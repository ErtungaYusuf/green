"""Non-custodial pooled mining.

Small miners find blocks rarely; pools smooth that out. In Bitcoin most pools
are custodial: the pool receives every reward and pays miners later, so a
handful of operators hold miners' coins and choose what goes into blocks.

Green pools pay in the block. A block's `miner` field may list up to
MAX_PAYEES addresses with weights (SPEC.md §5.1), and consensus splits the
reward and fees among them. The pool hands out a template whose payee list
follows the recent shares of each miner (PPLNS: the last WINDOW_SHARES
shares); whoever finds the block, every payee is paid by the block itself.
The pool never holds miners' coins, and a miner can see in every template what
it will be paid.

Protocol (JSON over HTTP on the pool's own port):
  GET  /template?address=<grn1...>  -> {"id", "header", "share_target"}
  POST /share {"id", "address", "nonce", "work", "transcript"}
       -> {"accepted": true, "block": "added" | null} or 400 with the reason
A share is the template's header with the miner's nonce, work field and
transcript whose ticket meets the share target (SHARE_FACTOR times easier than
the block target). The pool recomputes the work unit of every share.
"""

import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from . import params, work
from .chain import block_hash, fmt_target, seed_of, target_of, work_inputs
from .crypto import is_address, is_hex
from .node import BoundedServer, http_get, http_post

SHARE_FACTOR = 64              # shares are this much easier than blocks
WINDOW_SHARES = 1000           # PPLNS window
TEMPLATE_SECONDS = 10          # templates are refreshed this often (and on every new block)
FEE_PERCENT = 1                # the pool operator's share of each block
BAD_SHARE_LIMIT = 20           # invalid shares from one client before it is refused for a while
BAD_SHARE_PAUSE = 600


class Pool:
    def __init__(self, node, fee_address: str, host: str = "127.0.0.1", port: int = 0,
                 fee_percent: int = FEE_PERCENT):
        if not is_address(fee_address):
            raise ValueError("the pool needs a valid fee address")
        self.node = node
        self.fee_address = fee_address
        self.fee_percent = fee_percent
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.shares: deque = deque(maxlen=WINDOW_SHARES)     # (address, weight)
        self.templates: dict[str, dict] = {}                  # id -> {"header", "txs", "share_target", "made"}
        self.current: str | None = None
        self.seen: set[str] = set()                           # seeds of accepted shares (no double counting)
        self.bad: dict[str, list] = {}                        # client -> [count, until]
        self.stats = {"shares": 0, "rejected": 0, "blocks": 0}
        self.server = None

    # -- templates --

    def payees(self):
        """The payee list for the next template: recent shares by weight, plus the pool's fee."""
        weights: dict[str, int] = {}
        with self.lock:
            for addr, w in self.shares:
                weights[addr] = weights.get(addr, 0) + w
        if not weights:
            return self.fee_address
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:params.MAX_PAYEES - 1]
        total = sum(w for _, w in top)
        fee = max(1, total * self.fee_percent // (100 - self.fee_percent)) if self.fee_percent else 0
        merged = dict(top)
        if fee:
            merged[self.fee_address] = merged.get(self.fee_address, 0) + fee
        scale = max(1, -(-max(merged.values()) // params.MAX_PAYEE_WEIGHT))
        entries = sorted([a, max(1, w // scale)] for a, w in merged.items())
        return entries[0][0] if len(entries) == 1 else entries

    def template(self) -> dict:
        node = self.node
        with node.lock:
            tip = node.chain.tip
        with self.lock:
            cur = self.templates.get(self.current) if self.current else None
        if cur and cur["header"]["prev"] == tip and time.time() - cur["made"] < TEMPLATE_SECONDS:
            return cur
        miner = self.payees()
        with node.lock:
            header, txs, _ = node.chain.build_template(miner, list(node.mempool.values()),
                                                       signal_skip=node.no_signal)
        share_target = min(params.MAX_TARGET, target_of(header) * SHARE_FACTOR)
        tid = block_hash({"header": dict(header, transcript="00" * 32)})
        t = {"id": tid, "header": header, "txs": txs, "share_target": share_target, "made": time.time()}
        with self.lock:
            self.templates[tid] = t
            self.current = tid
            for old in [k for k, v in self.templates.items() if v["header"]["prev"] != tip]:
                del self.templates[old]                       # a new block makes old templates stale
        return t

    # -- shares --

    def submit(self, share: dict, client: str = "") -> dict:
        """Check a share; `client` (the submitter's IP) is what invalid shares are
        counted against, since payout addresses cost nothing to change."""
        addr = share.get("address")
        with self.lock:
            b = self.bad.get(client)
            if b and b[0] >= BAD_SHARE_LIMIT and time.time() < b[1]:
                raise PermissionError("too many invalid shares; paused")
            t = self.templates.get(share.get("id"))
        try:
            if not isinstance(addr, str) or not is_address(addr):
                raise ValueError("bad payout address")
            if t is None:
                raise ValueError("stale or unknown template: get a new one")
            return self._check(t, share, addr)
        except ValueError:
            with self.lock:
                b = self.bad.setdefault(client, [0, 0.0])
                b[0] += 1
                b[1] = time.time() + BAD_SHARE_PAUSE
                self.stats["rejected"] += 1
            raise

    def _check(self, t: dict, share: dict, addr: str) -> dict:
        nonce, w, transcript = share.get("nonce"), share.get("work"), share.get("transcript")
        if not (isinstance(nonce, int) and 0 <= nonce < 2**63 and is_hex(transcript, 64)):
            raise ValueError("bad nonce or transcript")
        if w != {"job": None, "unit": None}:
            raise ValueError("pool shares use filler work")
        header = dict(t["header"], nonce=nonce, work=w, transcript=transcript)
        seed = seed_of(header)
        key = seed.hex()
        with self.lock:
            if key in self.seen:
                raise ValueError("duplicate share")
        a, b = work_inputs(header, seed, self.node.blobs)
        if work.compute_unit(a, b, seed)[0] != transcript:
            raise ValueError("the transcript does not match the work")
        ticket = work.ticket_value(seed, transcript)
        if ticket > t["share_target"]:
            raise ValueError("the ticket does not meet the share target")
        weight = max(1, params.MAX_TARGET // t["share_target"])
        with self.lock:
            self.seen.add(key)
            if len(self.seen) > 100_000:
                self.seen.clear()
            self.shares.append((addr, weight))
            self.stats["shares"] += 1
        status = None
        if ticket <= target_of(header):
            status = self.node.submit_block({"header": header, "txs": t["txs"]})
            if status == "added":
                with self.lock:
                    self.stats["blocks"] += 1
                self.node.log(f"[pool] block found by {addr[:16]}... paying {len(header['miner']) if isinstance(header['miner'], list) else 1} payees")
        return {"accepted": True, "block": status}

    def status(self) -> dict:
        payees = self.payees()                       # (takes the lock itself)
        with self.lock:
            return {"miners": len({a for a, _ in self.shares}), "window": len(self.shares), **self.stats,
                    "fee_percent": self.fee_percent, "next_payees": payees}

    # -- server --

    def start(self) -> None:
        self.server = BoundedServer((self.host, self.port), _handler(self))
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True, name="pool").start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


def _public(t: dict) -> dict:
    return {"id": t["id"], "header": t["header"], "share_target": fmt_target(t["share_target"])}


def _handler(pool: Pool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        timeout = 15

        def log_message(self, *args):
            pass

        def _send(self, code: int, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/template":
                return self._send(200, _public(pool.template()))
            if u.path == "/status":
                return self._send(200, pool.status())
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                raw = self.headers.get("Content-Length", "0")
                if not raw.isdigit() or int(raw) > 64 * 1024:
                    return self._send(413, {"error": "too large"})
                share = json.loads(self.rfile.read(int(raw)) or b"null")
                if urlparse(self.path).path != "/share" or not isinstance(share, dict):
                    return self._send(404, {"error": "not found"})
                return self._send(200, pool.submit(share, client=self.client_address[0]))
            except PermissionError as e:
                return self._send(403, {"error": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            except Exception:
                return self._send(500, {"error": "internal error"})
    return Handler


class PoolMiner:
    """Mines for a pool: fetches templates, grinds nonces, submits shares."""

    def __init__(self, pool_url: str, address: str, gpu=None, log=print):
        if not is_address(address):
            raise ValueError("invalid payout address")
        self.pool = pool_url.rstrip("/")
        self.address = address
        self.gpu = gpu
        self.log = log
        self.running = True
        self.shares = self.blocks = self.rejected = 0

    def run(self, seconds: float | None = None) -> None:
        end = None if seconds is None else time.time() + seconds
        t, fetched = None, 0.0
        while self.running and (end is None or time.time() < end):
            if t is None or time.time() - fetched > TEMPLATE_SECONDS / 2:
                try:
                    t = http_get(f"{self.pool}/template", timeout=10)
                except (OSError, ValueError) as e:     # pool restarting or unreachable: keep trying
                    self.log(f"[pool miner] cannot reach the pool ({e}); retrying")
                    t = None
                    time.sleep(2)
                    continue
                fetched = time.time()
                header = dict(t["header"], work={"job": None, "unit": None})
                share_target = int(t["share_target"], 16)
                header["nonce"] = int.from_bytes(os.urandom(6), "big")
            transcript, ticket = self._ticket(header)
            if ticket <= share_target:
                try:
                    r = http_post(f"{self.pool}/share", {"id": t["id"], "address": self.address,
                                                          "nonce": header["nonce"], "work": header["work"],
                                                          "transcript": transcript}, timeout=10)
                    self.shares += 1
                    if r.get("block") == "added":
                        self.blocks += 1
                        self.log(f"[pool miner] found a block for the pool (paid in the block itself)")
                        t = None
                except OSError as e:
                    self.rejected += 1
                    t = None                              # likely stale: refresh the template
                    self.log(f"[pool miner] share refused: {e}")
            header["nonce"] += 1

    def _ticket(self, header) -> tuple[str, int]:
        seed = seed_of(header)
        a, b = work.filler_inputs(seed)
        if self.gpu is not None:
            return self.gpu.ticket(a, b, seed)
        transcript = work.compute_unit(a, b, seed)[0]
        return transcript, work.ticket_value(seed, transcript)

    def stop(self) -> None:
        self.running = False

