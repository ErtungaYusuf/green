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

Pools sell compute too. With each template a miner gets a few units of open
jobs (when the pool's balance can bond them). It computes them, which is the
same matrix product its ticket needs, and sends the results; the pool checks
each with Freivalds' test (O(n^2); a wrong result passes with probability
2^-124) and buffers it. The pool commits the results in batches and posts the
bond, and each commit names the miners who computed its units as payees
(SPEC.md §3.2): when the batch settles, consensus pays every miner for its
units. The pool keeps its fee and never holds what miners earn, from blocks
or from jobs.

Protocol (JSON over HTTP on the pool's own port):
  GET  /template?address=<grn1...>  -> {"id", "header", "share_target", "units": [[job, unit], ...]}
  GET  /job/<id>                    -> the job's matrices (the blob whose hash is the job id)
  POST /share {"id", "address", "nonce", "work", "transcript"}
       -> {"accepted": true, "block": "added" | null}, {"accepted": false, "stale": true}
          (a block came since that template: fetch the new one), or 400 with the reason
  POST /result {"address", "job", "unit", "result": base64 of the unit's int32 product}
       -> {"accepted": true} or 400 with the reason
A share is the template's header with the miner's nonce, work field (filler,
or a unit of an open job) and transcript whose ticket meets the share target
(SHARE_FACTOR times easier than the block target). The pool recomputes the
work unit of every share.
"""

import base64
import binascii
import json
import os
import random
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from . import params, tx as txmod, work
from .chain import (MAX_JOB_BLOB, BlobStore, InvalidBlock, MissingData, block_hash, fmt_target, seed_of,
                    target_of, work_inputs)
from .crypto import hhex, is_address, is_hex
from .node import COMMIT_INFLIGHT_CAP, BoundedServer, http_get, http_post
from .tx import InvalidTx

SHARE_FACTOR = 64              # shares are this much easier than blocks
WINDOW_SHARES = 1000           # PPLNS window
TEMPLATE_SECONDS = 10          # templates are refreshed this often (and on every new block)
MINER_REFRESH_SECONDS = 1.0    # miners ask for the current template this often (cheap: it is cached)
FEE_PERCENT = 1                # the pool operator's share of each block and each batch
BAD_SHARE_LIMIT = 20           # invalid shares or results from one client before it is refused for a while
BAD_SHARE_PAUSE = 600
UNITS_PER_MINER = 8            # useful units a miner holds at a time
ASSIGN_SECONDS = 120           # a unit not returned by then goes to someone else
MAX_SHARE_BODY = 64 * 1024
MAX_RESULT_BODY = 512 * 1024   # one unit's result (256 KiB) in base64, plus the envelope
SELL_SECONDS = 1.0             # how often a pool that does not mine itself commits and reveals


class Pool:
    def __init__(self, node, fee_address: str, host: str = "127.0.0.1", port: int = 0,
                 fee_percent: int = FEE_PERCENT, useful: bool = True):
        if not is_address(fee_address):
            raise ValueError("the pool needs a valid fee address")
        if not 0 <= fee_percent < 100:
            raise ValueError("the pool's fee is a percentage below 100")
        self.node = node
        self.fee_address = fee_address
        self.fee_percent = fee_percent
        # Selling units needs a key to commit with (the node's wallet) and posts its bonds.
        self.useful = useful and node.miner_key is not None
        self.host, self.port = host, port
        self.lock = threading.Lock()
        self.shares: deque = deque(maxlen=WINDOW_SHARES)     # (address, weight)
        self.templates: dict[str, dict] = {}                  # id -> {"header", "txs", "share_target", "made"}
        self.current: str | None = None
        self.seen: set[str] = set()                           # seeds of accepted shares (no double counting)
        self.bad: dict[str, list] = {}                        # client -> [count, until]
        self.assigned: dict[tuple, dict] = {}                 # (job, unit key) -> {"addr", "unit", "until"}
        self.units_by: dict[str, int] = {}                    # address -> results accepted
        self.stats = {"shares": 0, "rejected": 0, "blocks": 0, "useful_units": 0, "useful_shares": 0}
        self.server = None
        self.running = False

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

    # -- selling units --

    def assigned_keys(self) -> set:
        now = time.time()
        with self.lock:
            return {k for k, a in self.assigned.items() if a["until"] >= now}

    def assign(self, addr: str) -> list:
        """The open units `addr` should compute: those it holds, topped up to
        UNITS_PER_MINER from the best-paying jobs, as far as the pool's balance can
        bond them (units nobody can bond would be computed for nothing)."""
        if not self.useful or not is_address(addr):
            return []
        node = self.node
        now = time.time()
        with node.lock:
            state = node.chain.state
            busy = node._busy_units()
            in_flight = (sum(len(b["units"]) for b in node.batch_buf.values())
                         + sum(b["count"] for b in node.my_batches.values()))
            funds = node._pending.balance(node.miner_key.address) - 4 * params.MIN_TX_FEE
            for job_id, buf in node.batch_buf.items():          # bonds for units already waiting
                if job_id in state.jobs:
                    funds -= params.bond_for(state.jobs[job_id]["price"]) * len(buf["units"])
            jobs = sorted(((j, v) for j, v in state.jobs.items() if not v["closed"] and v["closes_at"] is None),
                          key=lambda jv: -jv[1]["price"])
            open_units = {j: state.open_units(j) for j, _ in jobs}
            with self.lock:
                for k in [k for k, a in self.assigned.items() if a["until"] < now]:
                    del self.assigned[k]
                for k, a in self.assigned.items():              # bonds for units out with miners
                    if k[0] in state.jobs:
                        funds -= params.bond_for(state.jobs[k[0]]["price"])
                mine = [[k[0], a["unit"]] for k, a in self.assigned.items() if a["addr"] == addr]
                room = min(UNITS_PER_MINER - len(mine), COMMIT_INFLIGHT_CAP - in_flight - len(self.assigned))
                for job_id, job in jobs:
                    bond = params.bond_for(job["price"])
                    for u in open_units[job_id]:
                        if room <= 0 or funds < bond:
                            break
                        key = (job_id, txmod.unit_key(u))
                        if key in busy or key in self.assigned:
                            continue
                        self.assigned[key] = {"addr": addr, "unit": list(u), "until": now + ASSIGN_SECONDS}
                        mine.append([job_id, list(u)])
                        funds -= bond
                        room -= 1
        return mine

    def submit_result(self, body: dict, client: str = "") -> dict:
        self._check_paused(client)
        try:
            return self._take_result(body)
        except ValueError:
            self._count_bad(client)
            raise

    def _take_result(self, body: dict) -> dict:
        addr, job_id, unit, data = body.get("address"), body.get("job"), body.get("unit"), body.get("result")
        if not (isinstance(addr, str) and is_address(addr) and is_hex(job_id, 64) and txmod._is_unit(unit)
                and isinstance(data, str) and len(data) <= MAX_RESULT_BODY):
            raise ValueError("malformed result")
        key = (job_id, txmod.unit_key(unit))
        with self.lock:
            a = self.assigned.get(key)
            if a is None or a["addr"] != addr:
                raise ValueError("this unit is not assigned to you (or the assignment expired)")
        try:
            c = txmod.decode_result(base64.b64decode(data, validate=True))
        except (ValueError, binascii.Error):
            raise ValueError("not a unit result") from None
        node = self.node
        with node.lock:
            job = node.chain.state.jobs.get(job_id)
        if job is None:
            with self.lock:
                self.assigned.pop(key, None)
            return {"accepted": False, "reason": "the job is gone"}     # not the miner's fault
        a_mat, b_mat = node.blobs.job(job_id, txmod.job_blob_size(job["dims"]))
        ua, ub = txmod.unit_inputs(a_mat, b_mat, unit)
        # A fresh secret seed: the miner cannot tailor a wrong result to pass.
        if not work.freivalds_check(ua, ub, c, os.urandom(32)):
            raise ValueError("the result is wrong (Freivalds check)")
        with self.lock:
            if self.assigned.get(key, {}).get("addr") != addr:
                raise ValueError("the assignment expired")
            del self.assigned[key]
            self.stats["useful_units"] += 1
            self.units_by[addr] = self.units_by.get(addr, 0) + 1
        node._buffer_result({"job": job_id, "unit": list(unit)}, c, by=addr)
        return {"accepted": True}

    def job_blob(self, job_id: str) -> bytes | None:
        """A job's matrices, for miners computing its units (open jobs only)."""
        if not is_hex(job_id, 64):
            return None
        with self.node.lock:
            if job_id not in self.node.chain.state.jobs:
                return None
        return self.node.blobs.get(job_id)

    def _seller(self) -> None:
        while self.running:
            time.sleep(SELL_SECONDS)
            if self.node.mining:
                continue                              # the miner loop commits and reveals
            try:
                self.node.sell_step()
            except Exception as e:
                self.node.log(f"[pool] error selling units: {type(e).__name__}: {e}")

    # -- shares --

    def _check_paused(self, client: str) -> None:
        with self.lock:
            b = self.bad.get(client)
            if b and b[0] >= BAD_SHARE_LIMIT and time.time() < b[1]:
                raise PermissionError("too many invalid shares; paused")

    def _count_bad(self, client: str) -> None:
        with self.lock:
            b = self.bad.setdefault(client, [0, 0.0])
            b[0] += 1
            b[1] = time.time() + BAD_SHARE_PAUSE
            self.stats["rejected"] += 1

    def submit(self, share: dict, client: str = "") -> dict:
        """Check a share; `client` (the submitter's IP) is what invalid shares are
        counted against, since payout addresses cost nothing to change."""
        self._check_paused(client)
        addr = share.get("address")
        with self.lock:
            t = self.templates.get(share.get("id"))
        with self.node.lock:
            tip = self.node.chain.tip
        try:
            if not isinstance(addr, str) or not is_address(addr):
                raise ValueError("bad payout address")
            if t is None or t["header"]["prev"] != tip:
                # Work on a template a new block replaced: normal, and cheap to refuse,
                # so it is not held against the miner.
                return {"accepted": False, "stale": True}
            return self._check(t, share, addr)
        except ValueError:
            self._count_bad(client)
            raise

    def _check(self, t: dict, share: dict, addr: str) -> dict:
        nonce, w, transcript = share.get("nonce"), share.get("work"), share.get("transcript")
        if not (isinstance(nonce, int) and 0 <= nonce < 2**63 and is_hex(transcript, 64)):
            raise ValueError("bad nonce or transcript")
        if not isinstance(w, dict) or set(w) != {"job", "unit"}:
            raise ValueError("bad work field")
        job_size = 0
        if w != {"job": None, "unit": None}:
            # A unit of an open job: a block needs the job in the state it builds on.
            with self.node.lock:
                job = self.node.chain.state.jobs.get(w["job"]) if is_hex(w["job"], 64) else None
            if job is None or not txmod._is_unit(w["unit"]):
                raise ValueError("work must be filler or a unit of an open job")
            job_size = txmod.job_blob_size(job["dims"])
        header = dict(t["header"], nonce=nonce, work=w, transcript=transcript)
        seed = seed_of(header)
        key = seed.hex()
        with self.lock:
            if key in self.seen:
                raise ValueError("duplicate share")
        try:
            a, b = work_inputs(header, seed, self.node.blobs, job_size)
        except (InvalidBlock, InvalidTx, MissingData) as e:
            raise ValueError(f"cannot do that work unit: {e}") from None
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
            self.stats["useful_shares"] += w["job"] is not None
        status = None
        if ticket <= target_of(header):
            try:
                status = self.node.submit_block({"header": header, "txs": t["txs"]})
            except (InvalidBlock, MissingData) as e:     # e.g. its job closed since the template
                self.node.log(f"[pool] a share that met the block target was not a valid block: {e}")
                status = "invalid"
            if status == "added":
                with self.lock:
                    self.stats["blocks"] += 1
                self.node.log(f"[pool] block found by {addr[:16]}... paying {len(header['miner']) if isinstance(header['miner'], list) else 1} payees")
        return {"accepted": True, "block": status}

    def status(self) -> dict:
        payees = self.payees()                       # (takes the lock itself)
        with self.lock:
            return {"miners": len({a for a, _ in self.shares}), "window": len(self.shares), **self.stats,
                    "fee_percent": self.fee_percent, "next_payees": payees, "selling_units": self.useful,
                    "units_out": len(self.assigned)}

    # -- server --

    def start(self) -> None:
        self.server = BoundedServer((self.host, self.port), _handler(self))
        self.port = self.server.server_address[1]
        self.running = True
        self.node.pool = self
        threading.Thread(target=self.server.serve_forever, daemon=True, name="pool").start()
        if self.useful:
            threading.Thread(target=self._seller, daemon=True, name="pool-seller").start()

    def stop(self) -> None:
        self.running = False
        if self.node.pool is self:
            self.node.pool = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()


def _public(t: dict, units: list) -> dict:
    return {"id": t["id"], "header": t["header"], "share_target": fmt_target(t["share_target"]), "units": units}


def _handler(pool: Pool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        timeout = 15

        def log_message(self, *args):
            pass

        def _send(self, code: int, body, raw: bool = False):
            data = body if raw else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream" if raw else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/template":
                addr = parse_qs(u.query).get("address", [""])[0]
                return self._send(200, _public(pool.template(), pool.assign(addr)))
            if u.path.startswith("/job/"):
                data = pool.job_blob(u.path[len("/job/"):])
                if data is None:
                    return self._send(404, {"error": "no such open job"})
                return self._send(200, data, raw=True)
            if u.path == "/status":
                return self._send(200, pool.status())
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                limit = MAX_RESULT_BODY if path == "/result" else MAX_SHARE_BODY
                raw = self.headers.get("Content-Length", "0")
                if not raw.isdigit() or int(raw) > limit:
                    return self._send(413, {"error": "too large"})
                body = json.loads(self.rfile.read(int(raw)) or b"null")
                if path not in ("/share", "/result") or not isinstance(body, dict):
                    return self._send(404, {"error": "not found"})
                client = self.client_address[0]
                if path == "/result":
                    return self._send(200, pool.submit_result(body, client=client))
                return self._send(200, pool.submit(body, client=client))
            except PermissionError as e:
                return self._send(403, {"error": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            except Exception:
                return self._send(500, {"error": "internal error"})
    return Handler


class PoolMiner:
    """Mines for a pool: fetches templates, computes the units of open jobs the pool
    assigns (sending their results), grinds nonces, submits shares."""

    def __init__(self, pool_url: str, address: str, gpu=None, log=print):
        if not is_address(address):
            raise ValueError("invalid payout address")
        self.pool = pool_url.rstrip("/")
        self.address = address
        self.gpu = gpu
        self.log = log
        self.running = True
        self.blobs = BlobStore()                  # job matrices, fetched from the pool
        self.shares = self.blocks = self.rejected = self.useful = 0

    def run(self, seconds: float | None = None) -> None:
        end = None if seconds is None else time.time() + seconds
        t, fetched = None, 0.0
        while self.running and (end is None or time.time() < end):
            if t is None or time.time() - fetched > MINER_REFRESH_SECONDS:
                try:
                    t = http_get(f"{self.pool}/template?address={self.address}", timeout=10)
                except (OSError, ValueError) as e:     # pool restarting or unreachable: keep trying
                    self.log(f"[pool miner] cannot reach the pool ({e}); retrying")
                    t = None
                    time.sleep(2)
                    continue
                fetched = time.time()
                share_target = int(t["share_target"], 16)
                delivered = 0
                for job_id, unit in t.get("units", []):     # what the pool still wants: none are done
                    if not self.running:
                        break
                    try:
                        delivered += self._useful_unit(t, job_id, unit, share_target)
                    except (OSError, ValueError, InvalidBlock, InvalidTx, MissingData) as e:
                        self.log(f"[pool miner] could not deliver unit {unit} of job {job_id[:12]}...: {e}")
                if delivered:
                    fetched = 0.0                     # there is more paid work: ask for it right away
                header = dict(t["header"], work={"job": None, "unit": None})
                header["nonce"] = int.from_bytes(os.urandom(6), "big")
            transcript, ticket = self._ticket(header)
            if ticket <= share_target and not self._share(t, header, transcript):
                t = None                              # likely stale: refresh the template
                continue
            header["nonce"] += 1

    def _share(self, t: dict, header: dict, transcript: str) -> bool:
        try:
            r = http_post(f"{self.pool}/share", {"id": t["id"], "address": self.address, "nonce": header["nonce"],
                                                  "work": header["work"], "transcript": transcript}, timeout=10)
        except OSError as e:
            self.rejected += 1
            self.log(f"[pool miner] share refused: {e}")
            return False
        if r.get("stale"):
            return False                              # a new block: the caller fetches the new template
        self.shares += 1
        if r.get("block") == "added":
            self.blocks += 1
            self.log("[pool miner] found a block for the pool (paid in the block itself)")
        return True

    def _useful_unit(self, t: dict, job_id: str, unit: list, share_target: int) -> bool:
        """Compute one assigned unit: its ticket may be a share, and its product is a
        result the pool sells (and pays us for, in the batch's commit)."""
        if not self.blobs.has(job_id):
            data = http_get(f"{self.pool}/job/{job_id}", timeout=120, raw=True, limit=MAX_JOB_BLOB)
            if hhex(data) != job_id:
                raise ValueError("the pool sent data that is not the job")
            for old in [k for k in self.blobs.keys() if k != job_id][:max(0, len(self.blobs.keys()) - 3)]:
                self.blobs.delete(old)               # keep a few jobs; each can be 32 MiB
            self.blobs.put(data)
        header = dict(t["header"], work={"job": job_id, "unit": list(unit)}, nonce=random.getrandbits(48))
        seed = seed_of(header)
        a, b = work_inputs(header, seed, self.blobs, MAX_JOB_BLOB)
        transcript, product = work.compute_unit(a, b, seed)
        if work.ticket_value(seed, transcript) <= share_target:
            self._share(t, header, transcript)
        blob = txmod.encode_result(product)
        r = http_post(f"{self.pool}/result", {"address": self.address, "job": job_id, "unit": list(unit),
                                               "result": base64.b64encode(blob).decode()}, timeout=60)
        if r.get("accepted"):
            self.useful += 1
        return bool(r.get("accepted"))

    def _ticket(self, header) -> tuple[str, int]:
        seed = seed_of(header)
        a, b = work.filler_inputs(seed)
        if self.gpu is not None:
            return self.gpu.ticket(a, b, seed)
        transcript = work.compute_unit(a, b, seed)[0]
        return transcript, work.ticket_value(seed, transcript)

    def stop(self) -> None:
        self.running = False
