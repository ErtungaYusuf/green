"""Compute channels: AI compute paid by the result, at the speed of the network.

The job market (SPEC.md §3.2) is asynchronous: a buyer posts a job, miners
deliver within blocks, and a dispute game keeps them honest. That suits big
batch work, but training a model takes thousands of matrix products, each
depending on the one before, and a block per product is days per epoch.
Channels take the chain out of the loop:

1. The buyer opens a channel to a provider (a miner's node): `channel_open`
   locks coins for that provider until a height. One transaction, one block.
2. The buyer sends matrix products straight to the provider and checks every
   result with Freivalds' test (verify_product) before using it: O(n^2) work,
   a secret random vector, and a wrong result passes with probability below
   2^-120. Correctness needs no chain and no waiting.
3. With each request the buyer sends a voucher: its signature over the total
   it owes for every result it has received and checked so far. The provider
   claims the highest voucher on chain whenever it likes before the channel
   expires (`channel_claim`); what it never claimed returns to the buyer.

Neither side can cheat the other by much. The buyer pays only for results that
passed its check; the provider computes ahead of payment for at most CREDIT
results, then waits for a voucher. A provider that returns a wrong result is
not paid for it and is dropped.

Wire protocol (HTTP on the provider's port; the body is raw bytes, C order):
  GET  /offer -> {"provider", "price" (yaprak per 2^24 multiply-adds), "max_dim", "credit", "chain"}
  POST /matmul, headers X-Channel, X-Shape "m,k,n", and with a payment X-Voucher-Total, X-Voucher;
       body A (m*k int8) then B (k*n int8) -> body C (m*n int32 little-endian), header X-Owed;
       402 pay first (or the channel is spent), 404 unknown channel, 409 channel expiring
  POST /voucher {"channel", "total", "voucher"} -> {"vouched"}     (paying without a request)
  GET  /channel/<id> -> {"owed", "vouched", "claimed"}               (the provider's books)
"""

import http.client
import json
import math
import os
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

from . import params, tx as txmod
from .crypto import Key, is_hex, verify
from .node import BoundedServer, http_get, http_post
from .chain import MissingData
from .tx import InvalidTx, make_tx

UNIT_MACS = params.UNIT ** 3        # 2^24 multiply-adds: what a price is quoted per
DEFAULT_PRICE = 1_000               # yaprak per unit
MAX_DIM = 4096                      # m, k, n of one request; k <= 4096 keeps every int32 sum exact
CREDIT = 4                          # results a provider delivers ahead of payment
CLAIM_MARGIN = 20                   # blocks before expiry when a provider stops serving and claims
CLAIM_MIN = 20 * params.MIN_TX_FEE  # unclaimed vouchers worth this much are claimed (fees stay <= 5%)
CLAIM_SECONDS = 5.0
SAVE_SECONDS = 1.0
OFFER_REFRESH = 1200               # providers renew their offer this often (offers live an hour)
OFFER_RETRY = 10                   # ...and retry this often until the network lists it


class PaymentRequired(Exception):
    pass


class ChannelEnding(Exception):
    pass


class WrongResult(Exception):
    """The provider's product failed Freivalds' test: it is not paid and should be dropped."""


class ProviderError(Exception):
    pass


def _private(host: str) -> bool:
    """Loopback, private and link-local addresses: not where strangers' offers should point a buyer."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost"
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified


def price_of(m: int, k: int, n: int, unit_price: int) -> int:
    return unit_price * max(1, math.ceil(m * k * n / UNIT_MACS))


SOUNDNESS_BITS = 120               # a wrong product passes verify_product with probability below 2^-120


def verify_product(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Is c == a @ b? Freivalds' test over the integers: with r a secret random
    matrix of entries in [0, 2^s), a wrong c has a nonzero row d with d·r = 0 for
    at most one value of any entry of r, so each column of r catches it with
    probability at least 1 - 2^-s; enough columns give 2^-SOUNDNESS_BITS.

    s is chosen so every sum stays below 2^53: float64 is then exact, whatever
    order the BLAS adds in, and the check costs a few matrix-vector products."""
    m, k = a.shape
    k2, n = b.shape
    if k != k2 or c.shape != (m, n):
        return False
    # Magnitudes as Python ints: np.abs of an int8 -128 is still -128.
    max_a = max(int(a.max(initial=0)), -int(a.min(initial=0)))
    max_b = max(int(b.max(initial=0)), -int(b.min(initial=0)))
    bound = k * max_a * max_b                 # no entry of a @ b exceeds this
    cf = c.astype(np.float64)
    if max(float(cf.max(initial=0)), -float(cf.min(initial=0))) > bound:
        return False
    # Largest sums: a @ (b @ r) <= k * max_a * k * max_b * 2^s and c @ r <= n * bound * 2^s.
    worst = max(k * max_a * k * max_b, n * bound, 1)
    s = min(30, 52 - worst.bit_length())
    if s < 8:
        raise ValueError("matrices too large for an exact check")
    rounds = -(-SOUNDNESS_BITS // s)
    rng = np.random.default_rng(int.from_bytes(os.urandom(32), "big"))
    r = rng.integers(0, 1 << s, size=(n, rounds)).astype(np.float64)
    lhs = a.astype(np.float64) @ (b.astype(np.float64) @ r)
    return bool(np.array_equal(lhs, cf @ r))


# --- the provider (a miner's node selling products) ------------------------------

class Provider:
    def __init__(self, node, key: Key, host: str = "127.0.0.1", port: int = 0, price: int = DEFAULT_PRICE,
                 max_dim: int = MAX_DIM, credit: int = CREDIT, url: str | None = None):
        if not 1 <= max_dim <= MAX_DIM or price < 1 or credit < 1:
            raise ValueError("bad provider settings")
        self.node, self.key, self.address = node, key, key.address
        self.host, self.port = host, port
        self.price, self.max_dim, self.credit = price, max_dim, credit
        self.url = url                    # how buyers reach us; advertised over the network when set
        self.announced = 0.0
        self.listed = False               # the network took our offer (we hold OFFER_MIN_BALANCE)
        self.path = os.path.join(node.data_dir, "provider.json")
        self.lock = threading.Lock()
        self.book: dict[str, dict] = self._load()   # channel -> {"owed", "delivered", "voucher": [total, sig]}
        self.dirty = False
        self.stats = {"products": 0, "units": 0, "refused": 0, "claims": 0}
        self.server = None
        self.running = False

    # -- books --

    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            data, self.dirty = json.dumps(self.book), False
        with open(self.path + ".tmp", "w") as f:
            f.write(data)
        os.replace(self.path + ".tmp", self.path)

    def offer(self) -> dict:
        return {"provider": self.address, "price": self.price, "max_dim": self.max_dim, "credit": self.credit,
                "chain": self.node.chain.chain_id}

    def signed_offer(self) -> dict:
        """Our offer, signed, for the network's provider list (p2p.py, GET /providers)."""
        offer = {"provider": self.address, "pub": self.key.pub_hex, "url": self.url, "price": self.price,
                 "max_dim": self.max_dim, "credit": self.credit, "time": int(time.time())}
        from .p2p import offer_bytes
        offer["sig"] = self.key.sign(offer_bytes(self.node.chain.chain_id, offer))
        return offer

    def announce(self) -> bool:
        self.announced = time.time()
        self.listed = bool(self.url) and self.node.p2p.add_offer(self.signed_offer())
        return self.listed

    def _channel(self, cid: str) -> dict:
        node = self.node
        with node.lock:
            ch = node.chain.state.channels.get(cid) if is_hex(cid, 64) else None
            height = node.chain.height
            ch = dict(ch) if ch else None
        if ch is None or ch["provider"] != self.address:
            raise LookupError("no open channel to this provider with that id (is it confirmed yet?)")
        if height + CLAIM_MARGIN >= ch["expires"]:
            raise ChannelEnding("the channel expires soon: open a new one")
        return ch

    def take_voucher(self, cid: str, ch: dict, total, sig) -> int:
        """Record a voucher if it is valid and better than the one we hold; returns the vouched total."""
        with self.lock:
            e = self.book.setdefault(cid, {"owed": 0, "delivered": [], "voucher": None})
            held = e["voucher"][0] if e["voucher"] else 0
        if total is None or total <= held:
            return held
        if not (isinstance(total, int) and total <= ch["amount"] and isinstance(sig, str)
                and verify(ch["pub"], txmod.voucher_bytes(self.node.chain.chain_id, cid, total), sig)):
            raise PaymentRequired("the voucher is not valid for this channel")
        with self.lock:
            if not e["voucher"] or total > e["voucher"][0]:
                e["voucher"] = [total, sig]
                e["delivered"] = [x for x in e["delivered"] if x > total]
                self.dirty = True
            return e["voucher"][0]

    def matmul(self, cid: str, shape: tuple, body: bytes, total=None, sig=None) -> tuple[bytes, int]:
        m, k, n = shape
        if not all(isinstance(d, int) and 1 <= d <= self.max_dim for d in shape):
            raise ValueError(f"every dimension must be 1..{self.max_dim}")
        if len(body) != m * k + k * n:
            raise ValueError("the body is not A and B of that shape")
        ch = self._channel(cid)
        vouched = self.take_voucher(cid, ch, total, sig)
        price = price_of(m, k, n, self.price)
        with self.lock:
            e = self.book[cid]
            if len([x for x in e["delivered"] if x > vouched]) >= self.credit:
                self.stats["refused"] += 1
                raise PaymentRequired(f"pay for the results delivered first (owed {e['owed']}, vouched {vouched})")
            if e["owed"] + price > ch["amount"]:
                self.stats["refused"] += 1
                raise PaymentRequired("the channel has too little left for this product: open another")
            e["owed"] += price                    # booked first, so parallel requests cannot overdraw
            e["delivered"].append(e["owed"])
            owed = e["owed"]
            self.dirty = True
        a = np.frombuffer(body, dtype=np.int8, count=m * k).reshape(m, k)
        b = np.frombuffer(body, dtype=np.int8, offset=m * k).reshape(k, n)
        # float64 is exact here (every sum stays below 2^31) and uses the fast BLAS.
        c = (a.astype(np.float64) @ b.astype(np.float64)).astype("<i4")
        with self.lock:
            self.stats["products"] += 1
            self.stats["units"] += price // self.price
        return c.tobytes(), owed

    def books(self, cid: str) -> dict:
        with self.lock:
            e = self.book.get(cid)
        if e is None:
            raise LookupError("no such channel in this provider's books")
        with self.node.lock:
            ch = self.node.chain.state.channels.get(cid)
            claimed = ch["paid"] if ch else None
        return {"owed": e["owed"], "vouched": e["voucher"][0] if e["voucher"] else 0, "claimed": claimed}

    # -- getting paid --

    def claim_due(self, force: bool = False) -> list[str]:
        """Claim vouchers worth claiming (or all, with `force`); returns the claimed channels."""
        node = self.node
        with node.lock:
            height = node.chain.height
            state = node.chain.state
            queued = {t["channel"] for t in node.mempool.values() if t["type"] == "channel_claim"}
            with self.lock:
                due = []
                for cid, e in self.book.items():
                    ch = state.channels.get(cid)
                    v = e["voucher"]
                    if ch is None or not v or v[0] <= ch["paid"] or cid in queued or height + 1 > ch["expires"]:
                        continue
                    unclaimed = v[0] - ch["paid"]
                    if unclaimed >= params.MIN_TX_FEE and (force or unclaimed >= CLAIM_MIN
                                                           or height + CLAIM_MARGIN >= ch["expires"]):
                        due.append((cid, v[0], v[1]))
        done = []
        for cid, total, sig in due:
            t = make_tx(self.key, node.chain.chain_id, "channel_claim", node.pending_nonce(self.address),
                        params.MIN_TX_FEE, channel=cid, total=total, voucher=sig)
            try:
                node.submit_tx(t, stem=False)
                done.append(cid)
                self.stats["claims"] += 1
            except (InvalidTx, MissingData) as e:
                node.log(f"[provider] claim on channel {cid[:12]}... refused: {e}")
        return done

    def _housekeeping(self) -> None:
        last_claim = 0.0
        while self.running:
            time.sleep(SAVE_SECONDS)
            try:
                self._save()
                # Renew the offer; until it is listed (a new miner has no balance yet), keep trying.
                if self.url and time.time() - self.announced >= (OFFER_REFRESH if self.listed else OFFER_RETRY):
                    self.announce()
                if time.time() - last_claim >= CLAIM_SECONDS:
                    last_claim = time.time()
                    self.claim_due()
                    self._forget_ended()
            except Exception as e:
                self.node.log(f"[provider] housekeeping error: {type(e).__name__}: {e}")

    def _forget_ended(self) -> None:
        with self.node.lock:
            live = set(self.node.chain.state.channels)
        with self.lock:
            for cid in [c for c in self.book if c not in live]:
                del self.book[cid]                # ended (claimed in full or expired): nothing more to do
                self.dirty = True

    # -- server --

    def start(self) -> None:
        self.server = BoundedServer((self.host, self.port), _handler(self))
        self.port = self.server.server_address[1]
        self.running = True
        if self.url == "":                       # "advertise me at whatever port I got" (tests, local nets)
            self.url = f"http://{'127.0.0.1' if self.host in ('0.0.0.0', '::') else self.host}:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True, name="provider").start()
        if self.url:
            self.announce()
        threading.Thread(target=self._housekeeping, daemon=True, name="provider-books").start()

    def stop(self) -> None:
        self.running = False
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        self._save()


def _handler(provider: Provider):
    max_body = 2 * provider.max_dim * provider.max_dim

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 120

        def log_message(self, *args):
            pass

        def _send(self, code: int, body, extra: dict | None = None):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream" if isinstance(body, bytes)
                             else "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _guard(self, fn):
            try:
                return fn()
            except PaymentRequired as e:
                return self._send(402, {"error": str(e)})
            except LookupError as e:
                return self._send(404, {"error": str(e)})
            except ChannelEnding as e:
                return self._send(409, {"error": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            except Exception:
                return self._send(500, {"error": "internal error"})

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/offer":
                return self._send(200, provider.offer())
            if path.startswith("/channel/"):
                return self._guard(lambda: self._send(200, provider.books(path[len("/channel/"):])))
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            raw = self.headers.get("Content-Length", "")
            if not raw.isdigit() or int(raw) > max_body:
                self.close_connection = True
                return self._send(413, {"error": "too large"})
            body = self.rfile.read(int(raw))
            if path == "/matmul":
                return self._guard(lambda: self._matmul(body))
            if path == "/voucher":
                return self._guard(lambda: self._voucher(body))
            return self._send(404, {"error": "not found"})

        def _matmul(self, body: bytes):
            shape = tuple(int(x) for x in self.headers.get("X-Shape", "").split(","))
            if len(shape) != 3:
                raise ValueError("X-Shape must be m,k,n")
            total = self.headers.get("X-Voucher-Total")
            c, owed = provider.matmul(self.headers.get("X-Channel", ""), shape, body,
                                      int(total) if total else None, self.headers.get("X-Voucher"))
            return self._send(200, c, {"X-Owed": str(owed)})

        def _voucher(self, body: bytes):
            m = json.loads(body)
            ch = provider._channel(m["channel"]) if m.get("channel") else None
            if ch is None:
                raise LookupError("no channel given")
            return self._send(200, {"vouched": provider.take_voucher(m["channel"], ch, m.get("total"),
                                                                     m.get("voucher"))})
    return Handler


# --- the buyer ------------------------------------------------------------------

class ComputeChannel:
    """A buyer's channel to one provider: open it, then call matmul like numpy."""

    def __init__(self, provider_url: str, key: Key, node_url: str, timeout: float = 300):
        self.url = provider_url.rstrip("/")
        self.node = node_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.offer = http_get(f"{self.url}/offer", timeout=30)
        self.chain_id = None
        self.cid = None
        self.amount = 0
        self.owed = 0                     # what the results received and checked so far cost
        self.bad = False                  # it returned a wrong result: do not use it again
        self.lock = threading.Lock()
        self.local = threading.local()    # one kept-alive connection per thread
        self.stats = {"products": 0, "macs": 0, "seconds": 0.0, "verify_seconds": 0.0}

    def open(self, amount: int, blocks: int = 4 * params.MIN_CHANNEL_BLOCKS, wait: float = 120) -> str:
        """Lock `amount` for this provider for `blocks` blocks; waits until the channel is confirmed."""
        st = http_get(f"{self.node}/status", timeout=30)
        if self.offer["chain"] != st["genesis"]:
            raise ProviderError("the provider is on another network")
        nonce = http_get(f"{self.node}/account/{self.key.address}", timeout=30)["pending_nonce"]
        t = make_tx(self.key, st["genesis"], "channel_open", nonce, params.MIN_TX_FEE,
                    provider=self.offer["provider"], amount=amount, expires=st["height"] + 1 + blocks)
        http_post(f"{self.node}/tx", t, timeout=30)
        self.chain_id, self.cid, self.amount = st["genesis"], txmod.txid(t), amount
        if not wait:
            return self.cid
        end = time.time() + wait
        while True:
            try:
                http_get(f"{self.node}/channel/{self.cid}", timeout=30)
                return self.cid
            except urllib.error.HTTPError as e:
                if e.code != 404 or time.time() > end:
                    raise
            time.sleep(0.5)

    def voucher_headers(self) -> dict:
        with self.lock:
            total = self.owed
        if not total:
            return {}
        return {"X-Voucher-Total": str(total), "X-Voucher": txmod.sign_voucher(self.key, self.chain_id, self.cid, total)}

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a @ b for int8 matrices, computed by the provider and checked here (int64 result)."""
        if self.bad:
            raise WrongResult("this provider returned a wrong result before")
        a = np.ascontiguousarray(a, dtype=np.int8)
        b = np.ascontiguousarray(b, dtype=np.int8)
        (m, k), (k2, n) = a.shape, b.shape
        if k != k2:
            raise ValueError("inner dimensions differ")
        started = time.time()
        headers = {"X-Channel": self.cid, "X-Shape": f"{m},{k},{n}", "Content-Type": "application/octet-stream",
                   **self.voucher_headers()}
        data = self._post(a.tobytes() + b.tobytes(), headers)
        if len(data) != 4 * m * n:
            self.bad = True
            raise WrongResult("the provider's answer has the wrong size")
        c = np.frombuffer(data, dtype="<i4").reshape(m, n)
        checked = time.time()
        if not verify_product(a, b, c):
            self.bad = True
            raise WrongResult("the provider's product failed Freivalds' test; it is not paid for it")
        with self.lock:
            self.owed += price_of(m, k, n, self.offer["price"])
            self.stats["products"] += 1
            self.stats["macs"] += m * k * n
            self.stats["seconds"] += time.time() - started
            self.stats["verify_seconds"] += time.time() - checked
        return c.astype(np.int64)

    def _connection(self) -> http.client.HTTPConnection:
        conn = getattr(self.local, "conn", None)
        if conn is None:
            u = urlparse(self.url)
            conn = self.local.conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=self.timeout)
        return conn

    def _post(self, body: bytes, headers: dict) -> bytes:
        end = time.time() + 60
        retried = False
        while True:
            conn = self._connection()
            try:
                conn.request("POST", "/matmul", body=body, headers=headers)
                r = conn.getresponse()
                data = r.read(4 * MAX_DIM * MAX_DIM + 1)
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                self.local.conn = None
                if retried:
                    raise ProviderError(f"cannot reach the provider: {e}") from None
                retried = True                    # a kept-alive connection the provider closed: once more
                continue
            if r.status == 200:
                return data
            reason = data.decode(errors="replace")
            if r.status == 404 and time.time() < end:     # the provider's node has not seen the channel yet
                time.sleep(0.5)
                continue
            if r.status == 402:
                raise PaymentRequired(reason)
            raise ProviderError(f"provider answered {r.status}: {reason}")

    def settle(self) -> int:
        """Send the voucher for everything received (the provider then claims it)."""
        with self.lock:
            total = self.owed
        if total:
            http_post(f"{self.url}/voucher", {"channel": self.cid, "total": total,
                                              "voucher": txmod.sign_voucher(self.key, self.chain_id, self.cid, total)},
                      timeout=30)
        return total


class ComputeCluster:
    """Several channels used as one. A large product is split by rows across the
    providers and computed in parallel; small ones go to the providers in turn. A
    provider that returns a wrong result is dropped and its share is computed by
    another one."""

    def __init__(self, channels: list[ComputeChannel], split_macs: int = 4 * UNIT_MACS):
        self.channels = list(channels)
        self.split_macs = split_macs
        self.pool = ThreadPoolExecutor(max(1, len(self.channels)), thread_name_prefix="compute")
        self.next = 0
        self.lock = threading.Lock()

    def live(self) -> list[ComputeChannel]:
        live = [c for c in self.channels if not c.bad]
        if not live:
            raise ProviderError("no honest provider left")
        return live

    def _one(self, a, b, prefer: ComputeChannel | None = None) -> np.ndarray:
        while True:
            live = self.live()
            if prefer is None or prefer.bad:
                with self.lock:
                    prefer = live[self.next % len(live)]
                    self.next += 1
            try:
                return prefer.matmul(a, b)
            except (WrongResult, ProviderError):
                prefer.bad = True
                prefer = None

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        live = self.live()
        m = a.shape[0]
        if len(live) == 1 or m * a.shape[1] * b.shape[1] < self.split_macs or m < len(live):
            return self._one(a, b)
        rows = np.array_split(np.arange(m), len(live))
        futures = [self.pool.submit(self._one, a[r[0]:r[-1] + 1], b, ch) for r, ch in zip(rows, live)]
        return np.vstack([f.result() for f in futures])

    @classmethod
    def from_market(cls, node_url: str, key: Key, amount_each: int, count: int = 3, max_price: int | None = None,
                    blocks: int = 4 * params.MIN_CHANNEL_BLOCKS, allow_local: bool = False) -> "ComputeCluster":
        """Find providers on the network (GET /providers on our node), take the `count`
        cheapest, and open a channel of `amount_each` to each."""
        offers = http_get(f"{node_url.rstrip('/')}/providers", timeout=30)
        picked = []
        for o in offers:
            host = urlparse(o["url"]).hostname or ""
            if (max_price is not None and o["price"] > max_price) or (not allow_local and _private(host)):
                continue
            try:
                ch = ComputeChannel(o["url"], key, node_url)
            except (OSError, ValueError):
                continue                          # unreachable: try the next one
            if ch.offer.get("provider") != o["provider"]:
                continue                          # the URL answers for someone else
            picked.append(ch)
            if len(picked) == count:
                break
        if not picked:
            raise ProviderError("no reachable provider on offer")
        for ch in picked:                         # one after another: each open uses the next nonce
            ch.open(amount_each, blocks=blocks, wait=0)
        end = time.time() + 120
        for ch in picked:
            while True:
                try:
                    http_get(f"{ch.node}/channel/{ch.cid}", timeout=30)
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 404 or time.time() > end:
                        raise
                    time.sleep(0.5)
        return cls(picked)

    def settle(self) -> int:
        return sum(c.settle() for c in self.channels if c.cid)

    @property
    def owed(self) -> int:
        return sum(c.owed for c in self.channels)
