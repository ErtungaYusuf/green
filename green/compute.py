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
import itertools
import json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np

from . import params, sealed, tx as txmod, work
from .crypto import Key, is_hex, verify
from .node import BoundedServer, http_get, http_post
from .p2p import NO_REDIRECTS
from .chain import MissingData
from .tx import InvalidTx, make_tx

UNIT_MACS = params.UNIT ** 3        # 2^24 multiply-adds: what a price is quoted per
DEFAULT_PRICE = 1_000               # yaprak per unit
MAX_DIM = 4096                      # m, k, n of one request; k <= 4096 keeps every int32 sum exact
CREDIT = 4                          # results a provider delivers ahead of payment
CLAIM_MARGIN = 20                   # blocks before expiry when a provider stops serving and claims
CLAIM_MIN = 40 * params.MIN_TX_FEE  # unclaimed vouchers worth this much are claimed (fees stay <= 5%)
CLAIM_SECONDS = 5.0
CLAIM_TX_BYTES = 450               # about the size of a claim, for its fee
COMPUTE_SLOTS = 2                  # products computed at once (the BLAS already uses every core)
IDLE_SECONDS = 30                  # a kept-alive connection with nothing to do is closed
DRAIN_SECONDS = 10                 # after a refusal, the unread body is read and dropped for at most this long
PARALLEL_MIN_SECONDS = 0.005       # products faster than this on average go one by one (threads cost more)
SAVE_SECONDS = 1.0
OFFER_REFRESH = 1200               # providers renew their offer this often (offers live an hour)
OFFER_RETRY = 10                   # ...and retry this often until the network lists it


class PaymentRequired(Exception):
    pass


class ChannelEnding(Exception):
    pass


class WrongResult(Exception):
    """The provider's product failed Freivalds' test: it is not paid and should be dropped."""


class RelayError(Exception):
    """A relayed request could not be carried (our node or its peers): not the provider's
    fault, so its channel is not given up on."""


class ProviderError(Exception):
    pass


def _private(host: str) -> bool:
    """Addresses that are not on the public internet (loopback, private, link-local,
    carrier-grade NAT): not where strangers' offers should point a buyer."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost"
    return not ip.is_global


def price_of(m: int, k: int, n: int, unit_price: int) -> int:
    return unit_price * max(1, math.ceil(m * k * n / UNIT_MACS))


SOUNDNESS_BITS = 120               # a wrong product passes verify_product with probability below 2^-120
MOD_MAX = 1 << 20                  # moduli of modular products (green/private.py): k * (p-1)^2 < 2^53


def product_bytes(shape: tuple, modulus: int | None) -> int:
    """The body of a product request: A then B, int8 (or int32 for a modular product)."""
    m, k, n = shape
    return (m * k + k * n) * (1 if modulus is None else 4)


def verify_product_mod(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: int) -> bool:
    """Is c == a @ b (mod p)? Freivalds' test in Z_p with random columns of entries in
    [0, 2^10): a wrong row passes one column with probability at most 2^-10, so 12
    columns give 2^-120. Every sum stays below 2^53 for k <= 4096 and p <= 2^20."""
    m, k = a.shape
    if b.shape[0] != k or c.shape != (m, b.shape[1]) or not 2 <= p <= MOD_MAX or k * (p - 1) ** 2 >= 2 ** 53:
        return False
    if c.min(initial=0) < 0 or c.max(initial=0) >= p:
        return False
    rng = np.random.default_rng(int.from_bytes(os.urandom(32), "big"))
    r = rng.integers(0, 1 << 10, size=(b.shape[1], -(-SOUNDNESS_BITS // 10))).astype(np.float64)
    br = np.mod(work.blas_matmul(b, r), p)
    return bool(np.array_equal(np.mod(work.blas_matmul(a, br), p), np.mod(work.blas_matmul(c, r), p)))


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
    lhs = work.blas_matmul(a, work.blas_matmul(b, r))
    return bool(np.array_equal(lhs, work.blas_matmul(cf, r)))


# --- the provider (a miner's node selling products) ------------------------------

class Provider:
    def __init__(self, node, key: Key, host: str = "127.0.0.1", port: int = 0, price: int = DEFAULT_PRICE,
                 max_dim: int = MAX_DIM, credit: int = CREDIT, url: str | None = None,
                 compute_slots: int = COMPUTE_SLOTS):
        if not 1 <= max_dim <= MAX_DIM or price < 1 or credit < 1 or compute_slots < 1:
            raise ValueError("bad provider settings")
        self.node, self.key, self.address = node, key, key.address
        self.host, self.port = host, port
        self.price, self.max_dim, self.credit = price, max_dim, credit
        self.url = url                    # how buyers reach us; advertised over the network when set
        self.announced = 0.0
        self.listed = False               # the network took our offer (we hold OFFER_MIN_BALANCE)
        self.path = os.path.join(node.data_dir, "provider.json")
        self.lock = threading.Lock()
        # channel -> {"base": amount settled by vouchers, "recent": [[ticket, price], ...] products after
        # that, "voucher": [total, sig] or None, "expires": height}
        self.book: dict[str, dict] = self._load()
        self.tickets = itertools.count(1)
        self.slots = threading.BoundedSemaphore(compute_slots)   # large products need ~400 MB each
        self.dirty = False
        self.stats = {"products": 0, "units": 0, "refused": 0, "claims": 0, "closed": 0}
        self.server = None
        self.running = False
        self.kx = sealed.provider_key(key.secret_hex())   # buyers seal requests to this (green/sealed.py)

    # -- books --

    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                book = json.load(f)
        except (OSError, ValueError):
            return {}
        for e in book.values():                  # books written by v0.15
            if "recent" not in e:
                e.update(base=e.pop("owed", 0), recent=[], expires=e.get("expires", 0))
                e.pop("delivered", None)
        return book

    def _save(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            data, self.dirty = json.dumps(self.book), False
        with open(self.path + ".tmp", "w") as f:
            f.write(data)
        os.replace(self.path + ".tmp", self.path)

    @staticmethod
    def _owed(e: dict) -> int:
        return e["base"] + sum(p for _, p in e["recent"])

    @staticmethod
    def _settle(e: dict) -> None:
        """Fold products the voucher covers into the base."""
        vouched = e["voucher"][0] if e["voucher"] else 0
        while e["recent"] and e["base"] + e["recent"][0][1] <= vouched:
            e["base"] += e["recent"].pop(0)[1]

    def offer(self) -> dict:
        return {"provider": self.address, "price": self.price, "max_dim": self.max_dim, "credit": self.credit,
                "chain": self.node.chain.chain_id, "kx": sealed.public_hex(self.kx)}

    def signed_offer(self) -> dict:
        """Our offer, signed, for the network's provider list (p2p.py, GET /providers)."""
        offer = {"provider": self.address, "pub": self.key.pub_hex, "url": self.url, "price": self.price,
                 "max_dim": self.max_dim, "credit": self.credit, "time": int(time.time()),
                 "kx": sealed.public_hex(self.kx)}
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

    def _entry(self, cid: str, ch: dict) -> dict:
        e = self.book.get(cid)
        if e is None:
            e = self.book[cid] = {"base": 0, "recent": [], "voucher": None, "expires": ch["expires"]}
        return e

    def take_voucher(self, cid: str, ch: dict, total, sig) -> int:
        """Record a voucher if it is valid and better than the one we hold; returns the vouched total."""
        with self.lock:
            e = self._entry(cid, ch)
            held = e["voucher"][0] if e["voucher"] else 0
        if total is None or total <= held:
            return held
        if not (type(total) is int and total <= ch["amount"] and isinstance(sig, str)
                and verify(ch["pub"], txmod.voucher_bytes(self.node.chain.chain_id, cid, total), sig, cache=False)):
            raise PaymentRequired("the voucher is not valid for this channel")
        with self.lock:
            if not e["voucher"] or total > e["voucher"][0]:
                e["voucher"] = [total, sig]
                self._settle(e)
                self.dirty = True
            return e["voucher"][0]

    def admit(self, cid: str, shape: tuple, total=None, sig=None) -> tuple[int, int]:
        """Everything about a product request except its data: shape, channel, payment,
        credit and funds. Checked before the body is read, so a stranger cannot make us
        read 32 MiB. Books the product; returns (ticket, owed). release() undoes it."""
        if len(shape) != 3 or not all(type(d) is int and 1 <= d <= self.max_dim for d in shape):
            raise ValueError(f"X-Shape must be three dimensions in 1..{self.max_dim}")
        ch = self._channel(cid)
        vouched = self.take_voucher(cid, ch, total, sig)
        price = price_of(*shape, self.price)
        with self.lock:
            e = self._entry(cid, ch)
            unpaid, running = 0, e["base"]
            for _, p in e["recent"]:
                running += p
                unpaid += running > vouched
            if unpaid >= self.credit:
                self.stats["refused"] += 1
                raise PaymentRequired(f"pay for the results delivered first (owed {self._owed(e)}, vouched {vouched})")
            if self._owed(e) + price > ch["amount"]:
                self.stats["refused"] += 1
                raise PaymentRequired("the channel has too little left for this product: open another")
            ticket = next(self.tickets)
            e["recent"].append([ticket, price])  # booked first, so parallel requests cannot overdraw
            self.dirty = True
            return ticket, self._owed(e)

    def release(self, cid: str, ticket: int) -> None:
        """A booked product that was never delivered (the request broke off)."""
        with self.lock:
            e = self.book.get(cid)
            if e is not None:
                e["recent"] = [x for x in e["recent"] if x[0] != ticket]
                self.dirty = True

    def compute(self, shape: tuple, body: bytes) -> bytes:
        m, k, n = shape
        if len(body) != m * k + k * n:
            raise ValueError("the body is not A and B of that shape")
        a = np.frombuffer(body, dtype=np.int8, count=m * k).reshape(m, k)
        b = np.frombuffer(body, dtype=np.int8, offset=m * k).reshape(k, n)
        with self.slots:
            # float64 is exact here (every sum stays below 2^31) and uses the fast BLAS.
            c = work.blas_matmul(a, b, "<i4")
        with self.lock:
            self.stats["products"] += 1
            self.stats["units"] += price_of(m, k, n, 1)
        return c.tobytes()

    def compute_mod(self, shape: tuple, modulus: int, body: bytes) -> bytes:
        """(A @ B) mod p for int32 A, B with entries in [0, p): the products of shares in
        green/private.py. float64 is exact while k * (p-1)^2 < 2^53."""
        m, k, n = shape
        if not 2 <= modulus <= MOD_MAX or k * (modulus - 1) ** 2 >= 2 ** 53:
            raise ValueError("modulus too large for an exact product")
        if len(body) != product_bytes(shape, modulus):
            raise ValueError("the body is not A and B of that shape")
        a = np.frombuffer(body, dtype="<i4", count=m * k).reshape(m, k)
        b = np.frombuffer(body, dtype="<i4", count=k * n, offset=4 * m * k).reshape(k, n)
        if min(a.min(initial=0), b.min(initial=0)) < 0 or max(a.max(initial=0), b.max(initial=0)) >= modulus:
            raise ValueError("entries must be in [0, modulus)")
        with self.slots:
            c = np.mod(work.blas_matmul(a, b), modulus).astype("<i4")
        with self.lock:
            self.stats["products"] += 1
            self.stats["units"] += price_of(m, k, n, 1)
        return c.tobytes()

    def serve_sealed(self, data: bytes) -> bytes:
        """Answer a sealed request (green/sealed.py) with a sealed answer. Raises SealError
        if it cannot be opened; everything else, refusals included, goes in the answer."""
        header, body, key = sealed.open_request(self.kx, data)
        path, h = header.get("path"), header.get("headers") or {}
        status, extra, out = 200, {}, b""
        try:
            if not isinstance(h, dict) or not all(isinstance(v, str) for v in h.values()):
                raise ValueError("bad headers")
            if path in ("/matmul", "/matmul_mod"):
                modulus = int(h.get("X-Modulus", "")) if path == "/matmul_mod" else None
                shape = tuple(int(x) for x in h.get("X-Shape", "").split(","))
                if len(shape) != 3 or len(body) != product_bytes(shape, modulus):
                    raise ValueError("X-Shape must be m,k,n and the body A and B of that shape")
                cid, total = h.get("X-Channel", ""), h.get("X-Voucher-Total")
                ticket, owed = self.admit(cid, shape, int(total) if total else None, h.get("X-Voucher"))
                try:
                    out = self.compute(shape, body) if modulus is None else self.compute_mod(shape, modulus, body)
                except BaseException:
                    self.release(cid, ticket)
                    raise
                extra = {"X-Owed": str(owed)}
            elif path == "/voucher":
                m = json.loads(body)
                ch = self._channel(m["channel"]) if m.get("channel") else None
                if ch is None:
                    raise LookupError("no channel given")
                out = json.dumps({"vouched": self.take_voucher(m["channel"], ch, m.get("total"), m.get("voucher"))}).encode()
            elif path == "/close":
                m = json.loads(body)
                out = json.dumps(self.close(m.get("channel", ""), m.get("total"), m.get("voucher"))).encode()
            else:
                status, out = 404, b'{"error": "not found"}'
        except PaymentRequired as e:
            status, out = 402, json.dumps({"error": str(e)}).encode()
        except LookupError as e:
            status, out = 404, json.dumps({"error": str(e)}).encode()
        except ChannelEnding as e:
            status, out = 409, json.dumps({"error": str(e)}).encode()
        except (ValueError, KeyError, TypeError) as e:
            status, out = 400, json.dumps({"error": str(e)}).encode()
        except Exception:
            status, out = 500, b'{"error": "internal error"}'
        return sealed.seal_answer(key, {"status": status, "headers": extra}, out)

    def matmul(self, cid: str, shape: tuple, body: bytes, total=None, sig=None) -> tuple[bytes, int]:
        """admit + compute, for callers that already hold the body."""
        ticket, owed = self.admit(cid, tuple(shape), total, sig)
        try:
            return self.compute(tuple(shape), body), owed
        except BaseException:
            self.release(cid, ticket)
            raise

    def books(self, cid: str) -> dict:
        with self.lock:
            e = self.book.get(cid)
            if e is None:
                raise LookupError("no such channel in this provider's books")
            owed, vouched = self._owed(e), e["voucher"][0] if e["voucher"] else 0
        with self.node.lock:
            ch = self.node.chain.state.channels.get(cid)
            claimed = ch["paid"] if ch else None
        return {"owed": owed, "vouched": vouched, "claimed": claimed}

    # -- getting paid --

    def _claim_fee(self) -> int:
        """A fee that should get a claim into the next blocks, from the node's estimate."""
        rate = self.node.fee_estimate(2)["fee_rate"]
        return max(2 * params.MIN_TX_FEE, int(rate * CLAIM_TX_BYTES) + 1 if rate else 0)   # 2 signatures

    def claim_due(self, force: bool = False, close: set | None = None) -> list[str]:
        """Claim vouchers worth claiming (all of them with `force`), and close the channels
        in `close`; returns the channels claimed. A claim still waiting in the mempool is
        replaced (same nonce, higher fee) when a channel is to be closed."""
        node, close = self.node, close or set()
        fee = self._claim_fee()
        with node.lock:
            height = node.chain.height
            state = node.chain.state
            queued = {t["channel"]: t for t in node.mempool.values()
                      if t["type"] == "channel_claim" and t["pub"] == self.key.pub_hex}
            with self.lock:
                due = []
                for cid, e in self.book.items():
                    ch = state.channels.get(cid)
                    v = e["voucher"]
                    if ch is None or not v or v[0] <= ch["paid"] or height + 1 > ch["expires"]:
                        continue
                    old = queued.get(cid)
                    if old is not None and (cid not in close or old.get("close")):
                        continue                  # already on its way
                    unclaimed = v[0] - ch["paid"]
                    if unclaimed >= fee and (force or cid in close or unclaimed >= CLAIM_MIN
                                             or height + CLAIM_MARGIN >= ch["expires"]):
                        due.append((cid, v[0], v[1], old))
        done = []
        for cid, total, sig, old in due:
            extra = {"close": True} if cid in close else {}
            if old is not None:                   # replace-by-fee: the closing claim takes its place
                nonce, claim_fee = old["nonce"], max(fee, old["fee"] + max(params.MIN_TX_FEE, old["fee"] // 10))
            else:
                nonce, claim_fee = node.pending_nonce(self.address), fee
            t = make_tx(self.key, node.chain.chain_id, "channel_claim", nonce, claim_fee, channel=cid,
                        total=total, voucher=sig, **extra)
            try:
                node.submit_tx(t, stem=False)
                done.append(cid)
                self.stats["claims"] += 1
                self.stats["closed"] += bool(extra)
            except (InvalidTx, MissingData) as e:
                node.log(f"[provider] claim on channel {cid[:12]}... refused: {e}")
        return done

    def close(self, cid: str, total, sig) -> dict:
        """The buyer is done: take its final voucher, claim it and close the channel, so
        what is left goes back to the buyer now rather than at expiry."""
        with self.node.lock:
            ch = self.node.chain.state.channels.get(cid) if is_hex(cid, 64) else None
            ch = dict(ch) if ch else None
        if ch is None or ch["provider"] != self.address:
            raise LookupError("no open channel to this provider with that id")
        self.take_voucher(cid, ch, total, sig)
        if not self.claim_due(close={cid}):
            raise ChannelEnding("nothing left to claim that pays the claim's fee: the rest returns at expiry")
        return {"closing": True}

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
        """Drop the books of channels that are over for good: past their expiry by more than
        a reorg can undo (a channel briefly missing from the state, in a reorg, keeps them)."""
        with self.node.lock:
            height = self.node.chain.height
        with self.lock:
            for cid in [c for c, e in self.book.items() if height > e["expires"] + params.MAX_REORG_DEPTH]:
                del self.book[cid]
                self.dirty = True

    # -- server --

    def start(self) -> None:
        self.server = BoundedServer((self.host, self.port), _handler(self))
        self.port = self.server.server_address[1]
        self.running = True
        if self.url is None and self.node.portmap:   # behind a router: forward our port and advertise that
            m = self.node.map_port(self.port)
            if m is not None:
                self.url = f"http://{m.external_ip}:{m.external_port}"
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


def relay_post(node_url: str, provider: str, data: bytes, timeout: float = 300) -> bytes:
    """Hand a sealed request to our own node, which has a peer carry it (POST /relay)."""
    from .node import CLIENT_HEADER
    req = urllib.request.Request(f"{node_url.rstrip('/')}/relay", data=data, method="POST",
                                 headers={"Content-Type": "application/octet-stream", CLIENT_HEADER: "1",
                                          "X-Relay-To": provider})
    try:
        with NO_REDIRECTS.open(req, timeout=timeout) as r:
            return r.read(sealed.MAX_SEALED + 1)
    except urllib.error.HTTPError as e:
        try:
            reason = json.loads(e.read()).get("error", e.reason)
        except (ValueError, OSError):
            reason = e.reason
        raise OSError(f"relay failed: {reason}") from None


def _handler(provider: Provider):
    max_body = 2 * provider.max_dim * provider.max_dim

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = IDLE_SECONDS            # a kept-alive connection holds a server slot: not for long

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

        def _refuse(self, code: int, error: str):
            """Answer without keeping the body (it may be large and unpaid), then hang up."""
            self.close_connection = True
            self.send_response(code)
            data = json.dumps({"error": error}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self._drain()

        def _drain(self):
            """Read and drop the body before hanging up. Clients send it after the headers,
            so it may still be arriving; a socket closed on arriving data answers with a
            reset, and on Windows the reset discards the answer the client has not read
            yet -- a buyer would see "connection aborted" instead of "pay first" or "no
            such channel yet". Nothing is kept or computed, at most max_body bytes are
            read, and for at most DRAIN_SECONDS."""
            raw = self.headers.get("Content-Length", "")
            left = min(int(raw), max_body) if raw.isdigit() else 0
            end = time.monotonic() + DRAIN_SECONDS
            try:
                self.connection.shutdown(socket.SHUT_WR)          # the answer is complete
                self.connection.settimeout(DRAIN_SECONDS)
                while left > 0 and time.monotonic() < end:
                    chunk = self.rfile.read1(min(left, 1 << 16))
                    if not chunk:
                        break
                    left -= len(chunk)
            except OSError:
                pass

        def _guard(self, fn, unread: bool = False):
            fail = self._refuse if unread else (lambda code, error: self._send(code, {"error": error}))
            try:
                return fn()
            except PaymentRequired as e:
                return fail(402, str(e))
            except LookupError as e:
                return fail(404, str(e))
            except ChannelEnding as e:
                return fail(409, str(e))
            except (ValueError, KeyError, TypeError) as e:
                return fail(400, str(e))
            except Exception:
                return fail(500, "internal error")

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
            limit = {"/matmul_mod": 4 * max_body, "/sealed": sealed.MAX_SEALED}.get(path, max_body)
            if not raw.isdigit() or int(raw) > limit:
                return self._refuse(413, "too large")
            if path == "/matmul":
                return self._matmul(int(raw))
            if path == "/matmul_mod":
                return self._matmul(int(raw), modular=True)
            if path == "/sealed":
                try:
                    return self._send(200, provider.serve_sealed(self.rfile.read(int(raw))))
                except sealed.SealError as e:
                    return self._send(400, {"error": str(e)})
            if int(raw) > 4096:
                return self._refuse(413, "too large")
            body = self.rfile.read(int(raw))
            if path == "/voucher":
                return self._guard(lambda: self._voucher(body))
            if path == "/close":
                return self._guard(lambda: self._close(body))
            return self._send(404, {"error": "not found"})

        def _matmul(self, length: int, modular: bool = False):
            cid = self.headers.get("X-Channel", "")
            admitted = []
            modulus = []

            def check():
                shape = tuple(int(x) for x in self.headers.get("X-Shape", "").split(","))
                modulus.append(int(self.headers.get("X-Modulus", "")) if modular else None)
                if len(shape) != 3 or length != product_bytes(shape, modulus[0]):
                    raise ValueError("X-Shape must be m,k,n and the body A and B of that shape")
                total = self.headers.get("X-Voucher-Total")
                admitted.append((shape,) + provider.admit(cid, shape, int(total) if total else None,
                                                          self.headers.get("X-Voucher")))
            self._guard(check, unread=True)
            if not admitted:
                return                                # refused before reading the body
            shape, ticket, owed = admitted[0]
            try:
                body = self.rfile.read(length)
                c = provider.compute(shape, body) if modulus[0] is None else provider.compute_mod(shape, modulus[0], body)
            except BaseException:
                provider.release(cid, ticket)
                self.close_connection = True
                raise
            return self._send(200, c, {"X-Owed": str(owed)})

        def _voucher(self, body: bytes):
            m = json.loads(body)
            ch = provider._channel(m["channel"]) if m.get("channel") else None
            if ch is None:
                raise LookupError("no channel given")
            return self._send(200, {"vouched": provider.take_voucher(m["channel"], ch, m.get("total"),
                                                                     m.get("voucher"))})

        def _close(self, body: bytes):
            m = json.loads(body)
            return self._send(200, provider.close(m.get("channel", ""), m.get("total"), m.get("voucher")))
    return Handler


# --- the buyer ------------------------------------------------------------------

class ComputeChannel:
    """A buyer's channel to one provider: open it, then call matmul like numpy."""

    def __init__(self, provider_url: str, key: Key, node_url: str, timeout: float = 300, offer: dict | None = None,
                 relay: bool = False):
        """With relay=True every request is sealed to the provider (green/sealed.py) and
        carried by a random peer of our node (p2p.py): the provider never sees where it
        comes from, the peer never sees what it is. The offer then comes from the
        network's signed list, so we never contact the provider directly either."""
        self.url = provider_url.rstrip("/")
        self.node = node_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.relay = relay
        if offer is None:
            if relay:
                raise ValueError("a relayed channel needs the provider's signed offer from the network")
            offer = http_get(f"{self.url}/offer", timeout=30)
        elif "chain" not in offer:
            offer = dict(offer, chain=http_get(f"{self.node}/status", timeout=30)["genesis"])
        if relay and not is_hex(offer.get("kx"), 64):
            raise ProviderError("this provider cannot take sealed requests")
        self.offer = offer
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
        fields = {"provider": self.offer["provider"], "amount": amount, "expires": st["height"] + 1 + blocks}
        fee = params.MIN_TX_FEE
        try:                                      # what gets it into the next blocks when they are full
            rate = http_get(f"{self.node}/v1/fee?blocks=2", timeout=30)["fee_rate"]
            size = txmod.tx_size(make_tx(self.key, st["genesis"], "channel_open", nonce, fee, **fields))
            fee = max(fee, int(rate * size) + 1 if rate else 0)
        except (OSError, ValueError, KeyError):
            pass
        t = make_tx(self.key, st["genesis"], "channel_open", nonce, fee, **fields)
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

    def _exchange(self, path: str, headers: dict, body: bytes) -> tuple[int, bytes]:
        """One request: (status, answer). Sealed and relayed, or straight to the provider."""
        if self.relay:
            req, key = sealed.seal_request(self.offer["kx"], {"path": path, "headers": headers}, body)
            data = relay_post(self.node, self.offer["provider"], req, timeout=self.timeout)
            try:
                header, out = sealed.open_answer(key, data)
            except sealed.SealError as e:
                raise ProviderError(str(e)) from None
            return int(header.get("status", 500)), out
        conn = self._connection()
        try:
            conn.request("POST", path, body=body, headers=headers)
            r = conn.getresponse()
            return r.status, r.read(16 * MAX_DIM * MAX_DIM + 1)
        except (OSError, http.client.HTTPException):
            conn.close()
            self.local.conn = None
            raise

    def _post(self, body: bytes, headers: dict, path: str = "/matmul") -> bytes:
        end = time.time() + 60
        retried = False
        while True:
            try:
                status, data = self._exchange(path, headers, body)
            except (OSError, http.client.HTTPException) as e:
                if retried:
                    if self.relay:
                        raise RelayError(f"the request could not be carried: {e}") from None
                    raise ProviderError(f"cannot reach the provider: {e}") from None
                retried = True                    # a kept-alive connection the provider closed: once more
                continue
            if status == 200:
                return data
            reason = data.decode(errors="replace")
            if status == 404 and time.time() < end:     # the provider's node has not seen the channel yet
                time.sleep(0.5)
                continue
            if status == 402:
                raise PaymentRequired(reason)
            raise ProviderError(f"provider answered {status}: {reason}")

    def _call(self, path: str, obj: dict) -> dict:
        """A small JSON request (voucher, close)."""
        if not self.relay:
            return http_post(f"{self.url}{path}", obj, timeout=30)
        status, out = self._exchange(path, {"Content-Type": "application/json"}, json.dumps(obj).encode())
        if status != 200:
            raise ProviderError(f"provider answered {status}: {out.decode(errors='replace')}")
        return json.loads(out)

    def matmul_mod(self, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
        """(a @ b) mod p for entries in [0, p), computed by the provider and checked here."""
        if self.bad:
            raise WrongResult("this provider returned a wrong result before")
        a = np.ascontiguousarray(a, dtype="<i4")
        b = np.ascontiguousarray(b, dtype="<i4")
        (m, k), (k2, n) = a.shape, b.shape
        if k != k2:
            raise ValueError("inner dimensions differ")
        started = time.time()
        headers = {"X-Channel": self.cid, "X-Shape": f"{m},{k},{n}", "X-Modulus": str(p),
                   "Content-Type": "application/octet-stream", **self.voucher_headers()}
        data = self._post(a.tobytes() + b.tobytes(), headers, "/matmul_mod")
        if len(data) != 4 * m * n:
            self.bad = True
            raise WrongResult("the provider's answer has the wrong size")
        c = np.frombuffer(data, dtype="<i4").reshape(m, n).astype(np.int64)
        checked = time.time()
        if not verify_product_mod(a, b, c, p):
            self.bad = True
            raise WrongResult("the provider's modular product failed Freivalds' test; it is not paid for it")
        with self.lock:
            self.owed += price_of(m, k, n, self.offer["price"])
            self.stats["products"] += 1
            self.stats["macs"] += m * k * n
            self.stats["seconds"] += time.time() - started
            self.stats["verify_seconds"] += time.time() - checked
        return c

    def matmul_many(self, pairs: list) -> list[np.ndarray]:
        """Several independent products at once (each on its own connection, at most the
        provider's credit in flight): over a slow link they cost one round trip, not several."""
        if len(pairs) <= 1 or self.mean_seconds() < PARALLEL_MIN_SECONDS:
            return [self.matmul(a, b) for a, b in pairs]
        with self.lock:
            if not hasattr(self, "pool"):
                self.pool = ThreadPoolExecutor(max(1, min(8, self.offer.get("credit", CREDIT))),
                                               thread_name_prefix="channel")
        return [f.result() for f in [self.pool.submit(self.matmul, a, b) for a, b in pairs]]

    def mean_seconds(self) -> float:
        """Average time per product so far (0 before the first): small on a fast link."""
        with self.lock:
            return self.stats["seconds"] / self.stats["products"] if self.stats["products"] else 0.0

    def close(self) -> bool:
        """Done with this provider: send the final voucher and ask it to close the channel,
        so the unspent part comes back now instead of at expiry. False if it declined
        (the voucher is still delivered; the rest then returns at expiry)."""
        with self.lock:
            total = self.owed
        body = {"channel": self.cid, "total": total,
                "voucher": txmod.sign_voucher(self.key, self.chain_id, self.cid, total) if total else None}
        try:
            self._call("/close", body)
            return True
        except (urllib.error.HTTPError, ProviderError):
            self.settle()
            return False

    def settle(self) -> int:
        """Send the voucher for everything received (the provider then claims it)."""
        with self.lock:
            total = self.owed
        if total:
            self._call("/voucher", {"channel": self.cid, "total": total,
                                    "voucher": txmod.sign_voucher(self.key, self.chain_id, self.cid, total)})
        return total


class ComputeCluster:
    """Several channels used as one. A large product is split by rows across the
    providers and computed in parallel; small ones go to the providers in turn. A
    provider that returns a wrong result is dropped and its share is computed by
    another one."""

    def __init__(self, channels: list[ComputeChannel], split_macs: int = 4 * UNIT_MACS):
        self.channels = list(channels)
        self.split_macs = split_macs
        self.pool = ThreadPoolExecutor(max(2, 2 * len(self.channels)), thread_name_prefix="compute")
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

    def matmul_many(self, pairs: list) -> list[np.ndarray]:
        """Independent products in parallel, spread over the providers."""
        live = self.live()
        if len(pairs) <= 1 or max(c.mean_seconds() for c in live) < PARALLEL_MIN_SECONDS:
            return [self.matmul(a, b) for a, b in pairs]
        futures = [self.pool.submit(self._one, a, b, live[i % len(live)]) for i, (a, b) in enumerate(pairs)]
        return [f.result() for f in futures]

    def close(self) -> int:
        """Close every channel (what is unspent comes back now); returns how many closed."""
        return sum(c.close() for c in self.channels if c.cid)

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
                    blocks: int = 4 * params.MIN_CHANNEL_BLOCKS, allow_local: bool = False,
                    relay: bool = False) -> "ComputeCluster":
        """Find providers on the network (GET /providers on our node), take the `count`
        cheapest, and open a channel of `amount_each` to each (relay: every request sealed
        and carried by a peer; see ComputeChannel)."""
        offers = http_get(f"{node_url.rstrip('/')}/providers", timeout=30)
        picked = []
        for o in offers:
            host = urlparse(o["url"]).hostname or ""
            if (max_price is not None and o["price"] > max_price) or (not allow_local and _private(host)):
                continue
            if relay and not is_hex(o.get("kx"), 64):
                continue                          # an older provider: no sealed requests
            try:
                ch = ComputeChannel(o["url"], key, node_url, offer=o if relay else None, relay=relay)
            except (OSError, ValueError, ProviderError):
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
