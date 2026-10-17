"""Peer-to-peer protocol: encrypted, authenticated links between nodes,
announce-then-fetch relay, and headers-first sync. Specified in SPEC.md §9.

Transport. Each side sends a fresh X25519 key (32 bytes, top bit random so the
bytes do not stand out); both derive keys with HKDF-SHA256 salted with the
network's genesis hash, so nodes of another network cannot even read the first
frame. Every frame is ChaCha20-Poly1305: a 4-byte length sealed on its own
(20 bytes on the wire) followed by the sealed body, nonces are frame counters,
and both keys are re-derived every REKEY_FRAMES frames for forward secrecy
within long sessions. Nothing on the wire is plaintext.

Identity. A node has a long-term Ed25519 key (data_dir/node_key, separate from
any wallet). Its first frame, `version`, signs the session id together with
its role, which proves it holds the key and binds the key to this connection.
A peer given as <node_id>@host:port must present exactly that key, so a
man in the middle cannot pose as it; other peers are encrypted and identified
on first contact (the session id can be compared out of band).

Relay. Transactions and blocks are announced by hash (`inv`) and fetched only
by peers that lack them (`getdata`), so each crosses each link at most once.
Transaction announcements are batched per peer at random (exponential)
intervals. Two of the outbound slots relay blocks only, which hides them from
transaction-based topology probing.

Origin privacy (Dandelion++). A node's own transactions are not announced: they
travel a "stem" of single hops (`dtx`) to one outbound peer after another, and
only a node that is a diffuser for the current epoch (or has nowhere to pass
it) announces it normally. An observer that sees the announcement learns where
the stem ended, not where it began. Stem routes are fixed per epoch and per
incoming link, stem transactions are kept apart from the mempool and never
served to `getdata`, and every node on a stem diffuses the transaction itself
if it has not seen it announced when its random embargo timer runs out.

Compact blocks. A new block goes to peers as its header plus 6-byte ids of its
transactions, salted per message (`cmpct`); a peer rebuilds it from its own
mempool and asks only for what it lacks (`getblocktxn`/`blocktxn`). Block
propagation then costs a few kilobytes and, usually, no round trip.

Sync. Headers come first (`getheaders`/`headers`, block locators); each one is
checked against every header rule except its work unit, and a random sample
of work units plus the last one is recomputed, so a fake header chain is
caught before any block is fetched for it. Blocks are then downloaded from
several peers at once inside a moving window, their work units recomputed in
parallel threads, and connected in order with full validation.

Defence. Every frame is size-checked before it is read (large frames only for
data we asked that peer for), every read has a deadline, handshakes time out,
inbound slots are capped per network group with eviction, outbound peers are
spread over network groups, address gossip is rate limited, every peer's
message queue and send queue are bounded, and peers that send invalid blocks
or headers are banned.
"""

import hashlib
import ipaddress
import json
import os
import queue
import random
import select
import socket
import struct
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand

from . import params, snapshot, tx as txmod, work
from .chain import (MAX_JOB_BLOB, InvalidBlock, MissingData, block_hash, seed_of, share_id, target_of,
                    work_for_target)
from .crypto import address_of, canonical, hhex, is_hex, merkle_root
from .crypto import verify as crypto_verify
from .tx import InvalidTx
from .verify import ProcessChecker

PROTOCOL_VERSION = 1
AGENT = "green/0.20.0"

MAX_OUTBOUND = 8               # outbound connections the node keeps
BLOCK_RELAY_ONLY = 2           # ...of which relay blocks only
MAX_INBOUND = 32
MAX_INBOUND_PER_GROUP = 4      # per /16 (IPv4) or /32 (IPv6); local addresses are exempt
HANDSHAKE_SECONDS = 10
MAX_PENDING_HANDSHAKES = 32    # inbound connections still in their handshake, in total
MAX_PENDING_PER_HOST = 4       # ...and per address (local addresses exempt)
FRAME_SECONDS = 120            # a frame must arrive within this (plus time for its size)
IDLE_SECONDS = 600             # no frame at all for this long: disconnect
PING_SECONDS = 60
PONG_SECONDS = 300
MAX_FRAME = 4 * 1024 * 1024    # blocks, headers, transactions; blobs only on request
MAX_SEND_QUEUE = 64 * 1024 * 1024
MAX_PEER_QUEUE = 1000          # messages waiting to be processed, per peer
REKEY_FRAMES = 4096
MAX_INV = 1000
MAX_HEADERS = 2000
MAX_LOCATOR = 64
MAX_ADDR = 1000
KNOWN_INV = 20_000             # hashes remembered per peer as "they have it"
TRICKLE_MEAN_SECONDS = 0.5     # mean delay before a transaction announcement
TX_REQUEST_SECONDS = 10        # wait this long for a requested transaction, then ask someone else
BLOCK_REQUEST_SECONDS = 15
HEADERS_REQUEST_SECONDS = 30
BLOCK_WINDOW = 256             # blocks ahead of the tip that may be in flight or waiting
BLOCKS_PER_PEER = 16
HEADER_INDEX_MAX = 50_000      # headers held ahead of the chain
WORK_SAMPLES = 8               # work units recomputed per headers message (plus the last one)
VERIFY_THREADS = min(8, os.cpu_count() or 2)
PROCESS_BACKLOG = 32           # downloaded blocks waiting: check work in processes (green/verify.py)
UNCONNECTING_LIMIT = 8
FAST_SYNC_PEERS = 2            # a fast sync picks its snapshot once this many peers are connected...
FAST_SYNC_GRACE = 10.0         # ...or this many seconds after starting
BAN_SCORE = 100
BAN_SECONDS = 24 * 3600
ADDR_RATE = 0.1                # addresses accepted per second per peer, after a burst of MAX_ADDR
ADDR_NEW_MAX = 4096
ADDR_TRIED_MAX = 1024
ADDR_PER_SOURCE = 64           # new addresses kept from one source network group
RETRY_SECONDS = 60             # do not redial an address sooner than this
BLOB_UPLOAD_PER_MINUTE = 256 * 1024 * 1024
WORKERS = 8                    # message-processing threads
DANDELION_EPOCH_SECONDS = 600  # stem routes are redrawn this often
DANDELION_FLUFF_PROB = 0.1     # chance per epoch that a node diffuses the stem transactions it is passed
DANDELION_DESTINATIONS = 2     # outbound peers that carry stem transactions
EMBARGO_SECONDS = 15           # a stem transaction not seen announced by then (plus a random delay)...
EMBARGO_MEAN_EXTRA = 5         # ...is announced by this node itself
MAX_STEMPOOL = 1000
SHORT_ID_BYTES = 6
MAX_OFFERS = 1000              # compute offers kept (one per provider address)
OFFER_TTL = 3600               # an offer older than this is dropped; providers renew theirs
OFFER_MIN_BALANCE = 10_000_000  # yaprak a provider must hold for its offer to be relayed (0.1 GRN)
OFFERS_PER_MESSAGE = 100
OFFER_RATE = 0.2               # offers accepted per second per peer, after a burst of MAX_OFFERS
FEATURES = ["dtx", "cmpct", "offers", "shares"]   # optional messages this node understands

CMDS = ["version", "ping", "pong", "getaddr", "addr", "inv", "getdata", "notfound", "tx", "block",
        "getheaders", "headers", "getblob", "blob", "noblob", "getmempool", "getsnapinfo", "snapinfo",
        "dtx", "cmpct", "getblocktxn", "blocktxn", "offers", "share"]
CODE = {name: i + 1 for i, name in enumerate(CMDS)}
NAME = {i: name for name, i in CODE.items()}


class Misbehaving(Exception):
    def __init__(self, score: int, reason: str):
        super().__init__(reason)
        self.score = score


# --- addresses -----------------------------------------------------------------

def parse_address(text) -> tuple[str, int, str | None] | None:
    """'host:port', '[v6]:port' or '<node id>@host:port' -> (host, port, id or None)."""
    if not isinstance(text, str) or len(text) > 300:
        return None
    node_id = None
    if "@" in text:
        node_id, text = text.split("@", 1)
        if not is_hex(node_id, 64):
            return None
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        if not rest.startswith(":"):
            return None
        port_s = rest[1:]
    else:
        host, sep, port_s = text.rpartition(":")
        if not sep or ":" in host:
            return None
    if not host or not port_s.isdigit() or not 0 < int(port_s) < 65536:
        return None
    if any(c.isspace() or c in "/?#@" for c in host):
        return None
    return host.lower(), int(port_s), node_id


def addr_key(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def is_local(host: str) -> bool:
    """Loopback or private: exempt from the per-group limits (test nets, LANs)."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost"
    return ip.is_loopback or ip.is_private or ip.is_link_local


def gossip_ok(host: str, source: str) -> bool:
    """May an address heard from `source` go in the address book? Only IP addresses (a
    name would make us look up whatever a stranger chose), and only public ones: a
    private address from the internet would have us dial into our own network. Local
    peers may tell us local addresses (test networks, LANs)."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_global or is_local(source) and is_local(host)


def netgroup(host: str) -> str:
    """Peers in one group are likely controlled together: /16 for IPv4, /32 for IPv6."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "name:" + host
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv4Address):
        return "v4:" + ".".join(str(ip).split(".")[:2])
    return "v6:" + ip.exploded[:9]


class BoundedSet:
    """A set that forgets its oldest members beyond a size."""

    def __init__(self, size: int):
        self.size = size
        self._d: OrderedDict = OrderedDict()

    def add(self, x) -> None:
        self._d[x] = None
        self._d.move_to_end(x)
        while len(self._d) > self.size:
            self._d.popitem(last=False)

    def __contains__(self, x) -> bool:
        return x in self._d

    def discard(self, x) -> None:
        self._d.pop(x, None)

    def clear(self) -> None:
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


class AddrMan:
    """Address book: addresses heard of ("new") and ones we connected to ("tried").

    Bounded in size, and one source network group can only fill a small part of
    it, so a single attacker cannot flood it with its own addresses (the core of
    an eclipse attack).
    """

    def __init__(self, path: str | None):
        self.path = path
        self.entries: dict[str, dict] = {}
        self.lock = threading.Lock()
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    for e in json.load(f):
                        if parse_address(addr_key(e["host"], e["port"])):
                            self.entries[addr_key(e["host"], e["port"])] = e
            except (OSError, ValueError, KeyError, TypeError):
                self.entries = {}

    def add(self, host: str, port: int, node_id: str | None = None, source: str | None = None,
            seen: float | None = None) -> bool:
        key = addr_key(host, port)
        now = time.time()
        with self.lock:
            e = self.entries.get(key)
            if e is not None:
                e["seen"] = max(e["seen"], min(seen or now, now))
                if node_id and not e.get("id"):
                    e["id"] = node_id
                return False
            group = netgroup(source) if source else "self"
            new = [x for x in self.entries.values() if not x["tried"]]
            if sum(1 for x in new if x["source"] == group) >= ADDR_PER_SOURCE:
                return False
            if len(new) >= ADDR_NEW_MAX:
                oldest = min(new, key=lambda x: x["seen"])
                del self.entries[addr_key(oldest["host"], oldest["port"])]
            self.entries[key] = {"host": host, "port": port, "id": node_id, "seen": min(seen or now, now),
                                 "tried": False, "attempts": 0, "last_try": 0.0, "last_ok": 0.0, "source": group}
            return True

    def forget(self, key: str) -> None:
        with self.lock:
            self.entries.pop(key, None)

    def attempt(self, key: str) -> None:
        with self.lock:
            e = self.entries.get(key)
            if e:
                e["attempts"] += 1
                e["last_try"] = time.time()
                if e["attempts"] >= 10 and not e["last_ok"]:
                    del self.entries[key]               # never worked: forget it

    def good(self, host: str, port: int, node_id: str) -> None:
        key = addr_key(host, port)
        with self.lock:
            e = self.entries.get(key)
            if e is None:
                e = self.entries[key] = {"host": host, "port": port, "id": node_id, "seen": time.time(),
                                         "tried": False, "attempts": 0, "last_try": 0.0, "last_ok": 0.0,
                                         "source": "self"}
            e.update(id=node_id, tried=True, attempts=0, last_ok=time.time(), seen=time.time())
            tried = [x for x in self.entries.values() if x["tried"]]
            if len(tried) > ADDR_TRIED_MAX:
                min(tried, key=lambda x: x["last_ok"])["tried"] = False

    def select(self, skip_keys: set, skip_groups: set) -> dict | None:
        """An address to dial: half the time a tried one, otherwise a new one."""
        now = time.time()
        with self.lock:
            ok = [e for e in self.entries.values()
                  if addr_key(e["host"], e["port"]) not in skip_keys and now - e["last_try"] >= RETRY_SECONDS
                  and (is_local(e["host"]) or netgroup(e["host"]) not in skip_groups)]
        if not ok:
            return None
        tried = [e for e in ok if e["tried"]]
        new = [e for e in ok if not e["tried"]]
        pool = tried if tried and (not new or random.random() < 0.5) else new
        return dict(random.choice(pool))

    def sample(self, n: int) -> list[dict]:
        cutoff = time.time() - 7 * 86400
        with self.lock:
            fresh = [e for e in self.entries.values() if e["seen"] >= cutoff]
        random.shuffle(fresh)
        return [{"host": e["host"], "port": e["port"], "id": e["id"], "seen": int(e["seen"])} for e in fresh[:n]]

    def save(self) -> None:
        if not self.path:
            return
        with self.lock:
            data = list(self.entries.values())
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.entries)


# --- the encrypted transport -----------------------------------------------------

def _rekey(key: bytes) -> bytes:
    return HKDFExpand(hashes.SHA256(), 32, b"grn-p2p-rekey").derive(key)


class Cipher:
    """One direction of a session: sealed length, sealed body, counter nonces."""

    def __init__(self, len_key: bytes, body_key: bytes):
        self.len_key, self.body_key = len_key, body_key
        self.n = 0
        self._load()

    def _load(self) -> None:
        self.len_aead = ChaCha20Poly1305(self.len_key)
        self.body_aead = ChaCha20Poly1305(self.body_key)

    def _nonce(self) -> bytes:
        return b"\0\0\0\0" + struct.pack("<Q", self.n)

    def _advance(self) -> None:
        self.n += 1
        if self.n == REKEY_FRAMES:
            self.len_key, self.body_key = _rekey(self.len_key), _rekey(self.body_key)
            self.n = 0
            self._load()

    def seal(self, body: bytes) -> bytes:
        nonce = self._nonce()
        frame = self.len_aead.encrypt(nonce, struct.pack("<I", len(body)), None) + \
            self.body_aead.encrypt(nonce, body, None)
        self._advance()
        return frame

    def open_length(self, sealed: bytes) -> int:
        return struct.unpack("<I", self.len_aead.decrypt(self._nonce(), sealed, None))[0]

    def open_body(self, sealed: bytes) -> bytes:
        body = self.body_aead.decrypt(self._nonce(), sealed, None)
        self._advance()
        return body


def _recv_exact(sock, n: int, deadline: float, alive=lambda: True) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError("read deadline passed")
        ready, _, _ = select.select([sock], [], [], min(left, 1.0))
        if not alive():
            raise ConnectionError("closed")
        if not ready:
            continue
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        buf += chunk
    return bytes(buf)


def key_exchange(sock, initiator: bool, genesis_hash: str):
    """Ephemeral X25519 in both directions -> (send Cipher, receive Cipher, session id)."""
    eph = X25519PrivateKey.generate()
    mine = eph.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    # The top bit of an X25519 key is always 0; send it random and mask it on receipt.
    sock.sendall(mine[:31] + bytes([mine[31] | (os.urandom(1)[0] & 0x80)]))
    theirs = _recv_exact(sock, 32, time.time() + HANDSHAKE_SECONDS)
    theirs = theirs[:31] + bytes([theirs[31] & 0x7F])
    shared = eph.exchange(X25519PublicKey.from_public_bytes(theirs))
    if shared == bytes(32):
        raise ConnectionError("degenerate key exchange")
    i_pub, r_pub = (mine, theirs) if initiator else (theirs, mine)
    okm = HKDF(hashes.SHA256(), 160, b"grn-p2p-v1" + bytes.fromhex(genesis_hash), i_pub + r_pub).derive(shared)
    k = [okm[i * 32:(i + 1) * 32] for i in range(5)]
    i2r, r2i = Cipher(k[0], k[1]), Cipher(k[2], k[3])
    return (i2r, r2i, k[4]) if initiator else (r2i, i2r, k[4])


def auth_message(session_id: bytes, initiator: bool) -> bytes:
    return b"grn-p2p-auth-v1" + session_id + (b"I" if initiator else b"R")


def load_node_key(data_dir: str | None) -> Ed25519PrivateKey:
    """The node's long-term identity (not a wallet: it holds no coins)."""
    if data_dir is None:
        return Ed25519PrivateKey.generate()
    path = os.path.join(data_dir, "node_key")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Ed25519PrivateKey.from_private_bytes(f.read())
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                            serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return key


def pub_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def short_id(key: bytes, tid: str) -> str:
    return hashlib.blake2b(bytes.fromhex(tid), key=key, digest_size=SHORT_ID_BYTES).hexdigest()


def compact_key(bh: str, salt: str) -> bytes:
    """Short ids are keyed by the block and a fresh salt per message, so nobody can
    craft transactions whose ids collide in advance."""
    return bytes.fromhex(bh) + bytes.fromhex(salt)


def offer_bytes(chain_id: str, offer: dict) -> bytes:
    """What a provider signs to advertise its compute (green/compute.py)."""
    return canonical(["grn-offer", chain_id, {k: v for k, v in offer.items() if k != "sig"}])


OFFER_FIELDS = {"provider", "pub", "url", "price", "max_dim", "credit", "time", "sig"}


def offer_problem(chain_id: str, offer, now: float) -> str | None:
    """Why `offer` is not a valid signed offer (None if it is). Balance is checked separately."""
    if not isinstance(offer, dict) or set(offer) != OFFER_FIELDS:
        return "bad fields"
    url = offer["url"]
    hostport = parse_address(url.split("://", 1)[1].split("/")[0]) if isinstance(url, str) and "://" in url else None
    if not (len(url) <= 200 and url.split("://")[0] in ("http", "https") and hostport is not None):
        return "bad url"
    try:
        ipaddress.ip_address(hostport[0])     # an IP, not a name: a name could resolve into the buyer's network
    except ValueError:
        return "bad url"
    if not (txmod.is_uint(offer["price"]) and offer["price"] >= 1 and txmod.is_uint(offer["max_dim"])
            and 1 <= offer["max_dim"] <= 4096 and txmod.is_uint(offer["credit"]) and 1 <= offer["credit"] <= 1000
            and txmod.is_uint(offer["time"])):
        return "bad terms"
    if not now - OFFER_TTL <= offer["time"] <= now + 60:
        return "stale"
    if not is_hex(offer["pub"], 64) or address_of(offer["pub"]) != offer["provider"]:
        return "the key is not the provider's"
    if not crypto_verify(offer["pub"], offer_bytes(chain_id, offer), offer["sig"]):
        return "bad signature"
    return None


def encode(cmd: str, payload) -> bytes:
    if cmd == "blob":
        return bytes([CODE[cmd]]) + payload
    return bytes([CODE[cmd]]) + canonical(payload)


# --- one connection ----------------------------------------------------------------

class Peer:
    def __init__(self, p2p: "P2P", sock, host: str, port: int, inbound: bool, block_only: bool = False):
        self.p2p = p2p
        self.sock = sock
        self.host, self.port = host, port            # the socket's remote address
        self.inbound = inbound
        self.block_only = block_only                 # we chose not to relay transactions on this link
        self.listen_port = 0
        self.node_id = ""
        self.session_id = b""
        self.agent = ""
        self.relay = True                            # they want transaction announcements
        self.features: set = set()                   # optional messages they understand (FEATURES)
        self.best_height = 0
        self.best_work = 0
        self.first_block = 0                         # lowest height it has block bodies for
        self.snap_offers: list | None = None         # its answer to getsnapinfo
        self.known = BoundedSet(KNOWN_INV)
        self.trickle: list[str] = []
        self.next_trickle = 0.0
        self.inflight_blocks: dict[str, float] = {}
        self.blob_waits: dict[str, dict] = {}        # blob hash -> {"max", "event", "data"}
        self.headers_asked = 0.0                     # when we last sent getheaders, 0 if none open
        self.unconnecting = 0
        self.score = 0
        self.lock = threading.Lock()
        self.out_q: queue.Queue = queue.Queue()
        self.out_bytes = 0
        self.in_q: deque = deque()
        self.scheduled = False
        self.alive = True
        self.connected_at = time.time()
        self.last_recv = time.time()
        self.ping_nonce = None
        self.ping_sent = 0.0
        self.ping_ms = None
        self.last_block = 0.0
        self.last_tx = 0.0
        self.bytes_in = self.bytes_out = 0
        self.sent_addr = False
        self.sent_mempool = False
        self.addr_tokens = float(MAX_ADDR)
        self.offer_tokens = float(MAX_OFFERS)
        self.offer_time = time.time()
        self.addr_time = time.time()
        self.blob_budget = float(BLOB_UPLOAD_PER_MINUTE)
        self.blob_time = time.time()
        self.send_cipher = self.recv_cipher = None

    @property
    def name(self) -> str:
        return addr_key(self.host, self.port)

    @property
    def listen_addr(self) -> str | None:
        return addr_key(self.host, self.listen_port) if self.listen_port else None

    def info(self) -> dict:
        return {"addr": self.name, "listen": self.listen_addr, "id": self.node_id, "inbound": self.inbound,
                "block_only": self.block_only or not self.relay, "agent": self.agent,
                "height": self.best_height, "ping_ms": self.ping_ms,
                "session": self.session_id.hex()[:16], "connected": int(time.time() - self.connected_at),
                "bytes_in": self.bytes_in, "bytes_out": self.bytes_out, "score": self.score,
                "features": sorted(self.features)}

    # -- sending --

    def send(self, cmd: str, payload=None) -> None:
        if not self.alive:
            return
        body = encode(cmd, payload if payload is not None else {})
        with self.lock:
            self.out_bytes += len(body)
            too_much = self.out_bytes > MAX_SEND_QUEUE
        if too_much:
            self.close("not reading what we send")
            return
        self.out_q.put(body)

    def _writer(self) -> None:
        while self.alive:
            try:
                body = self.out_q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                frame = self.send_cipher.seal(body)
                self.sock.sendall(frame)
                self.bytes_out += len(frame)
                with self.lock:
                    self.out_bytes -= len(body)
            except OSError:
                self.close("send failed")
                return

    # -- receiving --

    def _reader(self) -> None:
        try:
            while self.alive:
                sealed_len = _recv_exact(self.sock, 20, time.time() + IDLE_SECONDS, lambda: self.alive)
                length = self.recv_cipher.open_length(sealed_len)
                with self.lock:
                    allowed = max([MAX_FRAME] + [w["max"] + 1 for w in self.blob_waits.values()])
                if length < 1 or length > allowed:
                    raise Misbehaving(BAN_SCORE, f"frame of {length} bytes")
                deadline = time.time() + FRAME_SECONDS + length / (128 * 1024)
                body = self.recv_cipher.open_body(_recv_exact(self.sock, length + 16, deadline, lambda: self.alive))
                self.bytes_in += 36 + length
                self.last_recv = time.time()
                cmd = NAME.get(body[0])
                if cmd is None:
                    raise Misbehaving(20, f"unknown message type {body[0]}")
                self.p2p._receive(self, cmd, body[1:])
        except Misbehaving as e:
            self.p2p.punish(self, e.score, str(e))
        except InvalidTag:
            self.close("frame failed authentication")
        except (OSError, ConnectionError, TimeoutError, struct.error, ValueError) as e:
            self.close(f"{type(e).__name__}: {e}")

    def close(self, reason: str = "") -> None:
        with self.lock:
            if not self.alive:
                return
            self.alive = False
            for w in self.blob_waits.values():
                w["event"].set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        self.p2p._closed(self, reason)


# --- the network manager -------------------------------------------------------------

class P2P:
    def __init__(self, node, host: str = "127.0.0.1", port: int = 0, peers=(), data_dir: str | None = None,
                 listen: bool = True, max_outbound: int | None = None, seeds=None, fast_sync: bool = False):
        self.node = node
        self.chain = node.chain
        self.log = node.log
        self.host = host
        self.port = port
        self.listen = listen
        self.max_outbound = MAX_OUTBOUND if max_outbound is None else max_outbound
        self.data_dir = data_dir
        self.key = load_node_key(data_dir)
        self.node_id = pub_hex(self.key)
        self.addrman = AddrMan(os.path.join(data_dir, "peers.json") if data_dir else None)
        self.bans: dict[str, float] = self._load_bans()
        self.lock = threading.Lock()
        self.peers: dict[int, Peer] = {}           # id(peer) -> peer
        self.manual: dict[str, dict] = {}          # addresses we were told to keep connected
        self.connecting: set[str] = set()
        self.pending: dict[str, int] = {}          # host -> inbound handshakes in progress
        self.my_host: str | None = None            # how outbound peers see us
        self.public_port: int | None = None        # our port as seen from outside, when the router maps one
        self.running = False
        self.server = None
        self._ready: queue.Queue = queue.Queue()
        self.seeds = list(params.SEED_NODES if seeds is None else seeds)
        # relay
        self.tx_wait: dict[str, dict] = {}         # txid -> {"peers": [...], "asked": peer, "deadline"}
        self.rejected = BoundedSet(50_000)         # transactions found invalid since the last block
        self._rejected_tip = None
        # Dandelion++ (stempool guarded by node.lock, routes by self.lock)
        self.stempool: OrderedDict[str, dict] = OrderedDict()   # txid -> {"tx", "embargo", "local"}
        self.stem_epoch = 0.0
        self.stem_fluff = False
        self.stem_dest: list[Peer] = []
        self.stem_route: dict[str, Peer] = {}
        # compute offers (green/compute.py): provider address -> signed offer (guarded by self.lock)
        self.offers: OrderedDict[str, dict] = OrderedDict()
        self.offers_waiting: OrderedDict[str, dict] = OrderedDict()   # valid, but not funded on our chain yet
        # compact blocks waiting for missing transactions (guarded by node.lock)
        self.partial: dict[str, dict] = {}         # hash -> {"header", "txs", "missing", "peer", "time"}
        # headers-first sync (guarded by node.lock)
        self.hindex: dict[str, dict] = {}          # hash -> {"header", "work"}: checked, block not connected
        self.best = self.chain.tip                 # most-work header known (chain or index)
        self.requested: dict[str, tuple] = {}      # block hash -> (peer, time)
        self.downloaded: dict[str, tuple] = {}     # block hash -> (block, peer, work check), waiting to connect
        self._path: list[str] = []
        self._path_for = None
        self.connect_lock = threading.Lock()
        self._connect_wake = threading.Event()
        # fast sync: None (off), "waiting" (for the header chain), "running", "done", "failed"
        self.snap_state = "waiting" if fast_sync else None
        self.started = time.time()
        self.verify_pool = ThreadPoolExecutor(VERIFY_THREADS, thread_name_prefix="verify")
        blobs_dir = getattr(getattr(node, "blobs", None), "path", None)
        self.checker = ProcessChecker(blobs_dir) if blobs_dir and getattr(node, "verify_processes", True) else None
        self.stats = {"blocks_relayed": 0, "txs_relayed": 0, "headers_received": 0, "blocks_downloaded": 0,
                      "work_samples": 0, "prechecked": 0, "checked_in_processes": 0, "banned": 0, "stem_relayed": 0, "stem_fluffed": 0,
                      "embargo_fired": 0, "cmpct_sent": 0, "cmpct_received": 0, "cmpct_rebuilt": 0,
                      "cmpct_txs_requested": 0, "cmpct_fallbacks": 0}
        for text in peers:
            self.add_manual(text)

    # -- lifecycle --

    def start(self) -> None:
        self.running = True
        if self.listen:
            fam = socket.AF_INET6 if ":" in self.host else socket.AF_INET
            self.server = socket.socket(fam, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.host, self.port))
            self.server.listen(64)
            self.port = self.server.getsockname()[1]
            threading.Thread(target=self._accept_loop, daemon=True, name="p2p-accept").start()
        for _ in range(WORKERS):
            threading.Thread(target=self._worker, daemon=True, name="p2p-worker").start()
        for target in (self._connect_loop, self._housekeeping, self._trickle_loop, self._connector):
            threading.Thread(target=target, daemon=True, name=target.__name__).start()

    def stop(self) -> None:
        self.running = False
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
        for p in self.all_peers():
            p.close("shutting down")
        self.verify_pool.shutdown(wait=False, cancel_futures=True)
        if self.checker is not None:
            self.checker.close()
        try:
            self.addrman.save()
            self._save_bans()
        except OSError:
            pass

    @property
    def address(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return addr_key(host, self.port)

    @property
    def full_address(self) -> str:
        return f"{self.node_id}@{self.address}"

    def all_peers(self) -> list[Peer]:
        with self.lock:
            return list(self.peers.values())

    def status(self) -> dict:
        peers = self.all_peers()
        return {"node_id": self.node_id, "listen": self.address if self.listen else None,
                "inbound": sum(p.inbound for p in peers), "outbound": sum(not p.inbound for p in peers),
                "known_addresses": len(self.addrman), "headers_ahead": len(self.hindex),
                "best_header_height": self._height_of(self.best), "blocks_in_flight": len(self.requested),
                "fast_sync": self.snap_state, "stempool": len(self.stempool), "offers": len(self.offers),
                "public": addr_key(self.my_host, self.public_port or self.port) if self.my_host else None,
                "stats": dict(self.stats)}

    # -- bans --

    def _ban_key(self, host: str, node_id: str = "") -> str:
        # Local addresses share one IP (test nets, LANs): ban the node, not everyone there.
        return f"id:{node_id}" if is_local(host) else f"ip:{host}"

    def _load_bans(self) -> dict:
        path = os.path.join(self.data_dir, "banlist.json") if self.data_dir else None
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    return {k: v for k, v in json.load(f).items() if v > time.time()}
            except (OSError, ValueError, AttributeError):
                pass
        return {}

    def _save_bans(self) -> None:
        if self.data_dir:
            with open(os.path.join(self.data_dir, "banlist.json"), "w") as f:
                json.dump({k: v for k, v in self.bans.items() if v > time.time()}, f)

    def is_banned(self, host: str, node_id: str = "") -> bool:
        now = time.time()
        return self.bans.get(f"ip:{host}", 0) > now or (node_id and self.bans.get(f"id:{node_id}", 0) > now)

    def punish(self, peer: Peer, score: int, reason: str) -> None:
        peer.score += score
        if peer.score >= BAN_SCORE:
            self.bans[self._ban_key(peer.host, peer.node_id)] = time.time() + BAN_SECONDS
            self.stats["banned"] += 1
            self.log(f"[security] peer {peer.name} ({peer.node_id[:12]}) banned for {BAN_SECONDS // 3600} h: {reason}")
            peer.close(f"banned: {reason}")

    # -- connections --

    def add_manual(self, text) -> bool:
        parsed = parse_address(text)
        if parsed is None:
            return False
        host, port, node_id = parsed
        key = addr_key(host, port)
        if key == self.address:
            return False
        with self.lock:
            self.manual[key] = {"host": host, "port": port, "id": node_id, "next": 0.0, "backoff": 1.0}
        return True

    def _accept_loop(self) -> None:
        while self.running:
            try:
                sock, addr = self.server.accept()
            except OSError:
                if not self.running:
                    return
                continue
            host = addr[0]
            if self.is_banned(host) or not self._may_handshake(host):
                sock.close()
                continue
            threading.Thread(target=self._inbound, args=(sock, host, addr[1]), daemon=True).start()

    def _may_handshake(self, host: str) -> bool:
        """Cheap checks before any work: pending-handshake caps and the per-group
        limit. Nobody is evicted here: an unauthenticated connection must never
        push out a peer (that happens in _register, after the handshake)."""
        inbound = [p for p in self.all_peers() if p.inbound]
        local = is_local(host)
        if not local and sum(netgroup(p.host) == netgroup(host) for p in inbound) >= MAX_INBOUND_PER_GROUP:
            return False
        with self.lock:
            if sum(self.pending.values()) >= MAX_PENDING_HANDSHAKES:
                return False
            if not local and self.pending.get(host, 0) >= MAX_PENDING_PER_HOST:
                return False
            self.pending[host] = self.pending.get(host, 0) + 1
        return True

    def _inbound(self, sock, host: str, port: int) -> None:
        try:
            self._setup(sock, host, port, True)
        except Exception as e:                       # never let a connection thread die silently
            if self.running:
                self.log(f"[p2p] connection from {addr_key(host, port)} failed: {type(e).__name__}: {e}")
        finally:
            with self.lock:
                self.pending[host] -= 1
                if not self.pending[host]:
                    del self.pending[host]

    @staticmethod
    def _eviction_candidate(inbound: list[Peer]) -> Peer | None:
        """Bitcoin's rule in short: protect the peers that are hard for an attacker to
        imitate (fastest, most recent useful blocks and transactions, longest
        connected), then evict the youngest peer of the most crowded network group."""
        rest = list(inbound)
        for key, n in ((lambda p: p.ping_ms if p.ping_ms is not None else 1e9, 4),
                       (lambda p: -p.last_block, 4), (lambda p: -p.last_tx, 4), (lambda p: p.connected_at, 4)):
            for p in sorted(rest, key=key)[:n]:
                rest.remove(p)
        if not rest:
            return None
        groups: dict[str, list] = {}
        for p in rest:
            groups.setdefault(netgroup(p.host), []).append(p)
        crowd = max(groups.values(), key=len)
        return max(crowd, key=lambda p: p.connected_at)

    def _connect_loop(self) -> None:
        while self.running:
            try:
                self._fill_outbound()
            except Exception as e:                    # never let the dialer die
                self.log(f"[p2p] dialer error: {e}")
            time.sleep(0.5)

    def _fill_outbound(self) -> None:
        now = time.time()
        peers = self.all_peers()
        connected = {p.listen_addr for p in peers if p.listen_addr} | {p.name for p in peers if not p.inbound}
        out = [p for p in peers if not p.inbound]
        with self.lock:
            manual = [dict(m, key=k) for k, m in self.manual.items()]
            busy = set(self.connecting)
        for m in manual:                              # configured peers first, with backoff
            if m["key"] in connected or m["key"] in busy or m["next"] > now:
                continue
            with self.lock:
                if m["key"] in self.manual:
                    self.manual[m["key"]]["next"] = now + m["backoff"]
                    self.manual[m["key"]]["backoff"] = min(60.0, m["backoff"] * 2)
            self._dial(m["host"], m["port"], m["id"], manual=True)
        if len(out) + len(busy) >= self.max_outbound:
            return
        block_only = sum(p.block_only for p in out) < BLOCK_RELAY_ONLY and len(out) >= self.max_outbound - BLOCK_RELAY_ONLY
        groups = {netgroup(p.host) for p in out if not is_local(p.host)}
        pick = self.addrman.select(connected | busy | {self.address}, groups)
        if pick is None:
            if not out and not manual:
                for text in self.seeds:
                    parsed = parse_address(text)
                    if parsed:
                        self.addrman.add(parsed[0], parsed[1], parsed[2], source=None)
            return
        if self.is_banned(pick["host"], pick.get("id") or ""):
            self.addrman.attempt(addr_key(pick["host"], pick["port"]))
            return
        self._dial(pick["host"], pick["port"], None, block_only=block_only)

    def _dial(self, host: str, port: int, node_id: str | None, manual: bool = False, block_only: bool = False) -> None:
        key = addr_key(host, port)
        with self.lock:
            if key in self.connecting:
                return
            self.connecting.add(key)
        self.addrman.attempt(key)

        def run():
            try:
                sock = socket.create_connection((host, port), timeout=5)
            except OSError:
                with self.lock:
                    self.connecting.discard(key)
                return
            try:
                self._setup(sock, host, port, False, expect_id=node_id, block_only=block_only, manual_key=key if manual else None)
            except Exception as e:                   # never let a connection thread die silently
                if self.running:
                    self.log(f"[p2p] connecting to {key} failed: {type(e).__name__}: {e}")
            finally:
                with self.lock:
                    self.connecting.discard(key)
        threading.Thread(target=run, daemon=True).start()

    def _setup(self, sock, host: str, port: int, inbound: bool, expect_id: str | None = None,
               block_only: bool = False, manual_key: str | None = None) -> None:
        """Key exchange, then authenticated version messages; on success the peer is live."""
        peer = Peer(self, sock, host, port, inbound, block_only)
        try:
            sock.settimeout(HANDSHAKE_SECONDS)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            peer.send_cipher, peer.recv_cipher, peer.session_id = key_exchange(sock, not inbound, self.chain.genesis_hash)
            sig = self.key.sign(auth_message(peer.session_id, not inbound)).hex()
            with self.node.lock:
                if not self.running:              # the node is stopping: its database may be closed
                    raise ConnectionError("shutting down")
                height, tip = self.chain.height, self.chain.tip
                work = self.chain.meta[tip]["chainwork"]
                first = self.chain.first_block
            mine = {"proto": PROTOCOL_VERSION, "agent": AGENT, "id": self.node_id, "sig": sig,
                    "port": (self.public_port or self.port) if self.listen else 0, "relay": not block_only,
                    "height": height, "tip": tip, "work": str(work), "you": host, "time": int(time.time()),
                    "first": first, "features": FEATURES}
            sock.sendall(peer.send_cipher.seal(encode("version", mine)))
            deadline = time.time() + HANDSHAKE_SECONDS
            length = peer.recv_cipher.open_length(_recv_exact(sock, 20, deadline))
            if length > 64 * 1024:
                raise ConnectionError("oversized version")
            body = peer.recv_cipher.open_body(_recv_exact(sock, length + 16, deadline))
            if body[0] != CODE["version"]:
                raise ConnectionError("first message is not version")
            v = json.loads(body[1:])
            self._check_version(peer, v, inbound, expect_id)
        except (OSError, ConnectionError, TimeoutError, InvalidTag, InvalidSignature, ValueError,
                KeyError, TypeError, struct.error) as e:
            try:
                sock.close()
            except OSError:
                pass
            if isinstance(e, ConnectionError) and "ourselves" in str(e):
                self.addrman.forget(addr_key(host, port))
            self._debug(f"handshake with {addr_key(host, port)} failed: {type(e).__name__}: {e}")
            return
        sock.settimeout(None)
        if not self._register(peer):
            sock.close()
            return
        if manual_key is not None:
            with self.lock:
                if manual_key in self.manual:
                    self.manual[manual_key]["backoff"] = 1.0
        threading.Thread(target=peer._reader, daemon=True, name="p2p-read").start()
        threading.Thread(target=peer._writer, daemon=True, name="p2p-write").start()
        self._on_connect(peer)

    def _check_version(self, peer: Peer, v: dict, inbound: bool, expect_id: str | None) -> None:
        if not isinstance(v, dict) or v.get("proto") != PROTOCOL_VERSION:
            raise ConnectionError("unsupported protocol version")
        node_id = v["id"]
        if not is_hex(node_id, 64):
            raise ConnectionError("bad node id")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(node_id)).verify(
            bytes.fromhex(v["sig"]), auth_message(peer.session_id, inbound))
        if expect_id and node_id != expect_id:
            raise ConnectionError("peer presented a different node key than expected")
        if node_id == self.node_id:
            raise ConnectionError("connected to ourselves")
        if self.is_banned(peer.host, node_id):
            raise ConnectionError("banned")
        port = v.get("port", 0)
        peer.node_id = node_id
        peer.listen_port = port if isinstance(port, int) and 0 <= port < 65536 else 0
        peer.agent = str(v.get("agent", ""))[:64]
        peer.relay = bool(v.get("relay", True))
        features = v.get("features")
        if isinstance(features, list) and len(features) <= 32:
            peer.features = {f for f in features if isinstance(f, str) and f in FEATURES}
        peer.best_height = v["height"] if isinstance(v.get("height"), int) else 0
        peer.first_block = v["first"] if isinstance(v.get("first"), int) and v["first"] >= 0 else 0
        try:
            peer.best_work = int(v.get("work", "0"))
        except (TypeError, ValueError):
            peer.best_work = 0
        you = v.get("you")
        if not inbound and isinstance(you, str) and parse_address(addr_key(you, 1)):
            self.my_host = you

    def _register(self, peer: Peer) -> bool:
        victim = None
        with self.lock:
            if any(p.node_id == peer.node_id for p in self.peers.values()):
                return False                         # already connected to this node
            inbound = [p for p in self.peers.values() if p.inbound]
            if peer.inbound and len(inbound) >= MAX_INBOUND:
                victim = self._eviction_candidate(inbound)
                if victim is None:
                    return False                     # every inbound peer is protected
            self.peers[id(peer)] = peer
        if victim is not None:
            victim.close("evicted for a new inbound peer")
        return True

    def _on_connect(self, peer: Peer) -> None:
        self.log(f"Peer connected: {peer.name} {peer.node_id[:12]} ({'inbound' if peer.inbound else 'outbound'}"
                 f"{', blocks only' if peer.block_only else ''})")
        if not peer.inbound:
            self.addrman.good(peer.host, peer.port, peer.node_id)
            if not peer.block_only:
                peer.send("getaddr")
                self._advertise_self(peer)
        elif peer.listen_port:
            self.addrman.add(peer.host, peer.listen_port, peer.node_id, source=peer.host)
        if not peer.block_only and peer.relay:
            peer.send("getmempool")
        peer.next_trickle = time.time() + random.expovariate(1 / TRICKLE_MEAN_SECONDS)
        if "offers" in peer.features:
            offers = self.offer_list()
            for i in range(0, len(offers), OFFERS_PER_MESSAGE):
                peer.send("offers", {"offers": offers[i:i + OFFERS_PER_MESSAGE]})
        if peer.best_work > self._work_of(self.best):
            self._ask_headers(peer)

    def _advertise_self(self, peer: Peer) -> None:
        if self.listen and self.my_host:
            peer.send("addr", {"addrs": [{"host": self.my_host, "port": self.public_port or self.port, "id": self.node_id,
                                          "seen": int(time.time())}]})

    def _closed(self, peer: Peer, reason: str) -> None:
        with self.lock:
            was = self.peers.pop(id(peer), None)
        if was is None:
            return
        self._debug(f"peer {peer.name} disconnected: {reason}")
        with self.node.lock:
            for bh in list(peer.inflight_blocks):
                if self.requested.get(bh, (None,))[0] is peer:
                    del self.requested[bh]
            for bh in [h for h, p in self.partial.items() if p["peer"] is peer]:
                del self.partial[bh]
            for tid, w in list(self.tx_wait.items()):
                if peer in w["peers"]:
                    w["peers"].remove(peer)
                if w["asked"] is peer:
                    w["asked"] = None
                    w["deadline"] = 0.0

    def _debug(self, msg: str) -> None:
        debug = getattr(self.node, "debug", None)
        if debug is not None:
            debug(f"[p2p] {msg}")
        elif os.environ.get("GREEN_P2P_DEBUG"):
            self.log(f"[p2p] {msg}")

    # -- message intake --

    def _receive(self, peer: Peer, cmd: str, payload: bytes) -> None:
        """Runs on the peer's reader thread: answers to our own requests are handled
        here, everything else goes to the worker threads in arrival order."""
        if cmd == "blob":
            return self._on_blob(peer, payload)
        if cmd in ("ping", "pong", "noblob"):
            return self._dispatch(peer, cmd, json.loads(payload))
        with peer.lock:
            if len(peer.in_q) >= MAX_PEER_QUEUE:
                raise Misbehaving(BAN_SCORE, "flooding faster than it can be processed")
            peer.in_q.append((cmd, payload))
            if peer.scheduled:
                return
            peer.scheduled = True
        self._ready.put(peer)

    def _worker(self) -> None:
        while self.running:
            try:
                peer = self._ready.get(timeout=1)
            except queue.Empty:
                continue
            with peer.lock:
                item = peer.in_q.popleft() if peer.in_q else None
            if item is not None and peer.alive:
                cmd, payload = item
                try:
                    self._dispatch(peer, cmd, json.loads(payload))
                except Misbehaving as e:
                    self.punish(peer, e.score, str(e))
                except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as e:
                    self.punish(peer, 20, f"malformed {cmd}: {type(e).__name__}")
                except Exception as e:
                    self.log(f"[p2p] error handling {cmd} from {peer.name}: {type(e).__name__}: {e}")
            with peer.lock:
                again = bool(peer.in_q) and peer.alive
                peer.scheduled = again
            if again:
                self._ready.put(peer)

    def _dispatch(self, peer: Peer, cmd: str, m) -> None:
        if not isinstance(m, dict):
            raise Misbehaving(20, f"{cmd} payload is not an object")
        handler = getattr(self, f"_on_{cmd}", None)
        if handler is None or cmd == "version":
            raise Misbehaving(10, f"unexpected {cmd}")
        handler(peer, m)

    # -- small messages --

    def _on_ping(self, peer: Peer, m: dict) -> None:
        peer.send("pong", {"n": m.get("n")})

    def _on_pong(self, peer: Peer, m: dict) -> None:
        if peer.ping_nonce is not None and m.get("n") == peer.ping_nonce:
            peer.ping_ms = round((time.time() - peer.ping_sent) * 1000, 1)
            peer.ping_nonce = None

    def _on_getaddr(self, peer: Peer, m: dict) -> None:
        if peer.sent_addr or peer.block_only:
            return                                   # once per connection: no address scraping
        peer.sent_addr = True
        addrs = self.addrman.sample(MAX_ADDR)
        if self.listen and self.my_host:
            addrs.append({"host": self.my_host, "port": self.public_port or self.port, "id": self.node_id,
                          "seen": int(time.time())})
        peer.send("addr", {"addrs": addrs})

    def _on_addr(self, peer: Peer, m: dict) -> None:
        addrs = m["addrs"]
        if not isinstance(addrs, list) or len(addrs) > MAX_ADDR:
            raise Misbehaving(20, "oversized addr")
        if peer.block_only:
            return
        now = time.time()
        peer.addr_tokens = min(float(MAX_ADDR), peer.addr_tokens + (now - peer.addr_time) * ADDR_RATE)
        peer.addr_time = now
        fresh = []
        for a in addrs:
            if peer.addr_tokens < 1:
                break                                # rate limited: the rest is ignored
            peer.addr_tokens -= 1
            if not isinstance(a, dict):
                continue
            parsed = parse_address(addr_key(str(a.get("host", "")), int(a.get("port", 0) or 0)))
            node_id = a.get("id") if is_hex(a.get("id"), 64) else None
            if parsed is None or node_id == self.node_id or not gossip_ok(parsed[0], peer.host):
                continue
            seen = a.get("seen") if isinstance(a.get("seen"), int) else None
            if self.addrman.add(parsed[0], parsed[1], node_id, source=peer.host, seen=seen) and len(addrs) <= 10 \
                    and seen and now - seen < 600:
                fresh.append(a)
        if fresh:                                    # pass fresh self-announcements on to two peers
            for other in random.sample([p for p in self.all_peers() if p is not peer and not p.block_only],
                                       k=min(2, max(0, len(self.peers) - 1))):
                other.send("addr", {"addrs": fresh})

    # -- relay: announce, then fetch --

    def announce_tx(self, tid: str, source: Peer | None = None) -> None:
        self.stats["txs_relayed"] += 1
        for p in self.all_peers():
            if p is source or p.block_only or not p.relay or tid in p.known:
                continue
            with p.lock:
                p.trickle.append(tid)

    def announce_share(self, sid: str, share: dict, source: Peer | None = None) -> None:
        """Shares are small and few (about SHARE_FACTOR per block): sent whole to peers
        that speak them and have not seen them."""
        for p in self.all_peers():
            if p is not source and "shares" in p.features and sid not in p.known:
                p.known.add(sid)
                p.send("share", {"share": share})

    def _on_share(self, peer: Peer, m: dict) -> None:
        share = m["share"]
        try:
            sid = share_id(share)
        except (TypeError, ValueError):
            raise Misbehaving(20, "unencodable share") from None
        peer.known.add(sid)
        try:
            self.node.submit_share(share, source=peer)
        except InvalidBlock as e:                    # its work unit does not check out: a lie
            raise Misbehaving(BAN_SCORE, f"sent a fake share ({e})") from None

    def announce_block(self, bh: str, source: Peer | None = None) -> None:
        """Peers that speak compact blocks get the block at once as its header and
        short transaction ids; the others get an inv and fetch it."""
        self.stats["blocks_relayed"] += 1
        block = None
        for p in self.all_peers():
            if p is source or bh in p.known:
                continue
            p.known.add(bh)
            if "cmpct" in p.features:
                block = block or self.chain.block(bh)
                if block is not None:
                    p.send("cmpct", self._compact(block, p))
                    self.stats["cmpct_sent"] += 1
                    continue
            p.send("inv", {"items": [["block", bh]]})

    def _compact(self, block: dict, peer: Peer) -> dict:
        """Transactions the peer has seen announced (either way) travel as short ids;
        the rest, which it probably lacks, travel whole."""
        salt = os.urandom(8).hex()
        key = compact_key(block_hash(block), salt)
        ids, fill = [], []
        for i, t in enumerate(block["txs"]):
            tid = txmod.txid(t)
            if tid in peer.known:
                ids.append(short_id(key, tid))
            else:
                fill.append([i, t])
        out = {"header": block["header"], "salt": salt, "ids": ids, "fill": fill}
        if block.get("shares"):
            out["shares"] = block["shares"]          # few and small: sent whole
        return out

    def _on_cmpct(self, peer: Peer, m: dict) -> None:
        hdr, salt, ids, fill = m["header"], m["salt"], m["ids"], m["fill"]
        shares = m.get("shares", [])
        if not isinstance(shares, list) or len(shares) > params.MAX_BLOCK_SHARES:
            raise Misbehaving(20, "malformed compact block")
        if not (isinstance(hdr, dict) and is_hex(salt, 16) and isinstance(ids, list) and isinstance(fill, list)
                and len(ids) + len(fill) <= params.MAX_BLOCK_TXS
                and all(is_hex(s, 2 * SHORT_ID_BYTES) for s in ids)):
            raise Misbehaving(20, "malformed compact block")
        try:
            bh = block_hash({"header": hdr})
        except (TypeError, ValueError):
            raise Misbehaving(20, "unencodable compact block") from None
        peer.known.add(bh)
        peer.last_block = time.time()
        n = len(ids) + len(fill)
        txs: list = [None] * n
        for item in fill:
            if not (isinstance(item, list) and len(item) == 2 and type(item[0]) is int and 0 <= item[0] < n
                    and txs[item[0]] is None and isinstance(item[1], dict)):
                raise Misbehaving(20, "malformed compact block")
            txs[item[0]] = item[1]
        self.stats["cmpct_received"] += 1
        with self.node.lock:
            if self.chain.has_block(bh) or bh in self.chain.invalid or bh in self.partial or bh in self.downloaded:
                return
            orphan = hdr.get("prev") not in self.chain.blocks
            if not orphan:
                key = compact_key(bh, salt)
                pool: dict[str, tuple] = {}
                clash = set()
                for tid, t in list(self.node.mempool.items()) + [(k, e["tx"]) for k, e in self.stempool.items()]:
                    sid = short_id(key, tid)
                    if sid in pool and pool[sid][0] != tid:
                        clash.add(sid)
                    pool[sid] = (tid, t)
        if orphan:
            self.on_orphan(peer)                     # fetch the headers that lead to it
            return
        missing = []
        slots = iter(ids)
        for i in range(n):
            if txs[i] is None:
                sid = next(slots)
                if sid in pool and sid not in clash:
                    txs[i] = pool[sid][1]
                else:
                    missing.append(i)
        if missing:
            with self.node.lock:
                self.partial[bh] = {"header": hdr, "txs": txs, "shares": shares, "missing": missing, "peer": peer,
                                    "time": time.time()}
            self.stats["cmpct_txs_requested"] += len(missing)
            peer.send("getblocktxn", {"block": bh, "idx": missing})
            return
        self._complete(bh, hdr, txs, peer, shares)

    def _complete(self, bh: str, hdr: dict, txs: list, peer: Peer, shares: list = ()) -> None:
        try:
            ok = (merkle_root([txmod.txid(t) for t in txs]) == hdr.get("tx_root")
                  and merkle_root([share_id(sh) for sh in shares]) == hdr.get("shares_root"))
        except (TypeError, ValueError, KeyError):
            ok = False
        if not ok:
            # A short-id collision picked the wrong transaction, or the message carried a
            # bad one: neither proves the block wrong, so fetch the block itself.
            self.stats["cmpct_fallbacks"] += 1
            with self.node.lock:
                self.requested[bh] = (peer, time.time())
                peer.inflight_blocks[bh] = time.time()
            peer.send("getdata", {"items": [["block", bh]]})
            return
        self.stats["cmpct_rebuilt"] += 1
        self._submit({"header": hdr, "txs": txs, **({"shares": list(shares)} if shares else {})}, peer)

    def _on_getblocktxn(self, peer: Peer, m: dict) -> None:
        bh, idx = m["block"], m["idx"]
        block = self.chain.block(bh) if is_hex(bh, 64) else None
        if block is None:
            peer.send("notfound", {"items": [["block", bh]]})
            return
        count = len(block["txs"])
        if not isinstance(idx, list) or len(idx) > count or not all(type(i) is int and 0 <= i < count for i in idx):
            raise Misbehaving(20, "malformed getblocktxn")
        peer.send("blocktxn", {"block": bh, "txs": [block["txs"][i] for i in idx]})

    def _on_blocktxn(self, peer: Peer, m: dict) -> None:
        bh, got = m["block"], m["txs"]
        with self.node.lock:
            p = self.partial.get(bh)
            if p is None or p["peer"] is not peer:
                return                               # late (we gave up on it) or never asked
            del self.partial[bh]
        if not isinstance(got, list) or len(got) != len(p["missing"]) or not all(isinstance(t, dict) for t in got):
            raise Misbehaving(20, "malformed blocktxn")
        txs = p["txs"]
        for i, t in zip(p["missing"], got):
            txs[i] = t
        self._complete(bh, p["header"], txs, peer, p["shares"])

    # -- compute offers: providers advertise, every node lists them --

    def add_offer(self, offer: dict, source: Peer | None = None) -> bool:
        """Keep and relay a provider's signed offer if it is valid, newer than the one we
        hold, and its provider holds OFFER_MIN_BALANCE (which makes spam cost coins)."""
        if offer_problem(self.chain.chain_id, offer, time.time()) is not None:
            return False
        with self.node.lock:
            funded = self.chain.state.balance(offer["provider"]) >= OFFER_MIN_BALANCE
        with self.lock:
            held = self.offers.get(offer["provider"])
            if held is not None and held["time"] >= offer["time"]:
                return False
            if not funded:
                # It may have arrived before the block that funds its provider: look again later.
                self.offers_waiting.pop(offer["provider"], None)
                self.offers_waiting[offer["provider"]] = offer
                while len(self.offers_waiting) > 256:
                    self.offers_waiting.popitem(last=False)
                return False
            self.offers_waiting.pop(offer["provider"], None)
            self.offers.pop(offer["provider"], None)
            self.offers[offer["provider"]] = offer
            while len(self.offers) > MAX_OFFERS:
                self.offers.popitem(last=False)          # the least recently renewed
        for p in self.all_peers():
            if p is not source and "offers" in p.features and not p.block_only:
                p.send("offers", {"offers": [offer]})
        return True

    def offer_list(self) -> list[dict]:
        cutoff = time.time() - OFFER_TTL
        with self.lock:
            for addr in [a for a, o in self.offers.items() if o["time"] < cutoff]:
                del self.offers[addr]
            return list(self.offers.values())

    def _on_offers(self, peer: Peer, m: dict) -> None:
        offers = m["offers"]
        if not isinstance(offers, list) or len(offers) > OFFERS_PER_MESSAGE:
            raise Misbehaving(20, "oversized offers")
        if peer.block_only:
            return
        now = time.time()
        peer.offer_tokens = min(float(MAX_OFFERS), peer.offer_tokens + (now - peer.offer_time) * OFFER_RATE)
        peer.offer_time = now
        for offer in offers:
            if peer.offer_tokens < 1:
                return                                   # rate limited: the rest is ignored
            peer.offer_tokens -= 1
            problem = offer_problem(self.chain.chain_id, offer, now)
            if problem in ("bad signature", "the key is not the provider's", "bad fields"):
                raise Misbehaving(20, f"invalid offer ({problem})")
            if problem is None:
                self.add_offer(offer, source=peer)

    # -- Dandelion++: our transactions do not reveal where they came from --

    def _stem_target(self, source: Peer | None) -> Peer | None:
        """The outbound peer a stem transaction goes to next, or None to announce it here.

        Each epoch the node draws DANDELION_DESTINATIONS outbound peers and, with
        probability DANDELION_FLUFF_PROB, becomes a diffuser for the whole epoch.
        Each incoming link (and the node's own transactions) maps to one
        destination for the epoch, so the stem graph does not change from one
        transaction to the next: Dandelion++'s defence against linking a user's
        transactions by intersecting their paths."""
        now = time.time()
        with self.lock:
            if now >= self.stem_epoch:
                self.stem_epoch = now + DANDELION_EPOCH_SECONDS
                self.stem_fluff = random.random() < DANDELION_FLUFF_PROB
                self.stem_dest, self.stem_route = [], {}
            if source is not None and self.stem_fluff:
                return None
            alive = [p for p in self.stem_dest if p.alive]
            if len(alive) < DANDELION_DESTINATIONS:
                eligible = [p for p in self.peers.values() if not p.inbound and not p.block_only and p.relay
                            and p.alive and "dtx" in p.features and p not in alive]
                alive += random.sample(eligible, min(DANDELION_DESTINATIONS - len(alive), len(eligible)))
                self.stem_route = {k: v for k, v in self.stem_route.items() if v in alive}
            self.stem_dest = alive
            key = source.node_id if source is not None else ""
            dest = self.stem_route.get(key)
            if dest is None:
                choices = [p for p in alive if p is not source]
                if not choices:
                    return None
                dest = self.stem_route[key] = random.choice(choices)
            return None if dest is source else dest

    def stem_tx(self, t: dict, tid: str, source: Peer | None) -> bool:
        """Pass a checked transaction one hop along the stem; False: announce it now."""
        dest = self._stem_target(source)
        if dest is None:
            return False
        with self.node.lock:
            if tid not in self.stempool:
                if len(self.stempool) >= MAX_STEMPOOL:
                    return False
                embargo = time.time() + EMBARGO_SECONDS + random.expovariate(1 / EMBARGO_MEAN_EXTRA)
                self.stempool[tid] = {"tx": t, "embargo": embargo, "local": source is None}
        self.stats["stem_relayed"] += 1
        dest.send("dtx", {"tx": t})
        return True

    def _on_dtx(self, peer: Peer, m: dict) -> None:
        t = m["tx"]
        try:
            tid = txmod.txid(t)
        except (TypeError, ValueError, KeyError):
            raise Misbehaving(20, "unencodable transaction") from None
        if peer.block_only:
            return
        self._reset_rejected()
        with self.node.lock:
            if tid in self.node.mempool or tid in self.rejected:
                return
            looped = tid in self.stempool
        if looped:                                   # the stem came back around: it ends here
            self._fluff(tid)
            return
        try:
            self.node.submit_tx(t, source=peer, stem=True)
        except txmod.FutureNonce:
            pass                                     # its predecessor has not reached us yet
        except (InvalidTx, MissingData):
            self.rejected.add(tid)                   # policy can differ between nodes: no penalty

    def _fluff(self, tid: str, embargo: bool = False) -> None:
        """End a stem here: announce the transaction like any other."""
        with self.node.lock:
            e = self.stempool.pop(tid, None)
            if e is None or tid in self.node.mempool:
                return
        self.stats["embargo_fired" if embargo else "stem_fluffed"] += 1
        try:
            self.node.submit_tx(e["tx"], stem=False)
        except (InvalidTx, MissingData):
            pass                                     # mined meanwhile, or no longer valid

    def _trickle_loop(self) -> None:
        while self.running:
            now = time.time()
            for p in self.all_peers():
                if p.next_trickle > now:
                    continue
                p.next_trickle = now + random.expovariate(1 / TRICKLE_MEAN_SECONDS)
                with p.lock:
                    batch, p.trickle = p.trickle[:MAX_INV], p.trickle[MAX_INV:]
                batch = [t for t in batch if t not in p.known]
                for t in batch:
                    p.known.add(t)
                if batch:
                    p.send("inv", {"items": [["tx", t] for t in batch]})
            time.sleep(0.05)

    def _on_inv(self, peer: Peer, m: dict) -> None:
        items = m["items"]
        if not isinstance(items, list) or len(items) > MAX_INV:
            raise Misbehaving(20, "oversized inv")
        want_tx, want_blocks = [], []
        self._reset_rejected()
        with self.node.lock:
            for kind, h in items:
                if not is_hex(h, 64):
                    raise Misbehaving(20, "bad hash in inv")
                peer.known.add(h)
                if kind == "tx":
                    if peer.block_only or h in self.node.mempool or h in self.rejected:
                        continue
                    w = self.tx_wait.get(h)
                    if w is None:
                        if len(self.tx_wait) >= 50_000:
                            continue
                        w = self.tx_wait[h] = {"peers": [], "asked": None, "deadline": 0.0}
                    if peer not in w["peers"]:
                        w["peers"].append(peer)
                    if w["asked"] is None:
                        w["asked"], w["deadline"] = peer, time.time() + TX_REQUEST_SECONDS
                        want_tx.append(h)
                elif kind == "block":
                    if self.chain.has_block(h) or h in self.chain.invalid or h in self.requested \
                            or h in self.downloaded:
                        continue
                    self.requested[h] = (peer, time.time())
                    peer.inflight_blocks[h] = time.time()
                    want_blocks.append(h)
        if want_tx or want_blocks:
            peer.send("getdata", {"items": [["block", h] for h in want_blocks] + [["tx", h] for h in want_tx]})

    def _on_getdata(self, peer: Peer, m: dict) -> None:
        items = m["items"]
        if not isinstance(items, list) or len(items) > MAX_INV:
            raise Misbehaving(20, "oversized getdata")
        missing = []
        for kind, h in items:
            if kind == "tx":
                with self.node.lock:
                    t = self.node.mempool.get(h)
                if t is None:
                    missing.append([kind, h])
                else:
                    peer.send("tx", {"tx": t})
            elif kind == "block":
                block = self.chain.block(h) if is_hex(h, 64) else None
                if block is None:
                    missing.append([kind, h])
                else:
                    peer.send("block", {"block": block})
        if missing:
            peer.send("notfound", {"items": missing})

    def _on_notfound(self, peer: Peer, m: dict) -> None:
        with self.node.lock:
            for kind, h in m["items"][:MAX_INV]:
                if kind == "tx" and h in self.tx_wait and self.tx_wait[h]["asked"] is peer:
                    w = self.tx_wait[h]
                    w["peers"] = [p for p in w["peers"] if p is not peer]
                    w["asked"], w["deadline"] = None, 0.0
                elif kind == "block" and self.partial.get(h, {}).get("peer") is peer:
                    del self.partial[h]
                elif kind == "block" and self.requested.get(h, (None,))[0] is peer:
                    del self.requested[h]
                    peer.inflight_blocks.pop(h, None)
                    e = self.hindex.get(h)
                    if e is not None:                # it pruned since we connected: ask others for these
                        peer.first_block = max(peer.first_block, e["header"]["height"] + 1)

    def _reset_rejected(self) -> None:
        tip = self.chain.tip
        if tip != self._rejected_tip:                # a new block can make them valid
            self._rejected_tip = tip
            self.rejected.clear()

    def _on_tx(self, peer: Peer, m: dict) -> None:
        t = m["tx"]
        try:
            tid = txmod.txid(t)
        except (TypeError, ValueError, KeyError):
            raise Misbehaving(20, "unencodable transaction") from None
        peer.known.add(tid)
        with self.node.lock:
            self.tx_wait.pop(tid, None)
        peer.last_tx = time.time()
        try:
            self.node.submit_tx(t, source=peer)
        except txmod.FutureNonce:
            pass                                     # not "rejected": once its predecessor arrives, the
                                                     # next announcement of it is fetched again
        except (InvalidTx, MissingData):
            self.rejected.add(tid)                   # policy can differ between nodes: no penalty

    def _on_getmempool(self, peer: Peer, m: dict) -> None:
        if peer.sent_mempool or peer.block_only or not peer.relay:
            return
        peer.sent_mempool = True
        with self.node.lock:
            ids = list(self.node.mempool)[:5 * MAX_INV]
        with peer.lock:
            peer.trickle.extend(ids)

    # -- data blobs (job matrices, programs, results) --

    def fetch_blob(self, blob_hash: str, max_size: int, prefer: Peer | None = None) -> bytes | None:
        """Ask up to four peers for a blob; returns its bytes once one sends data that
        hashes right. Large frames are only accepted while such a request is open."""
        peers = [p for p in self.all_peers() if p.alive]
        random.shuffle(peers)
        if prefer in peers:
            peers.remove(prefer)
            peers.insert(0, prefer)
        for peer in peers[:4]:
            with peer.lock:
                if not peer.alive:
                    continue
                # Someone here already asked this peer for it (the watchtower and an API
                # client fetching one result at once): wait for that answer, do not skip
                # the peer -- with a single peer, skipping meant failing.
                wait = peer.blob_waits.get(blob_hash)
                asked = wait is None
                if asked:
                    wait = peer.blob_waits[blob_hash] = {"max": max_size, "event": threading.Event(), "data": None}
            if asked:
                peer.send("getblob", {"hash": blob_hash, "max": max_size})
            wait["event"].wait(10 + wait["max"] / (2 * 1024 * 1024))
            if asked:
                with peer.lock:
                    peer.blob_waits.pop(blob_hash, None)
            data = wait["data"]
            if data is not None and hhex(data) == blob_hash:
                return data
        return None

    def _on_getblob(self, peer: Peer, m: dict) -> None:
        h, limit = m["hash"], m["max"]
        data = self.node.blobs.get(h) if is_hex(h, 64) and self.node.is_public_blob(h) else None
        now = time.time()
        peer.blob_budget = min(float(BLOB_UPLOAD_PER_MINUTE),
                               peer.blob_budget + (now - peer.blob_time) * BLOB_UPLOAD_PER_MINUTE / 60)
        peer.blob_time = now
        if data is None or not isinstance(limit, int) or len(data) > limit or len(data) > peer.blob_budget:
            peer.send("noblob", {"hash": h})
            return
        peer.blob_budget -= len(data)
        peer.send("blob", data)

    def _on_blob(self, peer: Peer, data: bytes) -> None:
        h = hhex(data)
        with peer.lock:
            wait = peer.blob_waits.get(h)
        if wait is None:
            raise Misbehaving(BAN_SCORE, "sent data nobody asked for")
        wait["data"] = data
        wait["event"].set()

    def _on_noblob(self, peer: Peer, m: dict) -> None:
        with peer.lock:
            wait = peer.blob_waits.get(m.get("hash"))
        if wait is not None:
            wait["event"].set()

    # -- headers-first sync --

    def _lookup(self, bh: str) -> dict:
        e = self.hindex.get(bh)
        return e["header"] if e is not None else self.chain.header(bh)

    def _work_of(self, bh: str) -> int:
        e = self.hindex.get(bh)
        if e is not None:
            return e["work"]
        w = self.chain.chainwork(bh)
        return w if w is not None else 0

    def _height_of(self, bh: str) -> int:
        e = self.hindex.get(bh)
        return e["header"]["height"] if e is not None else (self.chain.header_or_none(bh) or {"height": 0})["height"]

    def _knows_header(self, bh: str) -> bool:
        return bh in self.hindex or self.chain.has_block(bh)

    def locator(self) -> list[str]:
        """Hashes from our best header back to genesis, dense near the tip, then
        doubling steps, so a peer finds where our chains part in one message."""
        out, cur = [], self.best
        while cur in self.hindex and len(out) < 10:
            out.append(cur)
            cur = self.hindex[cur]["header"]["prev"]
        height, step = self.chain.height, 1
        while height > 0 and len(out) < MAX_LOCATOR - 1:
            out.append(self.chain.main[height])
            if len(out) >= 10:
                step *= 2
            height -= step
        out.append(self.chain.genesis_hash)
        return out

    def _ask_headers(self, peer: Peer) -> None:
        if peer.headers_asked and time.time() - peer.headers_asked < HEADERS_REQUEST_SECONDS:
            return
        with self.node.lock:
            loc = self.locator()
        peer.headers_asked = time.time()
        peer.send("getheaders", {"locator": loc})

    def on_orphan(self, peer: Peer | None) -> None:
        if isinstance(peer, Peer):
            peer.headers_asked = 0.0
            self._ask_headers(peer)

    def _on_getheaders(self, peer: Peer, m: dict) -> None:
        loc = m["locator"]
        if not isinstance(loc, list) or len(loc) > MAX_LOCATOR:
            raise Misbehaving(20, "bad locator")
        start = 0
        with self.node.lock:
            for h in loc:
                n = self.chain.main_height(h) if is_hex(h, 64) else None
                if n is not None:
                    start = n
                    break
            hashes = self.chain.main[start + 1:start + 1 + MAX_HEADERS]
        headers = [self.chain.header(h) for h in hashes]
        peer.send("headers", {"headers": headers})

    def _on_headers(self, peer: Peer, m: dict) -> None:
        headers = m["headers"]
        if not isinstance(headers, list) or len(headers) > MAX_HEADERS:
            raise Misbehaving(20, "oversized headers")
        solicited = bool(peer.headers_asked)
        peer.headers_asked = 0.0
        if not headers:
            # Nothing past our best header: whatever it claimed, it has no more work.
            with self.node.lock:
                peer.best_work = min(peer.best_work, self._work_of(self.best))
            return
        hashes = []
        for i, hdr in enumerate(headers):
            try:
                hashes.append(block_hash({"header": hdr}))
            except (TypeError, ValueError):
                raise Misbehaving(BAN_SCORE, "unencodable header") from None
            if i and hdr.get("prev") != hashes[i - 1]:
                raise Misbehaving(BAN_SCORE, "headers do not form a chain")
        self.stats["headers_received"] += len(headers)
        new = []
        with self.node.lock:
            if not self._knows_header(headers[0].get("prev")):
                peer.unconnecting += 1
                if peer.unconnecting > UNCONNECTING_LIMIT:
                    raise Misbehaving(20, "headers that never connect")
                peer.headers_asked = 0.0
                self._ask_headers(peer)
                return
            peer.unconnecting = 0
            fork = self._fork_height(headers[0]["prev"])
            if fork < self.chain.height - params.MAX_REORG_DEPTH:
                return                               # forks off too deep to ever be accepted
            for hdr, bh in zip(headers, hashes):
                if bh in self.chain.invalid:
                    raise Misbehaving(BAN_SCORE, "header of a block known to be invalid")
                if self._knows_header(bh):
                    continue
                if len(self.hindex) >= HEADER_INDEX_MAX:
                    break
                try:
                    self.chain.validate_header(hdr, lookup=self._lookup)
                except InvalidBlock as e:
                    for x in new:
                        self.hindex.pop(x, None)
                    if e.permanent:
                        raise Misbehaving(BAN_SCORE, f"invalid header: {e}") from None
                    return                           # e.g. too far in the future *now*: try later
                self.hindex[bh] = {"header": hdr, "work": self._work_of(hdr["prev"]) + work_for_target(target_of(hdr))}
                new.append(bh)
        if not new:
            if len(headers) == MAX_HEADERS:
                self._ask_more(peer, hashes[-1])
            return
        # A fake header chain costs only hashes to make; real work units cost
        # matmuls. Recompute a random sample and the last one before trusting it.
        sample = set(random.sample(new, min(WORK_SAMPLES, len(new)))) | {new[-1]}
        if not all(self.verify_pool.map(lambda bh: self._sample_ok(bh, peer), sample)):
            with self.node.lock:
                for bh in new:
                    self.hindex.pop(bh, None)
            raise Misbehaving(BAN_SCORE, "header whose work unit does not check out")
        with self.node.lock:
            last = new[-1]
            if self.hindex[last]["work"] > self._work_of(self.best):
                self.best = last
            peer.best_work = max(peer.best_work, self.hindex[last]["work"])
            peer.best_height = max(peer.best_height, self.hindex[last]["header"]["height"])
        if len(headers) == MAX_HEADERS and solicited:
            self._ask_more(peer, hashes[-1])
        self._schedule()

    def _ask_more(self, peer: Peer, from_hash: str) -> None:
        if len(self.hindex) < HEADER_INDEX_MAX:
            peer.headers_asked = time.time()
            peer.send("getheaders", {"locator": [from_hash]})

    def _sample_ok(self, bh: str, peer: Peer) -> bool:
        hdr = self.hindex[bh]["header"]
        self.stats["work_samples"] += 1
        for _ in range(2):
            try:
                return self.chain.work_ok(hdr, self.chain.work_transcript(hdr))
            except MissingData as e:
                data = self.fetch_blob(e.blob_hash, MAX_JOB_BLOB, prefer=peer)
                if data is None:
                    return False                     # it must serve the data its headers use
                self.node.blobs.put(data)
            except InvalidBlock:
                return False
        return False

    def _fork_height(self, bh: str) -> int:
        """Height where the branch through `bh` meets our main chain."""
        while bh in self.hindex:
            bh = self.hindex[bh]["header"]["prev"]
        while self.chain.main_height(bh) is None:
            hdr = self.chain.header_or_none(bh)
            if hdr is None or hdr["height"] == 0:
                return 0
            bh = hdr["prev"]
        return self.chain.main_height(bh)

    # -- block download --

    def _download_path(self) -> list[str]:
        """Hashes from just above the chain to the best header, oldest first."""
        if self._path_for == self.best and self._path and self.chain.has_block(self._path[0]) is False:
            return self._path
        path, cur = [], self.best
        while cur in self.hindex:
            path.append(cur)
            cur = self.hindex[cur]["header"]["prev"]
        path.reverse()
        self._path, self._path_for = path, self.best
        return path

    def _schedule(self) -> None:
        """Hand out block requests for the next window of the best header chain."""
        if self.snap_state in ("waiting", "running"):
            self._maybe_fast_sync()
            return                                   # blocks wait until the snapshot is settled
        sends: dict[int, tuple] = {}
        now = time.time()
        with self.node.lock:
            if self._work_of(self.best) <= self.chain.meta[self.chain.tip]["chainwork"]:
                return
            path = self._download_path()
            peers = [p for p in self.all_peers() if p.alive]
            for bh in path[:BLOCK_WINDOW]:
                # (blocks of the path are above the tip, so connected ones are in memory)
                if bh in self.requested or bh in self.downloaded or bh in self.chain.blocks:
                    continue
                need, height = self.hindex[bh]["work"], self.hindex[bh]["header"]["height"]
                able = [p for p in peers if p.best_work >= need and p.first_block <= height
                        and len(p.inflight_blocks) < BLOCKS_PER_PEER]
                if not able:
                    break
                p = min(able, key=lambda x: len(x.inflight_blocks))
                p.inflight_blocks[bh] = now
                self.requested[bh] = (p, now)
                sends.setdefault(id(p), (p, []))[1].append(bh)
        for p, hashes in sends.values():
            p.send("getdata", {"items": [["block", h] for h in hashes]})

    def _on_block(self, peer: Peer, m: dict) -> None:
        block = m["block"]
        try:
            bh = block_hash(block)
        except (TypeError, ValueError, KeyError):
            raise Misbehaving(20, "unencodable block") from None
        peer.known.add(bh)
        peer.last_block = time.time()
        with self.node.lock:
            if self.requested.get(bh, (None,))[0] is peer:
                del self.requested[bh]
            peer.inflight_blocks.pop(bh, None)
            if self.chain.has_block(bh):
                return
            in_index = bh in self.hindex
            parent_known = block["header"]["prev"] in self.chain.blocks
            if in_index:
                # Its work unit is checked in the pool right away, in parallel with
                # other arrivals; the connector thread connects blocks in order.
                self.downloaded[bh] = (block, peer, self._check_work(block))
                self.stats["blocks_downloaded"] += 1
        if in_index:
            self._connect_wake.set()
            self._schedule()
            return
        if not parent_known:
            self.on_orphan(peer)                     # fetch the headers that lead to it
            return
        self._submit(block, peer)

    def _submit(self, block: dict, peer: Peer, relay: bool = True) -> str:
        """"added", "exists", "orphan", "invalid" (the header is bad) or "retry"
        (this copy or its data could not be used; fetch it again later)."""
        try:
            status = self.node.submit_block(block, source=peer, relay=relay)
        except InvalidBlock as e:
            if e.permanent:
                self.punish(peer, BAN_SCORE, f"sent an invalid block ({e})")
                return "invalid"
            self.punish(peer, 10, f"sent a block that could not be accepted ({e})")
            return "retry"
        except MissingData as e:
            self._debug(f"block from {peer.name} needs data nobody sent: {e}")
            return "retry"
        if status == "added":
            with self.node.lock:
                peer.best_work = max(peer.best_work, self._work_of(block_hash(block)))
                peer.best_height = max(peer.best_height, block["header"]["height"])   # (shown by `grn peers`)
        return status

    def _connector(self) -> None:
        """Connects downloaded blocks off the message threads: messages from one peer
        are handled one at a time, so connecting inside the handler would stop
        blocks from piling up and their work checks from running in parallel."""
        while self.running:
            self._connect_wake.wait(1.0)
            self._connect_wake.clear()
            try:
                self._connect_ready()
            except Exception as e:
                self.log(f"[p2p] connector error: {type(e).__name__}: {e}")

    def _connect_ready(self) -> None:
        """Connect downloaded blocks in chain order. Each one's work unit was sent to
        the verification threads when it arrived, so the sequential part is the ledger."""
        if not self.connect_lock.acquire(blocking=False):
            return                                   # someone else is connecting; it will see ours
        try:
            while True:
                with self.node.lock:
                    batch = []
                    for bh in self._download_path():
                        if bh in self.chain.blocks:
                            continue
                        if bh not in self.downloaded:
                            break
                        batch.append((bh,) + self.downloaded[bh])
                        if len(batch) >= 64:
                            break
                if not batch:
                    return
                for bh, block, peer, check in batch:
                    try:
                        found = check.result()           # usually done by now
                        if isinstance(found, list):      # transcripts from a worker process
                            self._apply_transcripts(block, found)
                    except Exception:
                        pass                             # add_block then checks the work itself
                    near_tip = self._height_of(self.best) - block["header"]["height"] <= 2
                    status = self._submit(block, peer, relay=near_tip)
                    with self.node.lock:
                        self.downloaded.pop(bh, None)
                        if status in ("added", "exists"):
                            self.hindex.pop(bh, None)
                            continue
                        if status == "invalid":
                            self._drop_branch(bh)
                    break                            # "retry"/"orphan": the scheduler asks again
        finally:
            self.connect_lock.release()

    def _check_work(self, block: dict):
        """A future for a downloaded block's work checks: worker processes during a long
        download (threads top out at ~2x under the interpreter lock), threads otherwise."""
        if self.checker is not None and len(self.downloaded) >= PROCESS_BACKLOG:
            try:
                return self.checker.submit([block["header"]] + list(block.get("shares", [])))
            except Exception:
                pass                                 # (closing, or a broken pool): threads instead
        return self.verify_pool.submit(self._precheck, block)

    def _apply_transcripts(self, block: dict, found: list) -> None:
        """Record what a worker process recomputed: the block's work unit, then its shares'."""
        hdr = block["header"]
        if found and found[0] is not None and self.chain.work_ok(hdr, found[0]):
            if len(self.chain.prechecked) > 10_000:
                self.chain.prechecked.clear()
            self.chain.prechecked.add(block_hash(block))
            self.stats["checked_in_processes"] += 1
        for sh, tr in zip(block.get("shares", []), found[1:]):
            if tr is not None and tr == sh["transcript"] \
                    and work.ticket_value(seed_of(sh), tr) <= self.chain.share_target(sh):
                self.chain.share_verified(share_id(sh))

    def _precheck(self, block: dict) -> None:
        hdr = block["header"]
        try:
            if self.chain.work_ok(hdr, self.chain.work_transcript(hdr)):
                if len(self.chain.prechecked) > 10_000:
                    self.chain.prechecked.clear()
                self.chain.prechecked.add(block_hash(block))
                self.stats["prechecked"] += 1
        except (MissingData, InvalidBlock, InvalidTx):
            pass                                     # add_block will fetch the data or reject it
        for sh in block.get("shares", []):           # the shares' work units too, in parallel
            try:
                self.chain.check_share_work(sh)
            except (MissingData, InvalidBlock, InvalidTx, TypeError, KeyError, ValueError):
                pass

    def _drop_branch(self, bh: str) -> None:
        """Forget a header whose block failed, and every header built on it."""
        doomed = {bh}
        changed = True
        while changed:
            changed = False
            for h, e in list(self.hindex.items()):
                if h not in doomed and e["header"]["prev"] in doomed:
                    doomed.add(h)
                    changed = True
        for h in doomed:
            self.hindex.pop(h, None)
            self.downloaded.pop(h, None)
        if self.best in doomed or self.best not in self.hindex and not self.chain.has_block(self.best):
            tip = self.chain.tip
            self.best = max(list(self.hindex) + [tip], key=self._work_of)
        self._path_for = None

    def _prune_index(self) -> None:
        """Headers of forks that fell below the reorg window can never be used;
        without this they would pile up until the index refused new headers."""
        floor = self.chain.height - params.MAX_REORG_DEPTH
        stale = [h for h, e in self.hindex.items() if e["header"]["height"] < floor]
        for h in stale:
            del self.hindex[h]
            self.downloaded.pop(h, None)
        if stale:
            self._path_for = None
            if self.best not in self.hindex and not self.chain.has_block(self.best):
                self.best = max(list(self.hindex) + [self.chain.tip], key=self._work_of)

    def on_new_tip(self) -> None:
        """Called by the node when its chain moves: keep `best` at least the tip."""
        with self.node.lock:
            if self._work_of(self.chain.tip) >= self._work_of(self.best):
                self.best = self.chain.tip
                self._path_for = None

    # -- fast sync from a snapshot --

    def _on_getsnapinfo(self, peer: Peer, m: dict) -> None:
        offers = [{"id": s["id"], "height": s["height"], "hash": s["hash"], "root": s["root"]}
                  for s in self.node.snapshots]
        peer.send("snapinfo", {"snapshots": offers})

    def _on_snapinfo(self, peer: Peer, m: dict) -> None:
        offers = m["snapshots"]
        if not isinstance(offers, list) or len(offers) > 16:
            raise Misbehaving(20, "bad snapinfo")
        peer.snap_offers = [o for o in offers if isinstance(o, dict) and is_hex(o.get("id"), 64)
                            and is_hex(o.get("hash"), 64) and isinstance(o.get("height"), int)]

    def _headers_settled(self) -> bool:
        """Some peer answered, and no peer claims more work than our best header."""
        peers = [p for p in self.all_peers() if p.alive]
        return bool(peers) and self.stats["headers_received"] > 0 and \
            not any(p.headers_asked for p in peers) and all(p.best_work <= self._work_of(self.best) for p in peers)

    def _maybe_fast_sync(self) -> None:
        if self.snap_state != "waiting" or not self._headers_settled():
            return
        if len([p for p in self.all_peers() if p.alive]) < FAST_SYNC_PEERS                 and time.time() - self.started < FAST_SYNC_GRACE:
            return                                   # a peer still connecting may be the one with a snapshot
        if self._height_of(self.best) < params.SNAPSHOT_INTERVAL + params.MAX_REORG_DEPTH:
            self.snap_state = "failed"               # too young a chain for any final snapshot
            self.log("[sync] no snapshot can exist yet; syncing from genesis")
            return
        self.snap_state = "running"
        threading.Thread(target=self._fast_sync, daemon=True, name="fast-sync").start()

    def _fast_sync(self) -> None:
        try:
            ok = self._try_snapshots()
        except Exception as e:
            self.log(f"[sync] fast sync error: {type(e).__name__}: {e}")
            ok = False
        if not ok:
            self.log("[sync] no usable snapshot; syncing from genesis")
        self.snap_state = "done" if ok else "failed"
        self._schedule()

    def _try_snapshots(self) -> bool:
        peers = [p for p in self.all_peers() if p.alive]
        for p in peers:
            p.snap_offers = None
            p.send("getsnapinfo")
        deadline = time.time() + 5
        while time.time() < deadline and any(p.snap_offers is None and p.alive for p in peers):
            time.sleep(0.05)
        with self.node.lock:
            best_height = self._height_of(self.best)
            on_best = set(self._download_path())         # headers above genesis on the best chain
            options = []
            for p in peers:
                for o in p.snap_offers or []:
                    e = self.hindex.get(o["hash"])
                    if e is None or o["hash"] not in on_best or e["header"]["height"] != o["height"]:
                        continue
                    if o["height"] > best_height - params.MAX_REORG_DEPTH or o.get("root") != e["header"]["state_root"]:
                        continue
                    options.append((o["height"], p, o))
        for height, peer, offer in sorted(options, key=lambda x: -x[0]):
            state = self._download_snapshot(peer, offer)
            if state is None:
                continue
            with self.node.lock:
                headers, cur = [], offer["hash"]
                while cur in self.hindex:
                    headers.append(self.hindex[cur]["header"])
                    cur = self.hindex[cur]["header"]["prev"]
                headers.reverse()
                self.node.adopt_snapshot(headers, state)
                for hdr in headers:
                    self.hindex.pop(block_hash({"header": hdr}), None)
                self._path_for = None
            return True
        return False

    def _download_snapshot(self, peer: Peer, offer: dict):
        """Fetch a manifest and its chunks (from that peer first, others if needed);
        returns the state, or None and punishes a peer that sent bad data."""
        expected_root = self.hindex[offer["hash"]]["header"]["state_root"]
        data = self.fetch_blob(offer["id"], snapshot.MAX_MANIFEST, prefer=peer)
        try:
            if data is None:
                return None
            manifest = snapshot.parse_manifest(data)
            if (manifest["height"], manifest["hash"], manifest["root"]) != (offer["height"], offer["hash"], expected_root):
                raise ValueError("manifest does not describe the offered snapshot")
            builder = snapshot.Builder(manifest)
            self.log(f"[sync] downloading the ledger at height {offer['height']} ({len(manifest['chunks'])} chunks)")
            for chunk in manifest["chunks"]:
                data = self.fetch_blob(chunk, snapshot.MAX_CHUNK, prefer=peer)
                if data is None:
                    return None                      # unavailable is not proof of lying
                builder.add(data)
            return builder.finish(expected_root)
        except ValueError as e:
            self.punish(peer, BAN_SCORE, f"sent a bad snapshot ({e})")
            return None

    # -- housekeeping --

    def _housekeeping(self) -> None:
        last_save = time.time()
        while self.running:
            time.sleep(1.0)
            try:
                self._tick()
                if self.checker is not None:
                    self.checker.stop_if_idle()      # worker processes only while a long sync lasts
                if time.time() - last_save > 300:
                    last_save = time.time()
                    self.addrman.save()
                    self._save_bans()
            except Exception as e:
                self.log(f"[p2p] housekeeping error: {type(e).__name__}: {e}")

    def _tick(self) -> None:
        now = time.time()
        peers = self.all_peers()
        for p in peers:
            if now - p.last_recv > IDLE_SECONDS or (p.ping_nonce is not None and now - p.ping_sent > PONG_SECONDS):
                p.close("timed out")
                continue
            if p.ping_nonce is None and now - p.ping_sent > PING_SECONDS:
                p.ping_nonce, p.ping_sent = random.getrandbits(48), now
                p.send("ping", {"n": p.ping_nonce})
            if p.headers_asked and now - p.headers_asked > HEADERS_REQUEST_SECONDS:
                p.headers_asked = 0.0
                self.punish(p, 10, "did not answer getheaders")
        retry_tx: dict[int, tuple] = {}
        with self.node.lock:
            for tid, w in list(self.tx_wait.items()):
                if w["asked"] is not None and w["deadline"] > now:
                    continue
                if w["asked"] is not None and w["asked"] in w["peers"]:
                    w["peers"].remove(w["asked"])
                live = [p for p in w["peers"] if p.alive]
                if not live or tid in self.node.mempool:
                    del self.tx_wait[tid]
                    continue
                w["peers"], w["asked"], w["deadline"] = live, live[0], now + TX_REQUEST_SECONDS
                retry_tx.setdefault(id(live[0]), (live[0], []))[1].append(tid)
            stalled = set()
            for bh, (p, t) in list(self.requested.items()):
                if now - t > BLOCK_REQUEST_SECONDS or not p.alive:
                    del self.requested[bh]
                    p.inflight_blocks.pop(bh, None)
                    if p.alive:
                        stalled.add(p)
            self._prune_index()
            behind = [p for p in peers if p.alive and p.best_work > self._work_of(self.best) and not p.headers_asked]
            # stems: announce what nobody announced before its embargo ran out
            due = [tid for tid, e in self.stempool.items() if e["embargo"] <= now or tid in self.node.mempool]
            # compact blocks whose missing transactions never came: fetch the whole block
            refetch = []
            for bh, p in list(self.partial.items()):
                if now - p["time"] > BLOCK_REQUEST_SECONDS:
                    del self.partial[bh]
                    if p["peer"].alive and not self.chain.has_block(bh):
                        self.requested[bh] = (p["peer"], now)
                        p["peer"].inflight_blocks[bh] = now
                        refetch.append((p["peer"], bh))
        for tid in due:
            self._fluff(tid, embargo=True)
        with self.lock:
            waiting = list(self.offers_waiting.values())
        for offer in waiting:                        # funded by now? (stale ones are dropped)
            if not self.add_offer(offer) and offer_problem(self.chain.chain_id, offer, now) is not None:
                with self.lock:
                    if self.offers_waiting.get(offer["provider"]) is offer:
                        del self.offers_waiting[offer["provider"]]
        for p, bh in refetch:
            p.send("getdata", {"items": [["block", bh]]})
        for p, tids in retry_tx.values():
            p.send("getdata", {"items": [["tx", t] for t in tids]})
        for p in stalled:                            # the blocks go to other peers; repeat stalling is a ban
            self.punish(p, 10, "did not deliver requested blocks")
        if behind:
            self._ask_headers(max(behind, key=lambda p: p.best_work))
        self._schedule()
        self._connect_wake.set()
