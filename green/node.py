"""A full node: HTTP API for clients, the peer-to-peer network (green/p2p.py),
mempool and miner.

Everything runs on this machine. By default the node only listens on
127.0.0.1, so nothing is reachable from outside unless you pass --host.

Security notes (details in SECURITY.md):
  * Nodes talk to each other only over the encrypted, authenticated P2P
    protocol; the HTTP API is for wallets and tools and cannot add peers or
    inject blocks from outside this machine.
  * Every network read has a size limit and a timeout; the server caps
    concurrent connections and rejects malformed Content-Length headers.
  * Admin endpoints (mining switch, adding peers, uploading job data) only
    answer requests from this machine.
  * Result data is only served once it has been revealed on chain, so a
    committed-but-unrevealed result cannot leak before it is paid.
  * Matmul delivery is optimistic: validators only see a batch's Merkle root,
    so a node checks every result delivered to its own wallet's jobs and
    disputes the wrong or withheld ones (the buyer-side watchtower).
"""

import ipaddress
import json
import os
import random
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import params, sealed, tx as txmod, work
from .chain import (BlobStore, Chain, InvalidBlock, MissingData, block_hash, fee_rate, seed_of, share_id,
                    target_of,
                    work_inputs)
from .crypto import Key, hhex, merkle_proof, merkle_root
from . import prune as prunemod, snapshot
from .chain import min_prune_keep
from .p2p import P2P
from .state import State
from .store import Store
from .tx import InvalidTx, make_tx

MAX_BLOB_BYTES = txmod.job_blob_size([params.MAX_JOB_UNITS_PER_DIM] * 3)
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CONCURRENT_REQUESTS = 64
REQUEST_TIMEOUT = 15
MAX_MEMPOOL = 20_000          # several full blocks
MAX_MEMPOOL_PER_SENDER = 100   # one key cannot fill the mempool on its own
RBF_MIN_BUMP_PERCENT = 10     # a replacement must pay this much more than the transaction it replaces
REVEAL_CONFIRMATIONS = 2      # wait until the commit has this many confirmations
COMMIT_INFLIGHT_CAP = 2 * params.MAX_BATCH_UNITS   # units a miner sells at once; the rest are lottery-only
BATCH_SECONDS = 2.0           # commit a partly filled batch after this long
BUMP_AFTER_BLOCKS = 3         # our own protocol transaction still waiting after this many blocks gets a higher fee
MAX_BUMP_FEE = 10_000_000     # ...up to this (0.1 GRN), so a fee spike cannot drain the wallet
BUMP_SECONDS = 5.0
MAX_SHARE_POOL = 16_384       # shares waiting to be included (~SHARE_FACTOR per block of the window)
# Shares mined on one parent block that a node checks and relays. The count before a
# block is random (geometric, mean SHARE_FACTOR): capped at a block's MAX_BLOCK_SHARES it
# would drop 13% of all shares, which later blocks of the window would have paid; at 4x
# it drops 0.03%, and still bounds a flood on one parent.
MAX_SHARES_PER_PARENT = 4 * params.MAX_BLOCK_SHARES
TEMPLATE_SECONDS = 5          # the miner rebuilds its block template at least this often
MINING_THREADS = max(1, min(4, (os.cpu_count() or 2) - 1))   # threads that prepare (CPU: compute) tries
SHARE_TEMPLATE_SECONDS = 1    # ...and for new shares at most this often
MAX_MANIFEST_BYTES = txmod.HASH_BYTES * params.MAX_BATCH_UNITS
RESPONSES_PER_ROUND = 32      # disputes answered per loop; the rest wait for mempool room
MAX_WAITERS = 16              # API clients long-polling for the next block at once
MAX_WAIT_SECONDS = 60
FEE_RECENT_BLOCKS = 6         # blocks the fee estimate looks back at
MAINTENANCE_SECONDS = 30      # how often a pruned node prunes and collects unused blobs
LOOPBACK = {"127.0.0.1", "::1"}
# Admin requests must come from this machine AND carry this header AND name this machine
# in Host. A web page open in a browser here can reach the loopback address too, but it
# cannot add a custom header to a cross-site request without a CORS preflight (which we
# never grant), and a DNS-rebinding page sends its own name as Host.
CLIENT_HEADER = "X-Green-Client"
LOCAL_HOST_NAMES = {"127.0.0.1", "localhost", "::1"}


def names_this_machine(host: str | None) -> bool:
    """Whether a Host header (with or without a port) is a loopback address or localhost."""
    h = (host or "").strip().lower()
    if h.startswith("["):                        # [::1]:8545
        h = h[1:h.find("]")] if "]" in h else ""
    elif h.count(":") == 1:                      # 127.0.0.1:8545
        h = h.split(":")[0]
    return h in LOCAL_HOST_NAMES


class TooLarge(ValueError):
    pass


def thread_stacks(frames: int = 12) -> dict:
    """Where every thread of this process is: thread name -> its last frames."""
    import sys
    import traceback
    names = {t.ident: t.name for t in threading.enumerate()}
    return {names.get(tid, str(tid)): traceback.format_stack(frame)[-frames:]
            for tid, frame in sys._current_frames().items()}


def tx_role(t: dict, addr: str) -> str | None:
    """How an address takes part in a transaction, named as the address index names it
    once the transaction is in a block (chain.py): sender, recipient, account, guardian."""
    if txmod.sender(t) == addr:
        return "sender"
    for field in ("to", "account", "guardian"):
        if t.get(field) == addr:
            return "recipient" if field == "to" else field
    return None


def _read_limited(resp, limit: int) -> bytes:
    data = resp.read(limit + 1)
    if len(data) > limit:
        raise TooLarge(f"response larger than {limit} bytes")
    return data


def http_get(url: str, timeout: float = 5.0, raw: bool = False, limit: int = MAX_RESPONSE_BYTES):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = _read_limited(r, limit)
    return data if raw else json.loads(data)


def wait_for_block(base_url: str, tip: str | None, timeout: float) -> str | None:
    """Wait until the node at `base_url` has a tip other than `tip`, at most `timeout`
    seconds (long-poll on GET /v1/wait); returns the tip. On a node without that
    endpoint it just sleeps, so clients work with older nodes too."""
    try:
        return http_get(f"{base_url}/v1/wait?tip={tip or ''}&timeout={timeout}", timeout=timeout + 10)["tip"]
    except (OSError, ValueError, KeyError):
        time.sleep(timeout)
        return tip


def http_post(url: str, body, timeout: float = 5.0, raw: bool = False):
    data = body if raw else json.dumps(body).encode()
    headers = {"Content-Type": "application/octet-stream" if raw else "application/json", CLIENT_HEADER: "1"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(_read_limited(r, MAX_RESPONSE_BYTES))


def _cpu_try(tr: dict) -> tuple[str, int]:
    """A try's transcript and ticket on the CPU (runs in the miner's threads)."""
    transcript = work.compute_unit(tr["a"], tr["b"], tr["seed"], product=False)[0]
    return transcript, work.ticket_value(tr["seed"], transcript)


class Node:
    def __init__(self, data_dir: str, genesis: dict, port: int = params.DEFAULT_PORT,
                 host: str = "127.0.0.1", peers=(), miner_key: Key | None = None, log=print,
                 use_gpu: bool = False, p2p_port: int = 0, listen: bool = True, seeds=None,
                 fast_sync: bool = False, p2p_host: str | None = None, no_signal=(), debug=None,
                 prune: bool = False, prune_keep: int = prunemod.PRUNE_KEEP_BLOCKS, dandelion: bool = True,
                 portmap: bool = False, portmap_gateway: tuple | None = None, verify_processes: bool = True):
        self.data_dir = data_dir
        self.port = port
        self.host = host
        self.url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
        self.log = log
        self.debug = debug                        # optional: detailed P2P messages (log level "debug")
        # A full node must run the exact wasmtime the network is pinned to; on any
        # other version it could judge a task challenge differently and fork.
        from . import wasm
        wasm.check_engine()
        self._gpu_lock = threading.Lock()
        self.gpu = self._init_gpu() if use_gpu else None
        self._gpu_checked = self.gpu                 # the GPU that passed its check (set_gpu)
        self.intensity = 100                      # percent of the time the miner works (GPU/CPU get the rest)
        self.lock = threading.RLock()
        self.tip_changed = threading.Condition(self.lock)   # wakes API clients waiting for a block
        self._waiters = 0
        self.mempool: dict[str, dict] = {}
        self.miner_key = miner_key
        self.no_signal = set(no_signal)           # deployments this miner chooses not to signal
        self.dandelion = dandelion                # our own transactions take a stem first (p2p.py)
        self.pool = None                          # a green.pool.Pool selling its miners' units through us
        self.portmap = portmap                    # ask the router to forward our P2P port (green/portmap.py)
        self.verify_processes = verify_processes  # check a long sync's work units in processes (green/verify.py)
        self.portmap_gateway = portmap_gateway    # (host, port); None: the default gateway, port 5351
        self.portmaps: list = []
        self._sell_lock = threading.Lock()        # one thread at a time commits and reveals batches
        self.prune = prune                        # keep recent block bodies and data in use only (green/prune.py)
        self.prune_keep = prune_keep
        if prune and prune_keep < min_prune_keep():
            raise ValueError(f"--prune-keep must be at least {min_prune_keep()} blocks")
        self.mining = False
        self.work_tasks = True                    # take paid WASM tasks while mining
        self.watch = True                         # check revealed task results, challenge frauds
        self.shutdown_requested = threading.Event()   # POST /shutdown; whoever runs the node stops it
        self._checked: dict[tuple, bool] = {}     # (task, output) -> our local verdict
        self.batch_buf: dict[str, dict] = {}      # job -> {"units", "results", "started"}: computed, not committed
        self.my_batches: dict[str, dict] = {}     # batch id -> {"job", "count", "root", "manifest"}: committed by us
        self._unit_ok: set[tuple] = set()         # (batch, unit_key) results we verified for our own jobs
        self._leaf_cache: dict[str, tuple] = {}   # manifest hash -> (root, leaves, set(leaves)), root-checked
        self._warned_bond = False
        self.stats = {"units": 0, "useful_units": 0, "blocks_found": 0, "shares_found": 0, "batches_committed": 0,
                      "results_revealed": 0, "tasks_done": 0, "frauds_caught": 0, "started": time.time()}
        self._recent_unit_times: deque = deque()
        self._task_inflight: dict[str, dict] = {}
        self._own_waiting: dict[tuple, int] = {}   # (sender, nonce) of our pending protocol tx -> height first seen
        self.share_pool: dict[str, dict] = {}      # share id -> share (SPEC.md §3.5), for our templates
        self._shares_on: dict[str, int] = {}        # parent block -> shares held that were mined on it
        self._share_cache: tuple = (None, -1, [])  # (tip, pool version, candidates)
        self._share_version = 0
        self._template = None                      # (key, made, header, txs, shares, share version)
        self._inflight_units = set()               # job units being tried right now (not yet buffered)
        self._committing: set = set()              # units taken from the buffer, their commit not yet in the mempool
        self._options: list | None = None          # open units for the batch of tries being built
        self._task_unrunnable: set[str] = set()
        self.running = True

        os.makedirs(data_dir, exist_ok=True)
        self.blobs = BlobStore(os.path.join(data_dir, "blobs"))
        self.store = Store(os.path.join(data_dir, "chain.db"))
        self.chain = Chain(genesis, self.blobs, store=self.store)
        if self.chain.height:
            self.log(f"Opened the chain at height {self.chain.height} (base state at {self.chain.base_height})")
        self._pending = self.chain.state.copy()
        self.snapshots: list[dict] = self.store.get_meta("snapshots", [])
        self.chain.on_snapshot = self._write_snapshot
        self.p2p = P2P(self, host=host if p2p_host is None else p2p_host, port=p2p_port, peers=peers, data_dir=data_dir, listen=listen, seeds=seeds,
                       fast_sync=fast_sync and self.chain.height == 0)

    # --- persistence ---

    def set_gpu(self, on: bool) -> None:
        """Mine on the GPU (checked against the CPU first) or on the CPU, from the next batch.
        The checked GPU is kept: switching back uses it without a second check, and the
        miner thread stays its only user (the GPU miner is not thread-safe)."""
        with self._gpu_lock:
            if on and self.gpu is None:
                if self._gpu_checked is None:
                    self._gpu_checked = self._init_gpu()
                self.gpu = self._gpu_checked
            elif not on:
                self.gpu = None

    def _init_gpu(self):
        """Load the GPU miner, but only trust it if it matches the CPU bit for bit.

        A GPU that produced a different digest would waste work on blocks every
        other node rejects, so a mismatch disables it rather than risking that.
        """
        from . import gpu
        try:
            miner = gpu.GPUMiner()                   # this node's own (a shared one is not thread-safe)
        except Exception:
            miner = None
        if miner is None:
            self.log("[gpu] no suitable GPU/driver found; mining on the CPU")
            return None
        for s in (b"gpu-selftest-1", b"gpu-selftest-2"):
            a, b = work.filler_inputs(s)
            tr_c, c_c = work.compute_unit(a, b, s)
            tr_g, _, c_g = miner.unit(a, b, s)
            import numpy as np
            if tr_c != tr_g or not np.array_equal(c_c, c_g):
                self.log("[gpu] self-test does not match the CPU; GPU disabled for safety, mining on the CPU")
                return None
        self.log(f"[gpu] {miner.name} verified, mining on the GPU")
        return miner

    def _write_snapshot(self, height: int, bh: str, state) -> None:
        """Called by the chain (inside the block's store transaction) when its base
        passes a snapshot height: write the ledger there as chunks, keep two."""
        manifest = snapshot.make_snapshot(state, height, bh, self.blobs.put)
        keep = [m for m in self.snapshots if m["height"] != height][-1:] + [manifest]
        live = {c for m in keep for c in m["chunks"]} | {m["id"] for m in keep}
        for m in self.snapshots:
            for blob in m["chunks"] + [m["id"]]:
                if blob not in live:
                    self.blobs.delete(blob)
        self.snapshots = keep
        self.store.set_meta("snapshots", keep)
        self.log(f"Snapshot of the ledger at height {height} written ({len(manifest['chunks'])} chunks)")

    def adopt_snapshot(self, headers: list[dict], state) -> None:
        """Install a verified snapshot (from the P2P fast sync) and restart the mempool on it."""
        with self.lock:
            self.chain.install_snapshot(headers, state)
            self.mempool = {}
            self._pending = self.chain.state.copy()
        self.log(f"Started from a verified snapshot at height {self.chain.height}; validating every block from here")

    def close(self) -> None:
        """Release the database (for a node that was never started, or after stop())."""
        self.store.close()

    # --- mempool ---

    def _rebuild_pending(self, must_keep: str | None = None) -> None:
        """Recompute the state after all mempool transactions; drop the stale ones.

        Arrival order, not nonce order: every transaction was accepted on top of
        the ones before it, so this keeps a payment and a spend of that payment
        (by another sender) in a valid order. If `must_keep` fails, its error is
        raised and nothing changes.
        """
        height = self.chain.height + 1
        s = self.chain.state.copy()
        keep = {}
        for tid, t in self.mempool.items():
            # The mempool is not a block: challenges beyond one block's adjudication
            # budget must wait for a later block, not be dropped.
            s.adjudication_used = 0
            s.responses_used = 0
            if tid == must_keep:
                self.chain.apply_tx(s, t, height)
                keep[tid] = t
                continue
            try:
                self.chain.apply_tx(s, t, height)
                keep[tid] = t
            except (InvalidTx, MissingData):
                pass
        self.mempool = keep
        self._pending = s

    def submit_tx(self, t: dict, source=None, stem: bool | None = None) -> str:
        """Validate a transaction and relay it. Our own transactions (and stem
        transactions from peers) first travel the Dandelion++ stem (p2p.py), so the
        network cannot tell which node they came from; the others go into the
        mempool and are announced."""
        try:
            tid = txmod.txid(t)
        except (TypeError, ValueError):
            raise InvalidTx("unencodable transaction") from None
        if stem is None:
            stem = source is None and self.dandelion
        announce_skip = source
        if stem:
            try:
                if self._stem(t, tid, source):
                    return tid
            except MissingData:
                pass                                 # it needs data: the normal path fetches it
            if source is not None:
                # The stem ends here. The peer that passed it on holds it only in its stem
                # pool, so it is announced to that peer as well.
                self.p2p.stats["stem_fluffed"] += 1
                announce_skip = None
        if self._with_blobs(lambda: self._add_tx(t, tid), source):
            self.p2p.announce_tx(tid, announce_skip)
        return tid

    def _stem_view(self, local_only: bool = False, skip: str | None = None) -> State:
        """A scratch state: the mempool's plus the stem transactions (only ours if
        `local_only`). Read it and drop it before releasing the lock."""
        s = State(self._pending)
        height = self.chain.height + 1
        for tid, e in self.p2p.stempool.items():
            if tid == skip or local_only and not e["local"]:
                continue
            s.adjudication_used = s.responses_used = 0
            try:
                self.chain.apply_tx(s, e["tx"], height)
            except (InvalidTx, MissingData):
                pass
        return s

    def _stem(self, t: dict, tid: str, source) -> bool:
        """Check a transaction against the mempool and the stem pool, then pass it on;
        False if it should be announced instead."""
        with self.lock:
            if tid in self.mempool:
                return True
            txmod.check_format(t, self.chain.chain_id)
            who = txmod.sender(t)
            if any(m["nonce"] == t["nonce"] and txmod.sender(m) == who for m in self.mempool.values()):
                return False                         # a replacement: the mempool's fee rules decide
            for other, e in list(self.p2p.stempool.items()):
                if e["local"] and e["tx"]["nonce"] == t["nonce"] and txmod.sender(e["tx"]) == who:
                    del self.p2p.stempool[other]     # a fee bump of our own stem transaction replaces it
                    return False                     # (and goes out at once: speed was the point)
            # A transaction follows its predecessor: on the same stem if that one is on our stem,
            # announced like it if it was announced (and is still on its way to other nodes). A
            # stem that overtook its predecessor would reach a node that cannot accept it yet.
            if source is None and t["nonce"] > self.chain.state.nonce(who) and not any(
                    e["tx"]["nonce"] == t["nonce"] - 1 and txmod.sender(e["tx"]) == who
                    for e in self.p2p.stempool.values()):
                return False
            s = self._stem_view(skip=tid)
            s.adjudication_used = s.responses_used = 0
            self.chain.apply_tx(s, t, self.chain.height + 1)
        return self.p2p.stem_tx(t, tid, source)

    def _add_tx(self, t: dict, tid: str) -> bool:
        with self.lock:
            if tid in self.mempool:
                return False
            txmod.check_format(t, self.chain.chain_id)
            who = txmod.sender(t)
            same = [k for k, m in self.mempool.items() if txmod.sender(m) == who]
            old = next((k for k in same if self.mempool[k]["nonce"] == t["nonce"]), None)
            if old is not None:
                return self._replace(old, t, tid)
            if len(same) >= MAX_MEMPOOL_PER_SENDER:
                raise InvalidTx("too many pending transactions from this sender")
            if len(self.mempool) < MAX_MEMPOOL:
                # Straight onto the pending state: a rejected transaction changes nothing
                # (tests/test_atomicity.py), and no layer is added per transaction.
                self._pending.adjudication_used = 0
                self._pending.responses_used = 0
                self.chain.apply_tx(self._pending, t, self.chain.height + 1)
                self.mempool[tid] = t
                return True
            # Full: the new transaction must outbid the cheapest one it pushes out.
            victim = self._eviction_candidate(exclude=who)
            if victim is None or fee_rate(self.mempool[victim]) >= fee_rate(t):
                raise InvalidTx("mempool is full and the fee rate is too low")
            saved = dict(self.mempool), self._pending
            del self.mempool[victim]
            self.mempool[tid] = t
            try:
                self._rebuild_pending(must_keep=tid)
            except (InvalidTx, MissingData):
                self.mempool, self._pending = saved
                raise
            return True

    def _eviction_candidate(self, exclude: str) -> str | None:
        """The lowest fee-rate transaction that is the last of its sender's queue
        (evicting an earlier one would strand the sender's later transactions)."""
        last: dict[str, tuple] = {}
        for tid, m in self.mempool.items():
            who = txmod.sender(m)
            if who != exclude and (who not in last or m["nonce"] > last[who][0]):
                last[who] = (m["nonce"], tid)
        if not last:
            return None
        return min((tid for _, tid in last.values()), key=lambda k: fee_rate(self.mempool[k]))

    def _replace(self, old: str, t: dict, tid: str) -> bool:
        """Replace-by-fee: the same sender and nonce, paying clearly more."""
        prev = self.mempool[old]
        need = prev["fee"] + max(params.MIN_TX_FEE, prev["fee"] * RBF_MIN_BUMP_PERCENT // 100)
        if t["fee"] < need:
            raise InvalidTx(f"replacement fee too low: must be at least {need}")
        saved = dict(self.mempool), self._pending
        self.mempool = {(tid if k == old else k): (t if k == old else v) for k, v in self.mempool.items()}
        try:
            self._rebuild_pending(must_keep=tid)
        except (InvalidTx, MissingData):
            self.mempool, self._pending = saved
            raise
        return True

    # --- shares (SPEC.md §3.5) ---

    def submit_share(self, share, source=None, own: bool = False) -> str | None:
        """Keep a share for our templates and pass it on. Returns its id if it is new and
        fits a block on our tip; None if it does not (too old, another branch, already
        included, data missing). Raises InvalidBlock if its work is fake: the sender lied."""
        if not isinstance(share, dict):
            raise InvalidBlock("a share is a header")
        try:
            sid = share_id(share)
        except (TypeError, ValueError):
            raise InvalidBlock("unencodable share") from None
        with self.lock:
            if sid in self.share_pool:
                return None
            if self._shares_on.get(share.get("prev"), 0) >= MAX_SHARES_PER_PARENT:
                return None                              # a flood on one parent: let it go unchecked
            state = self.chain.state
            problem = self.chain.share_problem(share, self.chain.tip, state.share_seen)
            job = share["work"]["job"] if problem is None else None
            if job is not None and job not in state.jobs:
                problem = "unknown job"
            job_size = txmod.job_blob_size(state.jobs[job]["dims"]) if job else 0
        if problem is not None:
            return None
        if own:                                      # we just computed it
            self.chain.share_verified(sid)
        else:
            try:
                self.chain.check_share_work(share, job_size)
            except MissingData:
                return None
        with self.lock:
            if sid in self.share_pool:                   # another peer's copy got here first
                return None
            self.share_pool[sid] = share
            self._shares_on[share["prev"]] = self._shares_on.get(share["prev"], 0) + 1
            while len(self.share_pool) > MAX_SHARE_POOL:
                self._drop_share(min(self.share_pool, key=lambda k: self.share_pool[k]["height"]))
            self._share_version += 1
        self.p2p.announce_share(sid, share, source)
        return sid

    def _drop_share(self, sid: str) -> None:
        """Forget a share (under self.lock)."""
        prev = self.share_pool.pop(sid)["prev"]
        left = self._shares_on.get(prev, 1) - 1
        if left > 0:
            self._shares_on[prev] = left
        else:
            self._shares_on.pop(prev, None)

    def share_candidates(self) -> list:
        """Shares a block on our tip may include (oldest first: they expire first), at most MAX_BLOCK_SHARES."""
        with self.lock:
            tip, version, cached = self._share_cache
            if tip == self.chain.tip and version == self._share_version:
                return cached
            state = self.chain.state
            keep = []
            for sid, sh in list(self.share_pool.items()):
                job = sh["work"]["job"]
                if self.chain.share_problem(sh, self.chain.tip, state.share_seen) is None \
                        and (job is None or job in state.jobs):
                    keep.append(sh)
                elif sh["height"] <= self.chain.height - params.SHARE_WINDOW + 1 or sid in state.share_seen:
                    self._drop_share(sid)            # it can never be included any more
            keep.sort(key=lambda sh: sh["height"])
            keep = keep[:params.MAX_BLOCK_SHARES]
            self._share_cache = (self.chain.tip, self._share_version, keep)
            return keep

    def pending_nonce(self, addr: str) -> int:
        with self.lock:
            return self._stem_view(local_only=True).nonce(addr)

    # --- blocks ---

    def submit_block(self, block: dict, source=None, relay: bool = True) -> str:
        status = self._with_blobs(lambda: self._add_block(block), source)
        if status == "orphan":
            self.p2p.on_orphan(source)            # fetch the headers that lead to it
        if status == "added" and relay:
            self.p2p.announce_block(block_hash(block), source)
        return status

    def _add_block(self, block: dict) -> str:
        with self.lock:
            old_tip = self.chain.tip
            abandoned = []
            status = self.chain.add_block(block)
            if self.chain.tip != old_tip:
                hdr = self.chain.tip_header()
                if hdr["prev"] != old_tip:
                    abandoned = self.chain.branch(old_tip)       # old tip is no longer on the main chain
                    self._restore_disconnected(abandoned)
                    self.log(f"[reorg] new heaviest chain, height {hdr['height']}")
                self._rebuild_pending()
                self.p2p.on_new_tip()
                self.tip_changed.notify_all()
            return status

    def _restore_disconnected(self, abandoned: list[str]) -> None:
        """After a reorg, put the transactions of abandoned blocks back in the mempool.

        Without this a payment confirmed on the losing branch would silently
        vanish. _rebuild_pending then drops the ones the new branch already has
        (their nonce is used) or that no longer apply.
        """
        back = {}
        for bh in reversed(abandoned):                  # oldest first
            for t in self.chain.blocks[bh]["txs"]:
                tid = txmod.txid(t)
                if tid not in self.mempool:
                    back[tid] = t
        # They are older than anything still waiting, which may build on them.
        rest = list(self.mempool.items())[:max(0, MAX_MEMPOOL - len(back))]
        self.mempool = {**back, **dict(rest)}
        if back:
            self.log(f"[reorg] {len(back)} transactions from {len(abandoned)} abandoned blocks returned to the mempool")

    def _with_blobs(self, fn, source=None):
        """Run fn; if it needs a data blob we lack, fetch it from peers and retry.

        Only ever fetches over the P2P network (preferring the peer that sent the
        transaction or block), never from an address named by a request, and
        never more than the size the protocol allows for that blob.
        """
        tried = set()
        while True:
            try:
                return fn()
            except MissingData as e:
                if e.blob_hash in tried or not self._fetch_blob(e.blob_hash, e.max_size, prefer=source):
                    raise
                tried.add(e.blob_hash)

    def _fetch_blob(self, blob_hash: str, max_size: int, prefer=None) -> bool:
        data = self.p2p.fetch_blob(blob_hash, max_size, prefer=prefer)
        if data is None or hhex(data) != blob_hash:
            return False
        self.blobs.put(data)
        return True

    def leaves(self, entry: dict, fetch: bool = False) -> list[str] | None:
        """A revealed batch's result hashes (in unit order), from its manifest.

        Only returned if they hash to the batch's revealed Merkle root, so a
        miner cannot hand out a list that differs from what it committed to.
        """
        m = entry.get("manifest")
        if not m:
            return None
        cached = self._leaf_cache.get(m)
        if cached is not None and cached[0] == entry["root"]:
            return cached[1]
        blob = self.blobs.get(m)
        if blob is None and fetch and self._fetch_blob(m, MAX_MANIFEST_BYTES):
            blob = self.blobs.get(m)
        found = txmod.manifest_leaves(blob, entry["count"], entry["root"])
        if found is not None:
            if len(self._leaf_cache) > 10_000:
                self._leaf_cache.clear()
            self._leaf_cache[m] = (entry["root"], found, set(found))
        return found

    def _revealed(self, state):
        """Every batch whose results are public: revealed live ones and settled ones."""
        for job in state.jobs.values():
            for b in job["batches"].values():
                if b["status"] == "revealed":
                    yield b
            yield from job["delivered"].values()

    def is_public_blob(self, blob_hash: str) -> bool:
        """Job data and revealed results are public; committed results are not (yet)."""
        with self.lock:
            state = self.chain.state
            if blob_hash in state.jobs:
                return True
            for m in self.snapshots:
                if blob_hash == m["id"] or blob_hash in m["chunks"]:
                    return True
            for b in self._revealed(state):
                if b["manifest"] == blob_hash:
                    return True
                if self.leaves(b) is not None and blob_hash in self._leaf_cache[b["manifest"]][2]:
                    return True
            for task in state.tasks.values():
                if blob_hash in (task["prog"], task["input"]) or task["output"] == blob_hash:
                    return True
            for t in self.mempool.values():
                if (t["type"] == "job_post" and t["job"] == blob_hash) or \
                        (t["type"] == "job_reveal" and t["manifest"] == blob_hash) or \
                        (t["type"] == "task_post" and blob_hash in (t["prog"], t["input"])):
                    return True
        return False

    # --- mining ---

    def _my_bond_ok(self, job: dict) -> bool:
        need = params.bond_for(job["price"]) + params.MIN_TX_FEE
        return self._pending.balance(self.miner_key.address) >= need

    def _busy_units(self) -> set:
        """(job, unit_key) pairs not to pick: in a mempool commit, computed and waiting in our
        buffer, being tried by our own miner right now, or handed to pool miners."""
        jobs = self.chain.state.jobs
        busy = {(t["job"], txmod.unit_key(u)) for t in self.mempool.values()
                if t["type"] == "job_commit" and t["job"] in jobs
                for u in txmod.bitmap_units(t["units"], jobs[t["job"]]["dims"]) or []}
        for job_id, buf in self.batch_buf.items():
            busy.update((job_id, txmod.unit_key(u)) for u in buf["units"])
        busy |= self._inflight_units                 # (the miner's batches are in flight a moment)
        busy |= self._committing                     # (and a batch between the buffer and the mempool)
        if self.pool is not None:
            busy |= self.pool.assigned_keys()        # handed to pool miners
        return busy

    def _work_options(self, busy: set) -> list:
        """The best-paying open units we can commit to, as (job, unit) pairs not in `busy`."""
        state = self.chain.state
        best_price, options = 0, []
        for job_id, job in state.jobs.items():
            if job["price"] < best_price:
                continue
            units = state.open_units(job_id)         # (a finished job answers at once)
            if not units or not self._my_bond_ok(job):
                continue
            units = [u for u in units if (job_id, txmod.unit_key(u)) not in busy]
            if not units:
                continue
            if job["price"] > best_price:
                best_price, options = job["price"], []
            options += [(job_id, u) for u in units]
        return options

    def _choose_work(self, busy: set) -> dict:
        """Prefer the best-paying unit we can commit to; fall back to filler work. `busy`:
        (job, unit key) pairs not to pick; the chosen one is added to it. Within one batch
        of tries the options are worked out once (self._options): scanning every job for
        every try made a miner 4x slower once a few hundred finished jobs piled up."""
        options = self._options if self._options is not None else self._work_options(busy)
        if not options:
            return {"job": None, "unit": None}
        job_id, unit = options.pop(random.randrange(len(options)))
        busy.add((job_id, txmod.unit_key(unit)))
        return {"job": job_id, "unit": unit}

    def _mine_loop(self) -> None:
        """Try nonces, in batches: on the GPU a batch is submitted before the previous
        one is collected, so the CPU prepares the next inputs while the GPU works."""
        addr = self.miner_key.address
        self.log(f"Mining started. Reward address: {addr}")
        if self.gpu is not None:
            self.gpu.discard()                       # (a restart after an error)
        self._inflight_units = set()
        pending = None                               # a GPU batch in flight: (tries, handle)
        # Tries are prepared (SHAKE-256 inputs, noise) and, on the CPU, computed in a few
        # threads: hashing and numpy release the interpreter lock (measured: 0.85 ms of
        # preparation per try alone, 0.41 ms each across 3 threads).
        pool = ThreadPoolExecutor(MINING_THREADS, thread_name_prefix="mine")
        try:
            while self.running:
                gpu = self.gpu                       # (it can be switched on or off while we run)
                if pending is not None and (not self.mining or pending[2] is not gpu):
                    self._finish_tries(pending[0], pending[2].collect(pending[1]))
                    pending = None
                if not self.mining:
                    time.sleep(0.2)
                    continue
                started = time.perf_counter()
                tries = self._next_tries(gpu.batch if gpu is not None else MINING_THREADS, pool)
                if not tries:
                    continue                         # (_next_tries waited)
                if gpu is None:
                    self._finish_tries(tries, list(pool.map(_cpu_try, tries)))
                else:
                    handle = gpu.submit_prepared([(tr["a1"], tr["b1"], tr["seed"]) for tr in tries])
                    if pending is not None:
                        self._finish_tries(pending[0], gpu.collect(pending[1]))
                    pending = (tries, handle, gpu)
                if self.intensity < 100:             # rest, so the hardware works intensity% of the time
                    time.sleep((time.perf_counter() - started) * (100 - self.intensity) / self.intensity)
        finally:
            pool.shutdown(wait=False)

    def _next_tries(self, count: int, pool: ThreadPoolExecutor) -> list[dict]:
        """`count` tries on the current block template, each with its own work unit and
        nonce and its inputs ready. The template (transactions, shares, state root) is
        rebuilt when the tip or the mempool changes, at least every TEMPLATE_SECONDS, and
        for new shares at most every SHARE_TEMPLATE_SECONDS: checking the share pool and
        building a template for every try cost more than the try itself."""
        addr = self.miner_key.address
        self._process_batches()
        with self.lock:
            key = (self.chain.tip, id(self.mempool), len(self.mempool))
            t, now = self._template, time.time()
            if t is None or t[0] != key or now - t[1] > TEMPLATE_SECONDS \
                    or (t[5] != self._share_version and now - t[1] > SHARE_TEMPLATE_SECONDS):
                version = self._share_version
                shares = self.share_candidates()
                base, txs, _ = self.chain.build_template(addr, list(self.mempool.values()),
                                                         signal_skip=self.no_signal, shares=shares)
                t = self._template = (key, now, base, txs, shares, version)
            _, _, base, txs, shares, _ = t
            ahead = base["time"] > time.time() + params.MAX_FUTURE_SECONDS - 5
            if ahead:
                self._template = None
            tip = self.chain.tip
            busy = self._busy_units()
            headers = []
            self._options = self._work_options(busy) if count and not ahead else None
            try:
                for _ in range(count if not ahead else 0):
                    header = dict(base, work=self._choose_work(busy), nonce=random.getrandbits(48))
                    if header["work"]["job"] is not None:
                        self._inflight_units.add((header["work"]["job"], txmod.unit_key(header["work"]["unit"])))
                    headers.append(header)
            finally:
                self._options = None
        if ahead:
            # The chain's clock (each block after the median of the last 11) ran ahead of ours,
            # as it does when blocks come many per second: a block now would be refused as
            # from the future, so wait for the clock instead of mining it -- outside the lock,
            # which waiting under starved the rest of the node (seen after a long outage).
            time.sleep(0.5)
            return []
        tries = []
        for header, inputs in zip(headers, pool.map(self._prepare_try, headers)):
            if isinstance(inputs, Exception):
                self.log(f"[warning] could not do the work unit: {inputs}")
                self._release_unit(header["work"])
                continue
            tries.append({"header": header, "tip": tip, "txs": txs, "shares": shares, **inputs})
        if not tries:
            time.sleep(1)                            # the work's data is missing: give it time to arrive
        return tries

    def _prepare_try(self, header: dict):
        """A try's seed and inputs (and A', B' for the GPU), or the error that stops it."""
        seed = seed_of(header)
        try:
            a, b = work_inputs(header, seed, self.blobs, self._job_size(header["work"]))
        except (MissingData, InvalidBlock, InvalidTx) as e:
            return e
        out = {"seed": seed, "a": a, "b": b}
        if self.gpu is not None:
            out["a1"], out["b1"] = self.gpu.factors(a, b, seed)
        return out

    def _release_unit(self, work_item: dict) -> None:
        if work_item["job"] is not None:
            with self.lock:
                self._inflight_units.discard((work_item["job"], txmod.unit_key(work_item["unit"])))

    def _finish_tries(self, tries: list, results: list) -> None:
        """What each try's ticket won: a share, a block, and for a job's unit a result to sell."""
        for tr, (transcript, ticket) in zip(tries, results):
            header, work_item = tr["header"], tr["header"]["work"]
            is_useful = work_item["job"] is not None
            self._count_unit(is_useful)
            # Delivery is separate from the lottery: results are collected into
            # batches (one commit + one reveal for up to MAX_BATCH_UNITS units),
            # and only a bounded number are in flight; the rest are just tickets.
            if is_useful and self._should_commit():
                self._buffer_result(work_item, work.recover_useful(tr["a"], tr["b"], tr["seed"]))
            self._release_unit(work_item)            # (now in the buffer, or not sold)
            if target_of(header) < ticket <= self.chain.share_target(header):
                # Not a block, but a share: paid from the reserve when a block includes it.
                try:
                    if self.submit_share(dict(header, transcript=transcript), own=True):
                        self.stats["shares_found"] += 1
                except (InvalidBlock, MissingData):
                    pass
            if ticket <= target_of(header):
                header["transcript"] = transcript
                block = {"header": header, "txs": tr["txs"], **({"shares": tr["shares"]} if tr["shares"] else {})}
                with self.lock:
                    stale = self.chain.tip != tr["tip"]
                if stale:
                    continue
                try:
                    status = self.submit_block(block)
                except (InvalidBlock, MissingData) as e:
                    self.log(f"[error] our own block was rejected: {e}")
                    continue
                if status == "added":
                    self.stats["blocks_found"] += 1
                    kind = "useful work" if work_item["job"] else "filler work"
                    self.log(f"*** BLOCK FOUND #{header['height']}  {block_hash(block)[:16]}...  "
                             f"({kind}, {len(tr['txs'])} txs, {len(tr['shares'])} shares)")
        self._flush_batches()

    def _job_size(self, work_item: dict) -> int:
        if work_item["job"] is None:
            return 0
        job = self.chain.state.jobs.get(work_item["job"])
        return txmod.job_blob_size(job["dims"]) if job else 0

    def _should_commit(self) -> bool:
        """Sell only a bounded number of units at a time (bonds are locked until payment)."""
        in_flight = (sum(len(b["units"]) for b in self.batch_buf.values())
                     + sum(b["count"] for b in self.my_batches.values()))
        return in_flight < COMMIT_INFLIGHT_CAP

    def _buffer_result(self, work_item: dict, useful, by: str | None = None) -> None:
        """Keep a computed unit for the next batch; `by` is the pool miner that computed
        it (None: this node), who the batch will pay for it."""
        blob = self.blobs.put(txmod.encode_result(useful))
        with self.lock:
            buf = self.batch_buf.setdefault(work_item["job"], {"units": [], "results": [], "by": [],
                                                               "started": time.time()})
            if list(work_item["unit"]) in [list(u) for u in buf["units"]]:
                return                               # computed twice: a batch may hold a unit only once
            buf["units"].append(work_item["unit"])
            buf["results"].append(blob)
            buf["by"].append(by)

    def _batch_payees(self, by: list) -> list | None:
        """Who a batch's payment goes to: each pool miner by the units it computed, the
        rest (this node's own units, the pool's fee) to this node. None: all to us."""
        if all(b is None for b in by):
            return None
        own = self.miner_key.address
        fee = self.pool.fee_percent if self.pool is not None else 0
        weights: dict[str, int] = {}
        for b in by:                                 # weights in hundredths: the fee is an exact percentage
            weights[b or own] = weights.get(b or own, 0) + (100 - fee)
        if fee:
            weights[self.pool.fee_address] = weights.get(self.pool.fee_address, 0) + fee * len(by)
        entries = sorted([a, w] for a, w in weights.items())
        return entries if len(entries) > 1 else None

    def sell_step(self) -> None:
        """Reveal what is due and commit full batches (the miner loop does this as it
        goes; a pool node that does not mine itself calls it on a timer)."""
        self._process_batches()
        self._flush_batches()

    def _flush_batches(self) -> None:
        """Commit a buffered batch once it is full, old enough, or its job has no open units left."""
        now = time.time()
        with self._sell_lock:
            with self.lock:
                state = self.chain.state
                busy = self._busy_units()
                ready = []
                for job_id, buf in list(self.batch_buf.items()):
                    more = job_id in state.jobs and any(
                        (job_id, txmod.unit_key(u)) not in busy for u in state.open_units(job_id))
                    if len(buf["units"]) >= params.MAX_BATCH_UNITS or not more or now - buf["started"] >= BATCH_SECONDS:
                        ready.append((job_id, self.batch_buf.pop(job_id)))
                        # Out of the buffer and not yet in the mempool, the units must still
                        # count as taken, or a pool miner (or our own) is handed them again.
                        self._committing.update((job_id, txmod.unit_key(u)) for u in ready[-1][1]["units"])
            for job_id, buf in ready:
                try:
                    self._commit_batch(job_id, buf)
                finally:                             # committed (mempool) or put back (buffer) by now
                    with self.lock:
                        self._committing.difference_update((job_id, txmod.unit_key(u)) for u in buf["units"])

    def _rebuffer(self, job_id: str, items: list) -> None:
        """Units pool miners computed that did not fit this batch wait for the next one
        (they are owed); this node's own leftovers are just dropped, as before."""
        theirs = [x for x in items if x[2] is not None]
        if not theirs:
            return
        with self.lock:
            buf = self.batch_buf.setdefault(job_id, {"units": [], "results": [], "by": [], "started": time.time()})
            for unit, result, by in theirs:
                buf["units"].append(unit)
                buf["results"].append(result)
                buf["by"].append(by)

    def _commit_batch(self, job_id: str, buf: dict) -> None:
        """Step 1 of selling: commit to the batch. The results stay private until the reveal."""
        addr = self.miner_key.address
        with self.lock:
            job = self.chain.state.jobs.get(job_id)
            bal = self._pending.balance(addr)
            nonce = self._pending.nonce(addr)
        if job is None:
            return
        items = list(zip(buf["units"], buf["results"], buf.get("by") or [None] * len(buf["units"])))
        # Units we cannot bond stay lottery tickets; selling them waits for funds.
        n = min(len(items), params.MAX_BATCH_UNITS,
                max(0, (bal - params.MIN_TX_FEE) // params.bond_for(job["price"])))
        if n == 0:
            if not self._warned_bond:
                self._warned_bond = True
                self.log("[info] not enough balance for a bond; selling compute starts after the first block reward")
            self._rebuffer(job_id, items)
            return
        # A batch pays at most MAX_PAYEES - 1 contributors besides the pool's fee.
        chosen, later, people = [], [], set()
        for item in items:
            if len(chosen) < n and (item[2] in people or len(people) < params.MAX_PAYEES - 1):
                chosen.append(item)
                people.add(item[2])
            else:
                later.append(item)
        self._rebuffer(job_id, later)
        dims = job["dims"]
        chosen.sort(key=lambda x: txmod.unit_index(x[0], dims))
        units, leaves = [u for u, _, _ in chosen], [r for _, r, _ in chosen]
        payees = self._batch_payees([b for _, _, b in chosen])
        bitmap = txmod.units_bitmap(units, dims)
        root = merkle_root(leaves)
        manifest = self.blobs.put(txmod.encode_manifest(leaves))
        c = txmod.commitment(addr, job_id, bitmap, root)
        extra = {"payees": payees} if payees else {}
        fields = {"job": job_id, "units": bitmap, "commitment": c, **extra}
        t = make_tx(self.miner_key, self.chain.chain_id, "job_commit", nonce,
                    self.protocol_fee("job_commit", nonce, **fields), **fields)
        try:
            self.submit_tx(t, stem=False)
        except (InvalidTx, MissingData) as e:
            self.log(f"[warning] could not commit the batch: {e}")
            self._rebuffer(job_id, chosen)
            return
        self.my_batches[c] = {"job": job_id, "count": len(chosen), "root": root, "manifest": manifest, **extra}
        self.stats["batches_committed"] += 1

    def _process_batches(self) -> None:
        with self._sell_lock:
            self._process_batches_locked()

    def _process_batches_locked(self) -> None:
        """Step 2: reveal our batches once their commit is buried; answer disputes against them."""
        addr = self.miner_key.address
        with self.lock:
            state = self.chain.state
            height = self.chain.height
            queued = {(t["type"], t.get("batch") or t.get("commitment")) for t in self.mempool.values()
                      if t["type"] in ("job_commit", "job_reveal")}
            answering = {(t["batch"], txmod.unit_key(t["unit"])) for t in self.mempool.values()
                         if t["type"] == "job_respond"}
            reveals, responds = [], []
            for bid, info in list(self.my_batches.items()):
                job = state.jobs.get(info["job"])
                batch = job["batches"].get(bid) if job else None
                if batch is None or batch["miner"] != addr:
                    if ("job_commit", bid) not in queued:
                        del self.my_batches[bid]       # paid, forfeit, expired or lost the race
                    continue
                if (batch["status"] == "committed" and height - batch["height"] + 1 >= REVEAL_CONFIRMATIONS
                        and height + 1 <= batch["expires"] and ("job_reveal", bid) not in queued):
                    reveals.append((info["job"], bid, info))
            # Disputes are answered from chain state and the manifest on disk, not
            # from memory, so this keeps working after a restart.
            for job_id, job in state.jobs.items():
                for bid, batch in job["batches"].items():
                    if batch["miner"] != addr or not batch["disputes"]:
                        continue
                    for key, d in batch["disputes"].items():
                        if height + 1 <= d["deadline"] and (bid, key) not in answering:
                            responds.append((job_id, job["dims"], bid, dict(batch), key))
        for job_id, bid, info in reveals:
            if self._try_tx("job_reveal", job=job_id, batch=bid, root=info["root"], manifest=info["manifest"]):
                self.stats["results_revealed"] += info["count"]
                self.log(f"Useful work delivered: job {job_id[:12]}... {info['count']} units (batch {bid[:12]}...)")
        for job_id, dims, bid, batch, key in responds[:RESPONSES_PER_ROUND]:
            leaves = self.leaves(batch)
            if leaves is None:
                self.log(f"[dispute] batch {bid[:12]}... is disputed but its manifest is not on this node")
                continue
            unit = [int(x) for x in key.split(",")]
            index = txmod.bitmap_units(batch["units"], dims).index(unit)
            if self._try_tx("job_respond", job=job_id, batch=bid, unit=unit, result=leaves[index],
                            proof=merkle_proof(leaves, index)):
                self.log(f"[dispute] batch {bid[:12]}... unit {unit} was disputed, proving it on chain")

    # --- general compute worker ---

    def _task_worker_loop(self) -> None:
        """Two jobs: earn by running posted WASM tasks, and guard the market.

        Worker: commit to a task's output, then reveal it to get paid.
        Watchtower: re-run every revealed result locally and challenge the wrong
        ones. An optimistic market is only as safe as its watchers -- without
        this, nobody would ever challenge and a lying worker would always be
        paid. A watchtower only challenges what it has already proven wrong
        itself, so honest results are never dragged into on-chain adjudication,
        and it earns the cheater's bond for the trouble.
        """
        from . import wasm
        addr = self.miner_key.address
        # Kept on the node, not in this function, so a restart of the loop after an
        # error still reveals what was committed (forgetting it would lose the bond).
        inflight = self._task_inflight       # task_id -> {"commitment","output","output_blob","height"}
        unrunnable = self._task_unrunnable
        while self.running:
            time.sleep(1.0)
            with self.lock:
                height = self.chain.height
                tasks = {tid: dict(t) for tid, t in self.chain.state.tasks.items()}
                busy = {t["task"] for t in self.mempool.values()
                        if t.get("type") in ("task_commit", "task_reveal", "task_challenge")}
                pending_commits = {t["task"]: t["commitment"] for t in self.mempool.values()
                                   if t.get("type") == "task_commit"}
                bal = self._pending.balance(addr)
            if self.watch:
                self._watch_tasks(tasks, busy, height, bal)
                self._watch_jobs(height, bal)
            if not (self.mining and self.work_tasks):
                continue
            # Reveal anything we committed that is now buried enough.
            for tid, info in list(inflight.items()):
                t = tasks.get(tid)
                r = t["reserved"] if t else None
                if not t or r is None or r.get("worker") != addr or r["commitment"] != info["commitment"]:
                    # Not on chain yet is not the same as lost: while our commit waits in
                    # the mempool, keep it -- dropping it here would leave the commit
                    # unrevealed, forfeit the bond and make us commit the task twice.
                    if pending_commits.get(tid) != info["commitment"]:
                        inflight.pop(tid, None)
                    continue
                if t["status"] != "open" or height > r["expires"]:
                    inflight.pop(tid, None)
                    continue
                if height - r["height"] >= REVEAL_CONFIRMATIONS and tid not in busy:
                    self.blobs.put(info["output_blob"])
                    self._try_tx("task_reveal", task=tid, output=info["output"])
                    inflight.pop(tid, None)
            # Pick one open, unreserved task we can afford and run it.
            if len(inflight) < COMMIT_INFLIGHT_CAP:
                for tid, t in tasks.items():
                    if t["status"] != "open" or tid in inflight or tid in busy or tid in unrunnable:
                        continue
                    if t["reserved"] is not None and height <= t["reserved"]["expires"]:
                        continue
                    if bal < params.task_bond(t["price"]) + params.MIN_TX_FEE:
                        continue
                    io = self._task_io(t)
                    if io is None:
                        continue
                    try:
                        output, _ = wasm.run(io[0], io[1], t["fuel"])
                    except wasm.ComputeError as e:
                        # Deterministic: it will fail the same way next time too.
                        unrunnable.add(tid)
                        self.log(f"[task] {tid[:12]} cannot be run, skipping: {e}")
                        continue
                    out_hash = wasm.result_hash(output)
                    c = txmod.task_commitment(addr, tid, out_hash)
                    if self._try_tx("task_commit", task=tid, commitment=c):
                        inflight[tid] = {"commitment": c, "output": out_hash,
                                         "output_blob": output, "height": height}
                        self.stats["tasks_done"] += 1
                        self.log(f"[task] {tid[:12]} computed, commitment sent")
                    break

    def _task_io(self, t: dict):
        """(program, input) bytes for a task, fetching them from peers if needed."""
        prog = self.blobs.get(t["prog"])
        inp = self.blobs.get(t["input"])
        if prog is None:
            self._fetch_blob(t["prog"], params.MAX_TASK_PROGRAM_BYTES)
        if inp is None:
            self._fetch_blob(t["input"], params.MAX_TASK_INPUT_BYTES)
        prog = prog or self.blobs.get(t["prog"])
        inp = inp if inp is not None else self.blobs.get(t["input"])
        return None if prog is None or inp is None else (prog, inp)

    def _watch_tasks(self, tasks: dict, busy: set, height: int, bal: int) -> None:
        from . import wasm
        addr = self.miner_key.address
        for tid, t in tasks.items():
            if t["status"] != "revealed" or t["worker"] == addr or tid in busy:
                continue
            if t["settle_at"] is None or height + 1 > t["settle_at"]:
                continue
            key = (tid, t["output"])
            if key not in self._checked:
                io = self._task_io(t)
                if io is None:
                    continue
                if len(self._checked) > 10_000:
                    self._checked.clear()
                self._checked[key] = wasm.verify(io[0], io[1], t["output"], t["fuel"])
            if self._checked[key]:
                continue
            fee = params.challenge_fee(t["fuel"], t["prog_size"])
            if bal < params.task_bond(t["price"]) + fee:
                self.log(f"[watchtower] {tid[:12]} has a false result, but the balance cannot cover the challenge bond")
                continue
            if self._try_tx("task_challenge", fee, task=tid):
                self.stats["frauds_caught"] += 1
                self.log(f"[watchtower] {tid[:12]} false result detected, challenge sent "
                         f"(worker {t['worker'][:12]}... will lose its bond)")

    def _watch_jobs(self, height: int, bal: int) -> None:
        """Check every result delivered to our own jobs; dispute wrong or withheld ones.

        Validators only see a Merkle root, so the buyer is the one who has to
        look. Freivalds makes that cheap (well under a millisecond per unit).
        Every bad unit can be disputed at once, and a single unanswered one
        costs the miner the bond of its whole batch.
        """
        addr = self.miner_key.address
        with self.lock:
            busy = {t["batch"] for t in self.mempool.values() if t["type"] in ("job_challenge", "job_respond")}
            todo = [(job_id, job["dims"], bid, {**b, "disputes": dict(b["disputes"]), "proven": set(b["proven"])})
                    for job_id, job in self.chain.state.jobs.items() if job["buyer"] == addr
                    for bid, b in job["batches"].items()
                    if b["status"] == "revealed" and b["miner"] != addr
                    and bid not in busy and height + 1 <= b["settle_at"]]
        for job_id, dims, bid, batch in todo:
            bad, missing = self._check_batch(job_id, dims, bid, batch)
            late = height + 1 >= batch["settle_at"] - params.JOB_CHALLENGE_BLOCKS // 2
            # Not fetched yet may just be slow; call data withheld only late in the window.
            units = [u for u in bad + (missing if late else [])
                     if txmod.unit_key(u) not in batch["disputes"] and txmod.unit_key(u) not in batch["proven"]]
            if not units:
                continue
            fee = txmod.job_challenge_fee(len(units))
            if bal < fee:
                self.log(f"[watchtower] batch {bid[:12]} is bad, but the balance cannot cover the dispute fee")
                continue
            if self._try_tx("job_challenge", fee, job=job_id, batch=bid, units=txmod.units_bitmap(units, dims)):
                self.stats["frauds_caught"] += 1
                bal -= fee
                self.log(f"[watchtower] job {job_id[:12]}: {len(bad)} wrong and {len(units) - len(bad)} withheld "
                         f"units disputed (miner {batch['miner'][:12]}... loses the batch bond unless it answers all)")

    def _check_batch(self, job_id: str, dims, bid: str, batch: dict):
        """(wrong units, unavailable units) of a revealed batch."""
        try:
            a, b = self.blobs.job(job_id, txmod.job_blob_size(dims))
        except (MissingData, InvalidTx):
            return [], []
        units = txmod.bitmap_units(batch["units"], dims)
        leaves = self.leaves(batch, fetch=True)
        if leaves is None:                       # no manifest that matches the root
            return [], units
        bad, missing = [], []
        for u, rh in zip(units, leaves):
            key = (bid, txmod.unit_key(u))
            if key in self._unit_ok:
                continue
            data = self.blobs.get(rh)
            if data is None and self._fetch_blob(rh, txmod.RESULT_BYTES):
                data = self.blobs.get(rh)
            if data is None:
                missing.append(u)
                continue
            try:
                c = txmod.decode_result(data)
            except ValueError:
                bad.append(u)
                continue
            ua, ub = txmod.unit_inputs(a, b, u)
            # Private randomness: the miner cannot know which checks we run.
            if not work.freivalds_check(ua, ub, c, os.urandom(32)):
                bad.append(u)
                continue
            if len(self._unit_ok) > 100_000:
                self._unit_ok.clear()
            self._unit_ok.add(key)
        return bad, missing

    def protocol_fee(self, kind: str, nonce: int, fee: int = params.MIN_TX_FEE, **fields) -> int:
        """A fee for one of our own transactions that should get it into the next blocks
        (the node's estimate for its size), and never below `fee`."""
        rate = self.fee_estimate(2)["fee_rate"]
        if not rate:
            return fee
        size = txmod.tx_size(make_tx(self.miner_key, self.chain.chain_id, kind, nonce, fee, **fields))
        return max(fee, min(MAX_BUMP_FEE, int(rate * size) + 1))       # the cap never undercuts a required fee

    def _try_tx(self, kind: str, fee: int = params.MIN_TX_FEE, **fields) -> bool:
        """The node's own protocol transactions (commits, reveals, answers, challenges)
        skip the stem: they have deadlines, their sender is the miner address blocks
        already show, and the loops that send them look for them in the mempool. They pay
        what the fee market asks, and _bump_stuck raises the fee of any that still wait."""
        nonce = self.pending_nonce(self.miner_key.address)
        fee = self.protocol_fee(kind, nonce, fee, **fields)
        t = make_tx(self.miner_key, self.chain.chain_id, kind, nonce, fee, **fields)
        try:
            self.submit_tx(t, stem=False)
            return True
        except (InvalidTx, MissingData) as e:
            self.log(f"[tx] {kind} rejected: {e}")
            return False

    def _bump_stuck(self) -> list[str]:
        """Replace our own protocol transactions that have waited BUMP_AFTER_BLOCKS blocks
        with the same transaction at a higher fee (replace-by-fee): a reveal or a dispute
        answer stuck behind a fee spike would otherwise miss its deadline and lose a bond.
        Returns the new txids."""
        if self.miner_key is None:
            return []
        me = self.miner_key.address
        with self.lock:
            height = self.chain.height
            mine = {(t["nonce"]): t for t in self.mempool.values() if txmod.sender(t) == me
                    and t["pub"] == self.miner_key.pub_hex}
            self._own_waiting = {k: h for k, h in self._own_waiting.items() if k[1] in mine}
            stuck = []
            for nonce, t in mine.items():
                first = self._own_waiting.setdefault((me, nonce), height)
                if height - first >= BUMP_AFTER_BLOCKS:
                    stuck.append(t)
        bumped = []
        for t in stuck:
            fields = {k: v for k, v in t.items() if k not in txmod.COMMON}
            need = t["fee"] + max(params.MIN_TX_FEE, t["fee"] * RBF_MIN_BUMP_PERCENT // 100)   # what RBF accepts
            fee = min(MAX_BUMP_FEE, max(self.protocol_fee(t["type"], t["nonce"], **fields),
                                        t["fee"] + max(params.MIN_TX_FEE, t["fee"] // 4)))
            if fee < need:
                continue                                  # at the cap: leave it
            new = make_tx(self.miner_key, self.chain.chain_id, t["type"], t["nonce"], fee, **fields)
            try:
                bumped.append(self.submit_tx(new, stem=False))
                with self.lock:
                    self._own_waiting[(me, t["nonce"])] = height      # give the new fee its own chance
                self.log(f"[fees] {t['type']} still waiting after {BUMP_AFTER_BLOCKS} blocks: fee raised "
                         f"{t['fee']} -> {fee}")
            except (InvalidTx, MissingData) as e:
                self.log(f"[fees] could not raise the fee of {t['type']}: {e}")
        return bumped

    def _bump_loop(self) -> None:
        while self.running:
            time.sleep(BUMP_SECONDS)
            self._bump_stuck()

    def _count_unit(self, useful: bool) -> None:
        now = time.time()
        with self.lock:
            self.stats["units"] += 1
            if useful:
                self.stats["useful_units"] += 1
            times = self._recent_unit_times          # the last 30 s, oldest first (rebuilding the list
            times.append(now)                        # for every unit cost O(units per 30 s) each)
            while now - times[0] >= 30:
                times.popleft()

    def units_per_second(self) -> float:
        times = self._recent_unit_times
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) / max(1e-9, times[-1] - times[0])

    # --- API views ---

    def status(self) -> dict:
        with self.lock:
            hdr = self.chain.tip_header()
            target = target_of(hdr)
            return {
                "coin": params.COIN_NAME, "genesis": self.chain.genesis_hash,
                "height": self.chain.height, "tip": self.chain.tip, "time": hdr["time"],
                "chainwork": str(self.chain.meta[self.chain.tip]["chainwork"]),
                "target": hdr["target"], "units_per_block": round(params.MAX_TARGET / (target + 1), 2),
                "peers": sorted(p.listen_addr or p.name for p in self.p2p.all_peers()),
                "p2p": self.p2p.status(),
                "mempool": len(self.mempool),
                "mining": self.mining, "miner": self.miner_key.address if self.miner_key else None,
                "gpu": self.gpu.name if self.gpu is not None else None, "intensity": self.intensity,
                "units_per_second": round(self.units_per_second(), 2), "stats": dict(self.stats),
                "warnings": self.warnings(),
                "pruned": self.prune, "first_block": self.chain.first_block,
                "reserve": self.chain.state.reserve, "burned": self.chain.state.burned,
                "shares_waiting": len(self.share_pool),
                "next_emission": dict(zip(("finder", "per_share", "inclusion_cut"),
                                          params.emission(self.chain.state.reserve))),
            }

    def maintain(self) -> dict:
        """Prune old block bodies and collect unused blobs (pruned nodes)."""
        with self.lock:
            dropped = self.chain.prune(self.prune_keep)
            height = self.chain.height
        result = prunemod.collect(self, height, keep=self.prune_keep)
        if dropped or result["deleted"]:
            self.log(f"[prune] dropped {dropped} old block bodies, deleted {result['deleted']} unused blobs")
        return {"bodies_dropped": dropped, **result}

    def _supervised(self, name: str, loop) -> None:
        """Run a long-lived loop; an unexpected error is logged and the loop restarts,
        instead of the thread dying silently. During shutdown it just ends."""
        while self.running:
            try:
                loop()
                return
            except Exception as e:
                if not self.running:
                    return                        # e.g. the database closed under it at shutdown
                self.log(f"[error] the {name} hit {type(e).__name__}: {e}; restarting it")
                time.sleep(1)

    def _maintenance_loop(self) -> None:
        while self.running:
            time.sleep(MAINTENANCE_SECONDS)
            try:
                self.maintain()
            except Exception as e:                # never stop maintaining because of one failure
                self.log(f"[prune] error: {type(e).__name__}: {e}")

    def wait_for_block(self, tip: str, timeout: float) -> dict:
        """Long-poll: return as soon as the tip is no longer `tip` (or after `timeout`).
        Lets wallets and agents react to new blocks without polling."""
        timeout = max(0.0, min(float(timeout), MAX_WAIT_SECONDS))
        with self.lock:
            if self._waiters >= MAX_WAITERS:
                raise OverflowError("too many clients waiting for a block")
            self._waiters += 1
            try:
                if self.chain.tip == tip:
                    self.tip_changed.wait_for(lambda: self.chain.tip != tip or not self.running, timeout)
                return {"tip": self.chain.tip, "height": self.chain.height, "changed": self.chain.tip != tip,
                        "mempool": len(self.mempool)}
            finally:
                self._waiters -= 1

    def fee_estimate(self, blocks: int = 2) -> dict:
        """A fee rate (yaprak per byte) that should get a transaction into one of the
        next `blocks` blocks: where it would land in the mempool's fee order, and
        what recent full blocks needed. Never below the minimum fee."""
        blocks = max(1, min(int(blocks), 100))
        with self.lock:
            pool = sorted(self.mempool.values(), key=fee_rate, reverse=True)
            recent = [self.chain.block(h) for h in self.chain.main[-FEE_RECENT_BLOCKS:]]
        room_bytes, room_txs = blocks * params.MAX_BLOCK_BYTES * 9 // 10, blocks * params.MAX_BLOCK_TXS * 9 // 10
        queue_rate, used = 0.0, 0
        for i, t in enumerate(pool):
            used += txmod.tx_size(t)
            if used > room_bytes or i + 1 > room_txs:
                queue_rate = fee_rate(t)              # to get in, outbid what sits at the cut
                break
        full_rates = []
        for b in recent:
            txs = b["txs"] if b else []
            if txs and (len(txs) >= params.MAX_BLOCK_TXS * 9 // 10
                        or sum(txmod.tx_size(t) for t in txs) >= params.MAX_BLOCK_BYTES * 9 // 10):
                full_rates.append(min(fee_rate(t) for t in txs))
        recent_rate = sorted(full_rates)[len(full_rates) // 2] if full_rates else 0.0
        rate = max(queue_rate, recent_rate) * 1.1 if max(queue_rate, recent_rate) else 0.0
        transfer_size = 330                           # a typical signed transfer, in bytes
        return {"blocks": blocks, "fee_rate": round(rate, 4),
                "transfer_fee": max(params.MIN_TX_FEE, int(rate * transfer_size) + 1 if rate else 0),
                "min_fee": params.MIN_TX_FEE, "mempool": len(pool),
                "basis": "mempool" if queue_rate >= recent_rate and queue_rate else
                         ("recent blocks" if recent_rate else "minimum")}

    # --- explorer and light-client views ---

    def tx_view(self, txid: str) -> dict | None:
        """A transaction: pending, or where it is on the main chain."""
        with self.lock:
            t = self.mempool.get(txid)
            if t is not None:
                return {"txid": txid, "status": "pending", "tx": t}
            e = self.p2p.stempool.get(txid)
            if e is not None and e["local"]:         # ours, still on its stem
                return {"txid": txid, "status": "pending", "relay": "stem", "tx": e["tx"]}
            loc = self.store.tx_location(txid)
            if loc is None:
                return None
            height, bh, pos = loc
            block = self.chain.block(bh)
            return {"txid": txid, "status": "confirmed", "height": height, "block": bh, "position": pos,
                    "confirmations": self.chain.height - height + 1,
                    "tx": block["txs"][pos] if block else None, "pruned": block is None}

    def address_history(self, addr: str, limit: int = 50, before: int | None = None) -> dict:
        """What an address took part in, newest first, plus what is pending for it."""
        limit = max(1, min(int(limit), 500))
        with self.lock:
            rows = self.store.address_history(addr, limit, before)
            mine = [(tid, e["tx"]) for tid, e in self.p2p.stempool.items() if e["local"]]    # still on a stem
            pending = [{"txid": tid, "type": t["type"], "role": tx_role(t, addr)}
                       for tid, t in list(self.mempool.items()) + mine if tx_role(t, addr)]
        items = [{"height": h, "position": pos, "role": role, "type": kind,
                  ("block" if role == "miner" else "txid"): ref} for h, pos, ref, role, kind in rows]
        return {"address": addr, "pending": pending, "items": items,
                "next": rows[-1][0] if len(rows) == limit else None}

    def tx_proof(self, txid: str) -> dict | None:
        """A Merkle proof that `txid` is in a main-chain block (green/light.py checks it)."""
        with self.lock:
            loc = self.store.tx_location(txid)
            block = self.chain.block(loc[1]) if loc else None
        if block is None:
            return None
        ids = [txmod.txid(t) for t in block["txs"]]
        return {"txid": txid, "block": loc[1], "height": loc[0], "index": loc[2], "count": len(ids),
                "proof": merkle_proof(ids, loc[2]), "header": block["header"]}

    def headers(self, start: int, count: int) -> list[dict]:
        count = max(1, min(int(count), 2000))
        with self.lock:
            hashes = self.chain.main[max(0, start):max(0, start) + count]
        return [self.chain.header(h) for h in hashes]

    def warnings(self) -> list[str]:
        """Things the operator must act on: e.g. a rule change this software does not know."""
        bits = self.chain.deployments.unknown_signals(self.chain.tip)
        return [f"most recent blocks signal an unknown rule change (version bit {b}): "
                f"upgrade this node before it activates" for b in bits]

    def account(self, addr: str) -> dict:
        with self.lock:
            pending = self._stem_view(local_only=True)
            view = {"address": addr,
                    "balance": self.chain.state.balance(addr), "nonce": self.chain.state.nonce(addr),
                    "pending_balance": pending.balance(addr), "pending_nonce": pending.nonce(addr),
                    "guard": None}
            g = self.chain.state.guards.get(addr)
            if g is not None:
                # What the account may still spend in the next block's period.
                window = (self.chain.height + 1) // g["period"]
                spent = g["spent"] if g["window"] == window else 0
                view["guard"] = {**g, "spend_left": max(0, g["limit"] - spent)}
            return view

    def channel_view(self, cid: str) -> dict | None:
        """An open compute channel (green/compute.py); None until it is confirmed and after it ends."""
        with self.lock:
            ch = self.chain.state.channels.get(cid)
            if ch is None:
                return None
            return {"id": cid, **ch, "left": ch["amount"] - ch["paid"], "height": self.chain.height}

    def supply(self) -> dict:
        """Audit: every coin sits in a balance, an escrow, a bond or a channel, or in the
        protocol's reserve, and together they are the fixed total minus the fees burned."""
        with self.lock:
            state, height = self.chain.state, self.chain.height
            balances = sum(state.balances.values())
            locked = state.locked_total()
            reserve, burned = state.reserve, state.burned
        return {"height": height, "balances": balances, "locked": locked, "reserve": reserve,
                "burned": burned, "total": balances + locked + reserve, "supply": params.TOTAL_SUPPLY - burned,
                "ok": balances + locked + reserve == params.TOTAL_SUPPLY - burned}

    def metrics(self) -> str:
        """Prometheus text format, for dashboards and alerts."""
        st = self.status()
        with self.lock:
            open_disputes = self.chain.state.open_disputes
            base = self.chain.base_height
        rows = [
            ("green_height", "gauge", "Height of the main chain", st["height"]),
            ("green_base_height", "gauge", "Height of the flat base state", base),
            ("green_peers", "gauge", "Verified peers", len(st["peers"])),
            ("green_mempool_transactions", "gauge", "Transactions waiting in the mempool", st["mempool"]),
            ("green_mining", "gauge", "1 if this node is mining", int(st["mining"])),
            ("green_units_per_second", "gauge", "Work units hashed per second", st["units_per_second"]),
            ("green_open_disputes", "gauge", "Matmul disputes waiting for an answer", open_disputes),
            ("green_blocks_found_total", "counter", "Blocks this node found", st["stats"]["blocks_found"]),
            ("green_work_units_total", "counter", "Work units this node hashed", st["stats"]["units"]),
            ("green_useful_units_total", "counter", "Hashed units that were buyers' work", st["stats"]["useful_units"]),
            ("green_results_delivered_total", "counter", "Matmul results this node delivered",
             st["stats"]["results_revealed"]),
            ("green_frauds_caught_total", "counter", "Frauds this node's watchtowers disputed",
             st["stats"]["frauds_caught"]),
            ("green_peers_inbound", "gauge", "Inbound P2P connections", st["p2p"]["inbound"]),
            ("green_peers_outbound", "gauge", "Outbound P2P connections", st["p2p"]["outbound"]),
            ("green_headers_ahead", "gauge", "Checked headers whose blocks are not connected yet",
             st["p2p"]["headers_ahead"]),
            ("green_blocks_in_flight", "gauge", "Blocks requested from peers", st["p2p"]["blocks_in_flight"]),
            ("green_peers_banned_total", "counter", "Peers banned for misbehaving", st["p2p"]["stats"]["banned"]),
            ("green_stempool_transactions", "gauge", "Transactions on their Dandelion++ stem through this node",
             st["p2p"]["stempool"]),
            ("green_stem_embargo_fired_total", "counter",
             "Stem transactions this node had to announce itself (a rising rate: stem peers drop them)",
             st["p2p"]["stats"]["embargo_fired"]),
            ("green_compact_blocks_rebuilt_total", "counter", "Compact blocks rebuilt from the mempool",
             st["p2p"]["stats"]["cmpct_rebuilt"]),
            ("green_compact_block_fallbacks_total", "counter", "Compact blocks that had to be fetched whole",
             st["p2p"]["stats"]["cmpct_fallbacks"]),
            ("green_bytes_received_total", "counter", "P2P bytes received",
             sum(p.bytes_in for p in self.p2p.all_peers())),
            ("green_bytes_sent_total", "counter", "P2P bytes sent", sum(p.bytes_out for p in self.p2p.all_peers())),
            ("green_supply_ok", "gauge", "1 if every coin is accounted for (supply audit)", int(self.supply()["ok"])),
            ("green_reserve", "gauge", "Coins in the protocol's reserve (yaprak)", st["reserve"]),
            ("green_shares_waiting", "gauge", "Shares waiting for a block", st["shares_waiting"]),
            ("green_shares_found_total", "counter", "Shares this node found", st["stats"]["shares_found"]),
            ("green_warnings", "gauge", "Warnings the operator should act on", len(st["warnings"])),
        ]
        out = []
        for name, kind, help_, value in rows:
            out += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}", f"{name} {value}"]
        return "\n".join(out) + "\n"

    def manifest_hints(self, job_id: str) -> list[str]:
        job = self.chain.state.jobs.get(job_id)
        if job is None:
            return []
        entries = [b for b in job["batches"].values() if b["status"] == "revealed"] + list(job["delivered"].values())
        return [b["manifest"] for b in entries if b["manifest"] and self.leaves(b) is None]

    def job_view(self, job_id: str, job: dict) -> dict:
        """`results` maps units to result hashes for every revealed batch whose manifest
        this node has (checked against the root); `pending` maps units still in a
        challenge window to their batch, which is what a buyer needs to dispute them."""
        dims = job["dims"]
        total = len(txmod.all_units(dims))
        results, pending, batches = {}, {}, []
        for bid, b in list(job["batches"].items()) + list(job["delivered"].items()):
            settled = bid in job["delivered"]
            keys = [txmod.unit_key(u) for u in txmod.bitmap_units(b["units"], dims)]
            status = "settled" if settled else b["status"]
            if status in ("revealed", "settled"):
                leaves = self.leaves(b)
                if leaves is not None:
                    results.update(zip(keys, leaves))
            if status == "revealed":
                pending.update((k, bid) for k in keys)
            if not settled:
                batches.append({"id": bid, "miner": b["miner"], "status": status, "units": keys,
                                "root": b["root"], "manifest": b["manifest"], "expires": b["expires"],
                                "settle_at": b["settle_at"], "disputes": sorted(b["disputes"]),
                                "proven": sorted(b["proven"])})
        return {"id": job_id, "buyer": job["buyer"], "dims": dims, "price": job["price"],
                "escrow": job["escrow"], "units_total": total, "units_done": len(job["done"]),
                "units_pending": len(pending), "units_reserved": len(job["reserved"]) - len(pending),
                "closes_at": job["closes_at"], "closed": job["closed"], "results": results,
                "pending": pending, "batches": batches, "height": self.chain.height}

    @staticmethod
    def task_view(task_id: str, task: dict) -> dict:
        return {"id": task_id, "buyer": task["buyer"], "prog": task["prog"], "input": task["input"],
                "prog_size": task["prog_size"], "fuel": task["fuel"], "price": task["price"],
                "status": task["status"], "worker": task["worker"], "output": task["output"],
                "settle_at": task["settle_at"],
                "challenge_fee": params.challenge_fee(task["fuel"], task["prog_size"])}

    # --- lifecycle ---

    def start(self, mine: bool = False) -> None:
        self.server = BoundedServer((self.host, self.port), make_handler(self))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.p2p.start()
        if self.portmap and self.p2p.listen:
            m = self.map_port(self.p2p.port)
            # A non-global "external" address (private, or 100.64/10 carrier-grade NAT, which
            # is_private does not cover) means another NAT further out: the mapping does not
            # make us reachable, so nothing is advertised.
            if m is not None and ipaddress.ip_address(m.external_ip).is_global:
                self.p2p.public_port = m.external_port
                self.p2p.my_host = m.external_ip          # what to tell peers to dial
        if self.prune:
            threading.Thread(target=self._maintenance_loop, daemon=True, name="maintenance").start()
        if self.miner_key:
            self.mining = mine
            threading.Thread(target=self._supervised, args=("miner", self._mine_loop), daemon=True).start()
            threading.Thread(target=self._supervised, args=("task worker", self._task_worker_loop),
                             daemon=True).start()
            threading.Thread(target=self._supervised, args=("fee bumper", self._bump_loop), daemon=True).start()

    def map_port(self, port: int):
        """Ask the router to forward `port` (kept renewed until stop); the PortMapper, or None."""
        from .portmap import PortMapper
        host, gw_port = self.portmap_gateway or (None, 5351)
        m = PortMapper(port, gateway=host, gateway_port=gw_port, log=self.log)
        if not m.start():
            return None
        self.portmaps.append(m)
        return m

    def stop(self) -> None:
        self.running = False
        for m in self.portmaps:
            m.stop()                             # remove the router's forwarding
        with self.lock:
            self.tip_changed.notify_all()        # release API clients waiting for a block
        self.server.shutdown()
        self.server.server_close()
        self.p2p.stop()
        with self.lock:                 # no block is being written while the database closes
            self.store.close()


class BoundedServer(ThreadingHTTPServer):
    """Threaded HTTP server that refuses connections beyond a fixed limit."""

    daemon_threads = True

    def __init__(self, addr, handler):
        super().__init__(addr, handler)
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address):
        pass  # a broken client connection is not our problem; never print tracebacks for it


def _route(path: str) -> list[str]:
    """Path segments, with the stable version prefix removed: /v1/status and /status
    are the same endpoint (the unprefixed form is kept for old clients)."""
    parts = [p for p in path.split("/") if p]
    return parts[1:] if parts[:1] == ["v1"] else parts


def make_handler(node: Node):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"        # one request per connection
        timeout = REQUEST_TIMEOUT            # applies to every socket read

        def log_message(self, *args):
            pass

        def _send(self, code: int, body, raw: bool = False):
            data = body if raw else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream" if raw else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self, limit: int) -> bytes:
            self._consumed = True
            raw = self.headers.get("Content-Length", "0").strip()
            if not raw.isdigit():
                raise ValueError("bad Content-Length")
            length = int(raw)
            if length > limit:
                raise TooLarge("request too large")
            return self.rfile.read(length)

        def _local(self) -> bool:
            return self.client_address[0] in LOOPBACK

        def do_GET(self):
            try:
                self._get()
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})
            except Exception:
                self._send(500, {"error": "internal error"})

        def _get(self):
            u = urlparse(self.path)
            parts = _route(u.path)
            q = parse_qs(u.query)
            if parts == ["status"]:
                return self._send(200, node.status())
            if parts[:1] == ["debug"] and parts[1:] in (["threads"], ["process"]):
                # For diagnosing a node that hangs, and for launchers (admin only).
                if not (self._local() and self.headers.get(CLIENT_HEADER) and names_this_machine(self.headers.get("Host"))):
                    return self._send(403, {"error": f"admin endpoint: only from this machine, with an {CLIENT_HEADER} header"})
                if parts[1] == "process":
                    return self._send(200, {"pid": os.getpid()})
                return self._send(200, thread_stacks())
            if parts == ["supply"]:
                return self._send(200, node.supply())
            if parts == ["metrics"]:
                data = node.metrics().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if parts == ["peers"]:
                return self._send(200, [p.info() for p in node.p2p.all_peers()])
            if parts == ["headers"]:
                return self._send(200, node.headers(int(q.get("from", ["0"])[0]), int(q.get("count", ["2000"])[0])))
            if len(parts) == 2 and parts[0] == "tx":
                view = node.tx_view(parts[1])
                return self._send(200, view) if view else self._send(404, {"error": "unknown transaction"})
            if len(parts) == 3 and parts[0] == "proof" and parts[1] == "tx":
                proof = node.tx_proof(parts[2])
                return self._send(200, proof) if proof else self._send(404, {"error": "no proof: unknown or pruned"})
            if len(parts) == 3 and parts[0] == "address" and parts[2] == "history":
                before = q.get("before", [None])[0]
                return self._send(200, node.address_history(parts[1], int(q.get("limit", ["50"])[0]),
                                                            int(before) if before is not None else None))
            if parts == ["fee"]:
                return self._send(200, node.fee_estimate(int(q.get("blocks", ["2"])[0])))
            if parts == ["wait"]:
                try:
                    return self._send(200, node.wait_for_block(q.get("tip", [""])[0],
                                                               float(q.get("timeout", ["30"])[0])))
                except OverflowError as e:
                    return self._send(503, {"error": str(e)})
            if parts == ["deployments"]:
                with node.lock:
                    return self._send(200, node.chain.deployments.info(node.chain.tip))
            if parts == ["chain"]:
                start = max(0, int(q.get("from", ["0"])[0]))
                with node.lock:
                    return self._send(200, {"hashes": node.chain.main[start:start + 5000]})
            if len(parts) == 2 and parts[0] == "block":
                block = node.chain.block(parts[1])
                return self._send(200, block) if block else self._send(404, {"error": "unknown block"})
            if len(parts) == 2 and parts[0] == "height":
                block = node.chain.block_at(int(parts[1]))
                return self._send(200, block) if block else self._send(404, {"error": "no such height"})
            if len(parts) == 2 and parts[0] == "account":
                return self._send(200, node.account(parts[1]))
            if parts == ["providers"]:
                return self._send(200, sorted(node.p2p.offer_list(), key=lambda o: (o["price"], o["provider"])))
            if len(parts) == 2 and parts[0] == "channel":
                view = node.channel_view(parts[1])
                return self._send(200, view) if view else self._send(404, {"error": "no open channel with that id"})
            if parts == ["mempool"]:
                with node.lock:
                    return self._send(200, list(node.mempool.values()))
            if parts == ["jobs"]:
                with node.lock:
                    jobs = [node.job_view(k, v) for k, v in node.chain.state.jobs.items()]
                return self._send(200, jobs)
            if len(parts) == 2 and parts[0] == "job":
                if self._local():
                    # A buyer asking its own node: fetch the manifests it lacks from peers
                    # first, so `results` covers everything that has been delivered.
                    with node.lock:
                        hints = node.manifest_hints(parts[1])
                    for m in hints:
                        node._fetch_blob(m, MAX_MANIFEST_BYTES)
                with node.lock:
                    job = node.chain.state.jobs.get(parts[1])
                    view = node.job_view(parts[1], job) if job else None
                return self._send(200, view) if view else self._send(404, {"error": "unknown job"})
            if parts == ["tasks"]:
                with node.lock:
                    return self._send(200, [node.task_view(k, v) for k, v in node.chain.state.tasks.items()])
            if len(parts) == 2 and parts[0] == "task":
                with node.lock:
                    task = node.chain.state.tasks.get(parts[1])
                    view = node.task_view(parts[1], task) if task else None
                return self._send(200, view) if view else self._send(404, {"error": "unknown task"})
            if len(parts) == 2 and parts[0] == "blob":
                data = None
                if node.is_public_blob(parts[1]):
                    data = node.blobs.get(parts[1])
                    # Results travel from the miner to the buyer on demand; validators never
                    # fetch them. Only local clients may make us fetch, so no amplification.
                    if data is None and self._local() and node._fetch_blob(parts[1], MAX_BLOB_BYTES):
                        data = node.blobs.get(parts[1])
                return self._send(200, data, raw=True) if data is not None else self._send(404, {"error": "unknown blob"})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                self._post()
            except TooLarge as e:
                self._send(413, {"error": str(e)})
            except MissingData as e:
                self._send(409, {"error": str(e)})
            except (InvalidTx, InvalidBlock) as e:
                self._send(400, {"error": str(e)})
            except PermissionError as e:
                self._send(403, {"error": str(e)})
            except (ValueError, KeyError, TypeError, RecursionError) as e:
                self._send(400, {"error": f"bad request: {e}"})
            except Exception:
                self._send(500, {"error": "internal error"})

        def _admin(self):
            if not (self._local() and self.headers.get(CLIENT_HEADER) and names_this_machine(self.headers.get("Host"))):
                # Read a small unread body first: closing a socket with unread data sends
                # a TCP reset, and the client would see that instead of the 403.
                raw = self.headers.get("Content-Length", "0").strip()
                if not getattr(self, "_consumed", False) and raw.isdigit() and int(raw) <= 64 * 1024:
                    self.rfile.read(int(raw))
                raise PermissionError(f"admin endpoint: only from this machine, with an {CLIENT_HEADER} header")

        def _post(self):
            parts = _route(urlparse(self.path).path)
            if parts == ["blob"]:
                self._admin()
                return self._send(200, {"hash": node.blobs.put(self._body(MAX_BLOB_BYTES))})
            if parts == ["relay"]:
                # A sealed request (green/sealed.py) for provider X-Relay-To, carried by a peer.
                self._admin()
                data = self._body(sealed.MAX_SEALED)
                try:
                    return self._send(200, node.p2p.relay(self.headers.get("X-Relay-To", ""), data), raw=True)
                except OSError as e:
                    return self._send(503, {"error": str(e)})
            body = json.loads(self._body(MAX_JSON_BYTES) or b"null")
            if parts == ["tx"]:
                return self._send(200, {"txid": node.submit_tx(body)})
            if parts == ["peers"]:
                self._admin()
                return self._send(200, {"added": node.p2p.add_manual(body["addr"])})
            if parts == ["mining"]:
                self._admin()
                if not node.miner_key:
                    return self._send(400, {"error": "node has no wallet"})
                if "intensity" in body:
                    level = body["intensity"]
                    if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 100:
                        return self._send(400, {"error": "intensity is a percentage, 1..100"})
                    node.intensity = level
                if "gpu" in body:
                    node.set_gpu(bool(body["gpu"]))
                if "on" in body:
                    node.mining = bool(body["on"])
                return self._send(200, {"mining": node.mining, "intensity": node.intensity,
                                        "gpu": node.gpu.name if node.gpu is not None else None})
            if parts == ["shutdown"]:
                self._admin()
                node.shutdown_requested.set()            # the process running the node stops it (cli.py)
                return self._send(200, {"stopping": True})
            return self._send(404, {"error": "not found"})

    return Handler
