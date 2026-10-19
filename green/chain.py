"""Blocks, validation, fork choice and the ledger's state transitions.

The ledger itself lives in diff layers (green/state.py): each block's state is
a thin layer over its parent's, and layers older than the reorg window are
merged into one flat base. With a Store (green/store.py) every accepted block
is persisted atomically and a restart loads the base plus recent diffs.
"""

import heapq
import os
import re
import threading
import time

from . import params, tx as txmod, versionbits, work
from .crypto import ZERO_HASH, canonical, h, hhex, is_address, is_hex, merkle_root, merkle_verify, verify
from .state import State
from .tx import InvalidTx

HEADER_FIELDS = {"version", "height", "prev", "time", "target", "tx_root", "shares_root", "state_root",
                 "miner", "work", "nonce", "msg", "transcript"}
# The base sits this far below the tip beyond the reorg window, so every block a
# fork may still replace has its own diff layer.
BASE_MARGIN = 32
# Blocks kept in memory below the base, for header lookbacks (the median time
# reaches back MTP_WINDOW blocks). Older blocks are read from the store.
HEADER_KEEP = 64
MAX_MSG_LEN = 200
# Checkpoints (SPEC.md §5.6): a network may name, in its genesis message, a key whose
# signed checkpoints every node follows until a height (or until the key signs an end):
# "checkpoints <Ed25519 public key, hex> until <height>". While they last, no chain
# that leaves out the latest checkpoint is accepted, however much work it has.
CHECKPOINT_RE = re.compile(r"checkpoints ([0-9a-f]{64}) until ([0-9]{1,10})")
MAX_CACHE = 100_000
MAX_JOB_BLOB = txmod.job_blob_size([params.MAX_JOB_UNITS_PER_DIM] * 3)
JOB_CACHE = 8


class InvalidBlock(Exception):
    """A block that cannot be accepted.

    `permanent` is False when the failure does not prove the header itself is
    bad (wrong body sent with a good header, a timestamp that is only *now* too
    far in the future, a fork that is too deep for *this* node). Only permanent
    failures are remembered, otherwise an attacker could get an honest block
    blacklisted by sending a corrupted copy of it first.
    """

    def __init__(self, msg: str, permanent: bool = True):
        super().__init__(msg)
        self.permanent = permanent


class MissingData(Exception):
    """A blob (job or result data) needed for validation is not available yet."""

    def __init__(self, blob_hash: str, max_size: int):
        super().__init__(f"missing data blob {blob_hash}")
        self.blob_hash = blob_hash
        self.max_size = max_size


# --- Blobs ---------------------------------------------------------------------

class BlobStore:
    """Content-addressed storage for job matrices and results."""

    def __init__(self, path: str | None = None):
        self.path = path
        self._mem: dict[str, bytes] = {}
        self._jobs: dict[str, tuple] = {}
        self._job_lock = threading.Lock()
        if path:
            os.makedirs(path, exist_ok=True)

    def put(self, data: bytes) -> str:
        key = hhex(data)
        if self.path:
            file = os.path.join(self.path, key)
            if not os.path.exists(file):
                with open(file + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(file + ".tmp", file)
        else:
            self._mem[key] = data
        return key

    def get(self, key: str) -> bytes | None:
        if not is_hex(key, 64):
            return None
        if self.path:
            file = os.path.join(self.path, key)
            if os.path.exists(file):
                with open(file, "rb") as f:
                    return f.read()
            return None
        return self._mem.get(key)

    def delete(self, key: str) -> None:
        if self.path:
            if is_hex(key, 64):
                try:
                    os.remove(os.path.join(self.path, key))
                except OSError:
                    pass
        else:
            self._mem.pop(key, None)

    def keys(self) -> list[str]:
        """Every stored blob's hash."""
        if self.path:
            return [n for n in os.listdir(self.path) if is_hex(n, 64)]
        return list(self._mem)

    def has(self, key: str) -> bool:
        if self.path:
            return is_hex(key, 64) and os.path.exists(os.path.join(self.path, key))
        return key in self._mem

    def job(self, job_id: str, size: int):
        """Decoded (A, B) for a job, or raise MissingData.

        A few recently used jobs stay decoded (a job can be 32 MiB, so the cache
        is small and least-recently-used jobs are dropped)."""
        with self._job_lock:
            cached = self._jobs.pop(job_id, None)
        if cached is None:
            blob = self.get(job_id)
            if blob is None:
                raise MissingData(job_id, size)
            try:
                cached = txmod.decode_job(blob)
            except ValueError as e:
                raise InvalidTx(f"malformed job data: {e}") from None
        with self._job_lock:                         # verification threads share the cache
            self._jobs[job_id] = cached              # most recently used last
            while len(self._jobs) > JOB_CACHE:
                del self._jobs[next(iter(self._jobs))]
        return cached


# --- Checkpoints ---------------------------------------------------------------

class InvalidCheckpoint(ValueError):
    pass


def checkpoint_authority(genesis: dict) -> tuple[str, int] | None:
    """(public key, last height) named by the genesis message, or None."""
    m = CHECKPOINT_RE.search(genesis["header"].get("msg", ""))
    return (m.group(1), int(m.group(2))) if m else None


def checkpoint_bytes(chain_id: str, height: int, block: str) -> bytes:
    return canonical(["green-checkpoint", chain_id, height, block])


def checkpoint_end_bytes(chain_id: str) -> bytes:
    return canonical(["green-checkpoint-end", chain_id])


# --- Block helpers -------------------------------------------------------------

def block_hash(block: dict) -> str:
    return hhex(canonical(block["header"]))


def seed_of(header: dict) -> bytes:
    return h(canonical({k: v for k, v in header.items() if k != "transcript"}))


def work_for_target(target: int) -> int:
    return 2**256 // (target + 1)


def target_of(header: dict) -> int:
    return int(header["target"], 16)


def fmt_target(target: int) -> str:
    return f"{target:064x}"


def work_inputs(header: dict, seed: bytes, blobs: BlobStore, job_size: int = 0):
    w = header["work"]
    if w["job"] is None:
        return work.filler_inputs(seed)
    try:
        a, b = blobs.job(w["job"], job_size)
    except InvalidTx as e:
        raise InvalidBlock(str(e)) from None
    mu, ku, nu = txmod.job_dims_units(a, b)
    i, l, j = w["unit"]
    if not (i < mu and l < ku and j < nu):
        raise InvalidBlock("work unit outside the job")
    return txmod.unit_inputs(a, b, w["unit"])


def solve(header: dict, blobs: BlobStore):
    """Do the work unit described by `header`.

    Returns (transcript, ticket_value, useful_result_or_None).
    """
    seed = seed_of(header)
    a, b = work_inputs(header, seed, blobs)
    transcript, useful = work.compute_unit(a, b, seed, product=header["work"]["job"] is not None)
    return transcript, work.ticket_value(seed, transcript), useful


def initial_state() -> State:
    """The ledger before the genesis block: every coin in the protocol's reserve."""
    s = State()
    s.reserve = params.TOTAL_SUPPLY
    s.commit()
    return s


def share_id(share: dict) -> str:
    return hhex(canonical(share))


def pay_block(s: State, hdr: dict, fees: int, shares: list) -> None:
    """A block's rewards (SPEC.md §3.5): from the reserve, the finder's part of the
    emission plus a cut of each share it includes, and each share's reward to the
    share's miner; of the fees, FEE_RESERVE_PERCENT back into the reserve,
    FEE_BURN_PERCENT destroyed and the rest to the finder. No coin is ever created:
    the supply is TOTAL_SUPPLY minus what was burned."""
    finder, per_share, cut = params.emission(s.reserve)
    to_reserve = fees * params.FEE_RESERVE_PERCENT // 100
    burn = fees * params.FEE_BURN_PERCENT // 100
    for sh in shares:
        for addr, amount in payouts(sh["miner"], per_share - cut):
            s.credit(addr, amount)
    for addr, amount in payouts(hdr["miner"], finder + cut * len(shares) + fees - to_reserve - burn):
        s.credit(addr, amount)
    s.reserve = s.reserve - finder - per_share * len(shares) + to_reserve
    if burn:
        s.burned += burn


def make_genesis(miner: str, msg: str, timestamp: int | None = None, on_try=None,
                 target: int | None = None) -> dict:
    """Mine the genesis block. It pays the first block's reward to `miner`. `target` is
    the network's starting difficulty (at most INITIAL_TARGET, i.e. never easier): a
    network that knows its miners starts where blocks come every 15 s, instead of
    paying out its first hours' coins in minutes while the difficulty catches up."""
    target = params.INITIAL_TARGET if target is None else target
    if not 0 < target <= params.INITIAL_TARGET:
        raise ValueError("the starting target must be positive and no easier than INITIAL_TARGET")
    blobs = BlobStore()
    first = initial_state().copy()
    header = {
        "version": versionbits.TOP_BITS, "height": 0, "prev": ZERO_HASH, "time": int(timestamp or time.time()),
        "target": fmt_target(target), "tx_root": merkle_root([]), "shares_root": merkle_root([]),
        "miner": miner, "work": {"job": None, "unit": None}, "nonce": 0, "msg": msg[:MAX_MSG_LEN],
    }
    pay_block(first, header, 0, [])
    header["state_root"] = first.commit()
    while True:
        transcript, ticket, _ = solve(header, blobs)
        if on_try:
            on_try(header["nonce"])
        if ticket <= target:
            header["transcript"] = transcript
            return {"header": header, "txs": []}
        header["nonce"] += 1


def asert(anchor_target: int, anchor_parent_time: int, anchor_height: int,
          parent_time: int, parent_height: int, halflife: int) -> int:
    """Per-block difficulty: aserti3-2d (ASERT, as specified for Bitcoin Cash in 2020).

    The target follows how far the chain is ahead of or behind its ideal
    schedule since the anchor: every `halflife` seconds behind schedule doubles
    the target (halves the difficulty), every `halflife` ahead halves it. It
    depends only on the parent and the anchor, so there are no windows to game,
    no oscillation, and no lookback. 2^x is approximated in 16-bit fixed point
    with the published cubic; integer only, so every node gets the same bits.
    Division rounds down (toward minus infinity).
    """
    time_delta = parent_time - anchor_parent_time
    height_delta = parent_height - anchor_height
    exponent = ((time_delta - params.TARGET_BLOCK_SECONDS * (height_delta + 1)) * 65536) // halflife
    shifts, frac = exponent >> 16, exponent & 0xFFFF
    factor = 65536 + ((195766423245049 * frac + 971821376 * frac * frac + 5127 * frac ** 3 + 2 ** 47) >> 48)
    target = anchor_target * factor
    target = target << shifts if shifts >= 0 else target >> -shifts
    return max(1, min(params.MAX_TARGET, target >> 16))


def valid_miner(miner) -> bool:
    """A block's `miner`: an address, or [[address, weight], ...] with 2..MAX_PAYEES
    entries sorted by address, addresses distinct, weights in 1..MAX_PAYEE_WEIGHT."""
    if isinstance(miner, str):
        return is_address(miner)
    return txmod.is_payee_list(miner)


def payouts(miner, total: int) -> list[tuple[str, int]]:
    """How a block's reward and fees (or a batch's payment) are split: by weight,
    rounded down, the remainder to the first payee."""
    if isinstance(miner, str):
        return [(miner, total)]
    weight = sum(w for _, w in miner)
    out = [(a, total * w // weight) for a, w in miner]
    out[0] = (out[0][0], out[0][1] + total - sum(x for _, x in out))
    return out


def min_prune_keep() -> int:
    """The fewest block bodies a pruned node may keep: every block a reorg can
    replace, plus the ones held in memory below the base."""
    return params.MAX_REORG_DEPTH + BASE_MARGIN + HEADER_KEEP + 1


def fee_rate(t: dict) -> float:
    """Fee per byte: what block space is sold for."""
    return t["fee"] / txmod.tx_size(t)


def block_order(candidates: list[dict]):
    """Highest fee per byte first, each sender's transactions in nonce order.

    The greedy rule: at every step take the best-paying transaction whose
    sender has nothing earlier left. Not consensus -- any order of valid
    transactions makes a valid block -- just what an honest miner earns most with.
    """
    queues: dict[str, list] = {}
    for t in candidates:
        try:
            queues.setdefault(txmod.sender(t), []).append(t)
        except (TypeError, KeyError, ValueError, AttributeError):
            continue
    heap = []
    for pub, q in queues.items():
        q.sort(key=lambda t: (t.get("nonce", 0), -t.get("fee", 0)))
        heapq.heappush(heap, (-fee_rate(q[0]), pub, 0))
    while heap:
        _, pub, i = heapq.heappop(heap)
        q = queues[pub]
        yield q[i]
        if i + 1 < len(q):
            heapq.heappush(heap, (-fee_rate(q[i + 1]), pub, i + 1))


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x < 2**63


# --- Chain -----------------------------------------------------------------------

class Chain:
    def __init__(self, genesis: dict, blobs: BlobStore, store=None):
        self.blobs = blobs
        self.store = store                       # green.store.Store, or None for an in-memory chain
        self.blocks: dict[str, dict] = {}        # recent blocks; older ones are in the store
        self.meta: dict[str, dict] = {}          # hash -> {"height", "chainwork"}, same span as blocks
        self.states: dict[str, State] = {}       # the base and every block above it
        self.invalid: set[str] = set()
        self.verified_results: set[tuple] = set()
        self.verified_shares: set[str] = set()   # shares whose work unit was recomputed and matched
        self.valid_programs: set[str] = set()
        self.verdicts: dict[tuple, bool] = {}
        # Blocks whose work unit was already recomputed and matched (in parallel,
        # ahead of connection, during sync), so add_block does not redo it.
        self.prechecked: set[str] = set()
        # Called with (height, block hash, flat state) whenever the base passes a
        # multiple of SNAPSHOT_INTERVAL (the node writes a snapshot there).
        self.on_snapshot = None
        self.deployments = versionbits.Deployments(self)

        self._check_genesis(genesis)
        g = block_hash(genesis)
        self.genesis_hash = g
        self.chain_id = g
        self.genesis = genesis
        self.initial_target = target_of(genesis["header"])   # the network's starting difficulty
        self.authority = checkpoint_authority(genesis)        # (key, until) or None
        self.checkpoint = store.get_meta("checkpoint", None) if store is not None else None
        self.checkpoint_end = store.get_meta("checkpoint_end", None) if store is not None else None
        if store is not None and store.get_meta("genesis") is not None:
            if store.get_meta("genesis") != g:
                raise ValueError("this data directory belongs to another network (different genesis)")
            self._open(store)
            return
        self.blocks[g] = genesis
        self.meta[g] = {"height": 0, "chainwork": work_for_target(self.initial_target)}
        base = self.apply_block(initial_state(), genesis).flattened()
        self.states[g] = base
        self.base_hash, self.base_height = g, 0
        self.tip = g
        self.main = [g]
        if store is not None:
            with store.transaction():
                store.put_block(g, genesis, 0, self.meta[g]["chainwork"])
                store.set_main_from(0, [g], [self._index_rows(g, genesis)])
                store.write_base(base.diff())
                store.set_meta("base", [g, 0])
                store.set_meta("genesis", g)
                store.set_meta("txindex", 1)

    def _open(self, store) -> None:
        """Load a chain from the store: the base state plus the diffs above it.

        Nothing is re-executed, so this takes the same time at height 100 or
        10 million (apart from reading the main-chain index).
        """
        self.base_hash, self.base_height = store.get_meta("base")
        self.main = store.main_hashes()
        self.tip = self.main[-1]
        for bh, height, work, block in store.blocks_from(max(0, self.base_height - HEADER_KEEP)):
            self.blocks[bh] = block
            self.meta[bh] = {"height": height, "chainwork": work}
        base = store.load_base()
        self.states = {self.base_hash: base}
        for bh, block in self.blocks.items():          # by height, so parents come first
            prev = block["header"]["prev"]
            if self.meta[bh]["height"] > self.base_height and prev in self.states:
                diff = store.get_diff(bh)
                if diff is not None:
                    self.states[bh] = State.from_diff(self.states[prev], diff)
        if self.tip not in self.states:
            raise ValueError("store is inconsistent: no state for the tip")
        if store.get_meta("txindex") != 1:
            self._reindex(store)

    def _reindex(self, store) -> None:
        """Build the transaction and address index from the stored blocks (a data
        folder made before v0.12 has none). Pruned bodies are skipped."""
        with store.transaction():
            store.clear_index()
            for start in range(0, len(self.main), 1000):
                rows = []
                for bh in self.main[start:start + 1000]:
                    block = self.block(bh)
                    if block is not None:
                        rows.append(self._index_rows(bh, block))
                store.add_index(rows)
            store.set_meta("txindex", 1)

    # -- queries --

    @property
    def height(self) -> int:
        return len(self.main) - 1

    @property
    def state(self) -> State:
        return self.states[self.tip]

    def tip_header(self) -> dict:
        return self.blocks[self.tip]["header"]

    def block(self, bh: str) -> dict | None:
        """Any known block, recent (memory) or old (store). None for blocks below a
        snapshot this node started from: it has their headers but not their bodies."""
        b = self.blocks.get(bh)
        if b is None and self.store is not None and isinstance(bh, str):
            b = self.store.get_block(bh)
        if b is None and bh == self.genesis_hash:
            b = self.genesis
        return b if b is None or b["txs"] is not None else None

    def header(self, bh: str) -> dict:
        """A block header; older ones come from the store, so lookbacks never depend on
        how many blocks stay in memory."""
        hdr = self.header_or_none(bh)
        if hdr is None:
            raise KeyError(f"unknown block {bh}")
        return hdr

    def header_or_none(self, bh: str) -> dict | None:
        b = self.blocks.get(bh)
        if b is not None:
            return b["header"]
        if bh == self.genesis_hash:
            return self.genesis["header"]
        if self.store is not None and isinstance(bh, str):
            return self.store.get_header(bh)
        return None

    def chainwork(self, bh: str) -> int | None:
        m = self.meta.get(bh)
        if m is not None:
            return m["chainwork"]
        return self.store.get_chainwork(bh) if self.store is not None else None

    def main_height(self, bh: str) -> int | None:
        """The height of `bh` if it is on the main chain, else None (any depth)."""
        hdr = self.header_or_none(bh)
        if hdr is None or not isinstance(hdr.get("height"), int):
            return None
        n = hdr["height"]
        return n if n < len(self.main) and self.main[n] == bh else None

    def has_block(self, bh: str) -> bool:
        return bh in self.blocks or bh == self.genesis_hash or (self.store is not None and self.store.has_block(bh))

    def block_at(self, height: int) -> dict | None:
        if 0 <= height < len(self.main):
            return self.block(self.main[height])
        return None

    def on_main(self, bh: str) -> bool:
        m = self.meta.get(bh)
        return m is not None and m["height"] < len(self.main) and self.main[m["height"]] == bh

    def branch(self, bh: str) -> list[str]:
        """Blocks from `bh` back to (not including) the main chain, newest first."""
        out = []
        while not self.on_main(bh):
            out.append(bh)
            bh = self.blocks[bh]["header"]["prev"]
        return out

    def ancestor_at(self, bh: str, height: int) -> str | None:
        """The block at `height` on the branch ending at `bh` (None if not known here)."""
        while bh in self.meta and self.meta[bh]["height"] > height:
            if self.on_main(bh):
                return self.main[height] if height < len(self.main) else None
            bh = self.blocks[bh]["header"]["prev"] if bh in self.blocks else None
        return bh if bh in self.meta and self.meta[bh]["height"] == height else None

    # -- checkpoints --

    def accept_checkpoint(self, cp: dict) -> bool:
        """Take a signed checkpoint (or the end of checkpoints); True if it is new. Raises
        InvalidCheckpoint for a forgery or one this network cannot take. A new checkpoint
        moves the tip onto its branch if the main chain left it out (a heavier chain an
        attacker mined, or one built before the checkpoint was heard)."""
        if self.authority is None:
            raise InvalidCheckpoint("this network has no checkpoints")
        key, until = self.authority
        if not isinstance(cp, dict):
            raise InvalidCheckpoint("malformed checkpoint")
        if cp.get("end") is True:
            if not (isinstance(cp.get("sig"), str) and verify(key, checkpoint_end_bytes(self.chain_id), cp["sig"])):
                raise InvalidCheckpoint("bad signature")
            if self.checkpoint_end is not None:
                return False
            self.checkpoint_end = {"end": True, "sig": cp["sig"]}
            if self.store is not None:
                self.store.set_meta("checkpoint_end", self.checkpoint_end)
            return True
        h, b, sig = cp.get("height"), cp.get("block"), cp.get("sig")
        if not (_is_int(h) and h > 0 and is_hex(b, 64) and isinstance(sig, str)):
            raise InvalidCheckpoint("malformed checkpoint")
        if not verify(key, checkpoint_bytes(self.chain_id, h, b), sig):
            raise InvalidCheckpoint("bad signature")
        if self.checkpoint_end is not None or h > until:
            return False                             # the checkpoint period is over
        cur = self.checkpoint
        if cur is not None and h <= cur["height"]:
            return False
        if cur is not None and b in self.meta and self.ancestor_at(b, cur["height"]) not in (None, cur["block"]):
            raise InvalidCheckpoint("does not extend the previous checkpoint")
        self.checkpoint = {"height": h, "block": b, "sig": sig}
        if self.store is not None:
            self.store.set_meta("checkpoint", self.checkpoint)
        self._follow_checkpoint()
        return True

    def _follow_checkpoint(self) -> None:
        """Put the tip on the checkpoint's branch: the most work block that descends from it."""
        cp = self.checkpoint
        h, b = cp["height"], cp["block"]
        if b not in self.meta or self.on_main(b):
            return
        best = max((k for k in self.states if self.meta[k]["height"] >= h and self.ancestor_at(k, h) == b),
                   key=lambda k: self.meta[k]["chainwork"], default=None)
        if best is None:
            return                                   # its branch left the reorg window here: resync
        if self.store is None:
            self._set_tip(best)
            return
        with self.store.transaction():
            self._set_tip(best)

    def checkpoint_status(self) -> dict:
        key, until = self.authority if self.authority else (None, None)
        return {"authority": key, "until": until, "checkpoint": self.checkpoint,
                "ended": self.checkpoint_end is not None, "end": self.checkpoint_end}

    def median_time_past(self, block_hash_: str, lookup=None) -> int:
        get = lookup or self.header
        times = []
        cur = get(block_hash_)
        while len(times) < params.MTP_WINDOW:
            times.append(cur["time"])
            if cur["height"] == 0:
                break
            cur = get(cur["prev"])
        return sorted(times)[len(times) // 2]

    # -- difficulty --

    def expected_target(self, prev_hash: str, lookup=None) -> int:
        """ASERT with the genesis block as anchor (its virtual parent one block
        interval before it), so block 1 has the genesis target."""
        prev = (lookup or self.header)(prev_hash)
        anchor_time = self.genesis["header"]["time"] - params.TARGET_BLOCK_SECONDS
        return asert(self.initial_target, anchor_time, 0, prev["time"], prev["height"], params.ASERT_HALFLIFE)

    # -- validation --

    @staticmethod
    def _check_header_format(hdr: dict) -> None:
        if not isinstance(hdr, dict) or set(hdr) != HEADER_FIELDS:
            raise InvalidBlock("header fields are wrong")
        if not all(_is_int(hdr[k]) for k in ("version", "height", "time", "nonce")):
            raise InvalidBlock("bad integer field")
        if not versionbits.valid_version(hdr["version"]):
            raise InvalidBlock("bad version: must be 0x20000000 plus signal bits")
        for k in ("prev", "tx_root", "shares_root", "state_root", "transcript", "target"):
            if not is_hex(hdr[k], 64):
                raise InvalidBlock(f"bad {k}")
        if not valid_miner(hdr["miner"]):
            raise InvalidBlock("bad miner: an address, or 2 to MAX_PAYEES [address, weight] pairs sorted by address")
        if not isinstance(hdr["msg"], str) or len(hdr["msg"]) > MAX_MSG_LEN:
            raise InvalidBlock("bad message")
        w = hdr["work"]
        if not isinstance(w, dict) or set(w) != {"job", "unit"}:
            raise InvalidBlock("bad work field")
        if w["job"] is None:
            if w["unit"] is not None:
                raise InvalidBlock("filler work has no unit")
        else:
            if not is_hex(w["job"], 64):
                raise InvalidBlock("bad work job")
            u = w["unit"]
            if not (isinstance(u, list) and len(u) == 3 and all(_is_int(x) for x in u)):
                raise InvalidBlock("bad work unit")

    def _check_activated_rules(self, block: dict) -> None:
        """Rules added by deployments (green/versionbits.py), enforced once ACTIVE.

        `testdummy` is the example every test network can schedule (like Bitcoin's
        regtest deployment of the same name); no real network schedules it. It
        tightens the header message limit from 200 to 100 characters, a soft fork:
        nodes that never heard of it still accept every block that follows it.
        """
        prev = block["header"]["prev"]
        if self.deployments.active("testdummy", prev) and len(block["header"]["msg"]) > 100:
            raise InvalidBlock("block message longer than 100 characters (testdummy rule)")

    def _check_ticket(self, hdr: dict, parent_state: State | None) -> None:
        seed = seed_of(hdr)
        job_size = 0
        if hdr["work"]["job"] is not None:
            job_size = txmod.job_blob_size(parent_state.jobs[hdr["work"]["job"]]["dims"])
        a, b = work_inputs(hdr, seed, self.blobs, job_size)
        transcript, _ = work.compute_unit(a, b, seed, product=False)
        if transcript != hdr["transcript"]:
            raise InvalidBlock("transcript does not match the work")
        if work.ticket_value(seed, transcript) > target_of(hdr):
            raise InvalidBlock("ticket does not meet the target")

    def _check_genesis(self, genesis: dict) -> None:
        if not isinstance(genesis, dict) or set(genesis) != {"header", "txs"}:
            raise InvalidBlock("bad genesis")
        hdr = genesis["header"]
        self._check_header_format(hdr)
        if hdr["height"] != 0 or hdr["prev"] != ZERO_HASH or genesis["txs"]:
            raise InvalidBlock("bad genesis")
        if not 0 < target_of(hdr) <= params.INITIAL_TARGET or hdr["work"]["job"] is not None:
            raise InvalidBlock("bad genesis")
        self._check_ticket(hdr, None)

    def check_header(self, block: dict, trusted: bool = False) -> None:
        self.validate_header(block["header"], trusted=trusted)

    def validate_header(self, hdr: dict, lookup=None, trusted: bool = False) -> None:
        """Every header rule except the work unit, which costs a matmul.

        `lookup` finds ancestors that are not blocks of this chain yet (the
        headers-first sync index); by default only the chain's own blocks.
        """
        self._check_header_format(hdr)
        prev = (lookup or self.header)(hdr["prev"])
        if hdr["height"] != prev["height"] + 1:
            raise InvalidBlock("wrong height")
        if hdr["time"] <= self.median_time_past(hdr["prev"], lookup):
            raise InvalidBlock("timestamp not after the median of recent blocks")
        if not trusted and hdr["time"] > time.time() + params.MAX_FUTURE_SECONDS:
            raise InvalidBlock("timestamp too far in the future", permanent=False)
        if target_of(hdr) != self.expected_target(hdr["prev"], lookup):
            raise InvalidBlock("wrong difficulty target")
        cp = self.checkpoint
        if cp is not None and hdr["height"] == cp["height"] and hhex(canonical(hdr)) != cp["block"]:
            raise InvalidBlock("conflicts with the checkpoint", permanent=False)

    def work_transcript(self, hdr: dict) -> str:
        """Recompute a header's work unit without any ledger state (thread safe).

        Used to check work ahead of connecting blocks: the job's data is found
        by its hash, so this needs no state, only the blob. Raises MissingData
        if the job's data is not here and InvalidBlock if the unit is outside
        the job.
        """
        seed = seed_of(hdr)
        a, b = work_inputs(hdr, seed, self.blobs, MAX_JOB_BLOB)
        return work.compute_unit(a, b, seed, product=False)[0]

    def share_problem(self, share, parent_hash: str, seen=()) -> str | None:
        """Why a block built on `parent_hash` may not include `share` (None if it may),
        apart from the share's work unit (check_share_work). A share is a header mined
        on one of the last SHARE_WINDOW blocks of that branch, not itself a block of it,
        with the target its parent called for, not included before (`seen`)."""
        try:
            self._check_header_format(share)
        except InvalidBlock as e:
            return str(e)
        sid = share_id(share)
        if sid in seen:
            return "share already included"
        parent = self.header_or_none(parent_hash)
        if parent is None:
            return "unknown parent"
        height = parent["height"] + 1
        if not height - params.SHARE_WINDOW <= share["height"] <= height:
            return "share outside the window"
        h, cur = parent["height"], parent_hash
        while h > share["height"] - 1:               # the branch's block at the share's height...
            if h == share["height"] and cur == sid:
                return "a block of this branch is not a share"
            cur = self.header(cur)["prev"]
            h -= 1
        if cur != share["prev"]:                     # ...and below it, the block the share was mined on
            return "share not mined on this branch"
        if target_of(share) != self.expected_target(share["prev"]):
            return "share has the wrong target"
        return None

    def share_target(self, share: dict) -> int:
        return min(params.MAX_TARGET, target_of(share) * params.SHARE_FACTOR)

    def check_share_work(self, share: dict, job_size: int = MAX_JOB_BLOB) -> None:
        """Recompute a share's work unit; InvalidBlock if it is not real work meeting the share target."""
        sid = share_id(share)
        if sid in self.verified_shares:
            return
        seed = seed_of(share)
        a, b = work_inputs(share, seed, self.blobs, job_size)
        transcript, _ = work.compute_unit(a, b, seed, product=False)
        if transcript != share["transcript"]:
            raise InvalidBlock("share transcript does not match its work")
        if work.ticket_value(seed, transcript) > self.share_target(share):
            raise InvalidBlock("share ticket does not meet the share target")
        self.share_verified(sid)

    def share_verified(self, sid: str) -> None:
        """Remember that a share's work unit checked out (a bounded cache)."""
        if len(self.verified_shares) > MAX_CACHE:
            self.verified_shares.clear()
        self.verified_shares.add(sid)

    def work_ok(self, hdr: dict, transcript: str) -> bool:
        return transcript == hdr["transcript"] and work.ticket_value(seed_of(hdr), transcript) <= target_of(hdr)

    def apply_tx(self, s: State, tx: dict, height: int) -> None:
        """Validate `tx` for a block at `height` and apply it to `s`.

        All checks run before any change, so `s` is untouched on error. The
        expensive or data-dependent checks (job data, Freivalds) run last, so
        a node never downloads data for a transaction that fails cheap checks.
        """
        txmod.check_format(tx, self.chain_id)
        addr = txmod.sender(tx)
        if tx["nonce"] != s.nonce(addr):
            raise (txmod.FutureNonce if tx["nonce"] > s.nonce(addr) else InvalidTx)(f"nonce must be {s.nonce(addr)}")
        fee = tx["fee"]
        if fee < txmod.min_fee(tx):
            raise InvalidTx(f"fee must be at least {txmod.min_fee(tx)}")
        bal = s.balance(addr)
        kind = tx["type"]
        # A guarded account answers to its guardian: frozen means no transaction at
        # all, otherwise spending is capped per period. Checked before any change.
        guard = s.guards.get(addr)
        if guard is not None:
            if guard["frozen"]:
                raise InvalidTx("account is frozen by its guardian")
            window = height // guard["period"]
            spent = (guard["spent"] if guard["window"] == window else 0) + txmod.outflow(tx)
            if spent > guard["limit"]:
                raise InvalidTx(f"over the guardian's spending limit ({guard['limit']} per {guard['period']} blocks)")

        if kind == "transfer":
            if bal < tx["amount"] + fee:
                raise InvalidTx("insufficient balance")
            if tx["amount"] < params.NEW_ACCOUNT_MIN and s.balance(tx["to"]) == 0 and s.nonce(tx["to"]) == 0:
                raise InvalidTx(f"a transfer that opens an account sends at least {params.NEW_ACCOUNT_MIN}")
            s.set_balance(addr, bal - tx["amount"] - fee)
            s.credit(tx["to"], tx["amount"])

        elif kind == "channel_open":
            if tx["provider"] == addr:
                raise InvalidTx("a channel is opened to someone else")
            if not height + params.MIN_CHANNEL_BLOCKS <= tx["expires"] <= height + params.MAX_CHANNEL_BLOCKS:
                raise InvalidTx(f"expires must be {params.MIN_CHANNEL_BLOCKS} to {params.MAX_CHANNEL_BLOCKS} "
                                f"blocks ahead")
            if bal < tx["amount"] + fee:
                raise InvalidTx("insufficient balance")
            cid = txmod.txid(tx)
            s.set_balance(addr, bal - tx["amount"] - fee)
            s.channels[cid] = {"buyer": addr, "pub": tx["pub"], "provider": tx["provider"],
                               "amount": tx["amount"], "paid": 0, "expires": tx["expires"]}
            s.channel_expiry.setdefault(tx["expires"] + 1, []).append(cid)

        elif kind == "channel_claim":
            ch = s.channels.get(tx["channel"])
            if ch is None:
                raise InvalidTx("unknown or ended channel")
            if ch["provider"] != addr:
                raise InvalidTx("only the channel's provider can claim")
            if height > ch["expires"]:
                raise InvalidTx("the channel has expired")
            if not ch["paid"] < tx["total"] <= ch["amount"]:
                raise InvalidTx("total must exceed what was claimed and not the channel's amount")
            pay = tx["total"] - ch["paid"]
            if pay < fee:
                raise InvalidTx("the claim does not cover its fee")
            if not verify(ch["pub"], txmod.voucher_bytes(self.chain_id, tx["channel"], tx["total"]), tx["voucher"]):
                raise InvalidTx("voucher not signed by the channel's buyer")
            # The fee comes out of the claim, so a new provider needs no balance.
            ch["paid"] = tx["total"]
            for payee, amount in payouts(tx.get("payees", addr), pay - fee):
                s.credit(payee, amount)
            if tx.get("close"):
                # The provider is done: the rest goes back to the buyer now, not at expiry.
                s.credit(ch["buyer"], ch["amount"] - tx["total"])
                del s.channels[tx["channel"]]

        elif kind == "job_post":
            if tx["job"] in s.jobs:
                raise InvalidTx("job already posted")
            mu, ku, nu = tx["dims"]
            escrow = tx["price"] * mu * ku * nu
            if bal < escrow + fee:
                raise InvalidTx("insufficient balance for escrow")
            a, b = self.blobs.job(tx["job"], txmod.job_blob_size(tx["dims"]))
            if txmod.job_dims_units(a, b) != tx["dims"]:
                raise InvalidTx("dims do not match the job data")
            s.set_balance(addr, bal - escrow - fee)
            s.jobs[tx["job"]] = {"buyer": addr, "dims": list(tx["dims"]), "price": tx["price"],
                                 "escrow": escrow, "done": {}, "delivered": {}, "reserved": {},
                                 "batches": {}, "closes_at": None, "closed": False}

        elif kind == "job_commit":
            job = s.jobs.get(tx["job"])
            if job is None or job["closed"]:
                raise InvalidTx("job is not open")
            if job["closes_at"] is not None:
                raise InvalidTx("job is being cancelled")
            units = txmod.bitmap_units(tx["units"], job["dims"])
            if units is None:
                raise InvalidTx("units bitmap does not fit the job")
            for u in units:
                key = txmod.unit_key(u)
                if key in job["done"]:
                    raise InvalidTx("unit already delivered")
                if key in job["reserved"]:
                    raise InvalidTx("unit is reserved by another commitment")
            bid = tx["commitment"]
            if bid in job["batches"] or bid in job["delivered"]:
                raise InvalidTx("batch already committed")
            bond = params.bond_for(job["price"]) * len(units)
            if bal < bond + fee:
                raise InvalidTx("insufficient balance for bond")
            s.set_balance(addr, bal - bond - fee)
            expires = height + params.RESERVE_BLOCKS
            job["batches"][bid] = {"miner": addr, "units": tx["units"], "count": len(units),
                                   "height": height, "expires": expires, "bond": bond,
                                   "status": "committed", "root": None, "manifest": None,
                                   "settle_at": None, "disputes": {}, "proven": set()}
            if "payees" in tx:                       # the payment is shared; the bond stays the sender's
                job["batches"][bid]["payees"] = tx["payees"]
            for u in units:
                job["reserved"][txmod.unit_key(u)] = bid
            s.schedule(expires + 1, "expire", tx["job"], bid)

        elif kind == "job_reveal":
            job, batch = self._batch(s, tx)
            if batch["status"] != "committed" or batch["miner"] != addr:
                raise InvalidTx("no commitment from this miner for the batch")
            if batch["height"] >= height:
                raise InvalidTx("reveal must come in a later block than the commitment")
            if height > batch["expires"]:
                raise InvalidTx("commitment expired")
            if bal < fee:
                raise InvalidTx("insufficient balance for fee")
            if txmod.commitment(addr, tx["job"], batch["units"], tx["root"]) != tx["batch"]:
                raise InvalidTx("root does not match the commitment")
            # No data is fetched: validators only learn the root. The manifest is a
            # hint for the buyer; a wrong or missing one just gets the units disputed.
            s.set_balance(addr, bal - fee)
            settle_at = height + params.JOB_CHALLENGE_BLOCKS
            batch.update(status="revealed", root=tx["root"], manifest=tx["manifest"], settle_at=settle_at)
            s.schedule(settle_at, "settle", tx["job"], tx["batch"])

        elif kind == "job_challenge":
            job, batch = self._batch(s, tx)
            if batch["status"] != "revealed":
                raise InvalidTx("batch has no revealed results to dispute")
            if height > batch["settle_at"]:
                raise InvalidTx("challenge window has closed")
            if len(tx["units"]) != len(batch["units"]) or int(tx["units"], 16) & ~int(batch["units"], 16):
                raise InvalidTx("disputed units are not in the batch")
            keys = [txmod.unit_key(u) for u in txmod.bitmap_units(tx["units"], job["dims"])]
            if any(k in batch["disputes"] or k in batch["proven"] for k in keys):
                raise InvalidTx("unit is already disputed or proven")
            if bal < fee:
                raise InvalidTx("insufficient balance for fee")
            s.set_balance(addr, bal - fee)
            # Answers are capped per block, so the deadline makes room for every open
            # dispute (this batch's included) to be answered before it.
            s.open_disputes += len(keys)
            deadline = height + params.JOB_RESPONSE_BLOCKS + s.open_disputes // params.MAX_BLOCK_JOB_RESPONSES
            for k in keys:
                batch["disputes"][k] = {"by": addr, "deadline": deadline}
            s.schedule(deadline + 1, "deadline", tx["job"], tx["batch"])

        elif kind == "job_respond":
            job, batch = self._batch(s, tx)
            key = txmod.unit_key(tx["unit"])
            d = batch["disputes"].get(key)
            if batch["status"] != "revealed" or d is None:
                raise InvalidTx("unit has no open dispute")
            if height > d["deadline"]:
                raise InvalidTx("response deadline has passed")
            if bal < fee:
                raise InvalidTx("insufficient balance for fee")
            units = txmod.bitmap_units(batch["units"], job["dims"])
            index = units.index(list(tx["unit"]))
            if not merkle_verify(tx["result"], index, batch["count"], tx["proof"], batch["root"]):
                raise InvalidTx("result is not under the batch's Merkle root")
            if s.responses_used >= params.MAX_BLOCK_JOB_RESPONSES:
                raise InvalidTx("block verification budget exhausted, try the next block")
            # Every validator fetches and checks the disputed unit; a wrong or missing
            # result cannot be answered, so the batch is forfeit at the deadline.
            self._verify_result(tx["job"], job["dims"], tx["unit"], tx["result"])
            s.responses_used += 1
            s.set_balance(addr, bal - fee)
            del batch["disputes"][key]
            batch["proven"].add(key)
            s.open_disputes -= 1
            if not batch["disputes"] and height >= batch["settle_at"]:     # window closed while disputed
                self._pay_batch(s, job, tx["batch"])

        elif kind == "job_cancel":
            job = s.jobs.get(tx["job"])
            if job is None or job["buyer"] != addr:
                raise InvalidTx("only the buyer can cancel a job")
            if job["closed"] or job["closes_at"] is not None:
                raise InvalidTx("job already cancelled")
            if bal < fee:
                raise InvalidTx("insufficient balance for fee")
            s.set_balance(addr, bal - fee)
            job["closes_at"] = height + params.CANCEL_DELAY
            s.closing.setdefault(job["closes_at"], []).append(tx["job"])

        elif kind == "guard_set":
            if tx["guardian"] == addr:
                raise InvalidTx("an account cannot be its own guardian")
            if bal < fee:
                raise InvalidTx("insufficient balance for fee")
            if guard is not None and (tx["guardian"] != guard["guardian"] or tx["limit"] > guard["limit"]
                                      or tx["period"] < guard["period"]):
                raise InvalidTx("only the guardian can loosen, hand over or remove a guard")
            s.set_balance(addr, bal - fee)
            # A new guard starts empty; a tightened one keeps what was already spent.
            s.guards[addr] = {"guardian": tx["guardian"], "limit": tx["limit"], "period": tx["period"],
                              "frozen": False, "window": height // tx["period"],
                              "spent": spent if guard is not None else 0}

        elif kind in ("guard_update", "guard_release"):
            target = s.guards.get(tx["account"])
            if target is None or target["guardian"] != addr:
                raise InvalidTx("only the account's guardian can change its guard")
            if kind == "guard_update" and tx["guardian"] == tx["account"]:
                raise InvalidTx("an account cannot be its own guardian")
            if bal < fee:
                raise InvalidTx("insufficient balance for fee")
            s.set_balance(addr, bal - fee)
            if kind == "guard_release":
                del s.guards[tx["account"]]
            else:
                target.update(guardian=tx["guardian"], limit=tx["limit"], period=tx["period"], frozen=tx["frozen"])

        elif kind == "task_post":
            task_id = txmod.task_id_of(tx)
            if task_id in s.tasks:
                raise InvalidTx("task already posted")
            if bal < tx["price"] + fee:
                raise InvalidTx("insufficient balance for task escrow")
            # Every node keeps the program and input, so a fraud can always be
            # proven later even if the buyer goes offline.
            prog = self.blobs.get(tx["prog"])
            if prog is None:
                raise MissingData(tx["prog"], params.MAX_TASK_PROGRAM_BYTES)
            inp = self.blobs.get(tx["input"])
            if inp is None:
                raise MissingData(tx["input"], params.MAX_TASK_INPUT_BYTES)
            if len(inp) > params.MAX_TASK_INPUT_BYTES:
                raise InvalidTx(f"input larger than {params.MAX_TASK_INPUT_BYTES} bytes")
            if fee < txmod.task_post_fee(len(prog), len(inp)):
                raise InvalidTx(f"fee must be at least {txmod.task_post_fee(len(prog), len(inp))}")
            self._validate_program(tx["prog"], prog)
            s.set_balance(addr, bal - tx["price"] - fee)
            s.tasks[task_id] = {"buyer": addr, "prog": tx["prog"], "input": tx["input"],
                                "prog_size": len(prog), "fuel": tx["fuel"], "price": tx["price"],
                                "escrow": tx["price"], "status": "open", "worker": None,
                                "output": None, "reserved": None, "settle_at": None}

        elif kind == "task_commit":
            task = s.tasks.get(tx["task"])
            if task is None or task["status"] != "open":
                raise InvalidTx("task is not open for work")
            r = task["reserved"]
            if r is not None and height <= r["expires"]:
                raise InvalidTx("task is reserved by another worker")
            bond = params.task_bond(task["price"])
            if bal < bond + fee:
                raise InvalidTx("insufficient balance for bond")
            if r is not None:
                s.credit(task["buyer"], r["bond"])          # previous worker never revealed
            s.set_balance(addr, s.balance(addr) - bond - fee)
            task["reserved"] = {"worker": addr, "commitment": tx["commitment"], "height": height,
                                "expires": height + params.TASK_RESERVE_BLOCKS, "bond": bond}

        elif kind == "task_reveal":
            task = s.tasks.get(tx["task"])
            if task is None or task["status"] != "open":
                raise InvalidTx("task is not open")
            r = task["reserved"]
            if r is None or r["worker"] != addr:
                raise InvalidTx("no commitment from this worker")
            if r["height"] >= height:
                raise InvalidTx("reveal must come in a later block than the commit")
            if height > r["expires"]:
                raise InvalidTx("commitment expired")
            if txmod.task_commitment(addr, tx["task"], tx["output"]) != r["commitment"]:
                raise InvalidTx("output does not match the commitment")
            if bal < fee:
                raise InvalidTx("insufficient balance for fee")
            s.set_balance(addr, bal - fee)
            task["status"] = "revealed"
            task["worker"] = addr
            task["output"] = tx["output"]
            task["settle_at"] = height + params.TASK_CHALLENGE_BLOCKS
            s.task_settle.setdefault(task["settle_at"], []).append(tx["task"])

        elif kind == "task_challenge":
            task = s.tasks.get(tx["task"])
            if task is None or task["status"] != "revealed":
                # "confirmed" results already survived a challenge: one re-execution per
                # task, otherwise anyone could make every validator re-run it forever.
                raise InvalidTx("task has no unconfirmed result to challenge")
            if height > task["settle_at"]:
                raise InvalidTx("challenge window has closed")
            cost = params.adjudication_cost(task["fuel"], task["prog_size"])
            if fee < params.challenge_fee(task["fuel"], task["prog_size"]):
                raise InvalidTx(f"fee must be at least {params.challenge_fee(task['fuel'], task['prog_size'])}")
            cbond = params.task_bond(task["price"])
            if bal < cbond + fee:
                raise InvalidTx("insufficient balance for challenge bond")
            if s.adjudication_used + cost > params.MAX_BLOCK_ADJUDICATION:
                raise InvalidTx("block adjudication budget exhausted, try the next block")
            honest = self._adjudicate(task)     # may raise MissingData: nothing is changed yet
            s.adjudication_used += cost
            s.set_balance(addr, bal - cbond - fee)
            wbond = task["reserved"]["bond"]
            if honest:
                # Wrong challenge: the challenger's bond goes to the worker, and the
                # result is now final.
                s.credit(task["worker"], cbond)
                task["status"] = "confirmed"
            else:
                # Proven fraud: worker's bond goes to the challenger, the escrow back to
                # the buyer; the task is abandoned so the buyer can repost.
                s.credit(addr, cbond + wbond)
                s.credit(task["buyer"], task["escrow"])
                task["escrow"] = 0
                task["reserved"] = None
                task["status"] = "failed"
                task["settle_at"] = None

        if guard is not None and addr in s.guards and kind != "guard_set":
            s.guards[addr].update(window=window, spent=spent)
        s.nonces[addr] = tx["nonce"] + 1

    @staticmethod
    def _batch(s: State, tx: dict):
        job = s.jobs.get(tx["job"])
        batch = job["batches"].get(tx["batch"]) if job else None
        if batch is None:
            raise InvalidTx("unknown batch")
        return job, batch

    @staticmethod
    def _release(s: State, job: dict, batch_id: str) -> dict:
        batch = job["batches"].pop(batch_id)
        for u in txmod.bitmap_units(batch["units"], job["dims"]):
            if job["reserved"].get(txmod.unit_key(u)) == batch_id:
                del job["reserved"][txmod.unit_key(u)]
        s.open_disputes -= len(batch["disputes"])
        return batch

    def _pay_batch(self, s: State, job: dict, batch_id: str) -> None:
        batch = self._release(s, job, batch_id)
        pay = job["price"] * batch["count"]
        job["escrow"] -= pay
        for addr, amount in payouts(batch.get("payees", batch["miner"]), pay):
            s.credit(addr, amount)
        s.credit(batch["miner"], batch["bond"])
        for u in txmod.bitmap_units(batch["units"], job["dims"]):
            job["done"][txmod.unit_key(u)] = batch_id
        job["delivered"][batch_id] = {"miner": batch["miner"], "units": batch["units"], "count": batch["count"],
                                      "root": batch["root"], "manifest": batch["manifest"]}
        if "payees" in batch:
            job["delivered"][batch_id]["payees"] = batch["payees"]

    def _drop_batch(self, s: State, job: dict, batch_id: str) -> None:
        """A batch that will not be paid. Its units reopen, or if the job is
        closed, their share of the escrow goes back to the buyer."""
        batch = self._release(s, job, batch_id)
        if job["closed"]:
            refund = job["price"] * batch["count"]
            job["escrow"] -= refund
            s.credit(job["buyer"], refund)

    def _job_events(self, s: State, height: int) -> None:
        """Batch deadlines that fall at `height`. Every event re-checks that the
        batch is still in the state it was scheduled for, so stale events are no-ops."""
        for kind, job_id, batch_id, _ in s.job_events.pop(height, []):
            job = s.jobs[job_id]
            batch = job["batches"].get(batch_id)
            if batch is None:
                continue
            if kind == "expire" and batch["status"] == "committed" and batch["expires"] == height - 1:
                # Never revealed: the bond compensates the buyer for the units it held up.
                s.credit(job["buyer"], batch["bond"])
                self._drop_batch(s, job, batch_id)
            elif (kind == "settle" and batch["status"] == "revealed" and batch["settle_at"] == height
                  and not batch["disputes"]):
                self._pay_batch(s, job, batch_id)
            elif kind == "deadline":
                late = [d for d in batch["disputes"].values() if d["deadline"] == height - 1]
                if late:
                    # A disputed unit went unanswered: the miner could not show a correct,
                    # available result, so the whole batch is forfeit to the challenger.
                    s.credit(late[0]["by"], batch["bond"])
                    self._drop_batch(s, job, batch_id)

    def _verify_result(self, job_id: str, dims, unit, result_hash: str) -> None:
        cache_key = (job_id, txmod.unit_key(unit), result_hash)
        if cache_key in self.verified_results:
            return
        blob = self.blobs.get(result_hash)
        if blob is None:
            raise MissingData(result_hash, txmod.RESULT_BYTES)
        try:
            c = txmod.decode_result(blob)
        except ValueError as e:
            raise InvalidTx(str(e)) from None
        a, b = self.blobs.job(job_id, txmod.job_blob_size(dims))
        ua, ub = txmod.unit_inputs(a, b, unit)
        seed = h(canonical(["freivalds", job_id, list(unit), result_hash]))
        if not work.freivalds_check(ua, ub, c, seed):
            raise InvalidTx("result is wrong (Freivalds check failed)")
        if len(self.verified_results) > MAX_CACHE:
            self.verified_results.clear()
        self.verified_results.add(cache_key)

    def _adjudicate(self, task: dict) -> bool:
        """Re-run a challenged task and return whether the worker's output is right.

        Deterministic, so every node reaches the same verdict. Needs the program
        and input blobs; the fuel budget (checked at post time) bounds the cost.
        """
        from . import wasm
        key = (task["prog"], task["input"], task["output"], task["fuel"])
        if key in self.verdicts:          # deterministic, so the verdict can be reused
            return self.verdicts[key]
        prog = self.blobs.get(task["prog"])
        input_blob = self.blobs.get(task["input"])
        if prog is None:
            raise MissingData(task["prog"], params.MAX_TASK_PROGRAM_BYTES)
        if input_blob is None:
            raise MissingData(task["input"], params.MAX_TASK_INPUT_BYTES)
        verdict = wasm.verify(prog, input_blob, task["output"], task["fuel"])
        if len(self.verdicts) > MAX_CACHE:
            self.verdicts.clear()
        self.verdicts[key] = verdict
        return verdict

    def _validate_program(self, prog_hash: str, prog: bytes) -> None:
        if prog_hash in self.valid_programs:
            return
        from . import wasm
        try:
            wasm.validate_module(prog)
        except wasm.ComputeError as e:
            raise InvalidTx(f"program rejected: {e}") from None
        if len(self.valid_programs) > MAX_CACHE:
            self.valid_programs.clear()
        self.valid_programs.add(prog_hash)

    @staticmethod
    def _end_channels(s: State, height: int) -> None:
        """Channels past their expiry: what the provider did not claim goes back to the buyer."""
        for cid in s.channel_expiry.pop(height, []):
            ch = s.channels.pop(cid, None)
            if ch is not None:
                s.credit(ch["buyer"], ch["amount"] - ch["paid"])

    def _settle_tasks(self, s: State, height: int) -> None:
        """Pay the worker for a revealed task that survived its challenge window."""
        for task_id in s.task_settle.pop(height, []):
            task = s.tasks[task_id]
            if task["status"] not in ("revealed", "confirmed") or task["settle_at"] != height:
                continue
            r = task["reserved"]
            s.credit(task["worker"], task["price"] + r["bond"])
            task["escrow"] -= task["price"]
            task["reserved"] = None
            task["status"] = "done"

    @staticmethod
    def _close_jobs(s: State, height: int) -> None:
        """Finish cancellations that reach their delay at `height`.

        Live batches keep their share of the escrow and settle as usual; only
        the rest comes back to the buyer now.
        """
        for job_id in s.closing.pop(height, []):
            job = s.jobs[job_id]
            live = sum(b["count"] for b in job["batches"].values())
            refund = job["escrow"] - job["price"] * live
            s.credit(job["buyer"], refund)
            job["escrow"] -= refund
            job["closed"] = True

    def apply_block(self, state: State, block: dict) -> State:
        """The state after `block`, checked against the state root its header commits to."""
        hdr = block["header"]
        txs = block["txs"]
        if not isinstance(txs, list):
            raise InvalidBlock("bad transaction list", permanent=False)
        try:
            ids = [txmod.txid(t) for t in txs]
        except (TypeError, ValueError):
            raise InvalidBlock("unencodable transaction", permanent=False) from None
        if merkle_root(ids) != hdr["tx_root"]:
            # The body is not the one the header commits to: blame the sender, not the header.
            raise InvalidBlock("tx_root mismatch", permanent=False)
        if len(txs) > params.MAX_BLOCK_TXS:
            raise InvalidBlock("too many transactions")
        if len(set(ids)) != len(ids):
            raise InvalidBlock("duplicate transaction")
        if len(canonical(txs)) > params.MAX_BLOCK_BYTES:
            raise InvalidBlock("block too large")
        try:
            sigs = sum(txmod.sig_count(t) for t in txs)
        except (TypeError, KeyError, AttributeError):
            sigs = 0                                 # malformed: apply_tx rejects it below
        if sigs > params.MAX_BLOCK_SIGS:
            raise InvalidBlock("too many signatures in the block")
        job_id = hdr["work"]["job"]
        if job_id is not None and job_id not in state.jobs:
            raise InvalidBlock("work references a job that was never posted")
        shares = block.get("shares", [])
        if "shares" in block and (not isinstance(shares, list) or not shares):
            raise InvalidBlock("a block without shares leaves the shares field out", permanent=False)
        try:
            share_ids = [share_id(sh) for sh in shares]
        except (TypeError, ValueError):
            raise InvalidBlock("unencodable share", permanent=False) from None
        if merkle_root(share_ids) != hdr["shares_root"]:
            raise InvalidBlock("shares_root mismatch", permanent=False)
        if len(shares) > params.MAX_BLOCK_SHARES:
            raise InvalidBlock("too many shares")
        if len(set(share_ids)) != len(share_ids):
            raise InvalidBlock("duplicate share")
        for sh in shares:                            # cheap checks for all, then the work units
            problem = self.share_problem(sh, hdr["prev"], state.share_seen)
            if problem:
                raise InvalidBlock(problem)
            if sh["work"]["job"] is not None and sh["work"]["job"] not in state.jobs:
                raise InvalidBlock("share work references a job that was never posted")
        for sh in shares:
            job = sh["work"]["job"]
            self.check_share_work(sh, txmod.job_blob_size(state.jobs[job]["dims"]) if job else 0)
        height = hdr["height"]
        s = state.copy()
        for t in txs:
            try:
                self.apply_tx(s, t, height)
            except InvalidTx as e:
                raise InvalidBlock(f"invalid transaction {txmod.txid(t)[:12]}: {e}") from None
        if self._finish_block(s, hdr, txs, shares) != hdr["state_root"]:
            # tx_root matched, so this is the body the header commits to: the header is wrong.
            raise InvalidBlock("state root does not match the state after the block")
        return s

    def _finish_block(self, s: State, hdr: dict, txs: list[dict], shares: list = ()) -> str:
        """End-of-block steps after the transactions; returns the new state root."""
        height = hdr["height"]
        self._job_events(s, height)
        self._close_jobs(s, height)
        self._settle_tasks(s, height)
        self._end_channels(s, height)
        for sid in s.share_expiry.pop(height - params.SHARE_WINDOW - 1, []):
            s.share_seen.pop(sid, None)              # too old to be included again anyway
        for sh in shares:
            sid = share_id(sh)
            s.share_seen[sid] = height
            s.share_expiry.setdefault(height, []).append(sid)
        pay_block(s, hdr, sum(t["fee"] for t in txs), list(shares))
        return s.commit()

    def _state_of(self, block_hash_: str) -> State:
        if block_hash_ in self.states:
            return self.states[block_hash_]
        path = []
        cur = block_hash_
        while cur not in self.states:
            if cur not in self.blocks or self.meta[cur]["height"] <= self.base_height:
                raise InvalidBlock("fork branches off below the base state", permanent=False)
            path.append(cur)
            cur = self.blocks[cur]["header"]["prev"]
        s = self.states[cur]
        for bh in reversed(path):
            s = self.apply_block(s, self.blocks[bh])
        return s

    def add_block(self, block: dict, trusted: bool = False) -> str:
        """Returns "added", "exists" or "orphan". Raises InvalidBlock / MissingData."""
        if not isinstance(block, dict) or set(block) - {"shares"} != {"header", "txs"} \
                or not isinstance(block["header"], dict):
            raise InvalidBlock("bad block", permanent=False)
        try:
            bh = block_hash(block)
        except (TypeError, ValueError):
            raise InvalidBlock("unencodable header", permanent=False) from None
        if self.has_block(bh):
            return "exists"
        if bh in self.invalid:
            raise InvalidBlock("block is known to be invalid")
        prev_hash = block["header"].get("prev")
        if not isinstance(prev_hash, str) or prev_hash not in self.blocks:
            if isinstance(prev_hash, str) and self.has_block(prev_hash):
                raise InvalidBlock("fork branches off too deep", permanent=False)    # parent is long buried
            return "orphan"
        try:
            # Cheap checks first, the work-unit recomputation last.
            self.check_header(block, trusted=trusted)
            # The rule is about where the fork leaves the main chain, not how old the
            # parent is: a side chain started long ago must not sneak in block by block.
            fork = prev_hash
            while not self.on_main(fork):
                fork = self.blocks[fork]["header"]["prev"]
            if self.meta[fork]["height"] < self.height - params.MAX_REORG_DEPTH:
                raise InvalidBlock("fork branches off too deep", permanent=False)
            cp = self.checkpoint
            if cp is not None and self.meta[prev_hash]["height"] + 1 > cp["height"] \
                    and self.ancestor_at(prev_hash, cp["height"]) != cp["block"]:
                raise InvalidBlock("conflicts with the checkpoint", permanent=False)
            self._check_activated_rules(block)
            parent_state = self._state_of(prev_hash)
            job_id = block["header"]["work"]["job"]
            if job_id is not None and job_id not in parent_state.jobs:
                raise InvalidBlock("work references a job that was never posted")
            if not trusted and bh not in self.prechecked:
                self._check_ticket(block["header"], parent_state)
            new_state = self.apply_block(parent_state, block)
        except InvalidBlock as e:
            if e.permanent:
                if len(self.invalid) > MAX_CACHE:
                    self.invalid.clear()
                self.invalid.add(bh)
            raise

        self.prechecked.discard(bh)
        prev_meta = self.meta[prev_hash]
        height = prev_meta["height"] + 1
        self.blocks[bh] = block
        self.meta[bh] = {"height": height,
                         "chainwork": prev_meta["chainwork"] + work_for_target(target_of(block["header"]))}
        self.states[bh] = new_state
        if self.store is None:
            if self.meta[bh]["chainwork"] > self.meta[self.tip]["chainwork"]:
                self._set_tip(bh)
            return "added"
        with self.store.transaction():          # the block, its diff and any tip/base move, atomically
            self.store.put_block(bh, block, height, self.meta[bh]["chainwork"])
            self.store.put_diff(bh, height, new_state.diff())
            if self.meta[bh]["chainwork"] > self.meta[self.tip]["chainwork"]:
                self._set_tip(bh)
        return "added"

    def _set_tip(self, bh: str) -> None:
        """Make `bh` the tip. Only the changed end of the main chain is touched."""
        new = self.branch(bh)[::-1]
        fork_height = self.meta[new[0]]["height"] - 1
        del self.main[fork_height + 1:]
        self.main.extend(new)
        self.tip = bh
        if self.store is not None:
            self.store.set_main_from(fork_height + 1, new, [self._index_rows(h, self.blocks[h]) for h in new])
        self._advance_base()

    @staticmethod
    def _index_rows(bh: str, block: dict):
        """Rows for the transaction and address index (green/store.py) of one block.
        Every address a transaction names is indexed with its role; the miner gets a
        "reward" row pointing at the block."""
        hdr = block["header"]
        height = hdr["height"]
        tx_rows = []
        addr_rows = [(addr, height, -1, bh, "miner", "reward") for addr, _ in payouts(hdr["miner"], 0)]
        for pos, t in enumerate(block["txs"] or []):
            tid = txmod.txid(t)
            tx_rows.append((tid, height, bh, pos))
            named = {txmod.sender(t): "sender"}
            for field in ("to", "account", "guardian"):
                if isinstance(t.get(field), str) and t[field] not in named:
                    named[t[field]] = field if field != "to" else "recipient"
            addr_rows += [(addr, height, pos, tid, role, t["type"]) for addr, role in named.items()]
        return tx_rows, addr_rows

    def _advance_base(self) -> None:
        """Merge layers that left the reorg window into the base.

        After this the chain holds one flat base plus a diff layer per block above
        it, so memory and time per block do not grow with the ledger's size.
        """
        target = self.height - params.MAX_REORG_DEPTH - BASE_MARGIN
        if target <= self.base_height:
            return
        base = self.states[self.base_hash]
        for height in range(self.base_height + 1, target + 1):
            layer = self.states[self.main[height]]
            base.absorb(layer)
            if self.store is not None:
                self.store.write_base(layer.diff())
            if self.on_snapshot is not None and height % params.SNAPSHOT_INTERVAL == 0:
                self.on_snapshot(height, self.main[height], base)
        old = self.states[self.main[target]]
        for st in self.states.values():
            if st.parent is old:
                st.rebase(base)
        # Keep a state only if its whole ancestry stays above the new base. The rest
        # are merged into the base, or belong to forks that left the main chain too
        # deep to ever be accepted (their parents' layers are now merged away).
        # One pass, parents before children (a parent is one block lower): a state is
        # kept when its parent is the base or a kept state. (Walking every state's
        # ancestry instead cost O(window^2) per block.)
        kept, dropped = {}, []
        good = {id(base)}
        for k in sorted(self.states, key=lambda k: self.meta[k]["height"]):
            if self.meta[k]["height"] <= target:
                continue                         # merged into the base, or a fork below it
            st = self.states[k]
            if id(st.parent) in good:
                good.add(id(st))
                kept[k] = st
            else:
                dropped.append(k)
        self.base_hash, self.base_height = self.main[target], target
        kept[self.base_hash] = base
        self.states = kept
        if self.store is not None:
            self.store.set_meta("base", [self.base_hash, target])
            self.store.drop_diffs_to(target)
            self.store.drop_diffs(dropped)
            # Old blocks live in the store; memory keeps a header window below the base.
            floor = target - HEADER_KEEP
            for k in [k for k, m in self.meta.items() if m["height"] < floor and k != self.genesis_hash]:
                del self.blocks[k]
                del self.meta[k]

    # -- snapshots --

    def install_snapshot(self, headers: list[dict], state: State) -> None:
        """Adopt a verified snapshot (green/snapshot.py): `headers` are blocks 1..H
        in order, already checked against every header rule, and `state` is the
        ledger after H, already checked against H's state root. Only for a
        chain that has nothing but its genesis block."""
        if self.store is None or self.height != 0:
            raise ValueError("a snapshot can only be installed into an empty, stored chain")
        if not headers or headers[0]["prev"] != self.genesis_hash:
            raise ValueError("snapshot headers do not start at genesis")
        if state.root() != headers[-1]["state_root"]:
            raise ValueError("snapshot state does not match its header")
        rows, work, prev = [], self.meta[self.genesis_hash]["chainwork"], self.genesis_hash
        for hdr in headers:
            bh = block_hash({"header": hdr})
            if hdr["prev"] != prev:
                raise ValueError("snapshot headers do not form a chain")
            work += work_for_target(target_of(hdr))
            rows.append((bh, hdr, work))
            prev = bh
        tip, height = prev, headers[-1]["height"]
        with self.store.transaction():
            for bh, hdr, w in rows:
                self.store.put_block(bh, {"header": hdr, "txs": None}, hdr["height"], w)
            self.store.set_main_from(1, [bh for bh, _, _ in rows])
            self.store.replace_base(state)
            self.store.set_meta("base", [tip, height])
            self.store.drop_diffs_to(height)
            self.store.set_meta("first_block", height + 1)
        self.blocks, self.meta, self.states = {}, {}, {}
        self._open(self.store)

    def prune(self, keep: int) -> int:
        """Drop the bodies of main-chain blocks more than `keep` below the tip (a
        pruned node keeps every header). `keep` must cover the reorg window and the
        blocks held in memory. Returns how many bodies were dropped."""
        if self.store is None:
            return 0
        if keep < min_prune_keep():
            raise ValueError(f"a pruned node must keep at least {min_prune_keep()} blocks "
                             f"(the reorg window plus the blocks held in memory)")
        below = self.height - keep
        if below <= self.first_block:
            return 0
        with self.store.transaction():
            dropped = self.store.prune_bodies(self.first_block, below)
            self.store.set_meta("first_block", below)
        self.store.reclaim_space()
        return dropped

    @property
    def first_block(self) -> int:
        """The lowest height whose block body this node has (above a snapshot it started from)."""
        return self.store.get_meta("first_block", 0) if self.store is not None else 0

    # -- mining --

    def build_template(self, miner: str, candidates: list[dict], now: int | None = None, signal_skip=(),
                       shares: list = ()):
        """Pick valid transactions from `candidates` and return (header, txs, state).
        `shares` are included as given: the caller passes shares it checked for a
        block on the tip (green.node.Node.share_candidates).

        The header commits to the state after the block, which does not depend on
        the nonce or the work unit, so it stays valid for every attempt."""
        height = self.tip_header()["height"] + 1
        s = self.state.copy()
        chosen = []
        size = 2                                    # canonical(txs) is "[" + ",".join(...) + "]"
        sigs = 0
        for t in block_order(candidates):
            if len(chosen) >= params.MAX_BLOCK_TXS:
                break
            if sigs + txmod.sig_count(t) > params.MAX_BLOCK_SIGS:
                continue
            t_size = txmod.tx_size(t) + (1 if chosen else 0)
            if size + t_size > params.MAX_BLOCK_BYTES:
                continue
            try:
                self.apply_tx(s, t, height)
            except (InvalidTx, MissingData):
                continue
            chosen.append(t)
            size += t_size
            sigs += txmod.sig_count(t)
        header = {
            "version": self.deployments.version_for(self.tip, signal_skip), "height": height, "prev": self.tip,
            "time": max(int(now or time.time()), self.median_time_past(self.tip) + 1),
            "target": fmt_target(self.expected_target(self.tip)),
            "tx_root": merkle_root([txmod.txid(t) for t in chosen]),
            "shares_root": merkle_root([share_id(sh) for sh in shares]), "miner": miner,
            "work": {"job": None, "unit": None}, "nonce": 0, "msg": "",
        }
        header["state_root"] = self._finish_block(s, header, chosen, list(shares))
        return header, chosen, s
