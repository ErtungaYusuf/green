"""Ledger state as copy-on-write diff layers over one flat base.

v0.4 copied the whole ledger for every block (and kept 120 copies), so a
block cost O(accounts) time and memory. Here a block's state is a thin layer
holding only the keys that block changed, on top of its parent's layers; reads
fall through to the parent. Layers older than the reorg window are merged into
a single flat base (the same idea as Ethereum's snapshot diff layers), so a
block costs O(keys it touches) and memory holds one ledger plus recent diffs.

Safety rules the chain relies on:
  * `State.copy()` freezes the parent: a state that has children can never be
    modified again (any attempt raises), so a committed block's state cannot
    be corrupted by later code.
  * A mutable value (a job, a task, an event list) found in an ancestor is
    copied into the top layer the first time it is accessed through
    `[]`/`get`/`pop`/`setdefault`, so nested mutation never reaches a parent.
    Iteration (`items()`, `values()`) is read-only.

State commitment: every state carries a lattice hash of its full content
(LtHash, Lewi et al. 2019; the construction Solana adopted for its accounts
hash). Each entry (map, key, value) is expanded with SHAKE-256 into 1024
16-bit lanes and the state's hash is the lane-wise sum of all entries modulo
2^16. Adding or removing an entry is one vector addition or subtraction, so a
block updates the hash in O(keys it changed), and the hash of a state does not
depend on the order its entries were written in. The state root in each block
header is BLAKE2b-256 of the 2048-byte lattice.
"""

import hashlib
from collections.abc import MutableMapping

import numpy as np

_MISSING = object()
LATTICE_LANES = 1024
_LATTICE_DOMAIN = b"grn-state-v1:"


class Layer(MutableMapping):
    __slots__ = ("data", "dead", "parent", "frozen", "copy_value")

    def __init__(self, parent: "Layer | None" = None, copy_value=None):
        self.data: dict = {}
        self.dead: set = set()          # keys deleted in this layer
        self.parent = parent
        self.frozen = False
        self.copy_value = copy_value    # deep-copies a mutable value; None for immutable ones

    # -- lookup --

    def _find(self, key):
        layer = self
        while layer is not None:
            v = layer.data.get(key, _MISSING)
            if v is not _MISSING:
                return v, layer
            if key in layer.dead:
                return _MISSING, None
            layer = layer.parent
        return _MISSING, None

    def _own(self, key):
        """The value for `key`, made local to this layer if it is mutable."""
        v, where = self._find(key)
        if v is _MISSING or where is self or self.frozen or self.copy_value is None:
            return v
        v = self.copy_value(v)
        self.data[key] = v
        return v

    def __getitem__(self, key):
        v = self._own(key)
        if v is _MISSING:
            raise KeyError(key)
        return v

    def get(self, key, default=None):
        v = self._own(key)
        return default if v is _MISSING else v

    def __contains__(self, key) -> bool:
        return self._find(key)[0] is not _MISSING

    # -- mutation --

    def _check(self):
        if self.frozen:
            raise RuntimeError("state is frozen: it has children, copy it to change it")

    def __setitem__(self, key, value):
        self._check()
        self.data[key] = value
        self.dead.discard(key)

    def __delitem__(self, key):
        self._check()
        if key not in self:
            raise KeyError(key)
        self.data.pop(key, None)
        if self.parent is not None:
            self.dead.add(key)

    def pop(self, key, default=_MISSING):
        self._check()
        v = self._own(key)
        if v is _MISSING:
            if default is _MISSING:
                raise KeyError(key)
            return default
        del self[key]
        return v

    def setdefault(self, key, default=None):
        self._check()
        v = self._own(key)
        if v is _MISSING:
            self[key] = default
            return default
        return v

    # -- iteration (read-only) --

    def _chain(self) -> list["Layer"]:
        layers, layer = [], self
        while layer is not None:
            layers.append(layer)
            layer = layer.parent
        return layers[::-1]                  # base first

    def _merged(self) -> dict:
        out: dict = {}
        for layer in self._chain():
            for k in layer.dead:
                out.pop(k, None)
            out.update(layer.data)
        return out

    def __iter__(self):
        if self.parent is None:
            return iter(list(self.data))
        return iter(list(self._merged()))

    def __len__(self) -> int:
        return len(self.data) if self.parent is None else len(self._merged())

    def items(self):
        return (self.data if self.parent is None else self._merged()).items()

    def values(self):
        return (self.data if self.parent is None else self._merged()).values()

    def keys(self):
        return (self.data if self.parent is None else self._merged()).keys()

    def depth(self) -> int:
        n, layer = 0, self
        while layer is not None:
            n, layer = n + 1, layer.parent
        return n


# -- deep copies of the mutable values ------------------------------------------

def copy_job(v: dict) -> dict:
    # Strings, "delivered" entries and bitmaps are never mutated in place, so they can be shared.
    return {**v, "done": dict(v["done"]), "reserved": dict(v["reserved"]), "delivered": dict(v["delivered"]),
            "batches": {b: {**x, "disputes": {u: dict(d) for u, d in x["disputes"].items()},
                            "proven": set(x["proven"])}
                        for b, x in v["batches"].items()}}


def copy_task(v: dict) -> dict:
    return {**v, "reserved": dict(v["reserved"]) if v["reserved"] else None}


def plain(v):
    """A value as plain JSON data (sets become sorted lists, keys strings)."""
    if isinstance(v, set):
        return sorted(v)
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [plain(x) for x in v]
    return v


def entry_bytes(name: str, key, value) -> bytes:
    from .crypto import canonical
    return hashlib.shake_256(_LATTICE_DOMAIN + canonical([name, key, plain(value)])).digest(2 * LATTICE_LANES)


def lattice_sum(entries: list[bytes]) -> np.ndarray:
    if not entries:
        return np.zeros(LATTICE_LANES, dtype=np.uint64)
    return np.frombuffer(b"".join(entries), dtype="<u2").reshape(-1, LATTICE_LANES).sum(axis=0, dtype=np.uint64)


def _lattice(added: list[bytes], removed: list[bytes], start: np.ndarray | None = None) -> np.ndarray:
    acc = np.zeros(LATTICE_LANES, dtype=np.uint64) if start is None else start.astype(np.uint64)
    # Subtraction modulo 2^16 is addition of the complement.
    acc = (acc + lattice_sum(added) + (65536 * len(removed) - lattice_sum(removed))) % 65536
    return acc.astype("<u2")


def lattice_root(lattice: np.ndarray) -> str:
    from .crypto import hhex
    return hhex(lattice.astype("<u2").tobytes())


# name -> how to copy a value of that map (None: immutable values)
MAPS = {
    "balances": None,
    "nonces": None,
    "jobs": copy_job,
    "tasks": copy_task,
    "closing": list,          # height -> jobs that close then
    "job_events": list,       # height -> [[kind, job_id, batch_id, unit_key]]
    "task_settle": list,      # height -> tasks that settle then
    "guards": dict,           # address -> {"guardian", "limit", "period", "frozen", "window", "spent"}
    "channels": dict,         # channel id -> {"buyer", "pub", "provider", "amount", "paid", "expires"}
    "channel_expiry": list,   # height -> channels that end then
    "protocol": None,         # "reserve" -> the coins not yet paid out (and fees flowing back)
    "share_seen": None,       # share id -> height of the block that included it
    "share_expiry": list,     # height -> shares included then (forgotten SHARE_WINDOW + 1 blocks later)
}


class State:
    """The ledger after some block. See the module docstring for the rules."""

    def __init__(self, parent: "State | None" = None):
        self.parent = parent
        for name, copy_value in MAPS.items():
            setattr(self, name, Layer(getattr(parent, name) if parent else None, copy_value))
        # job id -> {"buyer", "dims", "price", "escrow",
        #            "done": {unit_key: batch_id},             settled units and who delivered them
        #            "delivered": {batch_id: {"miner", "units", "count", "root", "manifest", ["payees"]}},
        #            "reserved": {unit_key: batch_id},         units held by a live batch
        #            "batches": {batch_id: {"miner", "units" (bitmap), "count", "height", "expires",
        #                                   "bond", "status": "committed" | "revealed",
        #                                   "root", "manifest", "settle_at",
        #                                   "disputes": {unit_key: {"by", "deadline"}},
        #                                   "proven": set of unit keys}},
        #            "closes_at": height or None, "closed": bool}
        # task id -> {"buyer","prog","input","fuel","price","escrow","status",
        #             "worker","output","reserved" or None,"settle_at" or None}
        # Disputes waiting for an answer, network wide. A new dispute's deadline is
        # pushed back by this backlog, so answers always fit in the blocks before it.
        self.open_disputes = parent.open_disputes if parent else 0
        # Per-block budgets. Deliberately not inherited: every block starts from
        # state.copy(), so they reset exactly at block boundaries.
        self.adjudication_used = 0
        self.responses_used = 0
        # The lattice hash of this state's content, valid once commit() has run
        # (a parentless state starts as the empty ledger with no open disputes).
        self.lattice = parent.lattice if parent else _lattice([entry_bytes("meta", "open_disputes", 0)], [])
        self._lattice_disputes = parent._lattice_disputes if parent else 0
        self.committed = False

    # -- the state commitment --

    def commit(self) -> str:
        """Fold this state's own changes into its lattice hash; returns the state root.

        Called once, after a block's last change: entries that changed leave the
        hash with their old value and enter it with the new one. O(keys changed).
        """
        if self.committed:
            raise RuntimeError("state already committed")
        added, removed = [], []
        for name, layer in self.layers():
            parent = layer.parent
            for k, v in layer.data.items():
                old = parent._find(k)[0] if parent is not None else _MISSING
                if old is not _MISSING:
                    if old == v:
                        continue                 # copied on access, not changed
                    removed.append(entry_bytes(name, k, old))
                added.append(entry_bytes(name, k, v))
            for k in layer.dead:
                old = parent._find(k)[0] if parent is not None else _MISSING
                if old is not _MISSING:
                    removed.append(entry_bytes(name, k, old))
        if self.open_disputes != self._lattice_disputes:
            removed.append(entry_bytes("meta", "open_disputes", self._lattice_disputes))
            added.append(entry_bytes("meta", "open_disputes", self.open_disputes))
            self._lattice_disputes = self.open_disputes
        self.lattice = _lattice(added, removed, self.lattice)
        self.committed = True
        return self.root()

    def root(self) -> str:
        return lattice_root(self.lattice)

    def recompute_lattice(self) -> np.ndarray:
        """The lattice hash computed from scratch over every entry (snapshots, tests)."""
        entries = [entry_bytes("meta", "open_disputes", self.open_disputes)]
        for name, layer in self.layers():
            entries += [entry_bytes(name, k, v) for k, v in layer.items()]
        return _lattice(entries, [])

    def _set_lattice(self, lattice: np.ndarray) -> None:
        self.lattice = lattice
        self._lattice_disputes = self.open_disputes
        self.committed = True

    # -- layering --

    def copy(self) -> "State":
        """A child state to apply the next block (or a candidate block) to.

        O(1): the child starts empty and reads through to this state, which is
        frozen from now on.
        """
        for name in MAPS:
            getattr(self, name).frozen = True
        return State(self)

    def layers(self):
        return [(name, getattr(self, name)) for name in MAPS]

    def depth(self) -> int:
        return self.balances.depth()

    def absorb(self, child: "State") -> None:
        """Merge `child`'s own layer into this flat state (used to move the base up).

        This state must be flat (no parent). Any other state stacked on this one
        sees the change, which is why only the chain's base is ever absorbed into.
        """
        for name, layer in child.layers():
            mine = getattr(self, name)
            assert mine.parent is None
            for k in layer.dead:
                mine.data.pop(k, None)
            mine.data.update(layer.data)
        self.open_disputes = child.open_disputes
        self._set_lattice(child.lattice)

    def rebase(self, new_parent: "State") -> None:
        """Point this state's layers at `new_parent` (which holds the same content)."""
        self.parent = new_parent
        for name, layer in self.layers():
            layer.parent = getattr(new_parent, name)

    def flattened(self) -> "State":
        """A new flat, mutable state with the same content."""
        flat = State()
        for name, layer in self.layers():
            target = getattr(flat, name)
            for k, v in layer.items():
                target.data[k] = v if MAPS[name] is None else MAPS[name](v)
        flat.open_disputes = self.open_disputes
        flat._set_lattice(self.lattice)
        return flat

    def diff(self) -> dict:
        """This state's own layer: what its block changed."""
        return {"maps": {name: {"set": dict(layer.data), "dead": set(layer.dead)} for name, layer in self.layers()},
                "open_disputes": self.open_disputes, "lattice": self.lattice.tobytes().hex()}

    @classmethod
    def from_diff(cls, parent: "State | None", diff: dict) -> "State":
        s = parent.copy() if parent is not None else cls()
        for name, part in diff["maps"].items():
            layer = getattr(s, name)
            layer.data.update(part["set"])
            layer.dead.update(part["dead"] if parent is not None else ())
        s.open_disputes = diff["open_disputes"]
        s._set_lattice(np.frombuffer(bytes.fromhex(diff["lattice"]), dtype="<u2").copy())
        return s

    # -- convenience --

    def balance(self, addr: str) -> int:
        return self.balances.get(addr, 0)

    def nonce(self, addr: str) -> int:
        return self.nonces.get(addr, 0)

    def set_balance(self, addr: str, amount: int) -> None:
        """Balances hold non-zero amounts only, so the state (and its root) does not
        depend on which addresses were ever touched with zero."""
        if amount:
            self.balances[addr] = amount
        elif addr in self.balances:
            del self.balances[addr]

    def credit(self, addr: str, amount: int) -> None:
        self.set_balance(addr, self.balance(addr) + amount)

    def open_units(self, job_id: str) -> list[list[int]]:
        """Units a miner could commit to now (neither delivered nor held by a live batch)."""
        from . import tx as txmod
        job = self.jobs[job_id]
        if job["closed"] or job["closes_at"] is not None:
            return []
        d = job["dims"]
        if len(job["done"]) >= d[0] * d[1] * d[2]:   # finished: answered without listing its units
            return []
        return [u for u in txmod.all_units(job["dims"])
                if txmod.unit_key(u) not in job["done"] and txmod.unit_key(u) not in job["reserved"]]

    def schedule(self, height: int, kind: str, job_id: str, batch_id: str, unit_key: str | None = None) -> None:
        self.job_events.setdefault(height, []).append([kind, job_id, batch_id, unit_key])

    @property
    def reserve(self) -> int:
        return self.protocol.get("reserve", 0)

    @reserve.setter
    def reserve(self, amount: int) -> None:
        self.protocol["reserve"] = amount

    @property
    def burned(self) -> int:
        """Fees destroyed so far: the supply is TOTAL_SUPPLY minus this."""
        return self.protocol.get("burned", 0)

    @burned.setter
    def burned(self, amount: int) -> None:
        self.protocol["burned"] = amount

    def locked_total(self) -> int:
        """Coins held by the protocol (escrow + bonds), for supply accounting."""
        jobs = sum(j["escrow"] + sum(b["bond"] for b in j["batches"].values()) for j in self.jobs.values())
        tasks = sum(t["escrow"] + (t["reserved"]["bond"] if t["reserved"] else 0) for t in self.tasks.values())
        channels = sum(c["amount"] - c["paid"] for c in self.channels.values())
        return jobs + tasks + channels

    def fingerprint(self) -> str:
        """A hash of the full content, to compare two states (tests, restarts)."""
        from .crypto import canonical, hhex
        return hhex(canonical({"maps": {name: plain(dict(layer.items())) for name, layer in self.layers()},
                               "open_disputes": self.open_disputes}))
