"""Rule changes without splitting the network: version bits (BIP 9, height based as in BIP 8).

A proposed rule change (a "deployment") gets a bit of the block header's
version, a start height and a timeout height. Miners that run software with
the new rule set the bit. The chain is cut into periods of SIGNAL_PERIOD
blocks; at the start of each period every deployment moves through

    DEFINED -> STARTED -> LOCKED_IN -> ACTIVE
                  \\-> FAILED

STARTED once the period starts at or after `start`; LOCKED_IN when at least
SIGNAL_THRESHOLD blocks of the previous period set the bit (or at the timeout
if the deployment says `lock_in_on_timeout`, a mandatory activation); FAILED at
the timeout otherwise; ACTIVE one period after locking in, but not before
`min_activation` (the "Speedy Trial" refinement). The state of a block depends
only on its ancestors, so every node computes the same thing, and a reorg that
crosses a period boundary simply follows the other branch.

Activated rules must only tighten what is valid (a soft fork): nodes that never
heard of a deployment keep accepting the chain. A node that sees an unknown bit
set in most recent blocks warns its operator that an upgrade is under way.
"""

import time

from . import params

DEFINED, STARTED, LOCKED_IN, ACTIVE, FAILED = "defined", "started", "locked_in", "active", "failed"
TOP_BITS = 0x20000000          # every header version is TOP_BITS | signal bits
TOP_MASK = 0xE0000000
SIGNAL_BITS = 29               # bits 0..28 can signal


def valid_version(version: int) -> bool:
    return isinstance(version, int) and version & TOP_MASK == TOP_BITS and version < 2**32


def signals(version: int, bit: int) -> bool:
    return valid_version(version) and (version >> bit) & 1 == 1


class Deployments:
    """Deployment states for a chain (green/chain.py), cached by block hash."""

    def __init__(self, chain):
        self.chain = chain
        self._states: dict[tuple, str] = {}       # (name, last block of the previous period) -> state
        self._counts: dict[tuple, int] = {}       # (last block of a period, bit) -> blocks that signalled
        self._warn_cache: tuple | None = None

    # -- walking the chain --

    def _height(self, bh: str) -> int:
        return self.chain.header(bh)["height"]

    def ancestor(self, bh: str, height: int) -> str:
        """The block at `height` on the branch ending at `bh`."""
        cur = bh
        while True:
            n = self.chain.main_height(cur)
            if n is not None:
                return self.chain.main[height]
            hdr = self.chain.header(cur)
            if hdr["height"] == height:
                return cur
            cur = hdr["prev"]

    def _signal_count(self, end: str, bit: int) -> int:
        """How many of the SIGNAL_PERIOD blocks ending at `end` set `bit`."""
        key = (end, bit)
        if key not in self._counts:
            count, cur = 0, end
            for _ in range(params.SIGNAL_PERIOD):
                hdr = self.chain.header(cur)
                count += signals(hdr["version"], bit)
                if hdr["height"] == 0:
                    break
                cur = hdr["prev"]
            if len(self._counts) > 10_000:
                self._counts.clear()
            self._counts[key] = count
        return self._counts[key]

    # -- states --

    def state(self, name: str, prev_hash: str) -> str:
        """The state of deployment `name` for a block whose parent is `prev_hash`."""
        d = params.DEPLOYMENTS.get(name)
        if d is None:
            return FAILED                              # unknown to this node: never enforced
        period = params.SIGNAL_PERIOD
        k = (self._height(prev_hash) + 1) // period
        # Walk down to the newest period whose state is known, then compute upward.
        ends, todo = {}, []
        while k > 0:
            end = self.ancestor(prev_hash, k * period - 1)
            ends[k] = end
            if (name, end) in self._states:
                break
            todo.append(k)
            k -= 1
        state = self._states[(name, ends[k])] if k > 0 else DEFINED
        for k in reversed(todo):
            state = self._next(name, d, state, k, ends[k])
            self._states[(name, ends[k])] = state
        return state

    def _next(self, name: str, d: dict, prev: str, k: int, end: str) -> str:
        """The state of period k, given period k-1's state; `end` is k-1's last block."""
        height = k * params.SIGNAL_PERIOD
        if prev == DEFINED:
            return STARTED if height >= d["start"] else DEFINED
        if prev == STARTED:
            if self._signal_count(end, d["bit"]) >= params.SIGNAL_THRESHOLD:
                return LOCKED_IN
            if height >= d["timeout"]:
                return LOCKED_IN if d.get("lock_in_on_timeout") else FAILED
            return STARTED
        if prev == LOCKED_IN:
            return ACTIVE if height >= d.get("min_activation", 0) else LOCKED_IN
        return prev                                    # ACTIVE and FAILED are final

    def active(self, name: str, prev_hash: str) -> bool:
        return self.state(name, prev_hash) == ACTIVE

    def version_for(self, prev_hash: str, skip=()) -> int:
        """The version a miner of the next block uses: signal every deployment it
        supports that is STARTED or LOCKED_IN (unless told not to)."""
        version = TOP_BITS
        for name, d in params.DEPLOYMENTS.items():
            if name not in skip and self.state(name, prev_hash) in (STARTED, LOCKED_IN):
                version |= 1 << d["bit"]
        return version

    def info(self, prev_hash: str) -> list[dict]:
        """Every known deployment with its state and this period's signalling so far."""
        period = params.SIGNAL_PERIOD
        height = self._height(prev_hash) + 1
        start = height - height % period
        out = []
        for name, d in params.DEPLOYMENTS.items():
            st = self.state(name, prev_hash)
            count, cur = 0, prev_hash
            while st == STARTED:
                hdr = self.chain.header(cur)
                if hdr["height"] < start:
                    break
                count += signals(hdr["version"], d["bit"])
                if hdr["height"] == 0:
                    break
                cur = hdr["prev"]
            out.append({"name": name, "bit": d["bit"], "state": st, "start": d["start"], "timeout": d["timeout"],
                        "min_activation": d.get("min_activation", 0),
                        "lock_in_on_timeout": bool(d.get("lock_in_on_timeout")),
                        "period_start": start, "signalled": count if st == STARTED else None,
                        "threshold": params.SIGNAL_THRESHOLD, "period": period})
        return out

    def unknown_signals(self, tip: str, window: int = 1000) -> list[int]:
        """Bits set by more than half of the last `window` blocks that this node has
        no deployment for: a rule change it does not know is being signalled.
        Recomputed at most every 30 seconds (a late warning by a few blocks is fine)."""
        if self._warn_cache and (self._warn_cache[0] == tip or time.time() - self._warn_cache[2] < 30):
            return self._warn_cache[1]
        known = {d["bit"] for d in params.DEPLOYMENTS.values()}
        counts = [0] * SIGNAL_BITS
        seen, cur = 0, tip
        while seen < window:
            hdr = self.chain.header(cur)
            v = hdr["version"]
            for bit in range(SIGNAL_BITS):
                counts[bit] += signals(v, bit)
            seen += 1
            if hdr["height"] == 0:
                break
            cur = hdr["prev"]
        bits = [b for b in range(SIGNAL_BITS) if b not in known and counts[b] * 2 > seen]
        self._warn_cache = (tip, bits, time.time())
        return bits
