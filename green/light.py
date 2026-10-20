"""A light client: verify payments without running a full node or trusting one.

Bitcoin's SPV in Green's terms. The client keeps only block headers, which it
checks itself: every header rule (linkage, heights, the median-time rule, the
ASERT target) on every header, plus the work unit of a random sample of headers
and of the newest one, recomputed on the client (green/p2p.py explains why a
sample is enough to make a forged header chain cost about as much as the real
one). A payment is then proven by a Merkle proof from its transaction id to the
`tx_root` of a header in that chain, buried under some number of blocks.

The node it talks to is trusted for nothing but availability: wrong headers,
wrong work or wrong proofs are detected. When the node's chain reorganizes, the
client finds the fork, checks the new branch the same way and takes it only if it
has more work than the headers it replaces, as a full node would.
"""

import random

from . import params
from .chain import BlobStore, Chain, InvalidBlock, MissingData, block_hash, target_of, work_for_target
from .crypto import merkle_verify
from .node import http_get

WORK_SAMPLES = 8          # work units recomputed per batch of headers (plus the newest)
BATCH = 2000


class LightClient:
    def __init__(self, node_url: str, genesis: dict):
        self.node = node_url.rstrip("/")
        self.chain = Chain(genesis, BlobStore())      # only for the header rules and work checks
        g = block_hash(genesis)
        self.headers: dict[str, dict] = {g: genesis["header"]}
        self.by_height: list[str] = [g]
        self.work = work_for_target(target_of(genesis["header"]))

    @property
    def height(self) -> int:
        return len(self.by_height) - 1

    def _lookup(self, bh: str) -> dict:
        return self.headers[bh]

    def sync(self) -> int:
        """Download and check headers up to the node's tip; returns the new height."""
        while True:
            batch = http_get(f"{self.node}/v1/headers?from={self.height + 1}&count={BATCH}", timeout=60,
                             limit=16 * 1024 * 1024)
            if not batch:
                return self.height
            if isinstance(batch[0], dict) and batch[0].get("prev") != self.by_height[-1]:
                self._reorg()                          # the node's chain no longer extends ours
                continue
            self._add(batch)
            if len(batch) < BATCH:
                return self.height

    def _add(self, batch: list[dict]) -> None:
        new = []
        try:
            for hdr in batch:
                if not isinstance(hdr, dict) or hdr.get("prev") != (new[-1][0] if new else self.by_height[-1]):
                    raise InvalidBlock("headers do not extend the chain")
                self.chain.validate_header(hdr, lookup=self._lookup)
                bh = block_hash({"header": hdr})
                new.append((bh, hdr))
                self.headers[bh] = hdr                 # visible to the next header's checks
            for bh, hdr in random.sample(new, min(WORK_SAMPLES, len(new))) + new[-1:]:
                if not self._work_ok(hdr):
                    raise InvalidBlock(f"header {bh[:12]} claims work that was not done")
        except (InvalidBlock, KeyError, TypeError):
            for h, _ in new:
                self.headers.pop(h, None)
            raise
        for bh, hdr in new:
            self.by_height.append(bh)
            self.work += work_for_target(target_of(hdr))

    def _reorg(self) -> None:
        """Switch to the node's branch: find where it leaves ours (within MAX_REORG_DEPTH,
        like a full node), check its headers, and keep it only if it has more work."""
        start = max(1, self.height - params.MAX_REORG_DEPTH)
        theirs = http_get(f"{self.node}/v1/headers?from={start}&count={BATCH}", timeout=60, limit=16 * 1024 * 1024)
        if not isinstance(theirs, list) or not theirs or not isinstance(theirs[0], dict) \
                or theirs[0].get("prev") != self.by_height[start - 1]:
            raise InvalidBlock("the node's chain leaves ours deeper than a reorganization can go")
        fork = start
        for hdr in theirs:
            if fork <= self.height and block_hash({"header": hdr}) == self.by_height[fork]:
                fork += 1
            else:
                break
        branch = theirs[fork - start:]
        dropped = self.by_height[fork:]
        dropped_work = sum(work_for_target(target_of(self.headers[bh])) for bh in dropped)
        self.by_height, saved_work = self.by_height[:fork], self.work
        self.work -= dropped_work
        try:
            self._add(branch)
            if self.work <= saved_work:
                raise InvalidBlock("the node's branch has no more work than ours")
        except (InvalidBlock, KeyError, TypeError):
            for bh in self.by_height[fork:]:
                self.headers.pop(bh, None)
            self.by_height, self.work = self.by_height[:fork] + dropped, saved_work
            raise

    def _work_ok(self, hdr: dict) -> bool:
        for _ in range(2):
            try:
                return self.chain.work_ok(hdr, self.chain.work_transcript(hdr))
            except MissingData as e:                 # useful work: fetch the job's public data
                try:
                    self.chain.blobs.put(http_get(f"{self.node}/v1/blob/{e.blob_hash}", raw=True, timeout=60,
                                                  limit=e.max_size))
                except OSError:
                    return False
            except InvalidBlock:
                return False
        return False

    def verify_payment(self, txid: str, min_confirmations: int = 6) -> dict:
        """Check that `txid` is in a block of the verified header chain with enough
        blocks on top. Returns the proof; raises ValueError if it does not hold."""
        p = http_get(f"{self.node}/v1/proof/tx/{txid}", timeout=30)
        hdr = self.headers.get(p.get("block")) if isinstance(p, dict) and isinstance(p.get("block"), str) else None
        if hdr is None or hdr["height"] > self.height or self.by_height[hdr["height"]] != p["block"]:
            raise ValueError("the block is not in the verified header chain (sync first?)")
        try:
            ok = (p.get("txid") == txid and type(p.get("index")) is int and type(p.get("count")) is int
                  and isinstance(p.get("proof"), list) and p["count"] < 2**32
                  and merkle_verify(txid, p["index"], p["count"], p["proof"], hdr["tx_root"]))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError("the inclusion proof does not match the block's tx_root")
        confirmations = self.height - hdr["height"] + 1
        if confirmations < min_confirmations:
            raise ValueError(f"only {confirmations} confirmations, {min_confirmations} wanted")
        return {**p, "confirmations": confirmations}
