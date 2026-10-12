"""A light client: verify payments without running a full node or trusting one.

Bitcoin's SPV in Green's terms. The client keeps only block headers, which it
checks itself: every header rule (linkage, heights, the median-time rule, the
ASERT target) on every header, plus the work unit of a random sample of headers
and of the newest one, recomputed on the client (green/p2p.py explains why a
sample is enough to make a forged header chain cost about as much as the real
one). A payment is then proven by a Merkle proof from its transaction id to the
`tx_root` of a header in that chain, buried under some number of blocks.

The node it talks to is trusted for nothing but availability: wrong headers,
wrong work or wrong proofs are detected.
"""

import random

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
        hdr = self.headers.get(p["block"])
        if hdr is None or self.by_height[hdr["height"]] != p["block"]:
            raise ValueError("the block is not in the verified header chain (sync first?)")
        if p["txid"] != txid or not merkle_verify(txid, p["index"], p["count"], p["proof"], hdr["tx_root"]):
            raise ValueError("the inclusion proof does not match the block's tx_root")
        confirmations = self.height - hdr["height"] + 1
        if confirmations < min_confirmations:
            raise ValueError(f"only {confirmations} confirmations, {min_confirmations} wanted")
        return {**p, "confirmations": confirmations}
