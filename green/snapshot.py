"""State snapshots: how a new node joins without replaying all of history.

Every SNAPSHOT_INTERVAL blocks a node writes the ledger at that height as
content-addressed chunks plus a manifest naming them. Snapshot heights sit at
the node's base, more than MAX_REORG_DEPTH below its tip, so they are final.

A new node started with fast sync first downloads and checks the header chain
(green/p2p.py), then fetches a manifest and its chunks from peers. Each chunk is
checked against its hash in the manifest as it arrives, and the lattice hash
of all entries must equal the state root that the snapshot block's header
commits to. That header is buried under the heaviest chain's work, so trusting
the snapshot is trusting that chain, the same assumption as Ethereum's snap
sync and stronger than Bitcoin's assumeutxo (which trusts a hash shipped in
the software). From the snapshot height on, every block is fully validated.

It also means a node never needs the matrices of jobs that were finished long
ago: only blocks after the snapshot are re-executed.
"""

import json

from .crypto import canonical, hhex, is_hex
from .state import MAPS, State, _lattice, entry_bytes, lattice_root, lattice_sum
from .store import dec, enc

CHUNK_BYTES = 1 << 20          # entries per chunk are cut at about this size
MAX_MANIFEST = 4 << 20
MAX_CHUNK = 8 << 20            # one entry (a large job) may push a chunk past CHUNK_BYTES


def make_snapshot(state: State, height: int, block_hash_: str, put_blob) -> dict:
    """Write `state` (the flat state after block `block_hash_`) as chunks via
    put_blob; returns the manifest (whose own hash is manifest["id"])."""
    chunks, current, size = [], [], 0
    for name in MAPS:
        for k, v in sorted(getattr(state, name).items(), key=lambda kv: (type(kv[0]).__name__, kv[0])):
            row = [name, k, v]
            text = enc(row)
            if current and size + len(text) > CHUNK_BYTES:
                chunks.append(put_blob(enc(current).encode()))
                current, size = [], 0
            current.append(row)
            size += len(text) + 1
    if current or not chunks:
        chunks.append(put_blob(enc(current).encode()))
    manifest = {"height": height, "hash": block_hash_, "root": state.root(),
                "open_disputes": state.open_disputes, "chunks": chunks}
    body = canonical(manifest)
    manifest["id"] = put_blob(body)
    return manifest


def parse_manifest(data: bytes) -> dict:
    """Check a manifest's shape (its content is checked against the header later)."""
    if len(data) > MAX_MANIFEST:
        raise ValueError("manifest too large")
    m = json.loads(data)
    if not isinstance(m, dict) or set(m) != {"height", "hash", "root", "open_disputes", "chunks"}:
        raise ValueError("bad manifest fields")
    if not (isinstance(m["height"], int) and m["height"] > 0 and is_hex(m["hash"], 64) and is_hex(m["root"], 64)):
        raise ValueError("bad manifest header")
    if not (isinstance(m["open_disputes"], int) and m["open_disputes"] >= 0):
        raise ValueError("bad manifest dispute count")
    if not (isinstance(m["chunks"], list) and 0 < len(m["chunks"]) <= 1_000_000
            and all(is_hex(c, 64) for c in m["chunks"])):
        raise ValueError("bad manifest chunk list")
    m["id"] = hhex(data)
    return m


class Builder:
    """Collects a snapshot's chunks, checking each, and rebuilds the state."""

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.state = State()
        self.sum = lattice_sum([entry_bytes("meta", "open_disputes", manifest["open_disputes"])])
        self.next = 0

    def add(self, data: bytes) -> None:
        """Add the next chunk (in manifest order). Raises ValueError if it is not that chunk."""
        if self.next >= len(self.manifest["chunks"]) or hhex(data) != self.manifest["chunks"][self.next]:
            raise ValueError("chunk does not match the manifest")
        rows = dec(data.decode())
        if not isinstance(rows, list):
            raise ValueError("bad chunk")
        entries = []
        for row in rows:
            if not (isinstance(row, list) and len(row) == 3 and row[0] in MAPS):
                raise ValueError("bad snapshot entry")
            name, key, value = row
            layer = getattr(self.state, name)
            if key in layer.data:
                raise ValueError("duplicate snapshot entry")
            layer.data[key] = value
            entries.append(entry_bytes(name, key, value))
        self.sum = self.sum + lattice_sum(entries)
        self.next += 1

    @property
    def complete(self) -> bool:
        return self.next == len(self.manifest["chunks"])

    def finish(self, expected_root: str) -> State:
        """The rebuilt state, if its lattice hash is the root the header commits to."""
        if not self.complete:
            raise ValueError("snapshot incomplete")
        lattice = _lattice([], [], self.sum % 65536)
        if lattice_root(lattice) != expected_root:
            raise ValueError("snapshot does not hash to the header's state root")
        self.state.open_disputes = self.manifest["open_disputes"]
        self.state._set_lattice(lattice)
        return self.state
