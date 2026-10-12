"""Snapshots: a new node starts from the ledger at a recent height, checked
against the state root in that block's header, and validates every block after
it. A snapshot that does not hash to the header's root is refused."""

import shutil
import socket
import time

import pytest

from green import chain as chainmod, params
from green.chain import BlobStore, Chain, make_genesis
from green.crypto import Key, canonical
from green.node import Node
from green.snapshot import Builder, make_snapshot, parse_manifest
from green.state import lattice_root
from green.store import Store
from helpers import T0, add, tx
from test_storage import build


@pytest.fixture(autouse=True)
def short_windows(monkeypatch):
    monkeypatch.setattr(params, "MAX_REORG_DEPTH", 4)
    monkeypatch.setattr(params, "SNAPSHOT_INTERVAL", 10)
    monkeypatch.setattr(chainmod, "BASE_MARGIN", 2)
    monkeypatch.setattr(chainmod, "HEADER_KEEP", 3)
    monkeypatch.setattr(params, "ASERT_HALFLIFE", 10**9)
    for name, value in (("RESERVE_BLOCKS", 3), ("JOB_CHALLENGE_BLOCKS", 3), ("JOB_RESPONSE_BLOCKS", 2)):
        monkeypatch.setattr(params, name, value)


def rebuild(blobs, manifest, root):
    builder = Builder(parse_manifest(blobs.get(manifest["id"])))
    for c in manifest["chunks"]:
        builder.add(blobs.get(c))
    return builder.finish(root)


def test_a_snapshot_rebuilds_the_exact_ledger_and_the_chain_continues(tmp_path):
    keys = [Key.generate() for _ in range(4)]
    genesis = make_genesis(keys[0].address, "snapshot", timestamp=T0)
    blobs = BlobStore(str(tmp_path / "blobs"))
    src = Chain(genesis, blobs, store=Store(str(tmp_path / "src.db")))
    manifests = []
    src.on_snapshot = lambda h, bh, st: manifests.append(make_snapshot(st, h, bh, blobs.put))
    build(src, keys, 70, seed=4)                          # transfers, jobs, sales, reorgs
    assert [m["height"] for m in manifests] == list(range(10, src.base_height + 1, 10))
    m = manifests[-1]
    hdr = src.header(m["hash"])
    assert m["root"] == hdr["state_root"] and src.main[m["height"]] == m["hash"]
    state = rebuild(blobs, m, hdr["state_root"])
    assert any(state.jobs.items()), "the snapshot should carry jobs (nested values, sets)"

    store = Store(str(tmp_path / "dst.db"))
    dst = Chain(genesis, blobs, store=store)
    dst.install_snapshot([src.header(src.main[h]) for h in range(1, m["height"] + 1)], state)
    assert dst.height == m["height"] and dst.state.root() == hdr["state_root"]
    assert dst.block(src.main[1]) is None and dst.first_block == m["height"] + 1   # bodies below are not here
    for bh in src.main[m["height"] + 1:]:
        assert dst.add_block(src.block(bh)) == "added"
    assert dst.tip == src.tip and dst.state.root() == src.state.root()
    assert dst.state.fingerprint() == src.state.fingerprint()
    store.close()
    again = Chain(genesis, blobs, store=Store(str(tmp_path / "dst.db")))              # and it survives a restart
    assert again.tip == src.tip and again.state.root() == src.state.root() and again.first_block == m["height"] + 1
    again.store.close()


def test_a_tampered_snapshot_is_refused(tmp_path):
    keys = [Key.generate() for _ in range(3)]
    genesis = make_genesis(keys[0].address, "tamper", timestamp=T0)
    blobs = BlobStore()
    src = Chain(genesis, blobs)
    manifests = []
    src.on_snapshot = lambda h, bh, st: manifests.append((make_snapshot(st, h, bh, blobs.put), st.flattened()))
    for _ in range(30):
        add(src, keys[0], [tx(src, keys[0], "transfer", to=keys[1].address, amount=5)])
    m, honest = manifests[-1]
    root = src.header(m["hash"])["state_root"]

    builder = Builder(m)
    with pytest.raises(ValueError, match="manifest"):
        builder.add(b'[["balances","x",1]]')                # not the chunk the manifest names
    fake = honest.flattened()
    fake.balances[keys[2].address] = 10**15               # coins from nowhere
    forged = make_snapshot(fake, m["height"], m["hash"], blobs.put)
    with pytest.raises(ValueError, match="state root"):
        rebuild(blobs, forged, root)                      # every chunk matches its manifest, the root does not
    for bad in (b"[]", b'{"height": 1}', canonical({**{k: m[k] for k in ("height", "hash", "root", "open_disputes")},
                                                    "chunks": ["zz"]})):
        with pytest.raises(ValueError):
            parse_manifest(bad)

    dst = Chain(genesis, BlobStore(), store=Store(str(tmp_path / "d.db")))
    headers = [src.header(src.main[h]) for h in range(1, m["height"] + 1)]
    with pytest.raises(ValueError, match="does not match"):
        dst.install_snapshot(headers, rebuild(blobs, forged, lattice_root(fake.recompute_lattice())))
    with pytest.raises(ValueError, match="genesis"):
        dst.install_snapshot(headers[1:], rebuild(blobs, m, root))
    add(dst, keys[0])
    with pytest.raises(ValueError, match="empty"):
        dst.install_snapshot(headers, rebuild(blobs, m, root))
    dst.store.close()


# --- over the network -------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(cond, timeout=60.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def source_nodes(tmp_path, count):
    """Nodes sharing one history with jobs in it; the first built it, the others copied it."""
    keys = [Key.generate() for _ in range(4)]
    genesis = make_genesis(keys[0].address, "fast sync", timestamp=T0)
    first = Node(str(tmp_path / "s0"), genesis, port=free_port(), log=lambda m: None)
    build(first.chain, keys, 60, seed=9)
    nodes = [first]
    for i in range(1, count):
        shutil.copytree(tmp_path / "s0" / "blobs", tmp_path / f"s{i}" / "blobs")
        n = Node(str(tmp_path / f"s{i}"), genesis, port=free_port(), log=lambda m: None)
        for bh in first.chain.main[1:]:
            assert n.chain.add_block(first.chain.block(bh)) == "added"
        nodes.append(n)
    return genesis, nodes


def test_a_new_node_joins_from_a_snapshot(tmp_path):
    genesis, (source,) = source_nodes(tmp_path, 1)
    assert source.snapshots, "the source should have written snapshots"
    snap = source.snapshots[-1]
    source.start()
    fresh = Node(str(tmp_path / "fresh"), genesis, port=free_port(), log=lambda m: None, fast_sync=True,
                 peers=[source.p2p.address])
    fresh.start()
    try:
        wait_for(lambda: fresh.chain.tip == source.chain.tip, what="fast sync")
        assert fresh.p2p.snap_state == "done"
        assert fresh.chain.first_block == snap["height"] + 1               # started at the newest snapshot
        assert fresh.chain.block(source.chain.main[1]) is None             # never downloaded old blocks
        assert fresh.chain.state.root() == source.chain.state.root()
    finally:
        fresh.stop()
        source.stop()


def test_a_lying_snapshot_server_is_banned_and_sync_falls_back(tmp_path):
    genesis, (honest, liar) = source_nodes(tmp_path, 2)
    snap = honest.snapshots[-1]
    root = honest.chain.header(snap["hash"])["state_root"]
    state = rebuild(honest.blobs, snap, root)
    state.balances[Key.generate().address] = 10**15
    forged = make_snapshot(state, snap["height"], snap["hash"], liar.blobs.put)
    body = {"height": snap["height"], "hash": snap["hash"], "root": root,   # claims the honest root
            "open_disputes": forged["open_disputes"], "chunks": forged["chunks"]}
    liar.snapshots = [dict(body, id=liar.blobs.put(canonical(body)))]
    honest.snapshots = []                                                  # only the liar offers one
    honest.start()
    liar.start()
    fresh = Node(str(tmp_path / "fresh"), genesis, port=free_port(), log=lambda m: None, fast_sync=True,
                 peers=[liar.p2p.address, honest.p2p.address])
    fresh.start()
    try:
        wait_for(lambda: fresh.chain.tip == honest.chain.tip, what="sync after the failed snapshot")
        assert fresh.p2p.snap_state == "failed" and fresh.p2p.stats["banned"] == 1
        assert fresh.p2p.is_banned("127.0.0.1", liar.p2p.node_id)
        assert fresh.chain.first_block == 0                               # replayed from genesis
        assert fresh.chain.state.root() == honest.chain.state.root()
    finally:
        for n in (fresh, honest, liar):
            n.stop()
