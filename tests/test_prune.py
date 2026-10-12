"""Pruned nodes: old block bodies go, headers and the ledger stay, data still in
use is kept, unused data is deleted only after a retention period, and a pruned
node still lets new nodes join by fast sync."""

import socket
import time

import numpy as np
import pytest

from green import chain as chainmod, params, prune as prunemod
from green.chain import BlobStore, Chain, make_genesis
from green.crypto import Key
from green.node import Node
from green.store import Store
from green.tx import make_tx
from helpers import T0, add, post_job, tx
from test_storage import reorg


@pytest.fixture(autouse=True)
def short_windows(monkeypatch):
    monkeypatch.setattr(params, "MAX_REORG_DEPTH", 4)
    monkeypatch.setattr(chainmod, "BASE_MARGIN", 2)
    monkeypatch.setattr(chainmod, "HEADER_KEEP", 3)              # minimum kept bodies: 4 + 2 + 3 + 1 = 10
    monkeypatch.setattr(params, "ASERT_HALFLIFE", 10**9)
    monkeypatch.setattr(params, "SNAPSHOT_INTERVAL", 10)
    monkeypatch.setattr(params, "CANCEL_DELAY", 2)


def busy_block(chain, keys, n=20):
    sender = keys[0]
    nonce = chain.state.nonce(sender.address)
    add(chain, keys[1], [make_tx(sender, chain.chain_id, "transfer", nonce + i, to=keys[1 + i % 3].address, amount=1)
                         for i in range(n)])


def test_old_bodies_go_headers_stay_and_the_chain_goes_on(tmp_path):
    import random
    keys = [Key.generate() for _ in range(4)]
    genesis = make_genesis(keys[0].address, "prune", timestamp=T0)
    store = Store(str(tmp_path / "chain.db"))
    chain = Chain(genesis, BlobStore(), store=store)
    for _ in range(120):
        busy_block(chain, keys)
    pages_before = store.size_pages()[0]
    body_bytes = store._q("SELECT SUM(LENGTH(body) - LENGTH(header)) FROM blocks WHERE height < ?",
                          (chain.height - 20,))[0][0]
    with pytest.raises(ValueError, match="at least 10"):
        chain.prune(5)
    assert chain.prune(20) == 100                                # heights 0..99
    assert chain.prune(20) == 0                                  # nothing new to drop
    assert chain.first_block == chain.height - 20
    old = chain.main[50]
    assert chain.block(old) is None and chain.header(old)["height"] == 50
    assert chain.block(chain.main[-1]) is not None and chain.block(chain.main[-20]) is not None
    pages, free = store.size_pages()
    page = store._q("PRAGMA page_size")[0][0]
    # The space of the dropped transactions went back to the disk (at least half of it:
    # pages are shared with headers and indexes), and nothing is left on the free list.
    assert (pages_before - pages) * page >= body_bytes / 2 and free == 0
    reorg(chain, keys, random.Random(1))                         # the window above is untouched
    busy_block(chain, keys)
    tip, root, first = chain.tip, chain.state.root(), chain.first_block
    store.close()
    again = Chain(genesis, BlobStore(), store=Store(str(tmp_path / "chain.db")))
    assert (again.tip, again.state.root(), again.first_block) == (tip, root, first)
    again.store.close()


def test_unused_data_is_deleted_only_after_the_retention_period(tmp_path, monkeypatch):
    monkeypatch.setattr(prunemod, "BLOB_RETENTION_BLOCKS", 5)
    key = Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(key.address, "gc", timestamp=T0), port=0, miner_key=key,
                log=lambda m: None, prune=True, prune_keep=10)
    chain = node.chain
    rng = np.random.default_rng(4)
    _, _, open_job = post_job(chain, key, rng)
    _, _, done_job = post_job(chain, key, rng)
    add(chain, key, [tx(chain, key, "job_cancel", job=done_job)])
    for _ in range(3):
        add(chain, key)                                          # the cancel takes effect
    assert chain.state.jobs[done_job]["closed"]
    junk = node.blobs.put(b"nobody needs this")
    a = np.zeros((256, 256), dtype=np.int8)
    from green import tx as txmod
    pending_job = node.blobs.put(txmod.encode_job(a, a))
    with node.lock:
        node._rebuild_pending()                                  # blocks were added to the chain directly
    node.submit_tx(make_tx(key, chain.chain_id, "job_post", chain.state.nonce(key.address),
                           txmod.min_fee({"type": "job_post", "dims": [1, 1, 1]}), job=pending_job,
                           dims=[1, 1, 1], price=10))
    for _ in range(12):
        add(chain, key)                                          # (the mempool job is not mined: add() ignores it)
    assert node.snapshots
    r = node.maintain()
    assert r["marked"] >= 2 and r["deleted"] == 0
    assert node.blobs.has(junk) and node.blobs.has(done_job)     # marked, not deleted yet
    for _ in range(5):
        add(chain, key)
    node.maintain()
    assert not node.blobs.has(junk) and not node.blobs.has(done_job)
    for kept in [open_job, pending_job] + node.snapshots[-1]["chunks"] + [node.snapshots[-1]["id"]]:
        assert node.blobs.has(kept), kept
    assert chain.first_block > 0                                 # bodies were pruned too
    node.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_pruned_node_still_lets_new_nodes_join(tmp_path):
    keys = [Key.generate() for _ in range(4)]
    genesis = make_genesis(keys[0].address, "pruned source", timestamp=T0)
    source = Node(str(tmp_path / "s"), genesis, port=free_port(), log=lambda m: None, prune=True, prune_keep=12)
    add(source.chain, keys[0], [tx(source.chain, keys[0], "transfer", to=keys[1].address, amount=10**6)])
    for _ in range(70):
        busy_block(source.chain, keys, n=2)
    source.maintain()
    assert source.chain.first_block == source.chain.height - 12 and source.snapshots
    source.start()
    fresh = Node(str(tmp_path / "f"), genesis, port=free_port(), log=lambda m: None, fast_sync=True,
                 peers=[source.p2p.address])
    fresh.start()
    try:
        end = time.time() + 60
        while fresh.chain.tip != source.chain.tip:
            assert time.time() < end, "fast sync from a pruned node did not finish"
            time.sleep(0.1)
        assert fresh.p2p.snap_state == "done" and fresh.chain.state.root() == source.chain.state.root()
    finally:
        fresh.stop()
        source.stop()
