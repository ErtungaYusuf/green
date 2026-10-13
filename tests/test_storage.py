"""The diff-layer state and the SQLite store: correct across reorgs and base
moves, restartable without re-executing anything, and crash-safe."""

import random

import numpy as np
import pytest

from green import chain as chainmod, params
from green.chain import BlobStore, Chain, InvalidBlock, block_hash, initial_state, make_genesis
from green.crypto import Key
from green.node import Node
from green.state import State
from green.store import Store
from helpers import T0, add, mine, post_job, sell, tx, unit_result


@pytest.fixture(autouse=True)
def short_windows(monkeypatch):
    """A small reorg window so the base moves many times in a short test."""
    monkeypatch.setattr(params, "MAX_REORG_DEPTH", 4)
    monkeypatch.setattr(chainmod, "BASE_MARGIN", 2)
    monkeypatch.setattr(chainmod, "HEADER_KEEP", 3)
    monkeypatch.setattr(params, "ASERT_HALFLIFE", 10**9)         # practically constant difficulty
    for name, value in (("RESERVE_BLOCKS", 3), ("JOB_CHALLENGE_BLOCKS", 3), ("JOB_RESPONSE_BLOCKS", 2)):
        monkeypatch.setattr(params, name, value)


def reference(chain: Chain) -> State:
    """The main chain replayed from genesis with full, flat copies (the v0.4 way)."""
    s = chain.apply_block(initial_state(), chain.block(chain.genesis_hash)).flattened()
    for bh in chain.main[1:]:
        s = chain.apply_block(s, chain.block(bh)).flattened()
    return s


def activity(chain, keys, pick, rng, jobs):
    """One random block's worth of transactions on top of the current tip."""
    actor = pick.choice(keys)
    roll = pick.random()
    if roll < 0.15 and len(jobs) < 6:
        a, b, job_id = post_job(chain, actor, rng, dims=(1, 1, pick.randint(1, 2)), price=pick.randint(1, 5000))
        jobs[job_id] = (a, b)
        return []
    live = [j for j in jobs if j in chain.state.jobs]      # a reorg can drop the block that posted a job
    if roll < 0.3 and live:
        job_id = pick.choice(live)
        units = chain.state.open_units(job_id)
        if units:
            a, b = jobs[job_id]
            sell(chain, actor, job_id, units, [unit_result(a, b, u) for u in units], settle=False)
            return []
    return [tx(chain, actor, "transfer", to=pick.choice(keys).address, amount=pick.randint(1, 1000))]


def reorg(chain, keys, pick):
    """Replace the last few blocks with a heavier branch."""
    depth = pick.randint(1, params.MAX_REORG_DEPTH - 1)
    parent = chain.main[-1 - depth]
    old_tip = chain.tip
    for i in range(depth + 1):
        block, _ = mine(chain, pick.choice(keys), [], t=T0 + 100_000 + 37 * len(chain.blocks) + i, parent=parent)
        assert chain.add_block(block) == "added"
        parent = block_hash(block)
    assert chain.tip == parent and chain.tip != old_tip


def build(chain, keys, steps, seed):
    pick, rng = random.Random(seed), np.random.default_rng(seed)
    jobs = {}
    for k in keys[1:]:
        add(chain, keys[0], [tx(chain, keys[0], "transfer", to=k.address, amount=10**7)])
    for step in range(steps):
        if step % 9 == 8:
            reorg(chain, keys, pick)
        else:
            add(chain, pick.choice(keys), activity(chain, keys, pick, rng, jobs))
    return jobs


def test_layered_state_matches_a_full_copy_replay():
    keys = [Key.generate() for _ in range(4)]
    chain = Chain(make_genesis(keys[0].address, "layers", timestamp=T0), BlobStore())
    build(chain, keys, 80, seed=1)
    assert chain.base_height > 30                      # the base moved many times
    assert chain.state.depth() <= params.MAX_REORG_DEPTH + chainmod.BASE_MARGIN + 2
    assert chain.state.fingerprint() == reference(chain).fingerprint()


def test_restart_opens_the_same_chain_without_re_executing(tmp_path, monkeypatch):
    keys = [Key.generate() for _ in range(4)]
    genesis = make_genesis(keys[0].address, "restart", timestamp=T0)
    blobs = BlobStore(str(tmp_path / "blobs"))
    store = Store(str(tmp_path / "chain.db"))
    chain = Chain(genesis, blobs, store=store)
    build(chain, keys, 50, seed=2)
    tip, height, base, print_ = chain.tip, chain.height, chain.base_height, chain.state.fingerprint()
    root = chain.state.root()
    store.close()

    def no_replay(*args, **kwargs):
        raise AssertionError("opening the chain re-executed a block")
    real = Chain.apply_block
    monkeypatch.setattr(Chain, "apply_block", no_replay)
    store = Store(str(tmp_path / "chain.db"))
    reopened = Chain(genesis, blobs, store=store)
    # Restore just this one; monkeypatch.undo() would also undo the fixtures' settings
    # (the easy test target among them), which made the rest of this test grind ~200x.
    monkeypatch.setattr(Chain, "apply_block", real)
    assert (reopened.tip, reopened.height, reopened.base_height) == (tip, height, base)
    assert reopened.state.fingerprint() == print_
    assert reopened.state.root() == root == reopened.tip_header()["state_root"]
    base_state = reopened.states[reopened.base_hash]     # the stored flat base keeps its commitment
    assert np.array_equal(base_state.lattice, base_state.recompute_lattice())
    # It keeps working: more blocks and reorgs on top of the reopened chain.
    pick = random.Random(3)
    for i in range(20):
        if i % 7 == 6:
            reorg(reopened, keys, pick)
        else:
            add(reopened, pick.choice(keys), [tx(reopened, keys[0], "transfer", to=keys[1].address, amount=1)])
    assert reopened.state.fingerprint() == reference(reopened).fingerprint()
    store.close()


def test_a_crash_mid_block_leaves_the_last_complete_block(tmp_path, monkeypatch):
    keys = [Key.generate() for _ in range(2)]
    genesis = make_genesis(keys[0].address, "crash", timestamp=T0)
    store = Store(str(tmp_path / "chain.db"))
    chain = Chain(genesis, BlobStore(), store=store)
    for _ in range(12):
        add(chain, keys[0], [tx(chain, keys[0], "transfer", to=keys[1].address, amount=5)])
    tip, print_ = chain.tip, chain.state.fingerprint()
    block, _ = mine(chain, keys[0], [tx(chain, keys[0], "transfer", to=keys[1].address, amount=5)])

    def power_cut(*args, **kwargs):
        raise OSError("power cut while writing")
    monkeypatch.setattr(store, "put_diff", power_cut)          # after the block row, before the commit
    with pytest.raises(OSError):
        chain.add_block(block)
    store.close()
    store = Store(str(tmp_path / "chain.db"))
    reopened = Chain(genesis, BlobStore(), store=store)
    assert reopened.tip == tip and not reopened.has_block(block_hash(block))
    assert reopened.state.fingerprint() == print_
    assert reopened.add_block(block) == "added"                # and the block can be accepted again
    store.close()


def test_old_blocks_leave_memory_but_stay_available(tmp_path):
    keys = [Key.generate() for _ in range(2)]
    store = Store(str(tmp_path / "chain.db"))
    chain = Chain(make_genesis(keys[0].address, "old", timestamp=T0), BlobStore(), store=store)
    for _ in range(40):
        add(chain, keys[0])
    first = chain.main[1]
    assert first not in chain.blocks                           # evicted from memory...
    assert chain.has_block(first) and block_hash(chain.block(first)) == first     # ...served from the store
    assert block_hash(chain.block_at(1)) == first
    assert len(chain.blocks) <= params.MAX_REORG_DEPTH + chainmod.BASE_MARGIN + chainmod.HEADER_KEEP + 3
    assert chain.add_block(chain.block(first)) == "exists"     # an old block re-sent by a peer
    store.close()


def test_side_chain_that_forked_too_deep_is_refused_block_by_block():
    """S11 was meant to refuse forks that leave the main chain more than
    MAX_REORG_DEPTH blocks below the tip, but v0.4 only checked the parent's
    height, so a side chain started long ago could keep growing next to the
    main chain. The rule now looks at the fork point."""
    keys = [Key.generate() for _ in range(2)]
    chain = Chain(make_genesis(keys[0].address, "deep", timestamp=T0), BlobStore())
    add(chain, keys[0])
    fork = chain.main[1]
    add(chain, keys[0])
    add(chain, keys[0])
    side = fork
    for n in range(2, 12):                    # the side chain stays one block behind the main chain
        block, _ = mine(chain, keys[1], [], t=T0 + 10 * n + 5, parent=side)
        try:
            chain.add_block(block)
        except InvalidBlock as e:
            assert "too deep" in str(e) and not e.permanent
            assert chain.height - 1 > params.MAX_REORG_DEPTH          # refused once the fork point is buried
            return
        side = block_hash(block)
        add(chain, keys[0])
    raise AssertionError("a side chain forked far below the tip kept being accepted")


def test_a_node_keeps_its_chain_and_identity_across_restarts(tmp_path):
    keys = [Key.generate() for _ in range(2)]
    genesis = make_genesis(keys[0].address, "restart node", timestamp=T0)
    d = str(tmp_path / "node")
    node = Node(d, genesis, port=0, miner_key=keys[0], log=lambda m: None)
    for _ in range(15):
        add(node.chain, keys[0], [tx(node.chain, keys[0], "transfer", to=keys[1].address, amount=3)])
    tip, root, node_id = node.chain.tip, node.chain.state.root(), node.p2p.node_id
    node.close()
    again = Node(d, genesis, port=0, miner_key=keys[0], log=lambda m: None)
    assert again.chain.tip == tip and again.chain.state.root() == root
    assert again.p2p.node_id == node_id                  # the P2P identity is kept in node_key
    again.close()


def test_a_data_folder_of_another_network_is_refused(tmp_path):
    a, b = Key.generate(), Key.generate()
    store = Store(str(tmp_path / "chain.db"))
    Chain(make_genesis(a.address, "net A", timestamp=T0), BlobStore(), store=store)
    with pytest.raises(ValueError, match="another network"):
        Chain(make_genesis(b.address, "net B", timestamp=T0), BlobStore(), store=store)
    store.close()
