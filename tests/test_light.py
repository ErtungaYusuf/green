"""Transaction and address index, inclusion proofs, and a light client that checks
headers and proofs itself instead of trusting the node."""

import socket

import pytest

from green import chain as chainmod
from green.chain import BlobStore, Chain, InvalidBlock, make_genesis
from green.crypto import Key, merkle_verify
from green.light import LightClient
from green.node import Node, http_get, http_post
from green.store import Store
from green.tx import txid
from helpers import T0, add, mine, tx


def test_the_index_follows_the_main_chain_through_a_reorg(tmp_path):
    alice, bob, carol = Key.generate(), Key.generate(), Key.generate()
    store = Store(str(tmp_path / "chain.db"))
    chain = Chain(make_genesis(alice.address, "index", timestamp=T0), BlobStore(), store=store)
    add(chain, alice)
    fork = chain.tip
    pay = tx(chain, alice, "transfer", to=bob.address, amount=7)
    add(chain, alice, [pay])
    height, block, pos = store.tx_location(txid(pay))
    assert (height, block, pos) == (2, chain.tip, 0)
    roles = {(r[3], r[4]) for r in store.address_history(bob.address)}
    assert roles == {("recipient", "transfer")}
    assert {r[3] for r in store.address_history(alice.address)} == {"sender", "miner"}
    # A heavier branch from before the payment, paying carol instead.
    other = tx(chain, alice, "transfer", to=carol.address, amount=9, nonce=0)
    parent = fork
    for i in range(3):
        block, _ = mine(chain, alice, [other] if i == 0 else [], t=T0 + 50_000 + i, parent=parent)
        assert chain.add_block(block) == "added"
        parent = chainmod.block_hash(block)
    assert chain.tip == parent
    assert store.tx_location(txid(pay)) is None                 # gone with the abandoned block
    assert store.address_history(bob.address) == []
    assert store.tx_location(txid(other))[0] == 2
    assert [r[3] for r in store.address_history(carol.address)] == ["recipient"]
    store.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served(tmp_path):
    keys = [Key.generate() for _ in range(3)]
    genesis = make_genesis(keys[0].address, "light", timestamp=T0)
    node = Node(str(tmp_path / "n"), genesis, port=free_port(), log=lambda m: None)
    pays = []
    for i in range(30):
        t = tx(node.chain, keys[0], "transfer", to=keys[1 + i % 2].address, amount=100 + i)
        pays.append(t)
        add(node.chain, keys[0], [t])
    node.start()
    yield node, genesis, keys, pays
    node.stop()


def test_explorer_endpoints(served):
    node, genesis, keys, pays = served
    v = http_get(f"{node.url}/v1/tx/{txid(pays[3])}")
    assert v["status"] == "confirmed" and v["tx"] == pays[3] and v["confirmations"] == node.chain.height - 4 + 1
    h = http_get(f"{node.url}/v1/address/{keys[1].address}/history?limit=5")
    assert len(h["items"]) == 5 and h["items"][0]["height"] > h["items"][-1]["height"] and h["next"] is not None
    more = http_get(f"{node.url}/v1/address/{keys[1].address}/history?limit=50&before={h['next']}")
    assert len(h["items"]) + len(more["items"]) == 15                 # every other payment went to keys[1]
    # A payment still waiting is listed with the role it will have once confirmed.
    with node.lock:
        node._rebuild_pending()                  # (the fixture built its blocks behind the node's back)
    waiting = tx(node.chain, keys[0], "transfer", to=keys[2].address, amount=7)
    http_post(f"{node.url}/tx", waiting)
    assert http_get(f"{node.url}/v1/address/{keys[2].address}/history")["pending"] == \
        [{"txid": txid(waiting), "type": "transfer", "role": "recipient"}]
    assert http_get(f"{node.url}/v1/address/{keys[0].address}/history")["pending"][0]["role"] == "sender"
    p = http_get(f"{node.url}/v1/proof/tx/{txid(pays[10])}")
    assert merkle_verify(p["txid"], p["index"], p["count"], p["proof"], p["header"]["tx_root"])
    assert [x["height"] for x in http_get(f"{node.url}/v1/headers?from=5&count=3")] == [5, 6, 7]


def test_a_light_client_verifies_payments_itself(served):
    node, genesis, keys, pays = served
    light = LightClient(node.url, genesis)
    assert light.sync() == node.chain.height
    r = light.verify_payment(txid(pays[5]), min_confirmations=6)
    assert r["confirmations"] == node.chain.height - 6 + 1
    with pytest.raises(ValueError, match="confirmations"):
        light.verify_payment(txid(pays[-1]), min_confirmations=6)


def test_a_light_client_catches_a_lying_node(served, monkeypatch):
    node, genesis, keys, pays = served
    light = LightClient(node.url, genesis)
    light.sync()
    real = node.tx_proof
    monkeypatch.setattr(node, "tx_proof", lambda t: {**real(txid(pays[2])), "txid": t})   # a proof for another tx
    with pytest.raises(ValueError, match="tx_root"):
        light.verify_payment(txid(pays[7]))
    monkeypatch.setattr(node, "tx_proof", real)       # (not undo(): that would also undo the test target)

    fresh = LightClient(node.url, genesis)
    good = node.headers

    def forged(start, count):                         # the newest header claims work never done
        out = good(start, count)
        return out[:-1] + [dict(out[-1], transcript="00" * 32)] if out else out
    monkeypatch.setattr(node, "headers", forged)
    with pytest.raises(InvalidBlock, match="work"):
        fresh.sync()
    assert fresh.height == 0 and len(fresh.headers) == 1        # nothing of the bad batch was kept


def test_an_older_data_folder_gets_its_index_built_on_open(tmp_path):
    alice, bob = Key.generate(), Key.generate()
    genesis = make_genesis(alice.address, "reindex", timestamp=T0)
    store = Store(str(tmp_path / "chain.db"))
    chain = Chain(genesis, BlobStore(), store=store)
    pays = []
    for _ in range(5):
        pays.append(tx(chain, alice, "transfer", to=bob.address, amount=3))
        add(chain, alice, [pays[-1]])
    store.clear_index()                                          # as if made by v0.11
    store.set_meta("txindex", None)
    store.close()
    store = Store(str(tmp_path / "chain.db"))
    Chain(genesis, BlobStore(), store=store)
    assert [store.tx_location(txid(p))[0] for p in pays] == [1, 2, 3, 4, 5]
    assert len(store.address_history(bob.address)) == 5
    store.close()


def test_a_light_client_follows_a_reorganization_with_more_work(served):
    """Blocks come every 15 s, so the chain a light client follows reorganizes now and
    then. It used to stop for good ("headers do not extend the chain"); it now finds the
    fork, checks the new branch and takes it, but only if it has more work."""
    node, genesis, keys, pays = served
    light = LightClient(node.url, genesis)
    light.sync()
    old_tip, fork = node.chain.tip, node.chain.main[node.chain.height - 2]
    parent = fork
    with node.lock:
        for i in range(4):                                      # a heavier branch off 3 blocks back
            block, _ = mine(node.chain, keys[0], t=T0 + 90_000 + i, parent=parent)
            assert node.chain.add_block(block) == "added"
            parent = chainmod.block_hash(block)
    assert node.chain.tip == parent and not node.chain.on_main(old_tip)
    assert light.sync() == node.chain.height and light.by_height[-1] == node.chain.tip
    assert light.verify_payment(txid(pays[5]))["confirmations"] == node.chain.height - 6 + 1


def test_a_light_client_keeps_its_chain_against_a_lighter_branch(served, monkeypatch):
    node, genesis, keys, pays = served
    light = LightClient(node.url, genesis)
    light.sync()
    mine_tip, height = light.by_height[-1], light.height
    side, parent = [], node.chain.main[height - 2]
    for i in range(2):                                          # shorter than ours: less work
        block, _ = mine(node.chain, keys[0], t=T0 + 95_000 + i, parent=parent)
        assert node.chain.add_block(block) == "added"
        side.append(block["header"])
        parent = chainmod.block_hash(block)
    good = node.headers

    def lying(start, count):                                    # serves the lighter branch as its chain
        out = [h for h in good(start, count) if h["height"] < height - 1]
        return out + side if start <= height - 1 else [dict(side[-1], height=height + 1, prev="00" * 32)]
    monkeypatch.setattr(node, "headers", lying)
    with pytest.raises(InvalidBlock, match="more work"):
        light.sync()
    assert light.by_height[-1] == mine_tip and light.height == height
