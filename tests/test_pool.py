"""Non-custodial pools: a block's reward split among payees by consensus, and a
pool whose blocks pay its miners directly."""

import socket
import threading
import time
import urllib.error

import pytest

from green import params, pool as poolmod
from green.chain import InvalidBlock, make_genesis, payouts, valid_miner
from green.crypto import Key
from green.node import Node, http_get, http_post
from green.pool import Pool, PoolMiner
from helpers import T0, grind, new_chain, tx


def test_a_block_can_pay_several_miners_by_weight():
    alice, bob, carol = (Key.generate() for _ in range(3))
    chain = new_chain(alice)
    payees = sorted([[bob.address, 3], [carol.address, 1]])
    pay = tx(chain, alice, "transfer", to=bob.address, amount=1)
    header, chosen, _ = chain.build_template(payees, [pay], now=T0 + 10)
    block, _ = grind(chain, header, chosen)
    bob_before = chain.state.balance(bob.address)
    assert chain.add_block(block) == "added"
    total = params.block_reward(1) + pay["fee"]
    split = dict(payouts(payees, total))
    assert sum(split.values()) == total and split[bob.address] >= 3 * split[carol.address]
    assert chain.state.balance(carol.address) == split[carol.address]
    assert chain.state.balance(bob.address) == bob_before + 1 + split[bob.address]
    assert chain.state.root() == block["header"]["state_root"]


def test_payee_lists_have_one_valid_form():
    a, b, c = sorted(Key.generate().address for _ in range(3))
    assert valid_miner(a) and valid_miner([[a, 1], [b, 2]])
    for bad in ([[a, 1]],                                   # one payee: use the plain address
                [[b, 1], [a, 1]],                           # not sorted
                [[a, 1], [a, 2]],                           # duplicate
                [[a, 0], [b, 1]], [[a, True], [b, 1]],      # weights are positive integers
                [[a, params.MAX_PAYEE_WEIGHT + 1], [b, 1]],
                [[k, 1] for k in sorted(Key.generate().address for _ in range(params.MAX_PAYEES + 1))],
                [[a, 1, 0], [b, 1]], "grn1notanaddress"):
        assert not valid_miner(bad), bad
    assert payouts([[a, 1], [b, 1], [c, 1]], 10) == [(a, 4), (b, 3), (c, 3)]   # the remainder goes first


def test_a_block_with_a_malformed_payee_list_is_rejected():
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    header, chosen, _ = chain.build_template(alice.address, [], now=T0 + 10)
    header["miner"] = [[bob.address, 1]]
    block, _ = grind(chain, header, chosen)
    with pytest.raises(InvalidBlock, match="miner"):
        chain.add_block(block)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def pool_node(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET // 8)   # about one share in 8 is a block
    operator = Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(operator.address, "pool"), port=free_port(), log=lambda m: None)
    node.start()
    pool = Pool(node, operator.address, fee_percent=2)
    pool.start()
    yield node, pool, operator
    pool.stop()
    node.stop()


def test_a_pool_pays_its_miners_in_the_blocks_themselves(pool_node):
    node, pool, operator = pool_node
    m1, m2 = Key.generate(), Key.generate()
    miners = [PoolMiner(f"http://127.0.0.1:{pool.port}", m.address, log=lambda m: None) for m in (m1, m2)]
    threads = [threading.Thread(target=m.run, args=(120,), daemon=True) for m in miners]
    for t in threads:
        t.start()

    def paid_both():
        for h in node.chain.main[1:]:
            miner = node.chain.header(h)["miner"]
            if isinstance(miner, list) and {m1.address, m2.address, operator.address} <= {a for a, _ in miner}:
                return True
        return False
    end = time.time() + 120
    while not paid_both():
        assert time.time() < end, f"no block paying both miners; pool {pool.status()}"
        time.sleep(0.2)
    for m in miners:
        m.stop()
    for t in threads:
        t.join(15)
    assert node.chain.state.balance(m1.address) > 0 and node.chain.state.balance(m2.address) > 0
    assert pool.stats["shares"] > pool.stats["blocks"] >= 1                    # most shares are not blocks
    assert http_get(f"http://127.0.0.1:{pool.port}/status")["miners"] == 2


def test_bad_shares_are_refused(pool_node, monkeypatch):
    node, pool, operator = pool_node
    monkeypatch.setattr(poolmod, "BAD_SHARE_LIMIT", 3)
    url = f"http://127.0.0.1:{pool.port}"
    t = http_get(f"{url}/template")
    miner = Key.generate().address
    header = dict(t["header"], work={"job": None, "unit": None}, nonce=1)
    transcript, ticket = PoolMiner(url, miner)._ticket(header)
    good = {"id": t["id"], "address": miner, "nonce": 1, "work": header["work"], "transcript": transcript}

    def refused(share, code=400):
        with pytest.raises(urllib.error.HTTPError) as e:
            http_post(f"{url}/share", share)
        assert e.value.code == code
    assert http_post(f"{url}/share", good)["accepted"]
    refused(good)                                                                 # duplicate
    refused(dict(good, nonce=2))                                                  # transcript of another nonce
    refused(dict(good, id="ab" * 32))                                             # unknown template
    refused(dict(good, nonce=3), code=403)                                        # paused after 3 bad ones


def test_a_pool_miner_survives_the_pool_going_away(pool_node):
    node, pool, operator = pool_node
    url = f"http://127.0.0.1:{pool.port}"
    logs = []
    miner = PoolMiner(url, Key.generate().address, log=logs.append)
    t = threading.Thread(target=miner.run, args=(30,), daemon=True)
    t.start()
    end = time.time() + 20
    while miner.shares == 0:
        assert time.time() < end, "no share accepted"
        time.sleep(0.1)
    pool.stop()                                        # the pool restarts...
    time.sleep(3)
    assert t.is_alive() and any("retrying" in m for m in logs)
    restarted = Pool(node, operator.address, port=pool.port)
    restarted.start()                                  # ...and the miner carries on with it
    before = miner.shares
    end = time.time() + 20
    while miner.shares == before:
        assert time.time() < end, "the miner did not come back to the restarted pool"
        time.sleep(0.1)
    miner.stop()
    t.join(15)
    restarted.stop()
