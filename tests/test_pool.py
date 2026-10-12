"""Non-custodial pools: a block's reward split among payees by consensus, and a
pool whose blocks pay its miners directly."""

import base64
import socket
import threading
import time
import urllib.error

import numpy as np
import pytest

from green import params, pool as poolmod, tx as txmod
from green.chain import InvalidBlock, make_genesis, payouts, valid_miner
from green.crypto import Key
from green.node import Node, http_get, http_post
from green.pool import Pool, PoolMiner
from green.tx import InvalidTx, make_tx
from helpers import T0, add, commit, grind, minted, new_chain, post_job, sell, supply, tx, unit_result


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
    # A share that is not also a block: a block would make the template stale for the rest.
    for nonce in range(10, 1000):
        header = dict(t["header"], work={"job": None, "unit": None}, nonce=nonce)
        transcript, ticket = PoolMiner(url, miner)._ticket(header)
        if ticket > int(t["header"]["target"], 16):
            break
    good = {"id": t["id"], "address": miner, "nonce": nonce, "work": header["work"], "transcript": transcript}

    def refused(share, code=400):
        with pytest.raises(urllib.error.HTTPError) as e:
            http_post(f"{url}/share", share)
        assert e.value.code == code
    assert http_post(f"{url}/share", good)["accepted"]
    refused(good)                                                                 # duplicate
    refused(dict(good, nonce=2))                                                  # transcript of another nonce
    assert http_post(f"{url}/share", dict(good, id="ab" * 32)) == {"accepted": False, "stale": True}
    refused(dict(good, address="grn1notanaddress"))                               # not held against: stale
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


# --- useful work through a pool: batches that pay the miners who computed them ---

def test_a_batch_can_pay_the_miners_who_computed_it():
    pool, buyer, m1, m2 = (Key.generate() for _ in range(4))
    chain = new_chain(pool)
    add(chain, pool, [tx(chain, pool, "transfer", to=buyer.address, amount=50_000_000)])
    a, b, job_id = post_job(chain, buyer, np.random.default_rng(7), dims=(1, 2, 1), price=1000)
    units = [[0, 0, 0], [0, 1, 0]]
    payees = sorted([[m1.address, 3], [m2.address, 1]])
    bid = sell(chain, pool, job_id, units, [unit_result(a, b, u) for u in units], payees=payees)
    split = dict(payouts(payees, 2 * 1000))
    assert chain.state.balance(m1.address) == split[m1.address] == 1500
    assert chain.state.balance(m2.address) == split[m2.address] == 500
    assert chain.state.jobs[job_id]["delivered"][bid]["payees"] == payees
    assert supply(chain) == minted(chain)               # the bond went back, nothing was created or lost


def test_batch_payees_have_one_valid_form():
    pool, buyer, m1 = (Key.generate() for _ in range(3))
    chain = new_chain(pool)
    add(chain, pool, [tx(chain, pool, "transfer", to=buyer.address, amount=50_000_000)])
    a, b, job_id = post_job(chain, buyer, np.random.default_rng(8), dims=(1, 1, 1))
    r = [unit_result(a, b, [0, 0, 0])]
    for bad in ([[m1.address, 1]], sorted([[m1.address, 1], [pool.address, 0]]),
                [[m1.address, 1], [m1.address, 2]], "not a list"):
        t, _ = commit(chain, pool, job_id, [[0, 0, 0]], r, payees=bad)
        with pytest.raises(InvalidTx, match="payees"):
            chain.apply_tx(chain.state.copy(), t, chain.height + 1)


def wait_until(cond, what, timeout=60, pool=None):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, f"timed out waiting for {what}" + (f"; pool {pool.status()}" if pool else "")
        time.sleep(0.2)


@pytest.fixture
def selling_pool(tmp_path, monkeypatch):
    # About a block a second: a commit must be revealed within RESERVE_BLOCKS (20) blocks,
    # five minutes on a real network; much faster blocks would race the pool's seller.
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET // 64)
    monkeypatch.setattr(params, "JOB_CHALLENGE_BLOCKS", 4)
    operator = Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(operator.address, "selling pool"), port=free_port(),
                miner_key=operator, log=lambda m: None)
    node.start()
    pool = Pool(node, operator.address, fee_percent=2)
    pool.start()
    yield node, pool, operator
    pool.stop()
    node.stop()


def post_job_on(node, buyer, a, b, price=1000):
    job_id = node.blobs.put(txmod.encode_job(a, b))
    dims = txmod.job_dims_units(a, b)
    node.submit_tx(make_tx(buyer, node.chain.chain_id, "job_post", node.pending_nonce(buyer.address),
                           txmod.min_fee({"type": "job_post", "dims": dims}), job=job_id, dims=dims, price=price))
    return job_id


def test_pool_miners_compute_jobs_and_the_batches_pay_them(selling_pool):
    node, pool, operator = selling_pool
    buyer, m1, m2 = Key.generate(), Key.generate(), Key.generate()
    miners = [PoolMiner(f"http://127.0.0.1:{pool.port}", m.address, log=lambda m: None) for m in (m1, m2)]
    threads = [threading.Thread(target=m.run, args=(180,), daemon=True) for m in miners]
    for t in threads:
        t.start()
    node.submit_tx(make_tx(operator, node.chain.chain_id, "transfer", 0, to=buyer.address, amount=50_000_000))
    wait_until(lambda: node.chain.state.balance(buyer.address) > 0, "the buyer's funds")
    rng = np.random.default_rng(9)
    n = params.UNIT
    a = rng.integers(-128, 128, size=(n, 3 * n), dtype=np.int8)
    b = rng.integers(-128, 128, size=(3 * n, n), dtype=np.int8)
    job_id = post_job_on(node, buyer, a, b)
    wait_until(lambda: len(node.chain.state.jobs.get(job_id, {}).get("done", {})) == 3,
               "all 3 units sold and paid", timeout=150, pool=pool)
    for m in miners:
        m.stop()
    for t in threads:
        t.join(15)
    job = node.chain.state.jobs[job_id]
    paid: dict[str, int] = {}
    for record in job["delivered"].values():
        assert isinstance(record.get("payees"), list), record       # paid by the batch, not to the pool
        for addr, amount in payouts(record["payees"], 1000 * record["count"]):
            paid[addr] = paid.get(addr, 0) + amount
    assert sum(paid.values()) == 3000 and set(paid) <= {m1.address, m2.address, operator.address}
    assert paid[operator.address] == 60                              # the pool's 2 percent, nothing more
    assert pool.stats["useful_units"] == 3 == miners[0].useful + miners[1].useful
    assert pool.stats["useful_shares"] >= 1          # a job's unit is a lottery ticket like any other
    for record in job["delivered"].values():                        # and the buyer gets the right products
        leaves = txmod.decode_manifest(node.blobs.get(record["manifest"]), record["count"])
        for unit, leaf in zip(txmod.bitmap_units(record["units"], [1, 3, 1]), leaves):
            assert np.array_equal(txmod.decode_result(node.blobs.get(leaf)), unit_result(a, b, unit))


def test_wrong_or_unassigned_results_are_refused(selling_pool):
    node, pool, operator = selling_pool
    url = f"http://127.0.0.1:{pool.port}"
    one = np.ones((params.UNIT, params.UNIT), dtype=np.int8)
    job_id = post_job_on(node, operator, one, one)
    miner = Key.generate()
    pool.useful = False                          # a helper mines the job's block, without taking its unit
    m = PoolMiner(url, Key.generate().address, log=lambda m: None)
    t = threading.Thread(target=m.run, args=(60,), daemon=True)
    t.start()
    wait_until(lambda: job_id in node.chain.state.jobs, "the job on chain")
    m.stop()
    t.join(15)
    pool.useful = True
    units = []
    wait_until(lambda: units.extend(http_get(f"{url}/template?address={miner.address}")["units"]) or units,
               "a unit assigned", timeout=30)
    job, unit = units[0]

    def refused(body):
        with pytest.raises(urllib.error.HTTPError) as e:
            http_post(f"{url}/result", body)
        assert e.value.code == 400
    wrong = base64.b64encode(txmod.encode_result(np.full((params.UNIT, params.UNIT), 7))).decode()
    refused({"address": miner.address, "job": job, "unit": unit, "result": wrong})                 # wrong product
    right = base64.b64encode(txmod.encode_result(unit_result(one, one, unit))).decode()
    refused({"address": Key.generate().address, "job": job, "unit": unit, "result": right})        # not theirs
    assert http_post(f"{url}/result", {"address": miner.address, "job": job, "unit": unit, "result": right})["accepted"]
    assert pool.stats["useful_units"] == 1 and pool.stats["rejected"] == 2


def test_a_batch_pays_at_most_31_contributors_and_the_rest_wait(tmp_path):
    operator, buyer = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(operator.address, "cap", timestamp=T0), port=free_port(),
                miner_key=operator, log=lambda m: None)
    try:
        chain = node.chain
        add(chain, operator, [tx(chain, operator, "transfer", to=buyer.address, amount=90_000_000)])
        a, b, job_id = post_job(chain, buyer, np.random.default_rng(3), dims=(2, 4, 5), price=1000)
        node._rebuild_pending()
        units = txmod.all_units([2, 4, 5])                     # 40 units by 40 different miners
        people = [Key.generate().address for _ in units]
        for u, who in zip(units, people):
            node._buffer_result({"job": job_id, "unit": u}, unit_result(a, b, u), by=who)
        node._flush_batches()
        commit_tx = next(t for t in node.mempool.values() if t["type"] == "job_commit")
        paid = {addr for addr, _ in commit_tx["payees"]}
        assert len(paid) == params.MAX_PAYEES - 1 and paid <= set(people)
        left = node.batch_buf[job_id]                          # owed: kept for the next batch
        assert len(left["units"]) == 40 - 31 and set(left["by"]) == set(people) - paid
    finally:
        node.close()
