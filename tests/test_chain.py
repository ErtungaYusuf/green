import numpy as np
import pytest

from green import params, tx as txmod
from green.chain import BlobStore, Chain, InvalidBlock, MissingData, block_hash, make_genesis, solve
from green.crypto import Key, canonical, merkle_proof
from green.tx import InvalidTx, make_tx
from helpers import (SALES, T0, add, challenge, commit, grind, minted, mine, new_chain, post_job, respond, reveal, sell,
                     settle_batch, supply, tx, unit_result)

FEE = params.MIN_TX_FEE


def test_genesis_pays_reward():
    alice = Key.generate()
    chain = new_chain(alice)
    assert chain.height == 0
    assert chain.state.balance(alice.address) == params.block_reward(0)


def test_transfer_and_rejections():
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    t = tx(chain, alice, "transfer", to=bob.address, amount=params.YAPRAK_PER_COIN)
    add(chain, bob, [t])
    assert chain.state.balance(bob.address) == params.YAPRAK_PER_COIN + params.block_reward(1) + FEE
    assert chain.state.balance(alice.address) == params.block_reward(0) - params.YAPRAK_PER_COIN - FEE

    s = chain.state.copy()
    h = chain.height + 1
    with pytest.raises(InvalidTx, match="nonce"):
        chain.apply_tx(s, t, h)  # replay
    forged = dict(tx(chain, alice, "transfer", to=bob.address, amount=1), amount=2)
    with pytest.raises(InvalidTx, match="signature"):
        chain.apply_tx(s, forged, h)
    with pytest.raises(InvalidTx, match="balance"):
        chain.apply_tx(s, tx(chain, alice, "transfer", to=bob.address, amount=10**15), h)
    with pytest.raises(InvalidTx, match="fee"):
        chain.apply_tx(s, tx(chain, alice, "transfer", to=bob.address, amount=1, fee=0), h)


def funded(n_keys, amount=2 * 10**7):
    """A chain plus keys; key 0 mines the genesis and funds the rest."""
    keys = [Key.generate() for _ in range(n_keys)]
    chain = new_chain(keys[0])
    add(chain, keys[0], [tx(chain, keys[0], "transfer", to=k.address, amount=amount, nonce=i)
                         for i, k in enumerate(keys[1:])])
    return chain, keys


def test_job_batch_commit_reveal_settle_pays_miner():
    rng = np.random.default_rng(1)
    chain, (buyer, miner) = funded(2)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 2, 1), price=1000)
    assert chain.state.jobs[job_id]["escrow"] == 2000

    # The miner does unit (0, 0, 0) as its lottery ticket and gets A_00 @ B_00 for free.
    block, useful = mine(chain, miner, work={"job": job_id, "unit": [0, 0, 0]})
    assert np.array_equal(useful, unit_result(a, b, [0, 0, 0]))
    chain.add_block(block)

    h0, before = chain.height, chain.state.balance(miner.address)
    units = [[0, 0, 0], [0, 1, 0]]
    commit_tx, sale = commit(chain, miner, job_id, units, [useful, unit_result(a, b, [0, 1, 0])])
    add(chain, miner, [commit_tx])
    bid = commit_tx["commitment"]
    job = chain.state.jobs[job_id]
    assert job["reserved"] == {"0,0,0": bid, "0,1,0": bid}
    assert job["batches"][bid]["bond"] == 2 * params.bond_for(1000)

    add(chain, miner, [reveal(chain, miner, job_id, bid, sale)])
    job = chain.state.jobs[job_id]
    assert job["batches"][bid]["status"] == "revealed" and job["done"] == {}   # not final yet
    assert job["batches"][bid]["root"] == sale.root

    settle_batch(chain, miner, job_id, bid)
    job = chain.state.jobs[job_id]
    assert job["done"] == {"0,0,0": bid, "0,1,0": bid}
    assert job["delivered"][bid]["root"] == sale.root and job["delivered"][bid]["manifest"] == sale.manifest
    assert job["escrow"] == 0 and job["reserved"] == {} and job["batches"] == {}
    # paid the price, got the bond back; its fees went to itself as the block producer
    rewards = sum(params.block_reward(h) for h in range(h0 + 1, chain.height + 1))
    assert chain.state.balance(miner.address) == before + 2000 + rewards
    assert supply(chain) == minted(chain)


def test_a_whole_job_is_delivered_with_two_transactions():
    rng = np.random.default_rng(11)
    chain, (buyer, miner) = funded(2)
    a, b, job_id = post_job(chain, buyer, rng, dims=(2, 2, 2), price=1000)
    units = txmod.all_units([2, 2, 2])
    commit_tx, sale = commit(chain, miner, job_id, units, [unit_result(a, b, u) for u in units])
    assert len(commit_tx["units"]) == 2                 # 8 units: a one-byte bitmap
    add(chain, miner, [commit_tx])
    bid = commit_tx["commitment"]
    add(chain, miner, [reveal(chain, miner, job_id, bid, sale)])
    settle_batch(chain, miner, job_id, bid)
    job = chain.state.jobs[job_id]
    assert len(job["done"]) == 8 and job["escrow"] == 0
    assert supply(chain) == minted(chain)


def test_validators_never_need_the_results():
    """Delivery only puts a 32-byte root on chain: a batch whose results and
    manifest never reached this node still settles, as long as nobody disputes it."""
    rng = np.random.default_rng(5)
    chain, (buyer, miner) = funded(2)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 2), price=1000)
    units = [[0, 0, 0], [0, 0, 1]]
    bid = sell(chain, miner, job_id, units, [unit_result(a, b, u) for u in units],
               publish=False, publish_manifest=False)
    job = chain.state.jobs[job_id]
    assert len(job["done"]) == 2 and job["escrow"] == 0
    sale = SALES[bid]
    assert not any(chain.blobs.has(r) for r in sale.leaves) and not chain.blobs.has(sale.manifest)


def test_reveal_rules():
    rng = np.random.default_rng(2)
    chain, (buyer, miner, other) = funded(3)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 1))
    c = unit_result(a, b, [0, 0, 0])

    commit_tx, sale = commit(chain, miner, job_id, [[0, 0, 0]], [c])
    bid = commit_tx["commitment"]
    s = chain.state.copy()
    h = chain.height + 1
    chain.apply_tx(s, commit_tx, h)
    with pytest.raises(InvalidTx, match="later block"):
        chain.apply_tx(s.copy(), reveal(chain, miner, job_id, bid, sale, nonce=1), h)
    add(chain, miner, [commit_tx])

    # Another miner cannot take the reserved unit, nor reveal the batch.
    other_commit, _ = commit(chain, other, job_id, [[0, 0, 0]], [c])
    with pytest.raises(InvalidTx, match="reserved"):
        chain.apply_tx(chain.state.copy(), other_commit, chain.height + 1)
    with pytest.raises(InvalidTx, match="no commitment"):
        chain.apply_tx(chain.state.copy(), reveal(chain, other, job_id, bid, sale), chain.height + 1)

    # A different root does not match the commitment.
    wrong = tx(chain, miner, "job_reveal", job=job_id, batch=bid, root="ab" * 32, manifest=sale.manifest)
    with pytest.raises(InvalidTx, match="commitment"):
        chain.apply_tx(chain.state.copy(), wrong, chain.height + 1)

    add(chain, miner, [reveal(chain, miner, job_id, bid, sale)])
    assert chain.state.jobs[job_id]["batches"][bid]["status"] == "revealed"


def test_batch_format_rules():
    rng = np.random.default_rng(3)
    chain, (buyer, miner) = funded(2)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 3))      # 3 units: one-byte bitmaps
    h = chain.height + 1
    for bitmap, match in (("01", "does not fit"),        # a bit past the last unit
                          ("e000", "does not fit"),      # wrong length
                          ("00", "non-empty bitmap"), ("E0", "bitmap"), ("e", "bitmap")):
        t = tx(chain, miner, "job_commit", job=job_id, units=bitmap, commitment="ab" * 32)
        with pytest.raises(InvalidTx, match=match):
            chain.apply_tx(chain.state.copy(), t, h)
    poor = Key.generate()
    broke, _ = commit(chain, poor, job_id, [[0, 0, 0]], [unit_result(a, b, [0, 0, 0])])
    with pytest.raises(InvalidTx, match="balance"):
        chain.apply_tx(chain.state.copy(), broke, h)


def test_wrong_result_is_disputed_and_the_whole_batch_is_forfeit():
    rng = np.random.default_rng(4)
    chain, (producer, buyer, cheat, honest) = funded(4)
    price = 10_000
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 2), price=price)
    units = [[0, 0, 0], [0, 0, 1]]
    good = [unit_result(a, b, u) for u in units]
    bad = good[1].copy()
    bad[17, 3] -= 1                                    # one wrong number out of 65,536
    bid = sell(chain, cheat, job_id, units, [good[0], bad], settle=False)

    # The buyer's client finds unit 1 wrong and disputes it.
    buyer_before = chain.state.balance(buyer.address)
    add(chain, producer, [challenge(chain, buyer, job_id, bid, [units[1]])])
    assert chain.state.jobs[job_id]["batches"][bid]["disputes"]["0,0,1"]["by"] == buyer.address

    # The cheat cannot answer: every validator's Freivalds check rejects the result,
    # and the right result is not under the root it committed to.
    with pytest.raises(InvalidTx, match="Freivalds"):
        chain.apply_tx(chain.state.copy(), respond(chain, cheat, job_id, bid, units[1]), chain.height + 1)
    forged = tx(chain, cheat, "job_respond", job=job_id, batch=bid, unit=units[1],
                result=chain.blobs.put(txmod.encode_result(good[1])), proof=merkle_proof(SALES[bid].leaves, 1))
    with pytest.raises(InvalidTx, match="Merkle"):
        chain.apply_tx(chain.state.copy(), forged, chain.height + 1)

    for _ in range(params.JOB_RESPONSE_BLOCKS + 2):
        add(chain, producer)
    job = chain.state.jobs[job_id]
    assert job["batches"] == {} and job["reserved"] == {} and job["done"] == {}
    assert job["escrow"] == 2 * price                  # nothing was paid out
    assert chain.state.open_disputes == 0
    # The challenger paid the dispute fee and receives the bond of the whole batch.
    assert chain.state.balance(buyer.address) == (buyer_before - txmod.job_challenge_fee(1)
                                                  + 2 * params.bond_for(price))

    # The units reopen and an honest miner delivers them.
    sell(chain, honest, job_id, units, good)
    job = chain.state.jobs[job_id]
    assert len(job["done"]) == 2 and job["escrow"] == 0
    assert supply(chain) == minted(chain)


def test_honest_result_survives_a_dispute():
    rng = np.random.default_rng(6)
    chain, (producer, buyer, miner) = funded(3)
    price = 10_000
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 1), price=price)
    bid = sell(chain, miner, job_id, [[0, 0, 0]], [unit_result(a, b, [0, 0, 0])], settle=False)
    miner_before = chain.state.balance(miner.address)

    add(chain, producer, [challenge(chain, buyer, job_id, bid, [[0, 0, 0]])])
    # The miner proves the unit: a Merkle proof, and every validator runs Freivalds.
    add(chain, producer, [respond(chain, miner, job_id, bid, [0, 0, 0])])
    batch = chain.state.jobs[job_id]["batches"][bid]
    assert batch["disputes"] == {} and batch["proven"] == {"0,0,0"}
    assert chain.state.open_disputes == 0

    settle_batch(chain, producer, job_id, bid)
    # Paid as usual minus the response fee: answering is neither a profit nor a loss.
    assert chain.state.balance(miner.address) == miner_before + price + params.bond_for(price) - FEE
    assert supply(chain) == minted(chain)


def test_withheld_result_cannot_be_answered():
    """Revealing a root but not the data does not get the miner paid once
    someone asks for it: without the data, no response is valid."""
    rng = np.random.default_rng(7)
    chain, (producer, buyer, miner) = funded(3)
    price = 10_000
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 1), price=price)
    bid = sell(chain, miner, job_id, [[0, 0, 0]], [unit_result(a, b, [0, 0, 0])], settle=False, publish=False)
    add(chain, producer, [challenge(chain, buyer, job_id, bid, [[0, 0, 0]])])
    with pytest.raises(MissingData):
        chain.apply_tx(chain.state.copy(), respond(chain, miner, job_id, bid, [0, 0, 0]), chain.height + 1)
    buyer_before = chain.state.balance(buyer.address)
    for _ in range(params.JOB_RESPONSE_BLOCKS + 2):
        add(chain, producer)
    assert bid not in chain.state.jobs[job_id]["batches"]
    assert chain.state.balance(buyer.address) == buyer_before + params.bond_for(price)
    assert supply(chain) == minted(chain)


def test_partial_withholding_cannot_get_paid():
    """v0.3 allowed one open dispute per batch: a miner could withhold most of a
    batch, answer the disputed unit by publishing it, and still be paid in full.
    Now every withheld unit can be disputed at once and each must be answered."""
    rng = np.random.default_rng(12)
    chain, (producer, buyer, miner) = funded(3)
    price = 10_000
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 4), price=price)
    units = txmod.all_units([1, 1, 4])
    results = [unit_result(a, b, u) for u in units]
    bid = sell(chain, miner, job_id, units, results, settle=False, publish=False)
    chain.blobs.put(txmod.encode_result(results[0]))          # only unit 0 is ever published
    withheld = units[1:]
    add(chain, producer, [challenge(chain, buyer, job_id, bid, withheld)])
    assert sorted(chain.state.jobs[job_id]["batches"][bid]["disputes"]) == ["0,0,1", "0,0,2", "0,0,3"]
    assert chain.state.open_disputes == 3

    # Answering one is not enough any more.
    chain.blobs.put(txmod.encode_result(results[1]))
    add(chain, producer, [respond(chain, miner, job_id, bid, units[1])])
    buyer_before = chain.state.balance(buyer.address)
    for _ in range(params.JOB_RESPONSE_BLOCKS + 2):
        add(chain, producer)
    job = chain.state.jobs[job_id]
    assert bid not in job["batches"] and job["done"] == {} and job["escrow"] == 4 * price
    assert chain.state.balance(buyer.address) == buyer_before + 4 * params.bond_for(price)
    assert chain.state.open_disputes == 0
    assert supply(chain) == minted(chain)


def test_answering_every_dispute_publishes_the_data_and_earns_nothing_extra():
    rng = np.random.default_rng(13)
    chain, (producer, buyer, miner) = funded(3)
    price = 10_000
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 3), price=price)
    units = txmod.all_units([1, 1, 3])
    results = [unit_result(a, b, u) for u in units]
    bid = sell(chain, miner, job_id, units, results, settle=False, publish=False)
    miner_before = chain.state.balance(miner.address)
    add(chain, producer, [challenge(chain, buyer, job_id, bid, units)])
    for r in results:
        chain.blobs.put(txmod.encode_result(r))                # the miner publishes when pressed
    add(chain, producer, [respond(chain, miner, job_id, bid, u, nonce=chain.state.nonce(miner.address) + i)
                          for i, u in enumerate(units)])
    assert all(chain.blobs.has(r) for r in SALES[bid].leaves)     # the buyer has every result now
    settle_batch(chain, producer, job_id, bid)
    # Withholding until disputed gained the miner nothing: it only paid response fees.
    assert chain.state.balance(miner.address) == miner_before + 3 * price + 3 * params.bond_for(price) - 3 * FEE


def test_dispute_rules():
    rng = np.random.default_rng(8)
    chain, (producer, buyer, miner) = funded(3)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 2))
    units = [[0, 0, 0], [0, 0, 1]]
    commit_tx, sale = commit(chain, miner, job_id, units[:1], [unit_result(a, b, units[0])])
    add(chain, producer, [commit_tx])
    bid = commit_tx["commitment"]
    h = chain.height + 1
    with pytest.raises(InvalidTx, match="no revealed results"):
        chain.apply_tx(chain.state.copy(), challenge(chain, buyer, job_id, bid, units[:1]), h)
    add(chain, producer, [reveal(chain, miner, job_id, bid, sale)])
    h = chain.height + 1
    with pytest.raises(InvalidTx, match="no open dispute"):
        chain.apply_tx(chain.state.copy(), respond(chain, miner, job_id, bid, units[0]), h)
    with pytest.raises(InvalidTx, match="not in the batch"):
        chain.apply_tx(chain.state.copy(), challenge(chain, buyer, job_id, bid, units[1:]), h)
    with pytest.raises(InvalidTx, match="fee"):
        chain.apply_tx(chain.state.copy(), challenge(chain, buyer, job_id, bid, units[:1], fee=FEE), h)
    add(chain, producer, [challenge(chain, buyer, job_id, bid, units[:1])])
    with pytest.raises(InvalidTx, match="already disputed"):
        chain.apply_tx(chain.state.copy(), challenge(chain, producer, job_id, bid, units[:1]), chain.height + 1)
    deadline = chain.state.jobs[job_id]["batches"][bid]["disputes"]["0,0,0"]["deadline"]
    with pytest.raises(InvalidTx, match="deadline"):
        chain.apply_tx(chain.state.copy(), respond(chain, miner, job_id, bid, units[0]), deadline + 1)
    add(chain, producer, [respond(chain, miner, job_id, bid, units[0])])
    with pytest.raises(InvalidTx, match="already disputed or proven"):    # no griefing the same unit twice
        chain.apply_tx(chain.state.copy(), challenge(chain, buyer, job_id, bid, units[:1]), chain.height + 1)
    settle_at = chain.state.jobs[job_id]["batches"][bid]["settle_at"]
    with pytest.raises(InvalidTx, match="window has closed"):
        chain.apply_tx(chain.state.copy(), challenge(chain, buyer, job_id, bid, units[:1]), settle_at + 1)


def test_dispute_deadlines_stretch_with_the_backlog(monkeypatch):
    """A flood of disputes cannot leave honest miners without room to answer:
    each new dispute's deadline leaves time for every open one ahead of it."""
    monkeypatch.setattr(params, "MAX_BLOCK_JOB_RESPONSES", 2)
    rng = np.random.default_rng(14)
    chain, (producer, buyer, miner) = funded(3)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 6))
    units = txmod.all_units([1, 1, 6])
    bid = sell(chain, miner, job_id, units, [unit_result(a, b, u) for u in units], settle=False)
    add(chain, producer, [challenge(chain, buyer, job_id, bid, units)])
    h = chain.height
    deadlines = {d["deadline"] for d in chain.state.jobs[job_id]["batches"][bid]["disputes"].values()}
    assert deadlines == {h + params.JOB_RESPONSE_BLOCKS + 6 // 2}
    # Two answers per block: all six fit before the deadline.
    for i in range(0, 6, 2):
        n0 = chain.state.nonce(miner.address)
        add(chain, producer, [respond(chain, miner, job_id, bid, u, nonce=n0 + k) for k, u in enumerate(units[i:i + 2])])
    assert chain.state.jobs[job_id]["batches"][bid]["disputes"] == {}


def test_responses_per_block_are_capped(monkeypatch):
    monkeypatch.setattr(params, "MAX_BLOCK_JOB_RESPONSES", 1)
    rng = np.random.default_rng(9)
    chain, (producer, buyer, miner) = funded(3)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 2))
    units = [[0, 0, 0], [0, 0, 1]]
    bid = sell(chain, miner, job_id, units, [unit_result(a, b, u) for u in units], settle=False)
    add(chain, producer, [challenge(chain, buyer, job_id, bid, units)])
    n0 = chain.state.nonce(miner.address)
    responds = [respond(chain, miner, job_id, bid, u, nonce=n0 + i) for i, u in enumerate(units)]
    block, _ = mine(chain, producer, responds)
    assert len(block["txs"]) == 1                      # the second waits for the next block
    chain.add_block(block)
    add(chain, producer, [responds[1]])
    assert chain.state.jobs[job_id]["batches"][bid]["disputes"] == {}


def test_unrevealed_commit_expires_and_bond_goes_to_buyer():
    rng = np.random.default_rng(10)
    chain, (producer, buyer, lazy, worker) = funded(4)
    price = 10_000
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 1), price=price)
    c = unit_result(a, b, [0, 0, 0])
    lazy_commit, _ = commit(chain, lazy, job_id, [[0, 0, 0]], [c])
    add(chain, producer, [lazy_commit])
    buyer_before = chain.state.balance(buyer.address)
    for _ in range(params.RESERVE_BLOCKS + 1):
        add(chain, producer)
    job = chain.state.jobs[job_id]
    assert job["batches"] == {} and job["reserved"] == {}
    assert chain.state.balance(buyer.address) == buyer_before + params.bond_for(price)
    assert chain.state.open_units(job_id) == [[0, 0, 0]]
    sell(chain, worker, job_id, [[0, 0, 0]], [c])
    assert supply(chain) == minted(chain)


def test_block_size_is_limited(monkeypatch):
    chain, (alice, bob) = funded(2)
    n0 = chain.state.nonce(alice.address)
    txs = [tx(chain, alice, "transfer", to=bob.address, amount=1, nonce=n0 + i) for i in range(10)]
    size = len(canonical(txs))
    monkeypatch.setattr(params, "MAX_BLOCK_BYTES", size // 2)
    block, _ = mine(chain, alice, txs)
    assert 0 < len(block["txs"]) < 10 and len(canonical(block["txs"])) <= size // 2
    monkeypatch.setattr(params, "MAX_BLOCK_BYTES", size)
    big, _ = mine(chain, alice, txs)
    assert len(big["txs"]) == 10
    monkeypatch.setattr(params, "MAX_BLOCK_BYTES", size - 1)
    with pytest.raises(InvalidBlock, match="too large") as e:
        chain.add_block(big)
    assert e.value.permanent


def test_tampered_blocks_are_rejected():
    alice = Key.generate()
    chain = new_chain(alice)
    block, _ = mine(chain, alice)
    block["header"]["nonce"] += 1                      # seed changes, transcript no longer matches
    with pytest.raises(InvalidBlock, match="transcript"):
        chain.add_block(block)

    block, _ = mine(chain, alice)
    block["header"]["miner"] = Key.generate().address  # steal the ticket
    with pytest.raises(InvalidBlock, match="transcript"):
        chain.add_block(block)

    rng = np.random.default_rng(2)
    a = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
    job_id = chain.blobs.put(txmod.encode_job(a, a))
    header, _, _ = chain.build_template(alice.address, [], now=T0 + 10)
    header["work"] = {"job": job_id, "unit": [0, 0, 0]}
    header["transcript"] = "00" * 32
    with pytest.raises(InvalidBlock, match="never posted"):
        chain.add_block({"header": header, "txs": []})


def test_a_block_must_commit_to_the_state_it_produces():
    """The header's state root is checked against the state after the block; a
    miner cannot claim an outcome the transactions do not produce."""
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    pay = tx(chain, alice, "transfer", to=bob.address, amount=5)
    header, chosen, after = chain.build_template(alice.address, [pay], now=T0 + 10)
    honest_root = header["state_root"]
    assert honest_root == after.root()
    header["state_root"] = "11" * 32
    block, _ = grind(chain, header, chosen)
    with pytest.raises(InvalidBlock, match="state root"):
        chain.add_block(block)
    assert block_hash(block) in chain.invalid           # the header itself is wrong: remembered
    header["state_root"] = honest_root
    block, _ = grind(chain, header, chosen)
    assert chain.add_block(block) == "added"
    assert chain.state.root() == honest_root == chain.tip_header()["state_root"]


def test_ticket_must_meet_target(monkeypatch):
    alice = Key.generate()
    chain = new_chain(alice)
    block, _ = mine(chain, alice)
    monkeypatch.setattr(chain, "expected_target", lambda prev, lookup=None: 1)
    block["header"]["target"] = f"{1:064x}"
    block["header"]["transcript"], _, _ = solve(block["header"], chain.blobs)
    with pytest.raises(InvalidBlock, match="target"):
        chain.add_block(block)


def test_heaviest_chain_wins_and_state_follows():
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    g = chain.tip
    a1, _ = mine(chain, alice)
    chain.add_block(a1)
    pay = tx(chain, alice, "transfer", to=bob.address, amount=7)
    b1, _ = mine(chain, bob, [pay], t=T0 + 11, parent=g)
    assert chain.add_block(b1) == "added"
    a1_hash, b1_hash = list(chain.blocks)[1:3]
    assert chain.tip == a1_hash                        # equal work: first seen stays
    b2, _ = mine(chain, bob, t=T0 + 20, parent=b1_hash)
    chain.add_block(b2)
    assert chain.height == 2 and chain.main[1] == b1_hash
    assert chain.state.balance(bob.address) == 7 + 2 * params.block_reward(1) + FEE


def mine_at(chain, key, t):
    header, _, _ = chain.build_template(key.address, [], now=t)
    while True:
        header["transcript"], ticket, _ = solve(header, chain.blobs)
        if ticket <= int(header["target"], 16):
            break
        header["nonce"] += 1
    assert chain.add_block({"header": header, "txs": []}) == "added"
    return int(header["target"], 16)


def test_difficulty_follows_the_schedule_every_block(monkeypatch):
    """ASERT: on schedule the target stays put; one half-life ahead of schedule
    halves it, one half-life behind doubles it -- adjusted at every block."""
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET // 8)
    monkeypatch.setattr(params, "ASERT_HALFLIFE", 600)
    alice = Key.generate()
    chain = Chain(make_genesis(alice.address, "", timestamp=T0), BlobStore())
    g = params.INITIAL_TARGET
    spacing = params.TARGET_BLOCK_SECONDS
    assert mine_at(chain, alice, T0 + spacing) == g                    # block 1: the anchor's target
    assert mine_at(chain, alice, T0 + 2 * spacing) == g                # exactly on schedule: unchanged
    # Block 3 comes one half-life (600 s) late: the target for block 4 doubles.
    mine_at(chain, alice, T0 + 3 * spacing + 600)
    assert int(chain.expected_target(chain.tip)) == 2 * g
    # Fast blocks pull it back down, every single block.
    targets = [mine_at(chain, alice, chain.tip_header()["time"] + 1) for _ in range(5)]
    assert all(b < a for a, b in zip(targets, targets[1:]))


def test_asert_matches_the_formula():
    from green.chain import asert
    T, H = params.TARGET_BLOCK_SECONDS, 3600
    anchor = 2**200
    for blocks_ahead in (-480, -240, 0, 240, 480):      # how many half-lives of drift: -2..2
        parent_height = 1000
        parent_time = (parent_height + 1) * T + blocks_ahead * T
        got = asert(anchor, 0, 0, parent_time, parent_height, H)
        want = anchor * 2 ** (blocks_ahead * T / H)
        assert abs(got - want) / want < 1e-4, (blocks_ahead, got / want)
    assert asert(params.MAX_TARGET, 0, 0, 10**9, 1, H) == params.MAX_TARGET    # clamped
    assert asert(1, 0, 0, 0, 10**6, H) == 1


def test_emission_matches_bitcoin_schedule():
    total = sum(params.block_reward(h * params.HALVING_INTERVAL) * params.HALVING_INTERVAL for h in range(64))
    assert 20_999_990 * params.YAPRAK_PER_COIN < total <= 21_000_000 * params.YAPRAK_PER_COIN
    years = params.HALVING_INTERVAL * params.TARGET_BLOCK_SECONDS / (365.25 * 86400)
    assert 3.9 < years < 4.1


def test_decoded_job_data_is_cached_within_a_bound():
    """A decoded job can be 32 MiB; posting many jobs must not grow memory without end."""
    from green import chain as chainmod
    blobs = BlobStore()
    rng = np.random.default_rng(3)
    ids = []
    for _ in range(chainmod.JOB_CACHE + 5):
        a = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
        ids.append(blobs.put(txmod.encode_job(a, a)))
    for job_id in ids:
        blobs.job(job_id, 0)
    assert len(blobs._jobs) == chainmod.JOB_CACHE
    assert list(blobs._jobs) == ids[-chainmod.JOB_CACHE:]          # the most recently used are kept
    blobs.job(ids[-chainmod.JOB_CACHE], 0)
    assert list(blobs._jobs)[-1] == ids[-chainmod.JOB_CACHE]       # and use refreshes an entry
