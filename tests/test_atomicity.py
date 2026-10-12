"""A rejected transaction leaves no trace in the state.

The mempool applies transactions straight onto its pending state, so every
transaction type must fail atomically -- including when it fails late, after
some of its checks already passed and after values were copied into the top
layer. Each case below is built to fail at the last possible check.
"""

import numpy as np
import pytest

wasmtime = pytest.importorskip("wasmtime")

from green import params, tx as txmod  # noqa: E402
from green.chain import MissingData  # noqa: E402
from green.crypto import Key, merkle_proof  # noqa: E402
from green.tx import InvalidTx  # noqa: E402
from helpers import SALES, add, challenge, commit, new_chain, post_job, respond, reveal, sell, tx, unit_result  # noqa: E402

PROG = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
  (func (export "green_run") (param i32 i32) (result i64)
    (i32.store (i32.const 0) (i32.const 7)) (i64.const 4)))""")


@pytest.fixture
def world(monkeypatch):
    """A chain with an open job, a revealed batch under dispute, and a revealed task."""
    monkeypatch.setattr(params, "TASK_RESERVE_BLOCKS", 3)
    monkeypatch.setattr(params, "TASK_CHALLENGE_BLOCKS", 4)
    keys = [Key.generate() for _ in range(5)]
    producer, buyer, miner, worker, poor = keys
    chain = new_chain(producer)
    add(chain, producer, [tx(chain, producer, "transfer", to=k.address, amount=2 * 10**7, nonce=i)
                          for i, k in enumerate(keys[1:4])])
    rng = np.random.default_rng(3)
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 3), price=10_000)
    units = txmod.all_units([1, 1, 3])
    results = [unit_result(a, b, u) for u in units]
    bad = results[2].copy()
    bad[0, 0] += 1
    bid = sell(chain, miner, job_id, units[:2], results[:2], settle=False, publish=False)
    bid_bad = sell(chain, miner, job_id, units[2:], [bad], settle=False)
    add(chain, producer, [challenge(chain, buyer, job_id, bid, units[:1])])
    add(chain, producer, [challenge(chain, buyer, job_id, bid_bad, units[2:])])
    ph, ih = chain.blobs.put(PROG), chain.blobs.put(b"x")
    post = tx(chain, buyer, "task_post", prog=ph, input=ih, fuel=10_000, price=10**6,
              fee=txmod.task_post_fee(len(PROG), 1))
    add(chain, producer, [post])
    task_id = txmod.task_id_of(post)
    # The worker answers to a guardian (the producer) with a tight limit.
    add(chain, producer, [tx(chain, worker, "guard_set", guardian=producer.address, limit=10**6, period=1000)])
    out = "ab" * 32                                           # a wrong output, to be challenged
    add(chain, producer, [tx(chain, worker, "task_commit", task=task_id,
                             commitment=txmod.task_commitment(worker.address, task_id, out))])
    add(chain, producer, [tx(chain, worker, "task_reveal", task=task_id, output=out)])
    return chain, keys, job_id, units, bid, bid_bad, task_id


def cases(chain, keys, job_id, units, bid, bid_bad, task_id):
    producer, buyer, miner, worker, poor = keys
    sale = SALES[bid]
    bitmap_all = txmod.units_bitmap(units, [1, 1, 3])
    t = chain.state.tasks[task_id]
    yield "transfer: balance", InvalidTx, tx(chain, poor, "transfer", to=buyer.address, amount=1)
    yield "job_post: already posted", InvalidTx, tx(chain, buyer, "job_post", job=job_id, dims=[1, 1, 3], price=1)
    # Two open units checked, then the third is reserved: fails inside the loop.
    yield "job_commit: reserved late in the bitmap", InvalidTx, tx(
        chain, worker, "job_commit", job=job_id, units=bitmap_all, commitment="cd" * 32)
    yield "job_commit: bond", InvalidTx, commit(chain, poor, job_id, [[0, 0, 2]], [np.zeros((256, 256))])[0]
    yield "job_reveal: wrong root", InvalidTx, tx(chain, miner, "job_reveal", job=job_id, batch=bid,
                                                  root="ab" * 32, manifest=sale.manifest)
    yield "job_challenge: already disputed", InvalidTx, challenge(chain, producer, job_id, bid, units[:2])
    yield "job_challenge: balance", InvalidTx, challenge(chain, poor, job_id, bid, units[1:2])
    yield "job_respond: proof for another unit", InvalidTx, tx(
        chain, miner, "job_respond", job=job_id, batch=bid, unit=units[0], result=sale.leaves[1],
        proof=merkle_proof(sale.leaves, 1))
    yield "job_respond: data withheld", MissingData, respond(chain, miner, job_id, bid, units[0])
    yield "job_respond: wrong result", InvalidTx, respond(chain, miner, job_id, bid_bad, units[2])
    yield "job_cancel: not the buyer", InvalidTx, tx(chain, miner, "job_cancel", job=job_id)
    yield "task_post: data missing", MissingData, tx(chain, buyer, "task_post", prog="ee" * 32, input="ff" * 32,
                                                     fuel=10, price=10, fee=10**6)
    yield "task_commit: not open", InvalidTx, tx(chain, producer, "task_commit", task=task_id, commitment="ab" * 32)
    yield "task_challenge: bond", InvalidTx, tx(chain, poor, "task_challenge", task=task_id,
                                                fee=params.challenge_fee(t["fuel"], t["prog_size"]))
    yield "transfer: over the guardian's limit", InvalidTx, tx(chain, worker, "transfer", to=buyer.address, amount=10**6)
    yield "guard_set: the account tries to loosen", InvalidTx, tx(chain, worker, "guard_set",
                                                                  guardian=producer.address, limit=10**9, period=1000)
    yield "guard_update: not the guardian", InvalidTx, tx(chain, buyer, "guard_update", account=worker.address,
                                                          guardian=buyer.address, limit=10**9, period=1, frozen=False)
    yield "guard_release: not the guardian", InvalidTx, tx(chain, miner, "guard_release", account=worker.address)


def test_every_rejected_transaction_leaves_no_trace(world):
    chain = world[0]
    seen = set()
    for name, error, t in cases(*world):
        s = chain.state.copy()
        before = s.fingerprint()
        with pytest.raises(error):
            chain.apply_tx(s, t, chain.height + 1)
        assert s.fingerprint() == before, f"{name}: a rejected transaction changed the state"
        assert s.adjudication_used == 0 and s.responses_used == 0, name
        seen.add(t["type"])
    assert {"transfer", "job_post", "job_commit", "job_reveal", "job_challenge", "job_respond", "job_cancel",
            "task_post", "task_commit", "task_challenge", "guard_set", "guard_update", "guard_release"} <= seen


def test_a_state_survives_many_failures_then_accepts_valid_work(world):
    """The same state object takes a string of failures, then still applies valid
    transactions correctly -- what the mempool does all day."""
    chain, keys, job_id, units, bid, bid_bad, task_id = world
    s = chain.state.copy()
    before = s.fingerprint()
    for _, _, t in cases(*world):
        try:
            chain.apply_tx(s, t, chain.height + 1)
        except (InvalidTx, MissingData):
            pass
    assert s.fingerprint() == before
    producer, buyer = keys[0], keys[1]
    t = chain.state.tasks[task_id]
    chain.apply_tx(s, tx(chain, producer, "task_challenge", task=task_id,
                         fee=params.challenge_fee(t["fuel"], t["prog_size"])), chain.height + 1)
    assert s.tasks[task_id]["status"] == "failed"                 # the wrong output is caught
    assert chain.state.tasks[task_id]["status"] == "revealed"     # and the chain's own state is untouched
