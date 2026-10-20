"""Compute-market tests: post / commit / reveal / settle, the fraud game, and the
limits that keep adjudication bounded (SECURITY.md, "Second review of the WASM market")."""

import pytest

wasmtime = pytest.importorskip("wasmtime")

from green import params, tx as txmod, wasm  # noqa: E402
from green.chain import InvalidBlock, MissingData  # noqa: E402
from green.crypto import Key  # noqa: E402
from green.tx import InvalidTx  # noqa: E402
from helpers import add, grind, mine, minted, new_chain, supply, tx  # noqa: E402

# Sum input bytes -> i32 at offset 0, return len 4.
SUM = wasmtime.wat2wasm(r"""(module
  (memory (export "memory") 2)
  (func (export "green_run") (param $p i32) (param $n i32) (result i64)
    (local $i i32) (local $acc i32)
    (block $done (loop $l
      (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
      (local.set $acc (i32.add (local.get $acc)
        (i32.load8_u (i32.add (local.get $p) (local.get $i)))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l)))
    (i32.store (i32.const 0) (local.get $acc))
    (i64.const 4)))""")

WITH_IMPORT = wasmtime.wat2wasm(r"""(module (import "env" "clock" (func))
  (memory (export "memory") 1)
  (func (export "green_run") (param i32 i32) (result i64) (i64.const 0)))""")


@pytest.fixture(autouse=True)
def fast_windows(monkeypatch):
    monkeypatch.setattr(params, "TASK_RESERVE_BLOCKS", 3)
    monkeypatch.setattr(params, "TASK_CHALLENGE_BLOCKS", 4)


def fund(chain, miner, who, amount=10**7):
    add(chain, miner, [tx(chain, miner, "transfer", to=who.address, amount=amount)])


def post_tx(chain, buyer, prog=SUM, data=b"abcde", fuel=1_000_000, price=10**6, **kw):
    ph, ih = chain.blobs.put(prog), chain.blobs.put(data)
    fee = kw.pop("fee", txmod.task_post_fee(len(prog), len(data)))
    return tx(chain, buyer, "task_post", prog=ph, input=ih, fuel=fuel, price=price, fee=fee, **kw)


def post_task(chain, buyer, data=b"abcde", fuel=1_000_000, price=10**6):
    post = post_tx(chain, buyer, data=data, fuel=fuel, price=price)
    add(chain, buyer, [post])
    return txmod.task_id_of(post), SUM, data


def true_output_hash(data):
    out, _ = wasm.run(SUM, data)
    return wasm.result_hash(out)


def commit_task(chain, worker, task_id, output_hash):
    c = txmod.task_commitment(worker.address, task_id, output_hash)
    return tx(chain, worker, "task_commit", task=task_id, commitment=c)


def challenge(chain, who, task_id, **kw):
    t = chain.state.tasks[task_id]
    fee = kw.pop("fee", params.challenge_fee(t["fuel"], t["prog_size"]))
    return tx(chain, who, "task_challenge", task=task_id, fee=fee, **kw)


def revealed_task(chain, miner, worker, output_hash=None, data=b"\x03\x04"):
    buyer = Key.generate()
    fund(chain, miner, buyer)
    fund(chain, miner, worker)
    task_id, _, data = post_task(chain, buyer, data)
    out = output_hash or true_output_hash(data)
    add(chain, miner, [commit_task(chain, worker, task_id, out)])
    add(chain, miner, [tx(chain, worker, "task_reveal", task=task_id, output=out)])
    return task_id, buyer


# --- lifecycle ---------------------------------------------------------------------

def test_honest_task_pays_worker():
    buyer, worker, miner = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, buyer)
    fund(chain, miner, worker)
    task_id, _, data = post_task(chain, buyer, b"\x01\x02\x03\x04\x05", price=10**6)
    assert chain.state.tasks[task_id]["escrow"] == 10**6

    out_hash = true_output_hash(data)
    add(chain, miner, [commit_task(chain, worker, task_id, out_hash)])
    bond = params.task_bond(10**6)
    add(chain, miner, [tx(chain, worker, "task_reveal", task=task_id, output=out_hash)])
    assert chain.state.tasks[task_id]["status"] == "revealed"

    while chain.state.tasks[task_id]["status"] != "done":     # paid only after the window
        before = chain.state.balance(worker.address)
        add(chain, miner)
    assert chain.state.balance(worker.address) == before + 10**6 + bond
    assert chain.state.tasks[task_id]["escrow"] == 0
    assert supply(chain) == minted(chain)


def test_fraud_is_caught_and_worker_slashed():
    cheat, hunter, miner = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, hunter)
    task_id, buyer = revealed_task(chain, miner, cheat, output_hash="ff" * 32)
    bond = params.task_bond(10**6)
    buyer_before = chain.state.balance(buyer.address)
    hunter_before = chain.state.balance(hunter.address)
    c = challenge(chain, hunter, task_id)
    add(chain, miner, [c])

    task = chain.state.tasks[task_id]
    assert task["status"] == "failed" and task["escrow"] == 0
    assert chain.state.balance(buyer.address) == buyer_before + 10**6
    assert chain.state.balance(hunter.address) == hunter_before + bond - c["fee"]
    assert supply(chain) == minted(chain)


def test_false_challenge_loses_bond_and_confirms_result():
    worker, troll, miner = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, troll)
    task_id, _ = revealed_task(chain, miner, worker)
    bond = params.task_bond(10**6)
    worker_before = chain.state.balance(worker.address)
    add(chain, miner, [challenge(chain, troll, task_id)])
    assert chain.state.balance(worker.address) == worker_before + bond
    assert chain.state.tasks[task_id]["status"] == "confirmed"
    while chain.state.tasks[task_id]["status"] != "done":
        add(chain, miner)
    assert supply(chain) == minted(chain)


def test_reveal_must_match_commitment_and_come_later():
    buyer, worker, miner = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, buyer)
    fund(chain, miner, worker)
    task_id, _, data = post_task(chain, buyer, b"\x05")
    out_hash = true_output_hash(data)
    commit = commit_task(chain, worker, task_id, out_hash)
    s = chain.state.copy()
    h = chain.height + 1
    chain.apply_tx(s, commit, h)
    reveal = tx(chain, worker, "task_reveal", task=task_id, output=out_hash, nonce=1)
    with pytest.raises(InvalidTx, match="later block"):
        chain.apply_tx(s.copy(), reveal, h)
    add(chain, miner, [commit])
    wrong = tx(chain, worker, "task_reveal", task=task_id, output="ab" * 32, nonce=1)
    with pytest.raises(InvalidTx, match="commitment"):
        chain.apply_tx(chain.state.copy(), wrong, chain.height + 1)


def test_expired_commit_lets_another_worker_take_over():
    buyer, slow, fast, miner = (Key.generate() for _ in range(4))
    chain = new_chain(miner)
    for k in (buyer, slow, fast):
        fund(chain, miner, k)
    task_id, _, data = post_task(chain, buyer, b"\x07\x07")
    out_hash = true_output_hash(data)
    add(chain, miner, [commit_task(chain, slow, task_id, out_hash)])
    buyer_before = chain.state.balance(buyer.address)
    for _ in range(params.TASK_RESERVE_BLOCKS + 1):
        add(chain, miner)
    add(chain, miner, [commit_task(chain, fast, task_id, out_hash)])
    assert chain.state.balance(buyer.address) == buyer_before + params.task_bond(10**6)
    assert chain.state.tasks[task_id]["reserved"]["worker"] == fast.address
    assert supply(chain) == minted(chain)


# --- bounded adjudication (each of these was possible before the hardening pass) --

def test_confirmed_result_cannot_be_challenged_again():
    """Otherwise a worker+challenger pair could make every validator re-run the
    program over and over for the cost of a fee, since the bond goes to itself."""
    worker, troll, miner = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, troll)
    task_id, _ = revealed_task(chain, miner, worker)
    add(chain, miner, [challenge(chain, troll, task_id)])
    again = challenge(chain, troll, task_id)
    with pytest.raises(InvalidTx, match="unconfirmed"):
        chain.apply_tx(chain.state.copy(), again, chain.height + 1)


def test_block_adjudication_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(params, "TASK_CHALLENGE_BLOCKS", 50)   # both windows must stay open
    miner = Key.generate()
    chain = new_chain(miner)
    hunter = Key.generate()
    fund(chain, miner, hunter, 10**8)
    t1, _ = revealed_task(chain, miner, Key.generate(), output_hash="11" * 32, data=b"\x01")
    t2, _ = revealed_task(chain, miner, Key.generate(), output_hash="22" * 32, data=b"\x02")
    one = params.adjudication_cost(1_000_000, len(SUM))
    monkeypatch.setattr(params, "MAX_BLOCK_ADJUDICATION", one)      # room for one re-run
    c1 = challenge(chain, hunter, t1)
    c2 = challenge(chain, hunter, t2, nonce=c1["nonce"] + 1)

    # A block producer that tries to cram both in is producing an invalid block...
    header, chosen, _ = chain.build_template(miner.address, [c1, c2])
    assert [t["task"] for t in chosen] == [t1]                       # ...so the template leaves one out
    block, _ = mine(chain, miner, [c1])
    block["txs"] = [c1, c2]
    from green.crypto import merkle_root
    block["header"]["tx_root"] = merkle_root([txmod.txid(t) for t in block["txs"]])
    # Re-mine the changed header: blocks come 10 s apart against a 15 s target, so
    # ASERT has tightened the target slightly and a single attempt can miss it.
    block, _ = grind(chain, block["header"], block["txs"])
    with pytest.raises(InvalidBlock, match="budget"):
        chain.add_block(block)
    # The leftover challenge fits in the next block.
    add(chain, miner, [c1])
    add(chain, miner, [c2])
    assert chain.state.tasks[t2]["status"] == "failed"


def test_challenge_fee_pays_for_the_re_execution():
    worker, hunter, miner = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, hunter)
    task_id, _ = revealed_task(chain, miner, worker, output_hash="ee" * 32)
    cheap = challenge(chain, hunter, task_id, fee=params.MIN_TX_FEE)
    with pytest.raises(InvalidTx, match="fee"):
        chain.apply_tx(chain.state.copy(), cheap, chain.height + 1)
    assert params.challenge_fee(params.MAX_TASK_FUEL, 0) > params.challenge_fee(1000, 0)


def test_failed_adjudication_leaves_state_untouched():
    """A challenge that cannot be judged yet (missing data) must not half-apply,
    or mempool and block templates would carry a corrupted balance."""
    worker, hunter, miner = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, hunter)
    task_id, _ = revealed_task(chain, miner, worker, output_hash="dd" * 32)
    del chain.blobs._mem[chain.state.tasks[task_id]["prog"]]
    s = chain.state.copy()
    before = s.balance(hunter.address)
    with pytest.raises(MissingData):
        chain.apply_tx(s, challenge(chain, hunter, task_id), chain.height + 1)
    assert s.balance(hunter.address) == before and s.adjudication_used == 0
    assert s.tasks[task_id]["status"] == "revealed"


# --- task_post guarantees -------------------------------------------------------------

def test_post_requires_the_data_so_fraud_can_always_be_proven():
    buyer, miner = Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, buyer)
    post = tx(chain, buyer, "task_post", prog="aa" * 32, input="bb" * 32, fuel=1000, price=10**5,
              fee=10**6)
    with pytest.raises(MissingData):
        chain.apply_tx(chain.state.copy(), post, chain.height + 1)


def test_post_rejects_bad_programs_and_unpaid_data():
    buyer, miner = Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, buyer, 10**8)
    h = chain.height + 1
    with pytest.raises(InvalidTx, match="import"):
        chain.apply_tx(chain.state.copy(), post_tx(chain, buyer, prog=WITH_IMPORT), h)
    with pytest.raises(InvalidTx, match="program rejected"):
        chain.apply_tx(chain.state.copy(), post_tx(chain, buyer, prog=b"\x00asm\x01\x00\x00\x00garbage"), h)
    with pytest.raises(InvalidTx, match="fee"):
        chain.apply_tx(chain.state.copy(), post_tx(chain, buyer, fee=params.MIN_TX_FEE), h)
    big = b"\x00" * (params.MAX_TASK_INPUT_BYTES + 1)
    with pytest.raises(InvalidTx, match="input larger"):
        chain.apply_tx(chain.state.copy(), post_tx(chain, buyer, data=big), h)
    too_long = post_tx(chain, buyer, fuel=params.MAX_TASK_FUEL + 1)
    with pytest.raises(InvalidTx, match="fuel"):
        txmod.check_format(too_long, chain.chain_id)


def test_money_is_conserved_through_random_task_activity():
    """Fuzz the task market: honest and lying workers, right and wrong challenges,
    expiring reservations. No coin may appear or vanish in any block."""
    import random
    pick = random.Random(7)
    miner = Key.generate()
    chain = new_chain(miner)
    keys = [Key.generate() for _ in range(4)]
    for k in keys:
        fund(chain, miner, k, 10**8)

    def valid(t):
        try:
            chain.apply_tx(chain.state.copy(), t, chain.height + 1)
            return True
        except InvalidTx:
            return False

    tasks, commits, seen = [], {}, set()
    for step in range(150):
        actor = pick.choice(keys)
        roll = pick.random()
        t = None
        if roll < 0.15 and len(tasks) < 10:
            data = bytes(pick.randrange(256) for _ in range(pick.randint(1, 8)))
            t = post_tx(chain, actor, data=data, price=pick.randint(10**4, 10**6))
            tasks.append((txmod.task_id_of(t), data))
        elif roll < 0.4 and tasks:
            tid, data = pick.choice(tasks)
            if tid in chain.state.tasks and tid not in commits:
                out = true_output_hash(data) if pick.random() < 0.6 else f"{pick.getrandbits(256):064x}"
                t = commit_task(chain, actor, tid, out)
                commits[tid] = (actor, out)
        elif roll < 0.65 and commits:
            tid = pick.choice(list(commits))
            worker, out = commits.pop(tid)
            t = tx(chain, worker, "task_reveal", task=tid, output=out)
        elif roll < 0.85 and tasks:
            tid, _ = pick.choice(tasks)
            if chain.state.tasks.get(tid, {}).get("status") == "revealed":
                t = challenge(chain, actor, tid)
        if t is not None and not valid(t):
            if t["type"] == "task_commit":
                commits.pop(t["task"], None)
            t = None
        add(chain, miner, [t] if t else [])
        assert supply(chain) == minted(chain), f"money leaked at step {step}"
        assert all(v >= 0 for v in chain.state.balances.values())
        seen |= {x["status"] for x in chain.state.tasks.values()}
    # the fuzz must actually have walked every path
    assert {"done", "failed", "confirmed"} <= seen, seen


def test_wasm_engine_version_is_pinned():
    assert wasm.installed_version() == params.WASMTIME_VERSION
    wasm.check_engine()


def test_a_block_can_name_only_so_much_new_data(monkeypatch):
    """Every validator fetches and checks the data a block's posts name: a job's
    matrices, a task's program (validated and instrumented, so it weighs more) and
    input. A block of thousands of posts used to name gigabytes. Past MAX_BLOCK_DATA a
    post waits for the next block, and a block that crams it in is invalid; the budget
    is checked before anything is fetched."""
    import numpy as np
    miner, buyer = Key.generate(), Key.generate()
    chain = new_chain(miner)
    fund(chain, miner, buyer, 10**8)
    job_bytes = txmod.job_blob_size([1, 1, 1])
    monkeypatch.setattr(params, "MAX_BLOCK_DATA", 2 * job_bytes)
    rng = np.random.default_rng(5)
    posts = []
    for i in range(3):
        a = rng.integers(-128, 128, size=(params.UNIT, params.UNIT), dtype=np.int8)
        job = chain.blobs.put(txmod.encode_job(a, a))
        posts.append(tx(chain, buyer, "job_post", job=job, dims=[1, 1, 1], price=10, nonce=i))
    _, chosen, _ = chain.build_template(miner.address, posts)
    assert chosen == posts[:2]                                           # the third waits
    block, _ = mine(chain, miner, posts[:2])
    block["txs"] = posts
    from green.crypto import merkle_root
    block["header"]["tx_root"] = merkle_root([txmod.txid(t) for t in posts])
    block, _ = grind(chain, block["header"], posts)
    with pytest.raises(InvalidBlock, match="budget"):
        chain.add_block(block)
    s = chain.state.copy()                                               # over the budget: refused
    for t in posts[:2]:                                                  # without fetching anything
        chain.apply_tx(s, t, chain.height + 1)
    unknown = tx(chain, buyer, "job_post", job="ab" * 32, dims=[1, 1, 1], price=10, nonce=2)
    with pytest.raises(InvalidTx, match="budget"):
        chain.apply_tx(s, unknown, chain.height + 1)
    first = tx(chain, buyer, "job_post", job="ab" * 32, dims=[1, 1, 1], price=10, nonce=0)
    with pytest.raises(MissingData):                                     # (with room, it would fetch)
        chain.apply_tx(chain.state.copy(), first, chain.height + 1)
    # A program weighs PROGRAM_DATA_WEIGHT times its size.
    monkeypatch.setattr(params, "MAX_BLOCK_DATA", len(SUM) * params.PROGRAM_DATA_WEIGHT + 4)
    t1, t2 = post_tx(chain, buyer, data=b"abcd", nonce=0), post_tx(chain, buyer, data=b"abce", nonce=1)
    s = chain.state.copy()
    chain.apply_tx(s, t1, chain.height + 1)
    with pytest.raises(InvalidTx, match="budget"):
        chain.apply_tx(s, t2, chain.height + 1)
