import socket
import time

import numpy as np
import pytest

from green import params, tx as txmod
from green.chain import make_genesis
from green.crypto import Key, merkle_root
from green.node import Node, http_get, http_post
from green.tx import make_tx

N = params.UNIT


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(cond, timeout=60.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def network(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET // 3)
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    monkeypatch.setattr(params, "JOB_CHALLENGE_BLOCKS", 8)
    monkeypatch.setattr(params, "JOB_RESPONSE_BLOCKS", 4)
    miner, buyer = Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, "network test")
    p1, p2 = free_port(), free_port()
    quiet = lambda msg: None  # noqa: E731
    n1 = Node(str(tmp_path / "n1"), genesis, port=p1, miner_key=miner, log=quiet)
    n1.start(mine=True)
    n2 = Node(str(tmp_path / "n2"), genesis, port=p2, peers=[n1.p2p.full_address], miner_key=buyer, log=quiet)
    n2.start(mine=False)
    yield n1, n2, miner, buyer
    n1.stop()
    n2.stop()


def test_two_nodes_sync_and_sell_compute(network):
    n1, n2, miner, buyer = network
    chain_id = n1.chain.chain_id

    # The miner's blocks reach the second node, and the nodes find each other.
    wait_for(lambda: n2.chain.height >= 3, what="sync")
    wait_for(lambda: n2.p2p.node_id in {p.node_id for p in n1.p2p.all_peers()}, what="peer connection")

    # Pay the buyer some coins through node 1.
    pay = make_tx(miner, chain_id, "transfer", n1.pending_nonce(miner.address), to=buyer.address,
                  amount=params.YAPRAK_PER_COIN)
    http_post(f"{n1.url}/tx", pay)
    wait_for(lambda: n2.chain.state.balance(buyer.address) > 0, what="payment")

    # The buyer posts a job to node 2 only; node 1 must fetch the data itself.
    rng = np.random.default_rng(7)
    a = rng.integers(-128, 128, size=(N, 2 * N), dtype=np.int8)
    b = rng.integers(-128, 128, size=(2 * N, N), dtype=np.int8)
    job_id = http_post(f"{n2.url}/blob", txmod.encode_job(a, b), raw=True)["hash"]
    fields = {"job": job_id, "dims": [1, 2, 1], "price": params.YAPRAK_PER_COIN // 10}
    post = make_tx(buyer, chain_id, "job_post", n2.pending_nonce(buyer.address),
                   txmod.min_fee({"type": "job_post", **fields}), **fields)
    http_post(f"{n2.url}/tx", post)

    def done():
        job = n2.chain.state.jobs.get(job_id)
        return job is not None and len(job["done"]) == 2
    wait_for(done, timeout=120, what="job completion (commit, reveal, challenge window)")

    job = http_get(f"{n2.url}/job/{job_id}")
    c = np.zeros((N, N), dtype=np.int64)
    for key, result_hash in job["results"].items():
        c += txmod.decode_result(http_get(f"{n2.url}/blob/{result_hash}", raw=True))
    assert np.array_equal(c, a.astype(np.int64) @ b.astype(np.int64))
    assert job["escrow"] == 0
    assert n1.stats["useful_units"] >= 2

    # Both nodes agree on the chain.
    wait_for(lambda: n1.chain.tip == n2.chain.tip, what="same tip")


def test_buyer_watchtower_catches_a_cheating_miner(tmp_path, monkeypatch):
    """Validators never look at matmul results, so the buyer's node has to. Nobody
    tells it to: it checks the delivered units, disputes the wrong one, the miner
    cannot answer, and the whole batch is forfeit."""
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    # instant blocks: give the cheater and the watchtower enough blocks of wall time
    monkeypatch.setattr(params, "RESERVE_BLOCKS", 5000)
    monkeypatch.setattr(params, "JOB_CHALLENGE_BLOCKS", 5000)
    monkeypatch.setattr(params, "JOB_RESPONSE_BLOCKS", 10)
    buyer, cheat = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "b"), make_genesis(buyer.address, "buyer"), port=free_port(),
                miner_key=buyer, log=lambda m: None)
    node._choose_work = lambda: {"job": None, "unit": None}     # leave the job to the cheater
    node.start(mine=True)
    try:
        wait_for(lambda: node.chain.state.balance(buyer.address) > 5 * params.YAPRAK_PER_COIN, what="reward")
        cid = node.chain.chain_id
        rng = np.random.default_rng(3)
        a = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
        b = rng.integers(-128, 128, size=(N, 2 * N), dtype=np.int8)
        job_id = node.blobs.put(txmod.encode_job(a, b))
        price = 10**6
        fields = {"job": job_id, "dims": [1, 1, 2], "price": price}
        post = make_tx(buyer, cid, "job_post", node.pending_nonce(buyer.address),
                       txmod.min_fee({"type": "job_post", **fields}), **fields)
        fund = make_tx(buyer, cid, "transfer", post["nonce"] + 1, to=cheat.address, amount=10**8)
        http_post(f"{node.url}/tx", post)
        http_post(f"{node.url}/tx", fund)
        wait_for(lambda: job_id in node.chain.state.jobs and node.chain.state.balance(cheat.address) > 0,
                 what="job posted")

        # The cheat skips half the work: unit (0,0,1) is made up.
        units = [[0, 0, 0], [0, 0, 1]]
        results = [txmod.unit_inputs(a, b, u) for u in units]
        results = [ua.astype(np.int64) @ ub.astype(np.int64) for ua, ub in results]
        results[1] = np.zeros_like(results[1])
        hashes = [node.blobs.put(txmod.encode_result(c)) for c in results]   # published, just wrong
        manifest = node.blobs.put(txmod.encode_manifest(hashes))
        root = merkle_root(hashes)
        bitmap = txmod.units_bitmap(units, [1, 1, 2])
        bid = txmod.commitment(cheat.address, job_id, bitmap, root)
        http_post(f"{node.url}/tx", make_tx(cheat, cid, "job_commit", 0, job=job_id, units=bitmap, commitment=bid))
        wait_for(lambda: bid in node.chain.state.jobs[job_id]["batches"], what="commit mined")
        committed_at = node.chain.height
        wait_for(lambda: node.chain.height > committed_at, what="next block")
        http_post(f"{node.url}/tx", make_tx(cheat, cid, "job_reveal", 1, job=job_id, batch=bid,
                                            root=root, manifest=manifest))

        wait_for(lambda: node.stats["frauds_caught"] >= 1, timeout=60, what="watchtower dispute")
        wait_for(lambda: bid not in node.chain.state.jobs[job_id]["batches"], timeout=60, what="batch forfeit")
        job = node.chain.state.jobs[job_id]
        assert job["done"] == {} and job["escrow"] == 2 * price      # nothing paid, units reopen
        assert sorted(node.chain.state.open_units(job_id)) == units
        # the cheat lost the bond of the whole batch, not just of the wrong unit
        assert node.chain.state.balance(cheat.address) == 10**8 - 2 * params.bond_for(price) - 2 * params.MIN_TX_FEE
    finally:
        node.stop()


def test_miner_node_answers_a_dispute_with_a_merkle_proof(tmp_path, monkeypatch):
    """A griefer disputes an honest unit. The miner's node notices on its own and
    answers with the result and its Merkle proof; the batch survives."""
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    monkeypatch.setattr(params, "RESERVE_BLOCKS", 5000)
    monkeypatch.setattr(params, "JOB_CHALLENGE_BLOCKS", 5000)
    monkeypatch.setattr(params, "JOB_RESPONSE_BLOCKS", 30)
    miner, buyer, griefer = Key.generate(), Key.generate(), Key.generate()
    node = Node(str(tmp_path / "m"), make_genesis(miner.address, "miner"), port=free_port(),
                miner_key=miner, log=lambda m: None)
    node.start(mine=True)
    try:
        wait_for(lambda: node.chain.state.balance(miner.address) > 5 * params.YAPRAK_PER_COIN, what="reward")
        cid = node.chain.chain_id
        n0 = node.pending_nonce(miner.address)
        for i, k in enumerate((buyer, griefer)):
            http_post(f"{node.url}/tx", make_tx(miner, cid, "transfer", n0 + i, to=k.address, amount=10**8))
        wait_for(lambda: node.chain.state.balance(griefer.address) > 0, what="funding")
        rng = np.random.default_rng(4)
        a = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
        b = rng.integers(-128, 128, size=(N, 2 * N), dtype=np.int8)
        job_id = node.blobs.put(txmod.encode_job(a, b))
        fields = {"job": job_id, "dims": [1, 1, 2], "price": 10**6}
        http_post(f"{node.url}/tx", make_tx(buyer, cid, "job_post", 0, txmod.min_fee({"type": "job_post", **fields}),
                                            **fields))

        def revealed():
            job = node.chain.state.jobs.get(job_id)
            return job and any(bt["status"] == "revealed" for bt in job["batches"].values())
        wait_for(revealed, timeout=60, what="batch revealed")
        bid, batch = next((k, v) for k, v in node.chain.state.jobs[job_id]["batches"].items()
                          if v["status"] == "revealed")
        unit = txmod.bitmap_units(batch["units"], [1, 1, 2])[0]
        key = txmod.unit_key(unit)
        dispute = make_tx(griefer, cid, "job_challenge", 0, txmod.job_challenge_fee(1), job=job_id, batch=bid,
                          units=txmod.units_bitmap([unit], [1, 1, 2]))
        http_post(f"{node.url}/tx", dispute)
        wait_for(lambda: key in node.chain.state.jobs[job_id]["batches"][bid]["proven"], timeout=60,
                 what="the miner's node answers the dispute")
        assert node.chain.state.jobs[job_id]["batches"][bid]["disputes"] == {}
        assert node.chain.state.open_disputes == 0
    finally:
        node.stop()


def test_miner_does_not_recommit_units_waiting_in_the_mempool(tmp_path):
    """With realistic block times a batch commit waits in the mempool. The miner
    must treat its units as taken; v0.4 briefly read the commit's bitmap as a
    list, so it picked them again and every new commit was rejected."""
    from helpers import T0, mine, tx
    key = Key.generate()
    node = Node(str(tmp_path / "c"), make_genesis(key.address, "busy", timestamp=T0), port=free_port(),
                miner_key=key, log=lambda m: None)
    rng = np.random.default_rng(8)
    a = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
    b = rng.integers(-128, 128, size=(N, 2 * N), dtype=np.int8)
    job_id = node.blobs.put(txmod.encode_job(a, b))
    block, _ = mine(node.chain, key, [tx(node.chain, key, "job_post", job=job_id, dims=[1, 1, 2], price=1000)])
    node._add_block(block)
    for unit in ([0, 0, 0], [0, 0, 1]):
        ua, ub = txmod.unit_inputs(a, b, unit)
        node._buffer_result({"job": job_id, "unit": unit}, ua.astype(np.int64) @ ub.astype(np.int64))
    node._flush_batches()
    assert [t["type"] for t in node.mempool.values()] == ["job_commit"]
    assert {(job_id, "0,0,0"), (job_id, "0,0,1")} <= node._busy_units()
    assert node._choose_work() == {"job": None, "unit": None}       # nothing left to sell: filler work


def test_node_works_a_wasm_task_end_to_end(tmp_path, monkeypatch):
    wasmtime = pytest.importorskip("wasmtime")
    from green import wasm
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)   # instant blocks
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    # A short half-life lets ASERT slow the instant blocks to ~1 per second within
    # seconds; otherwise a 3-block reservation lasts a fraction of the worker's
    # 1-second loop and the task can never be revealed (in production it lasts
    # 20 blocks x 15 s).
    monkeypatch.setattr(params, "ASERT_HALFLIFE", 20)
    monkeypatch.setattr(params, "TASK_RESERVE_BLOCKS", 3)
    monkeypatch.setattr(params, "TASK_CHALLENGE_BLOCKS", 4)

    sumwat = wasmtime.wat2wasm(r"""(module
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

    miner = Key.generate()
    genesis = make_genesis(miner.address, "task net")
    node = Node(str(tmp_path / "t"), genesis, port=free_port(), miner_key=miner, log=lambda m: None)
    node.start(mine=True)
    try:
        wait_for(lambda: node.chain.state.balance(miner.address) > 10 * params.YAPRAK_PER_COIN, what="reward")
        prog = node.blobs.put(sumwat)
        data = bytes([3, 4, 5])
        inp = node.blobs.put(data)
        fields = {"prog": prog, "input": inp, "fuel": 1_000_000, "price": params.YAPRAK_PER_COIN}
        post = make_tx(miner, node.chain.chain_id, "task_post", node.pending_nonce(miner.address),
                       txmod.task_post_fee(len(sumwat), len(data)), **fields)
        http_post(f"{node.url}/tx", post)
        task_id = txmod.task_id_of(post)

        wait_for(lambda: node.chain.state.tasks.get(task_id, {}).get("status") == "done",
                 timeout=60, what="task settled")
        # ASERT worked inside the live node: the instant blocks were slowed down.
        assert int(node.chain.tip_header()["target"], 16) < params.MAX_TARGET // 4
        task = node.chain.state.tasks[task_id]
        want = wasm.result_hash(wasm.run(sumwat, data)[0])
        assert task["output"] == want
        assert task["worker"] == miner.address
    finally:
        node.stop()


def test_watchtower_catches_a_lying_worker(tmp_path, monkeypatch):
    """Nobody is told to challenge: the node notices the wrong result on its own,
    proves it, and takes the cheater's bond."""
    wasmtime = pytest.importorskip("wasmtime")
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    # instant blocks: give the cheater and the watchtower enough blocks of wall time
    monkeypatch.setattr(params, "TASK_RESERVE_BLOCKS", 5000)
    monkeypatch.setattr(params, "TASK_CHALLENGE_BLOCKS", 5000)
    prog = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
      (func (export "green_run") (param i32 i32) (result i64)
        (i32.store (i32.const 0) (i32.const 42)) (i64.const 4)))""")

    miner, cheat = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "w"), make_genesis(miner.address, "watch"), port=free_port(),
                miner_key=miner, log=lambda m: None)
    node.work_tasks = False          # leave the task to the cheater
    node.start(mine=True)
    try:
        wait_for(lambda: node.chain.state.balance(miner.address) > 5 * params.YAPRAK_PER_COIN, what="reward")
        cid = node.chain.chain_id
        ph, ih = node.blobs.put(prog), node.blobs.put(b"x")
        post = make_tx(miner, cid, "task_post", node.pending_nonce(miner.address),
                       txmod.task_post_fee(len(prog), 1), prog=ph, input=ih, fuel=10_000, price=10**6)
        fund = make_tx(miner, cid, "transfer", post["nonce"] + 1, to=cheat.address, amount=10**8)
        http_post(f"{node.url}/tx", post)
        http_post(f"{node.url}/tx", fund)
        task_id = txmod.task_id_of(post)
        wait_for(lambda: task_id in node.chain.state.tasks and node.chain.state.balance(cheat.address) > 0,
                 what="task posted")

        lie = "ab" * 32                                  # the program always outputs 42; this is not its hash
        commit = make_tx(cheat, cid, "task_commit", 0, task=task_id,
                         commitment=txmod.task_commitment(cheat.address, task_id, lie))
        http_post(f"{node.url}/tx", commit)
        wait_for(lambda: (node.chain.state.tasks[task_id]["reserved"] or {}).get("worker") == cheat.address,
                 what="commit mined")
        committed_at = node.chain.height
        wait_for(lambda: node.chain.height > committed_at, what="next block")
        http_post(f"{node.url}/tx", make_tx(cheat, cid, "task_reveal", 1, task=task_id, output=lie))

        wait_for(lambda: node.chain.state.tasks[task_id]["status"] == "failed", timeout=60,
                 what="watchtower challenge")
        assert node.stats["frauds_caught"] >= 1
        assert node.chain.state.balance(cheat.address) < 10**8   # lost its bond
    finally:
        node.stop()


def test_worker_commits_once_when_blocks_are_slow(tmp_path, monkeypatch):
    """With realistic block times a commit waits in the mempool for a while.
    The worker used to forget it during that wait, never reveal it, lose the
    bond and commit the same task again. Instant test blocks hid this."""
    wasmtime = pytest.importorskip("wasmtime")
    # A few seconds per block on a CPU (a unit verifies in ~9 ms since v0.7). Block times
    # are random, so two can land a fraction of a second apart: the reservation leaves the
    # reveal 4 blocks of slack after its 2 confirmations (the real network leaves 18), or
    # this would test luck against the worker's 1 s loop instead of the worker.
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET // 400)
    monkeypatch.setattr(params, "TASK_RESERVE_BLOCKS", 6)
    monkeypatch.setattr(params, "TASK_CHALLENGE_BLOCKS", 2)
    prog = wasmtime.wat2wasm(r"""(module (memory (export "memory") 1)
      (func (export "green_run") (param i32 i32) (result i64)
        (i32.store (i32.const 0) (i32.const 7)) (i64.const 4)))""")
    key = Key.generate()
    node = Node(str(tmp_path / "s"), make_genesis(key.address, "slow"), port=free_port(),
                miner_key=key, log=lambda m: None)
    node.start(mine=True)
    try:
        ph, ih = node.blobs.put(prog), node.blobs.put(b"-")
        post = make_tx(key, node.chain.chain_id, "task_post", node.pending_nonce(key.address),
                       txmod.task_post_fee(len(prog), 1), prog=ph, input=ih, fuel=10_000,
                       price=params.YAPRAK_PER_COIN // 100)
        http_post(f"{node.url}/tx", post)
        task_id = txmod.task_id_of(post)
        wait_for(lambda: node.chain.state.tasks.get(task_id, {}).get("status") == "done",
                 timeout=150, what="task settled")
        assert node.stats["tasks_done"] == 1, "the worker committed the same task more than once"
    finally:
        node.stop()


def test_reorg_returns_abandoned_transactions_to_mempool(tmp_path):
    from helpers import T0, mine
    from green.chain import block_hash
    key, other = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "r"), make_genesis(key.address, "reorg", timestamp=T0), port=free_port(),
                miner_key=key, log=lambda m: None)
    chain = node.chain
    g = chain.tip
    pay = make_tx(key, chain.chain_id, "transfer", 0, to=other.address, amount=12345)
    a1, _ = mine(chain, key, [pay])                 # branch A confirms the payment
    node._add_block(a1)
    assert chain.state.balance(other.address) == 12345 and not node.mempool

    b1, _ = mine(chain, other, [], t=T0 + 11, parent=g)
    node._add_block(b1)
    b2, _ = mine(chain, other, [], t=T0 + 21, parent=block_hash(b1))
    node._add_block(b2)                             # branch B is heavier and never saw the payment
    assert chain.tip == block_hash(b2)
    assert chain.state.balance(other.address) == 2 * params.block_reward(1)
    # The payment is not lost: it is back in the mempool and will confirm on B.
    assert txmod.txid(pay) in node.mempool
    assert node._pending.balance(other.address) == 2 * params.block_reward(1) + 12345


def test_restart_loads_blocks_from_disk(network):
    n1, n2, miner, buyer = network
    wait_for(lambda: n1.chain.height >= 3, what="blocks")
    n1.mining = False
    time.sleep(0.5)
    height, tip = n1.chain.height, n1.chain.tip
    genesis = n1.chain.block(n1.chain.genesis_hash)
    reopened = Node(n1.data_dir, genesis, port=free_port(), miner_key=miner, log=lambda m: None)
    assert reopened.chain.height >= height
    assert reopened.chain.has_block(tip)


def test_supply_audit_and_metrics_on_a_live_node(network):
    n1, n2, miner, buyer = network
    wait_for(lambda: n1.chain.height >= 5, what="blocks")
    s = http_get(f"{n1.url}/supply")
    assert s["ok"] and s["total"] == s["minted"] and s["minted"] >= 6 * params.block_reward(0)
    import urllib.request
    with urllib.request.urlopen(f"{n1.url}/metrics", timeout=5) as r:
        assert r.headers["Content-Type"].startswith("text/plain")
        text = r.read().decode()
    values = {line.split()[0]: float(line.split()[1]) for line in text.splitlines() if line and not line.startswith("#")}
    for name in ("green_height", "green_peers", "green_mempool_transactions", "green_blocks_found_total",
                 "green_open_disputes", "green_frauds_caught_total"):
        assert name in values, name
    assert values["green_height"] >= 5 and values["green_mining"] == 1


def test_cli_guard_commands_end_to_end(tmp_path, monkeypatch):
    """The guardian workflow through the real command line, against a live node."""
    from green import cli
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    monkeypatch.setattr(params, "ASERT_HALFLIFE", 20)
    miner = Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(miner.address, "cli"), port=free_port(), miner_key=miner,
                log=lambda m: None)
    node.start(mine=True)
    try:
        agent_dir, guardian_dir = str(tmp_path / "agent"), str(tmp_path / "guardian")
        common = ["--node", node.url]
        for d in (agent_dir, guardian_dir):
            assert cli.main(["--data", d, "wallet", "new", "--no-password"]) == 0
        agent, guardian = cli.wallet_address(agent_dir), cli.wallet_address(guardian_dir)
        wait_for(lambda: node.chain.state.balance(miner.address) > 3 * params.YAPRAK_PER_COIN, what="reward")
        cid = node.chain.chain_id
        for i, to in enumerate((agent, guardian)):
            http_post(f"{node.url}/tx", make_tx(miner, cid, "transfer", node.pending_nonce(miner.address),
                                                to=to, amount=params.YAPRAK_PER_COIN))
        wait_for(lambda: node.chain.state.balance(guardian) > 0, what="funding")
        assert cli.main(["--data", agent_dir, *common, "guard", "set", guardian, "0.5", "1000"]) == 0
        wait_for(lambda: node.account(agent)["guard"] is not None, what="guard on chain")
        assert node.account(agent)["guard"]["limit"] == params.YAPRAK_PER_COIN // 2
        assert cli.main(["--data", guardian_dir, *common, "guard", "update", agent, "--freeze"]) == 0
        wait_for(lambda: node.account(agent)["guard"]["frozen"], what="freeze on chain")
        assert cli.main(["--data", agent_dir, *common, "send", guardian, "0.01"]) == 1      # refused: frozen
        assert cli.main(["--data", guardian_dir, *common, "guard", "release", agent]) == 0
        wait_for(lambda: node.account(agent)["guard"] is None, what="release on chain")
        assert cli.main(["--data", agent_dir, *common, "send", guardian, "0.01"]) == 0
        assert cli.main(["--data", agent_dir, *common, "supply"]) == 0
    finally:
        node.stop()
