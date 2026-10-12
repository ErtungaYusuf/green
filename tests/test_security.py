"""Regression tests for every issue found in the security audit (SECURITY.md).

Each test replays an attack that worked against v0.1 and checks it now fails.
"""

import json
import random
import socket
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from green import node as nodemod, params, tx as txmod, work
from green.chain import Chain, InvalidBlock, MissingData, make_genesis
from green.crypto import Key
from green.node import Node, TooLarge, http_get, http_post
from green.p2p import parse_address
from green.tx import InvalidTx
from helpers import (SALES, T0, add, challenge, commit, minted, mine, new_chain, post_job, respond, reveal,
                     settle_batch, supply, tx, unit_result)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def node(tmp_path):
    key = Key.generate()
    n = Node(str(tmp_path / "n"), make_genesis(key.address, "sec"), port=free_port(), miner_key=key,
             log=lambda m: None)
    n.start()
    yield n
    n.stop()


# --- consensus ------------------------------------------------------------------

def test_S1_mutated_copy_cannot_blacklist_honest_block():
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    txs = [tx(chain, alice, "transfer", to=bob.address, amount=1, nonce=i) for i in range(3)]
    honest, _ = mine(chain, alice, txs)
    for mutated_txs in (honest["txs"] + [honest["txs"][-1]], honest["txs"][:2], honest["txs"][::-1], []):
        with pytest.raises(InvalidBlock) as e:
            chain.add_block({"header": honest["header"], "txs": mutated_txs})
        assert not e.value.permanent
    assert chain.add_block(honest) == "added"


def test_S2_result_cannot_be_stolen():
    rng = np.random.default_rng(10)
    buyer, miner, thief = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(buyer)
    add(chain, buyer, [tx(chain, buyer, "transfer", to=miner.address, amount=10**7),
                       tx(chain, buyer, "transfer", to=thief.address, amount=10**7, nonce=1)])
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 1))
    c = unit_result(a, b, [0, 0, 0])
    commit_tx, sale = commit(chain, miner, job_id, [[0, 0, 0]], [c])
    add(chain, miner, [commit_tx])
    bid = commit_tx["commitment"]
    honest_reveal = reveal(chain, miner, job_id, bid, sale)

    # The thief saw the reveal (and so the root) in the mempool: it can neither
    # reveal the batch nor commit the unit, not even with the same root.
    h = chain.height + 1
    stolen = tx(chain, thief, "job_reveal", job=job_id, batch=bid, root=sale.root, manifest=sale.manifest, fee=10**6)
    with pytest.raises(InvalidTx, match="no commitment"):
        chain.apply_tx(chain.state.copy(), stolen, h)
    bitmap = commit_tx["units"]
    thief_commit = tx(chain, thief, "job_commit", job=job_id, units=bitmap, fee=10**6,
                      commitment=txmod.commitment(thief.address, job_id, bitmap, sale.root))
    with pytest.raises(InvalidTx, match="reserved"):
        chain.apply_tx(chain.state.copy(), thief_commit, h)
    add(chain, miner, [honest_reveal])
    settle_batch(chain, miner, job_id, bid)
    assert chain.state.jobs[job_id]["done"]["0,0,0"] == bid
    assert chain.state.jobs[job_id]["delivered"][bid]["miner"] == miner.address
    with pytest.raises(InvalidTx, match="already delivered"):
        chain.apply_tx(chain.state.copy(), thief_commit, chain.height + 1)


def test_S3_cancel_cannot_dodge_payment():
    rng = np.random.default_rng(11)
    buyer, miner = Key.generate(), Key.generate()
    chain = new_chain(buyer)
    add(chain, buyer, [tx(chain, buyer, "transfer", to=miner.address, amount=10**7)])
    a, b, job_id = post_job(chain, buyer, rng, dims=(1, 1, 2), price=10**6)
    c = unit_result(a, b, [0, 0, 0])
    commit_tx, sale = commit(chain, miner, job_id, [[0, 0, 0]], [c])
    add(chain, miner, [commit_tx])
    bid = commit_tx["commitment"]
    # The buyer sees the result coming and cancels first.
    add(chain, buyer, [tx(chain, buyer, "job_cancel", job=job_id)])
    miner_before = chain.state.balance(miner.address)
    add(chain, buyer, [reveal(chain, miner, job_id, bid, sale)])
    # No new commitments after a cancel.
    late, _ = commit(chain, miner, job_id, [[0, 0, 1]], [unit_result(a, b, [0, 0, 1])])
    with pytest.raises(InvalidTx, match="cancelled"):
        chain.apply_tx(chain.state.copy(), late, chain.height + 1)
    # The cancel takes effect while the batch is still in its challenge window: the
    # batch keeps its share of the escrow, the buyer gets back only the rest.
    buyer_before, h0 = chain.state.balance(buyer.address), chain.height
    while not chain.state.jobs[job_id]["closed"]:
        add(chain, buyer)
    rewards = sum(params.block_reward(h) for h in range(h0 + 1, chain.height + 1))
    assert chain.state.balance(buyer.address) == buyer_before + 10**6 + rewards
    assert bid in chain.state.jobs[job_id]["batches"]
    settle_batch(chain, buyer, job_id, bid)
    assert chain.state.balance(miner.address) == miner_before + 10**6 + params.bond_for(10**6) - params.MIN_TX_FEE
    assert chain.state.jobs[job_id]["escrow"] == 0
    assert supply(chain) == minted(chain)


def test_S4_transactions_cannot_be_replayed_on_another_network():
    alice, bob = Key.generate(), Key.generate()
    testnet, mainnet = new_chain(alice, "testnet"), new_chain(alice, "mainnet")
    t = tx(testnet, alice, "transfer", to=bob.address, amount=10**7)
    testnet.apply_tx(testnet.state.copy(), t, 1)
    with pytest.raises(InvalidTx, match="signature"):
        mainnet.apply_tx(mainnet.state.copy(), t, 1)


def test_S7_job_data_pays_for_its_storage():
    alice = Key.generate()
    chain = new_chain(alice)
    a = np.zeros((4 * 256, 4 * 256), dtype=np.int8)
    job_id = chain.blobs.put(txmod.encode_job(a, a))
    free = tx(chain, alice, "job_post", job=job_id, dims=[4, 4, 4], price=1, fee=params.MIN_TX_FEE)
    with pytest.raises(InvalidTx, match="fee"):
        chain.apply_tx(chain.state.copy(), free, 1)
    assert txmod.min_fee(free) == params.MIN_TX_FEE + txmod.job_blob_size([4, 4, 4])


def test_S9_timestamp_must_pass_median_time():
    alice = Key.generate()
    chain = new_chain(alice)
    for i in range(1, 6):
        add(chain, alice, t=T0 + 100 * i)
    block, _ = mine(chain, alice, t=T0 + 1)
    block["header"]["time"] = T0 + 300               # == median of the last blocks
    from green.chain import solve
    block["header"]["transcript"], _, _ = solve(block["header"], chain.blobs)
    with pytest.raises(InvalidBlock, match="median"):
        chain.add_block(block)


def test_S10_future_block_is_not_blacklisted():
    alice = Key.generate()
    chain = new_chain(alice)
    block, _ = mine(chain, alice, t=int(time.time()) + 3600)
    with pytest.raises(InvalidBlock, match="future") as e:
        chain.add_block(block)
    assert not e.value.permanent


def test_S11_deep_forks_are_refused(monkeypatch):
    monkeypatch.setattr(params, "MAX_REORG_DEPTH", 3)
    alice = Key.generate()
    chain = new_chain(alice)
    g = chain.tip
    for _ in range(5):
        add(chain, alice)
    fork, _ = mine(chain, alice, t=T0 + 1, parent=g)
    with pytest.raises(InvalidBlock, match="too deep") as e:
        chain.add_block(fork)
    assert not e.value.permanent


def test_S16_money_is_conserved_under_random_activity(monkeypatch):
    """Fuzz: random transfers, jobs, batch commits, reveals, disputes, responses and
    cancels -- honest, cheating and withholding miners -- never create or lose coins."""
    for name, value in (("RESERVE_BLOCKS", 6), ("JOB_CHALLENGE_BLOCKS", 4),
                        ("JOB_RESPONSE_BLOCKS", 3), ("CANCEL_DELAY", 5)):
        monkeypatch.setattr(params, name, value)
    paths = {"paid": 0, "forfeit": 0, "expired": 0, "dropped_after_close": 0}
    pay, drop = Chain._pay_batch, Chain._drop_batch

    def counting_pay(self, s, job, batch_id):
        paths["paid"] += 1
        return pay(self, s, job, batch_id)

    def counting_drop(self, s, job, batch_id):
        paths["forfeit" if job["batches"][batch_id]["disputes"] else "expired"] += 1
        paths["dropped_after_close"] += job["closed"]
        return drop(self, s, job, batch_id)
    monkeypatch.setattr(Chain, "_pay_batch", counting_pay)
    monkeypatch.setattr(Chain, "_drop_batch", counting_drop)

    rng = np.random.default_rng(99)
    pick = random.Random(99)
    keys = [Key.generate() for _ in range(4)]
    by_addr = {k.address: k for k in keys}
    chain = new_chain(keys[0])
    for k in keys[1:]:
        add(chain, keys[0], [tx(chain, keys[0], "transfer", to=k.address, amount=10**7)])
    jobs = {}           # job id -> (a, b)
    committed = []      # (miner, job_id, batch id) of commits that made it on chain
    meta = {}
    seen = set()
    rejected = {}

    def next_nonce(key, txs):
        return chain.state.nonce(key.address) + sum(1 for t in txs if t["pub"] == key.pub_hex)

    def open_units(job_id, bt):
        units = txmod.bitmap_units(bt["units"], chain.state.jobs[job_id]["dims"])
        return [u for u in units if txmod.unit_key(u) not in bt["disputes"] and txmod.unit_key(u) not in bt["proven"]]

    for step in range(200):
        actor = pick.choice(keys)
        txs = []
        roll = pick.random()
        state = chain.state
        h = chain.height + 1
        if roll < 0.1 and len(jobs) < 8:
            a, b, job_id = post_job(chain, actor, rng, dims=(1, pick.randint(1, 2), pick.randint(1, 3)),
                                    price=pick.randint(1, 10**5))
            jobs[job_id] = (a, b)
        elif roll < 0.35 and jobs:
            job_id = pick.choice(list(jobs))
            units = state.open_units(job_id)
            if units and state.balance(actor.address) > 10**6:
                units = pick.sample(units, pick.randint(1, len(units)))
                a, b = jobs[job_id]
                results = [unit_result(a, b, u) for u in units]
                if pick.random() < 0.3:
                    results[-1] = results[-1] + 1        # some miners cheat
                t, _ = commit(chain, actor, job_id, units, results, publish=pick.random() > 0.15,
                              publish_manifest=pick.random() > 0.1)
                txs.append(t)
                meta[t["commitment"]] = (actor, job_id, t["commitment"])
        elif roll < 0.55 and committed:
            miner, job_id, bid = committed.pop()
            txs.append(reveal(chain, miner, job_id, bid))
        elif roll < 0.62:
            revealed = [(j, bid, bt) for j, job in state.jobs.items() for bid, bt in job["batches"].items()
                        if bt["status"] == "revealed" and open_units(j, bt)]
            if revealed:                                  # a griefer disputes anything
                job_id, bid, bt = pick.choice(revealed)
                txs.append(challenge(chain, actor, job_id, bid, [pick.choice(open_units(job_id, bt))]))
        elif roll < 0.9 and jobs:
            job_id = pick.choice(list(jobs))
            job = state.jobs[job_id]
            if job["closes_at"] is None and not job["closed"]:
                txs.append(tx(chain, by_addr[job["buyer"]], "job_cancel", job=job_id))
        elif roll < 0.95:
            # Agents put themselves under guardians; guardians freeze, loosen and release.
            g = state.guards.get(actor.address)
            if g is None:
                guardian = pick.choice([k for k in keys if k is not actor])
                txs.append(tx(chain, actor, "guard_set", guardian=guardian.address,
                              limit=pick.randint(10**5, 10**7), period=pick.randint(2, 12)))
            else:
                guardian = by_addr[g["guardian"]]
                if pick.random() < 0.3:
                    txs.append(tx(chain, guardian, "guard_release", account=actor.address))
                else:
                    txs.append(tx(chain, guardian, "guard_update", account=actor.address, guardian=guardian.address,
                                  limit=pick.randint(10**5, 10**7), period=pick.randint(2, 12),
                                  frozen=pick.random() < 0.2))
        else:
            other = pick.choice(keys)
            if state.balance(actor.address) > 10**5:
                txs.append(tx(chain, actor, "transfer", to=other.address, amount=pick.randint(1, 10**4)))
        # Buyers check what they get (the node's watchtower) and usually dispute every
        # bad or unreachable unit at once.
        for job_id, job in state.jobs.items():
            a, b = jobs[job_id]
            buyer = by_addr[job["buyer"]]
            for bid, bt in job["batches"].items():
                if bt["status"] != "revealed" or pick.random() > 0.7:
                    continue
                sale = SALES[bid]
                if chain.blobs.has(sale.manifest):
                    bad = [u for u, r in zip(sale.units, sale.leaves) if not chain.blobs.has(r)
                           or not np.array_equal(txmod.decode_result(chain.blobs.get(r)), unit_result(a, b, u))]
                else:
                    bad = list(sale.units)               # no manifest: nothing can be checked
                bad = [u for u in bad if u in open_units(job_id, bt)]
                if bad:
                    txs.append(challenge(chain, buyer, job_id, bid, bad, nonce=next_nonce(buyer, txs)))
        # Miners usually defend their batches (a cheat's or withholder's answer is invalid).
        for job_id, job in state.jobs.items():
            for bid, bt in job["batches"].items():
                if bt["disputes"] and pick.random() < 0.6:
                    miner = by_addr[bt["miner"]]
                    for key in list(bt["disputes"])[:3]:
                        unit = [int(x) for x in key.split(",")]
                        txs.append(respond(chain, miner, job_id, bid, unit, nonce=next_nonce(miner, txs)))
        # Keep only what is valid right now (expired, wrong, withheld, cancelled...).
        valid, s = [], state.copy()
        for t in txs:
            before = s.fingerprint()
            try:
                chain.apply_tx(s, t, h)
                valid.append(t)
                seen.add(t["type"])
                if t["type"] == "job_commit":
                    committed.append(meta[t["commitment"]])
            except (InvalidTx, MissingData):
                # A rejected transaction must leave no trace: the mempool applies in place.
                assert s.fingerprint() == before, f"a rejected {t['type']} changed the state at step {step}"
                rejected[t["type"]] = rejected.get(t["type"], 0) + 1
        add(chain, pick.choice(keys), valid)
        assert supply(chain) == minted(chain), f"money leaked at step {step}"
        # (helpers.add also checked that the incremental state root equals a full recomputation)
        assert all(v >= 0 for v in chain.state.balances.values())
        open_total = 0
        for job in chain.state.jobs.values():
            # Escrow is exactly what is still owed: every undelivered unit while the job
            # is open, only the units of live batches once it is closed.
            live = sum(bt["count"] for bt in job["batches"].values())
            owed = live if job["closed"] else len(txmod.all_units(job["dims"])) - len(job["done"])
            assert job["escrow"] == job["price"] * owed, f"escrow drifted at step {step}"
            open_total += sum(len(bt["disputes"]) for bt in job["batches"].values())
        assert chain.state.open_disputes == open_total, f"dispute backlog drifted at step {step}"
        for addr, g in chain.state.guards.items():
            assert g["spent"] <= g["limit"] or g["window"] != (chain.height // g["period"]), \
                f"a guarded account overspent at step {step}"
    # make sure the fuzz really went through every path
    jobs_state = chain.state.jobs.values()
    assert sum(len(j["done"]) for j in jobs_state) >= 3
    assert any(j["closes_at"] is not None for j in jobs_state)
    assert {"job_commit", "job_reveal", "job_challenge", "job_respond", "job_cancel",
            "guard_set", "guard_update", "guard_release"} <= seen, seen
    assert all(paths.values()), paths
    assert sum(rejected.values()) >= 5, rejected        # the atomicity check really ran on failures


def test_S20_freivalds_cannot_be_fooled_by_modular_wraparound():
    """The field Freivalds check reduces mod p. Without a range check, C = AB - p*E
    would pass every round (it is AB mod p); the bound on |C| must stop it."""
    rng = np.random.default_rng(20)
    a = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
    b = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
    c = a.astype(np.int64) @ b.astype(np.int64)
    p = params.FREIVALDS_PRIME
    assert work.freivalds_check(a, b, c.astype(np.int32), b"seed")
    e = np.zeros_like(c)
    e[3, 9] = 1
    assert not work.freivalds_check(a, b, c - p * e, b"seed")
    wrapped = c.astype(np.int32)
    wrapped[0, 0] = np.iinfo(np.int32).min                 # |INT32_MIN| overflows in int32
    assert not work.freivalds_check(a, b, wrapped, b"seed")
    for seed in range(50):                                 # any single wrong entry is caught
        bad = c.copy()
        bad[seed % 256, (seed * 7) % 256] += 1 + seed
        assert not work.freivalds_check(a, b, bad, bytes([seed]))
    assert not work.freivalds_check(a, b, c[:, :255], b"seed")     # wrong shape


def test_S21_freivalds_is_much_cheaper_than_recomputing():
    """Verification has to stay far cheaper than the work, or checking every
    delivered result would cost the buyer as much as doing it."""
    import timeit
    rng = np.random.default_rng(21)
    a = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
    b = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
    c = a.astype(np.int64) @ b.astype(np.int64)
    check = min(timeit.repeat(lambda: work.freivalds_check(a, b, c, b"s"), number=5, repeat=3))
    redo = min(timeit.repeat(lambda: a.astype(np.int64) @ b.astype(np.int64), number=5, repeat=3))
    assert check * 5 < redo, (check, redo)


# --- network --------------------------------------------------------------------

def test_S5_http_requests_cannot_inject_peers_or_blocks(node):
    """Peers and blocks only come over the P2P protocol. The old HTTP gossip
    endpoints are gone, and no header can make the node connect anywhere."""
    for i in range(40):
        req = urllib.request.Request(f"{node.url}/tx", data=b"{}", method="POST",
                                     headers={"X-Green-From": f"http://10.0.0.{i}:9999"})
        try:
            urllib.request.urlopen(req, timeout=3)
        except urllib.error.HTTPError:
            pass
    for path, body in (("/announce", {"url": "http://127.0.0.1:1"}), ("/block", {"header": {}, "txs": []})):
        with pytest.raises(urllib.error.HTTPError) as e:
            http_post(node.url + path, body)
        assert e.value.code == 404
    time.sleep(1)
    assert node.p2p.all_peers() == [] and len(node.p2p.addrman) == 0


def test_S5b_peer_addresses_are_strictly_parsed():
    assert parse_address("127.0.0.1:8546") == ("127.0.0.1", 8546, None)
    assert parse_address("[::1]:8546") == ("::1", 8546, None)
    assert parse_address("ab" * 32 + "@node.example:1") == ("node.example", 1, "ab" * 32)
    for bad in ["http://127.0.0.1:8546", "127.0.0.1", "h:99999", "h:0", "::1:8546", "a b:1", "h:1/x",
                "zz@h:1", "u@" + "ab" * 32 + "@h:1", 42, None, "x" * 400]:
        assert parse_address(bad) is None, bad


def raw_request(port: int, payload: bytes, timeout: float = 5) -> bytes:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.sendall(payload)
            chunks = []
            while True:
                try:
                    data = s.recv(65536)
                except socket.timeout:
                    break
                if not data:
                    break
                chunks.append(data)
            return b"".join(chunks)
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
        return b""      # the server dropped us (over the connection cap) -> no response


def test_S6_bad_content_length_is_rejected_quickly(node):
    for cl in (b"-1", b"abc", b"99999999999"):
        started = time.time()
        resp = raw_request(node.port, b"POST /tx HTTP/1.1\r\nHost: x\r\nContent-Length: " + cl + b"\r\n\r\n{}")
        assert resp.startswith(b"HTTP/1.0 4"), resp[:80]
        assert time.time() - started < 3


def test_S12_admin_endpoints_refuse_remote_callers(node, monkeypatch):
    monkeypatch.setattr(nodemod, "LOOPBACK", set())       # pretend every caller is remote
    for path, body in (("/mining", {"on": True}), ("/peers", {"addr": "1.2.3.4:5"})):
        with pytest.raises(urllib.error.HTTPError) as e:
            http_post(node.url + path, body)
        assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        http_post(node.url + "/blob", b"x" * 100, raw=True)
    assert e.value.code == 403
    assert not node.mining


def test_S13_committed_results_are_not_served_before_reveal(node):
    c = np.ones((256, 256), dtype=np.int64)
    secret = node.blobs.put(txmod.encode_result(c))
    with pytest.raises(urllib.error.HTTPError) as e:
        http_get(f"{node.url}/blob/{secret}", raw=True)
    assert e.value.code == 404


def test_S14_responses_are_size_limited(node):
    big = np.zeros((1024, 1024), dtype=np.int8)
    job_id = http_post(f"{node.url}/blob", txmod.encode_job(big, big), raw=True)["hash"]
    node.mempool["x"] = {"type": "job_post", "job": job_id}       # make it public
    assert len(http_get(f"{node.url}/blob/{job_id}", raw=True, limit=3 * 2**20)) > 2**20
    with pytest.raises(TooLarge):
        http_get(f"{node.url}/blob/{job_id}", raw=True, limit=1000)


def test_S15_mempool_is_capped(node, monkeypatch):
    monkeypatch.setattr(nodemod, "MAX_MEMPOOL", 2)
    key = node.miner_key
    for i in range(2):
        node.submit_tx(txmod.make_tx(key, node.chain.chain_id, "transfer", i, to=Key.generate().address, amount=1))
    with pytest.raises(InvalidTx, match="full"):
        node.submit_tx(txmod.make_tx(key, node.chain.chain_id, "transfer", 2, to=Key.generate().address, amount=1))


def test_S17_oversized_body_is_rejected(node):
    resp = raw_request(node.port, b"POST /tx HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n"
                       % (nodemod.MAX_JSON_BYTES + 1))
    assert resp.startswith(b"HTTP/1.0 413")


def test_S18_connection_flood_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(nodemod, "MAX_CONCURRENT_REQUESTS", 3)
    monkeypatch.setattr(nodemod, "REQUEST_TIMEOUT", 2)
    key = Key.generate()
    n = Node(str(tmp_path / "f"), make_genesis(key.address, "flood"), port=free_port(), log=lambda m: None)
    n.start()
    try:
        idle = [socket.create_connection(("127.0.0.1", n.port)) for _ in range(3)]
        time.sleep(0.3)
        # The slots are full: an extra connection is dropped instead of piling up threads.
        assert raw_request(n.port, b"GET /status HTTP/1.0\r\n\r\n", timeout=1) == b""
        # Idle connections time out, and the node serves again.
        time.sleep(2.5)
        assert json.loads(raw_request(n.port, b"GET /status HTTP/1.0\r\n\r\n").split(b"\r\n\r\n", 1)[1])["height"] == 0
        for s in idle:
            s.close()
    finally:
        n.stop()


def test_S19_data_is_only_fetched_after_cheap_checks(node):
    """A transaction that fails cheap checks never makes the node download its data."""
    poor = Key.generate()
    job_id = "ab" * 32
    t = txmod.make_tx(poor, node.chain.chain_id, "job_post", 0, fee=txmod.min_fee(
        {"type": "job_post", "dims": [16, 16, 16]}), job=job_id, dims=[16, 16, 16], price=10**8)
    calls = []
    node._fetch_blob = lambda *a, **k: calls.append(a) or False
    with pytest.raises(InvalidTx, match="balance"):
        node.submit_tx(t)
    assert calls == []


def test_S22_a_node_loop_survives_an_unexpected_error(tmp_path, monkeypatch):
    """The miner keeps going after an error in one step instead of dying silently."""
    key = Key.generate()
    n = Node(str(tmp_path / "n"), make_genesis(key.address, "survive"), port=free_port(), miner_key=key,
             log=lambda m: logs.append(m))
    logs = []
    real = n._process_batches
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return real()
    monkeypatch.setattr(n, "_process_batches", flaky)
    n.start(mine=True)
    try:
        end = time.time() + 30
        while n.chain.height < 5:
            assert time.time() < end, "mining stopped after the error"
            time.sleep(0.1)
        assert any("restarting" in m and "boom" in m for m in logs)
    finally:
        n.stop()
