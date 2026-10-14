"""Instant compute through channels: a model trained on products bought from
providers, checked result by result, paid by voucher and claimed on chain."""

import json
import socket
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from green import compute, params, tx as txmod
from green.chain import make_genesis
from green.compute import ComputeChannel, ComputeCluster, Provider, WrongResult, price_of, verify_product
from green.crypto import Key
from green.node import Node
from green.train import MLP, digits, local_matmul, train


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(cond, what, timeout=60):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, f"timed out waiting for {what}"
        time.sleep(0.05)


@pytest.fixture
def market(tmp_path):
    """A mining node whose operator sells products, and a buyer with coins."""
    buyer, miner = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(buyer.address, "instant compute"), port=free_port(),
                miner_key=miner, log=lambda m: None)
    node.start(mine=True)
    providers = []

    def provider(key=miner, **kw):
        p = Provider(node, key, **kw)
        p.start()
        providers.append(p)
        return p
    yield node, buyer, provider
    for p in providers:
        p.stop()
    node.stop()


def channel(node, buyer, p, amount=5_000_000):
    ch = ComputeChannel(f"http://127.0.0.1:{p.port}", buyer, node.url)
    ch.open(amount)
    return ch


def test_verify_product_is_exact_for_any_allowed_shape():
    rng = np.random.default_rng(1)
    for m, k, n in ((1, 1, 1), (7, 300, 5), (64, 4096, 200), (300, 128, 2048)):
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        c = local_matmul(a, b)
        assert verify_product(a, b, c)
        bad = c.copy()
        bad[rng.integers(m), rng.integers(n)] += 1
        assert not verify_product(a, b, bad)
        bad[0, 0] = c[0, 0] + params.FREIVALDS_PRIME            # equal mod p, caught by the bound
        assert not verify_product(a, b, bad)
    assert not verify_product(a, b, c[:, :-1])                  # wrong shape
    edge = np.full((64, 4096), -128, dtype=np.int8)             # the largest products there are
    edge[0, 0] = 1                                              # (and np.abs(int8 -128) is -128)
    assert verify_product(edge, edge.T.copy(), local_matmul(edge, edge.T.copy()))


def test_a_model_trains_on_bought_compute_and_the_provider_is_paid(market):
    node, buyer, provider = market
    p = provider()
    ch = channel(node, buyer, p)
    x, y = digits(3000, seed=1)
    xt, yt = digits(1000, seed=2)
    remote, local = MLP([256, 64, 10], seed=3), MLP([256, 64, 10], seed=3)
    losses = train(remote, x, y, 40, matmul=ch.matmul, lr=0.1, seed=4)
    train(local, x, y, 40, matmul=local_matmul, lr=0.1, seed=4)
    # Every remote product is exact, so the model is the one this machine would have trained.
    assert all(np.array_equal(a, b) for a, b in zip(remote.W, local.W))
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) / 2 and remote.accuracy(xt, yt) > 0.5
    assert ch.stats["products"] == remote.products == 5 * 40
    assert ch.owed == remote.products * price_of(128, 256, 64, p.price)    # every product here is one unit
    assert ch.settle() == ch.owed
    p.claim_due(force=True)
    wait_until(lambda: node.chain.state.channels[ch.cid]["paid"] == ch.owed, "the claim on chain")


def test_a_wrong_product_is_caught_unpaid_and_the_cluster_moves_on(market, monkeypatch):
    node, buyer, provider = market
    honest, cheat = provider(), provider(key=Key.generate())
    real = cheat.compute

    def lying(*args, **kw):
        c = bytearray(real(*args, **kw))
        c[0] ^= 1                                               # one bit of one entry
        return bytes(c)
    monkeypatch.setattr(cheat, "compute", lying)
    good, bad = channel(node, buyer, honest), channel(node, buyer, cheat)
    rng = np.random.default_rng(5)
    a = rng.integers(-128, 128, (64, 32), dtype=np.int8)
    b = rng.integers(-128, 128, (32, 16), dtype=np.int8)
    with pytest.raises(WrongResult):
        bad.matmul(a, b)
    assert bad.owed == 0 and bad.bad                            # not paid, not used again
    cluster = ComputeCluster([bad, good], split_macs=1)
    big = rng.integers(-128, 128, (512, 512), dtype=np.int8)
    assert np.array_equal(cluster.matmul(big, big), local_matmul(big, big))
    assert good.owed > 0 and bad.owed == 0


def test_providers_extend_only_limited_credit(market, monkeypatch):
    node, buyer, provider = market
    p = provider(credit=2)
    ch = channel(node, buyer, p)
    url = f"http://127.0.0.1:{p.port}/matmul"
    body = bytes(4) + bytes(4)                                  # a 2x2 by 2x2 product of zeros

    def post(headers):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"X-Channel": ch.cid, "X-Shape": "2,2,2", **headers})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
    assert post({}) == 200 and post({}) == 200
    assert post({}) == 402                                      # two results unpaid: pay first
    price = price_of(2, 2, 2, p.price)
    forged = txmod.sign_voucher(Key.generate(), node.chain.chain_id, ch.cid, 2 * price)
    assert post({"X-Voucher-Total": str(2 * price), "X-Voucher": forged}) == 402
    paid = txmod.sign_voucher(buyer, node.chain.chain_id, ch.cid, 2 * price)
    assert post({"X-Voucher-Total": str(2 * price), "X-Voucher": paid}) == 200
    req = urllib.request.Request(url, data=body, method="POST", headers={"X-Channel": "ab" * 32, "X-Shape": "2,2,2"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 404                                  # no such channel
    monkeypatch.setattr(compute, "CLAIM_MARGIN", 10**6)         # every channel now "expires soon"
    assert post({"X-Voucher-Total": str(2 * price), "X-Voucher": paid}) == 409
    books = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{p.port}/channel/{ch.cid}", timeout=10).read())
    assert books["vouched"] == 2 * price and books["owed"] == 3 * price


def test_a_refusal_reaches_the_client_whose_body_comes_later(market):
    """A provider refuses a product from its headers and never keeps the body. HTTP
    clients send the body after the headers (http.client: a second send), so it can
    arrive after the provider has answered and hung up; a closed socket answers it with
    a reset, which on Windows discards the answer the client has not read yet (seen as
    WinError 10053 in a test). The answer must arrive anyway."""
    node, buyer, provider = market
    p = provider()
    for m, k, n in ((2, 2, 2), (256, 512, 256)):                # 8 bytes and 256 KiB
        body = bytes(m * k + k * n)
        with socket.create_connection(("127.0.0.1", p.port), timeout=10) as s:
            s.sendall(f"POST /matmul HTTP/1.1\r\nHost: x\r\nX-Channel: {'ab' * 32}\r\nX-Shape: {m},{k},{n}\r\n"
                      f"Content-Length: {len(body)}\r\n\r\n".encode())
            time.sleep(0.3)                                     # the provider answers from the headers
            s.sendall(body)
            time.sleep(0.3)                                     # and the client reads late
            answer = s.makefile("rb").read()
        assert answer.startswith(b"HTTP/1.1 404 ") and b'"error"' in answer   # no such channel


def test_a_buyer_finds_providers_on_the_network_and_trains(tmp_path, monkeypatch):
    monkeypatch.setattr(compute, "OFFER_RETRY", 0.5)
    buyer, miner = Key.generate(), Key.generate()
    genesis = make_genesis(buyer.address, "market")
    seller = Node(str(tmp_path / "s"), genesis, port=free_port(), miner_key=miner, log=lambda m: None)
    seller.start(mine=True)
    mine = Node(str(tmp_path / "b"), genesis, port=free_port(), log=lambda m: None, peers=[seller.p2p.address])
    mine.start()
    p = Provider(seller, miner, url="")                         # advertise at the port it gets
    p.start()
    try:
        wait_until(lambda: [o["provider"] for o in mine.p2p.offer_list()] == [miner.address],
                   "the offer to reach the buyer's node")
        cluster = ComputeCluster.from_market(mine.url, buyer, 2_000_000, count=3, allow_local=True)
        assert [c.offer["provider"] for c in cluster.channels] == [miner.address]
        model = MLP([256, 32, 10], seed=1)
        x, y = digits(1000, seed=1)
        losses = train(model, x, y, 15, matmul=cluster.matmul, lr=0.1)
        assert losses[-1] < losses[0] and cluster.owed > 0
        # Without allow_local, offers pointing at local addresses are skipped (they are for tests).
        assert compute._private("127.0.0.1") and compute._private("10.0.0.5") and compute._private("100.64.0.9")
        assert not compute._private("8.8.8.8")
    finally:
        p.stop()
        mine.stop()
        seller.stop()


def test_offers_must_be_signed_and_funded(tmp_path):
    from green import p2p
    from test_p2p import RawPeer
    rich, poor = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(rich.address, "offers"), port=free_port(), log=lambda m: None)
    node.start()
    try:
        def offer(key, **change):
            o = {"provider": key.address, "pub": key.pub_hex, "url": "http://203.0.113.7:8548", "price": 5,
                 "max_dim": 1024, "credit": 4, "time": int(time.time())}
            o["sig"] = key.sign(p2p.offer_bytes(node.chain.chain_id, o))
            o.update(change)
            return o
        assert node.p2p.add_offer(offer(rich))
        assert not node.p2p.add_offer(offer(poor))              # holds no coins: spam would be free
        assert not node.p2p.add_offer(offer(rich, price=1))     # changed after signing
        assert not node.p2p.add_offer(offer(rich, time=int(time.time()) - p2p.OFFER_TTL - 10))
        assert [o["price"] for o in node.p2p.offer_list()] == [5]
        raw = RawPeer(node, features=["offers"])
        assert raw.recv_until("offers")["offers"][0]["provider"] == rich.address   # sent on connect
        for _ in range(5):
            raw.send("offers", {"offers": [offer(rich, sig="00" * 64)]})
        assert raw.closed()                                     # forged offers add up to a ban
    finally:
        node.stop()


def test_requests_are_checked_before_their_body_is_read(market):
    node, buyer, provider = market
    p = provider()
    ch = channel(node, buyer, p)
    for cid, shape, length, code in (("ab" * 32, "4096,4096,4096", 2 * 4096 * 4096, b"404"),   # no such channel
                                     (ch.cid, "4096,4096,4096", 2 * 4096 * 4096 - 1, b"400")):  # wrong length
        s = socket.create_connection(("127.0.0.1", p.port), timeout=5)
        s.sendall(f"POST /matmul HTTP/1.1\r\nHost: x\r\nX-Channel: {cid}\r\nX-Shape: {shape}\r\n"
                  f"Content-Length: {length}\r\n\r\n".encode())
        started = time.time()
        reply = s.recv(4096)                                    # without sending a byte of the 32 MiB
        assert reply.split(b" ")[1] == code and time.time() - started < 3
        s.close()


def test_a_request_that_breaks_off_is_not_billed(market):
    node, buyer, provider = market
    p = provider()
    ch = channel(node, buyer, p)
    s = socket.create_connection(("127.0.0.1", p.port), timeout=5)
    s.sendall(f"POST /matmul HTTP/1.1\r\nHost: x\r\nX-Channel: {ch.cid}\r\nX-Shape: 64,64,64\r\n"
              f"Content-Length: {2 * 64 * 64}\r\n\r\n".encode() + bytes(100))
    s.close()                                                   # gone halfway through the body
    time.sleep(0.5)
    assert p.books(ch.cid)["owed"] == 0
    ticket, owed = p.admit(ch.cid, (2, 2, 2))
    p.admit(ch.cid, (2, 2, 2))
    p.release(ch.cid, ticket)
    assert p.books(ch.cid)["owed"] == price_of(2, 2, 2, p.price)


def test_books_outlive_a_reorg_and_go_after_expiry(market):
    node, buyer, provider = market
    p = provider()
    ch = channel(node, buyer, p)
    ch.matmul(np.ones((4, 4), np.int8), np.ones((4, 4), np.int8))
    ch.settle()
    p._forget_ended()                       # whatever the state says, a live channel's vouchers are kept
    assert p.books(ch.cid)["vouched"] == ch.owed
    with p.lock:
        p.book[ch.cid]["expires"] = node.chain.height - params.MAX_REORG_DEPTH - 1
    p._forget_ended()                       # over for good: beyond any reorg
    with pytest.raises(LookupError):
        p.books(ch.cid)


def test_the_buyer_closes_and_gets_the_rest_back_at_once(market):
    node, buyer, provider = market
    p = provider()
    ch = channel(node, buyer, p, amount=5_000_000)
    for _ in range(12):                                         # 12 units: more than a claim's fee
        ch.matmul(np.ones((8, 8), np.int8), np.ones((8, 8), np.int8))
    before = node.chain.state.balance(buyer.address)
    assert ch.close()
    wait_until(lambda: ch.cid not in node.chain.state.channels, "the channel to close")
    assert node.chain.state.balance(buyer.address) == before + 5_000_000 - ch.owed


def test_independent_products_go_out_together(market, monkeypatch):
    monkeypatch.setattr(compute, "PARALLEL_MIN_SECONDS", 0)     # in parallel even on a fast local link
    node, buyer, provider = market
    p1, p2 = provider(), provider(key=Key.generate())
    cluster = ComputeCluster([channel(node, buyer, p1), channel(node, buyer, p2)])
    rng = np.random.default_rng(9)
    pairs = [(rng.integers(-128, 128, (32, 48), dtype=np.int8), rng.integers(-128, 128, (48, 16), dtype=np.int8))
             for _ in range(6)]
    assert all(np.array_equal(c, local_matmul(a, b)) for c, (a, b) in zip(cluster.matmul_many(pairs), pairs))
    assert p1.stats["products"] == 3 and p2.stats["products"] == 3
    one = cluster.channels[0]
    assert all(np.array_equal(c, local_matmul(a, b)) for c, (a, b) in zip(one.matmul_many(pairs), pairs))


def test_offer_urls_name_an_ip_address():
    from green import p2p
    key = Key.generate()
    for url, ok in (("http://203.0.113.7:8548", True), ("http://[2001:db8::7]:8548", True),
                    ("http://provider.example:8548", False), ("ftp://203.0.113.7:21", False)):
        o = {"provider": key.address, "pub": key.pub_hex, "url": url, "price": 5, "max_dim": 1024, "credit": 4,
             "time": int(time.time())}
        o["sig"] = key.sign(p2p.offer_bytes("cc" * 32, o))
        assert (p2p.offer_problem("cc" * 32, o, time.time()) is None) == ok, url


def test_a_close_replaces_a_claim_still_waiting(market):
    node, buyer, provider = market
    p = provider()
    ch = channel(node, buyer, p, amount=5_000_000)
    node.mining = False                                         # so the first claim waits in the mempool
    for _ in range(12):
        ch.matmul(np.ones((8, 8), np.int8), np.ones((8, 8), np.int8))
    ch.settle()
    assert p.claim_due(force=True) == [ch.cid]
    for _ in range(12):
        ch.matmul(np.ones((8, 8), np.int8), np.ones((8, 8), np.int8))
    assert ch.close()                                           # replaces the waiting claim
    (claim,) = [t for t in node.mempool.values() if t["type"] == "channel_claim"]
    assert claim.get("close") and claim["total"] == ch.owed
    before = node.chain.state.balance(buyer.address)
    node.mining = True
    wait_until(lambda: ch.cid not in node.chain.state.channels, "the channel to close")
    assert node.chain.state.balance(buyer.address) == before + 5_000_000 - ch.owed
