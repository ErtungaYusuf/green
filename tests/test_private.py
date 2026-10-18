"""Private compute: sealed requests carried by a peer (green/sealed.py, p2p.py) and
data hidden from the providers (green/private.py)."""

import os
import socket
import time

import numpy as np
import pytest

from green import compute, p2p, params, sealed
from green.chain import make_genesis
from green.compute import ComputeChannel, ComputeCluster, Provider, verify_product_mod
from green.crypto import Key
from green.node import Node
from green.private import PRIMES, PrivateCompute
from green.tx import make_tx


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(cond, what, timeout=60):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, f"timed out waiting for {what}"
        time.sleep(0.1)


def test_a_sealed_request_opens_only_for_its_provider_and_unaltered():
    provider, other = sealed.provider_key(os.urandom(32).hex()), sealed.provider_key(os.urandom(32).hex())
    req, key = sealed.seal_request(sealed.public_hex(provider), {"path": "/matmul", "headers": {"X-A": "1"}}, b"body")
    header, body, pkey = sealed.open_request(provider, req)
    assert header == {"path": "/matmul", "headers": {"X-A": "1"}} and body == b"body"
    assert sealed.open_answer(key, sealed.seal_answer(pkey, {"status": 200}, b"ok")) == ({"status": 200}, b"ok")
    with pytest.raises(sealed.SealError):
        sealed.open_request(other, req)                            # not for it
    tampered = req[:-1] + bytes([req[-1] ^ 1])
    with pytest.raises(sealed.SealError):
        sealed.open_request(provider, tampered)                    # altered on the way
    with pytest.raises(sealed.SealError):
        sealed.open_answer(key, sealed.seal_answer(pkey, {}, b"x")[:-1] + b"\x00")
    req2, _ = sealed.seal_request(sealed.public_hex(provider), {"path": "/matmul", "headers": {"X-A": "1"}}, b"body")
    assert req2[:32] != req[:32]                                   # a fresh key for every request


@pytest.fixture
def market(tmp_path, monkeypatch):
    """A mining node with three providers (three keys), and a buyer with coins."""
    monkeypatch.setattr(compute, "OFFER_RETRY", 0.5)
    buyer, miner = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(buyer.address, "private"), port=free_port(), miner_key=miner,
                log=lambda m: None)
    node.start(mine=True)
    providers = []
    for _ in range(3):
        p = Provider(node, Key.generate())
        p.start()
        providers.append(p)
    yield node, buyer, providers
    for p in providers:
        p.stop()
    node.stop()


def cluster_of(node, buyer, providers) -> ComputeCluster:
    channels = [ComputeChannel(f"http://127.0.0.1:{p.port}", buyer, node.url) for p in providers]
    for ch in channels:
        ch.open(2_000_000, wait=0)
    for ch in channels:
        wait_until(lambda: ch.cid in node.chain.state.channels, "the channel")
    return ComputeCluster(channels)


def test_modular_products_are_exact_and_checked(market):
    node, buyer, providers = market
    p = PRIMES[0]
    rng = np.random.default_rng(3)
    a = rng.integers(0, p, (37, 4096 // 2))
    b = rng.integers(0, p, (4096 // 2, 29))
    body = a.astype("<i4").tobytes() + b.astype("<i4").tobytes()
    c = np.frombuffer(providers[0].compute_mod((37, 2048, 29), p, body), "<i4").astype(np.int64)
    exact = np.array([[sum(int(x) * int(y) for x, y in zip(a[i], b[:, j])) % p for j in range(29)] for i in range(37)])
    assert np.array_equal(c.reshape(37, 29), exact)
    assert verify_product_mod(a, b, exact, p)
    wrong = exact.copy()
    wrong[5, 7] = (wrong[5, 7] + 1) % p
    assert not verify_product_mod(a, b, wrong, p)
    with pytest.raises(ValueError):                               # entries outside [0, p)
        providers[0].compute_mod((1, 1, 1), p, np.array([p, 1], "<i4").tobytes())


def test_split_and_secret_give_the_exact_product_and_shares_look_random(market):
    node, buyer, providers = market
    cluster = cluster_of(node, buyer, providers)
    seen = []
    for ch in cluster.channels:                                   # record what each provider is sent
        real = ch.matmul_mod

        def spy(x, y, p, real=real, ch=ch):
            seen.append((ch.offer["provider"], p, x.copy(), y.copy()))
            return real(x, y, p)
        ch.matmul_mod = spy
    rng = np.random.default_rng(9)
    a = rng.integers(-128, 128, (70, 150), dtype=np.int8)
    b = rng.integers(-128, 128, (150, 45), dtype=np.int8)
    exact = a.astype(np.int64) @ b.astype(np.int64)
    pc = PrivateCompute(cluster)
    assert np.array_equal(pc.matmul(a, b, "split"), exact)
    assert np.array_equal(pc.matmul(a, b, "secret"), exact)
    a[0, 0], b[0, 0] = -128, -128                                 # extremes too
    assert np.array_equal(pc.matmul(a, b, "secret"), a.astype(np.int64) @ b.astype(np.int64))
    assert len({who for who, *_ in seen}) == 3                    # three different providers
    for who, p, x, y in seen:
        left = x[:, :150]
        assert not np.array_equal(left, a.astype(np.int64) % p)   # never the data itself
        assert abs(float(left.mean()) / p - 0.5) < 0.05           # uniform-looking residues
    with pytest.raises(ValueError):
        pc.matmul(a, b, "nope")


def test_requests_travel_sealed_through_a_peer(tmp_path, monkeypatch):
    """The buyer's node never contacts the provider: a peer carries the sealed request
    to the provider on offer, and only to a provider on offer."""
    monkeypatch.setattr(compute, "OFFER_RETRY", 0.5)
    buyer, miner = Key.generate(), Key.generate()
    genesis = make_genesis(buyer.address, "relay")
    seller = Node(str(tmp_path / "s"), genesis, port=free_port(), miner_key=miner, log=lambda m: None)
    seller.start(mine=True)
    mine = Node(str(tmp_path / "b"), genesis, port=free_port(), log=lambda m: None, peers=[seller.p2p.address])
    mine.start()
    p = Provider(seller, miner, url="")
    p.start()
    try:
        wait_until(lambda: [o.get("kx") for o in mine.p2p.offer_list()] == [sealed.public_hex(p.kx)], "the offer")
        cluster = ComputeCluster.from_market(mine.url, buyer, 2_000_000, count=1, allow_local=True, relay=True)
        assert cluster.channels[0].relay
        rng = np.random.default_rng(1)
        a = rng.integers(-128, 128, (40, 64), dtype=np.int8)
        b = rng.integers(-128, 128, (64, 30), dtype=np.int8)
        assert np.array_equal(PrivateCompute(cluster).matmul(a, b, "split"), a.astype(np.int64) @ b.astype(np.int64))
        assert seller.p2p.stats.get("relayed", 0) >= 1 and p.stats["products"] >= 1
        ch = cluster.channels[0]
        assert ch.settle() == ch.owed > 0                         # the voucher travels sealed too
        wait_until(lambda: p.books(ch.cid)["vouched"] == ch.owed, "the provider to hold the voucher")
        with pytest.raises(OSError, match="no such provider"):
            mine.p2p.relay(Key.generate().address, b"\x00" * 64)
        monkeypatch.setattr(p2p, "RELAY_PER_PEER", 0)            # every carrier busy: wait, then give up
        started = time.time()
        with pytest.raises(OSError, match="every peer is busy"):
            mine.p2p.relay(miner.address, b"\x00" * 64, timeout=1.5)
        assert time.time() - started >= 1.5
        monkeypatch.setattr(p2p, "RELAY_PER_PEER", 4)
        assert np.array_equal(PrivateCompute(cluster).matmul(a, b, "split"), a.astype(np.int64) @ b.astype(np.int64))

        def down(*args, **kw):                                    # our node cannot carry it
            raise OSError("relay failed: no peer can carry the request")
        monkeypatch.setattr(compute, "relay_post", down)
        with pytest.raises(compute.RelayError):
            cluster.matmul(a, b)
        assert not ch.bad                                          # not the provider's fault
    finally:
        p.stop()
        mine.stop()
        seller.stop()
