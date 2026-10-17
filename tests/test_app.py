"""The app: buying compute, training, the compute proxy, mining and sending from one
place, and the guards that keep web pages open in a browser on this computer out."""

import json
import socket
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from green import app as appmod
from green import compute, params
from green.chain import make_genesis
from green.compute import Provider
from green.crypto import Key
from green.node import Node


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(cond, what, timeout=60):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, f"timed out waiting for {what}"
        time.sleep(0.1)


@pytest.fixture
def market(tmp_path, monkeypatch):
    """A mining node selling compute (alice), the buyer's own node (carol), and the app."""
    monkeypatch.setattr(compute, "OFFER_RETRY", 0.5)
    carol, alice = Key.generate(), Key.generate()
    genesis = make_genesis(carol.address, "app")
    seller = Node(str(tmp_path / "s"), genesis, port=free_port(), miner_key=alice, log=lambda m: None)
    seller.start(mine=True)
    mine = Node(str(tmp_path / "b"), genesis, port=free_port(), miner_key=carol, log=lambda m: None,
                peers=[seller.p2p.address])
    mine.start()
    p = Provider(seller, alice, url="")
    p.start()
    app = appmod.App([appmod.Account("carol", "buyer", carol, mine.url),
                      appmod.Account("alice", "miner", alice, seller.url)], buyer="carol", port=0, log=lambda m: None)
    app.start()
    try:
        wait_until(lambda: [o["provider"] for o in mine.p2p.offer_list()] == [alice.address], "alice's offer")
        yield app, seller, mine
    finally:
        app.stop()
        p.stop()
        mine.stop()
        seller.stop()


def call(app, path, body=None, token=True, headers=None, raw=None) -> bytes:
    h = {"Content-Type": "application/json"}
    if token:
        h["X-App-Token"] = app.token
    h.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(app.url + path, data=data, headers=h, method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def state(app) -> dict:
    return json.loads(call(app, "/api/state", token=False))


def test_the_app_buys_checked_compute_and_trains_on_the_network(market):
    app, seller, mine = market
    assert app.token.encode() in call(app, "/", token=False)                # the page carries its token
    r = json.loads(call(app, "/api/product", {"account": "carol", "m": 300, "k": 200, "n": 100}))
    assert r["checked"] and r["matches_local"] and r["cost"] > 0 and r["shape"] == [300, 200, 100]
    assert state(app)["accounts"][0]["buying"]["products"] >= 1
    jid = json.loads(call(app, "/api/train", {"account": "carol", "steps": 20}))["job"]
    wait_until(lambda: state(app)["jobs"][-1]["state"] != "running", "the training job", timeout=120)
    job = state(app)["jobs"][-1]
    assert job["id"] == jid and job["state"] == "done", job["lines"]
    assert job["result"]["products"] == 20 * 5 and job["result"]["paid"] > 0
    assert json.loads(call(app, "/api/close", {"account": "carol"}))["closed"] == 1
    assert state(app)["accounts"][0]["buying"] is None


def test_the_proxy_answers_programs_like_a_provider(market):
    app, seller, mine = market
    rng = np.random.default_rng(1)
    a = rng.integers(-128, 128, (64, 300), dtype=np.int8)
    b = rng.integers(-128, 128, (300, 40), dtype=np.int8)
    out = call(app, "/v1/matmul", token=False, headers={"X-Shape": "64,300,40", "Content-Type": "application/octet-stream"},
               raw=a.tobytes() + b.tobytes())
    c = np.frombuffer(out, "<i4").reshape(64, 40)
    assert np.array_equal(c, a.astype(np.int64) @ b.astype(np.int64))
    assert app.proxy["products"] == 1 and app.proxy["macs"] == 64 * 300 * 40
    for shape in ("64,300,41", "64,300,39"):                               # body and shape must agree exactly
        with pytest.raises(urllib.error.HTTPError) as e:
            call(app, "/v1/matmul", token=False, headers={"X-Shape": shape}, raw=a.tobytes() + b.tobytes())
        assert e.value.code == 400
    assert app.proxy["products"] == 1


def test_mining_and_sending_from_the_app(market):
    app, seller, mine = market
    assert not mine.mining
    assert json.loads(call(app, "/api/mining", {"account": "carol", "on": True}))["mining"] and mine.mining
    call(app, "/api/mining", {"account": "carol", "on": False})
    assert not mine.mining
    wait_until(lambda: seller.chain.state.balance(seller.miner_key.address) > params.YAPRAK_PER_COIN, "alice's coins")
    before = mine.chain.state.balance(mine.miner_key.address)
    txid = json.loads(call(app, "/api/send", {"from": "alice", "to": "carol", "amount": "0.5"}))["txid"]
    assert len(txid) == 64
    wait_until(lambda: mine.chain.state.balance(mine.miner_key.address) == before + params.YAPRAK_PER_COIN // 2,
               "the coins to arrive")
    with pytest.raises(urllib.error.HTTPError) as e:
        call(app, "/api/send", {"from": "alice", "to": "nobody", "amount": "1"})
    assert e.value.code == 400


def test_web_pages_in_a_browser_cannot_use_the_app(market):
    """A page from another site can make the browser send simple requests to 127.0.0.1,
    but not read the token or add custom headers (no CORS), and a DNS-rebinding page
    names its own host."""
    app, seller, mine = market

    def refused(path, **kw):
        with pytest.raises(urllib.error.HTTPError) as e:
            call(app, path, **kw)
        return e.value.code
    assert refused("/api/send", body={"from": "alice", "to": "carol", "amount": "1"}, token=False) == 403
    assert refused("/api/mining", body={"account": "carol", "on": True}, headers={"Host": "evil.example:8600"}) == 403
    assert refused("/", token=False, headers={"Host": "evil.example"}) == 403            # the token stays here
    assert refused("/api/state", token=False, headers={"Host": "evil.example"}) == 403
    assert refused("/v1/matmul", token=False, raw=b"\x00" * 8, headers={"Content-Type": "text/plain"}) == 400
    assert refused("/v1/matmul", token=False, raw=b"\x00" * 8,
                   headers={"X-Shape": "2,2,2", "Host": "evil.example"}) == 403
    assert not mine.mining and app.proxy["products"] == 0 and mine.chain.state.nonce(seller.miner_key.address) == 0
