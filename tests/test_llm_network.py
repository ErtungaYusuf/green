"""Language-model layers served by miners' nodes on a network: found through the signed
offers, paid through payment channels, the vouchers claimed in blocks."""

import time

import numpy as np
import pytest

from green import compute
from green.chain import make_genesis
from green.compute import Provider
from green.crypto import Key
from green.llm.model import model_id
from green.llm.pipeline import Pipeline, Stage, StageRefused, cheapest_route
from green.node import Node
from green.p2p import offer_problem
from test_app import free_port, wait_until
from test_llm import model, tiny_dir  # noqa: F401  (fixtures)


@pytest.fixture
def network(tmp_path, monkeypatch, model, tiny_dir):  # noqa: F811
    """Alice and bob mine and each serve half the layers; carol buys."""
    monkeypatch.setattr(compute, "OFFER_RETRY", 0.5)
    carol, alice, bob = Key.generate(), Key.generate(), Key.generate()
    genesis = make_genesis(carol.address, "llm")
    quiet = dict(log=lambda m: None)
    na = Node(str(tmp_path / "a"), genesis, port=free_port(), miner_key=alice, **quiet)
    na.start(mine=True)
    nb = Node(str(tmp_path / "b"), genesis, port=free_port(), miner_key=bob, peers=[na.p2p.address], **quiet)
    nb.start(mine=True)
    nc = Node(str(tmp_path / "c"), genesis, port=free_port(), miner_key=carol, peers=[na.p2p.address], **quiet)
    nc.start()
    mid = model_id(tiny_dir)
    providers, stages = [], []
    for node, key, layers, price in ((na, alice, range(0, 2), 1000), (nb, bob, range(2, 4), 2000)):
        p = Provider(node, key, url="")
        p.start()
        st = Stage(model, layers, key, provider=p, price=price)
        st.serve()
        st.advertise(st.url, mid)
        providers.append(p)
        stages.append(st)
    try:
        def listed():
            offers = [o for o in nc.p2p.offer_list() if o.get("llm", {}).get("model") == mid]
            return {o["provider"] for o in offers} == {alice.address, bob.address}
        wait_until(listed, "both stages' offers", timeout=120)
        yield {"nodes": (na, nb, nc), "keys": (alice, bob, carol), "stages": stages, "mid": mid}
    finally:
        for st in stages:
            st.stop()
        for p in providers:
            p.stop()
        for n in (nc, nb, na):
            n.stop()


def test_a_buyer_finds_the_stages_pays_them_and_they_claim_it(network, model):  # noqa: F811
    na, nb, nc = network["nodes"]
    alice, bob, carol = network["keys"]
    pipe = Pipeline.from_market(nc.url, carol, model, network["mid"], amount_each=10_000_000)
    assert [st["provider"] for st in pipe.stages] == [alice.address, bob.address]
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    out, st = pipe.generate(prompt, max_new=10, draft=0)
    pipe.finish_audits()
    h, pos = model.embed_tokens(prompt + out[:-1]), np.arange(len(prompt) + 9)
    for i in range(4):
        h = model.forward(i, h, pos, [])["h_out"]
    assert [int(t) for t in np.argmax(model.logits(h)[len(prompt) - 1:], axis=1)] == out
    tokens = len(prompt) + 9                                  # every token that went through the layers
    assert [s["owed"] for s in pipe.stages] == [1000 * 2 * tokens, 2000 * 2 * tokens]
    cids = [s["channel"].cid for s in pipe.stages]
    paid = pipe.close()
    assert paid == {alice.address: 1000 * 2 * tokens, bob.address: 2000 * 2 * tokens}
    # the providers claim their vouchers and close the channels, in a block
    wait_until(lambda: all(c not in na.chain.state.channels for c in cids), "the closing claims", timeout=120)


def test_an_unpaid_or_forged_pass_is_refused(network, model):  # noqa: F811
    na, nb, nc = network["nodes"]
    alice, bob, carol = network["keys"]
    stages = [s.info for s in network["stages"]]
    free = Pipeline(model, stages)                            # no key: no payment at all
    with pytest.raises(StageRefused, match="open a channel"):
        free.generate([1, 2, 3], max_new=2, draft=0)
    paid = Pipeline.from_market(nc.url, carol, model, network["mid"], amount_each=10_000_000)
    mallory = Key.generate()                                   # signs passes on carol's channels
    paid.key = mallory
    with pytest.raises(StageRefused, match="not signed by the channel's owner"):
        paid.generate([1, 2, 3], max_new=2, draft=0)


def test_offers_with_layers_are_checked_and_routes_are_the_cheapest():
    k = Key.generate()
    base = {"provider": k.address, "pub": k.pub_hex, "url": "http://10.0.0.1:1", "price": 1, "max_dim": 64,
            "credit": 4, "time": int(time.time())}
    from green.p2p import offer_bytes
    for llm, ok in (({"url": "http://10.0.0.1:2", "model": "ab" * 32, "layers": [0, 12], "price": 5}, True),
                    ({"url": "http://example.com:2", "model": "ab" * 32, "layers": [0, 12], "price": 5}, False),
                    ({"url": "http://10.0.0.1:2", "model": "ab" * 32, "layers": [12, 12], "price": 5}, False),
                    ({"url": "http://10.0.0.1:2", "model": "xyz", "layers": [0, 12], "price": 5}, False),
                    ({"url": "http://10.0.0.1:2", "model": "ab" * 32, "layers": [0, 12], "price": 0}, False)):
        o = dict(base, llm=llm)
        o["sig"] = k.sign(offer_bytes("c" * 64, o))
        assert (offer_problem("c" * 64, o, time.time()) is None) == ok, llm

    def offer(name, a, b, price):
        return {"provider": name, "llm": {"layers": [a, b], "price": price}}
    offers = [offer("whole", 0, 8, 10), offer("x", 0, 4, 4), offer("y", 4, 8, 4), offer("z", 4, 8, 9)]
    assert [o["provider"] for o in cheapest_route(offers, 8)] == ["x", "y"]          # 32 + 32 < 80
    assert [o["provider"] for o in cheapest_route(offers[:1] + [offer("x", 0, 4, 20), offers[2]], 8)] == ["whole"]
    assert [o["provider"] for o in cheapest_route(offers[:1] + [offer("x", 0, 4, 16), offers[2]], 8)] == ["whole"]  # a tie: fewer hops
    assert cheapest_route([offer("x", 0, 4, 1)], 8) is None
