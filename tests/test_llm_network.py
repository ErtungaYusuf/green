"""Language-model layers served by miners' nodes on a network: found through the signed
offers, paid through payment channels, the vouchers claimed in blocks."""

import contextlib
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


@contextlib.contextmanager
def serving(tmp_path, monkeypatch, model, tiny_dir, specs):  # noqa: F811
    """A network of miners each serving layers (`specs`: [(layers, price, cheat)]), and a
    buyer's node, carol's, holding the genesis coins."""
    monkeypatch.setattr(compute, "OFFER_RETRY", 0.5)
    carol = Key.generate()
    genesis = make_genesis(carol.address, "llm")
    quiet = dict(log=lambda m: None)
    nodes, keys, providers, stages = [], [], [], []
    for n, (layers, price, cheat) in enumerate(specs):
        key = Key.generate()
        node = Node(str(tmp_path / f"m{n}"), genesis, port=free_port(), miner_key=key,
                    peers=[nodes[0].p2p.address] if nodes else [], **quiet)
        node.start(mine=True)
        nodes.append(node)
        keys.append(key)
    nc = Node(str(tmp_path / "carol"), genesis, port=free_port(), miner_key=carol, peers=[nodes[0].p2p.address],
              **quiet)
    nc.start()
    mid = model_id(tiny_dir)
    try:
        for node, key, (layers, price, cheat) in zip(nodes, keys, specs):
            p = Provider(node, key, url="")
            p.start()
            providers.append(p)
            st = Stage(model, layers, key, provider=p, price=price, cheat=cheat)
            st.serve()
            st.advertise(st.url, mid)
            stages.append(st)

        def listed():
            offers = [o for o in nc.p2p.offer_list() if o.get("llm", {}).get("model") == mid]
            return {o["provider"] for o in offers} == {k.address for k in keys}
        wait_until(listed, "every stage's offer", timeout=120)
        yield {"nodes": nodes + [nc], "keys": keys + [carol], "stages": stages, "mid": mid}
    finally:
        for st in stages:
            st.stop()
        for p in providers:
            p.stop()
        for n in [nc] + nodes[::-1]:
            n.stop()


@pytest.fixture
def network(tmp_path, monkeypatch, model, tiny_dir):  # noqa: F811
    """Alice and bob mine and each serve half the layers; carol buys."""
    with serving(tmp_path, monkeypatch, model, tiny_dir, [(range(0, 2), 1000, None), (range(2, 4), 2000, None)]) as net:
        yield net


def local_tokens(model, prompt, out):  # noqa: F811
    h, pos = model.embed_tokens(prompt + out[:-1]), np.arange(len(prompt) + len(out) - 1)
    for i in range(model.n_layers):
        h = model.forward(i, h, pos, [])["h_out"]
    return [int(t) for t in np.argmax(model.logits(h)[len(prompt) - 1:], axis=1)]


def test_a_buyer_finds_the_stages_pays_them_and_they_claim_it(network, model):  # noqa: F811
    na, nb, nc = network["nodes"]
    alice, bob, carol = network["keys"]
    pipe = Pipeline.from_market(nc.url, carol, model, network["mid"], amount_each=10_000_000)
    assert [st["provider"] for st in pipe.stages] == [alice.address, bob.address]
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    out, st = pipe.generate(prompt, max_new=10, draft=0)
    assert local_tokens(model, prompt, out) == out and st["replaced"] == 0
    tokens = len(prompt) + 9                                  # every token that went through the layers
    assert [s["owed"] for s in pipe.stages] == [1000 * 2 * tokens, 2000 * 2 * tokens]
    cids = [s["channel"].cid for s in pipe.stages]
    paid = pipe.close()
    assert paid == {alice.address: 1000 * 2 * tokens, bob.address: 2000 * 2 * tokens}
    # The providers claim their vouchers in a block, closing the channels. If a provider's
    # own periodic claim already took the whole voucher, the close has nothing left to
    # claim (a claim must raise the total) and the channel just runs to its expiry.
    def claimed(cid, total):
        ch = na.chain.state.channels.get(cid)
        return ch is None or ch["paid"] == total
    wait_until(lambda: all(claimed(c, t) for c, t in zip(cids, paid.values())), "the claims", timeout=120)


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


def test_a_cheating_stage_is_swapped_for_a_spare_and_the_text_made_again(tmp_path, monkeypatch, model,  # noqa: F811
                                                                      tiny_dir):  # noqa: F811
    """Bob and dave both serve layers 2-4; bob is closer, so he is chosen, and cheats."""
    specs = [(range(0, 2), 1000, None), (range(2, 4), 1000, "skip"), (range(2, 4), 1000, None)]
    with serving(tmp_path, monkeypatch, model, tiny_dir, specs) as net:
        alice, bob, dave, carol = net["keys"]
        bob_url = net["stages"][1].url
        monkeypatch.setattr("green.llm.pipeline.ping", lambda url, tries=3: 0.001 if url == bob_url else 0.002)
        pipe = Pipeline.from_market(net["nodes"][-1].url, carol, model, net["mid"], amount_each=10_000_000)
        assert [s["provider"] for s in pipe.stages] == [alice.address, bob.address]
        assert [s["provider"] for s in pipe.stages[1]["spares"]] == [dave.address]
        prompt = [2, 7, 1, 8, 2, 8]
        out, st = pipe.generate(prompt, max_new=8, draft=0)
        assert st["replaced"] == 1 and [s["provider"] for s in pipe.stages] == [alice.address, dave.address]
        assert local_tokens(model, prompt, out) == out                  # the model's own text, made again
        paid = pipe.close()
        assert paid[dave.address] > 0


class ToyTokenizer:
    """Stands in for a real tokenizer over the tiny model's 300 tokens."""

    class Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        return self.Enc([b % 290 for b in text.encode()][-40:])

    def decode(self, ids):
        return " ".join(map(str, ids))

    def token_to_id(self, text):
        return 299


def test_an_agent_asks_the_networks_model_through_mcp(network, model):  # noqa: F811
    from green.llm import api
    from green.mcp import Server
    carol = network["keys"][-1]
    pipe = Pipeline.from_market(network["nodes"][-1].url, carol, model, network["mid"], amount_each=10_000_000)

    class Backend:                                            # what NetworkModel does, with the toy tokenizer
        def complete(self, req):
            return api.complete(pipe, ToyTokenizer(), 299, "tiny", 2, req)

        def close(self):
            return pipe.close()
    s = Server(network["nodes"][-1].url, carol, max_spend=1_000_000, llm=Backend())
    try:
        assert "green_chat" in {t["name"] for t in s.tools}
        r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "green_chat", "arguments": {"prompt": "hello", "max_tokens": 5}}})["result"]
        out = r["structuredContent"]
        assert out["tokens"] == 5 and out["miners"] == 2 and out["cost_grn"] > 0
        assert s.spent == int(round(out["cost_grn"] * 1e8))
    finally:
        s.close()
    assert Server(network["nodes"][-1].url, carol).tools[-1]["name"] != "green_chat"   # no model, no tool
