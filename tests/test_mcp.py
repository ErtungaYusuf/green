"""The MCP server: Green as tools for an AI agent, over JSON-RPC on stdin/stdout."""

import io
import json

import numpy as np

from green import params
from green.crypto import Key
from green.mcp import Server
from test_app import free_port, market  # noqa: F401  (the fixture)


def rpc(server, *msgs) -> list:
    out = io.StringIO()
    server.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in msgs) + "not json\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def call(server, name, **args) -> dict:
    return server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})["result"]


def test_the_protocol_without_a_node():
    s = Server(f"http://127.0.0.1:{free_port()}", Key.generate())
    init, tools, status, unknown, parse = rpc(
        s, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},          # a notification: no answer
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "green_status"}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "rm_rf"}})
    assert init["result"]["protocolVersion"] == "2025-03-26" and init["result"]["serverInfo"]["name"] == "green"
    assert {t["name"] for t in tools["result"]["tools"]} == {
        "green_status", "green_wallet", "green_compute_offers", "green_matmul", "green_send"}
    assert status["result"]["isError"] and "no Green node" in status["result"]["content"][0]["text"]
    assert unknown["error"]["code"] == -32602
    assert parse["error"]["code"] == -32700
    assert call(s, "green_status", bogus=1)["isError"]                      # unknown arguments are refused
    assert call(Server(s.node, None), "green_wallet")["isError"]            # no wallet, no wallet tools


def test_an_agent_buys_checked_compute_and_stays_within_its_limit(market):  # noqa: F811
    app, seller, mine = market
    carol = app.accounts["carol"].key
    s = Server(mine.url, carol, max_spend=params.YAPRAK_PER_COIN, allow_local=True)
    try:
        assert call(s, "green_wallet")["structuredContent"]["address"] == carol.address
        offers = call(s, "green_compute_offers")["structuredContent"]["offers"]
        assert [o["provider"] for o in offers] == [seller.miner_key.address]
        a, b = [[1, -2, 3], [127, -128, 0]], [[1, 0], [0, 1], [5, 5]]
        r = call(s, "green_matmul", a=a, b=b, budget_grn=0.05)["structuredContent"]
        assert r["product"] == (np.array(a) @ np.array(b)).tolist() and r["cost_grn"] > 0
        big = call(s, "green_matmul", m=200, k=300, n=100)["structuredContent"]
        assert big["shape"] == [200, 300, 100] and len(big["product_sha256"]) == 64
        assert call(s, "green_matmul", a=[[300]], b=[[1]])["isError"]       # not int8
        over = call(s, "green_send", to=Key.generate().address, amount_grn=2)
        assert over["isError"] and "spending limit" in over["content"][0]["text"]
        sent = call(s, "green_send", to=Key.generate().address, amount_grn=0.1)["structuredContent"]
        assert len(sent["txid"]) == 64
        assert 0 < s.spent < params.YAPRAK_PER_COIN
    finally:
        s.close()
