"""Green as tools for an AI agent: a Model Context Protocol (MCP) server.

    grn --data agent mcp                  # an MCP client starts this and talks over stdin/stdout

An agent that speaks MCP (Claude, Cursor, a custom one) gets tools to read the network,
its wallet and the compute on offer, buy matrix products from the network's miners
(every product checked with Freivalds' test before it is paid for) and send GRN. The
wallet stays in the --data folder on this computer; an encrypted wallet takes its
password from GREEN_WALLET_PASSWORD_FILE or GREEN_WALLET_PASSWORD. --max-spend caps
what the agent can spend while the server runs, so an owner can hand it a budget.

The protocol is JSON-RPC 2.0, one message per line on stdin/stdout; anything else this
process writes goes to stderr.
"""

import contextlib
import json
import sys
import time

import numpy as np

from . import mainnet, params
from .crypto import Key
from .node import http_get, http_post
from .p2p import AGENT
from .tx import make_tx, min_fee, tx_size

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_SIDE = 2048                     # the largest matrix side the tools accept
RETURN_LIMIT = 4096                 # products with more entries come back as a digest and a corner

_EMPTY = {"type": "object", "properties": {}}
TOOLS = [
    {"name": "green_status",
     "description": "The Green network as this node sees it: height, peers, mining, fee per transfer and how "
                    "much this session may still spend.",
     "inputSchema": _EMPTY},
    {"name": "green_wallet",
     "description": "This agent's address and balance in GRN. Others pay the agent by sending GRN to the address.",
     "inputSchema": _EMPTY},
    {"name": "green_compute_offers",
     "description": "Miners selling compute (int8 matrix multiplication) and their prices in GRN per unit of "
                    "2^24 multiply-adds, cheapest first.",
     "inputSchema": _EMPTY},
    {"name": "green_matmul",
     "description": "Buy the product A @ B of two int8 matrices from the network's miners. Give the matrices "
                    "(entries -128..127), or only m, k and n to buy a product of random matrices of that size. "
                    "The result is checked with Freivalds' test before it is paid for; the answer says what it cost.",
     "inputSchema": {"type": "object", "properties": {
         "a": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}},
               "description": "m x k matrix of int8 values"},
         "b": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}},
               "description": "k x n matrix of int8 values"},
         "m": {"type": "integer"}, "k": {"type": "integer"}, "n": {"type": "integer"},
         "budget_grn": {"type": "number", "description": "GRN to put into payment channels the first time "
                                                         "(default 0.3); what is not spent comes back"}}}},
    {"name": "green_send",
     "description": "Send GRN from this agent's wallet to another address (grn1...).",
     "inputSchema": {"type": "object", "properties": {
         "to": {"type": "string"}, "amount_grn": {"type": "number"}}, "required": ["to", "amount_grn"]}},
]


class ToolError(Exception):
    """Reported to the agent as a failed tool call, not as a protocol error."""


class Server:
    def __init__(self, node: str, key: Key | None, max_spend: int | None = None, allow_local: bool = False):
        self.node, self.key = node.rstrip("/"), key
        self.max_spend, self.spent, self.allow_local = max_spend, 0, allow_local
        self.cluster = None

    # -- the tools --

    def _get(self, path: str):
        try:
            return http_get(self.node + path, timeout=30)
        except OSError as e:
            raise ToolError(f"no Green node answers at {self.node} ({e}); start one with: grn node") from None

    def _need_key(self) -> Key:
        if self.key is None:
            raise ToolError("this server has no wallet; create one with: grn --data <folder> wallet new")
        return self.key

    def _charge(self, amount: int) -> None:
        if self.max_spend is not None and self.spent + amount > self.max_spend:
            left = params.fmt_amount(self.max_spend - self.spent)
            raise ToolError(f"over this session's spending limit ({left} left)")

    def green_status(self) -> dict:
        st, fee = self._get("/status"), self._get("/v1/fee?blocks=2")
        return {"network": "main" if mainnet.is_mainnet(st["genesis"]) else "test " + st["genesis"][:16], "height": st["height"],
                "peers": st["p2p"]["outbound"] + st["p2p"]["inbound"], "mining": st["mining"],
                "transfer_fee_grn": fee["transfer_fee"] / params.YAPRAK_PER_COIN,
                "spend_left_grn": None if self.max_spend is None
                else (self.max_spend - self.spent) / params.YAPRAK_PER_COIN}

    def green_wallet(self) -> dict:
        key = self._need_key()
        acct = self._get(f"/account/{key.address}")
        return {"address": key.address, "balance_grn": acct["balance"] / params.YAPRAK_PER_COIN,
                "with_pending_grn": acct["pending_balance"] / params.YAPRAK_PER_COIN}

    def green_compute_offers(self) -> dict:
        offers = sorted(self._get("/v1/providers"), key=lambda o: o["price"])
        return {"unit": "2^24 multiply-adds",
                "offers": [{"provider": o["provider"], "price_grn_per_unit": o["price"] / params.YAPRAK_PER_COIN,
                            "max_side": o["max_dim"]} for o in offers[:10]]}

    def green_matmul(self, a=None, b=None, m=None, k=None, n=None, budget_grn: float = 0.3) -> dict:
        from .compute import ComputeCluster, ProviderError
        key = self._need_key()
        if a is not None or b is not None:
            try:
                a, b = np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)
            except (TypeError, ValueError):
                raise ToolError("a and b must be matrices of integers") from None
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or 0 in a.shape + b.shape:
                raise ToolError("a must be m x k and b k x n")
            if a.min() < -128 or a.max() > 127 or b.min() < -128 or b.max() > 127:
                raise ToolError("entries must be int8: -128..127")
            a, b = a.astype(np.int8), b.astype(np.int8)
        elif m and k and n:
            rng = np.random.default_rng()
            a = rng.integers(-128, 128, (int(m), int(k)), dtype=np.int8)
            b = rng.integers(-128, 128, (int(k), int(n)), dtype=np.int8)
        else:
            raise ToolError("give the matrices a and b, or the sizes m, k and n")
        if max(a.shape + b.shape) > MAX_SIDE:
            raise ToolError(f"matrix sides are limited to {MAX_SIDE}")
        if self.cluster is None:
            budget = int(budget_grn * params.YAPRAK_PER_COIN)
            if budget < 300_000:
                raise ToolError("budget_grn must be at least 0.003")
            if self.max_spend is not None:
                budget = min(budget, self.max_spend - self.spent)
            try:
                self.cluster = ComputeCluster.from_market(self.node, key, budget // 3, count=3,
                                                          allow_local=self.allow_local)
            except (OSError, ValueError, ProviderError) as e:
                raise ToolError(f"could not open payment channels: {e}") from None
        before, t = self.cluster.owed, time.perf_counter()
        try:
            c = self.cluster.matmul(a, b)
        except Exception as e:                         # a provider failed: nothing was paid for it
            raise ToolError(f"the product failed: {type(e).__name__}: {e}") from None
        cost = self.cluster.owed - before
        self.spent += cost
        out = {"shape": [a.shape[0], a.shape[1], b.shape[1]], "seconds": round(time.perf_counter() - t, 3),
               "cost_grn": cost / params.YAPRAK_PER_COIN, "checked": "Freivalds' test, before paying",
               "providers": len(self.cluster.channels)}
        if c.size <= RETURN_LIMIT:
            out["product"] = c.tolist()
        else:
            import hashlib
            out["product_sha256"] = hashlib.sha256(c.astype("<i8").tobytes()).hexdigest()
            out["product_corner"] = c[:4, :4].tolist()
        return out

    def green_send(self, to: str, amount_grn: float) -> dict:
        from .crypto import is_address
        key = self._need_key()
        if not is_address(to):
            raise ToolError("not a Green address")
        amount = int(round(amount_grn * params.YAPRAK_PER_COIN))
        if amount <= 0:
            raise ToolError("the amount must be positive")
        chain_id = self._get("/status")["genesis"]
        nonce = self._get(f"/account/{key.address}")["pending_nonce"]
        fields = {"to": to, "amount": amount}
        rate = self._get("/v1/fee?blocks=2")["fee_rate"]
        fee = min_fee({"type": "transfer", **fields})
        fee = max(fee, int(rate * tx_size(make_tx(key, chain_id, "transfer", nonce, fee, **fields))) + 1 if rate else 0)
        self._charge(amount + fee)
        try:
            txid = http_post(f"{self.node}/tx", make_tx(key, chain_id, "transfer", nonce, fee, **fields),
                             timeout=30)["txid"]
        except OSError as e:
            body = e.read().decode(errors="replace") if hasattr(e, "read") else str(e)
            raise ToolError(f"the node refused it: {body}") from None
        self.spent += amount + fee
        return {"txid": txid, "amount_grn": amount / params.YAPRAK_PER_COIN, "fee_grn": fee / params.YAPRAK_PER_COIN}

    def close(self) -> None:
        if self.cluster is not None:
            with contextlib.suppress(Exception):
                self.cluster.close()
            self.cluster = None

    # -- the protocol --

    def handle(self, msg: dict) -> dict | None:
        """One JSON-RPC message in, its response out (None for a notification)."""
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:
            return None                                # notifications/initialized and the like
        params_ = msg.get("params") or {}
        if method == "initialize":
            asked = params_.get("protocolVersion")
            result = {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "green", "version": AGENT.split("/")[1]},
                      "instructions": "Green (GRN): buy verified GPU compute from an open network of miners and "
                                      "hold or send GRN. Start with green_wallet and green_compute_offers."}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name, args = params_.get("name"), params_.get("arguments") or {}
            if name not in {t["name"] for t in TOOLS}:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool: {name}"}}
            try:
                with contextlib.redirect_stdout(sys.stderr):   # stdout carries only the protocol
                    out = getattr(self, name)(**args)
                result = {"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out}
            except TypeError as e:
                result = {"content": [{"type": "text", "text": f"bad arguments: {e}"}], "isError": True}
            except ToolError as e:
                result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, inp=None, out=None) -> None:
        inp, out = inp or sys.stdin, out or sys.stdout
        try:
            for line in inp:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
                else:
                    reply = self.handle(msg) if isinstance(msg, dict) else \
                        {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
                if reply is not None:
                    out.write(json.dumps(reply) + "\n")
                    out.flush()
        finally:
            self.close()
