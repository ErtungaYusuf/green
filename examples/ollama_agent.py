"""An AI agent on the Green network, its decisions made by a local language model.

    python examples/ollama_agent.py --localnet localnet          # try it on a local network
    python examples/ollama_agent.py --node http://127.0.0.1:8545  # on the main network

The model (through Ollama, default qwen3:1.7b) gets a goal and four tools: look at its
wallet, look at the compute on offer, buy a matrix product from the network's miners,
and finish with a report. It decides what to call and when; this program only carries
out the calls. Its wallet lives in its own folder and never leaves this computer; every
product it buys is checked (Freivalds' test) before it is paid for.

With --localnet, one of the local network's miners first sends the agent 2 GRN, as a
customer would pay it for its work.
"""

import argparse
import inspect
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from green import params, tx as txmod  # noqa: E402
from green.cli import create_wallet, wallet_path  # noqa: E402
from green.compute import ComputeCluster  # noqa: E402
from green.crypto import Key  # noqa: E402
from green.node import http_get, http_post, wait_for_block  # noqa: E402

GOAL = ("You are an autonomous AI agent with your own wallet on the Green network, where GPU miners sell "
        "matrix multiplication. Your job: get a 512 x 512 x 512 matrix product computed by the network for "
        "at most 0.5 GRN. Check your wallet first, then look at the offers, then buy the product, then "
        "finish with a short report of what you paid and whether the result was verified. Use the tools; "
        "do not invent results.")

TOOLS = [
    {"type": "function", "function": {"name": "check_wallet", "description": "Your address and balance in GRN.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_offers",
                                      "description": "Miners selling compute and their prices (GRN per 2^24 multiply-adds).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "buy_matrix_product",
        "description": "Buy the product of random int8 matrices of the given size (m x k x n) from the network; "
                       "the result is verified before it is paid for.",
        "parameters": {"type": "object", "properties": {
            "m": {"type": "integer"}, "k": {"type": "integer"}, "n": {"type": "integer"},
            "max_cost_grn": {"type": "number", "description": "the most you will pay"}},
            "required": ["m", "k", "n", "max_cost_grn"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End with a report for your owner.",
                                      "parameters": {"type": "object", "properties": {"report": {"type": "string"}},
                                                     "required": ["report"]}}},
]


class Agent:
    def __init__(self, node: str, key: Key, allow_local: bool, log=print):
        self.node, self.key, self.allow_local, self.log = node.rstrip("/"), key, allow_local, log
        self.cluster = None

    def check_wallet(self) -> dict:
        acct = http_get(f"{self.node}/account/{self.key.address}", timeout=10)
        return {"address": self.key.address, "balance_grn": acct["balance"] / params.YAPRAK_PER_COIN,
                "pending_grn": acct["pending_balance"] / params.YAPRAK_PER_COIN}

    def list_offers(self) -> dict:
        offers = http_get(f"{self.node}/providers", timeout=10)
        return {"offers": [{"miner": o["provider"][:16] + "...", "price_grn": o["price"] / params.YAPRAK_PER_COIN,
                            "largest_side": o["max_dim"]} for o in offers[:5]]}

    def buy_matrix_product(self, m: int, k: int, n: int, max_cost_grn: float) -> dict:
        m, k, n = (max(1, min(int(x), 2048)) for x in (m, k, n))
        if self.cluster is None:
            budget = int(max_cost_grn * params.YAPRAK_PER_COIN)
            self.log(f"   (opening payment channels to the cheapest miners, {max_cost_grn} GRN in all)")
            self.cluster = ComputeCluster.from_market(self.node, self.key, max(budget // 3, 100_000), count=3,
                                                      allow_local=self.allow_local)
        rng = np.random.default_rng()
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        before, t = self.cluster.owed, time.perf_counter()
        c = self.cluster.matmul(a, b)
        seconds = time.perf_counter() - t
        cost = (self.cluster.owed - before) / params.YAPRAK_PER_COIN
        same = bool(np.array_equal(c, a.astype(np.int64) @ b.astype(np.int64)))
        return {"shape": [m, k, n], "seconds": round(seconds, 3), "cost_grn": cost, "verified_by_freivalds": True,
                "matches_a_local_check": same, "providers": len(self.cluster.channels)}

    def close(self):
        if self.cluster is not None:
            self.cluster.close()


def chat(model: str, messages: list, host: str) -> dict:
    body = json.dumps({"model": model, "messages": messages, "tools": TOOLS, "stream": False, "think": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]


def fund_from_localnet(root: str, agent: Key, node: str, amount: int) -> None:
    from green import app as appmod
    accounts, _ = appmod.localnet_accounts(root)
    payer = max((a for a in accounts if a.role.startswith("miner")),
                key=lambda a: http_get(f"{a.node}/account/{a.key.address}", timeout=10)["balance"])
    chain_id = http_get(f"{payer.node}/status", timeout=10)["genesis"]
    nonce = http_get(f"{payer.node}/account/{payer.key.address}", timeout=10)["pending_nonce"]
    t = txmod.make_tx(payer.key, chain_id, "transfer", nonce, to=agent.address, amount=amount)
    http_post(f"{payer.node}/tx", t, timeout=10)
    print(f"{payer.name} sent the agent {params.fmt_amount(amount)}; waiting for the next block...")
    end = time.time() + 180
    while http_get(f"{node}/account/{agent.address}", timeout=10)["balance"] < amount:
        if time.time() > end:
            raise SystemExit("the payment did not arrive")
        tip = http_get(f"{node}/status", timeout=10)["tip"]
        wait_for_block(node, tip, 30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--node", default="http://127.0.0.1:18505", help="a Green node's API")
    ap.add_argument("--localnet", help="a local network's folder: fund the agent from one of its miners")
    ap.add_argument("--data", default="agent", help="the agent's wallet folder")
    ap.add_argument("--model", default="qwen3:1.7b")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    if not os.path.exists(wallet_path(args.data)):
        os.makedirs(args.data, exist_ok=True)
        key = create_wallet(args.data, None)            # the agent's own key, on this computer only
    else:
        with open(wallet_path(args.data), encoding="utf-8") as f:
            w = json.load(f)
        from green import hd
        key = hd.key_from_phrase(w["phrase"], w.get("passphrase", ""), w["path"])
    print(f"Agent wallet: {key.address}  (model {args.model})")
    if args.localnet:
        if http_get(f"{args.node}/account/{key.address}", timeout=10)["balance"] < params.YAPRAK_PER_COIN:
            fund_from_localnet(args.localnet, key, args.node, 2 * params.YAPRAK_PER_COIN)

    agent = Agent(args.node, key, allow_local=bool(args.localnet))
    messages = [{"role": "system", "content": GOAL}, {"role": "user", "content": "Begin."}]
    try:
        for step in range(args.steps):
            msg = chat(args.model, messages, args.ollama)
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                print(f"agent says: {msg.get('content', '').strip()[:400]}")
                messages.append({"role": "user", "content": "Use a tool for your next step, or call finish."})
                continue
            for call in calls:
                name, fargs = call["function"]["name"], call["function"].get("arguments") or {}
                print(f"agent calls {name}({json.dumps(fargs)})")
                if name == "finish":
                    print(f"\nREPORT: {fargs.get('report', '')}")
                    return
                try:
                    fn = getattr(agent, name)
                    # small models invent arguments; keep only the ones the tool takes
                    known = inspect.signature(fn).parameters
                    result = fn(**{k: v for k, v in fargs.items() if k in known})
                except Exception as e:                    # the model hears what went wrong and decides again
                    result = {"error": f"{type(e).__name__}: {e}"}
                print(f"   -> {json.dumps(result)}")
                messages.append({"role": "tool", "content": json.dumps(result), "tool_name": name})
        print("the agent ran out of steps")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
