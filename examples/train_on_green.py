"""Train a neural network by paying miners, as it happens.

A throwaway network on this machine: two miners whose nodes also sell compute
(they advertise signed offers over the P2P network), and a buyer with coins
and a node of its own. The buyer finds the miners through its node, opens a
compute channel to each (one transaction each), and trains a digit classifier:
every matrix product of every forward and backward pass is computed by the
miners, checked by the buyer with Freivalds' test, and paid with a voucher. At
the end the miners claim what they earned on chain.

    python examples/train_on_green.py [steps]

Everything here is real protocol; only the network is small and local, so
the miners' products cross a loopback link instead of the internet.
"""

import os
import shutil
import socket
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green import params                                          # noqa: E402
from green.chain import make_genesis                              # noqa: E402
from green.compute import ComputeCluster, Provider                # noqa: E402
from green.crypto import Key                                      # noqa: E402
from green.node import Node                                       # noqa: E402
from green.train import MLP, digits, train                        # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    params.INITIAL_TARGET = params.MAX_TARGET // 50          # quick blocks for a demo
    params.TARGET_BLOCK_SECONDS = 2
    tmp = tempfile.mkdtemp()
    buyer = Key.generate()
    genesis = make_genesis(buyer.address, "a model trained by miners")   # the buyer's 1.25 GRN
    nodes, providers = [], []
    try:
        for i in range(2):                                    # two miners that sell compute
            key = Key.generate()
            n = Node(os.path.join(tmp, f"miner{i}"), genesis, port=free_port(), miner_key=key, log=lambda m: None,
                     peers=[nodes[0].p2p.address] if nodes else [])
            n.start(mine=True)
            p = Provider(n, key, price=2_000 + 500 * i, url="")
            p.start()
            nodes.append(n)
            providers.append(p)
        mine = Node(os.path.join(tmp, "buyer"), genesis, port=free_port(), log=lambda m: None,
                    peers=[n.p2p.address for n in nodes])
        mine.start()
        nodes.append(mine)
        log("Two miners are mining and selling matrix products; waiting for their offers to reach the buyer...")
        while len(mine.p2p.offer_list()) < 2:
            time.sleep(0.5)
        for o in sorted(mine.p2p.offer_list(), key=lambda o: o["price"]):
            log(f"  offer: {o['provider'][:20]}... {o['price']} yaprak per unit at {o['url']}")

        cluster = ComputeCluster.from_market(mine.url, buyer, amount_each=params.YAPRAK_PER_COIN // 10, count=2,
                                             allow_local=True)
        log(f"Opened {len(cluster.channels)} compute channels (0.1 GRN each, one transaction each, confirmed)")

        x, y = digits(8000, seed=1)
        xt, yt = digits(2000, seed=2)
        model = MLP([256, 128, 10], seed=3)
        started = time.time()

        def report(step, loss):
            log(f"  step {step:4d}  loss {loss:.3f}  test accuracy {model.accuracy(xt, yt):.3f}  "
                f"({(time.time() - started) / step * 1000:.0f} ms per step, paid so far "
                f"{params.fmt_amount(cluster.owed)})")
        log(f"Training a 256-128-10 network on 16x16 digits for {steps} steps; every product is done by the miners:")
        train(model, x, y, steps, matmul=cluster.matmul, lr=0.1, seed=4, report=report)
        elapsed = time.time() - started
        reference = MLP([256, 128, 10], seed=3)
        train(reference, x, y, steps, lr=0.1, seed=4)                  # float32 on this machine
        log(f"Done in {elapsed:.1f} s: {model.products} products, all checked. Test accuracy "
            f"{model.accuracy(xt, yt):.3f} (the same network trained in float32 here: "
            f"{reference.accuracy(xt, yt):.3f})")

        owed = {c.offer["provider"]: c.owed for c in cluster.channels}
        cluster.settle()
        for p in providers:
            p.claim_due(force=True)
        log("Vouchers sent; the miners claim their earnings on chain (they also claimed as they went)...")
        end = time.time() + 60
        while time.time() < end:                  # settled: a claim for what is left would not pay its fee
            with mine.lock:
                paid = {c.offer["provider"]: mine.chain.state.channels[c.cid]["paid"] for c in cluster.channels}
                pending = any(t["type"] == "channel_claim" for n in nodes for t in n.mempool.values())
            if not pending and all(owed[a] - paid[a] < params.MIN_TX_FEE for a in owed):
                break
            time.sleep(0.5)
        for c in cluster.channels:
            a = c.offer["provider"]
            log(f"  {a[:20]}...: {c.stats['products']} products, {params.fmt_amount(paid[a])} claimed on chain "
                f"(claim fees come out of it), {owed[a] - paid[a]} yaprak left unclaimed")
        log(f"The buyer paid {params.fmt_amount(sum(paid.values()))} for {model.products} products; the rest of each "
            f"channel returns to it when the channel expires.")
    finally:
        for p in providers:
            p.stop()
        for n in nodes:
            n.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
