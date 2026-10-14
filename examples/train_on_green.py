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
    genesis = make_genesis(buyer.address, "a model trained by miners")   # the buyer's ~0.25 GRN
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
        with mine.lock:
            before = mine.chain.state.balance(buyer.address)
        closing = [c for c in cluster.channels if c.close()]   # final voucher; the miner claims and closes
        log(f"Training done: closing the channels ({len(closing)} of {len(cluster.channels)} agreed; a miner "
            f"declines only when what is left to claim would not pay a claim's fee)...")
        end = time.time() + 60
        while time.time() < end:
            with mine.lock:
                open_ = [c for c in closing if c.cid in mine.chain.state.channels]
            if not open_:
                break
            time.sleep(0.5)
        with mine.lock:
            back = mine.chain.state.balance(buyer.address) - before
        for c in cluster.channels:
            log(f"  {c.offer['provider'][:20]}...: {c.stats['products']} products, paid "
                f"{params.fmt_amount(owed[c.offer['provider']])} (claimed on chain by the miner)")
        log(f"The buyer paid {params.fmt_amount(sum(owed.values()))} for {model.products} products; the unspent "
            f"{params.fmt_amount(back)} of the closed channels came back at once"
            + (" (the others return theirs at expiry)." if len(closing) < len(cluster.channels) else "."))
    finally:
        for p in providers:
            p.stop()
        for n in nodes:
            n.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
