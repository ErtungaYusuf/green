"""How fast is compute bought through a channel? A local node mines and sells
products; a buyer opens a channel and times products of several sizes and a
training run, against doing the same on this machine.

    python bench/compute_bench.py
"""

import os
import shutil
import socket
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from green import params  # noqa: E402
from green.chain import make_genesis  # noqa: E402
from green.compute import ComputeChannel, Provider, verify_product  # noqa: E402
from green.crypto import Key  # noqa: E402
from green.node import Node  # noqa: E402
from green.train import MLP, digits, local_matmul, train  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def timed(fn, repeat):
    fn()                                                   # warm up
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t) / repeat


def main():
    params.INITIAL_TARGET = params.MAX_TARGET              # a block whenever the miner tries
    tmp = tempfile.mkdtemp()
    buyer, miner = Key.generate(), Key.generate()
    node = Node(os.path.join(tmp, "n"), make_genesis(buyer.address, "bench"), port=free_port(),
                miner_key=miner, log=lambda m: None)
    node.start(mine=True)
    provider = Provider(node, miner)
    provider.start()
    try:
        ch = ComputeChannel(f"http://127.0.0.1:{provider.port}", buyer, node.url)
        t = time.perf_counter()
        ch.open(100_000_000)
        print(f"channel opened and confirmed in {time.perf_counter() - t:.2f} s (one block)")
        rng = np.random.default_rng(1)
        print(f"{'product (m x k x n)':>22} {'channel':>10} {'local':>10} {'check':>10} {'GMAC/s':>8}")
        for m, k, n in ((128, 256, 64), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)):
            a = rng.integers(-128, 128, (m, k), dtype=np.int8)
            b = rng.integers(-128, 128, (k, n), dtype=np.int8)
            repeat = 20 if m <= 512 else 3
            remote = timed(lambda: ch.matmul(a, b), repeat)
            local = timed(lambda: local_matmul(a, b), repeat)
            c = local_matmul(a, b)
            check = timed(lambda: verify_product(a, b, c), repeat)
            print(f"{m:>6} x {k:>5} x {n:>5}  {remote * 1000:8.1f}ms {local * 1000:8.1f}ms {check * 1000:8.1f}ms "
                  f"{m * k * n / remote / 1e9:8.2f}")
        x, y = digits(6000, seed=1)
        xt, yt = digits(2000, seed=2)
        model = MLP([256, 128, 10], seed=3)
        steps = 300
        t = time.perf_counter()
        train(model, x, y, steps, matmul=ch.matmul, lr=0.1, seed=4)
        elapsed = time.perf_counter() - t
        print(f"training 256-128-10 on digits through the channel: {steps} steps in {elapsed:.1f} s "
              f"({elapsed / steps * 1000:.0f} ms per step, {model.products} products), "
              f"test accuracy {model.accuracy(xt, yt):.3f}")
        print(f"paid: {ch.owed} yaprak ({params.fmt_amount(ch.owed)}) for {ch.stats['products']} products")
    finally:
        provider.stop()
        node.stop()
        node.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
