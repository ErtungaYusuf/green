"""How fast is compute bought through a channel? A local node mines and sells
products; a buyer opens a channel and times products of several sizes and a
training run, against doing the same on this machine. With --rtt and --mbit
the buyer reaches the provider through a proxy that adds that round-trip time
and limits the bandwidth each way, like a link over the internet.

    python bench/compute_bench.py [--rtt MS] [--mbit MBIT]
"""

import argparse
import os
import queue
import shutil
import socket
import sys
import tempfile
import threading
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


class SlowLink:
    """A TCP proxy that delays every byte by rtt/2 and limits each direction to `mbit`."""

    def __init__(self, target_port: int, rtt_ms: float, mbit: float):
        self.target, self.delay = target_port, rtt_ms / 2000
        self.rate = mbit * 1e6 / 8 if mbit else 0.0
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(16)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, _ = self.server.accept()
            except OSError:
                return
            u = socket.create_connection(("127.0.0.1", self.target))
            for s in (c, u):
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            done = threading.Semaphore(0)
            for src, dst in ((c, u), (u, c)):
                q = queue.Queue()
                threading.Thread(target=self._read, args=(src, q), daemon=True).start()
                threading.Thread(target=self._write, args=(dst, q, done), daemon=True).start()
            threading.Thread(target=self._close, args=(c, u, done), daemon=True).start()

    def _read(self, src, q):
        free_at = time.perf_counter()               # when the bandwidth-limited link is free again
        while True:
            try:
                data = src.recv(65536)
            except OSError:
                data = b""
            now = time.perf_counter()
            if self.rate:
                free_at = max(free_at, now) + len(data) / self.rate
            q.put((max(free_at, now) + self.delay, data))
            if not data:
                return

    def _write(self, dst, q, done):
        while True:
            due, data = q.get()
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            try:
                if not data:                        # pass the end on gracefully (a close here could reset
                    dst.shutdown(socket.SHUT_WR)    # the other direction, losing data in flight)
                    break
                dst.sendall(data)
            except OSError:
                break
        done.release()

    @staticmethod
    def _close(c, u, done):
        done.acquire()
        done.acquire()                              # both directions finished
        for s in (c, u):
            s.close()


def timed(fn, repeat):
    fn()                                                   # warm up
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t) / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtt", type=float, default=0, help="round-trip time to add, in milliseconds")
    ap.add_argument("--mbit", type=float, default=0, help="bandwidth each way, in Mbit/s (0: unlimited)")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    params.INITIAL_TARGET = params.MAX_TARGET              # a block whenever the miner tries
    tmp = tempfile.mkdtemp()
    buyer, miner = Key.generate(), Key.generate()
    node = Node(os.path.join(tmp, "n"), make_genesis(buyer.address, "bench"), port=free_port(),
                miner_key=miner, log=lambda m: None)
    node.start(mine=True)
    provider = Provider(node, miner)
    provider.start()
    try:
        port = provider.port
        if args.rtt or args.mbit:
            port = SlowLink(provider.port, args.rtt, args.mbit).port
            print(f"link: {args.rtt:g} ms round trip, {args.mbit or 'unlimited'} Mbit/s each way")
        ch = ComputeChannel(f"http://127.0.0.1:{port}", buyer, node.url)
        t = time.perf_counter()
        ch.open(100_000_000)
        print(f"channel opened and confirmed in {time.perf_counter() - t:.2f} s (one block)")
        rng = np.random.default_rng(1)
        print(f"{'product (m x k x n)':>22} {'channel':>10} {'local':>10} {'check':>10} {'GMAC/s':>8}")
        sizes = ((128, 256, 64), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048))
        for m, k, n in sizes if not args.mbit else sizes[:3]:
            a = rng.integers(-128, 128, (m, k), dtype=np.int8)
            b = rng.integers(-128, 128, (k, n), dtype=np.int8)
            repeat = (20 if m <= 512 else 3) if not (args.rtt or args.mbit) else 3
            remote = timed(lambda: ch.matmul(a, b), repeat)
            local = timed(lambda: local_matmul(a, b), repeat)
            c = local_matmul(a, b)
            check = timed(lambda: verify_product(a, b, c), repeat)
            print(f"{m:>6} x {k:>5} x {n:>5}  {remote * 1000:8.1f}ms {local * 1000:8.1f}ms {check * 1000:8.1f}ms "
                  f"{m * k * n / remote / 1e9:8.2f}")
        x, y = digits(6000, seed=1)
        xt, yt = digits(2000, seed=2)
        steps = args.steps
        for name, mm in (("one at a time", lambda a, b: ch.matmul(a, b)), ("backward pairs together", ch.matmul)):
            model = MLP([256, 128, 10], seed=3)
            t = time.perf_counter()
            train(model, x, y, steps, matmul=mm, lr=0.1, seed=4)
            elapsed = time.perf_counter() - t
            print(f"training 256-128-10 on digits, {name}: {steps} steps in {elapsed:.1f} s "
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
