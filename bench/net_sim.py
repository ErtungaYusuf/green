"""A small network on this machine: N nodes over the real P2P protocol, two of
them mining, a stream of transactions, then a network split and its healing.

Measures how fast blocks and transactions reach every node, checks that all
nodes end on one chain with identical ledgers (same state root), and that a
split heals by a reorg to the heavier side.

    python bench/net_sim.py [nodes] [seconds per phase]
"""

import os
import random
import shutil
import socket
import statistics
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from green import p2p, params                # noqa: E402
from green.chain import make_genesis         # noqa: E402
from green.crypto import Key                 # noqa: E402
from green.node import Node                  # noqa: E402
from green.tx import make_tx                 # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Watch:
    """Records when each node first saw each block and transaction."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.block_seen: dict[str, dict[int, float]] = {}
        self.tx_seen: dict[str, dict[int, float]] = {}
        self.tx_sent: dict[str, float] = {}
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            now = time.time()
            for i, n in enumerate(self.nodes):
                with n.lock:
                    tip_chain = n.chain.main[-30:]
                    pool = list(n.mempool)
                for bh in tip_chain:
                    self.block_seen.setdefault(bh, {}).setdefault(i, now)
                for tid in pool:
                    self.tx_seen.setdefault(tid, {}).setdefault(i, now)
            time.sleep(0.02)

    def spread(self, seen: dict[str, dict[int, float]], start: dict[str, float] | None = None):
        """Seconds from first sighting (or submission) until every node had it."""
        out = []
        for h, by_node in seen.items():
            if len(by_node) == len(self.nodes):
                t0 = start[h] if start and h in start else min(by_node.values())
                out.append(max(by_node.values()) - t0)
        return out


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    phase = float(sys.argv[2]) if len(sys.argv) > 2 else 40
    params.TARGET_BLOCK_SECONDS = 3
    params.INITIAL_TARGET = params.MAX_TARGET // 60
    params.ASERT_HALFLIFE = 60
    p2p.TRICKLE_MEAN_SECONDS = 0.2
    p2p.MAX_OUTBOUND = 4
    keys = [Key.generate() for _ in range(count)]
    genesis = make_genesis(keys[0].address, "net sim")
    tmp = tempfile.mkdtemp()
    nodes = []
    try:
        for i in range(count):
            seeds = [nodes[j].p2p.address for j in random.Random(i).sample(range(i), min(2, i))] if i else []
            n = Node(os.path.join(tmp, f"n{i}"), genesis, port=free_port(), miner_key=keys[i],
                     peers=seeds, log=lambda m: None)
            n.start(mine=False)
            n.work_tasks = False
            nodes.append(n)
        miners = [0, count // 2]                          # one miner in each half
        for i in miners:
            nodes[i].mining = True
        watch = Watch(nodes)
        t_end = time.time() + phase
        sent = 0
        rng = random.Random(1)
        while time.time() < t_end:                        # phase 1: one network, transactions flowing
            sender = keys[miners[0]]
            node = nodes[miners[0]]
            if node.chain.state.balance(sender.address) > 10**7:
                t = make_tx(sender, node.chain.chain_id, "transfer", node.pending_nonce(sender.address),
                            to=rng.choice(keys).address, amount=1)
                try:
                    tid = node.submit_tx(t)
                    watch.tx_sent[tid] = time.time()
                    sent += 1
                except Exception:
                    pass
            time.sleep(0.25)
        heights = [n.chain.height for n in nodes]
        blocks = watch.spread(watch.block_seen)
        txs = watch.spread(watch.tx_seen, watch.tx_sent)
        degree = [len(n.p2p.all_peers()) for n in nodes]
        print(f"phase 1: {count} nodes, peers per node {min(degree)}-{max(degree)}, heights {min(heights)}-{max(heights)}")
        if blocks:
            print(f"  block reached every node in: median {statistics.median(blocks):.2f} s, "
                  f"max {max(blocks):.2f} s ({len(blocks)} blocks)")
        if txs:
            print(f"  transaction reached every mempool in: median {statistics.median(txs):.2f} s, "
                  f"max {max(txs):.2f} s ({len(txs)} of {sent} sent)")

        # phase 2: split the network in two halves, each with a miner
        half = set(range(count // 2, count))
        for i, n in enumerate(nodes):
            for j, m in enumerate(nodes):
                if (i in half) != (j in half):
                    n.p2p.bans[f"id:{m.p2p.node_id}"] = time.time() + 10**6
            for p in n.p2p.all_peers():
                if any(p.node_id == nodes[j].p2p.node_id for j in range(count) if (i in half) != (j in half)):
                    p.close("simulated partition")
        time.sleep(phase / 2)
        sides = {nodes[0].chain.tip, nodes[count // 2].chain.tip}
        print(f"phase 2: split; the halves are at heights {nodes[0].chain.height} and {nodes[count // 2].chain.height} "
              f"({'diverged' if len(sides) == 2 else 'same tip'})")

        # phase 3: heal (mining stops, so convergence is measured on fixed chains)
        for n in nodes:
            n.mining = False
        healed_at = time.time()
        for n in nodes:
            n.p2p.bans.clear()
            for m in nodes:
                if m is not n:
                    n.p2p.addrman.add("127.0.0.1", m.p2p.port, m.p2p.node_id)
        deadline = time.time() + phase
        while time.time() < deadline:
            tips = {n.chain.tip for n in nodes}
            if len(tips) == 1:
                break
            time.sleep(0.1)
        tips = {n.chain.tip for n in nodes}
        roots = {n.chain.state.root() for n in nodes}
        print(f"phase 3: healed; one tip after {time.time() - healed_at:.2f} s: {len(tips) == 1}, "
              f"identical ledgers (state roots): {len(roots) == 1}, height {nodes[0].chain.height}")
        sent_bytes = [sum(p.bytes_out for p in n.p2p.all_peers()) for n in nodes]
        print(f"  bytes sent per node on live links: {min(sent_bytes):,}-{max(sent_bytes):,}")
        watch.running = False
        assert len(tips) == 1 and len(roots) == 1, "the network did not converge"
    finally:
        for n in nodes:
            n.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
