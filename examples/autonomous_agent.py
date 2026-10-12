"""A self-contained autonomous agent living on the Green network.

One process that starts its own mining node (how the agent earns), then runs an
autonomous loop that spends its earnings to run its own neural-network
inferences on the network. No human in the loop after launch: it mines, watches
its balance, and buys compute for its goals when it can afford them.

    python examples/autonomous_agent.py

Everything runs locally on a throwaway network.
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green import params                              # noqa: E402
from green.agent import Agent, Goal                   # noqa: E402
from green.chain import make_genesis                  # noqa: E402
from green.crypto import Key                          # noqa: E402
from green.nn import QuantMLP                          # noqa: E402
from green.node import Node                            # noqa: E402


def main():
    # Fast local network so the demo does not take long.
    params.INITIAL_TARGET = params.MAX_TARGET // 500
    params.TARGET_BLOCK_SECONDS = 2

    data = tempfile.mkdtemp(prefix="green-agent-")
    key = Key.generate()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    log(f"Agent born. Address: {key.address}")
    node = Node(data, make_genesis(key.address, "autonomous agent"), port=0 or _free(),
                miner_key=key, log=log, use_gpu=True)
    node.start(mine=True)
    log(f"The agent runs its own node ({node.url}) and has started earning by mining")

    mlp = QuantMLP.random([256, 512, 256, 10], seed=7)
    rng = np.random.default_rng(1)
    goals = [Goal(f"inference-{i}", mlp, rng.integers(-127, 128, size=(256, 4), dtype=np.int8))
             for i in range(3)]

    agent = Agent(node.url, key.secret_hex(),
                  reserve=3 * params.YAPRAK_PER_COIN, price=params.YAPRAK_PER_COIN // 100, log=log)
    try:
        done = agent.run(goals, poll=2.0, max_wait=120)
        log("--- summary ---")
        log(f"Once its earnings passed the reserve, {len(done)} goals were run autonomously")
        for g in done:
            ok = np.array_equal(g.result, mlp.forward_local(g.x))
            log(f"  {g.name}: predictions {g.predictions.tolist()}  (matches the local reference: {'YES' if ok else 'NO'})")
        log(f"Current balance: {params.fmt_amount(agent.balance())} (what is above the reserve is net earnings)")
    finally:
        node.stop()
        shutil.rmtree(data, ignore_errors=True)


def _free():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
