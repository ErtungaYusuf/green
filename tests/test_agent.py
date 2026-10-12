"""An autonomous agent earns on the network and spends on its own inference goals."""

import socket
import time

import numpy as np
import pytest

from green import params
from green.agent import Agent, Goal
from green.chain import make_genesis
from green.crypto import Key
from green.nn import QuantMLP
from green.node import Node, http_post
from green.tx import make_tx


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def net(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    miner = Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(miner.address, "agent"), port=free_port(),
                miner_key=miner, log=lambda m: None)
    node.start(mine=True)
    yield node, miner
    node.stop()


def wait_balance(node, addr, at_least, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if node.chain.state.balance(addr) >= at_least:
            return
        time.sleep(0.2)
    raise AssertionError("funds did not arrive")


def test_agent_waits_for_funds_then_completes_goals(net):
    node, miner = net
    agent_key = Key.generate()
    # Fund the agent from the miner, but only a little at first.
    wait_balance(node, miner.address, 3 * params.YAPRAK_PER_COIN)
    pay = make_tx(miner, node.chain.chain_id, "transfer", node.pending_nonce(miner.address),
                  to=agent_key.address, amount=3 * params.YAPRAK_PER_COIN)
    http_post(f"{node.url}/tx", pay)
    wait_balance(node, agent_key.address, 3 * params.YAPRAK_PER_COIN)

    mlp = QuantMLP.random([256, 256, 10], seed=5)
    rng = np.random.default_rng(1)
    goals = [Goal(f"classify-{i}", mlp, rng.integers(-127, 128, size=(256, 4), dtype=np.int8))
             for i in range(2)]

    logs = []
    agent = Agent(node.url, agent_key.secret_hex(), reserve=params.YAPRAK_PER_COIN,
                  price=params.YAPRAK_PER_COIN // 100, log=logs.append)
    done = agent.run(goals, poll=0.5, max_wait=60)

    assert len(done) == 2
    for g in done:
        # bit-exact against the local reference
        assert np.array_equal(g.result, mlp.forward_local(g.x))
        assert g.predictions is not None
    # the agent never let its balance fall below the reserve
    assert agent.balance() >= params.YAPRAK_PER_COIN
    assert agent.spent >= 0


def test_agent_refuses_goals_it_cannot_afford(net):
    node, miner = net
    agent_key = Key.generate()
    # Agent has no coins; with a high reserve it must refuse to spend.
    mlp = QuantMLP.random([256, 256, 10], seed=6)
    x = np.random.default_rng(2).integers(-127, 128, size=(256, 4), dtype=np.int8)
    agent = Agent(node.url, agent_key.secret_hex(), reserve=5 * params.YAPRAK_PER_COIN,
                  log=lambda m: None)
    done = agent.run([Goal("no-budget", mlp, x)], poll=0.2, max_wait=2)
    assert done == []
    assert agent.spent == 0


def test_a_guarded_agent_obeys_its_guardian(net):
    """The off switch: a person caps what the agent spends, and can freeze it."""
    node, miner = net
    agent_key = Key.generate()
    wait_balance(node, miner.address, 3 * params.YAPRAK_PER_COIN)
    cid = node.chain.chain_id
    http_post(f"{node.url}/tx", make_tx(miner, cid, "transfer", node.pending_nonce(miner.address),
                                        to=agent_key.address, amount=3 * params.YAPRAK_PER_COIN))
    wait_balance(node, agent_key.address, 3 * params.YAPRAK_PER_COIN)
    # The agent accepts the miner as its guardian, with a tiny allowance.
    http_post(f"{node.url}/tx", make_tx(agent_key, cid, "guard_set", 0, guardian=miner.address,
                                        limit=params.YAPRAK_PER_COIN // 1000, period=10**6))
    end = time.time() + 30
    while node.account(agent_key.address)["guard"] is None and time.time() < end:
        time.sleep(0.2)
    mlp = QuantMLP.random([256, 256, 10], seed=5)
    x = np.random.default_rng(3).integers(-127, 128, size=(256, 4), dtype=np.int8)
    logs = []
    agent = Agent(node.url, agent_key.secret_hex(), reserve=0, price=params.YAPRAK_PER_COIN // 100, log=logs.append)
    assert agent.run([Goal("too-expensive", mlp, x)], poll=0.2, max_wait=1) == []
    assert any("guardian allows" in m for m in logs)
    # The guardian freezes it: the agent stops at once instead of waiting for funds.
    http_post(f"{node.url}/tx", make_tx(miner, cid, "guard_update", node.pending_nonce(miner.address),
                                        account=agent_key.address, guardian=miner.address,
                                        limit=10 * params.YAPRAK_PER_COIN, period=10**6, frozen=True))
    end = time.time() + 30
    while not (node.account(agent_key.address)["guard"] or {}).get("frozen") and time.time() < end:
        time.sleep(0.2)
    logs.clear()
    start = time.time()
    assert agent.run([Goal("anything", mlp, x)], poll=0.2, max_wait=60) == []
    assert time.time() - start < 5 and any("frozen by its guardian" in m for m in logs)
