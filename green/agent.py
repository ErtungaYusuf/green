"""An autonomous agent that lives on the Green network.

The point of Green is that compute is money: an agent can pay the network to
run its own AI workloads, and can earn by contributing compute (mining and
doing others' jobs). This module is the decision loop that ties the two sides
together, so an agent can sustain itself without a human in the loop:

  * it watches its own balance;
  * when it has a surplus above a reserve, it spends on a pending goal (here, a
    neural-network inference bought from the network);
  * when it is short, it waits and keeps earning.

The earning side is just a mining Node with the agent's wallet (it already
mines and works tasks). This module is the spending side plus the policy that
decides when to act. On a single machine the miner and the buyer can be the
same wallet (so a purchase only costs fees); on a real network they differ and
the same code pays a stranger. The policy is what makes it an agent rather than
a script.
"""

import time

import numpy as np

from . import params
from .nn import MatmulClient, QuantMLP, predictions
from .node import http_get, wait_for_block


class Goal:
    """One thing the agent wants computed: an inference over an input batch."""

    def __init__(self, name: str, mlp: QuantMLP, x: np.ndarray):
        self.name = name
        self.mlp = mlp
        self.x = x
        self.result = None
        self.predictions = None

    def cost_estimate(self, price: int) -> int:
        """Rough upper bound on what running this goal will cost (escrow+fees)."""
        total = 0
        h_cols = self.x.shape[1]
        for w in self.mlp.weights:
            mu = -(-w.shape[0] // params.UNIT)
            ku = -(-w.shape[1] // params.UNIT)
            nu = -(-h_cols // params.UNIT)
            total += price * mu * ku * nu + params.MIN_TX_FEE
        return total


class Agent:
    """Autonomous participant: earns in the background, spends on goals it can afford."""

    def __init__(self, node_url: str, secret_hex: str, reserve: int = 2 * params.YAPRAK_PER_COIN,
                 price: int = params.YAPRAK_PER_COIN // 100, log=print):
        from .crypto import Key
        self.node = node_url.rstrip("/")
        self.key = Key.from_hex(secret_hex)
        self.reserve = reserve
        self.price = price
        self.log = log
        self.client = MatmulClient(self.node, secret_hex)
        self.earned_start = None
        self.spent = 0
        self.frozen = False

    def account(self) -> dict:
        return http_get(f"{self.node}/account/{self.key.address}")

    def balance(self) -> int:
        return http_get(f"{self.node}/account/{self.key.address}")["balance"]

    def decide(self, goal: Goal) -> bool:
        """Policy: act only if the surplus over the reserve covers the goal, and
        only within what the agent's guardian allows."""
        acct = self.account()
        bal = acct["balance"]
        if self.earned_start is None:
            self.earned_start = bal
        need = goal.cost_estimate(self.price)
        guard = acct.get("guard")
        if guard and guard["frozen"]:
            self.frozen = True
            self.log(f"[agent] frozen by its guardian {guard['guardian'][:16]}...; stopping")
            return False
        if guard and need > guard["spend_left"]:
            self.log(f"[agent] the guardian allows {params.fmt_amount(guard['spend_left'])} more this period, "
                     f"'{goal.name}' needs ~{params.fmt_amount(need)}; waiting")
            return False
        if bal - need < self.reserve:
            self.log(f"[agent] not enough budget for '{goal.name}' "
                     f"(balance {params.fmt_amount(bal)}, needs ~{params.fmt_amount(need)} + "
                     f"reserve {params.fmt_amount(self.reserve)}); still earning")
            return False
        return True

    def pursue(self, goal: Goal, timeout: float = 180) -> Goal:
        """Run one goal on the network (blocking), recording what it cost."""
        before = self.balance()
        self.log(f"[agent] sending '{goal.name}' to the network ({goal.x.shape[1]} samples, "
                 f"{len(goal.mlp.weights)} layers)")
        goal.result = goal.mlp.forward_green(self.client, goal.x, self.price, timeout,
                                             log=lambda m: self.log("   " + m))
        goal.predictions = predictions(goal.result)
        after = self.balance()
        self.spent += max(0, before - after)
        self.log(f"[agent] '{goal.name}' done -> predictions {goal.predictions.tolist()}")
        return goal

    def run(self, goals: list[Goal], poll: float = 2.0, max_wait: float = 300) -> list[Goal]:
        """Autonomously pursue a queue of goals, waiting for funds when short."""
        pending = list(goals)
        done = []
        waited = 0.0
        tip = None
        while pending:
            goal = pending[0]
            if self.decide(goal):
                done.append(self.pursue(goal))
                pending.pop(0)
                waited = 0.0
            elif self.frozen:
                break
            else:
                if waited >= max_wait:
                    self.log(f"[agent] no budget built up in {int(max_wait)} s, stopping")
                    break
                # Funds only change with a block: wait for the next one (at most `poll` s).
                started = time.time()
                tip = wait_for_block(self.node, tip, poll)
                waited += time.time() - started
        net = self.spent
        self.log(f"[agent] {len(done)}/{len(goals)} goals completed, net spend {params.fmt_amount(net)}")
        return done
