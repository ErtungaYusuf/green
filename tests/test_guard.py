"""Guarded accounts: an autonomous agent stays accountable to a person.

A guardian caps what the account can spend per period and can freeze it. The
account can only make its own guard stricter; it cannot loosen, hand over or
remove it -- that is the whole point of an off switch.
"""

import numpy as np
import pytest

from green import params, tx as txmod
from green.crypto import Key
from green.tx import InvalidTx
from helpers import add, new_chain, post_job, supply, minted, tx

FEE = params.MIN_TX_FEE


@pytest.fixture
def world():
    producer, agent, guardian, shop = (Key.generate() for _ in range(4))
    chain = new_chain(producer)
    add(chain, producer, [tx(chain, producer, "transfer", to=k.address, amount=5 * 10**7, nonce=i)
                          for i, k in enumerate((agent, guardian))])
    return chain, producer, agent, guardian, shop


def guard(chain, agent, guardian, limit, period, **kw):
    return tx(chain, agent, "guard_set", guardian=guardian.address, limit=limit, period=period, **kw)


def update(chain, guardian, agent, limit, period, frozen=False, new_guardian=None, **kw):
    return tx(chain, guardian, "guard_update", account=agent.address,
              guardian=(new_guardian or guardian).address, limit=limit, period=period, frozen=frozen, **kw)


def pay(chain, who, to, amount, **kw):
    return tx(chain, who, "transfer", to=to.address, amount=amount, **kw)


def rejected(chain, t, match):
    with pytest.raises(InvalidTx, match=match):
        chain.apply_tx(chain.state.copy(), t, chain.height + 1)


def test_spending_is_capped_per_period_and_resets(world):
    chain, producer, agent, guardian, shop = world
    limit = 10**6
    add(chain, producer, [guard(chain, agent, guardian, limit=limit, period=50)])
    add(chain, producer, [pay(chain, agent, shop, limit - FEE)])               # exactly the limit, fee included
    rejected(chain, pay(chain, agent, shop, 1), "spending limit")
    while (chain.height + 1) // 50 == (chain.height) // 50:                  # wait for the next period
        add(chain, producer)
    add(chain, producer)
    add(chain, producer, [pay(chain, agent, shop, limit - FEE)])               # a fresh allowance
    assert chain.state.balance(shop.address) == 2 * (limit - FEE)
    assert supply(chain) == minted(chain)


def test_escrow_counts_as_spending(world):
    chain, producer, agent, guardian, shop = world
    add(chain, producer, [guard(chain, agent, guardian, limit=10**5, period=1000)])
    rng = np.random.default_rng(1)
    a = rng.integers(-128, 128, size=(256, 256), dtype=np.int8)
    job_id = chain.blobs.put(txmod.encode_job(a, a))
    post = tx(chain, agent, "job_post", job=job_id, dims=[1, 1, 1], price=10**6)
    rejected(chain, post, "spending limit")                                    # 1M escrow > 100k limit


def test_the_account_can_only_tighten_its_own_guard(world):
    chain, producer, agent, guardian, shop = world
    add(chain, producer, [guard(chain, agent, guardian, limit=10**6, period=100)])
    rejected(chain, guard(chain, agent, guardian, limit=10**7, period=100), "only the guardian")   # looser limit
    rejected(chain, guard(chain, agent, guardian, limit=10**6, period=10), "only the guardian")    # shorter period
    rejected(chain, guard(chain, agent, shop, limit=10**6, period=100), "only the guardian")       # new guardian
    rejected(chain, tx(chain, agent, "guard_release", account=agent.address), "only the account's guardian")
    rejected(chain, guard(chain, agent, agent, limit=1, period=1), "its own guardian")
    add(chain, producer, [guard(chain, agent, guardian, limit=10**5, period=200)])                 # stricter: fine
    g = chain.state.guards[agent.address]
    assert (g["limit"], g["period"], g["guardian"]) == (10**5, 200, guardian.address)


def test_the_guardian_freezes_loosens_hands_over_and_releases(world):
    chain, producer, agent, guardian, shop = world
    add(chain, producer, [guard(chain, agent, guardian, limit=10**5, period=100)])
    add(chain, producer, [update(chain, guardian, agent, limit=10**5, period=100, frozen=True)])
    for t in (pay(chain, agent, shop, 1), guard(chain, agent, guardian, limit=1, period=100)):
        rejected(chain, t, "frozen")                                          # nothing at all while frozen
    add(chain, producer, [update(chain, guardian, agent, limit=10**7, period=100, frozen=False)])
    add(chain, producer, [pay(chain, agent, shop, 5 * 10**6)])                 # loosened by the guardian
    successor = Key.generate()
    add(chain, producer, [update(chain, guardian, agent, limit=10**7, period=100, new_guardian=successor)])
    rejected(chain, tx(chain, guardian, "guard_release", account=agent.address), "only the account's guardian")
    add(chain, producer, [pay(chain, producer, successor, 10**6)])
    add(chain, producer, [tx(chain, successor, "guard_release", account=agent.address)])
    assert agent.address not in chain.state.guards
    add(chain, producer, [pay(chain, agent, shop, 2 * 10**7)])                 # free again
    assert supply(chain) == minted(chain)


def test_strangers_cannot_touch_a_guard(world):
    chain, producer, agent, guardian, shop = world
    add(chain, producer, [guard(chain, agent, guardian, limit=10**5, period=100)])
    rejected(chain, update(chain, producer, agent, limit=10**9, period=100), "only the account's guardian")
    rejected(chain, tx(chain, producer, "guard_release", account=agent.address), "only the account's guardian")
    rejected(chain, tx(chain, producer, "guard_release", account=shop.address), "only the account's guardian")


def test_incoming_coins_are_never_blocked(world):
    chain, producer, agent, guardian, shop = world
    add(chain, producer, [guard(chain, agent, guardian, limit=0, period=10)])
    add(chain, producer, [update(chain, guardian, agent, limit=0, period=10, frozen=True)])
    before = chain.state.balance(agent.address)
    add(chain, producer, [pay(chain, producer, agent, 12345)])
    assert chain.state.balance(agent.address) == before + 12345
