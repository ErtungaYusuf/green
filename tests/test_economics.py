"""Money (SPEC.md §3.5): a fixed supply paid out from the protocol's reserve with no
halvings, fees flowing back into it, and shares that pay every miner for its work."""

import random
import socket
import time

import numpy as np
import pytest

from green import params
from green.chain import InvalidBlock, block_hash, make_genesis, share_id, solve, target_of
from green.crypto import Key
from green.node import Node, http_get
from helpers import T0, add, grind, mine, minted, new_chain, supply, tx


@pytest.fixture
def shares_on(monkeypatch):
    monkeypatch.setattr(params, "SHARES_PERCENT", 80)        # (the test default gives it all to the finder)


def make_share(chain, miner, parent=None, t=None):
    """A header mined on `parent` (default: the tip) that no chain contains: with the
    easy test target its ticket meets any share target, like an orphaned block."""
    saved = chain.tip
    if parent is not None:
        chain.tip = parent
    try:
        header, chosen, _ = chain.build_template(miner.address, [], now=t or T0 + 10 * (chain.height + 1) + 3)
    finally:
        chain.tip = saved
    header["nonce"] = random.getrandbits(40)
    return grind(chain, header, chosen)[0]["header"]


def with_shares(chain, key, shares, txs=()):
    header, chosen, _ = chain.build_template(key.address, list(txs), now=T0 + 10 * (chain.height + 1), shares=shares)
    block, _ = grind(chain, header, chosen)
    if shares:
        block["shares"] = list(shares)
    return block


def test_the_supply_never_changes_and_half_of_every_fee_goes_back():
    keys = [Key.generate() for _ in range(5)]
    chain = new_chain(keys[0])
    rng = random.Random(1)
    for _ in range(25):
        rich = max(keys, key=lambda k: chain.state.balance(k.address))
        pays = [tx(chain, rich, "transfer", to=rng.choice(keys).address, amount=rng.randint(1, 10**6),
                   fee=params.MIN_TX_FEE * rng.randint(1, 50))]
        before, fees = chain.state.reserve, pays[0]["fee"]
        finder, _, _ = params.emission(before)
        add(chain, rng.choice(keys), pays)
        assert chain.state.reserve == before - finder + fees * params.FEE_RESERVE_PERCENT // 100
        assert supply(chain) == minted(chain) == params.TOTAL_SUPPLY


def test_shares_pay_their_miners_from_the_reserve(shares_on):
    finder, a, b = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(finder)
    add(chain, finder)
    shares = [make_share(chain, a), make_share(chain, a), make_share(chain, b)]
    reserve = chain.state.reserve
    f, per_share, cut = params.emission(reserve)
    before = chain.state.balance(finder.address)
    assert chain.add_block(with_shares(chain, finder, shares)) == "added"
    assert chain.state.balance(a.address) == 2 * (per_share - cut)
    assert chain.state.balance(b.address) == per_share - cut
    assert chain.state.balance(finder.address) == before + f + 3 * cut
    assert chain.state.reserve == reserve - f - 3 * per_share
    assert supply(chain) == params.TOTAL_SUPPLY
    # The emission splits exactly: the finder's part is the share of it not meant for shares, fixed
    # whether it includes 3 shares or 64, so leaving others' shares out would gain it nothing.
    e = reserve // params.EMISSION_DIVISOR
    assert f + params.SHARE_FACTOR * per_share == e
    assert abs(f - e * (100 - params.SHARES_PERCENT) // 100) <= params.SHARE_FACTOR


def test_a_miner_that_never_finds_a_block_is_still_paid(shares_on):
    """The point of shares: rewards follow work, not luck."""
    big, small = Key.generate(), Key.generate()
    chain = new_chain(big)
    for _ in range(6):
        add(chain, big, [])
        block = with_shares(chain, big, [make_share(chain, small)])
        assert chain.add_block(block) == "added"
    assert all(chain.block(h)["header"]["miner"] == big.address for h in chain.main)     # no block of its own
    assert chain.state.balance(small.address) > 0


def test_shares_that_break_the_rules_make_the_block_invalid(shares_on):
    finder, a = Key.generate(), Key.generate()
    chain = new_chain(finder)
    for _ in range(3):
        add(chain, finder)
    good = make_share(chain, a)

    def refused(shares, match):
        with pytest.raises(InvalidBlock, match=match):
            chain.add_block(with_shares(chain, finder, shares))
    refused([good, good], "duplicate share")
    refused([dict(good, transcript="00" * 32)], "share transcript does not match")
    refused([dict(good, target="00" * 31 + "01")], "wrong target")
    refused([chain.block(chain.main[2])["header"]], "a block of this branch is not a share")
    assert chain.add_block(with_shares(chain, finder, [good])) == "added"
    refused([good], "already included")                         # once only
    # A share mined on a block of another branch does not count on this one.
    side, _ = mine(chain, a, t=T0 + 15, parent=chain.main[1])
    assert chain.add_block(side) == "added" and chain.tip != block_hash(side)
    refused([make_share(chain, a, parent=block_hash(side), t=T0 + 16)], "not mined on this branch")
    # Nor does one mined more than SHARE_WINDOW blocks ago.
    old = make_share(chain, a, parent=chain.main[1], t=T0 + 17)
    for _ in range(params.SHARE_WINDOW):
        add(chain, finder)
    refused([old], "outside the window")


def test_an_orphaned_block_becomes_a_share(shares_on):
    """Two blocks at the same height: the loser's work is not wasted."""
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    add(chain, alice)
    g = chain.tip
    a2, _ = mine(chain, alice)
    b2, _ = mine(chain, bob, t=T0 + 25, parent=g)
    assert chain.add_block(a2) == "added" and chain.add_block(b2) == "added"
    assert chain.tip == block_hash(a2)                          # first seen wins; b2 is orphaned
    assert chain.add_block(with_shares(chain, alice, [b2["header"]])) == "added"
    assert chain.state.balance(bob.address) == params.emission(chain.states[block_hash(a2)].reserve)[1] \
        - params.emission(chain.states[block_hash(a2)].reserve)[2]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_shares_travel_the_network_and_fake_ones_get_the_sender_banned(tmp_path, shares_on, monkeypatch):
    from test_p2p import RawPeer, wait_for
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET // 64)   # most tickets: shares, not blocks
    big, small = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(big.address, "shares"), port=free_port(), miner_key=big,
                log=lambda m: None)
    node.start(mine=True)
    try:
        wait_for(lambda: node.chain.height >= 2, timeout=60, what="blocks")
        node.mining = False
        time.sleep(0.5)
        # A share from a miner that never finds a block, sent over P2P.
        with node.lock:
            header, _, _ = node.chain.build_template(small.address, [])
        header["nonce"] = random.getrandbits(40)
        while True:                                             # one try at a time: a share, not a block
            header["transcript"], ticket, _ = solve(header, node.chain.blobs)
            if target_of(header) < ticket <= node.chain.share_target(header):
                break
            header["nonce"] += 1
        share = {"header": header}
        raw = RawPeer(node, features=["shares"])
        raw.send("share", {"share": share["header"]})
        wait_for(lambda: share_id(share["header"]) in node.share_pool, what="the share in the pool")
        node.mining = True
        wait_for(lambda: node.chain.state.balance(small.address) > 0, timeout=60, what="the share paid")
        assert node.stats["shares_found"] > 0                   # its own near-misses became shares too
        assert http_get(f"{node.url}/supply")["ok"]
        fake = dict(share["header"], nonce=share["header"]["nonce"] + 1)
        raw.send("share", {"share": fake})                      # a transcript that is not its work
        assert raw.closed()
    finally:
        node.stop()
