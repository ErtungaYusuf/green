"""Compute channels on chain: a buyer locks coins for a provider, the provider
claims what the buyer's vouchers say it is owed, the rest returns at expiry."""

import pytest

from green import params, tx as txmod
from green.chain import payouts
from green.crypto import Key
from green.tx import InvalidTx
from helpers import add, minted, new_chain, supply, tx


def open_channel(chain, buyer, provider, amount=1_000_000, blocks=None):
    t = tx(chain, buyer, "channel_open", provider=provider.address, amount=amount,
           expires=chain.height + 1 + (blocks or params.MIN_CHANNEL_BLOCKS))
    add(chain, buyer, [t])
    return txmod.txid(t)


def claim(chain, provider, cid, total, signer, **kw):
    voucher = txmod.sign_voucher(signer, chain.chain_id, cid, total)
    return tx(chain, provider, "channel_claim", channel=cid, total=total, voucher=voucher, **kw)


def rejected(chain, t, match):
    with pytest.raises(InvalidTx, match=match):
        chain.apply_tx(chain.state.copy(), t, chain.height + 1)


def test_a_provider_claims_what_the_vouchers_say():
    buyer, provider = Key.generate(), Key.generate()
    chain = new_chain(buyer)
    cid = open_channel(chain, buyer, provider)
    assert chain.state.channels[cid]["amount"] == 1_000_000
    assert supply(chain) == minted(chain)                        # locked coins are still counted
    fee = params.MIN_TX_FEE
    add(chain, buyer, [claim(chain, provider, cid, 300_000, buyer)])
    assert chain.state.balance(provider.address) == 300_000 - fee   # the fee comes out of the claim
    add(chain, buyer, [claim(chain, provider, cid, 450_000, buyer)])
    assert chain.state.balance(provider.address) == 450_000 - 2 * fee
    assert chain.state.channels[cid]["paid"] == 450_000
    assert supply(chain) == minted(chain)


def test_claims_that_break_the_rules_are_refused():
    buyer, provider, mallory = Key.generate(), Key.generate(), Key.generate()
    chain = new_chain(buyer)
    cid = open_channel(chain, buyer, provider)
    add(chain, buyer, [claim(chain, provider, cid, 200_000, buyer)])
    rejected(chain, claim(chain, provider, cid, 200_000, buyer), "exceed what was claimed")
    rejected(chain, claim(chain, provider, cid, 1_000_001, buyer), "exceed what was claimed")
    rejected(chain, claim(chain, provider, cid, 300_000, mallory), "voucher not signed")
    rejected(chain, claim(chain, mallory, cid, 300_000, buyer), "only the channel's provider")
    rejected(chain, claim(chain, provider, cid, 200_000 + params.MIN_TX_FEE - 1, buyer), "does not cover its fee")
    rejected(chain, claim(chain, provider, "ab" * 32, 300_000, buyer), "unknown or ended")
    # A voucher is for one channel: it cannot be replayed on another one to the same provider.
    other = open_channel(chain, buyer, provider)
    v = txmod.sign_voucher(buyer, chain.chain_id, cid, 500_000)
    rejected(chain, tx(chain, provider, "channel_claim", channel=other, total=500_000, voucher=v), "voucher not signed")


def test_opening_rules():
    buyer, provider = Key.generate(), Key.generate()
    chain = new_chain(buyer)
    h = chain.height + 1
    for expires in (h + params.MIN_CHANNEL_BLOCKS - 1, h + params.MAX_CHANNEL_BLOCKS + 1):
        rejected(chain, tx(chain, buyer, "channel_open", provider=provider.address, amount=5, expires=expires),
                 "expires must be")
    rejected(chain, tx(chain, buyer, "channel_open", provider=buyer.address, amount=5,
                       expires=h + params.MIN_CHANNEL_BLOCKS), "to someone else")
    rejected(chain, tx(chain, buyer, "channel_open", provider=provider.address, amount=10**18,
                       expires=h + params.MIN_CHANNEL_BLOCKS), "insufficient")
    a, b = Key.generate(), Key.generate()
    ms = txmod.make_multisig_tx(chain.chain_id, 1, [a.pub_hex, b.pub_hex], "channel_open", 0,
                                provider=provider.address, amount=5, expires=h + params.MIN_CHANNEL_BLOCKS)
    with pytest.raises(InvalidTx, match="single key"):
        txmod.check_format(txmod.add_signature(ms, a, chain.chain_id), chain.chain_id)


def test_the_rest_returns_to_the_buyer_at_expiry(monkeypatch):
    monkeypatch.setattr(params, "MIN_CHANNEL_BLOCKS", 3)
    buyer, provider = Key.generate(), Key.generate()
    chain = new_chain(buyer)
    cid = open_channel(chain, buyer, provider, amount=1_000_000, blocks=3)
    expires = chain.state.channels[cid]["expires"]
    add(chain, buyer, [claim(chain, provider, cid, 250_000, buyer)])
    while chain.height < expires:
        add(chain, buyer)
    late = claim(chain, provider, cid, 400_000, buyer)
    rejected(chain, late, "expired")
    rewards_before = chain.state.balance(buyer.address)
    add(chain, buyer)                                            # the block after expiry ends it
    assert cid not in chain.state.channels
    assert chain.state.balance(buyer.address) == rewards_before + params.block_reward(chain.height) + 750_000
    rejected(chain, claim(chain, provider, cid, 400_000, buyer), "unknown or ended")
    assert supply(chain) == minted(chain)


def test_a_claim_can_pay_several_addresses():
    buyer, pool, m1, m2 = (Key.generate() for _ in range(4))
    chain = new_chain(buyer)
    cid = open_channel(chain, buyer, pool)
    payees = sorted([[m1.address, 3], [m2.address, 1]])
    add(chain, buyer, [claim(chain, pool, cid, 400_000 + params.MIN_TX_FEE, buyer, payees=payees)])
    split = dict(payouts(payees, 400_000))
    assert chain.state.balance(m1.address) == split[m1.address] == 300_000
    assert chain.state.balance(m2.address) == 100_000 and chain.state.balance(pool.address) == 0


def test_a_provider_can_close_a_channel_early(monkeypatch):
    monkeypatch.setattr(params, "MIN_CHANNEL_BLOCKS", 3)
    buyer, provider = Key.generate(), Key.generate()
    chain = new_chain(buyer)
    cid = open_channel(chain, buyer, provider, amount=1_000_000, blocks=3)
    expires = chain.state.channels[cid]["expires"]
    rejected(chain, claim(chain, provider, cid, 300_000, buyer, close=False), "close is true")
    before = chain.state.balance(buyer.address)
    add(chain, buyer, [claim(chain, provider, cid, 300_000, buyer, close=True)])
    assert cid not in chain.state.channels                       # the rest came back at once
    # (the buyer also mined that block: its reward and the claim's fee)
    assert chain.state.balance(buyer.address) == before + 700_000 + params.block_reward(chain.height) + params.MIN_TX_FEE
    assert chain.state.balance(provider.address) == 300_000 - params.MIN_TX_FEE
    while chain.height <= expires:                               # the expiry event finds nothing to refund
        add(chain, buyer)
    assert supply(chain) == minted(chain)
