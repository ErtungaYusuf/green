"""Addresses are checksummed (bech32m), so a typo is rejected instead of burning coins."""

import pytest

from green.crypto import ADDRESS_HRP, Key, address_of, bech32m_decode, bech32m_encode, h, is_address
from green.tx import InvalidTx
from helpers import new_chain, tx

CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def test_bip350_vectors():
    for v in ("A1LQFN3A", "a1lqfn3a", "abcdef1l7aum6echk45nj3s0wdvt2fg8x9yrzpqzd3ryx",
              "split1checkupstagehandshakeupstreamerranterredcaperredlc445v", "?1v759aa"):
        assert bech32m_decode(v) is not None, v
    for bad in ("A1LQFN3a", "a1lqfn3b", "1lqfn3a", "a1lqfn", "a1lqfn3a" + "\x7f", "a1lqfnba"):
        assert bech32m_decode(bad) is None, bad


def test_address_round_trip():
    k = Key.generate()
    assert k.address.startswith(ADDRESS_HRP + "1") and len(k.address) == 42
    assert is_address(k.address)
    assert bech32m_decode(k.address) == (ADDRESS_HRP, h(bytes.fromhex(k.pub_hex))[:20])
    assert address_of(k.pub_hex) == k.address == bech32m_encode(ADDRESS_HRP, h(bytes.fromhex(k.pub_hex))[:20])


def test_every_single_character_typo_is_rejected():
    addr = Key.generate().address
    for i in range(len(ADDRESS_HRP) + 1, len(addr)):
        for c in CHARSET:
            if c != addr[i]:
                assert not is_address(addr[:i] + c + addr[i + 1:]), (i, c)


def test_swapped_neighbours_are_rejected():
    addr = Key.generate().address
    for i in range(len(ADDRESS_HRP) + 1, len(addr) - 1):
        if addr[i] != addr[i + 1]:
            assert not is_address(addr[:i] + addr[i + 1] + addr[i] + addr[i + 2:]), i


def test_only_the_canonical_spelling_is_an_address():
    addr = Key.generate().address
    assert bech32m_decode(addr.upper()) is not None      # valid bech32m...
    assert not is_address(addr.upper())                  # ...but one account must have one spelling
    assert not is_address("g" + "ab" * 20)               # the old unchecked format
    other_net = bech32m_encode("bc", bytes(20))
    assert not is_address(other_net)
    assert not is_address(bech32m_encode(ADDRESS_HRP, bytes(19)))


def test_transfer_to_a_mistyped_address_is_refused():
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    typo = bob.address[:-1] + ("q" if bob.address[-1] != "q" else "p")
    with pytest.raises(InvalidTx, match="recipient"):
        chain.apply_tx(chain.state.copy(), tx(chain, alice, "transfer", to=typo, amount=1), 1)
