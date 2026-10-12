"""The fee market: block templates sell space by fee rate, the mempool is bounded
by fee rate, and a stuck transaction can be replaced by paying more."""

import socket

import pytest

from green import node as nodemod, params
from green.chain import make_genesis
from green.crypto import Key
from green.node import Node
from green.tx import InvalidTx
from helpers import T0, mine, new_chain, tx

FEE = params.MIN_TX_FEE


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def funded_chain(n):
    keys = [Key.generate() for _ in range(n)]
    chain = new_chain(keys[0])
    block, _ = mine(chain, keys[0], [tx(chain, keys[0], "transfer", to=k.address, amount=10**7, nonce=i)
                                     for i, k in enumerate(keys[1:])])
    chain.add_block(block)
    return chain, keys


@pytest.fixture
def node(tmp_path):
    """An unstarted node (no server, no miner) with funded keys."""
    keys = [Key.generate() for _ in range(6)]
    n = Node(str(tmp_path / "n"), make_genesis(keys[0].address, "fees", timestamp=T0), port=free_port(),
             miner_key=keys[0], log=lambda m: None)
    block, _ = mine(n.chain, keys[0], [tx(n.chain, keys[0], "transfer", to=k.address, amount=10**7, nonce=i)
                                       for i, k in enumerate(keys[1:])])
    n._add_block(block)
    return n, keys


def test_block_template_sells_space_by_fee_rate(monkeypatch):
    chain, (miner, a, b, c) = funded_chain(4)
    txs = [tx(chain, k, "transfer", to=miner.address, amount=1, fee=fee)
           for k, fee in ((a, FEE), (b, 5 * FEE), (c, 3 * FEE))]
    monkeypatch.setattr(params, "MAX_BLOCK_TXS", 2)
    block, _ = mine(chain, miner, txs)
    assert [t["fee"] for t in block["txs"]] == [5 * FEE, 3 * FEE]


def test_block_template_keeps_each_senders_nonce_order(monkeypatch):
    chain, (miner, a, b) = funded_chain(3)
    cheap = tx(chain, a, "transfer", to=miner.address, amount=1, fee=FEE)
    rich = tx(chain, a, "transfer", to=miner.address, amount=1, fee=50 * FEE, nonce=cheap["nonce"] + 1)
    mid = tx(chain, b, "transfer", to=miner.address, amount=1, fee=3 * FEE)
    block, _ = mine(chain, miner, [rich, mid, cheap])
    assert block["txs"].index(cheap) < block["txs"].index(rich)     # otherwise the block is invalid
    assert chain.add_block(block) == "added"
    chain2, (m2, a2, b2) = funded_chain(3)
    cheap2 = tx(chain2, a2, "transfer", to=m2.address, amount=1, fee=FEE)
    rich2 = tx(chain2, a2, "transfer", to=m2.address, amount=1, fee=50 * FEE, nonce=cheap2["nonce"] + 1)
    mid2 = tx(chain2, b2, "transfer", to=m2.address, amount=1, fee=3 * FEE)
    monkeypatch.setattr(params, "MAX_BLOCK_TXS", 1)
    block2, _ = mine(chain2, m2, [rich2, mid2, cheap2])
    assert block2["txs"] == [mid2]                 # the rich tx is unreachable without the cheap one


def test_replace_by_fee(node):
    n, (_, a, b, *_) = node
    first = tx(n.chain, a, "transfer", to=b.address, amount=5, fee=2 * FEE)
    n.submit_tx(first)
    stingy = tx(n.chain, a, "transfer", to=b.address, amount=5, fee=2 * FEE + 1)
    with pytest.raises(InvalidTx, match="replacement fee too low"):
        n.submit_tx(stingy)
    bumped = tx(n.chain, a, "transfer", to=b.address, amount=5, fee=4 * FEE)
    n.submit_tx(bumped)
    assert list(n.mempool.values()) == [bumped]
    assert n.pending_nonce(a.address) == first["nonce"] + 1


def test_invalid_replacement_changes_nothing(node):
    n, (_, a, b, *_) = node
    first = tx(n.chain, a, "transfer", to=b.address, amount=5)
    n.submit_tx(first)
    broke = tx(n.chain, a, "transfer", to=b.address, amount=10**12, fee=10 * FEE)
    with pytest.raises(InvalidTx, match="balance"):
        n.submit_tx(broke)
    assert list(n.mempool.values()) == [first]


def test_full_mempool_evicts_the_cheapest(node, monkeypatch):
    n, (_, a, b, c, d, e) = node
    monkeypatch.setattr(nodemod, "MAX_MEMPOOL", 3)
    for k, fee in ((a, FEE), (b, 2 * FEE), (c, 3 * FEE)):
        n.submit_tx(tx(n.chain, k, "transfer", to=e.address, amount=1, fee=fee))
    n.submit_tx(tx(n.chain, d, "transfer", to=e.address, amount=1, fee=int(1.5 * FEE)))
    assert sorted(t["fee"] for t in n.mempool.values()) == [int(1.5 * FEE), 2 * FEE, 3 * FEE]
    assert n.pending_nonce(a.address) == n.chain.state.nonce(a.address)      # a's tx is gone
    with pytest.raises(InvalidTx, match="fee rate is too low"):
        n.submit_tx(tx(n.chain, e, "transfer", to=a.address, amount=1, fee=int(1.2 * FEE)))
    assert len(n.mempool) == 3


def test_one_sender_cannot_fill_the_mempool(node, monkeypatch):
    n, (_, a, b, *_) = node
    monkeypatch.setattr(nodemod, "MAX_MEMPOOL_PER_SENDER", 2)
    start = n.chain.state.nonce(a.address)
    for i in range(2):
        n.submit_tx(tx(n.chain, a, "transfer", to=b.address, amount=1, nonce=start + i))
    with pytest.raises(InvalidTx, match="too many pending"):
        n.submit_tx(tx(n.chain, a, "transfer", to=b.address, amount=1, nonce=start + 2))


def test_a_payment_and_its_spend_survive_a_new_block(node):
    """The mempool used to be re-sorted by nonce after every block, which put a
    new account's spend (nonce 0) before the payment funding it, and dropped it."""
    n, (miner, a, *_) = node
    newcomer, shop = Key.generate(), Key.generate()
    pay = tx(n.chain, a, "transfer", to=newcomer.address, amount=10**6)
    assert pay["nonce"] == 0
    n.submit_tx(tx(n.chain, a, "transfer", to=shop.address, amount=1))   # a's nonce 0: pay must be 1
    pay = tx(n.chain, a, "transfer", to=newcomer.address, amount=10**6, nonce=1)
    n.submit_tx(pay)
    spend = tx(n.chain, newcomer, "transfer", to=shop.address, amount=10**5)
    n.submit_tx(spend)
    block, _ = mine(n.chain, miner, [])
    n._add_block(block)
    assert {t["nonce"] for t in n.mempool.values() if t["pub"] == newcomer.pub_hex} == {0}
    assert len(n.mempool) == 3
