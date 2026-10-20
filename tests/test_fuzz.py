"""Fuzzing what strangers can send: signed transactions with random, mostly well-formed
fields, applied to a ledger with an open job, a disputed batch, a channel and a guard;
and random messages to every P2P handler of a live node.

A transaction may be refused only with InvalidTx (or MissingData, for data the node
must fetch first); any other exception would stop a node in the middle of a block.
Every transaction that is accepted takes exactly its fee out of circulation: balances,
escrows, bonds, channels and the reserve add up to what they did, minus the fee.
A P2P handler may only refuse (Misbehaving, or a malformed-message error the worker
punishes); anything else would leave a peer unpunished and the node in an unknown state.
"""

import random

import numpy as np
import pytest

from green import p2p, params, tx as txmod
from green.chain import MissingData, make_genesis
from green.crypto import Key
from green.tx import InvalidTx, make_tx
from helpers import add, challenge, commit, new_chain, post_job, reveal, tx
from test_p2p import RawPeer, nodes, wait_for  # noqa: F401  (nodes is a fixture)

CHAOS = [0, -1, 2**63, 2**64, True, None, "", 1.5, float("nan"), [], {}, "zz", "00" * 32, [[]], [None]]


def circulating(s) -> int:
    return sum(s.balances.values()) + s.locked_total() + s.reserve


@pytest.mark.parametrize("seed", range(4))
def test_random_signed_transactions_are_refused_cleanly_and_conserve_money(seed):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    producer, alice, bob, carol = (Key.generate() for _ in range(4))
    chain = new_chain(producer)
    add(chain, producer, [tx(chain, producer, "transfer", to=k.address, amount=3 * 10**7, nonce=i)
                          for i, k in enumerate((alice, bob, carol))])
    a, b, job = post_job(chain, alice, nrng, dims=(1, 2, 1), price=1000)
    ctx, sale = commit(chain, bob, job, [[0, 0, 0]], [a[:, :params.UNIT].astype(np.int64) @ b[:params.UNIT].astype(np.int64)])
    add(chain, producer, [ctx])
    bid = ctx["commitment"]
    add(chain, producer, [reveal(chain, bob, job, bid, sale)])
    add(chain, producer, [challenge(chain, alice, job, bid, [[0, 0, 0]])])          # an open dispute
    opening = tx(chain, alice, "channel_open", provider=bob.address, amount=10**6,
                 expires=chain.height + 1 + params.MIN_CHANNEL_BLOCKS)
    add(chain, producer, [opening])
    cid = txmod.txid(opening)
    add(chain, producer, [tx(chain, carol, "guard_set", guardian=bob.address, limit=10**6, period=100)])
    keys = [alice, bob, carol, producer]
    addrs = [k.address for k in keys] + [Key.generate().address]
    total = [0]

    def payees():
        return sorted([[x, rng.choice([1, 2, 1000, params.MAX_PAYEE_WEIGHT])] for x in rng.sample(addrs, rng.randint(2, 4))])
    gen = {
        "to": lambda: rng.choice(addrs), "amount": lambda: rng.choice([1, 500, 10**6, 10**8, 2**62]),
        "job": lambda: rng.choice([job, job, "ab" * 32]), "dims": lambda: [rng.randint(1, 3) for _ in range(3)],
        "price": lambda: rng.choice([1, 1000, 10**6, 2**40]), "units": lambda: rng.choice(["80", "40", "c0", "20", "8000"]),
        "commitment": lambda: rng.choice([bid, "cd" * 32]), "batch": lambda: rng.choice([bid, bid, "cd" * 32]),
        "root": lambda: rng.choice([sale.root, "ee" * 32]), "manifest": lambda: rng.choice([sale.manifest, "ee" * 32]),
        "unit": lambda: rng.choice([[0, 0, 0], [0, 1, 0], [1, 0, 0], [15, 15, 15]]),
        "result": lambda: rng.choice([sale.leaves[0], "ee" * 32]), "proof": lambda: rng.choice([[], ["ee" * 32]]),
        "prog": lambda: "ab" * 32, "input": lambda: "ab" * 32, "fuel": lambda: rng.choice([1, 10**6]),
        "task": lambda: "ab" * 32, "output": lambda: "ab" * 32, "guardian": lambda: rng.choice(addrs),
        "limit": lambda: rng.choice([0, 1000, 10**9]), "period": lambda: rng.choice([1, 100, 10**8]),
        "frozen": lambda: rng.choice([True, False]), "account": lambda: rng.choice(addrs),
        "provider": lambda: rng.choice(addrs), "channel": lambda: rng.choice([cid, cid, "ab" * 32]),
        "expires": lambda: chain.height + rng.choice([0, params.MIN_CHANNEL_BLOCKS + 1, params.MAX_CHANNEL_BLOCKS + 5]),
        "total": lambda: total[0], "payees": payees, "close": lambda: rng.choice([True, True, False]),
        "voucher": lambda: txmod.sign_voucher(rng.choice([alice, alice, bob]), chain.chain_id, cid,
                                              total[0] + rng.choice([0, 0, 0, 1])),
    }

    def field(name):
        return rng.choice(CHAOS) if rng.random() < 0.08 else gen[name]()
    before = circulating(chain.state)
    outcomes = {"accepted": 0, "refused": 0}
    for _ in range(5000):
        kind = rng.choice(list(txmod.FIELDS))
        total[0] = rng.choice([1, 1000, 5000, 10**6, 10**6 + 1])
        fields = {f: field(f) for f in txmod.FIELDS[kind]}
        fields.update({f: field(f) for f in txmod.OPTIONAL.get(kind, ()) if rng.random() < 0.4})
        sender = rng.choice(keys)
        try:
            t = make_tx(sender, chain.chain_id, kind, chain.state.nonce(sender.address),
                        rng.choice([None, None, None, 0, 10**7]), **fields)
        except (TypeError, ValueError, OverflowError):
            continue                                         # cannot even be encoded and signed
        s = chain.state.copy()
        try:
            chain.apply_tx(s, t, chain.height + 1)
        except (InvalidTx, MissingData):
            outcomes["refused"] += 1
            continue
        outcomes["accepted"] += 1
        assert circulating(s) == before - t["fee"], (kind, fields)
    assert outcomes["accepted"] > 100 and outcomes["refused"] > 1000      # it reached both sides


def test_random_messages_to_every_handler_only_get_refused(nodes):
    rng = random.Random(7)
    key = Key.generate()
    node = nodes("n", make_genesis(key.address, "fuzz"), key=key)
    raw = RawPeer(node, features=list(p2p.FEATURES))
    wait_for(lambda: any(p.node_id == raw.node_id for p in node.p2p.all_peers()))
    peer = next(p for p in node.p2p.all_peers() if p.node_id == raw.node_id)
    blk = node.chain.block(node.chain.tip)
    shapes = {
        "inv": {"items": [["tx", "00" * 32]]}, "getdata": {"items": [["block", node.chain.tip]]},
        "headers": {"headers": [blk["header"]]}, "getheaders": {"locator": [node.chain.tip], "stop": None},
        "block": {"block": blk}, "cmpct": {"header": blk["header"], "salt": "00" * 8, "ids": [], "fill": []},
        "getblob": {"hash": "00" * 32, "max": 100}, "addr": {"addrs": [{"host": "1.2.3.4", "port": 1}]},
        "checkpoint": {"cp": {"height": 1, "block": "00" * 32, "sig": "00" * 64}},
        "share": {"share": blk["header"]}, "getblocktxn": {"block": node.chain.tip, "idx": [0]},
    }
    values = CHAOS + [2**200, -2**63, 10**30, float("inf"), "x" * 10000, "ff" * 64, "a" * 70, node.chain.tip]

    def value(depth=0):
        r = rng.random()
        if depth < 3 and r < 0.15:
            return [value(depth + 1) for _ in range(rng.randint(0, 4))]
        if depth < 3 and r < 0.3:
            return {rng.choice(["hash", "height", "items", "header", "block", "txs", "max", "sig", "ids"]): value(depth + 1)
                    for _ in range(rng.randint(0, 4))}
        return rng.choice(values)
    allowed = (p2p.Misbehaving, ValueError, KeyError, TypeError, AttributeError, RecursionError)
    commands = [c for c in p2p.CMDS if c not in p2p.BINARY and c != "version"]
    for _ in range(3000):
        cmd = rng.choice(commands)
        m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in shapes.get(cmd, {}).items()}
        for _ in range(rng.randint(1, 3)):
            if m and rng.random() < 0.7:
                k = rng.choice(list(m))
                m[k] = value() if rng.random() < 0.6 or not isinstance(m[k], dict) or not m[k] else {**m[k], rng.choice(list(m[k])): value()}
            else:
                m[rng.choice(["hash", "items", "headers", "cp", "block", "max", "header", "share"])] = value()
        peer.alive = True                                     # keep handling even once it would be banned
        try:
            node.p2p._dispatch(peer, cmd, m)
        except allowed:
            pass
    assert node.status()["height"] >= 0                       # still answering
