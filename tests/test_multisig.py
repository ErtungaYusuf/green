"""Multisignature accounts: k of n keys spend together, signed one cosigner at a time."""

import pytest

from green import params, tx as txmod
from green.chain import InvalidBlock
from green.crypto import Key
from green.tx import InvalidTx, add_signature, make_multisig_tx, multisig_address
from helpers import add, grind, new_chain, tx


def account(k=2, n=3):
    keys = [Key.generate() for _ in range(n)]
    pubs = sorted(key.pub_hex for key in keys)
    return keys, pubs, multisig_address(k, pubs)


def signed(chain, keys, pubs, k, signers, kind="transfer", nonce=None, **fields):
    addr = multisig_address(k, pubs)
    t = make_multisig_tx(chain.chain_id, k, pubs, kind, chain.state.nonce(addr) if nonce is None else nonce,
                         txmod.min_fee({"type": kind, **fields}), **fields)
    for key in signers:
        t = add_signature(t, key, chain.chain_id)
    return t


def test_k_of_n_keys_spend_from_a_multisig_account():
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    keys, pubs, addr = account(2, 3)
    assert addr.startswith("grn1") and addr == multisig_address(2, list(reversed(pubs)))   # order does not matter
    assert multisig_address(1, pubs) != addr and multisig_address(2, pubs[:2]) != addr     # k and keys both count
    add(chain, alice, [tx(chain, alice, "transfer", to=addr, amount=10**8)])                # anyone can pay it

    pay = signed(chain, keys, pubs, 2, keys[:2], to=bob.address, amount=1234)
    add(chain, alice, [pay])
    assert chain.state.balance(bob.address) == 1234
    assert chain.state.nonce(addr) == 1

    s = chain.state.copy()
    h = chain.height + 1
    for signers, reason in ((keys[:1], "exactly 2"), (keys, "exactly 2"), ([], "exactly 2")):
        with pytest.raises(InvalidTx, match=reason):
            chain.apply_tx(s, signed(chain, keys, pubs, 2, signers, to=bob.address, amount=1), h)
    outsider = Key.generate()
    with pytest.raises(InvalidTx, match="not one of"):
        add_signature(signed(chain, keys, pubs, 2, [], to=bob.address, amount=1), outsider, chain.chain_id)
    forged = signed(chain, keys, pubs, 2, keys[:2], to=bob.address, amount=1)
    forged["amount"] = 10**7                                                   # changed after signing
    with pytest.raises(InvalidTx, match="bad signature"):
        chain.apply_tx(s, forged, h)
    other_net = signed(chain, keys, pubs, 2, [], to=bob.address, amount=1)
    for key in keys[:2]:
        other_net = add_signature(other_net, key, "ab" * 32)                   # signed for another network
    with pytest.raises(InvalidTx, match="bad signature"):
        chain.apply_tx(s, other_net, h)


def test_malformed_multisig_accounts_are_refused():
    chain = new_chain(Key.generate())
    keys, pubs, _ = account(2, 3)
    good = signed(chain, keys, pubs, 2, keys[:2], to=keys[0].address, amount=1)
    for pub in ({"k": 2, "keys": list(reversed(pubs))},                       # not sorted
                {"k": 2, "keys": [pubs[0], pubs[0], pubs[1]]},                # duplicate key
                {"k": 0, "keys": pubs}, {"k": 4, "keys": pubs},               # k out of range
                {"k": 1, "keys": pubs[:1]},                                   # a single key is not multisig
                {"k": 2, "keys": sorted(Key.generate().pub_hex for _ in range(params.MAX_MULTISIG_KEYS + 1))},
                {"k": 2, "keys": pubs, "extra": 1}):
        with pytest.raises(InvalidTx, match="multisig account"):
            txmod.check_format({**good, "pub": pub}, chain.chain_id)
    with pytest.raises(InvalidTx, match="slot"):
        txmod.check_format({**good, "sig": good["sig"][:2]}, chain.chain_id)


def test_a_multisig_guardian_needs_its_quorum_to_freeze_an_agent():
    """The guardian of an agent no longer has to be a single key."""
    alice, agent = Key.generate(), Key.generate()
    chain = new_chain(alice)
    keys, pubs, guardian = account(2, 3)
    add(chain, alice, [tx(chain, alice, "transfer", to=guardian, amount=10**7),
                       tx(chain, alice, "transfer", to=agent.address, amount=10**7, nonce=1)])
    add(chain, alice, [tx(chain, agent, "guard_set", guardian=guardian, limit=10**6, period=10)])
    freeze = dict(account=agent.address, guardian=guardian, limit=10**6, period=10, frozen=True)
    with pytest.raises(InvalidTx):
        chain.apply_tx(chain.state.copy(), signed(chain, keys, pubs, 2, keys[:1], "guard_update", **freeze),
                       chain.height + 1)
    add(chain, alice, [signed(chain, keys, pubs, 2, keys[1:], "guard_update", **freeze)])
    assert chain.state.guards[agent.address]["frozen"]


def test_a_block_has_a_signature_budget(monkeypatch):
    monkeypatch.setattr(params, "MAX_BLOCK_SIGS", 5)
    alice, bob = Key.generate(), Key.generate()
    chain = new_chain(alice)
    keys, pubs, addr = account(3, 3)
    add(chain, alice, [tx(chain, alice, "transfer", to=addr, amount=10**8)])
    two = [signed(chain, keys, pubs, 3, keys, to=bob.address, amount=1, nonce=i) for i in range(2)]   # 6 signatures
    header, chosen, _ = chain.build_template(alice.address, two)
    assert chosen == two[:1]                                                  # the template stays in budget
    header, _, _ = chain.build_template(alice.address, two[:1])
    from green.crypto import merkle_root
    header["tx_root"] = merkle_root([txmod.txid(t) for t in two])
    block, _ = grind(chain, header, two)
    with pytest.raises(InvalidBlock, match="too many signatures"):
        chain.add_block(block)


def test_cli_multisig_workflow_end_to_end(tmp_path, monkeypatch, capsys):
    """Three people make a 2-of-3 account; two of them sign a payment from it, each
    with their own wallet, passing one transaction file along."""
    import json
    import socket
    import time

    from green import cli
    from green.chain import make_genesis
    from green.node import Node
    from green.tx import make_tx
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    monkeypatch.setattr(params, "ASERT_HALFLIFE", 20)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    miner, payee = Key.generate(), Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(miner.address, "multisig cli"), port=port, miner_key=miner,
                log=lambda m: None)
    node.start(mine=True)

    def wait(cond, what):
        end = time.time() + 60
        while not cond():
            assert time.time() < end, f"timed out waiting for {what}"
            time.sleep(0.1)
    try:
        dirs = [str(tmp_path / f"m{i}") for i in range(3)]
        pubs = []
        for d in dirs:
            assert cli.main(["--data", d, "wallet", "new", "--no-password"]) == 0
            capsys.readouterr()
            assert cli.main(["--data", d, "wallet", "show"]) == 0
            pubs.append(capsys.readouterr().out.split("Public key : ")[1].split()[0])
        assert cli.main(["multisig", "address", "2", *pubs]) == 0
        addr = capsys.readouterr().out.strip()
        assert addr == multisig_address(2, pubs)
        wait(lambda: node.chain.state.balance(miner.address) > 2 * params.YAPRAK_PER_COIN, "reward")
        node.submit_tx(make_tx(miner, node.chain.chain_id, "transfer", node.pending_nonce(miner.address),
                               to=addr, amount=params.YAPRAK_PER_COIN))
        wait(lambda: node.chain.state.balance(addr) > 0, "funding the account")
        txfile = str(tmp_path / "pay.json")
        common = ["--node", node.url]
        assert cli.main([*common, "multisig", "new", "2", *pubs, "--to", payee.address, "--amount", "0.25",
                         "--out", txfile]) == 0
        assert cli.main(["--data", dirs[2], "multisig", "sign", txfile]) == 0
        assert cli.main([*common, "multisig", "send", txfile]) == 1             # one signature is not enough
        assert cli.main(["--data", dirs[0], "multisig", "sign", txfile]) == 0
        with open(txfile) as f:
            assert sum(s is not None for s in json.load(f)["tx"]["sig"]) == 2
        assert cli.main([*common, "multisig", "send", txfile]) == 0
        wait(lambda: node.chain.state.balance(payee.address) == params.YAPRAK_PER_COIN // 4, "the payment")
    finally:
        node.stop()
