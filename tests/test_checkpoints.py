"""Checkpoints: a network's founder signs which blocks are final while the network is
young; no chain that leaves the latest checkpoint out is accepted, however much work it
has, and the period ends at a height or when the founder signs its end."""

import time

import pytest

from green import params
from green.chain import (BlobStore, Chain, InvalidBlock, InvalidCheckpoint, block_hash, checkpoint_bytes,
                         checkpoint_end_bytes, make_genesis)
from green.crypto import Key
from helpers import T0, add, mine


def network(founder: Key, until: int = 1000):
    miner = Key.generate()
    genesis = make_genesis(miner.address, f"test network | checkpoints {founder.pub_hex} until {until}", timestamp=T0)
    return Chain(genesis, BlobStore()), miner


def sign(founder: Key, chain: Chain, height: int, block: str) -> dict:
    return {"height": height, "block": block, "sig": founder.sign(checkpoint_bytes(chain.chain_id, height, block))}


def side_branch(chain: Chain, key: Key, parent: str, count: int) -> list[dict]:
    """`count` blocks on `parent`, mined but not added (a rival chain)."""
    blocks, prev = [], parent
    saved = (chain.tip, chain.main[:])
    for i in range(count):
        height = chain.meta[prev]["height"] + 1
        block, _ = mine(chain, key, parent=prev, t=T0 + 10 * height + 7)
        assert chain.add_block(block) in ("added", "orphan")
        blocks.append(block)
        prev = block_hash(block)
    return blocks


def test_the_founder_is_named_by_the_genesis_and_only_their_signature_counts():
    founder, stranger = Key.generate(), Key.generate()
    chain, miner = network(founder, until=50)
    assert chain.authority == (founder.pub_hex, 50)
    for _ in range(3):
        add(chain, miner)
    cp = sign(founder, chain, 2, chain.main[2])
    with pytest.raises(InvalidCheckpoint, match="bad signature"):
        chain.accept_checkpoint(sign(stranger, chain, 2, chain.main[2]))
    with pytest.raises(InvalidCheckpoint, match="bad signature"):
        chain.accept_checkpoint(dict(cp, height=3))                      # the signature covers the height
    assert chain.accept_checkpoint(cp) is True and chain.checkpoint["block"] == chain.main[2]
    assert chain.accept_checkpoint(cp) is False                           # already have it
    assert chain.accept_checkpoint(sign(founder, chain, 1, chain.main[1])) is False   # older
    assert chain.accept_checkpoint(sign(founder, chain, 51, "ab" * 32)) is False      # past "until"
    plain = Chain(make_genesis(miner.address, "no checkpoints", timestamp=T0), BlobStore())
    with pytest.raises(InvalidCheckpoint, match="no checkpoints"):
        plain.accept_checkpoint(cp)


def test_a_heavier_chain_that_leaves_out_the_checkpoint_is_refused():
    founder, attacker = Key.generate(), Key.generate()
    chain, miner = network(founder)
    for _ in range(4):
        add(chain, miner)
    honest_tip = chain.tip
    chain.accept_checkpoint(sign(founder, chain, 3, chain.main[3]))
    # The attacker forks at block 2 and mines more blocks than the honest chain has.
    with pytest.raises(InvalidBlock, match="conflicts with the checkpoint"):
        side_branch(chain, attacker, chain.main[2], 6)
    assert chain.tip == honest_tip


def test_without_the_checkpoint_the_same_attack_would_win():
    founder, attacker = Key.generate(), Key.generate()
    chain, miner = network(founder)
    for _ in range(4):
        add(chain, miner)
    honest = chain.main[3]
    side_branch(chain, attacker, chain.main[2], 6)
    assert not chain.on_main(honest)                                      # most work wins, as in Bitcoin


def test_a_checkpoint_pulls_a_node_off_an_attackers_heavier_chain():
    founder, attacker = Key.generate(), Key.generate()
    chain, miner = network(founder)
    for _ in range(4):
        add(chain, miner)
    honest = chain.main[3]
    side_branch(chain, attacker, chain.main[2], 6)                        # heard first: it is heavier
    assert not chain.on_main(honest)
    assert chain.accept_checkpoint(sign(founder, chain, 3, honest)) is True
    assert chain.on_main(honest) and chain.ancestor_at(chain.tip, 3) == honest
    add(chain, miner)                                                     # and mining goes on from there
    assert chain.ancestor_at(chain.tip, 3) == honest


def test_checkpoints_only_move_forward_along_one_chain():
    founder, attacker = Key.generate(), Key.generate()
    chain, miner = network(founder)
    for _ in range(4):
        add(chain, miner)
    rival = side_branch(chain, attacker, chain.main[1], 2)               # lighter: heights 2 and 3 off block 1
    assert chain.on_main(chain.main[2]) and not chain.on_main(block_hash(rival[1]))
    chain.accept_checkpoint(sign(founder, chain, 2, chain.main[2]))
    with pytest.raises(InvalidCheckpoint, match="does not extend"):
        chain.accept_checkpoint(sign(founder, chain, 3, block_hash(rival[1])))
    assert chain.checkpoint["height"] == 2


def test_a_checkpoint_on_a_block_nobody_has_halts_nothing():
    """A stolen key signs a checkpoint on a made-up block. Enforced at once, it made
    every next block "conflict with the checkpoint", for good: ending the period keeps
    the last checkpoint. A checkpoint is enforced only once its block is here."""
    founder = Key.generate()
    chain, miner = network(founder)
    for _ in range(4):
        add(chain, miner)
    chain.accept_checkpoint(sign(founder, chain, 3, chain.main[3]))
    assert chain.accept_checkpoint(sign(founder, chain, 5, "ab" * 32)) is True     # passed on, not enforced
    assert chain.checkpoint["height"] == 3 and chain.pending_checkpoint["block"] == "ab" * 32
    chain.accept_checkpoint({"end": True, "sig": founder.sign(checkpoint_end_bytes(chain.chain_id))})
    for _ in range(3):
        add(chain, miner)                                                 # the network goes on


def test_a_checkpoint_on_a_branch_nobody_has_seen_cannot_undo_the_last_one():
    """A stolen key and a private branch forked below the last checkpoint: signed
    before the branch was seen, its checkpoint used to be taken unchecked, and the
    branch then replaced the checkpointed block. Now it waits for its block, which
    never comes in: the branch leaves the last checkpoint out."""
    founder, thief = Key.generate(), Key.generate()
    chain, miner = network(founder)
    for _ in range(4):
        add(chain, miner)
    shadow = Chain(chain.block(chain.main[0]), BlobStore())             # the thief's own node
    shadow.add_block(chain.block(chain.main[1]))
    rival = []
    for _ in range(4):
        block, _ = mine(shadow, thief, t=T0 + 10 * (shadow.height + 1) + 7)
        assert shadow.add_block(block) == "added"
        rival.append(block)                                               # heights 2..5 off block 1
    honest_2 = chain.main[2]
    chain.accept_checkpoint(sign(founder, chain, 2, honest_2))
    chain.accept_checkpoint(sign(founder, chain, 3, block_hash(rival[1])))
    for block in rival:
        try:
            chain.add_block(block)
        except InvalidBlock:
            pass                                                          # past the checkpoint: refused
    assert chain.checkpoint["block"] == honest_2 and not chain.has_block(block_hash(rival[1]))
    assert block_hash(chain.block_at(2)) == honest_2 and chain.ancestor_at(chain.tip, 2) == honest_2


def test_a_checkpoint_heard_before_its_block_is_taken_when_the_block_arrives():
    founder = Key.generate()
    chain, miner = network(founder)
    for _ in range(3):
        add(chain, miner)
    follower = Chain(chain.block(chain.main[0]), BlobStore())
    for h in (1, 2):
        follower.add_block(chain.block(chain.main[h]))
    cp = sign(founder, chain, 3, chain.main[3])
    assert follower.accept_checkpoint(cp) is True and follower.checkpoint is None
    follower.add_block(chain.block(chain.main[3]))
    assert follower.checkpoint == cp and follower.pending_checkpoint is None


def test_the_founder_ends_the_checkpoint_period():
    founder = Key.generate()
    chain, miner = network(founder)
    for _ in range(3):
        add(chain, miner)
    chain.accept_checkpoint(sign(founder, chain, 1, chain.main[1]))
    with pytest.raises(InvalidCheckpoint):
        chain.accept_checkpoint({"end": True, "sig": Key.generate().sign(checkpoint_end_bytes(chain.chain_id))})
    end = {"end": True, "sig": founder.sign(checkpoint_end_bytes(chain.chain_id))}
    assert chain.accept_checkpoint(end) is True and chain.checkpoint_status()["ended"]
    assert chain.accept_checkpoint(sign(founder, chain, 2, chain.main[2])) is False   # no more checkpoints
    assert chain.checkpoint["height"] == 1                                # the last one stays final


def test_live_nodes_follow_the_founder_and_refuse_a_heavier_attack(tmp_path):
    """The founder's node signs checkpoints as it mines; a follower hears them over the
    network. An attacker mines a longer chain alone, then connects: the follower keeps the
    founder's chain, and a node joining afterwards picks it too."""
    from test_p2p import start_node, wait_for
    founder, miner, attacker_key = Key.generate(), Key.generate(), Key.generate()
    genesis = make_genesis(miner.address, f"live | checkpoints {founder.pub_hex} until 100000")
    started = []
    try:
        a = start_node(tmp_path / "a", genesis, key=miner, mine=True, checkpoint_key=founder)
        started.append(a)
        b = start_node(tmp_path / "b", genesis, peers=[a.p2p.address])
        started.append(b)
        wait_for(lambda: b.chain.checkpoint and b.chain.checkpoint["height"] >= 8, timeout=60, what="checkpoints")
        a.mining = False
        time.sleep(2)                                    # (the miner finishes the batch it has)
        wait_for(lambda: b.chain.tip == a.chain.tip, timeout=30, what="the follower to catch up")
        cp = b.chain.checkpoint
        honest_tip, honest_height = a.chain.tip, a.chain.height
        assert honest_height < params.MAX_REORG_DEPTH      # so the checkpoint, not the depth rule, is what holds
        c = start_node(tmp_path / "c", genesis, key=attacker_key, mine=True)          # alone, no peers
        started.append(c)
        wait_for(lambda: c.chain.height > honest_height + 10, timeout=60, what="the attacker's longer chain")
        c.mining = False
        assert c.chain.chainwork(c.chain.tip) > a.chain.chainwork(honest_tip)
        b.p2p.add_manual(c.p2p.address)
        wait_for(lambda: any(p.node_id == c.p2p.node_id for p in b.p2p.all_peers()), what="the attacker connected")
        time.sleep(3)
        assert b.chain.tip == a.chain.tip == honest_tip, (b.chain.height, a.chain.height, c.chain.height)
        assert b.chain.ancestor_at(b.chain.tip, cp["height"]) == cp["block"] and not b.chain.on_main(c.chain.tip)
        d = start_node(tmp_path / "d", genesis, peers=[a.p2p.address, c.p2p.address])        # joins later
        started.append(d)
        wait_for(lambda: d.chain.tip == honest_tip, timeout=60, what="the new node to pick the founder's chain")
        assert http_checkpoint(d)["checkpoint"]["block"] == cp["block"] or d.chain.checkpoint["height"] >= cp["height"]
    finally:
        for n in started:
            n.stop()


def http_checkpoint(node) -> dict:
    import json
    import urllib.request
    with urllib.request.urlopen(f"{node.url}/checkpoint", timeout=10) as r:
        return json.loads(r.read())


def test_the_founders_commands_from_key_to_end(tmp_path, capsys):
    """grn checkpoint key, genesis --checkpoints, a node signing them, status, end."""
    import json
    from green import cli
    from green.node import Node
    from test_p2p import free_port, wait_for
    d = str(tmp_path / "founder")
    assert cli.main(["--data", d, "checkpoint", "key", "--no-password"]) == 0
    pub = json.load(open(cli.checkpoint_key_path(d)))["pub"]
    assert cli.main(["--data", d, "wallet", "new", "--no-password"]) == 0
    assert cli.main(["--data", d, "genesis", "--msg", "founded", "--checkpoints", "--checkpoint-until", "5000"]) == 0
    genesis = json.load(open(f"{d}/genesis.json"))
    assert json.load(open(cli.checkpoint_key_path(d)))["pub"] == pub      # the key file is untouched
    assert f"checkpoints {pub} until 5000" in genesis["header"]["msg"]
    key = cli.load_checkpoint_key(d)
    node = Node(d, genesis, port=free_port(), miner_key=cli.require_wallet(d), log=lambda m: None, checkpoint_key=key)
    node.start(mine=True)
    try:
        wait_for(lambda: node.chain.checkpoint is not None, timeout=60, what="a checkpoint")
        url = ["--node", node.url]
        capsys.readouterr()
        assert cli.main(url + ["checkpoint", "status"]) == 0
        out = capsys.readouterr().out
        assert pub in out and "until     : block 5000" in out.lower().replace("until     :", "until     :")
        assert cli.main(url + ["--data", d, "checkpoint", "end"]) == 1          # needs --yes
        assert cli.main(url + ["--data", d, "checkpoint", "end", "--yes"]) == 0
        assert node.chain.checkpoint_status()["ended"]
        height = node.chain.checkpoint["height"]
        wait_for(lambda: node.chain.height > height + params.CHECKPOINT_EVERY + params.CHECKPOINT_DEPTH + 2,
                 timeout=60, what="more blocks")
        assert node.chain.checkpoint["height"] == height                         # no new checkpoints
    finally:
        node.stop()
    again = Node(d, genesis, port=free_port(), log=lambda m: None)              # a restart remembers both
    again.start()
    try:
        assert again.chain.checkpoint["height"] == height and again.chain.checkpoint_status()["ended"]
    finally:
        again.stop()


def test_the_checkpoint_key_is_encrypted_with_a_password(tmp_path, monkeypatch):
    from green import cli
    d = str(tmp_path / "f")
    monkeypatch.setenv(cli.CHECKPOINT_PASSWORD_ENV, "a good password")
    assert cli.main(["--data", d, "checkpoint", "key"]) == 0
    assert "secret" not in open(cli.checkpoint_key_path(d)).read()
    key = cli.load_checkpoint_key(d)
    monkeypatch.setenv(cli.CHECKPOINT_PASSWORD_ENV, "the wrong one")
    with pytest.raises(cli.CliError, match="wrong checkpoint key password"):
        cli.load_checkpoint_key(d)
    assert cli.main(["--data", d, "checkpoint", "key"]) == 1                   # never overwritten
    monkeypatch.setenv(cli.CHECKPOINT_PASSWORD_ENV, "a good password")
    assert cli.load_checkpoint_key(d).pub_hex == key.pub_hex
