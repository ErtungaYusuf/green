"""Rule changes by version bits: signalling, lock-in, activation, failure,
mandatory activation, reorgs across a period boundary, and old nodes that never
heard of the change (it must be a soft fork for them)."""

import socket

import pytest

from green import params, versionbits as vb
from green.chain import BlobStore, Chain, InvalidBlock, block_hash, make_genesis
from green.crypto import Key
from green.node import Node, http_get
from helpers import T0, grind

LONG = "x" * 150                                       # allowed before testdummy, not after


@pytest.fixture(autouse=True)
def short_periods(monkeypatch):
    monkeypatch.setattr(params, "SIGNAL_PERIOD", 10)
    monkeypatch.setattr(params, "SIGNAL_THRESHOLD", 9)
    monkeypatch.setattr(params, "DEPLOYMENTS", {"testdummy": {"bit": 28, "start": 10, "timeout": 50}})


def make(chain, key, signal=True, msg="", parent=None, t=None, version=None):
    saved = chain.tip
    if parent is not None:
        chain.tip = parent
    try:
        header, chosen, _ = chain.build_template(key.address, [], now=t or T0 + 10 * (chain.height + 1),
                                                 signal_skip=() if signal else ("testdummy",))
    finally:
        chain.tip = saved
    header["msg"] = msg
    if version is not None:
        header["version"] = version
    return grind(chain, header, chosen)[0]


def grow(chain, key, until, signal=True):
    while chain.height < until:
        assert chain.add_block(make(chain, key, signal)) == "added"


def state(chain):
    return chain.deployments.state("testdummy", chain.tip)        # for the next block


def test_signalled_change_locks_in_and_activates():
    key = Key.generate()
    chain = Chain(make_genesis(key.address, "vb", timestamp=T0), BlobStore())
    grow(chain, key, 8)
    assert state(chain) == vb.DEFINED
    assert make(chain, key)["header"]["version"] == vb.TOP_BITS          # nothing to signal yet
    grow(chain, key, 9)
    assert state(chain) == vb.STARTED                                   # block 10 opens the window
    assert make(chain, key)["header"]["version"] == vb.TOP_BITS | 1 << 28
    grow(chain, key, 19)
    assert state(chain) == vb.LOCKED_IN                                 # all 10 blocks of period 1 signalled
    assert chain.add_block(make(chain, key, msg=LONG)) == "added"       # not enforced while locked in
    grow(chain, key, 29)
    assert state(chain) == vb.ACTIVE
    with pytest.raises(InvalidBlock, match="testdummy"):
        chain.add_block(make(chain, key, msg=LONG))
    assert chain.add_block(make(chain, key, msg="short")) == "added"
    assert make(chain, key)["header"]["version"] == vb.TOP_BITS          # signalling stops once active
    info = chain.deployments.info(chain.tip)[0]
    assert (info["name"], info["state"], info["bit"]) == ("testdummy", vb.ACTIVE, 28)


def test_too_few_signals_fail_at_the_timeout():
    key = Key.generate()
    chain = Chain(make_genesis(key.address, "vb fail", timestamp=T0), BlobStore())
    grow(chain, key, 9)
    for _ in range(10):                                                  # 8 of 10: one short of 90%
        signal = chain.height % 10 not in (3, 7)
        assert chain.add_block(make(chain, key, signal)) == "added"
    assert state(chain) == vb.STARTED
    grow(chain, key, 49, signal=False)
    assert state(chain) == vb.FAILED
    grow(chain, key, 70, signal=True)
    assert state(chain) == vb.FAILED                                     # final
    assert chain.add_block(make(chain, key, msg=LONG)) == "added"


def test_mandatory_activation_and_a_minimum_height(monkeypatch):
    key = Key.generate()
    monkeypatch.setattr(params, "DEPLOYMENTS", {"testdummy": {"bit": 28, "start": 10, "timeout": 50,
                                                              "lock_in_on_timeout": True}})
    chain = Chain(make_genesis(key.address, "vb lot", timestamp=T0), BlobStore())
    grow(chain, key, 49, signal=False)
    assert state(chain) == vb.LOCKED_IN                                  # nobody signalled; locked in anyway
    grow(chain, key, 59, signal=False)
    assert state(chain) == vb.ACTIVE

    monkeypatch.setattr(params, "DEPLOYMENTS", {"testdummy": {"bit": 28, "start": 10, "timeout": 50,
                                                              "min_activation": 40}})
    chain = Chain(make_genesis(key.address, "vb trial", timestamp=T0), BlobStore())
    grow(chain, key, 19)
    assert state(chain) == vb.LOCKED_IN
    grow(chain, key, 29)
    assert state(chain) == vb.LOCKED_IN                                  # waits for its minimum height
    grow(chain, key, 39)
    assert state(chain) == vb.ACTIVE


def test_a_reorg_across_a_period_boundary_follows_the_other_branch():
    key = Key.generate()
    chain = Chain(make_genesis(key.address, "vb reorg", timestamp=T0), BlobStore())
    grow(chain, key, 19)                                                 # period 1 fully signalled
    locked_tip = chain.tip
    assert state(chain) == vb.LOCKED_IN
    fork = chain.main[15]
    parent = fork
    for i in range(6):                                                   # 16'..21', not signalling, longer
        block = make(chain, key, signal=False, parent=parent, t=T0 + 100_000 + i)
        assert chain.add_block(block) == "added"
        parent = block_hash(block)
    assert chain.tip == parent and chain.height == 21
    assert state(chain) == vb.STARTED                                    # this branch: 6 of 10 signalled
    assert chain.deployments.state("testdummy", locked_tip) == vb.LOCKED_IN   # the other branch keeps its state


def test_header_version_rules():
    key = Key.generate()
    chain = Chain(make_genesis(key.address, "vb version", timestamp=T0), BlobStore())
    with pytest.raises(InvalidBlock, match="version"):
        chain.add_block(make(chain, key, version=1))                     # old style: no top bits
    with pytest.raises(InvalidBlock, match="version"):
        chain.add_block(make(chain, key, version=0x40000000))
    assert chain.add_block(make(chain, key, version=vb.TOP_BITS | 1 << 5)) == "added"   # unknown bits are fine


def test_nodes_that_never_heard_of_the_change_follow_the_chain(monkeypatch):
    """A soft fork: everything valid under the new rule is valid under the old one."""
    key = Key.generate()
    genesis = make_genesis(key.address, "vb old", timestamp=T0)
    new = Chain(genesis, BlobStore())
    grow(new, key, 35)
    assert state(new) == vb.ACTIVE
    blocks = [new.block(h) for h in new.main[1:]]
    monkeypatch.setattr(params, "DEPLOYMENTS", {})                      # an old node's view
    old = Chain(genesis, BlobStore())
    for b in blocks:
        assert old.add_block(b) == "added"
    assert old.tip == new.tip and old.state.root() == new.state.root()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_node_warns_about_an_unknown_rule_change_and_reports_deployments(tmp_path):
    key = Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(key.address, "vb node", timestamp=T0), port=free_port(),
                miner_key=key, log=lambda m: None)
    for _ in range(12):                                                  # most blocks signal bit 7: unknown here
        assert node.chain.add_block(make(node.chain, key, version=vb.TOP_BITS | 1 << 7)) == "added"
    node.start()
    try:
        assert any("version bit 7" in w for w in node.status()["warnings"])
        rows = http_get(f"{node.url}/deployments")
        assert rows[0]["name"] == "testdummy" and rows[0]["state"] == vb.STARTED and rows[0]["signalled"] == 0
    finally:
        node.stop()


def test_exactly_the_threshold_locks_in():
    key = Key.generate()
    chain = Chain(make_genesis(key.address, "vb edge", timestamp=T0), BlobStore())
    grow(chain, key, 9)
    for _ in range(10):                                                  # 9 of 10 = SIGNAL_THRESHOLD
        assert chain.add_block(make(chain, key, signal=chain.height % 10 != 4)) == "added"
    assert state(chain) == vb.LOCKED_IN
