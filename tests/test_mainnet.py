"""The main network ships with the software: a node given no genesis joins it, and only
its nodes dial the seeds."""

import json
from types import SimpleNamespace

import pytest

from green import cli, mainnet, params
from green.chain import block_hash, make_genesis
from green.crypto import Key
from green.node import Node
from test_p2p import free_port


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    g = make_genesis(Key.generate().address, "a main network")
    path = tmp_path / "mainnet.json"
    path.write_text(json.dumps(g), encoding="utf-8")
    monkeypatch.setattr(mainnet, "PATH", str(path))
    monkeypatch.setattr(params, "SEED_NODES", ["203.0.113.7:8546"])
    return g


def test_a_node_given_no_genesis_joins_the_main_network(tmp_path, bundled):
    data = str(tmp_path / "me")
    g = cli.node_genesis(SimpleNamespace(data=data, genesis=None))
    assert block_hash(g) == block_hash(bundled) == mainnet.genesis_hash()
    assert json.load(open(f"{data}/genesis.json")) == bundled                # kept for the next start
    other = make_genesis(Key.generate().address, "my own network")
    (tmp_path / "other.json").write_text(json.dumps(other))
    mine = cli.node_genesis(SimpleNamespace(data=str(tmp_path / "own"), genesis=str(tmp_path / "other.json")))
    assert block_hash(mine) == block_hash(other)                            # --genesis still wins
    with pytest.raises(cli.CliError, match="no genesis file"):
        cli.node_genesis(SimpleNamespace(data=str(tmp_path / "x"), genesis=str(tmp_path / "missing.json")))


def test_before_the_main_network_exists_a_genesis_is_needed(tmp_path, monkeypatch):
    monkeypatch.setattr(mainnet, "PATH", str(tmp_path / "none.json"))
    with pytest.raises(cli.CliError, match="no genesis file"):
        cli.node_genesis(SimpleNamespace(data=str(tmp_path / "me"), genesis=None))


def test_only_main_network_nodes_dial_the_seeds(tmp_path, bundled):
    on_main = Node(str(tmp_path / "a"), bundled, port=free_port(), log=lambda m: None)
    elsewhere = Node(str(tmp_path / "b"), make_genesis(Key.generate().address, "local"), port=free_port(),
                     log=lambda m: None)
    try:
        assert on_main.p2p.seeds == ["203.0.113.7:8546"]
        assert elsewhere.p2p.seeds == []
    finally:
        on_main.store.close()
        elsewhere.store.close()
