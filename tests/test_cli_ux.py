"""The everyday commands: `grn` alone, `grn use`, receive, balance, send with a question
first, and --json for programs and AI agents."""

import json

import pytest

from green import cli
from green.chain import make_genesis
from green.crypto import Key
from green.node import Node
from test_app import free_port


def run(capsys, *argv) -> tuple[int, str]:
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_use_remembers_the_folder_and_json_answers_for_agents(tmp_path, capsys, monkeypatch):
    wallet = tmp_path / "w"
    assert run(capsys, "--data", str(wallet), "wallet", "new", "--no-password")[0] == 0
    assert run(capsys, "use", str(wallet))[0] == 0
    code, text = run(capsys, "--json", "receive")                   # no --data: remembered
    addr = json.loads(text)["address"]
    assert code == 0 and addr.startswith("grn1")
    code, text = run(capsys, "--json", "--node", f"http://127.0.0.1:{free_port()}")   # no node there
    view = json.loads(text)
    assert code == 0 and view["address"] == addr and view["node"] is None
    with open(cli.USER_CONFIG) as f:
        assert json.load(f)["data"] == str(wallet)


def test_balance_and_send_with_a_question_first(tmp_path, capsys, monkeypatch):
    alice = tmp_path / "alice"
    assert run(capsys, "--data", str(alice), "wallet", "new", "--no-password")[0] == 0
    key = cli.load_wallet(str(alice))
    node = Node(str(tmp_path / "n"), make_genesis(key.address, "ux"), port=free_port(), miner_key=key,
                log=lambda m: None)
    node.start()
    common = ["--data", str(alice), "--node", node.url]
    try:
        code, text = run(capsys, *common, "--json", "balance")
        assert code == 0 and json.loads(text)["balance"] > 0
        bob = Key.generate().address
        assert run(capsys, *common, "send", bob, "1")[0] == 1          # not a terminal and no --yes: refused
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        assert run(capsys, *common, "send", bob, "1")[0] == 1          # asked, and said no
        code, text = run(capsys, *common, "--json", "send", bob, "1", "--yes")
        assert code == 0 and json.loads(text)["to"] == bob and len(json.loads(text)["txid"]) == 64
    finally:
        node.stop()


def test_ask_without_a_model_says_how_to_choose_one(tmp_path, capsys):
    key_dir = tmp_path / "a"
    assert run(capsys, "--data", str(key_dir), "wallet", "new", "--no-password")[0] == 0
    key = cli.load_wallet(str(key_dir))
    node = Node(str(tmp_path / "n"), make_genesis(key.address, "ask"), port=free_port(), log=lambda m: None)
    node.start()
    try:
        assert cli.main(["--data", str(key_dir), "--node", node.url, "ask", "hi"]) == 1
        assert "grn use --model" in capsys.readouterr().err
    finally:
        node.stop()
