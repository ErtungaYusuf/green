"""Developer and operator surface: the versioned API, fee estimates, long-polling
for blocks, the config file, password files, log files and metrics."""

import logging
import socket
import threading
import time
import urllib.error

import pytest

from green import cli, node as nodemod, params, tx as txmod
from green.chain import make_genesis
from green.crypto import Key
from green.node import Node, http_get
from green.tx import make_tx
from helpers import T0, grind


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def node(tmp_path):
    key = Key.generate()
    n = Node(str(tmp_path / "n"), make_genesis(key.address, "api", timestamp=T0), port=free_port(), miner_key=key,
             log=lambda m: None)
    n.start()
    yield n
    n.stop()


def next_block(n):
    header, txs, _ = n.chain.build_template(n.miner_key.address, list(n.mempool.values()),
                                            now=T0 + 10 * (n.chain.height + 1))
    return grind(n.chain, header, txs)[0]


def test_the_versioned_api_matches_the_old_paths(node):
    assert http_get(f"{node.url}/v1/status")["genesis"] == http_get(f"{node.url}/status")["genesis"]
    assert http_get(f"{node.url}/v1/account/{node.miner_key.address}")["balance"] > 0


def test_fee_estimate_follows_the_mempool(node, monkeypatch):
    est = http_get(f"{node.url}/v1/fee?blocks=2")
    assert est["basis"] == "minimum" and est["transfer_fee"] == params.MIN_TX_FEE
    monkeypatch.setattr(params, "MAX_BLOCK_TXS", 10)
    key = node.miner_key
    for i in range(15):                                   # fee rates rise with i
        node.submit_tx(make_tx(key, node.chain.chain_id, "transfer", i, params.MIN_TX_FEE * (i + 1),
                               to=Key.generate().address, amount=1))
    one = node.fee_estimate(1)                            # room for 9 (90% of 10): must beat the 10th best
    tenth_best = sorted((t["fee"] / txmod.tx_size(t) for t in node.mempool.values()), reverse=True)[9]
    assert one["basis"] == "mempool" and one["fee_rate"] >= tenth_best
    assert one["transfer_fee"] > params.MIN_TX_FEE
    assert node.fee_estimate(3)["basis"] == "minimum"     # three blocks hold all 15


def test_long_poll_returns_on_the_next_block(node, monkeypatch):
    tip = node.chain.tip
    out = {}
    t = threading.Thread(target=lambda: out.update(r=http_get(f"{node.url}/v1/wait?tip={tip}&timeout=20", timeout=30)))
    started = time.time()
    t.start()
    time.sleep(0.5)
    node.submit_block(next_block(node))
    t.join(10)
    assert out["r"]["changed"] and out["r"]["tip"] == node.chain.tip and time.time() - started < 5
    r = http_get(f"{node.url}/v1/wait?tip={node.chain.tip}&timeout=0.5")
    assert not r["changed"]                               # nothing new: returns at the timeout
    monkeypatch.setattr(nodemod, "MAX_WAITERS", 0)
    with pytest.raises(urllib.error.HTTPError) as e:
        http_get(f"{node.url}/v1/wait?tip={node.chain.tip}&timeout=1")
    assert e.value.code == 503


def test_metrics_include_the_supply_audit(node):
    import urllib.request
    with urllib.request.urlopen(f"{node.url}/metrics", timeout=5) as r:
        text = r.read().decode()
    assert "green_supply_ok 1" in text and "green_warnings 0" in text


def test_config_file_settings_and_command_line_overrides(tmp_path, monkeypatch):
    data = tmp_path / "d"
    data.mkdir()
    (data / "green.toml").write_text('[node]\np2p_port = 9999\nmine = true\nlog_level = "debug"\n'
                                     'peer = ["127.0.0.1:1"]\n', encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "cmd_node", lambda args: seen.append(args))
    assert cli.main(["--data", str(data), "node"]) == 0
    assert (seen[0].p2p_port, seen[0].mine, seen[0].log_level, seen[0].peer) == (9999, True, "debug", ["127.0.0.1:1"])
    assert cli.main(["--data", str(data), "node", "--p2p-port", "7777", "--log-level", "warning"]) == 0
    assert (seen[1].p2p_port, seen[1].log_level) == (7777, "warning")       # the command line wins
    for bad in ('[node]\nprot = 1\n', '[node]\nport = "8545"\n', '[node]\nmine = 1\n', '[node]\nlog_level = "loud"\n',
                '[node\n'):
        (data / "green.toml").write_text(bad, encoding="utf-8")
        assert cli.main(["--data", str(data), "node"]) == 1             # refused with a message, not ignored


def test_wallet_password_from_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GREEN_WALLET_PASSWORD", "correct horse")
    key = cli.create_wallet(str(tmp_path), "correct horse")
    monkeypatch.delenv("GREEN_WALLET_PASSWORD")
    pw = tmp_path / "password"
    pw.write_text("correct horse\n", encoding="utf-8")
    monkeypatch.setenv("GREEN_WALLET_PASSWORD_FILE", str(pw))
    assert cli.load_wallet(str(tmp_path)).address == key.address
    monkeypatch.setenv("GREEN_WALLET_PASSWORD_FILE", str(tmp_path / "missing"))
    with pytest.raises(cli.CliError, match="password file"):
        cli.load_wallet(str(tmp_path))


def test_log_file_gets_the_messages(tmp_path):
    path = tmp_path / "node.log"
    logger = cli.setup_logging("info", str(path))
    logger.info("hello from the node")
    logger.debug("not at info level")
    for h in logger.handlers:
        h.flush()
    text = path.read_text(encoding="utf-8")
    assert "hello from the node" in text and "not at info level" not in text
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    assert logging.getLogger("green").handlers == []
