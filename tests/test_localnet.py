"""A local network of five people, started and stopped the way a user does it:
`grn localnet start`, separate node processes, `grn localnet stop`."""

import os
import random
import socket
import subprocess
import sys
import time

import pytest

from green import localnet
from green.node import http_get

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_base() -> int:
    """A base port whose API, P2P and compute ports are all free."""
    for _ in range(50):
        base = random.randrange(20000, 40000, 1000)
        ports = [base + off + i for off in (1, 101, 201) for i in range(len(localnet.PERSONAS))]
        try:
            for port in ports:
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue
    raise RuntimeError("no free port range")


def grn(*args, timeout=240):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, "-m", "green", *args], capture_output=True, text=True, timeout=timeout,
                          cwd=ROOT, env=env)


def test_five_people_start_a_network_buy_and_sell_and_stop(tmp_path):
    root, base = str(tmp_path / "ln"), free_base()
    r = grn("localnet", "start", "--dir", root, "--base-port", str(base))
    assert r.returncode == 0, r.stdout + r.stderr
    net = localnet.load(root)
    try:
        assert [p["name"] for p in net["personas"]] == ["alice", "bob", "carol", "dave", "erin"]
        carol = next(p for p in net["personas"] if p["name"] == "carol")
        end = time.time() + 120

        def settled():
            rows = localnet.status(root)
            if not all(row["running"] and row["peers"] >= 3 for row in rows):
                return False
            heights = [row["height"] for row in rows]          # (miners keep moving the tip: close is enough)
            offers = http_get(f"{localnet.url(carol)}/providers", timeout=5)
            return max(heights) - min(heights) <= 2 and min(heights) >= 5 and len(offers) == 3
        while not settled():
            assert time.time() < end, localnet.status(root)
            time.sleep(1)
        rows = {row["name"]: row for row in localnet.status(root)}
        assert rows["carol"]["balance"] > 0 and not rows["carol"]["mining"]      # the genesis reward is hers
        assert rows["alice"]["mining"] and rows["erin"]["peers"] >= 3
        assert grn("localnet", "start", "--dir", root, "--base-port", str(base)).returncode == 0   # already running

        def bought():                            # the buyers' processes act: coins first, then compute
            rows = localnet.status(root)
            return any(r.get("buying") and (r["buying"]["coins_bought"] or r["buying"]["products"]) for r in rows)
        end = time.time() + 150
        while not bought():
            assert time.time() < end, localnet.status(root)
            time.sleep(2)
    finally:
        r = grn("localnet", "stop", "--dir", root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(row["running"] for row in localnet.status(root))
    for p in net["personas"]:                    # each node shut down cleanly, each buyer after it
        with open(os.path.join(p["data"], "node.log"), encoding="utf-8", errors="replace") as f:
            log = f.read()
        assert "Stopped." in log and "Traceback" not in log, (p["name"], log[-2000:])
        if p["buys"]:
            with open(os.path.join(p["data"], "buyer.log"), encoding="utf-8", errors="replace") as f:
                log = f.read()
            assert "node is gone; stopping" in log and "Traceback" not in log, (p["name"], log[-2000:])
    assert not os.path.exists(os.path.join(root, "pids.json"))        # every process was seen to exit


def test_reset_deletes_only_a_local_network(tmp_path):
    """`grn localnet reset --dir data` must not delete a real node's folder."""
    real = tmp_path / "data"
    real.mkdir()
    (real / "wallet.json").write_text("{}")
    with pytest.raises(localnet.LocalnetError):
        localnet.reset(str(real), log=lambda m: None)
    assert (real / "wallet.json").exists()
    ln = str(tmp_path / "ln")
    localnet.create(ln, base_port=free_base(), log=lambda m: None)
    localnet.reset(ln, log=lambda m: None)
    assert not os.path.exists(ln)
    localnet.reset(ln, log=lambda m: None)                     # nothing there: nothing to do
