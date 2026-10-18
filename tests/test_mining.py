"""Controlling a running miner: on/off, how much of the time it works, GPU or CPU."""

import socket
import time
import urllib.error

import pytest

from green.chain import make_genesis
from green.crypto import Key
from green.node import Node, http_get, http_post


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def node(tmp_path):
    key = Key.generate()
    n = Node(str(tmp_path / "n"), make_genesis(key.address, "mining"), port=free_port(), miner_key=key,
             log=lambda m: None)
    n.start()
    yield n
    n.stop()


def units_in(node, seconds: float) -> int:
    u = node.stats["units"]
    time.sleep(seconds)
    return node.stats["units"] - u


def test_intensity_and_gpu_can_change_while_mining(node):
    r = http_post(f"{node.url}/mining", {"on": True, "intensity": 100})
    assert r["mining"] and r["intensity"] == 100
    time.sleep(1)
    full = units_in(node, 3)
    assert http_post(f"{node.url}/mining", {"intensity": 10})["intensity"] == 10
    time.sleep(1)
    tenth = units_in(node, 3)
    assert 0 < tenth < full / 2, (tenth, full)                   # it rests ~90% of the time
    st = http_get(f"{node.url}/status")
    assert st["intensity"] == 10 and st["mining"]
    for gpu in (True, False, True):                              # switching never stops the miner
        r = http_post(f"{node.url}/mining", {"gpu": gpu, "intensity": 100})
        assert (r["gpu"] is None) == (not gpu or node.gpu is None)
        assert units_in(node, 1.5) > 0
    for bad in (0, 101, "50", True):
        with pytest.raises(urllib.error.HTTPError) as e:
            http_post(f"{node.url}/mining", {"intensity": bad})
        assert e.value.code == 400
    http_post(f"{node.url}/mining", {"on": False})
    time.sleep(1)
    assert units_in(node, 1) <= (node.gpu.batch if node.gpu else 4)
