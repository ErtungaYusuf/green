"""The quantized MLP forward pass runs on a live Green node, bit-exact."""

import socket

import numpy as np
import pytest

from green import params
from green.chain import make_genesis
from green.crypto import Key
from green.nn import MatmulClient, QuantMLP, predictions
from green.node import Node


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mining_node(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)     # instant blocks
    monkeypatch.setattr(params, "TARGET_BLOCK_SECONDS", 1)
    key = Key.generate()
    node = Node(str(tmp_path / "n"), make_genesis(key.address, "nn"), port=free_port(),
                miner_key=key, log=lambda m: None)
    node.start(mine=True)
    # let the miner accumulate coins to pay job escrow
    import time
    end = time.time() + 30
    while node.chain.state.balance(key.address) < 5 * params.YAPRAK_PER_COIN and time.time() < end:
        time.sleep(0.2)
    yield node, key
    node.stop()


def test_mlp_forward_runs_on_green_bit_exact(mining_node):
    node, key = mining_node
    mlp = QuantMLP.random([256, 512, 256, 10], seed=3)
    rng = np.random.default_rng(1)
    x = rng.integers(-127, 128, size=(256, 8), dtype=np.int8)      # 8 samples

    local = mlp.forward_local(x)
    client = MatmulClient(node.url, key.secret_hex())
    green = mlp.forward_green(client, x, price=params.YAPRAK_PER_COIN // 100, timeout=90)

    assert np.array_equal(local, green), "Green forward pass diverged from the reference"
    assert green.shape == (10, 8)
    assert np.array_equal(predictions(local), predictions(green))
    # calibration should keep the classifier non-degenerate (not every sample one class)
    assert len(set(predictions(green).tolist())) > 1


def test_single_layer_matmul_matches_numpy(mining_node):
    node, key = mining_node
    client = MatmulClient(node.url, key.secret_hex())
    rng = np.random.default_rng(2)
    a = rng.integers(-127, 128, size=(300, 400), dtype=np.int8)     # not a multiple of 256
    b = rng.integers(-127, 128, size=(400, 120), dtype=np.int8)
    c = client.matmul(a, b, price=params.YAPRAK_PER_COIN // 100, timeout=90)
    assert np.array_equal(c, a.astype(np.int64) @ b.astype(np.int64))
