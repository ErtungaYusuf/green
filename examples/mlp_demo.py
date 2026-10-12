"""Run a quantized MLP forward pass on a running Green node.

    python examples/mlp_demo.py --node http://127.0.0.1:8545 --data data

Start a mining node first (python -m green node --mine). The demo submits each
layer's matmul to the network, applies the activation locally, and checks the
network's result against a local reference.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green import params                                   # noqa: E402
from green.nn import MatmulClient, QuantMLP, predictions   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Quantized MLP inference on Green")
    ap.add_argument("--node", default="http://127.0.0.1:8545")
    ap.add_argument("--data", default="data", help="wallet folder (paying needs coins)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--price", default="0.01", help="GRN per work unit")
    args = ap.parse_args()

    wallet = os.path.join(args.data, "wallet.json")
    if not os.path.exists(wallet):
        sys.exit(f"No wallet: {wallet}. First run: python -m green --data {args.data} wallet new")
    with open(wallet) as f:
        w = json.load(f)
    if "secret" not in w:
        sys.exit("This demo needs an unencrypted wallet (wallet new --no-password).")

    mlp = QuantMLP.random([256, 512, 256, 10], seed=3)
    rng = np.random.default_rng(1)
    x = rng.integers(-127, 128, size=(256, args.samples), dtype=np.int8)

    print(f"Model: 256 -> 512 -> 256 -> 10 (int8), {args.samples} samples")
    print("Computing the local reference...")
    local = mlp.forward_local(x)

    print(f"Running on the network ({args.node}):")
    client = MatmulClient(args.node, w["secret"])
    green = mlp.forward_green(client, x, params.parse_amount(args.price), timeout=180, log=lambda m: print("  " + m))

    ok = np.array_equal(local, green)
    print(f"\nNetwork result bit-identical to the local reference: {'YES' if ok else 'NO'}")
    print(f"Class predictions (network): {predictions(green).tolist()}")
    print(f"Class predictions (local):   {predictions(local).tolist()}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
