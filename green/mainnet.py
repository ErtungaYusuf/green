"""The Green main network: its genesis block ships with the software.

A node started without a genesis file joins this network, and only a node on this
network dials params.SEED_NODES (a local or test network never reaches out to them).
"""

import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks", "mainnet.json")


def genesis() -> dict | None:
    """The main network's genesis block, or None before it exists."""
    try:
        with open(PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def genesis_hash() -> str | None:
    from .chain import block_hash
    g = genesis()
    return block_hash(g) if g is not None else None


def is_mainnet(chain_id: str) -> bool:
    return chain_id is not None and chain_id == genesis_hash()
