"""How the ledger scales: block cost vs. number of accounts, memory, restart time.

    python bench/state_bench.py

Accounts are injected straight into the state (signing a million transfers
would only measure Ed25519); blocks are real: built, solved and validated.
"""

import os
import shutil
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

from green import params                       # noqa: E402
from green.chain import BlobStore, Chain, make_genesis   # noqa: E402
from green.crypto import Key, bech32m_encode   # noqa: E402
from green.node import Node                    # noqa: E402

params.INITIAL_TARGET = params.MAX_TARGET      # every ticket wins: measure the ledger, not the lottery
from helpers import T0, mine, tx               # noqa: E402


def fake_address(i: int) -> str:
    return bech32m_encode("grn", i.to_bytes(20, "big"))


def block_cost(n_accounts: int, blocks: int = 20) -> tuple[float, float, float]:
    """(ms per block, ms of it spent on the ledger, MB held by the chain) with n_accounts in the ledger.

    Most of a block's time is re-computing its winning work unit (fixed numpy
    work); the ledger part is timed separately.
    """
    key, other = Key.generate(), Key.generate()
    chain = Chain(make_genesis(key.address, "bench", timestamp=T0), BlobStore())
    chain.state.balances.update({fake_address(i): 1 for i in range(n_accounts)})
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    took, ledger = [], []
    for _ in range(blocks):
        t = tx(chain, key, "transfer", to=other.address, amount=1)
        block, _ = mine(chain, key, [t])
        start = time.perf_counter()
        chain.apply_block(chain.state, block)
        ledger.append(time.perf_counter() - start)
        start = time.perf_counter()
        assert chain.add_block(block) == "added"
        took.append(time.perf_counter() - start)
    held = (tracemalloc.get_traced_memory()[0] - base) / 2**20
    tracemalloc.stop()
    took.sort()
    ledger.sort()
    return took[len(took) // 2] * 1000, ledger[len(ledger) // 2] * 1000, held


def restart_cost(height: int) -> float:
    """Seconds for a node to open a chain of `height` blocks from disk."""
    d = tempfile.mkdtemp(prefix="green-bench-")
    try:
        key, other = Key.generate(), Key.generate()
        genesis = make_genesis(key.address, "bench", timestamp=T0)
        node = Node(d, genesis, port=0, miner_key=key, log=lambda m: None)
        for _ in range(height):
            block, _ = mine(node.chain, key, [tx(node.chain, key, "transfer", to=other.address, amount=1)])
            node._add_block(block)
        node.close()
        start = time.perf_counter()
        reopened = Node(d, genesis, port=0, miner_key=key, log=lambda m: None)
        took = time.perf_counter() - start
        assert reopened.chain.height == height
        reopened.close()
        return took
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    print("accounts     ms/block   ledger ms   MB held by 20 blocks")
    for n in (0, 10_000, 100_000, 1_000_000):
        ms, ledger_ms, mb = block_cost(n)
        print(f"{n:>9,}  {ms:>9.1f}  {ledger_ms:>9.2f}  {mb:>10.1f}")
    for h in (100, 400):
        print(f"restart with {h} blocks: {restart_cost(h):.2f} s")
