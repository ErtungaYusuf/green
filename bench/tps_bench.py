"""How many transfers a node can check and apply per second, and where the time goes.

    python bench/tps_bench.py [n_txs]

Builds one block of n signed transfers from n different accounts (the worst case:
every transaction touches two fresh accounts), then times the parts of checking it
apart from the work unit: signatures alone, the block's transactions applied as a
node that saw them in its mempool does (signatures already checked), and cold, as a
node seeing them for the first time does.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

from green import params, tx as txmod          # noqa: E402
from green.chain import BlobStore, Chain, make_genesis   # noqa: E402
from green import crypto                        # noqa: E402
from green.crypto import Key, verify            # noqa: E402

params.INITIAL_TARGET = params.MAX_TARGET
from helpers import T0, mine                    # noqa: E402


def main(n: int) -> None:
    params.MAX_BLOCK_TXS = max(params.MAX_BLOCK_TXS, n)
    params.MAX_BLOCK_SIGS = max(params.MAX_BLOCK_SIGS, n)
    params.MAX_BLOCK_BYTES = max(params.MAX_BLOCK_BYTES, 400 * n)
    miner = Key.generate()
    chain = Chain(make_genesis(miner.address, "tps", timestamp=T0), BlobStore())
    senders = [Key.generate() for _ in range(n)]
    for k in senders:
        chain.state.balances[k.address] = 10 * params.YAPRAK_PER_COIN
    chain.states[chain.tip] = chain.state
    t = time.perf_counter()
    txs = [txmod.make_tx(k, chain.chain_id, "transfer", 0, params.MIN_TX_FEE, to=Key.generate().address,
                         amount=params.YAPRAK_PER_COIN) for k in senders]
    sign = time.perf_counter() - t
    size = sum(txmod.tx_size(x) for x in txs) / n

    crypto._good_sigs.clear()
    t = time.perf_counter()
    for x in txs:
        assert verify(x["pub"], txmod.signing_bytes(x, chain.chain_id), x["sig"])
    sigs = time.perf_counter() - t

    block, _ = mine(chain, miner, txs)
    state = chain.state
    best_cold = best_warm = float("inf")
    for _ in range(3):
        crypto._good_sigs.clear()
        t = time.perf_counter()
        chain.apply_block(state, block)
        best_cold = min(best_cold, time.perf_counter() - t)
    for x in txs:
        txmod.check_format(x, chain.chain_id)      # as the mempool does on arrival
    for _ in range(3):
        t = time.perf_counter()
        chain.apply_block(state, block)
        best_warm = min(best_warm, time.perf_counter() - t)
    print(f"{n} transfers, {size:.0f} bytes each; signing {n / sign:,.0f}/s")
    print(f"signatures alone: {n / sigs:,.0f}/s ({sigs / n * 1e6:.0f} us each)")
    print(f"block applied, cold (signatures checked): {n / best_cold:,.0f} tx/s ({best_cold * 1000:.0f} ms)")
    print(f"block applied, warm (seen in the mempool): {n / best_warm:,.0f} tx/s ({best_warm * 1000:.0f} ms)")
    per_block = params.TARGET_BLOCK_SECONDS
    print(f"at {per_block} s blocks, a block of this size is {n / per_block:,.0f} tx/s of capacity")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
