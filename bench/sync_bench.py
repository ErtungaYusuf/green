"""How fast a new node catches up: full sync (every block re-executed, work
units checked in parallel) versus fast sync (a verified snapshot, then only the
blocks after it), over the real P2P protocol on this machine.

    python bench/sync_bench.py [blocks]
"""

import os
import shutil
import socket
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from green import params                     # noqa: E402
from green.chain import make_genesis, solve  # noqa: E402
from green.crypto import Key                 # noqa: E402
from green.node import Node                  # noqa: E402
from green.tx import make_tx                 # noqa: E402

T0 = 1_000_000


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build(chain, keys, blocks):
    """Blocks with a few transfers each, mined at the easiest target."""
    for h in range(1, blocks + 1):
        sender = keys[h % len(keys)] if h > len(keys) else keys[0]
        txs = []
        if chain.state.balance(sender.address) > 10**6:
            nonce = chain.state.nonce(sender.address)
            txs = [make_tx(sender, chain.chain_id, "transfer", nonce + i, to=keys[(h + i + 1) % len(keys)].address,
                           amount=1) for i in range(3)]
        header, chosen, _ = chain.build_template(keys[h % len(keys)].address, txs, now=T0 + 15 * h)
        while True:
            header["transcript"], ticket, _ = solve(header, chain.blobs)
            if ticket <= int(header["target"], 16):
                break
            header["nonce"] += 1
        assert chain.add_block({"header": header, "txs": chosen}) == "added"


def timed_sync(data, genesis, source, fast):
    node = Node(data, genesis, port=free_port(), log=lambda m: None, peers=[source.p2p.address], fast_sync=fast)
    started = time.perf_counter()
    node.start()
    while node.chain.tip != source.chain.tip:
        time.sleep(0.02)
    took = time.perf_counter() - started
    replayed = node.chain.height - node.chain.first_block + 1
    node.stop()
    return took, replayed


def main():
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    params.INITIAL_TARGET = params.MAX_TARGET
    params.ASERT_HALFLIFE = 10**9
    params.SNAPSHOT_INTERVAL = 100
    keys = [Key.generate() for _ in range(8)]
    genesis = make_genesis(keys[0].address, "sync bench", timestamp=T0)
    tmp = tempfile.mkdtemp()
    try:
        source = Node(os.path.join(tmp, "source"), genesis, port=free_port(), log=lambda m: None)
        t = time.perf_counter()
        build(source.chain, keys, blocks)
        print(f"built {blocks} blocks in {time.perf_counter() - t:.1f} s; snapshots at "
              f"{[s['height'] for s in source.snapshots]}")
        source.start()
        for fast in (False, True):
            took, replayed = timed_sync(os.path.join(tmp, f"fresh{int(fast)}"), genesis, source, fast)
            print(f"{'fast sync' if fast else 'full sync'}: {took:.2f} s, {replayed} blocks re-executed, "
                  f"{blocks / took:.0f} blocks/s overall")
        source.stop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
