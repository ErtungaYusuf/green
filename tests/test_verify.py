"""Checking work units in worker processes during a long sync (green/verify.py)."""

import os
import random
from types import SimpleNamespace

from green import p2p, params, work
from green.chain import block_hash, make_genesis, seed_of, share_id
from green.crypto import Key
from green.node import Node
from green.verify import ProcessChecker
from helpers import T0, add, grind, new_chain
from test_p2p import QUIET, free_port, start_node, wait_for


def share_for(chain, miner):
    """A header mined on the tip that no chain contains (with the easy test target it
    meets any share target, like an orphaned block)."""
    header, chosen, _ = chain.build_template(miner.address, [], now=T0 + 10 * (chain.height + 1) + 3)
    header["nonce"] = random.getrandbits(40)
    return grind(chain, header, chosen)[0]["header"]


def add_with_shares(chain, finder, miners):
    shares = [share_for(chain, m) for m in miners]
    header, chosen, _ = chain.build_template(finder.address, [], now=T0 + 10 * (chain.height + 1), shares=shares)
    block, _ = grind(chain, header, chosen)
    block["shares"] = shares
    assert chain.add_block(block) == "added"
    return block


def test_workers_recompute_transcripts_with_a_single_threaded_blas(monkeypatch):
    key = Key.generate()
    chain = new_chain(key)
    for _ in range(3):
        add(chain, key)
    headers = [chain.block(bh)["header"] for bh in chain.main[1:]]
    forged = dict(headers[0], transcript="00" * 32)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    checker = ProcessChecker(None, workers=2)
    try:
        found = checker.submit(headers + [forged, {"height": 1}]).result(timeout=120)
        assert found[:3] == [hd["transcript"] for hd in headers]
        assert found[3] == headers[0]["transcript"] != forged["transcript"]   # the work's, not the claim
        assert found[4] is None                                               # malformed: nothing
        assert checker.pool.submit(os.getenv, "OPENBLAS_NUM_THREADS").result(timeout=60) == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "3"                      # ours untouched
    finally:
        checker.close()


def test_only_matching_transcripts_count_as_checked(monkeypatch):
    monkeypatch.setattr(params, "SHARES_PERCENT", 80)
    finder, other = Key.generate(), Key.generate()
    chain = new_chain(finder)
    add(chain, finder)
    block = add_with_shares(chain, finder, [other, other])
    sh = block["shares"]
    host = SimpleNamespace(chain=chain, stats={"checked_in_processes": 0})
    chain.prechecked.clear()
    chain.verified_shares.clear()
    p2p.P2P._apply_transcripts(host, block, ["11" * 32, sh[0]["transcript"], "22" * 32])
    assert block_hash(block) not in chain.prechecked and host.stats["checked_in_processes"] == 0
    assert chain.verified_shares == {share_id(sh[0])}
    p2p.P2P._apply_transcripts(host, block, [block["header"]["transcript"], None, sh[1]["transcript"]])
    assert block_hash(block) in chain.prechecked and host.stats["checked_in_processes"] == 1
    assert chain.verified_shares == {share_id(sh[0]), share_id(sh[1])}
    # A share whose transcript is its own but misses the share target is not counted.
    monkeypatch.setattr(chain, "share_target", lambda share: 0)
    chain.verified_shares.clear()
    p2p.P2P._apply_transcripts(host, block, [None, sh[0]["transcript"], sh[1]["transcript"]])
    assert not chain.verified_shares


def test_a_long_sync_checks_blocks_and_shares_in_processes(tmp_path, monkeypatch):
    """With a long download queued, the blocks' and their shares' work units are
    recomputed in worker processes, none in the node's own, and the node ends up on
    the same chain and state."""
    monkeypatch.setattr(params, "SHARES_PERCENT", 80)
    monkeypatch.setattr(p2p, "PROCESS_BACKLOG", 0)          # every download counts as long
    keys = [Key.generate() for _ in range(3)]
    genesis = make_genesis(keys[0].address, "process sync", timestamp=T0)
    source = Node(str(tmp_path / "source"), genesis, port=free_port(), log=QUIET)
    for i in range(12):
        add_with_shares(source.chain, keys[i % 3], [keys[(i + 1) % 3], keys[(i + 2) % 3]])
    share_seeds = {seed_of(sh) for bh in source.chain.main[1:] for sh in source.chain.block(bh).get("shares", [])}
    assert len(share_seeds) == 24
    computed = set()
    real = work.compute_unit

    def counting(a, b, seed, product=True):
        computed.add(seed)
        return real(a, b, seed, product)
    monkeypatch.setattr(work, "compute_unit", counting)
    source.start()
    fresh = None
    try:
        fresh = start_node(tmp_path / "fresh", genesis, peers=[source.p2p.address])
        wait_for(lambda: fresh.chain.tip == source.chain.tip, timeout=120, what="sync through worker processes")
        assert fresh.p2p.stats["checked_in_processes"] == 12
        assert not computed & share_seeds                    # no share was recomputed here
        assert fresh.chain.state.root() == source.chain.state.root()
    finally:
        if fresh is not None:
            fresh.stop()
        source.stop()


def _alive(pid: int) -> bool:
    if os.name == "nt":                          # os.kill(pid, 0) would terminate it on Windows
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                 # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_workers_die_with_a_killed_node():
    """A node killed without shutting down left its 8 workers running for hours."""
    import subprocess
    import sys
    import time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = ("import sys, time; sys.path.insert(0, sys.argv[1]); from green.verify import start_pool; "
              "pool = start_pool(2); time.sleep(1); print(*pool._processes, flush=True); time.sleep(120)")
    proc = subprocess.Popen([sys.executable, "-c", script, root], stdout=subprocess.PIPE, text=True)
    try:
        pids = [int(x) for x in proc.stdout.readline().split()]
        assert len(pids) == 2 and all(_alive(p) for p in pids)
    finally:
        proc.kill()                              # no shutdown, as when a node is killed
        proc.wait()
    end = time.time() + 30
    while any(_alive(p) for p in pids):
        assert time.time() < end, "the workers outlived their node"
        time.sleep(0.2)
