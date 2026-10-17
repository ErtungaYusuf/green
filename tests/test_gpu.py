"""GPU miner tests. Skipped automatically when there is no GPU / wgpu.

The critical property is that the GPU reproduces the CPU transcript and result
bit for bit: a divergent GPU would mine blocks every other node rejects.
"""

import numpy as np
import pytest

from green import work
from green.gpu import get_miner

miner = get_miner()
pytestmark = pytest.mark.skipif(miner is None, reason="no GPU / wgpu available")


def test_gpu_matches_cpu_bit_for_bit():
    for s in range(12):
        seed = b"gpu-%d" % s
        a, b = work.filler_inputs(seed)
        tr_cpu, c_cpu = work.compute_unit(a, b, seed)
        tr_gpu, ticket, c_gpu = miner.unit(a, b, seed)
        assert tr_gpu == tr_cpu, f"transcript differs at seed {s}"
        assert np.array_equal(c_gpu, c_cpu), f"useful result differs at seed {s}"
        assert ticket == work.ticket_value(seed, tr_gpu)


def test_gpu_matches_cpu_on_real_job_matrices():
    rng = np.random.default_rng(5)
    N = work.N
    a = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
    b = rng.integers(-128, 128, size=(N, N), dtype=np.int8)
    seed = b"job-unit"
    tr_cpu, c_cpu = work.compute_unit(a, b, seed)
    tr_gpu, _, c_gpu = miner.unit(a, b, seed)
    assert tr_gpu == tr_cpu
    assert np.array_equal(c_gpu, c_cpu)
    # The recovered result really is A @ B.
    assert np.array_equal(c_gpu, a.astype(np.int64) @ b.astype(np.int64))


def test_ticket_only_path_matches_full_unit():
    a, b = work.filler_inputs(b"hot")
    tr1, tk1 = miner.ticket(a, b, b"hot")
    tr2, tk2, _ = miner.unit(a, b, b"hot")
    assert tr1 == tr2 and tk1 == tk2


def test_batches_in_both_slots_match_the_cpu_try_by_try():
    """Full and partial batches, two outstanding at once (the miner's pipeline): every
    try's transcript and ticket are the CPU's, in order."""
    def tries(tag, count):
        out = []
        for i in range(count):
            seed = b"%s-%d" % (tag, i)
            a, b = work.filler_inputs(seed)
            out.append((a, b, seed))
        return out
    first, second, third = tries(b"one", miner.batch), tries(b"two", 3), tries(b"three", miner.batch)
    h1 = miner.submit(first)
    h2 = miner.submit(second)
    with pytest.raises(RuntimeError):
        miner.submit(third)                                   # both slots busy
    got = miner.collect(h1) + miner.collect(h2)
    got += miner.collect(miner.submit(third))
    for (a, b, seed), (transcript, ticket) in zip(first + second + third, got):
        expect = work.compute_unit(a, b, seed, product=False)[0]
        assert transcript == expect and ticket == work.ticket_value(seed, expect), seed
    assert len(got) == 2 * miner.batch + 3
