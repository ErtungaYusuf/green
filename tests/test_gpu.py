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
