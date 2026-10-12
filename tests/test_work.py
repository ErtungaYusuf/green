"""The fast transcript digest is bit for bit the definition, and the work unit
it feeds is unchanged."""

import numpy as np

from green import work


def test_fast_digest_equals_the_definition():
    rng = np.random.default_rng(7)
    sizes = [8, 8 * 3, 8 * 1001, (1 << 16) - 8, 1 << 16, (1 << 16) + 8, 3 * (1 << 16) + 24,
             work.T ** 3 * work.BLK * work.BLK]
    for size in sizes:
        x = rng.integers(-2**31, 2**31, size=size, dtype=np.int64).astype(np.int32)
        assert work.transcript_digest(x) == work.digest_reference(x), size
    edge = np.array([0, -1, 2**31 - 1, -2**31] * 2, dtype=np.int32)
    assert work.transcript_digest(edge) == work.digest_reference(edge)


def test_work_unit_transcript_is_unchanged():
    """Recompute a unit's transcript the direct way and compare."""
    for i in range(3):
        seed = bytes([i]) * 32
        a, b = work.filler_inputs(seed)
        transcript, c = work.compute_unit(a, b, seed)
        af, bf = a.astype(np.float64), b.astype(np.float64)
        u, v, uf, vf = work.noise_factors(seed)
        products = np.matmul(work._tiles(af + u @ v)[:, :, None], work._tiles(bf + uf @ vf)[None, :, :])
        assert transcript == work.digest_reference(products.astype(np.int32).reshape(-1))
        assert np.array_equal(c, a.astype(np.int64) @ b.astype(np.int64))
