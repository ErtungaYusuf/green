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


def test_the_work_unit_matches_its_pinned_vector():
    """Consensus: a fixed filler unit's transcript. If this changes, every block's
    ticket changes, so it must only change on purpose (SPEC.md §5.2)."""
    seed = bytes([7]) * 32
    a, b = work.filler_inputs(seed)
    assert work.compute_unit(a, b, seed, product=False)[0] == "22f044a7ddbf20f469bd9cdec7aad9723b48abc62bbbd2b7e58a6cafbfd87a2e"


def test_the_transcript_alone_skips_the_product_and_matches():
    seed = bytes([9]) * 32
    a, b = work.filler_inputs(seed)
    full, c = work.compute_unit(a, b, seed)
    alone, none = work.compute_unit(a, b, seed, product=False)
    assert full == alone and none is None and c is not None


def test_large_products_split_over_threads_are_exact():
    """blas_matmul splits a large product into row or column blocks computed in threads."""
    rng = np.random.default_rng(5)
    for m, k, n in ((700, 100, 600), (70, 1000, 600), (600, 100, 700), (5, 7, 3)):
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int8)
        assert (m * k * n >= work.PARALLEL_MACS) == (m > 5)
        assert np.array_equal(work.blas_matmul(a, b, np.int64), a.astype(np.int64) @ b.astype(np.int64)), (m, k, n)
