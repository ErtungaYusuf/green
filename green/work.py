"""Proof of Useful Work built on integer matrix multiplication.

Why matrix multiplication: it is the bulk of the FLOPs in both training and
inference of neural networks, its result can be checked in O(n^2) with
Freivalds' algorithm, and with integer inputs it is bit-for-bit deterministic
on any hardware.

How one work unit becomes one lottery ticket (idea follows Komargodski, Schen
and Weinstein, "Proofs of Useful Work from Arbitrary Matrix Multiplication"):

  1. seed = hash(block header without transcript). It binds the ticket to the
     previous block, the transactions, the miner's address and a nonce.
  2. Low-rank noise E = U V and F = U' V' is derived from the seed.
  3. The miner multiplies A' = A + E by B' = B + F tile by tile. The product of
     every tile pair (A'_il * B'_lj) is part of the transcript.
  4. ticket = hash(seed || hash(transcript)); the block is valid when
     ticket < target, exactly like Bitcoin's hash < target.
  5. The useful result is recovered cheaply:
        A B = A'B' - A F - E B'      (both corrections are rank-r, O(r n^2))

Because the noise changes with every seed, a transcript can never be reused,
and because rank(noise) >= tile size, already knowing A*B does not make the
transcript cheaper. Every ticket therefore costs one full matmul whether the
matrices came from a paying customer or are random filler: buying your own
jobs gives no mining advantage.

Delivered results are checked with Freivalds' algorithm over a prime field
(freivalds_check), about 25x cheaper than recomputing the unit.
"""

import hashlib

import numpy as np

from . import params
from .crypto import h

N = params.UNIT
BLK = params.BLOCK
RANK = params.NOISE_RANK
T = N // BLK

assert N % BLK == 0
assert RANK >= BLK


def prg(seed: bytes, label: bytes, count: int) -> np.ndarray:
    """Deterministic pseudo-random bytes (SHAKE-256) as a uint8 array."""
    return np.frombuffer(hashlib.shake_256(seed + label).digest(count), dtype=np.uint8)


def _ternary(seed: bytes, label: bytes, shape: tuple[int, int]) -> np.ndarray:
    count = shape[0] * shape[1]
    return ((prg(seed, label, count) % 3).astype(np.float64) - 1.0).reshape(shape)


def noise_factors(seed: bytes):
    """Low-rank noise factors with entries in {-1, 0, 1}."""
    u = _ternary(seed, b"U", (N, RANK))
    v = _ternary(seed, b"V", (RANK, N))
    uf = _ternary(seed, b"Uf", (N, RANK))
    vf = _ternary(seed, b"Vf", (RANK, N))
    return u, v, uf, vf


def filler_inputs(seed: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Random int8 matrices used when nobody is buying compute."""
    a = prg(seed, b"fillA", N * N).view(np.int8).reshape(N, N)
    b = prg(seed, b"fillB", N * N).view(np.int8).reshape(N, N)
    return a, b


def _tiles(m: np.ndarray) -> np.ndarray:
    # (N, N) -> (T, T, BLK, BLK) where [r, c] is the tile at row r, column c.
    return m.reshape(T, BLK, T, BLK).transpose(0, 2, 1, 3)


def compute_unit(a: np.ndarray, b: np.ndarray, seed: bytes) -> tuple[str, np.ndarray]:
    """Do one work unit.

    Returns (transcript_hash_hex, a @ b). Integer values are carried in
    float64, which is exact while every partial sum stays below 2^53 (here it
    stays below 2^31), so any BLAS / GPU summation order gives the same bits.
    """
    assert a.shape == (N, N) and b.shape == (N, N)
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    u, v, uf, vf = noise_factors(seed)
    a1 = af + u @ v
    b1 = bf + uf @ vf

    # products[i, l, j] = A'_il @ B'_lj -- the transcript.
    products = np.matmul(_tiles(a1)[:, :, None], _tiles(b1)[None, :, :])
    transcript = transcript_digest(products.astype(np.int32).reshape(-1))

    c1 = products.sum(axis=1).transpose(0, 2, 1, 3).reshape(N, N)
    c = c1 - (af @ uf) @ vf - u @ (v @ b1)
    return transcript, np.rint(c).astype(np.int64)


# MurmurHash3-style u32 mixers. Each multiplier is odd, so the per-element mix
# is a bijection on u32; the lanes are combined with XOR, which is commutative,
# so the digest is independent of evaluation order -- the CPU computes it in one
# vectorised pass and the GPU with parallel atomics, bit for bit the same.
_C1 = np.uint32(0xCC9E2D51)
_C2 = np.uint32(0x1B873593)
_C3 = np.uint32(0xE6546B64)
_GOLDEN = np.uint32(0x9E3779B9)
_F1 = np.uint32(0x85EBCA6B)
_F2 = np.uint32(0xC2B2AE35)
DIGEST_LANES = 8


def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def _mix(values_u32: np.ndarray, idx_u32: np.ndarray) -> np.ndarray:
    k = values_u32 * _C1
    k = _rotl(k, 15) * _C2
    hh = (idx_u32 * _GOLDEN) ^ k
    hh = _rotl(hh, 13) * np.uint32(5) + _C3
    hh ^= hh >> np.uint32(16)
    hh *= _F1
    hh ^= hh >> np.uint32(13)
    hh *= _F2
    hh ^= hh >> np.uint32(16)
    return hh


def transcript_digest(products_flat_i32: np.ndarray) -> str:
    """256-bit commitment to the tile-product stream (see gpu.py for the match).

    Position-dependent nonlinear mixing means the digest cannot be produced
    without every product value -- there is no linear shortcut such as reusing
    A@B -- so it binds the work exactly as a plain hash of the products would,
    while staying cheap enough to compute on the mining device.

    This is _mix (the definition) computed in cache-sized chunks with in-place
    operations, 3.6x faster than the direct form and bit for bit the same
    (tests/test_work.py checks it against digest_reference).
    """
    assert products_flat_i32.size % DIGEST_LANES == 0
    values = products_flat_i32.astype(np.uint32, copy=False)
    n = values.size
    gold = _golden_index(n)
    lanes = np.zeros(DIGEST_LANES, dtype=np.uint32)
    k = np.empty(min(n, _DIGEST_CHUNK), np.uint32)
    t = np.empty_like(k)
    with np.errstate(over="ignore"):
        for start in range(0, n, _DIGEST_CHUNK):
            m = min(_DIGEST_CHUNK, n - start)
            kk, tt = k[:m], t[:m]
            np.multiply(values[start:start + m], _C1, out=kk)
            _rotl_inplace(kk, tt, 15)
            np.multiply(kk, _C2, out=kk)
            np.bitwise_xor(kk, gold[start:start + m], out=kk)
            _rotl_inplace(kk, tt, 13)
            np.multiply(kk, np.uint32(5), out=kk)
            np.add(kk, _C3, out=kk)
            for shift, mult in ((16, _F1), (13, _F2), (16, None)):
                np.right_shift(kk, np.uint32(shift), out=tt)
                np.bitwise_xor(kk, tt, out=kk)
                if mult is not None:
                    np.multiply(kk, mult, out=kk)
            # XOR the chunk's rows together (lane = index % DIGEST_LANES); XOR is
            # associative and commutative, so folding halves gives the same lanes.
            rows = kk.reshape(-1, DIGEST_LANES)
            while rows.shape[0] > 1:
                if rows.shape[0] % 2:
                    lanes ^= rows[-1]
                    rows = rows[:-1]
                half = rows.shape[0] // 2
                rows = rows[:half] ^ rows[half:]
            lanes ^= rows[0]
    return lanes.astype("<u4").tobytes().hex()


def digest_reference(products_flat_i32: np.ndarray) -> str:
    """transcript_digest in its direct form: the definition, kept for tests."""
    with np.errstate(over="ignore"):
        values = products_flat_i32.astype(np.uint32, copy=False)
        idx = np.arange(values.size, dtype=np.uint32)
        mixed = _mix(values, idx).reshape(-1, DIGEST_LANES)
        lanes = np.bitwise_xor.reduce(mixed, axis=0).astype("<u4")
    return lanes.tobytes().hex()


_DIGEST_CHUNK = 1 << 16          # 256 KiB per temporary: stays in L2 cache
_golden_cache: dict[int, np.ndarray] = {}


def _golden_index(n: int) -> np.ndarray:
    g = _golden_cache.get(n)
    if g is None:
        with np.errstate(over="ignore"):
            g = np.arange(n, dtype=np.uint32) * _GOLDEN
        g.setflags(write=False)
        _golden_cache[n] = g
    return g


def _rotl_inplace(x: np.ndarray, tmp: np.ndarray, r: int) -> None:
    np.left_shift(x, np.uint32(r), out=tmp)
    np.right_shift(x, np.uint32(32 - r), out=x)
    np.bitwise_or(x, tmp, out=x)


def ticket_value(seed: bytes, transcript_hex: str) -> int:
    return int.from_bytes(h(seed + bytes.fromhex(transcript_hex)), "big")


def recover_useful(a: np.ndarray, b: np.ndarray, seed: bytes) -> np.ndarray:
    """A @ B via the cheap low-rank correction, without the transcript.

    Identical to the result compute_unit returns, but skips hashing -- used by
    a miner that already found a ticket and now needs the deliverable value.
    """
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    u, v, uf, vf = noise_factors(seed)
    a1 = af + u @ v
    b1 = bf + uf @ vf
    c = a1 @ b1 - (af @ uf) @ vf - u @ (v @ b1)
    return np.rint(c).astype(np.int64)


def freivalds_check(a: np.ndarray, b: np.ndarray, c: np.ndarray, seed: bytes,
                    rounds: int = params.FREIVALDS_ROUNDS) -> bool:
    """Probabilistically check c == a @ b in O(n^2 * rounds).

    Works modulo the prime p = FREIVALDS_PRIME with random vectors drawn from
    all of Z_p rather than {0, 1}: a wrong c survives one round with
    probability <= 1/p instead of 1/2, so 4 rounds give 2^-124 where the {0, 1}
    version needed 64 rounds for 2^-64 -- and cost ~70% of recomputing a @ b.

    Reducing mod p is only sound if a wrong c stays wrong mod p, i.e. no entry
    of a @ b - c can be a nonzero multiple of p. So c is first checked against
    the largest value a @ b can hold (k * max|a| * max|b|); a cheater who
    shifted an entry by p is rejected there.

    The random vectors come from `seed`, which must commit to c so a cheater
    cannot pick c after seeing them (or be private to the verifier).
    """
    n, k = a.shape
    k2, m = b.shape
    if k != k2 or c.shape != (n, m):
        return False
    p = params.FREIVALDS_PRIME
    a64, b64, c64 = a.astype(np.int64), b.astype(np.int64), c.astype(np.int64)
    max_a = int(np.abs(a64).max(initial=0))
    max_b = int(np.abs(b64).max(initial=0))
    bound = k * max_a * max_b
    # Exactness: |a@b - c| < p, and no int64 product sum below can overflow.
    if 2 * bound >= p or k * max(max_a, max_b) * p >= 2**63 or m * bound * p >= 2**63:
        raise ValueError("matrices too large for an exact modular Freivalds check")
    if int(np.abs(c64).max(initial=0)) > bound:
        return False
    raw = prg(seed, b"freivalds", 8 * m * rounds).view("<u8").reshape(m, rounds)
    r = (raw % np.uint64(p)).astype(np.int64)
    lhs = (a64 @ ((b64 @ r) % p)) % p
    rhs = (c64 @ r) % p
    return bool(np.array_equal(lhs, rhs))
