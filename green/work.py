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
     every tile pair (A'_il * B'_lj) is part of the transcript, which is
     digested with a per-element nonlinear mix (transcript_digest).
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
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import params
from .crypto import h

N = params.UNIT
BLK = params.BLOCK
RANK = params.NOISE_RANK
T = N // BLK

assert N % BLK == 0
assert RANK >= BLK
# Tile products are carried in float32 (compute_unit): exact while every partial sum
# of BLK terms of |A'| * |B'| <= (128 + RANK)^2 stays below 2^24. Today 819,200: a 20x
# margin, and parameters leaving less than 8x are refused here. (The whole product, N
# terms, would have under 3x; v0.7 rejected float32 for that.)
assert BLK * (128 + RANK) ** 2 * 8 <= 2 ** 24


def prg(seed: bytes, label: bytes, count: int) -> np.ndarray:
    """Deterministic pseudo-random bytes (SHAKE-256) as a uint8 array."""
    return np.frombuffer(hashlib.shake_256(seed + label).digest(count), dtype=np.uint8)


def _ternary(seed: bytes, label: bytes, shape: tuple[int, int]) -> np.ndarray:
    count = shape[0] * shape[1]
    return ((prg(seed, label, count) % 3).astype(np.float32) - np.float32(1)).reshape(shape)


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


_scratch = threading.local()       # per-thread buffers: large arrays reused, not reallocated per unit


def _buffers() -> dict:
    b = getattr(_scratch, "b", None)
    if b is None:
        b = _scratch.b = {name: np.empty((N, N), np.float32) for name in ("a1", "b1", "e", "p")}
        b["k"] = np.empty(N * N, np.uint32)
        b["t"] = np.empty(N * N, np.uint32)
    return b


def compute_unit(a: np.ndarray, b: np.ndarray, seed: bytes, product: bool = True):
    """Do one work unit.

    Returns (transcript_hash_hex, a @ b), or (transcript_hash_hex, None) with
    product=False: mining filler and checking someone's work need only the
    transcript. The deliverable A @ B comes from recover_useful.

    Every value is an integer carried exactly in float32: the noise sums at most
    RANK terms in {-1, 0, 1}, so |A'|, |B'| <= 128 + RANK = 160, and every partial
    sum of a tile product stays below BLK * 160^2 = 819,200, far under 2^24,
    whatever order the BLAS adds in.
    """
    assert a.shape == (N, N) and b.shape == (N, N)
    u, v, uf, vf = noise_factors(seed)
    buf = _buffers()
    a1, b1, e = buf["a1"], buf["b1"], buf["e"]
    np.add(a, np.matmul(u, v, out=e), out=a1)
    np.add(b, np.matmul(uf, vf, out=e), out=b1)
    transcript = _unit_digest(a1, b1, buf)
    if not product:
        return transcript, None
    return transcript, recover_useful(a, b, seed)


def _unit_digest(a1: np.ndarray, b1: np.ndarray, buf: dict) -> str:
    """transcript_digest of the tile-product stream, without ever holding the stream.

    For each l, P_l = A'[:, l] @ B'[l, :] (an N x BLK column block times a BLK x N
    row block: T BLAS calls instead of T^3 tiny products) holds the products of
    every tile pair (i, l, j), and is digested in place. In P_l's row-major layout
    element (R, C), with R = i*BLK + r and C = j*BLK + c, is stream position
    ((i*T + l)*T + j)*BLK^2 + r*BLK + c = base(R, C) + l*T*BLK^2, and its lane
    (position mod 8) is C mod 8, so P_l's rows of 8 fall on the lanes in order.
    The working set (about 1.5 MB) stays in a core's cache: units checked in
    parallel processes do not fight over memory bandwidth, which materialising
    the 2 MB stream did (measured: 5x the parallel throughput, 1.5x alone).
    """
    p, k, t = buf["p"], buf["k"], buf["t"]
    acc = buf["e"].view(np.uint32).reshape(-1)        # the noise buffer, free again
    acc.fill(0)
    base = _gold_base()
    with np.errstate(over="ignore"):
        for l in range(T):
            np.matmul(a1[:, l * BLK:(l + 1) * BLK], b1[l * BLK:(l + 1) * BLK, :], out=p)
            np.copyto(k.view(np.int32).reshape(N, N), p, casting="unsafe")
            np.add(base, np.uint32(l * T * BLK * BLK * int(_GOLDEN) & 0xFFFFFFFF), out=t)
            np.bitwise_xor(k, t, out=k)
            _lowbias32(k, t)
            np.bitwise_xor(acc, k, out=acc)
    return _lanes_hex(np.bitwise_xor.reduce(acc.reshape(-1, DIGEST_LANES), axis=0))


# The digest mixes each transcript value with its position and passes it through
# lowbias32 (Chris Wellons' 32-bit integer hash: two xorshift-multiply rounds,
# a bijection on u32 with near-ideal avalanche), then XORs it into one of
# DIGEST_LANES lanes. XOR is commutative, so the digest does not depend on the
# order of evaluation: the CPU computes it in a few vectorised passes and the GPU
# with parallel atomics, bit for bit the same (green/gpu.py).
_GOLDEN = np.uint32(0x9E3779B9)
_M1 = np.uint32(0x7FEB352D)
_M2 = np.uint32(0x846CA68B)
DIGEST_LANES = 8
assert BLK % DIGEST_LANES == 0       # _unit_digest: each row of 8 in P_l falls on the lanes in order


def transcript_digest(products_flat_i32: np.ndarray, buffers: dict | None = None) -> str:
    """256-bit commitment to the tile-product stream: lane[i mod 8] ^= mix(v_i, i) with
    mix(v, i) = lowbias32(v XOR i * 0x9E3779B9).

    Position-dependent nonlinear mixing means the digest cannot be produced
    without every product value -- there is no linear shortcut such as reusing
    A@B -- so it binds the work exactly as a plain hash of the products would.
    Computed in cache-sized chunks with in-place operations (tests/test_work.py
    checks it against digest_reference, the direct form).
    """
    assert products_flat_i32.size % DIGEST_LANES == 0
    values = products_flat_i32.astype(np.uint32, copy=False)
    n = values.size
    gold = _golden_index(n)
    lanes = np.zeros(DIGEST_LANES, dtype=np.uint32)
    buf = buffers or _buffers()
    k, t = buf["k"], buf["t"]
    with np.errstate(over="ignore"):
        for start in range(0, n, k.size):
            m = min(k.size, n - start)
            kk, tt = k[:m], t[:m]
            np.bitwise_xor(values[start:start + m], gold[start:start + m], out=kk)
            _lowbias32(kk, tt)
            lanes ^= np.bitwise_xor.reduce(kk.reshape(-1, DIGEST_LANES), axis=0)
    return _lanes_hex(lanes)


def _lowbias32(k: np.ndarray, t: np.ndarray) -> None:
    """k = lowbias32(k) in place (t is scratch of the same size)."""
    for shift, mult in ((16, _M1), (15, _M2), (16, None)):
        np.right_shift(k, np.uint32(shift), out=t)
        np.bitwise_xor(k, t, out=k)
        if mult is not None:
            np.multiply(k, mult, out=k)


def _lanes_hex(lanes: np.ndarray) -> str:
    return lanes.astype("<u4").tobytes().hex()


def _mix(values_u32: np.ndarray, idx_u32: np.ndarray) -> np.ndarray:
    h = values_u32 ^ (idx_u32 * _GOLDEN)
    h ^= h >> np.uint32(16)
    h *= _M1
    h ^= h >> np.uint32(15)
    h *= _M2
    h ^= h >> np.uint32(16)
    return h


def digest_reference(products_flat_i32: np.ndarray) -> str:
    """transcript_digest in its direct form: the definition, kept for tests."""
    with np.errstate(over="ignore"):
        values = products_flat_i32.astype(np.uint32, copy=False)
        idx = np.arange(values.size, dtype=np.uint32)
        mixed = _mix(values, idx).reshape(-1, DIGEST_LANES)
        lanes = np.bitwise_xor.reduce(mixed, axis=0).astype("<u4")
    return lanes.tobytes().hex()


_golden_cache: dict[int, np.ndarray] = {}
_gold_base_cache: list = []


def _gold_base() -> np.ndarray:
    """base(R, C) * GOLDEN over an N x N block, row-major: stream position of
    P_0's element (R, C) (see _unit_digest), times the golden constant."""
    if not _gold_base_cache:
        rows = np.arange(N, dtype=np.uint32)[:, None]
        cols = np.arange(N, dtype=np.uint32)[None, :]
        with np.errstate(over="ignore"):
            pos = ((rows // BLK) * np.uint32(T * T * BLK * BLK) + (cols // BLK) * np.uint32(BLK * BLK)
                   + (rows % BLK) * np.uint32(BLK) + cols % BLK)
            g = (pos * _GOLDEN).reshape(-1)
        g.setflags(write=False)
        _gold_base_cache.append(g)
    return _gold_base_cache[0]


def _golden_index(n: int) -> np.ndarray:
    g = _golden_cache.get(n)
    if g is None:
        with np.errstate(over="ignore"):
            g = np.arange(n, dtype=np.uint32) * _GOLDEN
        g.setflags(write=False)
        _golden_cache[n] = g
    return g


def ticket_value(seed: bytes, transcript_hex: str) -> int:
    return int.from_bytes(h(seed + bytes.fromhex(transcript_hex)), "big")


def recover_useful(a: np.ndarray, b: np.ndarray, seed: bytes = b"") -> np.ndarray:
    """A @ B, the deliverable. With A and B at hand the BLAS computes it directly
    (float64 is exact: every sum stays below 2^23) faster than recovering it from
    the noisy tile products through the low-rank corrections, which give the same
    integers; `seed` is not needed and kept for callers."""
    return blas_matmul(a, b, np.int64)


PARALLEL_MACS = 1 << 25            # products at least this large are split over the cores
_matmul_pool: list = []


def blas_matmul(a: np.ndarray, b: np.ndarray, dtype=np.float64) -> np.ndarray:
    """a @ b through the BLAS in float64 (exact for integers while every sum stays below
    2^53), returned as `dtype`.

    The BLAS runs one thread per call (green/__init__.py: many small products, such as
    work units, are faster that way), so a large product is split into row blocks
    computed on every core in threads, which the BLAS lets run in parallel. Measured
    on 20 cores against a BLAS thread per core: 1024^3 in 10 ms (12), 4096^3 in 1.06 s
    (1.13); one BLAS thread alone took 35 ms and 5.7 s."""
    m, k = a.shape
    n = b.shape[1]
    rows = m >= n                                  # split the longer side of the result
    parts = min(os.cpu_count() or 1, max(m, n) // 64) if m * k * n >= PARALLEL_MACS else 1
    if parts <= 1:
        return (a.astype(np.float64, copy=False) @ b.astype(np.float64, copy=False)).astype(dtype, copy=False)
    if not _matmul_pool:
        _matmul_pool.append(ThreadPoolExecutor(os.cpu_count() or 1, thread_name_prefix="matmul"))
    out = np.empty((m, n), dtype)
    edges = [max(m, n) * i // parts for i in range(parts + 1)]
    af = None if rows else a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False) if rows else None

    def block(i: int) -> None:
        lo, hi = edges[i], edges[i + 1]
        if rows:
            out[lo:hi] = a[lo:hi].astype(np.float64, copy=False) @ bf
        else:
            out[:, lo:hi] = af @ b[:, lo:hi].astype(np.float64, copy=False)
    list(_matmul_pool[0].map(block, range(parts)))
    return out


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
