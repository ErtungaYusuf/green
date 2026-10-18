"""Compute that hides the data from the miners who do it.

Two ways, chosen per request:

split   The rows of A and the columns of B are shuffled with a secret permutation and
        the product is cut into a grid of tiles spread over the providers, so no
        provider sees the whole problem or the order of what it sees. Costs the same
        as a plain product. It hides structure, not values: a provider still sees the
        numbers in its tiles.

secret  Replicated secret sharing over three providers. For each of two primes p
        below 2^20, A = A1 + A2 + A3 and B = B1 + B2 + B3 (mod p) with A1, A2, B1, B2
        uniformly random. Provider i gets only [A_i | A_i+1] and [B_i + B_i+1 ; B_i],
        and returns their product mod p, which is A_i B_i + A_i B_i+1 + A_i+1 B_i; the
        three answers add up to A B mod p, and the two primes give A B exactly by the
        Chinese remainder theorem. Any two of three shares are uniformly random, so a
        provider learns nothing about A or B -- unless two of the three providers
        pool what they saw. Costs twelve products' worth of compute (two primes,
        three providers, inner dimension doubled).

Either way every answer is checked (Freivalds' test, over the integers or mod p)
before it is used or paid for, and with a relayed cluster (ComputeCluster.from_market
relay=True) providers do not learn who is asking either.
"""

import math
import secrets

import numpy as np

from . import sealed
from .compute import ComputeChannel, ComputeCluster, ProviderError, WrongResult

PRIMES = (1048573, 1048571)        # the two largest primes below 2^20; their product exceeds 2 max|A B|
TILE_BYTES = sealed.MAX_SEALED - 64 * 1024     # one request or answer, with room for the sealing
MODES = ("split", "secret")


class PrivateCompute:
    def __init__(self, cluster: ComputeCluster):
        self.cluster = cluster

    def matmul(self, a: np.ndarray, b: np.ndarray, mode: str = "split") -> np.ndarray:
        """a @ b for int8 matrices (int64 result), with the data hidden as `mode` says."""
        a = np.ascontiguousarray(a, dtype=np.int8)
        b = np.ascontiguousarray(b, dtype=np.int8)
        if a.shape[1] != b.shape[0]:
            raise ValueError("inner dimensions differ")
        if mode == "split":
            return self._split(a, b)
        if mode == "secret":
            return self._secret(a, b)
        raise ValueError(f"mode must be one of {MODES}")

    # -- split --

    def _split(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        (m, k), n = a.shape, b.shape[1]
        rng = np.random.default_rng(secrets.randbits(128))
        rows, cols = rng.permutation(m), rng.permutation(n)
        ap, bp = a[rows], b[:, cols]                  # (A B)[rows][:, cols] = ap @ bp
        live = self.cluster.live()
        side = math.ceil(math.sqrt(len(live)))        # at least as many tiles as providers
        gr, gc = min(m, side), min(n, side)
        while True:                                   # ...and each request and answer fits a sealed message
            tr, tc = -(-m // gr), -(-n // gc)
            if tr * k + k * tc <= TILE_BYTES and 4 * tr * tc <= TILE_BYTES:
                break
            if (tr >= tc or gc == n) and gr < m:
                gr += 1
            elif gc < n:
                gc += 1
            else:
                raise ValueError("inner dimension too large for one request")
        tiles = [(r, c) for r in range(gr) for c in range(gc)]
        start = int(rng.integers(len(live)))
        out = np.empty((m, n), dtype=np.int64)
        futures = []
        for t, (r, c) in enumerate(tiles):
            r0, r1, c0, c1 = r * tr, min(m, (r + 1) * tr), c * tc, min(n, (c + 1) * tc)
            if r0 >= r1 or c0 >= c1:
                continue
            ch = live[(start + t) % len(live)]
            futures.append((r0, r1, c0, c1, self.cluster.pool.submit(
                self.cluster._one, ap[r0:r1], bp[:, c0:c1], ch)))
        for r0, r1, c0, c1, f in futures:
            out[r0:r1, c0:c1] = f.result()
        result = np.empty_like(out)
        result[np.ix_(rows, cols)] = out
        return result

    # -- secret --

    def _secret(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        live = self.cluster.live()
        if len(live) < 3:
            raise ProviderError("secret sharing needs three providers that do not pool what they see")
        (m, k), n = a.shape, b.shape[1]
        residues = []
        for p in PRIMES:
            acc = np.zeros((m, n), dtype=np.int64)
            kc = max(1, min(2048, min(ch.offer.get("max_dim", 4096) for ch in live) // 2))
            for k0 in range(0, k, kc):
                for attempt in (1, 2):                # a provider that cheats is dropped; fresh shares, once more
                    try:
                        acc = (acc + self._shared_product(a[:, k0:k0 + kc], b[k0:k0 + kc], p)) % p
                        break
                    except (WrongResult, ProviderError):
                        if attempt == 2 or len(self.cluster.live()) < 3:
                            raise
            residues.append(acc)
        (p1, p2), (r1, r2) = PRIMES, residues
        inv = pow(p1, -1, p2)
        c = r1 + p1 * (((r2 - r1) % p2) * inv % p2)  # the residue mod p1 p2 (< 2^40, fits int64)
        mod = p1 * p2
        return np.where(c > mod // 2, c - mod, c)

    def _shared_product(self, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
        """(a @ b) mod p from three providers, each seeing two random shares of a and b."""
        rng = np.random.default_rng(secrets.randbits(128))
        A, B = a.astype(np.int64) % p, b.astype(np.int64) % p
        As = [rng.integers(0, p, A.shape), rng.integers(0, p, A.shape)]
        As.append((A - As[0] - As[1]) % p)
        Bs = [rng.integers(0, p, B.shape), rng.integers(0, p, B.shape)]
        Bs.append((B - Bs[0] - Bs[1]) % p)
        parties = [self.cluster.live()[i] for i in rng.permutation(len(self.cluster.live()))[:3]]
        futures = []
        for i, ch in enumerate(parties):              # party i's whole share goes to one provider only
            x = np.hstack([As[i], As[(i + 1) % 3]])
            y = np.vstack([(Bs[i] + Bs[(i + 1) % 3]) % p, Bs[i]])
            futures.append(self.cluster.pool.submit(_tiled_mod, ch, x, y, p))
        total = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
        for f in futures:
            total = (total + f.result()) % p
        return total


def _tiled_mod(ch: ComputeChannel, x: np.ndarray, y: np.ndarray, p: int) -> np.ndarray:
    """(x @ y) mod p from one provider, in requests that each fit a sealed message."""
    (m, k), n = x.shape, y.shape[1]
    tc = n
    while 4 * (k * tc) > TILE_BYTES // 2 and tc > 1:
        tc = -(-tc // 2)
    tr = max(1, min(m, (TILE_BYTES - 4 * k * tc) // (4 * (k + tc))))
    out = np.empty((m, n), dtype=np.int64)
    for r0 in range(0, m, tr):
        for c0 in range(0, n, tc):
            out[r0:r0 + tr, c0:c0 + tc] = ch.matmul_mod(x[r0:r0 + tr], y[:, c0:c0 + tc], p)
    return out
