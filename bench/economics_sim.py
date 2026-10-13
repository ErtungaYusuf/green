"""Green's money in numbers (SPEC.md §3.5): the emission curve, where rewards settle
once the reserve has been handed out, and what shares do for a small miner.

    python bench/economics_sim.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from green import params  # noqa: E402

COIN = params.YAPRAK_PER_COIN
BLOCKS_PER_YEAR = 365 * 86400 // params.TARGET_BLOCK_SECONDS
KEEP = 1 - 1 / params.EMISSION_DIVISOR


def emission_curve():
    print("Emission (no halvings; every coin comes from the fixed reserve):")
    print(f"  {'year':>4} {'reward/block':>13} {'paid out':>12} {'growth of coins in hands':>26}")
    reserve = params.TOTAL_SUPPLY
    for year in range(0, 41):
        out = reserve * (1 - KEEP ** BLOCKS_PER_YEAR)
        held = params.TOTAL_SUPPLY - reserve
        if year in (0, 1, 2, 4, 8, 12, 16, 20, 30, 40):
            growth = f"{out / held * 100:6.2f}%/yr" if held else "(first year)"
            print(f"  {year:>4} {reserve / params.EMISSION_DIVISOR / COIN:>10.4f} GRN {held / COIN / 1e6:>9.2f}M "
                  f"{growth:>26}")
        reserve -= out
    print("  The total supply never changes; what grows is only the part handed out, more slowly each year.")


def steady_state():
    print("\nWhere the block reward settles once the initial reserve is handed out (fees recycled 50%):")
    for fees in (0.01, 0.1, 1.0):
        reserve = fees * COIN * params.FEE_RESERVE_PERCENT / 100 * params.EMISSION_DIVISOR
        reward = reserve / params.EMISSION_DIVISOR
        print(f"  fees {fees:5.2f} GRN/block -> reserve settles at {reserve / COIN / 1e6:6.2f}M GRN, "
              f"emission {reward / COIN:.3f} GRN/block + the finder's {100 - params.FEE_RESERVE_PERCENT}% of fees")


def small_miner(share_of_hashrate: float, days: int = 30, trials: int = 20000, seed: int = 1):
    """Income spread for a miner with `share_of_hashrate` of the network, with and without shares."""
    rng = random.Random(seed)
    blocks = days * 86400 / params.TARGET_BLOCK_SECONDS
    lam_blocks = blocks * share_of_hashrate
    lam_shares = blocks * params.SHARE_FACTOR * share_of_hashrate
    finder_part = (100 - params.SHARES_PERCENT) / 100
    share_part = params.SHARES_PERCENT / 100

    def poisson(lam):
        if lam > 50:
            return max(0, round(rng.gauss(lam, math.sqrt(lam))))
        threshold, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= rng.random()
            if p < threshold:
                return k
            k += 1
    old, new = [], []
    for _ in range(trials):
        b = poisson(lam_blocks)
        s = poisson(lam_shares)
        old.append(b / lam_blocks)                                    # all or nothing: 100% to blocks
        new.append(finder_part * b / lam_blocks + share_part * s / lam_shares)
    def stats(xs):
        mean = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
        nothing = sum(x == 0 for x in xs) / len(xs)
        return sd, nothing
    (sd_old, zero_old), (sd_new, zero_new) = stats(old), stats(new)
    print(f"  {share_of_hashrate:>8.4%} of the network, {days} days: expects {lam_blocks:6.2f} blocks and "
          f"{lam_shares:7.1f} shares; income spread {sd_old * 100:5.1f}% -> {sd_new * 100:5.1f}%, "
          f"months with nothing {zero_old * 100:5.1f}% -> {zero_new * 100:4.1f}%")


if __name__ == "__main__":
    emission_curve()
    steady_state()
    print("\nA small miner's monthly income (spread = standard deviation around what its work deserves):")
    for f in (0.01, 0.001, 0.0001, 0.00001):
        small_miner(f)
