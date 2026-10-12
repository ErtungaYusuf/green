"""Difficulty algorithms under hashrate shocks: the v0.5 windowed retarget vs ASERT.

    python bench/difficulty_sim.py

Honest miners, exponential block times (a lottery), the same random draws for
both algorithms. Hashrate: steady, then x10 (big GPUs arrive), then /100
(they leave all at once, the case that stalls chains).
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MAX_TARGET = 2**256 - 1
T = 15                      # target seconds per block


def legacy_target(chain, interval=30, factor=4):
    """v0.5: retarget every 30 blocks over the last 30 intervals, at most 4x."""
    height = len(chain)
    old = chain[-1][1]
    if height % interval:
        return old
    first = chain[max(0, height - interval - 1)]
    intervals = (height - 1) - max(0, height - interval - 1)
    expected = intervals * T
    actual = min(max(max(1, chain[-1][0] - first[0]), expected // factor), expected * factor)
    return max(1, min(MAX_TARGET, old * actual // expected))


def asert_target(chain, halflife, anchor_target):
    """aserti3-2d (Bitcoin Cash, 2020) with the genesis block as anchor."""
    from green.chain import asert
    parent_time, _ = chain[-1]
    return asert(anchor_target, chain[0][0] - T, 0, parent_time, len(chain) - 1, halflife)


def simulate(next_target, phases, seed=7):
    rng = random.Random(seed)
    start_target = MAX_TARGET // (phases[0][1] * T)        # calibrated to the first hashrate
    chain = [(0, start_target)]                            # (time, target)
    out = []
    for blocks, hashrate in phases:
        times = []
        for _ in range(blocks):
            target = next_target(chain)
            tries = MAX_TARGET / target
            dt = rng.expovariate(1.0) * tries / hashrate
            t = chain[-1][0] + max(1, round(dt))
            chain.append((t, target))
            times.append(t - chain[-2][0])
        out.append(times)
    return out


def simulate_switching(next_target, blocks=6000, base=100, switchers=500, seed=11):
    """Rational miners: `switchers` units/s join only while the difficulty is at
    least 10% below what the always-on miners alone would justify, and leave
    when it is not. Window algorithms are known to oscillate under this."""
    rng = random.Random(seed)
    fair = MAX_TARGET // (base * T)
    chain = [(0, fair)]
    times = []
    for _ in range(blocks):
        target = next_target(chain)
        hashrate = base + (switchers if target >= fair * 1.1 else 0)
        dt = rng.expovariate(1.0) * (MAX_TARGET / target) / hashrate
        chain.append((chain[-1][0] + max(1, round(dt)), target))
        times.append(chain[-1][0] - chain[-2][0])
    return times


def report_switching(name, times, window=60):
    avgs = [sum(times[i:i + window]) / window for i in range(0, len(times) - window, window)]
    worst = max(abs(a - T) for a in avgs)
    drift = sum(times) - T * len(times)
    print(f"  {name:34} mean {sum(times) / len(times):5.1f} s  worst 15-min average off by {worst:5.1f} s  "
          f"schedule drift {drift / 3600:+6.1f} h over {len(times)} blocks")


def recovery(times, window=60, tolerance=0.2):
    """Blocks until the moving average of block times stays within 20% of T."""
    for i in range(len(times) - window):
        avg = sum(times[i:i + window]) / window
        if abs(avg - T) / T <= tolerance:
            return i
    return None


def report(name, result):
    labels = ["steady", "x10 hashrate", "/100 hashrate"]
    print(f"\n{name}")
    for label, times in zip(labels, result):
        mean = sum(times) / len(times)
        hours_to_recover = None
        r = recovery(times)
        if r is not None:
            hours_to_recover = sum(times[:r]) / 3600
        tail = times[len(times) // 2:]
        sd = math.sqrt(sum((x - T) ** 2 for x in tail) / len(tail))
        rec = f"{hours_to_recover:.2f} h ({r} blocks)" if r is not None else "never"
        print(f"  {label:14} mean {mean:7.1f} s  longest gap {max(times) / 60:6.1f} min  "
              f"back within 20%: {rec:20}  spread in 2nd half {sd:5.1f} s")


if __name__ == "__main__":
    phases = [(2000, 100), (3000, 1000), (3000, 10)]
    report("legacy: every 30 blocks, max 4x", simulate(legacy_target, phases))
    for halflife in (1800, 3600, 7200):
        start = MAX_TARGET // (phases[0][1] * T)
        report(f"ASERT, half-life {halflife // 60} min",
               simulate(lambda c, hl=halflife, a=start: asert_target(c, hl, a), phases))

    print()
    print("switching miners (100 units/s always on, 500 units/s that come and go)")
    report_switching("legacy: every 30 blocks, max 4x", simulate_switching(legacy_target))
    fair = MAX_TARGET // (100 * T)
    for halflife in (1800, 3600, 7200):
        report_switching(f"ASERT, half-life {halflife // 60} min",
                         simulate_switching(lambda c, hl=halflife: asert_target(c, hl, fair)))
