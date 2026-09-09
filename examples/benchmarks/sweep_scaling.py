#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""How the cost of a Monte Carlo sweep grows with the four-volume.

A sweep attempts N4 moves, so a sweep whose cost tracked the cost of a move is
linear in N4 and the fitted exponent below is 1.  An exponent near 2 means each
move is walking something that grows with the volume -- the failure #970 was
opened for, where a vertex's incidence list was scanned per proposal.

    python examples/benchmarks/sweep_scaling.py --volumes 25000 50000 100000

Report the exponent, not the seconds: absolute timings move with the machine and
with whatever else it is running, while the exponent is the property that says
whether the sweep scales.
"""
import argparse
import math
import time

import tessera


class SweepScaling:
    """Times sweeps at a series of four-volumes on one growing complex."""

    K0 = 2.2
    DELTA = 0.6
    K4 = 0.5

    def __init__(self, seed, build, target, sweeps):
        self.sweeps = sweeps
        sig = tessera.Signature(4, tessera.Lorentzian)
        self.spacetime = tessera.Spacetime(
            tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
            tessera.PREFERRED, tessera.Toroid())
        self.spacetime.setSeed(seed)
        self.spacetime.build(build)
        self.cdt = tessera.CDTSimulation(self.spacetime, self.K0, self.K4,
                                         self.DELTA, 1.0 / target, target)
        self.cdt.setSeed(seed)
        self.cdt.tune()

    @property
    def volume(self):
        return self.spacetime.getN41() + self.spacetime.getN32()

    def time_at(self, target_volume):
        """Grow to ``target_volume``, then seconds per sweep there."""
        while self.volume < target_volume:
            self.cdt.sweep(5)
        reached = self.volume
        started = time.time()
        self.cdt.sweep(self.sweeps)
        return reached, (time.time() - started) / self.sweeps

    @staticmethod
    def exponent(points):
        """Slope of log(seconds) against log(N4): the scaling exponent."""
        if len(points) < 2:
            return float("nan")
        xs = [math.log(n4) for n4, _ in points]
        ys = [math.log(dt) for _, dt in points]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        denominator = sum((x - mx) ** 2 for x in xs)
        if denominator == 0.0:
            return float("nan")
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volumes", type=int, nargs="+",
                   default=[25000, 50000, 100000],
                   help="Four-volumes to time a sweep at")
    p.add_argument("--sweeps", type=int, default=20,
                   help="Sweeps timed per volume (default: 20)")
    p.add_argument("--build", type=int, default=1600,
                   help="Initial build size (default: 1600)")
    p.add_argument("--target", type=int, default=20000,
                   help="Volume-fixing target N41 (default: 20000)")
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    bench = SweepScaling(args.seed, args.build, args.target, args.sweeps)
    print(f"{'N4':>9}  {'s/sweep':>10}")
    points = []
    for target_volume in sorted(args.volumes):
        n4, seconds = bench.time_at(target_volume)
        points.append((n4, seconds))
        print(f"{n4:9d}  {seconds:10.4f}", flush=True)

    print(f"\nfitted exponent: {SweepScaling.exponent(points):.2f} "
          f"(1 = linear in N4, 2 = every move walks the whole complex)")


if __name__ == "__main__":
    main()
