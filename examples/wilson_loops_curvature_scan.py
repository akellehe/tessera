#!/usr/bin/env python3
"""Hinge-loop curvature scan with the Wilson-loop observable.

Walks every hinge of a small CDT triangulation, builds the elementary
loop of top-simplices around it, and records the Wilson-loop value in
``DEFICIT_ANGLE`` mode. The result is the local Levi-Civita holonomy
trace at each hinge — a direct, per-hinge curvature reading on the
triangulation.

Companion to ``docs/source/wilson_loops.md``.

Mode reference:

* ``DEFICIT_ANGLE`` on a hinge loop: W = ((d-2) + 2 cos(eps_h)) / d.
  Flat hinge ⇒ W = 1; large positive or negative deficit pulls W toward
  (d-2)/d and below.
* ``COMBINATORIAL``: W = loop length; ``enclosed_hinges`` and
  ``contractible`` carry the topological readings instead.
* ``CAUSAL``: W = signed time-orientation winding; non-zero values
  mark loops that cross the CDT foliation.

Run::

    python examples/wilson_loops_curvature_scan.py \\
        --n-simplices 200 --d 3 --equilibrate 200 \\
        --out-json /tmp/wilson-scan.json

For a quick smoke test (no I/O)::

    python examples/wilson_loops_curvature_scan.py --n-simplices 80 --d 3
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

import tessera


def build_spacetime(n_simplices: int, d: int):
    """Toroidal CDT triangulation at the requested target volume."""
    sig    = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st     = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                                  tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    return st


def equilibrate(st, n_sweeps: int):
    """Run a few Metropolis sweeps so the triangulation isn't a perfect
    flat staircase — gives some curvature variation across hinges.
    Skip if n_sweeps == 0."""
    if n_sweeps <= 0:
        return
    target = st.get_n41() if hasattr(st, "get_n41") else 0
    if target == 0:
        return
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / max(target, 1), target)
    cdt.sweep(n_sweeps)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-simplices", type=int, default=200,
                    help="target volume for the CDT triangulation")
    p.add_argument("--d", type=int, default=3,
                    help="spacetime dimension (2, 3, or 4)")
    p.add_argument("--equilibrate", type=int, default=200,
                    help="number of Metropolis sweeps before measurement; "
                          "0 = measure the flat initial build")
    p.add_argument("--out-json", default=None,
                    help="optional path for the result record")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[setup] d={args.d}, target volume={args.n_simplices}, "
          f"equilibrate={args.equilibrate}")
    st = build_spacetime(args.n_simplices, args.d)
    equilibrate(st, args.equilibrate)

    # Sweep every hinge in DEFICIT_ANGLE mode.
    wl = tessera.WilsonLoop(st)
    wl.measure_all_hinges(tessera.WilsonMode.DEFICIT_ANGLE)
    meas = wl.get_measurements()
    if not meas:
        print("[abort] no measurable hinges in the triangulation — "
              "is the build size too small?")
        return

    values = [m.value for m in meas]
    sizes  = [m.loop_size for m in meas]
    avg_by_size = wl.get_average_by_size()

    # The deficit-angle Wilson value sits at 1 for a flat hinge; the
    # deviation tells you the local curvature scale. (d-2)/d is the
    # asymptote for eps = π/2 (max single-hinge deficit reachable).
    flat_value = 1.0
    print()
    print(f"[result] n hinges measured: {len(values)}")
    print(f"[result] mean W:            {statistics.mean(values):+.4f}")
    print(f"[result] stdev W:           {statistics.stdev(values):+.4f}" if len(values) > 1 else "")
    print(f"[result] min / max W:       {min(values):+.4f} / {max(values):+.4f}")
    print(f"[result] deviation from flat (mean |W - 1|): "
          f"{statistics.mean(abs(v - flat_value) for v in values):.4f}")
    print()

    print("[result] mean W by loop size (Creutz-ratio shape):")
    for size in sorted(avg_by_size):
        n_at_size = sum(1 for s in sizes if s == size)
        print(f"    L = {size:>3}   <W> = {avg_by_size[size]:+.4f}   "
              f"(n = {n_at_size})")

    if args.out_json:
        record = {
            "config": {
                "d": args.d,
                "n_simplices": args.n_simplices,
                "equilibrate": args.equilibrate,
            },
            "summary": {
                "n_hinges": len(values),
                "mean_value": statistics.mean(values),
                "min_value": min(values),
                "max_value": max(values),
                "mean_abs_deviation_from_flat":
                    statistics.mean(abs(v - flat_value) for v in values),
            },
            "by_size": {int(k): float(v) for k, v in avg_by_size.items()},
            "values": values,
            "sizes": sizes,
        }
        with open(args.out_json, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\n[wrote] {args.out_json}")


if __name__ == "__main__":
    main()
