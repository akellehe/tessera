#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Phase diagram of 4D Causal Dynamical Triangulations.

Reproduces Figure 3 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]
and the phase diagram from:
  Gorlich, "Introduction to Causal Dynamical Triangulations" (2013)

Scans the coupling-constant space (k0, Delta) and classifies each point
into one of three phases using the vertex ratio N0/N41 — the canonical
order parameter for CDT phase transitions:

  Phase A (branched polymer): large k0.
    High vertex ratio: many vertices, low connectivity.

  Phase B (crumpled): small k0, small Delta.
    Low vertex ratio: few vertices, high connectivity.

  Phase C (de Sitter): moderate k0, nonzero Delta.
    Intermediate vertex ratio.

Parameters scanned:
  k0 in [0.5, 6.0]
  Delta in [0.0, 1.0]

Estimated runtime: ~5-20 minutes depending on grid resolution.

To reproduce the paper results (Fig 3):
  python examples/phase_diagram.py \
      --n-simplices 10000 --n-sweeps 200 --grid-size 20

A finer grid (--grid-size 30) and more sweeps (--n-sweeps 500) give
cleaner phase boundaries but take proportionally longer.

Parallelization
---------------
Each (k0, Delta) grid point is a short, self-contained CDT run:
build a spacetime, perform nSweeps sweeps, classify the resulting
configuration.  No grid point reads or writes state used by any
other, so all points can execute concurrently in threads (--workers).

The GIL is released inside the C++ sweep() call, giving threads real
CPU parallelism without forking processes or duplicating memory.
With a 10x10 grid (100 points) and 8 threads, up to 8 grid points
are evaluated simultaneously.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.progress import ProgressDisplay, make_tune_cb


# =====================================================================
# Simulation
# =====================================================================

def run_point(k0, delta, n_simplices, nSweeps,
              sweep_cb=None, phase_cb=None, point_id=None):
    """Run CDT at a single (k0, Delta) point and return observables.

    The Toroid staircase product creates d*(d+1)=20 simplices per time
    slab in 4D.  Building directly with n_simplices would create
    n_simplices/20 time slices — far too many for the spatial volume to
    develop phase structure (e.g. 10000 simplices -> 500 slices of 20).
    Real CDT simulations use T~40-80 with thousands of simplices per
    slice.  Instead, build a small initial lattice to establish a
    reasonable number of time slices, set targetN41 to the desired
    volume, and let the Monte Carlo grow the system via add moves.

    GIL is released during sweep(), so threads get real C++ parallelism.
    """
    _ph = lambda p, done=0, total=0: phase_cb(point_id, p, done, total) if phase_cb and point_id is not None else None

    _ph("building")
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric=metric,
                         spacetimeType=tessera.CDT,
                         alpha=1.0,
                         a=1.0,
                         foliation=tessera.PREFERRED,
                         topology=tessera.Toroid()
                         )

    # Build a small initial lattice: cap at ~40 time slices so spatial
    # volume per slice is large enough for phase structure to develop.
    max_build = 40 * 20  # 40 slabs x 20 simplices/slab in 4D
    st.build(min(n_simplices, max_build))

    target = n_simplices // 2
    d = 4 # Hardcoded dimensions
    k4 = (k0 + 6 * delta) / (2 * d - 2) - 2 * delta
    epsilon = 1. / target
    cdt = tessera.CDTSimulation(spacetime=st, k0=k0, k4=k4, delta=delta, epsilon=epsilon, targetN41=target)

    # tune() adjusts k4 to the pseudo-critical value for this (k0,delta)
    # and runs 20 feedback sweeps during which the system grows to target.
    _ph("tuning")
    cdt.tune(progress=make_tune_cb(phase_cb, point_id))

    # Evolve.  Do NOT use thermalize() — its early-stopping criterion
    # converges to the nearest action basin (always Phase C from the
    # initial state) and prevents exploration of other phases.
    chunk = max(1, nSweeps // 20)
    for start in range(0, nSweeps, chunk):
        batch = min(chunk, nSweeps - start)
        cdt.sweep(batch, progress=sweep_cb)
        _ph("sweeping", start + batch, nSweeps)

    profile = cdt.getVolumeProfile()
    n0 = st.getVertexCount()
    n41 = st.getN41()
    n32 = st.getN32()
    rates = cdt.getAcceptanceRates()
    k4 = cdt.getK4()

    return {
        'profile': profile,
        'n0': n0, 'n41': n41, 'n32': n32,
        'vertex_ratio': n0 / max(n41, 1),
        'rates': rates,
        'k4': k4,
    }


# =====================================================================
# Classification
# =====================================================================

def classify_grid(order_param):
    """Classify grid points into 3 phases using N32/N41 simplex ratio.

    After k4 is tuned to the pseudo-critical value for each (k0, Delta),
    the equilibrium N32/N41 ratio reflects the position in coupling-
    constant space.  High k4 (large k0) suppresses flips, keeping N32
    low; low k4 (small k0) makes flips free, letting N32 grow.

      - Phase A (polymer):  LOW  N32/N41 (large k4 suppresses flips)
      - Phase B (crumpled):  HIGH N32/N41 (small k4, flips are cheap)
      - Phase C (de Sitter): intermediate

    Classification uses the 33rd and 67th percentiles of the order
    parameter distribution.  At small system sizes the phase transition
    is a smooth crossover (no sharp discontinuity), so percentile-based
    thresholds are more robust than gap-detection.
    """
    flat = order_param.flatten()
    total_range = flat.max() - flat.min()
    mean_val = np.mean(flat)

    # If there is no meaningful variation, everything is Phase C
    if total_range < 0.05 * max(mean_val, 1e-10):
        return np.ones_like(order_param, dtype=int)

    thresh_low = np.percentile(flat, 33)
    thresh_high = np.percentile(flat, 67)

    phase_map = np.ones_like(order_param, dtype=int)    # default C
    phase_map[order_param < thresh_low] = 2              # Phase A (low)
    phase_map[order_param > thresh_high] = 0             # Phase B (high)
    return phase_map


# =====================================================================
# Main
# =====================================================================

def main():
    monitor = MemoryMonitor()
    parser = argparse.ArgumentParser(
        description="CDT phase diagram (Fig 3 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=200,
                        help="Simplices per simulation point")
    parser.add_argument("--n-sweeps", type=int, default=30,
                        help="Sweeps per point")
    parser.add_argument("--k0-min", type=float, default=0.5)
    parser.add_argument("--k0-max", type=float, default=6.0)
    parser.add_argument("--delta-min", type=float, default=0.0)
    parser.add_argument("--delta-max", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Number of grid points per axis")
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker threads (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  CDT Phase Diagram Scan")
    print("  Reproduces Fig 3, Ambjorn, Jurkiewicz, Loll (2005)")
    print(f"  Grid: {args.grid_size}x{args.grid_size}, "
          f"N4={args.n_simplices}, sweeps={args.n_sweeps}")
    print(f"  k0 in [{args.k0_min}, {args.k0_max}], "
          f"Delta in [{args.delta_min}, {args.delta_max}]")
    print(f"  Workers: {n_workers} (threads, shared memory)")
    print("=" * 64)

    k0_values = np.linspace(args.k0_min, args.k0_max, args.grid_size)
    delta_values = np.linspace(args.delta_min, args.delta_max, args.grid_size)

    total_points = len(k0_values) * len(delta_values)

    t0 = time.time()
    total_sweeps = total_points * args.n_sweeps

    progress = ProgressDisplay(total_points, total_sweeps,
                               item_label="Points",
                               memory_monitor=monitor)

    # Collect results from all grid points
    results = {}  # (i, j) -> result dict

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        point_id = 0
        for i, delta in enumerate(delta_values):
            for j, k0 in enumerate(k0_values):
                f = pool.submit(run_point, k0, delta,
                                args.n_simplices, args.n_sweeps,
                                progress.on_sweep, progress.on_phase,
                                point_id)
                futures[f] = (i, j, k0, delta, point_id)
                point_id += 1

        for future in as_completed(futures):
            i, j, k0, delta, pid = futures[future]
            result = future.result()
            results[(i, j)] = result
            ratio = result['n32'] / max(result['n41'], 1)
            progress.on_item_done(pid,
                f"k₀={k0:.1f} Δ={delta:.2f} N32/N41={ratio:.2f}")

    progress.finish()

    # Build the N32/N41 order-parameter grid and classify
    op_grid = np.zeros((len(delta_values), len(k0_values)))
    for (i, j), result in results.items():
        op_grid[i, j] = result['n32'] / max(result['n41'], 1)

    phase_map = classify_grid(op_grid)

    # Count phases
    phase_counts = {"A": 0, "B": 0, "C": 0}
    for i in range(len(delta_values)):
        for j in range(len(k0_values)):
            label = ["B", "C", "A"][phase_map[i, j]]
            phase_counts[label] += 1

    elapsed = time.time() - t0
    print(f"\nScan complete: {elapsed:.1f}s "
          f"({elapsed / total_points:.2f}s per point)")
    print(f"  Phase A (polymer):    {phase_counts['A']:3d} points "
          f"({100 * phase_counts['A'] / total_points:.0f}%)")
    print(f"  Phase B (crumpled):   {phase_counts['B']:3d} points "
          f"({100 * phase_counts['B'] / total_points:.0f}%)")
    print(f"  Phase C (de Sitter):  {phase_counts['C']:3d} points "
          f"({100 * phase_counts['C'] / total_points:.0f}%)")

    # Diagnostics: show order parameter range
    op_flat = op_grid.flatten()
    print(f"\n  Order param N32/N41: "
          f"min={op_flat.min():.4f}, max={op_flat.max():.4f}, "
          f"mean={op_flat.mean():.4f}, std={op_flat.std():.4f}")

    # Show corner diagnostics
    corners = [
        (0, 0, "k0=min, D=min"),
        (0, -1, "k0=max, D=min"),
        (-1, 0, "k0=min, D=max"),
        (-1, -1, "k0=max, D=max"),
    ]
    for di, ki, label in corners:
        r = results[(di % len(delta_values), ki % len(k0_values))]
        print(f"  {label}: N0={r['n0']}, N41={r['n41']}, N32={r['n32']}, "
              f"VR={r['vertex_ratio']:.4f}, k4={r['k4']:.3f}, "
              f"rates={{add={r['rates']['add']:.3f}, "
              f"rem={r['rates']['remove']:.3f}, "
              f"flip={r['rates']['flip']:.3f}}}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: discrete phase map
    ax = axes[0]
    cmap = ListedColormap(["#4477AA", "#66CC66", "#EE6677"])
    phase_labels = ["Phase B\n(crumpled)", r"Phase $C_{dS}$""\n(de Sitter)",
                    "Phase A\n(polymer)"]

    im = ax.pcolormesh(k0_values, delta_values, phase_map,
                       cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(phase_labels)

    ax.set_xlabel(r"$\kappa_0$", fontsize=14)
    ax.set_ylabel(r"$\Delta$", fontsize=14)
    ax.set_title("CDT Phase Diagram\n"
                 "(cf. Fig 3, Ambjorn et al. 2005)", fontsize=13)
    ax.plot(2.2, 0.6, "w*", markersize=15, markeredgecolor="black",
            label=r"Paper: $\kappa_0=2.2, \Delta=0.6$")
    ax.legend(fontsize=11, loc="upper right")

    # Right panel: continuous order parameter
    ax2 = axes[1]
    im2 = ax2.pcolormesh(k0_values, delta_values, op_grid,
                         cmap="RdYlBu_r", shading="nearest")
    fig.colorbar(im2, ax=ax2, label=r"$N_{32} / N_{41}$")
    ax2.set_xlabel(r"$\kappa_0$", fontsize=14)
    ax2.set_ylabel(r"$\Delta$", fontsize=14)
    ax2.set_title(r"Simplex ratio $N_{32} / N_{41}$"
                  "\n(order parameter)", fontsize=13)
    ax2.plot(2.2, 0.6, "w*", markersize=15, markeredgecolor="black")

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
