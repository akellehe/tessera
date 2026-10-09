#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Distribution of N_4^{(3,2)} at fixed N_4^{(4,1)}.

Reproduces Figure 2 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

In 4D CDT there are two types of four-simplices: (4,1)-type (with 4
vertices at time tau and 1 at tau+1) and (3,2)-type (with 3 at tau and
2 at tau+1).  The volume-fixing term in the action constrains
N_4^{(4,1)}, but the number of (3,2)-simplices N_4^{(3,2)} fluctuates
freely.  Figure 2 shows that the distribution of N_4^{(3,2)} is sharply
peaked, demonstrating that the two types of simplices are strongly
correlated in the de Sitter phase.

Parameters: k0 = 2.2, Delta = 0.6.

To reproduce the paper results (Fig 2):
  python examples/n32_distribution.py \
      --target-volumes 40000 80000 160000 --n-therm 500 \
      --n-meas 500 --meas-interval 50

The paper uses N4 = 40k, 80k, 160k to show the sharply peaked
N32 distribution narrows with increasing volume.

Parallelization
---------------
The script runs one simulation per target volume (e.g. N41 = 5k, 10k,
20k).  Each target volume builds its own spacetime with independent
coupling constants and topology — there is no shared state — so all
volumes execute concurrently in threads (--workers).

The GIL is released inside the C++ sweep() call, giving threads real
CPU parallelism without forking processes or duplicating memory.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.progress import ProgressDisplay, make_tune_cb


def _volume_worker(vol_id, target_n41, n_therm, n_meas, meas_interval,
                   sweep_cb=None, phase_cb=None):
    """Run one target-volume simulation: build, thermalize, collect N32/N41.

    Each volume is a fully independent simulation.  The GIL is released
    during sweep(), so multiple threads run in parallel.
    """
    _ph = lambda p, done=0, total=0: phase_cb(vol_id, p, done, total) if phase_cb else None

    _ph("building")
    n_build = target_n41 * 2
    max_build = 80 * 20  # cap at ~80 time slices (20 simplices/slab in 4D)
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(min(n_build, max_build))
    target = st.getN41() if n_build <= max_build else target_n41
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)

    _ph("tuning")
    cdt.tune(progress=make_tune_cb(phase_cb, vol_id))

    chunk = max(1, n_therm // 20)
    for start in range(0, n_therm, chunk):
        batch = min(chunk, n_therm - start)
        cdt.sweep(batch, progress=sweep_cb)
        _ph("thermalizing", start + batch, n_therm)

    n32_samples = []
    n41_samples = []
    for i in range(n_meas):
        cdt.sweep(meas_interval, progress=sweep_cb)
        n32_samples.append(st.getN32())
        n41_samples.append(st.getN41())
        _ph("measuring", i + 1, n_meas)

    return (target_n41,
            np.array(n32_samples, dtype=float),
            np.array(n41_samples, dtype=float))


def main():
    monitor = MemoryMonitor()
    parser = argparse.ArgumentParser(
        description="N_4^{(3,2)} distribution at fixed N_4^{(4,1)} "
                    "(Fig 2 of hep-th/0505154)")
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-meas", type=int, default=200)
    parser.add_argument("--meas-interval", type=int, default=3)
    parser.add_argument("--target-volumes", type=int, nargs="+",
                        default=None,
                        help="Target N41 values (default: 5000 10000 20000)")
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker threads (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    # Run at multiple target volumes
    # Paper (Fig 2) uses N̄₄ = 40k, 80k, 160k.  We use smaller but
    # still large volumes to keep runtime under ~10 minutes.
    target_n41_values = args.target_volumes or [5000, 10000, 20000]
    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  N32 Distribution at Fixed N41")
    print("  Reproduces Fig 2, Ambjorn, Jurkiewicz, Loll (2005)")
    print("  Parameters: k0=2.2, Delta=0.6")
    print(f"  Target volumes: {target_n41_values}")
    print(f"  Workers: {n_workers} (threads, shared memory)")
    print("=" * 64)
    colors = ["black", "red", "green", "blue"]

    fig, ax = plt.subplots(figsize=(10, 7))
    t_total = time.time()

    # Each target volume is independent — run all in parallel.
    n_vols = len(target_n41_values)
    sweeps_per_vol = args.n_therm + args.n_meas * args.meas_interval
    total_sweeps = n_vols * sweeps_per_vol

    progress = ProgressDisplay(n_vols, total_sweeps, item_label="Volumes",
                                   memory_monitor=monitor)

    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_volume_worker, vid, tv, args.n_therm,
                        args.n_meas, args.meas_interval,
                        progress.on_sweep, progress.on_phase):
            (vid, tv)
            for vid, tv in enumerate(target_n41_values)
        }
        for f in as_completed(futures):
            vid, tv = futures[f]
            target_n41, n32_arr, n41_arr = f.result()
            results[target_n41] = (n32_arr, n41_arr)
            progress.on_item_done(vid, f"N₄₁≈{target_n41:,}")

    progress.finish()

    for tv in target_n41_values:
        n32_arr, n41_arr = results[tv]
        print(f"  N41~{tv:,}: "
              f"mean={n41_arr.mean():,.0f}  std={n41_arr.std():,.0f}  "
              f"range=[{n41_arr.min():,.0f}, {n41_arr.max():,.0f}]")
        print(f"  N32: mean={n32_arr.mean():,.0f}  "
              f"std={n32_arr.std():,.0f}  "
              f"range=[{n32_arr.min():,.0f}, {n32_arr.max():,.0f}]")
        print(f"  N32/N41 ratio: "
              f"{n32_arr.mean()/max(n41_arr.mean(),1):.3f}")

    # Plot in order of target volume
    for idx, target_n41 in enumerate(target_n41_values):
        n32_arr, n41_arr = results[target_n41]

        # Plot histogram (normalized)
        if len(n32_arr) > 10 and n32_arr.std() > 0:
            hist, edges = np.histogram(n32_arr, bins=30, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, hist, "-o", markersize=3,
                    color=colors[idx % len(colors)],
                    label=fr"$\tilde{{N}}_4 \approx {int(n41_arr.mean())}$")

    ax.set_xlabel(r"$N_4^{(3,2)}$", fontsize=14)
    ax.set_ylabel(r"$P(N_4^{(3,2)})$", fontsize=14)
    ax.set_title(
        r"Distribution of $N_4^{(3,2)}$ at fixed $N_4^{(4,1)}$"
        "\n"r"($\kappa_0=2.2$, $\Delta=0.6$; cf. Fig 2, Ambjorn et al. 2005)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
