#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Volume-volume correlator and Hausdorff dimension measurement.

Reproduces Figures 7-8, 12 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

Paper parameters: k0=2.2, Delta=0.6, N4=10k-160k, T=80.

To reproduce the paper results (Figs 7-8, 12):
  python examples/volume_scaling.py \
      --n-simplices 80000 --n-therm 500 --n-meas 200 \
      --meas-interval 50

This runs at N4 = 40k, 80k, 160k (half, full, double --n-simplices).
The paper uses three system sizes to demonstrate D_H = 4 scaling
collapse of the volume-volume correlator.

Parallelization
---------------
The script runs CDT at three system sizes (N4/2, N4, 2*N4) plus an
extra run at the largest size for the volume-difference distribution —
four independent simulations in total.  Each builds its own spacetime
and Markov chain, sharing no mutable state, so all four run
concurrently in threads (--workers).

The GIL is released inside the C++ sweep() call, giving threads real
CPU parallelism without forking processes or duplicating memory.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.progress import ProgressDisplay, make_tune_cb


def _profiles_worker(run_id, n_simplices, n_therm, n_meas, meas_interval,
                     sweep_cb=None, phase_cb=None):
    """Run one system-size simulation and collect volume profiles.

    Each size is independent.  The GIL is released during sweep(),
    so multiple threads run in parallel.
    """
    _ph = lambda p, done=0, total=0: phase_cb(run_id, p, done, total) if phase_cb else None

    _ph("building")
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    max_build = 80 * 20  # cap at ~80 time slices (20 simplices/slab in 4D)
    st.build(min(n_simplices, max_build))
    target = st.getN41() if n_simplices <= max_build else n_simplices // 2
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)

    _ph("tuning")
    cdt.tune(progress=make_tune_cb(phase_cb, run_id))

    chunk = max(1, n_therm // 20)
    for start in range(0, n_therm, chunk):
        batch = min(chunk, n_therm - start)
        cdt.sweep(batch, progress=sweep_cb)
        _ph("thermalizing", start + batch, n_therm)

    profiles = []
    for i in range(n_meas):
        cdt.sweep(meas_interval, progress=sweep_cb)
        profiles.append(np.array(cdt.getVolumeProfile(), dtype=float))
        _ph("measuring", i + 1, n_meas)
    return n_simplices, profiles


def compute_volume_correlator(profiles, stalk_volume=None):
    """Volume-volume correlator C_{N4}(delta) (Eq. 7 of the paper)."""
    T = max(len(p) for p in profiles)
    correlator = np.zeros(T)
    count = np.zeros(T)
    for p in profiles:
        n3 = np.array(p, dtype=float)
        t = len(n3)
        s = stalk_volume if stalk_volume is not None else max(np.min(n3), 0)
        n3_shifted = n3 - s / 2.0
        n4_eff = np.sum(n3) - t * s
        if n4_eff <= 0:
            continue
        for delta in range(t):
            c = 0.0
            for tau in range(t):
                c += n3_shifted[tau] * n3_shifted[(tau + delta) % t]
            correlator[delta] += 4.0 * c / (n4_eff ** 2)
            count[delta] += 1
    count[count == 0] = 1
    return correlator / count


def main():
    monitor = MemoryMonitor()
    parser = argparse.ArgumentParser(
        description="Volume-volume correlator (Figs 7-8, 12)")
    parser.add_argument("--n-simplices", type=int, default=500)
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-meas", type=int, default=30)
    parser.add_argument("--meas-interval", type=int, default=5)
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker threads (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  Volume-Volume Correlator & Hausdorff Dimension")
    print("  Reproduces Figs 7-8, 12, Ambjorn, Jurkiewicz, Loll (2005)")
    print("  Parameters: k0=2.2, Delta=0.6")
    print(f"  Workers: {n_workers} (threads, shared memory)")
    print("=" * 64)

    sizes = [args.n_simplices // 2, args.n_simplices, args.n_simplices * 2]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Store correlators for D_H scan
    all_corrs = {}
    t_total = time.time()

    # ---- Run all sizes + Fig 12 run in parallel ----
    # The 3 sizes for Fig 7 plus an extra run at the largest size for Fig 12
    # are all independent simulations.
    all_runs = sizes + [sizes[-1]]  # 4th entry is for Fig 12
    n_runs = len(all_runs)
    sweeps_per_run = args.n_therm + args.n_meas * args.meas_interval
    total_sweeps = n_runs * sweeps_per_run

    progress = ProgressDisplay(n_runs, total_sweeps, item_label="Runs",
                                   memory_monitor=monitor)

    size_profiles = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_profiles_worker, idx, n4, args.n_therm,
                        args.n_meas, args.meas_interval,
                        progress.on_sweep, progress.on_phase):
            (idx, n4)
            for idx, n4 in enumerate(all_runs)
        }
        for f in as_completed(futures):
            idx, n4 = futures[f]
            _, profiles = f.result()
            size_profiles[idx] = profiles
            progress.on_item_done(idx,
                f"N₄={n4:,}, {len(profiles)} profiles")

    progress.finish()

    # ---- Fig 7: Rescaled volume-volume correlator ----
    ax_corr = axes[0, 0]
    colors = ["red", "green", "blue"]
    for i, n4 in enumerate(sizes):
        profiles = size_profiles[i]
        avg_slices = np.mean([len(p) for p in profiles])
        avg_vol = np.mean([np.sum(p) for p in profiles])

        corr = compute_volume_correlator(profiles)
        T = len(corr)
        if T < 2:
            continue

        avg_n4_eff = np.mean([np.sum(p) for p in profiles])
        all_corrs[avg_n4_eff] = corr

        D_H = 4.0
        delta = np.arange(T)
        x = delta / max(avg_n4_eff ** (1.0 / D_H), 1e-10)
        c_rescaled = corr * (avg_n4_eff ** (1.0 / D_H))

        ax_corr.plot(x, c_rescaled, "o", markersize=3, color=colors[i],
                     label=f"$N_4 \\approx {int(avg_n4_eff)}$")

    ax_corr.set_xlabel(r"$x = \delta / (\tilde{N}_4^{\mathrm{eff}})^{1/D_H}$",
                       fontsize=12)
    ax_corr.set_ylabel(r"$c_{\tilde{N}_4}(x)$", fontsize=12)
    ax_corr.set_title(
        r"Rescaled volume-volume correlator ($D_H=4$)"
        "\n(cf. Fig 7, Ambjorn, Jurkiewicz, Loll,\n"
        r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax_corr.legend(fontsize=10)
    ax_corr.grid(True, alpha=0.3)

    # ---- Fig 8: D_H from best overlap ----
    ax_dh = axes[0, 1]
    D_H_range = np.linspace(2.0, 6.0, 40)
    errors = []

    if len(all_corrs) >= 2:
        # Pick the smallest correlator as reference, measure overlap with others
        sorted_keys = sorted(all_corrs.keys())
        ref_key = sorted_keys[len(sorted_keys) // 2]
        ref_corr = all_corrs[ref_key]

        for D_H_trial in D_H_range:
            total_err = 0
            n_pairs = 0
            for n4_eff, corr in all_corrs.items():
                if n4_eff == ref_key:
                    continue
                T1, T2 = len(ref_corr), len(corr)
                # Rescale both to same x-axis
                x_ref = np.arange(T1) / max(ref_key ** (1.0 / D_H_trial), 1e-10)
                x_other = np.arange(T2) / max(n4_eff ** (1.0 / D_H_trial), 1e-10)
                c_ref = ref_corr * (ref_key ** (1.0 / D_H_trial))
                c_other = corr * (n4_eff ** (1.0 / D_H_trial))
                # Interpolate and compare on common x range
                x_max = min(x_ref[-1], x_other[-1])
                x_common = np.linspace(0, x_max, 20)
                if x_max <= 0:
                    continue
                c1 = np.interp(x_common, x_ref, c_ref)
                c2 = np.interp(x_common, x_other, c_other)
                norm = np.mean(np.abs(c1) + np.abs(c2)) + 1e-10
                total_err += np.mean((c1 - c2) ** 2) / (norm ** 2)
                n_pairs += 1
            errors.append(total_err / max(n_pairs, 1))

        errors = np.array(errors)
        if errors.max() > 0:
            errors /= errors.max()
        ax_dh.plot(D_H_range, errors, "k-", linewidth=1.5)
        best_dh = D_H_range[np.argmin(errors)]
        ax_dh.axvline(x=best_dh, color="blue", linestyle="--", alpha=0.5,
                      label=f"Best: $D_H \\approx {best_dh:.1f}$")

    ax_dh.axvline(x=4.0, color="gray", linestyle=":", alpha=0.7,
                  label=r"Paper: $D_H=4$")
    ax_dh.set_xlabel(r"$D_H$", fontsize=12)
    ax_dh.set_ylabel("Overlap error (normalized)", fontsize=12)
    ax_dh.set_title(
        "Hausdorff dimension estimate\n"
        "(cf. Fig 8, Ambjorn, Jurkiewicz, Loll,\n"
        r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax_dh.legend(fontsize=10)
    ax_dh.grid(True, alpha=0.3)

    # ---- Fig 12: Volume difference distribution ----
    ax_vdiff = axes[1, 0]
    profiles = size_profiles[3]  # extra run at largest size
    all_z = []
    for p in profiles:
        for tau in range(len(p) - 1):
            n3_total = p[tau] + p[tau + 1]
            if n3_total <= 0:
                continue
            diff = abs(p[tau + 1] - p[tau])
            z = diff / max(n3_total ** 0.5, 1e-10)
            all_z.append(z)

    if len(all_z) > 10:
        all_z = np.array(all_z)
        hist, edges = np.histogram(all_z, bins=25, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_vdiff.plot(centers, hist, "o", markersize=4, color="blue",
                      label="Measured")
        try:
            def gaussian(x, A, c):
                return A * np.exp(-c * x ** 2)
            mask = hist > 0
            popt, _ = curve_fit(gaussian, centers[mask], hist[mask],
                                p0=[hist.max(), 1.0], maxfev=5000)
            x_fit = np.linspace(0, centers.max(), 100)
            ax_vdiff.plot(x_fit, gaussian(x_fit, *popt), "r-", linewidth=1.5,
                          label=fr"Gaussian: $e^{{-{popt[1]:.2f} z^2}}$")
        except Exception:
            pass

    ax_vdiff.set_xlabel(r"$z$", fontsize=12)
    ax_vdiff.set_ylabel(r"$P_{N_3}(z)$", fontsize=12)
    ax_vdiff.set_title(
        r"Rescaled volume differences ($D_2=2$)"
        "\n(cf. Fig 12, Ambjorn, Jurkiewicz, Loll,\n"
        r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax_vdiff.legend(fontsize=10)
    ax_vdiff.grid(True, alpha=0.3)

    if len(all_corrs) >= 2 and errors.max() > 0:
        print(f"\n  Best-fit Hausdorff dimension: D_H = {best_dh:.2f} "
              f"(paper: D_H = 4)")

    # ---- Volume profile with individual configs ----
    ax_prof = axes[1, 1]
    for p in profiles[-5:]:
        ax_prof.plot(p, alpha=0.3, color="blue", linewidth=0.8)
    max_len = max(len(p) for p in profiles)
    avg = np.zeros(max_len)
    cnt = np.zeros(max_len)
    for p in profiles:
        avg[:len(p)] += p
        cnt[:len(p)] += 1
    cnt[cnt == 0] = 1
    avg /= cnt
    ax_prof.plot(avg, "k-", linewidth=2, label="Average")
    ax_prof.set_xlabel(r"$\tau$", fontsize=12)
    ax_prof.set_ylabel(r"$N_3(\tau)$", fontsize=12)
    ax_prof.set_title("Volume profile (individual configs + average)")
    ax_prof.legend(fontsize=10)
    ax_prof.grid(True, alpha=0.3)

    fig.suptitle(
        r"Volume scaling analysis ($\kappa_0=2.2$, $\Delta=0.6$)",
        fontsize=14, y=1.01)
    fig.tight_layout()

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
