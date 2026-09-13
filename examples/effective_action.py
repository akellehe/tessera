#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Effective action and minisuperspace comparison.

Reproduces Figures 11-13 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

The effective action for CDT is extracted by measuring how the spatial
volume evolves between adjacent time slices.  For large three-volumes
N_3(tau), the Euclidean effective action takes the form (Eq. 38):

    S_eff ~ sum_tau [ c1/N_3(tau) * (Delta N_3 / Delta tau)^2
                     + c2 * N_3^alpha - lambda * N_3(tau) ]

The paper shows that:
  1. The kinetic term scaling dimension is D_2 ~ 2 (Fig 11).
  2. The distribution of rescaled volume differences is Gaussian (Fig 12).
  3. The volume-volume correlator from the effective action matches the
     Monte Carlo data (Fig 13), confirming that CDT dynamics is
     described by the minisuperspace action (Eq. 40):

        S = (1/G) integral[ a(tau) (da/dtau)^2 + a(tau) - lambda a^3(tau) ] dtau

     but with a flipped sign on the kinetic term compared to the classical
     Einstein-Hilbert action (the "conformal mode problem" is solved
     nonperturbatively).

Parameters: k0 = 2.2, Delta = 0.6.

To reproduce the paper results (Figs 11-13):
  python examples/effective_action.py \
      --n-simplices 160000 --n-therm 500 --n-meas 200 \
      --meas-interval 50

The paper uses N4 ~ 160k to extract D_2 = 2 scaling (Fig 11),
the Gaussian volume-difference distribution (Fig 12), and the
cos^3 minisuperspace fit (Fig 13).

Parallelization
---------------
The n_meas volume-profile measurements are distributed across --workers
independent Markov chains, each with its own spacetime.  This is valid
because the analysis (D_2 scaling, volume-difference distribution,
average profile) only requires a set of equilibrium configurations — it
does not matter whether they come from one chain or many.  Independent
chains are actually preferable: they eliminate within-chain
autocorrelation, giving statistically cleaner samples.

The action-tracking subplot (bottom right) intentionally stays
sequential: it visualizes the temporal evolution of S_Regge on a single
chain, which is inherently serial.

The GIL is released inside the C++ sweep() call, so threads achieve
real CPU parallelism without forking processes or duplicating memory.
"""
import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.progress import ProgressDisplay, SingleTaskProgress, make_tune_cb


def _collect_worker(worker_id, n_simplices, n_therm, n_meas, interval,
                    sweep_cb=None, phase_cb=None, seed=None):
    """Run one independent Markov chain and collect volume profiles.

    Each worker builds, thermalizes, and measures its own spacetime.
    Independent chains give truly decorrelated samples.  The GIL is
    released during sweep(), so multiple threads run in parallel.
    """
    _ph = lambda p, done=0, total=0: phase_cb(worker_id, p, done, total) if phase_cb else None

    _ph("building")
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    max_build = 80 * 20  # cap at ~80 time slices (20 simplices/slab in 4D)
    if seed is not None:
        # Both generators decide the outcome: the spacetime's drives the build,
        # the simulation's drives the sweeps. Offset by worker so the chains
        # stay independent, which is what decorrelated samples require.
        st.setSeed(seed + worker_id)
    st.build(min(n_simplices, max_build))
    target = st.getN41() if n_simplices <= max_build else n_simplices // 2
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    if seed is not None:
        cdt.setSeed(seed + worker_id)

    _ph("tuning")
    cdt.tune(progress=make_tune_cb(phase_cb, worker_id))

    chunk = max(1, n_therm // 20)
    for start in range(0, n_therm, chunk):
        batch = min(chunk, n_therm - start)
        cdt.sweep(batch, progress=sweep_cb)
        _ph("thermalizing", start + batch, n_therm)

    profiles = []
    for i in range(n_meas):
        cdt.sweep(interval, progress=sweep_cb)
        profiles.append(np.array(cdt.getVolumeProfile(), dtype=float))
        _ph("measuring", i + 1, n_meas)

    return worker_id, profiles


def main():
    monitor = MemoryMonitor()
    parser = argparse.ArgumentParser(
        description="Effective action analysis "
                    "(Figs 11-13 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=500)
    parser.add_argument("--n-therm", type=int, default=50)
    parser.add_argument("--n-meas", type=int, default=40)
    parser.add_argument("--meas-interval", type=int, default=5)
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Independent Markov chains in parallel "
                             "(default: min(cpus, 8))")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed both generators that decide the outcome, "
                             "making the run reproducible across processes")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  Effective Action & Minisuperspace Comparison")
    print("  Reproduces Figs 11-13, Ambjorn, Jurkiewicz, Loll (2005)")
    print("  Parameters: k0=2.2, Delta=0.6")
    print(f"  N4={args.n_simplices}, therm={args.n_therm}, "
          f"meas={args.n_meas}, interval={args.meas_interval}")
    print(f"  Workers: {n_workers} independent chains (threads, shared memory)")
    print("=" * 64)

    print("\nCollecting configurations...")
    t0 = time.time()
    t_total = time.time()

    # Distribute measurements across independent Markov chains.
    # Each chain thermalizes independently — the resulting profiles are
    # truly decorrelated, which is statistically better than one long chain.
    profiles = []
    meas_per_worker = math.ceil(args.n_meas / n_workers)

    # Compute total sweeps across all chains for the progress bar
    actual_workers = 0
    remaining = args.n_meas
    worker_meas = []
    for w in range(n_workers):
        n = min(meas_per_worker, remaining)
        if n <= 0:
            break
        remaining -= n
        worker_meas.append(n)
        actual_workers += 1
    total_sweeps = sum(
        args.n_therm + n * args.meas_interval for n in worker_meas)

    progress = ProgressDisplay(actual_workers, total_sweeps,
                               item_label="Chains",
                               memory_monitor=monitor)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for w, n in enumerate(worker_meas):
            f = pool.submit(_collect_worker, w, args.n_simplices,
                            args.n_therm, n, args.meas_interval,
                            progress.on_sweep, progress.on_phase, args.seed)
            futures[f] = (w, n)

        for f in as_completed(futures):
            wid, n = futures[f]
            _, worker_profiles = f.result()
            profiles.extend(worker_profiles)
            progress.on_item_done(wid,
                f"{len(worker_profiles)} profiles")

    progress.finish()

    avg_slices = np.mean([len(p) for p in profiles])
    avg_vol = np.mean([np.sum(p) for p in profiles])
    print(f"  {len(profiles)} configurations in {time.time()-t0:.1f}s")
    print(f"  Avg slices: {avg_slices:.0f}, avg total volume: {avg_vol:,.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- Fig 11: D_2 from finite-size scaling ----
    # Measure distribution of volume differences at different N_3 values
    ax_d2 = axes[0, 0]

    D_2_range = np.linspace(1.0, 3.0, 30)
    errors = []

    all_diffs_by_n3 = {}
    for p in profiles:
        for tau in range(len(p) - 1):
            n3 = p[tau] + p[tau + 1]
            if n3 <= 0:
                continue
            diff = abs(p[tau + 1] - p[tau])
            bucket = int(n3 / 50) * 50
            if bucket not in all_diffs_by_n3:
                all_diffs_by_n3[bucket] = []
            all_diffs_by_n3[bucket].append(diff)

    # For each D_2, compute rescaled distributions and measure overlap
    for D_2 in D_2_range:
        all_z = []
        for bucket, diffs in all_diffs_by_n3.items():
            n3 = max(bucket, 1)
            z = np.array(diffs) / (n3 ** (1.0 / D_2))
            all_z.extend(z.tolist())
        if len(all_z) < 10:
            errors.append(1e10)
            continue
        all_z = np.array(all_z)
        errors.append(np.std(all_z))

    errors = np.array(errors)
    if np.any(errors < 1e9):
        errors_norm = errors / errors[errors < 1e9].max()
        ax_d2.plot(D_2_range, errors_norm, "k-", linewidth=1.5)
    ax_d2.axvline(x=2.0, color="gray", linestyle=":", label=r"$D_2=2$")
    ax_d2.set_xlabel(r"$D_2$", fontsize=12)
    ax_d2.set_ylabel("Overlap error (normalized)", fontsize=12)
    ax_d2.set_title(r"Scaling dimension $D_2$ from finite-size scaling"
                    "\n(cf. Fig 11, Ambjorn et al. 2005)")
    ax_d2.legend(fontsize=10)
    ax_d2.grid(True, alpha=0.3)

    # ---- Fig 12: Volume difference distribution ----
    ax_dist = axes[0, 1]

    all_z = []
    for p in profiles:
        for tau in range(len(p) - 1):
            n3_total = p[tau] + p[tau + 1]
            if n3_total <= 0:
                continue
            diff = abs(p[tau + 1] - p[tau])
            z = diff / max(n3_total ** 0.5, 1e-10)  # D_2 = 2
            all_z.append(z)

    if len(all_z) > 10:
        all_z = np.array(all_z)
        hist, edges = np.histogram(all_z, bins=25, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_dist.plot(centers, hist, "o", markersize=4, color="blue",
                     label="Measured")

        try:
            def gauss(x, A, c):
                return A * np.exp(-c * x**2)
            popt, _ = curve_fit(gauss, centers[hist > 0], hist[hist > 0],
                                p0=[hist.max(), 1.0], maxfev=5000)
            x_fit = np.linspace(0, centers.max(), 100)
            ax_dist.plot(x_fit, gauss(x_fit, *popt), "r-", linewidth=1.5,
                         label=fr"$e^{{-{popt[1]:.1f} z^2}}$")
        except Exception:
            pass

    ax_dist.set_xlabel(r"$z$", fontsize=12)
    ax_dist.set_ylabel(r"$P_{N_3}(z)$", fontsize=12)
    ax_dist.set_title("Volume difference distribution\n"
                      "(cf. Fig 12, Ambjorn et al. 2005)")
    ax_dist.legend(fontsize=10)
    ax_dist.grid(True, alpha=0.3)

    # ---- Fig 13: Volume profile vs minisuperspace prediction ----
    # Following the paper: center each configuration's profile on its peak
    # before averaging (to prevent smearing from peak-position fluctuations
    # on the torus), subtract the stalk, then normalise so the peak = 1.
    ax_mss = axes[1, 0]

    # Pad all profiles to the same length (the modal length)
    lengths = [len(p) for p in profiles]
    T = max(lengths)

    # Center on peak, subtract the stalk, and normalise the peak to 1 in a
    # single shared C++ pass (see VolumeProfile.centeredAverage).
    avg_centered = np.asarray(tessera.VolumeProfile.centeredAverage(
        profiles, subtractStalk=True, normalizePeak=True))

    # x-axis: centered time so peak is at 0
    tau_centered = np.arange(T) - T // 2

    # Minisuperspace prediction: cos^3(pi * tau / T)
    tau_dense = np.linspace(-T / 2, T / 2, 300)
    cos3_pred = np.maximum(np.cos(np.pi * tau_dense / T), 0) ** 3

    ax_mss.plot(tau_centered, avg_centered, "ko", markersize=4,
                label="Monte Carlo")
    ax_mss.plot(tau_dense, cos3_pred, "r-", linewidth=1.5,
                label=r"$\cos^3(\pi\tau/T)$")
    ax_mss.set_xlabel(r"$\tau - \tau_{\mathrm{peak}}$", fontsize=12)
    ax_mss.set_ylabel(r"$N_3 / N_3^{\mathrm{max}}$", fontsize=12)
    ax_mss.set_title("Volume profile vs minisuperspace prediction\n"
                     "(cf. Fig 13, Ambjorn et al. 2005)")
    ax_mss.legend(fontsize=10)
    ax_mss.grid(True, alpha=0.3)

    # ---- Action tracking over simulation ----
    # Build a fresh chain, tune k4 to pseudo-critical, thermalize into
    # de Sitter, then track action evolution in equilibrium.
    ax_action = axes[1, 1]

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(args.n_simplices)
    target = st.getN41()  # [RU] eq. 6: volume-fix targets N41
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)

    n_track = max(100, args.n_therm * 2)
    actions = []
    volumes = []
    prog2 = SingleTaskProgress(memory_monitor=monitor)
    prog2.phase("tuning", total=20)
    cdt.tune(progress=prog2.on_tick)
    prog2.phase("thermalizing", total=args.n_therm)
    cdt.sweep(args.n_therm, progress=prog2.on_tick)
    prog2.phase("tracking action", total=n_track)
    for _ in range(n_track):
        cdt.sweep(1)
        actions.append(cdt.computeAction())
        volumes.append(st.getSimplexCount())
        prog2.on_tick()
    prog2.finish()

    ax_action.plot(actions, "b-", linewidth=0.8, alpha=0.7)
    ax_action.set_xlabel("Sweep", fontsize=12)
    ax_action.set_ylabel(r"$S_{\mathrm{Regge}}$", fontsize=12)
    ax_action.set_title("Regge action evolution")
    ax_action.grid(True, alpha=0.3)

    # Inset: volume evolution
    ax_vol = ax_action.twinx()
    ax_vol.plot(volumes, "r-", linewidth=0.8, alpha=0.5)
    ax_vol.set_ylabel(r"$N_4$", color="red", fontsize=12)
    ax_vol.tick_params(axis="y", labelcolor="red")

    fig.suptitle(r"Effective action analysis ($\kappa_0=2.2$, $\Delta=0.6$)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    print(f"\nAction range: [{min(actions):.2f}, {max(actions):.2f}], "
          f"final N4={volumes[-1]:,}")
    print(f"Total elapsed: {time.time()-t_total:.1f}s")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
