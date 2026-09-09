#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Volume profiles in the three CDT phases (A, B, C).

Reproduces Figures 4, 5, 6 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

Paper parameters: k0=2.2, Delta=0.6, N4=10k-362k, T=80.
Our parameters are smaller; the qualitative shape differences between
phases emerge at N4 > ~5000.

The Regge action is (Eq. 2 of hep-th/0505154):

  S_E = -(k0 + 6*Delta)*N0 + (k4 + 2*Delta)*N41 + (k4 + Delta)*N32

To reproduce the paper results (Figs 4-6):
  python examples/volume_profile_phases.py \
      --n-simplices 80000 --n-therm 500 --n-meas 100 \
      --meas-interval 20

The paper uses N4 up to 362k; 80k is sufficient to clearly
distinguish the three phase profiles (blob, crumpled, polymer).

Parallelization
---------------
The three phases (A, B, C_dS) use different coupling constants (k0,
Delta) and each builds its own spacetime from scratch.  No state is
shared between phases, so all three run concurrently in threads
(--workers).

The GIL is released inside the C++ sweep() call, giving threads real
CPU parallelism without forking processes or duplicating memory.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.progress import ProgressDisplay, make_tune_cb


def _phase_worker(phase_id, label, k0, delta, n_simplices, n_therm, n_meas,
                  meas_interval, sweep_cb=None, phase_cb=None):
    """Run one phase simulation: build, tune, thermalize, measure.

    Each phase uses different coupling constants and is fully independent.
    The GIL is released during sweep(), so threads run in parallel.
    The per-sweep callback reacquires the GIL only briefly (microseconds)
    between sweeps that each take milliseconds-to-seconds at large N4,
    so the overhead is negligible.
    """
    _ph = lambda p, done=0, total=0: phase_cb(phase_id, p, done, total) if phase_cb else None

    _ph("building")
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    max_build = 80 * 20  # cap at ~80 time slices (20 simplices/slab in 4D)
    st.build(min(n_simplices, max_build))
    target = st.getN41() if n_simplices <= max_build else n_simplices // 2
    cdt = tessera.CDTSimulation(st, k0, 0.5, delta, 1.0 / target, target)

    _ph("tuning")
    cdt.tune(progress=make_tune_cb(phase_cb, phase_id))

    chunk = max(1, n_therm // 20)
    for start in range(0, n_therm, chunk):
        batch = min(chunk, n_therm - start)
        cdt.sweep(batch, progress=sweep_cb)
        _ph("thermalizing", start + batch, n_therm)

    profiles = []
    for i in range(n_meas):
        cdt.sweep(meas_interval, progress=sweep_cb)
        profiles.append(cdt.getVolumeProfile())
        _ph("measuring", i + 1, n_meas)

    return label, profiles, cdt.getAcceptanceRates(), cdt.getK4()


def average_profile(profiles):
    """Peak-centered average of volume profiles via the C++ VolumeProfile.

    On a torus the de Sitter blob can sit at any time slice and its
    position diffuses along the Markov chain.  Naive bin-by-bin averaging
    smears the blob into uniform noise, so each profile is circularly
    rolled to align its peak at T//2 before averaging — the technique
    described in the CDT literature (Ambjorn et al., 2005).
    """
    return np.asarray(tessera.VolumeProfile.centeredAverage(profiles))


def plot_universe_surface(profile, title, ax, color_map=cm.coolwarm):
    T = len(profile)
    tau = np.linspace(0, 1, T)
    radius = np.sqrt(np.maximum(profile, 0))
    if radius.max() > 0:
        radius = radius / radius.max()

    n_theta = 60
    theta = np.linspace(0, 2 * np.pi, n_theta)
    tau_grid, theta_grid = np.meshgrid(tau, theta)
    r_grid = np.tile(radius, (n_theta, 1))

    X = r_grid * np.cos(theta_grid)
    Y = r_grid * np.sin(theta_grid)
    Z = tau_grid

    ax.plot_surface(X, Y, Z, cmap=color_map, alpha=0.85, edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\tau$")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)


def main():
    monitor = MemoryMonitor()
    parser = argparse.ArgumentParser(
        description="CDT volume profiles in phases A, B, C "
                    "(Figs 4-6 of hep-th/0505154)")
    parser.add_argument("--n-simplices", type=int, default=5000,
                        help="Target number of simplices (>=5000 for "
                             "cos^3 to emerge)")
    parser.add_argument("--n-therm", type=int, default=200,
                        help="Thermalization sweeps")
    parser.add_argument("--n-meas", type=int, default=30,
                        help="Number of measurement configurations")
    parser.add_argument("--meas-interval", type=int, default=20,
                        help="Sweeps between measurements for "
                             "decorrelation")
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker threads (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to write the figures to; the surface and "
                             "profile plots get the _surface and _profile "
                             "suffixes.  Without it they are displayed "
                             "interactively")
    args = parser.parse_args()

    n_workers = max(1, args.workers)

    print("=" * 64)
    print("  CDT Volume Profiles in Phases A, B, C")
    print("  Reproduces Figs 4-6, Ambjorn, Jurkiewicz, Loll (2005)")
    print(f"  N4={args.n_simplices}, therm={args.n_therm}, "
          f"meas={args.n_meas}, interval={args.meas_interval}")
    print(f"  Workers: {n_workers} (threads, shared memory)")
    print("=" * 64)

    phases = {
        "Phase A\n"r"($\kappa_0=5.0,\;\Delta=0$)": (5.0, 0.0),
        "Phase B\n"r"($\kappa_0=1.6,\;\Delta=0$)": (1.6, 0.0),
        r"Phase $C_{dS}$""\n"r"($\kappa_0=2.2,\;\Delta=0.6$)": (2.2, 0.6),
    }

    fig_surf = plt.figure(figsize=(18, 6))
    fig_surf.suptitle(
        "CDT Universe Snapshots  (cf. Figs 4-6, Ambjorn, Jurkiewicz, Loll,\n"
        r"$\it{Reconstructing\ the\ Universe}$, Phys. Rev. D 72, 2005"
        f"  [N4={args.n_simplices}])",
        fontsize=13)

    fig_line, ax_line = plt.subplots(figsize=(10, 6))
    t_total = time.time()

    # All three phases are independent — run in parallel.
    n_phases = len(phases)
    sweeps_per_phase = args.n_therm + args.n_meas * args.meas_interval
    progress = ProgressDisplay(n_phases, n_phases * sweeps_per_phase,
                               item_label="Phases",
                               memory_monitor=monitor)

    phase_results = {}
    label_to_pid = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for pid, (label, (k0, delta)) in enumerate(phases.items()):
            label_to_pid[label] = pid
            f = pool.submit(_phase_worker, pid, label, k0, delta,
                            args.n_simplices, args.n_therm,
                            args.n_meas, args.meas_interval,
                            progress.on_sweep, progress.on_phase)
            futures[f] = label

        for f in as_completed(futures):
            label, profiles, rates, k4 = f.result()
            phase_results[label] = (profiles, rates)
            short = label.split("\n")[0]
            progress.on_item_done(label_to_pid[label], short)

    progress.finish()

    for label, (profiles, rates) in phase_results.items():
        short = label.split("\n")[0]
        avg = average_profile(profiles)
        print(f"  {short}: slices={len(avg)}, "
              f"peak N3={avg.max():.1f}, rates={rates}")

    phase_c_avg = None  # track for cos^3 overlay

    for idx, (label, (k0, delta)) in enumerate(phases.items()):
        profiles, rates = phase_results[label]
        avg = average_profile(profiles)

        ax_surf = fig_surf.add_subplot(1, 3, idx + 1, projection="3d")
        plot_universe_surface(avg, label, ax_surf)

        short_label = label.split("\n")[0]
        tau = np.arange(len(avg)) - len(avg) // 2
        ax_line.plot(tau, avg, "o-", label=short_label,
                     linewidth=2, markersize=4)

        if "C_{dS}" in label:
            phase_c_avg = avg

    # Overlay cos^3 reference on the line plot (Eq. 28, hep-th/0505154).
    # Profiles are centered on their peak (at T//2), so the blob is at
    # the origin and the reference curve is simply cos^3(pi*tau/T).
    if phase_c_avg is not None:
        T_ref = len(phase_c_avg)
        stalk = float(np.min(phase_c_avg))
        amplitude = float(phase_c_avg.max()) - stalk
        tau_ref = np.linspace(-T_ref / 2, T_ref / 2, 200)
        cos3_ref = stalk + amplitude * np.maximum(
            np.cos(np.pi * tau_ref / T_ref), 0) ** 3
        ax_line.plot(tau_ref, cos3_ref, "k--",
                     alpha=0.4, linewidth=1.5, label=r"$\cos^3$ reference")

    ax_line.set_xlabel(r"$\tau - \tau_{\mathrm{peak}}$", fontsize=13)
    ax_line.set_ylabel(r"$N_3(\tau)$", fontsize=13)
    ax_line.set_title(
        r"Spatial volume profile $N_3(\tau)$"
        "\n(cf. Figs 4-6, Ambjorn, Jurkiewicz, Loll, "
        r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax_line.legend(fontsize=11)
    ax_line.grid(True, alpha=0.3)

    fig_surf.tight_layout(rect=[0, 0, 1, 0.90])
    fig_line.tight_layout()

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")
    if args.save:
        fig_surf.savefig(args.save.replace(".png", "_surface.png"), dpi=150)
        fig_line.savefig(args.save.replace(".png", "_profile.png"), dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
