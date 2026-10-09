#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Spectral dimension measurement via dual-graph heat-kernel diffusion.

Reproduces Figures 9-10 from:
  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
  Phys. Rev. D 72 (2005) [hep-th/0505154]

The spectral dimension D_S(sigma) is measured by diffusing on the dual
graph of the triangulation (each d-simplex is a node; neighbours share a
(d-1)-face).  The return probability

    P(sigma) = (1/|V|) Tr exp(-sigma L_sym)

-- the probability that a diffusing walker returns to its starting
simplex after diffusion time sigma -- scales as

    P(sigma) ~ sigma^{-D_S/2}

so the spectral dimension is extracted via

    D_S(sigma) = -2  d log P(sigma) / d log sigma

The dual-graph construction, the Krylov-Lanczos heat-kernel diffusion,
and the finite-difference D_S extraction all run in C++:
``st.getDualGraph()`` returns a :class:`tessera.SparseGraph`, whose
``returnProbability`` / ``spectralDimensionCurve`` methods are the same
machinery the modularity sweep and the emergent-geometry pipeline use.

Key results from the paper (k0=2.2, Delta=0.6, t=80):
  D_S(sigma -> infinity) = 4.02 +/- 0.1   (large-scale dimension)
  D_S(sigma -> 0)        = 1.80 +/- 0.25  (short-distance dimension)
  Best fit:  D_S(sigma) = 4.02 - 119/(54 + sigma)    [Eq. 29]

Parameters: k0 = 2.2, Delta = 0.6.

To reproduce the paper results (Figs 9-10):
  python examples/spectral_dimension.py \
      --n-simplices 160000 --n-therm 500 --n-configs 50 \
      --n-walks 100 --max-sigma 500 --sweeps-between 50

Parallelization
---------------
Each "configuration" is an independent Markov chain: it builds its own
spacetime, thermalizes from a cold start, and measures its own return
probability.  Because no state is shared between configurations, they can
run concurrently in threads (--workers).  The GIL is released inside the
C++ sweep() call, so threads achieve real parallelism without forking
separate processes and without duplicating memory.

The final D_S(sigma) curve is the average over the per-configuration
return-probability curves.  Mixing independent chains is standard
practice in lattice Monte Carlo -- it is statistically equivalent to (and
better decorrelated than) taking the same number of measurements from a
single long chain.
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


# ---------------------------------------------------------------------------
# Diffusion-time grid
# ---------------------------------------------------------------------------

def make_sigma_grid(max_sigma, n_sigma, sigma_min=1.0):
    """Log-spaced diffusion-time grid in [sigma_min, max_sigma].

    A geometric grid samples the small-sigma (UV) and large-sigma (IR)
    regimes evenly in log space, which is where the centered
    finite-difference D_S(sigma) is read off.
    """
    return np.geomspace(sigma_min, max_sigma, n_sigma)


# ---------------------------------------------------------------------------
# Worker for parallel configurations
# ---------------------------------------------------------------------------

def _worker(cfg_id, n_simplices, n_therm, sweeps_between,
            n_walks, sigmas, sweep_cb=None, phase_cb=None):
    """Run one independent configuration: build spacetime, thermalize,
    then read the dual-graph return probability P(sigma) from C++.

    Returns ``(cfg_id, P, N, avg_degree, elapsed)`` where ``P`` is the
    return-probability curve over ``sigmas`` (or None for an empty/edgeless
    dual graph).  The GIL is released during sweep(), so multiple threads
    get real C++ parallelism without duplicating process memory.
    """
    _ph = lambda p, done=0, total=0: phase_cb(cfg_id, p, done, total) if phase_cb else None

    _ph("building")
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    # Toroid::build creates n_simplices/(d*(d+1)) = n_simplices/20 time
    # slabs at d=4, each only ~5 spatial vertices wide. Building the full
    # size directly yields a long thin tube (~1D dual graph), which traps
    # diffusion at D_S~1. Match volume_profile_phases.py: cap the initial
    # build at T=80 slabs (paper value) and let the Metropolis chain grow
    # the volume sideways to target via (2,8)/(8,2) moves.
    max_build = 80 * 20  # 80 time slabs x 20 simplices/slab in d=4
    st.build(min(n_simplices, max_build))
    target = st.getN41() if n_simplices <= max_build else n_simplices // 2
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)

    _ph("tuning")
    cdt.tune(progress=make_tune_cb(phase_cb, cfg_id))

    chunk = max(1, n_therm // 20)
    for start in range(0, n_therm, chunk):
        batch = min(chunk, n_therm - start)
        cdt.sweep(batch, progress=sweep_cb)
        _ph("thermalizing", start + batch, n_therm)

    chunk = max(1, sweeps_between // 20)
    for start in range(0, sweeps_between, chunk):
        batch = min(chunk, sweeps_between - start)
        cdt.sweep(batch, progress=sweep_cb)
        _ph("decorrelating", start + batch, sweeps_between)

    _ph("diffusing")
    t0 = time.time()
    sg = st.getDualGraph()
    N = sg.nNodes()
    if N == 0 or sg.nEdges() == 0:
        return cfg_id, None, 0, 0.0, 0.0

    # Heat-kernel return probability on the dual graph, averaged over
    # min(n_walks, N) random start vertices.  The Krylov-Lanczos diffusion
    # and Hutchinson trace estimate run inside SparseGraph; seeding with
    # cfg_id keeps each configuration's start-vertex subsample reproducible.
    P = sg.returnProbability(list(sigmas), m=n_walks, seed=cfg_id)
    avg_nbr = 2.0 * sg.nEdges() / N
    elapsed = time.time() - t0
    return cfg_id, P, N, avg_nbr, elapsed


# ---------------------------------------------------------------------------
# Spectral dimension extraction
# ---------------------------------------------------------------------------

def compute_spectral_dimension(sigmas, P):
    """D_S(sigma) = -2 d(log P) / d(log sigma) via the C++ centered
    finite-difference (``SparseGraph.spectralDimensionCurve``).

    Returns ``(sigma_values, D_S_values)`` over the finite entries of the
    curve (NaNs, where P <= 0, are dropped).
    """
    sigmas = np.asarray(sigmas, dtype=float)
    ds = np.asarray(
        tessera.SparseGraph.spectralDimensionCurve(list(sigmas), list(P)))
    finite = np.isfinite(ds)
    return sigmas[finite], ds[finite]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    monitor = MemoryMonitor()
    parser = argparse.ArgumentParser(
        description="Spectral dimension D_S(sigma) measurement "
                    "(Figs 9-10 of hep-th/0505154)")
    # Defaults sized so D_S rises cleanly through ~3 toward ~4 at long
    # sigma while staying cheaper than the full paper reproduction. For
    # full paper reproduction (Figs 9-10 of AJL 2005) use
    #   --n-simplices 160000 --n-therm 500 --n-configs 50
    #   --n-walks 100 --max-sigma 500 --sweeps-between 50.
    parser.add_argument("--n-simplices", type=int, default=80000,
                        help="Initial number of simplices")
    parser.add_argument("--n-therm", type=int, default=800,
                        help="Thermalization sweeps")
    parser.add_argument("--n-configs", type=int, default=20,
                        help="Number of independent configurations to average")
    parser.add_argument("--n-walks", type=int, default=50,
                        help="Random start vertices per configuration "
                             "(Hutchinson subsample for the trace estimate)")
    parser.add_argument("--max-sigma", type=float, default=400.0,
                        help="Maximum diffusion time sigma")
    parser.add_argument("--sigma-min", type=float, default=1.0,
                        help="Minimum diffusion time sigma")
    parser.add_argument("--n-sigma", type=int, default=60,
                        help="Number of log-spaced sigma grid points")
    parser.add_argument("--sweeps-between", type=int, default=80,
                        help="Sweeps between configurations for decorrelation")
    parser.add_argument("--workers", type=int,
                        default=min(os.cpu_count() or 1, 8),
                        help="Parallel worker processes (default: min(cpus, 8))")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to write the figure to; without it the "
                             "figure is displayed interactively")
    args = parser.parse_args()

    n_workers = max(1, args.workers)
    sigmas = make_sigma_grid(args.max_sigma, args.n_sigma, args.sigma_min)

    print("=" * 64)
    print("  Spectral Dimension Measurement via Dual-Graph Heat Kernel")
    print("  Reproduces Figs 9-10, Ambjorn, Jurkiewicz, Loll (2005)")
    print("  Parameters: k0=2.2, Delta=0.6")
    print(f"  Configs: {args.n_configs}, walks/config: {args.n_walks}, "
          f"sigma: [{args.sigma_min:g}, {args.max_sigma:g}] ({args.n_sigma} pts)")
    print(f"  Workers: {n_workers} (threads, shared memory)")
    print("=" * 64)

    t_total = time.time()

    all_P = []

    sweeps_per_cfg = args.n_therm + args.sweeps_between
    total_sweeps = args.n_configs * sweeps_per_cfg

    # Threads share address space — no memory duplication.
    # The GIL is released inside sweep(), so threads get real C++ parallelism.
    progress = ProgressDisplay(args.n_configs, total_sweeps,
                                   memory_monitor=monitor)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _worker, cfg, args.n_simplices, args.n_therm,
                args.sweeps_between, args.n_walks, sigmas,
                progress.on_sweep, progress.on_phase
            ): cfg
            for cfg in range(args.n_configs)
        }
        for future in as_completed(futures):
            cfg_id, P, N, avg_nbr, elapsed = future.result()
            if P is not None:
                all_P.append(P)
                _, ds = compute_spectral_dimension(sigmas, P)
                if len(ds):
                    n_tail = max(1, len(ds) // 5)
                    ds_small = float(np.mean(ds[:n_tail]))
                    ds_large = float(np.mean(ds[-n_tail:]))
                    info = f"N₄≈{N:,}  D_S(small)={ds_small:.2f}  D_S(large)={ds_large:.2f}"
                else:
                    info = f"N₄≈{N:,}"
            else:
                info = "empty"
            progress.on_item_done(cfg_id, info)

    progress.finish()

    if not all_P:
        print("No data collected.")
        return

    print(f"\nCollected {len(all_P)} configurations")

    # Average the per-configuration return-probability curves, then extract
    # the spectral dimension from the grand-mean curve.
    avg_P = np.mean(np.asarray(all_P), axis=0)
    sigma_vals, D_S_vals = compute_spectral_dimension(sigmas, avg_P)

    # --- Plot (Fig 9/10 style) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(sigma_vals, D_S_vals, "b-", linewidth=1.5, label="Measured")
    sigma_fit = np.linspace(10, args.max_sigma, 200)
    D_S_fit = 4.02 - 119.0 / (54.0 + sigma_fit)
    ax1.plot(sigma_fit, D_S_fit, "r--", linewidth=1.5,
             label=r"Fit: $D_S = 4.02 - \frac{119}{54+\sigma}$")
    ax1.set_xlabel(r"Diffusion time $\sigma$", fontsize=13)
    ax1.set_ylabel(r"$D_S(\sigma)$", fontsize=13)
    ax1.set_title(r"Spectral dimension $D_S(\sigma)$"
                  "\n(cf. Fig 9, Ambjorn, Jurkiewicz, Loll,\n"
                  r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    ax2.plot(sigma_vals, D_S_vals, "k-", linewidth=1.5, label="Measured")
    ax2.plot(sigma_fit, D_S_fit, "g--", linewidth=1,
             label=r"$4.02 - 119/(54+\sigma)$")
    ax2.axhline(y=4.02, color="gray", linestyle=":", alpha=0.5,
                label=r"$D_S(\infty)=4.02$")
    ax2.axhline(y=1.80, color="gray", linestyle=":", alpha=0.5,
                label=r"$D_S(0)\approx 1.80$")
    ax2.set_xlabel(r"$\sigma$", fontsize=13)
    ax2.set_ylabel(r"$D_S$", fontsize=13)
    ax2.set_title(r"Spectral dimension with reference values"
                  "\n(cf. Fig 10, Ambjorn, Jurkiewicz, Loll,\n"
                  r"$\it{Reconstructing\ the\ Universe}$, 2005)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)

    fig.tight_layout()

    if len(sigma_vals) > 0:
        n_tail = max(1, len(D_S_vals) // 5)
        D_S_large = np.mean(D_S_vals[-n_tail:])
        n_head = max(1, len(D_S_vals) // 5)
        D_S_small = np.mean(D_S_vals[:n_head])
        print(f"\nResults:")
        print(f"  D_S(large scale)  = {D_S_large:.2f}  (paper: 4.02 +/- 0.1)")
        print(f"  D_S(small scale)  = {D_S_small:.2f}  (paper: 1.80 +/- 0.25)")

    print(f"Total elapsed: {time.time()-t_total:.1f}s")

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
