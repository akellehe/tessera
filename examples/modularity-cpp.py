#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""Modularity sweep on CDT spacetimes via transactional Pachner moves.

A C++-driven counterpart to ``examples/modularity.py``.  Instead of
walking an abstract synthetic graph, this driver:

  1. Builds a CDT spacetime per top-simplex dimension d (default
     d ∈ {2, 3, 4, 5}).
  2. Optionally runs ``cdt.tune()`` + a few Monte Carlo sweeps to
     thermalize.
  3. Runs :class:`tessera.ModularityOptimizer`, which drives the
     spacetime via Pachner moves with Q-direction acceptance.  Q is
     measured on the spacetime's vertex/edge 1-skeleton; D_S is
     measured on the dual graph (top simplices).
  4. Plots D_S vs Q for each dimension d on a single overlay
     (one curve per d, viridis colormap).

See ``docs/source/modularity-plan.md`` for the design rationale.

Reference:
  examples/modularity.py — the original Python-only implementation.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import matplotlib
# Force a non-interactive backend before pyplot import: the macOS
# native backend (used by default on darwin) crashes during process
# shutdown when this script saves the figure and exits without
# entering a GUI event loop.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tessera

# Reuse the TTY-aware ProgressBar from modularity.py — same look as
# the original Python-only sweep so output is uniform across the two
# scripts.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from modularity import ProgressBar  # noqa: E402


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_parser():
    p = argparse.ArgumentParser(
        description="Modularity sweep on CDT spacetimes via "
                    "transactional Pachner moves.")
    p.add_argument("--dimensions", type=int, nargs="+",
                   default=[2, 3, 4, 5],
                   help="Top-simplex dimensions to sweep.  One overlay "
                        "curve per dimension.")
    p.add_argument("--n-simplices", type=int, default=200,
                   help="Initial build size (passed to "
                        "Spacetime.build).  Toroid yields d*(d+1) "
                        "simplices/slab so the actual count depends "
                        "on slab quantization.")
    p.add_argument("--cdt-thermalize", type=int, default=0,
                   help="Optional CDT Monte Carlo sweeps after "
                        "tuning, before the modularity optimizer "
                        "takes over.  0 = skip thermalization.")
    p.add_argument("--target-n-modules", type=int, default=4,
                   help="M (modulo partition).  Vertex labels are "
                        "implicit: label(v) = v.id %% M.")
    p.add_argument("--direction", choices=["up", "down", "both"],
                   default="both",
                   help="Sweep direction(s).")
    p.add_argument("--target-dq", type=float, default=0.05,
                   help="Q increment between recorded measurements.")
    p.add_argument("--max-iterations", type=int, default=400,
                   help="Hard cap on iterations per direction.")
    p.add_argument("--epsilon-q-max", type=float, default=0.01,
                   help="Up-sweep early-exit tolerance.")
    p.add_argument("--n-diffusion-walks", type=int, default=80,
                   help="Diffusion walks per D_S measurement.")
    p.add_argument("--max-sigma", type=float, default=200.0,
                   help="Max diffusion time per measurement.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed.")
    p.add_argument("--save", type=str, default="./modularity-cpp.png",
                   help="Output plot path.")
    return p


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_cdt(d, n_simplices):
    sig = tessera.Signature(d, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    st.build(n_simplices)
    target = max(st.getN41(), 1)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    return cdt, st


def _make_config(args):
    cfg = tessera.ModularityOptimizerConfig()
    cfg.targetDq = args.target_dq
    cfg.maxIterations = args.max_iterations
    cfg.nDiffusionWalks = args.n_diffusion_walks
    cfg.maxSigma = args.max_sigma
    cfg.epsilonQMax = args.epsilon_q_max
    cfg.targetNModules = args.target_n_modules
    return cfg


def _sweep_with_bar(opt, cdt, st, direction, max_iter, label_prefix, M):
    """Run one sweep direction with a TTY-aware progress bar (matches
    the look of the original modularity.py output)."""
    desc = f"{label_prefix} {'↑ up  ' if direction == 'up' else '↓ down'}"
    initial_q = st.modularityOnSkeleton(M)
    with ProgressBar(total=max_iter, desc=desc) as bar:
        # Initial state line.
        bar.update(current=0, force=True,
                   Q=f"{initial_q:.4f}", dQ="+0.000",
                   N4=st.getTopSimplexCount(),
                   N0=st.getVertexCount(),
                   meas=1)

        def cb(it, mi, q, n_meas):
            bar.update(
                current=it,
                Q=f"{q:.4f}",
                dQ=f"{q - initial_q:+.3f}",
                N4=st.getTopSimplexCount(),
                N0=st.getVertexCount(),
                meas=n_meas,
                ok=opt.getNAccepted(),
                ko=opt.getNRolledBack(),
            )

        ms = opt.sweep(cdt, direction, progress=cb)
        bar.update(force=True,
                   Q=f"{ms[-1].Q:.4f}",
                   dQ=f"{ms[-1].Q - initial_q:+.3f}",
                   N4=st.getTopSimplexCount(),
                   N0=st.getVertexCount(),
                   meas=len(ms),
                   ok=opt.getNAccepted(),
                   ko=opt.getNRolledBack())

    logger.info(
        "  %s done: %d iter, %d accepted, %d rolled back, "
        "%d no-eligible-move, %d measurements; "
        "Q: %.4f → %.4f (Δ%+.4f)",
        direction, ms[-1].iter,
        opt.getNAccepted(), opt.getNRolledBack(),
        opt.getNNoMove(), len(ms),
        ms[0].Q, ms[-1].Q, ms[-1].Q - ms[0].Q,
    )
    return ms


def _run_one_d(d, args):
    """Build a d-dim CDT spacetime, optionally thermalize, then run the
    modularity optimizer up + down (depending on --direction)."""
    cdt, st = _build_cdt(d, args.n_simplices)
    label = (f"CDT(d={d}, Toroid) on {st.getTopSimplexCount()} simplices, "
             f"{st.getVertexCount()} vertices")
    logger.info("--- d=%d ---", d)
    logger.info("Initial: %s", label)

    if args.cdt_thermalize > 0:
        logger.info("Tuning + thermalizing %d CDT sweeps",
                    args.cdt_thermalize)
        cdt.tune()
        cdt.sweep(args.cdt_thermalize)
        logger.info("Thermalized: %d simplices, %d vertices",
                    st.getTopSimplexCount(), st.getVertexCount())

    cfg = _make_config(args)
    opt = tessera.ModularityOptimizer(cfg, seed=args.seed + d)
    M = args.target_n_modules
    measurements = []
    if args.direction in ("up", "both"):
        measurements.extend(_sweep_with_bar(
            opt, cdt, st, "up", args.max_iterations, f"d={d}", M))
    if args.direction in ("down", "both"):
        # Fresh CDT so the down sweep starts from the same initial
        # state, not the up-swept end state.
        cdt2, st2 = _build_cdt(d, args.n_simplices)
        if args.cdt_thermalize > 0:
            cdt2.tune()
            cdt2.sweep(args.cdt_thermalize)
        opt2 = tessera.ModularityOptimizer(
            cfg, seed=args.seed + d + 10000)
        measurements.extend(_sweep_with_bar(
            opt2, cdt2, st2, "down", args.max_iterations, f"d={d}", M))
    return measurements, label


def _plot_dimension_sweep(measurements_by_d, direction, save_path):
    if not measurements_by_d:
        return
    ds = sorted(measurements_by_d.keys())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=ds[0], vmax=ds[-1] if len(ds) > 1
                         else ds[0] + 1)

    fig, (ax_l, ax_s) = plt.subplots(1, 2, figsize=(13, 5))
    for d in ds:
        ms = measurements_by_d[d]
        if not ms:
            continue
        Q = np.array([m.Q for m in ms])
        Dl = np.array([m.dsLarge for m in ms])
        Ds = np.array([m.dsSmall for m in ms])
        order = np.argsort(Q)
        color = cmap(norm(d))
        ax_l.plot(Q[order], Dl[order], color=color, marker="o",
                  linestyle="-", linewidth=1.3, markersize=4,
                  label=f"d={d}")
        ax_s.plot(Q[order], Ds[order], color=color, marker="s",
                  linestyle="-", linewidth=1.3, markersize=4,
                  label=f"d={d}")

    for ax, ds_label in [(ax_l, r"$D_S$ (large $\sigma$)"),
                         (ax_s, r"$D_S$ (small $\sigma$)")]:
        ax.set_xlabel(r"Newman-Girvan $Q$ (M-modulo partition)",
                      fontsize=13)
        ax.set_ylabel(ds_label, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle(f"Spectral dimension vs modularity, "
                 f"d={ds[0]}..{ds[-1]} ({direction}-directed Pachner sweep)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def main():
    args = _make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    measurements_by_d = {}
    for d in args.dimensions:
        try:
            ms, label = _run_one_d(d, args)
        except Exception as exc:
            logger.warning("d=%d: failed (%s); skipping", d, exc)
            continue
        measurements_by_d[d] = ms
        logger.info("d=%d: %d measurements, Q range [%.3f, %.3f]",
                    d, len(ms),
                    min(m.Q for m in ms),
                    max(m.Q for m in ms))

    elapsed = time.time() - t0
    n_total = sum(len(v) for v in measurements_by_d.values())
    logger.info("Done in %.1fs across %d dimensions, %d measurements total",
                elapsed, len(measurements_by_d), n_total)

    _plot_dimension_sweep(measurements_by_d, args.direction, args.save)
    logger.info("Saved %s", args.save)


if __name__ == "__main__":
    main()
