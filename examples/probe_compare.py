"""Head-to-head comparison: does raising max_build help D_S?
And does raising target N4 help?

Each scenario runs a single config with the same compute budget.
We measure D_S at fixed sigmas.

Scenarios:
  A: max_build=1600, target=8000   (current default ratio)
  B: max_build=8000, target=8000   (init == target — no growth needed)
  C: max_build=1600, target=20000  (5x bigger N4 budget)

A and C share the "narrow start" but differ in target.  B has
"wide start" but small target.  C has "narrow start" and big target.

D_S(sigma) is read off the dual-graph heat-kernel spectral-dimension
curve (``tessera.SparseGraph``), the same estimator spectral_dimension.py
and the modularity sweep use.
"""
import argparse
import time

import numpy as np

import tessera

# Log-spaced sigma grid resolution for the D_S(sigma) curve.
N_SIGMA = 64


def ds_at_targets(st, sigma_targets, max_sigma, n_walks, seed):
    """D_S at each target sigma, read off the dual-graph spectral-
    dimension curve.  Returns ``(sg, [D_S or None per target])``;
    ``(None, [...])`` for an empty / edgeless dual graph."""
    sg = st.getDualGraph()
    if sg.nNodes() < 2 or sg.nEdges() == 0:
        return None, [None] * len(sigma_targets)
    sigmas = np.geomspace(1.0, float(max_sigma), N_SIGMA)
    P = sg.returnProbability(list(sigmas), m=n_walks, seed=seed)
    ds = np.asarray(
        tessera.SparseGraph.spectralDimensionCurve(list(sigmas), list(P)))
    out = []
    for s_t in sigma_targets:
        idx = int(np.argmin(np.abs(sigmas - s_t)))
        d = ds[idx]
        out.append(float(d) if np.isfinite(d) else None)
    return sg, out


def slice_widths(st):
    times = list(st.getTimeSlices())
    return [len(st.getVerticesAtTime(t)) for t in times]


def run(label, max_build, target, n_therm, max_sigma, n_walks, seed):
    sig_obj = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig_obj)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                          tessera.Toroid())
    initial = min(target, max_build)  # build at most max_build simplices
    st.build(initial)

    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    t0 = time.time()
    cdt.tune()
    t_tune = time.time() - t0

    t0 = time.time()
    cdt.sweep(n_therm)
    t_therm = time.time() - t0

    sigma_targets = [4, 12, 30, 60]
    sg, ds = ds_at_targets(st, sigma_targets, max_sigma, n_walks, seed)
    if sg is None:
        print(f"[{label}] empty triangulation")
        return
    widths = slice_widths(st)
    n_layers = len(widths)
    n4 = st.getTopSimplexCount()
    n41 = st.getN41()

    print(f"\n=== {label} ===")
    print(f"  setup: max_build={max_build} target={target} n_therm={n_therm}")
    print(f"  build: initial={initial} simplices, {n_layers} time layers (T)")
    print(f"  final: N4={n4}  N41={n41}  layers={n_layers}")
    print(f"  slice widths: min={min(widths)} mean={sum(widths)/len(widths):.1f} "
          f"max={max(widths)}")
    ds_str = "  ".join(f"DS@{s}={d:.2f}" if d is not None else f"DS@{s}=N/A"
                       for s, d in zip(sigma_targets, ds))
    print(f"  {ds_str}")
    print(f"  timing: tune={t_tune:.1f}s therm={t_therm:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--n-therm", type=int, default=400)
    ap.add_argument("--max-sigma", type=float, default=80.0)
    ap.add_argument("--n-walks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    scenarios = {
        "A": dict(max_build=1600,  target=8000),   # current default ratio
        "B": dict(max_build=8000,  target=8000),   # init == target
        "C": dict(max_build=1600,  target=20000),  # bigger N4
    }
    if args.scenario == "all":
        keys = ["A", "B", "C"]
    else:
        keys = [args.scenario]
    for k in keys:
        s = scenarios[k]
        run(k, s["max_build"], s["target"], args.n_therm,
            args.max_sigma, args.n_walks, args.seed)


if __name__ == "__main__":
    main()
