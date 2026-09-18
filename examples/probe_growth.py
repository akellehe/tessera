"""How fast does D_S(sigma) converge with thermalization?

Smaller, faster setup: target ~ 4000 N41 simplices.  Each checkpoint
prints volume, slice-width stats, and spectral-dimension samples at
specific sigmas (well below sqrt(N4) saturation).

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


def measure(st, n_walks, max_sigma, sigma_targets, seed=0):
    sg, ds_at = ds_at_targets(st, sigma_targets, max_sigma, n_walks, seed)
    if sg is None:
        return None
    return {
        "N4": st.getTopSimplexCount(),
        "N41": st.getN41(),
        "ds_at": ds_at,
        "widths": slice_widths(st),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-simplices", type=int, default=8000)
    ap.add_argument("--max-build", type=int, default=1600)
    ap.add_argument("--total-therm", type=int, default=4000)
    ap.add_argument("--n-checkpoints", type=int, default=12)
    ap.add_argument("--max-sigma", type=float, default=80.0)
    ap.add_argument("--n-walks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    sigma_targets = [4, 12, 30, 60]

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                          tessera.Toroid())
    initial_size = min(args.n_simplices, args.max_build)
    st.build(initial_size)
    target = (st.getN41() if args.n_simplices <= args.max_build
              else args.n_simplices // 2)
    print(f"build: N4={st.getTopSimplexCount()} target={target}", flush=True)

    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    cdt.tune()
    print(f"tune:  N4={st.getTopSimplexCount()} N41={st.getN41()}", flush=True)

    header = (f"{'sweeps':>7} {'N4':>5} {'N41':>5} {'ws_min':>6} "
              f"{'ws_mean':>7} {'ws_max':>6} "
              + " ".join(f"DS@{s:>3}" for s in sigma_targets)
              + f" {'sweep_s':>7}")
    print(header, flush=True)

    # Logarithmic checkpoints: 0, 25, 50, 100, 200, 400, 800, ...
    checkpoints = [0]
    s = 25
    while s <= args.total_therm:
        checkpoints.append(s)
        s *= 2
    if checkpoints[-1] != args.total_therm:
        checkpoints.append(args.total_therm)
    sweeps_done = 0

    def log_row(sweeps, m, dt):
        widths = m["widths"]
        ds_str = " ".join(f"{d:>5.2f}" if d is not None else "  N/A"
                          for d in m["ds_at"])
        print(f"{sweeps:>7} {m['N4']:>5} {m['N41']:>5} {min(widths):>6} "
              f"{sum(widths)/len(widths):>7.1f} {max(widths):>6} {ds_str} "
              f"{dt:>7.2f}", flush=True)

    m = measure(st, args.n_walks, args.max_sigma, sigma_targets, args.seed)
    if m:
        log_row(0, m, 0.0)

    for next_chk in checkpoints[1:]:
        batch = next_chk - sweeps_done
        t0 = time.time()
        cdt.sweep(batch)
        dt = time.time() - t0
        sweeps_done = next_chk
        m = measure(st, args.n_walks, args.max_sigma, sigma_targets, args.seed)
        if m is None:
            print(f"{sweeps_done:>7} (empty)", flush=True)
        else:
            log_row(sweeps_done, m, dt)


if __name__ == "__main__":
    main()
