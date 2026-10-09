"""Probe the geometry that spectral_dimension.py actually feeds into the
diffusion process.  Goal: validate or refute the narrow-tube hypothesis.

We measure:
  * Volume profile N3(tau) — how thick is each spatial slice?
  * Spatial-vertex count per slice — a thin tube has ~5 verts/slice.
  * Dual-graph degree distribution — a quasi-1D tube has mean degree ~2.
  * Spectral dimension at sigma -> 0 and large sigma at three checkpoints:
      (0) post-build, (1) post-tune, (2) post-therm.

The dual-graph diffusion + spectral-dimension extraction run in C++:
``st.getDualGraph()`` returns a :class:`tessera.SparseGraph` whose
``returnProbability`` / ``spectralDimensionCurve`` give the heat-kernel
return probability and D_S(sigma) curve.

Run it small first to sanity-check, then larger.  Resource-friendly:
single thread, single configuration.
"""
import argparse
import time

import numpy as np

import tessera

# Log-spaced sigma grid resolution for the D_S(sigma) curve.
N_SIGMA = 48


def dual_spectral_dimension(st, sigmas, n_walks, seed):
    """Heat-kernel D_S(sigma) curve on the dual graph via SparseGraph.

    Returns ``(sg, ds)`` where ``ds`` is the spectral-dimension curve
    aligned with ``sigmas`` (NaN where the return probability is
    non-positive), or ``(sg, None)`` for an empty / edgeless dual graph.
    """
    sg = st.getDualGraph()
    if sg.nNodes() == 0 or sg.nEdges() == 0:
        return sg, None
    P = sg.returnProbability(list(sigmas), m=n_walks, seed=seed)
    ds = np.asarray(
        tessera.SparseGraph.spectralDimensionCurve(list(sigmas), list(P)))
    return sg, ds


def slice_widths(st):
    """Return the number of vertices at each time slice."""
    times = list(st.getTimeSlices())
    return {t: len(st.getVerticesAtTime(t)) for t in times}


def volume_profile(st):
    """N4(t): per-slab top-simplex counts via the C++ VolumeProfile.

    ``VolumeProfile`` counts every top simplex by its minimum-time slice
    (``getTi()``); for a well-formed CDT this is identical to the old
    2-distinct-times filter, since every causal top simplex spans exactly
    two adjacent slices.  We drop empty interior slices so the reported
    stats still describe populated slabs.
    """
    vp = tessera.VolumeProfile()
    vp.compute(st)
    return [c for c in vp.getProfile() if c > 0]


def report(label, st, max_sigma=200.0, n_walks=20, seed=0):
    N = st.getTopSimplexCount()
    n41 = st.getN41()
    n32 = st.getN32()
    nverts = st.getVertexCount()
    profile = volume_profile(st)
    widths = slice_widths(st)

    sigmas = np.geomspace(1.0, float(max_sigma), N_SIGMA)
    sg, ds = dual_spectral_dimension(st, sigmas, n_walks, seed)
    if ds is None:
        print(f"[{label}] empty triangulation")
        return
    deg = np.array([sg.degree(i) for i in range(sg.nNodes())])
    finite = ds[np.isfinite(ds)]
    if finite.size:
        n_head = max(1, len(finite) // 5)
        n_tail = max(1, len(finite) // 5)
        ds_small = float(np.mean(finite[:n_head]))
        ds_large = float(np.mean(finite[-n_tail:]))
    else:
        ds_small = ds_large = float("nan")

    if profile:
        n4_min = min(profile)
        n4_max = max(profile)
        n4_mean = sum(profile) / len(profile)
        n_slabs = len(profile)
    else:
        n4_min = n4_max = n4_mean = n_slabs = 0
    if widths:
        w_min = min(widths.values())
        w_max = max(widths.values())
        w_mean = sum(widths.values()) / len(widths)
        n_layers = len(widths)
    else:
        w_min = w_max = w_mean = n_layers = 0

    print(f"\n=== {label} ===")
    print(f"  N4_total={N}  N41={n41}  N32={n32}  Nverts={nverts}")
    print(f"  Time layers={n_layers}  spatial-verts/slice min/mean/max="
          f"{w_min}/{w_mean:.1f}/{w_max}")
    print(f"  Slabs={n_slabs}  N4/slab min/mean/max="
          f"{n4_min}/{n4_mean:.1f}/{n4_max}")
    print(f"  Dual-graph degree min/mean/max="
          f"{deg.min()}/{deg.mean():.2f}/{deg.max()}")
    print(f"  D_S(small sigma~{sigmas[0]:.0f})={ds_small:.2f}  "
          f"D_S(large sigma~{sigmas[-1]:.0f})={ds_large:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-simplices", type=int, default=8000)
    ap.add_argument("--max-build", type=int, default=1600,
                    help="Cap initial build at this many simplices "
                         "(80*20=1600 is current default).")
    ap.add_argument("--n-therm", type=int, default=200)
    ap.add_argument("--max-sigma", type=float, default=200.0)
    ap.add_argument("--n-walks", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Probe: n_simplices={args.n_simplices} max_build={args.max_build} "
          f"n_therm={args.n_therm}")

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                          tessera.Toroid())
    st.build(min(args.n_simplices, args.max_build))
    target = st.getN41() if args.n_simplices <= args.max_build else args.n_simplices // 2
    print(f"Initial build done.  N4={st.getTopSimplexCount()} "
          f"N41={st.getN41()}  target N41={target}")
    report("post-build", st, args.max_sigma, args.n_walks, args.seed)

    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    t0 = time.time()
    cdt.tune()
    print(f"\nTune done in {time.time()-t0:.1f}s")
    report("post-tune", st, args.max_sigma, args.n_walks, args.seed)

    t0 = time.time()
    chunk = max(1, args.n_therm // 10)
    for start in range(0, args.n_therm, chunk):
        batch = min(chunk, args.n_therm - start)
        cdt.sweep(batch)
        if (start // chunk) % 2 == 0:
            print(f"  therm progress: {start+batch}/{args.n_therm} sweeps; "
                  f"N4={st.getTopSimplexCount()} N41={st.getN41()}  "
                  f"elapsed={time.time()-t0:.1f}s")
    print(f"Thermalization done in {time.time()-t0:.1f}s")
    report("post-therm", st, args.max_sigma, args.n_walks, args.seed)


if __name__ == "__main__":
    main()
