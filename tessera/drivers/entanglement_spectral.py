# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers.entanglement_spectral -- the spectral dimension of the
qubit information network: the dimension a diffusing walker sees.

Definitions
-----------
* The network is the one of ``tessera.drivers.entanglement_complex``: n
  qubits named A, B, C, ... evolved by one SWAP^alpha interaction per time
  slice on a pair drawn by the schedule, on the mixed engine (the global
  density matrix) or the pure one (a state vector). S(X) is the von Neumann
  entropy in nats of the one-qubit reduced state of X and
  I(X:Y) = S(X) + S(Y) - S(XY) the mutual information of the pair.
* The walk. The walker diffuses on a graph with the qubits as vertices. With
  the mutual-information walk (the default) the edge (X, Y) has weight
  w(X,Y) = I(X:Y) for every pair above the floor, the convention of
  ``tessera.quantum.holography.EmergentGraph``: the weighted Laplacian is
  L = D - W with D the diagonal of weighted degrees. With the skeleton walk
  the graph is the unweighted 1-skeleton of the entanglement complex at
  scale r, an edge of weight 1 for every pair whose length, in the chosen
  length mode of ``entanglement_complex.target_lengths`` (scaled to mean
  edge length 1), is at most r.
* The return probability P(sigma) = (1/n) Tr exp(-sigma L) is the
  probability that a walker started on a uniformly random vertex is found
  there again after diffusion time sigma; it is computed exactly, as the
  full trace, with a Krylov space as large as the graph (the default depth
  of 30 of ``EmergentGraph.return_probability`` is far off at large sigma on
  these graphs). The spectral dimension is
  D_S(sigma) = -2 d ln P / d ln sigma: on a flat d-dimensional space
  P ~ sigma^(-d/2) and D_S = d. On a graph P -> 1 as sigma -> 0, so D_S
  rises from 0; at sigma of the order of the inverse edge weight it
  overshoots the dimension of the lattice the graph approximates (the
  infinite chain peaks at 1.21, the square grid at about 2.4); then it
  settles on the dimension and, on a finite graph, falls back to 0 once
  P -> (components)/n. Three readings are reported. The peak of the curve
  with its sigma. D_half, the curve at the half-decay time sigma_half where
  ln P is halfway between 0 and ln(1/n): a long chain gives 0.99 and square
  grids 1.9 to 2.0 there, so it reads the dimension where the peak
  overshoots, and on the small graphs of a qubit network it is the reading
  to compare between schedules; it does not exist (nan) while the graph has
  more than sqrt(n) components, since P then never falls that far. And the
  Ambjorn-Loll form
  D_S(sigma) = D_inf - C / (B + sigma) of ``AmbjornLollFit``, fitted on the
  rising window up to the peak, where its form applies, reported with its
  reduced chi squared for comparison with the CDT literature.
* Time. One graph per slice, so the curve and its readings are functions of
  the slice index t; ``--pairs both`` runs the all-pairs and the chain
  schedules on the same seeds and overlays them.

Figure
------
Three panels: P(sigma) on log axes, D_S(sigma) with D_half marked and the
rising-window fit, both for the final slice of each schedule with the
earlier slices faint; and the peak and D_half against the slice index.
"""

import argparse
import itertools
import math
from fractions import Fraction

import numpy as np

from tessera.drivers import entanglement_complex as ec

WALK_MODES = ("mi", "skeleton")
PAIR_MODES = ("all", "chain", "both")
SIGMA_MIN, SIGMA_MAX, SIGMA_COUNT = 1e-2, 1e3, 60


def holography():
    """The holography submodule of the quantum subsystem."""
    from tessera import quantum
    holo = getattr(quantum, "holography", None)
    if holo is None:
        from tessera import _tessera
        holo = _tessera.quantum.holography
    return holo


def sigma_grid(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, count=SIGMA_COUNT):
    """`count` diffusion times spaced evenly in log sigma."""
    if not 0 < sigma_min < sigma_max or count < 4:
        raise ValueError("the sigma grid needs 0 < sigma_min < sigma_max and at least 4 points")
    return np.geomspace(sigma_min, sigma_max, count)


# ============================================================ graphs

def mi_edges(MI, floor):
    """(X, Y, I(X:Y)) for every pair above the floor."""
    MI = np.asarray(MI, dtype=float)
    return [(i, j, float(MI[i, j])) for i, j in itertools.combinations(range(len(MI)), 2)
            if MI[i, j] > floor]


def skeleton_edges(MI, S, floor, mode, scale):
    """(X, Y, 1) for every pair whose length in `mode` (mean edge length 1)
    is at most `scale`."""
    D, W = ec.target_lengths(MI, mode, floor, S)
    if D is None:
        return []
    return [(i, j, 1.0) for i, j in itertools.combinations(range(len(W)), 2)
            if W[i, j] > 0 and D[i, j] <= scale]


def graph(n, edges):
    """The ``EmergentGraph`` on n vertices with the weighted edges."""
    return holography().EmergentGraph.from_weighted_edges(int(n), [(int(i), int(j), float(w))
                                                                  for i, j, w in edges])


def return_probability(n, edges, sigmas):
    """P(sigma) = (1/n) Tr exp(-sigma L), exact: every vertex is a start and
    the Krylov space is as large as the graph; 1 everywhere without edges."""
    sigmas = [float(s) for s in sigmas]
    if not edges:
        return np.ones(len(sigmas))
    g = graph(n, edges)
    return np.asarray(g.return_probability(sigmas, int(n) + 1, int(n), 0), dtype=float)


def readings(n, sigmas, P):
    """The spectral-dimension curve of a return probability and its three
    readings: D_S by centred finite differences and smoothed by the local
    quadratic fit; the peak with its sigma; D_half at the half-decay time
    sigma_half (ln P halfway between 0 and ln(1/n)), interpolated on the
    grid; and the Ambjorn-Loll fit on the rising window up to the peak."""
    holo = holography()
    sigmas = np.asarray(sigmas, dtype=float)
    P = np.asarray(P, dtype=float)
    dS = np.asarray(holo.EmergentGraph.spectral_dimension(list(sigmas), list(P)), dtype=float)
    if len(sigmas) >= 5:
        smooth = np.asarray(holo.EmergentGraph.spectral_dimension_smoothed(
            list(sigmas), list(P), 5, 2), dtype=float)
    else:
        smooth = dS.copy()
    finite = np.isfinite(dS)
    peak = sigma_peak = d_half = sigma_half = float("nan")
    fit = {"D_inf": float("nan"), "C": float("nan"), "B": float("nan"), "chi2": float("nan"),
           "D_short": float("nan")}
    if finite.any() and n > 1:
        k = int(np.nanargmax(np.where(finite, dS, -np.inf)))
        peak, sigma_peak = float(dS[k]), float(sigmas[k])
        half = 0.5 * math.log(1.0 / n)
        lnP = np.log(np.clip(P, 1e-300, None))
        crossing = np.nonzero((lnP[:-1] >= half) & (lnP[1:] < half))[0]
        if len(crossing):
            i = int(crossing[0])
            frac = (lnP[i] - half) / (lnP[i] - lnP[i + 1])
            lns = np.log(sigmas)
            sigma_half = float(math.exp(lns[i] + frac * (lns[i + 1] - lns[i])))
            d_half = float(np.interp(math.log(sigma_half), lns[finite], dS[finite]))
        window = sigmas <= sigma_peak
        if window.sum() >= 4:
            f = holo.AmbjornLollFit.fit(list(sigmas[window]), list(dS[window]))
            fit = {"D_inf": float(f.d_infinity), "C": float(f.C), "B": float(f.B),
                   "chi2": float(f.chi_squared),
                   "D_short": (float(f.d_infinity - f.C / f.B) if f.B != 0 else float("nan"))}
    return {"sigmas": sigmas, "P": P, "dS": dS, "dS_smooth": smooth, "fit": fit,
            "peak": peak, "sigma_peak": sigma_peak, "D_half": d_half, "sigma_half": sigma_half}


def spectral_curve(n, edges, sigmas):
    """The return probability of the graph on the sigma grid and its
    readings (see ``readings``)."""
    sigmas = np.asarray(sigmas, dtype=float)
    out = readings(n, sigmas, return_probability(n, edges, sigmas))
    out["edges"] = len(edges)
    return out


def fitted_curve(fit, sigmas):
    """D_S(sigma) = D_inf - C / (B + sigma) on the grid."""
    sigmas = np.asarray(sigmas, dtype=float)
    return fit["D_inf"] - fit["C"] / (fit["B"] + sigmas)


# ============================================================ analysis

def analyse(result, sigmas, walk="mi", floor=1e-12, length_mode="log1p", scale=1.0):
    """One spectral curve per slice of a simulation result (with time slices)."""
    if walk not in WALK_MODES:
        raise ValueError("the walk must be one of %s" % (WALK_MODES,))
    slices = result["slices"]
    if slices is None:
        raise ValueError("the simulation must record time slices (timesteps > 0)")
    n = len(result["names"])
    curves = []
    for sl in slices:
        if walk == "mi":
            edges = mi_edges(sl["MI"], floor)
        else:
            edges = skeleton_edges(sl["MI"], sl["S"], floor, length_mode, scale)
        c = spectral_curve(n, edges, sigmas)
        c["t"] = int(sl["t"])
        curves.append(c)
    return {"n": n, "names": result["names"], "state": result.get("state", "mixed"),
            "walk": walk, "floor": floor, "length_mode": length_mode, "scale": scale,
            "sigmas": np.asarray(sigmas, dtype=float), "curves": curves, "final": curves[-1],
            "T": int(slices[-1]["t"]), "interaction": result["interaction"]}


def run(n, timesteps, seed, pair_mode, state, alpha, sigmas, walk, floor, length_mode, scale,
        log=ec._silent):
    """Simulate one schedule and analyse every slice."""
    result = ec.simulate(n, None, alpha, seed, "global", timesteps, pair_mode, False, log, state)
    rep = analyse(result, sigmas, walk, floor, length_mode, scale)
    rep["pair_mode"] = pair_mode
    return rep


def print_report(reps, log=print):
    for rep in reps:
        f = rep["final"]
        walk = ("mutual information as edge weight" if rep["walk"] == "mi" else
                "1-skeleton at scale %g of the %s length" % (rep["scale"], rep["length_mode"]))
        log("SPECTRAL DIMENSION (%d qubits, %s schedule, %s global state; walk: %s)"
            % (rep["n"], rep["pair_mode"], rep["state"], walk))
        log("  sigma grid: %d points from %g to %g" % (len(rep["sigmas"]), rep["sigmas"][0],
                                                       rep["sigmas"][-1]))
        for c in rep["curves"]:
            log("  slice %3d  edges %3d  peak D_S = %.3f at sigma = %.3g   D_half = %.3f at "
                "sigma_half = %.3g   rising-window fit: D_inf = %.3f  D_short = %.3f  "
                "chi2 = %.3g"
                % (c["t"], c["edges"], c["peak"], c["sigma_peak"], c["D_half"],
                   c["sigma_half"], c["fit"]["D_inf"], c["fit"]["D_short"], c["fit"]["chi2"]))
        log("  final slice t = %d: D_S(sigma) =" % rep["T"])
        for s, p, d in zip(f["sigmas"], f["P"], f["dS"]):
            log("    sigma = %9.4g   P = %.6f   D_S = %s" % (s, p, "%.4f" % d if math.isfinite(d)
                                                              else "nan"))


# ============================================================ figure

def draw(reps, title, save=None):
    import matplotlib.pyplot as plt
    from tessera.drivers import baryon_poles as bp

    fig = plt.figure(figsize=(17, 5.8), facecolor=bp.SURFACE)
    grid = fig.add_gridspec(1, 3, wspace=0.28, left=0.05, right=0.98, top=0.84, bottom=0.13)
    axes = [fig.add_subplot(grid[k]) for k in range(3)]
    for ax in axes:
        bp.style_axis(ax)
    for r, rep in enumerate(reps):
        colour = ec.CATEGORICAL[(2 * r) % len(ec.CATEGORICAL)]
        label = "%s pairs" % rep["pair_mode"]
        curves = rep["curves"]
        for c in curves[:-1]:
            axes[0].plot(c["sigmas"], c["P"], color=colour, alpha=0.12, linewidth=0.8)
            axes[1].plot(c["sigmas"], c["dS"], color=colour, alpha=0.12, linewidth=0.8)
        f = rep["final"]
        axes[0].plot(f["sigmas"], f["P"], color=colour, linewidth=2.0, label=label)
        axes[1].plot(f["sigmas"], f["dS"], color=colour, linewidth=2.0, label=label)
        if math.isfinite(f["fit"]["B"]) and f["fit"]["B"] != 0:
            rising = f["sigmas"] <= f["sigma_peak"]
            axes[1].plot(f["sigmas"][rising], fitted_curve(f["fit"], f["sigmas"][rising]),
                         color=colour, linewidth=1.2, linestyle="--",
                         label="rising-window fit: D_inf = %.2f" % f["fit"]["D_inf"])
        if math.isfinite(f["sigma_half"]):
            axes[1].scatter([f["sigma_half"]], [f["D_half"]], s=60, color=colour,
                            edgecolor=bp.INK, zorder=4,
                            label="D_half = %.2f at sigma_half = %.3g" % (f["D_half"],
                                                                         f["sigma_half"]))
        t = [c["t"] for c in curves]
        axes[2].plot(t, [c["peak"] for c in curves], color=colour, linewidth=1.2,
                     linestyle=":", label="%s: peak" % label)
        axes[2].plot(t, [c["D_half"] for c in curves], color=colour, linewidth=2.0,
                     label="%s: D_half" % label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("diffusion time sigma", color=bp.INK_MUTED)
    axes[0].set_ylabel("return probability P(sigma)", color=bp.INK_MUTED)
    axes[0].set_title("return probability, final slice (earlier slices faint)", loc="left",
                      color=bp.INK, fontsize=10)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("diffusion time sigma", color=bp.INK_MUTED)
    axes[1].set_ylabel("spectral dimension D_S(sigma)", color=bp.INK_MUTED)
    axes[1].set_title("spectral dimension, final slice", loc="left", color=bp.INK, fontsize=10)
    axes[2].set_xlabel("slice t", color=bp.INK_MUTED)
    axes[2].set_ylabel("dimension", color=bp.INK_MUTED)
    axes[2].set_title("the readings against the slice", loc="left", color=bp.INK, fontsize=10)
    for ax in axes:
        ax.legend(frameon=False, labelcolor=bp.INK_MUTED, fontsize=8)
    fig.suptitle(title, x=0.02, ha="left", color=bp.INK)
    if save:
        fig.savefig(save, dpi=150, facecolor=bp.SURFACE)
    return fig


# ============================================================ command line

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The spectral dimension of the qubit information network: the "
                    "dimension a walker diffusing on the mutual-information graph sees, "
                    "slice by slice.")
    parser.add_argument("--qubits", type=int, default=8, help="number of qubits (default 8)")
    parser.add_argument("--timesteps", type=int, default=16,
                        help="interactions, one per time slice (default 16)")
    parser.add_argument("--seed", type=int, default=0, help="seed (default 0)")
    parser.add_argument("--swap-power", default="1/2",
                        help="alpha of the SWAP^alpha interaction, a fraction (default 1/2)")
    parser.add_argument("--state", choices=ec.STATE_MODES, default="mixed",
                        help="the engine: the global density matrix, or a pure global state "
                             "(default mixed)")
    parser.add_argument("--pairs", choices=PAIR_MODES, default="both",
                        help="the schedule's pairs: all pairs, the nearest neighbours of an "
                             "open chain, or both overlaid on the same seeds (default both)")
    parser.add_argument("--walk", choices=WALK_MODES, default="mi",
                        help="mi: edge weight I(X:Y); skeleton: the unweighted 1-skeleton of "
                             "the complex at --scale (default mi)")
    parser.add_argument("--length", choices=ec.LENGTH_MODES, default="log1p",
                        help="length mode of the skeleton walk (default log1p)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale r of the skeleton walk, in units of the mean edge length "
                             "(default 1)")
    parser.add_argument("--mi-floor", type=float, default=1e-12,
                        help="a pair with I at or below this has no edge (default 1e-12)")
    parser.add_argument("--sigma-min", type=float, default=SIGMA_MIN,
                        help="smallest diffusion time (default %g)" % SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=SIGMA_MAX,
                        help="largest diffusion time (default %g)" % SIGMA_MAX)
    parser.add_argument("--sigma-count", type=int, default=SIGMA_COUNT,
                        help="points of the log-spaced sigma grid (default %d)" % SIGMA_COUNT)
    parser.add_argument("--verbose", action="store_true",
                        help="print the simulations' own slice-by-slice reports too")
    parser.add_argument("--save", default=None, help="write the figure to this file")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args(argv)

    if args.timesteps < 1:
        parser.error("--timesteps must be at least 1")
    try:
        Fraction(args.swap_power)
    except (ValueError, ZeroDivisionError):
        parser.error("--swap-power must be a number such as 1/2 or 0.25")
    try:
        sigmas = sigma_grid(args.sigma_min, args.sigma_max, args.sigma_count)
    except ValueError as exc:
        parser.error(str(exc))

    modes = ("all", "chain") if args.pairs == "both" else (args.pairs,)
    reps = [run(args.qubits, args.timesteps, args.seed, mode, args.state, args.swap_power,
                sigmas, args.walk, args.mi_floor, args.length, args.scale,
                print if args.verbose else ec._silent) for mode in modes]
    print_report(reps)

    if args.save or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        draw(reps, "%d qubits, %d interactions of %s: spectral dimension of the information "
                   "network (%s walk)" % (args.qubits, args.timesteps, reps[0]["interaction"],
                                          args.walk), args.save)
        if not args.no_show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
