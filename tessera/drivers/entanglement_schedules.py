# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers.entanglement_schedules -- how pairwise mutual information
relaxes when the interacting pair is drawn from all pairs (the complete
graph) or from the nearest neighbours of an open chain.

For each schedule the time-slice loop of
``tessera.drivers.entanglement_complex`` runs on the same seeds, and every
slice records

* the sum of all pairwise mutual informations I(X:Y), in nats;
* the total correlation C = sum_i S(i) - S_global, in nats;
* the mean chain separation |i - j| weighted by I(i:j), in sites.

The summed mutual information is fitted, after the first tenth of the run,
as floor + A exp(-t / t_relax) by least squares on log(sum - floor) with the
floor set to 0.95 of the smallest value; the separation is compared with the
sqrt(t) growth a diffusive spread would give.
"""

import itertools

import numpy as np


def run(n, timesteps, seed, pair_mode, alpha="1/2"):
    """One run: the summed pairwise I, C and the I-weighted separation per slice."""
    from tessera.drivers import entanglement_complex as ec
    res = ec.simulate(n, 0, alpha, seed, "global", timesteps, pair_mode)
    slices = res["slices"]
    sum_i = np.array([s["sumI"] for s in slices])
    total = np.array([s["C"] for s in slices])
    sep = []
    for s in slices:
        MI = s["MI"]
        num = sum(MI[i, j] * (j - i) for i, j in itertools.combinations(range(n), 2))
        den = sum(MI[i, j] for i, j in itertools.combinations(range(n), 2))
        sep.append(num / den if den > 0 else np.nan)
    return sum_i, total, np.array(sep)


def fit_relaxation(y, skip):
    """floor + A exp(-t / t_relax) fitted on log(y - floor), floor = 0.95 min(y);
    returns (floor, A, t_relax), with t_relax = inf when y does not decay."""
    t = np.arange(len(y))[skip:]
    floor = y.min() * 0.95
    z = np.log(np.clip(y[skip:] - floor, 1e-12, None))
    slope, intercept = np.polyfit(t, z, 1)
    return floor, np.exp(intercept), -1.0 / slope if slope < 0 else np.inf


def compare(n, timesteps, seeds, alpha="1/2", jobs=1):
    """Run both schedules on every seed; returns per-schedule means and runs."""
    tasks = [(n, timesteps, seed, mode, alpha) for mode in ("all", "chain") for seed in seeds]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(min(jobs, len(tasks))) as pool:
            results = pool.starmap(run, tasks)
    else:
        results = [run(*task) for task in tasks]
    data = {}
    for mode in ("all", "chain"):
        runs = [r for task, r in zip(tasks, results) if task[3] == mode]
        data[mode] = {"sumI": np.mean([r[0] for r in runs], axis=0),
                      "C": np.mean([r[1] for r in runs], axis=0),
                      "sep": np.nanmean([r[2] for r in runs], axis=0),
                      "runs": runs}
    return data


def draw(data, n, timesteps, seeds, alpha, skip, fits, save=None):
    import matplotlib.pyplot as plt
    from tessera.drivers import baryon_poles as bp
    from tessera.drivers.entanglement_complex import CATEGORICAL

    colour = {"all": CATEGORICAL[0], "chain": CATEGORICAL[1]}
    t = np.arange(timesteps + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=bp.SURFACE)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.14, wspace=0.25)
    for ax in axes:
        bp.style_axis(ax)
        ax.set_xlabel("interactions t", color=bp.INK_MUTED)
    panels = (("sumI", "sum of pairwise I(X:Y) [nats]", "pairwise mutual information drains"),
              ("C", "C = sum S_i - S_global [nats]", "total correlation saturates"),
              ("sep", "I-weighted mean separation |i - j| [sites]",
               "how far correlations have spread along the chain index"))
    for ax, (key, ylabel, title) in zip(axes, panels):
        index = ("sumI", "C", "sep").index(key)
        for mode in ("all", "chain"):
            for r in data[mode]["runs"]:
                ax.plot(t, r[index], color=colour[mode], linewidth=0.8, alpha=0.25)
            ax.plot(t, data[mode][key], color=colour[mode], linewidth=2,
                    label="%s pairs" % mode)
        ax.set_ylabel(ylabel, color=bp.INK_MUTED)
        ax.set_title(title, loc="left", fontsize=10.5, color=bp.INK)
    for mode in ("all", "chain"):
        floor, A, tr = fits[mode]
        if np.isfinite(tr):
            axes[0].plot(t[skip:], floor + A * np.exp(-t[skip:] / tr), color=colour[mode],
                         linewidth=1, linestyle=(0, (4, 3)),
                         label="%s: fit, relaxation time %.1f" % (mode, tr))
    axes[0].set_yscale("log")
    c = data["chain"]["sep"]
    if np.isfinite(c[skip]) and c[skip] > 0:
        ref = c[skip] * np.sqrt(np.maximum(t, 1) / max(skip, 1))
        axes[2].plot(t[skip:], ref[skip:], color=bp.BASELINE, linewidth=1,
                     linestyle=(0, (4, 3)), label="sqrt(t) reference (diffusive spread)")
    for ax in axes:
        ax.legend(fontsize=8, frameon=False, labelcolor=bp.INK_MUTED)
    fig.suptitle("%d qubits, SWAP^(%s) on random pairs: complete graph against nearest-"
                 "neighbour chain (mean of seeds %s; thin lines: single runs)"
                 % (n, alpha, list(seeds)), fontsize=11, color=bp.INK, x=0.05, ha="left")
    if save:
        fig.savefig(save, dpi=150, facecolor=bp.SURFACE)
    return fig


def main(argv=None):
    """Compare the two schedules: a text table on standard output, or with
    ``--json`` the per-run curves as JSON; the figure with ``--save``."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="tessera.drivers.entanglement_schedules",
        description="Compare how pairwise mutual information relaxes with random "
                    "interactions on all pairs and on nearest neighbours of a chain.")
    parser.add_argument("--qubits", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--swap-power", default="1/2")
    parser.add_argument("--jobs", type=int, default=1,
                        help="run this many (schedule, seed) simulations in parallel processes")
    parser.add_argument("--json", action="store_true",
                        help="write the per-run curves as JSON on standard output")
    parser.add_argument("--save", default=None, help="write the figure to this path")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args(argv)

    n, T = args.qubits, args.timesteps
    data = compare(n, T, args.seeds, args.swap_power, args.jobs)
    skip = max(2, T // 10)
    fits = {mode: fit_relaxation(data[mode]["sumI"], skip) for mode in ("all", "chain")}
    if args.json:
        payload = {"qubits": n, "timesteps": T, "seeds": args.seeds,
                   "swap_power": args.swap_power, "entropy_unit": "nats"}
        for mode in ("all", "chain"):
            payload[mode] = {key: [r[i].tolist() for r in data[mode]["runs"]]
                             for i, key in enumerate(("sumI", "C", "sep"))}
            payload[mode]["relaxation_time"] = fits[mode][2]
        json.dump(payload, sys.stdout, allow_nan=True)
        sys.stdout.write("\n")
        return 0

    print("%d qubits, %d interactions, seeds %s, SWAP^(%s); %d runs averaged per schedule"
          % (n, T, args.seeds, args.swap_power, len(data["all"]["runs"])))
    print("schedule   sumI(0)  sumI(T)   C(T)   relaxation time   half-life   sep(0)   sep(T)")
    for mode in ("all", "chain"):
        d, tr = data[mode], fits[mode][2]
        print("%-8s  %8.4f %8.4f %7.4f   %9.1f steps   %8.1f   %6.3f   %6.3f"
              % (mode, d["sumI"][0], d["sumI"][-1], d["C"][-1], tr, tr * np.log(2),
                 d["sep"][0], d["sep"][-1]))
    if args.save or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        draw(data, n, T, args.seeds, args.swap_power, skip, fits, args.save)
        if not args.no_show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
