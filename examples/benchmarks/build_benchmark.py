#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Build-time benchmarks for tessera simplicial complexes.

Measures wall-clock time to build CDT triangulations across dimensions
(1D through 4D) and a range of target sizes.  Results are written to a
structured JSON log and a set of publication-quality plots.

Usage:
    python examples/benchmarks/build_benchmark.py --save docs/source/assets/benchmarks/
    python examples/benchmarks/build_benchmark.py          # interactive plots

The JSON log is always written to <save_dir>/benchmark_results.json (or
./benchmark_results.json if --save is not given).
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

import tessera


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DIMENSIONS = [1, 2, 3, 4]
SIZES = [500, 2_000, 10_000, 50_000, 100_000]
REPEATS = 5

DIMENSION_COLORS = {1: "#636363", 2: "#2171b5", 3: "#238b45", 4: "#cb181d"}
DIMENSION_MARKERS = {1: "s", 2: "o", 3: "D", 4: "^"}

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def build_once(dim, target_n):
    """Build a single triangulation, return timing and stats."""
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                         tessera.PREFERRED, tessera.Toroid())

    t0 = time.perf_counter()
    st.build(target_n)
    elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": elapsed,
        "vertices": st.getVertexCount(),
        "edges": st.getEdgeList().size(),
        "simplices": st.getTopSimplexCount(),
    }


def run_benchmarks(dimensions, sizes, repeats):
    """Run the full benchmark matrix and return structured results."""
    total = len(dimensions) * len(sizes) * repeats
    results = []

    pbar = tqdm(total=total, desc="Benchmarking", unit="run")
    for dim in dimensions:
        for target_n in sizes:
            trial_results = []
            for rep in range(repeats):
                r = build_once(dim, target_n)
                r.update({"dimension": dim, "target_n": target_n,
                          "repeat": rep})
                trial_results.append(r)
                pbar.update(1)

            # Compute summary for this (dim, target_n) combination
            times = [t["elapsed_s"] for t in trial_results]
            simplices = trial_results[0]["simplices"]
            vertices = trial_results[0]["vertices"]
            edges = trial_results[0]["edges"]

            summary = {
                "dimension": dim,
                "target_n": target_n,
                "actual_simplices": simplices,
                "vertices": vertices,
                "edges": edges,
                "time_mean_s": np.mean(times),
                "time_std_s": np.std(times),
                "time_min_s": np.min(times),
                "time_max_s": np.max(times),
                "simplices_per_sec": simplices / np.mean(times)
                                     if np.mean(times) > 0 else 0,
                "trials": trial_results,
            }
            results.append(summary)

            tqdm.write(
                f"  {dim}D  target={target_n:>7,}  "
                f"actual={simplices:>7,}  "
                f"verts={vertices:>7,}  edges={edges:>8,}  "
                f"time={np.mean(times):.4f}s "
                f"(+/- {np.std(times):.4f}s)")

    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_build_time(results, ax):
    """Log-log plot of build time vs. target simplex count."""
    for dim in DIMENSIONS:
        subset = [r for r in results if r["dimension"] == dim]
        if not subset:
            continue
        x = [r["target_n"] for r in subset]
        y = [r["time_mean_s"] for r in subset]
        yerr = [r["time_std_s"] for r in subset]
        ax.errorbar(x, y, yerr=yerr, fmt=f"{DIMENSION_MARKERS[dim]}-",
                    color=DIMENSION_COLORS[dim], capsize=3,
                    linewidth=2, markersize=7, label=f"{dim}D")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target simplices", fontsize=12)
    ax.set_ylabel("Build time (seconds)", fontsize=12)
    ax.set_title("Build Time vs. Complex Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, title="Dimension", title_fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))


def plot_throughput(results, ax):
    """Bar chart of build throughput (simplices/sec) by dimension."""
    for dim in DIMENSIONS:
        subset = [r for r in results if r["dimension"] == dim]
        if not subset:
            continue
        x = [r["target_n"] for r in subset]
        y = [r["simplices_per_sec"] for r in subset]
        offsets = np.linspace(-0.15, 0.15, len(DIMENSIONS))
        dim_idx = DIMENSIONS.index(dim)
        x_pos = np.arange(len(x)) + offsets[dim_idx]
        bars = ax.bar(x_pos, y, width=0.08, color=DIMENSION_COLORS[dim],
                      label=f"{dim}D", edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([f"{s:,}" for s in SIZES], fontsize=9)
    ax.set_xlabel("Target simplices", fontsize=12)
    ax.set_ylabel("Simplices / second", fontsize=12)
    ax.set_title("Build Throughput", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, title="Dimension", title_fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))


def plot_scaling_ratios(results, ax):
    """Vertices and edges per simplex across dimensions."""
    bar_width = 0.35
    x_pos = np.arange(len(DIMENSIONS))

    # Use the largest-size run for each dimension for stable ratios
    verts_per = []
    edges_per = []
    for dim in DIMENSIONS:
        subset = [r for r in results if r["dimension"] == dim]
        if not subset:
            verts_per.append(0)
            edges_per.append(0)
            continue
        r = subset[-1]  # largest size
        s = max(r["actual_simplices"], 1)
        verts_per.append(r["vertices"] / s)
        edges_per.append(r["edges"] / s)

    ax.bar(x_pos - bar_width / 2, verts_per, bar_width,
           color=[DIMENSION_COLORS[d] for d in DIMENSIONS],
           edgecolor="white", linewidth=0.5, label="Vertices / simplex",
           alpha=0.85)
    ax.bar(x_pos + bar_width / 2, edges_per, bar_width,
           color=[DIMENSION_COLORS[d] for d in DIMENSIONS],
           edgecolor="white", linewidth=0.5, label="Edges / simplex",
           alpha=0.5, hatch="//")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{d}D" for d in DIMENSIONS], fontsize=12)
    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Ratio", fontsize=12)
    ax.set_title("Complex Density by Dimension",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)


def plot_actual_vs_target(results, ax):
    """Actual simplices built vs. target requested."""
    for dim in DIMENSIONS:
        subset = [r for r in results if r["dimension"] == dim]
        if not subset:
            continue
        x = [r["target_n"] for r in subset]
        y = [r["actual_simplices"] for r in subset]
        ax.plot(x, y, f"{DIMENSION_MARKERS[dim]}-",
                color=DIMENSION_COLORS[dim],
                linewidth=2, markersize=7, label=f"{dim}D")

    lims = [min(SIZES) * 0.5, max(SIZES) * 2]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target simplices", fontsize=12)
    ax.set_ylabel("Actual simplices built", fontsize=12)
    ax.set_title("Actual vs. Requested Size",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, title="Dimension", title_fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    fmt = ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def create_all_plots(results, save_dir=None):
    """Generate the full benchmark figure set."""
    # --- Main 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ver = getattr(tessera, "__version__", "?")
    fig.suptitle(f"tessera v{ver} Build Benchmarks", fontsize=16,
                 fontweight="bold", y=0.98)

    plot_build_time(results, axes[0, 0])
    plot_throughput(results, axes[0, 1])
    plot_actual_vs_target(results, axes[1, 0])
    plot_scaling_ratios(results, axes[1, 1])

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        path = os.path.join(save_dir, "build_benchmarks.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")

    # --- Hero chart: build time only (for docs intro) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_build_time(results, ax2)
    fig2.tight_layout()

    if save_dir:
        path2 = os.path.join(save_dir, "build_time.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"Saved {path2}")

    # --- Throughput standalone ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_throughput(results, ax3)
    fig3.tight_layout()

    if save_dir:
        path3 = os.path.join(save_dir, "build_throughput.png")
        fig3.savefig(path3, dpi=150, bbox_inches="tight")
        print(f"Saved {path3}")

    if not save_dir:
        plt.show()


# ---------------------------------------------------------------------------
# JSON log
# ---------------------------------------------------------------------------


def build_log(results):
    """Build the structured JSON log."""
    # Strip raw trial data for the summary (keep it in 'detailed')
    summary = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "trials"}
        summary.append(entry)

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "tessera_version": getattr(tessera, "__version__", "unknown"),
            "repeats_per_point": len(results[0]["trials"]) if results else 0,
        },
        "results": summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build-time benchmarks for tessera simplicial complexes")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save plots and JSON log")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help=f"Repetitions per (dim, size) pair "
                             f"(default: {REPEATS})")
    args = parser.parse_args()

    repeats = args.repeats

    print("=" * 64)
    print("  tessera Build Benchmarks")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Sizes:      {[f'{s:,}' for s in SIZES]}")
    print(f"  Repeats:    {repeats}")
    print(f"  Platform:   {platform.platform()}")
    print(f"  Python:     {sys.version.split()[0]}")
    print("=" * 64)
    print()

    t0 = time.time()
    results = run_benchmarks(DIMENSIONS, SIZES, repeats)
    elapsed = time.time() - t0

    print(f"\nAll benchmarks complete in {elapsed:.1f}s")

    # --- Summary table ---
    print(f"\n{'Dim':>4} {'Target':>10} {'Actual':>10} "
          f"{'Verts':>10} {'Edges':>10} "
          f"{'Time (s)':>12} {'Simpl/sec':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['dimension']:>3}D {r['target_n']:>10,} "
              f"{r['actual_simplices']:>10,} "
              f"{r['vertices']:>10,} {r['edges']:>10,} "
              f"{r['time_mean_s']:>10.4f}  "
              f"{r['simplices_per_sec']:>12,.0f}")

    # --- Write JSON ---
    log = build_log(results)
    json_dir = args.save or "."
    json_path = os.path.join(json_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nJSON log: {json_path}")

    # --- Plots ---
    create_all_plots(results, save_dir=args.save)


if __name__ == "__main__":
    main()
