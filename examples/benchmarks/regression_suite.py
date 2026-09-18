#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Multi-task performance regression suite for tessera.

Runs a fixed catalogue of micro-benchmarks against the installed
``tessera`` extension, writes a structured JSON record into
``benchmarks/runs/<UTC-timestamp>-<git-sha>.json`` (or a directory you
pick), and — when given a ``--baseline`` JSON — diffs the current run
against it and exits non-zero if any task slowed by more than
``--threshold`` percent.

Designed to be fast enough to wire into a post-build hook: the
``--quick`` profile runs in under ~30 seconds on a workstation. The
default profile (heavier sizes, more repeats) targets ~60-90 seconds.

Task catalogue covers the hot paths exercised by the recent
consolidations:

  * ``build_cdt_*``           - Spacetime::build (CDT, multiple dims).
  * ``dual_adjacency_*``      - Spacetime::getDualAdjacency (dualNeighbors).
  * ``poset_from_spacetime``  - Poset::fromSpacetime (indexByKey helper).
  * ``causet_chain``          - quantum::Causet::chainFrom.
  * ``sparse_spectral_dim``   - SparseGraph::spectralDimension via the
                                shared SpectralGraph::diagonalHeatKernel
                                Krylov-Lanczos + Padé-13 backbone.
  * ``emergent_return_prob``  - EmergentGraph::returnProbability via
                                the same backbone (weighted Laplacian).
  * ``wilson_loop_hinges``    - WilsonLoop::measureAllHinges (the
                                consolidated bfsFindCycles path).
  * ``pachner_sweep``         - CDT sweep throughput (propose / apply /
                                rollback round-trip).

Quantum-only tasks (``emergent_return_prob``, ``causet_chain``) are
auto-skipped when the build was compiled without TESSERA_QUANTUM.

Usage examples:

    # Plain run — writes JSON, prints summary table.
    python examples/benchmarks/regression_suite.py

    # Fast smoke run.
    python examples/benchmarks/regression_suite.py --quick

    # Compare against a previous baseline; exit 1 on > 15% slowdown.
    python examples/benchmarks/regression_suite.py \\
        --baseline benchmarks/runs/2026-05-13T14-23-45Z-abc1234.json

    # Just list the catalogue without running anything.
    python examples/benchmarks/regression_suite.py --list

    # Subset selection — useful for iterative profiling.
    python examples/benchmarks/regression_suite.py \\
        --tasks sparse_spectral_dim,emergent_return_prob

    # Compare two existing runs without running new benchmarks.
    python examples/benchmarks/regression_suite.py \\
        --baseline benchmarks/runs/A.json \\
        --current  benchmarks/runs/B.json --no-run
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Cap thread pools to 10 — the tessera box is shared. Mirrors the
# ``feedback_cpu_cap`` policy. Respect the user if they already set
# the env var.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_var, "10")

import tessera  # noqa: E402  (after env-var setup)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "runs"
DEFAULT_THRESHOLD_PCT = 15.0
DEFAULT_REPEATS_QUICK = 2
DEFAULT_REPEATS_FULL = 4


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------


@dataclass
class BenchTask:
    """A single benchmark unit.

    ``setup`` runs once and returns whatever state ``run`` needs (often
    a pre-built Spacetime so that the timing isolates the operation
    under test rather than incidental build cost). ``run`` is the
    operation that gets timed, called ``repeats`` times.
    """

    name: str
    category: str
    params: dict[str, Any]
    run: Callable[[], dict[str, Any]]
    repeats: int = DEFAULT_REPEATS_FULL
    skip_reason: str = ""


@dataclass
class TaskResult:
    name: str
    category: str
    params: dict[str, Any]
    repeats: int
    time_mean_s: float
    time_std_s: float
    time_min_s: float
    time_max_s: float
    extras: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "task":         self.name,
            "category":     self.category,
            "params":       self.params,
            "repeats":      self.repeats,
            "time_mean_s":  self.time_mean_s,
            "time_std_s":   self.time_std_s,
            "time_min_s":   self.time_min_s,
            "time_max_s":   self.time_max_s,
            "extras":       self.extras,
            "skipped":      self.skipped,
            "skip_reason":  self.skip_reason,
        }


def _has_quantum() -> bool:
    return hasattr(tessera, "quantum")


def _has_emergent_graph() -> bool:
    try:
        from tessera.quantum.holography import EmergentGraph  # noqa
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Task constructors
#
# Each helper returns the ``run`` closure for a single timing point. The
# task generators below assemble them with parameter combinations.
# ---------------------------------------------------------------------------


def _make_build_cdt(dim: int, target: int):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)

    def run():
        st = tessera.Spacetime(
            metric, tessera.CDT, 1.0, 1.0,
            tessera.PREFERRED, tessera.Toroid())
        st.build(target)
        return {"simplices": st.getTopSimplexCount(),
                "vertices":  st.getVertexCount()}

    return run


def _make_dual_adjacency(dim: int, target: int):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    # Build once outside the timed path — we're measuring the
    # facet → coface walk, not the construction.
    st = tessera.Spacetime(
        metric, tessera.CDT, 1.0, 1.0,
        tessera.PREFERRED, tessera.Toroid())
    st.build(target)

    def run():
        rows, cols, n = st.getDualAdjacency()
        return {"n_edges_directed": len(rows), "n_nodes": int(n)}

    return run


def _make_poset_from_spacetime(dim: int, target: int):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(
        metric, tessera.CDT, 1.0, 1.0,
        tessera.PREFERRED, tessera.Toroid())
    st.build(target)

    # Poset is only exposed on the quantum submodule — the caller must
    # have already guarded with ``_has_quantum``.
    Poset = tessera.quantum.Poset

    def run():
        p = Poset.fromSpacetime(st)
        return {"n_nodes":  p.getNodeCount,
                "n_covers": p.getCoverCount()}

    return run


def _make_causet_chain(dim: int, target: int):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(
        metric, tessera.CDT, 1.0, 1.0,
        tessera.PREFERRED, tessera.Toroid())
    st.build(target)

    def run():
        chain = tessera.quantum.Causet.chainFrom(st)
        return {"n_sites": chain.nSites,
                "n_hopping_pairs": len(chain.hoppingPairs)}

    return run


def _make_sparse_spectral_dim(dim: int, target: int,
                                  n_walks: int = 8, n_times: int = 12,
                                  krylov: int = 12):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(
        metric, tessera.CDT, 1.0, 1.0,
        tessera.PREFERRED, tessera.Toroid())
    st.build(target)
    g = st.getDualGraph()

    import numpy as np

    def run():
        # Exercise the Krylov-Lanczos + Padé-13 path directly with a
        # deterministic start set so timings are stable.
        starts = list(range(min(n_walks, g.nNodes())))
        times = np.logspace(-1, 1, n_times).tolist()
        K = g.diagonalHeatKernel(starts, times, krylovDim=krylov)
        return {"n_nodes": g.nNodes(),
                "n_diag_entries": sum(len(row) for row in K)}

    return run


def _make_emergent_return_prob(n_nodes: int, n_sigmas: int,
                                  krylov: int = 12):
    """Synthetic toroidal 2D lattice → EmergentGraph; measure
    ``returnProbability`` (the SpectralGraph base path on weighted L)."""
    from tessera.quantum.holography import EmergentGraph
    import numpy as np

    # Small 2D torus: side ≈ sqrt(n_nodes).
    side = int(round(n_nodes ** 0.5))
    side = max(side, 2)
    n = side * side
    edges = []
    for i in range(side):
        for j in range(side):
            v = i * side + j
            w1 = i * side + ((j + 1) % side)
            w2 = ((i + 1) % side) * side + j
            edges.append((v, w1, 1.0))
            edges.append((v, w2, 1.0))
    g = EmergentGraph.fromWeightedEdges(n, edges)
    sigmas = np.logspace(-1, 1, n_sigmas).tolist()

    def run():
        P = g.returnProbability(sigmas, krylov)
        return {"n_nodes": g.nVertices,
                "n_edges": g.nEdges,
                "P_min": min(P),
                "P_max": max(P)}

    return run


def _make_wilson_loop_hinges(dim: int, target: int):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(
        metric, tessera.CDT, 1.0, 1.0,
        tessera.PREFERRED, tessera.Toroid())
    st.build(target)
    wl = tessera.WilsonLoop(st)

    def run():
        # Reset between runs so we measure measurement throughput,
        # not append-to-vector overhead.
        wl.reset()
        wl.measureAllHinges(tessera.WilsonMode.COMBINATORIAL)
        return {"n_measurements": len(wl.getMeasurements())}

    return run


def _make_pachner_sweep(dim: int, target: int, sweeps: int):
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(
        metric, tessera.CDT, 1.0, 1.0,
        tessera.PREFERRED, tessera.Toroid())
    st.build(target)
    sim = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)

    def run():
        sim.sweep(sweeps)
        return {"simplices": st.getTopSimplexCount(),
                "sweeps":    sweeps}

    return run


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def _build_tasks(quick: bool) -> list[BenchTask]:
    """Materialise the full task catalogue. ``quick`` halves sizes and
    repeats so the suite runs under ~30s on a workstation."""
    repeats = DEFAULT_REPEATS_QUICK if quick else DEFAULT_REPEATS_FULL
    has_q = _has_quantum()
    has_eg = has_q and _has_emergent_graph()

    build_targets_2d = [600] if quick else [600, 2400]
    build_targets_3d = [400] if quick else [400, 1600]
    build_targets_4d = [400] if quick else [400, 1200]
    dual_target = 800 if quick else 2400
    poset_target = 600 if quick else 2000
    chain_target = 400 if quick else 1200
    spectral_target = 600 if quick else 1600
    spectral_walks = 6 if quick else 12
    spectral_times = 10 if quick else 16
    em_nodes = 64 if quick else 256
    em_sigmas = 8 if quick else 14
    wilson_target = 200 if quick else 600
    pachner_target = 200 if quick else 400
    pachner_sweeps = 20 if quick else 60

    tasks: list[BenchTask] = []

    # ── Build path ──────────────────────────────────────────────────
    for t in build_targets_2d:
        tasks.append(BenchTask(
            name=f"build_cdt_2d_n{t}",
            category="build",
            params={"dim": 2, "target": t},
            run=_make_build_cdt(2, t),
            repeats=repeats))
    for t in build_targets_3d:
        tasks.append(BenchTask(
            name=f"build_cdt_3d_n{t}",
            category="build",
            params={"dim": 3, "target": t},
            run=_make_build_cdt(3, t),
            repeats=repeats))
    for t in build_targets_4d:
        tasks.append(BenchTask(
            name=f"build_cdt_4d_n{t}",
            category="build",
            params={"dim": 4, "target": t},
            run=_make_build_cdt(4, t),
            repeats=repeats))

    # ── Graph extraction & traversal ───────────────────────────────
    tasks.append(BenchTask(
        name=f"dual_adjacency_3d_n{dual_target}",
        category="graph",
        params={"dim": 3, "target": dual_target},
        run=_make_dual_adjacency(3, dual_target),
        repeats=repeats))

    poset_task = BenchTask(
        name=f"poset_from_spacetime_3d_n{poset_target}",
        category="graph",
        params={"dim": 3, "target": poset_target},
        run=(_make_poset_from_spacetime(3, poset_target)
                if has_q else (lambda: {})),
        repeats=repeats,
        skip_reason="" if has_q else "tessera.quantum not available")
    tasks.append(poset_task)

    chain_task = BenchTask(
        name=f"causet_chain_2d_n{chain_target}",
        category="graph",
        params={"dim": 2, "target": chain_target},
        run=(_make_causet_chain(2, chain_target)
                if has_q else (lambda: {})),
        repeats=repeats,
        skip_reason="" if has_q else "tessera.quantum not available")
    tasks.append(chain_task)

    # ── Spectral / heat-kernel ─────────────────────────────────────
    tasks.append(BenchTask(
        name=f"sparse_spectral_dim_3d_n{spectral_target}",
        category="spectral",
        params={"dim": 3, "target": spectral_target,
                "walks": spectral_walks, "times": spectral_times},
        run=_make_sparse_spectral_dim(3, spectral_target,
                                          n_walks=spectral_walks,
                                          n_times=spectral_times),
        repeats=repeats))

    em_task = BenchTask(
        name=f"emergent_return_prob_n{em_nodes}",
        category="spectral",
        params={"n_nodes": em_nodes, "n_sigmas": em_sigmas},
        run=(_make_emergent_return_prob(em_nodes, em_sigmas)
                if has_eg else (lambda: {})),
        repeats=repeats,
        skip_reason="" if has_eg else "EmergentGraph not available")
    tasks.append(em_task)

    # ── Wilson loops & Pachner sweep ───────────────────────────────
    tasks.append(BenchTask(
        name=f"wilson_loop_hinges_3d_n{wilson_target}",
        category="observable",
        params={"dim": 3, "target": wilson_target},
        run=_make_wilson_loop_hinges(3, wilson_target),
        repeats=repeats))

    tasks.append(BenchTask(
        name=f"pachner_sweep_2d_n{pachner_target}_s{pachner_sweeps}",
        category="pachner",
        params={"dim": 2, "target": pachner_target,
                "sweeps": pachner_sweeps},
        run=_make_pachner_sweep(2, pachner_target, pachner_sweeps),
        repeats=repeats))

    return tasks


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_task(task: BenchTask, warmup: bool) -> TaskResult:
    if task.skip_reason:
        return TaskResult(
            name=task.name, category=task.category, params=task.params,
            repeats=0, time_mean_s=0.0, time_std_s=0.0,
            time_min_s=0.0, time_max_s=0.0,
            skipped=True, skip_reason=task.skip_reason)

    if warmup:
        # One untimed run to warm caches / JIT compilation paths.
        task.run()

    times: list[float] = []
    extras: dict[str, Any] = {}
    for _ in range(task.repeats):
        t0 = time.perf_counter()
        extras = task.run() or {}
        times.append(time.perf_counter() - t0)

    return TaskResult(
        name=task.name, category=task.category, params=task.params,
        repeats=task.repeats,
        time_mean_s=statistics.fmean(times),
        time_std_s=(statistics.stdev(times) if len(times) > 1 else 0.0),
        time_min_s=min(times),
        time_max_s=max(times),
        extras=extras)


def _print_progress(idx: int, total: int, task: BenchTask) -> None:
    prefix = f"[{idx:>2}/{total}]"
    if task.skip_reason:
        print(f"{prefix} {task.name:<50} SKIP ({task.skip_reason})")
    else:
        print(f"{prefix} {task.name:<50} running...", flush=True)


def _print_result(task: BenchTask, result: TaskResult) -> None:
    if result.skipped:
        return
    print(f"        {result.time_mean_s*1000:>10.2f} ms  "
          f"(+/- {result.time_std_s*1000:>7.2f} ms, "
          f"min {result.time_min_s*1000:>7.2f} / "
          f"max {result.time_max_s*1000:>7.2f}, "
          f"n={result.repeats})")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _git_info() -> dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), *args],
                text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    sha = _git("rev-parse", "--short", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"sha": sha, "branch": branch, "dirty": dirty}


def _build_metadata(quick: bool) -> dict[str, Any]:
    git = _git_info()
    return {
        "timestamp":       datetime.datetime.now(
                              datetime.timezone.utc).isoformat(),
        "git":             git,
        "tessera_version": getattr(tessera, "__version__", "unknown"),
        "tessera_quantum": _has_quantum(),
        "python_version":  sys.version.split()[0],
        "platform":        platform.platform(),
        "machine":         platform.machine(),
        "hostname":        platform.node(),
        "cpu_count":       os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "profile":         "quick" if quick else "full",
    }


def _output_filename(metadata: dict[str, Any]) -> str:
    """Sortable filename combining UTC timestamp + short SHA. Format
    looks like ``2026-05-13T14-23-45Z-abc1234.json`` so an
    alphabetical sort matches chronological order."""
    iso = metadata["timestamp"]
    # Drop microseconds for a tidier filename.
    if "." in iso:
        iso = iso.split(".")[0] + iso[iso.index("+"):] if "+" in iso else iso.split(".")[0]
    ts = (iso.replace("+00:00", "Z")
                .replace(":", "-"))
    git = metadata["git"]
    sha = git.get("sha") or "nogit"
    if git.get("dirty"):
        sha += "-dirty"
    return f"{ts}-{sha}.json"


# ---------------------------------------------------------------------------
# Comparison / regression assertion
# ---------------------------------------------------------------------------


def _load_record(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    return data.get("metadata", {}), data.get("results", [])


def _compare(
    baseline_results: list[dict[str, Any]],
    current_results:  list[dict[str, Any]],
    threshold_pct: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(report_rows, regressions)`` where each report row is a
    dict with task / baseline / current / pct_change keys, and
    ``regressions`` is the subset where pct_change exceeds the
    threshold (slower)."""
    by_name = {r["task"]: r for r in baseline_results}
    rows: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []

    for cur in current_results:
        if cur.get("skipped"):
            continue
        base = by_name.get(cur["task"])
        if base is None or base.get("skipped"):
            rows.append({
                "task": cur["task"],
                "baseline_ms": None,
                "current_ms":  cur["time_mean_s"] * 1000,
                "pct_change":  None,
                "verdict":     "new",
            })
            continue
        b_ms = base["time_mean_s"] * 1000
        c_ms = cur["time_mean_s"] * 1000
        pct = (c_ms - b_ms) / b_ms * 100.0 if b_ms > 0 else 0.0
        verdict = (
            "REGRESSED" if pct > threshold_pct else
            "improved"  if pct < -threshold_pct else
            "ok")
        row = {
            "task":         cur["task"],
            "baseline_ms":  b_ms,
            "current_ms":   c_ms,
            "pct_change":   pct,
            "verdict":      verdict,
        }
        rows.append(row)
        if verdict == "REGRESSED":
            regressions.append(row)

    # Surface tasks that disappeared from the current run.
    cur_names = {r["task"] for r in current_results}
    for base in baseline_results:
        if base["task"] not in cur_names:
            rows.append({
                "task":         base["task"],
                "baseline_ms":  base["time_mean_s"] * 1000,
                "current_ms":   None,
                "pct_change":   None,
                "verdict":      "missing",
            })

    return rows, regressions


def _print_comparison(rows: list[dict[str, Any]], threshold_pct: float) -> None:
    print()
    print(f"Comparison (threshold {threshold_pct:.1f}%)")
    print(f"{'task':<48} {'baseline':>12} {'current':>12} "
          f"{'change':>10}  verdict")
    print("-" * 100)
    for r in rows:
        b = (f"{r['baseline_ms']:>10.2f} ms"
                if r["baseline_ms"] is not None else f"{'-':>13}")
        c = (f"{r['current_ms']:>10.2f} ms"
                if r["current_ms"] is not None else f"{'-':>13}")
        if r["pct_change"] is not None:
            sign = "+" if r["pct_change"] >= 0 else ""
            pc = f"{sign}{r['pct_change']:>7.1f}%"
        else:
            pc = f"{'-':>9}"
        print(f"{r['task']:<48} {b} {c} {pc}  {r['verdict']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="tessera multi-task performance regression suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"directory to write per-build JSON files "
                          f"(default: {DEFAULT_OUT_DIR})")
    p.add_argument("--quick", action="store_true",
                    help="smaller sizes and fewer repeats")
    p.add_argument("--tasks", type=str, default="",
                    help="comma-separated subset of task names "
                          "(default: all)")
    p.add_argument("--list", action="store_true",
                    help="list available tasks and exit")
    p.add_argument("--no-warmup", action="store_true",
                    help="skip the untimed warm-up run before each task")
    p.add_argument("--baseline", type=Path, default=None,
                    help="baseline JSON to compare against; if any task "
                          "slows by more than --threshold %% the suite "
                          "exits with code 1")
    p.add_argument("--current", type=Path, default=None,
                    help="for --no-run: an existing run JSON to compare "
                          "against the baseline instead of running new "
                          "benchmarks")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT,
                    help=f"regression threshold in %% slowdown "
                          f"(default: {DEFAULT_THRESHOLD_PCT})")
    p.add_argument("--no-run", action="store_true",
                    help="do not run benchmarks; just compare --current "
                          "to --baseline")
    p.add_argument("--no-write", action="store_true",
                    help="do not write the JSON record")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = _build_metadata(args.quick)

    # ── List mode ──────────────────────────────────────────────────
    if args.list:
        for t in _build_tasks(args.quick):
            tag = f"[{t.category}]"
            print(f"{tag:<12} {t.name:<48} repeats={t.repeats}"
                  + (f"  SKIP({t.skip_reason})" if t.skip_reason else ""))
        return 0

    # ── Compare-only mode ──────────────────────────────────────────
    if args.no_run:
        if not args.baseline or not args.current:
            print("--no-run requires both --baseline and --current",
                  file=sys.stderr)
            return 2
        _, baseline_results = _load_record(args.baseline)
        _, current_results  = _load_record(args.current)
        rows, regressions = _compare(
            baseline_results, current_results, args.threshold)
        _print_comparison(rows, args.threshold)
        return 1 if regressions else 0

    # ── Run mode ───────────────────────────────────────────────────
    tasks = _build_tasks(args.quick)
    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t.name in wanted]
        if not tasks:
            print(f"No tasks match --tasks={args.tasks}", file=sys.stderr)
            return 2

    print("=" * 72)
    print("  tessera regression suite")
    print(f"  profile:        {metadata['profile']}")
    print(f"  tessera:        v{metadata['tessera_version']}"
          f"  (quantum={metadata['tessera_quantum']})")
    print(f"  git:            {metadata['git'].get('sha','?')}"
          f" on {metadata['git'].get('branch','?')}"
          f"{'  (dirty)' if metadata['git'].get('dirty') else ''}")
    print(f"  threads (OMP):  {metadata['omp_num_threads'] or 'unset'}")
    print(f"  tasks:          {len(tasks)}")
    print("=" * 72)
    print()

    t_total_0 = time.perf_counter()
    results: list[TaskResult] = []
    for i, task in enumerate(tasks, 1):
        _print_progress(i, len(tasks), task)
        r = _run_task(task, warmup=not args.no_warmup)
        _print_result(task, r)
        results.append(r)

    print()
    print(f"All tasks done in {time.perf_counter() - t_total_0:.1f}s")

    # ── Persist ────────────────────────────────────────────────────
    out_path: Path | None = None
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / _output_filename(metadata)
        out_path.write_text(json.dumps({
            "metadata": metadata,
            "results":  [r.to_json() for r in results],
        }, indent=2))
        print(f"Wrote {out_path}")

    # ── Baseline comparison ────────────────────────────────────────
    if args.baseline is not None:
        _, baseline_results = _load_record(args.baseline)
        rows, regressions = _compare(
            baseline_results,
            [r.to_json() for r in results],
            args.threshold)
        _print_comparison(rows, args.threshold)
        if regressions:
            print()
            print(f"FAIL: {len(regressions)} task(s) slowed by "
                  f"more than {args.threshold:.1f}%")
            return 1
        print()
        print(f"PASS: no task slowed by more than {args.threshold:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
