# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Sweep qudit operator transfer over boundary geometry, framing and thickness.

Every case pairs two generic simplex-boundary geometries, reads their
eigenframes as the two charts, and relaxes one prism between them.  The chart is
the full width of the boundary, so ``d`` cells carry a qudit of dimension ``d``,
and every chart vector is an isolated-boundary eigenstate by construction.  See
:mod:`qudit_boundary_geometry` for the construction and for why the amplitude
fixture cannot reach ``d >= 3``.

The configuration space is the product of qudit dimension, prism thickness,
attachment permutation and geometry seed.  It is enumerated, shuffled, and then
truncated, in that order, so a truncated run is an unbiased sample of the whole
space rather than a prefix of it.  Two caps bound memory: ``--max-attachments``
samples the permutation group before the product is formed, which matters
because it grows factorially, and ``--max-configurations`` truncates the
shuffled product.

Records stream to JSONL as each case finishes.  The driver holds counters and
never accumulates records, so its memory does not grow with the sweep.  Output
does not default under ``/tmp``: this host empties it at every boot.

Cases are not budgeted.  Restarts and iterations are set high enough not to
bind, and each case ends when it converges or when interior growth is
exhausted.
``--max-growth`` remains finite because it bounds memory rather than time, and
the validated configurations need no growth at all.

Run:

    python examples/cobordism/qudit_operator_sweep.py --dimensions 3 4 \\
        --jobs 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import concurrent.futures  # noqa: E402
import numpy as np  # noqa: E402

try:
    from examples.cobordism.qudit_boundary_geometry import (
        BoundaryGeometry,
        QuditPairCobordism,
    )
except ModuleNotFoundError:
    from qudit_boundary_geometry import (
        BoundaryGeometry,
        QuditPairCobordism,
    )


_DEFAULT_OUTPUT = Path.home() / "cobordism-runs" / "qudit_operator_sweep.jsonl"
_UNBUDGETED_RESTARTS = 64
_UNBUDGETED_ITERATIONS = 100000


def permutation_class(permutation):
    """The cycle type of a permutation, which labels its conjugacy class."""
    seen = set()
    cycles = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        length = 0
        current = start
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        cycles.append(length)
    return tuple(sorted(cycles, reverse=True))


def sample_attachments(dimension, limit, rng):
    """Attachment permutations for one dimension, capped before the product.

    The permutation group grows factorially, so it is sampled here rather than
    after the product is formed.  The identity is always retained, and one
    representative of every cycle type is retained before the remainder is
    sampled, so the framing structure stays legible under a tight cap.
    """
    everything = [tuple(p) for p in itertools.permutations(range(dimension))]
    if limit is None or limit >= len(everything):
        return everything
    identity = tuple(range(dimension))
    chosen = [identity]
    seen_classes = {permutation_class(identity)}
    for permutation in everything:
        if len(chosen) >= limit:
            break
        cycle_type = permutation_class(permutation)
        if cycle_type in seen_classes:
            continue
        seen_classes.add(cycle_type)
        chosen.append(permutation)
    remainder = [p for p in everything if p not in set(chosen)]
    rng.shuffle(remainder)
    chosen.extend(remainder[:max(0, limit - len(chosen))])
    return chosen


def build_configurations(dimensions, layer_counts, geometry_seeds,
                         max_attachments, shuffle_seed,
                         max_configurations=None):
    """Enumerate the configuration product, shuffle it, then truncate."""
    rng = np.random.default_rng(shuffle_seed)
    configurations = []
    for dimension in dimensions:
        attachments = sample_attachments(dimension, max_attachments, rng)
        for layers, attachment, seed in itertools.product(
                layer_counts, attachments, range(geometry_seeds)):
            configurations.append({
                "name": (f"d{dimension}_l{layers}_"
                         f"a{''.join(str(v) for v in attachment)}_g{seed}"),
                "dimension": int(dimension),
                "layers": int(layers),
                "attachment": list(attachment),
                "attachment_class": list(permutation_class(attachment)),
                "geometry_seed": int(seed),
            })
    order = rng.permutation(len(configurations))
    configurations = [configurations[index] for index in order]
    for position, configuration in enumerate(configurations):
        configuration["schedule_index"] = position
    if max_configurations is not None:
        configurations = configurations[:max_configurations]
    return configurations


def evaluate(configuration, options):
    """Run one configuration and return its record."""
    started = time.time()
    record = dict(configuration)
    record["schema_version"] = 1
    try:
        rng = np.random.default_rng(
            (configuration["geometry_seed"] + 1) * 100003
            + configuration["dimension"] * 101
            + configuration["layers"])
        dimension = configuration["dimension"]
        fixture = QuditPairCobordism(
            dimension,
            BoundaryGeometry.random(dimension, rng),
            BoundaryGeometry.random(dimension, rng),
            layers=configuration["layers"],
            attachment_permutation=configuration["attachment"],
        )
        measured = fixture.transfer(
            epsilon=options["epsilon"],
            restarts=options["restarts"],
            max_growth=options["max_growth"],
            max_iterations=options["max_iterations"],
            seed=configuration["geometry_seed"],
            held_out_count=options["held_out_count"],
        )
        record.update(measured)
        record["status"] = ("converged"
                            if measured["residual"] < options["epsilon"]
                            else "unconverged")
    except Exception as error:                      # noqa: BLE001
        record["status"] = "rejected" if isinstance(
            error, ValueError) else "error"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)[:400]
    record["duration_seconds"] = time.time() - started
    return record


def write_record(handle, record):
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def summarize(counters, totals):
    summary = {
        "status_counts": dict(counters["status"]),
        "status_by_dimension": {key: dict(value) for key, value
                                in counters["by_dimension"].items()},
        "status_by_layers": {key: dict(value) for key, value
                             in counters["by_layers"].items()},
        "status_by_attachment_class": {key: dict(value) for key, value
                                       in counters["by_class"].items()},
    }
    for name, values in totals.items():
        if values:
            summary[name] = {
                "count": len(values),
                "minimum": float(min(values)),
                "median": float(np.median(values)),
                "maximum": float(max(values)),
            }
    return summary


def run_sweep(configurations, options, output, jobs):
    counters = {
        "status": Counter(),
        "by_dimension": {},
        "by_layers": {},
        "by_class": {},
    }
    totals = {"converged_residual": [], "emergent_transfer_error": [],
              "growth_steps": [], "duration_seconds": []}
    started = time.time()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with output.open("w", encoding="utf-8") as handle:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(evaluate, configuration, options):
                       configuration for configuration in configurations}
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                completed += 1
                status = record["status"]
                counters["status"][status] += 1
                for key, bucket in (("dimension", "by_dimension"),
                                    ("layers", "by_layers")):
                    label = str(record[key])
                    counters[bucket].setdefault(label, Counter())[status] += 1
                label = "".join(str(v) for v in record["attachment_class"])
                counters["by_class"].setdefault(label, Counter())[status] += 1
                totals["duration_seconds"].append(record["duration_seconds"])
                if status == "converged":
                    totals["converged_residual"].append(record["residual"])
                if "emergent_transfer_error_max" in record:
                    totals["emergent_transfer_error"].append(
                        record["emergent_transfer_error_max"])
                if "growth_steps" in record:
                    totals["growth_steps"].append(record["growth_steps"])
                write_record(handle, record)
                print(f"[{completed}/{len(configurations)}] "
                      f"{record['name']}: {status}", flush=True)
    summary = summarize(counters, totals)
    summary["case_count"] = completed
    summary["wall_duration_seconds"] = time.time() - started
    summary["options"] = options
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[3, 4],
                        help="qudit dimensions; each boundary carries this "
                             "many cells")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 4],
                        help="prism thicknesses to sweep; one layer has no "
                             "interior and is not admissible")
    parser.add_argument("--geometry-seeds", type=int, default=4,
                        help="boundary geometry pairs per combination")
    parser.add_argument("--max-attachments", type=int, default=24,
                        help="attachment permutations kept per dimension, "
                             "sampled before the product is formed")
    parser.add_argument("--max-configurations", type=int, default=512,
                        help="cases kept after the product is shuffled")
    parser.add_argument("--shuffle-seed", type=int, default=20260829)
    parser.add_argument("--epsilon", type=float, default=1e-16)
    parser.add_argument("--max-growth", type=int, default=8,
                        help="interior growth ceiling; this bounds memory, "
                             "not time")
    parser.add_argument("--restarts", type=int, default=_UNBUDGETED_RESTARTS)
    parser.add_argument("--max-iterations", type=int,
                        default=_UNBUDGETED_ITERATIONS)
    parser.add_argument("--held-out-count", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if any(layers < 2 for layers in args.layers):
        raise SystemExit(
            "a prism of one layer has no interior; sweep layers >= 2")
    if any(dimension < 3 for dimension in args.dimensions):
        raise SystemExit("this sweep covers qutrits and wider")
    configurations = build_configurations(
        args.dimensions, args.layers, args.geometry_seeds,
        args.max_attachments, args.shuffle_seed, args.max_configurations)
    options = {
        "epsilon": float(args.epsilon),
        "max_growth": int(args.max_growth),
        "restarts": int(args.restarts),
        "max_iterations": int(args.max_iterations),
        "held_out_count": int(args.held_out_count),
    }
    print(f"configurations: {len(configurations)}  output: {args.output}",
          flush=True)
    summary = run_sweep(configurations, options, args.output, args.jobs)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                            encoding="utf-8")
    print(json.dumps(summary["status_counts"], sort_keys=True))
    print("records:", args.output)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
