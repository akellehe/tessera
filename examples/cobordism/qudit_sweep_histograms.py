# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Histograms of which qudit sweep configurations converge and which do not.

Reads the JSONL that :mod:`qudit_operator_sweep` streams and renders four
panels: the outcome split across qudit dimension, across prism thickness, and
across attachment cycle type, and the distribution of the emergent transfer
error.  The first three are the configuration axes the sweep varies; the fourth
is the one operator-level number that measures the fitted bulk rather than
restating pinned restrictions.

Outcomes are drawn in the reserved status palette rather than in series colors,
and every segment carries its count, so the split never rests on color alone.

Run:

    python examples/cobordism/qudit_sweep_histograms.py \\
        ~/cobordism-runs/qudit_operator_sweep.jsonl --output figures/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


_SURFACE = "#fcfcfb"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"

# The reserved status palette. Converged and unconverged are the two outcomes a
# healthy sweep produces; anything that raised is folded into one failed slot,
# which keeps the drawn set to three and clears the separation floors.
_STATUS_COLORS = {
    "converged": "#0ca30c",
    "unconverged": "#fab219",
    "failed": "#d03b3b",
}
_STATUS_ORDER = ("converged", "unconverged", "failed")


def load_records(paths):
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        raise ValueError("no records found")
    return records


def outcome(record):
    """Fold the raised statuses; they are not distinct outcomes here."""
    status = record.get("status")
    return status if status in ("converged", "unconverged") else "failed"


def group_counts(records, key):
    grouped = defaultdict(Counter)
    for record in records:
        grouped[key(record)][outcome(record)] += 1
    return grouped


def _sorted_labels(grouped):
    def sort_key(label):
        try:
            return (0, float(label), "")
        except ValueError:
            return (1, 0.0, label)
    return sorted(grouped, key=sort_key)


def draw_outcome_panel(axes, grouped, title, ylabel):
    """One horizontal stacked bar per configuration value, with counts."""
    labels = _sorted_labels(grouped)
    positions = np.arange(len(labels))
    left = np.zeros(len(labels))
    for status in _STATUS_ORDER:
        widths = np.array([grouped[label].get(status, 0) for label in labels],
                          dtype=float)
        if not widths.any():
            continue
        axes.barh(positions, widths, left=left, height=0.62,
                  color=_STATUS_COLORS[status], label=status,
                  edgecolor=_SURFACE, linewidth=2.0)
        for position, width, start in zip(positions, widths, left):
            if width <= 0:
                continue
            axes.text(start + width / 2.0, position, f"{int(width)}",
                      ha="center", va="center", fontsize=9,
                      color=_PRIMARY_INK)
        left = left + widths
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, color=_SECONDARY_INK)
    axes.set_ylabel(ylabel, color=_SECONDARY_INK)
    axes.set_xlabel("cases", color=_MUTED_INK)
    axes.set_title(title, color=_PRIMARY_INK, loc="left", fontsize=11)
    axes.invert_yaxis()
    _recede(axes)


def draw_transfer_panel(axes, records):
    """The emergent transfer error, which spans orders of magnitude."""
    values = [record["emergent_transfer_error_max"]
              for record in records
              if record.get("emergent_transfer_error_max", 0.0) > 0.0]
    if not values:
        axes.set_axis_off()
        axes.text(0.5, 0.5, "no emergent transfer measurements",
                  ha="center", va="center", color=_MUTED_INK)
        return
    exponents = np.log10(values)
    bins = np.arange(np.floor(exponents.min()), np.ceil(exponents.max()) + 1.0,
                     0.5)
    axes.hist(exponents, bins=bins, color=_STATUS_COLORS["converged"],
              edgecolor=_SURFACE, linewidth=2.0)
    axes.set_xlabel("log10 emergent transfer error", color=_MUTED_INK)
    axes.set_ylabel("cases", color=_SECONDARY_INK)
    axes.set_title(
        f"Emergent transfer error (n={len(values)}, median "
        f"{np.median(values):.1e})",
        color=_PRIMARY_INK, loc="left", fontsize=11)
    _recede(axes, integer_axis="y")


def _recede(axes, integer_axis="x"):
    axes.set_facecolor(_SURFACE)
    # Counts are integers; fractional ticks on a count axis are meaningless.
    getattr(axes, f"{integer_axis}axis").set_major_locator(
        MaxNLocator(integer=True))
    axes.grid(True, axis="x", color=_GRIDLINE, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_BASELINE)
    axes.tick_params(colors=_MUTED_INK, length=0)


def render(records, output):
    figure, panels = plt.subplots(2, 2, figsize=(13, 8))
    figure.patch.set_facecolor(_SURFACE)
    draw_outcome_panel(panels[0][0],
                       group_counts(records, lambda r: str(r["dimension"])),
                       "Outcome by qudit dimension", "dimension")
    draw_outcome_panel(panels[0][1],
                       group_counts(records, lambda r: str(r["layers"])),
                       "Outcome by prism thickness", "layers")
    draw_outcome_panel(
        panels[1][0],
        group_counts(records, lambda r: "".join(
            str(value) for value in r["attachment_class"])),
        "Outcome by attachment cycle type", "cycle type")
    draw_transfer_panel(panels[1][1], records)

    handles, labels = panels[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels),
                  frameon=False, labelcolor=_SECONDARY_INK,
                  bbox_to_anchor=(0.5, -0.01))
    figure.suptitle("Qudit operator transfer: configuration outcomes",
                    color=_PRIMARY_INK, x=0.02, ha="left", fontsize=13)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=_SURFACE)
    plt.close(figure)
    return output


def table_view(records):
    """The same counts as text, which the contrast relief rule requires."""
    lines = []
    for title, key in (("dimension", lambda r: str(r["dimension"])),
                       ("layers", lambda r: str(r["layers"])),
                       ("cycle type", lambda r: "".join(
                           str(v) for v in r["attachment_class"]))):
        grouped = group_counts(records, key)
        lines.append(f"\n{title:<12}" + "".join(
            f"{status:>14}" for status in _STATUS_ORDER))
        for label in _sorted_labels(grouped):
            lines.append(f"{label:<12}" + "".join(
                f"{grouped[label].get(status, 0):>14}"
                for status in _STATUS_ORDER))
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path.home() / "cobordism-runs"
                        / "qudit_sweep_histograms.png")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    records = load_records(args.records)
    print(f"records: {len(records)}")
    print(table_view(records))
    print("\nfigure:", render(records, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
