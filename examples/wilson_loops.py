#!/usr/bin/env python3
"""Wilson loop visualization: compute and display loops in all three modes.

Builds a CDT spacetime, generates hinge/geodesic/dual-lattice loops,
evaluates each in COMBINATORIAL, DEFICIT_ANGLE, and CAUSAL modes, then
renders the loop path overlaid on a 3D force-directed layout of the
spatial slice.

Usage:
    python examples/wilson_loops.py
    python examples/wilson_loops.py --n-simplices 200 --save wilson.gif
"""
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.plot import force_layout_3d
from tessera.utils.progress import SingleTaskProgress


# =========================================================================
# Dual-graph layout: one node per top-simplex, edges between adjacent ones
# =========================================================================

def _skey(s):
    """Vertex-ID frozenset as a stable key for a simplex."""
    return frozenset(v.getId() for v in s.getVertices())


def _build_dual_complex(st):
    """Build the full dual complex: nodes = top-simplices, edges = shared facets.

    Topology comes straight from ``Spacetime.getDualAdjacency()`` (computed in
    C++ — far faster than walking simplices/facets/cofaces from Python, and it
    sidesteps the ``getCofaces`` detached-copy hazard).  Node ``i`` is the
    ``i``-th entry of ``getTopSimplices()``, matching the COO indexing.

    The per-edge timelike flag is the only thing not in the COO: two adjacent
    top simplices share exactly their common ``(d-1)``-facet, i.e. the
    intersection of their vertex-id sets, so we classify it as timelike when
    those shared vertices span more than one time slice.

    Returns (key_to_idx, top_simplices, dual_edges, edge_types).
    """
    top_simplices = st.getTopSimplices()
    key_to_idx = {}
    vid_sets = []
    vid_time = {}
    for i, s in enumerate(top_simplices):
        verts = s.getVertices()
        ids = frozenset(v.getId() for v in verts)
        key_to_idx[ids] = i
        vid_sets.append(ids)
        for v in verts:
            vid_time[v.getId()] = round(v.getTime())

    rows, cols, _n = st.getDualAdjacency()
    edge_list = []
    edge_types = []
    seen = set()
    for a, b in zip(rows, cols):
        if a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in seen:
            continue
        seen.add(edge)
        shared = vid_sets[edge[0]] & vid_sets[edge[1]]
        is_tl = len({vid_time[v] for v in shared}) > 1
        edge_list.append(edge)
        edge_types.append(is_tl)

    return key_to_idx, top_simplices, edge_list, edge_types


def _loop_indices(loop, key_to_idx):
    """Map loop simplices to indices in the dual-complex layout."""
    return [key_to_idx[_skey(s)] for s in loop.simplices
            if _skey(s) in key_to_idx]


def _spread_loop_nodes(pos, loop_indices, blend=0.5):
    """Nudge loop nodes toward a circular arrangement so the loop is visible.

    Blends between the full-complex position (blend=0) and a circle
    centered at the loop centroid (blend=1).  Returns a copy of pos.
    """
    pos = pos.copy()
    n = len(loop_indices)
    if n < 3:
        return pos

    loop_pos = pos[loop_indices]
    centroid = loop_pos.mean(axis=0)

    # Compute a circle in the plane of best fit (PCA)
    centered = loop_pos - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto first two principal components
    u = Vt[0]
    v = Vt[1]
    # Use a radius that ensures the loop is visually open
    spread = np.linalg.norm(centered, axis=1).mean()
    all_spread = np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean()
    radius = max(spread * 2.0, all_spread * 0.4)

    for i, li in enumerate(loop_indices):
        angle = 2 * math.pi * i / n
        circle_pt = centroid + radius * (math.cos(angle) * u +
                                          math.sin(angle) * v)
        pos[li] = (1 - blend) * pos[li] + blend * circle_pt

    return pos


def _render_frame(pos, dual_edges, edge_types, loop_indices, title,
                  show_dual_dotted=False,
                  fig_size=(7, 7), elev=25, azim=45):
    """Render the full dual complex with the loop path highlighted.

    Dual edges colored by type: blue = timelike, black = spacelike.
    If show_dual_dotted=True, dual edges are drawn as dotted lines.
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")

    loop_set = set(loop_indices)

    # Draw dual-complex edges, colored by type
    if dual_edges:
        segs = [[pos[a], pos[b]] for a, b in dual_edges]
        if show_dual_dotted:
            colors = [(0.15, 0.30, 0.80, 0.8) if tl
                      else (0.0, 0.0, 0.0, 0.7)
                      for tl in edge_types]
            lc = Line3DCollection(segs, linewidths=1.0, colors=colors,
                                  linestyles="dotted")
        else:
            colors = [(0.15, 0.30, 0.80, 0.75) if tl
                      else (0.0, 0.0, 0.0, 0.6)
                      for tl in edge_types]
            lc = Line3DCollection(segs, linewidths=0.7, colors=colors)
        ax.add_collection(lc)

    # Draw dual-complex nodes
    node_colors = []
    node_sizes = []
    for i in range(len(pos)):
        if i in loop_set:
            node_colors.append("red")
            node_sizes.append(70)
        else:
            node_colors.append("lightgray")
            node_sizes.append(12)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c=node_colors, s=node_sizes,
               edgecolors="gray", linewidths=0.2,
               depthshade=True)

    # Draw loop path (red, thick)
    if len(loop_indices) >= 2:
        loop_segs = []
        for i in range(len(loop_indices)):
            j = (i + 1) % len(loop_indices)
            loop_segs.append([pos[loop_indices[i]], pos[loop_indices[j]]])
        lc_loop = Line3DCollection(loop_segs, linewidths=3.0,
                                   colors="red", alpha=0.9)
        ax.add_collection(lc_loop)

        # Node labels on loop
        for idx, li in enumerate(loop_indices):
            ax.text(pos[li, 0], pos[li, 1], pos[li, 2] + 0.15,
                    str(idx), fontsize=8, ha="center", va="bottom",
                    fontweight="bold", color="darkred", zorder=11)

    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    fig.subplots_adjust(right=0.95, left=0.05, top=0.92, bottom=0.05)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


# =========================================================================
# Main
# =========================================================================

def main():
    monitor = MemoryMonitor()
    p = argparse.ArgumentParser(
        description="Wilson loop visualization across three evaluation modes")
    p.add_argument("--n-simplices", type=int, default=50)
    p.add_argument("--save", type=str, default="wilson_loops.gif")
    args = p.parse_args()

    prog = SingleTaskProgress(memory_monitor=monitor)

    # Build spacetime
    prog.phase("building", extra=f"{args.n_simplices} simplices")
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(args.n_simplices)
    target = st.getN41()
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / target, target)
    prog.phase("tuning", total=20)
    cdt.tune(progress=prog.on_tick)
    prog.phase("thermalizing", total=10)
    cdt.sweep(10, progress=prog.on_tick)
    print(f"  Vertices: {st.getVertexCount()}, "
          f"Top simplices: {st.getTopSimplexCount()}")

    # Create WilsonLoop
    wl = tessera.WilsonLoop(st)

    # Find the hinge with the largest fan (most top-simplices around it)
    hinge = None
    best_fan = 0
    for s in st.getSimplices():
        if len(s.getVertices()) != 3:
            continue
        loop_candidate = wl.hingeLoop(s)
        fan = len(loop_candidate)
        if fan > best_fan:
            best_fan = fan
            hinge = s

    # Find a top-simplex for geodesic/dual-lattice
    start_simplex = None
    for s in st.getSimplices():
        if len(s.getVertices()) == 5:
            start_simplex = s
            break

    # Generate loops
    loops = {}
    if hinge is not None:
        loop = wl.hingeLoop(hinge)
        if len(loop) >= 2:
            loops["Hinge"] = loop
    if start_simplex is not None:
        loop = wl.geodesicLoop(start_simplex)
        if len(loop) >= 2:
            loops["Geodesic"] = loop
        loop = wl.dualLatticeLoop(start_simplex, 8)
        if len(loop) >= 2:
            loops["Dual-lattice"] = loop

    if not loops:
        print("No loops found. Try a larger spacetime.")
        return

    # Evaluate and print results
    modes = [
        ("COMBINATORIAL", tessera.WilsonMode.COMBINATORIAL),
        ("DEFICIT_ANGLE", tessera.WilsonMode.DEFICIT_ANGLE),
        ("CAUSAL",        tessera.WilsonMode.CAUSAL),
    ]

    print(f"\n{'Loop':<15} {'Mode':<16} {'Value':>8} {'Size':>5} "
          f"{'Hinges':>7} {'Winding':>8}")
    print("-" * 65)
    for loop_name, loop in loops.items():
        for mode_name, mode in modes:
            r = wl.evaluate(loop, mode)
            print(f"{loop_name:<15} {mode_name:<16} {r.value:>8.4f} "
                  f"{r.loopSize:>5} {r.enclosedHinges:>7} "
                  f"{r.causalWindingNumber:>8}")

    # Measure all hinge loops for area-law plot
    wl.measureAllHinges(tessera.WilsonMode.DEFICIT_ANGLE)
    measurements = wl.getMeasurements()
    avg = wl.getAverageBySize()
    if avg:
        print(f"\nHinge loop averages by size:")
        for size, val in sorted(avg.items()):
            print(f"  size={size}: W={val:.4f}")

    # Build full dual-complex layout (shared across all loop types)
    prog.phase("layouting", extra="dual complex")
    key_to_idx, top_simplices, dual_edges, edge_types = _build_dual_complex(st)
    dual_pos = force_layout_3d(
        len(top_simplices), dual_edges,
        spring_k=0.05, repulsion_k=0.3, repulsion_cap=300, iters=300)

    # Render GIF: for each loop type, show rotating views
    prog.phase("rendering", extra=args.save)
    frames = []
    azimuths = list(range(0, 360, 1))

    for loop_name, loop in loops.items():
        r = wl.evaluate(loop, tessera.WilsonMode.DEFICIT_ANGLE)
        title = (f"{loop_name} loop (size={r.loopSize}, "
                 f"W={r.value:.3f})")
        lidx = _loop_indices(loop, key_to_idx)
        dotted = (loop_name == "Dual-lattice")

        frame_pos = _spread_loop_nodes(dual_pos, lidx, blend=0.8)

        for az in azimuths:
            img = _render_frame(frame_pos, dual_edges, edge_types, lidx,
                                title, show_dual_dotted=dotted,
                                elev=25, azim=az)
            frames.append(img)

    # Assemble GIF
    pil_frames = [Image.fromarray(f[:, :, :3]) for f in frames]
    pil_frames[0].save(args.save, save_all=True,
                       append_images=pil_frames[1:],
                       duration=50, loop=0)
    prog.finish(f"saved {args.save}")


if __name__ == "__main__":
    main()
