#!/usr/bin/env python3
"""Solve the discrete Einstein equations for a point mass on a CDT complex and
render the result as a GIF, one frame per spatial time slice.

What is being solved
--------------------
A Regge geometry stores its shape in the squared lengths ``l^2`` of the edges of
a triangulation; there are no coordinates. In Lorentzian signature ``l^2`` is
real and signed: positive for a spacelike edge, negative for a timelike one. The
action of that geometry is

    S = S_Regge + S_matter
    S_Regge  = sum over hinges h of |*h| * eps_h
    S_matter = -M * sum over timelike worldline edges e of sqrt(-l^2_e)

A *hinge* is a triangle, the codimension-two face about which the geometry can
carry curvature. ``|*h|`` is the hinge's circumcentric dual content and ``eps_h``
its deficit angle -- how far the dihedral angles meeting at that hinge fall short
of filling the space around it. ``M`` is the mass and the *worldline* is the chain
of timelike edges the mass travels along, so ``S_matter`` is minus the mass times
its proper time.

The discrete Einstein equations are the statement that ``S`` is *stationary*, not
that it is small:

    dS / d(l^2_e) = 0   for every edge e

``S`` is unbounded below and must never be descended directly. The quantity this
script extremizes is therefore the squared norm of the action gradient,

    F = sum over edges e of |dS / d(l^2_e)|^2

which is non-negative and vanishes exactly when the Regge equations hold. ``F``
is a nonlinear least-squares objective in the edge variables and is minimized as
such, with the exact analytic gradient and Hessian of ``S`` supplying the
residual and its Jacobian; no finite differences are taken.

``ReggeSolver`` evaluates the action and its derivatives and does not relax the
geometry itself. Its ``actionGradientExact`` is the gradient of the
gravitational term ``S_Regge`` alone -- it does not see the matter configuration
-- so this script adds the matter term ``dS_matter/d(l^2_e) = M / (2 sqrt(-l^2_e))``
on the worldline's timelike edges. Without it the mass would not source any
curvature and ``--mass`` would have no effect on the geometry.

The analytic assembly also does not reach every edge. An edge it never reaches
is reported with a gradient of exactly zero and an empty Hessian column, whether
or not the action responds to that edge. Those components are recomputed here as
a central difference of the action so the residual counts them, and those edges
are held fixed during the relaxation because there is no descent direction along
them. The run reports how much of the final residual they carry.

Convergence criterion
---------------------
The relaxation stops when the trust-region solver's step, objective change, or
projected gradient falls below its tolerances, or when the evaluation budget
``--max-iters`` is spent. It is reported as *converged* only when the final
residual satisfies ``F <= --tol``, which is the statement that the Regge
equations are solved to that tolerance. Every free edge is bounded so that its
``l^2`` keeps its sign: a spacelike edge stays spacelike and a timelike edge
stays timelike. Changing an edge's causal character is a discrete change to the
causal structure, not a relaxation of the geometry.

What the frames show
--------------------
Each frame is one spatial time slice, drawn as a two-dimensional radial
embedding centered on the vertex the mass occupies in that slice. A vertex's
radius is its shortest-path distance in the slice from that center, measured in
edges, and its angle comes from a force-directed layout, so the mass sits at the
visual center of every frame. Color is the vertex's area-weighted deficit angle,
averaged over the spatial triangles that contain it.

A spacelike triangle in a four-dimensional Lorentzian complex has a timelike
normal plane, so the angles meeting at it include boosts as well as rotations
and its deficit angle is complex. The real part is the rotation, the angle
defect, and is the curvature in the ordinary sense; that is what the heat map
shows. The imaginary part is the boost, or rapidity, content, and the run
reports the largest one it encountered so the size of what the map leaves out is
stated rather than assumed.

The weighting differs between the two quantities on purpose. The action the
relaxation extremizes weights each hinge by its circumcentric dual content
``|*h|``, which is what the exact analytic gradient differentiates. The heat map
weights each hinge by its own triangle area ``A_h``, which reads as a local
curvature density at the vertex.

Usage (standalone):
    python examples/curvature_slice_gif.py --n-simplices 600 --mass 1.0 \
        --seed 20260909 --save point_mass.gif

Usage (as a library):
    from curvature_slice_gif import render_curvature_gif
    render_curvature_gif(st, solver, worldline, "curvature.gif")
"""
import argparse
import cmath
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.collections import LineCollection
from PIL import Image
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, vstack as sparse_vstack

import tessera
from tessera.utils.memory_monitor import MemoryMonitor
from tessera.utils.plot import (build_spacetime, time_slices, spatial_subgraph,
                              bfs_distances, save_gif, radial_layout_2d)
from tessera.utils.progress import SingleTaskProgress


# =========================================================================
# Per-vertex curvature
# =========================================================================

def _vertex_curvatures(verts, solver, t):
    """The area-weighted angle defect at each vertex of the time slice *t*.

    A vertex's value is the mean of ``A_h * eps_h`` -- the triangle's own area
    times its deficit angle -- over the spatial triangles containing it, where
    "spatial" means all three of the triangle's vertices sit in slice *t*.

    That product is complex. A spacelike triangle in a four-dimensional
    Lorentzian complex has a timelike normal plane, so the angles meeting at it
    include boosts as well as rotations: the real part of the deficit is the
    rotation, or angle defect, and is the curvature in the ordinary sense, while
    the imaginary part is the boost, or rapidity, content. The heat map shows the
    real part, and the largest imaginary part encountered is returned alongside
    it so the size of the boost content the map does not show is reported rather
    than left unstated.

    Returns ``(curvatures, max_boost_part)``.
    """
    curvatures = np.zeros(len(verts))
    max_boost = 0.0
    for idx, v in enumerate(verts):
        total = 0.0 + 0.0j
        count = 0
        for s in v.getSimplices():
            sv = s.getVertices()
            if len(sv) != 3:
                continue
            if not all(round(hv.getTime()) == t for hv in sv):
                continue
            eps = complex(solver.deficitAngle(s))
            area = complex(tessera.ReggeSolver.hingeArea(s))
            total += eps * area
            count += 1
        if count > 0:
            mean = total / count
            curvatures[idx] = mean.real
            max_boost = max(max_boost, abs(mean.imag))
    return curvatures, max_boost


# =========================================================================
# 2D radial layout: center at origin, radius = BFS distance
# =========================================================================

def _radial_layout_2d(verts, edges, center_vid, bfs_dist, *,
                      iters=200, prev_angles=None, seed=42):
    """Place vertices in 2D: radius = BFS distance, angles from force layout.

    The radius-constrained angular solve runs in C++
    (``tessera.ForceLayout.layout2D`` via :func:`radial_layout_2d`); this
    wrapper keeps the per-frame angular-continuity bookkeeping in Python:
    seeding initial angles from the previous frame and recording the solved
    angles for the next one.

    Returns (pos_2d, vid_to_idx, edge_idx, angles_by_dist) where
    angles_by_dist maps BFS distance to list of (angle, vid) for seeding
    the next frame.
    """
    rng = np.random.default_rng(seed)
    n = len(verts)
    vid_to_idx = {v.getId(): i for i, v in enumerate(verts)}
    center_idx = vid_to_idx[center_vid]

    # Group vertices by BFS distance
    max_d = max(bfs_dist.values()) if bfs_dist else 1

    # Initialize: place on circles by BFS distance with angular spread.
    # Radius is pinned to the BFS distance; the center stays at the origin.
    init_pos = np.zeros((n, 2))
    target_radii = np.zeros(n)
    for v in verts:
        idx = vid_to_idx[v.getId()]
        d = bfs_dist.get(v.getId(), max_d + 1)
        if v.getId() == center_vid:
            continue
        r = float(d)
        target_radii[idx] = r
        # Try to use previous frame's angular structure for stability
        if prev_angles and d in prev_angles and prev_angles[d]:
            # Pick the angle closest to a uniform distribution
            used = len([1 for vi in verts[:idx]
                       if bfs_dist.get(vi.getId(), -1) == d])
            n_at_d = sum(1 for vi in verts
                        if bfs_dist.get(vi.getId(), -1) == d)
            base_angle = prev_angles[d][0] if prev_angles[d] else 0
            angle = base_angle + 2 * math.pi * used / max(n_at_d, 1)
        else:
            angle = rng.uniform(0, 2 * math.pi)
        init_pos[idx] = [r * math.cos(angle), r * math.sin(angle)]

    # Build edge index
    edge_idx = []
    for e in edges:
        si = vid_to_idx.get(e.getSource().getId())
        ti = vid_to_idx.get(e.getTarget().getId())
        if si is not None and ti is not None:
            edge_idx.append((si, ti))

    # Radius-constrained, tangential-only angular solve (C++).
    pos = radial_layout_2d(n, edge_idx, target_radii, center_idx=center_idx,
                           init_pos=init_pos, iters=iters, seed=seed)

    # Record angles for next frame
    angles_by_dist = {}
    for v in verts:
        idx = vid_to_idx[v.getId()]
        d = bfs_dist.get(v.getId(), max_d + 1)
        if v.getId() == center_vid:
            continue
        angle = math.atan2(pos[idx, 1], pos[idx, 0])
        angles_by_dist.setdefault(d, []).append(angle)
    for d in angles_by_dist:
        angles_by_dist[d].sort()

    return pos, vid_to_idx, edge_idx, angles_by_dist


# =========================================================================
# Render a 2D frame
# =========================================================================

def _render_2d_frame(pos, edge_idx, curvatures, center_idx, t,
                     vmin, vmax, fig_size, cmap_name, axis_limit):
    """Render a 2D radial frame with curvature heatmap."""
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect("equal")

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Draw edges
    if edge_idx:
        segs = [[pos[s], pos[t]] for s, t in edge_idx]
        lc = LineCollection(segs, linewidths=0.5, colors=(0.7, 0.7, 0.7, 0.5))
        ax.add_collection(lc)

    # Draw vertices colored by curvature
    sizes = np.full(len(pos), 30.0)
    sizes[center_idx] = 120.0
    ax.scatter(pos[:, 0], pos[:, 1],
               c=curvatures, cmap=cmap, norm=norm,
               s=sizes, edgecolors="k", linewidths=0.3, zorder=5)

    # Center vertex star
    ax.scatter([pos[center_idx, 0]], [pos[center_idx, 1]],
               c="black", s=150, marker="*", zorder=10)

    # Fixed axis limits centered on origin
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_axis_off()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Area-weighted angle defect  Re(A_h eps_h)",
                   fontsize=9)

    ax.set_title(f"t = {t}", fontsize=12, pad=10)

    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


# =========================================================================
# Stationary-action relaxation
# =========================================================================

def _edge_key(edge):
    """An edge's identity as an ordered pair of vertex ids."""
    a, b = edge.getSource().getId(), edge.getTarget().getId()
    return (min(a, b), max(a, b))


def _worldline_edges(st, worldline, edge_index):
    """The positions of the worldline's timelike edges in the edge list.

    Walks consecutive worldline vertices and keeps the edge joining them when
    one exists and is timelike, which is the same traversal and the same causal
    test ``ReggeSolver.matterAction`` applies. A worldline vertex pair with no
    joining edge carries no proper time and so is skipped.
    """
    ids = [v.getId() for v in worldline]
    edges = st.getEdgeList().toVector()
    positions = []
    for a, b in zip(ids, ids[1:]):
        position = edge_index.get((min(a, b), max(a, b)))
        if position is not None and edges[position].isTimelike():
            positions.append(position)
    return positions


def _matter_derivatives(squared_lengths, worldline_positions, mass):
    """The matter action's first and second derivatives in the edge variables.

    ``S_matter = -M * sum_e sqrt(-l^2_e)`` over the worldline's timelike edges,
    so on each of those edges

        dS_matter / d(l^2_e)    = M / (2 * (-l^2_e)^(1/2))
        d2S_matter / d(l^2_e)^2 = M / (4 * (-l^2_e)^(3/2))

    and both vanish on every other edge. The second derivative is diagonal
    because an edge's proper-time contribution depends on its own length alone.

    Returns ``(gradient, hessian_diagonal)`` as real arrays over all edges.
    """
    gradient = np.zeros(len(squared_lengths))
    hessian_diagonal = np.zeros(len(squared_lengths))
    for position in worldline_positions:
        proper_time_squared = -squared_lengths[position]
        gradient[position] = mass / (2.0 * proper_time_squared ** 0.5)
        hessian_diagonal[position] = mass / (4.0 * proper_time_squared ** 1.5)
    return gradient, hessian_diagonal


def _require_finite(values, what):
    """Reject a non-finite derivative rather than carry it into the objective.

    ``Simplex.dualVolumeGradient`` divides by the principal square root of a
    facet's circumradius offset, which is exactly zero on a hinge whose
    circumradius coincides with one of its facets'. The action stays finite
    there -- it only multiplies by that root -- but its derivatives do not, and
    one such hinge poisons every edge derivative it contributes to. The
    objective is undefined in that case, so the relaxation stops and says so
    rather than reporting a residual of ``nan``.
    """
    finite = np.isfinite(values)
    if finite.all():
        return
    raise RuntimeError(
        f"{np.count_nonzero(~finite)} of {finite.size} {what} are not finite. "
        "A hinge whose circumradius coincides with a facet's makes the analytic "
        "dual-volume derivative diverge, leaving the stationarity residual "
        "undefined. Rebuild at a different --seed or --n-simplices.")


def _gravitational_hessian(solver):
    """``d2S_Regge/d(l^2_e)d(l^2_f)`` as a sparse matrix in edge-list order.

    Two edges couple only through a hinge they share, so the matrix is sparse
    and ``ReggeSolver.actionHessianExactSparse`` assembles it directly as
    coordinate lists rather than densifying it.
    """
    rows, cols, values, dim = solver.actionHessianExactSparse()
    values = np.asarray(values, dtype=complex)
    _require_finite(values, "action Hessian entries")
    return coo_matrix((values, (np.asarray(rows, dtype=int),
                                np.asarray(cols, dtype=int))),
                      shape=(dim, dim)).tocsc()


def covered_edges(solver):
    """The edges whose second derivative of the action the engine supplies.

    An edge with an identically empty Hessian column is one the analytic
    assembly never reaches: ``actionGradientExact`` reports its gradient as
    exactly zero and the Hessian offers no leverage on it, even where the
    action does respond to it. Those edges are held fixed during the relaxation
    -- there is no descent direction along them -- while their stationarity
    residual is still counted, so the reported residual stays honest.

    Returns a boolean mask over the edge list.
    """
    hessian = _gravitational_hessian(solver)
    column_peak = np.asarray(np.abs(hessian).max(axis=0).todense()).ravel()
    return column_peak > 0.0


def action_gradient(solver, edges, squared_lengths, worldline_positions, mass,
                    uncovered):
    """``dS/d(l^2_e)`` for every edge, gravity and matter together.

    The gravitational part is ``ReggeSolver.actionGradientExact``, the exact
    analytic gradient of the dual Regge action, which is matter-independent. On
    the edges named by *uncovered* that analytic gradient is reported as exactly
    zero whether or not the action responds to them, so those components are
    recomputed as a central difference of ``dualReggeAction`` in the edge's own
    ``l^2``. The step preserves the edge's sign, and therefore its causal
    character, on both legs. The matter term is added last.
    """
    gravity = np.asarray(solver.actionGradientExact(), dtype=complex)
    _require_finite(gravity, "gravitational action gradient entries")
    for position in np.nonzero(uncovered)[0]:
        edge = edges[position]
        squared = complex(squared_lengths[position])
        step = max(abs(squared) * 1e-5, 1e-9)
        edge.setLength(cmath.sqrt(squared + step))
        forward = complex(solver.dualReggeAction())
        edge.setLength(cmath.sqrt(squared - step))
        backward = complex(solver.dualReggeAction())
        edge.setLength(cmath.sqrt(squared))
        gravity[position] = (forward - backward) / (2.0 * step)
    _require_finite(gravity, "action gradient entries")
    matter, _ = _matter_derivatives(squared_lengths, worldline_positions, mass)
    return gravity + matter


def relax_to_stationary_action(st, solver, worldline, mass, *,
                               tol=1e-6, max_iters=100, band=4.0,
                               progress=None):
    """Move the edge lengths to a stationary point of the action.

    Minimizes ``F = sum_e |dS/d(l^2_e)|^2`` over the squared edge lengths. ``F``
    is a sum of squares, so it is minimized as a least-squares problem: the
    residual is the real and imaginary parts of the action gradient stacked into
    one real vector and its Jacobian is the action's analytic Hessian, with a
    trust-region reflective solve honouring the per-edge bounds.

    An edge is free when it has a definite causal character and the engine
    supplies its second derivative -- see :func:`covered_edges`. A free edge's
    bound keeps the sign of its ``l^2`` and lets its magnitude move by up to a
    factor of *band* either way, so a spacelike edge stays spacelike and a
    timelike edge stays timelike. Changing an edge's causal character is a
    discrete change to the causal structure, not a relaxation of the geometry.

    Returns ``(converged, F, held_F, n_evaluations)``, where *held_F* is the
    part of the final residual carried by the edges that were held fixed, an
    irreducible floor for this relaxation, and *converged* is ``F <= tol``: the
    Regge equations are satisfied to that residual.
    """
    edges = st.getEdgeList().toVector()
    edge_index = {_edge_key(e): i for i, e in enumerate(edges)}
    worldline_positions = _worldline_edges(st, worldline, edge_index)

    initial = np.array([(complex(e.getLength()) ** 2).real for e in edges])
    definite = np.array([e.isTimelike() or e.isSpacelike() for e in edges])
    covered = covered_edges(solver)
    free = np.nonzero(definite & covered)[0]
    uncovered = ~covered
    if free.size == 0:
        raise RuntimeError("no edge is both causally definite and covered by "
                           "the analytic second derivative")

    magnitudes = np.abs(initial[free])
    lower = np.where(initial[free] > 0.0, magnitudes / band, -band * magnitudes)
    upper = np.where(initial[free] > 0.0, band * magnitudes, -magnitudes / band)

    squared_lengths = initial.copy()
    evaluations = {"n": 0}

    def apply(x):
        """Write the free variables back onto the edges as complex lengths."""
        squared_lengths[free] = x
        for position in free:
            # l^2 is not stored: an edge carries its complex length and l^2 is
            # that length squared, so write the principal root. The root is real
            # for a positive l^2 and imaginary for a negative one, which is
            # exactly the spacelike/timelike distinction the bounds preserve.
            edges[position].setLength(
                cmath.sqrt(complex(squared_lengths[position])))

    def gradient_at(x):
        apply(x)
        return action_gradient(solver, edges, squared_lengths,
                               worldline_positions, mass, uncovered)

    def residual(x):
        evaluations["n"] += 1
        gradient = gradient_at(x)
        if progress is not None:
            progress.on_tick()
        return np.concatenate([gradient.real, gradient.imag])

    def jacobian(x):
        apply(x)
        hessian = _gravitational_hessian(solver)
        _, matter_diagonal = _matter_derivatives(
            squared_lengths, worldline_positions, mass)
        columns = hessian[:, free].tolil()
        for column, position in enumerate(free):
            if matter_diagonal[position] != 0.0:
                columns[position, column] += matter_diagonal[position]
        columns = columns.tocsr()
        return sparse_vstack([columns.real, columns.imag], format="csr")

    result = least_squares(
        residual, initial[free], jac=jacobian, bounds=(lower, upper),
        method="trf", tr_solver="lsmr", max_nfev=max_iters,
        xtol=1e-12, ftol=1e-12, gtol=1e-12)

    final = gradient_at(result.x)
    objective = float(np.vdot(final, final).real)
    held = final[uncovered]
    return (bool(objective <= tol), objective,
            float(np.vdot(held, held).real), evaluations["n"])


# =========================================================================
# Public entry point
# =========================================================================

def render_curvature_gif(st, solver, worldline, output_path="curvature.gif",
                         *, fig_size=(6, 6), cmap_name="RdYlBu_r",
                         layoutIters=200, frame_duration_ms=500,
                         **_kwargs):
    """Render a GIF with one frame per time slice showing the spatial
    subgraph as a 2D radial embedding with curvature heat map.

    The point mass (worldline vertex) is always at the visual center. A
    vertex's radius is its shortest-path distance in edges from that center and
    its color is the area-weighted angle defect of the spatial triangles
    containing it, as described in :func:`_vertex_curvatures`.
    """
    wl_by_time = {}
    for v in worldline:
        wl_by_time[round(v.getTime())] = v

    times = time_slices(st)

    # --- Pass 1: compute layouts and curvatures ---
    slice_data = []
    prev_angles = None
    max_boost_deficit = 0.0
    for t in times:
        if t not in wl_by_time:
            continue
        center = wl_by_time[t]
        verts, edges = spatial_subgraph(st, t)
        if len(verts) < 3:
            continue

        bfs_dist = bfs_distances(center, st)
        reachable_ids = set(bfs_dist.keys())
        verts = [v for v in verts if v.getId() in reachable_ids]
        edges = [e for e in edges
                 if e.getSource().getId() in reachable_ids
                 and e.getTarget().getId() in reachable_ids]
        if len(verts) < 3:
            continue

        pos, vid_to_idx, edge_idx, prev_angles = _radial_layout_2d(
            verts, edges, center.getId(), bfs_dist,
            iters=layoutIters, prev_angles=prev_angles)

        curvatures, max_boost = _vertex_curvatures(verts, solver, t)
        max_boost_deficit = max(max_boost_deficit, max_boost)
        center_idx = vid_to_idx[center.getId()]
        slice_data.append((t, pos, edge_idx, curvatures, center_idx))

    if not slice_data:
        print("  No renderable time slices found.")
        return output_path

    print(f"  largest boost part of an area-weighted deficit: "
          f"{max_boost_deficit:.4g}; the map shows the angle defect, the real "
          f"part")

    # Global curvature scale
    all_curv = np.concatenate([d[3] for d in slice_data])
    vmin, vmax = float(all_curv.min()), float(all_curv.max())
    if abs(vmax - vmin) < 1e-12:
        vmin -= 0.5
        vmax += 0.5

    # Global axis limit: max radius across all frames + padding
    max_radius = max(np.linalg.norm(d[1], axis=1).max() for d in slice_data)
    axis_limit = max_radius * 1.3

    # --- Pass 2: render frames ---
    frames = []
    for t, pos, edge_idx, curvatures, center_idx in slice_data:
        img = _render_2d_frame(pos, edge_idx, curvatures, center_idx, t,
                               vmin, vmax, fig_size, cmap_name, axis_limit)
        frames.append(img)

    save_gif(frames, output_path, duration_ms=frame_duration_ms)
    return output_path


# =========================================================================
# Standalone CLI
# =========================================================================

def main():
    monitor = MemoryMonitor()
    p = argparse.ArgumentParser(
        description="Solve the discrete Einstein equations for a point mass "
                    "and render each spatial slice as a curvature heat map")
    p.add_argument("--n-simplices", type=int, default=600,
                   help="four-simplices to build before thermalizing; the "
                        "builder caps the complex at 80 time slices")
    p.add_argument("--mass", type=float, default=1.0,
                   help="the point mass M in the proper-time action "
                        "S_matter = -M * sum sqrt(-l^2)")
    p.add_argument("--max-iters", type=int, default=100,
                   help="evaluation budget for the relaxation")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="residual F = ||dS/dl^2||^2 at or below which the "
                        "Regge equations count as solved")
    p.add_argument("--band", type=float, default=4.0,
                   help="the factor by which an edge's |l^2| may move; its "
                        "sign, and so its causal character, is always kept")
    p.add_argument("--seed", type=int, default=None,
                   help="seed both the build and the sweeps, making the "
                        "geometry and the figure reproducible")
    p.add_argument("--save", type=str, default="curvature_slices.gif")
    args = p.parse_args()

    prog = SingleTaskProgress(memory_monitor=monitor)
    prog.phase("building", extra=f"{args.n_simplices} simplices")

    st, _ = build_spacetime(args.n_simplices, seed=args.seed)

    verts = st.getVertexList().toVector()
    center = max(verts, key=lambda v: v.degree())

    matter = tessera.MatterConfiguration()
    worldline = tessera.MatterConfiguration.buildWorldline(center, st)
    matter.setWorldlineMass(center, args.mass, st)

    solver = tessera.ReggeSolver(st, matter)
    n_edges = len(st.getEdgeList().toVector())
    print(f"  {n_edges} edges, {len(time_slices(st))} time slices, "
          f"worldline of {len(worldline)} vertices")

    covered = covered_edges(solver)
    print(f"  {int(covered.sum())} of {n_edges} edges carry an analytic second "
          f"derivative and are free to move; the rest are held fixed")

    prog.phase("relaxing", total=args.max_iters)
    converged, F, held_F, evaluations = relax_to_stationary_action(
        st, solver, worldline, args.mass, tol=args.tol,
        max_iters=args.max_iters, band=args.band, progress=prog)
    print(f"  {'solved' if converged else 'not solved'} to --tol={args.tol:g}: "
          f"F = ||dS/dl^2||^2 = {F:.6g} after {evaluations} gradient "
          f"evaluations")
    print(f"  of that residual, {held_F:.6g} "
          f"({100.0 * held_F / F if F else 0.0:.1f}%) sits on the held edges "
          f"and is a floor this relaxation cannot reach")

    prog.phase("rendering", extra=args.save)
    render_curvature_gif(st, solver, worldline, args.save)
    prog.finish(f"saved {args.save}")


if __name__ == "__main__":
    main()
