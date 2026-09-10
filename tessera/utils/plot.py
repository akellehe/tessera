"""Shared utilities for tessera example scripts.

Thin Python wrappers over C++ internals for spacetime construction,
graph analysis, and force-directed layout — plus matplotlib helpers
for 3D rendering and GIF assembly.

Usage::

    from tessera.utils.plot import build_spacetime, spatial_subgraph, force_layout_3d
    from tessera.utils.plot import render_frame, draw_edges, save_gif
"""
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

import tessera


# =========================================================================
# Spacetime construction
# =========================================================================

def build_spacetime(n_simplices, *, k0=2.2, k4=0.5, delta=0.6,
                    epsilon=None, nSweeps=10, topology=None, seed=None):
    """Build, tune, and thermalize a 4D Lorentzian CDT spacetime.

    If *epsilon* is None (the default), it is set to ``1/target_N41``
    so the volume-fixing penalty stays proportionate at any lattice size.

    If *seed* is given it is applied to both random number generators that
    decide the outcome -- the spacetime's, which drives the initial build, and
    the simulation's, which drives the Monte Carlo sweeps -- so the returned
    geometry is reproducible across processes. Left as None, each generator
    seeds itself from ``std::random_device`` and the geometry differs run to run.

    Returns (spacetime, cdt_simulation).
    """
    metric = tessera.Metric(
        coordinateFree=True,
        signature=tessera.Signature(dimensions=4,
                                  signatureType=tessera.Lorentzian),
    )
    topo = topology or tessera.Toroid()
    st = tessera.Spacetime(
        metric=metric, spacetimeType=tessera.CDT,
        alpha=1.0, a=1.0,
        foliation=tessera.PREFERRED, topology=topo,
    )
    # Cap at ~80 time slices so spatial volume per slice is large enough
    # for meaningful geometry.  The staircase product creates d*(d+1)=20
    # simplices per slab in 4D; building directly with large n_simplices
    # would create n/20 slices (e.g. 160k → 8000 razor-thin slices).
    max_build = 80 * 20  # 80 slabs × 20 simplices/slab in 4D
    if seed is not None:
        st.setSeed(seed)
    st.build(min(n_simplices, max_build))
    target = st.getN41() if n_simplices <= max_build else n_simplices // 2
    if epsilon is None:
        epsilon = 1.0 / max(target, 1)
    cdt = tessera.CDTSimulation(
        spacetime=st, k0=k0, k4=k4, delta=delta,
        epsilon=epsilon, targetN41=target,
    )
    if seed is not None:
        cdt.setSeed(seed)
    cdt.tune()
    if nSweeps > 0:
        cdt.sweep(nSweeps)
    return st, cdt


# =========================================================================
# Time-slice utilities (delegates to C++)
# =========================================================================

def time_slices(st):
    """Sorted list of integer time values present in the spacetime."""
    return st.getTimeSlices()


def spatial_subgraph(st, t):
    """Return ``(vertices, spacelike_edges)`` for time slice *t*.

    Delegates to the C++ ``Spacetime::getSpatialSubgraph``.
    """
    return st.getSpatialSubgraph(t)


def bfs_distances(center, st_or_verts=None, edges=None, *, max_depth=None):
    """BFS shortest-path distances from *center* through spacelike edges.

    If called with a Spacetime object::

        bfs_distances(center, st, max_depth=5)

    delegates to the C++ ``Spacetime::bfsDistances``.

    If called with explicit verts and edges (legacy)::

        bfs_distances(center, verts, edges, max_depth=5)

    falls back to a Python BFS.
    """
    if isinstance(st_or_verts, tessera.Spacetime):
        return st_or_verts.bfsDistances(center,
                                        -1 if max_depth is None else max_depth)
    # Legacy path: explicit verts + edges
    from collections import defaultdict, deque
    verts, edges_ = st_or_verts, edges
    adj = defaultdict(list)
    for e in edges_:
        s, t = e.getSource().getId(), e.getTarget().getId()
        adj[s].append(t)
        adj[t].append(s)
    dist = {center.getId(): 0}
    queue = deque([center.getId()])
    while queue:
        vid = queue.popleft()
        if max_depth is not None and dist[vid] >= max_depth:
            continue
        for nbr in adj[vid]:
            if nbr not in dist:
                dist[nbr] = dist[vid] + 1
                queue.append(nbr)
    return dist


# =========================================================================
# Force-directed 3D layout (delegates to C++)
# =========================================================================

def force_layout_3d(n, edges, *, center_idx=None, init_pos=None,
                    spring_k=0.01, repulsion_k=0.5, rest_lengths=None,
                    iters=300, cooling=0.995, repulsion_cap=200,
                    seed=42):
    """Spring-electrical force-directed layout in 3D.

    Delegates to ``tessera.ForceLayout.layout3D`` (C++) for performance.

    Returns an ``(n, 3)`` numpy array of positions.
    """
    flat = tessera.ForceLayout.layout3D(
        n=n,
        edges=edges,
        centerIdx=center_idx if center_idx is not None else -1,
        initPos=list(init_pos.ravel()) if init_pos is not None else [],
        restLengths=list(rest_lengths) if rest_lengths is not None else [],
        springK=spring_k,
        repulsionK=repulsion_k,
        iters=iters,
        cooling=cooling,
        repulsionCap=repulsion_cap,
        seed=seed,
    )
    return np.array(flat).reshape(n, 3)


def radial_layout_2d(n, edges, target_radii, *, center_idx=None,
                     init_pos=None, spring_k=0.02, repulsion_k=0.3,
                     rest_lengths=None, iters=200, cooling=0.995,
                     repulsion_cap=200, initial_step=0.3, seed=42):
    """Radius-constrained 2D radial force-directed layout.

    Delegates to ``tessera.ForceLayout.layout2D`` (C++).  Each node's radius
    is pinned to ``target_radii``; only the angular coordinate is solved.

    Returns an ``(n, 2)`` numpy array of positions.
    """
    flat = tessera.ForceLayout.layout2D(
        n=n,
        edges=edges,
        targetRadii=list(target_radii),
        centerIdx=center_idx if center_idx is not None else -1,
        initPos=list(init_pos.ravel()) if init_pos is not None else [],
        restLengths=list(rest_lengths) if rest_lengths is not None else [],
        springK=spring_k,
        repulsionK=repulsion_k,
        iters=iters,
        cooling=cooling,
        repulsionCap=repulsion_cap,
        initialStep=initial_step,
        seed=seed,
    )
    return np.array(flat).reshape(n, 2)


def layout_from_spacetime(verts, edges, **kwargs):
    """Convenience: build index mappings and run force_layout_3d.

    Returns ``(positions, vid_to_idx, edge_idx_list)``.
    """
    vid_to_idx = {v.getId(): i for i, v in enumerate(verts)}
    edge_idx = []
    rest_lens = []
    for e in edges:
        si = vid_to_idx.get(e.getSource().getId())
        ti = vid_to_idx.get(e.getTarget().getId())
        if si is not None and ti is not None:
            edge_idx.append((si, ti))
            # Layout-only collapse of the complex signed l^2 to a positive
            # rest length |Re l^2| (Im and the causal sign are deliberately
            # NOT drawn), with an epsilon floor so null edges don't pin two
            # vertices together — the multicobordism animation's convention
            # (#581).
            rest_lens.append(
                math.sqrt(max(abs((e.getLength()**2).real), 1e-6)))

    pos = force_layout_3d(len(verts), edge_idx,
                          rest_lengths=rest_lens, **kwargs)
    return pos, vid_to_idx, edge_idx


# =========================================================================
# PCA alignment (for frame-to-frame consistency)
# =========================================================================

def pca_align(pos, prev_axes=None):
    """Align positions to PCA axes with sign/permutation consistency.

    Returns ``(aligned_positions, axes_3x3)``.
    """
    from itertools import permutations as perms

    centroid = pos.mean(axis=0)
    centered = pos - centroid
    if centered.shape[0] < 2:
        return centered, np.eye(3)

    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = Vt[:3]

    if prev_axes is not None:
        best_perm, best_score = None, -1.0
        for perm in perms(range(min(3, len(axes)))):
            score = sum(abs(np.dot(axes[perm[i]], prev_axes[i]))
                        for i in range(3))
            if score > best_score:
                best_score = score
                best_perm = perm
        axes = axes[list(best_perm)]
        for i in range(3):
            if np.dot(axes[i], prev_axes[i]) < 0:
                axes[i] = -axes[i]

    return centered @ axes.T, axes


# =========================================================================
# 3D rendering (matplotlib — stays in Python)
# =========================================================================

def render_frame(draw_fn, *, figsize=(7, 7), elev=25, azim=45, title=None):
    """Create a 3D figure, call ``draw_fn(ax)``, and return an RGBA array.

    Example::

        def draw(ax):
            draw_edges(ax, pos, edges)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

        img = render_frame(draw, title="My plot", azim=30)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    draw_fn(ax)

    if title:
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


def draw_edges(ax, pos, edges, *, edge_types=None,
               timelike_color=(0.15, 0.30, 0.80, 0.75),
               spacelike_color=(0.0, 0.0, 0.0, 0.6),
               default_color=(0.5, 0.5, 0.5, 0.5),
               linewidth=0.7, linestyle="solid"):
    """Draw edges as a Line3DCollection.

    If *edge_types* is provided (list of bools, True=timelike), edges
    are colored by type.  Otherwise all edges use *default_color*.
    """
    if not edges:
        return
    segs = [[pos[a], pos[b]] for a, b in edges]
    if edge_types is not None:
        colors = [timelike_color if tl else spacelike_color
                  for tl in edge_types]
    else:
        colors = default_color
    lc = Line3DCollection(segs, linewidths=linewidth, colors=colors,
                          linestyles=linestyle)
    ax.add_collection(lc)


# =========================================================================
# GIF assembly (PIL — stays in Python)
# =========================================================================

def save_gif(frames, path, *, duration_ms=200):
    """Save a list of RGBA numpy arrays as an animated GIF.

    Drops the alpha channel automatically.
    """
    pil_frames = [Image.fromarray(f[:, :, :3]) for f in frames]
    pil_frames[0].save(path, save_all=True,
                       append_images=pil_frames[1:],
                       duration=duration_ms, loop=0)
