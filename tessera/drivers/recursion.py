# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Several ticks of the level recursion, with the grown-cell rule.

This driver runs the whitepaper's level recursion (v17, Section 15) for several
ticks on a level-0 host of three-sheeted unit-monopole tetrahedra that share
faces. One tick is the Section 15 box followed by the interaction stage: the
reduction produces the response vertices and their fibers, and the interactions
among those fibers attach the cells of the next level, whose squared lengths
and connection values come from the grown-cell rule. Nothing is tuned toward a
target; every number is reported as it comes out.

Mode
----
Labelled controlled synthesis (WP §3.2): the host, its sheets and its
monopoles are declared, not emerged.

The host
--------
The base complex is a fan of ``--tetrahedra`` tetrahedra around the edge
(0, 1): tetrahedron k has vertices (0, 1, 2 + k, 3 + k), so consecutive
tetrahedra share a face. Every edge has squared length ``--edge-squared``.
The declared default is two tetrahedra: with the host's bounding-cut sector
held, the Newton solve on a fan of three or four tetrahedra stops at the
sector boundary unconverged and says so, while the fan of two is stationary
in it as built.
Every tetrahedron carries a unit Dirac monopole (WP §11.1): the outward
principal face angles theta_f, read in the outward orientation of
`MonopoleSupport.tetrahedron(1)` on the tetrahedron's ascending vertices, are
the least-norm solution of "total outward flux 2 pi mu on every tetrahedron",
and the edge phases are the solution of d phi = theta - 2 pi n, with the
integer Dirac string n placed on one unshared face of each tetrahedron. A face
shared by two tetrahedra is outward for one and inward for the other, which is
why the symmetric quarter-turn configuration of a lone tetrahedron is not
available here; the monopole number of every tetrahedron is read back from its
face holonomies. The level is three isomorphic, disjoint sheets of the base
(WP §8), attached sheet to sheet.

One tick
--------
At level l (a complex K_l of three sheets of a base complex):

1. both edge fields relax to holomorphic stationarity of the joint action,
   primal Regge + the paper's linear stiffness (1/kappa)(1/2)||l - l0||^2
   + the holonomy term (Villain by default, ``--holonomy``), in strict
   emergence (no carried density in the equations; WP §7), with l0 the
   level's own lengths as the level was built. The three sheets are relaxed
   as one shared base field (WP v17 §8): the solve's variables are the base
   complex's squared lengths and links, written to every sheet. The Regge term is read on
   Riemann sheets continued from the real projection of the level's starting
   geometry (`ReggeBranch.Continued`). A level with no interior hinge has a
   Regge term that is zero by structure; the record and the progress line
   say so. The monopole sector is boundary data held only on a declared
   cluster's bounding cut (WP v17 §3, §11.1). At level 0 the declared
   cluster is the host, one per sheet. Its bounding cut is every face of a
   host tetrahedron that no other host tetrahedron shares, oriented outward.
   The monopole number through that cut is held, and so are the moduli of
   the face holonomies on it, so a holonomy on the unit circle stays there.
   A bulk face, such as the face two host tetrahedra share, is not held: its
   holonomy relaxes freely, modulus included. A grown level has no declared
   cluster, so nothing is held there. Every tetrahedron's monopole number is
   read before and after the relaxation and recorded as bulk data;
2. the Section 15 box runs on the base: the partition of the covariant
   edge-mode operator h_1(z, U) of sheet 0 (`LevelRecursion`, the library's
   `PersistentPartition`, a modularity sweep over the declared resolutions),
   the Riesz fibers of the declared rank, the Feshbach reduction with its
   determinant-factorization certificate, the transports M_vw and the labeled
   sum. On sheet-diagonal couplings the reduction acts on the base and carries
   the sheet factor along (WP §15), so the other two sheets carry the same
   fibers; the sheet isomorphism of the relaxed level is certified. A
   component becomes a response vertex only if it persists across the stated
   range of scales (WP §5), which is every declared resolution by default
   (``persistence_required``). The other components are recorded by name as
   rejected;
3. the interaction stage: each accepted component's fiber is the Riesz band
   of the image pencil (A~_1^U, M_1^U) restricted to the component's edges, so
   its geometric image Z = G_1^U Y is supported on the component and its
   chain Y = M_1^U Z on the component's one-ring (spec Prop. 5.3, WP §5).
   The left frame is the geometric image of the dual band
   (spec Prop. 4). Two response vertices interact exactly when their image
   supports share a top simplex, which is exactly when their coupling block
   Z_v^T M_1 Z_w can be nonzero. Four vertices that interact pairwise span a
   grown 3-simplex (the flag complex of the interaction graph);
4. the grown-cell rule (WP v17 §15): each fiber Y_v of h_1(z, U) is paired
   with the fiber Y_v^vee of the same component of h_1(z, U^{-1}), selected
   by the same rule. The dual frame is normalized so that the edge integrals
   of the two geometric images reproduce the level-0 partition of unity,
   det((Z_v^vee)^T Z_v) = 3, the number of edges at a vertex of a 3-simplex.
   The gauge-invariant inherited pairing on the determinant line is
   g_vv = det((Y_v^vee)^T G_1^U Y_v) and
   g_vw = det((Y_v^vee)^T G_1^U Y_w) / det M_vw, read as |T| Gamma, so
   C Gamma = g / 20. The block of C Gamma off v_0 is inverted to g/C, with
   C = 14400 / det(g/C), and the squared lengths are the quadratic form of g
   on the edge vectors. The row-sum defect ||g 1|| / ||g|| of every grown cell
   is its certificate. The connection of a grown edge (v < w) is
   U_vw = det M_vw. It carries the units of the operator (1/length^2), and
   no dimensionless replacement has passed the level-0 and gauge checks. The
   grown cells are glued in ascending lexicographic order of their vertices.
   A cell is added only if the complex stays a manifold with boundary
   (`SurgicalCone.validate`); a cell that fails is recorded with the
   violation. An edge shared by several grown cells receives one value from
   each; the level carries their mean and reports their spread;
5. the reads, behind the certificate firewall: every tetrahedron of the base
   of K_l, read as a three-sheeted host of its own (``baryon_poles``), gives
   the v16 quark verdicts (`QuarkConditions`), the isospin-doublet reading
   (`IsospinDoublet`), the spin decomposition of the occupied modes on the
   T-averaged operator, and the baryon poles, quasi-free and with the
   Section 7 quartic with the connection's phase fluctuations eliminated, for
   every declared content. A content names occupations of the bands of the
   covariant operator h_1 in ascending order of real part, which on the
   monopole host are one simple mode per sheet, not spin doublets
   (``baryon_poles``, "What a content names"); the poles are read for every
   doublet content of the T-averaged operator and labelled by it.

The next tick runs on K_{l+1}. The recursion stops at a level with no grown
3-simplex, and says so.

Running it
----------
::

    python -m tessera.drivers.recursion run --ticks 3 --json recursion.json \\
        --out recursion.png [--live]

``--live`` draws each completed tick while the run proceeds, on an interactive
matplotlib backend, with the computation on a worker thread and the main
thread servicing the GUI event loop. Closing the window switches the run to
headless; it continues and still writes every output. With ``--json`` every
tick's record is appended to ``<json stem>.points.jsonl`` the moment the tick
completes. A non-interactive backend, and WebAgg, are refused by name.
"""

import argparse
import cmath
import itertools
import json
import math
import sys
import time

import numpy as np

import tessera as T
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import baryon_poles as bp

#: Declared inputs, recorded with every run.
DECLARED_TETRAHEDRA = 2
DECLARED_EDGE_SQUARED = 8.0
DECLARED_MONOPOLE = 1
DECLARED_TICKS = 3
DECLARED_KAPPA = 1.0
DECLARED_BETA = 1.0
DECLARED_RESOLUTIONS = (1.0, 1.5, 2.0, 2.5, 3.0)
DECLARED_BAND_RANK = 1
DECLARED_CONTOUR_NODES = 64
#: The value of the coordinate pairing det((Z_v^vee)^T Z_v) of a fiber's two
#: geometric images: the number of edges at a vertex of a 3-simplex, which is
#: the pairing of the images of the level-0 vertex chains, so that a level-0
#: vertex read as a fiber keeps its own normalization. It is the declared unit
#: calibration of the grown lengths.
IMAGE_PAIRING_UNIT = 3.0
SHEETS = 3
#: The outward faces of `MonopoleSupport.tetrahedron`, by local vertex.
FIXTURE_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
LIVE_POLL_INTERVAL = bp.LIVE_POLL_INTERVAL


# ---------------------------------------------------------------- the host


def fan(count):
    """The base cells of the declared host: ``count`` tetrahedra around the
    edge (0, 1), consecutive ones sharing a face."""
    return [[0, 1, 2 + k, 3 + k] for k in range(count)]


def _sorted_simplices(cells, order):
    return sorted({tuple(sorted(s)) for c in cells
                   for s in itertools.combinations(sorted(c), order)})


def _oriented_face(triangle, face_index):
    """The canonical face index and the sign of the oriented triangle relative
    to the ascending orientation."""
    ascending = tuple(sorted(triangle))
    positions = [ascending.index(v) for v in triangle]
    sign = 1
    for i in range(3):
        for j in range(i + 1, 3):
            if positions[i] > positions[j]:
                sign = -sign
    return face_index[ascending], sign


def monopole_connection(cells, monopole=DECLARED_MONOPOLE):
    """The declared monopole connection of the base (see the module
    docstring): the principal face angles, the Dirac string, the edge phases
    on the ascending orientation, and the consistency residual of
    d phi = theta - 2 pi n."""
    cells = [sorted(c) for c in cells]
    edges = _sorted_simplices(cells, 2)
    faces = _sorted_simplices(cells, 3)
    edge_index = {e: i for i, e in enumerate(edges)}
    face_index = {f: i for i, f in enumerate(faces)}
    outward = np.zeros((len(cells), len(faces)))
    for t, cell in enumerate(cells):
        for p, q, r in FIXTURE_FACES:
            f, sign = _oriented_face((cell[p], cell[q], cell[r]), face_index)
            outward[t, f] += sign
    flux = 2.0 * math.pi * monopole * np.ones(len(cells))
    theta = outward.T @ np.linalg.solve(outward @ outward.T, flux)
    coboundary = np.zeros((len(faces), len(edges)))
    for f, (a, b, c) in enumerate(faces):
        coboundary[f, edge_index[(a, b)]] += 1.0
        coboundary[f, edge_index[(b, c)]] += 1.0
        coboundary[f, edge_index[(a, c)]] -= 1.0
    boundary = np.zeros((len(cells), len(faces)))
    for t, cell in enumerate(cells):
        for i in range(4):
            face = tuple(v for k, v in enumerate(cell) if k != i)
            boundary[t, face_index[face]] += (-1) ** i
    shared = {f for f in range(len(faces))
              if np.count_nonzero(boundary[:, f]) > 1}
    string = np.zeros(len(faces))
    turns = np.rint(boundary @ theta / (2.0 * math.pi))
    for t in range(len(cells)):
        free = [f for f in range(len(faces))
                if boundary[t, f] != 0 and f not in shared and string[f] == 0]
        string[free[0]] = turns[t] / boundary[t, free[0]]
    target = theta - 2.0 * math.pi * string
    phases, *_ = np.linalg.lstsq(coboundary, target, rcond=None)
    return {
        "edges": edges,
        "faces": faces,
        "face_angles": theta,
        "dirac_string": string,
        "phases": phases,
        "residual": float(np.linalg.norm(coboundary @ phases - target)),
    }


def build_level(cells, squared_lengths, links, sheets=SHEETS):
    """A level: ``sheets`` disjoint copies of the base ``cells`` (vertices
    0..n-1), sheet t on vertices t n .. t n + n - 1, with ``squared_lengths``
    and ``links`` (dicts on ascending base edges) on corresponding edges."""
    count = 1 + max(max(c) for c in cells)
    tuples = [[v + t * count for v in c] for t in range(sheets) for c in cells]
    spacetime = T.Spacetime.fromVertexTuples(3, tuples, 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        a = int(edge.getSource().getId())
        b = int(edge.getTarget().getId())
        sheet = a // count
        x, y = a - sheet * count, b - sheet * count
        key = (min(x, y), max(x, y))
        link = complex(links[key])
        if x > y:
            link = 1.0 / link
        edge.setLength(cmath.sqrt(complex(squared_lengths[key])))
        edge.setPhase(complex(-1j * cmath.log(link)))
    return spacetime, count


def level_zero(config):
    """The declared level-0 base: cells, squared lengths and links."""
    cells = fan(config["tetrahedra"])
    connection = monopole_connection(cells, config["monopole"])
    z = {e: complex(config["edge_squared"]) for e in connection["edges"]}
    links = {e: cmath.exp(1j * p) for e, p in
             zip(connection["edges"], connection["phases"])}
    return cells, z, links, connection


def sheet_fields(spacetime, count, sheets=SHEETS):
    """Per sheet, the squared length and the link of every ascending base
    edge, read off the level's mesh."""
    fields = [({}, {}) for _ in range(sheets)]
    for edge in spacetime.getEdgeList().toVector():
        a = int(edge.getSource().getId())
        b = int(edge.getTarget().getId())
        sheet = a // count
        x, y = a - sheet * count, b - sheet * count
        link = cmath.exp(1j * complex(edge.getPhase()))
        if x > y:
            x, y, link = y, x, 1.0 / link
        fields[sheet][0][(x, y)] = complex(edge.getLength()) ** 2
        fields[sheet][1][(x, y)] = link
    return fields


def monopole_numbers(cells, links):
    """The monopole number of every base tetrahedron, read with
    `MonopoleSupport` on its ascending vertices."""
    fixture = obs.MonopoleSupport.tetrahedron(1)
    numbers = []
    for cell in cells:
        c = sorted(cell)
        values = [links[(c[i], c[j])] for i, j in fixture.edges]
        support = obs.MonopoleSupport(4, fixture.edges, fixture.faces,
                                      obs.MonopoleSupport.u1Part(values))
        numbers.append(int(support.monopoleNumber().monopole_number))
    return numbers


# ------------------------------------------------------------ relaxation


def held_sectors(cells, numbers, count, sheets=SHEETS):
    """The monopole sectors held as boundary data during a relaxation: every
    tetrahedron of every sheet, its bounding cut the four faces of
    `MonopoleSupport.tetrahedron` in their outward orientation on its ascending
    vertices, with its declared monopole number."""
    sectors = []
    for t in range(sheets):
        for cell, number in zip(cells, numbers):
            c = [v + t * count for v in sorted(cell)]
            sector = cob.HeldMonopoleSector()
            sector.faces = [[c[p], c[q], c[r]] for p, q, r in FIXTURE_FACES]
            sector.monopole_number = int(number)
            sectors.append(sector)
    return sectors


def outward_faces(cell):
    """The four faces of a tetrahedron, each oriented outward
    (`MonopoleSupport.tetrahedron`'s orientation on its ascending
    vertices)."""
    c = sorted(cell)
    return [(c[p], c[q], c[r]) for p, q, r in FIXTURE_FACES]


def bounding_cut(cells):
    """The bounding cut of the cluster made of ``cells``: every face of one of
    its tetrahedra that no other of its tetrahedra shares, oriented outward. A
    face two of the tetrahedra share is bulk and is not on the cut."""
    counts = {}
    for cell in cells:
        for face in outward_faces(cell):
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
    return [face for cell in cells for face in outward_faces(cell)
            if counts[tuple(sorted(face))] == 1]


def _oriented_link(links, a, b):
    return links[(a, b)] if a < b else 1.0 / links[(b, a)]


def face_holonomy(links, face):
    """The holonomy U_ab U_bc U_ca of an oriented face."""
    a, b, c = face
    return (_oriented_link(links, a, b) * _oriented_link(links, b, c)
            * _oriented_link(links, c, a))


def cut_monopole_number(faces, links):
    """The monopole number through a cut: the sum of the principal arguments
    of its outward face holonomies over 2 pi, as `HolomorphicRelaxation`
    reads a held sector."""
    total = sum(cmath.phase(face_holonomy(links, f)) for f in faces)
    return int(round(total / (2.0 * math.pi)))


def cut_sectors(faces, number, count, sheets=SHEETS):
    """The held sectors of a declared cluster present on every sheet: per
    sheet, the outward faces of its bounding cut with the monopole number
    through it."""
    sectors = []
    for t in range(sheets):
        sector = cob.HeldMonopoleSector()
        sector.faces = [[v + t * count for v in face] for face in faces]
        sector.monopole_number = int(number)
        sectors.append(sector)
    return sectors


def level_edge_classes(spacetime, count):
    """The shared base field of a level built by `build_level` (WP v17 §8):
    for every edge in `getEdgeList()` order, the index of its ascending base
    edge among the level's base edges, and the orientation of its stored link
    relative to the base edge (+1 when stored ascending, -1 otherwise)."""
    keys, classes, orientations = {}, [], []
    for edge in spacetime.getEdgeList().toVector():
        a = int(edge.getSource().getId())
        b = int(edge.getTarget().getId())
        sheet = a // count
        x, y = a - sheet * count, b - sheet * count
        key = (min(x, y), max(x, y))
        classes.append(keys.setdefault(key, len(keys)))
        orientations.append(1 if x < y else -1)
    return classes, orientations


def relax_level(spacetime, config, sectors=None, count=None):
    """Step 1: both edge fields relax to holomorphic stationarity of the joint
    action, strict emergence, with the level's lengths as built as l0, and
    with the declared monopole sectors held as boundary data. With ``count``,
    the base vertex count of a level built by `build_level`, the sheets are
    relaxed as one shared base field (`level_edge_classes`), so they stay
    identical exactly."""
    declaration = bp.action_declaration(spacetime, config["kappa"],
                                        config["beta"],
                                        config["regge_hinges"],
                                        holonomy=config["holonomy"])
    action = cob.JointAction(spacetime, declaration)
    held = dict(config)
    held["held_sectors"] = list(sectors or [])
    geometry = bp.relaxation_declaration(held)
    if count is not None:
        classes, orientations = level_edge_classes(spacetime, count)
        geometry.edge_classes = classes
        geometry.edge_class_orientations = orientations
    relaxation = cob.HolomorphicRelaxation(action, geometry)
    started = time.time()
    report = relaxation.solve()
    return {
        "converged": bool(report.converged),
        "initial_residual": float(report.initial_residual_norm),
        "residual": float(report.residual_norm),
        "iterations": len(report.steps),
        "zero_guard_damped_steps": int(report.zero_guard_damped_steps),
        "sector_guard_damped_steps": int(report.sector_guard_damped_steps),
        "sector_monopole_numbers": list(report.sector_monopole_numbers),
        "held_modulus_drift": float(report.held_modulus_drift),
        "shared_sheet_geometry": count is not None,
        "rank_tolerance": float(geometry.rank_tolerance),
        "jacobian_ranks": [int(st.jacobian_rank) for st in report.steps],
        "rank_gaps": [float(st.rank_gap) for st in report.steps],
        "regge_hinges": config["regge_hinges"],
        "regge_hinge_count": int(report.regge_hinge_count),
        "regge_structurally_zero": bool(report.regge_structurally_zero),
        "regge_off_principal_angles": int(report.regge_off_principal_angles),
        "action": complex(report.action),
        "seconds": time.time() - started,
    }


# ------------------------------------------------------- the box on the base


def base_operator(cells, z, links):
    """The Whitney complex of the base, its canonical edges and top simplices,
    the covariant chain-Hodge instance of sheet 0 and its h_1(z, U)."""
    tuples = [sorted(c) for c in cells]
    complex_ = cob.ChainComplex.fromTopCells(tuples)
    edges = [tuple(e) for e in complex_.kSimplexVertices(1)]
    tops = [tuple(t) for t in complex_.kSimplexVertices(3)]
    squared = [complex(z[e]) for e in edges]
    connection = ch.Connection(complex_, [complex(links[e]) for e in edges])
    hodge = ch.ChainHodge(complex_, squared)
    covariant = ch.CovariantChainHodge(hodge, connection)
    dual = ch.CovariantChainHodge(hodge, connection.inverse())
    operator = np.asarray(covariant.covariantOperator(1))
    dual_operator = np.asarray(dual.covariantOperator(1))
    pencil, dual_pencil = covariant.pencil(1), dual.pencil(1)
    return {"complex": complex_, "edges": edges, "tops": tops,
            "covariant": covariant, "operator": operator,
            "dual_operator": dual_operator,
            "pencil": np.asarray(pencil.A), "metric": np.asarray(pencil.B),
            "dual_pencil": np.asarray(dual_pencil.A),
            "dual_metric": np.asarray(dual_pencil.B)}


def recursion_turn(operator, config):
    """One turn of the Section 15 box on the base operator."""
    declaration = cob.LevelRecursionDeclaration()
    declaration.resolutions = list(config["resolutions"])
    bands = cob.RecursionBandDeclaration()
    bands.selection = cob.RecursionBandSelection.LowestModes
    bands.band_rank = config["band_rank"]
    bands.contour_nodes = config["contour_nodes"]
    declaration.bands = bands
    n = operator.shape[0]
    recursion = cob.LevelRecursion.overPencil(list(operator.reshape(-1)), [],
                                              n, declaration)
    recursion.advance()
    return recursion.level(0)


def riesz_band(block, rank, nodes):
    """The band of a component block by the rule `LevelRecursion` declares as
    `LowestModes`: the ``rank`` eigenvalues first in ascending real part are
    enclosed by the circle about their mean whose radius sits halfway to the
    nearest excluded eigenvalue; the projector is the contour integral by the
    trapezoidal rule; the right frame is the leading left singular vectors of
    the projector and the left frame its algebraic dual."""
    values = np.linalg.eigvals(block)
    ordered = sorted(values, key=lambda v: (v.real, v.imag))
    r = min(rank, len(ordered))
    centre = np.mean(ordered[:r])
    inside = max(abs(v - centre) for v in ordered[:r])
    rest = [abs(v - centre) for v in ordered[r:]]
    radius = 0.5 * (inside + min(rest)) if rest else inside + max(1.0, inside)
    n = block.shape[0]
    projector = np.zeros((n, n), dtype=complex)
    for k in range(nodes):
        root = np.exp(2j * np.pi * k / nodes)
        zeta = centre + radius * root
        projector += (radius * root / nodes) * np.linalg.inv(
            zeta * np.eye(n) - block)
    left_singular, _, _ = np.linalg.svd(projector)
    right = left_singular[:, :r]
    left = np.linalg.solve(right.conj().T @ right, right.conj().T @ projector)
    return {"frame": right, "left": left, "projector": projector,
            "eigenvalues": list(ordered[:r]), "centre": complex(centre),
            "radius": float(radius)}


def normalize_image_pairing(image, dual_image,
                            unit=IMAGE_PAIRING_UNIT):
    """The dual image with its first column divided by
    det((Z^vee)^T Z) / unit, so that det((Z^vee)^T Z) = unit afterwards. No
    root is taken. Returns the normalized dual image and the divisor."""
    pairing = complex(np.linalg.det(dual_image.T @ image))
    if pairing == 0 or not np.isfinite(abs(pairing)):
        raise ValueError("a fiber's two geometric images pair singularly")
    scale = pairing / unit
    out = np.array(dual_image, dtype=complex)
    out[:, 0] /= scale
    return out, scale


def fibers_for_partition(base, partition, rank, nodes):
    """The fibers of every component, supported on their geometric images
    (spec Prop. 5.3).

    The band of a component is the Riesz band of the image pencil
    (A~_1^U, M_1^U) restricted to the component's edges, selected by the rule
    of `riesz_band`. Its frame is the geometric image Z = G_1^U Y, which is
    zero off the component. The chain frame is Y = M_1^U Z, which is zero off
    the component's one-ring. The dual band is the same for U^{-1}. The dual
    image is normalized to det((Z^vee)^T Z) = 3 (`normalize_image_pairing`).
    The left frame is the geometric image of the dual band,
    Y~^T = B^{-1} (Z^vee)^T with B = (Y^vee)^T G_1^U Y = (Z^vee)^T M_1^U Z
    (spec Prop. 4), so Y~^T Y = I.

    Returns a dict of per-fiber lists: ``frames`` (Y), ``images`` (Z),
    ``lefts`` (Y~^T), ``duals`` (Y^vee), ``dual_images`` (Z^vee) and
    ``reads``."""
    pencil, metric = base["pencil"], base["metric"]
    dual_pencil, dual_metric = base["dual_pencil"], base["dual_metric"]
    n = pencil.shape[0]
    out = {key: [] for key in ("frames", "images", "lefts", "duals",
                               "dual_images", "reads")}
    for part in partition:
        index = list(part)
        block = np.ix_(index, index)
        band = riesz_band(np.linalg.solve(metric[block], pencil[block]),
                          rank, nodes)
        dual = riesz_band(np.linalg.solve(dual_metric[block],
                                          dual_pencil[block]), rank, nodes)
        r = band["frame"].shape[1]
        image = np.zeros((n, r), dtype=complex)
        dual_image = np.zeros((n, dual["frame"].shape[1]), dtype=complex)
        image[index, :] = band["frame"]
        dual_image[index, :] = dual["frame"]
        if dual_image.shape[1] == r:
            dual_image, _ = normalize_image_pairing(image, dual_image)
        frame = metric @ image
        dual_frame = dual_metric @ dual_image
        restriction = dual_image.T @ metric @ image
        left = np.linalg.solve(restriction, dual_image.T)
        mismatch = (max(abs(a - b) for a, b in zip(
            sorted(band["eigenvalues"], key=lambda v: (v.real, v.imag)),
            sorted(dual["eigenvalues"], key=lambda v: (v.real, v.imag))))
            if len(dual["eigenvalues"]) == r else float("inf"))
        out["frames"].append(frame)
        out["images"].append(image)
        out["lefts"].append(left)
        out["duals"].append(dual_frame)
        out["dual_images"].append(dual_image)
        out["reads"].append({
            "projector": band["projector"],
            "eigenvalues": band["eigenvalues"],
            "dual_eigenvalue_mismatch": float(mismatch),
            "restriction_determinant": complex(np.linalg.det(restriction)),
            "pairing_defect": float(np.linalg.norm(left @ frame
                                                   - np.eye(r))),
        })
    return out


def transport_matrix(pencil, images, lefts):
    """M_vw = Y~_v^T h_1 Y_w = Y~_v^T A~_1^U Z_w for every pair, in the
    frames given (spec Prop. 5.4: the transfer from the pencil block)."""
    return {(v, w): lefts[v] @ pencil @ images[w]
            for v in range(len(images)) for w in range(len(images))}


def level_record(level):
    """The partition and its certificates."""
    return {
        "partition": [list(p) for p in level.partition],
        "resolutions": list(level.resolutions),
        "selected_resolution": float(level.selected_resolution),
        "component_persistence": list(level.component_persistence),
        "band_ranks": [int(b.rank) for b in level.bands],
        "bands_accepted": [bool(b.accepted) for b in level.bands],
        "isolation_gaps": [float(b.isolation_gap) for b in level.bands],
        "projector_idempotency": [float(b.projector_idempotency)
                                  for b in level.bands],
        "pairing_defects": [float(b.pairing_defect) for b in level.bands],
        "band_eigenvalues": [[complex(v) for v in b.eigenvalues]
                             for b in level.bands],
        "gram_defect": float(level.gram_defect),
        "modes": int(level.modes),
        "fock_stage_dimension": float(level.fock_stage_dimension),
        "response_dimension": int(level.response_dimension),
        "determinant_residual": float(level.determinant_residual),
        "certificate": level.certificate.describe(),
        "reduction_certificate": level.reduction_certificate.describe(),
    }


# ------------------------------------------------------ the interaction stage


def interaction_graph(partition, edges, tops):
    """Two response vertices interact when some top simplex of the base
    contains an edge of each (their Whitney-metric coupling block is nonzero,
    spec Prop. 7.1). Returns the adjacency as a set of pairs (v < w)."""
    in_top = [{i for i, e in enumerate(edges) if set(e) <= set(t)}
              for t in tops]
    parts = [set(p) for p in partition]
    pairs = set()
    for v, w in itertools.combinations(range(len(parts)), 2):
        if any(parts[v] & top and parts[w] & top for top in in_top):
            pairs.add((v, w))
    return pairs


def grown_cells(vertex_count, pairs):
    """The 3-simplices of the flag complex: four response vertices that
    interact pairwise."""
    return [q for q in itertools.combinations(range(vertex_count), 4)
            if all(p in pairs for p in itertools.combinations(q, 2))]


def inherited_pairing(frames, duals, transports, covariant, images=None):
    """The gauge-invariant inherited pairing of the grown-cell rule: the
    dual-connection frame of v paired with the image of the frame of w through
    G_1^U on the determinant line, divided by U_vw = det M_vw off the
    diagonal (`GrownCellRule.gaugeInvariantPairing`). ``images`` are the
    geometric images G_1^U Y when the caller already holds them exactly; they
    are solved for otherwise."""
    n = len(frames)
    if n == 0:
        return np.zeros((0, 0), dtype=complex)
    if images is None:
        images = [np.asarray(covariant.applyG(1, Y)) for Y in frames]
    connection = np.ones((n, n), dtype=complex)
    for (v, w), block in transports.items():
        if v != w:
            connection[v, w] = (
                complex(ch.GrownCellRule.transportConnection(block))
                if block.shape[0] == block.shape[1] and block.size else
                complex("nan"))
    return np.asarray(ch.GrownCellRule.gaugeInvariantPairing(
        duals, images, connection))


def level_rule_shift(cells, z, links):
    """The grown-cell rule applied to a level's own tetrahedra as if they were
    grown cells: on each tetrahedron alone, the vertex fibers are the exact
    chains of the twisted coboundary, their dual-connection partners those of
    the inverse connection, and the transports the edge links. The relative
    shift of the rule's squared lengths from the tetrahedron's own is zero for
    a pure-gauge connection; with face holonomies it is the curvature-induced
    shift of the rule."""
    fixture_edges = list(itertools.combinations(range(4), 2))
    single = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    shifts = []
    for cell in cells:
        c = sorted(cell)
        s_local = [complex(z[(c[i], c[j])]) for i, j in fixture_edges]
        block = np.asarray(
            ch.WhitneyMass.topSimplexBlocks(single, s_local, 1)[0].block)
        U = {}
        for (i, j) in fixture_edges:
            U[(i, j)] = complex(links[(c[i], c[j])])
            U[(j, i)] = 1.0 / U[(i, j)]
        for v in range(4):
            U[(v, v)] = 1.0

        def dressed(link):
            return np.array([[block[a, b] * link[(fixture_edges[a][0],
                                                  fixture_edges[b][0])]
                              for b in range(6)] for a in range(6)])

        def coboundary(link):
            out = np.zeros((6, 4), dtype=complex)
            for a, (x, y) in enumerate(fixture_edges):
                out[a, x], out[a, y] = -1.0, link[(x, y)]
            return out

        inverse = {k: 1.0 / u for k, u in U.items()}
        forward = dressed(U)
        frames = forward @ coboundary(U)
        duals = dressed(inverse) @ coboundary(inverse)
        images = np.linalg.solve(forward, frames)
        connection = np.array([[U[(v, w)] for w in range(4)]
                               for v in range(4)])
        pairing = np.asarray(ch.GrownCellRule.gaugeInvariantPairing(
            [duals[:, [v]] for v in range(4)],
            [images[:, [v]] for v in range(4)], connection))
        read = ch.GrownCellRule.invertVertexPairing(pairing)
        rule = np.array(read.squaredLengths)
        shifts.append({"cell": c,
                       "relative_shift": float(np.max(np.abs(
                           rule / np.array(s_local) - 1.0))),
                       "row_sum_defect": float(read.rowSumDefect)})
    return shifts


def manifold_violation(cells):
    """None when the complex of ``cells`` is a manifold with boundary
    (`SurgicalCone.validate`); otherwise the violation it names."""
    spacetime = T.Spacetime.fromVertexTuples(
        3, [[int(v) for v in c] for c in cells], 1.0, 0.0)
    ok, reason = cob.SurgicalCone(spacetime).validate()
    return None if ok else str(reason)


def grow(cells, pairing, transports):
    """The grown-cell rule on every grown 3-simplex: lengths from the
    inherited pairing, connection from det M. The cells are glued in
    ascending lexicographic order of their vertices, and a cell is added only
    if the complex of the cells added so far stays a manifold with boundary;
    a cell that fails is recorded with the violation under ``"rejected"``.
    Returns the per-cell reads and the per-edge fields of the next level
    (response-vertex labels)."""
    reads, per_edge_z, glued = [], {}, []
    for cell in sorted(tuple(sorted(c)) for c in cells):
        index = list(cell)
        block = pairing[np.ix_(index, index)]
        entry = {"vertices": index, "pairing": block}
        try:
            read = ch.GrownCellRule.invertVertexPairing(block)
        except ValueError as error:
            entry["failed"] = str(error)
            reads.append(entry)
            continue
        violation = manifold_violation(glued + [index])
        if violation is not None:
            entry["rejected"] = violation
            reads.append(entry)
            continue
        glued.append(index)
        entry.update({
            "row_sum_defect": float(read.rowSumDefect),
            "asymmetry": float(read.asymmetry),
            "scale": complex(read.scale),
            "volume": complex(read.volume),
            "squared_lengths": [complex(v) for v in read.squaredLengths],
            "frame_invariant_ratios": np.asarray(read.frameInvariantRatios),
        })
        for m, (i, j) in enumerate(itertools.combinations(range(4), 2)):
            per_edge_z.setdefault((index[i], index[j]), []).append(
                complex(read.squaredLengths[m]))
        reads.append(entry)
    links, groupoid = {}, {}
    for (v, w) in per_edge_z:
        forward = ch.GrownCellRule.transportConnection(transports[(v, w)])
        backward = ch.GrownCellRule.transportConnection(transports[(w, v)])
        links[(v, w)] = complex(forward)
        groupoid[(v, w)] = float(abs(forward * backward - 1.0))
    z = {e: complex(np.mean(values)) for e, values in per_edge_z.items()}
    spread = {e: float(max(abs(x - z[e]) for x in values) / abs(z[e]))
              if abs(z[e]) > 0 else 0.0 for e, values in per_edge_z.items()}
    return reads, z, links, spread, groupoid


# ----------------------------------------------------------------- the reads


def cell_reads(cells, z, links, config):
    """Every base tetrahedron read as a three-sheeted host of its own: the
    quark verdicts, the isospin-doublet reading and the baryon poles of every
    declared content (`baryon_poles.evaluate_content`). A tetrahedron of the
    declared level-0 host is a declared host of its own, so its four faces
    are its bounding cut and are held. A tetrahedron of a grown level is not
    declared, so nothing on it is held (``hold_cell_sectors``)."""
    fixture = obs.MonopoleSupport.tetrahedron(1)
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group())
    chosen = cells if config["max_cells"] is None else \
        cells[:config["max_cells"]]
    out = []
    for cell in chosen:
        c = sorted(cell)
        host_cell = {
            "squared_lengths": [z[(c[i], c[j])] for i, j in fixture.edges],
            "links": [links[(c[i], c[j])] for i, j in fixture.edges],
        }
        cell_config = bp.default_config(
            kappas=[config["kappa"]], betas=[config["beta"]],
            regge_hinges=config["regge_hinges"], holonomy=config["holonomy"],
            elimination=config["elimination"],
            selected_contents=[tuple(x) for x in config["contents"]])
        cell_config["host_cell"] = host_cell
        cell_config["isospin_doublet"] = True
        number = monopole_numbers([c], links)[0]
        cell_config["held_sectors"] = (
            held_sectors([[0, 1, 2, 3]], [number], 4)
            if config.get("hold_cell_sectors", True) else [])
        point = bp.scan_point(config["kappa"], config["beta"], cell_config,
                              alignment)
        out.append({"cell": c, "host_cell": host_cell,
                    "failed_contents": point["failed_contents"],
                    "contents": point["contents"],
                    "ratios": point["ratios"],
                    "pole_table": point["pole_table"]})
    return out


def _verdict_summary(record):
    quark = record.get("quark_conditions")
    if not quark:
        return None
    return {"certified": quark["certified"],
            "status": {c["name"]: c["status"] for c in quark["conditions"]}}


def _lowest_poles(record):
    """Per spin and column, the lowest pole of a content over its doublet
    contents (`baryon_poles.lowest_poles`), without the doublet content."""
    out = {}
    for name in ("quasi_free", "with_quartic"):
        for key, best in bp.lowest_poles(record, name).items():
            if best is not None:
                out.setdefault(key, {})[name] = best[0]
    return out


def _truncation_summary(record):
    """The Section 7 quartic's truncation certificates of one content: how
    large the induced displacement is and how much of the operator's change is
    beyond linear order."""
    quartic = record.get("quartic")
    if not quartic:
        return None
    truncation = quartic.get("truncation") or {}
    keys = ("induced_displacement_norm", "induced_length_norm",
            "induced_phase_norm", "operator_relative_remainder",
            "action_relative_remainder")
    return {k: truncation.get(k) for k in keys}


def _doublet_summary(record):
    doublet = record.get("isospin_doublet")
    if not isinstance(doublet, dict):
        return doublet
    return {name: {k: v for k, v in read.items()
                   if k in ("status", "found", "candidates", "summary")}
            if isinstance(read, dict) else read
            for name, read in doublet.items()}


# ------------------------------------------------------------------- a tick


def interaction_stage(base, partition, config):
    """Steps 3-4 for a given partition of the level's coordinates: the
    image-supported fibers for U and U^{-1}, the transports, the interaction
    graph, the grown 3-simplices and the grown-cell rule on each, and the
    locality certificate: the largest pairing and transport between two
    response vertices whose image supports share no top simplex."""
    fibers = fibers_for_partition(base, partition, config["band_rank"],
                                  config["contour_nodes"])
    frames, images = fibers["frames"], fibers["images"]
    transports = transport_matrix(base["pencil"], images, fibers["lefts"])
    pairs = interaction_graph(partition, base["edges"], base["tops"])
    grown = grown_cells(len(frames), pairs)
    pairing = inherited_pairing(frames, fibers["duals"], transports,
                                base["covariant"], images=images)
    reads, z, links, spread, groupoid = grow(grown, pairing, transports)
    apart = [(v, w) for v, w in itertools.permutations(range(len(frames)), 2)
             if (min(v, w), max(v, w)) not in pairs]
    locality = {
        "pairs_without_a_shared_top_simplex": len(apart) // 2,
        "largest_pairing_between_them": max(
            (float(abs(pairing[v, w])) for v, w in apart), default=0.0),
        "largest_transport_between_them": max(
            (float(np.linalg.norm(transports[(v, w)])) for v, w in apart),
            default=0.0),
    }
    return {"frames": frames, "images": images, "duals": fibers["duals"],
            "dual_images": fibers["dual_images"], "lefts": fibers["lefts"],
            "fibers": fibers["reads"], "transports": transports,
            "pairs": pairs, "pairing": pairing, "reads": reads, "z": z,
            "links": links, "spread": spread, "groupoid": groupoid,
            "locality": locality}


def persistent_components(level, edges, required):
    """The components of the level's partition that persist across at least
    ``required`` adjacent declared resolutions (WP §5: a candidate is
    accepted only if it stays stable across the stated range of scales), and
    a record of every other component, named by its index and its edges."""
    accepted, rejected = [], []
    for index, (part, persistence) in enumerate(
            zip(level.partition, level.component_persistence)):
        if persistence >= required:
            accepted.append(list(part))
        else:
            rejected.append({
                "component": index,
                "edges": ["%d-%d" % edges[c] for c in part],
                "persistence": float(persistence),
                "required": int(required)})
    return accepted, rejected


def tick(index, cells, z, links, config):
    """One tick on the level whose base is ``cells`` with fields ``z`` and
    ``links``. Returns the tick's record and the next level's base (or None
    when no 3-simplex grows). Tick 0 is level 0, the declared host, whose
    bounding cut is held; every later level is grown, and nothing on it is
    held."""
    started = time.time()
    spacetime, count = build_level(cells, z, links)
    declared = index == 0
    bulk_before = monopole_numbers(cells, links)
    cut = bounding_cut(cells) if declared else []
    cut_before = cut_monopole_number(cut, links) if declared else None
    held = {"faces": [list(f) for f in cut],
            "monopole_number_before": cut_before}
    try:
        relaxation = relax_level(
            spacetime, config,
            cut_sectors(cut, cut_before, count) if declared else [],
            count=count)
    except ValueError as error:
        # a declared refusal of the library (a face holonomy outside the
        # domain of the holonomy term): the level has no stationary point to
        # read, so the recursion stops here and says why
        triangles = _sorted_simplices(cells, 3)
        holonomies = [links[(a, b)] * links[(b, c)] / links[(a, c)]
                      for a, b, c in triangles]
        record = {
            "tick": index,
            "level": {
                "vertices": 1 + max(max(c) for c in cells),
                "cells": [sorted(c) for c in cells],
                "squared_lengths": {"%d-%d" % e: v for e, v in z.items()},
                "links": {"%d-%d" % e: v for e, v in links.items()},
                "face_holonomies": holonomies,
                "held_cut": held,
                "bulk_monopole_numbers_before": bulk_before,
            },
            "relaxation": {"failed": str(error)},
            "summary": {"response_vertices": 0, "interactions": 0,
                        "grown_cells": 0, "failed_cells": 0,
                        "rejected_cells": 0, "row_sum_defects": []},
            "reads": [],
            "stopped": "the level's relaxation was refused: %s" % error,
            "seconds": time.time() - started,
        }
        return record, None
    fields = sheet_fields(spacetime, count)
    base_z, base_links = fields[0]
    if declared:
        held["monopole_number_after"] = cut_monopole_number(cut, base_links)
        held["sector_monopole_numbers_after"] = \
            relaxation["sector_monopole_numbers"]
    # the sheets are compared by their gauge-invariant data: squared lengths
    # and face holonomies (links alone may differ by a gauge transformation,
    # along which the action is flat)
    triangles = _sorted_simplices(cells, 3)

    def holonomy(links_of_sheet, triangle):
        a, b, c = triangle
        return (links_of_sheet[(a, b)] * links_of_sheet[(b, c)]
                / links_of_sheet[(a, c)])

    isomorphism = max(
        max(max(abs(fields[t][0][e] - base_z[e]) / abs(base_z[e])
                for e in base_z),
            max(abs(holonomy(fields[t][1], f) - holonomy(base_links, f))
                / abs(holonomy(base_links, f)) for f in triangles))
        for t in range(1, SHEETS))
    base = base_operator(cells, base_z, base_links)
    level = recursion_turn(base["operator"], config)
    partition, rejected = persistent_components(
        level, base["edges"], config["persistence_required"])
    stage = interaction_stage(base, partition, config)
    frames, pairs = stage["frames"], stage["pairs"]
    transports = stage["transports"]
    reads, next_z, next_links = stage["reads"], stage["z"], stage["links"]
    spread, groupoid = stage["spread"], stage["groupoid"]
    kept = [r for r in reads if "failed" not in r and "rejected" not in r]
    gated = [r for r in reads if "rejected" in r]
    record = {
        "tick": index,
        "level": {
            "vertices": 1 + max(max(c) for c in cells),
            "edges": len(base["edges"]),
            "tetrahedra": len(base["tops"]),
            "cells": [sorted(c) for c in cells],
            "squared_lengths": {"%d-%d" % e: v for e, v in base_z.items()},
            "links": {"%d-%d" % e: v for e, v in base_links.items()},
            "held_cut": held,
            "bulk_monopole_numbers_before": bulk_before,
            "bulk_monopole_numbers_after": monopole_numbers(cells,
                                                            base_links),
            "sheet_isomorphism_residual": float(isomorphism),
            "rule_shift": level_rule_shift(cells, base_z, base_links),
        },
        "relaxation": relaxation,
        "partition": level_record(level),
        "response_components": partition,
        "rejected_components": rejected,
        "fibers": {
            "pairing_defect": [f["pairing_defect"] for f in stage["fibers"]],
            "restriction_determinant": [
                f["restriction_determinant"] for f in stage["fibers"]],
            "dual_eigenvalue_mismatch": [
                f["dual_eigenvalue_mismatch"] for f in stage["fibers"]],
        },
        "locality": stage["locality"],
        "interactions": sorted(pairs),
        "transport_norms": {"%d-%d" % p: float(np.linalg.norm(
            transports[p])) for p in sorted(pairs)},
        "grown_cells": reads,
        "grown_edges": {
            "%d-%d" % e: {"squared_length": next_z[e],
                          "cell_spread": spread[e],
                          "connection": next_links[e],
                          "groupoid_defect": groupoid[e]}
            for e in sorted(next_z)},
    }
    reads_config = dict(config)
    reads_config["hold_cell_sectors"] = declared
    record["reads"] = cell_reads(cells, base_z, base_links, reads_config)
    record["summary"] = {
        "held_cut_monopole_numbers": (
            [cut_before, held["monopole_number_after"]] if declared
            else None),
        "bulk_monopole_numbers_before": bulk_before,
        "bulk_monopole_numbers_after": record["level"][
            "bulk_monopole_numbers_after"],
        "regge_structurally_zero": relaxation["regge_structurally_zero"],
        "rule_shift_on_level": max(
            (r["relative_shift"] for r in record["level"]["rule_shift"]),
            default=None),
        "rejected_components": len(rejected),
        "response_vertices": len(frames),
        "interactions": len(pairs),
        "grown_cells": len(kept),
        "failed_cells": sum(1 for r in reads if "failed" in r),
        "rejected_cells": len(gated),
        "row_sum_defects": [r["row_sum_defect"] for r in kept],
        "quark_verdicts": [[_verdict_summary(c) for c in cell["contents"]]
                           for cell in record["reads"]],
        "isospin_doublet": [[_doublet_summary(c) for c in cell["contents"]]
                            for cell in record["reads"]],
        "lowest_poles": [[{"content": c["content"],
                           "poles": _lowest_poles(c),
                           "quartic_truncation": _truncation_summary(c)}
                          for c in cell["contents"]]
                         for cell in record["reads"]],
    }
    record["seconds"] = time.time() - started
    if not kept:
        record["stopped"] = ("no grown 3-simplex: the flag complex of the "
                             "interaction graph has no four pairwise "
                             "interacting response vertices whose cell "
                             "could be glued")
        return record, None
    used = sorted({v for r in kept for v in r["vertices"]})
    relabel = {v: i for i, v in enumerate(used)}
    record["lineage"] = {str(v): relabel.get(v) for v in range(len(frames))}
    next_cells = [[relabel[v] for v in r["vertices"]] for r in kept]
    kept_edges = {tuple(sorted(e)) for r in kept
                  for e in itertools.combinations(r["vertices"], 2)}
    next_edges = {tuple(sorted((relabel[a], relabel[b]))): (a, b)
                  for (a, b) in next_z if (a, b) in kept_edges}
    z_out = {e: next_z[src] for e, src in next_edges.items()}
    links_out = {e: next_links[src] for e, src in next_edges.items()}
    return record, (next_cells, z_out, links_out)



# ------------------------------------------------------------------ the drive


def default_config(ticks=DECLARED_TICKS, tetrahedra=DECLARED_TETRAHEDRA,
                   kappa=DECLARED_KAPPA, beta=DECLARED_BETA,
                   resolutions=DECLARED_RESOLUTIONS,
                   band_rank=DECLARED_BAND_RANK,
                   edge_squared=DECLARED_EDGE_SQUARED,
                   holonomy=bp.DECLARED_HOLONOMY,
                   elimination=bp.DECLARED_ELIMINATION,
                   selected_contents=None, max_cells=None,
                   persistence_required=None):
    """The declared configuration, recorded with every run. ``max_cells``
    limits how many tetrahedra per tick are read as hosts, for quick checks;
    it changes no number of the cells it keeps. ``persistence_required`` is
    how many adjacent declared resolutions a component must persist across
    to become a response vertex; by default every declared resolution, the
    stated range of scales of WP §5."""
    config = bp.default_config(kappas=[kappa], betas=[beta],
                               edge_squared=edge_squared,
                               holonomy=holonomy, elimination=elimination,
                               selected_contents=selected_contents)
    config.update({
        "mode": "controlled synthesis",
        "ticks": ticks,
        "tetrahedra": tetrahedra,
        "monopole": DECLARED_MONOPOLE,
        "sheets": SHEETS,
        "kappa": kappa,
        "beta": beta,
        "resolutions": list(resolutions),
        "band_rank": band_rank,
        "contour_nodes": DECLARED_CONTOUR_NODES,
        "emergence": "strict",
        "fibers": ("Riesz bands of the image pencil (A~_1^U, M_1^U) on each "
                   "component's edges: geometric images supported on the "
                   "component"),
        "persistence_required": (len(resolutions)
                                 if persistence_required is None
                                 else int(persistence_required)),
        "pairing": ("det((Y_v^vee)^T G_1^U Y_w) / det M_vw off the diagonal, "
                    "dual images normalized to det((Z_v^vee)^T Z_v) = %g, "
                    "read as |T| Gamma" % IMAGE_PAIRING_UNIT),
        "connection": ("U_vw = det M_vw, M_vw = Y~_v^T A~_1^U Z_w; it carries "
                       "the operator's units, 1/length^2"),
        "regge_branch": ("continued from the real projection of each level's "
                         "starting geometry"),
        "monopole_sectors": ("held on the bounding cut of the declared level-0 "
                             "host, per sheet; read and never held in the "
                             "bulk and on every grown level"),
        "gluing": ("grown cells in ascending lexicographic order, each added "
                   "only if the complex stays a manifold with boundary"),
        "edge_value": "mean over the grown cells containing the edge",
        "max_cells": max_cells,
    })
    return config


def points_path(json_path):
    """The append-only JSON-lines file beside ``json_path``."""
    return bp.points_path(json_path)


def drive(config, progress=False, on_frame=None, stop_requested=None,
          points_file=None):
    """Every tick. `on_frame(frames, index)` is called after each tick with
    the completed ticks; with `points_file` the configuration and the host are
    its first line and every tick is appended the moment it completes."""
    cells, z, links, connection = level_zero(config)
    host = {
        "cells": cells,
        "face_angles_over_pi": [float(a / math.pi)
                                for a in connection["face_angles"]],
        "dirac_string": [int(n) for n in connection["dirac_string"]],
        "connection_residual": connection["residual"],
        "monopole_numbers": monopole_numbers(cells, links),
    }
    if points_file is not None:
        with open(points_file, "w"):
            pass
        _append_line(points_file, {"config": config, "host": host})
    frames = []
    state = (cells, z, links)
    stopped = False
    for index in range(config["ticks"]):
        if stop_requested is not None and stop_requested():
            stopped = True
            break
        record, state = tick(index, *state, config)
        frames.append(record)
        if points_file is not None:
            _append_line(points_file, record)
        if progress:
            s = record["summary"]
            sys.stdout.write(
                "tick %d: %d response vertices, %d interactions, %d grown "
                "cells, row-sum defects %s (%.0f s)\n"
                % (index, s["response_vertices"], s["interactions"],
                   s["grown_cells"],
                   ["%.3g" % d for d in s["row_sum_defects"]],
                   record["seconds"]))
            for line in _notices(record):
                sys.stdout.write("  " + line + "\n")
            sys.stdout.flush()
        if on_frame is not None:
            on_frame(frames, len(frames) - 1)
        if state is None:
            break
    return {"config": _jsonable(config), "host": host, "ticks": frames,
            "stopped": stopped}


def _notices(record):
    """What a tick left out or refused, by name: a Regge term that is zero by
    structure, components rejected for persistence, and grown cells rejected
    by the manifold gate."""
    lines = []
    relaxation = record.get("relaxation", {})
    if relaxation.get("regge_structurally_zero"):
        lines.append(
            "the Regge term is structurally zero on this level: it has %d "
            "hinges under the %s hinge rule, so the lengths relaxed without "
            "it" % (relaxation["regge_hinge_count"],
                    relaxation["regge_hinges"]))
    for rejected in record.get("rejected_components", []):
        lines.append(
            "component %d (edges %s) rejected: it persists over %g of the "
            "%d required resolutions"
            % (rejected["component"], ", ".join(rejected["edges"]),
               rejected["persistence"], rejected["required"]))
    for cell in record.get("grown_cells", []):
        if "rejected" in cell:
            lines.append("grown cell %s rejected by the manifold gate: %s"
                         % (cell["vertices"], cell["rejected"]))
    return lines


def _jsonable(value):
    """JSON-ready form of a record: arrays become nested lists, complex
    numbers ``{"re", "im"}`` as in `baryon_poles`."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return bp._jsonable(value)


def _append_line(path, record):
    bp._append_line(path, _jsonable(record))


# ---------------------------------------------------------------- live mode

SURFACE = bp.SURFACE
INK = bp.INK
INK_MUTED = bp.INK_MUTED


def draw_frame(figure, frames, index):
    """One frame: per tick, the counts of the recursion and the row-sum
    defects of the grown cells; for the latest tick, the lowest quasi-free
    spin-1/2 and spin-3/2 poles of every read cell and content."""
    figure.clear()
    figure.patch.set_facecolor(SURFACE)
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    for axis in (left, right):
        axis.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(colors=INK_MUTED)
    done = frames[:index + 1]
    x = np.arange(len(done))
    for key, colour, label in (("response_vertices", "#2a78d6",
                                "response vertices"),
                               ("grown_cells", "#eb6834", "grown cells")):
        left.plot(x, [f["summary"][key] for f in done], marker="o",
                  linewidth=2, color=colour, label=label)
    twin = left.twinx()
    for i, f in enumerate(done):
        defects = f["summary"]["row_sum_defects"]
        twin.plot([i] * len(defects), defects, "_", markersize=14,
                  color="#1baf7a")
    twin.set_ylabel("row-sum defect ||g 1|| / ||g||", color=INK_MUTED)
    left.set_xticks(x)
    left.set_xlabel("tick", color=INK)
    left.set_ylabel("count", color=INK)
    left.set_title("the recursion per tick", color=INK, fontsize=10)
    left.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")

    read = [f for f in done if f.get("reads")]
    latest = read[-1] if read else done[-1]
    labels, half, three = [], [], []
    for cell in latest.get("reads", []):
        for record in cell["contents"]:
            labels.append("%s\n%s" % ("".join(map(str, cell["cell"])),
                                      "".join(map(str, record["content"]))))
            lowest = bp.lowest_poles(record, "quasi_free")
            for store, key in ((half, str(bp.SPIN_HALF)),
                               (three, str(bp.SPIN_THREE_HALVES))):
                best = lowest[key]
                store.append(best[0].real if best is not None else np.nan)
    positions = np.arange(len(labels))
    right.plot(positions, half, "o", markersize=7, color="#1baf7a",
               label="spin 1/2")
    right.plot(positions, three, "s", markersize=7, color="#4a3aa7",
               markerfacecolor="none", markeredgewidth=2, label="spin 3/2")
    right.set_xticks(positions)
    right.set_xticklabels(labels, fontsize=6)
    right.set_xlabel("cell and content", color=INK)
    right.set_ylabel("Re s (quasi-free)", color=INK)
    right.set_title("baryon poles at tick %d" % latest["tick"], color=INK,
                    fontsize=10)
    right.legend(frameon=False, fontsize=8, labelcolor=INK)
    figure.suptitle("level recursion with the grown-cell rule (%d ticks "
                    "done)" % len(done), color=INK)
    figure.tight_layout()


def drive_live(config, progress=False, points_file=None):
    """The same `drive` on a worker thread, drawing each completed tick on the
    main thread; the window is shown once, never raised, and pumped with
    `canvas.start_event_loop`. Closing it switches the run to headless."""
    import queue
    import threading

    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend()
    name = backend.lower()
    is_webagg = "webagg" in name
    if name not in bp._interactive_backends() or is_webagg:
        webagg = (" WebAgg is also refused because matplotlib implements "
                  "pause() there as a blocking server loop, so the worker's "
                  "later frames and outputs are never consumed."
                  if is_webagg else "")
        raise RuntimeError(
            "--live needs an interactive matplotlib backend; this process "
            "has %r.%s Install the Qt backend with "
            "`pip install -e \".[live]\"`, or select another local GUI "
            "backend; otherwise drop --live and read the rendered --out. "
            "The drive is identical either way." % (backend, webagg))
    matplotlib.rcParams["figure.raise_window"] = False
    if not plt.isinteractive():
        plt.ion()
    figure = plt.figure(figsize=(13, 6))
    plt.show(block=False)
    ready = queue.Queue()
    published = {}
    outcome = {}
    stop = threading.Event()
    closed = threading.Event()

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    def on_close(event):
        if not closed.is_set():
            closed.set()
            sys.stdout.write(
                "the live window was closed; the run continues headless and "
                "still writes every output\n")
            sys.stdout.flush()

    figure.canvas.mpl_connect("close_event", on_close)

    def worker():
        try:
            outcome["result"] = drive(config, progress=progress,
                                      on_frame=publish,
                                      stop_requested=stop.is_set,
                                      points_file=points_file)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            ready.put(None)

    thread = threading.Thread(target=worker, name="level-recursion")
    thread.start()
    main_error = None
    try:
        while not closed.is_set():
            try:
                index = ready.get_nowait()
            except queue.Empty:
                figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
                continue
            if index is None or closed.is_set():
                break
            draw_frame(figure, published["frames"], index)
            figure.canvas.draw_idle()
            figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
    except BaseException as error:
        main_error = error
        stop.set()
    finally:
        thread.join()
        if not closed.is_set():
            plt.close(figure)
    if main_error is not None:
        raise main_error
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def render(result, path):
    """The final frame as a PNG, on a file backend."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(13, 6))
    if result["ticks"]:
        draw_frame(figure, result["ticks"], len(result["ticks"]) - 1)
    figure.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def summary(result):
    """Every tick as text."""
    lines = ["mode: controlled synthesis; host monopole numbers %s"
             % result["host"]["monopole_numbers"]]
    for record in result["ticks"]:
        if "failed" in record["relaxation"]:
            lines.append("tick %d: the relaxation was refused: %s"
                         % (record["tick"], record["relaxation"]["failed"]))
            continue
        s = record["summary"]
        p = record["partition"]
        level = record["level"]
        lines.append(
            "tick %d: level with %d vertices, %d edges, %d tetrahedra per "
            "sheet; relaxation converged %s (residual %.3g); bulk monopole "
            "numbers %s before relaxation, %s after"
            % (record["tick"], level["vertices"], level["edges"],
               level["tetrahedra"], record["relaxation"]["converged"],
               record["relaxation"]["residual"],
               level.get("bulk_monopole_numbers_before"),
               level.get("bulk_monopole_numbers_after")))
        held = level.get("held_cut") or {}
        if held.get("faces"):
            lines.append(
                "  held bounding cut %s: monopole number %s before, %s after"
                % (held["faces"], held.get("monopole_number_before"),
                   held.get("monopole_number_after")))
        for line in _notices(record):
            lines.append("  " + line)
        lines.append(
            "  partition %s at resolution %g; fibers accepted %s; isolation "
            "gaps %s; determinant residual %.3g"
            % ([len(c) for c in p["partition"]], p["selected_resolution"],
               p["bands_accepted"],
               ["%.3g" % g for g in p["isolation_gaps"]],
               p["determinant_residual"]))
        lines.append(
            "  %d response vertices, %d interactions, %d grown cells"
            % (s["response_vertices"], s["interactions"], s["grown_cells"]))
        for cell in record["grown_cells"]:
            if "failed" in cell:
                lines.append("    cell %s failed: %s" % (cell["vertices"],
                                                         cell["failed"]))
                continue
            if "rejected" in cell:
                continue
            lines.append(
                "    cell %s: row-sum defect %.4g, asymmetry %.3g, z %s"
                % (cell["vertices"], cell["row_sum_defect"],
                   cell["asymmetry"],
                   [bp._complex_text(v) for v in cell["squared_lengths"]]))
        for cell in record["reads"]:
            for c in cell["contents"]:
                verdict = _verdict_summary(c)
                spin = (c.get("spin_decomposition") or {}).get(
                    "occupied_state")
                lines.append(
                    "    host cell %s content %s (quarks per band of h_1): "
                    "%s; quark certified %s; quarks per doublet of h-bar_1 "
                    "%s; lowest poles %s; quartic truncation %s"
                    % (cell["cell"], c["content"],
                       "failed: " + c["failed"] if "failed" in c else "read",
                       verdict["certified"] if verdict else None,
                       {k: bp._complex_text(v) for k, v in spin.items()}
                       if spin else None,
                       {k: {n: bp._complex_text(v) if v is not None else None
                            for n, v in d.items()}
                        for k, d in _lowest_poles(c).items()},
                       _truncation_summary(c)))
        if "stopped" in record:
            lines.append("  stopped: " + record["stopped"])
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m tessera.drivers.recursion",
        description="Several ticks of the level recursion with the grown-cell "
                    "rule, on three-sheeted unit-monopole tetrahedra sharing "
                    "faces.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the declared recursion")
    run.add_argument("--ticks", type=int, default=DECLARED_TICKS,
                     help="number of ticks (default %d)" % DECLARED_TICKS)
    run.add_argument("--tetrahedra", type=int, default=DECLARED_TETRAHEDRA,
                     help="tetrahedra in the level-0 fan (default %d)"
                          % DECLARED_TETRAHEDRA)
    run.add_argument("--kappa", type=float, default=DECLARED_KAPPA,
                     help="kappa = 8 pi G (default %g)" % DECLARED_KAPPA)
    run.add_argument("--beta", type=float, default=DECLARED_BETA,
                     help="beta of the holonomy term (default %g)"
                          % DECLARED_BETA)
    run.add_argument("--resolutions", type=float, nargs="+",
                     default=list(DECLARED_RESOLUTIONS),
                     help="modularity resolutions of the persistent partition "
                          "(default %s)" % (DECLARED_RESOLUTIONS,))
    run.add_argument("--persistence-required", type=int, default=None,
                     help="how many adjacent declared resolutions a component "
                          "must persist across to become a response vertex "
                          "(default: every declared resolution)")
    run.add_argument("--band-rank", type=int, default=DECLARED_BAND_RANK,
                     help="rank of every fiber (default %d)"
                          % DECLARED_BAND_RANK)
    run.add_argument("--edge-squared", type=float,
                     default=DECLARED_EDGE_SQUARED,
                     help="squared edge length of level 0 (default %g)"
                          % DECLARED_EDGE_SQUARED)
    run.add_argument("--holonomy", choices=tuple(bp.HOLONOMY_FORMS),
                     default=bp.DECLARED_HOLONOMY)
    run.add_argument("--eliminate", choices=bp.ELIMINATIONS,
                     default=bp.DECLARED_ELIMINATION)
    run.add_argument("--contents", type=int, nargs=3, action="append",
                     default=None, metavar=("N1", "N2", "N3"),
                     help="a content to read on every cell (repeatable; "
                          "default all ten)")
    run.add_argument("--max-cells", type=int, default=None,
                     help="read at most this many tetrahedra per tick as "
                          "hosts (a quick-check limit; default all)")
    run.add_argument("--json", default=None,
                     help="write every record here at the end; each tick is "
                          "also appended, as it completes, to "
                          "<json stem>.points.jsonl")
    run.add_argument("--out", default=None,
                     help="write the final frame as a PNG here")
    run.add_argument("--live", action="store_true",
                     help="draw each completed tick while the run proceeds; "
                          "the outputs are identical")
    run.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = default_config(
        ticks=args.ticks, tetrahedra=args.tetrahedra, kappa=args.kappa,
        beta=args.beta, resolutions=args.resolutions,
        band_rank=args.band_rank, edge_squared=args.edge_squared,
        holonomy=args.holonomy, elimination=args.eliminate,
        selected_contents=[tuple(c) for c in args.contents]
        if args.contents else None, max_cells=args.max_cells,
        persistence_required=args.persistence_required)
    points_file = points_path(args.json) if args.json else None
    result = (drive_live(config, progress=not args.quiet,
                         points_file=points_file) if args.live
              else drive(config, progress=not args.quiet,
                         points_file=points_file))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(_jsonable(result), handle, indent=1)
    if args.out:
        render(result, args.out)
    if not args.quiet:
        sys.stdout.write(summary(result) + "\n")
    return result


if __name__ == "__main__":
    main()
