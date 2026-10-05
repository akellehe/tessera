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
The declared default is two tetrahedra, the fan whose declared fields are
stationary as built with the host's bounding-cut sector held: the residual
norm of its level is at rounding before the drive takes a step. The fans of
three and of four start at residual norms of order one, and their drives end
short of stationarity with the stop the engine names (no move and no scaled
step lowers the residual norm), a part of the points their line searches
score having no residual.
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
   primal Regge + the Villain holonomy term (the Villain weight summed to
   the declared order, ``--villain-order``, ten by default), with the
   cosmological term -(1/kappa) Lambda sum_T V_T when a cosmological
   constant is declared (``--cosmological-constant``, none by default), in
   strict emergence (no carried density in the equations; WP §7). The
   spectral-moment part of S_0 is the
   holomorphic spectral constraint of WP v17 §3.4, which belongs to
   controlled synthesis on an occupied fiber and so enters only the per-cell
   reads below (stated there as the eigenvalue of each occupied band by
   default, ``--fiber-pinning``). The three sheets are relaxed
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
   (`SurgicalCone.validate`); a cell that fails is read like any other,
   recorded with its read and the violation, and left unattached. An edge
   shared by several grown cells receives one value from
   each; the level carries their mean and reports their spread. The grown
   base then takes the combinatorial moves of the growth step
   (`pachner_stage`): a stage-1 search of `MultiCobordism` in unforced
   emergence under the declared objective (``--pachner-objective``,
   `PACHNER_OBJECTIVES`: the stationarity of the joint action over the
   level's sheets, or the engine's joint stationarity objective), the four
   Pachner kinds
   alone (no cone-out, cone-in or disposition move), run as the emergence
   driver runs its stage 1, over every candidate move of the base or over
   a drawn sample of them (``--pachner-candidate-moves``,
   ``--pachner-updates``, ``--pachner-depth``,
   ``--pachner-length``; zero updates run none); the next level is built
   from the base that comes
   back;
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
   doublet content of the T-averaged operator and labelled by it. Each
   content's mean field is solved by the `MultiCobordism` drive of the joint
   system (`cell_solve`), its bands chosen at the host and followed by
   continuation (``--band-selection``); the solve's accepted updates,
   committed moves, final force, stop reason and joint-Jacobian rank gap
   are reported with the content. A read whose
   precondition does not hold at its declared tolerance is made and flagged:
   a geometry that is not Kontsevich-Segal allowable is read with that flag
   and its margin, and a relaxed cell that does not meet the preconditions
   of its own spin read is read in the frame of the declared symmetric host
   with the flag that names the precondition (``baryon_poles``, "Flags").
   A solve whose squared lengths overflowed the double leaves no finite
   geometry, and its content is recorded as having no value, by name.

The next tick runs on K_{l+1}. The recursion stops at a level with no grown
3-simplex, and at a level one of whose steps has no value: the level as
built, its relaxation, its operator, the turn of the Section 15 box, or the
interaction stage. The tick's record says which and why (``stopped``), and
keeps every read the tick made: a tick whose turn has no value still reads
its cells. A step that has no value is one at which the library or the
driver raises one of `baryon_poles.NO_VALUE_ERRORS`; any other error is a
defect of the code, and the drive writes what the tick reached to the points
file before it raises it.

The report is per doublet content
---------------------------------
Every (host cell, content, doublet content) read is reported on its own, and
nothing is averaged over doublet contents or over contents
(``baryon_poles``, "The report is per doublet content"): as each tick
completes, and again in the final summary, stdout carries one line per
(content, doublet content) pair of every host cell with both spins and both
columns, every pole with its multiplicity and its certificates; then, per
host cell, the labelled "lowest over" minima naming the doublet content (and
the content) each came from, and the ratios naming the pairs they compare.
The tick's ``summary.poles`` keeps every pair's poles beside the labelled
minima, the cell's ``pole_table`` has one row per pole, and the live frame
draws every pole of every pair as its own mark.

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
from tessera.drivers import cell_solve

#: Declared inputs, recorded with every run.
DECLARED_TETRAHEDRA = 2
DECLARED_EDGE_SQUARED = 8.0
DECLARED_MONOPOLE = 1
DECLARED_TICKS = 3
DECLARED_KAPPA = 1.0
DECLARED_BETA = 1.0
DECLARED_RESOLUTIONS = (1.0, 1.5, 2.0, 2.5, 3.0)
DECLARED_BAND_RANK = 1
#: The combinatorial moves of the growth step (`pachner_stage`): how many
#: stage-1 updates of `MultiCobordism` run on each grown level's base and how
#: many moves deep the search goes when no single move lowers the objective.
#: They are the emergence driver's declared stage-1 unit
#: (`emergence.DECLARED_STAGE1_ITERS`, `DECLARED_COMBINATORIAL_DEPTH`). Every
#: update scores every candidate move of the base (the library's complete
#: walk, `enumerate_move_specifications`), so no candidate count and no seed
#: is declared. Zero updates run no move. The length is the emergence
#: driver's alternative schedule (`emergence.DECLARED_COMBINATORIAL_LENGTH`):
#: sequences of exactly that many moves searched first, backing off one move
#: at a time; zero keeps the deepening schedule of the depth.
DECLARED_PACHNER_UPDATES = 1
DECLARED_PACHNER_DEPTH = 1
DECLARED_PACHNER_LENGTH = 0
#: The objective the growth step's search scores a base with
#: (``--pachner-objective``). ``joint-action``, the declared one, is the
#: stationarity of the joint action the level's relaxation descends
#: (`cell_solve.StationarityObjective` over the level's sheets): its Regge
#: term is read on the sheet continued from the level as grown and has one
#: value there. ``engine`` is `JointStationarityObjective`: the Regge action
#: in its dual form with every dihedral angle on its principal sheet, and
#: the Hodge spectral entropies. A grown level whose squared lengths are
#: real to rounding lies on the cuts of that sheet, where the sign of an
#: imaginary part at rounding selects the value, so under ``engine`` the
#: search's scores depend on that rounding.
PACHNER_OBJECTIVES = ("joint-action", "engine")
#: How many candidates each update of the growth step's search draws at
#: random (``--pachner-candidate-moves``): zero scores every candidate move
#: of the base, and with a depth or a length above one every composition of
#: that many moves; a positive number draws that many candidates, each a
#: sequence of the search's depth or length drawn move by move.
DECLARED_PACHNER_CANDIDATE_MOVES = 0
#: The hinges of the primal Regge sum of every level's and every cell's
#: action (``--regge-hinges``): the interior hinges, or all of them.
REGGE_HINGES = ("interior", "all")
DECLARED_REGGE_HINGES = "interior"
DECLARED_PACHNER_OBJECTIVE = "joint-action"
#: The degrees the joint stationarity objective is declared over on the base:
#: the register degree and the Hodge degrees, the emergence driver's
#: (`emergence.DECLARED_REGISTER_DEGREES`, `DECLARED_HODGE_DEGREES`).
PACHNER_REGISTER_DEGREES = (1,)
PACHNER_HODGE_DEGREES = (0, 1, 2, 3)
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


def monopole_connection(cells, monopole=DECLARED_MONOPOLE,
                        rank_tolerance=bp.DECLARED_TOLERANCE):
    """The declared monopole connection of the base (see the module
    docstring): the principal face angles, the Dirac string, the edge phases
    on the ascending orientation, and the consistency residual of
    d phi = theta - 2 pi n. The edge phases are the minimum-norm solution of
    that system (`numpy.linalg.lstsq`), a singular value of the face
    coboundary below ``rank_tolerance`` times the largest counted as zero;
    the solver reads a threshold of one or more as the machine epsilon."""
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
    phases, *_ = np.linalg.lstsq(coboundary, target, rcond=rank_tolerance)
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
    connection = monopole_connection(
        cells, config["monopole"],
        bp.declared_tolerance(config, "rank_tolerance"))
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


def monopole_numbers(cells, links,
                     tolerance=bp.DECLARED_CERTIFICATE_TOLERANCE):
    """The monopole number of every base tetrahedron, read with
    `MonopoleSupport` on its ascending vertices at ``tolerance`` (the
    support's unit-modulus tolerance and the tolerance of the monopole
    read)."""
    fixture = obs.MonopoleSupport.tetrahedron(1)
    numbers = []
    for cell in cells:
        c = sorted(cell)
        values = [links[(c[i], c[j])] for i, j in fixture.edges]
        support = obs.MonopoleSupport(4, fixture.edges, fixture.faces,
                                      obs.MonopoleSupport.u1Part(values),
                                      tolerance)
        numbers.append(int(support.monopoleNumber(tolerance).monopole_number))
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


def cell_orientations(cells):
    """A coherent orientation of the cluster made of the tetrahedra
    ``cells``: for each, +1 when its ascending vertex order has the
    orientation of the cluster and -1 when it has the opposite one. The face
    opposite the i-th ascending vertex of a tetrahedron carries the induced
    sign (-1)^i, and two tetrahedra that share a face induce opposite
    orientations on it, so across a shared face opposite the i-th vertex of
    one and the j-th of the other the signs are related by
    s' = -s (-1)^(i + j). The first tetrahedron of every connected
    component, in the order given, is +1. A cluster with no coherent
    orientation (a face of three tetrahedra, or a one-sided cluster) has no
    outward faces, which is a ValueError that names the face."""
    ordered = [tuple(sorted(c)) for c in cells]
    owners = {}
    for t, cell in enumerate(ordered):
        for i in range(4):
            face = tuple(v for k, v in enumerate(cell) if k != i)
            owners.setdefault(face, []).append((t, i))
    for face, shared in owners.items():
        if len(shared) > 2:
            raise ValueError(
                "the face %s is shared by %d tetrahedra, so the cluster has "
                "no coherent orientation" % (list(face), len(shared)))
    signs = [0] * len(ordered)
    for start in range(len(ordered)):
        if signs[start]:
            continue
        signs[start] = 1
        frontier = [start]
        while frontier:
            t = frontier.pop()
            cell = ordered[t]
            for i in range(4):
                face = tuple(v for k, v in enumerate(cell) if k != i)
                for other, j in owners[face]:
                    if other == t:
                        continue
                    sign = -signs[t] * (-1) ** (i + j)
                    if signs[other] == 0:
                        signs[other] = sign
                        frontier.append(other)
                    elif signs[other] != sign:
                        raise ValueError(
                            "the tetrahedra on the face %s cannot be "
                            "oriented coherently, so the cluster has no "
                            "outward faces" % (list(face),))
    return signs


def bounding_cut(cells):
    """The bounding cut of the cluster made of ``cells``: every face of one of
    its tetrahedra that no other of its tetrahedra shares, oriented outward
    for the coherent orientation of the cluster (`cell_orientations`): the
    face of `outward_faces` on a tetrahedron whose ascending order has the
    cluster's orientation, and its reverse on one whose ascending order has
    the opposite one. A face two of the tetrahedra share is bulk and is not
    on the cut."""
    signs = cell_orientations(cells)
    counts = {}
    for cell in cells:
        for face in outward_faces(cell):
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
    return [face if sign > 0 else (face[0], face[2], face[1])
            for cell, sign in zip(cells, signs)
            for face in outward_faces(cell)
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


def sheet_base(spacetime, count):
    """Sheet 0 of a level built by `build_level` as a complex of its own:
    the base cells on vertices 0..count-1, every edge carrying the length and
    the phase the level's sheet 0 stores for it."""
    cells = [cell for cell in cell_solve.top_cells(spacetime)
             if max(cell) < count]
    stored = {}
    for a, b, length, phase in cell_solve.edge_fields(spacetime):
        if a < count and b < count:
            stored[(a, b)] = (length, phase)
            stored[(b, a)] = (length, -phase)
    base = T.Spacetime.fromVertexTuples(3, [list(c) for c in cells], 1.0, 0.0)
    for edge in base.getEdgeList().toVector():
        length, phase = stored[(int(edge.getSource().getId()),
                                int(edge.getTarget().getId()))]
        edge.setLength(length)
        edge.setPhase(phase)
    return base


def base_fields(spacetime):
    """A base complex as a level's base: its cells, squared lengths and
    links (dicts on ascending edges) with its vertices relabeled 0..n-1 in
    ascending order, and the relabeling."""
    raw = cell_solve.top_cells(spacetime)
    used = sorted({v for c in raw for v in c})
    relabel = {v: i for i, v in enumerate(used)}
    cells = sorted([relabel[v] for v in c] for c in raw)
    z, links = {}, {}
    for a, b, length, phase in cell_solve.edge_fields(spacetime):
        x, y = relabel[a], relabel[b]
        # the inverse of `build_level`'s writing of the fields
        link = cmath.exp(1j * phase)
        if x > y:
            link = 1.0 / link
        key = (min(x, y), max(x, y))
        z[key] = length ** 2
        links[key] = link
    return cells, z, links, relabel


def level_system(base, config, sectors, sheets):
    """The stationarity system of the joint action on the sheeted support of
    a level's base complex (`cell_solve.GeometricSystem`), the declared
    monopole sectors held on the sheets of ``base`` as it stands, with the
    configuration that carries them."""
    host = cell_solve.sheeted_support(base, sheets)
    held = dict(config)
    held["held_sectors"] = list(sectors or [])
    villain_order = bp.declared_villain_order(config)
    cosmological_constant = bp.declared_cosmological_constant(config)

    def declare(complex_):
        return bp.action_declaration(
            complex_, config["kappa"], config["beta"],
            config["regge_hinges"], villain_order=villain_order,
            cosmological_constant=cosmological_constant)

    def geometry_of(support):
        return bp.support_geometry(held, support, host)

    return cell_solve.GeometricSystem(declare, geometry_of, sheets), held


def relax_level(spacetime, config, sectors=None, count=None):
    """Step 1: the level driven to holomorphic stationarity of the joint
    action by `MultiCobordism` (`cell_solve.solve`), its Pachner moves
    included, strict emergence, with the level's lengths as built as l0, and
    with the declared monopole sectors held as boundary data. With
    ``count``, the base vertex count of a level built by `build_level`, the
    sheets are one shared base field: the drive runs on the base complex
    (`sheet_base`) and scores the system of all the sheets, so they stay
    identical exactly. Without it the drive runs on ``spacetime`` itself,
    every edge its own coordinate.

    The relaxed fields are written to ``spacetime`` when the base the drive
    ended on has the cells it started with. When committed moves left other
    cells, ``spacetime`` is left as it was; the record's ``moved_base`` is
    then the base the drive ended on (`base_fields`: cells, squared lengths
    and links on its vertices relabeled 0..n-1), with the relabeling keyed
    by the drive's name of each vertex (`cell_solve.VertexNames`): a vertex
    of the level as built is named by its id there, and a vertex the drive
    inserted by a name above every id of the level as built, so that no key
    names two vertices; None otherwise.

    The record's ``converged`` is `baryon_poles.solve_converged`: the
    residual norm at the point the drive ended on (``residual``) at or below
    the declared ``step_tolerance``. It is the norm the drive minimised: with
    a declared spectral-moment stiffness, the residual of the action with
    the stiffness (``residual_includes_stiffness``). At the start, where the
    stiffness is declared, its gradient vanishes, so ``initial_residual`` is
    the action's alone. ``accepted_updates`` is the number of
    relaxation updates the drive accepted; the engine's iterations, which
    ``--iteration-limit`` counts, are each one update of the Pachner moves
    and a relaxation of several such updates."""
    shared = count is not None
    sheets = SHEETS if shared else 1
    base = sheet_base(spacetime, count) if shared else spacetime
    system, held = level_system(base, config, sectors, sheets)
    start = system.point(base)
    start_moduli = start.relaxation.held_log_moduli()
    initial = float(np.linalg.norm(start.relaxation.residual()))
    regge_hinge_count = int(start.relaxation.action.regge_hinge_count())
    regge_structurally_zero = bool(
        start.relaxation.action.regge_structurally_zero())
    drive = cell_solve.solve(base, system, **bp.solve_arguments(held))
    final = drive["spacetime"]
    end = system.point(final)
    # the norm the drive minimised: with a declared stiffness, its gradient
    # is in the residual
    residual, with_stiffness = drive["objective"].residual_norm(final)
    end_moduli = end.relaxation.held_log_moduli()
    drift = (max((abs(a - b) for a, b in zip(end_moduli, start_moduli)),
                 default=0.0)
             if len(end_moduli) == len(start_moduli) else math.nan)
    action = end.relaxation.action
    reported = action.reported_value()
    cosmological = bp.declared_cosmological_constant(config)
    updates = drive["objective"].updates
    converged = bp.solve_converged(
        residual, bp.declared_tolerance(held, "step_tolerance"))
    moved_base = None
    if drive["changed"]:
        cells, z, links, relabel = base_fields(final)
        names = drive["vertex_names"]
        moved_base = (cells, z, links,
                      {names[v]: label for v, label in relabel.items()})
    elif shared:
        stored = {}
        for a, b, length, phase in cell_solve.edge_fields(final):
            stored[(a, b)] = (length, phase)
            stored[(b, a)] = (length, -phase)
        for edge in spacetime.getEdgeList().toVector():
            a = int(edge.getSource().getId())
            b = int(edge.getTarget().getId())
            sheet = a // count
            length, phase = stored[(a - sheet * count, b - sheet * count)]
            edge.setLength(length)
            edge.setPhase(phase)
    return {
        "method": "MultiCobordism drive of the joint action's stationarity",
        "converged": bool(converged),
        "stop_reason": "converged" if converged else drive["stop_reason"],
        "stop_detail": str(drive["stop_detail"]),
        "initial_residual": initial,
        "residual": residual,
        "residual_includes_stiffness": with_stiffness,
        "accepted_updates": int(drive["accepted_updates"]),
        "moves_committed": int(drive["moves_committed"]),
        "pachner_moves": bool(drive["moves"]),
        "combinatorial_depth": int(drive["combinatorial_depth"]),
        "combinatorial_length": int(drive["combinatorial_length"]),
        "candidate_moves": int(drive["candidate_moves"]),
        "complex_before": drive["complex_before"],
        "complex_after": drive["complex_after"],
        "changed": bool(drive["changed"]),
        "moved_base": moved_base,
        "residual_trace": list(drive["trace"]),
        # the margin of the base the drive ended on; the drive excludes no
        # geometry for it unless the admissibility gate is declared
        "kontsevich_segal_margin": float(
            cob.HodgeLaplacian.kontsevichSegalMargin(final)),
        "admissibility_gate": bool(held.get("admissibility_gate", False)),
        # the scored complexes without a residual, in all and by reason:
        # a candidate move or a trial of the line search on which the
        # declared system or a declared stiffness has no value
        "undefined_points": len(drive["objective"].undefined),
        "undefined_reasons": cell_solve.undefined_reasons(
            drive["objective"].undefined),
        "sector_monopole_numbers": list(
            end.relaxation.sector_monopole_numbers()),
        "held_modulus_drift": float(drift),
        "shared_sheet_geometry": shared,
        "rank_tolerance": float(bp.declared_tolerance(held,
                                                      "rank_tolerance")),
        "jacobian_ranks": [int(u["jacobian_rank"]) for u in updates],
        "rank_gaps": [float(u["rank_gap"]) for u in updates],
        # every term of the action at every point a step was proposed from
        # (--trace-terms); empty otherwise
        "term_trace": [bp.term_records(u["measured"]) for u in updates
                       if u.get("measured")],
        "regge_hinges": config["regge_hinges"],
        "regge_hinge_count": regge_hinge_count,
        "regge_structurally_zero": regge_structurally_zero,
        "regge_off_principal_angles": int(
            action.regge_off_principal_angles()),
        "action": complex(reported.value),
        "action_available": bool(reported.available),
        "seconds": float(drive["seconds"]),
        # with a declared cosmological constant, its term, the sum of the
        # volumes and the Regge term at the point the drive ended on: along
        # a dilation the action is stationary where the Regge term is minus
        # three times the cosmological term
        **({"cosmological_constant": cosmological,
            "cosmological_term": complex(action.cosmological_term()),
            "volume_sum": complex(action.volume_sum()),
            "regge_term": complex(action.regge_term())}
           if cosmological != 0.0 else {}),
    }


# ------------------------------------------------------- the box on the base


def proposition_three(instance):
    """The covariance certificate of a `CovariantChainHodge` instance
    (specification Proposition 3) as a record: whether every measured
    property holds at the certificate's tolerance, the tolerance, and each
    property that does not hold as the library states it, with its residual.
    The instance is built and read whether or not the certificate holds."""
    certificate = instance.certificate()
    return {"holds": bool(certificate.holds),
            "tolerance": float(certificate.tolerance),
            "failed": [str(entry)
                       for entry in getattr(certificate, "failed", [])]}


def base_operator(cells, z, links):
    """The Whitney complex of the base, its canonical edges and top simplices,
    the covariant chain-Hodge instance of sheet 0 and its h_1(z, U), with
    the covariance certificates of the instances of U and of U^{-1}
    (`proposition_three`) under ``certificates``."""
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
            "certificates": {
                "connection": proposition_three(covariant),
                "inverse_connection": proposition_three(dual)},
            "covariant": covariant, "operator": operator,
            "dual_operator": dual_operator,
            "pencil": np.asarray(pencil.A), "metric": np.asarray(pencil.B),
            "dual_pencil": np.asarray(dual_pencil.A),
            "dual_metric": np.asarray(dual_pencil.B)}


def recursion_turn(operator, config):
    """One turn of the Section 15 box on the base operator."""
    declaration = cob.LevelRecursionDeclaration()
    declaration.resolutions = list(config["resolutions"])
    declaration.tolerance = bp.declared_tolerance(config, "recursion_tolerance")
    declaration.rank_tolerance = bp.declared_tolerance(
        config, "quotient_rank_tolerance")
    bands = cob.RecursionBandDeclaration()
    bands.selection = cob.RecursionBandSelection.LowestModes
    bands.band_rank = config["band_rank"]
    declaration.bands = bands
    n = operator.shape[0]
    recursion = cob.LevelRecursion.overPencil(list(operator.reshape(-1)), [],
                                              n, declaration)
    recursion.advance()
    return recursion.level(0)


def riesz_band(block, rank, tolerance):
    """The band of a component block by the rule `LevelRecursion` declares as
    `LowestModes` in ascending real part: the ``rank`` eigenvalues first in
    ascending real part, then imaginary part, every key compared at
    ``tolerance`` relative to the block's Frobenius norm, are the band. The
    selection is recorded as the circle about their mean whose radius sits
    halfway to the nearest excluded eigenvalue, infinite when nothing is
    excluded. The projector is the exact spectral projector onto the band's
    invariant subspace, P = Phi PhiTilde^T from the block's complex Schur form
    (`LevelRecursion.read_band`): the right frame Phi is an orthonormal basis
    of the subspace, the leading Schur vectors after the selected eigenvalues
    are reordered to the front, and the left frame PhiTilde^T is its
    algebraic dual, PhiTilde^T Phi = I, from the Sylvester equation of the
    reordered form. For a diagonalizable block P = V_B (V^-1)_B over the
    selected eigenvalues. A band of the whole block has the identity as its
    projector and the canonical basis as both frames, exactly. The read
    carries the projector's certificates: its idempotency residual, the
    pairing defect of the frames, the residual of the invariant subspace and
    the isolation gap of the band, with ``accepted`` saying whether all of
    them hold at ``tolerance``. A selection that separates two eigenvalues
    equal at ``tolerance`` is read like any other and comes back with
    ``accepted`` false and the isolation gap that says why. A selection
    that separates two eigenvalues that are equal exactly has no projector
    (the Sylvester equation is singular), and the library raises a
    ValueError that names the eigenvalue."""
    block = np.asarray(block, dtype=complex)
    n = block.shape[0]
    bands = cob.RecursionBandDeclaration()
    bands.selection = cob.RecursionBandSelection.LowestModes
    bands.band_rank = int(rank)
    bands.order = cob.OccupationOrder.AscendingRealPart
    read = cob.LevelRecursion.read_band(
        [complex(v) for v in block.reshape(-1)], n, bands, 0,
        float(tolerance))
    right = np.asarray(read.frame, dtype=complex).reshape(n, read.rank)
    left = np.asarray(read.left_frame, dtype=complex).reshape(read.rank, n)
    return {"frame": right, "left": left, "projector": right @ left,
            "eigenvalues": [complex(v) for v in read.eigenvalues],
            "centre": complex(read.contour_centre),
            "radius": float(read.contour_radius),
            "encloses_everything": bool(read.encloses_everything),
            "isolation_gap": float(read.isolation_gap),
            "projector_idempotency": float(read.projector_idempotency),
            "pairing_defect": float(read.pairing_defect),
            "invariant_subspace_residual": float(
                read.invariant_subspace_residual),
            "accepted": bool(read.accepted)}


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


def fibers_for_partition(base, partition, rank, tolerance):
    """The fibers of every component, supported on their geometric images
    (spec Prop. 5.3).

    The band of a component is the Riesz band of the image pencil
    (A~_1^U, M_1^U) restricted to the component's edges, selected by the rule
    of `riesz_band` and read exactly at ``tolerance``. Its frame is the
    geometric image Z = G_1^U Y, which is
    zero off the component. The chain frame is Y = M_1^U Z, which is zero off
    the component's one-ring. The dual band is the same for U^{-1}. The dual
    image is normalized to det((Z^vee)^T Z) = 3 (`normalize_image_pairing`).
    The left frame is the geometric image of the dual band,
    Y~^T = B^{-1} (Z^vee)^T with B = (Y^vee)^T G_1^U Y = (Z^vee)^T M_1^U Z
    (spec Prop. 4), so Y~^T Y = I.

    Returns a dict of per-fiber lists: ``frames`` (Y), ``images`` (Z),
    ``lefts`` (Y~^T), ``duals`` (Y^vee), ``dual_images`` (Z^vee) and
    ``reads``, each read carrying the band's projector and its certificates
    (`riesz_band`), the mismatch of the dual band's eigenvalues, the
    restriction determinant and the pairing defect of the level frames."""
    pencil, metric = base["pencil"], base["metric"]
    dual_pencil, dual_metric = base["dual_pencil"], base["dual_metric"]
    n = pencil.shape[0]
    out = {key: [] for key in ("frames", "images", "lefts", "duals",
                               "dual_images", "reads")}
    for part in partition:
        index = list(part)
        block = np.ix_(index, index)
        band = riesz_band(np.linalg.solve(metric[block], pencil[block]),
                          rank, tolerance)
        dual = riesz_band(np.linalg.solve(dual_metric[block],
                                          dual_pencil[block]), rank,
                          tolerance)
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
            "isolation_gap": band["isolation_gap"],
            "encloses_everything": band["encloses_everything"],
            "projector_idempotency": band["projector_idempotency"],
            "invariant_subspace_residual": band["invariant_subspace_residual"],
            "accepted": band["accepted"],
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
        "encloses_everything": [bool(b.encloses_everything)
                                for b in level.bands],
        "projector_idempotency": [float(b.projector_idempotency)
                                  for b in level.bands],
        "pairing_defects": [float(b.pairing_defect) for b in level.bands],
        "invariant_subspace_residuals": [
            float(b.invariant_subspace_residual) for b in level.bands],
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


def level_rule_shift(cells, z, links, rank_tolerance=bp.DECLARED_TOLERANCE):
    """The grown-cell rule applied to a level's own tetrahedra as if they were
    grown cells: on each tetrahedron alone, the vertex fibers are the exact
    chains of the twisted coboundary, their dual-connection partners those of
    the inverse connection, and the transports the edge links. The relative
    shift of the rule's squared lengths from the tetrahedron's own is zero for
    a pure-gauge connection; with face holonomies it is the curvature-induced
    shift of the rule. ``rank_tolerance`` is the rule's own
    (`GrownCellRule.invertVertexPairing`)."""
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
        read = ch.GrownCellRule.invertVertexPairing(pairing, rank_tolerance)
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


def grow(cells, pairing, transports, rank_tolerance=bp.DECLARED_TOLERANCE):
    """The grown-cell rule on every grown 3-simplex: lengths from the
    inherited pairing, connection from det M. ``rank_tolerance`` is the
    fraction of the largest pivot of a cell's metric block at or below which
    a pivot counts as zero (`GrownCellRule.invertVertexPairing`). A cell
    whose rule has no value (a block singular at that tolerance, a transport
    between two of its vertices that carries no connection) is recorded with
    the library's reason under ``"failed"``. Every other cell is read. The
    cells are attached in ascending lexicographic order of their vertices,
    and a cell is attached only if the complex of the cells attached so far
    stays a manifold with boundary (WP v18 §15); a cell that would break it
    keeps its read and carries the violation under ``"rejected"``, and it
    gives the next level no edge.
    Returns the per-cell reads and the per-edge fields of the next level
    (response-vertex labels)."""
    reads, per_edge_z, glued, connections = [], {}, [], {}
    for cell in sorted(tuple(sorted(c)) for c in cells):
        index = list(cell)
        block = pairing[np.ix_(index, index)]
        entry = {"vertices": index, "pairing": block}
        try:
            read = ch.GrownCellRule.invertVertexPairing(block,
                                                        rank_tolerance)
            for i, j in itertools.combinations(range(4), 2):
                edge = (index[i], index[j])
                if edge not in connections:
                    forward = complex(ch.GrownCellRule.transportConnection(
                        transports[edge]))
                    backward = complex(ch.GrownCellRule.transportConnection(
                        transports[edge[::-1]]))
                    connections[edge] = (
                        forward, float(abs(forward * backward - 1.0)))
        except bp.NO_VALUE_ERRORS as error:
            entry["failed"] = str(error)
            reads.append(entry)
            continue
        entry.update({
            "row_sum_defect": float(read.rowSumDefect),
            "asymmetry": float(read.asymmetry),
            "scale": complex(read.scale),
            "volume": complex(read.volume),
            "squared_lengths": [complex(v) for v in read.squaredLengths],
            "frame_invariant_ratios": np.asarray(read.frameInvariantRatios),
        })
        try:
            violation = manifold_violation(glued + [index])
        except bp.NO_VALUE_ERRORS as error:
            violation = "the manifold test has no value: %s" % error
        if violation is not None:
            entry["rejected"] = violation
            reads.append(entry)
            continue
        glued.append(index)
        for m, (i, j) in enumerate(itertools.combinations(range(4), 2)):
            per_edge_z.setdefault((index[i], index[j]), []).append(
                complex(read.squaredLengths[m]))
        reads.append(entry)
    links = {edge: connections[edge][0] for edge in per_edge_z}
    groupoid = {edge: connections[edge][1] for edge in per_edge_z}
    z = {e: complex(np.mean(values)) for e, values in per_edge_z.items()}
    spread = {e: float(max(abs(x - z[e]) for x in values) / abs(z[e]))
              if abs(z[e]) > 0 else 0.0 for e, values in per_edge_z.items()}
    return reads, z, links, spread, groupoid


# ----------------------------------------------------------------- the reads


def cell_reads(cells, z, links, config):
    """Every base tetrahedron read as a three-sheeted host of its own: the
    quark verdicts, the isospin-doublet reading and the baryon poles of every
    declared content (`baryon_poles.evaluate_content`), each content's spin
    read in the relaxed cell's own frame when the cell meets the
    preconditions of that read and in the frame of the declared symmetric
    host, flagged, when it does not (`baryon_poles.spin_frame`). Every cell's
    read names its contents with no value (``failed_contents``) and its
    flagged contents (``flagged_contents``). A cell whose read as a host has
    no value before any content is reached (`baryon_poles.NO_VALUE_ERRORS`)
    is recorded with the reason under ``failed`` and no content, and the
    other cells are read. A tetrahedron of the declared
    level-0 host is a declared host of its own, so its four faces are its
    bounding cut and are held. A tetrahedron of a grown level is not
    declared, so nothing on it is held (``hold_cell_sectors``)."""
    fixture = obs.MonopoleSupport.tetrahedron(1)
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
            regge_hinges=config["regge_hinges"],
            elimination=config["elimination"],
            selected_contents=[tuple(x) for x in config["contents"]])
        cell_config["host_cell"] = host_cell
        cell_config["isospin_doublet"] = True
        # the order of the Villain weight and a declared cosmological
        # constant (parts of the action), and the mean-field solver and the
        # tolerances the run declared (none of which changes an equation)
        for key in ("villain_order", "cosmological_constant",
                    "band_selection", "fiber_moments",
                    "fiber_pinning", "kappa_role", "trace_terms") + tuple(
                        key for key, _ in bp.TOLERANCES) + tuple(
                            key for key, _, _ in bp.LIMITS) + tuple(
                                key for key, _, _ in bp.SOLVE_OPTIONS):
            if key in config:
                cell_config[key] = config[key]
        try:
            number = monopole_numbers(
                [c], links,
                bp.declared_tolerance(config, "certificate_tolerance"))[0]
            cell_config["held_sectors"] = (
                held_sectors([[0, 1, 2, 3]], [number], 4)
                if config.get("hold_cell_sectors", True) else [])
            point = bp.scan_point(config["kappa"], config["beta"],
                                  cell_config)
        except bp.NO_VALUE_ERRORS as error:
            out.append({"cell": c, "host_cell": host_cell,
                        "failed": str(error), "failed_contents": [],
                        "flagged_contents": [], "contents": [],
                        "ratios": {}, "pole_table": {}})
            continue
        out.append({"cell": c, "host_cell": host_cell,
                    "failed_contents": point["failed_contents"],
                    "flagged_contents": point["flagged_contents"],
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


def _content_poles(record, config=None):
    """The poles of one content record for the tick's summary: for every
    doublet content, both spins and both columns with every pole, its
    multiplicity and the read's failed certificates; and, separately
    labelled, the lowest pole of each column and spin over the doublet
    contents with the doublet content it came from and every tied one
    (`baryon_poles.lowest_poles`)."""
    pairs = []
    for read in record.get("doublet_reads") or []:
        sectors = {}
        for j2 in bp.SPINS:
            entry = read["sectors"].get(j2)
            if not entry:
                continue
            sectors[j2] = {}
            for name in bp.COLUMNS:
                column = entry.get(name) or {}
                sectors[j2][name] = {
                    "poles": list(column.get("poles") or []),
                    "multiplicity": list(column.get("multiplicity") or []),
                    "failed_certificates": list(
                        column.get("failed_certificates") or []),
                }
        pairs.append({"doublet_content": list(read["doublet_content"]),
                      "sectors": sectors})
    return {"per_doublet_content": pairs,
            "lowest_over_doublet_contents": {
                name: bp.lowest_poles(
                    record, name, bp.declared_tolerance(config, "tie_tolerance"))
                for name in bp.COLUMNS}}


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
    response vertices whose image supports share no top simplex.

    A component whose fiber has no value (`baryon_poles.NO_VALUE_ERRORS`: a
    band whose selection separates two eigenvalues that are equal exactly,
    two geometric images that pair singularly, a singular restriction) is
    not a response vertex: it is named under ``failed_components`` with its
    edges and the reason, and the stage is made on the others, which
    ``partition`` lists in the order of the response vertices."""
    fibers = {key: [] for key in ("frames", "images", "lefts", "duals",
                                  "dual_images", "reads")}
    kept, failed = [], []
    for index, part in enumerate(partition):
        try:
            one = fibers_for_partition(
                base, [part], config["band_rank"],
                bp.declared_tolerance(config, "recursion_tolerance"))
        except bp.NO_VALUE_ERRORS as error:
            failed.append({"component": index,
                           "edges": ["%d-%d" % base["edges"][c]
                                     for c in part],
                           "failed": str(error)})
            continue
        kept.append(list(part))
        for key in fibers:
            fibers[key] += one[key]
    partition = kept
    frames, images = fibers["frames"], fibers["images"]
    transports = transport_matrix(base["pencil"], images, fibers["lefts"])
    pairs = interaction_graph(partition, base["edges"], base["tops"])
    grown = grown_cells(len(frames), pairs)
    pairing = inherited_pairing(frames, fibers["duals"], transports,
                                base["covariant"], images=images)
    reads, z, links, spread, groupoid = grow(
        grown, pairing, transports,
        bp.declared_tolerance(config, "grown_cell_rank_tolerance"))
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
            "locality": locality, "partition": partition,
            "failed_components": failed}


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


def checked_pachner_objective(name):
    """The growth step's objective by name, one of `PACHNER_OBJECTIVES`."""
    if name not in PACHNER_OBJECTIVES:
        raise ValueError("the growth step's objective is one of %s; got %r"
                         % (", ".join(PACHNER_OBJECTIVES), name))
    return name


def pachner_stage(cells, z, links, config):
    """The combinatorial moves of the growth step, on the next level's base:
    a stage-1 search of `MultiCobordism` in unforced emergence under the
    config's ``pachner_objective`` (`PACHNER_OBJECTIVES`): the engine's
    `JointStationarityObjective` (the Regge action and the Hodge spectral
    entropies stationary at one metric), or the stationarity of the joint
    action over the level's sheets, nothing held, which is the objective of
    the level's relaxation. The search is restricted to the four Pachner
    kinds: no cone-out, cone-in or disposition move, so the base stays the
    manifold it is, with the boundary it has. It runs as the emergence
    driver runs its stage 1, over every candidate move of the base rather
    than a drawn sample: ``pachner_updates`` updates, each scoring every
    move and committing the best that lowers the objective by more than the
    config's ``move_tolerance``, deepening to ``pachner_depth``-move
    sequences when no single move does, or, with a ``pachner_length`` above
    zero, searching sequences of exactly that many moves first and backing
    off one move at a time (the two schedules are alternatives,
    `cell_solve.checked_schedule`). No candidate is excluded for its
    Kontsevich-Segal margin unless the config declares the engine's
    admissibility gate (``admissibility_gate``, off by default), under
    which a proposed geometry is admissible when its margin is at least
    minus the config's ``admissibility_tolerance``; the margin of the base
    the search ends on is in the record either way. The walk is
    complete and reproducible, so there is no seed. The base is one sheet;
    the level is rebuilt from it
    with every sheet identical. Returns the base after the committed moves,
    its vertices relabeled 0..n-1 (``vertex_relabeling`` in the record), with
    the stage's record; zero updates return the base as it is."""
    updates = int(config.get("pachner_updates", 0))
    depth, length = cell_solve.checked_schedule(
        config.get("pachner_depth", DECLARED_PACHNER_DEPTH),
        config.get("pachner_length", DECLARED_PACHNER_LENGTH))
    name = checked_pachner_objective(
        config.get("pachner_objective", DECLARED_PACHNER_OBJECTIVE))
    drawn = int(config.get("pachner_candidate_moves",
                           DECLARED_PACHNER_CANDIDATE_MOVES))
    record = {
        "updates": updates, "depth": depth, "length": length,
        "candidate_moves": drawn,
        "candidates": ("every candidate move of the base, scored in full"
                       if drawn <= 0 else
                       "%d candidates drawn at random at every update, each "
                       "scored in full" % drawn),
        "moves": ("the four Pachner kinds; no cone-out, cone-in or "
                  "disposition move"),
        "objective_name": name,
        "admissibility_gate": bool(config.get("admissibility_gate", False)),
        # a declared cosmological constant is a term of the joint action and
        # so of the joint-action objective; the engine's objective has no
        # such term
        **({"cosmological_constant": bp.declared_cosmological_constant(config),
            "cosmological_term_in_objective": name != "engine"}
           if bp.declared_cosmological_constant(config) != 0.0 else {}),
        "objective": (
            "joint stationarity: the Regge action and the Hodge spectral "
            "entropies of degrees %s stationary at one metric"
            % (list(PACHNER_HODGE_DEGREES),) if name == "engine" else
            "the norm of the joint action's stationarity residual over the "
            "level's %d sheets, nothing held" % SHEETS),
        "before": {"vertices": 1 + max(max(c) for c in cells),
                   "edges": len(z), "cells": len(cells)},
    }
    if updates <= 0:
        record["after"] = dict(record["before"])
        record["changed"] = False
        return cells, z, links, record
    spacetime, _ = build_level(cells, z, links, sheets=1)
    MC = cob.MultiCobordism
    if name == "engine":
        node = MC(spacetime, [], [], list(PACHNER_REGISTER_DEGREES), 1.0, 0,
                  0, False)
        node.set_objective(cob.JointStationarityObjective())
        node.set_hodge_degrees(list(PACHNER_HODGE_DEGREES))
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.STRICT)
        node.should_propose_surgery = False
    else:
        system, _ = level_system(spacetime, config, [], SHEETS)
        # the Regge sheets of the search start at the level as grown
        system.begin(spacetime)
        objective = cell_solve.StationarityObjective(system)
        objective.begin()
        node = cell_solve.cell_node(spacetime, objective)
    node.move_tolerance = bp.declared_tolerance(config, "move_tolerance")
    node.admissibility_gate = bool(config.get("admissibility_gate", False))
    node.admissibility_tolerance = bp.declared_tolerance(
        config, "admissibility_tolerance")
    record["objective_before"] = float(node.objective())
    record["trace"] = [float(value) for value in node.run_stage1(
        max_steps=updates, n_candidate_moves=max(drawn, 0),
        grow_boundaries=False, max_lookahead=depth,
        combinatorial_breadth=length)]
    record["objective_after"] = float(node.objective())
    # the margin of the base the search ends on: a number of the record,
    # which excludes nothing unless the admissibility gate is declared
    record["kontsevich_segal_margin"] = float(
        cob.HodgeLaplacian.kontsevichSegalMargin(node.spacetime()))
    cells_out, z_out, links_out, relabel = base_fields(node.spacetime())
    record["vertex_relabeling"] = {str(v): relabel[v]
                                   for v in sorted(relabel)}
    record["after"] = {"vertices": len(relabel), "edges": len(z_out),
                       "cells": len(cells_out)}
    record["changed"] = cells_out != sorted(sorted(c) for c in cells)
    return cells_out, z_out, links_out, record


def _counts():
    """The counts of a tick's summary before any of them is read."""
    return {"response_vertices": 0, "interactions": 0, "grown_cells": 0,
            "failed_cells": 0, "rejected_cells": 0, "row_sum_defects": []}


def _made(target, key, read, without_value):
    """``target[key]`` is ``read()``; when the read has no value
    (`baryon_poles.NO_VALUE_ERRORS`) it is None, and ``without_value[key]``
    is the reason."""
    try:
        target[key] = read()
    except bp.NO_VALUE_ERRORS as error:
        target[key] = None
        without_value[key] = str(error)
    return target[key]


def tick(index, cells, z, links, config):
    """One tick on the level whose base is ``cells`` with fields ``z`` and
    ``links``. Returns the tick's record and the next level's base (or None
    when no 3-simplex grows). Tick 0 is level 0, the declared host, whose
    bounding cut is held; every later level is grown, and nothing on it is
    held.

    A step of the tick that has no value (`baryon_poles.NO_VALUE_ERRORS`) is
    recorded by name with the library's reason, every read that does not
    depend on it is made, and the tick returns no next level, its record
    saying where it stopped and why (``stopped``):

    * the level as built (a link that is zero, a cluster with no coherent
      orientation): ``stopped`` alone, with the level's fields as given;
    * the relaxation (``relaxation.failed``; for example a face holonomy
      outside the domain of the holonomy term): there is no relaxed level,
      so nothing further is read;
    * the relaxed level's operator, or the turn of the Section 15 box
      (``partition.failed``; for example a band whose selection separates
      two eigenvalues that are equal exactly, or a level at the dense
      crossover): the level, its relaxation and the reads of its cells are
      kept;
    * the interaction stage (``interaction.failed``): the partition and the
      reads of the cells are kept.

    A read of the relaxed level that has no value (the monopole numbers, the
    sheet isomorphism residual, the rule shift) is None, with the reason
    under ``level.without_value``; the level's operator is read whether or
    not its covariance certificate holds, and the certificates are recorded
    under ``level.operator_certificates`` (`proposition_three`); a component
    whose fiber has no value is
    named under ``failed_components`` (`interaction_stage`); a cell whose
    read has no value is recorded in ``reads`` with the reason
    (`cell_reads`); and a growth step whose search has no value leaves the
    grown base as it is, with the reason under ``pachner.failed``.

    Any other error is a defect of the code: it is raised, carrying what the
    tick reached as its ``tick_record`` attribute, which `drive` writes to
    the points file."""
    started = time.time()
    record = {"tick": index, "summary": _counts(), "reads": []}
    try:
        following = _tick(record, index, cells, z, links, config)
    except Exception as error:
        record.setdefault("stopped", "the tick ended on an error: %s: %s"
                          % (type(error).__name__, error))
        record["seconds"] = time.time() - started
        error.tick_record = record
        raise
    record["seconds"] = time.time() - started
    return record, following


def _tick(record, index, cells, z, links, config):
    """The steps of `tick`, written into ``record`` as they are made; returns
    the next level's base, or None."""
    declared = index == 0
    certificate_tolerance = bp.declared_tolerance(config,
                                                  "certificate_tolerance")
    cells_before = [sorted(c) for c in cells]
    held = {"faces": [], "monopole_number_before": None}
    level = record["level"] = {
        "vertices": 1 + max(max(c) for c in cells),
        "edges": len(z),
        "tetrahedra": len(cells),
        "cells": cells_before,
        "cells_before": cells_before,
        "squared_lengths": {"%d-%d" % e: v for e, v in z.items()},
        "links": {"%d-%d" % e: v for e, v in links.items()},
        "held_cut": held,
    }
    try:
        spacetime, count = build_level(cells, z, links)
        bulk_before = monopole_numbers(cells, links, certificate_tolerance)
        cut = bounding_cut(cells) if declared else []
        cut_before = cut_monopole_number(cut, links) if declared else None
    except bp.NO_VALUE_ERRORS as error:
        record["stopped"] = "the level has no value as built: %s" % error
        return None
    held["faces"] = [list(f) for f in cut]
    held["monopole_number_before"] = cut_before
    # the monopole numbers of the cells the level has before its relaxation
    # (`cells_before`); a committed move leaves other cells, which
    # `bulk_monopole_numbers_after` is read on (`cells`)
    level["bulk_monopole_numbers_before"] = bulk_before
    try:
        relaxation = relax_level(
            spacetime, config,
            cut_sectors(cut, cut_before, count) if declared else [],
            count=count)
    except bp.NO_VALUE_ERRORS as error:
        # the library names no stationary point to read (for example a face
        # holonomy outside the domain of the holonomy term), so the
        # recursion stops here and says why
        triangles = _sorted_simplices(cells, 3)
        level["face_holonomies"] = [
            links[(a, b)] * links[(b, c)] / links[(a, c)]
            for a, b, c in triangles]
        record["relaxation"] = {"failed": str(error)}
        record["stopped"] = "the level's relaxation was refused: %s" % error
        return None
    record["relaxation"] = relaxation
    moved_base = relaxation.pop("moved_base", None)
    try:
        if moved_base is not None:
            # committed Pachner moves left other cells: the level is the one
            # the drive ended on, every sheet a copy of it, on its vertices
            # relabeled 0..n-1. ``vertex_relabeling`` takes each vertex from
            # the drive's name to its id on the moved level: a name that is
            # an id of ``cells_before`` names that vertex, and a vertex the
            # drive inserted has a name above every id of ``cells_before``
            # (`cell_solve.VertexNames`), so no name stands for two vertices;
            # ``cells_after`` names the moved level's cells so. The held cut
            # is carried by its vertices: a move of the four Pachner kinds
            # changes no boundary face and removes no boundary vertex, so
            # every face of the cut is a boundary face of the moved base,
            # which the record checks (``faces_after_on_boundary``)
            cells, moved_z, moved_links, relabel = moved_base
            spacetime, count = build_level(cells, moved_z, moved_links)
            relaxation["vertex_relabeling"] = {str(v): relabel[v]
                                               for v in sorted(relabel)}
            name_of = {label: name for name, label in relabel.items()}
            level["cells_after"] = [sorted(name_of[v] for v in cell)
                                    for cell in cells]
            kept_faces = [tuple(relabel[v] for v in face) for face in cut
                          if all(v in relabel for v in face)]
            owners = {}
            for cell in cells:
                for face in itertools.combinations(sorted(cell), 3):
                    owners[face] = owners.get(face, 0) + 1
            held["faces_after"] = [list(f) for f in kept_faces]
            held["faces_after_on_boundary"] = (
                len(kept_faces) == len(cut) and all(
                    owners.get(tuple(sorted(f))) == 1 for f in kept_faces))
            cut = kept_faces
        fields = sheet_fields(spacetime, count)
        base_z, base_links = fields[0]
    except bp.NO_VALUE_ERRORS as error:
        record["stopped"] = ("the relaxed level's fields have no value: %s"
                             % error)
        return None
    without_value = {}
    level.update({
        "vertices": 1 + max(max(c) for c in cells),
        "edges": len(_sorted_simplices(cells, 2)),
        "tetrahedra": len(cells),
        "cells": [sorted(c) for c in cells],
        "squared_lengths": {"%d-%d" % e: v for e, v in base_z.items()},
        "links": {"%d-%d" % e: v for e, v in base_links.items()},
    })
    if declared:
        _made(held, "monopole_number_after",
              lambda: cut_monopole_number(cut, base_links), without_value)
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

    def isomorphism():
        return float(max(
            max(max(abs(fields[t][0][e] - base_z[e]) / abs(base_z[e])
                    for e in base_z),
                max(abs(holonomy(fields[t][1], f) - holonomy(base_links, f))
                    / abs(holonomy(base_links, f)) for f in triangles))
            for t in range(1, SHEETS)))

    _made(level, "bulk_monopole_numbers_after",
          lambda: monopole_numbers(cells, base_links, certificate_tolerance),
          without_value)
    _made(level, "sheet_isomorphism_residual", isomorphism, without_value)
    _made(level, "rule_shift", lambda: level_rule_shift(
        cells, base_z, base_links,
        bp.declared_tolerance(config, "grown_cell_rank_tolerance")),
        without_value)
    if without_value:
        level["without_value"] = without_value

    # the Section 15 box on the base and the interaction stage
    stage = None
    step = "the level's operator has no value"
    try:
        base = base_operator(cells, base_z, base_links)
        level["operator_certificates"] = base["certificates"]
        step = "the recursion's turn has no value"
        turn = recursion_turn(base["operator"], config)
        turn_record = level_record(turn)
        partition, rejected = persistent_components(
            turn, base["edges"], config["persistence_required"])
    except bp.NO_VALUE_ERRORS as error:
        # there is no partition, no fiber and no grown cell to carry on, so
        # the recursion stops here and says why, with the level, its
        # relaxation and the reads of its cells kept
        record["partition"] = {"failed": str(error)}
        record["stopped"] = "%s: %s" % (step, error)
    else:
        record["partition"] = turn_record
        record["response_components"] = partition
        record["rejected_components"] = rejected
        try:
            stage = interaction_stage(base, partition, config)
            fibers = stage["fibers"]
            interaction = {
                "response_components": stage["partition"],
                "failed_components": stage["failed_components"],
                "fibers": {key: [f[key] for f in fibers] for key in (
                    "pairing_defect", "isolation_gap",
                    "projector_idempotency", "invariant_subspace_residual",
                    "accepted", "restriction_determinant",
                    "dual_eigenvalue_mismatch")},
                "locality": stage["locality"],
                "interactions": sorted(stage["pairs"]),
                "transport_norms": {
                    "%d-%d" % p: float(np.linalg.norm(
                        stage["transports"][p]))
                    for p in sorted(stage["pairs"])},
                "grown_cells": stage["reads"],
                "grown_edges": {
                    "%d-%d" % e: {"squared_length": stage["z"][e],
                                  "cell_spread": stage["spread"][e],
                                  "connection": stage["links"][e],
                                  "groupoid_defect": stage["groupoid"][e]}
                    for e in sorted(stage["z"])},
            }
        except bp.NO_VALUE_ERRORS as error:
            stage = None
            record["interaction"] = {"failed": str(error)}
            record["stopped"] = ("the interaction stage has no value: %s"
                                 % error)
        else:
            record.update(interaction)

    # the reads of the level's cells need its cells and its fields alone
    reads_config = dict(config)
    reads_config["hold_cell_sectors"] = declared
    record["reads"] = cell_reads(cells, base_z, base_links, reads_config)

    grown = stage["reads"] if stage is not None else []
    kept = [r for r in grown if "failed" not in r and "rejected" not in r]
    summary = _counts()
    summary.update({
        "held_cut_monopole_numbers": (
            [cut_before, held["monopole_number_after"]] if declared
            else None),
        "bulk_monopole_numbers_before": bulk_before,
        "bulk_monopole_numbers_after": level["bulk_monopole_numbers_after"],
        "regge_structurally_zero": relaxation["regge_structurally_zero"],
        "rule_shift_on_level": max(
            (r["relative_shift"] for r in level["rule_shift"] or []),
            default=None),
        "operator_certificates_hold": (
            all(c["holds"] for c in level["operator_certificates"].values())
            if "operator_certificates" in level else None),
        "rejected_components": len(record.get("rejected_components") or []),
        "failed_components": len(record.get("failed_components") or []),
        "failed_read_cells": sum(1 for cell in record["reads"]
                                 if "failed" in cell),
        "quark_verdicts": [[_verdict_summary(c) for c in cell["contents"]]
                           for cell in record["reads"]],
        "isospin_doublet": [[_doublet_summary(c) for c in cell["contents"]]
                            for cell in record["reads"]],
        "poles": [[dict(_content_poles(c, config), content=c["content"],
                        quartic_truncation=_truncation_summary(c))
                   for c in cell["contents"]]
                  for cell in record["reads"]],
    })
    if stage is not None:
        summary.update({
            "response_vertices": len(stage["frames"]),
            "interactions": len(stage["pairs"]),
            "grown_cells": len(kept),
            "failed_cells": sum(1 for r in grown if "failed" in r),
            "rejected_cells": sum(1 for r in grown if "rejected" in r),
            "row_sum_defects": [r["row_sum_defect"] for r in kept],
        })
    record["summary"] = summary
    if "stopped" in record:
        return None
    if not kept:
        record["stopped"] = ("no grown 3-simplex: the flag complex of the "
                             "interaction graph has no four pairwise "
                             "interacting response vertices whose cell "
                             "could be glued")
        return None
    next_z, next_links = stage["z"], stage["links"]
    used = sorted({v for r in kept for v in r["vertices"]})
    relabel = {v: i for i, v in enumerate(used)}
    lineage = {v: relabel.get(v) for v in range(len(stage["frames"]))}
    next_cells = [[relabel[v] for v in r["vertices"]] for r in kept]
    kept_edges = {tuple(sorted(e)) for r in kept
                  for e in itertools.combinations(r["vertices"], 2)}
    next_edges = {tuple(sorted((relabel[a], relabel[b]))): (a, b)
                  for (a, b) in next_z if (a, b) in kept_edges}
    z_out = {e: next_z[src] for e, src in next_edges.items()}
    links_out = {e: next_links[src] for e, src in next_edges.items()}
    # the combinatorial moves on the grown base, before it is the next level
    before = {"vertices": len(used), "edges": len(z_out),
              "cells": len(next_cells)}
    try:
        moved_cells, moved_z, moved_links, pachner = pachner_stage(
            next_cells, z_out, links_out, config)
    except bp.NO_VALUE_ERRORS as error:
        # the search over the moves has no value on this base: the grown
        # base is the next level as it is, and the record says why
        pachner = {"failed": str(error),
                   "updates": int(config.get("pachner_updates", 0)),
                   "before": before, "after": dict(before), "changed": False}
    else:
        next_cells, z_out, links_out = moved_cells, moved_z, moved_links
    record["pachner"] = pachner
    # a response vertex's lineage names a vertex of the next level, so it
    # takes the labels the growth step's moves leave (None for a vertex a
    # move removed)
    labels = pachner.get("vertex_relabeling")
    record["lineage"] = {
        str(v): (label if labels is None or label is None
                 else labels.get(str(label)))
        for v, label in lineage.items()}
    record["summary"]["pachner"] = {
        "updates": pachner["updates"],
        "changed": pachner["changed"],
        "cells": [pachner["before"]["cells"], pachner["after"]["cells"]],
        "objective": [pachner.get("objective_before"),
                      pachner.get("objective_after")],
    }
    if "failed" in pachner:
        record["summary"]["pachner"]["failed"] = pachner["failed"]
    return next_cells, z_out, links_out



# ------------------------------------------------------------------ the drive


def checked_run(ticks, tetrahedra, kappa, beta, resolutions, band_rank,
                persistence_required, max_cells, pachner_updates,
                pachner_depth, pachner_length, pachner_candidate_moves=0):
    """The declarations of a run that have no meaning, each named: a count
    that is negative, a coupling that is zero or not finite (the action
    carries 1/kappa), an empty window of resolutions or a resolution that is
    not positive, a band without a mode, and a schedule of the growth step
    that `cell_solve.checked_schedule` does not accept."""
    def whole(name, value, least):
        if int(value) != value or int(value) < least:
            raise ValueError("%s is an integer of at least %d; got %r"
                             % (name, least, value))

    whole("the number of ticks", ticks, 0)
    whole("the number of tetrahedra of the declared host", tetrahedra, 1)
    whole("the band rank", band_rank, 1)
    whole("the number of updates of the growth step", pachner_updates, 0)
    whole("the number of candidates the growth step draws",
          pachner_candidate_moves, 0)
    if persistence_required is not None:
        whole("the number of resolutions a component persists across",
              persistence_required, 0)
    if max_cells is not None:
        whole("the number of cells read per tick", max_cells, 0)
    for name, value in (("kappa", kappa), ("beta", beta)):
        if not math.isfinite(value):
            raise ValueError("%s is a finite number; got %r" % (name, value))
    if kappa == 0:
        raise ValueError("kappa is not zero: the action carries 1/kappa")
    resolutions = list(resolutions)
    if not resolutions:
        raise ValueError("the window of resolutions has no resolution in it")
    for resolution in resolutions:
        if not (math.isfinite(resolution) and resolution > 0):
            raise ValueError("a resolution is a positive finite number; got "
                             "%r" % (resolution,))
    cell_solve.checked_schedule(pachner_depth, pachner_length)


def default_config(ticks=DECLARED_TICKS, tetrahedra=DECLARED_TETRAHEDRA,
                   kappa=DECLARED_KAPPA, beta=DECLARED_BETA,
                   resolutions=DECLARED_RESOLUTIONS,
                   band_rank=DECLARED_BAND_RANK,
                   edge_squared=DECLARED_EDGE_SQUARED,
                   elimination=bp.DECLARED_ELIMINATION,
                   selected_contents=None, max_cells=None,
                   persistence_required=None,
                   band_selection=bp.DECLARED_BAND_SELECTION,
                   fiber_moments=bp.DECLARED_FIBER_MOMENTS,
                   fiber_pinning=bp.DECLARED_FIBER_PINNING, tolerances=None,
                   trace_terms=False,
                   pachner_updates=DECLARED_PACHNER_UPDATES,
                   pachner_depth=DECLARED_PACHNER_DEPTH,
                   pachner_length=DECLARED_PACHNER_LENGTH,
                   pachner_objective=DECLARED_PACHNER_OBJECTIVE,
                   pachner_candidate_moves=DECLARED_PACHNER_CANDIDATE_MOVES,
                   limits=None,
                   villain_order=bp.DECLARED_VILLAIN_ORDER, solve=None,
                   regge_hinges=DECLARED_REGGE_HINGES,
                   cosmological_constant=bp.DECLARED_COSMOLOGICAL_CONSTANT):
    """The declared configuration, recorded with every run. ``max_cells``
    limits how many tetrahedra per tick are read as hosts, for quick checks;
    it changes no number of the cells it keeps. ``persistence_required`` is
    how many adjacent declared resolutions a component must persist across
    to become a response vertex; by default every declared resolution, the
    stated range of scales of WP §5. ``tolerances`` sets any of
    `baryon_poles.TOLERANCES` by key; every tolerance is carried into every
    cell's config. ``limits`` declares any of `baryon_poles.LIMITS` by key
    (none by default: no count and no time ends a solve); every declared
    limit is carried into every solve of the run. ``villain_order`` is the
    order the Villain weight of the holonomy term is summed to
    (`baryon_poles.DECLARED_VILLAIN_ORDER`); it is carried into every level's
    action and every cell's config. ``solve`` sets any of
    `baryon_poles.SOLVE_OPTIONS` by key, for the drive of every level's
    relaxation and of every cell's solve. ``regge_hinges`` names the hinges
    of the primal Regge sum of every level's and every cell's action.
    ``cosmological_constant`` is Lambda
    (`baryon_poles.DECLARED_COSMOLOGICAL_CONSTANT`): a nonzero value is
    recorded and carried into every level's, every cell's and the growth
    step's action; zero leaves the term and the key out.

    A declaration that has no meaning is named here, before a tick is
    computed (`checked_run`)."""
    checked_run(ticks, tetrahedra, kappa, beta, resolutions, band_rank,
                persistence_required, max_cells, pachner_updates,
                pachner_depth, pachner_length, pachner_candidate_moves)
    config = bp.default_config(kappas=[kappa], betas=[beta],
                               regge_hinges=regge_hinges,
                               edge_squared=edge_squared,
                               elimination=elimination,
                               selected_contents=selected_contents,
                               band_selection=band_selection,
                               fiber_moments=fiber_moments,
                               fiber_pinning=fiber_pinning,
                               tolerances=tolerances, trace_terms=trace_terms,
                               limits=limits, villain_order=villain_order,
                               solve=solve,
                               cosmological_constant=cosmological_constant)
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
        "pachner_updates": int(pachner_updates),
        "pachner_depth": int(pachner_depth),
        "pachner_length": int(pachner_length),
        "pachner_objective": checked_pachner_objective(pachner_objective),
        "pachner_candidate_moves": int(pachner_candidate_moves),
        "pachner_stage": ("stage-1 updates of MultiCobordism on each grown "
                          "level's base (one sheet) under the declared "
                          "pachner_objective, the four Pachner kinds "
                          "alone: no cone-out, cone-in or disposition move; "
                          "every candidate move of the base scored, no "
                          "sample and no seed; zero updates run none"),
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
    its first line and every tick is appended the moment it completes. A tick
    that ends on an error which is not one of
    `baryon_poles.NO_VALUE_ERRORS` is a defect of the code: what the tick
    reached (the ``tick_record`` the error carries) is appended to the
    points file, and the error is raised."""
    cells, z, links, connection = level_zero(config)
    host = {
        "cells": cells,
        "face_angles_over_pi": [float(a / math.pi)
                                for a in connection["face_angles"]],
        "dirac_string": [int(n) for n in connection["dirac_string"]],
        "connection_residual": connection["residual"],
        "monopole_numbers": monopole_numbers(
            cells, links,
            bp.declared_tolerance(config, "certificate_tolerance")),
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
        try:
            record, state = tick(index, *state, config)
        except Exception as error:
            reached = getattr(error, "tick_record", None)
            if reached is not None and points_file is not None:
                _append_line(points_file, _reached(reached))
            raise
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
            for line in read_lines(record):
                sys.stdout.write(line + "\n")
            sys.stdout.flush()
        if on_frame is not None:
            on_frame(frames, len(frames) - 1)
        if state is None:
            break
    return {"config": _jsonable(config), "host": host, "ticks": frames,
            "stopped": stopped}


def _reached(record):
    """What a tick reached before it ended on an error, as the points file
    can hold it: the record with every entry that has no JSON form replaced
    by its `repr`."""
    def plain(value):
        value = _jsonable(value)
        if isinstance(value, dict):
            return {key: plain(entry) for key, entry in value.items()}
        if isinstance(value, list):
            return [plain(entry) for entry in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)
    return plain(record)


def _notices(record):
    """What a tick left out or flagged, by name: a Regge term that is zero
    by structure, reads of the relaxed level that have no value, a
    covariance certificate of the level's operator that does not hold,
    components
    rejected for persistence, components whose fiber has no value, grown
    cells left unattached by the manifold gate, cells whose read has no
    value, and a growth step whose search has no value."""
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
    for key, reason in sorted(((record.get("level") or {})
                               .get("without_value") or {}).items()):
        lines.append("the level's %s has no value: %s" % (key, reason))
    for name, certificate in sorted(((record.get("level") or {})
                                     .get("operator_certificates")
                                     or {}).items()):
        if not certificate["holds"]:
            lines.append(
                "the covariance certificate of the level's operator (%s) "
                "does not hold at the tolerance %.3g, and the level is read "
                "on it as built: %s"
                % (name.replace("_", " "), certificate["tolerance"],
                   "; ".join(certificate["failed"])
                   or "the library names no property"))
    for failed in record.get("failed_components", []):
        lines.append("component %d (edges %s) is not a response vertex: its "
                     "fiber has no value: %s"
                     % (failed["component"], ", ".join(failed["edges"]),
                        failed["failed"]))
    for cell in record.get("grown_cells", []):
        if "rejected" in cell:
            lines.append("grown cell %s rejected by the manifold gate: %s"
                         % (cell["vertices"], cell["rejected"]))
    for cell in record.get("reads") or []:
        if "failed" in cell:
            lines.append("host cell %s has no read: %s"
                         % (cell["cell"], cell["failed"]))
    if "failed" in (record.get("pachner") or {}):
        lines.append("the growth step's search over the Pachner moves has no "
                     "value, and the grown base is the next level as it is: "
                     "%s" % record["pachner"]["failed"])
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


def frame_data(frames, index):
    """What one frame draws, as data: the recursion's counts and the row-sum
    defects of its grown cells per tick; every pole of every (host cell,
    content, doublet content) of the latest tick that read cells
    (`baryon_poles.pole_marks`, one group per host cell and content); and the
    ratio by 2T reading of every read host cell in both columns with the two
    pairs it compares and the mean-field solve behind each
    (`baryon_poles.ratio_row`)."""
    done = frames[:index + 1]
    read = [f for f in done if f.get("reads")]
    latest = read[-1] if read else done[-1]
    cells = latest.get("reads") or []
    marks, spans, slots = bp.pole_marks(
        [("%s\n%s" % (bp._digits(cell["cell"]), bp._digits(record["content"])),
          record) for cell in cells for record in cell["contents"]])
    rows = []
    for cell in cells:
        where = "cell %s" % bp._digits(cell["cell"])
        solves = bp.content_solves(cell["contents"])
        for name in bp.COLUMNS:
            rows.append(bp.ratio_row(where, name, ((cell.get("ratios") or {})
                                                   .get(name) or {})
                                     .get("by_2T_reading"), solves))
    return {"tick": latest["tick"], "done": len(done),
            "counts": [{"tick": f["tick"],
                        "response_vertices": f["summary"]["response_vertices"],
                        "grown_cells": f["summary"]["grown_cells"],
                        "row_sum_defects": list(
                            f["summary"]["row_sum_defects"])} for f in done],
            "marks": marks, "groups": spans, "slots": slots, "ratios": rows}


def draw_frame(figure, frames, index):
    """One frame (`frame_data`): the poles of every (host cell, content,
    doublet content) of the latest tick that read cells, quasi-free and with
    the quartic, each pole its own mark; below them the recursion's counts
    per tick, the row-sum defects of its grown cells, the ratio by 2T reading
    per host cell against the target, and the listing of the pairs each ratio
    compares."""
    data = frame_data(frames, index)
    figure.clear()
    figure.patch.set_facecolor(SURFACE)
    grid = figure.add_gridspec(3, 4, height_ratios=(1.2, 1.2, 1.1),
                               width_ratios=(1.0, 1.0, 1.1, 1.6))
    quasi_free = figure.add_subplot(grid[0, :])
    quartic = figure.add_subplot(grid[1, :])
    counts = figure.add_subplot(grid[2, 0])
    defects = figure.add_subplot(grid[2, 1])
    ratio = figure.add_subplot(grid[2, 2])
    pairs = figure.add_subplot(grid[2, 3])
    for axis, name in ((quasi_free, "quasi_free"), (quartic, "with_quartic")):
        bp.draw_pole_panel(axis, data, name,
                           "poles %s at tick %d: every (host cell, content, "
                           "doublet content)" % (bp.SERIES_LABEL[name],
                                                 data["tick"]),
                           "host cell and content (quarks per band of h_1)")
    for axis in (counts, defects):
        bp.style_axis(axis)
    ticks = [row["tick"] for row in data["counts"]]
    for key, colour, label in (("response_vertices", "#2a78d6",
                                "response vertices"),
                               ("grown_cells", "#eb6834", "grown cells")):
        counts.plot(ticks, [row[key] for row in data["counts"]], marker="o",
                    linewidth=2, color=colour, label=label)
    counts.set_xticks(ticks)
    counts.set_xlabel("tick", color=INK, fontsize=8)
    counts.set_ylabel("count", color=INK, fontsize=8)
    counts.set_title("the recursion per tick", color=INK, fontsize=9)
    counts.legend(frameon=False, fontsize=7, labelcolor=INK)
    for row in data["counts"]:
        defects.plot([row["tick"]] * len(row["row_sum_defects"]),
                     row["row_sum_defects"], linestyle="none", marker="_",
                     markersize=14, color="#1baf7a")
    defects.set_xticks(ticks)
    defects.set_xlabel("tick", color=INK, fontsize=8)
    defects.set_ylabel("||g 1|| / ||g||", color=INK, fontsize=8)
    defects.set_title("row-sum defect of each grown cell", color=INK,
                      fontsize=9)
    bp.draw_ratio_panel(ratio, data["ratios"],
                        "ratio by 2T reading per host cell, tick %d"
                        % data["tick"])
    bp.draw_pairs_panel(pairs, data["ratios"])
    figure.suptitle("level recursion with the grown-cell rule, reported per "
                    "doublet content (%d ticks done)" % data["done"],
                    color=INK)
    figure.tight_layout()


def drive_live(config, progress=False, points_file=None, keep_open=False):
    """The same `drive` on a worker thread, drawing each completed tick on the
    main thread; the window is shown once, never raised, and pumped with
    `canvas.start_event_loop`. While a tick is being computed the window says
    which one and for how long. Closing it switches the run to headless. When
    the run ends the figure is closed, or, with `keep_open`, left on screen
    with its final frame for `bp.hold_live_window` once the outputs are
    written."""
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
    figure = plt.figure(figsize=bp.FIGURE_SIZE)
    plt.show(block=False)
    ready = queue.Queue()
    published = {}
    outcome = {}
    stop = threading.Event()
    closed = threading.Event()
    finished = threading.Event()

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    def on_close(event):
        # Only a close by the user while the run is in progress is reported;
        # the driver's own close at the end, and a close after the end, are
        # not.
        if closed.is_set():
            return
        closed.set()
        if not finished.is_set():
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
    def running(done, seconds):
        if done >= config["ticks"]:
            return "the run is finishing: %s" % bp.elapsed_text(seconds)
        return ("tick %d of %d is running: %s elapsed; its frame is drawn "
                "when the tick completes"
                % (done, config["ticks"], bp.elapsed_text(seconds)))

    main_error = None
    done = 0
    since = time.monotonic()
    shown = None
    try:
        while not closed.is_set():
            try:
                index = ready.get_nowait()
            except queue.Empty:
                now = time.monotonic()
                if shown is None or now - shown >= bp.LIVE_STATUS_INTERVAL:
                    bp.draw_status(figure, running(done, now - since))
                    shown = now
                figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
                continue
            if index is None or closed.is_set():
                break
            draw_frame(figure, published["frames"], index)
            done, since, shown = index + 1, time.monotonic(), None
            figure.canvas.draw_idle()
            figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
    except BaseException as error:
        main_error = error
        stop.set()
    finally:
        thread.join()
        if not closed.is_set():
            finished.set()
            if keep_open and main_error is None and "error" not in outcome:
                bp.draw_status(figure, "the run is complete; writing the "
                                       "outputs")
                figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
            else:
                plt.close(figure)
    if main_error is not None:
        raise main_error
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def render(result, path):
    """The final frame as a PNG, drawn on its own Agg canvas, so an open live
    window and the session's backend are left as they are."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=bp.FIGURE_SIZE)
    FigureCanvasAgg(figure)
    if result["ticks"]:
        draw_frame(figure, result["ticks"], len(result["ticks"]) - 1)
    figure.savefig(path, dpi=120, facecolor=SURFACE)


def summary(result):
    """Every tick as text. A tick that stopped says why: a level with no
    value as built and a relaxation the library refused are one line each;
    a tick whose operator, turn of the Section 15 box or interaction stage
    has no value is reported with its level, its relaxation, what it read
    before the stop and the reads of its cells, then its stop."""
    lines = ["mode: controlled synthesis; host monopole numbers %s"
             % result["host"]["monopole_numbers"]]
    for record in result["ticks"]:
        if "relaxation" not in record:
            lines.append("tick %d: stopped: %s"
                         % (record["tick"], record.get("stopped")))
            continue
        if "failed" in record["relaxation"]:
            lines.append("tick %d: the relaxation was refused: %s"
                         % (record["tick"], record["relaxation"]["failed"]))
            continue
        s = record["summary"]
        level = record["level"]
        lines.append(
            "tick %d: level with %d vertices, %d edges, %d tetrahedra per "
            "sheet; relaxation converged %s (residual %.3g%s); bulk monopole "
            "numbers %s before relaxation, %s after"
            % (record["tick"], level["vertices"], level["edges"],
               level["tetrahedra"], record["relaxation"]["converged"],
               record["relaxation"]["residual"],
               ", stopped: %s (%s)" % (record["relaxation"]["stop_reason"],
                                       record["relaxation"]["stop_detail"])
               if "stop_reason" in record["relaxation"] else "",
               level.get("bulk_monopole_numbers_before"),
               level.get("bulk_monopole_numbers_after")))
        if level.get("cells_after") is not None:
            lines.append(
                "  committed moves changed the level's cells: the numbers "
                "before are of the cells %s, those after of the cells %s "
                "(by the drive's vertex names; on the level's vertices %s)"
                % (level["cells_before"], level["cells_after"],
                   level["cells"]))
        lines += bp.term_trace_lines(record["relaxation"], "    ")
        held = level.get("held_cut") or {}
        if held.get("faces"):
            lines.append(
                "  held bounding cut %s: monopole number %s before, %s after"
                % (held["faces"], held.get("monopole_number_before"),
                   held.get("monopole_number_after")))
        for line in _notices(record):
            lines.append("  " + line)
        p = record.get("partition") or {}
        if p and "failed" not in p:
            lines.append(
                "  partition %s at resolution %g; fibers accepted %s; "
                "isolation gaps %s; determinant residual %.3g"
                % ([len(c) for c in p["partition"]],
                   p["selected_resolution"], p["bands_accepted"],
                   ["%.3g" % g for g in p["isolation_gaps"]],
                   p["determinant_residual"]))
        if "grown_cells" in record:
            lines.append(
                "  %d response vertices, %d interactions, %d grown cells"
                % (s["response_vertices"], s["interactions"],
                   s["grown_cells"]))
        for cell in record.get("grown_cells", []):
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
        lines += read_lines(record)
        if "stopped" in record:
            lines.append("  stopped: " + record["stopped"])
    return "\n".join(lines)


def _conditions_text(record):
    """The quark conditions of one content by number, name and status."""
    quark = record.get("quark_conditions") or {}
    return ", ".join(
        " ".join(str(part) for part in (c.get("number"), c["name"],
                                         c["status"]) if part is not None)
        for c in quark.get("conditions") or [])


def _content_line(cell, c):
    """One content of one host cell: whether it was read (or why its read
    has no value), its mean-field solve (`baryon_poles.relaxation_text`), its
    anchor atlas (`baryon_poles.anchor_text`), its spectral fingerprint
    (`baryon_poles.fingerprint_text`), its quark verdict and every
    quark condition's status, its quarks per doublet of h-bar_1, its
    quartic truncation certificates and, when the read is flagged, its flags
    (`baryon_poles.flags_text`)."""
    head = "host cell %s content %s (quarks per band of h_1): " % (
        cell["cell"], c["content"])
    if "failed" in c:
        line = head + "no value: " + c["failed"]
        if c.get("relaxation"):
            line += "; " + bp.relaxation_text(c["relaxation"])
        return line
    verdict = _verdict_summary(c)
    spin = (c.get("spin_decomposition") or {}).get("occupied_state")
    flagged = bp.flags_text(c)
    return head + ("read%s; quark certified %s; %s; %squarks per doublet of "
                   "h-bar_1 %s; quartic truncation %s; quark conditions %s"
                   % (", " + flagged if flagged else "",
                      verdict["certified"] if verdict else None,
                      bp.relaxation_text(c.get("relaxation")),
                      (bp.anchor_text(c["anchor"]) + "; "
                       if "anchor" in c else "")
                      + (bp.fingerprint_text(c["spectral_fingerprint"])
                         + "; " if "spectral_fingerprint" in c else ""),
                      {k: bp._complex_text(v) for k, v in spin.items()}
                      if spin else None,
                      _truncation_summary(c), _conditions_text(c)))


def read_lines(record):
    """The per-cell reads of one tick as text. For every host cell: one line
    per content (`_content_line`), each read content followed by one line per
    (content, doublet content) pair with both spins and both columns
    (`baryon_poles.pair_line`); then the cell's labelled "lowest over" minima
    (`baryon_poles.lowest_lines`) and its ratios with the pairs they compare
    (`baryon_poles.ratio_lines`). A cell whose read has no value is one line
    with the reason."""
    lines = []
    for cell in record.get("reads") or []:
        prefix = "host cell %s " % (cell["cell"],)
        if "failed" in cell:
            lines.append("    %sno value: %s" % (prefix, cell["failed"]))
            continue
        for c in cell["contents"]:
            lines.append("    " + _content_line(cell, c))
            lines += bp.term_trace_lines(c.get("relaxation"), "        ")
            if "failed" not in c:
                lines += ["      " + line
                          for line in bp.content_pair_lines(c, prefix)]
        lines += bp.lowest_lines(
            cell["contents"], "    " + prefix,
            bp.number(cell.get("tie_tolerance", bp.DECLARED_TIE_TOLERANCE)))
        lines += bp.ratio_lines(cell.get("ratios"), "    " + prefix)
    return lines


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
                          "of each level's box; they reach the level's box "
                          "alone, and the recursion read of every cell "
                          "(its quark condition 1) is taken at the "
                          "library's own resolutions, which its record "
                          "names under resolutions (default %s)"
                          % (DECLARED_RESOLUTIONS,))
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
    run.add_argument("--eliminate", choices=bp.ELIMINATIONS,
                     default=bp.DECLARED_ELIMINATION,
                     help="the fluctuations the Section 7 quartic eliminates "
                          "in every cell's read, as in baryon_poles "
                          "(default %s)" % bp.DECLARED_ELIMINATION)
    run.add_argument("--regge-hinges", choices=REGGE_HINGES,
                     default=DECLARED_REGGE_HINGES,
                     help="hinges of the primal Regge sum of every level's "
                          "and every cell's action (default %s)"
                          % DECLARED_REGGE_HINGES)
    run.add_argument("--contents", type=int, nargs=3, action="append",
                     default=None, metavar=("N1", "N2", "N3"),
                     help="a content to read on every cell (repeatable; "
                          "default all ten)")
    run.add_argument("--max-cells", type=int, default=None,
                     help="read at most this many tetrahedra per tick as "
                          "hosts (a quick-check limit; default all)")
    run.add_argument("--pachner-updates", type=int,
                     default=DECLARED_PACHNER_UPDATES,
                     help="stage-1 updates of the Pachner-move search on each "
                          "grown level's base (the four Pachner kinds under "
                          "the joint stationarity objective; no surgery); 0 "
                          "runs none (default %d)" % DECLARED_PACHNER_UPDATES)
    run.add_argument("--pachner-depth", type=int,
                     default=DECLARED_PACHNER_DEPTH,
                     help="how many moves deep the search goes when no single "
                          "move lowers the objective (default %d)"
                          % DECLARED_PACHNER_DEPTH)
    run.add_argument("--pachner-length", type=int,
                     default=DECLARED_PACHNER_LENGTH,
                     help="a fixed composition length for the growth step's "
                          "search: sequences of exactly this many moves are "
                          "searched first, backing off one move at a time; "
                          "the alternative to --pachner-depth, 0 keeps the "
                          "deepening schedule (default %d)"
                          % DECLARED_PACHNER_LENGTH)
    run.add_argument("--pachner-objective", choices=PACHNER_OBJECTIVES,
                     default=DECLARED_PACHNER_OBJECTIVE,
                     help="what the growth step's search scores a base with: "
                          "joint-action, the stationarity of the joint "
                          "action the level's relaxation descends, its Regge "
                          "term on the sheet continued from the level as "
                          "grown; engine, the Regge action on the principal "
                          "sheet of its dihedral angles with the Hodge "
                          "spectral entropies, whose value on a level with "
                          "squared lengths real to rounding depends on the "
                          "signs of their imaginary parts (default %s)"
                          % DECLARED_PACHNER_OBJECTIVE)
    run.add_argument("--pachner-candidate-moves", type=int,
                     default=DECLARED_PACHNER_CANDIDATE_MOVES,
                     help="how many candidates each update of the growth "
                          "step's search draws at random, each a sequence of "
                          "the search's depth or length; 0 scores every "
                          "candidate (default %d)"
                          % DECLARED_PACHNER_CANDIDATE_MOVES)
    run.add_argument("--json", default=None,
                     help="write every record here at the end; each tick is "
                          "also appended, as it completes, to "
                          "<json stem>.points.jsonl")
    run.add_argument("--out", default=None,
                     help="write the final frame as a PNG here")
    run.add_argument("--live", action="store_true",
                     help="draw each completed tick while the run proceeds; "
                          "the outputs are identical")
    bp.add_action_arguments(run)
    bp.add_cosmological_constant_argument(run)
    bp.add_mean_field_arguments(run)
    bp.add_tolerance_arguments(run)
    bp.add_limit_arguments(run)
    bp.add_solve_arguments(run)
    run.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = default_config(
        ticks=args.ticks, tetrahedra=args.tetrahedra, kappa=args.kappa,
        beta=args.beta, resolutions=args.resolutions,
        band_rank=args.band_rank, edge_squared=args.edge_squared,
        elimination=args.eliminate,
        selected_contents=[tuple(c) for c in args.contents]
        if args.contents else None, max_cells=args.max_cells,
        persistence_required=args.persistence_required,
        band_selection=args.band_selection,
        fiber_moments=args.fiber_moments,
        fiber_pinning=args.fiber_pinning,
        tolerances=bp.tolerances_from(args), trace_terms=args.trace_terms,
        pachner_updates=args.pachner_updates,
        pachner_depth=args.pachner_depth,
        pachner_length=args.pachner_length,
        pachner_objective=args.pachner_objective,
        pachner_candidate_moves=args.pachner_candidate_moves,
        regge_hinges=args.regge_hinges,
        limits=bp.limits_from(args),
        solve=bp.solve_options_from(args),
        villain_order=args.villain_order,
        cosmological_constant=args.cosmological_constant)
    # what the run's numbers depend on beside its declarations: the command
    # line, the commit, the thread count and the linear algebra library
    config["environment"] = bp.environment_record(argv)
    points_file = points_path(args.json) if args.json else None
    result = (drive_live(config, progress=not args.quiet,
                         points_file=points_file, keep_open=True)
              if args.live
              else drive(config, progress=not args.quiet,
                         points_file=points_file))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(_jsonable(result), handle, indent=1, allow_nan=False)
    if args.out:
        render(result, args.out)
    if not args.quiet:
        sys.stdout.write(summary(result) + "\n")
    if args.live:
        bp.hold_live_window("the run is complete and every output is "
                            "written; close this window to exit")
    return result


if __name__ == "__main__":
    main()
