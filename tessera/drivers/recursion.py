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
   level's own lengths as the level was built;
2. the Section 15 box runs on the base: the partition of the covariant
   edge-mode operator h_1(z, U) of sheet 0 (`LevelRecursion`, the library's
   `PersistentPartition`, a modularity sweep over the declared resolutions),
   the Riesz fibers of the declared rank, the Feshbach reduction with its
   determinant-factorization certificate, the transports M_vw and the labeled
   sum. On sheet-diagonal couplings the reduction acts on the base and carries
   the sheet factor along (WP §15), so the other two sheets carry the same
   fibers; the sheet isomorphism of the relaxed level is certified;
3. the interaction stage: two response vertices interact exactly when their
   supports share a top simplex (the coupling block of the Whitney metric is
   nonzero; spec Prop. 7.1), and four that interact pairwise span a grown
   3-simplex (the flag complex of the interaction graph);
4. the grown-cell rule (WP v17 §15): the inherited pairing on the
   determinant line, g_vw = det(Y_v^T G_1^U Y_w), is read as |T| Gamma, so
   C Gamma = g / 20; the block of C Gamma off v_0 is inverted to g/C,
   C = 14400 / det(g/C), and the squared lengths are the quadratic form of g
   on the edge vectors. The row-sum defect ||g 1|| / ||g|| of every grown cell
   is its certificate. The connection of a grown edge (v < w) is
   U_vw = det M_vw. An edge shared by several grown cells receives one value
   from each; the level carries their mean and reports their spread;
5. the reads, behind the certificate firewall: every tetrahedron of the base
   of K_l, read as a three-sheeted host of its own (``baryon_poles``), gives
   the v16 quark verdicts (`QuarkConditions`), the isospin-doublet reading
   (`IsospinDoublet`) and the baryon poles, quasi-free and with the Section 7
   quartic with the connection's phase fluctuations eliminated, for every
   declared content.

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
DECLARED_TETRAHEDRA = 3
DECLARED_EDGE_SQUARED = 8.0
DECLARED_MONOPOLE = 1
DECLARED_TICKS = 3
DECLARED_KAPPA = 1.0
DECLARED_BETA = 1.0
DECLARED_RESOLUTIONS = (1.0, 1.5, 2.0, 2.5, 3.0)
DECLARED_BAND_RANK = 1
DECLARED_CONTOUR_NODES = 64
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


def relax_level(spacetime, config):
    """Step 1: both edge fields relax to holomorphic stationarity of the joint
    action, strict emergence, with the level's lengths as built as l0."""
    declaration = bp.action_declaration(spacetime, config["kappa"],
                                        config["beta"],
                                        config["regge_hinges"],
                                        holonomy=config["holonomy"])
    action = cob.JointAction(spacetime, declaration)
    relaxation = cob.HolomorphicRelaxation(action,
                                           bp.relaxation_declaration(config))
    started = time.time()
    report = relaxation.solve()
    return {
        "converged": bool(report.converged),
        "initial_residual": float(report.initial_residual_norm),
        "residual": float(report.residual_norm),
        "iterations": len(report.steps),
        "zero_guard_damped_steps": int(report.zero_guard_damped_steps),
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
    covariant = ch.CovariantChainHodge(ch.ChainHodge(complex_, squared),
                                       connection)
    operator = np.asarray(covariant.covariantOperator(1))
    return {"complex": complex_, "edges": edges, "tops": tops,
            "covariant": covariant, "operator": operator}


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


def fiber_frames(level):
    """The right frame Y_v of every fiber (level dimension by r_v) and the
    offsets of the fibers in the labeled sum."""
    n = int(level.dimension)
    modes = int(level.modes)
    embedding = np.asarray(level.embedding).reshape(n, modes)
    frames, offsets, start = [], [], 0
    for band in level.bands:
        r = int(band.rank)
        frames.append(embedding[:, start:start + r])
        offsets.append((start, start + r))
        start += r
    return frames, offsets


def transport_blocks(level, offsets):
    """M_vw, the (v, w) block of the fiber operator (Hom(E_w, E_v))."""
    modes = int(level.modes)
    fiber = np.asarray(level.fiber_operator).reshape(modes, modes)
    return {(v, w): fiber[offsets[v][0]:offsets[v][1],
                          offsets[w][0]:offsets[w][1]]
            for v in range(len(offsets)) for w in range(len(offsets))}


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


def inherited_pairing(frames, covariant):
    """g_vw = det(Y_v^T G_1^U Y_w) over all response vertices."""
    images = [np.asarray(covariant.applyG(1, Y)) for Y in frames]
    return np.asarray(ch.GrownCellRule.determinantPairing(frames, images))


def grow(cells, pairing, transports):
    """The grown-cell rule on every grown 3-simplex: lengths from the
    inherited pairing, connection from det M. Returns the per-cell reads and
    the per-edge fields of the next level (response-vertex labels)."""
    reads, per_edge_z = [], {}
    for cell in cells:
        index = list(cell)
        block = pairing[np.ix_(index, index)]
        entry = {"vertices": index, "pairing": block}
        try:
            read = ch.GrownCellRule.invertVertexPairing(block)
        except ValueError as error:
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
    declared content (`baryon_poles.evaluate_content`)."""
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
    out = {}
    for key, entry in record.get("sectors", {}).items():
        out[key] = {name: entry[name]["lowest_pole"]
                    for name in ("quasi_free", "with_quartic")}
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


def tick(index, cells, z, links, config):
    """One tick on the level whose base is ``cells`` with fields ``z`` and
    ``links``. Returns the tick's record and the next level's base (or None
    when no 3-simplex grows)."""
    started = time.time()
    spacetime, count = build_level(cells, z, links)
    declared_monopoles = monopole_numbers(cells, links)
    try:
        relaxation = relax_level(spacetime, config)
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
                "declared_monopole_numbers": declared_monopoles,
            },
            "relaxation": {"failed": str(error)},
            "summary": {"response_vertices": 0, "interactions": 0,
                        "grown_cells": 0, "failed_cells": 0,
                        "row_sum_defects": []},
            "reads": [],
            "stopped": "the level's relaxation was refused: %s" % error,
            "seconds": time.time() - started,
        }
        return record, None
    fields = sheet_fields(spacetime, count)
    base_z, base_links = fields[0]
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
    frames, offsets = fiber_frames(level)
    transports = transport_blocks(level, offsets)
    pairs = interaction_graph(level.partition, base["edges"], base["tops"])
    grown = grown_cells(len(frames), pairs)
    pairing = inherited_pairing(frames, base["covariant"])
    reads, next_z, next_links, spread, groupoid = grow(grown, pairing,
                                                       transports)
    kept = [r for r in reads if "failed" not in r]
    record = {
        "tick": index,
        "level": {
            "vertices": 1 + max(max(c) for c in cells),
            "edges": len(base["edges"]),
            "tetrahedra": len(base["tops"]),
            "cells": [sorted(c) for c in cells],
            "squared_lengths": {"%d-%d" % e: v for e, v in base_z.items()},
            "links": {"%d-%d" % e: v for e, v in base_links.items()},
            "declared_monopole_numbers": declared_monopoles,
            "monopole_numbers": monopole_numbers(cells, base_links),
            "sheet_isomorphism_residual": float(isomorphism),
        },
        "relaxation": relaxation,
        "partition": level_record(level),
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
    record["reads"] = cell_reads(cells, base_z, base_links, config)
    record["summary"] = {
        "response_vertices": len(frames),
        "interactions": len(pairs),
        "grown_cells": len(kept),
        "failed_cells": len(reads) - len(kept),
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
                             "interacting response vertices")
        return record, None
    used = sorted({v for r in kept for v in r["vertices"]})
    relabel = {v: i for i, v in enumerate(used)}
    record["lineage"] = {str(v): relabel.get(v) for v in range(len(frames))}
    next_cells = [[relabel[v] for v in r["vertices"]] for r in kept]
    next_edges = {tuple(sorted((relabel[a], relabel[b]))): (a, b)
                  for (a, b) in next_z}
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
                   selected_contents=None, max_cells=None):
    """The declared configuration, recorded with every run. ``max_cells``
    limits how many tetrahedra per tick are read as hosts, for quick checks;
    it changes no number of the cells it keeps."""
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
        "pairing": "det(Y_v^T G_1^U Y_w), read as |T| Gamma",
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
            sys.stdout.flush()
        if on_frame is not None:
            on_frame(frames, len(frames) - 1)
        if state is None:
            break
    return {"config": _jsonable(config), "host": host, "ticks": frames,
            "stopped": stopped}


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

    latest = done[-1]
    labels, half, three = [], [], []
    for cell in latest["reads"]:
        for record in cell["contents"]:
            labels.append("%s\n%s" % ("".join(map(str, cell["cell"])),
                                      "".join(map(str, record["content"]))))
            for store, key in ((half, str(bp.SPIN_HALF)),
                               (three, str(bp.SPIN_THREE_HALVES))):
                entry = record.get("sectors", {}).get(key)
                pole = entry["quasi_free"]["lowest_pole"] if entry else None
                store.append(pole.real if pole is not None else np.nan)
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
        s = record["summary"]
        p = record["partition"]
        if "failed" in record["relaxation"]:
            lines.append("tick %d: the relaxation was refused: %s"
                         % (record["tick"], record["relaxation"]["failed"]))
            continue
        lines.append(
            "tick %d: level with %d vertices, %d edges, %d tetrahedra per "
            "sheet; relaxation converged %s (residual %.3g); monopole "
            "numbers %s as built, %s relaxed"
            % (record["tick"], record["level"]["vertices"],
               record["level"]["edges"], record["level"]["tetrahedra"],
               record["relaxation"]["converged"],
               record["relaxation"]["residual"],
               record["level"]["declared_monopole_numbers"],
               record["level"]["monopole_numbers"]))
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
            lines.append(
                "    cell %s: row-sum defect %.4g, asymmetry %.3g, z %s"
                % (cell["vertices"], cell["row_sum_defect"],
                   cell["asymmetry"],
                   [bp._complex_text(v) for v in cell["squared_lengths"]]))
        for cell in record["reads"]:
            for c in cell["contents"]:
                verdict = _verdict_summary(c)
                lines.append(
                    "    host cell %s content %s: %s; quark certified %s; "
                    "lowest poles %s; quartic truncation %s"
                    % (cell["cell"], c["content"],
                       "failed: " + c["failed"] if "failed" in c else "read",
                       verdict["certified"] if verdict else None,
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
        if args.contents else None, max_cells=args.max_cells)
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
