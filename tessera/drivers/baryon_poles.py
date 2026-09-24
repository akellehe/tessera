# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The nucleon-to-Delta pole ratio by controlled synthesis.

This driver computes, with the whitepaper's own mechanisms (v16), the ratio of
the spin-1/2 and spin-3/2 bound-state poles of three quarks on one host, and
reports it against the target: the proton mass over the Breit-Wigner mass of the
Delta(1232), m_N / m_Delta = 938.272 / 1232 = 0.7616, and its square 0.5800.
Nothing here is tuned toward the target. The design, with every step mapped to
the paper and to the library, is ``reports/design/nucleon_delta_by_synthesis.md``
in the notes repository.

Mode
----
Labelled controlled synthesis (WP §3.2): the host is prepared, not emerged.

The host
--------
Three isomorphic, disjoint sheets of the regular tetrahedron (squared edge
length ``a^2``, 8 by default, the paper's own units), each carrying the
symmetric unit Dirac monopole, whose every outward face holonomy is
exp(2 pi i / 4) (WP §10 condition 2, §11.1). The sheets are the literal case of
the sheet convention (WP §8), attached sheet to sheet.

One scan point
--------------
For each declared (kappa, beta), with kappa = 8 pi G in lattice units and beta
the coupling of the face-holonomy term, and for each content (the number of
quarks in each of the three bands of the host), the driver

1. relaxes both edge fields under the joint action
   S = (1/kappa) S_Regge(primal) + (1/kappa) (1/2) ||l - l0||^2 + S_hol
       + tr(Gamma h_1)
   whose holonomy term S_hol is declared by ``--holonomy``: the Villain form
   -beta_V sum_tau log W(F_tau), W(F) = sum_m exp(-m^2/(2 beta)) F^m,
   beta_V = beta / <m^2>_beta (the default, the paper's holonomy term), or the
   Wilson form beta sum_tau (1 - cos Theta_tau) (the paper's stand-in). Both
   have the bare connection stiffness beta L_1^up at trivial holonomy; at the
   host's quarter-turn holonomies the Wilson stiffness vanishes and the
   Villain stiffness does not;
   with certificates-blind mean-field backreaction to self-consistency
   (`HolomorphicRelaxation` inside `SelfConsistentMeanField`), the carried
   density being the content's band filling;
2. runs one turn of the level recursion (`LevelRecursion`) and reads the fibre
   certificates;
3. reads the seven v16 quark conditions by name (`QuarkConditions`);
4. forms the colour-singlet three-quark states of the content, one quark per
   sheet, sorted by total spin with `SharpSpin`;
5. builds the three-particle operator, both quasi-free (dGamma(h_1)) and with
   the paper's Section 7 geometric quartic about the self-consistent point
   (`DressedFluctuation.effectiveAction`), and reads the pole of every
   (content, spin) sector with `BoundStatePole`.

The nucleon pole is the lowest spin-1/2 pole over the contents and the Delta
pole the lowest spin-3/2 pole, "lowest" meaning smallest real part, which is the
library's declared `OccupationOrder.AscendingRealPart`. The poles are complex
and are reported as complex; the ratio s_N / s_Delta is compared with 0.5800,
the mass-squared reading (the pole of a Laplace-type operator has the dimension
of p^2, and the paper takes no square root of it), and the ratio of moduli and
the ratio of real parts are reported beside it.

The operator the spin is read on
-------------------------------
The covariant operator h_1(z, U) of the specification (Definition 2) is not
itself invariant under the projective rotation action D_1(g) at the monopole
connection: its six eigenvalues per sheet are simple. The whitepaper reads the
spin content on the T-averaged operator (WP §11.1 line 497: "The action on edge
modes is canonical even where the twisted operator is not"; "the T-averaged
twisted edge Laplacian"), h-bar_1 = (1/|T|) sum_g D_1(g)^{-1} h_1 D_1(g), with
T = A_4 the rotation group of the tetrahedron. The driver does the same: the
bands the carried density fills, the spin sectors and every pole are read on
h-bar_1 (and, for the Section 7 quartic, on the eliminated three-particle
operator averaged over the same diagonal action). The departure of h_1 itself
from the symmetric form is reported beside every record as `symmetry_residual`.

The same paragraph's caveat is carried with every Delta candidate: on the
tetrahedron the j = 3/2 quartet restricts to 2' + 2'', so "spin-1/2 with
triality" and "half of spin-3/2" are indistinguishable. A Delta reading is a
degenerate 2' + 2'' pair of three-quark states and a nucleon reading is a 2;
each sector reports its restriction, and the ratios are reported both by sharp
spin and by 2T reading.

Running it
----------
::

    python -m tessera.drivers.baryon_poles run \\
        --kappa 0.25 0.5 1 2 4 --beta 0.5 1 2 5 --json poles.json \\
        --out poles.png [--holonomy {villain,wilson}] [--live]

``--isospin-doublet`` adds, to every content's record, the isospin-doublet
observation of `tessera.drivers.isospin_doublet` on h_1 and on h-bar_1; every
other output is unchanged.

``--live`` draws each completed scan point while the scan runs, on an
interactive matplotlib backend, with the computation on a worker thread and the
main thread servicing the GUI event loop; the outputs are identical with or
without it. A non-interactive backend, and WebAgg, are refused by name.
"""

import argparse
import cmath
import itertools
import json
import math
import os
import sys
import time

import numpy as np

import tessera as T
from tessera import cobordism as cob
from tessera import observables as obs

#: The target: the proton mass over the Delta(1232) Breit-Wigner mass (PDG).
PROTON_MASS_MEV = 938.272
DELTA_MASS_MEV = 1232.0
TARGET_MASS_RATIO = PROTON_MASS_MEV / DELTA_MASS_MEV
TARGET_MASS_SQUARED_RATIO = TARGET_MASS_RATIO ** 2

#: The declared scan: kappa = 8 pi G (lattice units) and beta. The paper fixes
#: neither (WP §7); the grid spans its weak-coupling regime, the runaway scale
#: of about two it names, and the beta below which it reports the induced
#: stiffness dominating the bare one.
DECLARED_KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0)
DECLARED_BETAS = (0.5, 1.0, 2.0, 5.0)

#: The declared holonomy term: the Villain form, the paper's holonomy term.
DECLARED_HOLONOMY = "villain"
#: The forms --holonomy accepts, and the library's name for each.
HOLONOMY_FORMS = {"villain": cob.HolonomyForm.Villain,
                  "wilson": cob.HolonomyForm.Wilson}

#: Which fluctuations the Section 7 quartic eliminates: the squared lengths and
#: the link phases (the pure-gauge phase directions, the null space of A, are
#: projected out by the Drazin inverse), or the squared lengths alone.
DECLARED_ELIMINATION = "lengths-and-phases"
ELIMINATIONS = ("lengths-and-phases", "lengths")
#: The resonance disc of the Drazin inverse, relative to the spectral radius of
#: A: an eigenvalue of A inside it is a zero-stiffness direction.
DECLARED_GAUGE_RESONANCE_RADIUS = 1e-10
#: The Cauchy rule for the diamagnetic term along a pure-gauge direction.
DECLARED_WARD_CONTOUR_RADIUS = 0.1
DECLARED_WARD_CONTOUR_NODES = 8
#: The relative distance to a zero of the Villain weight W that no face
#: holonomy may come within along a Newton step (step control only).
DECLARED_HOLONOMY_ZERO_MARGIN = 0.05

#: The squared edge length of the regular tetrahedron (the paper's a^2 = 8).
DECLARED_EDGE_SQUARED = 8.0
#: The unit Dirac monopole.
DECLARED_MONOPOLE = 1
#: Sheets: the colour multiplicity.
SHEETS = 3
#: Base cells (edges) of one tetrahedron.
BASE_EDGES = 6
#: Relative separation at or below which ordered eigenvalues form one band.
DECLARED_BAND_TOLERANCE = 1e-8
#: Tolerances of the certificates this driver grades.
DECLARED_CERTIFICATE_TOLERANCE = 1e-8

#: How often the --live main thread services the GUI event loop.
LIVE_POLL_INTERVAL = 0.05

SPIN_HALF = 0.75
SPIN_THREE_HALVES = 3.75


# ---------------------------------------------------------------- the host


def monopole_support():
    """The library's symmetric unit-monopole tetrahedron (the fixture the
    whitepaper's tetrahedral statements are made about)."""
    return obs.MonopoleSupport.tetrahedron(DECLARED_MONOPOLE)


def build_host(edge_squared=DECLARED_EDGE_SQUARED, cell=None):
    """The three-sheeted host: three disjoint regular tetrahedra, vertices
    4 t .. 4 t + 3 on sheet t, each carrying the monopole connection of
    `MonopoleSupport.tetrahedron(1)` on corresponding edges.

    With ``cell``, a mapping with the six ``squared_lengths`` and the six
    ``links`` U_e of one tetrahedron on the ascending orientation of its edges
    in the order of `MonopoleSupport.edges`, every sheet carries that
    tetrahedron's data instead: this is how `tessera.drivers.recursion` reads
    a cell of a level as a three-sheeted host.

    The mesh stores the phase phi of U = exp(i phi) on each edge's stored
    orientation. The declared connection values have unit modulus, so their
    phases are their arguments; that is the declaration of the host's data, not
    a readout, and the monopole number is read back from the face holonomies of
    the built host (`MonopoleSupport.monopoleNumber`).
    """
    cells = [[4 * t + k for k in range(4)] for t in range(SHEETS)]
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    support = monopole_support()
    length = cmath.sqrt(complex(edge_squared))
    pairs = [tuple(e) for e in support.edges]
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        sheet = source // 4
        if cell is None:
            link = support.transport(source - 4 * sheet, target - 4 * sheet)
            edge.setLength(length)
            edge.setPhase(complex(cmath.phase(link)))
            continue
        a, b = source - 4 * sheet, target - 4 * sheet
        m = pairs.index((min(a, b), max(a, b)))
        link = complex(cell["links"][m])
        if a > b:
            link = 1.0 / link
        edge.setLength(cmath.sqrt(complex(cell["squared_lengths"][m])))
        # U = exp(i phi); the principal logarithm is a coordinate on the
        # stored field and returns U exactly
        edge.setPhase(complex(-1j * cmath.log(link)))
    return spacetime


def edge_records(spacetime):
    """(source, target) vertex ids of every edge in `getEdgeList()` order."""
    return [(int(edge.getSource().getId()), int(edge.getTarget().getId()))
            for edge in spacetime.getEdgeList().toVector()]


def canonical_edges():
    """The canonical degree-one cells of the host, ascending vertex pairs in
    the chain complex's order: sheet-major, and within a sheet the pairs of
    `MonopoleSupport` in the same order."""
    pairs = []
    for t in range(SHEETS):
        for a, b in itertools.combinations(range(4), 2):
            pairs.append((4 * t + a, 4 * t + b))
    return pairs


def sheet_links(spacetime, sheet):
    """U_e on the ascending orientation of each edge of one sheet, in the
    order of `MonopoleSupport.edges`."""
    stored = {}
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        link = cmath.exp(1j * complex(edge.getPhase()))
        stored[(source, target)] = link
        stored[(target, source)] = 1.0 / link
    return [stored[(4 * sheet + a, 4 * sheet + b)]
            for a, b in itertools.combinations(range(4), 2)]


def sheet_squared_lengths(spacetime, sheet):
    """z_e of one sheet's edges in the same order."""
    stored = {}
    for edge in spacetime.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        z = complex(edge.getLength()) ** 2
        stored[(source, target)] = stored[(target, source)] = z
    return [stored[(4 * sheet + a, 4 * sheet + b)]
            for a, b in itertools.combinations(range(4), 2)]


# ---------------------------------------------------------------- the action


def action_declaration(spacetime, kappa, beta, regge_hinges="interior",
                       matter_weight=1.0, reference_lengths=None,
                       holonomy=DECLARED_HOLONOMY):
    """The joint action of the calculation (WP §3, §7), with the holonomy term
    in the declared form, ``"villain"`` or ``"wilson"``."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = 1.0 / kappa
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = (cob.ReggeHinges.Interior
                                if regge_hinges == "interior"
                                else cob.ReggeHinges.All)
    declaration.stiffness_weight = 1.0 / kappa
    declaration.reference_lengths = (
        list(reference_lengths) if reference_lengths is not None else
        [complex(edge.getLength())
         for edge in spacetime.getEdgeList().toVector()])
    declaration.holonomy_weight = beta
    declaration.holonomy_form = HOLONOMY_FORMS[holonomy]
    declaration.matter_weight = matter_weight
    return declaration


def relaxation_declaration(config):
    """The inner holomorphic Newton solve."""
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = True
    geometry.relax_multipliers = False
    geometry.maximum_iterations = config["newton_iterations"]
    geometry.tolerance = config["newton_tolerance"]
    geometry.jacobian_mode = cob.HolomorphicJacobianMode.RealAxisDifference
    geometry.contour_radius = config["jacobian_radius"]
    geometry.holonomy_zero_margin = config["holonomy_zero_margin"]
    return geometry


def mean_field_declaration(content, config, band_symmetry=None):
    """Band filling with the content's occupations (WP §7 line 250), the bands
    read on the T-averaged operator when the rotation action is supplied
    (WP §11.1 line 497)."""
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    if band_symmetry is not None:
        declaration.band_symmetry = [list(np.asarray(d).reshape(-1))
                                     for d in band_symmetry]
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = [float(n) for n in content]
    declaration.band_tolerance = config["band_tolerance"]
    declaration.occupation_order = cob.OccupationOrder.AscendingRealPart
    declaration.maximum_iterations = config["mean_field_iterations"]
    declaration.tolerance = config["mean_field_tolerance"]
    declaration.geometry = relaxation_declaration(config)
    return declaration


def matrix(flat):
    """A flat row-major square matrix as a numpy array."""
    flat = np.asarray(flat, dtype=complex)
    order = int(round(math.sqrt(flat.size)))
    return flat.reshape(order, order)


# ------------------------------------------------------- symmetry and spin


def rotation_group():
    """The twelve rotations of the tetrahedron, T = A_4."""
    return obs.MonopoleSupport.tetrahedralRotations()


def rotation_action(supports):
    """D_1(g) on the 18 microscopic edge cells, one 6 x 6 block per sheet from
    that sheet's `MonopoleSupport.edgeRepresentation`, for the twelve
    rotations of `rotation_group()`. The canonical cell order is sheet-major
    and each sheet's edges are in the order of `MonopoleSupport.edges`."""
    actions = []
    for g in rotation_group():
        d = np.zeros((SHEETS * BASE_EDGES, SHEETS * BASE_EDGES),
                     dtype=complex)
        for t, support in enumerate(supports):
            sl = slice(t * BASE_EDGES, (t + 1) * BASE_EDGES)
            d[sl, sl] = np.asarray(support.edgeRepresentation(g))
        actions.append(d)
    return actions


def rotation_averaged(operator, actions):
    """The T-averaged operator (1/|T|) sum_g D_1(g)^{-1} X D_1(g): the
    operator the whitepaper reads the spin content on ("the T-averaged
    twisted edge Laplacian", WP §11.1 line 497)."""
    return sum(np.linalg.solve(d, operator @ d) for d in actions) / len(actions)


def sheet_support(spacetime, sheet):
    """The `MonopoleSupport` of one sheet of the relaxed host, with its faces
    in the outward orientation of the library fixture. `MonopoleSupport`
    refuses a connection off the unit circle; the U(1) part is taken
    explicitly (`MonopoleSupport.u1Part`) and the departure is reported."""
    fixture = monopole_support()
    links = sheet_links(spacetime, sheet)
    departure = max(abs(abs(u) - 1.0) for u in links)
    support = obs.MonopoleSupport(4, fixture.edges, fixture.faces,
                                  obs.MonopoleSupport.u1Part(links))
    return support, departure


def aligned_doublet_frame(support, group):
    """The edge basis in which the three doublets 2, 2', 2'' carry one common
    SU(2) action, each up to its own Z_3 character.

    The three doublets are the eigenspaces of the rotation-averaged edge
    operator (`MonopoleSupport.rotationAveragedEdgeOperator`); each appears
    once in the edge cochains, so they are the isotypic components of the
    projective action D_1(g) and do not depend on the operator averaged. With
    R(g) the action on the reference doublet (the coexact j = 1/2 doublet
    `spinRead` names) and M_d(g) the action on doublet d, the character is
    chi_d(g) = tr M_d(g) / tr R(g) where tr R(g) != 0 and one otherwise (the
    Klein four-group, which is the kernel of the Z_3 character), and the
    intertwiner T_d = sum_g chi_d(g)^{-1} M_d(g) X R(g)^{-1} carries R to
    chi_d^{-1} M_d (Schur averaging from a fixed seed X). Columns 2 c + s of
    the returned frame are spin state s of carrier c, the order
    `SharpSpin.doubletSpinMatrices` uses. The carriers are ordered by the
    eigenvalue of the averaged operator, ascending, which is the reading the
    `spinRead` bands are listed in.
    """
    read = support.spinRead(group)
    averaged = support.rotationAveragedEdgeOperator(support.edgeLaplacian(),
                                                    group)
    values, vectors = np.linalg.eigh(np.asarray(averaged))
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    blocks = [vectors[:, 2 * c:2 * c + 2] for c in range(3)]
    representations = [np.asarray(support.edgeRepresentation(g))
                       for g in group]
    reference = int(read.doublet_index)
    if reference >= 3:
        raise RuntimeError("the support carries no j = 1/2 doublet: %s"
                           % read.certificate.describe())
    basis_r = blocks[reference]
    actions = {c: [blocks[c].conj().T @ d @ blocks[c]
                   for d in representations] for c in range(3)}
    seed = np.array([[0.7, 0.2 + 0.1j], [-0.3j, 1.1]])
    frame = np.zeros((BASE_EDGES, BASE_EDGES), dtype=complex)
    residual = 0.0
    for c in range(3):
        if c == reference:
            intertwiner = np.eye(2, dtype=complex)
            characters = [1.0] * len(group)
        else:
            characters = []
            for m, r in zip(actions[c], actions[reference]):
                tr_r = np.trace(r)
                characters.append(np.trace(m) / tr_r
                                  if abs(tr_r) > 1e-9 else 1.0)
            intertwiner = sum(
                (1.0 / chi) * m @ seed @ np.linalg.inv(r)
                for chi, m, r in zip(characters, actions[c],
                                     actions[reference]))
            intertwiner = intertwiner / np.sqrt(np.linalg.det(intertwiner))
        for chi, m, r in zip(characters, actions[c], actions[reference]):
            residual = max(residual, np.max(np.abs(
                m @ intertwiner - chi * intertwiner @ r)))
        frame[:, 2 * c:2 * c + 2] = blocks[c] @ intertwiner
    # The Z_3 = 2T/Q_8 label of each carrier: its character on a declared
    # element of order three, the first three-cycle of `rotation_group()`,
    # read as exp(2 pi i t / 3). Which of the two nontrivial characters is
    # called 2' is the sheet-rotation convention of WP §8 (omega versus
    # omega^2); the label is recorded, not derived.
    three_cycle = next(k for k, g in enumerate(group)
                       if sum(1 for x, y in enumerate(g) if x != y) == 3)
    trialities = []
    for c in range(3):
        if c == reference:
            trialities.append(0)
            continue
        r = actions[reference][three_cycle]
        chi = np.trace(actions[c][three_cycle]) / np.trace(r)
        trialities.append(int(round(np.angle(chi) / (2 * np.pi / 3))) % 3)
    return {
        "frame": frame,
        "trialities": trialities,
        "three_cycle": list(group[three_cycle]),
        "averaged_eigenvalues": [float(v) for v in values],
        "reference_carrier": reference,
        "intertwining_residual": float(residual),
        "spin_read": read,
    }


def edge_spin_matrices():
    """J_a on the 18 microscopic-frame modes: the three doublets as three
    spin-1/2 carriers (`SharpSpin.doubletSpinMatrices(3)`), times the identity
    on the sheets, in the mode order b * 3 + t (base mode b, sheet t)."""
    base = obs.SharpSpin.doubletSpinMatrices(3)
    return [np.kron(np.asarray(j), np.eye(SHEETS)) for j in base]


def sheet_generators():
    """The eight Gell-Mann generators lambda_a / 2 of the sheet space, lifted to
    the 18 modes as I_6 (x) lambda_a / 2 in the mode order b * 3 + t."""
    lam = np.zeros((8, 3, 3), dtype=complex)
    lam[0][0, 1] = lam[0][1, 0] = 1
    lam[1][0, 1], lam[1][1, 0] = -1j, 1j
    lam[2][0, 0], lam[2][1, 1] = 1, -1
    lam[3][0, 2] = lam[3][2, 0] = 1
    lam[4][0, 2], lam[4][2, 0] = -1j, 1j
    lam[5][1, 2] = lam[5][2, 1] = 1
    lam[6][1, 2], lam[6][2, 1] = -1j, 1j
    lam[7] = np.diag([1, 1, -2]) / math.sqrt(3)
    return [np.kron(np.eye(BASE_EDGES), l / 2.0) for l in lam]


def colour_casimir(state):
    """The quadratic Casimir sum_a dGamma(lambda_a / 2)^2 of the sheet algebra
    applied to a Fock vector, as three `SharpSpin.applyTotalSpinSquared` calls.
    A colour singlet is annihilated by it."""
    generators = sheet_generators()
    zero = np.zeros_like(generators[0])
    groups = [generators[0:3], generators[3:6], [generators[6],
                                                   generators[7], zero]]
    out = np.zeros_like(state)
    for group in groups:
        out = out + np.asarray(obs.SharpSpin.applyTotalSpinSquared(group,
                                                                   state))
    return out


# ------------------------------------------------- the three-particle space

_FOCK_INDEX = {}


def fock_position(modes, mode_count=SHEETS * BASE_EDGES):
    """The Fock index and sign of the wedge of the ascending modes, read off
    `SharpSpin.determinant` once per tuple and cached."""
    key = (tuple(modes), mode_count)
    if key not in _FOCK_INDEX:
        vector = np.asarray(obs.SharpSpin.determinant(list(modes),
                                                      mode_count))
        index = int(np.flatnonzero(vector)[0])
        _FOCK_INDEX[key] = (index, complex(vector[index]))
    return _FOCK_INDEX[key]


def occupation_basis(mode_count=SHEETS * BASE_EDGES, particles=3):
    """The ascending tuples in lexicographic order: the basis of
    `ManyBodySpaceRead`."""
    return [tuple(c) for c in itertools.combinations(range(mode_count),
                                                     particles)]


def to_fock(coefficients, basis, mode_count=SHEETS * BASE_EDGES):
    """A vector over the occupation basis as a Fock vector."""
    out = np.zeros(1 << mode_count, dtype=complex)
    for amplitude, modes in zip(coefficients, basis):
        if amplitude != 0:
            index, sign = fock_position(modes, mode_count)
            out[index] += amplitude * sign
    return out


def singlet_states(content):
    """The colour-singlet, one-quark-per-sheet states of a content, as
    vectors over the occupation basis.

    Base mode b = 2 c + s (carrier c in band order, spin s). A symmetric base
    state of three quarks is a multiset {b1, b2, b3}; its colour singlet is
    sum over the distinct orderings of c+_{b1, sheet 0} c+_{b2, sheet 1}
    c+_{b3, sheet 2} |0>, with mode index b * 3 + t. The carrier counts of the
    multiset must equal the content.
    """
    basis = occupation_basis()
    position = {modes: k for k, modes in enumerate(basis)}
    states, labels = [], []
    carriers = [c for c, n in enumerate(content) for _ in range(n)]
    seen = set()
    for spins in itertools.product((0, 1), repeat=3):
        multiset = tuple(sorted(2 * c + s for c, s in zip(carriers, spins)))
        if multiset in seen:
            continue
        seen.add(multiset)
        vector = np.zeros(len(basis), dtype=complex)
        for ordering in set(itertools.permutations(multiset)):
            modes = [b * SHEETS + t for t, b in enumerate(ordering)]
            ascending = sorted(modes)
            # the wedge in slot order equals the ascending wedge times the
            # sign of the sorting permutation
            permutation = [modes.index(m) for m in ascending]
            vector[position[tuple(ascending)]] += _permutation_sign(
                permutation)
        states.append(vector)
        labels.append(multiset)
    return np.column_stack(states), labels


def _permutation_sign(permutation):
    sign, seen = 1, [False] * len(permutation)
    for start in range(len(permutation)):
        if seen[start]:
            continue
        length, k = 0, start
        while not seen[k]:
            seen[k] = True
            k = permutation[k]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def left_inverse(columns):
    """The bilinear left inverse (V^T V)^{-1} V^T of a full-column-rank set of
    coefficient vectors, the dual basis in the transpose pairing."""
    return np.linalg.solve(columns.T @ columns, columns.T)


def spin_sectors(states):
    """Split a colour-singlet space by total spin: J^2 applied to each basis
    state with `SharpSpin.applyTotalSpinSquared`, the block in the basis, and
    its eigenvectors grouped by eigenvalue (3/4 or 15/4)."""
    basis = occupation_basis()
    spins = edge_spin_matrices()
    images = []
    for k in range(states.shape[1]):
        fock = to_fock(states[:, k], basis)
        image = np.asarray(obs.SharpSpin.applyTotalSpinSquared(spins, fock))
        images.append(image)
    # J^2 is real symmetric in the occupation basis, so the block is read by
    # the bilinear pairing on the Fock vectors.
    focks = np.column_stack([to_fock(states[:, k], basis)
                             for k in range(states.shape[1])])
    gram = focks.T @ focks
    block = np.linalg.solve(gram, focks.T @ np.column_stack(images))
    values, vectors = np.linalg.eig(block)
    sectors = {}
    for j2 in (SPIN_HALF, SPIN_THREE_HALVES):
        picked = [k for k in range(len(values))
                  if abs(values[k] - j2) < 1e-6]
        if picked:
            sectors[j2] = states @ vectors[:, picked]
    return sectors, [complex(v) for v in values]


# -------------------------------------------------------------- the solve


def _geometric_action(spacetime, kappa, beta, config):
    """The geometric part of the action (matter off), with the host's declared
    reference lengths rather than the relaxed ones."""
    return cob.JointAction(
        spacetime, action_declaration(spacetime, kappa, beta,
                                      config["regge_hinges"],
                                      matter_weight=0.0,
                                      reference_lengths=config[
                                          "reference_lengths"],
                                      holonomy=config["holonomy"]))


def fluctuation_couplings(spacetime, phases):
    """O_a = dh_1/df_a for the retained fluctuations, in the order: the 18
    squared lengths z_e, then (when ``phases``) the 18 link phases phi_e of
    U_e = exp(i phi_e) on each edge's stored orientation, both in
    `getEdgeList()` order. `HodgeLaplacian.laplacianPhaseGradient` is the
    derivative in the canonical (ascending) phase, so the stored one carries
    the orientation sign."""
    hodge = cob.HodgeLaplacian(spacetime,
                               cob.HodgeLaplacian.defaultWeightConvention(),
                               cob.HodgeMetricSource.WhitneyPencil)
    records = edge_records(spacetime)
    couplings = [matrix(hodge.laplacianGradient(1, a, b)) for a, b in records]
    if phases:
        couplings += [(1.0 if a < b else -1.0)
                      * matrix(hodge.laplacianPhaseGradient(1, a, b))
                      for a, b in records]
    return couplings


def gauge_directions(spacetime, phases):
    """A basis of the pure-gauge directions in the fluctuation coordinates,
    one per vertex but the last of each sheet: no length component, and delta phi_xy = chi_y - chi_x on each
    stored edge (x, y), with chi the indicator of the vertex. None when the
    phases are not retained."""
    if not phases:
        return None
    records = edge_records(spacetime)
    # one vertex of each sheet is left out: the indicators of a whole sheet
    # sum to the constant, which moves no link, so the rest are a basis
    vertices = sorted({v for pair in records for v in pair})
    vertices = [v for v in vertices if v % 4 != 3]
    edges = len(records)
    directions = np.zeros((2 * edges, len(vertices)), dtype=complex)
    for column, vertex in enumerate(vertices):
        for row, (x, y) in enumerate(records):
            directions[edges + row, column] = ((1.0 if y == vertex else 0.0)
                                               - (1.0 if x == vertex else 0.0))
    return directions


def bare_stiffness(spacetime, kappa, beta, config, phases):
    """A, the Hessian of the geometric part of the action in the retained
    fluctuation coordinates (z_e; phi_e), and the checks on how it was read.

    The length block is `HolomorphicRelaxation.jacobian` of the geometric
    action with the matter term off. With the phases retained the relaxation
    also relaxes the links, whose coordinate is the Maurer-Cartan increment
    delta = i d(phi), so the cross blocks are i times the Jacobian's and the
    phase block is minus its link block. The phase block is taken from the
    exact analytic `JointAction.holonomy_hessian` (negated for the same
    reason), whose pure-gauge null space is exact to rounding; the Jacobian's
    link block is kept only as a check on it."""
    geometric = _geometric_action(spacetime, kappa, beta, config)
    declaration = relaxation_declaration(config)
    declaration.relax_links = phases
    solve = cob.HolomorphicRelaxation(geometric, declaration)
    count = solve.variable_count()
    jacobian = np.asarray(solve.jacobian()).reshape(solve.equation_count(),
                                                    count)
    n = len(spacetime.getEdgeList().toVector())
    if not phases:
        return jacobian, {}
    analytic = -np.asarray(geometric.holonomy_hessian()).reshape(n, n)
    stiffness = np.zeros((2 * n, 2 * n), dtype=complex)
    stiffness[:n, :n] = jacobian[:n, :n]
    stiffness[:n, n:] = 1j * jacobian[:n, n:]
    stiffness[n:, :n] = 1j * jacobian[n:, :n]
    stiffness[n:, n:] = analytic
    scale = max(np.abs(stiffness).max(), 1.0)
    return stiffness, {
        "cross_block_norm": float(np.linalg.norm(stiffness[:n, n:])),
        "cross_block_asymmetry": float(
            np.linalg.norm(stiffness[:n, n:] - stiffness[n:, :n].T) / scale),
        "phase_block_jacobian_departure": float(
            np.abs(-jacobian[n:, n:] - analytic).max() / scale),
    }


def drazin_elimination(stiffness, directions, radius):
    """The generalized inverse the elimination uses: the Drazin inverse A^D of
    A at zero, with the Riesz projector Pi_0 onto the generalized null space
    (`chainhodge.PencilSchur.feshbach` with every coordinate interior and the
    resonance disc of the declared relative radius about zero). A^D inverts A
    on ran(I - Pi_0) and is zero on ran(Pi_0), so the zero-stiffness
    pure-gauge directions contribute nothing and every other direction is
    integrated out.

    A is complex symmetric, so Pi_0^T = Pi_0 and ran(I - Pi_0) is orthogonal
    to ran(Pi_0) in the transpose pairing. With R a basis of ran(I - Pi_0),
    A^D = R (R^T A R)^{-1} R^T exactly: the fluctuation is carried in the
    reduced coordinates f = R g, whose stiffness R^T A R is nonsingular, and
    `DressedFluctuation` eliminates g. The identity is measured, not assumed.
    """
    size = stiffness.shape[0]
    read = T.chainhodge.PencilSchur.feshbach(
        stiffness, np.zeros_like(stiffness), 0j, [], 1e-12, radius)
    record = {"coordinates": int(size),
              "resonance_radius": float(read.resonanceRadius),
              "resonance_enclosure": float(read.resonanceEnclosure),
              "resonance_separation": float(read.resonanceSeparation)}
    if not read.interiorSingular:
        drazin = np.linalg.inv(stiffness)
        basis = np.eye(size, dtype=complex)
        null = np.zeros_like(stiffness)
    else:
        drazin = np.asarray(read.interiorInverse)
        null = np.asarray(read.nullProjector)
        left, _, _ = np.linalg.svd(np.asarray(read.rangeProjector))
        basis = left[:, :int(read.interiorRank)]
    reduced = basis.T @ stiffness @ basis
    rebuilt = basis @ np.linalg.solve(reduced, basis.T)
    record.update({
        "null_dimension": int(size - basis.shape[1]),
        "eliminated_dimension": int(basis.shape[1]),
        "projector_idempotency": float(
            np.linalg.norm(null @ null - null) / np.linalg.norm(null))
        if null.any() else 0.0,
        "drazin_identity_residual": float(
            np.linalg.norm(stiffness @ drazin @ stiffness - stiffness)
            / np.linalg.norm(stiffness)),
        "reduction_residual": float(np.linalg.norm(rebuilt - drazin)
                                    / np.linalg.norm(drazin)),
        "reduced_conditioning": float(np.linalg.cond(reduced)),
    })
    if directions is not None:
        # the pure-gauge directions lie in ran(Pi_0), and Pi_0 has no more
        # rank than they span
        gauge_rank = int(np.linalg.matrix_rank(directions))
        record["gauge_dimension"] = gauge_rank
        record["gauge_projector_residual"] = float(
            np.linalg.norm(null @ directions - directions)
            / np.linalg.norm(directions))
        record["null_space_is_pure_gauge"] = bool(
            record["null_dimension"] == gauge_rank
            and record["gauge_projector_residual"] < 1e-8)
    return drazin, basis, reduced, record


def occupied_projector(carrier, occupied):
    """The Riesz projector of the carrier onto its ``occupied`` modes of
    smallest real part (`OccupationOrder.AscendingRealPart`)."""
    values, vectors = np.linalg.eig(carrier)
    order = sorted(range(len(values)),
                   key=lambda k: (values[k].real, values[k].imag))[:occupied]
    return vectors[:, order] @ np.linalg.inv(vectors)[order, :]


def ward_read(spacetime, carrier, couplings, directions, config):
    """The Ward identity of the retained fluctuations: (D - Pi(0)) g = 0 on
    every pure-gauge direction g.

    Pi(0) is `DressedFluctuation.paramagnetic(0)` of the carrier with the
    declared couplings. D g is the diamagnetic term along g,
    (D g)_a = tr(P_occ d_g O_a), with P_occ the occupied Riesz projector and
    d_g O_a the derivative of the coupling O_a along the pure-gauge direction,
    formed by the Cauchy rule on a circle of the declared radius in the
    complex gauge parameter: the direction is a gauge transformation for
    every complex value of the parameter, so O_a is entire along it and the
    rule converges geometrically in the node count. The residual is
    ||(D - Pi(0)) g|| / (||Pi(0)|| ||g||), maximized over the directions, and
    ``paramagnetic_alone`` is the largest same ratio for Pi(0) g by itself,
    the size the identity cancels. The
    geometry is restored exactly afterwards."""
    if directions is None:
        return {"directions": 0}
    n = carrier.shape[0]
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = n
    declaration.carrier = list(carrier.reshape(-1))
    declaration.couplings = [list(o.reshape(-1)) for o in couplings]
    declaration.occupied_modes = 3
    try:
        paramagnetic = np.asarray(cob.DressedFluctuation(
            declaration).paramagnetic(0j)).reshape(len(couplings),
                                                   len(couplings))
    except ValueError as error:
        return {"directions": int(directions.shape[1]),
                "unmeasured": str(error)}
    projector = occupied_projector(carrier, 3)
    edges = spacetime.getEdgeList().toVector()
    saved = [complex(edge.getPhase()) for edge in edges]
    radius = config["ward_contour_radius"]
    nodes = config["ward_contour_nodes"]
    residuals, paramagnetic_parts = [], []
    try:
        for column in range(directions.shape[1]):
            g = directions[:, column]
            shift = g[len(edges):]
            derivative = [np.zeros((n, n), dtype=complex) for _ in couplings]
            for k in range(nodes):
                root = cmath.exp(2j * math.pi * k / nodes)
                for edge, phase, step in zip(edges, saved, shift):
                    edge.setPhase(phase + radius * root * step)
                weight = root.conjugate() / (nodes * radius)
                for a, o in enumerate(fluctuation_couplings(spacetime, True)):
                    derivative[a] += weight * o
            for edge, phase in zip(edges, saved):
                edge.setPhase(phase)
            diamagnetic = np.array([np.sum(projector * d.T)
                                    for d in derivative])
            polarization = paramagnetic @ g
            scale = np.linalg.norm(paramagnetic) * np.linalg.norm(g)
            residuals.append(float(np.linalg.norm(diamagnetic - polarization)
                                   / scale))
            paramagnetic_parts.append(float(np.linalg.norm(polarization)
                                            / scale))
    finally:
        for edge, phase in zip(edges, saved):
            edge.setPhase(phase)
    return {"directions": int(directions.shape[1]),
            "contour_radius": radius, "contour_nodes": nodes,
            "residual": max(residuals),
            "paramagnetic_alone": max(paramagnetic_parts)}


def truncation_certificates(spacetime, kappa, beta, config, couplings,
                            stiffness, induced):
    """The two remainders the exact elimination sets aside (WP §7), measured
    along the induced displacement f* = -A^D<J> in every retained coordinate
    (the squared lengths, and the link phases when retained): the cubic
    remainder of the geometric action relative to its quadratic term, and the
    second-order remainder of h_1 relative to its linear term. The geometry is
    restored exactly afterwards. The displaced action value needs log W of the
    Villain term; a displacement that reaches a zero of W has no value there,
    and the action remainder is then reported as unmeasured with the reason."""
    edges = spacetime.getEdgeList().toVector()
    n = len(edges)
    saved_lengths = [complex(edge.getLength()) for edge in edges]
    saved_phases = [complex(edge.getPhase()) for edge in edges]
    phases = len(induced) == 2 * n

    before = _geometric_action(spacetime, kappa, beta, config)
    s0 = complex(before.value())
    gradient = np.asarray(before.length_stationarity())
    if phases:
        gradient = np.concatenate(
            [gradient, 1j * np.asarray(before.link_stationarity())])
    h0 = matrix(before.carrier_operator())
    s1, unmeasured = None, None
    try:
        for edge, length, df in zip(edges, saved_lengths, induced[:n]):
            edge.setLength(cmath.sqrt(length * length + df))
        if phases:
            for edge, phase, dp in zip(edges, saved_phases, induced[n:]):
                edge.setPhase(phase + dp)
        after = _geometric_action(spacetime, kappa, beta, config)
        h1 = matrix(after.carrier_operator())
        try:
            s1 = complex(after.value())
        except ValueError as error:
            unmeasured = str(error)
    finally:
        for edge, length, phase in zip(edges, saved_lengths, saved_phases):
            edge.setLength(length)
            edge.setPhase(phase)
    quadratic = 0.5 * induced @ stiffness @ induced
    linear_h = sum(f * o for f, o in zip(induced, couplings))
    out = {
        "induced_displacement_norm": float(np.linalg.norm(induced)),
        "induced_length_norm": float(np.linalg.norm(induced[:n])),
        "induced_phase_norm": float(np.linalg.norm(induced[n:]))
        if phases else 0.0,
        "action_quadratic_term": complex(quadratic),
        "operator_relative_remainder": float(
            np.linalg.norm(h1 - h0 - linear_h) / np.linalg.norm(linear_h))
        if np.linalg.norm(linear_h) > 0 else None,
    }
    if s1 is None:
        out.update({"action_cubic_remainder": None,
                    "action_relative_remainder": None,
                    "action_unmeasured": unmeasured})
    else:
        cubic = s1 - s0 - gradient @ induced - quadratic
        out.update({"action_cubic_remainder": complex(cubic),
                    "action_relative_remainder":
                        float(abs(cubic) / abs(quadratic))
                        if quadratic != 0 else None})
    return out


def second_quantized(one_particle, basis=None):
    """dGamma(X) on the three-particle space, in the occupation basis: the
    linear coefficient of the exact cubic polynomial
    t -> Lambda^3(I + t X) = I + t dGamma(X) + t^2 (...) + t^3 (...), read off
    by the combination (8 f(1/2) - 8 f(-1/2) - f(1) + f(-1)) / 6."""
    identity = np.eye(one_particle.shape[0], dtype=complex)

    def f(t):
        return third_exterior_power(identity + t * one_particle, basis)
    return (8 * f(0.5) - 8 * f(-0.5) - f(1.0) + f(-1.0)) / 6.0


def dense_quartic_reference(couplings, drazin, frame, dual):
    """-1/2 sum_ab (A^D)_ab J_a J_b on the three-particle space of the fiber,
    assembled densely from A^D itself rather than through the reduced
    coordinates, with J_a = dGamma(Phi~^T O_a Phi). The sum over b is taken
    inside dGamma, which is linear, so only two dense three-particle matrices
    are held at a time."""
    in_frame = [dual @ o @ frame for o in couplings]
    total = None
    for a, o in enumerate(in_frame):
        weighted = sum(drazin[a, b] * in_frame[b]
                       for b in range(len(in_frame)))
        term = second_quantized(o) @ second_quantized(weighted)
        total = term if total is None else total + term
    return -0.5 * total


def eliminate_fluctuations(spacetime, action, carrier, kappa, beta, config):
    """Everything the exact elimination of Section 7 needs at the relaxed
    point, and the record of what was eliminated and how it was certified.

    The retained fluctuations are the 18 squared lengths and, under
    ``lengths-and-phases``, the 18 link phases; A is the bare stiffness of the
    geometric action in them, <J>_a = tr(Gamma O_a) the carried force, and the
    elimination about the self-consistent point is
    -1/2 (J - <J>)^T A^D (J - <J>): a one-body shift sum_a (A^D <J>)_a O_a, a
    constant -1/2 <J>^T A^D <J>, and the quartic -1/2 J^T A^D J, the last
    through the reduced coordinates of `drazin_elimination`."""
    phases = config["elimination"] == "lengths-and-phases"
    couplings = fluctuation_couplings(spacetime, phases)
    stiffness, stiffness_checks = bare_stiffness(spacetime, kappa, beta,
                                                 config, phases)
    directions = gauge_directions(spacetime, phases)
    drazin, basis, reduced, drazin_record = drazin_elimination(
        stiffness, directions, config["gauge_resonance_radius"])
    covariance = matrix(action.declaration.covariance)
    expectation = np.array([np.sum(covariance * o.T) for o in couplings])
    n = len(spacetime.getEdgeList().toVector())
    force_check = float(np.linalg.norm(
        expectation[:n] - np.asarray(action.hellmann_feynman_length_force())))
    if phases:
        force_check = max(force_check, float(np.linalg.norm(
            expectation[n:]
            - 1j * np.asarray(action.hellmann_feynman_link_force()))))
    induced = -drazin @ expectation
    record = {
        "rule": config["elimination"],
        "retained_coordinates": (
            "18 squared lengths z_e, then 18 link phases phi_e on the stored "
            "orientations, getEdgeList order" if phases
            else "18 squared lengths z_e in getEdgeList order; links held"),
        "generalized_inverse": (
            "Drazin inverse of A at zero with the Riesz projector onto its "
            "null space (PencilSchur.feshbach), applied through the reduced "
            "coordinates f = R g, R a basis of ran(I - Pi_0)"),
        "stiffness_checks": stiffness_checks,
        "drazin": drazin_record,
        "expectation_force_check": force_check,
        "ward_identity": ward_read(spacetime, carrier, couplings, directions,
                                   config),
    }
    return {
        "couplings": couplings,
        "stiffness": stiffness,
        "drazin": drazin,
        "reduced_stiffness": reduced,
        "reduced_couplings": [sum(basis[a, k] * couplings[a]
                                  for a in range(len(couplings)))
                              for k in range(basis.shape[1])],
        "induced": induced,
        "shift": sum(-f * o for f, o in zip(induced, couplings)),
        "constant": -0.5 * expectation @ drazin @ expectation,
        "truncation": truncation_certificates(spacetime, kappa, beta, config,
                                              couplings, stiffness, induced),
        "record": record,
    }


def contour_of(block):
    """The declared pole contour of a sector: centre tr(H)/dim, radius 1.5 times
    the Gershgorin radius about that centre plus a floor of 1e-3 times its
    scale, so every eigenvalue is enclosed by Gershgorin's theorem."""
    dimension = block.shape[0]
    centre = np.trace(block) / dimension
    radius = 0.0
    for i in range(dimension):
        off = np.sum(np.abs(block[i])) - abs(block[i, i])
        radius = max(radius, abs(block[i, i] - centre) + off)
    radius = 1.5 * radius + 1e-3 * max(1.0, abs(centre))
    return complex(centre), float(radius)


def sector_poles(operator, sector_states):
    """The compressed operator of a sector, its leakage, and its poles."""
    dual = left_inverse(sector_states)
    image = operator @ sector_states
    block = dual @ image
    leakage = np.linalg.norm(image - sector_states @ block) / max(
        np.linalg.norm(image), 1e-300)
    centre, radius = contour_of(block)
    config = cob.BoundStatePoleConfig()
    read = cob.BoundStatePole.poles(block, np.eye(block.shape[0]),
                                    list(range(block.shape[0])), centre,
                                    radius, config)
    return block, float(leakage), read


def contents():
    """All ten occupations of the three bands by three quarks."""
    return [c for c in itertools.product(range(4), repeat=3) if sum(c) == 3]


IRREP_NAMES = ("2", "2'", "2''")


def restriction(j2, triality):
    """The 2T content of a three-quark spin sector of total triality tau.

    The three doublets carry the Z_3 = 2T/Q_8 characters 0, 1, 2 (2, 2',
    2''). A spin-1/2 multiplet of total triality tau restricts to the doublet
    of label tau; a spin-3/2 multiplet restricts to 2' (x) chi^tau plus
    2'' (x) chi^tau, the j = 3/2 quartet being 2' + 2'' on the tetrahedron
    (WP §11.1 line 499). A tetrahedral support therefore cannot tell spin 1/2
    with triality from half of spin 3/2."""
    if abs(j2 - SPIN_HALF) < 1e-9:
        return [IRREP_NAMES[triality % 3]]
    return [IRREP_NAMES[(1 + triality) % 3], IRREP_NAMES[(2 + triality) % 3]]


def third_exterior_power(one_particle, basis=None):
    """Lambda^3 of a one-particle map in the occupation basis:
    entry (I, J) is the 3 x 3 minor det(X[I, J]) of the ascending tuples I, J,
    the coefficient of the wedge I in the image of the wedge J."""
    index = np.array(basis if basis is not None else occupation_basis())
    minors = one_particle[index[:, None, :, None], index[None, :, None, :]]
    return np.linalg.det(minors)


def rotation_averaged_many_body(operator, actions, frame, dual):
    """The three-particle operator averaged over the diagonal rotation action,
    (1/|T|) sum_g Lambda^3(D_g)^{-1} H Lambda^3(D_g), with D_g the rotation in
    the fibre frame. For H = dGamma(X) this is dGamma of the T-averaged X, and
    for the eliminated quartic it is the elimination with every coupling
    rotated, so it is the T-average of WP line 497 applied to the whole
    operator the poles are read on."""
    total = np.zeros_like(operator)
    for d in actions:
        lifted = third_exterior_power(dual @ d @ frame)
        total += np.linalg.solve(lifted, operator @ lifted)
    return total / len(actions)


def relax_content(content, kappa, beta, config, actions):
    """Steps 1-2 for one content: a fresh host relaxed to self-consistency,
    the carried density filling the bands of the T-averaged operator."""
    spacetime = build_host(config["edge_squared"], config.get("host_cell"))
    declaration = action_declaration(spacetime, kappa, beta,
                                     config["regge_hinges"],
                                     holonomy=config["holonomy"])
    config.setdefault("reference_lengths",
                      list(declaration.reference_lengths))
    action = cob.JointAction(spacetime, declaration)
    solve = cob.SelfConsistentMeanField(
        action, mean_field_declaration(content, config, actions))
    report = solve.solve()
    return spacetime, solve.action, report


def _micro_frame(alignments):
    """The aligned base frame of each sheet lifted to the microscopic cells;
    column b * 3 + t is base mode b on sheet t."""
    frame = np.zeros((SHEETS * BASE_EDGES, SHEETS * BASE_EDGES),
                     dtype=complex)
    for t, alignment in enumerate(alignments):
        for b in range(BASE_EDGES):
            frame[t * BASE_EDGES:(t + 1) * BASE_EDGES, b * SHEETS + t] = \
                alignment["frame"][:, b]
    return frame


def _block_scalar(in_frame):
    """The per-carrier energies of an operator in the aligned frame and its
    relative departure from block-scalar form (one number per doublet times
    the sheets)."""
    energies = []
    rest = in_frame.copy()
    for c in range(3):
        sl = slice(2 * c * SHEETS, 2 * (c + 1) * SHEETS)
        energy = complex(np.trace(in_frame[sl, sl]) / (2 * SHEETS))
        energies.append(energy)
        rest[sl, sl] -= energy * np.eye(2 * SHEETS)
    return energies, float(np.linalg.norm(rest) / np.linalg.norm(in_frame))


def evaluate_content(content, kappa, beta, config, alignment):
    """One content at one scan point: relaxation, recursion, quark conditions,
    states, spin, operators and poles."""
    started = time.time()
    declared_actions = rotation_action([monopole_support()] * SHEETS)
    spacetime, action, report = relax_content(content, kappa, beta, config,
                                              declared_actions)
    carrier = matrix(action.carrier_operator())

    # The spin is read with the canonical projective action D_1(g) of the
    # declared host (the symmetric monopole configuration): the relaxation
    # under h_1, which is not itself rotation-covariant, moves the fields off
    # the symmetric configuration, and how far is reported below as the
    # gauge-compensation residual of each rotation at the relaxed connection.
    supports = [sheet_support(spacetime, t) for t in range(SHEETS)]
    compensation = max(
        support.gaugeCompensation(g).residual
        for support, _ in supports for g in rotation_group())
    actions = declared_actions
    alignments = [alignment] * SHEETS
    frame = _micro_frame(alignments)
    dual = np.linalg.inv(frame)
    averaged = rotation_averaged(carrier, actions)
    band_energies, averaged_residual = _block_scalar(dual @ averaged @ frame)
    _, covariant_residual = _block_scalar(dual @ carrier @ frame)
    carrier_of_band = [int(c) for c in
                       np.argsort([e.real for e in band_energies])]
    trialities = alignments[0]["trialities"]

    # the quartic's ingredients, on h_1 itself: the retained fluctuations
    # (the squared lengths, and the link phases unless only the lengths are
    # declared), their couplings O_a, the bare stiffness A of the geometric
    # action and its Drazin inverse A^D
    fluctuations = eliminate_fluctuations(spacetime, action, carrier, kappa,
                                          beta, config)
    couplings = fluctuations["couplings"]
    shift = fluctuations["shift"]
    constant = fluctuations["constant"]
    truncation = fluctuations["truncation"]

    def many_body(carrier_matrix, coupling_matrices):
        declaration = cob.DressedFluctuationDeclaration()
        declaration.carrier_dimension = SHEETS * BASE_EDGES
        declaration.carrier = list(carrier_matrix.reshape(-1))
        declaration.couplings = [list(o.reshape(-1))
                                 for o in coupling_matrices]
        declaration.bare_stiffness = list(
            fluctuations["reduced_stiffness"].reshape(-1))
        declaration.occupied_modes = 3
        dressed = cob.DressedFluctuation(declaration)
        return dressed.effective_action(list(frame.reshape(-1)),
                                        list(dual.reshape(-1)), 3)

    # the reduced coordinates f = R g carry the Drazin elimination
    coupling_matrices = fluctuations["reduced_couplings"]
    # quasi-free: dGamma of the T-averaged operator
    quasi_free_read = many_body(averaged, coupling_matrices)
    dimension = int(quasi_free_read.dimension)
    quasi_free = np.asarray(quasi_free_read.one_body).reshape(dimension,
                                                               dimension)
    # with the quartic: the whole eliminated operator averaged over the
    # diagonal rotation action, term by term (a rotated carrier and rotated
    # couplings give the rotated many-body operator)
    quartic_read = many_body(carrier + shift, coupling_matrices)
    eliminated = np.asarray(quartic_read.effective_action).reshape(
        dimension, dimension)
    with_quartic = rotation_averaged_many_body(eliminated, actions, frame,
                                               dual) + \
        constant * np.eye(dimension)

    # states of the content, by carrier (the content is in band order)
    carrier_content = [0, 0, 0]
    for band, n in enumerate(content):
        carrier_content[carrier_of_band[band]] = n
    triality = sum(n * t for n, t in zip(carrier_content, trialities)) % 3
    states, labels = singlet_states(carrier_content)
    sectors, j2_values = spin_sectors(states)
    basis = occupation_basis()
    spins = edge_spin_matrices()

    sector_reads = {}
    for j2, sector in sectors.items():
        irreps = restriction(j2, triality)
        entry = {
            "dimension": int(sector.shape[1]),
            "total_triality": int(triality),
            "restriction_to_2T": irreps,
            "nucleon_reading": "2" in irreps,
            "delta_reading": sorted(irreps) == sorted(["2'", "2''"]),
            "tetrahedral_ambiguity": (
                "on a tetrahedral support spin 1/2 with triality and half of "
                "spin 3/2 are indistinguishable (WP line 499): this sector "
                "restricts to %s" % " + ".join(irreps)),
        }
        for name, operator in (("quasi_free", quasi_free),
                               ("with_quartic", with_quartic)):
            block, leakage, read = sector_poles(operator, sector)
            poles = [complex(p) for p in read.poles]
            lowest = min(poles, key=lambda p: (p.real, p.imag)) \
                if poles else None
            # the lowest pole's right and left eigenvectors, for the spin
            # and colour certificates
            values, right = np.linalg.eig(block)
            k = int(np.argmin(np.abs(values - lowest))) if lowest is not None \
                else 0
            values_left, left = np.linalg.eig(block.T)
            kl = int(np.argmin(np.abs(values_left - values[k])))
            right_state = to_fock(sector @ right[:, k], basis)
            left_state = to_fock(left_inverse(sector).T @ left[:, kl], basis)
            spin = obs.SharpSpin.read(spins, right_state, left_state, j2,
                                      DECLARED_CERTIFICATE_TOLERANCE)
            casimir = np.linalg.norm(colour_casimir(right_state)) / \
                np.linalg.norm(right_state)
            entry[name] = {
                "poles": poles,
                "multiplicity": [int(m) for m in read.multiplicity],
                "lowest_pole": lowest,
                "failed_certificates": list(read.failed_certificates),
                "zero_count_defect": float(read.zero_count_defect),
                "continuation_movement": [float(x) for x in
                                          read.continuation_movement],
                "contour": [complex(read.centre), float(read.radius)],
                "compression_leakage": leakage,
                "sharp_spin": bool(spin.sharp),
                "spin_right_residual": float(spin.right_residual),
                "spin_left_residual": float(spin.left_residual),
                "spin_expectation": complex(spin.expectation),
                "determinant_count": int(spin.determinant_count),
                "colour_casimir_residual": float(casimir),
            }
        sector_reads[j2] = entry

    recursion = recursion_read(spacetime, config)
    quark = quark_conditions(spacetime, alignment, recursion,
                             averaged_residual, report)
    truncation_read = action.holonomy_truncation()
    record = {
        "content": list(content),
        "holonomy": config["holonomy"],
        "elimination": config["elimination"],
        "carrier_content": carrier_content,
        "seconds": time.time() - started,
        "relaxation": {
            "converged": bool(report.converged),
            "force_norm": float(report.force_norm),
            "covariance_change": float(report.covariance_change),
            "purity_defect": float(report.purity_defect),
            "spectral_gap": float(report.spectral_gap),
            "band_ranks": [int(r) for r in report.band_ranks],
            "iterations": len(report.steps),
            "action": complex(report.action),
            "terms": {name: complex(getattr(action, name + "_term")())
                      for name in ("regge", "stiffness", "holonomy",
                                   "matter", "spectral")},
            "regge_hinge_count": int(action.regge_hinge_count()),
            "edge_lengths": [complex(e.getLength()) for e in
                             spacetime.getEdgeList().toVector()],
            "face_holonomies": [complex(f) for f in
                                action.face_holonomies()],
            "holonomy_truncation": {
                "tolerance": float(truncation_read.tolerance),
                "declared_term_count": int(
                    truncation_read.declared_term_count),
                "maximum_term_count": int(truncation_read.maximum_term_count),
                "relative_value_tail": float(
                    truncation_read.relative_value_tail),
                "relative_first_tail": float(
                    truncation_read.relative_first_tail),
                "relative_second_tail": float(
                    truncation_read.relative_second_tail),
            },
            "unit_circle_departure": max(dep for _, dep in supports),
            "symmetry_departure": float(compensation),
            "monopole_numbers": [int(sp.monopoleNumber().monopole_number)
                                 for sp, _ in supports],
            "occupied_energy": complex(report.occupied_energy),
            "link_force_norm": float(np.linalg.norm(
                action.link_stationarity())),
            "zero_guard_damped_steps": int(report.zero_guard_damped_steps),
            "holonomy_zero_distance": float(action.holonomy_zero_distance()),
        },
        "covariant_spectrum": sorted(
            [complex(v) for v in np.linalg.eigvals(carrier)],
            key=lambda v: (v.real, v.imag)),
        "averaged_spectrum": sorted(
            [complex(v) for v in np.linalg.eigvals(averaged)],
            key=lambda v: (v.real, v.imag)),
        "band_energies": band_energies,
        "carrier_of_band": carrier_of_band,
        "trialities": trialities,
        "symmetry_residual": covariant_residual,
        "averaged_symmetry_residual": averaged_residual,
        "j2_values": j2_values,
        "quartic": {
            "constant": complex(constant),
            "stiffness_asymmetry": float(quartic_read.stiffness_asymmetry),
            "stiffness_conditioning": float(
                quartic_read.stiffness_conditioning),
            "frame_pairing_defect": float(quartic_read.frame_pairing_defect),
            "certificate": quartic_read.certificate.describe(),
            "truncation": truncation,
            "fluctuations": fluctuations["record"],
        },
        "sectors": {str(k): v for k, v in sector_reads.items()},
        "recursion": recursion,
        "quark_conditions": quark,
    }
    if config.get("isospin_doublet"):
        # The isospin-doublet observation (WP §10) on h_1 and its T-average,
        # added only when requested so that the default record is unchanged.
        from tessera.drivers import isospin_doublet
        spinorial = bool(monopole_support().cocycle(rotation_group())
                         .nontrivial)
        record["isospin_doublet"] = isospin_doublet.observe_host(
            carrier, actions, spinorial)
    return record


def recursion_read(spacetime, config):
    """One turn of the level recursion on h_1 of the relaxed host (WP §15),
    keeping every mode of each component as its fibre."""
    declaration = cob.LevelRecursionDeclaration()
    bands = cob.RecursionBandDeclaration()
    bands.selection = cob.RecursionBandSelection.LowestModes
    bands.band_rank = BASE_EDGES
    declaration.bands = bands
    recursion = cob.LevelRecursion.overSpacetime(
        spacetime, 1, cob.HodgeMetricSource.WhitneyPencil, declaration)
    recursion.advance()
    level = recursion.level(1) if recursion.level_count() > 1 else \
        recursion.level(0)
    return {
        "partition": [list(p) for p in level.partition],
        "band_ranks": [int(b.rank) for b in level.bands],
        "bands_accepted": [bool(b.accepted) for b in level.bands],
        "isolation_gaps": [float(b.isolation_gap) for b in level.bands],
        "projector_idempotency": [float(b.projector_idempotency)
                                  for b in level.bands],
        "pairing_defects": [float(b.pairing_defect) for b in level.bands],
        "gram_defect": float(level.gram_defect),
        "modes": int(level.modes),
        "transport_norms": [float(np.linalg.norm(np.asarray(t.block)))
                            for t in level.transports],
        "fock_stage_dimension": float(level.fock_stage_dimension),
        "determinant_residual": float(level.determinant_residual),
        "certificate": level.certificate.describe(),
        "reduction_certificate": level.reduction_certificate.describe(),
    }


def quark_conditions(spacetime, alignment, recursion, symmetry_residual,
                     report):
    """The seven v16 quark conditions by name (`QuarkConditions`), from what a
    single-level synthesis measures. Anything that needs several cobordism
    frames is left unmeasured and so reads "not evaluable"."""
    E = obs.QuarkConditionEvidence
    spin = alignment["spin_read"]
    supports = [sheet_support(spacetime, t) for t in range(SHEETS)]
    monopoles = [s.monopoleNumber() for s, _ in supports]
    sheeting = obs.SheetedSupport(SHEETS, BASE_EDGES)
    isomorphism = sheeting.certifyIsomorphism(
        [np.array(sheet_squared_lengths(spacetime, t)) for t in range(SHEETS)],
        [np.array(sheet_links(spacetime, t)) for t in range(SHEETS)],
        DECLARED_CERTIFICATE_TOLERANCE)
    attachment = obs.SheetAttachment.attachmentMatrix(
        SHEETS, [obs.ConnectingSimplex(t, t, 1.0) for t in range(SHEETS)])
    doublet = spin.bands[spin.doublet_index] if spin.half_integer_doublet \
        else None
    accepted = all(recursion["bands_accepted"])
    evidence = [
        [E("persistent-support", accepted,
           "partition %s" % recursion["partition"]),
         E("localized-projector-rank", accepted,
           "ranks %s, idempotency %s" % (recursion["band_ranks"],
                                         recursion["projector_idempotency"])),
         E("contour-separation", all(g > 0 for g in
                                     recursion["isolation_gaps"]),
           "isolation gaps %s" % recursion["isolation_gaps"]),
         E("successor-overlap", None, "a single level has no successor"),
         E("multi-frame-lifetime", None,
           "a single level spans one cobordism frame"),
         E("external-leakage", all(n < DECLARED_CERTIFICATE_TOLERANCE
                                   for n in recursion["transport_norms"]),
           "transport norms %s" % recursion["transport_norms"])],
        [E("three-sheeted-support", True, "%d sheets" % SHEETS),
         E("sheet-isomorphism", bool(isomorphism.isomorphic),
           "length residual %.3g, connection residual %.3g"
           % (isomorphism.squared_length_residual,
              isomorphism.connection_residual)),
         E("protected-base-band",
           bool(doublet is not None and doublet.spinor_doublet
                and all(m.odd for m in monopoles)
                and spin.cocycle.nontrivial),
           "monopole numbers %s, cocycle nontrivial %s, commutator phase %s"
           % ([m.monopole_number for m in monopoles],
              spin.cocycle.nontrivial, spin.cocycle.commutator_phase)),
         E("base-band-sector", bool(doublet is not None and doublet.coexact),
           "coexact residual %s" % (doublet.coexact_residual
                                    if doublet is not None else None)),
         E("fibre-lift", symmetry_residual < 1e-6,
           "the T-averaged h_1 (WP line 497) in the aligned frame is "
           "block-scalar on each doublet times the sheets to relative "
           "residual %.3g" % symmetry_residual)],
        [E("anchor-profile-nonzero", None,
           "the dressed anchor atlas is not read by this driver"),
         E("anchor-covariance", None, "not read"),
         E("anchor-transitions", None, "not read"),
         E("anchor-stable-across-frames", None,
           "a single level spans one cobordism frame")],
        [E("odd-occupation-parity", True,
           "one occupied mode of the fibre per quark: parity (-1)^1")],
        [E("color-transport-full-rank",
           bool(attachment.certificate.holds()),
           "det S = %s (sheet-to-sheet attachment)"
           % complex(attachment.determinant)),
         E("base-transport-leakage", all(
             n < DECLARED_CERTIFICATE_TOLERANCE
             for n in recursion["transport_norms"]),
           "inter-component transport norms %s" % recursion["transport_norms"]),
         E("transport-over-lifetime", None,
           "a single level has no lifetime")],
        [E("lineage-intersection", None,
           "a lineage needs a cobordism history")],
        [E("refinement-stability", None, "no refinement is run"),
         E("relabeling-stability", None, "no relabeling is run")],
    ]
    verdict = obs.QuarkConditions.evaluate(evidence)
    return {
        "certified": bool(verdict.certified),
        "conditions": [{
            "number": int(c.number), "name": c.name,
            "status": c.status.name,
            "missing": list(c.missing), "failing": list(c.failing),
            "evidence": [{"name": e.name, "held": e.held,
                          "detail": e.detail} for e in c.evidence],
        } for c in verdict.conditions],
    }


# --------------------------------------------------------------- the scan


def scan_point(kappa, beta, config, alignment, on_content=None):
    """Every content at one (kappa, beta), and the ratios."""
    records = []
    for content in config.get("contents") or contents():
        try:
            record = evaluate_content(content, kappa, beta, config,
                                      alignment)
        except ValueError as error:
            # a declared refusal of the library at this content (a band that
            # cannot hold the occupation, a face holonomy outside the domain
            # of the holonomy term): recorded with its reason, and the content
            # supplies no pole
            record = {"content": list(content),
                      "holonomy": config["holonomy"],
                      "elimination": config["elimination"],
                      "failed": str(error), "sectors": {}}
        records.append(record)
        if on_content is not None:
            on_content(record)
    return {"kappa": kappa, "beta": beta, "holonomy": config["holonomy"],
            "elimination": config["elimination"],
            "failed_contents": [record["content"] for record in records
                                if "failed" in record],
            "contents": records,
            "ratios": ratios(records), "pole_table": pole_table(records)}


def pole_table(records):
    """Per column (quasi-free, with the quartic) and per spin, every content's
    lowest pole in ascending order of real part, so the content that supplies
    the nucleon and the Delta pole, and any tie between spins inside one
    content, can be read off the record."""
    table = {}
    for name in ("quasi_free", "with_quartic"):
        table[name] = {}
        for j2 in (str(SPIN_HALF), str(SPIN_THREE_HALVES)):
            rows = []
            for record in records:
                entry = record["sectors"].get(j2)
                if entry and entry[name]["lowest_pole"] is not None:
                    rows.append({"content": record["content"],
                                 "pole": entry[name]["lowest_pole"],
                                 "restriction_to_2T":
                                     entry.get("restriction_to_2T")})
            rows.sort(key=lambda row: (row["pole"].real, row["pole"].imag))
            table[name][j2] = rows
    return table


def _pair(s_n, s_d, extra):
    out = {
        "nucleon_pole": s_n, "delta_pole": s_d,
        "pole_ratio": s_n / s_d,
        "modulus_ratio": abs(s_n) / abs(s_d),
        "real_part_ratio": s_n.real / s_d.real,
        "target_mass_squared_ratio": TARGET_MASS_SQUARED_RATIO,
        "target_mass_ratio": TARGET_MASS_RATIO,
    }
    out.update(extra)
    return out


def ratios(records):
    """The nucleon and Delta poles over the contents, and their ratios against
    the target, in two pairings.

    * By spin: the lowest sharp spin-1/2 pole over the lowest sharp spin-3/2
      pole. Each candidate carries its restriction to 2T, and the Delta
      candidate the tetrahedral ambiguity: its sector is a Delta reading only
      when it restricts to the degenerate pair 2' + 2''; a 2 in its
      restriction is indistinguishable from spin 1/2 with triality.
    * By 2T reading: the lowest pole whose sector restricts to a 2 (a nucleon
      reading) over the lowest pole whose sector restricts to 2' + 2'' (a Delta
      reading).
    """
    out = {}
    for name in ("quasi_free", "with_quartic"):
        by_spin, nucleon_reading, delta_reading = {}, None, None
        for record in records:
            for j2 in (str(SPIN_HALF), str(SPIN_THREE_HALVES)):
                entry = record["sectors"].get(j2)
                if not entry or entry[name]["lowest_pole"] is None:
                    continue
                pole = entry[name]["lowest_pole"]
                candidate = (pole, record["content"], j2,
                             entry.get("restriction_to_2T"))
                if j2 not in by_spin or pole.real < by_spin[j2][0].real:
                    by_spin[j2] = candidate
                if entry.get("nucleon_reading") and (
                        nucleon_reading is None
                        or pole.real < nucleon_reading[0].real):
                    nucleon_reading = candidate
                if entry.get("delta_reading") and (
                        delta_reading is None
                        or pole.real < delta_reading[0].real):
                    delta_reading = candidate
        result = {}
        if str(SPIN_HALF) in by_spin and str(SPIN_THREE_HALVES) in by_spin:
            n = by_spin[str(SPIN_HALF)]
            d = by_spin[str(SPIN_THREE_HALVES)]
            restriction_d = d[3] or []
            result["by_spin"] = _pair(n[0], d[0], {
                "nucleon_content": n[1], "delta_content": d[1],
                "nucleon_restriction": n[3], "delta_restriction": d[3],
                "delta_is_a_delta_reading":
                    sorted(restriction_d) == sorted(["2'", "2''"]),
                "delta_ambiguous_with_spin_half": "2" in restriction_d,
            })
        else:
            result["by_spin"] = None
        if nucleon_reading is not None and delta_reading is not None:
            result["by_2T_reading"] = _pair(
                nucleon_reading[0], delta_reading[0], {
                    "nucleon_content": nucleon_reading[1],
                    "nucleon_spin_j_j_plus_1": float(nucleon_reading[2]),
                    "delta_content": delta_reading[1],
                    "delta_spin_j_j_plus_1": float(delta_reading[2]),
                })
        else:
            result["by_2T_reading"] = None
        out[name] = result
    return out


def default_config(kappas=DECLARED_KAPPAS, betas=DECLARED_BETAS,
                   edge_squared=DECLARED_EDGE_SQUARED,
                   regge_hinges="interior", selected_contents=None,
                   holonomy=DECLARED_HOLONOMY,
                   elimination=DECLARED_ELIMINATION):
    """The declared configuration, recorded with every run. The contents
    default to all ten; a subset is for tests and quick checks and changes no
    number of the contents it keeps."""
    return {
        "mode": "controlled synthesis",
        "contents": [list(c) for c in (selected_contents or contents())],
        "kappas": list(kappas),
        "betas": list(betas),
        "edge_squared": edge_squared,
        "regge_hinges": regge_hinges,
        "holonomy": holonomy,
        "elimination": elimination,
        "gauge_resonance_radius": DECLARED_GAUGE_RESONANCE_RADIUS,
        "ward_contour_radius": DECLARED_WARD_CONTOUR_RADIUS,
        "ward_contour_nodes": DECLARED_WARD_CONTOUR_NODES,
        "holonomy_zero_margin": DECLARED_HOLONOMY_ZERO_MARGIN,
        "band_tolerance": DECLARED_BAND_TOLERANCE,
        "newton_iterations": 40,
        "newton_tolerance": 1e-11,
        "jacobian_radius": 1e-4,
        "mean_field_iterations": 40,
        "mean_field_tolerance": 1e-9,
        "target_mass_ratio": TARGET_MASS_RATIO,
        "target_mass_squared_ratio": TARGET_MASS_SQUARED_RATIO,
    }


def points_path(json_path):
    """The append-only JSON-lines file beside ``json_path`` that holds one
    line per completed scan point (and a first line with the configuration
    and the host), so a stopped run loses nothing."""
    root, _ = os.path.splitext(os.fspath(json_path))
    return root + ".points.jsonl"


def _append_line(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(_jsonable(record)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def drive(config, progress=False, on_frame=None, stop_requested=None,
          points_file=None):
    """The whole scan. `on_frame(frames, index)` is called after each scan
    point with the list of completed points; the computation is the same with
    or without it. With `points_file`, the configuration and the host are
    written as the first line and every scan point's full record is appended
    as one JSON line the moment the point completes."""
    alignment = aligned_doublet_frame(monopole_support(), rotation_group())
    frames = []
    host = {
        "monopole": _monopole_record(alignment["spin_read"]),
        "averaged_eigenvalues": alignment["averaged_eigenvalues"],
        "reference_carrier": alignment["reference_carrier"],
        "intertwining_residual": alignment["intertwining_residual"],
    }
    if points_file is not None:
        with open(points_file, "w"):
            pass
        _append_line(points_file, {"config": config, "host": host})
    for kappa in config["kappas"]:
        for beta in config["betas"]:
            if stop_requested is not None and stop_requested():
                return {"config": _jsonable(config), "host": host,
                        "points": frames, "stopped": True}
            point = scan_point(kappa, beta, config, alignment)
            frames.append(point)
            if points_file is not None:
                _append_line(points_file, point)
            if progress:
                r = point["ratios"]
                sys.stdout.write(
                    "kappa=%g beta=%g  quasi-free %s  with quartic %s\n"
                    % (kappa, beta, _ratio_text(r["quasi_free"]["by_spin"]),
                       _ratio_text(r["with_quartic"]["by_spin"])))
                sys.stdout.flush()
            if on_frame is not None:
                on_frame(frames, len(frames) - 1)
    return {"config": _jsonable(config), "host": host, "points": frames,
            "stopped": False}


def _monopole_record(read):
    return {
        "monopole_number": int(read.monopole.monopole_number),
        "odd": bool(read.monopole.odd),
        "cocycle_nontrivial": bool(read.cocycle.nontrivial),
        "commutator_phase": complex(read.cocycle.commutator_phase),
        "half_integer_doublet": bool(read.half_integer_doublet),
        "doublet_index": int(read.doublet_index),
        "bands": [{"eigenvalue": float(b.eigenvalue),
                   "dimension": int(b.dimension),
                   "spinor_doublet": bool(b.spinor_doublet),
                   "coexact": bool(b.coexact),
                   "coexact_residual": float(b.coexact_residual),
                   "irreducibility_score": float(b.irreducibility_score)}
                  for b in read.bands],
        "certificate": read.certificate.describe(),
    }


def _ratio_text(ratio):
    if ratio is None:
        return "no pole pair"
    return "s_N/s_D = %s (target %.4f)" % (_complex_text(ratio["pole_ratio"]),
                                           TARGET_MASS_SQUARED_RATIO)


def _complex_text(value):
    return "%.6g%+.3gi" % (value.real, value.imag)


def _jsonable(value):
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.complexfloating):
        return {"re": float(value.real), "im": float(value.imag)}
    return value


# ------------------------------------------------------------ the drawing

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SERIES = {"quasi_free": "#2a78d6", "with_quartic": "#eb6834"}
SERIES_LABEL = {"quasi_free": "quasi-free dGamma(h)",
                "with_quartic": "with the Section 7 quartic"}


def draw_frame(figure, frames, index):
    """One frame: the pole ratio at every completed scan point against the
    target, and the poles of the latest point by content and spin. Cheap:
    two small axes and a few dozen marks."""
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
    labels = ["k=%g\nb=%g" % (p["kappa"], p["beta"]) for p in done]
    x = np.arange(len(done))
    for name in ("quasi_free", "with_quartic"):
        values = [p["ratios"][name]["by_spin"]["pole_ratio"].real
                  if p["ratios"][name]["by_spin"] else np.nan for p in done]
        left.plot(x, values, marker="o", markersize=6, linewidth=2,
                  color=SERIES[name], label=SERIES_LABEL[name])
    left.axhline(TARGET_MASS_SQUARED_RATIO, color=INK_MUTED, linewidth=1,
                 linestyle="--")
    left.text(0, TARGET_MASS_SQUARED_RATIO, " target (m_N/m_D)^2 = %.4f"
              % TARGET_MASS_SQUARED_RATIO, color=INK_MUTED, va="bottom",
              fontsize=8)
    left.set_xticks(x)
    left.set_xticklabels(labels, fontsize=7)
    left.set_ylabel("Re(s_N / s_Delta)", color=INK)
    left.set_title("pole ratio over the scan", color=INK, fontsize=10)
    left.legend(frameon=False, fontsize=8, labelcolor=INK)

    point = done[-1]
    names, half, three = [], [], []
    for record in point["contents"]:
        names.append("".join(str(n) for n in record["content"]))
        for store, key in ((half, str(SPIN_HALF)),
                           (three, str(SPIN_THREE_HALVES))):
            entry = record["sectors"].get(key)
            store.append(entry["quasi_free"]["lowest_pole"].real
                         if entry and entry["quasi_free"]["lowest_pole"]
                         is not None else np.nan)
    positions = np.arange(len(names))
    right.plot(positions, half, "o", markersize=8, color="#1baf7a",
               label="spin 1/2")
    right.plot(positions, three, "s", markersize=8, color="#4a3aa7",
               label="spin 3/2", markerfacecolor="none", markeredgewidth=2)
    right.set_xticks(positions)
    right.set_xticklabels(names, fontsize=7)
    right.set_xlabel("content (quarks per band, ascending band)", color=INK)
    right.set_ylabel("Re s (quasi-free)", color=INK)
    right.set_title("poles at kappa=%g, beta=%g" % (point["kappa"],
                                                   point["beta"]),
                    color=INK, fontsize=10)
    right.legend(frameon=False, fontsize=8, labelcolor=INK)
    figure.suptitle("nucleon-to-Delta poles by controlled synthesis "
                    "(%d of the scan done)" % len(done), color=INK)
    figure.tight_layout()


def _interactive_backends():
    """The interactive matplotlib backends, lowercased (asked of matplotlib,
    as `emergence._interactive_backends` does)."""
    from tessera.drivers.emergence import _interactive_backends as backends
    return backends()


def drive_live(config, progress=False, points_file=None):
    """The same `drive`, on a worker thread, drawing each completed scan point
    on the main thread. Refuses a non-interactive backend and WebAgg by name,
    as `tessera.drivers.emergence` does.

    Closing the window does not stop the scan: the figure's close event
    switches the run to headless, the main thread stops drawing and waits for
    the worker, and every output is still written."""
    import queue
    import threading

    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend()
    name = backend.lower()
    is_webagg = "webagg" in name
    if name not in _interactive_backends() or is_webagg:
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
    # The window is shown once and then only repainted: raising it or pumping
    # events through `plt.pause` (which calls `show`) would bring it to the
    # front and take the keyboard focus on every frame.
    matplotlib.rcParams["figure.raise_window"] = False
    if not plt.isinteractive():
        plt.ion()
    figure = plt.figure(figsize=(13, 6))
    plt.show(block=False)
    ready = queue.Queue()
    published = {}
    outcome = {}
    stop = threading.Event()

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    closed = threading.Event()

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

    thread = threading.Thread(target=worker, name="baryon-poles")
    thread.start()
    main_error = None
    try:
        while not closed.is_set():
            try:
                index = ready.get_nowait()
            except queue.Empty:
                figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
                continue
            if index is None:
                break
            if closed.is_set():
                break
            draw_frame(figure, published["frames"], index)
            figure.canvas.draw_idle()
            figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
    except BaseException as error:
        main_error = error
        stop.set()
    finally:
        # Headless from here: the worker is waited for, not stopped, unless the
        # main thread itself failed or was interrupted.
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
    if result["points"]:
        draw_frame(figure, result["points"], len(result["points"]) - 1)
    figure.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(figure)


def summary(result):
    """The ratios of every scan point as text."""
    lines = ["mode: controlled synthesis; target m_N/m_Delta = %.4f, "
             "(m_N/m_Delta)^2 = %.4f" % (TARGET_MASS_RATIO,
                                         TARGET_MASS_SQUARED_RATIO)]
    for point in result["points"]:
        for name in ("quasi_free", "with_quartic"):
            for pairing in ("by_spin", "by_2T_reading"):
                r = point["ratios"][name][pairing]
                head = "kappa=%g beta=%g %-12s %-13s" % (
                    point["kappa"], point["beta"], name, pairing)
                if r is None:
                    lines.append(head + " no pole pair")
                    continue
                note = ""
                if pairing == "by_spin":
                    note = " Delta restriction %s%s" % (
                        "+".join(r["delta_restriction"] or []),
                        " (ambiguous with spin 1/2)"
                        if r["delta_ambiguous_with_spin_half"] else "")
                lines.append(
                    head + " s_N=%s (content %s) s_D=%s (content %s) "
                    "s_N/s_D=%s |s_N|/|s_D|=%.6f Re/Re=%.6f%s"
                    % (_complex_text(r["nucleon_pole"]), r["nucleon_content"],
                       _complex_text(r["delta_pole"]), r["delta_content"],
                       _complex_text(r["pole_ratio"]), r["modulus_ratio"],
                       r["real_part_ratio"], note))
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m tessera.drivers.baryon_poles",
        description="The nucleon-to-Delta pole ratio by controlled synthesis "
                    "on the three-sheeted unit-monopole tetrahedron.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the declared scan")
    run.add_argument("--kappa", type=float, nargs="+",
                     default=list(DECLARED_KAPPAS),
                     help="kappa = 8 pi G values (default %s)"
                          % (DECLARED_KAPPAS,))
    run.add_argument("--beta", type=float, nargs="+",
                     default=list(DECLARED_BETAS),
                     help="beta values of the holonomy term (default %s)"
                          % (DECLARED_BETAS,))
    run.add_argument("--holonomy", choices=tuple(HOLONOMY_FORMS),
                     default=DECLARED_HOLONOMY,
                     help="the holonomy term: villain, the paper's "
                          "heat-kernel form, or wilson, its plaquette "
                          "stand-in (default %s)" % DECLARED_HOLONOMY)
    run.add_argument("--eliminate", choices=ELIMINATIONS,
                     default=DECLARED_ELIMINATION,
                     help="the fluctuations the Section 7 quartic eliminates: "
                          "the squared lengths and the link phases (the "
                          "pure-gauge phases projected out by the Drazin "
                          "inverse), or the squared lengths alone (default "
                          "%s)" % DECLARED_ELIMINATION)
    run.add_argument("--edge-squared", type=float,
                     default=DECLARED_EDGE_SQUARED,
                     help="squared edge length of the tetrahedron (default "
                          "%g, the paper's a^2)" % DECLARED_EDGE_SQUARED)
    run.add_argument("--regge-hinges", choices=("interior", "all"),
                     default="interior",
                     help="hinges of the primal Regge sum (default interior)")
    run.add_argument("--json", default=None,
                     help="write every record here at the end; each scan "
                          "point is also appended, as it completes, to the "
                          "JSON-lines file beside it (<json stem>.points.jsonl)")
    run.add_argument("--out", default=None,
                     help="write the final frame as a PNG here")
    run.add_argument("--live", action="store_true",
                     help="draw each completed scan point while the scan "
                          "runs; the outputs are identical. Needs an "
                          "interactive matplotlib backend (the 'live' extra, "
                          "PyQt6)")
    run.add_argument("--isospin-doublet", action="store_true",
                     help="add the isospin-doublet observation (WP §10, "
                          "`IsospinDoublet`) to every content's record; the "
                          "other outputs are unchanged")
    run.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = default_config(args.kappa, args.beta, args.edge_squared,
                            args.regge_hinges, holonomy=args.holonomy,
                            elimination=args.eliminate)
    if args.isospin_doublet:
        config["isospin_doublet"] = True
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
