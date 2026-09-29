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
quarks in each of the three lowest bands of the covariant operator h_1, see
"What a content names" below), the driver

1. relaxes both edge fields under the joint action
   S = (1/kappa) S_Regge(primal) + S_hol + tr(Gamma h_1)
       + sum_{j=1}^{m_c} xi_j (p_j(h_C) - p_j*)
   whose last term is the spectral-moment part of S_0, the holomorphic
   spectral constraints of WP v17 §3.4 on the occupied fiber: P_C is the
   Riesz projector of the bands the content occupies (rank r),
   h_C = P_C h_1 P_C on its range, p_j(h_C) = tr(h_C^j) its power sums, the
   targets p_j* their values at the host (controlled synthesis pins the
   carrier) and the xi_j independent complex multipliers solved for with the
   geometry. m_c is declared by ``--fiber-moments`` (r, every moment of the
   fiber, by default; 1 pins the trace alone). kappa = 8 pi G enters only
   through the Regge weight; the linear stand-in (1/kappa) (1/2) ||l - l0||^2
   for the spectral-moment part is available by name
   (``--stiffness linear-stand-in``) and off by default. The holonomy term
   S_hol is declared by ``--holonomy``: the Villain form
   -beta_V sum_tau log W(F_tau), W(F) = sum_m exp(-m^2/(2 beta)) F^m,
   beta_V = beta / <m^2>_beta (the default, the paper's holonomy term), or the
   Wilson form beta sum_tau (1 - cos Theta_tau) (the paper's stand-in). Both
   have the bare connection stiffness beta L_1^up at trivial holonomy; at the
   host's quarter-turn holonomies the Wilson stiffness vanishes and the
   Villain stiffness does not;
   with certificates-blind mean-field backreaction to self-consistency
   (`SelfConsistentMeanField`), the carried density being the content's band
   filling of h_1. The fixed point is solved by Newton's method on the joint
   system (WP v17 lines 259 and 263), the covariance rebuilt at every point
   (``--mean-field-method joint-newton``, the default; the alternation of a
   geometry relaxation and a re-occupation is kept as the named fallback
   ``alternation``). The content's bands are chosen at the host in ascending
   order of real part and followed from there by continuation (WP v17 line
   151; ``--band-selection sort-every-iterate`` re-sorts at every iterate
   instead), and every iterate's band overlaps and any crossing are
   recorded. The three sheets are relaxed as one shared base field (WP v17
   §8, "Sheet convention (adopted)"): the solve's variables are the base
   tetrahedron's six squared lengths and six links, written to every sheet,
   and the force on each is the sum of the forces on the corresponding edges
   of the three sheets. A solve that reaches no fixed point says why, by
   name (for example, the iterations ran out, no damped step reduced the
   residual, no stationary point in the declared monopole sector, or the
   lengths ran off), and the poles are not read on a geometry whose lengths
   ran off or that is not Kontsevich-Segal allowable: that read is refused by
   name, with the margin;
2. runs one turn of the level recursion (`LevelRecursion`) and reads the fibre
   certificates;
3. reads the seven v16 quark conditions by name (`QuarkConditions`);
4. reads the spin content of the occupied state on the T-averaged operator:
   the overlap of each occupied band of h_1 with each doublet 2, 2', 2'' of
   h-bar_1, and the number of quarks in each doublet, and the anchor atlas of
   the base band on the relaxed connection (`anchor_atlas_read`, quark
   condition 3): its projective profile over the cell's faces, the
   connection-dressed covariance, the determinant-line transitions and the
   invariant coordinates; and the spectral fingerprint of the fiber under the
   declared vertex relabeling and the declared refinement of the cell
   (`spectral_fingerprint_read`, quark condition 7);
5. for every doublet content (n_2, n_2', n_2''), forms the colour-singlet
   three-quark states, one quark per sheet, sorted by the total spin of the
   constructed lift (`SharpSpin.read` under the aligned SU(2) action of item
   4) and labelled by the types of the binary tetrahedral group each sector
   restricts to;
6. builds the three-particle operator, both quasi-free (dGamma(h-bar_1)) and
   with the paper's Section 7 geometric quartic about the self-consistent
   point (`DressedFluctuation.effectiveAction`), and reads the poles of every
   (doublet content, spin) sector with `BoundStatePole`, each pole with its
   own spinor certificate, the isotypic projector equations of WP v18 §11.1
   and §14 for the sector's 2T types (`SharpSpin.isotypicRead`), its
   spin-lift read (the J^2 eigen-equations, `SharpSpin.read`) and its colour
   certificate.

The report is per doublet content
---------------------------------
Every (content, doublet content) pair is reported on its own, and nothing is
averaged over doublet contents or over contents:

* the stdout summary of a scan point (`point_lines`) has one line per pair,
  carrying both spins and both columns (quasi-free and with the quartic),
  every pole with its multiplicity and its spin and colour certificates, and
  each read's bound-state certificates and compression leakage;
* the minima follow as separate, labelled "lowest over" summaries: per
  content, the lowest pole of each spin and column over its doublet
  contents, naming the doublet content it came from; and the lowest over
  every pair, naming the content and the doublet content. Every pair whose
  pole ties with a minimum (`DECLARED_TIE_TOLERANCE`) is named beside it;
* the pole table (`pole_table`) has one row per pole of every pair, in
  ascending order of real part, each row naming its content and doublet
  content;
* the live frame (`draw_frame`) draws every pole of every pair as its own
  mark, labelled by its content and doublet content, and keeps the ratio as
  a second panel beside a listing of the pairs each ratio compares.

The ratio is read by 2T type (WP v18 §11.1 and §14): the nucleon pole is
the lowest pole over every (content, doublet content) pair whose sector
restricts to a 2, and the Delta pole the lowest whose sector restricts to the
degenerate pair 2' + 2'', "lowest" meaning smallest real part, which is the
library's declared `OccupationOrder.AscendingRealPart`; the ratio names the
two pairs it compares. The pairing by the total spin of the constructed lift,
the lowest sharp spin-1/2 pole over the lowest sharp spin-3/2 pole, is
reported beside it as the spin-lift reading, because the type fixes the
representation of 2T and not the continuum spin value. The poles are complex
and are reported as complex; the ratio
s_N / s_Delta is compared with 0.7616 = m_N / m_Delta, the rest-energy reading
of WP v18 §13.3 (under the first-order flow of §7 the pole of the spatial
pencil is the complex frequency of the bound cluster, its energy at rest; the
theory carries the pole and takes no square root of it), and the ratio of
moduli and the ratio of real parts are reported beside it. The one average on the way to a pole is the T-average that defines
h-bar_1 (WP v17 §9), which is the operator the poles are read on, not a way
of reporting them.

Two operators: the covariance on h_1, the spin on h-bar_1
---------------------------------------------------------
The carried density is built from the covariant operator h_1(z, U) of the
specification (Definition 2) itself: a stationary pair has "Gamma* a
projector onto modes of h(z*)" (WP v17 §7 line 262), and a density built from
the bands of h_1 commutes with h_1, so the matter term tr(Gamma h_1) at fixed
Gamma is gauge invariant (its Ward current is divergence-free) and the rule is
covariant under a gauge transformation of any sheet.

h_1 is not itself invariant under the projective rotation action D_1(g) at the
monopole connection: its six eigenvalues per sheet are simple. The whitepaper
reads the spin content on the T-averaged operator (WP v17 §9 line 506: "The
action on edge modes is canonical even where the twisted operator is not";
"the T-averaged twisted edge Laplacian"), h-bar_1 = (1/|T|) sum_g D_1(g)^{-1}
h_1 D_1(g), with T = A_4 the rotation group of the tetrahedron. The driver
does the same: the spin decomposition of the occupied modes, the spin
sectors and every pole are read on h-bar_1 (and, for the Section 7 quartic, on
the eliminated three-particle operator averaged over the same diagonal
action). The departure of h_1 itself from the symmetric form is reported
beside every record as `symmetry_residual`.

What a content names
--------------------
A content (n_0, n_1, n_2) places n_b quarks in band b of h_1, the bands being
groups of degenerate eigenvalues in ascending order of real part. On the
monopole host each band of h_1 is one simple mode per sheet, rank three, not a
spin doublet, so a content names no doublet. The record carries, beside it,
how the occupied bands decompose into the doublets of h-bar_1
(``spin_decomposition``), and the pole sectors are read for every doublet
content and labelled by it (``doublet_reads``); no mapping from a content to a
doublet content is made.

The same paragraph's caveat is carried with every Delta candidate: on the
tetrahedron the j = 3/2 quartet restricts to 2' + 2'', so "spin-1/2 with
triality" and "half of spin-3/2" are indistinguishable. A Delta reading is a
degenerate 2' + 2'' pair of three-quark states and a nucleon reading is a 2;
each sector reports its restriction, and the ratios are reported by 2T
reading and, beside it, by the spin of the lift.

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

Every tolerance of the stack is an option (``--rank-tolerance`` is tau, the
Newton solve's rank decision; the others are listed by `TOLERANCES`), each
defaulting to 1e-15 and each recorded in the configuration. None changes an
equation.
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
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs

#: The target: the proton mass over the Delta(1232) Breit-Wigner mass (PDG).
#: The pole ratio s_N / s_Delta is compared with it directly, because the pole
#: is the complex rest energy of the bound cluster (WP v18 §13.3), not a mass
#: squared.
PROTON_MASS_MEV = 938.272
DELTA_MASS_MEV = 1232.0
TARGET_MASS_RATIO = PROTON_MASS_MEV / DELTA_MASS_MEV

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
#: The declared value of every tolerance of the stack (`TOLERANCES`): each
#: is its own option of both drivers (`add_tolerance_arguments`) and each
#: defaults to this.
DECLARED_TOLERANCE = 1e-15
#: Relative separation at or below which ordered eigenvalues form one band.
DECLARED_BAND_TOLERANCE = DECLARED_TOLERANCE
#: tau, the relative singular-value threshold of the Newton solve's rank
#: decision. The Jacobian is a two-node real-axis difference at relative
#: radius 1e-4 (``jacobian_radius``), whose rounding error is of order
#: epsilon / radius, about 2e-12, relative to its entries, so a singular value
#: below about 1e-11 of the largest cannot be told from zero. The declared
#: threshold sits below that floor: a singular value counts as zero only
#: when it is below the rounding of the Jacobian itself, and a null direction
#: of the Jacobian (a gauge direction, or a redundant constraint row) is
#: inverted at its rounding size rather than left out of the step.
DECLARED_RANK_TOLERANCE = DECLARED_TOLERANCE
#: Tolerances of the certificates this driver grades.
DECLARED_CERTIFICATE_TOLERANCE = DECLARED_TOLERANCE

#: Every tolerance of the stack, by config key, with what it thresholds.
#: `add_tolerance_arguments` offers each as ``--<key, with dashes>``,
#: `default_config` records each, and the recursion driver carries each into
#: every cell's config. None changes an equation.
TOLERANCES = (
    ("rank_tolerance",
     "tau, the relative singular-value threshold of the Newton solve's rank "
     "decision: a singular value of the Jacobian below this fraction of the "
     "largest counts as zero in the minimum-norm step"),
    ("newton_tolerance",
     "the residual norm at or below which the holomorphic Newton solve of "
     "the geometry is converged"),
    ("mean_field_tolerance",
     "the force norm at or below which the geometry and the covariance are "
     "self-consistent"),
    ("band_tolerance",
     "the relative separation at or below which consecutive ordered "
     "eigenvalues of the carrier form one band"),
    ("certificate_tolerance",
     "the tolerance every certificate graded here holds against: the spinor "
     "and spin-lift reads, the sheet isomorphism, the transport leakage, "
     "the anchor atlas and the spectral fingerprint"),
    ("allowability_tolerance",
     "the Kontsevich-Segal margin at or below which a relaxed geometry is "
     "read as the boundary of allowability and its pole read refused"),
    ("tie_tolerance",
     "the relative separation of real parts within which two poles tie in "
     "the ascending-real-part order"),
    ("degeneracy_tolerance",
     "the separation within which eigenvalues of the rotation-averaged edge "
     "Laplacian form one band in the spin read and in the fingerprint"),
    ("pole_newton_tolerance",
     "the Newton step, relative to the contour radius, below which the pole "
     "read's refinement of a root ends"),
    ("pole_zero_count_tolerance",
     "how far the pole read's argument-principle count may sit from an "
     "integer before the read refuses"),
    ("pole_rank_tolerance",
     "the fraction of the largest singular value at or below which the pole "
     "read's Hankel and residue ranks treat one as zero"),
    ("fluctuation_tolerance",
     "the relative tolerance of the dressed fluctuation's certificates, and "
     "the size below which a collective mode's geometric component is zero"),
    ("recursion_tolerance",
     "the relative tolerance the level recursion's certificates hold "
     "against"),
    ("villain_tolerance",
     "the relative size below which a coefficient of the Villain series is "
     "left out"),
)


def declared_tolerance(config, key):
    """The tolerance ``key`` (`TOLERANCES`) of a config, or the declared
    value when the config leaves it out."""
    return float((config or {}).get(key, DECLARED_TOLERANCE))


def declared_tolerances(tolerances=None):
    """Every tolerance of `TOLERANCES` at its declared value, with those of
    ``tolerances`` (a mapping by key) in their place; a key outside
    `TOLERANCES` is refused."""
    out = {key: DECLARED_TOLERANCE for key, _ in TOLERANCES}
    unknown = sorted(set(tolerances or {}) - set(out))
    if unknown:
        raise ValueError("unknown tolerances %s; the declared ones are %s"
                         % (unknown, [key for key, _ in TOLERANCES]))
    out.update({key: float(value)
                for key, value in (tolerances or {}).items()})
    return out

#: How the mean field's fixed point is solved for (both solve the same
#: equations, WP v17 lines 259 and 263): Newton's method on the joint system,
#: the covariance rebuilt at every point (the default), or the alternation of
#: a geometry relaxation at fixed covariance and a re-occupation, kept as a
#: named fallback. The forms --mean-field-method accepts, and the library's
#: name for each.
DECLARED_MEAN_FIELD_METHOD = "joint-newton"
MEAN_FIELD_METHODS = {"joint-newton": cob.SelfConsistentMethod.JointNewton,
                      "alternation": cob.SelfConsistentMethod.Alternation}
#: Where a content's bands are chosen: once, at the declared host, by the
#: ascending real part, and then followed by continuation (WP v17 line 151: a
#: band is selected by a contour, not by sorting real parts); or re-selected
#: by sorting at every iterate, kept as a named option.
DECLARED_BAND_SELECTION = "continuation"
BAND_SELECTIONS = {"continuation": cob.BandSelection.Continuation,
                   "sort-every-iterate": cob.BandSelection.SortEveryIterate}
#: The length stiffness of the joint action. The spectral-moment part of S_0
#: is the holomorphic spectral constraint of WP v17 §3.4 on the occupied
#: fiber (``fiber_moments``); the linear stand-in (1/2 kappa^-1) ||l - l0||^2,
#: which modelled it, stays available by name and is off by default. Without
#: it kappa = 8 pi G enters only through the Regge weight 1/kappa.
DECLARED_STIFFNESS = "none"
STIFFNESS_FORMS = ("none", "linear-stand-in")
#: m_c, the number of power sums p_j(h_C), j = 1..m_c, of the occupied fiber
#: the mean-field solve pins at their values at the host (WP v17 §3.4: in
#: controlled synthesis the constraints pin a carrier): "r", the fiber's rank
#: (every moment of the fiber), or a count ("1" pins the trace alone, which
#: removes the uniform dilation of the Euler identity; "0" pins nothing).
DECLARED_FIBER_MOMENTS = "r"
#: The growth of the largest squared length, over its value at the host, at
#: which a Newton solve stops and reports that the lengths ran off. The linear
#: stiffness stand-in's force on a squared length saturates at 1/(2 kappa), so
#: the residual has a plateau at infinite length that the monotone residual
#: test would otherwise accept (a stop, not a change of the equations).
DECLARED_LENGTH_RUNAWAY_RATIO = 1e2
#: The Kontsevich-Segal margin (radians) at or below which a geometry is not
#: on the allowable side: min over tetrahedra of pi minus the sum of the
#: moduli of the arguments of the metric's eigenvalues is pi on a Euclidean
#: cell, zero on a real Lorentzian one and negative beyond. A margin within
#: this of zero is the boundary to within the rounding of the arguments, where
#: WP v17 line 151 reads a band only as the limit of an allowable family and
#: never alone, so no pole is read there.
DECLARED_ALLOWABILITY_TOLERANCE = DECLARED_TOLERANCE

#: How often the --live main thread services the GUI event loop.
LIVE_POLL_INTERVAL = 0.05
# While no new frame is ready, the live window's status line (what is running
# and for how long) is redrawn this often, in seconds.
LIVE_STATUS_INTERVAL = 2.0

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
                       holonomy=DECLARED_HOLONOMY,
                       stiffness=DECLARED_STIFFNESS,
                       villain_tolerance=DECLARED_TOLERANCE):
    """The joint action of the calculation (WP §3, §7), with the holonomy term
    in the declared form, ``"villain"`` or ``"wilson"``, and the length
    stiffness in the declared form: ``"none"`` (the default, kappa entering
    only through the Regge weight 1/kappa) or ``"linear-stand-in"``, the
    stand-in (1/2 kappa^-1) ||l - l0||^2 for the spectral-moment part of S_0,
    which the mean-field solve imposes instead as the constraints of WP v17
    §3.4 on the occupied fiber."""
    if stiffness not in STIFFNESS_FORMS:
        raise ValueError("the length stiffness is one of %s; got %r"
                         % (STIFFNESS_FORMS, stiffness))
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = 1.0 / kappa
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = (cob.ReggeHinges.Interior
                                if regge_hinges == "interior"
                                else cob.ReggeHinges.All)
    declaration.stiffness_weight = (1.0 / kappa
                                    if stiffness == "linear-stand-in"
                                    else 0.0)
    declaration.reference_lengths = (
        list(reference_lengths) if reference_lengths is not None else
        [complex(edge.getLength())
         for edge in spacetime.getEdgeList().toVector()])
    declaration.holonomy_weight = beta
    declaration.holonomy_form = HOLONOMY_FORMS[holonomy]
    declaration.villain_tolerance = villain_tolerance
    declaration.matter_weight = matter_weight
    return declaration


def relaxation_declaration(config):
    """The inner holomorphic Newton solve, over every edge of the complex as
    its own coordinate. `relax_content` ties the sheets to one shared base
    field on top of this (`share_sheet_geometry`)."""
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = True
    geometry.relax_multipliers = False
    geometry.maximum_iterations = config["newton_iterations"]
    geometry.tolerance = config["newton_tolerance"]
    geometry.jacobian_mode = cob.HolomorphicJacobianMode.RealAxisDifference
    geometry.contour_radius = config["jacobian_radius"]
    geometry.rank_tolerance = config["rank_tolerance"]
    geometry.holonomy_zero_margin = config["holonomy_zero_margin"]
    geometry.length_runaway_ratio = config.get(
        "length_runaway_ratio", DECLARED_LENGTH_RUNAWAY_RATIO)
    # the monopole sectors a caller holds as boundary data
    # (`tessera.drivers.recursion`); none by default
    geometry.held_sectors = list(config.get("held_sectors") or [])
    return geometry


def sheet_edge_classes(spacetime):
    """The shared base field of the sheeted host (WP v17 §8, "Sheet convention
    (adopted)": equal squared lengths and equal connection values on
    corresponding edges). For every edge in `getEdgeList()` order, the index
    of its base edge in the order of `MonopoleSupport.edges`, and the sign
    that relates its stored orientation to the base edge's ascending one:
    +1 when the stored edge runs from the lower to the higher local vertex,
    -1 otherwise. Corresponding edges of the three sheets carry one class."""
    pairs = [tuple(e) for e in monopole_support().edges]
    classes, orientations = [], []
    for source, target in edge_records(spacetime):
        a, b = source % 4, target % 4
        classes.append(pairs.index((min(a, b), max(a, b))))
        orientations.append(1 if a < b else -1)
    return classes, orientations


def share_sheet_geometry(geometry, spacetime):
    """Declare the sheets' fields one shared base field in a relaxation
    declaration: the solve's variables are the base complex's six squared
    lengths and six links, written to every sheet, and the equation of each
    is the sum of the corresponding edges' stationarity equations, i.e. the
    derivative of the whole action, backreaction included, along the shared
    coordinate. Identical sheets therefore stay identical exactly."""
    classes, orientations = sheet_edge_classes(spacetime)
    geometry.edge_classes = classes
    geometry.edge_class_orientations = orientations
    return geometry


def fiber_moment_count(declaration, action, setting):
    """m_c for a declared setting: an integer as it is, or ``"r"``, the rank
    of the occupied fiber, the sum of the ranks of the bands the content
    occupies as the library's own band rule reads them at the action's point
    (`BandFollower`)."""
    if str(setting) != "r":
        return int(setting)
    read = cob.BandFollower(declaration).read(action.carrier_operator())
    return int(sum(band.rank for band in read.bands))


def mean_field_declaration(content, config, spacetime=None):
    """Band filling with the content's occupations (WP v17 §7 line 250) on the
    bands of the covariant operator h_1 itself (ruling (a): Gamma* is a
    projector onto modes of h(z*), WP v17 §7 line 262), the bands chosen at
    the host in ascending order of real part
    (`OccupationOrder.AscendingRealPart`) and followed from there by
    continuation (``band_selection``, WP v17 line 151). The fixed point is
    solved by the declared ``mean_field_method``, Newton's method on the
    joint system by default. With ``spacetime``, the solve carries the
    sheeted host's shared base field (`share_sheet_geometry`). The number of
    the occupied fiber's power sums pinned is set by `relax_content`, which
    has the host to read the fiber's rank on (`fiber_moment_count`)."""
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = [float(n) for n in content]
    declaration.band_tolerance = config["band_tolerance"]
    declaration.occupation_order = cob.OccupationOrder.AscendingRealPart
    declaration.band_selection = BAND_SELECTIONS[
        config.get("band_selection", DECLARED_BAND_SELECTION)]
    declaration.method = MEAN_FIELD_METHODS[
        config.get("mean_field_method", DECLARED_MEAN_FIELD_METHOD)]
    declaration.maximum_iterations = config["mean_field_iterations"]
    declaration.tolerance = config["mean_field_tolerance"]
    geometry = relaxation_declaration(config)
    if spacetime is not None:
        share_sheet_geometry(geometry, spacetime)
    declaration.geometry = geometry
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


def aligned_doublet_frame(support, group,
                          degeneracy_tolerance=DECLARED_TOLERANCE,
                          tolerance=DECLARED_CERTIFICATE_TOLERANCE):
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
    read = support.spinRead(group, degeneracy_tolerance, tolerance)
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


_ISOTYPIC = {}


def lifted_rotation_maps(alignment, actions, frame, dual):
    """The binary tetrahedral group on the 18 microscopic-frame modes, and the
    character of each doublet on every element.

    Each declared rotation's D_1(g) on the microscopic cells is carried to
    the frame modes, dual @ D_1(g) @ frame, and scaled to determinant one on
    the reference doublet (the coexact j = 1/2 doublet of `spinRead`, on
    sheet 0). With both signs the 24 scaled maps form a linear representation
    of the double cover 2T, on which the element covering the 2 pi rotation
    acts as -1 (the cocycle of D_1(g) has the nontrivial class). The character
    of the doublet with Z_3 label t on an element is the trace of the
    element's block on that doublet, read on sheet 0. Returns the maps and
    the characters keyed by doublet name."""
    trialities = alignment["trialities"]
    reference = int(alignment["reference_carrier"])
    columns = {c: [(2 * c + s) * SHEETS for s in range(2)] for c in range(3)}
    maps, characters = [], {IRREP_NAMES[t]: [] for t in trialities}
    for d in actions:
        in_frame = dual @ d @ frame
        block = in_frame[np.ix_(columns[reference], columns[reference])]
        lifted = in_frame / np.sqrt(np.linalg.det(block))
        for sign in (1.0, -1.0):
            maps.append(sign * lifted)
            for c, t in enumerate(trialities):
                characters[IRREP_NAMES[t]].append(sign * complex(np.trace(
                    lifted[np.ix_(columns[c], columns[c])])))
    return maps, characters


def isotypic_projectors(alignment, actions, frame, dual):
    """The isotypic projector of each doublet type 2, 2', 2'' of the binary
    tetrahedral group on the three-particle sector of the 18 microscopic-frame
    modes, over the occupation basis (`occupation_basis`):
    P_rho = (2 / 24) sum over 2T of conj(chi_rho) Lambda^3 D
    (`SharpSpin.isotypicProjector`, WP v18 §11.1). They depend on the declared
    host and the aligned frame only, so they are formed once per alignment;
    the record of each carries its rank and its idempotency residual."""
    key = (id(alignment), tuple(alignment["trialities"]))
    if key not in _ISOTYPIC:
        maps, characters = lifted_rotation_maps(alignment, actions, frame,
                                                dual)
        projectors, record = {}, {}
        for name, chi in characters.items():
            projector = np.asarray(obs.SharpSpin.isotypicProjector(
                maps, chi, 2, 3))
            projectors[name] = projector
            record[name] = {
                "rank": int(round(np.trace(projector).real)),
                "idempotency_residual": float(
                    np.linalg.norm(projector @ projector - projector)
                    / max(1.0, np.linalg.norm(projector))),
            }
        _ISOTYPIC[key] = (projectors, record)
    return _ISOTYPIC[key]


# -------------------------------------------------------------- the solve


def _geometric_action(spacetime, kappa, beta, config):
    """The geometric part of the action (matter off), with the declared length
    stiffness and the host's declared reference lengths rather than the
    relaxed ones."""
    return cob.JointAction(
        spacetime, action_declaration(
            spacetime, kappa, beta, config["regge_hinges"],
            matter_weight=0.0,
            reference_lengths=config["reference_lengths"],
            holonomy=config["holonomy"],
            villain_tolerance=declared_tolerance(config, "villain_tolerance"),
            stiffness=config.get("stiffness", DECLARED_STIFFNESS)))


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
    declaration.tolerance = declared_tolerance(config, "fluctuation_tolerance")
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
    restored exactly afterwards. The action value at either point needs log W
    of the Villain term; where the logarithm is refused (a displacement that
    reaches a zero of W, or a W not resolved above its rounding), the value
    is unavailable, and the action remainder is then reported as unmeasured
    with the reason."""
    edges = spacetime.getEdgeList().toVector()
    n = len(edges)
    saved_lengths = [complex(edge.getLength()) for edge in edges]
    saved_phases = [complex(edge.getPhase()) for edge in edges]
    phases = len(induced) == 2 * n

    before = _geometric_action(spacetime, kappa, beta, config)
    reported = before.reported_value()
    s0 = complex(reported.value) if reported.available else None
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
        if s0 is None:
            unmeasured = reported.unavailable
        else:
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


def sector_poles(operator, sector_states, config=None):
    """The compressed operator of a sector, its leakage, and its poles, read
    at the config's pole tolerances (`TOLERANCES`)."""
    dual = left_inverse(sector_states)
    image = operator @ sector_states
    block = dual @ image
    leakage = np.linalg.norm(image - sector_states @ block) / max(
        np.linalg.norm(image), 1e-300)
    centre, radius = contour_of(block)
    pole_config = cob.BoundStatePoleConfig()
    pole_config.newton_tolerance = declared_tolerance(
        config, "pole_newton_tolerance")
    pole_config.zero_count_tolerance = declared_tolerance(
        config, "pole_zero_count_tolerance")
    pole_config.rank_tolerance = declared_tolerance(
        config, "pole_rank_tolerance")
    read = cob.BoundStatePole.poles(block, np.eye(block.shape[0]),
                                    list(range(block.shape[0])), centre,
                                    radius, pole_config)
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


def relax_content(content, kappa, beta, config):
    """Step 1 for one content: a fresh host relaxed to self-consistency as one
    shared base field, the carried density filling the bands of h_1, with the
    declared number of the occupied fiber's power sums pinned at the host."""
    spacetime = build_host(config["edge_squared"], config.get("host_cell"))
    declaration = action_declaration(
        spacetime, kappa, beta, config["regge_hinges"],
        holonomy=config["holonomy"],
        stiffness=config.get("stiffness", DECLARED_STIFFNESS),
        villain_tolerance=declared_tolerance(config, "villain_tolerance"))
    config.setdefault("reference_lengths",
                      list(declaration.reference_lengths))
    action = cob.JointAction(spacetime, declaration)
    mean_field = mean_field_declaration(content, config, spacetime)
    mean_field.fiber_moments = fiber_moment_count(
        mean_field, action,
        config.get("fiber_moments", DECLARED_FIBER_MOMENTS))
    solve = cob.SelfConsistentMeanField(action, mean_field)
    report = solve.solve()
    return spacetime, solve.action, report


class ReadRefused(ValueError):
    """A content's pole read refused by name on the geometry its mean-field
    solve reached: ``name`` is the refusal ("the squared lengths ran off",
    "not Kontsevich-Segal allowable"), the message says why with the numbers
    that decided it, and ``relaxation`` is the solve's record
    (`relaxation_record`)."""

    def __init__(self, name, message, relaxation):
        super().__init__(message)
        self.name = name
        self.relaxation = relaxation


def _method_name(method):
    return {v: k for k, v in MEAN_FIELD_METHODS.items()}[method]


def _band_selection_name(selection):
    return {v: k for k, v in BAND_SELECTIONS.items()}[selection]


def _band_record(band):
    return {"declared_index": int(band.declared_index),
            "occupation": float(band.occupation),
            "rank": int(band.rank),
            "eigenvalues": [complex(v) for v in band.eigenvalues],
            "positions": [int(p) for p in band.positions],
            "declared_positions": [int(p) for p in band.declared_positions],
            "overlap": complex(band.overlap),
            "crossed": bool(band.crossed),
            "ambiguous": bool(band.ambiguous)}


#: The imaginary part of the moment-constrained Hessian's quotient along the
#: Hellmann-Feynman force, relative to the Hessian's own scale, below which
#: the quotient is read as real: the difference-rule Jacobian at radius 1e-4
#: is accurate to about 1e-8 of its entries.
DECLARED_HESSIAN_REALITY = 1e-6


def hessian_sign(value, scale):
    """The sign of the moment-constrained Hessian along the Hellmann-Feynman
    force as a word: "positive" or "negative" when the quotient is real to
    ``DECLARED_HESSIAN_REALITY`` times the Hessian's scale (the real slice),
    "complex" when it is not, and "unread" when it is not a number."""
    value = complex(value)
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        return "unread"
    if abs(value.imag) > DECLARED_HESSIAN_REALITY * scale:
        return "complex"
    return "positive" if value.real > 0 else "negative"


def relaxation_record(report):
    """What a mean-field solve reached and how, as every content record
    carries it: the method and band selection that ran, whether the fixed
    point was reached, why the solve stopped (by name, with its detail), the
    iterations, the final force, the joint Jacobian's rank and rank gap at the
    end point, the Kontsevich-Segal margin and the growth of the lengths
    there, the occupied bands followed to the end point, the pinned moments of
    the occupied fiber (their number, targets, multipliers and residuals, in
    the operator's own unit), the moment-constrained action's Hessian on the
    range of the Hellmann-Feynman force with its sign (WP v17 line 265), and a
    per-iterate trace of the force, the covariance change, the pinned
    moments' residual and multipliers, every occupied band's overlap and
    places in the ascending real-part order, any crossing, and the Newton step
    taken from the iterate."""
    trace = []
    for step in report.steps:
        entry = {
            "iteration": int(step.iteration),
            "force_norm": float(step.force_norm),
            "covariance_change": float(step.covariance_change),
            "band_overlaps": [complex(b.overlap) for b in step.bands],
            "band_positions": [[int(p) for p in b.positions]
                               for b in step.bands],
            "band_crossing": bool(step.band_crossing),
            "band_isolation": float(step.band_isolation),
            "moment_residual_norm": float(step.moment_residual_norm),
            "multipliers": [complex(x) for x in step.multipliers],
            "stop_reason": cob.relaxation_stop_name(step.geometry_stop_reason),
        }
        if step.newton_iterated:
            newton = step.newton
            entry["newton"] = {
                "accepted": bool(newton.accepted),
                "damping": float(newton.damping),
                "step_norm": float(newton.step_norm),
                "jacobian_rank": int(newton.jacobian_rank),
                "rank_gap": float(newton.rank_gap),
                "linear_residual": float(newton.linear_residual),
                "constrained_step": bool(newton.constrained_step),
                "constrained_rank": int(newton.constrained_rank),
                "constrained_rank_gap": float(newton.constrained_rank_gap),
                "residual_test_dampings": int(newton.residual_test_dampings),
                "sector_guard_dampings": int(newton.sector_guard_dampings),
                "domain_guard_dampings": int(newton.domain_guard_dampings),
                "zero_guard_dampings": int(newton.zero_guard_dampings),
            }
        else:
            entry["geometry_residual_norm"] = float(
                step.geometry_residual_norm)
            entry["geometry_converged"] = bool(step.geometry_converged)
            if step.geometry_stop_detail:
                entry["stop_detail"] = step.geometry_stop_detail
        trace.append(entry)
    return {
        "method": _method_name(report.method),
        "band_selection": _band_selection_name(report.band_selection),
        "converged": bool(report.converged),
        "stop_reason": cob.relaxation_stop_name(report.stop_reason),
        "stop_detail": report.stop_detail,
        "iterations": int(report.iterations),
        "force_norm": float(report.force_norm),
        "covariance_change": float(report.covariance_change),
        "purity_defect": float(report.purity_defect),
        "spectral_gap": float(report.spectral_gap),
        "band_isolation": float(report.band_isolation),
        "band_ranks": [int(r) for r in report.band_ranks],
        "bands": [_band_record(b) for b in report.bands],
        "band_crossing_iterates": int(report.band_crossing_iterates),
        "lowest_band_overlap": float(report.lowest_band_overlap),
        "joint_jacobian": {
            "size": int(report.jacobian_size),
            "rank": int(report.jacobian_rank),
            "largest_singular_value": float(report.largest_singular_value),
            "smallest_retained_singular_value": float(
                report.smallest_retained_singular_value),
            "largest_discarded_singular_value": float(
                report.largest_discarded_singular_value),
            "rank_gap": float(report.rank_gap),
        },
        "kontsevich_segal_margin": float(report.kontsevich_segal_margin),
        "largest_length_ratio": float(report.largest_length_ratio),
        "fiber_rank": int(report.fiber_rank),
        "fiber_moments": len(report.moment_targets),
        "moment_scale": float(report.moment_scale),
        "moment_targets": [complex(x) for x in report.moment_targets],
        "multipliers": [complex(x) for x in report.multipliers],
        "moment_residuals": [complex(x) for x in report.moment_residuals],
        "hellmann_feynman_force_norm": float(
            report.hellmann_feynman_force_norm),
        "force_hessian": complex(report.force_hessian),
        "force_hessian_scale": float(report.force_hessian_scale),
        "force_hessian_sign": hessian_sign(report.force_hessian,
                                           report.force_hessian_scale),
        "action": complex(report.action),
        "action_available": bool(report.action_available),
        "action_unavailable": report.action_unavailable,
        "occupied_energy": complex(report.occupied_energy),
        "zero_guard_damped_steps": int(report.zero_guard_damped_steps),
        "trace": trace,
    }


def read_refusal(report, tolerance=DECLARED_ALLOWABILITY_TOLERANCE):
    """The name and the message of the refusal of a pole read on the geometry
    a mean-field solve reached, or None when the read may proceed. The read is
    refused when the squared lengths ran off (the low eigenvalues there are
    h_1 ~ 1/z at infinite length, not a bound state) and when the geometry is
    not Kontsevich-Segal allowable (bands are read on the allowable side,
    WP v17 line 151; a margin within ``tolerance`` of zero is the
    boundary). A solve that stopped for another reason is read,
    and its record says why it stopped."""
    if report.stop_reason == cob.RelaxationStop.LengthRunaway:
        return ("the squared lengths ran off",
                "the pole read is refused: the squared lengths ran off (%s)"
                % report.stop_detail)
    margin = float(report.kontsevich_segal_margin)
    if not margin > tolerance:
        return ("not Kontsevich-Segal allowable",
                "the pole read is refused: the geometry the mean-field "
                "solve reached is not Kontsevich-Segal allowable (margin "
                "%.3g, at or below the declared tolerance %.0e), and bands "
                "are read on the allowable side (WP v17 line 151)"
                % (margin, tolerance))
    return None


def _term(action, name):
    """One term of the joint action for the record, or its refusal by name:
    only the holonomy term needs log W, and a refused logarithm makes that
    reported value unavailable without ending the read."""
    try:
        return complex(getattr(action, name + "_term")())
    except ValueError as error:
        return {"unavailable": str(error)}


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


#: What a content names under ruling (a), recorded with every content.
CONTENT_MEANING = (
    "n_b quarks in band b of the covariant operator h_1 (bands of degenerate "
    "eigenvalues in ascending order of real part, "
    "OccupationOrder.AscendingRealPart); the carried density is "
    "Gamma = sum_b (n_b / r_b) P_b over the Riesz projectors P_b of h_1. On "
    "the monopole host the bands of h_1 are simple on each sheet, so each "
    "band is one mode per sheet (rank 3), not a spin doublet; how the "
    "occupied bands decompose into the doublets 2, 2', 2'' of the T-averaged "
    "operator is recorded as 'spin_decomposition'")


def band_projectors(operator, tolerance):
    """The bands of an operator as the band rule groups them: eigenvalues in
    ascending order of real part, consecutive ones within ``tolerance``
    relative to their size in one band, and each band's Riesz projector
    V_b (V^-1)_b. Returns (eigenvalues, projector) per band."""
    values, vectors = np.linalg.eig(operator)
    order = sorted(range(len(values)), key=lambda k: values[k].real)
    inverse = np.linalg.inv(vectors)
    groups, current = [], [order[0]]
    for previous, k in zip(order, order[1:]):
        if abs(values[k] - values[previous]) > tolerance * max(
                1.0, abs(values[previous])):
            groups.append(current)
            current = [k]
        else:
            current.append(k)
    groups.append(current)
    return [(values[g], vectors[:, g] @ inverse[g, :]) for g in groups]


def doublet_projectors(frame, dual, trialities):
    """The projector onto each doublet 2, 2', 2'' of the T-averaged operator,
    times the sheets: the isotypic components of the declared projective
    action D_1(g), read off the aligned frame (columns (2 c + s) * 3 + t of
    carrier c), keyed by the doublet's Z_3 label."""
    out = {}
    for c, t in enumerate(trialities):
        columns = [(2 * c + s) * SHEETS + sheet for s in range(2)
                   for sheet in range(SHEETS)]
        out[IRREP_NAMES[t]] = frame[:, columns] @ dual[columns, :]
    return out


def spin_decomposition(carrier, covariance, content, frame, dual, trialities,
                       tolerance):
    """The spin content of the occupied state, read on the T-averaged operator
    (WP v17 §9 line 506) after the covariance was built from h_1 (ruling (a)).

    * ``occupied_bands``: for every band b of h_1 the content occupies, its
      rank r_b, its eigenvalues, and its overlap with each doublet d of
      h-bar_1, tr(P_b Pbar_d) / r_b (complex, because P_b is a Riesz
      projector of a non-normal operator; the three overlaps sum to one);
    * ``occupied_state``: the number of quarks in each doublet,
      tr(Gamma Pbar_d), which sums to the number of quarks;
    * ``commutator``: ||[Gamma, h_1]|| / (||Gamma|| ||h_1||), zero to rounding
      when Gamma is built from the bands of h_1."""
    bands = band_projectors(carrier, tolerance)
    doublets = doublet_projectors(frame, dual, trialities)
    occupied = []
    for b, n in enumerate(content):
        if n == 0 or b >= len(bands):
            continue
        values, projector = bands[b]
        rank = len(values)
        occupied.append({
            "band": b, "occupation": int(n), "rank": rank,
            "eigenvalues": [complex(v) for v in values],
            "overlap": {name: complex(np.trace(projector @ d) / rank)
                        for name, d in doublets.items()},
        })
    commutator = covariance @ carrier - carrier @ covariance
    return {
        "band_ranks": [len(v) for v, _ in bands],
        "occupied_bands": occupied,
        "occupied_state": {name: complex(np.trace(covariance @ d))
                           for name, d in doublets.items()},
        "commutator": float(np.linalg.norm(commutator)
                            / (np.linalg.norm(covariance)
                               * np.linalg.norm(carrier))),
    }


_DOUBLET_SECTORS = {}


def doublet_sectors(doublet_content, trialities):
    """The colour-singlet, one-quark-per-sheet spin sectors of a doublet
    content (n_2, n_2', n_2''): the carrier content in the aligned frame's
    carrier order, the total triality, and the sectors by total spin. They
    depend on the declared frame only, so they are formed once."""
    carrier_of = {IRREP_NAMES[t]: c for c, t in enumerate(trialities)}
    if sorted(carrier_of.values()) != [0, 1, 2]:
        raise RuntimeError("the aligned frame's trialities %s are not the "
                           "three Z_3 characters" % (trialities,))
    carrier_content = [0, 0, 0]
    for name, n in zip(IRREP_NAMES, doublet_content):
        carrier_content[carrier_of[name]] = int(n)
    key = tuple(carrier_content)
    if key not in _DOUBLET_SECTORS:
        states, _ = singlet_states(carrier_content)
        _DOUBLET_SECTORS[key] = spin_sectors(states)[0]
    triality = sum(n * t for n, t in zip(carrier_content, trialities)) % 3
    return carrier_content, triality, _DOUBLET_SECTORS[key]


def pole_certificates(target, j2, sector, dual, eigen, basis, spins,
                      projector=None, spinor_type=None,
                      tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The certificates of the eigenvector of a sector's compressed block
    nearest ``target``: the right eigenvector and its left partner, as
    vectors over the occupation basis and as Fock vectors.

    * the spinor certificate of WP v18 §11.1 and §14: the isotypic projector
      equations (I - P)|Psi_R> = 0 and <Psi_L|(I - P) = 0 with ``projector``
      the projector onto the sector's 2T types (`SharpSpin.isotypicRead`),
      named ``spinor_type``; unmeasured when no projector is supplied;
    * the spin-lift read: `SharpSpin.read` of the sector's j(j+1) under the
      constructed SU(2) action, which states the continuum spin value an
      accepted lift supplies;
    * the relative residual of the colour Casimir on the right vector (zero
      for a colour singlet).

    ``eigen`` holds the right and the left eigen-decompositions of the block,
    ``dual`` the sector's bilinear left inverse."""
    values, right, values_left, left = eigen
    k = int(np.argmin(np.abs(values - target)))
    kl = int(np.argmin(np.abs(values_left - values[k])))
    right_sector = sector @ right[:, k]
    left_sector = dual.T @ left[:, kl]
    right_state = to_fock(right_sector, basis)
    left_state = to_fock(left_sector, basis)
    spin = obs.SharpSpin.read(spins, right_state, left_state, j2, tolerance)
    casimir = np.linalg.norm(colour_casimir(right_state)) / \
        np.linalg.norm(right_state)
    out = {
        "sharp_spinor": None,
        "spinor_type": spinor_type,
        "spinor_right_residual": None,
        "spinor_left_residual": None,
        "spinor_weight": None,
        "spin_lift_sharp": bool(spin.sharp),
        "spin_lift_right_residual": float(spin.right_residual),
        "spin_lift_left_residual": float(spin.left_residual),
        "spin_lift_expectation": complex(spin.expectation),
        "determinant_count": int(spin.determinant_count),
        "colour_casimir_residual": float(casimir),
    }
    if projector is not None:
        spinor = obs.SharpSpin.isotypicRead(projector, right_sector,
                                            left_sector, spinor_type or "",
                                            tolerance)
        out.update({
            "sharp_spinor": bool(spinor.sharp),
            "spinor_right_residual": float(spinor.right_residual),
            "spinor_left_residual": float(spinor.left_residual),
            "spinor_weight": complex(spinor.weight),
        })
    return out


def sector_entry(j2, triality, sector, operators, projectors=None,
                 config=None):
    """One spin sector's reads: its restriction to 2T, the weight of each
    isotypic type in it (``projectors``, `isotypic_projectors`: the trace of
    the type's projector compressed to the sector over the sector's
    dimension, which is one on the sector's own types and zero on the others
    when the constructed lift agrees with the finite group), and for each
    named many-body operator the poles of the compressed block, each pole's
    spinor, spin-lift and colour certificates (``pole_certificates``,
    parallel to ``poles``), and the lowest pole's certificates repeated
    beside it. The poles and the certificates are read at the config's
    tolerances (`TOLERANCES`)."""
    basis = occupation_basis()
    spins = edge_spin_matrices()
    irreps = restriction(j2, triality)
    certificate_tolerance = declared_tolerance(config, "certificate_tolerance")
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
    dual = left_inverse(sector)
    spinor_type = " + ".join(irreps)
    projector = None
    if projectors is not None:
        projector = sum(projectors[name] for name in irreps)
        entry["isotypic_weights"] = {
            name: complex(np.trace(dual @ p @ sector) / sector.shape[1])
            for name, p in projectors.items()}
        entry["spinor_type"] = spinor_type
    for name, operator in operators:
        block, leakage, read = sector_poles(operator, sector, config)
        poles = [complex(p) for p in read.poles]
        lowest = min(poles, key=lambda p: (p.real, p.imag)) if poles else None
        # every pole's right and left eigenvectors, for its spin and colour
        # certificates
        values, right = np.linalg.eig(block)
        values_left, left = np.linalg.eig(block.T)
        eigen = (values, right, values_left, left)
        certificates = [pole_certificates(p, j2, sector, dual, eigen, basis,
                                          spins, projector, spinor_type,
                                          certificate_tolerance)
                        for p in poles]
        # with no pole read, the certificates beside the lowest pole are
        # those of the block's first eigenvector
        lowest_certificates = (
            certificates[poles.index(lowest)] if lowest is not None else
            pole_certificates(values[0], j2, sector, dual, eigen, basis,
                              spins, projector, spinor_type,
                              certificate_tolerance))
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
            "pole_certificates": certificates,
        }
        entry[name].update(lowest_certificates)
    return entry


def evaluate_content(content, kappa, beta, config, alignment):
    """One content at one scan point: relaxation, recursion, quark conditions,
    states, spin, operators and poles."""
    started = time.time()
    declared_actions = rotation_action([monopole_support()] * SHEETS)
    spacetime, action, report = relax_content(content, kappa, beta, config)
    solve = relaxation_record(report)
    refusal = read_refusal(report,
                           declared_tolerance(config, "allowability_tolerance"))
    if refusal is not None:
        raise ReadRefused(refusal[0], refusal[1], solve)
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
    trialities = alignments[0]["trialities"]
    spin = spin_decomposition(carrier, matrix(action.declaration.covariance),
                              content, frame, dual, trialities,
                              config["band_tolerance"])
    # the isotypic projectors of the double cover on the three-particle
    # sector, the spinor certificate's projectors (WP v18 §11.1, §14)
    projectors, isotypic = isotypic_projectors(alignment, declared_actions,
                                               frame, dual)

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
        declaration.tolerance = declared_tolerance(config,
                                                   "fluctuation_tolerance")
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

    # the poles of the colour-singlet three-quark sectors of h-bar_1, for
    # every doublet content (n_2, n_2', n_2''): a content of this driver names
    # occupations of the bands of h_1, which are not doublets, so the sectors
    # are read for every doublet content and labelled by it
    doublet_reads = []
    for doublet_content in contents():
        sector_reads = {}
        carrier_content, triality, sectors = doublet_sectors(doublet_content,
                                                             trialities)
        for j2, sector in sectors.items():
            sector_reads[j2] = sector_entry(j2, triality, sector, (
                ("quasi_free", quasi_free), ("with_quartic", with_quartic)),
                projectors, config)
        doublet_reads.append({
            "doublet_content": list(doublet_content),
            "carrier_content": carrier_content,
            "total_triality": int(triality),
            "sectors": {str(k): v for k, v in sector_reads.items()},
        })

    certificate_tolerance = declared_tolerance(config, "certificate_tolerance")
    recursion = recursion_read(spacetime, config)
    anchor = anchor_atlas_read(spacetime, alignment, certificate_tolerance)
    fingerprint = spectral_fingerprint_read(spacetime, kappa, beta, config)
    quark = quark_conditions(spacetime, alignment, recursion,
                             averaged_residual, report, anchor, fingerprint,
                             certificate_tolerance)
    truncation_read = action.holonomy_truncation()
    record = {
        "content": list(content),
        "content_meaning": CONTENT_MEANING,
        "holonomy": config["holonomy"],
        "elimination": config["elimination"],
        "seconds": time.time() - started,
        "relaxation": dict(solve, **{
            "band_operator": "h_1 (the covariant operator itself)",
            "shared_sheet_geometry": True,
            "stiffness": config.get("stiffness", DECLARED_STIFFNESS),
            "kappa_role": config.get("kappa_role"),
            "terms": {name: _term(action, name)
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
            "link_force_norm": float(np.linalg.norm(
                action.link_stationarity())),
            "ward_current_divergence": float(np.max(np.abs(np.asarray(
                action.ward_current_divergence())))),
            "holonomy_zero_distance": float(action.holonomy_zero_distance()),
        }),
        "covariant_spectrum": sorted(
            [complex(v) for v in np.linalg.eigvals(carrier)],
            key=lambda v: (v.real, v.imag)),
        "averaged_spectrum": sorted(
            [complex(v) for v in np.linalg.eigvals(averaged)],
            key=lambda v: (v.real, v.imag)),
        "band_energies": band_energies,
        "trialities": trialities,
        "spin_decomposition": spin,
        "isotypic_projectors": isotypic,
        "symmetry_residual": covariant_residual,
        "averaged_symmetry_residual": averaged_residual,
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
        "doublet_reads": doublet_reads,
        "recursion": recursion,
        "anchor": anchor,
        "spectral_fingerprint": fingerprint,
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
    declaration.tolerance = declared_tolerance(config, "recursion_tolerance")
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
        # leakage is coupling between distinct components: the block M_vv of
        # a component with itself is its own fiber operator, not a leak
        "transport_norms": [float(np.linalg.norm(np.asarray(t.block)))
                            for t in level.transports
                            if t.from_component != t.to_component],
        "fock_stage_dimension": float(level.fock_stage_dimension),
        "determinant_residual": float(level.determinant_residual),
        "certificate": level.certificate.describe(),
        "reduction_certificate": level.reduction_certificate.describe(),
    }


#: The base vertex of the anchor atlas's path rule: breadth-first walks from
#: the cell's lowest vertex, inside the cell.
ANCHOR_BASE_POINT = 0


def anchor_atlas_read(spacetime, alignment,
                      tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The anchor atlas of the quark fiber's base band on the relaxed host,
    quark condition 3 (WP v18 §10): the dressed anchor
    (`chainhodge.DressedAnchor`) of the reference doublet of the aligned
    frame, the j = 1/2 doublet of the odd-monopole support (rank 2), on every
    sheet's relaxed connection over the atlas of the cell's four faces, with
    the breadth-first path rule from `ANCHOR_BASE_POINT`.

    Per sheet: the Lambda^2 coordinates of every face (the three maximal
    minors of the restricted frame), the number of anchoring faces, the
    connection-dressed covariance residual against the verification gauge,
    the determinant-line transition cocycle residual, the named refusals,
    and the invariant coordinates alpha_tau of the faces from the sheet's
    Whitney chain Hodge operator and the band's geometric images
    (`withInvariantCoordinates`), or why they were not attached. The summary
    takes the worst sheet: the read is anchored only when every sheet
    anchors, and carries the largest residuals."""
    fixture = monopole_support()
    complex_ = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
    canonical = [tuple(int(v) for v in e)
                 for e in complex_.kSimplexVertices(1)]
    fixture_edges = [tuple(int(v) for v in e) for e in fixture.edges]
    # the fixture's edge order carried to the complex's canonical one
    order = [fixture_edges.index(e) for e in canonical]
    faces = list(range(complex_.numSimplices(2)))
    paths = ch.DeclaredPaths.breadthFirst(complex_, ANCHOR_BASE_POINT)
    reference = int(alignment["reference_carrier"])
    frame = np.asarray(alignment["frame"])[:, 2 * reference:
                                           2 * reference + 2]
    phi = frame[order, :]
    sheets = []
    for t in range(SHEETS):
        links = [complex(u) for u in sheet_links(spacetime, t)]
        squared = [complex(z) for z in sheet_squared_lengths(spacetime, t)]
        connection = ch.Connection(complex_, [links[k] for k in order])
        read = ch.DressedAnchor.profile(complex_, connection, paths, faces,
                                        phi, tolerance)
        unattached = None
        try:
            base = ch.ChainHodge(complex_, [squared[k] for k in order],
                                 ch.Preset.L2)
            covariant = ch.CovariantChainHodge(base, connection)
            images = np.asarray(covariant.applyG(1, phi))
            dual_images = np.asarray(covariant.dual().applyG(1, phi))
            read = ch.DressedAnchor.withInvariantCoordinates(
                read, covariant, dual_images, images)
        except (ValueError, RuntimeError) as error:
            unattached = str(error)
        sheets.append({
            "anchored": bool(read.anchored),
            "anchoring_faces": int(read.anchoringFaces),
            "coordinates": [[complex(c) for c in row]
                            for row in read.coordinates],
            "invariant_coordinates": [complex(a) for a in
                                      read.invariantCoordinates],
            "invariant_coordinates_unattached": unattached,
            "coordinate_scale": float(read.coordinateScale),
            "covariance_residual": float(read.covarianceResidual),
            "transition_cocycle_residual": float(
                read.transitionCocycleResidual),
            "failed_certificates": list(read.failedCertificates),
        })
    return {
        "band": ("the reference doublet of the aligned frame, the j = 1/2 "
                 "doublet of the odd-monopole support, rank 2"),
        "base_point": ANCHOR_BASE_POINT,
        "path_rule": "breadth-first walks from the base point inside the "
                     "cell",
        "faces": [[int(v) for v in f] for f in complex_.kSimplexVertices(2)],
        "tolerance": tolerance,
        "anchored": all(s["anchored"] for s in sheets),
        "anchoring_faces": min(s["anchoring_faces"] for s in sheets),
        "covariance_residual": max(s["covariance_residual"] for s in sheets),
        "transition_cocycle_residual": max(
            s["transition_cocycle_residual"] for s in sheets),
        "invariant_coordinates_attached": all(
            s["invariant_coordinates_unattached"] is None for s in sheets),
        "failed_certificates": sorted({f for s in sheets
                                       for f in s["failed_certificates"]}),
        "sheets": sheets,
    }


def anchor_evidence(anchor):
    """The evidence of quark condition 3 from an anchor atlas read
    (`anchor_atlas_read`): the profile is not identically zero, the
    connection-dressed covariance holds, and the transition cocycle holds
    over at least two anchoring faces, each against the read's tolerance;
    stability across frames needs several cobordism frames and is not
    measured at one level. Without a read, every item is unmeasured."""
    E = obs.QuarkConditionEvidence
    if not anchor:
        return [E("anchor-profile-nonzero", None,
                  "the anchor atlas was not read"),
                E("anchor-covariance", None, "not read"),
                E("anchor-transitions", None, "not read"),
                E("anchor-stable-across-frames", None,
                  "a single level spans one cobordism frame")]
    tolerance = anchor["tolerance"]
    faces = [s["anchoring_faces"] for s in anchor["sheets"]]
    refused = ("; refused: " + ", ".join(anchor["failed_certificates"])
               if anchor["failed_certificates"] else "")
    return [
        E("anchor-profile-nonzero", bool(anchor["anchored"]),
          "%s; anchoring faces per sheet %s of %d%s"
          % (anchor["band"], faces, len(anchor["faces"]), refused)),
        E("anchor-covariance",
          bool(anchor["covariance_residual"] <= tolerance),
          "connection-dressed covariance residual %.3g against the "
          "verification gauge (tolerance %.0e)"
          % (anchor["covariance_residual"], tolerance)),
        E("anchor-transitions",
          bool(anchor["anchoring_faces"] >= 2
               and anchor["transition_cocycle_residual"] <= tolerance),
          "determinant-line transition cocycle residual %.3g over %d "
          "anchoring faces%s"
          % (anchor["transition_cocycle_residual"],
             anchor["anchoring_faces"],
             "" if anchor["invariant_coordinates_attached"]
             else "; invariant coordinates not attached")),
        E("anchor-stable-across-frames", None,
          "a single level spans one cobordism frame"),
    ]


def anchor_text(anchor):
    """One content's anchor atlas as text."""
    if not anchor:
        return "anchor atlas unread"
    return ("anchor atlas %s (%d of %d faces anchor on every sheet, "
            "covariance residual %.2g, transition cocycle residual %.2g%s%s)"
            % ("anchored" if anchor["anchored"] else "refused",
               anchor["anchoring_faces"], len(anchor["faces"]),
               anchor["covariance_residual"],
               anchor["transition_cocycle_residual"],
               "" if anchor["invariant_coordinates_attached"]
               else ", invariant coordinates not attached",
               "; " + ", ".join(anchor["failed_certificates"])
               if anchor["failed_certificates"] else ""))


#: The declared relabeling of quark condition 7: the transposition of the
#: cell's two lowest vertices, an odd permutation that no rotation of the
#: tetrahedron supplies, so that the check is of the code's independence of
#: the labels and not of a symmetry.
FINGERPRINT_RELABELING = (1, 0, 2, 3)
#: The floor on the fiber's subspace overlap with its re-extraction on the
#: refined cell (the cluster pipeline's minRefinementOverlap).
DECLARED_REFINEMENT_OVERLAP = 0.9
#: The local id of the centre vertex of a refined cell: above the four
#: vertices, so that the base vertex b(e) = min e of every original edge, and
#: with it the dressing of the operator on those edges, is unchanged.
REFINED_CENTRE = 4
#: The edges of a refined cell in the chain complex's canonical order.
REFINED_EDGES = tuple(itertools.combinations(range(5), 2))
#: The edges of a cell in the same order.
PAIRS = tuple(itertools.combinations(range(4), 2))


def centroid_squared_lengths(squared):
    """|v_i - c|^2 for the centroid c of the flat tetrahedron with the six
    squared edge lengths ``squared`` (the order of `MonopoleSupport.edges`):
    (1/4) sum_j d_ij^2 - (1/16) sum_{j<k} d_jk^2, the identity for the
    centroid of four points. On the regular tetrahedron of squared edge
    length 8 every value is 3, the squared circumradius."""
    d = {}
    for (a, b), z in zip(PAIRS, squared):
        d[(a, b)] = d[(b, a)] = complex(z)
    total = sum(complex(z) for z in squared)
    return [sum(d[(i, j)] for j in range(4) if j != i) / 4.0 - total / 16.0
            for i in range(4)]


def refined_host(spacetime):
    """The relaxed host with every sheet's tetrahedron refined by one stellar
    subdivision at its centroid: the declared refinement of quark condition
    7. The centre of sheet t is vertex 4 SHEETS + t, above every other id, so
    the base vertex b(e) = min e of every original edge is unchanged; the four
    new edges carry the centroid's squared lengths of the flat cell
    (`centroid_squared_lengths`) and the trivial link, so every boundary face
    and its holonomy are unchanged and the interior faces carry the boundary
    holonomies. Returns the spacetime and, per sheet, the ten squared lengths
    and links in the order of `REFINED_EDGES`."""
    cells, data = [], []
    for t in range(SHEETS):
        centre = 4 * SHEETS + t
        for i in range(4):
            cells.append([centre] + [4 * t + k for k in range(4) if k != i])
        squared = sheet_squared_lengths(spacetime, t)
        links = sheet_links(spacetime, t)
        radii = centroid_squared_lengths(squared)
        z, u = {}, {}
        for (a, b), value, link in zip(PAIRS, squared, links):
            z[(a, b)] = complex(value)
            u[(a, b)] = complex(link)
        for i in range(4):
            z[(i, REFINED_CENTRE)] = radii[i]
            u[(i, REFINED_CENTRE)] = 1.0 + 0j
        data.append({"squared_lengths": [z[e] for e in REFINED_EDGES],
                     "links": [u[e] for e in REFINED_EDGES]})
    refined = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    for edge in refined.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        if source >= 4 * SHEETS or target >= 4 * SHEETS:
            t = (source if source >= 4 * SHEETS else target) - 4 * SHEETS
        else:
            t = source // 4
        local = tuple(REFINED_CENTRE if v >= 4 * SHEETS else v - 4 * t
                      for v in (source, target))
        key = (min(local), max(local))
        value = data[t]["squared_lengths"][REFINED_EDGES.index(key)]
        link = data[t]["links"][REFINED_EDGES.index(key)]
        if local[0] > local[1]:
            link = 1.0 / link
        edge.setLength(cmath.sqrt(value))
        edge.setPhase(complex(-1j * cmath.log(link)))
    return refined, data


def refined_support(sheet_data):
    """The `MonopoleSupport` of one refined sheet: five vertices, the ten
    edges of `REFINED_EDGES`, the four boundary faces of the fixture in their
    outward orientation (the bounding cut, which the subdivision leaves as it
    was), and the U(1) part of the sheet's links."""
    fixture = monopole_support()
    return obs.MonopoleSupport(
        5, [list(e) for e in REFINED_EDGES], [list(f) for f in fixture.faces],
        obs.MonopoleSupport.u1Part([complex(u) for u in sheet_data["links"]]))


def refined_rotation_group():
    """The twelve rotations of the refined cell: those of the tetrahedron,
    fixing the centre."""
    return [list(g) + [REFINED_CENTRE] for g in rotation_group()]


def relabeled_host(spacetime, permutation):
    """The host with every sheet's vertices relabeled by ``permutation``."""
    cells = [[4 * t + k for k in range(4)] for t in range(SHEETS)]
    host = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    data = []
    for t in range(SHEETS):
        z, u = {}, {}
        for (a, b), value, link in zip(PAIRS,
                                       sheet_squared_lengths(spacetime, t),
                                       sheet_links(spacetime, t)):
            image = (permutation[a], permutation[b])
            key = (min(image), max(image))
            z[key] = complex(value)
            u[key] = (complex(link) if image[0] < image[1]
                      else 1.0 / complex(link))
        data.append((z, u))
    for edge in host.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        t = source // 4
        a, b = source - 4 * t, target - 4 * t
        z, u = data[t]
        link = u[(min(a, b), max(a, b))]
        if a > b:
            link = 1.0 / link
        edge.setLength(cmath.sqrt(z[(min(a, b), max(a, b))]))
        edge.setPhase(complex(-1j * cmath.log(link)))
    return host


def conjugated_group(group, permutation):
    """s g s^-1 for every g, as vertex permutations."""
    inverse = [0] * len(permutation)
    for x, y in enumerate(permutation):
        inverse[y] = x
    out = []
    for g in group:
        out.append([permutation[g[inverse[y]]]
                    for y in range(len(permutation))])
    return out


def _bands_of(support, group, tolerance=DECLARED_TOLERANCE):
    """The bands of the rotation-averaged edge Laplacian of a support, as
    orthonormal blocks in ascending order of eigenvalue."""
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), group))
    values, vectors = np.linalg.eigh(averaged)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    bands, start = [], 0
    for k in range(1, len(values) + 1):
        if k == len(values) or abs(values[k] - values[start]) > tolerance:
            bands.append((float(values[start]), vectors[:, start:k]))
            start = k
    return bands


def _subspace_overlap(a, b):
    """(sum_i cos^2 theta_i) / max(rank a, rank b) over the principal angles
    of two frames on the same cells: one exactly when their column spans
    coincide (the reading of `SpectralFiber::overlap`)."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    cosines = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    return float(np.sum(cosines ** 2) / max(a.shape[1], b.shape[1]))


def _carrier(host, kappa, beta, config):
    """h_1 of a host under the run's declared action, as a matrix."""
    declaration = action_declaration(
        host, kappa, beta, config["regge_hinges"], holonomy=config["holonomy"],
        stiffness=config.get("stiffness", DECLARED_STIFFNESS),
        villain_tolerance=declared_tolerance(config, "villain_tolerance"))
    return matrix(cob.JointAction(host, declaration).carrier_operator())


def _fiber_fingerprint(host, support, group, edges, kappa, beta, config):
    """The label-free fingerprint of sheet 0 of a host: the spectrum of h_1,
    the spectrum of its T-average, and the reference doublet (the coexact
    rank-two spinorial band of the averaged combinatorial Laplacian, the
    band the aligned frame's reference carrier is) with its energy on the
    averaged h_1 and its weight on every edge."""
    degeneracy_tolerance = declared_tolerance(config, "degeneracy_tolerance")
    read = support.spinRead(group, degeneracy_tolerance,
                            declared_tolerance(config, "certificate_tolerance"))
    carrier = _carrier(host, kappa, beta, config)
    block = carrier[:edges, :edges]
    actions = [np.asarray(support.edgeRepresentation(g)) for g in group]
    averaged = rotation_averaged(block, actions)
    bands = _bands_of(support, group, degeneracy_tolerance)
    index = int(read.doublet_index)
    doublet = bands[index][1] if index < len(bands) else None
    out = {
        "spectrum": sorted((complex(v) for v in np.linalg.eigvals(block)),
                           key=lambda v: (v.real, v.imag)),
        "averaged_spectrum": sorted(
            (complex(v) for v in np.linalg.eigvals(averaged)),
            key=lambda v: (v.real, v.imag)),
        "monopole_number": int(read.monopole.monopole_number),
        "doublet_found": (bool(read.half_integer_doublet)
                          and doublet is not None and doublet.shape[1] == 2),
    }
    if out["doublet_found"]:
        out["doublet_energy"] = complex(
            np.trace(doublet.conj().T @ averaged @ doublet) / 2)
        out["doublet_weights"] = [float(w.real) for w in
                                  np.diag(doublet @ doublet.conj().T)]
        out["doublet_frame"] = doublet
        out["averaged"] = averaged
        out["actions"] = actions
    return out


def spectral_fingerprint_read(spacetime, kappa, beta, config,
                              tolerance=None,
                              overlap_floor=DECLARED_REFINEMENT_OVERLAP):
    """Quark condition 7 (WP v18 §10) on the relaxed host: the spectral
    fingerprint of the fiber re-read under the declared relabeling and the
    declared refinement, without re-solving the mean field. The sheets are
    certified isomorphic (quark condition 2), so sheet 0 is read.

    The fingerprint (`_fiber_fingerprint`) is the spectrum of h_1, the
    spectrum of its T-average, and the reference doublet (the coexact
    rank-two spinorial band of the averaged combinatorial edge Laplacian,
    the band the aligned frame's reference carrier is) with its energy on
    the averaged h_1 and the weight it places on every edge, which is
    invariant under the gauge a relabeling induces through the dressing
    convention b(e) = min e.

    * Relabeling (`FINGERPRINT_RELABELING`): the host is rebuilt with its
      vertices relabeled (`relabeled_host`), the rotations conjugated, and
      the fingerprint re-read; the two spectra agree as multisets, and the
      doublet's energy and edge weights agree on the matched edges, each to
      ``tolerance``.
    * Refinement (`refined_host`): h_1 is formed on the refined host at the
      relaxed geometry and averaged with the refined support's projective
      action; the doublet's type is re-extracted as the band of the averaged
      operator, inside the isotypic component of that type in the refined
      action (`SharpSpin.isotypicProjector`), whose column span overlaps the
      original doublet most on the six shared edges, the refinement
      continuation. The overlap must reach ``overlap_floor`` with the rank
      preserved, and the refined support must carry a spinor doublet; the
      energy shift is reported beside them."""
    group = rotation_group()
    if tolerance is None:
        tolerance = declared_tolerance(config, "certificate_tolerance")
    support, _ = sheet_support(spacetime, 0)
    original = _fiber_fingerprint(spacetime, support, group, BASE_EDGES,
                                  kappa, beta, config)
    result = {"sheet": 0, "spectrum": original["spectrum"],
              "doublet_found": original["doublet_found"],
              "tolerance": tolerance}
    if not original["doublet_found"]:
        unread = {"held": False, "unread": "no spinor doublet on the host"}
        result["relabeling"] = dict(unread)
        result["refinement"] = dict(unread)
        return result
    result["doublet_energy"] = original["doublet_energy"]
    result["doublet_weights"] = original["doublet_weights"]
    # --- relabeling
    permutation = list(FINGERPRINT_RELABELING)
    moved_host = relabeled_host(spacetime, permutation)
    moved_support, _ = sheet_support(moved_host, 0)
    moved = _fiber_fingerprint(moved_host, moved_support,
                               conjugated_group(group, permutation),
                               BASE_EDGES, kappa, beta, config)
    scale = max(1.0, max(abs(v) for v in original["spectrum"]))
    spectrum_shift = max(abs(a - b) for a, b in
                         zip(original["spectrum"], moved["spectrum"])) / scale
    averaged_shift = max(abs(a - b) for a, b in
                         zip(original["averaged_spectrum"],
                             moved["averaged_spectrum"])) / scale
    relabeling = {"permutation": permutation,
                  "spectrum_shift": float(spectrum_shift),
                  "averaged_spectrum_shift": float(averaged_shift),
                  "doublet_found": moved["doublet_found"],
                  "monopole_number": moved["monopole_number"]}
    if moved["doublet_found"]:
        weight_shift = 0.0
        for m, (a, b) in enumerate(PAIRS):
            image = (permutation[a], permutation[b])
            n = PAIRS.index((min(image), max(image)))
            weight_shift = max(weight_shift, abs(
                original["doublet_weights"][m] - moved["doublet_weights"][n]))
        energy_shift = (abs(original["doublet_energy"]
                            - moved["doublet_energy"])
                        / max(1.0, abs(original["doublet_energy"])))
        relabeling.update({"doublet_weight_shift": float(weight_shift),
                           "doublet_energy_shift": float(energy_shift),
                           "held": bool(spectrum_shift <= tolerance
                                        and averaged_shift <= tolerance
                                        and weight_shift <= tolerance
                                        and energy_shift <= tolerance)})
    else:
        relabeling["held"] = False
    result["relabeling"] = relabeling
    # --- refinement
    refined, sheet_data = refined_host(spacetime)
    support_r = refined_support(sheet_data[0])
    group_r = refined_rotation_group()
    edges = len(REFINED_EDGES)
    fine = _fiber_fingerprint(refined, support_r, group_r, edges, kappa, beta,
                              config)
    refinement = {"convention": ("one stellar subdivision of each sheet's "
                                 "tetrahedron at the centroid of the flat "
                                 "cell, the new edges at the centroid's "
                                 "squared lengths with the trivial link; the "
                                 "mean field is not re-solved"),
                  "overlap_floor": overlap_floor,
                  "doublet_found": fine["doublet_found"],
                  "monopole_number": fine["monopole_number"],
                  "refined_spectrum": fine["spectrum"]}
    if fine["doublet_found"]:
        block = fine["doublet_frame"]
        maps, characters = [], []
        for g, d in zip(group_r, fine["actions"]):
            lifted = d / np.sqrt(np.linalg.det(block.conj().T @ d @ block))
            for sign in (1.0, -1.0):
                maps.append(sign * lifted)
                characters.append(sign * complex(
                    np.trace(block.conj().T @ lifted @ block)))
        isotypic = np.asarray(obs.SharpSpin.isotypicProjector(
            maps, characters, 2, 1))
        values, vectors = np.linalg.eig(isotypic)
        span = vectors[:, np.abs(values - 1.0) < 1e-6]
        compressed = np.linalg.pinv(span) @ fine["averaged"] @ span
        energies, mixing = np.linalg.eig(compressed)
        candidates = span @ mixing
        shared = [k for k, e in enumerate(REFINED_EDGES)
                  if REFINED_CENTRE not in e]
        original_frame = np.zeros((edges, 2), dtype=complex)
        for m, k in enumerate(shared):
            original_frame[k, :] = original["doublet_frame"][m, :]
        distinct = []
        for e in energies:
            if not any(abs(e - f) < 1e-6 for f in distinct):
                distinct.append(complex(e))
        best = None
        for energy in sorted(distinct, key=lambda v: (v.real, v.imag)):
            picked = candidates[:, np.abs(energies - energy) < 1e-6]
            overlap = _subspace_overlap(original_frame[shared, :],
                                        picked[shared, :])
            if best is None or overlap > best["overlap"]:
                best = {"overlap": overlap, "energy": energy,
                        "rank": int(picked.shape[1])}
        refinement.update({
            "isotypic_dimension": int(span.shape[1]),
            "refined_bands": sorted(distinct, key=lambda v: (v.real, v.imag)),
            "overlap": best["overlap"],
            "refined_energy": best["energy"],
            "refined_rank": best["rank"],
            "energy_shift": float(
                abs(best["energy"] - original["doublet_energy"])
                / max(1.0, abs(original["doublet_energy"]))),
            "held": bool(best["overlap"] >= overlap_floor
                         and best["rank"] == 2)})
    else:
        refinement["held"] = False
    result["refinement"] = refinement
    return result



def fingerprint_evidence(fingerprint):
    """The evidence of quark condition 7 from a spectral fingerprint read
    (`spectral_fingerprint_read`), or the two items unmeasured when there is
    none."""
    E = obs.QuarkConditionEvidence
    if not fingerprint:
        return [E("refinement-stability", None, "no refinement is run"),
                E("relabeling-stability", None, "no relabeling is run")]
    r = fingerprint["refinement"]
    l = fingerprint["relabeling"]
    if "overlap" in r:
        refinement = ("the reference doublet re-extracted on the refined "
                      "cell overlaps the original by %.4f on the shared "
                      "edges (floor %g, rank %d), energy shift %.3g, "
                      "isotypic dimension %d"
                      % (r["overlap"], r["overlap_floor"], r["refined_rank"],
                         r["energy_shift"], r["isotypic_dimension"]))
    elif r.get("unread"):
        refinement = r["unread"]
    else:
        refinement = "the refined support carries no spinor doublet"
    if "doublet_weight_shift" in l:
        relabeling = ("under the relabeling %s the spectra of h_1 and of its "
                      "T-average shift by %.3g and %.3g (relative), the "
                      "doublet's energy by %.3g and its edge weights by %.3g"
                      % (l["permutation"], l["spectrum_shift"],
                         l["averaged_spectrum_shift"],
                         l["doublet_energy_shift"],
                         l["doublet_weight_shift"]))
    elif l.get("unread"):
        relabeling = l["unread"]
    else:
        relabeling = "the relabeled host carries no spinor doublet"
    return [E("refinement-stability", bool(r["held"]), refinement),
            E("relabeling-stability", bool(l["held"]), relabeling)]


def fingerprint_text(fingerprint):
    """One content's spectral fingerprint reads as text."""
    if not fingerprint:
        return "spectral fingerprint unread"
    r = fingerprint["refinement"]
    l = fingerprint["relabeling"]
    relabeling = ("stable" if l["held"] else "unstable")
    if "doublet_weight_shift" in l:
        relabeling += " (spectrum shift %.2g, doublet weight shift %.2g)" % (
            l["spectrum_shift"], l["doublet_weight_shift"])
    refinement = ("stable" if r["held"] else "unstable")
    if "overlap" in r:
        refinement += " (overlap %.4f, energy shift %.2g)" % (
            r["overlap"], r["energy_shift"])
    elif r.get("unread"):
        refinement += " (%s)" % r["unread"]
    else:
        refinement += " (no spinor doublet on the refined support)"
    return "spectral fingerprint: relabeling %s; refinement %s" % (
        relabeling, refinement)


def quark_conditions(spacetime, alignment, recursion, symmetry_residual,
                     report, anchor=None, fingerprint=None,
                     tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The seven v16 quark conditions by name (`QuarkConditions`), from what a
    single-level synthesis measures: condition 3 from the anchor atlas read
    (`anchor_evidence`) and condition 7 from the spectral fingerprint read
    (`fingerprint_evidence`). Anything that needs several cobordism frames
    is left unmeasured and so reads "not evaluable"."""
    E = obs.QuarkConditionEvidence
    spin = alignment["spin_read"]
    supports = [sheet_support(spacetime, t) for t in range(SHEETS)]
    monopoles = [s.monopoleNumber() for s, _ in supports]
    sheeting = obs.SheetedSupport(SHEETS, BASE_EDGES)
    isomorphism = sheeting.certifyIsomorphism(
        [np.array(sheet_squared_lengths(spacetime, t)) for t in range(SHEETS)],
        [np.array(sheet_links(spacetime, t)) for t in range(SHEETS)],
        tolerance)
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
         E("external-leakage", all(n < tolerance
                                   for n in recursion["transport_norms"]),
           "inter-component transport norms %s"
           % recursion["transport_norms"])],
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
        anchor_evidence(anchor),
        [E("odd-occupation-parity", True,
           "one occupied mode of the fibre per quark: parity (-1)^1")],
        [E("color-transport-full-rank",
           bool(attachment.certificate.holds()),
           "det S = %s (sheet-to-sheet attachment)"
           % complex(attachment.determinant)),
         E("base-transport-leakage", all(
             n < tolerance
             for n in recursion["transport_norms"]),
           "inter-component transport norms %s" % recursion["transport_norms"]),
         E("transport-over-lifetime", None,
           "a single level has no lifetime")],
        [E("lineage-intersection", None,
           "a lineage needs a cobordism history")],
        fingerprint_evidence(fingerprint),
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
        except ReadRefused as refusal:
            # the pole read refused by name on the geometry the mean-field
            # solve reached: recorded with its reason and the solve's record,
            # and the content supplies no pole
            record = {"content": list(content),
                      "holonomy": config["holonomy"],
                      "elimination": config["elimination"],
                      "failed": str(refusal), "refusal": refusal.name,
                      "relaxation": refusal.relaxation, "doublet_reads": []}
        except ValueError as error:
            # a declared refusal of the library at this content (a band that
            # cannot hold the occupation, a face holonomy outside the domain
            # of the holonomy term): recorded with its reason, and the content
            # supplies no pole
            record = {"content": list(content),
                      "holonomy": config["holonomy"],
                      "elimination": config["elimination"],
                      "failed": str(error), "doublet_reads": []}
        records.append(record)
        if on_content is not None:
            on_content(record)
    return {"kappa": kappa, "beta": beta, "holonomy": config["holonomy"],
            "elimination": config["elimination"],
            "failed_contents": [record["content"] for record in records
                                if "failed" in record],
            "contents": records,
            "ratios": ratios(records,
                             declared_tolerance(config, "tie_tolerance")),
            "pole_table": pole_table(records)}


# ------------------------------------------------------------- the report
#
# The report is per doublet content: every (content, doublet content) pair is
# reported on its own, every pole of every sector is kept, and the minima are
# separate, labelled summaries that name the pair each came from. Nothing here
# averages over doublet contents or over contents.

#: The two spin sectors, keyed by j(j + 1) as the records key them.
SPINS = (str(SPIN_HALF), str(SPIN_THREE_HALVES))
SPIN_NAMES = {str(SPIN_HALF): "1/2", str(SPIN_THREE_HALVES): "3/2"}
#: The two operators every sector is read on: dGamma of the T-averaged h_1
#: (quasi-free) and the same with the Section 7 quartic.
COLUMNS = ("quasi_free", "with_quartic")
COLUMN_NAMES = {"quasi_free": "quasi-free", "with_quartic": "with quartic"}
#: Two poles whose real parts agree to this relative tolerance are tied for a
#: minimum, and every tied pair is named beside the minimum.
DECLARED_TIE_TOLERANCE = DECLARED_TOLERANCE


def sector_rows(records):
    """Every (content, doublet content) pair of a list of content records,
    with the doublet content's sectors: the rows the pole table, the minima
    and the ratios are taken over."""
    for record in records:
        for read in record.get("doublet_reads", []):
            yield record["content"], read["doublet_content"], read["sectors"]


def _tied(a, b, tolerance=DECLARED_TIE_TOLERANCE):
    """Whether two poles tie in the ascending-real-part order."""
    return abs(a.real - b.real) <= tolerance * max(1.0, abs(a), abs(b))


def lowest_of(candidates, tolerance=DECLARED_TIE_TOLERANCE):
    """The candidate whose ``pole`` has the smallest real part, the library's
    declared `OccupationOrder.AscendingRealPart` (the first such candidate in
    the order given), as a copy that lists under ``tied`` every other
    candidate whose pole's real part agrees with it to ``tolerance``, which
    it records as ``tie_tolerance``; None when there is no candidate."""
    candidates = list(candidates)
    best = None
    for candidate in candidates:
        if best is None or candidate["pole"].real < best["pole"].real:
            best = candidate
    if best is None:
        return None
    out = dict(best)
    out["tied"] = [c for c in candidates if c is not best
                   and _tied(c["pole"], best["pole"], tolerance)]
    out["tie_tolerance"] = tolerance
    return out


def pole_candidates(records, name, spins=SPINS, reading=None):
    """The lowest pole of every (content, doublet content, spin) sector of the
    records in the named column, one candidate per sector, labelled with its
    content, doublet content, spin and restriction to 2T, in pair-major and
    then spin order. With ``reading`` ("nucleon_reading" or "delta_reading"),
    only the sectors the 2T reading names."""
    out = []
    for content, doublet_content, sectors in sector_rows(records):
        for j2 in spins:
            entry = sectors.get(j2)
            if not entry or (reading is not None and not entry.get(reading)):
                continue
            pole = (entry.get(name) or {}).get("lowest_pole")
            if pole is not None:
                out.append({"pole": pole, "content": list(content),
                            "doublet_content": list(doublet_content),
                            "spin_j_j_plus_1": float(j2),
                            "restriction_to_2T": entry.get(
                                "restriction_to_2T")})
    return out


def lowest_poles(record, name, tolerance=DECLARED_TIE_TOLERANCE):
    """Per spin, the lowest pole of one content record over its doublet
    contents in the named column ("quasi_free" or "with_quartic"), labelled
    with the doublet content it came from and every tied doublet content
    (`lowest_of`); None for a spin no sector carries."""
    return {j2: lowest_of(pole_candidates([record], name, (j2,)), tolerance)
            for j2 in SPINS}


def lowest_over_pairs(records, name, tolerance=DECLARED_TIE_TOLERANCE):
    """Per spin, the lowest pole over every (content, doublet content) pair of
    the records in the named column, labelled with the pair it came from and
    every tied pair (`lowest_of`); None for a spin no sector carries."""
    return {j2: lowest_of(pole_candidates(records, name, (j2,)), tolerance)
            for j2 in SPINS}


def pole_rows(records):
    """Every pole of every (content, doublet content, spin, column) sector of
    the records, one row per distinct pole, in the records' order. A row
    names its content, doublet content, spin and column, and carries the
    pole, its multiplicity, whether it is the lowest of its sector, the
    sector's restriction to 2T, the pole's own spinor, spin-lift and colour
    certificates, and the sector read's bound-state certificates and
    compression leakage.
    Nothing is dropped and nothing is combined."""
    rows = []
    for content, doublet_content, sectors in sector_rows(records):
        for j2 in SPINS:
            entry = sectors.get(j2)
            if not entry:
                continue
            for name in COLUMNS:
                read = entry.get(name) or {}
                poles = read.get("poles") or []
                multiplicity = read.get("multiplicity") or [None] * len(poles)
                certificates = read.get("pole_certificates") or \
                    [{}] * len(poles)
                for pole, count, certificate in zip(poles, multiplicity,
                                                    certificates):
                    rows.append({
                        "content": list(content),
                        "doublet_content": list(doublet_content),
                        "spin_j_j_plus_1": float(j2),
                        "column": name,
                        "pole": pole,
                        "multiplicity": count,
                        "lowest_in_sector": pole == read.get("lowest_pole"),
                        "restriction_to_2T": entry.get("restriction_to_2T"),
                        "sharp_spinor": certificate.get("sharp_spinor"),
                        "spin_lift_sharp": certificate.get("spin_lift_sharp"),
                        "colour_casimir_residual": certificate.get(
                            "colour_casimir_residual"),
                        "failed_certificates": list(
                            read.get("failed_certificates") or []),
                        "compression_leakage": read.get(
                            "compression_leakage"),
                    })
    return rows


def pole_table(records):
    """Per column (quasi-free, with the quartic) and per spin, every pole of
    every (content, doublet content) pair (`pole_rows`) in ascending order of
    real part, each row naming its content and doublet content, so the pair
    that supplies any pole, and any tie between pairs or between spins inside
    one pair, can be read off the record."""
    rows = pole_rows(records)
    table = {}
    for name in COLUMNS:
        table[name] = {}
        for j2 in SPINS:
            picked = [row for row in rows if row["column"] == name
                      and row["spin_j_j_plus_1"] == float(j2)]
            picked.sort(key=lambda row: (row["pole"].real, row["pole"].imag))
            table[name][j2] = picked
    return table


def _pair(s_n, s_d, extra):
    out = {
        "nucleon_pole": s_n, "delta_pole": s_d,
        "pole_ratio": s_n / s_d,
        "modulus_ratio": abs(s_n) / abs(s_d),
        "real_part_ratio": s_n.real / s_d.real,
        "target_mass_ratio": TARGET_MASS_RATIO,
    }
    out.update(extra)
    return out


def _pair_labels(candidates):
    """The (content, doublet content, spin) labels of tied candidates."""
    return [{"content": c["content"], "doublet_content": c["doublet_content"],
             "spin_j_j_plus_1": c["spin_j_j_plus_1"]} for c in candidates]


def ratios(records, tolerance=DECLARED_TIE_TOLERANCE):
    """The nucleon and Delta poles over every (content, doublet content)
    pair, and their ratios against the target, in two pairings. Each names
    the two pairs it compares, and every pair tied with either pole.

    * By 2T reading, the reading of WP v18 §11.1 and §14: the lowest pole
      whose sector restricts to a 2 (a nucleon reading) over the lowest pole
      whose sector restricts to 2' + 2'' (a Delta reading). The type is what
      a finite cluster certifies; it fixes the representation of 2T and not
      the continuum spin value.
    * By the spin of the lift: the lowest sharp spin-1/2 pole over the lowest
      sharp spin-3/2 pole under the constructed SU(2) action. Each candidate
      carries its restriction to 2T, and the Delta candidate the tetrahedral
      ambiguity: its sector is a Delta reading only when it restricts to the
      degenerate pair 2' + 2''; a 2 in its restriction is indistinguishable
      from spin 1/2 with triality.
    """
    out = {}
    for name in COLUMNS:
        result = {}
        n = lowest_of(pole_candidates(records, name,
                                      reading="nucleon_reading"), tolerance)
        d = lowest_of(pole_candidates(records, name, reading="delta_reading"),
                      tolerance)
        if n is not None and d is not None:
            result["by_2T_reading"] = _pair(n["pole"], d["pole"], {
                "nucleon_content": n["content"],
                "nucleon_doublet_content": n["doublet_content"],
                "nucleon_spin_j_j_plus_1": n["spin_j_j_plus_1"],
                "nucleon_tied_pairs": _pair_labels(n["tied"]),
                "delta_content": d["content"],
                "delta_doublet_content": d["doublet_content"],
                "delta_spin_j_j_plus_1": d["spin_j_j_plus_1"],
                "delta_tied_pairs": _pair_labels(d["tied"]),
            })
        else:
            result["by_2T_reading"] = None
        by_spin = lowest_over_pairs(records, name)
        n, d = by_spin[str(SPIN_HALF)], by_spin[str(SPIN_THREE_HALVES)]
        if n is not None and d is not None:
            restriction_d = d["restriction_to_2T"] or []
            result["by_spin_lift"] = _pair(n["pole"], d["pole"], {
                "nucleon_content": n["content"], "delta_content": d["content"],
                "nucleon_doublet_content": n["doublet_content"],
                "delta_doublet_content": d["doublet_content"],
                "nucleon_restriction": n["restriction_to_2T"],
                "delta_restriction": d["restriction_to_2T"],
                "nucleon_tied_pairs": _pair_labels(n["tied"]),
                "delta_tied_pairs": _pair_labels(d["tied"]),
                "delta_is_a_delta_reading":
                    sorted(restriction_d) == sorted(["2'", "2''"]),
                "delta_ambiguous_with_spin_half": "2" in restriction_d,
            })
        else:
            result["by_spin_lift"] = None
        out[name] = result
    return out


# ----------------------------------------------------------- the report text


def _measured(value, form="%.2g"):
    return "unmeasured" if value is None else form % value


def column_text(read):
    """One column of one sector as text: every pole with its multiplicity and
    its own spinor, spin-lift and colour certificates, then whether the
    bound-state read
    was certified (or the certificates it failed, by name) and the sector's
    compression leakage."""
    poles = read.get("poles") or []
    multiplicity = read.get("multiplicity") or [None] * len(poles)
    certificates = read.get("pole_certificates") or [{}] * len(poles)
    parts = []
    for pole, count, certificate in zip(poles, multiplicity, certificates):
        sharp = certificate.get("sharp_spinor")
        spinor = ("spinor unmeasured" if sharp is None else
                  "spinor sharp" if sharp else
                  "spinor not sharp (residuals %s, %s)" % (
                      _measured(certificate.get("spinor_right_residual")),
                      _measured(certificate.get("spinor_left_residual"))))
        lifted = certificate.get("spin_lift_sharp")
        lift = ("lift unmeasured" if lifted is None else
                "lift sharp" if lifted else
                "lift not sharp (residuals %s, %s)" % (
                    _measured(certificate.get("spin_lift_right_residual")),
                    _measured(certificate.get("spin_lift_left_residual"))))
        parts.append("%s x%s [%s, %s, colour %s]" % (
            _complex_text(pole), "?" if count is None else count, spinor,
            lift, _measured(certificate.get("colour_casimir_residual"))))
    failed = read.get("failed_certificates") or []
    return "%s {read %s, leakage %s}" % (
        ", ".join(parts) if parts else "no pole",
        "failed " + ", ".join(failed) if failed else "certified",
        _measured(read.get("compression_leakage")))


def pair_line(content, read, prefix=""):
    """One (content, doublet content) pair as one line: both spins, and for
    each both columns with every pole and its certificates (`column_text`).
    A spin the doublet content carries no sector of says so."""
    parts = []
    for j2 in SPINS:
        entry = read["sectors"].get(j2)
        if not entry:
            parts.append("spin %s: no sector" % SPIN_NAMES[j2])
            continue
        parts.append("spin %s (restricts to %s): %s" % (
            SPIN_NAMES[j2], "+".join(entry.get("restriction_to_2T") or []),
            "; ".join("%s %s" % (COLUMN_NAMES[name],
                                 column_text(entry.get(name) or {}))
                      for name in COLUMNS)))
    triality = read.get("total_triality")
    return "%scontent %s, doublet content %s%s | %s" % (
        prefix, list(content), list(read["doublet_content"]),
        "" if triality is None else " (triality %d)" % triality,
        " | ".join(parts))


def _band_text(band):
    places = band["positions"]
    chosen = band["declared_positions"]
    return ("band %d holds %g in rank %d at %s, places %s (chosen at %s)%s%s"
            % (band["declared_index"], band["occupation"], band["rank"],
               ", ".join(_complex_text(v) for v in band["eigenvalues"][:1]),
               places, chosen, ", crossed" if band["crossed"] else "",
               ", splits a degenerate group" if band.get("ambiguous") else ""))


def relaxation_text(relaxation):
    """One content's mean-field solve as text: whether it converged, its
    force and iterations, and, where the record carries them, the method, why
    it stopped (by name, with its detail), the joint Jacobian's rank and rank
    gap at the end point, the Kontsevich-Segal margin and the growth of the
    lengths there, and the occupied bands followed from the host with their
    places in the ascending real-part order, the iterates at which a band had
    crossed another and the lowest overlap of a band with its previous
    projector."""
    if not relaxation or "converged" not in relaxation:
        return "mean field unrecorded"
    text = "mean field converged %s (force norm %.3g after %d iterations)" % (
        relaxation["converged"], relaxation["force_norm"],
        relaxation["iterations"])
    if "method" not in relaxation:
        return text
    jacobian = relaxation.get("joint_jacobian") or {}
    text += ("; method %s, stopped: %s (%s); joint Jacobian rank %d of %d, "
             "rank gap %.3g; Kontsevich-Segal margin %.3g; largest |z| %.3g "
             "times the host's" % (
                 relaxation["method"], relaxation["stop_reason"],
                 relaxation["stop_detail"], jacobian.get("rank", 0),
                 jacobian.get("size", 0), jacobian.get("rank_gap", math.nan),
                 relaxation["kontsevich_segal_margin"],
                 relaxation["largest_length_ratio"]))
    if "force_hessian" in relaxation:
        pinned = relaxation.get("fiber_moments", 0)
        if pinned:
            relative = [abs(r) / abs(t) if abs(t) > 0 else abs(r)
                        for r, t in zip(relaxation["moment_residuals"],
                                        relaxation["moment_targets"])]
            text += ("; %d of the occupied fiber's %d power sums pinned at "
                     "the host: multipliers %s, largest relative residual "
                     "|p_j(h_C) - p_j*| / |p_j*| %.2g" % (
                         pinned, relaxation["fiber_rank"],
                         "[%s]" % ", ".join(_complex_text(x) for x in
                                            relaxation["multipliers"]),
                         max(relative)))
        else:
            text += "; no power sum of the occupied fiber pinned"
        text += ("; Hessian on the range of the Hellmann-Feynman force %s "
                 "(%s), force %.3g" % (
                     _complex_text(relaxation["force_hessian"]),
                     relaxation["force_hessian_sign"],
                     relaxation["hellmann_feynman_force_norm"]))
    bands = relaxation.get("bands") or []
    text += ("; bands by %s: %s; a band had crossed another at %d of %d "
             "iterates, lowest overlap %.3g") % (
            relaxation["band_selection"],
            "; ".join(_band_text(b) for b in bands) if bands else "none",
            relaxation["band_crossing_iterates"],
            len(relaxation.get("trace") or []),
            relaxation["lowest_band_overlap"])
    if not relaxation.get("action_available", True):
        text += "; action unavailable: " + relaxation["action_unavailable"]
    return text


def content_pair_lines(record, prefix=""):
    """Every (content, doublet content) pair of one content record, one line
    each (`pair_line`); a refused content is one line with its reason and,
    for a read refused after the mean-field solve, the solve's record."""
    if "failed" in record:
        line = "%scontent %s: refused: %s" % (prefix, list(record["content"]),
                                              record["failed"])
        if record.get("relaxation"):
            line += "; " + relaxation_text(record["relaxation"])
        return [line]
    reads = record.get("doublet_reads") or []
    if not reads:
        return ["%scontent %s: no doublet content was read"
                % (prefix, list(record["content"]))]
    return [pair_line(record["content"], read, prefix) for read in reads]


def _lowest_text(best, name_content):
    if best is None:
        return "none"
    label = ("content %s, doublet content %s" % (best["content"],
                                                 best["doublet_content"])
             if name_content else
             "doublet content %s" % (best["doublet_content"],))
    text = "%s from %s" % (_complex_text(best["pole"]), label)
    if best.get("tied"):
        text += " (tied to %g with %s)" % (
            best.get("tie_tolerance", DECLARED_TIE_TOLERANCE), "; ".join(
            ("content %s, doublet content %s" % (c["content"],
                                                 c["doublet_content"])
             if name_content else "doublet content %s"
             % (c["doublet_content"],)) for c in best["tied"]))
    return text


def lowest_lines(records, prefix=""):
    """The labelled minima as text, after the per-pair lines, each line
    starting "lowest over": per content, the lowest pole (smallest real part)
    of each spin and column over its doublet contents, with the doublet
    content it came from; then the lowest over every (content, doublet
    content) pair, with the pair it came from. Ties are named."""
    lines = []
    for record in records:
        if "failed" in record:
            continue
        per_column = {name: lowest_poles(record, name) for name in COLUMNS}
        lines.append("%slowest over the doublet contents of content %s: %s"
                     % (prefix, list(record["content"]), "; ".join(
                         "spin %s %s %s" % (
                             SPIN_NAMES[j2], COLUMN_NAMES[name],
                             _lowest_text(per_column[name][j2], False))
                         for j2 in SPINS for name in COLUMNS)))
    per_column = {name: lowest_over_pairs(records, name) for name in COLUMNS}
    lines.append("%slowest over every (content, doublet content) pair: %s" % (
        prefix, "; ".join(
            "spin %s %s %s" % (SPIN_NAMES[j2], COLUMN_NAMES[name],
                               _lowest_text(per_column[name][j2], True))
            for j2 in SPINS for name in COLUMNS)))
    return lines


def _tie_text(r, role):
    tied = r.get(role + "_tied_pairs") or []
    if not tied:
        return ""
    return "; s_%s tied with %s" % ("N" if role == "nucleon" else "D",
                                    "; ".join(
        "content %s, doublet content %s" % (t["content"], t["doublet_content"])
        for t in tied))


def ratio_lines(point_ratios, prefix=""):
    """The ratios of a set of records as text, one line per column and
    pairing, each naming the two (content, doublet content) pairs it
    compares."""
    lines = []
    for name in COLUMNS:
        for pairing in ("by_2T_reading", "by_spin_lift"):
            r = ((point_ratios or {}).get(name) or {}).get(pairing)
            head = "%sratio %-12s %-13s" % (prefix, COLUMN_NAMES[name],
                                            pairing)
            if r is None:
                lines.append(head + " no pole pair")
                continue
            note = ""
            if pairing == "by_spin_lift":
                note = "; Delta restriction %s%s" % (
                    "+".join(r["delta_restriction"] or []),
                    " (ambiguous with spin 1/2)"
                    if r["delta_ambiguous_with_spin_half"] else "")
            lines.append(
                head + " s_N=%s (content %s, doublet content %s) s_D=%s "
                "(content %s, doublet content %s) s_N/s_D=%s "
                "|s_N|/|s_D|=%.6f Re/Re=%.6f; "
                "target m_N/m_Delta=%.4f%s%s%s"
                % (_complex_text(r["nucleon_pole"]), r["nucleon_content"],
                   r["nucleon_doublet_content"],
                   _complex_text(r["delta_pole"]), r["delta_content"],
                   r["delta_doublet_content"],
                   _complex_text(r["pole_ratio"]), r["modulus_ratio"],
                   r["real_part_ratio"], TARGET_MASS_RATIO, note,
                   _tie_text(r, "nucleon"), _tie_text(r, "delta")))
    return lines


def point_lines(point):
    """One scan point as text: every (content, doublet content) pair on its
    own line, then the labelled minima, then the ratios with the pairs they
    compare."""
    records = point["contents"]
    lines = ["kappa=%g beta=%g: %d contents, %d refused; one line per "
             "(content, doublet content) pair, poles s with multiplicity x"
             % (point["kappa"], point["beta"], len(records),
                sum(1 for record in records if "failed" in record))]
    for record in records:
        if "failed" not in record:
            lines.append("  content %s %s%s%s" % (
                list(record["content"]),
                relaxation_text(record.get("relaxation")),
                "; " + anchor_text(record["anchor"])
                if "anchor" in record else "",
                "; " + fingerprint_text(record["spectral_fingerprint"])
                if "spectral_fingerprint" in record else ""))
        lines += content_pair_lines(record, "  ")
    lines += lowest_lines(records, "  ")
    lines += ratio_lines(point.get("ratios"), "  ")
    return lines


def default_config(kappas=DECLARED_KAPPAS, betas=DECLARED_BETAS,
                   edge_squared=DECLARED_EDGE_SQUARED,
                   regge_hinges="interior", selected_contents=None,
                   holonomy=DECLARED_HOLONOMY,
                   elimination=DECLARED_ELIMINATION,
                   mean_field_method=DECLARED_MEAN_FIELD_METHOD,
                   band_selection=DECLARED_BAND_SELECTION,
                   stiffness=DECLARED_STIFFNESS,
                   fiber_moments=DECLARED_FIBER_MOMENTS, tolerances=None):
    """The declared configuration, recorded with every run. The contents
    default to all ten; a subset is for tests and quick checks and changes no
    number of the contents it keeps. ``tolerances`` sets any of `TOLERANCES`
    by key; the others are recorded at `DECLARED_TOLERANCE`."""
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
        "newton_iterations": 40,
        "jacobian_radius": 1e-4,
        "mean_field_iterations": 40,
        **declared_tolerances(tolerances),
        "mean_field_method": mean_field_method,
        "band_selection": band_selection,
        "stiffness": stiffness,
        "fiber_moments": str(fiber_moments),
        "kappa_role": (
            "kappa = 8 pi G enters only through the Regge weight 1/kappa; "
            "the length stiffness is the spectral-moment part of S_0, the "
            "holomorphic spectral constraint of WP v17 §3.4 on the occupied "
            "fiber" if stiffness == "none" else
            "kappa = 8 pi G enters through the Regge weight 1/kappa and the "
            "linear stiffness stand-in (1/2 kappa^-1) ||l - l0||^2"),
        "length_runaway_ratio": DECLARED_LENGTH_RUNAWAY_RATIO,
        "target_mass_ratio": TARGET_MASS_RATIO,
        "target_reading": "the pole is the complex rest energy of the bound "
                          "cluster under the first-order flow (WP v18 "
                          "§13.3); s_N / s_Delta is compared with m_N / "
                          "m_Delta",
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
    alignment = aligned_doublet_frame(
        monopole_support(), rotation_group(),
        declared_tolerance(config, "degeneracy_tolerance"),
        declared_tolerance(config, "certificate_tolerance"))
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
                sys.stdout.write("\n".join(point_lines(point)) + "\n")
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
BASELINE = "#c3c2b7"
GRID = "#e1e0d9"
SERIES = {"quasi_free": "#2a78d6", "with_quartic": "#eb6834"}
SERIES_LABEL = {"quasi_free": "quasi-free dGamma(h-bar_1)",
                "with_quartic": "with the Section 7 quartic"}
SERIES_MARKER = {"quasi_free": "o", "with_quartic": "D"}
#: The marks of the two spins in the pole panels: colour, marker, fill, and
#: the offset from the pair's slot, so that the two spins of one pair sit side
#: by side rather than on top of each other.
SPIN_STYLE = {
    str(SPIN_HALF): {"color": "#1baf7a", "marker": "o", "filled": True,
                     "offset": -0.2, "label": "spin 1/2"},
    str(SPIN_THREE_HALVES): {"color": "#4a3aa7", "marker": "s",
                             "filled": False, "offset": 0.2,
                             "label": "spin 3/2"},
}
#: The empty slots between two groups of pairs in a pole panel.
GROUP_GAP = 1.0
#: The size of the live window and of the rendered frame, in inches.
FIGURE_SIZE = (15, 10)
#: The largest font size of the listing of the pairs each ratio compares; the
#: listing is set smaller when its lines need it to fit the panel.
LISTING_FONT_SIZE = 6.0
#: How a pole panel marks the mean-field solve behind each group of pairs
#: (`solve_state`): the ink of the group's callout and label, the tint of the
#: band behind its pairs, the callout's sign, and the legend's name for the
#: band.
SOLVE_STYLE = {
    "converged": {"ink": "#1a7f37", "band": "#e2f2e7", "sign": "\u2713",
                  "label": "mean-field solve converged"},
    "not converged": {"ink": "#b3261e", "band": "#fbe6e3", "sign": "\u2717",
                      "label": "mean-field solve not converged"},
    "refused": {"ink": INK_MUTED, "band": "#ecebe6", "sign": "\u2717",
                "label": "pole read refused"},
}
#: The short names, in a callout, of why a mean-field solve stopped
#: (`cob.relaxation_stop_name`) and of the refusals of a pole read
#: (`read_refusal`).
STOP_SHORT = {
    "the declared iterations ran out": "ran out",
    "no damped step reduced the residual": "no descent",
    "every damped step left the domain of the action": "left the domain",
    "no stationary point in the declared monopole sector":
        "left the sector",
    "every damped step came within the declared margin of a zero of the "
    "Villain weight": "Villain zero",
    "the residual is at its floor on the held set": "held floor",
    "the squared lengths ran off": "lengths ran off",
    "an outer iteration made no progress": "no progress",
    "not Kontsevich-Segal allowable": "not KS-allowable",
}
#: The largest font size of the callouts; they are set smaller, all to one
#: size, when the narrowest group needs it.
CALLOUT_FONT_SIZE = 7.0


def _digits(values):
    """A content or a doublet content as its digits: [0, 2, 1] is 021."""
    return "".join(str(int(v)) for v in values)


def solve_state(record):
    """The mean-field solve behind one content record, as the plots mark it:
    ``state`` is "refused" when the content's pole read was refused (it reads
    no pole) and otherwise "converged" or "not converged" as the solve's
    record says; ``reason`` is the short name (`STOP_SHORT`) of why an
    unconverged solve stopped or why the read was refused, None when there is
    none; ``iterations`` is the solve's iteration count, None when
    unrecorded. None when the record carries neither a refusal nor a
    solve."""
    relaxation = record.get("relaxation") or {}
    iterations = relaxation.get("iterations")
    if "failed" in record:
        name = record.get("refusal")
        return {"state": "refused", "reason": STOP_SHORT.get(name, name),
                "iterations": iterations}
    if "converged" not in relaxation:
        return None
    if relaxation["converged"]:
        return {"state": "converged", "reason": None,
                "iterations": iterations}
    stop = relaxation.get("stop_reason")
    return {"state": "not converged", "reason": STOP_SHORT.get(stop, stop),
            "iterations": iterations}


def callout_lines(solve):
    """The lines of one group's callout (`solve_state`): the sign and the
    state, then why the solve stopped or the read was refused, then the
    solve's iterations, each where known."""
    lines = ["%s %s" % (SOLVE_STYLE[solve["state"]]["sign"], solve["state"])]
    if solve["reason"]:
        lines.append(solve["reason"])
    if solve["iterations"] is not None:
        count = int(solve["iterations"])
        lines.append("%d iteration%s" % (count, "" if count == 1 else "s"))
    return lines


def pole_marks(groups):
    """The marks of the pole panels: one mark per distinct pole of every
    (group, doublet content, spin, column) sector; nothing is combined.

    ``groups`` is a list of (label, content record). A group occupies one
    slot per doublet content its record read, in the record's order, and
    groups are separated by `GROUP_GAP` empty slots; a refused content, which
    reads no doublet content, occupies one slot labelled "refused". Returns
    the marks (each with its position ``x``, its pair's slot offset by its
    spin, the group label, the content, the doublet content, the spin, the
    column, the pole and its multiplicity), the groups (label, first and
    last slot, and the mean-field solve behind them, `solve_state`) and the
    slots (position and doublet content label)."""
    marks, spans, slots = [], [], []
    x = 0.0
    for label, record in groups:
        start = x
        reads = record.get("doublet_reads") or []
        if not reads:
            slots.append((x, "refused" if "failed" in record else "none"))
            x += 1.0
        for read in reads:
            slots.append((x, _digits(read["doublet_content"])))
            for j2 in SPINS:
                entry = read["sectors"].get(j2)
                if not entry:
                    continue
                for name in COLUMNS:
                    column = entry.get(name) or {}
                    poles = column.get("poles") or []
                    multiplicity = column.get("multiplicity") or \
                        [None] * len(poles)
                    for pole, count in zip(poles, multiplicity):
                        marks.append({
                            "x": x + SPIN_STYLE[j2]["offset"], "slot": x,
                            "group": label,
                            "content": list(record["content"]),
                            "doublet_content": list(read["doublet_content"]),
                            "spin": j2, "column": name, "pole": pole,
                            "multiplicity": count})
            x += 1.0
        spans.append({"label": label, "first": start, "last": x - 1.0,
                      "solve": solve_state(record)})
        x += GROUP_GAP
    return marks, spans, slots


def ratio_row(where, name, ratio, solves=None):
    """One ratio of a ratio panel, with the two (content, doublet
    content) pairs it compares and the mean-field solve behind each
    (``solves`` maps a content, as a tuple, to its `solve_state`); ``ratio``
    is None when there is no pole pair."""
    if ratio is None:
        return {"where": where, "column": name, "ratio": None,
                "nucleon": None, "delta": None}
    solves = solves or {}
    return {"where": where, "column": name, "ratio": ratio["pole_ratio"],
            "nucleon": {"content": list(ratio["nucleon_content"]),
                        "doublet_content": list(
                            ratio["nucleon_doublet_content"]),
                        "pole": ratio["nucleon_pole"],
                        "solve": solves.get(tuple(ratio["nucleon_content"]))},
            "delta": {"content": list(ratio["delta_content"]),
                      "doublet_content": list(ratio["delta_doublet_content"]),
                      "pole": ratio["delta_pole"],
                      "solve": solves.get(tuple(ratio["delta_content"]))}}


def content_solves(records):
    """Every content record's mean-field solve (`solve_state`), keyed by its
    content as a tuple, for `ratio_row`."""
    return {tuple(record["content"]): solve_state(record)
            for record in records}


def unconverged_poles(row):
    """The poles of a ratio row, N (the spin-1/2 pole) and D (the spin-3/2
    pole), whose content's mean-field solve is recorded and did not
    converge."""
    if row["ratio"] is None:
        return []
    return [role for role, key in (("N", "nucleon"), ("D", "delta"))
            if (row[key].get("solve") or {}).get("state", "converged")
            != "converged"]


def solve_tag(row):
    """What a ratio's listing says of the solves behind its two poles:
    "[both converged]" when both are recorded as converged, "[unconverged:
    ...]" naming the poles whose solve did not converge, and nothing when
    neither applies."""
    if row["ratio"] is None:
        return ""
    unconverged = unconverged_poles(row)
    if unconverged:
        return " [unconverged: %s]" % ", ".join(unconverged)
    if all(row[key].get("solve") for key in ("nucleon", "delta")):
        return " [both converged]"
    return ""


def ratio_pair_text(row):
    """The two pairs a ratio compares as a short label, each written
    content|doublet content: N for the spin-1/2 pole, D for the spin-3/2
    pole."""
    if row["ratio"] is None:
        return "no pole pair"
    return "N %s|%s / D %s|%s" % (
        _digits(row["nucleon"]["content"]),
        _digits(row["nucleon"]["doublet_content"]),
        _digits(row["delta"]["content"]),
        _digits(row["delta"]["doublet_content"]))


def frame_data(frames, index):
    """What one frame draws, as data: every pole of every (content, doublet
    content) pair of the latest scan point (`pole_marks`), and the ratio by
    2T reading of every completed scan point in both columns with the two
    pairs it compares and the mean-field solve behind each (`ratio_row`)."""
    done = frames[:index + 1]
    point = done[-1]
    marks, spans, slots = pole_marks(
        [(_digits(record["content"]), record)
         for record in point["contents"]])
    rows = []
    for p in done:
        where = "k=%g b=%g" % (p["kappa"], p["beta"])
        solves = content_solves(p["contents"])
        for name in COLUMNS:
            rows.append(ratio_row(where, name,
                                  p["ratios"][name]["by_2T_reading"], solves))
    return {"where": "kappa=%g, beta=%g" % (point["kappa"], point["beta"]),
            "done": len(done), "marks": marks, "groups": spans,
            "slots": slots, "ratios": rows}


def style_axis(axis):
    """The recessive chrome every panel shares."""
    axis.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(BASELINE)
    axis.tick_params(colors=INK_MUTED)


def draw_pole_panel(axis, data, name, title, group_label):
    """One pole panel: every pole of every pair in the named column as its
    own mark at its pair's slot, spin 1/2 and spin 3/2 side by side, on a
    symmetric logarithmic scale of Re s (the poles span several decades).
    Below each slot is its doublet content (quarks in 2, 2', 2'' of
    h-bar_1), and below each group its label (``group_label`` says what it
    names). Each group whose record carries a mean-field solve or a refusal
    (`solve_state`) has a band behind its pairs and a callout above them in
    its state's colour (`SOLVE_STYLE`): whether the solve converged, why it
    stopped or the read was refused, and its iterations
    (`callout_lines`)."""
    style_axis(axis)
    named = set()
    for group in data["groups"]:
        solve = group.get("solve")
        if solve is None:
            continue
        style = SOLVE_STYLE[solve["state"]]
        axis.axvspan(group["first"] - 0.5, group["last"] + 0.5,
                     facecolor=style["band"], edgecolor="none", zorder=0,
                     label=style["label"] if solve["state"] not in named
                     else "_nolegend_")
        named.add(solve["state"])
    for j2 in SPINS:
        style = SPIN_STYLE[j2]
        picked = [m for m in data["marks"]
                  if m["column"] == name and m["spin"] == j2]
        axis.plot([m["x"] for m in picked], [m["pole"].real for m in picked],
                  linestyle="none", marker=style["marker"], markersize=5,
                  markeredgewidth=1.5, color=style["color"],
                  markerfacecolor=style["color"] if style["filled"]
                  else "none", label=style["label"])
    axis.set_yscale("symlog", linthresh=1.0)
    axis.axhline(0.0, color=BASELINE, linewidth=0.8)
    for before, after in zip(data["groups"], data["groups"][1:]):
        axis.axvline(0.5 * (before["last"] + after["first"]), color=GRID,
                     linewidth=0.8)
    if data["slots"]:
        axis.set_xlim(data["slots"][0][0] - 0.8, data["slots"][-1][0] + 0.8)
    # a group of one slot has its label where its slot is: both ticks stay
    axis.xaxis.remove_overlapping_locs = False
    axis.set_xticks([0.5 * (g["first"] + g["last"]) for g in data["groups"]])
    axis.set_xticklabels([g["label"] for g in data["groups"]])
    axis.set_xticks([position for position, _ in data["slots"]], minor=True)
    axis.set_xticklabels([label for _, label in data["slots"]], minor=True)
    axis.tick_params(axis="x", which="major", length=0, pad=20, labelsize=7,
                     labelcolor=INK)
    axis.tick_params(axis="x", which="minor", length=2, labelsize=5,
                     labelrotation=90, labelcolor=INK_MUTED)
    axis.set_xlabel("%s; below each slot, its doublet content (quarks in "
                    "2, 2', 2'' of h-bar_1)" % group_label, color=INK,
                    fontsize=8)
    axis.set_ylabel("Re s (symmetric log)", color=INK, fontsize=8)
    left, right = axis.get_xlim()
    per_slot = (axis.get_position().width * axis.figure.get_figwidth()
                * 72.0 / (right - left))
    callouts = [(label, group, callout_lines(group["solve"]))
                for label, group in zip(axis.get_xticklabels(),
                                        data["groups"])
                if group.get("solve") is not None]
    size = min([CALLOUT_FONT_SIZE] + [
        (group["last"] - group["first"] + 0.9) * per_slot
        / (0.55 * max(len(line) for line in lines))
        for _, group, lines in callouts])
    size = max(3.5, size)
    tallest = 0.0
    for label, group, lines in callouts:
        style = SOLVE_STYLE[group["solve"]["state"]]
        label.set_color(style["ink"])
        axis.text(0.5 * (group["first"] + group["last"]), 1.01,
                  "\n".join(lines), transform=axis.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=size,
                  color=style["ink"], linespacing=1.1)
        tallest = max(tallest, 1.25 * size * len(lines))
    axis.set_title(title, color=INK, fontsize=9, pad=6.0 + tallest)
    axis.legend(frameon=False, fontsize=7, labelcolor=INK, loc="best")


def draw_ratio_panel(axis, rows, title):
    """The ratio Re(s_N / s_Delta) by 2T reading in both columns at every
    place a ratio was read (a scan point, or a host cell), one mark each,
    against the
    target; the pairs each compares are listed beside it
    (`draw_pairs_panel`). A ratio with a pole whose mean-field solve did not
    converge (`unconverged_poles`) is drawn hollow."""
    style_axis(axis)
    places = list(dict.fromkeys(row["where"] for row in rows))
    position = {place: k for k, place in enumerate(places)}
    hollow = False
    for name in COLUMNS:
        picked = [row for row in rows if row["column"] == name]
        for unconverged in (False, True):
            chosen = [row for row in picked
                      if bool(unconverged_poles(row)) == unconverged]
            if unconverged and not chosen:
                continue
            hollow = hollow or unconverged
            axis.plot([position[row["where"]] for row in chosen],
                      [row["ratio"].real if row["ratio"] is not None
                       else np.nan for row in chosen],
                      linestyle="none", marker=SERIES_MARKER[name],
                      markersize=7, markeredgewidth=1.5, color=SERIES[name],
                      markerfacecolor="none" if unconverged else SERIES[name],
                      label="_nolegend_" if unconverged
                      else SERIES_LABEL[name])
    if hollow:
        axis.plot([], [], linestyle="none", marker="o", markersize=7,
                  markeredgewidth=1.5, color=INK_MUTED,
                  markerfacecolor="none",
                  label="hollow: a pole's solve did not converge")
    axis.axhline(TARGET_MASS_RATIO, color=INK_MUTED, linewidth=1,
                 linestyle="--")
    axis.text(0, TARGET_MASS_RATIO, " target m_N/m_D = %.4f"
              % TARGET_MASS_RATIO, color=INK_MUTED, va="bottom", fontsize=7)
    axis.set_xticks(range(len(places)))
    axis.set_xticklabels([place.replace(" ", "\n") for place in places],
                         fontsize=6, rotation=90 if len(places) > 8 else 0)
    axis.set_ylabel("Re(s_N / s_Delta)", color=INK, fontsize=8)
    axis.set_title(title, color=INK, fontsize=9)
    axis.legend(frameon=False, fontsize=7, labelcolor=INK)


def draw_pairs_panel(axis, rows):
    """The listing of the pairs every ratio of the ratio panel compares, in a
    monospaced font: one line per place a ratio was read with both columns on
    it, or one line per ratio when that lets the font be larger, set small
    enough for every line to fit the panel (the full listing is also in the
    stdout summary). Each ratio says which of its poles come from a
    mean-field solve that did not converge (`solve_tag`)."""
    axis.axis("off")
    places = list(dict.fromkeys(row["where"] for row in rows))

    def entry(row):
        return "%s %s = %s%s" % (
            "quasi-free" if row["column"] == "quasi_free" else "quartic",
            ratio_pair_text(row),
            _complex_text(row["ratio"]) if row["ratio"] is not None else "-",
            solve_tag(row))

    joined = ["%-13s %s" % (place, ";  ".join(
        entry(row) for row in rows if row["where"] == place))
        for place in places]
    single = ["%-13s %s" % (row["where"], entry(row)) for row in rows]
    head = ("each ratio compares the lowest pole of a sector of type 2 (N)"
            "\nwith the lowest of a sector of type 2'+2'' (D) over every\n"
            "(content, doublet content) pair, written content|doublet "
            "content;\n[unconverged: N] marks a pole whose mean-field solve "
            "did not converge:")
    axis.text(0.0, 1.0, head, va="top", ha="left", fontsize=7, color=INK,
              transform=axis.transAxes)
    position = axis.get_position()
    height = (position.height * axis.figure.get_figheight() - 0.5) * 72.0
    width = position.width * axis.figure.get_figwidth() * 72.0

    def fitted(lines):
        longest = max((len(line) for line in lines), default=1)
        return min(LISTING_FONT_SIZE, height / (1.3 * max(1, len(lines))),
                   width / (0.62 * longest))

    lines = max((joined, single), key=fitted)
    axis.text(0.0, 0.74, "\n".join(lines), va="top", ha="left",
              fontsize=max(3.5, fitted(lines)), color=INK,
              family="monospace", transform=axis.transAxes)


def draw_frame(figure, frames, index):
    """One frame (`frame_data`): the poles of every (content, doublet
    content) pair of the latest scan point, quasi-free and with the quartic,
    each pole its own mark; below them the ratio by 2T reading over the scan
    against the target, and the listing of the pairs each ratio compares."""
    data = frame_data(frames, index)
    figure.clear()
    figure.patch.set_facecolor(SURFACE)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.2, 1.2, 1.1),
                               width_ratios=(1.0, 1.1))
    quasi_free = figure.add_subplot(grid[0, :])
    quartic = figure.add_subplot(grid[1, :])
    ratio = figure.add_subplot(grid[2, 0])
    pairs = figure.add_subplot(grid[2, 1])
    for axis, name in ((quasi_free, "quasi_free"), (quartic, "with_quartic")):
        draw_pole_panel(axis, data, name,
                        "poles %s at %s: every (content, doublet content) "
                        "pair" % (SERIES_LABEL[name], data["where"]),
                        "content (quarks per band of h_1)")
    draw_ratio_panel(ratio, data["ratios"],
                     "ratio by 2T reading over the scan")
    draw_pairs_panel(pairs, data["ratios"])
    figure.suptitle("nucleon-to-Delta poles by controlled synthesis, "
                    "reported per doublet content (%d of the scan done)"
                    % data["done"], color=INK)
    figure.tight_layout()


def _interactive_backends():
    """The interactive matplotlib backends, lowercased (asked of matplotlib,
    as `emergence._interactive_backends` does)."""
    from tessera.drivers.emergence import _interactive_backends as backends
    return backends()



def elapsed_text(seconds):
    """Elapsed wall time as m:ss, or as h:mm:ss from one hour on."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%d:%02d" % (minutes, secs)


def draw_status(figure, message):
    """Show `message` in the live window: centred while the figure has no
    frame yet, as one line along the bottom once it has. `draw_frame` clears
    the figure, so the line is drawn again whenever it is missing."""
    status = getattr(figure, "live_status", None)
    if status is None or status not in figure.texts:
        if figure.axes:
            figure.live_status = figure.text(
                0.5, 0.004, message, ha="center", va="bottom", fontsize=10,
                color=INK_MUTED)
        else:
            figure.live_status = figure.text(
                0.5, 0.5, message, ha="center", va="center", fontsize=14,
                color=INK, wrap=True)
    else:
        status.set_text(message)
    figure.canvas.draw_idle()


def hold_live_window(message):
    """Keep every open live window on screen, showing `message`, until the
    user closes it. The CLI calls this after every output is written; it
    returns at once, False, when no window is open (the user closed it
    during the run)."""
    import matplotlib.pyplot as plt
    numbers = plt.get_fignums()
    if not numbers:
        return False
    for number in numbers:
        draw_status(plt.figure(number), message)
    sys.stdout.write(message + "\n")
    sys.stdout.flush()
    plt.ioff()
    plt.show()
    return True


def drive_live(config, progress=False, points_file=None, keep_open=False):
    """The same `drive`, on a worker thread, drawing each completed scan point
    on the main thread. Refuses a non-interactive backend and WebAgg by name,
    as `tessera.drivers.emergence` does. While a point is being computed the
    window says which one and for how long.

    Closing the window does not stop the scan: the figure's close event
    switches the run to headless, the main thread stops drawing and waits for
    the worker, and every output is still written. When the scan ends the
    figure is closed, or, with `keep_open`, left on screen with its final
    frame for `hold_live_window` once the outputs are written."""
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
    figure = plt.figure(figsize=FIGURE_SIZE)
    plt.show(block=False)
    ready = queue.Queue()
    published = {}
    outcome = {}
    stop = threading.Event()

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    closed = threading.Event()
    finished = threading.Event()

    def on_close(event):
        # Only a close by the user while the scan runs is reported; the
        # driver's own close at the end, and a close after the end, are not.
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

    thread = threading.Thread(target=worker, name="baryon-poles")
    thread.start()
    total = len(config["kappas"]) * len(config["betas"])

    def running(done, seconds):
        if done >= total:
            return "the scan is finishing: %s" % elapsed_text(seconds)
        kappa = config["kappas"][done // len(config["betas"])]
        beta = config["betas"][done % len(config["betas"])]
        return ("scan point %d of %d (kappa %g, beta %g) is running: %s "
                "elapsed; its frame is drawn when the point completes"
                % (done + 1, total, kappa, beta, elapsed_text(seconds)))

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
                if shown is None or now - shown >= LIVE_STATUS_INTERVAL:
                    draw_status(figure, running(done, now - since))
                    shown = now
                figure.canvas.start_event_loop(LIVE_POLL_INTERVAL)
                continue
            if index is None:
                break
            if closed.is_set():
                break
            draw_frame(figure, published["frames"], index)
            done, since, shown = index + 1, time.monotonic(), None
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
            finished.set()
            if keep_open and main_error is None and "error" not in outcome:
                draw_status(figure, "the scan is complete; writing the "
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
    figure = Figure(figsize=FIGURE_SIZE)
    FigureCanvasAgg(figure)
    if result["points"]:
        draw_frame(figure, result["points"], len(result["points"]) - 1)
    figure.savefig(path, dpi=120, facecolor=SURFACE)


def summary(result):
    """Every scan point as text (`point_lines`): one line per (content,
    doublet content) pair with both spins and both columns, the labelled
    minima, and the ratios with the pairs they compare."""
    lines = ["mode: controlled synthesis; target m_N/m_Delta = %.4f (the "
             "pole is the rest energy, WP v18 §13.3)" % TARGET_MASS_RATIO]
    for point in result["points"]:
        lines += point_lines(point)
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
    add_mean_field_arguments(run)
    add_tolerance_arguments(run)
    run.add_argument("--quiet", action="store_true")
    return parser


def _tolerance(text):
    """A positive number, for the tolerance options."""
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0.0:
        raise argparse.ArgumentTypeError(
            "a tolerance is a positive number; got %r" % text)
    return value


def add_tolerance_arguments(parser):
    """One option per tolerance of the stack (`TOLERANCES`), each defaulting
    to `DECLARED_TOLERANCE`, for both drivers. None changes an equation."""
    for key, meaning in TOLERANCES:
        parser.add_argument("--" + key.replace("_", "-"), dest=key,
                            type=_tolerance, default=DECLARED_TOLERANCE,
                            help="%s (default %g)" % (meaning,
                                                      DECLARED_TOLERANCE))


def tolerances_from(args):
    """The parsed tolerance options, by config key (`TOLERANCES`)."""
    return {key: getattr(args, key) for key, _ in TOLERANCES}


def _fiber_moments(text):
    """``r`` or a non-negative integer, for --fiber-moments."""
    if text == "r":
        return text
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(
            "--fiber-moments is r or a non-negative integer; got %r" % text)
    return str(value)


def add_mean_field_arguments(parser):
    """The mean-field solver options both drivers accept. Neither changes an
    equation."""
    parser.add_argument("--mean-field-method",
                        choices=tuple(MEAN_FIELD_METHODS),
                        default=DECLARED_MEAN_FIELD_METHOD,
                        help="how the mean field's fixed point is solved: "
                             "Newton's method on the joint system, or the "
                             "alternation of a geometry relaxation and a "
                             "re-occupation, kept as a named fallback "
                             "(default %s)" % DECLARED_MEAN_FIELD_METHOD)
    parser.add_argument("--band-selection", choices=tuple(BAND_SELECTIONS),
                        default=DECLARED_BAND_SELECTION,
                        help="where a content's bands are chosen: at the "
                             "host, then followed by continuation, or "
                             "re-selected by sorting at every iterate "
                             "(default %s)" % DECLARED_BAND_SELECTION)
    parser.add_argument("--stiffness", choices=STIFFNESS_FORMS,
                        default=DECLARED_STIFFNESS,
                        help="the length stiffness of the joint action: none "
                             "(the spectral-moment part of S_0 is the "
                             "fiber constraint of --fiber-moments), or the "
                             "linear stand-in (1/2 kappa^-1) ||l - l0||^2 "
                             "(default %s)" % DECLARED_STIFFNESS)
    parser.add_argument("--fiber-moments", type=_fiber_moments,
                        default=DECLARED_FIBER_MOMENTS,
                        help="m_c, the power sums p_j(h_C), j = 1..m_c, of "
                             "the occupied fiber pinned at the host (WP v17 "
                             "§3.4): r, the fiber's rank, or a count "
                             "(default %s)" % DECLARED_FIBER_MOMENTS)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = default_config(args.kappa, args.beta, args.edge_squared,
                            args.regge_hinges, holonomy=args.holonomy,
                            elimination=args.eliminate,
                            mean_field_method=args.mean_field_method,
                            band_selection=args.band_selection,
                            stiffness=args.stiffness,
                            fiber_moments=args.fiber_moments,
                            tolerances=tolerances_from(args))
    if args.isospin_doublet:
        config["isospin_doublet"] = True
    points_file = points_path(args.json) if args.json else None
    result = (drive_live(config, progress=not args.quiet,
                         points_file=points_file, keep_open=True)
              if args.live
              else drive(config, progress=not args.quiet,
                         points_file=points_file))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(_jsonable(result), handle, indent=1)
    if args.out:
        render(result, args.out)
    if not args.quiet:
        sys.stdout.write(summary(result) + "\n")
    if args.live:
        hold_live_window("the run is complete and every output is written; "
                         "close this window to exit")
    return result


if __name__ == "__main__":
    main()
