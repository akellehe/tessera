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
   geometry. By default (``--fiber-pinning eigenvalues``) the constraints are
   stated as the eigenvalue of each occupied band, lambda_b = tr(P_b h_1)/r_b
   with P_b the band's Riesz projector, one constraint per band: on the
   sheeted host every occupied band is one eigenvalue repeated once per
   sheet, so these are the independent constraints among the r power sums,
   without the dependent rows. ``--fiber-pinning power-sums`` states them as
   the power sums, m_c of them by ``--fiber-moments`` (r, every moment of the
   fiber, by default; bands, one per occupied band; 1 pins the trace alone).
   kappa = 8 pi G enters only
   through the Regge weight. The holonomy term S_hol is the Villain form
   -beta_V sum_tau log W_M(F_tau), with the Villain weight
   W(F) = sum_m exp(-m^2/(2 beta)) F^m summed to the declared order M
   (``--villain-order``, the sum over |m| <= M, ten by default) and
   beta_V = beta / <m^2>_beta from the same sums: the paper's holonomy term
   at that order, with the bare connection stiffness beta L_1^up at trivial
   holonomy;
   with certificates-blind mean-field backreaction to self-consistency
   (`SelfConsistentMeanField`), the carried density being the content's band
   filling of h_1. The fixed point is found by `MultiCobordism`, whose
   mechanics are used as they are (`tessera.drivers.cell_solve`): its
   objective is the residual norm of the joint stationarity system (WP v17
   lines 259 and 263), the covariance rebuilt at every point; its stage 1
   scores every Pachner move of the base tetrahedron and commits one when
   the residual norm falls; its stage 2 searches along the step that solves
   the equations to the declared order about the point
   (``--direction-order``, 1 for Newton's step). The content's bands are
   chosen at the host in ascending order of real part and followed from
   there by continuation (WP v17 line 151; ``--band-selection
   sort-every-iterate`` re-sorts at every point instead), and every step
   proposal's band overlaps and any crossing are recorded. The three sheets
   are one shared base field (WP v17 §8, "Sheet convention (adopted)"): the
   drive's coordinates are the base tetrahedron's six squared lengths and
   six links, written to every sheet, and the equation of each is the sum of
   the equations of the corresponding edges of the three sheets. A solve
   that reaches no fixed point says why, by name (no move and no scaled
   step lowers the residual norm, the step has no value at the point
   reached, or a declared limit was reached), and the poles are read on the
   geometry the solve ended at; a cell whose complex a committed move
   changed is not a tetrahedron and has no pole read. A geometry that is not Kontsevich-Segal
   allowable is read like any other, and the content's record carries the
   flag "not Kontsevich-Segal allowable" with the margin ("Flags" below).
   A geometry whose squared lengths overflowed the double is not finite, so
   it has no pole to read, and the content's record says so by name;
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
reads the spin content of a symmetric cluster on the T-averaged operator (WP
v17 §9 line 506: "The action on edge modes is canonical even where the
twisted operator is not"; "the T-averaged twisted edge Laplacian"),
h-bar_1 = (1/|T|) sum_g D_1(g)^{-1} h_1 D_1(g), with T = A_4 the rotation
group of the tetrahedron and D_1(g) = rho_1(u_g) P_g the geometric permutation
dressed by the compensating gauge transformation of the cluster's own
connection (WP v18 §11.1). The driver does the same on the relaxed cell
itself: the rotations of the tetrahedron that leave the cell's squared lengths
invariant and carry its connection to a gauge-equivalent one, on every sheet
and at the certificate tolerance, are its rotation group (`cell_symmetry`).
The spin read of the cell itself has three preconditions, each decided at the
certificate tolerance: the cell has all twelve rotations, the spin read of
every sheet names its j = 1/2 doublet, and the sheets' doublet labels agree.
When the three hold, the actions D_1(g), the aligned doublet frame of every
sheet, the isotypic projectors and the anchor atlas are built from that cell's
own connection. When one does not hold, the spin is read in the frame of the
declared symmetric host instead, with the actions D_1(g) and the aligned
doublet frame of `monopole_support()`, and the content's record carries a
flag that names the precondition and gives the numbers that decided it
("Flags" below); the rotations the cell lacks are recorded with their
departures. The record names the frame that was used
(``relaxation.spin_frame``, `spin_frame`). In either frame the spin
decomposition of the occupied modes, the spin sectors and every pole are read
on h-bar_1 (and, for the Section 7 quartic, on the eliminated three-particle
operator averaged over the same diagonal action). The departure of h_1 itself
from the symmetric form is reported beside every record as
`symmetry_residual`.

Flags
-----
A flag is the record of a precondition of a read that does not hold at its
declared tolerance. The read is made all the same, and the flag is carried
beside its result: a dict with the precondition's ``name``, a ``detail``
sentence and the numbers that decided it. A content's record lists its flags
under ``flags``, an empty list when every precondition holds. The names are
"not Kontsevich-Segal allowable" (the geometry the mean-field solve reached,
`geometry_flags`), "not tetrahedrally symmetric", "no j = 1/2 doublet", "the
sheets' doublet labels disagree" and "the doublet labels are not the three Z_3
characters" (the preconditions of the spin read of the cell itself,
`spin_frame`, which also flags a host frame whose reference doublet is not
certified), and "the elimination is outside the range of its expansion" (the
with-quartic read, `truncation_flags`, carried under ``quartic.truncation``
and named beside every ratio built from the content). A read with no value
at all is a different thing: a content whose solve overflowed the double, or
at which the library names no value (a band that cannot hold the occupation,
a face holonomy outside the domain of the holonomy term), has a record with
``failed`` and no pole.

What a content names
--------------------
A content (n_0, n_1, n_2) places n_b quarks in band b of h_1, the bands being
groups of degenerate eigenvalues in ascending order of real part. On the
monopole host each band of h_1 is one simple mode per sheet, rank three, not a
spin doublet, so a content names no doublet. The record carries, beside it,
how the bands the solved state occupies decompose into the doublets of
h-bar_1 (``spin_decomposition``), and the pole sectors are read for every
doublet content and labelled by it (``doublet_reads``); no mapping from a
content to a doublet content is made.

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
        --out poles.png [--live]

``--isospin-doublet`` adds, to every content's record, the isospin-doublet
observation of `tessera.drivers.isospin_doublet` on h_1 and on h-bar_1; every
other output is unchanged.

``--live`` draws each completed scan point while the scan runs, on an
interactive matplotlib backend, with the computation on a worker thread and the
main thread servicing the GUI event loop; the outputs are identical with or
without it. A non-interactive backend, and WebAgg, are refused by name.

Every tolerance of the stack is an option (``--rank-tolerance`` is tau, the
step's rank decision; the others are listed by `TOLERANCES`), each
defaulting to 1e-15 and each recorded in the configuration. None changes an
equation.

``--villain-order`` is M, the order the Villain weight of the holonomy term
is summed to, an integer from 1 to 10 (10 by default), recorded in the
configuration. It is part of the action: the holonomy term is the function
the order defines, and every content's record carries the reported distance
of the order-M sums from their infinite series (``holonomy_truncation``).
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
from scipy.optimize import linear_sum_assignment

import tessera as T
from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs
from tessera.drivers import cell_solve

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

#: Which fluctuations the Section 7 quartic eliminates: the squared lengths and
#: the link phases (the pure-gauge phase directions, the null space of A, are
#: projected out by the Drazin inverse), or the squared lengths alone.
DECLARED_ELIMINATION = "lengths-and-phases"
ELIMINATIONS = ("lengths-and-phases", "lengths")
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
#: tau, the relative singular-value threshold of the step's rank
#: decision. The Jacobian is assembled analytically, so its entries carry a
#: rounding error of order epsilon relative to their scale, and a singular
#: value below that of the largest cannot be told from zero. The declared
#: threshold sits at that floor: a singular value counts as zero only when it
#: is below the rounding of the Jacobian itself, and a null direction of the
#: Jacobian (a gauge direction, or a redundant constraint row) is inverted at
#: its rounding size rather than left out of the step.
DECLARED_RANK_TOLERANCE = DECLARED_TOLERANCE
#: Tolerances of the certificates this driver grades.
DECLARED_CERTIFICATE_TOLERANCE = DECLARED_TOLERANCE

#: Every tolerance of the stack, by config key, with what it thresholds.
#: `add_tolerance_arguments` offers each as ``--<key, with dashes>``,
#: `default_config` records each, and the recursion driver carries each into
#: every cell's config. None changes an equation.
TOLERANCES = (
    ("rank_tolerance",
     "tau, the relative singular-value threshold of the step's rank "
     "decision: a singular value of the Jacobian below this fraction of the "
     "largest counts as zero in the minimum-norm step; and the same threshold "
     "on the singular values of the face coboundary in the minimum-norm "
     "solve for the declared monopole connection's edge phases"),
    ("step_tolerance",
     "the amount by which a trial of the line search must lower the "
     "residual norm of the stationarity equations to be accepted, and the "
     "residual norm at or below which a level's geometry is stationary"),
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
     "read as the boundary of allowability and its pole read is flagged"),
    ("tie_tolerance",
     "the relative separation of real parts within which two poles tie in "
     "the ascending-real-part order"),
    ("degeneracy_tolerance",
     "the separation within which eigenvalues of the rotation-averaged edge "
     "Laplacian form one band in the spin read and in the fingerprint"),
    ("pole_rank_tolerance",
     "the relative tolerance of every decision of the pole read: the "
     "fraction of a sector block's largest singular value at or below which "
     "two eigenvalues form one pole, and the fraction of the largest singular "
     "value at or below which a singular value counts as zero in the rank of "
     "a residue and of the powers of a pole's nilpotent part"),
    ("fluctuation_tolerance",
     "the relative tolerance of the dressed fluctuation's certificates, and "
     "the size below which a collective mode's geometric component is zero"),
    ("recursion_tolerance",
     "the relative tolerance the level recursion's certificates hold "
     "against"),
    ("spin_sector_tolerance",
     "the size at or below which the measured departures of the spin split "
     "of a colour-singlet block hold: the residual of the block's "
     "two-eigenvalue polynomial (J^2 - 3/4)(J^2 - 15/4), the leakage of J^2 "
     "out of the block, and the distance of each sector projector's trace "
     "from an integer; the split itself is read from the polynomial "
     "projectors and compares no eigenvalue"),
    ("character_tolerance",
     "the modulus at or below which the measured trace of a half turn on the "
     "reference doublet holds as zero; a doublet's character on a half turn "
     "is one by the rotation's cycle type, and no trace is compared to "
     "decide it"),
    ("elimination_tolerance",
     "the relative threshold of the rank decisions of the Drazin "
     "elimination's Feshbach read of the bare stiffness A, and the fraction "
     "of the largest singular value of the pure-gauge directions at or below "
     "which a singular value counts as zero in their rank"),
    ("pure_gauge_tolerance",
     "the relative residual of the null projector of A on the pure-gauge "
     "directions at or below which the null space of A is read as pure "
     "gauge"),
    ("gauge_resonance_radius",
     "the radius of the resonance disc about zero, relative to the spectral "
     "radius of the bare stiffness A, inside which an eigenvalue of A is a "
     "zero-stiffness direction of the Drazin inverse"),
    ("truncation_tolerance",
     "the remainder of the elimination's expansion along the induced "
     "displacement, relative to the term the expansion keeps (the cubic "
     "remainder of the geometric action over its quadratic term, and the "
     "second-order remainder of h_1 over its linear term), above which the "
     "with-quartic read is flagged as outside the range of the expansion"),
    ("hessian_reality_tolerance",
     "the imaginary part of the moment-constrained Hessian's quotient along "
     "the Hellmann-Feynman force, relative to the Hessian's scale, at or "
     "below which the quotient is read as real"),
    ("fibre_lift_tolerance",
     "the relative residual of the T-averaged h_1 from a block-scalar "
     "operator on each doublet times the sheets at or below which the "
     "fibre lift of quark condition 2 holds"),
    ("isotypic_tolerance",
     "the separation at or below which an eigenvalue of the isotypic "
     "projector of the refined cell is one, and at or below which two "
     "eigenvalues of the averaged operator on that isotypic component form "
     "one refined band, in the spectral fingerprint's refinement read; and "
     "the fraction of the largest singular value of that isotypic "
     "component's basis at or below which a singular value counts as zero "
     "in its pseudo-inverse"),
    ("attachment_rank_tolerance",
     "the smallest singular value at or above which the sheet-to-sheet "
     "attachment matrix of quark condition 5 has full rank"),
    ("quotient_rank_tolerance",
     "the relative threshold of the rank decisions of the level recursion's "
     "interior solves: a pivot or a singular value of an interior block at "
     "or below this fraction of the block's largest counts as zero"),
    ("grown_cell_rank_tolerance",
     "the fraction of the largest pivot of a grown cell's metric block (the "
     "inherited pairing on the vertices other than the first) at or below "
     "which a pivot counts as zero, so that the block has no inverse and the "
     "grown-cell rule reads no lengths from it"),
    ("move_tolerance",
     "the amount by which a Pachner move must lower the residual norm of "
     "the stationarity equations to be committed"),
    ("admissibility_tolerance",
     "the Kontsevich-Segal margin, in radians, down to minus which a "
     "geometry proposed by a Pachner move is admissible"),
    ("isospin_grouping_tolerance",
     "the relative width within which eigenvalues form one band of the "
     "isospin-doublet detector"),
    ("isospin_projector_tolerance",
     "the relative idempotency defect at or below which a band's Riesz "
     "projector is certified by the isospin-doublet detector"),
    ("isospin_invariance_tolerance",
     "the relative commutator at or below which a band of the "
     "isospin-doublet detector is invariant under a symmetry element or a "
     "sheet matrix unit"),
    ("isospin_commutant_tolerance",
     "the relative eigenvalue cut of the null space of the commutator map "
     "and of the centre of the commutant in the isospin-doublet detector"),
    ("isospin_isotypic_tolerance",
     "the relative width within which eigenvalues of the commutant's generic "
     "central element form one isotypic component, and the relative "
     "singular-value cut of the rank of each isotypic block, in the "
     "isospin-doublet detector"),
    ("isospin_hermiticity_tolerance",
     "the relative departure of an operator from its adjoint at or below "
     "which the isospin-doublet detector reads it in the Hermitian regime"),
    ("isospin_transport_leakage_tolerance",
     "the relative leakage of a frame-to-frame transport at or below which "
     "the isospin-doublet detector certifies the transport"),
    ("isospin_intertwining_tolerance",
     "the relative intertwining residual of a transport against the "
     "rotation and colour actions at or below which the isospin-doublet "
     "detector certifies it"),
)

#: The tolerances of the isospin-doublet detector, by config key, with the
#: field of `observables.IsospinDoubletConfig` each one sets.
ISOSPIN_TOLERANCES = (
    ("isospin_grouping_tolerance", "grouping_tolerance"),
    ("isospin_projector_tolerance", "projector_tolerance"),
    ("isospin_invariance_tolerance", "invariance_tolerance"),
    ("isospin_commutant_tolerance", "commutant_tolerance"),
    ("isospin_isotypic_tolerance", "isotypic_tolerance"),
    ("isospin_hermiticity_tolerance", "hermiticity_tolerance"),
    ("isospin_transport_leakage_tolerance", "transport_leakage_tolerance"),
    ("isospin_intertwining_tolerance", "intertwining_tolerance"),
)


def declared_tolerance(config, key):
    """The tolerance ``key`` (`TOLERANCES`) of a config, or the declared
    value when the config leaves it out."""
    return float((config or {}).get(key, DECLARED_TOLERANCE))


def isospin_doublet_config(config=None):
    """The `observables.IsospinDoubletConfig` of a config: every tolerance of
    the detector (`ISOSPIN_TOLERANCES`) at the config's value, or at the
    declared value when the config leaves it out. The detector's other
    thresholds stay at the library's values."""
    out = obs.IsospinDoubletConfig()
    for key, field in ISOSPIN_TOLERANCES:
        setattr(out, field, declared_tolerance(config, key))
    return out


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


#: M, the order the Villain weight W(F) = sum_m exp(-m^2/(2 beta)) F^m of the
#: holonomy term is summed to: the term is defined by the sum over |m| <= M
#: (`cob.VillainCharacter`). The weight is an infinite series with no closed
#: elementary form, so its order is declared; ten is the largest order that
#: can be declared and the default. `add_action_arguments` offers it as
#: ``--villain-order``, `default_config` records it, and the recursion driver
#: carries it into every cell's config.
DECLARED_VILLAIN_ORDER = cob.VillainCharacter.maximum_order


def declared_villain_order(config):
    """The order of the Villain weight (`DECLARED_VILLAIN_ORDER`) of a
    config, or the declared value when the config leaves it out."""
    return checked_villain_order((config or {}).get("villain_order",
                                                   DECLARED_VILLAIN_ORDER))


def checked_villain_order(value):
    """``value`` as an order of the Villain weight: an integer from 1 to
    `cob.VillainCharacter.maximum_order`. Anything else is an error."""
    maximum = cob.VillainCharacter.maximum_order
    if isinstance(value, bool) or int(value) != value \
            or not 1 <= int(value) <= maximum:
        raise ValueError("the order of the Villain weight is an integer from "
                         "1 to %d; got %r" % (maximum, value))
    return int(value)


#: The limits a user may declare on a solve, by config key: the type of each
#: and what it ends. None is declared by default, and then nothing ends a
#: solve but the end of its drive, however long it runs.
#: `add_limit_arguments` offers each as ``--<key, with dashes>``,
#: `default_config` records each, and the recursion driver carries each into
#: every cell's config. A declared limit that is reached ends the solve by
#: name ("a declared limit was reached"). None changes an equation.
LIMITS = (
    ("iteration_limit", int,
     "the number of iterations of the drive, each one update of the Pachner "
     "moves and a relaxation of the geometry, after which a solve stops"),
    ("update_limit", int,
     "the number of relaxation updates after each update of the Pachner "
     "moves at which the relaxation is left for the next iteration"),
    ("time_limit_seconds", float,
     "the wall-clock time of one solve, in seconds, after which it stops"),
)

#: The order of the step the drive proposes: the stationarity equations are
#: solved to this order about the point (``direction_order``). Order 1 is
#: Newton's step.
DECLARED_DIRECTION_ORDER = 1

#: The options of a solve's drive, by config key, with the declared value of
#: each and what it sets. `add_solve_arguments` offers each on the command
#: line, `default_config` records each, and the recursion driver carries
#: each into every cell's config.
SOLVE_OPTIONS = (
    ("direction_order", DECLARED_DIRECTION_ORDER,
     "p, the order to which the proposed step solves the stationarity "
     "equations about a point, an integer from 1 to %d: 1 is Newton's step"
     % cell_solve.MAXIMUM_DIRECTION_ORDER),
    ("band_reference", cell_solve.DECLARED_BAND_REFERENCE,
     "where a content's bands are followed from: the last accepted point of "
     "the drive (previous) or the host throughout (host)"),
    ("pachner_moves", True,
     "whether the drive scores and commits Pachner moves of the base "
     "complex beside relaxing its geometry"),
    ("combinatorial_depth", cell_solve.DECLARED_COMBINATORIAL_DEPTH,
     "how many Pachner moves deep the drive searches for a sequence that "
     "lowers the residual norm: when no single move does, it deepens to "
     "sequences of two moves, then three, up to this many, each scored and "
     "committed as a whole; 1 is single moves only"),
    ("combinatorial_length", cell_solve.DECLARED_COMBINATORIAL_LENGTH,
     "a fixed composition length for the search over Pachner moves: "
     "sequences of exactly this many moves are searched first, and the "
     "search backs off one move at a time only when nothing at the current "
     "length lowers the residual norm; 0 keeps the deepening schedule of "
     "the combinatorial depth, of which it is the alternative"),
    ("candidate_moves", 0,
     "the number of Pachner moves drawn per update of the moves; 0 scores "
     "every move"),
    ("moment_stiffness_weight", 0.0,
     "the weight of a spectral-moment stiffness of the degree-1 operator "
     "about the host, added to the action on every sheet; 0 declares none"),
    ("moment_stiffness_coefficients", (),
     "the coefficients of the stiffness's moments of orders 1, 2, ..., "
     "needed with a nonzero weight"),
    ("pinned_vertices", (),
     "base vertices the drive holds: an edge both of whose endpoints are "
     "pinned keeps its squared length and its link; none by default"),
)


def checked_direction_order(value):
    """The order of the step as an integer from 1 to the largest order."""
    if (isinstance(value, bool) or int(value) != value
            or not 1 <= int(value) <= cell_solve.MAXIMUM_DIRECTION_ORDER):
        raise ValueError("the order of the step is an integer from 1 to %d; "
                         "got %r" % (cell_solve.MAXIMUM_DIRECTION_ORDER,
                                     value))
    return int(value)


def declared_solve_options(options=None):
    """Every option of `SOLVE_OPTIONS` by key, at its declared value unless
    ``options`` (a mapping by key) sets it. A key outside `SOLVE_OPTIONS` is
    an error."""
    out = {key: (list(value) if isinstance(value, tuple) else value)
           for key, value, _ in SOLVE_OPTIONS}
    unknown = sorted(set(options or {}) - set(out))
    if unknown:
        raise ValueError("unknown solve options %s; the ones that can be "
                         "set are %s" % (unknown, sorted(out)))
    for key, value in (options or {}).items():
        out[key] = list(value) if isinstance(value, (tuple, list)) else value
    out["direction_order"] = checked_direction_order(out["direction_order"])
    out["combinatorial_depth"], out["combinatorial_length"] = \
        cell_solve.checked_schedule(out["combinatorial_depth"],
                                    out["combinatorial_length"])
    if int(out["candidate_moves"]) < 0:
        raise ValueError("candidate_moves is how many Pachner moves are "
                         "drawn per update, or 0 for every move; got %r"
                         % (out["candidate_moves"],))
    if out["band_reference"] not in cell_solve.BAND_REFERENCES:
        raise ValueError("the band reference is one of %s; got %r"
                         % (", ".join(cell_solve.BAND_REFERENCES),
                            out["band_reference"]))
    return out


def declared_limits(limits=None):
    """Every limit of `LIMITS` by key: None (not declared) unless ``limits``
    (a mapping by key) declares it. A key outside `LIMITS` is an error."""
    out = {key: None for key, _, _ in LIMITS}
    unknown = sorted(set(limits or {}) - set(out))
    if unknown:
        raise ValueError("unknown limits %s; the ones that can be declared "
                         "are %s" % (unknown, [key for key, _, _ in LIMITS]))
    kinds = {key: kind for key, kind, _ in LIMITS}
    out.update({key: None if value is None else kinds[key](value)
                for key, value in (limits or {}).items()})
    return out

#: Where a content's bands are chosen: once, at the declared host, by the
#: ascending real part, and then followed by continuation (WP v17 line 151: a
#: band is selected by a contour, not by sorting real parts); or re-selected
#: by sorting at every iterate, kept as a named option.
DECLARED_BAND_SELECTION = "continuation"
BAND_SELECTIONS = {"continuation": cob.BandSelection.Continuation,
                   "sort-every-iterate": cob.BandSelection.SortEveryIterate}
#: m_c, the number of power sums p_j(h_C), j = 1..m_c, of the occupied fiber
#: the mean-field solve pins at their values at the host (WP v17 §3.4: in
#: controlled synthesis the constraints pin a carrier): "r", the fiber's rank
#: (every moment of the fiber), or a count ("1" pins the trace alone, which
#: removes the uniform dilation of the Euler identity; "0" pins nothing).
DECLARED_FIBER_MOMENTS = "r"
#: What the fiber constraints pin (``fiber_pinning``): the eigenvalue of each
#: occupied band, lambda_b = tr(P_b h_1) / r_b, one constraint per band
#: (``"eigenvalues"``, the default; on the sheeted host every occupied band is
#: one eigenvalue repeated once per sheet, so these are the independent
#: constraints among the fiber's power sums), or the power sums p_j(h_C) of
#: the fiber (``"power-sums"``, WP v17 §3.4 as written), as many as
#: ``--fiber-moments`` says.
DECLARED_FIBER_PINNING = "eigenvalues"
FIBER_PINNINGS = {"eigenvalues": cob.FiberConstraintForm.BandEigenvalues,
                  "power-sums": cob.FiberConstraintForm.PowerSums}
#: The Kontsevich-Segal margin (radians) at or below which a geometry is not
#: on the allowable side: min over tetrahedra of pi minus the sum of the
#: moduli of the arguments of the metric's eigenvalues is pi on a Euclidean
#: cell, zero on a real Lorentzian one and negative beyond. A margin within
#: this of zero is the boundary to within the rounding of the arguments, where
#: WP v17 line 151 reads a band only as the limit of an allowable family and
#: never alone, so a pole read there carries the flag "not Kontsevich-Segal
#: allowable" (`geometry_flags`).
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


def build_base(edge_squared=DECLARED_EDGE_SQUARED, cell=None):
    """The base cell of the three-sheeted host: one tetrahedron on vertices
    0..3 carrying what every sheet of `build_host` carries, the monopole
    connection of `MonopoleSupport.tetrahedron(1)` at the squared length
    ``edge_squared``, or the six squared lengths and links of ``cell``. Its
    squared lengths and links are the shared base field of the host, whose
    sheets are copies of it (`cell_solve.sheeted_support`)."""
    spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    support = monopole_support()
    length = cmath.sqrt(complex(edge_squared))
    pairs = [tuple(e) for e in support.edges]
    for edge in spacetime.getEdgeList().toVector():
        a = int(edge.getSource().getId())
        b = int(edge.getTarget().getId())
        if cell is None:
            edge.setLength(length)
            edge.setPhase(complex(cmath.phase(support.transport(a, b))))
            continue
        m = pairs.index((min(a, b), max(a, b)))
        link = complex(cell["links"][m])
        if a > b:
            link = 1.0 / link
        edge.setLength(cmath.sqrt(complex(cell["squared_lengths"][m])))
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


#: The hinges the primal Regge sum runs over, by the name the configuration
#: and the command line give them (``regge_hinges``, ``--regge-hinges``).
REGGE_HINGES = {"interior": cob.ReggeHinges.Interior,
                "all": cob.ReggeHinges.All}


def checked_regge_hinges(name):
    """The hinges of the primal Regge sum by name, one of `REGGE_HINGES`."""
    if name not in REGGE_HINGES:
        raise ValueError("the hinges of the Regge sum are one of %s; got %r"
                         % (", ".join(REGGE_HINGES), name))
    return REGGE_HINGES[name]


def action_declaration(spacetime, kappa, beta, regge_hinges="interior",
                       matter_weight=1.0,
                       villain_order=DECLARED_VILLAIN_ORDER):
    """The joint action of the calculation (WP §3, §7): the primal Regge term
    with weight 1/kappa over the hinges ``regge_hinges`` names
    (`REGGE_HINGES`; another name has no value), the Villain holonomy term
    with coupling beta and the Villain weight summed to ``villain_order``,
    and the mean-field term; kappa enters only through the Regge weight, and
    the spectral-moment part of S_0 is imposed by the mean-field solve as
    the constraints of WP v17 §3.4 on the occupied fiber."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = 1.0 / kappa
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = checked_regge_hinges(regge_hinges)
    declaration.holonomy_weight = beta
    declaration.villain_order = villain_order
    declaration.matter_weight = matter_weight
    return declaration


def relaxation_declaration(config):
    """The stationarity system over the squared lengths and the links, every
    edge of the complex its own coordinate. `relax_content` ties the sheets
    to one shared base field on top of this (`support_geometry`)."""
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = True
    geometry.relax_links = True
    geometry.relax_multipliers = False
    geometry.rank_tolerance = config["rank_tolerance"]
    # every recorded iterate carries every term of the action with its value
    # and gradient norm (--trace-terms); changes no step
    geometry.record_terms = bool(config.get("trace_terms", False))
    # the monopole sectors a caller holds as boundary data
    # (`tessera.drivers.recursion`); none by default
    geometry.held_sectors = list(config.get("held_sectors") or [])
    return geometry


def support_geometry(config, support, host):
    """`relaxation_declaration` on a sheeted support
    (`cell_solve.sheeted_support`): one squared length and one link per base
    edge, written to every sheet, and the config's held sectors, which are
    declared on the sheeted ``host``, carried to the support's vertices
    (`cell_solve.sectors_on`)."""
    geometry = relaxation_declaration(config)
    geometry.held_sectors = cell_solve.sectors_on(
        list(config.get("held_sectors") or []), host, support)
    geometry.edge_classes = list(support.classes)
    geometry.edge_class_orientations = list(support.orientations)
    return geometry


def node_configuration(config):
    """What the config declares on the `MultiCobordism` node of a solve
    before its drive, as the function `cell_solve.solve` calls with the
    node: the admissibility tolerance of the engine's Kontsevich-Segal gate,
    and, when declared, a spectral-moment stiffness about the host
    (``moment_stiffness_weight`` with ``moment_stiffness_coefficients``, the
    engine's `set_moment_stiffness` on the degree-1 operator) and a pinned
    region (``pinned_vertices``, base vertices whose edges the engine holds).
    Neither a stiffness nor a pinned region is declared by default."""
    def configure(node):
        node.admissibility_tolerance = declared_tolerance(
            config, "admissibility_tolerance")
        weight = float(config.get("moment_stiffness_weight") or 0.0)
        if weight != 0.0:
            node.set_moment_stiffness(
                weight, [1], [float(c) for c in
                              config.get("moment_stiffness_coefficients")
                              or []])
        pinned = [int(v) for v in config.get("pinned_vertices") or []]
        if pinned:
            node.declare_pinned_region("pinned", pinned)
    return configure


def solve_arguments(config):
    """The options of `cell_solve.solve` a config declares: the tolerances,
    the order of the step, the Pachner moves and the limits."""
    return {
        "tolerance": declared_tolerance(config, "step_tolerance"),
        "move_tolerance": declared_tolerance(config, "move_tolerance"),
        "direction_order": checked_direction_order(
            config.get("direction_order", DECLARED_DIRECTION_ORDER)),
        "moves": bool(config.get("pachner_moves", True)),
        "combinatorial_depth": int(config.get(
            "combinatorial_depth", cell_solve.DECLARED_COMBINATORIAL_DEPTH)),
        "combinatorial_length": int(config.get(
            "combinatorial_length",
            cell_solve.DECLARED_COMBINATORIAL_LENGTH)),
        "candidate_moves": int(config.get("candidate_moves", 0)),
        "iteration_limit": config.get("iteration_limit"),
        "update_limit": config.get("update_limit"),
        "time_limit_seconds": config.get("time_limit_seconds"),
        "configure": node_configuration(config),
    }


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


def fiber_moment_count(declaration, action, setting,
                       pinning=DECLARED_FIBER_PINNING):
    """m_c, the number of fiber constraints for a declared setting: an
    integer as it is (zero pins nothing); ``"bands"``, the number of bands
    the content occupies; or ``"r"``, which pinning the band eigenvalues
    (``pinning`` "eigenvalues") is every occupied band, one constraint per
    band, and pinning the power sums is the rank of the occupied fiber, the
    sum of the ranks of the bands the content occupies as the library's own
    band rule reads them at the action's point (`BandFollower`), the paper's
    j = 1..r. On a sheeted host every occupied band is one degenerate
    eigenvalue, one per sheet, so ``"bands"`` power sums are as many as the
    fiber has independent constraints."""
    setting = str(setting)
    if setting == "bands" or (setting == "r" and pinning == "eigenvalues"):
        return int(sum(1 for n in declaration.band_occupations if n > 0))
    if setting != "r":
        return int(setting)
    read = cob.BandFollower(declaration).read(action.carrier_operator())
    return int(sum(band.rank for band in read.bands))


def mean_field_declaration(content, config, spacetime=None):
    """Band filling with the content's occupations (WP v17 §7 line 250) on the
    bands of the covariant operator h_1 itself (ruling (a): Gamma* is a
    projector onto modes of h(z*), WP v17 §7 line 262), the bands chosen at
    the host in ascending order of real part
    (`OccupationOrder.AscendingRealPart`) and followed from there by
    continuation (``band_selection``, WP v17 line 151). With ``spacetime``, the
    system carries the sheeted host's shared base field (`share_sheet_geometry`). The number of
    the occupied fiber's power sums pinned is set by `relax_content`, which
    has the host to read the fiber's rank on (`fiber_moment_count`)."""
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = [float(n) for n in content]
    declaration.band_tolerance = config["band_tolerance"]
    declaration.occupation_order = cob.OccupationOrder.AscendingRealPart
    declaration.band_selection = BAND_SELECTIONS[
        config.get("band_selection", DECLARED_BAND_SELECTION)]
    declaration.tolerance = config["mean_field_tolerance"]
    declaration.fiber_constraint_form = FIBER_PINNINGS[
        config.get("fiber_pinning", DECLARED_FIBER_PINNING)]
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


def sheet_support(spacetime, sheet,
                  tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The `MonopoleSupport` of one sheet of the relaxed host, with its faces
    in the outward orientation of the library fixture. `MonopoleSupport`
    refuses a connection whose moduli depart from one by more than
    ``tolerance``; the U(1) part is taken explicitly
    (`MonopoleSupport.u1Part`) and the departure is reported."""
    fixture = monopole_support()
    links = sheet_links(spacetime, sheet)
    departure = max(abs(abs(u) - 1.0) for u in links)
    support = obs.MonopoleSupport(4, fixture.edges, fixture.faces,
                                  obs.MonopoleSupport.u1Part(links),
                                  tolerance)
    return support, departure


def cell_symmetry(spacetime, supports, tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The rotation group of the relaxed cell itself: the rotations of the
    tetrahedron (`rotation_group()`) under which, on every sheet, the squared
    lengths are invariant and the connection is symmetric up to gauge
    (`MonopoleSupport.gaugeCompensation` on the sheet's support), each within
    ``tolerance``.

    The whitepaper's spin read is made on a symmetric cluster: "the
    configuration is symmetric up to gauge, so every rotation g of the cluster
    is implemented on k-cochains by D_k(g) = rho_k(u_g) P_g" (WP v18 §11.1),
    and refinement-stable spinor doublets are predicted on tetrahedrally
    symmetric supports and nowhere else (WP v18 §4). A relaxed cell is
    therefore read against the group it has when that is the whole
    tetrahedral group, and in the frame of the declared symmetric host, with
    a flag, when it is not (`spin_frame`). The
    connection enters through the sheet support, that is through its U(1)
    part; the departure of the links from the unit circle is reported by
    `sheet_support` and is not part of this decision.

    Per rotation: its vertex permutation, the length departure
    max_e |z_{g e} - z_e| / max_e |z_e| over the sheets, the largest
    gauge-compensation residual over the sheets, and whether both are within
    tolerance. ``group`` lists the symmetric rotations, ``order`` their
    number, and ``tetrahedral`` says whether every rotation of the tetrahedron
    is one."""
    edges = list(itertools.combinations(range(4), 2))
    index = {edge: k for k, edge in enumerate(edges)}
    lengths = [np.asarray(sheet_squared_lengths(spacetime, t))
               for t in range(SHEETS)]
    scale = max(float(np.max(np.abs(z))) for z in lengths)
    rotations = []
    for g in rotation_group():
        image = [index[tuple(sorted((g[a], g[b])))] for a, b in edges]
        length_departure = max(float(np.max(np.abs(z[image] - z)))
                               for z in lengths) / scale
        reads = [support.gaugeCompensation(g, tolerance)
                 for support, _ in supports]
        rotations.append({
            "rotation": [int(v) for v in g],
            "length_departure": length_departure,
            "compensation_residual": max(read.residual for read in reads),
            "symmetric": bool(length_departure <= tolerance
                              and all(read.symmetric for read in reads)),
        })
    group = [r["rotation"] for r in rotations if r["symmetric"]]
    return {
        "rotations": rotations,
        "group": group,
        "order": len(group),
        "tetrahedral": len(group) == len(rotations),
        "tolerance": float(tolerance),
        "length_departure": max(r["length_departure"] for r in rotations),
        "compensation_residual": max(r["compensation_residual"]
                                     for r in rotations),
    }


class NoSpinorDoublet(RuntimeError):
    """`aligned_doublet_frame` on a support none of whose carriers is a
    certified j = 1/2 doublet, so no reference doublet exists to align the
    others to."""


def _band_certificates(block, representations, coexact, nontrivial,
                       tolerance):
    """The certificates of one band of a rotation-invariant edge operator
    (``block``, an orthonormal frame of it), as
    `MonopoleSupport.spinorBands` grades them: the invariance of its
    projector under the projective action, max_g ||D(g) P - P D(g)||_max;
    the irreducibility score (1 / |G|) sum_g |tr D(g)|_W|^2, one on an
    irreducible band; the coexact residual ||P_coexact V - V||_max; whether
    it is a spinor doublet (rank two, invariant and irreducible at
    ``tolerance``, under a nontrivial projective class) and whether it lies
    in the coexact sector at ``tolerance``."""
    projector = block @ block.conj().T
    invariance = max(float(np.max(np.abs(d @ projector - projector @ d)))
                     for d in representations)
    score = float(sum(abs(np.trace(block.conj().T @ d @ block)) ** 2
                      for d in representations) / len(representations))
    residual = float(np.max(np.abs(coexact @ block - block)))
    return {
        "dimension": int(block.shape[1]),
        "invariance_residual": invariance,
        "irreducibility_score": score,
        "coexact_residual": residual,
        "spinor_doublet": bool(nontrivial and block.shape[1] == 2
                               and invariance <= tolerance
                               and abs(score - 1.0) <= tolerance),
        "coexact": bool(residual <= tolerance),
    }


def symmetry_bands(support, group, nontrivial,
                   degeneracy_tolerance=DECLARED_TOLERANCE,
                   tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The symmetry-protected bands of the rotation-averaged edge Laplacian
    of a support, in ascending order of eigenvalue, each with an orthonormal
    frame, the place of its first mode in that order (``start``) and its
    certificates (`_band_certificates`): the reading of
    `MonopoleSupport.spinorBands`, with the frames.

    Under a nontrivial projective class (``nontrivial``) every invariant
    subspace has even rank, so the eigenvalues of a rotation-invariant
    operator are equal in pairs exactly: the two eigenvalues of a pair are
    one unit whatever the rounding of the eigensolver separates them by, and
    a band is a run of units each of which starts within
    ``degeneracy_tolerance`` of the band's first eigenvalue. Under the
    trivial class a unit is one eigenvalue."""
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), group))
    values, vectors = np.linalg.eigh(averaged)
    order = np.argsort(values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    representations = [np.asarray(support.edgeRepresentation(g))
                       for g in group]
    coexact = np.asarray(support.coexactProjector(tolerance))
    unit = 2 if nontrivial and len(values) % 2 == 0 else 1
    bands, start = [], 0
    while start < len(values):
        stop = start + unit
        while (stop < len(values)
               and abs(values[stop] - values[start]) <= degeneracy_tolerance):
            stop += unit
        block = vectors[:, start:stop]
        band = _band_certificates(block, representations, coexact,
                                  nontrivial, tolerance)
        band.update({"eigenvalue": float(np.mean(values[start:stop])),
                     "start": start, "frame": block})
        bands.append(band)
        start = stop
    return bands


def nearest_doublet_text(bands, tolerance):
    """Why no band of `symmetry_bands` is the reference doublet, as text: the
    certificates of the band nearest the coexact sector against the
    tolerance they are graded at."""
    if not bands:
        return "the support has no band"
    band = min(bands, key=lambda b: b["coexact_residual"])
    return ("no band is a certified coexact spinor doublet at the tolerance "
            "%.3g: the band nearest the coexact sector (eigenvalue %.6g, "
            "rank %d) has coexact residual %.3g, invariance residual %.3g "
            "and irreducibility score 1 %+.3g"
            % (tolerance, band["eigenvalue"], band["dimension"],
               band["coexact_residual"], band["invariance_residual"],
               band["irreducibility_score"] - 1.0))


def reference_doublet(bands):
    """The j = 1/2 doublet among the bands of `symmetry_bands`: the first
    rank-two, invariant, irreducible, spinorial band lying in the coexact
    sector (the rule of `MonopoleSupport.spinRead`), or None when there is
    none."""
    for band in bands:
        if band["spinor_doublet"] and band["coexact"]:
            return band
    return None


def aligned_doublet_frame(support, group,
                          degeneracy_tolerance=DECLARED_TOLERANCE,
                          tolerance=DECLARED_CERTIFICATE_TOLERANCE,
                          character_tolerance=DECLARED_TOLERANCE,
                          certified=True):
    """The edge basis in which the three doublets 2, 2', 2'' carry one common
    SU(2) action, each up to its own Z_3 character.

    The three doublets are the eigenspaces of the rotation-averaged edge
    operator (`MonopoleSupport.rotationAveragedEdgeOperator`); each appears
    once in the edge cochains, so they are the isotypic components of the
    projective action D_1(g) and do not depend on the operator averaged. The
    carriers are the three pairs of consecutive eigenvectors in ascending
    order of eigenvalue: under the nontrivial projective class the
    eigenvalues are equal in pairs exactly, so the pairing does not depend
    on how far the eigensolver's rounding separates the two of a pair.

    The reference carrier is the carrier that holds the coexact j = 1/2
    doublet: the first band of `symmetry_bands` that is a certified spinor
    doublet in the coexact sector (`reference_doublet`), named by the place
    of its modes among the carriers and not by its index among the bands.
    With no such band, `NoSpinorDoublet` is raised when ``certified`` is
    true; otherwise the reference is the carrier of least coexact residual
    and the returned ``reference_certified`` is false, with the carrier's
    certificates under ``reference_doublet``.

    With R(g) the action on the reference doublet and M_d(g) the action on
    doublet d, the character is chi_d(g) = tr M_d(g) / tr R(g) on a rotation
    that is not a half turn, and one on a half turn (a rotation of order
    two, the Klein four-group, on which tr R(g) = 0 exactly and which is the
    kernel of the Z_3 character). A half turn is known by the cycle type of
    its permutation, so no computed trace is compared to decide it; the
    largest |tr R(g)| over the half turns is returned as
    ``half_turn_trace_residual``, with whether it is at or below
    ``character_tolerance`` (``half_turn_trace_held``). The intertwiner
    T_d = sum_g chi_d(g)^{-1} M_d(g) X R(g)^{-1} carries R to chi_d^{-1} M_d
    (Schur averaging from a fixed seed X). Columns 2 c + s of the returned
    frame are spin state s of carrier c, the order
    `SharpSpin.doubletSpinMatrices` uses.
    """
    read = support.spinRead(group, degeneracy_tolerance, tolerance)
    nontrivial = bool(read.cocycle.nontrivial)
    averaged = support.rotationAveragedEdgeOperator(support.edgeLaplacian(),
                                                    group)
    values, vectors = np.linalg.eigh(np.asarray(averaged))
    order = np.argsort(values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    blocks = [vectors[:, 2 * c:2 * c + 2] for c in range(3)]
    representations = [np.asarray(support.edgeRepresentation(g))
                       for g in group]
    bands = symmetry_bands(support, group, nontrivial, degeneracy_tolerance,
                           tolerance)
    doublet = reference_doublet(bands)
    coexact = np.asarray(support.coexactProjector(tolerance))
    carriers = [_band_certificates(block, representations, coexact,
                                   nontrivial, tolerance)
                for block in blocks]
    if doublet is not None:
        reference = doublet["start"] // 2
    elif certified:
        raise NoSpinorDoublet(
            "the support carries no j = 1/2 doublet: the carriers' coexact "
            "residuals are %s, invariance residuals %s and irreducibility "
            "scores %s at the tolerance %.3g, the projective class is %s "
            "(%s)" % (
                ["%.3g" % c["coexact_residual"] for c in carriers],
                ["%.3g" % c["invariance_residual"] for c in carriers],
                ["%.17g" % c["irreducibility_score"] for c in carriers],
                tolerance, "nontrivial" if nontrivial else "trivial",
                read.certificate.describe()))
    else:
        reference = min(range(3),
                        key=lambda c: carriers[c]["coexact_residual"])
    actions = {c: [blocks[c].conj().T @ d @ blocks[c]
                   for d in representations] for c in range(3)}
    # a half turn: a rotation other than the identity whose square is the
    # identity
    half_turns = [any(x != y for x, y in enumerate(g))
                  and all(g[g[x]] == x for x in range(len(g)))
                  for g in group]
    half_turn_trace = max(
        (float(abs(np.trace(r))) for r, half in zip(actions[reference],
                                                    half_turns) if half),
        default=0.0)
    seed = np.array([[0.7, 0.2 + 0.1j], [-0.3j, 1.1]])
    frame = np.zeros((BASE_EDGES, BASE_EDGES), dtype=complex)
    residual = 0.0
    for c in range(3):
        if c == reference:
            intertwiner = np.eye(2, dtype=complex)
            characters = [1.0] * len(group)
        else:
            characters = [1.0 if half else np.trace(m) / np.trace(r)
                          for m, r, half in zip(actions[c],
                                                actions[reference],
                                                half_turns)]
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
        "reference_certified": doublet is not None,
        "reference_doublet": dict(
            carriers[reference],
            eigenvalue=float(np.mean(values[2 * reference:
                                            2 * reference + 2]))),
        "intertwining_residual": float(residual),
        "half_turn_trace_residual": half_turn_trace,
        "half_turn_trace_held": bool(half_turn_trace <= character_tolerance),
        "spin_read": read,
        # the bands the frame is read on, without their frames
        "bands": [{key: value for key, value in band.items()
                   if key != "frame"} for band in bands],
    }


#: The two frames a content's spin is read in (`spin_frame`), as the record
#: names them under ``relaxation.spin_frame``.
SPIN_FRAME_OF_THE_CELL = "the relaxed cell's own rotation group"
SPIN_FRAME_OF_THE_HOST = "the declared symmetric host"


def spin_frame(supports, symmetry, degeneracy_tolerance=DECLARED_TOLERANCE,
               tolerance=DECLARED_CERTIFICATE_TOLERANCE,
               character_tolerance=DECLARED_TOLERANCE):
    """The frame the spin of a relaxed cell is read in: the projective
    rotation action D_1(g) on the 18 microscopic edge cells
    (`rotation_action`) and the aligned doublet frame of every sheet
    (`aligned_doublet_frame`), with the flags of the read.

    ``supports`` are the sheets' (`MonopoleSupport`, departure) pairs
    (`sheet_support`) and ``symmetry`` is the cell's `cell_symmetry` at
    ``tolerance``. The spin read of the cell itself has three preconditions,
    each defined on the one before it:

    1. the cell is tetrahedrally symmetric: all twelve rotations of the
       tetrahedron leave its squared lengths invariant and carry its
       connection to a gauge-equivalent one, at ``tolerance``;
    2. the spin read of every sheet's own support names a j = 1/2 doublet, so
       that every sheet has an aligned frame;
    3. the aligned frames of the sheets carry the same doublet labels (the
       Z_3 labels of the three carriers and the reference carrier);
    4. those labels are the three Z_3 characters, one on each carrier, so
       that a doublet content names a carrier content.

    When the four hold, the frame is the cell's own: the actions and the
    aligned frames are built from the sheets' supports. When one does not
    hold, the frame is that of the declared symmetric host: the actions and
    the aligned frame of `monopole_support()`, the same on every sheet, and
    the first precondition that does not hold is recorded as a flag, a dict
    with its ``name`` ("not tetrahedrally symmetric", "no j = 1/2 doublet",
    "the sheets' doublet labels disagree", "the doublet labels are not the
    three Z_3 characters"), a ``detail`` sentence and the numbers that
    decided it. The read proceeds in either frame.

    The host's frame is read whether or not its reference doublet is
    certified at the tolerances (`aligned_doublet_frame` with ``certified``
    false): when it is not, the reference is the carrier of least coexact
    residual and the flag "the host's reference doublet is not certified"
    carries that carrier's certificates.

    Returns a dict with ``name`` (`SPIN_FRAME_OF_THE_CELL` or
    `SPIN_FRAME_OF_THE_HOST`), ``actions``, ``alignments`` (one per sheet)
    and ``flags``."""
    group = rotation_group()
    flags, own = [], None
    if not symmetry["tetrahedral"]:
        flags.append({
            "name": "not tetrahedrally symmetric",
            "detail": (
                "the relaxed cell has %d of the %d rotations of the "
                "tetrahedron as symmetries at the certificate tolerance %.3g "
                "(largest length departure %.3g, largest gauge-compensation "
                "residual %.3g)"
                % (symmetry["order"], len(symmetry["rotations"]), tolerance,
                   symmetry["length_departure"],
                   symmetry["compensation_residual"])),
            "order": int(symmetry["order"]),
            "rotations": len(symmetry["rotations"]),
            "tolerance": float(tolerance),
            "length_departure": float(symmetry["length_departure"]),
            "compensation_residual": float(
                symmetry["compensation_residual"]),
        })
    else:
        try:
            own = [aligned_doublet_frame(support, group, degeneracy_tolerance,
                                         tolerance, character_tolerance)
                   for support, _ in supports]
        except NoSpinorDoublet as missing:
            flags.append({
                "name": "no j = 1/2 doublet",
                "detail": "the spin read of the relaxed cell names no "
                          "reference doublet: %s" % missing,
                "degeneracy_tolerance": float(degeneracy_tolerance),
                "tolerance": float(tolerance),
            })
        else:
            labels = [[list(a["trialities"]), int(a["reference_carrier"])]
                      for a in own]
            if any(label != labels[0] for label in labels):
                flags.append({
                    "name": "the sheets' doublet labels disagree",
                    "detail": "the aligned frames of the sheets carry the "
                              "doublet labels (trialities, reference "
                              "carrier) %s" % labels,
                    "labels": labels,
                })
                own = None
            elif sorted(own[0]["trialities"]) != [0, 1, 2]:
                flags.append({
                    "name": "the doublet labels are not the three Z_3 "
                            "characters",
                    "detail": "the aligned frame of the relaxed cell carries "
                              "the trialities %s on its three carriers, so a "
                              "doublet content names no carrier content"
                              % (list(own[0]["trialities"]),),
                    "trialities": list(own[0]["trialities"]),
                })
                own = None
    if own is not None:
        return {"name": SPIN_FRAME_OF_THE_CELL,
                "actions": rotation_action([support
                                            for support, _ in supports]),
                "alignments": own, "flags": flags}
    host = monopole_support()
    alignment = aligned_doublet_frame(host, group, degeneracy_tolerance,
                                      tolerance, character_tolerance,
                                      certified=False)
    if not alignment["reference_certified"]:
        doublet = alignment["reference_doublet"]
        flags.append({
            "name": "the host's reference doublet is not certified",
            "detail": (
                "no carrier of the declared symmetric host is a certified "
                "j = 1/2 doublet at the certificate tolerance %.3g; the "
                "reference is carrier %d, the carrier of least coexact "
                "residual (coexact residual %.3g, invariance residual %.3g, "
                "irreducibility score %.17g)"
                % (tolerance, alignment["reference_carrier"],
                   doublet["coexact_residual"],
                   doublet["invariance_residual"],
                   doublet["irreducibility_score"])),
            "tolerance": float(tolerance),
            "reference_carrier": int(alignment["reference_carrier"]),
            "coexact_residual": float(doublet["coexact_residual"]),
            "invariance_residual": float(doublet["invariance_residual"]),
            "irreducibility_score": float(doublet["irreducibility_score"]),
        })
    return {"name": SPIN_FRAME_OF_THE_HOST,
            "actions": rotation_action([host] * SHEETS),
            "alignments": [alignment] * SHEETS,
            "flags": flags}


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


def spin_sectors(states, tolerance=DECLARED_TOLERANCE):
    """Split a colour-singlet space of three spin-1/2 quarks by total spin.

    J^2 is applied to each basis state with
    `SharpSpin.applyTotalSpinSquared` and B is its block in the basis. Three
    spins 1/2 add to 1/2 or 3/2, so B has the eigenvalues 3/4 and 15/4 only
    and its sector projectors are polynomials of it,

        P_3/2 = (B - 3/4) / 3,        P_1/2 = (15/4 - B) / 3,

    which no eigenvalue is computed or compared for. A sector's dimension is
    the trace of its projector, an integer, and its basis is the image of
    the projector: with the states orthonormalized (the Cholesky factor of
    their Gram matrix), the projector is Hermitian and the basis is its
    eigenvectors of the largest eigenvalues, as many as the dimension,
    orthonormal as Fock vectors, and real when the states are.

    Returns the sectors by j(j + 1) (a sector of dimension zero is left
    out) and the measurements of the read, none of which decides anything:

    * ``dimensions`` and ``trace_departure``: per sector, the dimension and
      the distance of the projector's trace from that integer;
    * ``polynomial_residual``: ||(B - 3/4)(B - 15/4)||_max, zero exactly
      when B has those two eigenvalues only and is diagonalizable;
    * ``leakage``: ||J^2 V - V B||_max over the basis states V as Fock
      vectors, zero exactly when their span is invariant under J^2;
    * ``eigenvalues``: the eigenvalues of B as computed;
    * ``tolerance`` and ``held``: whether the three departures are at or
      below ``tolerance``."""
    basis = occupation_basis()
    spins = edge_spin_matrices()
    focks = np.column_stack([to_fock(states[:, k], basis)
                             for k in range(states.shape[1])])
    images = np.column_stack([
        np.asarray(obs.SharpSpin.applyTotalSpinSquared(spins, focks[:, k]))
        for k in range(states.shape[1])])
    # J^2 is Hermitian (real symmetric in the occupation basis), so its
    # block in the basis is read by the Hermitian pairing on the Fock
    # vectors.
    gram = focks.conj().T @ focks
    block = np.linalg.solve(gram, focks.conj().T @ images)
    count = block.shape[0]
    identity = np.eye(count)
    # with G = L L^H, the coefficients c = L^-H x are those of the
    # orthonormalized states, in which the block is the Hermitian
    # L^H B L^-H
    factor = np.linalg.cholesky(gram)
    hermitian = factor.conj().T @ block @ np.linalg.inv(factor.conj().T)
    hermitian = (hermitian + hermitian.conj().T) / 2
    if not np.any(hermitian.imag):
        # real states: a real symmetric block, whose eigenvectors are real
        hermitian = hermitian.real
    projectors = {SPIN_HALF: (SPIN_THREE_HALVES * identity - hermitian) / 3,
                  SPIN_THREE_HALVES: (hermitian - SPIN_HALF * identity) / 3}
    sectors, dimensions, departures = {}, {}, {}
    for j2, projector in projectors.items():
        trace = float(np.trace(projector).real)
        dimension = max(0, min(count, int(round(trace))))
        dimensions[j2] = dimension
        departures[j2] = abs(trace - dimension)
        if dimension == 0:
            continue
        _, vectors = np.linalg.eigh(projector)
        picked = vectors[:, count - dimension:]
        sectors[j2] = states @ np.linalg.solve(factor.conj().T, picked)
    polynomial = float(np.max(np.abs(
        (block - SPIN_HALF * identity)
        @ (block - SPIN_THREE_HALVES * identity)))) if count else 0.0
    leakage = float(np.max(np.abs(images - focks @ block))) if count else 0.0
    read = {
        "dimensions": dimensions,
        "trace_departure": departures,
        "polynomial_residual": polynomial,
        "leakage": leakage,
        "eigenvalues": [complex(v) for v in np.linalg.eigvals(block)],
        "tolerance": float(tolerance),
        "held": bool(polynomial <= tolerance and leakage <= tolerance
                     and all(d <= tolerance for d in departures.values())),
    }
    return sectors, read




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
    (`SharpSpin.isotypicProjector`, WP v18 §11.1). They are formed from the
    relaxed cell's own actions and aligned frame; the record of each carries
    its rank and its idempotency residual."""
    maps, characters = lifted_rotation_maps(alignment, actions, frame, dual)
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
    return projectors, record


# -------------------------------------------------------------- the solve


def _geometric_action(spacetime, kappa, beta, config):
    """The geometric part of the action (matter off)."""
    return cob.JointAction(
        spacetime, action_declaration(
            spacetime, kappa, beta, config["regge_hinges"],
            matter_weight=0.0,
            villain_order=declared_villain_order(config)))


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


def drazin_elimination(stiffness, directions, radius,
                       tolerance=DECLARED_TOLERANCE,
                       pure_gauge_tolerance=DECLARED_TOLERANCE):
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

    A stiffness that is zero by structure is eliminated like any other. The
    length stiffness is the Regge term's, which is zero on a complex without
    an interior hinge, so the lengths-only elimination there has A = 0. The
    Drazin inverse of the zero matrix is the zero matrix: every coordinate
    lies in the null space, Pi_0 is the identity, ran(I - Pi_0) is the zero
    subspace, R has no column and the reduced stiffness is the 0 x 0 matrix.
    The elimination then integrates out no fluctuation and contributes
    nothing, and the record says so: ``zero_by_structure`` is whether A is
    the zero matrix exactly, ``eliminated_dimension`` is zero, the two
    residuals are zero exactly (A A^D A - A and R (R^T A R)^{-1} R^T - A^D
    are both the zero matrix) and ``reduced_conditioning`` is None, because
    a 0 x 0 matrix has no condition number. The same holds whenever every
    eigenvalue of A is zero, where A^D is also the zero matrix; for such an
    A other than zero, A A^D A - A = -A and the first residual is one.

    Returns A^D, R, R^T A R and the record: the number of coordinates, the
    resonance disc with its enclosure and separation, whether A is zero by
    structure, the dimensions of the null space and of the eliminated space,
    the idempotency residual of Pi_0, the residual of A A^D A = A relative
    to A, the residual of the reduced form against A^D relative to A^D, the
    condition number of the reduced stiffness and, with ``directions``, how
    the null space compares with the pure-gauge directions.

    ``tolerance`` is the relative threshold of the rank decisions of the
    Feshbach read and of the rank of the pure-gauge directions (the number
    of their singular values above that fraction of the largest), and
    ``pure_gauge_tolerance`` the residual
    ||Pi_0 G - G|| / ||G|| on the pure-gauge directions G at or below which
    the null space is read as pure gauge.
    """
    size = stiffness.shape[0]
    read = T.chainhodge.PencilSchur.feshbach(
        stiffness, np.zeros_like(stiffness), 0j, [], tolerance, radius)
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
    record["zero_by_structure"] = bool(not stiffness.any())
    reduced = basis.T @ stiffness @ basis
    if basis.shape[1] == 0:
        # every coordinate lies in the generalized null space: A^D is the
        # zero matrix, which the empty reduced form rebuilds exactly
        drazin = np.zeros_like(stiffness)
        scale = float(np.linalg.norm(stiffness))
        identity_residual = 1.0 if scale > 0.0 else 0.0
        reduction_residual, conditioning = 0.0, None
    else:
        rebuilt = basis @ np.linalg.solve(reduced, basis.T)
        identity_residual = float(
            np.linalg.norm(stiffness @ drazin @ stiffness - stiffness)
            / np.linalg.norm(stiffness))
        reduction_residual = float(np.linalg.norm(rebuilt - drazin)
                                   / np.linalg.norm(drazin))
        conditioning = float(np.linalg.cond(reduced))
    record.update({
        "null_dimension": int(size - basis.shape[1]),
        "eliminated_dimension": int(basis.shape[1]),
        "projector_idempotency": float(
            np.linalg.norm(null @ null - null) / np.linalg.norm(null))
        if null.any() else 0.0,
        "drazin_identity_residual": identity_residual,
        "reduction_residual": reduction_residual,
        "reduced_conditioning": conditioning,
    })
    if directions is not None:
        # the pure-gauge directions lie in ran(Pi_0), and Pi_0 has no more
        # rank than they span
        singular = np.linalg.svd(directions, compute_uv=False)
        gauge_rank = (int(np.sum(singular > tolerance * singular[0]))
                      if singular.size and singular[0] > 0.0 else 0)
        record["gauge_dimension"] = gauge_rank
        record["gauge_projector_residual"] = float(
            np.linalg.norm(null @ directions - directions)
            / np.linalg.norm(directions))
        record["null_space_is_pure_gauge"] = bool(
            record["null_dimension"] == gauge_rank
            and record["gauge_projector_residual"] <= pure_gauge_tolerance)
    return drazin, basis, reduced, record


def occupied_projector(carrier, occupied):
    """The Riesz projector of the carrier onto its ``occupied`` modes of
    smallest real part (`OccupationOrder.AscendingRealPart`)."""
    values, vectors = np.linalg.eig(carrier)
    order = sorted(range(len(values)),
                   key=lambda k: (values[k].real, values[k].imag))[:occupied]
    return vectors[:, order] @ np.linalg.inv(vectors)[order, :]


def gauge_vertex_function(records, phase_shift):
    """chi, the vertex function of a pure-gauge direction, and the departure
    of the direction from the coboundary of chi.

    A pure-gauge direction moves the phase of each stored edge (x, y) by
    chi_y - chi_x (`gauge_directions`). ``records`` are the stored edges
    (`edge_records`) and ``phase_shift`` the shift of each. chi is read
    along a spanning tree of each connected component: its lowest vertex is
    the root and carries chi = 0, the vertices are reached breadth first,
    the edges of a vertex taken in the order of ``records``, and each tree
    edge gives chi_y = chi_x + shift or chi_x = chi_y - shift. The constant
    of a component is the one freedom of chi and moves no link.

    Returns chi as a dictionary over the vertices and the Euclidean norm of
    shift - (chi_y - chi_x) over all the edges: zero to rounding for a
    coboundary, and the size of what the tree could not absorb for a shift
    that is not one."""
    vertices = sorted({v for pair in records for v in pair})
    neighbours = {v: [] for v in vertices}
    for (x, y), shift in zip(records, phase_shift):
        neighbours[x].append((y, complex(shift)))
        neighbours[y].append((x, -complex(shift)))
    chi = {}
    for root in vertices:
        if root in chi:
            continue
        chi[root] = 0j
        reached = [root]
        for vertex in reached:
            for other, shift in neighbours[vertex]:
                if other not in chi:
                    chi[other] = chi[vertex] + shift
                    reached.append(other)
    departure = math.sqrt(sum(
        abs(complex(shift) - (chi[y] - chi[x])) ** 2
        for (x, y), shift in zip(records, phase_shift)))
    return chi, departure


def gauge_generator(spacetime, phase_shift):
    """Lambda_g, the generator of a pure-gauge direction on the edge
    cochains, and the direction's departure from a coboundary.

    The direction is the gauge transformation U_xy -> g_x^{-1} U_xy g_y with
    g = exp(i chi) (`gauge_vertex_function`). An edge cochain is carried in
    the frame at the lowest vertex of its cell (`CovariantChainHodge`), so
    the transformation acts on the cochains by the diagonal matrix
    D = diag(exp(i chi_{min sigma})) over the degree-one cells sigma in the
    chain complex's order, the mode order of h_1, and h_1 goes to
    D^{-1} h_1 D. The generator is Lambda_g = diag(chi_{min sigma})."""
    chi, departure = gauge_vertex_function(edge_records(spacetime),
                                           phase_shift)
    cells = cob.ChainComplex.fromSpacetime(spacetime).kSimplexVertices(1)
    return (np.diag([chi[min(int(v) for v in cell)] for cell in cells]),
            departure)


def gauge_derivative(coupling, generator):
    """d_g O_a, the derivative of a coupling along a pure-gauge direction.

    Along the direction h_1 is D(s)^{-1} h_1 D(s) with
    D(s) = exp(i s Lambda_g) at every geometry, so each coupling
    O_a = dh_1 / df_a is D(s)^{-1} O_a D(s) and its derivative at s = 0 is
    the commutator i (O_a Lambda_g - Lambda_g O_a), a closed form."""
    return 1j * (coupling @ generator - generator @ coupling)


#: The state the Ward identity of the retained fluctuations is read on
#: (`ward_read`).
WARD_STATE = ("the three modes of h_1 of smallest real part (the occupied "
              "projector of DressedFluctuation), not the bands the content "
              "fills")


def ward_read(spacetime, carrier, couplings, directions, config):
    """The Ward identity of the retained fluctuations: (D - Pi(0)) g = 0 on
    every pure-gauge direction g.

    Pi(0) is `DressedFluctuation.paramagnetic(0)` of the carrier with the
    declared couplings. D g is the diamagnetic term along g,
    (D g)_a = tr(P_occ d_g O_a), with P_occ the occupied Riesz projector and
    d_g O_a the derivative of the coupling O_a along the pure-gauge
    direction, the commutator of O_a with the direction's generator on the
    edge cochains (`gauge_derivative`, `gauge_generator`). The residual is
    ||(D - Pi(0)) g|| / (||Pi(0)|| ||g||), maximized over the directions, and
    ``paramagnetic_alone`` is the largest same ratio for Pi(0) g by itself,
    the size the identity cancels. The generator is read from the phase
    components of a direction; ``coboundary_departure`` is the largest
    distance of a direction's phase shift from the coboundary of the vertex
    function the generator is built on (`gauge_vertex_function`), zero to
    rounding on a pure-gauge direction, and a direction that is not pure
    gauge is read all the same with its departure beside the residual.

    The state is `WARD_STATE`: Pi(0) and P_occ are those of the three modes
    of the carrier of smallest real part, the occupied projector
    `DressedFluctuation` forms, whatever the content the cell was solved
    for. The record names it under ``state``."""
    if directions is None:
        return {"directions": 0, "state": WARD_STATE}
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
                "state": WARD_STATE,
                "unmeasured": str(error)}
    projector = occupied_projector(carrier, 3)
    edges = len(spacetime.getEdgeList().toVector())
    residuals, paramagnetic_parts, departures = [], [], []
    for column in range(directions.shape[1]):
        g = directions[:, column]
        generator, departure = gauge_generator(spacetime, g[edges:])
        departures.append(departure)
        diamagnetic = np.array([
            np.sum(projector * gauge_derivative(o, generator).T)
            for o in couplings])
        polarization = paramagnetic @ g
        scale = np.linalg.norm(paramagnetic) * np.linalg.norm(g)
        residuals.append(float(np.linalg.norm(diamagnetic - polarization)
                               / scale))
        paramagnetic_parts.append(float(np.linalg.norm(polarization)
                                        / scale))
    return {"directions": int(directions.shape[1]),
            "state": WARD_STATE,
            "gauge_derivative": (
                "d_g O_a = i (O_a Lambda_g - Lambda_g O_a), the commutator "
                "with the generator of the gauge direction on the edge "
                "cochains"),
            "coboundary_departure": max(departures),
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
    with the reason.

    The elimination keeps the quadratic term of the action and the linear
    term of h_1, so the with-quartic read is inside the range of its
    expansion when both relative remainders are at or below the config's
    ``truncation_tolerance``. When a measured one is above it, or the
    action's is unmeasured, the read is made as it is and ``flags`` holds
    the flag `QUARTIC_OUTSIDE_EXPANSION` with the remainders and the induced
    displacement (`truncation_flags`); with no induced displacement nothing
    is eliminated and there is no flag."""
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
    out["flags"] = truncation_flags(
        out, declared_tolerance(config, "truncation_tolerance"))
    return out


#: The flag of a with-quartic read whose elimination is outside the range of
#: its expansion (`truncation_flags`).
QUARTIC_OUTSIDE_EXPANSION = ("the elimination is outside the range of its "
                             "expansion")


def truncation_flags(truncation, tolerance=DECLARED_TOLERANCE):
    """The flag of the with-quartic read from the measurements of
    `truncation_certificates`: an empty list when nothing is eliminated (the
    induced displacement is zero) or when both relative remainders are
    measured and at or below ``tolerance``; otherwise the one flag
    `QUARTIC_OUTSIDE_EXPANSION`, a dict with its ``name``, a ``detail``
    sentence, and the numbers that decided it: the two relative remainders
    (None for one that is unmeasured, with the reason), the norms of the
    induced displacement and the tolerance."""
    displacement = float(truncation["induced_displacement_norm"])
    if displacement == 0.0:
        return []
    operator = truncation.get("operator_relative_remainder")
    action = truncation.get("action_relative_remainder")
    unmeasured = truncation.get("action_unmeasured")
    outside = [value for value in (operator, action)
               if value is not None and not value <= tolerance]
    if not outside and not unmeasured:
        return []
    action_text = ("unmeasured (%s)" % unmeasured if action is None
                   and unmeasured else _measured(action, "%.3g"))
    return [{
        "name": QUARTIC_OUTSIDE_EXPANSION,
        "detail": (
            "along the induced displacement (norm %.3g: squared lengths "
            "%.3g, link phases %.3g rad) the second-order remainder of h_1 "
            "relative to its linear term is %s and the cubic remainder of "
            "the geometric action relative to its quadratic term is %s, "
            "against the declared tolerance %.3g; the with-quartic poles "
            "are read as they are" % (
                displacement, truncation["induced_length_norm"],
                truncation["induced_phase_norm"],
                _measured(operator, "%.3g"), action_text, tolerance)),
        "operator_relative_remainder": operator,
        "action_relative_remainder": action,
        "action_unmeasured": unmeasured,
        "induced_displacement_norm": displacement,
        "induced_length_norm": float(truncation["induced_length_norm"]),
        "induced_phase_norm": float(truncation["induced_phase_norm"]),
        "tolerance": float(tolerance),
    }]


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
        stiffness, directions,
        declared_tolerance(config, "gauge_resonance_radius"),
        declared_tolerance(config, "elimination_tolerance"),
        declared_tolerance(config, "pure_gauge_tolerance"))
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


def many_body_operators(carrier, couplings, stiffness, frame, dual,
                        tolerance=DECLARED_TOLERANCE):
    """The two operators the poles are read on, on the three-particle space
    of a fiber's frame (the occupation basis, `occupation_basis`):
    dGamma(Phi~^T h Phi) of the carrier h, and the same with the quartic of
    the eliminated fluctuations, -1/2 J^T A^-1 J over the reduced couplings
    and their stiffness (`DressedFluctuation.effective_action`). Returns the
    one-body operator and the read of the eliminated one: the operator
    (``effective_action``), the number of fluctuations eliminated, the
    stiffness's asymmetry and conditioning, the frames' pairing defect
    ||Phi~^T Phi - I|| and the read's certificate.

    When the elimination integrates out no fluctuation (the reduced
    stiffness is 0 by 0 and there is no reduced coupling: a stiffness that
    is zero by structure, `drazin_elimination`), the quartic is the zero
    operator and the eliminated operator is the one-body operator itself,
    formed here as dGamma of the carrier in the frame (`second_quantized`);
    there is no stiffness, so its asymmetry and conditioning are None."""
    stiffness = np.asarray(stiffness, dtype=complex)
    if len(couplings) == 0 and stiffness.size == 0:
        in_frame = dual @ carrier @ frame
        one_body = second_quantized(in_frame)
        pairing = dual @ frame
        return one_body, {
            "effective_action": one_body,
            "eliminated_dimension": 0,
            "stiffness_asymmetry": None,
            "stiffness_conditioning": None,
            "frame_pairing_defect": float(np.linalg.norm(
                pairing - np.eye(pairing.shape[0]))),
            "certificate": ("no fluctuation is eliminated: the reduced "
                            "stiffness is 0 by 0, the quartic is the zero "
                            "operator and the operator is dGamma of the "
                            "carrier in the frame"),
        }
    declaration = cob.DressedFluctuationDeclaration()
    declaration.carrier_dimension = carrier.shape[0]
    declaration.carrier = list(np.asarray(carrier,
                                          dtype=complex).reshape(-1))
    declaration.couplings = [list(np.asarray(o, dtype=complex).reshape(-1))
                             for o in couplings]
    declaration.bare_stiffness = list(stiffness.reshape(-1))
    declaration.occupied_modes = 3
    declaration.tolerance = tolerance
    read = cob.DressedFluctuation(declaration).effective_action(
        list(frame.reshape(-1)), list(dual.reshape(-1)), 3)
    dimension = int(read.dimension)
    return (np.asarray(read.one_body).reshape(dimension, dimension), {
        "effective_action": np.asarray(read.effective_action).reshape(
            dimension, dimension),
        "eliminated_dimension": len(couplings),
        "stiffness_asymmetry": float(read.stiffness_asymmetry),
        "stiffness_conditioning": float(read.stiffness_conditioning),
        "frame_pairing_defect": float(read.frame_pairing_defect),
        "certificate": read.certificate.describe(),
    })


def sector_poles(operator, sector_states, config=None):
    """The compressed operator of a sector, its leakage, and its poles: the
    eigenvalues of the compressed block, read exactly from its complex Schur
    form by `BoundStatePole` (the block is the whole pencil, with the identity
    metric and every coordinate on the interface), each pole with its
    algebraic multiplicity, its residue (minus the spectral projector onto
    its generalized eigenspace) and the residue's rank, the residual of its
    invariant subspace and its separation from the other poles. Two
    eigenvalues within the config's ``pole_rank_tolerance`` (`TOLERANCES`)
    times the block's largest singular value form one pole."""
    dual = left_inverse(sector_states)
    image = operator @ sector_states
    block = dual @ image
    leakage = np.linalg.norm(image - sector_states @ block) / max(
        np.linalg.norm(image), 1e-300)
    pole_config = cob.BoundStatePoleConfig()
    pole_config.rank_tolerance = declared_tolerance(
        config, "pole_rank_tolerance")
    read = cob.BoundStatePole.poles(block, np.eye(block.shape[0]),
                                    list(range(block.shape[0])), pole_config)
    return block, float(leakage), read


def contents():
    """All ten occupations of the three bands by three quarks."""
    return [c for c in itertools.product(range(4), repeat=3) if sum(c) == 3]


def checked_contents(selected):
    """Contents as declared, each one of the ten of `contents`: three
    non-negative integers, the occupations of the three bands, that sum to
    the three quarks. Another content has no value: the pole read is of
    three-quark sectors."""
    allowed = set(contents())
    out = []
    for content in selected:
        try:
            numbers = tuple(content)
            valid = (all(int(n) == n for n in numbers)
                     and tuple(int(n) for n in numbers) in allowed)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                "a content is three non-negative integers that sum to three "
                "(the occupations of three bands by three quarks); got %r"
                % (content,))
        out.append(tuple(int(n) for n in numbers))
    return out


IRREP_NAMES = ("2", "2'", "2''")


def restriction(j2, triality):
    """The 2T content of a three-quark spin sector of total triality tau.

    The three doublets carry the Z_3 = 2T/Q_8 characters 0, 1, 2 (2, 2',
    2''). A spin-1/2 multiplet of total triality tau restricts to the doublet
    of label tau; a spin-3/2 multiplet restricts to 2' (x) chi^tau plus
    2'' (x) chi^tau, the j = 3/2 quartet being 2' + 2'' on the tetrahedron
    (WP §11.1 line 499). A tetrahedral support therefore cannot tell spin 1/2
    with triality from half of spin 3/2."""
    if j2 == SPIN_HALF:
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
    """Step 1 for one content: the host cell driven to self-consistent
    stationarity by `MultiCobordism` (`cell_solve.solve`), its Pachner moves
    included, as one shared base field, the carried density filling the
    bands of h_1, with the declared number of the occupied fiber's
    constraints pinned at the host.

    The drive runs on the base tetrahedron (`build_base`) and scores the
    three-sheeted system built from it. Returns the three-sheeted complex of
    the base the drive ended on, the action there (carrying the covariance,
    the constraints and their multipliers), the end-point report
    (`SelfConsistentMeanField.read`) and the drive's record."""
    base = build_base(config["edge_squared"], config.get("host_cell"))
    host = cell_solve.sheeted_support(base, SHEETS)
    villain_order = declared_villain_order(config)

    def declare(spacetime):
        return action_declaration(spacetime, kappa, beta,
                                  config["regge_hinges"],
                                  villain_order=villain_order)

    # the number of constraints of the occupied fiber, read at the host;
    # their targets and unit are the host's (`cell_solve.ContentSystem`)
    moments = fiber_moment_count(
        mean_field_declaration(content, config),
        cob.JointAction(host.spacetime, declare(host.spacetime)),
        config.get("fiber_moments", DECLARED_FIBER_MOMENTS),
        config.get("fiber_pinning", DECLARED_FIBER_PINNING))

    def mean_field_of(support):
        declaration = mean_field_declaration(content, config)
        declaration.geometry = support_geometry(config, support, host)
        declaration.fiber_moments = moments
        return declaration

    system = cell_solve.ContentSystem(
        declare, mean_field_of, base, SHEETS,
        config.get("band_reference", cell_solve.DECLARED_BAND_REFERENCE))
    start_scale = max(abs(length * length)
                      for _, _, length, _ in cell_solve.edge_fields(base))
    drive = cell_solve.solve(base, system, **solve_arguments(config))
    support, action, report = system.read(drive["spacetime"], start_scale)
    return support.spacetime, action, report, drive


#: The errors by which the library and the drivers say that an operation
#: has no value (a refused logarithm, a singular block, a decomposition that
#: did not converge, a division by zero): `std::invalid_argument` and
#: `std::domain_error` arrive as ValueError, `std::runtime_error` as
#: RuntimeError, an overflow or a zero divisor as ArithmeticError. A step of
#: a run that meets one is recorded by name with the reason, and the rest of
#: the run is made. Any other error is a defect of the code and is raised.
NO_VALUE_ERRORS = (ValueError, ArithmeticError, RuntimeError)


def solve_converged(residual_norm, tolerance=DECLARED_TOLERANCE):
    """Whether a solve converged, by the one definition every solve's record
    uses (a content's and a level's): the residual norm of the system its
    drive descends, read at the point the drive ended on, is a number at or
    below the declared ``step_tolerance``."""
    return bool(residual_norm <= tolerance)


class ReadWithoutValue(ValueError):
    """A content's pole read that has no value on what its mean-field solve
    reached: ``name`` is the reason by name ("the squared lengths overflowed
    the double"), the message says why with the numbers that decided it,
    ``relaxation`` is the solve's record (`relaxation_record`), and
    ``records`` holds any further read made before the value was found
    missing, keyed as the content record would key it. A read whose
    precondition does not hold is not this: it is made and flagged
    (`geometry_flags`, `spin_frame`)."""

    def __init__(self, name, message, relaxation, **records):
        super().__init__(message)
        self.name = name
        self.relaxation = relaxation
        self.records = records


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
            "ambiguous": bool(band.ambiguous),
            "overfilled": bool(getattr(band, "overfilled", False))}


def band_flags(report):
    """The flags of the occupied bands a mean-field solve ended on: one flag
    per band that holds more particles than its rank
    (`OccupiedBand.overfilled`), with the band, its rank and its occupation.
    The band is filled as declared and the reads are made on it; the flag
    says the filling is not a density of at most one particle per mode."""
    flags = []
    for band in report.bands:
        if not getattr(band, "overfilled", False):
            continue
        flags.append({
            "name": "a band holds more particles than its rank",
            "detail": ("the declared band %d has rank %d and holds %g "
                       "particles, so its filling %g per mode is above one"
                       % (band.declared_index, band.rank, band.occupation,
                          band.occupation / band.rank if band.rank
                          else math.inf)),
            "declared_index": int(band.declared_index),
            "rank": int(band.rank),
            "occupation": float(band.occupation),
        })
    return flags


def eigenbasis_marks(operator):
    """The marks of the eigenbasis the bands of ``operator`` (flat,
    row-major) are read in (`BandFollower.read`): whether a block's
    eigenvector matrix is singular at the threshold of its LU decomposition
    (``defective``; the bands are then read with the inverse as computed)
    and the smallest reciprocal condition number of a block's eigenvector
    matrix. None when the library reports neither. The marks are the
    operator's own, so they are read with one occupied mode, which no
    spectrum refuses."""
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.OccupiedProjector
    declaration.occupied_modes = 1
    read = cob.BandFollower(declaration).read(list(operator))
    if not hasattr(read, "defective"):
        return None
    return {"defective": bool(read.defective),
            "reciprocal_condition": float(
                read.eigenbasis_reciprocal_condition)}


def hessian_sign(value, scale, tolerance=DECLARED_TOLERANCE):
    """The sign of the moment-constrained Hessian along the Hellmann-Feynman
    force as a word: "positive", "negative" or "zero" when the quotient is
    real to ``tolerance`` times the Hessian's scale (the real slice),
    "complex" when it is not, and "unread" when the quotient or the scale is
    not a number."""
    value = complex(value)
    if not (math.isfinite(value.real) and math.isfinite(value.imag)
            and math.isfinite(scale)):
        return "unread"
    if abs(value.imag) > tolerance * scale:
        return "complex"
    if value.real == 0:
        return "zero"
    return "positive" if value.real > 0 else "negative"


def _proposal_record(update):
    """One step proposal of a drive (`cell_solve.StationarityObjective`):
    the residual norm and the step's rank decisions where it was formed, the
    base complex there, and the point's measurements (the force, the
    covariance change, the bands, the pinned moments)."""
    entry = {key: update[key] for key in (
        "residual_norm", "step_norm", "jacobian_rank",
        "largest_singular_value", "smallest_retained_singular_value",
        "largest_discarded_singular_value", "rank_gap", "constrained_step",
        "constrained_rank", "constrained_rank_gap", "linear_residual",
        "complex")}
    step = update.get("measured")
    if step is not None:
        entry.update({
            "force_norm": float(step.force_norm),
            "covariance_change": float(step.covariance_change),
            "band_overlaps": [complex(b.overlap) for b in step.bands],
            "band_positions": [[int(p) for p in b.positions]
                               for b in step.bands],
            "band_crossing": bool(step.band_crossing),
            "band_isolation": float(step.band_isolation),
            "moment_residual_norm": float(step.moment_residual_norm),
            "multipliers": [complex(x) for x in step.multipliers],
            "terms": term_records(step.terms),
        })
    return entry


def relaxation_record(report, drive,
                      hessian_reality_tolerance=DECLARED_TOLERANCE,
                      step_tolerance=DECLARED_TOLERANCE):
    """What a content's solve reached and how, as every content record
    carries it. From the drive (`cell_solve.solve`): whether it converged
    (``converged``, `solve_converged`: the residual norm at the point the
    drive ended on, ``residual``, the last entry of its trace, at or below
    ``step_tolerance``), why it stopped (by name, with its detail), the
    number of accepted relaxation updates (``accepted_updates``) and of
    committed Pachner moves, the base complex before and after, the trace of
    the residual norm, and one entry per step proposal (``trace``). From the
    end-point report (`SelfConsistentMeanField.read`): whether the force and
    the pinned moments are at the mean-field tolerance there
    (``self_consistent``, with the report's sentence), the final
    force, the joint Jacobian's rank and rank gap, the Kontsevich-Segal
    margin and the growth of the lengths, the occupied bands followed to the
    end point, the pinned moments of the occupied fiber (their number,
    targets, multipliers and residuals, in the operator's own unit), and the
    moment-constrained action's Hessian on the range of the Hellmann-Feynman
    force with its sign (WP v17 line 265)."""
    objective = drive["objective"]
    trace = [_proposal_record(update) for update in objective.updates]
    measured = [update["measured"] for update in objective.updates
                if update.get("measured") is not None]
    overlaps = [abs(complex(band.overlap)) for step in measured
                for band in step.bands]
    residual = float(drive["trace"][-1]) if len(drive["trace"]) else math.nan
    converged = solve_converged(residual, step_tolerance)
    return {
        "method": "MultiCobordism drive of the joint action's stationarity",
        "direction_order": int(objective.direction_order),
        "band_selection": _band_selection_name(report.band_selection),
        "converged": converged,
        "residual": residual,
        "step_tolerance": float(step_tolerance),
        "self_consistent": bool(report.converged),
        "self_consistent_detail": str(report.stop_detail),
        "stop_reason": "converged" if converged else drive["stop_reason"],
        "stop_detail": ("the residual norm %.3g is at or below the step "
                        "tolerance %.3g; the drive ended: %s"
                        % (residual, step_tolerance, drive["stop_detail"])
                        if converged else drive["stop_detail"]),
        "drive_stop_reason": drive["stop_reason"],
        "accepted_updates": int(drive["accepted_updates"]),
        "moves_committed": int(drive["moves_committed"]),
        "pachner_moves": bool(drive["moves"]),
        "combinatorial_depth": int(drive["combinatorial_depth"]),
        "combinatorial_length": int(drive["combinatorial_length"]),
        "candidate_moves": int(drive["candidate_moves"]),
        "complex_before": drive["complex_before"],
        "complex_after": drive["complex_after"],
        "complex_changed": bool(drive["changed"]),
        "residual_trace": list(drive["trace"]),
        "undefined_points": len(objective.undefined),
        "seconds": float(drive["seconds"]),
        "force_norm": float(report.force_norm),
        "covariance_change": (float(measured[-1].covariance_change)
                              if measured else 0.0),
        "purity_defect": float(report.purity_defect),
        "spectral_gap": float(report.spectral_gap),
        "band_isolation": float(report.band_isolation),
        "band_ranks": [int(r) for r in report.band_ranks],
        "bands": [_band_record(b) for b in report.bands],
        "band_crossing_iterates": sum(1 for step in measured
                                      if step.band_crossing),
        "lowest_band_overlap": min(overlaps) if overlaps else 1.0,
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
        "fiber_pinning": {form: name for name, form in FIBER_PINNINGS.items()}[
            report.fiber_constraint_form],
        "moment_scale": float(report.moment_scale),
        "moment_targets": [complex(x) for x in report.moment_targets],
        "multipliers": [complex(x) for x in report.multipliers],
        "moment_residuals": [complex(x) for x in report.moment_residuals],
        "hellmann_feynman_force_norm": float(
            report.hellmann_feynman_force_norm),
        "force_hessian": complex(report.force_hessian),
        "force_hessian_scale": float(report.force_hessian_scale),
        "force_hessian_sign": hessian_sign(report.force_hessian,
                                           report.force_hessian_scale,
                                           hessian_reality_tolerance),
        "action": complex(report.action),
        "action_available": bool(report.action_available),
        "action_unavailable": report.action_unavailable,
        "occupied_energy": complex(report.occupied_energy),
        "trace": trace,
    }


#: Why a solved cell has no pole read, by name.
NO_VALUE_OVERFLOW = "the squared lengths overflowed the double"
NO_VALUE_MOVED = "the cell is not a tetrahedron after its Pachner moves"
NO_VALUE_READ = "a read the poles are built on has no value"


def geometry_without_value(spacetime, drive):
    """The name and the message of why the complex a content's solve reached
    has no pole to read, or None when it has. There are two such complexes.
    One whose squared lengths are not finite: the only bound on a length is
    the datatype's, and a squared length beyond the largest finite double is
    not a number, so there is no finite geometry to read an operator on. And
    one whose base is not one tetrahedron: the pole read is defined on the
    three-sheeted tetrahedron (its eighteen edge modes, its rotation group,
    its doublets), and has no definition on another complex. The test is on
    the base the drive ended on (``drive["spacetime"]``), one cell of four
    vertices and six edges, whatever its vertices are called and whatever
    moves led to it: a drive whose moves returned the base to a tetrahedron
    is read. A solve that stopped for any other reason is read, and its
    record says why it stopped."""
    for index, (_, _, length, _) in enumerate(
            cell_solve.edge_fields(spacetime)):
        squared = length * length
        if not (math.isfinite(squared.real) and math.isfinite(squared.imag)):
            return (NO_VALUE_OVERFLOW,
                    "%s (|z| is not finite on edge %d), so there is no "
                    "finite geometry to read a pole on"
                    % (NO_VALUE_OVERFLOW, index))
    after = cell_solve.complex_counts(drive["spacetime"])
    if after != {"vertices": 4, "edges": 6, "cells": 1}:
        return (NO_VALUE_MOVED,
                "%s (%d committed move updates left a base complex of %d "
                "vertices, %d edges and %d cells), and the pole read is "
                "defined on the three-sheeted tetrahedron"
                % (NO_VALUE_MOVED, drive["moves_committed"],
                   after["vertices"], after["edges"], after["cells"]))
    return None


def geometry_flags(report, tolerance=DECLARED_ALLOWABILITY_TOLERANCE):
    """The flags of the geometry a mean-field solve reached: the
    preconditions of the pole read that the geometry does not meet, each a
    dict with its ``name``, a ``detail`` sentence and the numbers that
    decided it. The poles are read on a flagged geometry, and the flags are
    carried beside them.

    The one precondition on the geometry is that it is Kontsevich-Segal
    allowable: bands are read on the allowable side (WP v17 line 151), and a
    margin within ``tolerance`` of zero is the boundary. A geometry whose
    margin is not above ``tolerance`` carries the flag "not Kontsevich-Segal
    allowable" with the margin and the tolerance. An empty list says the
    geometry is allowable."""
    margin = float(report.kontsevich_segal_margin)
    if margin > tolerance:
        return []
    return [{
        "name": "not Kontsevich-Segal allowable",
        "detail": ("the geometry the mean-field solve reached is not "
                   "Kontsevich-Segal allowable (margin %.3g, at or below "
                   "the declared tolerance %.0e), and bands are read on the "
                   "allowable side (WP v17 line 151)" % (margin, tolerance)),
        "kontsevich_segal_margin": margin,
        "tolerance": float(tolerance),
    }]


def content_flags(record, column=None):
    """Every flag of one content record: the flags of its geometry and of
    its spin frame (``flags``) and the flag of its with-quartic read
    (``quartic.truncation.flags``, `truncation_flags`), each name once.
    With ``column`` "quasi_free", the flags that bear on that column, which
    leaves out the with-quartic read's."""
    flags = list(record.get("flags") or [])
    names = {flag["name"] for flag in flags}
    quartic = ((record.get("quartic") or {}).get("truncation") or {}).get(
        "flags") or []
    flags += [flag for flag in quartic if flag["name"] not in names]
    if column == "quasi_free":
        flags = [flag for flag in flags
                 if flag["name"] != QUARTIC_OUTSIDE_EXPANSION]
    return flags


def flags_text(record):
    """The flags of one content record as text (`content_flags`: those of
    `geometry_flags`, `spin_frame` and `truncation_flags`): each flag's name
    with its detail, and the frame the spin was read in when the record
    names it. Empty when the record carries no flag."""
    flags = content_flags(record)
    if not flags:
        return ""
    text = "flagged: " + "; ".join("%s (%s)" % (flag["name"], flag["detail"])
                                   for flag in flags)
    frame = (record.get("relaxation") or {}).get("spin_frame")
    if frame:
        text += "; spin read in the frame of %s" % frame
    return text


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
    """The bands of an operator as the band rule groups them, read by the
    rule itself (`cobordism.BandFollower.read`, the read the mean field
    fills its bands by): eigenvalues in ascending order of real part, the
    groups of that read at the band tolerance ``tolerance``, and each band's
    Riesz projector V_b (V^-1)_b from the read's right and left
    eigenvectors. Returns (eigenvalues, projector) per band, in the order of
    the read's ``ranks``."""
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    # the read needs a filling to be declared; one particle in the lowest
    # band fits a band of any rank and plays no part in the grouping
    declaration.band_occupations = [1.0]
    declaration.band_tolerance = tolerance
    declaration.occupation_order = cob.OccupationOrder.AscendingRealPart
    operator = np.asarray(operator, dtype=complex)
    read = cob.BandFollower(declaration).read(list(operator.reshape(-1)))
    count = operator.shape[0]
    values = np.asarray(read.eigenvalues, dtype=complex)
    vectors = np.asarray(read.eigenvectors, dtype=complex).reshape(count,
                                                                   count)
    inverse = np.asarray(read.left_eigenvectors, dtype=complex).reshape(
        count, count)
    # the read's declared order: ascending real part, equal keys in the
    # order of its eigenvalues (a stable sort, as the read's own)
    order = sorted(range(count), key=lambda k: values[k].real)
    bands, start = [], 0
    for rank in read.ranks:
        group = order[start:start + int(rank)]
        bands.append((values[group], vectors[:, group] @ inverse[group, :]))
        start += int(rank)
    return bands


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

    * ``band_occupations``: tr(Gamma P_b) of every band b of h_1 at the
      point read, in ascending order of the bands (`band_projectors`): the
      number of quarks the state holds in that band, the declared occupation
      on a band the content fills and zero on the others;
    * ``occupied_bands``: the bands the state occupies, as many as the
      content has nonzero occupations: the bands of largest
      |tr(Gamma P_b)|, in ascending order. They are read off the state and
      not off the content's indices, because a band followed by continuation
      keeps its declared index while its place among the bands of the point
      changes when it crosses another. Each with its place ``band``, its
      measured ``occupation`` tr(Gamma P_b), its rank r_b, its eigenvalues,
      and its overlap with each doublet d of h-bar_1, tr(P_b Pbar_d) / r_b
      (complex, because P_b is a Riesz projector of a non-normal operator;
      the three overlaps sum to one);
    * ``occupied_state``: the number of quarks in each doublet,
      tr(Gamma Pbar_d), which sums to the number of quarks;
    * ``commutator``: ||[Gamma, h_1]|| / (||Gamma|| ||h_1||), zero to rounding
      when Gamma is built from the bands of h_1."""
    bands = band_projectors(carrier, tolerance)
    doublets = doublet_projectors(frame, dual, trialities)
    occupations = [complex(np.trace(covariance @ projector))
                   for _, projector in bands]
    filled = sum(1 for n in content if n != 0)
    # a stable sort, so equal occupations keep the bands' ascending order
    places = sorted(sorted(range(len(bands)),
                           key=lambda b: -abs(occupations[b]))[:filled])
    occupied = []
    for b in places:
        values, projector = bands[b]
        rank = len(values)
        occupied.append({
            "band": b, "occupation": occupations[b], "rank": rank,
            "eigenvalues": [complex(v) for v in values],
            "overlap": {name: complex(np.trace(projector @ d) / rank)
                        for name, d in doublets.items()},
        })
    commutator = covariance @ carrier - carrier @ covariance
    return {
        "band_ranks": [len(v) for v, _ in bands],
        "band_occupations": occupations,
        "occupied_bands": occupied,
        "occupied_state": {name: complex(np.trace(covariance @ d))
                           for name, d in doublets.items()},
        "commutator": float(np.linalg.norm(commutator)
                            / (np.linalg.norm(covariance)
                               * np.linalg.norm(carrier))),
    }


_DOUBLET_SECTORS = {}


class DoubletLabelsWithoutValue(ValueError):
    """A doublet content read against an aligned frame whose trialities are
    not the three Z_3 characters, one on each carrier: the doublets 2, 2',
    2'' then name no three carriers, and the doublet content has no carrier
    content."""


def _carrier_content(doublet_content, trialities):
    """The carrier content a doublet content (n_2, n_2', n_2'') names in an
    aligned frame's carrier order."""
    if sorted(int(t) for t in trialities) != [0, 1, 2]:
        raise DoubletLabelsWithoutValue(
            "the aligned frame's trialities %s are not the three Z_3 "
            "characters" % (list(trialities),))
    carrier_of = {IRREP_NAMES[t]: c for c, t in enumerate(trialities)}
    carrier_content = [0, 0, 0]
    for name, n in zip(IRREP_NAMES, doublet_content):
        carrier_content[carrier_of[name]] = int(n)
    return carrier_content


def _doublet_sector_entry(carrier_content, tolerance):
    key = (tuple(carrier_content), float(tolerance))
    if key not in _DOUBLET_SECTORS:
        states, _ = singlet_states(carrier_content)
        _DOUBLET_SECTORS[key] = spin_sectors(states, tolerance)
    return _DOUBLET_SECTORS[key]


def doublet_sectors(doublet_content, trialities,
                    tolerance=DECLARED_TOLERANCE):
    """The colour-singlet, one-quark-per-sheet spin sectors of a doublet
    content (n_2, n_2', n_2''): the carrier content in the aligned frame's
    carrier order, the total triality, and the sectors by total spin
    (`spin_sectors`, whose measurements `doublet_sector_read` returns). They
    depend on the declared frame and the tolerance only, so they are formed
    once for each. An aligned frame whose trialities are not the three Z_3
    characters gives a doublet content no carrier content
    (`DoubletLabelsWithoutValue`)."""
    carrier_content = _carrier_content(doublet_content, trialities)
    triality = sum(n * t for n, t in zip(carrier_content, trialities)) % 3
    return (carrier_content, triality,
            _doublet_sector_entry(carrier_content, tolerance)[0])


def doublet_sector_read(doublet_content, trialities,
                        tolerance=DECLARED_TOLERANCE):
    """The measurements of the spin split of a doublet content's
    colour-singlet states (the second return of `spin_sectors`): the
    sectors' dimensions, the departure of each projector's trace from its
    integer, the residual of the block's two-eigenvalue polynomial, the
    leakage of J^2 out of the states' span, and whether they hold at
    ``tolerance``."""
    return _doublet_sector_entry(
        _carrier_content(doublet_content, trialities), tolerance)[1]


def sector_spin_images(sector, dual, basis, spins):
    """What the certificates of a sector's poles are read from, formed once
    for the sector: the Fock vectors of the sector's states (``right``,
    one column per state) and of its bilinear dual (``left``, column k the
    k-th row of ``dual``), the image of each right column under J^2
    (``right_spin``) and of each left column under its transpose
    (``left_spin``; the left eigen-equation <Psi_L|(J^2 - j(j+1)) = 0 is the
    right one for (J^2)^T, which is J^2 of the transposed one-particle spin
    matrices), and the image of each right column under the colour Casimir
    (``right_casimir``)."""
    count = sector.shape[1]
    right = np.column_stack([to_fock(sector[:, k], basis)
                             for k in range(count)])
    left = np.column_stack([to_fock(dual[k, :], basis)
                            for k in range(count)])
    transposed = [np.asarray(j).T for j in spins]
    apply = obs.SharpSpin.applyTotalSpinSquared
    return {
        "right": right,
        "left": left,
        "right_spin": np.column_stack([np.asarray(apply(spins, right[:, k]))
                                       for k in range(count)]),
        "left_spin": np.column_stack([np.asarray(apply(transposed,
                                                       left[:, k]))
                                      for k in range(count)]),
        "right_casimir": np.column_stack([colour_casimir(right[:, k])
                                          for k in range(count)]),
    }


def pole_projectors(read, dimension):
    """The spectral projector of a compressed block onto the generalized
    eigenspace of each pole of ``read`` (`sector_poles`), in the order of
    the poles: minus the pole's residue, which `BoundStatePole` forms from
    the block's Schur form (for the identity metric on a block with no
    interior the residue is minus the spectral projector itself)."""
    return [-np.asarray(residue, dtype=complex).reshape(dimension, dimension)
            for residue in read.residue]


def _ratio(numerator, pairing):
    """A matrix element over its pairing as a complex number; not a number
    when the pairing vanishes or is not finite, and never an exception."""
    pairing = complex(pairing)
    if pairing == 0 or not np.isfinite(pairing):
        return complex(math.nan, math.nan)
    return complex(numerator) / pairing


def _worst_relative(images, states):
    """sup over the span of the columns of ``states`` of ||A psi|| / ||psi||,
    for ``images`` the columns' images under A: the largest singular value
    of the images in an orthonormal basis of the span. It does not depend on
    the basis the span is given in."""
    orthonormal, triangular = np.linalg.qr(states)
    in_basis = np.linalg.solve(triangular.T, images.T).T
    return float(np.linalg.norm(in_basis, 2))


def pole_certificates(projector, multiplicity, j2, sector, dual, images,
                      isotypic=None, spinor_type=None,
                      tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The certificates of one pole of a sector's compressed block, read on
    the pole's eigenspace: ``projector`` is the block's spectral projector
    onto it (`pole_projectors`) and ``multiplicity`` its dimension.

    The right eigenspace is the range of the projector and the left one the
    range of its transpose, both taken from the one projector, so a state
    and its partner come from one decomposition and their pairing does not
    vanish. Every number below is a property of the eigenspace and not of a
    basis of it, so the certificates of a pole of multiplicity above one do
    not depend on the vectors an eigensolver returns inside it:

    * the spinor certificate of WP v18 §11.1 and §14: the isotypic projector
      equations (I - P)|Psi_R> = 0 and <Psi_L|(I - P) = 0 with ``isotypic``
      the projector onto the sector's 2T types (`isotypic_projectors`),
      named ``spinor_type``; each residual is the largest
      ||(I - P) psi|| / ||psi|| over the eigenspace, and the weight is
      tr(P Pi) / tr(Pi), the mean of <Psi_L|P|Psi_R> / <Psi_L|Psi_R> over a
      biorthogonal basis; unmeasured when no projector is supplied;
    * the spin-lift read: the two eigen-equations of J^2 at the sector's
      j(j+1) under the constructed SU(2) action, each residual the largest
      ||(J^2 - j(j+1)) psi|| / ||psi|| over the eigenspace, and the
      biorthogonal expectation tr(J^2 Pi) / tr(Pi), which states the
      continuum spin value an accepted lift supplies;
    * the largest relative residual of the colour Casimir over the right
      eigenspace (zero for colour singlets);
    * ``determinant_count``: the number of determinants that carry a nonzero
      amplitude of some state of the right eigenspace.

    A certificate is sharp when both of its residuals are at or below
    ``tolerance`` and its expectation (or weight) is a finite number: a
    pairing that vanishes or is not a number is never read as sharp.
    ``images`` are the sector's `sector_spin_images` and ``dual`` its
    bilinear left inverse."""
    count = int(multiplicity)
    left_vectors, _, right_h = np.linalg.svd(projector)
    right = left_vectors[:, :count]
    left = right_h[:count, :].T
    states = images["right"] @ right
    partners = images["left"] @ left
    right_residual = _worst_relative(
        images["right_spin"] @ right - j2 * states, states)
    left_residual = _worst_relative(
        images["left_spin"] @ left - j2 * partners, partners)
    expectation = _ratio(
        np.trace(images["left"].T @ images["right_spin"] @ projector),
        np.trace(images["left"].T @ images["right"] @ projector))
    finite = bool(np.isfinite(expectation))
    out = {
        "eigenspace_dimension": count,
        "sharp_spinor": None,
        "spinor_type": spinor_type,
        "spinor_right_residual": None,
        "spinor_left_residual": None,
        "spinor_weight": None,
        "spin_lift_sharp": bool(finite and right_residual <= tolerance
                                and left_residual <= tolerance),
        "spin_lift_right_residual": right_residual,
        "spin_lift_left_residual": left_residual,
        "spin_lift_expectation": expectation,
        "determinant_count": int(np.count_nonzero(np.any(states != 0,
                                                         axis=1))),
        "colour_casimir_residual": _worst_relative(
            images["right_casimir"] @ right, states),
    }
    if isotypic is not None:
        patterns = sector @ right
        dual_patterns = dual.T @ left
        spinor_right = _worst_relative(patterns - isotypic @ patterns,
                                       patterns)
        spinor_left = _worst_relative(
            dual_patterns - isotypic.T @ dual_patterns, dual_patterns)
        weight = _ratio(np.trace(dual @ isotypic @ sector @ projector),
                        np.trace(dual @ sector @ projector))
        out.update({
            "sharp_spinor": bool(np.isfinite(weight)
                                 and spinor_right <= tolerance
                                 and spinor_left <= tolerance),
            "spinor_right_residual": spinor_right,
            "spinor_left_residual": spinor_left,
            "spinor_weight": weight,
        })
    return out


def lowest_pole(poles):
    """The pole of smallest real part, then smallest imaginary part, among
    those whose real part is a number; the first pole when none is; None
    for no pole. The choice does not depend on where in the list a pole
    that is not a number stands."""
    numbers = [p for p in poles if not math.isnan(p.real)]
    if numbers:
        return min(numbers, key=lambda p: (p.real, p.imag))
    return poles[0] if poles else None


def sector_entry(j2, triality, sector, operators, projectors=None,
                 config=None):
    """One spin sector's reads: its restriction to 2T, the weight of each
    isotypic type in it (``projectors``, `isotypic_projectors`: the trace of
    the type's projector compressed to the sector over the sector's
    dimension, which is one on the sector's own types and zero on the others
    when the constructed lift agrees with the finite group), and for each
    named many-body operator the poles of the compressed block with their
    multiplicities and the exact certificates of the read (the residual of
    each pole's invariant subspace, the rank of each residue, the separation
    of each pole from the others and the block's scale, `sector_poles`),
    each pole's spinor, spin-lift and colour certificates on its eigenspace
    (``pole_certificates``, parallel to ``poles``), and the lowest pole's
    certificates repeated beside it. The poles and the certificates are read
    at the config's tolerances (`TOLERANCES`)."""
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
    images = sector_spin_images(sector, dual, basis, spins)
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
        lowest = lowest_pole(poles)
        # every pole's certificates, on the eigenspace its spectral
        # projector projects onto
        size = block.shape[0]
        certificates = [
            pole_certificates(pole_projector, multiplicity, j2, sector, dual,
                              images, projector, spinor_type,
                              certificate_tolerance)
            for pole_projector, multiplicity in zip(
                pole_projectors(read, size), read.multiplicity)]
        # with no pole read, the certificates beside the lowest pole are
        # those of the whole sector
        lowest_certificates = (
            certificates[poles.index(lowest)] if lowest is not None else
            pole_certificates(np.eye(size, dtype=complex), size, j2, sector,
                              dual, images, projector, spinor_type,
                              certificate_tolerance))
        entry[name] = {
            "poles": poles,
            "multiplicity": [int(m) for m in read.multiplicity],
            "lowest_pole": lowest,
            "failed_certificates": list(read.failed_certificates),
            "subspace_residual": [float(x) for x in read.subspace_residual],
            "residue_rank": [int(r) for r in read.residue_rank],
            "separation": [float(x) for x in read.separation],
            "scale": float(read.scale),
            "compression_leakage": leakage,
            "pole_certificates": certificates,
        }
        entry[name].update(lowest_certificates)
    return entry


def evaluate_content(content, kappa, beta, config):
    """One content at one scan point: relaxation, recursion, quark conditions,
    states, spin, operators and poles.

    The poles are read on the geometry the mean-field solve reached, whatever
    it is. A precondition of the read that does not hold at its declared
    tolerance is recorded as a flag under ``flags`` (`geometry_flags` for
    the geometry, `spin_frame` for the spin read of the cell itself), and
    the frame the spin was read in is named under ``relaxation.spin_frame``.
    A solve that left no finite geometry, or a complex that is not the
    tetrahedron, has no pole read and raises `ReadWithoutValue`
    (`geometry_without_value`).

    A read of the solved cell that has no value (`NO_VALUE_ERRORS`) is
    recorded by name and the rest is kept. The reads the poles are built on
    (the spin frame, the spin decomposition, the eliminated fluctuations,
    the many-body operators, the sectors, the end-point measurements of the
    solve) leave the content without a pole: `ReadWithoutValue` with the
    name `NO_VALUE_READ`, the read that had no value and the library's
    reason, carrying the solve's record. The reads beside the poles (the
    level-recursion read, the anchor atlas, the spectral fingerprint, the
    quark conditions, the isospin doublet) leave the poles as they are: the
    record omits that read and carries a flag that names it with the
    reason.

    A band the solve ended on that holds more particles than its rank
    (`band_flags`) and a carrier operator whose eigenbasis is singular at
    the threshold of its LU decomposition (`eigenbasis_marks`, recorded
    under ``carrier_eigenbasis``) are read as they are and flagged."""
    started = time.time()
    spacetime, action, report, drive = relax_content(content, kappa, beta,
                                                     config)
    solve = relaxation_record(
        report, drive,
        declared_tolerance(config, "hessian_reality_tolerance"),
        declared_tolerance(config, "step_tolerance"))
    missing = geometry_without_value(spacetime, drive)
    if missing is not None:
        raise ReadWithoutValue(missing[0], missing[1], solve)
    flags = geometry_flags(
        report, declared_tolerance(config, "allowability_tolerance"))
    flags = flags + band_flags(report)
    # the read in progress, by name, for the record of one without a value
    stage = ["the carrier operator"]
    try:
        return _content_reads(content, kappa, beta, config, started,
                              spacetime, action, report, solve, flags, stage)
    except ReadWithoutValue:
        raise
    except NO_VALUE_ERRORS as error:
        raise ReadWithoutValue(
            NO_VALUE_READ, "%s: %s (%s)" % (NO_VALUE_READ, stage[0], error),
            solve) from error


def _content_reads(content, kappa, beta, config, started, spacetime, action,
                   report, solve, flags, stage):
    """The reads of `evaluate_content` on the cell its solve reached;
    ``stage`` (a one-entry list) names the read in progress."""
    carrier = matrix(action.carrier_operator())
    certificate_tolerance = declared_tolerance(config, "certificate_tolerance")

    # The relaxation under h_1, which is not itself rotation-covariant, moves
    # the fields off the symmetric configuration. The rotation group of the
    # relaxed cell is measured (`cell_symmetry`), and the spin is read with
    # the projective action D_1(g) of the cell itself when the cell meets
    # the preconditions of that read, and with that of the declared symmetric
    # host, flagged, when it does not (`spin_frame`).
    stage[0] = "the spin frame"
    supports = [sheet_support(spacetime, t, certificate_tolerance)
                for t in range(SHEETS)]
    symmetry = cell_symmetry(spacetime, supports, certificate_tolerance)
    spin_read_frame = spin_frame(
        supports, symmetry, declared_tolerance(config, "degeneracy_tolerance"),
        certificate_tolerance,
        declared_tolerance(config, "character_tolerance"))
    flags = flags + spin_read_frame["flags"]
    actions = spin_read_frame["actions"]
    alignments = spin_read_frame["alignments"]
    frame = _micro_frame(alignments)
    dual = np.linalg.inv(frame)
    averaged = rotation_averaged(carrier, actions)
    band_energies, averaged_residual = _block_scalar(dual @ averaged @ frame)
    _, covariant_residual = _block_scalar(dual @ carrier @ frame)
    trialities = alignments[0]["trialities"]
    stage[0] = "the spin decomposition"
    spin = spin_decomposition(carrier, matrix(action.declaration.covariance),
                              content, frame, dual, trialities,
                              config["band_tolerance"])
    # the isotypic projectors of the double cover on the three-particle
    # sector, the spinor certificate's projectors (WP v18 §11.1, §14)
    projectors, isotypic = isotypic_projectors(alignments[0], actions, frame,
                                               dual)

    # the quartic's ingredients, on h_1 itself: the retained fluctuations
    # (the squared lengths, and the link phases unless only the lengths are
    # declared), their couplings O_a, the bare stiffness A of the geometric
    # action and its Drazin inverse A^D
    stage[0] = "the elimination of the fluctuations"
    fluctuations = eliminate_fluctuations(spacetime, action, carrier, kappa,
                                          beta, config)
    couplings = fluctuations["couplings"]
    shift = fluctuations["shift"]
    constant = fluctuations["constant"]
    truncation = fluctuations["truncation"]
    # the with-quartic read's flag is a flag of the content: it is counted
    # among the flagged contents and named in the content's line
    names = {flag["name"] for flag in flags}
    flags = flags + [flag for flag in truncation.get("flags") or []
                     if flag["name"] not in names]

    # the reduced coordinates f = R g carry the Drazin elimination
    coupling_matrices = fluctuations["reduced_couplings"]
    stage[0] = "the many-body operators"
    # quasi-free: dGamma of the T-averaged operator; with the quartic: the
    # whole eliminated operator averaged over the diagonal rotation action,
    # term by term (a rotated carrier and rotated couplings give the rotated
    # many-body operator)
    quasi_free, _ = many_body_operators(
        averaged, coupling_matrices, fluctuations["reduced_stiffness"],
        frame, dual, declared_tolerance(config, "fluctuation_tolerance"))
    dimension = quasi_free.shape[0]
    _, quartic = many_body_operators(
        carrier + shift, coupling_matrices, fluctuations["reduced_stiffness"],
        frame, dual, declared_tolerance(config, "fluctuation_tolerance"))
    with_quartic = rotation_averaged_many_body(
        quartic["effective_action"], actions, frame, dual) + \
        constant * np.eye(dimension)

    # the poles of the colour-singlet three-quark sectors of h-bar_1, for
    # every doublet content (n_2, n_2', n_2''): a content of this driver names
    # occupations of the bands of h_1, which are not doublets, so the sectors
    # are read for every doublet content and labelled by it
    stage[0] = "the poles of the spin sectors"
    doublet_reads = []
    for doublet_content in contents():
        sector_reads = {}
        carrier_content, triality, sectors = doublet_sectors(
            doublet_content, trialities,
            declared_tolerance(config, "spin_sector_tolerance"))
        for j2, sector in sectors.items():
            sector_reads[j2] = sector_entry(j2, triality, sector, (
                ("quasi_free", quasi_free), ("with_quartic", with_quartic)),
                projectors, config)
        doublet_reads.append({
            "doublet_content": list(doublet_content),
            "carrier_content": carrier_content,
            "total_triality": int(triality),
            "sectors": {str(k): v for k, v in sector_reads.items()},
            # the measurements of the spin split the sectors come from,
            # keyed by j(j + 1) as the sectors are
            "spin_split": {
                name: ({str(k): v for k, v in value.items()}
                       if isinstance(value, dict) else value)
                for name, value in doublet_sector_read(
                    doublet_content, trialities,
                    declared_tolerance(
                        config, "spin_sector_tolerance")).items()},
        })

    # the reads beside the poles: one without a value is flagged by name and
    # left out of the record, and the poles stand
    def beside(name, read):
        try:
            return read()
        except NO_VALUE_ERRORS as error:
            flags.append({"name": "%s has no value" % name,
                          "detail": str(error)})
            return None

    eigenbasis = beside("the eigenbasis of the carrier operator",
                        lambda: eigenbasis_marks(action.carrier_operator()))
    if eigenbasis and eigenbasis["defective"]:
        flags.append({
            "name": "the carrier operator is defective at the LU threshold",
            "detail": ("an eigenvector matrix of the carrier operator is "
                       "singular at the threshold of its LU decomposition "
                       "(smallest reciprocal condition number %.3g), and the "
                       "bands are read with its inverse as computed"
                       % eigenbasis["reciprocal_condition"]),
            "reciprocal_condition": eigenbasis["reciprocal_condition"],
        })
    recursion = beside("the level-recursion read",
                       lambda: recursion_read(spacetime, config))
    anchor = beside("the anchor atlas", lambda: anchor_atlas_read(
        spacetime, alignments, certificate_tolerance))
    fingerprint = beside("the spectral fingerprint",
                         lambda: spectral_fingerprint_read(
                             spacetime, kappa, beta, config))
    quark = None
    if recursion is None or anchor is None or fingerprint is None:
        flags.append({
            "name": "the quark conditions have no value",
            "detail": "they are evaluated on the level-recursion read, the "
                      "anchor atlas and the spectral fingerprint, and one "
                      "of the three has no value"})
    else:
        quark = beside("the quark conditions", lambda: quark_conditions(
            spacetime, alignments, recursion, averaged_residual, report,
            anchor, fingerprint, certificate_tolerance,
            fibre_lift_tolerance=declared_tolerance(config,
                                                    "fibre_lift_tolerance"),
            attachment_rank_tolerance=declared_tolerance(
                config, "attachment_rank_tolerance"),
            covariance=matrix(action.declaration.covariance),
            frame_name=spin_read_frame["name"],
            degeneracy_tolerance=declared_tolerance(
                config, "degeneracy_tolerance")))
    stage[0] = "the end-point measurements of the solve"
    truncation_read = action.holonomy_truncation()
    record = {
        "content": list(content),
        "content_meaning": CONTENT_MEANING,
        "elimination": config["elimination"],
        "seconds": time.time() - started,
        "flags": flags,
        "relaxation": dict(solve, **{
            "band_operator": "h_1 (the covariant operator itself)",
            "shared_sheet_geometry": True,
            "kappa_role": config.get("kappa_role"),
            "terms": {name: _term(action, name)
                      for name in ("regge", "holonomy", "matter",
                                   "spectral")},
            "regge_hinge_count": int(action.regge_hinge_count()),
            "edge_lengths": [complex(e.getLength()) for e in
                             spacetime.getEdgeList().toVector()],
            "face_holonomies": [complex(f) for f in
                                action.face_holonomies()],
            "holonomy_truncation": {
                "order": int(truncation_read.order),
                "relative_value_tail": float(
                    truncation_read.relative_value_tail),
                "relative_first_tail": float(
                    truncation_read.relative_first_tail),
                "relative_second_tail": float(
                    truncation_read.relative_second_tail),
            },
            "unit_circle_departure": max(dep for _, dep in supports),
            "symmetry_departure": symmetry["compensation_residual"],
            "length_departure": symmetry["length_departure"],
            "symmetry": symmetry,
            "spin_frame": spin_read_frame["name"],
            "frames": [{
                "trialities": a["trialities"],
                "reference_carrier": a["reference_carrier"],
                "reference_certified": a.get("reference_certified"),
                "half_turn_trace_residual": a.get(
                    "half_turn_trace_residual"),
                "averaged_eigenvalues": a["averaged_eigenvalues"],
                "intertwining_residual": a["intertwining_residual"],
            } for a in alignments],
            "monopole_numbers": [
                int(sp.monopoleNumber(certificate_tolerance).monopole_number)
                for sp, _ in supports],
            "link_force_norm": float(np.linalg.norm(
                action.link_stationarity())),
            "ward_current_divergence": float(np.max(np.abs(np.asarray(
                action.ward_current_divergence())))),
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
            "eliminated_dimension": quartic["eliminated_dimension"],
            "stiffness_asymmetry": quartic["stiffness_asymmetry"],
            "stiffness_conditioning": quartic["stiffness_conditioning"],
            "frame_pairing_defect": quartic["frame_pairing_defect"],
            "certificate": quartic["certificate"],
            "truncation": truncation,
            "fluctuations": fluctuations["record"],
        },
        "doublet_reads": doublet_reads,
    }
    for key, read in (("recursion", recursion), ("anchor", anchor),
                      ("spectral_fingerprint", fingerprint),
                      ("quark_conditions", quark),
                      ("carrier_eigenbasis", eigenbasis)):
        if read is not None:
            record[key] = read
    if config.get("isospin_doublet"):
        # The isospin-doublet observation (WP §10) on h_1 and its T-average,
        # added only when requested so that the default record is unchanged.
        from tessera.drivers import isospin_doublet
        spinorial = all(bool(a["spin_read"].cocycle.nontrivial)
                        for a in alignments)
        doublet = beside("the isospin doublet",
                         lambda: isospin_doublet.observe_host(
                             carrier, actions, spinorial, config))
        if doublet is not None:
            record["isospin_doublet"] = doublet
    return record


def recursion_read(spacetime, config):
    """One turn of the level recursion on h_1 of the relaxed host (WP §15),
    keeping every mode of each component as its fibre.

    The recursion is built over the microscopic pencil and `advance` takes
    one turn of the box, which is `level(0)`, the first completed turn (the
    microscopic level itself is `response_pencil(0)`, never a completed
    turn). Building it or taking the turn refuses by name at or above the
    declared dense crossover, or when a level is reduced to nothing; a
    refusal is recorded here as it is, with the number of completed turns
    and no band, so that quark condition 1 reads "not evaluable"
    (`quark_conditions`) instead of failing the content."""
    declaration = cob.LevelRecursionDeclaration()
    bands = cob.RecursionBandDeclaration()
    bands.selection = cob.RecursionBandSelection.LowestModes
    bands.band_rank = BASE_EDGES
    declaration.bands = bands
    declaration.tolerance = declared_tolerance(config, "recursion_tolerance")
    declaration.rank_tolerance = declared_tolerance(
        config, "quotient_rank_tolerance")
    recursion = None
    try:
        recursion = cob.LevelRecursion.overSpacetime(
            spacetime, 1, cob.HodgeMetricSource.WhitneyPencil, declaration)
        recursion.advance()
    except ValueError as refusal:
        return {"levels": int(recursion.level_count())
                if recursion is not None else 0,
                "refusal": str(refusal)}
    level = recursion.level(0)
    return {
        "levels": int(recursion.level_count()),
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
        # the size the leakage is measured against, in its own unit
        "fiber_norms": [float(np.linalg.norm(np.asarray(t.block)))
                        for t in level.transports
                        if t.from_component == t.to_component],
        "fock_stage_dimension": float(level.fock_stage_dimension),
        "determinant_residual": float(level.determinant_residual),
        "certificate": level.certificate.describe(),
        "reduction_certificate": level.reduction_certificate.describe(),
    }


#: The base vertex of the anchor atlas's path rule: breadth-first walks from
#: the cell's lowest vertex, inside the cell.
ANCHOR_BASE_POINT = 0


def anchor_atlas_read(spacetime, alignments,
                      tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The anchor atlas of the quark fiber's base band on the relaxed host,
    quark condition 3 (WP v18 §10): the dressed anchor
    (`chainhodge.DressedAnchor`) of the reference doublet of each sheet's
    aligned frame (``alignments``, one per sheet), the j = 1/2 doublet of the
    odd-monopole support (rank 2), on that sheet's relaxed connection over
    the atlas of the cell's four faces, with the breadth-first path rule from
    `ANCHOR_BASE_POINT`.

    Per sheet: the Lambda^2 coordinates of every face (the three maximal
    minors of the restricted frame), the number of anchoring faces, the
    connection-dressed covariance residual against the verification gauge,
    the determinant-line transition cocycle residual, the named refusals,
    and the invariant coordinates alpha_tau of the faces from the sheet's
    Whitney chain Hodge operator and the band's geometric images
    (`withInvariantCoordinates`), or why they were not attached, and the
    Proposition 3 certificate of that operator (whether it holds, and each
    property that does not, with its residual). The summary
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
    sheets = []
    for t in range(SHEETS):
        reference = int(alignments[t]["reference_carrier"])
        frame = np.asarray(alignments[t]["frame"])[:, 2 * reference:
                                                   2 * reference + 2]
        phi = frame[order, :]
        links = [complex(u) for u in sheet_links(spacetime, t)]
        squared = [complex(z) for z in sheet_squared_lengths(spacetime, t)]
        connection = ch.Connection(complex_, [links[k] for k in order])
        read = ch.DressedAnchor.profile(complex_, connection, paths, faces,
                                        phi, tolerance)
        unattached = None
        operator_certificate = None
        try:
            base = ch.ChainHodge(complex_, [squared[k] for k in order],
                                 ch.Preset.L2)
            covariant = ch.CovariantChainHodge(base, connection)
            # the operator is built whatever the residuals of Proposition 3
            # are; the coordinates are attached and the certificate says
            # which property, if any, does not hold on this sheet
            certificate = covariant.certificate()
            operator_certificate = {
                "holds": bool(certificate.holds),
                "failed": [str(entry) for entry in certificate.failed],
                "tolerance": float(certificate.tolerance),
                "condition_estimate": float(certificate.conditionEstimate)}
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
            "operator_certificate": operator_certificate,
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
        "operator_certificates_hold": all(
            s["operator_certificate"] is not None
            and s["operator_certificate"]["holds"] for s in sheets),
        "operator_certificates_failed": sorted(
            {entry for s in sheets
             for entry in (s["operator_certificate"] or {}).get("failed",
                                                                [])}),
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


def refined_support(sheet_data, tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The `MonopoleSupport` of one refined sheet: five vertices, the ten
    edges of `REFINED_EDGES`, the four boundary faces of the fixture in their
    outward orientation (the bounding cut, which the subdivision leaves as it
    was), and the U(1) part of the sheet's links, whose moduli the support
    holds to one within ``tolerance``."""
    fixture = monopole_support()
    return obs.MonopoleSupport(
        5, [list(e) for e in REFINED_EDGES], [list(f) for f in fixture.faces],
        obs.MonopoleSupport.u1Part([complex(u) for u in sheet_data["links"]]),
        tolerance)


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


def multiset_distance(first, second):
    """The distance of two spectra as multisets: the smallest, over the
    one-to-one pairings of their entries, of the largest separation of a
    pair. It does not depend on the order either spectrum is listed in: two
    eigenvalues whose real parts agree to rounding are paired with their own
    images whichever way a sort by real part lists them."""
    a = np.asarray(first, dtype=complex)
    b = np.asarray(second, dtype=complex)
    if len(a) != len(b):
        raise ValueError("the spectra have %d and %d eigenvalues and are not "
                         "one multiset" % (len(a), len(b)))
    if len(a) == 0:
        return 0.0
    distances = np.abs(a[:, None] - b[None, :])
    thresholds = np.unique(distances)
    # the smallest threshold at which the pairs within it contain a
    # one-to-one pairing
    low, high = 0, len(thresholds) - 1
    while low < high:
        middle = (low + high) // 2
        beyond = (distances > thresholds[middle]).astype(float)
        rows, columns = linear_sum_assignment(beyond)
        if beyond[rows, columns].sum() == 0.0:
            high = middle
        else:
            low = middle + 1
    return float(thresholds[low])


def _relative_shift(distance, scale):
    """A distance relative to a scale of its own unit; zero for a zero
    distance whatever the scale, and infinite for a nonzero distance on a
    zero scale."""
    if distance == 0.0:
        return 0.0
    return float(distance / scale) if scale > 0.0 else math.inf


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
        host, kappa, beta, config["regge_hinges"],
        villain_order=declared_villain_order(config))
    return matrix(cob.JointAction(host, declaration).carrier_operator())


def _fiber_fingerprint(host, support, group, edges, kappa, beta, config):
    """The label-free fingerprint of sheet 0 of a host: the spectrum of h_1,
    the spectrum of its T-average, and the reference doublet (the coexact
    rank-two spinorial band of the averaged combinatorial Laplacian, the
    band the aligned frame's reference carrier is: `symmetry_bands`,
    `reference_doublet`) with its energy on the averaged h_1 and its weight
    on every edge. The doublet's frame is the frame of the band that carries
    the certificates, so the band found and the band read are one."""
    degeneracy_tolerance = declared_tolerance(config, "degeneracy_tolerance")
    certificate_tolerance = declared_tolerance(config,
                                               "certificate_tolerance")
    read = support.spinRead(group, degeneracy_tolerance,
                            certificate_tolerance)
    carrier = _carrier(host, kappa, beta, config)
    block = carrier[:edges, :edges]
    actions = [np.asarray(support.edgeRepresentation(g)) for g in group]
    averaged = rotation_averaged(block, actions)
    bands = symmetry_bands(support, group, bool(read.cocycle.nontrivial),
                           degeneracy_tolerance, certificate_tolerance)
    band = reference_doublet(bands)
    doublet = None if band is None else band["frame"]
    out = {
        "spectrum": sorted((complex(v) for v in np.linalg.eigvals(block)),
                           key=lambda v: (v.real, v.imag)),
        "averaged_spectrum": sorted(
            (complex(v) for v in np.linalg.eigvals(averaged)),
            key=lambda v: (v.real, v.imag)),
        "monopole_number": int(read.monopole.monopole_number),
        "doublet_found": doublet is not None,
        "band_certificates": [
            {key: value for key, value in b.items() if key != "frame"}
            for b in bands],
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
      continuation. The isotypic component is the image of the projector:
      its dimension is the projector's trace, an integer, and its basis the
      eigenvectors whose eigenvalues are nearest one, as many as that
      dimension, with the trace's departure from the integer and the
      projector's idempotency defect reported. The averaged operator
      commutes with the action on that component, which is copies of one
      rank-two type, so its eigenvalues there are equal in pairs exactly: a
      pair is one unit, and consecutive units within the config's
      ``isotypic_tolerance`` form one band. The overlap must reach
      ``overlap_floor`` with the rank preserved, and the refined support
      must carry a spinor doublet; the energy shift is reported beside
      them."""
    group = rotation_group()
    # the tolerance the bands' certificates are graded at
    band_tolerance = declared_tolerance(config, "certificate_tolerance")
    if tolerance is None:
        tolerance = band_tolerance
    support, _ = sheet_support(spacetime, 0, tolerance)
    original = _fiber_fingerprint(spacetime, support, group, BASE_EDGES,
                                  kappa, beta, config)
    result = {"sheet": 0, "spectrum": original["spectrum"],
              "doublet_found": original["doublet_found"],
              "band_certificates": original["band_certificates"],
              "tolerance": tolerance}
    if not original["doublet_found"]:
        # the fingerprint is the reference doublet's: with none on the host
        # there is nothing to re-read, and the two items are not evaluable
        unread = {"held": None,
                  "unread": "no spinor doublet on the host: %s"
                  % nearest_doublet_text(original["band_certificates"],
                                         band_tolerance)}
        result["relabeling"] = dict(unread)
        result["refinement"] = dict(unread)
        return result
    result["doublet_energy"] = original["doublet_energy"]
    result["doublet_weights"] = original["doublet_weights"]
    # --- relabeling
    permutation = list(FINGERPRINT_RELABELING)
    moved_host = relabeled_host(spacetime, permutation)
    moved_support, _ = sheet_support(moved_host, 0, tolerance)
    moved = _fiber_fingerprint(moved_host, moved_support,
                               conjugated_group(group, permutation),
                               BASE_EDGES, kappa, beta, config)
    # every shift is relative to the size of h_1's own spectrum, so the
    # verdict does not move with the unit the operator is expressed in
    scale = max(abs(v) for v in original["spectrum"])
    spectrum_shift = _relative_shift(multiset_distance(
        original["spectrum"], moved["spectrum"]), scale)
    averaged_shift = _relative_shift(multiset_distance(
        original["averaged_spectrum"], moved["averaged_spectrum"]), scale)
    relabeling = {"permutation": permutation,
                  "spectrum_shift": float(spectrum_shift),
                  "averaged_spectrum_shift": float(averaged_shift),
                  "doublet_found": moved["doublet_found"],
                  "band_certificates": moved["band_certificates"],
                  "monopole_number": moved["monopole_number"]}
    if moved["doublet_found"]:
        weight_shift = 0.0
        for m, (a, b) in enumerate(PAIRS):
            image = (permutation[a], permutation[b])
            n = PAIRS.index((min(image), max(image)))
            weight_shift = max(weight_shift, abs(
                original["doublet_weights"][m] - moved["doublet_weights"][n]))
        energy_shift = _relative_shift(
            abs(original["doublet_energy"] - moved["doublet_energy"]), scale)
        relabeling.update({"doublet_weight_shift": float(weight_shift),
                           "doublet_energy_shift": float(energy_shift),
                           "held": bool(spectrum_shift <= tolerance
                                        and averaged_shift <= tolerance
                                        and weight_shift <= tolerance
                                        and energy_shift <= tolerance)})
    else:
        relabeling["held"] = False
        relabeling["unread"] = (
            "no spinor doublet on the relabeled host: %s"
            % nearest_doublet_text(moved["band_certificates"],
                                   band_tolerance))
    result["relabeling"] = relabeling
    # --- refinement
    refined, sheet_data = refined_host(spacetime)
    support_r = refined_support(sheet_data[0], tolerance)
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
                  "band_certificates": fine["band_certificates"],
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
        isotypic_tolerance = declared_tolerance(config, "isotypic_tolerance")
        # the image of a projector: as many eigenvectors as its trace, those
        # whose eigenvalues are nearest one
        trace = complex(np.trace(isotypic))
        dimension = max(0, min(len(values), int(round(trace.real))))
        nearest = sorted(range(len(values)),
                         key=lambda k: abs(values[k] - 1.0))[:dimension]
        span = vectors[:, nearest]
        refinement.update({
            "isotypic_trace_departure": float(abs(trace - dimension)),
            "isotypic_idempotency": float(np.max(np.abs(
                isotypic @ isotypic - isotypic))),
        })
        # the pseudo-inverse of the span counts a singular value at or below
        # the declared tolerance times the largest as zero
        compressed = (np.linalg.pinv(span, rcond=isotypic_tolerance)
                      @ fine["averaged"] @ span)
        energies, mixing = np.linalg.eig(compressed)
        candidates = span @ mixing
        shared = [k for k, e in enumerate(REFINED_EDGES)
                  if REFINED_CENTRE not in e]
        original_frame = np.zeros((edges, 2), dtype=complex)
        for m, k in enumerate(shared):
            original_frame[k, :] = original["doublet_frame"][m, :]
        # the units of the component: its eigenvalues in ascending order,
        # two at a time (one copy of the rank-two type each), and the bands:
        # runs of units each within the tolerance of the run's first
        ascending = sorted(range(len(energies)),
                           key=lambda k: (energies[k].real, energies[k].imag))
        unit = 2 if len(ascending) % 2 == 0 else 1
        # the tolerance is relative to the size of the component's energies
        energy_scale = float(np.max(np.abs(energies))) if len(energies) \
            else 0.0
        runs, start = [], 0
        while start < len(ascending):
            stop = start + unit
            while (stop < len(ascending)
                   and abs(energies[ascending[stop]]
                           - energies[ascending[start]])
                   <= isotypic_tolerance * energy_scale):
                stop += unit
            runs.append(ascending[start:stop])
            start = stop
        distinct = [complex(np.mean(energies[run])) for run in runs]
        best = None
        for energy, run in zip(distinct, runs):
            picked = candidates[:, run]
            overlap = _subspace_overlap(original_frame[shared, :],
                                        picked[shared, :])
            if best is None or overlap > best["overlap"]:
                best = {"overlap": overlap, "energy": energy,
                        "rank": int(picked.shape[1])}
        refinement.update({
            "isotypic_dimension": int(span.shape[1]),
            "refined_bands": distinct,
        })
        if best is None:
            # the doublet's type does not occur in the refined action: there
            # is no band to continue the doublet to
            refinement.update({
                "held": False,
                "unread": "the isotypic component of the doublet's type on "
                          "the refined cell is empty (the projector's trace "
                          "is %.3g)" % trace.real})
        else:
            refinement.update({
                "overlap": best["overlap"],
                "refined_energy": best["energy"],
                "refined_rank": best["rank"],
                "energy_shift": _relative_shift(
                    abs(best["energy"] - original["doublet_energy"]), scale),
                "held": bool(best["overlap"] >= overlap_floor
                             and best["rank"] == 2)})
    else:
        refinement["held"] = False
        refinement["unread"] = (
            "no spinor doublet on the refined support: %s"
            % nearest_doublet_text(fine["band_certificates"],
                                   band_tolerance))
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
    # a read that was not made (``held`` None: no spinor doublet on the host
    # to take the fingerprint of) is not evaluable, with the reason
    return [E("refinement-stability",
              None if r["held"] is None else bool(r["held"]), refinement),
            E("relabeling-stability",
              None if l["held"] is None else bool(l["held"]), relabeling)]


def fingerprint_text(fingerprint):
    """One content's spectral fingerprint reads as text."""
    if not fingerprint:
        return "spectral fingerprint unread"
    r = fingerprint["refinement"]
    l = fingerprint["relabeling"]
    def verdict(read):
        return ("unread" if read["held"] is None
                else "stable" if read["held"] else "unstable")

    relabeling = verdict(l)
    if "doublet_weight_shift" in l:
        relabeling += " (spectrum shift %.2g, doublet weight shift %.2g)" % (
            l["spectrum_shift"], l["doublet_weight_shift"])
    elif l.get("unread"):
        relabeling += " (%s)" % l["unread"]
    refinement = verdict(r)
    if "overlap" in r:
        refinement += " (overlap %.4f, energy shift %.2g)" % (
            r["overlap"], r["energy_shift"])
    elif r.get("unread"):
        refinement += " (%s)" % r["unread"]
    else:
        refinement += " (no spinor doublet on the refined support)"
    return "spectral fingerprint: relabeling %s; refinement %s" % (
        relabeling, refinement)


def sheets_of(spacetime):
    """The sheets of a complex as it stands, in the literal case of the sheet
    convention (WP v18 §8, "Sheet convention (adopted)": k isomorphic copies
    of a base complex, disjoint, the sheet number a superselection datum of
    the complex): its connected components.

    ``components`` lists the connected components, each as the ascending ids
    of its vertices, in ascending order of their smallest vertex, and
    ``count`` is their number. ``cells`` gives, per component, the number of
    its cells of every degree from zero to the dimension. ``copies`` says
    whether every component is a copy of the first under the correspondence
    that sends the j-th smallest vertex of one to the j-th smallest vertex of
    the other, the correspondence `build_host` labels the sheets by: the
    cells of every degree are then the same sets of positions. The squared
    lengths and the connection on corresponding edges are the separate read
    of `SheetedSupport.certifyIsomorphism`."""
    complex_ = cob.ChainComplex.fromSpacetime(spacetime)
    dimension = int(complex_.dimension())
    vertices = sorted(int(cell[0]) for cell in complex_.kSimplexVertices(0))
    parent = {v: v for v in vertices}

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in complex_.kSimplexVertices(1):
        low, high = sorted((root(int(a)), root(int(b))))
        parent[high] = low
    members = {}
    for v in vertices:
        members.setdefault(root(v), []).append(v)
    components = [members[r] for r in sorted(members)]
    sheet = {v: t for t, component in enumerate(components)
             for v in component}
    place = {v: j for component in components
             for j, v in enumerate(component)}
    shapes = [[set() for _ in range(dimension + 1)] for _ in components]
    for degree in range(dimension + 1):
        for cell in complex_.kSimplexVertices(degree):
            cell = [int(v) for v in cell]
            shapes[sheet[cell[0]]][degree].add(
                tuple(sorted(place[v] for v in cell)))
    return {"count": len(components),
            "components": components,
            "cells": [[len(cells) for cells in shape] for shape in shapes],
            "copies": all(shape == shapes[0] for shape in shapes)}


def sheet_occupations(spacetime, covariance, sheets, degree=1):
    """n_t, the occupation every sheet of ``sheets`` (`sheets_of`) carries:
    the trace of the covariance Gamma over the carrier's cells on that sheet
    (WP v18 §7: tr Gamma = N, the number of occupied modes). The carrier's
    cells are the degree-``degree`` cells of the complex in the chain
    complex's order, which is the order of the modes of h_degree and so of
    Gamma. Raises `ValueError` when the covariance is not a matrix over those
    cells."""
    cells = cob.ChainComplex.fromSpacetime(spacetime).kSimplexVertices(degree)
    gamma = np.asarray(covariance, dtype=complex)
    if gamma.shape != (len(cells), len(cells)):
        raise ValueError(
            "the covariance has shape %s and the carrier has %d cells of "
            "degree %d" % (gamma.shape, len(cells), degree))
    sheet = {v: t for t, component in enumerate(sheets["components"])
             for v in component}
    occupations = [0j] * sheets["count"]
    for index, cell in enumerate(cells):
        occupations[sheet[int(cell[0])]] += gamma[index, index]
    return occupations


def sheet_count_evidence(spacetime):
    """The evidence "three-sheeted-support" of quark condition 2 (WP v18 §10:
    the colour-spin fibre is on a three-sheeted support), measured on the
    complex (`sheets_of`): it holds when the complex has `SHEETS` connected
    components and they are copies of one base complex. The detail carries
    the measured number of components and the cells of each."""
    sheets = sheets_of(spacetime)
    held = sheets["count"] == SHEETS and sheets["copies"]
    degrees = len(sheets["cells"][0]) - 1
    if sheets["copies"]:
        detail = ("sheet number %d: the connected components of the complex, "
                  "copies of one base complex of %s cells of degrees 0 to %d "
                  "under the ascending correspondence of their vertices"
                  % (sheets["count"], sheets["cells"][0], degrees))
    else:
        detail = ("%d connected components that are not copies of one base "
                  "complex under the ascending correspondence of their "
                  "vertices: cells of degrees 0 to %d per component %s"
                  % (sheets["count"], degrees, sheets["cells"]))
    return obs.QuarkConditionEvidence("three-sheeted-support", held, detail)


def occupation_parity_evidence(spacetime, covariance,
                               tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The evidence "odd-occupation-parity" of quark condition 4 (WP v18 §10:
    a quark is a single occupied mode of its fibre, so its occupation parity
    is odd; §12: a baryon carried by one three-sheeted cluster has one
    occupied mode per sheet), measured on the solved state: the occupation
    of every sheet is the trace of the covariance on that sheet's modes
    (`sheet_occupations`), and its parity is that of the integer it is
    within ``tolerance`` of. The evidence holds when every sheet's
    occupation is an odd integer. The detail carries the occupations, their
    sum tr Gamma and, for a sheet whose occupation is not an integer at the
    tolerance, its distance from the nearest one. Without a covariance, or
    with one that is not over the carrier's cells, the evidence is not
    evaluable and says why."""
    name = "odd-occupation-parity"
    E = obs.QuarkConditionEvidence
    if covariance is None:
        return E(name, None, "the covariance of the solved state was not "
                             "handed to the read")
    sheets = sheets_of(spacetime)
    try:
        occupations = sheet_occupations(spacetime, covariance, sheets)
    except ValueError as error:
        return E(name, None, str(error))

    def number(n):
        return ("%.16g" % n.real if n.imag == 0.0
                else "%.16g%+.3gi" % (n.real, n.imag))

    held, parities = True, []
    for n in occupations:
        if not (math.isfinite(n.real) and math.isfinite(n.imag)):
            held = False
            parities.append("%s is not a number" % number(n))
            continue
        nearest = int(round(n.real))
        distance = abs(n - nearest)
        if distance > tolerance:
            held = False
            parities.append("%s is %.3g from the integer %d, above the "
                            "tolerance %.3g"
                            % (number(n), distance, nearest, tolerance))
            continue
        held = held and nearest % 2 == 1
        parities.append("(-1)^%d" % nearest)
    detail = ("occupation per sheet %s (the trace of the covariance on each "
              "sheet's modes), tr Gamma = %s; parity per sheet %s"
              % ([number(n) for n in occupations],
                 number(sum(occupations, 0j)), parities))
    return E(name, held, detail)


def cell_base_bands(supports, degeneracy_tolerance=DECLARED_TOLERANCE,
                    tolerance=DECLARED_CERTIFICATE_TOLERANCE):
    """The base band of every sheet of a cell, read on the sheet's own
    support (``supports``, the (`MonopoleSupport`, departure) pairs of
    `sheet_support`): the projective class of the support's spin read
    (``nontrivial``), its symmetry-protected bands (`symmetry_bands`) and
    the band that is its reference doublet (`reference_doublet`), or, when
    no band is a certified coexact spinor doublet, the band of least coexact
    residual with ``certified`` false. This is the cell's own read, whatever
    frame the spin of the cell is read in (`spin_frame`)."""
    group = rotation_group()
    out = []
    for support, _ in supports:
        read = support.spinRead(group, degeneracy_tolerance, tolerance)
        nontrivial = bool(read.cocycle.nontrivial)
        bands = symmetry_bands(support, group, nontrivial,
                               degeneracy_tolerance, tolerance)
        doublet = reference_doublet(bands)
        band = doublet if doublet is not None else min(
            bands, key=lambda b: b["coexact_residual"])
        out.append({
            "nontrivial": nontrivial,
            "commutator_phase": complex(read.cocycle.commutator_phase),
            "certified": doublet is not None,
            "band": {key: value for key, value in band.items()
                     if key != "frame"},
        })
    return out


#: Why the evidence "color-transport-full-rank" of quark condition 5 is not
#: evaluated on the read of one cell.
COLOUR_TRANSPORT_UNREAD = (
    "the colour transport S_AB is the attachment pattern of the simplices "
    "that connect the sheets of two clusters A and B (WP v18 §8, §9), and "
    "the read of one cell has one cluster and no connecting simplex; the "
    "sheet-to-sheet attachment of the default rule, for which S is the "
    "identity, is a declared rule and not a measurement")


def quark_conditions(spacetime, alignments, recursion, symmetry_residual,
                     report, anchor=None, fingerprint=None,
                     tolerance=DECLARED_CERTIFICATE_TOLERANCE,
                     fibre_lift_tolerance=DECLARED_TOLERANCE,
                     attachment_rank_tolerance=DECLARED_TOLERANCE,
                     covariance=None, frame_name=None,
                     degeneracy_tolerance=DECLARED_TOLERANCE):
    """The seven v16 quark conditions by name (`QuarkConditions`), from what a
    single-level synthesis measures: condition 3 from the anchor atlas read
    (`anchor_evidence`) and condition 7 from the spectral fingerprint read
    (`fingerprint_evidence`). The protected base band and its sector are
    read on the cell: the reference doublet of every sheet's own support
    with its certificates (`cell_base_bands`), beside the support's monopole
    number and the projective class of its spin read; every sheet must
    carry them. The reference doublets of the aligned frames the spin is
    read in (``alignments``, one per sheet, `aligned_doublet_frame`) are
    named beside them in the evidence's detail with the frame they belong
    to (``frame_name``, `spin_frame`): when the frame is the declared
    symmetric host's they are the host's and not the cell's. The colour
    transport of condition 5 is not evaluated on one cell
    (`COLOUR_TRANSPORT_UNREAD`). Anything that needs several
    cobordism frames is left unmeasured and so reads "not evaluable", and so
    does every piece of evidence read from the recursion's completed turn
    when the recursion refused to take one (`recursion_read`). The number of
    sheets of condition 2 is measured on the complex (`sheet_count_evidence`)
    and the occupation parity of condition 4 on ``covariance``, the
    covariance of the solved state as a matrix over the carrier's cells
    (`occupation_parity_evidence`); without it that evidence is not
    evaluable. The
    certificates are graded at ``tolerance``; the fibre lift holds when
    ``symmetry_residual`` is at or below ``fibre_lift_tolerance``, and the
    sheet-to-sheet attachment of the declared default rule, which is
    reported in the detail of the colour transport's evidence, has full rank
    when its smallest singular value reaches ``attachment_rank_tolerance``.
    The bands of the cell's own supports are grouped at
    ``degeneracy_tolerance``. The leakage between components holds when
    every inter-component transport norm is at or below ``tolerance`` times
    the largest norm of a component's own fiber operator (the read's
    ``fiber_norms``), so it does not move with the unit of the operator."""
    E = obs.QuarkConditionEvidence
    supports = [sheet_support(spacetime, t, tolerance) for t in range(SHEETS)]
    monopoles = [s.monopoleNumber(tolerance) for s, _ in supports]
    own = cell_base_bands(supports, degeneracy_tolerance, tolerance)
    frame_doublets = [a["reference_doublet"] for a in alignments]
    frame_text = (
        "; the aligned frame the spin is read in (%s) has reference "
        "doublets with spinor doublet %s, coexact residuals %s"
        % (frame_name or "unnamed",
           [bool(d["spinor_doublet"]) for d in frame_doublets],
           [d["coexact_residual"] for d in frame_doublets]))
    sheeting = obs.SheetedSupport(SHEETS, BASE_EDGES)
    isomorphism = sheeting.certifyIsomorphism(
        [np.array(sheet_squared_lengths(spacetime, t)) for t in range(SHEETS)],
        [np.array(sheet_links(spacetime, t)) for t in range(SHEETS)],
        tolerance)
    # the declared default rule attaches sheet to sheet; its attachment is
    # reported beside the evidence, which it does not decide
    declared_rule = obs.SheetAttachment.attachmentMatrix(
        SHEETS, [obs.ConnectingSimplex(t, t, 1.0) for t in range(SHEETS)],
        attachment_rank_tolerance)
    produced = "refusal" not in recursion

    def from_level(name, held, detail):
        """Evidence read from the recursion's completed turn, or not
        evaluable, with the recursion's refusal, when it took none."""
        if not produced:
            return E(name, None, "the recursion took no turn: %s"
                     % recursion["refusal"])
        return E(name, held(), detail())

    accepted = all(recursion["bands_accepted"]) if produced else None

    def leakage():
        """The inter-component transport norms relative to the largest norm
        of a component's own fiber operator (``fiber_norms``), which is in
        their unit; a read that carries no fiber norm gives them as they
        are."""
        fibers = recursion.get("fiber_norms") or []
        scale = max(fibers) if fibers else 1.0
        return [_relative_shift(n, scale)
                for n in recursion["transport_norms"]]

    def leakage_text():
        fibers = recursion.get("fiber_norms") or []
        return ("inter-component transport norms %s%s"
                % (recursion["transport_norms"],
                   ", %s of the largest fiber operator norm %.3g"
                   % (leakage(), max(fibers)) if fibers else ""))
    evidence = [
        [from_level("persistent-support", lambda: accepted,
                    lambda: "partition %s" % recursion["partition"]),
         from_level("localized-projector-rank", lambda: accepted,
                    lambda: "ranks %s, idempotency %s"
                    % (recursion["band_ranks"],
                       recursion["projector_idempotency"])),
         from_level("contour-separation",
                    lambda: all(g > 0 for g in recursion["isolation_gaps"]),
                    lambda: "isolation gaps %s" % recursion["isolation_gaps"]),
         E("successor-overlap", None, "a single level has no successor"),
         E("multi-frame-lifetime", None,
           "a single level spans one cobordism frame"),
         from_level("external-leakage",
                    lambda: all(n <= tolerance for n in leakage()),
                    leakage_text)],
        [sheet_count_evidence(spacetime),
         E("sheet-isomorphism", bool(isomorphism.isomorphic),
           "length residual %.3g, connection residual %.3g"
           % (isomorphism.squared_length_residual,
              isomorphism.connection_residual)),
         E("protected-base-band",
           bool(all(o["band"]["spinor_doublet"] for o in own)
                and all(m.odd for m in monopoles)
                and all(o["nontrivial"] for o in own)),
           "on the cell's own supports: monopole numbers %s, cocycle "
           "nontrivial %s, commutator phases %s, base band a spinor doublet "
           "%s (rank %s, invariance residuals %s, irreducibility scores %s)"
           "%s"
           % ([m.monopole_number for m in monopoles],
              [o["nontrivial"] for o in own],
              [o["commutator_phase"] for o in own],
              [o["band"]["spinor_doublet"] for o in own],
              [o["band"]["dimension"] for o in own],
              [o["band"]["invariance_residual"] for o in own],
              [o["band"]["irreducibility_score"] for o in own],
              frame_text)),
         E("base-band-sector",
           bool(all(o["certified"] and o["band"]["coexact"] for o in own)),
           "on the cell's own supports: a certified coexact spinor doublet "
           "%s, coexact residuals %s of the band nearest the coexact sector"
           "%s"
           % ([o["certified"] for o in own],
              [o["band"]["coexact_residual"] for o in own], frame_text)),
         E("fibre-lift", symmetry_residual <= fibre_lift_tolerance,
           "the T-averaged h_1 (WP line 497) in the aligned frame is "
           "block-scalar on each doublet times the sheets to relative "
           "residual %.3g" % symmetry_residual)],
        anchor_evidence(anchor),
        [occupation_parity_evidence(spacetime, covariance, tolerance)],
        [E("color-transport-full-rank", None,
           "%s (det S = %s, full rank at the tolerance %.3g: %s)"
           % (COLOUR_TRANSPORT_UNREAD, complex(declared_rule.determinant),
              attachment_rank_tolerance,
              bool(declared_rule.certificate.holds()))),
         from_level("base-transport-leakage",
                    lambda: all(n <= tolerance for n in leakage()),
                    leakage_text),
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


def scan_point(kappa, beta, config, on_content=None):
    """Every content at one (kappa, beta), and the ratios.

    A content whose read is flagged is a content like any other: its record
    carries its poles and lists its flags (``flags``), and
    ``flagged_contents`` names it. A content with no value has a record with
    ``failed`` (the message) and no pole, and ``failed_contents`` names it:
    one whose solve left no geometry the poles are defined on, or whose
    solved cell has a read without a value that the poles are built on
    (`ReadWithoutValue`, with ``reason`` the reason by name and the solve's
    record), and one whose solve itself has no value (`NO_VALUE_ERRORS`: the
    library's message). Every other content of the point is read."""
    records = []
    for content in config.get("contents") or contents():
        try:
            record = evaluate_content(content, kappa, beta, config)
        except ReadWithoutValue as missing:
            # the mean-field solve left no finite geometry to read: recorded
            # with the reason by name and the solve's record, and the content
            # supplies no pole
            record = dict({"content": list(content),
                           "elimination": config["elimination"],
                           "failed": str(missing), "reason": missing.name,
                           "relaxation": missing.relaxation,
                           "doublet_reads": []}, **missing.records)
        except NO_VALUE_ERRORS as error:
            # the solve of this content has no value (a face holonomy outside
            # the domain of the holonomy term, a decomposition that did not
            # converge): recorded with the library's message, and the content
            # supplies no pole
            record = {"content": list(content),
                      "elimination": config["elimination"],
                      "failed": str(error), "doublet_reads": []}
        records.append(record)
        if on_content is not None:
            on_content(record)
    return {"kappa": kappa, "beta": beta,
            "elimination": config["elimination"],
            "failed_contents": [record["content"] for record in records
                                if "failed" in record],
            "flagged_contents": [record["content"] for record in records
                                 if record.get("flags")],
            "contents": records,
            # the tolerance every lowest pole of the point is picked at
            "tie_tolerance": declared_tolerance(config, "tie_tolerance"),
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
#: Two poles whose real parts agree to this tolerance, relative to the
#: larger of their moduli, are tied for a minimum, and every tied pair is
#: named beside the minimum.
DECLARED_TIE_TOLERANCE = DECLARED_TOLERANCE


def sector_rows(records):
    """Every (content, doublet content) pair of a list of content records,
    with the doublet content's sectors: the rows the pole table, the minima
    and the ratios are taken over."""
    for record in records:
        for read in record.get("doublet_reads", []):
            yield record["content"], read["doublet_content"], read["sectors"]


def _tied(a, b, tolerance=DECLARED_TIE_TOLERANCE):
    """Whether two poles tie in the ascending-real-part order: their real
    parts differ by at most ``tolerance`` times the larger of their moduli,
    so the verdict does not move with the unit the poles are expressed in.
    A pole that is not a number ties with nothing."""
    return abs(a.real - b.real) <= tolerance * max(abs(a), abs(b))


def lowest_of(candidates, tolerance=DECLARED_TIE_TOLERANCE):
    """The candidate whose ``pole`` has the smallest real part, the library's
    declared `OccupationOrder.AscendingRealPart` (the first such candidate in
    the order given), as a copy that lists under ``tied`` every other
    candidate whose pole's real part agrees with it to ``tolerance``, which
    it records as ``tie_tolerance``; None when there is no candidate."""
    candidates = list(candidates)
    best = None
    for candidate in candidates:
        # a pole whose real part is not a number is below nothing: it is the
        # lowest only when no candidate has a real part that is a number
        if best is None or candidate["pole"].real < best["pole"].real \
                or (math.isnan(best["pole"].real)
                    and not math.isnan(candidate["pole"].real)):
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
    """The ratios of a nucleon pole to a Delta pole: of the complex poles,
    of their moduli and of their real parts. A ratio whose denominator
    vanishes (a Delta pole at zero, or on the imaginary axis for the real
    parts) is not a number, and the pair is recorded all the same."""
    out = {
        "nucleon_pole": s_n, "delta_pole": s_d,
        "pole_ratio": _ratio(s_n, s_d),
        "modulus_ratio": _ratio(abs(s_n), abs(s_d)).real,
        "real_part_ratio": _ratio(s_n.real, s_d.real).real,
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

    Each pairing names, under ``nucleon_flags`` and ``delta_flags``, the
    flags of the content each pole is read from that bear on the column
    (`content_flags`): a ratio built from a flagged read says so beside its
    value.
    """
    flags_of = {}
    for record in records:
        if "content" not in record:
            continue
        for name in COLUMNS:
            flags_of[(tuple(record["content"]), name)] = [
                flag["name"] for flag in content_flags(record, name)]

    def flagged(name, n, d):
        return {"nucleon_flags": flags_of.get((tuple(n["content"]), name),
                                              []),
                "delta_flags": flags_of.get((tuple(d["content"]), name), [])}

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
                **flagged(name, n, d),
            })
        else:
            result["by_2T_reading"] = None
        by_spin = lowest_over_pairs(records, name, tolerance)
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
                **flagged(name, n, d),
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


def term_records(terms):
    """The terms of the action at one recorded point (`ActionTermRecord`) as
    plain records: name, the term's label as it stands in the action, its
    weight, the bare factor the weight multiplies, whether the value is
    their product, the value, and the norm of the term's stationarity
    gradient on the relaxed coordinates. The list ends with the sum of the
    constraint terms and the whole action."""
    return [{"name": str(term.name), "label": str(term.label),
             "weight": complex(term.weight), "bare": complex(term.bare),
             "factored": bool(term.factored), "value": complex(term.value),
             "gradient_norm": float(term.gradient_norm)} for term in terms]


def term_trace(relaxation):
    """The per-iterate terms of a solve record: the level relaxation's
    ``term_trace`` (the starting point first), or the mean-field iterates'
    ``terms``; empty when the solve recorded none."""
    if not relaxation:
        return []
    if relaxation.get("term_trace"):
        return relaxation["term_trace"]
    return [step.get("terms") or [] for step in relaxation.get("trace") or []]


def term_trace_lines(relaxation, prefix):
    """The trace of the action's terms through a solve (``--trace-terms``),
    one line per iterate: every term's value and the norm of its stationarity
    gradient on the relaxed coordinates, and, from the second iterate on, the
    change of the value and the improvement of the gradient norm (the
    previous norm less the current one: positive when the term's equations
    came closer to holding, negative when they moved away) since the
    previous iterate. Empty when the solve recorded no terms."""
    trace = [terms for terms in term_trace(relaxation) if terms]
    lines = []
    previous = None

    def changes(term, before):
        if before is None:
            return ""
        change = term["value"] - before["value"]
        return " [value %+.3g%+.3gi, improved %+.3g]" % (
            change.real, change.imag,
            before["gradient_norm"] - term["gradient_norm"])

    def line(term, before, indent, head):
        if term["factored"]:
            body = "%s = (%s) x (%s) = %s" % (
                head, _complex_text(term["weight"]),
                _complex_text(term["bare"]), _complex_text(term["value"]))
        elif term["name"] == "holonomy":
            body = "%s = %s (beta = %s)" % (head, _complex_text(term["value"]),
                                           _complex_text(term["weight"]))
        else:
            body = "%s = %s" % (head, _complex_text(term["value"]))
        return "%s%s%s; gradient %.3g%s" % (
            prefix, " " * indent, body, term["gradient_norm"],
            changes(term, before))

    for index, terms in enumerate(trace):
        by_name = {term["name"]: term for term in terms}
        earlier = {term["name"]: term for term in previous or []}
        total = by_name.get("action")
        if total is not None:
            lines.append("%siterate %d: S = %s; stationarity residual %.3g%s"
                         % (prefix, index, _complex_text(total["value"]),
                            total["gradient_norm"],
                            changes(total, earlier.get("action"))))
        else:
            lines.append("%siterate %d:" % (prefix, index))
        for name in ("regge", "holonomy", "matter"):
            if name in by_name:
                term = by_name[name]
                lines.append(line(term, earlier.get(name), 2, term["label"]))
        if "constraints" in by_name:
            term = by_name["constraints"]
            lines.append(line(term, earlier.get("constraints"), 2,
                              term["label"]))
        for term in terms:
            if term["name"].startswith("constraint "):
                lines.append(line(term, earlier.get(term["name"]), 4,
                                  term["label"]))
        previous = terms
    return lines


def relaxation_text(relaxation):
    """One content's mean-field solve as text: whether it converged
    (`solve_converged`) with its residual norm, its force norm and whether
    the end point is self-consistent at the mean-field tolerance, its number
    of accepted relaxation updates, and, where the record carries them, why
    it stopped (by name, with its detail), the joint Jacobian's rank and rank
    gap at the end point, the Kontsevich-Segal margin and the growth of the
    lengths there, and the occupied bands followed from the host with their
    places in the ascending real-part order, the iterates at which a band had
    crossed another and the lowest overlap of a band with its previous
    projector."""
    if not relaxation or "converged" not in relaxation:
        return "mean field unrecorded"
    measures = []
    if "residual" in relaxation:
        measures.append("residual norm %.3g" % relaxation["residual"])
    measures.append("force norm %.3g" % relaxation["force_norm"])
    if "self_consistent" in relaxation:
        measures.append("self-consistent %s" % relaxation["self_consistent"])
    text = "solve converged %s (%s)" % (relaxation["converged"],
                                        ", ".join(measures))
    if relaxation.get("accepted_updates") is not None:
        count = int(relaxation["accepted_updates"])
        text += " after %d accepted update%s" % (count,
                                                 "" if count == 1 else "s")
    if "stop_reason" not in relaxation:
        return text
    jacobian = relaxation.get("joint_jacobian") or {}
    text += ("; stopped: %s (%s); joint Jacobian rank %d of %d, "
             "rank gap %.3g; Kontsevich-Segal margin %.3g; largest |z| %.3g "
             "times the host's" % (
                 relaxation["stop_reason"],
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
            multipliers = "[%s]" % ", ".join(
                _complex_text(x) for x in relaxation["multipliers"])
            if relaxation.get("fiber_pinning", "power-sums") == "eigenvalues":
                text += ("; the eigenvalues of %d occupied bands (fiber rank "
                         "%d) pinned at the host: multipliers %s, largest "
                         "relative residual |lambda_b - lambda_b*| / "
                         "|lambda_b*| %.2g" % (
                             pinned, relaxation["fiber_rank"], multipliers,
                             max(relative)))
            else:
                text += ("; %d of the occupied fiber's %d power sums pinned "
                         "at the host: multipliers %s, largest relative "
                         "residual |p_j(h_C) - p_j*| / |p_j*| %.2g" % (
                             pinned, relaxation["fiber_rank"], multipliers,
                             max(relative)))
        else:
            text += "; no constraint of the occupied fiber pinned"
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
    each (`pair_line`); a content with no value is one line with the reason
    and, where the mean-field solve was made, the solve's record."""
    if "failed" in record:
        line = "%scontent %s: no value: %s" % (prefix, list(record["content"]),
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


def lowest_lines(records, prefix="", tolerance=DECLARED_TIE_TOLERANCE):
    """The labelled minima as text, after the per-pair lines, each line
    starting "lowest over": per content, the lowest pole (smallest real part)
    of each spin and column over its doublet contents, with the doublet
    content it came from; then the lowest over every (content, doublet
    content) pair, with the pair it came from. Ties are named at
    ``tolerance``, the tie tolerance the records' ratios were read at."""
    lines = []
    for record in records:
        if "failed" in record:
            continue
        per_column = {name: lowest_poles(record, name, tolerance)
                      for name in COLUMNS}
        lines.append("%slowest over the doublet contents of content %s: %s"
                     % (prefix, list(record["content"]), "; ".join(
                         "spin %s %s %s" % (
                             SPIN_NAMES[j2], COLUMN_NAMES[name],
                             _lowest_text(per_column[name][j2], False))
                         for j2 in SPINS for name in COLUMNS)))
    per_column = {name: lowest_over_pairs(records, name, tolerance)
                  for name in COLUMNS}
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


def _ratio_flag_text(r):
    """The flags of the contents a ratio is built from, as text; empty when
    neither is flagged."""
    parts = ["s_%s read from a flagged content (%s)" % (symbol, "; ".join(
        r[role + "_flags"]))
             for role, symbol in (("nucleon", "N"), ("delta", "D"))
             if r.get(role + "_flags")]
    return "".join("; " + part for part in parts)


def ratio_lines(point_ratios, prefix=""):
    """The ratios of a set of records as text, one line per column and
    pairing, each naming the two (content, doublet content) pairs it
    compares and the flags of the contents they are read from."""
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
                "target m_N/m_Delta=%.4f%s%s%s%s"
                % (_complex_text(r["nucleon_pole"]), r["nucleon_content"],
                   r["nucleon_doublet_content"],
                   _complex_text(r["delta_pole"]), r["delta_content"],
                   r["delta_doublet_content"],
                   _complex_text(r["pole_ratio"]), r["modulus_ratio"],
                   r["real_part_ratio"], TARGET_MASS_RATIO, note,
                   _tie_text(r, "nucleon"), _tie_text(r, "delta"),
                   _ratio_flag_text(r)))
    return lines


def point_lines(point):
    """One scan point as text: how many contents were read, how many of them
    have no value and how many are flagged; every content's line, with its
    mean-field solve, its anchor atlas, its spectral fingerprint and its
    flags (`flags_text`); every (content, doublet content) pair on its own
    line; then the labelled minima, then the ratios with the pairs they
    compare."""
    records = point["contents"]
    lines = ["kappa=%g beta=%g: %d contents, %d without a value, %d flagged; "
             "one line per (content, doublet content) pair, poles s with "
             "multiplicity x"
             % (point["kappa"], point["beta"], len(records),
                sum(1 for record in records if "failed" in record),
                sum(1 for record in records if record.get("flags")))]
    for record in records:
        if "failed" not in record:
            flagged = flags_text(record)
            lines.append("  content %s %s%s%s%s" % (
                list(record["content"]),
                relaxation_text(record.get("relaxation")),
                "; " + anchor_text(record["anchor"])
                if "anchor" in record else "",
                "; " + fingerprint_text(record["spectral_fingerprint"])
                if "spectral_fingerprint" in record else "",
                "; " + flagged if flagged else ""))
        lines += term_trace_lines(record.get("relaxation"), "    ")
        lines += content_pair_lines(record, "  ")
    lines += lowest_lines(records, "  ",
                          point.get("tie_tolerance", DECLARED_TIE_TOLERANCE))
    lines += ratio_lines(point.get("ratios"), "  ")
    return lines


def default_config(kappas=DECLARED_KAPPAS, betas=DECLARED_BETAS,
                   edge_squared=DECLARED_EDGE_SQUARED,
                   regge_hinges="interior", selected_contents=None,
                   elimination=DECLARED_ELIMINATION,
                   band_selection=DECLARED_BAND_SELECTION,
                   fiber_moments=DECLARED_FIBER_MOMENTS,
                   fiber_pinning=DECLARED_FIBER_PINNING, tolerances=None,
                   trace_terms=False, limits=None,
                   villain_order=DECLARED_VILLAIN_ORDER, solve=None):
    """The declared configuration, recorded with every run. The contents
    default to all ten; a subset is for tests and quick checks and changes no
    number of the contents it keeps. ``tolerances`` sets any of `TOLERANCES`
    by key; the others are recorded at `DECLARED_TOLERANCE`. ``limits``
    declares any of `LIMITS` by key; the others are recorded as None, not
    declared. ``villain_order`` is the order the Villain weight of the
    holonomy term is summed to (`DECLARED_VILLAIN_ORDER`). ``solve`` sets
    any of `SOLVE_OPTIONS` by key; the others are recorded at their declared
    values. A content that is not one of the ten (`checked_contents`), and a
    name of the hinges, the elimination, the band selection or the fiber
    pinning that is not declared, has no value and is refused here, before
    anything is computed."""
    checked_regge_hinges(regge_hinges)
    if elimination not in ELIMINATIONS:
        raise ValueError("the elimination is one of %s; got %r"
                         % (", ".join(ELIMINATIONS), elimination))
    if band_selection not in BAND_SELECTIONS:
        raise ValueError("the band selection is one of %s; got %r"
                         % (", ".join(BAND_SELECTIONS), band_selection))
    if fiber_pinning not in FIBER_PINNINGS:
        raise ValueError("the fiber pinning is one of %s; got %r"
                         % (", ".join(FIBER_PINNINGS), fiber_pinning))
    return {
        "mode": "controlled synthesis",
        "contents": [list(c) for c in checked_contents(
            selected_contents or contents())],
        "kappas": list(kappas),
        "betas": list(betas),
        "edge_squared": edge_squared,
        "regge_hinges": regge_hinges,
        "villain_order": checked_villain_order(villain_order),
        "elimination": elimination,
        **declared_tolerances(tolerances),
        **declared_limits(limits),
        **declared_solve_options(solve),
        "band_selection": band_selection,
        "fiber_moments": str(fiber_moments),
        "fiber_pinning": fiber_pinning,
        "trace_terms": bool(trace_terms),
        "kappa_role": (
            "kappa = 8 pi G enters only through the Regge weight 1/kappa; "
            "the spectral-moment part of S_0 is the holomorphic spectral "
            "constraint of WP v17 §3.4 on the occupied fiber"),
        "target_mass_ratio": TARGET_MASS_RATIO,
        "target_reading": "the pole is the complex rest energy of the bound "
                          "cluster under the first-order flow (WP v18 "
                          "§13.3); s_N / s_Delta is compared with m_N / "
                          "m_Delta",
    }


def _git_commit(directory):
    """The commit a directory's checkout is at, read from its `.git` (a
    directory, or the file of a linked worktree that names one) without
    running git: ``HEAD``, and for a symbolic HEAD the ref's file, in the
    worktree's own git directory or the common one, or its line of
    ``packed-refs``. Returns the checkout's root, the ref and the commit;
    None for each that is not found."""
    root = os.path.abspath(directory)
    while not os.path.exists(os.path.join(root, ".git")):
        parent = os.path.dirname(root)
        if parent == root:
            return {"root": None, "ref": None, "commit": None}
        root = parent
    git = os.path.join(root, ".git")
    try:
        if os.path.isfile(git):
            with open(git) as handle:
                git = os.path.normpath(os.path.join(
                    root, handle.read().split("gitdir:", 1)[1].strip()))
        common = git
        if os.path.isfile(os.path.join(git, "commondir")):
            with open(os.path.join(git, "commondir")) as handle:
                common = os.path.normpath(os.path.join(git,
                                                       handle.read().strip()))
        with open(os.path.join(git, "HEAD")) as handle:
            head = handle.read().strip()
        if not head.startswith("ref:"):
            return {"root": root, "ref": None, "commit": head}
        ref = head.split("ref:", 1)[1].strip()
        for base in (git, common):
            path = os.path.join(base, ref)
            if os.path.isfile(path):
                with open(path) as handle:
                    return {"root": root, "ref": ref,
                            "commit": handle.read().strip()}
        packed = os.path.join(common, "packed-refs")
        if os.path.isfile(packed):
            with open(packed) as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return {"root": root, "ref": ref, "commit": parts[0]}
        return {"root": root, "ref": ref, "commit": None}
    except (OSError, IndexError):
        return {"root": root, "ref": None, "commit": None}


def environment_record():
    """What a run's numbers depend on beside its configuration, recorded
    once in the header of its files: the command line as run (``argv``),
    the thread counts the environment declares (``OMP_NUM_THREADS`` and the
    linear algebra library's own variables, None for one that is not set)
    with the number of processors, the linear algebra numpy is built on,
    the versions of Python, numpy and the package, and the commit of the
    checkout the package is imported from (`_git_commit`). Decisions taken
    at the declared tolerances on values that differ in their last bits
    between thread counts depend on these."""
    try:
        build = np.show_config(mode="dicts").get("Build Dependencies", {})
        linear_algebra = {
            name: {key: build[name].get(key)
                   for key in ("name", "version", "openblas configuration")
                   if build[name].get(key) is not None}
            for name in ("blas", "lapack") if name in build}
    except (TypeError, AttributeError):
        linear_algebra = {}
    return {
        "argv": [str(argument) for argument in sys.argv],
        "threads": {name: os.environ.get(name)
                    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                 "MKL_NUM_THREADS")},
        "processors": os.cpu_count(),
        "linear_algebra": linear_algebra,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "package": str(getattr(T, "__version__", None)
                       or getattr(getattr(T, "_tessera", None),
                                  "__version__", None) or "unknown"),
        "checkout": _git_commit(os.path.dirname(os.path.abspath(
            T.__file__))),
    }


def points_path(json_path):
    """The append-only JSON-lines file beside ``json_path`` that holds one
    line per completed scan point (and a first line with the configuration
    and the host), so a stopped run loses nothing."""
    root, _ = os.path.splitext(os.fspath(json_path))
    return root + ".points.jsonl"


def _append_line(path, record):
    """One record appended to a points file as one line of JSON; a value
    that is not JSON is refused by the writer and never written."""
    with open(path, "a") as handle:
        handle.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def drive(config, progress=False, on_frame=None, stop_requested=None,
          points_file=None):
    """The whole scan. `on_frame(frames, index)` is called after each scan
    point with the list of completed points; the computation is the same with
    or without it. With `points_file`, the configuration and the host are
    written as the first line and every scan point's full record is appended
    as one JSON line the moment the point completes."""
    # the declared host's own read, recorded once; every relaxed cell is read
    # against its own rotation group in `evaluate_content`
    declared = aligned_doublet_frame(
        monopole_support(), rotation_group(),
        declared_tolerance(config, "degeneracy_tolerance"),
        declared_tolerance(config, "certificate_tolerance"),
        declared_tolerance(config, "character_tolerance"), certified=False)
    frames = []
    host = {
        "monopole": _monopole_record(declared["spin_read"],
                                     declared["bands"]),
        "averaged_eigenvalues": declared["averaged_eigenvalues"],
        "reference_carrier": declared["reference_carrier"],
        "reference_certified": declared["reference_certified"],
        "reference_doublet": declared["reference_doublet"],
        "trialities": declared["trialities"],
        "intertwining_residual": declared["intertwining_residual"],
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
            point = scan_point(kappa, beta, config)
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


def _monopole_record(read, bands):
    """The record of a support's monopole and spin read: the monopole
    number, the projective class, and the symmetry-protected bands the
    driver reads (``bands``, the certificates of `symmetry_bands`, the bands
    `aligned_doublet_frame` takes its reference doublet from) with whether
    one of them is the j = 1/2 doublet and its place among them. The reading
    of the library's own grouping (`MonopoleSupport.spinRead`) is recorded
    beside it under ``library`` and named as the library's."""
    doublet = reference_doublet(bands)
    return {
        "monopole_number": int(read.monopole.monopole_number),
        "odd": bool(read.monopole.odd),
        "cocycle_nontrivial": bool(read.cocycle.nontrivial),
        "commutator_phase": complex(read.cocycle.commutator_phase),
        "half_integer_doublet": doublet is not None,
        "doublet_index": (len(bands) if doublet is None
                          else next(k for k, band in enumerate(bands)
                                    if band is doublet)),
        "bands": [{"eigenvalue": float(b["eigenvalue"]),
                   "dimension": int(b["dimension"]),
                   "spinor_doublet": bool(b["spinor_doublet"]),
                   "coexact": bool(b["coexact"]),
                   "coexact_residual": float(b["coexact_residual"]),
                   "invariance_residual": float(b["invariance_residual"]),
                   "irreducibility_score": float(b["irreducibility_score"])}
                  for b in bands],
        "library": {
            "half_integer_doublet": bool(read.half_integer_doublet),
            "doublet_index": int(read.doublet_index),
            "bands": [{"eigenvalue": float(b.eigenvalue),
                       "dimension": int(b.dimension),
                       "spinor_doublet": bool(b.spinor_doublet),
                       "coexact": bool(b.coexact),
                       "coexact_residual": float(b.coexact_residual),
                       "irreducibility_score": float(
                           b.irreducibility_score)}
                      for b in read.bands],
        },
        "certificate": read.certificate.describe(),
    }


def _complex_text(value):
    return "%.6g%+.3gi" % (value.real, value.imag)


#: How a float that is not finite is written: JSON has no NaN and no
#: infinity, so each is the string of its name (`_jsonable`, `number`).
NON_FINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def _real(value):
    """A float as JSON holds it: itself when finite, and the string "nan",
    "inf" or "-inf" otherwise."""
    value = float(value)
    if math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"


def _jsonable(value):
    """The JSON-ready form of a record: a complex number is ``{"re", "im"}``,
    an array a nested list, a numpy scalar the Python number or truth value
    it holds, every key a string, and a float that is not finite the string
    of its name (`NON_FINITE`: "nan", "inf", "-inf"; the parts of a complex
    number likewise), which `number` reads back. The result holds no value
    that is not JSON."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (complex, np.complexfloating)):
        return {"re": _real(value.real), "im": _real(value.imag)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _real(value)
    return value


def number(value):
    """A number read back from a written record: the float of a string that
    names one that is not finite ("nan", "inf", "-inf"), the complex number
    of a written ``{"re", "im"}`` pair, and the value itself otherwise."""
    if isinstance(value, str) and value in NON_FINITE:
        return NON_FINITE[value]
    if isinstance(value, dict) and set(value) == {"re", "im"}:
        return complex(number(value["re"]), number(value["im"]))
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
    "no value": {"ink": INK_MUTED, "band": "#ecebe6", "sign": "\u2717",
                 "label": "pole read has no value"},
}
#: The short names, in a callout, of why a content's solve stopped
#: (`cell_solve.solve`) and of why a pole read has no value
#: (`geometry_without_value`).
STOP_SHORT = {
    cell_solve.STOP_STATIONARY: "no descent",
    cell_solve.STOP_NO_STEP: "no step",
    cell_solve.STOP_DECLARED_LIMIT: "declared limit",
    NO_VALUE_OVERFLOW: "lengths overflowed",
    NO_VALUE_MOVED: "cell moved",
    NO_VALUE_READ: "read without a value",
}
#: The largest font size of the callouts; they are set smaller, all to one
#: size, when the narrowest group needs it.
CALLOUT_FONT_SIZE = 7.0


def _digits(values):
    """A content or a doublet content as its digits: [0, 2, 1] is 021."""
    return "".join(str(int(v)) for v in values)


def solve_state(record):
    """The mean-field solve behind one content record, as the plots mark it:
    ``state`` is "no value" when the content's pole read has none (its
    record carries ``failed`` and no pole) and otherwise "converged" or "not
    converged" as the solve's record says, whether or not the read is
    flagged; ``reason`` is the short name (`STOP_SHORT`) of why an
    unconverged solve stopped or of why the read has no value, None when the
    record names none; ``accepted_updates`` is the number of relaxation
    updates the solve's drive accepted, None when unrecorded. None when the
    record carries neither ``failed`` nor a solve."""
    relaxation = record.get("relaxation") or {}
    updates = relaxation.get("accepted_updates")
    if "failed" in record:
        name = record.get("reason")
        return {"state": "no value", "reason": STOP_SHORT.get(name, name),
                "accepted_updates": updates}
    if "converged" not in relaxation:
        return None
    if relaxation["converged"]:
        return {"state": "converged", "reason": None,
                "accepted_updates": updates}
    stop = relaxation.get("stop_reason")
    return {"state": "not converged", "reason": STOP_SHORT.get(stop, stop),
            "accepted_updates": updates}


def callout_lines(solve):
    """The lines of one group's callout (`solve_state`): the sign and the
    state, then why the solve stopped or the read has no value, then the
    number of relaxation updates the solve accepted, each where known."""
    lines = ["%s %s" % (SOLVE_STYLE[solve["state"]]["sign"], solve["state"])]
    if solve["reason"]:
        lines.append(solve["reason"])
    if solve["accepted_updates"] is not None:
        count = int(solve["accepted_updates"])
        lines.append("%d update%s" % (count, "" if count == 1 else "s"))
    return lines


def pole_marks(groups):
    """The marks of the pole panels: one mark per distinct pole of every
    (group, doublet content, spin, column) sector; nothing is combined.

    ``groups`` is a list of (label, content record). A group occupies one
    slot per doublet content its record read, in the record's order, and
    groups are separated by `GROUP_GAP` empty slots; a content with no value,
    which reads no doublet content, occupies one slot labelled "no value".
    Returns
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
            slots.append((x, "no value" if "failed" in record else "none"))
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
    names). Each group whose record carries a mean-field solve or has no
    value (`solve_state`) has a band behind its pairs and a callout above
    them in its state's colour (`SOLVE_STYLE`): whether the solve converged,
    why it stopped or the read has no value, and its accepted updates
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
    run.add_argument("--regge-hinges", choices=tuple(REGGE_HINGES),
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
    add_action_arguments(run)
    add_mean_field_arguments(run)
    add_tolerance_arguments(run)
    add_limit_arguments(run)
    add_solve_arguments(run)
    run.add_argument("--quiet", action="store_true")
    return parser


def _villain_order(text):
    """An integer from 1 to the largest order, for --villain-order."""
    try:
        return checked_villain_order(int(text))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--villain-order is an integer from 1 to %d; got %r"
            % (cob.VillainCharacter.maximum_order, text))


def add_action_arguments(parser):
    """The options of the joint action every driver accepts. Each is part of
    the action: it changes the equations that are solved."""
    parser.add_argument("--villain-order", type=_villain_order,
                        default=DECLARED_VILLAIN_ORDER,
                        help="M, the order the Villain weight of the "
                             "holonomy term is summed to: the term is "
                             "defined by the sum over |m| <= M; an integer "
                             "from 1 to %d (default %d)"
                             % (cob.VillainCharacter.maximum_order,
                                DECLARED_VILLAIN_ORDER))


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


def add_limit_arguments(parser):
    """One option per limit a user may declare on a solve (`LIMITS`), for
    both drivers. None is declared unless the option is given."""
    for key, kind, meaning in LIMITS:
        parser.add_argument("--" + key.replace("_", "-"), dest=key,
                            type=kind, default=None,
                            help="%s (not declared by default: no count and "
                                 "no time ends a solve)" % meaning)


def limits_from(args):
    """The parsed limit options, by config key (`LIMITS`); None where the
    option was not given."""
    return {key: getattr(args, key) for key, _, _ in LIMITS}


def _direction_order(text):
    """An integer from 1 to the largest order, for --direction-order."""
    try:
        return checked_direction_order(int(text))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--direction-order is an integer from 1 to %d; got %r"
            % (cell_solve.MAXIMUM_DIRECTION_ORDER, text))


def add_solve_arguments(parser):
    """One option per option of a solve's drive (`SOLVE_OPTIONS`), for every
    driver."""
    meaning = {key: text for key, _, text in SOLVE_OPTIONS}
    parser.add_argument("--direction-order", type=_direction_order,
                        default=DECLARED_DIRECTION_ORDER,
                        help="%s (default %d)" % (meaning["direction_order"],
                                                  DECLARED_DIRECTION_ORDER))
    parser.add_argument("--band-reference",
                        choices=cell_solve.BAND_REFERENCES,
                        default=cell_solve.DECLARED_BAND_REFERENCE,
                        help="%s (default %s)"
                             % (meaning["band_reference"],
                                cell_solve.DECLARED_BAND_REFERENCE))
    parser.add_argument("--no-pachner-moves", dest="pachner_moves",
                        action="store_false",
                        help="relax the geometry alone: the drive scores no "
                             "Pachner move (by default it scores every one)")
    parser.add_argument("--combinatorial-depth", type=int,
                        default=cell_solve.DECLARED_COMBINATORIAL_DEPTH,
                        help="%s (default %d)"
                             % (meaning["combinatorial_depth"],
                                cell_solve.DECLARED_COMBINATORIAL_DEPTH))
    parser.add_argument("--combinatorial-length", "--combinatorial-breadth",
                        dest="combinatorial_length", type=int,
                        default=cell_solve.DECLARED_COMBINATORIAL_LENGTH,
                        help="%s (default %d)"
                             % (meaning["combinatorial_length"],
                                cell_solve.DECLARED_COMBINATORIAL_LENGTH))
    parser.add_argument("--candidate-moves", type=int, default=0,
                        help="%s (default 0)" % meaning["candidate_moves"])
    parser.add_argument("--moment-stiffness-weight", type=float, default=0.0,
                        help="%s (default 0)"
                             % meaning["moment_stiffness_weight"])
    parser.add_argument("--moment-stiffness-coefficients", type=float,
                        nargs="+", default=[],
                        help=meaning["moment_stiffness_coefficients"])
    parser.add_argument("--pinned-vertices", type=int, nargs="+", default=[],
                        help=meaning["pinned_vertices"])


def solve_options_from(args):
    """The parsed options of a solve's drive, by config key
    (`SOLVE_OPTIONS`)."""
    return {key: getattr(args, key) for key, _, _ in SOLVE_OPTIONS}


def _fiber_moments(text):
    """``r``, ``bands`` or a non-negative integer, for --fiber-moments."""
    if text in ("r", "bands"):
        return text
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 0:
        raise argparse.ArgumentTypeError(
            "--fiber-moments is r, bands or a non-negative integer; got %r"
            % text)
    return str(value)


def add_mean_field_arguments(parser):
    """The mean-field solver options both drivers accept. Neither changes an
    equation."""
    parser.add_argument("--band-selection", choices=tuple(BAND_SELECTIONS),
                        default=DECLARED_BAND_SELECTION,
                        help="where a content's bands are chosen: at the "
                             "host, then followed by continuation, or "
                             "re-selected by sorting at every iterate "
                             "(default %s)" % DECLARED_BAND_SELECTION)
    parser.add_argument("--fiber-pinning", choices=tuple(FIBER_PINNINGS),
                        default=DECLARED_FIBER_PINNING,
                        help="what the fiber constraints pin at the host: "
                             "eigenvalues, the eigenvalue of each occupied "
                             "band, one constraint per band; or power-sums, "
                             "the power sums p_j(h_C) of the occupied fiber "
                             "(WP v17 §3.4), as many as --fiber-moments "
                             "(default %s)" % DECLARED_FIBER_PINNING)
    parser.add_argument("--fiber-moments", type=_fiber_moments,
                        default=DECLARED_FIBER_MOMENTS,
                        help="m_c, the number of fiber constraints pinned at "
                             "the host: r, every occupied band's eigenvalue "
                             "or, pinning power sums, the fiber's rank of "
                             "them; bands, one per occupied band; or a "
                             "count, 0 pinning nothing (default %s)"
                             % DECLARED_FIBER_MOMENTS)
    parser.add_argument("--trace-terms", action="store_true",
                        help="record and print, at every iterate of every "
                             "relaxation, each term of the joint action with "
                             "its value, the norm of its stationarity "
                             "gradient on the relaxed coordinates, and the "
                             "change of both since the previous iterate")


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = default_config(args.kappa, args.beta, args.edge_squared,
                            args.regge_hinges,
                            elimination=args.eliminate,
                            band_selection=args.band_selection,
                            fiber_moments=args.fiber_moments,
                            fiber_pinning=args.fiber_pinning,
                            tolerances=tolerances_from(args),
                            trace_terms=args.trace_terms,
                            limits=limits_from(args),
                            villain_order=args.villain_order,
                            solve=solve_options_from(args))
    if args.isospin_doublet:
        config["isospin_doublet"] = True
    # what the numbers of the run depend on beside the configuration,
    # written once in the header
    config["environment"] = environment_record()
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
        hold_live_window("the run is complete and every output is written; "
                         "close this window to exit")
    return result


if __name__ == "__main__":
    main()
