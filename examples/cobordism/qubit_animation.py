# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Test geometric harmonic state/operator correspondence on a live cobordism.

The primary operator evidence is the target-independent ``correspondence``
readout: whole-kernel transport, basis and held-out reconstruction, declared
coordinate norms and Choi data. It reports the missing two-qubit tensor
register explicitly. No matrix-valued connection is prescribed.

The synthesis setup described below is retained as the historical
selected-state experiment. Its coupling-to-chi fit is not a full gate
realization certificate; the independent whole-kernel readout decides which
operator claims the accepted scalar geometry supports.

This is the qubit experiment of ``docs/design/qubit_cobordism_spec.md``.
Two flat qubit tori (``SimplicialQubit.flat_torus``) are the boundary of a
3-complex whose bulk starts as the collar between them
(``MultiCobordism.seed_collar``) and is then synthesized by the shared stage-1
and stage-2 loop against the two-body target chi of spec S5 -- the XY
flip-flop of two spin-1/2 at ``--J`` and ``--time`` -- while each torus keeps
representing its input state through the zero mode of its OWN Laplacian.

The block residual of spec D2,
``1 - |<psi(tau_in)|psi(tau_hat)>|^2``, uses tau_hat, the ratio of the
transported periods of the holomorphic form of the block's own Laplacian on
its live surface (``MultiCobordism.block_qubit``), and enters the objective at
``--input-weight``. The input surfaces are held by default so the bulk is
fitted to the declared inputs; ``--no-pin-boundary`` restores the historical
freely relaxed boundary. Each block holds its marking
(``set_input_marking``), which fixes the cycles the periods are taken over
and, by A.B = +1, the orientation the surface is read in.

The zero mode of the ENTIRE cobordism is the OUTPUT state (spec R1): it is
read at each torus as its coefficients in that torus's live frame -- the zero
mode of its own Laplacian normalized by its marking, derived by the engine at
every read -- and never held. The objective in force is the node's default
(``legacy``): the Regge stationarity term when ``--regge`` is on, plus Gamma
times r_U, where r_U is the weighted sum of the two block residuals and the
two-body residual. The tori carry complex lengths, so the node runs the
complex locus.

Every read-out of spec S6 is a frame channel, present or ``Absent``. Per block
the frame carries the block residual with its weight; the output state -- the
coefficients of the whole's zero mode in the block's live frame next to the
input (1, tau_in), with the leak of that fit; the former own-kernel leak as a
labelled diagnostic; and the qubit read on the block's live surface (tau-hat,
the Bloch vector, the J residual, the Delaunay and condition diagnostics, and
the Fubini-Study and Weil-Petersson distances to the input), the object the
block residual scores. For the whole it carries the Betti numbers; boundary
components with Euler characteristics; completion status; monodromy between
the two markings; restricted leak of each input line in the whole's zero
mode; the two-body read (transfer in the derived period frames, its leak
against chi, Schmidt spectrum and rank, and reversal residual); and the
objective terms. Unsupported representations are refused explicitly rather
than silently scored.

Run from the repository root::

    OMP_NUM_THREADS=8 python examples/cobordism/qubit_animation.py run \
      --tau-a 0.3+1.1j --tau-b=-0.2+0.8j --grid 3 --steps 4 \
      --out ~/cobordism-runs/qubit-cobordism/run.gif \
      --json ~/cobordism-runs/qubit-cobordism/run.json

``--tau-b=-0.2+0.8j`` needs the ``=`` because a leading minus is otherwise
read as an option name.
"""

import argparse
import itertools
import math
import sys
import warnings

from numpy.linalg import LinAlgError

import tessera as T

import emergence_animation as ea
import harmonic_correspondence as hc

cob = T.cobordism
obs = T.observables
MC = cob.MultiCobordism

# Shared measurement and presentation primitives.  Keeping the aliases local
# preserves the established example-module surface while all implementations
# remain single-sourced in the neutral animation module.
Absent = ea.Absent
Terminator = ea.Terminator
DriveResult = ea.DriveResult
_finite = ea._finite
_json_safe = ea._json_safe
_absent_panel = ea._absent_panel
_wrap = ea._wrap

# Shared drive and presentation defaults are aliases so the two entrypoints
# cannot acquire different engine-unit semantics.
DECLARED_STEPS = ea.DECLARED_STEPS
DECLARED_CANDIDATE_MOVES = ea.DECLARED_CANDIDATE_MOVES
DECLARED_STAGE1_ITERS = ea.DECLARED_STAGE1_ITERS
DECLARED_STAGE2_ITERS = ea.DECLARED_STAGE2_ITERS
DECLARED_COMBINATORIAL_DEPTH = ea.DECLARED_COMBINATORIAL_DEPTH
DECLARED_SURGICAL_DEPTH = DECLARED_COMBINATORIAL_DEPTH
DECLARED_COMBINATORIAL_LENGTH = ea.DECLARED_COMBINATORIAL_LENGTH
DECLARED_COMBINATORIAL_BREADTH = DECLARED_COMBINATORIAL_LENGTH
DECLARED_TOLERANCE = ea.DECLARED_TOLERANCE
DECLARED_PATIENCE = ea.DECLARED_PATIENCE
DECLARED_REGISTER_DEGREES = ea.DECLARED_REGISTER_DEGREES
# The qubit experiment uses the shared degree-one analysis and all-degree
# stationarity defaults. Betti numbers remain a separate topological read.
DECLARED_HODGE_DEGREES = ea.DECLARED_HODGE_DEGREES
DECLARED_ANALYSIS_DEGREES = ea.DECLARED_ANALYSIS_DEGREES
DECLARED_SEED = ea.DECLARED_SEED
DECLARED_INPUTS = "qubit"
LIVE_POLL_INTERVAL = ea.LIVE_POLL_INTERVAL
_LEGACY_UNSET = ea._LEGACY_UNSET

#: Which space the two-body target is scored against (`--readout`), as a
#: comma-separated set that is SUMMED. Each name is the SPACE the reading takes
#: the state from, so a name cannot suggest the wrong one.
#:
#: `transfer` is (Z_A^v)^T A~_1 Z_B: the whole complex's degree-1 operator read
#: as the coupling block between the two boundary frames. It is 4-dimensional
#: and factorizes across the boundaries, which is what lets it carry an
#: entangled target.
#:
#: `bulk` would be ker L_1(W - dW): the Laplacian on interior cells with the
#: boundary REMOVED, read through a Choi frame. The engine API accepts an
#: explicitly supplied frame, but this driver has no canonical choice of d^2
#: interior edges. It is refused rather than scoring a constant 1.0.
#:
#: `whole` is ker L_1(W): the boundary INCLUDED, read through the blocks'
#: markings. Two boundary tori give rank 2 and accept an explicitly declared
#: 2-vector output state. Four give rank 4 only as a direct sum of torus
#: cohomologies, not in the tensor-product coordinates of the two-qubit target,
#: so this driver refuses that dimensional coincidence.
#:
#: This is part of the OBJECTIVE. The two-body residual is a term in r_U, so
#: every choice prices stage-1 moves. Transfer also drives stage-2 descent;
#: whole is invariant under length motion at fixed marked topology and has a
#: zero analytic direction there.
DECLARED_READOUT = "transfer"
#: How many tori bound the cobordism (`--tori`): two in the runnable driver.
#: Four remains an accepted compatibility value so legacy invocations receive
#: the precise representation mismatch below instead of an argparse mystery;
#: the paired four-torus construction is available only as a low-level
#: geometric diagnostic.
#:
#: TWO is one torus per input state, the collar of spec S3.
#:
#: FOUR carries each state on a PAIR -- itself and its orientation reversal at
#: -conj(tau), which keeps the modulus in the upper half-plane where a torus
#: modulus has to live and gives the holomorphic form the conjugate periods.
#: The host becomes two collars joined along a removed tetrahedron; gluing
#: along a sphere is a connected sum, which adds no first homology, so
#: b_1 = 2 + 2 = 4 with nothing dying on the boundary.
#:
#: The count decides which readings can run at all. Although
#: rank(H^1(W) -> H^1(dW)) = b_1(dW)/2 gives rank 4 at four tori, that space is
#: a direct sum rather than a two-qubit tensor product. `_READOUT_TORI` records
#: and enforces the distinction.
#:
#: Four needs three collar layers. A prism cell spans two adjacent layers, so
#: the all-interior cell the join removes exists only with two interior layers.
DECLARED_TORI = 2

#: The mapping class that relabels the far surface before the collar
#: identifies the two. "none" is the PRODUCT collar, whose two boundary tori
#: are homologous, so its monodromy is the identity and it carries its input
#: straight through -- measured immovable, ||M - I|| stays at 5e-15 under a
#: 75% jitter of every squared length, because the monodromy is the induced
#: map on H^1 and no metric touches it. "swap" exchanges the two cycles,
#: giving M = [[0, 1], [1, 0]] and the propagator tau -> 1/tau.
#:
#: Only these two: a mapping class must be a SIMPLICIAL automorphism of the
#: torus's triangulation to collar at all, and on this one a Dehn twist sends
#: the diagonal to a step that is no edge (it is refused by name). The swap is
#: orientation-reversing, so the twisted host is an I-bundle that is not a
#: product.
DECLARED_COLLAR_TWIST = "none"

#: The causal character the COLLAR'S INTERIOR edges are seeded with.
#: Defaults to `spacelike`, which is what `seed_collar` already wires, so the
#: flag changes no existing run until it is asked to. It is recorded under
#: `interior_disposition` rather than `edge_disposition` because the latter
#: is the NEUTRAL driver's key for the whole host's edges, and a qubit
#: configuration carries none of those. The command-line flag stays
#: `--edge-disposition`.
#:
#: `seed_collar` writes each torus's own lengths onto its edges and wires
#: every other edge to 1.0, so without this a qubit host is all-spacelike in
#: its bulk and cannot say otherwise.
#:
#: The tori's own edges are never written: they carry the declared input
#: moduli, so a disposition over them would change the input states rather
#: than the bulk they sit in. `spacelike` is therefore a no-op here rather
#: than a rewrite, since it is already what the collar wires.
#:
#: The vocabulary is `emergence_animation.EdgeDisposition`, shared with the
#: neutral driver so the two cannot drift.
DECLARED_INTERIOR_DISPOSITION = ea.EdgeDisposition.SPACELIKE

#: How the whole-complex reading pairs its harmonic columns against the input
#: blocks (`--whole-pairing`).
#:
#: "periods" integrates each column over the marked cycles. That is how a
#: state is DEFINED on a boundary torus, where the Hodge star is an
#: endomorphism of H^1 and a modulus is the period ratio of the holomorphic
#: line it selects. On the three-dimensional bulk there is no such line -- the
#: star sends 1-forms to 2-forms -- and the reading is purely topological: a
#: change of metric moves the harmonic representative by exactly a coboundary
#: (100% of a change of 1.9 times the cochain's own norm, measured), and a
#: coboundary's period around a closed cycle telescopes to zero. The residual
#: does not move, 0.02430251774083792 to 0.02430251774083781 under a 75%
#: jitter of every squared length, so a run scored on it cannot improve.
#:
#: "gram" contracts through the chain metric instead. The metric content is
#: in the harmonic space either way; the period pairing is simply the one
#: contraction that annihilates it, and the same jitter moves the Gram by 76%.
DECLARED_WHOLE_PAIRING = "periods"

#: The engine's whole-reading pairings by their command-line names.
_WHOLE_PAIRINGS = {
    "periods": MC.WholePairing.PERIODS,
    "gram": MC.WholePairing.GRAM,
}

#: The state the whole complex's harmonic form is meant to be (`--output-state`),
#: as a modulus tau: the target is psi(tau) = (1, tau)/|(1, tau)|.
#:
#: The harmonic of the whole Laplacian IS a state. On the supported two-torus
#: host its rank is 2, and this names that 2-vector. It is required because the
#: ordinary two-body target has four tensor-product amplitudes and cannot stand
#: in for it.
DECLARED_OUTPUT_STATE = None

#: The two input moduli of the qubit mode, both in the upper half plane
#: (spec S1): the pair the T1-T3 tests measured on.
DECLARED_TAU_A = complex(0.3, 1.1)
DECLARED_TAU_B = complex(-0.2, 0.8)
#: Grid of each flat torus (n x n). Below 3 the grid is not a simplicial
#: complex, which `SimplicialQubit.flat_torus` refuses.
DECLARED_GRID = 3
#: The exchange coupling J and the time t of the flip-flop
#: `H = hbar J (s1+ s2- + s1- s2+)`; only their product `J t` enters the
#: amplitudes, but both are the experiment's declared parameters.
DECLARED_COUPLING = 1.0
DECLARED_TIME = 0.05
#: Weight of each block's own-Laplacian residual of spec D2 inside r_U --
#: the T5 findings note's chosen weight. The seed sits at the residual's
#: minimum (each torus IS its input torus there, residual zero), so the
#: weight sets how far the bulk's relaxation is allowed to pull the tori's
#: own conformal structures away from their inputs; at 1e4 two units leave
#: both residuals below 1e-3. Above about 1e5 the drive is stationary at the
#: seed for a reason outside this term: the engine's Regge stationarity is
#: discontinuous across the real locus (measured on origin/main: an
#: imaginary displacement of 1e-12 in one squared length moves it by 1.5),
#: and the residual's descent direction, holomorphic in z, always has one.
DECLARED_INPUT_WEIGHT = 1e4
#: Whether a CONE move may extend the boundary of W (`--extend-boundary`).
#: False, which is also the engine's default. This is a qubit-mode choice:
#: its two surface input blocks declare the boundary up front, and a cone that
#: hands dW a face it did not have changes what the cobordism IS rather than
#: how it is shaped. The neutral mode's M0 is only a pinned region; it holds
#: coordinates without declaring a topological boundary, so that mode refuses
#: a non-default value. A cone-in buries the facet it stands on and exposes
#: the new cell's others; a cone-out exposes every facet of the cell it
#: removes. The choice is recorded in the run document.
DECLARED_EXTEND_BOUNDARY = False
#: Whether the WHOLE cobordism's leak of each input state is scored in the
#: objective beside the block's own residual (`--score-leak`).
#:
#: Two different numbers measure whether an input torus still carries its
#: state. The block's own residual builds the Laplacian of that one torus in
#: isolation and asks whether the torus, judged alone, still represents what it
#: was handed. The leak builds the Laplacian of the entire cobordism, takes its
#: zero mode, and asks how much of the input coefficients fails to lie in it --
#: the torus's relationship to everything else. Scoring only the first leaves
#: the second unconstrained and it drifts upward as the two-body term falls.
#:
#: Off, which is the engine's default, so a run is unchanged unless asked.
DECLARED_SCORE_LEAK = False
#: Extra input pairs the SAME bulk must also satisfy (`--state`), as
#: "tau_a,tau_b" strings. Empty means the single pair of --tau-a/--tau-b.
#:
#: A bulk fitted to one pair reproduces that pair and nothing else: measured on
#: a converged geometry, its own pair reads 1.44e-11 and three others read 0.26
#: to 0.65, under every one of the 108 attachment automorphisms. That is what
#: the arithmetic predicts rather than a defect -- one 2x2 transfer against one
#: pair is six real constraints against some eighty free bulk coordinates, so
#: nothing ever asked the geometry to be a MAP.
#:
#: Each extra pair adds six more constraints on the same bulk. Every case
#: shares one triangulation, one gluing and one bulk; its boundary metric,
#: target and marking coefficients describe that input. Use with
#: --pin-boundary: with the boundary free the cases would fight over
#: coordinates the relaxation may move, which is not the experiment.
DECLARED_STATES = None

DECLARED_REGGE = True
#: Product layers of the collar seed. Spec S3 seeds the MINIMAL manifold
#: between the tori, which is one layer; nothing beyond it is templated.
DECLARED_COLLAR_LAYERS = 1
#: Degrees the Betti numbers of the 3-dimensional qubit host are reported at
#: (the collar is [1, 2, 1, 0]).
DECLARED_QUBIT_BETTI_DEGREES = (0, 1, 2, 3)
#: Tolerance on |tau_read - tau_in| for the SEED's surface read, which is
#: exact on a flat torus (spec S1: the read returns tau_in to rounding).
DECLARED_TAU_TOLERANCE = 1e-9
#: The tori's labels and drawing colours (the layout highlight and the traces
#: read from the same table, so a colour and its label cannot disagree).
#:
#: Four entries because `--conjugate-inputs` carries each state on a PAIR of
#: tori: A with its orientation reversal A*, B with B*. A conjugate shares its
#: partner's hue at a lighter value, since it is the same state seen the other
#: way round rather than a third and fourth input.
DECLARED_TORUS_LABELS = ("A", "B")
DECLARED_TORUS_COLOURS = ("#d2691e", "#1f8a70")
#: The same, for a four-torus boundary: each state beside its conjugate, which
#: shares its partner's hue at a lighter value, since it is the same state seen
#: the other way round rather than a third and fourth input.
DECLARED_CONJUGATE_TORUS_LABELS = ("A", "A*", "B", "B*")
DECLARED_CONJUGATE_TORUS_COLOURS = ("#d2691e", "#e8a76b", "#1f8a70", "#6fbfa8")

# the qubit host -- two flat tori on their collar (qubit cobordism spec)
# =====================================================================

def two_qubit_flip_flop(psi, phi):
    """chi of spec S5 for two spin-1/2 under the XY flip-flop.

    `H_int = hbar J (s1+ s2- + s1- s2+)` maps the product `psi (x) phi` to
    `chi = (s- psi)(s+ phi)^T + (s+ psi)(s- phi)^T` in the |0>, |1> bases of
    the two qubits -- rows qubit A, columns qubit B, `s-|0> = |1>`. The
    first-order amplitude to an orthogonal final state is `-i J t chi`; chi
    itself is the two-body target the engine's projective leak scores the
    transfer against (`MultiCobordism.set_two_body_target`), read in the
    tori's period frames where `f_A <-> |0>` and `f_B <-> |1>`.
    """
    import numpy as np

    lowering = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    raising = lowering.T
    psi = np.asarray(psi, dtype=complex).reshape(2)
    phi = np.asarray(phi, dtype=complex).reshape(2)
    return (np.outer(lowering @ psi, raising @ phi)
            + np.outer(raising @ psi, lowering @ phi))


#: The two-qubit gates a run may fit, as 4 x 4 matrices on the pair space
#: (`--operator`). Each is UNITARY, unlike `H_int`, which is Hermitian and
#: has two zero singular values. What a run fits is still the gate's OUTPUT
#: on one input pair: the transfer is 2 x 2, so its flattening is a two-qubit
#: STATE, and a gate's Choi state needs four-dimensional frames on each
#: boundary, which one torus (b_1 = 2, one qubit) cannot supply.
DECLARED_GATES = {
    "swap": [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    "sqrt_swap": [[1, 0, 0, 0], [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                  [0, 0.5 - 0.5j, 0.5 + 0.5j, 0], [0, 0, 0, 1]],
    "iswap": [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
    "cnot": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    # Charge conserving: each commutes with the total magnetisation
    # N = (sz (x) 1 + 1 (x) sz)/2, so it is block diagonal on |00>, the
    # one-excitation pair {|01>, |10>}, and |11>.
    "cz": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    "sqrt_iswap": [[1, 0, 0, 0],
                   [0, 0.7071067811865476, 0.7071067811865476j, 0],
                   [0, 0.7071067811865476j, 0.7071067811865476, 0],
                   [0, 0, 0, 1]],
    "cphase_third": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                     [0, 0, 0, 0.5 + 0.8660254037844386j]],
    # NOT charge conserving: each moves weight between the sectors of N.
    "cnot_reversed": [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
    "ch": [[1, 0, 0, 0], [0, 1, 0, 0],
           [0, 0, 0.7071067811865476, 0.7071067811865476],
           [0, 0, 0.7071067811865476, -0.7071067811865476]],
    "xx": [[0.7071067811865476, 0, 0, -0.7071067811865476j],
           [0, 0.7071067811865476, -0.7071067811865476j, 0],
           [0, -0.7071067811865476j, 0.7071067811865476, 0],
           [-0.7071067811865476j, 0, 0, 0.7071067811865476]],
}
#: The operator a run fits, by name. `flip_flop` is the interaction
#: Hamiltonian of spec S5; the rest are the gates above.
DECLARED_OPERATOR = "flip_flop"
#: Whether the input tori are HELD while the bulk relaxes (`--pin-boundary`,
#: `--no-pin-boundary`).
#:
#: The block residual of spec D2 is a function of tau_hat alone, and tau_hat is
#: a conformal invariant, so reshaping a boundary within its conformal class
#: costs the objective nothing and the relaxation does it: measured, individual
#: boundary edges move 10-28% while tau_hat holds to thirteen digits. A bulk
#: fitted to that reshaped boundary does not work with the input it was given.
#: Pinning removes the freedom rather than pricing it.
#:
#: ON by default (#1022). It used to be off, on the reading that the spec's
#: Do-not list rules pinned regions out of this experiment, and the reshaping
#: was accepted as the price. Once stage 1 could walk its whole move space the
#: price came due: the stronger search commits a different bulk, the
#: relaxation then takes the unheld boundary somewhere else, and the modulus
#: itself moves rather than only the individual edges. Measured on the two-unit
#: qubit drive, the same run twice:
#:
#:     unheld  residuals 0 0 -> 2.787845e-03 5.718170e-06   cells 54 -> 54
#:     held    residuals 0 0 -> 0.000000e+00 0.000000e+00   cells 54 -> 56
#:
#: Note which one grows the bulk. The unheld run spends its whole descent
#: reshaping the boundary and commits no cell; the held one puts that descent
#: into the thing the search is actually over.
#:
#: `--no-pin-boundary` restores the old behaviour for a run that wants the
#: relaxation to fit its boundary rather than hold it -- which is a real use,
#: and why the hold is a default here rather than a rule in the engine.
DECLARED_PIN_BOUNDARY = True

#: Whether every declared state is attached to ONE shared boundary at a time,
#: rather than each being scored against a boundary of its own
#: (`--pin-boundary-state`).
#:
#: A `TwoBodyCase` carries its own squared lengths on the boundary edges, and
#: `twoBodyResidualsPerCaseOn` writes them into the live complex, scores that
#: case, and restores. The boundary is therefore a parameter supplied per case,
#: not a coordinate the optimiser owns -- which is why the engine's
#: `set_two_body_cases` says it "Expects a pinned boundary", and why releasing
#: the geometric pin does nothing at all for a multi-state run. Measured, the
#: same three-unit drive at the same seed:
#:
#:     substituted, pinned  7.826605e-01 -> 3.358045e-01   8.625074e-01 -> 5.547834e-01
#:     substituted, free    7.826605e-01 -> 3.358045e-01   8.625074e-01 -> 5.547834e-01
#:
#: Bit-identical. `writeCaseBoundary` iterates the case's boundary list, so a
#: case whose list is EMPTY is a no-op and is scored on the live shared complex
#: using only its own coefficients and target; the gradient path uses the same
#: write/restore, so the two agree. Building the cases that way attaches every
#: state to the same boundary at one time and leaves that boundary free to
#: relax and to be changed combinatorially. Same run, boundary payload removed:
#:
#:     shared, free         7.826605e-01 -> 3.358045e-01   8.409688e-01 -> 4.571637e-01
#:
#: The first case cannot move: its declared boundary IS the live geometry, so
#: substituting it was always a no-op. The second both starts lower -- it is
#: read on the shared complex rather than on a private torus -- and ends lower.
#:
#: OFF by default: substitution answers "does the operator work for this state
#: in its own attachment context", which is a different and equally real
#: question. This flag answers "does ONE bulk carry the operator for all of
#: them at once", which is the one a bulk operator has to pass. Note that with
#: one shared boundary the block has a single tau_hat while each state declares
#: its own coefficients, so the states genuinely compete; the per-state
#: residuals cannot all reach zero unless the operator really is realisable on
#: one attachment. That competition is the measurement, not a defect.
DECLARED_PIN_BOUNDARY_STATE = False


#: The engine's readings by their command-line names. One table, so a name
#: that parses is a name the engine accepts. Each is named for the SPACE it
#: takes the state from, so a name cannot suggest the wrong one.
#: How many boundary tori each reading needs, and why. One table, enforced by
#: `build_config`, so a combination that cannot produce a number is refused
#: rather than scored as a constant 1.0 -- a constant term carries no gradient,
#: so the flag would silently be a no-op while the run looked like it was
#: optimizing something.
_READOUT_TORI = {
    "transfer": ((2,), "it reads between exactly two frames, so four tori "
                       "would leave two of them out"),
    "bulk": ((), "this driver has no declared Choi frame for the interior "
                  "edges; use the low-level geometric-operator API with an "
                  "explicit frame"),
    "whole": ((2,), "four marked tori provide a rank-4 direct-sum period "
                      "frame, not the tensor-product coordinates of the "
                      "declared two-qubit target"),
    "operator": ((), "its paired frame is a direct sum, while the declared "
                       "two-qubit gate acts on a tensor product; equal 4 by 4 "
                       "dimensions do not identify those spaces"),
}
#: What --tori, --readout, --output-state and --layers mean TOGETHER, printed
#: as the run subcommand's epilog. Written here beside `_READOUT_TORI`, which
#: enforces it, so the documentation and the refusal cannot drift apart.
#:
#: No per-cent signs anywhere in this string: argparse interpolates the epilog
#: and a bare one is read as a conversion.
QUBIT_READOUT_NOTES = """
qubit mode: how --tori, --readout and --output-state fit together

  --tori is how many tori bound the cobordism.

    2   one torus per input state; the collar of spec S3, one layer.
    4   each state on a PAIR, itself and its orientation reversal at
        -conj(tau). The host is two collars joined along a removed
        tetrahedron, which needs three layers, and --layers is raised to
        three on its own.

  Why the count matters: for a compact oriented 3-manifold
  rank(H^1(W) -> H^1(dW)) = b_1(dW)/2. Two tori leave a harmonic space of
  rank 2, four leave rank 4. The latter is a direct sum of torus cohomologies,
  not the tensor product of two qubits; matching dimensions do not identify
  those representations.

  --readout is which space the target is scored against, as a
  comma-separated set that is SUMMED. At the low-level engine API each is the
  same projective leak on the same scale, with 1.0 the convention for a
  geometry that cannot name a state. This driver refuses every configuration
  known to be structurally incapable before it drives, so no runnable flag
  silently contributes that constant. The choice is part of the OBJECTIVE:
  the residual is a term in r_U, so it prices stage-1 moves. Transfer also
  drives stage-2 descent; the period-normalized whole reading has zero length
  direction while the marked topology stays fixed.

    transfer   (Z_A^v)^T A~_1 Z_B, the whole complex's degree-1 operator
               read as the coupling block between the two boundary frames.
               2x2, scored against chi. --tori 2 only: it reads between
               exactly two frames, so four tori would leave two out.

    bulk       refused by this driver. The engine's low-level geometric
               operator reads ker L_1(W - dW) through an EXPLICIT Choi frame
               of d^2 interior edges. This driver has no geometrically
               canonical frame to supply, and choosing one implicitly would
               change the answer.

    whole      ker L_1(W), the boundary INCLUDED, read through the blocks'
               markings. At --tori 2 the harmonic rank is 2, so it needs
               --output-state, a 2-dimensional target. At --tori 4 the
               period coordinates form a direct sum rather than the tensor
               coordinates of chi, so that combination is refused.
               The product collar's two harmonic frames must agree, so this
               reading also refuses --collar-twist swap.

    operator   refused by this driver. Stacking a torus and its conjugate
               gives a rank-4 DIRECT-SUM frame, while a two-qubit gate acts
               on a rank-4 TENSOR PRODUCT. The matching dimensions do not
               supply the missing tensor-product identification. A future
               tensor/Fock readout must name that representation explicitly.

  --output-state names the state the whole Laplacian's harmonic is meant to
  be, as a modulus tau; the target is psi(tau) = (1, tau) normalized. Read
  by every --readout set containing whole, including transfer,whole, and
  refused when whole is absent.

  Writing a state. Every state here is a modulus tau, and the state it names
  is psi(tau) = (1, tau) normalized -- so tau = psi_1 / psi_0. The table is
  the usual qubit states in that coordinate:

    state              tau               input? output? on the command line
    |0>                0                 no     yes     --output-state 0
    |1>                infinity          no     no      (not writable at all)
    |+>                1                 no     yes     --output-state 1
    |->                -1                no     yes     --output-state=-1
    |+i>               1j                YES    yes     --tau-a 1j
    |-i>               -1j               no     yes     --output-state=-1j
    (sqrt3/2, 1/2)     0.57735           no     yes     --output-state 0.57735
    (3/5, 4i/5)        1.33333j          YES    yes     --tau-a 1.33333j
    near |0>           0.1j              YES    yes     --tau-a 0.1j
    near |1>           10j               YES    yes     --tau-a 10j
    equal, phase pi/4  0.70711+0.70711j  YES    yes     --tau-a 0.70711+0.70711j
    tilted right       0.5+1j            YES    yes     --tau-a 0.5+1j
    tilted left        -0.5+1j           YES    yes     --tau-b=-0.5+1j
    tall               2.5j              YES    yes     --tau-a 2.5j

  A LEADING MINUS needs the equals form. argparse reads a bare -0.2+0.8j as an
  option, so --tau-b=-0.2+0.8j works and --tau-b -0.2+0.8j does not. The same
  goes for --tau-a, --state and --output-state.

  Two different constraints are at work, and they are not the same one.

    An INPUT state -- --tau-a, --tau-b, --state -- is a TORUS, and a torus
    modulus must lie in the upper half-plane. flat_torus refuses Im tau <= 0
    by name. So an input can only be a state with Im(psi_1 / psi_0) > 0:
    half the Bloch sphere. |0>, |1>, |+> and |-> are NOT expressible as
    inputs, and neither is any other state whose amplitude ratio is real.

    An --output-state is a state and nothing else, so any finite tau does.
    Only |1> is out of reach there, because (1, tau) is never (0, 1) for
    finite tau; a large |tau| approaches it, which is what "near |1>" is.

  Which metric each reading uses:

    transfer   the chain-level WHITNEY PENCIL, at degree 1. The frames are
               derived live from the blocks' own covariant pencil zero mode
               and the dressed Whitney mass matrix M_1^U, so lengths and
               link phases both enter. A node on the diagonal-weight metric
               is refused by name rather than read through a metric it did
               not ask for.

    bulk       when called through the low-level API with an explicit frame,
               the live SIGNED HODGE WEIGHTS,
               W_1^-1 d_1^T W_0 d_1 + d_2 W_2^-1 d_2^T W_1, whose right
               kernel is taken by SVD because the Lorentzian operator is
               generally non-normal. The combinatorial unit-weight operator
               d_1^T d_1 + d_2 d_2^T is available in the engine and is NOT
               used here: it is topology-only, so it would score the same
               for every geometry with the same cells and the term would be
               a constant under stage 2.

    whole      the chain-level WHITNEY PENCIL, at degree 1, same guard as
               transfer. Its harmonic band and period matrix move with the
               lengths, but normalization into the live period frame cancels
               that basis motion at fixed marked topology. It can therefore
               price topology-changing stage-1 moves, while its correct
               stage-2 length gradient is zero.

    operator   the low-level paired transfer uses the chain-level WHITNEY
               PENCIL, but its frame is a direct sum and is not wired here as
               a two-qubit tensor-product operator.

  Transfer is metric-sensitive. Whole is period-frame intrinsic: at fixed
  marked topology its value is unchanged by length motion, but a
  topology-changing stage-1 candidate may change the available harmonic space
  or period map. Other metric-agnostic readings in the engine include the
  combinatorial unit-weight bulk operator above, and anything read at degree
  0, where the representation is the U(1) connection Laplacian and the metric
  source does not enter at all -- measured in #936, where the Whitney and
  diagonal-weight records came out bit-identical.

  What that leaves:

    --readout             --tori 2                --tori 4
    transfer              yes                     refused
    bulk                  refused                refused
    whole                 needs --output-state    refused
    operator              refused                refused

  Accepted rows compose: transfer,whole is accepted at --tori 2 when an
  --output-state is supplied, and its aggregate residual is the sum of the
  transfer and whole components. A selection containing a refused row or an
  incompatible torus count is refused when the config is built, by name and
  with the reason. A reading that cannot produce a number would otherwise
  score a constant 1.0 forever, and a constant term has no gradient -- the
  flag would be a silent no-op while the run looked like it was optimizing
  something.

  The driver therefore has no four-torus objective. The value 4 is retained
  so existing commands fail with the scientific reason; construct the paired
  host through the low-level engine API for a geometric diagnostic.

  Every accepted reading is case-specific. `transfer` uses the gate's image
  of the declared input pair; `whole` uses the explicit --output-state.
"""


_READOUT_MODES = {
    "transfer": MC.ReadoutMode.TRANSFER,
    "bulk": MC.ReadoutMode.BULK,
    "whole": MC.ReadoutMode.WHOLE,
    "operator": MC.ReadoutMode.OPERATOR,
}


def _readout_names(readout):
    """The readings a `--readout` value names, in the order written.

    A comma-separated SET rather than one choice: a run can be scored under
    more than one reading, and summing them needs no special name for each
    pairing. Refused rather than silently dropped when a name is unknown or
    the set is empty, since a two-body term scored against nothing is not a
    term.
    """
    names = [name.strip() for name in str(readout).split(",") if name.strip()]
    if not names:
        raise ValueError("readout names no reading: expected a comma-separated "
                         "set from %s" % ", ".join(sorted(_READOUT_MODES)))
    for name in names:
        if name not in _READOUT_MODES:
            raise ValueError("unknown readout %r: expected a comma-separated "
                             "set from %s"
                             % (name, ", ".join(sorted(_READOUT_MODES))))
    return list(dict.fromkeys(names))


def _validate_readout_host(names, tori):
    """Refuse a reading whose declared target has no matching host space."""
    for name in names:
        allowed, why = _READOUT_TORI[name]
        if not allowed:
            raise ValueError("--readout %s is unavailable: %s" % (name, why))
        if tori not in allowed:
            raise ValueError(
                "--readout %s needs --tori %s, not %d: %s"
                % (name, " or ".join(str(n) for n in allowed), tori, why))


def gate_image(name, psi, phi):
    """A gate's image of the product state, as the 2 x 2 the transfer meets.

    `G (psi (x) phi)` is a vector of C^4 and reshapes to 2 x 2. It is
    bilinear in the two states because the product is and G is linear, which
    is what lets it stand where `chi` of spec S5 stands: same shape, same
    projective scoring, same frames.
    """
    import numpy as np

    gate = np.asarray(DECLARED_GATES[name], dtype=complex)
    product = np.kron(np.asarray(psi, dtype=complex).reshape(2),
                      np.asarray(phi, dtype=complex).reshape(2))
    return {"coupling": float("nan"), "time": float("nan"), "Jt": float("nan"),
            "psi": np.asarray(psi, dtype=complex).reshape(2),
            "phi": np.asarray(phi, dtype=complex).reshape(2),
            "chi": (gate @ product).reshape(2, 2),
            # The gate beside its image of this input pair, for a complete
            # algebra record even though the driver scores the image itself.
            "gate": gate,
            "product_state": product.reshape(2, 2),
            "first_order_amplitudes": (gate @ product).reshape(2, 2),
            "exact_amplitudes": (gate @ product).reshape(2, 2)}


def flip_flop_evolution(psi, phi, coupling, time):
    """The exact two-qubit evolution at `J t`, recorded next to first order.

    `[H_int, N] = 0` for the total magnetization `N`, so the propagator is
    block diagonal on the sectors of `N`: |00> and |11> (size 1, untouched)
    and {|01>, |10>} (size 2, where `H_int` is `J sigma_x` and the
    exponential is `cos(Jt) - i sin(Jt) sigma_x`). The exact amplitudes are
    kept beside `-i J t chi` so a run's transfer can be compared with the
    algebra at the declared `J t`, as the spin-3/2 experiment does.
    """
    import numpy as np

    psi = np.asarray(psi, dtype=complex).reshape(2)
    phi = np.asarray(phi, dtype=complex).reshape(2)
    angle = float(coupling) * float(time)
    propagator = np.eye(4, dtype=complex)
    propagator[1, 1] = propagator[2, 2] = math.cos(angle)
    propagator[1, 2] = propagator[2, 1] = -1j * math.sin(angle)
    product = np.kron(psi, phi)
    chi = two_qubit_flip_flop(psi, phi)
    return {"coupling": float(coupling), "time": float(time), "Jt": angle,
            "psi": psi, "phi": phi, "chi": chi,
            # The propagator used for the exact ordinary state image.
            "gate": propagator,
            "product_state": product.reshape(2, 2),
            "first_order_amplitudes": -1j * angle * chi,
            "exact_amplitudes": (propagator @ product).reshape(2, 2)}


def _torus_fiber(torus, ids):
    """The torus's state as a fiber on its edges in host ids (spec S1, S2).

    Degree 1, one cell per torus edge in the torus's own edge order carried
    through the collar's id map, the holomorphic form as the single image
    column, and the HARMONIC contour of the torus's own pencil on the fiber
    (spec R7: the zero mode, never the band above it, which is the engine's
    default contour). Exactly the fiber the T2 tests attach.
    """
    import numpy as np

    fiber = cob.BoundaryFiber()
    fiber.degree = 1
    fiber.cells = [sorted((ids[int(i)], ids[int(j)])) for i, j in torus.edges()]
    fiber.images = np.asarray(torus.holomorphic_form()).reshape(-1, 1)
    fiber.contour = cob.PencilLayer.harmonic_contour(
        cob.PencilLayer.assemble([torus.spacetime()]), 1)
    return fiber


def _dispose_interior(host, tori, ids, disposition, seed):
    """Give the collar's INTERIOR edges a causal character.

    `seed_collar` writes each torus's own lengths onto its edges and wires
    every other edge to 1.0, so a qubit host is all-spacelike in its bulk with
    no way to say otherwise. This gives those edges the same closed vocabulary
    the neutral driver uses (`EdgeDisposition`).

    The tori's OWN edges are left exactly as the surfaces built them. They
    carry the declared input moduli, so a disposition written over them would
    be silently changing the input states rather than the bulk they are
    embedded in.
    """
    if disposition == ea.EdgeDisposition.SPACELIKE:
        return  # what the collar already wires, so nothing to write
    boundary = set()
    for index, torus in enumerate(tori):
        mapping = ids[index]
        for edge in torus.spacetime().getEdgeList().toVector():
            if edge is None or edge.getSource() is None or edge.getTarget() is None:
                continue
            u = mapping[int(edge.getSource().getId())]
            v = mapping[int(edge.getTarget().getId())]
            boundary.add((min(u, v), max(u, v)))
    interior = []
    for edge in host.getEdgeList().toVector():
        if edge is None or edge.getSource() is None or edge.getTarget() is None:
            continue
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        if (min(u, v), max(u, v)) not in boundary:
            interior.append(edge)
    ea._seed_lengths(host, disposition, seed, edges=interior)


#: How four tori are joined into one host: along a removed tetrahedron (a
#: sphere, the direct sum) or by a tube between the far tori (a 1-handle,
#: whose far boundary is one genus-2 surface).
DECLARED_JOINS = ("sphere", "tube")
DECLARED_JOIN = "sphere"
#: The tube's shape: prism layers, the height of one layer, and the scale of
#: the interior rings about their centroid -- the knobs of the neck.
DECLARED_TUBE_LAYERS = 2
DECLARED_TUBE_LENGTH = 1.0
DECLARED_TUBE_WAIST = 1.0


def _tube_face(grid):
    """The attachment face of the tube on a flat torus: the triangle
    `{(1, 1), (2, 1), (2, 2)}` in `SimplicialQubit.flat_torus`'s vertex ids
    (`vid(i, j) = i * grid + j`), whose three edges lie on neither marking
    cycle (row 0 is A, column 0 is B)."""
    n = int(grid)
    if n < 3:
        raise ValueError("the tube needs a grid of at least 3")
    vid = lambda i, j: i * n + j  # noqa: E731
    return [vid(1, 1), vid(2, 1), vid(2, 2)]
#: The layered flip passes on the collar's far torus. Each pass flips every
#: edge of one direction of the grid (row `(1,0)`, column `(0,1)` or
#: diagonal `(1,1)`): the tetrahedron on the edge's two boundary triangles is
#: attached, the edge becomes interior and the quadrilateral's other diagonal
#: the new boundary edge. The flipped triangulation is the grid again in a
#: sheared basis, so it is relabelled by the lattice map `L` (new grid
#: coordinates -> old), and the monodromy the relabelled marking reads is
#: `L^T`: the Dehn twists `T_B`, `T_A` and the quarter turn `S`. Offsets are
#: grid steps from the flipped edge's first vertex `v`: the tetrahedron, the
#: flipped edge `(v, v + edge)`, the new edge `(v + new[0], v + new[1])`.
DECLARED_FLIP_PASSES = {
    "b": {"edge": (1, 0), "tetrahedron": ((0, -1), (0, 0), (1, 0), (1, 1)), "new": ((0, -1), (1, 1)),
          "lattice": ((1, 0), (1, 1)), "name": "T_B: flip every row edge, relabel (x, y) -> (x, y + x)"},
    "a": {"edge": (0, 1), "tetrahedron": ((-1, 0), (0, 0), (0, 1), (1, 1)), "new": ((-1, 0), (1, 1)),
          "lattice": ((1, 1), (0, 1)), "name": "T_A: flip every column edge, relabel (x, y) -> (x + y, y)"},
    "s": {"edge": (1, 1), "tetrahedron": ((0, 0), (1, 0), (1, 1), (0, 1)), "new": ((1, 0), (0, 1)),
          "lattice": ((0, 1), (-1, 0)), "name": "S: flip every diagonal, relabel (x, y) -> (y, -x)"},
}
#: The flipped edge, now interior, is shortened by this factor so the layered
#: tetrahedron has volume; the far torus keeps its flat lengths on every
#: boundary edge, the new ones included.
DECLARED_FLIP_FACTOR = 0.7


def _far_flip_passes(config):
    """The pass sequence named by `far_flips` (`"b,a"` applies `b` first)."""
    raw = config.get("far_flips")
    if not raw:
        return []
    names = [name.strip() for name in str(raw).split(",") if name.strip()]
    for name in names:
        if name not in DECLARED_FLIP_PASSES:
            raise ValueError("unknown flip pass %r; expected a sequence of %s" % (name, ", ".join(DECLARED_FLIP_PASSES)))
    return names


def _flip_lattice_map(config):
    """`L`, the composed lattice map of the passes (new grid coordinates ->
    old), and the monodromy it predicts on the far marking, `M = L^T` times
    the collar twist's induced map on the right (the twist relabels first)."""
    import numpy as np
    lattice = np.eye(2, dtype=int)
    for name in _far_flip_passes(config):
        lattice = lattice @ np.array(DECLARED_FLIP_PASSES[name]["lattice"], dtype=int)
    twist = np.array(_TWIST_MATRICES[str(config.get("collar_twist", DECLARED_COLLAR_TWIST))], dtype=int)
    return lattice, lattice.T @ twist


def _layer_far_flips(host, far_ids, grid, tau_far, passes, factor):
    """Attach the passes' tetrahedra to the far torus and rebuild the host.

    Returns the new host, the far torus's id map composed with the
    relabelling (surface vertex `(x, y)` -> the host vertex at old grid
    position `L (x, y)`), and `L`. Every boundary edge of the far torus,
    the new ones included, carries its flat length in the torus's lattice
    (`e_1 = 1/n`, `e_2 = tau/n`, an edge of old displacement `d` has length
    `|d_0 e_1 + d_1 e_2|` with `d` taken unreduced, the diagonal of the
    flipped quadrilateral); a flipped edge, now interior, is shortened by
    `factor`. The passes of one sequence act on the current relabelled
    grid, and the flips within a pass are on disjoint triangle pairs.
    """
    import numpy as np
    n = int(grid)
    vid = lambda x, y: (x % n) * n + (y % n)  # noqa: E731
    cells = [tuple(sorted(int(v.getId()) for v in cell.getVertices())) for cell in host.getTopSimplices()]
    lengths = {}
    for edge in host.getEdgeList().toVector():
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        lengths[(min(u, v), max(u, v))] = complex(edge.getLength())
    e1, e2 = 1.0 / n, complex(tau_far) / n
    lattice = np.eye(2, dtype=int)

    def host_of(x, y):
        old = lattice @ np.array([x, y])
        return int(far_ids[vid(int(old[0]), int(old[1]))])

    def key(a, b):
        return (min(a, b), max(a, b))

    for name in passes:
        spec = DECLARED_FLIP_PASSES[name]
        new_cells, new_lengths, flipped = [], {}, []
        for x in range(n):
            for y in range(n):
                tet = tuple(sorted(host_of(x + dx, y + dy) for dx, dy in spec["tetrahedron"]))
                if len(set(tet)) != 4:
                    raise ValueError("flip pass %r: the tetrahedron at (%d, %d) repeats a vertex; the grid is too small" % (name, x, y))
                new_cells.append(tet)
                (ax, ay), (bx, by) = spec["new"]
                p, q = host_of(x + ax, y + ay), host_of(x + bx, y + by)
                if key(p, q) in lengths:
                    raise ValueError("flip pass %r: the new edge at (%d, %d) is already an edge" % (name, x, y))
                displacement = lattice @ np.array([bx - ax, by - ay])
                new_lengths[key(p, q)] = complex(abs(displacement[0] * e1 + displacement[1] * e2))
                flipped.append(key(host_of(x, y), host_of(x + spec["edge"][0], y + spec["edge"][1])))
        cells.extend(new_cells)
        lengths.update(new_lengths)
        for edge in flipped:
            lengths[edge] = lengths[edge] * float(factor)
        lattice = lattice @ np.array(spec["lattice"], dtype=int)
    ok, why = cob.ChainComplex.dualComplexIsValid(cells, 3)
    if not ok:
        raise ValueError("the layered flips are not a manifold-with-boundary: %s" % why)
    rebuilt = T.spacetime.Spacetime.fromCells(3, cells, 1.0, 0.0)
    for edge in rebuilt.getEdgeList().toVector():
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        edge.setLength(lengths[key(u, v)])
        edge.setPhase(0.0)
    relabelled = {vid(x, y): host_of(x, y) for x in range(n) for y in range(n)}
    return rebuilt, relabelled, lattice


def _collar_twist(config, grid):
    """The permutation of the far surface's base indices, or none.

    A flat torus's base index is its vertex's rank in ascending id order,
    which on the `grid x grid` lattice is `i * grid + j`. The swap is
    `(i, j) -> (j, i)`, the only mapping class that is both a simplicial
    automorphism of this triangulation and projectively non-trivial: the
    identity and the negation act trivially on a modulus, and a Dehn twist is
    not simplicial here.
    """
    name = str(config.get("collar_twist", DECLARED_COLLAR_TWIST))
    if name == "none":
        return []
    if name != "swap":
        raise ValueError("unknown --collar-twist %r; expected 'none' or 'swap'" % (name,))
    n = int(grid)
    return [(k % n) * n + (k // n) for k in range(n * n)]


def _host_marking(torus, ids):
    """The torus's marking as cycles of directed host steps `(u, v)`.

    `MultiCobordism.monodromy`'s convention: a step contributes `+h(u, v)`
    when `u < v` and `-h(u, v)` otherwise, so a torus step (edge index, sign)
    becomes the edge's `(i, j)` as `(i -> j)` for +1 and `(j -> i)` for -1,
    mapped through the id map.
    """
    edges = torus.edges()

    def cycle(steps):
        out = []
        for e, sign in steps:
            i, j = edges[int(e)]
            u, v = ids[int(i)], ids[int(j)]
            out.append((u, v) if sign > 0 else (v, u))
        return out

    return [cycle(torus.cycle_A()), cycle(torus.cycle_B())]


def _quiet(build):
    """Run a construction with its Python warnings silenced.

    `SimplicialQubit`'s constructor reports the spec's diagnostics --
    Delaunay violations, the section-13 condition numbers, the section-9
    branch note -- as Python warnings. A frame records them as channel
    values (`SimplicialQubit.warnings`) instead, so they are silenced here
    rather than printed once per frame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def _assert_seed_reads_the_torus(node, index, torus):
    """The engine's read of the seed surface must be the input torus.

    `block_qubit` reads the block's own triangles with the host's lengths
    over the block's marking, in the orientation with A.B = +1 -- the
    orientation the flat torus was built in (spec section 12), so on the
    collar seed it must return tau_in to rounding: that is what the block
    residual of D2 is zero for. Asserted by name once, on the seed, so a
    marking carried wrongly through the id map fails here rather than as a
    residual that never closes.
    """
    read = _quiet(lambda: node.block_qubit(index))
    if read.intersection_number() < 0:
        raise RuntimeError("the seed surface of block %d reads with A.B = %g; "
                           "the marking does not fix the orientation"
                           % (index, read.intersection_number()))
    if abs(complex(read.tau()) - complex(torus.tau())) > DECLARED_TAU_TOLERANCE:
        raise RuntimeError(
            "the seed surface of block %d reads tau = %s against tau_in = %s"
            % (index, read.tau(), torus.tau()))


class QubitInputs:
    """What the qubit factory hands every frame read: the two input tori and
    how they sit in the host.

    Nothing here is a measurement. These are the INPUTS of spec S1 and the
    id-mapped data the read-outs of spec S6 need: the tori and their moduli,
    the id maps, the attached fiber cells (host ids, in the torus's edge
    order), the markings as host steps, the orientation flag of each surface
    read, the two-body target chi with the algebra it comes from, and the
    objective the node descends. `to_json` puts them in the run document
    once, next to the config; a frame never repeats them.
    """

    __slots__ = ("tori", "tau_in", "coefficients_in", "vertex_ids", "cells",
                 "markings", "algebra", "weight", "regge", "grid", "layers",
                 "objective_name", "objective_terms", "torus_warnings", "seed")

    def __init__(self, tori, tau_in, vertex_ids, cells, markings,
                 algebra, weight, regge, grid, layers, objective_name,
                 objective_terms, torus_warnings, seed):
        self.tori = list(tori)
        self.tau_in = [complex(tau) for tau in tau_in]
        #: The input coefficients (1, tau_in) of each torus in its own frame
        #: (spec section 2, "State at a block"): what the block's marking
        #: carries and the block residual of D2 holds the torus's own
        #: holomorphic form at.
        self.coefficients_in = [[1.0 + 0j, complex(tau)] for tau in tau_in]
        self.vertex_ids = [dict(mapping) for mapping in vertex_ids]
        self.cells = [[list(cell) for cell in block] for block in cells]
        self.markings = [[list(cycle) for cycle in marking]
                         for marking in markings]
        self.algebra = algebra
        self.weight = float(weight)
        self.regge = bool(regge)
        self.grid = int(grid)
        self.layers = int(layers)
        self.objective_name = str(objective_name)
        self.objective_terms = [str(name) for name in objective_terms]
        self.torus_warnings = [[str(w) for w in notes] for notes in torus_warnings]
        self.seed = dict(seed)

    @property
    def labels(self):
        if len(self.tori) > len(DECLARED_TORUS_LABELS):
            return DECLARED_CONJUGATE_TORUS_LABELS[:len(self.tori)]
        return DECLARED_TORUS_LABELS[:len(self.tori)]

    @property
    def colours(self):
        if len(self.tori) > len(DECLARED_TORUS_COLOURS):
            return DECLARED_CONJUGATE_TORUS_COLOURS[:len(self.tori)]
        return DECLARED_TORUS_COLOURS[:len(self.tori)]

    def highlight(self):
        """The tori's edges for the layout panel: (label, colour, id pairs)."""
        return [(label, colour, {tuple(cell) for cell in cells})
                for label, colour, cells in zip(self.labels,
                                                self.colours,
                                                self.cells)]

    def to_json(self):
        import numpy as np

        def matrix(value):
            return [[complex(z) for z in row] for row in np.asarray(value)]

        # Written by SHAPE rather than by a list of names: the gate itself is
        # recorded beside its image (#1050) and a fifth name to remember is a
        # fifth chance to forget one.
        def entry(key, value):
            if isinstance(value, str):
                return str(value)
            if key in ("coupling", "time", "Jt"):
                return float(value)
            array = np.asarray(value)
            if array.ndim >= 2:
                return matrix(array)
            return [complex(z) for z in array]

        algebra = {key: entry(key, value)
                   for key, value in self.algebra.items()}
        return _json_safe({
            "labels": list(self.labels),
            "tau_in": self.tau_in,
            "coefficients_in": self.coefficients_in,
            "bloch_in": [[float(x) for x in torus.bloch()] for torus in self.tori],
            # Keep the historical (misnamed) vertex-count field for readers of
            # existing run documents; the unambiguous fields are additive.
            "grid": [len(torus.vertices()) for torus in self.tori],
            "grid_size": self.grid,
            "vertices_per_torus": [len(torus.vertices()) for torus in self.tori],
            "layers": self.layers,
            "vertex_ids": self.vertex_ids,
            "cells": self.cells,
            "markings": [[[list(step) for step in cycle] for cycle in marking]
                         for marking in self.markings],
            "input_weight": self.weight,
            "regge": self.regge,
            "objective": self.objective_name,
            "objective_terms": self.objective_terms,
            "torus_warnings": self.torus_warnings,
            "seed": self.seed,
            "algebra": algebra,
        })


def _as_state_pairs(text):
    """A colon-delimited list of moduli, read two at a time.

    `"a:b:c:d"` is the input pairs (a, b) and (c, d) -- one flag saying what
    several repeated ones used to. An odd count is refused rather than
    silently dropping the last modulus, since a pair with one state is not a
    two-body input.
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        items = [str(part) for part in text]
    else:
        items = [part.strip() for part in str(text).split(":") if part.strip()]
    if not items:
        return []
    # Two shapes, told apart by the comma. Each item may itself be a `a,b`
    # pair -- the shape a repeated --state used to take, and the one every
    # recorded run is written in -- or the items may be bare moduli read two
    # at a time. Mixing them is refused rather than guessed at.
    commas = [("," in item) for item in items]
    if any(commas):
        if not all(commas):
            raise ValueError(
                "--state mixes `a,b` pairs with bare moduli: write either "
                "0.5+0.9j,0.1+1.3j:1j,1j or 0.5+0.9j:0.1+1.3j:1j:1j")
        pairs = []
        for item in items:
            halves = [half.strip() for half in item.split(",")]
            if len(halves) != 2:
                raise ValueError("a state pair is two moduli separated by a "
                                 "comma, like 0.5+0.9j,0.1+1.3j -- got %r" % (item,))
            pairs.append((_as_complex(halves[0]), _as_complex(halves[1])))
        return pairs
    parts = items
    if len(parts) % 2 != 0:
        raise ValueError(
            "--state is a colon-delimited list of moduli read in PAIRS, like "
            "0.5+0.9j:0.1+1.3j:1j:1j -- got %d of them, which leaves one "
            "unpaired" % len(parts))
    try:
        moduli = [_as_complex(part) for part in parts]
    except ValueError as error:
        raise ValueError("--state is a colon-delimited list of moduli: %s"
                         % error) from None
    return [(moduli[n], moduli[n + 1]) for n in range(0, len(moduli), 2)]


def _two_body_case(pair, tori, ids, grid, operator, config):
    """One case as boundary, chi, decomposition, paired target and inputs.

    The boundary is the two tori at this pair's moduli, mapped onto the host by
    the seeding's vertex correspondence. The bulk is not listed: it is what the
    fit is solving for and is shared by every case. The paired direct-sum
    target is absent because the driver supplies no tensor/direct-sum
    identification; the final vector is the case's coefficients in
    input-block/cycle order.
    """
    import numpy as np

    # At four tori each state of the pair is carried with its orientation
    # reversal at -conj(tau), in the block order the seeding used:
    # (tau_a, -conj(tau_a), tau_b, -conj(tau_b)).
    moduli = list(pair)
    if len(ids) == 4:
        moduli = [tau for tau in pair for tau in (tau, -complex(tau).conjugate())]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at_pair = [obs.SimplicialQubit.flat_torus(complex(tau), grid, grid)
                   for tau in moduli]
    # An EMPTY boundary is the whole of `--pin-boundary-state`. The engine's
    # `writeCaseBoundary` iterates this list to write the case's lengths into
    # the live complex before scoring and restores them after, so an empty one
    # is a no-op: the case is then read on the shared complex, against its own
    # coefficients and target, alongside every other state rather than in place
    # of them. The gradient path iterates the same list, so value and gradient
    # agree without either being told which mode is in force.
    boundary = []
    if not config.get("pin_boundary_state", DECLARED_PIN_BOUNDARY_STATE):
        for index, torus in enumerate(at_pair):
            mapping = ids[index]
            for (i, j), length in zip(torus.edges(), torus.lengths()):
                source, target = mapping[int(i)], mapping[int(j)]
                boundary.append((source, target, complex(length) ** 2))
    # The operator's image of THIS case's two states is the case's output.
    # Derived, never declared: a run that names one output state cannot mean it
    # for several different inputs.
    forward = [np.asarray(at_pair[0].state()),
               np.asarray(at_pair[2 if len(ids) == 4 else 1].state())]
    if operator == "flip_flop":
        algebra = flip_flop_evolution(forward[0], forward[1],
                                      config["coupling"], config["time"])
    else:
        algebra = gate_image(operator, forward[0], forward[1])
    coefficients = [coefficient for tau in moduli
                    for coefficient in (1.0 + 0j, complex(tau))]
    # The fourth slot is retained for the engine tuple API. This driver does
    # not identify its paired direct-sum frame with a tensor two-state space.
    return (boundary, algebra["chi"], True, None, coefficients)


def seed_qubit_host(config):
    """The qubit host as seeded, and nothing wired on it: the tori, their
    input moduli, the collar seed, the id maps from each torus to the host,
    and each torus's marking as host steps.

    This is the first half of `build_qubit_node` -- everything a node is
    built FROM -- and all a read of the fixed complex needs. No node, no
    objective, no readout is created, so it seeds every host `--tori` names,
    including the four-torus direct sum that every declared readout refuses.
    """
    tau_in = [complex(*config["tau_a"]), complex(*config["tau_b"])]
    grid = int(config["grid"])
    with warnings.catch_warnings():
        # The tori's construction notes are recorded from `warnings()` in
        # `QubitInputs`, not printed.
        warnings.simplefilter("ignore")
        # With --conjugate-inputs each state is carried by a PAIR of tori,
        # itself and its orientation reversal at -conj(tau). Four boundary
        # tori give b_1(dW) = 8, so rank(H^1(W) -> H^1(dW)) = 4 -- the
        # dimension a 4-dimensional target needs, against the 2 that two tori
        # leave. -conj(tau) rather than conj(tau) keeps the modulus in the
        # upper half-plane, where a torus modulus has to live, and gives the
        # holomorphic form the conjugate periods.
        if int(config.get("tori", DECLARED_TORI)) == 4:
            tau_in = [tau for tau in tau_in
                      for tau in (tau, -tau.conjugate())]
        # The BASE pair, before the conjugates: what a case is written in, so
        # `_two_body_case` expands it once rather than twice.
        tori = [obs.SimplicialQubit.flat_torus(tau, grid, grid)
                for tau in tau_in]
    join = str(config.get("join", DECLARED_JOIN))
    if join not in DECLARED_JOINS:
        raise ValueError("unknown join %r; expected one of %s" % (join, ", ".join(DECLARED_JOINS)))
    if int(config.get("tori", DECLARED_TORI)) == 4 and join == "tube":
        # Two collars joined by a TUBE between their far tori: the far
        # boundary is one genus-2 surface, the connected sum of the two far
        # tori through the tube, and the near tori are untouched. The tube
        # joins two components and adds no loop: b_1 = 2 + 2 = 4 with
        # nothing invisible to the boundary. The attachment face is the
        # triangle at grid position (1, 1) of each far torus, which shares
        # no edge with the marking cycles (row 0 and column 0).
        tube = cob.TubeSpec()
        tube.layers = int(config.get("tube_layers", DECLARED_TUBE_LAYERS))
        tube.length = float(config.get("tube_length", DECLARED_TUBE_LENGTH))
        tube.waist = float(config.get("tube_waist", DECLARED_TUBE_WAIST))
        tube.face_a = _tube_face(grid)
        tube.face_b = _tube_face(grid)
        seed = MC.seed_tubed_collars([torus.spacetime() for torus in tori],
                                     int(config["layers"]), _collar_twist(config, grid), tube)
    elif int(config.get("tori", DECLARED_TORI)) == 4:
        # Two collars joined along a removed tetrahedron. Gluing along a sphere
        # is a connected sum, which adds no first homology, so b_1 = 2 + 2 = 4
        # with nothing dying on the boundary. Needs three layers: a prism cell
        # spans two adjacent layers, so the all-interior cell the join removes
        # exists only with two interior layers.
        seed = MC.seed_joined_collars([torus.spacetime() for torus in tori],
                                      int(config["layers"]),
                                      _collar_twist(config, grid))
    else:
        seed = MC.seed_collar(tori[0].spacetime(), tori[1].spacetime(),
                              config["layers"], _collar_twist(config, grid))
    ids = [{int(k): int(v) for k, v in mapping.items()}
           for mapping in seed.vertex_ids]
    passes = _far_flip_passes(config)
    if passes:
        import types
        if len(tori) != 2:
            raise ValueError("--far-flips layers flips on the collar's far torus and needs --tori 2")
        host, ids[1], _ = _layer_far_flips(seed.host, ids[1], grid, tau_in[1], passes,
                                           float(config.get("flip_factor", DECLARED_FLIP_FACTOR)))
        seed = types.SimpleNamespace(host=host, vertex_ids=ids)
    _dispose_interior(seed.host, tori, ids,
                      config.get("interior_disposition", DECLARED_INTERIOR_DISPOSITION),
                      int(config["seed"]))
    markings = [_host_marking(torus, ids[index])
                for index, torus in enumerate(tori)]
    return tori, tau_in, seed, ids, markings


def build_qubit_node(config):
    """The qubit node: two flat tori on their collar, seeded as the input
    blocks with their state fibers, markings and the two-body target.

    Step by step the setup the T1-T3 tests measured under (spec S1-S3, S5),
    with D2/D3 as revised: the tori `SimplicialQubit.flat_torus(tau, n, n)`;
    the collar `MultiCobordism.seed_collar` (or two joined collars for four
    tori), with incompatible readout/host combinations refused by name;
    the node with the degree-1 register on the complex locus (the tori carry
    complex lengths and pure-gauge phases), the Regge term per `regge` and
    the Whitney pencil as its metric source; each torus's vertex set one
    input block (`seed_inputs`); each torus's holomorphic form attached as a
    degree-1 fiber on its edges with the harmonic contour
    (`attach_input_fiber`); each torus's marking with its input coefficients
    (1, tau_in) on its block (`set_input_marking`), from which the engine
    derives the block's live frame at every read; chi of spec S5 as the
    Choi-decomposed two-body target (`set_two_body_target`), 2 x 2 in the
    ordinary transfer frames; the block residuals --
    the leak of (1, tau_in) in the zero mode of each block's OWN Laplacian --
    in r_U at the input weight. The shared drive holds the input regions by
    default, unless `--no-pin-boundary` is selected. No objective is injected:
    the node's default objective is what T2 and T3 measured under, and
    `QubitInputs.objective_name` records it.

    Returns the node and its `QubitInputs`.
    """
    import numpy as np

    _validate_readout_host(
        _readout_names(config.get("readout", DECLARED_READOUT)),
        int(config.get("tori", DECLARED_TORI)))
    tori, tau_in, seed, ids, markings = seed_qubit_host(config)
    grid = int(config["grid"])
    node = MC(seed.host, [[1.0 + 0j]] * len(tori), [],
              degrees=list(config["register_degrees"]), seed=config["seed"],
              einstein_hilbert=bool(config["regge"]),
              real_squared_lengths_only=False,
              metric_source=cob.HodgeMetricSource.WhitneyPencil)
    node.seed_inputs([sorted(mapping.values()) for mapping in ids])
    node.use_fiber_residuals(True)
    node.set_input_residual_weight(config["input_weight"])
    node.score_whole_complex_leak(bool(config.get("score_leak",
                                                  DECLARED_SCORE_LEAK)))
    cells = []
    for index, torus in enumerate(tori):
        fiber = _torus_fiber(torus, ids[index])
        node.attach_input_fiber(index, fiber, fiber.cells)
        cells.append([list(cell) for cell in node.inputs[index].fiber.cells])
    for index, torus in enumerate(tori):
        node.set_input_marking(index, markings[index],
                               [1.0 + 0j, complex(tau_in[index])])
    operator = config.get("operator", DECLARED_OPERATOR)
    if operator == "flip_flop":
        second = 2 if len(tori) == 4 else 1
        algebra = flip_flop_evolution(np.asarray(tori[0].state()),
                                      np.asarray(tori[second].state()),
                                      config["coupling"], config["time"])
    else:
        second = 2 if len(tori) == 4 else 1
        algebra = gate_image(operator, np.asarray(tori[0].state()),
                             np.asarray(tori[second].state()))
    algebra["operator"] = operator
    node.set_two_body_target(algebra["chi"], True)
    if config.get("output_state") is not None:
        import numpy as np
        tau_out = complex(*config["output_state"])
        # Scale the real and imaginary components before taking a complex
        # magnitude: ``abs(1.7e308 + 1.7e308j)`` overflows even though both
        # declared components are finite and the projective state is valid.
        scale = max(1.0, abs(tau_out.real), abs(tau_out.imag))
        state = np.array([1.0 / scale, tau_out / scale], dtype=complex)
        node.set_output_state_target(state / np.linalg.norm(state))
    node.set_whole_pairing(
        _WHOLE_PAIRINGS[str(config.get("whole_pairing", DECLARED_WHOLE_PAIRING))])
    node.set_readout_modes([_READOUT_MODES[name]
                            for name in _readout_names(config.get("readout", DECLARED_READOUT))])
    # Several input pairs on ONE bulk (#1017). The first pair is --tau-a/--tau-b
    # and is what the collar was seeded from, so it is always case zero; the
    # extra pairs are the same tori at different moduli. A case carries its
    # boundary metric and target, plus the marking coefficients needed by the
    # whole-harmonic reading; the triangulation and bulk remain shared.
    extra = _as_state_pairs(config.get("states"))
    base_pair = (complex(*config["tau_a"]), complex(*config["tau_b"]))
    # Cases are set for every four-torus construction as well as for additional
    # input pairs so a whole-harmonic read can use each case's own marking
    # coefficients. The driver currently refuses a tensor target on that
    # direct-sum host, but the low-level case representation remains complete.
    if extra or len(ids) == 4:
        node.set_two_body_cases(
            [_two_body_case(pair, tori, ids, grid, operator, config)
             for pair in [base_pair] + extra])
    for index, torus in enumerate(tori):
        _assert_seed_reads_the_torus(node, index, torus)
    host = node.spacetime()
    inputs = QubitInputs(
        tori, tau_in, ids, cells, markings, algebra,
        config["input_weight"], config["regge"], grid, config["layers"],
        node.objective_name, MC.objective_term_names(),
        [torus.warnings() for torus in tori],
        {"cells": len(host.getTopSimplices()),
         "vertices": len(host.getVertexList().toVector()),
         "edges": len(host.getEdgeList().toVector())})
    return node, inputs

def _boundary_components(faces, regions):
    """The connected components of a boundary, each with its Euler
    characteristic and the input block it lies in.

    Two boundary faces are joined when they share a ridge (a face of one
    dimension less), which is how a closed surface's triangles hang
    together. The Euler characteristic is the alternating count of every
    sub-simplex of the component's faces -- 0 for a torus. A component whose
    vertices all lie in one input block's vertex set is labelled with that
    block; one that spans blocks, or lies outside every block, carries None.
    """
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for index, face in enumerate(faces):
        for ridge in itertools.combinations(face, len(face) - 1):
            other = owner.setdefault(ridge, index)
            if other != index:
                parent[find(other)] = find(index)
    groups = {}
    for index in range(len(faces)):
        groups.setdefault(find(index), []).append(index)
    components = []
    for members in groups.values():
        simplices = set()
        for index in members:
            face = faces[index]
            for size in range(1, len(face) + 1):
                simplices.update(itertools.combinations(face, size))
        vertices = {s[0] for s in simplices if len(s) == 1}
        block = None
        for candidate, region in enumerate(regions):
            if vertices <= region:
                block = candidate
                break
        components.append({
            "faces": len(members),
            "vertices": len(vertices),
            "euler_characteristic": sum((-1) ** (len(s) - 1) for s in simplices),
            "block": block,
        })
    components.sort(key=lambda c: (c["block"] is None, c["block"] or 0,
                                   -c["faces"]))
    return components

# =====================================================================
# qubit frame reads
# =====================================================================

class QubitFrame(ea.AnimationFrame):
    """Every qubit measurement drawn on one accepted geometry."""

    QUBIT_CHANNELS = ("blocks", "leaks", "monodromy", "two_body", "boundary",
                      "completion", "correspondence")

    def __init__(self, node, spacetime, step, config, inputs):
        self.step = step
        self.config = config
        self.spacetime = spacetime
        self.inputs = inputs
        self.objective = self._read_objective(node)
        self.layout = self._read_layout(spacetime, inputs.highlight())
        self.dual = ea.Absent(
            "dual curvature is drawn from the triangle hinges of a "
            "4-dimensional host; the qubit host is 3-dimensional")
        for name in self.CERTIFICATE_CHANNELS:
            setattr(self, name, ea.Absent(
                "not read in the qubit input mode: the baryon certificates "
                "are the neutral mode's instrument"))
        self.betti = self._read_betti(spacetime, config)
        self._read_qubit_channels(node, spacetime, inputs)

    def _read_qubit_channels(self, node, spacetime, inputs):
        """The read-outs of spec S6 over the live complex, all read-only."""
        self.blocks = [self._read_block(node, inputs, index)
                       for index in range(len(inputs.tori))]
        self.leaks = self._read_restricted_leaks(spacetime, inputs)
        self.monodromy = self._read_monodromy(spacetime, inputs)
        self.two_body = self._read_two_body(node, self.config)
        self.boundary = self._read_boundary(spacetime, node)
        self.completion = self._read_completion(node)
        try:
            self.correspondence = hc.measure_geometry(node)
        except (ValueError, RuntimeError, KeyError, LinAlgError) as error:
            self.correspondence = Absent("whole-kernel correspondence unavailable: %s" % error)

    # ---- 12. the qubit blocks (spec S6, per block) ------------------

    @staticmethod
    def _read_block(node, inputs, index):
        """One input block: its residual, the output state read at the block,
        and its qubit read.

        The residual is `own_state_residual` (spec D2) -- the leak of the
        block's input state in the holomorphic form of its OWN Laplacian on
        its live surface, `1 - |<psi(tau_in)|psi(tau_hat)>|^2` -- reported
        with the weight it is scored at. The output state at the block
        (`read_input_state`, spec R1) is the coefficients of the whole's zero
        mode in the block's live frame (the zero mode of its own Laplacian
        normalized by its marking), next to the input, with the leak of that
        fit (`output_leak`): read, never held. The former own-kernel leak
        (`fiber_residual_for_input_block`) rides along as a labelled
        diagnostic: a frame always contains its own coefficients, so it is
        zero for every state and is not scored. The qubit read is the
        engine's `block_qubit` -- `SimplicialQubit` on the block's own
        triangles with the host's live lengths, over the block's marking, in
        the orientation with A.B = +1 -- the very object the residual is
        computed from (tau-hat, the Bloch vector, the spec's J residual, the
        Delaunay and condition diagnostics) and the two distances of spec
        section 11 to the input torus. All but the residual are read-outs
        (spec section 7).
        """
        torus = inputs.tori[index]
        row = {"label": inputs.labels[index],
               "tau_in": inputs.tau_in[index],
               "input": [complex(z) for z in inputs.coefficients_in[index]],
               "weight": inputs.weight}
        try:
            row["residual"] = _finite(node.own_state_residual(index))
        except Exception as error:                        # noqa: BLE001
            row["residual"] = Absent("block residual refused: %s" % error)
        try:
            state = node.read_input_state(index)
        except Exception as error:                        # noqa: BLE001
            row["output_leak"] = Absent("output state read refused: %s"
                                        % error)
            row["coefficients"] = Absent("output state at the block refused: "
                                         "%s" % error)
        else:
            row["output_leak"] = _finite(state.residual)
            if state.obstruction:
                row["coefficients"] = Absent(
                    "state at the block obstructed: %s" % state.obstruction)
            else:
                row["coefficients"] = [complex(z) for z in state.coefficients]
            row["harmonic_rank"] = int(state.harmonic_rank)
            row["frame_rank"] = int(state.frame_rank)
        try:
            row["own_kernel_leak"] = _finite(
                node.fiber_residual_for_input_block(index))
        except Exception as error:                        # noqa: BLE001
            row["own_kernel_leak"] = Absent("own-kernel leak refused: %s"
                                            % error)
        try:
            read = _quiet(lambda: node.block_qubit(index))
        except (KeyError, ValueError, RuntimeError) as error:
            # `block_qubit` names every refusal itself, a torn surface among
            # them, so there is nothing to check for it here.
            row["read"] = Absent("qubit read refused: %s" % error)
            return row
        row["read"] = {
            "tau": complex(read.tau()),
            "bloch": [float(x) for x in read.bloch()],
            "j_residual": _finite(read.j_residual()),
            "non_delaunay_edges": len(read.non_delaunay_edges()),
            "negative_weight_edges": len(read.negative_weight_edges()),
            "condition_m1": _finite(read.condition_m1()),
            "condition_g": _finite(read.condition_g()),
            "near_degenerate": bool(read.near_degenerate()),
            "marking_swapped": bool(read.marking_swapped()),
            "warnings": [str(w) for w in read.warnings()],
            "vertices": len(read.vertices()),
            "edges": len(read.edges()),
            "faces": len(read.faces()),
            "fubini_study_distance": _finite(
                obs.fubini_study_distance(read, torus)),
            "weil_petersson_distance": _finite(
                obs.weil_petersson_distance(read, torus)),
        }
        return row

    # ---- 13. the whole's zero mode restricted to each torus ---------

    @staticmethod
    def _read_restricted_leaks(spacetime, inputs):
        """The leak of each input line in the WHOLE's zero mode, restricted
        to that torus's edges (spec S6; a read-out channel per R1, never the
        definition of the output state).

        The whole -- bulk and boundary edges in one operator -- is assembled
        as one chain-level Whitney pencil, its degree-1 zero mode read on the
        harmonic contour (R7) and restricted to the torus's cells
        (`PencilLayer.read_boundary_fiber`); the leak is the least-squares
        residual of the torus's holomorphic form in those restricted images.
        The restriction is topological in its periods and metric in its
        representative (spec section 6), so this need not vanish even on
        the collar seed; it is recorded, not ruled.
        """
        import numpy as np

        try:
            assembled = cob.PencilLayer.assemble([spacetime])
            contour = cob.PencilLayer.harmonic_contour(assembled, 1)
        except Exception as error:                        # noqa: BLE001
            return Absent("the whole's zero mode could not be read: %s"
                          % error)
        rows = []
        rank = None
        for index, torus in enumerate(inputs.tori):
            try:
                read = cob.PencilLayer.read_boundary_fiber(
                    assembled, 1, contour, inputs.cells[index])
            except Exception as error:                    # noqa: BLE001
                rows.append(Absent("restricted read refused: %s" % error))
                continue
            images = np.asarray(read.images)
            target = np.asarray(torus.holomorphic_form()).reshape(-1)
            if images.ndim != 2 or images.shape[0] != target.shape[0] \
                    or images.shape[1] == 0:
                rows.append(Absent("the zero mode restricted to torus %s has "
                                   "shape %s against %d edges"
                                   % (inputs.labels[index], images.shape,
                                      target.shape[0])))
                continue
            coefficients = np.linalg.lstsq(images, target, rcond=None)[0]
            leak = (np.linalg.norm(images @ coefficients - target) ** 2
                    / np.linalg.norm(target) ** 2)
            rank = int(images.shape[1])
            rows.append({"leak": _finite(leak), "rank": rank})
        return {"per_block": rows, "harmonic_rank": rank}

    # ---- 14. the monodromy between the two markings -----------------

    @staticmethod
    def _read_monodromy(spacetime, inputs):
        """The monodromy of each collar, without dropping conjugate pairs."""
        import numpy as np

        pairs = [(0, 1)] if len(inputs.markings) == 2 else [(0, 1), (2, 3)]
        rows = []
        for left, right in pairs:
            label = "%s -> %s" % (inputs.labels[left], inputs.labels[right])
            try:
                read = MC.monodromy(spacetime, inputs.markings[left],
                                    inputs.markings[right])
            except Exception as error:                    # noqa: BLE001
                rows.append(Absent("%s monodromy refused: %s" % (label, error)))
                continue
            if read.obstruction:
                rows.append(Absent(
                    "%s monodromy obstructed: %s (Betti %s, harmonic rank %d)"
                    % (label, read.obstruction, list(read.betti),
                       read.harmonic_rank)))
                continue
            matrix = np.asarray(read.monodromy)
            rows.append({
                "label": label,
                "betti": [int(b) for b in read.betti],
                "harmonic_rank": int(read.harmonic_rank),
                "monodromy": [[complex(z) for z in row] for row in matrix],
                "rounded": [[int(x) for x in row] for row in read.rounded],
                "rounding_residual": _finite(read.rounding_residual),
                "fit_residual": _finite(read.fit_residual),
            })
        if len(rows) == 1:
            return rows[0]
        return {"pairs": rows}

    # ---- 15. the two-body read ----------------------------------------

    @staticmethod
    def _read_two_body(node, config):
        """`read_two_body`: the transfer in two ordinary or paired frames.

        The matrix is 2 x 2 for two tori and 4 x 4 for four. For two inputs the
        residual is the aggregate selected-readout objective over all declared
        cases. A four-input paired transfer is geometric diagnostic data only,
        so its residual is absent (the engine reports NaN). Per-case residuals,
        the Schmidt spectrum/rank, reversal residual, and block own-kernel
        diagnostics are recorded alongside it.
        """
        import numpy as np

        try:
            read = node.read_two_body()
        except Exception as error:                        # noqa: BLE001
            return Absent("two-body read refused: %s" % error)
        transfer = np.asarray(read.transfer)
        result = {"in_frames": bool(read.in_frames),
                "derived_frames": bool(read.derived_frames),
                "choi_decomposed": bool(read.choi_decomposed),
                "selected_readouts": _readout_names(config["readout"]),
                "transfer": [[complex(z) for z in row] for row in transfer],
                "shape": [int(n) for n in transfer.shape],
                "residual": _finite(read.residual),
                "singular_values": [_finite(s) for s in read.singular_values],
                "schmidt_rank": int(read.schmidt_rank),
                "reversal_residual": _finite(read.reversal_residual),
                # One residual PER STATE when several are fitted at once
                # (#1029). For a two-input read, `residual` above is their sum,
                # exactly the quantity
                # the drive minimises.
                "state_residuals": [_finite(r) for r in
                              node.two_body_residuals_per_case()],
                "input_fiber_residuals": [_finite(r) for r in
                                          read.input_fiber_residuals]}
        if "whole" in result["selected_readouts"]:
            obstruction = str(node.whole_harmonic_obstruction)
            if obstruction:
                result["whole_obstruction"] = obstruction
        return result

    # ---- 16. the boundary and the completion status -----------------

    @staticmethod
    def _read_boundary(spacetime, node):
        """The boundary of W as connected components, each with its Euler
        characteristic and the input block it lies in (`getBoundary` split
        on shared ridges; 0 for a torus)."""
        try:
            faces = [tuple(int(v) for v in face)
                     for face in spacetime.getBoundary()]
        except Exception as error:                        # noqa: BLE001
            return Absent("boundary unavailable: %s" % error)
        if not faces:
            return Absent("the complex is closed: no boundary face")
        regions = [set(int(v) for v in block.vertices) for block in node.inputs]
        components = _boundary_components(faces, regions)
        return {"count": len(components), "faces": len(faces),
                "components": components}

    @staticmethod
    def _read_completion(node):
        """`bridge_phase_complete` and the uncovered torus faces: whether the
        boundary of W is exactly the declared input tori (true by construction
        on the collar; a cone-out dent reopens it)."""
        try:
            uncovered = node.uncovered_input_faces()
            complete = bool(node.bridge_phase_complete())
        except Exception as error:                        # noqa: BLE001
            return Absent("completion status unavailable: %s" % error)
        return {"bridge_phase_complete": complete,
                "uncovered_faces": len(uncovered)}

    def to_json(self):
        document = {
            "step": self.step,
            "objective": self.objective,
            "clusters": self.clusters,
            "bands": self.bands,
            "anchors": self.anchors,
            "transports": self.transports,
            "statistics": self.statistics,
            "crossings": self.crossings,
            "spin": self.spin,
            "betti": self.betti,
            "verdict": self.verdict,
        }
        for name in self.QUBIT_CHANNELS:
            document[name] = getattr(self, name)
        return ea._json_safe(document)


def _tau_text(value):
    return "n/a" if value is None else "%.5f%+.5fi" % (value.real, value.imag)


def _report_qubit(frame):
    """Report the selected-state objective and independent operator evidence."""
    if not isinstance(frame.correspondence, Absent):
        claim = frame.correspondence.get("requested_gate", {})
        if claim:
            print("  whole-kernel gate certificate: %s" % claim["obstruction"])
    def block_residual(index):
        if isinstance(frame.blocks, Absent):
            return "absent"
        if index >= len(frame.blocks):
            return "absent"
        value = frame.blocks[index].get("residual")
        return "absent" if not isinstance(value, float) else "%.2e" % value

    def tau_hat(index):
        if isinstance(frame.blocks, Absent):
            return "absent"
        if index >= len(frame.blocks):
            return "absent"
        read = frame.blocks[index].get("read")
        return "absent" if isinstance(read, Absent) else _tau_text(read["tau"])

    def leak(index):
        if isinstance(frame.leaks, Absent):
            return "absent"
        rows = frame.leaks["per_block"]
        if index >= len(rows) or isinstance(rows[index], Absent):
            return "absent"
        value = rows[index].get("leak")
        return "absent" if value is None else "%.2e" % value

    if isinstance(frame.two_body, Absent):
        two_body, readouts = "absent", "readout"
    else:
        value = frame.two_body.get("residual")
        two_body = "absent" if value is None else "%.4f" % value
        readouts = ",".join(frame.two_body.get("selected_readouts", [])) or "readout"
    betti = ("absent" if isinstance(frame.betti, Absent)
             else [frame.betti["numbers"][d]
                   for d in sorted(frame.betti["numbers"])])
    if isinstance(frame.monodromy, Absent):
        monodromy = "absent"
    else:
        rows = frame.monodromy.get("pairs", [frame.monodromy])
        monodromy = ["absent" if isinstance(row, Absent)
                     else row["rounded"] for row in rows]
    total = frame.objective.get("total")
    labels = frame.inputs.labels
    blocks = " ".join("%s=%s" % (label, block_residual(index))
                      for index, label in enumerate(labels))
    leaks = " ".join("%s=%s" % (label, leak(index))
                     for index, label in enumerate(labels))
    moduli = " ".join("%s=%s" % (label, tau_hat(index))
                      for index, label in enumerate(labels))
    sys.stdout.write(
        "[step %2d] objective %s | blocks %s | %s %s | leaks %s "
        "| tau %s | betti %s | monodromy %s\n"
        % (frame.step, "n/a" if total is None else "%.6g" % total,
           blocks, readouts, two_body, leaks, moduli, betti, monodromy))
    sys.stdout.flush()

# ---- the qubit mode's panels (spec D4) ----------------------------------

def _block_value(frame, index, key):
    """One block's channel value on a frame, or None where it is absent."""
    if isinstance(frame.blocks, Absent) or index >= len(frame.blocks):
        return None
    value = frame.blocks[index].get(key)
    return None if isinstance(value, Absent) else value


def _block_read(frame, index):
    """One block's qubit read on a frame, or None where it is absent."""
    read = _block_value(frame, index, "read")
    return read if isinstance(read, dict) else None


def _leak_value(frame, index):
    if isinstance(frame.leaks, Absent):
        return None
    rows = frame.leaks["per_block"]
    if index >= len(rows) or isinstance(rows[index], Absent):
        return None
    return rows[index]["leak"]


def _qubit_reason(frame):
    """Why a qubit panel has nothing to draw on this frame."""
    if isinstance(frame.blocks, Absent):
        return frame.blocks.reason
    reasons = [row["read"].reason for row in frame.blocks
               if isinstance(row.get("read"), Absent)]
    return "; ".join(reasons) or "no block read on this frame"


def _panel_residuals(axis, frames):
    """Every residual of spec S6 that has a trace, on a log scale: the two
    block residuals (the leak of each input state in the holomorphic form of
    the block's own Laplacian, spec D2), the two-body leak against chi, and
    the two restricted leaks of the input lines in the whole's zero mode --
    the output state's distance from the inputs. Non-positive values have
    no place on a log axis and are left out rather than clipped to a floor
    that would read as a measurement."""
    title = "residuals (log scale)"
    last = frames[-1]
    if last.inputs is None or isinstance(last.blocks, Absent):
        return _absent_panel(axis, title, _qubit_reason(last))
    labels = last.inputs.labels
    series = []
    for index, (label, colour) in enumerate(zip(labels, last.inputs.colours)):
        series.append(("block %s (weight %g)" % (label, last.inputs.weight),
                       colour, "-", lambda f, i=index: _block_value(f, i, "residual")))
        series.append((r"leak %s in the whole's $\ker L_1$" % label, colour, ":",
                       lambda f, i=index: _leak_value(f, i)))
    # One trace per fitted state, so a step that trades one against another is
    # visible. With a single state this is exactly the one two-body trace the
    # panel always drew.
    state_residuals = (last.two_body.get("state_residuals", [])
                 if not isinstance(last.two_body, Absent) else [])
    selected = (", ".join(last.two_body.get("selected_readouts", []))
                if not isinstance(last.two_body, Absent) else "readout")
    if len(state_residuals) > 1:
        for index in range(len(state_residuals)):
            shade = _state_colour(index, len(state_residuals))
            series.append(("state %d (%s)" % (index, selected), shade, "--",
                           lambda f, i=index: _state_residual_value(f, i)))
        series.append(("sum over %d states" % len(state_residuals), "#1f4e79", "-",
                       lambda f: _state_residual_sum(f)))
    else:
        series.append(("selected readout (%s)" % selected, "#1f4e79", "--",
                       lambda f: None if isinstance(f.two_body, Absent)
                       else f.two_body["residual"]))
    drawn = 0
    for label, colour, style, value_of in series:
        points = [(f.step, value_of(f)) for f in frames]
        points = [(s, v) for s, v in points
                  if isinstance(v, float) and math.isfinite(v) and v > 0.0]
        if not points:
            continue
        axis.plot([s for s, _ in points], [v for _, v in points], marker="o",
                  markersize=2.2, linewidth=0.9, linestyle=style, color=colour,
                  label=label)
        drawn += 1
    if not drawn:
        return _absent_panel(axis, title, "no residual carried a positive "
                                          "finite value")
    axis.set_yscale("log")
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("engine unit", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.25, linewidth=0.4, which="both")
    axis.legend(fontsize=5, loc="best", frameon=True, framealpha=0.85)


def _panel_moduli(axis, frames):
    """The two tau-hat trajectories on the upper half plane, tau_in marked.

    tau-hat is the torus's own conformal structure, which the block residual
    of D2 holds at tau_in: the panel shows how far it is allowed to drift
    under synthesis, next to the Weil-Petersson distance that measures it."""
    title = r"$\hat\tau$ on the upper half plane (star: $\tau_{\mathrm{in}}$)"
    last = frames[-1]
    if last.inputs is None or isinstance(last.blocks, Absent):
        return _absent_panel(axis, title, _qubit_reason(last))
    drawn = 0
    for index, (label, colour) in enumerate(zip(last.inputs.labels,
                                                last.inputs.colours)):
        tau_in = last.inputs.tau_in[index]
        axis.plot([tau_in.real], [tau_in.imag], marker="*", markersize=9,
                  color=colour, linestyle="none", zorder=3)
        path = [_block_read(f, index) for f in frames]
        path = [read["tau"] for read in path if read is not None]
        if not path:
            continue
        drawn += 1
        axis.plot([t.real for t in path], [t.imag for t in path], marker="o",
                  markersize=2.2, linewidth=0.8, color=colour, zorder=2,
                  label=r"$\hat\tau_{%s}$ = %s" % (label, _tau_text(path[-1])))
        axis.plot([path[-1].real], [path[-1].imag], marker="o", markersize=5,
                  color=colour, linestyle="none", zorder=4)
    if not drawn:
        return _absent_panel(axis, title, _qubit_reason(last))
    axis.set_title(title, fontsize=8)
    axis.axhline(0.0, linewidth=0.5, color="#333333")
    low, high = axis.get_ylim()
    axis.set_ylim(min(0.0, low), max(high, 0.1))
    axis.set_xlabel(r"$\mathrm{Re}\,\tau$", fontsize=6)
    axis.set_ylabel(r"$\mathrm{Im}\,\tau$", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.25, linewidth=0.4)
    axis.legend(fontsize=5, loc="best", frameon=True, framealpha=0.85)


def _panel_bloch(axis, frame):
    """The Bloch hemisphere the tori can represent (Im tau > 0, i.e.
    r_y > 0), seen from +y: the (r_x, r_z) disk, the read vectors as arrows
    and the input vectors as hollow markers, r_y written beside each."""
    title = (r"Bloch hemisphere $r_y > 0$, seen from $+y$"
             "\n"
             r"(arrow: $\hat r$ of the live read; ring: $r$ of the input)")
    if frame.inputs is None or isinstance(frame.blocks, Absent):
        return _absent_panel(axis, title, _qubit_reason(frame))
    import numpy as np

    theta = np.linspace(0.0, 2.0 * math.pi, 181)
    axis.plot(np.cos(theta), np.sin(theta), color="#bbbbbb", linewidth=0.6)
    axis.axhline(0.0, linewidth=0.4, color="#cccccc")
    axis.axvline(0.0, linewidth=0.4, color="#cccccc")
    drawn = 0
    for index, (label, colour) in enumerate(zip(frame.inputs.labels,
                                                frame.inputs.colours)):
        r_in = [float(x) for x in frame.inputs.tori[index].bloch()]
        axis.plot([r_in[0]], [r_in[2]], marker="o", markersize=7,
                  markerfacecolor="none", markeredgecolor=colour,
                  linestyle="none", zorder=2)
        read = _block_read(frame, index)
        if read is None:
            continue
        drawn += 1
        r = read["bloch"]
        axis.annotate("", xy=(r[0], r[2]), xytext=(0.0, 0.0),
                      arrowprops=dict(arrowstyle="->", color=colour,
                                      linewidth=1.3), zorder=3)
        axis.text(r[0], r[2], r" %s  $r_y$=%.3f" % (label, r[1]), fontsize=5.5,
                  color=colour, va="center")
    if not drawn:
        return _absent_panel(axis, title, _qubit_reason(frame))
    axis.set_title(title, fontsize=8)
    axis.set_xlim(-1.2, 1.2)
    axis.set_ylim(-1.2, 1.2)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$r_x$", fontsize=6)
    axis.set_ylabel(r"$r_z$", fontsize=6)
    axis.tick_params(labelsize=6)


def _panel_transfer(axis, frame):
    """Show one matrix component and name the aggregate scored beside it."""
    title = "selected qubit readout"
    if frame.inputs is None:
        return _absent_panel(axis, title, _qubit_reason(frame))
    if isinstance(frame.two_body, Absent):
        return _absent_panel(axis, title, frame.two_body.reason)
    import numpy as np

    read = frame.two_body
    names = read.get("selected_readouts", _readout_names(frame.config["readout"]))
    residual = read.get("residual")
    residual_text = "n/a" if residual is None else "%.4g" % residual
    component_note = ""
    aggregate_note = ""
    if "operator" in names:
        target = frame.inputs.algebra.get("two_state_vector")
        target_label = r"G|\psi\rangle\langle\phi|G^\dagger"
    elif "transfer" in names:
        target = frame.inputs.algebra["chi"]
        target_label = r"\chi"
        if len(names) > 1:
            component_note = " (transfer component)"
            if "whole" in names:
                aggregate_note = " (includes whole)"
    else:
        reason = read.get("whole_obstruction")
        if reason:
            return _absent_panel(axis, title, reason)
        axis.set_title("%s\naggregate leak %s" % (", ".join(names), residual_text),
                       fontsize=7)
        axis.text(0.5, 0.55,
                  _wrap("No transfer matrix represents this objective. "
                        "The whole read scores the harmonic state itself.", 42),
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=6, color="#666666")
        axis.set_xticks([])
        axis.set_yticks([])
        return
    if target is None:
        return _absent_panel(axis, title, "the selected readout has no recorded target")
    transfer = np.abs(np.asarray(read["transfer"], dtype=complex))
    target = np.abs(np.asarray(target, dtype=complex))
    if transfer.shape != target.shape:
        return _absent_panel(axis, title, "the selected transfer is %s but its target is %s"
                             % (transfer.shape, target.shape))
    rows, columns = target.shape
    grid = np.full((rows, 2 * columns + 1), np.nan)
    grid[:, :columns] = transfer / max(float(transfer.max()), 1e-300)
    grid[:, columns + 1:] = target / max(float(target.max()), 1e-300)
    axis.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    def ink(value):
        # Legible on both ends of the colour map.
        return "#ffffff" if value > 0.6 else "#333333"

    for i in range(rows):
        for j in range(columns):
            axis.text(j, i, "%.3g" % transfer[i, j], ha="center", va="center",
                      fontsize=5.5, color=ink(grid[i, j]))
            axis.text(columns + 1 + j, i, "%.3g" % target[i, j], ha="center",
                      va="center", fontsize=5.5, color=ink(grid[i, columns + 1 + j]))
    axis.set_xticks(list(range(columns)) + list(range(columns + 1, 2 * columns + 1)))
    axis.set_xticklabels([r"$|%d\rangle$" % j for j in range(columns)] * 2, fontsize=6)
    axis.set_yticks(range(rows))
    axis.set_yticklabels([r"$|%d\rangle$" % i for i in range(rows)], fontsize=6)
    # A blank band above the matrices carries their labels inside the axes,
    # clear of the title.
    axis.set_ylim(rows - 0.5, -1.3)
    axis.text((columns - 1) / 2.0, -0.85, r"$|T_{AB}| / \max|T_{AB}|$",
              ha="center", fontsize=6, color="#333333")
    axis.text(columns + 1 + (columns - 1) / 2.0, -0.85,
              r"$|%s| / \max|%s|$" % (target_label, target_label),
              ha="center", fontsize=6, color="#333333")
    spectrum = ", ".join("%.3g" % s for s in read["singular_values"]
                         if s is not None)
    reversal = read.get("reversal_residual")
    reversal_text = "n/a" if reversal is None else "%.1e" % reversal
    axis.set_title("%s%s: %s\naggregate leak %s%s, $\\sigma$ (%s) rank %d, reversal %s%s"
                   % (title, component_note, ", ".join(names), residual_text,
                      aggregate_note, spectrum,
                      read["schmidt_rank"], reversal_text,
                      "" if read["in_frames"] else ", identity frames"),
                   fontsize=6.5)


def _panel_topology(axis, frame):
    """The whole's topology as text: Betti numbers, the boundary components
    with their Euler characteristics, the completion status, the harmonic
    rank and the monodromy between the two markings with its residuals."""
    title = "the whole: Betti, boundary, monodromy"
    if frame.inputs is None:
        return _absent_panel(axis, title, _qubit_reason(frame))
    lines = []
    if isinstance(frame.betti, Absent):
        lines.append("Betti: " + frame.betti.reason)
    else:
        numbers = frame.betti["numbers"]
        lines.append("Betti %s" % [numbers[d] for d in sorted(numbers)])
    if isinstance(frame.boundary, Absent):
        lines.append("boundary: " + frame.boundary.reason)
    else:
        parts = ["chi=%d (%d faces, %s)"
                 % (c["euler_characteristic"], c["faces"],
                    "block %s" % frame.inputs.labels[c["block"]]
                    if c["block"] is not None else "no single block")
                 for c in frame.boundary["components"]]
        lines.append("boundary: %d component(s): %s"
                     % (frame.boundary["count"], "; ".join(parts)))
    if isinstance(frame.completion, Absent):
        lines.append("completion: " + frame.completion.reason)
    else:
        lines.append("boundary of W is the %d input tori: %s (%d uncovered face(s))"
                     % (len(frame.inputs.tori),
                        "yes" if frame.completion["bridge_phase_complete"] else "no",
                        frame.completion["uncovered_faces"]))
    if isinstance(frame.leaks, Absent):
        lines.append("zero mode of the whole: " + frame.leaks.reason)
    else:
        lines.append("harmonic rank of the whole: %s"
                     % frame.leaks["harmonic_rank"])
    if isinstance(frame.monodromy, Absent):
        lines.append("monodromy: " + frame.monodromy.reason)
    else:
        rows = frame.monodromy.get("pairs", [frame.monodromy])
        for m in rows:
            if isinstance(m, Absent):
                lines.append("monodromy: " + m.reason)
                continue
            lines.append("monodromy %s rounded %s (rounding %s, fit %s)"
                         % (m.get("label", "A -> B"), m["rounded"],
                            "n/a" if m["rounding_residual"] is None
                            else "%.1e" % m["rounding_residual"],
                            "n/a" if m["fit_residual"] is None
                            else "%.1e" % m["fit_residual"]))
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    y = 0.95
    for line in lines:
        wrapped = _wrap(line, 48)
        axis.text(0.03, y, wrapped, transform=axis.transAxes, fontsize=5.8,
                  va="top", ha="left", color="#333333")
        y -= 0.095 * (wrapped.count("\n") + 1)

def _suptitle(frame, last_step):
    """Name the qubit inputs, objective, and post-hoc status on the figure."""
    inputs = frame.inputs
    second = 2 if len(inputs.tori) == 4 else 1
    jt = inputs.algebra.get("Jt")
    interaction = ("J t = %g" % jt
                   if isinstance(jt, float) and math.isfinite(jt)
                   else "operator %s" % inputs.algebra["operator"])
    return ("qubit cobordism -- engine unit %d of %d -- tau_A %s, tau_B "
            "%s, %d tori on %dx%d grids, %d layers, %s, readout %s, "
            "input weight %g, Regge term %s, objective %s (read-outs post-hoc)"
            % (frame.step, last_step, _tau_text(inputs.tau_in[0]),
               _tau_text(inputs.tau_in[second]), len(inputs.tori), inputs.grid,
               inputs.grid, inputs.layers, interaction, frame.config["readout"],
               inputs.weight, "on" if inputs.regge else "off",
               inputs.objective_name))


def _state_residual_value(frame, index):
    """State `index`'s own residual on that frame, or None if unmeasured."""
    if isinstance(frame.two_body, Absent):
        return None
    values = frame.two_body.get("state_residuals", [])
    return values[index] if index < len(values) else None


def _state_residual_sum(frame):
    """What the drive actually minimises: the sum over the fitted states."""
    if isinstance(frame.two_body, Absent):
        return None
    values = [v for v in frame.two_body.get("state_residuals", [])
              if isinstance(v, float) and math.isfinite(v)]
    return sum(values) if values else None


def _state_colour(index, count):
    """A distinct shade per state, dark to light, so many states stay legible.

    Deliberately not the torus colours: those name the two BOUNDARY blocks, and
    reusing them here would suggest a correspondence that does not exist -- a
    state is a pair over both tori, not one of them.
    """
    import matplotlib

    return matplotlib.colormaps["viridis"](
        0.12 + 0.76 * (index / max(1, count - 1)))


def _panel_correspondence(axis, frame):
    """Show whole-kernel operators, never relabel selected-state chi as a gate."""
    import numpy as np

    read = frame.correspondence
    title = "whole harmonic operators |T|"
    if isinstance(read, Absent):
        return _absent_panel(axis, title, read.reason)
    if read.get("obstruction"):
        return _absent_panel(axis, title, read["obstruction"])
    matrices, labels, notes = [], [], []
    for name, value in read["readouts"].items():
        if not value["identifiable"]:
            notes.append("%s: %s" % (name, value["obstruction"]))
            continue
        matrix = np.asarray(value["operator"], dtype=complex)
        matrices.append(matrix)
        labels.extend("%s %d" % (name, j) for j in range(matrix.shape[1]))
        notes.append("%s: kernel %.1e; held-out %.1e; isometry %.1e" % (
            name, value["kernel_residual"],
            max(value["held_out"]["errors"].values()),
            value["quantum"]["coordinate_isometry_error"]))
    if not matrices:
        return _absent_panel(axis, title, "; ".join(notes))
    matrix = np.hstack(matrices)
    axis.imshow(np.abs(matrix), cmap="viridis", aspect="auto")
    axis.set_title(title, fontsize=8)
    axis.set_xticks(range(matrix.shape[1]), labels, fontsize=6)
    axis.set_yticks(range(matrix.shape[0]), range(matrix.shape[0]), fontsize=6)
    axis.set_ylabel("output coordinate", fontsize=6)
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, "%.2g%+.2gj" % (value.real, value.imag),
                  color="white", ha="center", va="center", fontsize=6,
                  bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none"})
    notes.append("Two-qubit gate: NOT CERTIFIED (missing tensor register)")
    axis.set_xlabel("\n".join(notes), fontsize=6)


_QUBIT_PANEL_ORDER = ("objective", "residuals", "moduli", "bloch",
                      "transfer", "topology", "layout", "betti")

_QUBIT_PRIMARY_PANELS = [
    ("objective", ea._panel_objective),
    ("residuals", _panel_residuals),
    ("moduli", _panel_moduli),
    ("bloch", _panel_bloch),
    ("transfer", _panel_correspondence),
    ("topology", _panel_topology),
    ("layout", ea._panel_layout),
    ("betti", ea._panel_betti),
]
#: The qubit instrument is EXACTLY the panels above. It deliberately does not
#: inherit the remainder of `ea._PANELS`: clusters, bands, anchors, transports,
#: statistics, crossings, mass, spin, verdict and the two dual-graph panels
#: measure the neutral emergence question (does matter emerge from the
#: geometry), not this one. The qubit experiment never reads them, so drawing
#: them here filled eleven of the figure's nineteen panels with quantities
#: that are not about the qubit -- and shrank the eight that are.
_PANELS = list(_QUBIT_PRIMARY_PANELS)
_QUBIT_PANELS = _PANELS
_TRACE_PANELS = ("objective", "residuals", "moduli")
#: Of the shared placed panels only `layout` survives into the qubit set; the
#: two dual-graph panels went with the emergence instrument. Filtered rather
#: than rewritten so that a panel added to `ea._PLACED_PANELS` and adopted into
#: `_QUBIT_PANEL_ORDER` is still handed its placement.
_PLACED_PANELS = tuple(name for name in ea._PLACED_PANELS
                       if name in _QUBIT_PANEL_ORDER)

#: The grid these panels lay out on, derived from how many there are rather
#: than fixed. Eight panels resolve to 2x4: no unused cells, and a
#: columns-to-rows ratio of exactly 2, which is the shape of both canvases
#: below and so gives square cells. Adding or removing a qubit panel re-derives
#: it; nothing here has to be kept in step by hand.
DECLARED_PANEL_GRID = ea._grid_for(len(_PANELS))

#: The qubit live window and rendered canvas. Private (not `DECLARED_`) because
#: they are a property of this figure's panels, not a knob of the shared
#: experiment.
#:
#: Each keeps the width of the emergence canvas it replaces and takes the
#: height its own row count needs, which is what makes the remaining panels
#: bigger rather than merely fewer. On the live window a cell goes from
#: 3.6 x 2.5 inches (4x5 on 18x10) to 4.5 x 4.5 (2x4 on 18x9) -- 2.25 times the
#: area. On the rendered canvas it goes from 5.0 x 3.0 (4x5 on 20x12) to
#: 5.0 x 5.0 (2x4 on 20x10), 1.67 times the area, on a smaller file.
_QUBIT_LIVE_FIGSIZE = (18, 9)
_QUBIT_RENDER_FIGSIZE = (20, 10)


def panels_for(_config=None):
    """The fixed panel order of the standalone qubit experiment."""
    return _PANELS


def _block_geometry(node, inputs, index):
    """One input block's own surface, in the same shape as the whole."""
    block = {
        "label": inputs.labels[index],
        "vertices": sorted(int(v) for v in inputs.vertex_ids[index].values()),
        "marking": [[list(step) for step in cycle]
                    for cycle in inputs.markings[index]],
        "coefficients": [complex(z) for z in inputs.coefficients_in[index]],
        "tau_in": inputs.tau_in[index],
    }
    surface = MC.block_surface_subcomplex(node.inputs[index], node.spacetime())
    if surface is None:
        # A torn surface carries no state and no geometry; say so rather
        # than write a complex that is not the torus.
        block["surface"] = None
        return block
    cells = [[int(v.getId()) for v in cell.getVertices()]
             for cell in surface.getTopSimplices()]
    edges = []
    phases = []
    for edge in surface.getEdgeList().toVector():
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        squared = complex(edge.getLength()) ** 2
        edges.append([source, target, squared.real, squared.imag])
        phase = complex(edge.getPhase())
        if phase != 0:
            phases.append([source, target, phase.real, phase.imag])
    # The same shape as the whole, times included, so one loader reads either.
    block["surface"] = {"dimensions": len(cells[0]) - 1 if cells else 0,
                        "cells": cells, "edges": edges,
                        "vertex_times": [[int(v.getId()), float(v.getTime())]
                                         for v in surface.getVertexList().toVector()]}
    if phases:
        block["surface"]["edge_phases"] = phases
    return block


def geometry_document(node, inputs=None, source=None):
    """The live whole complex plus each qubit input block's surface."""
    document = ea.geometry_document(node, source=source)
    if inputs is not None:
        document["blocks"] = [
            _block_geometry(node, inputs, index)
            for index in range(len(inputs.tori))]
    return ea._json_safe(document)


def _configure_qubit_node(node, inputs, config):
    """Apply the qubit boundary policy before the first frame is read."""
    node.set_boundary_may_extend(bool(config["extend_boundary"]))
    if config.get("pin_boundary", DECLARED_PIN_BOUNDARY):
        # The input states are inputs: the attachment already put the correct
        # boundary in place, so stage 2 has nothing to improve there. A pinned
        # region zeroes the descent on the edges inside it, which is exactly
        # the block's own surface.
        for index in range(len(node.inputs)):
            node.declare_pinned_region(
                "input%d" % index,
                set(int(v) for v in node.inputs[index].vertices))


def stall_causes(config, frames):
    """Why a run that never improved never improved, named.

    A drive that does not move exits as `tolerance-reached`, which reads as
    convergence. Two configurations reach it while doing nothing, and a
    reader who is not told cannot tell them from a run that arrived
    somewhere. Reported from a real run as "exits after one try, ignoring
    --patience" -- patience was working and counting genuine stalls; nothing
    said what was stalling.

    Diagnostic only: nothing here changes the terminator, the objective, or
    any recorded value.
    """
    causes = []
    names = _readout_names(config.get("readout", DECLARED_READOUT))
    if "whole" in names and config.get("whole_pairing",
                                       DECLARED_WHOLE_PAIRING) == "periods":
        causes.append(
            "--readout whole under --whole-pairing periods CANNOT respond to "
            "the geometry. The reading pairs cohomology with homology, so it "
            "sees only a cohomology class; a change of metric moves the "
            "harmonic representative by exactly a coboundary, and a "
            "coboundary's period around a closed cycle telescopes to zero. "
            "Every unit is a stall by construction. Try --whole-pairing gram, "
            "which contracts through the chain metric instead, or "
            "--readout transfer")
    disposition = config.get("interior_disposition",
                             DECLARED_INTERIOR_DISPOSITION)
    if disposition != ea.EdgeDisposition.SPACELIKE:
        causes.append(
            "--edge-disposition %s seeds the collar's interior away from the "
            "lengths seed_collar wires. `random` in particular gives every "
            "interior edge a uniformly random phase, which has been measured "
            "to leave no improving move on the first unit" % (disposition,))
    return causes


def stall_report(config, frames):
    """The lines to print when a drive never moved, or none if it did.

    "Never moved" is measured against the run's own tolerance and over the
    whole run, not a single unit: a run that improves and then stops has
    converged or stalled on its geometry, which is a result. A run whose last
    unit reads the same as its first did nothing at all, and that is a
    configuration fault worth naming.
    """
    if len(frames) < 2:
        return []
    first = ea._objective_total(frames[0])
    last = ea._objective_total(frames[-1])
    if first is None or last is None:
        return []
    if not ea._converged(first, last, config.get("tolerance", 0.0)):
        return []
    causes = stall_causes(config, frames)
    if not causes:
        return []
    lines = ["the objective did not move over %d engine unit%s (%s throughout). "
             "That is a configuration, not a result:"
             % (len(frames) - 1, "" if len(frames) == 2 else "s",
                ea._format_objective_total(frames[-1]))]
    lines.extend("  * " + cause for cause in causes)
    return lines


def drive(config, progress=False, on_frame=None, on_node=None, on_setup=None,
          stop_requested=None):
    """Drive the qubit cobordism through the shared engine-unit loop."""
    result = ea._drive(
        config, build_qubit_node, QubitFrame, _report_qubit,
        configure_node=_configure_qubit_node, progress=progress,
        on_frame=on_frame, on_node=on_node, on_setup=on_setup,
        stop_requested=stop_requested)
    if progress:
        for line in stall_report(config, result.frames):
            sys.stdout.write(line + "\n")
    return result


def draw_frame(figure, frames, index, placed=None):
    """Draw one qubit frame with the qubit panels and title."""
    return ea._draw_frame(figure, frames, index, _PANELS, _suptitle,
                          _TRACE_PANELS, _PLACED_PANELS, placed,
                          grid=DECLARED_PANEL_GRID)


def drive_live(config, progress=False, on_node=None, on_setup=None):
    """Drive the qubit cobordism while displaying completed frames."""
    return ea._drive_live(
        config, drive, draw_frame, progress=progress, on_node=on_node,
        on_setup=on_setup, thread_name="qubit-animation-drive",
        figsize=_QUBIT_LIVE_FIGSIZE)


def render(frames, path):
    """Render qubit frames through the shared animation renderer."""
    return ea._render(frames, path, draw_frame,
                      figsize=_QUBIT_RENDER_FIGSIZE)

def _as_complex(value):
    """A complex from a number, a string like ``0.3+1.1j`` or a ``[re, im]``
    pair (the config's JSON form)."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("a modulus is a complex number or a [re, im] "
                             "pair, got %r" % (value,))
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, str):
        return complex(value.replace(" ", ""))
    return complex(value)

def _modulus_value(name, value):
    """Return a finite upper-half-plane torus modulus."""
    try:
        tau = _as_complex(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s is a modulus tau, e.g. 0.3+1.1j; got %r"
                         % (name, value)) from None
    if not (math.isfinite(tau.real) and math.isfinite(tau.imag)
            and tau.imag > 0.0):
        raise ValueError(
            "%s must lie in the upper half plane (Im > 0), got %r: the "
            "poles |0> and |1> are limits reached by pinching, not inputs "
            "(spec S1)" % (name, tau))
    return tau


def _state_value(name, value):
    """Return a finite projective-state coordinate."""
    try:
        state = _as_complex(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s is a complex state coordinate tau, e.g. "
                         "0.3+1.1j; got %r" % (name, value)) from None
    if not (math.isfinite(state.real) and math.isfinite(state.imag)):
        raise ValueError("%s must be finite, got %r" % (name, state))
    return state

def _complex_argument(text):
    """The CLI's complex parser: ``0.3+1.1j``, ``1.1j``, ``-0.2+0.8j``."""
    try:
        return _as_complex(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "%r is not a complex number; write it like 0.3+1.1j" % (text,))

def _add_qubit_run_arguments(run):
    run.add_argument("--readout", default=DECLARED_READOUT,
                     help="which space the two-body target is scored "
                          "against, as a comma-separated set that is SUMMED. "
                          "'transfer' is the whole complex's degree-1 "
                          "operator read as the coupling block between the "
                          "two boundary frames; 'bulk' is reserved but "
                          "refused because this driver declares no Choi frame; "
                          "'operator' is refused because its paired frame is "
                          "a direct sum, not a two-qubit tensor product; "
                          "'whole' is ker L_1(W), the boundary INCLUDED, read "
                          "through the blocks' markings. This is part of the "
                          "OBJECTIVE, not the reporting: the residual is a "
                          "term in r_U (default %s)" % DECLARED_READOUT)
    run.add_argument("--edge-disposition", dest="interior_disposition",
                     choices=list(ea.EdgeDisposition.ALL),
                     default=DECLARED_INTERIOR_DISPOSITION,
                     help="the causal character the COLLAR'S INTERIOR edges "
                          "are seeded with. seed_collar writes each torus's "
                          "own lengths onto its edges and wires every other "
                          "edge to 1.0, so without this the bulk is "
                          "all-spacelike and cannot say otherwise. The tori's "
                          "own edges are never written -- they carry the "
                          "declared input moduli, so a disposition over them "
                          "would change the input states rather than the bulk "
                          "they sit in, which makes 'spacelike' a no-op here "
                          "rather than a rewrite. 'foliated' reads its "
                          "layering from the whole host, so it follows the "
                          "collar's own layers: edges spanning a layer are "
                          "timelike and edges within one spacelike.")
    run.add_argument("--whole-pairing", dest="whole_pairing",
                     choices=sorted(_WHOLE_PAIRINGS),
                     default=DECLARED_WHOLE_PAIRING,
                     help="how --readout whole pairs its harmonic columns "
                          "against the input blocks. 'periods' integrates "
                          "each column over the marked cycles: how a state is "
                          "defined on a boundary torus, but TOPOLOGICAL on "
                          "the three-dimensional bulk, where the Hodge star "
                          "sends 1-forms to 2-forms so no holomorphic line "
                          "exists for a metric to select. There a change of "
                          "metric moves the harmonic representative by "
                          "exactly a coboundary, whose period around a closed "
                          "cycle telescopes to zero, so the residual does not "
                          "move at all -- measured, 1e-16 under a 75 per cent "
                          "jitter of every squared length -- and a run scored "
                          "on it cannot improve. 'gram' contracts through the "
                          "chain metric instead; the same jitter moves that "
                          "by 76 per cent. Default 'periods', which is what "
                          "every recorded run used.")
    run.add_argument("--collar-twist", dest="collar_twist",
                     choices=("none", "swap"), default=DECLARED_COLLAR_TWIST,
                     help="the mapping class that relabels the far surface "
                          "before the collar identifies the two. 'none' is the "
                          "PRODUCT collar: its two boundary tori are "
                          "homologous, so its monodromy M = P_B P_A^-1 is the "
                          "identity and it carries its input straight through. "
                          "That is topological, not a matter of tuning -- M is "
                          "the induced map on H^1, and it stays at ||M - I|| = "
                          "5e-15 under a 75%% jitter of every squared length, "
                          "so NO relaxation reaches a different propagator. "
                          "'swap' exchanges the two cycles, giving "
                          "M = [[0, 1], [1, 0]], det -1, the propagator "
                          "tau -> 1/tau, and an I-bundle that is not a "
                          "product. Only these two are offered because a "
                          "mapping class must be a simplicial automorphism of "
                          "the triangulation to collar at all, and a Dehn "
                          "twist is not one here. The whole-complex harmonic "
                          "reading requires the product collar's frame "
                          "agreement, so --readout whole refuses swap. A "
                          "continuous gate lives in --readout transfer, which "
                          "the same jitter moves by two orders of magnitude.")
    run.add_argument("--tori", type=int, choices=(2, 4),
                     default=DECLARED_TORI,
                     help="how many tori bound the cobordism (default torus "
                          "count %d). Two is one per "
                          "input state, the collar of spec S3. Four carries "
                          "each state on a PAIR, itself and its orientation "
                          "reversal at -conj(tau), on two collars joined "
                          "along a removed tetrahedron. The count decides "
                          "which readings can run: "
                          "rank(H^1(W) -> H^1(dW)) = b_1(dW)/2, so two tori "
                          "leave a harmonic space of rank 2 and four leave 4, "
                          "against a 4-dimensional target. --readout "
                          "transfer and whole need two; bulk and operator are "
                          "refused by this driver, and the four-torus direct "
                          "sum is not identified with a two-qubit tensor "
                          "product. Thus 4 is a compatibility value that "
                          "fails with the specific representation reason; "
                          "the paired host itself remains a low-level "
                          "diagnostic. Four needs three collar layers"
                          % DECLARED_TORI)
    run.add_argument("--layers", type=int, default=DECLARED_COLLAR_LAYERS,
                     help="qubit mode: product layers in the collar seed, at "
                          "least one. --tori 4 requires three, so smaller "
                          "values are normalized to three (default %d)"
                          % DECLARED_COLLAR_LAYERS)
    run.add_argument("--output-state", dest="output_state", default=None,
                     help="the state the whole complex's harmonic form is "
                          "meant to be, as a modulus tau (e.g. 0.3+1.1j): the "
                          "target is psi(tau) = (1, tau) normalized. The "
                          "harmonic IS a state and its dimension is b_1(W), "
                          "so at --tori 2 it is 2-dimensional and this names "
                          "it. At --tori 2 every accepted readout set "
                          "containing whole requires this option; a set "
                         "without whole refuses it")
    run.add_argument("--tau-a", type=_complex_argument, default=DECLARED_TAU_A,
                     help="qubit mode: the modulus of torus A in the upper "
                          "half plane, e.g. 0.3+1.1j (default %s)"
                          % DECLARED_TAU_A)
    run.add_argument("--tau-b", type=_complex_argument, default=DECLARED_TAU_B,
                     help="qubit mode: the modulus of torus B; a value with "
                          "a negative real part needs the = form, "
                          "--tau-b=-0.2+0.8j (default %s)" % DECLARED_TAU_B)
    run.add_argument("--grid", type=int, default=DECLARED_GRID,
                     help="qubit mode: each flat torus is a grid x grid "
                          "triangulation, at least 3 (default %d)"
                          % DECLARED_GRID)
    run.add_argument("--J", dest="coupling", type=float,
                     default=DECLARED_COUPLING,
                     help="qubit mode: the exchange coupling J of the "
                          "flip-flop H = hbar J (s1+ s2- + s1- s2+) "
                          "(default %g)" % DECLARED_COUPLING)
    run.add_argument("--time", type=float, default=DECLARED_TIME,
                     help="qubit mode: the time t of the recorded exact "
                          "evolution and first-order amplitude -i J t chi "
                          "(default %g)" % DECLARED_TIME)
    run.add_argument("--input-weight", type=float,
                     default=DECLARED_INPUT_WEIGHT,
                     help="qubit mode: weight of each block's residual (its "
                          "OWN Laplacian's zero mode against the torus input "
                          "coefficients) in r_U (default %g)"
                          % DECLARED_INPUT_WEIGHT)
    run.add_argument("--regge", action=argparse.BooleanOptionalAction,
                     default=DECLARED_REGGE,
                     help="qubit mode: keep the Regge stationarity term in "
                          "the objective (--no-regge leaves r_U alone; "
                          "default %s)" % ("on" if DECLARED_REGGE else "off"))
    run.add_argument("--state", dest="states", default=None,
                     metavar="TAU:TAU[:TAU:TAU...]",
                     help="ADDITIONAL input pairs the same bulk must also "
                          "satisfy, as a colon-delimited list of moduli read "
                          "two at a time: --state 0.5+0.9j:0.1+1.3j:1j:1j is "
                          "the pairs (0.5+0.9j, 0.1+1.3j) and (1j, 1j). An "
                          "odd count is refused. --tau-a/--tau-b are always "
                          "the first pair. A single --output-state cannot be "
                          "the answer for several different inputs, so the "
                          "two options are refused together")
    run.add_argument("--score-leak", action="store_true",
                     dest="score_leak", default=DECLARED_SCORE_LEAK,
                     help="also score the WHOLE cobordism's leak of each "
                          "input state, beside the block's own residual and "
                          "at the same --input-weight. Off by default. The "
                          "block's own residual reads one torus in isolation "
                          "and cannot see the bulk at all; the leak reads the "
                          "whole complex's zero mode. With a held boundary "
                          "the own residual sits at rounding and what it "
                          "contributes is negligible, so without this the "
                          "objective is effectively the two-body term alone")
    run.add_argument("--operator", default=DECLARED_OPERATOR,
                     choices=["flip_flop"] + sorted(DECLARED_GATES),
                     help="qubit mode: the operator whose output the transfer "
                          "is fitted to. flip_flop is the interaction "
                          "Hamiltonian of spec S5 at --J and --time; the rest "
                          "are unitary two-qubit gates, fitted through their "
                          "image of the product state (default %s)"
                          % DECLARED_OPERATOR)
    run.add_argument("--pin-boundary-state", action="store_true",
                     dest="pin_boundary_state",
                     default=DECLARED_PIN_BOUNDARY_STATE,
                     help="qubit mode: attach every declared state to ONE "
                          "shared boundary at a time, instead of scoring each "
                          "against a boundary of its own (default %s). Each "
                          "case normally carries its own squared lengths, "
                          "which are written into the complex for that case's "
                          "read and restored after, so the boundary is a "
                          "parameter supplied per case rather than a "
                          "coordinate the drive owns -- and --no-pin-boundary "
                          "is measurably inert. This drops that payload, so "
                          "every state is read on the live shared complex and "
                          "the boundary is free to relax and to be changed "
                          "combinatorially. It releases the geometric pin, "
                          "and is refused together with --pin-boundary. "
                          "Needs --state: with a single input pair no cases "
                          "are set and nothing is substituted"
                          % ("on" if DECLARED_PIN_BOUNDARY_STATE else "off"))
    run.add_argument("--pin-boundary", action=argparse.BooleanOptionalAction,
                     dest="pin_boundary", default=None,
                     help="qubit mode: hold the input tori fixed while the "
                          "bulk relaxes (default %s). The block residual sees "
                          "only tau_hat, so with --no-pin-boundary the "
                          "boundary is reshaped freely within its conformal "
                          "class and the bulk is fitted to a boundary that is "
                          "not the input it was given"
                          % ("on" if DECLARED_PIN_BOUNDARY else "off"))
    run.add_argument("--extend-boundary", action="store_true",
                     dest="extend_boundary",
                     default=DECLARED_EXTEND_BOUNDARY,
                     help="qubit mode: let a cone move extend the declared "
                          "surface boundary of W. Off by default: a cone that "
                          "hands dW a face it did not have changes what the "
                          "cobordism is. Neutral mode refuses this option; "
                          "its pinned M0 does not declare fixed topology")

def build_config(steps=DECLARED_STEPS, seed=DECLARED_SEED,
                 stage1_iters=DECLARED_STAGE1_ITERS,
                 stage2_iters=DECLARED_STAGE2_ITERS,
                 tolerance=DECLARED_TOLERANCE,
                 patience=DECLARED_PATIENCE,
                 candidate_moves=DECLARED_CANDIDATE_MOVES,
                 combinatorial_depth=_LEGACY_UNSET,
                 combinatorial_length=_LEGACY_UNSET,
                 backsteps=ea.DECLARED_BACKSTEPS,
                 readout=DECLARED_READOUT,
                 tori=DECLARED_TORI,
                 collar_twist=DECLARED_COLLAR_TWIST,
                 interior_disposition=DECLARED_INTERIOR_DISPOSITION,
                 whole_pairing=DECLARED_WHOLE_PAIRING,
                 output_state=DECLARED_OUTPUT_STATE,
                 layers=DECLARED_COLLAR_LAYERS,
                 tau_a=DECLARED_TAU_A,
                 tau_b=DECLARED_TAU_B,
                 grid=DECLARED_GRID,
                 coupling=DECLARED_COUPLING,
                 time=DECLARED_TIME,
                 input_weight=DECLARED_INPUT_WEIGHT,
                 regge=DECLARED_REGGE,
                 extend_boundary=DECLARED_EXTEND_BOUNDARY,
                 score_leak=DECLARED_SCORE_LEAK,
                 states=DECLARED_STATES,
                 operator=DECLARED_OPERATOR,
                 pin_boundary=None,
                 pin_boundary_state=DECLARED_PIN_BOUNDARY_STATE,
                 *, surgical_depth=_LEGACY_UNSET,
                 combinatorial_breadth=_LEGACY_UNSET):
    """Validate and record one qubit-animation run."""
    config = ea._build_common_config(
        steps=steps, seed=seed, stage1_iters=stage1_iters,
        stage2_iters=stage2_iters, tolerance=tolerance, patience=patience,
        candidate_moves=candidate_moves,
        combinatorial_depth=combinatorial_depth,
        combinatorial_length=combinatorial_length,
        surgical_depth=surgical_depth,
        combinatorial_breadth=combinatorial_breadth,
        backsteps=backsteps)

    grid = ea._cpp_int_value("grid", grid)
    layers = ea._cpp_int_value("collar layers", layers)
    tori = ea._integer_value("tori", tori)
    coupling = ea._finite_value("J", coupling)
    time = ea._finite_value("time", time)
    input_weight = ea._finite_value("input weight", input_weight)
    if input_weight <= 0.0:
        raise ValueError("input weight must be a positive finite number, "
                         "got %r" % input_weight)

    regge = ea._boolean_value("regge", regge)
    extend_boundary = ea._boolean_value(
        "extend_boundary", extend_boundary)
    score_leak = ea._boolean_value("score_leak", score_leak)
    pin_boundary_state = ea._boolean_value("pin_boundary_state",
                                           pin_boundary_state)
    # `pin_boundary` defaults to None, meaning "not asked for either way",
    # which is what lets --pin-boundary-state release the geometric pin without
    # silently overriding a pin the caller asked for by name. Behind a pin the
    # new flag would be inert: the pin zeroes the descent on exactly the block
    # edges the shared attachment exists to let move.
    if pin_boundary is None:
        pin_boundary = False if pin_boundary_state else DECLARED_PIN_BOUNDARY
    else:
        pin_boundary = ea._boolean_value("pin_boundary", pin_boundary)
        if pin_boundary and pin_boundary_state:
            raise ValueError(
                "--pin-boundary and --pin-boundary-state are refused together: "
                "the first freezes the block edges and the second exists to "
                "let them relax to a boundary that serves every state at once, "
                "so the pin would leave the second with nothing to move. Drop "
                "--pin-boundary (it is released automatically) or drop "
                "--pin-boundary-state")

    if collar_twist not in ("none", "swap"):
        raise ValueError("unknown collar twist %r: expected none or swap"
                         % (collar_twist,))
    if interior_disposition not in ea.EdgeDisposition.ALL:
        raise ValueError("unknown edge disposition %r: expected one of %s"
                         % (interior_disposition, ", ".join(ea.EdgeDisposition.ALL)))
    if whole_pairing not in _WHOLE_PAIRINGS:
        raise ValueError("unknown whole pairing %r: expected one of %s"
                         % (whole_pairing, ", ".join(sorted(_WHOLE_PAIRINGS))))
    operators = ("flip_flop",) + tuple(sorted(DECLARED_GATES))
    if operator not in operators:
        raise ValueError("unknown operator %r: expected one of %s"
                         % (operator, ", ".join(operators)))
    if operator == "flip_flop" and not math.isfinite(coupling * time):
        raise ValueError("J times time must be finite, got %r times %r"
                         % (coupling, time))

    tau_a = _modulus_value("tau_a", tau_a)
    tau_b = _modulus_value("tau_b", tau_b)
    moduli = {"tau_a": [tau_a.real, tau_a.imag],
              "tau_b": [tau_b.real, tau_b.imag]}
    if grid < 3:
        raise ValueError("grid must be at least 3 (below 3 the torus grid is "
                         "not a simplicial complex), got %r" % (grid,))
    names = _readout_names(readout)
    if layers < 1:
        raise ValueError("collar layers must be at least one, got %r" % layers)
    if tori not in (2, 4):
        raise ValueError("--tori is two or four, got %r: one torus per input "
                         "state, or a conjugate pair each" % (tori,))

    if isinstance(states, str) and any(
            not part.strip() for part in states.split(":")):
        raise ValueError("--state contains an empty modulus; write every "
                         "input pair explicitly")
    pairs = _as_state_pairs(states)
    for pair_index, pair in enumerate(pairs, 1):
        for state_index, tau in enumerate(pair, 1):
            _modulus_value("--state pair %d modulus %d"
                           % (pair_index, state_index), tau)
    # Cases are set for extra input pairs and for the four-torus construction,
    # and nowhere else. With a single pair on two tori nothing is substituted,
    # so removing the substitution removes nothing and the flag would reduce to
    # --no-pin-boundary under a name that promises more. Say so rather than
    # accept it and quietly do something else.
    #
    # The `tori != 4` arm keeps the condition honest against the builder, which
    # sets cases at four tori with no --state at all. It is not offered as a
    # remedy below because no reading currently accepts four tori -- every
    # entry in _READOUT_TORI is either two-only or unavailable -- so a reader
    # sent that way would only meet a second refusal.
    if pin_boundary_state and not pairs and tori != 4:
        raise ValueError(
            "--pin-boundary-state needs more than one attached state: it "
            "removes the per-case boundary substitution, and a single input "
            "pair sets no cases and substitutes nothing. Declare the other "
            "states with --state -- or, if what you want is simply a boundary "
            "free to reshape within its conformal class, that is "
            "--no-pin-boundary, which is what this would reduce to here")

    output_tau = (None if output_state is None
                  else _state_value("--output-state", output_state))

    # Four tori structurally need two interior layers. Record the effective
    # value that the builder will use rather than a lower, ignored request.
    if tori == 4:
        layers = max(3, layers)

    # A reading that STRUCTURALLY cannot produce a number is refused here
    # rather than scoring a constant 1.0 forever. A constant term carries no
    # gradient, so the flag would silently be a no-op and the run would look
    # like it was optimizing something it was not.
    _validate_readout_host(names, tori)
    if "whole" in names and collar_twist == "swap":
        raise ValueError("--collar-twist swap cannot be used with --readout "
                         "whole: their harmonic frames do not agree")
    if output_tau is not None and "whole" not in names:
        raise ValueError(
            "--output-state can only be used when --readout includes whole")
    if output_tau is not None and pairs:
        raise ValueError(
            "--output-state cannot be used with --state: one declared output "
            "cannot be the answer for several different inputs")
    # A modulus names a 2-vector. It cannot name a vector in the four-torus
    # direct sum, independently of that host's tensor-product obstruction.
    if output_tau is not None and tori == 4:
        raise ValueError(
            "--output-state cannot be used with --tori 4: a modulus names a "
            "2-dimensional state, while that harmonic space has rank 4")
    if "whole" in names and tori == 2 and output_tau is None:
        raise ValueError(
            "--readout whole at --tori 2 needs --output-state: the harmonic of "
            "the whole Laplacian has rank 2 there, so it carries a "
            "2-dimensional state, and the two-body target it would otherwise "
            "score against is 4-dimensional")
    if names == ["whole"] and tori == 2 and output_tau is not None:
        ignored = []
        if operator != DECLARED_OPERATOR:
            ignored.append("--operator")
        if coupling != DECLARED_COUPLING:
            ignored.append("--J")
        if time != DECLARED_TIME:
            ignored.append("--time")
        if ignored:
            raise ValueError(
                "--readout whole with a declared --output-state does not use "
                "%s; non-default values would be ignored"
                % ", ".join(ignored))
    if operator != "flip_flop" and (
            coupling != DECLARED_COUPLING or time != DECLARED_TIME):
        raise ValueError("--J and --time apply only to --operator flip_flop; "
                         "the named gate %s ignores them" % operator)

    config.update({
        "readout": ",".join(names),
        "tori": tori,
        "collar_twist": collar_twist,
        "interior_disposition": interior_disposition,
        "whole_pairing": whole_pairing,
        "output_state": (None if output_tau is None
                         else [output_tau.real, output_tau.imag]),
        "register_degrees": list(DECLARED_REGISTER_DEGREES),
        "hodge_degrees": list(DECLARED_HODGE_DEGREES),
        "degrees": list(DECLARED_ANALYSIS_DEGREES),
        "betti_degrees": list(DECLARED_QUBIT_BETTI_DEGREES),
        # The qubit parameters (spec D4). The moduli are [re, im] pairs so
        # the config stays JSON as it is written.
        "inputs": DECLARED_INPUTS,
        "tau_a": moduli["tau_a"],
        "tau_b": moduli["tau_b"],
        "grid": grid,
        "layers": layers,
        "coupling": coupling,
        "time": time,
        "input_weight": input_weight,
        "regge": regge,
        "extend_boundary": extend_boundary,
        "score_leak": score_leak,
        # An empty LIST when none were given, as it has always been: a record
        # written before this flag existed reads the same.
        "states": ([] if states is None
                   else list(states) if isinstance(states, (list, tuple))
                   else str(states)),
        "operator": operator,
        "pin_boundary": pin_boundary,
        "pin_boundary_state": pin_boundary_state,
    })
    return config



# ---- verify: the cobordism functor, read on a fixed complex -------------

#: The jitter the documented invariance was measured at: every squared length
#: times `1 + F xi`, `xi` a seeded draw from the unit disc.
DECLARED_VERIFY_JITTER = 0.75
#: "Exact", for a read of a fixed complex.
DECLARED_VERIFY_TOLERANCE = 1e-12
#: Where `verify` writes its record. Never /tmp, which is wiped at boot.
DECLARED_VERIFY_OUT = "~/cobordism-runs/functor-verify"

#: The matrix each `--collar-twist` induces on `H^1` in the marked basis:
#: the product collar is the identity, the swap `(i, j) -> (j, i)` exchanges
#: the two marking cycles (reading P23-P24).
_TWIST_MATRICES = {"none": [[1, 0], [0, 1]], "swap": [[0, 1], [1, 0]]}


def _dense(matrix):
    """A bound matrix -- scipy sparse or dense -- as a numpy array."""
    import numpy as np
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _numeric_rank(matrix):
    """Rank by SVD under `MultiCobordism.monodromy`'s tolerance rule."""
    import numpy as np
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    if singular.size == 0:
        return 0
    return int((singular > 1e-9 * max(1.0, float(singular[0]))).sum())


def _fit_monodromy(periods_a, periods_b):
    """`M` with `P_B = M P_A` (least squares, exact when `P_A` is invertible),
    its integer rounding, the rounding residual, and the fit residual: the
    same numbers `MultiCobordism.monodromy` reports, from given periods."""
    import numpy as np
    periods_a = np.asarray(periods_a)
    periods_b = np.asarray(periods_b)
    matrix = np.linalg.lstsq(periods_a.T, periods_b.T, rcond=None)[0].T
    rounded = np.rint(matrix.real).astype(int)
    rounding = float(np.abs(matrix - rounded).max())
    fit = float(np.linalg.norm(matrix @ periods_a - periods_b)
                / max(np.linalg.norm(periods_b), 1e-300))
    return matrix, rounded, rounding, fit


def _intersection_form(cycles):
    """`J (+) ... (+) J` for `cycles` marking cycles, two per torus."""
    import numpy as np
    j = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return np.kron(np.eye(cycles // 2), j)


def _twist_orientation(twist):
    """The sign of the twist's action on a surface's orientation: `det phi*`.
    The swap exchanges the two marking cycles, reversing orientation."""
    import numpy as np
    return int(round(float(np.linalg.det(np.asarray(_TWIST_MATRICES[twist], dtype=float)))))


def _boundary_form(cycles_a, cycles_b, twist, flipped=False):
    """The boundary intersection form `Omega = J_A (+) s J_B` from DECLARED
    data, not from `M`: the A-side tori carry the bulk's induced orientation
    (`+1`), the B-side tori the opposite (`-1`) times the orientation of the
    relabelling applied to them (`det phi*`: `+1` for none, `-1` for swap).
    `flipped` negates the B-side signs, the control that must not vanish."""
    import numpy as np
    sign = -_twist_orientation(twist)
    if flipped:
        sign = -sign
    omega_a = _intersection_form(cycles_a)
    omega_b = sign * _intersection_form(cycles_b)
    return np.block([[omega_a, np.zeros((omega_a.shape[0], omega_b.shape[1]))],
                     [np.zeros((omega_b.shape[0], omega_a.shape[1])), omega_b]])


def _orthonormal_image(matrix):
    """An orthonormal basis `Q` of the column space of `matrix` (`Q^H Q = I`),
    its numerical rank under `monodromy`'s tolerance rule, the singular
    values, and the tolerance used. Normalisation only: the SUBSPACE is what
    is returned, the pairing evaluated on it stays bilinear."""
    import numpy as np
    matrix = np.asarray(matrix)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = 1e-9 * max(1.0, largest)
    rank = int((singular > threshold).sum())
    return u[:, :rank], rank, singular, threshold


def _lagrangian_check(label, periods_a, periods_b, twist, tolerance):
    """R5: the restriction SUBSPACE `im [P_A; P_B]` is isotropic for the
    declared boundary form, and not for the control with the B-side
    orientations flipped.

    Evaluated on an orthonormal basis `Q` of the subspace, so the verdict is
    a property of the subspace: rescaling the harmonic basis by `s` rescales
    `|L^T Omega L|` by `|s|^2` against a fixed tolerance, and `P = 1e-7 I`
    was failing the control for that reason alone. The pairing is the
    bilinear `Q^T Omega Q`, not `Q^H Omega Q`; the Hermitian SVD is only how
    `Q` is normalised. A rank-deficient stack is refused by name here rather
    than scored, since isotropy of the wrong subspace means nothing (R2
    carries the rank verdict).

    The passing sign is not searched for and not derived from `det M`: the
    2x2 identity `M^T J M = det(M) J` does not extend to the combined map
    (`diag(S, S)` has `det +1` and reverses the form), so the form is built
    from the sides and the twist alone.
    """
    import numpy as np
    stacked = np.vstack([periods_a, periods_b])
    basis, rank, singular, threshold = _orthonormal_image(stacked)
    cycles_a, cycles_b = periods_a.shape[0], periods_b.shape[0]
    # The Lagrangian dimension: half the boundary's b_1 (the reading's
    # half-lives-half-dies). For one collar of a four-torus host the stack is
    # 4 cycles by 4 harmonic columns and the pair's restriction subspace is
    # 2-dimensional; the whole's is 8 by 4 and 4-dimensional.
    expected = (cycles_a + cycles_b) // 2
    conditioning = float(singular[0] / singular[rank - 1]) if rank else float("inf")
    if rank != expected:
        return _check(
            "R5:" + label,
            "the restriction subspace is isotropic for the declared boundary form",
            {"refusal": "the stacked periods have rank %d against the Lagrangian dimension %d "
                        "(half the %d boundary cycles); isotropy is not evaluated on the wrong "
                        "subspace" % (rank, expected, cycles_a + cycles_b),
             "singular_values": [float(x) for x in singular], "rank_tolerance": threshold},
            tolerance, False)
    declared = float(np.linalg.norm(
        basis.T @ _boundary_form(cycles_a, cycles_b, twist) @ basis))
    flipped = float(np.linalg.norm(
        basis.T @ _boundary_form(cycles_a, cycles_b, twist, flipped=True) @ basis))
    return _check(
        "R5:" + label,
        "the restriction subspace is isotropic for the declared boundary form "
        "J_A (+) (-det phi*) J_B, and not for the B-side-flipped control",
        {"declared_form_norm": declared, "flipped_control_norm": flipped,
         "b_side_sign": -_twist_orientation(twist), "rank": rank,
         "rank_tolerance": threshold, "conditioning": conditioning},
        tolerance, declared <= tolerance and flipped > tolerance)


def _harmonic_certificates(label, read, op, tolerance):
    """H1-H3: the band `restriction` returned IS the degree-1 harmonic space.

    Rank equal to `b_1` (R1) is necessary and not sufficient: on complex
    lengths the operator is not Hermitian, and a contour band can hold
    generalised or near-zero modes that solve neither harmonic equation.
    In the chain formulation (`ChainHodge`, dressed by "the transpose
    carries U") the equations are, for the cochain image `Z = G_1^U Phi` of
    the chain frame `Phi`:

        closed      (d_2^{U^-1})^T Z = 0     -- on the images
        co-closed   d_1^U Phi = 0            -- on the frame

    each measured as a relative residual (H1). The kernel is then read
    independently of the contour (`harmonicChains`): its nullity must equal
    the band's rank and `b_1`, and the two spans must agree (H2). The band's
    own certificate -- projector idempotency, rank tolerance, singular gap,
    resolvent bound -- is surfaced and must be finite (H3). A kernel-dimension
    mismatch is the geometry failing `dim ker L_1 = b_1`, the reading's
    condition, and is named as that.
    """
    import numpy as np
    images = np.asarray(read.images)
    frame = np.asarray(read.frame)
    rows = []
    try:
        d_dual_2 = _dense(op.twistedBoundaryDual(2))
        d_1 = _dense(op.twistedBoundary(1))
    except Exception as error:                            # noqa: BLE001
        rows.append(_check("H1:" + label, "closed and co-closed residuals of the band",
                           {"refusal": "boundary operators unavailable: %s" % error},
                           tolerance, False))
        return rows, {}
    if d_dual_2.shape[0] != images.shape[0] or d_1.shape[1] != frame.shape[0]:
        rows.append(_check("H1:" + label, "closed and co-closed residuals of the band",
                           {"refusal": "shape mismatch: d2 %s, d1 %s, images %s, frame %s"
                                       % (d_dual_2.shape, d_1.shape, images.shape, frame.shape)},
                           tolerance, False))
        return rows, {}
    closed = float(np.linalg.norm(d_dual_2.T @ images) / max(np.linalg.norm(images), 1e-300))
    coclosed = float(np.linalg.norm(d_1 @ frame) / max(np.linalg.norm(frame), 1e-300))
    rows.append(_check(
        "H1:" + label,
        "every column is closed ((d_2^{U^-1})^T Z = 0) and co-closed (d_1^U Phi = 0)",
        {"closed_residual": closed, "coclosed_residual": coclosed},
        tolerance, closed <= tolerance and coclosed <= tolerance))
    try:
        kernel = op.harmonicChains(1)
        kernel_images = np.asarray(kernel.images)
        nullity = int(kernel.nullity)
        gap = float(kernel.gap)
    except Exception as error:                            # noqa: BLE001
        rows.append(_check("H2:" + label, "an independent kernel read agrees with the band",
                           {"refusal": "harmonicChains unavailable: %s" % error},
                           tolerance, False))
        nullity, gap, kernel_images = None, float("nan"), None
    if kernel_images is not None:
        forward = float(np.linalg.norm(kernel_images - images @ np.linalg.lstsq(images, kernel_images, rcond=None)[0])
                        / max(np.linalg.norm(kernel_images), 1e-300))
        backward = float(np.linalg.norm(images - kernel_images @ np.linalg.lstsq(kernel_images, images, rcond=None)[0])
                         / max(np.linalg.norm(images), 1e-300))
        b1 = int(read.betti[1]) if len(read.betti) > 1 else None
        measured = {"kernel_nullity": nullity, "band_rank": int(read.harmonic_rank), "b1": b1,
                    "span_residual_kernel_in_band": forward, "span_residual_band_in_kernel": backward,
                    "kernel_gap": gap}
        if nullity != read.harmonic_rank or nullity != b1:
            measured["failure"] = ("dim ker L_1 = %s against b_1 = %s and band rank %d: the geometry "
                                   "fails the condition dim ker L_1 = b_1" % (nullity, b1, read.harmonic_rank))
        rows.append(_check(
            "H2:" + label,
            "an independent kernel read has nullity = band rank = b_1 and the same span",
            measured, tolerance,
            nullity == read.harmonic_rank == b1 and forward <= tolerance and backward <= tolerance))
    cert = read.certificate
    fields = {"node_count": int(cert.nodeCount), "idempotency": float(cert.idempotency),
              "rank": int(cert.rank), "rank_tolerance": float(cert.rankTolerance),
              "singular_gap": float(cert.singularGap), "resolvent_max": float(cert.resolventMax)}
    finite = all(math.isfinite(v) for v in fields.values() if isinstance(v, float))
    rows.append(_check(
        "H3:" + label,
        "the band's own certificate is finite and its projector is idempotent",
        fields, tolerance, finite and fields["idempotency"] <= tolerance))
    return rows, fields


def _exactness_check(delta, reference, coboundary, tolerance):
    """J2: the canonical representative moves by O(1) under the jitter and
    the move is exact -- in `im d_0` -- to the declared tolerance.

    Measured as the exactness RESIDUAL `|delta - d_0 c| / |delta|` with `c` the
    least-squares fit, never as the projected-length ratio `|d_0 c| / |delta|`:
    that ratio sits within `5e-11` of one for `delta = (1, 1e-5)` against
    `span((1, 0))` while the residual is `1e-5`. No motion at all is named,
    and fails, since the check is that the representative moves.
    """
    import numpy as np
    norm = float(np.linalg.norm(delta))
    if norm == 0.0:
        return _check("J2", "the canonical representative moves O(1) and the move is exact (in im d_0)",
                      {"relative_move": 0.0, "exactness_residual": None,
                       "note": "the representative did not move"}, tolerance, False)
    relative = norm / float(np.linalg.norm(reference))
    coefficients = np.linalg.lstsq(coboundary, delta, rcond=None)[0]
    residual = float(np.linalg.norm(delta - coboundary @ coefficients) / norm)
    return _check("J2", "the canonical representative moves O(1) and the move is exact (in im d_0)",
                  {"relative_move": relative, "exactness_residual": residual},
                  tolerance, relative > 1e-3 and residual <= tolerance)


def _read_restriction(spacetime, markings):
    """`MultiCobordism.restriction`, refusing by name rather than guessing.
    Returns the unpacked read and the read itself, whose frame and
    certificate the harmonic checks need."""
    import numpy as np
    read = MC.restriction(spacetime, markings)
    if read.obstruction:
        raise RuntimeError("restriction refused: %s" % read.obstruction)
    return (list(int(b) for b in read.betti), int(read.harmonic_rank),
            np.asarray(read.images), [np.asarray(p) for p in read.periods], read)


def _jitter_lengths(spacetime, fraction, seed):
    """Every squared length times `1 + fraction * xi`, `xi` drawn from the
    unit disc with `seed`; returns the restorer. The stored length moves on
    its own sheet: `l -> l sqrt(1 + fraction xi)` with the principal root,
    continuous because `1 + fraction xi` has positive real part."""
    import cmath
    import numpy as np
    rng = np.random.default_rng(int(seed))
    original = []
    for edge in spacetime.getEdgeList().toVector():
        length = complex(edge.getLength())
        original.append((edge, length))
        radius = math.sqrt(rng.random())
        angle = 2.0 * math.pi * rng.random()
        xi = complex(radius * math.cos(angle), radius * math.sin(angle))
        edge.setLength(length * cmath.sqrt(1.0 + fraction * xi))

    def restore():
        for edge, length in original:
            edge.setLength(length)
    return restore


def _copy_lengths_to_torus(host, torus_spacetime, ids):
    """The host's live lengths onto the torus's own edges through its id map,
    so the torus's own operator is assembled on the lengths the bulk carries
    at this instant."""
    live = {}
    for edge in host.getEdgeList().toVector():
        u = int(edge.getSource().getId())
        v = int(edge.getTarget().getId())
        live[(min(u, v), max(u, v))] = complex(edge.getLength())
    for edge in torus_spacetime.getEdgeList().toVector():
        a = int(edge.getSource().getId())
        b = int(edge.getTarget().getId())
        if a not in ids or b not in ids:
            raise RuntimeError("torus vertex %d/%d has no host id" % (a, b))
        key = (min(ids[a], ids[b]), max(ids[a], ids[b]))
        if key not in live:
            raise RuntimeError("torus edge %s is not a host edge" % (key,))
        edge.setLength(live[key])


def _covariant_period_gram(op, contour, walks):
    """The boundary pairing in the coordinates of a marking, gauge covariant:
    `G = (P^v)^-T (Phi~^T M Z) P^-1 = (P^v)^-T P^-1`, with `Z` the band's
    cochain images, `Phi~` its native left frame -- the covariant dual
    partner, normalised by `Phi~^T M Z = I` (the `(F^v)^T M_own F = I` rule
    of spec S4) -- `P` the transported periods of `Z` on the connection and
    `P^v` those of `Phi~` on the dual connection, both over the SAME walks
    so the base-point factors cancel.

    A primal-primal `Z^T M Z` transforms by `rho^T G rho` under a vertex
    gauge (`CovariantChainHodge`, property (vi)) and is not covariant:
    measured 0.76 relative movement under a U(1) gauge on the seeded torus,
    against 1e-15 for this pairing. At zero phases the two agree exactly,
    since there `Phi~ = Z B^-T` with `B = Phi^T Z`. Bilinear throughout.
    """
    import numpy as np
    band = op.band(1, contour)
    images = np.asarray(band.images)
    left = np.asarray(band.leftFrame)
    if left.size == 0 or left.shape != images.shape:
        raise RuntimeError("the band has no left frame (isotropic pairing): the covariant "
                           "Gram is not defined")
    dual = op.dual()
    periods = np.array([[op.connection().transportedPeriod(images[:, a], walk)
                         for a in range(images.shape[1])] for walk in walks])
    dual_periods = np.array([[dual.connection().transportedPeriod(left[:, a], walk)
                              for a in range(left.shape[1])] for walk in walks])
    if periods.shape[0] != periods.shape[1]:
        raise RuntimeError("the marking has %d cycles against a rank-%d band"
                           % (periods.shape[0], periods.shape[1]))
    return np.linalg.inv(dual_periods).T @ np.linalg.inv(periods)


def _torus_gram(torus, host, ids):
    """The torus's own boundary pairing in the coordinates of its marking, on
    the host's live lengths (`_covariant_period_gram` on the torus's own
    operator)."""
    spacetime = torus.spacetime()
    _copy_lengths_to_torus(host, spacetime, ids)
    marking = _host_marking(torus, {int(v): int(v)
                                    for pair in torus.edges() for v in pair})
    assembled = cob.PencilLayer.assemble([spacetime])
    contour = cob.PencilLayer.harmonic_contour(assembled, 1)
    walks = [[(int(u), int(v)) for u, v in cycle] for cycle in marking]
    return _covariant_period_gram(assembled.op, contour, walks)


def _unitarity_defect(monodromy, gram_a, gram_b):
    """`|M^T G_B M - G_A| / |G_A|`: whether the monodromy carries the bilinear
    Whitney pairing of one end to the other. Zero is the metric condition;
    nothing in the topology implies it."""
    import numpy as np
    return float(np.linalg.norm(monodromy.T @ gram_b @ monodromy - gram_a)
                 / max(np.linalg.norm(gram_a), 1e-300))


def _check(identifier, statement, measured, tolerance, passed):
    return {"id": identifier, "statement": statement, "measured": measured,
            "tolerance": tolerance, "pass": bool(passed)}


def _pair_checks(label, periods_a, periods_b, twist, tolerance, expect_twist=True, expected=None):
    """R2-R5 for one marked pair, and the fitted monodromy for the rest.
    `twist` names the relabelling applied to the B side; `expect_twist`
    says whether R4 compares `M` with that twist's induced map (a single
    collar) or not (the combined four-torus map); `expected` overrides the
    map R4 compares with (the flip passes' prediction)."""
    import numpy as np
    checks = []
    matrix, rounded, rounding, fit = _fit_monodromy(periods_a, periods_b)
    cycles = periods_a.shape[0]
    ranks = (_numeric_rank(periods_a), _numeric_rank(periods_b),
             _numeric_rank(np.vstack([periods_a, periods_b])))
    checks.append(_check(
        "R2:" + label,
        "rank P_A = rank P_B = rank [P_A; P_B] = %d, half of b_1 of the two ends" % cycles,
        {"rank_a": ranks[0], "rank_b": ranks[1], "rank_stacked": ranks[2]},
        None, ranks == (cycles, cycles, cycles)))
    det = int(round(float(np.linalg.det(rounded)))) if rounded.size else 0
    checks.append(_check(
        "R3:" + label, "M is an integer matrix with det +-1",
        {"rounding_residual": rounding, "fit_residual": fit, "det": det},
        tolerance, rounding <= tolerance and fit <= tolerance and det in (1, -1)))
    if expect_twist:
        if expected is None:
            expected = np.asarray(_TWIST_MATRICES[twist])
        expected = np.asarray(expected)
        checks.append(_check(
            "R4:" + label, "M equals the twist's induced map phi* (%s)" % twist,
            {"rounded": rounded.tolist(), "expected": expected.tolist(),
             "difference": float(np.abs(matrix - expected).max())},
            tolerance, rounded.shape == expected.shape
            and (rounded == expected).all()
            and float(np.abs(matrix - expected).max()) <= tolerance))
    checks.append(_lagrangian_check(label, periods_a, periods_b, twist, tolerance))
    return checks, matrix, rounded


def _labelled(pairs, labels):
    return ["%s->%s" % (labels[a], labels[b]) for a, b in pairs]


def _far_torus_modulus(host, torus, far_ids):
    """The flipped far torus's own modulus in the relabelled marking: its
    boundary triangles on the far vertices with the host's lengths, read by
    `SimplicialQubit` (section 6-9) with the standard walks of the
    relabelled grid as the marking."""
    import numpy as np
    far = set(far_ids.values())
    faces = []
    from collections import Counter
    count = Counter()
    for cell in host.getTopSimplices():
        ids_ = sorted(int(v.getId()) for v in cell.getVertices())
        for skip in range(4):
            count[tuple(v for n, v in enumerate(ids_) if n != skip)] += 1
    faces = [face for face, c in count.items() if c == 1 and set(face) <= far]
    # SimplicialQubit wants consistently oriented faces: propagate the
    # counterclockwise orientation of the relabelled grid's face at (0, 0)
    # across shared edges (each edge traversed oppositely by its two faces).
    n = int(round(len(far_ids) ** 0.5))
    root = tuple(far_ids[(x % n) * n + (y % n)] for x, y in ((0, 0), (1, 0), (1, 1)))
    from collections import defaultdict, deque
    by_edge = defaultdict(list)
    for index, face in enumerate(faces):
        for a in range(3):
            u, v = face[a], face[(a + 1) % 3]
            by_edge[(min(u, v), max(u, v))].append(index)
    oriented = [None] * len(faces)
    start = next(index for index, face in enumerate(faces) if set(face) == set(root))
    oriented[start] = root
    queue = deque([start])

    def traverses(order, u, v):
        return any(order[a] == u and order[(a + 1) % 3] == v for a in range(3))
    while queue:
        index = queue.popleft()
        face = oriented[index]
        for a in range(3):
            u, v = face[a], face[(a + 1) % 3]
            for other in by_edge[(min(u, v), max(u, v))]:
                if other == index or oriented[other] is not None:
                    continue
                order = list(faces[other])
                if not traverses(order, v, u):
                    order = [order[0], order[2], order[1]]
                oriented[other] = tuple(order)
                queue.append(other)
    faces = oriented
    lengths = {}
    for edge in host.getEdgeList().toVector():
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        lengths[(min(u, v), max(u, v))] = complex(edge.getLength())
    vertices = sorted(far)
    index = {v: n for n, v in enumerate(vertices)}
    edges = sorted({(min(index[u], index[v]), max(index[u], index[v])) for face in faces for u in face for v in face if u != v})
    edge_index = {e: n for n, e in enumerate(edges)}
    edge_lengths = [lengths[(vertices[e[0]], vertices[e[1]])] for e in edges]
    local_faces = [tuple(index[v] for v in face) for face in faces]

    def cycle(steps):
        out = []
        for u, v in steps:
            a, b = index[int(u)], index[int(v)]
            out.append((edge_index[(min(a, b), max(a, b))], 1 if a < b else -1))
        return out
    marking = _host_marking(torus, far_ids)
    read = obs.SimplicialQubit(vertices=list(range(len(vertices))), edges=edges, faces=local_faces,
                               lengths=edge_lengths, cycle_A=cycle(marking[0]), cycle_B=cycle(marking[1]))
    return complex(read.tau()), len(faces)


def _flip_checks(config, tori, host, ids, markings, monodromy, lattice, predicted, tolerance):
    """F1-F4 on the flipped collar."""
    import numpy as np
    passes = _far_flip_passes(config)
    n = int(config["grid"])
    layers = int(config["layers"])
    checks = []
    rounded = np.asarray(monodromy["rounded"], dtype=int)
    betti = MC.betti(host)
    cells = len(host.getTopSimplices())
    expected_cells = 6 * n * n * layers + n * n * len(passes)
    checks.append(_check(
        "F1", "the layered collar: %d tetrahedra per pass on the far torus, a manifold with betti [1, 2, 1, 0]" % (n * n),
        {"cells": cells, "expected_cells": expected_cells, "betti": [int(b) for b in betti], "passes": passes},
        None, cells == expected_cells and [int(b) for b in betti] == [1, 2, 1, 0]))
    order = None
    power = np.eye(2, dtype=int)
    for k in range(1, 13):
        power = power @ rounded
        if (power == np.eye(2, dtype=int)).all():
            order = k
            break
    det = int(round(float(np.linalg.det(rounded))))
    twist = str(config.get("collar_twist", DECLARED_COLLAR_TWIST))
    expected_order = None
    if twist == "none":
        expected_order = 4 if all(name == "s" for name in passes) and len(passes) % 2 == 1 else None
    checks.append(_check(
        "F2", "M is the predicted lattice map L^T (times the twist), det +1 for a twist, and of infinite order for a "
              "Dehn twist (M^k != I, k <= 12)",
        {"monodromy": rounded.tolist(), "predicted": predicted.tolist(), "lattice": lattice.tolist(),
         "det": det, "order": order if order is not None else "infinite (> 12)",
         "difference": float(np.abs(np.asarray(monodromy["matrix"]) - predicted).max())},
        tolerance, (rounded == predicted).all() and det == (1 if twist == "none" else -1)
        and (order is None if not all(name == "s" for name in passes) else order in (1, 2, 4))))
    tau_far = complex(tori[1].tau())
    if det == 1:
        m = predicted
        expected_tau = (m[1, 0] + m[1, 1] * tau_far) / (m[0, 0] + m[0, 1] * tau_far)
        read_tau, faces = _far_torus_modulus(host, tori[1], ids[1])
        checks.append(_check(
            "F3", "the flipped far torus's own modulus in the relabelled marking is the far modulus moved by M: "
                  "tau' = (M_21 + M_22 tau) / (M_11 + M_12 tau)",
            {"read": read_tau, "expected": expected_tau, "far_modulus": tau_far, "faces": faces,
             "distance": abs(read_tau - expected_tau)},
            1e-9, abs(read_tau - expected_tau) <= 1e-9))
    else:
        checks.append(_check("F3", "the far modulus check is made for orientation-preserving relabellings only",
                             {"det": det}, None, True))
    # F4: composition order on one host -- the passes in both orders.
    if len(passes) >= 2 or (len(passes) == 1 and twist == "swap"):
        def read(order_names, twist_name):
            other = dict(config)
            other["far_flips"] = ",".join(order_names)
            other["collar_twist"] = twist_name
            _, _, seed_, _, markings_ = seed_qubit_host(other)
            _, _, _, periods_, _ = _read_restriction(seed_.host, markings_)
            return _fit_monodromy(periods_[0], periods_[1])[1]
        forward, backward = list(passes), list(reversed(passes))
        m_forward = read(forward, twist)
        m_backward = read(backward, twist)
        singles = {}
        for name in set(passes):
            singles[name] = read([name], "none")
        product_forward = np.eye(2, dtype=int)
        for name in forward:
            product_forward = singles[name] @ product_forward
        product_backward = np.eye(2, dtype=int)
        for name in backward:
            product_backward = singles[name] @ product_backward
        twist_matrix = np.array(_TWIST_MATRICES[twist], dtype=int)
        product_forward = product_forward @ twist_matrix
        product_backward = product_backward @ twist_matrix
        differ = int(np.abs(m_forward - m_backward).max()) if forward != backward else None
        checks.append(_check(
            "F4", "composition on one host: M(passes) = M(last) ... M(first) . M(twist), and the reversed order "
                  "gives a different matrix when the passes do not commute",
            {"forward": forward, "M_forward": m_forward.tolist(), "product_forward": product_forward.tolist(),
             "backward": backward, "M_backward": m_backward.tolist(), "product_backward": product_backward.tolist(),
             "orders_differ_by": differ, "twist": twist},
            None, (m_forward == product_forward).all() and (m_backward == product_backward).all()
            and (differ is None or differ >= 1 or (singles and all(name == passes[0] for name in passes)))))
    return checks


def verify(config, jitter=DECLARED_VERIFY_JITTER,
           tolerance=DECLARED_VERIFY_TOLERANCE):
    """Read the seeded host and check the cobordism functor on it.

    R1-R5 are the restriction theorem and the monodromy's integrality on the
    fixed complex; J1-J3 the metric independence of `M` against the metric
    dependence of the representative and its Gram under a jitter of every
    length; C1 the composition law in the twist group across three seeded
    hosts; U1 the metric-dependent transport, reported and shown to move.
    Every row carries its measured value; nothing is optimised.
    """
    import numpy as np
    if str(config.get("surface", DECLARED_SURFACE)) == "decagon":
        raise ValueError("verify reads the torus hosts; the genus-2 decagon collar is read by `theta --surface "
                         "decagon` (checks E1-E6)")
    if str(config.get("join", DECLARED_JOIN)) == "tube":
        raise ValueError("verify reads the sphere-joined hosts, whose four boundary components are tori; "
                         "the tube-joined host, whose far boundary is one genus-2 surface, is read by "
                         "`theta` (checks G1-G4)")
    twist = str(config.get("collar_twist", DECLARED_COLLAR_TWIST))
    layers = int(config["layers"])
    tori, _, seed, ids, markings = seed_qubit_host(config)
    host = seed.host
    labels = (DECLARED_CONJUGATE_TORUS_LABELS if len(tori) > 2
              else DECLARED_TORUS_LABELS)[:len(tori)]
    pairs = [(0, 1)] if len(tori) == 2 else [(0, 1), (2, 3)]
    names = _labelled(pairs, labels)
    checks = []
    values = {"tori": len(tori), "layers": layers, "collar_twist": twist,
              "grid": int(config["grid"]), "seed": int(config["seed"]),
              "cells": len(host.getTopSimplices()),
              "edges": len(host.getEdgeList().toVector())}

    # ---- R1: the whole's zero mode is its first cohomology ------------
    betti, rank, images, periods, read = _read_restriction(host, markings)
    boundary_b1 = sum(int(p.shape[0]) for p in periods)
    values.update({"betti": betti, "harmonic_rank": rank,
                   "boundary_b1": boundary_b1})
    checks.append(_check(
        "R1", "harmonic rank = b_1(W) = b_1(dW) / 2",
        {"harmonic_rank": rank, "b1": betti[1] if len(betti) > 1 else None,
         "boundary_b1": boundary_b1},
        None, len(betti) > 1 and rank == betti[1] == boundary_b1 // 2))

    # ---- H1-H3: the band is the harmonic space, certified ----------------
    assembled = cob.PencilLayer.assemble([host])
    rows, values["band_certificate"] = _harmonic_certificates("seed", read, assembled.op, tolerance)
    checks.extend(rows)

    # ---- R2-R5 per marked pair -----------------------------------------
    monodromies = {}
    passes = _far_flip_passes(config)
    lattice, predicted = _flip_lattice_map(config)
    for (a, b), name in zip(pairs, names):
        rows, matrix, rounded = _pair_checks(name, periods[a], periods[b],
                                             twist, tolerance, expected=predicted)
        checks.extend(rows)
        monodromies[name] = {"matrix": matrix, "rounded": rounded}
    if passes:
        checks.extend(_flip_checks(config, tori, host, ids, markings, monodromies[names[0]],
                                   lattice, predicted, tolerance))
        values["far_flips"] = list(passes)
        values["flip_lattice"] = lattice.tolist()
    values["monodromy"] = {name: {"matrix": [[complex(z) for z in row]
                                             for row in m["matrix"]],
                                  "rounded": m["rounded"].tolist()}
                           for name, m in monodromies.items()}

    # ---- the direct sum, at four tori ---------------------------------
    a_side = [0, 2] if len(tori) == 4 else [0]
    b_side = [1, 3] if len(tori) == 4 else [1]
    periods_a = np.vstack([periods[i] for i in a_side])
    periods_b = np.vstack([periods[i] for i in b_side])
    if len(tori) == 4:
        rows, whole, whole_rounded = _pair_checks("whole", periods_a, periods_b,
                                                  twist, tolerance, expect_twist=False)
        checks.extend(rows)
        blocks = [monodromies[name]["matrix"] for name in names]
        block_diagonal = np.block([[blocks[0], np.zeros_like(blocks[0])],
                                   [np.zeros_like(blocks[1]), blocks[1]]])
        off = float(np.linalg.norm(whole - block_diagonal))
        checks.append(_check(
            "D1", "the four-torus monodromy is the direct sum of the two collars'",
            {"off_block_norm": off, "rounded": whole_rounded.tolist()},
            tolerance, off <= tolerance))
        values["monodromy"]["whole"] = {
            "matrix": [[complex(z) for z in row] for row in whole],
            "rounded": whole_rounded.tolist()}

    # ---- J1-J3: jitter every length -----------------------------------
    coboundary = _dense(assembled.op.twistedBoundary(1))
    if coboundary.shape[0] != images.shape[0]:
        coboundary = coboundary.T
    canonical_before = images @ np.linalg.inv(periods_a)
    # J3's Gram is the whole band's covariant pairing in the A-side marking
    # coordinates -- the same dual-frame rule as U1 -- not Z^T M Z.
    a_walks = [[(int(u), int(v)) for u, v in cycle] for i in a_side for cycle in markings[i]]
    gram_before = _covariant_period_gram(
        assembled.op, cob.PencilLayer.harmonic_contour(assembled, 1), a_walks)
    grams_before = {index: _torus_gram(torus, host, ids[index])
                    for index, torus in enumerate(tori)}
    restore = _jitter_lengths(host, float(jitter), int(config["seed"]))
    try:
        _, rank_after, images_after, periods_after, read_after = _read_restriction(host, markings)
        assembled_after = cob.PencilLayer.assemble([host])
        for (a, b), name in zip(pairs, names):
            matrix_after, _, _, _ = _fit_monodromy(periods_after[a], periods_after[b])
            move = float(np.abs(matrix_after - monodromies[name]["matrix"]).max())
            checks.append(_check(
                "J1:" + name, "the jitter leaves M fixed",
                {"max_entry_move": move, "rank_after": rank_after},
                tolerance, move <= tolerance and rank_after == rank))
        rows, values["band_certificate_jittered"] = _harmonic_certificates(
            "jittered", read_after, assembled_after.op, tolerance)
        checks.extend(rows)
        periods_a_after = np.vstack([periods_after[i] for i in a_side])
        canonical_after = images_after @ np.linalg.inv(periods_a_after)
        delta = canonical_after - canonical_before
        checks.append(_exactness_check(delta, canonical_before, coboundary, tolerance))
        gram_after = _covariant_period_gram(
            assembled_after.op, cob.PencilLayer.harmonic_contour(assembled_after, 1), a_walks)
        gram_move = float(np.linalg.norm(gram_after - gram_before)
                          / np.linalg.norm(gram_before))
        checks.append(_check(
            "J3", "the whole band's covariant pairing in the marking coordinates moves O(1)",
            {"relative_move": gram_move, "pairing_convention": "dual_frame_whitney"},
            None, gram_move > 1e-3))
        grams_after = {index: _torus_gram(torus, host, ids[index])
                       for index, torus in enumerate(tori)}
        for (a, b), name in zip(pairs, names):
            matrix = monodromies[name]["matrix"]
            before = _unitarity_defect(matrix, grams_before[a], grams_before[b])
            after = _unitarity_defect(matrix, grams_after[a], grams_after[b])
            # A diagnostic, not a theorem: the metric condition is reported,
            # and an isometric configuration (identical tori on the product
            # collar) is a legitimate result, not a failure.
            checks.append(_check(
                "U1:" + name,
                "the transport defect |M^T G_B M - G_A| / |G_A| before and after jitter, "
                "G the covariant dual-frame pairing in marking coordinates (the metric condition, reported)",
                {"before": before, "after": after, "move": abs(after - before),
                 "isometric_before": bool(before <= tolerance),
                 "pairing_convention": "dual_frame_whitney"},
                tolerance, math.isfinite(before) and math.isfinite(after)))
    finally:
        restore()
        for index, torus in enumerate(tori):
            _copy_lengths_to_torus(host, torus.spacetime(), ids[index])

    # ---- C1: composition in the twist group ---------------------------
    def monodromy_of(layer_count, twist_name):
        # The twist-group control is read on plain collars: no flip passes.
        other = dict(config)
        other["layers"] = int(layer_count)
        other["collar_twist"] = twist_name
        other["far_flips"] = ""
        _, _, other_seed, _, other_markings = seed_qubit_host(other)
        _, _, _, other_periods, _ = _read_restriction(other_seed.host, other_markings)
        return _fit_monodromy(other_periods[0], other_periods[1])[0]

    single = {name: monodromy_of(layers, name) for name in ("none", "swap")}
    double = {name: monodromy_of(2 * layers, name) for name in ("none", "swap")}
    products = {"none": single["swap"] @ single["swap"],
                "swap": single["swap"] @ single["none"]}
    for name in ("none", "swap"):
        move = float(np.abs(double[name] - products[name]).max())
        checks.append(_check(
            "C1:" + name,
            "M(2L, %s) = M(L, swap) . M(L, %s): gluing composes in the twist group"
            % (name, "swap" if name == "none" else "none"),
            {"glued": np.rint(double[name].real).astype(int).tolist(),
             "product": np.rint(products[name].real).astype(int).tolist(),
             "max_entry_move": move},
            tolerance, move <= tolerance))
    values["composition"] = {
        "single": {k: [[complex(z) for z in row] for row in v] for k, v in single.items()},
        "double": {k: [[complex(z) for z in row] for row in v] for k, v in double.items()}}
    return {"checks": checks, "values": values, "jitter": float(jitter),
            "tolerance": float(tolerance),
            "all_pass": all(row["pass"] for row in checks),
            "certifies": VERIFY_CERTIFIES}


#: What a passing `verify` certifies, and what it does not. The exact column
#: of the derivation's ledger, on this host; none of the conditional column.
VERIFY_CERTIFIES = (
    "On this seeded host: the contour band is the degree-1 harmonic space "
    "(closed and co-closed to tolerance, independent kernel of the same "
    "dimension and span, finite band certificate; the reading's condition "
    "dim ker L_1 = b_1 holds); the restriction subspace is Lagrangian for the "
    "declared boundary form; the monodromy is an integer matrix of det +-1 "
    "equal to the twist's induced map, unchanged by a jitter that moves the "
    "representative by O(1) exactly within im d_0; the twist group composes "
    "across separately seeded hosts; four tori give the direct sum. With "
    "layered flip passes on the far torus (F1-F4): the Dehn twists T_B, T_A "
    "and the quarter turn S are read as the predicted integer monodromies, of "
    "infinite order for the twists, with the flipped far torus's own modulus "
    "moved by M, and passes stacked on ONE host compose in order, the two "
    "orders of a non-commuting pair giving different matrices. "
    "NOT certified: any seam gluing between separately built cobordisms, "
    "quantum unitarity (the transport defect is the bilinear dual-frame pairing, "
    "gauge covariant, and reported), "
    "or a tensor-product register (four tori are a direct sum).")


def _print_verify_table(record):
    width = max(len(row["id"]) for row in record["checks"])
    for row in record["checks"]:
        measured = row["measured"]
        if isinstance(measured, dict):
            measured = ", ".join("%s=%s" % (k, _short(v)) for k, v in measured.items())
        print("%-*s  %-4s  %s\n%s      %s" % (
            width, row["id"], "pass" if row["pass"] else "FAIL", row["statement"],
            " " * width, measured))
    print("\n%s: %d of %d checks pass" % (
        "PASS" if record["all_pass"] else "FAIL",
        sum(row["pass"] for row in record["checks"]), len(record["checks"])))


def _short(value):
    if isinstance(value, float):
        return "%.3e" % value
    return str(value)


def _add_verify_arguments(verify_parser):
    verify_parser.add_argument("--tori", type=int, choices=(2, 4), default=DECLARED_TORI,
                               help="two tori (one collar) or four (two collars joined "
                                    "through a removed tetrahedron: the direct sum)")
    verify_parser.add_argument("--layers", type=int, default=DECLARED_COLLAR_LAYERS,
                               help="product layers of the collar seed; the composition "
                                    "check also seeds 2x this")
    verify_parser.add_argument("--collar-twist", dest="collar_twist",
                               choices=("none", "swap"), default=DECLARED_COLLAR_TWIST,
                               help="the mapping class the far surface is relabelled by")
    verify_parser.add_argument("--grid", type=int, default=DECLARED_GRID,
                               help="each torus is a grid x grid lattice")
    verify_parser.add_argument("--seed", type=int, default=DECLARED_SEED,
                               help="seed for the interior disposition and the jitter")
    verify_parser.add_argument("--tau-a", type=_complex_argument, default=DECLARED_TAU_A)
    verify_parser.add_argument("--tau-b", type=_complex_argument, default=DECLARED_TAU_B)
    verify_parser.add_argument("--far-flips", dest="far_flips", default="",
                               help="layered flip passes on the far torus, a comma-separated sequence of "
                                    "b (T_B: rows), a (T_A: columns), s (S: diagonals), applied in order")
    verify_parser.add_argument("--flip-factor", dest="flip_factor", type=float, default=DECLARED_FLIP_FACTOR,
                               help="the flipped, now interior, edge is shortened by this factor (default %g)"
                                    % DECLARED_FLIP_FACTOR)
    verify_parser.add_argument("--jitter", type=float, default=DECLARED_VERIFY_JITTER,
                               help="every squared length times 1 + F xi, xi from the "
                                    "unit disc (default %g)" % DECLARED_VERIFY_JITTER)
    verify_parser.add_argument("--tol", type=float, default=DECLARED_VERIFY_TOLERANCE,
                               help="what 'exact' means for a read of a fixed complex "
                                    "(default %g)" % DECLARED_VERIFY_TOLERANCE)
    verify_parser.add_argument("--out", default=DECLARED_VERIFY_OUT,
                               help="directory the JSON record is written under "
                                    "(default %s)" % DECLARED_VERIFY_OUT)
    verify_parser.add_argument("--json", default=None,
                               help="an explicit record path instead of --out/<name>.json")


def _verify_config(args):
    """The host seed, and only that: the keys `seed_qubit_host` reads.

    Not `build_config`: that validates a readout against the torus count,
    and every declared readout refuses four tori, while a read of the fixed
    complex has no readout to validate. The four-torus direct sum is exactly
    one of the hosts this subcommand exists to read.
    """
    layers = int(args.layers)
    if int(args.tori) == 4:
        # Two joined collars need two interior layers for the all-interior
        # cell the join removes to exist (`build_config` applies the same floor).
        layers = max(3, layers)
    config = {"seed": int(args.seed), "tori": int(args.tori),
              "collar_twist": str(args.collar_twist), "layers": layers,
              "tau_a": [args.tau_a.real, args.tau_a.imag],
              "tau_b": [args.tau_b.real, args.tau_b.imag],
              "grid": int(args.grid),
              "interior_disposition": DECLARED_INTERIOR_DISPOSITION,
              "join": str(getattr(args, "join", DECLARED_JOIN)),
              "far_flips": str(getattr(args, "far_flips", "") or ""),
              "flip_factor": float(getattr(args, "flip_factor", DECLARED_FLIP_FACTOR)),
              "surface": str(getattr(args, "surface", DECLARED_SURFACE)),
              "rotation": int(getattr(args, "rotation", DECLARED_ROTATION))}
    if config["surface"] == "decagon":
        config["layers"] = int(args.layers)
    if config["join"] == "tube":
        if config["tori"] != 4:
            raise ValueError("--join tube needs --tori 4: the tube connects the two collars' far tori")
        # The tube connects the collars itself; no interior cell is removed,
        # so one layer per collar suffices.
        config["layers"] = int(args.layers)
        config.update({"tube_layers": int(args.tube_layers), "tube_length": float(args.tube_length),
                       "tube_waist": float(args.tube_waist)})
    return config


def verify_main(args):
    import json
    import os
    config = _verify_config(args)
    record = verify(config, jitter=args.jitter, tolerance=args.tol)
    record["config"] = dict(config)
    path = args.json
    if path is None:
        directory = os.path.expanduser(args.out)
        os.makedirs(directory, exist_ok=True)
        # The moduli are part of the host, so they are part of the name: two
        # runs at different tau must not write the same record.
        flips = ("-f" + config["far_flips"].replace(",", "")) if config.get("far_flips") else ""
        path = os.path.join(directory, "verify-%dt-%s-L%d-g%d-s%d-a%s-b%s%s.json" % (
            args.tori, args.collar_twist, args.layers, args.grid, args.seed,
            _tau_slug(args.tau_a), _tau_slug(args.tau_b), flips))
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)
    _print_verify_table(record)
    print("record: %s" % path)
    return 0 if record["all_pass"] else 1


def _tau_slug(tau):
    """A modulus as a filename fragment: `0.3+1.1j` -> `0.3+1.1j`, kept
    readable rather than hashed."""
    return ("%g%+gj" % (tau.real, tau.imag)).replace(" ", "")


def _json_default(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError("cannot serialise %r" % (type(value),))


# ---- theta: the boundary lattice quantized at level k ---------------------

#: Where `theta` writes its record. Never /tmp, which is wiped at boot.
DECLARED_THETA_OUT = "~/cobordism-runs/theta-register"
#: Level 2 is a qubit per torus.
DECLARED_THETA_LEVEL = 2

#: The four-torus restriction read orders a torus's two marking cycles
#: together, (A_0, B_0, A_2, B_2); the register's symplectic form orders all
#: A cycles before all B cycles, (A_0, A_2, B_0, B_2). This permutation
#: matrix takes the first ordering to the second.
_PAIR_TO_STANDARD = None


def _pair_to_standard():
    import numpy as np
    global _PAIR_TO_STANDARD
    if _PAIR_TO_STANDARD is None:
        perm = np.zeros((4, 4))
        for target, source in enumerate((0, 2, 1, 3)):
            perm[target, source] = 1.0
        _PAIR_TO_STANDARD = perm
    return _PAIR_TO_STANDARD


def _theta_synthetic(register, tolerance):
    """T1-T7 on the generators and on synthetic period matrices: what the
    register is, before any geometry is read."""
    import numpy as np
    rng = np.random.default_rng(11)
    generators = register.generators()
    forms = register.level_two_forms() if register.level == 2 else {}
    checks = []
    moduli = [0.3 + 1.1j, -0.2 + 0.8j, -0.7 + 2.3j]
    # T1, T2: closure and unitarity, every generator at every modulus.
    worst_residual, worst_defect = 0.0, 0.0
    for name in ("S", "T", "swap"):
        for tau in moduli:
            fit = register.weil(generators[name], tau, rng)
            worst_residual = max(worst_residual, fit.residual)
            worst_defect = max(worst_defect, fit.unitarity_defect)
    checks.append(_check("T1", "the theta space is closed under the modular action: fit residual",
                         {"worst_residual": worst_residual, "matrices": ["S", "T", "swap"],
                          "moduli": len(moduli)}, tolerance, worst_residual <= tolerance))
    checks.append(_check("T2", "rho is unitary (antiunitary for the orientation-reversing swap)",
                         {"worst_unitarity_defect": worst_defect,
                          "swap_antiunitary": register.weil(generators["swap"], moduli[0], rng).antiunitary},
                         tolerance, worst_defect <= tolerance))
    # T3: the projective group law, S.S, S.T, (S.T)^3 against S.S, and the swap.
    tau = moduli[0]
    def fitted(matrix, omega):
        return register.weil(matrix, omega, rng)
    def composed(first, second, omega):
        # first after second: second acts at omega, first at second's image.
        sec = fitted(second, omega)
        omega_mid = register.orientation_reversal(omega) if sec.antiunitary else omega
        step = second @ np.diag([1.0, -1.0]) if sec.antiunitary else second
        omega_mid = register.siegel(step, omega_mid)[0]
        return fitted(first, omega_mid).compose(sec)
    law = {}
    for label, (first, second) in {"S.S": ("S", "S"), "S.T": ("S", "T"), "T.S": ("T", "S"),
                                   "swap.S": ("swap", "S"), "S.swap": ("S", "swap")}.items():
        direct = fitted(generators[first] @ generators[second], tau)
        chained = composed(generators[first], generators[second], tau)
        law[label] = register.projective_distance(direct.matrix, chained.matrix) if direct.antiunitary == chained.antiunitary else float("inf")
    checks.append(_check("T3", "projective group law: rho(M1 M2) = phase . rho(M1) rho(M2)",
                         {k: v for k, v in law.items()}, tolerance, max(law.values()) <= tolerance))
    # T4: level-2 closed forms.
    if forms:
        s_dist = register.projective_distance(fitted(generators["S"], tau).matrix, forms["S"])
        t_dist = register.projective_distance(fitted(generators["T"], tau).matrix, forms["T"])
        checks.append(_check("T4", "level 2: rho(S) is Hadamard and rho(T) is diag(1, i), each up to a phase",
                             {"hadamard_distance": s_dist, "phase_gate_distance": t_dist},
                             tolerance, max(s_dist, t_dist) <= tolerance))
    # T5: direct sum -> tensor product.
    omega = np.diag([moduli[0], moduli[1]])
    z = register.samples(2, rng)
    genus2 = register.basis(z, omega)
    product = np.einsum("is,js->ijs", register.basis(z[:, :1], moduli[0]),
                        register.basis(z[:, 1:], moduli[1])).reshape(genus2.shape)
    factor = float(np.linalg.norm(genus2 - product) / np.linalg.norm(genus2))
    direct_sum = register.direct_sum(generators["S"], generators["T"])
    joint = fitted(direct_sum, omega)
    kron = np.kron(fitted(generators["S"], moduli[0]).matrix, fitted(generators["T"], moduli[1]).matrix)
    kron_dist = register.projective_distance(joint.matrix, kron)
    rank = register.schmidt_rank(joint.matrix, (register.level, register.level))
    checks.append(_check("T5", "direct sum -> tensor product: the genus-2 basis factorises and rho(M1 (+) M2) = rho(M1) (x) rho(M2)",
                         {"basis_factorisation": factor, "kron_distance": kron_dist, "schmidt_rank": rank},
                         tolerance, factor <= tolerance and kron_dist <= tolerance and rank == 1))
    # T6: a mixing symplectic matrix entangles.
    shear = fitted(generators["shear"], omega)
    rank = register.schmidt_rank(shear.matrix, (register.level, register.level))
    measured = {"residual": shear.residual, "schmidt_rank": rank}
    passed = shear.residual <= tolerance and rank > 1
    if forms:
        measured["controlled_z_distance"] = register.projective_distance(shear.matrix, forms["shear"])
        passed = passed and measured["controlled_z_distance"] <= tolerance
    checks.append(_check("T6", "the shear [[I, B], [0, I]] mixes the two tori and entangles (level 2: controlled-Z)",
                         measured, tolerance, passed))
    # T7: polarization independence.
    span = max(register.projective_distance(fitted(generators[name], moduli[0]).matrix,
                                            fitted(generators[name], moduli[2]).matrix)
               for name in ("S", "T"))
    omega_other = np.array([[moduli[2], 0.15 + 0.05j], [0.15 + 0.05j, moduli[1]]])
    span = max(span, register.projective_distance(fitted(generators["shear"], omega).matrix,
                                                  fitted(generators["shear"], omega_other).matrix))
    checks.append(_check("T7", "polarization independence: the fitted rho is the same at every Omega",
                         {"worst_distance": span}, tolerance, span <= tolerance))
    return checks


def _theta_native(config, register, tolerance):
    """T8: the register on the seeded host -- moduli from the tori, the
    monodromy from the geometry -- and what it says about entanglement."""
    import numpy as np
    rng = np.random.default_rng(int(config["seed"]))
    twist = str(config.get("collar_twist", DECLARED_COLLAR_TWIST))
    tori, _, seed, ids, markings = seed_qubit_host(config)
    host = seed.host
    labels = (DECLARED_CONJUGATE_TORUS_LABELS if len(tori) > 2 else DECLARED_TORUS_LABELS)[:len(tori)]
    pairs = [(0, 1)] if len(tori) == 2 else [(0, 1), (2, 3)]
    checks, values = [], {"tori": len(tori), "collar_twist": twist,
                          "moduli": [complex(torus.tau()) for torus in tori]}
    fits = {}
    for a, b in pairs:
        name = "%s->%s" % (labels[a], labels[b])
        read = MC.monodromy(host, markings[a], markings[b])
        if read.obstruction:
            checks.append(_check("T8:" + name, "the monodromy is read", {"refusal": read.obstruction}, tolerance, False))
            continue
        matrix = np.asarray(read.rounded, dtype=float)
        tau = complex(tori[a].tau())
        fit = register.weil(matrix, tau, rng)
        fits[(a, b)] = fit
        expected = None
        passes = _far_flip_passes(config)
        _, predicted = _flip_lattice_map(config)
        if passes:
            forms = register.level_two_forms()
            hadamard, phase = forms["S"], forms["T"]
            closed = {"b": phase, "a": hadamard @ np.conj(phase) @ hadamard, "s": hadamard}
            if register.level == 2 and twist == "none" and len(passes) == 1:
                expected = closed[passes[0]]
            elif register.level == 2 and twist == "none" and passes == ["b", "b"]:
                expected = np.diag([1.0, -1.0]).astype(complex)
            values["predicted_monodromy"] = predicted.tolist()
            values["monodromy_matches_prediction"] = bool((np.asarray(read.rounded, dtype=int) == predicted).all())
        elif twist == "none":
            expected = np.eye(register.level, dtype=complex)
        elif register.level == 2:
            # swap = S^-1 . R: rho(swap) = rho(S^-1) K, and rho(S^-1) is Hadamard up to a phase
            expected = register.level_two_forms()["S"]
        distance = register.projective_distance(fit.matrix, expected) if expected is not None else float("nan")
        image = register.siegel(matrix @ np.diag([1.0, -1.0]), register.orientation_reversal(tau))[0] if fit.antiunitary else register.siegel(matrix, tau)[0]
        checks.append(_check(
            "T8:" + name,
            "the geometric monodromy acts on the level-%d register: unitary for the product collar, antiunitary for the swap" % register.level,
            {"monodromy": matrix.astype(int).tolist(), "residual": fit.residual,
             "unitarity_defect": fit.unitarity_defect, "antiunitary": fit.antiunitary,
             "distance_to_expected": distance, "tau_in": tau, "monodromy_tau_in": complex(image[0, 0]),
             "tau_out_declared": complex(tori[b].tau())},
            tolerance, fit.residual <= tolerance and fit.unitarity_defect <= tolerance
            and (expected is None or distance <= tolerance)
            and fit.antiunitary == (twist == "swap")
            and (not passes or (np.asarray(read.rounded, dtype=int) == predicted).all())))
        values["rho:" + name] = [[complex(x) for x in row] for row in fit.matrix]
        if passes and len(passes) >= 2 and twist == "none":
            other = dict(config)
            other["far_flips"] = ",".join(reversed(passes))
            _, _, seed_r, _, markings_r = seed_qubit_host(other)
            read_r = MC.monodromy(seed_r.host, markings_r[a], markings_r[b])
            fit_r = register.weil(np.asarray(read_r.rounded, dtype=float), tau, rng)
            apart = register.projective_distance(fit.matrix, fit_r.matrix)
            commute = (np.asarray(read.rounded, dtype=int) == np.asarray(read_r.rounded, dtype=int)).all()
            checks.append(_check(
                "T9", "composition order on the register: the passes in the two orders give different Weil "
                      "matrices exactly when their monodromies differ",
                {"passes": passes, "reversed": list(reversed(passes)),
                 "monodromy": np.asarray(read.rounded, dtype=int).tolist(),
                 "monodromy_reversed": np.asarray(read_r.rounded, dtype=int).tolist(),
                 "projective_distance": apart},
                tolerance, (apart > 1e-3) != bool(commute)))
    if len(tori) == 4 and len(fits) == 2:
        _, _, _, periods, _ = _read_restriction(host, markings)
        periods_a = np.vstack([periods[0], periods[2]])
        periods_b = np.vstack([periods[1], periods[3]])
        whole, rounded, rounding, _ = _fit_monodromy(periods_a, periods_b)
        perm = _pair_to_standard()
        standard = perm @ rounded.astype(float) @ perm.T
        omega = np.diag([complex(tori[0].tau()), complex(tori[2].tau())])
        fit = register.weil(standard, omega, rng)
        kron = np.kron(fits[(0, 1)].matrix, fits[(2, 3)].matrix)
        rank = register.schmidt_rank(fit.matrix, (register.level, register.level))
        off = float(np.linalg.norm(whole - np.block([[whole[:2, :2], np.zeros((2, 2))], [np.zeros((2, 2)), whole[2:, 2:]]])))
        checks.append(_check(
            "T8:whole",
            "four tori: rho(M_whole) = rho(M_01) (x) rho(M_23), Schmidt rank 1 -- a join along a sphere cannot entangle",
            {"residual": fit.residual, "kron_distance": register.projective_distance(fit.matrix, kron),
             "schmidt_rank": rank, "monodromy_off_block_norm": off, "rounding_residual": rounding,
             "antiunitary": fit.antiunitary},
            tolerance, fit.residual <= tolerance and register.projective_distance(fit.matrix, kron) <= tolerance
            and rank == 1 and fit.antiunitary == (twist == "swap")))
        values["rho:whole"] = [[complex(x) for x in row] for row in fit.matrix]
    return checks, values


def _boundary_faces(host):
    """The host's boundary triangles (facets of exactly one top cell) as
    sorted vertex triples."""
    from collections import Counter
    count = Counter()
    for cell in host.getTopSimplices():
        tuple_ = sorted(int(v.getId()) for v in cell.getVertices())
        for skip in range(len(tuple_)):
            count[tuple(v for n, v in enumerate(tuple_) if n != skip)] += 1
    return [face for face, n in count.items() if n == 1]


def _surface_component(faces, seed_vertex):
    """The connected component (by shared edges) of the closed surface
    `faces` containing `seed_vertex`."""
    from collections import defaultdict, deque
    by_edge = defaultdict(list)
    for index, face in enumerate(faces):
        for a in range(3):
            u, v = face[a], face[(a + 1) % 3]
            by_edge[(min(u, v), max(u, v))].append(index)
    start = next(index for index, face in enumerate(faces) if seed_vertex in face)
    seen = {start}
    queue = deque([start])
    while queue:
        index = queue.popleft()
        face = faces[index]
        for a in range(3):
            u, v = face[a], face[(a + 1) % 3]
            for other in by_edge[(min(u, v), max(u, v))]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
    return [faces[index] for index in sorted(seen)]


def _host_lengths(host):
    """`{(u, v): length}` over the host's edges, `u < v`."""
    out = {}
    for edge in host.getEdgeList().toVector():
        if edge is None or edge.getSource() is None or edge.getTarget() is None:
            continue
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        out[(min(u, v), max(u, v))] = complex(edge.getLength())
    return out


def _genus_two_boundary(host, tori, ids, markings):
    """The far boundary of the tube-joined host as a `SurfacePeriods` read:
    its faces (the boundary component containing the far tori), its
    lengths, the marking `(A_1, A_2 | B_1, B_2)` from the two far tori, and
    the root face `{(0,0), (1,0), (1,1)}` of the first far torus in its own
    counterclockwise order, which fixes the orientation."""
    from tessera.quantum import SurfacePeriods
    far = (1, 3)
    grid = int(round(len(tori[0].vertices()) ** 0.5))
    vid = lambda i, j: i * grid + j  # noqa: E731
    faces = _surface_component(_boundary_faces(host), ids[far[0]][vid(0, 0)])
    root = tuple(ids[far[0]][v] for v in (vid(0, 0), vid(1, 0), vid(1, 1)))
    a_cycles = [markings[far[0]][0], markings[far[1]][0]]
    b_cycles = [markings[far[0]][1], markings[far[1]][1]]
    return SurfacePeriods(faces, _host_lengths(host), a_cycles, b_cycles, root_face=root)


def _euler_characteristics(host):
    """The Euler characteristic of each boundary component, by shared edges."""
    faces = _boundary_faces(host)
    remaining = list(faces)
    out = []
    while remaining:
        component = _surface_component(remaining, remaining[0][0])
        members = set(component)
        remaining = [face for face in remaining if face not in members]
        vertices = {v for face in component for v in face}
        edges = {(min(u, v), max(u, v)) for face in component for a in range(3)
                 for u, v in [(face[a], face[(a + 1) % 3])]}
        out.append(len(vertices) - len(edges) + len(component))
    return sorted(out)


def _theta_tube(config, register, tolerance, sweep=None):
    """G1-G5 on the tube-joined host.

    G1: the topology of the tube -- b_1(W) = 4, the restriction subspace of
        rank 4 = b_1(dW)/2 and isotropic for the declared form (the two near
        tori on the bulk's near side, the genus-2 surface carrying each far
        torus's induced orientation), the flipped control nonvanishing;
    G2: the monodromy from the near tori to the genus-2 surface is
        block-diagonal, each block the twist's map, so rho is a product
        operator: the tube alone couples no cycle (operator entanglement is
        topological);
    G3: the genus-2 surface's own period matrix, read from its harmonic
        1-forms: rank 4, Im Omega positive definite, Omega_12 != 0 (the neck
        couples the harmonic forms: metric), symmetry residual reported;
    G4: the geometric state at z = 0 and at the register's sample points has
        entanglement entropy exactly 0 for the diagonal control
        diag(Omega_11, Omega_22) and the measured entropy for the read Omega;
    G5: the neck sweep, Omega_12 and the entropies against the waist and the
        length of the tube (the discrete Ryu-Takayanagi curve), recorded.
    """
    import numpy as np
    rng = np.random.default_rng(int(config["seed"]))
    twist = str(config.get("collar_twist", DECLARED_COLLAR_TWIST))
    k = register.level
    checks, values = [], {"join": "tube", "collar_twist": twist,
                          "tube": {key: config[key] for key in ("tube_layers", "tube_length", "tube_waist")}}

    def read(cfg):
        tori, _, seed, ids, markings = seed_qubit_host(cfg)
        host = seed.host
        betti, rank, _, periods, _ = _read_restriction(host, markings)
        near = np.vstack([periods[0], periods[2]])   # (A_1, B_1, A_2, B_2) on the near tori
        far = np.vstack([periods[1], periods[3]])    # the same order on the genus-2 surface
        surface = _genus_two_boundary(host, tori, ids, markings)
        return tori, host, betti, rank, near, far, surface

    tori, host, betti, rank, near, far, surface = read(config)
    values.update({"moduli": [complex(torus.tau()) for torus in tori], "betti": betti,
                   "harmonic_rank": rank, "euler_characteristics": _euler_characteristics(host),
                   "cells": len(host.getTopSimplices())})
    # ---- G1 -------------------------------------------------------------
    stacked_rank = _numeric_rank(np.vstack([near, far]))
    boundary_b1 = int(near.shape[0] + far.shape[0])
    iso = _lagrangian_check("tube", near, far, twist, tolerance)
    checks.append(_check(
        "G1", "tube host: b_1(W) = 4 = rank(H^1(W) -> H^1(dW)) = b_1(dW)/2, boundary components T^2, T^2, "
              "Sigma_2, the restriction isotropic for the declared form and not for the flipped control",
        {"betti": betti, "harmonic_rank": rank, "stacked_rank": stacked_rank, "boundary_b1": boundary_b1,
         "euler_characteristics": values["euler_characteristics"],
         "declared_form_norm": iso["measured"]["declared_form_norm"],
         "flipped_control_norm": iso["measured"]["flipped_control_norm"]},
        tolerance, len(betti) > 1 and betti[1] == 4 and rank == 4 and stacked_rank == 4
        and boundary_b1 == 8 and values["euler_characteristics"] == [-2, 0, 0] and iso["pass"]))
    # ---- G2 -------------------------------------------------------------
    whole, rounded, rounding, fit = _fit_monodromy(near, far)
    off = float(np.linalg.norm(whole - np.block([[whole[:2, :2], np.zeros((2, 2))],
                                                 [np.zeros((2, 2)), whole[2:, 2:]]])))
    expected = np.asarray(_TWIST_MATRICES[twist], dtype=float)
    block_distance = max(float(np.abs(rounded[:2, :2] - expected).max()),
                         float(np.abs(rounded[2:, 2:] - expected).max()))
    perm = _pair_to_standard()
    standard = perm @ rounded.astype(float) @ perm.T
    omega_declared = np.diag([complex(tori[1].tau()), complex(tori[3].tau())])
    weil = register.weil(standard, omega_declared, rng)
    schmidt = register.schmidt_rank(weil.matrix, (k, k))
    checks.append(_check(
        "G2", "the monodromy from the near tori to the genus-2 surface is block-diagonal, each block the "
              "twist's map, and rho is a product operator: the tube alone couples no cycle",
        {"monodromy": rounded.astype(int).tolist(), "off_block_norm": off, "rounding_residual": rounding,
         "fit_residual": fit, "block_distance_to_twist": block_distance, "weil_residual": weil.residual,
         "operator_schmidt_rank": schmidt, "antiunitary": weil.antiunitary},
        tolerance, off <= tolerance and rounding <= tolerance and fit <= tolerance and block_distance == 0.0
        and weil.residual <= tolerance and schmidt == 1 and weil.antiunitary == (twist == "swap")))
    # ---- G3 -------------------------------------------------------------
    report = surface.report()
    omega = surface.omega
    moduli_far = [complex(tori[1].tau()), complex(tori[3].tau())]
    diagonal_distance = [abs(omega[0, 0] - moduli_far[0]), abs(omega[1, 1] - moduli_far[1])]
    checks.append(_check(
        "G3", "the genus-2 surface's own period matrix: harmonic rank 4, Im Omega positive definite, "
              "Omega_12 != 0 -- the neck couples the harmonic forms (symmetry residual reported)",
        {"omega": omega, "omega_12": complex(omega[0, 1]), "omega_21": complex(omega[1, 0]),
         "symmetry_residual": report["symmetry_residual"],
         "imaginary_eigenvalues": report["imaginary_eigenvalues"], "harmonic_rank": report["harmonic_rank"],
         "orientation_flipped": report["orientation_flipped"], "j_residual": report["j_residual"],
         "diagonal_distance_to_far_moduli": diagonal_distance, "surface": {key: report[key] for key in ("faces", "edges", "vertices")},
         "warnings": report["warnings"]},
        tolerance, report["harmonic_rank"] == 4 and report["positive"] and not report["orientation_flipped"]
        and abs(omega[0, 1]) > tolerance))
    # ---- G4 -------------------------------------------------------------
    def entropies(omega_matrix):
        zero = register.entanglement_entropy(register.coherent_state(np.zeros(2), omega_matrix), (k, k))[0]
        sampled = [register.entanglement_entropy(register.coherent_state(z, omega_matrix), (k, k))[0]
                   for z in register.samples(2, rng)]
        return zero, float(np.mean(sampled)), float(np.max(sampled))
    control = np.diag(np.diag(omega))
    zero_c, mean_c, max_c = entropies(control)
    zero, mean, maximum = entropies(omega)
    values["entropy"] = {"zero": zero, "mean": mean, "max": maximum,
                         "control_zero": zero_c, "control_mean": mean_c, "control_max": max_c}
    checks.append(_check(
        "G4", "the geometric state of the genus-2 boundary is entangled between its two tori by the neck: "
              "entropy exactly 0 for the diagonal control, the measured entropy for the read Omega",
        {"entropy_bits_at_zero": zero, "entropy_bits_mean": mean, "entropy_bits_max": maximum,
         "control_entropy_max": max_c, "omega_12": complex(omega[0, 1])},
        tolerance, max_c <= tolerance and maximum > tolerance))
    # ---- G5 -------------------------------------------------------------
    if sweep:
        rows = []
        for waist, length in sweep:
            cfg = dict(config)
            cfg.update({"tube_waist": float(waist), "tube_length": float(length)})
            try:
                _, _, betti_s, rank_s, near_s, far_s, surface_s = read(cfg)
            except Exception as error:  # a degenerate neck refuses by name; the row records it
                rows.append({"waist": float(waist), "length": float(length), "refusal": str(error)})
                continue
            omega_s = surface_s.omega
            rep = surface_s.report()
            zero_s, mean_s, max_s = entropies(omega_s)
            _, rounded_s, _, _ = _fit_monodromy(near_s, far_s)
            rows.append({"waist": float(waist), "length": float(length), "omega": omega_s,
                         "omega_12": complex(omega_s[0, 1]), "abs_omega_12": float(abs(omega_s[0, 1])),
                         "entropy_zero": zero_s, "entropy_mean": mean_s, "entropy_max": max_s,
                         "symmetry_residual": rep["symmetry_residual"], "positive": rep["positive"],
                         "harmonic_rank": rep["harmonic_rank"], "betti": betti_s,
                         "monodromy": rounded_s.astype(int).tolist(),
                         "diagonal_distance_to_far_moduli": [abs(omega_s[0, 0] - moduli_far[0]),
                                                             abs(omega_s[1, 1] - moduli_far[1])]})
        values["sweep"] = rows
        good = [row for row in rows if "refusal" not in row]
        checks.append(_check(
            "G5", "neck sweep: every point read (rank 4, Im Omega > 0, integer block-diagonal monodromy); "
                  "Omega_12 and the entropies against the waist and the length are recorded",
            {"points": len(rows), "refused": len(rows) - len(good),
             "abs_omega_12_range": [min(r["abs_omega_12"] for r in good), max(r["abs_omega_12"] for r in good)] if good else None,
             "entropy_mean_range": [min(r["entropy_mean"] for r in good), max(r["entropy_mean"] for r in good)] if good else None},
            tolerance, bool(good) and all(r["positive"] and r["harmonic_rank"] == 4 for r in good)))
    return checks, values


#: The boundary surface the theta host is built on: the flat tori, or the
#: Z/10-symmetric genus-2 surface (the decagon with opposite sides
#: identified), whose collar twisted by a rotation reads an entangling
#: monodromy.
DECLARED_SURFACES = ("tori", "decagon")
DECLARED_SURFACE = "tori"
DECLARED_ROTATION = 2


def _marking_in_host(cycles, ids):
    return [[(int(ids[u]), int(ids[v])) for u, v in cycle] for cycle in cycles]


def seed_genus_two_host(config):
    """The collar of the symmetric genus-2 surface with its far end
    relabelled by the rotation `r^k` (`rotation` = k): the surface, the
    seed, the id maps and the two markings `[A_1, B_1, A_2, B_2]` as host
    steps. `k = 0` is the product collar, the control."""
    from tessera.quantum import SymmetricGenusTwo
    surface = SymmetricGenusTwo()
    k = int(config.get("rotation", DECLARED_ROTATION)) % 10
    twist = surface.rotation(k) if k else []
    if twist and not surface.is_automorphism(twist):
        raise RuntimeError("r^%d is not a simplicial automorphism of the decagon surface" % k)
    seed = MC.seed_collar(surface.spacetime(), surface.spacetime(), int(config["layers"]), twist)
    ids = [{int(a): int(b) for a, b in mapping.items()} for mapping in seed.vertex_ids]
    _dispose_interior(seed.host, [surface, surface], ids,
                      config.get("interior_disposition", DECLARED_INTERIOR_DISPOSITION), int(config["seed"]))
    markings = [_marking_in_host(surface.marking(), ids[0]), _marking_in_host(surface.marking(), ids[1])]
    return surface, seed, ids, markings


def _single_qubit_cliffords(register):
    """The 24 single-qubit Clifford matrices modulo phase, generated by the
    Hadamard and phase gates."""
    import numpy as np
    forms = register.level_two_forms()
    generators = [forms["S"], forms["T"]]
    found = [np.eye(2, dtype=complex)]
    frontier = list(found)
    while frontier:
        nxt = []
        for element in frontier:
            for g in generators:
                candidate = g @ element
                if all(register.projective_distance(candidate, other) > 1e-9 for other in found):
                    found.append(candidate)
                    nxt.append(candidate)
        frontier = nxt
    return found


def _theta_genus_two(config, register, tolerance):
    """E1-E6 on the twisted collar of the symmetric genus-2 surface.

    E1: b_1(W) = 4 = rank(H^1(W) -> H^1(dW)) = b_1(dW)/2, the restriction
        isotropic for the declared form (the far side reversed, the
        rotation orientation-preserving), the flipped control nonvanishing;
    E2: the monodromy is the rotation's induced map on H_1 in the symplectic
        marking, integer, det +1, of the rotation's order (5 for r^2), and
        not block-diagonal: it mixes the two torus factors;
    E3: the surface's own period matrix is a fixed point of M, the marking
        formula Omega = (M_BA + M_BB Omega)(M_AA + M_AB Omega)^-1: the
        rotation is an isometry;
    E4: rho_2(M) fitted from the transformation law at Omega: residual and
        unitarity at machine precision, rho(M)^order proportional to I,
        operator Schmidt rank > 1;
    E5: entangling power: rho(M) takes a product state to a Bell pair (one
        bit) and is at projective distance O(1) from every A (x) B and every
        SWAP . A (x) B with A, B single-qubit Cliffords;
    E6: the control, the product collar (rotation 0): M = I, no entangling
        power.
    """
    import numpy as np
    rng = np.random.default_rng(int(config["seed"]))
    k = int(config.get("rotation", DECLARED_ROTATION)) % 10
    checks, values = [], {"surface": "decagon", "rotation": k}
    surface, seed, ids, markings = seed_genus_two_host(config)
    host = seed.host
    report = surface.report()
    values["surface_report"] = {key: report[key] for key in ("euler_characteristic", "intersection_form_z",
                                                             "symplectic_basis_z", "harmonic_rank", "positive",
                                                             "symmetry_residual", "imaginary_eigenvalues")}
    betti, rank, _, periods, _ = _read_restriction(host, markings)
    near, far = periods[0], periods[1]
    values.update({"betti": [int(b) for b in betti], "harmonic_rank": rank, "cells": len(host.getTopSimplices())})
    iso = _lagrangian_check("decagon", near, far, "none", tolerance)
    stacked_rank = _numeric_rank(np.vstack([near, far]))
    checks.append(_check(
        "E1", "genus-2 collar: b_1(W) = 4 = rank(H^1(W) -> H^1(dW)) = b_1(dW)/2, the restriction isotropic for "
              "the declared form (far side reversed, rotation orientation-preserving), flipped control nonvanishing",
        {"betti": values["betti"], "harmonic_rank": rank, "stacked_rank": stacked_rank,
         "declared_form_norm": iso["measured"]["declared_form_norm"],
         "flipped_control_norm": iso["measured"]["flipped_control_norm"], "cells": values["cells"]},
        tolerance, len(betti) > 1 and betti[1] == 4 and rank == 4 and stacked_rank == 4 and iso["pass"]))
    # E2
    whole, rounded, rounding, fit = _fit_monodromy(near, far)
    perm = _pair_to_standard()
    m = perm @ rounded @ perm.T                       # (A_1, A_2, B_1, B_2)
    predicted = surface.induced_map(k)
    det = int(round(float(np.linalg.det(m))))
    order = None
    power = np.eye(4, dtype=int)
    for n in range(1, 21):
        power = power @ m
        if (power == np.eye(4, dtype=int)).all():
            order = n
            break
    expected_order = {0: 1, 5: 2}.get(k, 5 if k % 2 == 0 else 10)
    off = float(np.abs(m[:2, 2:]).max() + np.abs(m[2:, :2]).max()) if k else 0.0
    # the register's symplectic form
    j = register.symplectic_form(2)
    symplectic = float(np.abs(m.T @ j @ m - j).max())
    mixes = bool(k not in (0, 5)) and not (np.abs(m[0, 1]) + np.abs(m[1, 0]) + np.abs(m[0, 3]) + np.abs(m[1, 2])
                                          + np.abs(m[2, 1]) + np.abs(m[3, 0]) + np.abs(m[2, 3]) + np.abs(m[3, 2]) == 0)
    checks.append(_check(
        "E2", "the monodromy is the rotation's induced map on H_1 in the symplectic marking: integer, det +1, "
              "symplectic, of the rotation's order, and mixing the two torus factors",
        {"monodromy": m.tolist(), "predicted": predicted.tolist(), "rounding_residual": rounding,
         "fit_residual": fit, "det": det, "order": order, "expected_order": expected_order,
         "symplectic_defect": symplectic, "mixes_the_factors": mixes},
        tolerance, (m == predicted).all() and rounding <= tolerance and fit <= tolerance and det == 1
        and order == expected_order and symplectic == 0.0 and (mixes or k in (0, 5))))
    # E3
    omega = surface.periods.omega
    m_aa, m_ab, m_ba, m_bb = m[:2, :2], m[:2, 2:], m[2:, :2], m[2:, 2:]
    moved = (m_ba + m_bb @ omega) @ np.linalg.inv(m_aa + m_ab @ omega)
    fixed = float(np.linalg.norm(moved - omega) / np.linalg.norm(omega))
    checks.append(_check(
        "E3", "the surface's own period matrix is a fixed point of M: the rotation is an isometry",
        {"omega": omega, "moved": moved, "relative_distance": fixed, "imaginary_eigenvalues": report["imaginary_eigenvalues"],
         "symmetry_residual": report["symmetry_residual"]},
        1e-9, fixed <= 1e-9 and report["positive"]))
    # E4
    weil = register.weil(m.astype(float), omega, rng)
    rho = weil.matrix
    power = np.eye(4, dtype=complex)
    for _ in range(expected_order):
        power = rho @ power
    schmidt = register.schmidt_rank(rho, (register.level, register.level))
    checks.append(_check(
        "E4", "rho_2(M) from the transformation law at Omega: unitary, of the rotation's order up to a phase, "
              "operator Schmidt rank > 1",
        {"residual": weil.residual, "unitarity_defect": weil.unitarity_defect, "antiunitary": weil.antiunitary,
         "order_defect": register.projective_distance(power, np.eye(4)), "operator_schmidt_rank": schmidt},
        tolerance, weil.residual <= tolerance and weil.unitarity_defect <= tolerance and not weil.antiunitary
        and register.projective_distance(power, np.eye(4)) <= 1e-9 and (schmidt > 1 or k in (0, 5))))
    values["rho"] = [[complex(x) for x in row] for row in rho]
    # E5
    def entropy_of(state):
        return register.entanglement_entropy(rho @ state, (2, 2))[0]
    # A Clifford gate sends product stabilizer states to stabilizer states,
    # whose entanglement is exactly 0 or 1 bit, so the 36 products of the six
    # single-qubit stabilizer states decide the entangling power exactly; a
    # Clifford that is not a local Clifford or a swap times one sends some of
    # them to a Bell pair. Random product states are reported alongside.
    stabilizer = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2),
                  np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)]
    products = [np.kron(u, v).astype(complex) for u in stabilizer for v in stabilizer]
    entropies = [entropy_of(state) for state in products]
    sampled = []
    for _ in range(64):
        u = rng.normal(size=2) + 1j * rng.normal(size=2)
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        sampled.append(entropy_of(np.kron(u / np.linalg.norm(u), v / np.linalg.norm(v))))
    cliffords = _single_qubit_cliffords(register)
    swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
    local_distance = min(register.projective_distance(rho, np.kron(a, b)) for a in cliffords for b in cliffords)
    swap_distance = min(register.projective_distance(rho, swap @ np.kron(a, b)) for a in cliffords for b in cliffords)
    entangling = k not in (0, 5)
    checks.append(_check(
        "E5", "entangling power: rho(M) takes a product state to a Bell pair (one bit) and is O(1) from every "
              "local Clifford A (x) B and every SWAP . A (x) B",
        {"max_entropy_bits": max(entropies), "bell_pairs_among_36_product_stabilizer_states": int(sum(e > 1 - 1e-9 for e in entropies)),
         "computational_basis_entropies_bits": [entropies[0], entropies[1], entropies[6], entropies[7]],
         "sampled_product_states_mean_bits": float(np.mean(sampled)), "sampled_product_states_max_bits": float(max(sampled)),
         "distance_to_local_cliffords": local_distance,
         "distance_to_swap_times_local": swap_distance, "cliffords": len(cliffords)},
        1e-9, (max(entropies) >= 1.0 - 1e-9 and local_distance > 1e-3 and swap_distance > 1e-3) if entangling
        else (max(entropies) <= 1e-9 and min(local_distance, swap_distance) <= 1e-9)))
    values["entropies"] = entropies
    # E6: the control
    if k:
        control = dict(config)
        control["rotation"] = 0
        rows, control_values = _theta_genus_two(control, register, tolerance)
        by_id = {row["id"]: row for row in rows}
        checks.append(_check(
            "E6", "the control: the product collar reads M = I with no entangling power",
            {"monodromy": by_id["E2"]["measured"]["monodromy"], "max_entropy_bits": by_id["E5"]["measured"]["max_entropy_bits"],
             "all_pass": all(row["pass"] for row in rows)},
            tolerance, all(row["pass"] for row in rows) and by_id["E2"]["measured"]["monodromy"] == np.eye(4, dtype=int).tolist()))
    return checks, values


def theta(config, level=DECLARED_THETA_LEVEL, tolerance=DECLARED_VERIFY_TOLERANCE, sweep=None):
    from tessera.quantum import ThetaRegister
    register = ThetaRegister(level=level, tolerance=tolerance)
    checks = _theta_synthetic(register, tolerance)
    if str(config.get("surface", DECLARED_SURFACE)) == "decagon":
        native, values = _theta_genus_two(config, register, tolerance)
    elif str(config.get("join", DECLARED_JOIN)) == "tube":
        native, values = _theta_tube(config, register, tolerance, sweep=sweep)
    else:
        native, values = _theta_native(config, register, tolerance)
    checks.extend(native)
    values["level"] = int(level)
    return {"checks": checks, "values": values, "tolerance": float(tolerance),
            "all_pass": all(row["pass"] for row in checks), "certifies": THETA_CERTIFIES}


THETA_CERTIFIES = (
    "On the level-k theta quantization of the boundary marking lattices: the "
    "theta space is closed under the integer monodromies (fit residual), each "
    "monodromy acts unitarily (antiunitarily when orientation-reversing) with "
    "the Weil matrix independent of the modulus, the level-2 generators are the "
    "Hadamard and phase gates, a direct sum of tori is a tensor product of "
    "registers on which a block-diagonal monodromy is a product operator, and "
    "a monodromy mixing two tori (the shear) is entangling (controlled-Z at "
    "level 2). On the seeded hosts the geometric monodromies are read and act "
    "as stated; the four-torus join along a sphere is block-diagonal and "
    "therefore cannot entangle; with layered flip passes the geometric Dehn "
    "twists act as the level-2 generators diag(1, i), H diag(1, -i) H and the "
    "Hadamard matrix, and two passes in the two orders give different Weil "
    "matrices exactly when their monodromies differ. On the tube-joined host "
    "(G1-G5): the tube alone leaves the monodromy block-diagonal (a product "
    "operator: operator entanglement is topological), while the genus-2 far "
    "boundary's own period matrix couples the two tori (Omega_12 != 0, metric) "
    "and the geometric state it defines is entangled between them by that "
    "coupling, exactly zero for the diagonal control. On the decagon collar "
    "(E1-E6): the collar of the Z/10-symmetric genus-2 surface relabelled by "
    "the order-5 rotation reads an integer symplectic monodromy of order 5 that "
    "mixes the two torus factors, fixes the surface's own period matrix, and "
    "acts on the level-2 register as a two-qubit Clifford gate that takes a "
    "product state to a Bell pair and is O(1) from every local Clifford and "
    "every swap times a local Clifford -- an entangling monodromy realized "
    "geometrically. NOT certified: a Dehn twist on a genus-2 boundary, the "
    "symmetry of a discrete period matrix beyond its reported residual, "
    "non-abelian levels, or any change to the modulus register.")


def _add_theta_arguments(theta_parser):
    for name, kwargs in (
            ("--tori", dict(type=int, choices=(2, 4), default=DECLARED_TORI)),
            ("--layers", dict(type=int, default=DECLARED_COLLAR_LAYERS)),
            ("--collar-twist", dict(dest="collar_twist", choices=("none", "swap"), default=DECLARED_COLLAR_TWIST)),
            ("--grid", dict(type=int, default=DECLARED_GRID)),
            ("--seed", dict(type=int, default=DECLARED_SEED)),
            ("--tau-a", dict(type=_complex_argument, default=DECLARED_TAU_A)),
            ("--tau-b", dict(type=_complex_argument, default=DECLARED_TAU_B))):
        theta_parser.add_argument(name, **kwargs)
    theta_parser.add_argument("--join", choices=DECLARED_JOINS, default=DECLARED_JOIN,
                              help="how four tori are joined: along a removed tetrahedron (sphere, the direct "
                                   "sum) or by a tube between the far tori (a 1-handle; the far boundary is one "
                                   "genus-2 surface, read by G1-G4)")
    theta_parser.add_argument("--tube-layers", dest="tube_layers", type=int, default=DECLARED_TUBE_LAYERS,
                              help="prism layers of the tube (default %d)" % DECLARED_TUBE_LAYERS)
    theta_parser.add_argument("--tube-length", dest="tube_length", type=float, default=DECLARED_TUBE_LENGTH,
                              help="height of one tube layer (default %g)" % DECLARED_TUBE_LENGTH)
    theta_parser.add_argument("--tube-waist", dest="tube_waist", type=float, default=DECLARED_TUBE_WAIST,
                              help="scale of the tube's interior rings about their centroid (default %g)"
                                   % DECLARED_TUBE_WAIST)
    theta_parser.add_argument("--surface", choices=DECLARED_SURFACES, default=DECLARED_SURFACE,
                              help="the boundary surface: the flat tori, or the Z/10-symmetric genus-2 decagon "
                                   "surface whose collar is twisted by a rotation (E1-E6)")
    theta_parser.add_argument("--rotation", type=int, default=DECLARED_ROTATION,
                              help="the rotation r^k relabelling the decagon collar's far end (default %d, order 5)"
                                   % DECLARED_ROTATION)
    theta_parser.add_argument("--neck-waists", dest="neck_waists", default=None,
                              help="comma-separated waists to sweep on the tube host (G5), e.g. 1,0.7,0.5,0.3")
    theta_parser.add_argument("--neck-lengths", dest="neck_lengths", default=None,
                              help="comma-separated tube lengths to sweep on the tube host (G5), e.g. 0.5,1,2,4")
    theta_parser.add_argument("--far-flips", dest="far_flips", default="",
                              help="layered flip passes on the far torus (b, a, s; see verify)")
    theta_parser.add_argument("--flip-factor", dest="flip_factor", type=float, default=DECLARED_FLIP_FACTOR)
    theta_parser.add_argument("--level", type=int, default=DECLARED_THETA_LEVEL,
                              help="the quantization level k; 2 is a qubit per torus (default %d)" % DECLARED_THETA_LEVEL)
    theta_parser.add_argument("--tol", type=float, default=DECLARED_VERIFY_TOLERANCE)
    theta_parser.add_argument("--out", default=DECLARED_THETA_OUT)
    theta_parser.add_argument("--json", default=None)


def theta_main(args):
    import json
    import os
    config = _verify_config(args)
    sweep = None
    if config.get("join") == "tube" and (args.neck_waists or args.neck_lengths):
        waists = [float(x) for x in (args.neck_waists or str(config["tube_waist"])).split(",")]
        lengths = [float(x) for x in (args.neck_lengths or str(config["tube_length"])).split(",")]
        sweep = [(w, l) for w in waists for l in lengths]
    record = theta(config, level=args.level, tolerance=args.tol, sweep=sweep)
    record["config"] = dict(config)
    path = args.json
    if path is None:
        directory = os.path.expanduser(args.out)
        os.makedirs(directory, exist_ok=True)
        tube = ""
        if config.get("surface") == "decagon":
            tube = "-decagon-r%d" % config["rotation"]
        if config.get("join") == "tube":
            tube = "-tube-tL%d-len%g-w%g" % (config["tube_layers"], config["tube_length"], config["tube_waist"])
            if sweep:
                tube += "-sweep%d" % len(sweep)
        flips = ("-f" + config["far_flips"].replace(",", "")) if config.get("far_flips") else ""
        path = os.path.join(directory, "theta-k%d-%dt-%s-L%d-g%d-s%d-a%s-b%s%s%s.json" % (
            args.level, args.tori, args.collar_twist, config["layers"], args.grid, args.seed,
            _tau_slug(args.tau_a), _tau_slug(args.tau_b), tube, flips))
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)
    _print_verify_table(record)
    print("record: %s" % path)
    return 0 if record["all_pass"] else 1

def build_parser():
    parser = argparse.ArgumentParser(
        description="Animate the qubit cobordism and its declared read-outs.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="drive the qubit cobordism and render the overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=QUBIT_READOUT_NOTES)
    ea._add_common_run_arguments(
        run, out_default="qubit_animation.gif",
        geometry_help=(
            "also write the FINAL complex here (schema 1: cells, edges as "
            "[src, tgt, Re l^2, Im l^2], vertex times, and the qubit blocks "
            "with their markings), the one output a run can be rebuilt "
            "from. Written however the drive ends, an interrupt included"))
    _add_qubit_run_arguments(run)
    verify_parser = sub.add_parser(
        "verify", help="read the seeded host and check the cobordism functor on it",
        description="Seed the qubit host and READ it: the restriction theorem, the "
                    "monodromy's integrality and its metric independence, the "
                    "composition law in the twist group, the direct sum at four "
                    "tori, and the metric-dependent transport. No stage runs; "
                    "every row prints its measured value beside its tolerance.")
    _add_verify_arguments(verify_parser)
    theta_parser = sub.add_parser(
        "theta", help="quantize the boundary lattices at level k and read the monodromies on that register",
        description="The theta quantization register: each torus's marking lattice quantized at "
                    "level k (2 = a qubit). A direct sum of tori is a tensor product there, and "
                    "every integer monodromy acts by its Weil matrix, unitary or antiunitary. "
                    "T1-T7 are the register on synthetic data; T8 reads the seeded host.")
    _add_theta_arguments(theta_parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        return verify_main(args)
    if args.command == "theta":
        return theta_main(args)
    try:
        config = build_config(
            steps=args.steps, seed=args.seed,
            stage1_iters=args.stage_one_iterations,
            stage2_iters=args.stage_two_iterations,
            tolerance=args.tolerance, patience=args.patience,
            candidate_moves=args.candidate_moves,
            combinatorial_depth=args.combinatorial_depth,
            combinatorial_length=args.combinatorial_length,
            backsteps=args.backsteps,
            surgical_depth=getattr(args, "surgical_depth", _LEGACY_UNSET),
            combinatorial_breadth=getattr(
                args, "combinatorial_breadth", _LEGACY_UNSET),
            readout=args.readout, tori=args.tori,
            output_state=args.output_state, collar_twist=args.collar_twist,
            interior_disposition=args.interior_disposition,
            whole_pairing=args.whole_pairing,
            layers=args.layers, tau_a=args.tau_a, tau_b=args.tau_b,
            grid=args.grid, coupling=args.coupling, time=args.time,
            input_weight=args.input_weight, regge=args.regge,
            extend_boundary=args.extend_boundary,
            score_leak=args.score_leak, states=args.states,
            operator=args.operator, pin_boundary=args.pin_boundary,
            pin_boundary_state=args.pin_boundary_state)
        ea._validate_output_paths(args.out, args.json, args.geometry)
    except ValueError as error:
        parser.error(str(error))
    return ea._run_from_args(
        args, config, drive, drive_live, render, geometry_document)


if __name__ == "__main__":
    sys.exit(main())
