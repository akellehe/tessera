# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Animate unforced emergence and the certificates the whitepaper names.

ONE documented command drives the unmodified joint Regge-Hodge stationarity
objective on a neutral complex and, after every engine unit, draws what the
paper's certificates actually read off the accepted geometry.

What is driven, and what is only read
-------------------------------------
In neutral mode the dynamics is `MultiCobordism` in
`SimulationMode.EMERGENCE` / `EmergenceSubmode.STRICT` with
`JointStationarityObjective`. Nothing the neutral certificate panels compute
enters that objective: the firewall is structural (a static `objectiveOf` over
declared scalars), and every panel is a read-only measurement over the accepted
geometry through the library's own observable classes. No target is pinned, no
register is forced, and no residual against a prescribed carrier is scored.
The neutral seed's declared boundary region is held fixed; that removes
coordinates from relaxation without adding a pinned objective. Qubit mode is
the explicit exception: its selected readout and declared state are scored as
the target that mode exists to fit.

The paper's ontology, and what this driver refuses to draw
----------------------------------------------------------
A quark is sought as a PERSISTENT MODULAR SPECTRAL CLUSTER carrying an
anchored rank-three band -- never as a hole. Betti numbers are drawn, because
the paper keeps them as independent topological observables, but they are
never a quark count and never a success condition. There is no color-register
count, no `{1, omega, omega^2}` target and no singlet residual anywhere in
this file, because the construction those belong to is not the one the paper
specifies.

The panels, and the class that feeds each
-----------------------------------------
1.  objective trace                 -- the node's own objective terms
2.  drawing layout of the complex   -- persisted 1-skeleton, stabilized for
                                       drawing, coloured by causal class
3.  dual spatial curvature          -- `Re eps*|star|`, from timelike hinges
4.  dual temporal curvature         -- `Im eps*|star|`, from spacelike hinges
5.  persistent modular clusters     -- `PersistentModularity`
6.  fiber rank / gap / localization -- `SpectralFiber`, `SpectralFiberTracker`
7.  anchor profile                  -- `ColorAnchor` (score, max term,
                                       participation ratio, phase dispersion)
8.  transports and holonomy         -- `FiberConnection`
9.  exchange and rotation           -- `ExchangeHolonomy`
10. crossing readouts               -- `CrossingReadouts` (sign of Re pi_perp
                                       per crossing, crossing mass, the
                                       one-third baryon sum, charge power)
11. <J^2> and Var(J^2)              -- `CovarianceState` Wick reads
12. Betti numbers                   -- `Spacetime` (topological observable)
13. verdict and named reasons       -- `ParticleClusters.classifyBaryon`

Two of those are DRAWING, not measurement, and are kept out of the record
accordingly: the layout positions and the dual curvature appear in no
`to_json` block. In the complex panel, POSITION is a drawing artefact while
COLOUR is a measurement -- the causal class of each edge's own interval from
`arg(l^2)` -- and the panel says so on its face, because stabilizing the
picture and colouring it together make it look more physical than it is.

Refusals are first class. On accessible hosts most channels are absent, and an
absent panel states WHAT is absent and WHY rather than drawing a blank or a
zero. A zero is a measurement; an absence is not, and the two are never
conflated.

Running it
----------
Cap parallelism; this box may be shared::

    OMP_NUM_THREADS=8 .venv-build/bin/python \\
      examples/cobordism/emergence_animation.py run --size 6 --steps 6 \\
      --out emergence.gif

``--out`` accepts ``.gif`` (pillow) or ``.mp4`` (ffmpeg); ``.png`` renders the
final frame alone. ``--json`` additionally writes the per-frame measurements,
so a panel can be checked against a number.

``--edge-disposition`` chooses the seed's causal character: ``random`` (the
default, magnitude one with the real/imaginary split drawn per edge),
``spacelike`` (``l^2 = +1``), ``timelike`` (``l^2 = -1``), ``lightlike``
(``l^2 = i``), or ``foliated`` (a PRESCRIBED light cone -- timelike between
hop layers of M0, spacelike within one). Only ``foliated`` prescribes a causal
order; it is labelled as such wherever it is reported and is never presented
as emergent.

The qubit input mode
--------------------
``--inputs qubit`` drives the qubit cobordism of
``docs/design/qubit_cobordism_spec.md`` through this SAME loop (spec R5, D4):
two flat qubit tori (``SimplicialQubit.flat_torus``) are the boundary of a
3-complex whose bulk starts as the collar between them
(``MultiCobordism.seed_collar``) and is then synthesized by stage 1 and stage 2
against the two-body target chi of spec S5 -- the XY flip-flop of two
spin-1/2 at ``--J`` and ``--time`` -- while each torus keeps representing
its input state through the zero mode of its OWN Laplacian: the block
residual of spec D2, ``1 - |<psi(tau_in)|psi(tau_hat)>|^2`` with tau_hat the
ratio of the transported periods of the holomorphic form of the block's own
Laplacian on its live surface (``MultiCobordism.block_qubit``), is in the
objective at ``--input-weight``. The input surfaces are held by default so the
bulk is fitted to the declared inputs; ``--no-pin-boundary`` restores the
historical freely relaxed boundary. Each block holds its
marking (``set_input_marking``), which fixes the cycles the periods are taken
over and, by A.B = +1, the orientation the surface is read in. The zero mode
of the ENTIRE cobordism is the OUTPUT state (spec R1): it is read at each
torus as its coefficients in that torus's live frame (the zero mode of its
own Laplacian normalized by its marking, derived by the engine at every
read) and never held. The objective in force is the node's default
(``legacy``): the Regge stationarity term when ``--regge`` is on, plus Gamma
times r_U, where r_U is the weighted sum of the two block residuals and the
two-body residual. The tori carry complex lengths, so the node runs the
complex locus. Every read-out of spec S6 is a frame channel, present or
``Absent``: per block the block residual with its weight, the output state --
the coefficients of the whole's zero mode in the block's live frame next to
the input (1, tau_in), with the leak of that fit -- the former own-kernel
leak as a labelled diagnostic, and the qubit read on the block's live
surface (tau-hat, the Bloch vector, the J residual, the Delaunay and
condition diagnostics, the Fubini-Study and Weil-Petersson distances to the
input), the object the block residual scores; for the whole the Betti
numbers, the boundary components with their Euler characteristics, the
completion status, the monodromy between the two markings (periods with
parallel transport), the restricted leak of each input line in the whole's
zero mode, the two-body read (the transfer in the derived period frames, its
leak against chi, the Schmidt spectrum and rank, the reversal residual) and
the objective terms. Records go under
``~/cobordism-runs/qubit-cobordism/``::

    OMP_NUM_THREADS=8 python examples/cobordism/emergence_animation.py run \\
      --inputs qubit --tau-a 0.3+1.1j --tau-b=-0.2+0.8j --grid 3 --steps 4 \\
      --out ~/cobordism-runs/qubit-cobordism/run.gif \\
      --json ~/cobordism-runs/qubit-cobordism/run.json

(``--tau-b=-0.2+0.8j`` with the ``=``: a value that starts with ``-`` is
otherwise read as an option name.)
"""

import argparse
import cmath
import itertools
import json
import math
import numbers
import os
import random
import sys
import warnings

import tessera as T

cob = T.cobordism
obs = T.observables
qu = T.quantum
MC = cob.MultiCobordism


# =====================================================================
# declared parameters -- fixed before any datum is examined
# =====================================================================

#: Stellar Pachner adds applied to the seed 4-ball.
DECLARED_SIZE = 6
#: The declared name of the incoming boundary region. Named once here and
#: passed as this constant, never spelled at a call site: `regionHandle`
#: raises by name on an undeclared region, so a mis-spelling that reached
#: the engine would refuse rather than silently scope to the whole complex.
M0_REGION = "m0"
#: Engine units to drive. One unit is one stage-1 update plus one stage-2
#: relaxation -- the engine's deterministic unit (#579).
DECLARED_STEPS = 6
#: Candidate moves offered to stage 1 per unit.
DECLARED_CANDIDATE_MOVES = 6
#: Stage-1 combinatorial updates per engine unit.
DECLARED_STAGE1_ITERS = 1
#: Stage-2 relaxation iterations per unit.
DECLARED_STAGE2_ITERS = 12
#: How many moves deep stage 1 searches for an objective-lowering SEQUENCE.
#:
#: When a batch of single moves finds no improvement the search deepens
#: iteratively -- 2-move sequences, then 3, up to this many -- committing an
#: F-lowering sequence as a whole. One means single moves only.
#:
#: The deepening covers EVERY move kind the stage-1 draw offers: the four
#: Pachner moves plus the cone-outs and cone-ins, not the surgical moves
#: alone. The depth is over the whole draw, not over a subset of it.
#: ``--surgical-depth`` remains a hidden command-line and Python API alias.
DECLARED_COMBINATORIAL_DEPTH = 1
DECLARED_SURGICAL_DEPTH = DECLARED_COMBINATORIAL_DEPTH
#: Fixed composition length for the combinatorial search, or zero for the
#: schedule above.
#:
#: Non-zero runs the depth ladder the other way round: stage 1 searches
#: sequences of exactly this many moves FIRST, and backs off one move at a
#: time -- to this many minus one, then minus two, down to single moves --
#: only when nothing at the current length lowers the objective.
#:
#: The two schedules answer different questions. The deepening schedule finds
#: a single improving move whenever one exists and only ever looks at pairs
#: on a plateau; the backing-off schedule asks whether a composition of this
#: length improves a complex that no shorter composition improves, which means
#: looking there first. Zero keeps the deepening schedule.
#: ``--combinatorial-breadth`` remains a hidden command-line and Python API
#: alias.
DECLARED_COMBINATORIAL_LENGTH = 0
DECLARED_COMBINATORIAL_BREADTH = DECLARED_COMBINATORIAL_LENGTH
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
#: The state the whole complex's harmonic form is meant to be (`--output-state`),
#: as a modulus tau: the target is psi(tau) = (1, tau)/|(1, tau)|.
#:
#: The harmonic of the whole Laplacian IS a state. On the supported two-torus
#: host its rank is 2, and this names that 2-vector. It is required because the
#: ordinary two-body target has four tensor-product amplitudes and cannot stand
#: in for it.
DECLARED_OUTPUT_STATE = None
#: Absolute objective tolerance. Two roles, both absolute and never relative.
#:
#: Stage 2 backs its line search off until a trial lowers the exact selected
#: objective by at least this much; and the drive stops once a whole engine
#: unit fails to improve the objective by it. A run that stops that way says
#: so -- the terminator is recorded, never inferred from a short trace.
DECLARED_TOLERANCE = 1e-12
#: CONSECUTIVE units that must fail to improve the objective before the drive
#: stops.
#:
#: A unit that does not improve the objective has not established that the run
#: is finished. Stage 1 draws `DECLARED_CANDIDATE_MOVES` candidate moves at
#: random each unit, so a unit that commits nothing is one unlucky draw and
#: the next unit draws again; ending on the first of them spends a single draw
#: and calls the result convergence.
#:
#: One -- stop on the first stalled unit -- is the drive's long-standing
#: behaviour and stays the default, so raising it is a deliberate act by a
#: caller who wants the extra draws rather than a silent change of meaning for
#: every run already recorded.
DECLARED_PATIENCE = 1
#: Objective register degrees.
DECLARED_REGISTER_DEGREES = (1,)
#: Laplacian degrees the Hodge entropy term is scored at.
#:
#: Declared independently of the register degrees, which answer the unrelated
#: question of where a register is constructed. All four carry distinct
#: information here: the discrete weighted operator does not inherit the
#: continuum Hodge duality that would make k and 4-k the same condition counted
#: twice on a closed 4-manifold.
#:
#: Scoring more degrees makes the objective see more of the SPECTRUM, not more
#: of the TOPOLOGY. Exact zero modes are omitted from the entropy and from its
#: derivative, so each degree is blind to a change in its own kernel dimension.
DECLARED_HODGE_DEGREES = (0, 1, 2, 3)
#: Post-hoc analysis degrees. Separate from the objective's domain.
DECLARED_ANALYSIS_DEGREES = (1,)
#: Modularity resolution the clusters are read at.
DECLARED_RESOLUTION = 1.0
#: Node seed and host seed.
DECLARED_SEED = 7
DECLARED_HOST_SEED = 3
#: Degrees the Betti numbers are reported at.
DECLARED_BETTI_DEGREES = (0, 1, 2)


class InputMode:
    """What the drive's node is built from, as a closed vocabulary.

    `drive` selects its node factory by this value: the neutral host of the
    existing mode, or two qubit tori on their collar. Named constants rather
    than bare strings: the value is written by `build_config` and compared
    by `drive`, `draw_frame` and every mode-dependent read, and a typo in any
    of them would select a different experiment without failing.
    """

    #: The neutral refined 4-ball with M0 held -- unforced emergence, the
    #: existing mode and the default.
    NEUTRAL = "neutral"
    #: Two flat qubit tori as the boundary of a 3-complex seeded as their
    #: collar and synthesized against the two-body target (the qubit
    #: cobordism spec, delta D4).
    QUBIT = "qubit"

    #: Every accepted value, in help order.
    ALL = (NEUTRAL, QUBIT)


#: The input mode when the caller names none.
DECLARED_INPUTS = InputMode.NEUTRAL
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
#: How often the `--live` main thread services the GUI event loop while the
#: worker computes, in seconds -- about twenty times a second.
#:
#: The main thread has nothing to do between finished units except keep the
#: window alive, and an engine unit can take minutes. Waiting in the event loop
#: at this interval rather than in a blocking queue read is what keeps the
#: window responsive; the cost is one wakeup per interval, which is nothing
#: beside a unit.
LIVE_POLL_INTERVAL = 0.05
#: Whether the Regge stationarity term is in the objective (the engine's
#: `einstein_hilbert`); off, r_U is the whole objective.
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


class EdgeDisposition:
    """The causal character the seed's edges are given, as a closed vocabulary.

    `Edge` stores the length `l`, not the squared length, so a disposition is
    written through `l` and read as `l^2`. A purely imaginary length squares
    to a real negative, which IS the timelike condition, so the whole
    vocabulary is expressible as a choice of `arg l`.

    Named constants rather than bare strings: each value is written where the
    host is built and compared where the seed is assigned, and a typo in
    either place would not fail to compile -- it would silently select a
    different causal structure.
    """

    #: Magnitude one, real/imaginary split uniformly at random per edge:
    #: `l = e^{i a}`, so `l^2 = e^{2 i a}` sweeps the unit circle. The only
    #: setting that prescribes no causal structure, and the default.
    RANDOM = "random"
    #: `l = 1`, so `l^2 = +1` on every edge.
    SPACELIKE = "spacelike"
    #: `l = i`, so `l^2 = -1` on every edge.
    TIMELIKE = "timelike"
    #: `l = (1 + i)/sqrt(2)`, so `Re(l) = Im(l) > 0` and `l^2 = i` on every
    #: edge: the argument sits on the light cone, `arg(l^2) = pi/2`. The
    #: interval vanishes while the edge keeps unit extent, so this is the
    #: NON-TRIVIAL lightlike case, distinct from the degenerate `l = 0`.
    LIGHTLIKE = "lightlike"
    #: Timelike BETWEEN hop layers of M0, spacelike WITHIN a layer. A
    #: PRESCRIBED foliation: it imposes a causal order rather than letting one
    #: emerge, and must be reported as such wherever it is used.
    FOLIATED = "foliated"

    #: Every accepted value, in help order.
    ALL = (RANDOM, SPACELIKE, TIMELIKE, LIGHTLIKE, FOLIATED)


#: The seed disposition when the caller names none.
DECLARED_EDGE_DISPOSITION = EdgeDisposition.RANDOM


class Terminator:
    """Why a drive stopped, as a closed vocabulary.

    A short trace is ambiguous on its face: it looks the same whether the run
    exhausted its units or converged. The terminator says which, so a reader
    never has to infer it from the frame count.

    Named constants rather than bare strings: the value is written where the
    loop exits and compared where a document is read, and a typo in either
    place would produce a run that reports a terminator nothing matches.
    """

    #: The unit budget ran out. The run has not converged; it stopped.
    STEPS = "steps-exhausted"
    #: `patience` consecutive engine units failed to improve the objective by
    #: the declared absolute tolerance. The run stopped early and says so.
    #:
    #: This is a STALL, not an achievement: it says the drive stopped moving,
    #: never that the objective reached any particular value. A run that
    #: stops here at a large residual has stopped just as surely as one that
    #: stops at machine zero, and `DriveResult.stalls` records how many units
    #: it took so the two are told apart in the document.
    TOLERANCE = "tolerance-reached"
    #: A cooperative stop callback requested cancellation between engine
    #: calls. Live-mode interrupts use this to join the worker before exposing
    #: its geometry; a direct caller can also distinguish cancellation from an
    #: exhausted unit budget.
    CANCELLED = "cancelled"

    #: Every value a drive may report.
    ALL = (STEPS, TOLERANCE, CANCELLED)


class DriveResult:
    """A drive's frames together with WHY it stopped.

    The terminator belongs to the loop, not to any one frame: a frame is a
    measurement of the complex at a unit, while the terminator is a fact
    about the run that produced them. Returning them together keeps a frame
    from carrying a field that is meaningless for every frame but the last.

    `inputs` is what the node factory handed the reads -- the `QubitInputs`
    of the qubit mode, or None in the neutral mode -- a fact about the run
    as well, recorded once in the run document rather than on every frame.

    `stalls` is how many CONSECUTIVE units failed to improve the objective
    when the drive stopped. It is recorded rather than inferred because the
    terminator alone cannot distinguish a run that stopped on its first
    stalled unit from one that stopped after twenty of them, and those are
    different claims about the geometry.
    """

    __slots__ = ("frames", "terminator", "inputs", "stalls")

    def __init__(self, frames, terminator, inputs=None, stalls=0):
        if terminator not in Terminator.ALL:
            raise ValueError(
                "unknown terminator %r: expected one of %s"
                % (terminator, ", ".join(Terminator.ALL)))
        self.frames = frames
        self.terminator = terminator
        self.inputs = inputs
        self.stalls = int(stalls)

    def __len__(self):
        return len(self.frames)


class CausalClass:
    """How one edge reads causally: LABELS for `Edge`'s own classification.

    Not a second definition of causal type. `causal_class` dispatches on
    `Edge.isSpacelike()` and its siblings, so this panel and every certificate
    elsewhere answer from ONE classifier. A driver-local rule would eventually
    disagree with the engine about a physical property, which is exactly the
    condition that makes a picture untrustworthy.

    Causal type is the ARGUMENT of `l^2`. Writing `l = |l| e^{i a}` gives
    `l^2 = |l|^2 e^{2 i a}`, so the three definite dispositions sit at
    `arg(l^2) = 0`, `+/-pi` and `+/-pi/2`, and anything else is genuinely
    MIXED. Classifying on the SIGN of `Re(l^2)` would discard `Im(l^2)`, which
    is `2x^2 != 0` exactly at the lightlike point -- a fully null `l^2 = 0`
    does not exist non-trivially -- so the interval's sign cannot carry the
    whole statement, and every generic edge would be forced into a definite
    bucket it does not belong in.
    """

    #: `arg(l^2) ~ 0`: `l^2` real positive.
    SPACELIKE = "spacelike"
    #: `arg(l^2) ~ +/-pi`: `l^2` real negative.
    TIMELIKE = "timelike"
    #: `arg(l^2) ~ +/-pi/2`: `l^2` purely imaginary and NONZERO, reached at
    #: `Re(l) = +/- Im(l)`, both nonzero. A populated, physical case.
    LIGHTLIKE = "lightlike"
    #: A generic argument: no definite causal character. NOT rounded to the
    #: nearest of the three, which would report a definiteness the geometry
    #: does not have. Under a uniformly drawn argument this is the common
    #: case, and its FRACTION across engine units is the diagnostic of whether
    #: relaxation imposes a disposition. An all-mixed panel is a finding.
    MIXED = "mixed"
    #: `l = 0` in both parts: an absent edge, which is not a causal type at
    #: all. Reported apart so it can never be read as lightlike.
    DEGENERATE = "degenerate"

    ALL = (SPACELIKE, TIMELIKE, LIGHTLIKE, MIXED, DEGENERATE)


#: One colour per causal class, and the legend reads from this map, so the
#: drawn colour and its label can never disagree. Mixed is deliberately the
#: neutral grey: it is the absence of a definite character, not a fourth one.
DECLARED_CAUSAL_COLOURS = {
    CausalClass.SPACELIKE: "#1f4e79",
    CausalClass.TIMELIKE: "#a33227",
    CausalClass.LIGHTLIKE: "#c9a227",
    CausalClass.MIXED: "#8c8c8c",
    CausalClass.DEGENERATE: "#7a5aa8",
}

#: Stabilization of the drawing layout. Classical MDS is defined only up to
#: rotation, reflection and scale and is globally sensitive, so a small change
#: to the complex reshuffles the whole cloud. Scale is already fixed by the
#: RMS normalization the layout read performs; these three remove the
#: orientation ambiguity, ease the positions, and ease the view.
DECLARED_LAYOUT_EASE = 0.3
DECLARED_LAYOUT_VIEW_EASE = 0.25
DECLARED_LAYOUT_PAD = 0.18

#: The two dual-curvature channels. Diverging maps, because both channels are
#: SIGNED and centred at zero -- a sequential map would make a saddle
#: indistinguishable from a peak.
DECLARED_HEAT_CMAP_SPATIAL = "coolwarm"
DECLARED_HEAT_CMAP_TEMPORAL = "PuOr"
#: Clip the heat range at this percentile of |curvature| so one extreme cell
#: cannot flatten the rest of the field to a single colour.
DECLARED_HEAT_CLIP_PERCENTILE = 95
#: A curvature channel whose largest magnitude is at or below this is reported
#: as identically zero. Separate from the null-length tolerance above: that
#: one asks whether an edge sits on the light cone, this one asks whether a
#: whole channel has anything to draw.
DECLARED_HEAT_ZERO_TOLERANCE = 1e-9


def causal_class(edge):
    """The `CausalClass` of one edge, read from the LIBRARY predicates.

    Takes the live `Edge` rather than a length so the panel and the engine
    answer from one classifier. `Edge` classifies on `arg(l^2)` with its own
    declared angular tolerance; re-deriving that here would put a second
    definition of a physical property in the driver, and the two would drift.

    The order matters and mirrors the library's: degenerate is tested FIRST,
    because `arg(0)` is `0` and an absent edge would otherwise read spacelike.
    """
    if edge.isDegenerate():
        return CausalClass.DEGENERATE
    if edge.isSpacelike():
        return CausalClass.SPACELIKE
    if edge.isTimelike():
        return CausalClass.TIMELIKE
    if edge.isNull():
        return CausalClass.LIGHTLIKE
    return CausalClass.MIXED


# =====================================================================
# the neutral host -- a cobordism, because the readouts need a boundary
# =====================================================================

def boundary_vertices(spacetime):
    """The vertices of the incoming boundary M0, or [] if the complex is closed.

    A boundary facet is a (d-1)-simplex with exactly one coface. This is the
    same rule the crossing panel applies, kept in one place so the host and
    the readout cannot disagree about what M0 is.

    The facets are reached through the TOP cells rather than through
    `getSimplices()`: the lower skeleton is not materialized as registered
    simplices until something builds it, so `getSimplices()` on a fresh
    complex returns the top cells alone and would report every complex as
    closed.

    TWO passes, and the order matters. `getFacets()` REGISTERS the coface
    relation as a side effect, so a facet's coface count is only complete
    once every top cell has been visited. Counting in the same pass that
    materializes would see each facet before its second cell had registered
    and report the whole complex as boundary.
    """
    facets = {}
    for top in spacetime.getTopSimplices():
        for facet in top.getFacets():
            facets[facet.__hash__()] = facet
    boundary = set()
    for facet in facets.values():
        if len(facet.getCofaces()) == 1:
            for vertex in facet.getVertices():
                boundary.add(int(vertex.getId()))
    return sorted(boundary)


def _edge_endpoints(edge):
    """An edge's two vertex ids, through the bound accessors.

    `Edge` exposes its endpoints as vertices, not as a key pair, so an id is
    read through `getSource`/`getTarget` rather than by indexing.
    """
    return int(edge.getSource().getId()), int(edge.getTarget().getId())


def _hop_layers(spacetime, sources):
    """Hop distance from `sources` over the 1-skeleton, per vertex id.

    This is the SAME layering `CrossingReadouts::temporalFunction` derives
    from M0, recomputed here only to assign the causal character consistently
    with it. Assigning by any other partition would put a causal edge inside
    a layer, which the temporal-function certificate names and refuses.
    """
    layer = {vertex: 0 for vertex in sources}
    frontier = list(sources)
    adjacency = {}
    for edge in spacetime.getEdgeList().toVector():
        a, b = _edge_endpoints(edge)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for vertex in frontier:
            for neighbour in adjacency.get(vertex, ()):
                if neighbour not in layer:
                    layer[neighbour] = depth
                    nxt.append(neighbour)
        frontier = nxt
    return layer


def _seed_lengths(spacetime, disposition, seed):
    """Write the seed length on every edge, per the chosen disposition.

    Every setting carries magnitude one, so the dispositions differ ONLY in
    `arg l` -- in causal character, never in scale. Reproducible from `seed`
    through a private generator, so the global random state is untouched.
    """
    if disposition not in EdgeDisposition.ALL:
        raise ValueError(
            "unknown edge disposition %r: expected one of %s"
            % (disposition, ", ".join(EdgeDisposition.ALL)))
    edges = spacetime.getEdgeList().toVector()
    if disposition == EdgeDisposition.SPACELIKE:
        for edge in edges:
            edge.setLength(complex(1.0, 0.0))            # l^2 = +1
        return
    if disposition == EdgeDisposition.TIMELIKE:
        for edge in edges:
            edge.setLength(complex(0.0, 1.0))            # l^2 = -1
        return
    if disposition == EdgeDisposition.LIGHTLIKE:
        # Re(l) == Im(l) > 0 with |l| = 1, so l^2 = i exactly: the interval
        # vanishes on an edge of unit extent. Both parts are the SAME double,
        # so x^2 - t^2 cancels to exactly zero and arg(l^2) is exactly pi/2.
        component = math.sqrt(0.5)
        for edge in edges:
            edge.setLength(complex(component, component))  # l^2 = i
        return
    if disposition == EdgeDisposition.RANDOM:
        generator = random.Random(seed)
        for edge in edges:
            angle = generator.uniform(0.0, 2.0 * math.pi)
            edge.setLength(cmath.exp(1j * angle))        # |l| = 1
        return
    # FOLIATED: the causal character follows the hop layering of M0, which is
    # the same layering the temporal function derives. Edges spanning layers
    # are timelike and edges inside one spacelike -- a prescribed light cone.
    layer = _hop_layers(spacetime, boundary_vertices(spacetime))
    for edge in edges:
        a, b = _edge_endpoints(edge)
        spans_layers = layer.get(a, 0) != layer.get(b, 0)
        edge.setLength(complex(0.0, 1.0) if spans_layers
                       else complex(1.0, 0.0))


def build_cobordism_host(n_refine=DECLARED_SIZE, seed=DECLARED_HOST_SEED,
                         disposition=DECLARED_EDGE_DISPOSITION):
    """A single simplex, refined -- the canonical seed.

    The paper's crossing readouts live on a cobordism: `tau` is the Lorentzian
    distance FROM the incoming boundary M0, and the surfaces are its level
    sets. A CLOSED complex has no such surface, so those readouts cannot run
    on one at any size. A single 4-simplex is a 4-BALL, so it HAS a boundary
    -- `S^3 = M0` -- structurally, rather than by carving one out of a closed
    manifold. The whitepaper prescribes no host topology; it specifies
    cobordisms with `∂W = M0 ⊔ M1`, and this is the smallest complex that is
    one.

    The seed's causal character is `disposition`, an `EdgeDisposition`. Every
    setting carries magnitude one, so they differ only in `arg l` -- in causal
    character, never in scale. The refinement runs on a uniform real unit
    length REGARDLESS of the disposition, so the topology a given `seed`
    produces is the same under all four and the settings are comparable as a
    controlled variable; the disposition is written afterwards, over the
    finished complex.

    NEUTRAL otherwise: no holes, no pinned carrier, no boundary blocks, no
    target register. Whatever the run comes to carry is read afterwards.
    `foliated` is the exception and is not neutral: it prescribes a causal
    order rather than letting one emerge, and is labelled as such.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                     1.0, 1.0, T.PREFERRED, T.SolidSimplex(4))
    st.build()
    for edge in st.getEdgeList().toVector():
        edge.setLength(complex(1.0, 0.0))
    applied = 0
    for step in range(seed, seed + n_refine * 4):
        move = T.AddMove(st, step, False, T.PachnerMode.PreGeometric, False)
        if move.propose() and move.apply():
            applied += 1
        if applied >= n_refine:
            break
    _seed_lengths(st, disposition, seed)
    return st


# =====================================================================
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
    boundary = []
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
    if int(config.get("tori", DECLARED_TORI)) == 4:
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
    markings = [_host_marking(torus, ids[index])
                for index, torus in enumerate(tori)]
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


#: The node factories `drive` selects by `config["inputs"]`. The neutral
#: mode's factory is spelled inside `drive` itself, because it IS the
#: existing drive; only the other modes are registered here.
NODE_FACTORIES = {InputMode.QUBIT: build_qubit_node}


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
# small helpers -- unknown is None, never zero
# =====================================================================

def _finite(value):
    """A float, or None when the value is not a finite measurement."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reasons(read, attribute="failedCertificates"):
    """The NAMED reasons a read refused, as a list of strings."""
    named = getattr(read, attribute, None)
    if not named:
        return []
    return [str(reason) for reason in named]


class Absent:
    """A channel that has no measurement, carrying WHY.

    Not a zero and not an empty list: a statement that the measurement does
    not exist, with the reason the paper's certificate names.
    """

    __slots__ = ("reason",)

    def __init__(self, reason):
        self.reason = str(reason)

    def __repr__(self):                                   # pragma: no cover
        return "Absent(%r)" % self.reason

    def to_json(self):
        return {"absent": True, "reason": self.reason}


def _json_safe(value):
    """Recursively convert a measurement block to JSON-safe values."""
    if isinstance(value, Absent):
        return value.to_json()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, complex):
        return [_json_safe(value.real), _json_safe(value.imag)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# =====================================================================
# the read -- every panel's measurement, all read-only
# =====================================================================

class EmergenceFrame:
    """Every measurement drawn on one frame, assembled over one geometry.

    Read-only: nothing here touches the geometry or the objective. Each
    channel is either a measurement or an `Absent` carrying the named reason,
    so a panel can always say what it is showing.
    """

    #: The channels of the neutral mode's instrument: the paper's
    #: certificates, read in the order their reads chain.
    CERTIFICATE_CHANNELS = ("clusters", "bands", "anchors", "transports",
                            "statistics", "crossings", "spin", "verdict")
    #: The channels of the qubit mode's instrument: the read-outs of spec S6.
    QUBIT_CHANNELS = ("blocks", "leaks", "monodromy", "two_body", "boundary",
                      "completion")

    def __init__(self, node, spacetime, step, config, inputs=None):
        self.step = step
        self.config = config
        self.spacetime = spacetime
        #: The `QubitInputs` the frame was read against, or None in the
        #: neutral mode. Not a measurement and not in the record: the run
        #: document carries the inputs once.
        self.inputs = inputs
        self.objective = self._read_objective(node)
        self.layout = self._read_layout(
            spacetime, inputs.highlight() if inputs is not None else ())
        if inputs is None:
            # Drawing-only, like `layout`: neither appears in `to_json`, so
            # the record is unchanged by anything the figure needs.
            self.dual = self._read_dual_curvature(spacetime)
            self._read_certificates(spacetime, config)
            for name in self.QUBIT_CHANNELS:
                setattr(self, name, Absent(
                    "no qubit inputs: the neutral mode drives no input torus"))
            return
        # The qubit host is 3-dimensional and its hinges are edges; the dual
        # curvature panels read triangle hinges of a 4-dimensional host.
        self.dual = Absent("dual curvature is drawn from the triangle hinges "
                           "of a 4-dimensional host; the qubit host is "
                           "3-dimensional")
        for name in self.CERTIFICATE_CHANNELS:
            setattr(self, name, Absent(
                "not read in the qubit input mode: the baryon certificates "
                "are the neutral mode's instrument"))
        self.betti = self._read_betti(spacetime, config)
        self._read_qubit_channels(node, spacetime, inputs)

    def _read_certificates(self, spacetime, config):
        """The paper's certificate channels, in the order the reads chain:
        the bands need the clusters, the anchors the bands' states, the
        transports and crossings the accepted bands, the verdict the quark
        reads."""
        self.clusters = self._read_clusters(spacetime, config)
        self.bands = self._read_bands(spacetime, config)
        self.states = self._build_states()
        self.anchors = self._read_anchors()
        self.transports = self._read_transports(spacetime)
        self.statistics = self._read_statistics(spacetime)
        self.crossings = self._read_crossings(spacetime)
        self.spin = self._read_spin()
        self.betti = self._read_betti(spacetime, config)
        self.verdict = self._read_verdict()

    def _read_qubit_channels(self, node, spacetime, inputs):
        """The read-outs of spec S6 over the live complex, all read-only."""
        self.blocks = [self._read_block(node, inputs, index)
                       for index in range(len(inputs.tori))]
        self.leaks = self._read_restricted_leaks(spacetime, inputs)
        self.monodromy = self._read_monodromy(spacetime, inputs)
        self.two_body = self._read_two_body(node, self.config)
        self.boundary = self._read_boundary(spacetime, node)
        self.completion = self._read_completion(node)

    # ---- 1. the objective -------------------------------------------

    @staticmethod
    def _read_objective(node):
        terms = node.objective_terms()
        # `objective_of` is static over the terms record, so the total is a
        # fold of the terms the frame already holds. `node.objective()` would
        # be the same double, reached by computing those terms a second time
        # and discarding all but the scalar -- on a four-torus complex that is
        # a second whole-complex band read, and the band is the whole cost.
        block = {"total": _finite(MC.objective_of(terms))}
        for name in MC.objective_term_names():
            block[name] = _finite(getattr(terms, name, None))
        # Which DEGREE the Hodge share came from, not only the total. The
        # unweighted norm is carried alongside the weighted share so the raw
        # spread across degrees stays visible rather than folded into the
        # weighting.
        block["hodge_by_degree"] = [
            {"degree": contribution.degree,
             "weight": _finite(contribution.weight),
             "gradient_norm_squared":
                 _finite(contribution.gradient_norm_squared),
             "contribution": _finite(contribution.contribution)}
            for contribution in node.hodge_degree_contributions]
        return block

    # ---- 2. the drawing layout --------------------------------------

    @staticmethod
    def _read_layout(spacetime, highlight=()):
        """2-D coordinates per vertex for DRAWING ONLY.

        Classical multidimensional scaling on graph shortest paths under
        |l^2|^(1/2). This is a picture, not a spacetime coordinate system:
        it carries no causal content and no position here means anything
        physical.

        `highlight` names edge sets to draw apart -- `(label, colour, id
        pairs)` per set, the qubit mode's input tori -- and rides along as
        `highlight` in the returned block, indexed like `edges`; empty in the
        neutral mode.
        """
        try:
            import numpy as np
            from scipy.sparse.csgraph import shortest_path
        except ImportError:
            return Absent("numpy/scipy unavailable: no drawing layout")
        edges = spacetime.getEdgeList().toVector()
        vertices = sorted({int(v.getId())
                           for edge in edges
                           for v in (edge.getSource(), edge.getTarget())})
        if len(vertices) < 2:
            return Absent("fewer than two vertices: nothing to lay out")
        index = {v: i for i, v in enumerate(vertices)}
        n = len(vertices)
        weights = np.full((n, n), np.inf)
        np.fill_diagonal(weights, 0.0)
        pairs = []
        lengths = []
        classes = []
        arguments = []
        for edge in edges:
            a = index.get(int(edge.getSource().getId()))
            b = index.get(int(edge.getTarget().getId()))
            if a is None or b is None:
                continue
            length = complex(edge.getLength())
            w = math.sqrt(max(abs(length ** 2), 1e-6))
            weights[a, b] = weights[b, a] = min(weights[a, b], w)
            pairs.append((a, b))
            # Carried per drawn edge so the causal colouring reads the same
            # `l` the geometry holds, rather than re-deriving it from a
            # disposition setting that only describes how the SEED was built.
            lengths.append(length)
            # Classified by the LIBRARY, from the live edge -- see
            # `causal_class`. `arg(l^2)` rides along as the measured value, so
            # a reader can see where an edge actually sits rather than only
            # which bucket it fell in.
            classes.append(causal_class(edge))
            arguments.append(edge.squaredArgument())
        distances = shortest_path(weights, method="D", directed=False)
        finite = np.isfinite(distances)
        if not finite.any():
            return Absent("no finite graph distance: complex is disconnected")
        distances[~finite] = distances[finite].max() * 1.5
        squared_distances = distances ** 2
        centering = np.eye(n) - np.ones((n, n)) / n
        gram = -0.5 * centering @ squared_distances @ centering
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1][:2]
        coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
        coords = coords - coords.mean(0)
        rms = math.sqrt((coords ** 2).sum(1).mean()) or 1.0
        marked = []
        for label, colour, keys in highlight:
            drawn = [(index[u], index[v]) for u, v in sorted(keys)
                     if u in index and v in index]
            marked.append({"label": label, "colour": colour, "edges": drawn,
                           "missing": len(keys) - len(drawn)})
        return {"coords": {vertices[i]: tuple(coords[i] / rms)
                           for i in range(n)},
                "edges": pairs,
                "edge_lengths": lengths,
                "edge_intervals": [(z * z).real for z in lengths],
                "edge_causal_classes": classes,
                "edge_causal_arguments": arguments,
                "vertices": vertices,
                "highlight": marked}

    # ---- 2b. dual curvature, for the two heat panels ----------------

    @staticmethod
    def _read_dual_curvature(spacetime):
        """Per-top-cell Regge curvature, both channels, for DRAWING ONLY.

        Curvature in Regge calculus lives on the hinges -- the `(d-2)`
        simplices, which in 4D are the triangles. Each is weighted by its own
        dual measure and summed onto the top cells that contain it, giving one
        signed value per dual node.

        The Lorentzian deficit is COMPLEX and the two parts are different
        physics, so they are kept apart rather than collapsed to a magnitude:
        `Re eps * |star|` is the rotation angle-defect carried by timelike
        hinges, and `Im eps * |star|` is the boost / light-cone content
        carried by spacelike hinges -- those whose normal plane is timelike.
        Both keep their sign, so a saddle stays distinguishable from a peak.

        Like the drawing layout, this is computed for the figure and is NOT
        part of the record: it appears in no `to_json` block.
        """
        try:
            import numpy as np  # noqa: F401  (parity with the layout read)
        except ImportError:
            return Absent("numpy unavailable: no dual curvature")
        hinge = {}
        for simplex in spacetime.getSimplices():
            vertices = simplex.getVertices()
            if len(vertices) != 3:
                continue
            key = tuple(sorted(int(v.getId()) for v in vertices))
            try:
                deficit = complex(simplex.deficitAngle())
                weight = abs(complex(simplex.dualVolume()))
            except RuntimeError:
                # A boundary or degenerate hinge carries no deficit. Only that
                # geometric failure is absorbed -- a contract failure
                # (TypeError, ValueError) must propagate rather than render as
                # zero curvature, which would be indistinguishable from a flat
                # hinge.
                continue
            hinge[key] = (deficit.real * weight, deficit.imag * weight)
        if not hinge:
            # The hinges are the triangles, and they are only enumerable once
            # the lower skeleton has been materialized. Saying so beats
            # drawing a field of zeros that would read as "measured, flat".
            return Absent("no hinge carries a deficit: the lower skeleton "
                          "is not materialized")
        cells = []
        for cell in spacetime.getTopSimplices():
            ids = sorted(int(v.getId()) for v in cell.getVertices())
            faces = [tuple(t) for t in itertools.combinations(ids, 3)]
            cells.append({
                "vertices": ids,
                "spatial": sum(hinge.get(f, (0.0, 0.0))[0] for f in faces),
                "temporal": sum(hinge.get(f, (0.0, 0.0))[1] for f in faces),
            })
        if not cells:
            return Absent("no top cells: nothing to draw a dual over")
        rows, cols, _count = spacetime.getDualAdjacency()
        return {"cells": cells,
                "adjacency": list(zip(list(rows), list(cols))),
                "hinges_with_curvature": len(hinge)}

    # ---- 3. persistent modular clusters -----------------------------

    def _read_clusters(self, spacetime, config):
        modularity = obs.PersistentModularity.fromSpacetime(spacetime)
        settings = obs.PersistentModularityConfig()
        settings.resolutions = [config["resolution"]]
        settings.baseSeed = config["seed"]
        report = modularity.scanResolutions(settings)
        if not report.slices:
            self.components = []
            return Absent("modularity returned no resolution slice")
        self.slice = report.slices[0]
        self.components = list(self.slice.components)
        if not self.components:
            return Absent("no persistent cluster at the analysis resolution")
        sizes = [len(list(c.support)) for c in self.components]
        return {"count": len(self.components),
                "sizes": sizes,
                "resolution": config["resolution"],
                "modularity": _finite(getattr(self.slice, "modularity", None))}

    # ---- 4. fibers: rank, gap, localization -------------------------

    def _read_bands(self, spacetime, config):
        self.candidates = []
        self.candidate_components = []
        if not self.components:
            return Absent("no cluster to carry a band")
        settings = obs.SpectralFiberConfig()
        settings.degrees = list(config["degrees"])
        tracker = obs.SpectralFiberTracker(spacetime, settings)
        rows = []
        for component in self.components:
            for degree in config["degrees"]:
                self.candidate_components.append(component)
                try:
                    read = tracker.enumerateBands(component.support, degree)
                except Exception as error:                # noqa: BLE001
                    self.candidates.append(None)
                    rows.append({"accepted": False,
                                 "reason": "band enumeration failed: %s"
                                           % error})
                    continue
                chosen = None
                for fiber in read.fibers:
                    if fiber.accepted():
                        chosen = fiber
                        break
                self.candidates.append(chosen)
                if chosen is None:
                    named = []
                    for fiber in read.fibers:
                        named.extend(_reasons(fiber.certificate(),
                                              "failedCertificates"))
                    rows.append({"accepted": False,
                                 "offered": len(read.fibers),
                                 "reason": ", ".join(sorted(set(named)))
                                           or "no band met the certificate"})
                    continue
                certificate = chosen.certificate()
                rows.append({
                    "accepted": True,
                    "degree": int(chosen.degree()),
                    "rank": int(chosen.rank()),
                    "lowerGap": _finite(certificate.lowerGap),
                    "upperGap": _finite(certificate.upperGap),
                    "localization": _finite(certificate.localization),
                    "localizationExcess": _finite(
                        certificate.localizationExcess),
                    "gramDefect": _finite(certificate.gramDefect),
                })
        if not rows:
            return Absent("no band read was attempted")
        return {"rows": rows,
                "accepted": sum(1 for r in rows if r.get("accepted"))}

    def _build_states(self):
        """The pure Slater covariance of each accepted band projector.

        A band whose restricted metric fails positivity supplies no quasi-free
        covariance; that band's slot stays None rather than becoming a zero.
        """
        states = []
        for fiber in self.candidates:
            if fiber is None:
                states.append(None)
                continue
            try:
                states.append(
                    qu.CovarianceState.fromBandProjector(fiber.projector()))
            except Exception:                             # noqa: BLE001
                states.append(None)
        return states

    # ---- 5. the anchor profile --------------------------------------

    def _read_anchors(self):
        """The quark reads, which carry the triangle-anchor profile.

        `classifyQuark` is the library's own evaluation of the paper's quark
        conditions, and its read reports the anchor score, maximal term,
        participation ratio and determinant-phase dispersion the paper asks
        for -- together with every certificate that failed, NAMED.
        """
        self.quarks = []
        self.candidate_quarks = [None] * len(self.candidates)
        accepted = [f for f in self.candidates if f is not None]
        if not accepted:
            return Absent("no accepted band: nothing to anchor")
        classifier = obs.ParticleClusters()
        rows = []
        for index, fiber in enumerate(self.candidates):
            if fiber is None:
                continue
            evidence = obs.QuarkCandidateEvidence()
            evidence.colorBand = fiber
            components = getattr(self, "candidate_components",
                                 self.components)
            if index < len(components):
                evidence.component = components[index].id
            state = self.states[index] if index < len(self.states) else None
            if state is not None:
                evidence.parityRead = state.wickParity()
                evidence.occupationRead = state.wickTotalNumber()
            # This overlay reads ONE cobordism frame, so the frame lifetime is
            # one and its adjacent-frame overlap is vacuously one. Both are
            # MEASURED facts about this read.
            evidence.frameLifetime = 1.0
            evidence.frameMinOverlap = 1.0
            try:
                read = classifier.classifyQuark(evidence)
            except Exception as error:                    # noqa: BLE001
                rows.append({"reason": "quark read failed: %s" % error})
                continue
            self.candidate_quarks[index] = read
            self.quarks.append(read)
            rows.append({
                "classification": str(read.classification),
                "colorRank": int(read.colorRank),
                "score": _finite(read.triangleAnchorScore),
                "maxTerm": _finite(read.triangleAnchorMaxTerm),
                "participationRatio": _finite(
                    read.triangleAnchorParticipation),
                "phaseDispersion": _finite(read.anchorPhaseDispersion),
                "phaseCoherence": _finite(read.anchorPhaseCoherence),
                "reasons": _reasons(read),
            })
        if not rows:
            return Absent("no band produced a quark read")
        return {"rows": rows,
                "certified": sum(1 for r in rows
                                 if r.get("classification") == "quark")}

    # ---- 6. transports, leakage, holonomy ---------------------------

    def _read_transports(self, spacetime):
        accepted = [f for f in self.candidates if f is not None]
        if len(accepted) < 2:
            return Absent("fewer than two accepted bands: no ordered pair to "
                          "transport between")
        connection = obs.FiberConnection()
        rows = []
        for i, to_fiber in enumerate(accepted):
            for j, from_fiber in enumerate(accepted):
                if i == j:
                    continue
                if to_fiber.degree() != from_fiber.degree():
                    continue
                if to_fiber.rank() != from_fiber.rank():
                    continue
                try:
                    read = connection.transportOnSpacetime(
                        spacetime, to_fiber, from_fiber)
                except Exception as error:                # noqa: BLE001
                    rows.append({
                        "accepted": False,
                        "leakage": None,
                        "regime": "",
                        "reason": "transport failed: %s" % error,
                    })
                    continue
                reason = str(getattr(read, "rejectionReason", "") or "")
                rows.append({
                    "accepted": bool(read.accepted),
                    "leakage": _finite(getattr(read, "leakage", None)),
                    "regime": str(getattr(read, "regime", "")),
                    "reason": reason,
                })
        if not rows:
            return Absent("no same-degree same-rank pair: transport is "
                          "defined only between matching bands")
        return {"rows": rows,
                "accepted": sum(1 for r in rows if r["accepted"]),
                "total": len(rows)}

    # ---- 7. exchange and rotation characters ------------------------

    def _read_statistics(self, spacetime):
        accepted = [f for f in self.candidates if f is not None]
        if len(accepted) < 2:
            return Absent("the Berry-cancelled exchange needs two odd "
                          "clusters to interchange")
        return Absent("no exchange cobordism was constructed on this host: "
                      "the reference loop requires a non-exchanging motion "
                      "of the same geometric footprint")

    # ---- 8. the world-tube crossing readouts ------------------------

    @staticmethod
    def _m0_vertices(spacetime):
        """The incoming boundary's vertices, or a named absence on failure.

        Delegates to the module-level rule the host is built against, so the
        host and the readout cannot disagree about what M0 is. A closed
        complex has none, and then there is no M0 -- which the paper's
        readouts require, so the channel refuses rather than inventing one.
        """
        try:
            return boundary_vertices(spacetime)
        except Exception as error:                        # noqa: BLE001
            return Absent("incoming boundary read failed: %s" % error)

    @staticmethod
    def _regular_crossing_level(temporal):
        """A regular central level derived from the measured Re(tau).

        The range midpoint is already regular unless it is itself a vertex
        value. In that case, use the midpoint of the nearest adjacent gap;
        crossing at a vertex is not a regular level-set read.
        """
        values = []
        for value in getattr(temporal, "tau", ()):
            try:
                real = float(complex(value).real)
            except (TypeError, ValueError):
                continue
            if math.isfinite(real):
                values.append(real)
        levels = sorted(set(values))
        if len(levels) < 2:
            raise ValueError("Re tau has fewer than two distinct finite "
                             "vertex values")
        centre = 0.5 * (levels[0] + levels[-1])
        if centre not in levels:
            return centre
        gaps = [(0.5 * (lower + upper), lower, upper)
                for lower, upper in zip(levels, levels[1:])
                if upper > lower]
        return min(gaps, key=lambda gap: (abs(gap[0] - centre), gap[0]))[0]

    def _read_crossings(self, spacetime):
        self.crossing_candidate_quarks = []
        self.crossing_mass_read = None
        self.crossing_baryon_read = None
        self.crossing_candidate_read_failures = []
        accepted = [f for f in self.candidates if f is not None]
        if not accepted:
            return Absent("no accepted band: no world tube to cross a level")
        boundary = self._m0_vertices(spacetime)
        if isinstance(boundary, Absent):
            return boundary
        if not boundary:
            return Absent(
                "closed host: no incoming boundary M0, so tau has no "
                "reference surface. The crossing readouts are defined on a "
                "cobordism with boundary M0 + M1")
        try:
            temporal = obs.CrossingReadouts.temporalFunction(
                spacetime, boundary)
        except Exception as error:                        # noqa: BLE001
            return Absent("temporal function unavailable: %s" % error)
        if not temporal.certified:
            return Absent("Re tau is not a certified temporal function (%s); "
                          "no level set is admissible"
                          % (", ".join(_reasons(temporal))
                             or "no reason named"))
        tubes = []
        quark_tubes = []
        quark_reads = []
        candidate_quarks = getattr(
            self, "candidate_quarks", [None] * len(self.candidates))
        for index, fiber in enumerate(self.candidates):
            if fiber is None:
                continue
            tube = obs.WorldTubeInput()
            tube.tubeId = "band-%d" % index
            tube.band = fiber
            tube.orientation = +1
            quark = (candidate_quarks[index]
                     if index < len(candidate_quarks) else None)
            certified = (quark is not None
                         and str(quark.classification) == "quark")
            tube.certifiedQuarkTube = certified
            winding = (getattr(quark, "determinantWinding", None)
                       if quark is not None else None)
            if winding is not None:
                tube.determinantWinding = int(winding)
            tubes.append(tube)
            if certified:
                quark_tubes.append(tube)
                quark_reads.append(quark)
        try:
            level = self._regular_crossing_level(temporal)
        except ValueError as error:
            return Absent("no regular Re tau crossing level: %s" % error)
        block = {"level": level, "tubes": len(tubes)}
        mass = None
        try:
            mass = obs.CrossingReadouts.crossingMass(tubes, temporal,
                                                    level, 0.0)
            block["crossingMass"] = _finite(mass.crossingMass)
            block["admissible"] = int(mass.admissibleCrossings)
            block["refused"] = int(mass.refusedCrossings)
            block["calibrated"] = bool(mass.calibrated)
            block["units"] = str(mass.units)
        except Exception as error:                        # noqa: BLE001
            block["crossingMass"] = Absent("crossing mass failed: %s" % error)
            if len(tubes) == 3 and len(quark_tubes) == 3:
                self.crossing_candidate_read_failures.append(
                    "candidate crossing mass failed: %s" % error)
        baryon = None
        try:
            baryon = obs.CrossingReadouts.baryonNumber(tubes, temporal,
                                                      level, 0.0)
            block["baryonNumber"] = _finite(baryon.baryonNumber)
            block["quarkTubes"] = int(baryon.quarkTubes)
            # Named defects, not a count: a tube whose crossing sign
            # disagrees with its determinant-line winding is reported, never
            # silently resolved.
            block["signDefects"] = [str(d) for d in baryon.signDefects]
        except Exception as error:                        # noqa: BLE001
            block["baryonNumber"] = Absent("baryon sum failed: %s" % error)
            if len(tubes) == 3 and len(quark_tubes) == 3:
                self.crossing_candidate_read_failures.append(
                    "candidate baryon sum failed: %s" % error)
        if not quark_tubes:
            block["baryonNote"] = ("no tube carries a quark certificate, so "
                                   "the one-third sum has no term")
        elif ("quarkTubes" in block and not block["quarkTubes"]):
            block["baryonNote"] = (
                "certified quark tubes exist, but none has an admissible "
                "crossing at this level")

        # A baryon candidate has exactly three constituent reads. Keep the
        # corresponding crossing bundle together and never pad it with an
        # uncertified tube or a default-constructed quark read.
        if len(quark_tubes) >= 3:
            candidate_tubes = quark_tubes[:3]
            self.crossing_candidate_quarks = quark_reads[:3]
            if len(tubes) == 3 and len(quark_tubes) == 3:
                self.crossing_mass_read = mass
                self.crossing_baryon_read = baryon
            else:
                try:
                    self.crossing_mass_read = (
                        obs.CrossingReadouts.crossingMass(
                            candidate_tubes, temporal, level, 0.0))
                except Exception as error:                # noqa: BLE001
                    self.crossing_candidate_read_failures.append(
                        "candidate crossing mass failed: %s" % error)
                try:
                    self.crossing_baryon_read = (
                        obs.CrossingReadouts.baryonNumber(
                            candidate_tubes, temporal, level, 0.0))
                except Exception as error:                # noqa: BLE001
                    self.crossing_candidate_read_failures.append(
                        "candidate baryon sum failed: %s" % error)
        try:
            profile = obs.CrossingReadouts.chargePowerProfile(
                tubes, temporal, level)
            block["chargePower"] = {
                "eigenvalues": [_finite(x) for x in profile.eigenvalues],
                "power": [_finite(x) for x in profile.power],
                "normalized": bool(profile.normalized),
                "monopole": _finite(profile.monopole),
                "reasons": _reasons(profile),
            }
        except Exception as error:                        # noqa: BLE001
            block["chargePower"] = Absent("charge power failed: %s" % error)
        signs = []
        for tube in tubes:
            try:
                crossing = obs.CrossingReadouts.crossing(tube, temporal,
                                                         level)
            except Exception as error:                    # noqa: BLE001
                signs.append({
                    "tubeId": str(tube.tubeId),
                    "sign": None,
                    "admissible": False,
                    "perpendicular": None,
                    "reasons": ["crossing read failed: %s" % error],
                })
                continue
            signs.append({
                "tubeId": str(crossing.tubeId),
                "sign": int(crossing.sign),
                "admissible": bool(crossing.admissible),
                "perpendicular": complex(crossing.perpendicular),
                "reasons": _reasons(crossing),
            })
        block["crossings"] = signs
        if self.crossing_candidate_read_failures:
            block["candidateReadFailures"] = list(
                self.crossing_candidate_read_failures)
        return block

    # ---- 9. <J^2> and Var(J^2) --------------------------------------

    def _read_spin(self):
        accepted = [f for f in self.candidates if f is not None]
        if not accepted:
            return Absent("no accepted band: no covariance to Wick-contract")
        live = [s for s in self.states if s is not None]
        if not live:
            return Absent("no band projector yielded a quasi-free covariance; "
                          "a band whose restricted metric fails positivity "
                          "supplies no state")
        certified = [q for q in self.quarks
                     if str(q.classification) == "quark"]
        if len(certified) < 3:
            return Absent("the total-space J^2 read needs three certified "
                          "quark clusters; %d covariance state(s) exist and "
                          "%d cluster(s) are certified quarks"
                          % (len(live), len(certified)))
        return Absent("three certified quarks exist but no total-space spin "
                      "operator was assembled on this host")

    # ---- 10. Betti numbers ------------------------------------------

    @staticmethod
    def _read_betti(spacetime, config):
        """Independent topological observables.

        The paper keeps Betti numbers as observables in their own right. They
        are NOT a quark count and NOT a success condition: the quark is sought
        as a persistent modular spectral cluster, not as a hole.
        """
        try:
            values = list(MC.betti(spacetime))
        except Exception as error:                        # noqa: BLE001
            return Absent("Betti numbers unavailable: %s" % error)
        numbers = {degree: (int(values[degree]) if degree < len(values)
                            else None)
                   for degree in config["betti_degrees"]}
        if all(v is None for v in numbers.values()):
            return Absent("the complex reports no Betti number at the "
                          "declared degrees")
        return {"numbers": numbers}

    # ---- 11. the verdict --------------------------------------------

    def _read_verdict(self):
        accepted = [f for f in self.candidates if f is not None]
        certified = [q for q in self.quarks
                     if str(q.classification) == "quark"]
        if len(certified) < 3:
            return Absent(
                "the proton certificate is evaluated on three certified "
                "quark clusters; %d band(s) accepted, %d quark read(s), %d "
                "certified" % (len(accepted), len(self.quarks),
                               len(certified)))
        evidence = obs.BaryonCandidateEvidence()
        crossing_quarks = getattr(self, "crossing_candidate_quarks", [])
        chosen = (crossing_quarks if len(crossing_quarks) == 3
                  else certified[:3])
        evidence.quarks = chosen
        crossing_mass = getattr(self, "crossing_mass_read", None)
        crossing_baryon = getattr(self, "crossing_baryon_read", None)
        if crossing_mass is not None:
            evidence.crossingMass = crossing_mass
        if crossing_baryon is not None:
            evidence.crossingBaryon = crossing_baryon
        read_failures = list(getattr(
            self, "crossing_candidate_read_failures", []))
        try:
            read = obs.ParticleClusters().classifyBaryon(evidence)
        except Exception as error:                        # noqa: BLE001
            detail = ("; " + "; ".join(read_failures)
                      if read_failures else "")
            return Absent("classifier refused: %s%s" % (error, detail))
        result = {"classification": str(read.classification),
                  "confidence": _finite(read.confidence),
                  "reasons": _reasons(read)}
        if read_failures:
            result["readFailures"] = read_failures
        return result

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

    # ---- serialization ----------------------------------------------

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
        # The qubit channels ride along only when they were read: the
        # neutral mode's record keeps its schema byte for byte.
        if self.inputs is not None:
            for name in self.QUBIT_CHANNELS:
                document[name] = getattr(self, name)
        return _json_safe(document)


# =====================================================================
# the drive -- unforced emergence, one frame per engine unit
# =====================================================================

def drive(config, progress=False, on_frame=None, on_node=None, on_setup=None,
          stop_requested=None):
    """Drive unforced emergence, reading a frame after every engine unit.

    `on_frame(frames, index)` is called as each unit completes, so a caller
    can display a run while it is still running. `on_node(node)` is the
    backward-compatible node-only setup observer; `on_setup(node, inputs)`
    also carries the qubit inputs needed to serialize its boundary blocks.
    `stop_requested()` is checked between engine calls so a live worker can
    finish cooperatively before interrupted geometry is inspected. The
    callbacks are otherwise observers: they do not alter an uninterrupted
    drive. It is the ONLY difference
    between a live drive and a headless one: the loop, the engine calls and
    the frames are the same either way, so a live view cannot diverge from
    the run it claims to be showing.

    The node is built by a FACTORY selected by `config["inputs"]`: the
    neutral host with its held M0 region (the default, spelled here because
    it is the existing drive) or one of `NODE_FACTORIES` -- the qubit mode's
    two tori on their collar. The factory returns the node and what the frame
    reads need to know about its inputs (None for the neutral host); the loop
    below is the same for every factory, so the modes cannot diverge in how
    they are driven, only in what they drive.

    Returns a `DriveResult` carrying the frames, the terminator and the
    inputs.
    """
    combinatorial_depth = _cpp_int_value(
        "combinatorial depth", _config_aliased_value(
            config, "combinatorial_depth", "surgical_depth",
            DECLARED_COMBINATORIAL_DEPTH))
    combinatorial_length = _cpp_int_value(
        "combinatorial length", _config_aliased_value(
            config, "combinatorial_length", "combinatorial_breadth",
            DECLARED_COMBINATORIAL_LENGTH))
    if combinatorial_length > 0 and (
            combinatorial_depth != DECLARED_COMBINATORIAL_DEPTH):
        raise ValueError(
            "combinatorial_depth and combinatorial_length select alternative "
            "search schedules; a non-default depth would be ignored when "
            "length is nonzero")

    def neutral_node(config):
        host = build_cobordism_host(config["size"], config["host_seed"],
                                    config["edge_disposition"])
        node = MC(host, [], [], list(config["register_degrees"]), 1.0,
                  config["seed"])
        node.set_objective(cob.JointStationarityObjective())
        # Declared here rather than inherited from the register degrees
        # above: the degrees a register is constructed at and the degrees
        # whose entropy should be stationary are different questions.
        node.set_hodge_degrees(list(config["hodge_degrees"]))
        node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                                 MC.EmergenceSubmode.STRICT)
        # M0 is HELD, not targeted. Declaring the region says only WHICH
        # cells do not vary -- the paper's fixed boundary with a relaxed
        # bulk. No pinned objective is set, so the bulk objective scores the
        # whole cobordism including M0 and the run stays bit-identical to an
        # unpinned one in everything except which coordinates are free.
        node.declare_pinned_region(M0_REGION, set(boundary_vertices(host)))
        return node, None

    input_mode = config.get("inputs", DECLARED_INPUTS)
    if input_mode == InputMode.NEUTRAL:
        factory = neutral_node
    elif input_mode in NODE_FACTORIES:
        factory = NODE_FACTORIES[input_mode]
    else:
        raise ValueError("unknown input mode %r: expected one of %s"
                         % (input_mode, ", ".join(InputMode.ALL)))
    node, inputs = factory(config)
    if on_node is not None:
        on_node(node)
    if on_setup is not None:
        on_setup(node, inputs)
    # Only the qubit surface blocks declare a fixed topological boundary. M0
    # in neutral mode is a pinned region: it holds geometry but deliberately
    # does not arm the topology gate.
    if input_mode == InputMode.QUBIT:
        node.set_boundary_may_extend(bool(config["extend_boundary"]))
    if config.get("pin_boundary", DECLARED_PIN_BOUNDARY) and inputs is not None:
        # The input states are inputs: the attachment already put the correct
        # boundary in place, so stage 2 has nothing to improve there. A pinned
        # region zeroes the descent on the edges inside it, which is exactly
        # the block's own surface.
        for index in range(len(node.inputs)):
            node.declare_pinned_region(
                "input%d" % index, set(int(v) for v in node.inputs[index].vertices))

    # EVERY frame reads `node.spacetime()`, never the host handed to the
    # constructor. Stage 1 REPLACES the node's complex when it commits a move,
    # so the host stops being the complex the node is driving from the first
    # committed move onward. Reading it would freeze every panel at the initial
    # geometry while the objective tracked something else entirely -- the two
    # diverge silently, with no error and no empty frame to give it away.
    frames = [EmergenceFrame(node, node.spacetime(), 0, config, inputs)]
    if progress:
        _report(frames[-1])
    if on_frame is not None:
        on_frame(frames, 0)
    terminator = Terminator.STEPS
    patience = max(1, int(config.get("patience", DECLARED_PATIENCE)))
    # CONSECUTIVE stalled units, reset by any unit that improves. Counting
    # consecutively rather than cumulatively is the point of the knob: a run
    # that stalls, recovers and stalls again is still making progress, and
    # a cumulative count would end it for the arithmetic of its history
    # rather than for anything true of its current geometry.
    stalls = 0
    for step in range(1, config["steps"] + 1):
        if stop_requested is not None and stop_requested():
            terminator = Terminator.CANCELLED
            break
        before = _objective_total(frames[-1])
        # Stage 1 before stage 2 within a unit (spec S4): a committed
        # combinatorial move rebuilds the complex from a snapshot of its
        # cells and every edge's length and phase, and stage 2 then relaxes
        # every edge of the rebuilt complex.
        list(node.run_stage1(
            max_steps=config["stage1_iters"],
            n_candidate_moves=config["candidate_moves"],
            max_lookahead=combinatorial_depth,
            combinatorial_breadth=combinatorial_length))
        if stop_requested is not None and stop_requested():
            terminator = Terminator.CANCELLED
            break
        list(node.run_stage2(max_iters=config["stage2_iters"],
                             tolerance=config["tolerance"]))
        if stop_requested is not None and stop_requested():
            terminator = Terminator.CANCELLED
            break
        frames.append(EmergenceFrame(node, node.spacetime(), step, config,
                                     inputs))
        if progress:
            _report(frames[-1])
        if on_frame is not None:
            on_frame(frames, step)
        # The unit is complete and its frame is published before the exit is
        # considered, so a run that stops here has still reported the unit
        # that stopped it.
        #
        # A stall on the FINAL unit reports `tolerance-reached` even though
        # the budget also ran out. Both are true, and this is the more
        # informative of the two: a reader learns the run had stopped moving,
        # and can still see it used its whole budget from the unit count,
        # which the document and the stdout line both carry.
        if _converged(before, _objective_total(frames[-1]),
                      config["tolerance"]):
            stalls += 1
            if stalls >= patience:
                terminator = Terminator.TOLERANCE
                break
        else:
            stalls = 0
    return DriveResult(frames, terminator, inputs, stalls)


def _format_objective_total(frame):
    """A frame's objective for a human, or a named absence.

    The stall line quotes the value the run stopped at so a reader is not
    left to infer from the word `tolerance` that it must have been small.
    """
    total = _objective_total(frame)
    return "unmeasured" if total is None else "%.6e" % total


def _objective_total(frame):
    """A frame's scalar objective, or None where it was not measured."""
    total = frame.objective.get("total")
    return total if isinstance(total, (int, float)) else None


def _converged(before, after, tolerance):
    """Whether one engine unit failed to improve the objective by `tolerance`.

    ABSOLUTE, never relative: the test is on the improvement itself, not on
    its ratio to the objective, so the same tolerance means the same thing at
    every scale the objective happens to take.

    An unmeasured objective at either end is not convergence. It is an
    absence, and a run may not stop on one -- stopping there would report a
    converged run on the strength of a number nobody read.
    """
    if before is None or after is None:
        return False
    if not (math.isfinite(before) and math.isfinite(after)):
        return False
    # A unit that RAISES the objective has also failed to improve it by the
    # tolerance, so this stops on that too rather than running on.
    return (before - after) < tolerance


def _report(frame):
    """One line per frame on stdout, so a long run is legible while it runs."""
    if frame.inputs is not None:
        return _report_qubit(frame)
    verdict = (frame.verdict.reason if isinstance(frame.verdict, Absent)
               else frame.verdict["classification"])
    clusters = ("absent" if isinstance(frame.clusters, Absent)
                else frame.clusters["count"])
    bands = ("absent" if isinstance(frame.bands, Absent)
             else frame.bands["accepted"])
    total = frame.objective.get("total")
    sys.stdout.write(
        "[step %2d] objective %s | clusters %s | accepted bands %s | %s\n"
        % (frame.step,
           "n/a" if total is None else "%.6g" % total,
           clusters, bands, verdict))
    sys.stdout.flush()


def _tau_text(value):
    return "n/a" if value is None else "%.5f%+.5fi" % (value.real, value.imag)


def _report_qubit(frame):
    """The qubit mode's stdout line: the residuals, the moduli, the topology."""
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


# =====================================================================
# the overlay -- an absent panel still says what is absent, and why
# =====================================================================

class StableLayout:
    """Jitter-free drawing positions: the layout read's normalized embedding,
    rigidly aligned to the previous frame and eased toward it, with an eased
    auto-fit view.

    Classical MDS is defined only up to rotation, reflection and scale, and it
    is globally sensitive, so a small change to the complex can reshuffle the
    whole cloud. Scale is already fixed by the read's RMS normalization; this
    removes the orientation ambiguity by Procrustes, eases the positions, and
    auto-fits the view, so the structure stays legible as the complex changes.

    PRESENTATION ONLY. It consumes what the layout read measured and produces
    where to draw it; it feeds nothing back into any measurement or record.
    """

    def __init__(self, ease=DECLARED_LAYOUT_EASE,
                 view_ease=DECLARED_LAYOUT_VIEW_EASE,
                 pad=DECLARED_LAYOUT_PAD):
        self._previous = None
        self._view = None
        self.ease = ease
        self.view_ease = view_ease
        self.pad = pad

    def place(self, coords):
        """Stabilized positions for one frame's raw layout coordinates."""
        import numpy as np

        current = {v: np.asarray(p, dtype=float) for v, p in coords.items()}
        if len(current) < 2 or self._previous is None:
            # The first frame defines the frame of reference; there is nothing
            # to align to and nothing to ease toward.
            self._previous = current
            return {v: tuple(p) for v, p in current.items()}
        shared = [v for v in current if v in self._previous]
        if len(shared) >= 2:
            cur = np.array([current[v] for v in shared])
            ref = np.array([self._previous[v] for v in shared])
            cur_centre, ref_centre = cur.mean(0), ref.mean(0)
            u, _s, vt = np.linalg.svd((cur - cur_centre).T
                                      @ (ref - ref_centre))
            rotation = u @ vt          # rotation/reflection only, no scale
            aligned = {v: (p - cur_centre) @ rotation + ref_centre
                       for v, p in current.items()}
        else:
            # Too little in common to define an alignment. Taking the raw
            # embedding is honest; inventing a rotation from one point is not.
            aligned = current
        eased = {}
        for vertex, target in aligned.items():
            previous = self._previous.get(vertex)
            # A vertex that has just appeared has nowhere to ease FROM, so it
            # takes its target outright rather than sliding in from a position
            # it never occupied.
            eased[vertex] = (target if previous is None
                             else previous + self.ease * (target - previous))
        self._previous = eased
        return {v: tuple(p) for v, p in eased.items()}

    def view(self, coords):
        """An eased bounding box around the current cloud.

        Never grow-only: a view that could only expand would shrink the
        structure to an unreadable dot as soon as one frame spread out.
        """
        import numpy as np

        points = np.array(list(coords.values()), dtype=float)
        low, high = points.min(0), points.max(0)
        pad = self.pad * max(high[0] - low[0], high[1] - low[1], 1e-6)
        box = [low[0] - pad, high[0] + pad, low[1] - pad, high[1] + pad]
        if self._view is None:
            self._view = box
        else:
            self._view = [self._view[i]
                          + self.view_ease * (box[i] - self._view[i])
                          for i in range(4)]
        return self._view


def place_frame(state, frame):
    """One frame's stabilized placement, advancing `state` by one link.

    The single step both drivers share. `stabilize` walks it over a finished
    run and the live path calls it as each unit completes; because the
    alignment is a CHAIN, the two agree only if they feed the same state the
    same frames in the same order. Sharing the step makes that structural
    rather than a coincidence two call sites have to maintain.

    `None` where the layout itself is absent, so the chain skips a frame it
    cannot place rather than aligning the next one to nothing.
    """
    if isinstance(frame.layout, Absent):
        return None
    coords = state.place(frame.layout["coords"])
    return {"coords": coords, "view": state.view(coords)}


def stabilize(frames):
    """Every frame's stabilized positions and view, computed once in order.

    Precomputed rather than accumulated during drawing because the alignment
    is a chain: each frame is aligned to the one before it. A renderer that
    redraws a frame, or draws only the last, would otherwise get a different
    picture depending on what it had drawn before.

    Returns one entry per frame, `None` where the layout itself is absent.
    """
    state = StableLayout()
    return [place_frame(state, frame) for frame in frames]


def _absent_panel(axis, title, reason):
    """Draw a legible statement of absence. Never a blank, never a zero."""
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#bbbbbb")
        spine.set_linestyle(":")
    axis.text(0.5, 0.5, _wrap(reason, 34), transform=axis.transAxes,
              ha="center", va="center", fontsize=6.5, color="#666666",
              style="italic", wrap=True)


def _wrap(text, width):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width)[:6])


def _panel_objective(axis, frames):
    axis.set_title("objective trace", fontsize=8)
    totals = [f.objective.get("total") for f in frames]
    steps = [f.step for f in frames]
    finite = [(s, t) for s, t in zip(steps, totals) if t is not None]
    if not finite:
        return _absent_panel(axis, "objective trace",
                             "the objective returned no finite value")
    axis.plot([s for s, _ in finite], [t for _, t in finite],
              marker="o", markersize=2.5, linewidth=1.0, color="#1f4e79")
    axis.set_xlabel("engine unit", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.25, linewidth=0.4)


def _panel_layout(axis, frame, placement=None):
    # POSITION is a drawing artefact; COLOUR is a measurement. The two are
    # deliberately named apart in the title, because stabilizing the layout
    # and colouring the edges together make the picture look more physical
    # than it is -- where a vertex sits still means nothing at all.
    title = "complex -- position: drawing only | colour: causal arg(l^2)"
    if isinstance(frame.layout, Absent):
        return _absent_panel(axis, title, frame.layout.reason)
    from matplotlib.lines import Line2D

    coords = (placement or {}).get("coords") or frame.layout["coords"]
    axis.set_title(title, fontsize=8)
    vertices = frame.layout["vertices"]
    classes = frame.layout["edge_causal_classes"]
    seen = []
    for (a, b), causal in zip(frame.layout["edges"], classes):
        va = coords[vertices[a]]
        vb = coords[vertices[b]]
        axis.plot([va[0], vb[0]], [va[1], vb[1]], linewidth=0.7,
                  color=DECLARED_CAUSAL_COLOURS[causal], zorder=1)
        if causal not in seen:
            seen.append(causal)
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    axis.scatter(xs, ys, s=6, color="#333333", zorder=2)
    counts = {c: classes.count(c) for c in seen}
    handles = [Line2D([0], [0], color=DECLARED_CAUSAL_COLOURS[c],
                      linewidth=1.4,
                      label="%s (%d)" % (_CAUSAL_LEGEND[c], counts[c]))
               for c in CausalClass.ALL if c in counts]
    # The drawn boundary: the qubit mode's input tori, drawn over the causal
    # colouring in their own colours so the surfaces the residuals hold can
    # be told from the bulk between them. Empty in the neutral mode.
    for mark in frame.layout.get("highlight", []):
        for a, b in mark["edges"]:
            va = coords[vertices[a]]
            vb = coords[vertices[b]]
            axis.plot([va[0], vb[0]], [va[1], vb[1]], linewidth=1.7,
                      color=mark["colour"], alpha=0.9, zorder=3)
        handles.append(Line2D([0], [0], color=mark["colour"], linewidth=2.0,
                              label="torus %s (%d edges)"
                                    % (mark["label"], len(mark["edges"]))))
    if handles:
        axis.legend(handles=handles, fontsize=5, loc="upper right",
                    frameon=True, framealpha=0.85, borderpad=0.3,
                    handlelength=1.2)
    # No disagreement note: the colouring now dispatches on `Edge`'s own
    # predicates (see `causal_class`), so the panel and every certificate
    # elsewhere answer from one classifier and cannot disagree.
    if placement and placement.get("view"):
        view = placement["view"]
        axis.set_xlim(view[0], view[1])
        axis.set_ylim(view[2], view[3])
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")


#: What each causal class means, spelled out in the legend rather than left to
#: the colour alone. `degenerate` is the one that needs saying, since an absent
#: edge also has a vanishing interval but is not lightlike.
#: Legend text, stated in the quantity the classification actually reads --
#: `arg(l^2)` -- so the label cannot suggest a rule the classifier does not use.
_CAUSAL_LEGEND = {
    CausalClass.SPACELIKE: "spacelike  arg l^2 = 0",
    CausalClass.TIMELIKE: "timelike  arg l^2 = +/-pi",
    CausalClass.LIGHTLIKE: "lightlike  arg l^2 = +/-pi/2, l != 0",
    CausalClass.MIXED: "mixed  no definite arg",
    CausalClass.DEGENERATE: "degenerate  l = 0",
}


def _panel_dual(axis, frame, placement, channel, title, cmap):
    """One dual-curvature panel: dual nodes at primal cell centroids, edges
    across shared facets, heat-coloured by one signed channel."""
    if isinstance(frame.dual, Absent):
        return _absent_panel(axis, title, frame.dual.reason)
    if isinstance(frame.layout, Absent):
        return _absent_panel(axis, title,
                             "no drawing layout: nowhere to place the dual")
    import numpy as np

    coords = (placement or {}).get("coords") or frame.layout["coords"]
    cells = frame.dual["cells"]
    positions = np.full((len(cells), 2), np.nan)
    values = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        here = [coords[v] for v in cell["vertices"] if v in coords]
        if here:
            positions[i] = np.mean(np.asarray(here, dtype=float), axis=0)
        values[i] = cell[channel]
    finite = np.all(np.isfinite(positions), axis=1)
    axis.set_title("%s  (%d cells)" % (title, len(cells)), fontsize=8)
    if not finite.any():
        return _absent_panel(
            axis, title, "no dual node has a drawable position")
    for a, b in frame.dual["adjacency"]:
        if a < len(cells) and b < len(cells) and finite[a] and finite[b]:
            axis.plot([positions[a, 0], positions[b, 0]],
                      [positions[a, 1], positions[b, 1]],
                      color="0.82", linewidth=0.4, zorder=1)
    shown = values[finite]
    magnitude = np.abs(shown)
    if magnitude.max() <= DECLARED_HEAT_ZERO_TOLERANCE:
        # A channel that is identically zero would draw as one flat colour and
        # read as "measured, uniform". Say which channel vanished and why it
        # can: no hinge of the kind that carries it.
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(0.5, 0.5,
                  _wrap("identically 0: no hinge carries %s curvature here"
                        % channel, 30),
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=6.5, color="#666666", style="italic")
        return
    limit = (float(np.percentile(magnitude, DECLARED_HEAT_CLIP_PERCENTILE))
             if finite.sum() >= 5 else float(magnitude.max()))
    if not limit > 0:
        limit = float(magnitude.max()) or 1.0
    clipped = np.clip(shown, -limit, limit)
    scatter = axis.scatter(positions[finite, 0], positions[finite, 1],
                           c=clipped, cmap=cmap, vmin=-limit, vmax=limit,
                           s=18, zorder=2, edgecolors="0.3", linewidths=0.2)
    # Horizontal and beneath the panel: a vertical bar steals width from the
    # axes, which pulls the centred title off its own panel and into the
    # neighbour's colourbar.
    bar = axis.figure.colorbar(scatter, ax=axis, orientation="horizontal",
                               fraction=0.05, pad=0.04, aspect=40)
    # The centre is the whole point of a diverging map: without the zero tick
    # labelled, a reader cannot tell a saddle from a peak.
    bar.set_ticks([-limit, 0.0, limit])
    bar.ax.tick_params(labelsize=5)
    bar.set_label("signed, 0 at centre", fontsize=5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")


def _panel_dual_spatial(axis, frame, placement=None):
    return _panel_dual(
        axis, frame, placement, "spatial",
        "dual spatial: Re eps*|star| (timelike hinges)",
        DECLARED_HEAT_CMAP_SPATIAL)


def _panel_dual_temporal(axis, frame, placement=None):
    return _panel_dual(
        axis, frame, placement, "temporal",
        "dual temporal: Im eps*|star| (spacelike hinges)",
        DECLARED_HEAT_CMAP_TEMPORAL)


def _panel_clusters(axis, frame):
    title = "persistent modular clusters"
    if isinstance(frame.clusters, Absent):
        return _absent_panel(axis, title, frame.clusters.reason)
    sizes = frame.clusters["sizes"]
    axis.set_title(title, fontsize=8)
    axis.bar(range(len(sizes)), sizes, color="#608c64", width=0.7)
    axis.set_xlabel("cluster", fontsize=6)
    axis.set_ylabel("support cells", fontsize=6)
    axis.tick_params(labelsize=6)


def _panel_bands(axis, frame):
    title = "fibers: rank, gap, localization"
    if isinstance(frame.bands, Absent):
        return _absent_panel(axis, title, frame.bands.reason)
    rows = frame.bands["rows"]
    accepted = [r for r in rows if r.get("accepted")]
    if not accepted:
        reason = rows[0].get("reason") if rows else "no band read"
        return _absent_panel(axis, title,
                             "no band met its certificate: %s" % reason)
    axis.set_title(title, fontsize=8)
    ranks = [r["rank"] for r in accepted]
    axis.bar(range(len(ranks)), ranks, color="#608c64", width=0.5,
             label="rank")
    gaps = [(index, row.get("lowerGap"))
            for index, row in enumerate(accepted)
            if row.get("lowerGap") is not None]
    if gaps:
        twin = axis.twinx()
        twin.plot([index for index, _ in gaps],
                  [gap for _, gap in gaps], marker="s", markersize=3,
                  linewidth=0.8, color="#bc8836", label="lower gap")
        twin.tick_params(labelsize=6)
    else:
        axis.text(0.98, 0.92, "lower gap unmeasured",
                  transform=axis.transAxes, ha="right", va="top",
                  fontsize=5.5, color="#666666", style="italic")
    axis.set_xlabel("accepted band", fontsize=6)
    axis.tick_params(labelsize=6)


def _panel_anchors(axis, frame):
    title = "anchor profile (quark reads)"
    if isinstance(frame.anchors, Absent):
        return _absent_panel(axis, title, frame.anchors.reason)
    rows = frame.anchors["rows"]
    measured = [r for r in rows if r.get("score") is not None]
    if not measured:
        reasons = sorted({x for r in rows for x in (r.get("reasons") or [])})
        return _absent_panel(
            axis, title,
            "no quark read reported an anchor score: %s"
            % (", ".join(reasons[:3]) or "no reason named"))
    axis.set_title(title, fontsize=8)
    labels = ["score", "max term", "participation", "phase disp."]
    first = measured[0]
    values = [first.get("score"), first.get("maxTerm"),
              first.get("participationRatio"), first.get("phaseDispersion")]
    available = [(label, value) for label, value in zip(labels, values)
                 if value is not None]
    shown_labels = [label for label, _ in available]
    shown = [value for _, value in available]
    axis.barh(range(len(shown)), shown, color="#bc8836", height=0.6)
    axis.set_yticks(range(len(shown_labels)))
    axis.set_yticklabels(shown_labels, fontsize=6)
    axis.set_xlim(0.0, max(1.0, max(shown) * 1.15))
    axis.tick_params(labelsize=6)
    axis.text(0.98, 0.06, "%d certified" % frame.anchors["certified"],
              transform=axis.transAxes, ha="right", fontsize=6,
              color="#666666")


def _panel_transports(axis, frame):
    title = "transports: accepted vs rejected"
    if isinstance(frame.transports, Absent):
        return _absent_panel(axis, title, frame.transports.reason)
    accepted = frame.transports["accepted"]
    total = frame.transports["total"]
    axis.set_title(title, fontsize=8)
    axis.bar([0, 1], [accepted, total - accepted],
             color=["#608c64", "#b0302a"], width=0.6)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["accepted", "rejected"], fontsize=6)
    axis.tick_params(labelsize=6)
    reasons = sorted({r["reason"] for r in frame.transports["rows"]
                      if not r["accepted"] and r.get("reason")})
    if reasons:
        axis.set_xlabel(_wrap("rejected: " + ", ".join(reasons[:2]), 42),
                        fontsize=5.5, color="#666666", style="italic")


def _panel_statistics(axis, frame):
    title = "exchange / rotation characters"
    if isinstance(frame.statistics, Absent):
        return _absent_panel(axis, title, frame.statistics.reason)
    return _absent_panel(axis, title, "no character was measured")


def _panel_crossings(axis, frame):
    title = "world-tube crossings: sgn Re pi_perp"
    if isinstance(frame.crossings, Absent):
        return _absent_panel(axis, title, frame.crossings.reason)
    crossings = frame.crossings.get("crossings") or []
    if not crossings:
        return _absent_panel(axis, title,
                             "no tube crossed the level; %d refused"
                             % frame.crossings.get("refused", 0))
    admissible = [c for c in crossings if c["admissible"]]
    if not admissible:
        reasons = sorted({r for c in crossings for r in c["reasons"]})
        return _absent_panel(axis, title,
                             "every crossing refused: %s"
                             % (", ".join(reasons) or "no reason named"))
    axis.set_title(title, fontsize=8)
    signs = [c["sign"] for c in admissible]
    axis.bar(range(len(signs)), signs,
             color=["#1f4e79" if s > 0 else "#b0302a" for s in signs],
             width=0.6)
    axis.axhline(0.0, linewidth=0.5, color="#333333")
    axis.set_ylim(-1.4, 1.4)
    axis.set_xlabel("admissible crossing", fontsize=6)
    axis.tick_params(labelsize=6)


def _panel_mass(axis, frame):
    title = "crossing mass and one-third baryon sum"
    if isinstance(frame.crossings, Absent):
        return _absent_panel(axis, title, frame.crossings.reason)
    mass = frame.crossings.get("crossingMass")
    baryon = frame.crossings.get("baryonNumber")
    mass_reason = mass.reason if isinstance(mass, Absent) else None
    baryon_reason = baryon.reason if isinstance(baryon, Absent) else None
    mass = None if mass_reason is not None else mass
    baryon = None if baryon_reason is not None else baryon
    if mass is None and baryon is None:
        reasons = [reason for reason in (mass_reason, baryon_reason)
                   if reason]
        return _absent_panel(axis, title,
                             "; ".join(reasons)
                             or "no admissible crossing contributes a term")
    axis.set_title(title, fontsize=8)
    notes = [frame.crossings.get("baryonNote", "")]
    if mass_reason:
        notes.append("crossing mass unavailable: %s" % mass_reason)
    if baryon_reason:
        notes.append("baryon sum unavailable: %s" % baryon_reason)
    note = "; ".join(part for part in notes if part)
    units = frame.crossings.get("units", "")
    axis.text(0.05, 0.72, "crossing mass: %s" % ("n/a" if mass is None
                                                 else "%.6g" % mass),
              transform=axis.transAxes, fontsize=7)
    axis.text(0.05, 0.52, "units: %s" % (units or "unknown"),
              transform=axis.transAxes, fontsize=6, color="#666666")
    axis.text(0.05, 0.32, "B = %s" % ("n/a" if baryon is None
                                      else "%.4g" % baryon),
              transform=axis.transAxes, fontsize=7)
    if note:
        axis.text(0.05, 0.10, _wrap(note, 40), transform=axis.transAxes,
                  fontsize=5.5, color="#666666", style="italic")
    axis.set_xticks([])
    axis.set_yticks([])


def _panel_spin(axis, frame):
    title = "<J^2> and Var(J^2)"
    if isinstance(frame.spin, Absent):
        return _absent_panel(axis, title, frame.spin.reason)
    return _absent_panel(axis, title, "no sharp-spin read was assembled")


def _panel_betti(axis, frame):
    title = "Betti numbers (independent observable)"
    if isinstance(frame.betti, Absent):
        return _absent_panel(axis, title, frame.betti.reason)
    numbers = frame.betti["numbers"]
    measured = [(degree, numbers[degree]) for degree in sorted(numbers)
                if numbers[degree] is not None]
    if not measured:
        return _absent_panel(axis, title,
                             "no requested Betti degree was measured")
    degrees = [degree for degree, _ in measured]
    values = [value for _, value in measured]
    axis.set_title(title, fontsize=8)
    axis.bar([str(d) for d in degrees], values, color="#5a7ca0", width=0.6)
    axis.set_xlabel("degree", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.text(0.5, 0.92, "not a quark count", transform=axis.transAxes,
              ha="center", fontsize=5.5, color="#666666", style="italic")
    missing = [str(degree) for degree in sorted(numbers)
               if numbers[degree] is None]
    if missing:
        axis.text(0.5, 0.82,
                  "unmeasured degree(s): %s" % ", ".join(missing),
                  transform=axis.transAxes, ha="center", fontsize=5.5,
                  color="#666666", style="italic")


def _panel_verdict(axis, frame):
    title = "verdict and named reasons"
    axis.set_title(title, fontsize=8)
    axis.set_xticks([])
    axis.set_yticks([])
    if isinstance(frame.verdict, Absent):
        axis.text(0.5, 0.72, "no verdict", transform=axis.transAxes,
                  ha="center", fontsize=9, color="#b0302a")
        axis.text(0.5, 0.36, _wrap(frame.verdict.reason, 40),
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=6, color="#666666", style="italic")
        return
    axis.text(0.5, 0.74, frame.verdict["classification"],
              transform=axis.transAxes, ha="center", fontsize=9,
              color="#1f4e79")
    reasons = frame.verdict.get("reasons") or []
    axis.text(0.5, 0.34, _wrap(", ".join(reasons) or "no reason named", 40),
              transform=axis.transAxes, ha="center", va="center",
              fontsize=6, color="#666666")


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


#: Panels whose painter also takes the frame's stabilized placement. Named
#: rather than detected by signature, so adding a painter that needs it is a
#: deliberate act rather than something that silently starts working.
_PLACED_PANELS = ("layout", "dual_spatial", "dual_temporal")

#: Panels whose painter takes the frames so far rather than one frame: the
#: traces. Named for the same reason as `_PLACED_PANELS`.
_TRACE_PANELS = ("objective", "residuals", "moduli")

#: Every panel, in the neutral mode's order: the existing panels first, the
#: qubit mode's after them. One set of painters serves both modes; a panel
#: whose channel the mode does not read draws its named absence.
_PANELS = [
    ("objective", _panel_objective),
    ("layout", _panel_layout),
    ("dual_spatial", _panel_dual_spatial),
    ("dual_temporal", _panel_dual_temporal),
    ("clusters", _panel_clusters),
    ("bands", _panel_bands),
    ("anchors", _panel_anchors),
    ("transports", _panel_transports),
    ("statistics", _panel_statistics),
    ("crossings", _panel_crossings),
    ("mass", _panel_mass),
    ("spin", _panel_spin),
    ("betti", _panel_betti),
    ("verdict", _panel_verdict),
    ("residuals", _panel_residuals),
    ("moduli", _panel_moduli),
    ("bloch", _panel_bloch),
    ("transfer", _panel_transfer),
    ("topology", _panel_topology),
]

#: The same panels in the qubit mode's order: the panels that read that
#: mode's channels first, the certificate panels (absent there) after.
_QUBIT_PANEL_ORDER = ("objective", "residuals", "moduli", "bloch", "transfer",
                      "topology", "layout", "betti")
_QUBIT_PANELS = ([panel for name in _QUBIT_PANEL_ORDER
                  for panel in _PANELS if panel[0] == name]
                 + [panel for panel in _PANELS
                    if panel[0] not in _QUBIT_PANEL_ORDER])


def panels_for(config):
    """The panel order for a run's input mode (one painter set, two orders)."""
    if config.get("inputs", DECLARED_INPUTS) == InputMode.QUBIT:
        return _QUBIT_PANELS
    return _PANELS


#: Grid the panels are laid out on. Wide enough for every panel with room to
#: spare; the spare axes are removed rather than left as empty boxes, which
#: would read as absent measurements.
DECLARED_PANEL_GRID = (4, 5)


def _suptitle(frame, last_step):
    """The figure's title: what was driven, and that the read-outs are
    post-hoc."""
    if frame.inputs is not None:
        inputs = frame.inputs
        second = 2 if len(inputs.tori) == 4 else 1
        jt = inputs.algebra.get("Jt")
        interaction = ("J t = %g" % jt if isinstance(jt, float) and math.isfinite(jt)
                       else "operator %s" % inputs.algebra["operator"])
        return ("qubit cobordism -- engine unit %d of %d -- tau_A %s, tau_B "
                "%s, %d tori on %dx%d grids, %d layers, %s, readout %s, "
                "input weight %g, Regge term %s, objective %s (read-outs post-hoc)"
                % (frame.step, last_step, _tau_text(inputs.tau_in[0]),
                   _tau_text(inputs.tau_in[second]), len(inputs.tori), inputs.grid,
                   inputs.grid, inputs.layers, interaction, frame.config["readout"],
                   inputs.weight, "on" if inputs.regge else "off",
                   inputs.objective_name))
    disposition = frame.config.get("edge_disposition",
                                   DECLARED_EDGE_DISPOSITION)
    # `foliated` prescribes a causal order rather than letting one emerge, so
    # a frame drawn under it must say so on its face and never read as
    # emergent.
    seed_note = ("seed %s -- a PRESCRIBED foliation, not emergent"
                 % disposition if disposition == EdgeDisposition.FOLIATED
                 else "seed %s" % disposition)
    return ("unforced Regge-Hodge emergence -- engine unit %d of %d -- %s "
            "(certificates read post-hoc, firewalled from the objective)"
            % (frame.step, last_step, seed_note))


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


def draw_frame(figure, frames, index, placed=None):
    """Draw one frame's panels onto a figure.

    `placed` is `stabilize(frames)`. Passing it is optional so a caller can
    draw a single frame without it, in which case the raw layout is used and
    the picture is correct but unaligned.
    """
    figure.clear()
    frame = frames[index]
    placement = placed[index] if placed else None
    panels = panels_for(frame.config)
    rows, columns = DECLARED_PANEL_GRID
    axes = figure.subplots(rows, columns)
    flat = [ax for row in axes for ax in row]
    for axis in flat[len(panels):]:
        figure.delaxes(axis)
    for (name, painter), axis in zip(panels, flat):
        if name in _TRACE_PANELS:
            painter(axis, frames[:index + 1])
        elif name in _PLACED_PANELS:
            painter(axis, frame, placement)
        else:
            painter(axis, frame)
    figure.suptitle(_suptitle(frame, frames[-1].step), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.95))


def _interactive_backends():
    """The interactive matplotlib backends, lowercased.

    Asked of matplotlib rather than hard-coded, so the set cannot drift out
    of step with the installed version. The fallback names the file-only
    backends instead, which is the smaller and far more stable list.
    """
    try:
        from matplotlib.backends import BackendFilter, backend_registry
        return {name.lower() for name in
                backend_registry.list_builtin(BackendFilter.INTERACTIVE)}
    except ImportError:                     # matplotlib < 3.9
        import matplotlib
        try:
            return {name.lower() for name in matplotlib.rcsetup.interactive_bk}
        except AttributeError:
            return set()


def drive_live(config, progress=False, on_node=None, on_setup=None):
    """Drive while drawing each unit as it completes, then return the result.

    The compute runs on a worker thread and the figure is drawn on the main
    one, because a GUI toolkit may only be driven from the thread that owns
    it. That is safe here rather than merely conventional: `run_stage1` and
    `run_stage2` release the GIL, so the worker genuinely proceeds while the
    main thread draws, and the worker only ever APPENDS to the frame list
    while the main thread reads indices it has already been handed.

    The drive itself is `drive`, unchanged and un-forked. A live run and a
    headless one execute the same loop with the same engine calls; only the
    callback differs. There is no second code path to diverge.
    """
    import queue
    import threading

    import matplotlib
    import matplotlib.pyplot as plt

    # Checked on the BACKEND, not on whether a figure can be created: a
    # file-only backend like Agg makes figures perfectly well and simply
    # shows nothing, so creating one successfully proves nothing about
    # whether the caller will ever see a frame. Without this the flag would
    # silently degrade into a slower headless run.
    backend = matplotlib.get_backend()
    backend_name = backend.lower()
    is_webagg = "webagg" in backend_name
    if backend_name not in _interactive_backends() or is_webagg:
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
    if not plt.isinteractive():
        plt.ion()
    figure = plt.figure(figsize=(18, 10))

    ready = queue.Queue()
    published = {}
    outcome = {}
    stop = threading.Event()

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    def worker():
        try:
            outcome["result"] = drive(config, progress=progress,
                                      on_frame=publish, on_node=on_node,
                                      on_setup=on_setup,
                                      stop_requested=stop.is_set)
        except BaseException as exc:            # re-raised on the main thread
            outcome["error"] = exc
        finally:
            ready.put(None)

    thread = threading.Thread(target=worker, name="emergence-drive")
    thread.start()
    # The live path stabilizes through the SAME chained step a headless render
    # walks, advanced once per unit as it completes. Frames are published in
    # index order, so the state sees exactly the sequence `stabilize` would
    # feed it and the two produce identical placements. Accumulated here
    # rather than precomputed because there is no finished run to walk yet.
    state = StableLayout()
    placed = []
    main_error = None
    try:
        while True:
            try:
                index = ready.get_nowait()
            except queue.Empty:
                # POLLED, never blocked. A blocking `get` parks the main thread for
                # the whole of an engine unit, and for that entire interval the GUI
                # event loop is never serviced -- no redraw, no input -- which the
                # desktop reports as a hung application. The worker is computing
                # with the GIL released, so there is nothing to gain by sleeping in
                # the queue rather than in the event loop.
                #
                # `plt.pause` both pumps the event loop and sleeps for the
                # interval, so this waits at LIVE_POLL_INTERVAL without spinning,
                # whether or not the backend has an event loop of its own.
                plt.pause(LIVE_POLL_INTERVAL)
                continue
            if index is None:
                break
            frames = published["frames"]
            while len(placed) <= index:
                placed.append(place_frame(state, frames[len(placed)]))
            draw_frame(figure, frames, index, placed)
            figure.canvas.draw_idle()
            # Yields to the GUI event loop; a backend without one still returns.
            plt.pause(LIVE_POLL_INTERVAL)
    except BaseException as error:
        # Never inspect or serialize the node while its worker can still mutate
        # it. The event is observed between the two engine stages and units;
        # joining makes interrupted geometry a stable snapshot.
        main_error = error
        stop.set()
    finally:
        thread.join()
        plt.close(figure)
    if main_error is not None:
        raise main_error
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def source_commit():
    """The commit this run's code was at, or Absent outside a git work tree.

    A geometry dump is the only output a run cannot recompute from the others,
    and reproducing a figure means knowing which code produced it. `dirty`
    carries as much as `head`: a sha read from a tree with uncommitted changes
    names a commit that does not describe what ran, and a record that said
    otherwise would be worse than one that said nothing.

    Absent rather than guessed when git is unavailable or the tree is not a
    repository -- the same convention every other unmeasurable quantity in
    this document uses.
    """
    import subprocess

    def git(*arguments):
        return subprocess.run(("git",) + arguments, cwd=os.path.dirname(
            os.path.abspath(__file__)), capture_output=True, text=True,
            check=True, timeout=10).stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return Absent("the run is not inside a readable git work tree")
    return {"head": head, "branch": branch, "dirty": dirty}


def geometry_document(node, inputs=None, source=None):
    """The node's live complex, in the schema a rebuild already reads.

    The campaign worker's geometry dump (schema 1, rebuilt verbatim by
    `tests/cobordism/_causal_specimen.rebuild_spacetime`): the dimension, the
    top cells in their intrinsic vertex order, every edge as
    `[source, target, Re l^2, Im l^2]`, and the per-vertex times. That is
    what `Spacetime.fromCells` plus `setLength` and `setTime` need to bring
    the complex back exactly, which the run document cannot do: it records
    measurements of a geometry, never the geometry.

    WHY the squared length rather than the length: `l^2` is the coordinate
    the relaxation moves and the quantity every formula takes, and the
    complex square root has two branches, so writing `l` and reading it back
    would be a branch choice made twice. WHY the times: they are state a
    `Spacetime` carries and `fromCells` does not derive.

    With `inputs` (the qubit mode) each block is recorded too, in the same
    shape as the whole: its own surface as `cells` and `edges`, so a torus
    loads and can be fiddled with on its own -- straight into
    `SimplicialQubit`, without carrying the bulk -- next to its vertex set,
    its marking as directed host steps, and its input coefficients, which a
    rebuilt Spacetime alone cannot supply. The block's edges are the host's
    on those vertices, so a torus and the whole agree edge for edge; the
    surface is written anyway because reconstructing which of the whole's
    edges belong to a torus needs the block, which is the thing being
    recorded.

    Phases are a separate field, written per edge alongside the squared
    length and only where some phase is nonzero.
    """
    spacetime = node.spacetime()
    cells = [[int(v.getId()) for v in cell.getVertices()]
             for cell in spacetime.getTopSimplices()]
    if not cells:
        raise ValueError("the node's complex has no top cell: nothing to write")
    # The container exposes no dimension of its own, so it is read off the
    # top cells: a d-simplex has d + 1 vertices. A complex whose top cells
    # disagree is not one `fromCells` could rebuild, and says so here rather
    # than on the read.
    sizes = sorted({len(cell) for cell in cells})
    if len(sizes) != 1:
        raise ValueError("the top cells have %s vertices: not a pure complex"
                         % ", ".join(str(n) for n in sizes))
    edges = []
    phases = []
    for edge in spacetime.getEdgeList().toVector():
        source_vertex = int(edge.getSource().getId())
        target_vertex = int(edge.getTarget().getId())
        squared = complex(edge.getLength()) ** 2
        edges.append([source_vertex, target_vertex,
                      squared.real, squared.imag])
        phase = complex(edge.getPhase())
        if phase != 0:
            phases.append([source_vertex, target_vertex,
                           phase.real, phase.imag])
    document = {
        "schema": 1,
        "source": source_commit() if source is None else source,
        "dimensions": sizes[0] - 1,
        "cells": cells,
        "edges": edges,
        "vertex_times": [[int(v.getId()), float(v.getTime())]
                         for v in spacetime.getVertexList().toVector()],
    }
    if phases:
        document["edge_phases"] = phases
    if inputs is not None:
        document["blocks"] = [
            _block_geometry(node, inputs, index)
            for index in range(len(inputs.tori))]
    return _json_safe(document)


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


def render(frames, path):
    """Render the overlay to a GIF, MP4 or a single PNG of the last frame."""
    if not frames:
        raise ValueError("cannot render an empty frame sequence")
    lowered = os.fspath(path).lower()
    if not lowered.endswith((".gif", ".mp4", ".png")):
        raise ValueError("--out must end in .gif, .mp4, or .png; got %r"
                         % os.fspath(path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(20, 12))
    # Computed once, in frame order: the alignment is a chain, so a renderer
    # that redraws a frame or draws only the last must still see the same
    # positions it would have seen drawing them all in sequence.
    try:
        placed = stabilize(frames)
        if lowered.endswith(".png"):
            draw_frame(figure, frames, len(frames) - 1, placed)
            figure.savefig(path, dpi=110)
            return path
        import matplotlib.animation as animation

        def update(index):
            draw_frame(figure, frames, index, placed)
            return []

        movie = animation.FuncAnimation(figure, update, frames=len(frames),
                                        interval=900, blit=False, repeat=False)
        writer = "pillow" if lowered.endswith(".gif") else "ffmpeg"
        movie.save(path, writer=writer, dpi=100)
    finally:
        plt.close(figure)
    return path


# =====================================================================
# CLI
# =====================================================================

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


_LEGACY_UNSET = object()
_UINT64_MAX = (1 << 64) - 1
_CPP_INT_MAX = (1 << 31) - 1


def _integer_value(name, value):
    """Return an integer without silently truncating a numeric value."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("%s must be an integer, got %r" % (name, value))
    return int(value)


def _cpp_int_value(name, value):
    """Return a value representable by the C++ ``int`` API it reaches."""
    value = _integer_value(name, value)
    if value > _CPP_INT_MAX:
        raise ValueError("%s must be at most %d to fit the engine's integer "
                         "API, got %r" % (name, _CPP_INT_MAX, value))
    return value


def _finite_value(name, value):
    """Return a finite real number, with a named refusal for API callers."""
    if isinstance(value, bool):
        raise ValueError("%s must be a finite number, got %r" % (name, value))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s must be a finite number, got %r"
                         % (name, value)) from None
    if not math.isfinite(number):
        raise ValueError("%s must be finite, got %r" % (name, value))
    return number


def _boolean_value(name, value):
    """Refuse truthy objects that would otherwise change a declared flag."""
    if not isinstance(value, bool):
        raise ValueError("%s must be true or false, got %r" % (name, value))
    return value


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


def _aliased_value(canonical, legacy, default, canonical_name, legacy_name):
    """Resolve one renamed API keyword, refusing contradictory spellings."""
    if canonical is _LEGACY_UNSET:
        return default if legacy is _LEGACY_UNSET else legacy
    if legacy is not _LEGACY_UNSET and canonical != legacy:
        raise ValueError("%s and its legacy alias %s disagree: %r != %r"
                         % (canonical_name, legacy_name, canonical, legacy))
    return canonical


def _config_aliased_value(config, canonical_name, legacy_name, default):
    """Read a renamed stored-config key and refuse divergent duplicates."""
    canonical = config.get(canonical_name, _LEGACY_UNSET)
    legacy = config.get(legacy_name, _LEGACY_UNSET)
    return _aliased_value(canonical, legacy, default, canonical_name,
                          legacy_name)


def _reject_unused_options(inputs, options):
    """Reject non-default mode-specific values before they become no-ops."""
    changed = [name for name, differs in options if differs]
    if changed:
        raise ValueError(
            "%s input mode does not use %s; non-default values would be "
            "ignored" % (inputs, ", ".join(changed)))


def build_config(size=DECLARED_SIZE, steps=DECLARED_STEPS, seed=DECLARED_SEED,
                 host_seed=DECLARED_HOST_SEED,
                 resolution=DECLARED_RESOLUTION,
                 edge_disposition=DECLARED_EDGE_DISPOSITION,
                 stage1_iters=DECLARED_STAGE1_ITERS,
                 stage2_iters=DECLARED_STAGE2_ITERS,
                 tolerance=DECLARED_TOLERANCE,
                 patience=DECLARED_PATIENCE,
                 candidate_moves=DECLARED_CANDIDATE_MOVES,
                 combinatorial_depth=_LEGACY_UNSET,
                 combinatorial_length=_LEGACY_UNSET,
                 readout=DECLARED_READOUT,
                 tori=DECLARED_TORI,
                 collar_twist=DECLARED_COLLAR_TWIST,
                 output_state=DECLARED_OUTPUT_STATE,
                 layers=DECLARED_COLLAR_LAYERS,
                 inputs=DECLARED_INPUTS, tau_a=DECLARED_TAU_A,
                 tau_b=DECLARED_TAU_B, grid=DECLARED_GRID,
                 coupling=DECLARED_COUPLING, time=DECLARED_TIME,
                 input_weight=DECLARED_INPUT_WEIGHT, regge=DECLARED_REGGE,
                 extend_boundary=DECLARED_EXTEND_BOUNDARY,
                 score_leak=DECLARED_SCORE_LEAK,
                 states=DECLARED_STATES,
                 operator=DECLARED_OPERATOR,
                 pin_boundary=DECLARED_PIN_BOUNDARY,
                 *, surgical_depth=_LEGACY_UNSET,
                 combinatorial_breadth=_LEGACY_UNSET):
    combinatorial_depth = _aliased_value(
        combinatorial_depth, surgical_depth, DECLARED_COMBINATORIAL_DEPTH,
        "combinatorial_depth", "surgical_depth")
    combinatorial_length = _aliased_value(
        combinatorial_length, combinatorial_breadth,
        DECLARED_COMBINATORIAL_LENGTH, "combinatorial_length",
        "combinatorial_breadth")

    if inputs not in InputMode.ALL:
        raise ValueError("unknown input mode %r: expected one of %s"
                         % (inputs, ", ".join(InputMode.ALL)))

    size = _integer_value("size", size)
    steps = _integer_value("steps", steps)
    seed = _integer_value("seed", seed)
    host_seed = _integer_value("host seed", host_seed)
    stage1_iters = _cpp_int_value("stage-one iterations", stage1_iters)
    stage2_iters = _cpp_int_value("stage-two iterations", stage2_iters)
    patience = _integer_value("patience", patience)
    candidate_moves = _cpp_int_value("candidate moves", candidate_moves)
    combinatorial_depth = _cpp_int_value(
        "combinatorial depth", combinatorial_depth)
    combinatorial_length = _cpp_int_value(
        "combinatorial length", combinatorial_length)
    grid = _cpp_int_value("grid", grid)
    layers = _cpp_int_value("collar layers", layers)
    tori = _integer_value("tori", tori)

    if size < 0:
        raise ValueError("size must be non-negative, got %r" % size)
    if steps < 0:
        raise ValueError("steps must be non-negative, got %r" % steps)
    for name, value in (("seed", seed), ("host seed", host_seed)):
        if not 0 <= value <= _UINT64_MAX:
            raise ValueError("%s must be an unsigned 64-bit integer between "
                             "0 and %d, got %r"
                             % (name, _UINT64_MAX, value))
    if size and host_seed > _UINT64_MAX - (4 * size - 1):
        raise ValueError(
            "host seed plus the %d refinement attempts must stay within the "
            "unsigned 64-bit range; got host seed %r and size %r"
            % (4 * size, host_seed, size))
    if stage1_iters < 1:
        raise ValueError("stage-one iterations must be at least 1, got %r"
                         % stage1_iters)
    if stage2_iters < 1:
        raise ValueError("stage-two iterations must be at least 1, got %r"
                         % stage2_iters)
    if combinatorial_depth < 1:
        raise ValueError("combinatorial depth must be at least 1, got %r"
                         % combinatorial_depth)
    if combinatorial_length < 0:
        raise ValueError(
            "combinatorial length is how many moves stage 1 composes into "
            "one candidate before it starts backing off, so it cannot be "
            "negative; zero keeps the deepening schedule. Got %r"
            % combinatorial_length)
    if (combinatorial_length > 0 and
            combinatorial_depth != DECLARED_COMBINATORIAL_DEPTH):
        raise ValueError(
            "--combinatorial-depth and --combinatorial-length select "
            "alternative search schedules; a non-default depth would be "
            "ignored when length is nonzero")
    if candidate_moves < 0:
        raise ValueError("candidate_moves is how many move specifications "
                         "stage 1 draws per unit, or 0 for every candidate "
                         "there is, so it may not be negative; got %r"
                         % candidate_moves)
    if patience < 1:
        raise ValueError("patience is a count of consecutive stalled units "
                         "and must be at least 1, got %r" % patience)

    resolution = _finite_value("resolution", resolution)
    tolerance = _finite_value("tolerance", tolerance)
    coupling = _finite_value("J", coupling)
    time = _finite_value("time", time)
    input_weight = _finite_value("input weight", input_weight)
    if resolution <= 0.0:
        raise ValueError("resolution must be a positive finite number, got %r"
                         % resolution)
    if tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite absolute "
                         "threshold, got %r" % tolerance)
    if input_weight <= 0.0:
        raise ValueError("input weight must be a positive finite number, "
                         "got %r" % input_weight)

    regge = _boolean_value("regge", regge)
    extend_boundary = _boolean_value("extend_boundary", extend_boundary)
    score_leak = _boolean_value("score_leak", score_leak)
    pin_boundary = _boolean_value("pin_boundary", pin_boundary)

    if edge_disposition not in EdgeDisposition.ALL:
        raise ValueError(
            "unknown edge disposition %r: expected one of %s"
            % (edge_disposition, ", ".join(EdgeDisposition.ALL)))
    if collar_twist not in ("none", "swap"):
        raise ValueError("unknown collar twist %r: expected none or swap"
                         % (collar_twist,))
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
    output_tau = (None if output_state is None
                  else _state_value("--output-state", output_state))

    if inputs == InputMode.NEUTRAL:
        _reject_unused_options(inputs, (
            ("--readout", names != [DECLARED_READOUT]),
            ("--tori", tori != DECLARED_TORI),
            ("--collar-twist", collar_twist != DECLARED_COLLAR_TWIST),
            ("--output-state", output_tau is not None),
            ("--layers", layers != DECLARED_COLLAR_LAYERS),
            ("--tau-a", tau_a != DECLARED_TAU_A),
            ("--tau-b", tau_b != DECLARED_TAU_B),
            ("--grid", grid != DECLARED_GRID),
            ("--J", coupling != DECLARED_COUPLING),
            ("--time", time != DECLARED_TIME),
            ("--input-weight", input_weight != DECLARED_INPUT_WEIGHT),
            ("--regge/--no-regge", regge != DECLARED_REGGE),
            ("--score-leak", score_leak != DECLARED_SCORE_LEAK),
            ("--state", bool(pairs)),
            ("--operator", operator != DECLARED_OPERATOR),
            ("--pin-boundary/--no-pin-boundary",
             pin_boundary != DECLARED_PIN_BOUNDARY),
            ("--extend-boundary",
             extend_boundary != DECLARED_EXTEND_BOUNDARY),
        ))
    else:
        _reject_unused_options(inputs, (
            ("--size", size != DECLARED_SIZE),
            ("--host-seed", host_seed != DECLARED_HOST_SEED),
            ("--resolution", resolution != DECLARED_RESOLUTION),
            ("--edge-disposition",
             edge_disposition != DECLARED_EDGE_DISPOSITION),
        ))

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
    return {
        "size": size,
        "steps": steps,
        "seed": seed,
        "host_seed": host_seed,
        "resolution": resolution,
        "edge_disposition": edge_disposition,
        "candidate_moves": candidate_moves,
        "stage1_iters": stage1_iters,
        "tolerance": tolerance,
        "patience": patience,
        "combinatorial_depth": combinatorial_depth,
        "combinatorial_length": combinatorial_length,
        "readout": ",".join(names),
        "tori": tori,
        "collar_twist": collar_twist,
        "output_state": (None if output_tau is None
                         else [output_tau.real, output_tau.imag]),
        "stage2_iters": stage2_iters,
        "register_degrees": list(DECLARED_REGISTER_DEGREES),
        "hodge_degrees": list(DECLARED_HODGE_DEGREES),
        "degrees": list(DECLARED_ANALYSIS_DEGREES),
        "betti_degrees": list(DECLARED_QUBIT_BETTI_DEGREES
                              if inputs == InputMode.QUBIT
                              else DECLARED_BETTI_DEGREES),
        # The qubit mode's parameters (spec D4). The moduli are kept as
        # [re, im] pairs so the config stays JSON as it is written.
        "inputs": inputs,
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
    }


def _complex_argument(text):
    """The CLI's complex parser: ``0.3+1.1j``, ``1.1j``, ``-0.2+0.8j``."""
    try:
        return _as_complex(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "%r is not a complex number; write it like 0.3+1.1j" % (text,))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Animate unforced emergence and the paper's certificates.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="drive emergence and render the overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=QUBIT_READOUT_NOTES)
    run.add_argument("--size", type=int, default=DECLARED_SIZE)
    run.add_argument("--steps", type=int, default=DECLARED_STEPS)
    run.add_argument("--seed", type=int, default=DECLARED_SEED)
    run.add_argument("--host-seed", type=int, default=DECLARED_HOST_SEED)
    run.add_argument("--resolution", type=float, default=DECLARED_RESOLUTION)
    run.add_argument("--edge-disposition", choices=list(EdgeDisposition.ALL),
                     default=DECLARED_EDGE_DISPOSITION,
                     help="causal character of the seed's edges: random "
                          "(default, magnitude one with the real/imaginary "
                          "split drawn per edge), spacelike (l^2 = +1), "
                          "timelike (l^2 = -1), lightlike (l^2 = i), or "
                          "foliated (a PRESCRIBED "
                          "light cone: timelike between hop layers of M0, "
                          "spacelike within one)")
    run.add_argument("--stage-one-iterations", type=int,
                     default=DECLARED_STAGE1_ITERS,
                     help="combinatorial updates stage 1 may commit per "
                          "engine unit (default %d)" % DECLARED_STAGE1_ITERS)
    run.add_argument("--stage-two-iterations", type=int,
                     default=DECLARED_STAGE2_ITERS,
                     help="relaxation iterations stage 2 runs per engine "
                          "unit (default %d)" % DECLARED_STAGE2_ITERS)
    run.add_argument("--combinatorial-depth", type=int,
                     default=_LEGACY_UNSET,
                     help="how many moves deep stage 1 searches for an "
                          "objective-lowering SEQUENCE: when single moves "
                          "find no improvement it tries 2-move sequences, "
                          "then 3, up to this many, committing a lowering "
                          "sequence whole. The depth covers EVERY move kind "
                          "in the draw -- the four Pachner moves and the "
                          "cone-outs and cone-ins alike -- not the surgical "
                          "moves alone (default %d, single moves)"
                          % DECLARED_COMBINATORIAL_DEPTH)
    run.add_argument("--surgical-depth", type=int,
                     dest="surgical_depth",
                     default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    run.add_argument("--combinatorial-length", type=int,
                     default=_LEGACY_UNSET,
                     help="how many moves stage 1 composes into ONE candidate "
                          "before it starts backing off. Non-zero runs the "
                          "search the other way round from "
                          "--combinatorial-depth (the two schedules are "
                          "mutually exclusive): "
                          "sequences of exactly this many moves are tried "
                          "FIRST, and the search shortens by one move -- to "
                          "this many minus one, then minus two, down to "
                          "single moves -- only when nothing at the current "
                          "length lowers the objective. It asks whether a "
                          "composition of this length improves a complex that "
                          "no shorter composition improves. Combined with "
                          "--candidate-moves 0 the search at every length is "
                          "exhaustive, which costs the move space raised to "
                          "the length. Default %d, the deepening schedule"
                          % DECLARED_COMBINATORIAL_LENGTH)
    run.add_argument("--combinatorial-breadth", type=int,
                     dest="combinatorial_breadth",
                     default=argparse.SUPPRESS, help=argparse.SUPPRESS)
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
    run.add_argument("--tolerance", type=float, default=DECLARED_TOLERANCE,
                     help="ABSOLUTE objective tolerance (default %g). Stage "
                          "2 backs its line search off until a trial lowers "
                          "the objective by at least this much, and the run "
                          "EXITS once a whole engine unit fails to improve "
                          "it by this much. Never relative"
                          % DECLARED_TOLERANCE)
    run.add_argument("--candidate-moves", type=int,
                     dest="candidate_moves",
                     default=DECLARED_CANDIDATE_MOVES,
                     help="how many move specifications stage 1 draws per "
                          "unit (default %d). The draw picks a KIND uniformly "
                          "from six and only then a site within it, so this "
                          "many draws is this many divided by six samples per "
                          "kind, against site sets of order the cell count. "
                          "Six draws covers about a fiftieth of the move space "
                          "and the run then reports itself combinatorially "
                          "stationary; measured, 200 draws commits moves where "
                          "6 and 50 commit none. A candidate costs 73-106ms "
                          "against 16-25 minutes for a relaxation unit. "
                          "ZERO means EVERY candidate rather than a sample of "
                          "them: the moves are then addressed by site instead "
                          "of drawn, which is both complete and reproducible "
                          "(about 739 candidates on a 3x3 collar, ~55s)"
                          % DECLARED_CANDIDATE_MOVES)
    run.add_argument("--patience", type=int, default=DECLARED_PATIENCE,
                     help="how many CONSECUTIVE units may fail to improve "
                          "the objective by --tolerance before the run stops "
                          "(default %d). Stage 1 draws its candidate moves "
                          "at random, so a unit that commits nothing is one "
                          "unlucky draw; raising this gives the run that "
                          "many more draws at the same geometry before it "
                          "calls the stall an ending. Any unit that does "
                          "improve resets the count"
                          % DECLARED_PATIENCE)
    run.add_argument("--inputs", choices=list(InputMode.ALL),
                     default=DECLARED_INPUTS,
                     help="what the node is built from: neutral (default, "
                          "the refined 4-ball of unforced emergence) or "
                          "qubit (two flat qubit tori on their collar, "
                          "synthesized against the two-body flip-flop "
                          "target; the qubit cobordism spec)")
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
    run.add_argument("--pin-boundary", action=argparse.BooleanOptionalAction,
                     dest="pin_boundary", default=DECLARED_PIN_BOUNDARY,
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
    run.add_argument("--live", action="store_true",
                     help="draw each frame as it is computed instead of only "
                          "at the end; still writes --out and --json. Needs "
                          "a local GUI matplotlib backend (WebAgg is refused "
                          "because its pause loop blocks), which the "
                          "'live' extra supplies (pip install -e \".[live]\"); "
                          "without one it fails by name rather than running "
                          "headless")
    run.add_argument("--out", default="emergence_animation.gif",
                     help="GIF or MP4 animation of every frame, or PNG of "
                          "the final frame")
    run.add_argument("--json", default=None,
                     help="also write the per-frame measurements here")
    run.add_argument("--geometry", default=None,
                     help="also write the FINAL complex here (schema 1: "
                          "cells, edges as [src, tgt, Re l^2, Im l^2], vertex "
                          "times, and the qubit blocks with their markings), "
                          "the one output a run can be rebuilt from. Written "
                          "however the drive ends, an interrupt included")
    run.add_argument("--quiet", action="store_true")
    return parser


def _validate_output_paths(out_path, json_path, geometry_path):
    """Refuse formats and aliases before an expensive drive starts."""
    if out_path:
        lowered = os.fspath(out_path).lower()
        if not lowered.endswith((".gif", ".mp4", ".png")):
            raise ValueError("--out must end in .gif, .mp4, or .png; got %r"
                             % os.fspath(out_path))
    paths = [(name, os.fspath(path))
             for name, path in (("--out", out_path), ("--json", json_path),
                                ("--geometry", geometry_path)) if path]
    seen = {}
    existing = []
    for name, path in paths:
        if os.path.isdir(path):
            raise ValueError("%s must name a file, not the directory %r"
                             % (name, path))
        canonical = os.path.normcase(os.path.realpath(
            os.path.abspath(os.path.expanduser(path))))
        if canonical in seen:
            raise ValueError("%s and %s resolve to the same output path %r"
                             % (seen[canonical], name, path))
        for previous_name, previous_path in existing:
            try:
                same_file = os.path.samefile(path, previous_path)
            except OSError:
                same_file = False
            if same_file:
                raise ValueError(
                    "%s and %s refer to the same output file %r"
                    % (previous_name, name, path))
        seen[canonical] = name
        if os.path.exists(path):
            existing.append((name, path))


def _write_json_document(path, document):
    """Serialize strictly before opening the destination for replacement."""
    payload = json.dumps(_json_safe(document), indent=2, sort_keys=True,
                         allow_nan=False)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(
            size=args.size, steps=args.steps, seed=args.seed,
            host_seed=args.host_seed, resolution=args.resolution,
            edge_disposition=args.edge_disposition,
            stage1_iters=args.stage_one_iterations,
            stage2_iters=args.stage_two_iterations,
            tolerance=args.tolerance, patience=args.patience,
            candidate_moves=args.candidate_moves,
            combinatorial_depth=args.combinatorial_depth,
            combinatorial_length=args.combinatorial_length,
            surgical_depth=getattr(args, "surgical_depth", _LEGACY_UNSET),
            combinatorial_breadth=getattr(
                args, "combinatorial_breadth", _LEGACY_UNSET),
            readout=args.readout, tori=args.tori,
            output_state=args.output_state, collar_twist=args.collar_twist,
            layers=args.layers, inputs=args.inputs, tau_a=args.tau_a,
            tau_b=args.tau_b, grid=args.grid, coupling=args.coupling,
            time=args.time, input_weight=args.input_weight, regge=args.regge,
            extend_boundary=args.extend_boundary, score_leak=args.score_leak,
            states=args.states, operator=args.operator,
            pin_boundary=args.pin_boundary)
        _validate_output_paths(args.out, args.json, args.geometry)
    except ValueError as error:
        parser.error(str(error))
    # Held from the moment the node exists, so the geometry is written even
    # when the drive is interrupted: an interrupted run's complex is exactly
    # the one worth keeping, and it is the only output that cannot be
    # recomputed from the others.
    driven = {}
    def remember_setup(node, inputs):
        driven.update(node=node, inputs=inputs)

    # Provenance describes the code that starts the run. Sampling after a long
    # drive can instead record a checkout or edit made while that run was in
    # progress.
    source = source_commit() if args.json or args.geometry else None
    try:
        result = (drive_live(config, progress=not args.quiet,
                             on_setup=remember_setup)
                  if args.live
                  else drive(config, progress=not args.quiet,
                             on_setup=remember_setup))
    except BaseException as error:
        if args.geometry and "node" in driven:
            try:
                _write_geometry(args.geometry, driven["node"],
                                driven.get("inputs"), args.quiet, source)
            except BaseException as geometry_error:
                # Recovery output is secondary: retain the engine or UI error
                # that ended the drive, but make the failed recovery visible.
                error.add_note("could not write recovery geometry %r: %s"
                               % (os.fspath(args.geometry), geometry_error))
        raise
    frames = result.frames
    # Geometry is the only irreplaceable output. Write it before derivative
    # records so a JSON serialization failure cannot suppress the snapshot.
    if args.geometry and "node" in driven:
        _write_geometry(args.geometry, driven["node"], result.inputs,
                        args.quiet, source)
    if not args.quiet and result.terminator == Terminator.TOLERANCE:
        sys.stdout.write(
            "exited on a STALL, not on a target: %d consecutive engine unit%s "
            "improved the objective by less than %g, after %d of %d units. "
            "The objective stopped moving at %s; it did not reach any "
            "particular value\n"
            % (result.stalls, "" if result.stalls == 1 else "s",
               config["tolerance"], frames[-1].step, config["steps"],
               _format_objective_total(frames[-1])))
    if args.json:
        document = {"config": config,
                    "source": source,
                    "terminator": result.terminator,
                    "stalls": result.stalls,
                    "frames": [f.to_json() for f in frames]}
        if result.inputs is not None:
            # The inputs once, next to the config: what the tori are, how
            # they sit in the host, chi and the algebra it comes from, and
            # the objective the node descended.
            document["inputs"] = result.inputs.to_json()
        _write_json_document(args.json, document)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % args.json)
    if args.out:
        path = render(frames, args.out)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % path)
    return 0


def _write_geometry(path, node, inputs, quiet, source=None):
    _write_json_document(path, geometry_document(node, inputs, source))
    if not quiet:
        sys.stdout.write("wrote %s\n" % path)


if __name__ == "__main__":
    sys.exit(main())
