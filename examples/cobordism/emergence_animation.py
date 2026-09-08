# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Animate unforced emergence and the certificates the whitepaper names.

ONE documented command drives the unmodified joint Regge-Hodge stationarity
objective on a neutral complex and, after every engine unit, draws what the
paper's certificates actually read off the accepted geometry.

What is driven, and what is only read
-------------------------------------
The dynamics is `MultiCobordism` in `SimulationMode.EMERGENCE` /
`EmergenceSubmode.STRICT` with `JointStationarityObjective`.
Nothing this file computes enters that objective: the firewall is structural
(a static `objectiveOf` over declared scalars), and every panel below is a
read-only measurement over the accepted geometry through the library's own
observable classes. No target is pinned, no register is forced, no residual
against a prescribed carrier is scored.

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
                                       drawing, coloured by the interval
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
COLOUR is a measurement -- the causal class of each edge's own interval
`Re(l^2)` -- and the panel says so on its face, because stabilizing the
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
``spacelike`` (``l^2 = +1``), ``timelike`` (``l^2 = -1``), or ``foliated`` (a
PRESCRIBED light cone -- timelike between hop layers of M0, spacelike within
one). Only ``foliated`` prescribes a causal order; it is labelled as such
wherever it is reported and is never presented as emergent.

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
objective at ``--input-weight``; nothing is pinned. Each block holds its
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
DECLARED_SURGICAL_DEPTH = 1
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
#: False, which is also the engine's default: this driver states its
#: boundary up front -- the two input tori of the qubit mode, the held M0 of
#: the neutral one -- and a cone that hands dW a face it did not have
#: changes what the cobordism IS rather than how it is shaped. A cone-in
#: buries the facet it stands on and exposes the new cell's others; a
#: cone-out exposes every facet of the cell it removes. Declared here
#: anyway, so the run document records which it was.
DECLARED_EXTEND_BOUNDARY = False
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
#: The two tori's labels and drawing colours (the layout highlight and the
#: traces read from the same pair, so a colour and its label cannot disagree).
DECLARED_TORUS_LABELS = ("A", "B")
DECLARED_TORUS_COLOURS = ("#d2691e", "#1f8a70")


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

    #: Every value a drive may report.
    ALL = (STEPS, TOLERANCE)


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
#: Whether the input tori are HELD while the bulk relaxes (`--pin-boundary`).
#: The block residual of spec D2 is a function of tau_hat alone, and tau_hat
#: is a conformal invariant, so reshaping a boundary within its conformal
#: class costs the objective nothing and the relaxation does it: measured,
#: individual boundary edges move 10-28% while tau_hat holds to thirteen
#: digits. A bulk fitted to that reshaped boundary does not work with the
#: input it was given. Pinning removes the freedom rather than pricing it.
#: Off by default because the spec's Do-not list still says pinned regions
#: are not used in this experiment.
DECLARED_PIN_BOUNDARY = False


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
                 "markings", "algebra", "weight", "regge", "layers",
                 "objective_name", "objective_terms", "torus_warnings", "seed")

    def __init__(self, tori, tau_in, vertex_ids, cells, markings,
                 algebra, weight, regge, layers, objective_name,
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
        self.layers = int(layers)
        self.objective_name = str(objective_name)
        self.objective_terms = [str(name) for name in objective_terms]
        self.torus_warnings = [[str(w) for w in notes] for notes in torus_warnings]
        self.seed = dict(seed)

    @property
    def labels(self):
        return DECLARED_TORUS_LABELS[:len(self.tori)]

    def highlight(self):
        """The tori's edges for the layout panel: (label, colour, id pairs)."""
        return [(label, colour, {tuple(cell) for cell in cells})
                for label, colour, cells in zip(self.labels,
                                                DECLARED_TORUS_COLOURS,
                                                self.cells)]

    def to_json(self):
        import numpy as np

        def matrix(value):
            return [[complex(z) for z in row] for row in np.asarray(value)]

        algebra = {key: (str(value) if isinstance(value, str)
                         else float(value) if key in ("coupling", "time", "Jt")
                         else matrix(value) if key in ("chi", "product_state",
                                                       "first_order_amplitudes",
                                                       "exact_amplitudes")
                         else [complex(z) for z in np.asarray(value)])
                   for key, value in self.algebra.items()}
        return _json_safe({
            "labels": list(self.labels),
            "tau_in": self.tau_in,
            "coefficients_in": self.coefficients_in,
            "bloch_in": [[float(x) for x in torus.bloch()] for torus in self.tori],
            "grid": [len(torus.vertices()) for torus in self.tori],
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


def build_qubit_node(config):
    """The qubit node: two flat tori on their collar, seeded as the input
    blocks with their state fibers, markings and the two-body target.

    Step by step the setup the T1-T3 tests measured under (spec S1-S3, S5),
    with D2/D3 as revised: the tori `SimplicialQubit.flat_torus(tau, n, n)`;
    the collar `MultiCobordism.seed_collar`, one gated whole refused by name;
    the node with the degree-1 register on the complex locus (the tori carry
    complex lengths and pure-gauge phases), the Regge term per `regge` and
    the Whitney pencil as its metric source; each torus's vertex set one
    input block (`seed_inputs`); each torus's holomorphic form attached as a
    degree-1 fiber on its edges with the harmonic contour
    (`attach_input_fiber`); each torus's marking with its input coefficients
    (1, tau_in) on its block (`set_input_marking`), from which the engine
    derives the block's live frame at every read; chi of spec S5 as the
    Choi-decomposed two-body target (`set_two_body_target`), 2 x 2 in the
    derived frames; the block residuals -- the leak of (1, tau_in) through
    the live frame in the whole's zero mode on the block's edges -- in r_U at
    the input weight. No region is pinned and no objective is injected: the
    node's default objective is what T2 and T3 measured under, and
    `QubitInputs.objective_name` records it.

    Returns the node and its `QubitInputs`.
    """
    import numpy as np

    tau_in = [complex(*config["tau_a"]), complex(*config["tau_b"])]
    grid = int(config["grid"])
    with warnings.catch_warnings():
        # The tori's construction notes are recorded from `warnings()` in
        # `QubitInputs`, not printed.
        warnings.simplefilter("ignore")
        tori = [obs.SimplicialQubit.flat_torus(tau, grid, grid)
                for tau in tau_in]
    seed = MC.seed_collar(tori[0].spacetime(), tori[1].spacetime(),
                          config["layers"])
    ids = [{int(k): int(v) for k, v in mapping.items()}
           for mapping in seed.vertex_ids]
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [],
              degrees=list(config["register_degrees"]), seed=config["seed"],
              einstein_hilbert=bool(config["regge"]),
              real_squared_lengths_only=False,
              metric_source=cob.HodgeMetricSource.WhitneyPencil)
    node.seed_inputs([sorted(mapping.values()) for mapping in ids])
    node.use_fiber_residuals(True)
    node.set_input_residual_weight(config["input_weight"])
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
        algebra = flip_flop_evolution(np.asarray(tori[0].state()),
                                      np.asarray(tori[1].state()),
                                      config["coupling"], config["time"])
    else:
        algebra = gate_image(operator, np.asarray(tori[0].state()),
                             np.asarray(tori[1].state()))
    algebra["operator"] = operator
    node.set_two_body_target(algebra["chi"], True)
    for index, torus in enumerate(tori):
        _assert_seed_reads_the_torus(node, index, torus)
    host = node.spacetime()
    inputs = QubitInputs(
        tori, tau_in, ids, cells, markings, algebra,
        config["input_weight"], config["regge"], config["layers"],
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
        return [value.real, value.imag]
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
        self.two_body = self._read_two_body(node)
        self.boundary = self._read_boundary(spacetime, node)
        self.completion = self._read_completion(node)

    # ---- 1. the objective -------------------------------------------

    @staticmethod
    def _read_objective(node):
        terms = node.objective_terms()
        block = {"total": _finite(node.objective())}
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
        if not self.components:
            return Absent("no cluster to carry a band")
        settings = obs.SpectralFiberConfig()
        settings.degrees = list(config["degrees"])
        tracker = obs.SpectralFiberTracker(spacetime, settings)
        rows = []
        for component in self.components:
            for degree in config["degrees"]:
                try:
                    read = tracker.enumerateBands(component.support, degree)
                except Exception as error:                # noqa: BLE001
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
            if index < len(self.components):
                evidence.component = self.components[index].id
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
                except Exception:                         # noqa: BLE001
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
        """The incoming boundary's vertices, or an empty list if closed.

        Delegates to the module-level rule the host is built against, so the
        host and the readout cannot disagree about what M0 is. A closed
        complex has none, and then there is no M0 -- which the paper's
        readouts require, so the channel refuses rather than inventing one.
        """
        try:
            return boundary_vertices(spacetime)
        except Exception:                                 # noqa: BLE001
            return []

    def _read_crossings(self, spacetime):
        accepted = [f for f in self.candidates if f is not None]
        if not accepted:
            return Absent("no accepted band: no world tube to cross a level")
        boundary = self._m0_vertices(spacetime)
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
        for index, fiber in enumerate(accepted):
            tube = obs.WorldTubeInput()
            tube.tubeId = "band-%d" % index
            tube.band = fiber
            tube.orientation = +1
            tube.certifiedQuarkTube = False
            tubes.append(tube)
        levels = [float(x) for x in temporal.layer] if temporal.layer else []
        level = max(levels) / 2.0 if levels else 0.0
        block = {"level": level, "tubes": len(tubes)}
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
        if not block.get("quarkTubes"):
            block["baryonNote"] = ("no tube carries a quark certificate, so "
                                   "the one-third sum has no term")
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
            except Exception:                             # noqa: BLE001
                continue
            signs.append({
                "tubeId": str(crossing.tubeId),
                "sign": int(crossing.sign),
                "admissible": bool(crossing.admissible),
                "perpendicular": complex(crossing.perpendicular),
                "reasons": _reasons(crossing),
            })
        block["crossings"] = signs
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
        try:
            read = obs.ParticleClusters().classifyBaryon(evidence)
        except Exception as error:                        # noqa: BLE001
            return Absent("classifier refused: %s" % error)
        return {"classification": str(read.classification),
                "confidence": _finite(read.confidence),
                "reasons": _reasons(read)}

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
        """`MultiCobordism.monodromy` with both markings: the integer matrix
        relating them through the whole's zero mode (spec S6), with the Betti
        numbers, the harmonic rank and the rounding and fit residuals. An
        obstructed read names its obstruction and is Absent."""
        import numpy as np

        try:
            read = MC.monodromy(spacetime, inputs.markings[0],
                                inputs.markings[1])
        except Exception as error:                        # noqa: BLE001
            return Absent("monodromy read refused: %s" % error)
        if read.obstruction:
            return Absent("monodromy read obstructed: %s (Betti %s, harmonic "
                          "rank %d)" % (read.obstruction, list(read.betti),
                                        read.harmonic_rank))
        matrix = np.asarray(read.monodromy)
        return {"betti": [int(b) for b in read.betti],
                "harmonic_rank": int(read.harmonic_rank),
                "monodromy": [[complex(z) for z in row] for row in matrix],
                "rounded": [[int(x) for x in row] for row in read.rounded],
                "rounding_residual": _finite(read.rounding_residual),
                "fit_residual": _finite(read.fit_residual)}

    # ---- 15. the two-body read ----------------------------------------

    @staticmethod
    def _read_two_body(node):
        """`read_two_body`: the transfer in the two derived period frames
        (2 x 2 for two qubits), its projective leak against chi, the Schmidt
        spectrum and rank of the Choi state, the reversal residual, and the
        blocks' own-kernel fiber leaks as the engine reports them (a
        diagnostic; the scored block residuals are in the `blocks`
        channel)."""
        import numpy as np

        try:
            read = node.read_two_body()
        except Exception as error:                        # noqa: BLE001
            return Absent("two-body read refused: %s" % error)
        transfer = np.asarray(read.transfer)
        return {"in_frames": bool(read.in_frames),
                "derived_frames": bool(read.derived_frames),
                "choi_decomposed": bool(read.choi_decomposed),
                "transfer": [[complex(z) for z in row] for row in transfer],
                "shape": [int(n) for n in transfer.shape],
                "residual": _finite(read.residual),
                "singular_values": [_finite(s) for s in read.singular_values],
                "schmidt_rank": int(read.schmidt_rank),
                "reversal_residual": _finite(read.reversal_residual),
                "input_fiber_residuals": [_finite(r) for r in
                                          read.input_fiber_residuals]}

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
        boundary of W is exactly the two tori (true by construction on the
        collar; a cone-out dent reopens it)."""
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

def drive(config, progress=False, on_frame=None, on_node=None):
    """Drive unforced emergence, reading a frame after every engine unit.

    `on_frame(frames, index)` is called as each unit completes, so a caller
    can display a run while it is still running. `on_node(node)` is called
    once, as soon as the node exists, so a caller holds the object the loop
    drives and can write its geometry even if the loop is interrupted. Both
    are observers: neither is consulted, so a drive with them and a drive
    without them take the same steps. It is the ONLY difference
    between a live drive and a headless one: the loop, the engine calls and
    the frames are the same either way, so a live view cannot diverge from
    the run it claims to be showing.

    The node is built by a FACTORY selected by `config["inputs"]`: the
    neutral host with its unpinned node (the default, spelled here because
    it is the existing drive) or one of `NODE_FACTORIES` -- the qubit mode's
    two tori on their collar. The factory returns the node and what the
    frame reads need to know about its inputs (None for the neutral host);
    the loop below is the same for every factory, so the modes cannot
    diverge in how they are driven, only in what they drive.

    Returns a `DriveResult` carrying the frames, the terminator and the
    inputs.
    """
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

    factory = NODE_FACTORIES.get(config.get("inputs", DECLARED_INPUTS),
                                 neutral_node)
    node, inputs = factory(config)
    if on_node is not None:
        on_node(node)
    # One place, both modes: the boundary gate is a property of the drive, not
    # of what is being driven, so neither factory decides it (spec R8: the
    # host is emergent, and this says which growth counts as emergence).
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
        before = _objective_total(frames[-1])
        # Stage 1 before stage 2 within a unit (spec S4): a committed
        # combinatorial move rebuilds the complex from a snapshot of its
        # cells and every edge's length and phase, and stage 2 then relaxes
        # every edge of the rebuilt complex.
        list(node.run_stage1(max_steps=config["stage1_iters"],
                             n_candidate_moves=config["candidate_moves"],
                             max_lookahead=config["surgical_depth"]))
        list(node.run_stage2(max_iters=config["stage2_iters"],
                             tolerance=config["tolerance"]))
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
        value = frame.blocks[index].get("residual")
        return "absent" if not isinstance(value, float) else "%.2e" % value

    def tau_hat(index):
        if isinstance(frame.blocks, Absent):
            return "absent"
        read = frame.blocks[index].get("read")
        return "absent" if isinstance(read, Absent) else _tau_text(read["tau"])

    def leak(index):
        if isinstance(frame.leaks, Absent):
            return "absent"
        row = frame.leaks["per_block"][index]
        return "absent" if isinstance(row, Absent) else "%.2e" % row["leak"]

    two_body = ("absent" if isinstance(frame.two_body, Absent)
                else "%.4f" % frame.two_body["residual"])
    betti = ("absent" if isinstance(frame.betti, Absent)
             else [frame.betti["numbers"][d]
                   for d in sorted(frame.betti["numbers"])])
    monodromy = ("absent" if isinstance(frame.monodromy, Absent)
                 else frame.monodromy["rounded"])
    total = frame.objective.get("total")
    sys.stdout.write(
        "[step %2d] objective %s | blocks %s %s | two-body %s | leaks %s %s "
        "| tau %s %s | betti %s | monodromy %s\n"
        % (frame.step, "n/a" if total is None else "%.6g" % total,
           block_residual(0), block_residual(1), two_body, leak(0), leak(1),
           tau_hat(0), tau_hat(1), betti, monodromy))
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
    title = "complex -- position: drawing only | colour: interval Re(l^2)"
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
#: the colour alone. Every label names the INTERVAL, because that is what is
#: being drawn -- `degenerate` is the one that needs saying, since an absent
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
    gaps = [r["lowerGap"] or 0.0 for r in accepted]
    axis.bar(range(len(ranks)), ranks, color="#608c64", width=0.5,
             label="rank")
    twin = axis.twinx()
    twin.plot(range(len(gaps)), gaps, marker="s", markersize=3,
              linewidth=0.8, color="#bc8836", label="lower gap")
    twin.tick_params(labelsize=6)
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
    shown = [v if v is not None else 0.0 for v in values]
    axis.barh(range(len(shown)), shown, color="#bc8836", height=0.6)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=6)
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
    if isinstance(mass, Absent):
        return _absent_panel(axis, title, mass.reason)
    if mass is None and baryon is None:
        return _absent_panel(axis, title,
                             "no admissible crossing contributes a term")
    axis.set_title(title, fontsize=8)
    note = frame.crossings.get("baryonNote", "")
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
    degrees = sorted(numbers)
    values = [numbers[d] if numbers[d] is not None else 0 for d in degrees]
    axis.set_title(title, fontsize=8)
    axis.bar([str(d) for d in degrees], values, color="#5a7ca0", width=0.6)
    axis.set_xlabel("degree", fontsize=6)
    axis.tick_params(labelsize=6)
    axis.text(0.5, 0.92, "not a quark count", transform=axis.transAxes,
              ha="center", fontsize=5.5, color="#666666", style="italic")


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
    for index, (label, colour) in enumerate(zip(labels, DECLARED_TORUS_COLOURS)):
        series.append(("block %s (weight %g)" % (label, last.inputs.weight),
                       colour, "-", lambda f, i=index: _block_value(f, i, "residual")))
        series.append((r"leak %s in the whole's $\ker L_1$" % label, colour, ":",
                       lambda f, i=index: _leak_value(f, i)))
    series.append((r"two-body vs $\chi$", "#1f4e79", "--",
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
                                                DECLARED_TORUS_COLOURS)):
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
                                                DECLARED_TORUS_COLOURS)):
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
    """|T| and |chi| side by side, each scaled by its own maximum: T is the
    whole's pencil-operator block between the two period frames (a metric
    quantity, one inverse power of the length scale), chi the algebra's
    target; the projective leak the engine scores, the Schmidt spectrum and
    the reversal residual are in the title."""
    title = r"$|T_{AB}|$ (period frames) vs $|\chi|$ (spec S5)"
    if frame.inputs is None:
        return _absent_panel(axis, title, _qubit_reason(frame))
    if isinstance(frame.two_body, Absent):
        return _absent_panel(axis, title, frame.two_body.reason)
    import numpy as np

    transfer = np.abs(np.asarray(frame.two_body["transfer"], dtype=complex))
    chi = np.abs(np.asarray(frame.inputs.algebra["chi"], dtype=complex))
    if transfer.shape != chi.shape:
        return _absent_panel(axis, title, "the transfer is %s but chi is %s: "
                                          "not read in the period frames"
                             % (transfer.shape, chi.shape))
    rows, columns = chi.shape
    grid = np.full((rows, 2 * columns + 1), np.nan)
    grid[:, :columns] = transfer / max(float(transfer.max()), 1e-300)
    grid[:, columns + 1:] = chi / max(float(chi.max()), 1e-300)
    axis.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    def ink(value):
        # Legible on both ends of the colour map.
        return "#ffffff" if value > 0.6 else "#333333"

    for i in range(rows):
        for j in range(columns):
            axis.text(j, i, "%.3g" % transfer[i, j], ha="center", va="center",
                      fontsize=5.5, color=ink(grid[i, j]))
            axis.text(columns + 1 + j, i, "%.3g" % chi[i, j], ha="center",
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
              r"$|\chi| / \max|\chi|$",
              ha="center", fontsize=6, color="#333333")
    read = frame.two_body
    spectrum = ", ".join("%.3g" % s for s in read["singular_values"]
                         if s is not None)
    axis.set_title("%s\nleak %.4g, Schmidt $\\sigma$ (%s) rank %d, reversal %.1e%s"
                   % (title, read["residual"] if read["residual"] is not None
                      else float("nan"), spectrum, read["schmidt_rank"],
                      read["reversal_residual"]
                      if read["reversal_residual"] is not None
                      else float("nan"),
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
        lines.append("boundary of W is the two tori: %s (%d uncovered face(s))"
                     % ("yes" if frame.completion["bridge_phase_complete"]
                        else "no", frame.completion["uncovered_faces"]))
    if isinstance(frame.leaks, Absent):
        lines.append("zero mode of the whole: " + frame.leaks.reason)
    else:
        lines.append("harmonic rank of the whole: %s"
                     % frame.leaks["harmonic_rank"])
    if isinstance(frame.monodromy, Absent):
        lines.append("monodromy: " + frame.monodromy.reason)
    else:
        m = frame.monodromy
        lines.append("monodromy rounded %s (rounding %s, fit %s)"
                     % (m["rounded"],
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
        return ("qubit cobordism -- engine unit %d of %d -- tau_A %s, tau_B "
                "%s, %dx%d tori, J t = %g, input weight %g, Regge term %s, "
                "objective %s (read-outs post-hoc, tau read-out only)"
                % (frame.step, last_step, _tau_text(inputs.tau_in[0]),
                   _tau_text(inputs.tau_in[1]), frame.config["grid"],
                   frame.config["grid"], inputs.algebra["Jt"], inputs.weight,
                   "on" if inputs.regge else "off", inputs.objective_name))
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


def drive_live(config, progress=False, on_node=None):
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
    if backend.lower() not in _interactive_backends():
        raise RuntimeError(
            "--live needs an interactive matplotlib backend; this process "
            "has %r, which renders to files and shows no window. A run "
            "under it would compute every frame and display none of them. "
            "Set MPLBACKEND to an interactive backend (webagg needs no "
            "display), or drop --live and read the rendered --out: the "
            "drive is identical either way." % backend)
    if not plt.isinteractive():
        plt.ion()
    figure = plt.figure(figsize=(18, 10))

    ready = queue.Queue()
    published = {}
    outcome = {}

    def publish(frames, index):
        published["frames"] = frames
        ready.put(index)

    def worker():
        try:
            outcome["result"] = drive(config, progress=progress,
                                      on_frame=publish, on_node=on_node)
        except BaseException as exc:            # re-raised on the main thread
            outcome["error"] = exc
        finally:
            ready.put(None)

    thread = threading.Thread(target=worker, name="emergence-drive",
                              daemon=True)
    thread.start()
    # The live path stabilizes through the SAME chained step a headless render
    # walks, advanced once per unit as it completes. Frames are published in
    # index order, so the state sees exactly the sequence `stabilize` would
    # feed it and the two produce identical placements. Accumulated here
    # rather than precomputed because there is no finished run to walk yet.
    state = StableLayout()
    placed = []
    while True:
        index = ready.get()
        if index is None:
            break
        frames = published["frames"]
        while len(placed) <= index:
            placed.append(place_frame(state, frames[len(placed)]))
        draw_frame(figure, frames, index, placed)
        figure.canvas.draw_idle()
        # Yields to the GUI event loop; a backend without one still returns.
        plt.pause(0.001)
    thread.join()
    plt.close(figure)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def geometry_document(node, inputs=None):
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
        source = int(edge.getSource().getId())
        target = int(edge.getTarget().getId())
        squared = complex(edge.getLength()) ** 2
        edges.append([source, target, squared.real, squared.imag])
        phase = complex(edge.getPhase())
        if phase != 0:
            phases.append([source, target, phase.real, phase.imag])
    document = {
        "schema": 1,
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(20, 12))
    # Computed once, in frame order: the alignment is a chain, so a renderer
    # that redraws a frame or draws only the last must still see the same
    # positions it would have seen drawing them all in sequence.
    placed = stabilize(frames)
    lowered = path.lower()
    if lowered.endswith(".png") or len(frames) == 1:
        draw_frame(figure, frames, len(frames) - 1, placed)
        figure.savefig(path, dpi=110)
        plt.close(figure)
        return path
    import matplotlib.animation as animation

    def update(index):
        draw_frame(figure, frames, index, placed)
        return []

    movie = animation.FuncAnimation(figure, update, frames=len(frames),
                                    interval=900, blit=False, repeat=False)
    writer = "pillow" if lowered.endswith(".gif") else "ffmpeg"
    movie.save(path, writer=writer, dpi=100)
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


def build_config(size=DECLARED_SIZE, steps=DECLARED_STEPS, seed=DECLARED_SEED,
                 host_seed=DECLARED_HOST_SEED,
                 resolution=DECLARED_RESOLUTION,
                 edge_disposition=DECLARED_EDGE_DISPOSITION,
                 stage1_iters=DECLARED_STAGE1_ITERS,
                 stage2_iters=DECLARED_STAGE2_ITERS,
                 tolerance=DECLARED_TOLERANCE,
                 patience=DECLARED_PATIENCE,
                 surgical_depth=DECLARED_SURGICAL_DEPTH,
                 inputs=DECLARED_INPUTS, tau_a=DECLARED_TAU_A,
                 tau_b=DECLARED_TAU_B, grid=DECLARED_GRID,
                 coupling=DECLARED_COUPLING, time=DECLARED_TIME,
                 input_weight=DECLARED_INPUT_WEIGHT, regge=DECLARED_REGGE,
                 extend_boundary=DECLARED_EXTEND_BOUNDARY,
                 operator=DECLARED_OPERATOR,
                 pin_boundary=DECLARED_PIN_BOUNDARY):
    if edge_disposition not in EdgeDisposition.ALL:
        raise ValueError(
            "unknown edge disposition %r: expected one of %s"
            % (edge_disposition, ", ".join(EdgeDisposition.ALL)))
    if inputs not in InputMode.ALL:
        raise ValueError("unknown input mode %r: expected one of %s"
                         % (inputs, ", ".join(InputMode.ALL)))
    # Refused rather than clamped: a caller who writes 0 means something the
    # drive cannot do (never stop on a stall), and silently reading it as 1
    # would run the opposite of what was asked.
    if int(patience) < 1:
        raise ValueError("patience is a count of consecutive stalled units "
                         "and must be at least 1, got %r" % (patience,))
    moduli = {}
    for label, value in (("tau_a", tau_a), ("tau_b", tau_b)):
        tau = _as_complex(value)
        if not (math.isfinite(tau.real) and math.isfinite(tau.imag)
                and tau.imag > 0.0):
            raise ValueError(
                "%s must lie in the upper half plane (Im > 0), got %r: the "
                "poles |0> and |1> are limits reached by pinching, not "
                "inputs (spec S1)" % (label, tau))
        moduli[label] = [tau.real, tau.imag]
    if grid < 3:
        raise ValueError("grid must be at least 3 (below 3 the torus grid is "
                         "not a simplicial complex), got %r" % (grid,))
    if not (input_weight > 0.0 and math.isfinite(input_weight)):
        raise ValueError("input weight must be a positive finite number, "
                         "got %r" % (input_weight,))
    for label, value in (("J", coupling), ("time", time)):
        if not math.isfinite(value):
            raise ValueError("%s must be finite, got %r" % (label, value))
    if stage1_iters < 1:
        raise ValueError("stage-one iterations must be at least 1, got %r"
                         % (stage1_iters,))
    if stage2_iters < 1:
        raise ValueError("stage-two iterations must be at least 1, got %r"
                         % (stage2_iters,))
    if surgical_depth < 1:
        raise ValueError("surgical depth must be at least 1, got %r"
                         % (surgical_depth,))
    if not (tolerance > 0.0 and math.isfinite(tolerance)):
        raise ValueError("tolerance must be a positive finite absolute "
                         "threshold, got %r" % (tolerance,))
    return {
        "size": size,
        "steps": steps,
        "seed": seed,
        "host_seed": host_seed,
        "resolution": resolution,
        "edge_disposition": edge_disposition,
        "candidate_moves": DECLARED_CANDIDATE_MOVES,
        "stage1_iters": stage1_iters,
        "tolerance": tolerance,
        "patience": patience,
        "surgical_depth": surgical_depth,
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
        "grid": int(grid),
        "layers": DECLARED_COLLAR_LAYERS,
        "coupling": float(coupling),
        "time": float(time),
        "input_weight": float(input_weight),
        "regge": bool(regge),
        "extend_boundary": bool(extend_boundary),
        "operator": str(operator),
        "pin_boundary": bool(pin_boundary),
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
    run = sub.add_parser("run", help="drive emergence and render the overlay")
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
                          "timelike (l^2 = -1), or foliated (a PRESCRIBED "
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
    run.add_argument("--surgical-depth", type=int,
                     default=DECLARED_SURGICAL_DEPTH,
                     help="how many moves deep stage 1 searches for an "
                          "objective-lowering SEQUENCE: when single moves "
                          "find no improvement it tries 2-move sequences, "
                          "then 3, up to this many, committing a lowering "
                          "sequence whole. The depth covers EVERY move kind "
                          "in the draw -- the four Pachner moves and the "
                          "cone-outs and cone-ins alike -- not the surgical "
                          "moves alone (default %d, single moves)"
                          % DECLARED_SURGICAL_DEPTH)
    run.add_argument("--tolerance", type=float, default=DECLARED_TOLERANCE,
                     help="ABSOLUTE objective tolerance (default %g). Stage "
                          "2 backs its line search off until a trial lowers "
                          "the objective by at least this much, and the run "
                          "EXITS once a whole engine unit fails to improve "
                          "it by this much. Never relative"
                          % DECLARED_TOLERANCE)
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
                     help="qubit mode: weight of each block's residual (the "
                          "whole's zero mode against the torus's input "
                          "coefficients in its live frame) in r_U (default %g)"
                          % DECLARED_INPUT_WEIGHT)
    run.add_argument("--regge", action=argparse.BooleanOptionalAction,
                     default=DECLARED_REGGE,
                     help="qubit mode: keep the Regge stationarity term in "
                          "the objective (--no-regge leaves r_U alone; "
                          "default %s)" % ("on" if DECLARED_REGGE else "off"))
    run.add_argument("--operator", default=DECLARED_OPERATOR,
                     choices=["flip_flop"] + sorted(DECLARED_GATES),
                     help="qubit mode: the operator whose output the transfer "
                          "is fitted to. flip_flop is the interaction "
                          "Hamiltonian of spec S5 at --J and --time; the rest "
                          "are unitary two-qubit gates, fitted through their "
                          "image of the product state (default %s)"
                          % DECLARED_OPERATOR)
    run.add_argument("--pin-boundary", action="store_true",
                     dest="pin_boundary", default=DECLARED_PIN_BOUNDARY,
                     help="qubit mode: hold the input tori fixed while the "
                          "bulk relaxes. The block residual sees only tau_hat, "
                          "so without this the boundary is reshaped freely "
                          "within its conformal class and the bulk is fitted "
                          "to a boundary that is not the input it was given")
    run.add_argument("--extend-boundary", action="store_true",
                     dest="extend_boundary",
                     default=DECLARED_EXTEND_BOUNDARY,
                     help="let a cone move extend the boundary of W. Off by "
                          "default: a cone that hands dW a face it did not "
                          "have changes what the cobordism is, and this "
                          "driver states its boundary up front")
    run.add_argument("--live", action="store_true",
                     help="draw each frame as it is computed instead of only "
                          "at the end; still writes --out and --json. Needs "
                          "an interactive matplotlib backend and fails by "
                          "name without one")
    run.add_argument("--out", default="emergence_animation.gif",
                     help="GIF, MP4, or PNG of the final frame")
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = build_config(args.size, args.steps, args.seed, args.host_seed,
                          args.resolution, args.edge_disposition,
                          args.stage_one_iterations,
                          args.stage_two_iterations,
                          tolerance=args.tolerance,
                          patience=args.patience,
                          surgical_depth=args.surgical_depth,
                          inputs=args.inputs, tau_a=args.tau_a,
                          tau_b=args.tau_b, grid=args.grid,
                          coupling=args.coupling, time=args.time,
                          input_weight=args.input_weight, regge=args.regge,
                          extend_boundary=args.extend_boundary,
                          operator=args.operator,
                          pin_boundary=args.pin_boundary)
    # Held from the moment the node exists, so the geometry is written even
    # when the drive is interrupted: an interrupted run's complex is exactly
    # the one worth keeping, and it is the only output that cannot be
    # recomputed from the others.
    driven = {}
    try:
        result = (drive_live(config, progress=not args.quiet,
                             on_node=lambda node: driven.update(node=node))
                  if args.live
                  else drive(config, progress=not args.quiet,
                             on_node=lambda node: driven.update(node=node)))
    except KeyboardInterrupt:
        if args.geometry and "node" in driven:
            _write_geometry(args.geometry, driven["node"], None, args.quiet)
        raise
    frames = result.frames
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
                    "terminator": result.terminator,
                    "stalls": result.stalls,
                    "frames": [f.to_json() for f in frames]}
        if result.inputs is not None:
            # The inputs once, next to the config: what the tori are, how
            # they sit in the host, chi and the algebra it comes from, and
            # the objective the node descended.
            document["inputs"] = result.inputs.to_json()
        with open(args.json, "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % args.json)
    if args.geometry and "node" in driven:
        _write_geometry(args.geometry, driven["node"], result.inputs, args.quiet)
    if args.out:
        path = render(frames, args.out)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % path)
    return 0


def _write_geometry(path, node, inputs, quiet):
    with open(path, "w") as handle:
        json.dump(geometry_document(node, inputs), handle, indent=2,
                  sort_keys=True)
    if not quiet:
        sys.stdout.write("wrote %s\n" % path)


if __name__ == "__main__":
    sys.exit(main())
