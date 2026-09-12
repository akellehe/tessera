# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Animate unforced emergence and the certificates the whitepaper names.

ONE documented command drives the unmodified joint Regge-Hodge stationarity
objective on a neutral complex and, after every engine unit, draws what the
paper's certificates actually read off the accepted geometry.

What is driven, and what is only read
-------------------------------------
The dynamics is `MultiCobordism` in
`SimulationMode.EMERGENCE` / `EmergenceSubmode.STRICT` with
`JointStationarityObjective`. Nothing the neutral certificate panels compute
enters that objective: the firewall is structural (a static `objectiveOf` over
declared scalars), and every panel is a read-only measurement over the accepted
geometry through the library's own observable classes. No target is pinned, no
register is forced, and no residual against a prescribed carrier is scored.
The seed's declared boundary region is held fixed; that removes coordinates
from relaxation without adding a pinned objective. The qubit experiment is
implemented separately by ``qubit_animation.py``.

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


#: How often the `--live` main thread services the GUI event loop while the
#: worker computes, in seconds -- about twenty times a second.
#:
#: The main thread has nothing to do between finished units except keep the
#: window alive, and an engine unit can take minutes. Waiting in the event loop
#: at this interval rather than in a blocking queue read is what keeps the
#: window responsive; the cost is one wakeup per interval, which is nothing
#: beside a unit.
LIVE_POLL_INTERVAL = 0.05
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

    `inputs` is any experiment-specific description returned by the node
    factory, or None when there is none. It is recorded once in the run
    document rather than repeated on every frame.

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

class AnimationFrame:
    """Every measurement drawn on one frame, assembled over one geometry.

    Read-only: nothing here touches the geometry or the objective. Each
    channel is either a measurement or an `Absent` carrying the named reason,
    so a panel can always say what it is showing.
    """

    CERTIFICATE_CHANNELS = ("clusters", "bands", "anchors", "transports",
                            "statistics", "crossings", "spin", "verdict")

    def __init__(self, node, spacetime, step, config, inputs=None):
        self.step = step
        self.config = config
        self.spacetime = spacetime
        self.inputs = None
        self.objective = self._read_objective(node)
        self.layout = self._read_layout(spacetime)
        # Drawing-only, like `layout`: neither appears in `to_json`, so the
        # record is unchanged by anything the figure needs.
        self.dual = self._read_dual_curvature(spacetime)
        self._read_certificates(spacetime, config)

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

    # ---- 1. the objective -------------------------------------------

    @staticmethod
    def _read_objective(node):
        terms = node.objective_terms()
        # `objective_of` is static over the terms record, so the total is a
        # fold of the terms the frame already holds. `node.objective()` would
        # be the same double, reached by computing those terms a second time
        # and discarding all but the scalar -- on a large complex that is a
        # second whole-complex band read, and the band is the whole cost.
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
        pairs)` per set -- and rides along as `highlight` in the returned
        block, indexed like `edges`; empty when the caller declares none.
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
        return _json_safe(document)


class EmergenceFrame(AnimationFrame):
    """The neutral emergence instrument retained as the public frame type."""


# =====================================================================
# the drive -- unforced emergence, one frame per engine unit
# =====================================================================

def build_emergence_node(config):
    """Build the neutral host and its unforced-emergence node."""
    host = build_cobordism_host(config["size"], config["host_seed"],
                                config["edge_disposition"])
    node = MC(host, [], [], list(config["register_degrees"]), 1.0,
              config["seed"])
    node.set_objective(cob.JointStationarityObjective())
    # Register construction and entropy stationarity have independent degree
    # domains, so the latter is always declared explicitly.
    node.set_hodge_degrees(list(config["hodge_degrees"]))
    node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                             MC.EmergenceSubmode.STRICT)
    # M0 is held, not targeted. It remains part of the scored cobordism.
    node.declare_pinned_region(M0_REGION, set(boundary_vertices(host)))
    return node, None


def _drive(config, node_factory, frame_factory, reporter, *,
           configure_node=None, progress=False, on_frame=None, on_node=None,
           on_setup=None, stop_requested=None):
    """Drive one animation node through the shared engine-unit loop.

    `on_frame(frames, index)` is called as each unit completes, so a caller
    can display a run while it is still running. `on_node(node)` is the
    backward-compatible node-only setup observer; `on_setup(node, inputs)`
    also carries any experiment-specific input description.
    `stop_requested()` is checked between engine calls so a live worker can
    finish cooperatively before interrupted geometry is inspected. The
    callbacks are otherwise observers: they do not alter an uninterrupted
    drive. It is the ONLY difference
    between a live drive and a headless one: the loop, the engine calls and
    the frames are the same either way, so a live view cannot diverge from
    the run it claims to be showing.

    The supplied factory, frame type, reporter, and optional setup hook define
    the experiment. The engine calls, convergence rule, callback ordering, and
    cancellation behavior remain single-sourced here.

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

    node, inputs = node_factory(config)
    if on_node is not None:
        on_node(node)
    if on_setup is not None:
        on_setup(node, inputs)
    if configure_node is not None:
        configure_node(node, inputs, config)

    # EVERY frame reads `node.spacetime()`, never the host handed to the
    # constructor. Stage 1 REPLACES the node's complex when it commits a move,
    # so the host stops being the complex the node is driving from the first
    # committed move onward. Reading it would freeze every panel at the initial
    # geometry while the objective tracked something else entirely -- the two
    # diverge silently, with no error and no empty frame to give it away.
    frames = [frame_factory(node, node.spacetime(), 0, config, inputs)]
    if progress:
        reporter(frames[-1])
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
        frames.append(frame_factory(node, node.spacetime(), step, config,
                                    inputs))
        if progress:
            reporter(frames[-1])
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


def drive(config, progress=False, on_frame=None, on_node=None, on_setup=None,
          stop_requested=None):
    """Drive the neutral emergence example."""
    return _drive(
        config, build_emergence_node, EmergenceFrame, _report,
        progress=progress, on_frame=on_frame, on_node=on_node,
        on_setup=on_setup, stop_requested=stop_requested)


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
    # Caller-declared edge sets are drawn over the causal colouring so their
    # marked surfaces can be distinguished from the surrounding complex.
    for mark in frame.layout.get("highlight", []):
        for a, b in mark["edges"]:
            va = coords[vertices[a]]
            vb = coords[vertices[b]]
            axis.plot([va[0], vb[0]], [va[1], vb[1]], linewidth=1.7,
                      color=mark["colour"], alpha=0.9, zorder=3)
        handles.append(Line2D([0], [0], color=mark["colour"], linewidth=2.0,
                              label="%s (%d edges)"
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


#: Panels whose painter also takes the frame's stabilized placement. Named
#: rather than detected by signature, so adding a painter that needs it is a
#: deliberate act rather than something that silently starts working.
_PLACED_PANELS = ("layout", "dual_spatial", "dual_temporal")

#: Panels whose painter takes the frames so far rather than one frame: the
#: traces. Named for the same reason as `_PLACED_PANELS`.
_TRACE_PANELS = ("objective",)

#: Every panel in the neutral emergence instrument.
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
]


def panels_for(_config=None):
    """Return the fixed panel order of the neutral emergence example."""
    return _PANELS


#: Grid the panels are laid out on. Wide enough for every panel with room to
#: spare; the spare axes are removed rather than left as empty boxes, which
#: would read as absent measurements.
DECLARED_PANEL_GRID = (4, 5)


def _suptitle(frame, last_step):
    """The figure's title: what was driven, and that the read-outs are
    post-hoc."""
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




def _draw_frame(figure, frames, index, panels, title_fn, trace_panels,
                placed_panels, placed=None):
    """Draw one frame's panels onto a figure.

    `placed` is `stabilize(frames)`. Passing it is optional so a caller can
    draw a single frame without it, in which case the raw layout is used and
    the picture is correct but unaligned.
    """
    figure.clear()
    frame = frames[index]
    placement = placed[index] if placed else None
    rows, columns = DECLARED_PANEL_GRID
    axes = figure.subplots(rows, columns)
    flat = [ax for row in axes for ax in row]
    for axis in flat[len(panels):]:
        figure.delaxes(axis)
    for (name, painter), axis in zip(panels, flat):
        if name in trace_panels:
            painter(axis, frames[:index + 1])
        elif name in placed_panels:
            painter(axis, frame, placement)
        else:
            painter(axis, frame)
    figure.suptitle(title_fn(frame, frames[-1].step), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.95))


def draw_frame(figure, frames, index, placed=None):
    """Draw one neutral emergence frame."""
    return _draw_frame(figure, frames, index, _PANELS, _suptitle,
                       _TRACE_PANELS, _PLACED_PANELS, placed)


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


def _drive_live(config, driver, drawer, progress=False, on_node=None,
                on_setup=None, thread_name="animation-drive"):
    """Drive with supplied experiment callbacks while drawing live frames.

    The compute runs on a worker thread and the figure is drawn on the main
    one, because a GUI toolkit may only be driven from the thread that owns
    it. That is safe here rather than merely conventional: `run_stage1` and
    `run_stage2` release the GIL, so the worker genuinely proceeds while the
    main thread draws, and the worker only ever APPENDS to the frame list
    while the main thread reads indices it has already been handed.

    The supplied driver is unchanged and un-forked. A live run and a
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
            outcome["result"] = driver(
                config, progress=progress, on_frame=publish,
                on_node=on_node, on_setup=on_setup,
                stop_requested=stop.is_set)
        except BaseException as exc:            # re-raised on the main thread
            outcome["error"] = exc
        finally:
            ready.put(None)

    thread = threading.Thread(target=worker, name=thread_name)
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
            drawer(figure, frames, index, placed)
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


def drive_live(config, progress=False, on_node=None, on_setup=None):
    """Drive neutral emergence while displaying each completed frame."""
    return _drive_live(config, drive, draw_frame, progress, on_node, on_setup,
                       "emergence-drive")


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
    return _json_safe(document)




def _render(frames, path, drawer):
    """Render frames with ``drawer`` to a GIF, MP4, or final-frame PNG."""
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
            drawer(figure, frames, len(frames) - 1, placed)
            figure.savefig(path, dpi=110)
            return path
        import matplotlib.animation as animation

        def update(index):
            drawer(figure, frames, index, placed)
            return []

        movie = animation.FuncAnimation(figure, update, frames=len(frames),
                                        interval=900, blit=False, repeat=False)
        writer = "pillow" if lowered.endswith(".gif") else "ffmpeg"
        movie.save(path, writer=writer, dpi=100)
    finally:
        plt.close(figure)
    return path


def render(frames, path):
    """Render the neutral overlay to a GIF, MP4, or final-frame PNG."""
    return _render(frames, path, draw_frame)


# =====================================================================
# CLI
# =====================================================================

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


def _build_common_config(
        steps=DECLARED_STEPS, seed=DECLARED_SEED,
        stage1_iters=DECLARED_STAGE1_ITERS,
        stage2_iters=DECLARED_STAGE2_ITERS,
        tolerance=DECLARED_TOLERANCE, patience=DECLARED_PATIENCE,
        candidate_moves=DECLARED_CANDIDATE_MOVES,
        combinatorial_depth=_LEGACY_UNSET,
        combinatorial_length=_LEGACY_UNSET, *,
        surgical_depth=_LEGACY_UNSET,
        combinatorial_breadth=_LEGACY_UNSET):
    """Validate and return the engine schedule shared by both examples."""
    combinatorial_depth = _aliased_value(
        combinatorial_depth, surgical_depth, DECLARED_COMBINATORIAL_DEPTH,
        "combinatorial_depth", "surgical_depth")
    combinatorial_length = _aliased_value(
        combinatorial_length, combinatorial_breadth,
        DECLARED_COMBINATORIAL_LENGTH, "combinatorial_length",
        "combinatorial_breadth")

    steps = _integer_value("steps", steps)
    seed = _integer_value("seed", seed)
    stage1_iters = _cpp_int_value("stage-one iterations", stage1_iters)
    stage2_iters = _cpp_int_value("stage-two iterations", stage2_iters)
    patience = _integer_value("patience", patience)
    candidate_moves = _cpp_int_value("candidate moves", candidate_moves)
    combinatorial_depth = _cpp_int_value(
        "combinatorial depth", combinatorial_depth)
    combinatorial_length = _cpp_int_value(
        "combinatorial length", combinatorial_length)

    if steps < 0:
        raise ValueError("steps must be non-negative, got %r" % steps)
    if not 0 <= seed <= _UINT64_MAX:
        raise ValueError(
            "seed must be an unsigned 64-bit integer between 0 and %d, got %r"
            % (_UINT64_MAX, seed))
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
        raise ValueError(
            "candidate_moves is how many move specifications stage 1 draws "
            "per unit, or 0 for every candidate there is, so it may not be "
            "negative; got %r" % candidate_moves)
    if patience < 1:
        raise ValueError(
            "patience is a count of consecutive stalled units and must be at "
            "least 1, got %r" % patience)

    tolerance = _finite_value("tolerance", tolerance)
    if tolerance <= 0.0:
        raise ValueError(
            "tolerance must be a positive finite absolute threshold, got %r"
            % tolerance)
    return {
        "steps": steps,
        "seed": seed,
        "candidate_moves": candidate_moves,
        "stage1_iters": stage1_iters,
        "stage2_iters": stage2_iters,
        "tolerance": tolerance,
        "patience": patience,
        "combinatorial_depth": combinatorial_depth,
        "combinatorial_length": combinatorial_length,
    }


def build_config(size=DECLARED_SIZE, steps=DECLARED_STEPS,
                 seed=DECLARED_SEED, host_seed=DECLARED_HOST_SEED,
                 resolution=DECLARED_RESOLUTION,
                 edge_disposition=DECLARED_EDGE_DISPOSITION,
                 stage1_iters=DECLARED_STAGE1_ITERS,
                 stage2_iters=DECLARED_STAGE2_ITERS,
                 tolerance=DECLARED_TOLERANCE,
                 patience=DECLARED_PATIENCE,
                 candidate_moves=DECLARED_CANDIDATE_MOVES,
                 combinatorial_depth=_LEGACY_UNSET,
                 combinatorial_length=_LEGACY_UNSET, *,
                 surgical_depth=_LEGACY_UNSET,
                 combinatorial_breadth=_LEGACY_UNSET):
    """Build the configuration for neutral emergence."""
    config = _build_common_config(
        steps=steps, seed=seed, stage1_iters=stage1_iters,
        stage2_iters=stage2_iters, tolerance=tolerance, patience=patience,
        candidate_moves=candidate_moves,
        combinatorial_depth=combinatorial_depth,
        combinatorial_length=combinatorial_length,
        surgical_depth=surgical_depth,
        combinatorial_breadth=combinatorial_breadth)

    size = _integer_value("size", size)
    host_seed = _integer_value("host seed", host_seed)
    if size < 0:
        raise ValueError("size must be non-negative, got %r" % size)
    if not 0 <= host_seed <= _UINT64_MAX:
        raise ValueError(
            "host seed must be an unsigned 64-bit integer between 0 and %d, "
            "got %r" % (_UINT64_MAX, host_seed))
    if size and host_seed > _UINT64_MAX - (4 * size - 1):
        raise ValueError(
            "host seed plus the %d refinement attempts must stay within the "
            "unsigned 64-bit range; got host seed %r and size %r"
            % (4 * size, host_seed, size))

    resolution = _finite_value("resolution", resolution)
    if resolution <= 0.0:
        raise ValueError("resolution must be a positive finite number, got %r"
                         % resolution)
    if edge_disposition not in EdgeDisposition.ALL:
        raise ValueError(
            "unknown edge disposition %r: expected one of %s"
            % (edge_disposition, ", ".join(EdgeDisposition.ALL)))

    config.update({
        "size": size,
        "host_seed": host_seed,
        "resolution": resolution,
        "edge_disposition": edge_disposition,
        "register_degrees": list(DECLARED_REGISTER_DEGREES),
        "hodge_degrees": list(DECLARED_HODGE_DEGREES),
        "degrees": list(DECLARED_ANALYSIS_DEGREES),
        "betti_degrees": list(DECLARED_BETTI_DEGREES),
        "inputs": "neutral",
    })
    return config

def _add_common_run_arguments(run, out_default, geometry_help):
    """Add the schedule and output options shared by both examples."""
    run.add_argument("--steps", type=int, default=DECLARED_STEPS)
    run.add_argument("--seed", type=int, default=DECLARED_SEED)
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
                          "objective-lowering sequence (default %d)"
                          % DECLARED_COMBINATORIAL_DEPTH)
    run.add_argument("--surgical-depth", type=int,
                     dest="surgical_depth",
                     default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    run.add_argument("--combinatorial-length", type=int,
                     default=_LEGACY_UNSET,
                     help="fixed composition length tried before shorter "
                          "sequences; mutually exclusive with a non-default "
                          "--combinatorial-depth (default %d)"
                          % DECLARED_COMBINATORIAL_LENGTH)
    run.add_argument("--combinatorial-breadth", type=int,
                     dest="combinatorial_breadth",
                     default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    run.add_argument("--tolerance", type=float, default=DECLARED_TOLERANCE,
                     help="absolute objective tolerance (default %g)"
                          % DECLARED_TOLERANCE)
    run.add_argument("--candidate-moves", type=int,
                     dest="candidate_moves",
                     default=DECLARED_CANDIDATE_MOVES,
                     help="move specifications sampled per stage-1 unit; "
                          "zero enumerates every candidate (default %d)"
                          % DECLARED_CANDIDATE_MOVES)
    run.add_argument("--patience", type=int, default=DECLARED_PATIENCE,
                     help="consecutive stalled units allowed before stopping "
                          "(default %d)" % DECLARED_PATIENCE)
    run.add_argument("--live", action="store_true",
                     help="draw each completed frame while the drive runs; "
                          "still writes the requested outputs. Needs the "
                          "'live' extra (pip install -e \".[live]\")")
    run.add_argument("--out", default=out_default,
                     help="GIF or MP4 animation of every frame, or PNG of "
                          "the final frame")
    run.add_argument("--json", default=None,
                     help="also write the per-frame measurements here")
    run.add_argument("--geometry", default=None, help=geometry_help)
    run.add_argument("--quiet", action="store_true")


def build_parser():
    """Build the neutral-emergence command-line parser."""
    parser = argparse.ArgumentParser(
        description="Animate unforced emergence and the paper's certificates.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="drive neutral emergence and render the overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    run.add_argument("--size", type=int, default=DECLARED_SIZE)
    run.add_argument("--host-seed", type=int, default=DECLARED_HOST_SEED)
    run.add_argument("--resolution", type=float, default=DECLARED_RESOLUTION)
    run.add_argument("--edge-disposition", choices=list(EdgeDisposition.ALL),
                     default=DECLARED_EDGE_DISPOSITION,
                     help="causal character of the seed's edges: random, "
                          "spacelike, timelike, lightlike, or foliated")
    _add_common_run_arguments(
        run, "emergence_animation.gif",
        "also write the final complex here (schema 1: cells, edges as "
        "[src, tgt, Re l^2, Im l^2], and vertex times); written however "
        "the drive ends, including an interrupt")
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


def _write_geometry(path, node, inputs, quiet, source=None):
    """Write neutral geometry through the established helper seam."""
    _write_json_document(path, geometry_document(node, inputs, source))
    if not quiet:
        sys.stdout.write("wrote %s\n" % path)


def _run_from_args(args, config, driver, live_driver, renderer,
                   geometry_builder):
    """Run one configured example with shared output and recovery handling."""
    _validate_output_paths(args.out, args.json, args.geometry)
    driven = {}

    def remember_setup(node, inputs):
        driven.update(node=node, inputs=inputs)

    def write_geometry(path, node, inputs):
        # Preserve the neutral helper's public/test seam while allowing an
        # experiment to supply a richer geometry schema.
        if geometry_builder is geometry_document:
            return _write_geometry(path, node, inputs, args.quiet, source)
        _write_json_document(path, geometry_builder(node, inputs, source))
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % path)

    source = source_commit() if args.json or args.geometry else None
    try:
        result = (
            live_driver(config, progress=not args.quiet,
                        on_setup=remember_setup)
            if args.live
            else driver(config, progress=not args.quiet,
                        on_setup=remember_setup))
    except BaseException as error:
        if args.geometry and "node" in driven:
            try:
                write_geometry(args.geometry, driven["node"],
                               driven.get("inputs"))
            except BaseException as geometry_error:
                error.add_note("could not write recovery geometry %r: %s"
                               % (os.fspath(args.geometry), geometry_error))
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

    if args.geometry and "node" in driven:
        write_geometry(args.geometry, driven["node"], result.inputs)
    if args.json:
        document = {
            "config": config,
            "source": source,
            "terminator": result.terminator,
            "stalls": result.stalls,
            "frames": [frame.to_json() for frame in frames],
        }
        if result.inputs is not None:
            document["inputs"] = result.inputs.to_json()
        _write_json_document(args.json, document)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % args.json)
    if args.out:
        rendered = renderer(frames, args.out)
        if not args.quiet:
            sys.stdout.write("wrote %s\n" % rendered)
    return 0


def _extract_legacy_inputs(arguments):
    """Remove the former --inputs option and return its last declared value."""
    cleaned = []
    selected = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--inputs":
            if index + 1 >= len(arguments):
                raise ValueError("--inputs needs neutral or qubit")
            selected = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("--inputs="):
            selected = argument.partition("=")[2]
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    if selected is not None and selected not in ("neutral", "qubit"):
        raise ValueError("unknown input mode %r: expected neutral or qubit"
                         % selected)
    return selected, cleaned


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        legacy_inputs, arguments = _extract_legacy_inputs(arguments)
    except ValueError as error:
        build_parser().error(str(error))
    if legacy_inputs == "qubit":
        if not any(argument == "--out" or argument.startswith("--out=")
                   for argument in arguments):
            # The legacy entrypoint historically wrote this name. The new
            # canonical entrypoint has its own qubit-specific default.
            arguments.extend(("--out", "emergence_animation.gif"))
        # Direct script execution loads this file as ``__main__``. Publish
        # that module under its import name before the dedicated entrypoint
        # imports it, avoiding a second copy of every shared class and helper.
        if __name__ == "__main__":
            sys.modules.setdefault("emergence_animation",
                                   sys.modules[__name__])
        import qubit_animation
        return qubit_animation.main(arguments)

    parser = build_parser()
    args = parser.parse_args(arguments)
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
                args, "combinatorial_breadth", _LEGACY_UNSET))
        _validate_output_paths(args.out, args.json, args.geometry)
    except ValueError as error:
        parser.error(str(error))
    return _run_from_args(args, config, drive, drive_live, render,
                          geometry_document)


if __name__ == "__main__":
    sys.exit(main())
