# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#778 — the complete recursive baryon simulation, replay, and animation.

The capstone driver of epic #763. ONE documented command starts from a
documented NEUTRAL initial complex, runs the unforced joint Regge-Hodge
emergence dynamics under the #776 no-feedback firewall, runs the recursive
post-hoc analysis over the accepted geometry, and reports a particle verdict
from the library's own classifier. A second command replays the persisted
checkpoints with cold caches and verifies every stored verdict and content
hash. A third runs a headless size campaign. A fourth renders the geometric
hierarchy.

**A rigorous NEGATIVE result is valid completion**, and is the measured
outcome at accessible sizes (#777). Design spec §21.4: "An unforced proton is
a scientific success condition, not a software completion condition. The
software is complete if it can return a rigorous negative result." Nothing in
this file is tuned toward a verdict, and the verdict itself is
``ParticleClusters::classifyBaryon``'s own ``classification`` string, relayed
verbatim.

The four verdicts, and nothing else
-----------------------------------
``no baryon`` / ``baryon candidate`` / ``certified proton`` /
``quasi-free sharp-spin obstruction``. There is no target-dependent success
string anywhere in the vocabulary, and no target-dependent code path produces
it: the SAME evidence bundle is assembled from whatever the emergence run
produced — possibly nothing — and handed to the classifier, which names every
missing or failed certificate. ``VERDICTS`` is the complete list and a test
asserts it.

The neutral initial complex
---------------------------
The bare boundary of a 5-simplex — a combinatorial closed S^4, the smallest
closed 4-manifold triangulation — refined by ``--size`` PreGeometric stellar
Pachner adds at a fixed host seed, then given a mild deterministic
non-uniform metric (``l^2 = 1 + 0.01*(index mod 6)``). NEUTRAL means: no
holes, no color windows, no pinned carrier, no boundary blocks, no target
register, no proton-specific term. It is the same host #777 measured on, so
this run's numbers are directly comparable with that study's ensemble.

What is exact, what is certified numerical, what is heuristic
-------------------------------------------------------------
Printed by ``--help`` and emitted in every run document under
``quantity_classes``:

* **EXACT** (closed form / algebraic identity, residual at double round-off):
  static Schur (Kron) reduction; the shifted Feshbach-Schur response pencil;
  second-quantized subset sums and the assembled hopping block; the
  ``Lambda^3 C^3`` singlet Gram and ``F_3`` unitarity; the graded (Pauli)
  amplitude; the spin double cover ``exp(2 pi Sigma) = -I``; the composable
  near-isometry budget; the closed determinant winding (an integer); the
  center sector's cube-root branch relation; the Berry-cancelled exchange and
  rotation characters; the covariance's Hermiticity and its Wick expansions.
* **CERTIFIED NUMERICAL** (a solver result carrying a #764 ``Certificate``
  with a measured residual against a declared tolerance): band isolation and
  projector residuals; band acceptance; the covariance purity defect; derived
  transport acceptance and leakage; labeled-fiber-sum Gram defects; the
  response network's coverage residual; the anchor's calibration margin; the
  heat-kernel spectral dimension.
* **HEURISTIC** (a discovery step with NO exactness claim, reported with its
  own ``HeuristicDiscovery`` grade): modularity community discovery and its
  resolution/persistence tracking; which band of a component is taken as the
  candidate; ``QuarkRead``/``BaryonRead`` ``confidence`` (a passed-gate
  fraction, not a probability); the MDS drawing layout, which is NOT a
  spacetime coordinate system.

Reused machinery, all merged on main
------------------------------------
``MultiCobordism`` (modes, the analysis overlay, the schema-4 checkpoint,
``replay_checkpoint``, ``refinement_decision``/``refine_geometry``),
``PersistentModularity``, ``SpectralFiberTracker``/``SpectralFiber``,
``RecursiveQuotient`` (static / Feshbach / response network / labeled fiber
sum), ``ColorFiber``/``ColorAnchor``, ``FiberConnection`` (transport,
holonomy, determinant winding, fundamental lift), ``ExchangeHolonomy``,
``ParticleClusters`` (quark, bound supercomponent, baryon),
``CovarianceState``, ``LazyFockEngine``, ``OccupationSpectra``,
``AnalyticCache``, and ``Certificate``. Nothing here reimplements any of
them; this file only drives them and records what they returned.

Running it
----------
Cap parallelism to 8 threads; this box is shared::

    OMP_NUM_THREADS=8 .venv-build/bin/python \\
      examples/cobordism/recursive_baryon_simulation.py run --out run.json

    OMP_NUM_THREADS=8 .venv-build/bin/python \\
      examples/cobordism/recursive_baryon_simulation.py replay --from run.json

    OMP_NUM_THREADS=8 .venv-build/bin/python \\
      examples/cobordism/recursive_baryon_simulation.py campaign \\
      --out camp.json

    OMP_NUM_THREADS=8 .venv-build/bin/python \\
      examples/cobordism/recursive_baryon_simulation.py animate \\
      --from run.json --out overlay.png

``run`` exits 0 whether or not a proton emerges — the exit code reports
whether the SOFTWARE ran, never whether the physics obliged. ``replay`` exits
non-zero only when a stored verdict or content hash fails to reproduce.

Reproducibility, exactly
------------------------
Every emitted document carries its ``config_hash``, ``commit``, seeds and
sizes, and embeds every schema-4 checkpoint it produced, so
``MultiCobordism.replay_checkpoint`` reproduces each frame from the record
alone. #579/#776 measured that the engine's move draw is NOT
process-deterministic past the first committed move; a fresh rebuild from
(config, seed, commit) therefore reproduces the first committed move and the
whole relaxation but not a longer trajectory. That limit is a property of the
engine, is stated in every document under ``reproducibility``, and is not
papered over: the CHECKPOINT is the faithful record, and the replay path
replays it.
"""

import argparse
import cmath
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

import tessera as T

cob = T.cobordism
MC = cob.MultiCobordism
EH = T.ExchangeHolonomy
QU = T.quantum

#: The schema version of the RUN document this file writes and reads. The
#: schema-4 CHECKPOINTS it embeds are `MultiCobordism`'s, and are versioned
#: independently.
RUN_SCHEMA_VERSION = 1

#: The COMPLETE verdict vocabulary. Not one entry
#: is a target-dependent success string, and no other string may be emitted.
VERDICTS = (
    "no baryon",
    "baryon candidate",
    "certified proton",
    "quasi-free sharp-spin obstruction",
)

#: `BaryonRead::classification`'s hyphenated spelling, mapped onto the design
#: spec's prose vocabulary. The library produces the left column; this file
#: never invents a verdict of its own.
LIBRARY_VERDICTS = {
    "no-baryon": "no baryon",
    "baryon-candidate": "baryon candidate",
    "certified-proton": "certified proton",
    "quasi-free-sharp-spin-obstruction": "quasi-free sharp-spin obstruction",
}


# =====================================================================
# DECLARED constants — fixed before any datum was examined, identical at
# every size and every seed. Nothing below is retuned per size, and none
# of it was chosen by looking at a particle verdict.
# =====================================================================

#: The fast default host refinement count and the documented larger mode.
DECLARED_SIZE_FAST = 12
DECLARED_SIZE_LARGE = 30

#: Drive steps: how many (stage-1 update + stage-2 relaxation + refinement
#: decision) units the emergence run takes. One unit is the engine's
#: deterministic unit (#579/#776).
DECLARED_DRIVE_STEPS_FAST = 2
DECLARED_DRIVE_STEPS_LARGE = 4

#: The node seed. A seed LABELS an attempt; it does not reproduce one (#579).
DECLARED_SEED = 7

#: The host construction seed, held fixed so "size" varies alone.
DECLARED_HOST_SEED = 3

#: Hodge degrees the post-hoc ANALYSIS enumerates bands at.
DECLARED_DEGREES = (1,)

#: The degrees the emergence objective's register term is configured at.
#: Fixed at one: `MultiCobordism` declares joint Hodge-entropy stationarity
#: over degrees >= 1, so this is the objective's DECLARED domain, not a tuning
#: choice. (Since #805 a degree-zero d(Hodge)/dz does exist -- L_0 is
#: holomorphic in z -- so the restriction is a decision about the objective,
#: not a missing derivative.) The ANALYSIS degrees above are a separate,
#: post-hoc knob.
DECLARED_REGISTER_DEGREES = (1,)

#: The resolution the component/fiber/reduction layer is READ at: gamma = 1,
#: the standard Newman-Girvan value — a literature default, declared here so
#: it is visibly not a fitted choice.
DECLARED_ANALYSIS_RESOLUTION = 1.0

#: The #765 resolution SCAN. Persistence lifetime is identically 1 under a
#: single resolution, so the classifier's `persistence` gate is structurally
#: unpassable there (#777 §4); the scan pass is where it is reachable.
DECLARED_RESOLUTION_SCAN = (0.5, 1.0, 2.0)

#: Stage-1 candidate draws per update and the stage-2 iteration budget.
DECLARED_CANDIDATE_MOVES = 6
DECLARED_STAGE2_ITERS = 12

#: Geometry-only, particle-blind refinement thresholds (#776
#: `RefinementIndicators`). Both are DIMENSIONLESS mesh-quality criteria, so
#: they mean the same thing at every size, and neither is a particle read.
#: `None` = the indicator never fires (the shipped default).
#:
#:   * `curvature_concentration` — max|eps_h| over mean|eps_h| across the
#:     (d-2)-hinges. Above 4 the curvature is piling onto one hinge.
#:   * `mesh_quality` — min|vol| / max|vol| over top cells. Below 0.1 a cell
#:     is an order of magnitude smaller than the largest and is degenerating.
DECLARED_REFINEMENT_THRESHOLDS = {
    "regge_stationarity_residual": None,
    "hodge_stationarity_residual": None,
    "curvature_concentration": 4.0,
    "mesh_quality": 0.1,
    "solver_error": None,
}

#: Cells committed per refinement event, through the EXISTING gated cone-in.
DECLARED_REFINEMENT_CELLS = 1

#: Declared dimensionless shifted-response probe points, as fractions of the
#: operator's own spectral scale, with a declared window half-width.
DECLARED_SHIFT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DECLARED_WINDOW_HALF_WIDTH_FRACTION = 0.1

#: Declared AMLS (Craig-Bampton) retained-mode cutoff.
DECLARED_AMLS_MODE_CUTOFF = 4

#: The heat-kernel diffusion times, as a fixed geometric grid. The EXISTING
#: estimator (`Spacetime.getSpectralDimensionOnSkeleton`) is used unchanged.
DECLARED_SIGMAS = tuple(0.05 * (1.5 ** i) for i in range(20))
DECLARED_KRYLOV_DIM = 64

#: The pinned near-four-dimensional baseline (#777 §11):
#: docs/source/quantum-experiments/overview/h_ds4_status.md.
PINNED_DS_BASELINE = 4.245
PINNED_DS_BASELINE_SIGMA = 0.024

#: The declared closed rotation loop and the plane it turns in.
DECLARED_ROTATION_STEPS = 16
DECLARED_ROTATION_PLANE = (0, 1)
DECLARED_ROTATION_DIMENSIONS = (3, 4)

#: The exchange fixture's ring size, step count and injected common Berry
#: (metric) phase per step. The phase is nonzero ON PURPOSE: an uncancelled
#: loop determinant must be visibly not a sign.
DECLARED_EXCHANGE_CELLS = 8
DECLARED_EXCHANGE_STEPS = 8
DECLARED_BERRY_PHASE = 0.7

#: Campaign sizes and seeds. At least three sizes, so a scaling trend is
#: distinguishable from a single-point coincidence.
DECLARED_CAMPAIGN_SIZES = (6, 12, 20)
DECLARED_CAMPAIGN_SEEDS = (7, 11)

#: A matrix with more than this many entries goes to the versioned binary
#: sidecar instead of inline JSON.
DECLARED_JSON_MATRIX_LIMIT = 64

#: The relative tolerance a replayed continuous aggregate may differ by. #776
#: measured that a hand-built host's edge list is in construction rather than
#: `fromCells` order, so the same weights accumulate in a different order.
#: Every DISCRETE verdict must match exactly; only continuous aggregates get
#: this budget, and the measured difference is always reported.
DECLARED_REPLAY_TOLERANCE = 1e-12

#: Machine-precision bar for the exactness fixtures.
DECLARED_EXACT_TOLERANCE = 1e-12

#: Certificate gates, in the exact order `ParticleClusters` evaluates them.
#: `failedCertificates[0]` is therefore the FIRST failing certificate.
QUARK_GATE_ORDER = (
    "persistence", "localization", "parity-odd", "occupation-one",
    "color-rank-three", "color-rank-stability", "anchor",
    "anchor-stability", "transport-leakage", "winding", "winding-unit",
    "refinement-stability",
)
BARYON_GATE_ORDER = (
    "constituent-quarks", "bound-supercomponent", "color-singlet",
    "color-flux-zero", "baryon-flux-unit", "composite-parity-odd",
    "flavor-uud", "electric-flux-unit", "spin-expectation", "sharp-spin",
    "rotation-character", "spin-lift", "finite-radius", "profile-stability",
    "crossing-readouts",
)

#: The exact / certified-numerical / heuristic classification, emitted in
#: every document and printed by `--help`.
QUANTITY_CLASSES = {
    "exact": [
        "static Schur (Kron) reduction",
        "shifted Feshbach-Schur response pencil",
        "second-quantized subset sums and the assembled hopping block",
        "F_3 unitarity and the Lambda^3 C^3 singlet Gram",
        "the graded (Pauli) amplitude",
        "spin double cover exp(2 pi Sigma) = -I",
        "composable near-isometry budget",
        "closed determinant winding (an integer)",
        "center sector and its cube-root branch relation",
        "Berry-cancelled exchange and rotation characters",
        "covariance Hermiticity and the Wick expansions",
    ],
    "certified_numerical": [
        "band isolation gaps, projector and eigen residuals",
        "band acceptance",
        "covariance purity defect",
        "derived transport acceptance and leakage",
        "labeled-fiber-sum Gram defect",
        "response-network coverage residual",
        "triangle-anchor calibration margin",
        "heat-kernel spectral dimension",
    ],
    "heuristic": [
        "modularity community discovery, its resolution and persistence",
        "which band of a component is taken as the candidate",
        "QuarkRead / BaryonRead confidence (a passed-gate fraction)",
        "the MDS drawing layout, which is NOT a spacetime coordinate system",
    ],
}


# =====================================================================
# the neutral host
# =====================================================================

def build_neutral_host(n_refine, seed=DECLARED_HOST_SEED):
    """The documented NEUTRAL initial complex.

    The bare boundary of a 5-simplex — a combinatorial closed S^4 — refined
    by ``n_refine`` PreGeometric stellar Pachner adds, then given the same
    mild deterministic non-uniform metric. NEUTRAL means every structure the
    epic looks for is absent by construction: no holes, no color windows, no
    pinned carrier, no boundary blocks, no target register.

    Byte-identical in construction to ``tests/cobordism/_closed_s4.py`` and
    to #777's host — deliberately a standalone copy (the ``_holed_surface``
    convention) so the example cannot drift when a test fixture is edited,
    and so an example never imports from the test tree.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                     1.0, 1.0, T.PREFERRED, T.SimplexBoundarySphere(4))
    st.build()
    for edge in st.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(1.0)))
    applied = 0
    for step in range(seed, seed + n_refine * 4):
        move = T.AddMove(st, step, False, T.PachnerMode.PreGeometric, False)
        if move.propose() and move.apply():
            applied += 1
        if applied >= n_refine:
            break
    for index, edge in enumerate(st.getEdgeList().toVector()):
        edge.setLength(cmath.sqrt(complex(1.0 + 0.01 * (index % 6))))
    return st


# =====================================================================
# small helpers — unknown is null, never zero
# =====================================================================

def _finite(value):
    """A float JSON can carry, or None. Unknown is NEVER zero."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _complex_pair(value):
    if value is None:
        return None
    try:
        z = complex(value)
    except (TypeError, ValueError):
        return None
    return [_finite(z.real), _finite(z.imag)]


def _histogram(values):
    out = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _int_histogram(values):
    out = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def mean_sd(values):
    """(mean, sample sd, n) over the finite entries; None where undefined."""
    xs = [v for v in (_finite(v) for v in values) if v is not None]
    if not xs:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(xs) / len(xs)
    if len(xs) < 2:
        return {"mean": mean, "sd": None, "n": len(xs)}
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": mean, "sd": math.sqrt(var), "n": len(xs)}


def linear_fit(xs, ys):
    """OLS ``y = a + b x`` with standard errors and R^2.

    None below three finite pairs: a fit through two points has no residual
    degrees of freedom and therefore no honest uncertainty.
    """
    pairs = [(x, y) for x, y in zip(xs, ys)
             if _finite(x) is not None and _finite(y) is not None]
    if len(pairs) < 3:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    n = len(x)
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx <= 0.0:
        return None
    b = float(((x - xbar) * (y - ybar)).sum() / sxx)
    a = float(ybar - b * xbar)
    resid = y - (a + b * x)
    sse = float((resid ** 2).sum())
    sst = float(((y - ybar) ** 2).sum())
    dof = n - 2
    s2 = sse / dof if dof > 0 else float("nan")
    se_b = math.sqrt(s2 / sxx) if math.isfinite(s2) and s2 >= 0 else None
    se_a = (math.sqrt(s2 * (1.0 / n + xbar ** 2 / sxx))
            if math.isfinite(s2) and s2 >= 0 else None)
    return {"intercept": a, "intercept_se": _finite(se_a),
            "slope": b, "slope_se": _finite(se_b),
            "r_squared": _finite(1.0 - sse / sst) if sst > 0 else None,
            "n": n}


def canonical_json(value):
    """The canonical text a content hash is taken over: sorted keys, no
    incidental whitespace, and a fixed float repr."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def content_hash(value):
    """SHA-256 of `canonical_json(value)` — the block's content hash."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def current_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                     # noqa: BLE001
        return ""


def _rss_bytes():
    """Current resident set size, from /proc; None where unavailable."""
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:                                     # noqa: BLE001
        return None


def _peak_rss_bytes():
    """Peak RSS of this process so far (monotone — never a per-member cost)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# =====================================================================
# the versioned binary sidecar
# =====================================================================

class MatrixSidecar:
    """Versioned binary sidecar for matrices too large for JSON.

    One uncompressed ``.npz`` per run (NPY format 1.0 — a versioned binary
    container, not an ad-hoc blob). Every array carries the SHA-256 of its
    own C-contiguous bytes in the JSON document, and the container file
    carries its own SHA-256, so a replay verifies both the file and each
    array it claims to hold.
    """

    FORMAT = "npz-npy-1.0"

    def __init__(self, path):
        self.path = path
        self._arrays = {}

    def store(self, name, matrix):
        """Record `matrix` and return its JSON descriptor.

        Small matrices are inlined (as split real/imaginary lists) so a small
        run needs no sidecar file at all; large ones go to the container.
        """
        array = np.ascontiguousarray(np.asarray(matrix, dtype=complex))
        descriptor = {
            "name": name,
            "shape": [int(d) for d in array.shape],
            "dtype": "complex128",
            "entries": int(array.size),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        if array.size <= DECLARED_JSON_MATRIX_LIMIT:
            descriptor["storage"] = "inline"
            descriptor["real"] = [float(v) for v in array.real.reshape(-1)]
            descriptor["imag"] = [float(v) for v in array.imag.reshape(-1)]
        else:
            descriptor["storage"] = "sidecar"
            descriptor["format"] = self.FORMAT
            descriptor["file"] = os.path.basename(self.path)
            self._arrays[name] = array
        return descriptor

    def write(self):
        """Flush the container. Returns its descriptor, or None when the run
        produced no matrix large enough to need one."""
        if not self._arrays:
            return None
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(self.path, **self._arrays)
        with open(self.path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        return {"file": os.path.basename(self.path), "path": self.path,
                "format": self.FORMAT, "arrays": sorted(self._arrays),
                "sha256": digest}

    @staticmethod
    def load(descriptor, directory):
        """Rehydrate one descriptor's matrix, verifying its content hash.

        Returns (matrix, ok, reason). `ok` is False with a NAMED reason when
        the bytes are missing or hash differently — never a silent zero.
        """
        shape = tuple(descriptor["shape"])
        if descriptor["storage"] == "inline":
            flat = np.array(descriptor["real"], dtype=float) + \
                1j * np.array(descriptor["imag"], dtype=float)
            array = np.ascontiguousarray(flat.reshape(shape))
        else:
            path = os.path.join(directory, descriptor["file"])
            if not os.path.exists(path):
                return None, False, f"sidecar file missing: {path}"
            with np.load(path) as handle:
                if descriptor["name"] not in handle:
                    return None, False, (
                        f"sidecar {path} has no array {descriptor['name']!r}")
                array = np.ascontiguousarray(handle[descriptor["name"]])
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if digest != descriptor["sha256"]:
            return array, False, (
                f"content hash mismatch for {descriptor['name']!r}: "
                f"stored {descriptor['sha256']}, recomputed {digest}")
        return array, True, None


# =====================================================================
# the recursive readout — the SAME assembly rule the C++ overlay applies
# =====================================================================

class RecursiveReadout:
    """Assemble the recursive hierarchy over one relaxed complex.

    This mirrors ``RecursiveFiberSimulation``'s documented assembly rule
    exactly — components at the analysis resolution, one candidate band per
    component read (the FIRST accepted one), one derived transport per
    ordered pair of same-degree same-rank candidate bands, the pure Slater
    covariance of each accepted band projector — so that the layers the
    schema-4 checkpoint does NOT carry (the response hierarchy, the
    determinant/projective/center holonomy sectors, the covariance
    matrices) are built on identical evidence. The run
    cross-checks its own quark reads against the checkpoint's and records
    the agreement, so a divergence could never hide.

    Read-only: nothing here touches the geometry or the objective.
    """

    def __init__(self, spacetime, config, seed):
        self.spacetime = spacetime
        self.config = config
        self.seed = seed
        self.degrees = list(config["degrees"])
        self.classifier = T.ParticleClusters()
        self.connection = T.FiberConnection()
        self._build()

    # ---- construction ------------------------------------------------

    def _build(self):
        modularity = T.PersistentModularity.fromSpacetime(self.spacetime)
        scan_config = T.PersistentModularityConfig()
        scan_config.resolutions = list(self.config["resolution_scan"])
        scan_config.baseSeed = self.seed
        self.scan_report = modularity.scanResolutions(scan_config)

        analysis_config = T.PersistentModularityConfig()
        analysis_config.resolutions = [self.config["analysis_resolution"]]
        analysis_config.baseSeed = self.seed
        self.report = modularity.scanResolutions(analysis_config)
        self.slice = self.report.slices[0] if self.report.slices else None
        self.components = list(self.slice.components) if self.slice else []
        self.next_level = (list(self.slice.hierarchy[1])
                           if self.slice and len(self.slice.hierarchy) > 1
                           else [])

        fiber_config = T.SpectralFiberConfig()
        fiber_config.degrees = self.degrees
        self.tracker = T.SpectralFiberTracker(self.spacetime, fiber_config)
        self.band_reads = []
        self.band_component = []
        for index, component in enumerate(self.components):
            for degree in self.degrees:
                self.band_reads.append(
                    self.tracker.enumerateBands(component.support, degree))
                self.band_component.append(index)
        # The candidate band of each component read: the FIRST accepted one,
        # the band every downstream read is assembled around. Its POSITION is
        # recorded beside it so a consumer identifies it by index rather than
        # by object identity through the binding.
        self.candidate = []
        self.candidate_index = []
        for read in self.band_reads:
            position = self._first_accepted_index(read)
            self.candidate_index.append(position)
            self.candidate.append(None if position < 0
                                  else read.fibers[position])

        # the pure Slater covariance of each accepted candidate band
        self.states = []
        for fiber in self.candidate:
            if fiber is None:
                self.states.append(None)
                continue
            try:
                self.states.append(
                    QU.CovarianceState.fromBandProjector(fiber.projector()))
            except Exception:                             # noqa: BLE001
                self.states.append(None)

        self._build_transports()
        self._build_quarks()
        self._build_bindings()

    @staticmethod
    def _first_accepted_index(band_read):
        for position, fiber in enumerate(band_read.fibers):
            if fiber.accepted():
                return position
        return -1

    def _build_transports(self):
        self.transports = []
        for a, to_fiber in enumerate(self.candidate):
            if to_fiber is None:
                continue
            for b, from_fiber in enumerate(self.candidate):
                if a == b or from_fiber is None:
                    continue
                if self.band_reads[a].degree != self.band_reads[b].degree:
                    continue
                if to_fiber.rank() != from_fiber.rank():
                    continue
                try:
                    read = self.connection.transportOnSpacetime(
                        self.spacetime, to_fiber, from_fiber)
                except Exception:                         # noqa: BLE001
                    continue
                self.transports.append({"from": b, "to": a, "read": read})
        self.accepted_transports = [t["read"] for t in self.transports
                                    if t["read"].accepted]

    def _persistence_of(self, component_id):
        """The modularity RESOLUTION-slice lifetime of a component.

        REPORT-ONLY since #808: the `persistence` certificate is gated on
        the COBORDISM-FRAME lifetime, so a resolution-slice count — which is
        identically 1 at a single resolution — can no longer make the gate
        structurally unpassable, and a modularity read can no longer veto an
        otherwise certified fiber.
        """
        lifetime, overlap = None, None
        for track in self.report.tracks:
            for member in track.members:
                if member == component_id:
                    lifetime = float(track.lastSlice - track.firstSlice + 1)
                    overlap = float(track.minAdjacentOverlap)
        return lifetime, overlap

    def _build_quarks(self):
        self.quarks = []
        self.quark_band = []
        for index, fiber in enumerate(self.candidate):
            if fiber is None:
                continue
            evidence = T.QuarkCandidateEvidence()
            component_id = self.components[self.band_component[index]].id
            evidence.component = component_id
            evidence.colorBand = fiber
            if self.states[index] is not None:
                evidence.parityRead = self.states[index].wickParity()
                evidence.occupationRead = self.states[index].wickTotalNumber()
            lifetime, overlap = self._persistence_of(component_id)
            if lifetime is not None:
                evidence.persistenceLifetime = lifetime
                evidence.persistenceMinOverlap = overlap
            # This overlay reads ONE cobordism frame, so the frame lifetime
            # is one and its adjacent-frame overlap is vacuously one.  Both
            # are MEASURED facts about this read: `persistence` fails
            # because the candidate was seen in a single frame, which is a
            # statement about the evidence, not about the measurement.
            evidence.frameLifetime = 1.0
            evidence.frameMinOverlap = 1.0
            self.quarks.append(self.classifier.classifyQuark(evidence))
            self.quark_band.append(index)

    def _build_bindings(self):
        self.bindings = []
        self.bound_candidates = []
        if not self.quarks:
            self.binding_reason = "no candidate band produced a quark read"
            return
        if not self.next_level:
            self.binding_reason = (
                "the modularity hierarchy has no next level at the declared "
                "analysis resolution, so there is no supercomponent for a "
                "three-quark binding to live in")
            return
        self.binding_reason = None
        for index, quark in enumerate(self.quarks):
            candidate = T.BoundCandidateEvidence()
            candidate.quark = quark
            candidate.support = list(
                self.band_reads[self.quark_band[index]].support)
            links = [t["read"] for t in self.transports
                     if t["from"] == self.quark_band[index]
                     or t["to"] == self.quark_band[index]]
            candidate.mutualTransports = links
            self.bound_candidates.append(candidate)
        self.bindings = list(self.classifier.boundSupercomponentSearch(
            self.next_level, self.bound_candidates))
        if not self.bindings:
            certified = sum(1 for q in self.quarks
                            if q.classification == "quark")
            self.binding_reason = (
                f"the bound-supercomponent search examined "
                f"{len(self.next_level)} next-level components against "
                f"{len(self.quarks)} quark reads and emitted no binding: "
                f"{certified} of them are CERTIFIED quark candidates, and an "
                f"uncertified candidate is never counted toward a "
                f"supercomponent's membership")

    # ---- the baryon verdict ------------------------------------------

    def baryon_evidence(self):
        """The evidence bundle, and a NAMED account of what is missing.

        Deliberately target-independent: the same bundle is built whatever
        the run produced. The three constituents are the ones the LIBRARY's
        own bound-supercomponent search grouped (its richest binding, ties
        broken by the canonical component id); when it grouped fewer than
        three, the remaining legs stay default-constructed and the
        classifier's structural `constituent-quarks` gate names the gap.
        Nothing is padded with a fabricated read.
        """
        evidence = T.BaryonCandidateEvidence()
        missing = []
        chosen = []
        binding = None
        if self.bindings:
            binding = max(
                self.bindings,
                key=lambda b: (len(b.quarkIndices),
                               b.boundComponent.canonicalHash()))
            evidence.binding = binding
            evidence.boundComponent = binding.boundComponent
            chosen = [self.quarks[i] for i in binding.quarkIndices][:3]
        else:
            missing.append(
                self.binding_reason
                or "the bound-supercomponent search returned no binding")
        if len(chosen) == 3:
            evidence.quarks = chosen
        else:
            missing.append(
                f"the bound-supercomponent search grouped {len(chosen)} "
                "certified quark candidates, not three; the remaining "
                "constituent legs are UNSUPPLIED")
            if chosen:
                padded = list(chosen) + [T.QuarkRead()] * (3 - len(chosen))
                evidence.quarks = padded

        # Colour columns need three ANCHORED rank-three bands. No accepted
        # band on this host is rank three (#777 §4), so the columns stay
        # zero and `color-singlet` fails BY NAME rather than by a fabricated
        # triad.
        anchored = [q for q in chosen if q.triangleAnchorScore is not None]
        if len(anchored) < 3:
            missing.append(
                f"{len(anchored)} of 3 constituents carry a triangle-anchored "
                "rank-three colour band, so the colour columns are UNSUPPLIED")

        # The composite's rotation character needs a spinor carrier on the
        # emergent geometry. The rotation read stays default-constructed and
        # `rotation-character` fails by name.
        carrier = self.spinor_carrier()
        if not carrier["supplied"]:
            missing.append(carrier["reason"])

        # The quasi-free spin reads. This driver supplies NO composite
        # covariance state, and says so as a statement about itself rather
        # than as a claim about the geometry: `wickSpinSquaredExpectation`
        # takes CALLER-SUPPLIED one-particle spin matrices, and #777 §9
        # measured that a declared pairing convention dominates Var(J^2)
        # almost entirely. Quoting a convention-dependent number as a
        # physical one is exactly what a rigorous readout must not do.
        found = bool(binding is not None and binding.found)
        missing.append(
            ("a bound supercomponent WAS found, but this driver supplies no "
             if found else
             "no certified bound composite was found, and this driver "
             "supplies no ")
            + "composite covariance state, so <J^2> and Var(J^2) are "
              "UNSUPPLIED; #777 §9 measured that a declared one-particle "
              "spin convention would dominate them anyway")

        # The mass-radius battery seeds its shells from emergent holes, and
        # the neutral host is built WITHOUT any — a property of the host's
        # construction, not of this run. No `ScaleProfileSample` is supplied
        # and `finite-radius`/`profile-stability` fail by name.
        missing.append(
            "this driver supplies no refinement-window ScaleProfileSample: "
            "the existing mass-radius battery seeds its shells from register "
            "holes, and the documented neutral host is built without any")
        return evidence, missing

    def spinor_carrier(self):
        """Whether any accepted emergent band supplies a spinor frame."""
        for fiber in self.candidate:
            if fiber is None:
                continue
            rows = np.array(fiber.rightFrame()).shape[0]
            if rows in (EH.spinorDimension(3), EH.spinorDimension(4)):
                return {"supplied": True, "reason": None, "rows": int(rows)}
        return {"supplied": False, "rows": None, "reason": (
            "no accepted band frame has the spinor row count; "
            "ExchangeHolonomy needs the rotation-loop frame's row count to "
            "equal the spinor dimension, and an accepted band's frame lives "
            "on its component's cells")}

    def baryon_read(self):
        evidence, missing = self.baryon_evidence()
        return self.classifier.classifyBaryon(evidence), missing


# =====================================================================
# the persisted layers
# =====================================================================

def raw_geometry_block(spacetime):
    """Versioned raw geometry: the top cells and the canonical edge lengths.

    Serialized in the SAME canonical endpoint order the schema-4 checkpoint
    uses, so the block is a pure function of the geometry.
    """
    # The vertex tuple is kept in its INTRINSIC stored order — the cell's own
    # orientation — and only the outer cell LIST is put in a canonical order,
    # so the block is a pure function of the geometry without imposing a
    # vertex convention.
    cells = sorted([int(v.getId()) for v in cell.getVertices()]
                   for cell in spacetime.getTopSimplices())
    edges = {}
    for edge in spacetime.getEdgeList().toVector():
        a, b = edge.getSource().getId(), edge.getTarget().getId()
        edges[(min(a, b), max(a, b))] = complex(edge.getLength())
    return {
        # A top cell of a d-complex has d+1 vertices; that IS the metric
        # dimension `Spacetime.fromCells` wants back.
        "dimensions": (len(cells[0]) - 1) if cells else 0,
        "cells": cells,
        "edges": [{"a": int(a), "b": int(b), "length": _complex_pair(z)}
                  for (a, b), z in sorted(edges.items())],
        "cell_count": len(cells),
        "edge_count": len(edges),
        "vertex_count": len(spacetime.getVertexList().toVector()),
    }


def edge_mode_block(readout, checkpoint):
    """Versioned edge-mode data: which cells carry the one-particle modes.

    The carried modes of an emergence run are the cells each accepted
    candidate band lives on. The checkpoint's own `edge_quantum_data` block
    travels beside it for cross-check.
    """
    modes = []
    for index, fiber in enumerate(readout.candidate):
        if fiber is None:
            continue
        modes.append({
            "band_read": index,
            "degree": int(fiber.degree()),
            "rank": int(fiber.rank()),
            "cells": [[int(v) for v in cell] for cell in fiber.cellVertices()],
            "eigenvalues": [_complex_pair(v) for v in fiber.eigenvalues()],
            "weight_diagonal": [_complex_pair(v)
                                for v in fiber.weightDiagonal()],
        })
    return {
        "carried_mode_bands": len(modes),
        "modes": modes,
        "checkpoint_edge_quantum_data": checkpoint.get("edge_quantum_data"),
    }


def covariance_block(readout, checkpoint, sidecar):
    """The covariance-layer state: Gamma per accepted band, plus its Wick
    reads, purity certificate and the vacuum-embedding (inductive
    compatibility) defect. Matrices travel through the versioned sidecar."""
    entries = []
    for index, state in enumerate(readout.states):
        if state is None:
            fiber = readout.candidate[index]
            entries.append({
                "band_read": index, "available": False,
                "reason": ("no accepted band on this component read"
                           if fiber is None
                           else "the band projector produced no covariance"),
                "gamma": None})
            continue
        gamma = np.array(state.gamma())
        entries.append({
            "band_read": index,
            "available": True,
            "reason": None,
            "modes": int(state.modeCount()),
            "purity_defect": _finite(state.purityDefect()),
            "occupation": _finite(complex(state.wickTotalNumber().value).real),
            "parity": _finite(complex(state.wickParity().value).real),
            "vacuum_embedding_defect": _finite(
                _vacuum_embedding_defect(state)),
            "gamma": sidecar.store(f"gamma_band_{index}", gamma),
        })
    available = [e for e in entries if e["available"]]
    return {
        "states": entries,
        "state_count": len(available),
        "purity_defect_max": _finite(max(
            (e["purity_defect"] for e in available
             if e["purity_defect"] is not None), default=None)),
        "vacuum_embedding_defect_max": _finite(max(
            (e["vacuum_embedding_defect"] for e in available
             if e["vacuum_embedding_defect"] is not None), default=None)),
        "checkpoint_covariance": checkpoint.get("covariance"),
    }


def _vacuum_embedding_defect(state):
    """Falsifier 7: padding Gamma with empty modes must change no amplitude."""
    gamma = np.array(state.gamma())
    modes = gamma.shape[0]
    padded = np.zeros((modes + 2, modes + 2), dtype=complex)
    padded[:modes, :modes] = gamma
    embedded = QU.CovarianceState(padded)
    return max(
        abs(complex(state.wickTotalNumber().value)
            - complex(embedded.wickTotalNumber().value)),
        abs(complex(state.wickParity().value)
            - complex(embedded.wickParity().value)))


def fock_block(readout, checkpoint, config):
    """The global Fock state / DAG.

    Built ONLY for the oracle and explicitly non-Gaussian boundary sectors,
    which is the rule: the quasi-free production representation
    is the covariance above. When the oracle is selected and the operator's
    band projectors are oblique, ``LazyFockEngine`` refuses and its own
    message is recorded as the absence's named reason (#776 finding 7).
    """
    out = {
        "selected": bool(config["fock_oracle"]),
        "present": False,
        "reason": None,
        "nodes": None,
        "modes": None,
        "exact": None,
        "discarded_norm": None,
        "checkpoint_fock_oracle": checkpoint.get("fock_oracle"),
    }
    if not config["fock_oracle"]:
        out["reason"] = (
            "the lazy Fock DAG is an ORACLE / non-Gaussian-boundary path and "
            "is not selected; the quasi-free sector is carried exactly by the "
            "covariance layer (--fock-oracle selects it)")
        return out
    out["reason"] = "no accepted band"
    for index, fiber in enumerate(readout.candidate):
        if fiber is None:
            continue
        projector = np.array(fiber.projector())
        modes = int(projector.shape[0])
        out["modes"] = modes
        # ‖P − P†‖ in the PLAIN inner product. The engine's own premise is
        # P² = P = P^t̄ in the band's signed (Krein) inner product, so this
        # number is a diagnostic beside the engine's verdict, never a
        # substitute for it: the refusal message below is the authority.
        out["hermiticity_defect"] = _finite(float(
            np.abs(projector - projector.conj().T).max()))
        try:
            engine = QU.LazyFockEngine(modes)
            slater = engine.slaterFromProjector(list(range(modes)),
                                                projector, 1e-9)
            out["present"] = bool(slater.state.valid())
            out["nodes"] = (int(slater.state.nodeCount())
                            if out["present"] else None)
            out["discarded_norm"] = (_finite(slater.state.discardedNorm())
                                     if out["present"] else None)
            out["exact"] = bool(engine.exactMode())
            out["reason"] = (None if out["present"]
                             else "the engine returned an invalid state")
        except Exception as error:                        # noqa: BLE001
            # The engine's OWN refusal message is the named reason (#776
            # finding 7): at k >= 1 the signed-weight operator's band
            # projectors are oblique, so no exact Slater reference exists.
            out["reason"] = f"{type(error).__name__}: {error}"
        break
    return out


def response_hierarchy_block(readout, config):
    """The response hierarchy: static Schur, the shifted Feshbach windows,
    AMLS, the response network's vertices/stalks/edges, and the realization
    attempt. This is what the animation's response panel draws."""
    out = {"built": False, "reason": None, "regime": None,
           "static": None, "shifted": [], "amls": None,
           "response_network": None, "realization": None,
           "labeled_fiber_sums": []}
    if not readout.components:
        out["reason"] = "no components discovered"
        return out
    supports = [list(component.support) for component in readout.components]
    degree = readout.degrees[0]
    try:
        quotient = cob.RecursiveQuotient.overVertexSupports(
            readout.spacetime, degree, supports,
            cob.RecursiveQuotient.Options())
    except Exception as error:                            # noqa: BLE001
        out["reason"] = f"{type(error).__name__}: {error}"
        return out
    out["built"] = True
    out["regime"] = str(quotient.regime)
    out["dimension"] = int(quotient.dimension)
    out["component_count"] = int(quotient.componentCount)
    out["interface_cells"] = len(quotient.interfaceIndices)

    static = quotient.staticReduction()
    out["static"] = {
        "solve_residual": _finite(static.solveResidual),
        "compatibility_residual": _finite(static.compatibilityResidual),
        "kept_coordinates": len(static.coordinates),
        "certificate_grade": str(static.certificate.grade),
        "certificate_holds": bool(static.certificate.holds()),
    }

    eigenvalues = []
    for band_read in readout.band_reads:
        eigenvalues.extend(list(band_read.coveredEigenvalues))
    scale = max((abs(complex(v)) for v in eigenvalues), default=0.0) or 1.0
    half = config["window_half_width_fraction"] * scale
    for fraction in config["shift_fractions"]:
        lam = fraction * scale
        entry = {"fraction": fraction, "lambda": _finite(lam),
                 "window": [_finite(lam - half), _finite(lam + half)]}
        try:
            read = quotient.feshbach(lam, lam - half, lam + half)
            entry.update({
                "resonant": bool(read.resonant),
                "solve_residual": _finite(read.solveResidual),
                "compatibility_residual": _finite(read.compatibilityResidual),
                "determinant_residual": _finite(read.determinantResidual),
                "certificate_grade": str(read.certificate.grade),
                "certificate_holds": bool(read.certificate.holds()),
            })
        except Exception as error:                        # noqa: BLE001
            entry.update({"available": False,
                          "reason": f"{type(error).__name__}: {error}"})
        out["shifted"].append(entry)

    try:
        amls = quotient.craigBampton(0.0, scale, config["amls_mode_cutoff"],
                                     -1.0)
        out["amls"] = {
            "available": True,
            "retained_modes": [int(m) for m in amls.retainedModes],
            "discarded_mode_gap": _finite(amls.discardedModeGap),
            "max_eigen_residual": _finite(
                max((abs(v) for v in amls.eigenResiduals), default=None)),
            "certificate_grade": str(amls.certificate.grade),
            "certificate_holds": bool(amls.certificate.holds()),
        }
    except Exception as error:                            # noqa: BLE001
        out["amls"] = {"available": False, "retained_modes": None,
                       "max_eigen_residual": None,
                       "reason": f"{type(error).__name__}: {error}"}

    try:
        network = quotient.responseNetwork()
        out["response_network"] = {
            "stalk_dimensions": [int(d) for d in network.stalkDimensions],
            "stalk_coordinates": [[int(c) for c in stalk]
                                  for stalk in network.stalkCoordinates],
            "edges": [[int(edge.from_component), int(edge.to_component)]
                      for edge in network.edges],
            "edge_count": len(network.edges),
            "coverage_residual": _finite(network.coverageResidual),
            "certificate_grade": str(network.certificate.grade),
            "certificate_holds": bool(network.certificate.holds()),
        }
    except Exception as error:                            # noqa: BLE001
        out["response_network"] = {
            "reason": f"{type(error).__name__}: {error}",
            "stalk_dimensions": None, "edges": None}

    try:
        sheaf = quotient.sheafRealization()
        if sheaf.emitted and sheaf.simplicial:
            kind = "simplicial_sheaf"
        elif sheaf.emitted:
            kind = "cellular_sheaf"
        else:
            kind = "general_response_network"
        out["realization"] = {
            "type": kind,
            "emitted": bool(sheaf.emitted),
            "simplicial": bool(sheaf.simplicial),
            # An unemitted realization reconstructed nothing, so it has no
            # residual. The struct default is 0.0, which would read as
            # "exact"; unknown is null here, never zero.
            "reconstruction_residual": (_finite(sheaf.reconstructionResidual)
                                        if sheaf.emitted else None),
            "certificate_grade": str(sheaf.certificate.grade),
            "certificate_holds": bool(sheaf.certificate.holds()),
        }
    except Exception as error:                            # noqa: BLE001
        out["realization"] = {"type": None,
                              "reason": f"{type(error).__name__}: {error}"}

    try:
        labeled = quotient.labeledFiberSum()
        defect = _finite(labeled.gramDefect)
        # The retained-fiber Gram G = J* W J is BIMODAL, for the structural
        # reason #777 §5 records: an all-positive weight diagonal makes the
        # embedding an exact isometry, while a single negative (Krein) weight
        # flips one diagonal entry to -1 and puts ||G - I|| at exactly 2. The
        # regime is classified, never averaged across.
        if defect is None:
            regime = None
        elif defect < 1e-12:
            regime = "isometric"
        elif abs(defect - 2.0) < 1e-1:
            regime = "signature_flipped"
        else:
            regime = "intermediate"
        out["labeled_fiber_sums"].append({
            "degree": int(degree),
            "nominal_rank": int(labeled.nominalRank),
            "effective_rank": int(labeled.effectiveRank),
            "gram_defect": defect,
            "gram_defect_regime": regime,
            "quotient_nullity": int(labeled.quotientNullity),
            "certificate_grade": str(labeled.certificate.grade),
            "certificate_holds": bool(labeled.certificate.holds()),
        })
    except Exception as error:                            # noqa: BLE001
        out["labeled_fiber_sums"] = [
            {"degree": int(degree),
             "reason": f"{type(error).__name__}: {error}"}]
    return out


def fibers_block(readout, sidecar):
    """Fibers and their signatures — the band layer, with the Krein
    signature and the accept/reject certificate of each."""
    entries = []
    for index, band_read in enumerate(readout.band_reads):
        bands = []
        for position, fiber in enumerate(band_read.fibers):
            certificate = fiber.certificate()
            band = {
                "index": position,
                "degree": int(fiber.degree()),
                "rank": int(fiber.rank()),
                "accepted": bool(fiber.accepted()),
                "band_center": _complex_pair(fiber.bandCenter()),
                "lower_gap": _finite(certificate.lowerGap),
                "upper_gap": _finite(certificate.upperGap),
                "localization": _finite(certificate.localization),
                "gram_defect": _finite(certificate.gramDefect),
                "eigen_residual": _finite(certificate.eigenResidual),
                "projector_residual": _finite(certificate.projectorResidual),
                "nearest_discarded_separation": _finite(
                    certificate.nearestDiscardedSeparation),
                "localization_excess": _finite(
                    certificate.localizationExcess),
                "projector_norm": _finite(certificate.projectorNorm),
                "frame_condition_number": _finite(
                    certificate.frameConditionNumber),
                "self_adjoint": bool(certificate.selfAdjoint),
                "krein_positive": int(certificate.positiveSignature),
                "krein_negative": int(certificate.negativeSignature),
            }
            # The CANDIDATE band — the first accepted one — is the band every
            # downstream read is assembled around, so its projector is the
            # one the record has to carry. Persisting all of them would be
            # bulk, not evidence.
            if position == readout.candidate_index[index]:
                band["candidate"] = True
                band["projector"] = sidecar.store(
                    f"projector_read_{index}_band_{position}",
                    np.array(fiber.projector()))
            bands.append(band)
        entries.append({
            "band_read": index,
            "component": readout.components[
                readout.band_component[index]].id.canonicalHash(),
            "degree": int(band_read.degree),
            "support": [int(v) for v in band_read.support],
            "regime": str(band_read.regime),
            "solver_path": str(band_read.solverPath),
            "bands": bands,
        })
    all_bands = [b for e in entries for b in e["bands"]]
    accepted = [b for b in all_bands if b["accepted"]]
    return {
        "reads": entries,
        "total": len(all_bands),
        "accepted": len(accepted),
        "rank_histogram": _int_histogram(b["rank"] for b in all_bands),
        "accepted_rank_histogram": _int_histogram(b["rank"] for b in accepted),
        "rank_three_accepted": sum(1 for b in accepted if b["rank"] == 3),
        "self_adjoint": sum(1 for b in all_bands if b["self_adjoint"]),
        "krein_indefinite": sum(1 for b in all_bands
                                if b["krein_negative"] > 0),
    }


def transports_block(readout):
    """Derived transports with the determinant, projective and center
    sectors, the closed determinant winding, and the Wilson value.

    Every channel that has no accepted link reports `null` with the SAME
    named reason — never a fabricated value.
    """
    details = []
    for record in readout.transports:
        read = record["read"]
        details.append({
            "from": record["from"], "to": record["to"],
            "degree": int(read.degree),
            "rank": int(read.rank),
            "numerical_rank": int(read.numericalRank),
            "accepted": bool(read.accepted),
            "leakage": _finite(read.leakage),
            "overlap_condition_number": _finite(read.overlapConditionNumber),
            "frame_condition_number": _finite(read.frameConditionNumber),
            "determinant_phase": _complex_pair(read.determinantPhase),
            "rejection_reason": read.rejectionReason or None,
            "regime": str(read.regime),
            "krein_to": [int(read.toPositiveSignature),
                         int(read.toNegativeSignature)],
            "krein_from": [int(read.fromPositiveSignature),
                           int(read.fromNegativeSignature)],
        })
    accepted = readout.accepted_transports
    out = {
        "derived": len(details),
        "accepted": len(accepted),
        "leakage_min": _finite(min((d["leakage"] for d in details
                                    if d["leakage"] is not None),
                                   default=None)),
        "leakage_max": _finite(max((d["leakage"] for d in details
                                    if d["leakage"] is not None),
                                   default=None)),
        "rejection_reasons": _histogram(d["rejection_reason"]
                                        for d in details),
        "detail": details,
    }
    absent = ("no accepted derived transport on this complex: the derived "
              "transfer is the off-diagonal Hodge block between two "
              "components' cells, and between rank-1 bands of a vertex "
              "partition's components that block contracts to a numerically "
              "zero overlap")
    if not accepted:
        for channel in ("full", "determinant", "projective", "center",
                        "winding"):
            out[channel] = {"available": False, "reason": absent}
        return out

    try:
        holonomy = readout.connection.holonomy(accepted)
        out["full"] = {
            "available": True,
            "closed": bool(holonomy.closed),
            "rank": int(holonomy.rank),
            "loop_length": int(holonomy.loopLength),
            "normalized_trace": _complex_pair(holonomy.normalizedTrace),
            "unitary": bool(holonomy.unitary),
            "unitarity_residual": _finite(holonomy.unitarityResidual),
            "certificate_grade": str(holonomy.certificate.grade),
            "certificate_holds": bool(holonomy.certificate.holds()),
        }
        out["determinant"] = {
            "available": True,
            "determinant": _complex_pair(holonomy.determinant),
            "modulus": _finite(abs(complex(holonomy.determinant))),
        }
        if holonomy.unitary:
            gate = _colour_gate()
            representative = np.array(
                T.FiberConnection.projectiveRepresentative(
                    holonomy.holonomy, gate))
            out["projective"] = {
                "available": True,
                "representative_trace": _complex_pair(
                    complex(np.trace(representative))),
                "adjoint_trace": _complex_pair(holonomy.adjointTrace),
                "center_blind": True,
            }
        else:
            out["projective"] = {
                "available": False,
                "reason": ("the closed holonomy is not unitary to the "
                           "declared tolerance, so it has no projective "
                           "representative")}
    except Exception as error:                            # noqa: BLE001
        message = f"{type(error).__name__}: {error}"
        out["full"] = {"available": False, "reason": message}
        out["determinant"] = {"available": False, "reason": message}
        out["projective"] = {"available": False, "reason": message}

    if matrix_rank_is_three(readout):
        centers = {}
        for branch in (0, 1, 2):
            try:
                lift = readout.connection.fundamentalLift(
                    accepted, _colour_gate(), branch)
                centers[str(branch)] = {
                    "valid": bool(lift.valid),
                    "center_sector": int(lift.centerSector),
                    "lift_trace": _complex_pair(lift.liftTrace),
                    "determinant_residual": _finite(lift.detResidual),
                    "invalid_reason": lift.invalidReason or None,
                }
            except Exception as error:                    # noqa: BLE001
                centers[str(branch)] = {
                    "valid": False,
                    "invalid_reason": f"{type(error).__name__}: {error}"}
        out["center"] = {"available": True, "branches": centers}
    else:
        out["center"] = {"available": False, "reason": (
            "the Z_3 center sector is defined on rank-three links; the "
            "accepted transports are not rank three")}

    try:
        winding = readout.connection.closedFamilyWinding(accepted)
        out["winding"] = {
            "available": winding.winding is not None,
            "winding": winding.winding,
            "closure": str(winding.windingClosure),
            "closure_defect": _finite(winding.closureDefect),
            "max_phase_step": _finite(winding.maxPhaseStep),
            "invalidation_reason": winding.invalidationReason or None,
            "certificate_holds": bool(winding.certificate.holds()),
        }
        if (winding.winding is None
                and not out["winding"]["invalidation_reason"]):
            out["winding"]["reason"] = "the winding did not certify"
    except Exception as error:                            # noqa: BLE001
        out["winding"] = {"available": False, "winding": None,
                          "reason": f"{type(error).__name__}: {error}"}
    return out


def matrix_rank_is_three(readout):
    return bool(readout.accepted_transports) and all(
        int(read.rank) == 3 for read in readout.accepted_transports)


def statistics_block(readout):
    """Berry-cancelled exchange and rotation characters.

    Exact on the DECLARED analytic carriers at every size; honestly absent
    on the emergent one, with the reason named.
    """
    out = {"declared_carrier": {}, "emergent_carrier": None,
           "spin_lift": None}
    plane_a, plane_b = DECLARED_ROTATION_PLANE
    for dimension in DECLARED_ROTATION_DIMENSIONS:
        spinor = np.array(EH.transverseSpinorFrame(plane_a, plane_b,
                                                   dimension))
        weights = np.ones(spinor.shape[0], dtype=complex)
        rotation = EH.rotationLoopFrames(spinor, plane_a, plane_b, dimension,
                                         1, DECLARED_ROTATION_STEPS)
        reference = EH.referenceLoopFrames(spinor, DECLARED_ROTATION_STEPS)
        character = EH.rotationCharacter(EH.loopHolonomy(rotation, weights),
                                         EH.loopHolonomy(reference, weights))
        vector0 = np.eye(dimension, dtype=complex)[:, :1]
        vector_rotation = EH.vectorLoopFrames(vector0, plane_a, plane_b,
                                              dimension, 1,
                                              DECLARED_ROTATION_STEPS)
        vector_reference = EH.referenceLoopFrames(vector0,
                                                  DECLARED_ROTATION_STEPS)
        vector_character = EH.rotationCharacter(
            EH.loopHolonomy(vector_rotation,
                            np.ones(dimension, dtype=complex)),
            EH.loopHolonomy(vector_reference,
                            np.ones(dimension, dtype=complex)))
        matrix = np.array(EH.spinorRotation(2.0 * math.pi, plane_a, plane_b,
                                            dimension))
        out["declared_carrier"][str(dimension)] = {
            "spinor_character": _complex_pair(character.character),
            "spinor_sign": int(character.characterSign),
            "spinor_certificate_holds": bool(character.certificate.holds()),
            "vector_character": _complex_pair(vector_character.character),
            "vector_sign": int(vector_character.characterSign),
            "double_cover_residual": _finite(float(
                np.abs(matrix + np.eye(matrix.shape[0])).max())),
        }
    out["emergent_carrier"] = readout.spinor_carrier()
    out["spin_lift"] = {
        "status": None,
        "reason": ("no emergent tangent-frame atlas: ExchangeHolonomy."
                   "spinLift needs Cech SO(d) edge rotations over a cover, "
                   "and the relaxed complex supplies none")}
    return out


def scan_particles_block(scan_checkpoint):
    """The C++ overlay's OWN reads on the declared resolution scan.

    With one resolution the persistence lifetime is identically 1, so the
    `persistence` gate is structurally unpassable and would be the first
    failing certificate for reasons of the MEASUREMENT rather than the
    physics (#777 §4). The scan pass is where persistence is reachable, and
    both readings travel together so neither can be mistaken for the other.
    """
    quarks = scan_checkpoint["particles"]["quarks"]
    first = {}
    for quark in quarks:
        failed = quark.get("failed_certificates") or []
        key = failed[0] if failed else "none"
        first[key] = first.get(key, 0) + 1
    return {
        "resolutions": scan_checkpoint["analysis"]["resolutions"],
        "quark_reads": len(quarks),
        "classifications": _histogram(q["classification"] for q in quarks),
        "first_failing_certificate": dict(sorted(first.items())),
        "all_failing_certificates": _histogram(
            name for q in quarks
            for name in (q.get("failed_certificates") or [])),
        "bound_supercomponents": len(
            scan_checkpoint["particles"]["bound_supercomponents"]),
        "bound_supercomponents_found": sum(
            1 for b in scan_checkpoint["particles"]["bound_supercomponents"]
            if b["found"]),
        "baryons": len(scan_checkpoint["particles"]["baryons"]),
        "baryon_classifications": _histogram(
            b["classification"]
            for b in scan_checkpoint["particles"]["baryons"]),
    }


def particles_block(readout, checkpoint):
    """Quark, bound-supercomponent and baryon reads, plus the agreement
    between this driver's Python-side reads and the C++ overlay's."""
    quarks = []
    for index, read in enumerate(readout.quarks):
        quarks.append({
            "component": read.component.canonicalHash(),
            "classification": read.classification,
            "confidence": _finite(read.confidence),
            "color_rank": int(read.colorRank),
            "exterior_parity": int(read.exteriorParity),
            "triangle_anchor_score": _finite(read.triangleAnchorScore),
            "determinant_winding": read.determinantWinding,
            "winding_closure": str(read.windingClosure),
            "baryon_flux": _finite(read.baryonFlux),
            "isospin": _finite(read.isospin),
            "electric_flux": _finite(read.electricFlux),
            "occupation_total": _finite(read.occupationTotal),
            "transport_count": int(read.transportCount),
            "transport_leakage_max": _finite(read.transportLeakageMax),
            "persistence_lifetime": _finite(read.persistenceLifetime),
            "frame_lifetime": _finite(read.frameLifetime),
            "localization": _finite(read.localization),
            "failed_certificates": list(read.failedCertificates),
        })
    bindings = []
    for read in readout.bindings:
        bindings.append({
            "bound_component": read.boundComponent.canonicalHash(),
            "found": bool(read.found),
            "constituents": len(read.quarks),
            "lifetime_overlap": _finite(read.lifetimeOverlap),
            "min_containment": _finite(read.minContainment),
            "transport_leakage_max": _finite(read.transportLeakageMax),
            "failed_certificates": list(read.failedCertificates),
        })
    first_failure = {}
    for quark in quarks:
        failed = quark["failed_certificates"]
        key = failed[0] if failed else "none"
        first_failure[key] = first_failure.get(key, 0) + 1

    checkpoint_quarks = checkpoint.get("particles", {}).get("quarks", [])
    agreement = {
        "checkpoint_quark_reads": len(checkpoint_quarks),
        "driver_quark_reads": len(quarks),
        "classifications_match": (
            _histogram(q["classification"] for q in quarks)
            == _histogram(q["classification"] for q in checkpoint_quarks)),
        "failed_certificates_match": (
            [sorted(q["failed_certificates"]) for q in quarks]
            == [sorted(q.get("failed_certificates") or [])
                for q in checkpoint_quarks]),
    }
    return {
        "quarks": quarks,
        "quark_reads": len(quarks),
        "certified_quarks": sum(1 for q in quarks
                                if q["classification"] == "quark"),
        "classifications": _histogram(q["classification"] for q in quarks),
        "first_failing_certificate": dict(sorted(first_failure.items())),
        "all_failing_certificates": _histogram(
            name for q in quarks for name in q["failed_certificates"]),
        "bound_supercomponents": bindings,
        "bindings_found": sum(1 for b in bindings if b["found"]),
        "gluons": [],
        "gluons_reason": (
            "an even colour-octet read needs a certified colour bilinear on "
            "a rank-three band; none exists on this complex"),
        "checkpoint_agreement": agreement,
    }


def verdict_block(readout):
    """The particle verdict, relayed VERBATIM from the library classifier.

    No target-dependent code path: the same evidence bundle is assembled
    whatever the run produced, and `ParticleClusters::classifyBaryon` returns
    the classification and names every failed or missing certificate.
    """
    read, missing = readout.baryon_read()
    if read.classification not in LIBRARY_VERDICTS:
        raise AssertionError(
            f"unknown classification {read.classification!r}: the verdict "
            f"vocabulary is {sorted(LIBRARY_VERDICTS)}")
    return {
        "verdict": LIBRARY_VERDICTS[read.classification],
        "library_classification": read.classification,
        "vocabulary": list(VERDICTS),
        "failed_certificates": list(read.failedCertificates),
        "first_failing_certificate": (read.failedCertificates[0]
                                      if read.failedCertificates else None),
        "missing_evidence": missing,
        "confidence": _finite(read.confidence),
        "confidence_is": ("the fraction of the fifteen gates that passed — "
                          "a passed-gate fraction, NOT a probability"),
        "certificate_grade": str(read.certificate.grade),
        "certificate_holds": bool(read.certificate.holds()),
        "color_gram_determinant": _finite(read.colorGramDeterminant),
        "color_wedge": _complex_pair(read.colorWedge),
        "color_flux": _finite(read.colorFlux),
        "baryon_flux": read.baryonFlux,
        "electric_flux": read.electricFlux,
        "total_j2": read.totalJ2,
        "total_j2_variance": read.totalJ2Variance,
        "sharp_spin": bool(read.sharpSpin),
        "quasi_free_class_swept": bool(read.quasiFreeClassSwept),
        "class_variance_floor": _finite(read.classVarianceFloor),
        "rotation_character": _complex_pair(read.rotationCharacter),
        "exterior_parity": int(read.exteriorParity),
        "flavor_pattern": read.flavorPattern or None,
        "total_isospin": read.totalIsospin,
        "radius_finite": bool(read.radiusFinite),
        "profile_stable": bool(read.profileStable),
        "describe": read.describe(),
    }


def spectral_dimension_block(spacetime, config):
    """The EXISTING heat-kernel estimator on the weighted 1-skeleton of top
    simplices, reused verbatim, against the pinned near-4D baseline."""
    tops = list(spacetime.getTopSimplices())
    if not tops:
        return {"available": False, "reason": "no top simplices"}
    top_k = len(next(iter(tops)).getVertices()) - 1
    curve = list(spacetime.getSpectralDimensionOnSkeleton(
        list(config["sigmas"]), config["krylov_dim"], T.AllSimplexFilter(),
        top_k, 1))
    peak = max(curve) if curve else None
    return {
        "available": True,
        "sigmas": [_finite(s) for s in config["sigmas"]],
        "curve": [_finite(v) for v in curve],
        "peak": _finite(peak),
        "peak_sigma": (_finite(config["sigmas"][curve.index(peak)])
                       if peak is not None else None),
        "top_k": top_k,
        "baseline": PINNED_DS_BASELINE,
        "baseline_sigma": PINNED_DS_BASELINE_SIGMA,
        "deviation_from_baseline": (_finite(peak - PINNED_DS_BASELINE)
                                    if peak is not None else None),
        "baseline_source":
            "docs/source/quantum-experiments/overview/h_ds4_status.md",
    }


# =====================================================================
# exactness fixtures — analytic / dense references
# =====================================================================


def _declared_anchor_gate():
    """An OPEN gate for the pure-algebra controls below.

    Those controls exercise an algebraic identity of the cube-root branch on a
    matrix of their own construction, not a colour reading of an emergent
    band, so they declare their own anchor rather than borrowing one. Two
    faces SHARING the boundary edge rows {0, 1} are required: an isolated face
    has no overlap partner and reports unknown determinant-phase coherence,
    which the acceptance predicate refuses.
    """
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    band = np.zeros((4, 3), dtype=complex)
    band[0, 0] = 1.0
    band[1, 1] = 1.0
    band[2, 2] = np.sqrt(0.8)
    band[3, 2] = np.sqrt(0.2)
    anchor = T.ColorAnchor([T.OrientedTriangle([0, 1, 2], [1, 1, 1]),
                            T.OrientedTriangle([0, 1, 3], [1, 1, 1])])
    anchor.declareWeights([0.75, 0.25])
    return T.ColorAnchor.gateFor(anchor.evaluate(band, weights))


def _colour_gate(anchor_profile=None):
    """The triangle-anchor gate the exactness contract requires before any
    colour-specific kernel runs.

    A band with no measured anchor gets a CLOSED gate, so the projective
    representative and the centre lift refuse rather than emit an unlicensed
    colour datum; the refusal travels with the read.
    """
    if anchor_profile is None:
        return T.AnchorGate()
    return T.ColorAnchor.gateFor(anchor_profile)

def _record_fixture(out, name, residual, tolerance, grade, detail=None):
    out.append({
        "name": name,
        "residual": _finite(residual),
        "tolerance": tolerance,
        "grade": grade,
        "exact": bool(residual is not None and residual <= tolerance),
        "detail": detail or {},
    })


def _synthetic_unit_fiber(base_id, rank):
    """A rank-`rank` fiber with an identity frame on `rank` synthetic cells,
    built through the PUBLIC `SpectralFiber.fromRecord` schema — the
    documented replay/serialization route. Standalone here so the example
    never imports from the test tree."""
    def split(name, values, record):
        array = np.asarray(values, dtype=complex).reshape(-1)
        record[name + "_re"] = [float(v.real) for v in array]
        record[name + "_im"] = [float(v.imag) for v in array]

    right = np.eye(rank, dtype=complex)
    record = {
        "schema_version": 2, "record_type": "spectral_fiber",
        "cells": [[base_id + i] for i in range(rank)],
        "rows": rank, "rank": rank,
        "certificate": {
            "degree": 1, "rank": rank, "lower_gap": 1.0, "upper_gap": 1.0,
            "localization": 0.5, "projector_residual": 1e-16,
            "eigen_residual": 1e-16, "left_residual": 1e-16,
            "gram_defect": 0.0, "projector_norm": 1.0,
            "frame_condition_number": 1.0,
            "nearest_discarded_separation": 1.0,
            "localization_support_fraction": 0.5,
            "localization_excess": 0.0,
            "positive_signature": rank, "negative_signature": 0,
            "frequency_lower": 0.0, "frequency_upper": 2.0,
            "self_adjoint": True, "accepted": True,
            "certificate": {"grade": "certified-numerical",
                            "domain": "band-window",
                            "regime": "positive-semidefinite",
                            "residual": 1e-15, "conditioning": 1.0,
                            "dense_reference_error": float("nan"),
                            "tolerance": 1e-9}}}
    split("eigenvalues", [1.0 + 0j] * rank, record)
    split("right_frame", right, record)
    split("left_frame", right, record)
    split("weights", np.ones(rank, dtype=complex), record)
    return T.SpectralFiber.fromRecord(record)


def _ring_mode(position, cells):
    """A localized unit mode at ring position `position`."""
    k = int(math.floor(position)) % cells
    fraction = position - math.floor(position)
    vector = np.zeros(cells, dtype=complex)
    vector[k] += math.cos(fraction * math.pi / 2.0)
    vector[(k + 1) % cells] += math.sin(fraction * math.pi / 2.0)
    return vector


def _translation_frames(positions, cells, steps, distance):
    frames = []
    for step in range(steps):
        shift = distance * step / steps
        frames.append(np.stack(
            [_ring_mode((p + shift) % cells, cells) for p in positions],
            axis=1))
    return frames


def exactness_fixtures():
    """The named exactness fixtures, each against an analytic or dense
    reference computed independently in this file.

    Design spec §21.1 items 2, 3, 4, 11, 12, 13, 15 — the set the #778 ticket
    names: static Schur, shifted Feshbach, second-quantized subset-sum and
    hopping, triangle anchor, center branch, and Berry cancellation.
    """
    out = []

    # ---- 1. static Schur (Kron) reduction, hand-solvable path ----------
    operator = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=complex)
    quotient = cob.RecursiveQuotient.overMatrix(
        operator.reshape(-1).tolist(), 3, [], [[0, 1], [1, 2]],
        cob.RecursiveQuotient.Options())
    static = quotient.staticReduction()
    kept = len(static.coordinates)
    effective = np.array(static.effectiveOperator).reshape(kept, kept)
    interface = list(quotient.interfaceIndices)
    interior = [i for i in range(3) if i not in interface]
    analytic = (operator[np.ix_(interface, interface)]
                - operator[np.ix_(interface, interior)]
                @ np.linalg.inv(operator[np.ix_(interior, interior)])
                @ operator[np.ix_(interior, interface)])
    _record_fixture(out, "static_schur_path",
                    float(np.abs(effective - analytic).max()),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"interface": interface, "interior": interior,
                     "solve_residual": _finite(static.solveResidual),
                     "compatibility_residual":
                         _finite(static.compatibilityResidual),
                     "reference": "L_BB - L_BI L_II^{-1} L_IB, computed here"})

    # ---- 2. shifted Feshbach-Schur pencil ------------------------------
    lam = 0.3
    read = quotient.feshbach(lam, lam - 0.1, lam + 0.1)
    response = np.array(read.response).reshape(kept, kept)
    shifted = (operator[np.ix_(interface, interface)]
               - lam * np.eye(len(interface))
               - operator[np.ix_(interface, interior)]
               @ np.linalg.inv(operator[np.ix_(interior, interior)]
                               - lam * np.eye(len(interior)))
               @ operator[np.ix_(interior, interface)])
    _record_fixture(out, "shifted_feshbach_pencil",
                    float(np.abs(response - shifted).max()),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"lambda": lam, "resonant": bool(read.resonant),
                     "solve_residual": _finite(read.solveResidual),
                     "determinant_residual": _finite(read.determinantResidual),
                     "reference":
                         "(L_BB - lam I) - L_BI (L_II - lam I)^{-1} L_IB"})
    # ...and the same fixture demonstrating that the STATIC Schur complement
    # does NOT preserve the pencil's nonzero eigenvalue:
    # a nonzero separation is the POINT, so this row records a difference and
    # is exact when the separation is resolved.
    separation = float(np.abs(np.linalg.eigvals(effective)
                              - np.linalg.eigvals(shifted)).max())
    _record_fixture(out, "static_schur_does_not_preserve_the_pencil",
                    0.0 if separation > 1e-6 else 1.0,
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"eigenvalue_separation": separation,
                     "note": "static Schur is NOT the shifted response; a "
                             "resolved separation is the required outcome"})

    # ---- 3. second-quantized subset sums -------------------------------
    import itertools
    spectrum = [0.5 + 0j, 1.25 + 0j, 2.0 + 0j, 3.5 + 0j]
    key = (lambda z: (z.real, z.imag))
    got = sorted((complex(v) for v in
                  cob.OccupationSpectra.subsetSums(spectrum, 2)), key=key)
    reference = sorted((sum(c) for c in itertools.combinations(spectrum, 2)),
                       key=key)
    residual = (max(abs(a - b) for a, b in zip(got, reference))
                if len(got) == len(reference) else float("inf"))
    _record_fixture(out, "second_quantized_subset_sum", residual,
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"particles": 2, "terms": len(got),
                     "reference":
                         "itertools.combinations sums, computed here"})

    # ---- 4. second-quantized hopping block -----------------------------
    block_a = np.diag([1.0, 2.0]).astype(complex)
    block_b = np.diag([0.5]).astype(complex)
    coupling = np.array([[0.3 + 0.1j], [0.2 - 0.05j]])
    flat = cob.OccupationSpectra.hoppingBlock(
        block_a.reshape(-1).tolist(), 2, block_b.reshape(-1).tolist(), 1,
        coupling.reshape(-1).tolist())
    assembled = np.array(flat).reshape(3, 3)
    dense = np.zeros((3, 3), dtype=complex)
    dense[:2, :2] = block_a
    dense[2:, 2:] = block_b
    dense[:2, 2:] = coupling
    dense[2:, :2] = coupling.conj().T
    hopping_residual = float(np.abs(assembled - dense).max())
    eigen = np.linalg.eigvalsh(assembled)
    occupied = sorted((complex(v) for v in cob.OccupationSpectra.subsetSums(
        [complex(v) for v in eigen], 2)), key=key)
    occupied_reference = sorted(
        (sum(c) for c in itertools.combinations(
            [complex(v) for v in eigen], 2)), key=key)
    _record_fixture(out, "second_quantized_hopping",
                    max(hopping_residual,
                        max(abs(a - b) for a, b in
                            zip(occupied, occupied_reference))),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"block_residual": hopping_residual,
                     "reference": "dense block assembly, computed here"})

    # ---- 5. triangle anchor --------------------------------------------
    triangle = T.OrientedTriangle([0, 1, 2], [1, -1, 1])
    anchor = T.ColorAnchor([triangle])
    profile = anchor.evaluate(np.eye(3, dtype=complex), np.ones(3))
    literal_residual = abs(float(profile.score) - 1.0)
    # an abstract rank-three band whose support misses the atlas scores 0
    frame = np.zeros((6, 3), dtype=complex)
    frame[0, 0] = frame[1, 1] = frame[2, 2] = 1.0
    disjoint = T.ColorAnchor([T.OrientedTriangle([3, 4, 5], [1, -1, 1])])
    disjoint_profile = disjoint.evaluate(frame, np.ones(6))
    # post-hoc re-weighting is refused once data have been evaluated
    reweighting_refused = False
    try:
        anchor.declareWeights([1.0])
    except Exception:                                     # noqa: BLE001
        reweighting_refused = True
    empty_atlas_refused = False
    try:
        T.ColorAnchor([])
    except Exception:                                     # noqa: BLE001
        empty_atlas_refused = True
    _record_fixture(out, "triangle_anchor",
                    max(literal_residual, abs(float(disjoint_profile.score)),
                        0.0 if reweighting_refused else 1.0,
                        0.0 if empty_atlas_refused else 1.0),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"literal_triangle_score": _finite(profile.score),
                     "disjoint_atlas_score": _finite(disjoint_profile.score),
                     "calibration_margin": _finite(profile.calibration_margin),
                     "post_hoc_reweighting_refused": reweighting_refused,
                     "empty_atlas_refused": empty_atlas_refused,
                     "reference": "|det A|^2 = 1 exactly at full "
                                  "concentration on the anchoring face"})

    # ---- 6. center branch ----------------------------------------------
    connection = T.FiberConnection()
    to_fiber = _synthetic_unit_fiber(1, 3)
    from_fiber = _synthetic_unit_fiber(11, 3)
    samples = 8
    family = [connection.transport(
        to_fiber, from_fiber,
        np.diag([cmath.exp(2j * math.pi * k / samples), 1.0, 1.0]
                ).astype(complex)) for k in range(samples)]
    omega = cmath.exp(2j * math.pi / 3.0)
    # The section-5 literal-triangle profile is a SINGLE face, so its
    # determinant-phase coherence is unknown and the acceptance predicate
    # refuses it. This centre fixture is an algebraic branch identity, so it
    # declares its own two-face anchor instead of borrowing that one.
    gate = _declared_anchor_gate()
    lifts = [connection.fundamentalLift(family, gate, branch)
             for branch in (0, 1, 2)]
    sectors = {int(lift.centerSector) for lift in lifts}
    base = complex(lifts[0].liftTrace)
    branch_residual = max(
        abs(complex(lifts[s].liftTrace) - base * omega ** (-s))
        for s in (0, 1, 2))
    unitary = np.array(connection.holonomy(family).holonomy)
    adjoint = np.array(T.FiberConnection.adjointRepresentation(unitary))
    adjoint_shifted = np.array(
        T.FiberConnection.adjointRepresentation(omega * unitary))
    adjoint_residual = float(np.abs(adjoint - adjoint_shifted).max())
    fundamental_spread = float(np.abs(
        np.array(T.FiberConnection.projectiveRepresentative(unitary, gate))
        - np.array(T.FiberConnection.projectiveRepresentative(
            omega * unitary, gate))).max())
    _record_fixture(out, "center_branch",
                    max(branch_residual, adjoint_residual,
                        0.0 if len(sectors) == 1 else 1.0,
                        0.0 if fundamental_spread > 1e-6 else 1.0),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"center_sectors": sorted(sectors),
                     "branch_trace_residual": branch_residual,
                     "adjoint_branch_blindness": adjoint_residual,
                     "fundamental_center_spread": fundamental_spread,
                     "reference": "Tr H~(s) = Tr H~(0) * omega^{-s}; "
                                  "Ad(omega U) = Ad(U)"})

    # ---- 6b. closed determinant winding is an integer ------------------
    winding = connection.closedFamilyWinding(family)
    _record_fixture(out, "closed_determinant_winding",
                    (abs(float(winding.winding) - 1.0)
                     if winding.winding is not None else float("inf")),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"winding": winding.winding,
                     "closure": str(winding.windingClosure),
                     "closure_defect": _finite(winding.closureDefect),
                     "reference":
                         "one declared turn of the determinant phase"})

    # ---- 7. Berry cancellation -----------------------------------------
    cells, steps = DECLARED_EXCHANGE_CELLS, DECLARED_EXCHANGE_STEPS
    weights = cmath.exp(1j * DECLARED_BERRY_PHASE) * np.ones(cells,
                                                             dtype=complex)
    single = _translation_frames([0, 4], cells, steps, 4)
    double = _translation_frames([0, 4], cells, 2 * steps, 8)
    raw = complex(EH.loopHolonomy(single, weights).determinant)
    single_character = EH.exchangeCharacter(
        EH.loopHolonomy(single, weights),
        EH.loopHolonomy([single[0]] * len(single), weights))
    double_character = EH.exchangeCharacter(
        EH.loopHolonomy(double, weights),
        EH.loopHolonomy([double[0]] * len(double), weights))
    raw_is_sign = min(abs(raw - 1.0), abs(raw + 1.0)) < 1e-6
    _record_fixture(out, "berry_cancellation",
                    max(abs(complex(single_character.character) + 1.0),
                        abs(complex(double_character.character) - 1.0),
                        1.0 if raw_is_sign else 0.0),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"raw_determinant": _complex_pair(raw),
                     "raw_is_a_sign": raw_is_sign,
                     "single_exchange": _complex_pair(
                         single_character.character),
                     "double_exchange": _complex_pair(
                         double_character.character),
                     "reference": "matched single/double exchange ratios are "
                                  "-1 / +1 while the raw loop carries an "
                                  "arbitrary common Berry phase"})

    # ---- 8. the sharp-spin dichotomy fixtures --------------------------
    jx, jy, jz = _declared_spin_matrices(2)
    sharp = QU.CovarianceState.fromOccupations(np.array([1.0, 0.0]))
    _record_fixture(out, "sharp_spin_half",
                    max(abs(complex(sharp.wickSpinSquaredExpectation(
                            jx, jy, jz).value) - 0.75),
                        abs(complex(sharp.wickSpinSquaredVariance(
                            jx, jy, jz).value))),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"reference": "<J^2> = 3/4 exactly with zero variance"})
    root = 1.0 / math.sqrt(2.0)
    sx = np.array([[0, root, 0], [root, 0, root], [0, root, 0]], dtype=complex)
    sy = np.array([[0, -1j * root, 0], [1j * root, 0, -1j * root],
                   [0, 1j * root, 0]], dtype=complex)
    sz = np.diag([1.0, 0.0, -1.0]).astype(complex)

    def pad(block):
        padded = np.zeros((4, 4), dtype=complex)
        padded[1:, 1:] = block
        return padded

    orbital = np.zeros((4, 1), dtype=complex)
    orbital[0, 0] = math.sqrt(5.0 / 8.0)
    orbital[1, 0] = math.sqrt(3.0 / 8.0)
    generic = QU.CovarianceState.fromSlaterFrame(orbital)
    _record_fixture(out, "generic_slater_is_not_a_sharp_spin",
                    max(abs(complex(generic.wickSpinSquaredExpectation(
                            pad(sx), pad(sy), pad(sz)).value) - 0.75),
                        abs(complex(generic.wickSpinSquaredVariance(
                            pad(sx), pad(sy), pad(sz)).value) - 15.0 / 16.0)),
                    DECLARED_EXACT_TOLERANCE, "exact",
                    {"reference": "<J^2> = 3/4 with Var = 15/16 — the right "
                                  "expectation is NOT a sharp spin "
                                  "certificate"})
    return out


def _declared_spin_matrices(mode_count, offset=0):
    """J_alpha = (+) sigma_alpha / 2 over consecutive mode pairs.

    A DECLARED readout convention, not a derived one. #777 §9 measured that
    Var(J^2) depends on the pairing offset almost entirely, so this is only
    used for the analytic spin fixtures, where the pairing is part of the
    fixture's own statement.
    """
    jx = np.zeros((mode_count, mode_count), dtype=complex)
    jy = np.zeros((mode_count, mode_count), dtype=complex)
    jz = np.zeros((mode_count, mode_count), dtype=complex)
    index = offset
    while index + 1 < mode_count:
        a, b = index, index + 1
        jx[a, b] = jx[b, a] = 0.5
        jy[a, b] = -0.5j
        jy[b, a] = 0.5j
        jz[a, a] = 0.5
        jz[b, b] = -0.5
        index += 2
    return jx, jy, jz


# =====================================================================
# the run
# =====================================================================

def make_config(size=DECLARED_SIZE_FAST, seed=DECLARED_SEED,
                host_seed=DECLARED_HOST_SEED,
                drive_steps=DECLARED_DRIVE_STEPS_FAST,
                submode="strict", refine=True, fock_oracle=False,
                degrees=DECLARED_DEGREES,
                resolution_scan=DECLARED_RESOLUTION_SCAN,
                analysis_resolution=DECLARED_ANALYSIS_RESOLUTION):
    return {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "size": int(size),
        "seed": int(seed),
        "host_seed": int(host_seed),
        "drive_steps": int(drive_steps),
        "emergence_submode": submode,
        "refine": bool(refine),
        "fock_oracle": bool(fock_oracle),
        "degrees": list(degrees),
        "register_degrees": list(DECLARED_REGISTER_DEGREES),
        "resolution_scan": list(resolution_scan),
        "analysis_resolution": float(analysis_resolution),
        "candidate_moves": DECLARED_CANDIDATE_MOVES,
        "stage2_iters": DECLARED_STAGE2_ITERS,
        "refinement_thresholds": dict(DECLARED_REFINEMENT_THRESHOLDS),
        "refinement_cells": DECLARED_REFINEMENT_CELLS,
        "shift_fractions": list(DECLARED_SHIFT_FRACTIONS),
        "window_half_width_fraction": DECLARED_WINDOW_HALF_WIDTH_FRACTION,
        "amls_mode_cutoff": DECLARED_AMLS_MODE_CUTOFF,
        "sigmas": list(DECLARED_SIGMAS),
        "krylov_dim": DECLARED_KRYLOV_DIM,
        "exact_tolerance": DECLARED_EXACT_TOLERANCE,
        "replay_tolerance": DECLARED_REPLAY_TOLERANCE,
        "json_matrix_limit": DECLARED_JSON_MATRIX_LIMIT,
    }


def config_hash_of(config):
    return hashlib.md5(canonical_json(config).encode("utf-8")).hexdigest()


def _apply_refinement_thresholds(node, config):
    thresholds = MC.RefinementIndicators()
    declared = config["refinement_thresholds"]
    # Upper bounds crossed from BELOW; infinity = never fires.
    for name in ("regge_stationarity_residual", "hodge_stationarity_residual",
                 "curvature_concentration", "solver_error"):
        value = declared[name]
        setattr(thresholds, name,
                float("inf") if value is None else float(value))
    # A LOWER bound crossed from above; 0 = never fires.
    value = declared["mesh_quality"]
    thresholds.mesh_quality = 0.0 if value is None else float(value)
    node.set_refinement_thresholds(thresholds)


def _analysis_config(degrees, resolutions, cadence=1, cold=False, fock=False):
    config = MC.AnalysisConfig()
    config.enabled = True
    config.cadence = cadence
    config.degrees = list(degrees)
    config.resolutions = list(resolutions)
    config.cold_caches = cold
    config.fock_oracle = fock
    return config


def run_simulation(config, commit=None, sidecar_path=None, progress=False):
    """One complete simulation: neutral host, emergence drive, recursive
    analysis, verdict. Returns the versioned run document."""
    started = time.time()
    commit = current_commit() if commit is None else commit
    config_hash = config_hash_of(config)
    mean_field = config["emergence_submode"] == "certificates-blind-mean-field"
    submode = (MC.EmergenceSubmode.CERTIFICATES_BLIND_MEAN_FIELD if mean_field
               else MC.EmergenceSubmode.STRICT)

    host = build_neutral_host(config["size"], config["host_seed"])
    host_cells = len(host.getTopSimplices())
    # The node's REGISTER degrees are the objective's domain (>= 1); the
    # analysis degrees below are a separate, post-hoc knob.
    node = MC(host, [], [], list(config["register_degrees"]), 1.0,
              config["seed"])
    node.set_objective(cob.JointStationarityObjective())
    node.set_simulation_mode(MC.SimulationMode.EMERGENCE, submode)
    node.set_provenance(config_hash, commit)
    node.set_analysis_config(_analysis_config(
        config["degrees"], [config["analysis_resolution"]],
        fock=config["fock_oracle"]))
    _apply_refinement_thresholds(node, config)

    steps = []
    checkpoints = []
    trace = []
    drive_started = time.time()
    for step in range(config["drive_steps"]):
        step_started = time.time()
        stage1 = list(node.run_stage1(
            max_steps=1, n_candidate_moves=config["candidate_moves"]))
        stage2 = list(node.run_stage2(max_iters=config["stage2_iters"]))
        trace.extend(stage1)
        trace.extend(stage2)
        indicators = node.refinement_indicators()
        decision = node.refinement_decision()
        refined = 0
        if config["refine"] and decision.refine:
            refined = int(node.refine_geometry(config["refinement_cells"]))
        node.run_recursive_analysis()
        checkpoint = json.loads(node.checkpoint_json)
        checkpoints.append(checkpoint)
        steps.append({
            "step": step,
            "stage1_updates": len(stage1),
            "stage2_iterations": len(stage2),
            "lookahead_depth": int(node.last_stage1_lookahead),
            "stage2_stationary": bool(node.last_stage2_stationary),
            "objective": _finite(checkpoint["objective"]["total"]),
            "regge_stationarity": _finite(
                checkpoint["objective"]["regge_stationarity"]),
            "hodge_stationarity": _finite(
                checkpoint["objective"]["hodge_stationarity"]),
            "carried_state_energy": _finite(
                checkpoint["objective"]["carried_state_energy"]),
            "indicators": {
                "regge_stationarity_residual":
                    _finite(indicators.regge_stationarity_residual),
                "hodge_stationarity_residual":
                    _finite(indicators.hodge_stationarity_residual),
                "curvature_concentration":
                    _finite(indicators.curvature_concentration),
                "mesh_quality": _finite(indicators.mesh_quality),
                "solver_error": _finite(indicators.solver_error),
            },
            "refinement": {
                "enabled": bool(config["refine"]),
                "decision": bool(decision.refine),
                "trigger": decision.trigger or None,
                "cells_committed": refined,
            },
            "cells": len(node.st.getTopSimplices()),
            "edges": len(node.st.getEdgeList().toVector()),
            "cache": checkpoint["analysis"],
            "seconds": time.time() - step_started,
        })
        if progress:
            print(f"  step {step}: F={steps[-1]['objective']!r} "
                  f"cells={steps[-1]['cells']} "
                  f"refine={steps[-1]['refinement']['decision']} "
                  f"{steps[-1]['seconds']:.2f}s", flush=True)
    drive_seconds = time.time() - drive_started

    # The SCAN pass on the same relaxed geometry: with a single resolution
    # the overlay's persistence lifetime is identically 1, so `persistence`
    # is the first failing certificate for STRUCTURAL reasons (#777 §4). Both
    # passes are recorded so that cannot be mistaken for a physical result.
    node.set_analysis_config(_analysis_config(
        config["degrees"], config["resolution_scan"],
        fock=config["fock_oracle"]))
    node.run_recursive_analysis()
    scan_checkpoint = json.loads(node.checkpoint_json)
    node.set_analysis_config(_analysis_config(
        config["degrees"], [config["analysis_resolution"]],
        fock=config["fock_oracle"]))
    analysis_started = time.time()
    node.run_recursive_analysis()
    analysis_seconds = time.time() - analysis_started
    checkpoint = json.loads(node.checkpoint_json)
    # The final pass replaces the last drive step's (same geometry, one more
    # pass). With no drive step at all — a bare analysis of the neutral host —
    # it IS the only frame.
    if checkpoints:
        checkpoints[-1] = checkpoint
    else:
        checkpoints.append(checkpoint)
    spacetime = node.st

    readout_started = time.time()
    readout = RecursiveReadout(spacetime, config, config["seed"])
    readout_seconds = time.time() - readout_started

    sidecar = MatrixSidecar(sidecar_path or "recursive_baryon_sidecar.npz")
    document = {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": "recursive_baryon_simulation",
        "provenance": {
            "seed": config["seed"],
            "host_seed": config["host_seed"],
            "config_hash": config_hash,
            "commit": commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(started)),
            "threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "config": config,
        "quantity_classes": QUANTITY_CLASSES,
        "host": {
            "kind": "neutral_closed_s4",
            "description": (
                "the bare boundary of a 5-simplex (a combinatorial closed "
                "S^4) refined by n PreGeometric stellar Pachner adds at a "
                "fixed host seed, then given a mild deterministic "
                "non-uniform metric; no hole, no colour window, no pinned "
                "carrier, no boundary block, no target register"),
            "n_refine": config["size"],
            "initial_cells": host_cells,
            "cells": len(spacetime.getTopSimplices()),
            "edges": len(spacetime.getEdgeList().toVector()),
            "vertices": len(spacetime.getVertexList().toVector()),
        },
        "drive": {
            "mode": checkpoint["mode"],
            "emergence_submode": checkpoint["emergence_submode"],
            "objective_name": cob.ObjectiveName.JOINT_STATIONARITY,
            "steps": steps,
            "objective_trace": [_finite(v) for v in trace],
            "refinement_events": sum(s["refinement"]["cells_committed"]
                                     for s in steps),
            "refinement_thresholds": config["refinement_thresholds"],
            "refinement_never_fires": sorted(
                name for name, value in
                config["refinement_thresholds"].items() if value is None),
            "firewall": {
                "objective_terms": list(MC.objective_term_names()),
                "refinement_indicators": list(
                    MC.refinement_indicator_names()),
                "carried_state_energy": _finite(
                    checkpoint["objective"]["carried_state_energy"]),
                "carried_state_energy_weight": _finite(
                    checkpoint["objective"]["carried_state_energy_weight"]),
                "statement": (
                    "objectiveOf and refinementDecisionOf are STATIC over "
                    "their records, so neither can reach an analysis member; "
                    "the only channel from the carried state to the geometry "
                    "is carried_state_energy, identically zero outside the "
                    "labeled certificates_blind_mean_field sub-mode"),
                "carried_state_present": bool(
                    checkpoint["edge_quantum_data"]["carried_state_present"]),
                "mean_field_note": (
                    "this driver adopts NO carried state, so even under the "
                    "certificates_blind_mean_field sub-mode the one permitted "
                    "coupling is exactly zero; #776 owns the mean-field "
                    "schedule and this run does not exercise it"
                    if (config["emergence_submode"]
                    == "certificates-blind-mean-field")
                    else "the strict sub-mode zeroes the coupling weight by "
                         "construction"),
            },
        },
        "checkpoints": checkpoints,
        "checkpoint": checkpoint,
        "scan_checkpoint": scan_checkpoint,
        "raw_geometry": raw_geometry_block(spacetime),
        "edge_mode_data": edge_mode_block(readout, checkpoint),
        "hierarchy": hierarchy_block(readout, checkpoint),
        "response_hierarchy": response_hierarchy_block(readout, config),
        "fibers": fibers_block(readout, sidecar),
        "transports": transports_block(readout),
        "covariance": covariance_block(readout, checkpoint, sidecar),
        "fock": fock_block(readout, checkpoint, config),
        "statistics": statistics_block(readout),
        "particles": particles_block(readout, checkpoint),
        "particles_resolution_scan": scan_particles_block(scan_checkpoint),
        "spectral_dimension": spectral_dimension_block(spacetime, config),
        "exactness": exactness_fixtures(),
        "reproducibility": {
            "statement": (
                "the engine's move draw is NOT process-deterministic past "
                "the first committed move (#579, re-measured in #776), so a "
                "fresh rebuild from (config, seed, commit) reproduces the "
                "first committed move and the whole relaxation but not a "
                "longer trajectory; the schema-4 CHECKPOINT is the faithful "
                "record and `replay` replays it"),
            "deterministic_unit": "one stage-1 update plus one stage-2 "
                                  "relaxation",
            "drive_steps": config["drive_steps"],
        },
    }
    document["verdict"] = verdict_block(readout)
    document["certificates"] = certificates_block(document)
    sidecar_descriptor = sidecar.write()
    document["sidecar"] = sidecar_descriptor
    document["runtime"] = {
        "wall_seconds": time.time() - started,
        "drive_seconds": drive_seconds,
        "analysis_seconds": analysis_seconds,
        "readout_seconds": readout_seconds,
        "rss_bytes": _rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    document["content_hashes"] = content_hashes_of(document)
    return document


def hierarchy_block(readout, checkpoint):
    """The persistent component hierarchy at the analysis resolution, plus
    the declared resolution scan and its persistence tracks."""
    def slice_of(slice_read):
        return {
            "gamma": _finite(slice_read.gamma),
            "q": _finite(slice_read.objectiveValue),
            "q_complex": [_finite(slice_read.q.real),
                          _finite(slice_read.q.imag)],
            "levels": int(slice_read.levels),
            "restart_spread": _finite(slice_read.restartSpread),
            "component_count": len(slice_read.components),
            "hierarchy_sizes": [len(level) for level in slice_read.hierarchy],
            "components": [
                {
                    "id": component.id.canonicalHash(),
                    "level": int(component.id.level()),
                    "volume": len(component.support),
                    "support": [int(v) for v in component.support],
                    "strength": _finite(component.strength),
                    "conductance": _finite(component.conductance),
                    "internal_weight": _finite(component.internalWeight),
                    "modularity_contribution":
                        [_finite(component.modularityContribution.real),
                         _finite(component.modularityContribution.imag)],
                }
                for component in slice_read.components],
        }

    return {
        "analysis_resolution": readout.config["analysis_resolution"],
        "analysis_slice": (slice_of(readout.slice) if readout.slice else None),
        "next_level_components": len(readout.next_level),
        "resolution_scan": [slice_of(s) for s in readout.scan_report.slices],
        "tracks": [
            {
                "lifetime": int(track.lastSlice - track.firstSlice + 1),
                "min_adjacent_overlap": _finite(track.minAdjacentOverlap),
                "mean_conductance": _finite(track.meanConductance),
                "members": len(track.members),
            }
            for track in readout.scan_report.tracks],
        "max_depth": max((int(s.levels) for s in readout.scan_report.slices),
                         default=0),
        "checkpoint_hierarchy_slices": len(checkpoint.get("hierarchy", [])),
        "note": ("component discovery is HEURISTIC (modularity), and #776 "
                 "measured that the canonical component hash is not fully "
                 "label-free, so nothing here is keyed on that hash across a "
                 "relabeling"),
    }


def certificates_block(document):
    """Every certificate the run reported, with its grade and status — the
    one place a reader can see what did and did not certify.

    Three statuses, deliberately distinguished: ``holds`` (evaluated and
    certified), ``refused`` (out of domain or unsupplied evidence, with the
    reason named — a correct refusal, not a failure), and ``failed``
    (evaluated and did not certify).
    """
    entries = []

    def add(name, grade, holds, residual=None, reason=None, refused=False):
        entries.append({
            "name": name, "grade": grade, "holds": bool(holds),
            "status": "holds" if holds else ("refused" if refused
                                             else "failed"),
            "residual": _finite(residual), "reason": reason})

    response = document["response_hierarchy"]
    if response.get("static"):
        add("static-reduction", response["static"]["certificate_grade"],
            response["static"]["certificate_holds"],
            response["static"]["solve_residual"])
    for window in response.get("shifted", []):
        if "certificate_grade" in window:
            add(f"shifted-response@{window['fraction']}",
                window["certificate_grade"], window["certificate_holds"],
                window["solve_residual"])
    amls = response.get("amls") or {}
    if amls.get("available"):
        add("amls-craig-bampton", amls["certificate_grade"],
            amls["certificate_holds"], amls["max_eigen_residual"])
    else:
        add("amls-craig-bampton", None, False, None, amls.get("reason"),
            refused=True)
    network = response.get("response_network") or {}
    if network.get("certificate_grade"):
        add("response-network", network["certificate_grade"],
            network["certificate_holds"], network["coverage_residual"])
    else:
        add("response-network", None, False, None, network.get("reason"),
            refused=True)
    realization = response.get("realization") or {}
    if realization.get("certificate_grade"):
        add("sheaf-realization", realization["certificate_grade"],
            realization["emitted"], realization["reconstruction_residual"],
            None if realization["emitted"] else
            "not emitted: a cellular-sheaf Laplacian is self-adjoint and the "
            "regime is non-normal, so the general response network is "
            "correctly retained",
            refused=not realization["emitted"])
    for entry in response.get("labeled_fiber_sums", []):
        if entry.get("certificate_grade"):
            add(f"labeled-fiber-sum@k={entry['degree']}",
                entry["certificate_grade"], entry["certificate_holds"],
                entry["gram_defect"])

    covariance = document["covariance"]
    add("covariance-purity", "certified-numerical",
        covariance["purity_defect_max"] is not None
        and covariance["purity_defect_max"] < 1e-9,
        covariance["purity_defect_max"],
        None if covariance["state_count"] else "no accepted band state",
        refused=not covariance["state_count"])
    add("inductive-embedding", "exact",
        covariance["vacuum_embedding_defect_max"] is not None
        and covariance["vacuum_embedding_defect_max"] < 1e-12,
        covariance["vacuum_embedding_defect_max"])

    transports = document["transports"]
    add("derived-transport", "certified-numerical",
        transports["accepted"] > 0, transports["leakage_min"],
        None if transports["accepted"] else
        "; ".join(sorted(k for k in transports["rejection_reasons"]
                         if k != "None")) or "no derived transport")
    for channel in ("full", "determinant", "projective", "center", "winding"):
        entry = transports.get(channel) or {}
        add(f"holonomy-{channel}", "exact" if entry.get("available") else None,
            bool(entry.get("available")), None,
            None if entry.get("available") else entry.get("reason"),
            refused=not entry.get("available"))

    for fixture in document["exactness"]:
        add(f"fixture:{fixture['name']}", fixture["grade"], fixture["exact"],
            fixture["residual"])

    for name in document["verdict"]["failed_certificates"]:
        add(f"baryon:{name}", "heuristic-discovery", False, None,
            "named as failed or missing by ParticleClusters::classifyBaryon",
            refused=True)

    return {
        "entries": entries,
        "held": sum(1 for e in entries if e["status"] == "holds"),
        "refused": sum(1 for e in entries if e["status"] == "refused"),
        "failed_count": sum(1 for e in entries if e["status"] == "failed"),
        "total": len(entries),
        "failed": [e["name"] for e in entries if not e["holds"]],
        "status_note": (
            "`refused` means out of domain or evidence unsupplied, with the "
            "reason named — a correct refusal, not a failure; `failed` means "
            "evaluated and not certified"),
    }


#: The blocks a replay must reproduce byte-for-byte through their content
#: hash. `runtime`, `provenance` and the sidecar descriptor are deliberately
#: excluded: a wall time is not a verdict.
HASHED_BLOCKS = (
    "raw_geometry", "edge_mode_data", "hierarchy", "response_hierarchy",
    "fibers", "transports", "covariance", "fock", "statistics", "particles",
    "particles_resolution_scan", "spectral_dimension", "exactness",
    "verdict", "certificates", "checkpoint", "scan_checkpoint",
)


def content_hashes_of(document):
    return {name: content_hash(document[name]) for name in HASHED_BLOCKS
            if name in document}


# =====================================================================
# replay
# =====================================================================

def _worst_relative_difference(left, right, path=""):
    """(worst relative difference, where) over two JSON documents.

    A DISCRETE mismatch (a differing string, bool, int, key set or length)
    returns infinity: those must be identical.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf"), f"{path}: key sets differ"
        worst, where = 0.0, ""
        for key in sorted(left):
            value, spot = _worst_relative_difference(
                left[key], right[key], f"{path}.{key}")
            if value > worst:
                worst, where = value, spot
        return worst, where
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf"), f"{path}: lengths differ"
        worst, where = 0.0, ""
        for index, (a, b) in enumerate(zip(left, right)):
            value, spot = _worst_relative_difference(a, b, f"{path}[{index}]")
            if value > worst:
                worst, where = value, spot
        return worst, where
    if isinstance(left, bool) or isinstance(right, bool):
        return (0.0, "") if left == right else (float("inf"), path)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if left == right:
            return 0.0, ""
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return abs(float(left) - float(right)) / scale, path
    return (0.0, "") if left == right else (float("inf"), path)


#: The checkpoint blocks a cold replay must reproduce. #776 measured these
#: byte-identical from any optimizer-produced complex; every one is a pure
#: function of the accepted geometry.
REPLAY_COMPARED_BLOCKS = (
    "schema_version", "emergence_submode", "raw_complex", "edge_quantum_data",
    "objective", "hierarchy", "fibers", "labeled_fiber_sums", "transports",
    "covariance", "fock_oracle", "particles", "certificates",
)

#: The blocks a replay is NOT expected to reproduce, each with the reason it
#: is a property of the REPLAYING PROCESS rather than of the geometry.
REPLAY_EXCLUDED_BLOCKS = {
    "mode": "the replayed document is stamped \"replay\" by design",
    "geometry_revision":
        "the metric revision key counts THIS process's metric writes",
    "refinement":
        "solver_error is the magnitude of the last accepted stage-2 "
        "improvement, and a replay never relaxed",
    "analysis":
        "the pass and cache counters are this process's, and a replay runs "
        "with every cache cold by design",
    "provenance": "the replay stamps its own provenance",
    "invalidated_ancestry":
        "invalidation is relative to the accepted move a replay did not make",
}


def discrete_verdicts(checkpoint):
    """Every DISCRETE read of a checkpoint — what a replay must reproduce
    exactly, with no tolerance at all."""
    return {
        "mode": checkpoint["mode"],
        "emergence_submode": checkpoint["emergence_submode"],
        "raw_complex": checkpoint["raw_complex"],
        "component_ids": [[c["id"] for c in s["components"]]
                          for s in checkpoint["hierarchy"]],
        "component_supports": [[c["support"] for c in s["components"]]
                               for s in checkpoint["hierarchy"]],
        "band_ranks": [f["rank"] for f in checkpoint["fibers"]],
        "band_accepted": [f["accepted"] for f in checkpoint["fibers"]],
        "labeled_sum_ranks": [[s["nominal_rank"], s["effective_rank"]]
                              for s in checkpoint["labeled_fiber_sums"]],
        "transport_accepted": [t["accepted"]
                               for t in checkpoint["transports"]],
        "transport_ranks": [t["numerical_rank"]
                            for t in checkpoint["transports"]],
        "quark_classifications": [q["classification"]
                                  for q in checkpoint["particles"]["quarks"]],
        "quark_failed": [q["failed_certificates"]
                         for q in checkpoint["particles"]["quarks"]],
        "bound_supercomponent_found": [
            b["found"]
            for b in checkpoint["particles"]["bound_supercomponents"]],
        "baryon_classifications": [
            b["classification"] for b in checkpoint["particles"]["baryons"]],
        "active_modes": checkpoint["covariance"]["active_modes"],
        "fock_present": checkpoint["fock_oracle"]["present"],
    }


def replay_document(document, directory=None, progress=False):
    """Cold-cache replay: reproduce every stored verdict and content hash.

    Every checkpoint is rebuilt through ``MultiCobordism.replay_checkpoint``,
    which disables every cache and recomputes every derived hierarchy and
    certificate. Every DISCRETE verdict must match exactly; continuous
    aggregates get the declared ``replay_tolerance`` and the measured
    difference is always reported, so a real divergence could never hide.
    """
    started = time.time()
    version = document.get("schema_version")
    if version != RUN_SCHEMA_VERSION:
        raise ValueError(
            f"unknown run schema_version {version!r}: this build writes and "
            f"accepts {RUN_SCHEMA_VERSION}")
    directory = directory or "."
    config = document["config"]
    tolerance = config.get("replay_tolerance", DECLARED_REPLAY_TOLERANCE)

    frames = []
    for index, checkpoint in enumerate(document["checkpoints"]):
        stored = json.dumps(checkpoint)
        checkpoint_version = MC.checkpoint_version_of(stored)
        replayed = json.loads(MC.replay_checkpoint(stored))
        stored_verdicts = discrete_verdicts(checkpoint)
        replayed_verdicts = discrete_verdicts(replayed)
        # `mode` is stamped "replay" on the replayed document by design.
        stored_verdicts.pop("mode")
        replayed_verdicts.pop("mode")
        discrete_ok = stored_verdicts == replayed_verdicts
        comparable = {k: checkpoint[k] for k in REPLAY_COMPARED_BLOCKS
                      if k in checkpoint}
        comparable_replayed = {k: replayed[k] for k in REPLAY_COMPARED_BLOCKS
                               if k in replayed}
        worst, where = _worst_relative_difference(comparable,
                                                  comparable_replayed)
        # What was NOT compared, and why — a reader must be able to see the
        # exclusions rather than infer them from a silence.
        excluded = {}
        for key, reason in REPLAY_EXCLUDED_BLOCKS.items():
            if key not in checkpoint:
                continue
            excluded[key] = {
                "reason": reason,
                "identical": checkpoint[key] == replayed.get(key),
            }
        frames.append({
            "frame": index,
            "checkpoint_schema_version": checkpoint_version,
            "discrete_verdicts_identical": discrete_ok,
            "byte_identical": comparable == comparable_replayed,
            "compared_blocks": list(comparable),
            "excluded_blocks": excluded,
            "worst_relative_difference": _finite(worst),
            "worst_at": where or None,
            "within_tolerance": bool(worst <= tolerance),
            "cold_caches": replayed["analysis"]["cold_caches"],
        })
        if progress:
            print(f"  frame {index}: discrete={discrete_ok} "
                  f"worst={worst:.3g} at {where or '-'}", flush=True)

    # the content hashes of every persisted block
    recomputed = content_hashes_of(document)
    hashes = []
    for name, stored_hash in sorted(document["content_hashes"].items()):
        again = recomputed.get(name)
        hashes.append({"block": name, "stored": stored_hash,
                       "recomputed": again, "match": stored_hash == again})

    # the sidecar's own bytes, and every matrix descriptor's content hash
    sidecar_report = {"present": document.get("sidecar") is not None,
                      "arrays": [], "file_hash_match": None}
    if document.get("sidecar"):
        path = os.path.join(directory,
                            os.path.basename(document["sidecar"]["file"]))
        if os.path.exists(path):
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            sidecar_report["file_hash_match"] = (
                digest == document["sidecar"]["sha256"])
        else:
            sidecar_report["file_hash_match"] = False
            sidecar_report["reason"] = f"sidecar file missing: {path}"
    for descriptor in _matrix_descriptors(document):
        _matrix, ok, reason = MatrixSidecar.load(descriptor, directory)
        sidecar_report["arrays"].append({"name": descriptor["name"],
                                         "storage": descriptor["storage"],
                                         "verified": ok, "reason": reason})

    # the exactness fixtures, recomputed cold
    fresh = {f["name"]: f for f in exactness_fixtures()}
    fixtures = []
    for stored in document["exactness"]:
        again = fresh.get(stored["name"])
        fixtures.append({
            "name": stored["name"],
            "stored_residual": stored["residual"],
            "replayed_residual": again["residual"] if again else None,
            "both_exact": bool(stored["exact"] and again and again["exact"]),
        })

    # the verdict itself, recomputed from the replayed final checkpoint's
    # complex — the strongest statement the replay can make
    verdict = {"stored": document["verdict"]["verdict"],
               "replayed": None, "match": None, "reason": None}
    try:
        rebuilt = _spacetime_from_raw(document["raw_geometry"])
        readout = RecursiveReadout(rebuilt, config, config["seed"])
        recomputed = verdict_block(readout)
        verdict["replayed"] = recomputed["verdict"]
        verdict["match"] = verdict["replayed"] == verdict["stored"]
        verdict["failed_certificates_match"] = (
            list(recomputed["failed_certificates"])
            == list(document["verdict"]["failed_certificates"]))
    except Exception as error:                            # noqa: BLE001
        verdict["reason"] = f"{type(error).__name__}: {error}"
        verdict["match"] = False

    report = {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": "recursive_baryon_replay",
        "source_config_hash": document["provenance"]["config_hash"],
        "source_commit": document["provenance"]["commit"],
        "replay_commit": current_commit(),
        "tolerance": tolerance,
        "frames": frames,
        "frames_discrete_identical": sum(
            1 for f in frames if f["discrete_verdicts_identical"]),
        "frames_byte_identical": sum(1 for f in frames if f["byte_identical"]),
        "frames_total": len(frames),
        "content_hashes": hashes,
        "content_hashes_matched": sum(1 for h in hashes if h["match"]),
        "sidecar": sidecar_report,
        "exactness": fixtures,
        "verdict": verdict,
        "wall_seconds": time.time() - started,
    }
    report["verified"] = bool(
        all(f["discrete_verdicts_identical"] and f["within_tolerance"]
            for f in frames)
        and all(h["match"] for h in hashes)
        and all(a["verified"] for a in sidecar_report["arrays"])
        and (sidecar_report["file_hash_match"] is not False)
        and all(f["both_exact"] for f in fixtures)
        and verdict["match"])
    return report


def _matrix_descriptors(document):
    """Every matrix descriptor the run persisted, wherever it lives."""
    out = []
    for entry in document["covariance"]["states"]:
        if entry.get("gamma"):
            out.append(entry["gamma"])
    for read in document["fibers"]["reads"]:
        for band in read["bands"]:
            if band.get("projector"):
                out.append(band["projector"])
    return out


def _spacetime_from_raw(raw):
    """Rebuild the complex from the persisted raw geometry.

    Uses the SAME entry point the checkpoint replay uses,
    ``Spacetime.fromCells`` — the faithful rebuild route (#579).
    """
    cells = [[int(v) for v in cell] for cell in raw["cells"]]
    spacetime = T.Spacetime.fromCells(int(raw["dimensions"]), cells)
    lengths = {(int(e["a"]), int(e["b"])): complex(e["length"][0],
                                                  e["length"][1])
               for e in raw["edges"]}
    for edge in spacetime.getEdgeList().toVector():
        a, b = edge.getSource().getId(), edge.getTarget().getId()
        key = (min(a, b), max(a, b))
        if key in lengths:
            edge.setLength(lengths[key])
    return spacetime


# =====================================================================
# the headless campaign
# =====================================================================

def run_campaign(sizes, seeds, base_config, out_dir=None, progress=False):
    """A headless size campaign and its aggregate scaling report.

    At least three sizes. Every member is run and
    RECORDED, including one that fails: a silently dropped seed would make
    the aggregate a selection.
    """
    import shutil
    import tempfile

    started = time.time()
    commit = current_commit()
    members = []
    # Without --member-dir the member documents are not kept, so their
    # sidecars are scratch: they go to a temporary directory that is removed,
    # rather than littering the working directory with orphans.
    scratch = None if out_dir else tempfile.mkdtemp(prefix="rbs-campaign-")
    for size in sizes:
        for seed in seeds:
            config = dict(base_config)
            config["size"] = int(size)
            config["seed"] = int(seed)
            member_started = time.time()
            rss_before = _rss_bytes()
            record = {"size": int(size), "seed": int(seed)}
            sidecar = os.path.join(out_dir or scratch,
                                   f"sidecar_{size}_{seed}.npz")
            try:
                document = run_simulation(config, commit=commit,
                                          sidecar_path=sidecar)
            except Exception as error:                    # noqa: BLE001
                record.update({
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "wall_seconds": time.time() - member_started,
                })
                members.append(record)
                if progress:
                    print(f"  size {size} seed {seed}: FAILED "
                          f"{record['error']}", flush=True)
                continue
            # Cache statistics over EVERY analysis pass the member ran, not
            # just the last: the last pass is warm by construction (two
            # passes already ran on the same complex), so quoting it alone
            # would report a hit rate the run did not pay for.
            cache = document["checkpoint"]["analysis"]
            passes = [step["cache"] for step in document["drive"]["steps"]]
            passes.append(document["scan_checkpoint"]["analysis"])
            passes.append(cache)
            record.update({
                "ok": True,
                "config_hash": document["provenance"]["config_hash"],
                "cells": document["host"]["cells"],
                "edges": document["host"]["edges"],
                "vertices": document["host"]["vertices"],
                "verdict": document["verdict"]["verdict"],
                "first_failing_certificate":
                    document["verdict"]["first_failing_certificate"],
                "failed_certificates":
                    document["verdict"]["failed_certificates"],
                "components": (document["hierarchy"]["analysis_slice"]
                               ["component_count"]
                               if document["hierarchy"]["analysis_slice"]
                               else None),
                "bands": document["fibers"]["total"],
                "accepted_bands": document["fibers"]["accepted"],
                "rank_three_accepted": document["fibers"]
                                               ["rank_three_accepted"],
                "derived_transports": document["transports"]["derived"],
                "accepted_transports": document["transports"]["accepted"],
                "transport_leakage_min": document["transports"]["leakage_min"],
                "certified_quarks": document["particles"]["certified_quarks"],
                "quark_reads": document["particles"]["quark_reads"],
                "peak_spectral_dimension":
                    document["spectral_dimension"].get("peak"),
                "purity_defect_max": document["covariance"]
                                             ["purity_defect_max"],
                "wall_seconds": document["runtime"]["wall_seconds"],
                "drive_seconds": document["runtime"]["drive_seconds"],
                "analysis_seconds": document["runtime"]["analysis_seconds"],
                "readout_seconds": document["runtime"]["readout_seconds"],
                "rss_bytes": document["runtime"]["rss_bytes"],
                "rss_delta_bytes": (
                    (document["runtime"]["rss_bytes"] - rss_before)
                    if (document["runtime"]["rss_bytes"] is not None
                        and rss_before is not None) else None),
                "peak_rss_bytes": document["runtime"]["peak_rss_bytes"],
                "analysis_passes": len(passes),
                "cache_hits": sum(p["cache_hits"] for p in passes),
                "cache_misses": sum(p["cache_misses"] for p in passes),
                "cache_invalidations": sum(p["cache_invalidations"]
                                           for p in passes),
                "cache_entries": cache["cache_entries"],
                "cache_hits_final_pass": cache["cache_hits"],
                "cache_misses_final_pass": cache["cache_misses"],
                "exactness_all_exact": all(f["exact"]
                                           for f in document["exactness"]),
            })
            if out_dir:
                path = os.path.join(out_dir, f"member_{size}_{seed}.json")
                with open(path, "w") as handle:
                    json.dump(document, handle)
                record["document"] = os.path.basename(path)
            members.append(record)
            if progress:
                print(f"  size {size} seed {seed}: {record['verdict']!r} "
                      f"cells={record['cells']} "
                      f"{record['wall_seconds']:.2f}s", flush=True)
    if scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": "recursive_baryon_campaign",
        "commit": commit,
        "sizes": [int(s) for s in sizes],
        "seeds": [int(s) for s in seeds],
        "base_config": base_config,
        "members": members,
        "members_total": len(members),
        "members_ok": sum(1 for m in members if m["ok"]),
        "members_failed": [m for m in members if not m["ok"]],
        "member_documents": bool(out_dir),
        "member_documents_note": (
            "each member's full run document and its sidecar are written to "
            "--member-dir; without it a member's matrices are scratch and are "
            "removed, so only the aggregate below survives"),
        "aggregate": aggregate_campaign(members),
        "wall_seconds": time.time() - started,
    }


def aggregate_campaign(members):
    """The scaling report: runtime, memory, cache statistics and verdicts,
    per size, with a log-log scaling exponent where three sizes exist."""
    ok = [m for m in members if m["ok"]]
    by_size = {}
    for member in ok:
        by_size.setdefault(member["size"], []).append(member)
    rows = []
    for size in sorted(by_size):
        group = by_size[size]
        cache_hits = sum(m["cache_hits"] for m in group)
        cache_total = cache_hits + sum(m["cache_misses"] for m in group)
        rows.append({
            "size": size,
            "members": len(group),
            "cells": mean_sd(m["cells"] for m in group),
            "edges": mean_sd(m["edges"] for m in group),
            "wall_seconds": mean_sd(m["wall_seconds"] for m in group),
            "drive_seconds": mean_sd(m["drive_seconds"] for m in group),
            "analysis_seconds": mean_sd(m["analysis_seconds"] for m in group),
            "readout_seconds": mean_sd(m["readout_seconds"] for m in group),
            "rss_bytes": mean_sd(m["rss_bytes"] for m in group),
            "rss_delta_bytes": mean_sd(m["rss_delta_bytes"] for m in group),
            "peak_rss_bytes": mean_sd(m["peak_rss_bytes"] for m in group),
            "analysis_passes": mean_sd(m["analysis_passes"] for m in group),
            "cache_hits": mean_sd(m["cache_hits"] for m in group),
            "cache_hits_final_pass": mean_sd(m["cache_hits_final_pass"]
                                             for m in group),
            "cache_misses": mean_sd(m["cache_misses"] for m in group),
            "cache_invalidations": mean_sd(m["cache_invalidations"]
                                           for m in group),
            "cache_entries": mean_sd(m["cache_entries"] for m in group),
            "cache_hit_fraction": (cache_hits / cache_total
                                   if cache_total else None),
            "components": mean_sd(m["components"] for m in group),
            "bands": mean_sd(m["bands"] for m in group),
            "accepted_bands": mean_sd(m["accepted_bands"] for m in group),
            "rank_three_accepted": mean_sd(m["rank_three_accepted"]
                                           for m in group),
            "accepted_transports": mean_sd(m["accepted_transports"]
                                           for m in group),
            "transport_leakage_min": mean_sd(m["transport_leakage_min"]
                                             for m in group),
            "certified_quarks": mean_sd(m["certified_quarks"] for m in group),
            "peak_spectral_dimension": mean_sd(m["peak_spectral_dimension"]
                                               for m in group),
            "verdicts": _histogram(m["verdict"] for m in group),
        })

    def scaling(field):
        xs = [math.log10(row["cells"]["mean"]) for row in rows
              if row["cells"]["mean"]]
        ys = [math.log10(row[field]["mean"]) for row in rows
              if row[field]["mean"] and row[field]["mean"] > 0]
        if len(xs) != len(ys):
            return None
        fit = linear_fit(xs, ys)
        if fit is None:
            return {"available": False,
                    "reason": "fewer than three sizes carry this observable, "
                              "so no honest uncertainty exists"}
        return {"available": True, "exponent": fit["slope"],
                "exponent_se": fit["slope_se"], "r_squared": fit["r_squared"],
                "note": "log10(seconds) vs log10(top cells)"}

    return {
        "by_size": rows,
        "scaling": {field: scaling(field) for field in
                    ("wall_seconds", "drive_seconds", "analysis_seconds",
                     "readout_seconds")},
        "memory_note": (
            "rss_bytes is the process RSS at the member's end and "
            "peak_rss_bytes is the process peak SO FAR — monotone, hence "
            "never a per-member cost; rss_delta_bytes is the per-member "
            "change and is the honest per-member figure"),
        "verdicts": _histogram(m["verdict"] for m in ok),
        "first_failing_certificates": _histogram(
            m["first_failing_certificate"] for m in ok),
        "failed_certificate_totals": _histogram(
            name for m in ok for name in m["failed_certificates"]),
    }


# =====================================================================
# the animation overlay
# =====================================================================

def _layout_from_raw(raw):
    """2-D classical-MDS coordinates per vertex from the persisted edge
    graph, normalized to unit RMS radius.

    A DRAWING LAYOUT, not a spacetime coordinate system: it is derived from
    graph shortest paths under |l^2|^{1/2} and carries no causal content.
    """
    from scipy.sparse.csgraph import shortest_path

    vertices = sorted({int(v) for cell in raw["cells"] for v in cell})
    if len(vertices) < 2:
        return {v: np.zeros(2) for v in vertices}
    index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    weights = np.full((n, n), np.inf)
    np.fill_diagonal(weights, 0.0)
    for edge in raw["edges"]:
        a, b = index.get(int(edge["a"])), index.get(int(edge["b"]))
        if a is None or b is None:
            continue
        length = complex(edge["length"][0], edge["length"][1])
        w = math.sqrt(max(abs(length ** 2), 1e-6))
        weights[a, b] = weights[b, a] = min(weights[a, b], w)
    distances = shortest_path(weights, method="D", directed=False)
    finite = np.isfinite(distances)
    distances[~finite] = (distances[finite].max() * 1.5
                          if finite.any() else 1.0)
    squared = distances ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    coords = vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))
    coords = coords - coords.mean(0)
    rms = math.sqrt((coords ** 2).sum(1).mean()) or 1.0
    return {vertices[i]: coords[i] / rms for i in range(n)}


def _absent(axis, title, reason):
    """Draw an ABSENT panel that says what is absent and WHY.

    The honest answer on this host is that most channels are absent, so an
    absent panel must still be a legible statement rather than an empty
    frame.
    """
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("#f5f5f5")
    axis.text(0.5, 0.62, "ABSENT", ha="center", va="center", fontsize=13,
              color="#b00020", fontweight="bold", transform=axis.transAxes)
    axis.text(0.5, 0.40, _wrap(reason, 46), ha="center", va="center",
              fontsize=6.5, color="#333333", transform=axis.transAxes)


def _wrap(text, width):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width))


def render_overlay(document, path, frame=-1):
    """Render the recursive-hierarchy overlay of one persisted frame.

    Reads exactly the same checkpoint data the headless path reads — the
    persisted run document — so the two can never disagree. Panels:

    1. component world tubes over the drawing layout;
    2. response vertices and stalks with the response-network edges;
    3. the fiber layer, with triangle-anchored rank-three bands highlighted;
    4. accepted and rejected derived transports;
    5. the determinant / projective / center holonomy channels and winding;
    6. the Berry-cancelled exchange and rotation characters;
    7. certificate failures;
    8. the verdict banner and its named reasons.

    Every panel that has nothing to draw says ABSENT and names the reason.
    """
    import matplotlib
    if not os.environ.get("DISPLAY") or path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    checkpoint = document["checkpoints"][frame]
    raw = (document["raw_geometry"] if frame in (-1, len(
        document["checkpoints"]) - 1) else _raw_from_checkpoint(checkpoint))
    layout = _layout_from_raw(raw)

    figure, axes = plt.subplots(2, 4, figsize=(19, 9.5))
    figure.suptitle(
        f"#778 recursive baryon simulation — verdict: "
        f"{document['verdict']['verdict'].upper()}   "
        f"(frame {frame if frame >= 0 else len(document['checkpoints']) - 1}"
        f" of {len(document['checkpoints'])}, "
        f"{document['host']['cells']} top cells, seed "
        f"{document['provenance']['seed']}, config "
        f"{document['provenance']['config_hash'][:8]})",
        fontsize=12)

    _panel_world_tubes(axes[0][0], document, layout, raw)
    _panel_response(axes[0][1], document, layout)
    _panel_fibers(axes[0][2], document)
    _panel_transports(axes[0][3], document, layout)
    _panel_holonomy(axes[1][0], document, Circle)
    _panel_statistics(axes[1][1], document)
    _panel_certificates(axes[1][2], document)
    _panel_verdict(axes[1][3], document)

    figure.text(0.5, 0.012,
                "The layout is a classical-MDS DRAWING of the 1-skeleton, "
                "not a spacetime coordinate system.",
                ha="center", fontsize=7, color="#555555")
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    if path:
        figure.savefig(path, dpi=120)
        plt.close(figure)
        return path
    return figure


def _raw_from_checkpoint(checkpoint):
    return checkpoint["raw_complex"]


def _component_colors(count):
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap("tab10")
    return [colormap(i % 10) for i in range(max(count, 1))]


def _panel_world_tubes(axis, document, layout, raw):
    hierarchy = document["hierarchy"]["analysis_slice"]
    axis.set_title("component world tubes (analysis resolution)", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    if not hierarchy or not hierarchy["components"]:
        _absent(axis, "component world tubes",
                "the modularity discovery returned no component")
        return
    for edge in raw["edges"]:
        a, b = layout.get(int(edge["a"])), layout.get(int(edge["b"]))
        if a is None or b is None:
            continue
        axis.plot([a[0], b[0]], [a[1], b[1]], color="#cccccc", linewidth=0.5,
                  zorder=1)
    colors = _component_colors(len(hierarchy["components"]))
    for index, component in enumerate(hierarchy["components"]):
        points = np.array([layout[v] for v in component["support"]
                           if v in layout])
        if not len(points):
            continue
        axis.scatter(points[:, 0], points[:, 1], s=26, color=colors[index],
                     edgecolors="black", linewidths=0.3, zorder=3,
                     label=f"c{index} |{component['volume']}|")
        centroid = points.mean(0)
        axis.text(centroid[0], centroid[1], str(index), fontsize=8,
                  ha="center", va="center", zorder=4,
                  bbox={"boxstyle": "circle,pad=0.15", "fc": "white",
                        "ec": colors[index], "lw": 1.0})
    axis.legend(fontsize=6, loc="upper right", framealpha=0.85)
    axis.set_xlabel(
        f"Q={hierarchy['q']:.3f}  levels={hierarchy['levels']}  "
        f"next-level components="
        f"{document['hierarchy']['next_level_components']}",
        fontsize=7)


def _stalk_positions(document, layout, count):
    """Positions for `count` response stalks.

    The first ones sit on their component's centroid; any stalk BEYOND the
    discovered components — the reduction covers every cell, so an uncovered
    remainder gets its own stalk — goes on an outer ring so it is visibly a
    different kind of object rather than a coincident dot at the origin.
    Returns (positions, number drawn on the ring).
    """
    hierarchy = document["hierarchy"]["analysis_slice"]
    positions = []
    for component in (hierarchy["components"] if hierarchy else []):
        points = np.array([layout[v] for v in component["support"]
                           if v in layout])
        positions.append(points.mean(0) if len(points) else None)
    positions = positions[:count]
    extra = max(0, count - len(positions))
    radius = 1.4 * max(
        (float(np.linalg.norm(p)) for p in positions if p is not None),
        default=1.0)
    unplaced = [i for i, p in enumerate(positions) if p is None]
    ring = unplaced + list(range(len(positions), count))
    positions += [None] * extra
    for order, index in enumerate(ring):
        angle = 2.0 * math.pi * order / max(len(ring), 1)
        positions[index] = np.array([radius * math.cos(angle),
                                     radius * math.sin(angle)])
    return positions, extra


def _panel_response(axis, document, layout):
    response = document["response_hierarchy"]
    network = response.get("response_network") or {}
    axis.set_title("response vertices and stalks", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    stalks = network.get("stalk_dimensions")
    if not response.get("built") or stalks is None:
        _absent(axis, "response vertices and stalks",
                response.get("reason") or network.get("reason")
                or "the recursive reduction produced no response network")
        return
    centroids, extra = _stalk_positions(document, layout, len(stalks))
    for a, b in network.get("edges") or []:
        if a < len(centroids) and b < len(centroids):
            axis.plot([centroids[a][0], centroids[b][0]],
                      [centroids[a][1], centroids[b][1]],
                      color="#3366cc", linewidth=1.6, zorder=2)
    colors = _component_colors(len(stalks))
    scale = max(float(d) for d in stalks) or 1.0
    for index, dimension in enumerate(stalks):
        point = centroids[index]
        axis.scatter([point[0]], [point[1]],
                     s=80 + 520 * float(dimension) / scale,
                     color=colors[index], edgecolors="black", linewidths=0.6,
                     zorder=3)
        axis.text(point[0], point[1], str(dimension), fontsize=7,
                  ha="center", va="center", color="white", zorder=4)
    coverage = network.get("coverage_residual")
    label = (f"stalk dims {stalks}   edges={network.get('edge_count')}   "
             f"coverage residual="
             + (f"{coverage:.2g}" if coverage is not None else "unknown")
             + f"\nrealization="
               f"{(response.get('realization') or {}).get('type')}")
    if extra:
        label += (f"   ({extra} stalk(s) beyond the discovered components — "
                  "the reduction covers every cell, so an uncovered "
                  "remainder gets its own stalk; drawn on the outer ring)")
    axis.set_xlabel(_wrap(label, 78), fontsize=6.5)


def _panel_fibers(axis, document):
    fibers = document["fibers"]
    axis.set_title("fibers: rank spectrum and anchoring", fontsize=9)
    if not fibers["total"]:
        _absent(axis, "fibers", "no band was enumerated on this complex")
        return
    ranks = sorted(int(k) for k in fibers["rank_histogram"])
    counts = [fibers["rank_histogram"][str(r)] for r in ranks]
    accepted = [fibers["accepted_rank_histogram"].get(str(r), 0)
                for r in ranks]
    # Ranks 1..3 always appear on the axis, so a missing rank three is
    # visible as an empty slot rather than as an axis that never mentions it.
    ranks = sorted(set(ranks) | {1, 2, 3})
    counts = [fibers["rank_histogram"].get(str(r), 0) for r in ranks]
    accepted = [fibers["accepted_rank_histogram"].get(str(r), 0)
                for r in ranks]
    positions = np.arange(len(ranks))
    axis.bar(positions - 0.18, counts, width=0.34, color="#bbbbbb",
             label="enumerated")
    axis.bar(positions + 0.18, accepted, width=0.34, color="#3366cc",
             label="accepted")
    axis.set_xticks(positions)
    axis.set_xticklabels([str(r) for r in ranks])
    axis.set_xlim(-0.6, len(ranks) - 0.4)
    axis.set_xlabel("band rank", fontsize=8)
    axis.set_ylabel("bands", fontsize=8)
    axis.legend(fontsize=7, loc="upper right")
    anchored = sum(1 for q in document["particles"]["quarks"]
                   if q["triangle_anchor_score"] is not None)
    if fibers["rank_three_accepted"] == 0:
        index_three = ranks.index(3)
        axis.annotate("rank 3\nABSENT", xy=(index_three, 0),
                      xytext=(index_three, max(max(counts), 1) * 0.22),
                      ha="center", va="bottom", fontsize=7, color="#b00020",
                      fontweight="bold")
        axis.text(0.5, 0.66,
                  _wrap("NO triangle-anchored rank-three fiber: every "
                        "accepted band is rank " +
                        ", ".join(sorted(fibers["accepted_rank_histogram"]))
                        + ", so the colour-rank-three gate is unreachable "
                          "and with it the anchor, the colour wedge, the "
                          "flavour read and the charge read",
                        42),
                  transform=axis.transAxes, ha="center", va="center",
                  fontsize=6.2, color="#b00020",
                  bbox={"boxstyle": "round,pad=0.35", "fc": "#fff3f3",
                        "ec": "#b00020", "lw": 0.8})
    axis.set_title(
        f"fibers: {fibers['accepted']}/{fibers['total']} accepted, "
        f"{fibers['rank_three_accepted']} rank-3 anchored "
        f"({anchored} anchored quark reads)", fontsize=9)


def _panel_transports(axis, document, layout):
    transports = document["transports"]
    axis.set_title("derived transports: accepted vs rejected", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    if not transports["derived"]:
        _absent(axis, "derived transports",
                "no ordered pair of same-degree, same-rank candidate bands "
                "existed, so no transport was derived")
        return
    band_reads = len(document["fibers"]["reads"])
    centroids, _extra = _stalk_positions(document, layout, band_reads)
    if not centroids:
        _absent(axis, "derived transports", "no component centroid to draw on")
        return
    for entry in transports["detail"]:
        a, b = entry["from"], entry["to"]
        if a >= len(centroids) or b >= len(centroids):
            continue
        style = "-" if entry["accepted"] else "--"
        color = "#1a7f37" if entry["accepted"] else "#b00020"
        axis.plot([centroids[a][0], centroids[b][0]],
                  [centroids[a][1], centroids[b][1]], style, color=color,
                  linewidth=1.2, alpha=0.8, zorder=2)
    for index, point in enumerate(centroids):
        axis.scatter([point[0]], [point[1]], s=90, color="#444444", zorder=3)
        axis.text(point[0], point[1], str(index), fontsize=7, color="white",
                  ha="center", va="center", zorder=4)
    reasons = "; ".join(f"{k} x{v}" for k, v in
                        transports["rejection_reasons"].items() if k != "None")
    axis.set_xlabel(
        _wrap(f"{transports['accepted']}/{transports['derived']} accepted, "
              f"leakage min {transports['leakage_min']}  |  "
              f"rejected: {reasons or 'none'}", 62), fontsize=6.5)


def _panel_holonomy(axis, document, Circle):
    transports = document["transports"]
    axis.set_title("determinant / projective / center holonomy, winding",
                   fontsize=9)
    channels = ("full", "determinant", "projective", "center", "winding")
    available = [c for c in channels
                 if (transports.get(c) or {}).get("available")]
    if not available:
        _absent(axis, "holonomy channels",
                (transports.get("determinant") or {}).get("reason")
                or "no accepted derived transport on this complex")
        return
    axis.add_patch(Circle((0, 0), 1.0, fill=False, color="#999999", lw=0.8))
    axis.set_xlim(-1.3, 1.3)
    axis.set_ylim(-1.3, 1.3)
    axis.set_aspect("equal")
    lines = []
    determinant = transports.get("determinant") or {}
    if determinant.get("available"):
        z = complex(*determinant["determinant"])
        axis.plot([0, z.real], [0, z.imag], color="#3366cc", lw=1.8)
        axis.scatter([z.real], [z.imag], color="#3366cc", zorder=3)
        lines.append(f"det = {z:.4g} (|det| = {abs(z):.6g})")
    center = transports.get("center") or {}
    if center.get("available"):
        for branch, entry in sorted(center["branches"].items()):
            if not entry.get("lift_trace"):
                continue
            z = complex(*entry["lift_trace"]) / 3.0
            axis.scatter([z.real], [z.imag], marker="s", zorder=3,
                         label=f"branch {branch} (sector "
                               f"{entry['center_sector']})")
            lines.append(f"branch {branch}: sector {entry['center_sector']}")
        axis.legend(fontsize=6, loc="lower left")
    winding = transports.get("winding") or {}
    if winding.get("available"):
        lines.append(f"winding nu = {winding['winding']} "
                     f"({winding['closure']})")
    axis.text(0.02, 0.98, "\n".join(lines) or "-", transform=axis.transAxes,
              fontsize=6.5, va="top", ha="left")


def _panel_statistics(axis, document):
    statistics = document["statistics"]
    axis.set_title("Berry-cancelled exchange / rotation characters",
                   fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    lines = ["DECLARED analytic carriers (exact at every size):"]
    for dimension, entry in sorted(statistics["declared_carrier"].items()):
        spinor = complex(*entry["spinor_character"])
        vector = complex(*entry["vector_character"])
        lines.append(f"  d={dimension}: spinor chi(2pi) = {spinor.real:+.6f}")
        lines.append(f"        vector chi(2pi) = {vector.real:+.6f}")
        lines.append("        |exp(2pi Sigma)+I| = "
                     f"{entry['double_cover_residual']:.1e}")
    for fixture in document["exactness"]:
        if fixture["name"] == "berry_cancellation":
            detail = fixture["detail"]
            lines.append("  single exchange = "
                         f"{complex(*detail['single_exchange']).real:+.6f}")
            lines.append("  double exchange = "
                         f"{complex(*detail['double_exchange']).real:+.6f}")
            lines.append("  raw loop det    = "
                         f"{complex(*detail['raw_determinant']):.4g}"
                         f" (a sign: {detail['raw_is_a_sign']})")
    lines.append("")
    lines.append("EMERGENT carrier:")
    carrier = statistics["emergent_carrier"]
    if carrier["supplied"]:
        lines.append(f"  supplied, {carrier['rows']} frame rows")
    else:
        lines.append("  ABSENT --")
        lines.append("  " + _wrap(carrier["reason"], 48).replace(
            "\n", "\n  "))
    lines.append("  spin lift ABSENT --")
    lines.append("  " + _wrap(statistics["spin_lift"]["reason"], 48).replace(
        "\n", "\n  "))
    axis.text(0.03, 0.96, "\n".join(lines), transform=axis.transAxes,
              fontsize=6.0, va="top", ha="left", family="monospace")


def _panel_certificates(axis, document):
    verdict = document["verdict"]
    failures = verdict["failed_certificates"]
    axis.set_title("certificate failures", fontsize=9)
    if not failures:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.text(0.5, 0.5, "every certificate held", ha="center",
                  va="center", fontsize=11, color="#1a7f37",
                  transform=axis.transAxes)
        return
    order = [name for name in BARYON_GATE_ORDER if name in failures]
    order += [name for name in failures if name not in BARYON_GATE_ORDER]
    positions = np.arange(len(order))
    first = verdict["first_failing_certificate"]
    colors = ["#b00020" if name == first else "#e08b8b" for name in order]
    axis.barh(positions, [1.0] * len(order), color=colors)
    axis.set_yticks(positions)
    axis.set_yticklabels(order, fontsize=6.5)
    axis.invert_yaxis()
    axis.set_xticks([])
    axis.set_xlabel(f"{len(failures)} of {len(BARYON_GATE_ORDER)} baryon "
                    f"gates failed; first = {first}", fontsize=7)
    quark_failures = document["particles"]["first_failing_certificate"]
    if quark_failures:
        axis.text(0.98, 0.02,
                  "quark first-failures: " + ", ".join(
                      f"{k} x{v}" for k, v in quark_failures.items()),
                  transform=axis.transAxes, fontsize=6, ha="right",
                  va="bottom")


def _panel_verdict(axis, document):
    verdict = document["verdict"]
    axis.set_title("verdict and its named reasons", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    color = {"certified proton": "#1a7f37",
             "baryon candidate": "#b08900",
             "quasi-free sharp-spin obstruction": "#b08900",
             "no baryon": "#b00020"}[verdict["verdict"]]
    axis.text(0.5, 0.92, verdict["verdict"].upper(), ha="center", va="center",
              fontsize=15, color=color, fontweight="bold",
              transform=axis.transAxes)
    confidence = verdict["confidence"]
    lines = [f"classifier: {verdict['library_classification']}",
             ("confidence: "
              + (f"{confidence:.4g}" if confidence is not None else "unknown")
              + " (passed-gate fraction, NOT a probability)"),
             f"certificate: {verdict['certificate_grade']} "
             f"holds={verdict['certificate_holds']}",
             "",
             "missing evidence, named:"]
    for reason in verdict["missing_evidence"]:
        lines.append("  - " + _wrap(reason, 56).replace("\n", "\n    "))
    axis.text(0.02, 0.82, "\n".join(lines), transform=axis.transAxes,
              fontsize=6.2, va="top", ha="left")


def render_animation(document, path, fps=1):
    """Render EVERY persisted frame. A GIF when more than one exists,
    otherwise a single still. Both read the same checkpoint data.

    Returns ``(path, frames rendered)``. The per-frame stills are scratch
    and are removed with their temporary directory — the animation is the
    artifact.
    """
    frames = len(document["checkpoints"])
    if frames <= 1 or not path.lower().endswith((".gif", ".mp4")):
        render_overlay(document, path, frame=frames - 1)
        return path, 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index in range(frames):
            frame_path = os.path.join(directory, f"frame_{index:04d}.png")
            render_overlay(document, frame_path, frame=index)
            paths.append(frame_path)
        figure = plt.figure(figsize=(19, 9.5))
        axis = figure.add_axes((0, 0, 1, 1))
        axis.set_axis_off()
        artists = []
        for frame_path in paths:
            image = plt.imread(frame_path)
            artists.append([axis.imshow(image, animated=True)])
        movie = animation.ArtistAnimation(figure, artists,
                                          interval=int(1000 / max(fps, 1)),
                                          blit=False)
        writer = "pillow" if path.lower().endswith(".gif") else "ffmpeg"
        movie.save(path, writer=writer)
        plt.close(figure)
    return path, frames


# =====================================================================
# reporting
# =====================================================================

def print_run_summary(document, stream=sys.stdout):
    write = stream.write
    verdict = document["verdict"]
    write("\n" + "=" * 72 + "\n")
    write(f"VERDICT: {verdict['verdict']}\n")
    write("=" * 72 + "\n")
    write(f"  classifier      : {verdict['library_classification']}\n")
    write(f"  vocabulary      : {', '.join(VERDICTS)}\n")
    write(f"  first failure   : {verdict['first_failing_certificate']}\n")
    write(f"  failed gates    : {len(verdict['failed_certificates'])} of "
          f"{len(BARYON_GATE_ORDER)}"
          f" — {', '.join(verdict['failed_certificates'])}\n")
    write("  missing evidence:\n")
    for reason in verdict["missing_evidence"]:
        write(f"    - {reason}\n")
    write("\nwhat the run produced\n")
    hierarchy = document["hierarchy"]["analysis_slice"]
    write(f"  host            : {document['host']['cells']} top cells, "
          f"{document['host']['edges']} edges, "
          f"{document['host']['vertices']} vertices\n")
    write(f"  components      : "
          f"{hierarchy['component_count'] if hierarchy else 0} at gamma="
          f"{document['hierarchy']['analysis_resolution']}, next level "
          f"{document['hierarchy']['next_level_components']}\n")
    write(f"  bands           : {document['fibers']['accepted']} accepted of "
          f"{document['fibers']['total']}, rank histogram "
          f"{document['fibers']['accepted_rank_histogram']}, rank-3 accepted "
          f"{document['fibers']['rank_three_accepted']}\n")
    write(f"  transports      : {document['transports']['accepted']} accepted "
          f"of {document['transports']['derived']}, leakage min "
          f"{document['transports']['leakage_min']}\n")
    write(f"  quark reads     : {document['particles']['quark_reads']}, "
          f"certified {document['particles']['certified_quarks']}, first "
          f"failures {document['particles']['first_failing_certificate']}\n")
    write(f"  covariance      : {document['covariance']['state_count']} "
          f"states, worst purity defect "
          f"{document['covariance']['purity_defect_max']}\n")
    write(f"  spectral dim    : peak "
          f"{document['spectral_dimension'].get('peak')} against the pinned "
          f"{PINNED_DS_BASELINE} baseline\n")
    inexact = [f["name"] for f in document["exactness"] if not f["exact"]]
    exact = len(document["exactness"]) - len(inexact)
    write(f"  exact fixtures  : {exact} of "
          f"{len(document['exactness'])} exact"
          + (f"; FAILED {inexact}" if inexact else "") + "\n")
    certificates = document["certificates"]
    write(f"  certificates    : {certificates['held']} held, "
          f"{certificates['refused']} refused with a named reason, "
          f"{certificates['failed_count']} failed, of "
          f"{certificates['total']}\n")
    write(f"  refinement      : "
          f"{document['drive']['refinement_events']} cells committed by the "
          f"declared geometry-only rule\n")
    write(f"  runtime         : "
          f"{document['runtime']['wall_seconds']:.2f} s wall "
          f"({document['runtime']['drive_seconds']:.2f} s drive, "
          f"{document['runtime']['analysis_seconds']:.2f} s analysis, "
          f"{document['runtime']['readout_seconds']:.2f} s readout)\n")
    write("\nA rigorous negative is valid completion.\n")


def print_replay_summary(report, stream=sys.stdout):
    write = stream.write
    write("\n" + "=" * 72 + "\n")
    write(f"REPLAY {'VERIFIED' if report['verified'] else 'FAILED'}\n")
    write("=" * 72 + "\n")
    write(f"  frames          : {report['frames_discrete_identical']} of "
          f"{report['frames_total']} discrete-identical, "
          f"{report['frames_byte_identical']} byte-identical\n")
    worst = max((f["worst_relative_difference"] or 0.0)
                for f in report["frames"]) if report["frames"] else 0.0
    write(f"  worst continuous: {worst:.3g} against tolerance "
          f"{report['tolerance']:.1g}\n")
    write(f"  content hashes  : {report['content_hashes_matched']} of "
          f"{len(report['content_hashes'])} matched\n")
    unmatched = [h["block"] for h in report["content_hashes"]
                 if not h["match"]]
    if unmatched:
        write(f"    MISMATCHED    : {unmatched}\n")
    write(f"  sidecar         : file hash "
          f"{report['sidecar']['file_hash_match']}, "
          f"{sum(1 for a in report['sidecar']['arrays'] if a['verified'])} of "
          f"{len(report['sidecar']['arrays'])} arrays verified\n")
    write(f"  fixtures        : "
          f"{sum(1 for f in report['exactness'] if f['both_exact'])} of "
          f"{len(report['exactness'])} exact on both paths\n")
    write(f"  verdict         : stored {report['verdict']['stored']!r}, "
          f"replayed {report['verdict']['replayed']!r}, match "
          f"{report['verdict']['match']}\n")
    if report["verdict"].get("reason"):
        write(f"    reason        : {report['verdict']['reason']}\n")
    write(f"  runtime         : {report['wall_seconds']:.2f} s\n")


def print_campaign_summary(report, stream=sys.stdout):
    write = stream.write
    aggregate = report["aggregate"]
    write("\n" + "=" * 72 + "\n")
    write(f"CAMPAIGN — {report['members_ok']} of {report['members_total']} "
          f"members ran\n")
    write("=" * 72 + "\n")
    failed = report["members_failed"]
    if failed:
        write(f"  FAILED members (reported, never dropped): "
              f"{[(m['size'], m['seed'], m['error']) for m in failed]}\n")
    header = ("  size  cells   wall s    drive s   analysis s  RSS MiB   "
              "hits/miss  inval  rank3  acc.tr  verdicts\n")
    write(header)
    for row in aggregate["by_size"]:
        rss = row["rss_bytes"]["mean"]
        write(f"  {row['size']:>4}  {row['cells']['mean']:>5.0f}  "
              f"{row['wall_seconds']['mean']:>7.2f}  "
              f"{row['drive_seconds']['mean']:>8.2f}  "
              f"{row['analysis_seconds']['mean']:>10.3f}  "
              f"{(rss / 1048576.0 if rss else 0):>7.1f}  "
              f"{row['cache_hits']['mean']:>4.0f}/"
              f"{row['cache_misses']['mean']:<4.0f}  "
              f"{row['cache_invalidations']['mean']:>5.0f}  "
              f"{row['rank_three_accepted']['mean']:>5.1f}  "
              f"{row['accepted_transports']['mean']:>6.1f}  "
              f"{row['verdicts']}\n")
    for field, fit in aggregate["scaling"].items():
        if fit and fit.get("available"):
            write(f"  scaling {field}: exponent {fit['exponent']:.2f} "
                  f"+- {fit['exponent_se']:.2f} (R^2 = "
                  f"{fit['r_squared']:.3f}) vs top cells\n")
        elif fit:
            write(f"  scaling {field}: {fit['reason']}\n")
    write(f"  verdicts        : {aggregate['verdicts']}\n")
    write(f"  first failures  : {aggregate['first_failing_certificates']}\n")
    write(f"  runtime         : {report['wall_seconds']:.2f} s\n")


# =====================================================================
# CLI
# =====================================================================

_EPILOG = """
verdict vocabulary (the complete list; no target-dependent success string)
  no baryon | baryon candidate | certified proton
  | quasi-free sharp-spin obstruction

what is exact, certified numerical, or heuristic
  EXACT               static Schur (Kron) reduction; the shifted
                      Feshbach-Schur pencil; second-quantized subset sums and
                      the assembled hopping block; F_3 unitarity and the
                      Lambda^3 C^3 singlet Gram; the graded (Pauli) amplitude;
                      exp(2 pi Sigma) = -I; the composable near-isometry
                      budget; the closed determinant winding; the center
                      sector's cube-root branch relation; the Berry-cancelled
                      exchange and rotation characters; covariance
                      Hermiticity and the Wick expansions.
  CERTIFIED NUMERICAL band isolation, projector and eigen residuals; band
                      acceptance; the covariance purity defect; derived
                      transport acceptance and leakage; labeled-fiber-sum
                      Gram defects; the response network's coverage residual;
                      the anchor's calibration margin; the heat-kernel
                      spectral dimension.
  HEURISTIC           modularity community discovery and its
                      resolution/persistence tracking; which band of a
                      component is taken as the candidate; QuarkRead /
                      BaryonRead confidence (a passed-gate fraction, not a
                      probability); the MDS drawing layout, which is NOT a
                      spacetime coordinate system.

An unforced proton is a scientific success condition, not a software
completion condition. `run` exits 0 whether or not one
emerges; `replay` exits non-zero only when a stored verdict or content hash
fails to reproduce.
"""


def _add_run_arguments(parser):
    parser.add_argument("--size", type=int, default=DECLARED_SIZE_FAST,
                        help="host refinement count "
                             "(default: the fast %(default)s)")
    parser.add_argument("--seed", type=int, default=DECLARED_SEED,
                        help="node seed; a seed LABELS an attempt "
                             "(default: %(default)s)")
    parser.add_argument("--host-seed", type=int, default=DECLARED_HOST_SEED,
                        help="host construction seed (default: %(default)s)")
    parser.add_argument("--drive-steps", type=int,
                        default=DECLARED_DRIVE_STEPS_FAST,
                        help="stage-1 + stage-2 units to drive "
                             "(default: %(default)s)")
    parser.add_argument("--large", action="store_true",
                        help=f"the documented larger mode: size "
                             f"{DECLARED_SIZE_LARGE}, "
                             f"{DECLARED_DRIVE_STEPS_LARGE} drive steps")
    parser.add_argument("--submode", choices=("strict",
                                              "certificates-blind-mean-field"),
                        default="strict",
                        help="the labeled emergence sub-mode "
                             "(default: %(default)s)")
    parser.add_argument("--no-refine", action="store_true",
                        help="disable the declared geometry-only "
                             "refinement rule")
    parser.add_argument("--fock-oracle", action="store_true",
                        help="build the lazy Fock DAG (oracle / non-Gaussian "
                             "boundary path only)")
    parser.add_argument("--degrees", default=",".join(
        str(k) for k in DECLARED_DEGREES),
        help="comma-separated Hodge degrees the bands are enumerated at "
             "(default: %(default)s; degree 0 reads the POSITIVE U(1) "
             "connection graph Laplacian -- SpectralFiber's induced-subgraph "
             "D - A, not the Hodge L_0 -- where the Fock oracle has an exact "
             "reference)")


def _config_from_arguments(args):
    size = DECLARED_SIZE_LARGE if args.large else args.size
    steps = (DECLARED_DRIVE_STEPS_LARGE if args.large
             else args.drive_steps)
    degrees = tuple(int(k) for k in args.degrees.split(",") if k.strip())
    return make_config(size=size, seed=args.seed, host_seed=args.host_seed,
                       drive_steps=steps, submode=args.submode,
                       refine=not args.no_refine,
                       fock_oracle=args.fock_oracle,
                       degrees=degrees or DECLARED_DEGREES)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="recursive_baryon_simulation.py",
        description=__doc__.split("\n\n")[0],
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="the one documented end-to-end command",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_run_arguments(run)
    run.add_argument("--out", default="recursive_baryon_run.json",
                     help="run document path (default: %(default)s)")
    run.add_argument("--sidecar", default=None,
                     help="binary matrix sidecar path "
                          "(default: <out>.sidecar.npz)")
    run.add_argument("--animate", default=None,
                     help="also render the overlay to this PNG/GIF")
    run.add_argument("--quiet", action="store_true")

    replay = subparsers.add_parser(
        "replay", help="cold-cache replay of a persisted run document")
    replay.add_argument("--from", dest="source", required=True,
                        help="run document written by `run`")
    replay.add_argument("--out", default=None,
                        help="replay report path")
    replay.add_argument("--quiet", action="store_true")

    campaign = subparsers.add_parser(
        "campaign", help="headless size campaign and scaling report")
    campaign.add_argument("--sizes", default=",".join(
        str(s) for s in DECLARED_CAMPAIGN_SIZES),
        help="comma-separated host sizes (default: %(default)s)")
    campaign.add_argument("--seeds", default=",".join(
        str(s) for s in DECLARED_CAMPAIGN_SEEDS),
        help="comma-separated node seeds (default: %(default)s)")
    campaign.add_argument("--drive-steps", type=int,
                          default=DECLARED_DRIVE_STEPS_FAST)
    campaign.add_argument("--no-refine", action="store_true")
    campaign.add_argument("--out", default="recursive_baryon_campaign.json")
    campaign.add_argument("--member-dir", default=None,
                          help="write each member's full run document here")
    campaign.add_argument("--quiet", action="store_true")

    animate = subparsers.add_parser(
        "animate", help="render the overlay from a persisted run document")
    animate.add_argument("--from", dest="source", required=True)
    animate.add_argument("--out", default="recursive_baryon_overlay.png")
    animate.add_argument("--frame", type=int, default=-1,
                         help="frame index, or -1 for the last")
    animate.add_argument("--all-frames", action="store_true",
                         help="render every frame (a GIF when --out is .gif)")

    fixtures = subparsers.add_parser(
        "fixtures", help="run the exactness fixtures alone")
    fixtures.add_argument("--out", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = _config_from_arguments(args)
        sidecar = args.sidecar or (
            os.path.splitext(args.out)[0] + ".sidecar.npz")
        if not args.quiet:
            print("#778 recursive baryon simulation — "
                  f"size {config['size']}, "
                  f"seed {config['seed']}, {config['drive_steps']} drive "
                  f"steps, submode {config['emergence_submode']}", flush=True)
        document = run_simulation(config, sidecar_path=sidecar,
                                  progress=not args.quiet)
        with open(args.out, "w") as handle:
            json.dump(document, handle)
        if not args.quiet:
            print_run_summary(document)
            print(f"\nrun document -> {args.out}")
            if document.get("sidecar"):
                print(f"sidecar      -> {document['sidecar']['path']}")
        if args.animate:
            render_animation(document, args.animate)
            if not args.quiet:
                print(f"overlay      -> {args.animate}")
        # Exits 0 whether or not a proton emerged: the exit code reports
        # whether the SOFTWARE ran, never whether the physics obliged.
        return 0

    if args.command == "replay":
        with open(args.source) as handle:
            document = json.load(handle)
        report = replay_document(
            document, directory=os.path.dirname(os.path.abspath(args.source)),
            progress=not args.quiet)
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(report, handle)
        if not args.quiet:
            print_replay_summary(report)
            if args.out:
                print(f"\nreplay report -> {args.out}")
        return 0 if report["verified"] else 1

    if args.command == "campaign":
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        base = make_config(drive_steps=args.drive_steps,
                           refine=not args.no_refine)
        if args.member_dir:
            os.makedirs(args.member_dir, exist_ok=True)
        report = run_campaign(sizes, seeds, base, out_dir=args.member_dir,
                              progress=not args.quiet)
        with open(args.out, "w") as handle:
            json.dump(report, handle)
        if not args.quiet:
            print_campaign_summary(report)
            print(f"\ncampaign report -> {args.out}")
        return 0

    if args.command == "animate":
        with open(args.source) as handle:
            document = json.load(handle)
        if args.all_frames:
            render_animation(document, args.out)
        else:
            render_overlay(document, args.out, frame=args.frame)
        print(f"overlay -> {args.out}")
        return 0

    if args.command == "fixtures":
        fixtures = exactness_fixtures()
        for fixture in fixtures:
            status = "exact" if fixture["exact"] else "FAILED"
            print(f"  {fixture['name']:<45} {status:>7}  residual "
                  f"{fixture['residual']:.3g} <= {fixture['tolerance']:.0e}")
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(fixtures, handle)
        return 0 if all(f["exact"] for f in fixtures) else 1

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
