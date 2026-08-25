# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#777 — multiscale convergence, the covariance-only dichotomy, and spectral 4D.

The Wave 4 validation experiment of epic #763. Everything else in the epic is
machinery; this driver is the instrument that turns the machinery into a
measurement. It answers one question with numbers: **do recursive components,
color/statistics certificates, amplitudes, and the near-four-dimensional
spacetime regime converge together under refinement?**

A rigorous NEGATIVE result is a valid outcome and is reported as one. No
threshold in this file is chosen by looking at a verdict: every threshold is
either a shipped library default or a value DECLARED in the block below,
before any datum was examined, and it is identical at every size and seed.

What is driven, and what is only read
-------------------------------------
The dynamics is the merged, unmodified `MultiCobordism` joint Regge-Hodge
stationarity objective in `SimulationMode.EMERGENCE` /
`EmergenceSubmode.STRICT`. Nothing in this file enters that objective: the
#776 firewall (a static `objectiveOf` over five declared scalars) makes that
structural, and this driver only *reads* the accepted geometry afterwards
through `runRecursiveAnalysis`'s schema-4 checkpoint and the public observable
classes.

Reused machinery, all merged on main: `MultiCobordism` (modes, checkpoint,
`replayCheckpoint`, the analysis overlay), `PersistentModularity`,
`SpectralFiber`, `RecursiveQuotient`, `ColorFiber` / `ColorAnchor`,
`FiberConnection`, `ExchangeHolonomy`, `ParticleClusters`, `CovarianceState`,
`LazyFockEngine`, `Certificate`, `AnalyticCache`, and the EXISTING
heat-kernel spectral-dimension estimator
`Spacetime.getSpectralDimensionOnSkeleton`. The spectral-dimension definition
is reused verbatim, never replaced.

Measured per (size, seed)
-------------------------
hierarchy depth; response-network type and realization residual; component
volume/boundary scaling; modularity resolution and persistence; fiber gaps and
Krein signatures; exact static-response residuals; shifted (Feshbach) and AMLS
(Craig-Bampton) residuals over declared frequency windows; amplitude Gram
defect and inductive-embedding compatibility; triangle anchoring; full /
determinant / projective / center holonomies; determinant winding;
Berry-cancelled exchange and rotation characters; spin-lift status; charge;
particle verdicts and the first-failing-certificate distribution;
`<J^2>` and `Var(J^2)` under a DECLARED one-particle spin convention; and the
heat-kernel spectral dimension on the interaction-history complex.

Unknown values are `null`, never zero. A `classification: "none"` read is
DATA and is reported as such.

Running it
----------
Cap parallelism to 8 threads; this box is shared::

    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \\
      BLIS_NUM_THREADS=8 .venv-build/bin/python \\
      examples/cobordism/multiscale_validation.py --quick --out quick.json

``--quick`` runs a reduced ensemble (3 sizes x 2 seeds), MEASURED at 30 s, so
the harness itself is exercisable in CI-adjacent time. The full study (5 sizes
x 5 seeds) is the default and was MEASURED at 246 s on a 16-core box at 8
threads — 9.5 +- 7.7 s per member, dominated by the stage-2 relaxation, which
grows from 1.7 s at size 6 to 20 s at size 44. Both modes run every negative
control, every analytic invariant, the threshold ladder and the cold-replay
check. Each run's own wall time is recorded in the output under ``runtime``.

Reproducibility, exactly
------------------------
Every emitted point carries its `config_hash`, `commit`, `size`, and `seed`,
and the run's schema-4 checkpoint is embedded, so `MultiCobordism.
replayCheckpoint` reproduces the point exactly from the record alone. A fresh
rebuild from (config, seed, commit) reproduces the FIRST committed move and the
whole stage-2 relaxation, but #579/#776 measured that the engine's move draw is
not process-deterministic past the first committed move; the drive is therefore
deliberately ONE stage-1 update plus a full relaxation, which is the engine's
deterministic unit. That limitation is a property of the engine, is stated in
the output under ``reproducibility``, and is not papered over.
"""

import argparse
import cmath
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

import tessera as T

cob = T.cobordism
MC = cob.MultiCobordism
EH = T.ExchangeHolonomy
QU = T.quantum

SCHEMA_VERSION = 1

# =====================================================================
# DECLARED constants — fixed before any datum was examined, identical at
# every size and every seed. Nothing below is retuned per size, and none
# of it was chosen by looking at a particle verdict.
# =====================================================================

#: Host refinement counts. "Size" of the experiment; at least three.
DECLARED_SIZES_FULL = (6, 12, 20, 30, 44)
DECLARED_SIZES_QUICK = (6, 12, 20)

#: The node seed set. A seed LABELS an attempt; it does not reproduce one
#: (#579). The ensemble is the seed set, not any single seed.
DECLARED_SEEDS_FULL = (7, 11, 13, 17, 19)
DECLARED_SEEDS_QUICK = (7, 11)

#: The host construction seed, held fixed so "size" varies alone.
DECLARED_HOST_SEED = 3

#: Hodge degrees the register/band layer is enumerated at.
DECLARED_DEGREES = (1,)

#: The resolution the component/fiber/reduction layer is READ at: gamma = 1,
#: the standard Newman-Girvan value. A literature default, declared here so it
#: is visibly not a fitted choice, and identical at every size and seed.
DECLARED_ANALYSIS_RESOLUTION = 1.0

#: The #765 resolution SCAN, for the modularity-resolution and persistence
#: measurements. Ascending; the analysis resolution is one of its points.
DECLARED_RESOLUTION_SCAN = (0.5, 1.0, 2.0)

#: The engine drive: one stage-1 combinatorial update (the deterministic
#: unit) then a bounded stage-2 relaxation.
DECLARED_CANDIDATE_MOVES = 6
DECLARED_STAGE2_ITERS = 12

#: The heat-kernel diffusion times, as a fixed geometric grid. The EXISTING
#: estimator (`Spacetime.getSpectralDimensionOnSkeleton`) is used unchanged.
DECLARED_SIGMAS = tuple(0.05 * (1.5 ** i) for i in range(20))
DECLARED_KRYLOV_DIM = 64

#: The pinned near-four-dimensional baseline this study compares against:
#: docs/source/quantum-experiments/overview/h_ds4_status.md, peak D_S at
#: T = 20k with its quoted uncertainty, and the geometric extrapolation.
PINNED_DS_BASELINE = 4.245
PINNED_DS_BASELINE_SIGMA = 0.024
PINNED_DS_EXTRAPOLATION = 4.07

#: Declared dimensionless shifted-response probe points, as fractions of the
#: operator's own spectral scale, with a declared window half-width. The grid
#: is dimensionless so it is literally the same grid at every size.
DECLARED_SHIFT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DECLARED_WINDOW_HALF_WIDTH_FRACTION = 0.1

#: Declared AMLS mode cutoff (Craig-Bampton retained-mode count).
DECLARED_AMLS_MODE_CUTOFF = 4

#: Relative tolerances at which two eigenvalues count as one near-degenerate
#: cluster, for the multiplicity histogram. A declared LADDER, so "no
#: degeneracy" is a statement about a range of tolerances rather than about
#: one lucky cut. Never fitted.
DECLARED_DEGENERACY_TOLERANCES = (1e-9, 1e-6, 1e-3)
DECLARED_DEGENERACY_TOLERANCE = DECLARED_DEGENERACY_TOLERANCES[1]

#: The steps of the declared closed rotation loop, and the rotation plane.
DECLARED_ROTATION_STEPS = 16
DECLARED_ROTATION_PLANE = (0, 1)
DECLARED_ROTATION_DIMENSIONS = (3, 4)

#: Threshold-sensitivity ladder: multiplicative perturbations applied to the
#: shipped band-isolation thresholds, spanning several decades so the scan
#: actually crosses the acceptance boundary instead of sitting on one side of
#: it. Raising the isolation floor can only ever REDUCE acceptance, so the
#: ladder cannot be a search for a desired verdict; it is the uncertainty
#: curve reported alongside the result.
DECLARED_THRESHOLD_SCAN = (1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3,
                           1.0e4, 1.0e5, 1.0e6)

#: Control seeds (kept apart from the ensemble seeds so a control can never
#: be mistaken for a measurement).
DECLARED_CONTROL_SEED = 20260823

#: "Var(J^2) approximately zero" is the SAME statement the baryon classifier
#: makes, so the two layers read one number: `ParticleClusters`' own
#: sharp-spin tolerance. Declared here, never fitted, and never widened to
#: reach a verdict.
DECLARED_VAR_J2_ZERO_TOLERANCE = (
    T.ParticleClustersConfig().spinVarianceTolerance)

#: Certificate gates, in the exact order `ParticleClusters` evaluates them.
#: `failedCertificates[0]` is therefore the FIRST failing certificate.
QUARK_GATE_ORDER = (
    "persistence", "localization", "parity-odd", "occupation-one",
    "color-rank-three", "color-rank-stability", "anchor",
    "anchor-stability", "transport-leakage", "winding", "winding-unit",
    "refinement-stability",
)


# =====================================================================
# host
# =====================================================================

def build_host(n_refine, seed=DECLARED_HOST_SEED):
    """The refined closed-S4 emergence host.

    The bare boundary-of-Delta^5 sphere refined by ``n_refine`` PreGeometric
    stellar Pachner adds, then given the same mild deterministic non-uniform
    metric. Byte-identical in construction to ``tests/cobordism/_closed_s4.py``
    — deliberately a standalone copy (the ``_holed_surface`` convention) so
    the example cannot drift when a test fixture is edited, and so an example
    never imports from the test tree.
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
# small numeric helpers — everything reports uncertainty
# =====================================================================

def _finite(value):
    """A float that JSON can carry, or None. Unknown is NEVER zero."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _score_scalar(value):
    """The real ordered scalar of a (possibly complex) modularity score.

    Q is complex in general (#849): its magnitude is how much structure a
    partition has and its argument is what kind.  A comparison needs one real
    number, and this is the same rule the library's ``objectiveValue`` uses --
    the score itself where Q is real, its magnitude where it is not.  The
    unreduced value is reported alongside wherever this appears, so nothing is
    lost by taking it.
    """
    if value is None:
        return None
    z = complex(value)
    return _finite(z.real if z.imag == 0.0 else abs(z))


def _complex_pair(value):
    if value is None:
        return None
    z = complex(value)
    return [_finite(z.real), _finite(z.imag)]


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
    """Ordinary least squares y = a + b x with standard errors and R^2.

    Returns None when fewer than three finite pairs are available — a fit
    through two points has no residual degrees of freedom and no honest
    uncertainty, so none is reported.
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
    return {
        "intercept": a, "intercept_se": _finite(se_a),
        "slope": b, "slope_se": _finite(se_b),
        "r_squared": _finite(1.0 - sse / sst) if sst > 0 else None,
        "n": n,
    }


def inverse_size_fit(sizes, values):
    """Fit y = y_inf + c / N over size means and classify the trend.

    `intercept` is the extrapolated N -> infinity value with its standard
    error. The verdict vocabulary describes the FIT and nothing more; three to
    five sizes cannot prove a continuum limit and this study never claims one:

    * ``exactly_constant`` — every size mean is identical, so there is no
      trend to fit;
    * ``flat_within_uncertainty`` — the 1/N coefficient is not separated from
      zero by two standard errors;
    * ``converging`` — the coefficient IS resolved and a 1/N law explains
      more than 90% of the spread, so the extrapolated limit is meaningful;
    * ``trending_but_not_inverse_size`` — the coefficient is resolved but a
      1/N law does not explain the spread, so the observable is moving with
      size in some other way and the extrapolation is NOT to be trusted;
    * ``insufficient_points`` — fewer than three sizes carry the observable.
    """
    finite = [v for v in values if _finite(v) is not None]
    if len(finite) == len(values) and len(set(finite)) == 1:
        fit = linear_fit([1.0 / s for s in sizes], values)
        return {"fit": fit, "verdict": "exactly_constant",
                "extrapolated_limit": finite[0],
                "extrapolated_limit_se": 0.0}
    fit = linear_fit([1.0 / s for s in sizes], values)
    if fit is None:
        return {"fit": None, "verdict": "insufficient_points"}
    slope, se = fit["slope"], fit["slope_se"]
    r2 = fit["r_squared"]
    if se is None or r2 is None:
        verdict = "insufficient_points"
    elif abs(slope) <= 2.0 * se:
        verdict = "flat_within_uncertainty"
    elif r2 > 0.9:
        verdict = "converging"
    else:
        verdict = "trending_but_not_inverse_size"
    return {"fit": fit, "verdict": verdict,
            "extrapolated_limit": fit["intercept"],
            "extrapolated_limit_se": fit["intercept_se"]}


def pearson(xs, ys):
    """Pearson r with a Fisher-z 95% interval; None when undefined."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if _finite(x) is not None and _finite(y) is not None]
    if len(pairs) < 4:
        return None
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if abs(r) >= 1.0 - 1e-15:
        return {"r": r, "n": n, "ci95": None}
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    lo, hi = z - 1.96 * se, z + 1.96 * se
    return {"r": r, "n": n,
            "ci95": [math.tanh(lo), math.tanh(hi)]}


# =====================================================================
# the DECLARED one-particle spin convention
# =====================================================================

def declared_spin_matrices(mode_count, offset=0):
    """J_alpha = direct sum of sigma_alpha / 2 over consecutive mode pairs.

    This is a DECLARED READOUT CONVENTION, not a quantity derived from the
    geometry. The epic's own `CovarianceState::wickSpinSquaredExpectation`
    takes CALLER-SUPPLIED one-particle spin matrices, so a convention must be
    named; this one pairs modes in the covariance's own deterministic order
    (2k, 2k+1), leaving an odd trailing mode spinless. `offset` shifts the
    pairing by one mode and exists ONLY to measure how much the read depends
    on the arbitrary pairing — that spread is reported as uncertainty, never
    used to select a value.

    Two consequences dominate the reading and are stated up front:

    1. When EVERY mode is paired (`mode_count - offset` even), any rank-1
       covariance is exactly a j = 1/2 eigenstate: `<J^2> = 3/4` and
       `Var(J^2) = 0` are identities of the READOUT, carrying no information
       about the geometry. Such reads are labelled `trivial_rank1` and can
       never be evidence of a proton.
    2. When a mode is left UNPAIRED it is a spin-0 mode, and the very same
       rank-1 state becomes a j = 1/2 + j = 0 superposition with a genuinely
       nonzero `Var(J^2)` — the sharp-spin negative fixture, produced
       here by the pairing convention rather than by the physics. The
       difference between the two arms is reported as
       `var_j2_pairing_spread`, and it is the honest uncertainty on every
       `Var(J^2)` this study quotes.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2.0
    sz = np.diag([1.0, -1.0]).astype(complex) / 2.0

    def blocks(sigma):
        out = np.zeros((mode_count, mode_count), dtype=complex)
        start = offset
        while start + 2 <= mode_count:
            out[start:start + 2, start:start + 2] = sigma
            start += 2
        return out

    return blocks(sx), blocks(sy), blocks(sz)


# =====================================================================
# one ensemble member
# =====================================================================

def _overlay_config(degrees, resolutions, cadence=1, cold=False, fock=False):
    cfg = MC.AnalysisConfig()
    cfg.enabled = True
    cfg.cadence = cadence
    cfg.degrees = list(degrees)
    cfg.resolutions = list(resolutions)
    cfg.cold_caches = cold
    cfg.fock_oracle = fock
    return cfg


def _first_accepted(band_read):
    for fiber in band_read.fibers:
        if fiber.accepted():
            return fiber
    return None


def _degeneracy_histogram(eigenvalues, tolerance):
    """Cluster-size histogram of near-degenerate eigenvalues.

    Reported RAW. A robust fourfold cluster is recorded as a fourfold
    cluster; this driver does not name it Kahler-Dirac taste.
    """
    values = sorted((complex(v) for v in eigenvalues),
                    key=lambda z: (z.real, z.imag))
    if not values:
        return {}
    scale = max(abs(v) for v in values) or 1.0
    histogram = {}
    size = 1
    for previous, current in zip(values, values[1:]):
        if abs(current - previous) <= tolerance * scale:
            size += 1
        else:
            histogram[size] = histogram.get(size, 0) + 1
            size = 1
    histogram[size] = histogram.get(size, 0) + 1
    return {str(k): v for k, v in sorted(histogram.items())}


def _spin_reads(state, rank):
    """<J^2>, Var(J^2) and the pairing-sensitivity spread on one state."""
    modes = state.modeCount()
    if modes < 2:
        return {"available": False, "reason": "fewer than two modes",
                "j2": None, "var_j2": None}
    out = {"available": True, "reason": None, "modes": modes, "rank": rank}
    values = []
    variances = []
    for offset in (0, 1):
        if modes < 2 + offset:
            continue
        jx, jy, jz = declared_spin_matrices(modes, offset)
        expectation = state.wickSpinSquaredExpectation(jx, jy, jz)
        variance = state.wickSpinSquaredVariance(jx, jy, jz)
        values.append(complex(expectation.value).real)
        variances.append(complex(variance.value).real)
        if offset == 0:
            out["j2"] = _finite(complex(expectation.value).real)
            out["j2_imaginary_leakage"] = _finite(
                abs(complex(expectation.value).imag))
            out["var_j2"] = _finite(complex(variance.value).real)
            out["var_j2_imaginary_leakage"] = _finite(
                abs(complex(variance.value).imag))
            out["j2_certificate"] = str(expectation.certificate.grade)
            out["var_j2_certificate"] = str(variance.certificate.grade)
    out["j2_pairing_spread"] = _finite(max(values) - min(values)) if values else None
    out["var_j2_pairing_spread"] = (
        _finite(max(variances) - min(variances)) if variances else None)
    # The dominant honesty caveats, carried on every read.
    out["unpaired_modes"] = modes % 2
    out["trivial_rank1"] = bool(rank == 1 and modes % 2 == 0)
    out["rank1_with_unpaired_mode"] = bool(rank == 1 and modes % 2 == 1)
    variance_value = out.get("var_j2")
    spread = out.get("var_j2_pairing_spread")
    out["var_j2_dominated_by_convention"] = bool(
        spread is not None and variance_value is not None
        and spread > max(abs(variance_value), 1e-12))
    return out


def _vacuum_embedding_defect(state):
    """Inductive compatibility on the covariance: pad Gamma with empty modes.

    Falsifier 7 ("adding vacuum modes changes already-computed amplitudes").
    The embedded state must reproduce every Wick read exactly.
    """
    gamma = np.array(state.gamma())
    modes = gamma.shape[0]
    padded = np.zeros((modes + 2, modes + 2), dtype=complex)
    padded[:modes, :modes] = gamma
    embedded = QU.CovarianceState(padded)
    defects = [
        abs(complex(state.wickTotalNumber().value)
            - complex(embedded.wickTotalNumber().value)),
        abs(complex(state.wickParity().value)
            - complex(embedded.wickParity().value)),
    ]
    if modes >= 2:
        jx, jy, jz = declared_spin_matrices(modes)
        big = np.zeros((modes + 2, modes + 2), dtype=complex)
        reads = []
        for small in (jx, jy, jz):
            padded_j = big.copy()
            padded_j[:modes, :modes] = small
            reads.append(padded_j)
        defects.append(abs(
            complex(state.wickSpinSquaredExpectation(jx, jy, jz).value)
            - complex(embedded.wickSpinSquaredExpectation(*reads).value)))
    return _finite(max(defects))


def _pauli_exclusion_defect(state):
    """The graded (Pauli) identity on the emergent state.

    `wickGramDeterminant(V, V)` with a REPEATED column is a determinant with
    two equal columns, so the graded value is exactly zero. Its ungraded
    counterpart is the permanent of the same 2x2 Gram, which is 2 |v* Gamma v|^2
    and generically nonzero. Returned as (graded, ungraded): the grading is
    working exactly when the first is at round-off and the second is not.
    """
    gamma = np.array(state.gamma())
    modes = gamma.shape[0]
    if modes < 1:
        return None, None
    vector = np.zeros((modes, 1), dtype=complex)
    # The declared probe: the mode carrying the largest occupation.
    vector[int(np.argmax(np.abs(np.diag(gamma)))), 0] = 1.0
    frame = np.hstack([vector, vector])
    graded = abs(complex(state.wickGramDeterminant(frame, frame).value))
    overlap = complex((vector.conj().T @ gamma @ vector)[0, 0])
    ungraded = 2.0 * abs(overlap) ** 2
    return _finite(graded), _finite(ungraded)


def run_member(size, seed, config, commit, config_hash):
    """One (size, seed) ensemble member. Returns its measurement record."""
    started = time.time()
    host = build_host(size, config["host_seed"])
    node = MC(host, [], [], list(config["degrees"]), 1.0, seed)
    node.set_objective(cob.JointStationarityObjective())
    node.set_simulation_mode(MC.SimulationMode.EMERGENCE,
                             MC.EmergenceSubmode.STRICT)
    node.set_provenance(config_hash, commit)
    node.set_analysis_config(_overlay_config(
        config["degrees"], [config["analysis_resolution"]]))

    drive_start = time.time()
    trace = list(node.run_stage1(max_steps=1,
                                 n_candidate_moves=config["candidate_moves"]))
    trace += list(node.run_stage2(max_iters=config["stage2_iters"]))
    drive_seconds = time.time() - drive_start

    # TWO overlay passes on the SAME relaxed geometry, both declared:
    #
    #   * the SCAN pass, given the whole ascending resolution scan, so
    #     `PersistentModularity` tracks span adjacent resolutions and the
    #     classifier's `persistence` gate is reachable at all; and
    #   * the ANALYSIS pass, given the single declared analysis resolution
    #     gamma = 1, whose component layer is the one every Python-side read
    #     and every aggregate below is taken on.
    #
    # Both are recorded. Since #808 the `persistence` certificate reads the
    # COBORDISM-FRAME lifetime, not the modularity resolution-slice count, so
    # a single-resolution pass no longer makes that gate structurally
    # unpassable; the two passes are still both recorded, because the
    # resolution-slice numbers are a real diagnostic of how stable the
    # modularity PROPOSAL is under the resolution parameter.
    node.set_analysis_config(_overlay_config(config["degrees"],
                                             config["resolution_scan"]))
    node.run_recursive_analysis()
    scan_doc = json.loads(node.checkpoint_json)

    node.set_analysis_config(_overlay_config(
        config["degrees"], [config["analysis_resolution"]]))
    analysis_start = time.time()
    node.run_recursive_analysis()
    analysis_seconds = time.time() - analysis_start
    checkpoint = node.checkpoint_json
    doc = json.loads(checkpoint)
    spacetime = node.st

    record = {
        "size": size,
        "seed": seed,
        "config_hash": config_hash,
        "commit": commit,
        "cells": len(spacetime.getTopSimplices()),
        "edges": len(spacetime.getEdgeList().toVector()),
        "vertices": len(spacetime.getVertexList().toVector()),
        "objective_trace": [_finite(v) for v in trace],
        "drive_seconds": drive_seconds,
        "analysis_seconds": analysis_seconds,
        "checkpoint": doc,
    }

    # ---- objective, stationarity residuals, refinement -----------------
    objective = doc["objective"]
    refinement = doc["refinement"]
    record["stationarity"] = {
        "objective_total": _finite(objective["total"]),
        "regge_residual": _finite(objective["regge_stationarity"]),
        "hodge_residual": _finite(objective["hodge_stationarity"]),
        "joint_residual": _finite(
            (objective["regge_stationarity"] or 0.0)
            + (objective["hodge_stationarity"] or 0.0)),
        "curvature_concentration": _finite(
            refinement["curvature_concentration"]),
        "mesh_quality": _finite(refinement["mesh_quality"]),
        "solver_error": _finite(refinement["solver_error"]),
        "refine": refinement["refine"],
        "refine_trigger": refinement["trigger"] or None,
    }

    # ---- hierarchy depth, modularity, persistence ----------------------
    # The SCAN measures resolution dependence and persistence; the component
    # layer everything downstream is read on is the DECLARED analysis
    # resolution (gamma = 1, Newman-Girvan).
    modularity = T.PersistentModularity.fromSpacetime(spacetime)
    mod_config = T.PersistentModularityConfig()
    mod_config.resolutions = list(config["resolution_scan"])
    mod_config.baseSeed = seed
    report = modularity.scanResolutions(mod_config)
    slices = []
    for slice_read in report.slices:
        slices.append({
            "gamma": _finite(slice_read.gamma),
            "q": _finite(slice_read.objectiveValue),
            "q_complex": _complex_pair(slice_read.q),
            "levels": int(slice_read.levels),
            "restart_spread": _finite(slice_read.restartSpread),
            "components": len(slice_read.components),
            "hierarchy_sizes": [len(level) for level in slice_read.hierarchy],
        })
    record["hierarchy"] = {
        "slices": slices,
        "max_depth": max((s["levels"] for s in slices), default=0),
        "tracks": [
            {
                "lifetime": int(track.lastSlice - track.firstSlice + 1),
                "min_adjacent_overlap": _finite(track.minAdjacentOverlap),
                "mean_conductance": _finite(track.meanConductance),
                "members": len(track.members),
            }
            for track in report.tracks
        ],
    }

    analysis_config = T.PersistentModularityConfig()
    analysis_config.resolutions = [config["analysis_resolution"]]
    analysis_config.baseSeed = seed
    analysis_report = modularity.scanResolutions(analysis_config)
    components = (analysis_report.slices[0].components
                  if analysis_report.slices else [])
    record["analysis_resolution"] = config["analysis_resolution"]
    analysis_slice = (analysis_report.slices[0]
                      if analysis_report.slices else None)
    record["analysis_modularity"] = {
        "gamma": config["analysis_resolution"],
        "q": (_finite(analysis_slice.objectiveValue)
              if analysis_slice else None),
        "q_complex": (_complex_pair(analysis_slice.q)
                      if analysis_slice else None),
        "levels": int(analysis_slice.levels) if analysis_slice else None,
        "restart_spread": (_finite(analysis_slice.restartSpread)
                           if analysis_slice else None),
        "components": len(components),
        "hierarchy_sizes": ([len(level) for level in analysis_slice.hierarchy]
                            if analysis_slice else []),
    }
    record["components"] = [
        {
            "volume": len(component.support),
            "strength": _complex_pair(component.strength),
            "conductance": _finite(component.conductance),
            "internal_weight": _complex_pair(component.internalWeight),
            "modularity_contribution": _complex_pair(
                component.modularityContribution),
        }
        for component in components
    ]

    # ---- spectral bands, Krein signatures, degeneracy -------------------
    fiber_config = T.SpectralFiberConfig()
    fiber_config.degrees = list(config["degrees"])
    tracker = T.SpectralFiberTracker(spacetime, fiber_config)
    band_reads = []
    bands = []
    all_eigenvalues = []
    for component in components:
        for degree in config["degrees"]:
            band_read = tracker.enumerateBands(component.support, degree)
            band_reads.append(band_read)
            all_eigenvalues.extend(list(band_read.coveredEigenvalues))
            for fiber in band_read.fibers:
                certificate = fiber.certificate()
                bands.append({
                    "degree": int(fiber.degree()),
                    "rank": int(fiber.rank()),
                    "accepted": bool(fiber.accepted()),
                    "lower_gap": _finite(certificate.lowerGap),
                    "upper_gap": _finite(certificate.upperGap),
                    "localization": _finite(certificate.localization),
                    "gram_defect": _finite(certificate.gramDefect),
                    "eigen_residual": _finite(certificate.eigenResidual),
                    "projector_residual": _finite(
                        certificate.projectorResidual),
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
                })
    accepted_bands = [b for b in bands if b["accepted"]]
    record["bands"] = {
        "total": len(bands),
        "accepted": len(accepted_bands),
        "rank_histogram": _histogram(b["rank"] for b in bands),
        "accepted_rank_histogram": _histogram(
            b["rank"] for b in accepted_bands),
        "rank_three_accepted": sum(1 for b in accepted_bands
                                   if b["rank"] == 3),
        "gap_min": _finite(min((b["lower_gap"] for b in accepted_bands
                                if b["lower_gap"] is not None), default=None)),
        "krein_indefinite": sum(1 for b in bands if b["krein_negative"] > 0),
        "self_adjoint": sum(1 for b in bands if b["self_adjoint"]),
        "regimes": sorted({str(r.regime) for r in band_reads}),
        "solver_paths": sorted({str(r.solverPath) for r in band_reads}),
        "degeneracy_histogram": _degeneracy_histogram(
            all_eigenvalues, config["degeneracy_tolerance"]),
        "degeneracy_histograms": {
            repr(tolerance): _degeneracy_histogram(all_eigenvalues, tolerance)
            for tolerance in config["degeneracy_tolerances"]},
        "detail": bands,
    }

    # ---- the recursive reduction: static, shifted, AMLS, network -------
    record["reduction"] = _reduction_reads(spacetime, components,
                                           config, all_eigenvalues)

    # ---- amplitudes: labeled fiber sum + inductive compatibility -------
    record["amplitudes"] = _amplitude_reads(doc, band_reads, config)

    # ---- transports, holonomies, winding -------------------------------
    record["gauge"] = _gauge_reads(spacetime, band_reads, doc)

    # ---- statistics: exchange, rotation, spin lift ---------------------
    record["statistics"] = _statistics_reads(band_reads)

    # ---- particles, charge, verdict stability --------------------------
    record["particles"] = _particle_reads(doc)
    record["particles_resolution_scan"] = _particle_reads(scan_doc)

    # ---- the covariance-only dichotomy reads ---------------------------
    record["covariance"] = _covariance_reads(band_reads, doc)

    # ---- spectral dimension on the interaction-history complex ---------
    top_k = len(next(iter(spacetime.getTopSimplices())).getVertices()) - 1
    curve = list(spacetime.getSpectralDimensionOnSkeleton(
        list(config["sigmas"]), config["krylov_dim"],
        T.AllSimplexFilter(), top_k, 1))
    peak = max(curve) if curve else None
    record["spectral_dimension"] = {
        "sigmas": [_finite(s) for s in config["sigmas"]],
        "curve": [_finite(v) for v in curve],
        "peak": _finite(peak),
        "peak_sigma": _finite(config["sigmas"][curve.index(peak)])
        if peak is not None else None,
        "top_k": top_k,
        "baseline": PINNED_DS_BASELINE,
        "baseline_sigma": PINNED_DS_BASELINE_SIGMA,
        "deviation_from_baseline": (
            _finite(peak - PINNED_DS_BASELINE) if peak is not None else None),
        "baseline_source":
            "docs/source/quantum-experiments/overview/h_ds4_status.md",
    }

    record["wall_seconds"] = time.time() - started
    return record


def _histogram(values):
    out = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _reduction_reads(spacetime, components, config, eigenvalues):
    """Static / shifted / AMLS reduction and the response-network type."""
    out = {
        "built": False, "reason": None, "regime": None,
        "static": None, "shifted": [], "amls": None,
        "response_network": None, "realization": None,
    }
    if not components:
        out["reason"] = "no components discovered"
        return out
    supports = [list(component.support) for component in components]
    degree = config["degrees"][0]
    try:
        quotient = cob.RecursiveQuotient.overVertexSupports(
            spacetime, degree, supports, cob.RecursiveQuotient.Options())
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
        "certificate_residual": _finite(static.certificate.residual),
        "certificate_tolerance": _finite(static.certificate.tolerance),
    }

    scale = max((abs(complex(value)) for value in eigenvalues), default=0.0)
    if scale <= 0.0:
        scale = 1.0
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
        amls = quotient.craigBampton(0.0, scale,
                                     config["amls_mode_cutoff"], -1.0)
        out["amls"] = {
            "available": True,
            "retained_modes": int(amls.retainedModes),
            "discarded_mode_gap": _finite(amls.discardedModeGap),
            "max_eigen_residual": _finite(
                max((abs(v) for v in amls.eigenResiduals), default=None)),
            "certificate_grade": str(amls.certificate.grade),
            "certificate_holds": bool(amls.certificate.holds()),
        }
    except Exception as error:                            # noqa: BLE001
        out["amls"] = {"available": False,
                       "reason": f"{type(error).__name__}: {error}",
                       "retained_modes": None,
                       "max_eigen_residual": None}

    try:
        network = quotient.responseNetwork()
        out["response_network"] = {
            "stalk_dimensions": [int(d) for d in network.stalkDimensions],
            "edges": len(network.edges),
            "coverage_residual": _finite(network.coverageResidual),
            "certificate_grade": str(network.certificate.grade),
            "certificate_holds": bool(network.certificate.holds()),
        }
    except Exception as error:                            # noqa: BLE001
        out["response_network"] = {
            "reason": f"{type(error).__name__}: {error}"}

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
            # reconstruction residual. The struct's default is 0.0, which
            # would read as "exact"; unknown is null here, never zero.
            "reconstruction_residual": (
                _finite(sheaf.reconstructionResidual) if sheaf.emitted
                else None),
            "certificate_grade": str(sheaf.certificate.grade),
            "certificate_holds": bool(sheaf.certificate.holds()),
        }
    except Exception as error:                            # noqa: BLE001
        out["realization"] = {"type": None,
                              "reason": f"{type(error).__name__}: {error}"}
    return out


def _amplitude_reads(doc, band_reads, config):
    """Amplitude Gram defect and inductive-embedding compatibility."""
    sums = doc.get("labeled_fiber_sums", [])
    out = {
        "labeled_fiber_sums": [
            {
                "degree": entry["degree"],
                "nominal_rank": entry["nominal_rank"],
                "effective_rank": entry["effective_rank"],
                "gram_defect": _finite(entry["gram_defect"]),
                "quotient_nullity": entry["quotient_nullity"],
                "certificate_grade": entry["certificate"]["grade"],
                "certificate_holds": entry["certificate"]["holds"],
            }
            for entry in sums
        ],
        "gram_defect_max": _finite(
            max((entry["gram_defect"] for entry in sums
                 if entry["gram_defect"] is not None), default=None)),
    }
    # The Gram of the retained-fiber embedding is G = J* W J and the defect is
    # ||G - I||. With an all-positive weight diagonal that is an exact
    # isometry; a single negative (Krein) weight flips one diagonal entry to
    # -1 and puts the defect at 2. The regime label records WHICH of those two
    # the run landed in instead of averaging across a bimodal quantity.
    worst = out["gram_defect_max"]
    if worst is None:
        out["gram_defect_regime"] = None
    elif worst < 1e-12:
        out["gram_defect_regime"] = "isometric"
    elif abs(worst - 2.0) < 1e-1:
        out["gram_defect_regime"] = "signature_flipped"
    else:
        out["gram_defect_regime"] = "intermediate"
    # Inductive compatibility: the vacuum embedding must preserve every
    # amplitude exactly (falsifier 7).
    defects = []
    for band_read in band_reads:
        fiber = _first_accepted(band_read)
        if fiber is None:
            continue
        try:
            state = QU.CovarianceState.fromBandProjector(fiber.projector())
        except Exception:                                 # noqa: BLE001
            continue
        defect = _vacuum_embedding_defect(state)
        if defect is not None:
            defects.append(defect)
    out["vacuum_embedding_defect_max"] = (
        _finite(max(defects)) if defects else None)
    out["vacuum_embedding_states"] = len(defects)
    # The exact composable near-isometry budget identity (#768).
    out["near_isometry_budget_identity"] = _finite(abs(
        cob.RecursiveQuotient.composeNearIsometryBudget(1e-3, 2e-3)
        - (1e-3 + 2e-3 + 1e-3 * 2e-3)))
    return out


def _colour_gate(anchor_profile=None):
    """The triangle-anchor gate the exactness contract requires before any
    colour-specific kernel runs.

    A band with no measured anchor gets a CLOSED gate, so the projective
    representative and the centre lift refuse instead of emitting an
    unlicensed colour datum. That refusal is recorded, not swallowed.
    """
    if anchor_profile is None:
        return T.AnchorGate()
    return T.ColorAnchor.gateFor(anchor_profile)


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


def _gauge_reads(spacetime, band_reads, doc):
    """Transports, the four holonomy channels, and determinant winding."""
    connection = T.FiberConnection()
    candidates = []
    for band_read in band_reads:
        fiber = _first_accepted(band_read)
        if fiber is not None:
            candidates.append(fiber)

    transports = []
    accepted = []
    for i, to_fiber in enumerate(candidates):
        for j, from_fiber in enumerate(candidates):
            if i == j or to_fiber.degree() != from_fiber.degree():
                continue
            if to_fiber.rank() != from_fiber.rank():
                continue
            try:
                read = connection.transportOnSpacetime(
                    spacetime, to_fiber, from_fiber)
            except Exception:                             # noqa: BLE001
                continue
            transports.append({
                "rank": int(read.rank),
                "numerical_rank": int(read.numericalRank),
                "accepted": bool(read.accepted),
                "leakage": _finite(read.leakage),
                "overlap_condition_number": _finite(
                    read.overlapConditionNumber),
                "frame_condition_number": _finite(read.frameConditionNumber),
                "determinant_phase": _complex_pair(read.determinantPhase),
                "rejection_reason": read.rejectionReason or None,
                "regime": str(read.regime),
                "krein_to": [int(read.toPositiveSignature),
                             int(read.toNegativeSignature)],
                "krein_from": [int(read.fromPositiveSignature),
                               int(read.fromNegativeSignature)],
            })
            if read.accepted:
                accepted.append(read)

    out = {
        "candidate_bands": len(candidates),
        "transports": len(transports),
        "accepted_transports": len(accepted),
        "leakage_min": _finite(min((t["leakage"] for t in transports
                                    if t["leakage"] is not None),
                                   default=None)),
        "leakage_max": _finite(max((t["leakage"] for t in transports
                                    if t["leakage"] is not None),
                                   default=None)),
        "rejection_reasons": _string_histogram(
            t["rejection_reason"] for t in transports),
        "detail": transports[:32],
        "holonomy": None,
        "determinant_winding": None,
        "center": None,
    }

    if len(accepted) >= 1:
        try:
            holonomy = connection.holonomy(accepted)
            # No anchor profile is measured in this gauge pass, so the gate is
            # closed and the colour kernels below refuse by contract.
            gate = _colour_gate()
            projective = (
                T.FiberConnection.projectiveRepresentative(
                    holonomy.holonomy, gate)
                if holonomy.unitary and gate.accepted else None)
            out["holonomy"] = {
                "closed": bool(holonomy.closed),
                "rank": int(holonomy.rank),
                "loop_length": int(holonomy.loopLength),
                "normalized_trace": _complex_pair(holonomy.normalizedTrace),
                "determinant": _complex_pair(holonomy.determinant),
                "adjoint_trace": _complex_pair(holonomy.adjointTrace),
                "unitary": bool(holonomy.unitary),
                "unitarity_residual": _finite(holonomy.unitarityResidual),
                "projective_available": projective is not None,
                "certificate_grade": str(holonomy.certificate.grade),
                "certificate_holds": bool(holonomy.certificate.holds()),
            }
        except Exception as error:                        # noqa: BLE001
            out["holonomy"] = {"available": False,
                               "reason": f"{type(error).__name__}: {error}"}
        try:
            winding = connection.closedFamilyWinding(accepted)
            out["determinant_winding"] = {
                "winding": winding.winding,
                "closure": str(winding.windingClosure),
                "closure_defect": _finite(winding.closureDefect),
                "max_phase_step": _finite(winding.maxPhaseStep),
                "invalidation_reason": winding.invalidationReason or None,
                "certificate_holds": bool(winding.certificate.holds()),
            }
        except Exception as error:                        # noqa: BLE001
            out["determinant_winding"] = {
                "winding": None,
                "reason": f"{type(error).__name__}: {error}"}
        centers = {}
        for branch in (0, 1, 2):
            try:
                lift = connection.fundamentalLift(accepted, _colour_gate(),
                                                  branch)
                centers[str(branch)] = {
                    "valid": bool(lift.valid),
                    "center_sector": int(lift.centerSector),
                    "lift_trace": _complex_pair(lift.liftTrace),
                    "det_residual": _finite(lift.detResidual),
                    "invalid_reason": lift.invalidReason or None,
                }
            except Exception as error:                    # noqa: BLE001
                centers[str(branch)] = {
                    "valid": False,
                    "invalid_reason": f"{type(error).__name__}: {error}"}
        out["center"] = centers
    else:
        out["holonomy"] = {
            "available": False,
            "reason": "no accepted derived transport on this complex"}
        out["determinant_winding"] = {
            "winding": None,
            "reason": "no accepted derived transport on this complex"}
        out["center"] = {
            "available": False,
            "reason": "no accepted derived transport on this complex"}
    # What the overlay itself recorded, for cross-check.
    out["checkpoint_transports"] = len(doc.get("transports", []))
    out["checkpoint_accepted_transports"] = sum(
        1 for t in doc.get("transports", []) if t["accepted"])
    return out


def _string_histogram(values):
    out = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _statistics_reads(band_reads):
    """Berry-cancelled exchange/rotation characters and the spin-lift status.

    The 2pi rotation and exchange characters are exact on their DECLARED
    carriers (the canonical transverse spinor frame and the vector control),
    and those are reported as analytic invariants that must stay exact at
    every size. Whether the EMERGENT geometry supplies such a carrier is a
    separate question, answered here honestly: `ExchangeHolonomy` requires the
    loop frame's row count to equal the spinor dimension, and an accepted
    band's frame lives on the component's cells, so no emergent band supplies
    one. That read is `null` with its reason, never a fabricated sign.
    """
    out = {"rotation": {}, "emergent_carrier": None, "spin_lift": None}
    for dimension in DECLARED_ROTATION_DIMENSIONS:
        plane_a, plane_b = DECLARED_ROTATION_PLANE
        entry = {}
        spinor = np.array(EH.transverseSpinorFrame(plane_a, plane_b,
                                                   dimension))
        rows = spinor.shape[0]
        weights = np.ones(rows, dtype=complex)
        rotation = EH.rotationLoopFrames(spinor, plane_a, plane_b, dimension,
                                         1, DECLARED_ROTATION_STEPS)
        reference = EH.referenceLoopFrames(spinor, DECLARED_ROTATION_STEPS)
        character = EH.rotationCharacter(EH.loopHolonomy(rotation, weights),
                                         EH.loopHolonomy(reference, weights))
        entry["spinor_character"] = _complex_pair(character.character)
        entry["spinor_sign"] = int(character.characterSign)
        entry["spinor_certificate_holds"] = bool(
            character.certificate.holds())
        vector0 = np.eye(dimension, dtype=complex)[:, :1]
        vector_rotation = EH.vectorLoopFrames(vector0, plane_a, plane_b,
                                              dimension, 1,
                                              DECLARED_ROTATION_STEPS)
        vector_reference = EH.referenceLoopFrames(vector0,
                                                  DECLARED_ROTATION_STEPS)
        vector_weights = np.ones(dimension, dtype=complex)
        vector_character = EH.rotationCharacter(
            EH.loopHolonomy(vector_rotation, vector_weights),
            EH.loopHolonomy(vector_reference, vector_weights))
        entry["vector_character"] = _complex_pair(vector_character.character)
        entry["vector_sign"] = int(vector_character.characterSign)
        # The exact double-cover identity exp(2 pi Sigma_ab) = -I.
        matrix = np.array(EH.spinorRotation(2.0 * math.pi, plane_a, plane_b,
                                            dimension))
        entry["double_cover_residual"] = _finite(
            float(np.abs(matrix + np.eye(matrix.shape[0])).max()))
        out["rotation"][str(dimension)] = entry

    # Can any emergent accepted band supply a spinor carrier?
    reason = None
    supplied = False
    for band_read in band_reads:
        fiber = _first_accepted(band_read)
        if fiber is None:
            continue
        rows = np.array(fiber.rightFrame()).shape[0]
        if rows in (EH.spinorDimension(3), EH.spinorDimension(4)):
            supplied = True
            break
    if not supplied:
        reason = ("no accepted band frame has the spinor row count; the "
                  "physical 2pi rotation certificate is not applicable to "
                  "the emergent carrier at this scale")
    out["emergent_carrier"] = {"supplied": supplied, "reason": reason}
    out["spin_lift"] = {
        "status": None,
        "reason": ("no emergent tangent-frame atlas: ExchangeHolonomy.spinLift "
                   "needs Cech SO(d) edge rotations over a cover, and the "
                   "relaxed complex supplies none at this scale"),
    }
    return out


#: The ONE proton certificate the dichotomy's branch point is about. The
#: whitepaper's premise is "every OTHER certificate is met inside the
#: covariance-only theory, but Var(J^2) fails to converge to zero", and
#: `ParticleClusters::classifyBaryon` reaches its own
#: `quasi-free-sharp-spin-obstruction` verdict when this is the ONLY failure.
#: A read whose failures are a subset of {this} is one whose non-spin
#: certificates all hold.
SHARP_SPIN_CERTIFICATE = "sharp-spin"


def _proton_verdict(baryon):
    """One BaryonRead record, reduced to what the branch point reads.

    `all_non_spin_certificates_hold` is the whitepaper's premise, taken from
    the classifier's own named failures: every certificate except the sharp
    total-space spin one held. It is never inferred from a missing read --
    an absent Var(J^2) is a NAMED `sharp-spin` failure, and an absent
    anything else keeps the premise false.
    """
    failed = set(baryon.get("failed_certificates") or [])
    return {
        "size": None,
        "seed": None,
        "bound_component": baryon.get("bound_component"),
        "classification": baryon.get("classification"),
        "confidence": _finite(baryon.get("confidence")),
        "failed_certificates": sorted(failed),
        "all_non_spin_certificates_hold": failed <= {SHARP_SPIN_CERTIFICATE},
        "sharp_spin": baryon.get("sharp_spin"),
        "total_j2": baryon.get("total_j2"),
        "total_j2_variance": baryon.get("total_j2_variance"),
        "quasi_free_class_swept": baryon.get("quasi_free_class_swept"),
        "class_variance_floor": _finite(baryon.get("class_variance_floor")),
    }


def _particle_reads(doc):
    """Verdicts, charge, anchoring, and the first-failing-certificate map."""
    quarks = doc["particles"]["quarks"]
    bindings = doc["particles"]["bound_supercomponents"]
    baryons = doc["particles"]["baryons"]
    first_failure = {}
    for quark in quarks:
        failed = quark.get("failed_certificates") or []
        key = failed[0] if failed else "none"
        first_failure[key] = first_failure.get(key, 0) + 1
    return {
        "quark_reads": len(quarks),
        "classifications": _string_histogram(
            q["classification"] for q in quarks),
        "first_failing_certificate": dict(sorted(first_failure.items())),
        "all_failing_certificates": _string_histogram(
            name for q in quarks for name in (q.get("failed_certificates") or [])),
        "triangle_anchor_scores": [_finite(q["triangle_anchor_score"])
                                   for q in quarks],
        "anchored": sum(1 for q in quarks
                        if q["triangle_anchor_score"] is not None),
        "determinant_windings": [q["determinant_winding"] for q in quarks],
        "baryon_flux": [q["baryon_flux"] for q in quarks],
        "electric_flux": [q["electric_flux"] for q in quarks],
        "isospin": [q["isospin"] for q in quarks],
        "confidence": [_finite(q["confidence"]) for q in quarks],
        # The §16.2 SEARCH layer: one record per next-level component that
        # contains at least one CERTIFIED quark candidate.
        "bound_supercomponents": len(bindings),
        "bound_supercomponents_found": sum(1 for b in bindings if b["found"]),
        "bound_supercomponent_failed_certificates": _string_histogram(
            name for b in bindings
            for name in (b.get("failed_certificates") or [])),
        # The §16.4 VERDICT layer: one BaryonRead per binding that grouped
        # exactly three certified constituents.
        "baryons": len(baryons),
        "baryon_classifications": _string_histogram(
            b["classification"] for b in baryons),
        "baryon_failed_certificates": _string_histogram(
            name for b in baryons for name in (b.get("failed_certificates") or [])),
        "proton_verdicts": [_proton_verdict(b) for b in baryons],
        # Counted over the baryon reads the overlay ACTUALLY produced -- a
        # measurement, with its denominator (`baryons`) beside it. Zero
        # baryon reads means the bound-supercomponent search grouped three
        # certified constituents nowhere, which is itself a measured result
        # and not an unrun classifier.
        "certified_protons": sum(1 for b in baryons
                                 if b["classification"] == "certified-proton"),
        "certified_proton_candidates": sum(
            1 for b in baryons if b["classification"] == "baryon-candidate"),
        "sharp_spin_obstructions": sum(
            1 for b in baryons
            if b["classification"] == "quasi-free-sharp-spin-obstruction"),
    }


def _covariance_reads(band_reads, doc):
    """The quasi-free layer: purity, spin, Pauli grading, per-candidate."""
    entries = []
    for band_read in band_reads:
        fiber = _first_accepted(band_read)
        if fiber is None:
            continue
        try:
            state = QU.CovarianceState.fromBandProjector(fiber.projector())
        except Exception as error:                        # noqa: BLE001
            entries.append({"available": False,
                            "reason": f"{type(error).__name__}: {error}"})
            continue
        graded, ungraded = _pauli_exclusion_defect(state)
        entry = {
            "available": True,
            "rank": int(fiber.rank()),
            "modes": int(state.modeCount()),
            "purity_defect": _finite(state.purityDefect()),
            "occupation": _finite(complex(state.wickTotalNumber().value).real),
            "parity": _finite(complex(state.wickParity().value).real),
            "pauli_graded": graded,
            "pauli_ungraded": ungraded,
        }
        entry.update(_spin_reads(state, int(fiber.rank())))
        entries.append(entry)
    checkpoint_covariance = doc.get("covariance", {})
    return {
        "states": entries,
        "purity_defect_max": _finite(
            max((e.get("purity_defect") for e in entries
                 if e.get("purity_defect") is not None), default=None)),
        "checkpoint_purity_defect": _finite(
            checkpoint_covariance.get("purity_defect")),
        "checkpoint_active_modes": checkpoint_covariance.get("active_modes"),
    }


# =====================================================================
# negative controls — each must FIRE. One that silently passes is a bug
# in the instrument, and is reported as such.
# =====================================================================

def _control(name, expectation, fired, detail):
    return {"name": name, "expectation": expectation,
            "fired": bool(fired), "detail": detail}


def _certified_read_vector(spacetime, components, tracker, config):
    """The DECLARED summary of what the certificate layer reads on a complex.

    One vector, computed identically on a relaxed complex and on a destroyed
    one, so a control compares like with like.
    """
    bands = 0
    accepted = 0
    localization = []
    gaps = []
    modularity_q = None
    for component in components:
        band_read = tracker.enumerateBands(component.support,
                                           config["degrees"][0])
        bands += len(band_read.fibers)
        for fiber in band_read.fibers:
            certificate = fiber.certificate()
            if fiber.accepted():
                accepted += 1
                value = _finite(certificate.localization)
                if value is not None:
                    localization.append(value)
            for gap in (certificate.lowerGap, certificate.upperGap):
                value = _finite(gap)
                if value is not None:
                    gaps.append(value)
    try:
        graph = T.PersistentModularity.fromSpacetime(spacetime)
        ids = list(graph.cellIds())
        index_of = {cell: i for i, cell in enumerate(ids)}
        labels = [0] * len(ids)
        for label, component in enumerate(components):
            for cell in component.support:
                if cell in index_of:
                    labels[index_of[cell]] = label
        modularity_q = _score_scalar(graph.modularityGamma(
            labels, config["analysis_resolution"]))
    except Exception:                                     # noqa: BLE001
        modularity_q = None
    return {
        "components": float(len(components)),
        "bands": float(bands),
        "accepted_bands": float(accepted),
        "accepted_fraction": (accepted / bands) if bands else None,
        "mean_localization": mean_sd(localization)["mean"],
        "mean_gap": mean_sd(gaps)["mean"],
        "modularity_q": modularity_q,
    }


def negative_controls(size, config):
    """The eight mandated controls, each with its expected failure."""
    controls = []
    rng = np.random.default_rng(DECLARED_CONTROL_SEED)

    # --- shared relaxed host for the geometry-side controls --------------
    host = build_host(size, config["host_seed"])
    node = MC(host, [], [], list(config["degrees"]), 1.0,
              config["seeds"][0])
    node.set_objective(cob.JointStationarityObjective())
    list(node.run_stage1(max_steps=1,
                         n_candidate_moves=config["candidate_moves"]))
    list(node.run_stage2(max_iters=config["stage2_iters"]))
    spacetime = node.st
    mod_config = T.PersistentModularityConfig()
    mod_config.resolutions = [config["analysis_resolution"]]
    mod_config.baseSeed = config["seeds"][0]
    modularity = T.PersistentModularity.fromSpacetime(spacetime)
    report = modularity.scanResolutions(mod_config)
    components = report.slices[0].components if report.slices else []
    fiber_config = T.SpectralFiberConfig()
    fiber_config.degrees = list(config["degrees"])
    tracker = T.SpectralFiberTracker(spacetime, fiber_config)
    baseline_reads = _certified_read_vector(spacetime, components, tracker,
                                            config)
    baseline_accepted = baseline_reads["accepted_bands"]

    # --- 1. shuffled phases ---------------------------------------------
    # The certified reads must NOT be blind to the geometry: randomizing every
    # complex edge phase on a relaxed complex has to move the declared read
    # vector. Each component of that vector is reported with its own relative
    # change, so a read that turns out to carry no phase information at all is
    # visible as such instead of hiding inside a single boolean.
    shuffled_host = build_host(size, config["host_seed"])
    shuffled_node = MC(shuffled_host, [], [], list(config["degrees"]), 1.0,
                       config["seeds"][0])
    shuffled_node.set_objective(cob.JointStationarityObjective())
    list(shuffled_node.run_stage1(
        max_steps=1, n_candidate_moves=config["candidate_moves"]))
    list(shuffled_node.run_stage2(max_iters=config["stage2_iters"]))
    shuffled = shuffled_node.st
    lengths = [edge.getLength() for edge in shuffled.getEdgeList().toVector()]
    phases = rng.uniform(0.0, 2.0 * math.pi, size=len(lengths))
    for edge, length, phase in zip(shuffled.getEdgeList().toVector(),
                                   lengths, phases):
        edge.setLength(abs(length) * cmath.exp(1j * float(phase)))
    shuffled_components = (T.PersistentModularity.fromSpacetime(shuffled)
                           .scanResolutions(mod_config).slices)
    shuffled_components = (shuffled_components[0].components
                           if shuffled_components else [])
    shuffled_tracker = T.SpectralFiberTracker(shuffled, fiber_config)
    shuffled_reads = _certified_read_vector(shuffled, shuffled_components,
                                            shuffled_tracker, config)
    changes = {}
    moved = False
    for name, baseline_value in baseline_reads.items():
        other = shuffled_reads.get(name)
        if baseline_value is None or other is None:
            changes[name] = None
            continue
        scale = max(abs(baseline_value), abs(other), 1.0)
        relative = abs(other - baseline_value) / scale
        changes[name] = _finite(relative)
        if relative > 1e-9:
            moved = True
    controls.append(_control(
        "shuffled_phases",
        "randomizing every complex edge phase changes the certified read "
        "vector; a read that does not move carries no phase information and "
        "is named",
        moved,
        {"baseline": {k: _finite(v) for k, v in baseline_reads.items()},
         "shuffled": {k: _finite(v) for k, v in shuffled_reads.items()},
         "relative_change": changes,
         "unmoved_reads": sorted(k for k, v in changes.items()
                                 if v is not None and v <= 1e-9)}))

    # --- 2. destroyed modularity ----------------------------------------
    cell_ids = list(modularity.cellIds())
    true_labels = [0] * len(cell_ids)
    index_of = {cell: i for i, cell in enumerate(cell_ids)}
    for label, component in enumerate(components):
        for cell in component.support:
            if cell in index_of:
                true_labels[index_of[cell]] = label
    true_q = modularity.modularityGamma(true_labels, 1.0)
    random_labels = list(true_labels)
    rng.shuffle(random_labels)
    random_q = modularity.modularityGamma(random_labels, 1.0)
    controls.append(_control(
        "destroyed_modularity",
        "randomly permuting the discovered partition's labels collapses Q",
        _score_scalar(random_q) < _score_scalar(true_q),
        {"discovered_q": _score_scalar(true_q),
         "shuffled_q": _score_scalar(random_q),
         "discovered_q_complex": _complex_pair(true_q),
         "shuffled_q_complex": _complex_pair(random_q),
         "cells": len(cell_ids)}))

    # --- 3. modularity resolution-limit graph ----------------------------
    controls.append(_resolution_limit_control())

    # --- 4. unanchored rank-three band -----------------------------------
    controls.append(_unanchored_band_control())

    # --- 5. closed spectral / rank gaps ----------------------------------
    closed_config = T.SpectralFiberConfig()
    closed_config.degrees = list(config["degrees"])
    closed_config.minRelativeGap = 1e9
    closed_config.gapDominance = 1e9
    closed_tracker = T.SpectralFiberTracker(spacetime, closed_config)
    closed_accepted = 0
    for component in components:
        band_read = closed_tracker.enumerateBands(component.support,
                                                  config["degrees"][0])
        closed_accepted += sum(1 for f in band_read.fibers if f.accepted())
    controls.append(_control(
        "closed_spectral_and_rank_gaps",
        "forcing the isolation floor past any achievable gap accepts no band",
        closed_accepted == 0 and baseline_accepted > 0,
        {"baseline_accepted": baseline_accepted,
         "closed_accepted": closed_accepted}))

    # --- 6. cube-root branch changes -------------------------------------
    controls.append(_cube_root_branch_control(rng))

    # --- 7. uncancelled Berry loops --------------------------------------
    controls.append(_berry_control())

    # --- 8. disabled grading ---------------------------------------------
    controls.append(_grading_control(spacetime, components, tracker, config))

    return controls


def _resolution_limit_control():
    """A ring of cliques: modularity at gamma = 1 provably MERGES cliques.

    The Fortunato-Barthelemy resolution limit. Sixteen 4-cliques in a ring,
    joined by single edges: the discovered component count at gamma = 1 is
    below sixteen, and a higher gamma resolves more of them. The control
    fires when the merge is observed.
    """
    cliques, clique_size = 16, 4
    src, tgt, weight = [], [], []
    for c in range(cliques):
        base = c * clique_size
        for a in range(clique_size):
            for b in range(a + 1, clique_size):
                src.append(base + a)
                tgt.append(base + b)
                weight.append(1.0)
        nxt = ((c + 1) % cliques) * clique_size
        src.append(base)
        tgt.append(nxt)
        weight.append(1.0)
    graph = T.PersistentModularity.fromWeightedEdges(src, tgt, weight, [])
    config = T.PersistentModularityConfig()
    config.resolutions = [1.0, 4.0]
    config.baseSeed = DECLARED_CONTROL_SEED
    report = graph.scanResolutions(config)
    counts = [len(s.components) for s in report.slices]
    at_one = counts[0] if counts else 0
    at_four = counts[1] if len(counts) > 1 else 0
    return _control(
        "modularity_resolution_limit",
        "a ring of 16 4-cliques is MERGED at gamma = 1 (fewer than 16 "
        "components) and a higher gamma resolves more of them",
        at_one < cliques and at_four > at_one,
        {"cliques": cliques, "components_at_gamma_1": at_one,
         "components_at_gamma_4": at_four})


def _unanchored_band_control():
    """An abstract rank-three frame with no oriented-triangle atlas.

    ColorAnchor cannot be declared on an empty atlas, so the anchor evidence
    is MISSING and no score exists. The control fires when the refusal
    happens and when an atlas whose triangles do not meet the band's support
    scores exactly zero.
    """
    detail = {}
    fired = True
    try:
        T.ColorAnchor([])
        detail["empty_atlas_refused"] = False
        fired = False
    except Exception as error:                            # noqa: BLE001
        detail["empty_atlas_refused"] = True
        detail["empty_atlas_reason"] = f"{type(error).__name__}: {error}"

    # A rank-three frame supported on edges 0..2 of a six-edge complex, and a
    # declared triangle on edges 3,4,5 — the band is not anchored there.
    edges = 6
    frame = np.zeros((edges, 3), dtype=complex)
    frame[0, 0] = frame[1, 1] = frame[2, 2] = 1.0
    weights = np.ones(edges, dtype=float)
    triangle = T.OrientedTriangle([3, 4, 5], [1, -1, 1])
    anchor = T.ColorAnchor([triangle])
    profile = anchor.evaluate(frame, weights)
    detail["disjoint_atlas_score"] = _finite(profile.score)
    detail["disjoint_atlas_max_term"] = _finite(profile.max_term)
    fired = fired and abs(profile.score) < 1e-14
    return _control(
        "unanchored_rank_three_band",
        "an empty atlas is refused, and a rank-three band whose support "
        "misses the declared triangles scores exactly zero",
        fired, detail)


def _cube_root_branch_control(rng):
    """Cube-root branch: projective observables agree, center lifts differ."""
    matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    unitary, _ = np.linalg.qr(matrix)
    unitary = unitary / np.linalg.det(unitary) ** (1.0 / 3.0)
    representatives = []
    adjoints = []
    gate = _declared_anchor_gate()
    for factor in (1.0, cmath.exp(2j * math.pi / 3.0),
                   cmath.exp(4j * math.pi / 3.0)):
        branch = unitary * factor
        representatives.append(
            np.array(T.FiberConnection.projectiveRepresentative(branch, gate)))
        adjoints.append(
            np.array(T.FiberConnection.adjointRepresentation(branch)))
    adjoint_spread = max(float(np.abs(adjoints[0] - other).max())
                         for other in adjoints[1:])
    center_spread = max(
        float(np.abs(representatives[0] - other).max())
        for other in representatives[1:])
    return _control(
        "cube_root_branch_change",
        "the adjoint (PU(3)) image is branch-independent while the "
        "fundamental representatives expose distinct center lifts",
        adjoint_spread < 1e-10 and center_spread > 1e-6,
        {"adjoint_spread": _finite(adjoint_spread),
         "fundamental_center_spread": _finite(center_spread)})


def _berry_control():
    """The raw loop determinant carries the Berry phase and is NOT a sign."""
    steps, cells, omega = 8, 8, 0.7

    def mode(position):
        k = int(math.floor(position)) % cells
        fraction = position - math.floor(position)
        vector = np.zeros(cells, dtype=complex)
        vector[k] += math.cos(fraction * math.pi / 2.0)
        vector[(k + 1) % cells] += math.sin(fraction * math.pi / 2.0)
        return vector

    frames = []
    for t in range(steps):
        shift = 4.0 * t / steps
        frames.append(np.stack([mode((0 + shift) % cells),
                                mode((4 + shift) % cells)], axis=1))
    weights = math.e ** (1j * omega) * np.ones(cells, dtype=complex)
    loop = EH.loopHolonomy(frames, weights)
    reference = EH.loopHolonomy([frames[0]] * steps, weights)
    character = EH.exchangeCharacter(loop, reference)
    raw = complex(loop.determinant)
    raw_is_sign = min(abs(raw - 1.0), abs(raw + 1.0)) < 1e-6
    cancelled = complex(character.character)
    return _control(
        "uncancelled_berry_loop",
        "the RAW exchange-loop determinant is not +-1 while the "
        "reference-cancelled character is exactly -1",
        (not raw_is_sign) and abs(cancelled + 1.0) < 1e-12,
        {"raw_determinant": _complex_pair(raw),
         "cancelled_character": _complex_pair(cancelled),
         "character_sign": int(character.characterSign),
         "certificate_holds": bool(character.certificate.holds())})


def _grading_control(spacetime, components, tracker, config):
    """Grading on/off on the EMERGENT covariance state.

    The graded amplitude `<a*(v) a*(v) a(v) a(v)> = det(V* Gamma V)` with a
    repeated column is a determinant with two equal columns: exactly zero
    (Pauli exclusion). Its ungraded counterpart is the permanent of the same
    Gram, 2 |v* Gamma v|^2, which is generically nonzero. The control fires
    when the shipped graded read is at round-off and the ungraded one is not.
    """
    for component in components:
        band_read = tracker.enumerateBands(component.support,
                                           config["degrees"][0])
        fiber = _first_accepted(band_read)
        if fiber is None:
            continue
        state = QU.CovarianceState.fromBandProjector(fiber.projector())
        graded, ungraded = _pauli_exclusion_defect(state)
        if graded is None:
            continue
        return _control(
            "disabled_grading",
            "the graded (exterior) amplitude with a repeated mode is exactly "
            "zero by Pauli exclusion, while the ungraded (permanent) "
            "counterpart is not",
            graded < 1e-12 and ungraded > 1e-6,
            {"graded_amplitude": graded, "ungraded_amplitude": ungraded})
    return _control(
        "disabled_grading",
        "the graded amplitude with a repeated mode is exactly zero",
        False,
        {"reason": "no accepted band supplied a covariance state"})


# =====================================================================
# analytic invariants — the small exact fixtures that must stay exact
# =====================================================================

def analytic_invariants():
    """Exact identities that must hold at machine precision, every run."""
    out = []

    def record(name, residual, tolerance, detail=None):
        out.append({"name": name, "residual": _finite(residual),
                    "tolerance": tolerance,
                    "exact": bool(residual is not None
                                  and residual <= tolerance),
                    "detail": detail})

    # F_3 unitary with unit-modulus determinant.
    frame = np.array(T.ColorFiber.fourierFrame())
    record("fourier_frame_unitary",
           float(np.abs(frame.conj().T @ frame - np.eye(3)).max()), 1e-14)
    record("fourier_frame_unit_determinant",
           abs(abs(complex(np.linalg.det(frame))) - 1.0), 1e-14)

    # The Lambda^3 C^3 singlet Gram is exactly 1 for any SU(3) frame.
    rng = np.random.default_rng(DECLARED_CONTROL_SEED)
    worst = 0.0
    for _ in range(8):
        matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        unitary, _ = np.linalg.qr(matrix)
        unitary = unitary / np.linalg.det(unitary) ** (1.0 / 3.0)
        worst = max(worst,
                    abs(complex(T.ColorFiber.singletGram(unitary)) - 1.0))
    record("su3_singlet_gram_invariance", worst, 1e-12)

    # exp(2 pi Sigma_ab) = -I in d = 3 and d = 4.
    for dimension in DECLARED_ROTATION_DIMENSIONS:
        matrix = np.array(EH.spinorRotation(2.0 * math.pi, 0, 1, dimension))
        record(f"spin_double_cover_d{dimension}",
               float(np.abs(matrix + np.eye(matrix.shape[0])).max()), 1e-14)

    # The single-mode spin-1/2 fixture: <J^2> = 3/4 exactly, Var = 0.
    jx, jy, jz = declared_spin_matrices(2)
    state = QU.CovarianceState.fromOccupations(np.array([1.0, 0.0]))
    record("sharp_spin_half_expectation",
           abs(complex(state.wickSpinSquaredExpectation(jx, jy, jz).value)
               - 0.75), 1e-14)
    record("sharp_spin_half_variance",
           abs(complex(state.wickSpinSquaredVariance(jx, jy, jz).value)),
           1e-12)

    # The mandated NEGATIVE spin fixture: <J^2> = 3/4 with nonzero variance.
    root = 1.0 / math.sqrt(2.0)
    sx1 = np.array([[0, root, 0], [root, 0, root], [0, root, 0]],
                   dtype=complex)
    sy1 = np.array([[0, -1j * root, 0], [1j * root, 0, -1j * root],
                    [0, 1j * root, 0]], dtype=complex)
    sz1 = np.diag([1.0, 0.0, -1.0]).astype(complex)

    def pad(block):
        out_matrix = np.zeros((4, 4), dtype=complex)
        out_matrix[1:, 1:] = block
        return out_matrix

    orbital = np.zeros((4, 1), dtype=complex)
    orbital[0, 0] = math.sqrt(5.0 / 8.0)
    orbital[1, 0] = math.sqrt(3.0 / 8.0)
    generic = QU.CovarianceState.fromSlaterFrame(orbital)
    expectation = complex(generic.wickSpinSquaredExpectation(
        pad(sx1), pad(sy1), pad(sz1)).value)
    variance = complex(generic.wickSpinSquaredVariance(
        pad(sx1), pad(sy1), pad(sz1)).value)
    record("generic_slater_expectation_three_quarters",
           abs(expectation - 0.75), 1e-13)
    record("generic_slater_variance_is_fifteen_sixteenths",
           abs(variance - 15.0 / 16.0), 1e-12,
           {"note": "the right expectation with nonzero variance is NOT a "
                    "certified sharp spin"})

    # The exact composable near-isometry budget.
    record("near_isometry_budget",
           abs(cob.RecursiveQuotient.composeNearIsometryBudget(1e-3, 2e-3)
               - (1e-3 + 2e-3 + 1e-3 * 2e-3)), 1e-18)

    # The Berry-cancelled single exchange is exactly -1.
    berry = _berry_control()
    record("berry_cancelled_single_exchange",
           abs(complex(*berry["detail"]["cancelled_character"]) + 1.0), 1e-12)
    return out


# =====================================================================
# aggregation: the two named studies plus the uncertainty budget
# =====================================================================

def _by_size(runs, extractor):
    grouped = {}
    for run in runs:
        grouped.setdefault(run["size"], []).append(extractor(run))
    return {size: mean_sd(values) for size, values in sorted(grouped.items())}


def _convergence(runs, extractor, name):
    grouped = _by_size(runs, extractor)
    sizes = [s for s, stat in grouped.items() if stat["mean"] is not None]
    means = [grouped[s]["mean"] for s in sizes]
    if len(sizes) < 3:
        return {"observable": name, "by_size": grouped,
                "fit": None, "verdict": "insufficient_sizes"}
    result = inverse_size_fit(sizes, means)
    result.update({"observable": name, "by_size": grouped})
    return result


def proton_verdicts_of(runs):
    """Every BaryonRead the overlay produced across the ensemble.

    Stamped with the member that produced it. An EMPTY list is a real
    measurement -- the overlay ran the §16.2 search on every member and it
    grouped three certified constituents nowhere, so the §16.4 classifier
    had no three-cluster candidate to classify.
    """
    out = []
    for run in runs:
        for verdict in run["particles"].get("proton_verdicts", []):
            entry = dict(verdict)
            entry["size"] = run["size"]
            entry["seed"] = run["seed"]
            out.append(entry)
    return out


def var_j2_zero_limit(fit):
    """Does the accepted class's Var(J^2) extrapolate to ZERO under 1/N?

    TRI-STATE, and ``None`` is not ``False``. The whitepaper's obstruction
    is "Var(J^2) FAILS to converge to zero across refinement", which cannot
    be asserted from a fit that was never made or from one whose own
    verdict says the extrapolation is not to be trusted. The measured
    limit, its standard error, the zero band and the fit's own verdict all
    travel with the answer.

    The statement is about the extrapolated LIMIT, not about the trend: an
    observable can converge beautifully to a nonzero value, and that is the
    obstruction, not the absence of one.
    """
    out = {"converges_to_zero": None,
           "extrapolated_limit": None,
           "extrapolated_limit_se": None,
           "tolerance": DECLARED_VAR_J2_ZERO_TOLERANCE,
           "zero_band": None,
           "fit_verdict": None,
           "reason": None}
    if not fit:
        out["reason"] = (
            "fewer than three sizes carry a Var(J^2) on an accepted "
            "non-trivial band, so no 1/N refinement fit exists and the "
            "limit is UNMEASURED")
        return out
    out["fit_verdict"] = fit.get("verdict")
    limit = _finite(fit.get("extrapolated_limit"))
    se = _finite(fit.get("extrapolated_limit_se"))
    out["extrapolated_limit"] = limit
    out["extrapolated_limit_se"] = se
    if out["fit_verdict"] in ("insufficient_points",
                              "trending_but_not_inverse_size"):
        out["reason"] = (
            f"the 1/N fit reports `{out['fit_verdict']}`, so its "
            "extrapolation is not to be trusted and the limit is UNMEASURED")
        return out
    if limit is None:
        out["reason"] = ("the 1/N fit produced no finite extrapolated "
                         "limit, so the limit is UNMEASURED")
        return out
    # The zero band is the wider of the fit's own two-sigma interval and the
    # declared sharp-spin tolerance: a limit inside it is zero as far as
    # this measurement can tell, and a limit outside it is not.
    band = DECLARED_VAR_J2_ZERO_TOLERANCE
    if se is not None:
        band = max(band, 2.0 * se)
    out["zero_band"] = band
    out["converges_to_zero"] = abs(limit) <= band
    out["reason"] = (
        f"extrapolated limit {limit:.6g} "
        + (f"+- {se:.6g} " if se is not None else "")
        + ("within" if out["converges_to_zero"] else "outside")
        + f" the zero band {band:.6g}, fit verdict `{out['fit_verdict']}`")
    return out


def dichotomy(runs, proton_verdicts=None):
    """The covariance-only dichotomy: Var(J^2) on accepted candidates.

    Three outcomes are possible and exactly one is returned:

    * ``covariance_only_proton`` — every proton certificate holds on an
      accepted candidate class AND Var(J^2) converges to zero;
    * ``quasi_free_sharp_spin_obstruction`` — every OTHER certificate holds
      but Var(J^2) does not converge to zero;
    * ``inconclusive`` — the branch point was not reached. Five distinct
      situations produce it and each names itself in ``reason``: no quark
      candidate was certified at all; this call was given no proton verdict
      list (an INSTRUMENT gap); the overlay ran the baryon classifier but the
      bound-supercomponent search grouped three certified constituents
      nowhere (a measured emptiness); three-cluster candidates were
      classified and none completed the non-spin certificates; or they did
      and the refinement behaviour of Var(J^2) was never measured, so
      neither branch can be claimed.

    Between the two informative outcomes the branch is decided first by the
    CLASSIFIER's own verdict — a ``certified-proton`` carries the sharp-spin
    certificate, a ``quasi-free-sharp-spin-obstruction`` is its negation over
    a SWEPT covariance-only class — and, when the classifier reached neither,
    by whether the accepted class's Var(J^2) extrapolates to zero under 1/N
    refinement (``var_j2_zero_limit``).

    ``proton_verdicts`` is the ensemble's ``BaryonRead`` reductions --
    ``proton_verdicts_of(runs)`` builds it from the overlay's own
    ``particles.baryons`` block. ``None`` means the branch point could not be
    evaluated, and the proton counts stay ``null`` rather than zero.
    """
    certified = []
    trivial = []
    unpaired = []
    convention_dominated = 0
    spreads = []
    for run in runs:
        for state in run["covariance"]["states"]:
            if not state.get("available") or state.get("var_j2") is None:
                continue
            entry = {"size": run["size"], "seed": run["seed"],
                     "rank": state.get("rank"),
                     "modes": state.get("modes"),
                     "unpaired_modes": state.get("unpaired_modes"),
                     "j2": state.get("j2"),
                     "var_j2": state.get("var_j2"),
                     "j2_pairing_spread": state.get("j2_pairing_spread"),
                     "var_j2_pairing_spread": state.get(
                         "var_j2_pairing_spread")}
            if state.get("var_j2_dominated_by_convention"):
                convention_dominated += 1
            if state.get("var_j2_pairing_spread") is not None:
                spreads.append(state["var_j2_pairing_spread"])
            if state.get("trivial_rank1"):
                trivial.append(entry)
            elif state.get("rank1_with_unpaired_mode"):
                unpaired.append(entry)
            else:
                certified.append(entry)

    def failure_map(block):
        counts = {}
        for run in runs:
            for name, count in run[block]["first_failing_certificate"].items():
                counts[name] = counts.get(name, 0) + count
        total = sum(counts.values())
        return {
            "first_failing_certificate": dict(sorted(counts.items())),
            "first_failing_certificate_fraction": {
                name: (count / total if total else None)
                for name, count in sorted(counts.items())},
            "all_failing_certificates": {
                name: sum(run[block]["all_failing_certificates"].get(name, 0)
                          for run in runs)
                for name in sorted({key for run in runs
                                    for key in run[block][
                                        "all_failing_certificates"]})},
            "quark_reads_total": total,
        }

    primary = failure_map("particles")
    scan = failure_map("particles_resolution_scan")
    total_reads = primary["quark_reads_total"]

    accepted_candidates = sum(
        count for run in runs
        for name, count in run["particles"]["classifications"].items()
        if name != "none")

    # The proton counts come from the verdicts THIS call was given. With no
    # verdict list the classifier's output never reached here, so the counts
    # are UNKNOWN (null), never zero -- a zero would claim a proton count
    # that was never computed.
    if proton_verdicts is None:
        certified_protons = None
        certified_proton_candidates = None
        sharp_spin_obstructions = None
    else:
        certified_protons = sum(
            1 for v in proton_verdicts
            if v.get("classification") == "certified-proton")
        certified_proton_candidates = sum(
            1 for v in proton_verdicts
            if v.get("classification") == "baryon-candidate")
        sharp_spin_obstructions = sum(
            1 for v in proton_verdicts
            if v.get("classification")
            == "quasi-free-sharp-spin-obstruction")

    out = {
        "accepted_candidates": accepted_candidates,
        "proton_verdicts": (None if proton_verdicts is None
                            else list(proton_verdicts)),
        "proton_verdicts_total": (None if proton_verdicts is None
                                  else len(proton_verdicts)),
        "certified_protons": certified_protons,
        "certified_proton_candidates": certified_proton_candidates,
        "sharp_spin_obstructions": sharp_spin_obstructions,
        "first_failing_certificate": primary["first_failing_certificate"],
        "first_failing_certificate_fraction":
            primary["first_failing_certificate_fraction"],
        "all_failing_certificates": primary["all_failing_certificates"],
        "resolution_scan_pass": scan,
        "resolution_scan_note": (
            "the `particles` block is read at the declared analysis "
            "resolution gamma = 1; the `resolution_scan_pass` block is the "
            "same classifier fed the whole declared resolution scan.  Since "
            "#808 the `persistence` certificate reads the COBORDISM-FRAME "
            "lifetime, so neither pass can fail it for a reason that is not "
            "physics -- this overlay reads ONE frame, and the reported "
            "lifetime of 1 is a measured fact about the evidence.  The two "
            "passes now differ only in the modularity resolution-slice "
            "diagnostics, which are reported and never gate"),
        "quark_reads_total": total_reads,
        "var_j2_on_nontrivial_bands": certified,
        "var_j2_on_fully_paired_rank1_bands": trivial,
        "var_j2_on_rank1_bands_with_an_unpaired_mode": unpaired,
        "rank1_triviality_note": (
            "with every mode paired, a rank-1 covariance is exactly a "
            "j = 1/2 eigenstate under this convention, so <J^2> = 3/4 and "
            "Var(J^2) = 0 are identities of the READOUT, not evidence of a "
            "proton; with a mode left unpaired the SAME rank-1 state picks up "
            "a spin-0 admixture and a genuinely nonzero Var(J^2), which is "
            "likewise an artifact of the convention"),
        "spin_convention_dominance": {
            "reads_where_the_pairing_spread_exceeds_the_value":
                convention_dominated,
            "reads_total": len(trivial) + len(unpaired) + len(certified),
            "var_j2_pairing_spread": mean_sd(spreads),
            "max_var_j2_pairing_spread": _finite(max(spreads))
            if spreads else None,
            "meaning": (
                "the DECLARED one-particle spin convention is not supplied by "
                "the geometry; the pairing spread is how much Var(J^2) moves "
                "when the arbitrary pairing is shifted by one mode, and where "
                "it exceeds the value itself the read carries no geometric "
                "information at all"),
        },
    }
    if certified:
        sizes = sorted({e["size"] for e in certified})
        by_size = {}
        for size in sizes:
            by_size[size] = mean_sd([e["var_j2"] for e in certified
                                     if e["size"] == size])
        out["var_j2_by_size"] = by_size
        if len(sizes) >= 3:
            out["var_j2_convergence"] = inverse_size_fit(
                sizes, [by_size[s]["mean"] for s in sizes])
    if accepted_candidates == 0:
        out["classification"] = "inconclusive"
        out["reason"] = (
            "no quark candidate was certified at any size or seed, so the "
            "accepted covariance-only class is EMPTY and the dichotomy is "
            "not reached; the first-failing-certificate distribution above "
            "is the result. A second, independent obstruction is recorded in "
            "`spin_convention_dominance`: the geometry supplies no spin "
            "structure, so even a certified candidate could not be given a "
            "Var(J^2) that means anything until one is derived rather than "
            "declared")
    elif proton_verdicts is None:
        out["classification"] = "inconclusive"
        out["reason"] = (
            "quark candidates were classified, but this call was given no "
            "proton verdict list at all, so the branch point cannot be "
            "evaluated here. This is an instrument gap, NOT a measurement "
            "that the dichotomy failed")
    elif not proton_verdicts:
        out["classification"] = "inconclusive"
        out["reason"] = (
            "quark candidates were classified and the analysis overlay ran "
            "the baryon classifier, but the bound-supercomponent search "
            "grouped three certified constituents nowhere, so no "
            "three-cluster candidate was classified and the branch point is "
            "not reached. This is a measured emptiness of the accepted "
            "three-cluster class, not an unrun instrument")
    else:
        # The branch point proper: every OTHER proton certificate holds on an
        # accepted class, and the question is only whether Var(J^2) converges.
        others_hold = [v for v in proton_verdicts
                       if v.get("all_non_spin_certificates_hold")]
        if not others_hold:
            out["classification"] = "inconclusive"
            out["reason"] = (
                f"{len(proton_verdicts)} three-cluster candidate(s) were "
                "classified and none completed the non-spin proton "
                "certificates, so the sharp-spin branch point is not reached")
        else:
            proton_reads = [v for v in others_hold
                            if v.get("classification") == "certified-proton"]
            obstructed = [
                v for v in others_hold
                if v.get("classification")
                == "quasi-free-sharp-spin-obstruction"]
            limit = var_j2_zero_limit(out.get("var_j2_convergence"))
            out["var_j2_zero_limit"] = limit
            out["non_spin_certificates_hold"] = len(others_hold)
            if proton_reads:
                # The CLASSIFIER itself certified the sharp spin on an
                # accepted candidate: every one of its fifteen gates held,
                # Var(J^2) included.
                out["classification"] = "covariance_only_proton"
                out["reason"] = (
                    f"{len(proton_reads)} accepted three-cluster "
                    "candidate(s) carry the COMPLETE proton certificate, "
                    "sharp spin included, so an exact covariance-only proton "
                    "exists in this ensemble")
            elif obstructed:
                # The classifier reached its own branch point: sharp-spin is
                # the ONLY failure and the accepted covariance-only class was
                # swept, so the variance floor is a property of the class.
                out["classification"] = "quasi_free_sharp_spin_obstruction"
                floors = [v.get("class_variance_floor")
                          for v in obstructed
                          if v.get("class_variance_floor") is not None]
                out["reason"] = (
                    f"{len(obstructed)} accepted three-cluster candidate(s) "
                    "hold every proton certificate except the sharp spin, "
                    "over a SWEPT accepted covariance-only class"
                    + (f" whose Var(J^2) floor is {min(floors):.6g}"
                       if floors else "")
                    + "; a non-Gaussian mechanism is required")
            elif limit["converges_to_zero"] is True:
                out["classification"] = "covariance_only_proton"
                out["reason"] = (
                    "every other proton certificate holds on the accepted "
                    "class and Var(J^2) extrapolates to zero under 1/N "
                    "refinement (" + limit["reason"] + ")")
            elif limit["converges_to_zero"] is False:
                out["classification"] = "quasi_free_sharp_spin_obstruction"
                out["reason"] = (
                    "every other proton certificate holds on the accepted "
                    "class but Var(J^2) does not extrapolate to zero under "
                    "1/N refinement (" + limit["reason"] + "); a "
                    "non-Gaussian mechanism is required")
            else:
                # UNMEASURED is not a measured failure: the whitepaper's
                # obstruction is "Var(J^2) FAILS to converge", which no
                # absent fit can establish.
                out["classification"] = "inconclusive"
                out["reason"] = (
                    "every other proton certificate holds on the accepted "
                    "class, but the refinement behaviour of Var(J^2) was "
                    "never measured, so neither branch can be claimed ("
                    + limit["reason"] + ")")
    return out


def stationarity_defect_correlation(runs):
    """The CONJECTURAL scaling relation, never a theorem.

    The whitepaper explicitly rejects the Hellmann-Feynman/envelope argument
    that a Regge-Hodge stationary point makes the transport Gram defect
    stationary: the defect is not the optimized functional. This is therefore
    reported as a measured correlation with its fit uncertainty and nothing
    more.
    """
    residual = []
    gram = []
    leakage = []
    for run in runs:
        joint = run["stationarity"]["joint_residual"]
        residual.append(joint)
        gram.append(run["amplitudes"]["gram_defect_max"])
        leakage.append(run["gauge"]["leakage_min"])
    log_residual = [math.log10(v) if _finite(v) and v > 0 else None
                    for v in residual]
    log_gram = [math.log10(v) if _finite(v) and v > 0 else None
                for v in gram]

    def undefined_reason(values):
        finite = [v for v in values if _finite(v) is not None]
        if len(finite) < 4:
            return "fewer than four finite points"
        if len(set(finite)) == 1:
            return (f"the series is constant at {finite[0]}, so its variance "
                    "is exactly zero and no correlation is defined")
        return None
    return {
        "status": "CONJECTURAL scaling relation, not a theorem",
        "rejected_argument": (
            "the envelope/Hellmann-Feynman argument does not make the first "
            "variation of the transport Gram defect vanish at a Regge-Hodge "
            "stationary point, because the defect is not the optimized "
            "functional (whitepaper, Conclusion)"),
        "n_points": len(runs),
        "residual_vs_gram_defect": pearson(residual, gram),
        "residual_vs_gram_defect_undefined_reason": (
            undefined_reason(gram) if pearson(residual, gram) is None
            else None),
        "residual_vs_transport_leakage": pearson(residual, leakage),
        "residual_vs_transport_leakage_undefined_reason": (
            undefined_reason(leakage) if pearson(residual, leakage) is None
            else None),
        "log_log_fit": linear_fit(log_residual, log_gram),
        "log_log_fit_meaning": (
            "slope = the measured power p in gram_defect ~ residual^p; the "
            "standard error is the honest uncertainty on p and the R^2 says "
            "how much of the spread a power law explains at all"),
        "points": [
            {"size": run["size"], "seed": run["seed"],
             "joint_stationarity_residual":
                 run["stationarity"]["joint_residual"],
             "gram_defect_max": run["amplitudes"]["gram_defect_max"],
             "transport_leakage_min": run["gauge"]["leakage_min"]}
            for run in runs
        ],
    }


def threshold_sensitivity(size, config):
    """How much the band-acceptance read moves under declared threshold
    perturbations. An UNCERTAINTY measurement; nothing is selected by it."""
    host = build_host(size, config["host_seed"])
    node = MC(host, [], [], list(config["degrees"]), 1.0, config["seeds"][0])
    node.set_objective(cob.JointStationarityObjective())
    list(node.run_stage1(max_steps=1,
                         n_candidate_moves=config["candidate_moves"]))
    list(node.run_stage2(max_iters=config["stage2_iters"]))
    spacetime = node.st
    mod_config = T.PersistentModularityConfig()
    mod_config.resolutions = [config["analysis_resolution"]]
    mod_config.baseSeed = config["seeds"][0]
    report = T.PersistentModularity.fromSpacetime(
        spacetime).scanResolutions(mod_config)
    components = report.slices[0].components if report.slices else []
    baseline = T.SpectralFiberConfig()
    rows = []
    for factor in config["threshold_scan"]:
        scan = T.SpectralFiberConfig()
        scan.degrees = list(config["degrees"])
        scan.minRelativeGap = baseline.minRelativeGap * factor
        scan.gapDominance = baseline.gapDominance * factor
        tracker = T.SpectralFiberTracker(spacetime, scan)
        accepted = total = rank_three = 0
        for component in components:
            band_read = tracker.enumerateBands(component.support,
                                               config["degrees"][0])
            total += len(band_read.fibers)
            for fiber in band_read.fibers:
                if fiber.accepted():
                    accepted += 1
                    if fiber.rank() == 3:
                        rank_three += 1
        rows.append({"factor": factor, "bands": total, "accepted": accepted,
                     "rank_three_accepted": rank_three,
                     "min_relative_gap": _finite(scan.minRelativeGap),
                     "gap_dominance": _finite(scan.gapDominance)})
    accepted_values = [r["accepted"] for r in rows]
    return {
        "size": size,
        "note": ("a sensitivity scan reported as uncertainty; no threshold "
                 "was selected using a desired verdict"),
        "rows": rows,
        "accepted_spread": max(accepted_values) - min(accepted_values),
        "rank_three_accepted_ever": max(r["rank_three_accepted"]
                                        for r in rows),
    }


#: Relative tolerance the cold-replay comparison allows on CONTINUOUS
#: aggregates. #776 measured and declared this: on a checkpoint whose edge
#: list is in construction rather than `fromCells` order the same weights
#: accumulate in a different order, so the modularity aggregates agree to
#: double round-off and not to the bit. Every DISCRETE read is compared
#: exactly, with no tolerance at all.
DECLARED_REPLAY_TOLERANCE = 1e-12


def _worst_relative_difference(left, right, path=""):
    """(max relative difference, first structurally differing path).

    Numbers are compared relatively; every other leaf must be exactly equal.
    A structural difference returns infinity so it can never hide inside a
    tolerance.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf"), path or "<root>"
        worst, where = 0.0, None
        for key in left:
            value, spot = _worst_relative_difference(
                left[key], right[key], f"{path}.{key}" if path else key)
            if value > worst:
                worst, where = value, spot
        return worst, where
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf"), path or "<root>"
        worst, where = 0.0, None
        for index, (a, b) in enumerate(zip(left, right)):
            value, spot = _worst_relative_difference(a, b, f"{path}[{index}]")
            if value > worst:
                worst, where = value, spot
        return worst, where
    if isinstance(left, bool) or isinstance(right, bool):
        return (0.0, None) if left == right else (float("inf"), path)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(float(left)), abs(float(right)))
        if scale == 0.0:
            return 0.0, None
        difference = abs(float(left) - float(right)) / scale
        return difference, (path if difference else None)
    return (0.0, None) if left == right else (float("inf"), path)


def replay_check(runs):
    """Cold replay of every stored checkpoint.

    Every DISCRETE read — classification, named failed certificates, band
    ranks and acceptance, component supports and ids, transport counts, baryon
    verdicts, the raw complex — must be byte-identical. Continuous aggregates
    are allowed the declared double-round-off tolerance and their measured
    worst relative difference is REPORTED, so a real divergence can never be
    mistaken for accumulation order.
    """
    results = []
    for run in runs:
        checkpoint = json.dumps(run["checkpoint"])
        try:
            replayed = json.loads(MC.replay_checkpoint(checkpoint))
        except Exception as error:                        # noqa: BLE001
            results.append({"size": run["size"], "seed": run["seed"],
                            "replayed": False,
                            "reason": f"{type(error).__name__}: {error}"})
            continue
        blocks = ("hierarchy", "fibers", "labeled_fiber_sums", "transports",
                  "covariance", "particles", "certificates", "raw_complex")
        identical = {}
        worst = {}
        where = {}
        for name in blocks:
            mine = run["checkpoint"].get(name)
            theirs = replayed.get(name)
            identical[name] = theirs == mine
            difference, spot = _worst_relative_difference(mine, theirs)
            worst[name] = _finite(difference)
            where[name] = spot
        within = {name: (worst[name] is not None
                         and worst[name] <= DECLARED_REPLAY_TOLERANCE)
                  for name in blocks}
        results.append({
            "size": run["size"], "seed": run["seed"],
            "replayed": True,
            "mode": replayed.get("mode"),
            "blocks_identical": identical,
            "blocks_within_tolerance": within,
            "worst_relative_difference": worst,
            "worst_difference_at": {k: v for k, v in where.items() if v},
            "discrete_verdicts_identical":
                _discrete_verdicts(run["checkpoint"])
                == _discrete_verdicts(replayed),
            "all_identical": all(identical.values()),
            "all_within_tolerance": all(within.values()),
            "tolerance": DECLARED_REPLAY_TOLERANCE,
        })
    return results


def _discrete_verdicts(document):
    """Every DISCRETE read of a checkpoint — no tolerance may apply to these."""
    return {
        "raw_complex": document.get("raw_complex"),
        "component_ids": [
            [component["id"] for component in slice_read["components"]]
            for slice_read in document.get("hierarchy", [])],
        "component_supports": [
            [component["support"] for component in slice_read["components"]]
            for slice_read in document.get("hierarchy", [])],
        "band_ranks": [(fiber["degree"], fiber["rank"], fiber["accepted"])
                       for fiber in document.get("fibers", [])],
        "labeled_sum_ranks": [
            (entry["degree"], entry["nominal_rank"], entry["effective_rank"],
             entry["quotient_nullity"])
            for entry in document.get("labeled_fiber_sums", [])],
        "transport_verdicts": [(entry["rank"], entry["numerical_rank"],
                                entry["accepted"])
                               for entry in document.get("transports", [])],
        "active_modes": document.get("covariance", {}).get("active_modes"),
        "quark_verdicts": [
            (quark["component"], quark["classification"], quark["color_rank"],
             quark["exterior_parity"], quark["winding_closure"],
             tuple(quark["failed_certificates"]))
            for quark in document.get("particles", {}).get("quarks", [])],
        "bound_supercomponent_verdicts": [
            (binding["bound_component"], binding["found"],
             binding["constituents"], tuple(binding["failed_certificates"]))
            for binding in document.get("particles", {}).get(
                "bound_supercomponents", [])],
        "baryon_verdicts": [
            (baryon["bound_component"], baryon["classification"],
             tuple(baryon["quarks"]), baryon["exterior_parity"],
             baryon["flavor_pattern"],
             tuple(baryon["failed_certificates"]))
            for baryon in document.get("particles", {}).get("baryons", [])],
    }


def aggregate(runs):
    """Every dimensionless observable with a convergence fit and uncertainty."""
    observables = {
        "accepted_band_fraction": lambda r: (
            r["bands"]["accepted"] / r["bands"]["total"]
            if r["bands"]["total"] else None),
        "rank_three_accepted_bands": lambda r: r["bands"]["rank_three_accepted"],
        "hierarchy_max_depth": lambda r: r["analysis_modularity"]["levels"],
        "modularity_q": lambda r: r["analysis_modularity"]["q"],
        "modularity_restart_spread": lambda r: (
            r["analysis_modularity"]["restart_spread"]),
        "components": lambda r: len(r["components"]),
        "mean_component_volume": lambda r: (
            sum(c["volume"] for c in r["components"]) / len(r["components"])
            if r["components"] else None),
        "mean_component_conductance": lambda r: mean_sd(
            [c["conductance"] for c in r["components"]])["mean"],
        "amplitude_gram_defect": lambda r: r["amplitudes"]["gram_defect_max"],
        "self_adjoint_band_fraction": lambda r: (
            r["bands"]["self_adjoint"] / r["bands"]["total"]
            if r["bands"]["total"] else None),
        "vacuum_embedding_defect": lambda r: (
            r["amplitudes"]["vacuum_embedding_defect_max"]),
        "static_solve_residual": lambda r: (
            r["reduction"]["static"]["solve_residual"]
            if r["reduction"].get("static") else None),
        "response_coverage_residual": lambda r: (
            (r["reduction"].get("response_network") or {}).get(
                "coverage_residual")),
        "transport_leakage_min": lambda r: r["gauge"]["leakage_min"],
        "accepted_transports": lambda r: r["gauge"]["accepted_transports"],
        "covariance_purity_defect": lambda r: (
            r["covariance"]["purity_defect_max"]),
        "joint_stationarity_residual": lambda r: (
            r["stationarity"]["joint_residual"]),
        "spectral_dimension_peak": lambda r: r["spectral_dimension"]["peak"],
        "certified_quark_fraction": lambda r: (
            1.0 - (r["particles"]["classifications"].get("none", 0)
                   / r["particles"]["quark_reads"])
            if r["particles"]["quark_reads"] else None),
        "certified_quark_fraction_resolution_scan": lambda r: (
            1.0 - (r["particles_resolution_scan"]["classifications"].get(
                "none", 0)
                   / r["particles_resolution_scan"]["quark_reads"])
            if r["particles_resolution_scan"]["quark_reads"] else None),
        "failed_certificates_per_quark": lambda r: (
            sum(r["particles"]["all_failing_certificates"].values())
            / r["particles"]["quark_reads"]
            if r["particles"]["quark_reads"] else None),
    }
    return {name: _convergence(runs, extractor, name)
            for name, extractor in observables.items()}


def spectral_dimension_verdict(runs, aggregates):
    """Does the recursive construction preserve, improve, or destroy 4D?"""
    fit = aggregates["spectral_dimension_peak"]
    by_size = fit.get("by_size", {})
    sizes = sorted(by_size)
    peaks = [by_size[s]["mean"] for s in sizes]
    limit = fit.get("extrapolated_limit")
    limit_se = fit.get("extrapolated_limit_se")
    monotone = all(b is not None and a is not None and b >= a - 1e-12
                   for a, b in zip(peaks, peaks[1:]))
    if limit is None:
        verdict = "not_established"
    elif limit_se is not None and abs(limit - PINNED_DS_BASELINE) <= 2.0 * (
            limit_se + PINNED_DS_BASELINE_SIGMA):
        verdict = "preserves_the_near_four_dimensional_regime"
    elif limit > PINNED_DS_BASELINE:
        verdict = "exceeds_the_pinned_baseline"
    else:
        verdict = "does_not_reach_the_near_four_dimensional_regime"
    return {
        "verdict": verdict,
        "peak_by_size": {str(s): by_size[s] for s in sizes},
        "monotone_in_size": monotone,
        "extrapolated_peak": limit,
        "extrapolated_peak_se": limit_se,
        "pinned_baseline": PINNED_DS_BASELINE,
        "pinned_baseline_sigma": PINNED_DS_BASELINE_SIGMA,
        "pinned_extrapolation": PINNED_DS_EXTRAPOLATION,
        "estimator": ("Spacetime.getSpectralDimensionOnSkeleton — the "
                      "EXISTING heat-kernel estimator, reused unchanged"),
        "caveat": (
            "the pinned baseline was measured on interaction-history "
            "complexes of 2.5k-20k events under a beta scan; this study "
            "measures the SAME estimator on the emergence host at 30-200 "
            "top cells, so the comparison bounds the regime reached here "
            "and is not a like-for-like reproduction of the baseline"),
    }


def degeneracy_report(runs):
    """Any unexplained near-fourfold degeneracy, reported without naming it."""
    ladder = {}
    for tolerance in DECLARED_DEGENERACY_TOLERANCES:
        key = repr(tolerance)
        combined = {}
        for run in runs:
            histograms = run["bands"].get("degeneracy_histograms", {})
            for size_key, count in histograms.get(key, {}).items():
                combined[size_key] = combined.get(size_key, 0) + count
        fourfold = combined.get("4", 0)
        total = sum(combined.values())
        ladder[key] = {
            "cluster_size_histogram": dict(sorted(
                combined.items(), key=lambda kv: int(kv[0]))),
            "fourfold_clusters": fourfold,
            "clusters_total": total,
            "fourfold_fraction": (fourfold / total) if total else None,
            "largest_cluster": max((int(k) for k in combined), default=None),
        }
    primary = ladder[repr(DECLARED_DEGENERACY_TOLERANCE)]
    return {
        "tolerance_ladder": ladder,
        "tolerance": DECLARED_DEGENERACY_TOLERANCE,
        "cluster_size_histogram": primary["cluster_size_histogram"],
        "fourfold_clusters": primary["fourfold_clusters"],
        "clusters_total": primary["clusters_total"],
        "fourfold_fraction": primary["fourfold_fraction"],
        "interpretation": (
            "reported RAW over a declared tolerance ladder. A robust fourfold "
            "degeneracy is NOT automatically Kahler-Dirac taste; naming a "
            "mechanism would require a prediction from the stated "
            "one-particle operator, which this study does not have"),
    }


def amplitude_regime_report(runs):
    """The bimodal retained-fiber Gram defect, and what tracks it."""
    regimes = _string_histogram(r["amplitudes"]["gram_defect_regime"]
                                for r in runs)
    paired = []
    for run in runs:
        regime = run["amplitudes"]["gram_defect_regime"]
        total = run["bands"]["total"]
        fraction = (run["bands"]["self_adjoint"] / total) if total else None
        paired.append({"size": run["size"], "seed": run["seed"],
                       "gram_defect_regime": regime,
                       "gram_defect_max": run["amplitudes"]["gram_defect_max"],
                       "self_adjoint_band_fraction": _finite(fraction),
                       "krein_indefinite_bands":
                           run["bands"]["krein_indefinite"]})
    agree = sum(1 for entry in paired
                if entry["gram_defect_regime"] is not None
                and entry["self_adjoint_band_fraction"] is not None
                and ((entry["gram_defect_regime"] == "isometric")
                     == (entry["self_adjoint_band_fraction"] > 0.5)))
    return {
        "regimes": regimes,
        "per_run": paired,
        "isometric_iff_self_adjoint_agreement": (
            agree / len(paired) if paired else None),
        "mechanism": (
            "the retained-fiber Gram is G = J* W J and the defect is "
            "||G - I||; an all-positive weight diagonal makes the embedding "
            "an exact isometry, while one negative (Krein) weight flips a "
            "diagonal entry to -1 and puts the defect at 2 — so the "
            "observable is bimodal by construction and a mean across the "
            "ensemble would be meaningless"),
    }


# =====================================================================
# driver
# =====================================================================

def make_config(quick, sizes=None, seeds=None):
    config = {
        "sizes": list(sizes if sizes else
                      (DECLARED_SIZES_QUICK if quick else DECLARED_SIZES_FULL)),
        "seeds": list(seeds if seeds else
                      (DECLARED_SEEDS_QUICK if quick else DECLARED_SEEDS_FULL)),
        "host_seed": DECLARED_HOST_SEED,
        "degrees": list(DECLARED_DEGREES),
        "analysis_resolution": DECLARED_ANALYSIS_RESOLUTION,
        "resolution_scan": list(DECLARED_RESOLUTION_SCAN),
        "candidate_moves": DECLARED_CANDIDATE_MOVES,
        "stage2_iters": DECLARED_STAGE2_ITERS,
        "sigmas": list(DECLARED_SIGMAS),
        "krylov_dim": DECLARED_KRYLOV_DIM,
        "shift_fractions": list(DECLARED_SHIFT_FRACTIONS),
        "window_half_width_fraction": DECLARED_WINDOW_HALF_WIDTH_FRACTION,
        "amls_mode_cutoff": DECLARED_AMLS_MODE_CUTOFF,
        "degeneracy_tolerances": list(DECLARED_DEGENERACY_TOLERANCES),
        "degeneracy_tolerance": DECLARED_DEGENERACY_TOLERANCE,
        "threshold_scan": list(DECLARED_THRESHOLD_SCAN),
        "control_seed": DECLARED_CONTROL_SEED,
        "objective_name": cob.ObjectiveName.JOINT_STATIONARITY,
        "simulation_mode": "emergence",
        "emergence_submode": "strict",
        "checkpoint_schema_version": MC.checkpoint_schema_version(),
    }
    return config


def config_hash_of(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def current_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="#777 multiscale convergence, covariance-only dichotomy, "
                    "and spectral-dimension validation.")
    parser.add_argument("--quick", action="store_true",
                        help="reduced ensemble (3 sizes x 2 seeds), minutes "
                             "rather than tens of minutes")
    parser.add_argument("--sizes", type=int, nargs="+",
                        help="override the declared refinement sizes")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="override the declared seed set")
    parser.add_argument("--out", default=None,
                        help="machine-readable JSON output path")
    parser.add_argument("--no-controls", action="store_true",
                        help="skip the negative controls (diagnostics only; "
                             "a published run MUST include them)")
    parser.add_argument("--no-replay", action="store_true",
                        help="skip the cold checkpoint replay check")
    parser.add_argument("--drop-checkpoints", action="store_true",
                        help="drop the embedded schema-4 checkpoints from the "
                             "output. They are embedded by default and are "
                             "what makes a plotted point exactly replayable, "
                             "so a published run keeps them")
    args = parser.parse_args(argv)

    config = make_config(args.quick, args.sizes, args.seeds)
    digest = config_hash_of(config)
    commit = current_commit()
    started = time.time()

    runs = []
    for size in config["sizes"]:
        for seed in config["seeds"]:
            print(f"[#777] size={size} seed={seed} ...", file=sys.stderr,
                  flush=True)
            runs.append(run_member(size, seed, config, commit, digest))

    aggregates = aggregate(runs)
    result = {
        "schema_version": SCHEMA_VERSION,
        "ticket": 777,
        "epic": 763,
        "mode": "quick" if args.quick else "full",
        "config": config,
        "config_hash": digest,
        "commit": commit,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "reproducibility": {
            "point_identity": "config_hash + commit + size + seed",
            "exact_replay": ("every run embeds its schema-4 checkpoint; "
                             "MultiCobordism.replay_checkpoint rebuilds the "
                             "raw complex with cold caches and reproduces the "
                             "verdicts"),
            "engine_limitation": (
                "the engine's move draw is NOT process-deterministic past the "
                "first committed move (#579, remeasured in #776), so a seed "
                "LABELS an attempt rather than reproducing it; the drive is "
                "deliberately one stage-1 update plus a full stage-2 "
                "relaxation, which IS deterministic"),
        },
        "runs": runs,
        "aggregates": aggregates,
        "dichotomy": dichotomy(runs,
                               proton_verdicts=proton_verdicts_of(runs)),
        "stationarity_defect_correlation":
            stationarity_defect_correlation(runs),
        "spectral_dimension_verdict":
            spectral_dimension_verdict(runs, aggregates),
        "degeneracy": degeneracy_report(runs),
        "amplitude_regime": amplitude_regime_report(runs),
        "analytic_invariants": analytic_invariants(),
        "uncertainty_budget": {
            "finite_size_drift": (
                "the size-to-size movement of each observable's mean; the "
                "1/N fit's slope in `aggregates`"),
            "ensemble_variance": (
                "the per-size sample sd over the seed set, in each "
                "observable's `by_size` block"),
            "solver_residual": (
                "the measured certificate residuals: band eigen/projector/"
                "Gram defects, static and shifted solve residuals, covariance "
                "purity defect"),
            "threshold_sensitivity": None,
        },
    }

    if not args.no_controls:
        control_size = config["sizes"][0]
        result["negative_controls"] = {
            "size": control_size,
            "controls": negative_controls(control_size, config),
        }
        result["negative_controls"]["all_fired"] = all(
            c["fired"] for c in result["negative_controls"]["controls"])
        result["uncertainty_budget"]["threshold_sensitivity"] = \
            threshold_sensitivity(control_size, config)

    if not args.no_replay:
        entries = replay_check(runs)
        result["replay"] = {
            "tolerance": DECLARED_REPLAY_TOLERANCE,
            "entries": entries,
            "all_replayed": all(e["replayed"] for e in entries),
            "all_discrete_verdicts_identical": all(
                e.get("discrete_verdicts_identical") for e in entries),
            "byte_identical": sum(1 for e in entries
                                  if e.get("all_identical")),
            "within_tolerance": sum(1 for e in entries
                                    if e.get("all_within_tolerance")),
            "members": len(entries),
        }

    if args.drop_checkpoints:
        for run in result["runs"]:
            run.pop("checkpoint", None)
        result["reproducibility"]["exact_replay"] = (
            "checkpoints were DROPPED from this output (--drop-checkpoints); "
            "the run is reproducible only up to the engine limitation below")

    result["runtime"] = {
        "wall_seconds": time.time() - started,
        "members": len(runs),
        "per_member_seconds": mean_sd([r["wall_seconds"] for r in runs]),
    }

    payload = json.dumps(result, indent=2, sort_keys=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        print(f"[#777] wrote {args.out} "
              f"({len(payload) / 1e6:.2f} MB, "
              f"{result['runtime']['wall_seconds']:.1f}s)", file=sys.stderr)
    else:
        print(payload)
    return result


if __name__ == "__main__":
    main()
