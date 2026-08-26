# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Sweep the fixed-boundary common-eigenspace operator construction.

The experiment separates three outcomes:

* verified: the solver reached its requested residual and all exact boundary,
  recovery, and held-out checks passed;
* budget_limited: the finite search budget ended above the requested residual,
  which is not a structural no-go result;
* rejected: the boundary data violated an explicit preparation precondition,
  such as a zero output or a state spanning isolated-boundary eigenspaces.

Run from the repository root:

    python examples/cobordism/geometric_operator_sweep.py --jobs 4
    python examples/cobordism/geometric_operator_sweep.py --profile exhaustive --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

try:
    from examples.cobordism.geometric_operators import (
        BoundaryPairCobordism,
        _matrix_payload as matrix_payload,
        squared_content_weights,
    )
except ModuleNotFoundError:
    from geometric_operators import (
        BoundaryPairCobordism,
        _matrix_payload as matrix_payload,
        squared_content_weights,
    )


_PROFILE_RANDOM_COUNTS = {
    "smoke": 1,
    "standard": 8,
    "exhaustive": 32,
}
_DEFAULT_OUTPUT = Path(
    "/tmp/cobordism/geometric_operator_sweep.json")
_TINY = np.finfo(float).tiny


def _unit(vector):
    vector = np.asarray(vector, dtype=complex)
    norm = float(np.linalg.norm(vector))
    if not norm > 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def _haar(dimension, rng):
    matrix = (
        rng.normal(size=(dimension, dimension))
        + 1j * rng.normal(size=(dimension, dimension))
    )
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    phases = np.ones(dimension, dtype=complex)
    nonzero = np.abs(diagonal) > 0.0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return q @ np.diag(phases.conj())


def _normalized_matrix(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    norm = float(np.linalg.norm(matrix))
    return matrix / norm if norm > 0.0 else matrix


def _case(
        name, family, operator, *, input_basis=None, solver_seed=0,
        common_eigenvalue=True, notes=""):
    operator = np.asarray(operator, dtype=complex)
    dimension = int(operator.shape[0])
    if input_basis is None:
        input_basis = np.eye(dimension, dtype=complex)
    return {
        "name": str(name),
        "family": str(family),
        "operator": operator,
        "input_basis": np.asarray(input_basis, dtype=complex),
        "solver_seed": int(solver_seed),
        "common_eigenvalue": bool(common_eigenvalue),
        "notes": str(notes),
    }


def _block_operator(charge_amplitude, logical):
    operator = np.zeros((3, 3), dtype=complex)
    operator[0, 0] = complex(charge_amplitude)
    operator[1:, 1:] = np.asarray(logical, dtype=complex)
    return operator


def _structured_cases():
    root_two = math.sqrt(2.0)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    pauli_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    pauli_z = np.diag([1.0, -1.0]).astype(complex)
    hadamard = np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=complex) / root_two
    cases = [
        _case("unitary_identity", "unitary", np.eye(2)),
        _case("unitary_minus_identity", "unitary", -np.eye(2)),
        _case("unitary_pauli_x", "unitary", pauli_x),
        _case("unitary_pauli_y", "unitary", pauli_y),
        _case("unitary_pauli_z", "unitary", pauli_z),
        _case("unitary_hadamard", "unitary", hadamard),
        _case(
            "unitary_phase_s", "unitary",
            np.diag([1.0, 1j])),
        _case(
            "unitary_phase_t", "unitary",
            np.diag([1.0, np.exp(0.25j * math.pi)])),
        _case(
            "unitary_complex_rotation", "unitary",
            np.array([
                [math.cos(0.37), -np.exp(-0.41j) * math.sin(0.37)],
                [np.exp(0.41j) * math.sin(0.37), math.cos(0.37)],
            ])),
        _case(
            "normal_complex_diagonal", "normal",
            np.diag([0.3 + 0.8j, -1.7 + 0.2j])),
    ]

    for exponent in (-12, -9, -6, -3, -1, 1, 3, 6, 9, 12):
        scale = 10.0 ** exponent
        cases.append(_case(
            f"scalar_1e{exponent:+03d}", "scale",
            scale * np.eye(2)))
    for exponent in (2, 4, 6, 8, 10, 12):
        ratio = 10.0 ** exponent
        cases.extend([
            _case(
                f"condition_small_1e{exponent:02d}",
                "condition", np.diag([1.0, 1.0 / ratio])),
            _case(
                f"condition_large_1e{exponent:02d}",
                "condition", np.diag([1.0, ratio])),
        ])
    for shear in (0.1, 1.0, 10.0, 1e2, 1e4, 1e6):
        cases.append(_case(
            f"jordan_shear_{shear:g}", "nonnormal",
            np.array([[1.0, shear], [0.0, 1.0]], dtype=complex)))

    dense_left = _unit([1.0, 1j])
    dense_right = _unit([1.0, np.exp(0.3j)])
    cases.extend([
        _case(
            "rank_one_dense", "singular",
            np.outer(dense_left, dense_right.conj())),
        _case(
            "rank_one_dense_scaled", "singular",
            1e3 * np.outer(dense_left, dense_right.conj())),
        _case(
            "rank_one_plus_projector", "singular",
            0.5 * np.ones((2, 2), dtype=complex)),
        _case(
            "rank_one_zero_first_column", "zero_output",
            np.array([[0.0, 1.0], [0.0, 1j]])),
        _case(
            "rank_one_zero_second_column", "zero_output",
            np.array([[1.0, 0.0], [1j, 0.0]])),
        _case("zero_operator", "zero_output", np.zeros((2, 2))),
    ])

    target = np.array(
        [[0.7 + 0.2j, -0.4j], [0.3 - 0.1j, -0.8 + 0.5j]],
        dtype=complex,
    )
    basis_rng = np.random.default_rng(73)
    cases.extend([
        _case(
            "basis_real_rotation", "input_basis", target,
            input_basis=np.array(
                [[1.0, -1.0], [1.0, 1.0]]) / root_two),
        _case(
            "basis_complex_haar", "input_basis", target,
            input_basis=_haar(2, basis_rng)),
        _case(
            "basis_column_rescaling", "input_basis", target,
            input_basis=np.diag([1e-9, 1e9])),
    ])
    for exponent in (2, 4, 6, 8, 10, 12):
        epsilon = 10.0 ** (-exponent)
        cases.append(_case(
            f"basis_near_collinear_1e{exponent:02d}",
            "input_basis", target,
            input_basis=np.array(
                [[1.0, 1.0], [0.0, epsilon]], dtype=complex),
        ))

    right = _unit([1.0, -0.4j])
    rank_one = np.outer(_unit([1.0, 2j]), right.conj())
    kernel = np.array([-np.conj(right[1]), np.conj(right[0])])
    complement = _unit([1.0, 0.2 + 0.4j])
    cases.append(_case(
        "rank_one_kernel_training_vector", "zero_output",
        rank_one,
        input_basis=np.column_stack((kernel, complement)),
        notes="A selected input lies exactly in the operator kernel.",
    ))

    cases.extend([
        _case(
            "qutrit_charge_preserving_identity", "charge_preserving",
            np.eye(3)),
        _case(
            "qutrit_charge_preserving_unitary", "charge_preserving",
            _block_operator(1.0, hadamard)),
        _case(
            "qutrit_charge_preserving_nonunitary", "charge_preserving",
            _block_operator(
                2.0, np.array([[1.0, 0.4j], [0.3, -0.7]]))),
        _case(
            "qutrit_charge_preserving_singular", "charge_preserving",
            _block_operator(
                0.5, np.outer(dense_left, dense_right.conj()))),
        _case(
            "qutrit_charge_swap", "charge_permuting",
            np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ])),
        _case(
            "qutrit_sector_cycle", "charge_permuting",
            np.array([
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ])),
        _case(
            "qutrit_charge_mixing_rotation", "charge_mixing",
            np.array([
                [math.cos(0.3), -math.sin(0.3), 0.0],
                [math.sin(0.3), math.cos(0.3), 0.0],
                [0.0, 0.0, 1.0],
            ])),
        _case(
            "qutrit_charge_preserving_mixed_inputs",
            "charge_mixing_input", np.eye(3),
            input_basis=_haar(3, np.random.default_rng(91)),
            notes=(
                "The operator preserves sectors, but the prepared input "
                "basis mixes isolated-boundary eigenspaces."),
        ),
        _case(
            "qutrit_zero_charge_output", "zero_output",
            _block_operator(0.0, np.eye(2))),
        _case(
            "qutrit_identity_individual_eigenvalues", "control",
            np.eye(3), common_eigenvalue=False,
            notes=(
                "Control only: independent Rayleigh quotients do not certify "
                "linear combinations as one eigenspace."),
        ),
    ])
    return cases


def _random_cases(profile, generation_seed):
    count = _PROFILE_RANDOM_COUNTS[profile]
    rng = np.random.default_rng(generation_seed)
    cases = []
    for index in range(count):
        unitary = _haar(2, rng)
        cases.append(_case(
            f"random_unitary_{index:03d}", "random_unitary", unitary))

        ginibre = _normalized_matrix(
            rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        cases.append(_case(
            f"random_ginibre_{index:03d}", "random_ginibre", ginibre))

        raw = (
            rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        hermitian = _normalized_matrix(raw + raw.conj().T)
        cases.append(_case(
            f"random_hermitian_{index:03d}",
            "random_hermitian", hermitian))

        chart = _haar(2, rng)
        eigenvalues = (
            10.0 ** rng.uniform(-3.0, 3.0, size=2)
            * np.exp(1j * rng.uniform(-math.pi, math.pi, size=2))
        )
        normal = chart @ np.diag(eigenvalues) @ chart.conj().T
        cases.append(_case(
            f"random_normal_{index:03d}", "random_normal", normal))

        chart = _haar(2, rng)
        triangular = np.array([
            [rng.normal() + 1j * rng.normal(),
             10.0 ** rng.uniform(-2.0, 4.0)
             * np.exp(1j * rng.uniform(-math.pi, math.pi))],
            [0.0, rng.normal() + 1j * rng.normal()],
        ])
        nonnormal = chart @ triangular @ chart.conj().T
        cases.append(_case(
            f"random_nonnormal_{index:03d}",
            "random_nonnormal", nonnormal))

        left = _haar(2, rng)
        right_chart = _haar(2, rng)
        log_condition = rng.uniform(0.0, 14.0)
        conditioned = (
            left @ np.diag([1.0, 10.0 ** (-log_condition)])
            @ right_chart.conj().T
        )
        cases.append(_case(
            f"random_conditioned_{index:03d}",
            "random_conditioned", conditioned))

        left_vector = _unit(
            rng.normal(size=2) + 1j * rng.normal(size=2))
        right_vector = _unit(
            rng.normal(size=2) + 1j * rng.normal(size=2))
        rank_one = (
            10.0 ** rng.uniform(-6.0, 6.0)
            * np.outer(left_vector, right_vector.conj())
        )
        cases.append(_case(
            f"random_rank_one_{index:03d}",
            "random_rank_one", rank_one))

        scaled = 10.0 ** rng.uniform(-14.0, 14.0) * ginibre
        cases.append(_case(
            f"random_scaled_{index:03d}",
            "random_scaled", scaled))

    qutrit_count = max(1, count // 2)
    for index in range(qutrit_count):
        charge = rng.normal() + 1j * rng.normal()
        if abs(charge) < 0.1:
            charge += 0.5
        logical = _normalized_matrix(
            rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        cases.append(_case(
            f"random_charge_preserving_{index:03d}",
            "random_charge_preserving",
            _block_operator(charge, logical)))

        permutation = rng.permutation(3)
        phases = np.exp(1j * rng.uniform(-math.pi, math.pi, size=3))
        monomial = np.zeros((3, 3), dtype=complex)
        monomial[permutation, np.arange(3)] = phases
        cases.append(_case(
            f"random_charge_permuting_{index:03d}",
            "random_charge_permuting", monomial))

        cases.append(_case(
            f"random_charge_mixing_{index:03d}",
            "random_charge_mixing", _haar(3, rng)))

    repeated = 2 if profile == "smoke" else (4 if profile == "standard" else 8)
    reference = np.array(
        [[0.4 + 0.7j, -0.2 + 0.1j],
         [0.8 - 0.3j, -0.5j]],
        dtype=complex,
    )
    for solver_seed in range(repeated):
        cases.append(_case(
            f"solver_seed_repeat_{solver_seed:02d}",
            "solver_seed", reference, solver_seed=solver_seed))
    return cases


def build_cases(profile="standard", generation_seed=20260826):
    if profile not in _PROFILE_RANDOM_COUNTS:
        raise ValueError(f"unknown sweep profile: {profile}")
    cases = _structured_cases()
    cases.extend(_random_cases(profile, int(generation_seed)))
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("sweep case names must be unique")
    return cases


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _effective_input_basis(input_basis):
    basis = np.asarray(input_basis, dtype=complex).copy()
    for column in range(basis.shape[1]):
        norm = float(np.linalg.norm(basis[:, column]))
        if norm > 0.0:
            basis[:, column] /= norm
    return basis


def _sector_pure(vector, tolerance=1e-12):
    vector = np.asarray(vector, dtype=complex)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return False
    return (
        abs(vector[0]) <= tolerance * norm
        or np.linalg.norm(vector[1:]) <= tolerance * norm
    )


def operator_diagnostics(case):
    operator = np.asarray(case["operator"], dtype=complex)
    input_basis = np.asarray(case["input_basis"], dtype=complex)
    singular_values = np.linalg.svd(operator, compute_uv=False)
    norm = float(np.linalg.norm(operator))
    spectral_norm = float(singular_values[0]) if singular_values.size else 0.0
    effective_basis = _effective_input_basis(input_basis)
    input_vectors = [
        input_basis[:, index] for index in range(input_basis.shape[1])]
    output_vectors = [operator @ vector for vector in input_vectors]
    dimension = operator.shape[0]
    charge_projector = np.zeros_like(operator)
    charge_projector[0, 0] = 1.0
    return {
        "dimension": int(dimension),
        "rank": int(np.linalg.matrix_rank(operator)),
        "frobenius_norm": norm,
        "spectral_norm": spectral_norm,
        "condition_number": _finite(np.linalg.cond(operator)),
        "smallest_singular_value": (
            float(singular_values[-1]) if singular_values.size else 0.0),
        "unitarity_error": float(np.linalg.norm(
            operator.conj().T @ operator - np.eye(dimension))),
        "normality_error": float(np.linalg.norm(
            operator.conj().T @ operator
            - operator @ operator.conj().T)),
        "charge_commutator_error": (
            float(np.linalg.norm(
                operator @ charge_projector
                - charge_projector @ operator))
            if dimension == 3 else None
        ),
        "input_basis_condition": _finite(np.linalg.cond(input_basis)),
        "prepared_input_condition": _finite(
            np.linalg.cond(effective_basis)),
        "minimum_training_output_norm": float(min(
            (np.linalg.norm(vector) for vector in output_vectors),
            default=0.0,
        )),
        "zero_training_outputs": int(sum(
            np.linalg.norm(vector) == 0.0 for vector in output_vectors)),
        "boundary_compatible_prediction": (
            True if dimension == 2 else all(
                _sector_pure(vector)
                for vector in input_vectors + output_vectors)
        ),
    }


def _quality_status(result, config, operator_norm):
    held_out_relative = (
        result["held_out_operator_error_max"]
        / max(float(operator_norm), _TINY)
    )
    result["held_out_operator_relative_error_max"] = float(
        held_out_relative)
    if not result["converged"]:
        return "budget_limited"
    exact_boundary = (
        result["boundary_preserved"]
        and result["boundary_drift"] == 0.0
        and result["restriction_error"] == 0.0
    )
    isolated = max(
        result["input_boundary_residuals"]
        + result["output_boundary_residuals"],
        default=0.0,
    ) < config["boundary_epsilon"]
    if not exact_boundary or not isolated:
        return "boundary_invariant_failure"
    if (
            result["operator_relative_error"] >= 1e-8
            or held_out_relative >= 1e-8):
        return "readout_unstable"
    if result["held_out_full_residual_max"] >= 1e-10:
        return "span_not_common"
    return "verified"


def _fit_attempt(case, config, diagnostics, budget, solver_seed):
    started = time.perf_counter()
    attempt = {
        "solver_seed": int(solver_seed),
        "search_budget": dict(budget),
    }
    try:
        fixture = BoundaryPairCobordism(
            include_charge_mode=(diagnostics["dimension"] == 3))
        with squared_content_weights():
            result = fixture.relax_operator(
                case["operator"],
                epsilon=config["epsilon"],
                boundary_epsilon=config["boundary_epsilon"],
                restarts=budget["restarts"],
                max_growth=budget["max_growth"],
                seed=solver_seed,
                max_iterations=budget["max_iterations"],
                held_out_count=config["held_out_count"],
                input_basis=case["input_basis"],
                common_eigenvalue=case["common_eigenvalue"],
            )
        attempt["fit"] = result
        attempt["status"] = _quality_status(
            result, config, diagnostics["frobenius_norm"])
    except ValueError as error:
        attempt["status"] = "rejected"
        attempt["reason"] = str(error)
    except Exception as error:
        attempt["status"] = "error"
        attempt["reason"] = f"{type(error).__name__}: {error}"
    attempt["duration_seconds"] = float(
        time.perf_counter() - started)
    return attempt


def _attempt_summary(attempt):
    summary = {
        "solver_seed": attempt["solver_seed"],
        "search_budget": attempt["search_budget"],
        "status": attempt["status"],
        "duration_seconds": attempt["duration_seconds"],
    }
    if "reason" in attempt:
        summary["reason"] = attempt["reason"]
    if "fit" in attempt:
        summary.update({
            "residual": attempt["fit"]["residual"],
            "growth_steps": attempt["fit"]["growth_steps"],
            "converged": attempt["fit"]["converged"],
        })
    return summary


def evaluate_case(case, config):
    started = time.perf_counter()
    diagnostics = operator_diagnostics(case)
    qutrit = diagnostics["dimension"] == 3
    initial_budget = (
        config["qutrit_budget"]
        if qutrit else config["neutral_budget"]
    )
    confirmation_budget = (
        config["qutrit_confirmation_budget"]
        if qutrit else config["neutral_confirmation_budget"]
    )
    attempts = []
    attempt_specs = [(case["solver_seed"], initial_budget)]
    for index in range(config["confirmation_attempts"]):
        attempt_specs.append((
            case["solver_seed"] + 104729 * (index + 1),
            confirmation_budget,
        ))

    for solver_seed, budget in attempt_specs:
        attempt = _fit_attempt(
            case, config, diagnostics, budget, solver_seed)
        attempts.append(attempt)
        if attempt["status"] != "budget_limited":
            break

    errors = [
        attempt for attempt in attempts
        if attempt["status"] == "error"]
    verified = [
        attempt for attempt in attempts
        if attempt["status"] == "verified"]
    fitted = [attempt for attempt in attempts if "fit" in attempt]
    if errors:
        chosen = errors[0]
    elif verified:
        chosen = verified[0]
    elif fitted:
        chosen = min(
            fitted, key=lambda attempt: attempt["fit"]["residual"])
    else:
        chosen = attempts[0]

    record = {
        "name": case["name"],
        "family": case["family"],
        "solver_seed": case["solver_seed"],
        "common_eigenvalue": case["common_eigenvalue"],
        "notes": case["notes"],
        "operator": matrix_payload(case["operator"]),
        "input_basis": matrix_payload(case["input_basis"]),
        "diagnostics": diagnostics,
        "status": chosen["status"],
        "selected_solver_seed": chosen["solver_seed"],
        "search_attempts": [
            _attempt_summary(attempt) for attempt in attempts],
        "duration_seconds": float(time.perf_counter() - started),
    }
    if "fit" in chosen:
        record["fit"] = chosen["fit"]
    if "reason" in chosen:
        record["reason"] = chosen["reason"]
    return record


def _evaluate_payload(payload):
    index, case, config = payload
    return index, evaluate_case(case, config)


def summarize(records):
    statuses = Counter(record["status"] for record in records)
    by_family = defaultdict(Counter)
    for record in records:
        by_family[record["family"]][record["status"]] += 1
    verified = [
        record for record in records if record["status"] == "verified"]
    attempted = [
        record for record in records if "fit" in record]
    rejected = [
        record for record in records if record["status"] == "rejected"]
    norms = [
        record["diagnostics"]["frobenius_norm"] for record in verified]
    conditions = [
        record["diagnostics"]["condition_number"]
        for record in verified
        if record["diagnostics"]["condition_number"] is not None
    ]
    residuals = [
        record["fit"]["residual"] for record in attempted]
    boundary_violations = [
        record["name"] for record in attempted
        if (
            not record["fit"]["boundary_preserved"]
            or record["fit"]["boundary_drift"] != 0.0
            or record["fit"]["restriction_error"] != 0.0
        )
    ]
    return {
        "case_count": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "family_status_counts": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(by_family.items())
        },
        "verified_operator_norm_range": (
            [min(norms), max(norms)] if norms else None),
        "verified_condition_number_range": (
            [min(conditions), max(conditions)] if conditions else None),
        "best_attempted_residual": min(residuals, default=None),
        "worst_attempted_residual": max(residuals, default=None),
        "maximum_verified_growth": max(
            (record["fit"]["growth_steps"] for record in verified),
            default=None,
        ),
        "boundary_invariant_violations": boundary_violations,
        "rejection_reasons": dict(sorted(Counter(
            record["reason"] for record in rejected).items())),
        "total_duration_seconds": float(sum(
            record["duration_seconds"] for record in records)),
    }


def run_sweep(
        profile="standard", generation_seed=20260826, jobs=1,
        epsilon=1e-16, boundary_epsilon=1e-12, restarts=4,
        max_growth=10, max_iterations=400, qutrit_restarts=8,
        qutrit_max_growth=12, qutrit_max_iterations=400,
        confirmation_attempts=0, confirmation_restarts=8,
        confirmation_max_growth=14, confirmation_max_iterations=600,
        qutrit_confirmation_restarts=12,
        qutrit_confirmation_max_growth=16,
        qutrit_confirmation_max_iterations=600,
        held_out_count=16, families=None, name_contains=None,
        progress=False):
    cases = build_cases(profile, generation_seed)
    if families:
        selected = set(families)
        cases = [case for case in cases if case["family"] in selected]
    if name_contains:
        cases = [
            case for case in cases if name_contains in case["name"]]
    if not cases:
        raise ValueError("sweep selection is empty")
    config = {
        "profile": profile,
        "generation_seed": int(generation_seed),
        "epsilon": float(epsilon),
        "boundary_epsilon": float(boundary_epsilon),
        "neutral_budget": {
            "restarts": int(restarts),
            "max_growth": int(max_growth),
            "max_iterations": int(max_iterations),
        },
        "qutrit_budget": {
            "restarts": int(qutrit_restarts),
            "max_growth": int(qutrit_max_growth),
            "max_iterations": int(qutrit_max_iterations),
        },
        "neutral_confirmation_budget": {
            "restarts": int(confirmation_restarts),
            "max_growth": int(confirmation_max_growth),
            "max_iterations": int(confirmation_max_iterations),
        },
        "qutrit_confirmation_budget": {
            "restarts": int(qutrit_confirmation_restarts),
            "max_growth": int(qutrit_confirmation_max_growth),
            "max_iterations": int(qutrit_confirmation_max_iterations),
        },
        "confirmation_attempts": int(confirmation_attempts),
        "held_out_count": int(held_out_count),
    }
    if jobs < 1:
        raise ValueError("jobs must be positive")

    started = time.perf_counter()
    records = [None] * len(cases)
    payloads = [
        (index, case, config) for index, case in enumerate(cases)]
    if jobs == 1:
        completed = map(_evaluate_payload, payloads)
        for count, (index, record) in enumerate(completed, start=1):
            records[index] = record
            if progress:
                print(
                    f"[{count}/{len(cases)}] "
                    f"{record['name']}: {record['status']}",
                    flush=True,
                )
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs) as executor:
            futures = [
                executor.submit(_evaluate_payload, payload)
                for payload in payloads
            ]
            for count, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1):
                index, record = future.result()
                records[index] = record
                if progress:
                    print(
                        f"[{count}/{len(cases)}] "
                        f"{record['name']}: {record['status']}",
                        flush=True,
                    )

    result = {
        "schema_version": 1,
        "method": (
            "fixed-boundary full-W common-eigenvalue relaxation"),
        "config": {
            **config,
            "jobs": int(jobs),
        },
        "wall_duration_seconds": float(
            time.perf_counter() - started),
        "summary": summarize(records),
        "cases": records,
    }
    json.dumps(result, allow_nan=False)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile", choices=sorted(_PROFILE_RANDOM_COUNTS),
        default="standard")
    parser.add_argument("--generation-seed", type=int, default=20260826)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-16)
    parser.add_argument(
        "--boundary-epsilon", type=float, default=1e-12)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--max-growth", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--qutrit-restarts", type=int, default=8)
    parser.add_argument("--qutrit-max-growth", type=int, default=12)
    parser.add_argument("--qutrit-max-iterations", type=int, default=400)
    parser.add_argument("--confirmation-attempts", type=int, default=0)
    parser.add_argument("--confirmation-restarts", type=int, default=8)
    parser.add_argument("--confirmation-max-growth", type=int, default=14)
    parser.add_argument(
        "--confirmation-max-iterations", type=int, default=600)
    parser.add_argument(
        "--qutrit-confirmation-restarts", type=int, default=12)
    parser.add_argument(
        "--qutrit-confirmation-max-growth", type=int, default=16)
    parser.add_argument(
        "--qutrit-confirmation-max-iterations", type=int, default=600)
    parser.add_argument("--held-out-count", type=int, default=16)
    parser.add_argument(
        "--family", action="append", dest="families",
        help="run only this family; repeat to select several")
    parser.add_argument(
        "--name-contains",
        help="run only cases whose name contains this text")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list:
        for case in build_cases(args.profile, args.generation_seed):
            if args.families and case["family"] not in args.families:
                continue
            if args.name_contains and args.name_contains not in case["name"]:
                continue
            print(f"{case['name']}\t{case['family']}")
        return 0

    result = run_sweep(
        profile=args.profile,
        generation_seed=args.generation_seed,
        jobs=args.jobs,
        epsilon=args.epsilon,
        boundary_epsilon=args.boundary_epsilon,
        restarts=args.restarts,
        max_growth=args.max_growth,
        max_iterations=args.max_iterations,
        qutrit_restarts=args.qutrit_restarts,
        qutrit_max_growth=args.qutrit_max_growth,
        qutrit_max_iterations=args.qutrit_max_iterations,
        confirmation_attempts=args.confirmation_attempts,
        confirmation_restarts=args.confirmation_restarts,
        confirmation_max_growth=args.confirmation_max_growth,
        confirmation_max_iterations=args.confirmation_max_iterations,
        qutrit_confirmation_restarts=(
            args.qutrit_confirmation_restarts),
        qutrit_confirmation_max_growth=(
            args.qutrit_confirmation_max_growth),
        qutrit_confirmation_max_iterations=(
            args.qutrit_confirmation_max_iterations),
        held_out_count=args.held_out_count,
        families=args.families,
        name_contains=args.name_contains,
        progress=args.progress,
    )
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = result["summary"]
    print("cases:", summary["case_count"])
    print(
        "statuses:",
        ", ".join(
            f"{name}={count}"
            for name, count in summary["status_counts"].items()),
    )
    print(
        "wall_seconds:",
        f"{result['wall_duration_seconds']:.3f}",
    )
    if not args.no_write:
        print("record:", args.output)
    return 1 if summary["status_counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
