# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Fixed-boundary spectral Choi synthesis and operator identifiability.

The primary experiment reproduces the method that existed when the 2026-06-24
report was written. It fixes the relative amplitudes of an explicit normalized
``vec(U)`` block, frees all other cochain amplitudes, varies only interior
geometry, and minimizes the eigenvalue-agnostic Rayleigh residual. The complete
cochain is normalized during evaluation, so the fixed block specifies a ray.
Boundary geometry is bit-identical; no Regge, period, harmonic, or charge
constraint enters that relaxation.

The second experiment retains the later period-based ``r_U`` implementation.
It distinguishes carrying one input/output pair, identifying boundary transport
from a complete basis, and promoting a framed bulk-minus-boundary kernel. These
are separate semantics and are reported separately.

The coupled boundary-value experiment supplies the missing semantics explicitly:
two disconnected boundary components are prepared as degenerate Laplacian
eigenspaces, their geometries and state restrictions are fixed, and one shared
bulk is fitted to a complete set of input/output pairs at a common full-complex
eigenvalue. Linear combinations of the fitted witnesses then test unseen inputs.

Run:

    python examples/cobordism/geometric_operators.py
    python examples/cobordism/geometric_operators.py --live
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import tessera


cob = tessera.cobordism
choi = tessera.quantum.ChoiJamiolkowski

_TOL = 1e-9
# Round-off of a squared residual of an exactly realizable period fit. The
# diagonal-weight operator lands at ~1e-27; the chain-level Whitney pencil,
# whose operator entries are an order of magnitude larger on the same fixture,
# lands at ~1e-20 (#931). Both are machine-precision zeros of the fit.
_TINY_PERIOD_RESIDUAL = 1e-18
_SPECTRAL_EPSILON = 1e-24
_CHARGE = np.ones(3, dtype=complex) / math.sqrt(3.0)
_CHARGE_PROJECTOR = np.outer(_CHARGE, _CHARGE.conj())
_SECTOR_BASIS = np.array(
    [
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
        [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0),
         -2.0 / math.sqrt(6.0)],
    ],
    dtype=complex,
)

_ICOSAHEDRON_FACES = [
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
    [1, 5, 10], [1, 10, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3],
    [3, 7, 8], [3, 8, 4], [4, 8, 9], [4, 9, 5], [5, 9, 10],
    [6, 10, 11], [7, 6, 11], [8, 7, 11], [9, 8, 11], [10, 9, 11],
]
_WINDOWS = [[0, 1, 2], [3, 7, 8], [4, 9, 5]]
_WINDOW_SET = {tuple(sorted(window)) for window in _WINDOWS}
_REGISTER_FACES = [
    face for face in _ICOSAHEDRON_FACES
    if tuple(sorted(face)) not in _WINDOW_SET
]
_GAMMA = {
    0: 3, 1: 7, 2: 8, 3: 4, 4: 0, 5: 2,
    6: 11, 7: 9, 8: 5, 9: 1, 10: 6, 11: 10,
}


@contextmanager
def squared_content_weights():
    previous = cob.HodgeLaplacian.defaultWeightConvention()
    cob.HodgeLaplacian.setDefaultWeightConvention(
        cob.HodgeWeightConvention.SquaredContent)
    try:
        yield
    finally:
        cob.HodgeLaplacian.setDefaultWeightConvention(previous)


def _edge_key(edge):
    a = int(edge.getSource().getId())
    b = int(edge.getTarget().getId())
    return min(a, b), max(a, b)


def _unit(vector):
    vector = np.asarray(vector, dtype=complex)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero vector has no normalized state")
    return vector / norm


def _matrix_payload(matrix):
    array = np.asarray(matrix, dtype=complex)
    return [[[float(value.real), float(value.imag)] for value in row]
            for row in array]


def _vector_payload(vector):
    return [[float(value.real), float(value.imag)]
            for value in np.asarray(vector, dtype=complex)]


def _phase_error(actual, expected):
    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    overlap = complex(np.vdot(expected.ravel(), actual.ravel()))
    if abs(overlap) == 0.0:
        return float(np.linalg.norm(actual - expected))
    return float(np.linalg.norm(
        actual - (overlap / abs(overlap)) * expected))


def lift_sector_operator(logical):
    logical = np.asarray(logical, dtype=complex)
    if logical.shape != (2, 2):
        raise ValueError("logical operator must be 2x2")
    return (
        _CHARGE_PROJECTOR
        + _SECTOR_BASIS.conj().T @ logical @ _SECTOR_BASIS
    )


def identity_operator():
    return np.eye(3, dtype=complex)


def cycle_operator():
    return np.array(
        [[0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]],
        dtype=complex,
    )


def generic_charge_preserving_operator(theta=0.37, phase=0.41):
    c = math.cos(theta)
    s = math.sin(theta)
    logical = np.array(
        [[c, -cmath.exp(-1j * phase) * s],
         [cmath.exp(1j * phase) * s, c]],
        dtype=complex,
    )
    return lift_sector_operator(logical)


def charge_leaking_operator(theta=0.43):
    basis = np.column_stack((_CHARGE, _SECTOR_BASIS.conj().T))
    c = math.cos(theta)
    s = math.sin(theta)
    in_basis = np.array(
        [[c, -s, 0.0],
         [s, c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=complex,
    )
    return basis @ in_basis @ basis.conj().T


def logical_operator(operator):
    return (
        _SECTOR_BASIS @ np.asarray(operator, dtype=complex)
        @ _SECTOR_BASIS.conj().T
    )


def charge_commutator_error(operator):
    operator = np.asarray(operator, dtype=complex)
    return float(np.linalg.norm(
        operator @ _CHARGE_PROJECTOR - _CHARGE_PROJECTOR @ operator))


def number_charge_commutator_error(operator):
    """Q=diag(0,1) charge-sector test for the historical qubit fixture."""
    operator = np.asarray(operator, dtype=complex)
    charge = np.diag([0.0, 1.0]).astype(complex)
    return float(np.linalg.norm(operator @ charge - charge @ operator))


class SpectralChoiCobordism:
    """Historical direct ``vec(U)`` inverse-eigenvector synthesis."""

    def __init__(self):
        self.spacetime = tessera.Spacetime.fromCells(
            2, [[0, 1, 2], [0, 1, 3]], 1.0, 0.0)
        for edge in self.spacetime.getEdgeList().toVector():
            edge.setLength(1.0)
            edge.setPhase(0.0)
        self.spacetime.materializeFacets()
        synthesis = cob.EigenstateSynthesis(self.spacetime, 0)
        self.support_cells = [[0], [1], [2], [3]]
        self.boundary_edges = {
            tuple(map(int, edge)) for edge in synthesis.boundaryEdges()
        }
        self.boundary_initial = self._boundary_snapshot()
        self.node = cob.MultiCobordism(
            host=self.spacetime,
            input_targets=[],
            output_targets=[],
            degrees=[0],
            einstein_hilbert=False,
        )

    def _boundary_snapshot(self):
        edge_map = {
            _edge_key(edge): edge
            for edge in self.spacetime.getEdgeList().toVector()
        }
        return {
            key: (
                complex(edge_map[key].getLength()) ** 2,
                complex(edge_map[key].getPhase()),
            )
            for key in self.boundary_edges
        }

    def relax_operator(
            self, operator, epsilon=_SPECTRAL_EPSILON, restarts=80,
            max_growth=4, seed=0, max_iterations=300,
            held_out_count=16):
        operator = np.asarray(operator, dtype=complex)
        if operator.shape != (2, 2):
            raise ValueError("historical fixture requires a 2x2 operator")
        target = np.asarray(choi.choiState(
            [complex(value) for value in operator.ravel()], 2),
            dtype=complex,
        )
        witness = self.node.relax_fixed_boundary_eigenstate(
            degree=0,
            support_cells=self.support_cells,
            target=[complex(value) for value in target],
            epsilon=float(epsilon),
            restarts=int(restarts),
            max_growth=int(max_growth),
            seed=int(seed),
            max_iterations=int(max_iterations),
        )

        synthesis = cob.EigenstateSynthesis(self.spacetime, 0)
        cell_index = {
            tuple(map(int, cell)): index
            for index, cell in enumerate(synthesis.cellSimplices())
        }
        state = np.asarray(witness.state, dtype=complex)
        support_state = _unit(np.asarray([
            state[cell_index[tuple(cell)]]
            for cell in self.support_cells
        ]))
        recovered = np.asarray(choi.operatorFromChoiState(
            [complex(value) for value in support_state], 2),
            dtype=complex,
        ).reshape(2, 2)
        phase = complex(np.vdot(operator.ravel(), recovered.ravel()))
        phase_aligned = (
            recovered * np.conj(phase / abs(phase))
            if abs(phase) > 0.0 else recovered
        )

        rng = np.random.default_rng(seed + 991)
        held_out_errors = []
        for _ in range(int(held_out_count)):
            input_state = _unit(
                rng.normal(size=2) + 1j * rng.normal(size=2))
            held_out_errors.append(float(np.linalg.norm(
                phase_aligned @ input_state - operator @ input_state)))

        boundary_final = self._boundary_snapshot()
        boundary_cells_final = {
            tuple(map(int, edge))
            for edge in synthesis.boundaryEdges()
        }
        boundary_drift = max((
            max(abs(boundary_final[key][component]
                    - self.boundary_initial[key][component])
                for component in range(2))
            for key in self.boundary_edges
        ), default=0.0)
        residual_cross_check = float(synthesis.residual(
            [complex(value) for value in state]))
        applied = np.asarray(synthesis.apply(
            [complex(value) for value in state]), dtype=complex)
        eigenvector_defect = float(math.sqrt(residual_cross_check))
        relative_eigenvector_defect = float(
            eigenvector_defect
            / max(float(np.linalg.norm(applied)), np.finfo(float).tiny)
        )
        return {
            "converged": bool(witness.converged),
            "residual": float(witness.residual),
            "residual_cross_check": residual_cross_check,
            "eigenvector_defect": eigenvector_defect,
            "relative_eigenvector_defect": relative_eigenvector_defect,
            "eigenvalue": float(witness.eigenvalue),
            "growth_steps": int(witness.growth_steps),
            "interior_vertex_count": int(
                witness.interior_vertex_count),
            "interior_edge_count": int(witness.interior_edge_count),
            "auxiliary_cell_count": int(witness.auxiliary_cell_count),
            "boundary_preserved": (
                boundary_cells_final == self.boundary_edges
                and boundary_final == self.boundary_initial),
            "boundary_drift": float(boundary_drift),
            "charge_commutator_error": (
                number_charge_commutator_error(operator)),
            "target_operator": _matrix_payload(operator),
            "recovered_operator": _matrix_payload(phase_aligned),
            "operator_error": float(np.linalg.norm(
                phase_aligned - operator)),
            "support_choi_overlap": float(abs(np.vdot(
                target, support_state))),
            "held_out_error_max": max(held_out_errors, default=0.0),
            "held_out_error_mean": (
                float(np.mean(held_out_errors))
                if held_out_errors else 0.0),
        }


def historical_spectral_experiment(
        epsilon=_SPECTRAL_EPSILON, restarts=80, max_growth=4,
        seed=0, max_iterations=300):
    phase_gate = np.diag([1.0, cmath.exp(0.41j)]).astype(complex)
    charge_changing = np.array(
        [[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    cases = {}
    for offset, (name, operator) in enumerate((
            ("charge_preserving_phase", phase_gate),
            ("charge_changing_x", charge_changing))):
        cases[name] = SpectralChoiCobordism().relax_operator(
            operator,
            epsilon=epsilon,
            restarts=restarts,
            max_growth=max_growth,
            seed=seed + offset,
            max_iterations=max_iterations,
        )
    checks = {
        "phase_gate_reaches_requested_precision": (
            cases["charge_preserving_phase"]["residual"] < epsilon),
        "charge_changing_gate_also_converges": (
            cases["charge_changing_x"]["residual"] < epsilon),
        "boundary_geometry_is_bit_identical": all(
            case["boundary_preserved"] for case in cases.values()),
        "support_recovers_each_choi_ray": all(
            case["support_choi_overlap"] > 1.0 - 1e-12
            for case in cases.values()),
        "held_out_application_matches": all(
            case["held_out_error_max"] < 1e-12
            for case in cases.values()),
    }
    return {
        "method": {
            "target": (
                "normalized vec(U) ray fixed on explicit 0-cells"),
            "free_state": "all non-support amplitudes",
            "free_geometry": "interior edges only",
            "residual": "||L psi - <psi,L psi> psi||^2",
            "regge": False,
            "period_constraints": False,
            "harmonic_eigenvalue": False,
            "charge_constraint": False,
            "epsilon": float(epsilon),
            "restarts": int(restarts),
            "max_growth": int(max_growth),
            "max_iterations": int(max_iterations),
            "seed": int(seed),
        },
        "cases": cases,
        "checks": checks,
        "interpretation": (
            "The historical inverse-eigenvector solver realizes the pinned "
            "Choi ray. Held-out application follows after unvectorizing that "
            "same pinned ray; it is not process learning or a target-free "
            "bulk readout. Charge conservation is neither encoded nor "
            "necessary for this numerical construction."
        ),
    }


class BoundaryPairCobordism:
    """Two prepared boundary circles joined by a relaxed annular bulk."""

    def __init__(self, include_charge_mode=False):
        base_circle = [[0, 1], [1, 2], [0, 2]]
        cells = tessera.Spacetime.prismCells(base_circle, 1, {})
        self.spacetime = tessera.Spacetime.fromCells(
            2, cells, 1.0, 0.0)
        for edge in self.spacetime.getEdgeList().toVector():
            edge.setLength(1.0)
            edge.setPhase(0.0)
        self.spacetime.materializeFacets()

        self.input_vertices = {0, 1, 2}
        self.output_vertices = {3, 4, 5}
        self.input_cells = [[vertex] for vertex in sorted(
            self.input_vertices)]
        self.output_cells = [[vertex] for vertex in sorted(
            self.output_vertices)]
        omega = cmath.exp(2j * math.pi / 3.0)
        neutral_basis = np.asarray(
            [[1.0, omega, omega * omega],
             [1.0, omega * omega, omega]],
            dtype=complex,
        ) / math.sqrt(3.0)
        self.include_charge_mode = bool(include_charge_mode)
        self.boundary_basis = (
            np.vstack((_CHARGE, neutral_basis))
            if self.include_charge_mode else neutral_basis
        )

        self.node = cob.MultiCobordism(
            host=self.spacetime,
            input_targets=[],
            output_targets=[],
            degrees=[0],
            einstein_hilbert=False,
        )
        self.node.declare_pinned_region(
            "input", self.input_vertices)
        self.node.declare_pinned_region(
            "output", self.output_vertices)
        self.edge_map = {
            _edge_key(edge): edge
            for edge in self.spacetime.getEdgeList().toVector()
        }
        self.pinned_edges = sorted(
            edge for edge in self.edge_map
            if self.node.edge_is_pinned(*edge)
        )
        self.free_edges = sorted(
            set(self.edge_map) - set(self.pinned_edges))
        if not self.free_edges:
            raise RuntimeError("annular fixture has no bulk edges")
        self.boundary_initial = self.boundary_snapshot()

    def boundary_snapshot(self):
        return {
            key: (
                complex(self.edge_map[key].getLength()) ** 2,
                complex(self.edge_map[key].getPhase()),
            )
            for key in self.pinned_edges
        }

    def relax_operator(
            self, operator, epsilon=1e-16, boundary_epsilon=1e-12,
            restarts=4, max_growth=8, seed=0, max_iterations=400,
            held_out_count=16, input_basis=None,
            common_eigenvalue=True):
        operator = np.asarray(operator, dtype=complex)
        dimension = self.boundary_basis.shape[0]
        if operator.shape != (dimension, dimension):
            raise ValueError(
                "boundary-pair fixture requires a square operator matching "
                "the prepared boundary basis")
        if not np.all(np.isfinite(operator)):
            raise ValueError("operator must contain only finite amplitudes")
        if input_basis is None:
            input_basis = np.eye(dimension, dtype=complex)
        input_basis = np.asarray(input_basis, dtype=complex)
        if input_basis.shape != (dimension, dimension):
            raise ValueError(
                "input basis must be square and match the operator")
        if not np.all(np.isfinite(input_basis)):
            raise ValueError(
                "input basis must contain only finite amplitudes")
        prepared_basis = input_basis.copy()
        for column in range(dimension):
            column_norm = float(np.linalg.norm(
                prepared_basis[:, column]))
            if not column_norm > 0.0:
                raise ValueError(
                    "input basis vectors must be nonzero")
            prepared_basis[:, column] /= column_norm
        if np.linalg.matrix_rank(prepared_basis) != dimension:
            raise ValueError("input basis must be linearly independent")

        input_states = input_basis.T @ self.boundary_basis
        output_states = (
            operator @ input_basis).T @ self.boundary_basis
        witness = self.node.relax_boundary_state_pairs(
            degree=0,
            input_region="input",
            input_cells=self.input_cells,
            input_states=input_states.tolist(),
            output_region="output",
            output_cells=self.output_cells,
            output_states=output_states.tolist(),
            common_eigenvalue=bool(common_eigenvalue),
            epsilon=float(epsilon),
            boundary_epsilon=float(boundary_epsilon),
            restarts=int(restarts),
            max_growth=int(max_growth),
            seed=int(seed),
            max_iterations=int(max_iterations),
        )

        synthesis = cob.EigenstateSynthesis(self.spacetime, 0)
        cell_index = {
            tuple(map(int, cell)): index
            for index, cell in enumerate(synthesis.cellSimplices())
        }
        input_indices = [
            cell_index[tuple(cell)] for cell in witness.input_cells
        ]
        output_indices = [
            cell_index[tuple(cell)] for cell in witness.output_cells
        ]
        states = np.asarray(witness.states, dtype=complex)
        fixed_inputs = np.asarray(witness.input_states, dtype=complex)
        fixed_outputs = np.asarray(witness.output_states, dtype=complex)
        measured_inputs = states[:, input_indices]
        measured_outputs = states[:, output_indices]
        restriction_error = float(max(
            np.max(np.abs(measured_inputs - fixed_inputs), initial=0.0),
            np.max(np.abs(measured_outputs - fixed_outputs), initial=0.0),
        ))

        input_coefficients = np.column_stack([
            self.boundary_basis.conj() @ state
            for state in measured_inputs
        ])
        output_coefficients = np.column_stack([
            self.boundary_basis.conj() @ state
            for state in measured_outputs
        ])
        recovered = output_coefficients @ np.linalg.inv(
            input_coefficients)

        rng = np.random.default_rng(seed + 1701)
        held_out_residuals = []
        held_out_input_errors = []
        held_out_output_errors = []
        held_out_operator_errors = []
        for _ in range(int(held_out_count)):
            coefficients = _unit(
                rng.normal(size=dimension)
                + 1j * rng.normal(size=dimension))
            combined = coefficients @ states
            expected_input = coefficients @ fixed_inputs
            expected_output = coefficients @ fixed_outputs
            held_out_input_errors.append(float(np.linalg.norm(
                combined[input_indices] - expected_input)))
            held_out_output_errors.append(float(np.linalg.norm(
                combined[output_indices] - expected_output)))
            logical_input = (
                self.boundary_basis.conj()
                @ combined[input_indices])
            logical_output = (
                self.boundary_basis.conj()
                @ combined[output_indices])
            held_out_operator_errors.append(float(np.linalg.norm(
                logical_output - operator @ logical_input)))
            held_out_residuals.append(float(synthesis.residual(
                [complex(value) for value in combined])))

        boundary_final = self.boundary_snapshot()
        boundary_drift = max((
            max(abs(boundary_final[key][component]
                    - self.boundary_initial[key][component])
                for component in range(2))
            for key in self.pinned_edges
        ), default=0.0)
        state_eigenvalues = np.asarray(
            witness.state_eigenvalues, dtype=float)
        return {
            "converged": bool(witness.converged),
            "residual": float(witness.residual),
            "residual_trace": [
                float(value) for value in witness.residual_trace],
            "state_residuals": [
                float(value) for value in witness.state_residuals],
            "state_eigenvalues": state_eigenvalues.tolist(),
            "common_eigenvalue": float(witness.eigenvalue),
            "common_eigenvalue_spread": float(np.max(
                np.abs(state_eigenvalues - witness.eigenvalue),
                initial=0.0,
            )),
            "growth_steps": int(witness.growth_steps),
            "free_edge_count": int(witness.free_edge_count),
            "auxiliary_cell_count": int(
                witness.auxiliary_cell_count),
            "input_boundary_residuals": [
                float(value)
                for value in witness.input_boundary_residuals
            ],
            "output_boundary_residuals": [
                float(value)
                for value in witness.output_boundary_residuals
            ],
            "boundary_preserved": (
                boundary_final == self.boundary_initial),
            "boundary_drift": float(boundary_drift),
            "restriction_error": restriction_error,
            "dimension": int(dimension),
            "common_eigenvalue_mode": bool(common_eigenvalue),
            "input_basis": _matrix_payload(input_basis),
            "input_basis_condition": float(np.linalg.cond(input_basis)),
            "prepared_input_condition": float(np.linalg.cond(
                input_coefficients)),
            "minimum_output_norm": float(min(
                np.linalg.norm(state) for state in output_states)),
            "target_operator": _matrix_payload(operator),
            "recovered_operator": _matrix_payload(recovered),
            "operator_error": float(np.linalg.norm(
                recovered - operator)),
            "operator_relative_error": float(
                np.linalg.norm(recovered - operator)
                / max(np.linalg.norm(operator), np.finfo(float).tiny)),
            "held_out_full_residual_max": max(
                held_out_residuals, default=0.0),
            "held_out_input_error_max": max(
                held_out_input_errors, default=0.0),
            "held_out_output_error_max": max(
                held_out_output_errors, default=0.0),
            "held_out_operator_error_max": max(
                held_out_operator_errors, default=0.0),
        }


def coupled_boundary_experiment(
        epsilon=1e-16, boundary_epsilon=1e-12, restarts=4,
        max_growth=8, seed=0, max_iterations=400):
    full_operator = generic_charge_preserving_operator()
    operator = logical_operator(full_operator)
    result = BoundaryPairCobordism().relax_operator(
        operator,
        epsilon=epsilon,
        boundary_epsilon=boundary_epsilon,
        restarts=restarts,
        max_growth=max_growth,
        seed=seed,
        max_iterations=max_iterations,
    )
    result["charge_commutator_error"] = charge_commutator_error(
        full_operator)
    result["method"] = {
        "boundary": (
            "two disconnected, independently prepared circle components"),
        "fixed_state_data": (
            "complete degree-zero cochain restriction on each component"),
        "fixed_geometry": "every intra-component edge length and phase",
        "free_geometry": "all non-pinned bulk edge weights and phases",
        "residual": (
            "sum_j ||L_W psi_j - lambda_bar psi_j||^2"),
        "common_eigenvalue": True,
        "regge": False,
        "period_constraints": False,
        "epsilon": float(epsilon),
        "boundary_epsilon": float(boundary_epsilon),
        "restarts": int(restarts),
        "max_growth": int(max_growth),
        "max_iterations": int(max_iterations),
        "seed": int(seed),
    }
    return result


class FrozenBoundaryTransport:
    """Target-free map from input/output restrictions of live ker L_1(W)."""

    def __init__(self, periods, sign_in, sign_out):
        periods = np.asarray(periods, dtype=complex)
        if periods.ndim != 2 or periods.shape[1] != 6:
            raise ValueError("period matrix must have six boundary columns")
        raw_in = periods[:, :3]
        raw_out = periods[:, 3:]
        self.input_restriction = (
            raw_in * np.asarray(sign_in)[None, :]
        ) @ _SECTOR_BASIS.conj().T
        self.output_restriction = (
            raw_out * np.asarray(sign_out)[None, :]
        ) @ _SECTOR_BASIS.conj().T
        if np.linalg.matrix_rank(self.input_restriction, tol=_TOL) != 2:
            raise ValueError("input restriction is not invertible")
        if np.linalg.matrix_rank(self.output_restriction, tol=_TOL) != 2:
            raise ValueError("output restriction is not invertible")
        self.transport = np.linalg.solve(
            self.input_restriction, self.output_restriction).T

    def apply(self, state):
        state = np.asarray(state, dtype=complex)
        if state.shape != (2,):
            raise ValueError("logical input must have two components")
        return self.transport @ state


class PeriodCobordism:
    """Fixed pair-of-pants prism driven only by explicit r_U constraints."""

    def __init__(self, twist=None, metric_source=None):
        cells = tessera.Spacetime.prismCells(
            _REGISTER_FACES, 1, twist or {})
        self.spacetime = tessera.Spacetime.fromCells(3, cells, 1.0, 0.0)
        for edge in self.spacetime.getEdgeList().toVector():
            a, b = _edge_key(edge)
            edge.setLength(1j if (a < 12) != (b < 12) else 1.0)
            edge.setPhase(0.0)
        self.spacetime.materializeFacets()

        self.synthesis = cob.EigenstateSynthesis(self.spacetime, 1)
        self.holes_in = [list(window) for window in _WINDOWS]
        self.holes_out = [
            [vertex + 12 for vertex in window] for window in _WINDOWS
        ]
        self.holes = self.holes_in + self.holes_out
        faces_out = [
            [vertex + 12 for vertex in face] for face in _REGISTER_FACES
        ]
        self.sign_in = np.asarray(
            cob.ChainComplex.endSignCovector(
                _REGISTER_FACES, self.holes_in),
            dtype=complex,
        )
        self.sign_out = np.asarray(
            cob.ChainComplex.endSignCovector(faces_out, self.holes_out),
            dtype=complex,
        )
        valid, reason = self.synthesis.dualComplexValid()
        if not valid:
            raise RuntimeError(f"invalid dual complex: {reason}")

        self.edge_map = {
            _edge_key(edge): edge
            for edge in self.spacetime.getEdgeList().toVector()
        }
        self.node = cob.MultiCobordism(
            host=self.spacetime,
            input_targets=[],
            output_targets=[],
            degrees=[1],
            gamma=1.0,
            seed=0,
            precone=0,
            should_propose_dispositions=False,
            precone_timelike=False,
            precone_alternate=False,
            balanced_edge_wiring=False,
            singular_value_ratio=False,
            einstein_hilbert=False,
            real_squared_lengths_only=True,
            metric_source=(metric_source if metric_source is not None
                           else cob.HodgeLaplacian.defaultMetricSource()),
        )
        # One region per boundary facet pins exactly the full boundary
        # subcomplex, including its lateral faces.
        for index, facet in enumerate(self.spacetime.getBoundary()):
            self.node.declare_pinned_region(
                f"boundary_facet_{index}", set(facet))

        expected_boundary = {
            tuple(map(int, edge)) for edge in self.synthesis.boundaryEdges()
        }
        actual_boundary = {
            edge for edge in self.edge_map
            if self.node.edge_is_pinned(*edge)
        }
        if actual_boundary != expected_boundary:
            raise RuntimeError("facet pinning did not select the full boundary")
        self.boundary_edges = sorted(expected_boundary)
        self.free_edges = sorted(set(self.edge_map) - expected_boundary)
        if not self.free_edges:
            raise RuntimeError("the experiment has no movable bulk edges")
        self.boundary_initial = self.boundary_snapshot()

    def boundary_snapshot(self):
        return {
            edge: complex(self.edge_map[edge].getLength()) ** 2
            for edge in self.boundary_edges
        }

    def boundary_drift(self):
        current = self.boundary_snapshot()
        return float(max(
            (abs(current[edge] - value)
             for edge, value in self.boundary_initial.items()),
            default=0.0,
        ))

    def free_squared_lengths(self):
        return np.asarray([
            complex(self.edge_map[edge].getLength()) ** 2
            for edge in self.free_edges
        ])

    def _raw_pair(self, raw_input, raw_output):
        return np.concatenate((
            self.sign_in * np.asarray(raw_input, dtype=complex),
            self.sign_out * np.asarray(raw_output, dtype=complex),
        ))

    def state_pair(self, operator, logical_input):
        logical_input = np.asarray(logical_input, dtype=complex)
        raw_input = _SECTOR_BASIS.conj().T @ logical_input
        raw_output = np.asarray(operator, dtype=complex) @ raw_input
        return self._raw_pair(raw_input, raw_output)

    def basis_pairs(self, operator):
        return [
            self.state_pair(operator, state)
            for state in np.eye(2, dtype=complex)
        ]

    def pin_state(self, name, operator, logical_input):
        target = self.state_pair(operator, logical_input)
        self.node.declare_register_constraint(
            name, 1, self.holes, [complex(value) for value in target])
        return target

    def pin_basis(self, operator):
        for index, target in enumerate(self.basis_pairs(operator)):
            self.node.declare_register_constraint(
                f"basis_{index}", 1, self.holes,
                [complex(value) for value in target])

    def hard_period_gap(self):
        return float(sum(
            self.synthesis.periodGapForPeriods(
                constraint["holes"], constraint["target"])
            for constraint in self.node.register_constraints()
        ))

    def snapshot(self):
        terms = self.node.objective_terms()
        return {
            "objective": float(self.node.objective()),
            "r_u": float(self.node.r_u(self.node.spacetime())),
            "hard_period_gap": self.hard_period_gap(),
            "regge_stationarity": float(terms.regge_stationarity),
            "register_residual_term": float(terms.register_residual),
            "boundary_drift": self.boundary_drift(),
            "max_imaginary_l2": float(np.max(
                np.abs(self.free_squared_lengths().imag), initial=0.0)),
        }

    def read_transport(self):
        flat = np.asarray(
            self.synthesis.cyclePeriods(self.holes), dtype=complex)
        if flat.size % len(self.holes) != 0:
            raise RuntimeError("period matrix has inconsistent shape")
        return FrozenBoundaryTransport(
            flat.reshape(-1, len(self.holes)),
            self.sign_in,
            self.sign_out,
        )

    def relax(self, iterations, alpha, animator=None):
        history = [self.snapshot()]
        if animator is not None:
            animator.update(history, self.free_squared_lengths().real)
        accepted_steps = 0
        for _ in range(int(iterations)):
            trace = self.node.run_stage2(
                beta=1.0, max_iters=1, alpha0=float(alpha),
                tolerance=1e-15)
            if not (len(trace) > 1 and trace[-1] < trace[0]):
                break
            accepted_steps += 1
            history.append(self.snapshot())
            if animator is not None:
                animator.update(
                    history, self.free_squared_lengths().real)
        return {
            "accepted_steps": accepted_steps,
            "history": history,
            "initial": history[0],
            "final": history[-1],
        }


class LiveAnimation:
    """Interactive geometry, period, and coupled-boundary diagnostics."""

    def __init__(self, edge_count):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import rcsetup

        backend = matplotlib.get_backend().lower()
        interactive = {name.lower() for name in rcsetup.interactive_bk}
        if backend not in interactive:
            raise RuntimeError(
                f"--live requires an interactive backend, got {backend}")
        self.plt = plt
        plt.ion()
        self.figure, (
            self.geometry_axis,
            self.residual_axis,
            self.transfer_axis,
        ) = plt.subplots(1, 3, figsize=(15, 4.5))
        self.weight_line, = self.geometry_axis.plot(
            np.arange(edge_count), np.zeros(edge_count), marker="o")
        self.r_u_line, = self.residual_axis.plot(
            [], [], marker="o", label="r_U")
        self.gap_line, = self.residual_axis.plot(
            [], [], marker="s", label="hard period gap")
        self.transfer_line, = self.transfer_axis.plot(
            [], [], marker="o", color="tab:green")
        self.geometry_axis.set(
            xlabel="free bulk edge index", ylabel="real squared length")
        self.residual_axis.set(
            xlabel="accepted step", ylabel="residual")
        self.transfer_axis.set(
            xlabel="interior-growth pass",
            ylabel="coupled full-W residual")
        self.residual_axis.set_yscale("log")
        self.transfer_axis.set_yscale("log")
        self.residual_axis.legend()
        self.figure.suptitle(
            "Geometric-operator relaxation with fixed boundaries")
        plt.show(block=False)

    def update(self, history, weights):
        steps = np.arange(len(history))
        tiny = np.finfo(float).tiny
        r_u = np.maximum([entry["r_u"] for entry in history], tiny)
        gap = np.maximum(
            [entry["hard_period_gap"] for entry in history], tiny)
        self.weight_line.set_ydata(weights)
        self.r_u_line.set_data(steps, r_u)
        self.gap_line.set_data(steps, gap)
        for axis in (self.geometry_axis, self.residual_axis):
            axis.relim()
            axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def update_transfer(self, residual_trace):
        tiny = np.finfo(float).tiny
        residuals = np.maximum(
            np.asarray(residual_trace, dtype=float), tiny)
        for stop in range(1, residuals.size + 1):
            self.transfer_line.set_data(
                np.arange(stop), residuals[:stop])
            self.transfer_axis.relim()
            self.transfer_axis.autoscale_view()
            self.figure.canvas.draw_idle()
            self.plt.pause(0.05)

    def finish(self):
        self.plt.ioff()
        self.plt.show()


def held_out_errors(fill, operator, seed, count=16):
    target = logical_operator(operator)
    transport = fill.read_transport()
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(int(count)):
        state = _unit(
            rng.normal(size=2) + 1j * rng.normal(size=2))
        errors.append(float(np.linalg.norm(
            transport.apply(state) - target @ state)))
    return {
        "target_logical": _matrix_payload(target),
        "transport": _matrix_payload(transport.transport),
        "transport_operator_error": float(np.linalg.norm(
            transport.transport - target)),
        "held_out_error_max": max(errors, default=0.0),
        "held_out_error_mean": (
            float(np.mean(errors)) if errors else 0.0),
    }


def _bulk_readout_payload(readout):
    payload = {
        "identifiable": bool(readout.identifiable),
        "obstruction": readout.obstruction,
        "bulk_cell_count": int(readout.bulk_cell_count),
        "kernel_dimension": int(readout.kernel_dimension),
        "frame_rank": int(readout.frame_rank),
        "frame_cells": [list(map(int, cell))
                        for cell in readout.frame_cells],
        "unitarity_error": float(readout.unitarity_error),
    }
    if readout.identifiable:
        payload["choi_state"] = _vector_payload(readout.choi_state)
        payload["operator"] = _matrix_payload(
            np.asarray(readout.operator_matrix).reshape(2, 2))
    return payload


def square_cycle_choi_control():
    base_cycle = [[0, 1], [1, 2], [2, 3], [0, 3]]
    cells = tessera.Spacetime.prismCells(base_cycle, 2, {})
    spacetime = tessera.Spacetime.fromCells(2, cells, 1.0, 0.0)
    spacetime.materializeFacets()
    node = cob.MultiCobordism(
        host=spacetime,
        input_targets=[],
        output_targets=[],
        degrees=[1],
        einstein_hilbert=False,
        real_squared_lengths_only=True,
    )
    readout = node.geometric_operator(2)
    if not readout.identifiable:
        raise RuntimeError(readout.obstruction)
    operator = np.asarray(
        readout.operator_matrix, dtype=complex).reshape(2, 2)
    state = np.asarray(readout.choi_state, dtype=complex)
    canonical_state = np.asarray(choi.choiState(
        [complex(value) for value in operator.ravel()], 2))
    sigma_y = np.array(
        [[0.0, -1j], [1j, 0.0]], dtype=complex)

    psi_a = _unit(np.array([0.31 + 0.2j, -0.7 + 0.1j]))
    psi_b = _unit(np.array([0.2 - 0.4j, 0.7 + 0.1j]))
    flat_operator = [complex(value) for value in operator.ravel()]
    amplitude = complex(choi.transitionAmplitude(
        [complex(value) for value in psi_a],
        flat_operator,
        [complex(value) for value in psi_b],
        2,
        2,
    ))
    transition = np.outer(psi_a, psi_b.conj()).ravel()
    dual_amplitude = complex(np.vdot(
        transition, operator.ravel()))

    edge_map = {
        _edge_key(edge): edge
        for edge in spacetime.getEdgeList().toVector()
    }
    for index, cell in enumerate(readout.bulk_cells):
        edge = edge_map[tuple(map(int, cell))]
        squared = complex(edge.getLength()) ** 2
        edge.setLength(cmath.sqrt(
            squared * (1.0 + 0.03 * (index + 1))))
    spacetime.materializeFacets()
    perturbed = node.geometric_operator(2)
    perturbed_operator = np.asarray(
        perturbed.operator_matrix, dtype=complex).reshape(2, 2)

    reordered_cells = list(readout.bulk_cells)
    reordered_cells[0], reordered_cells[1] = (
        reordered_cells[1], reordered_cells[0])
    reordered = node.geometric_operator(2, reordered_cells)
    reordered_operator = np.asarray(
        reordered.operator_matrix, dtype=complex).reshape(2, 2)

    result = _bulk_readout_payload(readout)
    result.update({
        "choi_convention_error": _phase_error(
            state, canonical_state),
        "metric_perturbation_error": _phase_error(
            perturbed_operator, operator),
        "frame_permutation_error": _phase_error(
            reordered_operator, operator),
        "charge_commutator_error": float(np.linalg.norm(
            operator @ sigma_y - sigma_y @ operator)),
        "transition_amplitude": [
            float(amplitude.real), float(amplitude.imag)],
        "choi_duality_error": float(abs(
            amplitude - dual_amplitude)),
    })
    return result


def run_experiment(
        iterations=12, alpha=0.05, seed=20260826, live=False,
        spectral_epsilon=_SPECTRAL_EPSILON, spectral_restarts=80,
        spectral_max_growth=4, spectral_max_iterations=300,
        spectral_seed=0, transfer_epsilon=1e-16,
        transfer_boundary_epsilon=1e-12, transfer_restarts=4,
        transfer_max_growth=8, transfer_max_iterations=400,
        transfer_seed=0):
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    if not float(alpha) > 0.0:
        raise ValueError("alpha must be positive")

    historical = historical_spectral_experiment(
        epsilon=spectral_epsilon,
        restarts=spectral_restarts,
        max_growth=spectral_max_growth,
        seed=spectral_seed,
        max_iterations=spectral_max_iterations,
    )
    if not all(historical["checks"].values()):
        failed = [
            name for name, passed in historical["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            "failed historical spectral checks: " + ", ".join(failed))

    with squared_content_weights():
        reflection = lift_sector_operator(np.diag([1.0, -1.0]))
        selected = np.array([1.0, 0.0], dtype=complex)
        unseen = np.array([0.0, 1.0], dtype=complex)
        single = PeriodCobordism()
        reflection_pair = single.pin_state(
            "selected_pair", reflection, selected)
        identity_pair = single.state_pair(
            identity_operator(), selected)
        single_transport = single.read_transport()
        single_case = {
            "diagnostics": single.snapshot(),
            "identity_and_reflection_pair_difference": float(
                np.linalg.norm(identity_pair - reflection_pair)),
            "unseen_reflection_error": float(np.linalg.norm(
                single_transport.apply(unseen) + unseen)),
            "transport": _matrix_payload(
                single_transport.transport),
            "bulk_readout": _bulk_readout_payload(
                single.node.geometric_operator(2)),
        }

        identity = PeriodCobordism()
        identity.pin_basis(identity_operator())
        identity_case = {
            "diagnostics": identity.snapshot(),
            **held_out_errors(identity, identity_operator(), seed),
            "bulk_readout": _bulk_readout_payload(
                identity.node.geometric_operator(2)),
        }

        cycle = PeriodCobordism(twist=_GAMMA)
        cycle.pin_basis(cycle_operator())
        cycle_case = {
            "diagnostics": cycle.snapshot(),
            **held_out_errors(
                cycle, cycle_operator(), seed + 101),
            "bulk_readout": _bulk_readout_payload(
                cycle.node.geometric_operator(2)),
        }

        target = generic_charge_preserving_operator()
        generic = PeriodCobordism()
        generic.pin_basis(target)
        animator = (
            LiveAnimation(len(generic.free_edges)) if live else None)
        coupled = coupled_boundary_experiment(
            epsilon=transfer_epsilon,
            boundary_epsilon=transfer_boundary_epsilon,
            restarts=transfer_restarts,
            max_growth=transfer_max_growth,
            seed=transfer_seed,
            max_iterations=transfer_max_iterations,
        )
        if animator is not None:
            animator.update_transfer(coupled["residual_trace"])
        relaxation = generic.relax(
            iterations, alpha, animator)
        generic_case = {
            "charge_commutator_error": (
                charge_commutator_error(target)),
            "relaxation": relaxation,
            **held_out_errors(generic, target, seed + 202),
            "bulk_readout": _bulk_readout_payload(
                generic.node.geometric_operator(2)),
        }

        leaking_target = charge_leaking_operator()
        leaking = PeriodCobordism()
        leaking.pin_basis(leaking_target)
        leaking_case = {
            "charge_commutator_error": (
                charge_commutator_error(leaking_target)),
            "diagnostics": leaking.snapshot(),
            **held_out_errors(
                leaking, leaking_target, seed + 303),
        }

        bulk_control = square_cycle_choi_control()
        checks = {
            "single_pair_has_tiny_period_residual": (
                single_case["diagnostics"]["r_u"]
                < _TINY_PERIOD_RESIDUAL),
            "single_pair_does_not_identify_operator": (
                single_case["unseen_reflection_error"] > 1.0),
            "complete_identity_generalizes": (
                identity_case["held_out_error_max"] < 1e-10),
            "mapping_class_generalizes": (
                cycle_case["held_out_error_max"] < 1e-10),
            "full_boundary_is_bit_identical": (
                relaxation["final"]["boundary_drift"] == 0.0),
            "objective_has_no_regge_term": (
                relaxation["final"]["regge_stationarity"] == 0.0),
            "residual_only_step_descends": (
                relaxation["final"]["r_u"]
                < relaxation["initial"]["r_u"]),
            "hard_gap_exposes_nonrealization": (
                relaxation["final"]["hard_period_gap"] > 1e-2
                and generic_case["transport_operator_error"] > 1e-2),
            "charge_conservation_is_not_sufficient": (
                generic_case["charge_commutator_error"] < 1e-12
                and generic_case["transport_operator_error"] > 1e-2),
            "charge_leak_is_detected": (
                leaking_case["charge_commutator_error"] > 1e-2
                and leaking_case["diagnostics"]["hard_period_gap"] > 1e-2),
            "rank_one_bulk_choi_is_promotable": (
                bulk_control["identifiable"]
                and bulk_control["unitarity_error"] < 1e-12),
            "choi_amplitude_identity_holds": (
                bulk_control["choi_duality_error"] < 1e-12),
            "coupled_boundary_pairs_converge": (
                coupled["converged"]
                and coupled["residual"] < transfer_epsilon),
            "coupled_boundary_data_are_exact": (
                coupled["boundary_preserved"]
                and coupled["boundary_drift"] == 0.0
                and coupled["restriction_error"] == 0.0),
            "coupled_operator_is_recovered": (
                coupled["operator_error"] < 1e-12),
            "coupled_span_generalizes": (
                coupled["held_out_input_error_max"] < 1e-12
                and coupled["held_out_output_error_max"] < 1e-12
                and coupled["held_out_full_residual_max"] < 1e-12),
        }
        if not all(checks.values()):
            failed = [
                name for name, passed in checks.items()
                if not passed
            ]
            raise RuntimeError(
                "failed checks: " + ", ".join(failed))

        result = {
            "method": {
                "primary": (
                    "historical fixed-boundary Rayleigh residual"),
                "coupled_boundary_value": (
                    "common-eigenvalue full-W Rayleigh residual"),
                "later_diagnostic": "ordered-period r_U only",
                "einstein_hilbert": False,
                "real_squared_lengths_only": True,
                "boundary": (
                    "full simplicial boundary pinned facet by facet"),
                "iterations": int(iterations),
                "alpha": float(alpha),
                "seed": int(seed),
            },
            "historical_fixed_boundary_spectral": historical,
            "coupled_boundary_state_transfer": coupled,
            "single_pair_ambiguity": single_case,
            "complete_basis_identity": identity_case,
            "complete_basis_mapping_class": cycle_case,
            "generic_charge_preserving": generic_case,
            "charge_leaking": leaking_case,
            "rank_one_bulk_choi_control": bulk_control,
            "checks": checks,
            "scientific_conclusion": (
                "The pre-paper method directly realizes a pinned Choi "
                "ray as a Laplacian eigenstate with frozen boundary "
                "geometry; it succeeds for both a charge-preserving "
                "phase gate and a charge-changing X gate. This does "
                "not establish charge conservation as a realizability "
                "criterion. The new two-boundary solve does recover a "
                "generic unitary on a complete prepared basis and extends "
                "to unseen linear combinations because all witnesses share "
                "one full-W eigenvalue. That is a boundary-conditioned "
                "graph-of-U certificate, not yet a target-free bulk-only "
                "operator: the later period experiment still shows that one "
                "pair is insufficient and that bulk promotion needs a frame "
                "and rank-one restriction."
            ),
        }
        if animator is not None:
            animator.finish()
        return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0])
    parser.add_argument(
        "--iterations", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--seed", type=int, default=20260826)
    parser.add_argument(
        "--spectral-epsilon", type=float,
        default=_SPECTRAL_EPSILON,
        help="historical Rayleigh-residual threshold")
    parser.add_argument(
        "--spectral-restarts", type=int, default=80)
    parser.add_argument(
        "--spectral-max-growth", type=int, default=4)
    parser.add_argument(
        "--spectral-max-iterations", type=int, default=300)
    parser.add_argument(
        "--spectral-seed", type=int, default=0)
    parser.add_argument(
        "--transfer-epsilon", type=float, default=1e-16,
        help="coupled full-W residual threshold")
    parser.add_argument(
        "--transfer-boundary-epsilon", type=float, default=1e-12,
        help="maximum isolated-boundary eigenresidual")
    parser.add_argument(
        "--transfer-restarts", type=int, default=4)
    parser.add_argument(
        "--transfer-max-growth", type=int, default=8)
    parser.add_argument(
        "--transfer-max-iterations", type=int, default=400)
    parser.add_argument(
        "--transfer-seed", type=int, default=0)
    parser.add_argument(
        "--live", action="store_true",
        help=(
            "animate bulk l^2, period residuals, and the coupled "
            "full-W residual"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "/tmp/cobordism/geometric_operators.json"))
    parser.add_argument(
        "--no-write", action="store_true",
        help="do not write the JSON record")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_experiment(
        iterations=args.iterations,
        alpha=args.alpha,
        seed=args.seed,
        live=args.live,
        spectral_epsilon=args.spectral_epsilon,
        spectral_restarts=args.spectral_restarts,
        spectral_max_growth=args.spectral_max_growth,
        spectral_max_iterations=args.spectral_max_iterations,
        spectral_seed=args.spectral_seed,
        transfer_epsilon=args.transfer_epsilon,
        transfer_boundary_epsilon=args.transfer_boundary_epsilon,
        transfer_restarts=args.transfer_restarts,
        transfer_max_growth=args.transfer_max_growth,
        transfer_max_iterations=args.transfer_max_iterations,
        transfer_seed=args.transfer_seed,
    )
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    historical = result["historical_fixed_boundary_spectral"]["cases"]
    print(
        "historical spectral:",
        "phase",
        f'{historical["charge_preserving_phase"]["residual"]:.6g},',
        "X",
        f'{historical["charge_changing_x"]["residual"]:.6g}',
    )
    coupled = result["coupled_boundary_state_transfer"]
    print(
        "coupled boundary transfer:",
        f'r_W={coupled["residual"]:.6g},',
        f'operator_error={coupled["operator_error"]:.6g},',
        f'held_out_r_W={coupled["held_out_full_residual_max"]:.6g}',
    )
    generic = result["generic_charge_preserving"]
    final = generic["relaxation"]["final"]
    print(
        "single-pair r_U:",
        f'{result["single_pair_ambiguity"]["diagnostics"]["r_u"]:.6g}',
    )
    print(
        "generic target:",
        f'r_U={final["r_u"]:.6g},',
        f'hard_gap={final["hard_period_gap"]:.6g},',
        f'transport_error={generic["transport_operator_error"]:.6g}',
    )
    print("conclusion:", result["scientific_conclusion"])
    if not args.no_write:
        print("record:", args.output)


if __name__ == "__main__":
    main()
