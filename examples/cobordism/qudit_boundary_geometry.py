# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Qudit operator transfer with the state carried by the boundary geometry.

The coupled fixture in :mod:`geometric_operators` presents a state as a cochain
of amplitudes laid on a boundary whose geometry is frozen at unit lengths and
zero phases.  That choice is load bearing in a way the construction does not
require.  Every pinned boundary state must be an eigenstate of the isolated
boundary, so a chart wider than one eigenvector must sit inside a degenerate
eigenspace, and on a connected boundary the only geometry carrying an
``(n-1)``-fold degeneracy is the equal-weight, zero-flux one.  Overall scale and
gauge leave it intact; any weight asymmetry or real flux splits it, and a
relative length asymmetry of ``1e-6`` already reaches the ``1e-12``
precondition.

This module carries the state in the boundary geometry instead.  A boundary
component is the boundary of a simplex, its geometry is generic, and its chart
is its own eigenframe.  Every chart vector is then an isolated-boundary
eigenstate by construction, at machine precision, with no degeneracy and no
symmetry: the precondition is satisfied rather than avoided.  The chart is the
full width of the boundary, so ``d`` cells carry a qudit of dimension ``d``
rather than ``d-1``.

The operator is read from the pair of geometries as ``V_out V_in^dagger`` rather
than prescribed, which makes every realizable operator unitary.  Prescribing a
state geometrically is also available through
:meth:`BoundaryGeometry.realizing`,
but one boundary carries at most one prescribed state at ``d = 3``, two from
``d = 4`` up, and never a complete frame.

The cobordism is a prism over that boundary.  At ``layers = 1`` a prism over the
boundary of a simplex has no interior at all: every vertex is boundary, every
witness amplitude is pinned, and a full-width chart leaves more equations than
unknowns, so no bulk exists to solve with at any budget.  ``layers >= 2``
supplies genuine interior vertices and free interior edges.

Run:

    python examples/cobordism/qudit_boundary_geometry.py
    python examples/cobordism/qudit_boundary_geometry.py --dimension 4 \
        --layers 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import tessera


cob = tessera.cobordism

_DEFAULT_EPSILON = 1e-16
_DEFAULT_BOUNDARY_EPSILON = 1e-12


def simplex_boundary_facets(dimension):
    """The facets of the boundary of a simplex on ``dimension`` vertices.

    ``dimension`` vertices give ``dimension`` facets, so the chart is exactly as
    wide as the qudit: a triangle for ``d = 3``, a tetrahedron surface for
    ``d = 4``, the five tetrahedra of a four-simplex boundary for ``d = 5``.
    """
    if dimension < 3:
        raise ValueError("a qudit boundary needs at least three cells")
    return [list(facet) for facet in itertools.combinations(
        range(dimension), dimension - 1)]


def vertex_pairs(dimension):
    """The one-skeleton of the simplex boundary, in a stable order."""
    return [(u, v)
            for u in range(dimension)
            for v in range(u + 1, dimension)]


def _edge_key(edge):
    a = int(edge.getSource().getId())
    b = int(edge.getTarget().getId())
    return min(a, b), max(a, b)


def _unit(vector):
    vector = np.asarray(vector, dtype=complex)
    norm = float(np.linalg.norm(vector))
    if not norm > 0.0:
        raise ValueError("zero vector has no normalized state")
    return vector / norm


def _matrix_payload(matrix):
    array = np.asarray(matrix, dtype=complex)
    return [[[float(value.real), float(value.imag)] for value in row]
            for row in array]


class BoundaryGeometry:
    """Edge weights and phases on the one-skeleton of a simplex boundary.

    The connection Laplacian the engine assembles at degree zero is
    ``L = D - A`` with ``A_uv = w_uv exp(i theta_uv)`` for ``u < v`` and
    ``w_uv = length^2``, so a weight and a phase per pair determine it.
    """

    def __init__(self, dimension, weights, phases):
        self.dimension = int(dimension)
        self.pairs = vertex_pairs(self.dimension)
        weights = np.asarray(weights, dtype=float)
        phases = np.asarray(phases, dtype=float)
        if weights.shape != (len(self.pairs),):
            raise ValueError("one weight per vertex pair is required")
        if phases.shape != (len(self.pairs),):
            raise ValueError("one phase per vertex pair is required")
        if not np.all(weights > 0.0):
            raise ValueError("edge weights must be positive")
        self.weights = weights
        self.phases = phases

    @classmethod
    def random(cls, dimension, rng, scale=0.4):
        """A generic geometry: unequal weights and nonzero flux."""
        weights = np.exp(rng.normal(scale=scale, size=len(
            vertex_pairs(dimension))))
        weights = weights / weights.sum()
        phases = rng.uniform(0.0, 2.0 * math.pi, size=len(
            vertex_pairs(dimension)))
        return cls(dimension, weights, phases)

    @classmethod
    def realizing(cls, state, attempts=24, seed=0):
        """A geometry whose Laplacian has ``state`` as an eigenvector.

        The weight scale is fixed while solving, because the residual vanishes
        trivially as the weights go to zero.
        """
        state = _unit(state)
        dimension = state.shape[0]
        pairs = vertex_pairs(dimension)

        def unpack(x):
            weights = np.exp(x[:len(pairs)])
            return weights / weights.sum(), x[len(pairs):2 * len(pairs)], x[-1]

        def residual(x):
            weights, phases, eigenvalue = unpack(x)
            deviation = (cls(dimension, weights, phases).laplacian() @ state
                         - eigenvalue * state)
            return np.concatenate([deviation.real, deviation.imag])

        best = None
        for attempt in range(attempts):
            rng = np.random.default_rng(seed + attempt)
            start = np.concatenate([
                rng.normal(size=len(pairs)),
                rng.uniform(0.0, 2.0 * math.pi, len(pairs)),
                [0.3],
            ])
            found = least_squares(residual, start, xtol=1e-15, ftol=1e-15,
                                  gtol=1e-15, max_nfev=60000)
            if best is None or found.cost < best.cost:
                best = found
        weights, phases, _ = unpack(best.x)
        geometry = cls(dimension, weights, phases)
        geometry.preparation_error = float(np.max(np.abs(residual(best.x))))
        return geometry

    def laplacian(self):
        """The degree-zero connection Laplacian, assembled algebraically."""
        matrix = np.zeros((self.dimension, self.dimension), dtype=complex)
        for (u, v), weight, phase in zip(self.pairs, self.weights, self.phases):
            matrix[u, v] = -weight * np.exp(1j * phase)
            matrix[v, u] = np.conj(matrix[u, v])
            matrix[u, u] += weight
            matrix[v, v] += weight
        return matrix

    def eigenframe(self):
        """The eigenvalues and eigenvectors that form this boundary's chart."""
        eigenvalues, frame = np.linalg.eigh(self.laplacian())
        return eigenvalues, frame

    def spacetime(self):
        """The isolated boundary component carrying this geometry."""
        facets = simplex_boundary_facets(self.dimension)
        boundary = tessera.Spacetime.fromCells(
            self.dimension - 2, facets, 1.0, 0.0)
        self.apply_to(boundary, offset=0)
        return boundary

    def apply_to(self, spacetime, offset):
        """Write the weights and phases onto ``spacetime``'s edges.

        Only edges whose endpoints both lie in ``[offset, offset + dimension)``
        are touched, so one call places one boundary component of a prism.
        """
        index = {pair: position for position, pair in enumerate(self.pairs)}
        for edge in spacetime.getEdgeList().toVector():
            source = int(edge.getSource().getId())
            target = int(edge.getTarget().getId())
            low, high = min(source, target), max(source, target)
            if low < offset or high >= offset + self.dimension:
                continue
            position = index[(low - offset, high - offset)]
            phase = float(self.phases[position])
            edge.setLength(complex(math.sqrt(float(self.weights[position]))))
            edge.setPhase(phase if source < target else -phase)

    def chart_residuals(self):
        """The isolated-boundary eigen-residual of each chart vector."""
        synthesis = cob.EigenstateSynthesis(self.spacetime(), 0)
        _, frame = self.eigenframe()
        return [float(synthesis.residual(
            [complex(value) for value in frame[:, column]]))
            for column in range(self.dimension)]


class QuditPairCobordism:
    """Two prepared simplex-boundary components joined by a relaxed prism.

    ``layers`` is the prism thickness.  It must be at least two: a single layer
    over a simplex boundary has no interior vertices, so every witness amplitude
    is pinned and a full-width chart admits no solution at any budget.
    """

    def __init__(self, dimension, input_geometry, output_geometry,
                 layers=3, attachment_permutation=None):
        self.dimension = int(dimension)
        if layers < 2:
            raise ValueError(
                "a prism of one layer has no interior; use layers >= 2")
        self.layers = int(layers)
        if attachment_permutation is None:
            attachment_permutation = tuple(range(self.dimension))
        attachment_permutation = tuple(
            int(value) for value in attachment_permutation)
        if sorted(attachment_permutation) != list(range(self.dimension)):
            raise ValueError(
                "the attachment permutation must be a permutation of the "
                "boundary cells")
        self.attachment_permutation = attachment_permutation
        self.effective_attachment = self._iterated_permutation(
            attachment_permutation, self.layers)
        self.input_geometry = input_geometry
        self.output_geometry = output_geometry

        base = simplex_boundary_facets(self.dimension)
        twist = {source: target
                 for source, target in enumerate(attachment_permutation)}
        cells = tessera.Spacetime.prismCells(base, self.layers, twist)
        self.spacetime = tessera.Spacetime.fromCells(
            self.dimension - 1, cells, 1.0, 0.0)
        for edge in self.spacetime.getEdgeList().toVector():
            edge.setLength(1.0)
            edge.setPhase(0.0)
        self.output_offset = self.dimension * self.layers
        input_geometry.apply_to(self.spacetime, offset=0)
        output_geometry.apply_to(self.spacetime, offset=self.output_offset)
        self.spacetime.materializeFacets()

        self.input_vertices = set(range(self.dimension))
        self.output_vertices = set(range(
            self.output_offset, self.output_offset + self.dimension))
        self.input_cells = [[vertex] for vertex in sorted(self.input_vertices)]
        self.output_cells = [[vertex]
                             for vertex in sorted(self.output_vertices)]

        self.node = cob.MultiCobordism(
            host=self.spacetime,
            input_targets=[],
            output_targets=[],
            degrees=[0],
            einstein_hilbert=False,
        )
        self.node.declare_pinned_region("input", self.input_vertices)
        self.node.declare_pinned_region("output", self.output_vertices)
        self.initial_interior_vertex_count = self.interior_vertex_count()
        self.edge_map = {_edge_key(edge): edge
                         for edge in self.spacetime.getEdgeList().toVector()}
        self.pinned_edges = sorted(
            key for key in self.edge_map
            if self.node.edge_is_pinned(*key))
        self.free_edges = sorted(set(self.edge_map) - set(self.pinned_edges))
        if not self.free_edges:
            raise RuntimeError("the prism has no free bulk edge")
        self.boundary_initial = self.boundary_snapshot()

    @staticmethod
    def _iterated_permutation(permutation, times):
        """``permutation`` composed with itself ``times`` times.

        ``prismCells`` applies its twist cumulatively per layer, so a prism of
        ``L`` layers glues its ends through the ``L``-th power of the twist.
        The effective attachment is that power, not the twist itself.
        """
        result = tuple(range(len(permutation)))
        for _ in range(times):
            result = tuple(permutation[value] for value in result)
        return result

    def boundary_snapshot(self):
        return {key: (complex(self.edge_map[key].getLength()) ** 2,
                      complex(self.edge_map[key].getPhase()))
                for key in self.pinned_edges}

    def interior_vertex_count(self):
        return (self.spacetime.getVertexCount()
                - len(self.input_vertices) - len(self.output_vertices))

    def _assembled_laplacian(self, synthesis):
        order = synthesis.order()
        matrix = np.zeros((order, order), dtype=complex)
        for column in range(order):
            basis = [0j] * order
            basis[column] = 1 + 0j
            matrix[:, column] = np.asarray(synthesis.apply(basis),
                                           dtype=complex)
        return matrix

    def emergent_output(self, synthesis, eigenvalue, input_indices,
                        output_indices, restriction):
        """The output a *new* input produces on the already-fitted geometry.

        Nothing about ``restriction`` was pinned during the relaxation.  The
        interior and outgoing amplitudes are solved from the fitted bulk by the
        unique interior lift of the boundary relation: the interior and outgoing
        rows of ``(L - lambda) z = 0`` determine them from the incoming
        restriction alone.  This is the one operator-level number in this
        module that linearity of the fitted witnesses does not already imply.
        """
        matrix = self._assembled_laplacian(synthesis)
        order = matrix.shape[0]
        shifted = matrix - eigenvalue * np.eye(order)
        unknown = [index for index in range(order)
                   if index not in set(input_indices)]
        rows = [index for index in range(order)
                if index not in set(input_indices)]
        block = shifted[np.ix_(rows, unknown)]
        driving = -shifted[np.ix_(rows, list(input_indices))] @ restriction
        solution, *_ = np.linalg.lstsq(block, driving, rcond=None)
        position = {index: place for place, index in enumerate(unknown)}
        return (np.array([solution[position[index]]
                          for index in output_indices], dtype=complex),
                float(np.linalg.cond(block)))

    def transfer(self, epsilon=_DEFAULT_EPSILON,
                 boundary_epsilon=_DEFAULT_BOUNDARY_EPSILON,
                 restarts=8, max_growth=4, seed=0, max_iterations=2000,
                 held_out_count=16):
        """Relax the bulk against the two boundaries' paired eigenframes.

        The pinned states are the chart vectors themselves, so the
        isolated-boundary eigenstate precondition holds by construction and
        nothing is bypassed to satisfy it.
        """
        input_values, input_frame = self.input_geometry.eigenframe()
        output_values, output_frame = self.output_geometry.eigenframe()
        represented = output_frame @ input_frame.conj().T

        witness = self.node.relax_boundary_state_pairs(
            degree=0,
            input_region="input",
            input_cells=self.input_cells,
            input_states=[input_frame[:, column].tolist()
                          for column in range(self.dimension)],
            output_region="output",
            output_cells=self.output_cells,
            output_states=[output_frame[:, column].tolist()
                           for column in range(self.dimension)],
            common_eigenvalue=True,
            epsilon=float(epsilon),
            boundary_epsilon=float(boundary_epsilon),
            restarts=int(restarts),
            max_growth=int(max_growth),
            seed=int(seed),
            max_iterations=int(max_iterations),
        )

        synthesis = cob.EigenstateSynthesis(self.spacetime, 0)
        cell_index = {tuple(map(int, cell)): position for position, cell
                      in enumerate(synthesis.cellSimplices())}
        input_indices = [cell_index[tuple(cell)]
                         for cell in witness.input_cells]
        output_indices = [cell_index[tuple(cell)]
                          for cell in witness.output_cells]
        states = np.asarray(witness.states, dtype=complex)
        fixed_inputs = np.asarray(witness.input_states, dtype=complex)
        fixed_outputs = np.asarray(witness.output_states, dtype=complex)
        measured_inputs = states[:, input_indices]
        measured_outputs = states[:, output_indices]
        restriction_error = float(max(
            np.max(np.abs(measured_inputs - fixed_inputs), initial=0.0),
            np.max(np.abs(measured_outputs - fixed_outputs), initial=0.0),
        ))

        input_coefficients = np.column_stack(
            [input_frame.conj().T @ state for state in measured_inputs])
        output_coefficients = np.column_stack(
            [output_frame.conj().T @ state for state in measured_outputs])
        recovered = (output_frame
                     @ output_coefficients
                     @ np.linalg.inv(input_coefficients)
                     @ input_frame.conj().T)
        operator_error = float(np.linalg.norm(recovered - represented))
        represented_norm = float(np.linalg.norm(represented))

        rng = np.random.default_rng(int(seed) + 1701)
        held_out_residuals = []
        held_out_operator_errors = []
        emergent_errors = []
        emergent_conditions = []
        for _ in range(int(held_out_count)):
            coefficients = _unit(rng.normal(size=self.dimension)
                                 + 1j * rng.normal(size=self.dimension))
            combined = coefficients @ states
            logical_input = input_frame.conj().T @ combined[input_indices]
            logical_output = output_frame.conj().T @ combined[output_indices]
            held_out_operator_errors.append(float(np.linalg.norm(
                output_frame @ logical_output
                - represented @ (input_frame @ logical_input))))
            held_out_residuals.append(float(synthesis.residual(
                [complex(value) for value in combined])))

            restriction = _unit(rng.normal(size=self.dimension)
                                + 1j * rng.normal(size=self.dimension))
            emergent, condition = self.emergent_output(
                synthesis, float(witness.eigenvalue), input_indices,
                output_indices, restriction)
            emergent_errors.append(float(np.linalg.norm(
                emergent - represented @ restriction)))
            emergent_conditions.append(condition)

        final = self.boundary_snapshot()
        boundary_drift = max(
            (max(abs(final[key][0] - self.boundary_initial[key][0]),
                 abs(final[key][1] - self.boundary_initial[key][1]))
             for key in self.pinned_edges), default=0.0)

        return {
            "dimension": self.dimension,
            "layers": self.layers,
            "attachment_permutation": list(self.attachment_permutation),
            "effective_attachment": list(self.effective_attachment),
            "residual": float(witness.residual),
            "growth_steps": int(witness.growth_steps),
            "common_eigenvalue": float(witness.eigenvalue),
            "free_edge_count": int(witness.free_edge_count),
            "auxiliary_cell_count": int(witness.auxiliary_cell_count),
            "interior_vertex_count": int(self.initial_interior_vertex_count),
            "grown_interior_vertex_count": int(self.interior_vertex_count()),
            "boundary_drift": float(abs(boundary_drift)),
            "restriction_error": restriction_error,
            "input_chart_residual_max": float(max(
                self.input_geometry.chart_residuals())),
            "output_chart_residual_max": float(max(
                self.output_geometry.chart_residuals())),
            "input_boundary_eigenvalues": [float(value)
                                           for value in input_values],
            "output_boundary_eigenvalues": [float(value)
                                            for value in output_values],
            "operator_error": operator_error,
            "operator_relative_error": (operator_error / represented_norm
                                        if represented_norm > 0.0 else
                                        operator_error),
            "operator_unitarity_error": float(np.max(np.abs(
                represented.conj().T @ represented
                - np.eye(self.dimension)))),
            "held_out_operator_error_max": float(max(held_out_operator_errors)),
            "held_out_full_residual_max": float(max(held_out_residuals)),
            "emergent_transfer_error_max": float(max(emergent_errors)),
            "emergent_transfer_error_mean": float(np.mean(emergent_errors)),
            "emergent_lift_condition_max": float(max(emergent_conditions)),
            "represented_operator": _matrix_payload(represented),
            "recovered_operator": _matrix_payload(recovered),
        }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimension", type=int, default=4,
                        help="qudit dimension; the boundary carries this many "
                             "cells")
    parser.add_argument("--layers", type=int, default=3,
                        help="prism thickness; must be at least two")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-growth", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    fixture = QuditPairCobordism(
        args.dimension,
        BoundaryGeometry.random(args.dimension, rng),
        BoundaryGeometry.random(args.dimension, rng),
        layers=args.layers,
    )
    record = fixture.transfer(restarts=args.restarts,
                              max_growth=args.max_growth,
                              max_iterations=args.max_iterations,
                              seed=args.seed)
    summary = {key: value for key, value in record.items()
               if key not in ("represented_operator", "recovered_operator")}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True),
                               encoding="utf-8")
        print("record:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
