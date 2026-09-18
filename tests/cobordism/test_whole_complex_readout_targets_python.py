# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Whole-complex readout targets for a spanning set of boundary-state pairs
(#936): both boundary components are inputs, the whole complex is the output.

`relax_whole_complex_readout_targets` fixes the boundary geometry and both
components' amplitudes, imposes the readout constraints EXACTLY (each
witness's free amplitudes live on the affine solution set of its readout
system), and minimizes only the common-eigenvalue Rayleigh residual. The
always-on tests pin validation, exactness, and the algebraic class-3
obstruction at short budgets; the realizability gates run under
TESSERA_SLOW_TESTS=1.
"""
import cmath
import itertools
import math
import os

import numpy as np
import pytest

import tessera
from tessera import cobordism as cob

_FULL = bool(os.environ.get("TESSERA_SLOW_TESTS"))
_OMEGA = cmath.exp(2j * math.pi / 3.0)
_MODES = np.array([[1.0, _OMEGA, _OMEGA ** 2], [1.0, _OMEGA ** 2, _OMEGA]], dtype=complex) / math.sqrt(3.0)
_BASE = [[0, 1], [1, 2], [0, 2]]


def annulus(layers):
    cells = tessera.Spacetime.prismCells(_BASE, layers, {})
    st = tessera.Spacetime.fromVertexTuples(2, cells, 1.0, 0.0)
    for e in st.getEdgeList().toVector():
        e.setLength(1.0)
        e.setPhase(0.0)
    st.materializeFacets()
    top = 3 * layers
    a, b = [0, 1, 2], [top, top + 1, top + 2]
    interior = [v for v in range(3 * (layers + 1)) if v not in a and v not in b]
    node = cob.MultiCobordism(st, [], [], [0], einstein_hilbert=False)
    node.declare_pinned_region("A", set(a))
    node.declare_pinned_region("B", set(b))
    return node, a, b, interior


def frame_readouts(interior, size):
    n = len(interior)
    dft = np.array([[cmath.exp(-2j * math.pi * r * i / n) for i in range(n)] for r in range(n)]) / math.sqrt(n)
    frame = dft[:size]
    return frame, [[([v], complex(frame[r, i])) for i, v in enumerate(interior)] for r in range(size)]


def amplitudes(logical):
    return (_MODES.T @ np.asarray(logical, dtype=complex)).tolist()


def haar(n, rng):
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]


def spanning_problem(seed=0):
    rng = np.random.default_rng(seed)
    operator = haar(4, rng)
    spanning = haar(4, rng)
    inputs = [spanning[:, j] for j in range(4)]
    return operator, inputs, [operator @ x for x in inputs]


def fit(node, a, b, readouts, inputs, targets, restarts, growth, iterations, epsilon=1e-16):
    return node.relax_whole_complex_readout_targets(
        0, "A", [[v] for v in a], [amplitudes(x[:2]) for x in inputs],
        "B", [[v] for v in b], [amplitudes(x[2:]) for x in inputs],
        readouts, [[complex(v) for v in t] for t in targets],
        True, epsilon, 1e-12, restarts, growth, seed=0, max_iterations=iterations)


class TestValidation:
    def test_refuses_empty_readouts_and_mismatched_targets(self):
        node, a, b, interior = annulus(3)
        _, readouts = frame_readouts(interior, 4)
        _, inputs, targets = spanning_problem()
        with pytest.raises(ValueError, match="at least one readout chain"):
            fit(node, a, b, [], inputs, targets, 1, 0, 5)
        with pytest.raises(ValueError, match="readout target row"):
            fit(node, a, b, readouts, inputs, [t[:3] for t in targets], 1, 0, 5)
        with pytest.raises(ValueError, match="witness count mismatch"):
            fit(node, a, b, readouts, inputs, targets[:3], 1, 0, 5)

    def test_refuses_absent_cell_and_inconsistent_readout(self):
        node, a, b, interior = annulus(3)
        _, inputs, targets = spanning_problem()
        with pytest.raises(ValueError, match="absent from the live complex"):
            fit(node, a, b, [[([99], 1.0 + 0j)]], inputs, [[1.0 + 0j]] * 4, 1, 0, 5)
        # A chain supported on a boundary cell alone is already fixed by the
        # input; a target that disagrees has no solution.
        pinned_value = amplitudes(inputs[0][:2])[0]
        with pytest.raises(ValueError, match="inconsistent with its fixed amplitudes"):
            fit(node, a, b, [[([a[0]], 1.0 + 0j)]], inputs, [[pinned_value + 1.0]] * 4, 1, 0, 5)

    def test_refuses_non_eigenstate_boundary_but_admits_zero_component(self):
        node, a, b, interior = annulus(3)
        _, readouts = frame_readouts(interior, 4)
        with pytest.raises(ValueError, match="not an isolated-boundary eigenstate"):
            node.relax_whole_complex_readout_targets(
                0, "A", [[v] for v in a], [[1.0 + 0j, 0.5 + 0j, 0j]], "B", [[v] for v in b],
                [amplitudes([1.0, 0.0])], readouts, [[0j] * 4], True, 1e-16, 1e-12, 1, 0, 0, 5)
        res = node.relax_whole_complex_readout_targets(
            0, "A", [[v] for v in a], [[0j, 0j, 0j]], "B", [[v] for v in b],
            [amplitudes([1.0, 0.0])], readouts, [[0j] * 4], True, 1e-16, 1e-12, 1, 0, 0, 5)
        assert res.boundary_residuals_a == [0.0]
        assert math.isfinite(res.residual)


class TestExactness:
    """At any budget the returned witnesses restrict to the fixed boundary
    amplitudes and read out to the targets to round-off; only geometry and the
    readout null space moved."""

    def test_readouts_and_restrictions_are_exact(self):
        node, a, b, interior = annulus(3)
        frame, readouts = frame_readouts(interior, 4)
        _, inputs, targets = spanning_problem()
        res = fit(node, a, b, readouts, inputs, targets, 1, 0, 10)
        assert res.readout_rank == 4
        assert res.auxiliary_cell_count == 4 * (len(interior) - 4)
        assert res.readout_deviation < 1e-12
        assert res.growth_steps == 0
        assert math.isfinite(res.residual) and math.isfinite(res.eigenvalue)
        synthesis = cob.EigenstateSynthesis(node.spacetime(), 0)
        index = {tuple(c): i for i, c in enumerate(synthesis.cellSimplices())}
        for j, x in enumerate(inputs):
            state = np.asarray(res.states[j])
            np.testing.assert_allclose(state[[index[(v,)] for v in a]], np.asarray(res.states_a[j]), atol=0)
            np.testing.assert_allclose(state[[index[(v,)] for v in b]], np.asarray(res.states_b[j]), atol=0)
            np.testing.assert_allclose(frame @ state[[index[(v,)] for v in interior]], np.asarray(res.targets[j]),
                                       atol=1e-12)
            np.testing.assert_allclose(np.asarray(res.readouts[j]), np.asarray(res.targets[j]), atol=1e-12)
            # joint boundary normalization scales the target by the same factor
            scale = np.linalg.norm(np.concatenate([res.states_a[j], res.states_b[j]]))
            np.testing.assert_allclose(np.asarray(res.targets[j]), targets[j] * scale / np.linalg.norm(x), atol=1e-12)

    def test_growth_keeps_readout_chains(self):
        node, a, b, interior = annulus(2)
        frame, readouts = frame_readouts(interior, 2)
        _, inputs, targets = spanning_problem()
        res = fit(node, a, b, readouts, inputs, [t[:2] for t in targets], 1, 1, 5, epsilon=1e-30)
        assert res.growth_steps == 1
        assert res.readout_deviation < 1e-12
        assert len(res.states[0]) > 3 * 3


class TestProductObstruction:
    """The product spanning set {e_a ⊕ e_b} is dependent in the direct sum:
    a linear bulk forces Σ_j c_j Θ_j = 0 for c = (1, −1, −1, 1)."""

    @staticmethod
    def obstruction(tensor_operator):
        c = np.array([1, -1, -1, 1], dtype=complex)
        return np.linalg.norm(sum(cj * tensor_operator[:, 2 * a + b]
                                  for cj, (a, b) in zip(c, [(0, 0), (0, 1), (1, 0), (1, 1)])))

    def test_tensor_operators_are_obstructed_and_one_particle_operators_are_not(self):
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        assert self.obstruction(cnot) == pytest.approx(2.0)
        assert self.obstruction(np.eye(4, dtype=complex)) == pytest.approx(2.0)
        operator, _, _ = spanning_problem(3)
        inputs = [np.concatenate([np.eye(2)[:, a], np.eye(2)[:, b]]) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        c = np.array([1, -1, -1, 1], dtype=complex)
        assert np.linalg.norm(sum(cj * (operator @ x) for cj, x in zip(c, inputs))) < 1e-14


class TestRealizability:
    """Full gates (TESSERA_SLOW_TESTS=1): the one-particle spanning set
    converges and the frozen bulk reads held-out inputs; the CNOT product
    spanning set floors."""

    def _extension(self, node, a, b, interior, frame, lam, amp_a, amp_b):
        synthesis = cob.EigenstateSynthesis(node.spacetime(), 0)
        n = synthesis.order()
        L = np.zeros((n, n), dtype=complex)
        for i in range(n):
            e = [0j] * n
            e[i] = 1.0 + 0j
            L[:, i] = synthesis.apply(e)
        index = {tuple(c): i for i, c in enumerate(synthesis.cellSimplices())}
        ia, ib = [index[(v,)] for v in a], [index[(v,)] for v in b]
        boundary = ia + ib
        free = [i for i in range(n) if i not in boundary]
        psi = np.zeros(n, dtype=complex)
        psi[ia], psi[ib] = amp_a, amp_b
        psi[free] = -np.linalg.lstsq(L[np.ix_(free, free)] - lam * np.eye(len(free)),
                                     L[np.ix_(free, boundary)] @ psi[boundary], rcond=None)[0]
        return frame @ psi[[index[(v,)] for v in interior]]

    def test_short_budget_reports(self):
        node, a, b, interior = annulus(3)
        _, readouts = frame_readouts(interior, 4)
        _, inputs, targets = spanning_problem()
        res = fit(node, a, b, readouts, inputs, targets, 1, 0, 20)
        assert math.isfinite(res.residual) and len(res.states) == 4

    def test_one_mode_per_side_spanning_set_then_held_out(self):
        """One mode per side (a two-dimensional direct sum): the bulk encodes a
        generic U(2) between the A and B modes. Measured: converges to 7.2e-17
        after 4 growths in ~25 s; held-out reads agree to ~1e-8 (the square
        root of the residual)."""
        if not _FULL:
            pytest.skip("full realizability gate: set TESSERA_SLOW_TESTS=1")
        node, a, b, interior = annulus(3)
        frame, readouts = frame_readouts(interior, 2)
        rng = np.random.default_rng(0)
        operator, spanning = haar(2, rng), haar(2, rng)
        # joint logical (ψ_0, 0 | φ_0, 0): mode 0 of A and mode 0 of B
        inputs = [np.array([spanning[0, j], 0.0, spanning[1, j], 0.0]) for j in range(2)]
        targets = [operator @ spanning[:, j] for j in range(2)]
        res = fit(node, a, b, readouts, inputs, targets, 8, 12, 1000)
        assert res.converged, f"one-mode fit did not converge: residual {res.residual:.3e}"
        x = rng.normal(size=2) + 1j * rng.normal(size=2)
        read = self._extension(node, a, b, interior, frame, res.eigenvalue,
                               np.asarray(amplitudes([x[0], 0.0])), np.asarray(amplitudes([x[1], 0.0])))
        assert np.linalg.norm(read - operator @ x) < 1e-6 * np.linalg.norm(x)
        # attachment rotation of A's cells is an automorphism of the circle:
        # mode 0 picks up the phase ω, and the read follows the rotated input.
        amp_a = np.asarray(amplitudes([x[0], 0.0]))[[1, 2, 0]]
        rotated = _MODES.conj() @ amp_a
        assert abs(rotated[1]) < 1e-12
        read = self._extension(node, a, b, interior, frame, res.eigenvalue, amp_a,
                               np.asarray(amplitudes([x[1], 0.0])))
        assert np.linalg.norm(read - operator @ np.array([rotated[0], x[1]])) < 1e-6 * np.linalg.norm(x)

    def test_qubit_spanning_set_reports(self):
        """Both modes per side (a four-dimensional direct sum, four witnesses at
        one common eigenvalue). Measured at 4 restarts, growth 8, 400
        iterations on the 3-layer annulus: residual 2.5e-4, not converged —
        the realizability question #901/#903 characterize, recorded as
        measured; the experiment script carries the larger-budget runs."""
        if not _FULL:
            pytest.skip("full realizability gate: set TESSERA_SLOW_TESTS=1")
        node, a, b, interior = annulus(3)
        _, readouts = frame_readouts(interior, 4)
        operator, inputs, targets = spanning_problem()
        res = fit(node, a, b, readouts, inputs, targets, 4, 8, 400)
        assert math.isfinite(res.residual) and res.readout_deviation < 1e-12
