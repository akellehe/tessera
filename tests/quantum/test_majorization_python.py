"""Python-level tests: pure majorization predicate, Poset
construction via :class:`Majorization`, and the end-to-end
:meth:`SchwingerModel.solve_with_majorization` pipeline through the
``tessera.quantum`` API.

Mirrors the C++-side tests in test_majorization.cpp,
test_schmidt_spectra.cpp, test_majorization_poset.cpp, and
test_schwinger_schmidt_cross_check.cpp at the Python boundary.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import unittest
from itertools import permutations

try:
    from tessera import Poset
    from tessera.quantum import (
        QuantumConfig,
        Interval,
        SchmidtSpectra,
        Majorization,
        StandardMajorization,
        SchwingerModel,
    )
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


# Tolerance for float comparisons.
TOL = 1e-10

# Module-level predicate used for the pure-function tests below. The
# classical {N1999} order with default tolerance is what the old free
# functions wrapped.
CLASSICAL = StandardMajorization() if HAVE_QUANTUM else None


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestStandardMajorization(unittest.TestCase):
    """Pure tests on the classical :class:`StandardMajorization` predicate."""

    def test_reflexivity(self) -> None:
        for v in ([1.0], [0.5, 0.5], [0.7, 0.2, 0.1]):
            self.assertTrue(CLASSICAL.majorizes(v, v))
            self.assertFalse(CLASSICAL.strictly_majorizes(v, v))

    def test_canonical_strict_pair(self) -> None:
        self.assertTrue(CLASSICAL.majorizes([1.0, 0.0], [0.5, 0.5]))
        self.assertFalse(CLASSICAL.majorizes([0.5, 0.5], [1.0, 0.0]))
        self.assertTrue(CLASSICAL.strictly_majorizes([1.0, 0.0], [0.5, 0.5]))

    def test_zero_padding_invariance(self) -> None:
        self.assertTrue(CLASSICAL.majorizes([0.5, 0.5, 0.0, 0.0], [0.5, 0.5]))
        self.assertTrue(CLASSICAL.majorizes([0.5, 0.5], [0.5, 0.5, 0.0]))
        self.assertFalse(CLASSICAL.strictly_majorizes([0.5, 0.5, 0.0], [0.5, 0.5]))

    def test_sort_invariance(self) -> None:
        a = [0.7, 0.2, 0.1]
        for perm in permutations(a):
            self.assertTrue(CLASSICAL.majorizes(list(perm), a))
            self.assertTrue(CLASSICAL.majorizes(a, list(perm)))

    def test_transitivity(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.5, 0.5, 0.0]
        c = [1/3, 1/3, 1/3]
        self.assertTrue(CLASSICAL.strictly_majorizes(a, b))
        self.assertTrue(CLASSICAL.strictly_majorizes(b, c))
        self.assertTrue(CLASSICAL.strictly_majorizes(a, c))

    def test_unequal_total_mass_rejected(self) -> None:
        self.assertFalse(CLASSICAL.majorizes([1.0, 0.0], [0.5, 0.5, 0.5]))
        self.assertFalse(CLASSICAL.majorizes([0.5, 0.5, 0.5], [1.0, 0.0]))

    def test_incomparable(self) -> None:
        a = [0.5, 0.4, 0.1]
        b = [0.6, 0.2, 0.2]
        self.assertFalse(CLASSICAL.majorizes(a, b))
        self.assertFalse(CLASSICAL.majorizes(b, a))

    def test_tol_is_effective(self) -> None:
        # Tighter tolerance: 1e-10 absorbs sub-1e-12 disagreements.
        loose = StandardMajorization(1e-10)
        a = [0.5, 0.5]
        b = [0.5 + 5e-13, 0.5 - 5e-13]
        self.assertTrue(loose.majorizes(a, b))
        self.assertTrue(loose.majorizes(b, a))


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestMajorizationPoset(unittest.TestCase):
    """Tests on :meth:`Majorization.poset_of` and the Poset struct."""

    def test_empty_input(self) -> None:
        p = Majorization.poset_of([])
        self.assertEqual(p.get_node_count, 0)
        self.assertEqual(p.covers, [])

    def test_single_node(self) -> None:
        p = Majorization.poset_of([[1.0]])
        self.assertEqual(p.get_node_count, 1)
        self.assertEqual(p.covers, [])

    def test_canonical_chain_transitive_reduction(self) -> None:
        spectra = [
            [1/3, 1/3, 1/3],  # 0 — most uniform
            [0.5, 0.5],        # 1 — middle
            [1.0],             # 2 — most concentrated
        ]
        p = Majorization.poset_of(spectra)
        self.assertEqual(p.get_node_count, 3)
        self.assertEqual(set(p.covers), {(2, 1), (1, 0)})

    def test_equivalent_spectra_no_strict_edges(self) -> None:
        p = Majorization.poset_of([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(p.covers, [])

    def test_covers_only_have_in_range_indices(self) -> None:
        p = Majorization.poset_of([[1.0], [0.5, 0.5], [1/3]*3])
        for a, b in p.covers:
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, p.get_node_count)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, p.get_node_count)
            self.assertNotEqual(a, b)

    def test_predicate_overload(self) -> None:
        """The predicate-explicit overload matches the tol overload at the
        default tolerance."""
        spectra = [[1/3]*3, [0.5, 0.5], [1.0]]
        p_tol  = Majorization.poset_of(spectra)
        p_pred = Majorization.poset_of(spectra, StandardMajorization())
        self.assertEqual(set(p_tol.covers), set(p_pred.covers))


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSolveWithMajorization(unittest.TestCase):
    """End-to-end pipeline tests: DMRG → Schmidt → Poset via
    :meth:`SchwingerModel.solve_with_majorization`."""

    @staticmethod
    def _basic_config(N: int, m: float = 0.0, g: float = 1.0,
                      L0: float = 0.0,
                      maxBondDim: int = 32, nSweeps: int = 8) -> "QuantumConfig":
        cfg = QuantumConfig()
        cfg.N = N; cfg.a = 1.0; cfg.g = g; cfg.m = m; cfg.L0 = L0
        cfg.max_bond_dim = maxBondDim; cfg.n_sweeps = nSweeps
        return cfg

    def test_n6_pipeline_basic(self) -> None:
        cfg = self._basic_config(N=6)
        r = SchwingerModel(cfg).solve_with_majorization()

        self.assertEqual(r.spectra.N, 6)
        self.assertLess(r.ground_state.energy, 0)
        self.assertGreater(r.ground_state.bond_dim, 0)

        expected_intervals = {
            (i, j) for i in range(1, 7) for j in range(i, 7)
            if not (i == 1 and j == 6)
        }
        seen = {(iv.i, iv.j) for iv in r.spectra.intervals}
        self.assertEqual(expected_intervals, seen)
        self.assertEqual(len(r.spectra.intervals), len(r.spectra.spectra))

    def test_spectra_normalize_to_one(self) -> None:
        cfg = self._basic_config(N=6)
        r = SchwingerModel(cfg).solve_with_majorization()
        for spec, iv in zip(r.spectra.spectra, r.spectra.intervals):
            total = sum(spec)
            self.assertAlmostEqual(
                total, 1.0, places=8,
                msg=f"interval [{iv.i}, {iv.j}] sums to {total}")

    def test_complement_symmetry(self) -> None:
        """Schmidt spectrum of [1, j] equals that of [j+1, N]."""
        cfg = self._basic_config(N=6, m=0.0)
        r = SchwingerModel(cfg).solve_with_majorization()
        by_iv = {(iv.i, iv.j): spec for iv, spec
                 in zip(r.spectra.intervals, r.spectra.spectra)}
        for j in range(1, 6):
            left  = sorted(by_iv[(1, j)],     reverse=True)
            right = sorted(by_iv[(j+1, 6)],   reverse=True)
            n = max(len(left), len(right))
            left  += [0.0] * (n - len(left))
            right += [0.0] * (n - len(right))
            for k in range(n):
                self.assertAlmostEqual(
                    left[k], right[k], places=8,
                    msg=f"[1, {j}] vs [{j+1}, 6] spectra disagree at idx {k}")

    def test_n4_product_state_limit(self) -> None:
        """At m → ∞ the GS is approximately a Néel product state."""
        cfg = self._basic_config(N=4, m=200.0, maxBondDim=32, n_sweeps=10)
        r = SchwingerModel(cfg).solve_with_majorization(tol=1e-3)
        for spec, iv in zip(r.spectra.spectra, r.spectra.intervals):
            self.assertAlmostEqual(
                max(spec), 1.0, places=3,
                msg=f"product-state limit broken at [{iv.i}, {iv.j}]: {spec}")
        self.assertEqual(r.poset.get_node_count, len(r.spectra.spectra))
        self.assertEqual(r.poset.covers, [])

    def test_poset_is_transitively_reduced(self) -> None:
        cfg = self._basic_config(N=6)
        r = SchwingerModel(cfg).solve_with_majorization()
        succ = {a: set() for a in range(r.poset.get_node_count)}
        for a, b in r.poset.covers:
            succ[a].add(b)
        for (a, b) in r.poset.covers:
            for c in range(r.poset.get_node_count):
                if c == a or c == b:
                    continue
                if c in succ[a] and b in succ[c]:
                    self.fail(
                        f"poset has cover ({a}, {b}) but intermediate "
                        f"path {a} → {c} → {b} also exists")

    def test_poset_irreflexive(self) -> None:
        cfg = self._basic_config(N=6)
        r = SchwingerModel(cfg).solve_with_majorization()
        for a, b in r.poset.covers:
            self.assertNotEqual(a, b, "self-loop in Hasse diagram")

    def test_poset_acyclic(self) -> None:
        cfg = self._basic_config(N=6)
        r = SchwingerModel(cfg).solve_with_majorization()

        in_deg = {a: 0 for a in range(r.poset.get_node_count)}
        succ   = {a: [] for a in range(r.poset.get_node_count)}
        for a, b in r.poset.covers:
            succ[a].append(b)
            in_deg[b] += 1
        queue = [n for n in range(r.poset.get_node_count) if in_deg[n] == 0]
        processed = 0
        while queue:
            n = queue.pop()
            processed += 1
            for m in succ[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)
        self.assertEqual(processed, r.poset.get_node_count,
                         "directed cycle present in Hasse cover edges")

    def test_consistency_with_solve(self) -> None:
        """solveWithMajorization should agree with solve on the energy /
        bondDim fields when run with the same config."""
        cfg = self._basic_config(N=6, m=0.125)
        r1 = SchwingerModel(cfg).solve()
        r2 = SchwingerModel(cfg).solve_with_majorization()
        self.assertAlmostEqual(r1.energy, r2.ground_state.energy, places=10)
        self.assertAlmostEqual(r1.operator_energy,
                               r2.ground_state.operator_energy, places=10)
        self.assertEqual(r1.bond_dim, r2.ground_state.bond_dim)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestPosetRepr(unittest.TestCase):
    def test_repr(self) -> None:
        p = Majorization.poset_of([[1.0], [0.5, 0.5]])
        text = repr(p)
        self.assertIn("Poset", text)
        self.assertIn("getNodeCount=2", text)
        self.assertIn("edges", text)

    def test_interval_repr(self) -> None:
        cfg = TestSolveWithMajorization._basic_config(N=4)
        r = SchwingerModel(cfg).solve_with_majorization()
        text = repr(r.spectra.intervals[0])
        self.assertIn("Interval", text)
        self.assertIn("i=", text)
        self.assertIn("j=", text)
