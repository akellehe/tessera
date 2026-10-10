"""Class-semantics tests for the OO surface of ``tessera.quantum``.

This file targets behaviour that is *intrinsic to the class shape* —
config introspection, statelessness, static-class non-instantiability,
predicate hierarchy semantics — distinct from the per-class workflow
correctness covered in test_schwinger_*.py, test_majorization_python.py,
test_causal_compare_python.py, and test_causet_chain_python.py:

* :class:`SchwingerModel`  — config introspection, statelessness (same
  config in, same result out), cross-method consistency.
* :class:`SchwingerQuench` — same, plus the ``predicate`` keyword on
  :meth:`compareCausalOrders`.
* :class:`Majorization`    — static-class semantics (no instances),
  predicate-explicit ``posetOf`` overload, polymorphism across the
  three variants.
* :class:`Causet`          — static-class semantics, idempotency.
* :class:`CausalOrders`    — standalone :meth:`fromSnapshots` factory.
* :class:`LogConcaveMajorization` — ``isLogConcave`` static utility
  edge cases.
* :class:`PeakRadialMajorization` — strictly-stronger-than-classical
  property at the canonical witness.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import unittest

try:
    from tessera.quantum import (
        QuantumConfig,
        TDVPConfig,
        SchwingerModel,
        SchwingerQuench,
        Majorization,
        Causet,
        CausalOrders,
        MajorizationPredicate,
        StandardMajorization,
        LogConcaveMajorization,
        PeakRadialMajorization,
    )
    from tessera import Poset
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _basic_quantum_config(N: int = 6, m: float = 0.0) -> "QuantumConfig":
    cfg = QuantumConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0; cfg.m = m; cfg.L0 = 0.0
    cfg.maxBondDim = 32; cfg.nSweeps = 8
    return cfg


def _light_quark_tdvp_config(N: int = 8, T: float = 0.4) -> "TDVPConfig":
    cfg = TDVPConfig()
    cfg.N = N; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.5; cfg.L0 = 0.0
    cfg.dmrgMaxBondDim = 32; cfg.dmrgNSweeps = 10
    cfg.i0 = 1; cfg.d = 3
    cfg.dt = 0.2; cfg.T = T; cfg.snapshotEvery = 1
    cfg.maxBondDim = 60
    cfg.cutoff = 1e-10; cfg.krylovDim = 12
    cfg.quiet = True; cfg.conserveQns = True
    return cfg


# ─── SchwingerModel ────────────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSchwingerModelClassSemantics(unittest.TestCase):
    """SchwingerModel binds a QuantumConfig and exposes solve /
    solveWithMajorization. It carries no other state, so repeated calls
    must produce identical numbers."""

    def test_config_property_reflects_constructor_argument(self) -> None:
        cfg = _basic_quantum_config(N=8, m=0.125)
        cfg.L0 = 0.25
        model = SchwingerModel(cfg)
        self.assertEqual(model.config.N, 8)
        self.assertEqual(model.config.m, 0.125)
        self.assertEqual(model.config.L0, 0.25)
        self.assertEqual(model.config.maxBondDim, cfg.maxBondDim)

    def test_solve_is_idempotent_on_same_instance(self) -> None:
        model = SchwingerModel(_basic_quantum_config(N=6, m=0.25))
        a = model.solve()
        b = model.solve()
        self.assertAlmostEqual(a.energy, b.energy, places=12)
        self.assertAlmostEqual(a.operatorEnergy, b.operatorEnergy, places=12)
        self.assertAlmostEqual(a.constant, b.constant, places=12)
        self.assertEqual(a.bondDim, b.bondDim)

    def test_solve_matches_across_instances_with_same_config(self) -> None:
        """Constructing two SchwingerModel objects from equivalent configs
        must give the same answer — there's no hidden per-instance state."""
        cfg = _basic_quantum_config(N=6, m=0.125)
        a = SchwingerModel(cfg).solve()
        b = SchwingerModel(cfg).solve()
        self.assertAlmostEqual(a.energy, b.energy, places=12)
        self.assertEqual(a.bondDim, b.bondDim)

    def test_solve_vs_solveWithMajorization_agree(self) -> None:
        """The two methods must produce the same ground-state diagnostics
        — solveWithMajorization is solve + extra Schmidt work."""
        cfg = _basic_quantum_config(N=6, m=0.125)
        model = SchwingerModel(cfg)
        a = model.solve()
        b = model.solveWithMajorization()
        self.assertAlmostEqual(a.energy, b.groundState.energy, places=10)
        self.assertAlmostEqual(a.operatorEnergy,
                               b.groundState.operatorEnergy, places=10)
        self.assertEqual(a.bondDim, b.groundState.bondDim)

    def test_solveWithMajorization_idempotent(self) -> None:
        model = SchwingerModel(_basic_quantum_config(N=4))
        a = model.solveWithMajorization()
        b = model.solveWithMajorization()
        self.assertEqual(a.spectra.N, b.spectra.N)
        self.assertEqual(len(a.spectra.intervals), len(b.spectra.intervals))
        self.assertEqual(set(a.poset.covers), set(b.poset.covers))


# ─── SchwingerQuench ───────────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestSchwingerQuenchClassSemantics(unittest.TestCase):
    """SchwingerQuench binds a TDVPConfig and exposes evolve /
    compareCausalOrders. Same statelessness contract as SchwingerModel."""

    def test_config_property_reflects_constructor_argument(self) -> None:
        cfg = _light_quark_tdvp_config(N=6, T=0.2)
        quench = SchwingerQuench(cfg)
        self.assertEqual(quench.config.N, 6)
        self.assertAlmostEqual(quench.config.T, 0.2)
        self.assertAlmostEqual(quench.config.dt, cfg.dt)

    def test_evolve_is_idempotent_on_same_instance(self) -> None:
        """Two TDVP runs on the same instance must give bit-identical
        snapshot energies and L_n profiles."""
        cfg = _light_quark_tdvp_config(N=6, T=0.2)
        quench = SchwingerQuench(cfg)
        a = quench.evolve()
        b = quench.evolve()
        self.assertEqual(len(a.snapshots), len(b.snapshots))
        for sa, sb in zip(a.snapshots, b.snapshots):
            self.assertAlmostEqual(sa.time, sb.time, places=12)
            self.assertAlmostEqual(sa.energy, sb.energy, places=10)
            self.assertEqual(sa.bondDim, sb.bondDim)
            for la, lb in zip(sa.lProfile, sb.lProfile):
                self.assertAlmostEqual(la, lb, places=10)

    def test_default_predicate_matches_explicit_standard(self) -> None:
        """compareCausalOrders without a predicate uses StandardMajorization
        internally; passing one explicitly with default tol must give the
        same report."""
        cfg = _light_quark_tdvp_config(N=6, T=0.2)
        quench = SchwingerQuench(cfg)
        r_default = quench.compareCausalOrders(vLr=1.0)
        r_explicit = quench.compareCausalOrders(
            vLr=1.0, predicate=StandardMajorization())
        self.assertEqual(r_default.majKind, "standard")
        self.assertEqual(r_explicit.majKind, "standard")
        self.assertEqual(r_default.nLabels, r_explicit.nLabels)
        # The maj poset is the only thing the predicate controls; on
        # identical inputs both Hasse cover lists should match.
        self.assertEqual(r_default.majVsLr.nConcordant,
                         r_explicit.majVsLr.nConcordant)
        self.assertEqual(r_default.majVsLr.nDiscordant,
                         r_explicit.majVsLr.nDiscordant)

    def test_majKind_tracks_predicate_name(self) -> None:
        cfg = _light_quark_tdvp_config(N=6, T=0.2)
        quench = SchwingerQuench(cfg)
        for pred, expected in (
            (StandardMajorization(),   "standard"),
            (LogConcaveMajorization(), "log-concave"),
            (PeakRadialMajorization(), "peak-radial"),
        ):
            r = quench.compareCausalOrders(vLr=1.0, predicate=pred)
            self.assertEqual(r.majKind, expected,
                             msg=f"predicate {expected} reported as {r.majKind}")


# ─── Majorization static utility class ─────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestMajorizationStaticClass(unittest.TestCase):
    """:class:`Majorization` is a static utility: it has no instances. Its
    two static methods are :meth:`posetOf` (predicate-driven Hasse-cover
    construction) and :meth:`agreement` (pairwise order statistics)."""

    def test_not_instantiable(self) -> None:
        """The C++ side declares ``Majorization()`` deleted; pybind11
        propagates that as a Python-level error on direct construction."""
        with self.assertRaises(Exception):
            Majorization()

    def test_posetOf_predicate_overload_matches_tol_overload(self) -> None:
        """The convenience tol overload is just a shortcut for
        ``posetOf(spectra, StandardMajorization(tol))``."""
        spectra = [[1/3]*3, [0.5, 0.5], [1.0]]
        a = Majorization.posetOf(spectra)                          # tol=1e-12
        b = Majorization.posetOf(spectra, StandardMajorization())  # equivalent
        self.assertEqual(a.getNodeCount, b.getNodeCount)
        self.assertEqual(set(a.covers), set(b.covers))

    def test_posetOf_with_each_predicate_variant(self) -> None:
        """Each predicate subclass produces a well-formed Hasse poset
        (no self-loops, all node indices in range)."""
        spectra = [[1.0], [0.5, 0.5], [1/3]*3]
        for pred in (StandardMajorization(),
                     LogConcaveMajorization(),
                     PeakRadialMajorization()):
            p = Majorization.posetOf(spectra, pred)
            self.assertEqual(p.getNodeCount, 3)
            for a, b in p.covers:
                self.assertNotEqual(a, b)
                self.assertGreaterEqual(a, 0)
                self.assertLess(a, 3)
                self.assertGreaterEqual(b, 0)
                self.assertLess(b, 3)

    def test_agreement_on_identical_poset_is_perfect(self) -> None:
        """A poset compared against itself gives τ = 1 and edit distance 0."""
        p = Majorization.posetOf([[1.0], [0.5, 0.5], [1/3]*3])
        agr = Majorization.agreement(p, p, p.getNodeCount)
        self.assertAlmostEqual(agr.kendallTau, 1.0)
        self.assertAlmostEqual(agr.hasseEditDistance, 0.0)

    def test_agreement_is_swap_symmetric_for_total_orders(self) -> None:
        """Kendall-τ is symmetric (τ(a, b) = τ(b, a)) since concordant /
        discordant counts don't depend on which poset is "first"."""
        p = Majorization.posetOf([[1.0], [0.5, 0.5], [1/3]*3])
        q = Majorization.posetOf([[1.0], [0.5, 0.5]])  # different label set
        # Use the larger label count so both posets fit.
        n = 3
        # Build q as a Poset of size 3 by re-encoding its covers verbatim.
        q3 = Poset()
        q3.getNodeCount = n
        q3.covers = q.covers
        ab = Majorization.agreement(p, q3, n)
        ba = Majorization.agreement(q3, p, n)
        self.assertAlmostEqual(ab.kendallTau, ba.kendallTau)
        self.assertAlmostEqual(ab.discordantFraction, ba.discordantFraction)
        self.assertAlmostEqual(ab.hasseEditDistance, ba.hasseEditDistance)
        # nOnlyA on (a, b) should equal nOnlyB on (b, a).
        self.assertEqual(ab.nOnlyA, ba.nOnlyB)
        self.assertEqual(ab.nOnlyB, ba.nOnlyA)


# ─── Causet static utility class ───────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestCausetStaticClass(unittest.TestCase):
    """:class:`Causet` is a static utility wrapping :meth:`chainFrom`."""

    def test_not_instantiable(self) -> None:
        with self.assertRaises(Exception):
            Causet()

    def test_chainFrom_is_idempotent(self) -> None:
        """Two calls on the same Spacetime must produce equivalent
        CausetChain values (same nSites, antichain layout, hopping pairs,
        cover edges)."""
        import tessera
        metric = tessera.Metric(True, tessera.Signature(4, tessera.Lorentzian))
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(20)
        a = Causet.chainFrom(st)
        b = Causet.chainFrom(st)
        self.assertEqual(a.nSites, b.nSites)
        self.assertEqual(list(a.times), list(b.times))
        self.assertEqual(list(a.vertexIds), list(b.vertexIds))
        self.assertEqual(sorted(a.hoppingPairs), sorted(b.hoppingPairs))
        self.assertEqual(set(a.partialOrder.covers),
                         set(b.partialOrder.covers))


# ─── MajorizationPredicate hierarchy ───────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestMajorizationPredicateHierarchy(unittest.TestCase):
    """The three concrete predicate subclasses share a single base contract.
    They must all be usable polymorphically wherever a
    :class:`MajorizationPredicate` is required."""

    def test_all_subclass_base(self) -> None:
        for pred in (StandardMajorization(),
                     LogConcaveMajorization(),
                     PeakRadialMajorization()):
            self.assertIsInstance(pred, MajorizationPredicate)

    def test_log_concave_is_a_standard_subclass(self) -> None:
        """:class:`LogConcaveMajorization` inherits from
        :class:`StandardMajorization` to share the tol member; the
        Python-level class hierarchy mirrors the C++ one."""
        self.assertTrue(issubclass(LogConcaveMajorization, StandardMajorization))

    def test_peak_radial_is_not_a_standard_subclass(self) -> None:
        """:class:`PeakRadialMajorization` is its own variant — not a
        sub-relation of standard majorization, so it does not inherit
        from StandardMajorization in the C++ hierarchy either."""
        self.assertFalse(issubclass(PeakRadialMajorization, StandardMajorization))

    def test_name_property_matches_string_contract(self) -> None:
        self.assertEqual(StandardMajorization().name,   "standard")
        self.assertEqual(LogConcaveMajorization().name, "log-concave")
        self.assertEqual(PeakRadialMajorization().name, "peak-radial")

    def test_canonical_strict_pair_classical(self) -> None:
        """(1, 0) ≻ (½, ½) under classical majorization."""
        std = StandardMajorization()
        self.assertTrue(std.majorizes([1.0, 0.0], [0.5, 0.5]))
        self.assertTrue(std.strictlyMajorizes([1.0, 0.0], [0.5, 0.5]))

    def test_log_concave_filters_non_log_concave_inputs(self) -> None:
        """A spectrum that fails log-concavity is incomparable in the
        log-concave variant even when classical majorization would
        relate it. ``(0.7, 0.2, 0.1)`` fails log-concavity because
        0.2² = 0.04 < 0.7 × 0.1 = 0.07."""
        lc = LogConcaveMajorization()
        # This spectrum is not log-concave; pairing it against ANY
        # other spectrum makes the comparison incomparable, even when
        # classical would relate them.
        non_lc = [0.7, 0.2, 0.1]
        lc_partner = [0.5, 0.4, 0.1]   # this one IS log-concave
        self.assertFalse(lc.majorizes(non_lc, lc_partner))
        self.assertFalse(lc.majorizes(lc_partner, non_lc))
        # Classical does relate them (top-1: 0.7 > 0.5; etc.).
        self.assertTrue(StandardMajorization().majorizes(non_lc, lc_partner))


# ─── PeakRadialMajorization ────────────────────────────────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestPeakRadialMajorization(unittest.TestCase):
    """Peak-radial dominance is strictly stronger than classical
    majorization: every peak-radial-relation implies the corresponding
    classical relation, but not vice versa."""

    def test_classical_canonical_strict_pair_also_holds_peak_radial(self) -> None:
        """(1, 0) ≻ (½, ½) under both classical AND peak-radial."""
        pr = PeakRadialMajorization()
        self.assertTrue(pr.majorizes([1.0, 0.0], [0.5, 0.5]))
        self.assertTrue(pr.strictlyMajorizes([1.0, 0.0], [0.5, 0.5]))

    def test_witness_where_classical_majorizes_but_peak_radial_does_not(self) -> None:
        """μ = (0.5, 0.5, 0) classically majorizes λ = (0.4, 0.3, 0.3):

        * top-1 sums: 0.5 ≥ 0.4 ✓
        * top-2 sums: 1.0 ≥ 0.7 ✓
        * total: 1.0 = 1.0 ✓

        But peak-radial fails: μ-ratios = (1, 1, 0), λ-ratios = (1, 0.75, 0.75).
        At i=2, μ-ratio = 1 > 0.75 = λ-ratio, so the entrywise
        condition λᵢ/λ₁ ≤ μᵢ/μ₁ is violated (it's the wrong direction
        for "μ more peaked than λ"). This is the canonical witness
        that peak-radial ⊂ classical strictly."""
        std = StandardMajorization()
        pr = PeakRadialMajorization()
        mu  = [0.5, 0.5, 0.0]
        lam = [0.4, 0.3, 0.3]
        self.assertTrue(std.majorizes(mu, lam))
        self.assertFalse(pr.majorizes(mu, lam))

    def test_tol_property(self) -> None:
        pr = PeakRadialMajorization(1e-9)
        self.assertAlmostEqual(pr.tol, 1e-9)


# ─── LogConcaveMajorization.isLogConcave (static method) ───────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestIsLogConcaveStatic(unittest.TestCase):
    """:meth:`LogConcaveMajorization.isLogConcave` is a static helper. The
    contract: True iff, after sorting descending and stripping trailing
    near-zero entries, s_i² ≥ s_{i-1} · s_{i+1} for every interior i."""

    def test_trivial_short_spectra(self) -> None:
        """Length ≤ 2 spectra are trivially log-concave."""
        self.assertTrue(LogConcaveMajorization.isLogConcave([]))
        self.assertTrue(LogConcaveMajorization.isLogConcave([1.0]))
        self.assertTrue(LogConcaveMajorization.isLogConcave([0.5, 0.5]))
        self.assertTrue(LogConcaveMajorization.isLogConcave([0.6, 0.4]))

    def test_uniform_distribution_is_log_concave(self) -> None:
        """(1/3, 1/3, 1/3) hits the equality case: s² = s·s holds
        identically. Treated as log-concave."""
        self.assertTrue(LogConcaveMajorization.isLogConcave([1/3, 1/3, 1/3]))

    def test_strictly_log_concave_example(self) -> None:
        """(0.5, 0.4, 0.1): 0.4² = 0.16 ≥ 0.5·0.1 = 0.05 ✓."""
        self.assertTrue(LogConcaveMajorization.isLogConcave([0.5, 0.4, 0.1]))

    def test_non_log_concave_example(self) -> None:
        """(0.7, 0.2, 0.1): 0.2² = 0.04 < 0.7·0.1 = 0.07."""
        self.assertFalse(LogConcaveMajorization.isLogConcave([0.7, 0.2, 0.1]))

    def test_trailing_zeros_stripped_before_check(self) -> None:
        """Trailing zeros are removed before the inequality is applied
        (otherwise s_{i+1} = 0 would make any spectrum log-concave at
        the tail, and a non-zero middle entry would force the body of
        the spectrum to be log-concave anyway). With trailing zeros
        stripped, (0.5, 0.4, 0, 0) reduces to (0.5, 0.4) which is
        trivially log-concave."""
        self.assertTrue(LogConcaveMajorization.isLogConcave([0.5, 0.4, 0.0, 0.0]))

    def test_permutation_invariance(self) -> None:
        """Input order doesn't matter — the implementation sorts first."""
        sorted_input    = [0.5, 0.4, 0.1]
        permuted_input  = [0.1, 0.5, 0.4]
        self.assertEqual(
            LogConcaveMajorization.isLogConcave(sorted_input),
            LogConcaveMajorization.isLogConcave(permuted_input),
        )

    def test_tol_makes_borderline_cases_pass(self) -> None:
        """A spectrum sitting on the equality boundary up to numerical
        noise should be classified as log-concave when ``tol`` is loose
        enough to absorb the wobble."""
        # Construct s = (1, 0.5, 0.25): s_1² = 0.25, s_0·s_2 = 0.25 — exactly
        # on the boundary.
        on_boundary = [1.0, 0.5, 0.25]
        self.assertTrue(LogConcaveMajorization.isLogConcave(on_boundary))
        # Now nudge s_1 slightly below sqrt(s_0 · s_2). With tol=1e-12 this
        # should register as non-log-concave; with tol=1e-3 it gets absorbed.
        nudged = [1.0, 0.5 - 1e-6, 0.25]
        self.assertFalse(LogConcaveMajorization.isLogConcave(nudged, tol=1e-12))
        self.assertTrue(LogConcaveMajorization.isLogConcave(nudged, tol=1e-3))


# ─── CausalOrders.fromSnapshots (standalone factory) ──────────────────────

@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestCausalOrdersFactory(unittest.TestCase):
    """:meth:`CausalOrders.fromSnapshots` is the standalone factory that
    :meth:`SchwingerQuench.compareCausalOrders` calls internally. Callers
    holding an existing snapshot list can use it directly without re-
    running the TDVP pipeline."""

    @classmethod
    def setUpClass(cls) -> None:  # noqa: D401
        cfg = _light_quark_tdvp_config(N=6, T=0.2)
        cfg.recordSpectra = True
        cls.snapshots = SchwingerQuench(cfg).evolve().snapshots

    def test_factory_produces_three_posets_over_shared_label_set(self) -> None:
        orders = CausalOrders.fromSnapshots(self.snapshots, vLr=1.0)
        n_labels = len(orders.labels)
        self.assertGreater(n_labels, 0)
        self.assertEqual(orders.maj.getNodeCount, n_labels)
        self.assertEqual(orders.lr.getNodeCount,  n_labels)
        self.assertEqual(orders.cs.getNodeCount,  n_labels)

    def test_default_predicate_matches_explicit_standard(self) -> None:
        """``predicate=None`` should produce identical orders to passing
        ``StandardMajorization()`` explicitly."""
        a = CausalOrders.fromSnapshots(self.snapshots, vLr=1.0)
        b = CausalOrders.fromSnapshots(
            self.snapshots, vLr=1.0, predicate=StandardMajorization())
        self.assertEqual(set(a.maj.covers), set(b.maj.covers))
        self.assertEqual(set(a.lr.covers),  set(b.lr.covers))
        self.assertEqual(set(a.cs.covers),  set(b.cs.covers))

    def test_lr_subset_of_cs_invariant_from_factory(self) -> None:
        """The structural invariant ≼_LR ⊂ ≼_cs (the strongest sanity
        check on the causal-comparison construction) must also hold when the
        orders are built directly via the factory."""
        orders = CausalOrders.fromSnapshots(self.snapshots, vLr=1.0)
        agr = Majorization.agreement(orders.lr, orders.cs,
                                       len(orders.labels))
        self.assertAlmostEqual(agr.kendallTau, 1.0, places=12)
        self.assertEqual(agr.nDiscordant, 0)

    def test_factory_rejects_snapshots_without_recordSpectra(self) -> None:
        """Without recorded spectra there's nothing to build ≼_maj from."""
        cfg = _light_quark_tdvp_config(N=4, T=0.2)
        cfg.recordSpectra = False
        snapshots = SchwingerQuench(cfg).evolve().snapshots
        with self.assertRaises(Exception):
            CausalOrders.fromSnapshots(snapshots, vLr=1.0)


if __name__ == "__main__":
    unittest.main()
