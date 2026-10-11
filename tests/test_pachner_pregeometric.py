# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Pre-geometric, boundary-fixed Pachner moves (#112).

The CDT Pachner hierarchy (Add / Remove / Flip / IFlip / Shift) is
generalized so the 3D bistellar moves run on a *pre-geometric*
(non-time-sliced, coordinate-free) simplicial complex via
``mode=PachnerMode.PreGeometric``:

  * Flip   — the 2 -> (d+1) bistellar flip (2->3 in 3D),
  * IFlip  — the (d+1) -> 2 inverse flip (3->2 in 3D),
  * Add    — the 1 -> (d+1) stellar subdivision (1->4 in 3D),
  * Remove — the (d+1) -> 1 stellar weld (4->1 in 3D).

The CDT-orientation / time-slice guard is dropped (a manifold-
preservation check stands in for it) and the move dimension is read off
the actual top cell rather than the metric signature.  A ``boundary_fixed``
flag additionally restricts a move to the interior so the boundary
face-set ``∂W`` (codim-1 faces in exactly one top cell) stays fixed.

What is verified here:

  * **Pre-geometric validity** — on a closed ``T^3`` a random sequence of
    interior moves keeps it a valid closed pseudomanifold and preserves
    homology (Betti numbers ``[1, 3, 3, 1]``).
  * **Boundary-fixed** — on a bounded complex (a solid tetrahedron, and a
    triangular bipyramid) interior moves leave ``∂W`` exactly unchanged
    while changing the interior.

  * **Dijkgraaf-Witten Z-invariance** (the headline T2 check) — the ℤ₂
    state sum ``Z(W)`` is invariant to machine precision across an interior
    Pachner sweep on the small closed 3-manifold ``S^2 x S^1``, for both the
    Trivial and Sign cocycles.

The CDT (foliated) path is unaffected — see the ``test_pachner_*`` /
``test_*cdt*`` suites for the non-regression coverage; a couple of
defaults are re-checked here.
"""

import itertools
import unittest

import tessera

cobordism = tessera.cobordism
PRE = tessera.PachnerMode.PreGeometric


# =====================================================================
# Helpers
# =====================================================================

def _spacetime(dim, topology=None):
    """A ``dim``-dimensional spacetime.  The signature dimension is what
    makes the (dim)-cells register as *top* simplices, so it must match
    the manifold dimension for ``get_random_top_simplex`` to see them."""
    sig = tessera.Signature(dim, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    topo = topology if topology is not None else tessera.SolidSimplex(dim)
    return tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                             tessera.PREFERRED, topo)


def _built(topology, dim=3):
    st = _spacetime(dim, topology)
    st.build()
    return st


def _circle():
    return tessera.SimplexBoundarySphere(1)  # S^1


def _t3():
    """T^3 = S^1 x S^1 x S^1 — a closed, coordinate-free 3-manifold."""
    return tessera.SimplicialProduct(
        tessera.SimplicialProduct(_circle(), _circle()), _circle())


def _tops(st):
    """Top-dimensional cells as sorted vertex-id tuples (plain ints)."""
    sizes = [len(s.get_vertices()) for s in st.get_simplices()]
    if not sizes:
        return []
    top = max(sizes)
    return [tuple(sorted(v.get_id() for v in s.get_vertices()))
            for s in st.get_simplices() if len(s.get_vertices()) == top]


def _facet_counts(tops):
    counts = {}
    for t in tops:
        for f in itertools.combinations(t, len(t) - 1):
            counts[f] = counts.get(f, 0) + 1
    return counts


def _boundary(tops):
    """``∂W``: the codim-1 faces contained in exactly one top cell."""
    return frozenset(f for f, c in _facet_counts(tops).items() if c == 1)


def _is_closed_pseudomanifold(tops):
    """Every codim-1 face is shared by exactly two top cells."""
    counts = _facet_counts(tops)
    return bool(counts) and all(c == 2 for c in counts.values())


def _betti(st):
    return cobordism.ChainComplex.from_spacetime(st).betti_numbers()


def _make(cls, st, seed, boundary_fixed=False):
    """A pre-geometric move of type ``cls`` (AddMove has the extra
    ``relabel`` positional argument)."""
    if cls is tessera.AddMove:
        return cls(st, seed, False, PRE, boundary_fixed)
    return cls(st, seed, PRE, boundary_fixed)


_BISTELLAR = (tessera.FlipMove, tessera.IFlipMove, tessera.AddMove)


def _run_sequence(st, n_moves, classes=_BISTELLAR, boundary_fixed=False,
                  max_attempts=4000, after=None):
    """Apply up to ``n_moves`` successful pre-geometric moves, cycling
    through ``classes``.  Calls ``after(st)`` after each success (e.g. to
    assert an invariant).  Returns the per-type success counts."""
    counts = {c.__name__: 0 for c in classes}
    seed = 0
    while sum(counts.values()) < n_moves and seed < max_attempts:
        progressed = False
        for cls in classes:
            m = _make(cls, st, seed, boundary_fixed)
            if m.propose() and m.apply():
                counts[cls.__name__] += 1
                progressed = True
                if after is not None:
                    after(st)
                break
        seed += 1
        if not progressed:
            continue
    return counts


def _d2_zero(st):
    """∂_{k-1} ∘ ∂_k = 0 for the chain complex of the current triangulation —
    the chain-complex axiom, and a non-corruption check on the moved complex."""
    return cobordism.ChainComplex.from_spacetime(st).boundary_composes_to_zero()


def _is_pseudomanifold_with_boundary(tops):
    """Every codim-1 face is shared by one or two top cells (1 ⇒ on ∂W)."""
    counts = _facet_counts(tops)
    return bool(counts) and all(c in (1, 2) for c in counts.values())


def _s2s1():
    """S^2 x S^1 — the vertex-minimal closed oriented 3-manifold fixture."""
    return tessera.SphereCircleProduct()


def _cylinder(n_simplices=24):
    """A 3D CDT cylinder Σ x [0, T]: closed spatial slices but open time
    boundaries, so ∂W (the two end slices) is non-empty."""
    st = _spacetime(3, tessera.Cylinder())
    st.build(n_simplices)
    return st


# =====================================================================
# Pre-geometric validity on the closed 3-torus
# =====================================================================

class TestPreGeometricClosedT3(unittest.TestCase):
    """A random sequence of interior bistellar moves on a closed T^3
    keeps it a valid closed pseudomanifold and preserves homology."""

    def test_fixture_starts_as_t3(self):
        st = _built(_t3())
        self.assertEqual(_betti(st), [1, 3, 3, 1])
        self.assertTrue(_is_closed_pseudomanifold(_tops(st)))
        # 3-manifold: top cells are tetrahedra.
        self.assertTrue(all(len(t) == 4 for t in _tops(st)))

    def test_random_interior_moves_preserve_homology(self):
        st = _built(_t3())
        st.set_seed(20240601)
        before_tops = set(_tops(st))

        counts = _run_sequence(st, n_moves=40)
        self.assertGreaterEqual(sum(counts.values()), 40,
                                f"too few moves fired: {counts}")
        # The bistellar moves actually fire (not just the always-on add).
        self.assertGreater(counts["FlipMove"] + counts["IFlipMove"], 0,
                           f"no 2<->3 flips fired: {counts}")

        # Still a closed 3-manifold with the homology of T^3.
        self.assertTrue(_is_closed_pseudomanifold(_tops(st)))
        self.assertEqual(_betti(st), [1, 3, 3, 1])
        # ... and genuinely retriangulated, not a no-op.
        self.assertNotEqual(set(_tops(st)), before_tops)

    def test_homology_preserved_after_each_move(self):
        st = _built(_t3())
        st.set_seed(7)

        def check(s):
            self.assertEqual(_betti(s), [1, 3, 3, 1])
            self.assertTrue(_is_closed_pseudomanifold(_tops(s)))

        counts = _run_sequence(st, n_moves=15, after=check)
        self.assertGreaterEqual(sum(counts.values()), 15)


# =====================================================================
# Boundary-fixed: triangular bipyramid (a flippable interior facet)
# =====================================================================

class TestBoundaryFixedBipyramid(unittest.TestCase):
    """Two tetrahedra glued on triangle 012 (apexes 3, 4).  The shared
    triangle is the only interior facet; the six outer triangles are
    ``∂W``.  A boundary-fixed 2->3 flip retriangulates the interior while
    leaving ``∂W`` exactly fixed."""

    @staticmethod
    def _bipyramid():
        st = _spacetime(3)
        v = [st.create_vertex(i) for i in range(5)]
        st.create_simplex([v[0], v[1], v[2], v[3]])  # tet 0123
        st.create_simplex([v[0], v[1], v[2], v[4]])  # tet 0124
        return st

    def test_setup_has_one_interior_facet(self):
        st = self._bipyramid()
        tops = _tops(st)
        self.assertEqual(len(tops), 2)
        # 6 outer triangles on the boundary, 012 interior.
        self.assertEqual(len(_boundary(tops)), 6)
        self.assertNotIn((0, 1, 2), _boundary(tops))

    def test_boundary_fixed_flip_keeps_boundary_changes_interior(self):
        st = self._bipyramid()
        boundary_before = _boundary(_tops(st))

        flipped = None
        for seed in range(200):
            m = _make(tessera.FlipMove, st, seed, boundary_fixed=True)
            if m.propose():
                self.assertEqual(m.mode(), PRE)
                self.assertTrue(m.boundary_fixed())
                self.assertTrue(m.apply())
                flipped = m
                break
        self.assertIsNotNone(flipped, "boundary-fixed flip never fired")

        tops_after = _tops(st)
        # ∂W is byte-identical; the interior changed (2 -> 3 tetrahedra).
        self.assertEqual(_boundary(tops_after), boundary_before)
        self.assertEqual(len(tops_after), 3)
        # The apex edge 3-4 now exists (it did not before the flip).
        self.assertTrue(any(3 in t and 4 in t for t in tops_after))

    def test_flip_then_iflip_round_trips_with_fixed_boundary(self):
        st = self._bipyramid()
        boundary = _boundary(_tops(st))

        for seed in range(200):
            m = _make(tessera.FlipMove, st, seed, boundary_fixed=True)
            if m.propose() and m.apply():
                break
        self.assertEqual(len(_tops(st)), 3)
        self.assertEqual(_boundary(_tops(st)), boundary)

        for seed in range(200):
            m = _make(tessera.IFlipMove, st, seed, boundary_fixed=True)
            if m.propose() and m.apply():
                break
        # Back to two tetrahedra, boundary still fixed throughout.
        self.assertEqual(len(_tops(st)), 2)
        self.assertEqual(_boundary(_tops(st)), boundary)


# =====================================================================
# Boundary-fixed: solid tetrahedron (∂ = S^2) under interior moves
# =====================================================================

class TestBoundaryFixedSolidSimplex(unittest.TestCase):
    """``SolidSimplex(3)`` is a single tetrahedron whose boundary is the
    4-triangle 2-sphere.  Interior moves grow the interior while ``∂W``
    stays exactly the original four faces."""

    S2 = frozenset([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])

    def test_solid_simplex_boundary_is_s2(self):
        st = _built(tessera.SolidSimplex(3))
        self.assertEqual(len(_tops(st)), 1)
        self.assertEqual(_boundary(_tops(st)), self.S2)

    def test_interior_moves_fix_boundary_and_grow_interior(self):
        st = _built(tessera.SolidSimplex(3))
        st.set_seed(101)
        n0_before = st.get_vertex_count()
        ntops_before = len(_tops(st))

        # After every successful boundary-fixed move ∂W is unchanged.
        self.assertEqual(_boundary(_tops(st)), self.S2)

        def check(s):
            self.assertEqual(_boundary(_tops(s)), self.S2)

        counts = _run_sequence(st, n_moves=12, boundary_fixed=True, after=check)
        self.assertGreaterEqual(sum(counts.values()), 12)
        self.assertGreater(counts["AddMove"], 0)

        # ∂W fixed, interior genuinely larger.
        self.assertEqual(_boundary(_tops(st)), self.S2)
        self.assertGreater(st.get_vertex_count(), n0_before)
        self.assertGreater(len(_tops(st)), ntops_before)


# =====================================================================
# Pre-geometric add / remove are a 1<->(d+1) inverse pair
# =====================================================================

class TestPreGeometricAddRemoveRoundTrip(unittest.TestCase):
    """The 1->(d+1) add and the (d+1)->1 remove invert each other, and
    the add's rollback restores the complex byte-for-byte."""

    S2 = frozenset([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])

    def _apply_add(self, st):
        for seed in range(50):
            m = _make(tessera.AddMove, st, seed)
            if m.propose() and m.apply():
                return m
        self.fail("pre-geometric add never fired")

    def test_add_creates_d_plus_1_cells_one_vertex(self):
        st = _built(tessera.SolidSimplex(3))
        n0, ntop = st.get_vertex_count(), len(_tops(st))
        self._apply_add(st)
        self.assertEqual(st.get_vertex_count(), n0 + 1)
        self.assertEqual(len(_tops(st)), ntop + 3)  # 1 -> d+1 = 4 in 3D
        # A 1->4 subdivision is interior: ∂W (the S^2) is untouched.
        self.assertEqual(_boundary(_tops(st)), self.S2)

    def test_add_rollback_restores_state(self):
        st = _built(tessera.SolidSimplex(3))
        before = sorted(_tops(st))
        n0 = st.get_vertex_count()
        m = self._apply_add(st)
        self.assertTrue(m.is_applied())
        m.rollback()
        self.assertFalse(m.is_applied())
        self.assertEqual(sorted(_tops(st)), before)
        self.assertEqual(st.get_vertex_count(), n0)

    def test_add_then_remove_returns_to_start(self):
        st = _built(tessera.SolidSimplex(3))
        before = sorted(_tops(st))
        n0 = st.get_vertex_count()

        self._apply_add(st)
        self.assertEqual(len(_tops(st)), len(before) + 3)

        # The (d+1)->1 weld targets the freshly inserted interior vertex
        # (the only vertex whose link is the boundary of a tetrahedron).
        for seed in range(200):
            m = _make(tessera.RemoveMove, st, seed)
            if m.propose() and m.apply():
                break
        else:
            self.fail("pre-geometric remove never fired")

        self.assertEqual(sorted(_tops(st)), before)
        self.assertEqual(st.get_vertex_count(), n0)


# =====================================================================
# CDT default path is undisturbed
# =====================================================================

class TestCDTDefaultsUnchanged(unittest.TestCase):
    """Moves default to the CDT regime; the new knobs are opt-in."""

    @staticmethod
    def _cdt():
        sig = tessera.Signature(4, tessera.Lorentzian)
        metric = tessera.Metric(True, sig)
        st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                               tessera.PREFERRED, tessera.Toroid())
        st.build(60)
        return st

    def test_default_mode_is_cdt(self):
        st = self._cdt()
        for cls in (tessera.FlipMove, tessera.IFlipMove, tessera.ShiftMove,
                    tessera.RemoveMove):
            m = cls(st, 0)
            self.assertEqual(m.mode(), tessera.PachnerMode.CDT)
            self.assertFalse(m.boundary_fixed())
        add = tessera.AddMove(st, 0)
        self.assertEqual(add.mode(), tessera.PachnerMode.CDT)
        self.assertFalse(add.boundary_fixed())

    def test_spacetimetype_cdt_not_shadowed(self):
        # Regression guard: tessera.CDT must remain SpacetimeType.CDT,
        # not PachnerMode.CDT (the enums share the member name "CDT").
        self.assertIs(tessera.CDT, tessera.SpacetimeType.CDT)

    def test_cdt_flip_still_proposes(self):
        st = self._cdt()
        ok = any(tessera.FlipMove(st, s).propose() for s in range(200))
        self.assertTrue(ok)

    def test_cdt_move_still_applies_on_foliated_complex(self):
        # The foliated (time-sliced) path is intact: a default CDT move proposes
        # AND applies on a built CDT spacetime, in CDT mode with no boundary
        # fixing — unaffected by the pre-geometric / boundary-fixed extensions.
        st = self._cdt()
        applied = None
        for seed in range(400):
            for cls in (tessera.FlipMove, tessera.ShiftMove, tessera.AddMove):
                m = cls(st, seed)
                if m.mode() != tessera.PachnerMode.CDT or m.boundary_fixed():
                    continue
                if m.propose() and m.apply():
                    applied = m
                    break
            if applied is not None:
                break
        self.assertIsNotNone(applied, "no CDT move applied on the foliated complex")
        self.assertEqual(applied.mode(), tessera.PachnerMode.CDT)
        self.assertFalse(applied.boundary_fixed())


# =====================================================================
# Dijkgraaf-Witten Z-invariance — the headline T2 check (#108)
# =====================================================================

class TestDijkgraafWittenZInvariance(unittest.TestCase):
    """The make-or-break T2 check: the Dijkgraaf-Witten ℤ₂ state sum
    ``Z(W)`` is invariant to machine precision across a sequence of
    interior Pachner moves.

    Run on the closed oriented 3-manifold ``S^2 x S^1`` (the vertex-minimal
    ``SphereCircleProduct``): the state sum brute-forces the whole flat
    space ``2^{|V|-1+b_1}``, so it needs a small triangulation — flips and
    iflips leave ``|V|`` fixed, and the few subdivisions are capped so the
    flat space stays enumerable.  Both normalized cocycles are checked: the
    untwisted ``Trivial`` and the ``Sign`` twist (which agrees on
    ``S^2 x S^1`` because its mod-2 cup cube vanishes)."""

    def _Z(self, st):
        zt = cobordism.DijkgraafWitten(
            st, cobordism.Cocycle.Trivial).partitionFunction()
        zs = cobordism.DijkgraafWitten(
            st, cobordism.Cocycle.Sign).partitionFunction()
        return zt, zs

    @unittest.skipUnless(hasattr(cobordism, "DijkgraafWitten"),
                         "cobordism.DijkgraafWitten unavailable (#108)")
    def test_z_invariant_under_interior_pachner_sweep(self):
        st = _built(tessera.SphereCircleProduct())
        st.set_seed(11)

        # Convention anchor: Z_Trivial(S^2 x S^1) = 2^{b_1 - 1} = 1, and the
        # Sign cocycle agrees here (the cup cube vanishes on S^2 x S^1).
        zt0, zs0 = self._Z(st)
        self.assertAlmostEqual(zt0.real, 1.0, places=9)
        self.assertAlmostEqual(zt0.imag, 0.0, places=9)
        self.assertAlmostEqual(abs(zs0 - zt0), 0.0, places=9)

        depth, seed = 0, 0
        counts = {"FlipMove": 0, "IFlipMove": 0, "AddMove": 0}
        classes = (tessera.FlipMove, tessera.IFlipMove, tessera.AddMove)
        while depth < 18 and seed < 6000:
            for cls in classes:
                # Cap |V| so the flat space (2^{|V|-1+b_1}) stays enumerable.
                if cls is tessera.AddMove and st.get_vertex_count() >= 16:
                    continue
                m = _make(cls, st, seed)
                if m.propose() and m.apply():
                    depth += 1
                    counts[cls.__name__] += 1
                    # Still a closed pseudomanifold ...
                    self.assertTrue(_is_closed_pseudomanifold(_tops(st)),
                                    f"not closed at depth {depth}")
                    # ... and Z(W) is invariant to machine precision, for
                    # both cocycles, at every depth of the sweep.
                    zt, zs = self._Z(st)
                    self.assertAlmostEqual(abs(zt - zt0), 0.0, places=9,
                                           msg=f"Z_Trivial drifted at depth {depth}")
                    self.assertAlmostEqual(abs(zs - zs0), 0.0, places=9,
                                           msg=f"Z_Sign drifted at depth {depth}")
                    break
            seed += 1

        self.assertGreaterEqual(depth, 15, f"too few moves fired: {counts}")
        self.assertGreater(counts["FlipMove"] + counts["IFlipMove"], 0,
                           f"no 2<->3 flips fired: {counts}")
        # A 1->4 subdivision changes |V|, hence the 1/2^|V| prefactor and the
        # flat space itself — Z staying put through it is the strong check.
        self.assertGreater(counts["AddMove"], 0, f"no subdivisions fired: {counts}")


# =====================================================================
# Boundary algebra: ∂² = 0 across an interior sweep
# =====================================================================

class TestPreGeometricBoundaryAlgebra(unittest.TestCase):
    """A random interior Pachner sweep keeps the simplicial *chain complex*
    coherent: ∂_{k-1} ∘ ∂_k = 0 at every step (no malformed or duplicated
    cells), complementing the facet-sharing pseudomanifold check."""

    def test_d_squared_zero_after_each_interior_move_on_t3(self):
        st = _built(_t3())
        st.set_seed(31337)
        self.assertTrue(_d2_zero(st))

        def check(s):
            self.assertTrue(_d2_zero(s))
            self.assertTrue(_is_closed_pseudomanifold(_tops(s)))

        counts = _run_sequence(st, n_moves=20, after=check)
        self.assertGreaterEqual(sum(counts.values()), 20,
                                f"too few moves fired: {counts}")
        self.assertTrue(_d2_zero(st))


# =====================================================================
# Homology preserved on S^2 x S^1
# =====================================================================

class TestPreGeometricHomologyS2xS1(unittest.TestCase):
    """The other closed 3-manifold from the headline checks: S^2 x S^1, with
    Betti numbers [1, 1, 1, 1]. A random interior sweep preserves them."""

    BETTI = [1, 1, 1, 1]

    def test_fixture_starts_as_s2xs1(self):
        st = _built(_s2s1())
        self.assertEqual(_betti(st), self.BETTI)
        self.assertTrue(_is_closed_pseudomanifold(_tops(st)))
        self.assertTrue(_d2_zero(st))
        self.assertTrue(all(len(t) == 4 for t in _tops(st)))

    def test_random_interior_moves_preserve_homology(self):
        st = _built(_s2s1())
        st.set_seed(909)
        before = set(_tops(st))

        def check(s):
            self.assertEqual(_betti(s), self.BETTI)
            self.assertTrue(_is_closed_pseudomanifold(_tops(s)))

        counts = _run_sequence(st, n_moves=18, after=check)
        self.assertGreaterEqual(sum(counts.values()), 18,
                                f"too few moves fired: {counts}")
        self.assertGreater(counts["FlipMove"] + counts["IFlipMove"], 0,
                           f"no 2<->3 flips fired: {counts}")
        self.assertEqual(_betti(st), self.BETTI)
        self.assertTrue(_is_closed_pseudomanifold(_tops(st)))
        self.assertNotEqual(set(_tops(st)), before)


# =====================================================================
# Boundary-fixed: CDT cylinder Σ x [0, T]  (∂W = the open-time end slices)
# =====================================================================

class TestBoundaryFixedCylinder(unittest.TestCase):
    """A 3D CDT cylinder is a bounded complex whose ∂W is the two open-time end
    slices. Boundary-fixed interior moves (the always-interior 1->4 subdivision
    in particular) grow the interior while leaving ∂W byte-identical."""

    def test_cylinder_is_a_bounded_pseudomanifold(self):
        st = _cylinder()
        tops = _tops(st)
        self.assertTrue(all(len(t) == 4 for t in tops))
        self.assertTrue(_is_pseudomanifold_with_boundary(tops))
        self.assertGreater(len(_boundary(tops)), 0)

    def test_interior_moves_fix_boundary_and_grow_interior(self):
        st = _cylinder()
        st.set_seed(2024)
        boundary_before = _boundary(_tops(st))
        n0 = st.get_vertex_count()
        ntops0 = len(_tops(st))
        self.assertGreater(len(boundary_before), 0)

        def check(s):
            self.assertEqual(_boundary(_tops(s)), boundary_before)

        counts = _run_sequence(st, n_moves=10, boundary_fixed=True, after=check)
        self.assertGreaterEqual(sum(counts.values()), 10,
                                f"too few moves fired: {counts}")
        self.assertGreater(counts["AddMove"], 0, f"no subdivisions fired: {counts}")
        # ∂W byte-identical; interior genuinely larger.
        self.assertEqual(_boundary(_tops(st)), boundary_before)
        self.assertGreater(st.get_vertex_count(), n0)
        self.assertGreater(len(_tops(st)), ntops0)


if __name__ == "__main__":
    unittest.main()
