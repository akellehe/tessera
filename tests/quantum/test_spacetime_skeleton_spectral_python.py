"""Acceptance tests for Spacetime::getSpectralDimensionOnSkeleton and the
SimplexFilter interface.

These are the architectural pieces that survived the #31 close-without-merge:
- SimplexFilter / AllSimplexFilter / PositiveGramDeterminantFilter
- Spacetime.get_spectral_dimension_on_skeleton (now the unified entry point for
  D_S on the weighted 1-skeleton, used by InteractionSimulation.get_spectral_dimension
  via delegation)

The MI/TDVP "holographic dual" bulk reconstruction was rolled back — the
right home for Spacetime-from-MI is per-Vertex state/basis/labels on
Spacetime, scoped under v0.2.2 (#41).
"""

from __future__ import annotations

import math
import unittest

try:
    import tessera
    from tessera import (
        AllSimplexFilter,
        PositiveGramDeterminantFilter,
        SimplexFilter,
        Spacetime,
    )
    HAVE_TESSERA = True
except ImportError:
    HAVE_TESSERA = False


@unittest.skipUnless(HAVE_TESSERA, "tessera unavailable")
class TestSimplexFilters(unittest.TestCase):
    """The two concrete SimplexFilter subclasses."""

    def test_all_filter_name(self) -> None:
        f = AllSimplexFilter()
        self.assertEqual(f.name(), "AllSimplexFilter")
        self.assertIn("AllSimplexFilter", repr(f))
        self.assertIsInstance(f, SimplexFilter)

    def test_positive_gram_filter_name(self) -> None:
        f = PositiveGramDeterminantFilter()
        self.assertEqual(f.name(), "PositiveGramDeterminantFilter")
        self.assertIsInstance(f, SimplexFilter)


@unittest.skipUnless(HAVE_TESSERA, "tessera unavailable")
class TestSpacetimeGetSpectralDimensionOnSkeleton(unittest.TestCase):
    """The unified D_S entry point on Spacetime."""

    def test_skeleton_dim_other_than_one_throws(self) -> None:
        st = Spacetime()
        st.build(num_simplices=10)
        with self.assertRaises(Exception):
            st.get_spectral_dimension_on_skeleton(
                [0.5, 1.0, 2.0], 12, AllSimplexFilter(),
                top_k=4, skeleton_dim=2)

    def test_topk_below_one_throws(self) -> None:
        st = Spacetime()
        st.build(num_simplices=10)
        with self.assertRaises(Exception):
            st.get_spectral_dimension_on_skeleton(
                [0.5, 1.0, 2.0], 12, AllSimplexFilter(),
                top_k=0, skeleton_dim=1)

    def test_topk_too_large_yields_zeros(self) -> None:
        """No top simplices of size topK+1=10 exist in a default 4D build,
        so the heat-kernel return is empty and D_S is uniformly zero."""
        st = Spacetime()
        st.build(num_simplices=10)
        ds = st.get_spectral_dimension_on_skeleton(
            [0.5, 1.0, 2.0], 12, AllSimplexFilter(),
            top_k=9, skeleton_dim=1)
        self.assertEqual(len(ds), 3)
        self.assertTrue(all(d == 0.0 for d in ds))

    def test_empty_spacetime_returns_zeros(self) -> None:
        st = Spacetime()  # never built — get_simplices() is empty
        ds = st.get_spectral_dimension_on_skeleton(
            [0.5, 1.0, 2.0], 12, AllSimplexFilter(),
            top_k=4, skeleton_dim=1)
        self.assertEqual(len(ds), 3)
        self.assertTrue(all(d == 0.0 for d in ds))

    def test_default_4d_build_produces_finite_ds(self) -> None:
        """A 4D CDT build should yield a non-empty 1-skeleton of 4-simplices
        and a defined D_S(σ) curve."""
        st = Spacetime()
        st.build(num_simplices=20)
        sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
        ds = st.get_spectral_dimension_on_skeleton(
            sigmas, 12, AllSimplexFilter(), top_k=4, skeleton_dim=1)
        self.assertEqual(len(ds), len(sigmas))
        finite_count = sum(1 for d in ds if math.isfinite(d))
        self.assertGreater(finite_count, 0,
            msg=f"expected at least one finite D_S; got {ds}")

    def test_positive_gram_filter_runs(self) -> None:
        """PositiveGramDeterminantFilter is a valid drop-in alternative."""
        st = Spacetime()
        st.build(num_simplices=20)
        sigmas = [0.5, 1.0, 2.0]
        ds_all = st.get_spectral_dimension_on_skeleton(
            sigmas, 12, AllSimplexFilter(), top_k=4, skeleton_dim=1)
        ds_pos = st.get_spectral_dimension_on_skeleton(
            sigmas, 12, PositiveGramDeterminantFilter(), top_k=4,
            skeleton_dim=1)
        self.assertEqual(len(ds_pos), len(sigmas))
        # ds_pos may match or differ from ds_all depending on which
        # simplices the default 4D build admits as metrically valid.
        # We just check both produced a well-formed array.
        for d in ds_all:
            self.assertTrue(math.isfinite(d) or d == 0.0 or math.isnan(d))
        for d in ds_pos:
            self.assertTrue(math.isfinite(d) or d == 0.0 or math.isnan(d))


if __name__ == "__main__":
    unittest.main()
