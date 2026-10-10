"""Python tests — :meth:`Causet.chainFrom` (Spacetime → causet-
chain extractor) plus the underlying :meth:`Poset.fromSpacetime`
factory exposed in the same module.

Skips cleanly when tessera was built without TESSERA_QUANTUM=1.
"""

from __future__ import annotations

import unittest

import tessera

try:
    from tessera import Poset
    from tessera.quantum import CausetChain, Causet
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def _build_default_cdt(num_simplices: int = 20) -> "tessera.Spacetime":
    """Tiny default-CDT Spacetime for testing — toroidal, alpha=1, a=1."""
    metric = tessera.Metric(True, tessera.Signature(4, tessera.Lorentzian))
    st = tessera.Spacetime(
        metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED, tessera.Toroid()
    )
    st.build(num_simplices)
    return st


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestCausetChainFrom(unittest.TestCase):
    """The extractor produces a self-consistent CausetChain on a built CDT."""

    def test_default_cdt_self_consistent(self) -> None:
        st = _build_default_cdt(num_simplices=20)
        chain = Causet.chainFrom(st)

        # nSites == sum of antichain sizes == len(vertexIds).
        self.assertEqual(
            sum(len(a) for a in chain.antichains), chain.nSites
        )
        self.assertEqual(len(chain.vertexIds), chain.nSites)

        # times sorted ascending, no duplicates.
        self.assertEqual(list(chain.times), sorted(set(chain.times)))

        # Flat layout: vertexIds is the concatenation of antichains
        # in time order.
        flat = [vid for ac in chain.antichains for vid in ac]
        self.assertEqual(list(chain.vertexIds), flat)

        # hoppingPairs invariants.
        layer_of_site: dict[int, int] = {}
        flat_idx = 0
        for layer_idx, ac in enumerate(chain.antichains):
            for _ in ac:
                layer_of_site[flat_idx] = layer_idx
                flat_idx += 1
        for i, j in chain.hoppingPairs:
            self.assertLess(i, j, "hopping pair must be canonicalised i<j")
            self.assertIn(i, layer_of_site)
            self.assertIn(j, layer_of_site)
            self.assertEqual(
                abs(layer_of_site[j] - layer_of_site[i]), 1,
                "hopping pair endpoints must be in adjacent layers"
            )

        # partialOrder Poset has nSites nodes and its covers are a
        # subset of the hopping pairs.
        self.assertEqual(chain.partialOrder.getNodeCount, chain.nSites)
        cover_set = set(chain.partialOrder.covers)
        hop_set = set(chain.hoppingPairs)
        self.assertTrue(
            cover_set <= hop_set,
            f"covers {cover_set} not subset of hops {hop_set}"
        )


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without TESSERA_QUANTUM=1")
class TestPosetFromSpacetimePython(unittest.TestCase):
    """Direct Python access to tessera.Poset.fromSpacetime."""

    def test_from_spacetime_returns_valid_poset(self) -> None:
        st = _build_default_cdt(num_simplices=20)
        poset = Poset.fromSpacetime(st)
        self.assertEqual(poset.getNodeCount, st.getVertexList().toVector().__len__())
        covers = poset.covers
        self.assertEqual(len(set(covers)), len(covers))
        for a, b in covers:
            self.assertNotEqual(a, b)
            self.assertGreaterEqual(a, 0)
            self.assertLess(a, poset.getNodeCount)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, poset.getNodeCount)

    def test_to_dot_renders(self) -> None:
        st = _build_default_cdt(num_simplices=20)
        poset = Poset.fromSpacetime(st)
        dot = poset.toDot()
        self.assertIn("digraph poset", dot)
        for a, b in poset.covers:
            self.assertIn(f"{a} -> {b}", dot)


if __name__ == "__main__":
    unittest.main()
