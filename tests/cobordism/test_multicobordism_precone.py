# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The `precone` knob (#533): pre-grow the emergent seed by N gated cone-in moves
before optimization.

`precone` is a constructor argument on the C++ `MultiCobordism` (the source of truth),
threaded through `ProtonSynthesis` (which builds the animation's nodes from a single Δ⁴ seed) and
surfaced as `--precone` on the animation. Each pre-cone adds one top cell on a fresh apex
over a facet and is accepted only through the same `dualComplexValid` gate stage 1 uses, so
the pre-growth stays a valid manifold-with-boundary — nothing is inserted by fiat. It is the
emergent analogue of a prebuilt host refinement.
"""
import unittest

import tessera
import cmath

cob = tessera.cobordism

_DIM = 4


def _single_delta4():
    """A single Δ⁴ simplex (one pentatope: 5 vertices, 1 top cell, Betti [1,0,0,0,0], a
    contractible 4-ball) with a uniform ℓ²=1 metric — the same minimal emergent seed
    `ProtonSynthesis.buildMinimalSeed` grows the proton out of."""
    sig = tessera.Signature(_DIM, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(_DIM))
    st.build()
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(1.0)))
    return st


def _n_cells(st):
    return len(st.getTopSimplices())


def _is_manifold(st, k=3):
    ok, _why = cob.EigenstateSynthesis(st, k).dualComplexValid()
    return ok


# Minimal, well-formed targets — precone runs in the ctor irrespective of the states, so a
# 1-vector input and output are enough to construct a MultiCobordism.
_IN = [[complex(1.0, 0.0)]]
_OUT = [[complex(1.0, 0.0)]]


class PreconeMultiCobordismTest(unittest.TestCase):
    """The `precone` flag directly on the `MultiCobordism` constructor."""

    def _mc(self, seed, precone):
        return cob.MultiCobordism(_single_delta4(), _IN, _OUT, degrees=[3],
                                  gamma=1.0, seed=seed, precone=precone)

    def test_zero_is_a_noop(self):
        # precone=0 leaves the single-Δ⁴ seed untouched: exactly one top cell.
        mc = self._mc(seed=1, precone=0)
        self.assertEqual(_n_cells(mc.st), 1)
        self.assertEqual(list(cob.MultiCobordism.betti(mc.st)), [1, 0, 0, 0, 0])

    def test_precone_grows_the_complex(self):
        # precone=N cones N fresh top cells into the seed (gated; best-effort), so the
        # complex grows well beyond the bare seed.
        mc = self._mc(seed=1, precone=8)
        self.assertGreater(_n_cells(mc.st), 1)
        self.assertGreaterEqual(_n_cells(mc.st), 1 + 4)   # most gated cone-ins succeed

    def test_precone_stays_a_valid_manifold_and_contractible(self):
        # Every accepted cone-in passes the dualComplexValid gate, so the grown complex
        # is still a manifold-with-boundary; cone-in caps (never opens) so the 4-ball
        # stays contractible (Betti [1,0,0,0,0]) — no hole is created.
        mc = self._mc(seed=2, precone=10)
        self.assertTrue(_is_manifold(mc.st))
        self.assertEqual(list(cob.MultiCobordism.betti(mc.st)), [1, 0, 0, 0, 0])

    def test_more_precone_grows_at_least_as_much(self):
        small = _n_cells(self._mc(seed=4, precone=3).st)
        large = _n_cells(self._mc(seed=4, precone=12).st)
        self.assertGreater(large, small)

    def test_reproducible_given_seed(self):
        # Same (seed, precone) → identical pre-growth (the RNG is the ctor seed).
        a = _n_cells(self._mc(seed=7, precone=6).st)
        b = _n_cells(self._mc(seed=7, precone=6).st)
        self.assertEqual(a, b)


class PreconeThroughProtonTest(unittest.TestCase):
    """`precone` threaded through `ProtonSynthesis` into both node factories — the path the
    animation drives. Also guards the id-capture fix: with precone>0 the ctor regrows the
    complex, so the factories must anchor `seed_inputs`/`seed_outputs` by vertex id (stable
    across the rebuild), not by a stale Vertex handle."""

    def test_recombination_node_precone_grows_and_seeds(self):
        bare = cob.ProtonSynthesis(seed=3, precone=0).recombination_node(3)
        grown = cob.ProtonSynthesis(seed=3, precone=8).recombination_node(3)
        self.assertEqual(_n_cells(bare.st), 1)              # bare single-Δ⁴ seed
        self.assertGreater(_n_cells(grown.st), 1)           # pre-grown
        self.assertTrue(_is_manifold(grown.st))
        # The id-capture fix held: the two inputs and two outputs were seeded on the grown
        # complex (non-empty regions), not lost to a dangling handle.
        self.assertEqual(len(grown.inputs), 2)
        self.assertEqual(len(grown.outputs), 2)
        for block in list(grown.inputs) + list(grown.outputs):
            self.assertTrue(list(block.vertices))

    def test_formation_node_precone_grows_and_seeds(self):
        bare = cob.ProtonSynthesis(seed=5, precone=0).formation_node(6)
        grown = cob.ProtonSynthesis(seed=5, precone=8).formation_node(6)
        self.assertEqual(_n_cells(bare.st), 1)
        self.assertGreater(_n_cells(grown.st), 1)
        self.assertTrue(_is_manifold(grown.st))
        self.assertEqual(len(grown.inputs), 2)              # diquark + third quark
        for block in grown.inputs:
            self.assertTrue(list(block.vertices))


if __name__ == "__main__":
    unittest.main()
