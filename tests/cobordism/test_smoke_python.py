# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Smoke tests for the cobordism subsystem scaffold (issue #62)."""

import itertools
import unittest

import tessera
import cmath

# Subsystem submodules are exposed as attributes of the `tessera` package
# (via tessera/__init__.py), matching mesh/spacetime/observables/simulations.
# Dotted `import tessera.cobordism` is not supported for the C++ subsystems
# (only the pure-Python `tessera.quantum` package allows it).
cobordism = tessera.cobordism


class TestCobordismScaffold(unittest.TestCase):

    def test_submodule_imports(self):
        self.assertTrue(hasattr(tessera, "cobordism"))
        self.assertTrue(cobordism._cobordism_smoke())

    def test_combinatorial_dimension_observable_empty(self):
        # Per-complex measurements are Observables (#62). Empty complex -> -1.
        st = tessera.Spacetime()
        self.assertEqual(cobordism.CombinatorialDimension().compute(st), -1.0)

    def test_combinatorial_dimension_is_intrinsic_not_declared(self):
        # Build S^2 = ∂Δ³ (triangles) inside a 4D-declared Spacetime. The
        # combinatorial dimension is 2, independent of the Spacetime's signature.
        st = tessera.Spacetime()
        V = [st.create_vertex(i, [0.0]) for i in range(4)]
        for a, b in itertools.combinations(range(4), 2):
            st.create_edge(V[a], V[b], cmath.sqrt(complex(1.0)))
        for combo in itertools.combinations(range(4), 3):
            st.create_simplex([V[i] for i in combo])
        self.assertEqual(cobordism.CombinatorialDimension().compute(st), 2.0)


if __name__ == "__main__":
    unittest.main()
