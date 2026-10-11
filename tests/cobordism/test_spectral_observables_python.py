# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Hodge spectral Observables (#95): SpectralGap and HarmonicDimension.

Thin scalar wrappers over HodgeLaplacian (mirroring how EulerCharacteristic /
Signature wrap ChainComplex). They carry the C4/C5 content as single numbers:
the spectral gap collapses at flux Phi=pi, and a nonzero U(1) flux lifts the
harmonic zero-mode (dim ker L0: 1 -> 0) while the flux-independent topological
b0 from ChainComplex is unchanged.
"""

import math
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism
obs = tessera.observables  # SpectralGap / HarmonicDimension live in the observables subsystem


# --------------------------------------------------------------------------- #
# Fixtures (same idioms as the HodgeLaplacian operator tests)
# --------------------------------------------------------------------------- #
def _build_topology(topology):
    sig = tessera.Signature(topology.dimension(), tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, topology)
    st.build()
    return st


def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.create_vertex(i) for i in range(num_vertices)]
    for s in simplices:
        st.create_simplex([verts[i] for i in s])
    return st


def _set_uniform(st, sq=1.0, phase=0.0):
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(sq)))
        e.set_phase(phase)


def _triangle(phi=0.0):
    """S^1 = boundary of a 2-simplex; total flux Phi placed on one edge."""
    st = _build_topology(tessera.SimplexBoundarySphere(1))
    _set_uniform(st, 1.0, 0.0)
    if phi:
        st.get_edge_list().to_vector()[0].set_phase(phi)
    return st


def _path():
    st = _from_simplices(3, [(0, 1), (1, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


def _testbed():
    st = _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    _set_uniform(st, 1.0, 0.0)
    return st


class TestSpectralGap(unittest.TestCase):

    def test_matches_operator_first_gap(self):
        # The Observable equals lambda_1 - lambda_0 read off HodgeLaplacian's
        # U(1) connection Laplacian D - A, the operator it is defined on
        # (SpectralGap::compute); eigenvalues() is L_0 of the default metric
        # source, a different operator since that source became the Whitney
        # pencil.
        for name, st in (("triangle", _triangle()), ("path", _path()),
                         ("testbed", _testbed())):
            with self.subTest(fixture=name):
                raw = np.asarray(
                    cob.HodgeLaplacian(st).connection_eigenvalues(),
                    dtype=complex)
                # D - A is Hermitian: real spectrum, asserted rather than
                # projected.
                np.testing.assert_allclose(raw.imag, 0.0, atol=1e-12)
                evals = sorted(raw.real)
                self.assertAlmostEqual(obs.SpectralGap().compute(st),
                                       evals[1] - evals[0], places=12)

    def test_triangle_zero_flux_gap_is_three(self):
        # {0, 3, 3} -> gap 3.
        self.assertAlmostEqual(obs.SpectralGap().compute(_triangle()), 3.0,
                               places=12)

    def test_gap_collapses_at_half_flux_quantum(self):
        # C4: Phi=pi -> {1, 1, 4}, the two lowest modes degenerate -> gap 0.
        self.assertAlmostEqual(obs.SpectralGap().compute(_triangle(math.pi)),
                               0.0, places=12)

    def test_gap_decreases_from_three_to_zero_with_flux(self):
        gaps = [obs.SpectralGap().compute(_triangle(phi))
                for phi in (0.0, math.pi / 3, math.pi / 2,
                            2 * math.pi / 3, math.pi)]
        for earlier, later in zip(gaps, gaps[1:]):
            self.assertLessEqual(later, earlier + 1e-9)
        self.assertAlmostEqual(gaps[0], 3.0, places=9)
        self.assertAlmostEqual(gaps[-1], 0.0, places=9)

    def test_empty_spacetime_returns_zero(self):
        self.assertEqual(obs.SpectralGap().compute(tessera.Spacetime()), 0.0)


class TestHarmonicDimension(unittest.TestCase):

    def test_zero_flux_equals_b0(self):
        # dim ker L0 at zero flux = number of connected components = b0.
        for name, st in (("triangle", _triangle()), ("path", _path()),
                         ("testbed", _testbed())):
            with self.subTest(fixture=name):
                b0 = cob.ChainComplex.from_spacetime(st).betti_numbers()[0]
                self.assertEqual(obs.HarmonicDimension().compute(st), float(b0))

    def test_flux_lifts_the_zero_mode(self):
        # C4/C5: a nonzero flux lifts the harmonic (1 -> 0); the topological b0
        # from ChainComplex is flux-independent and stays at 1.
        st0, stpi = _triangle(), _triangle(math.pi)
        self.assertEqual(obs.HarmonicDimension().compute(st0), 1.0)
        self.assertEqual(obs.HarmonicDimension().compute(stpi), 0.0)
        self.assertEqual(cob.ChainComplex.from_spacetime(stpi).betti_numbers()[0], 1)

    def test_matches_operator_harmonic_count(self):
        st = _triangle()
        hl = cob.HodgeLaplacian(st)
        # harmonics() is one Cochain per ker L_0 basis vector => len == dim ker.
        self.assertEqual(obs.HarmonicDimension().compute(st),
                         float(len(hl.harmonics())))

    def test_empty_spacetime_returns_zero(self):
        self.assertEqual(obs.HarmonicDimension().compute(tessera.Spacetime()), 0.0)


if __name__ == "__main__":
    unittest.main()
