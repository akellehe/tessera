# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The band-filling covariance rule of `SelfConsistentMeanField`.

Under `CovarianceRule.BandFilling` the ordered spectrum of h_1 is grouped into
bands of degenerate eigenvalues and band b of rank r_b carries the declared
occupation n_b spread evenly over it: Gamma = sum_b (n_b / r_b) P_b. The host is
the three-sheeted unit-monopole tetrahedron. Its covariant operator h_1 (the
Whitney pencil) has six simple eigenvalues per sheet at the monopole
connection, so each band is one base mode times the three sheets, rank three.
"""

import unittest

import numpy as np

import tessera as T
from tessera.drivers import baryon_poles as bp

cob = T.cobordism


def _mean_field(occupations, iterations=1):
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = list(occupations)
    declaration.maximum_iterations = iterations
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = False
    geometry.relax_links = False
    geometry.relax_multipliers = True
    geometry.maximum_iterations = 1
    declaration.geometry = geometry
    return declaration


def _action(matter_weight=1.0):
    spacetime = bp.build_host()
    declaration = bp.action_declaration(spacetime, 1.0, 1.0,
                                        matter_weight=matter_weight)
    return spacetime, cob.JointAction(spacetime, declaration)


class TestBandFilling(unittest.TestCase):
    def test_the_host_has_six_bands_of_rank_three(self):
        _, action = _action()
        report = cob.SelfConsistentMeanField(action,
                                             _mean_field([1, 1, 1])).solve()
        self.assertEqual(list(report.band_ranks), [3] * 6)

    def test_the_covariance_is_the_weighted_band_projector(self):
        _, action = _action()
        report = cob.SelfConsistentMeanField(action,
                                             _mean_field([1.5, 0, 0])).solve()
        gamma = bp.matrix(report.covariance)
        lowest = bp.matrix(action.occupation_projector(3))
        self.assertLess(np.max(np.abs(gamma - 0.5 * lowest)), 1e-10)
        self.assertLess(abs(np.trace(gamma) - 1.5), 1e-10)
        h = bp.matrix(action.carrier_operator())
        self.assertLess(np.max(np.abs(gamma @ h - h @ gamma)), 1e-9)
        # half-filled: Gamma^2 = Gamma / 2 on the band, so not a projector
        self.assertGreater(report.purity_defect, 0.1)

    def test_a_full_filling_is_a_projector(self):
        _, action = _action()
        report = cob.SelfConsistentMeanField(action,
                                             _mean_field([3, 0, 0])).solve()
        self.assertLess(report.purity_defect, 1e-9)

    def test_the_band_projector_does_not_depend_on_the_order_inside_a_band(
            self):
        """Every band's projector is a difference of prefix projectors that end
        on band boundaries; filling bands one and two evenly gives one third
        of their joint projector whatever order the eigensolver left inside
        each."""
        _, action = _action()
        report = cob.SelfConsistentMeanField(action,
                                             _mean_field([1, 1, 0])).solve()
        gamma = bp.matrix(report.covariance)
        joint = bp.matrix(action.occupation_projector(6))
        self.assertLess(np.max(np.abs(gamma - joint / 3.0)), 1e-10)

    def test_refusals(self):
        _, action = _action()
        for occupations in ([], [0, 0, 0], [-1, 2, 0]):
            with self.assertRaises(ValueError):
                cob.SelfConsistentMeanField(action, _mean_field(occupations))
        with self.assertRaises(ValueError):
            cob.SelfConsistentMeanField(action, _mean_field([4])).solve()
        with self.assertRaises(ValueError):
            cob.SelfConsistentMeanField(action,
                                        _mean_field([1] * 7)).solve()

    def test_bands_read_under_a_declared_symmetry(self):
        """With the rotation action declared, the bands are those of the
        T-averaged operator: three doublets times three sheets, rank six, and
        the covariance commutes with every D_1(g)."""
        _, action = _action()
        actions = bp.rotation_action([bp.monopole_support()] * bp.SHEETS)
        declaration = _mean_field([1, 1, 1])
        declaration.band_symmetry = [list(d.reshape(-1)) for d in actions]
        report = cob.SelfConsistentMeanField(action, declaration).solve()
        self.assertEqual(list(report.band_ranks), [6, 6, 6])
        gamma = bp.matrix(report.covariance)
        self.assertLess(abs(np.trace(gamma) - 3.0), 1e-10)
        for d in actions:
            self.assertLess(np.max(np.abs(d @ gamma - gamma @ d)), 1e-10)
        # one particle in each of the three bands, spread evenly: the identity
        # over the 18 cells divided by six
        self.assertLess(np.max(np.abs(gamma - np.eye(18) / 6.0)), 1e-10)

    def test_a_malformed_symmetry_is_refused(self):
        _, action = _action()
        declaration = _mean_field([1])
        declaration.band_symmetry = [[1.0, 0.0, 0.0, 1.0]]
        with self.assertRaises(ValueError):
            cob.SelfConsistentMeanField(action, declaration).solve()

    def test_the_projector_rule_is_unchanged(self):
        _, action = _action()
        declaration = _mean_field([1])
        declaration.covariance_rule = cob.CovarianceRule.OccupiedProjector
        declaration.occupied_modes = 3
        report = cob.SelfConsistentMeanField(action, declaration).solve()
        self.assertEqual(list(report.band_ranks), [])
        self.assertLess(report.purity_defect, 1e-9)


if __name__ == "__main__":
    unittest.main()
