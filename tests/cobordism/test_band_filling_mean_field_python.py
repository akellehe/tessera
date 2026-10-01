# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The band-filling covariance rule of `SelfConsistentMeanField`.

Under `CovarianceRule.BandFilling` the ordered spectrum of h_1 is grouped into
bands of degenerate eigenvalues and band b of rank r_b carries the declared
occupation n_b spread evenly over it: Gamma = sum_b (n_b / r_b) P_b. The host is
the three-sheeted unit-monopole tetrahedron. Its covariant operator h_1 (the
Whitney pencil) has six eigenvalues per sheet at the monopole connection, each
carried once by every sheet. The three computed copies of an eigenvalue differ
at rounding (by up to 4e-14), and the bands are read at the declared band
tolerance 1e-15, at which the eighteen eigenvalues group into eight bands of
ranks (2, 1, 3, 3, 3, 2, 1, 3) in ascending order.

Every test reads the host as built (`SelfConsistentMeanField.read`): the
lengths and the links are not variables of the declared system, so nothing
moves.
"""

import unittest

import numpy as np

import tessera as T
from tessera.drivers import baryon_poles as bp

cob = T.cobordism

#: The ranks of the bands of h_1 on the host at the declared band tolerance,
#: in ascending order of real part.
HOST_BAND_RANKS = [2, 1, 3, 3, 3, 2, 1, 3]


def _mean_field(occupations):
    """The band-filling declaration at the declared tolerances, over a
    system whose lengths and links are held, so h_1 and its band filling are
    the host's."""
    declaration = cob.SelfConsistentMeanFieldDeclaration()
    declaration.covariance_rule = cob.CovarianceRule.BandFilling
    declaration.band_occupations = list(occupations)
    geometry = cob.HolomorphicRelaxationDeclaration()
    geometry.relax_lengths = False
    geometry.relax_links = False
    geometry.relax_multipliers = True
    declaration.geometry = geometry
    return declaration


def _action(matter_weight=1.0):
    spacetime = bp.build_host()
    declaration = bp.action_declaration(spacetime, 1.0, 1.0,
                                        matter_weight=matter_weight)
    return spacetime, cob.JointAction(spacetime, declaration)


def _read(action, declaration):
    return cob.SelfConsistentMeanField(action, declaration).read()


class TestBandFilling(unittest.TestCase):
    def test_the_host_has_eight_bands_at_the_declared_tolerance(self):
        _, action = _action()
        declaration = _mean_field([1, 1, 1])
        self.assertEqual(declaration.band_tolerance, 1e-15)
        report = _read(action, declaration)
        self.assertEqual(list(report.band_ranks), HOST_BAND_RANKS)
        self.assertEqual(sum(report.band_ranks), 18)
        self.assertEqual([band.rank for band in report.bands], [2, 1, 3])

    def test_the_covariance_is_the_weighted_band_projector(self):
        """An occupation 3/2 of the lowest band, of rank two: Gamma is 3/4
        of the band's projector, which is the projector onto the two lowest
        modes."""
        _, action = _action()
        report = _read(action, _mean_field([1.5, 0, 0]))
        gamma = bp.matrix(report.covariance)
        (band,) = report.bands
        self.assertEqual(band.rank, 2)
        self.assertLess(np.max(np.abs(gamma - 0.75 * bp.matrix(
            band.projector))), 1e-15)
        lowest = bp.matrix(action.occupation_projector(2))
        self.assertLess(np.max(np.abs(gamma - 0.75 * lowest)), 1e-15)
        self.assertLess(abs(np.trace(gamma) - 1.5), 1e-14)
        h = bp.matrix(action.carrier_operator())
        self.assertLess(np.max(np.abs(gamma @ h - h @ gamma)), 1e-13)
        # Gamma^2 = (3/4) Gamma on the band, so not a projector
        self.assertAlmostEqual(report.purity_defect, 0.26525047569832727,
                               places=12)

    def test_a_full_filling_is_a_projector(self):
        """The two lowest bands, of ranks two and one, filled: Gamma is the
        projector onto the three lowest modes."""
        _, action = _action()
        report = _read(action, _mean_field([2, 1, 0]))
        self.assertLess(report.purity_defect, 1e-15)
        gamma = bp.matrix(report.covariance)
        lowest = bp.matrix(action.occupation_projector(3))
        self.assertLess(np.max(np.abs(gamma - lowest)), 1e-15)

    def test_the_band_projector_does_not_depend_on_the_order_inside_a_band(
            self):
        """Every band's projector is a difference of prefix projectors that end
        on band boundaries; filling the third and fourth bands, each of rank
        three, with one particle each gives one third of their joint
        projector whatever order the eigensolver left inside each."""
        _, action = _action()
        report = _read(action, _mean_field([0, 0, 1, 1]))
        self.assertEqual([(band.declared_index, band.rank)
                          for band in report.bands], [(2, 3), (3, 3)])
        gamma = bp.matrix(report.covariance)
        joint = (bp.matrix(action.occupation_projector(9))
                 - bp.matrix(action.occupation_projector(3)))
        self.assertLess(np.max(np.abs(gamma - joint / 3.0)), 1e-15)

    def test_refusals(self):
        """An occupation rule that names no particle, or a negative number
        of them, is refused when the system is posed; an occupation a band
        cannot hold, and more occupations than the spectrum has bands, when
        the bands are read."""
        _, action = _action()
        for occupations in ([], [0, 0, 0], [-1, 2, 0]):
            with self.assertRaises(ValueError):
                cob.SelfConsistentMeanField(action, _mean_field(occupations))
        with self.assertRaisesRegex(ValueError, "band 0 has rank 2 and "
                                    "cannot hold the declared occupation"):
            _read(action, _mean_field([4]))
        self.assertEqual(list(_read(action, _mean_field([1] * 8)).band_ranks),
                         HOST_BAND_RANKS)
        with self.assertRaisesRegex(ValueError, "9 band occupations were "
                                    "declared but the spectrum groups into "
                                    "only 8 bands"):
            _read(action, _mean_field([1] * 9))

    def test_bands_read_under_a_declared_symmetry(self):
        """With the rotation action declared, the bands are those of the
        T-averaged operator: three doublets times three sheets, rank six, and
        the covariance commutes with every D_1(g)."""
        _, action = _action()
        actions = bp.rotation_action([bp.monopole_support()] * bp.SHEETS)
        declaration = _mean_field([1, 1, 1])
        declaration.band_symmetry = [list(d.reshape(-1)) for d in actions]
        report = _read(action, declaration)
        self.assertEqual(list(report.band_ranks), [6, 6, 6])
        gamma = bp.matrix(report.covariance)
        self.assertLess(abs(np.trace(gamma) - 3.0), 1e-14)
        for d in actions:
            self.assertLess(np.max(np.abs(d @ gamma - gamma @ d)), 1e-14)
        # one particle in each of the three bands, spread evenly: the identity
        # over the 18 cells divided by six
        self.assertLess(np.max(np.abs(gamma - np.eye(18) / 6.0)), 1e-15)

    def test_a_malformed_symmetry_is_refused(self):
        _, action = _action()
        declaration = _mean_field([1])
        declaration.band_symmetry = [[1.0, 0.0, 0.0, 1.0]]
        with self.assertRaisesRegex(ValueError, "must be square over the "
                                    "carrier's cells"):
            _read(action, declaration)

    def test_the_projector_rule_is_unchanged(self):
        _, action = _action()
        declaration = _mean_field([1])
        declaration.covariance_rule = cob.CovarianceRule.OccupiedProjector
        declaration.occupied_modes = 3
        report = _read(action, declaration)
        self.assertEqual(list(report.band_ranks), [])
        self.assertLess(report.purity_defect, 1e-15)
        gamma = bp.matrix(report.covariance)
        lowest = bp.matrix(action.occupation_projector(3))
        self.assertLess(np.max(np.abs(gamma - lowest)), 1e-15)


if __name__ == "__main__":
    unittest.main()
