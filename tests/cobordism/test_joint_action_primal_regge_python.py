# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The primal Regge form of `JointAction`.

The primal term is Regge's sum over hinges of the hinge content times the
deficit angle; in three dimensions sum_e l_e eps_e. Under the default hinge
rule only interior hinges (those whose link closes) contribute. It is checked
against an independent evaluation: the value against `ReggeSolver`, and the
analytic length gradient against a central difference of the value along the
real axis. The force of a carried state is checked to be the matter term's
alone, the Regge term standing on the geometric side it balances.
"""

import cmath
import os
import sys
import unittest

import numpy as np

import tessera as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _joint_action_hosts import kuhn_ball, sphere3, tetrahedron  # noqa: E402

cob = T.cobordism


def _declaration(**overrides):
    """A declaration with only the named terms switched on."""
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.gravitational_weight = 0.0
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 0.0
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    for name, value in overrides.items():
        setattr(declaration, name, value)
    return declaration


def _perturbed(index, squared):
    """A deterministic non-uniform Euclidean metric near the flat one."""
    return squared * (1.0 + 0.03 * ((index * 7) % 5 - 2))


def _central_difference(spacetime, value, step=1e-6):
    """d value / d z_e by a real-axis central difference in each squared
    length, restoring every edge afterwards."""
    edges = spacetime.getEdgeList().toVector()
    gradient = []
    for edge in edges:
        length = complex(edge.getLength())
        z = length * length
        edge.setLength(cmath.sqrt(z + step))
        up = value()
        edge.setLength(cmath.sqrt(z - step))
        down = value()
        edge.setLength(length)
        gradient.append((up - down) / (2 * step))
    return np.array(gradient)


class TestDefaults(unittest.TestCase):
    def test_primal_interior_is_the_default(self):
        declaration = cob.JointActionDeclaration()
        self.assertEqual(declaration.regge_form, cob.ReggeForm.Primal)
        self.assertEqual(declaration.regge_hinges, cob.ReggeHinges.Interior)

    def test_the_term_names_are_the_four_terms(self):
        self.assertEqual(cob.JointAction.term_names(),
                         ["regge", "holonomy", "matter", "spectral"])


class TestPrimalRegge(unittest.TestCase):
    def test_closed_complex_primal_equals_regge_solver(self):
        """On a closed complex every hinge is interior, so both hinge rules
        give ReggeSolver.reggeAction."""
        spacetime = sphere3(squared=lambda i: 1.0 + 0.05 * (i % 3))
        reference = T.ReggeSolver(spacetime,
                                  T.MatterConfiguration()).reggeAction()
        for rule in (cob.ReggeHinges.Interior, cob.ReggeHinges.All):
            action = cob.JointAction(spacetime, _declaration(
                gravitational_weight=1.0, regge_hinges=rule))
            self.assertEqual(action.regge_hinge_count(), 10)
            self.assertLess(abs(action.regge_term() - reference),
                            1e-12 * max(1.0, abs(reference)))

    def test_a_lone_tetrahedron_has_no_interior_hinge(self):
        spacetime = tetrahedron(squared=lambda i: 8.0)
        interior = cob.JointAction(spacetime, _declaration(
            gravitational_weight=1.0))
        self.assertEqual(interior.regge_hinge_count(), 0)
        self.assertEqual(interior.regge_term(), 0)
        self.assertLess(np.max(np.abs(interior.length_stationarity())), 1e-15)
        every = cob.JointAction(spacetime, _declaration(
            gravitational_weight=1.0, regge_hinges=cob.ReggeHinges.All))
        self.assertEqual(every.regge_hinge_count(), 6)
        reference = T.ReggeSolver(spacetime,
                                  T.MatterConfiguration()).reggeAction()
        self.assertLess(abs(every.regge_term() - reference), 1e-12)

    def test_the_weight_scales_the_term(self):
        spacetime = sphere3(squared=lambda i: 1.0 + 0.05 * (i % 3))
        one = cob.JointAction(spacetime, _declaration(gravitational_weight=1.0))
        half = cob.JointAction(spacetime,
                               _declaration(gravitational_weight=0.5))
        self.assertLess(abs(half.regge_term() - 0.5 * one.regge_term()),
                        1e-13)

    def test_dual_form_stays_available(self):
        spacetime = sphere3(squared=lambda i: 1.0 + 0.05 * (i % 3))
        action = cob.JointAction(spacetime, _declaration(
            gravitational_weight=1.0, regge_form=cob.ReggeForm.Dual))
        reference = T.ReggeSolver(spacetime,
                                  T.MatterConfiguration()).dualReggeAction()
        self.assertLess(abs(action.regge_term() - reference), 1e-12)

    def _check_gradient(self, spacetime, rule):
        action = cob.JointAction(spacetime, _declaration(
            gravitational_weight=1.0, regge_hinges=rule))
        analytic = np.array(action.length_stationarity())
        numeric = _central_difference(spacetime,
                                      lambda: complex(action.regge_term()))
        scale = max(1.0, np.max(np.abs(numeric)))
        self.assertLess(np.max(np.abs(analytic - numeric)) / scale, 1e-6)
        return analytic

    def test_gradient_matches_a_central_difference_on_a_closed_complex(self):
        self._check_gradient(sphere3(squared=lambda i: 1.0 + 0.05 * (i % 3)),
                             cob.ReggeHinges.Interior)

    def test_gradient_matches_on_a_ball_under_both_hinge_rules(self):
        for rule in (cob.ReggeHinges.Interior, cob.ReggeHinges.All):
            spacetime = kuhn_ball(2, scale=_perturbed)
            action = cob.JointAction(spacetime, _declaration(
                gravitational_weight=1.0, regge_hinges=rule))
            self.assertGreater(action.regge_hinge_count(), 0)
            self._check_gradient(spacetime, rule)

    def test_the_interior_rule_drops_the_boundary_of_a_ball(self):
        spacetime = kuhn_ball(2)
        interior = cob.JointAction(spacetime, _declaration(
            gravitational_weight=1.0)).regge_hinge_count()
        every = cob.JointAction(spacetime, _declaration(
            gravitational_weight=1.0,
            regge_hinges=cob.ReggeHinges.All)).regge_hinge_count()
        # The Kuhn cube of two divisions per axis has one interior vertex; the
        # interior hinges are the edges incident to it, whose links close.
        self.assertGreater(every, interior)
        self.assertGreater(interior, 0)


class TestTheForceOfTheCarriedState(unittest.TestCase):
    def test_the_hellmann_feynman_force_carries_no_geometric_term(self):
        """The force of the carried state is the matter term alone; the Regge
        term (every hinge of a tetrahedron, so that it has a gradient) belongs
        to the geometric side it is balanced against."""
        spacetime = tetrahedron(squared=lambda i: 8.0 + 0.4 * i)
        declaration = _declaration(gravitational_weight=0.7,
                                   regge_hinges=cob.ReggeHinges.All,
                                   matter_weight=1.0)
        declaration.covariance = list((np.eye(6) / 2.0).reshape(-1))
        with_regge = cob.JointAction(spacetime, declaration)
        declaration.gravitational_weight = 0.0
        without = cob.JointAction(spacetime, declaration)
        force = np.array(with_regge.hellmann_feynman_length_force())
        self.assertLess(np.max(np.abs(
            force - np.array(without.length_stationarity()))), 1e-14)
        regge = np.array(cob.JointAction(spacetime, _declaration(
            gravitational_weight=0.7,
            regge_hinges=cob.ReggeHinges.All)).length_stationarity())
        self.assertGreater(np.max(np.abs(regge)), 1e-2)
        self.assertLess(np.max(np.abs(
            np.array(with_regge.length_stationarity()) - force - regge)),
            1e-14)


if __name__ == "__main__":
    unittest.main()
