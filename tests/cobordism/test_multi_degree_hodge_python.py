# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""#859 — the Hodge term over several degrees, decoupled from the register degree.

The term is summed over a degree list a caller declares HERE, and never over
the register degrees, which answer the unrelated question of where a register
is constructed. The guard that matters is that decoupling changed only the
term's CONFIGURATION and not the term: declaring the degrees a caller used
before reproduces the old value and the old descent direction to the BIT.
"""

import cmath
import math
import unittest

import tessera as T

cob = T.cobordism
MC = cob.MultiCobordism
MODE = cob.HodgeEntropyPhaseMode.IncludeComplexPhase


def _host(jitter=True):
    """A closed 4-manifold with a mild non-degenerate metric.

    The boundary of a 5-simplex is S^4, so every degree 0..4 has cells and no
    degree's entropy is trivially empty.
    """
    spacetime = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                            1.0, 1.0, T.PREFERRED, T.SimplexBoundarySphere(4))
    spacetime.build()
    for index, edge in enumerate(spacetime.getEdgeList().toVector()):
        squared = 1.0 + (0.01 * (index % 6) if jitter else 0.0)
        edge.setLength(cmath.sqrt(complex(squared)))
    return spacetime


def _node(hodge_degrees=None, weights=None, register_degrees=(1,),
          spacetime=None):
    node = MC(spacetime if spacetime is not None else _host(), [], [],
              list(register_degrees), 1.0, 7)
    node.set_objective(cob.JointStationarityObjective())
    if hodge_degrees is not None:
        node.set_hodge_degrees(list(hodge_degrees),
                               [] if weights is None else list(weights))
    return node


class HodgeDegreeDefaultTest(unittest.TestCase):
    """The default is degree zero alone, and it is not the register degrees."""

    def test_the_default_is_degree_zero(self):
        self.assertEqual(list(_node().hodge_degrees), [0])

    def test_the_default_weights_are_empty_meaning_uniform(self):
        self.assertEqual(list(_node().hodge_degree_weights), [])

    def test_the_default_does_not_follow_the_register_degrees(self):
        """The decoupling, stated as a test.

        A node declared over register degree 3 still scores the Hodge term at
        degree 0. If the list were inherited — as a default, a fallback, or an
        "if empty then" — this would read [3].
        """
        node = _node(register_degrees=(3,))
        self.assertEqual(list(node.hodge_degrees), [0])

    def test_the_term_does_not_depend_on_the_register_degrees(self):
        """Independence, measured rather than asserted.

        Two nodes over the SAME complex declaring the same Hodge degrees but
        different register degrees must score the identical Hodge term. Any
        surviving path from the register list into the term — a default, a
        fallback, a stray read — would make these differ.
        """
        spacetime = _host()
        one = _node(hodge_degrees=[2], register_degrees=(1,),
                    spacetime=spacetime)
        other = _node(hodge_degrees=[2], register_degrees=(3,),
                      spacetime=spacetime)
        self.assertEqual(one.objective_terms().hodge_stationarity,
                         other.objective_terms().hodge_stationarity)
        self.assertEqual(list(one.hodge_degrees), [2])
        self.assertEqual(list(other.hodge_degrees), [2])


class HodgeDegreeBitIdentityTest(unittest.TestCase):
    """Declaring today's degrees reproduces today's arithmetic exactly.

    The reference is not a remembered number but an independent reconstruction
    from the same primitives the pre-change code called, so the assertion is
    that the TERM is unchanged and only its configuration moved.
    """

    def test_the_term_equals_the_primitive_bitwise_at_every_degree(self):
        spacetime = _host()
        laplacian = cob.HodgeLaplacian(spacetime)
        for degree in (0, 1, 2, 3, 4):
            with self.subTest(degree=degree):
                node = _node(hodge_degrees=[degree], spacetime=spacetime)
                expected = (node.hodge_entropy_weight *
                            laplacian.spectralEntropyGradientNorm(degree, MODE))
                self.assertEqual(node.objective_terms().hodge_stationarity,
                                 expected)

    def test_the_direction_equals_the_reconstruction_bitwise(self):
        """The descent direction, not merely the value.

        Regge is switched off so the ascent is the Hodge contribution alone and
        the comparison is unambiguous.
        """
        spacetime = _host()
        laplacian = cob.HodgeLaplacian(spacetime)
        edge_count = len(spacetime.getEdgeList().toVector())
        degree = 1

        context = cob.ObjectiveContext()
        context.spacetime = spacetime
        context.hodge_degrees = [degree]
        context.hodge_entropy_weight = 1.0
        context.regge_weight = 0.0
        context.einstein_hilbert = False
        direction_context = cob.ObjectiveDirectionContext()
        direction_context.scalar = context
        direction_context.edge_count = edge_count

        produced = cob.JointStationarityObjective().direction(direction_context)

        base = laplacian.spectralEntropyGradient(degree, MODE)
        ascent_direction = [complex(component).conjugate() for component in base]
        derivative = laplacian.spectralEntropyGradientDirectionalDerivative(
            degree, ascent_direction, MODE)
        expected = [1.0 * 2.0 * complex(component).conjugate()
                    for component in derivative]

        self.assertEqual(len(list(produced.ascent)), edge_count)
        for index, (got, want) in enumerate(zip(produced.ascent, expected)):
            with self.subTest(edge=index):
                self.assertEqual(complex(got), complex(want))

    def test_the_baseline_equals_the_primitive_bitwise(self):
        spacetime = _host()
        laplacian = cob.HodgeLaplacian(spacetime)
        context = cob.ObjectiveContext()
        context.spacetime = spacetime
        context.hodge_degrees = [1]
        context.hodge_entropy_weight = 1.0
        context.regge_weight = 0.0
        context.einstein_hilbert = False
        direction_context = cob.ObjectiveDirectionContext()
        direction_context.scalar = context
        direction_context.edge_count = len(
            spacetime.getEdgeList().toVector())
        produced = cob.JointStationarityObjective().direction(direction_context)
        self.assertEqual(produced.baseline,
                         laplacian.spectralEntropyGradientNorm(1, MODE))

    def test_a_uniform_weight_of_one_changes_nothing(self):
        """Multiplying by exactly 1 is exact, so declaring it is a no-op."""
        implicit = _node(hodge_degrees=[1]).objective_terms().hodge_stationarity
        explicit = _node(hodge_degrees=[1],
                         weights=[1.0]).objective_terms().hodge_stationarity
        self.assertEqual(implicit, explicit)


class HodgeDegreeSumTest(unittest.TestCase):
    """Several degrees sum, and the recorded shares add back up."""

    def test_the_degrees_are_distinct_information(self):
        """k and 4-k are not one condition counted twice.

        The continuum Laplacians at k and n-k on a closed oriented n-manifold
        are isospectral through the Hodge star. The discrete weighted operator
        does NOT inherit that, which is what makes scoring several degrees worth
        doing rather than redundant.
        """
        norms = {}
        for degree in (0, 1, 2, 3, 4):
            node = _node(hodge_degrees=[degree])
            norms[degree] = node.objective_terms().hodge_stationarity
        self.assertNotEqual(norms[0], norms[4])
        self.assertNotEqual(norms[1], norms[3])
        self.assertEqual(len(set(norms.values())), 5)

    def test_the_total_is_the_sum_of_the_single_degree_terms(self):
        degrees = [0, 1, 2, 3]
        together = _node(hodge_degrees=degrees).objective_terms()
        apart = sum(_node(hodge_degrees=[degree]).objective_terms()
                    .hodge_stationarity for degree in degrees)
        self.assertAlmostEqual(together.hodge_stationarity, apart, delta=
                               abs(apart) * 1e-12)

    def test_the_shares_reproduce_the_term(self):
        node = _node(hodge_degrees=[0, 1, 2, 3])
        term = node.objective_terms().hodge_stationarity
        shares = sum(contribution.contribution
                     for contribution in node.hodge_degree_contributions)
        # To double round-off rather than to the bit: the term applies the
        # entropy weight once to the accumulated weighted norms, each share
        # carries its own multiply.
        self.assertAlmostEqual(shares, term, delta=abs(term) * 1e-12)

    def test_the_shares_are_reported_in_declaration_order(self):
        node = _node(hodge_degrees=[3, 1, 0])
        self.assertEqual(
            [contribution.degree
             for contribution in node.hodge_degree_contributions],
            [3, 1, 0])

    def test_an_unweighted_norm_is_reported_alongside_the_share(self):
        """The raw spread stays visible rather than folded into the weight."""
        node = _node(hodge_degrees=[1], weights=[4.0])
        contribution = node.hodge_degree_contributions[0]
        self.assertEqual(contribution.weight, 4.0)
        self.assertEqual(contribution.contribution,
                         node.hodge_entropy_weight * 4.0 *
                         contribution.gradient_norm_squared)
        unweighted = _node(hodge_degrees=[1]).hodge_degree_contributions[0]
        self.assertEqual(contribution.gradient_norm_squared,
                         unweighted.gradient_norm_squared)

    def test_an_objective_without_a_hodge_term_reports_no_breakdown(self):
        node = MC(_host(), [], [], [1], 1.0, 7)
        node.set_objective(cob.LegacyObjective())
        self.assertEqual(list(node.hodge_degree_contributions), [])


class HodgeDegreeWeightTest(unittest.TestCase):
    """A weight scales exactly one degree's share."""

    def test_a_weight_scales_only_its_own_degree(self):
        plain = {contribution.degree: contribution.contribution
                 for contribution in
                 _node(hodge_degrees=[0, 1, 2, 3]).hodge_degree_contributions}
        scaled = {contribution.degree: contribution.contribution
                  for contribution in
                  _node(hodge_degrees=[0, 1, 2, 3],
                        weights=[1.0, 3.0, 1.0, 1.0])
                  .hodge_degree_contributions}
        self.assertEqual(scaled[1], plain[1] * 3.0)
        for degree in (0, 2, 3):
            with self.subTest(degree=degree):
                self.assertEqual(scaled[degree], plain[degree])

    def test_a_zero_weight_removes_a_degree_from_the_total(self):
        without = _node(hodge_degrees=[0, 2]).objective_terms()
        with_muted = _node(hodge_degrees=[0, 1, 2],
                           weights=[1.0, 0.0, 1.0]).objective_terms()
        self.assertAlmostEqual(with_muted.hodge_stationarity,
                               without.hodge_stationarity,
                               delta=abs(without.hodge_stationarity) * 1e-12)

    def test_the_weight_reaches_the_descent_direction(self):
        """A weight must move the direction, not only the reported scalar."""
        spacetime = _host()
        edge_count = len(spacetime.getEdgeList().toVector())

        def ascent(weight):
            context = cob.ObjectiveContext()
            context.spacetime = spacetime
            context.hodge_degrees = [1]
            context.hodge_degree_weights = [weight]
            context.hodge_entropy_weight = 1.0
            context.regge_weight = 0.0
            context.einstein_hilbert = False
            direction_context = cob.ObjectiveDirectionContext()
            direction_context.scalar = context
            direction_context.edge_count = edge_count
            return list(cob.JointStationarityObjective()
                        .direction(direction_context).ascent)

        single = ascent(1.0)
        doubled = ascent(2.0)
        moved = [index for index, value in enumerate(single)
                 if abs(complex(value)) > 0.0]
        self.assertTrue(moved, "fixture moves no coordinate; nothing is tested")
        for index in moved:
            with self.subTest(edge=index):
                self.assertAlmostEqual(
                    abs(complex(doubled[index])),
                    2.0 * abs(complex(single[index])),
                    delta=abs(complex(single[index])) * 1e-9)


class HodgeDegreeValidationTest(unittest.TestCase):
    """A malformed declaration fails loudly and by name."""

    def test_an_empty_degree_list_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            _node().set_hodge_degrees([])
        self.assertIn("at least one degree", str(raised.exception))

    def test_a_negative_degree_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            _node().set_hodge_degrees([0, -1])
        self.assertIn("negative", str(raised.exception))

    def test_a_repeated_degree_is_refused(self):
        """A repeat would double-count while reading as distinct degrees."""
        with self.assertRaises(ValueError) as raised:
            _node().set_hodge_degrees([1, 2, 1])
        self.assertIn("more than once", str(raised.exception))

    def test_a_mismatched_weight_list_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            _node().set_hodge_degrees([0, 1], [1.0])
        self.assertIn("one weight per degree", str(raised.exception))

    def test_a_negative_weight_is_refused(self):
        with self.assertRaises(ValueError):
            _node().set_hodge_degrees([0], [-1.0])

    def test_a_non_finite_weight_is_refused(self):
        with self.assertRaises(ValueError):
            _node().set_hodge_degrees([0], [math.inf])

    def test_a_refused_declaration_leaves_the_previous_one_intact(self):
        node = _node(hodge_degrees=[1, 2])
        with self.assertRaises(ValueError):
            node.set_hodge_degrees([0, 0])
        self.assertEqual(list(node.hodge_degrees), [1, 2])


class HodgeDegreeFirewallTest(unittest.TestCase):
    """The new inputs are declared configuration, not a route to the engine."""

    def test_the_new_fields_are_on_the_enumerated_input_list(self):
        names = cob.ObjectiveContext.input_names()
        self.assertIn("hodge_degrees", names)
        self.assertIn("hodge_degree_weights", names)

    def test_the_input_list_matches_the_readable_surface(self):
        """Every enumerated input is readable, so the firewall list and the
        surface a Python objective actually sees cannot drift apart."""
        context = cob.ObjectiveContext()
        for name in cob.ObjectiveContext.input_names():
            with self.subTest(field=name):
                self.assertTrue(hasattr(context, name))

    def test_an_objective_scores_degrees_with_no_node_in_existence(self):
        """The breakdown is computed from plain data alone."""
        context = cob.ObjectiveContext()
        context.spacetime = _host()
        context.hodge_degrees = [0, 2]
        context.hodge_entropy_weight = 1.0
        context.einstein_hilbert = False
        context.regge_weight = 0.0
        contributions = (cob.JointStationarityObjective()
                         .hodge_degree_contributions(context))
        self.assertEqual([c.degree for c in contributions], [0, 2])
        for contribution in contributions:
            with self.subTest(degree=contribution.degree):
                self.assertGreater(contribution.gradient_norm_squared, 0.0)


class HodgeDegreeDriverTest(unittest.TestCase):
    """The animation declares its own Hodge degrees."""

    @staticmethod
    def _driver():
        from tessera.drivers import emergence
        return emergence

    def test_the_driver_declares_all_four_degrees(self):
        driver = self._driver()
        self.assertEqual(list(driver.DECLARED_HODGE_DEGREES), [0, 1, 2, 3])

    def test_the_driver_keeps_the_lists_separate(self):
        """The register degrees are unchanged by the Hodge declaration."""
        driver = self._driver()
        self.assertEqual(list(driver.DECLARED_REGISTER_DEGREES), [1])
        self.assertNotEqual(list(driver.DECLARED_HODGE_DEGREES),
                            list(driver.DECLARED_REGISTER_DEGREES))

    def test_the_config_carries_the_hodge_degrees(self):
        driver = self._driver()
        config = driver.build_config()
        self.assertEqual(config["hodge_degrees"], [0, 1, 2, 3])
        self.assertEqual(config["register_degrees"], [1])


if __name__ == "__main__":
    unittest.main()
