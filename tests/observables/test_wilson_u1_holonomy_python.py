# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""U(1) connection holonomy mode of observables::WilsonLoop (#114, T4).

WilsonMode.U1_CONNECTION accumulates the U(1) connection (``Edge.phase``)
around a closed cycle on the primal 1-skeleton, honoring each edge's stored
source->target orientation (``+phase`` forward, ``-phase`` reversed), and
returns the total holonomy mod 2*pi.

Validated against two independent references:

  * a numpy/hand oracle of the oriented phase sum (``_cycle_flux``), the same
    helper the Hodge-Laplacian tests use; and
  * the Stage-1 cycle flux carried by ``cobordism.HodgeLaplacian`` — recovered
    from the complex adjacency ``A[i,j] = squaredLength * exp(i*phase)`` it
    assembles (``_hodge_flux``), the cross-layer consistency T4 asserts.

The Z2 case (phases restricted to {0, pi}) checks that the bulk holonomy lands
in {0, pi} and matches the Stage-1 flux there.

Fixtures (shared idiom with tests/cobordism): build a HERMITIAN_WEIGHTED
spacetime directly from explicit simplex vertex tuples.
"""

import math
import types
import unittest

import numpy as np

import tessera
import cmath

cob = tessera.cobordism

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.create_vertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.create_simplex([verts[i] for i in simplex])
    return st


def _triangle():
    """S^1: the 1-skeleton cycle 0-1-2-0 (b1 = 1)."""
    return _from_simplices(3, [(0, 1), (1, 2), (2, 0)])


def _testbed():
    """Square 0-1-2-3-0 plus the diagonal 0-2 (b1 = 2)."""
    return _from_simplices(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


def _vertices_by_id(st):
    return {v.get_id(): v for v in st.get_vertex_list().to_vector()}


def _edge(st, a, b):
    for e in st.get_edge_list().to_vector():
        if {e.get_source().get_id(), e.get_target().get_id()} == {a, b}:
            return e
    raise KeyError((a, b))


def _set_phases(st, phases):
    """Assign edge phases from a {frozenset({a, b}): phi} map; unit positive
    weights everywhere, phase 0 on edges not in the map."""
    for e in st.get_edge_list().to_vector():
        key = frozenset({e.get_source().get_id(), e.get_target().get_id()})
        e.set_length(cmath.sqrt(complex(1.0)))
        e.set_phase(phases.get(key, 0.0))


# --------------------------------------------------------------------------- #
# Oracles and mod-2*pi helpers
# --------------------------------------------------------------------------- #
def _cycle_flux(st, cycle):
    """Hand oracle: directed holonomy sum of phase around a closed vertex-id
    cycle, honoring each edge's stored source->target orientation.

    The COMPACT part only. The connection phase is complex (C* = U(1) x R+),
    and a Wilson loop measures the U(1) winding: only Re has winding, while
    Im is a local scale that would turn the holonomy into an unbounded
    modulus."""
    total = 0.0
    n = len(cycle)
    for k in range(n):
        a, b = cycle[k], cycle[(k + 1) % n]
        e = _edge(st, a, b)
        if e.get_source().get_id() == a and e.get_target().get_id() == b:
            total += e.get_phase().real
        else:
            total -= e.get_phase().real
    return total


def _hodge_flux(st, cycle):
    """Stage-1 cycle flux read off cobordism.HodgeLaplacian's complex
    adjacency: arg(A[i,j]) is +phase along source->target and -phase reversed
    (unit weights), so the directed sum around the cycle is the same oriented
    holonomy the operator encodes in L = D - A."""
    ids = sorted(v.get_id() for v in st.get_vertex_list().to_vector())
    idx = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)
    A = np.array(cob.HodgeLaplacian(st).adjacency(), dtype=complex).reshape(n, n)
    total = 0.0
    m = len(cycle)
    for k in range(m):
        a, b = cycle[k], cycle[(k + 1) % m]
        total += np.angle(A[idx[a], idx[b]])
    return float(total)


def _wrap(theta):
    """Principal value in (-pi, pi] (matches WilsonLoop::principalAngle)."""
    r = math.remainder(theta, TWO_PI)
    if r <= -math.pi:
        r += TWO_PI
    return r


def _close_mod_2pi(a, b, tol=1e-9):
    """True iff a == b modulo 2*pi (robust to the +-pi boundary)."""
    return abs(math.remainder(a - b, TWO_PI)) < tol


def _holonomy(st, cycle):
    wl = tessera.WilsonLoop(st)
    by_id = _vertices_by_id(st)
    r = wl.evaluate_u1_connection([by_id[c] for c in cycle])
    if not hasattr(r, "loop_size"):  # degenerate/open cycles return {} unchanged
        return r
    # WilsonResult.value is complex-typed for the deficit-angle mode's sake and
    # in this mode holds the DERIVED residual_phase(), a real angle; this asserts
    # that rather than assuming it. The datum itself -- the unreduced complex
    # accumulation -- is passed through untouched for the tests that read it.
    v = complex(r.value)
    assert v.imag == 0.0, f"derived residual phase grew an imaginary part: {v}"
    return types.SimpleNamespace(value=v.real, loopSize=r.loop_size, read=r)


# --------------------------------------------------------------------------- #
# Holonomy vs the hand oracle, and total flux phi (mod 2*pi)
# --------------------------------------------------------------------------- #
class TestHolonomyVsOracle(unittest.TestCase):

    def test_zero_phase_is_trivial(self):
        st = _triangle()
        _set_phases(st, {})
        r = _holonomy(st, [0, 1, 2])
        self.assertEqual(r.loop_size, 3)
        self.assertAlmostEqual(r.value, 0.0, places=12)

    def test_matches_hand_oracle_single_edge(self):
        # All of Phi on one edge of the triangle; the holonomy equals the
        # oriented oracle sum exactly (both honor the stored orientation).
        for phi in (0.3, 1.0, 2.0, -0.7, math.pi / 2):
            with self.subTest(phi=phi):
                st = _triangle()
                _set_phases(st, {frozenset({0, 1}): phi})
                r = _holonomy(st, [0, 1, 2])
                self.assertEqual(r.loop_size, 3)
                self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

    def test_the_non_compact_part_carries_no_winding_but_is_still_measured(self):
        """Only the compact factor of C* = U(1) x R+ has winding, so an added
        imaginary part cannot move the WINDING. It does move the modulus, and
        both are reported: not quantizing is not a reason to discard (#872).

        The former version of this test asserted only the first half, which
        read as licence to drop Im(phase) entirely."""
        for phi, scale in ((0.3, 2.5), (-1.1, -0.75), (math.pi / 2, 4.0)):
            with self.subTest(phi=phi, scale=scale):
                compact = _triangle()
                _set_phases(compact, {frozenset({0, 1}): phi})
                twisted = _triangle()
                _set_phases(twisted, {frozenset({0, 1}): complex(phi, scale)})
                c = _holonomy(compact, [0, 1, 2])
                t = _holonomy(twisted, [0, 1, 2])
                # The winding is untouched -- the true half of the old claim.
                self.assertTrue(_close_mod_2pi(t.value, c.value))
                self.assertEqual(t.read.winding_number(), c.read.winding_number())
                # And the modulus is NOT: the scale is measured, not discarded.
                self.assertAlmostEqual(c.read.holonomy_modulus(), 1.0, places=12)
                self.assertAlmostEqual(t.read.holonomy_modulus(),
                                       math.exp(-scale), places=12)
                self.assertNotAlmostEqual(t.read.holonomy_modulus(), 1.0,
                                          places=6)

    def test_a_purely_non_compact_connection_is_measured_not_trivial(self):
        """An imaginary-only phase has zero winding, which the retired version
        of this test called 'trivial'. The real part being zero while the
        imaginary part is not is a MEASUREMENT, and the modulus reports it."""
        st = _triangle()
        _set_phases(st, {frozenset({0, 1}): complex(0.0, 1.7),
                         frozenset({1, 2}): complex(0.0, -0.4)})
        r = _holonomy(st, [0, 1, 2])
        sigma = complex(r.read.connection_accumulation)
        self.assertAlmostEqual(r.value, 0.0, places=12)   # no winding
        self.assertEqual(r.read.winding_number(), 0)
        self.assertAlmostEqual(sigma.real, 0.0, places=12)
        # The whole content of this connection lives in the part that used to
        # be thrown away.
        self.assertNotAlmostEqual(sigma.imag, 0.0, places=6)
        self.assertAlmostEqual(r.read.holonomy_modulus(),
                               math.exp(-sigma.imag), places=12)

    def test_equals_total_flux_mod_2pi(self):
        # Orient the cycle to hit the phased edge forward, so the holonomy is
        # exactly +phi (mod 2*pi). Covers Phi outside (-pi, pi] to exercise the
        # mod-2*pi reduction.
        for phi in (0.0, 1.0, math.pi, 4.0, 2 * math.pi, -5.0, 3 * math.pi):
            with self.subTest(phi=phi):
                st = _triangle()
                _set_phases(st, {frozenset({0, 1}): phi})
                e01 = _edge(st, 0, 1)
                s, t = e01.get_source().get_id(), e01.get_target().get_id()
                third = ({0, 1, 2} - {s, t}).pop()
                r = _holonomy(st, [s, t, third])
                self.assertTrue(_close_mod_2pi(r.value, phi),
                                f"holonomy {r.value} != Phi {phi} (mod 2pi)")
                # Reduced into the principal interval (-pi, pi].
                self.assertGreater(r.value, -math.pi - 1e-12)
                self.assertLessEqual(r.value, math.pi + 1e-12)

    def test_distributed_phases_sum(self):
        # Distinct phases on every edge of a testbed cycle; the holonomy is the
        # oriented sum over the three traversed edges.
        st = _testbed()
        _set_phases(st, {
            frozenset({0, 1}): 0.4,
            frozenset({1, 2}): -1.1,
            frozenset({0, 2}): 0.9,   # the (2, 0) leg of the cycle
        })
        r = _holonomy(st, [0, 1, 2])
        self.assertEqual(r.loop_size, 3)
        self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

    def test_orientation_reversal_negates(self):
        st = _testbed()
        _set_phases(st, {
            frozenset({0, 1}): 0.4,
            frozenset({1, 2}): -1.1,
            frozenset({0, 2}): 0.9,
        })
        fwd = _holonomy(st, [0, 1, 2]).value
        rev = _holonomy(st, [2, 1, 0]).value
        self.assertTrue(_close_mod_2pi(fwd, -rev))


# --------------------------------------------------------------------------- #
# Cross-layer: equals the Stage-1 HodgeLaplacian cycle flux
# --------------------------------------------------------------------------- #
class TestMatchesHodgeStageOneFlux(unittest.TestCase):

    def test_triangle_random_phase_matches_hodge(self):
        rng = np.random.default_rng(114)
        for _ in range(8):
            st = _triangle()
            _set_phases(st, {
                frozenset({0, 1}): float(rng.uniform(-math.pi, math.pi)),
                frozenset({1, 2}): float(rng.uniform(-math.pi, math.pi)),
                frozenset({2, 0}): float(rng.uniform(-math.pi, math.pi)),
            })
            r = _holonomy(st, [0, 1, 2])
            self.assertTrue(_close_mod_2pi(r.value, _hodge_flux(st, [0, 1, 2])))
            self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

    def test_testbed_both_cycles_match_hodge(self):
        # The b1 = 2 testbed has two independent cycles; each holonomy matches
        # the operator's Stage-1 flux for that signed-edge cycle.
        rng = np.random.default_rng(2024)
        st = _testbed()
        _set_phases(st, {
            frozenset({0, 1}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({1, 2}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({2, 3}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({3, 0}): float(rng.uniform(-math.pi, math.pi)),
            frozenset({0, 2}): float(rng.uniform(-math.pi, math.pi)),
        })
        for cycle in ([0, 1, 2], [0, 2, 3]):
            with self.subTest(cycle=cycle):
                r = _holonomy(st, cycle)
                self.assertTrue(_close_mod_2pi(r.value, _hodge_flux(st, cycle)))
                self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, cycle)))


# --------------------------------------------------------------------------- #
# Z2 case: phases in {0, pi} -> holonomy in {0, pi}, matching Stage-1 flux
# --------------------------------------------------------------------------- #
class TestZ2Holonomy(unittest.TestCase):

    def test_parity_lands_in_zero_or_pi(self):
        # k edges at phase pi; the bulk holonomy is pi for odd k, 0 for even k
        # (a pi edge contributes +-pi == pi mod 2*pi regardless of orientation).
        edges = [frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 0})]
        saw_zero = saw_pi = False
        for k in range(len(edges) + 1):
            with self.subTest(num_pi_edges=k):
                st = _triangle()
                _set_phases(st, {e: math.pi for e in edges[:k]})
                r = _holonomy(st, [0, 1, 2])
                expected = math.pi if (k % 2 == 1) else 0.0

                # Holonomy restricted to {0, pi}.
                self.assertTrue(
                    abs(r.value) < 1e-9 or abs(r.value - math.pi) < 1e-9,
                    f"Z2 holonomy {r.value} not in {{0, pi}}")
                self.assertTrue(_close_mod_2pi(r.value, expected))

                # Cross-layer: matches the Stage-1 flux there.
                self.assertTrue(_close_mod_2pi(r.value, _hodge_flux(st, [0, 1, 2])))
                self.assertTrue(_close_mod_2pi(r.value, _cycle_flux(st, [0, 1, 2])))

                saw_zero |= (expected == 0.0)
                saw_pi |= (expected == math.pi)
        self.assertTrue(saw_zero and saw_pi)  # both Z2 classes exercised


# --------------------------------------------------------------------------- #
# Degenerate / open cycles return an empty result
# --------------------------------------------------------------------------- #
class TestDegenerateCycles(unittest.TestCase):

    def test_too_few_vertices_is_empty(self):
        st = _triangle()
        _set_phases(st, {})
        by_id = _vertices_by_id(st)
        wl = tessera.WilsonLoop(st)
        self.assertEqual(wl.evaluate_u1_connection([]).loop_size, 0)
        self.assertEqual(wl.evaluate_u1_connection([by_id[0]]).loop_size, 0)

    def test_open_path_is_empty(self):
        # On the testbed, vertices 1 and 3 are not adjacent (no edge 1-3), so a
        # cycle through that pair is open and yields an empty result.
        st = _testbed()
        _set_phases(st, {})
        with self.assertRaises(KeyError):
            _edge(st, 1, 3)  # confirm the gap exists
        r = _holonomy(st, [0, 1, 3])
        self.assertEqual(r.loop_size, 0)
        self.assertEqual(r.value, 0.0)


# --------------------------------------------------------------------------- #
# The datum: the unreduced complex accumulation, and its derived views
# --------------------------------------------------------------------------- #
class TestConnectionAccumulation(unittest.TestCase):
    """`connection_accumulation` is the complete gauge-invariant datum; the
    holonomy, its modulus, the residual phase and the winding are derived from
    it. Reducing mod 2*pi at accumulation time -- what the code used to do --
    destroys the winding irrecoverably, so that is what these pin (#872)."""

    @staticmethod
    def _read(phases, cycle=(0, 1, 2)):
        st = _triangle()
        _set_phases(st, phases)
        by_id = _vertices_by_id(st)
        return tessera.WilsonLoop(st).evaluate_u1_connection(
            [by_id[c] for c in cycle])

    def test_both_components_accumulate(self):
        r = self._read({frozenset({0, 1}): complex(0.4, 0.2),
                        frozenset({1, 2}): complex(0.5, -0.1),
                        frozenset({2, 0}): complex(0.3, 0.05)})
        sigma = complex(r.connection_accumulation)
        self.assertAlmostEqual(sigma.real, 1.2, places=12)
        self.assertAlmostEqual(sigma.imag, 0.15, places=12)

    def test_the_holonomy_reconstructs_from_the_datum(self):
        r = self._read({frozenset({0, 1}): complex(0.4, 0.2),
                        frozenset({1, 2}): complex(0.5, -0.1)})
        sigma = complex(r.connection_accumulation)
        self.assertEqual(complex(r.holonomy()), cmath.exp(1j * sigma))
        self.assertEqual(r.holonomy_modulus(), math.exp(-sigma.imag))

    def test_winding_beyond_two_pi_is_recoverable(self):
        """The test the defect would have failed. `principalAngle` at
        accumulation time folded these into one another."""
        for turns in (0, 1, 2, -3):
            with self.subTest(turns=turns):
                per = (TWO_PI * turns + 0.25) / 3.0
                r = self._read({frozenset({0, 1}): complex(per, 0.0),
                                frozenset({1, 2}): complex(per, 0.0),
                                frozenset({2, 0}): complex(per, 0.0)})
                self.assertEqual(r.winding_number(), turns)
                self.assertAlmostEqual(r.residual_phase(), 0.25, places=9)
                # The full value is present, not the folded one.
                self.assertAlmostEqual(
                    complex(r.connection_accumulation).real,
                    TWO_PI * turns + 0.25, places=9)

    def test_a_real_connection_has_modulus_exactly_one(self):
        """The emergent cancellation, when it happens: a purely compact
        connection gives |H| = 1 exactly -- observed, not imposed."""
        r = self._read({frozenset({0, 1}): complex(0.4, 0.0),
                        frozenset({1, 2}): complex(0.5, 0.0)})
        self.assertEqual(complex(r.connection_accumulation).imag, 0.0)
        self.assertEqual(r.holonomy_modulus(), 1.0)

    def test_the_derived_angle_is_bit_identical_to_the_hand_oracle(self):
        """`value` is now derived from the complex datum. Re is linear over the
        sum, so it must be bitwise what accumulating Re alone produced."""
        for phases in ({frozenset({0, 1}): complex(0.4, 0.2),
                        frozenset({1, 2}): complex(-1.3, 5.0)},
                       {frozenset({0, 1}): complex(2.9, 0.0)},
                       {frozenset({2, 0}): complex(-0.75, -3.1)}):
            with self.subTest(phases=sorted(map(sorted, phases))):
                st = _triangle()
                _set_phases(st, phases)
                by_id = _vertices_by_id(st)
                r = tessera.WilsonLoop(st).evaluate_u1_connection(
                    [by_id[c] for c in (0, 1, 2)])
                self.assertEqual(complex(r.connection_accumulation).real,
                                 _cycle_flux(st, [0, 1, 2]))
                self.assertEqual(complex(r.value).real,
                                 _wrap(_cycle_flux(st, [0, 1, 2])))

    def test_a_gauge_transformation_moves_neither_component(self):
        """phi_uv -> phi_uv + chi_v - chi_u telescopes to zero around a closed
        loop, so the WHOLE complex sum is invariant. That the imaginary part is
        invariant too is what justifies reporting the modulus at all."""
        chi = {0: complex(0.3, -0.15),
               1: complex(-0.8, 0.4),
               2: complex(1.1, 0.9)}
        base = {frozenset({0, 1}): complex(0.4, 0.2),
                frozenset({1, 2}): complex(0.5, -0.1),
                frozenset({2, 0}): complex(0.3, 0.05)}
        before = self._read(base)

        st = _triangle()
        _set_phases(st, base)
        for e in st.get_edge_list().to_vector():
            u, v = e.get_source().get_id(), e.get_target().get_id()
            e.set_phase(complex(e.get_phase()) + chi[v] - chi[u])
        by_id = _vertices_by_id(st)
        after = tessera.WilsonLoop(st).evaluate_u1_connection(
            [by_id[c] for c in (0, 1, 2)])

        drift = abs(complex(after.connection_accumulation)
                    - complex(before.connection_accumulation))
        self.assertLess(drift, 1e-12, f"gauge drift {drift}")
        self.assertEqual(after.winding_number(), before.winding_number())
        self.assertAlmostEqual(after.holonomy_modulus(),
                               before.holonomy_modulus(), places=12)
        # Guard against a vacuous pass: the phases really did change.
        self.assertGreater(
            max(abs(complex(e.get_phase()) - base[frozenset(
                {e.get_source().get_id(), e.get_target().get_id()})])
                for e in st.get_edge_list().to_vector()), 0.5)

    def test_an_unread_cycle_leaves_the_datum_unmeasured(self):
        """NaN, never zero. A zero accumulation would read as a MEASURED
        trivial holonomy, which is exactly the confusion the modulus exists to
        avoid -- and it is what a default-constructed result would report if
        the field were zero-initialised."""
        st = _testbed()
        _set_phases(st, {})
        by_id = _vertices_by_id(st)
        wl = tessera.WilsonLoop(st)
        for cycle, why in (([], "empty"),
                           ([by_id[0]], "single vertex"),
                           ([by_id[0], by_id[1], by_id[3]], "open path")):
            with self.subTest(why=why):
                sigma = complex(
                    wl.evaluate_u1_connection(cycle).connection_accumulation)
                self.assertTrue(math.isnan(sigma.real), f"{why}: {sigma}")
                self.assertTrue(math.isnan(sigma.imag), f"{why}: {sigma}")


if __name__ == "__main__":
    unittest.main()
