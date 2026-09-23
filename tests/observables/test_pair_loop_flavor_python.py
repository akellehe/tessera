# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The pair-loop dual-basis flavor read's per-hole quantity (#1203).

The quantity the read computes per hole is
``I_h = sum_{c in boundary of h} W_c |psi_c|^2``: the squared Hodge norm of the
carried representative over the hole's boundary facets, with ``W_c`` the Hodge
metric weight of the cell. It is a norm of the representative and nothing else.
It was called a Dirac-Kaehler charge, which named an operator this tree does
not have: Section 12.1 of the whitepaper separates the occupation exterior
algebra, whose degree is occupation number, from the inhomogeneous cochain
space a Kaehler-Dirac field lives on, and no operator of the second kind exists
here.

The odd-one-out rule over the three pair loops is arithmetic on those
intensities and is tested against hand-computed values."""
import pytest

from tessera import observables as obs


class TestTheReadNamesWhatItComputes:
    def test_the_joint_read_reports_boundary_intensities(self):
        for field in ("hole_intensity", "loop_intensity"):
            assert hasattr(obs.PairLoopJointRead, field)

    def test_no_field_is_called_a_charge(self):
        """The retired names are gone, not aliased: a reader who asks for the
        old name gets an error rather than a quantity under a name that claims
        an operator the code does not have."""
        for retired in ("q", "loop_q", "charge", "loop_charge"):
            assert not hasattr(obs.PairLoopJointRead, retired)


class TestTheOddOneOut:
    """(odd loop index, rho): the loop whose boundary intensity sits farthest
    from the mean of the other two, and rho = |spread of the other two| /
    |that separation|."""

    def test_two_together_and_one_apart(self):
        # separations are |1 - 1.5| = 0.5, |1 - 1.5| = 0.5 and |2 - 1| = 1.
        odd, rho = obs.PairLoopFlavor.odd_one_out([1.0, 1.0, 2.0])
        assert odd == 2
        assert rho == pytest.approx(0.0)
        assert rho < obs.PairLoopFlavor.RHO_MAX

    def test_the_pair_that_is_not_tight_does_not_read_as_two_to_one(self):
        # separations are |0 - 1.75| = 1.75, |1 - 1.25| = 0.25 and
        # |2.5 - 0.5| = 2; the odd one is 2.5 and the other two are 1 apart
        # against a separation of 2.
        odd, rho = obs.PairLoopFlavor.odd_one_out([0.0, 1.0, 2.5])
        assert odd == 2
        assert rho == pytest.approx(0.5)
        assert not rho < obs.PairLoopFlavor.RHO_MAX

    def test_the_complement_hole_of_a_pair_loop(self):
        assert obs.PairLoopFlavor.complement_hole((0, 1)) == 2
        assert obs.PairLoopFlavor.complement_hole((0, 2)) == 1
        assert obs.PairLoopFlavor.complement_hole((1, 2)) == 0
