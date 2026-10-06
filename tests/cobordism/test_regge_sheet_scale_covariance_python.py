# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The continued Regge sheets follow a dilation of the squared lengths (#1427).

Terms used below:

* a *dilation* by s > 0 multiplies every squared length by s and leaves the
  links as they are;
* the fixture is the boundary of the 4-simplex of
  ``test_joint_action_properties_python`` (five tetrahedra, squared lengths
  8 (1 + 0.05 a_e) + i y b_e times s), at y = 0.4 (the complex metric) and at
  y = 0;
* the *sheet sign* of a continued root is +1 when Re(conj(c) p) >= 0 for the
  continued value c and the principal value p at the geometry the mesh holds.

`JointAction` reads the sheet sign of the root product of every dihedral
angle, of every hinge content and of every top-cell volume. It reads it from
the unit phases c/|c| and p/|p|, so a sign does not depend on the size of c
and p. On the module of db73d0e1 it was read from the product conj(c) p itself,
of order s^4 for the dihedral root products, which underflows below s of about
1e-80: the Regge term divided by s^(1/2) at y = 0.4 read 45.844285 + 0.824354 i
there, against 73.304611 + 0.885506 i, and the 14 off-principal angles read 0.

The scales run from 1e-150 (the dihedral cofactors, of order s^2, are normal
doubles) to 1e100 (constructing the action walks the squared top-cell volumes,
of order s^3, which overflow above about 1e102); the volume sum, of order
s^(3/2), is read from 1e-100.

Measured on the module of
``~/scratch/v18code-2026-10-05/scale-free-thresholds/build-3``: the largest
relative departure over these scales is 3.9e-16 for the Regge term and 4.0e-16
for the volume sum, at y = 0.4 and at y = 0; ``AGREEMENT`` is 1e-13.
"""

import pytest

from tessera import cobordism as cob
from tests.cobordism import test_joint_action_properties_python as P

#: Every fifth decade from 1e-150 to 1e100.
SCALES = [10.0 ** k for k in range(-150, 101, 5)]

#: The scales at which the volume sum is read (its squared values, of order
#: s^3, are normal doubles).
VOLUME_SCALES = [s for s in SCALES if s >= 1e-100]

#: The relative agreement asserted between a read divided by its power of s and
#: its value at s = 1.
AGREEMENT = 1e-13

IMAGINARY = [0.4, 0.0]


def _action(scale, imaginary):
    spacetime = P._sphere(scale, imaginary)
    return cob.JointAction(
        spacetime, P._declaration(regge=1.0, start=P._squared(spacetime)))


_REFERENCE = {}


def _reference(imaginary):
    if imaginary not in _REFERENCE:
        action = _action(1.0, imaginary)
        _REFERENCE[imaginary] = (complex(action.regge_term()),
                                 action.regge_off_principal_angles(),
                                 complex(action.volume_sum()))
    return _REFERENCE[imaginary]


@pytest.mark.parametrize("imaginary", IMAGINARY, ids=lambda y: f"y={y}")
@pytest.mark.parametrize("scale", SCALES, ids=lambda s: f"s={s:.0e}")
def test_the_regge_term_and_its_sheets_follow_the_dilation(scale, imaginary):
    """The Regge term divided by s^(1/2) agrees with its value at s = 1 to
    ``AGREEMENT``, and as many angles are off the principal sheet."""
    regge, off_principal, _ = _reference(imaginary)
    action = _action(scale, imaginary)
    value = complex(action.regge_term()) / scale ** 0.5
    assert abs(value - regge) <= AGREEMENT * abs(regge), (
        f"{value} against {regge} at s = {scale:.0e}")
    assert action.regge_off_principal_angles() == off_principal


def test_the_complex_metric_carries_off_principal_angles():
    """At y = 0.4 the continued walk leaves 14 angles off the principal
    sheet, so the sheet signs above are exercised."""
    assert _reference(0.4)[1] == 14


@pytest.mark.parametrize("imaginary", IMAGINARY, ids=lambda y: f"y={y}")
@pytest.mark.parametrize("scale", VOLUME_SCALES, ids=lambda s: f"s={s:.0e}")
def test_the_volume_sum_follows_the_dilation(scale, imaginary):
    """The sum of the continued top-cell volumes divided by s^(3/2) agrees
    with its value at s = 1 to ``AGREEMENT``."""
    _, _, volume = _reference(imaginary)
    value = complex(_action(scale, imaginary).volume_sum()) / scale ** 1.5
    assert abs(value - volume) <= AGREEMENT * abs(volume), (
        f"{value} against {volume} at s = {scale:.0e}")
