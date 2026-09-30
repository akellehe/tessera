# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Simplex geometry of the meshes the recursion driver builds.

Both drivers build their complexes with `Spacetime.fromVertexTuples` and write
complex edge lengths (`recursion.build_level`, `baryon_poles.build_host`); the
primal Regge term then reads each top simplex's content, dihedral angles and
deficits (`mesh::Simplex`). The expected values are the closed forms of the
regular unit d-simplex:

* content V_d = sqrt(d + 1) / (d! 2^{d/2}) = sqrt(det G) / d!, with Gram
  matrix G_ij = (1/2)(s_0i + s_0j - s_ij);
* dihedral angle arccos(1/d) at every hinge;
* Cayley-Menger determinant (-1)^{d+1} 2^d (d!)^2 V_d^2,

and, for signature, a triangle with one timelike leg whose Gram determinant
is -1, so its content is imaginary, sqrt(-1)/2. The refusals of a degenerate
simplex are named. On the continued sheet the Regge term is continuous across
an infinitesimal imaginary part of a squared length where the principal
(sheet-blind) angle jumps to its supplement, and on real Euclidean input the
two agree (`JointAction`, `ReggeBranch`; whitepaper v17 ledger: complex Regge
roots carry a Riemann-sheet label).
"""
import cmath
import math

import numpy as np
import pytest

import tessera as T
from tessera import cobordism as cob


def _regular(dimension, squared=1.0):
    spacetime = T.Spacetime.fromVertexTuples(
        dimension, [list(range(dimension + 1))], 1.0, 0.0)
    for edge in spacetime.getEdgeList().toVector():
        edge.setLength(cmath.sqrt(complex(squared)))
    return spacetime, spacetime.getTopSimplices()[0]


def _hinges(simplex):
    seen, out = set(), []
    for facet in simplex.getFacets():
        for hinge in facet.getFacets():
            key = tuple(sorted(v.getId() for v in hinge.getVertices()))
            if key not in seen:
                seen.add(key)
                out.append(hinge)
    return out


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_the_regular_unit_simplex_content(d):
    """V_d = sqrt(d + 1) / (d! 2^{d/2}): sqrt 3 / 4, 1 / (6 sqrt 2),
    sqrt 5 / 96 and sqrt 6 / (960 sqrt 2), and V_d = sqrt(det G) / d! from
    the simplex's own Gram matrix, whose entries are 1 on the diagonal and
    1/2 off it."""
    _, simplex = _regular(d)
    expected = math.sqrt(d + 1) / (math.factorial(d) * 2 ** (d / 2))
    assert complex(simplex.volume()) == pytest.approx(expected, rel=1e-14)
    gram = np.asarray(simplex.gramMatrix()).reshape(d, d)
    np.testing.assert_allclose(gram, 0.5 * (np.eye(d) + np.ones((d, d))),
                               atol=1e-15)
    assert cmath.sqrt(np.linalg.det(gram)) / math.factorial(d) == \
        pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_the_regular_dihedral_angle_is_arccos_one_over_d(d):
    """Every hinge of the regular unit d-simplex carries the interior
    dihedral angle arccos(1/d): pi/3, arccos(1/3), arccos(1/4), arccos(1/5),
    real on the principal branch."""
    _, simplex = _regular(d)
    hinges = _hinges(simplex)
    assert len(hinges) == math.comb(d + 1, d - 1)
    for hinge in hinges:
        angle = complex(simplex.dihedralAngle(hinge))
        assert angle.real == pytest.approx(math.acos(1.0 / d), abs=1e-14)
        assert abs(angle.imag) < 1e-15


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_the_cayley_menger_determinant(d):
    """det CM = (-1)^{d+1} 2^d (d!)^2 V_d^2: -3, 4, -5 and 6 for the regular
    unit simplices of dimension 2 to 5."""
    _, simplex = _regular(d)
    cm = np.asarray(simplex.cayleyMengerMatrix()).reshape(d + 2, d + 2)
    volume = math.sqrt(d + 1) / (math.factorial(d) * 2 ** (d / 2))
    expected = (-1) ** (d + 1) * 2 ** d * math.factorial(d) ** 2 * volume ** 2
    assert np.linalg.det(cm) == pytest.approx(expected, rel=1e-13)
    assert expected == pytest.approx((-1) ** (d + 1) * (d + 1), rel=1e-13)


def test_the_regular_tetrahedron_of_the_run_scales_as_squared_length():
    """The run's host cells are regular with squared length 8: the content
    scales as the cube of the length, 8^{3/2} V_3 = 8^{3/2} / (6 sqrt 2) =
    8 / 3, and the dihedral angle is still arccos(1/3)."""
    _, simplex = _regular(3, 8.0)
    assert complex(simplex.volume()) == pytest.approx(
        8.0 ** 1.5 / (6.0 * math.sqrt(2.0)), rel=1e-14)
    for hinge in _hinges(simplex):
        assert complex(simplex.dihedralAngle(hinge)).real == pytest.approx(
            math.acos(1.0 / 3.0), abs=1e-14)


def test_a_timelike_leg_gives_an_imaginary_content():
    """A triangle with s_01 = 1, s_02 = -1 (timelike), s_12 = 0 has
    G = diag(1, -1), det G = -1, so its content sqrt(det G) / 2 is imaginary
    of modulus 1/2 and its square is -1/4."""
    spacetime = T.Spacetime.fromVertexTuples(2, [[0, 1, 2]], 1.0, 0.0)
    squared = {(0, 1): 1.0, (0, 2): -1.0, (1, 2): 0.0}
    for edge in spacetime.getEdgeList().toVector():
        a, b = sorted((int(edge.getSource().getId()),
                       int(edge.getTarget().getId())))
        edge.setLength(cmath.sqrt(complex(squared[(a, b)])))
    simplex = spacetime.getTopSimplices()[0]
    content = complex(simplex.volume())
    assert abs(content.real) < 1e-15
    assert abs(content.imag) == pytest.approx(0.5, rel=1e-14)
    assert content ** 2 == pytest.approx(-0.25, abs=1e-15)


def test_a_degenerate_simplex_is_refused_by_name():
    """A flat tetrahedron (the fourth vertex in the plane of the first three:
    squared lengths of the unit square's corners with its diagonals) has zero
    content; its Hodge star is refused as 'primal volume is zero (degenerate
    simplex)' and its spacelike admissibility as 'inadmissible spacelike
    simplex'."""
    spacetime = T.Spacetime.fromVertexTuples(3, [[0, 1, 2, 3]], 1.0, 0.0)
    corners = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
    for edge in spacetime.getEdgeList().toVector():
        a, b = int(edge.getSource().getId()), int(edge.getTarget().getId())
        (xa, ya), (xb, yb) = corners[a], corners[b]
        edge.setLength(cmath.sqrt(complex((xa - xb) ** 2 + (ya - yb) ** 2)))
    simplex = spacetime.getTopSimplices()[0]
    assert abs(complex(simplex.volume())) < 1e-15
    with pytest.raises(RuntimeError, match="primal volume is zero "
                                           "\\(degenerate simplex\\)"):
        simplex.hodgeStar()
    with pytest.raises(RuntimeError, match="inadmissible spacelike simplex"):
        simplex.assertSpacelikeAdmissible()


def _regge(spacetime, branch):
    declaration = cob.JointActionDeclaration()
    declaration.carrier_degree = 1
    declaration.metric_source = cob.HodgeMetricSource.WhitneyPencil
    declaration.gravitational_weight = 1.0
    declaration.regge_form = cob.ReggeForm.Primal
    declaration.regge_hinges = cob.ReggeHinges.All
    declaration.regge_branch = branch
    declaration.stiffness_weight = 0.0
    declaration.reference_lengths = [
        complex(e.getLength()) for e in spacetime.getEdgeList().toVector()]
    declaration.holonomy_weight = 0.0
    declaration.matter_weight = 0.0
    declaration.regge_start_squared_lengths = [1.0 + 0j] * 6
    return complex(cob.JointAction(spacetime, declaration).regge_term())


def test_the_continued_sheet_is_continuous_where_the_principal_one_jumps():
    """The regular unit tetrahedron with one squared length 1 +- 1e-10 i: on
    the continued sheet (continued from the real Euclidean reference) the
    Regge term moves by O(1e-10) across the real axis, while on the principal
    sheet the two sides differ by O(1), the supplement jump; at the real point
    both sheets give 6 (2 pi - arccos(1/3))."""
    exact = 6 * (2 * math.pi - math.acos(1.0 / 3.0))
    values = {}
    for sign in (1, -1):
        spacetime, _ = _regular(3)
        first = spacetime.getEdgeList().toVector()[0]
        first.setLength(cmath.sqrt(1.0 + sign * 1e-10j))
        values[sign] = (_regge(spacetime, cob.ReggeBranch.Continued),
                        _regge(spacetime, cob.ReggeBranch.Principal))
    continued = [values[s][0] for s in (1, -1)]
    principal = [values[s][1] for s in (1, -1)]
    assert abs(continued[0] - continued[1]) < 1e-8
    assert abs(continued[0] - exact) < 1e-8
    assert abs(principal[0] - principal[1]) > 0.1
    real, _ = _regular(3)
    assert _regge(real, cob.ReggeBranch.Continued) == pytest.approx(
        exact, abs=1e-12)
    assert _regge(real, cob.ReggeBranch.Principal) == pytest.approx(
        exact, abs=1e-12)
