# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Every Gram and Cayley-Menger read follows a dilation of the squared lengths
(#1425).

Terms used below:

* a *dilation* by s > 0 multiplies every squared length by s and leaves the
  links as they are;
* a read *follows* the dilation when, divided by its power of s, it agrees with
  its value at s = 1 to the relative ``AGREEMENT``. On a d-simplex the powers
  are: the content s^(d/2), the Gram determinant s^d, a Cayley-Menger cofactor
  C_ij of the dihedral angle s^(d-1), the dihedral angle s^0, the squared
  circumradius s^1 and the content gradient d V / d l^2 s^(d/2 - 1). On a
  closed 3-complex the Regge term sum_h |h| eps_h goes as s^(1/2) and the
  volume sum as s^(3/2).

`Simplex::determinant` eliminates with partial pivoting and stops only on a
column whose every candidate is exactly zero, and the reads below test a Gram
or Cayley-Menger determinant, or the root product of two cofactors, against
exact zero only, so none of them depends on the unit of length. The fixtures
are a regular tetrahedron with every squared length s, and
the boundary of the 4-simplex of ``test_joint_action_properties_python`` (five
tetrahedra, squared lengths 8 (1 + 0.05 a_e) + i y b_e times s, links off the
unit circle) at y = 0.4 (the complex metric) and at y = 0 (the real one of the
#1417 probe).

Measured on the module of
``~/scratch/v18code-2026-10-05/determinant-without-absolute-cut/build-2``: the
largest relative departure over every read and every decade from s = 1e-30 to
1e30 is 3.3e-15 (the content gradients of the boundary of the 4-simplex); the
Regge term divided by s^(1/2) reads 73.304611 + 0.885506 i at y = 0.4 and
73.218453 at y = 0 at every scale. ``AGREEMENT`` is 1e-13, thirty times that.

`Simplex::assertSpacelikeAdmissible` (#1427) reads each leading minor of the
Gram matrix relative to vertex 0 divided by the product of its diagonal moduli
(the squared lengths of the edges from vertex 0 that span it), a ratio a
dilation leaves unchanged, and an edge is degenerate only at l = 0, so the
check is made, and decides the same way, at every scale from 1e-300 to 1e300.
"""

import cmath
import math

import pytest

import tessera as T
from tessera import cobordism as cob
from tests.cobordism import test_joint_action_properties_python as P

#: Every decade of the dilation from 1e-30 to 1e30.
SCALES = [10.0 ** k for k in range(-30, 31)]

#: The relative agreement asserted between a read divided by its power of s and
#: its value at s = 1.
AGREEMENT = 1e-13

#: The dimension of every top cell of the fixtures.
DIMENSION = 3


# -------------------------------------------------------------------- fixtures


def _tetrahedron(scale):
    """A regular tetrahedron with every squared length ``scale``, and its
    edge (2, 3) as the hinge of the dihedral reads."""
    spacetime = T.Spacetime()
    vertices = {i: spacetime.createVertex(i) for i in range(4)}
    edges = {}
    for a in range(4):
        for b in range(a + 1, 4):
            edges[(a, b)] = spacetime.createEdge(
                vertices[a], vertices[b], cmath.sqrt(complex(scale)))
    cell, _ = spacetime.createSimplex([vertices[i] for i in range(4)],
                                      list(edges.values()))
    hinge, _ = spacetime.createSimplex([vertices[2], vertices[3]],
                                       [edges[(2, 3)]])
    return spacetime, [(cell, hinge)]


def _sphere(scale, imaginary):
    """The boundary of the 4-simplex at ``scale``, each top cell with an edge
    of it as the hinge of the dihedral reads."""
    spacetime = P._sphere(scale, imaginary)
    cells = []
    for cell in spacetime.getSimplices():
        if len(cell.getVertices()) == DIMENSION + 1:
            cells.append((cell, cell.getFacets()[0].getFacets()[0]))
    return spacetime, cells


def _cell_reads(cells, scale):
    """Every read of every cell, divided by its power of ``scale``."""
    d = DIMENSION
    reads = {}
    for k, (cell, hinge) in enumerate(cells):
        content = complex(cell.volume())
        reads[f"content[{k}]"] = content / scale ** (d / 2)
        reads[f"Gram determinant[{k}]"] = (
            (math.factorial(d) * content) ** 2 / scale ** d)
        ok, cij, cii, cjj = cell.dihedralCofactors(hinge)
        reads[f"cofactors usable[{k}]"] = complex(float(ok))
        for name, value in (("C_ij", cij), ("C_ii", cii), ("C_jj", cjj)):
            reads[f"{name}[{k}]"] = complex(value) / scale ** (d - 1)
        reads[f"dihedral angle[{k}]"] = complex(cell.dihedralAngle(hinge))
        reads[f"squared circumradius[{k}]"] = (
            complex(cell.circumradiusSquared()) / scale)
        for edge, value in sorted(cell.volumeGradient().items()):
            reads[f"content gradient {edge}[{k}]"] = (
                complex(value) / scale ** (d / 2 - 1))
    return reads


def _reads(fixture, scale):
    if fixture == "regular tetrahedron":
        spacetime, cells = _tetrahedron(scale)
        return _cell_reads(cells, scale)
    imaginary = 0.4 if fixture == "boundary of the 4-simplex, y = 0.4" else 0.0
    spacetime, cells = _sphere(scale, imaginary)
    reads = _cell_reads(cells, scale)
    action = cob.JointAction(
        spacetime, P._declaration(regge=1.0, start=P._squared(spacetime)))
    reads["Regge term"] = complex(action.regge_term()) / scale ** 0.5
    reads["volume sum"] = complex(action.volume_sum()) / scale ** 1.5
    return reads


FIXTURES = ["regular tetrahedron", "boundary of the 4-simplex, y = 0.4",
            "boundary of the 4-simplex, y = 0"]

_REFERENCE = {}


def _reference(fixture):
    if fixture not in _REFERENCE:
        _REFERENCE[fixture] = _reads(fixture, 1.0)
    return _REFERENCE[fixture]


def _departures(reference, reads):
    """The relative departure of every read from its reference; a read that
    is missing departs by infinity."""
    out = {}
    for name, value in reference.items():
        if name not in reads:
            out[name] = math.inf
        elif value == 0:
            out[name] = abs(reads[name])
        else:
            out[name] = abs(reads[name] - value) / abs(value)
    return out


# ----------------------------------------------------------------------- tests


@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("scale", SCALES, ids=lambda s: f"s={s:.0e}")
def test_every_read_follows_the_dilation(fixture, scale):
    """The determinants, the contents, the cofactors and the angle of every
    top cell, and on the closed complex the Regge term and the volume sum,
    divided by their powers of s agree with their values at s = 1 to
    ``AGREEMENT``, with the same set of reads."""
    reference = _reference(fixture)
    reads = _reads(fixture, scale)
    assert sorted(reads) == sorted(reference)
    departures = _departures(reference, reads)
    worst = max(departures, key=departures.get)
    assert departures[worst] <= AGREEMENT, (
        f"{worst} departs by {departures[worst]:.3e} at s = {scale:.0e}: "
        f"{reads[worst]} against {reference[worst]}")


def test_the_regge_term_has_one_value_over_sixty_decades():
    """The Regge term of the complex-metric boundary of the 4-simplex divided
    by s^(1/2) is one number from s = 1e-30 to 1e30, and the continued
    sheets are followed at every scale."""
    values = [_reads("boundary of the 4-simplex, y = 0.4", s)["Regge term"]
              for s in SCALES]
    reference = _reference("boundary of the 4-simplex, y = 0.4")["Regge term"]
    spread = max(abs(v - reference) for v in values) / abs(reference)
    assert spread <= AGREEMENT


def test_a_gram_determinant_below_1e_300_is_read():
    """At s = 1e-101 the Gram determinant of the regular tetrahedron is about
    5e-304, a normal double: the content gradient and every other read divided
    by its power of s still agree with their values at s = 1."""
    scale = 1e-101
    reference = _reference("regular tetrahedron")
    reads = _reads("regular tetrahedron", scale)
    assert abs(reads["Gram determinant[0]"] * scale ** DIMENSION) < 1e-300
    departures = _departures(reference, reads)
    worst = max(departures, key=departures.get)
    assert departures[worst] <= AGREEMENT, (
        f"{worst} departs by {departures[worst]:.3e}")


def test_a_cofactor_root_product_below_1e_300_gives_the_angle():
    """At s = 1e-153 the Cayley-Menger cofactors of the regular tetrahedron
    are C_ij = 1e-306 and C_ii = C_jj = -3e-306, so the product of the roots
    of C_ii and C_jj is 3e-306, below 1e-300, and all are normal doubles:
    the cofactors divided by s^2 and the dihedral angle agree with their
    values at s = 1. (Its Gram determinant, of order 1e-459, is below the
    smallest double, so the content is not read here.)"""
    scale = 1e-153
    spacetime, cells = _tetrahedron(scale)
    cell, hinge = cells[0]
    ok, cij, cii, cjj = cell.dihedralCofactors(hinge)
    assert ok
    assert cii != 0 and cjj != 0
    assert math.sqrt(abs(cii)) * math.sqrt(abs(cjj)) < 1e-300
    reference = _reference("regular tetrahedron")
    for name, value in (("C_ij", cij), ("C_ii", cii), ("C_jj", cjj)):
        expected = reference[f"{name}[0]"]
        assert abs(complex(value) / scale ** 2 - expected) <= (
            AGREEMENT * abs(expected))
    expected = reference["dihedral angle[0]"]
    assert abs(complex(cell.dihedralAngle(hinge)) - expected) <= (
        AGREEMENT * abs(expected))


# ------------------------------------------------ spacelike admissibility

#: Every decade from 1e-30 to 1e30, and every tenth decade out to 1e-300 and
#: 1e300, where the squared lengths are still normal doubles.
WIDE_SCALES = sorted(set(SCALES) | {10.0 ** k for k in range(-300, 301, 10)})


def _tetrahedron_with(squared, scale):
    """A tetrahedron whose edge (a, b) has the squared length
    ``squared[(a, b)]`` times ``scale``."""
    spacetime = T.Spacetime()
    vertices = {i: spacetime.createVertex(i) for i in range(4)}
    edges = [spacetime.createEdge(vertices[a], vertices[b],
                                  cmath.sqrt(complex(scale * value)))
             for (a, b), value in sorted(squared.items())]
    cell, _ = spacetime.createSimplex([vertices[i] for i in range(4)], edges)
    return spacetime, cell


#: A regular tetrahedron: the leading minors of its Gram matrix relative to
#: vertex 0, each divided by the product of its diagonal moduli, are 1, 3/4
#: and 1/2.
REGULAR = {(a, b): 1.0 for a in range(4) for b in range(a + 1, 4)}

#: The face (0, 2, 3) has sides 1, 1 and 5^(1/2) > 1 + 1: the third leading
#: minor of the Gram matrix is -5/2.
VIOLATING = {**REGULAR, (2, 3): 5.0}


def test_a_small_regular_tetrahedron_is_spacelike_admissible():
    """A regular tetrahedron of squared length 1e-16 is positive definite and
    passes the spacelike admissibility check, as it does at s = 1."""
    spacetime, cells = _tetrahedron(1e-16)
    cells[0][0].assertSpacelikeAdmissible()


@pytest.mark.parametrize("scale", WIDE_SCALES, ids=lambda s: f"s={s:.0e}")
def test_spacelike_admissibility_follows_the_dilation(scale):
    """At every scale the regular tetrahedron passes at the default tol and at
    tol = 0.49 and is refused at tol = 0.51 (its smallest relative leading
    minor is 1/2), and the tetrahedron that violates a triangle inequality is
    refused."""
    spacetime, regular = _tetrahedron_with(REGULAR, scale)
    regular.assertSpacelikeAdmissible()
    regular.assertSpacelikeAdmissible(0.49)
    with pytest.raises(RuntimeError, match="leading minor 3"):
        regular.assertSpacelikeAdmissible(0.51)
    spacetime, violating = _tetrahedron_with(VIOLATING, scale)
    with pytest.raises(RuntimeError, match="not positive-definite"):
        violating.assertSpacelikeAdmissible()
