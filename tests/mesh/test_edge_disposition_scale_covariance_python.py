# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The causal disposition of an edge follows a dilation of the squared lengths
(#1427).

Terms used below:

* a *dilation* by s > 0 multiplies every squared length by s, so a length
  l = |l| e^{i a} becomes s^(1/2) l and keeps its argument;
* the *disposition* is which of `Edge.isSpacelike`, `isTimelike`, `isNull`,
  `isMixed` and `isDegenerate` holds.

The disposition is read from arg(l^2), which a dilation leaves unchanged, and
an edge is degenerate (absent) exactly when l = 0: an edge with no extent has
no argument. So the disposition of every nonzero edge is the same at every
scale at which its squared length is a normal double, s from 1e-300 to 1e300
here. On the module of db73d0e1 every edge with |l| <= 1e-12 (s <= 1e-24) read
degenerate, whatever its argument, and the causal weight map of
`PersistentModularity` was refused on such a complex.
"""

import cmath
import math

import pytest

import tessera
from tessera import Edge, Vertex

PM = tessera.observables.PersistentModularity

#: Every decade from 1e-30 to 1e30, and every tenth decade out to 1e-300 and
#: 1e300, where the squared lengths are still normal doubles.
SCALES = sorted({10.0 ** k for k in range(-30, 31)} |
                {10.0 ** k for k in range(-300, 301, 10)})

#: A unit length of each disposition: arg(l^2) = 0, pi, pi/2 and 0.6.
UNIT_LENGTHS = {
    "spacelike": 1.0 + 0j,
    "timelike": 1j,
    "lightlike": cmath.exp(0.25j * math.pi),
    "mixed": cmath.exp(0.3j),
}


def _edge(length):
    return Edge(Vertex(1, [0.0, 0.0, 0.0, 0.0]),
                Vertex(2, [0.0, 0.0, 0.0, 1.0]), complex(length))


def _disposition(edge):
    return (edge.isSpacelike(), edge.isTimelike(), edge.isNull(),
            edge.isMixed(), edge.isDegenerate())


def _scale_id(scale):
    return f"s={scale:.0e}"


@pytest.mark.parametrize("kind", sorted(UNIT_LENGTHS))
@pytest.mark.parametrize("scale", SCALES, ids=_scale_id)
def test_the_disposition_of_an_edge_follows_the_dilation(kind, scale):
    """An edge of length s^(1/2) l has the disposition of the edge of length
    l, and is not degenerate."""
    unit = UNIT_LENGTHS[kind]
    expected = _disposition(_edge(unit))
    assert sum(expected) == 1 and not expected[4]
    edge = _edge(math.sqrt(scale) * unit)
    assert abs(edge.getLength()) > 0.0
    assert _disposition(edge) == expected, (
        f"{kind} edge of modulus {abs(edge.getLength()):.1e} reads "
        f"{_disposition(edge)} against {expected}")


def test_only_a_zero_length_is_degenerate():
    """l = 0 is degenerate; the smallest positive double is not."""
    assert _disposition(_edge(0.0)) == (False, False, False, False, True)
    assert not _edge(5e-324).isDegenerate()


# ------------------------------------------- the causal census of a complex


def _causal_k6(scale):
    """K6 with spacelike edges inside the triples (0, 1, 2) and (3, 4, 5) and
    timelike edges between them, every squared length of modulus ``scale``."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.createVertex(i) for i in range(6)]
    for a in range(6):
        for b in range(a + 1, 6):
            st.createSimplex([verts[a], verts[b]])
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        same = (a < 3) == (b < 3)
        e.setLength(complex(math.sqrt(scale) * (1.0 if same else 1j)))
        e.setPhase(0.0)
    return st


def _census(read):
    return (read.spacelike, read.timelike, read.lightlike, read.mixed,
            read.degenerate, read.available, read.reason)


@pytest.mark.parametrize("scale", SCALES, ids=_scale_id)
def test_the_causal_census_of_a_complex_follows_the_dilation(scale):
    """`PersistentModularity.causalWeightAvailability` reads the same census
    (6 spacelike, 9 timelike edges, the map available) at every scale."""
    reference = _census(PM.causalWeightAvailability(_causal_k6(1.0)))
    assert reference == (6, 9, 0, 0, 0, True, "")
    assert _census(PM.causalWeightAvailability(_causal_k6(scale))) == reference
