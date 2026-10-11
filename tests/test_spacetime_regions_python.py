# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Spacetime's region, hole, Betti and edge-geometry reads.

These were static helpers of MultiCobordism (``betti``, ``emergent_holes``,
``boundary_facets``, ``subcomplexWithinVertexSet``) and moved onto the class
whose cells they read (https://github.com/akellehe/tessera/issues/1487).

``get_region`` is built with ``from_simplices``, so a region carries its
parent's edge lengths and phases without a second copy. The tests below pin
that, since the block reads used to copy the geometry again on top.
"""

import cmath

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

# The boundary of a tetrahedron: a 2-sphere.
_SPHERE = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
# The same sphere with the face (0, 1, 2) removed: a disk whose boundary is
# that face's three edges.
_DISK = [[0, 1, 3], [0, 2, 3], [1, 2, 3]]


def _edges_by_pair(st):
    return {(min(e.get_source().get_id(), e.get_target().get_id()),
             max(e.get_source().get_id(), e.get_target().get_id())): e
            for e in st.get_edge_list().to_vector()}


def _ascending_phase(edge):
    """The phase on the low-to-high orientation."""
    phase = edge.get_phase()
    return phase if edge.get_source().get_id() < edge.get_target().get_id() else 0 - phase


def _with_distinct_geometry(st):
    for k, (pair, edge) in enumerate(sorted(_edges_by_pair(st).items())):
        edge.set_length(cmath.sqrt(complex(2.0 + k, 0.0)))
        edge.set_phase(complex(0.1 * (k + 1), 0.0))
    return st


def test_betti_numbers_of_a_sphere_and_a_disk():
    assert list(tessera.Spacetime.from_vertex_tuples(2, _SPHERE).get_betti_numbers()) == [1, 0, 1]
    assert list(tessera.Spacetime.from_vertex_tuples(2, _DISK).get_betti_numbers()) == [1, 0, 0]


def test_the_disk_has_one_hole_and_the_sphere_none():
    disk = tessera.Spacetime.from_vertex_tuples(2, _DISK)
    assert [list(h) for h in disk.get_holes(1)] == [[0, 1, 2]]
    assert list(tessera.Spacetime.from_vertex_tuples(2, _SPHERE).get_holes(1)) == []
    # The boundary's facets are edges, so no hole of another degree exists.
    assert list(disk.get_holes(0)) == []


def test_a_region_is_the_cells_inside_the_vertex_set():
    sphere = tessera.Spacetime.from_vertex_tuples(2, _SPHERE)
    region = sphere.get_region({0, 1, 3})
    assert [sorted(v.get_id() for v in c.get_vertices()) for c in region.get_top_simplices()] == [[0, 1, 3]]
    assert sphere.get_region({0, 1}) is None


def test_a_region_carries_its_parents_lengths_and_phases():
    sphere = _with_distinct_geometry(tessera.Spacetime.from_vertex_tuples(2, _SPHERE))
    parent = _edges_by_pair(sphere)
    region = sphere.get_region({0, 1, 2, 3})
    for pair, edge in _edges_by_pair(region).items():
        assert edge.get_length() == pytest.approx(parent[pair].get_length(), rel=1e-15), pair
        assert _ascending_phase(edge) == pytest.approx(_ascending_phase(parent[pair]), rel=1e-15), pair


def test_copy_edge_geometry_from_matches_by_vertex_pair():
    sphere = _with_distinct_geometry(tessera.Spacetime.from_vertex_tuples(2, _SPHERE))
    face = tessera.Spacetime.from_vertex_tuples(2, [[0, 1, 2]])
    assert face.copy_edge_geometry_from(sphere)
    parent = _edges_by_pair(sphere)
    for pair, edge in _edges_by_pair(face).items():
        assert edge.get_length() == pytest.approx(parent[pair].get_length(), rel=1e-15), pair
        assert _ascending_phase(edge) == pytest.approx(_ascending_phase(parent[pair]), rel=1e-15), pair
    # An edge the source lacks: the copy reports it.
    stranger = tessera.Spacetime.from_vertex_tuples(2, [[0, 1, 7]])
    assert not stranger.copy_edge_geometry_from(sphere)
