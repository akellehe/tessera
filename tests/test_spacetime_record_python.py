# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Spacetime.record, Spacetime.from_record and Spacetime.relabeled.

One record of a complex's cells, edge geometry and vertex times replaces
``observables.LiveComplex`` and ``MultiCobordism``'s private snapshot
(https://github.com/akellehe/tessera/issues/1489). ``LiveComplex`` restored
squared lengths only, so its relabeled copy lost every link phase; the tests
below pin that the phases now carry across, on each pair's own orientation.
"""

import cmath

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

_SPHERE = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]


def _edges_by_pair(st):
    return {(min(e.get_source().get_id(), e.get_target().get_id()),
             max(e.get_source().get_id(), e.get_target().get_id())): e
            for e in st.get_edge_list().to_vector()}


def _ascending_phase(edge):
    phase = edge.get_phase()
    return phase if edge.get_source().get_id() < edge.get_target().get_id() else 0 - phase


def _sphere_with_geometry():
    st = tessera.Spacetime.from_vertex_tuples(2, _SPHERE)
    for k, (pair, edge) in enumerate(sorted(_edges_by_pair(st).items())):
        edge.set_length(cmath.sqrt(complex(2.0 + k, 0.5 * k)))
        edge.set_phase(complex(0.1 * (k + 1), 0.0))
    return st


def test_a_record_round_trips_lengths_phases_and_times():
    st = _sphere_with_geometry()
    record = st.record()
    assert record.dimensions == 2
    assert sorted(sorted(c) for c in record.cells) == sorted(sorted(c) for c in _SPHERE)
    rebuilt = tessera.Spacetime.from_record(record)
    original = _edges_by_pair(st)
    for pair, edge in _edges_by_pair(rebuilt).items():
        assert edge.get_length() == original[pair].get_length(), pair
        assert _ascending_phase(edge) == _ascending_phase(original[pair]), pair


def test_from_record_refuses_a_partial_record_unless_asked_to_default():
    record = _sphere_with_geometry().record()
    edges = dict(record.edges)
    del edges[(0, 1)]
    record.edges = edges
    with pytest.raises(IndexError):
        tessera.Spacetime.from_record(record)
    rebuilt = tessera.Spacetime.from_record(
        record, tessera.Spacetime.MissingEdgeGeometry.KEEP_DEFAULT)
    assert _edges_by_pair(rebuilt)[(0, 1)].get_phase() == 0


def test_a_relabeled_copy_carries_every_phase_on_its_own_orientation():
    st = _sphere_with_geometry()
    relabeled = st.relabeled(seed=7)
    vertex_map = dict(relabeled.vertex_map)
    assert sorted(vertex_map) == [0, 1, 2, 3]
    copied = _edges_by_pair(relabeled.spacetime)
    for (a, b), edge in _edges_by_pair(st).items():
        pa, pb = vertex_map[a], vertex_map[b]
        image = copied[(min(pa, pb), max(pa, pb))]
        assert image.get_length() == edge.get_length(), (a, b)
        expected = _ascending_phase(edge) if pa < pb else 0 - _ascending_phase(edge)
        assert _ascending_phase(image) == expected, (a, b)
        assert _ascending_phase(image) != 0, "a relabeling must not reset the connection"
