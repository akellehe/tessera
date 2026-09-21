# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The sparse storage of the boundary maps (#1157): `boundaryEntries` is the
stored form, the dense `boundaryMatrix` is its image, and a complex far too
large for dense maps is still built and checked."""
import numpy as np

import tessera
from tessera import cobordism as cob


def _dense(K, k):
    rows, cols = K.numSimplices(k - 1), K.numSimplices(k)
    return np.array(K.boundaryMatrix(k), dtype=int).reshape(rows, cols)


def test_entries_are_the_nonzeros_of_the_dense_map():
    K = cob.ChainComplex.fromTopCells(tessera.PeriodicKuhnGrid.cubic(3).cells())
    for k in range(1, 4):
        entries = K.boundaryEntries(k)
        dense = _dense(K, k)
        assert len(entries) == (k + 1) * K.numSimplices(k) == np.count_nonzero(dense)
        assert [c for _, c, _ in entries] == sorted(c for _, c, _ in entries)
        rebuilt = np.zeros_like(dense)
        for r, c, v in entries:
            assert v in (-1, 1)
            rebuilt[r, c] = v
        assert np.array_equal(rebuilt, dense)
    assert K.boundaryEntries(0) == [] and K.boundaryEntries(4) == []
    assert np.array_equal(_dense(K, 1) @ _dense(K, 2), np.zeros((27, 324), dtype=int))


def test_a_spacetime_complex_stores_the_same_entries():
    grid = tessera.PeriodicKuhnGrid.cubic(3)
    sig = tessera.Signature(3, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0, tessera.PREFERRED, grid)
    st.build()
    fromSpacetime = cob.ChainComplex.fromSpacetime(st)
    fromCells = cob.ChainComplex.fromTopCells(grid.cells())
    for k in range(1, 4):
        assert fromSpacetime.boundaryEntries(k) == fromCells.boundaryEntries(k)
    assert fromSpacetime.boundaryComposesToZero()


def test_a_large_complex_never_forms_a_dense_map():
    """16^3 vertices: the dense boundary maps would need about 21 GB."""
    grid = tessera.PeriodicKuhnGrid.cubic(16)
    K = cob.ChainComplex.fromTopCells(grid.cells())
    assert list(K.fVector()) == [4096, 28672, 49152, 24576]
    assert K.eulerCharacteristic() == 0
    assert K.boundaryComposesToZero()
    assert len(K.boundaryEntries(3)) == 4 * 24576
