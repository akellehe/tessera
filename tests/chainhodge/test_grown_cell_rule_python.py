# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The grown-cell rule (whitepaper v17, Section 3 and Section 15): the length
inversion of a degree-1 Whitney block returns the squared lengths of a
Euclidean, a complex and a Lorentzian tetrahedron exactly, reports the
Whitney-form residual of a block that is not of Whitney form, leaves the scale
undetermined in two dimensions, and the phase rule returns the determinant of
the transport."""
import numpy as np
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob

GCR = ch.GrownCellRule
WM = ch.WhitneyMass

TETRAHEDRON = cob.ChainComplex.fromTopCells([[0, 1, 2, 3]])
TRIANGLE = cob.ChainComplex.fromTopCells([[0, 1, 2]])


def _squared_lengths(points, metric):
    """Squared lengths z_ij = (x_j - x_i)^T eta (x_j - x_i) in lexicographic edge order."""
    n = len(points)
    return [complex((points[j] - points[i]) @ metric @ (points[j] - points[i]))
            for i in range(n) for j in range(i + 1, n)]


def _local_block(K, s, k=1):
    blocks = WM.topSimplexBlocks(K, s, k)
    assert len(blocks) == 1
    return np.asarray(blocks[0].block), list(blocks[0].edgeIndices)


def _tetrahedra():
    rng = np.random.default_rng(7)
    euclid = rng.normal(size=(4, 3))
    yield "euclidean", _squared_lengths(euclid, np.eye(3))
    complex_lengths = [z * (1.0 + 0.05j * (m + 1)) for m, z in
                       enumerate(_squared_lengths(euclid, np.eye(3)))]
    yield "complex", complex_lengths
    # A tetrahedron in Minkowski 2+1 with a timelike edge (0, 3).
    minkowski = np.diag([-1.0, 1.0, 1.0])
    points = np.array([[0.0, 0.0, 0.0], [0.1, 1.0, 0.0], [0.2, 0.3, 1.1], [1.5, 0.2, 0.3]])
    yield "lorentzian", _squared_lengths(points, minkowski)


@pytest.mark.parametrize("name,s", list(_tetrahedra()))
def test_level_zero_lengths_returned_exactly(name, s):
    block, edges = _local_block(TETRAHEDRON, s)
    inversion = GCR.invertWhitneyBlock(block)
    assert inversion.dimension == 3
    assert inversion.scaleDetermined
    expected = np.array([s[e] for e in edges])
    np.testing.assert_allclose(np.array(inversion.squaredLengths), expected,
                               rtol=1e-12, atol=0.0)
    assert inversion.relativeResidual < 1e-14
    assert inversion.asymmetry < 1e-14
    volume = WM.certificate(TETRAHEDRON, s).volumes[0]
    np.testing.assert_allclose(inversion.volume, volume, rtol=1e-12)


@pytest.mark.parametrize("name,s", list(_tetrahedra()))
def test_forward_map_reproduces_block(name, s):
    block, _ = _local_block(TETRAHEDRON, s)
    inversion = GCR.invertWhitneyBlock(block)
    rebuilt = np.asarray(GCR.whitneyBlock(inversion.scaledGradientGram))
    np.testing.assert_allclose(rebuilt, block, rtol=0.0, atol=1e-14 * np.abs(block).max())


def test_gradient_gram_annihilates_constants_at_level_zero():
    s = list(_tetrahedra())[0][1]
    block, _ = _local_block(TETRAHEDRON, s)
    gram = np.asarray(GCR.invertWhitneyBlock(block).scaledGradientGram)
    assert np.abs(gram @ np.ones(4)).max() < 1e-13 * np.abs(gram).max()


def test_block_off_the_whitney_family_reports_its_residual():
    s = list(_tetrahedra())[0][1]
    block, _ = _local_block(TETRAHEDRON, s)
    rng = np.random.default_rng(3)
    perturbation = rng.normal(size=(6, 6))
    perturbation = 0.5 * (perturbation + perturbation.T)
    # Remove the component inside the Whitney family so that the residual is known.
    design = np.array([np.asarray(GCR.whitneyBlock(_unit(a, b)))[np.triu_indices(6)]
                       for a in range(4) for b in range(a, 4)]).T.real
    upper = perturbation[np.triu_indices(6)]
    orthogonal = upper - design @ np.linalg.lstsq(design, upper, rcond=None)[0]
    off = np.zeros((6, 6))
    off[np.triu_indices(6)] = orthogonal
    off = off + np.triu(off, 1).T
    scaled = 1e-3 * np.abs(block).max() / np.abs(off).max() * off
    inversion = GCR.invertWhitneyBlock(block + scaled)
    np.testing.assert_allclose(inversion.residual,
                               np.linalg.norm(scaled[np.triu_indices(6)]), rtol=1e-9)
    np.testing.assert_allclose(np.array(inversion.squaredLengths),
                               np.array(GCR.invertWhitneyBlock(block).squaredLengths),
                               rtol=1e-9)


def _unit(a, b):
    m = np.zeros((4, 4), dtype=complex)
    m[a, b] = m[b, a] = 1.0
    return m


def test_two_dimensional_scale_is_undetermined():
    s = [1.0 + 0.0j, 2.0 + 0.0j, 1.5 + 0.0j]
    block, edges = _local_block(TRIANGLE, s)
    inversion = GCR.invertWhitneyBlock(block)
    assert inversion.dimension == 2
    assert not inversion.scaleDetermined
    assert inversion.squaredLengths == []
    assert np.isnan(inversion.scale.real)
    scaled = np.array(inversion.scaledSquaredLengths)
    expected = np.array([s[e] for e in edges])
    ratio = scaled / expected
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    # The same triangle at any scale gives the same block: the scale is not in it.
    block2, _ = _local_block(TRIANGLE, [4.0 * z for z in s])
    np.testing.assert_allclose(block2, block, rtol=1e-12)


def test_higher_dimension_and_bad_shapes_refused():
    with pytest.raises(ValueError, match="root"):
        GCR.invertWhitneyBlock(np.eye(10, dtype=complex))
    with pytest.raises(ValueError, match="square"):
        GCR.invertWhitneyBlock(np.zeros((6, 5), dtype=complex))


def test_phase_rule_rank_one_returns_the_edge_connection():
    for phi in [0.3, -1.2 + 0.4j, 2.9 - 0.1j]:
        U = np.exp(1j * phi)
        np.testing.assert_allclose(GCR.transportConnection(np.array([[U]])), U, rtol=1e-15)


def test_phase_rule_is_the_full_determinant():
    rng = np.random.default_rng(11)
    M = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    np.testing.assert_allclose(GCR.transportConnection(M), np.linalg.det(M), rtol=1e-13)
    # Frame changes at the two ends multiply it by det(g_v)^-1 det(g_w).
    gv = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    gw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    np.testing.assert_allclose(GCR.transportConnection(np.linalg.inv(gv) @ M @ gw),
                               np.linalg.det(M) * np.linalg.det(gw) / np.linalg.det(gv),
                               rtol=1e-12)
    with pytest.raises(ValueError, match="common-rank"):
        GCR.transportConnection(np.ones((2, 3), dtype=complex))
