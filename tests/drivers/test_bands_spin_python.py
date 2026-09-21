# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Stage 3 of the band-structure drivers (#1159): two sheets as the two spin
components, and spin-orbit coupling as the attachment block between them."""
import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp

import tessera.quantum as quantum
from tessera import cobordism as cob
from tessera.drivers.bands import potentials as pot
from tessera.drivers.bands import spin
from tessera.drivers.bands.crystal import CrystalCell, solve_pencil


def test_l_dot_s_has_a_quartet_and_a_doublet():
    matrix = spin.l_dot_s()
    assert np.abs(matrix - matrix.conj().T).max() < 1e-15
    assert np.linalg.eigvalsh(matrix) == pytest.approx([-1, -1, 0.5, 0.5, 0.5, 0.5], abs=1e-14)
    assert abs(np.trace(matrix)) < 1e-15             # the centre of gravity does not move


def test_two_sheets_double_every_level_and_a_constant_coupling_splits_them():
    cell = CrystalCell.cubic(1.0, 4, kinetic_scale=1.0)
    A, M = cell.pencil((0.3, -0.2, 0.1), pot.cosine_potential(cell, 5.0))
    single = solve_pencil(A, M, 4, -20.0).energies
    A2, M2 = spin.two_sheet_pencil(A, M)
    assert solve_pencil(A2, M2, 8, -20.0).energies == pytest.approx(np.repeat(single, 2), rel=1e-10)
    delta = 0.6
    coupled = A2 + delta * sp.kron(sp.csc_matrix([[0, 1], [1, 0]]), M)       # Delta sigma_x
    expected = np.sort(np.concatenate([single - delta, single + delta]))
    assert solve_pencil(coupled.tocsc(), M2, 8, -20.0).energies == pytest.approx(expected, rel=1e-10)


def test_spin_orbit_splits_an_exact_triplet_into_a_quartet_and_a_doublet():
    """The first shell of the free torus holds the three functions sin(2 pi x_a),
    an exactly degenerate triplet that transforms as a vector. With them as the
    projector functions the six levels move by xi / 2 (four) and -xi (two),
    3 xi / 2 apart, and the cosine triplet of the same shell does not move."""
    cell = CrystalCell.cubic(1.0, 6, kinetic_scale=1.0)
    A, M = cell.pencil()
    beta = np.sin(2 * np.pi * cell.fractional)
    beta /= np.sqrt(np.einsum("ij,ij->j", beta, M.real @ beta))
    shell = (beta[:, 0] @ (A.real @ beta[:, 0]))
    assert np.abs(A.real @ beta - shell * (M.real @ beta)).max() < 1e-10 * shell   # exact eigenvectors
    xi = 0.8
    A2, M2 = spin.two_sheet_pencil(A, M)
    read = solve_pencil((A2 + spin.spin_orbit_matrix(M, beta, xi)).tocsc(), M2, 14, -5.0)
    assert read.certified()
    expected = np.sort(np.concatenate([[0.0, 0.0], [shell - xi] * 2, [shell] * 6, [shell + xi / 2] * 4]))
    assert read.energies == pytest.approx(expected, abs=1e-8)
    quartet, doublet = read.energies[-1], read.energies[2]
    assert quartet - doublet == pytest.approx(1.5 * xi, abs=1e-8)


def test_the_mesh_splits_a_cubic_triplet_at_the_order_of_its_error():
    """Every cube is cut along the same body diagonal, so the mesh keeps the
    threefold axis along (1, 1, 1) and loses the fourfold ones: the p level of a
    spherical well is a doublet and a singlet, closing under refinement."""
    splittings = []
    for n in (8, 16):
        cell = CrystalCell.cubic(10.0, n, kinetic_scale=1.0)
        levels = cell.solve((0, 0, 0), 4, pot.gaussian_well(cell, 6.0, 1.6)).energies[1:4]
        assert min(levels[1] - levels[0], levels[2] - levels[1]) < 1e-8      # two of the three coincide
        splittings.append(np.ptp(levels))
    assert 0.0 < splittings[1] < 0.65 * splittings[0]


def test_the_coupling_block_is_a_hopping_term_on_fock_space():
    """Eight modes: two sheets of the tetrahedron's four levels with a constant
    coupling. The second-quantized block operator has the subset-sum spectrum of
    the one-particle block."""
    levels = np.array([0.0, 40.0, 40.0, 40.0])
    coupling = 3.0 * np.eye(4)
    block = np.array(cob.OccupationSpectra.hoppingBlock(list(np.diag(levels).ravel().astype(complex)), 4,
                                                        list(np.diag(levels).ravel().astype(complex)), 4,
                                                        list(coupling.ravel().astype(complex)))).reshape(8, 8)
    one_particle = np.linalg.eigvalsh(block)
    assert one_particle == pytest.approx(np.sort(np.concatenate([levels - 3.0, levels + 3.0])))
    rows, cols, values, n = quantum.FockDirectSum(4, 4).dGammaBlockCOO(np.diag(levels).astype(complex),
                                                                       np.diag(levels).astype(complex),
                                                                       coupling.astype(complex))
    fock = sp.csr_matrix((values, (rows, cols)), shape=(n, n)).toarray()
    assert n == 256 and np.abs(fock - fock.conj().T).max() < 1e-14
    expected = np.sort(np.array(cob.OccupationSpectra.fockSums(list(one_particle.astype(complex)))).real)
    assert scipy.linalg.eigvalsh(fock) == pytest.approx(expected, abs=1e-9)
