# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Two sheets as the two spin components, and spin-orbit coupling as the
attachment block between them.

Two copies of a cell that share one geometry carry the block-diagonal pencil
`SparsePencilComposition.directSum`. A spin-orbit term of the projector form

    H_so = xi sum_{m m' s s'} |beta_m> (L . S)_{m s, m' s'} <beta_m'| ,

with `beta_m` the three p-like projector functions r_m f(r) about an atom and
L . S the 6 x 6 matrix of the orbital angular momentum one coupled to spin one
half, enters the left-hand matrix of the pencil as `(M beta) (L . S) (M beta)^dagger`:
within a sheet for the part diagonal in spin, and as the coupling block between
the sheets for the rest. It has rank six, so it is kept in factored form.

L . S has the eigenvalues +1/2 (four times, total angular momentum 3/2) and -1
(twice, total angular momentum 1/2), so a p level splits into a quartet and a
doublet 3 xi / 2 apart, in the ratio 2 : 1 of their degeneracies, about an
unmoved centre of gravity.
"""
import itertools

import numpy as np
import scipy.sparse as sp

from tessera import chainhodge as ch

# The orbital angular momentum one in the Cartesian basis (x, y, z): (L_a)_{bc} = -i epsilon_abc.
_L = np.zeros((3, 3, 3), dtype=complex)
for a, b, c in itertools.permutations(range(3)):
    _L[a, b, c] = -1j * np.linalg.det(np.eye(3)[[a, b, c]])
_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def l_dot_s():
    """L . S on the six states (spin, Cartesian p orbital), spin the slow index."""
    return sum(np.kron(0.5 * _PAULI[a], _L[a]) for a in range(3))


def p_projectors(cell, center, width):
    """The nodal values of the three p-like functions r_m exp(-r^2 / 2 width^2)
    about the fractional `center` (nearest periodic image), M-orthonormalized."""
    offset = cell.fractional - np.asarray(center, dtype=float)
    offset -= np.rint(offset)
    r = offset @ cell.lattice
    beta = r * np.exp(-np.einsum("ij,ij->i", r, r) / (2.0 * width ** 2))[:, None]
    mass = cell.mass.dressed()
    gram = beta.T @ (mass @ beta)
    return beta @ np.linalg.inv(np.linalg.cholesky(gram.real)).T


def two_sheet_pencil(A, M):
    """The direct sum of a pencil with itself: both spin components, uncoupled."""
    sheet = ch.SparsePencil(0, sp.csc_matrix(A, dtype=complex), sp.csc_matrix(M, dtype=complex))
    both = ch.SparsePencilComposition.directSum(sheet, sheet)
    return sp.csc_matrix(both.A), sp.csc_matrix(both.M)


def spin_orbit_factors(M, beta, strength):
    """(P, C) with H_so = P C P^dagger on the two-sheet space: P = diag(M beta, M beta)
    (2n x 6) and C = strength * L . S."""
    loaded = M @ beta
    n = loaded.shape[0]
    P = np.zeros((2 * n, 6), dtype=complex)
    P[:n, :3] = loaded
    P[n:, 3:] = loaded
    return P, strength * l_dot_s()


def spin_orbit_matrix(M, beta, strength):
    """H_so assembled as a sparse matrix (small cells; it fills the support of beta)."""
    P, C = spin_orbit_factors(M, beta, strength)
    return sp.csc_matrix(P @ C @ P.conj().T)
