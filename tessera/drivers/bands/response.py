# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The derivative of a band energy with respect to the squared edge lengths: the
Hellmann-Feynman force of an occupied mode on the geometry.

For an M-normalized eigenpair (E, z) of the pencil
`A = kinetic_scale * d_1 M_1 d_1^T + M_0[V]`, `M = M_0`,

    dE/ds_e = z^dagger (dA/ds_e - E dM/ds_e) z
            = kinetic_scale * x^dagger (dM_1/ds_e) x + z^dagger (dM_0[V]/ds_e) z - E z^dagger (dM_0/ds_e) z,

with x = d_1^T z the edge differences of the mode. Every term is a contraction
of a Whitney mass derivative, evaluated per top simplex without forming a
derivative matrix (`WhitneyMass.derivativeContraction`). The boundary maps and
the Bloch phases do not depend on the lengths.

The Whitney Hodge Laplacian is homogeneous of degree -1 in the squared lengths,
so the kinetic levels obey Euler's identity sum_e s_e dE/ds_e = -E; a potential
held fixed at the vertices is homogeneous of degree zero and drops out of the
sum, leaving minus the kinetic part of the energy.
"""
import numpy as np
import scipy.sparse as sp

from tessera import chainhodge as ch


def length_derivative(cell, read, band, potential=None):
    """dE/ds_e for every edge (canonical order) of band `band` of the `BandRead`
    `read`, at fixed vertex values of the potential."""
    if not np.allclose(read.kappa, 0.0):
        raise NotImplementedError("length derivatives are implemented at the zone centre")
    u = read.vectors[:, band:band + 1]
    energy = read.energies[band]
    boundary = sp.csc_matrix(cell.base.boundary(1))
    x = boundary.T @ u
    s = cell.squared_lengths
    K = cell.complex
    kinetic = np.array(ch.WhitneyMass.derivativeContraction(K, s, 1, x.conj(), x))
    overlap = np.array(ch.WhitneyMass.derivativeContraction(K, s, 0, u.conj(), u))
    out = cell.kinetic_scale * kinetic - energy * overlap
    if potential is not None:
        # M_0[V] depends on the lengths through the volumes only, exactly as M_0
        # does: d M_0[V] / ds_e restricted to a top simplex is its block times
        # dln|T|/ds_e. The contraction of M_0 with the weighted pair (V u, u)
        # is not that, so differentiate the volumes directly.
        out = out + _weighted_mass_derivative(cell, u, np.asarray(potential))
    return out.real


def _weighted_mass_derivative(cell, u, potential):
    """z^dagger (dM_0[V]/ds_e) z per edge, from the per-simplex blocks of M_0:
    on a top simplex both M_0 and M_0[V] are |T| times a constant matrix, so
    dM_0[V]|_T / ds_e = M_0[V]|_T * (d|T|/ds_e) / |T|, and the logarithmic
    derivative of the volume is read off the degree-zero block and its derivative."""
    K, s = cell.complex, cell.squared_lengths
    blocks = ch.WhitneyMass.topSimplexBlocks(K, s, 0, ch.Branch.Continuation, True)
    out = np.zeros(K.numSimplices(1), dtype=complex)
    d = K.dimension()
    scale = float(np.prod(np.arange(1, d + 1))) / float(np.prod(np.arange(1, d + 4)))
    for block in blocks:
        vertices = np.array(block.cellIndices)
        volume = block.block[0, 0] * (d + 1) * (d + 2) / 2.0
        local_u = u[vertices, 0]
        local_v = potential[vertices]
        su, sv = local_u.sum(), local_v.sum()
        # sum_abc mu_abc conj(u_a) u_b V_c with mu = 1 + d_ab + d_ac + d_bc + 2 d_abc
        value = (np.conj(su) * su * sv + np.vdot(local_u, local_u) * sv
                 + np.conj(su) * np.dot(local_u, local_v) + np.vdot(local_u, local_v) * su
                 + 2.0 * np.sum(np.abs(local_u) ** 2 * local_v))
        for m, edge in enumerate(block.edgeIndices):
            dlog = block.derivative[m][0, 0] / block.block[0, 0]
            out[edge] += scale * volume * dlog * value
    return out


def euler_defect(cell, read, band, derivative, potential_energy=0.0):
    """sum_e s_e dE/ds_e + (E - <V>), which vanishes: the kinetic part of the
    energy is homogeneous of degree -1 and the potential part of degree 0."""
    s = np.array(cell.squared_lengths).real
    return float(np.dot(s, derivative) + (read.energies[band] - potential_energy))
