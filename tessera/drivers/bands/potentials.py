# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Model potentials with known answers, and the empirical pseudopotential of a
zinc-blende crystal with the plane-wave diagonalization it is compared against.

Every potential is returned as its values on the vertices of a `CrystalCell`;
the cell turns them into the weighted mass matrix that enters the pencil.
"""
import itertools

import numpy as np
import scipy.linalg
import scipy.special

from tessera.drivers.bands import HBAR2_OVER_2M, RYDBERG


# ---------------------------------------------------------------- separable cosine

def cosine_potential(cell, amplitude):
    """V(x) = amplitude * sum_a cos(2 pi x_a) in fractional coordinates: the
    separable Mathieu potential of a cubic cell."""
    return amplitude * np.cos(2.0 * np.pi * cell.fractional).sum(axis=1)


def mathieu_levels(amplitude, a, kinetic_scale, antiperiodic=False, count=4):
    """The lowest levels of -kinetic_scale u'' + amplitude cos(2 pi x / a) u = E u
    on one period, periodic (the zone centre) or antiperiodic (the zone edge).

    With z = pi x / a the equation is Mathieu's, y'' + (c - 2 q cos 2z) y = 0,
    with q = amplitude a^2 / (2 pi^2 kinetic_scale) and
    E = kinetic_scale (pi / a)^2 c. Periodic solutions in x have period pi in z
    and the characteristic values a_0, b_2, a_2, ...; antiperiodic ones have
    b_1, a_1, b_3, a_3, ...
    """
    q = amplitude * a * a / (2.0 * np.pi ** 2 * kinetic_scale)
    orders = range(1, 2 * count + 2, 2) if antiperiodic else range(0, 2 * count + 2, 2)
    values = []
    for m in orders:
        values.append(scipy.special.mathieu_a(m, q))
        if m > 0:
            values.append(scipy.special.mathieu_b(m, q))
    return kinetic_scale * (np.pi / a) ** 2 * np.sort(values)[:count]


def separable_levels(per_axis_levels, count):
    """The lowest sums of one level per axis, with multiplicity."""
    sums = sorted(sum(combo) for combo in itertools.product(*per_axis_levels))
    return np.array(sums[:count])


# ---------------------------------------------------------------- Gaussian well

def gaussian_well(cell, depth, width, center=(0.5, 0.5, 0.5)):
    """V(r) = -depth exp(-r^2 / 2 width^2) about the fractional `center`, summed
    over the nearest periodic images."""
    values = np.zeros(cell.size)
    for image in itertools.product((-1, 0, 1), repeat=3):
        offset = (cell.fractional - np.asarray(center) - np.asarray(image)) @ cell.lattice
        values -= depth * np.exp(-np.einsum("ij,ij->i", offset, offset) / (2.0 * width ** 2))
    return values


def radial_levels(potential, kinetic_scale, angular_momentum=0, count=1, radius=30.0, points=6000):
    """The lowest bound levels of -kinetic_scale (u'' - l(l+1) u / r^2) + V u = E u,
    u(0) = u(radius) = 0, for the isolated well: three uniform finite-difference
    grids, extrapolated in the squared spacing."""
    from tessera.drivers.bands.crystal import richardson
    spacings, levels = [], []
    for refinement in (1, 2, 4):
        n = points * refinement
        r = np.linspace(0.0, radius, n + 2)[1:-1]
        h = r[1] - r[0]
        diagonal = 2.0 * kinetic_scale / h ** 2 + potential(r) \
            + kinetic_scale * angular_momentum * (angular_momentum + 1) / r ** 2
        off = -kinetic_scale / h ** 2 * np.ones(n - 1)
        levels.append(scipy.linalg.eigvalsh_tridiagonal(diagonal, off, select="i",
                                                        select_range=(0, count - 1)))
        spacings.append(h)
    return richardson(spacings, np.array(levels))[0]


# ---------------------------------------------------------------- zinc blende

class ZincBlendeEPM:
    """The local empirical pseudopotential of a zinc-blende crystal,

        V(r) = sum_G [V_S(|G|^2) cos(G . tau) + i V_A(|G|^2) sin(G . tau)] e^{i G . r},

    over the reciprocal vectors of the face-centred cubic lattice, with the two
    atoms at +tau and -tau, tau = a (1, 1, 1) / 8, and the symmetric and
    antisymmetric form factors given by |G|^2 in units of (2 pi / a)^2.

    Reference: Cohen & Bergstresser, "Band structures and pseudopotential form
    factors for fourteen semiconductors of the diamond and zinc-blende
    structures", Physical Review 141, 789 (1966).
    """

    def __init__(self, a, symmetric, antisymmetric, kinetic_scale=HBAR2_OVER_2M):
        self.a = float(a)
        self.symmetric = dict(symmetric)
        self.antisymmetric = dict(antisymmetric)
        self.kinetic_scale = kinetic_scale
        self.tau = np.array([0.125, 0.125, 0.125])     # units of a
        self.components = {}
        top = max(list(self.symmetric) + list(self.antisymmetric))
        span = range(-int(np.sqrt(top)) - 1, int(np.sqrt(top)) + 2)
        for g in itertools.product(span, repeat=3):
            if len({abs(x) % 2 for x in g}) != 1:
                continue                                    # not a face-centred cubic reciprocal vector
            g2 = sum(x * x for x in g)
            phase = 2.0 * np.pi * np.dot(g, self.tau)
            value = self.symmetric.get(g2, 0.0) * np.cos(phase) + 1j * self.antisymmetric.get(g2, 0.0) * np.sin(phase)
            if abs(value) > 0.0:
                self.components[g] = value

    @classmethod
    def gallium_arsenide(cls):
        """The Cohen-Bergstresser form factors of GaAs (given in rydberg)."""
        return cls(a=5.64,
                   symmetric={3: -0.23 * RYDBERG, 8: 0.01 * RYDBERG, 11: 0.06 * RYDBERG},
                   antisymmetric={3: 0.07 * RYDBERG, 4: 0.05 * RYDBERG, 11: 0.01 * RYDBERG})

    # -- on a mesh

    def conventional_lattice(self):
        return self.a * np.eye(3)

    def primitive_lattice(self):
        return 0.5 * self.a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

    def potential(self, cell):
        """The vertex values of V on `cell` (real: V(-G) is the conjugate of V(G))."""
        values = np.zeros(cell.size, dtype=complex)
        for g, component in self.components.items():
            values += component * np.exp(2j * np.pi / self.a * (cell.positions @ np.array(g, dtype=float)))
        if np.abs(values.imag).max() > 1e-10 * max(1.0, np.abs(values.real).max()):
            raise ValueError("the form factors do not give a real potential")
        return values.real

    # -- plane waves

    def plane_wave_bands(self, k, count=8, cutoff=60.0):
        """The lowest `count` bands at the Cartesian momentum `k` (inverse
        angstrom) of the primitive cell, from plane waves with
        |k + G|^2 <= cutoff (2 pi / a)^2."""
        unit = 2.0 * np.pi / self.a
        kappa = np.asarray(k, dtype=float) / unit
        reach = int(np.sqrt(cutoff)) + 2
        basis = [g for g in itertools.product(range(-reach, reach + 1), repeat=3)
                 if len({abs(x) % 2 for x in g}) == 1 and np.sum((kappa + g) ** 2) <= cutoff]
        index = {g: i for i, g in enumerate(basis)}
        H = np.zeros((len(basis), len(basis)), dtype=complex)
        for g, i in index.items():
            H[i, i] = self.kinetic_scale * unit ** 2 * np.sum((kappa + g) ** 2)
            for dg, component in self.components.items():
                j = index.get(tuple(np.subtract(g, dg)))
                if j is not None:
                    H[i, j] += component
        return scipy.linalg.eigvalsh(H)[:count]

    def folded_bands(self, k, count, cutoff=60.0):
        """The bands of the conventional cubic cell at `k`: the primitive bands
        at `k` and at `k` plus each of the three zone-boundary vectors
        (2 pi / a) e_a that the cubic cell folds onto it."""
        unit = 2.0 * np.pi / self.a
        shifts = [np.zeros(3)] + [unit * np.eye(3)[axis] for axis in range(3)]
        levels = np.concatenate([self.plane_wave_bands(np.asarray(k) + shift, count, cutoff) for shift in shifts])
        return np.sort(levels)[:count]
