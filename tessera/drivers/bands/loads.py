# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The load of a function against the vertex functions of the mesh,

    b_v = int f(x) lambda_v(x) dx ,

by a quadrature of fixed degree on every top simplex. The separable part of a
pseudopotential enters the pencil as the low-rank term P D P^dagger whose columns
are these loads of the projector functions. `M beta`, the mass matrix on the
vertex values, is the load of the interpolant of the projector; the load of the
projector itself has no closed form, because the radial functions are tabulated.

The rule is the collapsed Gauss rule (the conical product of Gauss-Jacobi rules
of weights (1 - x)^(d-1), ..., (1 - x)^0): with n points per direction it is
exact for every polynomial of degree 2 n - 1 on the simplex. The simplices and
their volumes are the library's (`ChainComplex.orientedTopSimplices`, the
`volumes` of `WhitneyMass.certificate`).

A section of crystal momentum k is carried by the covariant Whitney pencil as
the vertex values u_v of its cell-periodic part, psi_v = exp(i k . x_v) u_v
being interpolated linearly. The load of a function centred on an ion at tau
is then sum over the ion's images R of b_v^R exp(-i k . (x_v - tau - R)), with
b_v^R the zone-centre load of the image: the loads are kept per vertex and per
displacement from the image (`LocalLoads`), so every momentum, and the
derivative with respect to the momentum, is a sum over the same entries.
"""
import itertools

import numpy as np
from scipy.special import roots_jacobi

from tessera import chainhodge as ch


def collapsed_gauss_rule(points, dimension=3):
    """Barycentric coordinates (Q, dimension + 1) and weights (Q,) summing to
    one: the collapsed Gauss rule of `points` per direction on a simplex,
    exact to degree 2 points - 1."""
    if points < 1:
        raise ValueError("a quadrature rule has at least one point per direction")
    axes = []
    for level in range(dimension):
        nodes, weights = roots_jacobi(points, dimension - 1 - level, 0.0)
        axes.append((0.5 * (nodes + 1.0), weights / weights.sum()))
    barycentric, weights = [], []
    for combination in itertools.product(*[range(points)] * dimension):
        remaining, coordinates, weight = 1.0, [], 1.0
        for level, index in enumerate(combination):
            nodes, axis_weights = axes[level]
            coordinates.append(nodes[index] * remaining)
            remaining *= 1.0 - nodes[index]
            weight *= axis_weights[index]
        barycentric.append([remaining] + coordinates)
        weights.append(weight)
    return np.array(barycentric), np.array(weights)


def radial_reach(radius, values):
    """The radius beyond which a tabulated function cannot change a sum in
    double precision: the last mesh point where it exceeds the machine epsilon
    times its largest magnitude."""
    magnitude = np.abs(np.asarray(values, dtype=float))
    present = np.nonzero(magnitude > np.finfo(float).eps * magnitude.max())[0]
    last = min(present[-1] + 1, len(radius) - 1) if len(present) else 0
    return float(np.asarray(radius)[last])


class LocalLoads:
    """The loads of functions centred on one ion, one entry per vertex and per
    Cartesian displacement of that vertex from an image of the ion."""

    def __init__(self, size, vertex, displacement, values):
        self.size, self.vertex, self.displacement, self.values = int(size), vertex, displacement, values

    def _scatter(self, values):
        out = np.zeros((self.size, values.shape[1]), dtype=values.dtype)
        np.add.at(out, self.vertex, values)
        return out

    def assemble(self, momentum=None):
        """The loads (vertices, functions): at the zone centre when `momentum`
        is None, else with the Bloch phase of the Cartesian momentum."""
        if momentum is None:
            return self._scatter(self.values)
        phase = np.exp(-1j * (self.displacement @ np.asarray(momentum, dtype=float)))
        return self._scatter(self.values * phase[:, None])

    def derivative(self, axis):
        """The derivative of `assemble` with respect to the Cartesian momentum
        component `axis`, at the zone centre."""
        return self._scatter(-1j * self.displacement[:, axis, None] * self.values)


class SimplexQuadrature:
    """The collapsed Gauss rule of `points` per direction on every top simplex
    of a `CrystalCell`."""

    def __init__(self, cell, points):
        self.cell, self.points = cell, int(points)
        vertex_index = {int(simplex[0]): i for i, simplex in enumerate(cell.complex.kSimplexVertices(0))}
        self.tops = np.array([[vertex_index[int(v)] for v in top] for top in cell.complex.orientedTopSimplices()])
        self.volumes = np.array(ch.WhitneyMass.certificate(cell.complex, list(cell.squared_lengths)).volumes).real
        self.barycentric, self.weights = collapsed_gauss_rule(self.points, self.tops.shape[1] - 1)
        self.moments = self.weights[:, None] * self.barycentric              # weight times lambda_a at each point
        # Every simplex is laid out from its first vertex along the shortest grid steps, which undoes the
        # periodic wrap (`PeriodicKuhnGrid.displacement` is the same steps, one pair at a time).
        divisions = np.array(cell.divisions)
        steps = cell.index[self.tops] - cell.index[self.tops[:, :1]]
        steps -= divisions * np.rint(steps / divisions).astype(int)
        self.shift = (cell.index[self.tops[:, :1]] + steps - cell.index[self.tops]) // divisions   # whole cells
        self.corners = (cell.index[self.tops[:, :1]] + steps) / divisions                          # fractional
        self.radius = float(np.sqrt(np.abs(np.asarray(cell.squared_lengths)).max()))

    def integrate(self, values):
        """The loads of an integrand given at the rule's points of every
        simplex, `values[simplex, point, function]`."""
        local = self.volumes[:, None, None] * np.einsum("qa,tqf->taf", self.moments, values, optimize=True)
        out = np.zeros((self.cell.size, values.shape[2]), dtype=local.dtype)
        np.add.at(out, self.tops, local)
        return out

    def at_points(self, vertex_values):
        """A piecewise-linear function at the rule's points, `[simplex, point]`."""
        return np.einsum("qa,ta...->tq...", self.barycentric, np.asarray(vertex_values)[self.tops])

    def loads(self, function, center, reach, images=(0,), block=2048):
        """`LocalLoads` of `function(offsets)` (rows of Cartesian offsets from the
        centre to an array [row, function]) centred on the fractional position
        `center` and on its images, over the simplices within `reach` of each."""
        cell = self.cell
        center = np.asarray(center, dtype=float)
        centroids = self.corners.mean(axis=1)
        keys, displacements, loads = [], [], []
        for image in itertools.product(images, repeat=3):
            origin = center + np.asarray(image, dtype=float)
            distance = np.linalg.norm((centroids - origin) @ cell.lattice, axis=1)
            near = np.nonzero(distance < reach + self.radius)[0]
            for start in range(0, len(near), block):
                chosen = near[start:start + block]
                corners = (self.corners[chosen] - origin) @ cell.lattice                   # (t, a, 3)
                offsets = np.einsum("qa,tax->tqx", self.barycentric, corners)
                values = np.asarray(function(offsets.reshape(-1, 3)))
                values = values.reshape(len(chosen), len(self.weights), -1)
                local = self.volumes[chosen, None, None] * np.einsum("qa,tqf->taf", self.moments, values, optimize=True)
                # One key per vertex and per whole-cell shift of the vertex from the image.
                shift = self.shift[chosen] - np.asarray(image, dtype=int)
                keys.append(np.concatenate([self.tops[chosen][..., None], shift], axis=2).reshape(-1, 4))
                displacements.append(corners.reshape(-1, 3))
                loads.append(local.reshape(-1, local.shape[2]))
        if not keys:
            return LocalLoads(cell.size, np.zeros(0, dtype=int), np.zeros((0, 3)), np.zeros((0, 0)))
        keys, displacements, loads = np.concatenate(keys), np.concatenate(displacements), np.concatenate(loads)
        unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        merged = np.zeros((len(unique), loads.shape[1]), dtype=loads.dtype)
        np.add.at(merged, inverse.ravel(), loads)
        return LocalLoads(cell.size, unique[:, 0], displacements[first], merged)
