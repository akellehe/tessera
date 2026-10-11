# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A periodic mesh graded toward the ions, and the Coulomb kernel on it.

The geometry of the framework is one squared length per edge, so a mesh is
graded by changing the lengths and nothing else. `GradedCell` keeps the
vertices, the simplices and the periodic identification of the Kuhn grid and
moves the vertices by a smooth periodic map of the cell (`IonGrading`): inside a
ball about every ion the distance r to the ion becomes

    rho(r) = r [1 - (1 - 1 / gamma) (1 - (r / r_c)^p)^3] ,

which is r / gamma at the ion (the mesh is gamma times finer there), joins the
identity at the radius r_c with two continuous derivatives, and is increasing
for every gamma >= 1. The even power p (`flatness`) sets how far from the ion
the compression holds: it has halved at (r / r_c)^p = 1 / (3 (gamma - 1)). The displacements of the ions add, and `GradedCell`
measures that the piecewise-linear image of the mesh keeps the orientation of
every top simplex (it refuses a folded mesh, which the squared lengths alone
would not show).

A crystal momentum is the flat connection U_vw = exp(i k . dx_vw) on the true
displacement of every edge, so the constant section is the plane wave
exp(i k . x) at the vertices wherever they are. It differs from the connection
on the grid steps by the gauge exp(i k . (x_v - X_v)), X_v being the vertex of
the grid: the spectra are the same, and the sections are the cell-periodic
parts with respect to the true positions, which is what the pair loads, the
projectors and the derivative with respect to the momentum assume.

Everything that used the translation invariance of the uniform grid has a
counterpart here that does not: `GradedCoulombKernel` inverts the stiffness
matrix by a sparse factorization in place of the Fourier transform, and its
zero-momentum constant is the same auxiliary-function integral taken on the
dressed stiffness matrix itself.
"""
import itertools

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tessera import chainhodge as ch
from tessera.drivers.bands import E2
from tessera.drivers.bands.crystal import CrystalCell


class IonGrading:
    """The radial compression toward `centers` (fractional coordinates, rows):
    within `radius` (one value, or one per centre, in the length unit of the
    lattice) of a centre the distance r to it becomes rho(r), with the
    `compression` gamma >= 1 (one value or one per centre). `compression = 1`
    is the identity. The displacements of the centres add. Where the balls of
    two centres overlap the sum is still smooth and periodic, and whether it is
    one-to-one on a mesh is measured there (`GradedCell.orientation_margin`);
    a centre that the pulls of the others would move is refused, because the
    mesh would then be fine about another point than the ion."""

    def __init__(self, centers, radius, compression, flatness=2):
        self.centers = np.atleast_2d(np.asarray(centers, dtype=float))
        self.flatness = int(flatness)
        if self.flatness < 2 or self.flatness % 2:
            raise ValueError("the flatness is an even power of the radius, so that the map is smooth at the ion")
        count = len(self.centers)
        self.radius = np.broadcast_to(np.asarray(radius, dtype=float), (count,)).copy()
        self.compression = np.broadcast_to(np.asarray(compression, dtype=float), (count,)).copy()
        if np.any(self.compression < 1.0) or np.any(self.radius <= 0.0):
            raise ValueError("a grading needs a compression of at least one and a positive radius")

    @classmethod
    def of_crystal(cls, crystal, radius, compression, flatness=2):
        """Toward every ion of an `abinitio.Crystal`; `radius` and `compression`
        are one value, one per ion, or a mapping from the element to a value."""
        per_ion = lambda value: ([value[pseudo.element] for pseudo, _ in crystal.ions]
                                 if isinstance(value, dict) else value)
        return cls([position for _, position in crystal.ions], per_ion(radius), per_ion(compression), flatness)

    def contraction(self, t, compression):
        """rho(r) / r at r = t r_c."""
        t = np.minimum(np.asarray(t, dtype=float), 1.0)
        return 1.0 - (1.0 - 1.0 / compression) * (1.0 - t ** self.flatness) ** 3

    def displacement(self, fractional, lattice):
        """The Cartesian displacement of the points `fractional` (rows)."""
        lattice = np.asarray(lattice, dtype=float)
        fractional = np.atleast_2d(np.asarray(fractional, dtype=float))
        moved = np.zeros_like(fractional)
        reach = [int(np.ceil(self.radius.max() / height)) for height in
                 np.abs(np.linalg.det(lattice)) / np.linalg.norm(np.cross(lattice[[1, 2, 0]], lattice[[2, 0, 1]]), axis=1)]
        images = list(itertools.product(*[range(-n, n + 1) for n in reach]))
        for center, radius, compression in zip(self.centers, self.radius, self.compression):
            offset = fractional - center
            offset -= np.rint(offset)
            for image in images:
                d = (offset - np.asarray(image, dtype=float)) @ lattice
                r = np.linalg.norm(d, axis=1)
                inside = r < radius
                if np.any(inside):
                    factor = self.contraction(r[inside] / radius, compression)
                    moved[inside] += (factor - 1.0)[:, None] * d[inside]
        return moved

    def place(self, fractional, lattice):
        """The fractional coordinates of the moved points."""
        lattice = np.asarray(lattice, dtype=float)
        if np.abs(self.displacement(self.centers, lattice)).max() > 1e-12 * np.abs(lattice).max():
            raise ValueError("the pulls on an ion do not cancel: the grading would move it")
        return np.asarray(fractional, dtype=float) + self.displacement(fractional, lattice) @ np.linalg.inv(lattice)


class IonRefinement:
    """Where the mesh is bisected: `shells` is a list of (radius, halvings), and
    every top simplex whose centroid lies within `radius` of a centre is
    bisected until its edges are `halvings` times halved (three bisections of a
    Kuhn simplex halve every edge)."""

    def __init__(self, centers, shells):
        self.centers = np.atleast_2d(np.asarray(centers, dtype=float))
        self.shells = sorted(((float(radius), int(halvings)) for radius, halvings in shells), reverse=True)

    @classmethod
    def of_crystal(cls, crystal, shells, elements=None):
        """About every ion of an `abinitio.Crystal`, or those of the `elements` only."""
        return cls([position for pseudo, position in crystal.ions if elements is None or pseudo.element in elements],
                   shells)

    def levels(self, fractional, lattice):
        """The number of bisections asked for at the points `fractional` (rows)."""
        wanted = np.zeros(len(fractional), dtype=int)
        for center in self.centers:
            offset = fractional - center
            offset -= np.rint(offset)
            nearest = np.full(len(fractional), np.inf)
            for image in itertools.product((-1, 0, 1), repeat=3):
                nearest = np.minimum(nearest, np.linalg.norm((offset - np.asarray(image, dtype=float)) @ lattice, axis=1))
            for radius, halvings in self.shells:
                wanted = np.where(nearest < radius, np.maximum(wanted, 3 * halvings), wanted)
        return wanted


class KuhnBisection:
    """The periodic Kuhn grid as ordered simplices, and their conforming
    bisection (Maubach, SIAM Journal on Scientific Computing 16, 210 (1995);
    Traxler, Computing 59, 115 (1997)).

    A Kuhn simplex is the path x_0, x_0 + e_p(1), x_0 + e_p(1) + e_p(2), x_0 +
    (1, 1, 1) through a cube, one per permutation p of the axes. A simplex
    (x_0, x_1, x_2, x_3) of level l is bisected at the midpoint z of the edge
    (x_0, x_k), k = 3 - (l mod 3), into (x_0, .., x_{k-1}, z, x_{k+1}, ..) and
    (x_1, .., x_k, z, x_{k+1}, ..) of level l + 1. Before an edge is cut, every
    simplex around it is brought to the state in which that edge is the one it
    would cut, so the mesh stays conforming: no vertex lies inside an edge or a
    face of a neighbour. Three levels halve every edge and return 8 Kuhn
    simplices of half the size, so the shapes do not degenerate.

    Coordinates are integers in units of 1 / `SCALE` of a grid step, unwrapped
    within a simplex and wrapped for the identity of a vertex."""
    SCALE = 1 << 10

    def __init__(self, divisions):
        self.divisions = np.array(divisions, dtype=np.int64)
        self.period = self.divisions * self.SCALE
        n1, n2, n3 = (int(n) for n in divisions)
        self.vertex_ids, self.vertex_coordinates = {}, []
        for i, j, k in itertools.product(range(n1), range(n2), range(n3)):           # the numbering of the grid
            self._vertex((i * self.SCALE, j * self.SCALE, k * self.SCALE))
        self.elements, self.around = {}, {}
        self.midpoints = {}
        self._next = 0
        for origin in itertools.product(range(n1), range(n2), range(n3)):
            for order in itertools.permutations(range(3)):
                corners = [np.array(origin, dtype=np.int64) * self.SCALE]
                for axis in order:
                    step = np.zeros(3, dtype=np.int64)
                    step[axis] = self.SCALE
                    corners.append(corners[-1] + step)
                self._add(tuple(tuple(int(x) for x in corner) for corner in corners), 0)

    def _vertex(self, coordinates):
        key = tuple(int(x) % int(p) for x, p in zip(coordinates, self.period))
        if key not in self.vertex_ids:
            self.vertex_ids[key] = len(self.vertex_coordinates)
            self.vertex_coordinates.append(key)
        return self.vertex_ids[key]

    @staticmethod
    def _edges(ids):
        return [frozenset((ids[a], ids[b])) for a, b in itertools.combinations(range(4), 2)]

    def _add(self, corners, level):
        ids = tuple(self._vertex(corner) for corner in corners)
        self.elements[self._next] = (corners, ids, level)
        for edge in self._edges(ids):
            self.around.setdefault(edge, set()).add(self._next)
        self._next += 1

    def _cut(self, element):
        _, ids, level = self.elements[element]
        return frozenset((ids[0], ids[3 - level % 3]))

    def refine(self, element, depth=0):
        """Bisect `element`, after whatever its neighbours need first."""
        if element not in self.elements:
            return
        if depth > 64:
            raise RuntimeError("the conforming closure did not terminate")
        edge = self._cut(element)
        while True:
            behind = [other for other in self.around[edge] if self._cut(other) != edge]
            if not behind:
                break
            for other in behind:
                self.refine(other, depth + 1)
        for other in list(self.around[edge]):
            corners, ids, level = self.elements.pop(other)
            for shared in self._edges(ids):
                self.around[shared].discard(other)
                if not self.around[shared]:
                    del self.around[shared]
            k = 3 - level % 3
            middle = tuple((a + b) // 2 for a, b in zip(corners[0], corners[k]))
            self.midpoints[frozenset((ids[0], ids[k]))] = self._vertex(middle)
            self._add(corners[:k] + (middle,) + corners[k + 1:], level + 1)
            self._add(corners[1:k + 1] + (middle,) + corners[k + 1:], level + 1)

    def refine_to(self, wanted):
        """Bisect until every simplex has the level `wanted(centroids)` asks for
        at its centroid (fractional coordinates, rows)."""
        while True:
            keys = list(self.elements)
            corners = np.array([self.elements[key][0] for key in keys], dtype=float)
            levels = np.array([self.elements[key][2] for key in keys])
            centroids = corners.mean(axis=1) / self.period
            short = np.nonzero(levels < wanted(centroids))[0]
            if len(short) == 0:
                return
            for index in short:
                self.refine(keys[index])

    def arrays(self):
        """(cells as rows of vertex ids, the fractional coordinates of the
        vertices, the edges as (v, w, fractional displacement from v to w) with
        v < w)."""
        corners = np.array([corners for corners, _, _ in self.elements.values()], dtype=np.int64)
        ids = np.array([ids for _, ids, _ in self.elements.values()], dtype=np.int64)
        v, w, d = [], [], []
        for a, b in itertools.combinations(range(4), 2):
            flip = ids[:, a] > ids[:, b]
            v.append(np.where(flip, ids[:, b], ids[:, a]))
            w.append(np.where(flip, ids[:, a], ids[:, b]))
            d.append(np.where(flip[:, None], -1, 1) * (corners[:, b] - corners[:, a]))
        v, w, d = np.concatenate(v), np.concatenate(w), np.concatenate(d)
        _, first = np.unique(v * len(self.vertex_coordinates) + w, return_index=True)
        fractional = np.array(self.vertex_coordinates, dtype=float) / self.period
        return ids, fractional, (v[first], w[first], d[first] / self.period)


class GradedCell(CrystalCell):
    """A `CrystalCell` on a mesh graded toward the ions: the Kuhn grid with its
    simplices bisected where `refinement` asks (more vertices there, the first
    of them the vertices of the grid in its numbering), and its vertices moved
    by `grading`. With neither it is the `CrystalCell`.

    `fractional` and `positions` are the true positions of the vertices.
    `orientation_margin` is the smallest ratio of the volume of a top simplex
    to its volume before the vertices were moved: the mesh is folded if it is
    not positive, which the squared lengths alone would not show, and such a
    grading is refused."""

    def __init__(self, lattice, divisions, grading=None, refinement=None, **kwargs):
        self.grading, self.refinement = grading, refinement
        super().__init__(lattice, divisions, **kwargs)

    def _top_cells(self):
        mesh = KuhnBisection(self.divisions)
        if self.refinement is not None:
            place = (lambda f: f) if self.grading is None else (lambda f: self.grading.place(f, self.lattice))
            mesh.refine_to(lambda centroids: self.refinement.levels(place(centroids), self.lattice))
        cells, self._reference, (v, w, d) = mesh.arrays()
        self._midpoints = mesh.midpoints
        moved = self._reference if self.grading is None else self.grading.place(self._reference, self.lattice)
        self.vertex_offsets = (moved - self._reference) @ self.lattice
        self._moved = moved
        count = len(self._reference)
        self._edge_keys = v * count + w                          # sorted: `arrays` returns them so
        self._edge_vectors = d @ self.lattice + self.vertex_offsets[w] - self.vertex_offsets[v]
        corners = lambda offsets: [(self._edge_lookup(cells[:, 0], cells[:, c], offsets)) for c in (1, 2, 3)]
        volume = lambda vectors: np.linalg.det(np.stack(vectors, axis=1))
        reference = volume(corners(False))
        self.orientation_margin = float((volume(corners(True)) / reference).min())
        if self.orientation_margin <= 0.0:
            raise ValueError("the grading folds the mesh: a top simplex has lost its orientation")
        self.largest_volume_ratio = float(np.abs(reference).max() / np.abs(reference).min())
        return [[int(x) for x in sorted(cell)] for cell in cells]

    def _edge_lookup(self, source, target, moved=True):
        """The displacement from `source` to `target` for pairs joined by an edge (zero on the diagonal)."""
        source, target = np.asarray(source, dtype=np.int64), np.asarray(target, dtype=np.int64)
        count = len(self._reference)
        low, high = np.minimum(source, target), np.maximum(source, target)
        where = np.searchsorted(self._edge_keys, low * count + high)
        where = np.minimum(where, len(self._edge_keys) - 1)
        found = self._edge_keys[where] == low * count + high
        if np.any(~found & (source != target)):
            raise ValueError("a stored entry joins vertices that are not joined by an edge")
        vectors = self._edge_vectors[where]
        if not moved:
            v, w = self._edge_keys[where] // count, self._edge_keys[where] % count
            vectors = vectors - self.vertex_offsets[w] + self.vertex_offsets[v]
        sign = np.where(source <= target, 1.0, -1.0) * found
        return vectors * sign[:, None]

    def entry_displacements(self, rows, cols):
        return self._edge_lookup(rows, cols)

    def _vertex_fractional(self):
        return self._moved

    @property
    def edge_displacements(self):
        if not hasattr(self, "_edge_displacements"):
            edges = np.asarray(self.edges, dtype=np.int64)
            self._edge_displacements = self._edge_lookup(edges[:, 0], edges[:, 1])
        return self._edge_displacements

    def _edge_squared_lengths(self):
        return [complex(value) for value in (self.edge_displacements ** 2).sum(axis=1)]

    def _links_at(self, kappa):
        k = np.asarray(kappa, dtype=float) @ self.reciprocal
        return [complex(value) for value in np.exp(1j * (self.edge_displacements @ k))]

    def _fundamental_cycle(self, axis):
        def path(v, w):
            middle = self._midpoints.get(frozenset((v, w)))
            return [(v, w)] if middle is None else path(v, middle) + path(middle, w)
        return [step for v, w in self.grid.fundamentalCycle(axis) for step in path(int(v), int(w))]

    def spacetime(self, kappa=(0.0, 0.0, 0.0)):
        """The cell as a `Spacetime` with the graded geometry declared on its
        edges: `Edge.setLength` with the length between the moved vertices, and
        `Edge.setPhase` with the angle of the link of the crystal momentum. A
        bisected mesh has no `Topology` of its own in the library yet."""
        if self.refinement is not None:
            raise NotImplementedError("a bisected mesh is not a Topology of the library; it is assembled from its "
                                      "chain complex and squared lengths")
        spacetime = super().spacetime(kappa)
        k = np.asarray(kappa, dtype=float) @ self.reciprocal
        for edge in spacetime.getEdgeList().toVector():
            displacement = self._edge_lookup([edge.getSource().getId()], [edge.getTarget().getId()])[0]
            edge.setLength(float(np.linalg.norm(displacement)))
            edge.setPhase(float(displacement @ k))
        return spacetime


# ---------------------------------------------------------------- the Coulomb kernel without translation invariance

class GradedCoulombKernel:
    """`coulomb.GridCoulombKernel` on a cell whose stiffness matrix does not
    commute with the grid translations, with the same meaning of every method
    and the same values on the Kuhn grid.

    The kernel on charges of the crystal momentum q is strength * A(q)^-1, the
    inverse of the stiffness matrix dressed by that momentum, applied through a
    sparse factorization (positive definite for q != 0; at the zone centre the
    constants are the kernel of A, one vertex is pinned, the load is neutralized
    by the uniform background and the potential is returned with zero mean, as
    `coulomb.CoulombKernel` does). What the Fourier transform named by the
    wavevector G = 0 is named here by what it is:

    - the G = 0 component of a load is its total charge 1^T rho;
    - the G = 0 entry of the kernel is the energy of the uniform normalized
      charge of momentum q, e(q) = strength * w^T A(q)^-1 w / V^2 with w the
      vertex weights (`momentum_entry`), which is strength / (n a(q)) on the
      grid; the kernel "with that entry replaced by z" is
      strength * A(q)^-1 + (z - e(q)) 1 1^T, an identity, so nothing downstream
      is approximated by the split;
    - the auxiliary function F(q) = (strength / n) sum_G 1 / a(G + q) is the
      energy of a unit point load, strength * (A(q)^-1)_pp, the same at every
      vertex p of the grid. Off the grid it depends on the vertex at second
      order in the mesh spacing, because F(q) averaged over q minus F(0) is the
      interaction with the periodic images, which is smooth at the load; it is
      taken at the `probes` (default: the vertex of largest weight, where the
      mesh is coarsest and farthest from an ion).
    """

    def __init__(self, cell, strength=4.0 * np.pi * E2, probes=None, factorizations=2):
        self.cell, self.strength, self.size = cell, float(strength), cell.size
        self.weights = np.asarray(cell.mass.dressed().real.sum(axis=1)).ravel()
        self.volume = float(self.weights.sum())
        self.probes = [int(np.argmax(self.weights))] if probes is None else [int(p) for p in probes]
        self._limit, self._factors = int(factorizations), {}
        stiffness = cell.stiffness.dressed().real.tocsc()
        self._centre = self._factorize(stiffness[1:, 1:])

    @staticmethod
    def _factorize(matrix):
        # Symmetric ordering and no pivoting: the matrices are positive definite.
        return spla.splu(sp.csc_matrix(matrix), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                         options=dict(SymmetricMode=True))

    @staticmethod
    def _key(kappa):
        if kappa is None:
            return None
        key = tuple(float(x) for x in kappa)
        return key if any(key) else None

    def _solve(self, key, columns):
        """A(q)^-1 columns; at the zone centre the pseudo-inverse on neutral loads, of zero mean."""
        if key is None:
            neutral = columns - np.outer(self.weights, columns.sum(axis=0)) / self.volume
            solved = np.zeros(columns.shape, dtype=complex)
            solved[1:] = self._centre.solve(neutral[1:].real) + 1j * self._centre.solve(neutral[1:].imag)
            return solved - (self.weights @ solved)[None, :] / self.volume
        if key not in self._factors:
            while len(self._factors) >= self._limit:
                self._factors.pop(next(iter(self._factors)))
            self._factors[key] = self._factorize(self.cell.stiffness.dressed(key))
        return self._factors[key].solve(columns)

    def momentum_entry(self, kappa):
        """e(q): the energy of the uniform normalized charge of momentum q;
        strength / (V q^2) in the continuum."""
        key = self._key(kappa)
        if key is None:
            raise ValueError("the entry at zero momentum is the zero-momentum constant")
        w = self.weights.astype(complex).reshape(-1, 1)
        return self.strength * float(np.real(np.vdot(w, self._solve(key, w)))) / self.volume ** 2

    def momentum_entry_limit(self, direction):
        """The limit of `momentum_entry` times q^2 as the momentum tends to zero
        along `direction`: strength / (q^ . S q^) with S = -1/2 sum_vw A_vw
        dx_vw dx_vw^T, the stiffness of the linear functions, which
        piecewise-linear elements carry exactly on any mesh: S = V."""
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        along = self.cell.stiffness.displacement @ direction
        return self.strength / float(-0.5 * np.real(self.cell.stiffness.data) @ along ** 2)

    def potential(self, rho, kappa=None, zero_momentum=None):
        """The potential of the load vectors `rho` (columns) at the crystal
        momentum `kappa` (the zone centre when None); `zero_momentum` replaces
        the energy of the uniform normalized charge by that value."""
        rho = np.asarray(rho, dtype=complex)
        columns = rho.reshape(self.size, -1)
        key = self._key(kappa)
        solved = self.strength * self._solve(key, columns)
        if zero_momentum is not None:
            entry = 0.0 if key is None else self.momentum_entry(key)
            solved = solved + (zero_momentum - entry) * columns.sum(axis=0)[None, :]
        return solved.reshape(rho.shape)

    def energy(self, rho_a, rho_b=None):
        rho_b = rho_a if rho_b is None else rho_b
        return np.vdot(rho_a, self.potential(rho_b))

    def potential_derivative(self, rho, axis):
        """The derivative of `potential` with respect to the Cartesian component
        `axis` of the crystal momentum at the zone centre,
        -strength * A^+ (dA / dk) A^+ rho, with dA / dk the derivative of the
        dressed stiffness matrix along a uniform change of the link phases
        (`CovariantChainHodge.sparsePencilPhaseDerivativeAlong` with the edge
        displacements as weights)."""
        if not hasattr(self, "_derivatives"):
            self._derivatives = {}
        if axis not in self._derivatives:
            weights = [float(w) for w in self.cell.edge_displacements[:, axis]]
            self._derivatives[axis] = sp.csc_matrix(self.cell.covariant().sparsePencilPhaseDerivativeAlong(weights).A)
        rho = np.asarray(rho, dtype=complex)
        columns = rho.reshape(self.size, -1)
        inner = self._derivatives[axis] @ self._solve(None, columns)
        return (-self.strength * self._solve(None, inner)).reshape(rho.shape)

    def auxiliary_function(self, kappa=None):
        """F(q): the energy of a unit point load at the probes (their mean)."""
        loads = np.zeros((self.size, len(self.probes)), dtype=complex)
        loads[self.probes, np.arange(len(self.probes))] = 1.0
        solved = self._solve(self._key(kappa), loads)
        return self.strength * float(np.mean(np.real(solved[self.probes, np.arange(len(self.probes))])))

    def zero_momentum_constant(self, refinements=(2, 3, 4, 6, 8), transfers=None):
        """The term a cell sampled at its zone centre leaves out at zero momentum
        transfer, for a normalized charge: the average of the auxiliary function
        over every momentum minus its value at the sampled momenta,

            c = < F(q) >_q - (1 / N) sum_t F(t) ,

        `transfers` being the momentum transfers of a momentum set (the zero
        transfer alone by default). The average has the integrable singularity
        strength / (V q^2), taken analytically as on the grid: g(q) = (1 / V)
        sum_K exp(-alpha |q + K|^2) / |q + K|^2 over the reciprocal lattice of
        the cell has the same singularity and the average 1 / (4 pi^(3/2)
        sqrt(alpha)); the remainder is bounded with a jump at the origin only,
        and its mean over grids of m^3 momenta errs by odd inverse powers of m
        from the third on, which the `refinements` remove. F(-q) = F(q), so half
        of the momenta are factorized."""
        cell = self.cell
        alpha = 0.25 * cell.spacing ** 2
        reach = [int(np.ceil(np.sqrt(40.0 / alpha) / np.linalg.norm(b))) + 1 for b in cell.reciprocal]
        K = np.stack(np.meshgrid(*[np.arange(-n, n + 1) for n in reach], indexing="ij"), axis=-1).reshape(-1, 3)

        def singular(q):
            k2 = (((K + q) @ cell.reciprocal) ** 2).sum(axis=1)
            k2 = k2[k2 > 1e-24]
            return float((np.exp(-alpha * k2) / k2).sum()) / self.volume

        means = []
        for m in refinements:
            total = 0.0
            for j in itertools.product(range(m), repeat=3):
                partner = tuple((-x) % m for x in j)
                if j > partner:
                    continue
                q = np.array(j, dtype=float) / m
                value = self.auxiliary_function(q) / self.strength - singular(q)
                total += value if j == partner else 2.0 * value
            means.append(total / m ** 3)
        m = np.array(refinements, dtype=float)
        design = np.column_stack([np.ones_like(m)] + [1.0 / m ** power for power in (3, 5, 7, 9)][:len(m) - 1])
        average = np.linalg.lstsq(design, np.array(means), rcond=None)[0][0] + 1.0 / (4.0 * np.pi ** 1.5 * np.sqrt(alpha))
        transfers = np.zeros((1, 3)) if transfers is None else np.atleast_2d(np.asarray(transfers, dtype=float))
        sampled = sum(self.auxiliary_function(t) for t in transfers) / len(transfers)
        return self.strength * average - sampled


# ---------------------------------------------------------------- the confined pseudo-atom on a cell

def pseudo_atom_levels(cell, pseudo, strength, count, center=(0.5, 0.5, 0.5), tolerance=1e-9):
    """The lowest `count` levels (rydberg) of the screened, confined pseudo-atom
    of `pseudopotential.screened_potential` at the fractional `center` of a
    cell in bohr with `kinetic_scale = 1`: the mesh counterpart of
    `pseudopotential.radial_levels`, on any cell. The local potential and the
    projector functions are taken at the true positions of the vertices."""
    from tessera.drivers.bands.abinitio import solve_with_projectors
    from tessera.drivers.bands.pseudopotential import real_harmonics, screened_potential
    offset = cell.fractional - np.asarray(center, dtype=float)
    offset -= np.rint(offset)
    offset = offset @ cell.lattice
    radius = np.linalg.norm(offset, axis=1)
    potential = screened_potential(pseudo, radius, strength)
    A, M = cell.pencil((0.0, 0.0, 0.0), potential)
    columns, spans = [], []
    for i, (l, _) in enumerate(pseudo.projectors):
        radial = pseudo.projector_at(i, radius)
        spans.append((i, l, len(columns)))
        columns += [radial * harmonic for harmonic in real_harmonics(l, offset)]
    D = np.zeros((len(columns), len(columns)))
    for i, l, start_i in spans:
        for j, l2, start_j in spans:
            if l == l2 and pseudo.D[i, j] != 0.0:
                for m in range(2 * l + 1):
                    D[start_i + m, start_j + m] = pseudo.D[i, j]
    P = M.real @ np.array(columns).T
    # The separable term is bounded below by its most negative coefficient times the projector norms.
    floor = float(potential.min()) + min(0.0, float(np.linalg.eigvalsh(D).min())) * float(
        np.linalg.eigvalsh(np.array(columns) @ P).max())
    values, _, residual, below = solve_with_projectors(A.real, M.real, P, D, count, floor - 1.0, tolerance)
    return values, residual, below
