"""The Z/10-symmetric genus-2 surface: the regular decagon with opposite
sides identified, triangulated by the barycentric subdivision of its cone.

``y^2 = x^5 - 1`` has the automorphism ``x -> zeta x``; its symmetric model
is the decagon ``w_0 ... w_9`` with side ``E_{i+5}`` glued to ``E_i``
reversed. The gluing has two vertex classes (``p``, the even corners; ``q``,
the odd), five side classes ``e_0 ... e_4`` and one face: ``chi = 2 - 5 + 1
= -2``. Coning from the centre ``c`` and subdividing barycentrically gives
a simplicial complex with 28 vertices (``c``, ``p``, ``q``, the five side
midpoints, the ten spoke midpoints, the ten face barycentres), 90 edges and
60 faces. The rotation ``r`` by ``2 pi / 10`` permutes the cells, so it is
a simplicial automorphism of order ten; ``r^2`` has order five and fixes
``c``, ``p`` and ``q``; ``r^5`` is the hyperelliptic involution, ``-1`` on
homology. The metric is the flat decagon's, so every ``r^k`` is an
isometry.

``H_1 = Z^4`` is spanned by the loops ``z_i = e_i + e_{i+1}`` (``i = 0..3``),
all based at ``p``; a symplectic marking ``(A_1, A_2 | B_1, B_2)`` is
extracted from them by integer symplectic reduction of the intersection
form, which ``SurfacePeriods`` reads from the surface itself.
"""
from __future__ import annotations

import math

import numpy as np

from .surface_periods import SurfacePeriods, _steps

__all__ = ["SymmetricGenusTwo"]


class SymmetricGenusTwo:
    """Vertex ids: ``0`` = ``c``, ``1`` = ``p``, ``2`` = ``q``, ``3..7`` =
    midpoints of ``e_0 .. e_4``, ``8..17`` = midpoints of the spokes
    ``s_0 .. s_9``, ``18..27`` = barycentres of the faces ``t_0 .. t_9``."""

    CENTRE, P, Q = 0, 1, 2

    def __init__(self, radius=1.0):
        self.radius = float(radius)
        self.faces = []          # oriented (counterclockwise) vertex triples
        self._build()
        self._metric()
        self._marking()

    # ------------------------------------------------------------ cells
    @staticmethod
    def side_mid(j):
        return 3 + (j % 5)

    @staticmethod
    def spoke_mid(i):
        return 8 + (i % 10)

    @staticmethod
    def barycentre(i):
        return 18 + (i % 10)

    @classmethod
    def corner(cls, i):
        """The decagon corner ``w_i`` as a vertex class: ``p`` for even ``i``."""
        return cls.P if i % 2 == 0 else cls.Q

    def _build(self):
        """The six triangles of each sector ``t_i = (c, w_i, w_{i+1})``,
        counterclockwise in the sector's own plane, with the identified
        side ``E_i`` and its midpoint ``m(e_{i mod 5})``."""
        faces = []
        for i in range(10):
            c, wi, wj = self.CENTRE, self.corner(i), self.corner(i + 1)
            b = self.barycentre(i)
            m_ci, m_cj, m_ij = self.spoke_mid(i), self.spoke_mid(i + 1), self.side_mid(i)
            # the barycentric subdivision of (c, w_i, w_{i+1}), each
            # triangle (b, vertex, midpoint) oriented counterclockwise
            faces.extend([(b, c, m_ci), (b, m_ci, wi), (b, wi, m_ij),
                          (b, m_ij, wj), (b, wj, m_cj), (b, m_cj, c)])
        self.faces = faces
        self.vertices = sorted({v for face in faces for v in face})
        assert self.vertices == list(range(28))
        self.edges = sorted({(min(u, v), max(u, v)) for face in faces for u, v in _steps(face)})
        assert len(self.edges) == 90 and len(faces) == 60
        # each edge in exactly two faces, oppositely traversed
        seen = {}
        for face in faces:
            for u, v in _steps(face):
                seen.setdefault((min(u, v), max(u, v)), []).append((u, v))
        for key, traversals in seen.items():
            assert len(traversals) == 2 and traversals[0] == traversals[1][::-1], key

    @property
    def euler_characteristic(self):
        return len(self.vertices) - len(self.edges) + len(self.faces)

    # ------------------------------------------------------------ metric
    def _metric(self):
        """Every sector is the same Euclidean isosceles triangle
        (circumradius ``radius``, apex ``2 pi / 10``); its barycentric
        points are placed in the plane and each edge's length is the
        distance of its endpoints. Identified edges get equal lengths by
        the symmetry, so ``lengths`` is well defined."""
        r = self.radius
        step = 2 * math.pi / 10
        lengths = {}
        for i in range(10):
            c = np.zeros(2)
            wi = r * np.array([math.cos(i * step), math.sin(i * step)])
            wj = r * np.array([math.cos((i + 1) * step), math.sin((i + 1) * step)])
            position = {self.CENTRE: c, self.corner(i): wi, self.corner(i + 1): wj,
                        self.barycentre(i): (c + wi + wj) / 3.0,
                        self.spoke_mid(i): (c + wi) / 2.0, self.spoke_mid(i + 1): (c + wj) / 2.0,
                        self.side_mid(i): (wi + wj) / 2.0}
            for face in self.faces[6 * i:6 * i + 6]:
                for u, v in _steps(face):
                    key = (min(u, v), max(u, v))
                    length = float(np.linalg.norm(position[u] - position[v]))
                    if key in lengths and abs(lengths[key] - length) > 1e-12:
                        raise RuntimeError("the metric is not symmetric on edge %r" % (key,))
                    lengths[key] = length
        self.lengths = lengths

    def spacetime(self):
        """The surface as a 2-dimensional ``Spacetime`` with its lengths and
        zero phases (the ``seedCollar`` input)."""
        import tessera as T
        cells = [list(face) for face in self.faces]
        surface = T.spacetime.Spacetime.fromVertexTuples(2, cells, 1.0, 0.0)
        for edge in surface.getEdgeList().toVector():
            u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
            edge.setLength(self.lengths[(min(u, v), max(u, v))])
            edge.setPhase(0.0)
        return surface

    # ------------------------------------------------------------ symmetry
    def rotation(self, k=1):
        """The vertex permutation of ``r^k`` (``r`` the rotation by
        ``2 pi / 10``): ``perm[v]`` is the image of vertex ``v``."""
        k = int(k) % 10
        perm = list(range(28))
        if k % 2:
            perm[self.P], perm[self.Q] = self.Q, self.P
        for j in range(5):
            perm[self.side_mid(j)] = self.side_mid(j + k)
        for i in range(10):
            perm[self.spoke_mid(i)] = self.spoke_mid(i + k)
            perm[self.barycentre(i)] = self.barycentre(i + k)
        return perm

    def is_automorphism(self, perm):
        faces = {frozenset(face) for face in self.faces}
        return all(frozenset(perm[v] for v in face) in faces for face in self.faces)

    # ------------------------------------------------------------ homology
    def side_walk(self, j, reverse=False):
        """The side class ``e_j`` (``j = 0..4``) as two steps through its
        midpoint, from ``w_j`` to ``w_{j+1}``; reversed on request."""
        j = int(j) % 5
        a, b = self.corner(j), self.corner(j + 1)
        m = self.side_mid(j)
        walk = [(a, m), (m, b)]
        return [(v, u) for u, v in reversed(walk)] if reverse else walk

    def loop(self, i):
        """``z_i = e_i + e_{i+1}`` as a closed walk based at ``p``."""
        i = int(i)
        if i % 2 == 0:      # e_i: p -> q, e_{i+1}: q -> p
            return self.side_walk(i) + self.side_walk(i + 1)
        return self.side_walk(i + 1) + self.side_walk(i)  # e_{i+1}: p -> q, then e_i: q -> p

    def loop_coefficients(self, i):
        """``z_i`` in the side basis ``e_0..e_4``."""
        out = np.zeros(5, dtype=int)
        out[i % 5] += 1
        out[(i + 1) % 5] += 1
        return out

    def rotation_on_sides(self, k=1):
        """The matrix of ``r^k`` on the side classes: ``r e_j = e_{j+1}`` for
        ``j < 4`` and ``r e_4 = -e_0`` (``E_5 = e_0^{-1}``)."""
        r = np.zeros((5, 5), dtype=int)
        for j in range(5):
            if j < 4:
                r[j + 1, j] = 1
            else:
                r[0, 4] = -1
        return np.linalg.matrix_power(r, int(k) % 10)

    def combined_walk(self, coefficients):
        """A closed walk at ``p`` for ``sum_i n_i z_i``: the loops
        concatenated, reversed for negative coefficients."""
        walk = []
        for i, n in enumerate(coefficients):
            loop = self.loop(i)
            for _ in range(abs(int(n))):
                walk.extend(loop if n > 0 else [(v, u) for u, v in reversed(loop)])
        return walk

    def _marking(self):
        """The intersection form of ``z_0..z_3`` from the surface, and the
        integer symplectic basis ``(A_1, A_2 | B_1, B_2)``."""
        loops = [self.loop(i) for i in range(4)]
        provisional = SurfacePeriods(self.faces, self.lengths, loops[:2], loops[2:], root_face=self.faces[0])
        self.intersection, self.intersection_residual = provisional.intersection_form(loops)
        self.symplectic_basis = _symplectic_reduction(self.intersection)   # rows: coefficients in z_0..z_3
        basis = self.symplectic_basis
        self.a_cycles = [self.combined_walk(basis[0]), self.combined_walk(basis[1])]
        self.b_cycles = [self.combined_walk(basis[2]), self.combined_walk(basis[3])]
        self.periods = SurfacePeriods(self.faces, self.lengths, self.a_cycles, self.b_cycles,
                                      root_face=self.faces[0])

    def marking(self):
        """``[A_1, B_1, A_2, B_2]`` as closed walks of vertex steps (the
        pair ordering `MultiCobordism.restriction` takes)."""
        return [self.a_cycles[0], self.b_cycles[0], self.a_cycles[1], self.b_cycles[1]]

    def induced_map(self, k=1):
        """The matrix of ``r^k`` on ``H_1`` in the symplectic basis, in the
        register's ordering ``(A_1, A_2, B_1, B_2)``: rows are the images.
        Read through periods, ``P(r^k A) = M P``, so this is the monodromy
        a collar twisted by ``r^k`` reads."""
        rot = self.rotation_on_sides(k)
        # z_i in side coordinates, images of z_i, then back to z coordinates
        z = np.array([self.loop_coefficients(i) for i in range(4)])          # (4, 5)
        images = (rot @ z.T).T                                              # (4, 5): r^k z_i
        # solve images = X z  (X integer) with the 4 independent z_i
        x = np.linalg.lstsq(z.T.astype(float), images.T.astype(float), rcond=None)[0].T
        if np.abs(x - np.rint(x)).max() > 1e-9:
            raise RuntimeError("r^%d does not preserve the span of z_0..z_3" % k)
        on_z = np.rint(x).astype(int)                                       # r^k z_i = sum_j on_z[i, j] z_j
        basis = self.symplectic_basis                                       # rows: A_1, A_2, B_1, B_2 in z coords
        inverse = np.rint(np.linalg.inv(basis.astype(float))).astype(int)
        # r^k(basis_i) = basis_i . on_z (as a row in z coords) = (basis_i on_z inverse) in basis coords
        return basis @ on_z @ inverse

    def report(self):
        rep = self.periods.report()
        rep.update({"euler_characteristic": self.euler_characteristic,
                    "intersection_form_z": self.intersection.tolist(),
                    "intersection_residual": self.intersection_residual,
                    "symplectic_basis_z": self.symplectic_basis.tolist()})
        return rep


def _symplectic_reduction(form):
    """An integer basis ``(a_1 .. a_g, b_1 .. b_g)`` of ``Z^{2g}`` with
    ``a_i . b_j = delta_ij`` and all other products zero, for a unimodular
    antisymmetric integer ``form``; rows are the basis vectors in the given
    coordinates. Symplectic Gram-Schmidt over the integers."""
    form = np.asarray(form, dtype=int)
    n = form.shape[0]
    remaining = [np.eye(n, dtype=int)[i] for i in range(n)]
    pairs = []
    while remaining:
        a = remaining.pop(0)
        products = [int(a @ form @ v) for v in remaining]
        if all(x == 0 for x in products):
            if any(v.any() for v in remaining):
                raise ValueError("the form is degenerate on the remaining lattice")
            break
        # an integer combination b of the remaining with a . b = gcd = 1
        g, coefficients = _extended_gcd_combination(products)
        if abs(g) != 1:
            raise ValueError("the form is not unimodular: gcd %d" % g)
        b = sum(int(c) * v for c, v in zip(coefficients, remaining))
        if int(a @ form @ b) == -1:
            b = -b
        pairs.append((a, b))
        # project the rest onto the symplectic complement of (a, b)
        new_remaining = []
        for v in remaining:
            w = v - int(v @ form @ b) * a + int(v @ form @ a) * b
            if w.any():
                new_remaining.append(w)
        # keep an independent set
        remaining = _independent(new_remaining)
    basis = np.array([p[0] for p in pairs] + [p[1] for p in pairs], dtype=int)
    check = basis @ form @ basis.T
    g = len(pairs)
    expected = np.block([[np.zeros((g, g), dtype=int), np.eye(g, dtype=int)], [-np.eye(g, dtype=int), np.zeros((g, g), dtype=int)]])
    if not (check == expected).all():
        raise RuntimeError("symplectic reduction failed: %r" % (check.tolist(),))
    if abs(int(round(np.linalg.det(basis.astype(float))))) != 1:
        raise RuntimeError("the symplectic basis is not unimodular")
    return basis


def _extended_gcd_combination(values):
    """``g = gcd(values)`` and integer coefficients with ``sum c_i v_i = g``."""
    g, coefficients = 0, []
    for value in values:
        if g == 0:
            g, coefficients = abs(int(value)), coefficients + [1 if value >= 0 else -1]
            if value == 0:
                coefficients[-1] = 0
            continue
        old_g = g
        x, y, g = _egcd(old_g, int(value))
        coefficients = [c * x for c in coefficients] + [y]
    return g, coefficients


def _egcd(a, b):
    """``x, y, g`` with ``a x + b y = g = gcd(a, b) >= 0``."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b != 0:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        a, x0, y0 = -a, -x0, -y0
    return x0, y0, a


def _independent(vectors):
    """A maximal linearly independent subset (over Q) in the given order."""
    kept = []
    for v in vectors:
        trial = np.array(kept + [v], dtype=float)
        if np.linalg.matrix_rank(trial) == len(kept) + 1:
            kept.append(v)
    return kept
