"""The Riemann period matrix of a closed, oriented, triangulated surface from
its own lengths: the genus-``g`` form of ``SimplicialQubit`` sections 3-9.

A boundary surface of a cobordism carries an intrinsic complex structure --
its harmonic 1-forms with the Hodge star of its own metric -- and that
structure is what the theta quantization register polarizes. For a torus
``SimplicialQubit`` reads the one number the structure has, the modulus
``tau``; for a surface of genus ``g`` the same construction reads the
``g x g`` period matrix ``Omega`` over a symplectic marking
``(A_1..A_g, B_1..B_g)``:

- cotangent weights on the edges and the harmonic space
  ``H = ker[d_1; d_0^T diag(w)]`` (closed and co-closed), of dimension ``2g``;
- the complex structure on ``H`` from the Whitney interpolants at the
  barycentres, ``J = G^{-1} R^T`` with ``G_ab = sum_t area_t <W_t h_a, W_t h_b>``
  and ``R_ab = sum_t area_t <rot90 W_t h_a, W_t h_b>``;
- the holomorphic differentials, the ``-i`` eigenspace of ``J``;
- the periods ``P_A``, ``P_B`` as signed sums along the marking cycles, and
  ``Omega = P_B P_A^{-1}``.

Everything is the real-locus construction: real lengths, real ``J``; a
complex length is refused by name. The faces are oriented consistently by
propagation from a root face, so ``J`` is one complex structure on the whole
surface; with the marking symplectic for that orientation ``Im Omega`` is
positive definite. If it comes out negative definite the marking is
symplectic for the opposite orientation and the conjugate structure is
taken (``orientation_flipped``); indefinite means the marking is not
symplectic for either, which is reported, not repaired.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

__all__ = ["SurfacePeriods"]


def _rot90(v):
    return np.array([-v[1], v[0]])


class SurfacePeriods:
    """The period matrix of a closed triangulated surface.

    Parameters
    ----------
    faces
        The triangles as vertex-id triples. Their orientation is NOT taken
        from the order given: one consistent orientation is propagated from
        ``root_face``.
    lengths
        ``{(u, v): length}`` for every edge (``u < v``); real, positive.
    a_cycles, b_cycles
        The marking: ``g`` cycles each, as lists of directed steps
        ``(u, v)`` along edges of the surface, in the order ``A_i`` pairs
        with ``B_i``. A step contributes ``+omega(u, v)`` when ``u < v`` and
        ``-omega(u, v)`` otherwise, the ``MultiCobordism.monodromy``
        convention.
    root_face
        A triple ``(u, v, w)`` naming a face WITH the cyclic order that is
        to count as positive; the orientation of every other face follows.
        Default: the first face as given.
    """

    def __init__(self, faces, lengths, a_cycles, b_cycles, root_face=None, tolerance=1e-12):
        self.tolerance = float(tolerance)
        self.faces = [tuple(int(v) for v in face) for face in faces]
        if not self.faces:
            raise ValueError("SurfacePeriods: no faces")
        if len(a_cycles) != len(b_cycles):
            raise ValueError("SurfacePeriods: %d A cycles against %d B cycles" % (len(a_cycles), len(b_cycles)))
        self.genus = len(a_cycles)
        self.warnings = []
        self._orient(root_face)
        self._index()
        self._lengths(lengths)
        self._geometry()
        self._weights()
        self._harmonic()
        self._complex_structure()
        self._periods(a_cycles, b_cycles)

    # ------------------------------------------------------------ orientation
    def _orient(self, root_face):
        """One consistent orientation from the root, by propagation across
        shared edges (each edge traversed oppositely by its two faces)."""
        by_edge = defaultdict(list)
        for index, face in enumerate(self.faces):
            if len(face) != 3 or len(set(face)) != 3:
                raise ValueError("SurfacePeriods: face %r is not a triangle" % (face,))
            for a in range(3):
                u, v = face[a], face[(a + 1) % 3]
                by_edge[(min(u, v), max(u, v))].append(index)
        for edge, owners in by_edge.items():
            if len(owners) != 2:
                raise ValueError("SurfacePeriods: edge %r belongs to %d faces; the surface is not closed"
                                 % (edge, len(owners)))
        root = 0
        oriented = [None] * len(self.faces)
        if root_face is not None:
            wanted = tuple(int(v) for v in root_face)
            for index, face in enumerate(self.faces):
                if set(face) == set(wanted):
                    root = index
                    oriented[root] = wanted
                    break
            else:
                raise ValueError("SurfacePeriods: the root face %r is no face of the surface" % (wanted,))
        else:
            oriented[root] = self.faces[root]
        queue = deque([root])
        while queue:
            index = queue.popleft()
            face = oriented[index]
            for a in range(3):
                u, v = face[a], face[(a + 1) % 3]
                for other in by_edge[(min(u, v), max(u, v))]:
                    if other == index:
                        continue
                    candidate = self.faces[other]
                    # The neighbour must traverse (v, u): rotate it so that
                    # v comes first and u second, reversing if necessary.
                    order = list(candidate)
                    if not _traverses(order, v, u):
                        order = [order[0], order[2], order[1]]
                    assert _traverses(order, v, u)
                    order = tuple(order)
                    if oriented[other] is None:
                        oriented[other] = order
                        queue.append(other)
                    elif oriented[other] != order and oriented[other] not in _rotations(order):
                        raise ValueError("SurfacePeriods: the surface is not orientable (face %r)" % (candidate,))
        if any(face is None for face in oriented):
            raise ValueError("SurfacePeriods: the faces do not form one connected surface")
        self.oriented = oriented

    # ------------------------------------------------------------ indexing
    def _index(self):
        vertices = sorted({v for face in self.faces for v in face})
        self.vertices = vertices
        self.vertex_index = {v: n for n, v in enumerate(vertices)}
        edges = sorted({(min(u, v), max(u, v)) for face in self.faces for u, v in _steps(face)})
        self.edges = edges
        self.edge_index = {e: n for n, e in enumerate(edges)}
        # Incidence, ``SimplicialQubit`` section 3: d0 (E x V) with -1 at the
        # smaller id and +1 at the larger; d1 (F x E) with the traversal sign.
        n_v, n_e, n_f = len(vertices), len(edges), len(self.faces)
        d0 = np.zeros((n_e, n_v))
        for n, (u, v) in enumerate(edges):
            d0[n, self.vertex_index[u]] = -1.0
            d0[n, self.vertex_index[v]] = 1.0
        d1 = np.zeros((n_f, n_e))
        for t, face in enumerate(self.oriented):
            for u, v in _steps(face):
                d1[t, self.edge_index[(min(u, v), max(u, v))]] = 1.0 if u < v else -1.0
        if np.abs(d1 @ d0).max() != 0.0:
            raise RuntimeError("SurfacePeriods: d1 d0 != 0")
        self.d0, self.d1 = d0, d1

    def _lengths(self, lengths):
        values = np.empty(len(self.edges))
        for n, edge in enumerate(self.edges):
            try:
                length = lengths[edge]
            except KeyError:
                raise ValueError("SurfacePeriods: no length for edge %r" % (edge,)) from None
            length = complex(length)
            if abs(length.imag) > self.tolerance * max(1.0, abs(length.real)):
                raise ValueError("SurfacePeriods: edge %r has a complex length %r; the reader is the real-locus "
                                 "construction" % (edge, length))
            if not length.real > 0.0:
                raise ValueError("SurfacePeriods: edge %r has a non-positive length %r" % (edge, length))
            values[n] = length.real
        self.lengths = values

    # ------------------------------------------------------------ geometry
    def _geometry(self):
        """Per oriented face (i, j, k): angles, area, the planar layout with
        i at the origin, j on the x-axis and k above (counterclockwise), and
        the barycentric gradients -- ``SimplicialQubit`` sections 4 and 7."""
        n_f = len(self.faces)
        self.angles = np.empty((n_f, 3))
        self.areas = np.empty(n_f)
        self.gradients = np.empty((n_f, 3, 2))
        for t, (i, j, k) in enumerate(self.oriented):
            a = self.lengths[self.edge_index[(min(j, k), max(j, k))]]
            b = self.lengths[self.edge_index[(min(k, i), max(k, i))]]
            c = self.lengths[self.edge_index[(min(i, j), max(i, j))]]
            angle = lambda cosine: math.acos(max(-1.0, min(1.0, cosine)))  # noqa: E731
            alpha = (angle((b * b + c * c - a * a) / (2 * b * c)),
                     angle((c * c + a * a - b * b) / (2 * c * a)),
                     angle((a * a + b * b - c * c) / (2 * a * b)))
            s = 0.5 * (a + b + c)
            area2 = s * (s - a) * (s - b) * (s - c)
            if not area2 > 0.0:
                raise ValueError("SurfacePeriods: face %r is degenerate (%g, %g, %g)" % ((i, j, k), a, b, c))
            area = math.sqrt(area2)
            p_i = np.array([0.0, 0.0])
            p_j = np.array([c, 0.0])
            p_k = np.array([b * math.cos(alpha[0]), b * math.sin(alpha[0])])
            self.angles[t] = alpha
            self.areas[t] = area
            self.gradients[t, 0] = _rot90(p_k - p_j) / (2 * area)
            self.gradients[t, 1] = _rot90(p_i - p_k) / (2 * area)
            self.gradients[t, 2] = _rot90(p_j - p_i) / (2 * area)
        defect = np.abs(self.gradients.sum(axis=1)).max()
        scale = max(1.0, float(np.abs(self.gradients).sum(axis=(1, 2)).max()))
        if defect > 1e-10 * scale:
            raise RuntimeError("SurfacePeriods: the barycentric gradients of a face do not sum to zero")

    def _weights(self):
        """Cotangent weights ``w_e = (cot alpha_e + cot beta_e) / 2``
        (``SimplicialQubit`` section 5); a violated intrinsic Delaunay
        condition is a warning, as there."""
        weights = np.zeros(len(self.edges))
        angle_sums = np.zeros(len(self.edges))
        for t, face in enumerate(self.oriented):
            for slot in range(3):
                u, v = face[slot], face[(slot + 1) % 3]
                e = self.edge_index[(min(u, v), max(u, v))]
                opposite = (slot + 2) % 3
                theta = self.angles[t, opposite]
                weights[e] += 0.5 * math.cos(theta) / math.sin(theta)
                angle_sums[e] += theta
        scale = max(1.0, float(np.abs(weights).max()))
        negative = int((weights < -1e-12 * scale).sum())
        non_delaunay = int((angle_sums > math.pi + 1e-12).sum())
        if negative or non_delaunay:
            self.warnings.append("%d edge(s) violate the intrinsic Delaunay condition and %d cotangent "
                                 "weight(s) are negative" % (non_delaunay, negative))
        self.weights = weights

    # ------------------------------------------------------------ harmonic space
    def _harmonic(self):
        """``H = null_space([d1; d0^T diag(w)])`` with scipy's rcond rule
        (``SimplicialQubit`` section 6); rank ``2g`` expected."""
        system = np.vstack([self.d1, self.d0.T @ np.diag(self.weights)])
        _, sigma, vt = np.linalg.svd(system)
        eps = np.finfo(float).eps
        threshold = eps * max(system.shape) * (sigma[0] if sigma.size else 0.0)
        rank = int((sigma > threshold).sum())
        self.harmonic = vt[rank:].T.copy()
        self.harmonic_rank = int(self.harmonic.shape[1])
        if self.harmonic_rank != 2 * self.genus:
            raise ValueError("SurfacePeriods: dim H = %d for a marking of genus %d (expected %d); the surface "
                             "or the weights are not what the marking says"
                             % (self.harmonic_rank, self.genus, 2 * self.genus))

    def _whitney(self, t, omega):
        """``W_t(omega)``, the Whitney interpolant of the 1-cochain at the
        barycentre of face ``t`` (``SimplicialQubit`` section 7)."""
        face = self.oriented[t]
        w = np.zeros(2)
        for slot in range(3):
            u, v = face[slot], face[(slot + 1) % 3]
            value = (1.0 if u < v else -1.0) * omega[self.edge_index[(min(u, v), max(u, v))]]
            w += value * (self.gradients[t, (slot + 1) % 3] - self.gradients[t, slot])
        return w / 3.0

    def _complex_structure(self):
        """``J = G^{-1} R^T`` on the harmonic space and its ``-i`` eigenspace,
        the holomorphic differentials (``SimplicialQubit`` sections 8-9)."""
        n = self.harmonic_rank
        gram = np.zeros((n, n))
        rot = np.zeros((n, n))
        for t in range(len(self.faces)):
            w = [self._whitney(t, self.harmonic[:, a]) for a in range(n)]
            for a in range(n):
                for b in range(n):
                    gram[a, b] += self.areas[t] * (w[a] @ w[b])
                    rot[a, b] += self.areas[t] * (_rot90(w[a]) @ w[b])
        self.gram, self.rotation = gram, rot
        self.J = np.linalg.solve(gram, rot.T)
        self.j_residual = float(np.linalg.norm(self.J @ self.J + np.eye(n)))
        values, vectors = np.linalg.eig(self.J)
        order = np.argsort(np.abs(values + 1j))
        picked = order[:self.genus]
        self.eigenvalues = values[picked]
        self.holomorphic = self.harmonic @ vectors[:, picked]     # (E, g)

    # ------------------------------------------------------------ periods
    def _periods(self, a_cycles, b_cycles):
        def period(cycle, omega):
            total = 0.0 + 0.0j
            for u, v in cycle:
                u, v = int(u), int(v)
                key = (min(u, v), max(u, v))
                if key not in self.edge_index:
                    raise ValueError("SurfacePeriods: marking step (%d, %d) is no edge of the surface" % (u, v))
                total += (1.0 if u < v else -1.0) * omega[self.edge_index[key]]
            return total
        g = self.genus
        periods_a = np.array([[period(a_cycles[i], self.holomorphic[:, k]) for k in range(g)] for i in range(g)])
        periods_b = np.array([[period(b_cycles[i], self.holomorphic[:, k]) for k in range(g)] for i in range(g)])
        if np.linalg.matrix_rank(periods_a) < g:
            raise ValueError("SurfacePeriods: the A-periods of the holomorphic differentials are singular; "
                             "the A cycles do not span half the homology")
        omega = periods_b @ np.linalg.inv(periods_a)
        self.orientation_flipped = False
        imaginary = np.linalg.eigvalsh(0.5 * (omega.imag + omega.imag.T))
        if imaginary.max() < 0.0:
            # The marking is symplectic for the opposite orientation: the
            # conjugate structure, whose holomorphic differentials are the
            # conjugates and whose period matrix is the conjugate.
            self.orientation_flipped = True
            self.holomorphic = np.conj(self.holomorphic)
            periods_a, periods_b, omega = np.conj(periods_a), np.conj(periods_b), np.conj(omega)
            imaginary = -imaginary[::-1]
        self.periods_a, self.periods_b = periods_a, periods_b
        self.omega = omega
        self.symmetry_residual = float(np.linalg.norm(omega - omega.T) / max(1.0, np.linalg.norm(omega)))
        self.imaginary_eigenvalues = imaginary
        self.positive = bool(imaginary.min() > 0.0)

    # ------------------------------------------------------------ summary
    def report(self):
        """The numbers a record keeps."""
        return {"genus": int(self.genus), "harmonic_rank": int(self.harmonic_rank),
                "omega": self.omega.copy(), "symmetry_residual": self.symmetry_residual,
                "imaginary_eigenvalues": [float(x) for x in self.imaginary_eigenvalues],
                "positive": self.positive, "orientation_flipped": self.orientation_flipped,
                "j_residual": self.j_residual,
                "j_eigenvalues": [complex(x) for x in self.eigenvalues],
                "warnings": list(self.warnings),
                "faces": int(len(self.faces)), "edges": int(len(self.edges)), "vertices": int(len(self.vertices))}


def _steps(face):
    return [(face[a], face[(a + 1) % 3]) for a in range(3)]


def _rotations(face):
    return {tuple(face[(a + s) % 3] for a in range(3)) for s in range(3)}


def _traverses(order, u, v):
    """Whether the cyclic order ``order`` steps from ``u`` to ``v``."""
    return any(order[a] == u and order[(a + 1) % 3] == v for a in range(3))
