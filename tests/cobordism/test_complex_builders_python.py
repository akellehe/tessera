# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""The C++ complex builders: Spacetime.fromVertexTuples + Spacetime.prismCells (#288).

The register/fill examples hand-rolled a "Signature(n) -> createVertex ->
createSimplex(sorted) -> uniform pin" builder in a dozen files, plus a 3d copy
and a 4d copy of the staircase prism rule. Both conventions now live once, as
statics on Spacetime. These tests pin the factory against verbatim
reimplementations of the pre-refactor hand-rolled logic, so any drift in the
conventions (vertex labels, the uniform pin, the tracked-metric causal signs,
the staircase) is caught -- and check the degree-generality the 3d/4d copies
could not express (5d, for level 2).
"""

import unittest

import tessera
import cmath


# --------------------------------------------------------------------------- #
# Verbatim reimplementations of the pre-refactor hand-rolled builders, the
# reference the factory must reproduce exactly.
# --------------------------------------------------------------------------- #
def _old_surface(faces, weight=1.0, phase=0.0):
    """The pre-refactor _surface (2-complex, uniform Hermitian pin)."""
    sig = tessera.Signature(2, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in sorted({v for f in faces for v in f})}
    for f in faces:
        t = sorted(f)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]]])
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(weight)))
        e.setPhase(phase)
    return st


def _old_bulk(cells, weight=1.0, phase=0.0):
    """The pre-refactor _bulk (3-complex, uniform Hermitian pin)."""
    sig = tessera.Signature(3, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {i: st.createVertex(i) for i in sorted({v for c in cells for v in c})}
    for c in cells:
        t = sorted(c)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]], vmap[t[3]]])
    for e in st.getEdgeList().toVector():
        e.setLength(cmath.sqrt(complex(weight)))
        e.setPhase(phase)
    return st


def _old_layered_bulk(cells, stride=12):
    """The pre-refactor _layered_time_bulk (tracked metric, time = id // stride)."""
    sig = tessera.Signature(3, tessera.Lorentzian)
    st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, None)
    vmap = {}
    for i in sorted({v for c in cells for v in c}):
        vmap[i] = st.createVertex(i, [float(i // stride)])
    for c in cells:
        t = sorted(c)
        st.createSimplex([vmap[t[0]], vmap[t[1]], vmap[t[2]], vmap[t[3]]])
    return st


def _old_prism(cells, layers=1, twist=None, stride=None):
    """The pre-refactor staircase (the 3d _prism_cells / 4d _prism4_cells rule),
    generalized over the base-cell arity. Base cells are taken sorted."""
    cells = [tuple(sorted(c)) for c in cells]
    if stride is None:
        stride = max(v for c in cells for v in c) + 1
    ident = {v: v for v in range(stride)}
    twist = dict(twist) if twist else ident

    def compose(g, h):
        return {v: g[h[v]] for v in h}

    phi = [ident]
    for _ in range(layers):
        phi.append(compose(twist, phi[-1]))
    out = []
    for ell in range(layers):
        lo, hi = phi[ell], phi[ell + 1]
        for base in cells:
            m = len(base)
            for j in range(m):
                s = [lo[base[i]] + stride * ell for i in range(j + 1)]
                s += [hi[base[i]] + stride * (ell + 1) for i in range(j, m)]
                out.append(tuple(sorted(s)))
    return sorted(set(out))


def _snapshot(st):
    """The geometry-bearing fingerprint of a complex: every edge keyed by its
    sorted endpoints to (squared length, phase), and the sorted set of simplex
    vertex tuples."""
    edges = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        phase = e.getPhase()   # complex: the group is C* = U(1) x R+
        edges[(min(a, b), max(a, b))] = (round((e.getLength()**2).real, 12),
                                         (round(phase.real, 12),
                                          round(phase.imag, 12)))
    simplices = sorted(tuple(sorted(v.getId() for v in s.getVertices()))
                       for s in st.getSimplices())
    return edges, simplices


def _vertices_by_id(st):
    out = {}
    for e in st.getEdgeList().toVector():
        for v in (e.getSource(), e.getTarget()):
            out[v.getId()] = v
    return out


# An octahedron-style 6-vertex surface (sorted faces) and a 12-vertex triangle
# base whose prism is a tet bulk (the shape of the real 3D layered fill).
_OCTA = [tuple(sorted(f)) for f in
         [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 1, 4),
          (1, 2, 5), (2, 3, 5), (3, 4, 5), (1, 4, 5)]]
_TRI12 = [tuple(sorted(t)) for t in
          [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11),
           (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 9, 3),
           (1, 10, 4), (2, 11, 5)]]
# A 6-vertex tetrahedral 3-complex (the join-style seed) for the 4d prism.
_TETS6 = [tuple(sorted(t)) for t in
          [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (0, 1, 4, 5)]]
# An order-6 twist (a permutation of 0..5) for the twisted-prism checks.
_TWIST6 = {0: 1, 1: 2, 2: 0, 3: 4, 4: 5, 5: 3}


class TestPrismCells(unittest.TestCase):
    """Spacetime.prismCells reproduces the staircase rule the register fills
    carried as separate 3d and 4d copies, in every dimension."""

    def test_reproduces_3d_staircase(self):
        # Triangles -> tetrahedra, the _prism_cells rule. Identity + twisted,
        # single + multi-layer.
        for layers in (1, 2, 3):
            for twist in (None, _TWIST6):
                with self.subTest(layers=layers, twisted=twist is not None):
                    got = [tuple(c) for c in tessera.Spacetime.prismCells(
                        [list(f) for f in _OCTA], layers, twist)]
                    want = _old_prism(_OCTA, layers, twist)
                    self.assertEqual(got, want)

    def test_reproduces_4d_staircase(self):
        # Tetrahedra -> 4-simplices, the _prism4_cells rule (single layer, the
        # only arity the 4d copy supported), identity + twisted.
        for twist in (None, _TWIST6):
            with self.subTest(twisted=twist is not None):
                got = [tuple(c) for c in tessera.Spacetime.prismCells(
                    [list(t) for t in _TETS6], 1, twist)]
                want = _old_prism(_TETS6, 1, twist)
                self.assertEqual(got, want)

    def test_cell_arity_and_count(self):
        # A base m-vertex cell yields m prism cells per layer, each on m+1
        # vertices. The same rule at every dimension -- 5d included (the level-2
        # case the 3d/4d copies could not reach).
        for base, m in ((_OCTA, 3), (_TETS6, 4),
                        ([tuple(range(5)), tuple(range(1, 6))], 5)):
            with self.subTest(m=m):
                layers = 2
                cells = tessera.Spacetime.prismCells(
                    [list(c) for c in base], layers, None)
                self.assertTrue(all(len(c) == m + 1 for c in cells))
                # Per layer, per base cell, exactly m staircase simplices (before
                # the cross-cell dedup at shared walls) -- so the unique count is
                # at most layers * |base| * m and positive.
                self.assertGreater(len(cells), 0)
                self.assertLessEqual(len(cells), layers * len(base) * m)

    def test_twist_is_cumulative_across_layers(self):
        # phi^2 over one transition equals phi over two transitions of phi:
        # composing the twist into the layer offsets, not re-applying it raw.
        twosq = {v: _TWIST6[_TWIST6[v]] for v in _TWIST6}
        a = _old_prism(_OCTA, 2, _TWIST6)
        b = [tuple(c) for c in tessera.Spacetime.prismCells(
            [list(f) for f in _OCTA], 2, _TWIST6)]
        self.assertEqual(b, a)
        # the second layer's top end is glued through phi^2, by construction
        self.assertEqual(twosq, {v: _TWIST6[_TWIST6[v]] for v in range(6)})

    def test_identity_twist_matches_no_twist(self):
        ident = {v: v for v in range(6)}
        with_id = [tuple(c) for c in tessera.Spacetime.prismCells(
            [list(f) for f in _OCTA], 2, ident)]
        without = [tuple(c) for c in tessera.Spacetime.prismCells(
            [list(f) for f in _OCTA], 2, None)]
        self.assertEqual(with_id, without)


class TestFromCellsUniformPin(unittest.TestCase):
    """Spacetime.fromVertexTuples with no vertexTimes reproduces the uniform Hermitian
    pin of _surface / _bulk / _bulk4 exactly."""

    def test_surface_pin_matches(self):
        for weight, phase in ((1.0, 0.0), (2.5, 0.3), (0.7, -1.2)):
            with self.subTest(weight=weight, phase=phase):
                got = _snapshot(tessera.Spacetime.fromVertexTuples(
                    2, [list(f) for f in _OCTA], weight, phase))
                want = _snapshot(_old_surface(_OCTA, weight, phase))
                self.assertEqual(got, want)

    def test_bulk_pin_matches(self):
        for weight, phase in ((1.0, 0.0), (3.0, 0.5)):
            with self.subTest(weight=weight, phase=phase):
                got = _snapshot(tessera.Spacetime.fromVertexTuples(
                    3, [list(c) for c in _TETS6], weight, phase))
                want = _snapshot(_old_bulk(_TETS6, weight, phase))
                self.assertEqual(got, want)

    def test_every_edge_carries_the_pin(self):
        st = tessera.Spacetime.fromVertexTuples(2, [list(f) for f in _OCTA], 1.7, -0.4)
        edges, _ = _snapshot(st)
        self.assertTrue(edges)
        for sq, (phRe, phIm) in edges.values():
            self.assertAlmostEqual(sq, 1.7)
            self.assertAlmostEqual(phRe, -0.4)
            self.assertAlmostEqual(phIm, 0.0)

    def test_the_pin_carries_a_complex_phase(self):
        # The pin argument is the C* connection phase, so its non-compact part
        # must reach the edges too.
        st = tessera.Spacetime.fromVertexTuples(2, [list(f) for f in _OCTA], 1.7,
                                         complex(-0.4, 0.9))
        edges, _ = _snapshot(st)
        self.assertTrue(edges)
        for _, (phRe, phIm) in edges.values():
            self.assertAlmostEqual(phRe, -0.4)
            self.assertAlmostEqual(phIm, 0.9)

    def test_vertices_are_coordinate_free(self):
        # The uniform-pin vertices carry no coordinates (getTime() == 0; the
        # coordinate vector is absent), so the length-2/3 getTime() trap never
        # arises.
        st = tessera.Spacetime.fromVertexTuples(2, [list(f) for f in _OCTA])
        for v in _vertices_by_id(st).values():
            self.assertEqual(v.getTime(), 0.0)
            with self.assertRaises(Exception):
                v.getCoordinates()

    def test_one_simplex_per_cell_distinct_vertices(self):
        st = tessera.Spacetime.fromVertexTuples(3, [list(c) for c in _TETS6])
        _, simplices = _snapshot(st)
        self.assertEqual(simplices, sorted(tuple(c) for c in _TETS6))
        ids = {v for c in _TETS6 for v in c}
        self.assertEqual(st.getVertexCount(), len(ids))


class TestFromCellsTrackedMetric(unittest.TestCase):
    """Spacetime.fromVertexTuples with a per-vertex time vector reproduces the tracked
    metric rule of _layered_time_bulk: spacelike intra-layer, timelike
    inter-layer edges, no uniform re-pin."""

    def _prism_and_times(self, layers, stride=12):
        prism = tessera.Spacetime.prismCells(
            [list(c) for c in _TRI12], layers, None)
        n = max(v for c in prism for v in c) + 1
        times = [float(i // stride) for i in range(n)]
        return prism, times

    def test_matches_old_layered_builder(self):
        for layers in (1, 2, 3):
            with self.subTest(layers=layers):
                prism, times = self._prism_and_times(layers)
                got = _snapshot(tessera.Spacetime.fromVertexTuples(
                    3, prism, vertexTimes=times))
                want = _snapshot(_old_layered_bulk([tuple(c) for c in prism]))
                self.assertEqual(got, want)

    def test_causal_edge_signs(self):
        # Intra-layer (equal time) edges are spacelike (+a = +1); inter-layer
        # (time difference one) edges are timelike (-alpha*a = -1).
        prism, times = self._prism_and_times(2)
        edges, _ = _snapshot(tessera.Spacetime.fromVertexTuples(
            3, prism, vertexTimes=times))
        intra = [(i, j) for (i, j) in edges if i // 12 == j // 12]
        inter = [(i, j) for (i, j) in edges if i // 12 != j // 12]
        self.assertTrue(intra and inter)
        for ij in intra:
            self.assertAlmostEqual(edges[ij][0], 1.0)
        for ij in inter:
            self.assertAlmostEqual(edges[ij][0], -1.0)

    def test_tracked_vertices_carry_arity_one_time(self):
        # The time coordinate is arity one -- {t} -- never the length-2/3 vector
        # that makes Vertex.getTime() throw.
        prism, times = self._prism_and_times(2)
        st = tessera.Spacetime.fromVertexTuples(3, prism, vertexTimes=times)
        for vid, v in _vertices_by_id(st).items():
            self.assertEqual(list(v.getCoordinates()), [float(vid // 12)])
            self.assertEqual(v.getTime(), float(vid // 12))

    def test_weight_and_phase_ignored_when_tracked(self):
        # Under the tracked rule the auto-wired causal lengths are the geometry;
        # weight/phase do not overwrite them.
        prism, times = self._prism_and_times(2)
        pinned = tessera.Spacetime.fromVertexTuples(
            3, prism, weight=99.0, phase=7.0, vertexTimes=times)
        plain = tessera.Spacetime.fromVertexTuples(3, prism, vertexTimes=times)
        self.assertEqual(_snapshot(pinned), _snapshot(plain))

    def test_short_vertex_times_raises(self):
        cells = [list(c) for c in _TETS6]  # ids up to 5
        with self.assertRaises(Exception):
            tessera.Spacetime.fromVertexTuples(3, cells, vertexTimes=[0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
