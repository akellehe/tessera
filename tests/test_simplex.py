# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import unittest

from tessera import Vertex, Simplex, Metric, Spacetime, Signature, SignatureType


class TestSimplex(unittest.TestCase):
    def setUp(self):
        self.spacetime = Spacetime()

    def test_get_faces(self):
        s1, _ = self.spacetime.create_simplex((4, 1))
        facets = s1.get_facets()
        self.assertEqual(len(facets), 5)
        tio, tfo = (0, 0)
        for vertex in s1.get_vertices():
            if vertex.get_time() == 0:
                tio += 1
            elif vertex.get_time() > 0:
                tfo += 1

        self.assertEqual(tio, 4)
        self.assertEqual(tfo, 1)

        nTimelike = 0
        for face in s1.get_facets():
            face.validate()
            self.assertEqual(len(face.get_vertices()), 4)
            self.assertEqual(len(face.get_edges()), 6)
            self.assertEqual(len(set([(e.get_source().get_id(), e.get_target().get_id()) for e in face.get_edges()])), 6)
            self.assertEqual(len(face.get_cofaces()), 1)
            if face.is_spatial():
                nTimelike += 1
                for timelikeFace in face.get_facets():
                    timelikeFace.validate()
                    self.assertTrue(timelikeFace.is_spatial())
                    self.assertEqual(len(timelikeFace.get_vertices()), 3)
                    self.assertEqual(len(timelikeFace.get_edges()), 3)
                    self.assertEqual(len(set([(e.get_source().get_id(), e.get_target().get_id()) for e in timelikeFace.get_edges()])), 3)
                    self.assertEqual(len(timelikeFace.get_cofaces()), 1)

        self.assertEqual(nTimelike, 1)

    def test_creating_oriented_simplices(self):
        ti, tf = (4, 1)
        s1, _ = self.spacetime.create_simplex((ti, tf))
        oti, otf = (0, 0)
        initialTime = 0
        finalTime = 0
        for vertex in s1.get_vertices():
            initialTime = min(initialTime, vertex.get_time())
            finalTime = max(finalTime, vertex.get_time())

        for vertex in s1.get_vertices():
            if vertex.get_time() == initialTime:
                oti += 1
            elif vertex.get_time() == finalTime:
                otf += 1

        self.assertEqual(oti, ti)
        self.assertEqual(otf, tf)

        ti, tf = (3, 2)
        s2, _ = self.spacetime.create_simplex((ti, tf))
        oti, otf = (0, 0)
        initialTime = 0
        finalTime = 0
        for vertex in s2.get_vertices():
            initialTime = min(initialTime, vertex.get_time())
            finalTime = max(finalTime, vertex.get_time())

        for vertex in s2.get_vertices():
            if vertex.get_time() == initialTime:
                oti += 1
            elif vertex.get_time() == finalTime:
                otf += 1

        for f in s1.get_facets():
            f.validate()
        for f in s2.get_facets():
            f.validate()
        self.assertEqual(oti, ti)
        self.assertEqual(otf, tf)

    @unittest.skip("Parity check not yet implemented")
    def test_parity(self):
        simplex41, _ = self.spacetime.create_simplex((4, 1))
        f1, f2, f3, f4, f5 = simplex41.get_facets()

        self.assertEqual(len(f1.get_vertices()), 4)

        # Disjoint faces have pairty flag=0
        self.assertEqual(f1.checkPairty(f2), 0)

        v1, v2, v3, v4 = f1.get_vertices()

        #The same face has pairty flag=1
        clone = Simplex([v1, v2, v3, v4])
        self.assertEqual(f1.checkPairty(clone), 1)

        # A single vertex swap has pairty flag=-1
        oneSwap = Simplex([v2, v1, v3, v4])
        self.assertEqual(f1.checkPairty(oneSwap), -1)

        # Two swaps has pairty flag=1
        twoSwaps = Simplex([v2, v1, v4, v3])
        self.assertEqual(f1.checkPairty(twoSwaps), 1)

        for f in simplex41.get_facets():
            f.validate()

    def test_get_edges(self):
        simplex41, _ = self.spacetime.create_simplex((4, 1))

        f1, f2, f3, f4, f5 = simplex41.get_facets()
        v1, v2, v3, v4 = sorted(v.get_id() for v in f1.get_vertices())
        e1, e2, e3, e4, e5, e6 = sorted([(e.get_source().get_id(), e.get_target().get_id()) for e in f1.get_edges()])

        """
(Pdb) e1
(1, 2)
(Pdb) e2
(1, 3)
(Pdb) e3
(1, 4)
(Pdb) e4
(2, 3)
(Pdb) e5
(2, 4)
(Pdb) e6
(3, 4)
        """

        # 1>2
        self.assertEqual(e1[0], 1)
        self.assertEqual(e1[1], 2)

        # 2>3
        self.assertEqual(e2[0], 1)
        self.assertEqual(e2[1], 3)

        # 1>3
        self.assertEqual(e3[0], 1)
        self.assertEqual(e3[1], 4)

        # 3>4
        self.assertEqual(e4[0], 2)
        self.assertEqual(e4[1], 3)

        # 2>4
        self.assertEqual(e5[0], 2)
        self.assertEqual(e5[1], 4)

        # 1>4
        self.assertEqual(e6[0], 3)
        self.assertEqual(e6[1], 4)


    def test_is_timelike14(self):
        st = Spacetime()
        simplex14, _ = st.create_simplex((1, 4))
        ntime = 0
        nspace = 0
        for facet in simplex14.get_facets():
            if facet.is_spatial():
                ntime += 1
            else:
                nspace += 1

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)

    def test_is_timelike41(self):
        st = Spacetime()
        simplex41, _ = st.create_simplex((4, 1))
        ntime = 0
        nspace = 0
        for facet in simplex41.get_facets():
            if facet.is_spatial():
                ntime += 1
            else:
                nspace += 1

        self.assertEqual(ntime, 1)
        self.assertEqual(nspace, 4)

    def test_is_timelike23(self):
        st = Spacetime()
        simplex23, _ = st.create_simplex((2, 3))
        ntime = 0
        nspace = 0
        for facet in simplex23.get_facets():
            if facet.is_spatial():
                ntime += 1
            else:
                nspace += 1

        self.assertEqual(ntime, 0)
        self.assertEqual(nspace, 5)

    def test_replace_vertex(self):
        st = Spacetime()
        simplex, _ = st.create_simplex((1, 2))
        facets1 = simplex.get_facets()
        self.assertEqual(len(simplex.get_vertices()), 3)
        self.assertEqual(len(simplex.get_edges()), 3)

        v0 = simplex.get_vertices()[0]
        v4 = st.create_vertex(4, [0])
        v0id = v0.get_id()

        simplex.replace_vertex(v0, v4)

        self.assertEqual(len(simplex.get_vertices()), 3)
        self.assertEqual(len(simplex.get_edges()), 3)

        self.assertNotIn(v0, simplex.get_vertices())
        self.assertNotIn(v0id, simplex.get_vertex_id_lookup())

        self.assertIn(v4, simplex.get_vertices())
        self.assertIn(4, simplex.get_vertex_id_lookup())

        facets2 = simplex.get_facets()

        for f in facets1:
            f.validate()
        for f in facets2:
            f.validate()

    def test_facets_are_registered_to_vertices(self):
        st = Spacetime()
        s1, created = st.create_simplex((1, 4))
        for v in s1.get_vertices():
            self.assertEqual(len(v.get_simplices()), 1)
            self.assertEqual(list(v.get_simplices())[0], s1)

        for v in s1.get_vertices():
            self.assertEqual(len(v.get_simplices()), 1)

        facets = s1.get_facets()
        for v in s1.get_vertices():
            # A 4-simplex has 4 facets + the 4-simplex itself = 5 simplices
            self.assertEqual(len(v.get_simplices()), 5)
            for facet in facets:
                if v in facet.get_vertices():
                    self.assertIn(facet, v.get_simplices())

        for facet in facets:
            cofaces = facet.get_cofaces()
            for coface in cofaces:
                self.assertTrue(coface.is_coface_to(facet))


if __name__ == '__main__':
    unittest.main()
