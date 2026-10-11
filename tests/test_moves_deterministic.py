# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Deterministic tests for each Pachner move's topology change, decoupled
from random selection and Metropolis acceptance.

Each test builds a known lattice using the low-level spacetime API,
applies the move's topology change directly, asserts the exact result,
then applies the inverse and asserts we return.

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
"""

import unittest
import tessera


def _make_spacetime():
    """Create a properly configured 4D Lorentzian spacetime."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    return st


def _top_simplices(st):
    """Return list of all top-dimensional (5-vertex) simplices."""
    return [s for s in st.get_simplices() if len(s.get_vertices()) == 5]


def _top_fps(st):
    """Set of fingerprints of all top simplices."""
    return {hash(s) for s in _top_simplices(st)}


def _assert_all_causal(test, st):
    """Every top simplex spans exactly 2 time slices."""
    for s in _top_simplices(st):
        times = {v.get_time() for v in s.get_vertices()}
        test.assertEqual(len(times), 2,
                         f"Non-causal: orientation={s.get_orientation().numeric()}")


def _is_valid_cdt_simplex(verts):
    """Check if vertex set spans exactly 2 times with valid CDT orientation."""
    times = {v.get_time() for v in verts}
    if len(times) != 2:
        return False
    o = tessera.TemporalOrientation(0, 0)  # dummy
    # Count vertices at each time
    time_list = sorted(times)
    ti_count = sum(1 for v in verts if v.get_time() == time_list[0])
    tf_count = len(verts) - ti_count
    d = len(verts) - 1
    if (ti_count, tf_count) in ((d, 1), (1, d), (d-1, 2), (2, d-1)):
        return True
    return False


def _assert_counts(test, st):
    """N4 = N41 + N32 and matches manual count."""
    n41, n32 = 0, 0
    for s in _top_simplices(st):
        o = s.get_orientation().numeric()
        if o in ((4, 1), (1, 4)):
            n41 += 1
        elif o in ((3, 2), (2, 3)):
            n32 += 1
        else:
            test.fail(f"Invalid orientation {o}")
    test.assertEqual(st.get_n41(), n41)
    test.assertEqual(st.get_n32(), n32)
    test.assertEqual(st.get_top_simplex_count(), n41 + n32)


# =====================================================================
# Cone — direct topology change via createSimplex
# =====================================================================

class TestConeForward(unittest.TestCase):
    """[BGL] Sec. 2.3.1: Cone a facet to a new vertex using createSimplex.

    This tests the raw topology primitive that underlies vertex insertion,
    separated from CDT move selection and acceptance.
    """

    def _cone_facet(self, st, top_simplex):
        """Find a non-timelike facet and cone it manually via createSimplex.

        Replicates the time-assignment logic from Simplex::cone(): the new
        vertex is placed at the facet's ti or tf time depending on the
        coface orientation, ensuring the new simplex spans exactly 2 times.

        Returns (new_simplex, new_vertex) or (None, None) if no facet found.
        """
        facet = None
        for f in top_simplex.get_facets():
            if not f.is_spatial():
                facet = f
                break
        if facet is None:
            return None, None

        # Determine correct time for new vertex (mirrors Simplex::cone logic)
        fti, ftf = facet.get_orientation().numeric()
        cti, ctf = top_simplex.get_orientation().numeric()
        facet_times = sorted({v.get_time() for v in facet.get_vertices()})
        if ctf > ftf:
            cone_time = facet_times[0]   # ti
        else:
            cone_time = facet_times[-1]  # tf

        max_id = max(v.get_id() for v in st.get_vertex_list().to_vector())
        new_vertex = st.create_vertex(max_id + 1, [cone_time])

        new_verts = list(facet.get_vertices()) + [new_vertex]
        new_simplex, created = st.create_simplex(new_verts)
        if not created:
            return None, None
        return new_simplex, new_vertex

    def test_cone_creates_one_new_top_simplex(self):
        """Coning a non-timelike facet creates exactly 1 new top simplex."""
        st = _make_spacetime()
        st.build(5)
        before_n4 = st.get_top_simplex_count()
        before_n0 = st.get_vertex_count()
        before_fps = _top_fps(st)

        top = _top_simplices(st)[0]
        new_simplex, new_vertex = self._cone_facet(st, top)
        self.assertIsNotNone(new_simplex, "Need a non-timelike facet")

        after_fps = _top_fps(st)
        gained = after_fps - before_fps
        self.assertEqual(len(gained), 1)
        self.assertEqual(st.get_top_simplex_count(), before_n4 + 1)
        self.assertEqual(st.get_vertex_count(), before_n0 + 1)
        self.assertEqual(len(new_simplex.get_vertices()), 5)
        self.assertTrue(new_simplex.has_vertex(new_vertex))

        o = new_simplex.get_orientation().numeric()
        self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)))
        _assert_counts(self, st)
        _assert_all_causal(self, st)

    def test_cone_then_remove_round_trip(self):
        """Cone a facet, then removeSimplex the result: N4 returns."""
        st = _make_spacetime()
        st.build(5)
        before_n4 = st.get_top_simplex_count()
        before_fps = _top_fps(st)

        top = _top_simplices(st)[0]
        new_simplex, _ = self._cone_facet(st, top)
        self.assertIsNotNone(new_simplex)
        self.assertEqual(st.get_top_simplex_count(), before_n4 + 1)

        st.remove_simplex(new_simplex)
        self.assertEqual(st.get_top_simplex_count(), before_n4)
        self.assertTrue(before_fps.issubset(_top_fps(st)))
        _assert_counts(self, st)

    def test_cone_five_times_then_remove_five(self):
        """5 cones then 5 removes: N4 returns to start each time."""
        st = _make_spacetime()
        st.build(10)
        start_n4 = st.get_top_simplex_count()

        added_simplices = []
        for i in range(5):
            top = _top_simplices(st)[0]
            new_simplex, _ = self._cone_facet(st, top)
            self.assertIsNotNone(new_simplex, f"Iteration {i}: need facet")
            added_simplices.append(new_simplex)
            self.assertEqual(st.get_top_simplex_count(), start_n4 + i + 1)
            _assert_counts(self, st)
            _assert_all_causal(self, st)

        for i, s in enumerate(reversed(added_simplices)):
            st.remove_simplex(s)
            expected = start_n4 + (4 - i)
            self.assertEqual(st.get_top_simplex_count(), expected,
                             f"Remove {i}: expected N4={expected}")
            _assert_counts(self, st)

        self.assertEqual(st.get_top_simplex_count(), start_n4)


# =====================================================================
# Flip (2 -> d) — direct topology change
# =====================================================================

class TestFlipForward(unittest.TestCase):
    """[BGL] Sec. 2.3.2: The (2,d) flip replaces 2 simplices sharing a (d-1)-face with d
    new simplices.  This is the topology change, decoupled from CDT::flip().
    """

    def _find_flippable(self, st):
        """Find a (d-1)-face with exactly 2 top cofaces where the flip
        produces only valid CDT simplices (spans exactly 2 times).
        Returns (facet, s1, s2, shared_verts, unique_verts) or None.
        """
        for top in _top_simplices(st):
            for facet in top.get_facets():
                cofaces = [c for c in facet.get_cofaces()
                           if len(c.get_vertices()) == 5]
                if len(cofaces) != 2:
                    continue
                s1, s2 = cofaces
                s1_vids = {v.get_id() for v in s1.get_vertices()}
                s2_vids = {v.get_id() for v in s2.get_vertices()}
                shared_vids = s1_vids & s2_vids
                unique_vids = s1_vids ^ s2_vids
                if len(shared_vids) != 4 or len(unique_vids) != 2:
                    continue
                all_verts = {v.get_id(): v for v in
                             list(s1.get_vertices()) + list(s2.get_vertices())}
                shared = [all_verts[vid] for vid in shared_vids]
                unique = [all_verts[vid] for vid in unique_vids]
                # Check that all d new simplices would be valid CDT
                valid = True
                for skip in range(4):
                    nv = [shared[i] for i in range(4) if i != skip] + unique
                    if not _is_valid_cdt_simplex(nv):
                        valid = False
                        break
                if not valid:
                    continue
                return facet, s1, s2, shared, unique
        return None

    def test_flip_2_to_d(self):
        """Remove 2, create d=4 new simplices with correct vertex sets."""
        st = _make_spacetime()
        st.build(20)
        before_n4 = st.get_top_simplex_count()
        before_n0 = st.get_vertex_count()

        result = self._find_flippable(st)
        if result is None:
            self.skipTest("No flippable configuration found")
        facet, s1, s2, shared, unique = result

        st.remove_simplex(s1)
        st.remove_simplex(s2)
        self.assertEqual(st.get_top_simplex_count(), before_n4 - 2)

        new_simplices = []
        for skip in range(4):
            verts = [shared[i] for i in range(4) if i != skip]
            verts.extend(unique)
            self.assertEqual(len(verts), 5)
            new_s, created = st.create_simplex(verts)
            if created:
                new_simplices.append(new_s)

        self.assertGreaterEqual(len(new_simplices), 1)
        self.assertGreater(st.get_top_simplex_count(), before_n4 - 2)
        self.assertEqual(st.get_vertex_count(), before_n0)

        for s in new_simplices:
            o = s.get_orientation().numeric()
            self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)),
                          f"New simplex has invalid orientation {o}")

        _assert_counts(self, st)
        _assert_all_causal(self, st)

    def test_flip_then_reverse_flip(self):
        """Flip forward, then flip back: N4 returns."""
        st = _make_spacetime()
        st.build(20)
        before_n4 = st.get_top_simplex_count()

        result = self._find_flippable(st)
        if result is None:
            self.skipTest("No flippable configuration found")
        _, s1, s2, shared, unique = result

        s1_verts = list(s1.get_vertices())
        s2_verts = list(s2.get_vertices())

        st.remove_simplex(s1)
        st.remove_simplex(s2)
        new_simplices = []
        for skip in range(4):
            verts = [shared[i] for i in range(4) if i != skip]
            verts.extend(unique)
            new_s, created = st.create_simplex(verts)
            if created:
                new_simplices.append(new_s)
        _assert_counts(self, st)

        for s in new_simplices:
            st.remove_simplex(s)
        st.create_simplex(s1_verts)
        st.create_simplex(s2_verts)

        self.assertEqual(st.get_top_simplex_count(), before_n4)
        _assert_counts(self, st)
        _assert_all_causal(self, st)


# =====================================================================
# Shift (3 -> 3) — direct topology change
# =====================================================================

class TestShiftForward(unittest.TestCase):
    """[BGL] Sec. 2.3.3: The (3,3) shift replaces 3 simplices sharing a (d-2)-face with
    3 new simplices.  Decoupled from CDT::shift().
    """

    def _find_shiftable(self, st):
        """Find 3 top simplices sharing 3 vertices (a triangle).
        Returns (sharing, shared_verts, unique_verts) or None.
        """
        tops = _top_simplices(st)
        for top in tops:
            verts = list(top.get_vertices())
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    for k in range(j + 1, len(verts)):
                        tri = [verts[i], verts[j], verts[k]]
                        sharing = []
                        for s in tri[0].get_simplices():
                            if len(s.get_vertices()) != 5:
                                continue
                            if s.has_vertex(tri[1]) and s.has_vertex(tri[2]):
                                sharing.append(s)
                        if len(sharing) != 3:
                            continue
                        all_vids = set()
                        for s in sharing:
                            for v in s.get_vertices():
                                all_vids.add(v.get_id())
                        if len(all_vids) != 6:
                            continue
                        all_verts = {}
                        for s in sharing:
                            for v in s.get_vertices():
                                all_verts[v.get_id()] = v
                        shared_v = []
                        unique_v = []
                        for vid, v in all_verts.items():
                            in_all = all(s.has_vertex(v) for s in sharing)
                            if in_all:
                                shared_v.append(v)
                            else:
                                unique_v.append(v)
                        if len(shared_v) == 3 and len(unique_v) == 3:
                            # Check new simplices would be valid CDT
                            valid = True
                            for skip in range(3):
                                nv = [shared_v[ii] for ii in range(3) if ii != skip] + unique_v
                                if not _is_valid_cdt_simplex(nv):
                                    valid = False
                                    break
                            if valid:
                                return sharing, shared_v, unique_v
        return None

    def test_shift_3_to_3(self):
        """Remove 3, create 3 new simplices with correct vertex sets."""
        st = _make_spacetime()
        st.build(200)
        target = st.get_n41()
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / max(target, 1), target)
        cdt.sweep(200)

        before_n4 = st.get_top_simplex_count()
        before_n0 = st.get_vertex_count()

        result = self._find_shiftable(st)
        if result is None:
            self.skipTest("No shiftable configuration found")
        sharing, shared, unique = result

        for s in sharing:
            st.remove_simplex(s)
        self.assertEqual(st.get_top_simplex_count(), before_n4 - 3)

        new_simplices = []
        for skip in range(3):
            verts = [shared[i] for i in range(3) if i != skip]
            verts.extend(unique)
            self.assertEqual(len(verts), 5)
            new_s, created = st.create_simplex(verts)
            if created:
                new_simplices.append(new_s)

        self.assertEqual(st.get_vertex_count(), before_n0)

        for s in new_simplices:
            o = s.get_orientation().numeric()
            self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)))

        _assert_counts(self, st)
        _assert_all_causal(self, st)

    def test_shift_then_reverse(self):
        """Shift forward, then shift back: N4 returns."""
        st = _make_spacetime()
        st.build(200)
        target = st.get_n41()
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / max(target, 1), target)
        cdt.sweep(200)

        before_n4 = st.get_top_simplex_count()

        result = self._find_shiftable(st)
        if result is None:
            self.skipTest("No shiftable configuration found")
        sharing, shared, unique = result

        old_vert_sets = [list(s.get_vertices()) for s in sharing]

        for s in sharing:
            st.remove_simplex(s)
        new_simplices = []
        for skip in range(3):
            verts = [shared[i] for i in range(3) if i != skip]
            verts.extend(unique)
            new_s, created = st.create_simplex(verts)
            if created:
                new_simplices.append(new_s)
        _assert_counts(self, st)

        for s in new_simplices:
            st.remove_simplex(s)
        for old_verts in old_vert_sets:
            st.create_simplex(old_verts)

        self.assertEqual(st.get_top_simplex_count(), before_n4)
        _assert_counts(self, st)
        _assert_all_causal(self, st)


# =====================================================================
# Iterated forward-backward
# =====================================================================

class TestIteratedConeRemove(unittest.TestCase):
    """[BGL] Sec. 2.3.1: Cone then removeSimplex, iterated many times."""

    def test_ten_iterations(self):
        st = _make_spacetime()
        st.build(10)
        start_n4 = st.get_top_simplex_count()

        for iteration in range(10):
            top = _top_simplices(st)[0]
            facet = None
            for f in top.get_facets():
                if not f.is_spatial():
                    facet = f
                    break
            self.assertIsNotNone(facet, f"Iter {iteration}: need facet")

            # Cone manually via create_simplex (replicate cone() time logic)
            fti, ftf = facet.get_orientation().numeric()
            cti, ctf = top.get_orientation().numeric()
            facet_times = sorted({v.get_time() for v in facet.get_vertices()})
            cone_time = facet_times[0] if ctf > ftf else facet_times[-1]

            max_id = max(v.get_id() for v in st.get_vertex_list().to_vector())
            vertex = st.create_vertex(max_id + 1, [cone_time])
            new_verts = list(facet.get_vertices()) + [vertex]
            new_s, created = st.create_simplex(new_verts)
            self.assertTrue(created, f"Iter {iteration}: simplex already existed")
            self.assertEqual(st.get_top_simplex_count(), start_n4 + 1,
                             f"Iter {iteration}: after cone")
            _assert_counts(self, st)
            _assert_all_causal(self, st)

            # Backward: remove
            st.remove_simplex(new_s)
            self.assertEqual(st.get_top_simplex_count(), start_n4,
                             f"Iter {iteration}: after remove")
            _assert_counts(self, st)


class TestIteratedFlip(unittest.TestCase):
    """[BGL] Sec. 2.3.2: Flip forward then backward, iterated."""

    def _find_flippable(self, st):
        for top in _top_simplices(st):
            for facet in top.get_facets():
                cofaces = [c for c in facet.get_cofaces()
                           if len(c.get_vertices()) == 5]
                if len(cofaces) != 2:
                    continue
                s1, s2 = cofaces
                s1_vids = {v.get_id() for v in s1.get_vertices()}
                s2_vids = {v.get_id() for v in s2.get_vertices()}
                shared_vids = s1_vids & s2_vids
                unique_vids = s1_vids ^ s2_vids
                if len(shared_vids) != 4 or len(unique_vids) != 2:
                    continue
                all_verts = {v.get_id(): v for v in
                             list(s1.get_vertices()) + list(s2.get_vertices())}
                shared = [all_verts[vid] for vid in shared_vids]
                unique = [all_verts[vid] for vid in unique_vids]
                # Check new simplices would be valid CDT
                valid = True
                for skip in range(4):
                    nv = [shared[i] for i in range(4) if i != skip] + unique
                    if not _is_valid_cdt_simplex(nv):
                        valid = False
                        break
                if not valid:
                    continue
                return s1, s2, shared, unique
        return None

    def test_five_iterations(self):
        st = _make_spacetime()
        st.build(20)
        start_n4 = st.get_top_simplex_count()

        for iteration in range(5):
            result = self._find_flippable(st)
            if result is None:
                return
            s1, s2, shared, unique = result
            s1_verts = list(s1.get_vertices())
            s2_verts = list(s2.get_vertices())

            st.remove_simplex(s1)
            st.remove_simplex(s2)
            new_simplices = []
            for skip in range(4):
                verts = [shared[i] for i in range(4) if i != skip]
                verts.extend(unique)
                new_s, created = st.create_simplex(verts)
                if created:
                    new_simplices.append(new_s)
            _assert_counts(self, st)
            _assert_all_causal(self, st)

            for s in new_simplices:
                st.remove_simplex(s)
            st.create_simplex(s1_verts)
            st.create_simplex(s2_verts)

            self.assertEqual(st.get_top_simplex_count(), start_n4,
                             f"Iter {iteration}: N4 should return")
            _assert_counts(self, st)
            _assert_all_causal(self, st)


if __name__ == "__main__":
    unittest.main()
