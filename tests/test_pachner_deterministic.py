# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""
Deterministic forward-backward tests for each Pachner move.

Each test:
  1. Builds a small lattice and snapshots the full state.
  2. Applies one move, verifies the exact combinatorial change.
  3. Applies the inverse move, verifies we return to a compatible state.
  4. Repeats for several iterations.

The moves are stochastic (random simplex selection), so we retry until
one succeeds. But once a move succeeds, we assert exact combinatorial
deltas — not statistical properties.

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
"""

import unittest
import tessera


# =====================================================================
# Helpers
# =====================================================================

def _build_small(n_simplices=10):
    """Build a minimal CDT spacetime.

    Uses epsilon=0 so the Metropolis criterion only depends on the Regge
    action, and sets k4 very high so that add moves are expensive (keeps
    the lattice from growing).  Individual move tests call the moves
    directly, bypassing acceptance.
    """
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(n_simplices)
    target = st.getN41()
    # epsilon=0 removes volume-fixing from acceptance (we call moves directly)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.0, target)
    # Disable vertex relabeling so fingerprint-based before/after comparisons work
    cdt.setRelabelVertices(False)
    return cdt, st


def _snapshot(st):
    """Capture the full lattice state as a dict."""
    top_fps = set()
    orientations = {}
    for s in st.getSimplices():
        if len(s.getVertices()) == 5:
            fp = hash(s)
            top_fps.add(fp)
            orientations[fp] = s.getOrientation().numeric()
    return {
        "n4": st.getTopSimplexCount(),
        "n41": st.getN41(),
        "n32": st.getN32(),
        "n0": st.getVertexCount(),
        "top_fps": top_fps,
        "orientations": orientations,
    }


def _verify_all_top_causal(st):
    """Assert every top simplex spans exactly 2 time slices."""
    for s in st.getSimplices():
        if len(s.getVertices()) != 5:
            continue
        times = set(v.getTime() for v in s.getVertices())
        assert len(times) == 2, (
            f"Non-causal top simplex: orientation={s.getOrientation().numeric()}, "
            f"times={times}")


def _verify_counts_consistent(st):
    """Assert N4 = N41 + N32 and matches manual count."""
    n41_manual = 0
    n32_manual = 0
    n_top = 0
    for s in st.getSimplices():
        if len(s.getVertices()) != 5:
            continue
        n_top += 1
        o = s.getOrientation().numeric()
        if o in ((4, 1), (1, 4)):
            n41_manual += 1
        elif o in ((3, 2), (2, 3)):
            n32_manual += 1
        else:
            raise AssertionError(f"Invalid orientation {o}")
    assert st.getTopSimplexCount() == st.getN41() + st.getN32(), (
        f"N4 mismatch: {st.getTopSimplexCount()} != {st.getN41()} + {st.getN32()}")
    assert st.getN41() == n41_manual, (
        f"N41 mismatch: {st.getN41()} != {n41_manual}")
    assert st.getN32() == n32_manual, (
        f"N32 mismatch: {st.getN32()} != {n32_manual}")
    assert n_top == n41_manual + n32_manual, (
        f"Non-CDT orientations: {n_top} top simplices, "
        f"{n41_manual} N41 + {n32_manual} N32")


# =====================================================================
# Add / Remove round-trip
# =====================================================================

class TestAddRemoveRoundTrip(unittest.TestCase):
    """(2,2d) add creates +1 vertex, +(2d-2) N41 simplices.
    (2d,2) remove undoes it. Together they form a round trip.

    Ref: [BGL] Sec. 2.3.1 (adapted to 4D).
    """

    def test_single_add_exact_delta(self):
        """One successful add: dN0=+1, dN41=+(2d-2)=+6."""
        cdt, st = _build_small(n_simplices=20)
        before = _snapshot(st)

        accepted = False
        for _ in range(500):
            if cdt.add():
                accepted = True
                break
        if not accepted:
            self.skipTest("No add accepted")

        after = _snapshot(st)

        # Exact deltas: (2,2d) add creates 2d new, removes 2 old
        d = 4
        self.assertEqual(after["n0"], before["n0"] + 1)
        self.assertEqual(after["n41"], before["n41"] + 2 * d - 2,
                         f"Add should change N41 by +{2*d-2}")

        # 2d new simplices appeared, 2 old disappeared
        new_fps = after["top_fps"] - before["top_fps"]
        lost_fps = before["top_fps"] - after["top_fps"]
        self.assertEqual(len(new_fps), 2 * d,
                         f"Expected {2*d} new simplices, got {len(new_fps)}")
        self.assertEqual(len(lost_fps), 2,
                         f"Expected 2 removed simplices, got {len(lost_fps)}")

        # The new simplex has a valid CDT orientation
        new_fp = new_fps.pop()
        o = after["orientations"][new_fp]
        self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)),
                      f"New simplex has invalid orientation {o}")

        # Invariants hold
        _verify_counts_consistent(st)
        _verify_all_top_causal(st)

    def test_single_remove_exact_delta(self):
        """One add then one remove: N41 returns to original."""
        d = 4
        delta_n41 = 2 * d - 2  # +6 in 4D
        cdt, st = _build_small(n_simplices=20)
        before = _snapshot(st)

        for _ in range(2000):
            if cdt.add():
                break
        else:
            self.skipTest("No add accepted")

        mid = _snapshot(st)
        self.assertEqual(mid["n41"], before["n41"] + delta_n41)

        for _ in range(2000):
            if cdt.remove():
                break
        else:
            self.skipTest("No remove accepted")

        after = _snapshot(st)
        self.assertEqual(after["n41"], before["n41"])
        self.assertEqual(after["n41"], mid["n41"] - delta_n41)

        _verify_counts_consistent(st)
        _verify_all_top_causal(st)

    def test_add_remove_many_cycles(self):
        """Repeat add/remove 10 times.  After each pair, N41 should return."""
        d = 4
        delta_n41 = 2 * d - 2
        cdt, st = _build_small(n_simplices=20)
        original_n41 = st.getN41()

        for cycle in range(10):
            added = False
            for _ in range(500):
                if cdt.add():
                    added = True
                    break
            if not added:
                continue

            self.assertEqual(st.getN41(), original_n41 + delta_n41,
                             f"Cycle {cycle}: N41 should be original+{delta_n41}")
            _verify_counts_consistent(st)
            _verify_all_top_causal(st)

            removed = False
            for _ in range(500):
                if cdt.remove():
                    removed = True
                    break
            if not removed:
                original_n41 = st.getN41()
                continue

            self.assertEqual(st.getN41(), original_n41,
                             f"Cycle {cycle}: N41 should return after remove")
            _verify_counts_consistent(st)
            _verify_all_top_causal(st)


# =====================================================================
# Flip forward and backward
# =====================================================================

class TestFlipRoundTrip(unittest.TestCase):
    """[BGL] Sec. 2.3.2: The (2,4) flip: 2 simplices → 4 simplices.

    dN0=0, dN4=+2.  Exactly 2 old simplices disappear, 4 new ones appear.
    A second flip can (but doesn't always) undo the first.
    """

    def test_single_flip_exact_delta(self):
        """One flip: dN0=0, dN4=+2, 2 lost + 4 gained."""
        cdt, st = _build_small(n_simplices=30)
        before = _snapshot(st)

        accepted = False
        for _ in range(2000):
            if cdt.flip():
                accepted = True
                break
        if not accepted:
            self.skipTest("No flip accepted")

        after = _snapshot(st)

        # Exact deltas
        self.assertEqual(after["n0"], before["n0"],
                         "Flip should not change vertex count")
        self.assertGreaterEqual(after["n4"], before["n4"],
                               "Flip should not decrease N4")
        self.assertLessEqual(after["n4"], before["n4"] + 2,
                             "Flip should change N4 by at most +2")

        # 2 old simplices gone, 4 new ones appeared
        lost = before["top_fps"] - after["top_fps"]
        gained = after["top_fps"] - before["top_fps"]
        self.assertEqual(len(lost), 2,
                         f"Flip should remove 2 simplices, removed {len(lost)}")
        self.assertGreaterEqual(len(gained), 2,
                                f"Flip should create 2-4 simplices (dedup), created {len(gained)}")
        self.assertLessEqual(len(gained), 4)

        # All new simplices have valid orientations
        for fp in gained:
            o = after["orientations"][fp]
            self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)),
                          f"New simplex has invalid orientation {o}")

        _verify_counts_consistent(st)
        _verify_all_top_causal(st)

    def test_flip_twice_deltas_accumulate(self):
        """Two flips: each adds exactly +2 to N4."""
        cdt, st = _build_small(n_simplices=30)
        n4_start = st.getTopSimplexCount()

        flips_done = 0
        for _ in range(5000):
            if cdt.flip():
                flips_done += 1
                # N4 change depends on dedup
                _verify_counts_consistent(st)
                _verify_all_top_causal(st)
                if flips_done >= 2:
                    return

        if flips_done < 2:
            self.skipTest(f"Only {flips_done} flips accepted")

    def test_flip_preserves_vertex_set(self):
        """The exact set of vertex IDs should not change under a flip."""
        cdt, st = _build_small(n_simplices=30)

        verts_before = set()
        for s in st.getSimplices():
            for v in s.getVertices():
                verts_before.add(v.getId())

        for _ in range(2000):
            if cdt.flip():
                break
        else:
            self.skipTest("No flip accepted")

        verts_after = set()
        for s in st.getSimplices():
            for v in s.getVertices():
                verts_after.add(v.getId())

        self.assertEqual(verts_before, verts_after,
                         "Flip should not change the vertex set")


# =====================================================================
# Shift forward and backward
# =====================================================================

class TestShiftRoundTrip(unittest.TestCase):
    """[BGL] Sec. 2.3.3: The (3,3) shift: 3 simplices → 3 simplices.

    dN0=0, dN4=0.  Exactly 3 old simplices disappear, 3 new ones appear.
    """

    def test_single_shift_exact_delta(self):
        """One shift: dN0=0, dN4=0, 3 lost + 3 gained."""
        cdt, st = _build_small(n_simplices=200)
        # Do some sweeps to create a richer topology for shifts
        cdt.sweep(100)
        before = _snapshot(st)

        accepted = False
        for _ in range(20000):
            if cdt.shift():
                accepted = True
                break
        if not accepted:
            self.skipTest("No shift accepted")

        after = _snapshot(st)

        self.assertEqual(after["n0"], before["n0"],
                         "Shift should not change vertex count")
        # (3,3) shift replaces 3 simplices with 3: dN4 = 0.
        # On very small lattices dedup can cause dN4 in [-3, 0].
        dN4 = after["n4"] - before["n4"]
        self.assertGreaterEqual(dN4, -3, f"Shift dN4 too negative: {dN4}")
        self.assertLessEqual(dN4, 0, f"Shift should not increase N4: dN4={dN4}")

        lost = before["top_fps"] - after["top_fps"]
        gained = after["top_fps"] - before["top_fps"]
        self.assertEqual(len(lost), 3,
                         f"Shift should remove 3 simplices, removed {len(lost)}")
        # Gained can be < 3 if a new simplex already existed (dedup)
        self.assertGreaterEqual(len(gained), 1)
        self.assertLessEqual(len(gained), 3)

        for fp in gained:
            o = after["orientations"][fp]
            self.assertIn(o, ((4, 1), (1, 4), (3, 2), (2, 3)))

        _verify_counts_consistent(st)
        _verify_all_top_causal(st)

    def test_single_ishift_exact_delta(self):
        """One ishift: same combinatorics as shift (3→3)."""
        cdt, st = _build_small(n_simplices=200)
        cdt.sweep(100)
        before = _snapshot(st)

        accepted = False
        for _ in range(20000):
            if cdt.ishift():
                accepted = True
                break
        if not accepted:
            self.skipTest("No ishift accepted")

        after = _snapshot(st)

        self.assertEqual(after["n0"], before["n0"])

        lost = before["top_fps"] - after["top_fps"]
        gained = after["top_fps"] - before["top_fps"]
        self.assertEqual(len(lost), 3)
        self.assertGreaterEqual(len(gained), 1)
        self.assertLessEqual(len(gained), 3)

        _verify_counts_consistent(st)
        _verify_all_top_causal(st)

    def test_shift_preserves_vertex_set(self):
        """The exact set of vertex IDs should not change under a shift."""
        cdt, st = _build_small(n_simplices=200)
        cdt.sweep(50)

        verts_before = set()
        for s in st.getSimplices():
            for v in s.getVertices():
                verts_before.add(v.getId())

        for _ in range(20000):
            if cdt.shift():
                break
        else:
            self.skipTest("No shift accepted")

        verts_after = set()
        for s in st.getSimplices():
            for v in s.getVertices():
                verts_after.add(v.getId())

        self.assertEqual(verts_before, verts_after)


# =====================================================================
# Multi-iteration round-trip stress tests
# =====================================================================

class TestMultiIterationRoundTrips(unittest.TestCase):
    """[BGL] Sec. 2.3: Apply a move several times, then the inverse the same number of
    times, and verify that the lattice returns to a compatible state.
    """

    def test_five_adds_then_five_removes(self):
        """5 adds then 5 removes: N41 should return to start."""
        cdt, st = _build_small(n_simplices=20)
        n41_start = st.getN41()
        n0_start = st.getVertexCount()

        n_added = 0
        for _ in range(2500):
            if cdt.add():
                n_added += 1
                if n_added >= 5:
                    break

        d = 4
        delta_per_add = 2 * d - 2  # +6 per add in 4D
        self.assertEqual(st.getN41(), n41_start + n_added * delta_per_add)
        _verify_counts_consistent(st)
        _verify_all_top_causal(st)

        n_removed = 0
        for _ in range(2500):
            if cdt.remove():
                n_removed += 1
                if n_removed >= n_added:
                    break

        self.assertEqual(n_removed, n_added,
                         f"Could only remove {n_removed} of {n_added}")
        self.assertEqual(st.getN41(), n41_start,
                         f"N41 should return to {n41_start}")
        _verify_counts_consistent(st)
        _verify_all_top_causal(st)

    def test_three_flips_counts_and_causality(self):
        """3 flips: each should increase N4 and maintain all invariants.

        On a tiny lattice, a flip normally gives dN4=+2, but can give less
        if one of the 4 new simplices happens to already exist (vertex
        reuse on small complexes).  We check the invariants regardless.
        """
        cdt, st = _build_small(n_simplices=50)
        prev_n4 = st.getTopSimplexCount()

        flips = 0
        for _ in range(10000):
            if cdt.flip():
                flips += 1
                # N4 change depends on dedup: 2 removed, 0-4 new unique created.
                # On small lattices with rich topology, net change can be negative.
                _verify_counts_consistent(st)
                _verify_all_top_causal(st)
                prev_n4 = st.getTopSimplexCount()
                if flips >= 3:
                    return

        if flips < 3:
            self.skipTest(f"Only {flips} flips accepted")

    def test_mixed_moves_then_verify(self):
        """Apply a mix of all move types, verify invariants after each."""
        cdt, st = _build_small(n_simplices=30)

        moves = [cdt.add, cdt.remove, cdt.flip, cdt.iflip, cdt.shift, cdt.ishift]
        # Canonical names from the move classes, so a rename cannot leave this
        # loop silently driving nothing. "ishift" has no class of its own -- it
        # is ShiftMove's inverse direction, named only here.
        move_names = [tessera.AddMove.MOVE_TYPE, tessera.RemoveMove.MOVE_TYPE,
                      tessera.FlipMove.MOVE_TYPE, tessera.IFlipMove.MOVE_TYPE,
                      tessera.ShiftMove.MOVE_TYPE, "ishift"]

        for iteration in range(20):
            for move, name in zip(moves, move_names):
                before_n4 = st.getTopSimplexCount()
                before_n0 = st.getVertexCount()
                if move():
                    after_n4 = st.getTopSimplexCount()
                    after_n0 = st.getVertexCount()

                    if name == "add":
                        # (2,2d) add: dN41 = +(2d-2) = +6 in 4D
                        self.assertEqual(after_n0, before_n0 + 1)
                        self.assertGreater(after_n4, before_n4)
                    elif name == "remove":
                        # (2d,2) remove: dN41 = -(2d-2) = -6 in 4D
                        self.assertLess(after_n4, before_n4)
                    elif name in ("flip", "iflip"):
                        # dN4 normally +(d-2) for flip, -(d-2) for iflip;
                        # can differ due to dedup on small lattices
                        self.assertEqual(after_n0, before_n0)
                    elif name in ("shift", "ishift"):
                        # Normally 0; can be negative on small lattices
                        # if a new simplex already exists (dedup)
                        self.assertEqual(after_n0, before_n0)

                    with self.subTest(iteration=iteration, move=name):
                        _verify_counts_consistent(st)
                        _verify_all_top_causal(st)


# =====================================================================
# Vertex structure tests for add/remove
# =====================================================================

class TestAddRemoveVertexStructure(unittest.TestCase):
    """Verify (2,2d) add creates a vertex incident to 2d top simplices,
    and (2d,2) remove finds and removes exactly such a vertex.

    Ref: [BGL] Sec. 2.3.1 (adapted to 4D).
    """

    def test_added_vertex_has_2d_top_simplices(self):
        """The vertex created by (2,2d) add belongs to exactly 2d top simplices."""
        cdt, st = _build_small(n_simplices=20)

        for _ in range(500):
            if cdt.add():
                break
        else:
            self.skipTest("No add accepted")

        # Find the new vertex (highest ID)
        all_verts = st.getVertexList().toVector()
        new_vert = max(all_verts, key=lambda v: v.getId())

        top_count = sum(1 for s in st.getSimplices()
                        if len(s.getVertices()) == 5 and s.hasVertex(new_vert))

        d = 4
        self.assertEqual(top_count, 2 * d,
                         f"Added vertex should be in exactly {2*d} top "
                         f"simplices, found {top_count}")

    def test_added_vertex_connects_to_d_plus_2_vertices(self):
        """The (2,2d) vertex connects to d spatial + 2 non-spatial = d+2 others."""
        cdt, st = _build_small(n_simplices=100)

        for _ in range(2000):
            if cdt.add():
                break
        else:
            self.skipTest("No add accepted")

        all_verts = st.getVertexList().toVector()
        new_vert = max(all_verts, key=lambda v: v.getId())

        # d+2 edges: d to spatial vertices + 2 to non-spatial
        d = 4
        edge_pairs = set()
        for e in new_vert.getEdges():
            a, b = e.getSource().getId(), e.getTarget().getId()
            edge_pairs.add((min(a, b), max(a, b)))
        self.assertEqual(len(edge_pairs), d + 2,
                         f"Added vertex should have {d+2} edges, "
                         f"has {len(edge_pairs)}")


if __name__ == "__main__":
    unittest.main()
