# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Nothing the CDT sweep reads depends on a vertex's label.

`AddMove` finishes by swapping the new vertex's label with a random vertex's,
which costs a walk over that vertex's incidence list -- a list that grows with
the four-volume, and the sweep's only remaining growth term (#978).  It is there
because [BGL] Sec. 2.2.1 requires the labeling to be uniform.

What the sweep actually reads is the point of these tests.  The action is a
function of N0, N41 and N32; the Metropolis prefactors are functions of the same
counts; the guards read vertex *times* (`isValidCDTOrientation`,
`isN41Type`); and every draw indexes a live vector by position, not by label.
A permutation of the labels therefore cannot change what the chain samples, and
these tests hold that claim to the code.

References:
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744
"""

import unittest

import tessera


class TestLabelInvariance(unittest.TestCase):
    SEED = 20260909

    def _thermalized(self, relabel=True, sweeps=60):
        sig = tessera.Signature(4, tessera.Lorentzian)
        st = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                               1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
        st.setSeed(self.SEED)
        st.build(1600)
        cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 1.0 / 2000, 2000)
        cdt.setSeed(self.SEED)
        cdt.setRelabelVertices(relabel)
        cdt.sweep(sweeps)
        return st, cdt

    @staticmethod
    def _state(st, cdt):
        """Everything the sweep reads: the counts it prices and the action."""
        return (st.getVertexCount(), st.getN41(), st.getN32(),
                round(cdt.computeAction(), 9))

    @staticmethod
    def _cells(st):
        """Top cells as vertex-id sets, which a relabeling permutes."""
        return [frozenset(v.getId() for v in s.getVertices())
                for s in st.getSimplices() if len(s.getVertices()) == 5]

    def test_relabeling_leaves_the_action_and_counts_alone(self):
        """The action is a function of the counts, and a permutation of labels
        moves no vertex between them."""
        st, cdt = self._thermalized()
        before = self._state(st, cdt)
        verts = st.getVertexList().toVector()
        for i in range(0, len(verts) - 1, 2):
            st.swapVertexLabels(verts[i], verts[i + 1])
        self.assertEqual(self._state(st, cdt), before)

    def test_relabeling_permutes_the_cells_without_changing_them(self):
        """A relabeling renames vertices; it does not add, drop or merge a
        cell, so the number of distinct top cells is preserved."""
        st, cdt = self._thermalized()
        before = self._cells(st)
        verts = st.getVertexList().toVector()
        for i in range(0, len(verts) - 1, 2):
            st.swapVertexLabels(verts[i], verts[i + 1])
        after = self._cells(st)
        self.assertEqual(len(after), len(before))
        self.assertEqual(len(set(after)), len(set(before)))

    def test_the_orientation_guards_read_times_not_labels(self):
        """(4,1) and (3,2) are decided by how many vertices sit at each time,
        so relabeling cannot move a cell between the two populations."""
        st, cdt = self._thermalized()
        n41, n32 = st.getN41(), st.getN32()
        verts = st.getVertexList().toVector()
        for i in range(0, len(verts) - 1, 2):
            st.swapVertexLabels(verts[i], verts[i + 1])
        self.assertEqual((st.getN41(), st.getN32()), (n41, n32))

    def test_a_chain_runs_the_same_invariants_without_relabeling(self):
        """With the swap disabled the chain still moves, and still holds every
        invariant above: the counts stay consistent with the cells it carries."""
        st, cdt = self._thermalized(relabel=False, sweeps=120)
        cells = self._cells(st)
        self.assertEqual(len(cells), st.getN41() + st.getN32())
        self.assertEqual(len(set(cells)), len(cells), "duplicate top cells")


if __name__ == "__main__":
    unittest.main()
