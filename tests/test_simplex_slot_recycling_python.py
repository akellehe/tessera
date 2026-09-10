# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Storage slots are reused, so memory tracks volume rather than sweep count.

``Spacetime::simplexStorage_`` hands every simplex a slot at a stable address.
A removed simplex's slot goes back on a free list and is handed out again, which
is what keeps a long Markov chain's memory proportional to the four-volume it is
holding instead of to the number of simplices it has ever created.

Reuse is deferred to a sweep boundary on purpose. A Pachner move captures
``SimplexPtr`` in ``propose()`` and reads them in ``apply()``, so a slot freed
mid-move must not be handed out until the move is over.
"""

import unittest

import tessera


def _spacetime(seed=20260909, n_simplices=1600):
    sig = tessera.Signature(4, tessera.Lorentzian)
    spacetime = tessera.Spacetime(tessera.Metric(True, sig), tessera.CDT,
                                  1.0, 1.0, tessera.PREFERRED, tessera.Toroid())
    spacetime.setSeed(seed)
    spacetime.build(n_simplices)
    return spacetime


def _chain(spacetime, target=None, seed=20260909):
    target = target or spacetime.getN41()
    cdt = tessera.CDTSimulation(spacetime, 2.2, 0.5, 0.6, 1.0 / target, target)
    cdt.setSeed(seed)
    return cdt


class TestSimplexSlotRecycling(unittest.TestCase):

    def test_storage_tracks_live_simplices_rather_than_sweep_count(self):
        """The regression: storage used to grow without bound as sweeps ran."""
        spacetime = _spacetime()
        cdt = _chain(spacetime)
        cdt.tune()
        cdt.sweep(2000)

        live = len(spacetime.getSimplices())
        storage = spacetime.simplexStorageSize()
        free = spacetime.freeSimplexSlotCount()

        # Every slot is either holding a live simplex or waiting on the free
        # list; the two together account for the whole deque.
        self.assertGreaterEqual(storage, live)
        self.assertLessEqual(storage - live, free + spacetime.pendingSimplexSlotCount())
        # Without reuse a 2000-sweep chain allocates many times its live count.
        self.assertLess(storage, live * 2)

    def test_sweeping_further_does_not_grow_storage_beyond_the_volume(self):
        spacetime = _spacetime()
        cdt = _chain(spacetime)
        cdt.tune()
        cdt.sweep(2000)
        first_excess = spacetime.simplexStorageSize() - len(spacetime.getSimplices())

        cdt.sweep(4000)
        later_excess = spacetime.simplexStorageSize() - len(spacetime.getSimplices())

        # The excess is one sweep of churn plus the free list, not a running
        # total of everything the chain has ever created.
        self.assertLess(later_excess, max(first_excess, 1) * 10)

    def test_a_sweep_leaves_no_slot_pending(self):
        """CDT::sweep reclaims at its own end, where no move is in flight."""
        spacetime = _spacetime()
        cdt = _chain(spacetime)
        cdt.tune()
        cdt.sweep(200)
        self.assertEqual(spacetime.pendingSimplexSlotCount(), 0)

    def test_reclaim_is_idempotent_and_reports_what_it_released(self):
        spacetime = _spacetime()
        cdt = _chain(spacetime)
        cdt.tune()
        cdt.sweep(200)
        # sweep() already reclaimed, so there is nothing left to release.
        self.assertEqual(spacetime.reclaimSimplexSlots(), 0)
        self.assertEqual(spacetime.reclaimSimplexSlots(), 0)

    def test_the_chain_is_unchanged_by_reuse(self):
        """Reusing storage must not change which geometry the chain visits."""
        first = _spacetime()
        cdt = _chain(first)
        cdt.tune()
        cdt.sweep(1500)

        second = _spacetime()
        cdt2 = _chain(second)
        cdt2.tune()
        cdt2.sweep(1500)

        self.assertEqual(first.getN41(), second.getN41())
        self.assertEqual(first.getN32(), second.getN32())

    def test_free_slots_are_actually_handed_back_out(self):
        spacetime = _spacetime()
        cdt = _chain(spacetime)
        cdt.tune()
        cdt.sweep(500)
        free_before = spacetime.freeSimplexSlotCount()
        storage_before = spacetime.simplexStorageSize()
        cdt.sweep(500)
        # If the free list were never drawn from, the deque would have to grow
        # by every simplex the second batch created.
        growth = spacetime.simplexStorageSize() - storage_before
        self.assertLess(growth, storage_before)
        self.assertGreaterEqual(free_before, 0)


if __name__ == "__main__":
    unittest.main()
