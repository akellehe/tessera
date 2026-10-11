# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The objective is a pure function of the complex, and the node's complex is
the one the node is driving.

#864 reported the objective descending across engine units while the complex
appeared bit-for-bit unchanged, which would have made it a function of hidden
mutable state. It is not. The complex being inspected was the `Spacetime`
handed to the constructor, and stage 1 REPLACES the node's complex when it
commits a move, so that handle goes stale from the first committed move onward.
The objective was tracking the node's live complex the whole time.

Two properties are pinned here because each failed silently and neither would
raise:

  * evaluating the objective twice on ONE complex returns the SAME double, so
    the objective is pure;
  * `spacetime()` is the complex to read after a drive, and it diverges from
    the constructor argument once a move lands.
"""

import cmath
import unittest
import pytest

import tessera as T


cob = T.cobordism


def _sphere4(jitter=True):
    """A closed 4-manifold with genuinely complex, non-degenerate lengths."""
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)),
                     T.CDT, 1.0, 1.0, T.PREFERRED,
                     T.SimplexBoundarySphere(4))
    st.build()
    for index, edge in enumerate(st.get_edge_list().to_vector()):
        z = (complex(1.0 + 0.017 * (index % 5), 0.013 * (1 + index % 4))
             if jitter else complex(1.0))
        edge.set_length(cmath.sqrt(z))
    return st


def _refined_ball4(n_refine=4, seed=3):
    """A single 4-simplex refined by stellar adds -- a 4-BALL with a boundary.

    Used where a test needs stage 1 to actually COMMIT a move. The closed
    `_sphere4` above is too rigid for that: a batch of candidates routinely
    finds none that lowers the objective, so an assertion resting on a
    committed move would skip rather than run there.
    """
    st = T.Spacetime(T.Metric(True, T.Signature(4, T.Lorentzian)), T.CDT,
                     1.0, 1.0, T.PREFERRED, T.SolidSimplex(4))
    st.build()
    for edge in st.get_edge_list().to_vector():
        edge.set_length(complex(1.0, 0.0))
    applied = 0
    for step in range(seed, seed + n_refine * 4):
        move = T.AddMove(st, step, False, T.PachnerMode.PreGeometric, False)
        if move.propose() and move.apply():
            applied += 1
        if applied >= n_refine:
            break
    for index, edge in enumerate(st.get_edge_list().to_vector()):
        z = complex(1.0 + 0.017 * (index % 5), 0.013 * (1 + index % 4))
        edge.set_length(cmath.sqrt(z))
    return st


def _node(st, seed=7):
    node = cob.MultiCobordism(st, [], [], degrees=[1], gamma=0.0, seed=seed)
    node.set_objective(cob.JointStationarityObjective())
    node.set_simulation_mode(cob.MultiCobordism.SimulationMode.EMERGENCE,
                             cob.MultiCobordism.EmergenceSubmode.STRICT)
    return node


def _fingerprint(st):
    """Everything that defines the complex, as comparable data.

    The cell tuples keep their INTRINSIC stored order rather than being sorted,
    so an orientation flip shows up here rather than being normalized away.

    The lattice is materialized to a fixpoint FIRST. `get_facets()` creates
    facets and wires coface links as a side effect, so a complex that has been
    scored has more simplices listed than one that has not -- through pure
    bookkeeping, with no geometry or topology added. Without this call a
    fingerprint taken before a drive and one taken after would differ for that
    reason alone, and the difference would look like a move that never
    happened.
    """
    st.materialize_facets()
    cells = [tuple(int(v.get_id()) for v in s.get_vertices())
             for s in st.get_simplices()]
    edges = [(int(e.get_source().get_id()), int(e.get_target().get_id()),
              complex(e.get_length()), complex(e.get_phase()))
             for e in st.get_edge_list().to_vector()]
    return sorted(cells), sorted(edges, key=lambda row: row[:2])


def _top_cells(st):
    """The maximal-dimension cells, as intrinsic vertex tuples.

    This is the one indicator that isolates a COMMITTED MOVE from the two
    things that also change a complex's appearance without being one:

      * stage 2 relaxes edge lengths IN PLACE, so lengths move while the node
        is still driving the very object handed to the constructor;
      * `get_facets()` materializes lower faces as a side effect of being read,
        so the simplex count grows through bookkeeping alone.

    Neither touches the top cells. Only surgery does.
    """
    st.materialize_facets()
    cells = [tuple(int(v.get_id()) for v in s.get_vertices())
             for s in st.get_simplices()]
    if not cells:
        return []
    top = max(len(cell) for cell in cells)
    return sorted(cell for cell in cells if len(cell) == top)


def _drive_until_the_node_replaces_its_complex(node, st, attempts=24):
    """Drive until stage 1 commits a move, which is when `st` goes stale.

    Stage 1 commits only a move that lowers the objective, and on a small host
    a batch of candidates can find none, so this is a search rather than a
    guarantee. Returns whether the node's top cells left `st` behind.
    """
    before = _top_cells(st)
    for _ in range(attempts):
        list(node.run_stage1(max_steps=1, n_candidate_moves=8))
        list(node.run_stage2(max_iters=6))
        if _top_cells(node.spacetime()) != before:
            return True
    return False


class ObjectiveIsPureInTheComplexTest(unittest.TestCase):
    """Same complex in, same double out."""

    def test_repeated_evaluation_on_one_complex_is_bit_identical(self):
        node = _node(_sphere4())
        first = node.objective()
        for _ in range(4):
            self.assertEqual(node.objective(), first)

    def test_a_second_node_on_the_same_complex_agrees_bit_identically(self):
        """Purity across INSTANCES, not merely across calls on one instance.

        A per-node cache would satisfy the repeated-call test above while still
        making the objective a function of that node's history.
        """
        st = _sphere4()
        self.assertEqual(_node(st).objective(), _node(st, seed=99).objective())

    def test_materializing_the_skeleton_does_not_move_the_objective(self):
        """The #850 invariant.

        `get_facets()` creates facets and wires coface links as a side effect,
        and `Simplex::dualVolume` walks those links, so an objective read
        before the lattice is complete could differ from one read after. It
        must not: materialization adds no geometry, so it may not change a
        geometric functional.
        """
        st = _sphere4()
        node = _node(st)
        before = node.objective()
        st.materialize_facets()
        self.assertEqual(node.objective(), before)


class TheNodeDrivesItsOwnComplexTest(unittest.TestCase):
    """`spacetime()` is the complex to read; the constructor argument is not."""

    def test_the_accessor_is_the_constructor_argument_before_any_drive(self):
        st = _sphere4()
        node = _node(st)
        self.assertEqual(_fingerprint(node.spacetime()), _fingerprint(st))

    def test_the_objective_matches_the_accessor_and_not_the_stale_handle(self):
        """The #864 measurement, as an assertion.

        After a drive that commits a move, scoring the constructor argument
        gives a DIFFERENT number from `objective()`, while scoring the
        accessor's complex gives the same one. That asymmetry is the whole
        defect: a reader holding the stale handle sees a frozen complex and
        cannot tell.
        """
        st = _refined_ball4()
        node = _node(st)
        if not _drive_until_the_node_replaces_its_complex(node, st):
            self.skipTest("no move landed on this host, so nothing went stale")

        live = node.objective_terms_for(node.spacetime())
        stale = node.objective_terms_for(st)
        total = (lambda t: t.regge_stationarity + t.hodge_stationarity
                 + t.connection_stationarity + t.register_residual
                 + t.action_magnitude + t.carried_state_energy)
        self.assertEqual(total(live), node.objective())
        self.assertNotEqual(total(stale), node.objective())

    # Measured at 49 s in CI, about 50 seconds: above the 30 s limit of the
    # pull-request tier, so it runs on main and nightly only.
    @pytest.mark.slow
    def test_the_stale_handle_stops_moving_once_a_move_lands(self):
        """The reason the animation's complex panel never updated.

        Note the handle is NOT frozen from construction: until the first
        committed move the node is still driving that very object, so stage 2
        relaxes its lengths in place. It goes stale at the moment stage 1
        replaces the node's complex, and from then on a driver reading it
        redraws one picture forever while the node evolves something else.
        """
        st = _refined_ball4()
        node = _node(st)
        if not _drive_until_the_node_replaces_its_complex(node, st):
            self.skipTest("no move landed on this host, so nothing went stale")

        stranded = _fingerprint(st)
        for _ in range(12):
            list(node.run_stage1(max_steps=1, n_candidate_moves=8))
            list(node.run_stage2(max_iters=6))
        # The stranded handle took no further update of any kind -- not a
        # length, not a cell -- while the node kept driving.
        self.assertEqual(_fingerprint(st), stranded)
        self.assertNotEqual(_top_cells(node.spacetime()), _top_cells(st))


if __name__ == "__main__":
    unittest.main()
