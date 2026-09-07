# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A cone move may not extend the boundary unless asked (#997).

A cone-in over a boundary facet buries that facet and exposes the new cell's
remaining facets; a cone-out removes a cell and exposes every facet it had.
Either hands ``dW`` faces it did not have, which silently changes what a
cobordism IS when its boundary is stated up front. ``set_boundary_may_extend``
gates it: with False a ``cone_out``, ``cone_in`` or ``cone_in_timelike``
candidate whose complex has a boundary facet the complex before the move
lacked is refused, beside the manifold gate and on the same footing. The engine's
default is False, and the gate applies only where a boundary is DECLARED
fixed (``has_fixed_boundary``: a surface input or output block, or a pinned
region), so a free emergent build -- a proton host grown from a simplex
seed, a CDT sweep -- is unrestricted either way and behaves exactly as it
did. ``emergence_animation.py`` records the choice in its run document.
"""
import os
import sys

import pytest

from tessera import cobordism as cob

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

MC = cob.MultiCobordism


def simplex_node(seed=0, precone=0):
    return MC(MC.seed_simplex(3), [], [], degrees=[1], seed=seed, precone=precone,
              einstein_hilbert=False)


def boundary(node):
    return {tuple(f) for f in MC.boundary_facets(node.spacetime())}


def cells(node):
    return sorted(tuple(sorted(v.getId() for v in c.getVertices()))
                  for c in node.spacetime().getTopSimplices())


def test_the_boundary_of_a_simplex_is_all_of_its_facets():
    node = simplex_node()
    faces = boundary(node)
    # a single 3-simplex: four triangles, each on exactly one cell
    assert len(cells(node)) == 1 and len(faces) == 4
    for facet in faces:
        assert len(facet) == 3
    assert all(sorted(f) == list(f) for f in faces), "facets are sorted tuples"


def test_the_engine_defaults_to_disallowing_it():
    node = simplex_node()
    assert node.boundary_may_extend is False


def test_a_node_without_a_declared_boundary_is_unrestricted():
    """A free emergent build declares no fixed boundary, so the gate does not
    apply to it and its cones behave exactly as they did before the gate."""
    free, control = simplex_node(seed=5, precone=4), simplex_node(seed=5, precone=4)
    assert not free.has_fixed_boundary
    control.set_boundary_may_extend(True)
    for node in (free, control):
        for _ in range(6):
            node.run_stage1(1, 6, grow_boundaries=False)
    assert cells(free) == cells(control)
    assert boundary(free) == boundary(control)


def collar_node(seed_value=7):
    """A node whose boundary IS declared: two flat tori on their collar, the
    surface input blocks of the qubit experiment."""
    import warnings

    from tessera import observables as obs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tori = [obs.SimplicialQubit.flat_torus(tau, 3, 3)
                for tau in (complex(0.3, 1.1), complex(-0.2, 0.8))]
    seed = MC.seed_collar(tori[0].spacetime(), tori[1].spacetime(), 1)
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1],
              seed=seed_value, einstein_hilbert=False)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    return node


def test_a_pinned_region_is_not_a_declared_boundary():
    """Pinning constrains the geometry and does not veto a topology change
    (`declare_pinned_region`), so it does not arm this gate."""
    node = simplex_node()
    node.declare_pinned_region("M0", set(range(2)))
    assert not node.has_fixed_boundary


def test_the_gate_refuses_a_cone_that_grows_a_declared_boundary():
    """With a boundary declared -- surface input blocks -- a cone-in on a
    fresh apex always hands it new faces, so no cone survives and the complex
    is untouched."""
    gated, open_ = collar_node(), collar_node()
    for node in (gated, open_):
        assert node.has_fixed_boundary
    open_.set_boundary_may_extend(True)
    assert gated.boundary_may_extend is False and open_.boundary_may_extend is True
    before_cells, before_boundary = cells(gated), boundary(gated)
    for _ in range(6):
        gated.run_stage1(1, 8, grow_boundaries=False)
    assert cells(gated) == before_cells, "no cone survived the gate"
    assert boundary(gated) == before_boundary


def test_the_flag_is_bit_identical_to_the_engine_default():
    """With the knob True the gate does nothing: two nodes that differ only
    in having been told so explicitly take the same moves."""
    explicit, default = simplex_node(seed=3, precone=4), simplex_node(seed=3, precone=4)
    for node in (explicit, default):
        node.declare_pinned_region("M0", set(range(2)))
    explicit.set_boundary_may_extend(True)
    default.set_boundary_may_extend(True)
    for node in (explicit, default):
        for _ in range(6):
            node.run_stage1(1, 6, grow_boundaries=False)
    assert cells(explicit) == cells(default)
    assert boundary(explicit) == boundary(default)


def test_the_driver_disallows_growth_unless_asked():
    parser = ea.build_parser()
    assert parser.parse_args(["run"]).extend_boundary is False
    assert parser.parse_args(["run", "--extend-boundary"]).extend_boundary is True
    assert ea.DECLARED_EXTEND_BOUNDARY is False
    assert ea.build_config()["extend_boundary"] is False
    assert ea.build_config(extend_boundary=True)["extend_boundary"] is True


def test_the_drive_applies_the_config_to_its_node(monkeypatch):
    """`drive` sets the knob on whatever its factory returned, in one place
    for both modes. The node is not otherwise reachable from a caller, so the
    factory is intercepted to hold it."""
    held = []
    original = ea.MC

    class Recording(original):
        def set_boundary_may_extend(self, allowed):
            held.append(bool(allowed))
            return original.set_boundary_may_extend(self, allowed)

    monkeypatch.setattr(ea, "MC", Recording)
    for requested in (False, True):
        held.clear()
        ea.drive(ea.build_config(size=4, steps=0, stage1_iters=1, stage2_iters=1,
                                 extend_boundary=requested), progress=False)
        assert held == [requested], held
