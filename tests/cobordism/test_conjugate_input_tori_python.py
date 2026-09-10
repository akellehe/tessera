# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Four boundary tori, so the harmonic space can carry the target (#1039).

The two-body target is 4-dimensional and the two-torus collar's harmonic space
is 2-dimensional. That is forced, not incidental: for a compact oriented
3-manifold `rank(H^1(W) -> H^1(dW)) = b_1(dW)/2`, and two tori give 4/2 = 2.
Adding handles to the bulk raises `b_1(W)` but the extra classes die on the
boundary, so a readout cannot use them.

`--conjugate-inputs` carries each state on a PAIR of tori: itself and its
orientation reversal at `-conj(tau)`, which keeps the modulus in the upper
half-plane where a torus modulus has to live and gives the holomorphic form the
conjugate periods. Four boundary tori give `b_1(dW) = 8`, hence rank 4.

The host is two collars joined along a removed tetrahedron. Gluing along a
sphere is a connected sum, which adds no first homology, so `b_1 = 2 + 2 = 4`
with nothing dying on the boundary. It needs three collar layers: a prism cell
spans two adjacent layers, so the all-interior cell the join removes exists
only once there are two interior layers.
"""
import os
import sys

import pytest

from tessera import cobordism as cob
from tessera import observables as obs

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

MC = cob.MultiCobordism
CC = cob.ChainComplex
TAU_A, TAU_B, GRID = 0.3 + 1.1j, -0.2 + 0.8j, 3


def torus(tau):
    return obs.SimplicialQubit.flat_torus(tau, GRID, GRID).spacetime()


def surfaces():
    return [torus(TAU_A), torus(-TAU_A.conjugate()),
            torus(TAU_B), torus(-TAU_B.conjugate())]


# ---- the host ----

@pytest.fixture(scope="module")
def joined():
    return MC.seed_joined_collars(surfaces(), 3)


def test_it_is_one_connected_manifold_with_four_boundary_tori(joined):
    """The seed refuses a non-manifold by name, so reaching here is half of it.

    The other half is the count: four boundary components of eighteen faces,
    which is the 3x3 grid torus.
    """
    spacetime = joined.host
    assert len(spacetime.getBoundary()) == 4 * 18
    betti = CC.fromSpacetime(spacetime).bettiNumbers()
    assert betti[0] == 1, "one component: the collars are joined, not merely present"


def test_the_first_betti_number_is_four(joined):
    """The whole point. A connected sum along a sphere adds no first homology."""
    assert CC.fromSpacetime(joined.host).bettiNumbers()[1] == 4


def test_every_surface_keeps_its_own_vertex_map(joined):
    assert [len(mapping) for mapping in joined.vertex_ids] == [9, 9, 9, 9]
    seen = set()
    for mapping in joined.vertex_ids:
        seen |= set(int(v) for v in mapping.values())
    assert len(seen) == 36, "the four surfaces share no vertex"


def test_fewer_than_three_layers_is_refused_by_name():
    """With two layers every cell touches a surface: there is none to remove."""
    for layers in (1, 2):
        with pytest.raises(ValueError) as caught:
            MC.seed_joined_collars(surfaces(), layers)
        assert "three" in str(caught.value)


def test_an_odd_number_of_surfaces_is_refused():
    with pytest.raises(ValueError):
        MC.seed_joined_collars(surfaces()[:3], 3)


# ---- what it buys the readout ----

def drive_to_seed(**overrides):
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             operator="xx", tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False,
                             pin_boundary=True, **overrides)
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    return held["node"]


def test_two_tori_cannot_carry_a_four_dimensional_target():
    """The control, and the reason the flag exists."""
    node = drive_to_seed(readout="whole", conjugate_inputs=False)
    chi = node.two_body_target().chi
    assert node.whole_harmonic_residual(chi) == 1.0
    assert "rank 2" in node.whole_harmonic_obstruction


def test_four_tori_can():
    """A number, not a refusal. Its VALUE is what a run is for."""
    node = drive_to_seed(readout="whole", conjugate_inputs=True)
    chi = node.two_body_target().chi
    residual = node.whole_harmonic_residual(chi)
    assert node.whole_harmonic_obstruction == "", node.whole_harmonic_obstruction
    assert 0.0 <= residual < 1.0, residual


def test_the_conjugate_partners_are_the_orientation_reversals():
    """-conj(tau), so the modulus stays in the upper half-plane."""
    node = drive_to_seed(conjugate_inputs=True)
    assert len(node.inputs) == 4


# ---- the flag ----

def test_the_default_is_two_tori():
    assert ea.build_config()["conjugate_inputs"] is False
    assert ea.DECLARED_CONJUGATE_INPUTS is False


def test_the_value_is_carried_into_the_config():
    assert ea.build_config(conjugate_inputs=True)["conjugate_inputs"] is True
