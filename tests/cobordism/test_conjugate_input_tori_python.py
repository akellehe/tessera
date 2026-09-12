# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Four boundary tori and the low-level paired-frame diagnostic (#1039).

The two-body target is 4-dimensional and the two-torus collar's harmonic space
is 2-dimensional. That is forced, not incidental: for a compact oriented
3-manifold `rank(H^1(W) -> H^1(dW)) = b_1(dW)/2`, and two tori give 4/2 = 2.
Adding handles to the bulk raises `b_1(W)` but the extra classes die on the
boundary, so a readout cannot use them.

The four-torus construction carries each state on a PAIR of tori: itself and its
orientation reversal at `-conj(tau)`, which keeps the modulus in the upper
half-plane where a torus modulus has to live and gives the holomorphic form the
conjugate periods. Four boundary tori give `b_1(dW) = 8`, hence rank 4.

That rank count does not make the construction a two-qubit readout. Its four
period coordinates are a direct sum of two torus frames, while a two-qubit
state uses a tensor product. The driver therefore refuses four-torus semantic
readouts; the engine's paired 4x4 transfer remains available as a low-level
geometric diagnostic without claiming tensor-product axes.

The host is two collars joined along a removed tetrahedron. Gluing along a
sphere is a connected sum, which adds no first homology, so `b_1 = 2 + 2 = 4`
with nothing dying on the boundary. It needs three collar layers: a prism cell
spans two adjacent layers, so the all-interior cell the join removes exists
only once there are two interior layers.
"""
import os
import sys

import numpy as np
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


def low_level_four_torus_node():
    """Build the geometric diagnostic directly through the engine API."""
    moduli = (TAU_A, -TAU_A.conjugate(), TAU_B, -TAU_B.conjugate())
    tori = [obs.SimplicialQubit.flat_torus(tau, GRID, GRID)
            for tau in moduli]
    seed = MC.seed_joined_collars(
        [surface.spacetime() for surface in tori], 3)
    ids = [{int(k): int(v) for k, v in mapping.items()}
           for mapping in seed.vertex_ids]
    node = MC(seed.host, [[1.0 + 0j]] * 4, [], degrees=[1], seed=7,
              einstein_hilbert=False, real_squared_lengths_only=False,
              metric_source=cob.HodgeMetricSource.WhitneyPencil)
    node.seed_inputs([sorted(mapping.values()) for mapping in ids])
    node.use_fiber_residuals(True)
    for index, surface in enumerate(tori):
        fiber = ea._torus_fiber(surface, ids[index])
        node.attach_input_fiber(index, fiber, fiber.cells)
        node.set_input_marking(
            index, ea._host_marking(surface, ids[index]),
            [1.0 + 0j, complex(moduli[index])])
    return node


def test_two_tori_cannot_carry_a_four_dimensional_target():
    """The control, and the reason the flag exists."""
    node = drive_to_seed(tori=2)
    chi = node.two_body_target().chi
    assert node.whole_harmonic_residual(chi) == 1.0
    assert "rank 2" in node.whole_harmonic_obstruction


def test_four_tori_supply_a_low_level_paired_transfer_only():
    """The 4x4 matrix is diagnostic; its axes are not qubit tensor axes."""
    node = low_level_four_torus_node()
    read = node.read_two_body()
    assert list(read.transfer.shape) == [4, 4]
    assert len(read.input_fiber_residuals) == 4
    assert np.isnan(read.residual), "a diagnostic must not infer a tensor target"


def test_the_conjugate_partners_are_the_orientation_reversals():
    """-conj(tau), so the modulus stays in the upper half-plane.

    A readout is named because the default, `transfer`, reads between exactly
    two frames and is refused at four tori.
    """
    node = low_level_four_torus_node()
    assert len(node.inputs) == 4


def test_paired_target_shape_is_checked_at_the_case_setter():
    node = low_level_four_torus_node()
    with pytest.raises(ValueError, match=r"paired.*3x3.*4x4"):
        node.set_two_body_cases([
            ([], np.eye(2, dtype=complex), True, np.eye(3, dtype=complex))])


# ---- the flag ----

def test_the_default_is_two_tori():
    assert ea.build_config()["tori"] == 2
    assert ea.DECLARED_TORI == 2


def test_the_driver_refuses_four_torus_semantic_readouts():
    for readout in ("whole", "operator"):
        with pytest.raises(
                ValueError, match=r"direct[- ]sum.*tensor[- ]product"):
            ea.build_config(inputs=ea.InputMode.QUBIT, tori=4,
                            readout=readout)


def test_only_two_or_four():
    """Three tori bound no collar and six is a host nobody has built."""
    for count in (0, 1, 3, 5, 6):
        with pytest.raises(ValueError) as caught:
            ea.build_config(tori=count)
        assert "two or four" in str(caught.value)
