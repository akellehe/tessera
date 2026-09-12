# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The driver refuses the paired-frame transfer as a qubit operator (#1048).

With conjugate pairs the boundary is four tori, two per state, and the transfer
between the two PAIRS is read between rank-4 frames. That makes it 4x4, but its
axes are direct sums of torus frames. A two-qubit gate's axes are tensor
products. Equal dimensions do not provide the missing identification, so the
driver refuses ``--readout operator`` rather than scoring an unrelated matrix.

The same mismatch rules out the four-torus whole-harmonic reading: its rank-4
period coordinates are also a direct sum, not the tensor coordinates of the
flattened two-qubit target. Four-torus paired reads remain low-level geometric
diagnostics only.

A reading that STRUCTURALLY cannot produce a number is refused when the config
is built, not scored as a constant 1.0. A constant term has no gradient, so the
flag would silently be a no-op while the run looked like it was optimizing.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import qubit_animation as qa  # noqa: E402

TAU_A, TAU_B, GRID = 0.3 + 1.1j, -0.2 + 0.8j, 3


# ---- the reading ----

def test_a_four_torus_case_does_not_invent_a_tensor_two_state_vector():
    """The tuple slot remains for the engine API, but has no driver value."""
    from tessera import cobordism as cob
    from tessera import observables as obs
    MC = cob.MultiCobordism

    def surface(tau):
        return obs.SimplicialQubit.flat_torus(complex(tau), GRID, GRID).spacetime()

    # A real seed: the case reads its boundary through the vertex maps.
    seed = MC.seed_joined_collars(
        [surface(TAU_A), surface(-TAU_A.conjugate()),
         surface(TAU_B), surface(-TAU_B.conjugate())], 3)
    ids = [{int(k): int(v) for k, v in mapping.items()}
           for mapping in seed.vertex_ids]
    boundary, chi, choi, tsv, coefficients = qa._two_body_case(
        (TAU_A, TAU_B), None, ids, GRID, "xx", {"coupling": 1.0, "time": 1.0})
    assert boundary
    assert choi is True
    assert tsv is None
    assert len(coefficients) == 8

    def state(tau):
        return np.asarray(
            obs.SimplicialQubit.flat_torus(complex(tau), GRID, GRID).state()).reshape(2)

    gate = np.asarray(qa.DECLARED_GATES["xx"], dtype=complex)
    psi = np.kron(state(TAU_A), state(TAU_B))
    assert np.allclose(chi, (gate @ psi).reshape(2, 2), atol=1e-13)


def test_the_reversed_torus_carries_Z_conj_psi():
    """Not conj(psi): reversing flips the sign of the second amplitude.

    The geometry does not undo that Z, so the backward wavefunction in the
    target is the reversed torus's OWN state.
    """
    from tessera import observables as obs

    def state(tau):
        return np.asarray(
            obs.SimplicialQubit.flat_torus(complex(tau), GRID, GRID).state()).reshape(2)

    for tau in (TAU_A, TAU_B, 1j, 0.5 + 0.9j):
        psi, partner = state(tau), state(-complex(tau).conjugate())
        assert not np.allclose(partner, psi.conj())
        assert np.allclose(partner, np.array([psi.conj()[0], -psi.conj()[1]]))


def test_equal_matrix_dimensions_do_not_make_the_operator_axes_compatible():
    """Four direct-sum axes are still not two tensor-product qubit axes."""
    with pytest.raises(ValueError,
                       match=r"direct[- ]sum.*tensor[- ]product"):
        qa.build_config(readout="operator", tori=4)


# ---- incompatible inputs are refused by name ----

@pytest.mark.parametrize("overrides,expected", [
    # At two tori the whole harmonic has rank 2, so it needs a 2-dimensional
    # target named rather than the 4-dimensional chi.
    (dict(readout="whole"), "needs --output-state"),
    (dict(readout="operator"), "unavailable"),
    (dict(readout="whole", tori=4), "direct-sum period frame"),
    (dict(readout="transfer", tori=4), "needs --tori 2"),
    (dict(readout="transfer,operator", tori=4), "needs --tori 2"),
    (dict(readout="bulk", layers=1), "unavailable"),
    (dict(readout="transfer", output_state="0.1+1.3j"), "includes whole"),
    (dict(readout=""), "no reading"),
    (dict(readout="nope"), "unknown readout"),
])
def test_refused(overrides, expected):
    with pytest.raises(ValueError) as caught:
        qa.build_config(**overrides)
    assert expected in str(caught.value), str(caught.value)


@pytest.mark.parametrize("overrides", [
    dict(readout="transfer"),
    dict(readout="whole", output_state="0.1+1.3j"),
])
def test_accepted(overrides):
    assert qa.build_config(**overrides)["readout"] == overrides["readout"]


def test_transfer_and_whole_compose_when_the_whole_target_is_declared():
    config = qa.build_config(readout="transfer,whole",
                             output_state="0.1+1.3j")
    assert config["readout"] == "transfer,whole"
    assert config["output_state"] == [0.1, 1.3]
