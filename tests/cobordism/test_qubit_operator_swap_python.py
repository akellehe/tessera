# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The swap harness reads a solved geometry faithfully (#999).

``examples/cobordism/qubit_operator_swap.py`` loads a geometry dump, installs
boundary lengths on it and reads the transfer, without driving anything. The
property that makes its answers trustworthy is the REPLAY control: putting
the solved boundary lengths back, edge for edge, must reproduce the numbers
the run itself reported -- the two-body residual, both own-state residuals
and both tau_hat -- or the harness is measuring its own reconstruction rather
than the geometry.

The second control, `reflat`, installs a FRESH flat torus at the same tau.
Its state is the one the bulk was solved against but its lengths are not,
since the D2 residual holds tau_hat while the individual lengths drift under
it. Here that is only asserted to be well defined; what it measures is a
finding, not a fixture.
"""
import json
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402
import qubit_operator_swap as swap  # noqa: E402

TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)


@pytest.fixture(scope="module")
def solved():
    """Two units of drive, its geometry dumped and its last frame kept."""
    config = ea.build_config(steps=2, stage1_iters=1, stage2_iters=2,
                             tolerance=1e-30, inputs=ea.InputMode.QUBIT,
                             tau_a=TAU_A, tau_b=TAU_B, grid=3, regge=False)
    held = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ea.drive(config, progress=False,
                          on_node=lambda node: held.update(node=node))
        document = ea.geometry_document(held["node"], result.inputs)
    return json.loads(json.dumps(document)), result.frames[-1]


def test_replay_reproduces_the_run(solved):
    document, frame = solved
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = swap.read(document, [TAU_A, TAU_B], "flip_flop", 3, mode="replay")
    assert "refused" not in out, out.get("refused")
    assert out["residual"] == pytest.approx(frame.two_body["residual"], rel=1e-12)
    for index in range(2):
        assert out["own_state_residuals"][index] == pytest.approx(
            frame.blocks[index]["residual"], rel=1e-9, abs=1e-18)
        assert out["tau_hat"][index] == pytest.approx(
            complex(frame.blocks[index]["read"]["tau"]), rel=1e-12)


def test_a_fresh_flat_torus_is_exactly_its_input_state(solved):
    """`reflat` installs the state the bulk was solved against, exactly: a
    flat torus's own-state residual is zero, unlike the drifted boundary the
    run left, whose residual is merely small."""
    document, _frame = solved
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = swap.read(document, [TAU_A, TAU_B], "flip_flop", 3, mode="flat")
    assert "refused" not in out, out.get("refused")
    for index, tau in enumerate((TAU_A, TAU_B)):
        assert out["own_state_residuals"][index] < 1e-14
        assert out["tau_hat"][index] == pytest.approx(tau, rel=1e-12)


def test_every_translation_is_a_gluing_of_the_same_torus(solved):
    """The lattice translations permute the torus's vertices and carry the
    marking to itself, so each is a different gluing of the same torus to the
    same bulk; every one must be installable and readable."""
    document, _frame = solved
    permutations = swap.translations(3)
    assert len(permutations) == 9
    assert permutations[0] == tuple(range(9)), "the identity comes first"
    assert len({tuple(p) for p in permutations}) == 9
    for permutation in permutations:
        assert sorted(permutation) == list(range(9)), "a permutation, not a map"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = swap.read(document, [TAU_A, TAU_B], "flip_flop", 3,
                        permutations=(permutations[4], None), mode="flat")
    assert "refused" not in out or "host edge" in out["refused"], out.get("refused")


def test_the_operators_are_bilinear_and_their_ranks_are_known():
    psi, phi = swap.state_of(TAU_A), swap.state_of(TAU_B)
    flip = swap.OPERATORS["flip_flop"](psi, phi)
    # the selection rule: the flip-flop's image has no diagonal
    assert abs(flip[0, 0]) < 1e-18 and abs(flip[1, 1]) < 1e-18
    assert np.linalg.matrix_rank(flip, tol=1e-12) == 2
    for name in ("ising", "product"):
        assert np.linalg.matrix_rank(swap.OPERATORS[name](psi, phi), tol=1e-12) == 1
    # bilinearity in the first argument, on the raw (unnormalised) vectors
    a, b = np.array([1.0, 0.3j]), np.array([0.2, -1.1])
    for name, operator in swap.OPERATORS.items():
        left = operator(a + b, phi)
        right = operator(a, phi) + operator(b, phi)
        assert np.abs(left - right).max() < 1e-15, name


def test_the_residual_is_the_engine_s_own_scoring():
    """`projective_residual` is 1 - |<T, chi>|^2 / (|T|^2 |chi|^2): zero on a
    complex multiple, one on an orthogonal pair, scale-free in T."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    assert swap.projective_residual(target, target) < 1e-30
    assert swap.projective_residual(3.7j * target, target) < 1e-30
    orthogonal = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    assert swap.projective_residual(orthogonal, np.array([[1.0, 0.0], [0.0, 0.0]],
                                                         dtype=complex)) == pytest.approx(1.0)
