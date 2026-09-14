"""The `verify` predicates on the cases the review of PR #1098 showed them
wrong on (issue #1099): R5's orientation sign is declared data, not
`-det M`; J2 measures the exactness residual, not the projected length;
U1 reports an isometric initial configuration instead of rejecting it.
"""
import os
import sys

import numpy as np

import tessera as T  # noqa: F401  (the driver imports the package)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))
import qubit_animation as qa  # noqa: E402

TOL = 1e-12
SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])


def test_r5_accepts_the_direct_sum_of_two_swaps():
    """`P_A = I_4`, `P_B = diag(S, S)`: each block reverses the form, the
    combined determinant is +1, and the graph is isotropic for the declared
    form with the swap's B-side sign. The old `-det M` rule rejected it."""
    periods_a = np.eye(4, dtype=complex)
    periods_b = np.block([[SWAP, np.zeros((2, 2))], [np.zeros((2, 2)), SWAP]]).astype(complex)
    row = qa._lagrangian_check("whole", periods_a, periods_b, "swap", TOL)
    assert row["pass"], row
    assert row["measured"]["declared_form_norm"] <= TOL
    assert row["measured"]["flipped_control_norm"] > 1.0
    assert row["measured"]["b_side_sign"] == +1
    # The sign is declared, not searched: told the relabelling was the
    # identity, the same periods fail, because the form is then the wrong one.
    assert not qa._lagrangian_check("whole", periods_a, periods_b, "none", TOL)["pass"]


def test_r5_product_collar_and_swap_collar_signs():
    identity = np.eye(2, dtype=complex)
    assert qa._lagrangian_check("A->B", identity, identity, "none", TOL)["pass"]
    assert qa._lagrangian_check("A->B", identity, SWAP.astype(complex), "swap", TOL)["pass"]
    assert not qa._lagrangian_check("A->B", identity, SWAP.astype(complex), "none", TOL)["pass"]


def test_j2_residual_rejects_a_near_projection():
    """`delta = (1, 1e-5)` against `span((1, 0))`: the projected-length ratio
    is within 5e-11 of one, the exactness residual is 1e-5."""
    delta = np.array([[1.0], [1e-5]], dtype=complex)
    coboundary = np.array([[1.0], [0.0]], dtype=complex)
    row = qa._exactness_check(delta, delta, coboundary, TOL)
    assert not row["pass"], row
    assert abs(row["measured"]["exactness_residual"] - 1e-5) < 1e-9
    fitted = np.linalg.lstsq(coboundary, delta, rcond=None)[0]
    ratio = np.linalg.norm(coboundary @ fitted) / np.linalg.norm(delta)
    assert 1.0 - ratio < 1e-9   # what the old predicate would have accepted


def test_j2_exact_motion_passes_and_no_motion_is_named():
    coboundary = np.array([[1.0], [0.0]], dtype=complex)
    reference = np.array([[1.0], [0.0]], dtype=complex)
    assert qa._exactness_check(np.array([[0.5], [0.0]], dtype=complex),
                               reference, coboundary, TOL)["pass"]
    still = qa._exactness_check(np.zeros((2, 1), dtype=complex), reference, coboundary, TOL)
    assert not still["pass"]
    assert still["measured"]["note"] == "the representative did not move"


def test_u1_reports_an_isometric_configuration():
    """Identical square tori on the product collar: `M = I` carries `G_A` to
    `G_B` exactly, so the initial defect is at rounding. Every check passes
    and U1 says so, rather than failing for the defect being small."""
    args = qa.build_parser().parse_args(["verify", "--tau-a", "1j", "--tau-b", "1j"])
    record = qa.verify(qa._verify_config(args))
    assert record["all_pass"], [row for row in record["checks"] if not row["pass"]]
    u1 = next(row for row in record["checks"] if row["id"] == "U1:A->B")
    assert u1["measured"]["isometric_before"]
    assert u1["measured"]["after"] > 1e-3
