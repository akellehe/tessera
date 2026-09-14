"""The `verify` predicates on the cases the review of PR #1098 showed them
wrong on (issue #1099): R5's orientation sign is declared data, not
`-det M`; J2 measures the exactness residual, not the projected length;
U1 reports an isometric initial configuration instead of rejecting it.
"""
import os
import sys

import numpy as np
import pytest

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


# ---- #1103: R5 on the subspace, and the harmonic certificates -------------

def test_r5_verdict_is_independent_of_basis_scale():
    """The same subspace and identity operator at every scale: rescaling the
    harmonic basis rescales |L^T Omega L| by |s|^2, which the old absolute
    check read as the control vanishing at s = 1e-7."""
    for scale in (1e-8, 1e-7, 1.0, 1e7, 1e8):
        periods = scale * np.eye(2, dtype=complex)
        row = qa._lagrangian_check("A->B", periods, periods, "none", TOL)
        assert row["pass"], (scale, row)
        assert row["measured"]["rank"] == 2


def test_r5_verdict_survives_a_common_complex_basis_change():
    rng = np.random.default_rng(3)
    periods_a = np.eye(4, dtype=complex)
    periods_b = np.block([[SWAP, np.zeros((2, 2))], [np.zeros((2, 2)), SWAP]]).astype(complex)
    for _ in range(5):
        change = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        change /= np.linalg.norm(change, 2)
        change += 0.5 * np.eye(4)   # well conditioned
        assert qa._lagrangian_check("whole", periods_a @ change, periods_b @ change, "swap", TOL)["pass"]
        assert not qa._lagrangian_check("whole", periods_a @ change, periods_b @ change, "none", TOL)["pass"]


def test_r5_rejects_a_genuinely_non_isotropic_subspace():
    """`M = diag(2, 1)` is not (anti-)symplectic: neither sign vanishes."""
    periods_a = np.eye(2, dtype=complex)
    periods_b = np.diag([2.0, 1.0]).astype(complex)
    row = qa._lagrangian_check("A->B", periods_a, periods_b, "none", TOL)
    assert not row["pass"]
    assert row["measured"]["declared_form_norm"] > 1e-3
    assert row["measured"]["flipped_control_norm"] > 1e-3


def test_r5_one_collar_of_a_four_torus_host_is_a_two_dimensional_subspace():
    """Four cycles against four harmonic columns, rank two: the pair's own
    restriction subspace is half its boundary b_1, not the column count."""
    periods_a = np.hstack([np.eye(2), np.zeros((2, 2))]).astype(complex)
    periods_b = np.hstack([np.eye(2), np.zeros((2, 2))]).astype(complex)
    row = qa._lagrangian_check("A->A*", periods_a, periods_b, "none", TOL)
    assert row["pass"], row
    assert row["measured"]["rank"] == 2
    # and a subspace too large to be Lagrangian is refused, not scored
    too_big = np.eye(4, dtype=complex)
    refused = qa._lagrangian_check("A->A*", too_big[:2], too_big[2:], "none", TOL)
    assert refused["pass"] is False and "rank 4 against the Lagrangian dimension 2" in refused["measured"]["refusal"]


def test_r5_refuses_a_rank_deficient_stack_by_name():
    periods_a = np.diag([1.0, 0.0]).astype(complex)
    periods_b = np.diag([1.0, 0.0]).astype(complex)
    row = qa._lagrangian_check("A->B", periods_a, periods_b, "none", TOL)
    assert not row["pass"]
    assert "rank 1 against the Lagrangian dimension 2" in row["measured"]["refusal"]
    assert "declared_form_norm" not in row["measured"]


_HOST = {}


def _host():
    """The seeded collar, its markings, its restriction read and its pencil."""
    if not _HOST:
        args = qa.build_parser().parse_args(["verify"])
        tori, _, seed, ids, markings = qa.seed_qubit_host(qa._verify_config(args))
        read = qa.MC.restriction(seed.host, markings)
        assert read.obstruction == "", read.obstruction
        _HOST["read"] = read
        _HOST["op"] = qa.cob.PencilLayer.assemble([seed.host]).op
        _HOST["markings"] = markings
        _HOST["host"] = seed.host
    return _HOST


class _FakeRead:
    """A restriction read with some fields replaced, for the certificate checks."""

    def __init__(self, real, **override):
        self.betti = list(real.betti)
        self.harmonic_rank = int(real.harmonic_rank)
        self.images = np.asarray(real.images)
        self.frame = np.asarray(real.frame)
        self.certificate = real.certificate
        for key, value in override.items():
            setattr(self, key, value)


def _rows(rows):
    return {row["id"]: row for row in rows}


def test_harmonic_certificates_pass_on_the_native_band():
    fixture = _host()
    rows, fields = qa._harmonic_certificates("seed", fixture["read"], fixture["op"], TOL)
    by_id = _rows(rows)
    assert by_id["H1:seed"]["pass"], by_id["H1:seed"]
    assert by_id["H1:seed"]["measured"]["closed_residual"] <= TOL
    assert by_id["H1:seed"]["measured"]["coclosed_residual"] <= TOL
    assert by_id["H2:seed"]["pass"], by_id["H2:seed"]
    assert by_id["H2:seed"]["measured"]["kernel_nullity"] == 2
    assert by_id["H3:seed"]["pass"], by_id["H3:seed"]
    assert fields["idempotency"] <= TOL


def test_an_incorrect_representative_with_unchanged_periods_is_rejected():
    """Perturb one column on an edge no marking walks: every transported
    period is unchanged, and the column is no longer closed."""
    fixture = _host()
    real = fixture["read"]
    marked = {(min(u, v), max(u, v)) for marking in fixture["markings"]
              for cycle in marking for u, v in cycle}
    interior = None
    for edge in fixture["host"].getEdgeList().toVector():
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        if (min(u, v), max(u, v)) not in marked:
            interior = (min(u, v), max(u, v))
            break
    assert interior is not None
    assembled = qa.cob.PencilLayer.assemble([fixture["host"]])
    row_index = assembled.cell_index(1, list(interior))
    images = np.asarray(real.images).copy()
    images[row_index, 0] += 1e-3
    fake = _FakeRead(real, images=images)
    # periods over every marking unchanged: the walks never touch that edge
    connection = assembled.op.connection()
    for index, marking in enumerate(fixture["markings"]):
        for c, cycle in enumerate(marking):
            walk = [(int(u), int(v)) for u, v in cycle]
            rotations = [walk[k:] + walk[:k] for k in range(len(walk))]
            reported = np.asarray(real.periods[index])[c, 0]
            assert any(abs(connection.transportedPeriod(images[:, 0], rot) - reported) < 1e-12
                       for rot in rotations)
    rows = _rows(qa._harmonic_certificates("seed", fake, fixture["op"], TOL)[0])
    assert not rows["H1:seed"]["pass"]
    assert rows["H1:seed"]["measured"]["closed_residual"] > 1e-6


def test_a_contaminated_band_is_rejected_by_both_equations_and_the_kernel():
    fixture = _host()
    real = fixture["read"]
    rng = np.random.default_rng(5)
    images = np.asarray(real.images).copy()
    images[:, 1] = rng.normal(size=images.shape[0]) + 1j * rng.normal(size=images.shape[0])
    rows = _rows(qa._harmonic_certificates("seed", _FakeRead(real, images=images), fixture["op"], TOL)[0])
    assert not rows["H1:seed"]["pass"]
    assert not rows["H2:seed"]["pass"]
    assert rows["H2:seed"]["measured"]["span_residual_band_in_kernel"] > 1e-3


def test_a_kernel_dimension_mismatch_is_named_as_the_geometry_failing_the_condition():
    fixture = _host()
    fake = _FakeRead(fixture["read"], harmonic_rank=3)
    rows = _rows(qa._harmonic_certificates("seed", fake, fixture["op"], TOL)[0])
    assert not rows["H2:seed"]["pass"]
    assert "fails the condition dim ker L_1 = b_1" in rows["H2:seed"]["measured"]["failure"]


def test_verify_certifies_sentence_is_in_the_record():
    args = qa.build_parser().parse_args(["verify"])
    record = qa.verify(qa._verify_config(args))
    assert record["certifies"] == qa.VERIFY_CERTIFIES
    assert "NOT certified" in record["certifies"]
    ids = {row["id"] for row in record["checks"]}
    assert {"H1:seed", "H2:seed", "H3:seed", "H1:jittered", "H2:jittered", "H3:jittered"} <= ids
    assert record["all_pass"], [row for row in record["checks"] if not row["pass"]]


# ---- #1109: the dual-frame pairing behind U1 and J3 ------------------------

def _torus_operator():
    """The seeded collar's torus A on the host's live lengths: its operator,
    contour, marking walks, and the primal pieces for the comparison."""
    fixture = _host()
    tori, _, seed, ids, _ = qa.seed_qubit_host(qa._verify_config(qa.build_parser().parse_args(["verify"])))
    spacetime = tori[0].spacetime()
    qa._copy_lengths_to_torus(fixture["host"], spacetime, ids[0])
    marking = qa._host_marking(tori[0], {int(v): int(v) for pair in tori[0].edges() for v in pair})
    assembled = qa.cob.PencilLayer.assemble([spacetime])
    contour = qa.cob.PencilLayer.harmonic_contour(assembled, 1)
    walks = [[(int(u), int(v)) for u, v in cycle] for cycle in marking]
    return spacetime, assembled.op, contour, walks


def _primal_period_gram(op, contour, walks):
    band = op.band(1, contour)
    images = np.asarray(band.images)
    mass = np.asarray(op.dressed(1).toarray())
    periods = np.array([[op.connection().transportedPeriod(images[:, a], w)
                         for a in range(images.shape[1])] for w in walks])
    inverse = np.linalg.inv(periods)
    return inverse.T @ (images.T @ mass @ images) @ inverse


def test_covariant_gram_equals_the_primal_one_at_zero_phases():
    """The seeded hosts carry no phases: the recorded U1/J3 numbers must be
    reproduced exactly, and here the two formulas coincide."""
    spacetime, op, contour, walks = _torus_operator()
    assert sum(abs(complex(e.getPhase())) > 0 for e in spacetime.getEdgeList().toVector()) == 0
    covariant = qa._covariant_period_gram(op, contour, walks)
    primal = _primal_period_gram(op, contour, walks)
    assert np.linalg.norm(covariant - primal) / np.linalg.norm(primal) <= TOL


@pytest.mark.parametrize("modulus_one", [True, False])
def test_covariant_gram_is_gauge_invariant_and_the_primal_one_is_not(modulus_one):
    """A random vertex gauge, U(1) or C*: the dual-frame pairing in marking
    coordinates is unchanged; the primal-primal one moves by O(1)."""
    spacetime, op, contour, walks = _torus_operator()
    rng = np.random.default_rng(3)
    vertices = [int(v.getId()) for v in spacetime.getVertexList().toVector()]
    gauge = {v: complex((1.0 if modulus_one else rng.uniform(0.7, 1.4)) * np.exp(1j * rng.uniform(0, 2 * np.pi)))
             for v in vertices}
    gauged = op.gauged(gauge)
    covariant, covariant_g = (qa._covariant_period_gram(o, contour, walks) for o in (op, gauged))
    primal, primal_g = (_primal_period_gram(o, contour, walks) for o in (op, gauged))
    assert np.linalg.norm(covariant_g - covariant) / np.linalg.norm(covariant) <= TOL
    assert np.linalg.norm(primal_g - primal) / np.linalg.norm(primal) > 0.1


def test_recorded_u1_and_j3_values_are_reproduced():
    """The values recorded for the default host before the move (PR #1104):
    U1 0.5708 -> 0.7564 under the jitter, J3 relative move 0.4488."""
    args = qa.build_parser().parse_args(["verify"])
    record = qa.verify(qa._verify_config(args))
    u1 = next(r for r in record["checks"] if r["id"] == "U1:A->B")["measured"]
    j3 = next(r for r in record["checks"] if r["id"] == "J3")["measured"]
    assert u1["pairing_convention"] == "dual_frame_whitney"
    assert abs(u1["before"] - 0.5708) < 5e-4 and abs(u1["after"] - 0.7564) < 5e-4
    assert abs(j3["relative_move"] - 0.4488) < 5e-4
    assert record["all_pass"], [row for row in record["checks"] if not row["pass"]]
