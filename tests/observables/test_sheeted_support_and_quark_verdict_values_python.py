# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The sheet convention and the v16 quark verdict as the recursion driver
reaches them.

`baryon_poles.quark_conditions` (called by `evaluate_content` for every
content of every host cell `recursion.cell_reads` reads) builds the evidence
of the seven conditions from `SheetedSupport.certifyIsomorphism`,
`SheetAttachment.attachmentMatrix`, the spin read of `MonopoleSupport` and the
recursion record, and grades it with `QuarkConditions.evaluate`. The tests
hold each piece to whitepaper v17 §8 (the sheet convention, lines 355-357:
the free operator h (x) I_3, every band E-bar (x) C^3, colour commuting with
the base dynamics and symmetries) and §14 (lines 586-589: the colour singlet
as the determinant wedge Omega_p(c_A ^ c_B ^ c_C) of three rays), drive every
one of the seven conditions to each of its three statuses with evidence built
for it, and run the driver's own `quark_conditions` on constructed inputs.
"""

import numpy as np
import pytest

from tessera import observables as obs
from tessera.drivers import baryon_poles as bp

E = obs.QuarkConditionEvidence
Q = obs.QuarkConditions
Passed = obs.QuarkConditionStatus.Passed
Failed = obs.QuarkConditionStatus.Failed
NotEvaluable = obs.QuarkConditionStatus.NotEvaluable


# ------------------------------------------------ the sheeted support


def _base_operator():
    """The averaged edge operator of the unit-monopole tetrahedron (WP v17
    line 506) with a non-Hermitian perturbation, so that nothing below rests
    on symmetry of the base."""
    support = obs.MonopoleSupport.tetrahedron(1)
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), obs.MonopoleSupport.tetrahedralRotations()))
    return averaged + 0.1j * np.arange(36).reshape(6, 6) / 36.0


def test_the_free_operator_is_h_tensor_the_identity_on_the_sheets():
    """WP v17 line 357: on a three-sheeted support the free operator is
    h (x) I_3. With the mode index b * 3 + s it is exactly np.kron(h, I_3),
    and its spectrum is the base spectrum with each value three times."""
    h = _base_operator()
    support = obs.SheetedSupport(3, 6)
    lifted = np.asarray(support.freeOperator(h))
    assert np.max(np.abs(lifted - np.kron(h, np.eye(3)))) == 0.0
    assert support.modeIndex(4, 2) == 14
    base = np.sort_complex(np.linalg.eigvals(h))
    sheeted = np.sort_complex(np.linalg.eigvals(lifted))
    np.testing.assert_allclose(sheeted, np.repeat(base, 3), atol=1e-12)


def test_a_lifted_band_is_the_colour_spin_fibre():
    """WP v17 line 357: every Riesz band E-bar of the base becomes
    E-bar (x) C^3. The j = 1/2 doublet of the unit monopole (rank 2) lifts to
    a rank-6 fibre equal to np.kron(E-bar, I_3), invariant under the free
    operator of the averaged Laplacian."""
    support = obs.MonopoleSupport.tetrahedron(1)
    averaged = np.asarray(support.rotationAveragedEdgeOperator(
        support.edgeLaplacian(), obs.MonopoleSupport.tetrahedralRotations()))
    values, vectors = np.linalg.eigh(averaged)
    band = vectors[:, np.abs(values - 4.0) < 1e-9]
    assert band.shape == (6, 2)
    sheeting = obs.SheetedSupport(3, 6)
    fibre = np.asarray(sheeting.liftBand(band))
    assert fibre.shape == (18, 6)
    assert np.max(np.abs(fibre - np.kron(band, np.eye(3)))) == 0.0
    free = np.asarray(sheeting.freeOperator(averaged))
    assert np.max(np.abs(free @ fibre - 4.0 * fibre)) < 1e-12


def test_colour_commutes_with_the_base_dynamics_and_symmetries():
    """WP v17 line 357: I (x) g, g in GL(3, C), commutes with h (x) I and
    with D (x) I; the measured commutator of a non-unitary frame against the
    base operator and against every rotation's edge action is zero to
    rounding."""
    g = np.array([[1.0, 2.0, 0.5j], [0.0, 1.5, -1.0], [0.3, 0.0, 2.0]])
    sheeting = obs.SheetedSupport(3, 6)
    assert sheeting.sheetCommutatorResidual(_base_operator(), g) < 1e-13
    support = obs.MonopoleSupport.tetrahedron(1)
    for rotation in obs.MonopoleSupport.tetrahedralRotations():
        action = np.asarray(support.edgeRepresentation(rotation))
        assert sheeting.sheetCommutatorResidual(action, g) < 1e-13
        lifted = np.asarray(sheeting.baseSymmetryOperator(action))
        assert np.max(np.abs(lifted - np.kron(action, np.eye(3)))) == 0.0
    frame = np.asarray(sheeting.sheetFrameOperator(g))
    assert np.max(np.abs(frame - np.kron(np.eye(6), g))) == 0.0


def test_the_driver_sheet_to_sheet_attachment_is_the_identity():
    """The driver's colour transport (condition 5, `quark_conditions`)
    attaches sheet t to sheet t with weight 1: the attachment matrix is I_3,
    its determinant is exactly 1, it is sheet-diagonal and certified full
    rank."""
    read = obs.SheetAttachment.attachmentMatrix(
        3, [obs.ConnectingSimplex(t, t, 1.0) for t in range(3)])
    assert np.max(np.abs(np.asarray(read.matrix) - np.eye(3))) == 0.0
    assert read.determinant == 1.0
    assert read.sheet_diagonal
    assert read.simplex_count == 3
    assert read.min_singular_value == pytest.approx(1.0, abs=1e-15)
    assert read.certificate.holds()


def test_a_cross_sheet_attachment_accumulates_its_weights():
    """Two simplices on (0, 1) with weights 2 and 3 and one on each diagonal
    entry with weight 1: S = [[1, 5, 0], [0, 1, 0], [0, 0, 1]], det S = 1, not
    sheet-diagonal."""
    simplices = [obs.ConnectingSimplex(0, 1, 2.0),
                 obs.ConnectingSimplex(0, 1, 3.0)] + \
        [obs.ConnectingSimplex(t, t, 1.0) for t in range(3)]
    read = obs.SheetAttachment.attachmentMatrix(3, simplices)
    expected = np.array([[1, 5, 0], [0, 1, 0], [0, 0, 1]], dtype=complex)
    assert np.max(np.abs(np.asarray(read.matrix) - expected)) == 0.0
    assert read.determinant == pytest.approx(1.0, abs=1e-14)
    assert not read.sheet_diagonal


def test_a_missing_sheet_attachment_is_rank_deficient():
    """Sheets 0 and 1 attached, sheet 2 not: det S = 0 and the full-rank
    certificate does not hold (condition 5 would fail)."""
    read = obs.SheetAttachment.attachmentMatrix(
        3, [obs.ConnectingSimplex(t, t, 1.0) for t in range(2)])
    assert read.determinant == 0.0
    assert read.min_singular_value == 0.0
    assert not read.certificate.holds()


# ---------------------------------------------------- the colour singlet


IDENTITY = [np.eye(3, dtype=complex)] * 3


@pytest.mark.parametrize("colors, expected", [
    (([1, 0, 0], [0, 1, 0], [0, 0, 1]), 1.0),
    (([0, 1, 0], [1, 0, 0], [0, 0, 1]), -1.0),
    (([1, 2, 0], [0, 1, 3], [4, 0, 1]), 25.0),
    (([1, 2, 3], [2, 4, 6], [0, 0, 1]), 0.0),
])
def test_the_singlet_amplitude_is_the_determinant_wedge(colors, expected):
    """WP v17 §14 lines 586-589: S_ABC = Omega_p(c-hat_A ^ c-hat_B ^ c-hat_C).
    With the determinant line trivialized by Omega_p(e_1 ^ e_2 ^ e_3) = 1 and
    identity transports it is the determinant of the three rays as columns. The rays (1, 2, 0), (0, 1, 3),
    (4, 0, 1) give 1 (1 - 0) - 0 + 4 (6 - 0) = 25; a repeated direction gives
    exactly zero and the nonvanishing certificate fails."""
    vectors = [np.array(c, dtype=complex) for c in colors]
    read = obs.ColorSinglet.amplitude(1.0, IDENTITY, vectors, 1e-12)
    assert read.amplitude == pytest.approx(expected, abs=1e-13)
    assert read.nonvanishing == (expected != 0.0)
    assert read.certificate.holds() == (expected != 0.0)


def test_the_singlet_amplitude_carries_the_trivialization_and_transports():
    """With Omega_p(e_1 ^ e_2 ^ e_3) = 2 and the transports
    S_pA = diag(2, 1, 1), S_pB = I and S_pC the cyclic permutation
    e_1 -> e_2 -> e_3 -> e_1, the amplitude is 2 det[S_pA c_A, c_B, S_pC c_C]:
    the rays e_1, e_2, e_3 are carried to 2 e_1, e_2, e_1 and the amplitude is
    exactly zero, while the rays e_1, e_2, e_2 are carried to 2 e_1, e_2, e_3
    and the amplitude is 2 * 2 = 4."""
    cyclic = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
    transports = [np.diag([2.0, 1.0, 1.0]).astype(complex),
                  np.eye(3, dtype=complex), cyclic]
    e = np.eye(3, dtype=complex)
    read = obs.ColorSinglet.amplitude(2.0, transports, [e[0], e[1], e[2]],
                                      1e-12)
    assert read.amplitude == 0.0
    read = obs.ColorSinglet.amplitude(2.0, transports, [e[0], e[1], e[1]],
                                      1e-12)
    assert read.amplitude == pytest.approx(4.0, abs=1e-14)


def test_the_singlet_amplitude_refuses_other_sheet_counts():
    with pytest.raises(ValueError):
        obs.ColorSinglet.amplitude(1.0, [np.eye(2)] * 3,
                                   [np.ones(2)] * 3, 1e-12)


# -------------------------------------------- the seven quark conditions


REQUIRED = {
    1: ["persistent-support", "localized-projector-rank",
        "contour-separation", "successor-overlap", "multi-frame-lifetime",
        "external-leakage"],
    2: ["three-sheeted-support", "sheet-isomorphism", "protected-base-band",
        "base-band-sector", "fibre-lift"],
    3: ["anchor-profile-nonzero", "anchor-covariance", "anchor-transitions",
        "anchor-stable-across-frames"],
    4: ["odd-occupation-parity"],
    5: ["color-transport-full-rank", "base-transport-leakage",
        "transport-over-lifetime"],
    6: ["lineage-intersection"],
    7: ["refinement-stability", "relabeling-stability"],
}
NAMES = ["persistent-cluster", "color-spin-fiber", "anchor-atlas",
         "odd-occupation", "color-transport", "lineage", "fingerprint"]


def test_the_seven_conditions_and_their_required_evidence():
    """The v16 conditions of WP §10 as `QuarkConditions` names them, and the
    certificates each requires (QuarkConditions.h)."""
    assert list(Q.condition_names()) == NAMES
    for number, names in REQUIRED.items():
        assert list(Q.required_evidence(number)) == names
        assert Q.statement(number)


@pytest.mark.parametrize("number", range(1, 8))
def test_every_condition_passes_on_all_required_evidence_held(number):
    read = Q.evaluate_condition(number, [E(n, True, "held")
                                         for n in REQUIRED[number]])
    assert read.status == Passed
    assert read.name == NAMES[number - 1]
    assert list(read.missing) == [] and list(read.failing) == []


@pytest.mark.parametrize("number", range(1, 8))
def test_every_condition_fails_on_its_last_certificate_failing(number):
    """The last required certificate measured and not holding, the others
    held: Failed, with exactly that certificate named."""
    names = REQUIRED[number]
    evidence = [E(n, True, "held") for n in names[:-1]] + \
        [E(names[-1], False, "measured and failed")]
    read = Q.evaluate_condition(number, evidence)
    assert read.status == Failed
    assert list(read.failing) == [names[-1]]


@pytest.mark.parametrize("number", range(1, 8))
def test_every_condition_is_not_evaluable_on_an_unmeasured_certificate(
        number):
    """The first required certificate left unmeasured (held None) and the
    rest held: NotEvaluable, never Passed, with it named missing; supplying no
    evidence at all names every required certificate missing."""
    names = REQUIRED[number]
    evidence = [E(names[0], None, "not measured")] + \
        [E(n, True, "held") for n in names[1:]]
    read = Q.evaluate_condition(number, evidence)
    assert read.status == NotEvaluable
    assert list(read.missing) == [names[0]]
    empty = Q.evaluate_condition(number, [])
    assert empty.status == NotEvaluable
    assert list(empty.missing) == names


def test_a_failure_outranks_an_absence_and_an_optional_can_only_fail():
    """Condition 2 with sheet isomorphism failed and the fibre lift
    unmeasured is Failed, not NotEvaluable; an additional certificate outside
    the required list fails the condition when it fails and blocks nothing
    when it holds."""
    evidence = [E("three-sheeted-support", True, ""),
                E("sheet-isomorphism", False, "length residual 22.3"),
                E("protected-base-band", True, ""),
                E("base-band-sector", True, ""),
                E("fibre-lift", None, "")]
    read = Q.evaluate_condition(2, evidence)
    assert read.status == Failed
    assert list(read.failing) == ["sheet-isomorphism"]
    assert list(read.missing) == ["fibre-lift"]
    held = [E(n, True, "") for n in REQUIRED[6]]
    assert Q.evaluate_condition(
        6, held + [E("determinant-winding", True, "")]).status == Passed
    assert Q.evaluate_condition(
        6, held + [E("determinant-winding", False, "")]).status == Failed


def test_the_verdict_certifies_only_when_all_seven_pass():
    all_held = [[E(n, True, "") for n in REQUIRED[k]] for k in range(1, 8)]
    verdict = Q.evaluate(all_held)
    assert verdict.certified
    one_open = [list(x) for x in all_held]
    one_open[5] = []
    verdict = Q.evaluate(one_open)
    assert not verdict.certified
    assert list(verdict.not_evaluable) == ["lineage"]
    assert list(verdict.failed) == []


def test_the_condition_refusals_are_named():
    with pytest.raises(ValueError, match="numbered 1 to 7; got 8"):
        Q.statement(8)
    with pytest.raises(ValueError, match="numbered 1 to 7; got 0"):
        Q.evaluate_condition(0, [])
    with pytest.raises(ValueError, match="every evidence item names the "
                                         "certificate it reports"):
        Q.evaluate_condition(4, [E("", True, "")])


# ------------------------------------------ the driver's quark verdict


def _recursion_record(accepted=True, transport_norms=()):
    return {"partition": [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],
                          [12, 13, 14, 15, 16, 17]],
            "band_ranks": [6, 6, 6],
            "bands_accepted": [accepted] * 3,
            "projector_idempotency": [1e-16] * 3,
            "isolation_gaps": [17.4] * 3,
            "transport_norms": list(transport_norms)}


def _driver_verdict(spacetime, symmetry_residual=0.0, **recursion):
    alignment = bp.aligned_doublet_frame(bp.monopole_support(),
                                         bp.rotation_group())
    return bp.quark_conditions(spacetime, alignment,
                               _recursion_record(**recursion),
                               symmetry_residual, None)


def _statuses(verdict):
    return {c["name"]: c["status"] for c in verdict["conditions"]}


def test_the_driver_verdict_on_the_declared_host():
    """On the declared host (three exact sheets, unit monopole, WP §10
    condition 2) with accepted bands, no inter-component transport and a
    zero fibre-lift residual, a single-level read passes conditions 2 and 4,
    leaves 1, 3, 5, 6 and 7 not evaluable (they need several frames, a
    lineage, the anchor atlas or a refinement), and certifies nothing."""
    verdict = _driver_verdict(bp.build_host())
    assert _statuses(verdict) == {
        "persistent-cluster": "NotEvaluable",
        "color-spin-fiber": "Passed",
        "anchor-atlas": "NotEvaluable",
        "odd-occupation": "Passed",
        "color-transport": "NotEvaluable",
        "lineage": "NotEvaluable",
        "fingerprint": "NotEvaluable"}
    assert verdict["certified"] is False
    two = verdict["conditions"][1]
    isomorphism = [e for e in two["evidence"]
                   if e["name"] == "sheet-isomorphism"][0]
    assert isomorphism["detail"] == "length residual 0, connection residual 0"
    assert verdict["conditions"][0]["missing"] == ["successor-overlap",
                                                   "multi-frame-lifetime"]


def test_the_driver_verdict_fails_condition_two_on_a_separated_sheet():
    """Moving one squared length of sheet 2 by 1e-6 (above the driver's
    tolerance 1e-8) fails condition 2 on sheet isomorphism alone, with the
    residual in the evidence."""
    spacetime = bp.build_host()
    edge = [e for e in spacetime.getEdgeList().toVector()
            if int(e.getSource().getId()) // 4 == 2][0]
    edge.setLength(np.sqrt(complex(edge.getLength()) ** 2 + 1e-6))
    verdict = _driver_verdict(spacetime)
    two = verdict["conditions"][1]
    assert two["status"] == "Failed"
    assert two["failing"] == ["sheet-isomorphism"]


def test_the_driver_verdict_fails_the_fibre_lift_above_its_threshold():
    """A fibre-lift residual of 0.818 (the run's content (0, 2, 1) on the
    first cell) is above the driver's 1e-6 and fails condition 2 on the
    fibre lift; a residual of 2.49e-11 (content (0, 1, 2)) passes."""
    failing = _driver_verdict(bp.build_host(), symmetry_residual=0.818)
    assert failing["conditions"][1]["failing"] == ["fibre-lift"]
    passing = _driver_verdict(bp.build_host(), symmetry_residual=2.49e-11)
    assert passing["conditions"][1]["status"] == "Passed"


def test_the_driver_verdict_fails_conditions_one_and_five_on_leakage():
    """An inter-component transport norm of 0.146 (the run's tick-0
    transport between its two response vertices) is above the driver's
    tolerance 1e-8: condition 1 fails on external leakage and condition 5 on
    base-transport leakage; rejected bands fail condition 1 on persistent
    support, localized projector rank and not on contour separation."""
    verdict = _driver_verdict(bp.build_host(), transport_norms=[0.146])
    assert verdict["conditions"][0]["failing"] == ["external-leakage"]
    assert verdict["conditions"][4]["failing"] == ["base-transport-leakage"]
    rejected = _driver_verdict(bp.build_host(), accepted=False)
    assert rejected["conditions"][0]["failing"] == [
        "persistent-support", "localized-projector-rank"]
