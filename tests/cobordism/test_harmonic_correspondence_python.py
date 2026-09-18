# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Whole-kernel identification, independent witnesses and geometric controls."""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tessera.drivers import harmonic as hc


def graph_fixture(operator):
    """Analytical fixture only, never a geometric synthesis result."""
    operator = np.asarray(operator, dtype=complex)
    rows, columns = operator.shape
    constraint = np.hstack((-operator, np.eye(rows)))
    identity = np.eye(rows + columns)
    return constraint.conj().T @ constraint, identity[:columns], identity[columns:]


@pytest.mark.parametrize("operator", [
    np.eye(2), np.array([[0, 1j], [1, 0]]),
    np.array([[1 + 0.3j, 2], [-0.4j, 0.2]]),
    np.array([[1], [2j], [3]]), np.zeros((2, 2)),
])
def test_exact_operator_and_independent_whole_witnesses(operator):
    pencil, rin, rout = graph_fixture(operator)
    read = hc.recover_whole_relation(pencil, rin, rout)
    assert read["identifiable"], read["obstruction"]
    np.testing.assert_allclose(read["operator"], operator, atol=1e-12)
    witness = np.asarray(read["basis_whole_cochains"])
    np.testing.assert_allclose(pencil @ witness, 0, atol=1e-12)
    np.testing.assert_allclose(rin @ witness, np.eye(rin.shape[0]), atol=1e-12)
    assert max(read["held_out"]["errors"].values()) < 1e-12


def test_kernel_rebasing_preserves_the_operator():
    operator = np.array([[1j, 2], [0.3, -1j]])
    pencil, rin, rout = graph_fixture(operator)
    kernel = np.vstack((np.eye(2), operator))
    for change in (np.eye(2), np.array([[2j, 1], [-0.5, 3]])):
        read = hc.recover_whole_relation(pencil, rin, rout, kernel=kernel @ change)
        assert read["identifiable"]
        np.testing.assert_allclose(read["operator"], operator, atol=1e-12)
    with pytest.raises(ValueError, match="does not span"):
        hc.recover_whole_relation(pencil, rin, rout, kernel=np.eye(4)[:, :2])


def test_hidden_modes_require_output_uniqueness():
    pencil = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=complex)
    rin = np.array([[1, 0, 0]])
    unique = hc.recover_whole_relation(pencil, rin, [[0, 1, 0]])
    assert unique["identifiable"] and unique["hidden_mode_count"] == 1
    ambiguous = hc.recover_whole_relation(pencil, rin, [[0, 1, 1]])
    assert not ambiguous["identifiable"]
    assert "input-invisible" in ambiguous["obstruction"]
    assert "operator" not in ambiguous
    tiny = hc.recover_whole_relation(pencil, rin, [[0, 1e-12, 1e-12]])
    assert not tiny["identifiable"]


def test_port_coordinate_covariance():
    operator = np.array([[1j, 2], [0.3, -1j]])
    pencil, rin, rout = graph_fixture(operator)
    incoming = np.array([[2, 1j], [0, 0.5]])
    outgoing = np.array([[1, -1], [0.3j, 2]])
    read = hc.recover_whole_relation(pencil, np.linalg.solve(incoming, rin),
                                     np.linalg.solve(outgoing, rout))
    assert read["identifiable"]
    np.testing.assert_allclose(read["operator"], np.linalg.solve(outgoing, operator @ incoming), atol=1e-12)


def test_missing_inputs_and_empty_kernel_are_obstructions():
    pencil, rin, rout = graph_fixture(np.eye(2))
    rin[1] = rin[0]
    read = hc.recover_whole_relation(pencil, rin, rout)
    assert not read["identifiable"] and "not onto" in read["obstruction"]
    read = hc.recover_whole_relation(np.eye(4), rin, rout)
    assert not read["identifiable"] and "no zero mode" in read["obstruction"]


def test_zero_pencil_and_invalid_data():
    read = hc.recover_whole_relation(np.zeros((2, 2)), np.eye(2), np.eye(2))
    assert read["identifiable"]
    for tolerance in (0, -1, np.nan, 1):
        with pytest.raises(ValueError, match="tolerance"):
            hc.recover_whole_relation(np.eye(2), np.eye(2), np.eye(2), tolerance=tolerance)
    with pytest.raises(ValueError, match="finite matrix"):
        hc.recover_whole_relation([[np.nan]], [[1]], [[1]])


def test_choi_conjugation_marginals_and_entropy():
    operator = np.array([[0, 1j], [1, 0]])
    read = hc.operator_quantum_data(operator)
    assert read["coordinate_unitary"]
    choi = read["choi"]
    vector = np.asarray(choi["normalized_vector"]) * choi["frobenius_norm"]
    x = np.array([0.2 + 0.4j, -0.7j])
    np.testing.assert_allclose(np.kron(np.eye(2), x[None, :]) @ vector, operator @ x)
    np.testing.assert_allclose(choi["input_marginal"], np.eye(2) / 2, atol=1e-12)
    assert choi["input_reference_entropy_nats"] == pytest.approx(np.log(2))
    assert not hc.operator_quantum_data(2 * operator)["coordinate_unitary"]
    zero = hc.operator_quantum_data(np.zeros((2, 2)))
    assert "zero operator" in zero["choi"]["obstruction"]
    assert hc.operator_quantum_data(np.diag([1, 0]))["choi"]["input_reference_entropy_nats"] == 0


def test_composition_by_internal_elimination():
    t1 = np.array([[1j, 0.3], [0.4, 1]])
    t2 = np.array([[0.5, -1j], [0.2j, 2]])
    eye, zero = np.eye(2), np.zeros((2, 2))
    constraint = np.block([[-t1, eye, zero], [zero, -t2, eye]])
    pencil = constraint.conj().T @ constraint
    retained, internal = [0, 1, 4, 5], [2, 3]
    schur = pencil[np.ix_(retained, retained)] - pencil[np.ix_(retained, internal)] @ np.linalg.solve(
        pencil[np.ix_(internal, internal)], pencil[np.ix_(internal, retained)])
    product = np.hstack((-t2 @ t1, eye))
    expected = product.conj().T @ np.linalg.solve(eye + t2 @ t2.conj().T, product)
    np.testing.assert_allclose(schur, expected, atol=1e-12)
    read = hc.recover_whole_relation(schur, np.eye(4)[:2], np.eye(4)[2:])
    np.testing.assert_allclose(read["operator"], t2 @ t1, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_choi_ray_survives_extreme_nonzero_operator_scale(scale):
    data = hc.operator_quantum_data(scale * np.eye(2))
    vector = np.asarray(data["choi"]["normalized_vector"])
    np.testing.assert_allclose(vector, np.eye(2).reshape(-1) / np.sqrt(2))
    assert data["choi"]["input_reference_entropy_nats"] == pytest.approx(np.log(2))
    assert not data["coordinate_unitary"]


@pytest.fixture
def geometry():
    from tessera.drivers import qubit as qa
    return qa.build_qubit_node(qa.build_config(steps=0, grid=3, regge=False))


def test_native_target_independence_and_tensor_obstruction(geometry):
    node, _ = geometry
    assert tuple(inspect.signature(hc.measure_geometry).parameters) == ("node",)
    before = hc.measure_geometry(node)
    node.set_two_body_target(np.array([[1j, 0], [0, -1j]]), True)
    after = hc.measure_geometry(node)
    for name in ("periods", "gram"):
        read = before["readouts"][name]
        assert read["identifiable"], read["obstruction"]
        assert read["kernel_dimension"] == 2
        assert max(read["held_out"]["errors"].values()) < 1e-10
        np.testing.assert_allclose(read["operator"], after["readouts"][name]["operator"], atol=1e-12)
    np.testing.assert_allclose(before["readouts"]["periods"]["operator"], np.eye(2), atol=1e-10)
    assert not before["readouts"]["gram"]["quantum"]["coordinate_unitary"]
    assert not before["requested_gate"]["certified"]
    assert before["requested_gate"]["required_input_dimension"] == 4


def test_scalar_bulk_geometry_changes_gram_not_period_transport(geometry):
    node, _ = geometry
    initial = hc.measure_geometry(node)
    random = np.random.default_rng(71)
    boundary = [set(block.vertices) for block in node.inputs]
    for edge in node.spacetime().getEdgeList().toVector():
        vertices = {edge.getSource().getId(), edge.getTarget().getId()}
        if any(vertices <= block for block in boundary):
            continue
        edge.setLength(edge.getLength() * (1 + 0.03 * random.normal()))
    changed = hc.measure_geometry(node)
    for name in ("periods", "gram"):
        assert changed["readouts"][name]["identifiable"]
    np.testing.assert_allclose(changed["readouts"]["periods"]["operator"],
                               initial["readouts"]["periods"]["operator"], atol=1e-9)
    assert np.linalg.norm(np.asarray(changed["readouts"]["gram"]["operator"]) -
                          initial["readouts"]["gram"]["operator"]) > 1e-4


def test_animation_record_and_render(tmp_path):
    from tessera.drivers import qubit as qa
    result = qa.drive(qa.build_config(steps=0, grid=3, regge=False))
    record = json.loads(json.dumps(result.frames[0].to_json(), allow_nan=False))
    assert not record["correspondence"]["requested_gate"]["certified"]
    assert "input_reference_entropy_nats" in record["correspondence"]["readouts"]["periods"]["quantum"]["choi"]
    path = tmp_path / "correspondence.png"
    qa.render(result.frames, str(path))
    assert path.stat().st_size > 1000


def test_native_scalar_phase_transport(geometry):
    node, _ = geometry
    for edge in node.spacetime().getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        edge.setPhase(0.013 * (v - u))
    read = hc.measure_geometry(node)["readouts"]["periods"]
    assert read["identifiable"]
    bases = [node.input_marking(i).base_vertex for i in (0, 1)]
    expected = np.exp(-1j * 0.013 * (bases[1] - bases[0])) * np.eye(2)
    np.testing.assert_allclose(read["operator"], expected, atol=1e-10)
    assert read["quantum"]["coordinate_unitary"]


@pytest.mark.parametrize("fixed_bases", [True, False])
@pytest.mark.parametrize("complex_lengths", [False, True])
def test_native_gram_gauge_covariance(geometry, fixed_bases, complex_lengths):
    """A pure gauge changes only declared port coordinates, not the operator's
    singular values or Choi entropy. Fixing both basepoints fixes those
    coordinates as well. Exercise both real and complex Whitney metrics.
    """
    node, _ = geometry
    edges = node.spacetime().getEdgeList().toVector()
    if complex_lengths:
        random = np.random.default_rng(1105)
        for edge in edges:
            edge.setLength(edge.getLength() * (1 + 0.02j * random.normal()))
    before = hc.measure_geometry(node)
    bases = [int(node.input_marking(i).base_vertex) for i in (0, 1)]

    def phase(vertex):
        return 0.0 if fixed_bases and vertex in bases else 0.3 * np.sin(vertex)

    for edge in edges:
        u, v = int(edge.getSource().getId()), int(edge.getTarget().getId())
        edge.setPhase(phase(v) - phase(u))
    after = hc.measure_geometry(node)
    coordinate_change = np.exp(-1j * (phase(bases[1]) - phase(bases[0])))
    for name in ("periods", "gram"):
        initial, changed = before["readouts"][name], after["readouts"][name]
        assert initial["identifiable"], initial["obstruction"]
        assert changed["identifiable"], changed["obstruction"]
        np.testing.assert_allclose(changed["operator"],
                                   coordinate_change * np.asarray(initial["operator"]),
                                   atol=1e-10, rtol=0)
        initial_choi, changed_choi = initial["quantum"]["choi"], changed["quantum"]["choi"]
        for field in ("schmidt_coefficients", "input_marginal", "output_marginal",
                      "input_reference_entropy_nats"):
            np.testing.assert_allclose(changed_choi[field], initial_choi[field],
                                       atol=1e-10, rtol=0)
    assert after["readouts"]["gram"]["observation_convention"] == "dual_frame_whitney"


def test_gram_readout_reuses_native_gram_block(geometry, monkeypatch):
    from tessera import chainhodge
    node, _ = geometry
    gram_block = chainhodge.PencilSchur.gramBlock
    calls = []

    def recorded_gram_block(mass, left, right):
        calls.append((left.shape[1], right.shape[1]))
        return gram_block(mass, left, right)

    monkeypatch.setattr(chainhodge.PencilSchur, "gramBlock", staticmethod(recorded_gram_block))
    read = hc.measure_geometry(node)["readouts"]["gram"]
    assert read["identifiable"], read["obstruction"]
    assert calls == [(2, read["cochain_count"])] * 2


def test_a_gram_obstruction_does_not_hide_period_transport(geometry):
    from types import SimpleNamespace
    node, _ = geometry

    class ObstructedFrame:
        def __getattr__(self, name):
            return getattr(node, name)

        def derive_input_frame(self, index):
            return SimpleNamespace(obstruction="isotropic own-frame pairing")

    read = hc.measure_geometry(ObstructedFrame())
    assert read["readouts"]["periods"]["identifiable"]
    assert not read["readouts"]["gram"]["identifiable"]
    assert "isotropic" in read["readouts"]["gram"]["obstruction"]


def test_numerical_failure_is_a_named_animation_absence(monkeypatch):
    from tessera.drivers import qubit as qa

    def failed_read(node):
        raise np.linalg.LinAlgError("test singular-value solve failure")

    monkeypatch.setattr(hc, "measure_geometry", failed_read)
    result = qa.drive(qa.build_config(steps=0, grid=3, regge=False))
    read = result.frames[0].correspondence
    assert isinstance(read, qa.Absent)
    assert "singular-value solve failure" in read.reason
