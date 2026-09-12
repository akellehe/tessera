# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Worked-solution tests for the emergence-animation process (#1070).

The fixture manifest is an independent oracle: computational-basis gate
actions, sphere homology, and the regular-simplex Regge deficit are stated as
solutions rather than copied from the driver's output.  These tests carry the
solutions through the gate, readout, CLI, and neutral-geometry paths.
"""

import itertools
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402


@pytest.fixture(scope="module")
def worked_solutions():
    path = Path(__file__).with_name("data") / "emergence_worked_solutions.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _solution_matrix(solutions, name):
    basis = solutions["basis_order"]
    columns = []
    for action in solutions["gate_actions"][name]:
        column = np.zeros(len(basis), dtype=complex)
        for ket, (real, imaginary) in action.items():
            column[basis.index(ket)] = complex(real, imaginary)
        columns.append(column)
    return np.column_stack(columns)


@pytest.fixture(params=sorted(ea.DECLARED_GATES))
def solved_gate(request, worked_solutions):
    assert set(worked_solutions["gate_actions"]) == set(ea.DECLARED_GATES)
    return request.param, _solution_matrix(worked_solutions, request.param)


def _basis_state(bit):
    state = np.zeros(2, dtype=complex)
    state[int(bit)] = 1.0
    return state


def _json_complex_matrix(value):
    return np.asarray([[complex(*entry) for entry in row] for row in value])


def test_every_operator_matches_its_solved_basis_action(solved_gate,
                                                        worked_solutions):
    name, expected = solved_gate
    actual_gate = np.asarray(ea.DECLARED_GATES[name], dtype=complex)
    np.testing.assert_allclose(actual_gate, expected, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(actual_gate.conj().T @ actual_gate,
                               np.eye(4), atol=1e-15, rtol=0.0)

    for column, bits in enumerate(worked_solutions["basis_order"]):
        image = ea.gate_image(name, _basis_state(bits[0]),
                              _basis_state(bits[1]))
        wanted = expected[:, column].reshape(2, 2)
        np.testing.assert_allclose(image["chi"], wanted,
                                   atol=1e-15, rtol=0.0)
        np.testing.assert_allclose(image["exact_amplitudes"], wanted,
                                   atol=1e-15, rtol=0.0)


@pytest.fixture(scope="module")
def sqrt_swap_cli_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sqrt-swap-run")
    run_path = directory / "run.json"
    geometry_path = directory / "geometry.json"
    result = ea.main([
        "run", "--inputs", "qubit", "--operator", "sqrt_swap",
        "--tau-a", "0.3+1.1j", "--tau-b=-0.2+0.8j", "--grid", "3",
        "--steps", "0", "--no-regge", "--json", str(run_path),
        "--geometry", str(geometry_path), "--out", "", "--quiet",
    ])
    assert result == 0
    with run_path.open(encoding="utf-8") as handle:
        run = json.load(handle)
    with geometry_path.open(encoding="utf-8") as handle:
        geometry = json.load(handle)
    return run, geometry


def test_a_solved_gate_reaches_both_cli_artifacts(sqrt_swap_cli_run,
                                                  worked_solutions):
    run, geometry = sqrt_swap_cli_run
    gate = _solution_matrix(worked_solutions, "sqrt_swap")
    tau_a, tau_b = 0.3 + 1.1j, -0.2 + 0.8j
    psi = np.array([1.0, tau_a]) / math.sqrt(1.0 + abs(tau_a) ** 2)
    phi = np.array([1.0, tau_b]) / math.sqrt(1.0 + abs(tau_b) ** 2)
    expected = (gate @ np.kron(psi, phi)).reshape(2, 2)

    assert run["config"]["operator"] == "sqrt_swap"
    assert run["terminator"] == ea.Terminator.STEPS
    assert len(run["frames"]) == 1
    algebra = run["inputs"]["algebra"]
    np.testing.assert_allclose(_json_complex_matrix(algebra["gate"]), gate,
                               atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(
        _json_complex_matrix(algebra["exact_amplitudes"]), expected,
        atol=1e-14, rtol=0.0)
    assert geometry["schema"] == 1
    assert geometry["source"] == run["source"]
    assert len(geometry["blocks"]) == 2
    for index, block in enumerate(geometry["blocks"]):
        assert block["tau_in"] == run["inputs"]["tau_in"][index]
        assert block["coefficients"] == run["inputs"][
            "coefficients_in"][index]


@pytest.fixture(scope="module")
def closed_s4():
    import tessera

    cells = [list(cell) for cell in itertools.combinations(range(6), 5)]
    spacetime = tessera.Spacetime.fromCells(4, cells, 1.0, 0.0)
    spacetime.materializeFacets()
    return spacetime


def test_closed_s4_neutral_reads_match_the_analytic_solution(
        closed_s4, worked_solutions):
    solved = worked_solutions["closed_s4"]
    counts = [0] * 5
    triangles = {}
    for simplex in closed_s4.getSimplices():
        vertices = tuple(sorted(int(v.getId())
                                for v in simplex.getVertices()))
        if 1 <= len(vertices) <= 5:
            counts[len(vertices) - 1] += 1
        if len(vertices) == 3:
            triangles[vertices] = simplex
    assert counts == solved["f_vector"]
    assert sum((-1) ** degree * count
               for degree, count in enumerate(counts)) == solved[
                   "euler_characteristic"]
    total_volume = sum(complex(cell.volume()).real
                       for cell in closed_s4.getTopSimplices())
    assert total_volume == pytest.approx(6.0 * math.sqrt(5.0) / 96.0,
                                         rel=1e-13)

    betti = ea.EmergenceFrame._read_betti(
        closed_s4, {"betti_degrees": list(range(5))})
    assert list(betti["numbers"].values()) == solved["betti"]

    deficit = 2.0 * math.pi - 3.0 * math.acos(0.25)
    assert len(triangles) == solved["hinges"]
    for triangle in triangles.values():
        assert complex(triangle.deficitAngle()).real == pytest.approx(deficit)
        assert complex(triangle.deficitAngle()).imag == pytest.approx(0.0)
    dual = ea.EmergenceFrame._read_dual_curvature(closed_s4)
    assert dual["hinges_with_curvature"] == solved["hinges"]
    assert len(dual["cells"]) == solved["top_cells"]
    for cell in dual["cells"]:
        faces = itertools.combinations(cell["vertices"], 3)
        expected = sum(deficit * abs(complex(triangles[tuple(face)].dualVolume()))
                       for face in faces)
        assert cell["spatial"] == pytest.approx(expected, rel=1e-13)
        assert cell["temporal"] == pytest.approx(0.0, abs=1e-15)


def test_closed_s4_geometry_record_keeps_the_solved_complex(
        closed_s4, worked_solutions):
    solved = worked_solutions["closed_s4"]
    source = {"head": "worked-solution", "branch": "fixture",
              "dirty": False}
    node = SimpleNamespace(spacetime=lambda: closed_s4)
    document = ea.geometry_document(node, source=source)
    assert document["source"] == source
    assert document["dimensions"] == 4
    assert len(document["cells"]) == solved["top_cells"]
    assert len(document["edges"]) == solved["f_vector"][1]
    assert len(document["vertex_times"]) == solved["vertices"]
    for _u, _v, real_squared, imaginary_squared in document["edges"]:
        assert real_squared == pytest.approx(1.0)
        assert imaginary_squared == pytest.approx(0.0)


@pytest.fixture(scope="module")
def composed_readout_node():
    config = ea.build_config(
        inputs="qubit", readout="transfer,whole",
        output_state="0.1+1.3j", operator="sqrt_swap",
        pin_boundary=True, input_weight=1.0, grid=3, steps=0,
        tau_a=0.3 + 1.1j, tau_b=-0.2 + 0.8j, regge=False)
    node, _inputs = ea.build_qubit_node(config)
    return node, config


def test_transfer_and_whole_are_additive_in_value_and_ascent(
        composed_readout_node):
    node, config = composed_readout_node
    modes = ea.MC.ReadoutMode
    readings = {
        "transfer": [modes.TRANSFER],
        "whole": [modes.WHOLE],
        "combined": [modes.TRANSFER, modes.WHOLE],
    }
    residuals = {}
    ascents = {}
    for name, selected in readings.items():
        node.set_readout_modes(selected)
        residuals[name] = node.two_body_residual_over_cases()
        ascents[name] = tuple(np.asarray(part)
                              for part in node.fiber_mode_ascent())
    assert residuals["combined"] == pytest.approx(
        residuals["transfer"] + residuals["whole"], rel=1e-13)

    edge_count = len(node.spacetime().getEdgeList().toVector())
    baseline = [np.zeros(edge_count, dtype=complex),
                np.zeros(edge_count, dtype=complex)]
    for index in range(2):
        for component, part in enumerate(node.own_state_residual_gradient(index)):
            values = np.asarray(part)
            if values.size:
                baseline[component] += config["input_weight"] * values
    for component in range(2):
        expected = ((ascents["transfer"][component] - baseline[component])
                    + (ascents["whole"][component] - baseline[component]))
        actual = ascents["combined"][component] - baseline[component]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
