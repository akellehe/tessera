# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The refusals and obstructions of the harmonic correspondence module
(`tessera.drivers.harmonic`) that its main tests
(``tests/cobordism/test_harmonic_correspondence_python.py``) do not reach, and
the package's statement about its driver modules.

Terms: a *pencil* is the whole zero-frequency operator whose kernel holds the
harmonic states; an *input readout* and an *output readout* are the matrices
that observe a whole cochain on the two ports; an *obstruction* is a named
reason the relation cannot be read, returned in place of a number.
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tessera.drivers import harmonic as hc


def _graph(operator):
    """The pencil and readouts whose unique harmonic relation is ``operator``:
    the kernel of C^dagger C with C = [-T, I] is the graph of T."""
    operator = np.asarray(operator, dtype=complex)
    rows, columns = operator.shape
    constraint = np.hstack((-operator, np.eye(rows)))
    identity = np.eye(rows + columns)
    return (constraint.conj().T @ constraint, identity[:columns],
            identity[columns:])


@pytest.mark.parametrize("value", [[1.0, 2.0], np.zeros((0, 2)),
                                   [[np.inf]], [[1.0, np.nan]]])
def test_a_matrix_that_is_not_nonempty_and_finite_is_refused(value):
    with pytest.raises(ValueError, match="nonempty finite matrix"):
        hc.recover_whole_relation(value, np.eye(1), np.eye(1))


def test_readouts_on_other_coordinates_are_refused():
    pencil, rin, rout = _graph(np.eye(2))
    with pytest.raises(ValueError, match="same whole-cochain coordinates"):
        hc.recover_whole_relation(pencil, rin[:, :3], rout)
    with pytest.raises(ValueError, match="same whole-cochain coordinates"):
        hc.recover_whole_relation(pencil[:3], rin, rout)


def test_a_kernel_frame_of_the_wrong_dimension_is_refused():
    pencil, rin, rout = _graph(np.eye(2))
    with pytest.raises(ValueError, match="wrong whole-cochain dimension"):
        hc.recover_whole_relation(pencil, rin, rout, kernel=np.eye(3))


def test_the_report_records_its_threshold_and_dimensions():
    operator = np.array([[1.0, 2.0j], [0.5, -1.0]])
    pencil, rin, rout = _graph(operator)
    read = hc.recover_whole_relation(pencil, rin, rout, tolerance=1e-10)
    assert read["identifiable"]
    assert read["cochain_count"] == 4
    assert read["kernel_dimension"] == 2
    assert read["input_dimension"] == read["output_dimension"] == 2
    assert read["relative_tolerance"] == 1e-10
    assert read["kernel_threshold"] == pytest.approx(
        1e-10 * read["pencil_singular_values"][0])
    assert read["hidden_mode_count"] == 0
    assert read["basis_input_error"] < 1e-12
    assert read["basis_output_error"] < 1e-12
    assert read["held_out"]["seed"] == 1096


def test_a_non_square_isometry_is_an_isometry_but_not_unitary():
    isometry = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    read = hc.operator_quantum_data(isometry)
    assert read["coordinate_isometry"]
    assert not read["coordinate_unitary"]
    assert read["tensor_dimensions"] == [3, 2]
    assert read["coordinate_isometry_error"] == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(read["choi"]["schmidt_coefficients"],
                               [1 / np.sqrt(2)] * 2, atol=1e-15)


def test_an_overflowing_isometry_error_is_named():
    read = hc.operator_quantum_data(np.array([[1e200, 0.0], [0.0, 1.0]]))
    assert read["coordinate_isometry_error"] == float("inf")
    assert not read["coordinate_isometry"]
    assert "floating-point range" in read["isometry_error_obstruction"]


def test_a_node_with_one_port_is_an_obstruction():
    node = SimpleNamespace(inputs=[object()])
    read = hc.measure_geometry(node)
    assert read == {"obstruction": "requires two declared ports; no "
                                   "tensor-factor map is defined for a direct "
                                   "sum"}


@pytest.fixture
def node():
    from tessera.drivers import qubit as qa
    built, _ = qa.build_qubit_node(qa.build_config(steps=0, grid=3,
                                                   regge=False))
    return built


def test_a_port_without_a_marking_is_an_obstruction(node):
    class Unmarked:
        def __getattr__(self, name):
            return getattr(node, name)

        def input_marking(self, index):
            return None if index == 1 else node.input_marking(index)

    read = hc.measure_geometry(Unmarked())
    assert read == {"obstruction": "both ports require geometric markings"}


def test_the_geometry_read_records_its_conventions(node):
    read = hc.measure_geometry(node)
    assert read["schema"] == 1 and read["degree"] == 1
    assert read["boundary_included"] is True
    assert read["readouts"]["periods"]["interpretation"].startswith(
        "transported marked periods")
    assert read["requested_gate"]["certified"] is False
    assert read["requested_gate"]["observed_input_dimension"] == 2


@pytest.mark.xfail(strict=True, reason=(
    "the tessera.drivers package docstring states that each of its listed "
    "modules (emergence, qubit, harmonic, fock) keeps a main() entry point "
    "for the command-line wrappers outside the repository, but "
    "tessera.drivers.harmonic defines no main()"))
def test_every_listed_driver_module_keeps_a_main_entry_point():
    for name in ("emergence", "qubit", "harmonic", "fock"):
        module = importlib.import_module("tessera.drivers." + name)
        assert callable(getattr(module, "main", None)), name
