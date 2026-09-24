# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Target-independent whole-kernel relations for the qubit experiment.

This module never assembles a target-built Laplacian. Its numerical helpers
also accept independent analytical fixtures in tests. Quantum norms below
are explicitly the identity metrics in the declared port coordinates.
"""

import numpy as np


def _matrix(value, name):
    array = np.asarray(value, dtype=complex)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError("%s must be a nonempty finite matrix" % name)
    return array


def operator_quantum_data(operator, tolerance=1e-9):
    """Choi and norm certificates in identity-metric port coordinates."""
    matrix = _matrix(operator, "operator")
    rows, columns = matrix.shape
    with np.errstate(over="ignore", invalid="ignore"):
        error = float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(columns)))
    if not np.isfinite(error):
        error = float("inf")
    result = {"metric": "identity in declared port coordinates",
              "coordinate_isometry_error": error,
              "coordinate_isometry": error <= tolerance,
              "coordinate_unitary": rows == columns and error <= tolerance,
              "tensor_factors": ["output", "input reference"],
              "tensor_dimensions": [rows, columns],
              "vectorization": "output-major: index = output * input_dimension + input"}
    peak = float(max(np.max(np.abs(matrix.real)), np.max(np.abs(matrix.imag))))
    if peak == 0.0:
        result["choi"] = {"obstruction": "zero operator has no normalized Choi ray"}
        return result
    scaled = matrix / peak
    scaled_norm = float(np.linalg.norm(scaled))
    norm = peak * scaled_norm
    normalized = scaled / scaled_norm
    singular = np.linalg.svd(normalized, compute_uv=False)
    probabilities = singular**2
    probabilities /= probabilities.sum()
    positive = probabilities[probabilities > 0]
    result["choi"] = {
        "frobenius_norm": norm,
        "normalization_scale": peak,
        "scaled_frobenius_norm": scaled_norm,
        "normalized_vector": normalized.reshape(-1).tolist(),
        "schmidt_coefficients": singular.tolist(),
        "input_marginal": (normalized.T @ normalized.conj()).tolist(),
        "output_marginal": (normalized @ normalized.conj().T).tolist(),
        "input_reference_entropy_nats": float(-np.sum(positive * np.log(positive))),
        "interpretation": "normalized single-operator vector; not total-system entropy",
    }
    if not np.isfinite(error):
        result["isometry_error_obstruction"] = "norm error exceeds floating-point range"
    return result


def recover_whole_relation(pencil, input_readout, output_readout, *,
                           kernel=None, tolerance=1e-9, seed=1096):
    """Recover a unique operator and independently reconstruct its witnesses.

    Only the whole zero-frequency pencil and geometric observation matrices
    enter. ``kernel`` optionally supplies a different full-kernel frame for
    invariance tests. Missing input directions and output-visible hidden
    modes are obstructions, not a least-squares operator.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must lie strictly between zero and one")
    pencil = _matrix(pencil, "pencil")
    rin = _matrix(input_readout, "input readout")
    rout = _matrix(output_readout, "output readout")
    size = pencil.shape[0]
    if pencil.shape != (size, size) or rin.shape[1] != size or rout.shape[1] != size:
        raise ValueError("pencil and readouts must use the same whole-cochain coordinates")
    _, singular, vh = np.linalg.svd(pencil, full_matrices=True)
    scale = float(singular[0])
    threshold = tolerance * scale
    rank = int(np.count_nonzero(singular > threshold))
    q = vh[rank:].conj().T
    report = {"identifiable": False, "obstruction": "",
              "cochain_count": size, "kernel_dimension": size - rank,
              "input_dimension": rin.shape[0], "output_dimension": rout.shape[0],
              "relative_tolerance": tolerance, "kernel_threshold": threshold,
              "pencil_singular_values": singular.tolist(),
              "kernel_definition": "algebraic zero kernel; no semisimplicity assertion",
              "witness_norm": "Euclidean geometric-image norm, computational only"}
    if not q.shape[1]:
        report["obstruction"] = "whole pencil has no zero mode at the recorded threshold"
        return report
    if kernel is not None:
        supplied = _matrix(kernel, "kernel frame")
        if supplied.shape[0] != size:
            raise ValueError("kernel frame has the wrong whole-cochain dimension")
        u, values, _ = np.linalg.svd(supplied, full_matrices=False)
        supplied_rank = int(np.count_nonzero(values > tolerance * values[0]))
        candidate = u[:, :supplied_rank]
        if supplied_rank != q.shape[1] or np.linalg.norm(
                candidate - q @ (q.conj().T @ candidate)) > tolerance:
            raise ValueError("kernel frame does not span the whole zero kernel")
        q = candidate
    normalized_pencil = pencil / scale if scale else pencil
    report["kernel_residual"] = float(np.linalg.norm(normalized_pencil @ q))
    a, b = rin @ q, rout @ q
    u, values, vh = np.linalg.svd(a, full_matrices=True)
    input_rank = int(np.count_nonzero(values > tolerance * values[0])) if values.size else 0
    report["input_rank"] = input_rank
    report["input_singular_values"] = values.tolist()
    if input_rank != rin.shape[0]:
        report["obstruction"] = "input readout is not onto: some inputs have no whole harmonic extension"
        return report
    report["input_condition"] = float(values[0] / values[-1])
    right_inverse = vh[:input_rank].conj().T @ np.diag(1 / values) @ u.conj().T
    hidden = vh[input_rank:].conj().T
    ambiguity = float(np.linalg.norm(b @ hidden))
    output_scale = float(np.linalg.norm(b))
    relative_ambiguity = ambiguity / output_scale if output_scale else 0.0
    report["hidden_mode_count"] = hidden.shape[1]
    report["output_ambiguity"] = ambiguity
    report["relative_output_ambiguity"] = relative_ambiguity
    if relative_ambiguity > tolerance:
        report["obstruction"] = "input-invisible harmonic modes change the output"
        return report
    operator = b @ right_inverse
    witnesses = q @ right_inverse
    report["operator"] = operator.tolist()
    report["basis_whole_cochains"] = witnesses.tolist()
    report["basis_input_error"] = float(np.linalg.norm(rin @ witnesses - np.eye(rin.shape[0])))
    report["basis_output_error"] = float(np.linalg.norm(rout @ witnesses - operator))
    report["basis_harmonic_error"] = float(np.linalg.norm(normalized_pencil @ witnesses)
                                             / max(float(np.linalg.norm(witnesses)), 1.0))

    # Solve the full pencil/input equations afresh. These witnesses do not
    # come from multiplying the recovered operator or a synthesis target.
    random = np.random.default_rng(seed)
    inputs = random.normal(size=(rin.shape[0], 4)) + 1j * random.normal(size=(rin.shape[0], 4))
    inputs /= np.linalg.norm(inputs, axis=0)
    equations = np.vstack((normalized_pencil, rin))
    rhs = np.vstack((np.zeros((size, inputs.shape[1])), inputs))
    held, _, _, _ = np.linalg.lstsq(equations, rhs, rcond=tolerance)
    outputs = rout @ held
    errors = {"input": float(np.linalg.norm(rin @ held - inputs)),
              "output": float(np.linalg.norm(outputs - operator @ inputs)),
              "harmonic": float(np.linalg.norm(normalized_pencil @ held)
                                / max(float(np.linalg.norm(held)), 1.0))}
    report["held_out"] = {"seed": seed, "inputs": inputs.tolist(),
                           "outputs": outputs.tolist(), "errors": errors}
    if max(errors.values()) > tolerance * 10 or report["kernel_residual"] > tolerance * 10:
        report["obstruction"] = "whole-state reconstruction failed at the recorded tolerance"
        return report
    report["identifiable"] = True
    report["quantum"] = operator_quantum_data(operator, tolerance)
    return report


def measure_geometry(node):
    """Read a two-port whole relation from native scalar simplicial data.

    This entrypoint deliberately accepts neither animation inputs nor a
    target. Markings and live frames are attachment data on the node.
    """
    from tessera import chainhodge, cobordism as cob

    if len(node.inputs) != 2:
        return {"obstruction": "requires two declared ports; no tensor-factor map is defined for a direct sum"}
    assembled = cob.PencilLayer.assemble([node.spacetime()])
    op = assembled.op
    pencil = np.asarray(op.pencilAux(1))
    # Minv is the inverse chain metric, i.e. the Whitney cochain mass.
    mass = np.asarray(op.Minv(1).todense())
    connection = op.connection()
    size = pencil.shape[0]
    identity = np.eye(size, dtype=complex)
    period_rows, gram_rows, gram_obstructions = [], [], []
    for index in range(2):
        marking = node.input_marking(index)
        if marking is None:
            return {"obstruction": "both ports require geometric markings"}
        period_rows.append(np.asarray([
            [connection.transportedPeriod(identity[:, column], walk)
             for column in range(size)] for walk in marking.cycles]))
        derived = node.derive_input_frame(index)
        if derived.obstruction:
            gram_obstructions.append("port %d: %s" % (index, derived.obstruction))
            continue
        frame = derived.frame
        indices = list(cob.PencilLayer.indices_of(assembled, 1, frame.cells))
        if len(indices) != len(frame.cells) or any(row < 0 or row >= size for row in indices):
            gram_obstructions.append("port %d has a cell absent from the whole complex" % index)
            continue
        embedded = np.zeros((size, frame.rank()), dtype=complex)
        # The native frame already supplies the covariant left partner,
        # normalized by (F^vee)^T M_own F = I. Pair it with whole cochains
        # through the WHOLE Whitney mass, using the shared Gram primitive.
        embedded[indices] = np.asarray(frame.dual_images)
        gram_rows.append(np.asarray(chainhodge.PencilSchur.gramBlock(mass, embedded, identity)))
    result = {"schema": 1, "degree": 1, "primitive_data": "simplex topology, scalar lengths and phases",
              "boundary_included": True, "readouts": {}}
    result["readouts"]["periods"] = recover_whole_relation(pencil, *period_rows)
    result["readouts"]["gram"] = (
        {"identifiable": False, "obstruction": "; ".join(gram_obstructions)}
        if gram_obstructions else recover_whole_relation(pencil, *gram_rows))
    result["readouts"]["periods"]["interpretation"] = "transported marked periods; topological collar control"
    result["readouts"]["gram"]["observation_convention"] = "dual_frame_whitney"
    result["readouts"]["gram"]["interpretation"] = (
        "bilinear (F^vee)^T M_whole observation; (F^vee)^T M_own F = I; "
        "not a positive Whitney inner product")
    result["requested_gate"] = {
        "certified": False, "required_input_dimension": 4,
        "observed_input_dimension": period_rows[0].shape[0],
        "obstruction": (
            "port dimensions %d -> %d do not supply a 4 -> 4 two-qubit gate; chi is selected-state data"
            % (period_rows[0].shape[0], period_rows[1].shape[0])
            if any(rows.shape[0] != 4 for rows in period_rows) else
            "no geometric two-qubit subsystem-factor map has been declared"),
    }
    result["synthesis_interpretation"] = "historical selected-state objective; convergence is not whole-operator realization"
    return result


def main(argv=None):
    """Read the two-port whole relation of a qubit node as built, with no
    target and no relaxation: `measure_geometry` on the node that
    `qubit.build_qubit_node` seeds from the declared configuration. The
    record is written as JSON on standard output with ``--json``, and as a
    short text summary otherwise."""
    import argparse
    import json
    import sys

    from tessera.drivers import qubit

    parser = argparse.ArgumentParser(
        prog="tessera.drivers.harmonic",
        description="Measure the harmonic state/operator correspondence of "
                    "the qubit node's geometry: the whole relation between "
                    "its two ports, read from the zero-frequency pencil.")
    parser.add_argument("--grid", type=int, default=qubit.DECLARED_GRID,
                        help="vertices per side of each flat torus "
                             "(default %d)" % qubit.DECLARED_GRID)
    parser.add_argument("--no-regge", action="store_true",
                        help="build the node without the Regge term")
    parser.add_argument("--json", action="store_true",
                        help="write the record as JSON on standard output")
    args = parser.parse_args(argv)
    config = qubit.build_config(steps=0, grid=args.grid,
                                regge=not args.no_regge)
    node, _ = qubit.build_qubit_node(config)
    record = measure_geometry(node)
    if args.json:
        json.dump(record, sys.stdout, indent=2, sort_keys=True,
                  default=qubit._json_default)
        sys.stdout.write("\n")
        return 0
    if "obstruction" in record:
        print("obstruction: %s" % record["obstruction"])
        return 0
    for name, read in record["readouts"].items():
        print("%-8s identifiable %s%s" % (
            name, read.get("identifiable"),
            "; " + read["obstruction"] if read.get("obstruction") else ""))
    print("gate:    certified %s; %s" % (record["requested_gate"]["certified"],
                                         record["requested_gate"]["obstruction"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
