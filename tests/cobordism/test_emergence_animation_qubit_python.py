# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The qubit input mode of ``examples/cobordism/emergence_animation.py``
(#963, T4 of ``docs/design/qubit_cobordism_spec.md``, delta D4).

The qubit cobordism runs through the SAME drive as the neutral mode (spec R5):
``drive`` takes its node from a factory selected by ``config["inputs"]`` --
the neutral host by default, ``build_qubit_node`` for ``--inputs qubit`` --
and the loop, the engine calls and the frames are the same either way. The
qubit factory is the T1-T3 setup: two ``SimplicialQubit.flat_torus`` inputs,
their collar as the host, each torus one input block with its holomorphic
form attached as a degree-1 fiber on the harmonic contour, its marking with
the input coefficients (1, tau_in) on the block (the engine derives the
frame live, spec D2/D3 as revised), chi of spec S5 as the two-body target,
the block residuals at the input weight, the node's default objective on the
complex locus.

Coverage:

* the drive produces one frame per engine unit plus the seed, publishing
  each to the live callback as it completes, and every channel of spec S6
  is a measurement or an ``Absent`` with a named reason;
* frame 0 is the collar seed: each block's tau-hat equals tau_in to 1e-9
  with both distances at zero, the block residuals of D2 at zero and the
  output-state leaks (the whole's zero mode
  against (1, tau_in) in the live frame) at T4's restricted leaks -- 3.1e-3
  and 9.3e-3 on 3x3 tori, NOT at rounding, because the whole's harmonic
  representative restricted to a torus differs from the torus's own -- with
  the coefficients of the whole's zero mode reported next to the inputs and
  the own-kernel leaks at their floor as diagnostics, the monodromy the
  identity, Betti numbers [1, 2, 1, 0], the boundary two tori (Euler
  characteristic 0 each), the two-body read in the derived period frames
  (2 x 2, Schmidt rank 2), r_U the weighted block residuals plus the
  two-body leak;
* after synthesis every channel is still read, the block residuals stay at
  their floor
  from their seed values (spec C2) and the surfaces are still the tori;
* the algebra: chi of spec S5 against an explicit computation, the exact
  block evolution against first order at small J t, the selection rule;
* ``to_json`` round-trips through ``json.dumps`` with complex numbers as
  [re, im] pairs, and the run document carries the inputs once;
* ``render`` writes a PNG and a two-frame GIF under Agg;
* the neutral mode is unchanged: its config defaults, its record schema
  (no qubit key), and its tests, which run untouched next to this file;
* the config refuses a modulus off the upper half plane, a grid below 3, a
  non-positive weight and an unknown mode by name, and the CLI parses the
  qubit flags, ``--tau-b=-0.2+0.8j`` included.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pytest

import tessera as T  # noqa: F401  (the driver imports the package)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)
GRID = 3
STEPS = 2
FLOOR = 1e-24
#: The output-state read's leaks on the 3x3 collar seed (T2-bis): the
#: restricted leaks of the input lines in the whole's zero mode, which T4
#: measured. Since the D2 wording of 2026-09-07 they are reported, not
#: scored: the block residual is the torus's own (zero on the seed).
SEED_OUTPUT_LEAKS = (3.099981154846e-3, 9.344558825278e-3)

_CACHE = {}


def _run():
    """One shared qubit drive on 3x3 tori, built once for the whole file."""
    if "result" not in _CACHE:
        seen = []
        config = ea.build_config(steps=STEPS, stage1_iters=1, stage2_iters=2,
                                 tolerance=1e-30, inputs=ea.InputMode.QUBIT,
                                 tau_a=TAU_A, tau_b=TAU_B, grid=GRID)
        result = ea.drive(
            config, progress=False,
            on_frame=lambda frames, index: seen.append((index, len(frames))))
        _CACHE.update(result=result, config=config, seen=seen)
    return _CACHE


def _present(value):
    return not isinstance(value, ea.Absent)


def _read(frame, index):
    read = frame.blocks[index]["read"]
    assert _present(read), read.reason
    return read


# --------------------------------------------------------------------------- #
# the drive and the channels
# --------------------------------------------------------------------------- #
def test_one_frame_per_unit_published_live():
    run = _run()
    frames = run["result"].frames
    assert len(frames) == STEPS + 1
    assert [f.step for f in frames] == list(range(STEPS + 1))
    assert run["result"].terminator == ea.Terminator.STEPS
    assert run["seen"] == [(i, i + 1) for i in range(STEPS + 1)]
    assert run["result"].inputs is not None
    assert all(f.inputs is run["result"].inputs for f in frames)


def test_every_channel_is_a_measurement_or_a_named_absence():
    for frame in _run()["result"].frames:
        for name in ea.EmergenceFrame.QUBIT_CHANNELS + ("betti", "objective"):
            value = getattr(frame, name)
            if isinstance(value, ea.Absent):
                assert value.reason.strip(), "%s is absent without a reason" % name
            else:
                assert isinstance(value, (dict, list)), name
        assert isinstance(frame.blocks, list) and len(frame.blocks) == 2
        for row in frame.blocks:
            assert row["label"] in ea.DECLARED_TORUS_LABELS
            for key in ("residual", "output_leak", "coefficients", "own_kernel_leak", "read"):
                value = row[key]
                assert _present(value) or value.reason.strip(), key
        # the neutral mode's certificates are absent by name here
        for name in ea.EmergenceFrame.CERTIFICATE_CHANNELS:
            value = getattr(frame, name)
            assert isinstance(value, ea.Absent) and "qubit input mode" in value.reason
        assert isinstance(frame.dual, ea.Absent) and "3-dimensional" in frame.dual.reason
        # the tori highlighted in the layout
        assert _present(frame.layout)
        assert [(m["label"], m["missing"]) for m in frame.layout["highlight"]] == [("A", 0), ("B", 0)]
        assert all(len(m["edges"]) == 27 for m in frame.layout["highlight"])


def test_frame_zero_is_the_collar_seed():
    run = _run()
    frame = run["result"].frames[0]
    inputs = run["result"].inputs
    assert inputs.tau_in == [TAU_A, TAU_B]
    assert inputs.objective_name == "legacy"
    assert inputs.weight == ea.DECLARED_INPUT_WEIGHT and inputs.regge
    assert inputs.seed == {"cells": 54, "vertices": 18, "edges": 90}
    for index, tau_in in enumerate((TAU_A, TAU_B)):
        row = frame.blocks[index]
        assert row["tau_in"] == tau_in and row["weight"] == ea.DECLARED_INPUT_WEIGHT
        assert row["input"] == [1.0 + 0j, tau_in] == inputs.coefficients_in[index]
        # the block residual (spec D2): the torus's own holomorphic form against
        # its input -- zero on the seed, where the surface is the flat torus
        assert 0.0 <= row["residual"] < 1e-12, row["residual"]
        # the output-state read (spec R1): the whole's zero mode against (1, tau_in)
        # in the live frame -- T4's restricted leak on the seed, not zero
        assert row["output_leak"] == pytest.approx(SEED_OUTPUT_LEAKS[index], rel=1e-6), row["output_leak"]
        assert row["harmonic_rank"] == 2 and row["frame_rank"] == 2
        coefficients = np.asarray(row["coefficients"])
        assert np.abs(coefficients - np.array([1.0, tau_in])).max() < 0.03
        # the former own-kernel leak is a diagnostic at its floor
        assert row["own_kernel_leak"] < FLOOR, row["own_kernel_leak"]
        read = _read(frame, index)
        assert abs(read["tau"] - tau_in) < 1e-9, read["tau"]
        assert read["fubini_study_distance"] < 1e-9 and read["weil_petersson_distance"] < 1e-9
        assert (read["vertices"], read["edges"], read["faces"]) == (9, 27, 18)
        assert not read["marking_swapped"] and not read["near_degenerate"]
        assert read["j_residual"] < 1e-12
        bloch = np.asarray(read["bloch"])
        assert abs(np.linalg.norm(bloch) - 1.0) < 1e-12 and bloch[1] > 0
        assert np.allclose(bloch, np.asarray(inputs.tori[index].bloch()), atol=1e-12)
    # the whole: Betti numbers, boundary, completion, monodromy, zero mode
    assert frame.betti["numbers"] == {0: 1, 1: 2, 2: 1, 3: 0}
    assert _present(frame.boundary)
    assert frame.boundary["count"] == 2 and frame.boundary["faces"] == 36
    assert [(c["euler_characteristic"], c["faces"], c["block"]) for c in frame.boundary["components"]] == [(0, 18, 0), (0, 18, 1)]
    assert frame.completion == {"bridge_phase_complete": True, "uncovered_faces": 0}
    assert _present(frame.monodromy), frame.monodromy
    assert frame.monodromy["betti"] == [1, 2, 1, 0] and frame.monodromy["harmonic_rank"] == 2
    assert frame.monodromy["rounded"] == [[1, 0], [0, 1]]
    assert frame.monodromy["rounding_residual"] < 1e-9 and frame.monodromy["fit_residual"] < 1e-9
    assert _present(frame.leaks) and frame.leaks["harmonic_rank"] == 2
    for index, row in enumerate(frame.leaks["per_block"]):
        assert _present(row) and 0.0 <= row["leak"] <= 1.0 and row["rank"] == 2
        # the leak of the input line is the output-state read's leak on the seed (the same target up to scale)
        assert row["leak"] == pytest.approx(frame.blocks[index]["output_leak"], rel=1e-9)
    # the two-body read in the period frames against chi of spec S5
    two_body = frame.two_body
    assert _present(two_body), two_body
    assert two_body["in_frames"] and two_body["derived_frames"] and two_body["choi_decomposed"]
    assert two_body["shape"] == [2, 2]
    assert 0.0 <= two_body["residual"] <= 1.0
    assert two_body["schmidt_rank"] == 2 and len(two_body["singular_values"]) == 2
    assert two_body["reversal_residual"] < 1e-8
    assert all(r < FLOOR for r in two_body["input_fiber_residuals"])
    transfer = np.asarray(two_body["transfer"])
    assert transfer.shape == (2, 2) and np.abs(transfer.imag).max() < 1e-12 * np.abs(transfer).max()
    # the objective: the node's default, Regge stationarity plus Gamma r_U
    assert frame.objective["total"] is not None
    assert frame.objective["regge_stationarity"] > 0
    # r_U: the two block residuals at their weight plus the two-body residual
    expected = two_body["residual"] + ea.DECLARED_INPUT_WEIGHT * sum(frame.blocks[i]["residual"] for i in range(2))
    assert frame.objective["register_residual"] == pytest.approx(expected, rel=1e-9)
    print("\n[T4] seed: blocks %s, output leaks %s (weight %g), coefficients %s, leaks %s, two-body %.6f, T %s" % (
        ["%.3e" % frame.blocks[i]["residual"] for i in range(2)],
        ["%.3e" % frame.blocks[i]["output_leak"] for i in range(2)], ea.DECLARED_INPUT_WEIGHT,
        [np.round(np.asarray(frame.blocks[i]["coefficients"]), 4).tolist() for i in range(2)],
        ["%.3e" % r["leak"] for r in frame.leaks["per_block"]], two_body["residual"],
        np.round(transfer.real, 6).tolist()))


def test_synthesis_reads_every_channel_and_holds_the_tori():
    run = _run()
    frames = run["result"].frames
    last = frames[-1]
    for index, tau_in in enumerate((TAU_A, TAU_B)):
        row = last.blocks[index]
        # The seed sits at the residual's minimum (the surface IS the input
        # torus there), so synthesis cannot lower it; spec C2 as the D2
        # wording reads it is that the torus keeps representing its input,
        # which the residual measures directly.
        assert isinstance(row["residual"], float) and 0.0 <= row["residual"] < 1e-2, row["residual"]
        assert _present(row["coefficients"]) and len(row["coefficients"]) == 2
        assert isinstance(row["own_kernel_leak"], float)
        read = _read(last, index)
        assert (read["vertices"], read["edges"], read["faces"]) >= (9, 27, 18), "the surface keeps the torus"
        # the residual IS the read: r = 1 - |<psi_in|psi_hat>|^2 = sin^2(d_FS)
        assert row["residual"] == pytest.approx(np.sin(read["fubini_study_distance"]) ** 2, rel=1e-9, abs=1e-18)
        assert read["weil_petersson_distance"] >= 0.0
        assert abs(np.linalg.norm(read["bloch"]) - 1.0) < 1e-12
    assert _present(last.monodromy) and _present(last.two_body) and _present(last.leaks)
    assert _present(last.boundary) and _present(last.completion) and _present(last.betti)
    assert last.two_body["in_frames"] and last.two_body["derived_frames"] and last.two_body["shape"] == [2, 2]
    totals = [f.objective["total"] for f in frames]
    assert all(t is not None for t in totals) and totals[-1] < totals[0]
    print("\n[T4] after %d units: objective %s, blocks %s, output leaks %s, two-body %s, tau-hat %s, monodromy %s" % (
        STEPS, ["%.4f" % t for t in totals],
        ["%.2e" % f.blocks[0]["residual"] + "/%.2e" % f.blocks[1]["residual"] for f in frames],
        ["%.2e" % f.blocks[0]["output_leak"] + "/%.2e" % f.blocks[1]["output_leak"] for f in frames],
        ["%.4f" % f.two_body["residual"] for f in frames],
        [str(_read(last, i)["tau"]) for i in range(2)], last.monodromy["rounded"]))


# --------------------------------------------------------------------------- #
# the algebra of spec S5
# --------------------------------------------------------------------------- #
def test_the_flip_flop_algebra():
    rng = np.random.default_rng(963)
    psi, phi = (rng.normal(size=2) + 1j * rng.normal(size=2) for _ in range(2))
    psi, phi = psi / np.linalg.norm(psi), phi / np.linalg.norm(phi)
    chi = ea.two_qubit_flip_flop(psi, phi)
    lowering = np.array([[0, 0], [1, 0]], dtype=complex)
    expected = np.outer(lowering @ psi, lowering.T @ phi) + np.outer(lowering.T @ psi, lowering @ phi)
    assert np.allclose(chi, expected, atol=1e-15)
    assert chi[0, 0] == 0 and chi[1, 1] == 0, "the selection rule: chi lives in the |01>, |10> sector"
    bilinear = (np.kron(lowering, lowering.T) + np.kron(lowering.T, lowering)) @ np.kron(psi, phi)
    assert np.allclose(chi, bilinear.reshape(2, 2)), "chi is the bilinear map's image of the product"
    small = ea.flip_flop_evolution(psi, phi, 2.0, 5e-5)
    product = np.kron(psi, phi).reshape(2, 2)
    assert np.allclose(small["product_state"], product)
    assert np.allclose(small["first_order_amplitudes"], -1j * 1e-4 * chi)
    velocity = (small["exact_amplitudes"] - product) / (-1j * 1e-4)
    assert np.abs(velocity - chi).max() < 1e-3, "the exact evolution's first order is -i J t chi"
    large = ea.flip_flop_evolution(psi, phi, 1.0, 0.7)
    exact = large["exact_amplitudes"]
    assert abs(np.linalg.norm(exact) - 1.0) < 1e-12, "unitary"
    assert exact[0, 0] == product[0, 0] and exact[1, 1] == product[1, 1], "the size-1 blocks are untouched"
    c, s = np.cos(0.7), np.sin(0.7)
    assert np.allclose(exact[0, 1], c * product[0, 1] - 1j * s * product[1, 0])
    assert np.allclose(exact[1, 0], c * product[1, 0] - 1j * s * product[0, 1])
    assert large["Jt"] == 0.7


# --------------------------------------------------------------------------- #
# serialization and rendering
# --------------------------------------------------------------------------- #
def test_to_json_round_trips_with_complex_numbers_as_pairs():
    run = _run()
    for frame in run["result"].frames:
        document = json.loads(json.dumps(frame.to_json()))
        assert document["step"] == frame.step
        for name in ea.EmergenceFrame.QUBIT_CHANNELS:
            assert name in document
        read = document["blocks"][0]["read"]
        assert read["tau"] == [frame.blocks[0]["read"]["tau"].real, frame.blocks[0]["read"]["tau"].imag]
        assert document["blocks"][0]["tau_in"] == [TAU_A.real, TAU_A.imag]
        assert len(document["two_body"]["transfer"]) == 2 and len(document["two_body"]["transfer"][0][0]) == 2
        assert document["monodromy"]["rounded"] == frame.monodromy["rounded"]
    inputs = json.loads(json.dumps(run["result"].inputs.to_json()))
    assert inputs["tau_in"] == [[TAU_A.real, TAU_A.imag], [TAU_B.real, TAU_B.imag]]
    assert inputs["grid"] == [GRID * GRID, GRID * GRID]
    assert inputs["grid_size"] == GRID
    assert inputs["vertices_per_torus"] == [GRID * GRID, GRID * GRID]
    assert inputs["objective"] == "legacy" and "register_residual" in inputs["objective_terms"]
    assert "reversed" not in inputs, "the engine's read fixes the orientation by A.B = +1"
    assert len(inputs["algebra"]["chi"]) == 2 and inputs["algebra"]["Jt"] == ea.DECLARED_COUPLING * ea.DECLARED_TIME
    assert inputs["markings"][0][0][0] == [0, 1] or len(inputs["markings"][0][0]) == GRID
    whole = json.dumps({"config": run["config"], "inputs": inputs,
                        "frames": [f.to_json() for f in run["result"].frames]}, sort_keys=True)
    assert json.loads(whole)["config"]["inputs"] == "qubit"


def test_render_png_and_gif_under_agg():
    frames = _run()["result"].frames
    with tempfile.TemporaryDirectory() as directory:
        png = os.path.join(directory, "qubit.png")
        gif = os.path.join(directory, "qubit.gif")
        one = os.path.join(directory, "one-frame.gif")
        ea.render(frames, png)
        ea.render(frames[:2], gif)
        ea.render(frames[:1], one)
        assert all(os.path.getsize(path) > 1024 for path in (png, gif, one))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(18, 10))
    try:
        ea.draw_frame(figure, frames, len(frames) - 1)
        assert "qubit cobordism" in figure._suptitle.get_text()
        assert "legacy" in figure._suptitle.get_text()
        titles = [axis.get_title() for axis in figure.axes]
        panel_text = titles + [text.get_text() for axis in figure.axes
                               for text in axis.texts]
        # The qubit panels name their quantities in mathtext (#993), so the
        # tokens are the symbols as they are written in the source, not the
        # glyphs they render as.
        for token in ("residuals", r"$\hat\tau$", "Bloch", r"|T_{AB}|", "the whole"):
            assert any(token in label for label in panel_text), token
    finally:
        plt.close(figure)
    assert [name for name, _ in ea.panels_for(_run()["config"])][:6] == list(ea._QUBIT_PANEL_ORDER[:6])
    rows, columns = ea.DECLARED_PANEL_GRID
    assert len(ea._PANELS) == len(ea._QUBIT_PANELS) <= rows * columns
    assert {name for name, _ in ea._PANELS} == {name for name, _ in ea._QUBIT_PANELS}


# --------------------------------------------------------------------------- #
# the neutral mode is unchanged
# --------------------------------------------------------------------------- #
def test_the_neutral_mode_is_the_default_and_keeps_its_record():
    config = ea.build_config()
    assert config["inputs"] == ea.InputMode.NEUTRAL == ea.DECLARED_INPUTS
    assert config["betti_degrees"] == list(ea.DECLARED_BETTI_DEGREES)
    assert ea.NODE_FACTORIES == {ea.InputMode.QUBIT: ea.build_qubit_node}
    result = ea.drive(ea.build_config(size=4, steps=0), progress=False)
    assert result.inputs is None and len(result.frames) == 1
    frame = result.frames[0]
    document = frame.to_json()
    for name in ea.EmergenceFrame.QUBIT_CHANNELS:
        assert name not in document
        assert isinstance(getattr(frame, name), ea.Absent)
    assert frame.layout["highlight"] == []
    assert [name for name, _ in ea.panels_for(config)] == [name for name, _ in ea._PANELS]
    assert [name for name, _ in ea._PANELS[:14]] == [
        "objective", "layout", "dual_spatial", "dual_temporal", "clusters", "bands", "anchors",
        "transports", "statistics", "crossings", "mass", "spin", "betti", "verdict"]


# --------------------------------------------------------------------------- #
# the config and the CLI
# --------------------------------------------------------------------------- #
def test_config_refusals_by_name():
    with pytest.raises(ValueError, match="tau_a must lie in the upper half plane"):
        ea.build_config(inputs=ea.InputMode.QUBIT, tau_a=0.3 - 1.1j)
    with pytest.raises(ValueError, match="tau_b must lie in the upper half plane"):
        ea.build_config(inputs=ea.InputMode.QUBIT, tau_b=0.5)
    with pytest.raises(ValueError, match="grid must be at least 3"):
        ea.build_config(inputs=ea.InputMode.QUBIT, grid=2)
    with pytest.raises(ValueError, match="input weight must be a positive finite number"):
        ea.build_config(inputs=ea.InputMode.QUBIT, input_weight=0.0)
    with pytest.raises(ValueError, match="J must be finite"):
        ea.build_config(inputs=ea.InputMode.QUBIT, coupling=float("nan"))
    with pytest.raises(ValueError, match="unknown input mode 'qbit'"):
        ea.build_config(inputs="qbit")
    config = ea.build_config(inputs=ea.InputMode.QUBIT, tau_a="0.3+1.1j", tau_b=[-0.2, 0.8])
    assert config["tau_a"] == [0.3, 1.1] and config["tau_b"] == [-0.2, 0.8]
    assert config["layers"] == ea.DECLARED_COLLAR_LAYERS == 1


def test_the_cli_parses_the_qubit_flags():
    parser = ea.build_parser()
    args = parser.parse_args(["run", "--inputs", "qubit", "--tau-a", "0.3+1.1j", "--tau-b=-0.2+0.8j",
                              "--grid", "4", "--J", "2.0", "--time", "0.1", "--input-weight", "1e3",
                              "--no-regge"])
    assert args.inputs == "qubit" and args.tau_a == TAU_A and args.tau_b == TAU_B
    assert args.grid == 4 and args.coupling == 2.0 and args.time == 0.1
    assert args.input_weight == 1e3 and args.regge is False
    defaults = parser.parse_args(["run"])
    assert defaults.inputs == ea.InputMode.NEUTRAL and defaults.regge is ea.DECLARED_REGGE
    assert defaults.tau_a == ea.DECLARED_TAU_A and defaults.grid == ea.DECLARED_GRID
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--inputs", "qbit"])
    with pytest.raises(argparse.ArgumentTypeError):
        ea._complex_argument("abc")


def test_main_writes_the_run_document():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "run.json")
        ea.main(["run", "--inputs", "qubit", "--tau-a", "0.3+1.1j", "--tau-b=-0.2+0.8j",
                 "--grid", "3", "--steps", "1", "--stage-two-iterations", "1",
                 "--tolerance", "1e-30", "--json", path, "--out", "", "--quiet"])
        with open(path) as handle:
            document = json.load(handle)
    assert document["config"]["inputs"] == "qubit" and document["config"]["grid"] == 3
    assert document["terminator"] in ea.Terminator.ALL
    assert document["inputs"]["objective"] == "legacy"
    assert len(document["frames"]) == 2
    for frame in document["frames"]:
        for name in ea.EmergenceFrame.QUBIT_CHANNELS:
            assert name in frame
        assert frame["blocks"][0]["read"]["tau"][1] > 0
