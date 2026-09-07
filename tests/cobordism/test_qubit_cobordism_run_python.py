# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The qubit cobordism run, pinned at small size (#964, T5 of
``docs/design/qubit_cobordism_spec.md``).

The run itself (eight units and more, the weight scan, the 4x4 synthesis,
the phased run) lives in the records under ``~/cobordism-runs/qubit-cobordism/t5/``
and is reported in ``docs/design/qubit_cobordism_findings.md``. This file pins
the headline numbers of those records that a fast drive can reproduce, through
the SAME driver (``examples/cobordism/emergence_animation.py``, qubit mode):

* C1 on the 3x3 collar seed: a manifold whose boundary is exactly the two
  tori (two components of Euler characteristic 0, 18 faces each, complete
  with no uncovered face), Betti numbers [1, 2, 1, 0], the monodromy the
  identity to rounding, 54 cells / 18 vertices / 90 edges;
* the seed numbers of the record ``seed-3x3.json`` to 1e-6: the two block
  residuals of D2 (the torus's own holomorphic form against its input: zero
  on the seed), the output-state read (the whole's zero mode against
  (1, tau_in) in each live frame: T5's leaks and coefficients), the
  own-kernel leaks at their floor, the two-body leak against chi(S5) at
  J t = 0.05 with its transfer and Schmidt spectrum, the objective and its
  terms (r_U is the two-body leak alone on the seed);
* two units of synthesis at the chosen weights lower the objective while the
  block residuals stay at their floor and the surfaces stay the tori (C2 as
  the D2 wording of 2026-09-07 reads it: the tori keep representing their
  inputs on their own).

Where ``tests/cobordism/test_emergence_animation_qubit_python.py`` (T4) checks
that every channel is read, this file checks the numbers. Two units with two
stage-2 iterations each: about a minute at OMP_NUM_THREADS=2.
"""
import os
import sys

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
UNITS = 2
#: The chosen weights of the T5 findings note, which are now the driver's
#: defaults: input weight 1e4, Gamma = 1, the Regge term on.
WEIGHT = 1e4
REGGE = True

#: The record ``seed-3x3.json`` (frame 0 of every 3x3 run of T5); the objective
#: terms are frame 0 of ``scan-w1e4-regge-on.json`` (the weight enters r_U only).
SEED = {
    "counts": {"cells": 54, "vertices": 18, "edges": 90},
    "output_leaks": (3.0999811548462074e-3, 9.344558825277608e-3),
    "coefficients": ((0.9982485452788615 + 0.0011426511214279456j,
                      0.2984679431474786 + 1.0950467357199427j),
                     (0.9978162036715472 + 0.003176008132991126j,
                      -0.20130813617101526 + 0.7840718840403786j)),
    "two_body": 0.5261747058239845,
    "singular_values": (0.02606146984983397, 0.0002611716874913035),
    "transfer": ((-0.015919575774081807, -0.019358393245779218),
                 (-0.004326900396088621, -0.005689118777237639)),
    "regge_stationarity": 123.12355190179969,
    # r_U on the seed is the two-body leak alone: both block residuals of D2
    # are zero there (each torus IS its input torus on the collar seed).
    "register_residual": 0.5261747060460291,
    "total": 123.64972660784572,
}
FLOOR = 1e-24
_CACHE = {}


def _run():
    if "result" not in _CACHE:
        config = ea.build_config(steps=UNITS, stage1_iters=1, stage2_iters=2,
                                 tolerance=1e-30, inputs=ea.InputMode.QUBIT,
                                 tau_a=TAU_A, tau_b=TAU_B, grid=GRID,
                                 input_weight=WEIGHT, regge=REGGE)
        _CACHE["result"] = ea.drive(config, progress=False)
    return _CACHE["result"]


def _present(value):
    return not isinstance(value, ea.Absent)


def test_c1_the_collar_seed():
    result = _run()
    frame = result.frames[0]
    assert result.inputs.seed == SEED["counts"]
    assert frame.betti["numbers"] == {0: 1, 1: 2, 2: 1, 3: 0}
    assert _present(frame.boundary) and frame.boundary["count"] == 2
    assert frame.boundary["faces"] == 36
    assert [(c["euler_characteristic"], c["faces"], c["vertices"], c["block"])
            for c in frame.boundary["components"]] == [(0, 18, 9, 0), (0, 18, 9, 1)]
    assert frame.completion == {"bridge_phase_complete": True, "uncovered_faces": 0}
    assert _present(frame.monodromy), frame.monodromy
    assert frame.monodromy["betti"] == [1, 2, 1, 0]
    assert frame.monodromy["harmonic_rank"] == 2
    assert frame.monodromy["rounded"] == [[1, 0], [0, 1]]
    assert np.abs(np.asarray(frame.monodromy["monodromy"]) - np.eye(2)).max() < 1e-12
    assert frame.monodromy["rounding_residual"] < 1e-12
    assert frame.monodromy["fit_residual"] < 1e-12


def test_the_seed_residuals_are_the_records():
    result = _run()
    frame = result.frames[0]
    assert result.inputs.weight == WEIGHT and result.inputs.regge is REGGE
    assert result.inputs.algebra["Jt"] == 0.05
    for index, tau_in in enumerate((TAU_A, TAU_B)):
        row = frame.blocks[index]
        assert 0.0 <= row["residual"] < 1e-12
        assert row["output_leak"] == pytest.approx(SEED["output_leaks"][index], rel=1e-6)
        assert row["harmonic_rank"] == 2 and row["frame_rank"] == 2
        coefficients = np.asarray(row["coefficients"])
        assert np.abs(coefficients - np.asarray(SEED["coefficients"][index])).max() < 1e-6
        assert row["own_kernel_leak"] < FLOOR
        read = row["read"]
        assert _present(read), read.reason
        assert abs(read["tau"] - tau_in) < 1e-9
        assert (read["vertices"], read["edges"], read["faces"]) == (9, 27, 18)
    leaks = frame.leaks
    assert _present(leaks) and leaks["harmonic_rank"] == 2
    for index, row in enumerate(leaks["per_block"]):
        assert row["leak"] == pytest.approx(SEED["output_leaks"][index], rel=1e-6)
    two_body = frame.two_body
    assert _present(two_body), two_body
    assert two_body["in_frames"] and two_body["derived_frames"]
    assert two_body["shape"] == [2, 2] and two_body["schmidt_rank"] == 2
    assert two_body["residual"] == pytest.approx(SEED["two_body"], rel=1e-6)
    assert np.allclose(two_body["singular_values"], SEED["singular_values"], rtol=1e-6)
    transfer = np.asarray(two_body["transfer"])
    assert np.abs(transfer.imag).max() < 1e-12
    assert np.allclose(transfer.real, np.asarray(SEED["transfer"]), rtol=1e-6, atol=0)
    assert two_body["reversal_residual"] < 1e-8
    objective = frame.objective
    assert objective["regge_stationarity"] == pytest.approx(SEED["regge_stationarity"], rel=1e-6)
    assert objective["register_residual"] == pytest.approx(SEED["register_residual"], rel=1e-6)
    assert objective["total"] == pytest.approx(SEED["total"], rel=1e-6)
    assert objective["register_residual"] == pytest.approx(
        two_body["residual"] + WEIGHT * sum(frame.blocks[i]["residual"] for i in range(2)), rel=1e-9)
    assert objective["register_residual"] == pytest.approx(two_body["residual"], rel=1e-9), \
        "on the seed the block residuals are zero: r_U is the two-body leak alone"


def test_two_units_lower_the_objective_and_hold_the_tori():
    result = _run()
    first, last = result.frames[0], result.frames[-1]
    assert last.step == UNITS
    for index in range(2):
        assert first.blocks[index]["residual"] < 1e-12, "the seed is the input torus"
        after = last.blocks[index]["residual"]
        # the seed is the residual's minimum, so C2 under the D2 wording is
        # that the torus still represents its input after synthesis
        assert isinstance(after, float) and 0.0 <= after < 1e-3, (index, after)
        read = last.blocks[index]["read"]
        assert _present(read), read.reason
        assert (read["vertices"], read["edges"], read["faces"]) >= (9, 27, 18)
        assert after == pytest.approx(np.sin(read["fubini_study_distance"]) ** 2, rel=1e-9, abs=1e-18)
    assert last.objective["total"] < first.objective["total"]
    assert _present(last.monodromy) and last.monodromy["rounded"] == [[1, 0], [0, 1]]
    assert _present(last.two_body) and last.two_body["shape"] == [2, 2]
    assert last.betti["numbers"] == {0: 1, 1: 2, 2: 1, 3: 0}
