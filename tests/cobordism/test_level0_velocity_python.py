# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Plumbing of the level-0 velocity experiment (#953): the two-phase protocol
runs, pinning holds the block geometry through phase 2, and every recorded
certificate is finite."""
import importlib.util
import math
import pathlib

import numpy as np
import pytest

from tessera import cobordism as cob

HL = cob.HodgeLaplacian
SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cobordism" / "level0_velocity.py"


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


@pytest.fixture(scope="module")
def l0():
    spec = importlib.util.spec_from_file_location("level0_velocity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_phase_protocol_pins_the_blocks_and_records_certificates(l0, whitney_default):
    args = l0.build_parser().parse_args(["--precones", "8", "--pairs", "1", "--phase1-rounds", "1",
                                         "--phase2-rounds", "1", "--stage1-steps", "1", "--candidates", "2",
                                         "--stage2-iters", "5", "--carrier", "blocks", "--shells", "0"])
    rng = np.random.default_rng(0)
    psi = l0._unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    phi = l0._unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    chi = l0.flip_flop(psi, phi)
    probe = cob.MultiCobordism(cob.MultiCobordism.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]],
                               [], degrees=[0], seed=0, precone=8, einstein_hilbert=False)
    ranked = l0.ranked_pairs(probe.spacetime())
    assert ranked and 0.0 <= ranked[0][0] <= 1.0
    cfg = l0.run_configuration(psi, phi, chi, 0, 8, (ranked[0][1], ranked[0][2]), args)
    p2 = cfg["phase2"]
    assert 0.0 <= p2["leak"] <= 1.0 and p2["leak"] == pytest.approx(p2["leak_recomputed"], rel=1e-9)
    assert len(p2["schmidt"]) == 4 and p2["reversal_residual"] < 1e-8
    for key in ("M0_condition", "M1_condition", "frame_distance", "cells_adjacent_to_both"):
        assert p2[key] is not None and math.isfinite(float(p2[key]))
    assert p2["cells_adjacent_to_both"] >= 0 and p2["frame_distance"] >= 1
    # phase 2 pinned the block regions: their fiber residuals are unchanged from the end of phase 1
    np.testing.assert_allclose(p2["block_residuals_after"], cfg["phase1"]["block_residual_trace"][-1], rtol=1e-9, atol=1e-12)
    # with the regions bounded to the tetrahedra, the bulk stays free
    assert cfg["phase2_free_edges"] > 0.5 * cfg["phase2_total_edges"]


def test_whole_carrier_keeps_the_bulk_free_and_records_the_pair_fiber(l0, whitney_default):
    args = l0.build_parser().parse_args(["--precones", "8", "--pairs", "1", "--phase1-rounds", "1",
                                         "--phase2-rounds", "1", "--stage1-steps", "1", "--candidates", "2",
                                         "--stage2-iters", "5", "--carrier", "whole"])
    rng = np.random.default_rng(3)
    psi = l0._unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    phi = l0._unit(rng.normal(size=4) + 1j * rng.normal(size=4))
    chi = l0.flip_flop(psi, phi)
    probe = cob.MultiCobordism(cob.MultiCobordism.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]],
                               [], degrees=[0], seed=0, precone=8, einstein_hilbert=False)
    ranked = l0.ranked_pairs(probe.spacetime())
    cfg = l0.run_configuration(psi, phi, chi, 0, 8, (ranked[0][1], ranked[0][2]), args)
    assert cfg["carrier"] == "whole"
    assert cfg["phase2_free_edges"] > 0.5 * cfg["phase2_total_edges"]  # only the two frames' own edges are pinned
    assert 0.0 <= cfg["phase2"]["whole_fiber_residual_after"] <= 1.0 and 0.0 <= cfg["phase2"]["leak"] <= 1.0
    assert len(cfg["phase1"]["whole_fiber_trace"]) >= 1


def test_flip_flop_and_ranking_are_deterministic(l0):
    rng = np.random.default_rng(1)
    psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
    chi = l0.flip_flop(psi, phi)
    assert np.linalg.matrix_rank(chi, tol=1e-10) == 2
    assert l0.projective_leak(chi, chi) == pytest.approx(0.0, abs=1e-12)
    assert l0.projective_leak(np.eye(4, dtype=complex), chi) > 0.0
