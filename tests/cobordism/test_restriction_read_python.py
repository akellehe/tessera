"""`MultiCobordism.restriction`: one harmonic basis, every marking's periods.

`monodromy` reads two markings and fits `P_B = M P_A`; `restriction` reads
any number of markings and returns the basis `Z` those periods were taken
from. The checks here are the ones a consumer relies on: the two reads
agree on everything basis-independent (Betti numbers, rank, and `M`), the
periods really are the periods of the returned `Z` (transported over the
ordered marking, exactly as `monodromy` takes them), and a marking off the
whole is named rather than guessed.
"""
import os
import sys

import numpy as np
import pytest

import tessera as T

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))
from tessera.drivers import qubit as qa

MC = T.cobordism.MultiCobordism

GRID = 3
_CACHE = {}


def _seeded(collar_twist="none"):
    """The collar seed at `GRID`, its markings, and its host: read-only."""
    if collar_twist not in _CACHE:
        config = qa.build_config(grid=GRID, collar_twist=collar_twist)
        node, inputs = qa.build_qubit_node(config)
        _CACHE[collar_twist] = (node, inputs, node.spacetime())
    return _CACHE[collar_twist]


def _monodromy_from(periods_a, periods_b):
    return np.linalg.solve(periods_a.T, periods_b.T).T


def test_the_two_reads_agree_on_everything_basis_independent():
    _, inputs, st = _seeded()
    restricted = MC.restriction(st, inputs.markings)
    assert restricted.obstruction == "", restricted.obstruction
    direct = MC.monodromy(st, inputs.markings[0], inputs.markings[1])
    assert direct.obstruction == "", direct.obstruction
    assert list(restricted.betti) == list(direct.betti)
    assert restricted.harmonic_rank == direct.harmonic_rank == 2
    assert len(restricted.periods) == 2
    fitted = _monodromy_from(np.asarray(restricted.periods[0]),
                             np.asarray(restricted.periods[1]))
    np.testing.assert_allclose(fitted, np.asarray(direct.monodromy),
                               atol=1e-12, rtol=0)


def test_the_periods_are_the_periods_of_the_returned_basis():
    """`images` and `periods` come from ONE basis: re-taking the periods of
    `images` over the ordered marking reproduces `periods`, column by column,
    with the same transport `monodromy` uses."""
    _, inputs, st = _seeded()
    restricted = MC.restriction(st, inputs.markings)
    assert restricted.obstruction == "", restricted.obstruction
    images = np.asarray(restricted.images)
    assert images.shape[1] == restricted.harmonic_rank
    # The pencil the read assembled, re-assembled: the connection is the
    # same object either way, so the transported period of a column over a
    # marking cycle must agree with what the read reported.
    assembled = T.cobordism.PencilLayer.assemble([st])
    connection = assembled.op.connection()
    for index, marking in enumerate(inputs.markings):
        reported = np.asarray(restricted.periods[index])
        assert reported.shape == (len(marking), images.shape[1])
        # `restriction` orders each cycle into a closed walk and rotates it to
        # the marking's common base point before transporting, and a
        # transported sum depends on where the walk starts. The base point is
        # not reported, so every rotation of the closed walk is tried: exactly
        # the one the read used reproduces its periods.
        for c, cycle in enumerate(marking):
            walk = [(int(u), int(v)) for u, v in cycle]
            rotations = [walk[k:] + walk[:k] for k in range(len(walk))]
            matched = False
            for rotated in rotations:
                periods = np.array([connection.transportedPeriod(images[:, a], rotated)
                                    for a in range(images.shape[1])])
                if np.abs(periods - reported[c]).max() < 1e-12:
                    matched = True
                    break
            assert matched, (c, reported[c])


def test_a_marking_off_the_whole_is_named():
    _, inputs, st = _seeded()
    bogus = [[(10**9, 10**9 + 1)]]
    read = MC.restriction(st, inputs.markings + [bogus])
    assert "is not an edge of the whole" in read.obstruction
    assert len(read.periods) == 0


def test_the_swap_collar_reads_the_swap():
    """The one projectively non-trivial simplicial mapping class of the grid
    torus: `M` is the swap, integer, det -1, from the restriction's periods."""
    _, inputs, st = _seeded("swap")
    restricted = MC.restriction(st, inputs.markings)
    assert restricted.obstruction == "", restricted.obstruction
    fitted = _monodromy_from(np.asarray(restricted.periods[0]),
                             np.asarray(restricted.periods[1]))
    rounded = np.rint(fitted.real).astype(int)
    assert np.abs(fitted - rounded).max() < 1e-9
    assert rounded.tolist() == [[0, 1], [1, 0]]


def test_verify_subcommand_passes_on_the_seeded_collar(tmp_path):
    """`qubit_animation.py verify` reads the swap collar at the default grid
    and every check passes; about seven seconds, no stage runs."""
    import json
    import subprocess

    script = qa.__file__
    record = tmp_path / "verify.json"
    result = subprocess.run(
        [sys.executable, script, "verify", "--collar-twist", "swap",
         "--json", str(record)],
        capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
    document = json.loads(record.read_text())
    assert document["all_pass"]
    assert {row["id"] for row in document["checks"]} >= {
        "R1", "H1:seed", "H2:seed", "H3:seed", "R3:A->B", "R4:A->B", "R5:A->B",
        "J1:A->B", "H1:jittered", "H2:jittered", "H3:jittered", "J2", "J3",
        "U1:A->B", "C1:none", "C1:swap"}
