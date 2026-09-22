# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The inductive limit of the Fock stages over a refinement sequence (#1203).

The infinite Fock space is the direct limit of the finite stages under the
vacuum embedding, and the maps carried along it are consistent only when
``||iota_M V_M - V_{M+1} iota_M|| -> 0`` over a refinement sequence. That is a
statement about a SEQUENCE: one pair of stages gives one number and establishes
no limit. `LazyFockEngine.inductiveLimit` measures the defect at every adjacent
pair of a sequence on one and the same carried subspace, and
`tessera.drivers.fock` drives it over a stated fixture.

The expected defect is computed by hand rather than re-derived through the
bindings. With V_M the second quantization dGamma(h_M) of the stage's
one-particle operator, and a carried state b occupying only modes of the first
stage, the step from stage M to stage M+1 adds exactly the terms of dGamma that
involve the new mode M. On such a state the new mode is empty, so a_M and n_M
annihilate it and only ``sum_j h_Mj a_M^dagger a_j`` survives, sending b to the
orthonormal states ``b \\ {j} u {M}`` with amplitudes ``h_Mj``. Over the carried
subspace spanned by every occupation state of the first two modes the defect
matrix has the single row ``{M}`` carrying the two one-particle columns and the
two rows ``{0,M}``, ``{1,M}`` carrying the doubly occupied column, so its top
singular value is exactly ``sqrt(2) * |h_M0|`` when ``|h_M0| = |h_M1|``.

Skips cleanly when tessera was built without the quantum subsystem.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from tessera.drivers import fock

try:
    from tessera.quantum import LazyFockEngine  # noqa: F401
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def constant_coupling_stages(first=3, count=4, hopping=0.3, onsite=1.0):
    """A nested sequence whose added modes do NOT decouple: every off-diagonal
    entry is the same, so each new mode couples into the carried subspace
    exactly as strongly as the one before it."""
    largest = first + count - 1
    out = []
    for modes in range(first, largest + 1):
        h = np.full((modes, modes), hopping, dtype=complex)
        np.fill_diagonal(h, onsite)
        out.append(h)
    return out


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestTheDefectFallsAlongARefinementSequence(unittest.TestCase):
    """The acceptance: the defect falls along a refinement sequence."""

    def test_the_driver_measures_a_falling_defect(self):
        config = fock.build_config()
        read, record = fock.measure(config)
        self.assertEqual(len(read.defects), config["stages"] - 1)
        self.assertEqual(record["stageModes"],
                         list(range(config["first_stage_modes"],
                                    config["first_stage_modes"]
                                    + config["stages"])))
        self.assertEqual(read.activeDimension, 1 << config["active_modes"])
        self.assertTrue(read.falls)
        self.assertTrue(read.certificate.holds())
        # Every step falls by the fixture's decay ratio, so the worst ratio is
        # that ratio and the sequence is geometric.
        self.assertAlmostEqual(read.largestRatio, config["decay"], delta=1e-9)
        self.assertLess(read.lastDefect, read.defects[0])

    def test_each_defect_is_the_new_mode_coupling(self):
        """The hand-computed value: sqrt(2) * t * r ** M at the step that adds
        mode M, over the carried subspace of the first two modes."""
        config = fock.build_config(active_modes=2)
        read, _ = fock.measure(config)
        for step, defect in enumerate(read.defects):
            added = config["first_stage_modes"] + step
            expected = (math.sqrt(2.0) * config["hopping"]
                        * config["decay"] ** added)
            self.assertAlmostEqual(defect, expected, delta=1e-12 + 1e-9 * expected)

    def test_the_vacuum_alone_carries_no_defect(self):
        """A carried subspace of one active mode still sees the new mode, but a
        subspace that is the vacuum alone cannot: dGamma annihilates it, so both
        stages agree exactly and the defect is zero at every step."""
        config = fock.build_config()
        stages = fock.fock_stages(fock.one_particle_stages(config))
        engine = fock.qu.LazyFockEngine(len(stages[-1].modes))
        read = engine.inductiveLimit(stages, [[]])
        self.assertEqual(read.activeDimension, 1)
        self.assertEqual(max(read.defects), 0.0)
        # Nothing falls from zero to zero, and the read says so rather than
        # reporting a limit it did not measure.
        self.assertFalse(read.falls)
        self.assertEqual(read.largestRatio, 0.0)

    def test_a_sequence_that_does_not_decouple_does_not_fall(self):
        """A refinement whose added modes couple as strongly as the ones before
        them holds its defect flat. The certificate reports it; nothing here
        assumes a limit exists."""
        operators = constant_coupling_stages()
        read, record = fock.measure(fock.build_config(
            stages=len(operators), first_stage_modes=operators[0].shape[0]),
            operators=operators)
        self.assertFalse(read.falls)
        self.assertAlmostEqual(read.largestRatio, 1.0, delta=1e-12)
        expected = math.sqrt(2.0) * 0.3
        for defect in read.defects:
            self.assertAlmostEqual(defect, expected, delta=1e-12)
        self.assertFalse(record["falls"])

    def test_the_stage_map_is_the_second_quantization(self):
        """dGamma(h) is number-preserving and annihilates the vacuum, so its
        first column is exactly zero and its dimension is 2 ** modes."""
        stages = fock.fock_stages(fock.one_particle_stages(fock.build_config()))
        for stage in stages:
            modes = len(stage.modes)
            self.assertEqual(list(stage.support), list(stage.modes))
            self.assertEqual(np.asarray(stage.map).shape, (1 << modes,) * 2)
            np.testing.assert_array_equal(np.asarray(stage.map)[:, 0], 0.0)


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRefusals(unittest.TestCase):
    """A sequence that cannot establish a limit is refused by name."""

    def test_one_pair_of_stages_is_not_a_sequence(self):
        stages = fock.fock_stages(fock.one_particle_stages(fock.build_config()))
        engine = fock.qu.LazyFockEngine(len(stages[-1].modes))
        with self.assertRaisesRegex(ValueError, "at least two stages"):
            engine.inductiveLimit(stages[:1], [[], [0]])

    def test_a_sequence_too_short_to_fall_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least three stages"):
            fock.build_config(stages=2)

    def test_the_carried_subspace_must_fit_inside_the_first_stage(self):
        with self.assertRaisesRegex(ValueError, "inside the first stage"):
            fock.build_config(first_stage_modes=2, active_modes=3)

    def test_a_decay_ratio_that_does_not_decay_is_refused(self):
        with self.assertRaisesRegex(ValueError, "decay ratio"):
            fock.build_config(decay=1.0)

    def test_stages_that_are_not_nested_are_refused(self):
        operators = fock.one_particle_stages(fock.build_config())
        operators[1][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "not nested"):
            fock.fock_stages(operators)

    def test_a_sequence_that_does_not_grow_is_refused(self):
        operators = fock.one_particle_stages(fock.build_config())
        with self.assertRaisesRegex(ValueError, "a refinement sequence grows"):
            fock.fock_stages(list(reversed(operators)))


if __name__ == "__main__":
    unittest.main()
