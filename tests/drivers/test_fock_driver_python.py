# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The command line and the remaining refusals of the Fock inductive-limit
driver (`tessera.drivers.fock`).

The measurement itself is held in
``tests/quantum/test_fock_inductive_limit_python.py``; this file covers what a
caller reaches through `main` (the text and JSON outputs and the argument
parsing), the configuration refusals not exercised there, and the carried
subspace the driver declares. The fixture's defect at the step from stage M to
stage M + 1 is the coupling of the new mode into the carried subspace, of size
t r^M for hopping t and decay ratio r, so consecutive defects fall by r.

Skips cleanly when tessera was built without the quantum subsystem.
"""
from __future__ import annotations

import contextlib
import io
import json
import unittest

import numpy as np

from tessera.drivers import fock

try:
    from tessera.quantum import LazyFockEngine  # noqa: F401
    HAVE_QUANTUM = True
except ImportError:
    HAVE_QUANTUM = False


def run_main(argv):
    """`fock.main` with its standard output captured."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fock.main(argv)
    return code, out.getvalue()


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestTheCommandLine(unittest.TestCase):
    """`main` prints the defects, or writes them as JSON, and exits zero."""

    def test_the_json_record(self):
        code, out = run_main(["--stages", "4", "--first-stage-modes", "3",
                              "--active-modes", "2", "--decay", "0.5",
                              "--json"])
        self.assertEqual(code, 0)
        record = json.loads(out)
        self.assertEqual(record["stageModes"], [3, 4, 5, 6])
        self.assertEqual(record["activeModes"], 2)
        self.assertEqual(record["activeDimension"], 4)
        self.assertEqual(len(record["defects"]), 3)
        self.assertTrue(record["falls"])
        self.assertTrue(record["certified"])
        # consecutive defects fall by the declared decay ratio
        ratios = np.array(record["defects"][1:]) / np.array(
            record["defects"][:-1])
        np.testing.assert_allclose(ratios, 0.5, rtol=1e-10)
        self.assertAlmostEqual(record["largestRatio"], 0.5, places=10)
        self.assertEqual(sorted(record), sorted(
            ["stageModes", "activeModes", "activeDimension", "defects",
             "lastDefect", "largestRatio", "falls", "certified",
             "residual"]))

    def test_the_text_report(self):
        code, out = run_main([])
        self.assertEqual(code, 0)
        self.assertIn("stages (modes):      [3, 4, 5, 6, 7, 8]", out)
        self.assertIn("carried subspace:    2 modes, dimension 4", out)
        self.assertEqual(out.count("  defect "), fock.DECLARED_STAGES - 1)
        self.assertIn("  defect 3 -> 4:", out)
        self.assertIn("the defect falls:    True", out)

    def test_every_fixture_parameter_reaches_the_measurement(self):
        _, out = run_main(["--hopping", "2", "--onsite", "0.5",
                           "--onsite-step", "0", "--decay", "0.25",
                           "--json"])
        doubled = json.loads(out)
        _, out = run_main(["--decay", "0.25", "--json"])
        reference = json.loads(out)
        np.testing.assert_allclose(doubled["defects"],
                                   2.0 * np.array(reference["defects"]),
                                   rtol=1e-10)

    def test_a_bad_configuration_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "at least three stages"):
            run_main(["--stages", "2"])
        with self.assertRaisesRegex(ValueError, "decay ratio"):
            run_main(["--decay", "1.5"])

    def test_an_unparseable_argument_exits_with_usage(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as stop:
                fock.main(["--stages", "many"])
        self.assertEqual(stop.exception.code, 2)
        self.assertIn("--stages", err.getvalue())


@unittest.skipUnless(HAVE_QUANTUM, "tessera built without the quantum subsystem")
class TestRefusalsAndTheCarriedSubspace(unittest.TestCase):
    """The refusals `build_config`, `fock_stages` and `measure` make by name,
    and the carried subspace."""

    def test_a_first_stage_without_modes_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one mode"):
            fock.build_config(first_stage_modes=0, active_modes=0)

    def test_an_empty_carried_subspace_is_refused(self):
        with self.assertRaisesRegex(ValueError, "inside the first stage"):
            fock.build_config(active_modes=0)

    def test_a_non_square_stage_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nonempty square matrix"):
            fock.fock_stages([np.zeros((2, 3))])
        with self.assertRaisesRegex(ValueError, "nonempty square matrix"):
            fock.fock_stages([np.zeros((0, 0))])

    def test_supplied_operators_smaller_than_the_carried_subspace_are_refused(
            self):
        config = fock.build_config(first_stage_modes=3, active_modes=3)
        operators = [np.eye(m, dtype=complex) for m in (2, 3, 4)]
        with self.assertRaisesRegex(ValueError, "inside the first stage"):
            fock.measure(config, operators)

    def test_the_carried_subspace_is_every_occupation_of_its_modes(self):
        self.assertEqual(fock.active_basis(2), [[], [0], [1], [0, 1]])
        self.assertEqual(len(fock.active_basis(3)), 8)

    def test_the_fixture_is_nested_and_its_couplings_decay(self):
        config = fock.build_config(stages=3, first_stage_modes=2)
        operators = fock.one_particle_stages(config)
        self.assertEqual([h.shape[0] for h in operators], [2, 3, 4])
        for smaller, larger in zip(operators, operators[1:]):
            n = smaller.shape[0]
            np.testing.assert_array_equal(larger[:n, :n], smaller)
        largest = operators[-1]
        self.assertAlmostEqual(largest[3, 0].real, 0.25 ** 3)
        self.assertAlmostEqual(largest[2, 2].real, 1.0 + 2 * 0.5)

    def test_a_stage_is_the_second_quantization(self):
        """dGamma of a diagonal one-particle operator is diagonal on the Fock
        basis with the sums of the occupied entries."""
        (stage,) = fock.fock_stages([np.diag([1.0, 2.0])])
        self.assertEqual(list(stage.modes), [0, 1])
        values = sorted(np.real(np.diag(np.asarray(stage.map))))
        np.testing.assert_allclose(values, [0.0, 1.0, 2.0, 3.0], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
