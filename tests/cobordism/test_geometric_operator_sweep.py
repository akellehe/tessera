# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Acceptance tests for the geometric-operator realizability sweep."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(
    _HERE, "..", "..", "examples", "cobordism",
    "geometric_operator_sweep.py",
)
_SPEC = importlib.util.spec_from_file_location(
    "geometric_operator_sweep", _EXAMPLE)
SWEEP = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = SWEEP
_SPEC.loader.exec_module(SWEEP)


def _named_case(name, profile="smoke"):
    return next(
        case for case in SWEEP.build_cases(profile)
        if case["name"] == name
    )


_CONFIG = {
    "profile": "smoke",
    "generation_seed": 20260826,
    "epsilon": 1e-16,
    "boundary_epsilon": 1e-12,
    "neutral_budget": {
        "restarts": 4,
        "max_growth": 10,
        "max_iterations": 400,
    },
    "qutrit_budget": {
        "restarts": 8,
        "max_growth": 12,
        "max_iterations": 400,
    },
    "neutral_confirmation_budget": {
        "restarts": 8,
        "max_growth": 14,
        "max_iterations": 600,
    },
    "qutrit_confirmation_budget": {
        "restarts": 12,
        "max_growth": 16,
        "max_iterations": 600,
    },
    "confirmation_attempts": 0,
    "held_out_count": 4,
}


class CaseGenerationTest(unittest.TestCase):
    def test_profiles_are_deterministic_and_broad(self):
        expected_counts = {
            "smoke": 77,
            "standard": 144,
            "exhaustive": 376,
        }
        for profile, expected in expected_counts.items():
            first = SWEEP.build_cases(profile)
            second = SWEEP.build_cases(profile)
            self.assertEqual(len(first), expected)
            self.assertEqual(
                [case["name"] for case in first],
                [case["name"] for case in second],
            )
            np.testing.assert_array_equal(
                first[-1]["operator"], second[-1]["operator"])

        families = {
            case["family"] for case in SWEEP.build_cases("standard")}
        self.assertTrue({
            "unitary", "normal", "nonnormal", "singular", "scale",
            "condition", "input_basis", "charge_preserving",
            "charge_permuting", "charge_mixing",
        }.issubset(families))

    def test_all_relative_attachments_are_distinct_and_exhaustive(self):
        base = SWEEP.build_cases("smoke")
        expanded = SWEEP.expand_attachment_permutations(base)
        self.assertEqual(len(expanded), 6 * len(base))
        expected = {
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0),
        }
        self.assertEqual({
            case["attachment_permutation"] for case in expanded[:6]
        }, expected)

        signatures = set()
        for permutation in expected:
            fixture = SWEEP.BoundaryPairCobordism(
                attachment_permutation=permutation)
            signatures.add(tuple(sorted(
                tuple(sorted(int(vertex.getId())
                             for vertex in simplex.getVertices()))
                for simplex in fixture.spacetime.getTopSimplices()
            )))
        self.assertEqual(len(signatures), 6)

    def test_random_input_frames_are_reproducible_and_controlled(self):
        names = [
            "unitary_identity",
            "qutrit_charge_preserving_unitary",
            "qutrit_charge_swap",
            "rank_one_kernel_training_vector",
        ]
        cases = [_named_case(name) for name in names]
        first = SWEEP.randomize_input_bases(cases, 41)
        repeated = SWEEP.randomize_input_bases(cases, 41)
        distinct = SWEEP.randomize_input_bases(cases, 43)

        for left, right in zip(first, repeated):
            np.testing.assert_array_equal(
                left["input_basis"], right["input_basis"])
            self.assertEqual(left["input_seed"], right["input_seed"])
            self.assertTrue(left["input_basis_randomized"])
            self.assertEqual(
                np.linalg.matrix_rank(left["input_basis"]),
                left["operator"].shape[0],
            )
        self.assertTrue(any(
            not np.allclose(left["input_basis"], right["input_basis"])
            for left, right in zip(first, distinct)
        ))

        modes = {
            case["name"]: case["input_randomization"] for case in first}
        self.assertEqual(modes["unitary_identity"], "haar")
        self.assertEqual(
            modes["qutrit_charge_preserving_unitary"], "sector_haar")
        self.assertEqual(modes["qutrit_charge_swap"], "monomial")
        self.assertEqual(
            modes["rank_one_kernel_training_vector"], "specified_rays")

        for name in (
                "qutrit_charge_preserving_unitary",
                "qutrit_charge_swap"):
            case = next(case for case in first if case["name"] == name)
            self.assertTrue(
                SWEEP.operator_diagnostics(case)[
                    "boundary_compatible_prediction"])

        kernel_case = next(
            case for case in first
            if case["name"] == "rank_one_kernel_training_vector")
        output_norms = [
            np.linalg.norm(
                kernel_case["operator"] @
                kernel_case["input_basis"][:, column])
            for column in range(2)
        ]
        self.assertLess(min(output_norms), 1e-14)

        expanded = SWEEP.expand_attachment_permutations(first)
        for start in range(0, len(expanded), 6):
            for attached in expanded[start + 1:start + 6]:
                np.testing.assert_array_equal(
                    expanded[start]["input_basis"],
                    attached["input_basis"],
                )
                self.assertEqual(
                    expanded[start]["input_seed"],
                    attached["input_seed"],
                )

    def test_random_frames_preserve_boundary_classification(self):
        original = SWEEP.build_cases("exhaustive", 20260826)
        randomized = SWEEP.randomize_input_bases(original, 20260826)

        expected = {
            case["name"]: SWEEP.operator_diagnostics(case)[
                "boundary_compatible_prediction"]
            for case in original
        }
        actual = {
            case["name"]: SWEEP.operator_diagnostics(case)[
                "boundary_compatible_prediction"]
            for case in randomized
        }
        self.assertEqual(actual, expected)
        for case in randomized:
            prepared = SWEEP._effective_input_basis(
                case["input_basis"])
            self.assertEqual(
                np.linalg.matrix_rank(prepared),
                case["operator"].shape[0],
            )

    def test_shuffle_is_deterministic_diverse_and_exhaustive(self):
        expanded = SWEEP.expand_attachment_permutations(
            SWEEP.build_cases("smoke"))
        first = SWEEP.shuffle_cases(expanded, 47)
        repeated = SWEEP.shuffle_cases(expanded, 47)
        distinct = SWEEP.shuffle_cases(expanded, 53)

        def names(cases):
            return [case["name"] for case in cases]

        self.assertEqual(names(first), names(repeated))
        self.assertNotEqual(names(first), names(distinct))
        self.assertNotEqual(names(first), names(expanded))
        self.assertEqual(set(names(first)), set(names(expanded)))
        self.assertEqual(len(names(first)), len(set(names(first))))

        prefix = first[:12]
        self.assertGreater(len({
            case["name"].rsplit("__attachment_", 1)[0]
            for case in prefix
        }), 1)
        self.assertGreater(len({
            case["attachment_permutation"] for case in prefix
        }), 1)

    def test_diagnostics_distinguish_charge_from_boundary_compatibility(self):
        preserving = SWEEP.operator_diagnostics(_named_case(
            "qutrit_charge_preserving_identity"))
        permuting = SWEEP.operator_diagnostics(_named_case(
            "qutrit_charge_swap"))
        mixing = SWEEP.operator_diagnostics(_named_case(
            "qutrit_charge_mixing_rotation"))

        self.assertEqual(preserving["charge_commutator_error"], 0.0)
        self.assertTrue(preserving["boundary_compatible_prediction"])
        self.assertGreater(permuting["charge_commutator_error"], 1.0)
        self.assertTrue(permuting["boundary_compatible_prediction"])
        self.assertGreater(mixing["charge_commutator_error"], 0.1)
        self.assertFalse(mixing["boundary_compatible_prediction"])


class SweepClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = SWEEP.evaluate_case(
            _named_case("unitary_identity"), _CONFIG)
        randomized_identity = SWEEP.randomize_input_bases(
            [_named_case("unitary_identity")], 41)[0]
        cls.randomized_identity = SWEEP.evaluate_case(
            randomized_identity, _CONFIG)
        cls.rank_one = SWEEP.evaluate_case(
            _named_case("rank_one_dense"), _CONFIG)
        cls.column_rescaling = SWEEP.evaluate_case(
            _named_case("basis_column_rescaling"), _CONFIG)
        cls.zero = SWEEP.evaluate_case(
            _named_case("zero_operator"), _CONFIG)
        cls.charge_mixing = SWEEP.evaluate_case(
            _named_case("qutrit_charge_mixing_rotation"), _CONFIG)

    def test_unitary_and_dense_singular_maps_are_verified(self):
        for record in (
                self.identity, self.randomized_identity, self.rank_one):
            self.assertEqual(record["status"], "verified")
            self.assertLess(record["fit"]["residual"], 1e-16)
            self.assertEqual(record["fit"]["boundary_drift"], 0.0)
            self.assertEqual(record["fit"]["restriction_error"], 0.0)
            self.assertLess(
                record["fit"]["operator_relative_error"], 1e-12)
            self.assertLess(
                record["fit"]["held_out_full_residual_max"], 1e-12)

    def test_random_training_frame_is_recorded(self):
        record = self.randomized_identity
        self.assertTrue(record["input_basis_randomized"])
        self.assertEqual(record["input_randomization"], "haar")
        self.assertIsInstance(record["input_seed"], int)
        self.assertEqual(
            record["input_basis"], record["fit"]["input_basis"])

    def test_all_reattachments_have_explicit_chart_action(self):
        diagnostics = (
            self.identity["fit"]["input_gluing_permutations"])
        self.assertEqual(len(diagnostics), 6)
        self.assertEqual(
            sum(item["parity"] == "even" for item in diagnostics),
            3,
        )
        nonidentity = [
            item for item in diagnostics
            if item["permutation"] != [0, 1, 2]
        ]
        self.assertEqual(len(nonidentity), 5)
        self.assertGreater(
            max(item["naive_operator_error"] for item in nonidentity),
            1.0,
        )
        for diagnostic in diagnostics:
            self.assertLess(diagnostic["composed_operator_error"], 1e-12)
            self.assertLess(diagnostic["chart_corrected_error"], 1e-12)
        self.assertTrue(all(
            not diagnostic["input_only_combinatorial_automorphism"]
            for diagnostic in nonidentity
        ))

    def test_input_column_scaling_is_normalized_before_rank_check(self):
        self.assertEqual(self.column_rescaling["status"], "verified")
        self.assertLess(
            self.column_rescaling["fit"]["prepared_input_condition"],
            1.0 + 1e-12,
        )

    def test_zero_output_is_a_structural_rejection(self):
        self.assertEqual(self.zero["status"], "rejected")
        self.assertIn(
            "output states must be nonzero", self.zero["reason"])

    def test_charge_mixing_is_rejected_by_boundary_preparation(self):
        self.assertEqual(self.charge_mixing["status"], "rejected")
        self.assertIn(
            "output state is not an isolated-boundary eigenstate",
            self.charge_mixing["reason"],
        )

    def test_summary_is_strict_json(self):
        summary = SWEEP.summarize([
            self.identity, self.rank_one, self.zero, self.charge_mixing])
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["status_counts"], {
            "rejected": 2,
            "verified": 2,
        })
        self.assertEqual(summary["boundary_invariant_violations"], [])
        json.dumps(summary, allow_nan=False)


class CheckpointTest(unittest.TestCase):
    def test_all_attachment_results_resume_from_atomic_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "sweep.json")
            options = {
                "profile": "smoke",
                "name_contains": "zero_operator",
                "all_attachments": True,
                "randomize_inputs": True,
                "input_seed": 41,
                "shuffle_seed": 47,
                "held_out_count": 0,
                "checkpoint_path": checkpoint,
            }
            first = SWEEP.run_sweep(**options)
            resumed = SWEEP.run_sweep(**options, resume=True)
        self.assertTrue(first["complete"])
        self.assertEqual(first["expected_case_count"], 6)
        self.assertEqual(first["summary"]["case_count"], 6)
        self.assertEqual(first["summary"]["status_counts"], {"rejected": 6})
        self.assertEqual(first["cases"], resumed["cases"])
        self.assertTrue(all(
            case["input_basis_randomized"] for case in first["cases"]))
        self.assertEqual(
            {case["schedule_index"] for case in first["cases"]},
            set(range(6)),
        )
        self.assertEqual(
            {case["completion_index"] for case in first["cases"]},
            set(range(1, 7)),
        )


class CommandLineTest(unittest.TestCase):
    def test_profiles_filters_and_qutrit_budget_are_exposed(self):
        args = SWEEP.build_parser().parse_args([
            "--profile", "exhaustive",
            "--family", "unitary",
            "--name-contains", "identity",
            "--jobs", "3",
            "--qutrit-restarts", "9",
            "--checkpoint-every", "7",
            "--all-attachments",
            "--random-inputs",
            "--input-seed", "41",
            "--shuffle-seed", "47",
            "--resume",
            "--no-write",
        ])
        self.assertEqual(args.profile, "exhaustive")
        self.assertEqual(args.families, ["unitary"])
        self.assertEqual(args.name_contains, "identity")
        self.assertEqual(args.jobs, 3)
        self.assertEqual(args.qutrit_restarts, 9)
        self.assertEqual(args.checkpoint_every, 7)
        self.assertTrue(args.all_attachments)
        self.assertTrue(args.random_inputs)
        self.assertEqual(args.input_seed, 41)
        self.assertEqual(args.shuffle_seed, 47)
        self.assertTrue(args.resume)
        self.assertTrue(args.no_write)


if __name__ == "__main__":
    unittest.main()
