# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The command-line value parsers and refusals of the qubit cobordism driver
(`tessera.drivers.qubit`) not reached by its other tests.

Terms: a *modulus* tau is a complex number in the upper half plane, the shape
of a torus; a *state coordinate* is a finite complex number; the *verify* and
*theta* subcommands read the seeded host without running a stage, from the
host seed `_verify_config` extracts from their arguments.
"""
import argparse
import math

import pytest

from tessera.drivers import qubit as qa


@pytest.mark.parametrize("value,expected", [
    (0.5, 0.5 + 0j),
    ("0.3 + 1.1j", 0.3 + 1.1j),
    ([0.2, 0.8], 0.2 + 0.8j),
    ((1, -1), 1 - 1j),
])
def test_a_complex_is_read_from_a_number_a_string_or_a_pair(value, expected):
    assert qa._as_complex(value) == expected


def test_a_pair_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match=r"\[re, im\] pair"):
        qa._as_complex([1.0, 2.0, 3.0])


@pytest.mark.parametrize("value", [0.3 - 1.1j, 1.0, [0.0, 0.0],
                                   complex(math.inf, 1.0),
                                   complex(0.0, math.nan)])
def test_a_modulus_outside_the_upper_half_plane_is_refused(value):
    with pytest.raises(ValueError, match="upper half plane"):
        qa._modulus_value("tau_a", value)


def test_an_unreadable_modulus_is_refused_by_name():
    with pytest.raises(ValueError, match="tau_b is a modulus tau"):
        qa._modulus_value("tau_b", "north")
    assert qa._modulus_value("tau_a", "0.3+1.1j") == 0.3 + 1.1j


def test_a_state_coordinate_must_be_finite():
    assert qa._state_value("--output-state", "-0.4+0.95j") == -0.4 + 0.95j
    # the lower half plane is a valid state coordinate, unlike a modulus
    assert qa._state_value("--output-state", 0.1 - 2j) == 0.1 - 2j
    with pytest.raises(ValueError, match="must be finite"):
        qa._state_value("--output-state", complex(math.inf, 0.0))
    with pytest.raises(ValueError, match="complex state coordinate"):
        qa._state_value("--output-state", "up")


def test_the_modulus_slug_is_readable():
    assert qa._tau_slug(0.3 + 1.1j) == "0.3+1.1j"
    assert qa._tau_slug(-0.2 + 0.8j) == "-0.2+0.8j"
    assert qa._tau_slug(1j) == "0+1j"


def test_the_command_line_complex_parser_names_the_bad_value():
    with pytest.raises(argparse.ArgumentTypeError, match="'1.1i'"):
        qa._complex_argument("1.1i")


def test_a_lower_half_plane_modulus_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as stop:
        qa.main(["run", "--tau-a=0.3-1.1j", "--steps", "0", "--out", "",
                 "--quiet"])
    assert stop.value.code == 2
    assert "upper half plane" in capsys.readouterr().err


def test_an_unsupported_output_extension_is_a_usage_error(capsys, tmp_path):
    with pytest.raises(SystemExit) as stop:
        qa.main(["run", "--steps", "0", "--out", str(tmp_path / "frame.jpg"),
                 "--quiet"])
    assert stop.value.code == 2
    assert ".gif, .mp4, or .png" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["verify", "theta"])
def test_four_tori_need_at_least_three_layers(command):
    args = qa.build_parser().parse_args([command, "--tori", "4",
                                         "--layers", "1"])
    config = qa._verify_config(args)
    assert config["tori"] == 4 and config["layers"] == 3
    two = qa._verify_config(qa.build_parser().parse_args(
        [command, "--tori", "2", "--layers", "1"]))
    assert two["layers"] == 1


def test_a_tube_join_needs_four_tori():
    args = qa.build_parser().parse_args(["theta", "--join", "tube",
                                         "--tori", "2"])
    with pytest.raises(ValueError, match="--join tube needs --tori 4"):
        qa._verify_config(args)


def test_a_tube_join_keeps_its_declared_layers_and_shape():
    args = qa.build_parser().parse_args(
        ["theta", "--join", "tube", "--tori", "4", "--layers", "1",
         "--tube-layers", "2", "--tube-length", "1.5", "--tube-waist", "0.4"])
    config = qa._verify_config(args)
    assert config["layers"] == 1
    assert config["tube_layers"] == 2
    assert config["tube_length"] == 1.5 and config["tube_waist"] == 0.4


def test_the_verify_seed_records_the_moduli_as_pairs():
    args = qa.build_parser().parse_args(["verify", "--tau-a", "0.1+1.2j",
                                         "--seed", "7", "--grid", "3"])
    config = qa._verify_config(args)
    assert config["tau_a"] == [0.1, 1.2]
    assert config["seed"] == 7 and config["grid"] == 3
    assert config["interior_disposition"] == qa.DECLARED_INTERIOR_DISPOSITION
