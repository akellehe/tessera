# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Every help string must render (#1015).

``run --help`` raised rather than printing:

    TypeError: %c requires int or char
      File "argparse.py", line 627, in _expand_help
        return self._get_help_string(action) % params

A help string built with a ``%`` format of its own has to escape a literal
percent for that pass -- and argparse then interpolates the result a SECOND
time against its own params, where ``% c`` reads as a conversion specifier with
a space flag and demands an int.

Nothing in the suite formatted help, so a broken help string was invisible to
every existing test while ``--help`` was completely unusable. These render it,
which is the only way to find this class of bug: the string is well-formed
Python and only fails when argparse expands it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402


def parser():
    """The driver's own parser, however it is spelled."""
    for name in ("build_parser", "make_parser", "parser", "_parser"):
        factory = getattr(ea, name, None)
        if callable(factory):
            return factory()
    pytest.skip("the driver exposes no parser factory to render")


def test_the_top_level_help_renders():
    assert parser().format_help()


def test_every_subparser_help_renders():
    """Each subcommand, so a broken string cannot hide behind an unused one."""
    import argparse

    root = parser()
    rendered = 0
    for action in root._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, sub in action.choices.items():
            assert sub.format_help(), name
            rendered += 1
    assert rendered > 0, "no subparsers found to render"


def test_every_individual_help_string_expands():
    """Expanded one at a time, so a failure names the flag that caused it.

    `format_help` on the whole parser reports only the first breakage; this
    reports which option's string is at fault, which is what a reader needs.
    """
    import argparse

    root = parser()
    formatter = root._get_formatter()
    checked = 0
    for parent in [root] + [
            sub for action in root._actions
            if isinstance(action, argparse._SubParsersAction)
            for sub in action.choices.values()]:
        for action in parent._actions:
            if not action.help:
                continue
            try:
                formatter._expand_help(action)
            except (TypeError, ValueError) as error:
                raise AssertionError(
                    "help for %s does not expand: %s"
                    % ("/".join(action.option_strings) or action.dest, error))
            checked += 1
    assert checked > 0


def test_the_candidate_moves_help_survives_both_interpolations():
    """The specific regression, named.

    A percent sign adjacent to a letter is what broke: the driver's own format
    turns `%%` into `%`, and argparse then reads `% c` as a conversion.
    """
    import argparse

    root = parser()
    formatter = root._get_formatter()
    for action in root._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for sub in action.choices.values():
            for candidate in sub._actions:
                if "--candidate-moves" in candidate.option_strings:
                    text = formatter._expand_help(candidate)
                    assert "fiftieth" in text or "%" not in text, text
                    return
    pytest.skip("--candidate-moves is not on this parser")
