# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""`--live` needs a GUI toolkit, and the project has to say so (#1028).

`--live` opens a window. matplotlib on its own resolves to Agg, which renders
to files and shows nothing, so the flag refuses to start rather than compute
every frame and display none of them:

    RuntimeError: --live needs an interactive matplotlib backend; this process
    has 'agg', which renders to files and shows no window.

The refusal is right and the flag was still unusable from a clean install,
because nothing in the project declared a toolkit binding. A `live` extra
supplies one, `dev` picks it up, and both the error and the `--live` help name
the extra so whoever hits it is told what to install.

It stays its OWN extra rather than joining `examples`: every campaign on a
headless box renders to `--out` and has no use for Qt.
"""
import os
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO / "examples" / "cobordism"))

import emergence_animation as ea  # noqa: E402


@pytest.fixture(scope="module")
def project():
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]


def test_the_live_extra_declares_a_gui_binding(project):
    extra = project["optional-dependencies"]["live"]
    assert any(name.lower().startswith("pyqt6") for name in extra), extra


def test_dev_pulls_the_live_extra_in(project):
    """A developer running the example is the common case."""
    assert any("live" in name for name in
               project["optional-dependencies"]["dev"])


def test_it_is_not_a_core_dependency(project):
    """A headless campaign must not download Qt to import tessera."""
    assert not any("pyqt" in name.lower() for name in project["dependencies"])
    assert not any("pyqt" in name.lower()
                   for name in project["optional-dependencies"]["examples"])


def test_the_refusal_names_the_extra(monkeypatch):
    """Whoever hits it is told what to install, not left to work it out.

    Driven through `drive_live` itself rather than a helper, so the message
    under test is the one a caller actually gets.
    """
    import matplotlib
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    with pytest.raises(RuntimeError) as caught:
        ea.drive_live(ea.build_config(steps=1, size=4), progress=False)
    message = str(caught.value)
    assert "[live]" in message, message
    assert "agg" in message


def test_the_flag_help_names_the_extra():
    parser = ea.build_parser()
    live = [action for action in parser._subparsers._group_actions[0]
            .choices["run"]._actions if "--live" in action.option_strings]
    assert live and "[live]" in live[0].help
