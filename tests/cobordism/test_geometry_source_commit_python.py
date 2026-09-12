# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""A geometry dump records the commit that produced it (#1040).

The dump is the only output of a run that cannot be recomputed from the others,
and it carried no record of the code that wrote it. Reproducing a figure meant
guessing which commit had been checked out.

`dirty` carries as much as the sha. A sha read from a tree with uncommitted
changes names a commit that does not describe what ran, so a record that gave
the sha alone would be worse than one that said nothing.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

SMALL = 4


@pytest.fixture(scope="module")
def source():
    return ea.source_commit()


def test_it_names_the_head_this_file_is_checked_out_at(source):
    """Read from the repository, never from a recorded constant."""
    assert not isinstance(source, ea.Absent), source
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=os.path.dirname(os.path.abspath(ea.__file__)),
        capture_output=True, text=True, check=True).stdout.strip()
    assert source["head"] == head
    assert len(source["head"]) == 40


def test_it_says_whether_the_tree_was_dirty(source):
    assert isinstance(source["dirty"], bool)
    porcelain = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=os.path.dirname(os.path.abspath(ea.__file__)),
        capture_output=True, text=True, check=True).stdout.strip()
    assert source["dirty"] is bool(porcelain)


def test_it_names_the_branch(source):
    assert isinstance(source["branch"], str) and source["branch"]


def test_it_is_absent_rather_than_guessed_outside_a_work_tree(monkeypatch):
    """No git, no answer -- not a fabricated one."""
    def refuse(*_args, **_kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(subprocess, "run", refuse)
    absent = ea.source_commit()
    assert isinstance(absent, ea.Absent)
    assert "git" in absent.reason


def test_the_geometry_document_carries_it(source):
    config = ea.build_config(size=SMALL, steps=1, stage2_iters=1)
    result = ea.drive(config, progress=False)
    held = {}
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    document = ea.geometry_document(held["node"], source=source)
    assert document["schema"] == 1, "the addition is a new key, not a new schema"
    assert document["source"] == source
    assert result.frames
