# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Fixtures for the chain-level Hodge tests.

`svp_records` is the scaling verification plan's reporting format: one JSON
record per test instance, written to `records.jsonl` in a directory outside the
repository -- the session's pytest temporary directory by default, or the
directory named by the `TESSERA_SVP_RECORDS` environment variable. The path is
printed when the session's first recording test runs."""
import pytest

from tests.chainhodge._svp import Recorder


@pytest.fixture(scope="session")
def svp_records(tmp_path_factory):
    return Recorder(tmp_path_factory.mktemp("svp-records"))
