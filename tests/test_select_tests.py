# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The pull-request test selector picks the tests a change can affect.

``.github/scripts/select_tests.py`` maps the paths a pull request changes to
the pytest paths to run. These tests exercise each of its rules on a small
synthetic repository with four bound C++ areas (``alpha``, ``beta``,
``gamma``, ``delta``), two of them star-exported at the top level (``alpha``,
``gamma``), and a handful of test modules that reach the library in every way
the scanner resolves: a submodule import, an attribute on the package, a
top-level name, a Python package, a helper imported as a sibling or loaded
by path, a dynamic import.
The last tests run the selector on this repository itself. Nothing here
imports ``tessera``.
"""
import importlib.util
import io
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / ".github" / "scripts" / "select_tests.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location("select_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = _load_selector()

# Every file of the synthetic repository; directories are created on demand.
SYNTHETIC_FILES = {
    "CMakeLists.txt": "project(synthetic)\n",
    "pyproject.toml": "[project]\nname = 'synthetic'\n",
    "conftest.py": "",
    "README.md": "# synthetic\n",
    "docs/source/cpp_api.md": "{doxygenfile} Alpha.h\n",
    "examples/demo.py": "print('demo')\n",
    # C++: alpha is the base of everything, Shared.h is used by beta only,
    # gamma uses the root Logger, delta is a leaf, omega has no binding.
    "include/Logger.h": "#pragma once\n",
    "include/tessera_config.hpp.in": "#define TESSERA_VERSION \"@VERSION@\"\n",
    "include/alpha/Alpha.h": "#pragma once\n",
    "include/alpha/Shared.h": "#pragma once\n",
    "include/beta/Beta.h": '#pragma once\n#include "alpha/Shared.h"\n',
    "include/gamma/Gamma.h": "#pragma once\n",
    "include/delta/Delta.h": "#pragma once\n",
    "include/omega/Omega.h": "#pragma once\n",
    "src/Logger.cpp": '#include "Logger.h"\n',
    "src/bindings.cpp": '#include <pybind11/pybind11.h>\n',
    "src/alpha/Alpha.cpp": '#include "alpha/Alpha.h"\n',
    "src/alpha/Shared.cpp": '#include "alpha/Shared.h"\n',
    "src/alpha/Bindings.cpp": '#include "alpha/Alpha.h"\n#include "alpha/Shared.h"\n',
    "src/beta/Beta.cpp": '#include "beta/Beta.h"\n#include "alpha/Alpha.h"\n',
    "src/beta/Bindings.cpp": '#include "beta/Beta.h"\n',
    "src/gamma/Gamma.cpp": '#include "gamma/Gamma.h"\n#include "alpha/Alpha.h"\n#include "Logger.h"\n',
    "src/gamma/Bindings.cpp": '#include "gamma/Gamma.h"\n',
    "src/delta/Delta.cpp": '#include "delta/Delta.h"\n#include "alpha/Alpha.h"\n',
    "src/delta/Bindings.cpp": '#include "delta/Delta.h"\n',
    # Python package: alpha and gamma are star-exported at the top level.
    "tessera/__init__.py": (
        "from tessera._tessera import *\n"
        "from tessera._tessera import (alpha, beta, gamma, delta)\n"
        "from tessera._tessera.alpha import *\n"
        "from tessera._tessera.gamma import *\n"
    ),
    "tessera/drivers/__init__.py": "",
    "tessera/drivers/drive.py": "from tessera import beta\n\ndef run():\n    return beta.X()\n",
    "tessera/utils/__init__.py": "",
    "tessera/utils/tool.py": "def tool():\n    return 1\n",
    # Tests.
    "tests/__init__.py": "",
    "tests/alpha/__init__.py": "",
    "tests/alpha/test_alpha.py": "from tessera.alpha import Thing\n\ndef test_thing():\n    assert Thing\n",
    "tests/beta/__init__.py": "",
    "tests/beta/test_beta.py": "import tessera as T\n\ndef test_x():\n    assert T.beta.X\n",
    "tests/beta/_helper.py": "from tessera import beta\n\ndef make():\n    return beta.X()\n",
    "tests/beta/test_helper_user.py": "from tests.beta._helper import make\n\ndef test_make():\n    assert make\n",
    "tests/beta/test_sibling.py": (
        "import os\nimport sys\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "from _helper import make\n"
        "def test_sibling():\n    assert make\n"
    ),
    "tests/beta/test_by_path.py": (
        "import importlib.util\nimport os\n"
        "_H = os.path.join(os.path.dirname(__file__), '_helper.py')\n"
        "spec = importlib.util.spec_from_file_location('_helper', _H)\n"
        "def test_spec():\n    assert spec\n"
    ),
    "tests/beta/test_dump.py": (
        "import json\nimport os\n"
        "DUMP = os.path.join(os.path.dirname(__file__), 'data', 'dump.json')\n"
        "def test_dump():\n    assert os.path.exists(DUMP)\n"
    ),
    "tests/beta/data/dump.json": "{}\n",
    "tests/drivers/__init__.py": "",
    "tests/drivers/test_drive.py": "from tessera.drivers import drive\n\ndef test_run():\n    assert drive.run\n",
    "tests/test_top.py": "from tessera import Thing\n\ndef test_top():\n    assert Thing\n",
    "tests/test_gamma_attr.py": "import tessera\n\ndef test_y():\n    assert tessera.gamma.Y\n",
    "tests/test_dynamic.py": (
        "import importlib\nimport tessera\n"
        "def test_all(name):\n    assert importlib.import_module(f'tessera.{name}')\n"
    ),
    "tests/test_prefix.py": (
        "import importlib\n"
        "def test_driver(name):\n    assert importlib.import_module('tessera.drivers.' + name)\n"
    ),
    "tests/test_tool.py": "from tessera.utils import tool\n\ndef test_tool():\n    assert tool.tool() == 1\n",
    "tests/test_examples.py": (
        "import os\nEXAMPLES = os.path.join(os.path.dirname(__file__), '..', 'examples')\n"
        "def test_examples():\n    assert EXAMPLES\n"
    ),
    "tests/test_cpp_api_docs_coverage.py": "def test_page():\n    assert True\n",
    "tests/test_delta_only.py": "from tessera import delta as d\n\ndef test_d():\n    assert d.Z\n",
    "tests/test_fixture_user.py": (
        "import os\nSPECIMENS = os.path.join(os.path.dirname(__file__), 'fixtures', 'specimens')\n"
        "def test_specimens():\n    assert SPECIMENS\n"
    ),
    "tests/fixtures/specimens/one.json": "{}\n",
    "tests/fixtures/other/two.json": "{}\n",
}

#: The synthetic repository's test files (sixteen of them).
SYNTHETIC_TESTS = sorted(
    p for p in SYNTHETIC_FILES
    if p.startswith("tests/") and os.path.basename(p).startswith("test_") and p.endswith(".py"))

#: What every C++ change selects besides the tests reaching the areas.
BASELINE = set(selector.CPP_BASELINE)

#: The test that reaches everything is in every selection that is not full.
DYNAMIC = "tests/test_dynamic.py"


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    for rel, content in SYNTHETIC_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def tree(synthetic_root):
    return selector.Tree(str(synthetic_root))


def run(tree, *changed):
    selection = selector.select(tree, list(changed))
    return selection, selector.pytest_paths(selection)


# ---------------------------------------------------------------------------
# The scan of the synthetic tree.


def test_tree_scan(tree):
    assert tree.areas == {"alpha", "beta", "gamma", "delta", "omega"}
    assert tree.bound_areas == {"alpha", "beta", "gamma", "delta"}
    assert tree.star_exported_areas == {"alpha", "gamma"}
    assert len(tree.test_files) == len(SYNTHETIC_TESTS) == 16
    assert tree.includes["include/beta/Beta.h"] == {"include/alpha/Shared.h"}
    assert tree.included_by["include/Logger.h"] == {"src/Logger.cpp", "src/gamma/Gamma.cpp"}


def test_dependents_follow_includes_transitively(tree):
    assert tree.dependents("include/alpha/Shared.h") == {
        "include/beta/Beta.h", "src/beta/Beta.cpp", "src/beta/Bindings.cpp",
        "src/alpha/Shared.cpp", "src/alpha/Bindings.cpp"}


def test_import_graph_edges(tree):
    reverse = tree.reverse
    assert "tests.beta.test_beta" in reverse["cpp:beta"]
    assert "tests.alpha.test_alpha" in reverse["cpp:alpha"]
    assert "tests.test_gamma_attr" in reverse["cpp:gamma"]
    assert "tests.test_delta_only" in reverse["cpp:delta"]
    assert "tests.test_top" in reverse[selector.TOP_LEVEL_NAMES]
    assert "tests.test_dynamic" in reverse[selector.EVERYTHING]
    assert "tests.test_prefix" in reverse["tessera.drivers.drive"]
    assert "tests.test_prefix" not in reverse[selector.EVERYTHING]
    assert "tessera.drivers.drive" in reverse["cpp:beta"]
    assert "tests.beta.test_helper_user" in reverse["tests.beta._helper"]
    assert "tests.beta.test_by_path" in reverse["tests.beta._helper"]
    assert "tests.beta.test_sibling" in reverse["tests.beta._helper"]


# ---------------------------------------------------------------------------
# Rule 1: infrastructure selects the full fast tier.


@pytest.mark.parametrize("changed", [
    "CMakeLists.txt", "cmake/toolchain.cmake", "pyproject.toml", "conftest.py",
    "tests/conftest.py", "tests/beta/conftest.py", ".github/workflows/build.yml",
    ".github/scripts/select_tests.py", "third_party/lib/x.h", "src/bindings.cpp",
    "tessera/__init__.py", "tests/__init__.py",
])
def test_infrastructure_selects_everything(tree, changed):
    selection, paths = run(tree, changed)
    assert paths == [selector.FULL_TIER]
    assert "build or test infrastructure" in selection.full_reason


def test_unknown_path_selects_everything(tree):
    selection, paths = run(tree, "Makefile")
    assert paths == [selector.FULL_TIER]
    assert "no known mapping" in selection.full_reason


def test_no_changed_paths_selects_everything(tree):
    selection, paths = run(tree)
    assert paths == [selector.FULL_TIER]
    assert "empty or failed diff" in selection.full_reason


def test_one_full_path_makes_the_union_full(tree):
    _, paths = run(tree, "README.md", "CMakeLists.txt")
    assert paths == [selector.FULL_TIER]


# ---------------------------------------------------------------------------
# Rule 2: documentation.


@pytest.mark.parametrize("changed", ["docs/source/cpp_api.md", "README.md", "docs/Doxyfile"])
def test_documentation_selects_the_coverage_test(tree, changed):
    _, paths = run(tree, changed)
    assert paths == [selector.DOCS_COVERAGE_TEST]


# ---------------------------------------------------------------------------
# Rule 3: C++.


def test_cpp_source_selects_its_area_and_the_tests_reaching_it(tree):
    _, paths = run(tree, "src/beta/Beta.cpp")
    assert set(paths) == BASELINE | {"tests/beta/", DYNAMIC, "tests/test_prefix.py"}


def test_cpp_header_closure_crosses_areas(tree):
    # Shared.h is alpha's, but beta includes it; alpha is star-exported, so
    # the tests using top-level names are reached as well.
    selection, paths = run(tree, "include/alpha/Shared.h")
    assert not selection.is_full
    assert set(paths) == BASELINE | {
        "tests/alpha/", "tests/beta/", "tests/test_top.py", DYNAMIC, "tests/test_prefix.py"}


@pytest.mark.parametrize("changed", ["include/alpha/Alpha.h", "src/alpha/Alpha.cpp"])
def test_cpp_closure_over_most_areas_selects_everything(tree, changed):
    selection, paths = run(tree, changed)
    assert paths == [selector.FULL_TIER]
    assert "include closure covers 4 of 4 bound areas" in selection.full_reason


def test_bindings_translation_unit_affects_only_its_area(tree):
    _, paths = run(tree, "src/delta/Bindings.cpp")
    assert set(paths) == BASELINE | {"tests/test_delta_only.py", DYNAMIC}


def test_root_area_reaches_the_top_level_names(tree):
    _, paths = run(tree, "src/Logger.cpp")
    assert set(paths) == BASELINE | {"tests/test_gamma_attr.py", "tests/test_top.py", DYNAMIC}


def test_cpp_without_binding_selects_everything(tree):
    selection, paths = run(tree, "include/omega/Omega.h")
    assert paths == [selector.FULL_TIER]
    assert "no Python binding" in selection.full_reason


def test_non_cpp_under_include_selects_everything(tree):
    selection, paths = run(tree, "include/tessera_config.hpp.in")
    assert paths == [selector.FULL_TIER]
    assert "not a C++ source or header" in selection.full_reason


def test_deleted_cpp_source_maps_to_its_area(tree):
    _, paths = run(tree, "src/delta/Gone.cpp")
    assert set(paths) == BASELINE | {"tests/test_delta_only.py", DYNAMIC}


# ---------------------------------------------------------------------------
# Rule 4: Python under tessera/.


def test_python_module_selects_the_tests_reaching_it(tree):
    _, paths = run(tree, "tessera/drivers/drive.py")
    assert set(paths) == {
        "tests/drivers/test_drive.py", "tests/test_prefix.py", DYNAMIC, selector.EXAMPLES_TEST}


def test_python_package_init_is_reached_by_its_modules_importers(tree):
    _, paths = run(tree, "tessera/drivers/__init__.py")
    assert set(paths) == {
        "tests/drivers/test_drive.py", "tests/test_prefix.py", DYNAMIC, selector.EXAMPLES_TEST}


def test_python_module_outside_drivers_does_not_add_the_examples_test(tree):
    _, paths = run(tree, "tessera/utils/tool.py")
    assert set(paths) == {"tests/test_tool.py", DYNAMIC}


def test_non_python_under_tessera_selects_everything(tree):
    selection, paths = run(tree, "tessera/drivers/table.npz")
    assert paths == [selector.FULL_TIER]
    assert "not a Python module" in selection.full_reason


# ---------------------------------------------------------------------------
# Rule 5: examples.


def test_example_selects_the_examples_test_and_its_mentioners(tree):
    _, paths = run(tree, "examples/demo.py")
    assert set(paths) == {selector.EXAMPLES_TEST, DYNAMIC}


# ---------------------------------------------------------------------------
# Rule 6: tests.


def test_test_module_selects_itself(tree):
    _, paths = run(tree, "tests/alpha/test_alpha.py")
    assert set(paths) == {"tests/alpha/test_alpha.py", DYNAMIC}


def test_helper_selects_its_importers_and_mentioners(tree):
    _, paths = run(tree, "tests/beta/_helper.py")
    assert set(paths) == {
        "tests/beta/test_helper_user.py", "tests/beta/test_by_path.py",
        "tests/beta/test_sibling.py", DYNAMIC}


def test_test_package_init_selects_its_directory(tree):
    _, paths = run(tree, "tests/beta/__init__.py")
    assert paths == ["tests/beta/"]


def test_test_data_selects_its_directory_and_mentioners(tree):
    selection, paths = run(tree, "tests/beta/data/dump.json")
    assert "tests/beta/test_dump.py" in selection.paths
    assert set(paths) == {"tests/beta/", DYNAMIC}


def test_fixture_selects_the_tests_mentioning_it(tree):
    _, paths = run(tree, "tests/fixtures/specimens/one.json")
    assert set(paths) == {"tests/test_fixture_user.py", DYNAMIC}


def test_unmentioned_fixture_selects_everything(tree):
    selection, paths = run(tree, "tests/fixtures/other/two.json")
    assert paths == [selector.FULL_TIER]
    assert "no test mentions" in selection.full_reason


def test_file_directly_under_tests_selects_everything(tree):
    selection, paths = run(tree, "tests/notes.txt")
    assert paths == [selector.FULL_TIER]
    assert "directly under tests/" in selection.full_reason


# ---------------------------------------------------------------------------
# The closing guards and the output.


def test_union_of_several_changes(tree):
    _, paths = run(tree, "README.md", "tessera/utils/tool.py", "src/delta/Bindings.cpp")
    assert set(paths) == BASELINE | {"tests/test_tool.py", "tests/test_delta_only.py", DYNAMIC}


def test_selection_covering_most_test_files_becomes_the_whole_suite(tree):
    selection, paths = run(tree, *SYNTHETIC_TESTS)
    assert paths == [selector.FULL_TIER]
    assert "covers 16 of 16 test files" in selection.full_reason


def test_directories_absorb_the_files_under_them(tree):
    selection = selector.Selection()
    selection.add("x", "rule", {"tests/beta/", "tests/beta/test_beta.py", "tests/test_top.py"})
    assert selector.pytest_paths(selection) == ["tests/beta/", "tests/test_top.py"]


def test_every_decision_is_explained(tree):
    selection, paths = run(tree, "tessera/utils/tool.py")
    log = io.StringIO()
    selector._report(tree, selection, paths, log)
    text = log.getvalue()
    assert "tessera/utils/tool.py: Python module tessera.utils.tool" in text
    assert "selected 2 of 16 test files" in text


def test_main_reads_a_file_and_writes_the_selection(synthetic_root, tmp_path, capsys):
    changed = tmp_path / "changed.txt"
    changed.write_text("README.md\n\ntessera/utils/tool.py\n", encoding="utf-8")
    output = tmp_path / "selected.txt"
    status = selector.main([
        "--root", str(synthetic_root), "--changed-paths", str(changed), "--output", str(output)])
    assert status == 0
    assert output.read_text(encoding="utf-8").split() == sorted(
        [selector.DOCS_COVERAGE_TEST, DYNAMIC, "tests/test_tool.py"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "select_tests: selected 3 of 16 test files" in captured.err


def test_main_prints_to_stdout_by_default(synthetic_root, capsys):
    assert selector.main(["--root", str(synthetic_root), "CMakeLists.txt"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "tests/\n"
    assert "running the full fast tier" in captured.err


def test_main_falls_back_to_everything_when_the_selector_fails(synthetic_root, capsys, monkeypatch):
    def explode(*_):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(selector, "select", explode)
    assert selector.main(["--root", str(synthetic_root), "README.md"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "tests/\n"
    assert "synthetic failure" in captured.err
    assert "the selector failed" in captured.err


# ---------------------------------------------------------------------------
# The real repository.


@pytest.fixture(scope="module")
def real_tree():
    if not (REPO_ROOT / "include").is_dir() or not (REPO_ROOT / "tessera").is_dir():
        pytest.skip("not run from the repository")
    return selector.Tree(str(REPO_ROOT))


def test_real_tree_areas(real_tree):
    assert {"mesh", "spacetime", "observables", "simulations", "cobordism", "chainhodge", "quantum"} \
        <= real_tree.bound_areas
    assert real_tree.star_exported_areas == {"mesh", "spacetime", "observables", "simulations"}
    assert real_tree.notes == []


def test_real_tree_docs_only_change(real_tree):
    _, paths = run(real_tree, "docs/source/theory.md", "README.md")
    assert paths == [selector.DOCS_COVERAGE_TEST]


def test_real_tree_this_pull_request_runs_everything(real_tree):
    selection, paths = run(
        real_tree, ".github/workflows/build.yml", ".github/scripts/select_tests.py",
        "tests/test_select_tests.py")
    assert paths == [selector.FULL_TIER]
    assert "build or test infrastructure" in selection.full_reason


def test_real_tree_drivers_change_is_a_strict_subset(real_tree):
    selection, paths = run(real_tree, "tessera/drivers/baryon_poles.py")
    assert not selection.is_full
    assert selector.EXAMPLES_TEST in paths
    assert any(p.startswith("tests/drivers/") for p in paths)
    assert not any(p.startswith("tests/mesh/") for p in paths)
    assert all((REPO_ROOT / p).exists() for p in paths)


def test_real_tree_mesh_change_runs_everything(real_tree):
    selection, paths = run(real_tree, "src/mesh/Vertex.cpp")
    assert paths == [selector.FULL_TIER]
    assert "include closure covers" in selection.full_reason


def test_real_tree_selected_paths_exist(real_tree):
    for changed in ("src/quantum/Bindings.cpp", "tessera/utils/progress.py", "examples/wilson_loops.py"):
        selection, paths = run(real_tree, changed)
        assert not selection.is_full, changed
        assert all((REPO_ROOT / p).exists() for p in paths), changed
