#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Select the pytest paths a change can affect.

Usage::

    python3 .github/scripts/select_tests.py [--root DIR] [--output FILE]
                                            [--changed-paths FILE] [PATH ...]

The changed paths are given relative to the repository root, as
``git diff --name-only`` prints them: as positional arguments, or one per line
in the ``--changed-paths`` file (``-`` reads standard input). The selected
pytest paths are written one per line to standard output, or to ``--output``.
Every decision is explained on standard error, so a CI job log shows what was
selected and why. The exit status is always zero: when the selector cannot
decide, or fails internally, it selects the full fast tier rather than
failing the job.

The selection is a pure function of the working tree and the changed paths.
Nothing is imported from the repository: the Python import graph is read from
the source with :mod:`ast`, and the C++ include graph from the ``#include``
lines, so the script runs before the extension is built and needs only the
standard library.

Vocabulary
----------
* A *C++ area* is a directory directly under ``include/`` or ``src/``
  (``mesh``, ``spacetime``, ``observables``, ``simulations``, ``cobordism``,
  ``chainhodge``, ``quantum``, ...). Files directly under ``include/`` or
  ``src/`` form the *root area*. An area is *bound* when it has a
  ``src/<area>/Bindings.cpp``; a bound area is the ``tessera.<area>`` Python
  namespace.
* A *star-exported area* is one whose names ``tessera/__init__.py`` re-exports
  at the top level (``from tessera._tessera.<area> import *``). A test that
  uses a top-level name (``from tessera import Spacetime``, ``tessera.CDT``)
  is taken to reach every star-exported area and the root area, because the
  name's area is not resolved.
* The *Python import graph* has a node per module under ``tessera/`` and
  ``tests/`` and a node per bound C++ area. A module has an edge to every
  module it imports, resolved from ``import``/``from ... import`` statements
  (absolute and relative), from the attributes it reads on an imported module
  or on a plain alias of one (``T.cobordism``, ``tessera.drivers.qubit``,
  ``T = tessera``), from ``getattr``/``hasattr`` with a literal name, and from
  ``importlib.import_module``/``__import__`` of a literal name or of a name
  with a literal package prefix (``"tessera.drivers." + name`` reaches every
  module under ``tessera.drivers``). A module that imports ``tessera`` and
  uses it in a way the scanner cannot resolve (``getattr(tessera, name)``,
  ``from tessera import *``, a computed import with no package prefix)
  reaches every node. A test file that does not parse reaches every node.
* The *C++ include graph* has a node per file under ``include/`` and ``src/``
  and an edge from a file to every file its ``#include`` lines resolve to
  (relative to the including file's directory, then to ``include/`` and
  ``src/``). The *dependents* of a header are the files that include it,
  transitively. A source file's *interface* is the header of the same stem in
  its area; failing that, the headers of its own area that it includes
  directly; failing that, every header of its area. A ``Bindings.cpp`` has no
  interface: nothing includes it, so it affects only its own area.
* A *mention* is a string constant of a module that names a file: it is not a
  docstring, is at least five characters long, contains no whitespace, does
  not start with a dot, and either contains the file's basename, contains
  its stem (when the stem is at least five characters long), or equals the
  name of the file's parent directory when that directory holds no test
  module (``fixtures/causal_specimens``). Mentions catch the helpers and data
  loaded by path (``importlib.util.spec_from_file_location``,
  ``os.path.join``).

Rules
-----
Each changed path contributes a set of pytest paths; the selection is their
union. Any path that selects the full fast tier (``tests/``) makes the whole
selection the full fast tier. The rules, in the order they are tried:

1. Build and test infrastructure selects the full fast tier: ``CMakeLists.txt``,
   ``cmake/``, ``pyproject.toml``, every ``conftest.py``, ``.github/`` (the
   workflows and this script), ``third_party/``, ``src/bindings.cpp`` (the
   extension's entry point), ``tessera/__init__.py`` (the namespace
   registration) and ``tests/__init__.py``.
2. Documentation (``docs/`` and every ``*.md``) selects
   ``tests/test_cpp_api_docs_coverage.py``, which checks the C++ API page
   against ``include/``.
3. C++ under ``include/`` or ``src/``:
   * the affected files are the changed file, its dependents if it is a
     header, and the dependents of its interface headers if it is a source
     file; the affected areas are the areas of those files;
   * when the affected areas cover more than half of the bound areas, the
     closure has reached most of the library and the full fast tier runs;
   * otherwise the selection is ``tests/<area>/`` for every affected area with
     such a directory, every test module whose import graph reaches a bound
     affected area (or the top-level names, when the root or a star-exported
     area is affected), ``tests/drivers/`` (the drivers import most of the
     library) and ``tests/test_cpp_api_docs_coverage.py``;
   * a C++ change whose affected areas include no bound area and not the root
     area has no mapping to tests and selects the full fast tier.
   A file under ``include/`` or ``src/`` that is not C++ (``*.hpp.in``) selects
   the full fast tier.
4. Python under ``tessera/`` selects every test module whose import graph
   reaches the changed module (a package's ``__init__.py`` is reached by every
   import of the package or of a module in it). A change under
   ``tessera/drivers/`` also selects ``tests/test_examples.py``. A file under
   ``tessera/`` that is not Python selects the full fast tier.
5. ``examples/`` selects ``tests/test_examples.py`` (it runs the example
   scripts) and every test module that mentions ``examples`` or the changed
   file.
6. Under ``tests/``:
   * a test module selects itself (when it still exists) and every test module
     that imports or mentions it;
   * a helper module selects every test module that imports or mentions it;
     a ``conftest.py`` reached this way selects its directory;
   * a ``tests/<dir>/__init__.py`` selects ``tests/<dir>/``;
   * a data file under ``tests/<dir>/`` selects ``tests/<dir>/`` and every test
     module that mentions it; one under ``tests/fixtures/`` selects every test
     module that mentions it, and the full fast tier when none does;
   * any other file directly under ``tests/`` selects the full fast tier.
7. Any other path has no known mapping and selects the full fast tier.

Two guards close the selection: no changed paths at all (an empty or failed
diff) selects the full fast tier, and so does a selection that maps to no test
at all, since a change nothing exercises is more likely a gap in this mapping
than a dead file. When the selection covers at least nine tenths of the test
files it is replaced by ``tests/``, which runs the same tests with a shorter
command line.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import traceback
from collections import defaultdict, deque

CPP_HEADER_SUFFIXES = (".h", ".hpp", ".cuh", ".inl", ".tpp")
CPP_SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".cu")
CPP_SUFFIXES = CPP_HEADER_SUFFIXES + CPP_SOURCE_SUFFIXES

#: The area of the files directly under ``include/`` or ``src/`` (``Logger.h``,
#: ``Poset.cpp``): bound at the top level by ``src/bindings.cpp``.
ROOT_AREA = "<root>"

#: Import-graph node of the names ``tessera/__init__.py`` re-exports at the
#: top level (``tessera.Spacetime``, ``from tessera import CDT``).
TOP_LEVEL_NAMES = "tessera:<top-level names>"

#: Import-graph node a module reaches when it uses ``tessera`` in a way the
#: scanner cannot resolve; it has an edge to every other node.
EVERYTHING = "tessera:<everything>"

#: Test modules a C++ change always selects (the drivers import most of the
#: library; the coverage test reads ``include/``).
CPP_BASELINE = ("tests/drivers/", "tests/test_cpp_api_docs_coverage.py")

DOCS_COVERAGE_TEST = "tests/test_cpp_api_docs_coverage.py"
EXAMPLES_TEST = "tests/test_examples.py"
FULL_TIER = "tests/"

#: Paths and prefixes whose change selects the full fast tier (rule 1).
FULL_TIER_FILES = frozenset({
    "CMakeLists.txt",
    "pyproject.toml",
    "conftest.py",
    "src/bindings.cpp",
    "tessera/__init__.py",
    "tests/__init__.py",
})
FULL_TIER_PREFIXES = ("cmake/", ".github/", "third_party/")

#: Share of the test files above which the selection is replaced by ``tests/``.
WHOLE_SUITE_SHARE = 0.9

#: Shortest string constant that counts as a mention of a file.
MIN_MENTION_LENGTH = 5

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


def _cpp_area(rel_path: str) -> str:
    """The area of a path under ``include/`` or ``src/``."""
    parts = rel_path.split("/")
    return parts[1] if len(parts) > 2 else ROOT_AREA


def _is_test_file(rel_path: str) -> bool:
    name = os.path.basename(rel_path)
    return rel_path.startswith("tests/") and name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py"))


def _is_conftest(rel_path: str) -> bool:
    return os.path.basename(rel_path).startswith("conftest") and rel_path.endswith(".py")


def _module_name(rel_path: str) -> str:
    """The dotted module name of a Python file, relative to the repository root."""
    parts = rel_path[:-len(".py")].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _literal_prefix(node: ast.AST) -> tuple[str | None, bool]:
    """The literal prefix of a string expression, and whether it is the whole string.

    ``"a.b"`` gives ``("a.b", True)``; ``"a." + x`` and ``f"a.{x}"`` give
    ``("a.", False)``; an expression with no literal prefix gives ``(None, False)``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value, False
        return None, False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        prefix, _ = _literal_prefix(node.left)
        return prefix, False
    return None, False


class _ModuleScan:
    """What one Python module imports, read from its source with :mod:`ast`."""

    def __init__(self, tree: Tree, name: str, is_package: bool, source: str):
        self.tree = tree
        self.name = name
        self.is_package = is_package
        self.edges: set[str] = set()
        self.mentions: set[str] = set()
        self.unparsable = False
        try:
            root = ast.parse(source)
        except (SyntaxError, ValueError):
            self.unparsable = True
            return
        self._aliases: dict[str, str] = {}
        self._collect_imports(root)
        self._collect_uses(root)
        self._collect_mentions(root)

    # -- imports ---------------------------------------------------------

    def _package(self, level: int) -> str | None:
        """The package a relative import of the given level starts from."""
        base = self.name.split(".") if self.is_package else self.name.split(".")[:-1]
        if level - 1 > len(base):
            return None
        base = base[:len(base) - (level - 1)] if level > 1 else base
        return ".".join(base)

    def _collect_imports(self, root: ast.AST) -> None:
        self._alias_copies: set[int] = set()
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._add_dotted(alias.name)
                    bound = alias.asname or alias.name.split(".")[0]
                    self._aliases[bound] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = self._package(node.level)
                    if package is None:
                        continue
                    base = f"{package}.{node.module}" if node.module else package
                else:
                    base = node.module or ""
                self._add_dotted(base)
                for alias in node.names:
                    if alias.name == "*":
                        if self.tree.is_tessera_name(base):
                            self.edges.add(EVERYTHING)
                        continue
                    dotted = f"{base}.{alias.name}"
                    self._add_dotted(dotted)
                    self._aliases[alias.asname or alias.name] = dotted
            elif isinstance(node, ast.Call):
                self._collect_dynamic_import(node)
        # ``T = tessera`` makes ``T`` another name for the package.
        for node in ast.walk(root):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                    and node.value.id in self._aliases):
                self._alias_copies.add(id(node.value))
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._aliases[target.id] = self._aliases[node.value.id]

    def _add_dotted(self, dotted: str) -> None:
        nodes = self.tree.nodes_for(dotted)
        if not nodes:
            # A sibling imported by its bare name after
            # ``sys.path.insert(0, os.path.dirname(__file__))``:
            # ``from _causal_specimen import load_dump``.
            package = self._package(1)
            head = dotted.split(".")[0]
            if package and f"{package}.{head}" in self.tree.py_modules:
                nodes = self.tree.nodes_for(f"{package}.{dotted}")
        self.edges |= nodes

    def _collect_dynamic_import(self, call: ast.Call) -> None:
        """``importlib.import_module(x)`` and ``__import__(x)``.

        A literal name is an ordinary import. A name built from a literal
        prefix (``"tessera.drivers." + name``, ``f"tessera.{name}"``) reaches
        every module under the prefix. Anything else reaches everything.
        """
        func = call.func
        is_import_module = (isinstance(func, ast.Attribute) and func.attr == "import_module")
        is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
        if not (is_import_module or is_dunder_import) or not call.args:
            return
        prefix, complete = _literal_prefix(call.args[0])
        if prefix is None:
            self.edges.add(EVERYTHING)
        elif complete:
            self._add_dotted(prefix)
        else:
            self.edges |= self.tree.nodes_under(prefix)

    # -- attribute uses --------------------------------------------------

    def _collect_uses(self, root: ast.AST) -> None:
        """Resolve ``alias.attr...`` chains and bare uses of module aliases."""
        attribute_values: set[int] = set()
        literal_lookups: set[int] = set()
        for node in ast.walk(root):
            if isinstance(node, ast.Attribute):
                attribute_values.add(id(node.value))
                self._resolve_chain(node)
            elif isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Name) and func.id in ("getattr", "hasattr")
                        and len(node.args) >= 2 and isinstance(node.args[0], ast.Name)):
                    second = node.args[1]
                    if isinstance(second, ast.Constant) and isinstance(second.value, str):
                        literal_lookups.add(id(node.args[0]))
                        self._resolve_dotted_use(node.args[0].id, [second.value])
        for node in ast.walk(root):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and id(node) not in attribute_values
                    and id(node) not in literal_lookups
                    and id(node) not in self._alias_copies):
                self._resolve_dotted_use(node.id, [])

    def _resolve_chain(self, node: ast.Attribute) -> None:
        attrs: list[str] = []
        value: ast.AST = node
        while isinstance(value, ast.Attribute):
            attrs.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            attrs.reverse()
            self._resolve_dotted_use(value.id, attrs)

    def _resolve_dotted_use(self, name: str, attrs: list[str]) -> None:
        base = self._aliases.get(name)
        if base is None or not self.tree.is_tessera_name(base):
            return
        if attrs and attrs[0].startswith("__"):
            return
        if not attrs:
            # A bare use of the package itself (``getattr(tessera, name)``,
            # ``dir(tessera)``) can reach anything in it.
            if self.tree.is_extension_root(base):
                self.edges.add(EVERYTHING)
            return
        self.edges |= self.tree.nodes_for(".".join([base, *attrs]))

    # -- mentions --------------------------------------------------------

    def _collect_mentions(self, root: ast.AST) -> None:
        docstrings: set[int] = set()
        for node in ast.walk(root):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                docstrings.add(id(node.value))
        for node in ast.walk(root):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and len(node.value) >= MIN_MENTION_LENGTH
                    and not node.value.startswith(".")
                    and not any(c.isspace() for c in node.value)):
                self.mentions.add(node.value)


class Tree:
    """The repository's Python import graph and C++ include graph."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.notes: list[str] = []
        # C++
        self.cpp_files: set[str] = set()
        self.includes: dict[str, set[str]] = defaultdict(set)
        self.included_by: dict[str, set[str]] = defaultdict(set)
        self.areas: set[str] = set()
        self.bound_areas: set[str] = set()
        self.star_exported_areas: set[str] = set()
        # Python
        self.py_modules: dict[str, str] = {}      # dotted name -> path
        self.test_files: list[str] = []
        self.conftest_dirs: dict[str, str] = {}   # dotted name -> directory
        self.scans: dict[str, _ModuleScan] = {}
        self.reverse: dict[str, set[str]] = defaultdict(set)
        self._scan_cpp()
        self._scan_python()

    # -- repository walk -------------------------------------------------

    def _walk(self, top: str):
        base = os.path.join(self.root, top)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for filename in sorted(filenames):
                yield os.path.relpath(os.path.join(dirpath, filename), self.root).replace(os.sep, "/")

    def exists(self, rel_path: str) -> bool:
        return os.path.exists(os.path.join(self.root, rel_path))

    def _read(self, rel_path: str) -> str:
        with open(os.path.join(self.root, rel_path), encoding="utf-8", errors="replace") as handle:
            return handle.read()

    # -- C++ -------------------------------------------------------------

    def _scan_cpp(self) -> None:
        for top in ("include", "src"):
            for rel in self._walk(top):
                if rel.endswith(CPP_SUFFIXES):
                    self.cpp_files.add(rel)
                    area = _cpp_area(rel)
                    if area != ROOT_AREA:
                        self.areas.add(area)
        for area in self.areas:
            if f"src/{area}/Bindings.cpp" in self.cpp_files:
                self.bound_areas.add(area)
        for rel in self.cpp_files:
            for line in self._read(rel).splitlines():
                match = _INCLUDE_RE.match(line)
                if not match:
                    continue
                target = self._resolve_include(rel, match.group(1))
                if target is not None:
                    self.includes[rel].add(target)
                    self.included_by[target].add(rel)

    def _resolve_include(self, including: str, spec: str) -> str | None:
        candidates = [
            os.path.normpath(os.path.join(os.path.dirname(including), spec)),
            os.path.normpath(os.path.join("include", spec)),
            os.path.normpath(os.path.join("src", spec)),
        ]
        for candidate in candidates:
            candidate = candidate.replace(os.sep, "/")
            if candidate in self.cpp_files:
                return candidate
        return None

    def dependents(self, rel_path: str) -> set[str]:
        """Every file that includes ``rel_path``, transitively."""
        seen: set[str] = set()
        queue = deque([rel_path])
        while queue:
            current = queue.popleft()
            for dependent in self.included_by.get(current, ()):
                if dependent not in seen:
                    seen.add(dependent)
                    queue.append(dependent)
        return seen

    def interface_headers(self, source: str) -> set[str]:
        """The headers whose declarations a source file implements."""
        area = _cpp_area(source)
        stem = os.path.splitext(os.path.basename(source))[0]
        area_headers = {
            f for f in self.cpp_files
            if f.endswith(CPP_HEADER_SUFFIXES) and _cpp_area(f) == area}
        same_stem = {f for f in area_headers if os.path.splitext(os.path.basename(f))[0] == stem}
        if same_stem:
            return same_stem
        direct = {f for f in self.includes.get(source, ()) if f in area_headers}
        if direct:
            return direct
        return area_headers

    def affected_cpp(self, rel_path: str) -> tuple[set[str], str]:
        """The files a C++ change can affect, and how they were found."""
        basename = os.path.basename(rel_path)
        if rel_path.endswith(CPP_HEADER_SUFFIXES):
            dependents = self.dependents(rel_path)
            return {rel_path} | dependents, f"header; {len(dependents)} dependents"
        if basename.lower().endswith("bindings.cpp"):
            return {rel_path}, "bindings translation unit; nothing includes it"
        if rel_path not in self.cpp_files:
            return {rel_path}, "source not in the tree (deleted); its own area"
        headers = self.interface_headers(rel_path)
        affected = {rel_path}
        for header in headers:
            affected |= {header} | self.dependents(header)
        return affected, f"source; interface {sorted(headers)}; {len(affected) - 1} dependents"

    # -- Python ----------------------------------------------------------

    def _scan_python(self) -> None:
        sources: dict[str, tuple[str, bool]] = {}
        for top in ("tessera", "tests"):
            for rel in self._walk(top):
                if not rel.endswith(".py"):
                    continue
                name = _module_name(rel)
                is_package = rel.endswith("/__init__.py")
                self.py_modules[name] = rel
                if _is_test_file(rel):
                    self.test_files.append(rel)
                if _is_conftest(rel):
                    self.conftest_dirs[name] = os.path.dirname(rel) + "/"
                sources[name] = (rel, is_package)
        self.star_exported_areas = self._star_exported_areas()
        for name, (rel, is_package) in sources.items():
            if rel == "tessera/__init__.py":
                # The package registers every namespace; following its
                # imports would make every module reach every area.
                continue
            scan = _ModuleScan(self, name, is_package, self._read(rel))
            if scan.unparsable:
                self.notes.append(f"{rel} does not parse; it is taken to reach everything")
                scan.edges.add(EVERYTHING)
            self.scans[name] = scan
            for target in scan.edges:
                self.reverse[target].add(name)
        # Mentions: a module that names a helper file loads it by path.
        for helper, rel in self.py_modules.items():
            if not rel.startswith("tests/") or _is_test_file(rel):
                continue
            for name in self.mentioning(rel):
                self.reverse[helper].add(name)

    def _star_exported_areas(self) -> set[str]:
        rel = "tessera/__init__.py"
        if not self.exists(rel):
            return set()
        try:
            root = ast.parse(self._read(rel))
        except (SyntaxError, ValueError):
            return set(self.bound_areas)
        areas: set[str] = set()
        for node in ast.walk(root):
            if (isinstance(node, ast.ImportFrom) and node.module
                    and any(alias.name == "*" for alias in node.names)):
                parts = node.module.split(".")
                if parts[:2] == ["tessera", "_tessera"] and len(parts) == 3:
                    areas.add(parts[2])
        return areas

    def is_tessera_name(self, dotted: str) -> bool:
        return dotted == "tessera" or dotted.startswith("tessera.")

    def is_extension_root(self, dotted: str) -> bool:
        return dotted in ("tessera", "tessera._tessera")

    def nodes_for(self, dotted: str) -> set[str]:
        """The import-graph nodes an import or attribute chain reaches."""
        parts = dotted.split(".")
        if parts[0] == "tests":
            nodes = {dotted}
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix in self.py_modules:
                    nodes.add(prefix)
            return nodes
        if parts[0] != "tessera" or len(parts) == 1:
            return set()
        if parts[1] == "_tessera":
            if len(parts) == 2:
                return set()
            return {f"cpp:{parts[2]}"} if parts[2] in self.areas else {TOP_LEVEL_NAMES}
        if parts[1].startswith("_tessera_"):
            return {f"cpp:{parts[1][len('_tessera_'):]}"}
        nodes = {dotted}
        known = False
        for i in range(2, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix in self.py_modules:
                nodes.add(prefix)
                known = True
        if parts[1] in self.bound_areas:
            nodes.add(f"cpp:{parts[1]}")
        elif not known:
            # Neither a module nor a namespace: a top-level re-exported name.
            nodes.add(TOP_LEVEL_NAMES)
        return nodes

    def nodes_under(self, prefix: str) -> set[str]:
        """The nodes a computed import with a literal prefix can reach.

        The prefix must name at least a package below ``tessera`` or ``tests``
        (``tessera.drivers.``); a shorter prefix can reach anything.
        """
        parts = prefix.split(".")
        if parts[0] not in ("tessera", "tests") or len(parts) < 3:
            return {EVERYTHING}
        nodes = {name for name in self.py_modules
                 if name.startswith(prefix) or name == prefix.rstrip(".")}
        nodes |= {f"cpp:{area}" for area in self.bound_areas
                  if f"tessera.{area}".startswith(prefix)}
        return nodes or {EVERYTHING}

    def mentioning(self, rel_path: str) -> set[str]:
        """The modules whose string constants mention the file."""
        basename = os.path.basename(rel_path)
        stem = os.path.splitext(basename)[0]
        directory = os.path.dirname(rel_path)
        # The parent directory's name counts only for a data directory (one
        # with no test modules of its own): ``fixtures/causal_specimens``.
        parent = os.path.basename(directory)
        if any(os.path.dirname(f) == directory for f in self.test_files):
            parent = None
        found: set[str] = set()
        for name, scan in self.scans.items():
            for constant in scan.mentions:
                if (constant == parent or basename in constant
                        or (len(stem) >= MIN_MENTION_LENGTH and stem in constant)):
                    found.add(name)
                    break
        return found

    def reaching(self, targets: set[str]) -> set[str]:
        """Every module whose import graph reaches one of the targets."""
        seen: set[str] = set()
        # A module that reaches everything reaches every target; it and its
        # importers are part of every answer.
        queue = deque(set(targets) | {EVERYTHING})
        while queue:
            current = queue.popleft()
            for importer in self.reverse.get(current, ()):
                if importer not in seen:
                    seen.add(importer)
                    queue.append(importer)
        return seen

    def tests_reaching(self, targets: set[str]) -> set[str]:
        """The pytest paths of the modules reaching the targets."""
        paths: set[str] = set()
        for name in self.reaching(targets):
            rel = self.py_modules.get(name)
            if rel is None:
                continue
            if name in self.conftest_dirs:
                paths.add(self.conftest_dirs[name])
            elif _is_test_file(rel):
                paths.add(rel)
        return paths

    def test_files_under(self, path: str) -> set[str]:
        if path.endswith("/"):
            return {f for f in self.test_files if f.startswith(path)}
        return {path} if path in self.test_files else set()


class Selection:
    """The pytest paths to run, with the reason for each contribution."""

    def __init__(self):
        self.paths: set[str] = set()
        self.reasons: list[str] = []
        self.full_reason: str | None = None

    def add(self, changed: str, rule: str, paths) -> None:
        paths = sorted(set(paths))
        self.paths |= set(paths)
        self.reasons.append(f"{changed}: {rule} -> {', '.join(paths) if paths else 'nothing'}")

    def full(self, changed: str, rule: str) -> None:
        self.reasons.append(f"{changed}: {rule} -> {FULL_TIER} (full fast tier)")
        if self.full_reason is None:
            self.full_reason = f"{rule} ({changed})"

    @property
    def is_full(self) -> bool:
        return self.full_reason is not None


def _select_cpp(tree: Tree, changed: str, selection: Selection) -> None:
    if not changed.endswith(CPP_SUFFIXES):
        selection.full(changed, "not a C++ source or header under include/ or src/")
        return
    affected, how = tree.affected_cpp(changed)
    areas = {_cpp_area(f) for f in affected}
    bound = areas & tree.bound_areas
    if len(bound) > len(tree.bound_areas) / 2:
        selection.full(
            changed,
            f"the include closure covers {len(bound)} of {len(tree.bound_areas)} bound areas "
            f"({', '.join(sorted(bound))}; {how})")
        return
    if not bound and ROOT_AREA not in areas:
        selection.full(
            changed,
            f"the affected areas ({', '.join(sorted(areas))}) have no Python binding ({how})")
        return
    targets = {f"cpp:{area}" for area in bound}
    if ROOT_AREA in areas or bound & tree.star_exported_areas:
        targets.add(TOP_LEVEL_NAMES)
    paths = set(CPP_BASELINE)
    for area in sorted(bound):
        if tree.exists(f"tests/{area}/"):
            paths.add(f"tests/{area}/")
    paths |= tree.tests_reaching(targets)
    selection.add(
        changed,
        f"C++ ({how}); affected areas {', '.join(sorted(areas))}; "
        f"tests reaching {', '.join(sorted(targets))}",
        paths)


def _select_python(tree: Tree, changed: str, selection: Selection) -> None:
    if not changed.endswith(".py"):
        selection.full(changed, "not a Python module under tessera/")
        return
    name = _module_name(changed)
    paths = tree.tests_reaching({name})
    if changed.startswith("tessera/drivers/"):
        paths.add(EXAMPLES_TEST)
    selection.add(changed, f"Python module {name}; tests reaching it", paths)


def _select_examples(tree: Tree, changed: str, selection: Selection) -> None:
    names = tree.mentioning(changed)
    names |= {name for name, scan in tree.scans.items() if "examples" in scan.mentions}
    paths = tree.tests_reaching(names) | {EXAMPLES_TEST}
    for name in names:
        rel = tree.py_modules[name]
        if _is_test_file(rel):
            paths.add(rel)
    selection.add(changed, "example script; the examples test and the tests mentioning examples", paths)


def _select_tests(tree: Tree, changed: str, selection: Selection) -> None:
    directory = os.path.dirname(changed)
    top_level = directory == "tests"
    if changed.endswith(".py"):
        name = _module_name(changed)
        if changed.endswith("/__init__.py"):
            selection.add(changed, "test package; its directory", {directory + "/"})
            return
        importers = tree.tests_reaching({name})
        if _is_test_file(changed):
            own = {changed} if tree.exists(changed) else set()
            selection.add(changed, "test module; itself and the tests importing it", own | importers)
        else:
            if tree.exists(changed):
                for mentioner in tree.mentioning(changed):
                    rel = tree.py_modules[mentioner]
                    if _is_test_file(rel):
                        importers.add(rel)
                    importers |= tree.tests_reaching({mentioner})
            selection.add(changed, "test helper; the tests importing or mentioning it", importers)
        return
    if top_level:
        selection.full(changed, "a file directly under tests/ that is not a module")
        return
    mentioners = tree.mentioning(changed)
    paths: set[str] = set()
    for name in mentioners:
        rel = tree.py_modules[name]
        if _is_test_file(rel):
            paths.add(rel)
        paths |= tree.tests_reaching({name})
    if changed.startswith("tests/fixtures/"):
        if not paths:
            selection.full(changed, "a fixture file no test mentions")
            return
        selection.add(changed, "fixture file; the tests mentioning it", paths)
        return
    paths.add("/".join(directory.split("/")[:2]) + "/")
    selection.add(changed, "test data; its tests directory and the tests mentioning it", paths)


def select(tree: Tree, changed_paths) -> Selection:
    """Map the changed paths to the pytest paths to run."""
    selection = Selection()
    changed_paths = sorted({p.strip().replace(os.sep, "/") for p in changed_paths if p.strip()})
    if not changed_paths:
        selection.full("(no changed paths)", "an empty or failed diff")
        return selection
    for changed in changed_paths:
        if changed in FULL_TIER_FILES or changed.startswith(FULL_TIER_PREFIXES) or _is_conftest(changed):
            selection.full(changed, "build or test infrastructure")
        elif changed.startswith("docs/") or changed.endswith(".md"):
            selection.add(changed, "documentation; the C++ API coverage test", {DOCS_COVERAGE_TEST})
        elif changed.startswith(("include/", "src/")):
            _select_cpp(tree, changed, selection)
        elif changed.startswith("tessera/"):
            _select_python(tree, changed, selection)
        elif changed.startswith("examples/"):
            _select_examples(tree, changed, selection)
        elif changed.startswith("tests/"):
            _select_tests(tree, changed, selection)
        else:
            selection.full(changed, "no known mapping")
    if selection.is_full:
        return selection
    if FULL_TIER in selection.paths:
        selection.full("(union)", "a change reaches the conftest of tests/")
        return selection
    selection.paths = {p for p in selection.paths if tree.exists(p)}
    if not selection.paths:
        selection.full("(union)", "the changed paths map to no test")
        return selection
    selected = set()
    for path in selection.paths:
        selected |= tree.test_files_under(path)
    if tree.test_files and len(selected) >= WHOLE_SUITE_SHARE * len(tree.test_files):
        selection.full(
            "(union)",
            f"the selection covers {len(selected)} of {len(tree.test_files)} test files")
    return selection


def pytest_paths(selection: Selection) -> list[str]:
    """The selection as a sorted list of paths, files under selected directories dropped."""
    if selection.is_full:
        return [FULL_TIER]
    directories = sorted(p for p in selection.paths if p.endswith("/"))
    paths = [p for p in selection.paths if not any(p != d and p.startswith(d) for d in directories)]
    return sorted(paths)


def _report(tree: Tree, selection: Selection, paths: list[str], log) -> None:
    print("select_tests: reasons per changed path", file=log)
    for reason in selection.reasons:
        print(f"  {reason}", file=log)
    for note in tree.notes:
        print(f"  note: {note}", file=log)
    total = len(tree.test_files)
    if selection.is_full:
        print(f"select_tests: running the full fast tier ({FULL_TIER}, {total} test files) "
              f"because {selection.full_reason}", file=log)
        return
    selected = set()
    for path in paths:
        selected |= tree.test_files_under(path)
    print(f"select_tests: selected {len(selected)} of {total} test files via {len(paths)} paths:", file=log)
    for path in paths:
        print(f"  {path}", file=log)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", help="changed paths, relative to the repository root")
    parser.add_argument("--changed-paths", metavar="FILE",
                        help="file with one changed path per line ('-' for standard input)")
    parser.add_argument("--root", default=None,
                        help="repository root (default: two directories above this script)")
    parser.add_argument("--output", metavar="FILE",
                        help="write the selected pytest paths here instead of standard output")
    args = parser.parse_args(argv)

    root = args.root or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    changed = list(args.paths)
    if args.changed_paths:
        handle = sys.stdin if args.changed_paths == "-" else open(args.changed_paths, encoding="utf-8")
        with handle:
            changed.extend(line.rstrip("\n") for line in handle)

    try:
        tree = Tree(root)
        selection = select(tree, changed)
        paths = pytest_paths(selection)
        _report(tree, selection, paths, sys.stderr)
    except Exception:  # noqa: BLE001 - a selector failure must never fail the job
        traceback.print_exc(file=sys.stderr)
        print("select_tests: the selector failed; running the full fast tier", file=sys.stderr)
        paths = [FULL_TIER]

    text = "\n".join(paths) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
