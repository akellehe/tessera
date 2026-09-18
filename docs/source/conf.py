# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

import os, sys, pathlib, tomllib

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DOXY_XML = THIS_DIR.parent / "_doxygen" / "xml"

# Read version from pyproject.toml (single source of truth)
with open(REPO_ROOT / "pyproject.toml", "rb") as f:
    _pyproject = tomllib.load(f)
version = _pyproject["project"]["version"]
release = version

project = "tessera"
author = "tessera contributors"

extensions = [
    "myst_parser",
    "breathe",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
]

# Bibliographic references for the pages that use {cite} (the causal-order
# charter, causal_sets, wilson_loops, ...). All bib entries live in
# references.bib next to this conf.py so they're auto-discovered by
# sphinx-build.
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"

myst_enable_extensions = ["dollarmath", "amsmath", "substitution", "colon_fence"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_heading_anchors = 3

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

breathe_projects = {"tessera": str((pathlib.Path(__file__).parent.parent / "_doxygen" / "xml").resolve())}
breathe_default_project = "tessera"

# Prefix every auto-generated section label with the source-document
# path so identical section headings across files (`## Setup`,
# `## See also`, `## Results`, etc.) don't collide. Without this,
# Sphinx logs many "duplicate label" warnings on every build.
autosectionlabel_prefix_document = True

# The cpp_api.md page uses ``{doxygenfile}`` to pull in every Doxygen-
# generated header, and each header re-declares the ``tessera`` namespace,
# producing 'Duplicate C++ declaration / Duplicate ID: namespacetessera'
# warnings that can't be fixed at the source level without restructuring
# the Doxygen import. Suppress just that category. Real cpp warnings
# (missing declarations, etc.) are not in 'cpp.duplicate_declaration'.
suppress_warnings = [
    # Both spellings — Sphinx 7.x emits 'duplicate_declaration.cpp'.
    "duplicate_declaration.cpp",
    "duplicate_declaration",
    "docutils",  # paired 'Duplicate ID / explicit target name' warnings
                 # come from the same root cause as duplicate_declaration
    # cpp_api.md re-imports each Doxygen-generated header with
    # ``{doxygenfile}`` directives. Doxygen emits identically-named
    # sub-sections ("Implementation Details", "Example", "References",
    # ...) in many of them, which collides at autosectionlabel time.
    # Real same-file labels still work for navigation in our hand-
    # written pages; this only silences the imported Doxygen pages.
    "autosectionlabel.cpp_api",
    # The Schwinger subsystem page renders the tessera.quantum module
    # docstring via autodoc; that docstring repeats section titles
    # ("Quickstart", "Causal-order comparison", ...) that also appear as
    # hand-written headings on the page, colliding at autosectionlabel
    # time.
]


def setup(app):
    """Filter out the noisy 'duplicate object description' autodoc
    warnings for tessera.quantum re-exports. These come from
    tessera/quantum/__init__.py mirroring every pybind11 class out
    of _tessera.quantum at the package level — autodoc then sees
    the same attribute object under two namespace paths and warns.
    They're cosmetic noise; the real warning class (Sphinx domain
    'duplicate object description') is not exposed via
    ``suppress_warnings``, so a logging filter is the only clean
    way to silence just these without hiding everything else.
    """
    import logging

    class _DuplicateObjectDescriptionFilter(logging.Filter):
        def filter(self, record):  # noqa: D401
            return "duplicate object description" not in record.getMessage()

    for name in ("sphinx", "sphinx.domains", "sphinx.domains.python"):
        logging.getLogger(name).addFilter(_DuplicateObjectDescriptionFilter())

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"

# Make |version| and |release| available in .md files via substitution
myst_substitutions = {
    "version": version,
    "release": release,
}
