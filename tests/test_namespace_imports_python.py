# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The Python namespaces line up with the C++ ones.

pybind11 binds a submodule as an ATTRIBUTE of its parent, and Python's import
machinery does not look at attributes: ``from tessera.cobordism import X``
resolves through ``sys.modules['tessera.cobordism']``. Without an entry there
the attribute works and the import raises ``ModuleNotFoundError``, so the two
spellings of the same namespace disagree.

The subtler failure is a namespace resolving to two different objects. There is
a real ``tessera/quantum/`` package as well as a C++ ``quantum`` submodule, and
assigning the latter as an attribute shadowed the former until something
imported it -- so ``tessera.quantum.surface_periods`` raised while
``from tessera.quantum import surface_periods`` succeeded, and which one a
caller got depended on import order.
"""

import importlib
import sys

import pytest

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")

#: The namespaces that mirror a C++ namespace of the same name.
_NAMESPACES = ("chainhodge", "cobordism", "mesh", "observables", "simulations",
               "spacetime", "quantum")

#: One representative class per namespace, to prove `from ... import` resolves
#: a real binding rather than an empty module.
_REPRESENTATIVES = {
    "chainhodge": "ChainHodge",
    "cobordism": "HodgeLaplacian",
    "mesh": "Vertex",
    "observables": "WilsonLoop",
    "simulations": "ReggeSolver",
    "spacetime": "Spacetime",
    "quantum": "SchwingerModel",
}


@pytest.mark.parametrize("name", _NAMESPACES)
def test_the_namespace_is_importable_as_a_module(name):
    module = importlib.import_module(f"tessera.{name}")
    assert module is not None
    assert f"tessera.{name}" in sys.modules


@pytest.mark.parametrize("name", _NAMESPACES)
def test_the_attribute_and_the_module_are_the_same_object(name):
    """Otherwise which one a caller gets depends on import order."""
    imported = importlib.import_module(f"tessera.{name}")
    attribute = getattr(tessera, name)
    assert attribute is imported, (
        f"tessera.{name} as an attribute is a different object from "
        f"tessera.{name} as a module, so the two spellings can disagree")


@pytest.mark.parametrize("name,symbol", sorted(_REPRESENTATIVES.items()))
def test_from_import_resolves_a_real_binding(name, symbol):
    module = importlib.import_module(f"tessera.{name}")
    assert hasattr(module, symbol), (
        f"from tessera.{name} import {symbol} should resolve")


def test_quantum_carries_its_pure_python_modules_too():
    """The quantum namespace is the package, not the bare C++ submodule.

    The package re-exports the C++ names and adds the pure Python ones; the C++
    submodule alone has no surface_periods, so this is what distinguishes them.
    """
    import tessera.quantum as quantum
    assert hasattr(quantum, "SchwingerModel"), "the C++ names are re-exported"
    from tessera.quantum import surface_periods  # noqa: F401
    assert hasattr(quantum, "surface_periods"), (
        "the attribute must see the pure Python modules, not just the C++ ones")


def test_the_top_level_re_exports_still_work():
    """Backward compatibility: the flat spelling keeps resolving."""
    from tessera import Spacetime, CDT  # noqa: F401
    assert tessera.Spacetime is Spacetime
