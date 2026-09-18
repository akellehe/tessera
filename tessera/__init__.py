"""tessera -- Causal Set and CDT simulation library.

The heavy lifting lives in the C++ extension ``_tessera``. Classes are
organised into submodules whose names match their C++ namespaces:

* ``tessera.mesh``         — Vertex, Edge, Simplex, SimplexFilter, IDs
* ``tessera.spacetime``    — Spacetime, Metric, Signature, topologies, Pachner moves
* ``tessera.observables``  — SparseGraph, ModularityOptimizer, WilsonLoop, ...
* ``tessera.simulations``  — CDT, ReggeSolver, Simulation base
* ``tessera.quantum``      — Schwinger model, DMRG, TDVP, holography, InteractionSimulation

For backward compatibility every public class is also re-exported at the
top level, so ``from tessera import Spacetime`` continues to work alongside
the canonical ``from tessera.spacetime import Spacetime``.
"""

# Root-namespace classes / free functions (Poset, OrderAgreement,
# MatterConfiguration, HingeType, renderSpacetime, ForceLayout, ...).
from tessera._tessera import *                              # noqa: F401,F403
from tessera._tessera import __doc__                        # noqa: F401

# Subsystem submodules — make `tessera.mesh.Vertex` etc. importable.
from tessera._tessera import (                              # noqa: F401
    mesh,
    spacetime,
    observables,
    simulations,
    cobordism,
    chainhodge,
)

# Register them under their `tessera.*` names as well.
#
# pybind11 binds a submodule as an ATTRIBUTE of its parent. Python's import
# machinery does not look at attributes: `from tessera.cobordism import X`
# resolves through `sys.modules['tessera.cobordism']`, so without this the
# attribute works and the import raises ModuleNotFoundError. Registering the
# same object under both makes the two forms agree, rather than one of them
# being a second, subtly different module.
#
# `quantum` is deliberately absent: there is a real `tessera/quantum/` package
# on disk, and it re-exports the C++ submodule's names alongside its own pure
# Python ones. Registering the C++ submodule here would shadow it, which is the
# bug this replaced — `tessera.quantum` then meant the C++ module while
# `from tessera.quantum import ...` meant the package, so
# `tessera.quantum.surface_periods` raised while the import of the same name
# succeeded. Let the package own the name.
import sys as _sys                                          # noqa: E402

for _submodule in (mesh, spacetime, observables, simulations, cobordism,
                   chainhodge):
    _sys.modules[f"tessera.{_submodule.__name__.rsplit('.', 1)[-1]}"] = _submodule
del _submodule

# Backward-compat re-exports at top level. Star-import each submodule so
# existing scripts that do `from tessera import Spacetime`, `tessera.CDT`,
# etc. continue to work.
from tessera._tessera.mesh        import *                  # noqa: F401,F403
from tessera._tessera.spacetime   import *                  # noqa: F401,F403
from tessera._tessera.observables import *                  # noqa: F401,F403
from tessera._tessera.simulations import *                  # noqa: F401,F403
# NB: cobordism is intentionally NOT star-imported to the top level. It is a
# specialized subsystem (no backward-compat scripts) and some of its names
# would shadow core ones — e.g. cobordism.Signature vs the metric
# spacetime.Signature. Access it as ``tessera.cobordism.*``.
# Quantum is subsystem-namespaced too, and is the one namespace with a Python
# package of its own: `tessera/quantum/__init__.py` re-exports the C++ names and
# adds the pure Python modules (surface_periods, symmetric_genus_two, ...), so
# importing it is what gives the full surface. Scripts already use
# `tessera.quantum.*`, which resolves to that package.
