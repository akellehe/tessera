// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "spacetime/topologies/Topology.h"
#include "spacetime/topologies/Cylinder.h"
#include "spacetime/topologies/Sphere.h"
#include "spacetime/topologies/Toroid.h"
#include "simulations/CDT.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include "spacetime/pachner/ShiftMove.h"
#include "simulations/ReggeSolver.h"
#include "mesh/SimplexFilter.h"
#include "observables/ModularityOptimizer.h"
#include "observables/SparseGraph.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
#include "spacetime/Spacetime.h"
#include "ForceLayout.h"
#include "Poset.h"
#include "mesh/VertexList.h"
#include "mesh/EdgeList.h"
#include "spacetime/Signature.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "spacetime/Metric.h"
#include "Renderer.h"

#include <vector>
#include <algorithm>

namespace py = pybind11;

using namespace tessera;

// Defined in src/quantum/Bindings.cpp — always built.
void register_quantum(py::module_ m);

void register_mesh(py::module_ m);
void register_spacetime(py::module_ m);
void register_observables(py::module_ m);
void register_isospin_doublet(py::module_ m);
void register_simulations(py::module_ m);
void register_cobordism(py::module_ m);
void register_chainhodge(py::module_ m);
void register_matter(py::module_ m);

PYBIND11_MODULE(_tessera, m) {
  m.doc() = R"doc(
tessera -- Causal Set and CDT simulation library.

A C++ library (with Python bindings) for Causal Dynamical Triangulations
(CDT) and causal set theory simulations in arbitrary dimension.

Typical usage::

    import tessera

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(500)
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.get_n41())
    cdt.tune()
    cdt.sweep(100)

References:
  [RU]  Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
        Phys. Rev. D 72 (2005), arXiv:hep-th/0505154v2
  [BGL] Brunekreef, Gorlich, Loll, "Simulating CDT quantum gravity",
        arXiv:2310.16744v1 (2023)
)doc";

  // ========================================
  // Subsystem submodules — 1:1 with C++ namespaces
  // ========================================
  //
  // Each subsystem's classes live in its own Python submodule
  // (`tessera.mesh`, `tessera.spacetime`, ...). Backward-compat top-level
  // re-exports live in `tessera/__init__.py`.
  auto m_mesh        = m.def_submodule("mesh",
      "Vertex / Edge / Simplex primitives, SimplexFilter, and ID typedefs.");
  auto m_spacetime   = m.def_submodule("spacetime",
      "Spacetime simplicial complex, Metric, Signature, Foliation, "
      "topologies, Pachner moves.");
  auto m_observables = m.def_submodule("observables",
      "Observables on a Spacetime: SparseGraph, ModularityOptimizer, "
      "WilsonLoop, VolumeProfile.");
  auto m_simulations = m.def_submodule("simulations",
      "Monte Carlo simulations: CDT, ReggeSolver, Simulation base class.");
  auto m_cobordism   = m.def_submodule("cobordism",
      "Cobordisms between PL manifolds: characteristic numbers, verification, "
      "reconstruction.");
  auto m_matter      = m.def_submodule("matter",
      "Matter on a triangulation: MatterConfiguration, worldlines, hinge "
      "classification.");
  auto m_chainhodge  = m.def_submodule("chainhodge",
      "Chain-level Whitney Hodge pencil: sparse inverse chain metrics, "
      "branches, and instance certificates.");

  // --- Per-subsystem bindings (one file per subsystem) ---
  register_mesh(m_mesh);
  register_spacetime(m_spacetime);
  register_matter(m_matter);
  register_observables(m_observables);
  register_isospin_doublet(m_observables);
  register_simulations(m_simulations);
  register_cobordism(m_cobordism);
  register_chainhodge(m_chainhodge);

  // ========================================
  // Poset: finite partial orders and their comparison (include/Poset.h)
  // ========================================
  py::class_<Poset>(m, "Poset",
          R"doc(Hasse / cover representation of a finite partial order.

Nodes are integers ``0 .. getNodeCount - 1``. ``covers`` lists the cover
edges: each entry ``(a, b)`` means ``a`` strictly precedes ``b`` with no
intermediate node. The full strict order is the transitive closure of
the covers; see :func:`compare_orders` for pairwise statistics derived
from that closure.

Construct empty (``Poset()``) and resize via the ``get_node_count``
setter, or pass an integer to pre-populate node count
(``Poset(4)``). Mutate via :meth:`add_cover` (single edge) or the
``covers`` setter (whole list). The class makes no internal
consistency checks; callers are responsible for transitivity and
acyclicity.

See ``docs/source/causal_sets.md`` for the conceptual background.
)doc")
      .def(py::init<>())
      .def(py::init<int>(), py::arg("node_count"),
          "Construct with the node count pre-set to ``node_count``.")
      .def("add_cover", &Poset::addCover, py::arg("a"), py::arg("b"),
          R"doc(Add the cover edge ``a -> b`` (a strictly precedes b, no intermediate).

Both endpoints must already exist (call the int constructor or the
``get_node_count`` setter first). No deduplication is performed — adding
the same cover twice creates two parallel edges. Covers normally come
from a transitive reduction, where duplicates cannot arise.
)doc")
      .def_property("get_node_count",
          [](Poset const& p) { return p.getNodeCount(); },
          [](Poset& p, int n) { p.setNodeCount(n); },
          "Number of nodes. Setting grows the node set; nodes are not "
          "removed if you set a smaller value, and cover edges are "
          "preserved across resizes.")
      .def("get_cover_count", &Poset::getCoverCount,
          "Number of cover edges currently registered.")
      .def_property("covers",
          [](Poset const& p) { return p.covers(); },
          [](Poset& p, std::vector<std::pair<int, int>> const& covers) {
              p.setCovers(covers);
          },
          "Cover edges as a list of ``(a, b)`` pairs. Setting replaces "
          "the entire cover list in one pass.")
      .def("to_dot", &Poset::toDot,
          "Graphviz DOT representation of the Hasse diagram. "
          "Nodes labelled by their integer id; one directed edge per "
          "cover. Suitable for ``dot -Tsvg`` rendering.")
      .def_static("from_spacetime",
          [](py::object spacetime_obj) {
              auto const* st = spacetime_obj.cast<tessera::spacetime::Spacetime const*>();
              return tessera::Poset::fromSpacetime(*st);
          }, py::arg("spacetime"),
          R"doc(Build the causet partial order on a Spacetime's vertices.

Reads the directed-edge / timelike-edge subgraph as the strict
``precedes`` relation, then takes the transitive reduction to recover
cover edges. The result has one node per Spacetime vertex (in
ascending ID order); cover edges are between strictly comparable
vertices with no intermediate.
)doc")
      .def("__repr__", [](Poset const& p) {
          return "Poset(getNodeCount=" + std::to_string(p.getNodeCount()) +
                 ", covers=" + std::to_string(p.getCoverCount()) + " edges)";
      });

  py::class_<OrderAgreement>(m, "OrderAgreement",
          R"doc(Pairwise agreement statistics between two posets.

Counted over unordered pairs (i, j) with i < j:
* concordant — both orders relate the pair, in the same direction.
* discordant — both orders relate the pair, in opposite directions.
* only_a / only_b — exactly one order relates the pair.

Build via :meth:`Majorization.agreement(a, b, n_labels)`.
)doc")
      .def_readonly("kendall_tau",         &OrderAgreement::kendallTau)
      .def_readonly("discordant_fraction", &OrderAgreement::discordantFraction)
      .def_readonly("hasse_edit_distance",  &OrderAgreement::hasseEditDistance)
      .def_readonly("n_concordant",        &OrderAgreement::nConcordant)
      .def_readonly("n_discordant",        &OrderAgreement::nDiscordant)
      .def_readonly("n_comparable_both",    &OrderAgreement::nComparableBoth)
      .def_readonly("n_only_a",             &OrderAgreement::nOnlyA)
      .def_readonly("n_only_b",             &OrderAgreement::nOnlyB);

  m.def("compare_orders", &tessera::compareOrders,
      py::arg("a"), py::arg("b"), py::arg("n_labels"),
      R"doc(Pairwise agreement statistics between two posets on a shared label set.

Counts unordered pairs (i, j) with i < j in five disjoint buckets via
Floyd–Warshall transitive closures of `a` and `b`:

* concordant   — both orders relate the pair the same way
* discordant   — both orders relate the pair, opposite ways
* only-a       — `a` relates the pair, `b` does not
* only-b       — `b` relates the pair, `a` does not
* neither      — neither order relates the pair

Returns an :class:`OrderAgreement` with ``kendall_tau``,
``discordant_fraction``, ``hasse_edit_distance``, and the five counts.

Complexity: O(nLabels^3) for the transitive closure, O(nLabels^2) for
the pair counts. Practical up to a few thousand labels.

See ``docs/source/causal_sets.md`` for the methodology context.
)doc");

  // ========================================
  // ForceLayout
  // ========================================
  py::class_<ForceLayout>(m, "ForceLayout",
        "Fruchterman-Reingold spring-electrical graph layout.")
      .def_static("layout_3d", &ForceLayout::layout3D,
        py::arg("n"),
        py::arg("edges"),
        py::arg("center_idx") = -1,
        py::arg("init_pos") = std::vector<double>{},
        py::arg("rest_lengths") = std::vector<double>{},
        py::arg("spring_k") = 0.01,
        py::arg("repulsion_k") = 0.5,
        py::arg("iters") = 300,
        py::arg("cooling") = 0.995,
        py::arg("repulsion_cap") = 200,
        py::arg("seed") = 42,
        R"doc(Spring-electrical force-directed layout in 3D.

Returns a flat list of n*3 floats (row-major x,y,z positions).
Reshape to (n, 3) with numpy: ``np.array(result).reshape(n, 3)``.

Args:
    n: Number of nodes.
    edges: List of (i, j) index pairs.
    centerIdx: Pin this node at the origin (-1 to disable).
    initPos: Flat initial positions (length n*3). Random if empty.
    restLengths: Per-edge rest lengths. Unit if empty.
    springK: Spring constant (default 0.01).
    repulsionK: Coulomb constant (default 0.5).
    iters: Number of iterations (default 300).
    cooling: Step-size decay per iteration (default 0.995).
    repulsionCap: Max nodes for O(n^2) repulsion (default 200).
    seed: Random seed (default 42).)doc")
      .def_static("layout_2d", &ForceLayout::layout2D,
        py::arg("n"),
        py::arg("edges"),
        py::arg("target_radii") = std::vector<double>{},
        py::arg("groups") = std::vector<int>{},
        py::arg("center_idx") = -1,
        py::arg("init_pos") = std::vector<double>{},
        py::arg("rest_lengths") = std::vector<double>{},
        py::arg("spring_k") = 0.02,
        py::arg("repulsion_k") = 0.3,
        py::arg("iters") = 200,
        py::arg("cooling") = 0.995,
        py::arg("repulsion_cap") = 200,
        py::arg("initial_step") = 0.5,
        py::arg("seed") = 42,
        R"doc(Spring-electrical force-directed layout in 2D.

Returns a flat list of n*2 floats (row-major x,y positions).
Reshape to (n, 2) with numpy: ``np.array(result).reshape(n, 2)``.

Two optional constraints (independent, may be combined):
  * target_radii (length n) pins each node's radius — only the angle is
    solved (tangential forces, radii re-snapped each step).
  * groups (length n) scopes repulsion to nodes sharing a group id;
    when empty, repulsion is global (capped).

Args:
    n: Number of nodes.
    edges: List of (i, j) index pairs.
    targetRadii: Per-node pinned radius (length n enables radial mode).
    groups: Per-node group id (length n scopes repulsion).
    centerIdx: Pin this node at the origin (-1 to disable).
    initPos: Flat initial positions (length n*2). Random if empty.
    restLengths: Per-edge rest lengths. Unit if empty.
    springK: Spring constant (default 0.02).
    repulsionK: Coulomb constant (default 0.3).
    iters: Number of iterations (default 200).
    cooling: Step-size decay per iteration (default 0.995).
    repulsionCap: Max nodes per repulsion group (default 200).
    initialStep: Initial max displacement per iteration (default 0.5).
    seed: Random seed (default 42).)doc");

#ifdef TESSERA_VERSION
  m.attr("__version__") = TESSERA_VERSION;
#else
  m.attr("__version__") = "unknown";
#endif

  // Register the Schwinger / DMRG bindings as a `quantum` submodule so users
  // call them as `tessera._tessera.quantum.computeGroundState(...)` (typically
  // routed through `tessera.quantum` — see tessera/quantum/__init__.py).
  register_quantum(m.def_submodule("quantum",
      "Schwinger model + DMRG (docs/source/quantum-plan.md)."));
}

