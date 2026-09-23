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
#include "matter/MatterConfiguration.h"
#include "mesh/SimplexFilter.h"
#include "observables/ModularityOptimizer.h"
#include "observables/SparseGraph.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
#include "spacetime/Spacetime.h"
#include "ForceLayout.h"
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
    cdt = tessera.CDTSimulation(st, 2.2, 0.5, 0.6, 0.02, st.getN41())
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
  auto m_chainhodge  = m.def_submodule("chainhodge",
      "Chain-level Whitney Hodge pencil: sparse inverse chain metrics, "
      "branches, and instance certificates.");

  // --- Per-subsystem bindings (one file per subsystem) ---
  register_mesh(m_mesh);
  register_spacetime(m_spacetime);
  register_observables(m_observables);
  register_isospin_doublet(m_observables);
  register_simulations(m_simulations);
  register_cobordism(m_cobordism);
  register_chainhodge(m_chainhodge);

  // ========================================
  // MatterConfiguration
  // ========================================
  py::enum_<HingeType>(m, "HingeType")
      .value("SPATIAL", HingeType::SPATIAL)
      .value("TIMELIKE", HingeType::TIMELIKE);

  py::class_<MatterConfiguration>(m, "MatterConfiguration",
      R"doc(Intrinsic (coordinate-free) specification of stress-energy on a triangulation.

Matter is defined relationally: by assigning energy densities to vertices,
simplices, or as a function of geodesic distance from a reference vertex.

For point particles, the matter action is the proper-time action:
S_matter = -M Σ √(-ℓ²) along the worldline.)doc")
      .def(py::init<>())
      .def("setWorldlineMass", &MatterConfiguration::setWorldlineMass,
           py::arg("center"), py::arg("mass"), py::arg("spacetime"),
           R"doc(Assign a static point mass along its worldline through all time slices.

Traces a worldline from center through the foliation by following
timelike edges.  The matter action is the proper-time action:
S_matter = -M Σ √(-ℓ²) along the worldline.

Args:
    center: A vertex on the worldline (any time slice).
    mass: The mass in geometrized units (G=c=1).
    spacetime: The spacetime to trace through.)doc")
      .def("setEnergyDensity", &MatterConfiguration::setEnergyDensity,
           py::arg("simplex"), py::arg("rho"),
           R"doc(Assign energy density to a top-simplex.

Args:
    simplex: The simplex to assign density to.
    rho: Energy density in geometrized units.)doc")
      .def("setRadialProfile", &MatterConfiguration::setRadialProfile,
           py::arg("center"), py::arg("rhoOfR"),
           R"doc(Assign energy density as a function of geodesic distance.

Args:
    center: The reference vertex.
    rhoOfR: A callable taking distance (float) and returning density (float).)doc")
      .def_static("buildWorldline", &MatterConfiguration::buildWorldline,
           py::arg("center"), py::arg("spacetime"),
           py::return_value_policy::copy,
           R"doc(Trace a worldline from center through all time slices.

Returns a list of vertices, one per time slice, ordered by time.)doc")
      .def_static("classifyHinge", &MatterConfiguration::classifyHinge,
           py::arg("hinge"),
           R"doc(Classify a hinge as SPATIAL (all vertices at one time) or TIMELIKE.)doc");
  // ========================================
  // ForceLayout
  // ========================================
  py::class_<ForceLayout>(m, "ForceLayout",
        "Fruchterman-Reingold spring-electrical graph layout.")
      .def_static("layout3D", &ForceLayout::layout3D,
        py::arg("n"),
        py::arg("edges"),
        py::arg("centerIdx") = -1,
        py::arg("initPos") = std::vector<double>{},
        py::arg("restLengths") = std::vector<double>{},
        py::arg("springK") = 0.01,
        py::arg("repulsionK") = 0.5,
        py::arg("iters") = 300,
        py::arg("cooling") = 0.995,
        py::arg("repulsionCap") = 200,
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
      .def_static("layout2D", &ForceLayout::layout2D,
        py::arg("n"),
        py::arg("edges"),
        py::arg("targetRadii") = std::vector<double>{},
        py::arg("groups") = std::vector<int>{},
        py::arg("centerIdx") = -1,
        py::arg("initPos") = std::vector<double>{},
        py::arg("restLengths") = std::vector<double>{},
        py::arg("springK") = 0.02,
        py::arg("repulsionK") = 0.3,
        py::arg("iters") = 200,
        py::arg("cooling") = 0.995,
        py::arg("repulsionCap") = 200,
        py::arg("initialStep") = 0.5,
        py::arg("seed") = 42,
        R"doc(Spring-electrical force-directed layout in 2D.

Returns a flat list of n*2 floats (row-major x,y positions).
Reshape to (n, 2) with numpy: ``np.array(result).reshape(n, 2)``.

Two optional constraints (independent, may be combined):
  * targetRadii (length n) pins each node's radius — only the angle is
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

