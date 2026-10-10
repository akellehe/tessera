// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Pybind11 bindings for the matter subsystem (include/matter/), registered
// into the tessera.matter submodule by src/bindings.cpp.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "matter/MatterConfiguration.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace py = pybind11;
using namespace tessera::matter;

void register_matter(py::module_ m) {
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
      .def("set_worldline_mass", &MatterConfiguration::setWorldlineMass,
           py::arg("center"), py::arg("mass"), py::arg("spacetime"),
           R"doc(Assign a static point mass along its worldline through all time slices.

Traces a worldline from center through the foliation by following
timelike edges.  The matter action is the proper-time action:
S_matter = -M Σ √(-ℓ²) along the worldline.

Args:
    center: A vertex on the worldline (any time slice).
    mass: The mass in geometrized units (G=c=1).
    spacetime: The spacetime to trace through.)doc")
      .def("set_energy_density", &MatterConfiguration::setEnergyDensity,
           py::arg("simplex"), py::arg("rho"),
           R"doc(Assign energy density to a top-simplex.

Args:
    simplex: The simplex to assign density to.
    rho: Energy density in geometrized units.)doc")
      .def("set_radial_profile", &MatterConfiguration::setRadialProfile,
           py::arg("center"), py::arg("rho_of_r"),
           R"doc(Assign energy density as a function of geodesic distance.

Args:
    center: The reference vertex.
    rhoOfR: A callable taking distance (float) and returning density (float).)doc")
      .def_static("build_worldline", &MatterConfiguration::buildWorldline,
           py::arg("center"), py::arg("spacetime"),
           py::return_value_policy::copy,
           R"doc(Trace a worldline from center through all time slices.

Returns a list of vertices, one per time slice, ordered by time.)doc")
      .def_static("classify_hinge", &MatterConfiguration::classifyHinge,
           py::arg("hinge"),
           R"doc(Classify a hinge as SPATIAL (all vertices at one time) or TIMELIKE.)doc");
}
