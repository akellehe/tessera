// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: the entry point. The bound
// classes live in bindings/*Bindings.cpp, one translation unit per family,
// registered here in the order the single unit used to register them, so a
// base class precedes its derived classes and the type of every default
// argument is bound before a signature uses it
// (https://github.com/akellehe/tessera/issues/1453).

#include "bindings/BindingsCommon.h"

void register_observables_effective_topology(py::module_ &m);
void register_observables_graph_modularity(py::module_ &m);
void register_observables_spectral_fiber(py::module_ &m);
void register_observables_modularity_volume_simplicial_qubit(py::module_ &m);
void register_observables_spectral_gap_wilson(py::module_ &m);
void register_observables_register_observable(py::module_ &m);
void register_observables_dual_volume_sheet(py::module_ &m);
void register_observables_complex_transport_monopole(py::module_ &m);
void register_observables_color_fiber(py::module_ &m);
void register_observables_exchange_holonomy(py::module_ &m);
void register_observables_fiber_connection(py::module_ &m);
void register_observables_particle_clusters(py::module_ &m);
void register_observables_crossing_register_lineage(py::module_ &m);

void register_observables(py::module_ m) {
  register_observables_effective_topology(m);
  register_observables_graph_modularity(m);
  register_observables_spectral_fiber(m);
  register_observables_modularity_volume_simplicial_qubit(m);
  register_observables_spectral_gap_wilson(m);
  register_observables_register_observable(m);
  register_observables_dual_volume_sheet(m);
  register_observables_complex_transport_monopole(m);
  register_observables_color_fiber(m);
  register_observables_exchange_holonomy(m);
  register_observables_fiber_connection(m);
  register_observables_particle_clusters(m);
  register_observables_crossing_register_lineage(m);
}
