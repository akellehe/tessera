// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the entry point. The bound
// classes live in bindings/*Bindings.cpp, one translation unit per family,
// registered here in the order the single unit used to register them, so a
// base class precedes its derived classes and the type of every default
// argument is bound before a signature uses it
// (https://github.com/akellehe/tessera/issues/1453).

#include "bindings/BindingsCommon.h"

void register_cobordism_chain_complex(py::module_ &m);
void register_cobordism_hodge(py::module_ &m);
void register_cobordism_integer_linalg(py::module_ &m);
void register_cobordism_pencil(py::module_ &m);
void register_cobordism_multi_cobordism_read(py::module_ &m);
void register_cobordism_multi_cobordism(py::module_ &m);
void register_cobordism_dag_proton_certificate(py::module_ &m);
void register_cobordism_analytic_cache(py::module_ &m);
void register_cobordism_recursive_quotient(py::module_ &m);
void register_cobordism_joint_action(py::module_ &m);
void register_cobordism_holomorphic_relaxation(py::module_ &m);
void register_cobordism_self_consistent_mean_field(py::module_ &m);
void register_cobordism_ward_flux_bound_state(py::module_ &m);
void register_cobordism_dressed_fluctuation(py::module_ &m);
void register_cobordism_level_recursion(py::module_ &m);

void register_cobordism(py::module_ m) {
  // Smoke hook: lets tests assert the subsystem loaded. Single leading
  // underscore, to avoid Python name-mangling inside test classes.
  m.def("_cobordism_smoke", [] { return true; },
        "Returns True; confirms the cobordism subsystem is built and importable.");

  register_cobordism_chain_complex(m);
  register_cobordism_hodge(m);
  register_cobordism_integer_linalg(m);
  register_cobordism_pencil(m);
  register_cobordism_multi_cobordism_read(m);
  register_cobordism_multi_cobordism(m);
  register_cobordism_dag_proton_certificate(m);
  register_cobordism_analytic_cache(m);
  register_cobordism_recursive_quotient(m);
  register_cobordism_joint_action(m);
  register_cobordism_holomorphic_relaxation(m);
  register_cobordism_self_consistent_mean_field(m);
  register_cobordism_ward_flux_bound_state(m);
  register_cobordism_dressed_fluctuation(m);
  register_cobordism_level_recursion(m);
}
