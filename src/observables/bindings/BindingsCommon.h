#pragma once
// The shared prelude of the observables bindings: the includes, the
// namespace aliases and the helpers that every *Bindings.cpp translation
// unit in this directory uses. The bound classes themselves live in those
// units, one family each, and Bindings.cpp in the parent directory registers
// them in order (https://github.com/akellehe/tessera/issues/1453).

// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
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
#include "observables/EffectiveTopology.h"
#include "observables/ModularityOptimizer.h"
#include "observables/PersistentModularity.h"
#include "observables/SpectralFiber.h"
#include "observables/SparseGraph.h"
#include "cobordism/AnalyticCache.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
#include "observables/Spectral.h"
#include "observables/SimplicialQubit.h"
#include "chainhodge/CovariantChainHodge.h"
#include "observables/Record.h"
#include "observables/ClusterRegister.h"
#include "observables/RegisterContext.h"
#include "observables/RegisterObservable.h"
#include "observables/InteriorHinges.h"
#include "observables/LiveComplex.h"
#include "observables/SingletResidual.h"
#include "observables/BlockResiduals.h"
#include "observables/EmergentMass.h"
#include "observables/EmergentRadius.h"
#include "observables/PairLoopFlavor.h"
#include "observables/ObservableGates.h"
#include "observables/DualVolumeSigns.h"
#include "observables/ColorFiber.h"
#include "observables/SheetedColor.h"
#include "observables/ComplexTransport.h"
#include "observables/MonopoleSpin.h"
#include "observables/QuarkConditions.h"
#include "observables/CrossingReadouts.h"
#include "observables/ExchangeHolonomy.h"
#include "observables/FiberConnection.h"
#include "observables/ParticleClusters.h"
#include "observables/ClusterLineage.h"
#include "cobordism/ProtonSynthesis.h"
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

// Background for the observables bound here:
//   Ambjorn, Goerlich, Jurkiewicz, Loll, "Nonperturbative Quantum Gravity",
//   arXiv:1203.3591 -- causal dynamical triangulations, volume profiles.
//   Newman, "Modularity and community structure in networks",
//   arXiv:physics/0602124 -- modularity and the leading-eigenvector method.

namespace py = pybind11;
using namespace tessera;
using namespace tessera::observables;

namespace {

// A JSON-able observable Record -> a native Python object (dict/list/scalar).
py::object recordToPython(const Record &r) {
  switch (r.type()) {
    case Record::Type::Null:
      return py::none();
    case Record::Type::Bool:
      return py::bool_(r.asBool());
    case Record::Type::Int:
      return py::int_(static_cast<long long>(r.asInt()));
    case Record::Type::Double:
      return py::float_(r.asDouble());
    case Record::Type::String:
      return py::str(r.asString());
    case Record::Type::List: {
      py::list out;
      for (const auto &e : r.asList()) out.append(recordToPython(e));
      return out;
    }
    case Record::Type::Map: {
      py::dict out;
      for (const auto &kv : r.asMap()) {
        out[py::str(kv.first)] = recordToPython(kv.second);
      }
      return out;
    }
  }
  return py::none();
}

// A native Python object -> a Record (for report_delta testing). bool is checked
// before int (Python bool is an int subtype).
Record pythonToRecord(const py::handle &o) {
  if (o.is_none()) return Record();
  if (py::isinstance<py::bool_>(o)) return Record(o.cast<bool>());
  if (py::isinstance<py::int_>(o)) {
    return Record(static_cast<std::int64_t>(o.cast<long long>()));
  }
  if (py::isinstance<py::float_>(o)) return Record(o.cast<double>());
  if (py::isinstance<py::str>(o)) return Record(o.cast<std::string>());
  if (py::isinstance<py::dict>(o)) {
    Record::Map m;
    for (const auto &item : o.cast<py::dict>()) {
      m[item.first.cast<std::string>()] = pythonToRecord(item.second);
    }
    return Record(std::move(m));
  }
  if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
    Record::List l;
    for (const auto &e : o) l.push_back(pythonToRecord(e));
    return Record(std::move(l));
  }
  throw std::runtime_error(
      "report_delta: record leaves must be dict/list/str/float/int/bool/None");
}

// Emit the RegisterContext's surplus-selection warning as a Python UserWarning
// (the header's documented binding behavior).
void emitSelectionWarning(const RegisterContext &ctx) {
  if (!ctx.selectionWarning().empty()) {
    auto warnings = py::module_::import("warnings");
    warnings.attr("warn")(ctx.selectionWarning());
  }
}

}  // namespace

// Registers all tessera::observables classes into the `m` submodule
// (i.e. `tessera.observables`). Called from src/bindings.cpp's
// PYBIND11_MODULE entry point.
