#pragma once
// The shared prelude of the cobordism bindings: the includes, the namespace
// aliases and the helpers that every *Bindings.cpp translation unit in this
// directory uses. The bound classes themselves live in those units, one
// family each, and Bindings.cpp in the parent directory registers them in
// order (https://github.com/akellehe/tessera/issues/1453).

// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Pybind11 bindings for the cobordism subsystem. Lives outside tessera_core
// (which is pybind-free) so the static library can be reused without the
// Python dependency. Always added to _tessera's sources.

#include <limits>
#include <optional>
#include <tuple>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/AnalyticCache.h"
#include "cobordism/BoundStatePole.h"
#include "cobordism/Certificate.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/Characteristic.h"
#include "cobordism/DenseReference.h"
#include "cobordism/DressedFluctuation.h"
#include "cobordism/KuennethProduct.h"
#include "cobordism/LevelRecursion.h"
#include "cobordism/MappingCylinder.h"
#include "cobordism/SpacetimeComposition.h"
#include "cobordism/LowRankUpdate.h"
#include "cobordism/OccupationSpectra.h"
#include "cobordism/Cochain.h"
#include "cobordism/CombinatorialDimension.h"
#include "cobordism/CobordismDAG.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/MultiCobordism.h"
#include "observables/SimplicialQubit.h"
#include "cobordism/PencilLayer.h"
#include "cobordism/ProtonSynthesis.h"
#include "cobordism/ProtonIngredients.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/HolomorphicRelaxation.h"
#include "cobordism/IntegerLinalg.h"
#include "cobordism/JointAction.h"
#include "cobordism/SelfConsistentMeanField.h"
#include "cobordism/RecursiveQuotient.h"
#include "cobordism/SurgicalCone.h"
#include "cobordism/WardFlux.h"
#include "cobordism/Spectrum.h"
#include "spacetime/Spacetime.h"  // complete type required by pybind (typeid)

namespace py = pybind11;

using namespace tessera;
using namespace tessera::cobordism;

/// # PyCobordismObjective
///
/// The trampoline that lets a Python subclass supply the functional
/// `MultiCobordism` descends. Each override forwards to the Python method of
/// the corresponding snake_case name; the three with C++ defaults fall back to
/// the base implementation when a subclass does not define them.
///
/// The firewall survives the crossing: a Python objective is handed an
/// `ObjectiveContext`, plain data carrying geometry, a region, that region's
/// targets and scalar configuration. It receives no node and no callable that
/// closes over one, so no route to a component, fiber, transport, colour,
/// charge, flavour, exchange, spin certificate or verdict.
///
/// The override macros acquire the GIL themselves, so an engine entry point
/// that released it re-enters Python safely.
class PyCobordismObjective : public CobordismObjective {
 public:
  using CobordismObjective::CobordismObjective;

  [[nodiscard]] std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, CobordismObjective, name);
  }

  [[nodiscard]] std::vector<std::string> termNames() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::vector<std::string>, CobordismObjective,
                                "term_names", termNames);
  }

  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override {
    PYBIND11_OVERRIDE_PURE(ObjectiveTerms, CobordismObjective, terms, context);
  }

  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override {
    PYBIND11_OVERRIDE_PURE(ObjectiveDirection, CobordismObjective, direction,
                           context);
  }

  [[nodiscard]] bool isTargetConditioned() const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, CobordismObjective,
                                "is_target_conditioned", isTargetConditioned);
  }

  [[nodiscard]] ObjectiveScope scope() const override {
    PYBIND11_OVERRIDE(ObjectiveScope, CobordismObjective, scope);
  }

  [[nodiscard]] bool needsRegisterResidual() const override {
    PYBIND11_OVERRIDE_NAME(bool, CobordismObjective, "needs_register_residual",
                           needsRegisterResidual);
  }

  [[nodiscard]] double numericalRegisterResidualWeight(
      const ObjectiveContext &context) const override {
    PYBIND11_OVERRIDE_NAME(double, CobordismObjective,
                           "numerical_register_residual_weight",
                           numericalRegisterResidualWeight, context);
  }

  [[nodiscard]] std::vector<HodgeDegreeContribution> hodgeDegreeContributions(
      const ObjectiveContext &context) const override {
    PYBIND11_OVERRIDE_NAME(std::vector<HodgeDegreeContribution>,
                           CobordismObjective, "hodge_degree_contributions",
                           hodgeDegreeContributions, context);
  }
};

