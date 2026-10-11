// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Records of the objective: its per-objective contributions and register
// constraints.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_OBJECTIVE_H
#define TESSERA_COBORDISM_MULTICOBORDISM_OBJECTIVE_H

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "cobordism/CobordismObjective.h"

namespace tessera::cobordism::multicobordism {

/// An ordered, explicit period constraint, evaluated by
/// `EigenstateSynthesis::residualForPeriods`. No component permutation is
/// permitted: `target[i]` belongs to `holes[i]`.
struct RegisterConstraint {
  /// Identity; re-declaring this name replaces the constraint.
  std::string name;
  /// Cochain degree the periods are read at.
  int degree{1};
  /// The cycles the periods are taken over, as vertex-id tuples.
  std::vector<std::vector<std::uint64_t>> holes;
  /// One target period per hole, in the same order.
  std::vector<std::complex<double>> target;
};

/// One objective's decomposition, labelled by the objective that produced it
/// and the region it was scored over. The record carries a contribution per
/// objective rather than one summed record, so a reader can tell whether
/// descent came from the bulk or from the pinned region.
struct ObjectiveContribution {
  /// The objective's stable identifier.
  std::string objectiveName;
  /// The declared region, or empty for the whole cobordism.
  std::string regionName;
  /// That objective's terms over its own scope.
  ObjectiveTerms terms;
};

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_OBJECTIVE_H
