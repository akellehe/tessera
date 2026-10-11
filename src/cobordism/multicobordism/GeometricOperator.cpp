// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the geometric
// operator readout. One of the translation units that define the class's
// members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

GeometricOperatorReadout MultiCobordism::geometricOperator(
    int stateDimension, std::vector<std::vector<std::uint64_t>> frameCells,
    double tol, bool metric) const {
  return geometricOperatorOn(spacetime_, stateDimension, std::move(frameCells), tol, metric);
}

GeometricOperatorReadout MultiCobordism::geometricOperatorOn(
    const std::shared_ptr<Spacetime> &spacetime, int stateDimension,
    std::vector<std::vector<std::uint64_t>> frameCells, double tol,
    bool metric) const {
  return EigenstateSynthesis::geometricOperator(spacetime, stateDimension, std::move(frameCells),
                                                tol, metric, metricSource_);
}

}  // namespace tessera::cobordism
