// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Pachner-move sweeps that drive the modularity of the dual graph up or down
/// while tracking the spectral dimension.
/// Reference: Newman, "Modularity and community structure in networks",
/// arXiv:physics/0602124

#include "observables/ModularityOptimizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include "observables/SparseGraph.h"
#include "simulations/CDT.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/Spacetime.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

ScanReport ModularityOptimizer::discoverComponents(
    const Spacetime &st, const PersistentModularityConfig &cfg,
    PersistentModularity::WeightMap map) const {
  return PersistentModularity::fromSpacetime(st, map).scanResolutions(cfg);
}

std::unique_ptr<PachnerMove> ModularityOptimizer::proposeAny(CDT &cdt) {
  // Try the five move types in a fresh random order each call.
  std::array<int, 5> order{0, 1, 2, 3, 4};
  std::shuffle(order.begin(), order.end(), rng_);
  for (int t : order) {
    std::unique_ptr<PachnerMove> m;
    switch (t) {
      case 0: m = cdt.proposeAdd(); break;
      case 1: m = cdt.proposeRemove(); break;
      case 2: m = cdt.proposeFlip(); break;
      case 3: m = cdt.proposeIflip(); break;
      case 4: m = cdt.proposeShift(); break;
    }
    if (m) return m;
  }
  return nullptr;
}

ModularityMeasurement ModularityOptimizer::measure(
    CDT &cdt, int iter, const std::string &direction) {
  auto &st = *cdt.getSpacetime();
  double Q = st.modularityOnSkeleton(cfg_.targetNModules);
  // Measure the spectral dimension on the dual graph. A negative reading is
  // retried with fresh random-walk starts.
  SparseGraph g = st.getDualGraph();
  double dsSmall = std::numeric_limits<double>::quiet_NaN();
  double dsLarge = std::numeric_limits<double>::quiet_NaN();
  for (int retry = 0; retry <= cfg_.negativeRetryMax; ++retry) {
    auto [s, l] = g.spectralDimension(
        cfg_.nDiffusionWalks, cfg_.maxSigma, &rng_,
        /*tailFraction=*/0.2, /*nTimes=*/40,
        /*tMin=*/0.5, cfg_.krylovDim);
    dsSmall = s;
    dsLarge = l;
    if (std::isnan(dsSmall) || std::isnan(dsLarge)) break;
    if (dsSmall >= 0 && dsLarge >= 0) break;
  }
  return ModularityMeasurement{
      Q, dsSmall, dsLarge,
      st.getVertexCount(),
      st.getEdgeList()->size(),
      st.getSimplexCount(),
      iter, direction
  };
}

std::vector<ModularityMeasurement> ModularityOptimizer::sweep(
    CDT &cdt, const std::string &direction, ProgressCallback progress) {
  if (direction != "up" && direction != "down") {
    return {};
  }
  // Reset per-sweep counters.
  nAccepted_ = 0;
  nRolledBack_ = 0;
  nNoMove_ = 0;
  nMeasurements_ = 0;

  std::vector<ModularityMeasurement> measurements;
  measurements.reserve(static_cast<std::size_t>(cfg_.maxIterations / 4 + 4));

  // Initial measurement (iter 0).
  auto initial = measure(cdt, 0, direction);
  measurements.push_back(initial);
  nMeasurements_ = 1;
  double currentQ = initial.Q;

  // Maximum modularity of a balanced M-partition, 1 - 1/M; the up-sweep exits
  // early once it is within epsilon of this.
  double qMaxTarget = 0.0;
  if (cfg_.targetNModules > 0) {
    qMaxTarget = 1.0 - 1.0 / static_cast<double>(cfg_.targetNModules);
  }

  double sign = (direction == "up") ? +1.0 : -1.0;
  double nextThreshold = currentQ + sign * cfg_.targetDq;

  if (progress) {
    progress(0, cfg_.maxIterations, currentQ, measurements.size());
  }

  for (int it = 1; it <= cfg_.maxIterations; ++it) {
    auto move = proposeAny(cdt);
    if (!move) {
      ++nNoMove_;
      if (progress) progress(it, cfg_.maxIterations,
                             currentQ, measurements.size());
      continue;
    }

    // Modularity before the move is applied.
    double qBefore =
        cdt.getSpacetime()->modularityOnSkeleton(cfg_.targetNModules);
    if (!move->apply()) {
      ++nNoMove_;
      if (progress) progress(it, cfg_.maxIterations,
                             currentQ, measurements.size());
      continue;
    }
    double qAfter =
        cdt.getSpacetime()->modularityOnSkeleton(cfg_.targetNModules);

    // Accept only moves that push modularity in the requested direction.
    bool accepted = (direction == "up") ? (qAfter >= qBefore)
                                        : (qAfter <= qBefore);
    if (!accepted) {
      move->rollback();
      ++nRolledBack_;
      if (progress) progress(it, cfg_.maxIterations,
                             currentQ, measurements.size());
      continue;
    }

    ++nAccepted_;
    currentQ = qAfter;
    bool crossed = (direction == "up") ? (currentQ >= nextThreshold)
                                       : (currentQ <= nextThreshold);
    if (crossed) {
      auto m = measure(cdt, it, direction);
      measurements.push_back(m);
      ++nMeasurements_;
      currentQ = m.Q;
      nextThreshold = currentQ + sign * cfg_.targetDq;
    }

    // Up-sweep early exit near the maximum modularity.
    if (direction == "up" && cfg_.epsilonQMax > 0.0
        && currentQ >= qMaxTarget - cfg_.epsilonQMax) {
      if (progress) progress(it, cfg_.maxIterations,
                             currentQ, measurements.size());
      break;
    }

    if (progress) {
      progress(it, cfg_.maxIterations, currentQ, measurements.size());
    }
  }

  return measurements;
}

}  // namespace tessera::observables
