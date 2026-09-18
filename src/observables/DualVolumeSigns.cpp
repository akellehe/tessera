// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Sign audit of primal volumes, circumcentric dual volumes and the diagonal
/// Hodge star built from them.
/// Reference: Desbrun, Hirani, Leok, Marsden, "Discrete Exterior Calculus",
/// arXiv:math/0508341

#include "observables/DualVolumeSigns.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

bool DualVolumeSigns::isAllSpacelike(const Simplex &simplex) {
  for (const auto &edge : simplex.getEdges()) {
    if (!edge) continue;
    if (!edge->isSpacelike()) return false;
  }
  return true;
}

DualVolumeSigns::Report DualVolumeSigns::analyze(
    const std::shared_ptr<Spacetime> &spacetime) const {
  Report report;
  if (!spacetime) return report;

  // Accumulate per dimension; the map keeps dimensions sorted regardless of the
  // order getSimplices returns them in.
  std::map<int, DimensionReport> byDimension;
  std::map<int, double> starRatioSum;

  for (const auto &simplexPtr : spacetime->getSimplices()) {
    if (!simplexPtr) continue;
    const Simplex &simplex = *simplexPtr;

    // Orphans (sub-faces left by a Pachner move with no surviving top coface)
    // are no longer part of the complex and are excluded from the statistics.
    if (!simplex.hasTopCoface()) continue;

    const int nVertices = static_cast<int>(simplex.getVertices().size());
    if (nVertices == 0) continue;
    const int dimension = nVertices - 1;

    DimensionReport &entry = byDimension[dimension];
    entry.dimension = dimension;
    entry.nSimplices += 1;

    const bool allSpacelike = isAllSpacelike(simplex);
    if (allSpacelike) {
      entry.nAllSpacelike += 1;
    } else {
      entry.nMixedSignature += 1;
    }

    // The tallies are defined on real parts. On the real-Lorentzian locus every
    // quantity below is real, so the reads are exact; off that locus the
    // imaginary part is separate information this observable does not report.
    const std::complex<double> dualVolume = simplex.dualVolume();
    if (dualVolume.real() < 0.0) entry.nNegativeDualVolume += 1;

    // A negative barycentric coordinate places the circumcenter outside the
    // simplex on that vertex's side, violating well-centeredness.
    const std::vector<std::complex<double>> barycentric =
        simplex.circumcenterBarycentric();
    if (!barycentric.empty()) {
      const double smallest =
          std::min_element(barycentric.begin(), barycentric.end(),
                           [](const std::complex<double> &a,
                              const std::complex<double> &b) {
                             return a.real() < b.real();
                           })->real();
      if (smallest < 0.0) entry.nCircumcenterOutside += 1;
    }

    // Negative signed circumradius squared means the circumcenter-to-vertex
    // displacement is timelike — reachable only in Lorentzian signature.
    if (simplex.circumradiusSquared().real() < 0.0) entry.nNegativeCircumradius += 1;

    // The diagonal Hodge star entry is the dual-to-primal volume ratio. A
    // near-zero primal volume makes that ratio meaningless, so those simplices
    // are counted separately and excluded from the negative tally and from the
    // ratio statistics.
    const std::complex<double> volume = simplex.volume();
    if (std::abs(volume) <= tolerance_) {
      entry.nDegenerateVolume += 1;
      continue;
    }

    const double starRatio = (dualVolume / volume).real();
    if (starRatio < 0.0) {
      entry.nNegativeStar += 1;
      if (allSpacelike) {
        entry.nNegativeStarAllSpacelike += 1;
      } else {
        entry.nNegativeStarMixedSignature += 1;
      }
    }

    const int nRated = entry.nSimplices - entry.nDegenerateVolume;
    if (nRated == 1) {
      entry.minStarRatio = starRatio;
      entry.maxStarRatio = starRatio;
    } else {
      entry.minStarRatio = std::min(entry.minStarRatio, starRatio);
      entry.maxStarRatio = std::max(entry.maxStarRatio, starRatio);
    }
    starRatioSum[dimension] += starRatio;
  }

  report.dimensions.reserve(byDimension.size());
  for (auto &[dimension, entry] : byDimension) {
    const int nRated = entry.nSimplices - entry.nDegenerateVolume;
    if (nRated > 0) {
      entry.meanStarRatio = starRatioSum[dimension] / static_cast<double>(nRated);
    }
    report.nSimplices += nRated;
    report.nNegativeStar += entry.nNegativeStar;
    report.dimensions.push_back(entry);
  }

  return report;
}

double DualVolumeSigns::compute(const std::shared_ptr<Spacetime> &spacetime) {
  const Report report = analyze(spacetime);
  if (report.nSimplices == 0) return 0.0;
  return static_cast<double>(report.nNegativeStar) /
         static_cast<double>(report.nSimplices);
}

}  // namespace tessera::observables
