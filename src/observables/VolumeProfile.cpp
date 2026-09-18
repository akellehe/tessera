// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Spatial volume as a function of discrete time, and its peak-centred average
/// over measurements.
/// Reference: Ambjorn, Jurkiewicz, Loll, "Reconstructing the Universe",
/// arXiv:hep-th/0505154

#include "observables/VolumeProfile.h"
#include <algorithm>

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

double VolumeProfile::compute(const std::shared_ptr<Spacetime> &spacetime) {
  int topSize = spacetime->getMetric()->getSignature()->getDimensions() + 1;
  std::map<int, int> profile;
  for (const auto &s : spacetime->getSimplices()) {
    if (static_cast<int>(s->size()) != topSize) continue;
    int tMin = static_cast<int>(s->getTi());
    profile[tMin]++;
  }
  if (profile.empty()) {
    currentProfile.clear();
    return 0.0;
  }
  int tMinKey = profile.begin()->first;
  int tMaxKey = profile.rbegin()->first;
  currentProfile.assign(tMaxKey - tMinKey + 1, 0);
  for (const auto &[t, count] : profile) {
    currentProfile[t - tMinKey] = count;
  }
  // The peak spatial volume over the time slices.
  return static_cast<double>(*std::max_element(currentProfile.begin(), currentProfile.end()));
}

double VolumeProfile::update(const std::shared_ptr<Spacetime> &spacetime) {
  return compute(spacetime);
}

const std::vector<int> &VolumeProfile::getProfile() const {
  return currentProfile;
}

void VolumeProfile::measure(const std::shared_ptr<Spacetime> &spacetime) {
  compute(spacetime);
  measurements.push_back(currentProfile);
}

std::vector<double> VolumeProfile::getAverageProfile() const {
  if (measurements.empty()) return {};
  // Longest recorded profile.
  std::size_t maxLen = 0;
  for (const auto &m : measurements) maxLen = std::max(maxLen, m.size());
  std::vector<double> avg(maxLen, 0.0);
  for (const auto &m : measurements) {
    for (std::size_t i = 0; i < m.size(); ++i) {
      avg[i] += m[i];
    }
  }
  for (auto &v : avg) v /= static_cast<double>(measurements.size());
  return avg;
}

std::vector<double> VolumeProfile::getCenteredAverageProfile(
    bool subtractStalk, bool normalizePeak) const {
  std::vector<std::vector<double>> profiles;
  profiles.reserve(measurements.size());
  for (const auto &m : measurements)
    profiles.emplace_back(m.begin(), m.end());
  return centeredAverage(profiles, subtractStalk, normalizePeak);
}

std::vector<double> VolumeProfile::centeredAverage(
    const std::vector<std::vector<double>> &profiles,
    bool subtractStalk, bool normalizePeak) {
  if (profiles.empty()) return {};

  std::size_t maxLen = 0;
  for (const auto &p : profiles) maxLen = std::max(maxLen, p.size());
  if (maxLen == 0) return {};

  std::vector<double> avg(maxLen, 0.0);
  std::vector<double> arr(maxLen, 0.0);
  const std::size_t mid = maxLen / 2;

  for (const auto &p : profiles) {
    // Zero-pad to the common length.
    std::fill(arr.begin(), arr.end(), 0.0);
    for (std::size_t i = 0; i < p.size(); ++i) arr[i] = p[i];

    // Remove the constant stalk volume (min of the padded profile).
    if (subtractStalk) {
      double stalk = *std::min_element(arr.begin(), arr.end());
      for (auto &v : arr) v -= stalk;
    }

    // Circularly roll so the peak (the first maximum) sits at maxLen/2:
    // result[(i + shift) mod n] = arr[i].
    std::size_t peakIdx = static_cast<std::size_t>(
        std::max_element(arr.begin(), arr.end()) - arr.begin());
    std::size_t shift = (mid + maxLen - peakIdx % maxLen) % maxLen;
    for (std::size_t i = 0; i < maxLen; ++i)
      avg[(i + shift) % maxLen] += arr[i];
  }

  for (auto &v : avg) v /= static_cast<double>(profiles.size());

  if (normalizePeak) {
    double peak = *std::max_element(avg.begin(), avg.end());
    if (peak > 0.0)
      for (auto &v : avg) v /= peak;
  }
  return avg;
}

void VolumeProfile::reset() {
  currentProfile.clear();
  measurements.clear();
}

} // namespace tessera::observables
