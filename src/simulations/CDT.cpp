// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "simulations/CDT.h"
#include "Logger.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include "spacetime/pachner/ShiftMove.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>

namespace { // anonymous — local to this TU
template<typename T, std::size_t Cap = 8>
struct StackVec {
  std::array<T, Cap> data_{};
  std::uint8_t len_ = 0;
  void push_back(T v) noexcept { if (len_ < Cap) data_[len_++] = v; }
  T  operator[](std::size_t i) const noexcept { return data_[i]; }
  T &operator[](std::size_t i)       noexcept { return data_[i]; }
  std::size_t size()  const noexcept { return len_; }
  const T *begin() const noexcept { return data_.data(); }
  const T *end()   const noexcept { return data_.data() + len_; }
};
} // anon

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;

CDT::CDT(std::shared_ptr<Spacetime> spacetime_, double k0_, double k4_, double delta_,
         double epsilon_, std::size_t targetN41_, bool quadraticVolumeFix_)
    : spacetime(std::move(spacetime_)), k0(k0_), k4(k4_), delta(delta_),
      epsilon(epsilon_), targetN41(targetN41_), quadraticVolumeFix(quadraticVolumeFix_) {}

static int getDim(const std::shared_ptr<Spacetime> &st) {
  return st->getMetric()->getSignature()->getDimensions();
}

/// Check that a proposed simplex vertex set has a valid CDT orientation:
/// (d,1), (1,d), (d-1,2), or (2,d-1), AND spans exactly 2 time slices.
static bool isValidCDTOrientation(const VertexPtrs &verts, int d) {
  // Must span exactly 2 distinct times (CDT causality constraint)
  std::unordered_set<std::uint64_t> times;
  for (const auto &v : verts) {
    // Use floor cast (consistent with volume profile time binning)
    times.insert(static_cast<std::uint64_t>(v->getTime()));
  }
  if (times.size() != 2) return false;

  auto orient = TemporalOrientation::orientationOf(verts);
  auto [ti, tf] = orient.numeric();
  if ((ti == d && tf == 1) || (ti == 1 && tf == d)) return true;
  if ((ti == d - 1 && tf == 2) || (ti == 2 && tf == d - 1)) return true;
  return false;
}

static bool isN41Type(const SimplexPtr &s, int d) {
  auto [ti, tf] = s->getOrientation().numeric();
  return (ti == d && tf == 1) || (ti == 1 && tf == d);
}


/// Select a uniformly random N41-type top simplex.
/// Uses rejection sampling with a fallback linear scan.
SimplexPtr CDT::getRandomN41Simplex(int d) {
  int dPlus1 = d + 1;
  // Fast path: rejection sampling (effective when N41/N4 is not too small)
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto s = spacetime->getRandomTopSimplex();
    if (s && static_cast<int>(s->size()) == dPlus1 && isN41Type(s, d)) return s;
  }
  // Fallback: linear scan
  std::vector<SimplexPtr> matches;
  for (const auto &s : spacetime->getSimplices()) {
    if (static_cast<int>(s->size()) == dPlus1 && isN41Type(s, d))
      matches.push_back(s);
  }
  if (matches.empty()) return nullptr;
  std::uniform_int_distribution<std::size_t> dist(0, matches.size() - 1);
  return matches[dist(rng)];
}

// ========================================
// Action Computation
// ========================================

double CDT::computeAction() const {
  auto n0 = static_cast<double>(spacetime->getVertexCount());
  auto n41 = static_cast<double>(spacetime->getN41());
  auto n32 = static_cast<double>(spacetime->getN32());

  double regge = -(k0 + 6.0 * delta) * n0
               + (k4 + 2.0 * delta) * n41
               + (k4 + delta) * n32;
  double target = static_cast<double>(targetN41);
  double volumeFix;
  if (quadraticVolumeFix) {
    volumeFix = epsilon * (n41 - target) * (n41 - target);
  } else {
    volumeFix = epsilon * std::abs(n41 - target);
  }
  return regge + volumeFix;
}

// computeDeltaAction and accept are inlined in CDT.h

// ========================================
// (2, 2d) Add Move: vertex insertion at spatial face
// ========================================
bool CDT::add() {
  addAttempts++;
  AddMove move(spacetime.get(), &rng, relabelVertices_);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  addAccepted++;
  return true;
}

// ========================================
// (2d, 2) Remove Move: vertex deletion (blind guessing)
// ========================================
bool CDT::remove() {
  removeAttempts++;
  RemoveMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  removeAccepted++;
  return true;
}

// ========================================
// (2, d) Flip Move
// ========================================
bool CDT::flip() {
  flipAttempts++;
  FlipMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  flipAccepted++;
  return true;
}

// ========================================
// (d, 2) Inverse Flip Move
// ========================================
bool CDT::iflip() {
  iflipAttempts++;
  IFlipMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  if (!move.apply()) return false;
  iflipAccepted++;
  return true;
}

// ========================================
// (3, 3) Shift (self-inverse)
// ========================================
bool CDT::shift() {
  shiftAttempts++;
  if (shiftImpl()) {
    shiftAccepted++;
    return true;
  }
  return false;
}

bool CDT::ishift() {
  ishiftAttempts++;
  if (shiftImpl()) {
    ishiftAccepted++;
    return true;
  }
  return false;
}

bool CDT::shiftImpl() {
  ShiftMove move(spacetime.get(), &rng);
  if (!move.propose()) return false;
  double deltaS = computeDeltaAction(move.dN0(), move.dN41(), move.dN32());
  if (!accept(deltaS, move.metropolisLogPrefactor())) return false;
  return move.apply();
}

// ========================================
// Transactional move factories
// ========================================
//
// These hand the caller a fresh PachnerMove already bound to this
// simulation's spacetime and Markov-chain RNG.  Useful for the
// modularity-sweep optimizer (observables/ModularityOptimizer.h),
// which needs to layer custom acceptance (Q-direction filter) on top
// of the bare move mechanics.  Each factory calls ``propose()``;
// returns nullptr if no eligible target.

std::unique_ptr<PachnerMove> CDT::proposeAdd() {
  auto m = std::make_unique<AddMove>(spacetime.get(), &rng, relabelVertices_);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeRemove() {
  auto m = std::make_unique<RemoveMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeFlip() {
  auto m = std::make_unique<FlipMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeIflip() {
  auto m = std::make_unique<IFlipMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

std::unique_ptr<PachnerMove> CDT::proposeShift() {
  auto m = std::make_unique<ShiftMove>(spacetime.get(), &rng);
  if (!m->propose()) return nullptr;
  return m;
}

// ========================================
// Metropolis Sweep
// ========================================

int CDT::sweep() {
  int n4 = static_cast<int>(spacetime->getSimplexCount());
  if (n4 <= 0) n4 = 1;
  int accepted = 0;

  std::uniform_int_distribution<int> moveDist(0, 4);
  for (int i = 0; i < n4; ++i) {
    int moveType = moveDist(rng);
    bool result = false;
    switch (moveType) {
      case 0: result = add(); break;
      case 1: result = remove(); break;
      case 2: result = flip(); break;
      case 3: result = iflip(); break;
      case 4: result = shift(); break;
    }
    if (result) accepted++;
  }
  return accepted;
}

double CDT::measureVolumeDrift(int windowSweeps, std::size_t floorVolume,
                               std::size_t ceilingVolume) {
  // Least-squares slope of volume against sweep number. Sums are kept in the
  // centred form so the fit needs one pass and no storage.
  double n = 0.0, sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
  auto observe = [&](double x, double y) {
    n += 1.0; sumX += x; sumY += y; sumXX += x * x; sumXY += x * y;
  };
  observe(0.0, static_cast<double>(spacetime->getN41() + spacetime->getN32()));
  for (int i = 1; i <= windowSweeps; ++i) {
    sweep();
    const std::size_t now = spacetime->getN41() + spacetime->getN32();
    observe(static_cast<double>(i), static_cast<double>(now));
    if (now < floorVolume || now > ceilingVolume) break;
  }
  const double denominator = n * sumXX - sumX * sumX;
  const double mean = sumY / n;
  if (denominator <= 0.0 || mean <= 0.0) return 0.0;
  const double slope = (n * sumXY - sumX * sumY) / denominator;
  return slope / mean;
}

void CDT::locatePseudoCriticalCoupling(int windowSweeps, int bisectionSteps,
                                      double tolerance,
                                      const std::function<void()> &report) {
  // Criticality is a property of k4 against the entropy of the triangulations,
  // so the volume-fixing term plays no part in locating it.
  const double configuredEpsilon = epsilon;
  epsilon = 0.0;

  // Hold the configuration inside a band around the volume this search starts
  // at: a coupling far below critical inflates the complex, one far above
  // dismantles it, and the sign of the drift is settled long before the volume
  // leaves the band.
  const double entryVolume =
      static_cast<double>(spacetime->getN41() + spacetime->getN32());
  const std::size_t floorVolume = std::max<std::size_t>(
      static_cast<std::size_t>(entryVolume * (1.0 - kTuneVolumeBand)), 1);
  const std::size_t ceilingVolume = std::max<std::size_t>(
      static_cast<std::size_t>(entryVolume * (1.0 + kTuneVolumeBand)),
      floorVolume + 1);

  // Bracket the drift sign change. Below the pseudo-critical coupling the
  // volume grows and k4 has to rise; above it the volume shrinks.
  double drift = measureVolumeDrift(windowSweeps, floorVolume, ceilingVolume);
  report();
  double below = k4, above = k4;
  double width = 1.0;
  const bool startsBelowCritical = drift > 0.0;
  for (int i = 1; i < kTuneMaxBracketSteps; ++i) {
    if (startsBelowCritical) {
      below = k4;
      k4 += width;
    } else {
      above = k4;
      k4 -= width;
    }
    width *= 2.0;
    drift = measureVolumeDrift(windowSweeps, floorVolume, ceilingVolume);
    report();
    if (startsBelowCritical ? (drift <= 0.0) : (drift > 0.0)) {
      (startsBelowCritical ? above : below) = k4;
      break;
    }
  }

  // If the sign never changed the bracket is open on one side and the last k4
  // tried is the best estimate available, so the bisection is skipped.
  if (below < above) {
    int window = windowSweeps;
    for (int i = 0; i < bisectionSteps && above - below > tolerance; ++i) {
      k4 = 0.5 * (below + above);
      window = std::min(2 * window, kTuneMaxWindowSweeps);
      drift = measureVolumeDrift(window, floorVolume, ceilingVolume);
      report();
      if (drift > 0.0) below = k4;
      else above = k4;
    }
    k4 = 0.5 * (below + above);
  }

  epsilon = configuredEpsilon;
}

void CDT::tune(std::function<void(int,int)> progress) {
  int d = getDim(spacetime);
  if (d <= 1) return;  // CDT requires d >= 2

  // The action's per-simplex cost alone puts k4 here: it is where a single
  // (2,2d) add move has dS_Regge = -(k0+6Δ) + (2d-2)(k4+2Δ) = 0. That ignores
  // the entropy of the triangulations reachable at this volume, which is what
  // actually sets the pseudo-critical coupling, so this value only starts the
  // search (#965).
  k4 = (k0 + 6.0 * delta) / (2.0 * d - 2.0) - 2.0 * delta;

  const int totalSteps = kTuneMaxBracketSteps + kTuneBisectionSteps;
  int step = 0;
  auto report = [&progress, &step, totalSteps] {
    if (progress) progress(std::min(++step, totalSteps), totalSteps);
  };

  // The search runs at the volume the complex was built at. The pseudo-critical
  // coupling does depend on the volume, but weakly -- measured, it moves by
  // -0.023 between N4 = 1.5k and N4 = 6k -- while searching at the target volume
  // instead measures the (3,2) sector relaxing toward its equilibrium, which is
  // a transient over thousands of sweeps and not a property of the coupling.
  locatePseudoCriticalCoupling(kTuneWindowSweeps, kTuneBisectionSteps,
                               kTuneTolerance, report);

  if (progress) progress(totalSteps, totalSteps);
}

void CDT::thermalize() {
  double prevAction = computeAction();
  for (int i = 0; i < 200; ++i) {
    sweep();
    double action = computeAction();
    if (std::abs(action - prevAction) / (std::abs(prevAction) + 1e-10) < 0.01) {
      if (i > 20) return;
    }
    prevAction = action;
  }
}

// ========================================
// Observables
// ========================================

std::vector<int> CDT::getVolumeProfile() const {
  int d = getDim(spacetime);
  std::size_t dPlus1 = static_cast<std::size_t>(d + 1);
  std::map<int, int> profile;
  for (const auto &s : spacetime->getSimplices()) {
    if (s->size() != dPlus1) continue;
    int tMin = static_cast<int>(s->getTi());
    profile[tMin]++;
  }
  if (profile.empty()) return {};
  int tMax = profile.rbegin()->first;
  int tMin = profile.begin()->first;
  std::vector<int> result(tMax - tMin + 1, 0);
  for (const auto &[t, count] : profile) {
    result[t - tMin] = count;
  }
  return result;
}

std::map<std::string, double> CDT::getAcceptanceRates() const {
  auto rate = [](std::int64_t accepted, std::int64_t attempted) -> double {
    return attempted > 0 ? static_cast<double>(accepted) / static_cast<double>(attempted) : 0.0;
  };
  return {
    {::tessera::spacetime::AddMove::kMoveType, rate(addAccepted, addAttempts)},
    {::tessera::spacetime::RemoveMove::kMoveType, rate(removeAccepted, removeAttempts)},
    {::tessera::spacetime::FlipMove::kMoveType, rate(flipAccepted, flipAttempts)},
    {::tessera::spacetime::IFlipMove::kMoveType, rate(iflipAccepted, iflipAttempts)},
    {::tessera::spacetime::ShiftMove::kMoveType, rate(shiftAccepted, shiftAttempts)},
    {"ishift", rate(ishiftAccepted, ishiftAttempts)},
  };
}

const std::shared_ptr<Spacetime> &CDT::getSpacetime() const noexcept { return spacetime; }
double CDT::getK0() const noexcept { return k0; }
double CDT::getK4() const noexcept { return k4; }
double CDT::getDelta() const noexcept { return delta; }

} // namespace tessera::simulations
