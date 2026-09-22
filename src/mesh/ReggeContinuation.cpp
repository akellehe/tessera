// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/ReggeContinuation.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"

namespace tessera::mesh {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

double factorialOf(int k) {
  double f = 1.0;
  for (int i = 2; i <= k; ++i) f *= static_cast<double>(i);
  return f;
}

SimplexKey keyOf(const Simplex &s) {
  SimplexKey key;
  key.reserve(s.getVertices().size());
  for (const auto &v : s.getVertices()) key.push_back(v->getId());
  std::sort(key.begin(), key.end());
  return key;
}

std::string render(const SimplexKey &key) {
  std::string out = "(";
  for (std::size_t i = 0; i < key.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(key[i]);
  }
  return out + ")";
}

}  // namespace

ReggeContinuation::ReggeContinuation(std::shared_ptr<spacetime::Spacetime> spacetime)
    : spacetime_(std::move(spacetime)) {
  if (spacetime_ == nullptr)
    throw std::invalid_argument("ReggeContinuation: null spacetime");
  dimension_ = spacetime_->getMetric()->getSignature()->getDimensions();
  const auto cellSize = static_cast<std::size_t>(dimension_ + 1);
  // A hinge is a (d-2)-simplex, i.e. d-1 vertices. In two dimensions that is a
  // point, whose content is a count rather than a root.
  const auto hingeSize = static_cast<std::size_t>(dimension_ - 1);

  for (const auto &s : spacetime_->getSimplices()) {
    if (s == nullptr) continue;
    if (s->size() == cellSize) {
      CellState state;
      state.cell = s;
      state.content = SheetedSqrt(s->gramCache().gramDet);
      state.factorial = factorialOf(dimension_);
      cells_.emplace(keyOf(*s), std::move(state));
    } else if (s->size() == hingeSize && s->hasTopCoface()) {
      HingeState state;
      state.hinge = s;
      const int k = static_cast<int>(s->size()) - 1;
      state.hasContentRoot = k >= 1;
      if (state.hasContentRoot) {
        state.content = SheetedSqrt(s->gramCache().gramDet);
        state.factorial = factorialOf(k);
      }
      hinges_.emplace(keyOf(*s), std::move(state));
    }
  }

  // Incidence by enumeration over each cell's hinge-sized vertex subsets, so
  // the cost is C(d+1, d-1) lookups per cell rather than a scan over every
  // hinge. The subsets of a sorted key stay sorted, so they are hinge keys
  // directly.
  for (auto &[cellKey, cellState] : cells_) {
    if (hingeSize > cellKey.size()) continue;
    std::vector<bool> mask(cellKey.size(), false);
    std::fill(mask.begin(), mask.begin() + static_cast<std::ptrdiff_t>(hingeSize), true);
    do {
      SimplexKey hingeKey;
      hingeKey.reserve(hingeSize);
      for (std::size_t i = 0; i < cellKey.size(); ++i)
        if (mask[i]) hingeKey.push_back(cellKey[i]);
      const auto hinge = hinges_.find(hingeKey);
      if (hinge == hinges_.end()) continue;
      const auto cofactors = cellState.cell->dihedralCofactors(hinge->second.hinge);
      if (!cofactors.ok) continue;
      AngleState angle;
      angle.rootII = SheetedSqrt(cofactors.Cii);
      angle.rootJJ = SheetedSqrt(cofactors.Cjj);
      angle.angle = SheetedAcos(
          cosineOn(cofactors.Cij, angle.rootII.value(), angle.rootJJ.value()));
      hinge->second.cofaces.push_back(cellKey);
      angles_.emplace(AngleKey{cellKey, hingeKey}, std::move(angle));
    } while (std::prev_permutation(mask.begin(), mask.end()));
  }
}

std::complex<double> ReggeContinuation::cosineOn(std::complex<double> Cij,
                                                 std::complex<double> rootII,
                                                 std::complex<double> rootJJ) {
  const std::complex<double> denominator = rootII * rootJJ;
  if (std::abs(denominator) < 1e-300) return {0.0, 0.0};
  std::complex<double> r = -Cij / denominator;
  // Pinned to the +0 side for a real ratio, the side Simplex::dihedralAngle
  // takes: for |r| > 1 the sign of Im(theta) is decided by which side of the
  // arccosine's cut a real argument sits on, and leaving that to the sign of a
  // floating-point zero would make the boost orientation a rounding accident.
  if (r.imag() == 0.0) r = {r.real(), 0.0};
  return r;
}

void ReggeContinuation::advance() {
  maxRadicandTurn_ = 0.0;
  maxAngleStep_ = 0.0;

  for (auto &[key, state] : cells_) {
    state.content.advance(state.cell->gramCache().gramDet);
    maxRadicandTurn_ = std::max(maxRadicandTurn_, std::abs(state.content.lastStep()));
  }
  for (auto &[key, state] : hinges_) {
    if (!state.hasContentRoot) continue;
    state.content.advance(state.hinge->gramCache().gramDet);
    maxRadicandTurn_ = std::max(maxRadicandTurn_, std::abs(state.content.lastStep()));
  }
  for (auto &[key, state] : angles_) {
    const CellState &cell = cellState(key.first);
    const HingeState &hinge = hingeState(key.second);
    const auto cofactors = cell.cell->dihedralCofactors(hinge.hinge);
    if (!cofactors.ok) continue;
    state.rootII.advance(cofactors.Cii);
    state.rootJJ.advance(cofactors.Cjj);
    maxRadicandTurn_ = std::max(
        {maxRadicandTurn_, std::abs(state.rootII.lastStep()),
         std::abs(state.rootJJ.lastStep())});
    state.angle.advance(
        cosineOn(cofactors.Cij, state.rootII.value(), state.rootJJ.value()));
    maxAngleStep_ = std::max(maxAngleStep_, state.angle.lastStep());
  }
}

const ReggeContinuation::CellState &ReggeContinuation::cellState(
    const SimplexKey &cell) const {
  const auto it = cells_.find(cell);
  if (it == cells_.end())
    throw std::invalid_argument("ReggeContinuation: no top cell " + render(cell));
  return it->second;
}

const ReggeContinuation::HingeState &ReggeContinuation::hingeState(
    const SimplexKey &hinge) const {
  const auto it = hinges_.find(hinge);
  if (it == hinges_.end())
    throw std::invalid_argument("ReggeContinuation: no hinge " + render(hinge));
  return it->second;
}

const ReggeContinuation::AngleState &ReggeContinuation::angleState(
    const AngleKey &key) const {
  const auto it = angles_.find(key);
  if (it == angles_.end())
    throw std::invalid_argument("ReggeContinuation: hinge " + render(key.second) +
                                " is not a hinge of cell " + render(key.first));
  return it->second;
}

std::vector<SimplexKey> ReggeContinuation::cells() const {
  std::vector<SimplexKey> out;
  out.reserve(cells_.size());
  for (const auto &[key, state] : cells_) out.push_back(key);
  return out;
}

std::vector<SimplexKey> ReggeContinuation::hinges() const {
  std::vector<SimplexKey> out;
  out.reserve(hinges_.size());
  for (const auto &[key, state] : hinges_) out.push_back(key);
  return out;
}

std::complex<double> ReggeContinuation::volume(const SimplexKey &cell) const {
  const CellState &state = cellState(cell);
  return state.content.value() / state.factorial;
}

int ReggeContinuation::volumeSheet(const SimplexKey &cell) const {
  return cellState(cell).content.sheet();
}

int ReggeContinuation::volumeWinding(const SimplexKey &cell) const {
  return cellState(cell).content.winding();
}

std::complex<double> ReggeContinuation::hingeContent(const SimplexKey &hinge) const {
  const HingeState &state = hingeState(hinge);
  if (!state.hasContentRoot) return {1.0, 0.0};
  return state.content.value() / state.factorial;
}

int ReggeContinuation::hingeContentSheet(const SimplexKey &hinge) const {
  const HingeState &state = hingeState(hinge);
  return state.hasContentRoot ? state.content.sheet() : 0;
}

std::complex<double> ReggeContinuation::dihedralAngle(const SimplexKey &cell,
                                                      const SimplexKey &hinge) const {
  return angleState({cell, hinge}).angle.value();
}

int ReggeContinuation::angleBranchIndex(const SimplexKey &cell,
                                        const SimplexKey &hinge) const {
  return angleState({cell, hinge}).angle.branchIndex();
}

int ReggeContinuation::angleOrientation(const SimplexKey &cell,
                                        const SimplexKey &hinge) const {
  return angleState({cell, hinge}).angle.orientation();
}

std::pair<int, int> ReggeContinuation::angleCofactorSheets(
    const SimplexKey &cell, const SimplexKey &hinge) const {
  const AngleState &state = angleState({cell, hinge});
  return {state.rootII.sheet(), state.rootJJ.sheet()};
}

std::complex<double> ReggeContinuation::deficitAngle(const SimplexKey &hinge) const {
  const HingeState &state = hingeState(hinge);
  std::complex<double> sum(0.0, 0.0);
  for (const auto &cell : state.cofaces)
    sum += angleState({cell, hinge}).angle.value();
  return std::complex<double>(kTwoPi, 0.0) - sum;
}

std::complex<double> ReggeContinuation::action() const {
  std::complex<double> S(0.0, 0.0);
  for (const auto &[key, state] : hinges_) S += hingeContent(key) * deficitAngle(key);
  return S;
}

std::complex<double> ReggeContinuation::principalAction() const {
  // The same sum term for term, with every label reset to its principal value:
  // the content root, both cofactor roots and the inverse cosine. Built here
  // rather than read off Simplex so that the comparison is of sheets alone and
  // not of two different definitions of the hinge measure.
  std::complex<double> S(0.0, 0.0);
  for (const auto &[hingeKey, hinge] : hinges_) {
    std::complex<double> content(1.0, 0.0);
    if (hinge.hasContentRoot)
      content = principalSquareRoot(hinge.hinge->gramCache().gramDet) / hinge.factorial;
    std::complex<double> sum(0.0, 0.0);
    for (const auto &cellKey : hinge.cofaces) {
      const auto cofactors = cellState(cellKey).cell->dihedralCofactors(hinge.hinge);
      if (!cofactors.ok) continue;
      sum += principalArcCosine(cosineOn(cofactors.Cij,
                                         principalSquareRoot(cofactors.Cii),
                                         principalSquareRoot(cofactors.Cjj)));
    }
    S += content * (std::complex<double>(kTwoPi, 0.0) - sum);
  }
  return S;
}

bool ReggeContinuation::touchedBranchPoint() const {
  for (const auto &[key, state] : cells_)
    if (state.content.touchedBranchPoint()) return true;
  for (const auto &[key, state] : hinges_)
    if (state.hasContentRoot && state.content.touchedBranchPoint()) return true;
  for (const auto &[key, state] : angles_)
    if (state.rootII.touchedBranchPoint() || state.rootJJ.touchedBranchPoint())
      return true;
  return false;
}

}  // namespace tessera::mesh
