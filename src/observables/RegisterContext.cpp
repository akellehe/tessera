// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/RegisterContext.h"

#include <algorithm>
#include <complex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cobordism/ChainComplex.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/MultiCobordism.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "observables/InteriorHinges.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::cobordism::ChainComplex;
using ::tessera::cobordism::EigenstateSynthesis;
using ::tessera::cobordism::HodgeLaplacian;
using ::tessera::cobordism::MultiCobordism;

// Lazily built per-complex structures, held behind one shared_ptr so a
// `gauged()` copy shares every cache. The caches depend only on the live
// complex, the holes and the degree, never on the target, so sharing them
// across a gauge rotation is safe.
struct RegisterContext::Caches {
  std::unique_ptr<EigenstateSynthesis> synthesis;
  bool epsBuilt = false;
  std::vector<int> eps;
  bool weightsBuilt = false;
  std::vector<double> hodgeWeights;
  bool cellIndexBuilt = false;
  std::map<std::vector<std::uint64_t>, std::size_t> cellIndex;
  bool bettiBuilt = false;
  std::vector<int> betti;
  std::shared_ptr<InteriorHinges> interiorHinges;
};

namespace {

std::string formatHole(const std::vector<std::uint64_t> &hole) {
  std::ostringstream oss;
  oss << "(";
  for (std::size_t i = 0; i < hole.size(); ++i) {
    oss << (i ? ", " : "") << hole[i];
  }
  oss << ")";
  return oss.str();
}

std::string formatHoleList(
    const std::vector<std::vector<std::uint64_t>> &holes) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < holes.size(); ++i) {
    oss << (i ? ", " : "") << formatHole(holes[i]);
  }
  oss << "]";
  return oss.str();
}

}  // namespace

RegisterContext::RegisterContext(std::shared_ptr<Spacetime> spacetime,
                                 int count, int degree,
                                 std::vector<std::complex<double>> target)
    : spacetime_(std::move(spacetime)),
      degree_(degree),
      target_(std::move(target)),
      caches_(std::make_shared<Caches>()) {
  initialize(count, nullptr);
}

RegisterContext::RegisterContext(
    std::shared_ptr<Spacetime> spacetime,
    const std::vector<std::vector<std::uint64_t>> &holes, int count, int degree,
    std::vector<std::complex<double>> target)
    : spacetime_(std::move(spacetime)),
      degree_(degree),
      target_(std::move(target)),
      caches_(std::make_shared<Caches>()) {
  initialize(count, &holes);
}

void RegisterContext::initialize(
    int count, const std::vector<std::vector<std::uint64_t>> *explicitHoles) {
  if (count < 0 || degree_ < 0) {
    throw std::invalid_argument(
        "RegisterContext: count and degree must be non-negative");
  }
  if (!spacetime_) {
    throw std::invalid_argument("RegisterContext: null spacetime");
  }

  // A reader: no build, no solve, no materialize. Every census below comes from
  // the `const` query surface of an already-built, relaxed complex.
  const Spacetime &st = *spacetime_;

  topCellCount_ = static_cast<int>(st.getTopSimplices().size());
  // The spacetime dimension is the metric signature's dimension, never a
  // cell-size guess.
  dimensions_ = st.getMetric()->getSignature()->getDimensions();

  // "Causal content" is the negation of spacelike: it covers timelike and
  // lightlike edges and also mixed ones, which are not definitely spacelike
  // either. That is the conservative reading (the complex is not purely
  // spatial), but it is broader than "timelike or null": a complex of entirely
  // mixed edges reports causal content while carrying no definite causal
  // structure at all.
  causalContent_ = false;
  for (const auto *e : st.getEdgeList()->toVector()) {
    if (!e->isSpacelike()) {
      causalContent_ = true;
      break;
    }
  }

  // The emergent register census at the register degree (nothing placed).
  std::vector<std::vector<std::uint64_t>> emergent =
      MultiCobordism::emergentHoles(st, degree_);
  holesTotal_ = static_cast<int>(emergent.size());

  const std::vector<std::vector<std::uint64_t>> &candidates =
      explicitHoles ? *explicitHoles : emergent;

  // Hole selection is validated here: too few holes throws, and a surplus is a
  // recorded truncation that names the dropped holes rather than a silent
  // slice.
  if (static_cast<int>(candidates.size()) < count) {
    std::ostringstream oss;
    if (explicitHoles) {
      oss << "need >= " << count << " emergent holes, got "
          << candidates.size();
    } else {
      oss << "need >= " << count << " emergent holes at degree " << degree_
          << ", found " << candidates.size();
      if (!candidates.empty()) oss << ": " << formatHoleList(candidates);
      oss << " — this specimen cannot host the requested register";
    }
    throw std::invalid_argument(oss.str());
  }

  holes_.assign(candidates.begin(), candidates.begin() + count);
  droppedHoles_.assign(candidates.begin() + count, candidates.end());
  if (!droppedHoles_.empty()) {
    std::ostringstream oss;
    oss << "register selection: " << candidates.size()
        << " emergent holes; using the first " << count
        << " (emergentHoles order), dropping " << formatHoleList(droppedHoles_)
        << " — the confinement constraint ranges over ALL holes, so the read "
           "covers a sub-register";
    selectionWarning_ = oss.str();
  }
}

int RegisterContext::bK() const {
  const auto &b = betti();
  return (degree_ >= 0 && static_cast<std::size_t>(degree_) < b.size())
             ? b[degree_]
             : 0;
}

const std::vector<int> &RegisterContext::betti() const {
  if (!caches_->bettiBuilt) {
    caches_->betti = MultiCobordism::betti(*spacetime_);
    caches_->bettiBuilt = true;
  }
  return caches_->betti;
}

EigenstateSynthesis &RegisterContext::synthesis() const {
  // EigenstateSynthesis takes a non-const `shared_ptr<Spacetime>` because its
  // setWeights/setInteriorWeights drive the emergent build. Only its `const`
  // read methods (cellSimplices, carriedRepresentative, residualForPeriods) are
  // called here; the setters never are.
  if (!caches_->synthesis) {
    caches_->synthesis =
        std::make_unique<EigenstateSynthesis>(spacetime_, degree_);
  }
  return *caches_->synthesis;
}

const std::vector<int> &RegisterContext::epsilonSigns() const {
  if (!caches_->epsBuilt) {
    // Induced-orientation signs from ChainComplex::endSignCovector, over the top
    // cells in their intrinsic vertex order (never sorted: the stored order
    // carries the orientation) and the selected holes.
    std::vector<std::vector<std::uint64_t>> tops;
    tops.reserve(spacetime_->getTopSimplices().size());
    for (const auto *t : spacetime_->getTopSimplices()) {
      std::vector<std::uint64_t> vids;
      vids.reserve(t->getVertices().size());
      for (const auto *v : t->getVertices()) vids.push_back(v->getId());
      tops.push_back(std::move(vids));
    }
    caches_->eps = ChainComplex::endSignCovector(tops, holes_);
    caches_->epsBuilt = true;
  }
  return caches_->eps;
}

const std::vector<double> &RegisterContext::hodgeWeights() const {
  if (!caches_->weightsBuilt) {
    // weights(degree_) defaults to lorentzian = false, i.e. the |vol| weights,
    // which are real by construction, so .real() here is exact rather than a
    // projection.
    const auto w = HodgeLaplacian(spacetime_).weights(degree_);
    caches_->hodgeWeights.resize(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) caches_->hodgeWeights[i] = w[i].real();
    caches_->weightsBuilt = true;
  }
  return caches_->hodgeWeights;
}

const std::map<std::vector<std::uint64_t>, std::size_t> &
RegisterContext::cellIndex() const {
  if (!caches_->cellIndexBuilt) {
    const auto &cells = synthesis().cellSimplices();
    for (std::size_t i = 0; i < cells.size(); ++i) {
      std::vector<std::uint64_t> key = cells[i];
      std::sort(key.begin(), key.end());
      caches_->cellIndex.emplace(std::move(key), i);
    }
    caches_->cellIndexBuilt = true;
  }
  return caches_->cellIndex;
}

const std::shared_ptr<InteriorHinges> &RegisterContext::interiorHinges() const {
  if (!caches_->interiorHinges) {
    // The 4D hinge selection reads a const spacetime, so it is read-only by
    // construction. The live complex must already carry its facet skeleton; the
    // loader completes one after a bare fromCells rehydration.
    caches_->interiorHinges = std::make_shared<InteriorHinges>(
        std::shared_ptr<const Spacetime>(spacetime_), holes_);
  }
  return caches_->interiorHinges;
}

std::shared_ptr<RegisterContext> RegisterContext::gauged(double theta) const {
  // Shares this context's live spacetime and caches, rotating only the register
  // target by the global U(1) phase e^{i theta}.
  auto g = std::make_shared<RegisterContext>(*this);
  const std::complex<double> phase = std::exp(std::complex<double>(0.0, theta));
  for (auto &t : g->target_) t *= phase;
  return g;
}

}  // namespace tessera::observables
