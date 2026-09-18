// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/ExchangeHolonomy.h"

#include "cobordism/IntegerLinalg.h"
#include "quantum/GradedFock.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tessera::observables {

namespace {

using cd = std::complex<double>;
using cobordism::Certificate;
using cobordism::CertificateDomain;
using cobordism::CertificateRegime;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Severity order of the metric regimes (Positive < HermitianIndefinite <
// ComplexSymmetricPencil < NonNormal); a composed read reports the worst
// regime it touched.
int regimeSeverity(CertificateRegime regime) {
  switch (regime) {
    case CertificateRegime::PositiveSemidefinite:
      return 0;
    case CertificateRegime::HermitianIndefinite:
      return 1;
    case CertificateRegime::ComplexSymmetricPencil:
      return 2;
    case CertificateRegime::NonNormal:
      return 3;
  }
  return 3;
}

CertificateRegime worseRegime(CertificateRegime a, CertificateRegime b) {
  return regimeSeverity(a) >= regimeSeverity(b) ? a : b;
}

// ||M - sign * I||_F for the +-I closure tests.
double distanceToSignedIdentity(const Eigen::MatrixXcd &m, double sign) {
  const Eigen::MatrixXcd target =
      sign * Eigen::MatrixXcd::Identity(m.rows(), m.cols());
  return (m - target).norm();
}

// Deterministic sign fix of a real vector: flips so its first component of
// magnitude > tol is positive (the pi-branch axis rule).
void canonicalizeSign(Eigen::VectorXd &v, double tol = 1e-12) {
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (std::abs(v[i]) > tol) {
      if (v[i] < 0.0) v = -v;
      return;
    }
  }
}

// Deterministic phase fix of a complex vector: rotates so its first
// component of magnitude > tol is real positive.
void canonicalizePhase(Eigen::VectorXcd &v, double tol = 1e-12) {
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (std::abs(v[i]) > tol) {
      v *= std::conj(v[i]) / std::abs(v[i]);
      return;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// certified overlap transport
// ---------------------------------------------------------------------------

Eigen::MatrixXcd ExchangeHolonomy::polarUnitary(
    const Eigen::MatrixXcd &overlap) {
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
      overlap, Eigen::ComputeFullU | Eigen::ComputeFullV);
  return svd.matrixU() * svd.matrixV().adjoint();
}

Eigen::MatrixXcd ExchangeHolonomy::transportStep(
    const Eigen::MatrixXcd &from, const Eigen::MatrixXcd &to,
    const Eigen::VectorXcd &weights, TransportStepRead &read,
    const ExchangeHolonomyConfig &cfg) {
  const Eigen::MatrixXcd overlap =
      to.adjoint() * weights.asDiagonal() * from;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
      overlap, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto &sigma = svd.singularValues();
  read.minSingularValue = sigma.size() ? sigma[sigma.size() - 1] : 0.0;
  read.maxSingularValue = sigma.size() ? sigma[0] : 0.0;
  read.conditioning = read.minSingularValue > 0.0
                          ? read.maxSingularValue / read.minSingularValue
                          : std::numeric_limits<double>::infinity();
  read.certified = read.minSingularValue >= cfg.leakFloor &&
                   read.conditioning <= cfg.conditionCap;
  return svd.matrixU() * svd.matrixV().adjoint();
}

LoopHolonomyRead ExchangeHolonomy::finalizeLoop(
    Eigen::MatrixXcd holonomy, std::vector<TransportStepRead> stepReads,
    std::size_t steps, std::size_t rank, CertificateRegime regime,
    bool uncertifiedBand, const ExchangeHolonomyConfig &cfg) {
  LoopHolonomyRead read;
  read.steps = steps;
  read.rank = rank;
  read.uncertifiedBand = uncertifiedBand;
  read.stepReads = std::move(stepReads);

  bool allCertified = !read.stepReads.empty();
  read.minStepSingularValue = std::numeric_limits<double>::infinity();
  read.conditioning = 0.0;
  for (const TransportStepRead &s : read.stepReads) {
    allCertified = allCertified && s.certified;
    read.minStepSingularValue =
        std::min(read.minStepSingularValue, s.minSingularValue);
    read.conditioning = std::max(read.conditioning, s.conditioning);
  }
  if (read.stepReads.empty()) {
    read.minStepSingularValue = kNaN;
    read.conditioning = kNaN;
  }

  const bool structurallyValid = holonomy.size() > 0;
  if (structurallyValid) {
    read.holonomy = std::move(holonomy);
    read.determinant = read.holonomy.determinant();
    const Eigen::MatrixXcd gram = read.holonomy.adjoint() * read.holonomy;
    read.unitarityResidual =
        (gram - Eigen::MatrixXcd::Identity(gram.rows(), gram.cols())).norm() /
        std::sqrt(std::max<std::size_t>(1, rank));
  }

  const bool certified = structurallyValid && allCertified &&
                         !uncertifiedBand &&
                         read.unitarityResidual <= cfg.unitaryTolerance;
  read.certificate =
      certified ? Certificate::certifiedNumerical(
                      CertificateDomain::Static, regime,
                      read.unitarityResidual, read.conditioning,
                      cfg.unitaryTolerance)
                : Certificate::heuristicDiscovery(CertificateDomain::Static,
                                                  regime);
  return read;
}

cobordism::CertificateRegime ExchangeHolonomy::weightsRegime(
    const Eigen::VectorXcd &weights) {
  bool allReal = true;
  bool allPositive = true;
  for (Eigen::Index i = 0; i < weights.size(); ++i) {
    if (weights[i].imag() != 0.0) allReal = false;
    if (!(weights[i].real() > 0.0) || weights[i].imag() != 0.0)
      allPositive = false;
  }
  if (allPositive) return CertificateRegime::PositiveSemidefinite;
  if (allReal) return CertificateRegime::HermitianIndefinite;
  return CertificateRegime::NonNormal;
}

LoopHolonomyRead ExchangeHolonomy::loopHolonomy(
    const std::vector<Eigen::MatrixXcd> &frames,
    const Eigen::VectorXcd &weights, const ExchangeHolonomyConfig &cfg) {
  return loopHolonomyPerStep(
      frames, std::vector<Eigen::VectorXcd>(frames.size(), weights), cfg);
}

LoopHolonomyRead ExchangeHolonomy::loopHolonomyPerStep(
    const std::vector<Eigen::MatrixXcd> &frames,
    const std::vector<Eigen::VectorXcd> &stepWeights,
    const ExchangeHolonomyConfig &cfg) {
  if (frames.empty())
    throw std::invalid_argument("ExchangeHolonomy: empty frame loop");
  if (stepWeights.size() != frames.size())
    throw std::invalid_argument(
        "ExchangeHolonomy: stepWeights must have one entry per loop frame");
  const Eigen::Index rows = frames.front().rows();
  const Eigen::Index rank = frames.front().cols();
  for (const Eigen::MatrixXcd &f : frames) {
    if (f.rows() != rows || f.cols() != rank)
      throw std::invalid_argument(
          "ExchangeHolonomy: explicit frame loop with a shape mismatch (an "
          "explicit path with inconsistent shapes is a structural error)");
  }
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  for (const Eigen::VectorXcd &w : stepWeights) {
    if (w.size() != rows)
      throw std::invalid_argument(
          "ExchangeHolonomy: weights size must equal the frame row count");
    regime = worseRegime(regime, weightsRegime(w));
  }

  const std::size_t T = frames.size();
  Eigen::MatrixXcd holonomy = Eigen::MatrixXcd::Identity(rank, rank);
  std::vector<TransportStepRead> stepReads(T);
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t next = (t + 1) % T;
    stepReads[t].fromIndex = t;
    stepReads[t].toIndex = next;
    const Eigen::MatrixXcd r = transportStep(frames[t], frames[next],
                                             stepWeights[t], stepReads[t],
                                             cfg);
    holonomy = r * holonomy;
  }
  return finalizeLoop(std::move(holonomy), std::move(stepReads), T,
                      static_cast<std::size_t>(rank), regime,
                      /*uncertifiedBand=*/false, cfg);
}

struct ExchangeHolonomy::RestrictedPair {
  Eigen::MatrixXcd from;      // shared-cells x rank
  Eigen::MatrixXcd to;        // shared-cells x rank
  Eigen::VectorXcd weights;   // departing fiber's W on the shared cells
  std::size_t sharedCells = 0;
};

ExchangeHolonomy::RestrictedPair ExchangeHolonomy::restrictToSharedCells(
    const SpectralFiber &from, const SpectralFiber &to) {
  std::map<std::vector<std::uint64_t>, Eigen::Index> toRows;
  const auto &toCells = to.cellVertices();
  for (std::size_t r = 0; r < toCells.size(); ++r)
    toRows.emplace(toCells[r], static_cast<Eigen::Index>(r));

  const auto &fromCells = from.cellVertices();
  std::vector<std::pair<Eigen::Index, Eigen::Index>> shared;
  for (std::size_t r = 0; r < fromCells.size(); ++r) {
    const auto hit = toRows.find(fromCells[r]);
    if (hit != toRows.end())
      shared.emplace_back(static_cast<Eigen::Index>(r), hit->second);
  }

  const Eigen::MatrixXcd fromFrame = from.rightFrame();
  const Eigen::MatrixXcd toFrame = to.rightFrame();
  const Eigen::VectorXcd fromWeights = from.weightDiagonal();

  RestrictedPair pair;
  pair.sharedCells = shared.size();
  pair.from.resize(static_cast<Eigen::Index>(shared.size()),
                   fromFrame.cols());
  pair.to.resize(static_cast<Eigen::Index>(shared.size()), toFrame.cols());
  pair.weights.resize(static_cast<Eigen::Index>(shared.size()));
  for (std::size_t s = 0; s < shared.size(); ++s) {
    const auto idx = static_cast<Eigen::Index>(s);
    pair.from.row(idx) = fromFrame.row(shared[s].first);
    pair.to.row(idx) = toFrame.row(shared[s].second);
    pair.weights[idx] = fromWeights[shared[s].first];
  }
  return pair;
}

LoopHolonomyRead ExchangeHolonomy::fiberLoopHolonomy(
    const std::vector<SpectralFiber> &loop,
    const ExchangeHolonomyConfig &cfg) {
  if (loop.empty())
    throw std::invalid_argument("ExchangeHolonomy: empty fiber loop");

  const std::size_t T = loop.size();
  const std::size_t rank = loop.front().rank();
  bool uncertifiedBand = false;
  bool rankChanged = false;
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  for (const SpectralFiber &fiber : loop) {
    uncertifiedBand = uncertifiedBand || !fiber.accepted();
    rankChanged = rankChanged || fiber.rank() != rank;
    regime = worseRegime(regime, fiber.certificate().certificate.regime());
  }
  if (rankChanged) {
    // A rank change along the track breaks the band identity: the read is
    // returned uncertified, with no sign and no exception.
    return finalizeLoop(Eigen::MatrixXcd(), {}, T, rank, regime,
                        /*uncertifiedBand=*/true, cfg);
  }

  Eigen::MatrixXcd holonomy = Eigen::MatrixXcd::Identity(
      static_cast<Eigen::Index>(rank), static_cast<Eigen::Index>(rank));
  std::vector<TransportStepRead> stepReads(T);
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t next = (t + 1) % T;
    stepReads[t].fromIndex = t;
    stepReads[t].toIndex = next;
    const RestrictedPair pair = restrictToSharedCells(loop[t], loop[next]);
    if (pair.sharedCells == 0) {
      // No shared support: a total leak (min singular value 0).
      stepReads[t].minSingularValue = 0.0;
      stepReads[t].maxSingularValue = 0.0;
      stepReads[t].conditioning = std::numeric_limits<double>::infinity();
      stepReads[t].certified = false;
      continue;
    }
    const Eigen::MatrixXcd r = transportStep(pair.from, pair.to,
                                             pair.weights, stepReads[t], cfg);
    holonomy = r * holonomy;
  }
  return finalizeLoop(std::move(holonomy), std::move(stepReads), T, rank,
                      regime, uncertifiedBand, cfg);
}

// ---------------------------------------------------------------------------
// interferometric (Berry-cancelled) characters
// ---------------------------------------------------------------------------

HolonomyCharacterRead ExchangeHolonomy::characterAgainstReference(
    const LoopHolonomyRead &loop, const LoopHolonomyRead &reference,
    HolonomyChannel channel, const ExchangeHolonomyConfig &cfg) {
  HolonomyCharacterRead read;
  read.channel = channel;
  read.rawLoopDeterminant = loop.determinant;
  read.referenceDeterminant = reference.determinant;
  read.timingMatched = loop.steps == reference.steps;
  read.ranksMatched = loop.rank == reference.rank;

  const CertificateRegime regime =
      worseRegime(loop.certificate.regime(), reference.certificate.regime());
  const bool referenceInvertible =
      std::isfinite(std::abs(reference.determinant)) &&
      std::abs(reference.determinant) > 1e-12;
  if (referenceInvertible &&
      std::isfinite(std::abs(loop.determinant))) {
    read.character = loop.determinant / reference.determinant;
  }

  const double magnitudeResidual =
      std::isfinite(std::abs(read.character))
          ? std::abs(std::abs(read.character) - 1.0)
          : kNaN;
  const bool certified = loop.certificate.holds() &&
                         reference.certificate.holds() &&
                         read.timingMatched && read.ranksMatched &&
                         referenceInvertible &&
                         magnitudeResidual <= cfg.unitaryTolerance;
  const double conditioning =
      std::max(loop.conditioning, reference.conditioning);
  read.certificate =
      certified ? Certificate::certifiedNumerical(
                      CertificateDomain::Static, regime, magnitudeResidual,
                      conditioning, cfg.unitaryTolerance)
                : Certificate::heuristicDiscovery(CertificateDomain::Static,
                                                  regime);

  if (read.certificate.holds()) {
    const double toPlus = std::abs(read.character - cd(1.0, 0.0));
    const double toMinus = std::abs(read.character - cd(-1.0, 0.0));
    if (std::min(toPlus, toMinus) <= cfg.signTolerance) {
      read.characterSign = toPlus <= toMinus ? +1 : -1;
      read.signResidual = std::min(toPlus, toMinus);
    }
  }
  return read;
}

HolonomyCharacterRead ExchangeHolonomy::exchangeCharacter(
    const LoopHolonomyRead &exchangeLoop,
    const LoopHolonomyRead &referenceLoop,
    const ExchangeHolonomyConfig &cfg) {
  return characterAgainstReference(exchangeLoop, referenceLoop,
                                   HolonomyChannel::ParticleExchange, cfg);
}

HolonomyCharacterRead ExchangeHolonomy::rotationCharacter(
    const LoopHolonomyRead &rotationLoop,
    const LoopHolonomyRead &referenceLoop,
    const ExchangeHolonomyConfig &cfg) {
  return characterAgainstReference(rotationLoop, referenceLoop,
                                   HolonomyChannel::PhysicalRotation, cfg);
}

std::complex<double> ExchangeHolonomy::doublyCancelledRatio(
    const HolonomyCharacterRead &exchange,
    const HolonomyCharacterRead &rotation) {
  if (exchange.channel != HolonomyChannel::ParticleExchange)
    throw std::invalid_argument(
        "doublyCancelledRatio: the first read must be a ParticleExchange "
        "character (the channels are never interchangeable)");
  if (rotation.channel != HolonomyChannel::PhysicalRotation)
    throw std::invalid_argument(
        "doublyCancelledRatio: the second read must be a PhysicalRotation "
        "character (the channels are never interchangeable)");
  return exchange.character / rotation.character;
}

// ---------------------------------------------------------------------------
// structural permutation channel
// ---------------------------------------------------------------------------

int ExchangeHolonomy::permutationSign(
    const std::vector<std::size_t> &permutation) {
  const std::size_t m = permutation.size();
  if (m == 0) return 1;
  std::vector<std::size_t> all(m);
  std::iota(all.begin(), all.end(), std::size_t{0});
  // With every mode occupied, permutationParity is the sign of the
  // permutation itself (the algebraic wedge sign).
  return quantum::OccupationBitset::fromOccupiedModes(m, all)
      .permutationParity(permutation);
}

BlockPermutationRead ExchangeHolonomy::blockPermutation(
    const std::vector<std::vector<SpectralFiber>> &steps,
    const std::vector<std::vector<SpectralFiber>> &referenceSteps,
    const std::vector<std::vector<std::size_t>> &composites,
    const ExchangeHolonomyConfig &cfg) {
  BlockPermutationRead read;
  if (steps.empty() || steps.front().empty())
    throw std::invalid_argument(
        "ExchangeHolonomy::blockPermutation: empty tracking");

  const std::size_t T = steps.size();
  const std::size_t B = steps.front().size();

  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  bool premiseOk = true;
  bool uncertifiedBand = false;
  for (const auto &step : steps) {
    premiseOk = premiseOk && step.size() == B;
    for (const SpectralFiber &f : step) {
      uncertifiedBand = uncertifiedBand || !f.accepted();
      regime = worseRegime(regime, f.certificate().certificate.regime());
    }
  }
  premiseOk = premiseOk && !uncertifiedBand;

  const auto uncertified = [&read, regime]() {
    // A failed premise (gap closure, rank change, ambiguous matching, a
    // malformed reference) invalidates the whole read: no permutation and no
    // parity is reported from an uncertified experiment.
    read.blockPermutation.clear();
    read.compositePermutation.clear();
    read.blockParity = 0;
    read.modeParity = 0;
    read.compositeParity = 0;
    read.residualInBlockMotion = kNaN;
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::Static, regime);
    return read;
  };
  if (!premiseOk) return uncertified();

  for (const SpectralFiber &f : steps.front())
    read.blockRanks.push_back(f.rank());

  // Per-step block matching via SpectralFiberTracker::matchFibers. Every
  // match must be a certified continuation and the mapping a bijection.
  std::vector<std::vector<std::size_t>> stepMaps(T);
  std::vector<std::vector<Eigen::MatrixXcd>> stepTransports(T);
  double minOverlap = std::numeric_limits<double>::infinity();
  double worstConditioning = 0.0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t next = (t + 1) % T;
    const auto matches = SpectralFiberTracker::matchFibers(
        steps[t], steps[next], cfg.blockMatchThreshold);
    if (matches.size() != B) return uncertified();
    std::vector<std::size_t> map(B, B);
    std::vector<bool> hit(B, false);
    for (const FiberMatchRead &m : matches) {
      if (!m.certifiedContinuation) return uncertified();
      if (m.fromIndex >= B || m.toIndex >= B || hit[m.toIndex])
        return uncertified();
      map[m.fromIndex] = m.toIndex;
      hit[m.toIndex] = true;
      minOverlap = std::min(minOverlap, m.overlap.subspaceOverlap);
    }
    for (std::size_t b = 0; b < B; ++b)
      if (map[b] >= B) return uncertified();

    stepMaps[t] = map;
    stepTransports[t].resize(B);
    for (std::size_t b = 0; b < B; ++b) {
      TransportStepRead stepRead;
      stepRead.fromIndex = t;
      stepRead.toIndex = next;
      const RestrictedPair pair =
          restrictToSharedCells(steps[t][b], steps[next][map[b]]);
      if (pair.sharedCells == 0) return uncertified();
      stepTransports[t][b] =
          transportStep(pair.from, pair.to, pair.weights, stepRead, cfg);
      if (!stepRead.certified) return uncertified();
      worstConditioning = std::max(worstConditioning, stepRead.conditioning);
    }
  }
  read.minMatchOverlap = minOverlap;

  // Full-loop permutation: block b at t = 0 arrives at pi(b).
  std::vector<std::size_t> pi(B);
  for (std::size_t b = 0; b < B; ++b) {
    std::size_t pos = b;
    for (std::size_t t = 0; t < T; ++t) pos = stepMaps[t][pos];
    pi[b] = pos;
  }
  read.blockPermutation = pi;
  read.blockParity = permutationSign(pi);

  // Mode-level parity (the exchange statistic): blocks expanded to their
  // ranks, in-block order carried, giving the graded sign.
  std::vector<std::size_t> offsets(B, 0);
  std::size_t modeCount = 0;
  for (std::size_t b = 0; b < B; ++b) {
    offsets[b] = modeCount;
    modeCount += read.blockRanks[b];
  }
  std::vector<std::size_t> modePerm(modeCount);
  for (std::size_t b = 0; b < B; ++b)
    for (std::size_t k = 0; k < read.blockRanks[b]; ++k)
      modePerm[offsets[b] + k] = offsets[pi[b]] + k;
  read.modeParity = permutationSign(modePerm);

  // Optional composite-level view.
  if (!composites.empty()) {
    std::vector<std::size_t> owner(B, composites.size());
    for (std::size_t c = 0; c < composites.size(); ++c) {
      for (const std::size_t b : composites[c]) {
        if (b >= B || owner[b] != composites.size())
          throw std::invalid_argument(
              "ExchangeHolonomy::blockPermutation: composites must "
              "partition the block indices");
        owner[b] = c;
      }
    }
    for (std::size_t b = 0; b < B; ++b) {
      if (owner[b] == composites.size())
        throw std::invalid_argument(
            "ExchangeHolonomy::blockPermutation: composites must "
            "partition the block indices");
    }
    std::vector<std::size_t> compPerm(composites.size(), composites.size());
    bool wellDefined = true;
    for (std::size_t c = 0; c < composites.size() && wellDefined; ++c) {
      for (const std::size_t b : composites[c]) {
        const std::size_t target = owner[pi[b]];
        if (compPerm[c] == composites.size()) {
          compPerm[c] = target;
        } else if (compPerm[c] != target) {
          wellDefined = false;  // the composite's blocks scattered
          break;
        }
      }
    }
    if (wellDefined) {
      std::vector<bool> seen(composites.size(), false);
      for (const std::size_t c : compPerm) {
        if (c >= composites.size() || seen[c]) {
          wellDefined = false;
          break;
        }
        seen[c] = true;
      }
    }
    if (wellDefined) {
      read.compositePermutation = compPerm;
      read.compositeParity = permutationSign(compPerm);
    }
    // A scattered composite leaves the composite view empty; the block and
    // mode channels remain certified.
  }

  // Residual in-block motion after reference cancellation.
  read.residualInBlockMotion = Certificate::kUnmeasured;
  if (!referenceSteps.empty()) {
    bool refOk = referenceSteps.size() == T;
    for (const auto &step : referenceSteps)
      refOk = refOk && step.size() == B;
    if (refOk) {
      for (const auto &step : referenceSteps)
        for (const SpectralFiber &f : step)
          if (!f.accepted()) refOk = false;
    }
    std::vector<Eigen::MatrixXcd> refLoops(B);
    if (refOk) {
      std::vector<std::vector<std::size_t>> refMaps(T);
      std::vector<std::vector<Eigen::MatrixXcd>> refTransports(T);
      for (std::size_t t = 0; t < T && refOk; ++t) {
        const std::size_t next = (t + 1) % T;
        const auto matches = SpectralFiberTracker::matchFibers(
            referenceSteps[t], referenceSteps[next], cfg.blockMatchThreshold);
        if (matches.size() != B) {
          refOk = false;
          break;
        }
        refMaps[t].assign(B, B);
        refTransports[t].resize(B);
        for (const FiberMatchRead &m : matches) {
          if (!m.certifiedContinuation || m.fromIndex >= B ||
              m.toIndex >= B) {
            refOk = false;
            break;
          }
          refMaps[t][m.fromIndex] = m.toIndex;
          TransportStepRead stepRead;
          const RestrictedPair pair = restrictToSharedCells(
              referenceSteps[t][m.fromIndex], referenceSteps[next][m.toIndex]);
          if (pair.sharedCells == 0) {
            refOk = false;
            break;
          }
          refTransports[t][m.fromIndex] =
              transportStep(pair.from, pair.to, pair.weights, stepRead, cfg);
          if (!stepRead.certified) refOk = false;
        }
      }
      if (refOk) {
        // The reference must be non-exchanging: the identity full-loop map.
        for (std::size_t b = 0; b < B && refOk; ++b) {
          std::size_t pos = b;
          for (std::size_t t = 0; t < T; ++t) pos = refMaps[t][pos];
          refOk = refOk && pos == b;
        }
        if (refOk) {
          for (std::size_t b = 0; b < B; ++b) {
            Eigen::MatrixXcd u = Eigen::MatrixXcd::Identity(
                static_cast<Eigen::Index>(read.blockRanks[b]),
                static_cast<Eigen::Index>(read.blockRanks[b]));
            std::size_t pos = b;
            for (std::size_t t = 0; t < T; ++t) {
              u = refTransports[t][pos] * u;
              pos = refMaps[t][pos];
            }
            refLoops[b] = u;
          }
        }
      }
    }
    if (!refOk) return uncertified();

    // Per permutation cycle: compose the exchange track over the whole
    // cycle, cancel the visited blocks' reference loops in visit order.
    double worst = 0.0;
    std::vector<bool> done(B, false);
    for (std::size_t b0 = 0; b0 < B; ++b0) {
      if (done[b0]) continue;
      std::vector<std::size_t> visit;
      std::size_t b = b0;
      do {
        visit.push_back(b);
        done[b] = true;
        b = pi[b];
      } while (b != b0);

      Eigen::MatrixXcd u = Eigen::MatrixXcd::Identity(
          static_cast<Eigen::Index>(read.blockRanks[b0]),
          static_cast<Eigen::Index>(read.blockRanks[b0]));
      std::size_t pos = b0;
      for (std::size_t round = 0; round < visit.size(); ++round) {
        for (std::size_t t = 0; t < T; ++t) {
          u = stepTransports[t][pos] * u;
          pos = stepMaps[t][pos];
        }
      }
      Eigen::MatrixXcd refProduct = Eigen::MatrixXcd::Identity(
          static_cast<Eigen::Index>(read.blockRanks[b0]),
          static_cast<Eigen::Index>(read.blockRanks[b0]));
      for (const std::size_t visited : visit)
        refProduct = refLoops[visited] * refProduct;
      const Eigen::MatrixXcd cancelled = u * refProduct.adjoint();
      worst = std::max(
          worst, distanceToSignedIdentity(cancelled, 1.0) /
                     std::sqrt(static_cast<double>(read.blockRanks[b0])));
    }
    read.residualInBlockMotion = worst;
  }

  // Parities are exact integers given the verified matching premise.
  read.certificate = Certificate::structureExact(
      CertificateDomain::Static, regime, 1.0 - read.minMatchOverlap,
      worstConditioning, 1.0 - cfg.blockMatchThreshold);
  return read;
}

// ---------------------------------------------------------------------------
// the constructed total-space spin holonomy cycle
// ---------------------------------------------------------------------------

int ExchangeHolonomy::spinorDimension(int d) {
  if (d == 3) return 2;
  if (d == 4) return 4;
  throw std::invalid_argument(
      "ExchangeHolonomy: the spinor layer is implemented at d = 3 (Pauli) "
      "and d = 4 (the documented Euclidean Dirac layer) only");
}

Eigen::MatrixXcd ExchangeHolonomy::gamma(int a, int d) {
  spinorDimension(d);  // validates d
  if (a < 0 || a >= d)
    throw std::invalid_argument("ExchangeHolonomy: gamma axis out of range");
  Eigen::Matrix2cd s1, s2, s3, id2;
  s1 << cd(0, 0), cd(1, 0), cd(1, 0), cd(0, 0);
  s2 << cd(0, 0), cd(0, -1), cd(0, 1), cd(0, 0);
  s3 << cd(1, 0), cd(0, 0), cd(0, 0), cd(-1, 0);
  id2.setIdentity();
  if (d == 3) {
    if (a == 0) return s1;
    if (a == 1) return s2;
    return s3;
  }
  // The d = 4 Euclidean layer: kron(s1, s1), kron(s1, s2), kron(s1, s3),
  // kron(s2, I).
  const auto kron2 = [](const Eigen::Matrix2cd &p, const Eigen::Matrix2cd &q) {
    Eigen::MatrixXcd out(4, 4);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        out.block(2 * i, 2 * j, 2, 2) = p(i, j) * q;
    return out;
  };
  if (a == 0) return kron2(s1, s1);
  if (a == 1) return kron2(s1, s2);
  if (a == 2) return kron2(s1, s3);
  return kron2(s2, id2);
}

Eigen::MatrixXcd ExchangeHolonomy::spinGenerator(int a, int b, int d) {
  if (a == b)
    throw std::invalid_argument(
        "ExchangeHolonomy: a spin generator needs two distinct axes");
  const Eigen::MatrixXcd ga = gamma(a, d);
  const Eigen::MatrixXcd gb = gamma(b, d);
  return 0.25 * (ga * gb - gb * ga);
}

Eigen::MatrixXcd ExchangeHolonomy::spinorRotation(double theta, int a, int b,
                                                  int d) {
  if (a == b)
    throw std::invalid_argument(
        "ExchangeHolonomy: a spinor rotation needs two distinct axes");
  const int dim = spinorDimension(d);
  const Eigen::MatrixXcd plane = gamma(a, d) * gamma(b, d);
  return std::cos(theta / 2.0) * Eigen::MatrixXcd::Identity(dim, dim) +
         std::sin(theta / 2.0) * plane;
}

Eigen::MatrixXcd ExchangeHolonomy::transverseSpinorFrame(int a, int b,
                                                         int d) {
  const int dim = spinorDimension(d);
  const Eigen::MatrixXcd sigma = spinGenerator(a, b, d);
  // i Sigma is Hermitian with eigenvalues -+1/2; the transverse line is the
  // equal superposition of one vector from each eigenspace, with a
  // deterministic phase convention.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(cd(0, 1) * sigma);
  Eigen::VectorXcd low = solver.eigenvectors().col(0);
  Eigen::VectorXcd high = solver.eigenvectors().col(dim - 1);
  canonicalizePhase(low);
  canonicalizePhase(high);
  Eigen::MatrixXcd frame(dim, 1);
  frame.col(0) = (low + high) / std::sqrt(2.0);
  return frame;
}

std::vector<Eigen::MatrixXcd> ExchangeHolonomy::rotationLoopFrames(
    const Eigen::MatrixXcd &frame0, int a, int b, int d, int turns,
    int steps) {
  const int dim = spinorDimension(d);
  if (frame0.rows() != dim)
    throw std::invalid_argument(
        "ExchangeHolonomy: rotation frame rows must equal the spinor "
        "dimension");
  if (steps < 3)
    throw std::invalid_argument(
        "ExchangeHolonomy: a rotation loop needs at least 3 steps");
  std::vector<Eigen::MatrixXcd> frames;
  frames.reserve(static_cast<std::size_t>(steps));
  const double total = 2.0 * M_PI * static_cast<double>(turns);
  for (int t = 0; t < steps; ++t) {
    const double theta = total * static_cast<double>(t) /
                         static_cast<double>(steps);
    frames.push_back(spinorRotation(theta, a, b, d) * frame0);
  }
  return frames;
}

std::vector<Eigen::MatrixXcd> ExchangeHolonomy::referenceLoopFrames(
    const Eigen::MatrixXcd &frame0, int steps) {
  if (steps < 1)
    throw std::invalid_argument(
        "ExchangeHolonomy: a reference loop needs at least 1 step");
  return std::vector<Eigen::MatrixXcd>(static_cast<std::size_t>(steps),
                                       frame0);
}

std::vector<Eigen::MatrixXcd> ExchangeHolonomy::vectorLoopFrames(
    const Eigen::MatrixXcd &frame0, int a, int b, int d, int turns,
    int steps) {
  if (d < 2 || a < 0 || b < 0 || a >= d || b >= d || a == b)
    throw std::invalid_argument(
        "ExchangeHolonomy: vector rotation needs two distinct axes < d");
  if (frame0.rows() != d)
    throw std::invalid_argument(
        "ExchangeHolonomy: vector frame rows must equal d");
  if (steps < 3)
    throw std::invalid_argument(
        "ExchangeHolonomy: a rotation loop needs at least 3 steps");
  std::vector<Eigen::MatrixXcd> frames;
  frames.reserve(static_cast<std::size_t>(steps));
  const double total = 2.0 * M_PI * static_cast<double>(turns);
  for (int t = 0; t < steps; ++t) {
    const double theta = total * static_cast<double>(t) /
                         static_cast<double>(steps);
    Eigen::MatrixXd rot = Eigen::MatrixXd::Identity(d, d);
    rot(a, a) = std::cos(theta);
    rot(b, b) = std::cos(theta);
    rot(a, b) = -std::sin(theta);
    rot(b, a) = std::sin(theta);
    frames.push_back(rot.cast<cd>() * frame0);
  }
  return frames;
}

// ---------------------------------------------------------------------------
// the total-space spin read
// ---------------------------------------------------------------------------

Eigen::MatrixXcd ExchangeHolonomy::totalJSquaredOperator(int constituents) {
  if (constituents < 1 || constituents > 10)
    throw std::invalid_argument(
        "ExchangeHolonomy: totalJSquaredOperator supports 1..10 "
        "constituents (dense 2^n cap)");
  const Eigen::Index dim = Eigen::Index{1} << constituents;
  Eigen::Matrix2cd s[3];
  s[0] << cd(0, 0), cd(0.5, 0), cd(0.5, 0), cd(0, 0);
  s[1] << cd(0, 0), cd(0, -0.5), cd(0, 0.5), cd(0, 0);
  s[2] << cd(0.5, 0), cd(0, 0), cd(0, 0), cd(-0.5, 0);

  Eigen::MatrixXcd j2 = Eigen::MatrixXcd::Zero(dim, dim);
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::MatrixXcd total = Eigen::MatrixXcd::Zero(dim, dim);
    for (int site = 0; site < constituents; ++site) {
      Eigen::MatrixXcd op = Eigen::MatrixXcd::Ones(1, 1);
      for (int k = 0; k < constituents; ++k) {
        const Eigen::MatrixXcd factor =
            k == site ? Eigen::MatrixXcd(s[axis])
                      : Eigen::MatrixXcd(Eigen::Matrix2cd::Identity());
        Eigen::MatrixXcd next(op.rows() * 2, op.cols() * 2);
        for (Eigen::Index i = 0; i < op.rows(); ++i)
          for (Eigen::Index j = 0; j < op.cols(); ++j)
            next.block(2 * i, 2 * j, 2, 2) = op(i, j) * factor;
        op = std::move(next);
      }
      total += op;
    }
    j2 += total * total;
  }
  return j2;
}

double ExchangeHolonomy::totalJSquared(const Eigen::VectorXcd &state) {
  const Eigen::Index size = state.size();
  int constituents = 0;
  Eigen::Index dim = 1;
  while (dim < size && constituents <= 10) {
    dim <<= 1;
    ++constituents;
  }
  if (dim != size || constituents < 1)
    throw std::invalid_argument(
        "ExchangeHolonomy: totalJSquared needs a (C^2)^(tensor n) state "
        "with 1 <= n <= 10");
  const double norm2 = state.squaredNorm();
  if (!(norm2 > 0.0))
    throw std::invalid_argument(
        "ExchangeHolonomy: totalJSquared of a zero state");
  const Eigen::MatrixXcd j2 = totalJSquaredOperator(constituents);
  return ((state.adjoint() * j2 * state).value() / norm2).real();
}

// ---------------------------------------------------------------------------
// SO(d) -> Spin(d) lift machinery
// ---------------------------------------------------------------------------

struct ExchangeHolonomy::PlaneDecomposition {
  // Orthonormal plane pairs (u_k, v_k) with angles theta_k in (-pi, pi], plus
  // the +1 fixed axes. exp(sum theta_k (v u^T - u v^T)) = R.
  std::vector<Eigen::VectorXd> u{};
  std::vector<Eigen::VectorXd> v{};
  std::vector<double> angles{};
  std::vector<Eigen::VectorXd> fixedAxes{};
  double residual = 0.0;
};

ExchangeHolonomy::PlaneDecomposition ExchangeHolonomy::planeDecomposition(
    const Eigen::MatrixXd &rotation) {
  const Eigen::Index n = rotation.rows();
  if (rotation.cols() != n)
    throw std::invalid_argument("ExchangeHolonomy: rotation must be square");
  const double orthoResidual =
      (rotation.transpose() * rotation - Eigen::MatrixXd::Identity(n, n))
          .norm();
  if (orthoResidual > 1e-9 || rotation.determinant() < 0.0)
    throw std::invalid_argument(
        "ExchangeHolonomy: not a special orthogonal matrix (orthogonality "
        "residual above 1e-9 or determinant negative)");

  Eigen::RealSchur<Eigen::MatrixXd> schur(rotation);
  const Eigen::MatrixXd &t = schur.matrixT();
  const Eigen::MatrixXd &q = schur.matrixU();

  PlaneDecomposition dec;
  std::vector<Eigen::Index> minusOnes;
  Eigen::Index i = 0;
  while (i < n) {
    if (i + 1 < n && std::abs(t(i + 1, i)) > 1e-12) {
      const double angle = std::atan2(t(i + 1, i), t(i, i));
      dec.u.push_back(q.col(i));
      dec.v.push_back(q.col(i + 1));
      dec.angles.push_back(angle);
      i += 2;
    } else {
      if (t(i, i) < 0.0) {
        minusOnes.push_back(i);
      } else {
        dec.fixedAxes.push_back(q.col(i));
      }
      i += 1;
    }
  }
  // det = +1 guarantees an even count of -1 eigenvalues; pair them into
  // angle-pi planes with the deterministic axis-sign rule. The pi-branch
  // convention fixes the branch representative, not the conjugacy class.
  for (std::size_t k = 0; k + 1 < minusOnes.size(); k += 2) {
    Eigen::VectorXd u = q.col(minusOnes[k]);
    Eigen::VectorXd v = q.col(minusOnes[k + 1]);
    canonicalizeSign(u);
    canonicalizeSign(v);
    dec.u.push_back(u);
    dec.v.push_back(v);
    dec.angles.push_back(M_PI);
  }

  // Verify the decomposition rebuilds R (the certificate residual).
  Eigen::MatrixXd rebuilt = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t k = 0; k < dec.angles.size(); ++k) {
    const double c = std::cos(dec.angles[k]);
    const double s = std::sin(dec.angles[k]);
    rebuilt += c * (dec.u[k] * dec.u[k].transpose() +
                    dec.v[k] * dec.v[k].transpose()) +
               s * (dec.v[k] * dec.u[k].transpose() -
                    dec.u[k] * dec.v[k].transpose());
  }
  for (const Eigen::VectorXd &axis : dec.fixedAxes)
    rebuilt += axis * axis.transpose();
  dec.residual = (rebuilt - rotation).norm();
  if (dec.residual > 1e-6)
    throw std::invalid_argument(
        "ExchangeHolonomy: rotation plane decomposition failed to rebuild "
        "the input (non-normal Schur form)");
  return dec;
}

Eigen::MatrixXd ExchangeHolonomy::rotationLog(
    const Eigen::MatrixXd &rotation) {
  const PlaneDecomposition dec = planeDecomposition(rotation);
  const Eigen::Index n = rotation.rows();
  Eigen::MatrixXd log = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t k = 0; k < dec.angles.size(); ++k)
    log += dec.angles[k] * (dec.v[k] * dec.u[k].transpose() -
                            dec.u[k] * dec.v[k].transpose());
  return log;
}

Eigen::MatrixXcd ExchangeHolonomy::rotationToSpin(
    const Eigen::MatrixXd &rotation, int d) {
  const int dim = spinorDimension(d);
  if (rotation.rows() != d || rotation.cols() != d)
    throw std::invalid_argument(
        "ExchangeHolonomy: rotationToSpin needs a d x d rotation");
  const PlaneDecomposition dec = planeDecomposition(rotation);

  Eigen::MatrixXcd spin = Eigen::MatrixXcd::Identity(dim, dim);
  for (std::size_t k = 0; k < dec.angles.size(); ++k) {
    Eigen::MatrixXcd gu = Eigen::MatrixXcd::Zero(dim, dim);
    Eigen::MatrixXcd gv = Eigen::MatrixXcd::Zero(dim, dim);
    for (int axis = 0; axis < d; ++axis) {
      gu += dec.u[k][axis] * gamma(axis, d);
      gv += dec.v[k][axis] * gamma(axis, d);
    }
    // For orthonormal u perp v the plane bivector gamma(u) gamma(v) squares
    // to -I, so the factor is the closed-form half-angle rotation; factors
    // over orthogonal planes commute. The minus sign selects the
    // covering-homomorphism orientation S gamma(x) S^{-1} = gamma(R x); with
    // +sin the identity comes out with R^{-1}, an anti-homomorphism that
    // breaks noncommuting lift compositions such as the Cech triangle
    // products in spinLift.
    spin = (std::cos(dec.angles[k] / 2.0) *
                Eigen::MatrixXcd::Identity(dim, dim) -
            std::sin(dec.angles[k] / 2.0) * (gu * gv)) *
           spin;
  }
  return spin;
}

LoopLiftRead ExchangeHolonomy::loopLiftCharacter(
    const std::vector<Eigen::MatrixXd> &loop, int d,
    const ExchangeHolonomyConfig &cfg) {
  LoopLiftRead read;
  if (loop.size() < 3)
    throw std::invalid_argument(
        "ExchangeHolonomy: a lifted SO(d) loop needs at least 3 samples");
  const int dim = spinorDimension(d);

  const std::size_t T = loop.size();
  double maxAngle = 0.0;
  bool branchSafe = true;
  Eigen::MatrixXcd product = Eigen::MatrixXcd::Identity(dim, dim);
  for (std::size_t t = 0; t < T; ++t) {
    const Eigen::MatrixXd increment =
        loop[(t + 1) % T] * loop[t].transpose();
    const PlaneDecomposition dec = planeDecomposition(increment);
    for (const double angle : dec.angles)
      maxAngle = std::max(maxAngle, std::abs(angle));
    if (maxAngle >= M_PI - cfg.liftAngleMargin) branchSafe = false;
    product = rotationToSpin(increment, d) * product;
  }
  read.maxStepAngle = maxAngle;

  const double toPlus = distanceToSignedIdentity(product, 1.0);
  const double toMinus = distanceToSignedIdentity(product, -1.0);
  read.closureResidual = std::min(toPlus, toMinus);
  const double tolerance =
      std::max(1e-12, cfg.unitaryTolerance * static_cast<double>(T));
  const bool certified =
      branchSafe && read.closureResidual <= tolerance;
  if (certified) read.character = toPlus <= toMinus ? +1 : -1;
  read.certificate =
      certified ? Certificate::structureExact(
                      CertificateDomain::Static,
                      CertificateRegime::NonNormal, read.closureResidual,
                      1.0, tolerance)
                : Certificate::heuristicDiscovery(
                      CertificateDomain::Static,
                      CertificateRegime::NonNormal);
  return read;
}

bool ExchangeHolonomy::gf2Solve(std::vector<int> matrix, int rows, int cols,
                                std::vector<int> rhs,
                                std::vector<int> &solution) {
  // Gaussian elimination over GF(2) on the augmented system; returns the
  // particular solution with free variables set to 0 when consistent.
  std::vector<int> pivotCol(rows, -1);
  int rank = 0;
  for (int c = 0; c < cols && rank < rows; ++c) {
    int pivot = -1;
    for (int r = rank; r < rows; ++r) {
      if (matrix[static_cast<std::size_t>(r) * cols + c] & 1) {
        pivot = r;
        break;
      }
    }
    if (pivot < 0) continue;
    for (int k = 0; k < cols; ++k)
      std::swap(matrix[static_cast<std::size_t>(pivot) * cols + k],
                matrix[static_cast<std::size_t>(rank) * cols + k]);
    std::swap(rhs[pivot], rhs[rank]);
    for (int r = 0; r < rows; ++r) {
      if (r != rank && (matrix[static_cast<std::size_t>(r) * cols + c] & 1)) {
        for (int k = 0; k < cols; ++k)
          matrix[static_cast<std::size_t>(r) * cols + k] ^=
              matrix[static_cast<std::size_t>(rank) * cols + k];
        rhs[r] ^= rhs[rank];
      }
    }
    pivotCol[rank] = c;
    ++rank;
  }
  for (int r = rank; r < rows; ++r)
    if (rhs[r] & 1) return false;  // inconsistent: the obstruction
  solution.assign(static_cast<std::size_t>(cols), 0);
  for (int r = 0; r < rank; ++r)
    solution[static_cast<std::size_t>(pivotCol[r])] = rhs[r] & 1;
  return true;
}

std::string SpinLiftRead::describe() const {
  std::ostringstream out;
  out << "SpinLiftRead(";
  if (!certificate.holds() && !liftExists && !obstructed) {
    out << "UNCERTIFIED";
  } else if (liftExists) {
    out << "lift accepted";
  } else {
    out << "OBSTRUCTED (w2 nontrivial)";
  }
  out << ", triangles=" << triangleSigns.size()
      << ", cocycle residual=" << maxCocycleResidual
      << ", lift residual=" << maxLiftResidual << ")";
  return out.str();
}

SpinLiftRead ExchangeHolonomy::spinLift(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> &edges,
    const std::vector<Eigen::MatrixXd> &edgeRotations,
    const std::vector<std::vector<std::uint64_t>> &triangles, int d,
    const ExchangeHolonomyConfig &cfg) {
  spinorDimension(d);  // validates d
  if (edges.size() != edgeRotations.size())
    throw std::invalid_argument(
        "ExchangeHolonomy::spinLift: one rotation per edge required");
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> edgeIndex;
  for (std::size_t e = 0; e < edges.size(); ++e) {
    if (edges[e].first == edges[e].second)
      throw std::invalid_argument(
          "ExchangeHolonomy::spinLift: self-loop edge");
    if (edgeRotations[e].rows() != d || edgeRotations[e].cols() != d)
      throw std::invalid_argument(
          "ExchangeHolonomy::spinLift: edge rotations must be d x d");
    const auto key = std::minmax(edges[e].first, edges[e].second);
    if (!edgeIndex.emplace(std::make_pair(key.first, key.second), e).second)
      throw std::invalid_argument(
          "ExchangeHolonomy::spinLift: duplicate unordered edge");
  }

  // The directed transition g(i -> j) and its principal lift. g(j -> i) is
  // the transpose, lifted independently by the same principal rule.
  const auto directedRotation = [&](std::uint64_t i, std::uint64_t j) {
    const auto key = std::minmax(i, j);
    const auto hit = edgeIndex.find(std::make_pair(key.first, key.second));
    if (hit == edgeIndex.end())
      throw std::invalid_argument(
          "ExchangeHolonomy::spinLift: a triangle edge is missing from the "
          "edge list");
    const Eigen::MatrixXd &g = edgeRotations[hit->second];
    return edges[hit->second].first == i ? g : Eigen::MatrixXd(g.transpose());
  };

  SpinLiftRead read;
  read.maxCocycleResidual = 0.0;
  read.maxLiftResidual = 0.0;
  std::vector<int> w01;
  for (const auto &tri : triangles) {
    if (tri.size() != 3)
      throw std::invalid_argument(
          "ExchangeHolonomy::spinLift: triangles must have 3 vertices");
    const Eigen::MatrixXd gij = directedRotation(tri[0], tri[1]);
    const Eigen::MatrixXd gjk = directedRotation(tri[1], tri[2]);
    const Eigen::MatrixXd gki = directedRotation(tri[2], tri[0]);
    const double cocycleResidual =
        (gij * gjk * gki - Eigen::MatrixXd::Identity(d, d)).norm();
    read.maxCocycleResidual =
        std::max(read.maxCocycleResidual, cocycleResidual);

    const Eigen::MatrixXcd lifted = rotationToSpin(gij, d) *
                                    rotationToSpin(gjk, d) *
                                    rotationToSpin(gki, d);
    const double toPlus = distanceToSignedIdentity(lifted, 1.0);
    const double toMinus = distanceToSignedIdentity(lifted, -1.0);
    const int sign = toPlus <= toMinus ? +1 : -1;
    read.triangleSigns.push_back(sign);
    read.maxLiftResidual =
        std::max(read.maxLiftResidual, std::min(toPlus, toMinus));
    w01.push_back(sign < 0 ? 1 : 0);
  }

  const bool premiseOk =
      read.maxCocycleResidual <= cfg.cocycleTolerance &&
      read.maxLiftResidual <= cfg.cocycleTolerance;
  if (!premiseOk) {
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::Static, CertificateRegime::NonNormal);
    return read;
  }

  // Exact GF(2) coboundary decision via cobordism::gf2Rank
  // (rank(D) == rank([D | w])); the witness signs come from the augmented
  // solve, also exact.
  const int rows = static_cast<int>(triangles.size());
  const int cols = static_cast<int>(edges.size());
  std::vector<int> incidence(static_cast<std::size_t>(rows) * cols, 0);
  for (int t = 0; t < rows; ++t) {
    const auto &tri = triangles[static_cast<std::size_t>(t)];
    for (int corner = 0; corner < 3; ++corner) {
      const auto key = std::minmax(tri[static_cast<std::size_t>(corner)],
                                   tri[(corner + 1) % 3]);
      const std::size_t e =
          edgeIndex.at(std::make_pair(key.first, key.second));
      incidence[static_cast<std::size_t>(t) * cols + static_cast<int>(e)] ^=
          1;
    }
  }
  std::vector<int> augmented(static_cast<std::size_t>(rows) * (cols + 1));
  for (int t = 0; t < rows; ++t) {
    for (int e = 0; e < cols; ++e)
      augmented[static_cast<std::size_t>(t) * (cols + 1) + e] =
          incidence[static_cast<std::size_t>(t) * cols + e];
    augmented[static_cast<std::size_t>(t) * (cols + 1) + cols] =
        w01[static_cast<std::size_t>(t)];
  }
  const bool coboundary =
      cobordism::gf2Rank(incidence, rows, cols) ==
      cobordism::gf2Rank(augmented, rows, cols + 1);

  std::vector<int> flips;
  const bool solved = gf2Solve(incidence, rows, cols, w01, flips);
  if (solved != coboundary)
    throw std::logic_error(
        "ExchangeHolonomy::spinLift: gf2Rank and the augmented solve "
        "disagree on the coboundary decision");

  read.liftExists = coboundary;
  read.obstructed = !coboundary;
  if (coboundary) {
    read.edgeSigns.reserve(flips.size());
    for (const int f : flips) read.edgeSigns.push_back(f ? -1 : +1);
  }
  read.certificate = Certificate::structureExact(
      CertificateDomain::Static, CertificateRegime::NonNormal,
      std::max(read.maxCocycleResidual, read.maxLiftResidual), 1.0,
      cfg.cocycleTolerance);
  return read;
}

// ---------------------------------------------------------------------------
// channel-separation gauge actions
// ---------------------------------------------------------------------------

std::vector<Eigen::MatrixXcd> ExchangeHolonomy::reorientedFrames(
    const std::vector<Eigen::MatrixXcd> &frames,
    const std::vector<int> &cellSigns) {
  for (const int s : cellSigns)
    if (s != 1 && s != -1)
      throw std::invalid_argument(
          "ExchangeHolonomy: reorientation signs must be -1 or +1");
  std::vector<Eigen::MatrixXcd> out;
  out.reserve(frames.size());
  for (const Eigen::MatrixXcd &f : frames) {
    if (static_cast<std::size_t>(f.rows()) != cellSigns.size())
      throw std::invalid_argument(
          "ExchangeHolonomy: one reorientation sign per cell row required");
    Eigen::MatrixXcd g = f;
    for (Eigen::Index r = 0; r < g.rows(); ++r)
      if (cellSigns[static_cast<std::size_t>(r)] < 0) g.row(r) = -g.row(r);
    out.push_back(std::move(g));
  }
  return out;
}

std::vector<Eigen::MatrixXcd> ExchangeHolonomy::permutedCellFrames(
    const std::vector<Eigen::MatrixXcd> &frames,
    const std::vector<std::size_t> &rowPermutation) {
  std::vector<bool> seen(rowPermutation.size(), false);
  for (const std::size_t p : rowPermutation) {
    if (p >= rowPermutation.size() || seen[p])
      throw std::invalid_argument(
          "ExchangeHolonomy: rowPermutation must be a bijection");
    seen[p] = true;
  }
  std::vector<Eigen::MatrixXcd> out;
  out.reserve(frames.size());
  for (const Eigen::MatrixXcd &f : frames) {
    if (static_cast<std::size_t>(f.rows()) != rowPermutation.size())
      throw std::invalid_argument(
          "ExchangeHolonomy: rowPermutation size must equal the row count");
    Eigen::MatrixXcd g(f.rows(), f.cols());
    for (Eigen::Index r = 0; r < g.rows(); ++r)
      g.row(r) = f.row(static_cast<Eigen::Index>(
          rowPermutation[static_cast<std::size_t>(r)]));
    out.push_back(std::move(g));
  }
  return out;
}

}  // namespace tessera::observables
