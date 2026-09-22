// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/DressedFluctuation.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace tessera::cobordism {
namespace {

using complexd = std::complex<double>;

Eigen::MatrixXcd toMatrix(const std::vector<complexd> &flat, std::size_t rows,
                          std::size_t columns) {
  Eigen::MatrixXcd matrix(static_cast<Eigen::Index>(rows),
                          static_cast<Eigen::Index>(columns));
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column)
      matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
          flat[row * columns + column];
  return matrix;
}

std::vector<complexd> toFlat(const Eigen::MatrixXcd &matrix) {
  std::vector<complexd> flat(static_cast<std::size_t>(matrix.size()));
  for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    for (Eigen::Index column = 0; column < matrix.cols(); ++column)
      flat[static_cast<std::size_t>(row * matrix.cols() + column)] =
          matrix(row, column);
  return flat;
}

// The position of the pair (a, b) with a <= b in the row-major upper triangle
// of an R x R array.
std::size_t upperTrianglePosition(std::size_t a, std::size_t b,
                                  std::size_t count) {
  if (a > b) std::swap(a, b);
  return a * count - (a * (a + 1)) / 2 + b;
}

// The ascending N-element subsets of {0, ..., total - 1} in lexicographic
// order.
std::vector<std::vector<std::size_t>> subsets(std::size_t total,
                                              std::size_t chosen) {
  std::vector<std::vector<std::size_t>> out;
  if (chosen > total) return out;
  std::vector<std::size_t> current(chosen);
  std::iota(current.begin(), current.end(), std::size_t{0});
  while (true) {
    out.push_back(current);
    if (chosen == 0) break;
    std::size_t position = chosen;
    while (position > 0 && current[position - 1] == total - chosen + position - 1)
      --position;
    if (position == 0) break;
    ++current[position - 1];
    for (std::size_t later = position; later < chosen; ++later)
      current[later] = current[later - 1] + 1;
  }
  return out;
}

}  // namespace

DressedFluctuation::DressedFluctuation(DressedFluctuationDeclaration declaration)
    : declaration_(std::move(declaration)) {
  if (declaration_.carrierDimension <= 0)
    throw std::invalid_argument(
        "DressedFluctuation: the carrier dimension must be positive; got " +
        std::to_string(declaration_.carrierDimension));
  dimension_ = static_cast<std::size_t>(declaration_.carrierDimension);
  const std::size_t square = dimension_ * dimension_;
  if (declaration_.carrier.size() != square)
    throw std::invalid_argument(
        "DressedFluctuation: the carrier operator must be a square matrix over "
        "the " +
        std::to_string(dimension_) + " carrier coordinates; got " +
        std::to_string(declaration_.carrier.size()) + " entries");
  fluctuations_ = declaration_.couplings.size();
  for (std::size_t index = 0; index < fluctuations_; ++index)
    if (declaration_.couplings[index].size() != square)
      throw std::invalid_argument(
          "DressedFluctuation: coupling operator " + std::to_string(index) +
          " must be a square matrix over the " + std::to_string(dimension_) +
          " carrier coordinates; got " +
          std::to_string(declaration_.couplings[index].size()) + " entries");
  const std::size_t expectedSecond = (fluctuations_ * (fluctuations_ + 1)) / 2;
  if (!declaration_.secondDerivatives.empty() &&
      declaration_.secondDerivatives.size() != expectedSecond)
    throw std::invalid_argument(
        "DressedFluctuation: the second derivatives must be the " +
        std::to_string(expectedSecond) +
        " upper-triangular pairs of the declared fluctuations, or none at all; "
        "got " +
        std::to_string(declaration_.secondDerivatives.size()));
  for (std::size_t index = 0; index < declaration_.secondDerivatives.size();
       ++index)
    if (declaration_.secondDerivatives[index].size() != square)
      throw std::invalid_argument(
          "DressedFluctuation: second derivative " + std::to_string(index) +
          " must be a square matrix over the " + std::to_string(dimension_) +
          " carrier coordinates; got " +
          std::to_string(declaration_.secondDerivatives[index].size()) +
          " entries");
  if (!declaration_.bareStiffness.empty() &&
      declaration_.bareStiffness.size() != fluctuations_ * fluctuations_)
    throw std::invalid_argument(
        "DressedFluctuation: the bare stiffness must be a square matrix over "
        "the " +
        std::to_string(fluctuations_) + " declared fluctuations; got " +
        std::to_string(declaration_.bareStiffness.size()) + " entries");
  if (declaration_.occupiedModes > dimension_)
    throw std::invalid_argument(
        "DressedFluctuation: " + std::to_string(declaration_.occupiedModes) +
        " modes are declared occupied but the carrier has only " +
        std::to_string(dimension_));
  if (!(declaration_.continuumBroadening >= 0.0))
    throw std::invalid_argument(
        "DressedFluctuation: the declared continuum broadening must be zero or "
        "positive");

  const Eigen::MatrixXcd carrier =
      toMatrix(declaration_.carrier, dimension_, dimension_);
  const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(carrier);
  if (solver.info() != Eigen::Success)
    throw std::invalid_argument(
        "DressedFluctuation: the carrier operator has no eigendecomposition, so "
        "its modes cannot be split into occupied and empty ones");
  const Eigen::VectorXcd values = solver.eigenvalues();
  std::vector<Eigen::Index> order(static_cast<std::size_t>(values.size()));
  std::iota(order.begin(), order.end(), Eigen::Index{0});
  const bool byRealPart =
      declaration_.occupationOrder == OccupationOrder::AscendingRealPart;
  std::stable_sort(order.begin(), order.end(),
                   [&values, byRealPart](Eigen::Index a, Eigen::Index b) {
                     if (byRealPart) {
                       if (values(a).real() != values(b).real())
                         return values(a).real() < values(b).real();
                       return values(a).imag() < values(b).imag();
                     }
                     return std::abs(values(a)) < std::abs(values(b));
                   });
  Eigen::MatrixXcd right(static_cast<Eigen::Index>(dimension_),
                         static_cast<Eigen::Index>(dimension_));
  eigenvalues_.resize(dimension_);
  for (std::size_t column = 0; column < dimension_; ++column) {
    right.col(static_cast<Eigen::Index>(column)) =
        solver.eigenvectors().col(order[column]);
    eigenvalues_[column] = values(order[column]);
  }
  const Eigen::FullPivLU<Eigen::MatrixXcd> factor(right);
  if (!factor.isInvertible())
    throw std::invalid_argument(
        "DressedFluctuation: the carrier operator is defective, so it carries "
        "no matched pair of left and right mode frames");
  const Eigen::MatrixXcd left = factor.inverse();
  rightModes_ = toFlat(right);
  leftModes_ = toFlat(left);

  modeCurrents_.resize(fluctuations_);
  for (std::size_t index = 0; index < fluctuations_; ++index) {
    const Eigen::MatrixXcd coupling =
        toMatrix(declaration_.couplings[index], dimension_, dimension_);
    modeCurrents_[index] = toFlat(left * coupling * right);
  }

  diamagnetic_.assign(fluctuations_ * fluctuations_, complexd{0.0, 0.0});
  if (!declaration_.secondDerivatives.empty()) {
    const auto occupied = declaration_.occupiedModes;
    for (std::size_t a = 0; a < fluctuations_; ++a)
      for (std::size_t b = a; b < fluctuations_; ++b) {
        const Eigen::MatrixXcd second = toMatrix(
            declaration_.secondDerivatives[upperTrianglePosition(a, b,
                                                                 fluctuations_)],
            dimension_, dimension_);
        const Eigen::MatrixXcd transformed = left * second * right;
        complexd value{0.0, 0.0};
        for (std::size_t mode = 0; mode < occupied; ++mode)
          value += transformed(static_cast<Eigen::Index>(mode),
                               static_cast<Eigen::Index>(mode));
        diamagnetic_[a * fluctuations_ + b] = value;
        diamagnetic_[b * fluctuations_ + a] = value;
      }
  }
}

std::size_t DressedFluctuation::carrierDimension() const noexcept {
  return dimension_;
}

std::size_t DressedFluctuation::fluctuationCount() const noexcept {
  return fluctuations_;
}

std::size_t DressedFluctuation::particleHolePairCount() const noexcept {
  return declaration_.occupiedModes * (dimension_ - declaration_.occupiedModes);
}

std::vector<complexd> DressedFluctuation::carrierEigenvalues() const {
  return eigenvalues_;
}

std::vector<complexd> DressedFluctuation::particleHoleEnergies() const {
  std::vector<complexd> energies;
  energies.reserve(particleHolePairCount());
  for (std::size_t occupied = 0; occupied < declaration_.occupiedModes;
       ++occupied)
    for (std::size_t empty = declaration_.occupiedModes; empty < dimension_;
         ++empty)
      energies.push_back(eigenvalues_[empty] - eigenvalues_[occupied]);
  return energies;
}

std::vector<complexd> DressedFluctuation::modeCurrents(std::size_t index) const {
  if (index >= fluctuations_)
    throw std::out_of_range("DressedFluctuation::modeCurrents: " +
                            std::to_string(index) +
                            " names no declared fluctuation; there are " +
                            std::to_string(fluctuations_));
  return modeCurrents_[index];
}

std::vector<complexd> DressedFluctuation::diamagnetic() const {
  return diamagnetic_;
}

std::vector<complexd> DressedFluctuation::paramagnetic(
    complexd frequency) const {
  std::vector<complexd> polarization(fluctuations_ * fluctuations_,
                                     complexd{0.0, 0.0});
  const complexd broadening{0.0, -declaration_.continuumBroadening};
  for (std::size_t occupied = 0; occupied < declaration_.occupiedModes;
       ++occupied)
    for (std::size_t empty = declaration_.occupiedModes; empty < dimension_;
         ++empty) {
      const complexd gap =
          eigenvalues_[empty] - eigenvalues_[occupied] + broadening;
      const complexd denominator = gap * gap - frequency * frequency;
      if (std::abs(denominator) <=
          declaration_.tolerance *
              std::max(1.0, std::abs(gap * gap) + std::abs(frequency * frequency)))
        throw std::domain_error(
            "DressedFluctuation::paramagnetic: the requested frequency sits on "
            "the particle-hole energy of the pair (" +
            std::to_string(occupied) + ", " + std::to_string(empty) +
            "), where the polarization has a pole and no finite value");
      const complexd weight = gap / denominator;
      for (std::size_t a = 0; a < fluctuations_; ++a)
        for (std::size_t b = 0; b < fluctuations_; ++b) {
          const complexd forward =
              modeCurrents_[a][occupied * dimension_ + empty] *
              modeCurrents_[b][empty * dimension_ + occupied];
          const complexd reverse =
              modeCurrents_[a][empty * dimension_ + occupied] *
              modeCurrents_[b][occupied * dimension_ + empty];
          polarization[a * fluctuations_ + b] += weight * (forward + reverse);
        }
    }
  return polarization;
}

std::vector<complexd> DressedFluctuation::dressedStiffness(
    complexd frequency) const {
  std::vector<complexd> dressed = paramagnetic(frequency);
  for (std::size_t entry = 0; entry < dressed.size(); ++entry) {
    dressed[entry] = diamagnetic_[entry] - dressed[entry];
    if (!declaration_.bareStiffness.empty())
      dressed[entry] += declaration_.bareStiffness[entry];
  }
  return dressed;
}

std::vector<complexd> DressedFluctuation::inducedStiffness() const {
  std::vector<complexd> induced = paramagnetic(complexd{0.0, 0.0});
  for (std::size_t entry = 0; entry < induced.size(); ++entry)
    induced[entry] = diamagnetic_[entry] - induced[entry];
  return induced;
}

double DressedFluctuation::wardResidual(
    const std::vector<complexd> &gaugeDirection) const {
  if (gaugeDirection.size() != fluctuations_)
    throw std::invalid_argument(
        "DressedFluctuation::wardResidual: a pure-gauge direction must have "
        "one entry per declared fluctuation (" +
        std::to_string(fluctuations_) + "); got " +
        std::to_string(gaugeDirection.size()));
  const Eigen::MatrixXcd induced =
      toMatrix(inducedStiffness(), fluctuations_, fluctuations_);
  Eigen::VectorXcd direction(static_cast<Eigen::Index>(fluctuations_));
  for (std::size_t entry = 0; entry < fluctuations_; ++entry)
    direction(static_cast<Eigen::Index>(entry)) = gaugeDirection[entry];
  const double directionNorm = direction.norm();
  const double operatorNorm = induced.norm();
  if (directionNorm == 0.0 || operatorNorm == 0.0) return 0.0;
  return (induced * direction).norm() / (operatorNorm * directionNorm);
}

Certificate DressedFluctuation::wardCertificate(
    const std::vector<std::vector<complexd>> &gaugeDirections) const {
  double worst = Certificate::kUnmeasured;
  for (const auto &direction : gaugeDirections) {
    const double residual = wardResidual(direction);
    worst = std::isfinite(worst) ? std::max(worst, residual) : residual;
  }
  return Certificate::algebraicallyExact(CertificateDomain::Static,
                                         CertificateRegime::ComplexSymmetricPencil,
                                         worst, declaration_.tolerance);
}

std::vector<CollectiveMode> DressedFluctuation::collectiveModes() const {
  std::vector<CollectiveMode> modes;
  if (fluctuations_ == 0) return modes;

  // The channels of the partial-fraction expansion: each particle-hole pair
  // contributes the two rank-one factors of the symmetrized numerator, so a
  // pair with matrix elements u and v gives the factor pairs (u/2, v) and
  // (v/2, u).
  const complexd broadening{0.0, -declaration_.continuumBroadening};
  std::vector<complexd> channelEnergy;
  std::vector<Eigen::VectorXcd> channelLeft;
  std::vector<Eigen::VectorXcd> channelRight;
  std::vector<complexd> pairEnergies;
  for (std::size_t occupied = 0; occupied < declaration_.occupiedModes;
       ++occupied)
    for (std::size_t empty = declaration_.occupiedModes; empty < dimension_;
         ++empty) {
      const complexd bare = eigenvalues_[empty] - eigenvalues_[occupied];
      pairEnergies.push_back(bare);
      const complexd gap = bare + broadening;
      Eigen::VectorXcd forward(static_cast<Eigen::Index>(fluctuations_));
      Eigen::VectorXcd reverse(static_cast<Eigen::Index>(fluctuations_));
      for (std::size_t entry = 0; entry < fluctuations_; ++entry) {
        forward(static_cast<Eigen::Index>(entry)) =
            modeCurrents_[entry][occupied * dimension_ + empty];
        reverse(static_cast<Eigen::Index>(entry)) =
            modeCurrents_[entry][empty * dimension_ + occupied];
      }
      channelEnergy.push_back(gap);
      channelLeft.push_back(forward);
      channelRight.push_back(reverse);
      channelEnergy.push_back(gap);
      channelLeft.push_back(reverse);
      channelRight.push_back(forward);
    }
  const auto channels = channelEnergy.size();
  const auto fluctuations = static_cast<Eigen::Index>(fluctuations_);
  const auto channelCount = static_cast<Eigen::Index>(channels);
  const Eigen::Index order = fluctuations + 2 * channelCount;

  Eigen::MatrixXcd pencil = Eigen::MatrixXcd::Zero(order, order);
  Eigen::MatrixXcd mass = Eigen::MatrixXcd::Zero(order, order);
  Eigen::MatrixXcd bare = Eigen::MatrixXcd::Zero(fluctuations, fluctuations);
  for (std::size_t a = 0; a < fluctuations_; ++a)
    for (std::size_t b = 0; b < fluctuations_; ++b) {
      complexd value = diamagnetic_[a * fluctuations_ + b];
      if (!declaration_.bareStiffness.empty())
        value += declaration_.bareStiffness[a * fluctuations_ + b];
      bare(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) = value;
    }
  pencil.topLeftCorner(fluctuations, fluctuations) = bare;
  for (Eigen::Index channel = 0; channel < channelCount; ++channel) {
    const auto index = static_cast<std::size_t>(channel);
    const Eigen::VectorXcd column = 0.5 * channelLeft[index];
    const Eigen::VectorXcd row = channelRight[index];
    pencil.block(0, fluctuations + channel, fluctuations, 1) = -column;
    pencil.block(0, fluctuations + channelCount + channel, fluctuations, 1) =
        -column;
    pencil.block(fluctuations + channel, 0, 1, fluctuations) =
        -row.transpose();
    pencil.block(fluctuations + channelCount + channel, 0, 1, fluctuations) =
        row.transpose();
    pencil(fluctuations + channel, fluctuations + channel) =
        channelEnergy[index];
    pencil(fluctuations + channelCount + channel,
           fluctuations + channelCount + channel) = -channelEnergy[index];
    mass(fluctuations + channel, fluctuations + channel) = complexd{1.0, 0.0};
    mass(fluctuations + channelCount + channel,
         fluctuations + channelCount + channel) = complexd{1.0, 0.0};
  }

  // The finite eigenvalues of the pencil are the reciprocals of the nonzero
  // eigenvalues of (pencil - shift * mass)^{-1} mass, shifted back. The shift
  // is needed because the dressed stiffness is singular at zero frequency
  // whenever a pure-gauge direction exists, which is exactly the Ward identity.
  double scale = std::max(1.0, bare.norm());
  for (const complexd &energy : channelEnergy)
    scale = std::max(scale, std::abs(energy));
  const std::vector<complexd> candidateShifts{
      complexd{0.3719, 0.2341}, complexd{-0.6131, 0.4517},
      complexd{1.2837, -0.7193}, complexd{-1.9043, -1.1287}};
  Eigen::MatrixXcd resolvent;
  complexd shift{0.0, 0.0};
  bool shifted = false;
  for (const complexd &candidate : candidateShifts) {
    const complexd trial = candidate * scale;
    const Eigen::FullPivLU<Eigen::MatrixXcd> factor(pencil - trial * mass);
    if (!factor.isInvertible()) continue;
    resolvent = factor.solve(mass);
    shift = trial;
    shifted = true;
    break;
  }
  if (!shifted) return modes;

  const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(resolvent);
  if (solver.info() != Eigen::Success) return modes;

  for (Eigen::Index index = 0; index < solver.eigenvalues().size(); ++index) {
    const complexd reciprocal = solver.eigenvalues()(index);
    // A vanishing reciprocal is an infinite eigenvalue of the pencil, which the
    // deflated geometric block contributes and which is no frequency at all.
    if (std::abs(reciprocal) * scale <= 1e-10) continue;
    const complexd frequency = shift + complexd{1.0, 0.0} / reciprocal;
    bool onParticleHoleEnergy = false;
    for (const complexd &energy : channelEnergy)
      if (std::abs(frequency - energy) <= declaration_.tolerance * scale ||
          std::abs(frequency + energy) <= declaration_.tolerance * scale)
        onParticleHoleEnergy = true;
    if (onParticleHoleEnergy) continue;

    Eigen::VectorXcd geometric =
        solver.eigenvectors().col(index).head(fluctuations);
    const double whole = solver.eigenvectors().col(index).norm();
    if (whole == 0.0 || geometric.norm() <= declaration_.tolerance * whole)
      continue;
    geometric /= geometric.norm();

    Eigen::MatrixXcd dressed;
    try {
      dressed =
          toMatrix(dressedStiffness(frequency), fluctuations_, fluctuations_);
    } catch (const std::domain_error &) {
      continue;
    }
    const double operatorNorm = dressed.norm();
    if (operatorNorm == 0.0) continue;
    const double residual = (dressed * geometric).norm() / operatorNorm;
    if (!(residual <= declaration_.tolerance)) continue;

    CollectiveMode mode;
    mode.frequency = frequency;
    mode.polarization.resize(fluctuations_);
    for (std::size_t entry = 0; entry < fluctuations_; ++entry)
      mode.polarization[entry] = geometric(static_cast<Eigen::Index>(entry));
    mode.radiationRate = -2.0 * frequency.imag();
    mode.residual = residual;
    if (pairEnergies.empty()) {
      mode.continuumDistance = Certificate::kUnmeasured;
      mode.insideParticleHoleContinuum = false;
    } else {
      double nearest = std::abs(frequency - pairEnergies.front());
      double lowest = pairEnergies.front().real();
      double highest = pairEnergies.front().real();
      for (const complexd &energy : pairEnergies) {
        nearest = std::min(nearest, std::abs(frequency - energy));
        nearest = std::min(nearest, std::abs(frequency + energy));
        lowest = std::min(lowest, energy.real());
        highest = std::max(highest, energy.real());
      }
      mode.continuumDistance = nearest;
      const double real = std::abs(frequency.real());
      mode.insideParticleHoleContinuum = real >= lowest && real <= highest;
    }
    mode.certificate = Certificate::algebraicallyExact(
        CertificateDomain::BandWindow, CertificateRegime::ComplexSymmetricPencil,
        residual, declaration_.tolerance);
    modes.push_back(std::move(mode));
  }

  std::stable_sort(modes.begin(), modes.end(),
                   [](const CollectiveMode &a, const CollectiveMode &b) {
                     if (a.frequency.real() != b.frequency.real())
                       return a.frequency.real() < b.frequency.real();
                     return a.frequency.imag() < b.frequency.imag();
                   });
  return modes;
}

ManyBodySpaceRead DressedFluctuation::effectiveAction(
    const std::vector<complexd> &clusterFrame,
    const std::vector<complexd> &clusterDualFrame, std::size_t particles,
    std::size_t dimensionCap) const {
  if (declaration_.bareStiffness.empty())
    throw std::invalid_argument(
        "DressedFluctuation::effectiveAction: the elimination inverts the bare "
        "stiffness A, and none is declared");
  std::size_t rank = dimension_;
  Eigen::MatrixXcd right =
      Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(dimension_),
                                 static_cast<Eigen::Index>(dimension_));
  Eigen::MatrixXcd left = right;
  if (!clusterFrame.empty() || !clusterDualFrame.empty()) {
    if (clusterFrame.empty() || clusterDualFrame.empty())
      throw std::invalid_argument(
          "DressedFluctuation::effectiveAction: a cluster fiber is declared by "
          "both of its frames or by neither");
    if (clusterFrame.size() % dimension_ != 0)
      throw std::invalid_argument(
          "DressedFluctuation::effectiveAction: the cluster frame must be a "
          "carrier-dimension by rank matrix");
    rank = clusterFrame.size() / dimension_;
    if (clusterDualFrame.size() != rank * dimension_)
      throw std::invalid_argument(
          "DressedFluctuation::effectiveAction: the two cluster frames declare "
          "different fiber ranks");
    right = toMatrix(clusterFrame, dimension_, rank);
    left = toMatrix(clusterDualFrame, rank, dimension_);
  }
  if (particles > rank)
    throw std::invalid_argument(
        "DressedFluctuation::effectiveAction: " + std::to_string(particles) +
        " particles do not fit in a fiber of rank " + std::to_string(rank));

  const auto basis = subsets(rank, particles);
  if (basis.size() > dimensionCap)
    throw std::length_error(
        "DressedFluctuation::effectiveAction: the " + std::to_string(particles) +
        "-particle space of a rank-" + std::to_string(rank) +
        " fiber has dimension " + std::to_string(basis.size()) +
        ", above the declared cap of " + std::to_string(dimensionCap));

  ManyBodySpaceRead read;
  read.particles = particles;
  read.fiberRank = rank;
  read.dimension = basis.size();
  read.basis = basis;
  read.framePairingDefect =
      (left * right -
       Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(rank),
                                  static_cast<Eigen::Index>(rank)))
          .norm();

  // The one-body operators of the fiber: the carrier and every current,
  // restricted by the declared frames with the transpose pairing.
  const Eigen::MatrixXcd carrier =
      left * toMatrix(declaration_.carrier, dimension_, dimension_) * right;
  std::vector<Eigen::MatrixXcd> currents(fluctuations_);
  for (std::size_t index = 0; index < fluctuations_; ++index)
    currents[index] =
        left * toMatrix(declaration_.couplings[index], dimension_, dimension_) *
        right;

  const auto secondQuantize = [&basis, rank,
                               particles](const Eigen::MatrixXcd &operatorMatrix) {
    const auto order = static_cast<Eigen::Index>(basis.size());
    Eigen::MatrixXcd lifted = Eigen::MatrixXcd::Zero(order, order);
    std::vector<std::size_t> reduced;
    std::vector<std::size_t> grown;
    for (std::size_t column = 0; column < basis.size(); ++column) {
      const auto &occupation = basis[column];
      for (std::size_t position = 0; position < particles; ++position) {
        const std::size_t annihilated = occupation[position];
        const double annihilationSign = (position % 2 == 0) ? 1.0 : -1.0;
        reduced.assign(occupation.begin(), occupation.end());
        reduced.erase(reduced.begin() +
                      static_cast<std::ptrdiff_t>(position));
        for (std::size_t created = 0; created < rank; ++created) {
          const complexd entry =
              operatorMatrix(static_cast<Eigen::Index>(created),
                             static_cast<Eigen::Index>(annihilated));
          if (entry == complexd{0.0, 0.0}) continue;
          const auto place =
              std::lower_bound(reduced.begin(), reduced.end(), created);
          if (place != reduced.end() && *place == created) continue;
          const auto offset = static_cast<std::size_t>(place - reduced.begin());
          grown.assign(reduced.begin(), reduced.end());
          grown.insert(grown.begin() + static_cast<std::ptrdiff_t>(offset),
                       created);
          const double creationSign = (offset % 2 == 0) ? 1.0 : -1.0;
          const auto row = static_cast<std::size_t>(
              std::lower_bound(basis.begin(), basis.end(), grown) -
              basis.begin());
          lifted(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(column)) +=
              annihilationSign * creationSign * entry;
        }
      }
    }
    return lifted;
  };

  const Eigen::MatrixXcd oneBody = secondQuantize(carrier);
  read.oneBody = toFlat(oneBody);

  const Eigen::MatrixXcd stiffness =
      toMatrix(declaration_.bareStiffness, fluctuations_, fluctuations_);
  read.stiffnessAsymmetry =
      stiffness.norm() == 0.0
          ? 0.0
          : (stiffness - stiffness.transpose()).norm() / stiffness.norm();
  const Eigen::FullPivLU<Eigen::MatrixXcd> factor(stiffness);
  if (!factor.isInvertible())
    throw std::invalid_argument(
        "DressedFluctuation::effectiveAction: the declared bare stiffness is "
        "singular, so the fluctuation cannot be eliminated at its saddle");
  const auto fluctuationOrder = static_cast<Eigen::Index>(fluctuations_);
  const Eigen::MatrixXcd identity =
      Eigen::MatrixXcd::Identity(fluctuationOrder, fluctuationOrder);
  // The whole interaction is factored through the R x R geometric response, and
  // R is the number of retained fluctuations rather than any many-body
  // dimension, so the response is formed once and no four-index tensor is ever
  // built.
  const Eigen::MatrixXcd response = factor.solve(identity);
  read.stiffnessConditioning = stiffness.norm() * response.norm();

  const auto order = static_cast<Eigen::Index>(basis.size());
  std::vector<Eigen::MatrixXcd> lifted(fluctuations_);
  for (std::size_t index = 0; index < fluctuations_; ++index)
    lifted[index] = secondQuantize(currents[index]);

  Eigen::MatrixXcd quartic = Eigen::MatrixXcd::Zero(order, order);
  Eigen::MatrixXcd induced =
      Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(rank),
                             static_cast<Eigen::Index>(rank));
  for (std::size_t a = 0; a < fluctuations_; ++a)
    for (std::size_t b = 0; b < fluctuations_; ++b) {
      const complexd weight = -0.5 * response(static_cast<Eigen::Index>(a),
                                              static_cast<Eigen::Index>(b));
      quartic += weight * (lifted[a] * lifted[b]);
      induced += weight * (currents[a] * currents[b]);
    }
  read.quartic = toFlat(quartic);

  const Eigen::MatrixXcd inducedLifted = secondQuantize(induced);
  read.inducedOneBody = toFlat(inducedLifted);
  read.normalOrderedQuartic = toFlat(quartic - inducedLifted);
  read.effectiveAction = toFlat(oneBody + quartic);

  const Eigen::MatrixXcd residual = stiffness * response - identity;
  read.certificate = Certificate::structureExact(
      CertificateDomain::Static, CertificateRegime::ComplexSymmetricPencil,
      std::max(residual.norm(), read.framePairingDefect),
      read.stiffnessConditioning, declaration_.tolerance);
  return read;
}

}  // namespace tessera::cobordism
