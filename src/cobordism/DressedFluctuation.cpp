// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/DressedFluctuation.h"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
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

// The exactly decoupled blocks of a square matrix: the connected components
// of the graph that joins two indices when either entry between them is not
// zero. Each block lists its indices in ascending order, and the blocks are
// in the order of their smallest index.
std::vector<std::vector<Eigen::Index>> decoupledBlocks(
    const Eigen::MatrixXcd &matrix) {
  const Eigen::Index n = matrix.rows();
  std::vector<Eigen::Index> root(static_cast<std::size_t>(n));
  std::iota(root.begin(), root.end(), Eigen::Index{0});
  const auto find = [&root](Eigen::Index index) {
    while (root[static_cast<std::size_t>(index)] != index)
      index = root[static_cast<std::size_t>(index)];
    return index;
  };
  const complexd zero{0.0, 0.0};
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = i + 1; j < n; ++j) {
      if (matrix(i, j) == zero && matrix(j, i) == zero) continue;
      const Eigen::Index a = find(i);
      const Eigen::Index b = find(j);
      // the smaller root stays, so a block is named by its smallest index
      if (a != b)
        root[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);
    }
  std::vector<std::vector<Eigen::Index>> blocks;
  std::vector<Eigen::Index> blockOf(static_cast<std::size_t>(n), -1);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::Index top = find(i);
    if (blockOf[static_cast<std::size_t>(top)] < 0) {
      blockOf[static_cast<std::size_t>(top)] =
          static_cast<Eigen::Index>(blocks.size());
      blocks.emplace_back();
    }
    blocks[static_cast<std::size_t>(blockOf[static_cast<std::size_t>(top)])]
        .push_back(i);
  }
  return blocks;
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
  // Each exactly decoupled block of the carrier is decomposed on its own: a
  // mode is then supported on one block, and blocks that are equal entry for
  // entry (the sheets of a sheeted support) have equal eigenvalues and equal
  // modes, so the degeneracy the copies give is exact. One decomposition of
  // the whole matrix separates the copies of an eigenvalue by its rounding,
  // and a state whose occupied modes end inside such a group then has
  // particle-hole energies that are rounding and a polarization that is the
  // reciprocal of rounding. The frame of right modes of each block is
  // inverted as computed: when it is singular at the declared tolerance the
  // carrier is defective to that tolerance, the instance says so
  // (`carrierDefective`), and the left frame is the inverse every nonzero
  // pivot gives. An exactly zero pivot leaves no inverse. The reciprocal
  // condition is estimated with every nonzero pivot counted, as the inverse
  // is formed.
  const auto carrierOrder = static_cast<Eigen::Index>(dimension_);
  Eigen::VectorXcd values = Eigen::VectorXcd::Zero(carrierOrder);
  Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Zero(carrierOrder, carrierOrder);
  Eigen::MatrixXcd inverse = Eigen::MatrixXcd::Zero(carrierOrder, carrierOrder);
  bool withoutInverse = false;
  Eigen::Index first = 0;
  for (const std::vector<Eigen::Index> &block : decoupledBlocks(carrier)) {
    const auto size = static_cast<Eigen::Index>(block.size());
    Eigen::MatrixXcd part(size, size);
    for (Eigen::Index row = 0; row < size; ++row)
      for (Eigen::Index column = 0; column < size; ++column)
        part(row, column) = carrier(block[static_cast<std::size_t>(row)],
                                    block[static_cast<std::size_t>(column)]);
    const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(part);
    if (solver.info() != Eigen::Success)
      throw std::invalid_argument(
          "DressedFluctuation: the carrier operator has no eigendecomposition, "
          "so its modes cannot be split into occupied and empty ones");
    Eigen::FullPivLU<Eigen::MatrixXcd> factor(solver.eigenvectors());
    factor.setThreshold(declaration_.tolerance);
    if (!factor.isInvertible()) defective_ = true;
    factor.setThreshold(0.0);
    const bool invertible = factor.isInvertible();
    const double reciprocal = invertible ? factor.rcond() : 0.0;
    modeFrameReciprocalCondition_ =
        std::min(modeFrameReciprocalCondition_,
                 std::isfinite(reciprocal) ? reciprocal : 0.0);
    Eigen::MatrixXcd partInverse = Eigen::MatrixXcd::Zero(size, size);
    if (invertible) partInverse = factor.inverse();
    if (!invertible || !partInverse.allFinite()) withoutInverse = true;
    for (Eigen::Index mode = 0; mode < size; ++mode) {
      values(first + mode) = solver.eigenvalues()(mode);
      for (Eigen::Index row = 0; row < size; ++row) {
        const Eigen::Index index = block[static_cast<std::size_t>(row)];
        vectors(index, first + mode) = solver.eigenvectors()(row, mode);
        inverse(first + mode, index) = partInverse(mode, row);
      }
    }
    first += size;
  }
  if (withoutInverse)
    throw std::invalid_argument(
        "DressedFluctuation: the frame of the carrier operator's right modes "
        "has no finite inverse, so the carrier has no matched pair of left and "
        "right mode frames");
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
  // The modes in the declared occupation order; equal keys keep the order of
  // the blocks, so the order does not turn on the eigensolver's rounding.
  Eigen::MatrixXcd right(carrierOrder, carrierOrder);
  Eigen::MatrixXcd left(carrierOrder, carrierOrder);
  eigenvalues_.resize(dimension_);
  for (std::size_t column = 0; column < dimension_; ++column) {
    right.col(static_cast<Eigen::Index>(column)) = vectors.col(order[column]);
    left.row(static_cast<Eigen::Index>(column)) = inverse.row(order[column]);
    eigenvalues_[column] = values(order[column]);
  }
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

bool DressedFluctuation::carrierDefective() const noexcept { return defective_; }

double DressedFluctuation::modeFrameReciprocalCondition() const noexcept {
  return modeFrameReciprocalCondition_;
}

double DressedFluctuation::smallestRelativeGap() const {
  double largest = 0.0;
  for (const complexd &value : eigenvalues_)
    largest = std::max(largest, std::abs(value));
  double smallest = std::numeric_limits<double>::infinity();
  for (std::size_t occupied = 0; occupied < declaration_.occupiedModes;
       ++occupied)
    for (std::size_t empty = declaration_.occupiedModes; empty < dimension_;
         ++empty)
      smallest = std::min(
          smallest, std::abs(eigenvalues_[empty] - eigenvalues_[occupied]));
  if (!std::isfinite(smallest) || largest == 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  return smallest / largest;
}

double DressedFluctuation::poleProximity(complexd frequency) const {
  const complexd broadening{0.0, -declaration_.continuumBroadening};
  double nearest = std::numeric_limits<double>::infinity();
  bool measured = false;
  for (std::size_t occupied = 0; occupied < declaration_.occupiedModes;
       ++occupied)
    for (std::size_t empty = declaration_.occupiedModes; empty < dimension_;
         ++empty) {
      const complexd gap =
          eigenvalues_[empty] - eigenvalues_[occupied] + broadening;
      const double size = std::norm(gap) + std::norm(frequency);
      if (size == 0.0) return 0.0;
      nearest = std::min(nearest,
                         std::abs(gap * gap - frequency * frequency) / size);
      measured = true;
    }
  return measured ? nearest : std::numeric_limits<double>::quiet_NaN();
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
      // The polarization has no value where a denominator is zero. Anywhere
      // else it has one, however near a pole the frequency is; how near is
      // `poleProximity`.
      if (denominator == complexd{0.0, 0.0})
        throw std::domain_error(
            "DressedFluctuation::paramagnetic: the requested frequency is the "
            "particle-hole energy of the pair (" +
            std::to_string(occupied) + ", " + std::to_string(empty) +
            "), where the polarization has a pole and no value");
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
  // The unit of the frequency axis: the largest particle-hole energy. The
  // shift is placed in it and the reciprocal of an infinite eigenvalue is
  // read against it. When every particle-hole energy is zero the pencil has
  // no energy of its own and its norm is the one scale present; a pencil
  // that is zero has no mode.
  double scale = 0.0;
  for (const complexd &energy : channelEnergy)
    scale = std::max(scale, std::abs(energy));
  if (scale == 0.0) scale = pencil.norm();
  if (scale == 0.0) return modes;
  const std::vector<complexd> candidateShifts{
      complexd{0.3719, 0.2341}, complexd{-0.6131, 0.4517},
      complexd{1.2837, -0.7193}, complexd{-1.9043, -1.1287}};
  // The shift is chosen on the measured quality of its solve rather than on a
  // pivot threshold: the pencil mixes the scale of the stiffness with that of
  // the particle-hole energies, and a rank decision taken against its largest
  // pivot refuses matrices the solve handles to rounding. Every shift that is
  // not an eigenvalue gives the same finite eigenvalues, so the one whose
  // solve leaves the smallest residual is the one used, and every mode read
  // from it carries its own measured residual against the dressed stiffness.
  Eigen::MatrixXcd resolvent;
  complexd shift{0.0, 0.0};
  bool shifted = false;
  double smallestResidual = std::numeric_limits<double>::infinity();
  for (const complexd &candidate : candidateShifts) {
    const complexd trial = candidate * scale;
    const Eigen::MatrixXcd shiftedPencil = pencil - trial * mass;
    const Eigen::PartialPivLU<Eigen::MatrixXcd> factor(shiftedPencil);
    const Eigen::MatrixXcd solved = factor.solve(mass);
    const double solveResidual =
        (shiftedPencil * solved - mass).norm() / mass.norm();
    if (!(solveResidual < smallestResidual)) continue;
    smallestResidual = solveResidual;
    resolvent = solved;
    shift = trial;
    shifted = true;
  }
  // No candidate shift has a solve with a finite residual: the resolvent has
  // no value at any of them, and there is nothing to read modes from.
  if (!shifted) return modes;

  const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(resolvent);
  if (solver.info() != Eigen::Success) return modes;

  for (Eigen::Index index = 0; index < solver.eigenvalues().size(); ++index) {
    const complexd reciprocal = solver.eigenvalues()(index);
    // A vanishing reciprocal is an infinite eigenvalue of the pencil, which the
    // deflated geometric block contributes and which is no frequency at all:
    // a reciprocal at or below the declared tolerance in the unit of the
    // largest particle-hole energy is read as zero.
    if (std::abs(reciprocal) * scale <= declaration_.tolerance) continue;
    const complexd frequency = shift + complexd{1.0, 0.0} / reciprocal;

    Eigen::VectorXcd geometric =
        solver.eigenvectors().col(index).head(fluctuations);
    const double whole = solver.eigenvectors().col(index).norm();
    if (whole == 0.0 || geometric.norm() <= declaration_.tolerance * whole)
      continue;
    geometric /= geometric.norm();

    // The residual is measured against the size of the two terms that cancel
    // at a pole, the bare-plus-diamagnetic stiffness and the polarization,
    // and not against the dressed stiffness itself, which is what vanishes
    // there. A candidate whose frequency is a particle-hole energy exactly
    // is reported with its residual unmeasured, since the polarization has
    // no value there; where both terms are zero the dressed stiffness is
    // zero and the null-vector equation holds exactly.
    double residual = Certificate::kUnmeasured;
    try {
      const Eigen::MatrixXcd polarization =
          toMatrix(paramagnetic(frequency), fluctuations_, fluctuations_);
      const Eigen::MatrixXcd dressed = bare - polarization;
      const double cancellationScale = bare.norm() + polarization.norm();
      residual = cancellationScale == 0.0
                     ? 0.0
                     : (dressed * geometric).norm() / cancellationScale;
    } catch (const std::domain_error &) {
    }

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
    std::optional<std::size_t> dimensionCap) const {
  // With no retained fluctuation the elimination integrates out nothing and
  // the effective action is the one-body term. With retained fluctuations
  // and no bare stiffness there is nothing to invert.
  if (fluctuations_ > 0 && declaration_.bareStiffness.empty())
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

  // The dimension is rank choose particles; a declared cap is compared with
  // it before the space is enumerated.
  if (dimensionCap) {
    long double dimension = 1.0L;
    for (std::size_t term = 1; term <= particles; ++term)
      dimension = dimension * static_cast<long double>(rank - particles + term) /
                  static_cast<long double>(term);
    if (dimension > static_cast<long double>(*dimensionCap) + 0.5L)
      throw std::length_error(
          "DressedFluctuation::effectiveAction: the " +
          std::to_string(particles) + "-particle space of a rank-" +
          std::to_string(rank) + " fiber has dimension above the declared cap "
          "of " + std::to_string(*dimensionCap));
  }
  const auto basis = subsets(rank, particles);

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

  const auto order = static_cast<Eigen::Index>(basis.size());
  if (fluctuations_ == 0) {
    // Nothing is eliminated: no stiffness is inverted, the quartic and its
    // two parts are zero, and the effective action is the one-body term. A
    // 0 x 0 stiffness has no condition number.
    const Eigen::MatrixXcd none = Eigen::MatrixXcd::Zero(order, order);
    read.quartic = toFlat(none);
    read.inducedOneBody = toFlat(none);
    read.normalOrderedQuartic = toFlat(none);
    read.effectiveAction = toFlat(oneBody);
    read.stiffnessConditioning = std::numeric_limits<double>::quiet_NaN();
    read.stiffnessReciprocalCondition =
        std::numeric_limits<double>::quiet_NaN();
    read.certificate = Certificate::structureExact(
        CertificateDomain::Static, CertificateRegime::ComplexSymmetricPencil,
        read.framePairingDefect, 1.0, declaration_.tolerance);
    return read;
  }

  const Eigen::MatrixXcd stiffness =
      toMatrix(declaration_.bareStiffness, fluctuations_, fluctuations_);
  read.stiffnessAsymmetry =
      stiffness.norm() == 0.0
          ? 0.0
          : (stiffness - stiffness.transpose()).norm() / stiffness.norm();
  // The bare stiffness is inverted as computed. When it is singular at the
  // declared tolerance (a pivot at or below that fraction of the largest)
  // the read says so (`stiffnessSingular`) and is made with the inverse every
  // nonzero pivot gives; an exactly zero pivot leaves no inverse. The
  // reciprocal condition is estimated with every nonzero pivot counted, as
  // the inverse is formed.
  Eigen::FullPivLU<Eigen::MatrixXcd> factor(stiffness);
  factor.setThreshold(declaration_.tolerance);
  read.stiffnessSingular = !factor.isInvertible();
  factor.setThreshold(0.0);
  const bool invertible = factor.isInvertible();
  const double reciprocal = invertible ? factor.rcond() : 0.0;
  read.stiffnessReciprocalCondition =
      std::isfinite(reciprocal) ? reciprocal : 0.0;
  const auto fluctuationOrder = static_cast<Eigen::Index>(fluctuations_);
  const Eigen::MatrixXcd identity =
      Eigen::MatrixXcd::Identity(fluctuationOrder, fluctuationOrder);
  // The whole interaction is factored through the R x R geometric response, and
  // R is the number of retained fluctuations rather than any many-body
  // dimension, so the response is formed once and no four-index tensor is ever
  // built.
  Eigen::MatrixXcd response =
      Eigen::MatrixXcd::Zero(fluctuationOrder, fluctuationOrder);
  if (invertible) response = factor.solve(identity);
  if (!invertible || !response.allFinite())
    throw std::invalid_argument(
        "DressedFluctuation::effectiveAction: the declared bare stiffness has "
        "no finite inverse, so the fluctuation cannot be eliminated at its "
        "saddle");
  read.stiffnessConditioning = stiffness.norm() * response.norm();

  std::vector<Eigen::MatrixXcd> lifted(fluctuations_);
  for (std::size_t index = 0; index < fluctuations_; ++index)
    lifted[index] = secondQuantize(currents[index]);

  // -1/2 sum_ab A^-1_ab O_a O_b = -1/2 sum_a O_a (sum_b A^-1_ab O_b): one
  // product per fluctuation, of the lifted current, which second
  // quantization leaves sparse, with the response-contracted one.
  Eigen::MatrixXcd quartic = Eigen::MatrixXcd::Zero(order, order);
  Eigen::MatrixXcd induced =
      Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(rank),
                             static_cast<Eigen::Index>(rank));
  for (std::size_t a = 0; a < fluctuations_; ++a) {
    Eigen::MatrixXcd contracted = Eigen::MatrixXcd::Zero(order, order);
    Eigen::MatrixXcd contractedCurrent =
        Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(rank),
                               static_cast<Eigen::Index>(rank));
    for (std::size_t b = 0; b < fluctuations_; ++b) {
      const complexd weight = response(static_cast<Eigen::Index>(a),
                                       static_cast<Eigen::Index>(b));
      contracted += weight * lifted[b];
      contractedCurrent += weight * currents[b];
    }
    const Eigen::SparseMatrix<complexd> sparse = lifted[a].sparseView();
    quartic.noalias() -= 0.5 * (sparse * contracted);
    induced.noalias() -= 0.5 * (currents[a] * contractedCurrent);
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
