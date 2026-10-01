// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/LevelRecursion.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "chainhodge/RieszProjector.h"
#include "cobordism/HodgeLaplacian.h"
#include "spacetime/Spacetime.h"

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

bool ascendingSpectrum(const complexd &a, const complexd &b) {
  if (a.real() != b.real()) return a.real() < b.real();
  return a.imag() < b.imag();
}

std::string describe(const complexd &value) {
  return "(" + std::to_string(value.real()) + ", " +
         std::to_string(value.imag()) + ")";
}

// The keys of the declared occupation order, in the order they are compared:
// real then imaginary part, or modulus then real then imaginary part.
double orderKey(const complexd &value, OccupationOrder order,
                std::size_t level) {
  if (order == OccupationOrder::AscendingRealPart)
    return level == 0 ? value.real() : value.imag();
  if (level == 0) return std::abs(value);
  return level == 1 ? value.real() : value.imag();
}

// The declared order of a spectrum with every key compared at the resolution:
// the indices are sorted by the key of this level, split into runs whose
// consecutive keys differ by at most the resolution, and each run is ordered
// by the next key. A run at the last key is one eigenvalue with multiplicity
// and keeps the order the exact keys give it.
void orderAtResolution(const Eigen::VectorXcd &values, OccupationOrder order,
                       double resolution, std::vector<int> indices,
                       std::size_t level, std::vector<int> &ordered) {
  const std::size_t keyCount =
      order == OccupationOrder::AscendingRealPart ? 2 : 3;
  if (level == keyCount || indices.size() <= 1) {
    ordered.insert(ordered.end(), indices.begin(), indices.end());
    return;
  }
  std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
    return orderKey(values(a), order, level) < orderKey(values(b), order, level);
  });
  std::size_t start = 0;
  for (std::size_t index = 1; index <= indices.size(); ++index) {
    const bool runEnds =
        index == indices.size() ||
        orderKey(values(indices[index]), order, level) -
                orderKey(values(indices[index - 1]), order, level) >
            resolution;
    if (!runEnds) continue;
    orderAtResolution(values, order, resolution,
                      std::vector<int>(indices.begin() +
                                           static_cast<std::ptrdiff_t>(start),
                                       indices.begin() +
                                           static_cast<std::ptrdiff_t>(index)),
                      level + 1, ordered);
    start = index;
  }
}

}  // namespace

LevelRecursion LevelRecursion::overPencil(
    const std::vector<complexd> &pencil, const std::vector<complexd> &metric,
    int dimension, LevelRecursionDeclaration declaration) {
  if (dimension <= 0)
    throw std::invalid_argument(
        "LevelRecursion::overPencil: the dimension must be positive; got " +
        std::to_string(dimension));
  const std::size_t square =
      static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension);
  if (pencil.size() != square)
    throw std::invalid_argument(
        "LevelRecursion::overPencil: the pencil must be a square matrix over "
        "the " +
        std::to_string(dimension) + " coordinates; got " +
        std::to_string(pencil.size()) + " entries");
  if (!metric.empty() && metric.size() != square)
    throw std::invalid_argument(
        "LevelRecursion::overPencil: the metric must be a square matrix over "
        "the " +
        std::to_string(dimension) + " coordinates, or empty for the identity; "
        "got " +
        std::to_string(metric.size()) + " entries");
  if (declaration.resolutions.empty())
    throw std::invalid_argument(
        "LevelRecursion::overPencil: the partition sweep must name at least "
        "one modularity resolution");
  if (declaration.bands.bandRank == 0)
    throw std::invalid_argument(
        "LevelRecursion::overPencil: a band of rank zero encloses no "
        "eigenvalue and is no fiber");
  if (declaration.denseCrossover && dimension >= *declaration.denseCrossover)
    throw std::length_error(
        "LevelRecursion::overPencil: the microscopic level has " +
        std::to_string(dimension) +
        " coordinates, at or above the declared dense crossover of " +
        std::to_string(*declaration.denseCrossover) +
        ", and the response pencil is evaluated densely");
  LevelRecursion recursion;
  recursion.pencil_ = pencil;
  recursion.metric_ = metric;
  recursion.dimension_ = dimension;
  recursion.declaration_ = std::move(declaration);
  return recursion;
}

LevelRecursion LevelRecursion::overSpacetime(
    const std::shared_ptr<Spacetime> &spacetime, int degree,
    HodgeLaplacian::MetricSource metricSource,
    LevelRecursionDeclaration declaration) {
  if (!spacetime)
    throw std::invalid_argument(
        "LevelRecursion::overSpacetime: the complex is null");
  const HodgeLaplacian laplacian(spacetime,
                                 HodgeLaplacian::defaultWeightConvention(),
                                 metricSource);
  const std::vector<complexd> operatorMatrix = laplacian.laplacian(degree);
  if (operatorMatrix.empty())
    throw std::invalid_argument(
        "LevelRecursion::overSpacetime: the complex carries no cell of degree " +
        std::to_string(degree));
  const auto dimension = static_cast<int>(
      std::llround(std::sqrt(static_cast<double>(operatorMatrix.size()))));
  return overPencil(operatorMatrix, {}, dimension, std::move(declaration));
}

RecursiveQuotient::Options LevelRecursion::quotientOptions() const {
  RecursiveQuotient::Options options;
  options.tolerance = declaration_.tolerance;
  options.rankTolerance = declaration_.rankTolerance;
  // With no declared crossover every interior solve takes the dense path.
  options.denseCrossover =
      declaration_.denseCrossover.value_or(std::numeric_limits<int>::max());
  options.embeddingPolicy = FiberEmbeddingPolicy::CarryGramExactly;
  return options;
}

int LevelRecursion::responseDimension(std::size_t level) const {
  if (level > levels_.size())
    throw std::out_of_range(
        "LevelRecursion::responseDimension: level " + std::to_string(level) +
        " has not been built; " + std::to_string(levels_.size()) +
        " turns of the box have been taken");
  if (level == 0) return dimension_;
  return levels_[level - 1].responseDimension;
}

std::vector<complexd> LevelRecursion::responsePencil(
    std::size_t level, complexd lambda) const {
  if (level > levels_.size())
    throw std::out_of_range(
        "LevelRecursion::responsePencil: level " + std::to_string(level) +
        " has not been built; " + std::to_string(levels_.size()) +
        " turns of the box have been taken");
  // R_0(lambda) = A - lambda M. The spectral parameter enters here and nowhere
  // else: every level above is a plain supported block elimination of the level
  // below, which already carries it.
  std::vector<complexd> response = pencil_;
  const auto width = static_cast<std::size_t>(dimension_);
  for (std::size_t row = 0; row < width; ++row)
    for (std::size_t column = 0; column < width; ++column) {
      const complexd metric =
          metric_.empty() ? (row == column ? complexd{1.0, 0.0}
                                           : complexd{0.0, 0.0})
                          : metric_[row * width + column];
      response[row * width + column] -= lambda * metric;
    }
  int currentDimension = dimension_;
  for (std::size_t step = 0; step < level; ++step) {
    const RecursiveQuotient quotient = RecursiveQuotient::overMatrix(
        response, currentDimension, {}, levels_[step].partition,
        quotientOptions());
    const RecursiveQuotient::StaticReductionRead &reduction =
        quotient.staticReduction();
    response = reduction.effectiveOperator;
    currentDimension = static_cast<int>(reduction.coordinates.size());
  }
  return response;
}

RecursiveQuotient LevelRecursion::quotientAt(std::size_t level,
                                             complexd lambda) const {
  return RecursiveQuotient::overMatrix(responsePencil(level, lambda),
                                       responseDimension(level), {},
                                       levels_[level].partition,
                                       quotientOptions());
}

complexd LevelRecursion::responseDeterminant(std::size_t level,
                                             complexd lambda) const {
  const std::vector<complexd> response = responsePencil(level, lambda);
  const auto width = static_cast<std::size_t>(responseDimension(level));
  if (width == 0) return complexd{1.0, 0.0};
  return toMatrix(response, width, width).determinant();
}

complexd LevelRecursion::interiorDeterminant(std::size_t level,
                                             complexd lambda) const {
  if (level >= levels_.size())
    throw std::out_of_range(
        "LevelRecursion::interiorDeterminant: level " + std::to_string(level) +
        " has not been reduced; " + std::to_string(levels_.size()) +
        " turns of the box have been taken");
  const std::vector<complexd> response = responsePencil(level, lambda);
  const auto width = static_cast<std::size_t>(responseDimension(level));
  const Eigen::MatrixXcd matrix = toMatrix(response, width, width);
  const RecursiveQuotient quotient = RecursiveQuotient::overMatrix(
      response, static_cast<int>(width), {}, levels_[level].partition,
      quotientOptions());
  complexd determinant{1.0, 0.0};
  for (int component = 0; component < quotient.componentCount(); ++component) {
    const std::vector<int> &interior = quotient.interiorIndices(component);
    if (interior.empty()) continue;
    Eigen::MatrixXcd block(static_cast<Eigen::Index>(interior.size()),
                           static_cast<Eigen::Index>(interior.size()));
    for (std::size_t row = 0; row < interior.size(); ++row)
      for (std::size_t column = 0; column < interior.size(); ++column)
        block(static_cast<Eigen::Index>(row),
              static_cast<Eigen::Index>(column)) =
            matrix(interior[row], interior[column]);
    determinant *= block.determinant();
  }
  return determinant;
}

double LevelRecursion::determinantFactorizationResidual(
    std::size_t level, complexd lambda) const {
  const complexd whole = responseDeterminant(level, lambda);
  const complexd factored =
      interiorDeterminant(level, lambda) * responseDeterminant(level + 1, lambda);
  const double scale = std::max(std::abs(whole), std::abs(factored));
  if (scale == 0.0) return 0.0;
  return std::abs(whole - factored) / scale;
}

RecursionBandRead LevelRecursion::readBand(
    const std::vector<complexd> &block, int order,
    const RecursionBandDeclaration &bands, std::size_t component,
    double tolerance) {
  if (order <= 0)
    throw std::invalid_argument(
        "LevelRecursion::readBand: the block of component " +
        std::to_string(component) + " must have at least one coordinate; got " +
        std::to_string(order));
  const auto n = static_cast<std::size_t>(order);
  if (block.size() != n * n)
    throw std::invalid_argument(
        "LevelRecursion::readBand: the block of component " +
        std::to_string(component) + " must be a square matrix over its " +
        std::to_string(order) + " coordinates; got " +
        std::to_string(block.size()) + " entries");
  if (bands.bandRank == 0)
    throw std::invalid_argument(
        "LevelRecursion::readBand: a band of rank zero encloses no eigenvalue "
        "and is no fiber");
  const bool declared =
      bands.selection == RecursionBandSelection::DeclaredContours;
  if (declared && (component >= bands.contourCentres.size() ||
                   component >= bands.contourRadii.size()))
    throw std::invalid_argument(
        "LevelRecursion::readBand: the declared contours name " +
        std::to_string(bands.contourCentres.size()) + " centres and " +
        std::to_string(bands.contourRadii.size()) +
        " radii, none of them for component " + std::to_string(component));

  const Eigen::MatrixXcd matrix = toMatrix(block, n, n);
  const Eigen::ComplexSchur<Eigen::MatrixXcd> schur(matrix, true);
  if (schur.info() != Eigen::Success)
    throw std::runtime_error(
        "LevelRecursion::readBand: the complex Schur decomposition of the "
        "block of component " +
        std::to_string(component) + " did not converge");
  const Eigen::VectorXcd values = schur.matrixT().diagonal();
  // Two eigenvalues, or two order keys, at most this far apart are equal at
  // the declared tolerance: the rounding of the decomposition is relative to
  // the block's norm, not to the eigenvalues' own magnitudes.
  const double scale = matrix.norm();
  const double resolution = tolerance * scale;

  RecursionBandRead band;
  band.component = static_cast<int>(component);
  band.contourGap = std::numeric_limits<double>::infinity();
  std::vector<bool> selected(n, false);
  // Whether the membership of every eigenvalue in a declared circle is
  // decided at the declared tolerance; a derived circle decides none.
  bool membershipDecided = true;
  if (!declared) {
    std::vector<int> initial(n);
    std::iota(initial.begin(), initial.end(), 0);
    std::vector<int> ordered;
    orderAtResolution(values, bands.order, resolution, initial, 0, ordered);
    const std::size_t rank = std::min(bands.bandRank, n);
    complexd centre{0.0, 0.0};
    for (std::size_t index = 0; index < rank; ++index) {
      selected[static_cast<std::size_t>(ordered[index])] = true;
      band.eigenvalues.push_back(values(ordered[index]));
      centre += values(ordered[index]);
    }
    centre /= static_cast<double>(rank);
    double inside = 0.0;
    for (std::size_t index = 0; index < rank; ++index)
      inside = std::max(inside, std::abs(values(ordered[index]) - centre));
    double outside = std::numeric_limits<double>::infinity();
    for (std::size_t index = rank; index < n; ++index)
      outside = std::min(outside, std::abs(values(ordered[index]) - centre));
    band.contourCentre = centre;
    band.enclosesEverything = rank == n;
    band.contourRadius = band.enclosesEverything
                             ? std::numeric_limits<double>::infinity()
                             : 0.5 * (inside + outside);
  } else {
    const complexd centre = bands.contourCentres[component];
    const double radius = bands.contourRadii[component];
    // Membership is the strict comparison of an eigenvalue's distance from
    // the centre with the radius. How close the circle passes to an
    // eigenvalue is measured beside it: an eigenvalue on the circle at the
    // declared tolerance leaves its membership undecided at that tolerance,
    // which the read reports and does not act on.
    for (std::size_t index = 0; index < n; ++index) {
      const double distance = std::abs(values(index) - centre);
      const double gap = std::abs(distance - radius);
      band.contourGap = std::min(band.contourGap, gap);
      if (gap <= tolerance * std::max(distance, radius))
        membershipDecided = false;
      selected[index] = distance < radius;
    }
    for (std::size_t index = 0; index < n; ++index)
      if (selected[index]) band.eigenvalues.push_back(values(index));
    std::stable_sort(band.eigenvalues.begin(), band.eigenvalues.end(),
                     ascendingSpectrum);
    band.contourCentre = centre;
    band.contourRadius = radius;
    band.enclosesEverything = band.eigenvalues.size() == n;
  }
  band.rank = band.eigenvalues.size();

  // The isolation of the band: the closest a selected eigenvalue comes to an
  // excluded one. The band is read whatever its isolation, and the gap is
  // reported; a selection that separates two eigenvalues equal at the
  // declared tolerance is not accepted.
  band.isolationGap = std::numeric_limits<double>::infinity();
  complexd closestSelected{0.0, 0.0};
  for (std::size_t inside = 0; inside < n; ++inside) {
    if (!selected[inside]) continue;
    for (std::size_t outside = 0; outside < n; ++outside) {
      if (selected[outside]) continue;
      const double distance = std::abs(values(inside) - values(outside));
      if (distance < band.isolationGap) {
        band.isolationGap = distance;
        closestSelected = values(inside);
      }
    }
  }
  // The eigenvalues of the Sylvester operator Y -> T_11 Y - Y T_22 are the
  // differences of a selected and an excluded eigenvalue. When a selected and
  // an excluded eigenvalue are equal exactly, one of them is zero: the
  // equation is singular, and the projector onto a part of that eigenvalue's
  // invariant subspace has no value.
  if (band.isolationGap == 0.0)
    throw std::domain_error(
        "LevelRecursion::readBand: the selection of component " +
        std::to_string(component) + " takes the eigenvalue " +
        describe(closestSelected) +
        " of its block into the band and leaves an eigenvalue exactly equal "
        "to it out, so the Sylvester equation of the band's projector is "
        "singular and the projector onto a part of that eigenvalue's "
        "invariant subspace has no value");

  // The exact projector and its frames. A selection that encloses every
  // eigenvalue has the whole coordinate space as its invariant subspace: the
  // projector is the identity and the canonical basis is its frame, with
  // nothing to reorder and no Sylvester equation to solve. A selection that
  // encloses no eigenvalue has the zero subspace as its invariant subspace:
  // the projector is zero and the frames have no column. Every other
  // selection is read from the reordered Schur form.
  const auto blockOrder = static_cast<Eigen::Index>(n);
  const auto columns = static_cast<Eigen::Index>(band.rank);
  Eigen::MatrixXcd projector;
  Eigen::MatrixXcd right;
  Eigen::MatrixXcd left;
  if (band.rank == 0) {
    projector = Eigen::MatrixXcd::Zero(blockOrder, blockOrder);
    right = Eigen::MatrixXcd(blockOrder, 0);
    left = Eigen::MatrixXcd(0, blockOrder);
  } else if (band.enclosesEverything) {
    projector = Eigen::MatrixXcd::Identity(columns, columns);
    right = projector;
    left = projector;
  } else {
    const chainhodge::RieszProjectorRead riesz =
        chainhodge::rieszProjector(schur, selected);
    projector = riesz.projector;
    right = riesz.right;
    left = riesz.left.transpose();
  }
  if (band.rank == 0) {
    // The zero projector is idempotent exactly, and its empty frames pair to
    // the 0 x 0 identity and span the zero subspace, which is invariant.
    band.projectorIdempotency = 0.0;
    band.pairingDefect = 0.0;
    band.invariantSubspaceResidual = 0.0;
  } else {
    band.projectorIdempotency =
        (projector * projector - projector).norm() / projector.norm();
    band.pairingDefect =
        (left * right - Eigen::MatrixXcd::Identity(columns, columns)).norm();
    const Eigen::MatrixXcd reduced = left * matrix * right;
    band.invariantSubspaceResidual =
        scale > 0.0 ? (matrix * right - right * reduced).norm() / scale : 0.0;
  }
  band.frame = toFlat(right);
  band.leftFrame = toFlat(left);

  // What the read reports of itself. The preconditions of a certified fiber
  // are that the band has an eigenvalue, that its selection separates no two
  // eigenvalues equal at the declared tolerance, and that the membership of
  // every eigenvalue in a declared circle is decided at that tolerance; the
  // residuals are those of the projector that was formed. A residual that is
  // not a finite number is a residual that does not hold.
  const bool measured = std::isfinite(band.projectorIdempotency) &&
                        std::isfinite(band.pairingDefect) &&
                        std::isfinite(band.invariantSubspaceResidual);
  const bool preconditions = band.rank > 0 &&
                             band.isolationGap > resolution &&
                             membershipDecided && measured;
  const double residual =
      preconditions ? std::max({band.projectorIdempotency, band.pairingDefect,
                                band.invariantSubspaceResidual})
                    : std::numeric_limits<double>::infinity();
  band.accepted = residual <= tolerance;
  band.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, CertificateRegime::NonNormal, residual,
      Certificate::kUnmeasured, tolerance);
  return band;
}

void LevelRecursion::advanceTo(std::size_t levels) {
  while (levels_.size() < levels) advance();
}

void LevelRecursion::advance() {
  const std::size_t level = levels_.size();
  const int width = responseDimension(level);
  if (width <= 0)
    throw std::length_error(
        "LevelRecursion::advance: level " + std::to_string(level) +
        " has no coordinate left to partition, so the recursion has reduced "
        "the complex to nothing");
  if (declaration_.denseCrossover && width >= *declaration_.denseCrossover)
    throw std::length_error(
        "LevelRecursion::advance: level " + std::to_string(level) + " has " +
        std::to_string(width) +
        " coordinates, at or above the declared dense crossover of " +
        std::to_string(*declaration_.denseCrossover));

  RecursionLevelRead read;
  read.level = level;
  read.dimension = width;

  const std::vector<complexd> response =
      responsePencil(level, declaration_.referenceLambda);
  const Eigen::MatrixXcd operatorMatrix =
      toMatrix(response, static_cast<std::size_t>(width),
               static_cast<std::size_t>(width));

  const RecursiveQuotient::ResolvedPartitionRead resolved =
      RecursiveQuotient::persistentPartitionOverResolutions(
          response, width, declaration_.resolutions,
          declaration_.modularityRestarts, declaration_.modularitySeed,
          declaration_.persistenceOverlap);
  read.partition = resolved.components;
  read.resolutions = resolved.resolutions;
  read.selectedResolution = resolved.selectedResolution;
  read.componentPersistence = resolved.componentPersistence;
  read.worstPersistenceOverlap = resolved.worstOverlap;

  if (declaration_.bands.selection == RecursionBandSelection::DeclaredContours &&
      (declaration_.bands.contourCentres.size() != read.partition.size() ||
       declaration_.bands.contourRadii.size() != read.partition.size()))
    throw std::invalid_argument(
        "LevelRecursion::advance: the declared contours name " +
        std::to_string(declaration_.bands.contourCentres.size()) +
        " centres and " +
        std::to_string(declaration_.bands.contourRadii.size()) +
        " radii, but the partition of this level has " +
        std::to_string(read.partition.size()) + " components");

  // The certified fiber of every response vertex: the range of the exact
  // Riesz projector of its own block onto the band the declaration selects,
  // read over the block's coordinates and embedded into the level's.
  double worstFiberResidual = 0.0;
  bool everyFiberAccepted = true;
  for (std::size_t component = 0; component < read.partition.size();
       ++component) {
    const std::vector<int> &coordinates = read.partition[component];
    const auto blockOrder = static_cast<Eigen::Index>(coordinates.size());
    Eigen::MatrixXcd block(blockOrder, blockOrder);
    for (Eigen::Index row = 0; row < blockOrder; ++row)
      for (Eigen::Index column = 0; column < blockOrder; ++column)
        block(row, column) =
            operatorMatrix(coordinates[static_cast<std::size_t>(row)],
                           coordinates[static_cast<std::size_t>(column)]);

    RecursionBandRead band =
        readBand(toFlat(block), static_cast<int>(blockOrder),
                 declaration_.bands, component, declaration_.tolerance);
    const auto columns = static_cast<Eigen::Index>(band.rank);
    const Eigen::MatrixXcd right =
        toMatrix(band.frame, static_cast<std::size_t>(blockOrder), band.rank);
    const Eigen::MatrixXcd left = toMatrix(
        band.leftFrame, band.rank, static_cast<std::size_t>(blockOrder));
    Eigen::MatrixXcd embedded =
        Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(width), columns);
    Eigen::MatrixXcd dualEmbedded =
        Eigen::MatrixXcd::Zero(columns, static_cast<Eigen::Index>(width));
    for (Eigen::Index row = 0; row < blockOrder; ++row) {
      embedded.row(coordinates[static_cast<std::size_t>(row)]) =
          right.row(row);
      dualEmbedded.col(coordinates[static_cast<std::size_t>(row)]) =
          left.col(row);
    }
    band.frame = toFlat(embedded);
    band.leftFrame = toFlat(dualEmbedded);

    worstFiberResidual =
        std::max(worstFiberResidual, band.certificate.residual());
    if (!band.accepted) everyFiberAccepted = false;
    read.bands.push_back(std::move(band));
  }

  // The labeled sum: the fibers side by side, with the bilinear overlap
  // carried exactly and no claim that the internal sum is direct.
  std::size_t modes = 0;
  for (const RecursionBandRead &band : read.bands) modes += band.rank;
  read.modes = modes;
  Eigen::MatrixXcd embedding = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(width), static_cast<Eigen::Index>(modes));
  Eigen::MatrixXcd dualEmbedding = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(modes), static_cast<Eigen::Index>(width));
  std::vector<std::size_t> offsets;
  std::size_t offset = 0;
  for (const RecursionBandRead &band : read.bands) {
    offsets.push_back(offset);
    if (band.rank == 0) continue;
    embedding.middleCols(static_cast<Eigen::Index>(offset),
                         static_cast<Eigen::Index>(band.rank)) =
        toMatrix(band.frame, static_cast<std::size_t>(width), band.rank);
    dualEmbedding.middleRows(static_cast<Eigen::Index>(offset),
                             static_cast<Eigen::Index>(band.rank)) =
        toMatrix(band.leftFrame, band.rank, static_cast<std::size_t>(width));
    offset += band.rank;
  }
  read.embedding = toFlat(embedding);
  read.dualEmbedding = toFlat(dualEmbedding);
  const Eigen::MatrixXcd gram = dualEmbedding * embedding;
  read.gram = toFlat(gram);
  read.gramDefect =
      modes == 0 ? 0.0
                 : (gram - Eigen::MatrixXcd::Identity(
                               static_cast<Eigen::Index>(modes),
                               static_cast<Eigen::Index>(modes)))
                       .norm();
  const Eigen::MatrixXcd fiberOperator =
      dualEmbedding * operatorMatrix * embedding;
  read.fiberOperator = toFlat(fiberOperator);
  if (modes > 0) {
    const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> fiberSolver(fiberOperator);
    if (fiberSolver.info() == Eigen::Success) {
      for (Eigen::Index index = 0; index < fiberSolver.eigenvalues().size();
           ++index)
        read.fiberSpectrum.push_back(fiberSolver.eigenvalues()(index));
      std::stable_sort(read.fiberSpectrum.begin(), read.fiberSpectrum.end(),
                       ascendingSpectrum);
    }
  }
  for (std::size_t to = 0; to < read.bands.size(); ++to)
    for (std::size_t from = 0; from < read.bands.size(); ++from) {
      if (read.bands[to].rank == 0 || read.bands[from].rank == 0) continue;
      const Eigen::MatrixXcd block = fiberOperator.block(
          static_cast<Eigen::Index>(offsets[to]),
          static_cast<Eigen::Index>(offsets[from]),
          static_cast<Eigen::Index>(read.bands[to].rank),
          static_cast<Eigen::Index>(read.bands[from].rank));
      if (to != from && block.norm() == 0.0) continue;
      LevelTransport transport;
      transport.to = static_cast<int>(to);
      transport.from = static_cast<int>(from);
      transport.block = toFlat(block);
      read.transports.push_back(std::move(transport));
    }
  read.fockStageDimension =
      modes >= 1024 ? std::numeric_limits<double>::infinity()
                    : std::pow(2.0, static_cast<double>(modes));

  // The response step itself.
  const RecursiveQuotient quotient = RecursiveQuotient::overMatrix(
      response, width, {}, read.partition, quotientOptions());
  const RecursiveQuotient::StaticReductionRead &reduction =
      quotient.staticReduction();
  read.responseDimension = static_cast<int>(reduction.coordinates.size());
  read.reductionCertificate = reduction.certificate;

  bool retainedInteriorMode = false;
  for (const RecursiveQuotient::RetainedCoordinate &coordinate :
       reduction.coordinates)
    if (coordinate.kind == RetainedCoordinateKind::Harmonic ||
        coordinate.kind == RetainedCoordinateKind::Resonant)
      retainedInteriorMode = true;

  levels_.push_back(std::move(read));
  RecursionLevelRead &stored = levels_.back();
  if (retainedInteriorMode) {
    stored.determinantResidual = Certificate::kUnmeasured;
  } else {
    stored.determinantResidual = determinantFactorizationResidual(
        level, declaration_.referenceLambda);
  }
  const double residual =
      std::isfinite(stored.determinantResidual)
          ? std::max(worstFiberResidual, stored.determinantResidual)
          : worstFiberResidual;
  stored.certificate = Certificate::structureExact(
      CertificateDomain::BandWindow, quotient.regime(),
      everyFiberAccepted ? residual : std::numeric_limits<double>::infinity(),
      Certificate::kUnmeasured, declaration_.tolerance);
}

const RecursionLevelRead &LevelRecursion::level(std::size_t index) const {
  if (index >= levels_.size())
    throw std::out_of_range("LevelRecursion::level: " + std::to_string(index) +
                            " names no completed level; " +
                            std::to_string(levels_.size()) +
                            " turns of the box have been taken");
  return levels_[index];
}

std::vector<int> LevelRecursion::componentOfCoordinate(
    std::size_t index) const {
  const RecursionLevelRead &read = level(index);
  std::vector<int> owner(static_cast<std::size_t>(read.dimension), -1);
  for (std::size_t component = 0; component < read.partition.size();
       ++component)
    for (const int coordinate : read.partition[component])
      if (coordinate >= 0 && coordinate < read.dimension)
        owner[static_cast<std::size_t>(coordinate)] =
            static_cast<int>(component);
  return owner;
}

void LevelRecursion::recordVacuumEmbeddedModes(std::size_t index,
                                               std::size_t modes) {
  if (index >= levels_.size())
    throw std::out_of_range(
        "LevelRecursion::recordVacuumEmbeddedModes: " + std::to_string(index) +
        " names no completed level; " + std::to_string(levels_.size()) +
        " turns of the box have been taken");
  RecursionLevelRead &read = levels_[index];
  read.vacuumEmbeddedModes = modes;
  const std::size_t grown = read.modes + modes;
  read.fockStageDimension = grown >= 1024
                                ? std::numeric_limits<double>::infinity()
                                : std::pow(2.0, static_cast<double>(grown));
}

}  // namespace tessera::cobordism
