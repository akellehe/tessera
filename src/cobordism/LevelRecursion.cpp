// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/LevelRecursion.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "chainhodge/RieszBand.h"
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
  if (declaration.bands.contourNodes < 3)
    throw std::invalid_argument(
        "LevelRecursion::overPencil: a contour needs at least three quadrature "
        "nodes; got " +
        std::to_string(declaration.bands.contourNodes));
  if (dimension >= declaration.denseCrossover)
    throw std::length_error(
        "LevelRecursion::overPencil: the microscopic level has " +
        std::to_string(dimension) +
        " coordinates, at or above the declared dense crossover of " +
        std::to_string(declaration.denseCrossover) +
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
  options.denseCrossover = declaration_.denseCrossover;
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
  if (width >= declaration_.denseCrossover)
    throw std::length_error(
        "LevelRecursion::advance: level " + std::to_string(level) + " has " +
        std::to_string(width) +
        " coordinates, at or above the declared dense crossover of " +
        std::to_string(declaration_.denseCrossover));

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

  // The certified fiber of every response vertex: the range of the Riesz
  // projector of its own block over its own contour.
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

    RecursionBandRead band;
    band.component = static_cast<int>(component);
    band.contourNodes = declaration_.bands.contourNodes;

    const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(block);
    std::vector<complexd> values;
    if (solver.info() == Eigen::Success)
      for (Eigen::Index index = 0; index < solver.eigenvalues().size(); ++index)
        values.push_back(solver.eigenvalues()(index));

    std::size_t rank = 0;
    if (declaration_.bands.selection == RecursionBandSelection::LowestModes) {
      std::vector<complexd> ordered = values;
      const bool byRealPart =
          declaration_.bands.order == OccupationOrder::AscendingRealPart;
      std::stable_sort(ordered.begin(), ordered.end(),
                       [byRealPart](const complexd &a, const complexd &b) {
                         if (byRealPart) return ascendingSpectrum(a, b);
                         return std::abs(a) < std::abs(b);
                       });
      rank = std::min(declaration_.bands.bandRank, ordered.size());
      complexd centre{0.0, 0.0};
      for (std::size_t index = 0; index < rank; ++index)
        centre += ordered[index];
      if (rank > 0) centre /= static_cast<double>(rank);
      double inside = 0.0;
      for (std::size_t index = 0; index < rank; ++index)
        inside = std::max(inside, std::abs(ordered[index] - centre));
      double outside = std::numeric_limits<double>::infinity();
      for (std::size_t index = rank; index < ordered.size(); ++index)
        outside = std::min(outside, std::abs(ordered[index] - centre));
      band.contourCentre = centre;
      band.contourRadius = std::isfinite(outside)
                               ? 0.5 * (inside + outside)
                               : inside + std::max(1.0, inside);
      band.eigenvalues.assign(ordered.begin(),
                              ordered.begin() +
                                  static_cast<std::ptrdiff_t>(rank));
    } else {
      band.contourCentre = declaration_.bands.contourCentres[component];
      band.contourRadius = declaration_.bands.contourRadii[component];
      for (const complexd &value : values)
        if (std::abs(value - band.contourCentre) < band.contourRadius)
          band.eigenvalues.push_back(value);
      std::stable_sort(band.eigenvalues.begin(), band.eigenvalues.end(),
                       ascendingSpectrum);
      rank = band.eigenvalues.size();
    }

    band.isolationGap = values.empty()
                            ? std::numeric_limits<double>::quiet_NaN()
                            : std::numeric_limits<double>::infinity();
    for (const complexd &value : values)
      band.isolationGap = std::min(
          band.isolationGap,
          std::abs(std::abs(value - band.contourCentre) - band.contourRadius));

    const chainhodge::Contour contour = chainhodge::Contour::circle(
        band.contourCentre, band.contourRadius, band.contourNodes);
    Eigen::MatrixXcd projector = Eigen::MatrixXcd::Zero(blockOrder, blockOrder);
    const Eigen::MatrixXcd identity =
        Eigen::MatrixXcd::Identity(blockOrder, blockOrder);
    bool resolventFailed = false;
    for (std::size_t node = 0; node < contour.nodeCount(); ++node) {
      const Eigen::FullPivLU<Eigen::MatrixXcd> factor(
          contour.nodes[node] * identity - block);
      if (!factor.isInvertible()) {
        resolventFailed = true;
        break;
      }
      projector += contour.weights[node] * factor.solve(identity);
    }

    band.rank = rank;
    if (resolventFailed || rank == 0) {
      band.projectorIdempotency = std::numeric_limits<double>::infinity();
      band.pairingDefect = std::numeric_limits<double>::infinity();
      band.accepted = false;
      band.certificate = Certificate::certifiedNumerical(
          CertificateDomain::BandWindow, CertificateRegime::NonNormal,
          std::numeric_limits<double>::infinity(),
          Certificate::kUnmeasured, declaration_.tolerance);
      everyFiberAccepted = false;
      read.bands.push_back(std::move(band));
      continue;
    }

    band.projectorIdempotency =
        (projector * projector - projector).norm() /
        std::max(1.0, projector.norm());

    const Eigen::JacobiSVD<Eigen::MatrixXcd> decomposition(
        projector, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto columns = std::min<Eigen::Index>(
        static_cast<Eigen::Index>(rank), decomposition.matrixU().cols());
    const Eigen::MatrixXcd right = decomposition.matrixU().leftCols(columns);
    // The left frame is the one that pairs with the right frame by the
    // transpose to the identity and reproduces the projector; the
    // least-squares solve that computes a basis of the range is a means, and
    // the bilinear identity it is held to is the statement.
    const Eigen::MatrixXcd left =
        (right.adjoint() * right).ldlt().solve(right.adjoint() * projector);
    band.pairingDefect =
        (left * right - Eigen::MatrixXcd::Identity(columns, columns)).norm();

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
    band.rank = static_cast<std::size_t>(columns);

    const double residual =
        std::max(band.projectorIdempotency, band.pairingDefect);
    band.accepted = residual <= declaration_.tolerance &&
                    band.isolationGap > 0.0 &&
                    static_cast<std::size_t>(columns) == rank;
    band.certificate = Certificate::certifiedNumerical(
        CertificateDomain::BandWindow, CertificateRegime::NonNormal, residual,
        Certificate::kUnmeasured, declaration_.tolerance);
    worstFiberResidual = std::max(worstFiberResidual, residual);
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
