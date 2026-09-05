// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/MultiCobordism.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>

#include "Logger.h"
#include "chainhodge/BandDerivative.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/LevenbergMarquardt.h"
#include "cobordism/SurgicalCone.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeKey.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "quantum/ChoiJamiolkowski.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"
#include "spacetime/topologies/SolidSimplex.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;
using complexd = std::complex<double>;

namespace {

std::pair<std::uint64_t, std::uint64_t> edgeKey(
    const ::tessera::mesh::Edge *edge) {
  const auto sourceVertexId = edge->getSource()->getId();
  const auto targetVertexId = edge->getTarget()->getId();
  return {std::min(sourceVertexId, targetVertexId),
          std::max(sourceVertexId, targetVertexId)};
}

// z=l^2 is the optimization coordinate, while Edge stores l. Both square roots
// represent the same z; retain the root nearest the resident l so an accepted
// update cannot introduce an unrelated l -> -l branch jump.
complexd continuousSquareRoot(complexd z, complexd referenceLength) {
  const complexd principal = std::sqrt(z);
  return std::abs(principal - referenceLength) <=
                 std::abs(-principal - referenceLength)
             ? principal
             : -principal;
}

// The boundary facets of `spacetime`, as sorted vertex-id tuples, for membership tests.
std::set<std::vector<std::uint64_t>> boundaryFacetSet(const Spacetime &spacetime) {
  std::set<std::vector<std::uint64_t>> facets;
  for (auto facet : spacetime.getBoundary()) {  // getBoundary() returns a fresh copy
    std::sort(facet.begin(), facet.end());
    facets.insert(facet);
  }
  return facets;
}

using Cell = std::vector<std::uint64_t>;
using FixedCochainTarget = std::map<Cell, complexd>;

struct BoundaryComponentData {
  std::vector<Cell> facets;
  std::set<std::uint64_t> vertices;
};

std::vector<BoundaryComponentData> boundaryComponents(
    const Spacetime &spacetime) {
  std::vector<Cell> facets;
  for (auto facet : spacetime.getBoundary()) {
    std::sort(facet.begin(), facet.end());
    if (facet.empty() ||
        std::adjacent_find(facet.begin(), facet.end()) != facet.end())
      throw std::invalid_argument(
          "MultiCobordism::relaxBoundaryStatePairs: malformed boundary "
          "facet");
    facets.push_back(std::move(facet));
  }
  std::sort(facets.begin(), facets.end());
  facets.erase(std::unique(facets.begin(), facets.end()), facets.end());
  if (facets.empty()) return {};

  const std::size_t facetWidth = facets.front().size();
  for (const auto &facet : facets)
    if (facet.size() != facetWidth)
      throw std::invalid_argument(
          "MultiCobordism::relaxBoundaryStatePairs: non-pure boundary");

  auto boundary = Spacetime::fromCells(
      spacetime.getDimensions() - 1, facets, 1.0, 0.0);
  std::vector<BoundaryComponentData> components;
  std::map<std::uint64_t, std::size_t> componentByVertex;
  for (const auto &vertices : boundary->getConnectedComponents()) {
    const std::size_t componentIndex = components.size();
    BoundaryComponentData component;
    for (const auto *vertex : vertices) {
      const std::uint64_t vertexId = vertex->getId();
      if (!componentByVertex.emplace(vertexId, componentIndex).second)
        throw std::logic_error(
            "MultiCobordism::relaxBoundaryStatePairs: boundary vertex "
            "belongs to multiple components");
      component.vertices.insert(vertexId);
    }
    components.push_back(std::move(component));
  }
  for (const auto &facet : facets) {
    const auto owner = componentByVertex.find(facet.front());
    if (owner == componentByVertex.end())
      throw std::logic_error(
          "MultiCobordism::relaxBoundaryStatePairs: boundary facet has no "
          "component");
    for (const std::uint64_t vertexId : facet) {
      const auto component = componentByVertex.find(vertexId);
      if (component == componentByVertex.end() ||
          component->second != owner->second)
        throw std::logic_error(
            "MultiCobordism::relaxBoundaryStatePairs: boundary facet spans "
            "multiple components");
    }
    components[owner->second].facets.push_back(facet);
  }
  for (auto &component : components)
    std::sort(component.facets.begin(), component.facets.end());
  std::sort(components.begin(), components.end(),
            [](const BoundaryComponentData &left,
               const BoundaryComponentData &right) {
              return left.facets < right.facets;
            });
  return components;
}

std::set<Cell> componentCells(const BoundaryComponentData &component,
                              int degree) {
  const std::size_t width = static_cast<std::size_t>(degree) + 1;
  std::set<Cell> cells;
  for (const auto &facet : component.facets) {
    if (width > facet.size()) continue;
    Cell cell;
    cell.reserve(width);
    const std::function<void(std::size_t, std::size_t)> enumerate =
        [&](std::size_t start, std::size_t remaining) {
          if (remaining == 0) {
            cells.insert(cell);
            return;
          }
          for (std::size_t index = start;
               index + remaining <= facet.size(); ++index) {
            cell.push_back(facet[index]);
            enumerate(index + 1, remaining - 1);
            cell.pop_back();
          }
        };
    enumerate(0, width);
  }
  return cells;
}

bool finiteComplex(complexd value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

std::vector<complexd> normalized(std::vector<complexd> state) {
  double normSquared = 0.0;
  for (const complexd value : state) normSquared += std::norm(value);
  if (!(normSquared > 0.0) || !std::isfinite(normSquared))
    throw std::invalid_argument(
        "MultiCobordism: cochain state must be finite and nonzero");
  const double inverseNorm = 1.0 / std::sqrt(normSquared);
  for (complexd &value : state) value *= inverseNorm;
  return state;
}

struct BoundaryStateEvaluation {
  std::vector<double> residuals;
};

BoundaryStateEvaluation evaluateBoundaryStates(
    const std::shared_ptr<Spacetime> &spacetime,
    const BoundaryComponentData &component, int degree,
    const std::vector<Cell> &orderedCells,
    const std::vector<std::vector<complexd>> &states,
    HodgeLaplacian::MetricSource metricSource) {
  auto boundary = Spacetime::fromCells(spacetime->getDimensions() - 1,
                                       component.facets, 1.0, 0.0);
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           ::tessera::mesh::Edge *>
      parentEdges;
  for (auto *edge : spacetime->getEdgeList()->toVector())
    parentEdges.emplace(edgeKey(edge), edge);
  for (auto *edge : boundary->getEdgeList()->toVector()) {
    const auto parent = parentEdges.find(edgeKey(edge));
    if (parent == parentEdges.end())
      throw std::logic_error(
          "MultiCobordism::relaxBoundaryStatePairs: boundary edge is absent "
          "from the live cobordism");
    edge->setLength(parent->second->getLength());
    edge->setPhase(parent->second->getPhase());
  }

  EigenstateSynthesis synthesis(boundary, degree, metricSource);
  std::map<Cell, std::size_t> suppliedIndex;
  for (std::size_t index = 0; index < orderedCells.size(); ++index)
    suppliedIndex.emplace(orderedCells[index], index);

  BoundaryStateEvaluation evaluation;
  evaluation.residuals.reserve(states.size());
  for (const auto &state : states) {
    std::vector<complexd> canonical(synthesis.order());
    for (std::size_t index = 0; index < synthesis.order(); ++index) {
      const auto supplied =
          suppliedIndex.find(synthesis.cellSimplices()[index]);
      if (supplied == suppliedIndex.end())
        throw std::logic_error(
            "MultiCobordism::relaxBoundaryStatePairs: boundary cell frame "
            "does not match the isolated boundary");
      canonical[index] = state[supplied->second];
    }
    evaluation.residuals.push_back(synthesis.residual(canonical));
  }
  return evaluation;
}

struct FixedCochainOptimization {
  double residual{std::numeric_limits<double>::infinity()};
  double eigenvalue{0.0};
  std::size_t freeEdgeCount{0};
  /// Free amplitude coordinates summed over the witnesses: the auxiliary
  /// cells, minus the readout rank when a readout system is imposed.
  std::size_t auxiliaryCellCount{0};
  std::size_t readoutRank{0};
  std::vector<std::vector<complexd>> states;
  std::vector<double> stateResiduals;
  std::vector<double> stateEigenvalues;
};

/// Exact linear readout constraints on every witness (#936): chain `r` paired
/// with witness `j` must equal `targets[j][r]`. The auxiliary block of each
/// witness is parametrized on the affine solution set of its readout system,
/// so the constraints hold exactly at every iterate and no penalty enters.
struct ReadoutSystem {
  const std::vector<MultiCobordism::ReadoutChain> *chains{nullptr};
  const std::vector<std::vector<complexd>> *targets{nullptr};
};

/// The affine parametrization `auxiliary = offset + basis * coordinates` of one
/// witness's auxiliary block (identity when no readout system is imposed).
struct AffineAuxiliary {
  Eigen::VectorXcd offset;
  Eigen::MatrixXcd basis;
};

/// A warm start for one relaxation pass (#936): the previous pass's witnesses
/// keyed by cell (cells absent from the live complex, i.e. created by growth,
/// start at zero) together with the live edge geometry. The warm start is
/// descended first; the remaining restarts are drawn as before.
struct WarmStart {
  std::vector<std::map<Cell, complexd>> states;
};

FixedCochainOptimization optimizeFixedCochainTargets(
    const std::shared_ptr<Spacetime> &spacetime,
    EigenstateSynthesis &synthesis,
    const std::vector<FixedCochainTarget> &fixedTargets,
    const std::function<bool(std::uint64_t, std::uint64_t)> &edgeIsFree,
    bool commonEigenvalue, double epsilon, int restarts, std::uint64_t seed,
    int maxIterations, const ReadoutSystem *readoutSystem = nullptr,
    const WarmStart *warmStart = nullptr) {
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kWeightMinimum = 0.1;
  constexpr double kWeightMaximum = 10.0;
  constexpr double kPhaseBound = 2.0 * kPi;
  constexpr double kAuxiliaryBound = 5.0;

  const bool usePhases = synthesis.degree() == 0;
  const std::size_t order = synthesis.order();
  const auto &cells = synthesis.cellSimplices();

  std::vector<::tessera::mesh::Edge *> freeEdges;
  for (auto *edge : spacetime->getEdgeList()->toVector()) {
    const auto [a, b] = edgeKey(edge);
    if (edgeIsFree(a, b)) freeEdges.push_back(edge);
  }

  std::vector<std::vector<complexd>> fixedValues(
      fixedTargets.size(), std::vector<complexd>(order, complexd(0.0, 0.0)));
  std::vector<std::vector<std::size_t>> auxiliaryIndices(fixedTargets.size());
  std::vector<std::size_t> auxiliaryOffsets(fixedTargets.size(), 0);
  std::size_t totalAuxiliaryCount = 0;
  for (std::size_t stateIndex = 0; stateIndex < fixedTargets.size();
       ++stateIndex) {
    std::size_t matchedCount = 0;
    for (std::size_t cellIndex = 0; cellIndex < order; ++cellIndex) {
      const auto fixed = fixedTargets[stateIndex].find(cells[cellIndex]);
      if (fixed != fixedTargets[stateIndex].end()) {
        fixedValues[stateIndex][cellIndex] = fixed->second;
        ++matchedCount;
      } else {
        auxiliaryIndices[stateIndex].push_back(cellIndex);
      }
    }
    if (matchedCount != fixedTargets[stateIndex].size())
      throw std::invalid_argument(
          "MultiCobordism: a fixed cochain cell is absent from the live "
          "complex");
  }

  const bool hasAffine = readoutSystem != nullptr &&
                         readoutSystem->chains != nullptr &&
                         !readoutSystem->chains->empty();
  std::vector<AffineAuxiliary> affine(fixedTargets.size());
  std::size_t readoutRank = 0;
  if (hasAffine) {
    const auto &chains = *readoutSystem->chains;
    const auto &targets = *readoutSystem->targets;
    if (targets.size() != fixedTargets.size())
      throw std::invalid_argument(
          "MultiCobordism: readout target count does not match the witness "
          "count");
    std::map<Cell, std::size_t> cellIndex;
    for (std::size_t index = 0; index < order; ++index)
      cellIndex.emplace(cells[index], index);
    // The readout matrix over ALL cells, in canonical cell order.
    Eigen::MatrixXcd readoutMatrix = Eigen::MatrixXcd::Zero(
        static_cast<Eigen::Index>(chains.size()),
        static_cast<Eigen::Index>(order));
    for (std::size_t row = 0; row < chains.size(); ++row) {
      for (const auto &[cell, coefficient] : chains[row]) {
        const auto found = cellIndex.find(cell);
        if (found == cellIndex.end())
          throw std::invalid_argument(
              "MultiCobordism: a readout chain cell is absent from the live "
              "complex");
        readoutMatrix(static_cast<Eigen::Index>(row),
                      static_cast<Eigen::Index>(found->second)) +=
            coefficient;
      }
    }
    for (std::size_t stateIndex = 0; stateIndex < fixedTargets.size();
         ++stateIndex) {
      if (targets[stateIndex].size() != chains.size())
        throw std::invalid_argument(
            "MultiCobordism: a readout target row does not match the readout "
            "chain count");
      const auto &auxiliary = auxiliaryIndices[stateIndex];
      const Eigen::Index rows = static_cast<Eigen::Index>(chains.size());
      const Eigen::Index columns = static_cast<Eigen::Index>(auxiliary.size());
      // Right-hand side: the target minus the fixed cells' contribution.
      Eigen::VectorXcd rhs(rows);
      for (Eigen::Index row = 0; row < rows; ++row) {
        complexd fixedPart(0.0, 0.0);
        for (std::size_t cell = 0; cell < order; ++cell)
          fixedPart += readoutMatrix(row, static_cast<Eigen::Index>(cell)) *
                       fixedValues[stateIndex][cell];
        rhs[row] = targets[stateIndex][static_cast<std::size_t>(row)] -
                   fixedPart;
      }
      Eigen::MatrixXcd auxiliaryMatrix(rows, columns);
      for (Eigen::Index column = 0; column < columns; ++column)
        auxiliaryMatrix.col(column) = readoutMatrix.col(
            static_cast<Eigen::Index>(auxiliary[static_cast<std::size_t>(column)]));
      // Fixed cells contribute only to the right-hand side above; the fixed
      // values of the auxiliary block are zero there by construction.
      Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
          auxiliaryMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
      svd.setThreshold(1e-12);
      const Eigen::Index rank = columns == 0 ? 0 : svd.rank();
      Eigen::VectorXcd particular = Eigen::VectorXcd::Zero(columns);
      if (columns > 0) particular = svd.solve(rhs);
      const double inconsistency =
          (auxiliaryMatrix * particular - rhs).norm();
      if (!(inconsistency <= 1e-10 * std::max(1.0, rhs.norm())))
        throw std::invalid_argument(
            "MultiCobordism: the readout targets of witness " +
            std::to_string(stateIndex) +
            " are inconsistent with its fixed amplitudes on the live "
            "complex (readout system has no solution)");
      affine[stateIndex].offset = std::move(particular);
      affine[stateIndex].basis =
          columns == 0 ? Eigen::MatrixXcd(0, 0)
                       : Eigen::MatrixXcd(svd.matrixV().rightCols(columns - rank));
      if (stateIndex == 0) readoutRank = static_cast<std::size_t>(rank);
    }
  }
  // Free amplitude coordinates per witness: the auxiliary cells, or the
  // readout null-space dimension when a readout system is imposed.
  std::vector<std::size_t> coordinateCounts(fixedTargets.size(), 0);
  for (std::size_t stateIndex = 0; stateIndex < fixedTargets.size();
       ++stateIndex) {
    coordinateCounts[stateIndex] =
        hasAffine ? static_cast<std::size_t>(affine[stateIndex].basis.cols())
                  : auxiliaryIndices[stateIndex].size();
    auxiliaryOffsets[stateIndex] = totalAuxiliaryCount;
    totalAuxiliaryCount += coordinateCounts[stateIndex];
  }

  const std::size_t weightCount = freeEdges.size();
  const std::size_t phaseCount = usePhases ? freeEdges.size() : 0;
  const std::size_t parameterCount =
      weightCount + phaseCount + 2 * totalAuxiliaryCount;

  const auto buildStates =
      [&fixedValues, &auxiliaryIndices, &auxiliaryOffsets, &coordinateCounts,
       &affine, hasAffine, weightCount,
       phaseCount](const Eigen::VectorXd &parameters) {
        std::vector<std::vector<complexd>> states = fixedValues;
        for (std::size_t stateIndex = 0; stateIndex < states.size();
             ++stateIndex) {
          const Eigen::Index base = static_cast<Eigen::Index>(
              weightCount + phaseCount + 2 * auxiliaryOffsets[stateIndex]);
          if (!hasAffine) {
            for (std::size_t auxiliaryIndex = 0;
                 auxiliaryIndex < auxiliaryIndices[stateIndex].size();
                 ++auxiliaryIndex) {
              const Eigen::Index parameterIndex =
                  base + static_cast<Eigen::Index>(2 * auxiliaryIndex);
              states[stateIndex]
                    [auxiliaryIndices[stateIndex][auxiliaryIndex]] =
                  complexd(parameters[parameterIndex],
                           parameters[parameterIndex + 1]);
            }
            continue;
          }
          Eigen::VectorXcd coordinates(
              static_cast<Eigen::Index>(coordinateCounts[stateIndex]));
          for (Eigen::Index index = 0; index < coordinates.size(); ++index)
            coordinates[index] = complexd(parameters[base + 2 * index],
                                          parameters[base + 2 * index + 1]);
          const Eigen::VectorXcd auxiliary =
              affine[stateIndex].offset +
              affine[stateIndex].basis * coordinates;
          for (std::size_t auxiliaryIndex = 0;
               auxiliaryIndex < auxiliaryIndices[stateIndex].size();
               ++auxiliaryIndex)
            states[stateIndex][auxiliaryIndices[stateIndex][auxiliaryIndex]] =
                auxiliary[static_cast<Eigen::Index>(auxiliaryIndex)];
        }
        return states;
      };

  const auto writeGeometry =
      [&freeEdges, weightCount, phaseCount](const Eigen::VectorXd &parameters) {
        for (std::size_t index = 0; index < weightCount; ++index)
          freeEdges[index]->setLength(std::sqrt(complexd{
              parameters[static_cast<Eigen::Index>(index)], 0.0}));
        for (std::size_t index = 0; index < phaseCount; ++index)
          freeEdges[index]->setPhase(complexd{
              parameters[static_cast<Eigen::Index>(weightCount + index)],
              0.0});
      };

  const auto residual =
      [&synthesis, &buildStates, &writeGeometry, commonEigenvalue,
       order](const Eigen::VectorXd &parameters) {
        writeGeometry(parameters);
        std::vector<std::vector<complexd>> states = buildStates(parameters);
        std::vector<std::vector<complexd>> applied(states.size());
        std::vector<double> eigenvalues(states.size(), 0.0);
        for (std::size_t stateIndex = 0; stateIndex < states.size();
             ++stateIndex) {
          states[stateIndex] = normalized(std::move(states[stateIndex]));
          applied[stateIndex] = synthesis.apply(states[stateIndex]);
          complexd rayleigh(0.0, 0.0);
          for (std::size_t index = 0; index < order; ++index)
            rayleigh += std::conj(states[stateIndex][index]) *
                        applied[stateIndex][index];
          eigenvalues[stateIndex] = rayleigh.real();
        }
        const double sharedEigenvalue =
            std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0) /
            static_cast<double>(eigenvalues.size());
        Eigen::VectorXd values(static_cast<Eigen::Index>(
            2 * order * states.size()));
        for (std::size_t stateIndex = 0; stateIndex < states.size();
             ++stateIndex) {
          const double eigenvalue =
              commonEigenvalue ? sharedEigenvalue : eigenvalues[stateIndex];
          const Eigen::Index base =
              static_cast<Eigen::Index>(2 * order * stateIndex);
          for (std::size_t index = 0; index < order; ++index) {
            const complexd difference =
                applied[stateIndex][index] -
                eigenvalue * states[stateIndex][index];
            values[base + static_cast<Eigen::Index>(index)] =
                difference.real();
            values[base + static_cast<Eigen::Index>(order + index)] =
                difference.imag();
          }
        }
        return values;
      };

  const auto clamp =
      [weightCount, phaseCount,
       totalAuxiliaryCount, kWeightMinimum, kWeightMaximum, kPhaseBound,
       kAuxiliaryBound](const Eigen::VectorXd &parameters) {
        Eigen::VectorXd clamped = parameters;
        for (std::size_t index = 0; index < weightCount; ++index)
          clamped[static_cast<Eigen::Index>(index)] = std::clamp(
              clamped[static_cast<Eigen::Index>(index)], kWeightMinimum,
              kWeightMaximum);
        for (std::size_t index = 0; index < phaseCount; ++index) {
          const Eigen::Index phaseIndex =
              static_cast<Eigen::Index>(weightCount + index);
          clamped[phaseIndex] =
              std::clamp(clamped[phaseIndex], -kPhaseBound, kPhaseBound);
        }
        for (std::size_t index = 0; index < 2 * totalAuxiliaryCount; ++index) {
          const Eigen::Index auxiliaryIndex = static_cast<Eigen::Index>(
              weightCount + phaseCount + index);
          clamped[auxiliaryIndex] =
              std::clamp(clamped[auxiliaryIndex], -kAuxiliaryBound,
                         kAuxiliaryBound);
        }
        return clamped;
      };

  std::uniform_real_distribution<double> weightDistribution(
      kWeightMinimum, kWeightMaximum);
  std::uniform_real_distribution<double> phaseDistribution(-kPi, kPi);
  std::uniform_real_distribution<double> auxiliaryDistribution(-1.0, 1.0);
  const auto sample =
      [weightCount, phaseCount, totalAuxiliaryCount, weightDistribution,
       phaseDistribution,
       auxiliaryDistribution](std::mt19937_64 &randomEngine) mutable {
        Eigen::VectorXd parameters(static_cast<Eigen::Index>(
            weightCount + phaseCount + 2 * totalAuxiliaryCount));
        for (std::size_t index = 0; index < weightCount; ++index)
          parameters[static_cast<Eigen::Index>(index)] =
              weightDistribution(randomEngine);
        for (std::size_t index = 0; index < phaseCount; ++index)
          parameters[static_cast<Eigen::Index>(weightCount + index)] =
              phaseDistribution(randomEngine);
        for (std::size_t index = 0; index < 2 * totalAuxiliaryCount; ++index)
          parameters[static_cast<Eigen::Index>(weightCount + phaseCount +
                                                index)] =
              auxiliaryDistribution(randomEngine);
        return parameters;
      };

  const LevenbergMarquardt solver(maxIterations, epsilon);
  LevenbergMarquardt::Result best;
  int remainingRestarts = restarts;
  if (warmStart != nullptr) {
    if (warmStart->states.size() != fixedTargets.size())
      throw std::invalid_argument(
          "MultiCobordism: warm-start witness count does not match");
    Eigen::VectorXd start = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(parameterCount));
    for (std::size_t index = 0; index < weightCount; ++index)
      start[static_cast<Eigen::Index>(index)] =
          (freeEdges[index]->getLength() * freeEdges[index]->getLength())
              .real();
    for (std::size_t index = 0; index < phaseCount; ++index)
      start[static_cast<Eigen::Index>(weightCount + index)] =
          freeEdges[index]->getPhase().real();
    for (std::size_t stateIndex = 0; stateIndex < fixedTargets.size();
         ++stateIndex) {
      const auto &auxiliary = auxiliaryIndices[stateIndex];
      Eigen::VectorXcd previous = Eigen::VectorXcd::Zero(
          static_cast<Eigen::Index>(auxiliary.size()));
      for (std::size_t auxiliaryIndex = 0; auxiliaryIndex < auxiliary.size();
           ++auxiliaryIndex) {
        const auto found =
            warmStart->states[stateIndex].find(cells[auxiliary[auxiliaryIndex]]);
        if (found != warmStart->states[stateIndex].end())
          previous[static_cast<Eigen::Index>(auxiliaryIndex)] = found->second;
      }
      const Eigen::VectorXcd coordinates =
          hasAffine ? Eigen::VectorXcd(affine[stateIndex].basis.adjoint() *
                                       (previous - affine[stateIndex].offset))
                    : previous;
      const Eigen::Index base = static_cast<Eigen::Index>(
          weightCount + phaseCount + 2 * auxiliaryOffsets[stateIndex]);
      for (Eigen::Index index = 0; index < coordinates.size(); ++index) {
        start[base + 2 * index] = coordinates[index].real();
        start[base + 2 * index + 1] = coordinates[index].imag();
      }
    }
    best = solver.minimize(residual, clamp, clamp(start));
    remainingRestarts = restarts - 1;
  }
  if (remainingRestarts > 0 && !(best.cost < epsilon)) {
    auto trial = solver.multiRestart(residual, clamp, sample, parameterCount,
                                     remainingRestarts, seed, epsilon);
    if (trial.cost < best.cost) best = std::move(trial);
  }
  (void)residual(best.parameters);

  FixedCochainOptimization result;
  result.freeEdgeCount = freeEdges.size();
  result.auxiliaryCellCount = totalAuxiliaryCount;
  result.readoutRank = readoutRank;
  result.states = buildStates(best.parameters);
  result.stateEigenvalues.reserve(result.states.size());
  std::vector<std::vector<complexd>> unitStates;
  unitStates.reserve(result.states.size());
  for (const auto &state : result.states) {
    unitStates.push_back(normalized(state));
    result.stateEigenvalues.push_back(synthesis.rayleigh(state));
  }
  result.eigenvalue =
      std::accumulate(result.stateEigenvalues.begin(),
                      result.stateEigenvalues.end(), 0.0) /
      static_cast<double>(result.stateEigenvalues.size());
  result.stateResiduals.reserve(result.states.size());
  result.residual = 0.0;
  for (std::size_t stateIndex = 0; stateIndex < result.states.size();
       ++stateIndex) {
    const auto applied = synthesis.apply(unitStates[stateIndex]);
    const double eigenvalue =
        commonEigenvalue ? result.eigenvalue
                         : result.stateEigenvalues[stateIndex];
    double stateResidual = 0.0;
    for (std::size_t index = 0; index < order; ++index)
      stateResidual += std::norm(
          applied[index] - eigenvalue * unitStates[stateIndex][index]);
    result.stateResiduals.push_back(stateResidual);
    result.residual += stateResidual;
  }
  return result;
}

}  // namespace

MultiCobordism::MultiCobordism(
    std::shared_ptr<Spacetime> host,
    const std::vector<std::vector<complexd>> &inputTargets,
    const std::vector<std::vector<complexd>> &outputTargets,
    const std::vector<int> &degrees, double gamma, std::uint64_t seed,
    int precone, bool shouldProposeDispositions, bool preconeTimelike,
    bool preconeAlternate, bool balancedEdgeWiring, bool singularValueRatio,
    bool einsteinHilbert, bool realSquaredLengthsOnly,
    HodgeLaplacian::MetricSource metricSource)
    : spacetime_(std::move(host)),
      inputTargets_(inputTargets),
      outputTargets_(outputTargets),
      registerDegrees_(degrees),
      dualComplexGateDegree_(
          registerDegrees_.empty()
              ? 0
              : *std::max_element(registerDegrees_.begin(),
                                  registerDegrees_.end())),
      gamma_(gamma),
      balancedEdgeWiring_(balancedEdgeWiring),
      singularValueRatio_(singularValueRatio),
      einsteinHilbert_(einsteinHilbert),
      realSquaredLengthsOnly_(realSquaredLengthsOnly),
      metricSource_(metricSource),
      randomNumberGenerator_(seed) {
  // The wiring mode must reach the host BEFORE any precone growth below wires
  // its first edge (#690).
  if (spacetime_) spacetime_->setBalancedEdgeWiring(balancedEdgeWiring_);
  // Assigned in the body rather than the init list: the member is declared last,
  // and C++ initializes in DECLARATION order, so an init-list entry here would
  // reorder-warn. It is a plain bool with an in-class default, so nothing depends
  // on it being set earlier.
  shouldProposeDispositions_ = shouldProposeDispositions;
  // #776: the deterministic provenance stamp of every checkpoint this node
  // writes. Assigned in the body for the same declaration-order reason.
  seed_ = seed;
  // Install the built-in matching the default mode, so `objectiveSpec_` is
  // never null and a caller that never injects one descends exactly the
  // objective it descended before this became injectable.
  objectiveSpec_ = std::make_shared<LegacyObjective>();
  // Pre-grow the seed by `precone` gated cone-ins before any optimization, so the
  // stage-1 search starts from a larger complex grown emergently from the host (no
  // input/output block is seeded yet, so nothing is pinned — the gate is the only
  // constraint). `precone <= 0` leaves the host and RNG untouched.
  // `preconeTimelike` draws every cone-in as the TIMELIKE disposition (#613);
  // `preconeAlternate` instead ALTERNATES timelike/spacelike for balanced
  // causal content (it wins when both are set). Default: all-spacelike.
  if (precone > 0) preconeCells(precone, preconeTimelike, preconeAlternate);
}

std::vector<int> MultiCobordism::betti(const Spacetime &spacetime) {
  // Betti numbers are purely combinatorial, and the residual path calls this
  // on every objective evaluation while only edge lengths move (7.2% of a
  // live perf sample went to Smith normal form). The spacetime's structural
  // revision proves when the last computation is still exact (#681).
  if (const auto *cached = spacetime.cachedBettiNumbers()) return *cached;
  auto numbers = ChainComplex::fromSpacetime(spacetime).bettiNumbers();
  spacetime.storeBettiNumbers(numbers);
  return numbers;
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::emergentHoles(
    const Spacetime &spacetime, int registerDegree) {
  // The (k+2)-vertex tuples all of whose drop-one facets are boundary facets.
  std::set<std::vector<std::uint64_t>> boundaryFacets;
  for (auto boundaryFacet : spacetime.getBoundary()) {
    std::sort(boundaryFacet.begin(), boundaryFacet.end());
    boundaryFacets.insert(std::move(boundaryFacet));
  }
  std::vector<std::vector<std::uint64_t>> emergentHoleTuples;
  if (boundaryFacets.empty() ||
      static_cast<int>(boundaryFacets.begin()->size()) !=
          registerDegree + 1)  // facets must be k-cells
    return emergentHoleTuples;
  std::set<std::uint64_t> boundaryVertexIds;
  for (const auto &boundaryFacet : boundaryFacets)
    for (auto vertexId : boundaryFacet) boundaryVertexIds.insert(vertexId);
  std::set<std::vector<std::uint64_t>> emergentHoleSet;
  for (const auto &boundaryFacet : boundaryFacets) {
    for (auto candidateVertexId : boundaryVertexIds) {
      if (std::find(boundaryFacet.begin(), boundaryFacet.end(),
                    candidateVertexId) != boundaryFacet.end())
        continue;
      std::vector<std::uint64_t> candidateHole = boundaryFacet;
      candidateHole.push_back(candidateVertexId);
      std::sort(candidateHole.begin(), candidateHole.end());
      bool allFacetsAreBoundary = true;
      for (std::size_t droppedIndex = 0; droppedIndex < candidateHole.size();
           ++droppedIndex) {
        std::vector<std::uint64_t> dropOneFacet;
        for (std::size_t copyIndex = 0; copyIndex < candidateHole.size();
             ++copyIndex)
          if (copyIndex != droppedIndex)
            dropOneFacet.push_back(candidateHole[copyIndex]);
        if (!boundaryFacets.count(dropOneFacet)) {
          allFacetsAreBoundary = false;
          break;
        }
      }
      if (allFacetsAreBoundary) emergentHoleSet.insert(candidateHole);
    }
  }
  emergentHoleTuples.assign(emergentHoleSet.begin(), emergentHoleSet.end());
  return emergentHoleTuples;
}

double MultiCobordism::reggeActionGradient(
    const std::shared_ptr<Spacetime> &spacetime) {
  ReggeSolver reggeSolver(spacetime, MatterConfiguration());
  double squaredGradientNorm = 0.0;
  for (const auto &gradientComponent : reggeSolver.actionGradientExact())
    squaredGradientNorm += std::norm(gradientComponent);
  return squaredGradientNorm;
}

Eigen::VectorXcd MultiCobordism::targetStateVector(
    const std::vector<complexd> &targetState) {
  Eigen::VectorXcd targetVector(targetState.size());
  for (std::size_t componentIndex = 0; componentIndex < targetState.size();
       ++componentIndex)
    targetVector(componentIndex) = targetState[componentIndex];
  return targetVector;
}

std::vector<std::vector<std::uint64_t>> MultiCobordism::holesCarryingTheTarget(
    const Spacetime &spacetime, int registerDegree, std::size_t targetDimension) {
  auto emergentHoleTuples = emergentHoles(spacetime, registerDegree);
  // One target component per hole: holes beyond the target's width have no component
  // to carry and take no part in the fit.
  if (emergentHoleTuples.size() > targetDimension)
    emergentHoleTuples.resize(targetDimension);
  return emergentHoleTuples;
}

Eigen::MatrixXcd MultiCobordism::holePeriodMatrix(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    int degreeBettiNumber,
    const std::vector<std::vector<std::uint64_t>> &cycleHoles,
    std::size_t targetDimension, HodgeLaplacian::MetricSource metricSource) {
  EigenstateSynthesis eigenstateSynthesis(spacetime, registerDegree, metricSource);
  const auto flattenedCyclePeriods =
      eigenstateSynthesis.cyclePeriods(cycleHoles);  // rank x m, row-major
  const std::size_t holeCount = cycleHoles.size();
  // The row count of the flattened periods is the NUMERIC harmonic-kernel
  // dimension the synthesizer actually computed (HodgeLaplacian::harmonicMatrix
  // at its rank threshold, metric-dependent) — NOT necessarily the INTEGER
  // Betti number: on geometrically extreme complexes (e.g. deep-lookahead
  // candidates near the null-face locus) the numeric rank can
  // fall below the topological one, and indexing by the Betti count then read
  // past the end of the vector — the measured #636 segfault (thread 1 in
  // residualOfTargetStateAgainstHarmonic while scoring one). Bound every
  // index by the data's own shape; fewer usable harmonics honestly means a
  // LARGER residual, never an out-of-bounds read (a zero-column matrix reads as
  // the full leak in the caller).
  const std::size_t periodRowCount =
      holeCount == 0 ? 0 : flattenedCyclePeriods.size() / holeCount;
  const int harmonicRank =
      std::min(degreeBettiNumber, static_cast<int>(periodRowCount));
  Eigen::MatrixXcd periodMatrixTransposed = Eigen::MatrixXcd::Zero(
      static_cast<int>(targetDimension), std::max(harmonicRank, 0));
  for (int harmonicIndex = 0; harmonicIndex < harmonicRank; ++harmonicIndex)
    for (std::size_t holeIndex = 0; holeIndex < holeCount; ++holeIndex)
      periodMatrixTransposed(static_cast<int>(holeIndex), harmonicIndex) =
          flattenedCyclePeriods[static_cast<std::size_t>(harmonicIndex) * holeCount +
                                holeIndex];
  return periodMatrixTransposed;
}

Eigen::VectorXcd MultiCobordism::relabeledTargetVector(
    const Eigen::VectorXcd &targetVector, const std::vector<int> &relabeling) {
  Eigen::VectorXcd relabeled(targetVector.size());
  for (std::size_t holeIndex = 0; holeIndex < relabeling.size(); ++holeIndex)
    relabeled(holeIndex) = targetVector(relabeling[holeIndex]);
  return relabeled;
}

MultiCobordism::RelabelingMatch MultiCobordism::bestRelabelingOfTarget(
    const Eigen::MatrixXcd &periodMatrixTransposed,
    const Eigen::VectorXcd &targetVector,
    const std::set<std::vector<int>> &claimedMatchings, bool skipClaimed) {
  // min over the relabelings of the target components of ||pdT c - ts||^2 (lstsq c).
  // Total over EVERY configuration (#699): a non-finite period matrix (an
  // unbounded stage-2 trial overflowed the polynomial cell weights, so the
  // harmonic periods left double range) scores +inf — an infinitely bad
  // configuration the line search rejects — instead of handing non-finite
  // input to BDCSVD, whose compute/solve is undefined behavior with asserts
  // compiled out (measured: a general protection fault inside rank()).
  if (!periodMatrixTransposed.allFinite()) {
    std::vector<int> identityRelabeling(
        static_cast<std::size_t>(targetVector.size()));
    std::iota(identityRelabeling.begin(), identityRelabeling.end(), 0);
    return {std::numeric_limits<double>::infinity(), identityRelabeling, true};
  }
  Eigen::BDCSVD<Eigen::MatrixXcd> periodSvd(
      periodMatrixTransposed, Eigen::ComputeThinU | Eigen::ComputeThinV);
  RelabelingMatch bestMatch;
  std::vector<int> relabeling(static_cast<std::size_t>(targetVector.size()));
  std::iota(relabeling.begin(), relabeling.end(), 0);
  do {
    if (skipClaimed && claimedMatchings.count(relabeling)) continue;
    const Eigen::VectorXcd relabeledTarget =
        relabeledTargetVector(targetVector, relabeling);
    const Eigen::VectorXcd leastSquaresCoefficients =
        periodSvd.solve(relabeledTarget);
    const double residual =
        (periodMatrixTransposed * leastSquaresCoefficients - relabeledTarget)
            .squaredNorm();
    if (!bestMatch.scored || residual < bestMatch.residual)
      bestMatch = {residual, relabeling, true};
  } while (std::next_permutation(relabeling.begin(), relabeling.end()));
  return bestMatch;
}

double MultiCobordism::residualOfTargetStateAgainstHarmonic(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    const std::vector<complexd> &targetState,
    HodgeLaplacian::MetricSource metricSource) {
  // No other register to collide with: an empty claim set excludes nothing, so this
  // is the unconstrained min over the relabelings (`r_state`, the reference read-out).
  std::set<std::vector<int>> claimedMatchings;
  return residualOfTargetStateAgainstHarmonicWithDistinctMatching(
      spacetime, registerDegree, targetState, claimedMatchings, metricSource);
}

double MultiCobordism::residualOfTargetStateAgainstHarmonicWithDistinctMatching(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    const std::vector<complexd> &targetState,
    std::set<std::vector<int>> &claimedMatchings,
    HodgeLaplacian::MetricSource metricSource) {
  const Eigen::VectorXcd targetVector = targetStateVector(targetState);
  const double fullLeakResidual = targetVector.squaredNorm();  // zero-filled leak

  const auto bettiNumbers = betti(*spacetime);
  if (registerDegree < 0) //||  # TODO: why would we exit if registerDegree is higher than betti numbers?
    return fullLeakResidual;
  if (registerDegree >= static_cast<int>(bettiNumbers.size())) {
    CLOG(WARN_LEVEL, "register degree was higher than bettiNumbers!");
    return fullLeakResidual;
  }
  const int degreeBettiNumber = bettiNumbers[registerDegree];
  if (degreeBettiNumber == 0) return fullLeakResidual;

  const auto cycleHoles =
      holesCarryingTheTarget(*spacetime, registerDegree, targetState.size());
  if (cycleHoles.empty()) return fullLeakResidual;
  const Eigen::MatrixXcd periodMatrixTransposed =
      holePeriodMatrix(spacetime, registerDegree, degreeBettiNumber, cycleHoles,
                       targetState.size(), metricSource);
  // The matrix is bounded by the NUMERIC harmonic rank (see holePeriodMatrix,
  // #636): zero usable harmonics on a geometrically extreme candidate means the
  // register carries nothing — the full leak — not an SVD of a 0-column matrix.
  if (periodMatrixTransposed.cols() == 0) return fullLeakResidual;

  // The relabeling this register wins is withheld from the registers scored after it,
  // so no two of them are read against the same matching of components onto holes.
  RelabelingMatch match = bestRelabelingOfTarget(
      periodMatrixTransposed, targetVector, claimedMatchings, /*skipClaimed=*/true);
  if (!match.scored) {
    // Every relabeling is already claimed — more registers than the d! this target
    // admits. Restart the exclusion rather than return the empty minimum.
    claimedMatchings.clear();
    match = bestRelabelingOfTarget(periodMatrixTransposed, targetVector,
                                   claimedMatchings, /*skipClaimed=*/false);
  }
  claimedMatchings.insert(match.relabeling);
  return match.residual;
}

double MultiCobordism::residualForBoundaryBlock(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime) const {
  std::set<std::vector<int>> claimedMatchings;
  return residualForBoundaryBlockWithDistinctMatchings(boundaryBlock, spacetime,
                                                       claimedMatchings);
}

double MultiCobordism::residualForBoundaryBlockWithDistinctMatchings(
    const BoundaryBlock &boundaryBlock,
    const std::shared_ptr<Spacetime> &spacetime,
    std::set<std::vector<int>> &claimedMatchings) const {
  if (useFiberResiduals_ && boundaryBlock.fiber && boundaryBlock.fiber->images.cols() > 0)
    return fiberResidualForBoundaryBlock(boundaryBlock, spacetime);  // #940
  auto blockSubcomplex = spacetime->subcomplexWithinVertexSet(
    boundaryBlock.vertices);
  double residual = 0.0;
  if (!blockSubcomplex)  // no complex to read: the target leaks in full, per degree
    return static_cast<double>(registerDegrees_.size()) *
           targetStateVector(boundaryBlock.target).squaredNorm();
  // The sub-complex is a FRESH spacetime whose per-instance Betti slot (#681)
  // is empty, so without help every evaluation would re-run the Smith normal
  // form — per block, per line-search trial, per candidate (measured: 47.5%
  // of live-run cycles). The block's topology is a pure function of the
  // PARENT's cells and the vertex set, so the parent caches the numbers per
  // (structural revision, region fingerprint): on a hit, pre-seed the child's
  // slot so betti() inside the scoring below never computes; on a miss, store
  // the child's freshly computed numbers back on the parent (#705).
  //
  // The region is named by `Fingerprint::fingerprintOf` over its vertex
  // identifiers — the class's own hash, called as a static because a
  // `Fingerprint` INSTANCE holds only `kMax` identifiers and drops the rest
  // silently, while a block region grows across the complex.
  const std::uint64_t vertexSetKey =
      ::tessera::mesh::Fingerprint::fingerprintOf(boundaryBlock.vertices);
  if (const auto *cached =
          spacetime->cachedSubcomplexBettiNumbers(vertexSetKey))
    blockSubcomplex->storeBettiNumbers(*cached);
  for (int registerDegree : registerDegrees_)
    residual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
        blockSubcomplex, registerDegree, boundaryBlock.target, claimedMatchings);
  if (const auto *computed = blockSubcomplex->cachedBettiNumbers())
    spacetime->storeSubcomplexBettiNumbers(vertexSetKey, *computed);
  return residual;
}

double MultiCobordism::rU(const std::shared_ptr<Spacetime> &spacetime) const {
  // The cobordism residual. INPUTS are localized boundary sub-complexes (built near
  // a seed, held representable by these terms, not pinned) — each read off its own
  // region and weighted by inputResidualWeight_ so they are not out-competed by the
  // whole/output term.
  //
  // ONE claim set spans the whole evaluation: every register here is scored by the
  // same min-over-relabelings, so without it they all pick the same argmin matching
  // and the sum is smallest when the registers carry identical weights. The set
  // records each register's winning matching and withholds it from the ones after.
  std::set<std::vector<int>> claimedMatchings;
  double totalResidual = 0.0;
  // Explicit constraints are already framed: their hole and target ordering is
  // fixed by the caller, so they bypass emergent-hole relabeling entirely. This
  // is still the existing exact-period r_U; only the frame is no longer guessed.
  for (const auto &constraint : registerConstraints_)
    totalResidual += EigenstateSynthesis(spacetime, constraint.degree, metricSource_)
                         .residualForPeriods(constraint.holes,
                                             constraint.target);
  for (const auto &inputBlock : inputBlocks_)
    totalResidual += inputResidualWeight_ *
                     residualForBoundaryBlockWithDistinctMatchings(inputBlock, spacetime,
                                                   claimedMatchings);
  if (outputTargets_.size() == 1) {
    // A SINGLE output is the whole cobordism's output boundary: as in the Python
    // reference it is "the harmonic of the entire structure", NEVER a pinned
    // region. Read it off the WHOLE complex so the bulk loop drives the whole to
    // carry it (the output EMERGES; it is not frozen by seedOutputs).
    // In the singularValueRatio mode this period read is part of the
    // whole-complex term the ratio below replaces, so it is skipped — the
    // output target then names an EXPECTATION for the after-the-fact readout
    // (and sizes expectedRegisterCount), never a scored prescription.
    if (!singularValueRatio_)
      for (int registerDegree : registerDegrees_)
        totalResidual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
            spacetime, registerDegree, outputTargets_.front(), claimedMatchings);
  } else {
    // Multiple outputs (e.g. a 2->2 recombination → diquark ⊔ antidiquark) live in
    // distinct regions: read each off its own constructed block. EMPTY outputTargets
    // is the supported nothing-pinned-downstream shape (#555): no output term at
    // all — rU is the weighted input residuals alone, and the whole's final state
    // emerges (read after the fact, e.g. ProtonIngredients' singlet diagnostic).
    for (const auto &outputBlock : outputBlocks_)
      totalResidual +=
          residualForBoundaryBlockWithDistinctMatchings(outputBlock, spacetime,
                                                        claimedMatchings);
    if (inputBlocks_.empty() && outputBlocks_.empty())  // bare objective, nothing built yet
      for (int registerDegree : registerDegrees_)
        for (const auto &outputTarget : outputTargets_)
          totalResidual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
              spacetime, registerDegree, outputTarget, claimedMatchings);
  }
  if (singularValueRatio_) {
    // The whole-complex term in the ratio mode (#697): one scale-invariant
    // spectral-shape term per degree covers BOTH regimes the two terms below
    // split between — it reads the full spectrum, so it presses from the bare
    // seed (no topological threshold) and keeps pressing after the holes open
    // (the lower half keeps collapsing past the exact kernel).
    for (int registerDegree : registerDegrees_)
      totalResidual += singularValueHalfSumRatio(spacetime, registerDegree, metricSource_);
    return totalResidual;
  }
  // The pre-topological register signal (#644): the period residuals above are
  // STEP functions in the topology — exactly flat until a register exists — so
  // they carry no register-seeking gradient at a seed. The near-kernel residual
  // is the same functional continued below the topological threshold, and it
  // saturates at 0 the moment b_k reaches the expected count (see the header).
  // Fiber-form targets (#940) declare no register count and no hole: the
  // whole-complex fiber residual replaces the near-kernel hole-forcing term.
  if (useFiberResiduals_) {
    if (wholeFiberTarget_) totalResidual += fiberResidualOn(spacetime, *wholeFiberTarget_);
    if (twoBodyTarget_) totalResidual += twoBodyResidualOn(spacetime, *twoBodyTarget_);
    return totalResidual;
  }
  const std::size_t expectedRegisters = expectedRegisterCount();
  if (expectedRegisters > 0)
    for (int registerDegree : registerDegrees_)
      totalResidual += nearKernelResidual(spacetime, registerDegree,
                                          expectedRegisters, metricSource_);
  return totalResidual;
}

std::size_t MultiCobordism::expectedRegisterCount() const {
  std::size_t expected = 0;
  for (const auto &target : inputTargets_)
    expected = std::max(expected, target.size());
  for (const auto &target : outputTargets_)
    expected = std::max(expected, target.size());
  return expected;
}

double MultiCobordism::nearKernelResidual(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    std::size_t expectedRegisterCount, HodgeLaplacian::MetricSource metricSource) {
  if (expectedRegisterCount == 0) return 0.0;
  cobordism::HodgeLaplacian laplacian(spacetime, HodgeLaplacian::defaultWeightConvention(), metricSource);
  // METRIC operator, deliberately: the term must feel the continuously-valued
  // edge lengths, so stage 2 can tune the CAUSAL STRUCTURE toward null
  // directions and open near-kernels with no holes at all — that channel is
  // the point, not a loophole (measured: a build driven this way ends with
  // most edges timelike and spectral near-kernels but zero topological holes).
  // Whether such causal near-kernels can CARRY a register is the next level of
  // exploration; the semantics for reading them out are not implemented here.
  // Stage-1 surgery remains the other route to the same descent: a genuine
  // hole zeroes the same singular values exactly.
  const std::vector<std::complex<double>> flat =
      laplacian.laplacian(registerDegree, /*metric=*/true);
  const std::size_t n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  // No k-cells at all: every expected register is missing — the worst case on
  // the normalized scale, 1 per missing dimension.
  if (n == 0) return static_cast<double>(expectedRegisterCount);
  Eigen::MatrixXcd L(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      L(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * n + j];
  // Total over EVERY configuration (#699): a non-finite operator evaluates to
  // +inf (see bestRelabelingOfTarget) rather than reaching BDCSVD.
  if (!L.allFinite()) return std::numeric_limits<double>::infinity();
  // Singular values of the NON-normal signed operator: the smooth surrogate for
  // the eigenvalue magnitudes (they share the kernel exactly).
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(L);
  const Eigen::VectorXd sigma = svd.singularValues();  // descending
  double total = 0.0;
  for (Eigen::Index i = 0; i < sigma.size(); ++i) total += sigma[i] * sigma[i];
  // L identically zero: every mode is kernel — nothing left to open.
  if (total <= 0.0) return 0.0;
  const std::size_t m = std::min(expectedRegisterCount, n);
  double smallest = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double s = sigma[static_cast<Eigen::Index>(n - 1 - i)];
    smallest += s * s;
  }
  // n * (smallest m) / (all): scale-invariant (L is degree −1 in l^2, so a raw
  // spectral sum is a conformal-inflation descent channel; the ratio is degree
  // 0). Missing dimensions count 1 each — the generic-mode value.
  return static_cast<double>(n) * smallest / total +
         static_cast<double>(expectedRegisterCount - m);
}

std::vector<std::complex<double>> MultiCobordism::nearKernelResidualGradient(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    std::size_t expectedRegisterCount, HodgeLaplacian::MetricSource metricSource) {
  // d/dl^2 of  r = n * (sum of the m smallest sigma^2) / (sum of all sigma^2).
  //
  // The sigma^2 are the eigenvalues of the Hermitian H = L^dagger L, so first-order
  // perturbation gives d(sigma_i^2) = w_i^dagger (dL^dagger L + L^dagger dL) w_i for the
  // normalized eigenvector w_i — no singular-vector pair needed, and the
  // expression stays valid for the non-normal signed operator. The denominator
  // is tr(H), whose derivative is the trace of the same perturbation. Quotient
  // rule over the two, times n.
  //
  // COMPLEX throughout (#746): laplacianGradient is already complex, and the
  // return follows the same convention as the period-gap family,
  //   g = dr/d(Re l^2) - i dr/d(Im l^2),
  // so Re(g) and -Im(g) are the two directional derivatives.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  const ChainComplex chain = ChainComplex::fromSpacetime(*spacetime);
  const std::vector<std::vector<std::uint64_t>> oneCells = chain.kSimplexVertices(1);
  std::vector<complexd> gradient(oneCells.size(), complexd(0.0, 0.0));
  if (expectedRegisterCount == 0) return gradient;   // value is the constant 0

  cobordism::HodgeLaplacian laplacian(spacetime, HodgeLaplacian::defaultWeightConvention(), metricSource);
  const std::vector<complexd> flat =
      laplacian.laplacian(registerDegree, /*metric=*/true);
  const std::size_t n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  if (n == 0) return gradient;      // value is the constant expectedRegisterCount
  const Index N = static_cast<Index>(n);
  MatrixXcd L(N, N);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      L(static_cast<Index>(i), static_cast<Index>(j)) = flat[i * n + j];
  if (!L.allFinite()) return gradient;   // the value is +inf here (#699); no slope

  // H = L^dagger L is Hermitian positive semi-definite; its eigenvalues ARE the
  // sigma^2 the value reads, ascending here, and its eigenvectors give the exact
  // first-order response of each one.
  const MatrixXcd H = L.adjoint() * L;
  Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(H);
  const Eigen::VectorXd eigenvalues = solver.eigenvalues();      // ascending
  const MatrixXcd eigenvectors = solver.eigenvectors();
  double total = 0.0;
  for (Index i = 0; i < eigenvalues.size(); ++i) total += eigenvalues[i];
  if (total <= 0.0) return gradient;    // L identically zero: value is 0, flat
  const std::size_t m = std::min(expectedRegisterCount, n);
  double smallest = 0.0;
  for (std::size_t i = 0; i < m; ++i)
    smallest += eigenvalues[static_cast<Index>(i)];              // m smallest

  for (std::size_t edgeIndex = 0; edgeIndex < oneCells.size(); ++edgeIndex) {
    const std::vector<complexd> derivativeFlat = laplacian.laplacianGradient(
        registerDegree, oneCells[edgeIndex][0], oneCells[edgeIndex][1]);
    if (derivativeFlat.empty()) continue;
    MatrixXcd dL(N, N);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        dL(static_cast<Index>(i), static_cast<Index>(j)) =
            derivativeFlat[i * n + j];
    // L is HOLOMORPHIC in l^2 (the weights are polynomial in it), so the
    // complex derivative of H = L^dagger L is 2 L^dagger dL — NOT
    // dL^dagger L + L^dagger dL, which is the Hermitian combination and
    // therefore the REAL-direction derivative alone. Using the Hermitian form
    // makes w^dagger dH w real by construction and the gradient's imaginary
    // part identically zero, which is the whole thing this was meant to fix
    // (#746/#748). Re(2 L^dagger dL) reproduces the Hermitian value exactly, so
    // the real direction is unchanged.
    const MatrixXcd dH = 2.0 * (L.adjoint() * dL);
    // d(sum of the m smallest) and d(trace) from the same perturbation.
    complexd dSmallest(0.0, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      const auto w = eigenvectors.col(static_cast<Index>(i));
      dSmallest += w.dot(dH * w);      // w^dagger dH w, as a scalar
    }
    const complexd dTotal = dH.trace();
    gradient[edgeIndex] = static_cast<double>(n) *
                          (dSmallest * total - smallest * dTotal) /
                          (total * total);
  }
  return gradient;
}

double MultiCobordism::singularValueHalfSumRatio(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    HodgeLaplacian::MetricSource metricSource) {
  cobordism::HodgeLaplacian laplacian(spacetime, HodgeLaplacian::defaultWeightConvention(), metricSource);
  // The SAME operator nearKernelResidual reads (metric, signed, generally
  // non-normal — see its comment); the two terms are alternatives for the one
  // whole-complex slot in rU, so they must see the same spectrum.
  const std::vector<std::complex<double>> flat =
      laplacian.laplacian(registerDegree, /*metric=*/true);
  const std::size_t n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  // No k-cells: the worst case on the [0, 1] scale. Returning the perfect 0
  // here would reward deleting every k-cell over collapsing the spectrum.
  if (n == 0) return 1.0;
  const std::size_t h = n / 2;
  if (h == 0) return 0.0;  // a single mode: no pair of halves to compare
  Eigen::MatrixXcd L(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      L(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * n + j];
  // Total over EVERY configuration (#699): +inf, as nearKernelResidual.
  if (!L.allFinite()) return std::numeric_limits<double>::infinity();
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(L);
  const Eigen::VectorXd sigma = svd.singularValues();  // descending
  double upperHalfSum = 0.0;
  double lowerHalfSum = 0.0;
  for (std::size_t i = 0; i < h; ++i) {
    upperHalfSum += sigma[static_cast<Eigen::Index>(i)];
    lowerHalfSum += sigma[static_cast<Eigen::Index>(n - h + i)];
  }
  // L identically zero: every mode is kernel — nothing left to collapse.
  if (upperHalfSum <= 0.0) return 0.0;
  // Each lower-half value is bounded by its upper-half counterpart (descending
  // order), so the ratio lives in [0, 1]; and L is homogeneous of degree −1 in
  // l^2, so a uniform rescale scales every sigma alike and cancels — degree 0,
  // the same closed conformal-inflation channel as nearKernelResidual.
  return lowerHalfSum / upperHalfSum;
}

double MultiCobordism::hodgeEntropy() const {
  // Over the HODGE degrees, which are what the entropy is taken at. Reported
  // unweighted: the per-degree weights balance the stationarity residuals
  // against each other in the objective, and applying them to entropy VALUES
  // would report a number that is not any degree's entropy.
  double entropy = 0.0;
  for (int degree : hodgeDegrees_)
    entropy += HodgeLaplacian(spacetime_, HodgeLaplacian::defaultWeightConvention(), metricSource_).spectralEntropy(
        degree, hodgeEntropyPhaseMode_);
  return entropy;
}

double MultiCobordism::hodgeEntropyStationarity() const {
  // The entropy half of the objective, so it must read the same degrees and
  // the same weights the objective does. Reading the register degrees here
  // while the term reads the Hodge degrees would let an observation disagree
  // silently with the quantity being descended.
  //
  // Accumulated in the term's order — weighted norms summed, the entropy
  // weight applied by the caller — so `hodgeEntropyWeight() * this` reproduces
  // `ObjectiveTerms::hodgeStationarity` exactly.
  double residual = 0.0;
  for (std::size_t index = 0; index < hodgeDegrees_.size(); ++index) {
    const double weight = index < hodgeDegreeWeights_.size()
                              ? hodgeDegreeWeights_[index]
                              : 1.0;
    residual += weight * HodgeLaplacian(spacetime_, HodgeLaplacian::defaultWeightConvention(), metricSource_).spectralEntropyGradientNorm(
                             hodgeDegrees_[index], hodgeEntropyPhaseMode_);
  }
  return residual;
}

void MultiCobordism::requireObjectiveAcceptable(
    const std::shared_ptr<CobordismObjective> &objective) const {
  // The objective's own DECLARED domain, enforced here so the restriction
  // travels with the objective that declares it rather than living in the
  // engine as a special case. It is a declaration, not a capability limit.
  const int minimumDegree = objective->minimumRegisterDegree();
  for (int degree : registerDegrees_)
    if (degree < minimumDegree)
      throw std::invalid_argument(
          "MultiCobordism: objective '" + objective->name() +
          "' is declared over degrees >= " + std::to_string(minimumDegree) +
          "; got degree " + std::to_string(degree));
  // A scope can only carry a handle minted by `regionHandle`, which refuses an
  // undeclared name, so a mis-spelling cannot reach this point. Re-check
  // anyway: regions can be cleared after a handle was minted, and an objective
  // pointing at a region that no longer exists must fail loudly rather than
  // score nothing.
  const auto scope = objective->scope();
  if (scope.isWholeCobordism()) return;
  for (const auto &region : pinnedRegions_)
    if (region.name == scope.region.name()) return;
  throw std::invalid_argument(
      "MultiCobordism: the injected objective is scoped to pinned region "
      "'" + scope.region.name() + "', which is not declared");
}

void MultiCobordism::setObjective(
    std::shared_ptr<CobordismObjective> objective) {
  if (!objective)
    throw std::invalid_argument(
        "MultiCobordism: the injected objective must not be null");
  requireObjectiveAcceptable(objective);
  objectiveSpec_ = std::move(objective);
}

void MultiCobordism::setPinnedObjective(
    std::shared_ptr<CobordismObjective> objective) {
  if (!objective)
    throw std::invalid_argument(
        "MultiCobordism: the pinned-region objective must not be null");
  requireObjectiveAcceptable(objective);
  pinnedObjectiveSpec_ = std::move(objective);
}

RegionHandle MultiCobordism::regionHandle(const std::string &name) const {
  for (const auto &region : pinnedRegions_)
    if (region.name == name) return RegionHandle(name);
  throw std::invalid_argument(
      "MultiCobordism: no pinned region named '" + name + "' is declared");
}

std::string MultiCobordism::objectiveName() const {
  return objectiveSpec_->name();
}

bool MultiCobordism::compositeSupportsLocalizedDelta() const {
  // A localized delta differences the objective over the cells a move touches.
  // That shortcut is only honest while the scalar being reported is the one
  // being differenced, and with a pinned objective in force the reported scalar
  // is the SUM of two functionals over two different scopes. Differencing the
  // bulk alone would optimize a surrogate that is not the objective — the very
  // thing the localized path exists to avoid — so any pinned objective drops
  // the whole node back to global re-evaluation. Global is always correct,
  // merely more expensive.
  if (pinnedObjectiveSpec_) return false;
  return objectiveSpec_->supportsLocalizedDelta();
}

bool MultiCobordism::objectiveIsTargetConditioned() const {
  // The DISJUNCTION, not the bulk objective's answer. A search policy asks this
  // to find out whether the run it is driving is unforced, and a run whose
  // pinned region is held to a declared state is target-conditioned however
  // geometric the bulk objective is. Reporting the bulk alone would let a
  // policy believe it was unforced while a target steered part of the complex.
  if (pinnedObjectiveSpec_ && pinnedObjectiveSpec_->isTargetConditioned())
    return true;
  return objectiveSpec_->isTargetConditioned();
}

ObjectiveContext MultiCobordism::objectiveContextFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  return objectiveContextFor(spacetime, objectiveSpec_);
}

std::vector<std::size_t> MultiCobordism::scopedEdgeIndices(
    const std::shared_ptr<Spacetime> &spacetime,
    const std::set<std::uint64_t> &region,
    bool includesStraddlingEdges) const {
  std::vector<std::size_t> scored;
  if (!spacetime || region.empty()) return scored;
  const auto edges = spacetime->getEdgeList()->toVector();
  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto key = edges[edgeIndex]->getKey();
    const int endpointsInside = static_cast<int>(region.count(key.first)) +
                                static_cast<int>(region.count(key.second));
    if (endpointsInside == 0) continue;          // wholly bulk
    if (endpointsInside == 2) {                  // interior to the region
      scored.push_back(edgeIndex);
      continue;
    }
    // Exactly one endpoint inside: a straddling edge, in only where the
    // objective declared it so.
    if (includesStraddlingEdges) scored.push_back(edgeIndex);
  }
  return scored;
}

ObjectiveContext MultiCobordism::objectiveContextFor(
    const std::shared_ptr<Spacetime> &spacetime,
    const std::shared_ptr<CobordismObjective> &objective) const {
  // THE FIREWALL. Everything an objective can see is assembled here and
  // nowhere else, and every field of it is PLAIN DATA: the complex, the region
  // and its declared targets, the configured weights, and two precomputed
  // geometric scalars. Deliberately no `std::function`, because a bound
  // callable would capture `this` and hand the objective a route back into the
  // node — which is exactly the reachability the former `static objectiveOf`
  // denied by having no `this` at all. An objective therefore cannot consult a
  // component, fiber, transport, amplitude, colour, particle, charge, flavour,
  // exchange, spin certificate or verdict: it is handed nothing that leads
  // there.
  ObjectiveContext context;
  context.spacetime = spacetime;
  // The objective's DECLARED scope, resolved here. Declaring nothing means the
  // whole cobordism, which leaves `region` and `scoredEdges` empty and every
  // sum running over every coordinate exactly as it did before scopes existed.
  const ObjectiveScope scope = objective ? objective->scope() : ObjectiveScope{};
  context.region = {};
  if (!scope.isWholeCobordism()) {
    for (const auto &region : pinnedRegions_)
      if (region.name == scope.region.name()) {
        context.region = region.vertices;
        break;
      }
    // Present, even when empty: a region with no scored coordinate scores
    // nothing, which is not the whole cobordism.
    context.scoredEdges = scopedEdgeIndices(spacetime, context.region,
                                            scope.includesStraddlingEdges);
  }
  for (const auto &block : inputBlocks_) context.regionTargets.push_back(block.target);
  for (const auto &block : outputBlocks_) context.regionTargets.push_back(block.target);
  context.registerDegrees = registerDegrees_;
  // Declared independently and never inherited from the register degrees: the
  // degrees a register is constructed at and the degrees whose entropy should
  // be stationary are different questions, and defaulting one to the other
  // would reinstate that coupling in the implementation.
  context.hodgeDegrees = hodgeDegrees_;
  context.hodgeDegreeWeights = hodgeDegreeWeights_;
  context.reggeWeight = reggeWeight_;
  context.hodgeEntropyWeight = hodgeEntropyWeight_;
  context.connectionEntropyWeight = connectionEntropyWeight_;
  context.gamma = gamma_;
  context.carriedStateEnergyWeight = carriedStateEnergyWeight_;
  context.einsteinHilbert = einsteinHilbert_;
  context.hodgeEntropyPhaseMode = hodgeEntropyPhaseMode_;
  // Computed only where the objective declares it reads them, so a purely
  // geometric objective never pays for a target-conditioned quantity. Left NaN
  // rather than zero where not computed: unmeasured is not "measured zero".
  if (spacetime && objective && objective->needsRegisterResidual())
    context.registerResidual = rU(spacetime);
  if (spacetime && carriedStateEnergyWeight_ != 0.0)
    context.carriedStateEnergy = carriedStateEnergy(spacetime);
  return context;
}

void MultiCobordism::setHodgeEntropyWeight(double weight) {
  if (!std::isfinite(weight) || weight < 0.0)
    throw std::invalid_argument(
        "MultiCobordism: Hodge entropy weight must be finite and non-negative");
  hodgeEntropyWeight_ = weight;
}

void MultiCobordism::setConnectionEntropyWeight(double weight) {
  if (!std::isfinite(weight) || weight < 0.0)
    throw std::invalid_argument(
        "MultiCobordism: connection entropy weight must be finite and "
        "non-negative");
  connectionEntropyWeight_ = weight;
}

void MultiCobordism::setReggeWeight(double weight) {
  if (!std::isfinite(weight) || weight < 0.0)
    throw std::invalid_argument(
        "MultiCobordism: Regge weight must be finite and non-negative");
  reggeWeight_ = weight;
}

void MultiCobordism::setHodgeDegrees(std::vector<int> degrees,
                                     std::vector<double> weights) {
  if (degrees.empty())
    throw std::invalid_argument(
        "MultiCobordism: the Hodge degree list must name at least one degree; "
        "an empty list would silently score no entropy at all");
  for (int degree : degrees)
    if (degree < 0)
      throw std::invalid_argument(
          "MultiCobordism: Hodge degree " + std::to_string(degree) +
          " is negative; a degree indexes a Laplacian L_k with k >= 0");
  // A repeat would double-count that degree's share while reading as a list of
  // distinct degrees, so it is refused rather than silently summed twice.
  for (std::size_t outer = 0; outer < degrees.size(); ++outer)
    for (std::size_t inner = outer + 1; inner < degrees.size(); ++inner)
      if (degrees[outer] == degrees[inner])
        throw std::invalid_argument(
            "MultiCobordism: Hodge degree " + std::to_string(degrees[outer]) +
            " is listed more than once");
  if (!weights.empty()) {
    if (weights.size() != degrees.size())
      throw std::invalid_argument(
          "MultiCobordism: " + std::to_string(weights.size()) +
          " Hodge degree weights were given for " +
          std::to_string(degrees.size()) +
          " degrees; supply one weight per degree, or none for uniform");
    for (double weight : weights)
      if (!std::isfinite(weight) || weight < 0.0)
        throw std::invalid_argument(
            "MultiCobordism: a Hodge degree weight must be finite and "
            "non-negative");
  }
  hodgeDegrees_ = std::move(degrees);
  hodgeDegreeWeights_ = std::move(weights);
}

std::vector<HodgeDegreeContribution>
MultiCobordism::hodgeDegreeContributionsFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  // Read from the objective that owns the term, so a reported share is the one
  // that was descended rather than a second computation that could disagree.
  if (!objectiveSpec_) return {};
  return objectiveSpec_->hodgeDegreeContributions(
      objectiveContextFor(spacetime, objectiveSpec_));
}

std::vector<HodgeDegreeContribution> MultiCobordism::hodgeDegreeContributions()
    const {
  return hodgeDegreeContributionsFor(spacetime_);
}

std::vector<std::string> MultiCobordism::objectiveTermNames() {
  // The declaration order of `ObjectiveTerms`. Enumerated as data so the
  // no-feedback firewall is CHECKABLE rather than asserted in a comment: a
  // test reads this list and confirms no particle, fiber, transport, or
  // amplitude quantity is on it. Every objective records into these same
  // slots, so a record stays comparable across objectives.
  return CobordismObjective::declaredTermNames();
}

double MultiCobordism::objectiveOf(const ObjectiveTerms &terms) {
  // STATIC: no `this`, so the scalar the optimizer descends provably depends
  // on nothing but the declared terms.
  return CobordismObjective::total(terms);
}

std::vector<MultiCobordism::ObjectiveContribution>
MultiCobordism::objectiveContributionsFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  // One contribution per objective, in evaluation order, so a reader can tell
  // whether descent came from the bulk or from the pinned region. Summing them
  // reproduces `objectiveTermsFor` exactly.
  std::vector<ObjectiveContribution> contributions;
  const auto record = [&](const std::shared_ptr<CobordismObjective> &objective) {
    if (!objective) return;
    contributions.push_back(
        {objective->name(), objective->scope().region.name(),
         objective->terms(objectiveContextFor(spacetime, objective))});
  };
  record(objectiveSpec_);
  record(pinnedObjectiveSpec_);
  return contributions;
}

std::vector<MultiCobordism::ObjectiveContribution>
MultiCobordism::objectiveContributions() const {
  return objectiveContributionsFor(spacetime_);
}

MultiCobordism::ObjectiveTerms MultiCobordism::objectiveTermsFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  // The engine no longer knows which functional it is scoring: it assembles
  // the firewalled context and the injected objective decomposes itself.
  ObjectiveTerms terms = objectiveSpec_->terms(objectiveContextFor(spacetime));
  // A pinned-region objective ADDS its terms on top. The bulk objective keeps
  // scoring the entire cobordism INCLUDING the pinned interior, so a
  // boundary-interior edge contributes to both — additive by design rather than
  // double-counting to be corrected, because the bulk sees one coherent
  // cobordism and this is an additional hold on part of it.
  if (pinnedObjectiveSpec_) {
    const auto pinned = pinnedObjectiveSpec_->terms(
        objectiveContextFor(spacetime, pinnedObjectiveSpec_));
    terms.reggeStationarity += pinned.reggeStationarity;
    terms.hodgeStationarity += pinned.hodgeStationarity;
    terms.registerResidual += pinned.registerResidual;
    terms.actionMagnitude += pinned.actionMagnitude;
    terms.carriedStateEnergy += pinned.carriedStateEnergy;
  }
  return terms;
}

MultiCobordism::ObjectiveTerms MultiCobordism::objectiveTerms() const {
  return objectiveTermsFor(spacetime_);
}

double MultiCobordism::objectiveFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  if (!spacetime) return std::numeric_limits<double>::infinity();
  return objectiveOf(objectiveTermsFor(spacetime));
}

double MultiCobordism::objective() const { return objectiveFor(spacetime_); }

void MultiCobordism::declarePinnedRegion(PinnedRegion region) {
  for (auto &existing : pinnedRegions_)
    if (existing.name == region.name) {
      existing = std::move(region);
      return;
    }
  pinnedRegions_.push_back(std::move(region));
}

void MultiCobordism::clearPinnedRegions() { pinnedRegions_.clear(); }

void MultiCobordism::declareRegisterConstraint(RegisterConstraint constraint) {
  if (constraint.name.empty())
    throw std::invalid_argument(
        "MultiCobordism::declareRegisterConstraint: name is empty");
  if (constraint.degree < 0)
    throw std::invalid_argument(
        "MultiCobordism::declareRegisterConstraint: degree is negative");
  if (constraint.holes.size() != constraint.target.size())
    throw std::invalid_argument(
        "MultiCobordism::declareRegisterConstraint: hole/target size mismatch");
  const std::size_t expectedWidth =
      static_cast<std::size_t>(constraint.degree) + 2;
  for (auto &hole : constraint.holes) {
    if (hole.size() != expectedWidth)
      throw std::invalid_argument(
          "MultiCobordism::declareRegisterConstraint: malformed hole");
    std::sort(hole.begin(), hole.end());
    if (std::adjacent_find(hole.begin(), hole.end()) != hole.end())
      throw std::invalid_argument(
          "MultiCobordism::declareRegisterConstraint: repeated hole vertex");
  }
  for (auto &existing : registerConstraints_)
    if (existing.name == constraint.name) {
      existing = std::move(constraint);
      return;
    }
  registerConstraints_.push_back(std::move(constraint));
}

void MultiCobordism::clearRegisterConstraints() {
  registerConstraints_.clear();
}

MultiCobordism::FixedBoundaryEigenstateResult
MultiCobordism::relaxFixedBoundaryEigenstate(
    int degree, std::vector<std::vector<std::uint64_t>> supportCells,
    std::vector<complexd> target, double epsilon, int restarts, int maxGrowth,
    std::uint64_t seed, int maxIterations) {
  if (!spacetime_)
    throw std::invalid_argument(
        "MultiCobordism::relaxFixedBoundaryEigenstate: null spacetime");
  if (degree < 0)
    throw std::invalid_argument(
        "MultiCobordism::relaxFixedBoundaryEigenstate: degree is negative");
  if (supportCells.empty() || supportCells.size() != target.size())
    throw std::invalid_argument(
        "MultiCobordism::relaxFixedBoundaryEigenstate: support/target size "
        "mismatch or empty support");
  if (!(epsilon > 0.0) || !std::isfinite(epsilon))
    throw std::invalid_argument(
        "MultiCobordism::relaxFixedBoundaryEigenstate: epsilon must be finite "
        "and positive");
  if (restarts <= 0 || maxGrowth < 0 || maxIterations <= 0)
    throw std::invalid_argument(
        "MultiCobordism::relaxFixedBoundaryEigenstate: invalid search budget");

  const std::size_t expectedCellWidth = static_cast<std::size_t>(degree) + 1;
  std::set<std::vector<std::uint64_t>> uniqueSupport;
  for (auto &cell : supportCells) {
    if (cell.size() != expectedCellWidth)
      throw std::invalid_argument(
          "MultiCobordism::relaxFixedBoundaryEigenstate: malformed support "
          "cell");
    std::sort(cell.begin(), cell.end());
    if (std::adjacent_find(cell.begin(), cell.end()) != cell.end() ||
        !uniqueSupport.insert(cell).second)
      throw std::invalid_argument(
          "MultiCobordism::relaxFixedBoundaryEigenstate: repeated support "
          "cell");
  }

  double targetNormSquared = 0.0;
  for (const complexd value : target) targetNormSquared += std::norm(value);
  if (!(targetNormSquared > 0.0) || !std::isfinite(targetNormSquared))
    throw std::invalid_argument(
        "MultiCobordism::relaxFixedBoundaryEigenstate: target must be finite "
        "and nonzero");
  const double inverseTargetNorm = 1.0 / std::sqrt(targetNormSquared);
  for (complexd &value : target) value *= inverseTargetNorm;

  std::map<std::vector<std::uint64_t>, complexd> pinnedByCell;
  for (std::size_t index = 0; index < supportCells.size(); ++index)
    pinnedByCell.emplace(supportCells[index], target[index]);

  EigenstateSynthesis synthesis(spacetime_, degree, metricSource_);
  FixedCochainOptimization best;
  int growthSteps = 0;
  for (int pass = 0;; ++pass) {
    const auto interiorEdges = synthesis.interiorEdges();
    const std::set<std::pair<std::uint64_t, std::uint64_t>> freeEdgeKeys(
        interiorEdges.begin(), interiorEdges.end());
    best = optimizeFixedCochainTargets(
        spacetime_, synthesis, {pinnedByCell},
        [&freeEdgeKeys](std::uint64_t a, std::uint64_t b) {
          return freeEdgeKeys.count({std::min(a, b), std::max(a, b)}) != 0;
        },
        false, epsilon, restarts, seed + static_cast<std::uint64_t>(pass),
        maxIterations);
    if (best.residual < epsilon || growthSteps >= maxGrowth) break;
    if (!synthesis.growInterior(seed + 1000u +
                                static_cast<std::uint64_t>(pass)))
      break;
    ++growthSteps;
  }

  FixedBoundaryEigenstateResult result;
  result.converged = best.residual < epsilon;
  result.residual = best.residual;
  result.state = normalized(std::move(best.states.front()));
  result.eigenvalue = synthesis.rayleigh(result.state);
  result.degree = degree;
  result.growthSteps = growthSteps;
  result.interiorVertexCount = synthesis.interiorVertexCount();
  result.interiorEdgeCount = synthesis.numInteriorEdges();
  result.auxiliaryCellCount = best.auxiliaryCellCount;
  result.supportCells = std::move(supportCells);
  result.target = std::move(target);
  return result;
}

MultiCobordism::BoundaryStateTransferResult
MultiCobordism::relaxBoundaryStatePairs(
    int degree, std::string inputRegionName,
    std::vector<std::vector<std::uint64_t>> inputCells,
    std::vector<std::vector<complexd>> inputStates,
    std::string outputRegionName,
    std::vector<std::vector<std::uint64_t>> outputCells,
    std::vector<std::vector<complexd>> outputStates, bool commonEigenvalue,
    double epsilon, double boundaryEpsilon, int restarts, int maxGrowth,
    std::uint64_t seed, int maxIterations) {
  const std::string prefix =
      "MultiCobordism::relaxBoundaryStatePairs: ";
  if (!spacetime_)
    throw std::invalid_argument(prefix + "null spacetime");
  if (degree < 0 || degree >= spacetime_->getDimensions())
    throw std::invalid_argument(
        prefix + "degree must index a cell of the boundary");
  if (inputRegionName.empty() || outputRegionName.empty() ||
      inputRegionName == outputRegionName)
    throw std::invalid_argument(
        prefix + "input and output region names must be distinct and nonempty");
  if (inputStates.empty() || inputStates.size() != outputStates.size())
    throw std::invalid_argument(
        prefix + "input/output state-pair count mismatch or empty states");
  if (!(epsilon > 0.0) || !std::isfinite(epsilon) ||
      !(boundaryEpsilon > 0.0) || !std::isfinite(boundaryEpsilon))
    throw std::invalid_argument(
        prefix + "residual tolerances must be finite and positive");
  if (restarts <= 0 || maxGrowth < 0 || maxIterations <= 0)
    throw std::invalid_argument(prefix + "invalid search budget");

  const PinnedRegion *inputRegion = nullptr;
  const PinnedRegion *outputRegion = nullptr;
  for (const auto &region : pinnedRegions_) {
    if (region.name == inputRegionName) inputRegion = &region;
    if (region.name == outputRegionName) outputRegion = &region;
  }
  if (!inputRegion || !outputRegion)
    throw std::invalid_argument(
        prefix + "both named pinned regions must be declared");

  const auto components = boundaryComponents(*spacetime_);
  if (components.size() != 2)
    throw std::invalid_argument(
        prefix + "the live cobordism boundary must have exactly two "
                 "connected components");
  const BoundaryComponentData *inputComponent = nullptr;
  const BoundaryComponentData *outputComponent = nullptr;
  for (const auto &component : components) {
    if (component.vertices == inputRegion->vertices)
      inputComponent = &component;
    if (component.vertices == outputRegion->vertices)
      outputComponent = &component;
  }
  if (!inputComponent || !outputComponent ||
      inputComponent == outputComponent)
    throw std::invalid_argument(
        prefix + "each named region must equal one distinct boundary "
                 "component");

  const std::size_t cellWidth = static_cast<std::size_t>(degree) + 1;
  const auto canonicalizeFrame =
      [&prefix, cellWidth](std::vector<Cell> &frame,
                           const std::string &label) {
        std::set<Cell> unique;
        for (auto &cell : frame) {
          if (cell.size() != cellWidth)
            throw std::invalid_argument(
                prefix + label + " contains a malformed cell");
          std::sort(cell.begin(), cell.end());
          if (std::adjacent_find(cell.begin(), cell.end()) != cell.end() ||
              !unique.insert(cell).second)
            throw std::invalid_argument(
                prefix + label + " contains a repeated cell");
        }
        return unique;
      };
  const auto suppliedInputCells =
      canonicalizeFrame(inputCells, "input cell frame");
  const auto suppliedOutputCells =
      canonicalizeFrame(outputCells, "output cell frame");
  const auto expectedInputCells = componentCells(*inputComponent, degree);
  const auto expectedOutputCells = componentCells(*outputComponent, degree);
  if (suppliedInputCells.empty() ||
      suppliedInputCells != expectedInputCells)
    throw std::invalid_argument(
        prefix + "input cell frame must enumerate the complete degree-k "
                 "boundary component");
  if (suppliedOutputCells.empty() ||
      suppliedOutputCells != expectedOutputCells)
    throw std::invalid_argument(
        prefix + "output cell frame must enumerate the complete degree-k "
                 "boundary component");

  for (std::size_t stateIndex = 0; stateIndex < inputStates.size();
       ++stateIndex) {
    if (inputStates[stateIndex].size() != inputCells.size())
      throw std::invalid_argument(
          prefix + "input state width does not match its cell frame");
    if (outputStates[stateIndex].size() != outputCells.size())
      throw std::invalid_argument(
          prefix + "output state width does not match its cell frame");
    double inputNormSquared = 0.0;
    double outputNormSquared = 0.0;
    for (const complexd value : inputStates[stateIndex]) {
      if (!finiteComplex(value))
        throw std::invalid_argument(
            prefix + "input state must contain only finite amplitudes");
      inputNormSquared += std::norm(value);
    }
    for (const complexd value : outputStates[stateIndex]) {
      if (!finiteComplex(value))
        throw std::invalid_argument(
            prefix + "output state must contain only finite amplitudes");
      outputNormSquared += std::norm(value);
    }
    if (!(inputNormSquared > 0.0) || !std::isfinite(inputNormSquared))
      throw std::invalid_argument(
          prefix + "input states must be nonzero");
    if (!(outputNormSquared > 0.0) || !std::isfinite(outputNormSquared))
      throw std::invalid_argument(
          prefix + "output states must be nonzero");
    const double inverseInputNorm = 1.0 / std::sqrt(inputNormSquared);
    for (complexd &value : inputStates[stateIndex])
      value *= inverseInputNorm;
    for (complexd &value : outputStates[stateIndex])
      value *= inverseInputNorm;
  }

  const auto inputBoundary = evaluateBoundaryStates(
      spacetime_, *inputComponent, degree, inputCells, inputStates, metricSource_);
  const auto outputBoundary = evaluateBoundaryStates(
      spacetime_, *outputComponent, degree, outputCells, outputStates, metricSource_);
  for (const double residual : inputBoundary.residuals)
    if (!(residual < boundaryEpsilon))
      throw std::invalid_argument(
          prefix + "an input state is not an isolated-boundary eigenstate");
  for (const double residual : outputBoundary.residuals)
    if (!(residual < boundaryEpsilon))
      throw std::invalid_argument(
          prefix + "an output state is not an isolated-boundary eigenstate");

  EigenstateSynthesis synthesis(spacetime_, degree, metricSource_);
  for (const auto &[a, b] : synthesis.boundaryEdges())
    if (!edgeIsPinned(a, b))
      throw std::invalid_argument(
          prefix + "every geometric boundary edge must be held by a declared "
                   "pinned region");

  using Geometry = std::pair<complexd, complexd>;
  std::map<std::pair<std::uint64_t, std::uint64_t>, Geometry> pinnedGeometry;
  for (auto *edge : spacetime_->getEdgeList()->toVector()) {
    const auto key = edgeKey(edge);
    if (edgeIsPinned(key.first, key.second))
      pinnedGeometry.emplace(
          key, Geometry{edge->getLength(), edge->getPhase()});
  }

  std::vector<FixedCochainTarget> fixedTargets(inputStates.size());
  for (std::size_t stateIndex = 0; stateIndex < inputStates.size();
       ++stateIndex) {
    for (std::size_t cellIndex = 0; cellIndex < inputCells.size();
         ++cellIndex)
      fixedTargets[stateIndex].emplace(
          inputCells[cellIndex], inputStates[stateIndex][cellIndex]);
    for (std::size_t cellIndex = 0; cellIndex < outputCells.size();
         ++cellIndex)
      fixedTargets[stateIndex].emplace(
          outputCells[cellIndex], outputStates[stateIndex][cellIndex]);
  }

  FixedCochainOptimization best;
  std::vector<double> residualTrace;
  int growthSteps = 0;
  for (int pass = 0;; ++pass) {
    best = optimizeFixedCochainTargets(
        spacetime_, synthesis, fixedTargets,
        [this](std::uint64_t a, std::uint64_t b) {
          return !edgeIsPinned(a, b);
        },
        commonEigenvalue, epsilon, restarts,
        seed + static_cast<std::uint64_t>(pass), maxIterations);
    residualTrace.push_back(best.residual);
    if (best.residual < epsilon || growthSteps >= maxGrowth) break;
    if (!synthesis.growInterior(seed + 1000u +
                                static_cast<std::uint64_t>(pass)))
      break;
    ++growthSteps;
  }

  std::map<std::pair<std::uint64_t, std::uint64_t>, Geometry>
      finalPinnedGeometry;
  for (auto *edge : spacetime_->getEdgeList()->toVector()) {
    const auto key = edgeKey(edge);
    if (edgeIsPinned(key.first, key.second))
      finalPinnedGeometry.emplace(
          key, Geometry{edge->getLength(), edge->getPhase()});
  }
  if (finalPinnedGeometry != pinnedGeometry)
    throw std::logic_error(
        prefix + "pinned geometry changed during boundary-state relaxation");

  const auto finalInputBoundary = evaluateBoundaryStates(
      spacetime_, *inputComponent, degree, inputCells, inputStates, metricSource_);
  const auto finalOutputBoundary = evaluateBoundaryStates(
      spacetime_, *outputComponent, degree, outputCells, outputStates, metricSource_);

  BoundaryStateTransferResult result;
  result.converged = best.residual < epsilon;
  result.commonEigenvalue = commonEigenvalue;
  result.residual = best.residual;
  result.eigenvalue = best.eigenvalue;
  result.degree = degree;
  result.growthSteps = growthSteps;
  result.freeEdgeCount = best.freeEdgeCount;
  result.auxiliaryCellCount = best.auxiliaryCellCount;
  result.inputRegion = std::move(inputRegionName);
  result.outputRegion = std::move(outputRegionName);
  result.inputCells = std::move(inputCells);
  result.outputCells = std::move(outputCells);
  result.inputStates = std::move(inputStates);
  result.outputStates = std::move(outputStates);
  result.states = std::move(best.states);
  result.stateResiduals = std::move(best.stateResiduals);
  result.stateEigenvalues = std::move(best.stateEigenvalues);
  result.inputBoundaryResiduals = finalInputBoundary.residuals;
  result.outputBoundaryResiduals = finalOutputBoundary.residuals;
  result.residualTrace = std::move(residualTrace);
  return result;
}

MultiCobordism::WholeComplexReadoutResult
MultiCobordism::relaxWholeComplexReadoutTargets(
    int degree, std::string regionAName,
    std::vector<std::vector<std::uint64_t>> cellsA,
    std::vector<std::vector<complexd>> statesA, std::string regionBName,
    std::vector<std::vector<std::uint64_t>> cellsB,
    std::vector<std::vector<complexd>> statesB,
    std::vector<ReadoutChain> readouts,
    std::vector<std::vector<complexd>> targets, bool commonEigenvalue,
    double epsilon, double boundaryEpsilon, int restarts, int maxGrowth,
    std::uint64_t seed, int maxIterations) {
  const std::string prefix =
      "MultiCobordism::relaxWholeComplexReadoutTargets: ";
  if (!spacetime_)
    throw std::invalid_argument(prefix + "null spacetime");
  if (degree < 0 || degree >= spacetime_->getDimensions())
    throw std::invalid_argument(
        prefix + "degree must index a cell of the boundary");
  if (regionAName.empty() || regionBName.empty() ||
      regionAName == regionBName)
    throw std::invalid_argument(
        prefix + "the two region names must be distinct and nonempty");
  if (statesA.empty() || statesA.size() != statesB.size() ||
      statesA.size() != targets.size())
    throw std::invalid_argument(
        prefix + "witness count mismatch between the two boundary states and "
                 "the readout targets, or no witnesses");
  if (readouts.empty())
    throw std::invalid_argument(
        prefix + "at least one readout chain is required");
  if (!(epsilon > 0.0) || !std::isfinite(epsilon) ||
      !(boundaryEpsilon > 0.0) || !std::isfinite(boundaryEpsilon))
    throw std::invalid_argument(
        prefix + "residual tolerances must be finite and positive");
  if (restarts <= 0 || maxGrowth < 0 || maxIterations <= 0)
    throw std::invalid_argument(prefix + "invalid search budget");

  const PinnedRegion *regionA = nullptr;
  const PinnedRegion *regionB = nullptr;
  for (const auto &region : pinnedRegions_) {
    if (region.name == regionAName) regionA = &region;
    if (region.name == regionBName) regionB = &region;
  }
  if (!regionA || !regionB)
    throw std::invalid_argument(
        prefix + "both named pinned regions must be declared");

  const auto components = boundaryComponents(*spacetime_);
  if (components.size() != 2)
    throw std::invalid_argument(
        prefix + "the live cobordism boundary must have exactly two "
                 "connected components");
  const BoundaryComponentData *componentA = nullptr;
  const BoundaryComponentData *componentB = nullptr;
  for (const auto &component : components) {
    if (component.vertices == regionA->vertices) componentA = &component;
    if (component.vertices == regionB->vertices) componentB = &component;
  }
  if (!componentA || !componentB || componentA == componentB)
    throw std::invalid_argument(
        prefix + "each named region must equal one distinct boundary "
                 "component");

  const std::size_t cellWidth = static_cast<std::size_t>(degree) + 1;
  const auto canonicalizeCell = [&prefix, cellWidth](Cell &cell,
                                                     const std::string &label) {
    if (cell.size() != cellWidth)
      throw std::invalid_argument(prefix + label + " contains a malformed cell");
    std::sort(cell.begin(), cell.end());
    if (std::adjacent_find(cell.begin(), cell.end()) != cell.end())
      throw std::invalid_argument(prefix + label + " contains a malformed cell");
  };
  const auto canonicalizeFrame = [&prefix, &canonicalizeCell](
                                     std::vector<Cell> &frame,
                                     const std::string &label) {
    std::set<Cell> unique;
    for (auto &cell : frame) {
      canonicalizeCell(cell, label);
      if (!unique.insert(cell).second)
        throw std::invalid_argument(prefix + label +
                                    " contains a repeated cell");
    }
    return unique;
  };
  const auto suppliedCellsA = canonicalizeFrame(cellsA, "cell frame A");
  const auto suppliedCellsB = canonicalizeFrame(cellsB, "cell frame B");
  if (suppliedCellsA.empty() ||
      suppliedCellsA != componentCells(*componentA, degree))
    throw std::invalid_argument(
        prefix + "cell frame A must enumerate the complete degree-k boundary "
                 "component");
  if (suppliedCellsB.empty() ||
      suppliedCellsB != componentCells(*componentB, degree))
    throw std::invalid_argument(
        prefix + "cell frame B must enumerate the complete degree-k boundary "
                 "component");

  for (std::size_t row = 0; row < readouts.size(); ++row) {
    if (readouts[row].empty())
      throw std::invalid_argument(prefix + "readout chain " +
                                  std::to_string(row) + " is empty");
    std::set<Cell> seen;
    for (auto &[cell, coefficient] : readouts[row]) {
      canonicalizeCell(cell, "readout chain " + std::to_string(row));
      if (!seen.insert(cell).second)
        throw std::invalid_argument(prefix + "readout chain " +
                                    std::to_string(row) +
                                    " lists a cell twice");
      if (!finiteComplex(coefficient))
        throw std::invalid_argument(prefix + "readout chain " +
                                    std::to_string(row) +
                                    " has a non-finite coefficient");
    }
  }

  for (std::size_t witness = 0; witness < statesA.size(); ++witness) {
    if (statesA[witness].size() != cellsA.size())
      throw std::invalid_argument(
          prefix + "state A width does not match cell frame A");
    if (statesB[witness].size() != cellsB.size())
      throw std::invalid_argument(
          prefix + "state B width does not match cell frame B");
    if (targets[witness].size() != readouts.size())
      throw std::invalid_argument(
          prefix + "a readout target row does not match the readout chain "
                   "count");
    double jointNormSquared = 0.0;
    for (const complexd value : statesA[witness]) {
      if (!finiteComplex(value))
        throw std::invalid_argument(
            prefix + "boundary states must contain only finite amplitudes");
      jointNormSquared += std::norm(value);
    }
    for (const complexd value : statesB[witness]) {
      if (!finiteComplex(value))
        throw std::invalid_argument(
            prefix + "boundary states must contain only finite amplitudes");
      jointNormSquared += std::norm(value);
    }
    for (const complexd value : targets[witness])
      if (!finiteComplex(value))
        throw std::invalid_argument(
            prefix + "readout targets must contain only finite amplitudes");
    if (!(jointNormSquared > 0.0) || !std::isfinite(jointNormSquared))
      throw std::invalid_argument(
          prefix + "the joint boundary state of a witness must be nonzero");
    const double inverseNorm = 1.0 / std::sqrt(jointNormSquared);
    for (complexd &value : statesA[witness]) value *= inverseNorm;
    for (complexd &value : statesB[witness]) value *= inverseNorm;
    for (complexd &value : targets[witness]) value *= inverseNorm;
  }

  // Isolated-boundary eigenstate check on every NONZERO component
  // restriction; an exactly zero restriction is the zero input.
  const auto isZero = [](const std::vector<complexd> &state) {
    for (const complexd value : state)
      if (value != complexd(0.0, 0.0)) return false;
    return true;
  };
  const auto boundaryResiduals =
      [this, degree, &isZero, &prefix, boundaryEpsilon](
          const BoundaryComponentData &component,
          const std::vector<Cell> &cells,
          const std::vector<std::vector<complexd>> &states,
          const char *label, bool enforce) {
        std::vector<std::vector<complexd>> nonzero;
        std::vector<std::size_t> witnessOf;
        for (std::size_t witness = 0; witness < states.size(); ++witness) {
          if (isZero(states[witness])) continue;
          nonzero.push_back(states[witness]);
          witnessOf.push_back(witness);
        }
        std::vector<double> residuals(states.size(), 0.0);
        if (!nonzero.empty()) {
          const auto evaluation = evaluateBoundaryStates(
              spacetime_, component, degree, cells, nonzero, metricSource_);
          for (std::size_t index = 0; index < witnessOf.size(); ++index) {
            residuals[witnessOf[index]] = evaluation.residuals[index];
            if (enforce && !(evaluation.residuals[index] < boundaryEpsilon))
              throw std::invalid_argument(
                  prefix + "the state of witness " +
                  std::to_string(witnessOf[index]) + " on component " +
                  label + " is not an isolated-boundary eigenstate");
          }
        }
        return residuals;
      };
  (void)boundaryResiduals(*componentA, cellsA, statesA, "A", true);
  (void)boundaryResiduals(*componentB, cellsB, statesB, "B", true);

  EigenstateSynthesis synthesis(spacetime_, degree, metricSource_);
  for (const auto &[a, b] : synthesis.boundaryEdges())
    if (!edgeIsPinned(a, b))
      throw std::invalid_argument(
          prefix + "every geometric boundary edge must be held by a declared "
                   "pinned region");

  using Geometry = std::pair<complexd, complexd>;
  std::map<std::pair<std::uint64_t, std::uint64_t>, Geometry> pinnedGeometry;
  for (auto *edge : spacetime_->getEdgeList()->toVector()) {
    const auto key = edgeKey(edge);
    if (edgeIsPinned(key.first, key.second))
      pinnedGeometry.emplace(key,
                             Geometry{edge->getLength(), edge->getPhase()});
  }

  std::vector<FixedCochainTarget> fixedTargets(statesA.size());
  for (std::size_t witness = 0; witness < statesA.size(); ++witness) {
    for (std::size_t index = 0; index < cellsA.size(); ++index)
      fixedTargets[witness].emplace(cellsA[index], statesA[witness][index]);
    for (std::size_t index = 0; index < cellsB.size(); ++index)
      fixedTargets[witness].emplace(cellsB[index], statesB[witness][index]);
  }
  const ReadoutSystem readoutSystem{&readouts, &targets};

  FixedCochainOptimization best;
  std::vector<double> residualTrace;
  int growthSteps = 0;
  WarmStart warmStart;
  for (int pass = 0;; ++pass) {
    best = optimizeFixedCochainTargets(
        spacetime_, synthesis, fixedTargets,
        [this](std::uint64_t a, std::uint64_t b) {
          return !edgeIsPinned(a, b);
        },
        commonEigenvalue, epsilon, restarts,
        seed + static_cast<std::uint64_t>(pass), maxIterations,
        &readoutSystem, pass == 0 ? nullptr : &warmStart);
    residualTrace.push_back(best.residual);
    if (best.residual < epsilon || growthSteps >= maxGrowth) break;
    // Carry this pass's witnesses (by cell) and the live geometry into the
    // next pass as its first descent; cells created by growth start at zero.
    warmStart.states.assign(best.states.size(), {});
    for (std::size_t witness = 0; witness < best.states.size(); ++witness)
      for (std::size_t index = 0; index < synthesis.order(); ++index)
        warmStart.states[witness].emplace(synthesis.cellSimplices()[index],
                                          best.states[witness][index]);
    if (!synthesis.growInterior(seed + 1000u +
                                static_cast<std::uint64_t>(pass)))
      break;
    ++growthSteps;
  }

  std::map<std::pair<std::uint64_t, std::uint64_t>, Geometry>
      finalPinnedGeometry;
  for (auto *edge : spacetime_->getEdgeList()->toVector()) {
    const auto key = edgeKey(edge);
    if (edgeIsPinned(key.first, key.second))
      finalPinnedGeometry.emplace(
          key, Geometry{edge->getLength(), edge->getPhase()});
  }
  if (finalPinnedGeometry != pinnedGeometry)
    throw std::logic_error(
        prefix + "pinned geometry changed during the readout relaxation");

  // Readouts of the returned (unnormalized) witnesses in the live cell order.
  std::map<Cell, std::size_t> cellIndex;
  for (std::size_t index = 0; index < synthesis.order(); ++index)
    cellIndex.emplace(synthesis.cellSimplices()[index], index);
  std::vector<std::vector<complexd>> measuredReadouts(best.states.size());
  double readoutDeviation = 0.0;
  for (std::size_t witness = 0; witness < best.states.size(); ++witness) {
    measuredReadouts[witness].reserve(readouts.size());
    for (std::size_t row = 0; row < readouts.size(); ++row) {
      complexd value(0.0, 0.0);
      for (const auto &[cell, coefficient] : readouts[row]) {
        const auto found = cellIndex.find(cell);
        if (found == cellIndex.end())
          throw std::logic_error(
              prefix + "a readout chain cell vanished from the live complex");
        value += coefficient * best.states[witness][found->second];
      }
      measuredReadouts[witness].push_back(value);
      readoutDeviation = std::max(
          readoutDeviation, std::abs(value - targets[witness][row]));
    }
  }

  WholeComplexReadoutResult result;
  result.converged = best.residual < epsilon;
  result.commonEigenvalue = commonEigenvalue;
  result.residual = best.residual;
  result.eigenvalue = best.eigenvalue;
  result.degree = degree;
  result.growthSteps = growthSteps;
  result.freeEdgeCount = best.freeEdgeCount;
  result.auxiliaryCellCount = best.auxiliaryCellCount;
  result.readoutRank = best.readoutRank;
  result.regionA = std::move(regionAName);
  result.regionB = std::move(regionBName);
  result.boundaryResidualsA =
      boundaryResiduals(*componentA, cellsA, statesA, "A", false);
  result.boundaryResidualsB =
      boundaryResiduals(*componentB, cellsB, statesB, "B", false);
  result.cellsA = std::move(cellsA);
  result.cellsB = std::move(cellsB);
  result.statesA = std::move(statesA);
  result.statesB = std::move(statesB);
  result.targets = std::move(targets);
  result.readouts = std::move(measuredReadouts);
  result.readoutDeviation = readoutDeviation;
  result.states = std::move(best.states);
  result.stateResiduals = std::move(best.stateResiduals);
  result.stateEigenvalues = std::move(best.stateEigenvalues);
  result.residualTrace = std::move(residualTrace);
  return result;
}

MultiCobordism::GeometricOperatorReadout MultiCobordism::geometricOperator(
    int stateDimension, std::vector<std::vector<std::uint64_t>> frameCells,
    double tol, bool metric) const {
  GeometricOperatorReadout result;
  result.stateDimension = stateDimension;
  result.metric = metric;
  const auto obstruct = [&](std::string reason) {
    result.identifiable = false;
    result.obstruction = std::move(reason);
    result.choiState.clear();
    result.operatorMatrix.clear();
    return result;
  };
  if (!spacetime_)
    return obstruct("the cobordism has no spacetime");
  if (stateDimension <= 0)
    return obstruct("state dimension must be positive");
  if (!(tol > 0.0) || !std::isfinite(tol))
    return obstruct("kernel tolerance must be finite and positive");

  const std::size_t d = static_cast<std::size_t>(stateDimension);
  if (d > std::numeric_limits<std::size_t>::max() / d)
    return obstruct("state dimension overflows the Choi width");
  const std::size_t choiWidth = d * d;
  EigenstateSynthesis synthesis(spacetime_, 1, metricSource_);
  result.bulkCells = synthesis.bulkMinusBoundaryCells();
  result.bulkCellCount = result.bulkCells.size();
  if (result.bulkCells.empty())
    return obstruct("bulk-minus-boundary has no interior 1-cells");

  if (frameCells.empty()) {
    if (result.bulkCells.size() != choiWidth)
      return obstruct(
          "an ordered d^2 Choi frame is required when the bulk width differs");
    frameCells = result.bulkCells;
  }
  if (frameCells.size() != choiWidth)
    return obstruct("the Choi frame must contain exactly d^2 interior 1-cells");
  std::map<std::vector<std::uint64_t>, std::size_t> bulkIndex;
  for (std::size_t i = 0; i < result.bulkCells.size(); ++i)
    bulkIndex[result.bulkCells[i]] = i;
  std::set<std::vector<std::uint64_t>> usedFrameCells;
  std::vector<std::size_t> frameColumns;
  frameColumns.reserve(frameCells.size());
  for (auto &cell : frameCells) {
    std::sort(cell.begin(), cell.end());
    if (cell.size() != 2 || cell[0] == cell[1])
      return obstruct("a Choi frame entry is not an edge");
    const auto found = bulkIndex.find(cell);
    if (found == bulkIndex.end())
      return obstruct("a Choi frame edge is not in the bulk-minus-boundary");
    if (!usedFrameCells.insert(cell).second)
      return obstruct("the Choi frame repeats an interior edge");
    frameColumns.push_back(found->second);
  }
  result.frameCells = frameCells;

  const std::vector<complexd> flat =
      synthesis.bulkMinusBoundaryHarmonicMatrix(tol, metric);
  if (flat.size() % result.bulkCells.size() != 0)
    return obstruct("bulk harmonic matrix has an inconsistent shape");
  result.kernelDimension = flat.size() / result.bulkCells.size();
  if (result.kernelDimension == 0)
    return obstruct("bulk-minus-boundary kernel is empty");

  Eigen::MatrixXcd restricted(static_cast<Eigen::Index>(result.kernelDimension),
                              static_cast<Eigen::Index>(choiWidth));
  for (std::size_t row = 0; row < result.kernelDimension; ++row)
    for (std::size_t column = 0; column < choiWidth; ++column)
      restricted(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(column)) =
          flat[row * result.bulkCells.size() + frameColumns[column]];
  if (!restricted.allFinite())
    return obstruct("the framed bulk kernel is non-finite");

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(restricted, Eigen::ComputeThinU |
                                                         Eigen::ComputeThinV);
  const Eigen::VectorXd &singularValues = svd.singularValues();
  result.frameSingularValues.reserve(
      static_cast<std::size_t>(singularValues.size()));
  for (Eigen::Index i = 0; i < singularValues.size(); ++i)
    result.frameSingularValues.push_back(singularValues[i]);
  const double leading = singularValues.size() == 0 ? 0.0 : singularValues[0];
  const double cutoff = tol * std::max(1.0, leading);
  for (Eigen::Index i = 0; i < singularValues.size(); ++i)
    if (singularValues[i] > cutoff)
      ++result.frameRank;
  if (result.frameRank == 0)
    return obstruct("the framed bulk kernel carries no Choi component");
  if (result.frameRank != 1)
    return obstruct("the framed bulk kernel has rank " +
                    std::to_string(result.frameRank) +
                    "; no unique Choi ray exists");

  // A = U Sigma V^H: the direct row-state spanning A is conj(V.col(0)). This
  // remains invariant under a change of basis among the bulk kernel rows.
  Eigen::VectorXcd choi = svd.matrixV().col(0).conjugate();
  for (Eigen::Index i = 0; i < choi.size(); ++i)
    if (std::abs(choi[i]) > cutoff) {
      choi *= std::conj(choi[i]) / std::abs(choi[i]);
      break;
    }
  result.choiState.assign(choi.data(), choi.data() + choi.size());

  result.operatorMatrix =
      ::tessera::quantum::ChoiJamiolkowski::operatorFromChoiState(
          result.choiState, stateDimension);
  Eigen::MatrixXcd U(stateDimension, stateDimension);
  for (int row = 0; row < stateDimension; ++row)
    for (int column = 0; column < stateDimension; ++column)
      U(row, column) = result.operatorMatrix[static_cast<std::size_t>(row) * d +
                                             static_cast<std::size_t>(column)];
  result.unitarityError =
      (U.adjoint() * U -
       Eigen::MatrixXcd::Identity(stateDimension, stateDimension))
          .norm();
  result.identifiable = true;
  result.obstruction.clear();
  return result;
}

std::set<std::uint64_t> MultiCobordism::pinnedVertices() const {
  std::set<std::uint64_t> united;
  for (const auto &region : pinnedRegions_)
    united.insert(region.vertices.begin(), region.vertices.end());
  return united;
}

bool MultiCobordism::edgeIsPinned(std::uint64_t a, std::uint64_t b) const {
  // Both endpoints within ONE region. One pinned endpoint leaves the edge free to
  // relax, and two regions that each hold one endpoint do not pin the edge that
  // spans between them — that edge is bulk.
  for (const auto &region : pinnedRegions_)
    if (region.vertices.count(a) != 0 && region.vertices.count(b) != 0)
      return true;
  return false;
}

MultiCobordism::Snapshot MultiCobordism::snapshotOf(
    const Spacetime &spacetime) const {
  std::vector<std::vector<std::uint64_t>> cellVertexTuples;
  for (const auto &topSimplex : spacetime.getTopSimplices())
    cellVertexTuples.push_back(topSimplex->topTuple());
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> lengthsByEdge;
  for (const auto *edge : spacetime.getEdgeList()->toVector())
    lengthsByEdge[edgeKey(edge)] = edge->getLength();  // verbatim, branch-exact
  return {std::move(cellVertexTuples), std::move(lengthsByEdge)};
}

MultiCobordism::Snapshot MultiCobordism::snapshot() const {
  return snapshotOf(*spacetime_);
}

std::shared_ptr<Spacetime> MultiCobordism::build(
    const Snapshot &complexSnapshot) const {
  auto rebuiltSpacetime = Spacetime::fromCells(spacetime_->getDimensions(),
                                               complexSnapshot.first, 1.0, 0.0);
  // Candidate clones inherit the wiring mode so COMBINATORIAL MOVES scored on
  // them wire their new edges under the same convention (#690).
  rebuiltSpacetime->setBalancedEdgeWiring(balancedEdgeWiring_);
  for (auto *edge : rebuiltSpacetime->getEdgeList()->toVector()) {
    const auto savedEntry = complexSnapshot.second.find(edgeKey(edge));
    if (savedEntry != complexSnapshot.second.end())
      edge->setLength(savedEntry->second);  // verbatim, branch-exact
  }
  return rebuiltSpacetime;
}

MultiCobordism::MoveSpec MultiCobordism::drawRandomMoveSpecification(
    const Spacetime &spacetime) {
  // #613: with emergent dispositions ON the draw also offers a TIMELIKE cone-in
  // and a disposition flip on an existing edge. Both are ordinary candidate moves:
  // proposed at random, scored by deltaF, committed only if they lower F. Nothing
  // prescribes causal structure -- the objective decides whether it wants any.
  //
  // These discrete proposals remain useful for jumping directly between causal
  // sectors. Complex-z stage 2 can also rotate continuously around z=0; neither
  // path prescribes which causal structure the objective should prefer.
  static const char *baseMoveKinds[] = {kAddMove,  kRemoveMove, kFlipMove,
                                        kIFlipMove, kConeOut,   kConeIn};
  static const char *dispositionMoveKinds[] = {
      kAddMove, kRemoveMove,     kFlipMove,        kIFlipMove,
      kConeOut, kConeIn,         kConeInTimelike,  kFlipDisposition};
  const char *const *moveKinds =
      shouldProposeDispositions_ ? dispositionMoveKinds : baseMoveKinds;
  const std::size_t nMoveKinds = shouldProposeDispositions_ ? 8u : 6u;
  const std::string moveKind = moveKinds[randomNumberGenerator_() % nMoveKinds];

  // Flip the disposition of one existing edge, chosen uniformly. The payload is
  // the edge's two vertex ids.
  if (moveKind == kFlipDisposition) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> edgeEndpoints;
    if (spacetime.getEdgeList())
      for (const auto *edge : spacetime.getEdgeList()->toVector())
        if (edge != nullptr && edge->getSource() != nullptr &&
            edge->getTarget() != nullptr)
          edgeEndpoints.emplace_back(edge->getSource()->getId(),
                                     edge->getTarget()->getId());
    if (edgeEndpoints.empty()) return {kNoop, {}};
    const auto &chosen =
        edgeEndpoints[randomNumberGenerator_() % edgeEndpoints.size()];
    return {kFlipDisposition, {chosen.first, chosen.second}};
  }
  if (moveKind == kAddMove || moveKind == kRemoveMove ||
      moveKind == kFlipMove || moveKind == kIFlipMove)
    return {moveKind,
            {static_cast<std::uint64_t>(randomNumberGenerator_() % (1u << 31))}};
  if (moveKind == kConeOut) {
    std::vector<std::vector<std::uint64_t>> topCellTuples;
    for (const auto &topSimplex : spacetime.getTopSimplices())
      topCellTuples.push_back(topSimplex->topTuple());
    if (topCellTuples.empty()) return {kNoop, {}};
    return {kConeOut,
            topCellTuples[randomNumberGenerator_() % topCellTuples.size()]};
  }
  // cone_in and cone_in_timelike share a payload (the facet to cone onto); only
  // the apex-edge disposition differs when applied. Only a BOUNDARY facet (one
  // coface) can accept a cone — an interior facet already has two cofaces, so
  // coning it would be non-manifold and the gate rejects it after a full
  // build+apply+deltaF evaluation. Drawing from getBoundary() directly spends
  // the batch on the coneable set only (measured on a 13-cell build frame: the
  // old cell×dropped-vertex draw had 65 outcomes aliasing onto 41 facets, 17
  // coneable, with interior duds drawn twice as often as valid facets). The
  // facet's stored vertex order is passed through verbatim.
  auto boundaryFacets = spacetime.getBoundary();
  if (boundaryFacets.empty()) return {kNoop, {}};  // closed: nothing coneable
  return {moveKind,
          boundaryFacets[randomNumberGenerator_() % boundaryFacets.size()]};
}

bool MultiCobordism::applyMoveSpecification(
    const std::shared_ptr<Spacetime> &spacetime,
    const MoveSpec &moveSpecification) {
  const auto &moveKind = moveSpecification.first;
  CLOG(INFO_LEVEL, "Applying a ", moveKind, " move.");
  if (moveKind == kNoop) return false;
  bool moveWasApplied = false;
  if (moveKind == kAddMove || moveKind == kRemoveMove ||
      moveKind == kFlipMove || moveKind == kIFlipMove) {
    std::mt19937 moveRandomEngine(
        static_cast<std::uint32_t>(moveSpecification.second[0]));
    using ::tessera::spacetime::PachnerMode;
    if (moveKind == kAddMove) {
      ::tessera::spacetime::AddMove pachnerMove(
          spacetime.get(), &moveRandomEngine, false, PachnerMode::PreGeometric,
          false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == kRemoveMove) {
      ::tessera::spacetime::RemoveMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == kFlipMove) {
      ::tessera::spacetime::FlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else {
      ::tessera::spacetime::IFlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    }
  } else if (moveKind == kConeOut) {
    moveWasApplied =
        SurgicalCone(spacetime.get()).coneOut(moveSpecification.second).first;
  } else if (moveKind == kFlipDisposition) {
    // #613: negate one edge's squared length, carrying it across the light cone.
    // Spacelike <-> timelike is a DISCRETE step stage 2 cannot take (it would have
    // to pass through the singular l^2 = 0), which is why it is a move. Not gated
    // here -- deltaF and step()'s acceptance test gate it, exactly as for every
    // other move.
    if (payloadNamesAnEdge(moveSpecification.second) &&
        spacetime->getEdgeList()) {
      // O(1) via the EdgeList's fingerprint -> slot map, not an O(|E|) scan:
      // EdgeKey canonicalizes the endpoint pair, so orientation does not matter.
      const ::tessera::mesh::EdgeKey key(moveSpecification.second[0],
                                         moveSpecification.second[1]);
      if (auto *edge =
              spacetime->getEdgeList()->get(key.fingerprint.fingerprint())) {
        edge->setLength(std::sqrt(-(edge->getLength() * edge->getLength())));
        moveWasApplied = true;
      }
    }
  } else {
    moveWasApplied = SurgicalCone(spacetime.get())
                         .coneIn(moveSpecification.second,
                                 /*timelike=*/moveKind == kConeInTimelike)
                         .first;
  }
  if (!moveWasApplied) return false;
  // Manifold validity is the whole gate. A move that removes a pinned vertex is
  // accepted when what it leaves is a valid manifold in its own right: pinning
  // constrains the geometry, it does not veto a topology change.
  if (!EigenstateSynthesis(spacetime, dualComplexGateDegree_, metricSource_)
           .dualComplexValid()
           .first)
    return false;
  // Under the Whitney pencil the configuration space is the closure of the
  // Kontsevich–Segal allowable domain; a proposal outside it is not a member.
  return geometryAdmissible(spacetime);
}

bool MultiCobordism::geometryAdmissible(const std::shared_ptr<Spacetime> &spacetime) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil) return true;
  if (!spacetime) return false;
  constexpr double kBoundaryTolerance = 1e-12;
  return HodgeLaplacian::kontsevichSegalMargin(*spacetime) >= -kBoundaryTolerance;
}

double MultiCobordism::deltaF(
    const std::shared_ptr<Spacetime> &candidateSpacetime, double baseObjective,
    double baseResidualU,
    const std::set<std::vector<std::uint64_t>> &baseCellSet) const {
  // An objective that declares a localized exact delta is differenced over the
  // cells the move touches, below. One that does not depends on global spectra
  // or action magnitudes, so its true scalar difference is the only honest
  // score. It is more expensive, but it prevents stage 1 from optimizing a
  // surrogate different from the objective it reports.
  if (!compositeSupportsLocalizedDelta())
    return objectiveFor(candidateSpacetime) - baseObjective;

  std::set<std::vector<std::uint64_t>> candidateCellSet;
  for (const auto &topSimplex : candidateSpacetime->getTopSimplices())
    candidateCellSet.insert(topSimplex->topTuple());
  std::vector<std::vector<std::uint64_t>> touchedCells;
  for (const auto &cell : baseCellSet)
    if (!candidateCellSet.count(cell)) touchedCells.push_back(cell);
  for (const auto &cell : candidateCellSet)
    if (!baseCellSet.count(cell)) touchedCells.push_back(cell);

  // The touched-cell diff alone is NOT the whole affected set (#633): a
  // flip_disposition changes an edge's l^2 SIGN and changes no cells at all, so the
  // diff comes back empty and the geometry term would be scored as exactly 0 --
  // while flipping an edge between spacelike and timelike changes the deficit angle
  // of every hinge on it. So diff the edge l^2 values too, and pull in the top cells
  // incident to any edge that moved. Move-agnostic on purpose: this also covers
  // cone_in_timelike's apex edges and any future move that perturbs geometry without
  // changing cells, rather than special-casing a move kind.
  //
  // Widening is safe: Delta||grad S||^2 = after - before is exact over any FIXED
  // SUPERSET of the truly-affected edges, because every edge outside the set keeps
  // its gradient and cancels. A superset costs compute, never correctness.
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> baseLengths;
  for (const auto *edge : spacetime_->getEdgeList()->toVector())
    baseLengths[edgeKey(edge)] = edge->getLength();
  std::set<std::pair<std::uint64_t, std::uint64_t>> movedEdges;
  for (const auto *edge : candidateSpacetime->getEdgeList()->toVector()) {
    const auto key = edgeKey(edge);
    const auto found = baseLengths.find(key);
    if (found == baseLengths.end() ||
        found->second != edge->getLength())
      movedEdges.insert(key);
    if (found != baseLengths.end()) baseLengths.erase(found);
  }
  for (const auto &leftover : baseLengths)  // in base, absent from candidate
    movedEdges.insert(leftover.first);
  if (!movedEdges.empty()) {
    std::set<std::vector<std::uint64_t>> incidentCells;
    const auto collectIncident = [&](const Spacetime &spacetime) {
      for (const auto &topSimplex : spacetime.getTopSimplices()) {
        auto cellVertexIds = topSimplex->topTuple();
        for (const auto &moved : movedEdges) {
          const bool cellHoldsBothEndpoints =
              std::find(cellVertexIds.begin(), cellVertexIds.end(),
                        moved.first) != cellVertexIds.end() &&
              std::find(cellVertexIds.begin(), cellVertexIds.end(),
                        moved.second) != cellVertexIds.end();
          if (cellHoldsBothEndpoints) {
            incidentCells.insert(std::move(cellVertexIds));
            break;
          }
        }
      }
    };
    collectIncident(*spacetime_);
    collectIncident(*candidateSpacetime);
    for (const auto &cell : incidentCells)
      if (std::find(touchedCells.begin(), touchedCells.end(), cell) ==
          touchedCells.end())
        touchedCells.push_back(cell);
  }

  ReggeSolver baseReggeSolver(spacetime_, MatterConfiguration());
  ReggeSolver candidateReggeSolver(candidateSpacetime, MatterConfiguration());
  std::set<std::pair<std::uint64_t, std::uint64_t>> affectedEdgeSet;
  for (const auto &edgeEndpoints :
       baseReggeSolver.affectedEdgesOfCells(touchedCells))
    affectedEdgeSet.insert(edgeEndpoints);
  for (const auto &edgeEndpoints :
       candidateReggeSolver.affectedEdgesOfCells(touchedCells))
    affectedEdgeSet.insert(edgeEndpoints);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> affectedEdges(
      affectedEdgeSet.begin(), affectedEdgeSet.end());
  // Skipped entirely when the Einstein-Hilbert term is off (#724): scoring a
  // move by a term the objective does not contain would make stage 1 disagree
  // with `objective()` about which moves lower F.
  const double gradientDelta =
      einsteinHilbert_
          ? candidateReggeSolver.gradientNorm2OverEdges(affectedEdges) -
                baseReggeSolver.gradientNorm2OverEdges(affectedEdges)
          : 0.0;
  const double residualUDelta = rU(candidateSpacetime) - baseResidualU;
  return reggeWeight_ * gradientDelta + gamma_ * residualUDelta;
}

double MultiCobordism::step(int nCandidateMoves, int lookaheadDepth,
                            double baseObjective) {
  // The candidate loop below constructs one `ReggeSolver` on the LIVE
  // spacetime per candidate (`deltaF`), and that constructor materializes the
  // facet lattice lazily — a mutation of the shared object. On a live complex
  // whose facets are not yet materialized (a fresh seed, or the complex built
  // from the snapshot of a committed move) two OpenMP threads then race in
  // `Simplex::getFacets` and corrupt the simplex deque (measured on the Δ³
  // fiber drive, #940: SIGSEGV in `Spacetime::createSimplex` from two
  // candidate threads). Reach the fixpoint once, here, so every thread's
  // construction is read-only.
  spacetime_->materializeFacets();
  const auto currentSnapshot = snapshot();
  const double baseResidualU =
      compositeSupportsLocalizedDelta() ? rU(spacetime_) : 0.0;
  std::set<std::vector<std::uint64_t>> baseCellSet;
  for (const auto &topSimplex : spacetime_->getTopSimplices())
    baseCellSet.insert(topSimplex->topTuple());
  // ONE scoring rule at every depth: the localized, UNRELAXED deltaF (#714).
  // The two stages have separate jobs — the combinatorial moves exist to leave
  // a local minimum, the geometric update to descend to the minimum of the
  // region the complex then sits in — and scoring a candidate through a
  // relaxation mixed them, asking where a move would land after stage 2 rather
  // than whether the move itself improves the state. Relaxation now happens
  // only after a move is committed, bounded by the caller's relaxBudgetPerMove.
  double bestObjectiveDelta = -convergenceTolerance_;
  bool foundImprovingMove = false;
  Snapshot bestSnapshot;
  if (lookaheadDepth <= 1) {
    // Depth 1: every candidate starts from the SAME base complex, so the specs
    // can be pre-drawn serially (identical RNG order to the serial loop — the
    // per-seed draw sequence is unchanged) and the batch scored in parallel:
    // applyMoveSpecification is deterministic given its spec (it seeds a local
    // engine from the payload), build() constructs an independent complex, and
    // deltaF is const over it. The inner OpenMP region of the action gradient
    // serializes inside each worker (nesting off), so the batch parallelism is
    // the outer level. The reduction is the lexicographic (delta, index) min,
    // which reproduces the serial rule exactly: the EARLIEST candidate among
    // equals wins.
    std::vector<MoveSpec> specifications;
    specifications.reserve(static_cast<std::size_t>(nCandidateMoves));
    for (int candidateIndex = 0; candidateIndex < nCandidateMoves;
         ++candidateIndex)
      specifications.push_back(drawRandomMoveSpecification(*spacetime_));
    // Deduplicate exact (kind, payload) repeats before evaluating: the batch
    // samples with replacement, and on a small complex the same spec recurs
    // (the cone-in space can be a dozen-odd facets). Duplicates carry
    // identical deltas, so dropping every copy after the first cannot change
    // the lexicographic (delta, index) winner — the committed move is
    // bit-identical, only the wasted build+apply+deltaF evaluations go away.
    // The RNG stream is untouched (all nCandidateMoves draws happen above).
    // Pachner specs carry RNG-seed payloads, so only exact seed repeats
    // collapse there; the cone/disposition kinds dedup by actual site.
    {
      std::set<MoveSpec> seenSpecifications;
      std::vector<MoveSpec> distinctSpecifications;
      distinctSpecifications.reserve(specifications.size());
      for (auto &specification : specifications)
        if (seenSpecifications.insert(specification).second)
          distinctSpecifications.push_back(std::move(specification));
      specifications = std::move(distinctSpecifications);
    }
    const int distinctCount = static_cast<int>(specifications.size());
    std::vector<double> deltas(static_cast<std::size_t>(distinctCount),
                               std::numeric_limits<double>::infinity());
    std::vector<Snapshot> snapshots(static_cast<std::size_t>(distinctCount));
#pragma omp parallel for schedule(dynamic)
    for (int candidateIndex = 0; candidateIndex < distinctCount;
         ++candidateIndex) {
      auto candidateSpacetime = build(currentSnapshot);
      if (!applyMoveSpecification(
              candidateSpacetime,
              specifications[static_cast<std::size_t>(candidateIndex)]))
        continue;  // failed the gate: stays at +inf
      const double objectiveDelta =
          deltaF(candidateSpacetime, baseObjective, baseResidualU, baseCellSet);
      deltas[static_cast<std::size_t>(candidateIndex)] = objectiveDelta;
      if (objectiveDelta < -convergenceTolerance_)
        snapshots[static_cast<std::size_t>(candidateIndex)] =
            snapshotOf(*candidateSpacetime);
    }
    for (int candidateIndex = 0; candidateIndex < distinctCount;
         ++candidateIndex) {
      const double objectiveDelta =
          deltas[static_cast<std::size_t>(candidateIndex)];
      if (objectiveDelta < bestObjectiveDelta) {
        bestObjectiveDelta = objectiveDelta;
        bestSnapshot =
            std::move(snapshots[static_cast<std::size_t>(candidateIndex)]);
        foundImprovingMove = true;
      }
    }
  } else
  for (int candidateIndex = 0; candidateIndex < nCandidateMoves; ++candidateIndex) {
    // One candidate = `lookaheadDepth` gated random moves applied in sequence,
    // each drawn against the evolving candidate complex. The sequence is scored
    // — and, if best, committed — as a WHOLE, so an F-lowering pair whose first
    // move alone raises F is still an honest descent step. This deepened path
    // stays SERIAL: each draw is made against the candidate the previous move
    // left, so the sequence cannot be pre-drawn the way a depth-1 batch is.
    auto candidateSpacetime = build(currentSnapshot);
    bool wholeSequenceApplied = true;
    for (int moveIndex = 0; moveIndex < lookaheadDepth; ++moveIndex) {
      const auto moveSpecification =
          drawRandomMoveSpecification(*candidateSpacetime);
      if (!applyMoveSpecification(candidateSpacetime, moveSpecification)) {
        wholeSequenceApplied = false;  // one link failed the gate: discard the candidate
        break;
      }
    }
    if (!wholeSequenceApplied) continue;
    // Scored exactly as a depth-1 candidate is: the finished sequence diffed
    // against the base complex. deltaF is exact over any fixed superset of the
    // affected edges, so a multi-move candidate needs no special treatment.
    const double objectiveDelta =
        deltaF(candidateSpacetime, baseObjective, baseResidualU, baseCellSet);
    if (objectiveDelta < bestObjectiveDelta) {
      bestObjectiveDelta = objectiveDelta;
      // The snapshot carries the sequence's AS-BUILT geometry: nothing was
      // relaxed to earn the score, so nothing is being banked here either.
      bestSnapshot = snapshotOf(*candidateSpacetime);
      foundImprovingMove = true;
    }
  }
  if (foundImprovingMove) {
    spacetime_ = build(bestSnapshot);
    // The first committed move is what starts linking the bulk, so block
    // regions are settled from here on (#737).
    bulkConnected_ = true;
    // #776: the move is ALREADY committed — `bestObjectiveDelta` is fixed and
    // `spacetime_` already replaced — before the analysis overlay is offered
    // the chance to look at it. The overlay is post-hoc by construction: there
    // is no path from here back to the acceptance test above.
    noteAcceptedMove();
    return bestObjectiveDelta;
  }
  return 0.0;
}

void MultiCobordism::preconeCells(int count, bool timelike, bool alternate) {
  // Each cone-in cones a fresh apex onto a random codim-1 facet (a top cell with one
  // vertex dropped) and is committed only through applyMoveSpecification's
  // dualComplexValid gate — the same gated primitive the stage-1 draw uses, so the
  // pre-growth is sound (nothing inserted by fiat). On the single-Δ⁴ seed (a 4-ball)
  // a cone-in over a boundary facet is valid, so this enlarges the 4-ball; a draw
  // onto an already-saturated interior facet is rejected by the gate and retried.
  // `timelike` draws every cone-in as the TIMELIKE disposition (apex edges
  // ℓ² = −1); `alternate` instead interleaves timelike/spacelike cone-ins for
  // balanced causal content. Either way every edge sits at one uniform
  // magnitude |ℓ²| = 1; the default is the all-spacelike precone.
  constexpr int kAttemptsPerCone = 20;  // gated tries before giving up on one cone
  for (int conedSoFar = 0; conedSoFar < count; ++conedSoFar) {
    std::vector<std::vector<std::uint64_t>> topCellTuples;
    for (const auto &topSimplex : spacetime_->getTopSimplices())
      topCellTuples.push_back(topSimplex->topTuple());
    if (topCellTuples.empty()) return;  // nothing to cone onto
    bool coned = false;
    for (int attempt = 0; attempt < kAttemptsPerCone && !coned; ++attempt) {
      const auto &chosenCell =
          topCellTuples[randomNumberGenerator_() % topCellTuples.size()];
      const std::size_t droppedVertexIndex =
          randomNumberGenerator_() % chosenCell.size();
      std::vector<std::uint64_t> coneInFace;  // a codim-1 facet: drop one vertex
      for (std::size_t vertexIndex = 0; vertexIndex < chosenCell.size();
           ++vertexIndex)
        if (vertexIndex != droppedVertexIndex)
          coneInFace.push_back(chosenCell[vertexIndex]);
      auto candidateSpacetime = build(snapshot());
      const bool coneTimelike =
          alternate ? (conedSoFar % 2 == 0) : timelike;
      if (applyMoveSpecification(
              candidateSpacetime,
              {coneTimelike ? kConeInTimelike : kConeIn, coneInFace})) {
        spacetime_ = build(snapshotOf(*candidateSpacetime));
        coned = true;
      }
    }
    if (!coned) return;  // no valid cone-in found for this cell; stop early
  }
}

void MultiCobordism::growBlockRegions() {
  // Growth is a SETUP step: it runs only before the bulk is connected, and
  // only when a shell strictly LOWERS the block's residual (#737).
  //
  // Both conditions exist because the old rule had no stopping point. The gate
  // was "keep the shell unless the residual rises", and a block that is not
  // carrying sits at exactly the constant full-leak residual for ANY region
  // size — so every shell scored a change of exactly zero, was always kept,
  // and the region grew until it ran out of complex. Measured on a six-block
  // node: regions [21, 13, 15, 5, 13, 21] became [25, 25, 25, 25, 25, 25], the
  // whole complex, so all six blocks were reading one identical sub-complex and
  // differed only in their target vectors.
  if (bulkConnected_) return;   // the bulk is linked; the states stay as they are
  // Expand one block's READ WINDOW by a shell — the vertices of every top cell
  // touching it — so it gets room to open the holes that carry it. A block
  // already carrying (residual < tolerance) is left alone, so it stops growing
  // once it represents its state.
  //
  // This grows a SCORING REGION, never the cobordism's boundary: a block is a
  // vertex set plus a target, and that set selects the sub-complex the block's
  // residual is read over. Nothing here creates a cell, an edge, or a vertex —
  // the only write is to `block.vertices`, and every `spacetime_` access below
  // is a read.
  //
  // GATED on the block's own residual: a shell is kept only when it STRICTLY
  // lowers the block's r_U term, so region growth can never raise F and never
  // buys nothing. The earlier Δ <= 0 gate was chosen so a region too small to
  // hold a full cell (whose shells are exact ties) could still get started, but
  // that same allowance is what let a permanently-leaking block grow forever.
  const auto growOneShell = [this](BoundaryBlock &block) {
    const double residualBefore = residualForBoundaryBlock(block, spacetime_);
    if (residualBefore < inputCarriedTolerance_) return;
    std::set<std::uint64_t> expanded = block.vertices;
    for (const auto &topSimplex : spacetime_->getTopSimplices()) {
      auto cellVertexIds = topSimplex->topTuple();
      bool touchesRegion = false;
      for (auto vertexId : cellVertexIds)
        if (block.vertices.count(vertexId)) {
          touchesRegion = true;
          break;
        }
      if (touchesRegion)
        expanded.insert(cellVertexIds.begin(), cellVertexIds.end());
    }
    std::set<std::uint64_t> original = std::move(block.vertices);
    block.vertices = std::move(expanded);
    // STRICT: a shell is kept only if it actually improves the carry. A shell
    // that leaves the residual unchanged buys nothing and is what let the
    // regions sprawl, so it is reverted like a harmful one.
    if (residualForBoundaryBlock(block, spacetime_) >= residualBefore)
      block.vertices = std::move(original);
  };
  for (auto &inputBlock : inputBlocks_) growOneShell(inputBlock);
  // Localized OUTPUT blocks (a 2→2 recombination's diquark ⊔ antidiquark) grow the
  // same way; a SINGLE output reads off the whole and has no block here, so this is
  // a no-op for the formation node.
  for (auto &outputBlock : outputBlocks_) growOneShell(outputBlock);
}

std::vector<double> MultiCobordism::runStage1(int maxSteps, int nCandidateMoves,
                                                 bool growBoundaries,
                                                 int maxLookahead) {
  std::vector<double> objectiveTrace = {objective()};
  for (int stepIndex = 0; stepIndex < maxSteps; ++stepIndex)
    if (!stage1Update(nCandidateMoves, growBoundaries, objectiveTrace,
                      maxLookahead))
      break;
  return objectiveTrace;
}

bool MultiCobordism::stage1Update(int nCandidateMoves, bool growBoundaries,
                                  std::vector<double> &objectiveTrace,
                                  int maxLookahead) {
  // In target-conditioned modes the register is "carried" once summed r_U is
  // essentially zero. JointStationarity never consults this target diagnostic.
  constexpr double kRegisterCarriedTolerance = 1e-3;
  // INITIALIZATION ONLY: while establishing the boundary states, let each
  // not-yet-carrying block expand its scoring region by a shell so it can develop
  // the holes that carry its state. Off during the bulk evolution — the regions
  // are then frozen too. This never moves ∂W (see growBlockRegions).
  //
  // Growing a region CHANGES F and so must be booked into the trace (#607) —
  // though with the per-block gate in `growBlockRegions` (a shell that raises
  // the block's residual is reverted) the booked delta is now always <= 0.
  // `growBlockRegions` mutates only the blocks' scoring-region vertex sets and
  // never touches `spacetime_`, so `reggeActionGradient` is provably unchanged and the
  // whole objective change is `gamma_ * Δr_U` — exact, not an approximation of the
  // kind `deltaF` makes for the gradient term. Leaving it unbooked let the
  // accumulated trace drift arbitrarily far from `objective()` (measured at tens of
  // thousands on preconed hosts), and since the SAME accumulated quantity gates
  // acceptance, moves were being committed against a number that was not F.
  if (growBoundaries && objectiveSpec_->needsRegisterResidual()) {
    const double objectiveBeforeGrowth = objective();
    growBlockRegions();
    const double growthObjectiveDelta = objective() - objectiveBeforeGrowth;
    if (growthObjectiveDelta != 0.0)
      objectiveTrace.push_back(objectiveTrace.back() + growthObjectiveDelta);
  }
  // ITERATIVE-DEEPENING LOOKAHEAD: try single moves first (depth 1 — the cheap,
  // common case); only when that batch finds no improvement does the search
  // deepen to 2-move sequences, then 3, up to `maxLookahead`. A sequence is
  // scored and committed as a WHOLE, so an F-lowering pair whose first move
  // alone raises F — the plateau that used to need the trap-door escape — is
  // reached by honest descent rather than growth on faith.
  lastStage1LookaheadDepth_ = 0;  // report: nothing committed until proven otherwise
  // A stalled search is allowed to go WIDE as well as deep: depth 1 keeps the
  // caller's fast batch (the common, cheap case), while each deepened batch
  // scans on the order of a hundred candidate sequences — deep sequences die on
  // the gate chain far more often and the move space grows with depth, so a
  // handful of draws would badly under-sample it. The budget is only spent
  // when depth 1 already failed, i.e. exactly when it is worth it.
  constexpr int kDeepLookaheadCandidates = 128;
  // Candidates without a localized delta use an exact global objective. Score
  // the unchanged base geometry once for the entire iterative-deepening pass
  // instead of once per candidate endpoint (up to 524 redundant evaluations at
  // depth five).
  const double baseObjective =
      compositeSupportsLocalizedDelta() ? 0.0 : objectiveFor(spacetime_);
  for (int lookaheadDepth = 1; lookaheadDepth <= std::max(1, maxLookahead);
       ++lookaheadDepth) {
    const int batchSize =
        lookaheadDepth == 1 ? nCandidateMoves
                            : std::max(nCandidateMoves, kDeepLookaheadCandidates);
    const double objectiveDelta =
        step(batchSize, lookaheadDepth, baseObjective);
    if (objectiveDelta < -convergenceTolerance_) {
      // An F-lowering surgery sequence: progress.
      objectiveTrace.push_back(objectiveTrace.back() + objectiveDelta);
      lastStage1LookaheadDepth_ = lookaheadDepth;
      return true;
    }
  }
  // A target-free objective is done when no improving sequence is found: there
  // is nothing else it was trying to reach. A target-conditioned one halts when
  // the register is carried; otherwise it keeps drawing, because a random miss
  // is not proof that no target-improving sequence exists. `maxSteps` bounds
  // those retries.
  if (!objectiveSpec_->isTargetConditioned()) return false;
  return rU(spacetime_) >= kRegisterCarriedTolerance;
}

void MultiCobordism::seedInputs(const std::vector<std::uint64_t> &seeds) {
  seedBlocks(seeds, inputTargets_, inputBlocks_);
}

void MultiCobordism::seedOutputs(const std::vector<std::uint64_t> &seeds) {
  seedBlocks(seeds, outputTargets_, outputBlocks_);
}

void MultiCobordism::seedBlocks(
    const std::vector<std::uint64_t> &seeds,
    const std::vector<std::vector<complexd>> &targets,
    std::vector<BoundaryBlock> &destinationBlocks) {
  // Seed one boundary block per (seed vertex, target): its initial region is the seed
  // vertex's cell-neighbourhood. The block is NOT pre-grown here — runStage1's
  // growBlockRegions grows it under the objective, so the carrying topology is fully
  // emergent. The seed vertex is the only anchor (it distinguishes one input/output
  // from another); everything else emerges.
  for (std::size_t blockIndex = 0;
       blockIndex < targets.size() && blockIndex < seeds.size(); ++blockIndex) {
    const std::uint64_t seedVertexId = seeds[blockIndex];
    std::set<std::uint64_t> regionVertexIds;
    for (const auto &topSimplex : spacetime_->getTopSimplices()) {
      auto cellVertexIds = topSimplex->topTuple();
      if (std::find(cellVertexIds.begin(), cellVertexIds.end(), seedVertexId) !=
          cellVertexIds.end())
        regionVertexIds.insert(cellVertexIds.begin(), cellVertexIds.end());
    }
    destinationBlocks.push_back(BoundaryBlock{regionVertexIds, targets[blockIndex]});
  }
}

std::vector<double> MultiCobordism::runStage2(double beta, int maxIters,
                                                 double alpha0, double tolerance) {
  setReggeWeight(beta);
  std::vector<double> objectiveTrace = {objective()};
  double stepScale = alpha0;
  lastStage2Stationary_ = false;  // for maxIters == 0; each update reports its own
  for (int iterationIndex = 0; iterationIndex < maxIters; ++iterationIndex)
    if (!stage2Update(beta, tolerance, objectiveTrace, stepScale)) break;
  return objectiveTrace;
}

std::vector<double> MultiCobordism::run(int maxIters, int nCandidateMoves,
                                        bool growBoundaries, double beta,
                                        double alpha0, double tolerance,
                                        int maxLookahead,
                                        int relaxBudgetPerMove) {
  setReggeWeight(beta);
  std::vector<double> objectiveTrace = {objective()};
  double stepScale = alpha0;
  lastStage2Stationary_ = false;  // for maxIters == 0; each update reports its own
  // A single stalled stage-1 batch is a random-draw miss, not proof the moves
  // have no effect (measured on a timelike-preconed drive: committed moves landed
  // several stalled batches apart), so exhaustion is only concluded after this
  // many CONSECUTIVE no-effect iterations.
  constexpr int kConsecutiveNoEffectLimit = 3;
  int consecutiveNoEffect = 0;
  for (int iterationIndex = 0; iterationIndex < maxIters; ++iterationIndex) {
    // ONE combinatorial move (or lookahead sequence), then a FULL geometric
    // relaxation: stage-2 updates repeat until the absolute-improvement test
    // reports diminishing returns. Every committed move is therefore scored
    // from — and leaves behind — relaxed geometry (stage2Update re-reads the
    // edge list each call, picking up whatever the move just created).
    const bool stage1WantsAnotherIteration = stage1Update(
        nCandidateMoves, growBoundaries, objectiveTrace, maxLookahead);
    const bool moveCommitted = lastStage1LookaheadDepth_ > 0;
    // "Full" relaxation still needs a safety budget (as runStage2's maxIters):
    // near a slow descent tail the line search can accept a near-unbounded
    // number of threshold-sized micro-steps, so the stationarity test alone
    // does not bound the loop in practice. Caller-tunable (#666); the
    // stationarity test remains the real terminator.
    bool geometryRelaxed = false;
    for (int relaxIndex = 0; relaxIndex < relaxBudgetPerMove; ++relaxIndex) {
      if (!stage2Update(beta, tolerance, objectiveTrace, stepScale)) break;
      geometryRelaxed = true;
    }
    // "The combinatorial moves have no effect": nothing committed at any
    // lookahead depth AND nothing left to relax — but only after enough
    // consecutive misses to rule out draw noise.
    if (!moveCommitted && !geometryRelaxed)
      ++consecutiveNoEffect;
    else
      consecutiveNoEffect = 0;
    const bool wantsExit =
        (!stage1WantsAnotherIteration && !geometryRelaxed) ||
        consecutiveNoEffect >= kConsecutiveNoEffectLimit;
    if (wantsExit) {
      // The LAST geometric relaxation before exit runs at a much tighter
      // tolerance than the in-loop diminishing-returns cut. If the tighter pass
      // still finds descent the state was NOT truly stationary — the exit was
      // premature — so keep looping on the freshly relaxed geometry (which may
      // also enable new moves). Exit only once stationary at 1e-12 too.
      constexpr double kExitRelTol = 1e-12;
      bool tighterPassFoundDescent = false;
      for (int relaxIndex = 0; relaxIndex < relaxBudgetPerMove; ++relaxIndex) {
        if (!stage2Update(beta, kExitRelTol, objectiveTrace, stepScale)) break;
        tighterPassFoundDescent = true;
      }
      if (!tighterPassFoundDescent) break;
      consecutiveNoEffect = 0;  // it moved: not done after all
    }
  }
  return objectiveTrace;
}

bool MultiCobordism::stage2Update(double beta, double tolerance,
                                  std::vector<double> &objectiveTrace,
                                  double &stepScale) {
  // Reset-then-set: the flag reports THIS call's outcome, so in the combined
  // drive (`run`) it reflects the most recent geometric update instead of
  // latching true after a stationary point a later topology change reopened.
  // (`runStage2` is unaffected: there a true flag breaks its loop immediately.)
  lastStage2Stationary_ = false;
  // Within one `runStage2` call the topology is fixed (only edge lengths move), so
  // re-reading the edge list here is free; in the combined drive (`run`) it is what
  // picks up the edges a stage-1 move just created or removed.
  const auto &edges = spacetime_->getEdgeList()->toVector();
  const std::size_t edgeCount = edges.size();
  Eigen::VectorXcd lengths(edgeCount);
  Eigen::VectorXcd squaredLengths(edgeCount);
  for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
    lengths(edgeIndex) = edges[edgeIndex]->getLength();
    squaredLengths(edgeIndex) = lengths(edgeIndex) * lengths(edgeIndex);
  }
  // The connection phase is the node's OTHER edge field, and it relaxes on its
  // own coordinate: phi is not derived from l, so there is no square-root
  // branch to track and the trial is written directly.
  Eigen::VectorXcd phases(edgeCount);
  for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
    phases(edgeIndex) = edges[edgeIndex]->getPhase();
  const auto restoreEdgeLengths = [&]() {
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
      edges[edgeIndex]->setLength(lengths(edgeIndex));
      edges[edgeIndex]->setPhase(phases(edgeIndex));
    }
  };
  const auto setSquaredLengths = [&](const Eigen::VectorXcd &trialSquared) {
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      edges[edgeIndex]->setLength(continuousSquareRoot(
          trialSquared(edgeIndex), lengths(edgeIndex)));
  };
  const auto setPhases = [&](const Eigen::VectorXcd &trialPhases) {
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      edges[edgeIndex]->setPhase(trialPhases(edgeIndex));
  };
  auto fullObjective = [&]() { return objectiveFor(spacetime_); };

  // Return the steepest-ASCENT displacement in the complex z plane for a real
  // scalar. Stage 2 subtracts it. This finite-difference path is reserved for
  // r_U, whose target/block composition has no one closed-form derivative yet.
  const auto scalarAscentDirection = [&](const auto &functional) {
    Eigen::VectorXcd ascent = Eigen::VectorXcd::Zero(edgeCount);
    const double relativeStep =
        std::cbrt(std::numeric_limits<double>::epsilon());
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
      const double coordinateStep =
          relativeStep * std::max(std::abs(squaredLengths(edgeIndex)), 1.0);
      const auto evaluateAt = [&](complexd value) {
        edges[edgeIndex]->setLength(continuousSquareRoot(
            value, lengths(edgeIndex)));
        return functional();
      };
      const double realPlus =
          evaluateAt(squaredLengths(edgeIndex) + coordinateStep);
      const double realMinus =
          evaluateAt(squaredLengths(edgeIndex) - coordinateStep);
      double imaginaryDerivative = 0.0;
      if (!realSquaredLengthsOnly_) {
        const double imaginaryPlus =
            evaluateAt(squaredLengths(edgeIndex) +
                       complexd{0.0, coordinateStep});
        const double imaginaryMinus =
            evaluateAt(squaredLengths(edgeIndex) -
                       complexd{0.0, coordinateStep});
        imaginaryDerivative =
            (imaginaryPlus - imaginaryMinus) / (2.0 * coordinateStep);
      }
      edges[edgeIndex]->setLength(lengths(edgeIndex));
      ascent(edgeIndex) = complexd{
          (realPlus - realMinus) / (2.0 * coordinateStep),
          imaginaryDerivative};
    }
    return ascent;
  };

  // Explicit fixed-hole constraints already have an exact analytic r_U
  // gradient. When they are the whole residual and the run stays on real l^2,
  // use it instead of evaluating r_U twice per edge. Mixed or complex-locus
  // objectives retain the general numerical path.
  const bool explicitConstraintsAreWholeResidual =
      !registerConstraints_.empty() && inputTargets_.empty() &&
      outputTargets_.empty() && inputBlocks_.empty() && outputBlocks_.empty();
  const auto explicitConstraintAscentDirection = [&]() {
    Eigen::VectorXcd ascent = Eigen::VectorXcd::Zero(edgeCount);
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> edgeIndices;
    for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex)
      edgeIndices[edgeKey(edges[edgeIndex])] = edgeIndex;
    const auto oneCells =
        ChainComplex::fromSpacetime(*spacetime_).kSimplexVertices(1);
    for (const auto &constraint : registerConstraints_) {
      EigenstateSynthesis synthesis(spacetime_, constraint.degree, metricSource_);
      const std::vector<double> gradient = synthesis.residualForPeriodsGradient(
          constraint.holes, constraint.target);
      if (gradient.size() != oneCells.size())
        throw std::runtime_error(
            "MultiCobordism: explicit r_U gradient has wrong edge count");
      for (std::size_t i = 0; i < oneCells.size(); ++i) {
        const auto &cell = oneCells[i];
        if (cell.size() != 2)
          continue;
        const auto found = edgeIndices.find(
            {std::min(cell[0], cell[1]), std::max(cell[0], cell[1])});
        if (found != edgeIndices.end())
          ascent[static_cast<Eigen::Index>(found->second)] +=
              complexd{gradient[i], 0.0};
      }
    }
    return ascent;
  };

  Eigen::VectorXcd descentDirection = Eigen::VectorXcd::Zero(edgeCount);
  // The phase's own descent direction, empty unless the injected objective
  // declares a phi dependence. Kept separate from `descentDirection` because
  // the two are displacements in DIFFERENT coordinates — z and phi are distinct
  // fields, and mixing them is the error the two-field split exists to prevent.
  Eigen::VectorXcd phaseDescentDirection = Eigen::VectorXcd::Zero(edgeCount);
  // Exact acceptance baseline at the CURRENT state rather than
  // objectiveTrace.back(). Joint mode assembles it from the exact gradients
  // already needed for its direction; the other modes recompute their scalar.
  // In the combined drive (`run`) the trace is accumulated from stage-1 deltas
  // and can drift from the true objective, so it is not a safe line-search
  // gate.

  double currentObjective = 0.0;
  double trialStepScale = stepScale;
  bool objectiveImproved = false;
  try {
    // The engine no longer knows which functional it is scoring. It assembles
    // the firewalled context; the injected objective supplies its own analytic
    // direction and, where it has assembled its scalar along the way, the exact
    // baseline the line search gates on. Any NUMERICALLY differentiated
    // register-residual term is applied here by weight rather than inside the
    // objective: differencing a scalar over edge coordinates is engine
    // machinery, and handing an objective a callable that did it would mean
    // handing it a closure over this node.
    ObjectiveDirectionContext directionContext;
    directionContext.scalar = objectiveContextFor(spacetime_);
    directionContext.edgeCount = edgeCount;
    if (carriedStateEnergyWeight_ != 0.0)
      directionContext.carriedStateEnergyGradient =
          carriedStateEnergyGradient(spacetime_);
    const auto objectiveDirection = objectiveSpec_->direction(directionContext);
    descentDirection += objectiveDirection.ascent;
    if (objectiveDirection.phaseAscent.size() ==
        static_cast<Eigen::Index>(edgeCount))
      phaseDescentDirection += objectiveDirection.phaseAscent;
    if (objectiveDirection.baselineComputed)
      currentObjective = objectiveDirection.baseline;
    const double numericalResidualWeight =
        objectiveSpec_->numericalRegisterResidualWeight(
            directionContext.scalar);
    if (numericalResidualWeight != 0.0) {
      if (useFiberResiduals_) {
        // #947: every fiber-mode term of rU has an analytic gradient through
        // the band's Riesz projector and the frame transfer; the numerical
        // path is not used for them.
        const ResidualGradient analytic = fiberModeAscent();
        descentDirection += numericalResidualWeight * analytic.lengths;
        if (fiberPhaseDescent_ && analytic.phases.size() == static_cast<Eigen::Index>(edgeCount))
          phaseDescentDirection += numericalResidualWeight * analytic.phases;
      } else if (explicitConstraintsAreWholeResidual && realSquaredLengthsOnly_)
        descentDirection += numericalResidualWeight *
                            explicitConstraintAscentDirection();
      else
        descentDirection += numericalResidualWeight *
                            scalarAscentDirection(
                                [&]() { return rU(spacetime_); });
    }

    // The pinned-region objective's direction adds on top of the bulk's, the
    // same way its terms add on top of the bulk's terms. Its own declared scope
    // decides which coordinates it moves; the bulk keeps moving all of them.
    if (pinnedObjectiveSpec_) {
      ObjectiveDirectionContext pinnedContext;
      pinnedContext.scalar =
          objectiveContextFor(spacetime_, pinnedObjectiveSpec_);
      pinnedContext.edgeCount = edgeCount;
      if (carriedStateEnergyWeight_ != 0.0)
        pinnedContext.carriedStateEnergyGradient =
            directionContext.carriedStateEnergyGradient;
      const auto pinnedDirection =
          pinnedObjectiveSpec_->direction(pinnedContext);
      descentDirection += pinnedDirection.ascent;
      // The line search gates on the exact composite scalar, so a baseline
      // assembled from the bulk alone would understate it. Only the sum of both
      // objectives is the number being descended.
      if (objectiveDirection.baselineComputed &&
          pinnedDirection.baselineComputed)
        currentObjective = objectiveDirection.baseline + pinnedDirection.baseline;
      else if (objectiveDirection.baselineComputed)
        currentObjective = objectiveFor(spacetime_);
      const double pinnedNumericalWeight =
          pinnedObjectiveSpec_->numericalRegisterResidualWeight(
              pinnedContext.scalar);
      if (pinnedNumericalWeight != 0.0) {
        if (explicitConstraintsAreWholeResidual && realSquaredLengthsOnly_)
          descentDirection += pinnedNumericalWeight *
                              explicitConstraintAscentDirection();
        else
          descentDirection += pinnedNumericalWeight *
                              scalarAscentDirection(
                                  [&]() { return rU(spacetime_); });
      }
    }

    // A real-locus run varies Re(l^2) only. Analytic objectives encode both
    // real derivatives in the complex direction, so remove the imaginary
    // coordinate after every contribution has been assembled.
    if (realSquaredLengthsOnly_)
      for (Eigen::Index i = 0; i < descentDirection.size(); ++i)
        descentDirection[i] = complexd{descentDirection[i].real(), 0.0};

    restoreEdgeLengths();
    // Pinning enters HERE and only here: a pinned edge keeps its resident squared
    // length while the rest of the complex relaxes around it. Zeroing the descent
    // component is the whole mechanism — no clamp, no projection after the fact,
    // no special case in the line search, which then simply has no reason to move
    // the coordinate. With no region declared this loop does nothing.
    if (!pinnedRegions_.empty())
      for (std::size_t edgeIndex = 0; edgeIndex < edgeCount; ++edgeIndex) {
        const auto key = edges[edgeIndex]->getKey();
        if (edgeIsPinned(key.first, key.second)) {
          descentDirection(edgeIndex) = complexd{0.0, 0.0};
          // A pinned edge is held in BOTH its fields. "Do not change these"
          // that froze the length while the phase drifted would be a pin in
          // name only.
          phaseDescentDirection(edgeIndex) = complexd{0.0, 0.0};
        }
      }
    // No coordinate can move, so every backtracking trial would evaluate the
    // unchanged global objective and fail the strict-improvement gate. This is
    // common at exact stationary points, when every selected term is disabled,
    // and when every edge that could move is pinned.
    if (descentDirection.squaredNorm() == 0.0 &&
        phaseDescentDirection.squaredNorm() == 0.0) {
      lastStage2Stationary_ = true;
      return false;
    }
    // An objective whose direction assembly already produced its scalar handed
    // it back as the direction's baseline, taken above. Only evaluate the
    // functional again when it did not, rather than re-deriving the same
    // gradients at the unchanged base geometry.
    if (!objectiveDirection.baselineComputed)
      currentObjective = fullObjective();
    // Absolute improvement threshold: the same tolerance has the same meaning
    // at every objective scale.
    const double improvementThreshold = tolerance;
    for (int lineSearchIndex = 0; lineSearchIndex < 24; ++lineSearchIndex) {
      // This is the key coordinate correction: derivatives are with respect to
      // z=l^2, so subtract the direction from z, then map back to Edge's stored
      // l on the continuous square-root branch. No component of z is projected.
      setSquaredLengths(squaredLengths -
                        trialStepScale * descentDirection);
      // ONE line search over BOTH fields: the same step scale moves z and phi
      // together and the same strict-improvement gate accepts or rejects the
      // pair. Two searches would let one field buy an improvement the other
      // paid for, and the accepted state would not be a descent of the whole
      // objective.
      setPhases(phases - trialStepScale * phaseDescentDirection);
      if (!geometryAdmissible(spacetime_)) {
        // Outside the closure of the allowable domain: not a configuration,
        // so it is not scored; the step is shortened exactly as a non-improving
        // trial is.
        CLOG(INFO_LEVEL, "Trial geometry not Kontsevich-Segal admissible; shortening the step.");
        trialStepScale *= 0.5;
        continue;
      }
      const double trialObjective = fullObjective();
      CLOG(INFO_LEVEL, "-----------------------------------");
      CLOG(INFO_LEVEL, "Trial objective: ", trialObjective);
      CLOG(INFO_LEVEL, "Current objective: ", currentObjective);
      CLOG(INFO_LEVEL, "Improvement threshold: ", improvementThreshold);
      CLOG(INFO_LEVEL, "Improvement: ", currentObjective - trialObjective);
      CLOG(INFO_LEVEL, "-----------------------------------");
      if ((currentObjective - trialObjective) >= improvementThreshold) {
        CLOG(INFO_LEVEL, (currentObjective - trialObjective), "<=", improvementThreshold);
        CLOG(INFO_LEVEL, "Improved.");
        objectiveTrace.push_back(trialObjective);
        stepScale = std::min(stepScale * 1.3, 1.0);
        objectiveImproved = true;
        // #776 solver-error indicator: the magnitude of the improvement this
        // geometric update actually banked (0 once the relaxation is
        // stationary). A base numerical quantity — nothing derived.
        lastStage2Improvement_ = std::abs(currentObjective - trialObjective);
        currentObjective = trialObjective;
        break;
      }
      CLOG(INFO_LEVEL, (currentObjective - trialObjective), ">", improvementThreshold);
      CLOG(INFO_LEVEL, "Did not improve.");
      trialStepScale *= 0.5;
    }
  } catch (...) {
    // The error still propagates loudly — it just does not take the geometry with
    // it. The throw comes from a TRIAL the line search had not accepted, so the
    // complex the caller still holds must be the one it had on entry, not a
    // half-applied step everything downstream would then read.
    restoreEdgeLengths();
    throw;
  }
  if (!objectiveImproved) {
    restoreEdgeLengths();
    lastStage2Stationary_ = true;
    lastStage2Improvement_ = 0.0;  // #776: stationary means zero solver error
    return false;
  }
  (void)beta;  // run/runStage2 synchronize this with reggeWeight_ before entry.
  return true;
}

int MultiCobordism::directedConeOut(HolePlacementStrategy strategy, int maxOpen) {
  if (registerDegrees_.empty()) return 0;
  constexpr int kMaxCandidates = 40;  // bound the scan; interior-first surfaces openers early
  constexpr int kProbeOpeners = 3;    // stop once a few openers are in hand
  const int registerDegree = registerDegrees_.front();
  auto spacetime = spacetime_;
  int opened = 0;
  for (int iteration = 0; iteration < maxOpen; ++iteration) {
    const auto holesBefore = emergentHoles(*spacetime, registerDegree);
    const std::size_t holeCountBefore = holesBefore.size();
    std::set<std::uint64_t> holeVertices;
    for (const auto &hole : holesBefore) holeVertices.insert(hole.begin(), hole.end());
    const auto boundary = boundaryFacetSet(*spacetime);

    std::vector<std::vector<std::uint64_t>> cells;
    for (const auto *simplex : spacetime->getTopSimplices())
      cells.push_back(simplex->topTuple());
    // Order interior-first (fewest boundary facets → hole-creators first); the secondary key
    // then places cells sharing vertices with the existing holes last (AdjacentHolesLast, a
    // separated register) or first (AdjacentHolesFirst, a clustered one).
    const auto orderKey = [&](const std::vector<std::uint64_t> &cell) {
      int boundaryFacets = 0;
      for (std::size_t i = 0; i < cell.size(); ++i) {
        std::vector<std::uint64_t> facet;
        for (std::size_t j = 0; j < cell.size(); ++j)
          if (j != i) facet.push_back(cell[j]);
        if (boundary.count(facet)) ++boundaryFacets;
      }
      int shared = 0;
      for (auto vertexId : cell)
        if (holeVertices.count(vertexId)) ++shared;
      return std::pair<int, int>(
          boundaryFacets,
          strategy == HolePlacementStrategy::AdjacentHolesFirst ? -shared : shared);
    };
    std::sort(cells.begin(), cells.end(),
              [&](const std::vector<std::uint64_t> &a,
                  const std::vector<std::uint64_t> &b) { return orderKey(a) < orderKey(b); });

    // Scored by the INJECTED objective, so topology changes when the functional
    // in force wants it and not otherwise. Surgery is the only topology-changing
    // mechanism the engine has — Pachner moves are bistellar and preserve the PL
    // homeomorphism type, hence the Betti numbers, and geometric relaxation
    // changes no topology at all — so this probe is where a higher b_k becomes
    // reachable. It is never required: an objective indifferent to topology
    // finds no candidate that lowers it and nothing is committed.
    const double baseObjective = objectiveFor(spacetime);
    double bestObjective = baseObjective;
    std::vector<std::uint64_t> bestCell;
    int candidatesScanned = 0;
    int openersScanned = 0;
    SurgicalCone cone(spacetime.get());
    for (const auto &cell : cells) {
      if (candidatesScanned++ >= kMaxCandidates) break;
      // `coneOut` is itself gated on the full manifold check, so a candidate that
      // survives it leaves a valid manifold-with-boundary — including when it
      // removed a pinned vertex, which is a legitimate topology change.
      if (!cone.coneOut(cell).first) continue;  // gate rejected; nothing applied
      const bool opensHole =
          emergentHoles(*spacetime, registerDegree).size() > holeCountBefore;
      if (opensHole) {
        const double candidateObjective = objectiveFor(spacetime);
        if (candidateObjective < bestObjective) {
          bestObjective = candidateObjective;
          bestCell = cell;
        }
        ++openersScanned;
      }
      cone.rollback();
      if (opensHole && openersScanned >= kProbeOpeners) break;
    }
    if (bestCell.empty()) break;  // no opener lowers the objective
    if (!cone.coneOut(bestCell).first) break;
    ++opened;
  }
  return opened;
}

int MultiCobordism::directedConeIn(int maxClose) {
  if (registerDegrees_.empty()) return 0;
  constexpr int kMaxCandidates = 40;
  const int registerDegree = registerDegrees_.front();
  auto spacetime = spacetime_;
  int closed = 0;
  for (int iteration = 0; iteration < maxClose; ++iteration) {
    const auto holesBefore = emergentHoles(*spacetime, registerDegree);
    const std::size_t holeCountBefore = holesBefore.size();
    if (holeCountBefore == 0) break;
    const auto boundary = boundaryFacetSet(*spacetime);

    // Cap facets: drop-one facets of the current holes that lie on the boundary — capping
    // one (a cone-in over it) closes that hole.
    std::set<std::vector<std::uint64_t>> seen;
    std::vector<std::vector<std::uint64_t>> capFacets;
    for (const auto &hole : holesBefore) {
      for (std::size_t i = 0; i < hole.size(); ++i) {
        std::vector<std::uint64_t> facet;
        for (std::size_t j = 0; j < hole.size(); ++j)
          if (j != i) facet.push_back(hole[j]);
        std::sort(facet.begin(), facet.end());
        if (boundary.count(facet) && seen.insert(facet).second) capFacets.push_back(facet);
      }
    }

    // Scored by the INJECTED objective, exactly as the cone-out probe is: a
    // hole closes when the functional in force is lowered by closing it.
    const double baseObjective = objectiveFor(spacetime);
    double bestObjective = baseObjective;
    std::vector<std::uint64_t> bestFacet;
    int candidatesScanned = 0;
    SurgicalCone cone(spacetime.get());
    for (const auto &facet : capFacets) {
      if (candidatesScanned++ >= kMaxCandidates) break;
      if (!cone.coneIn(facet).first) continue;
      if (emergentHoles(*spacetime, registerDegree).size() < holeCountBefore) {
        const double candidateObjective = objectiveFor(spacetime);
        if (candidateObjective < bestObjective) {
          bestObjective = candidateObjective;
          bestFacet = facet;
        }
      }
      cone.rollback();
    }
    if (bestFacet.empty()) break;  // no cap lowers the objective
    if (!cone.coneIn(bestFacet).first) break;
    ++closed;
  }
  return closed;
}

void MultiCobordism::buildStep(BuildAction action, int maxSteps, int nCandidateMoves,
                               double stage2Beta, int stage2MaxIters,
                               double stage2Alpha0,
                               HolePlacementStrategy holePlacementStrategy) {
  switch (action) {
    case BuildAction::Grow:
      runStage1(maxSteps, nCandidateMoves, /*growBoundaries=*/true);
      break;
    case BuildAction::Evolve:
      runStage1(maxSteps, nCandidateMoves, /*growBoundaries=*/false);
      break;
    case BuildAction::Relax:
      runStage2(stage2Beta, stage2MaxIters, stage2Alpha0);
      break;
    case BuildAction::ConeOut:
      (void)directedConeOut(holePlacementStrategy);
      break;
    case BuildAction::ConeIn:
      (void)directedConeIn();
      break;
  }
}

// ---- fiber-form boundary targets (#916) ----

double MultiCobordism::fiberResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                       const BoundaryFiber &target) const {
  if (target.images.cols() == 0)
    throw std::logic_error("MultiCobordism::fiberResidualOn: the fiber target has no images");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::fiberResidualOn: the fiber residual is read on the chain-level "
                           "Whitney pencil; this node uses the diagonal-weight metric");
  const double targetNorm = target.images.squaredNorm();
  if (!(targetNorm > 0.0))
    throw std::logic_error("MultiCobordism::fiberResidualOn: the fiber target is zero");
  if (!spacetime) return 1.0;  // no complex to read: the target leaks in full
  // A candidate geometry the pencil refuses (a singular dressed metric, a
  // branch or allowability failure, a cell outside the complex) cannot carry
  // the state: it scores as the full leak, exactly as a block with no emerged
  // register does under the period residual. Contract errors still propagate.
  BoundaryFiber read;
  try {
    const AssembledPencil assembled = PencilLayer::assemble({spacetime});
    if (assembled.dimension() < target.degree) return 1.0;
    const std::vector<int> idx = PencilLayer::indicesOf(assembled, target.degree, target.cells);
    if (idx.size() != target.cells.size()) return 1.0;
    const chainhodge::Contour contour =
        target.contour.nodes.empty() ? PencilLayer::bandContour(assembled, target.degree, 1)
                                     : target.contour;
    read = PencilLayer::readBoundaryFiber(assembled, target.degree, contour, target.cells);
  } catch (const std::runtime_error &) {
    return 1.0;
  } catch (const std::invalid_argument &) {
    return 1.0;
  }
  if (read.images.cols() == 0 || read.images.rows() != target.images.rows()) return 1.0;
  // Least-squares fit of the target images in the band's images on the cells.
  const Eigen::MatrixXcd coefficients = read.images.colPivHouseholderQr().solve(target.images);
  const double leak = (read.images * coefficients - target.images).squaredNorm();
  return leak / targetNorm;
}

std::shared_ptr<Spacetime> MultiCobordism::blockSubcomplexWithGeometry(
    const BoundaryBlock &block, const std::shared_ptr<Spacetime> &spacetime) {
  auto sub = spacetime->subcomplexWithinVertexSet(block.vertices);
  if (!sub) return nullptr;
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *> parentEdges;
  for (auto *edge : spacetime->getEdgeList()->toVector()) parentEdges.emplace(edgeKey(edge), edge);
  for (auto *edge : sub->getEdgeList()->toVector()) {
    const auto parent = parentEdges.find(edgeKey(edge));
    if (parent == parentEdges.end())
      throw std::logic_error("MultiCobordism::blockSubcomplexWithGeometry: a block edge is absent from the parent");
    edge->setLength(parent->second->getLength());
    edge->setPhase(parent->second->getPhase());
  }
  sub->materializeFacets();
  return sub;
}

double MultiCobordism::fiberResidualForBoundaryBlock(
    const BoundaryBlock &boundaryBlock, const std::shared_ptr<Spacetime> &spacetime) const {
  if (!boundaryBlock.fiber || boundaryBlock.fiber->images.cols() == 0)
    throw std::logic_error("MultiCobordism::fiberResidualForBoundaryBlock: the block carries no fiber target");
  return fiberResidualOn(blockSubcomplexWithGeometry(boundaryBlock, spacetime), *boundaryBlock.fiber);
}

// ---- two-body cobordism map (#941) ----

void MultiCobordism::attachInputFiber(std::size_t index, BoundaryFiber fiber,
                                      std::vector<std::vector<std::uint64_t>> cells) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::attachInputFiber: input block index out of range");
  if (fiber.images.cols() == 0)
    throw std::invalid_argument("MultiCobordism::attachInputFiber: the fiber has no images");
  if (cells.size() != static_cast<std::size_t>(fiber.images.rows()))
    throw std::invalid_argument("MultiCobordism::attachInputFiber: one attachment cell per fiber row");
  const ChainComplex K = ChainComplex::fromSpacetime(*spacetime_);
  std::set<std::vector<std::uint64_t>> live;
  for (auto c : K.kSimplexVertices(fiber.degree)) {
    std::sort(c.begin(), c.end());
    live.insert(c);
  }
  std::set<std::vector<std::uint64_t>> seen;
  for (auto &c : cells) {
    std::sort(c.begin(), c.end());
    if (!live.count(c))
      throw std::invalid_argument("MultiCobordism::attachInputFiber: an attachment cell is absent from the "
                                  "live complex at degree " + std::to_string(fiber.degree));
    if (!seen.insert(c).second)
      throw std::invalid_argument("MultiCobordism::attachInputFiber: an attachment cell is repeated");
  }
  for (std::size_t other = 0; other < inputBlocks_.size(); ++other) {
    if (other == index || !inputBlocks_[other].fiber) continue;
    for (const auto &c : inputBlocks_[other].fiber->cells)
      if (seen.count(c))
        throw std::invalid_argument("MultiCobordism::attachInputFiber: attachment cell overlaps input fiber " +
                                    std::to_string(other));
  }
  // Attaching a fiber to cells makes those cells the block's: the block's
  // region grows to contain them, so the block's own sub-complex reads them
  // (a cell outside the region would score as the full leak forever).
  for (const auto &c : cells)
    for (const std::uint64_t v : c) inputBlocks_[index].vertices.insert(v);
  fiber.cells = std::move(cells);
  inputBlocks_[index].fiber = std::move(fiber);
}

void MultiCobordism::setInputBlockRegion(std::size_t index, std::set<std::uint64_t> vertices) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::setInputBlockRegion: input block index out of range");
  if (inputBlocks_[index].fiber)
    for (const auto &c : inputBlocks_[index].fiber->cells)
      for (const std::uint64_t v : c)
        if (!vertices.count(v))
          throw std::invalid_argument("MultiCobordism::setInputBlockRegion: an attached fiber cell lies outside "
                                      "the region");
  inputBlocks_[index].vertices = std::move(vertices);
}

void MultiCobordism::setTwoBodyTarget(Eigen::MatrixXcd chi, bool choiDecomposed) {
  if (chi.rows() == 0 || chi.cols() == 0 || !(chi.squaredNorm() > 0.0))
    throw std::invalid_argument("MultiCobordism::setTwoBodyTarget: the target must be a nonzero matrix");
  twoBodyTarget_ = TwoBodyTarget{std::move(chi), choiDecomposed};
}

std::pair<const BoundaryFiber *, const BoundaryFiber *> MultiCobordism::attachedInputFibers() const {
  std::vector<const BoundaryFiber *> attached;
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0) attached.push_back(&*block.fiber);
  if (attached.size() != 2)
    throw std::logic_error("MultiCobordism: the two-body map needs exactly two attached input fibers; " +
                           std::to_string(attached.size()) + " found");
  return {attached[0], attached[1]};
}

chainhodge::TransferResult MultiCobordism::frameTransferOn(const std::shared_ptr<Spacetime> &spacetime,
                                                            const BoundaryFiber &A,
                                                            const BoundaryFiber &B) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism: the two-body map is read on the chain-level Whitney pencil; "
                           "this node uses the diagonal-weight metric");
  if (A.degree != B.degree)
    throw std::logic_error("MultiCobordism: the two attached fibers are at different degrees");
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  // The full frames on the attached cells: unit images (and unit dual images),
  // so the transfer is the coupling block of the whole between the two frames.
  auto frame = [&](const BoundaryFiber &f) {
    BoundaryFiber out;
    out.degree = f.degree;
    out.cells = f.cells;
    const Eigen::Index r = static_cast<Eigen::Index>(f.cells.size());
    out.images = Eigen::MatrixXcd::Identity(r, r);
    out.dualImages = Eigen::MatrixXcd::Identity(r, r);
    return out;
  };
  return PencilLayer::transfer(assembled, A.degree, frame(A), frame(B));
}

double MultiCobordism::twoBodyResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                         const TwoBodyTarget &target) const {
  const auto [A, B] = attachedInputFibers();
  if (!spacetime) return 1.0;
  Eigen::MatrixXcd T;
  try {
    T = frameTransferOn(spacetime, *A, *B).forward;
  } catch (const std::runtime_error &) {
    return 1.0;  // a refused geometry cannot carry the map: full leak
  } catch (const std::invalid_argument &) {
    return 1.0;
  }
  if (T.rows() != target.chi.rows() || T.cols() != target.chi.cols())
    throw std::logic_error("MultiCobordism: the two-body target is " + std::to_string(target.chi.rows()) + "x" +
                           std::to_string(target.chi.cols()) + " but the attached frames give " +
                           std::to_string(T.rows()) + "x" + std::to_string(T.cols()));
  // Projective Frobenius leak: min_c ||c T - chi||^2 / ||chi||^2 (the same in
  // the state and operator readings, vec being linear).
  const double tt = T.squaredNorm();
  if (!(tt > 0.0)) return 1.0;
  const complexd overlap = (T.conjugate().cwiseProduct(target.chi)).sum();
  const double leak = target.chi.squaredNorm() - std::norm(overlap) / tt;
  return std::max(0.0, leak / target.chi.squaredNorm());
}

double MultiCobordism::twoBodyResidual() const {
  if (!twoBodyTarget_) throw std::logic_error("MultiCobordism::twoBodyResidual: no two-body target");
  return twoBodyResidualOn(spacetime_, *twoBodyTarget_);
}

MultiCobordism::TwoBodyRead MultiCobordism::readTwoBody() const {
  const auto [A, B] = attachedInputFibers();
  const chainhodge::TransferResult transfer = frameTransferOn(spacetime_, *A, *B);
  TwoBodyRead read;
  read.choiDecomposed = twoBodyTarget_ ? twoBodyTarget_->choiDecomposed : true;
  read.transfer = transfer.forward;
  read.choiState = Eigen::Map<const Eigen::VectorXcd>(transfer.forward.data(), transfer.forward.size());
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(transfer.forward);
  const Eigen::VectorXd sv = svd.singularValues();
  read.singularValues.assign(sv.data(), sv.data() + sv.size());
  read.schmidtRank = 0;
  for (Eigen::Index i = 0; i < sv.size(); ++i)
    if (sv.size() > 0 && sv(i) > 1e-10 * sv(0)) ++read.schmidtRank;
  read.reversalResidual = transfer.reversalResidual;
  read.residual = twoBodyTarget_ ? twoBodyResidualOn(spacetime_, *twoBodyTarget_)
                                 : std::numeric_limits<double>::quiet_NaN();
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0)
      read.inputFiberResiduals.push_back(fiberResidualForBoundaryBlock(block, spacetime_));
  read.cellsA = A->cells;
  read.cellsB = B->cells;
  return read;
}

// ---- analytic gradients of the fiber-mode residuals (#947) ----

namespace {

/// (∂/∂Re, ∂/∂Im) of a real objective whose holomorphic sensitivity to the
/// coordinate is `dF` (that is, dObjective = 2 Re(dF · dcoordinate)).
complexd packHolomorphic(complexd dF) {
  return complexd{2.0 * dF.real(), -2.0 * dF.imag()};
}

/// Canonical C_1 index of every live edge (EdgeList order), by vertex pair.
std::vector<std::size_t> canonicalEdgeIndices(const Spacetime &spacetime, const ChainComplex &K) {
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> canonical;
  const auto cells = K.kSimplexVertices(1);
  for (std::size_t j = 0; j < cells.size(); ++j) canonical[{cells[j][0], cells[j][1]}] = j;
  std::vector<std::size_t> out;
  for (const auto *edge : spacetime.getEdgeList()->toVector()) {
    const auto found = canonical.find(edgeKey(edge));
    if (found == canonical.end())
      throw std::logic_error("MultiCobordism: a live edge is absent from the chain complex");
    out.push_back(found->second);
  }
  return out;
}

}  // namespace

MultiCobordism::ResidualGradient MultiCobordism::fiberResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime, const BoundaryFiber &target) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::fiberResidualGradientOn: read on the chain-level Whitney pencil");
  if (!spacetime) throw std::invalid_argument("MultiCobordism::fiberResidualGradientOn: null spacetime");
  const auto edges = spacetime->getEdgeList()->toVector();
  ResidualGradient gradient;
  gradient.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  if (target.degree == 0) gradient.phases = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  const std::vector<int> idx = PencilLayer::indicesOf(assembled, target.degree, target.cells);
  const chainhodge::Contour contour =
      target.contour.nodes.empty() ? PencilLayer::bandContour(assembled, target.degree, 1) : target.contour;
  const chainhodge::Band band = assembled.op->band(target.degree, contour);
  if (band.rank() == 0) return gradient;  // full leak everywhere: no descent direction
  // The least-squares fit on the cells: u = psi - Z_T c, r = |u|^2 / |psi|^2.
  Eigen::MatrixXcd ZT(static_cast<Eigen::Index>(idx.size()), band.rank());
  for (std::size_t i = 0; i < idx.size(); ++i) ZT.row(static_cast<Eigen::Index>(i)) = band.images.row(idx[i]);
  const Eigen::MatrixXcd c = ZT.colPivHouseholderQr().solve(target.images);
  const Eigen::MatrixXcd u = target.images - ZT * c;
  const double norm = target.images.squaredNorm();
  const chainhodge::BandDerivative::ResolventFrames frames =
      chainhodge::BandDerivative::resolventFrames(*assembled.op, target.degree, contour, band.frame);
  const std::vector<std::size_t> canonical = canonicalEdgeIndices(*spacetime, assembled.complex());
  // Holomorphic sensitivity: d r = 2 Re(dF · dcoord) with dF = -tr(u^H dZ_T c)/|psi|^2.
  const auto sensitivity = [&](const Eigen::MatrixXcd &dZ) {
    complexd dF(0.0, 0.0);
    for (std::size_t i = 0; i < idx.size(); ++i) {
      const Eigen::RowVectorXcd row = dZ.row(idx[i]) * c;  // 1 x columns of psi
      dF += (u.row(static_cast<Eigen::Index>(i)).conjugate().cwiseProduct(row)).sum();
    }
    return -dF / norm;
  };
  for (std::size_t e = 0; e < edges.size(); ++e) {
    gradient.lengths[static_cast<Eigen::Index>(e)] = packHolomorphic(sensitivity(
        chainhodge::BandDerivative::imagesLengthDerivative(*assembled.op, frames, band.images, canonical[e])));
    if (target.degree == 0)
      gradient.phases[static_cast<Eigen::Index>(e)] = packHolomorphic(sensitivity(
          chainhodge::BandDerivative::imagesPhaseDerivative(*assembled.op, frames, band.images, canonical[e])));
  }
  return gradient;
}

MultiCobordism::ResidualGradient MultiCobordism::twoBodyResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime, const TwoBodyTarget &target) const {
  const auto [A, B] = attachedInputFibers();
  if (!spacetime) throw std::invalid_argument("MultiCobordism::twoBodyResidualGradientOn: null spacetime");
  const auto edges = spacetime->getEdgeList()->toVector();
  ResidualGradient gradient;
  gradient.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  if (A->degree == 0) gradient.phases = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  const std::vector<int> ia = PencilLayer::indicesOf(assembled, A->degree, A->cells);
  const std::vector<int> ib = PencilLayer::indicesOf(assembled, B->degree, B->cells);
  const Eigen::MatrixXcd Atilde = PencilLayer::pencil(assembled, A->degree).A;
  const auto block = [&](const Eigen::MatrixXcd &full) {
    Eigen::MatrixXcd out(static_cast<Eigen::Index>(ia.size()), static_cast<Eigen::Index>(ib.size()));
    for (std::size_t i = 0; i < ia.size(); ++i)
      for (std::size_t j = 0; j < ib.size(); ++j)
        out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = full(ia[i], ib[j]);
    return out;
  };
  const Eigen::MatrixXcd T = block(Atilde);
  const double tt = T.squaredNorm();
  if (!(tt > 0.0)) return gradient;
  const double cc = target.chi.squaredNorm();
  const complexd overlap = (T.conjugate().cwiseProduct(target.chi)).sum();  // <T, chi>
  const std::vector<std::size_t> canonical = canonicalEdgeIndices(*spacetime, assembled.complex());
  // r = 1 - |<T,chi>|^2 / (|T|^2 |chi|^2). For a holomorphic dT:
  //   d|<T,chi>|^2 = 2 Re( <T,chi> <chi,dT> ),  d|T|^2 = 2 Re( <T,dT> ),
  // so dr = 2 Re(dF · dcoord) with
  //   dF = -( <T,chi> <chi,dT> |T|^2 - |<T,chi>|^2 <T,dT> ) / (|T|^4 |chi|^2).
  const auto sensitivity = [&](const Eigen::MatrixXcd &dA) {
    const Eigen::MatrixXcd dT = block(dA);
    const complexd s1 = (target.chi.conjugate().cwiseProduct(dT)).sum();  // <chi, dT>
    const complexd s2 = (T.conjugate().cwiseProduct(dT)).sum();           // <T, dT>
    return -(overlap * s1 * tt - std::norm(overlap) * s2) / (tt * tt * cc);
  };
  for (std::size_t e = 0; e < edges.size(); ++e) {
    gradient.lengths[static_cast<Eigen::Index>(e)] = packHolomorphic(sensitivity(
        chainhodge::BandDerivative::pencilOperatorLengthDerivative(*assembled.op, A->degree, canonical[e])));
    if (A->degree == 0)
      gradient.phases[static_cast<Eigen::Index>(e)] = packHolomorphic(sensitivity(
          chainhodge::BandDerivative::pencilOperatorPhaseDerivative(*assembled.op, A->degree, canonical[e])));
  }
  return gradient;
}

MultiCobordism::ResidualGradient MultiCobordism::fiberModeAscent() const {
  const auto edges = spacetime_->getEdgeList()->toVector();
  ResidualGradient total;
  total.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  total.phases = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> parentIndex;
  for (std::size_t e = 0; e < edges.size(); ++e) parentIndex[edgeKey(edges[e])] = e;
  const auto accumulate = [&](const ResidualGradient &g, const Spacetime &on, double weight) {
    const auto onEdges = on.getEdgeList()->toVector();
    for (std::size_t e = 0; e < onEdges.size(); ++e) {
      const auto found = parentIndex.find(edgeKey(onEdges[e]));
      if (found == parentIndex.end()) continue;
      total.lengths[static_cast<Eigen::Index>(found->second)] += weight * g.lengths[static_cast<Eigen::Index>(e)];
      if (g.phases.size() == static_cast<Eigen::Index>(onEdges.size()))
        total.phases[static_cast<Eigen::Index>(found->second)] += weight * g.phases[static_cast<Eigen::Index>(e)];
    }
  };
  if (wholeFiberTarget_) accumulate(fiberResidualGradientOn(spacetime_, *wholeFiberTarget_), *spacetime_, 1.0);
  for (const auto &block : inputBlocks_) {
    if (!block.fiber || block.fiber->images.cols() == 0) continue;
    auto sub = blockSubcomplexWithGeometry(block, spacetime_);
    if (!sub) continue;
    try {
      accumulate(fiberResidualGradientOn(sub, *block.fiber), *sub, inputResidualWeight_);
    } catch (const std::runtime_error &) {
      // a refused geometry has no descent direction (its residual is the full leak)
    } catch (const std::invalid_argument &) {
    }
  }
  if (twoBodyTarget_) {
    try {
      accumulate(twoBodyResidualGradientOn(spacetime_, *twoBodyTarget_), *spacetime_, 1.0);
    } catch (const std::runtime_error &) {
    } catch (const std::invalid_argument &) {
    }
  }
  return total;
}

void MultiCobordism::setWholeComplexFiberTarget(BoundaryFiber fiber) {
  if (fiber.images.cols() == 0)
    throw std::invalid_argument("MultiCobordism::setWholeComplexFiberTarget: the fiber has no images");
  if (fiber.cells.size() != static_cast<std::size_t>(fiber.images.rows()))
    throw std::invalid_argument("MultiCobordism::setWholeComplexFiberTarget: one image row per cell");
  for (auto &c : fiber.cells) std::sort(c.begin(), c.end());
  wholeFiberTarget_ = std::move(fiber);
}

double MultiCobordism::wholeComplexFiberResidual() const {
  if (!wholeFiberTarget_)
    throw std::logic_error("MultiCobordism::wholeComplexFiberResidual: no whole-complex fiber target");
  return fiberResidualOn(spacetime_, *wholeFiberTarget_);
}

BoundaryFiber MultiCobordism::readWholeComplexFiber(const chainhodge::Contour *contour,
                                                    double kappa) const {
  if (!wholeFiberTarget_)
    throw std::logic_error("MultiCobordism::readWholeComplexFiber: no whole-complex fiber target");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::readWholeComplexFiber: read on the chain-level Whitney pencil; "
                           "this node uses the diagonal-weight metric");
  const AssembledPencil assembled = PencilLayer::assemble({spacetime_});
  const BoundaryFiber &target = *wholeFiberTarget_;
  const chainhodge::Contour chosen =
      contour ? *contour
              : (target.contour.nodes.empty() ? PencilLayer::bandContour(assembled, target.degree, 1)
                                              : target.contour);
  return PencilLayer::readBoundaryFiber(assembled, target.degree, chosen, target.cells, kappa);
}

double MultiCobordism::fiberResidualForInputBlock(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::fiberResidualForInputBlock: input block index out of range");
  return fiberResidualForBoundaryBlock(inputBlocks_[index], spacetime_);
}

std::shared_ptr<Spacetime> MultiCobordism::seedSimplex(int dimension, bool balancedEdges) {
  using namespace ::tessera::spacetime;
  if (dimension < 1)
    throw std::invalid_argument("MultiCobordism::seedSimplex: dimension must be at least one");
  auto metric = std::make_shared<Metric>(true, Signature(dimension, SignatureType::Lorentzian));
  std::shared_ptr<Topology> topology = std::make_shared<SolidSimplex>(dimension);
  auto host = std::make_shared<Spacetime>(metric, SpacetimeType::CDT, 1.0, 1.0,
                                          Foliation::PREFERRED, topology);
  host->build();
  // #690: the wiring mode is stamped before ANY growth, and the seed's own
  // uniform |l^2| = 1 edges honor it too (balanced: l = sqrt(1/2)*(1+i)).
  host->setBalancedEdgeWiring(balancedEdges);
  for (auto *edge : host->getEdgeList()->toVector())
    edge->setLength(balancedEdges ? Spacetime::balancedLength(1.0) : std::sqrt(complexd(1.0, 0.0)));
  return host;
}

void MultiCobordism::setInputFiber(std::size_t index, BoundaryFiber fiber) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::setInputFiber: input block index out of range");
  inputBlocks_[index].fiber = std::move(fiber);
}

void MultiCobordism::setOutputFiber(std::size_t index, BoundaryFiber fiber) {
  if (index >= outputBlocks_.size())
    throw std::out_of_range("MultiCobordism::setOutputFiber: output block index out of range");
  outputBlocks_[index].fiber = std::move(fiber);
}

const std::optional<BoundaryFiber> &MultiCobordism::inputFiber(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::inputFiber: input block index out of range");
  return inputBlocks_[index].fiber;
}

const std::optional<BoundaryFiber> &MultiCobordism::outputFiber(std::size_t index) const {
  if (index >= outputBlocks_.size())
    throw std::out_of_range("MultiCobordism::outputFiber: output block index out of range");
  return outputBlocks_[index].fiber;
}

MultiCobordism::FixedBoundaryEigenstateResult MultiCobordism::pinInputFibers(
    int degree, double epsilon, int restarts, int maxGrowth, std::uint64_t seed, int maxIterations) {
  if (inputBlocks_.size() != 2 || !inputBlocks_[0].fiber || !inputBlocks_[1].fiber)
    throw std::invalid_argument(
        "MultiCobordism::pinInputFibers: exactly two input blocks carrying fibers are required");
  const BoundaryFiber &A = *inputBlocks_[0].fiber;
  const BoundaryFiber &B = *inputBlocks_[1].fiber;
  if (A.degree != degree || B.degree != degree)
    throw std::invalid_argument("MultiCobordism::pinInputFibers: the input fibers are not at degree " +
                                std::to_string(degree));
  if (A.rank() != 1 || B.rank() != 1)
    throw std::invalid_argument("MultiCobordism::pinInputFibers: the input fibers have ranks " +
                                std::to_string(A.rank()) + " and " + std::to_string(B.rank()) +
                                "; the fixed-boundary fit pins one state, so only rank-one fibers are "
                                "pinned (a joint multi-column fit is not approximated column by column)");
  std::vector<std::vector<std::uint64_t>> support;
  std::vector<complexd> target;
  std::set<std::vector<std::uint64_t>> seen;
  auto append = [&](const BoundaryFiber &f, const char *name) {
    for (Eigen::Index r = 0; r < f.images.rows(); ++r) {
      std::vector<std::uint64_t> cell;
      for (const auto v : f.cells[static_cast<std::size_t>(r)]) cell.push_back(static_cast<std::uint64_t>(v));
      std::sort(cell.begin(), cell.end());
      if (!seen.insert(cell).second)
        throw std::invalid_argument(std::string("MultiCobordism::pinInputFibers: ") + name +
                                    " overlaps the other input fiber on a boundary cell; the two inputs "
                                    "are the disjoint components of the boundary");
      support.push_back(std::move(cell));
      target.push_back(f.images(r, 0));
    }
  };
  append(A, "input fiber 0");
  append(B, "input fiber 1");
  return relaxFixedBoundaryEigenstate(degree, std::move(support), std::move(target), epsilon, restarts,
                                      maxGrowth, seed, maxIterations);
}

BoundaryFiber MultiCobordism::readOutputFiber(std::size_t index, int degree,
                                              const chainhodge::Contour *contour, double kappa) {
  if (index >= outputBlocks_.size())
    throw std::out_of_range("MultiCobordism::readOutputFiber: output block index out of range");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::readOutputFiber: the fiber form of a target is read on the "
                           "chain-level Whitney pencil; this node uses the diagonal-weight metric");
  const AssembledPencil assembled = PencilLayer::assemble({spacetime_});
  const std::vector<std::uint64_t> region(outputBlocks_[index].vertices.begin(),
                                          outputBlocks_[index].vertices.end());
  const std::vector<int> idx = PencilLayer::cellsWithin(assembled, degree, region);
  if (idx.empty())
    throw std::invalid_argument("MultiCobordism::readOutputFiber: the output block carries no degree-" +
                                std::to_string(degree) + " cell");
  const auto cells = assembled.complex().kSimplexVertices(degree);
  std::vector<std::vector<std::uint64_t>> blockCells;
  for (const int j : idx) blockCells.push_back(cells[static_cast<std::size_t>(j)]);
  const chainhodge::Contour chosen =
      contour ? *contour : PencilLayer::harmonicContour(assembled, degree);
  BoundaryFiber fiber = PencilLayer::readBoundaryFiber(assembled, degree, chosen, blockCells, kappa);
  outputBlocks_[index].fiber = fiber;
  return fiber;
}

}  // namespace tessera::cobordism
