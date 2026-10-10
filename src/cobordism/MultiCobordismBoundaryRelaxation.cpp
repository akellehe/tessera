// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the boundary
// relaxations: the fixed-boundary eigenstate, boundary-state pairs and
// whole-complex readout targets. One of the translation units that define
// the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "MultiCobordismInternal.h"

namespace tessera::cobordism {

namespace {

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

  auto boundary = Spacetime::fromVertexTuples(
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
  auto boundary = Spacetime::fromVertexTuples(spacetime->getDimensions() - 1,
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

/// Exact linear readout constraints on every witness: chain `r` paired with
/// witness `j` must equal `targets[j][r]`. The auxiliary block of each witness
/// is parametrized on the affine solution set of its readout system, so the
/// constraints hold exactly at every iterate and no penalty enters.
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

/// A warm start for one relaxation pass: the previous pass's witnesses keyed by
/// cell, together with the live edge geometry. Cells absent from the live
/// complex, i.e. created by growth, start at zero. The warm start is descended
/// first and the remaining restarts are drawn at random.
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
    // The readout matrix over all cells, in canonical cell order.
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

  // Isolated-boundary eigenstate check on every nonzero component
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

}  // namespace tessera::cobordism
