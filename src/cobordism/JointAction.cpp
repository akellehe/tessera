// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/JointAction.h"

#include <algorithm>
#include <map>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "chainhodge/CovariantChainHodge.h"
#include "cobordism/ChainComplex.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using ::tessera::MatterConfiguration;
using ::tessera::simulations::ReggeSolver;
using complexd = std::complex<double>;

namespace {

/// The unordered vertex pair of an edge, as the key both the canonical cell
/// list and the mesh's edge list are indexed by.
std::pair<std::uint64_t, std::uint64_t> pairKey(std::uint64_t a,
                                                std::uint64_t b) {
  return a <= b ? std::make_pair(a, b) : std::make_pair(b, a);
}

/// A flat row-major matrix lifted into Eigen. An empty input gives a 0x0
/// matrix, which every consumer below treats as "no cell of this degree".
Eigen::MatrixXcd toMatrix(const std::vector<complexd> &flat, std::size_t order) {
  Eigen::MatrixXcd matrix(static_cast<Eigen::Index>(order),
                          static_cast<Eigen::Index>(order));
  if (order == 0) return matrix;
  for (std::size_t i = 0; i < order; ++i)
    for (std::size_t j = 0; j < order; ++j)
      matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * order + j];
  return matrix;
}

/// An Eigen matrix flattened back to the row-major layout the public interface
/// reports matrices in.
std::vector<complexd> toFlat(const Eigen::MatrixXcd &matrix) {
  const auto rows = static_cast<std::size_t>(matrix.rows());
  const auto columns = static_cast<std::size_t>(matrix.cols());
  std::vector<complexd> flat(rows * columns, complexd{0.0, 0.0});
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < columns; ++j)
      flat[i * columns + j] = matrix(static_cast<Eigen::Index>(i),
                                     static_cast<Eigen::Index>(j));
  return flat;
}

/// \f$ \operatorname{tr}(AB) = \sum_{ij}A_{ij}B_{ji} \f$, formed without
/// building the product.
complexd traceOfProduct(const Eigen::MatrixXcd &a, const Eigen::MatrixXcd &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols() || a.rows() == 0)
    return complexd{0.0, 0.0};
  complexd trace{0.0, 0.0};
  for (Eigen::Index i = 0; i < a.rows(); ++i)
    for (Eigen::Index j = 0; j < a.cols(); ++j) trace += a(i, j) * b(j, i);
  return trace;
}

/// Whether a query needs the carrier operator assembled.
///
/// The operator enters only through the matter term and the spectral
/// constraints, so an action of the Regge and face-holonomy terms alone never
/// pays for the Whitney pencil. The test is on the declaration rather than on
/// an assembled matrix, so the decision is made before the cost is incurred.
bool carrierIsNeeded(const JointActionDeclaration &declaration) {
  if (declaration.matterWeight != 0.0 && !declaration.covariance.empty())
    return true;
  return !declaration.momentConstraints.empty();
}

}  // namespace

// ------------------------------------------------------------------ workspace

/// The geometry-derived quantities every stationarity query needs, assembled
/// once per query rather than once per edge: the canonical cell lists, the
/// mesh-to-canonical edge correspondence, the carrier operator, and the
/// multiplier-weighted matrix the operator derivatives are contracted against.
///
/// Held in an anonymous-namespace struct rather than cached on the instance
/// because the geometry is the caller's to move between queries: a solver
/// writes a new squared length or a new connection phase and asks again, and a
/// cached operator would then answer for the previous point.
namespace {

struct ActionWorkspace {
  ChainComplex complex;
  /// The operator the carrier and every one of its derivatives is read from.
  /// One instance per query rather than one per edge: the Whitney pencil's
  /// factorizations are cached on the instance, so sharing it means the pencil
  /// is assembled once for a whole per-edge sweep.
  HodgeLaplacian hodge;
  /// The mesh's edges, in `getEdgeList()` order.
  std::vector<::tessera::mesh::Edge *> edges;
  /// For each mesh edge, its index among the canonical degree-one cells, or
  /// -1 for an edge the chain complex does not carry.
  std::vector<long long> canonicalOfEdge;
  /// For each mesh edge, \f$ +1 \f$ when its stored source-to-target
  /// orientation is the canonical one (ascending vertex id) and \f$ -1 \f$
  /// otherwise. The link stationarity is reported on the stored orientation and
  /// the current is odd under reversal, so this is the sign that relates the two.
  std::vector<double> storedSign;
  /// The canonical links \f$ U_e \in \mathbb{C}^{*} \f$, in canonical
  /// degree-one cell order.
  std::vector<complexd> links;
  /// The face holonomies \f$ \mathcal F_\tau \f$, in canonical degree-two cell
  /// order.
  std::vector<complexd> holonomies;
  /// The carrier operator \f$ h_k(z,U) \f$.
  Eigen::MatrixXcd carrier;
  /// The number of \f$ k \f$-cells.
  std::size_t carrierOrder = 0;

  ActionWorkspace(const std::shared_ptr<Spacetime> &spacetime,
                  int carrierDegree,
                  HodgeLaplacian::MetricSource metricSource,
                  bool wantCarrier);
};

ActionWorkspace::ActionWorkspace(const std::shared_ptr<Spacetime> &spacetime,
                                 int carrierDegree,
                                 HodgeLaplacian::MetricSource metricSource,
                                 bool wantCarrier)
    : complex(ChainComplex::fromSpacetime(*spacetime)),
      hodge(spacetime, HodgeLaplacian::defaultWeightConvention(),
            metricSource) {
  if (spacetime->getEdgeList()) edges = spacetime->getEdgeList()->toVector();

  const auto canonicalEdges = complex.kSimplexVertices(1);
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> indexOfPair;
  for (std::size_t index = 0; index < canonicalEdges.size(); ++index)
    if (canonicalEdges[index].size() == 2)
      indexOfPair[pairKey(canonicalEdges[index][0], canonicalEdges[index][1])] =
          index;

  canonicalOfEdge.assign(edges.size(), -1);
  storedSign.assign(edges.size(), 1.0);
  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto *edge = edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const std::uint64_t source = edge->getSource()->getId();
    const std::uint64_t target = edge->getTarget()->getId();
    storedSign[edgeIndex] = source < target ? 1.0 : -1.0;
    const auto found = indexOfPair.find(pairKey(source, target));
    if (found != indexOfPair.end())
      canonicalOfEdge[edgeIndex] = static_cast<long long>(found->second);
  }

  // The links come from the framework's own C* connection adapter, so the
  // reference orientation, the inverse convention U_yx = U_xy^{-1} and the
  // treatment of the non-compact component are the ones every other covariant
  // consumer uses.
  if (!canonicalEdges.empty())
    links = chainhodge::Connection::fromSpacetime(*spacetime, complex).links();

  // F_tau = prod_e U_e^{eps_tau e} over the incidences of the boundary map,
  // formed as an ordered product of links and their inverses. No sum of phases
  // and no logarithm is taken.
  if (complex.dimension() >= 2) {
    holonomies.assign(complex.numSimplices(2), complexd{1.0, 0.0});
    for (const auto &entry : complex.boundaryEntries(2)) {
      const auto row = static_cast<std::size_t>(entry.row);
      const auto column = static_cast<std::size_t>(entry.column);
      if (row >= links.size() || column >= holonomies.size()) continue;
      const complexd link = links[row];
      holonomies[column] *= entry.value > 0 ? link : complexd{1.0, 0.0} / link;
    }
  }

  if (!wantCarrier || carrierDegree < 0 ||
      carrierDegree > complex.dimension())
    return;
  carrierOrder = complex.numSimplices(carrierDegree);
  const auto flat = hodge.laplacian(carrierDegree, /*metric=*/true);
  if (flat.size() != carrierOrder * carrierOrder) {
    carrierOrder = 0;
    return;
  }
  carrier = toMatrix(flat, carrierOrder);
}

}  // namespace

// ------------------------------------------------------------------ the class

JointAction::JointAction(std::shared_ptr<Spacetime> spacetime,
                         JointActionDeclaration declaration)
    : spacetime_(std::move(spacetime)), declaration_(std::move(declaration)) {
  if (!spacetime_)
    throw std::invalid_argument(
        "JointAction: the action is defined over a complex; the spacetime is "
        "null");
  if (declaration_.carrierDegree < 0)
    throw std::invalid_argument(
        "JointAction: the carrier degree is a simplicial degree and must be "
        "non-negative; got " +
        std::to_string(declaration_.carrierDegree));
  for (const auto &constraint : declaration_.momentConstraints)
    if (constraint.order < 1)
      throw std::invalid_argument(
          "JointAction: a spectral moment order is the power j >= 1 of "
          "p_j(h) = tr(h^j); got " +
          std::to_string(constraint.order));
  if (!declaration_.covariance.empty()) {
    const ChainComplex complex = ChainComplex::fromSpacetime(*spacetime_);
    const std::size_t order =
        declaration_.carrierDegree <= complex.dimension()
            ? complex.numSimplices(declaration_.carrierDegree)
            : std::size_t{0};
    if (declaration_.covariance.size() != order * order)
      throw std::invalid_argument(
          "JointAction: the covariance is a square matrix over the " +
          std::to_string(order) + " cells of degree " +
          std::to_string(declaration_.carrierDegree) + "; got " +
          std::to_string(declaration_.covariance.size()) + " entries");
  }
}

void JointAction::setMultipliers(const std::vector<complexd> &multipliers) {
  if (multipliers.size() != declaration_.momentConstraints.size())
    throw std::invalid_argument(
        "JointAction::setMultipliers: one multiplier per declared constraint "
        "is required; got " +
        std::to_string(multipliers.size()) + " for " +
        std::to_string(declaration_.momentConstraints.size()) +
        " constraints");
  for (std::size_t index = 0; index < multipliers.size(); ++index)
    declaration_.momentConstraints[index].multiplier = multipliers[index];
}

std::vector<complexd> JointAction::multipliers() const {
  std::vector<complexd> values;
  values.reserve(declaration_.momentConstraints.size());
  for (const auto &constraint : declaration_.momentConstraints)
    values.push_back(constraint.multiplier);
  return values;
}

std::size_t JointAction::edgeCount() const {
  if (!spacetime_ || !spacetime_->getEdgeList()) return 0;
  return spacetime_->getEdgeList()->toVector().size();
}

std::vector<std::string> JointAction::termNames() {
  return {"regge", "holonomy", "matter", "spectral"};
}

std::vector<complexd> JointAction::carrierOperator() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  return toFlat(workspace.carrier);
}

std::vector<complexd> JointAction::faceHolonomies() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  return workspace.holonomies;
}

std::vector<complexd> JointAction::powerSums() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  std::vector<complexd> sums;
  sums.reserve(declaration_.momentConstraints.size());
  for (const auto &constraint : declaration_.momentConstraints) {
    if (workspace.carrierOrder == 0) {
      sums.emplace_back(0.0, 0.0);
      continue;
    }
    // Repeated multiplication rather than an eigendecomposition: the power sum
    // is defined for a defective operator, and forming it this way needs no
    // eigenvalue ordering and no similarity to diagonal form.
    Eigen::MatrixXcd power = workspace.carrier;
    for (int step = 1; step < constraint.order; ++step)
      power = power * workspace.carrier;
    sums.push_back(power.trace());
  }
  return sums;
}

std::vector<complexd> JointAction::momentResiduals() const {
  auto residuals = powerSums();
  for (std::size_t index = 0; index < residuals.size(); ++index)
    residuals[index] -= declaration_.momentConstraints[index].target;
  return residuals;
}

std::complex<double> JointAction::reggeTerm() const {
  if (declaration_.gravitationalWeight == 0.0) return complexd{0.0, 0.0};
  return declaration_.gravitationalWeight *
         ReggeSolver(spacetime_, MatterConfiguration()).dualReggeAction();
}

std::complex<double> JointAction::holonomyTerm() const {
  if (declaration_.holonomyWeight == 0.0) return complexd{0.0, 0.0};
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  complexd sum{0.0, 0.0};
  for (const complexd &holonomy : workspace.holonomies)
    sum += complexd{1.0, 0.0} -
           0.5 * (holonomy + complexd{1.0, 0.0} / holonomy);
  return declaration_.holonomyWeight * sum;
}

std::complex<double> JointAction::matterTerm() const {
  if (declaration_.matterWeight == 0.0 || declaration_.covariance.empty())
    return complexd{0.0, 0.0};
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  if (workspace.carrierOrder == 0) return complexd{0.0, 0.0};
  const Eigen::MatrixXcd gamma =
      toMatrix(declaration_.covariance, workspace.carrierOrder);
  return declaration_.matterWeight * traceOfProduct(gamma, workspace.carrier);
}

std::complex<double> JointAction::spectralTerm() const {
  const auto residuals = momentResiduals();
  complexd sum{0.0, 0.0};
  for (std::size_t index = 0; index < residuals.size(); ++index)
    sum += declaration_.momentConstraints[index].multiplier * residuals[index];
  return sum;
}

std::complex<double> JointAction::value() const {
  return reggeTerm() + holonomyTerm() + matterTerm() + spectralTerm();
}

namespace {

/// The matrix every operator derivative is contracted against in the
/// stationarity equations: \f$ A = w_M\Gamma + \sum_j \xi_j\,j\,h^{j-1} \f$.
///
/// Both the matter term and the spectral term are linear in \f$ h \f$'s
/// derivative through a trace, so assembling their coefficient once turns the
/// per-edge work into a single contraction instead of one per term.
Eigen::MatrixXcd contractionMatrix(const JointActionDeclaration &declaration,
                                   const Eigen::MatrixXcd &carrier,
                                   std::size_t order) {
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(order), static_cast<Eigen::Index>(order));
  if (order == 0) return matrix;
  if (declaration.matterWeight != 0.0 && !declaration.covariance.empty())
    matrix += declaration.matterWeight * toMatrix(declaration.covariance, order);
  for (const auto &constraint : declaration.momentConstraints) {
    if (constraint.multiplier == complexd{0.0, 0.0}) continue;
    // d tr(h^j) = j tr(h^{j-1} dh), the cyclic identity, exact for every
    // matrix including a defective one.
    Eigen::MatrixXcd power = Eigen::MatrixXcd::Identity(
        static_cast<Eigen::Index>(order), static_cast<Eigen::Index>(order));
    for (int step = 1; step < constraint.order; ++step) power = power * carrier;
    matrix += constraint.multiplier *
              static_cast<double>(constraint.order) * power;
  }
  return matrix;
}

}  // namespace

std::vector<complexd> JointAction::lengthStationarity() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  carrierIsNeeded(declaration_));
  const std::size_t edges = workspace.edges.size();
  std::vector<complexd> stationarity(edges, complexd{0.0, 0.0});

  if (declaration_.gravitationalWeight != 0.0) {
    const auto reggeGradient =
        ReggeSolver(spacetime_, MatterConfiguration()).actionGradientExact();
    for (std::size_t edgeIndex = 0;
         edgeIndex < edges && edgeIndex < reggeGradient.size(); ++edgeIndex)
      stationarity[edgeIndex] +=
          declaration_.gravitationalWeight * reggeGradient[edgeIndex];
  }

  const Eigen::MatrixXcd contraction = contractionMatrix(
      declaration_, workspace.carrier, workspace.carrierOrder);
  if (workspace.carrierOrder == 0 || contraction.isZero(0.0))
    return stationarity;

  for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
    const auto *edge = workspace.edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const auto derivative = workspace.hodge.laplacianGradient(
        declaration_.carrierDegree, edge->getSource()->getId(),
        edge->getTarget()->getId());
    if (derivative.size() != workspace.carrierOrder * workspace.carrierOrder)
      continue;
    stationarity[edgeIndex] += traceOfProduct(
        contraction, toMatrix(derivative, workspace.carrierOrder));
  }
  return stationarity;
}

namespace {

/// The canonical Maurer-Cartan derivative of the face-holonomy term,
/// \f$ U_e\,\partial S_{\rm hol}/\partial U_e
///     = -\tfrac12\sum_\tau \epsilon_{\tau e}
///       (\mathcal F_\tau-\mathcal F_\tau^{-1}) \f$,
/// one entry per canonical degree-one cell.
///
/// Exact in closed form: \f$ \mathcal F_\tau \f$ is a Laurent monomial in the
/// links, so \f$ U_e\,\partial\mathcal F_\tau/\partial U_e
/// = \epsilon_{\tau e}\mathcal F_\tau \f$ and
/// \f$ U_e\,\partial\mathcal F_\tau^{-1}/\partial U_e
/// = -\epsilon_{\tau e}\mathcal F_\tau^{-1} \f$.
std::vector<complexd> holonomyLinkDerivative(const ActionWorkspace &workspace,
                                             double weight) {
  std::vector<complexd> derivative(workspace.links.size(), complexd{0.0, 0.0});
  if (weight == 0.0 || workspace.complex.dimension() < 2) return derivative;
  for (const auto &entry : workspace.complex.boundaryEntries(2)) {
    const auto row = static_cast<std::size_t>(entry.row);
    const auto column = static_cast<std::size_t>(entry.column);
    if (row >= derivative.size() || column >= workspace.holonomies.size())
      continue;
    const complexd holonomy = workspace.holonomies[column];
    derivative[row] += -0.5 * weight * static_cast<double>(entry.value) *
                       (holonomy - complexd{1.0, 0.0} / holonomy);
  }
  return derivative;
}

}  // namespace

std::vector<complexd> JointAction::linkStationarity() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  carrierIsNeeded(declaration_));
  const std::size_t edges = workspace.edges.size();
  std::vector<complexd> stationarity(edges, complexd{0.0, 0.0});

  const auto holonomyPart =
      holonomyLinkDerivative(workspace, declaration_.holonomyWeight);
  for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
    const long long canonical = workspace.canonicalOfEdge[edgeIndex];
    if (canonical < 0) continue;
    stationarity[edgeIndex] +=
        workspace.storedSign[edgeIndex] *
        holonomyPart[static_cast<std::size_t>(canonical)];
  }

  const Eigen::MatrixXcd contraction = contractionMatrix(
      declaration_, workspace.carrier, workspace.carrierOrder);
  if (workspace.carrierOrder == 0 || contraction.isZero(0.0))
    return stationarity;

  for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
    const auto *edge = workspace.edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const auto derivative = workspace.hodge.laplacianPhaseGradient(
        declaration_.carrierDegree, edge->getSource()->getId(),
        edge->getTarget()->getId());
    if (derivative.size() != workspace.carrierOrder * workspace.carrierOrder)
      continue;
    // U d/dU = -i d/dphi on the canonical link U = e^{i phi}: a relation
    // between derivatives, which fixes no branch of the logarithm. The stored
    // orientation carries the same current up to the sign that relates it to
    // the canonical one, since j_yx = -j_xy.
    const complexd canonicalPart =
        complexd{0.0, -1.0} *
        traceOfProduct(contraction, toMatrix(derivative, workspace.carrierOrder));
    stationarity[edgeIndex] += workspace.storedSign[edgeIndex] * canonicalPart;
  }
  return stationarity;
}

namespace {

/// The same action with only the carried state's channel left on: the
/// geometric coefficients set to zero, the constraints dropped, and the matter
/// coefficient set to one. Its stationarity vectors are then exactly the
/// Hellmann-Feynman forces, assembled by the same code path the full equations
/// use rather than by a second implementation of the same sweep.
JointActionDeclaration forceOnlyDeclaration(
    const JointActionDeclaration &declaration) {
  JointActionDeclaration forceOnly = declaration;
  forceOnly.gravitationalWeight = 0.0;
  forceOnly.holonomyWeight = 0.0;
  forceOnly.matterWeight = 1.0;
  forceOnly.momentConstraints.clear();
  return forceOnly;
}

}  // namespace

std::vector<complexd> JointAction::hellmannFeynmanLengthForce() const {
  return JointAction(spacetime_, forceOnlyDeclaration(declaration_))
      .lengthStationarity();
}

std::vector<complexd> JointAction::hellmannFeynmanLinkForce() const {
  return JointAction(spacetime_, forceOnlyDeclaration(declaration_))
      .linkStationarity();
}

std::vector<complexd> JointAction::occupationNumbers() const {
  if (declaration_.covariance.empty()) return {};
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  const std::size_t order =
      declaration_.carrierDegree <= workspace.complex.dimension()
          ? workspace.complex.numSimplices(declaration_.carrierDegree)
          : std::size_t{0};
  std::vector<complexd> occupations;
  if (declaration_.covariance.size() != order * order) return occupations;
  occupations.reserve(order);
  for (std::size_t cell = 0; cell < order; ++cell)
    occupations.push_back(declaration_.covariance[cell * order + cell]);
  return occupations;
}

std::vector<complexd> JointAction::wardCurrentDivergence() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  const std::size_t vertices = workspace.complex.numSimplices(0);
  std::vector<complexd> divergence(vertices, complexd{0.0, 0.0});
  if (vertices == 0) return divergence;

  // The current on the canonical orientation, which is the orientation the
  // boundary map's incidences refer to.
  const auto stored = linkStationarity();
  std::vector<complexd> canonical(workspace.links.size(), complexd{0.0, 0.0});
  for (std::size_t edgeIndex = 0; edgeIndex < stored.size(); ++edgeIndex) {
    const long long index = workspace.canonicalOfEdge[edgeIndex];
    if (index < 0) continue;
    canonical[static_cast<std::size_t>(index)] =
        workspace.storedSign[edgeIndex] * stored[edgeIndex];
  }

  for (const auto &entry : workspace.complex.boundaryEntries(1)) {
    const auto row = static_cast<std::size_t>(entry.row);
    const auto column = static_cast<std::size_t>(entry.column);
    if (row >= divergence.size() || column >= canonical.size()) continue;
    divergence[row] += static_cast<double>(entry.value) * canonical[column];
  }
  return divergence;
}

std::vector<complexd> JointAction::stationarityResidual() const {
  const auto lengths = lengthStationarity();
  const auto links = linkStationarity();
  const auto moments = momentResiduals();
  std::vector<complexd> residual;
  residual.reserve(lengths.size() + links.size() + moments.size());
  residual.insert(residual.end(), lengths.begin(), lengths.end());
  residual.insert(residual.end(), links.begin(), links.end());
  residual.insert(residual.end(), moments.begin(), moments.end());
  return residual;
}

double JointAction::stationarityResidualNorm() const {
  double squared = 0.0;
  for (const complexd &component : stationarityResidual())
    squared += std::norm(component);
  return std::sqrt(squared);
}

std::vector<complexd> JointAction::momentGradient(std::size_t index) const {
  if (index >= declaration_.momentConstraints.size())
    throw std::out_of_range(
        "JointAction::momentGradient: no declared constraint at index " +
        std::to_string(index));
  const auto &constraint = declaration_.momentConstraints[index];
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  const std::size_t edges = workspace.edges.size();
  std::vector<complexd> gradient(2 * edges, complexd{0.0, 0.0});
  if (workspace.carrierOrder == 0) return gradient;

  Eigen::MatrixXcd power = Eigen::MatrixXcd::Identity(
      static_cast<Eigen::Index>(workspace.carrierOrder),
      static_cast<Eigen::Index>(workspace.carrierOrder));
  for (int step = 1; step < constraint.order; ++step)
    power = power * workspace.carrier;
  power = complexd{static_cast<double>(constraint.order), 0.0} * power;

  for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
    const auto *edge = workspace.edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const std::uint64_t source = edge->getSource()->getId();
    const std::uint64_t target = edge->getTarget()->getId();
    const auto lengthDerivative = workspace.hodge.laplacianGradient(
        declaration_.carrierDegree, source, target);
    if (lengthDerivative.size() ==
        workspace.carrierOrder * workspace.carrierOrder)
      gradient[edgeIndex] = traceOfProduct(
          power, toMatrix(lengthDerivative, workspace.carrierOrder));
    const auto phaseDerivative = workspace.hodge.laplacianPhaseGradient(
        declaration_.carrierDegree, source, target);
    if (phaseDerivative.size() ==
        workspace.carrierOrder * workspace.carrierOrder)
      gradient[edges + edgeIndex] =
          workspace.storedSign[edgeIndex] * complexd{0.0, -1.0} *
          traceOfProduct(power,
                         toMatrix(phaseDerivative, workspace.carrierOrder));
  }
  return gradient;
}

// ------------------------------------------------------------------- spectrum

namespace {

/// The eigendecomposition of the carrier operator, with the modes ordered by
/// the declared occupation rule.
struct OrderedSpectrum {
  Eigen::MatrixXcd eigenvectors;
  std::vector<complexd> eigenvalues;
  std::vector<Eigen::Index> order;
};

OrderedSpectrum orderedSpectrumOf(const Eigen::MatrixXcd &carrier,
                                  bool ascendingRealPart) {
  OrderedSpectrum spectrum;
  if (carrier.rows() == 0) return spectrum;
  const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(carrier);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error(
        "JointAction: the carrier operator's eigendecomposition did not "
        "converge");
  spectrum.eigenvectors = solver.eigenvectors();
  const Eigen::VectorXcd values = solver.eigenvalues();
  spectrum.order.resize(static_cast<std::size_t>(values.size()));
  std::iota(spectrum.order.begin(), spectrum.order.end(), Eigen::Index{0});
  std::stable_sort(spectrum.order.begin(), spectrum.order.end(),
                   [&values, ascendingRealPart](Eigen::Index a, Eigen::Index b) {
                     if (ascendingRealPart)
                       return values(a).real() < values(b).real();
                     return std::abs(values(a)) < std::abs(values(b));
                   });
  spectrum.eigenvalues.reserve(spectrum.order.size());
  for (const Eigen::Index index : spectrum.order)
    spectrum.eigenvalues.push_back(values(index));
  return spectrum;
}

}  // namespace

std::vector<complexd> JointAction::carrierEigenvalues() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  if (workspace.carrierOrder == 0) return {};
  const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(workspace.carrier);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error(
        "JointAction: the carrier operator's eigendecomposition did not "
        "converge");
  std::vector<complexd> values;
  values.reserve(workspace.carrierOrder);
  for (Eigen::Index index = 0; index < solver.eigenvalues().size(); ++index)
    values.push_back(solver.eigenvalues()(index));
  return values;
}

std::vector<complexd> JointAction::orderedCarrierEigenvalues(
    bool ascendingRealPart) const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  if (workspace.carrierOrder == 0) return {};
  return orderedSpectrumOf(workspace.carrier, ascendingRealPart).eigenvalues;
}

std::vector<complexd> JointAction::occupationProjector(
    std::size_t occupied, bool ascendingRealPart) const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  if (occupied > workspace.carrierOrder)
    throw std::invalid_argument(
        "JointAction::occupationProjector: " + std::to_string(occupied) +
        " occupied modes were asked for on a carrier of " +
        std::to_string(workspace.carrierOrder) + " cells");
  if (workspace.carrierOrder == 0) return {};

  const OrderedSpectrum spectrum =
      orderedSpectrumOf(workspace.carrier, ascendingRealPart);
  const Eigen::FullPivLU<Eigen::MatrixXcd> factorization(
      spectrum.eigenvectors);
  if (!factorization.isInvertible())
    throw std::invalid_argument(
        "JointAction::occupationProjector: the carrier operator is defective, "
        "so its eigenvectors do not span and no spectral projector onto a "
        "splitting of its spectrum follows from them");

  // Gamma = V diag(chi) V^{-1} with chi the occupation indicator. Writing Phi
  // for the occupied columns of V and PhiTilde^T for the matching rows of
  // V^{-1}, this is the whitepaper's Gamma = Phi PhiTilde^T: a matched
  // left/right frame pair, idempotent by construction, with no adjoint taken.
  Eigen::VectorXcd indicator = Eigen::VectorXcd::Zero(
      static_cast<Eigen::Index>(workspace.carrierOrder));
  for (std::size_t slot = 0; slot < occupied; ++slot)
    indicator(spectrum.order[slot]) = complexd{1.0, 0.0};
  const Eigen::MatrixXcd projector = spectrum.eigenvectors *
                                     indicator.asDiagonal() *
                                     factorization.inverse();
  return toFlat(projector);
}

}  // namespace tessera::cobordism
