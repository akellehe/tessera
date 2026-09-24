// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/JointAction.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <cmath>
#include <numbers>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "chainhodge/CovariantChainHodge.h"
#include "cobordism/ChainComplex.h"
#include "matter/MatterConfiguration.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/RiemannSheet.h"
#include "mesh/Simplex.h"
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

/// The sorted vertex ids of a mesh simplex, the key the chain complex and the
/// mesh agree on.
std::vector<std::uint64_t> sortedIds(const ::tessera::mesh::Simplex &simplex) {
  std::vector<std::uint64_t> ids;
  for (const auto &vertex : simplex.getVertices()) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}

/// The hinges the primal Regge sum runs over under \p rule.
///
/// A hinge is a \f$ (d-2) \f$-simplex that is a face of at least one top cell,
/// the set `simulations::ReggeSolver` collects. Under `ReggeHinges::Interior`
/// it is kept only when its link closes: every \f$ (d-1) \f$-face containing
/// it is shared by exactly two top cells. The coface counts are read from the
/// chain complex's own boundary maps, so the test is pure incidence and uses no
/// geometry.
std::vector<::tessera::mesh::Simplex *> primalHinges(
    const std::shared_ptr<Spacetime> &shared, ReggeHinges rule) {
  // Constructing the solver materializes the facet lattice down to the
  // hinges, which a freshly built complex does not hold.
  const ReggeSolver materialized(shared, MatterConfiguration());
  const Spacetime &spacetime = *shared;
  const int d = spacetime.getMetric()->getSignature()->getDimensions();
  std::vector<::tessera::mesh::Simplex *> hinges;
  if (d < 2) return hinges;
  for (auto *simplex : spacetime.getSimplices())
    if (simplex != nullptr &&
        static_cast<int>(simplex->size()) == d - 1 && simplex->hasTopCoface())
      hinges.push_back(simplex);
  if (rule == ReggeHinges::All) return hinges;

  const ChainComplex complex = ChainComplex::fromSpacetime(spacetime);
  if (complex.dimension() != d) return {};
  // Top cofaces per (d-1)-face, from the entries of the top boundary map.
  std::vector<int> topCofaces(complex.numSimplices(d - 1), 0);
  for (const auto &entry : complex.boundaryEntries(d))
    if (entry.row >= 0 &&
        static_cast<std::size_t>(entry.row) < topCofaces.size())
      ++topCofaces[static_cast<std::size_t>(entry.row)];
  // A hinge is on the boundary when some (d-1)-face containing it has fewer
  // or more than two top cofaces.
  std::vector<int> facesOfHinge(complex.numSimplices(d - 2), 0);
  std::vector<bool> closed(complex.numSimplices(d - 2), true);
  for (const auto &entry : complex.boundaryEntries(d - 1)) {
    const auto hinge = static_cast<std::size_t>(entry.row);
    const auto face = static_cast<std::size_t>(entry.column);
    if (hinge >= closed.size() || face >= topCofaces.size()) continue;
    ++facesOfHinge[hinge];
    if (topCofaces[face] != 2) closed[hinge] = false;
  }
  std::set<std::vector<std::uint64_t>> interior;
  const auto hingeVertices = complex.kSimplexVertices(d - 2);
  for (std::size_t index = 0; index < hingeVertices.size(); ++index)
    if (closed[index] && facesOfHinge[index] > 0) {
      auto key = hingeVertices[index];
      std::sort(key.begin(), key.end());
      interior.insert(std::move(key));
    }
  std::vector<::tessera::mesh::Simplex *> kept;
  for (auto *hinge : hinges)
    if (interior.count(sortedIds(*hinge)) > 0) kept.push_back(hinge);
  return kept;
}

}  // namespace

// ------------------------------------------------------------ the Villain form

namespace {

/// The largest term count the tail certification may raise \f$ M \f$ to.
constexpr std::size_t kVillainTermCeiling = 100000;

/// How far below the truncation bound \f$ |W| \f$ may fall before it is no
/// longer certified nonzero: a value within this many tail bounds of zero is
/// indistinguishable from a zero of the infinite series.
constexpr double kVillainZeroMargin = 1e3;

/// The largest distance from one of a ratio of consecutive \f$ W \f$ values
/// that the radial continuation accepts, so the principal logarithm of every
/// accepted ratio is the increment of the continued logarithm.
constexpr double kVillainRatioBound = 0.25;

/// The smallest step of the radial continuation, as a fraction of the path.
constexpr double kVillainMinimumStep = 1e-12;

}  // namespace

VillainCharacter::VillainCharacter(double beta, double tolerance)
    : beta_(beta), tolerance_(tolerance) {
  if (!(beta > 0.0) || !std::isfinite(beta))
    throw std::invalid_argument(
        "VillainCharacter: the heat-kernel coupling beta must be positive and "
        "finite; got " + std::to_string(beta));
  if (!(tolerance > 0.0) || !(tolerance < 1.0))
    throw std::invalid_argument(
        "VillainCharacter: the relative coefficient tolerance must lie in "
        "(0, 1); got " + std::to_string(tolerance));
  q_ = std::exp(-1.0 / (2.0 * beta_));
  // M_0: the least m >= 1 with exp(-m^2/(2 beta)) < tolerance.
  declaredTerms_ = 1;
  while (std::exp(-static_cast<double>(declaredTerms_ * declaredTerms_) /
                  (2.0 * beta_)) >= tolerance_)
    ++declaredTerms_;
  const VillainSeries trivial = series(complexd{1.0, 0.0});
  secondMoment_ = trivial.second.real() / trivial.value.real();
}

VillainSeries VillainCharacter::series(complexd holonomy) const {
  const double modulus = std::abs(holonomy);
  if (!(modulus > 0.0) || !std::isfinite(modulus))
    throw std::invalid_argument(
        "VillainCharacter::series: the face holonomy must be a finite nonzero "
        "complex number");
  const double r = std::max(modulus, 1.0 / modulus);
  const double logR = std::log(r);

  // rho_k(M), the bound on every tail ratio past M (see VillainSeries).
  auto ratio = [&](std::size_t m, int k) {
    const double grow = std::pow(static_cast<double>(m + 2) /
                                     static_cast<double>(m + 1),
                                 k);
    return grow * std::exp(-static_cast<double>(2 * m + 3) / (2.0 * beta_) +
                           logR);
  };
  // 2 t_{M+1} with t_m = m^k q^{m^2} r^m, the first omitted pair's bound.
  auto leading = [&](std::size_t m, int k) {
    const double next = static_cast<double>(m + 1);
    return 2.0 * std::pow(next, k) *
           std::exp(-next * next / (2.0 * beta_) + next * logR);
  };

  VillainSeries out;
  out.value = complexd{1.0, 0.0};
  double firstMagnitude = 0.0;
  double secondMagnitude = 0.0;
  const complexd inverse = complexd{1.0, 0.0} / holonomy;
  complexd up{1.0, 0.0};
  complexd down{1.0, 0.0};
  // Terms are added until M reaches M_0 and, for the holonomy at hand, the
  // geometric tail ratios are at most 1/2 and every tail bound is below the
  // declared tolerance relative to the sum of the moduli of the kept terms of
  // its series. Away from the unit circle the terms q^{m^2} r^m peak near
  // m = beta log r rather than at m = 0, so M_0 alone does not bound them.
  for (std::size_t m = 1;; ++m) {
    if (m > kVillainTermCeiling)
      throw std::invalid_argument(
          "VillainCharacter::series: no term count up to " +
          std::to_string(kVillainTermCeiling) +
          " certifies the tail at this holonomy");
    up *= holonomy;
    down *= inverse;
    const double md = static_cast<double>(m);
    const double coefficient = std::exp(-md * md / (2.0 * beta_));
    const double size = coefficient * (std::abs(up) + std::abs(down));
    out.value += coefficient * (up + down);
    out.magnitude += size;
    out.first += coefficient * md * (up - down);
    firstMagnitude += md * size;
    out.second += coefficient * md * md * (up + down);
    secondMagnitude += md * md * size;
    if (m < declaredTerms_ || ratio(m, 2) > 0.5) continue;
    const double valueTail = leading(m, 0) / (1.0 - ratio(m, 0));
    const double firstTail = leading(m, 1) / (1.0 - ratio(m, 1));
    const double secondTail = leading(m, 2) / (1.0 - ratio(m, 2));
    if (valueTail > tolerance_ * out.magnitude ||
        firstTail > tolerance_ * std::max(firstMagnitude, out.magnitude) ||
        secondTail > tolerance_ * std::max(secondMagnitude, out.magnitude))
      continue;
    out.termCount = m;
    out.valueTail = valueTail;
    out.firstTail = firstTail;
    out.secondTail = secondTail;
    return out;
  }
}

namespace {

/// Whether a truncated value of \f$ W \f$ is certified nonzero: it exceeds, by
/// the declared margin, its tail bound plus the rounding scale of the sum
/// (machine epsilon times the sum of the moduli of the kept terms).
bool certifiedNonzero(const VillainSeries &series) {
  const double uncertainty =
      series.valueTail +
      std::numeric_limits<double>::epsilon() * series.magnitude;
  return std::abs(series.value) > kVillainZeroMargin * uncertainty;
}

}  // namespace

complexd VillainCharacter::logarithm(complexd holonomy) const {
  const double modulus = std::abs(holonomy);
  if (!(modulus > 0.0) || !std::isfinite(modulus))
    throw std::invalid_argument(
        "VillainCharacter::logarithm: the face holonomy must be a finite "
        "nonzero complex number");
  // The start of the radial path, on the unit circle, where W is real and
  // positive: its principal logarithm is the real logarithm of a positive
  // number, and it is the branch the continuation carries.
  const complexd direction = holonomy / modulus;
  const VillainSeries start = series(direction);
  if (!(start.value.real() > 0.0) ||
      std::abs(start.value.imag()) > 1e-12 * start.value.real())
    throw std::logic_error(
        "VillainCharacter::logarithm: W is not real and positive on the unit "
        "circle, which contradicts its Poisson form");
  complexd accumulated{std::log(start.value.real()), 0.0};
  const double radial = std::log(modulus);
  if (radial == 0.0) return accumulated;

  auto at = [&](double t) {
    const VillainSeries point = series(direction * std::exp(t * radial));
    if (!certifiedNonzero(point))
      throw std::domain_error(
          "VillainCharacter::logarithm: W has a zero on the radial path to "
          "this holonomy, so log W has no value on the branch real on the "
          "unit circle there");
    return point.value;
  };

  double t = 0.0;
  double step = 0.125;
  complexd previous = start.value;
  while (t < 1.0) {
    step = std::min(step, 1.0 - t);
    const complexd middle = at(t + 0.5 * step);
    const complexd end = at(t + step);
    const complexd firstHalf = middle / previous;
    const complexd secondHalf = end / middle;
    if (std::abs(firstHalf - 1.0) > kVillainRatioBound ||
        std::abs(secondHalf - 1.0) > kVillainRatioBound) {
      step *= 0.5;
      if (step < kVillainMinimumStep)
        throw std::domain_error(
            "VillainCharacter::logarithm: the radial continuation cannot "
            "resolve W on the path to this holonomy");
      continue;
    }
    accumulated += std::log(firstHalf) + std::log(secondHalf);
    t += step;
    previous = end;
    step = std::min(2.0 * step, 0.125);
  }
  return accumulated;
}

double VillainCharacter::zeroDistance(complexd holonomy) const {
  const double modulus = std::abs(holonomy);
  if (!(modulus > 0.0) || !std::isfinite(modulus))
    throw std::invalid_argument(
        "VillainCharacter::zeroDistance: the face holonomy must be a finite "
        "nonzero complex number");
  double nearest = std::numeric_limits<double>::infinity();
  // The zeros are -q^{k} and -q^{-k} for odd k. Past the zero nearest to |F|
  // on either side the distance only grows, so the loop stops once both
  // families have passed |F| by a wide margin.
  for (std::size_t n = 1;; ++n) {
    const double k = static_cast<double>(2 * n - 1);
    const double inner = std::pow(q_, k);
    if (!(inner > 0.0)) break;
    const double outer = 1.0 / inner;
    nearest = std::min(nearest, std::abs(holonomy + inner) /
                                    std::min(inner, modulus));
    if (std::isfinite(outer))
      nearest = std::min(nearest, std::abs(holonomy + outer) /
                                      std::min(outer, modulus));
    if (inner < 1e-3 * modulus && outer > 1e3 * modulus) break;
  }
  return nearest;
}

complexd VillainCharacter::potential(complexd holonomy) const {
  return -matchedWeight() * logarithm(holonomy);
}

namespace {

VillainSeries certifiedSeries(const VillainCharacter &character,
                              complexd holonomy) {
  VillainSeries point = character.series(holonomy);
  if (!certifiedNonzero(point))
    throw std::domain_error(
        "VillainCharacter: W is not certified nonzero at this holonomy, so the "
        "potential's derivatives W'/W and W''/W are not defined there");
  return point;
}

}  // namespace

complexd VillainCharacter::firstDerivative(complexd holonomy) const {
  const VillainSeries point = certifiedSeries(*this, holonomy);
  return -matchedWeight() * point.first / point.value;
}

complexd VillainCharacter::secondDerivative(complexd holonomy) const {
  const VillainSeries point = certifiedSeries(*this, holonomy);
  const complexd mean = point.first / point.value;
  return -matchedWeight() * (point.second / point.value - mean * mean);
}

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

namespace {

/// The per-face potential of the declared holonomy form and its first two
/// Maurer-Cartan derivatives, \f$ \phi \f$, \f$ D\phi \f$ and \f$ D^2\phi \f$
/// with \f$ D=F\,d/dF \f$, the weight included. The Villain character is built
/// once per query and shared by every face.
class FacePotential {
 public:
  explicit FacePotential(const JointActionDeclaration &declaration)
      : weight_(declaration.holonomyWeight) {
    if (declaration.holonomyForm == HolonomyForm::Villain && weight_ > 0.0)
      villain_.emplace(weight_, declaration.villainTolerance);
  }

  [[nodiscard]] bool active() const { return weight_ != 0.0; }

  [[nodiscard]] complexd value(complexd holonomy) const {
    if (!active()) return complexd{0.0, 0.0};
    if (villain_) return villain_->potential(holonomy);
    return weight_ * (complexd{1.0, 0.0} -
                      0.5 * (holonomy + complexd{1.0, 0.0} / holonomy));
  }

  [[nodiscard]] complexd first(complexd holonomy) const {
    if (!active()) return complexd{0.0, 0.0};
    if (villain_) return villain_->firstDerivative(holonomy);
    return -0.5 * weight_ * (holonomy - complexd{1.0, 0.0} / holonomy);
  }

  [[nodiscard]] complexd second(complexd holonomy) const {
    if (!active()) return complexd{0.0, 0.0};
    if (villain_) return villain_->secondDerivative(holonomy);
    return -0.5 * weight_ * (holonomy + complexd{1.0, 0.0} / holonomy);
  }

  [[nodiscard]] const std::optional<VillainCharacter> &villain() const {
    return villain_;
  }

 private:
  double weight_;
  std::optional<VillainCharacter> villain_;
};

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
  if (declaration_.stiffnessWeight != 0.0) {
    const std::size_t edges = edgeCount();
    if (declaration_.referenceLengths.size() != edges)
      throw std::invalid_argument(
          "JointAction: the length stiffness needs one reference length per "
          "edge; got " +
          std::to_string(declaration_.referenceLengths.size()) + " for " +
          std::to_string(edges) + " edges");
  }
  if (declaration_.holonomyForm == HolonomyForm::Villain) {
    if (declaration_.holonomyWeight < 0.0)
      throw std::invalid_argument(
          "JointAction: the Villain holonomy term's coupling beta is a "
          "heat-kernel time and must be non-negative; got " +
          std::to_string(declaration_.holonomyWeight));
    if (!(declaration_.villainTolerance > 0.0) ||
        !(declaration_.villainTolerance < 1.0))
      throw std::invalid_argument(
          "JointAction: the Villain coefficient tolerance must lie in (0, 1); "
          "got " + std::to_string(declaration_.villainTolerance));
  }
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
  if (declaration_.reggeBranch == ReggeBranch::Continued) {
    const auto edges = spacetime_->getEdgeList()->toVector();
    const auto &declared = declaration_.reggeStartSquaredLengths;
    if (!declared.empty() && declared.size() != edges.size())
      throw std::invalid_argument(
          "JointAction: the continued Regge sheets need one starting squared "
          "length per edge; got " + std::to_string(declared.size()) + " for " +
          std::to_string(edges.size()) + " edges");
    for (std::size_t index = 0; index < edges.size(); ++index) {
      const auto *edge = edges[index];
      if (edge == nullptr || edge->getSource() == nullptr ||
          edge->getTarget() == nullptr)
        continue;
      const complexd length = edge->getLength();
      reggeStart_[pairKey(edge->getSource()->getId(),
                          edge->getTarget()->getId())] =
          declared.empty() ? length * length : declared[index];
    }
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
  return {"regge", "stiffness", "holonomy", "matter", "spectral"};
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

// ------------------------------------------------- the continued Regge sheets

struct JointAction::ReggeSheets {
  struct Hinge {
    ::tessera::mesh::Simplex *hinge{nullptr};
    /// The sheets of the hinge's dihedral angles, by top cell.
    ::tessera::mesh::Simplex::DihedralSheets angles;
    /// The sign relating the continued content root to `Simplex::volume`.
    int contentSign{1};
  };
  std::vector<Hinge> hinges;
  /// The number of dihedral angles whose declared sheet is not the principal
  /// one.
  std::size_t offPrincipal{0};
};

namespace {

using ::tessera::mesh::principalArcCosine;
using ::tessera::mesh::principalSquareRoot;
using ::tessera::mesh::SheetedAcos;
using ::tessera::mesh::SheetedSqrt;
using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;
using SquaredLengthField = std::map<EdgeKey, complexd>;

/// The largest turn about its branch point a root may make in one step of a
/// walk, and the largest distance an angle may move. Both are well inside the
/// half turn beyond which a continuation cannot tell two paths apart.
constexpr double kMaximumRootTurn = std::numbers::pi / 4.0;
constexpr double kMaximumAngleStep = 0.25;
/// The shortest step a walk refines to before it accepts a step as it is.
constexpr double kShortestStep = 1.0 / (1u << 30);

/// A real cosine pinned to the +0 side of the inverse cosine's cuts, as
/// `Simplex::dihedralAngle` pins it.
complexd pinnedCosine(complexd Cij, complexd rootProduct) {
  complexd r = -Cij / rootProduct;
  if (r.imag() == 0.0) r = {r.real(), 0.0};
  return r;
}

/// The squared length of \p key at parameter \p t on the straight segment from
/// \p from to \p to.
complexd along(const SquaredLengthField &from, const SquaredLengthField &to,
               const EdgeKey &key, double t) {
  const complexd a = from.at(key);
  const complexd b = to.at(key);
  return a + t * (b - a);
}

/// The Cayley-Menger matrix of the simplex on the sorted vertex ids \p ids at
/// parameter \p t, in the canonical frame `Simplex::cayleyMengerCanonical` uses.
std::vector<complexd> cayleyMenger(const std::vector<std::uint64_t> &ids,
                                   const SquaredLengthField &from,
                                   const SquaredLengthField &to, double t) {
  const int m = static_cast<int>(ids.size());
  const int n = m + 1;
  std::vector<complexd> B(static_cast<std::size_t>(n) * n, complexd{0.0, 0.0});
  for (int k = 1; k < n; ++k) {
    B[static_cast<std::size_t>(k)] = 1.0;
    B[static_cast<std::size_t>(k) * n] = 1.0;
  }
  for (int i = 0; i < m; ++i)
    for (int j = i + 1; j < m; ++j) {
      const complexd z = along(from, to, pairKey(ids[i], ids[j]), t);
      B[static_cast<std::size_t>(i + 1) * n + (j + 1)] = z;
      B[static_cast<std::size_t>(j + 1) * n + (i + 1)] = z;
    }
  return B;
}

/// The Gram determinant of the simplex on \p ids at parameter \p t, the
/// radicand of its content root.
complexd gramDeterminant(const std::vector<std::uint64_t> &ids,
                         const SquaredLengthField &from,
                         const SquaredLengthField &to, double t) {
  const int d = static_cast<int>(ids.size()) - 1;
  if (d < 1) return {1.0, 0.0};
  auto z = [&](int a, int b) -> complexd {
    return a == b ? complexd{0.0, 0.0}
                  : along(from, to, pairKey(ids[a], ids[b]), t);
  };
  std::vector<complexd> G(static_cast<std::size_t>(d) * d);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
      G[static_cast<std::size_t>(i) * d + j] =
          0.5 * (z(0, i + 1) + z(0, j + 1) - z(i + 1, j + 1));
  return ::tessera::mesh::Simplex::determinant(G, d);
}

/// The roots and angles of one top cell carried along a path of geometries:
/// the root of every diagonal Cayley-Menger cofactor (one per vertex, the
/// content of the opposite face) and the inverse cosine of every requested
/// vertex pair.
struct CellWalk {
  std::vector<std::uint64_t> ids;
  /// Border offsets (1-based) of the vertex pairs whose angles are carried.
  std::vector<std::pair<int, int>> pairs;
  std::vector<SheetedSqrt> roots;
  std::vector<SheetedAcos> angles;

  /// Declare every root and angle on its principal sheet at \p cofactors.
  void declare(const std::vector<complexd> &cofactors) {
    const int n = static_cast<int>(ids.size()) + 1;
    roots.clear();
    angles.clear();
    for (int p = 1; p < n; ++p)
      roots.emplace_back(cofactors[static_cast<std::size_t>(p) * n + p]);
    for (const auto &[i, j] : pairs)
      angles.emplace_back(pinnedCosine(
          cofactors[static_cast<std::size_t>(i) * n + j],
          roots[static_cast<std::size_t>(i - 1)].value() *
              roots[static_cast<std::size_t>(j - 1)].value()));
  }

  /// Advance every label to \p cofactors and return whether the step was
  /// fine enough: every root turned by at most `kMaximumRootTurn` and every
  /// angle moved by at most `kMaximumAngleStep`.
  bool advance(const std::vector<complexd> &cofactors) {
    const int n = static_cast<int>(ids.size()) + 1;
    bool fine = true;
    for (int p = 1; p < n; ++p) {
      auto &root = roots[static_cast<std::size_t>(p - 1)];
      root.advance(cofactors[static_cast<std::size_t>(p) * n + p]);
      if (std::abs(root.lastStep()) > kMaximumRootTurn) fine = false;
    }
    for (std::size_t index = 0; index < pairs.size(); ++index) {
      const auto [i, j] = pairs[index];
      angles[index].advance(pinnedCosine(
          cofactors[static_cast<std::size_t>(i) * n + j],
          roots[static_cast<std::size_t>(i - 1)].value() *
              roots[static_cast<std::size_t>(j - 1)].value()));
      if (angles[index].lastStep() > kMaximumAngleStep) fine = false;
    }
    return fine;
  }
};

/// The top cells containing \p hinge: the cells `Simplex::deficitAngle` sums
/// over, found through the hinge's first vertex.
std::vector<::tessera::mesh::Simplex *> topCellsAt(
    const ::tessera::mesh::Simplex &hinge, int dimension) {
  std::vector<::tessera::mesh::Simplex *> out;
  const auto &vertices = hinge.getVertices();
  if (vertices.empty()) return out;
  std::set<std::vector<std::uint64_t>> seen;
  for (const auto &cell : vertices.front()->getSimplices()) {
    if (cell == nullptr ||
        static_cast<int>(cell->size()) != dimension + 1)
      continue;
    bool containsAll = true;
    for (const auto &vertex : vertices)
      if (!cell->hasVertex(vertex)) {
        containsAll = false;
        break;
      }
    if (containsAll && seen.insert(sortedIds(*cell)).second)
      out.push_back(cell);
  }
  return out;
}

/// Walk \p state along the straight segment from \p from to \p to, refining
/// every step until it is fine enough, by bisection down to `kShortestStep`.
/// \p advance takes the parameter and returns whether the step it made was
/// fine; \p State is copied so that a refused step is undone.
template <typename State, typename Advance>
void walkSegment(State &state, Advance advance) {
  double t = 0.0;
  double step = 1.0;
  while (t < 1.0) {
    const double next = std::min(1.0, t + step);
    State trial = state;
    if (advance(trial, next) || step <= kShortestStep) {
      state = std::move(trial);
      t = next;
      step = std::min(1.0, 2.0 * step);
    } else {
      step *= 0.5;
    }
  }
}

}  // namespace

JointAction::ReggeSheets JointAction::reggeSheets() const {
  using ::tessera::mesh::Simplex;
  ReggeSheets out;
  const auto hinges = primalHinges(spacetime_, declaration_.reggeHinges);
  if (declaration_.reggeBranch == ReggeBranch::Principal) {
    for (auto *hinge : hinges) out.hinges.push_back({hinge, {}, 1});
    return out;
  }

  // The three geometries of the path: the Euclidean reference Re z0, the
  // starting geometry z0 and the geometry the mesh holds now.
  SquaredLengthField reference;
  SquaredLengthField now;
  for (const auto &[key, value] : reggeStart_)
    reference[key] = complexd{value.real(), 0.0};
  for (const auto *edge : spacetime_->getEdgeList()->toVector()) {
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const complexd length = edge->getLength();
    now[pairKey(edge->getSource()->getId(), edge->getTarget()->getId())] =
        length * length;
  }
  for (const auto &[key, value] : now)
    if (reggeStart_.find(key) == reggeStart_.end())
      throw std::logic_error(
          "JointAction: an edge of the mesh has no starting squared length for "
          "the continued Regge sheets; the triangulation changed after the "
          "action was constructed");
  const SquaredLengthField &start = reggeStart_;
  const int dimension =
      spacetime_->getMetric()->getSignature()->getDimensions();
  // The path: the Euclidean reference to the starting geometry, then the
  // starting geometry to the current one.
  const std::array<std::pair<const SquaredLengthField *,
                             const SquaredLengthField *>, 2>
      legs{{{&reference, &start}, {&start, &now}}};

  // Which vertex pairs of which top cell carry an angle of the sum.
  std::map<std::vector<std::uint64_t>, CellWalk> walks;
  std::map<std::vector<std::uint64_t>, Simplex *> cellOf;
  for (auto *hinge : hinges) {
    const auto hingeIds = sortedIds(*hinge);
    for (auto *cell : topCellsAt(*hinge, dimension)) {
      const auto cellIds = sortedIds(*cell);
      auto &walk = walks[cellIds];
      walk.ids = cellIds;
      cellOf[cellIds] = cell;
      std::vector<int> opposite;
      for (int k = 0; k < static_cast<int>(cellIds.size()); ++k)
        if (!std::binary_search(hingeIds.begin(), hingeIds.end(), cellIds[k]))
          opposite.push_back(k + 1);
      if (opposite.size() == 2)
        walk.pairs.emplace_back(opposite[0], opposite[1]);
    }
  }

  // Continue every cell's roots and angles along the two segments.
  std::map<std::pair<std::vector<std::uint64_t>, std::pair<int, int>>,
           std::pair<complexd, complexd>>
      continued;  // (root product, angle) per cell and vertex pair
  for (auto &[cellIds, walk] : walks) {
    const int n = static_cast<int>(cellIds.size()) + 1;
    walk.declare(Simplex::cofactorMatrix(
        cayleyMenger(cellIds, reference, reference, 0.0), n));
    for (const auto &[from, to] : legs)
      walkSegment(walk, [&, from = from, to = to](CellWalk &trial, double t) {
        return trial.advance(Simplex::cofactorMatrix(
            cayleyMenger(cellIds, *from, *to, t), n));
      });
    for (std::size_t index = 0; index < walk.pairs.size(); ++index) {
      const auto [i, j] = walk.pairs[index];
      continued[{cellIds, {i, j}}] = {
          walk.roots[static_cast<std::size_t>(i - 1)].value() *
              walk.roots[static_cast<std::size_t>(j - 1)].value(),
          walk.angles[index].value()};
    }
  }

  // Read each continued value as a sheet of the mesh's own cofactors, so the
  // value and derivative the mesh evaluates are on the continued sheet exactly
  // and the comparison does not depend on how the walk's last point rounds.
  for (auto *hinge : hinges) {
    ReggeSheets::Hinge entry;
    entry.hinge = hinge;
    const auto hingeIds = sortedIds(*hinge);
    for (auto *cell : topCellsAt(*hinge, dimension)) {
      const auto cellIds = sortedIds(*cell);
      std::vector<int> opposite;
      for (int k = 0; k < static_cast<int>(cellIds.size()); ++k)
        if (!std::binary_search(hingeIds.begin(), hingeIds.end(), cellIds[k]))
          opposite.push_back(k + 1);
      if (opposite.size() != 2) continue;
      const auto found = continued.find({cellIds, {opposite[0], opposite[1]}});
      const auto cofactors = cell->dihedralCofactors(hinge);
      if (found == continued.end() || !cofactors.ok) continue;
      const auto &[product, angle] = found->second;
      const complexd principal = principalSquareRoot(cofactors.Cii) *
                                 principalSquareRoot(cofactors.Cjj);
      Simplex::DihedralSheet sheet;
      sheet.rootSign =
          (std::conj(product) * principal).real() >= 0.0 ? 1 : -1;
      const complexd arccosine = principalArcCosine(pinnedCosine(
          cofactors.Cij, static_cast<double>(sheet.rootSign) * principal));
      double best = std::numeric_limits<double>::infinity();
      for (const int orientation : {1, -1}) {
        const double turns = std::round(
            (angle - static_cast<double>(orientation) * arccosine).real() /
            (2.0 * std::numbers::pi));
        const complexd candidate =
            2.0 * std::numbers::pi * turns +
            static_cast<double>(orientation) * arccosine;
        if (std::abs(candidate - angle) < best) {
          best = std::abs(candidate - angle);
          sheet.orientation = orientation;
          sheet.branchIndex = static_cast<int>(turns);
        }
      }
      if (sheet.rootSign != 1 || sheet.branchIndex != 0 ||
          sheet.orientation != 1)
        ++out.offPrincipal;
      entry.angles[cellIds] = sheet;
    }
    // The content root of the hinge, continued the same way.
    if (hingeIds.size() >= 2) {
      SheetedSqrt content(gramDeterminant(hingeIds, reference, reference, 0.0));
      for (const auto &[from, to] : legs)
        walkSegment(content, [&, from = from, to = to](SheetedSqrt &trial,
                                                       double t) {
          trial.advance(gramDeterminant(hingeIds, *from, *to, t));
          return std::abs(trial.lastStep()) <= kMaximumRootTurn;
        });
      entry.contentSign =
          (std::conj(content.value()) * hinge->volume()).real() >= 0.0 ? 1
                                                                        : -1;
    }
    out.hinges.push_back(std::move(entry));
  }
  return out;
}

std::complex<double> JointAction::reggeTerm() const {
  if (declaration_.gravitationalWeight == 0.0) return complexd{0.0, 0.0};
  if (declaration_.reggeForm == ReggeForm::Dual)
    return declaration_.gravitationalWeight *
           ReggeSolver(spacetime_, MatterConfiguration()).dualReggeAction();
  complexd sum{0.0, 0.0};
  for (const auto &entry : reggeSheets().hinges)
    sum += static_cast<double>(entry.contentSign) *
           ReggeSolver::hingeContent(entry.hinge) *
           entry.hinge->deficitAngle(entry.angles);
  return declaration_.gravitationalWeight * sum;
}

std::size_t JointAction::reggeHingeCount() const {
  return primalHinges(spacetime_, declaration_.reggeHinges).size();
}

bool JointAction::reggeStructurallyZero() const {
  return declaration_.gravitationalWeight != 0.0 &&
         declaration_.reggeForm == ReggeForm::Primal && reggeHingeCount() == 0;
}

std::size_t JointAction::reggeOffPrincipalAngles() const {
  if (declaration_.reggeForm != ReggeForm::Primal) return 0;
  return reggeSheets().offPrincipal;
}

std::complex<double> JointAction::stiffnessTerm() const {
  if (declaration_.stiffnessWeight == 0.0) return complexd{0.0, 0.0};
  const auto edges = spacetime_->getEdgeList()->toVector();
  complexd sum{0.0, 0.0};
  for (std::size_t index = 0; index < edges.size(); ++index) {
    const complexd stretch =
        edges[index]->getLength() - declaration_.referenceLengths[index];
    sum += 0.5 * stretch * stretch;
  }
  return declaration_.stiffnessWeight * sum;
}

std::complex<double> JointAction::holonomyTerm() const {
  const FacePotential potential(declaration_);
  if (!potential.active()) return complexd{0.0, 0.0};
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  complexd sum{0.0, 0.0};
  for (const complexd &holonomy : workspace.holonomies)
    sum += potential.value(holonomy);
  return sum;
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
  return reggeTerm() + stiffnessTerm() + holonomyTerm() + matterTerm() +
         spectralTerm();
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

  if (declaration_.gravitationalWeight != 0.0 &&
      declaration_.reggeForm == ReggeForm::Dual) {
    const auto reggeGradient =
        ReggeSolver(spacetime_, MatterConfiguration()).actionGradientExact();
    for (std::size_t edgeIndex = 0;
         edgeIndex < edges && edgeIndex < reggeGradient.size(); ++edgeIndex)
      stationarity[edgeIndex] +=
          declaration_.gravitationalWeight * reggeGradient[edgeIndex];
  }

  if (declaration_.gravitationalWeight != 0.0 &&
      declaration_.reggeForm == ReggeForm::Primal) {
    // d(sum_h |h| eps_h)/dz_e = sum_h (d|h|/dz_e eps_h + |h| d eps_h/dz_e),
    // both factors from the per-hinge analytic gradients. No Schlaefli
    // identity is invoked, because under the interior-hinge rule the sum over
    // a cell's hinges is incomplete and that identity does not apply.
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> indexOfEdge;
    for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
      const auto *edge = workspace.edges[edgeIndex];
      if (edge == nullptr || edge->getSource() == nullptr ||
          edge->getTarget() == nullptr)
        continue;
      indexOfEdge[pairKey(edge->getSource()->getId(),
                          edge->getTarget()->getId())] = edgeIndex;
    }
    // Every root and inverse cosine on its declared sheet (ReggeBranch): the
    // content root through its sign, the angles through their sheets.
    for (const auto &entry : reggeSheets().hinges) {
      auto *hinge = entry.hinge;
      const double sign = static_cast<double>(entry.contentSign);
      const complexd content = sign * ReggeSolver::hingeContent(hinge);
      const complexd deficit = hinge->deficitAngle(entry.angles);
      for (const auto &[key, derivative] : hinge->volumeGradient()) {
        const auto found = indexOfEdge.find(pairKey(key.first, key.second));
        if (found != indexOfEdge.end())
          stationarity[found->second] +=
              declaration_.gravitationalWeight * sign * derivative * deficit;
      }
      for (const auto &[key, derivative] :
           hinge->deficitAngleGradient(entry.angles)) {
        const auto found = indexOfEdge.find(pairKey(key.first, key.second));
        if (found != indexOfEdge.end())
          stationarity[found->second] +=
              declaration_.gravitationalWeight * content * derivative;
      }
    }
  }

  if (declaration_.stiffnessWeight != 0.0) {
    // d/dz [ (l - l0)^2 / 2 ] = (l - l0) dl/dz with dl/dz = 1/(2l) on the
    // branch the stored length already sits on.
    for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
      const auto *edge = workspace.edges[edgeIndex];
      if (edge == nullptr) continue;
      const complexd length = edge->getLength();
      stationarity[edgeIndex] +=
          declaration_.stiffnessWeight *
          (length - declaration_.referenceLengths[edgeIndex]) /
          (2.0 * length);
    }
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
///     = \sum_\tau \epsilon_{\tau e}\,D\phi(\mathcal F_\tau) \f$,
/// one entry per canonical degree-one cell.
///
/// Exact in closed form: \f$ \mathcal F_\tau \f$ is a Laurent monomial in the
/// links, so \f$ U_e\,\partial\mathcal F_\tau/\partial U_e
/// = \epsilon_{\tau e}\mathcal F_\tau \f$ and the chain rule gives
/// \f$ U_e\,\partial\phi(\mathcal F_\tau)/\partial U_e
/// = \epsilon_{\tau e}\,D\phi(\mathcal F_\tau) \f$.
std::vector<complexd> holonomyLinkDerivative(const ActionWorkspace &workspace,
                                             const FacePotential &potential) {
  std::vector<complexd> derivative(workspace.links.size(), complexd{0.0, 0.0});
  if (!potential.active() || workspace.complex.dimension() < 2)
    return derivative;
  std::vector<complexd> perFace(workspace.holonomies.size());
  for (std::size_t face = 0; face < perFace.size(); ++face)
    perFace[face] = potential.first(workspace.holonomies[face]);
  for (const auto &entry : workspace.complex.boundaryEntries(2)) {
    const auto row = static_cast<std::size_t>(entry.row);
    const auto column = static_cast<std::size_t>(entry.column);
    if (row >= derivative.size() || column >= perFace.size()) continue;
    derivative[row] += static_cast<double>(entry.value) * perFace[column];
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
      holonomyLinkDerivative(workspace, FacePotential(declaration_));
  for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
    const long long canonical = workspace.canonicalOfEdge[edgeIndex];
    if (canonical < 0 ||
        static_cast<std::size_t>(canonical) >= holonomyPart.size())
      continue;
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

std::vector<complexd> JointAction::holonomyHessian() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  const std::size_t edges = workspace.edges.size();
  std::vector<complexd> hessian(edges * edges, complexd{0.0, 0.0});
  const FacePotential potential(declaration_);
  if (!potential.active() || workspace.complex.dimension() < 2) return hessian;

  // Per face, the incidences (canonical edge, sign) of its boundary.
  const std::size_t faces = workspace.holonomies.size();
  std::vector<std::vector<std::pair<std::size_t, double>>> boundary(faces);
  for (const auto &entry : workspace.complex.boundaryEntries(2)) {
    const auto row = static_cast<std::size_t>(entry.row);
    const auto column = static_cast<std::size_t>(entry.column);
    if (row >= workspace.links.size() || column >= faces) continue;
    boundary[column].emplace_back(row, static_cast<double>(entry.value));
  }
  const std::size_t cells = workspace.links.size();
  std::vector<complexd> canonical(cells * cells, complexd{0.0, 0.0});
  for (std::size_t face = 0; face < faces; ++face) {
    const complexd curvature = potential.second(workspace.holonomies[face]);
    for (const auto &[a, sa] : boundary[face])
      for (const auto &[b, sb] : boundary[face])
        canonical[a * cells + b] += sa * sb * curvature;
  }
  // U_stored = U_canonical^{s}, so U d/dU on the stored orientation is s times
  // the canonical one, once per index.
  for (std::size_t e = 0; e < edges; ++e) {
    const long long ce = workspace.canonicalOfEdge[e];
    if (ce < 0) continue;
    for (std::size_t f = 0; f < edges; ++f) {
      const long long cf = workspace.canonicalOfEdge[f];
      if (cf < 0) continue;
      hessian[e * edges + f] =
          workspace.storedSign[e] * workspace.storedSign[f] *
          canonical[static_cast<std::size_t>(ce) * cells +
                    static_cast<std::size_t>(cf)];
    }
  }
  return hessian;
}

double JointAction::holonomyZeroDistance() const {
  const FacePotential potential(declaration_);
  if (!potential.villain()) return std::numeric_limits<double>::infinity();
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  double nearest = std::numeric_limits<double>::infinity();
  for (const complexd &holonomy : workspace.holonomies)
    nearest = std::min(nearest, potential.villain()->zeroDistance(holonomy));
  return nearest;
}

double JointAction::holonomyZeroClearance(
    const std::vector<complexd> &linkIncrements, double spacing) const {
  if (!(spacing > 0.0))
    throw std::invalid_argument(
        "JointAction::holonomyZeroClearance: the node spacing must be "
        "positive");
  const FacePotential potential(declaration_);
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  if (linkIncrements.size() != workspace.edges.size())
    throw std::invalid_argument(
        "JointAction::holonomyZeroClearance: one increment per edge is "
        "required; got " + std::to_string(linkIncrements.size()) + " for " +
        std::to_string(workspace.edges.size()) + " edges");
  if (!potential.villain() || workspace.complex.dimension() < 2)
    return std::numeric_limits<double>::infinity();
  const VillainCharacter &character = *potential.villain();

  // The increments on the canonical orientations: U_stored = U_canonical^{s}.
  std::vector<complexd> canonical(workspace.links.size(), complexd{0.0, 0.0});
  for (std::size_t e = 0; e < workspace.edges.size(); ++e) {
    const long long c = workspace.canonicalOfEdge[e];
    if (c < 0 || static_cast<std::size_t>(c) >= canonical.size()) continue;
    canonical[static_cast<std::size_t>(c)] +=
        workspace.storedSign[e] * linkIncrements[e];
  }
  std::vector<complexd> exponent(workspace.holonomies.size(),
                                 complexd{0.0, 0.0});
  for (const auto &entry : workspace.complex.boundaryEntries(2)) {
    const auto row = static_cast<std::size_t>(entry.row);
    const auto column = static_cast<std::size_t>(entry.column);
    if (row >= canonical.size() || column >= exponent.size()) continue;
    exponent[column] += static_cast<double>(entry.value) * canonical[row];
  }

  double nearest = std::numeric_limits<double>::infinity();
  for (std::size_t face = 0; face < exponent.size(); ++face) {
    const std::size_t intervals = std::max<std::size_t>(
        1, static_cast<std::size_t>(
               std::ceil(std::abs(exponent[face]) / spacing)));
    for (std::size_t node = 0; node <= intervals; ++node) {
      const double t =
          static_cast<double>(node) / static_cast<double>(intervals);
      nearest = std::min(
          nearest, character.zeroDistance(workspace.holonomies[face] *
                                          std::exp(t * exponent[face])));
    }
  }
  return nearest;
}

HolonomyTruncation JointAction::holonomyTruncation() const {
  HolonomyTruncation report;
  report.form = declaration_.holonomyForm;
  if (declaration_.holonomyForm != HolonomyForm::Villain) return report;
  report.tolerance = declaration_.villainTolerance;
  const FacePotential potential(declaration_);
  if (!potential.villain()) return report;
  const VillainCharacter &character = *potential.villain();
  report.declaredTermCount = character.declaredTermCount();
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  for (const complexd &holonomy : workspace.holonomies) {
    const VillainSeries point = character.series(holonomy);
    const double scale = std::abs(point.value);
    report.maximumTermCount = std::max(report.maximumTermCount,
                                       point.termCount);
    report.relativeValueTail =
        std::max(report.relativeValueTail, point.valueTail / scale);
    report.relativeFirstTail =
        std::max(report.relativeFirstTail, point.firstTail / scale);
    report.relativeSecondTail =
        std::max(report.relativeSecondTail, point.secondTail / scale);
  }
  return report;
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
  forceOnly.stiffnessWeight = 0.0;
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

std::vector<complexd> JointAction::canonicalWardCurrent() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  const auto stored = linkStationarity();
  std::vector<complexd> canonical(workspace.complex.numSimplices(1),
                                  complexd{0.0, 0.0});
  for (std::size_t edgeIndex = 0; edgeIndex < stored.size(); ++edgeIndex) {
    const long long index = workspace.canonicalOfEdge[edgeIndex];
    if (index < 0 || static_cast<std::size_t>(index) >= canonical.size())
      continue;
    canonical[static_cast<std::size_t>(index)] =
        workspace.storedSign[edgeIndex] * stored[edgeIndex];
  }
  return canonical;
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
  const std::vector<complexd> canonical = canonicalWardCurrent();

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
