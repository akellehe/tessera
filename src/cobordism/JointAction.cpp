// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/JointAction.h"

#include <algorithm>
#include <array>
#include <limits>
#include <functional>
#include <map>
#include <cmath>
#include <cstdio>
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

/// \f$ u=2^{-53} \f$, the unit roundoff of double precision: half the machine
/// epsilon, the largest relative error of one correctly rounded operation.
constexpr double kUnitRoundoff = 0.5 * std::numeric_limits<double>::epsilon();

/// The next double above \p value: an upper bound of the exact result of the
/// one rounded operation that produced \p value, since that result lies
/// within half a unit in the last place of it. Underflow is covered, because
/// the next double above zero is the least subnormal number.
double roundedUp(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}

/// The next double toward zero below a non-negative \p value: a lower bound
/// of the exact non-negative result of the one rounded operation that
/// produced \p value.
double roundedDown(double value) { return std::nextafter(value, 0.0); }

/// An upper bound of \f$ \gamma_k=ku/(1-ku) \f$. The product \f$ ku \f$ and
/// the difference \f$ 1-ku \f$ are exact in double precision for every count
/// below \f$ 2^{52} \f$, so the quotient is the one rounded operation.
double gammaBound(int count) {
  const double scaled = static_cast<double>(count) * kUnitRoundoff;
  return roundedUp(scaled / (1.0 - scaled));
}

/// A finite nonzero complex number as \f$ (x+iy)\,2^{e} \f$ with
/// \f$ \tfrac12\le\max(|x|,|y|)<1 \f$. The scaling is by a power of two and
/// is exact unless the smaller component underflows.
struct ScaledComplex {
  double real = 0.0;
  double imaginary = 0.0;
  int exponent = 0;
};

ScaledComplex scaledComplex(complexd value) {
  ScaledComplex scaled;
  std::frexp(std::max(std::abs(value.real()), std::abs(value.imag())),
             &scaled.exponent);
  scaled.real = std::scalbn(value.real(), -scaled.exponent);
  scaled.imaginary = std::scalbn(value.imag(), -scaled.exponent);
  return scaled;
}

/// A lower and an upper bound of the modulus of a complex number with finite
/// components, from the scaled squares, their sum, its root and the scaling
/// back, each rounded downward for the one and upward for the other.
struct ModulusBounds {
  double lower = 0.0;
  double upper = 0.0;
};

ModulusBounds modulusBounds(complexd value) {
  const ScaledComplex scaled = scaledComplex(value);
  const double realSquare = scaled.real * scaled.real;
  const double imaginarySquare = scaled.imaginary * scaled.imaginary;
  ModulusBounds bounds;
  bounds.lower = roundedDown(std::scalbn(
      roundedDown(std::sqrt(roundedDown(roundedDown(realSquare) +
                                        roundedDown(imaginarySquare)))),
      scaled.exponent));
  bounds.upper = roundedUp(std::scalbn(
      roundedUp(std::sqrt(roundedUp(roundedUp(realSquare) +
                                    roundedUp(imaginarySquare)))),
      scaled.exponent));
  return bounds;
}

/// \f$ 1/F=\bar s/|s|^2\cdot2^{-e} \f$ from the scaled \f$ F=s\,2^{e} \f$:
/// two squares, their sum and one division per component, so each component
/// carries three roundings (`VillainCharacter`, the rounding bound).
complexd complexReciprocal(complexd value) {
  const ScaledComplex scaled = scaledComplex(value);
  const double square = scaled.real * scaled.real +
                        scaled.imaginary * scaled.imaginary;
  return complexd{std::scalbn(scaled.real / square, -scaled.exponent),
                  std::scalbn(-scaled.imaginary / square, -scaled.exponent)};
}

/// The complex product formed as \f$ (ac-bd,\ ad+bc) \f$, whose relative
/// error is at most \f$ \sqrt2\,\gamma_2 \f$ (Higham, Lemma 3.5).
complexd complexProduct(complexd left, complexd right) {
  return complexd{left.real() * right.real() - left.imag() * right.imag(),
                  left.real() * right.imag() + left.imag() * right.real()};
}

/// Whether both components of a complex number are finite.
bool finiteComplex(complexd value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

/// A number as text with three significant digits (printf's "%.3g"), for the
/// messages of a refusal: `std::to_string` prints six fixed decimals and would
/// show a rounding bound of 1e-15 as zero.
std::string threeDigits(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof buffer, "%.3g", value);
  return buffer;
}

/// A complex number as text with every digit of its two doubles (printf's
/// "%.17g"), so that a refusal names the point it refuses.
std::string pointText(complexd value) {
  char buffer[80];
  std::snprintf(buffer, sizeof buffer, "(%.17g, %.17g)", value.real(),
                value.imag());
  return buffer;
}

void requireHolonomy(complexd holonomy, const char *caller) {
  if (!finiteComplex(holonomy) || holonomy == complexd{0.0, 0.0})
    throw std::invalid_argument(
        std::string("VillainCharacter::") + caller +
        ": the face holonomy must be a finite nonzero complex number");
}

}  // namespace

VillainCharacter::VillainCharacter(double beta, int order)
    : beta_(beta), order_(order) {
  if (!(beta > 0.0) || !std::isfinite(beta))
    throw std::invalid_argument(
        "VillainCharacter: the heat-kernel coupling beta must be positive and "
        "finite; got " + std::to_string(beta));
  if (order < 1 || order > maximumOrder)
    throw std::invalid_argument(
        "VillainCharacter: the order of the Villain weight is an integer "
        "from 1 to " + std::to_string(maximumOrder) + "; got " +
        std::to_string(order));
  coefficients_.reserve(static_cast<std::size_t>(order_) + 1);
  for (int m = 0; m <= order_; ++m) {
    const double md = static_cast<double>(m);
    coefficients_.push_back(std::exp(-md * md / (2.0 * beta_)));
  }
  const VillainSeries trivial = series(complexd{1.0, 0.0});
  secondMoment_ = trivial.second.real() / trivial.value.real();
}

VillainSeries VillainCharacter::series(complexd holonomy) const {
  requireHolonomy(holonomy, "series");
  const complexd inverse = complexReciprocal(holonomy);
  const ModulusBounds modulus = modulusBounds(holonomy);
  const double inverseModulus = roundedUp(1.0 / modulus.lower);

  VillainSeries out;
  out.order = order_;
  out.value = complexd{1.0, 0.0};
  out.magnitude = 1.0;
  complexd up{1.0, 0.0};
  complexd down{1.0, 0.0};
  double upModulus = 1.0;
  double downModulus = 1.0;
  for (int m = 1; m <= order_; ++m) {
    up = complexProduct(up, holonomy);
    down = complexProduct(down, inverse);
    const double md = static_cast<double>(m);
    const double coefficient = coefficients_[static_cast<std::size_t>(m)];
    const double firstCoefficient = coefficient * md;
    const double secondCoefficient = firstCoefficient * md;
    const complexd sum{up.real() + down.real(), up.imag() + down.imag()};
    const complexd difference{up.real() - down.real(),
                              up.imag() - down.imag()};
    out.value = complexd{out.value.real() + coefficient * sum.real(),
                         out.value.imag() + coefficient * sum.imag()};
    out.first =
        complexd{out.first.real() + firstCoefficient * difference.real(),
                 out.first.imag() + firstCoefficient * difference.imag()};
    out.second = complexd{out.second.real() + secondCoefficient * sum.real(),
                          out.second.imag() + secondCoefficient * sum.imag()};
    // The sums of the moduli of the terms, |F|^m and |F|^-m from the bounds
    // of |F|, every operation rounded upward.
    upModulus = roundedUp(upModulus * modulus.upper);
    downModulus = roundedUp(downModulus * inverseModulus);
    const double size = roundedUp(upModulus + downModulus);
    out.magnitude = roundedUp(out.magnitude + roundedUp(coefficient * size));
    const double firstWeight = roundedUp(coefficient * md);
    const double secondWeight = roundedUp(firstWeight * md);
    out.firstMagnitude =
        roundedUp(out.firstMagnitude + roundedUp(firstWeight * size));
    out.secondMagnitude =
        roundedUp(out.secondMagnitude + roundedUp(secondWeight * size));
  }
  out.roundingBound =
      finiteComplex(out.value)
          ? roundedUp(gammaBound(6 * order_ + 1) * out.magnitude)
          : std::numeric_limits<double>::infinity();

  // The reported tails (see VillainSeries). log|F| is read from the scaled
  // modulus, so it is finite for every finite nonzero holonomy.
  const ScaledComplex scaled = scaledComplex(holonomy);
  const double logR = std::abs(
      std::log(std::sqrt(scaled.real * scaled.real +
                         scaled.imaginary * scaled.imaginary)) +
      static_cast<double>(scaled.exponent) * std::numbers::ln2);
  // t_m = m^k q^{m^2} r^m, with the index as a double.
  auto term = [&](double m, int k) {
    return std::pow(m, k) * std::exp(-m * m / (2.0 * beta_) + m * logR);
  };
  // rho_k(N), the bound on every ratio t_{m+1}/t_m past N.
  auto ratio = [&](double index, int k) {
    return std::pow((index + 2.0) / (index + 1.0), k) *
           std::exp(-(2.0 * index + 3.0) / (2.0 * beta_) + logR);
  };
  auto tail = [&](int k) {
    double listed = 0.0;
    double index = static_cast<double>(order_);
    // The ratio bound decreases to zero in the index, so the loop ends at the
    // least index at which the geometric bound exists.
    while (!(ratio(index, k) < 1.0)) {
      index += 1.0;
      listed += 2.0 * term(index, k);
    }
    return listed + 2.0 * term(index + 1.0, k) / (1.0 - ratio(index, k));
  };
  out.valueTail = tail(0);
  out.firstTail = tail(1);
  out.secondTail = tail(2);
  return out;
}

bool VillainCharacter::certifiedNonzero(const VillainSeries &point) {
  // A sum that is not finite carries an infinite bound, which nothing exceeds.
  return modulusBounds(point.value).lower > point.roundingBound;
}

double VillainCharacter::secondDerivativeBound(double upperModulus,
                                               double lowerModulus) const {
  const double inverseModulus = roundedUp(1.0 / lowerModulus);
  double upPower = 1.0;
  double downPower = 1.0;
  double bound = 0.0;
  for (int m = 1; m <= order_; ++m) {
    const double md = static_cast<double>(m);
    upPower = roundedUp(upPower * upperModulus);
    downPower = roundedUp(downPower * inverseModulus);
    const double weight = roundedUp(
        roundedUp(coefficients_[static_cast<std::size_t>(m)] * md) * md);
    bound = roundedUp(bound +
                      roundedUp(weight * roundedUp(upPower + downPower)));
  }
  return bound;
}

double VillainCharacter::changeBound(const VillainSeries &node,
                                     double secondBound, double length) const {
  // |DW_M| at the node: the modulus of the computed sum plus its rounding
  // bound gamma_{6M+2} sum |m| c_m |F|^m.
  const double slope = roundedUp(
      modulusBounds(node.first).upper +
      roundedUp(gammaBound(6 * order_ + 2) * node.firstMagnitude));
  const double linear = roundedUp(slope * length);
  const double quadratic =
      roundedUp(0.5 * roundedUp(roundedUp(secondBound * length) * length));
  return roundedUp(linear + quadratic);
}

complexd VillainCharacter::logarithm(complexd holonomy) const {
  requireHolonomy(holonomy, "logarithm");
  const ModulusBounds modulus = modulusBounds(holonomy);
  const ScaledComplex scaled = scaledComplex(holonomy);
  // s_0 = fl(1/|F|), the scale of the first node.
  const double startScale =
      1.0 / std::scalbn(std::sqrt(scaled.real * scaled.real +
                                  scaled.imaginary * scaled.imaginary),
                        scaled.exponent);
  if (!(startScale > 0.0) || !std::isfinite(startScale))
    throw std::domain_error(
        "VillainCharacter::logarithm: the scale 1/|F| that carries the "
        "holonomy " + pointText(holonomy) +
        " to the unit circle is not a finite positive double, so the radial "
        "path has no first node and log W_M has no certified value there");

  // |F| s_0, the modulus of the ray's point at the first node, bounded above
  // and below.
  const double startUpper = roundedUp(modulus.upper * startScale);
  const double startLower = roundedDown(modulus.lower * startScale);
  // omega = u + eta / min(|F|, |F| s_0) and epsilon = omega / (1 - omega):
  // the distance of a computed node from the point of the ray it stands for.
  const double relativeDistance = roundedUp(
      kUnitRoundoff +
      roundedUp(std::numeric_limits<double>::denorm_min() /
                std::min(modulus.lower, startLower)));
  const double epsilon =
      roundedUp(relativeDistance / roundedDown(1.0 - relativeDistance));
  // e^{epsilon} <= 1/(1 - epsilon) and e^{-epsilon} >= 1 - epsilon.
  const double widen = roundedUp(1.0 / roundedDown(1.0 - epsilon));
  const double narrow = roundedDown(1.0 - epsilon);

  auto node = [&](double scale) {
    return scale == 1.0 ? holonomy
                        : complexd{holonomy.real() * scale,
                                   holonomy.imag() * scale};
  };

  // The start: the stretch of the ray between the unit circle and the first
  // node, of Maurer-Cartan length at most lambda, plus epsilon to the node.
  VillainSeries current = series(node(startScale));
  const double lambda =
      std::max({roundedUp(startUpper - 1.0),
                roundedUp(roundedUp(1.0 / startLower) - 1.0), 0.0});
  const double startBound = roundedUp(
      changeBound(current,
                  secondDerivativeBound(
                      roundedUp(std::max(1.0, startUpper) * widen),
                      roundedDown(std::min(1.0, startLower) * narrow)),
                  roundedUp(lambda + epsilon)) +
      current.roundingBound);
  if (std::abs(current.value.imag()) > startBound)
    throw std::logic_error(
        "VillainCharacter::logarithm: the computed W_M at the start of the "
        "radial path to the holonomy " + pointText(holonomy) +
        " has imaginary part " + threeDigits(current.value.imag()) +
        ", beyond its bound " + threeDigits(startBound) +
        ", which contradicts the reality of W_M on the unit circle");
  if (!(current.value.real() > startBound))
    throw std::domain_error(
        "VillainCharacter::logarithm: W_M at the point of the unit circle on "
        "the ray of the holonomy " + pointText(holonomy) + ", " +
        threeDigits(current.value.real()) + ", does not exceed its bound " +
        threeDigits(startBound) +
        ": W_M is zero, negative or below its rounding there, so log W_M has "
        "no value real on the unit circle on this ray");
  complexd accumulated{std::log(current.value.real()), 0.0};

  double currentScale = startScale;
  std::optional<VillainSeries> end;
  while (currentScale != 1.0) {
    // The first candidate is the whole remaining path; a candidate that is
    // not certified is replaced by the geometric mean of the two scales.
    if (!end) end = series(holonomy);
    double candidateScale = 1.0;
    VillainSeries candidate = *end;
    for (;;) {
      const double larger = std::max(currentScale, candidateScale);
      const double smaller = std::min(currentScale, candidateScale);
      const double secondBound = secondDerivativeBound(
          roundedUp(roundedUp(modulus.upper * larger) * widen),
          roundedDown(roundedDown(modulus.lower * smaller) * narrow));
      // L = (s_+ - s_-)/s_- + 2 epsilon >= log(s_+/s_-) + 2 epsilon.
      const double length =
          roundedUp(roundedUp(roundedUp(larger - smaller) / smaller) +
                    roundedUp(2.0 * epsilon));
      const double required =
          roundedUp(roundedUp(changeBound(current, secondBound, length) +
                              current.roundingBound) +
                    candidate.roundingBound);
      if (required < modulusBounds(current.value).lower) break;
      const double middle =
          std::sqrt(currentScale) * std::sqrt(candidateScale);
      if (!(smaller < middle && middle < larger))
        throw std::domain_error(
            "VillainCharacter::logarithm: the radial path to the holonomy " +
            pointText(holonomy) + " meets a zero of W_M at the resolution of "
            "double precision near " + pointText(node(currentScale)) +
            ", where |W_M| = " + threeDigits(std::abs(current.value)) +
            ": no step from there is certified, so log W_M has no value on "
            "the branch real on the unit circle");
      candidateScale = middle;
      candidate = series(node(candidateScale));
    }
    accumulated += std::log(candidate.value / current.value);
    currentScale = candidateScale;
    current = candidate;
  }
  if (!certifiedNonzero(current))
    throw std::domain_error(
        "VillainCharacter::logarithm: W_M is not certified nonzero at the "
        "holonomy " + pointText(holonomy) + ": |W_M| = " +
        threeDigits(std::abs(current.value)) +
        " does not exceed its rounding bound " +
        threeDigits(current.roundingBound) +
        ", so log W_M has no certified value there");
  return accumulated;
}

complexd VillainCharacter::potential(complexd holonomy) const {
  return -matchedWeight() * logarithm(holonomy);
}

namespace {

VillainSeries certifiedSeries(const VillainCharacter &character,
                              complexd holonomy) {
  VillainSeries point = character.series(holonomy);
  // The derivatives DW_M/W_M and D^2W_M/W_M are formed with W_M as summed
  // wherever it is not exactly zero. Whether its modulus exceeds its
  // rounding bound there is a measurement of the point
  // (`VillainCharacter::certifiedNonzero`, `HolonomyTruncation`), reported
  // and not required.
  if (point.value == complexd{0.0, 0.0} || !std::isfinite(point.value.real()) ||
      !std::isfinite(point.value.imag()))
    throw std::domain_error(
        "VillainCharacter: W_M is " +
        std::string(point.value == complexd{0.0, 0.0} ? "exactly zero"
                                                      : "not finite") +
        " at the holonomy " + pointText(holonomy) +
        ", so the potential's derivatives DW_M/W_M and D^2W_M/W_M have no "
        "value there");
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

/// The per-face potential of the holonomy term and its first two
/// Maurer-Cartan derivatives, \f$ \phi \f$, \f$ D\phi \f$ and \f$ D^2\phi \f$
/// with \f$ D=F\,d/dF \f$, the weight included. The Villain character is built
/// once per query and shared by every face.
class FacePotential {
 public:
  explicit FacePotential(const JointActionDeclaration &declaration)
      : weight_(declaration.holonomyWeight) {
    if (weight_ > 0.0)
      villain_.emplace(weight_, declaration.villainOrder);
  }

  [[nodiscard]] bool active() const { return villain_.has_value(); }

  [[nodiscard]] complexd value(complexd holonomy) const {
    if (!active()) return complexd{0.0, 0.0};
    return villain_->potential(holonomy);
  }

  [[nodiscard]] complexd first(complexd holonomy) const {
    if (!active()) return complexd{0.0, 0.0};
    return villain_->firstDerivative(holonomy);
  }

  [[nodiscard]] complexd second(complexd holonomy) const {
    if (!active()) return complexd{0.0, 0.0};
    return villain_->secondDerivative(holonomy);
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
  for (const auto &constraint : declaration_.momentConstraints) {
    if (constraint.form == SpectralConstraintForm::PowerSum &&
        constraint.order < 1)
      throw std::invalid_argument(
          "JointAction: a spectral moment order is the power j >= 1 of "
          "p_j(h) = tr(h^j); got " +
          std::to_string(constraint.order));
    if (constraint.form == SpectralConstraintForm::BandMean &&
        constraint.band >= declaration_.momentBandProjectors.size())
      throw std::invalid_argument(
          "JointAction: a band-mean constraint names band projector " +
          std::to_string(constraint.band) + ", but " +
          std::to_string(declaration_.momentBandProjectors.size()) +
          " band projectors are declared");
  }
  {
    if (declaration_.holonomyWeight < 0.0)
      throw std::invalid_argument(
          "JointAction: the Villain holonomy term's coupling beta is a "
          "heat-kernel time and must be non-negative; got " +
          std::to_string(declaration_.holonomyWeight));
    if (declaration_.villainOrder < 1 ||
        declaration_.villainOrder > VillainCharacter::maximumOrder)
      throw std::invalid_argument(
          "JointAction: the order of the Villain weight is an integer from 1 "
          "to " + std::to_string(VillainCharacter::maximumOrder) + "; got " +
          std::to_string(declaration_.villainOrder));
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
  if (!(declaration_.momentScale > 0.0) ||
      !std::isfinite(declaration_.momentScale))
    throw std::invalid_argument(
        "JointAction: the unit the power sums are measured in must be "
        "positive and finite; got " + std::to_string(declaration_.momentScale));
  if (!declaration_.momentProjector.empty()) {
    const ChainComplex complex = ChainComplex::fromSpacetime(*spacetime_);
    const std::size_t order =
        declaration_.carrierDegree <= complex.dimension()
            ? complex.numSimplices(declaration_.carrierDegree)
            : std::size_t{0};
    if (declaration_.momentProjector.size() != order * order)
      throw std::invalid_argument(
          "JointAction: the constraints' fiber projector is a square matrix "
          "over the " + std::to_string(order) + " cells of degree " +
          std::to_string(declaration_.carrierDegree) + "; got " +
          std::to_string(declaration_.momentProjector.size()) + " entries");
  }
  if (!declaration_.momentBandProjectors.empty()) {
    const ChainComplex complex = ChainComplex::fromSpacetime(*spacetime_);
    const std::size_t order =
        declaration_.carrierDegree <= complex.dimension()
            ? complex.numSimplices(declaration_.carrierDegree)
            : std::size_t{0};
    for (const auto &projector : declaration_.momentBandProjectors)
      if (projector.size() != order * order)
        throw std::invalid_argument(
            "JointAction: a band projector of the band-mean constraints is a "
            "square matrix over the " + std::to_string(order) +
            " cells of degree " + std::to_string(declaration_.carrierDegree) +
            "; got " + std::to_string(projector.size()) + " entries");
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

void JointAction::setCovariance(std::vector<complexd> covariance) {
  if (!covariance.empty() &&
      covariance.size() != declaration_.covariance.size()) {
    const ChainComplex complex = ChainComplex::fromSpacetime(*spacetime_);
    const std::size_t order =
        declaration_.carrierDegree <= complex.dimension()
            ? complex.numSimplices(declaration_.carrierDegree)
            : std::size_t{0};
    if (covariance.size() != order * order)
      throw std::invalid_argument(
          "JointAction::setCovariance: the covariance is a square matrix over "
          "the " + std::to_string(order) + " cells of degree " +
          std::to_string(declaration_.carrierDegree) + "; got " +
          std::to_string(covariance.size()) + " entries");
  }
  declaration_.covariance = std::move(covariance);
}

void JointAction::setMomentProjector(std::vector<complexd> projector) {
  if (!projector.empty() &&
      projector.size() != declaration_.momentProjector.size()) {
    const ChainComplex complex = ChainComplex::fromSpacetime(*spacetime_);
    const std::size_t order =
        declaration_.carrierDegree <= complex.dimension()
            ? complex.numSimplices(declaration_.carrierDegree)
            : std::size_t{0};
    if (projector.size() != order * order)
      throw std::invalid_argument(
          "JointAction::setMomentProjector: the fiber projector is a square "
          "matrix over the " + std::to_string(order) + " cells of degree " +
          std::to_string(declaration_.carrierDegree) + "; got " +
          std::to_string(projector.size()) + " entries");
  }
  declaration_.momentProjector = std::move(projector);
}

void JointAction::setMomentBandProjectors(
    std::vector<std::vector<complexd>> projectors) {
  if (projectors.size() != declaration_.momentBandProjectors.size())
    throw std::invalid_argument(
        "JointAction::setMomentBandProjectors: one projector per declared "
        "band is required; got " + std::to_string(projectors.size()) +
        " for " + std::to_string(declaration_.momentBandProjectors.size()) +
        " declared");
  for (std::size_t index = 0; index < projectors.size(); ++index)
    if (projectors[index].size() !=
        declaration_.momentBandProjectors[index].size())
      throw std::invalid_argument(
          "JointAction::setMomentBandProjectors: band projector " +
          std::to_string(index) + " has " +
          std::to_string(projectors[index].size()) + " entries; " +
          std::to_string(declaration_.momentBandProjectors[index].size()) +
          " are declared");
  declaration_.momentBandProjectors = std::move(projectors);
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

namespace {

/// The operator the spectral constraints' power sums are taken of: the
/// carrier \f$ h \f$, or its compression
/// \f$ P_{\mathcal C}hP_{\mathcal C} \f$ to the declared fiber, whose
/// nonzero spectrum is that of \f$ h_{\mathcal C} \f$ on
/// \f$ \operatorname{Ran}P_{\mathcal C} \f$.
Eigen::MatrixXcd constrainedOperator(const JointActionDeclaration &declaration,
                                     const Eigen::MatrixXcd &carrier,
                                     std::size_t order) {
  const complexd unit{1.0 / declaration.momentScale, 0.0};
  if (declaration.momentProjector.empty() || order == 0)
    return unit * carrier;
  const Eigen::MatrixXcd projector =
      toMatrix(declaration.momentProjector, order);
  return unit * (projector * carrier * projector);
}

/// The matrix the operator derivative is traced against in
/// \f$ dp_j=\operatorname{tr}(X_j\,dh) \f$ for the power sums in the declared
/// unit \f$ s \f$: \f$ (j/s)(h/s)^{j-1} \f$ by the cyclic identity, exact for
/// every matrix including a defective one, or, for the declared fiber at fixed
/// \f$ P_{\mathcal C} \f$,
/// \f$ (j/s)P_{\mathcal C}(P_{\mathcal C}hP_{\mathcal C}/s)^{j-1}P_{\mathcal C} \f$.
Eigen::MatrixXcd powerSumDerivative(const JointActionDeclaration &declaration,
                                    const Eigen::MatrixXcd &carrier,
                                    std::size_t order, int power) {
  const auto size = static_cast<Eigen::Index>(order);
  const Eigen::MatrixXcd constrained =
      constrainedOperator(declaration, carrier, order);
  Eigen::MatrixXcd result = Eigen::MatrixXcd::Identity(size, size);
  for (int step = 1; step < power; ++step) result = result * constrained;
  if (!declaration.momentProjector.empty()) {
    const Eigen::MatrixXcd projector =
        toMatrix(declaration.momentProjector, order);
    result = projector * result * projector;
  }
  return complexd{static_cast<double>(power) / declaration.momentScale, 0.0} *
         result;
}

/// The matrix the operator derivative is traced against for one declared
/// constraint: `powerSumDerivative` for a power sum, or, for a band mean at
/// fixed \f$ P_b \f$, \f$ P_b/(r_bs) \f$ with \f$ r_b=\operatorname{tr}P_b \f$
/// (zero when the projector has no trace).
Eigen::MatrixXcd constraintDerivative(
    const JointActionDeclaration &declaration, const Eigen::MatrixXcd &carrier,
    std::size_t order, const SpectralMomentConstraint &constraint) {
  if (constraint.form == SpectralConstraintForm::PowerSum)
    return powerSumDerivative(declaration, carrier, order, constraint.order);
  const Eigen::MatrixXcd projector =
      toMatrix(declaration.momentBandProjectors.at(constraint.band), order);
  const complexd rank = projector.trace();
  if (rank == complexd{0.0, 0.0})
    return Eigen::MatrixXcd::Zero(projector.rows(), projector.cols());
  return projector / (rank * declaration.momentScale);
}

/// The variation of \f$ M^p \f$ under \f$ \delta M \f$,
/// \f$ \sum_{m=0}^{p-1}M^m\,\delta M\,M^{p-1-m} \f$; zero for \f$ p=0 \f$.
Eigen::MatrixXcd powerVariation(const Eigen::MatrixXcd &matrix,
                                const Eigen::MatrixXcd &variation, int power) {
  Eigen::MatrixXcd result =
      Eigen::MatrixXcd::Zero(matrix.rows(), matrix.cols());
  if (power <= 0) return result;
  std::vector<Eigen::MatrixXcd> powers;
  powers.push_back(Eigen::MatrixXcd::Identity(matrix.rows(), matrix.cols()));
  for (int m = 1; m < power; ++m) powers.push_back(powers.back() * matrix);
  for (int m = 0; m < power; ++m)
    result += powers[static_cast<std::size_t>(m)] * variation *
              powers[static_cast<std::size_t>(power - 1 - m)];
  return result;
}

/// The variation of `constraintDerivative`'s matrix \f$ X_j \f$ under a
/// variation \p carrierVariation of the carrier at fixed projector and a
/// variation \p projectorVariation of the projector at fixed carrier (either
/// null for none), linear in each:
/// * a power sum of the whole carrier, \f$ X=(j/s)(h/s)^{j-1} \f$:
///   \f$ \delta X=(j/s)\sum_{m}(h/s)^m(\delta h/s)(h/s)^{j-2-m} \f$;
/// * a power sum of the declared fiber, \f$ X=(j/s)PM^{j-1}P \f$ with
///   \f$ M=PhP/s \f$: \f$ \delta M=(\delta P\,hP+P\,\delta h\,P+Ph\,\delta P)/s \f$
///   and \f$ \delta X=(j/s)(\delta P\,M^{j-1}P+P\,\delta(M^{j-1})P+PM^{j-1}\delta P) \f$;
/// * a band mean, \f$ X=P_b/(r_bs) \f$: \f$ \delta X=\delta P_b/(r_bs) \f$, the
///   rank a constant.
Eigen::MatrixXcd constraintDerivativeVariation(
    const JointActionDeclaration &declaration, const Eigen::MatrixXcd &carrier,
    std::size_t order, const SpectralMomentConstraint &constraint,
    const Eigen::MatrixXcd *carrierVariation,
    const Eigen::MatrixXcd *projectorVariation) {
  const auto size = static_cast<Eigen::Index>(order);
  const Eigen::MatrixXcd zero = Eigen::MatrixXcd::Zero(size, size);
  const double scale = declaration.momentScale;
  if (constraint.form == SpectralConstraintForm::BandMean) {
    if (projectorVariation == nullptr) return zero;
    const Eigen::MatrixXcd projector =
        toMatrix(declaration.momentBandProjectors.at(constraint.band), order);
    const complexd rank = projector.trace();
    if (rank == complexd{0.0, 0.0}) return zero;
    return *projectorVariation / (rank * scale);
  }
  const int power = constraint.order;
  const complexd unit{1.0 / scale, 0.0};
  const complexd weight{static_cast<double>(power) / scale, 0.0};
  if (declaration.momentProjector.empty()) {
    if (carrierVariation == nullptr) return zero;
    return weight * powerVariation(unit * carrier, unit * (*carrierVariation),
                                   power - 1);
  }
  const Eigen::MatrixXcd projector =
      toMatrix(declaration.momentProjector, order);
  const Eigen::MatrixXcd compressed = unit * (projector * carrier * projector);
  Eigen::MatrixXcd compressedVariation = zero;
  if (carrierVariation != nullptr)
    compressedVariation += unit * (projector * (*carrierVariation) * projector);
  if (projectorVariation != nullptr)
    compressedVariation +=
        unit * ((*projectorVariation) * carrier * projector +
                projector * carrier * (*projectorVariation));
  Eigen::MatrixXcd result =
      projector * powerVariation(compressed, compressedVariation, power - 1) *
      projector;
  if (projectorVariation != nullptr) {
    Eigen::MatrixXcd powered = Eigen::MatrixXcd::Identity(size, size);
    for (int step = 1; step < power; ++step) powered = powered * compressed;
    result += (*projectorVariation) * powered * projector +
              projector * powered * (*projectorVariation);
  }
  return weight * result;
}

}  // namespace

std::vector<complexd> JointAction::powerSums() const {
  return constraintValues();
}

std::vector<complexd> JointAction::constraintValues() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  std::vector<complexd> sums;
  sums.reserve(declaration_.momentConstraints.size());
  const Eigen::MatrixXcd constrained = constrainedOperator(
      declaration_, workspace.carrier, workspace.carrierOrder);
  for (const auto &constraint : declaration_.momentConstraints) {
    if (workspace.carrierOrder == 0) {
      sums.emplace_back(0.0, 0.0);
      continue;
    }
    if (constraint.form == SpectralConstraintForm::BandMean) {
      // tr(P_b h) / (r_b s): the band's mean eigenvalue in the unit.
      const Eigen::MatrixXcd projector = toMatrix(
          declaration_.momentBandProjectors.at(constraint.band),
          workspace.carrierOrder);
      const complexd rank = projector.trace();
      sums.push_back(rank == complexd{0.0, 0.0}
                         ? complexd{0.0, 0.0}
                         : traceOfProduct(projector, workspace.carrier) /
                               (rank * declaration_.momentScale));
      continue;
    }
    // Repeated multiplication rather than an eigendecomposition: the power sum
    // is defined for a defective operator, and forming it this way needs no
    // eigenvalue ordering and no similarity to diagonal form.
    Eigen::MatrixXcd power = constrained;
    for (int step = 1; step < constraint.order; ++step)
      power = power * constrained;
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

/// The vertex ids of a cell as text, for a refusal that names the cell.
template <typename Ids>
std::string idList(const Ids &ids) {
  std::string out = "[";
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index > 0) out += ", ";
    out += std::to_string(ids[index]);
  }
  return out + "]";
}

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
/// every step until it is fine enough, by bisection down to the resolution of
/// the parameter. \p advance takes the parameter and returns whether the step
/// it made was fine; \p State is copied so that a refused step is undone. The
/// bisection ends by itself: a step that is halved until it leaves the
/// parameter the number it was is the shortest step the datatype resolves,
/// and no shorter one exists. When no step down to that one is fine the sheet
/// cannot be followed along the segment there, which is a refusal of the
/// walk, and no step is accepted in its place. \p what names the quantity
/// walked, for the refusal.
/// @throws std::domain_error when no step the parameter resolves is fine.
template <typename State, typename Advance>
void walkSegment(State &state, Advance advance, const std::string &what) {
  double t = 0.0;
  double step = 1.0;
  while (t < 1.0) {
    const double next = std::min(1.0, t + step);
    if (next == t)
      throw std::domain_error(
          "JointAction: the continued Regge sheets cannot be followed: " +
          what + " makes no fine step from parameter " +
          std::to_string(t) + " of the segment at any step the parameter "
          "resolves (a fine step turns every root by at most pi/4 and moves "
          "every angle by at most 0.25)");
    State trial = state;
    if (advance(trial, next)) {
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
      walkSegment(
          walk,
          [&, from = from, to = to](CellWalk &trial, double t) {
            return trial.advance(Simplex::cofactorMatrix(
                cayleyMenger(cellIds, *from, *to, t), n));
          },
          "the dihedral angles of the cell on vertices " + idList(cellIds));
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
        walkSegment(
            content,
            [&, from = from, to = to](SheetedSqrt &trial, double t) {
              trial.advance(gramDeterminant(hingeIds, *from, *to, t));
              return std::abs(trial.lastStep()) <= kMaximumRootTurn;
            },
            "the content root of the hinge on vertices " + idList(hingeIds));
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
  return reggeTerm() + holonomyTerm() + matterTerm() +
         spectralTerm();
}

ReportedActionValue JointAction::reportedValue() const {
  ReportedActionValue reported;
  try {
    reported.value = value();
  } catch (const std::logic_error &refusal) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    reported.available = false;
    reported.value = complexd{nan, nan};
    reported.unavailable = refusal.what();
  }
  return reported;
}

namespace {

/// The matrix every operator derivative is contracted against in the
/// stationarity equations: \f$ A = w_M\Gamma + \sum_j \xi_j\,X_j \f$, with
/// \f$ X_j=jh^{j-1} \f$, or its compression to the declared fiber
/// (`powerSumDerivative`).
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
    matrix += constraint.multiplier *
              constraintDerivative(declaration, carrier, order, constraint);
  }
  return matrix;
}

}  // namespace

std::vector<complexd> JointAction::lengthStationarity() const {
  std::vector<complexd> stationarity;
  stationarityPart(StationarityPart::All, nullptr, &stationarity, nullptr);
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
  std::vector<complexd> stationarity;
  stationarityPart(StationarityPart::All, nullptr, nullptr, &stationarity);
  return stationarity;
}

namespace {

/// Add \f$ \operatorname{tr}(A\,\partial h/\partial z_e) \f$ to \p lengths and
/// \f$ \operatorname{tr}(A\,U_e\partial h/\partial U_e) \f$ on the stored
/// orientation to \p links, per edge, for the contraction matrix \p matrix
/// (a null output is not formed). Nothing is added when the matrix is zero.
void contractionSweep(const ActionWorkspace &workspace, int carrierDegree,
                      const Eigen::MatrixXcd &matrix,
                      std::vector<complexd> *lengths,
                      std::vector<complexd> *links) {
  if (workspace.carrierOrder == 0 || matrix.isZero(0.0)) return;
  const std::size_t edges = workspace.edges.size();
  for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
    const auto *edge = workspace.edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const std::uint64_t source = edge->getSource()->getId();
    const std::uint64_t target = edge->getTarget()->getId();
    if (lengths) {
      const auto derivative =
          workspace.hodge.laplacianGradient(carrierDegree, source, target);
      if (derivative.size() == workspace.carrierOrder * workspace.carrierOrder)
        (*lengths)[edgeIndex] += traceOfProduct(
            matrix, toMatrix(derivative, workspace.carrierOrder));
    }
    if (links) {
      const auto derivative = workspace.hodge.laplacianPhaseGradient(
          carrierDegree, source, target);
      if (derivative.size() != workspace.carrierOrder * workspace.carrierOrder)
        continue;
      // U d/dU = -i d/dphi on the canonical link U = e^{i phi}: a relation
      // between derivatives, which fixes no branch of the logarithm. The
      // stored orientation carries the same current up to the sign that
      // relates it to the canonical one, since j_yx = -j_xy.
      const complexd canonicalPart =
          complexd{0.0, -1.0} *
          traceOfProduct(matrix, toMatrix(derivative, workspace.carrierOrder));
      (*links)[edgeIndex] += workspace.storedSign[edgeIndex] * canonicalPart;
    }
  }
}

}  // namespace

void JointAction::stationarityPart(StationarityPart part,
                                   const std::vector<complexd> *contraction,
                                   std::vector<complexd> *lengths,
                                   std::vector<complexd> *links) const {
  const bool wantContraction = part == StationarityPart::All ||
                               part == StationarityPart::Contraction;
  const bool wantCarrier =
      part == StationarityPart::Contraction ||
      (part == StationarityPart::All && carrierIsNeeded(declaration_));
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource, wantCarrier);
  const std::size_t edges = workspace.edges.size();
  if (lengths) lengths->assign(edges, complexd{0.0, 0.0});
  if (links) links->assign(edges, complexd{0.0, 0.0});
  const bool regge = lengths != nullptr &&
                     (part == StationarityPart::All ||
                      part == StationarityPart::Regge);
  const bool holonomy = links != nullptr &&
                        (part == StationarityPart::All ||
                         part == StationarityPart::Holonomy);

  if (regge && declaration_.gravitationalWeight != 0.0 &&
      declaration_.reggeForm == ReggeForm::Dual) {
    const auto reggeGradient =
        ReggeSolver(spacetime_, MatterConfiguration()).actionGradientExact();
    for (std::size_t edgeIndex = 0;
         edgeIndex < edges && edgeIndex < reggeGradient.size(); ++edgeIndex)
      (*lengths)[edgeIndex] +=
          declaration_.gravitationalWeight * reggeGradient[edgeIndex];
  }

  if (regge && declaration_.gravitationalWeight != 0.0 &&
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
          (*lengths)[found->second] +=
              declaration_.gravitationalWeight * sign * derivative * deficit;
      }
      for (const auto &[key, derivative] :
           hinge->deficitAngleGradient(entry.angles)) {
        const auto found = indexOfEdge.find(pairKey(key.first, key.second));
        if (found != indexOfEdge.end())
          (*lengths)[found->second] +=
              declaration_.gravitationalWeight * content * derivative;
      }
    }
  }

  if (holonomy) {
    const auto holonomyPart =
        holonomyLinkDerivative(workspace, FacePotential(declaration_));
    for (std::size_t edgeIndex = 0; edgeIndex < edges; ++edgeIndex) {
      const long long canonical = workspace.canonicalOfEdge[edgeIndex];
      if (canonical < 0 ||
          static_cast<std::size_t>(canonical) >= holonomyPart.size())
        continue;
      (*links)[edgeIndex] +=
          workspace.storedSign[edgeIndex] *
          holonomyPart[static_cast<std::size_t>(canonical)];
    }
  }

  if (!wantContraction || workspace.carrierOrder == 0) return;
  const Eigen::MatrixXcd matrix =
      part == StationarityPart::Contraction
          ? toMatrix(*contraction, workspace.carrierOrder)
          : contractionMatrix(declaration_, workspace.carrier,
                              workspace.carrierOrder);
  contractionSweep(workspace, declaration_.carrierDegree, matrix, lengths,
                   links);
}

std::vector<ActionTermGradient> JointAction::termGradients() const {
  const std::size_t edges = edgeCount();
  std::vector<ActionTermGradient> terms;
  auto add = [&](const std::string &name, const std::string &label,
                 complexd weight, complexd value, bool factored,
                 StationarityPart part, const std::vector<complexd> *matrix,
                 bool wantLengths, bool wantLinks) {
    ActionTermGradient term;
    term.name = name;
    term.label = label;
    term.weight = weight;
    term.value = value;
    term.factored = factored;
    term.bare = !factored ? value
                : weight != complexd{0.0, 0.0} ? value / weight
                                               : complexd{0.0, 0.0};
    term.lengthStationarity.assign(edges, complexd{0.0, 0.0});
    term.linkStationarity.assign(edges, complexd{0.0, 0.0});
    if (wantLengths || wantLinks)
      stationarityPart(part, matrix,
                       wantLengths ? &term.lengthStationarity : nullptr,
                       wantLinks ? &term.linkStationarity : nullptr);
    terms.push_back(std::move(term));
  };
  add("regge", "(1/kappa) S_Regge",
      complexd{declaration_.gravitationalWeight, 0.0}, reggeTerm(), true,
      StationarityPart::Regge, nullptr,
      declaration_.gravitationalWeight != 0.0, false);
  complexd holonomy{0.0, 0.0};
  try {
    holonomy = holonomyTerm();
  } catch (const std::domain_error &) {
    holonomy = complexd{std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN()};
  }
  add("holonomy", "beta S_hol", complexd{declaration_.holonomyWeight, 0.0},
      holonomy, false, StationarityPart::Holonomy, nullptr, false,
      declaration_.holonomyWeight != 0.0);
  // The mean-field term: the contraction w_m Gamma alone.
  std::vector<complexd> weighted = declaration_.covariance;
  for (complexd &entry : weighted) entry *= declaration_.matterWeight;
  const bool matter = declaration_.matterWeight != 0.0 && !weighted.empty();
  add("matter", "w_m tr(Gamma h_1)", complexd{declaration_.matterWeight, 0.0},
      matterTerm(), true, StationarityPart::Contraction, &weighted, matter,
      matter);
  // Each constraint: xi_j times the analytic gradient of c_j.
  const std::vector<complexd> residuals = momentResiduals();
  for (std::size_t index = 0; index < declaration_.momentConstraints.size();
       ++index) {
    const auto &constraint = declaration_.momentConstraints[index];
    const complexd multiplier = constraint.multiplier;
    const std::string j = std::to_string(index + 1);
    ActionTermGradient term;
    term.name = "constraint " + j;
    term.label =
        "xi_" + j + " (c_" + j + " - c_" + j + "*), c_" + j + " = " +
        (constraint.form == SpectralConstraintForm::BandMean
             ? "lambda_" + std::to_string(constraint.band + 1) + " / s"
             : "p_" + std::to_string(constraint.order) + "(h_C / s)");
    term.weight = multiplier;
    term.bare = residuals[index];
    term.factored = true;
    term.value = multiplier * residuals[index];
    const std::vector<complexd> gradient = momentGradient(index);
    term.lengthStationarity.assign(edges, complexd{0.0, 0.0});
    term.linkStationarity.assign(edges, complexd{0.0, 0.0});
    for (std::size_t edge = 0; edge < edges && edge < gradient.size(); ++edge)
      term.lengthStationarity[edge] = multiplier * gradient[edge];
    for (std::size_t edge = 0; edge < edges && edges + edge < gradient.size();
         ++edge)
      term.linkStationarity[edge] = multiplier * gradient[edges + edge];
    terms.push_back(std::move(term));
  }
  return terms;
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



std::vector<complexd> JointAction::reggeHessian() const {
  const std::size_t edges = edgeCount();
  std::vector<complexd> hessian(edges * edges, complexd{0.0, 0.0});
  const double weight = declaration_.gravitationalWeight;
  if (weight == 0.0 || edges == 0) return hessian;
  if (declaration_.reggeForm == ReggeForm::Dual) {
    const auto dense =
        ReggeSolver(spacetime_, MatterConfiguration()).actionHessianExact();
    for (std::size_t e = 0; e < edges && e < dense.size(); ++e)
      for (std::size_t f = 0; f < edges && f < dense[e].size(); ++f)
        hessian[e * edges + f] = weight * dense[e][f];
    return hessian;
  }
  using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;
  std::map<EdgeKey, std::size_t> indexOfEdge;
  const auto meshEdges = spacetime_->getEdgeList()->toVector();
  for (std::size_t edgeIndex = 0; edgeIndex < edges && edgeIndex < meshEdges.size();
       ++edgeIndex) {
    const auto *edge = meshEdges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    indexOfEdge[pairKey(edge->getSource()->getId(),
                        edge->getTarget()->getId())] = edgeIndex;
  }
  auto add = [&](const EdgeKey &rowKey, const EdgeKey &columnKey,
                 complexd value) {
    const auto row = indexOfEdge.find(pairKey(rowKey.first, rowKey.second));
    const auto column =
        indexOfEdge.find(pairKey(columnKey.first, columnKey.second));
    if (row == indexOfEdge.end() || column == indexOfEdge.end()) return;
    hessian[row->second * edges + column->second] += value;
  };
  // d^2(sum_h |h| eps_h) = sum_h (d^2|h| eps_h + d|h| d eps_h + d eps_h d|h|
  // + |h| d^2 eps_h), every root and inverse cosine on its declared sheet.
  for (const auto &entry : reggeSheets().hinges) {
    auto *hinge = entry.hinge;
    const double sign = static_cast<double>(entry.contentSign);
    const complexd content = sign * ReggeSolver::hingeContent(hinge);
    const complexd deficit = hinge->deficitAngle(entry.angles);
    const auto contentGradient = hinge->volumeGradient();
    const auto deficitGradient = hinge->deficitAngleGradient(entry.angles);
    for (const auto &[keyF, unused] : contentGradient) {
      (void)unused;
      const std::map<EdgeKey, complexd> unit{{keyF, complexd{1.0, 0.0}}};
      for (const auto &[keyE, second] :
           hinge->volumeGradientDirectionalDerivative(unit))
        add(keyE, keyF, weight * sign * second * deficit);
    }
    for (const auto &[keyE, contentDerivative] : contentGradient)
      for (const auto &[keyF, deficitDerivative] : deficitGradient) {
        add(keyE, keyF, weight * sign * contentDerivative * deficitDerivative);
        add(keyF, keyE, weight * sign * contentDerivative * deficitDerivative);
      }
    for (const auto &[keys, second] : hinge->deficitAngleHessian(entry.angles))
      add(keys.first, keys.second, weight * content * second);
  }
  return hessian;
}

std::vector<complexd> JointAction::actionHessian(bool lengths,
                                                 bool links) const {
  const std::size_t edges = edgeCount();
  const std::size_t size = 2 * edges;
  Eigen::MatrixXcd hessian = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(size), static_cast<Eigen::Index>(size));
  auto at = [&](std::size_t row, std::size_t column) -> complexd & {
    return hessian(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(column));
  };
  if (lengths && declaration_.gravitationalWeight != 0.0) {
    const std::vector<complexd> regge = reggeHessian();
    for (std::size_t e = 0; e < edges; ++e)
      for (std::size_t f = 0; f < edges; ++f) at(e, f) += regge[e * edges + f];
  }
  if (links && declaration_.holonomyWeight > 0.0) {
    const std::vector<complexd> villain = holonomyHessian();
    for (std::size_t e = 0; e < edges; ++e)
      for (std::size_t f = 0; f < edges; ++f)
        at(edges + e, edges + f) += villain[e * edges + f];
  }
  if (!carrierIsNeeded(declaration_)) return toFlat(hessian);

  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  const std::size_t order = workspace.carrierOrder;
  if (order == 0) return toFlat(hessian);
  const int degree = declaration_.carrierDegree;
  const std::size_t cells = order * order;
  const Eigen::MatrixXcd matrix =
      contractionMatrix(declaration_, workspace.carrier, order);

  // The first derivatives of h in the relaxed coordinates: in z_e, and in the
  // Maurer-Cartan increment on the stored orientation, U d/dU = -i s d/dphi
  // with s the stored sign.
  std::vector<bool> carried(edges, false);
  std::vector<std::uint64_t> sources(edges, 0);
  std::vector<std::uint64_t> targets(edges, 0);
  std::vector<Eigen::MatrixXcd> lengthDerivative(edges);
  std::vector<Eigen::MatrixXcd> linkDerivative(edges);
  for (std::size_t e = 0; e < edges; ++e) {
    const auto *edge = workspace.edges[e];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    sources[e] = edge->getSource()->getId();
    targets[e] = edge->getTarget()->getId();
    const auto dz = workspace.hodge.laplacianGradient(degree, sources[e],
                                                      targets[e]);
    const auto dphi = workspace.hodge.laplacianPhaseGradient(
        degree, sources[e], targets[e]);
    if (dz.size() != cells || dphi.size() != cells) continue;
    carried[e] = true;
    lengthDerivative[e] = toMatrix(dz, order);
    linkDerivative[e] = complexd{0.0, -workspace.storedSign[e]} *
                        toMatrix(dphi, order);
  }
  const bool contracted = !matrix.isZero(0.0);

  // The connected component of the complex each edge lies in. The operator
  // is block diagonal over the components (the sheets of a sheeted support
  // are components), so its second derivative in two edges of different
  // components is exactly zero, and those pairs are not formed.
  std::map<std::uint64_t, std::uint64_t> root;
  const std::function<std::uint64_t(std::uint64_t)> find =
      [&](std::uint64_t vertex) {
        auto found = root.find(vertex);
        if (found == root.end()) found = root.emplace(vertex, vertex).first;
        if (found->second == vertex) return vertex;
        const std::uint64_t top = find(found->second);
        root[vertex] = top;
        return top;
      };
  for (std::size_t e = 0; e < edges; ++e) {
    if (!carried[e]) continue;
    const std::uint64_t a = find(sources[e]);
    const std::uint64_t b = find(targets[e]);
    if (a != b) root[std::max(a, b)] = std::min(a, b);
  }
  std::vector<std::uint64_t> component(edges, 0);
  for (std::size_t e = 0; e < edges; ++e)
    if (carried[e]) component[e] = find(sources[e]);

  // tr(A d^2 h): the length block from the directional derivatives of the
  // gradient, one direction per edge; the link block and the mixed block from
  // the phase Hessian and the mixed derivative, with -1 = (-i)^2 and -i on
  // the stored signs.
  if (contracted) {
    for (std::size_t f = 0; f < edges; ++f) {
      if (!carried[f]) continue;
      std::vector<complexd> direction(edges, complexd{0.0, 0.0});
      direction[f] = complexd{1.0, 0.0};
      const auto second =
          workspace.hodge.laplacianGradientDirectionalDerivative(degree,
                                                                 direction);
      for (std::size_t e = 0; e < edges; ++e) {
        if (!carried[e] || second[e].size() != cells) continue;
        at(e, f) += traceOfProduct(matrix, toMatrix(second[e], order));
      }
    }
    for (std::size_t e = 0; e < edges; ++e) {
      if (!carried[e]) continue;
      for (std::size_t f = 0; f < edges; ++f) {
        if (!carried[f] || component[e] != component[f]) continue;
        const auto phase = workspace.hodge.laplacianPhaseHessian(
            degree, sources[e], targets[e], sources[f], targets[f]);
        if (phase.size() == cells)
          at(edges + e, edges + f) +=
              -workspace.storedSign[e] * workspace.storedSign[f] *
              traceOfProduct(matrix, toMatrix(phase, order));
        const auto mixed = workspace.hodge.laplacianMixedDerivative(
            degree, sources[e], targets[e], sources[f], targets[f]);
        if (mixed.size() == cells) {
          const complexd value = complexd{0.0, -workspace.storedSign[f]} *
                                 traceOfProduct(matrix, toMatrix(mixed, order));
          at(e, edges + f) += value;
          at(edges + f, e) += value;
        }
      }
    }
  }

  // sum_j xi_j tr(d_y X_j d_x h), the variation of the power sums' matrices
  // through h at fixed projector, one column y at a time.
  bool constrained = false;
  for (const auto &constraint : declaration_.momentConstraints)
    if (constraint.multiplier != complexd{0.0, 0.0} &&
        constraint.form == SpectralConstraintForm::PowerSum)
      constrained = true;
  if (constrained) {
    for (std::size_t y = 0; y < size; ++y) {
      const std::size_t f = y < edges ? y : y - edges;
      if (!carried[f]) continue;
      const Eigen::MatrixXcd &dh =
          y < edges ? lengthDerivative[f] : linkDerivative[f];
      Eigen::MatrixXcd variation = Eigen::MatrixXcd::Zero(
          static_cast<Eigen::Index>(order), static_cast<Eigen::Index>(order));
      for (const auto &constraint : declaration_.momentConstraints) {
        if (constraint.multiplier == complexd{0.0, 0.0} ||
            constraint.form != SpectralConstraintForm::PowerSum)
          continue;
        variation += constraint.multiplier *
                     constraintDerivativeVariation(declaration_,
                                                   workspace.carrier, order,
                                                   constraint, &dh, nullptr);
      }
      if (variation.isZero(0.0)) continue;
      for (std::size_t x = 0; x < size; ++x) {
        const std::size_t e = x < edges ? x : x - edges;
        if (!carried[e]) continue;
        at(x, y) += traceOfProduct(
            variation, x < edges ? lengthDerivative[e] : linkDerivative[e]);
      }
    }
  }
  return toFlat(hessian);
}

CarrierDerivatives JointAction::carrierDerivatives() const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  const std::size_t edges = workspace.edges.size();
  const std::size_t order = workspace.carrierOrder;
  const std::size_t cells = order * order;
  CarrierDerivatives out;
  out.lengths.assign(edges, std::vector<complexd>(cells, complexd{0.0, 0.0}));
  out.links.assign(edges, std::vector<complexd>(cells, complexd{0.0, 0.0}));
  if (order == 0) return out;
  for (std::size_t e = 0; e < edges; ++e) {
    const auto *edge = workspace.edges[e];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const std::uint64_t source = edge->getSource()->getId();
    const std::uint64_t target = edge->getTarget()->getId();
    auto dz = workspace.hodge.laplacianGradient(declaration_.carrierDegree,
                                                source, target);
    if (dz.size() == cells) out.lengths[e] = std::move(dz);
    auto dphi = workspace.hodge.laplacianPhaseGradient(
        declaration_.carrierDegree, source, target);
    if (dphi.size() != cells) continue;
    // U d/dU = -i d/dphi on the canonical link, times the stored sign.
    const complexd factor{0.0, -workspace.storedSign[e]};
    for (complexd &entry : dphi) entry *= factor;
    out.links[e] = std::move(dphi);
  }
  return out;
}

std::vector<complexd> JointAction::stationarityStateVariation(
    const CarriedStateVariation &variation) const {
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/true);
  const std::size_t edges = workspace.edges.size();
  const std::size_t order = workspace.carrierOrder;
  std::vector<complexd> out(2 * edges, complexd{0.0, 0.0});
  if (order == 0) return out;
  const std::size_t cells = order * order;
  auto check = [&](const std::vector<complexd> &entry, const std::string &what) {
    if (!entry.empty() && entry.size() != cells)
      throw std::invalid_argument(
          "JointAction::stationarityStateVariation: the variation of " + what +
          " is a square matrix over the " + std::to_string(order) +
          " cells of degree " + std::to_string(declaration_.carrierDegree) +
          "; got " + std::to_string(entry.size()) + " entries");
  };
  check(variation.covariance, "the covariance");
  check(variation.momentProjector, "the fiber projector");
  if (!variation.bandProjectors.empty() &&
      variation.bandProjectors.size() !=
          declaration_.momentBandProjectors.size())
    throw std::invalid_argument(
        "JointAction::stationarityStateVariation: " +
        std::to_string(variation.bandProjectors.size()) +
        " band projector variations were given for " +
        std::to_string(declaration_.momentBandProjectors.size()) +
        " declared band projectors");
  for (const auto &entry : variation.bandProjectors)
    check(entry, "a band projector");

  // delta A = w_M delta Gamma + sum_j xi_j delta X_j, the latter through the
  // projectors alone.
  Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(order), static_cast<Eigen::Index>(order));
  if (declaration_.matterWeight != 0.0 && !declaration_.covariance.empty() &&
      !variation.covariance.empty())
    matrix += declaration_.matterWeight * toMatrix(variation.covariance, order);
  for (const auto &constraint : declaration_.momentConstraints) {
    if (constraint.multiplier == complexd{0.0, 0.0}) continue;
    const std::vector<complexd> *flat = nullptr;
    if (constraint.form == SpectralConstraintForm::BandMean) {
      if (variation.bandProjectors.empty()) continue;
      flat = &variation.bandProjectors.at(constraint.band);
    } else {
      if (declaration_.momentProjector.empty()) continue;
      flat = &variation.momentProjector;
    }
    if (flat->empty()) continue;
    const Eigen::MatrixXcd projectorVariation = toMatrix(*flat, order);
    matrix += constraint.multiplier *
              constraintDerivativeVariation(declaration_, workspace.carrier,
                                            order, constraint, nullptr,
                                            &projectorVariation);
  }
  std::vector<complexd> lengths(edges, complexd{0.0, 0.0});
  std::vector<complexd> links(edges, complexd{0.0, 0.0});
  contractionSweep(workspace, declaration_.carrierDegree, matrix, &lengths,
                   &links);
  std::copy(lengths.begin(), lengths.end(), out.begin());
  std::copy(links.begin(), links.end(),
            out.begin() + static_cast<std::ptrdiff_t>(edges));
  return out;
}

HolonomyTruncation JointAction::holonomyTruncation() const {
  HolonomyTruncation report;
  report.order = declaration_.villainOrder;
  const FacePotential potential(declaration_);
  if (!potential.villain()) return report;
  const VillainCharacter &character = *potential.villain();
  const ActionWorkspace workspace(spacetime_, declaration_.carrierDegree,
                                  declaration_.metricSource,
                                  /*wantCarrier=*/false);
  // std::max keeps its first argument when the comparison with the second is
  // false, so a face whose ratio is not a number (a sum of moduli that is
  // zero in double precision) leaves the maximum as it is.
  for (const complexd &holonomy : workspace.holonomies) {
    const VillainSeries point = character.series(holonomy);
    if (!VillainCharacter::certifiedNonzero(point)) ++report.uncertifiedFaces;
    const double margin = std::abs(point.value) / point.roundingBound;
    if (margin < report.smallestCertificateMargin)
      report.smallestCertificateMargin = margin;
    report.relativeValueTail = std::max(report.relativeValueTail,
                                        point.valueTail / point.magnitude);
    report.relativeFirstTail = std::max(
        report.relativeFirstTail, point.firstTail / point.firstMagnitude);
    report.relativeSecondTail = std::max(
        report.relativeSecondTail, point.secondTail / point.secondMagnitude);
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

  const Eigen::MatrixXcd power = constraintDerivative(
      declaration_, workspace.carrier, workspace.carrierOrder, constraint);

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
