// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/MonopoleSpin.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace tessera::observables {
namespace {

using ::tessera::cobordism::Certificate;
using ::tessera::cobordism::CertificateDomain;
using ::tessera::cobordism::CertificateRegime;

using cd = std::complex<double>;

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kTwoPi = 2.0 * kPi;

void requirePositive(double tolerance, const char* what) {
  if (!(tolerance > 0.0)) {
    std::ostringstream message;
    message << what << " must be positive; received " << tolerance << ".";
    throw std::invalid_argument(message.str());
  }
}

/// The composition (g h)(x) = g(h(x)) of two vertex permutations.
std::vector<std::size_t> composePermutations(
    const std::vector<std::size_t>& g, const std::vector<std::size_t>& h) {
  std::vector<std::size_t> out(h.size());
  for (std::size_t x = 0; x < h.size(); ++x) out[x] = g[h[x]];
  return out;
}

/// Whether two permutations are equal.
bool samePermutation(const std::vector<std::size_t>& g,
                     const std::vector<std::size_t>& h) {
  return g.size() == h.size() && std::equal(g.begin(), g.end(), h.begin());
}

/// The largest entry modulus of a matrix, zero for an empty one.
double maxAbs(const Eigen::MatrixXcd& m) {
  if (m.size() == 0) return 0.0;
  return m.cwiseAbs().maxCoeff();
}

}  // namespace

// ─── MonopoleSupport construction ──────────────────────────────────────────

MonopoleSupport::MonopoleSupport(std::size_t vertexCount,
                                 std::vector<std::array<std::size_t, 2>> edges,
                                 std::vector<std::array<std::size_t, 3>> faces,
                                 std::vector<Complex> connection,
                                 double unitModulusTolerance)
    : vertexCount_(vertexCount),
      edges_(std::move(edges)),
      faces_(std::move(faces)),
      connection_(std::move(connection)) {
  requirePositive(unitModulusTolerance,
                  "MonopoleSupport unit-modulus tolerance");
  if (vertexCount_ == 0) {
    throw std::invalid_argument(
        "MonopoleSupport: a support needs at least one vertex.");
  }
  if (connection_.size() != edges_.size()) {
    std::ostringstream message;
    message << "MonopoleSupport: " << edges_.size() << " edges were declared "
            << "but " << connection_.size()
            << " connection values were supplied; one value per edge is "
               "required.";
    throw std::invalid_argument(message.str());
  }
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const std::array<std::size_t, 2>& edge = edges_[e];
    if (edge[0] >= vertexCount_ || edge[1] >= vertexCount_) {
      std::ostringstream message;
      message << "MonopoleSupport: edge " << e << " joins vertices " << edge[0]
              << " and " << edge[1] << "; the support has " << vertexCount_
              << " vertices.";
      throw std::invalid_argument(message.str());
    }
    if (edge[0] >= edge[1]) {
      std::ostringstream message;
      message << "MonopoleSupport: edge " << e << " is stored as (" << edge[0]
              << ", " << edge[1]
              << "); every edge is stored with its smaller vertex first and "
                 "no edge joins a vertex to itself.";
      throw std::invalid_argument(message.str());
    }
    for (std::size_t f = 0; f < e; ++f) {
      if (edges_[f] == edge) {
        std::ostringstream message;
        message << "MonopoleSupport: edges " << f << " and " << e
                << " are the same pair (" << edge[0] << ", " << edge[1]
                << "); an edge is declared once.";
        throw std::invalid_argument(message.str());
      }
    }
    if (connection_[e] == cd(0.0, 0.0)) {
      std::ostringstream message;
      message << "MonopoleSupport: the connection value on edge " << e
              << " is zero, so it has no inverse and no holonomy.";
      throw std::invalid_argument(message.str());
    }
    // The monopole number is a charge of the U(1) part of the connection, and
    // every construction below — the unitarity of D_k(g), the rotation
    // average, the self-adjoint band decomposition — rests on that part alone.
    // A general GL(1, C) connection enters through the explicit `u1Part`
    // conversion, which is the caller's declared choice; it is never applied
    // silently here.
    const double modulusDefect = std::abs(std::abs(connection_[e]) - 1.0);
    if (modulusDefect > unitModulusTolerance) {
      std::ostringstream message;
      message << "MonopoleSupport: the connection value on edge " << e
              << " has modulus " << std::abs(connection_[e])
              << "; this kernel reads the U(1) part of the connection, so "
                 "every value must have unit modulus. Convert an unrestricted "
                 "connection with MonopoleSupport::u1Part first.";
      throw std::invalid_argument(message.str());
    }
  }
  for (std::size_t f = 0; f < faces_.size(); ++f) {
    const std::array<std::size_t, 3>& face = faces_[f];
    if (face[0] == face[1] || face[1] == face[2] || face[0] == face[2]) {
      std::ostringstream message;
      message << "MonopoleSupport: face " << f << " repeats a vertex.";
      throw std::invalid_argument(message.str());
    }
    for (std::size_t i = 0; i < 3; ++i) {
      const std::size_t x = face[i];
      const std::size_t y = face[(i + 1) % 3];
      if (x >= vertexCount_ || y >= vertexCount_) {
        std::ostringstream message;
        message << "MonopoleSupport: face " << f << " names vertex "
                << std::max(x, y) << "; the support has " << vertexCount_
                << " vertices.";
        throw std::invalid_argument(message.str());
      }
      if (edgeIndex(x, y) == edges_.size()) {
        std::ostringstream message;
        message << "MonopoleSupport: face " << f << " has boundary pair ("
                << x << ", " << y << "), which is not a declared edge.";
        throw std::invalid_argument(message.str());
      }
    }
  }
}

std::size_t MonopoleSupport::edgeIndex(std::size_t x, std::size_t y) const {
  const std::array<std::size_t, 2> key{std::min(x, y), std::max(x, y)};
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    if (edges_[e] == key) return e;
  }
  return edges_.size();
}

MonopoleSupport::Complex MonopoleSupport::transport(std::size_t x,
                                                    std::size_t y) const {
  const std::size_t e = edgeIndex(x, y);
  if (e == edges_.size()) {
    std::ostringstream message;
    message << "MonopoleSupport::transport: no declared edge joins vertices "
            << x << " and " << y << ".";
    throw std::invalid_argument(message.str());
  }
  return x < y ? connection_[e] : cd(1.0, 0.0) / connection_[e];
}

MonopoleSupport MonopoleSupport::tetrahedron(int monopoleNumber) {
  const double angle = kTwoPi * static_cast<double>(monopoleNumber) / 4.0;
  const std::vector<std::array<std::size_t, 2>> edges{
      {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  // Outward orientation of the four faces of the tetrahedron 0 1 2 3.
  const std::vector<std::array<std::size_t, 3>> faces{
      {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
  // Gauge-fixed on the spanning tree from vertex 0: U_{0j} = 1. The three
  // remaining values are read off the three faces containing vertex 0:
  // face (0,2,1) has holonomy 1/U_{12}, face (0,1,3) has U_{13}, and face
  // (0,3,2) has 1/U_{23}. Setting each equal to exp(i angle) gives the
  // values below, and the fourth face (1,2,3) then carries
  // U_{12} U_{23} / U_{13} = exp(-3 i angle) = exp(i angle) modulo 2 pi
  // whenever the four holonomies are equal, which is the closure identity of
  // a closed cut.
  const std::vector<cd> connection{cd(1.0, 0.0),
                                   cd(1.0, 0.0),
                                   cd(1.0, 0.0),
                                   std::polar(1.0, -angle),
                                   std::polar(1.0, angle),
                                   std::polar(1.0, -angle)};
  return MonopoleSupport(4, edges, faces, connection);
}

std::vector<MonopoleSupport::Permutation>
MonopoleSupport::tetrahedralRotations() {
  std::vector<Permutation> out;
  std::vector<std::size_t> perm{0, 1, 2, 3};
  do {
    int inversions = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = i + 1; j < 4; ++j) {
        if (perm[i] > perm[j]) ++inversions;
      }
    }
    if (inversions % 2 == 0) out.push_back(perm);
  } while (std::next_permutation(perm.begin(), perm.end()));
  // Put the identity first so a caller can rely on group element zero.
  const Permutation identity{0, 1, 2, 3};
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (samePermutation(out[i], identity)) {
      std::swap(out[0], out[i]);
      break;
    }
  }
  return out;
}

std::vector<MonopoleSupport::Complex> MonopoleSupport::u1Part(
    const std::vector<Complex>& connection) {
  std::vector<cd> out;
  out.reserve(connection.size());
  for (std::size_t e = 0; e < connection.size(); ++e) {
    const double modulus = std::abs(connection[e]);
    if (modulus == 0.0) {
      std::ostringstream message;
      message << "MonopoleSupport::u1Part: the connection value on edge " << e
              << " is zero and has no U(1) part.";
      throw std::invalid_argument(message.str());
    }
    out.push_back(connection[e] / modulus);
  }
  return out;
}

// ─── the monopole number ───────────────────────────────────────────────────

MonopoleNumberRead MonopoleSupport::monopoleNumber(double tolerance) const {
  requirePositive(tolerance, "MonopoleSupport::monopoleNumber tolerance");
  MonopoleNumberRead read;
  for (const cd& value : connection_) {
    read.unitModulusResidual =
        std::max(read.unitModulusResidual, std::abs(std::abs(value) - 1.0));
  }
  read.faceHolonomies.reserve(faces_.size());
  read.faceFluxes.reserve(faces_.size());
  read.branchMargin = kPi;
  for (const std::array<std::size_t, 3>& face : faces_) {
    const cd holonomy = transport(face[0], face[1]) *
                        transport(face[1], face[2]) *
                        transport(face[2], face[0]);
    read.faceHolonomies.push_back(holonomy);
    double flux = std::arg(holonomy);
    // The principal argument is taken on the half-open interval (-pi, pi], so
    // a holonomy on the negative real axis carries +pi. std::arg returns -pi
    // for such a value when its imaginary part is the negative zero, which is
    // a signed-zero artifact of the floating-point product and not the
    // mathematical principal value; the two are reconciled here so that the
    // read does not depend on how the product happened to round.
    if (std::abs(kPi + flux) <= tolerance) flux = kPi;
    read.faceFluxes.push_back(flux);
    read.totalFlux += flux;
    read.branchMargin = std::min(read.branchMargin, kPi - std::abs(flux));
  }
  read.onBranchCut = read.branchMargin <= tolerance;
  const double turns = read.totalFlux / kTwoPi;
  read.monopoleNumber = static_cast<int>(std::lround(turns));
  read.integralityResidual =
      std::abs(turns - static_cast<double>(read.monopoleNumber));
  const double residual =
      std::max(read.integralityResidual, read.unitModulusResidual);
  read.bundle = residual <= tolerance;
  read.odd = (read.monopoleNumber % 2) != 0;
  read.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, CertificateRegime::NonNormal, residual,
      tolerance);
  return read;
}

// ─── the twisted operators ─────────────────────────────────────────────────

Eigen::MatrixXcd MonopoleSupport::twistedCoboundary() const {
  Eigen::MatrixXcd d0 =
      Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(edges_.size()),
                             static_cast<Eigen::Index>(vertexCount_));
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const auto row = static_cast<Eigen::Index>(e);
    d0(row, static_cast<Eigen::Index>(edges_[e][0])) = cd(-1.0, 0.0);
    d0(row, static_cast<Eigen::Index>(edges_[e][1])) = connection_[e];
  }
  return d0;
}

Eigen::MatrixXcd MonopoleSupport::twistedFaceCoboundary() const {
  Eigen::MatrixXcd d1 =
      Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(faces_.size()),
                             static_cast<Eigen::Index>(edges_.size()));
  for (std::size_t f = 0; f < faces_.size(); ++f) {
    std::array<std::size_t, 3> sorted = faces_[f];
    std::sort(sorted.begin(), sorted.end());
    const std::size_t a = sorted[0];
    const std::size_t b = sorted[1];
    const std::size_t c = sorted[2];
    const auto row = static_cast<Eigen::Index>(f);
    d1(row, static_cast<Eigen::Index>(edgeIndex(b, c))) += transport(a, b);
    d1(row, static_cast<Eigen::Index>(edgeIndex(a, c))) += cd(-1.0, 0.0);
    d1(row, static_cast<Eigen::Index>(edgeIndex(a, b))) += cd(1.0, 0.0);
  }
  return d1;
}

Eigen::MatrixXcd MonopoleSupport::vertexLaplacian() const {
  const Eigen::MatrixXcd d0 = twistedCoboundary();
  return d0.adjoint() * d0;
}

Eigen::MatrixXcd MonopoleSupport::edgeLaplacian() const {
  const Eigen::MatrixXcd d0 = twistedCoboundary();
  const Eigen::MatrixXcd d1 = twistedFaceCoboundary();
  return d0 * d0.adjoint() + d1.adjoint() * d1;
}

Eigen::MatrixXcd MonopoleSupport::coexactProjector(double tolerance) const {
  requirePositive(tolerance, "MonopoleSupport::coexactProjector tolerance");
  const Eigen::MatrixXcd d0 = twistedCoboundary();
  const auto edgeCount = static_cast<Eigen::Index>(edges_.size());
  if (edgeCount == 0) return Eigen::MatrixXcd::Zero(0, 0);
  // ker((delta_0^U)^dagger) is the orthogonal complement of the column space
  // of delta_0^U, so the projector is I - P_image.
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(d0, Eigen::ComputeThinU);
  const Eigen::VectorXd singular = svd.singularValues();
  Eigen::MatrixXcd image =
      Eigen::MatrixXcd::Zero(edgeCount, edgeCount);
  // The rank of delta_0^U: its singular values above the tolerance times
  // the largest.
  const double largest = singular.size() > 0 ? singular(0) : 0.0;
  for (Eigen::Index i = 0; i < singular.size(); ++i) {
    if (singular(i) <= tolerance * largest) continue;
    image += svd.matrixU().col(i) * svd.matrixU().col(i).adjoint();
  }
  return Eigen::MatrixXcd::Identity(edgeCount, edgeCount) - image;
}

// ─── the projective representation ─────────────────────────────────────────

void MonopoleSupport::validateRotation(const Permutation& rotation) const {
  if (rotation.size() != vertexCount_) {
    std::ostringstream message;
    message << "MonopoleSupport: a rotation is a permutation of the "
            << vertexCount_ << " vertices; received " << rotation.size()
            << " images.";
    throw std::invalid_argument(message.str());
  }
  std::vector<char> seen(vertexCount_, 0);
  for (std::size_t x = 0; x < vertexCount_; ++x) {
    if (rotation[x] >= vertexCount_ || seen[rotation[x]] != 0) {
      std::ostringstream message;
      message << "MonopoleSupport: the supplied vertex map is not a "
                 "permutation; vertex "
              << x << " maps to " << rotation[x] << ".";
      throw std::invalid_argument(message.str());
    }
    seen[rotation[x]] = 1;
  }
  for (const std::array<std::size_t, 2>& edge : edges_) {
    if (edgeIndex(rotation[edge[0]], rotation[edge[1]]) == edges_.size()) {
      std::ostringstream message;
      message << "MonopoleSupport: the supplied rotation carries edge ("
              << edge[0] << ", " << edge[1] << ") to ("
              << rotation[edge[0]] << ", " << rotation[edge[1]]
              << "), which is not a declared edge, so it is not a symmetry of "
                 "this support.";
      throw std::invalid_argument(message.str());
    }
  }
}

GaugeCompensationRead MonopoleSupport::gaugeCompensation(
    const Permutation& rotation, double tolerance) const {
  requirePositive(tolerance, "MonopoleSupport::gaugeCompensation tolerance");
  validateRotation(rotation);

  // (g U)_{gx, gy} = U_{xy}, so (g U)_{xy} = U_{g^{-1}x, g^{-1}y}.
  std::vector<std::size_t> inverse(vertexCount_, 0);
  for (std::size_t x = 0; x < vertexCount_; ++x) inverse[rotation[x]] = x;
  const auto pushed = [&](std::size_t x, std::size_t y) {
    return transport(inverse[x], inverse[y]);
  };

  GaugeCompensationRead read;
  read.gauge = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(vertexCount_));
  std::vector<char> solved(vertexCount_, 0);
  // Breadth-first spanning tree from the lowest vertex, with u_g(root) = 1.
  std::vector<std::size_t> frontier{0};
  read.gauge(0) = cd(1.0, 0.0);
  solved[0] = 1;
  while (!frontier.empty()) {
    std::vector<std::size_t> next;
    for (const std::size_t x : frontier) {
      for (const std::array<std::size_t, 2>& edge : edges_) {
        std::size_t y = vertexCount_;
        if (edge[0] == x) y = edge[1];
        else if (edge[1] == x) y = edge[0];
        if (y == vertexCount_ || solved[y] != 0) continue;
        // (g U)_{xy} = u(x) U_{xy} u(y)^{-1}  =>  u(y) = u(x) U_{xy} / (gU)_{xy}
        read.gauge(static_cast<Eigen::Index>(y)) =
            read.gauge(static_cast<Eigen::Index>(x)) * transport(x, y) /
            pushed(x, y);
        solved[y] = 1;
        next.push_back(y);
      }
    }
    frontier = std::move(next);
  }
  for (std::size_t x = 0; x < vertexCount_; ++x) {
    if (solved[x] == 0) {
      std::ostringstream message;
      message << "MonopoleSupport::gaugeCompensation: vertex " << x
              << " is not reachable from vertex 0, so no spanning tree covers "
                 "the support and the gauge compensation is not determined.";
      throw std::invalid_argument(message.str());
    }
  }
  for (const std::array<std::size_t, 2>& edge : edges_) {
    const std::size_t x = edge[0];
    const std::size_t y = edge[1];
    const cd expected = read.gauge(static_cast<Eigen::Index>(x)) *
                        transport(x, y) /
                        read.gauge(static_cast<Eigen::Index>(y));
    read.residual = std::max(read.residual, std::abs(pushed(x, y) - expected));
  }
  read.symmetric = read.residual <= tolerance;
  return read;
}

Eigen::MatrixXcd MonopoleSupport::vertexRepresentation(
    const Permutation& rotation) const {
  const GaugeCompensationRead gauge = gaugeCompensation(rotation);
  const auto n = static_cast<Eigen::Index>(vertexCount_);
  Eigen::MatrixXcd d = Eigen::MatrixXcd::Zero(n, n);
  for (std::size_t x = 0; x < vertexCount_; ++x) {
    const std::size_t gx = rotation[x];
    d(static_cast<Eigen::Index>(gx), static_cast<Eigen::Index>(x)) =
        cd(1.0, 0.0) / gauge.gauge(static_cast<Eigen::Index>(gx));
  }
  return d;
}

Eigen::MatrixXcd MonopoleSupport::edgeRepresentation(
    const Permutation& rotation) const {
  const GaugeCompensationRead gauge = gaugeCompensation(rotation);
  const auto m = static_cast<Eigen::Index>(edges_.size());
  Eigen::MatrixXcd d = Eigen::MatrixXcd::Zero(m, m);
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const std::size_t x = edges_[e][0];
    const std::size_t y = edges_[e][1];
    const std::size_t gx = rotation[x];
    const std::size_t gy = rotation[y];
    const cd base = cd(1.0, 0.0) / gauge.gauge(static_cast<Eigen::Index>(gx));
    const auto target = static_cast<Eigen::Index>(edgeIndex(gx, gy));
    if (gx < gy) {
      d(target, static_cast<Eigen::Index>(e)) = base;
    } else {
      d(target, static_cast<Eigen::Index>(e)) = -transport(gy, gx) * base;
    }
  }
  return d;
}

double MonopoleSupport::intertwiningResidual(
    const Permutation& rotation) const {
  const Eigen::MatrixXcd d0 = twistedCoboundary();
  const Eigen::MatrixXcd vertexAction = vertexRepresentation(rotation);
  const Eigen::MatrixXcd edgeAction = edgeRepresentation(rotation);
  return maxAbs(d0 * vertexAction - edgeAction * d0);
}

CocycleRead MonopoleSupport::cocycle(const std::vector<Permutation>& group,
                                     int cochainDegree,
                                     double tolerance) const {
  requirePositive(tolerance, "MonopoleSupport::cocycle tolerance");
  if (group.empty()) {
    throw std::invalid_argument(
        "MonopoleSupport::cocycle: the rotation group must contain at least "
        "the identity.");
  }
  if (cochainDegree != 0 && cochainDegree != 1) {
    std::ostringstream message;
    message << "MonopoleSupport::cocycle: the cochain degree must be 0 "
               "(vertices) or 1 (edges); received "
            << cochainDegree << ".";
    throw std::invalid_argument(message.str());
  }
  const std::size_t order = group.size();
  std::vector<Eigen::MatrixXcd> actions;
  actions.reserve(order);
  for (const Permutation& g : group) {
    actions.push_back(cochainDegree == 0 ? vertexRepresentation(g)
                                         : edgeRepresentation(g));
  }
  // The index of each product gh inside the supplied group.
  std::vector<std::vector<std::size_t>> productIndex(
      order, std::vector<std::size_t>(order, order));
  for (std::size_t i = 0; i < order; ++i) {
    for (std::size_t j = 0; j < order; ++j) {
      const Permutation product = composePermutations(group[i], group[j]);
      for (std::size_t k = 0; k < order; ++k) {
        if (samePermutation(group[k], product)) {
          productIndex[i][j] = k;
          break;
        }
      }
      if (productIndex[i][j] == order) {
        std::ostringstream message;
        message << "MonopoleSupport::cocycle: the product of group elements "
                << i << " and " << j
                << " is not in the supplied set, so it is not a group and "
                   "varpi(g, h) has no D(gh) to be measured against.";
        throw std::invalid_argument(message.str());
      }
    }
  }

  CocycleRead read;
  read.groupOrder = order;
  Eigen::MatrixXcd phases = Eigen::MatrixXcd::Zero(
      static_cast<Eigen::Index>(order), static_cast<Eigen::Index>(order));
  for (std::size_t i = 0; i < order; ++i) {
    for (std::size_t j = 0; j < order; ++j) {
      const Eigen::MatrixXcd product = actions[i] * actions[j];
      const Eigen::MatrixXcd target = actions[productIndex[i][j]];
      // D(g) D(h) = varpi D(gh), so varpi is the ratio of the two matrices at
      // any entry where D(gh) does not vanish. The entry of largest modulus
      // is taken, which is the best conditioned one; the residual below then
      // checks that the SAME ratio reproduces every other entry, so a set of
      // permutations that does not act projectively is caught rather than
      // averaged over.
      Eigen::Index pivotRow = 0;
      Eigen::Index pivotColumn = 0;
      target.cwiseAbs().maxCoeff(&pivotRow, &pivotColumn);
      const cd phase =
          product(pivotRow, pivotColumn) / target(pivotRow, pivotColumn);
      phases(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          phase;
      read.scalarResidual = std::max(
          read.scalarResidual,
          maxAbs(product - phase * target));
      bool known = false;
      for (const cd& value : read.values) {
        if (std::abs(value - phase) <= tolerance) {
          known = true;
          break;
        }
      }
      if (!known) read.values.push_back(phase);
    }
  }
  std::sort(read.values.begin(), read.values.end(),
            [](const cd& a, const cd& b) {
              if (a.real() != b.real()) return a.real() < b.real();
              return a.imag() < b.imag();
            });

  for (std::size_t i = 0; i < order; ++i) {
    for (std::size_t j = 0; j < order; ++j) {
      if (productIndex[i][j] != productIndex[j][i]) continue;  // not commuting
      const cd forward =
          phases(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
      const cd backward =
          phases(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i));
      if (std::abs(backward) == 0.0) continue;
      const cd commutator = forward / backward;
      const double deviation = std::abs(commutator - cd(1.0, 0.0));
      if (deviation > read.maxCommutatorDeviation) {
        read.maxCommutatorDeviation = deviation;
        read.commutatorPhase = commutator;
        read.commutatorPair = {i, j};
      }
    }
  }
  read.nontrivial = read.maxCommutatorDeviation > tolerance;
  read.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, CertificateRegime::NonNormal,
      read.scalarResidual, tolerance);
  return read;
}

Eigen::MatrixXcd MonopoleSupport::rotationAveragedEdgeOperator(
    const Eigen::MatrixXcd& edgeOperator,
    const std::vector<Permutation>& group) const {
  if (group.empty()) {
    throw std::invalid_argument(
        "MonopoleSupport::rotationAveragedEdgeOperator: the rotation group "
        "must contain at least the identity.");
  }
  const auto m = static_cast<Eigen::Index>(edges_.size());
  if (edgeOperator.rows() != m || edgeOperator.cols() != m) {
    std::ostringstream message;
    message << "MonopoleSupport::rotationAveragedEdgeOperator: the operator "
               "is "
            << edgeOperator.rows() << "x" << edgeOperator.cols()
            << "; the support has " << edges_.size() << " edges.";
    throw std::invalid_argument(message.str());
  }
  Eigen::MatrixXcd averaged = Eigen::MatrixXcd::Zero(m, m);
  for (const Permutation& g : group) {
    const Eigen::MatrixXcd action = edgeRepresentation(g);
    averaged += action * edgeOperator * action.adjoint();
  }
  return averaged / static_cast<double>(group.size());
}

std::vector<SpinorBandRead> MonopoleSupport::spinorBands(
    const Eigen::MatrixXcd& operatorMatrix,
    const std::vector<Permutation>& group, bool nontrivialClass,
    double degeneracyTolerance, double tolerance) const {
  requirePositive(degeneracyTolerance,
                  "MonopoleSupport::spinorBands degeneracyTolerance");
  requirePositive(tolerance, "MonopoleSupport::spinorBands tolerance");
  if (group.empty()) {
    throw std::invalid_argument(
        "MonopoleSupport::spinorBands: the rotation group must contain at "
        "least the identity.");
  }
  const auto m = static_cast<Eigen::Index>(edges_.size());
  if (operatorMatrix.rows() != m || operatorMatrix.cols() != m) {
    std::ostringstream message;
    message << "MonopoleSupport::spinorBands: the operator is "
            << operatorMatrix.rows() << "x" << operatorMatrix.cols()
            << "; the support has " << edges_.size() << " edges.";
    throw std::invalid_argument(message.str());
  }
  // The departure of the operator from its adjoint relative to its own size.
  // The bands are read on the Hermitian part, and every band carries the
  // departure and whether it is within the tolerance.
  const double size = maxAbs(operatorMatrix);
  const double defect = maxAbs(operatorMatrix - operatorMatrix.adjoint());
  const double hermiticity = size > 0.0 ? defect / size : 0.0;
  const Eigen::MatrixXcd hermitianPart =
      0.5 * (operatorMatrix + operatorMatrix.adjoint());

  std::vector<Eigen::MatrixXcd> actions;
  actions.reserve(group.size());
  for (const Permutation& g : group) actions.push_back(edgeRepresentation(g));
  const Eigen::MatrixXcd coexact = coexactProjector(tolerance);

  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hermitianPart);
  const Eigen::VectorXd values = solver.eigenvalues();
  const Eigen::MatrixXcd vectors = solver.eigenvectors();

  // Under a nontrivial projective class every invariant subspace has even
  // rank, so the ascending eigenvalues of a rotation-invariant operator on an
  // even number of edges are equal in consecutive pairs exactly: a pair is
  // one unit whatever the eigensolver's rounding separates its two values
  // by. A band is a run of units each of which starts within the degeneracy
  // tolerance, taken relative to the operator's scale, of the band's first
  // eigenvalue.
  const Eigen::Index unit =
      nontrivialClass && values.size() % 2 == 0 ? 2 : 1;
  const double scale =
      values.size() > 0
          ? std::max(std::abs(values(0)), std::abs(values(values.size() - 1)))
          : 0.0;
  const double width0 = degeneracyTolerance * scale;

  std::vector<SpinorBandRead> bands;
  Eigen::Index start = 0;
  while (start < values.size()) {
    Eigen::Index stop = std::min(start + unit, values.size());
    while (stop < values.size() &&
           std::abs(values(stop) - values(start)) <= width0) {
      stop = std::min(stop + unit, values.size());
    }
    const Eigen::Index width = stop - start;
    const Eigen::MatrixXcd basis = vectors.block(0, start, m, width);
    const Eigen::MatrixXcd projector = basis * basis.adjoint();

    SpinorBandRead band;
    band.eigenvalue = values.segment(start, width).mean();
    band.dimension = static_cast<std::size_t>(width);
    double characterSum = 0.0;
    for (const Eigen::MatrixXcd& action : actions) {
      band.invarianceResidual =
          std::max(band.invarianceResidual,
                   maxAbs(action * projector - projector * action));
      const cd character = (basis.adjoint() * action * basis).trace();
      characterSum += std::norm(character);
    }
    band.irreducibilityScore =
        characterSum / static_cast<double>(actions.size());
    band.coexactResidual = maxAbs(coexact * basis - basis);
    band.coexact = band.coexactResidual <= tolerance;
    band.hermiticityDefect = hermiticity;
    band.hermitian = hermiticity <= tolerance;
    band.spinorDoublet = nontrivialClass && width == 2 &&
                         band.invarianceResidual <= tolerance &&
                         std::abs(band.irreducibilityScore - 1.0) <= tolerance;
    bands.push_back(band);
    start = stop;
  }
  return bands;
}

MonopoleSpinRead MonopoleSupport::spinRead(
    const std::vector<Permutation>& group, double degeneracyTolerance,
    double tolerance) const {
  MonopoleSpinRead read;
  read.monopole = monopoleNumber(tolerance);
  read.cocycle = cocycle(group, /*cochainDegree=*/1, tolerance);
  const Eigen::MatrixXcd averaged =
      rotationAveragedEdgeOperator(edgeLaplacian(), group);
  read.bands = spinorBands(averaged, group, read.cocycle.nontrivial,
                           degeneracyTolerance, tolerance);
  read.doubletIndex = read.bands.size();
  for (std::size_t i = 0; i < read.bands.size(); ++i) {
    if (read.bands[i].spinorDoublet && read.bands[i].coexact) {
      read.doubletIndex = i;
      read.halfIntegerDoublet = true;
      break;
    }
  }
  const double residual = std::max(read.monopole.certificate.residual(),
                                   read.cocycle.certificate.residual());
  const bool holds =
      read.monopole.certificate.holds() && read.cocycle.certificate.holds();
  read.certificate =
      holds ? Certificate::algebraicallyExact(CertificateDomain::Static,
                                              CertificateRegime::NonNormal,
                                              residual, tolerance)
            : Certificate::heuristicDiscovery(CertificateDomain::Static,
                                              CertificateRegime::NonNormal);
  return read;
}

// ─── SharpSpin ─────────────────────────────────────────────────────────────

namespace {

/// The mode count M of a 2^M-dimensional Fock space, or zero when the
/// dimension is not a power of two.
std::size_t modeCountOf(Eigen::Index dimension) {
  if (dimension <= 0) return 0;
  auto d = static_cast<std::size_t>(dimension);
  std::size_t modes = 0;
  while ((d & 1u) == 0u) {
    d >>= 1u;
    ++modes;
  }
  return d == 1u ? modes : 0;
}

/// Apply dGamma(L) = sum_ij L(i,j) a_i^dagger a_j to a Fock vector without
/// materializing the operator. The basis is n(b) = sum_i b_i 2^i, so mode i
/// is bit i, and the Jordan-Wigner sign of a mode is the parity of the
/// occupied modes below it.
Eigen::VectorXcd applyDGamma(const Eigen::MatrixXcd& oneParticle,
                             const Eigen::VectorXcd& state,
                             std::size_t modeCount) {
  Eigen::VectorXcd out = Eigen::VectorXcd::Zero(state.size());
  // The Jordan-Wigner sign of a mode is the parity of the occupied modes
  // below it, the same convention ExteriorAlgebra::creationMatrix documents.
  const auto prefixParity = [](std::size_t occupation, std::size_t mode) {
    const auto below = static_cast<unsigned long long>(
        occupation & ((std::size_t{1} << mode) - 1));
    return (std::popcount(below) % 2) != 0;
  };
  for (Eigen::Index n = 0; n < state.size(); ++n) {
    const cd amplitude = state(n);
    if (amplitude == cd(0.0, 0.0)) continue;
    const auto occupation = static_cast<std::size_t>(n);
    for (std::size_t j = 0; j < modeCount; ++j) {
      if ((occupation >> j & 1u) == 0u) continue;
      const bool signJ = prefixParity(occupation, j);
      const std::size_t removed = occupation & ~(std::size_t{1} << j);
      for (std::size_t i = 0; i < modeCount; ++i) {
        const cd coefficient =
            oneParticle(static_cast<Eigen::Index>(i),
                        static_cast<Eigen::Index>(j));
        if (coefficient == cd(0.0, 0.0)) continue;
        if ((removed >> i & 1u) != 0u) continue;
        const bool signI = prefixParity(removed, i);
        const std::size_t added = removed | (std::size_t{1} << i);
        const double sign = (signI == signJ) ? 1.0 : -1.0;
        out(static_cast<Eigen::Index>(added)) +=
            sign * coefficient * amplitude;
      }
    }
  }
  return out;
}

/// Name a declared limit that was reached.
[[noreturn]] void declaredLimitReached(const char* where, const char* what,
                                       std::size_t limit, double asked) {
  std::ostringstream message;
  message << where << ": the declared " << what << " limit of " << limit
          << " was reached: " << asked << " were asked for.";
  throw std::length_error(message.str());
}

/// A mode count M whose 2^M-entry Fock vector has an index (`indexable` is
/// the largest such M for the object being built), within the caller's
/// declared limit when there is one.
void requireFockModes(const char* where, std::size_t modeCount,
                      std::size_t indexable,
                      const std::optional<std::size_t>& modeLimit) {
  if (modeCount == 0 || modeCount > indexable) {
    std::ostringstream message;
    message << where << ": the mode count must lie between one and "
            << indexable
            << ", the largest count whose entries the index type counts; "
               "received "
            << modeCount << ".";
    throw std::invalid_argument(message.str());
  }
  if (modeLimit.has_value() && modeCount > *modeLimit)
    declaredLimitReached(where, "mode", *modeLimit,
                         static_cast<double>(modeCount));
}

}  // namespace

Eigen::VectorXcd SharpSpin::determinant(
    const std::vector<std::size_t>& occupiedModes, std::size_t modeCount,
    std::optional<std::size_t> modeLimit) {
  requireFockModes("SharpSpin::determinant", modeCount, kIndexableModes,
                   modeLimit);
  std::vector<char> seen(modeCount, 0);
  for (const std::size_t mode : occupiedModes) {
    if (mode >= modeCount) {
      std::ostringstream message;
      message << "SharpSpin::determinant: mode " << mode
              << " is outside the " << modeCount << "-mode space.";
      throw std::invalid_argument(message.str());
    }
    if (seen[mode] != 0) {
      std::ostringstream message;
      message << "SharpSpin::determinant: mode " << mode
              << " is occupied twice; a determinant occupies each mode at "
                 "most once and a repeated mode wedges to exactly zero.";
      throw std::invalid_argument(message.str());
    }
    seen[mode] = 1;
  }
  // a_{m_1}^dagger ... a_{m_n}^dagger on the vacuum, the rightmost factor
  // acting first. Creating mode i on an occupation carries the Jordan-Wigner
  // sign of the occupied modes below i, the convention of
  // `ExteriorAlgebra::creationMatrix`, so the wedge of basis modes is one
  // Fock basis state with a sign.
  std::uint64_t occupation = 0;
  bool negative = false;
  for (auto mode = occupiedModes.rbegin(); mode != occupiedModes.rend();
       ++mode) {
    const std::uint64_t bit = std::uint64_t{1} << *mode;
    if (std::popcount(occupation & (bit - 1)) % 2 != 0) negative = !negative;
    occupation |= bit;
  }
  Eigen::VectorXcd state = Eigen::VectorXcd::Zero(
      static_cast<Eigen::Index>(std::uint64_t{1} << modeCount));
  state(static_cast<Eigen::Index>(occupation)) =
      cd(negative ? -1.0 : 1.0, 0.0);
  return state;
}

Eigen::VectorXcd SharpSpin::determinantSuperposition(
    const std::vector<std::vector<std::size_t>>& occupations,
    const std::vector<Complex>& amplitudes, std::size_t modeCount,
    std::optional<std::size_t> modeLimit) {
  if (occupations.empty()) {
    throw std::invalid_argument(
        "SharpSpin::determinantSuperposition: a bounded superposition needs "
        "at least one determinant.");
  }
  if (occupations.size() != amplitudes.size()) {
    std::ostringstream message;
    message << "SharpSpin::determinantSuperposition: " << occupations.size()
            << " determinants were listed but " << amplitudes.size()
            << " amplitudes were supplied.";
    throw std::invalid_argument(message.str());
  }
  Eigen::VectorXcd state;
  for (std::size_t d = 0; d < occupations.size(); ++d) {
    const Eigen::VectorXcd term =
        determinant(occupations[d], modeCount, modeLimit);
    if (state.size() == 0) state = Eigen::VectorXcd::Zero(term.size());
    state += amplitudes[d] * term;
  }
  return state;
}

Eigen::VectorXcd SharpSpin::applyTotalSpinSquared(
    const std::array<Eigen::MatrixXcd, 3>& spinMatrices,
    const Eigen::VectorXcd& state) {
  const std::size_t modeCount = modeCountOf(state.size());
  if (modeCount == 0) {
    std::ostringstream message;
    message << "SharpSpin::applyTotalSpinSquared: the state has dimension "
            << state.size()
            << ", which is not 2^M for any mode count M, so it is not a Fock "
               "vector.";
    throw std::invalid_argument(message.str());
  }
  for (std::size_t a = 0; a < 3; ++a) {
    if (spinMatrices[a].rows() != static_cast<Eigen::Index>(modeCount) ||
        spinMatrices[a].cols() != static_cast<Eigen::Index>(modeCount)) {
      std::ostringstream message;
      message << "SharpSpin::applyTotalSpinSquared: spin matrix " << a
              << " is " << spinMatrices[a].rows() << "x"
              << spinMatrices[a].cols() << "; the state carries " << modeCount
              << " modes.";
      throw std::invalid_argument(message.str());
    }
  }
  Eigen::VectorXcd out = Eigen::VectorXcd::Zero(state.size());
  for (std::size_t a = 0; a < 3; ++a) {
    const Eigen::VectorXcd once =
        applyDGamma(spinMatrices[a], state, modeCount);
    out += applyDGamma(spinMatrices[a], once, modeCount);
  }
  return out;
}

Eigen::MatrixXcd SharpSpin::totalSpinSquaredMatrix(
    const std::array<Eigen::MatrixXcd, 3>& spinMatrices,
    std::optional<std::size_t> modeLimit) {
  const Eigen::Index modes = spinMatrices[0].rows();
  // the dense operator has 4^M entries, so M is indexable up to half the bits
  requireFockModes("SharpSpin::totalSpinSquaredMatrix",
                   modes > 0 ? static_cast<std::size_t>(modes) : 0,
                   kIndexableModes / 2, modeLimit);
  const auto modeCount = static_cast<std::size_t>(modes);
  const Eigen::Index dimension =
      static_cast<Eigen::Index>(std::size_t{1} << modeCount);
  Eigen::MatrixXcd out(dimension, dimension);
  for (Eigen::Index column = 0; column < dimension; ++column) {
    Eigen::VectorXcd basis = Eigen::VectorXcd::Zero(dimension);
    basis(column) = cd(1.0, 0.0);
    out.col(column) = applyTotalSpinSquared(spinMatrices, basis);
  }
  return out;
}

SharpSpinRead SharpSpin::read(
    const std::array<Eigen::MatrixXcd, 3>& spinMatrices,
    const Eigen::VectorXcd& rightState, const Eigen::VectorXcd& leftState,
    double targetEigenvalue, double tolerance) {
  requirePositive(tolerance, "SharpSpin::read tolerance");
  if (rightState.size() != leftState.size()) {
    std::ostringstream message;
    message << "SharpSpin::read: the right state has dimension "
            << rightState.size() << " and the left state "
            << leftState.size()
            << "; a biorthogonal pair lives in one space.";
    throw std::invalid_argument(message.str());
  }
  const double rightNorm = rightState.norm();
  const double leftNorm = leftState.norm();
  if (rightNorm == 0.0 || leftNorm == 0.0) {
    throw std::invalid_argument(
        "SharpSpin::read: neither state may be the zero vector; a zero vector "
        "satisfies every eigen-equation vacuously.");
  }

  SharpSpinRead read;
  read.targetEigenvalue = targetEigenvalue;
  for (Eigen::Index n = 0; n < rightState.size(); ++n) {
    if (rightState(n) != cd(0.0, 0.0)) ++read.determinantCount;
  }

  const Eigen::VectorXcd rightImage =
      applyTotalSpinSquared(spinMatrices, rightState);
  const Eigen::VectorXcd rightResidualVector =
      rightImage - targetEigenvalue * rightState;
  read.rightResidual = rightResidualVector.norm() / rightNorm;

  // The left eigen-equation <Psi_L|(J^2 - target I) = 0 is the right one for
  // the transposed operator, because the pairing is bilinear:
  // (Psi_L^T J^2)^T = (J^2)^T Psi_L. In the occupation basis the creation and
  // annihilation matrices are real and mutually transposed, so
  // dGamma(M)^T = dGamma(M^T); hence
  // (J^2)^T = sum_a (dGamma(J_a)^T)^2 = sum_a dGamma(J_a^T)^2, and applying
  // the same routine to the transposed one-particle spin matrices evaluates
  // the left residual exactly.
  std::array<Eigen::MatrixXcd, 3> transposed{
      spinMatrices[0].transpose(), spinMatrices[1].transpose(),
      spinMatrices[2].transpose()};
  const Eigen::VectorXcd leftImage =
      applyTotalSpinSquared(transposed, leftState);
  const Eigen::VectorXcd leftResidualVector =
      leftImage - targetEigenvalue * leftState;
  read.leftResidual = leftResidualVector.norm() / leftNorm;

  read.sharp =
      read.rightResidual <= tolerance && read.leftResidual <= tolerance;

  const cd pairing = (leftState.transpose() * rightState)(0, 0);
  if (std::abs(pairing) > 0.0) {
    const cd first = (leftState.transpose() * rightImage)(0, 0);
    const Eigen::VectorXcd rightImageTwice =
        applyTotalSpinSquared(spinMatrices, rightImage);
    const cd second = (leftState.transpose() * rightImageTwice)(0, 0);
    read.expectation = first / pairing;
    read.variance = second / pairing - read.expectation * read.expectation;
    read.varianceWouldAccept =
        std::abs(read.expectation - cd(targetEigenvalue, 0.0)) <= tolerance &&
        std::abs(read.variance) <= tolerance;
  }

  const double residual = std::max(read.rightResidual, read.leftResidual);
  read.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, CertificateRegime::NonNormal, residual,
      tolerance);
  return read;
}

std::array<Eigen::MatrixXcd, 3> SharpSpin::doubletSpinMatrices(
    std::size_t carrierCount) {
  if (carrierCount == 0) {
    throw std::invalid_argument(
        "SharpSpin::doubletSpinMatrices: at least one spin-one-half carrier "
        "is required.");
  }
  const std::size_t modeCount = 2 * carrierCount;
  Eigen::Matrix2cd sx;
  sx << cd(0.0, 0.0), cd(0.5, 0.0), cd(0.5, 0.0), cd(0.0, 0.0);
  Eigen::Matrix2cd sy;
  sy << cd(0.0, 0.0), cd(0.0, -0.5), cd(0.0, 0.5), cd(0.0, 0.0);
  Eigen::Matrix2cd sz;
  sz << cd(0.5, 0.0), cd(0.0, 0.0), cd(0.0, 0.0), cd(-0.5, 0.0);
  const std::array<Eigen::Matrix2cd, 3> pauli{sx, sy, sz};

  std::array<Eigen::MatrixXcd, 3> out;
  const auto dimension = static_cast<Eigen::Index>(modeCount);
  for (std::size_t a = 0; a < 3; ++a) {
    out[a] = Eigen::MatrixXcd::Zero(dimension, dimension);
    for (std::size_t c = 0; c < carrierCount; ++c) {
      const auto offset = static_cast<Eigen::Index>(2 * c);
      out[a].block(offset, offset, 2, 2) = pauli[a];
    }
  }
  return out;
}

// ---- the isotypic read: the sharp spinor certificate of WP v18 ------------

namespace {

/// The determinant of the square block of \p matrix at rows \p rows and
/// columns \p columns, by Gaussian elimination with partial pivoting on a
/// caller-owned buffer, so that a minor costs no allocation.
cd minorDeterminant(const Eigen::MatrixXcd& matrix,
                    const std::vector<std::size_t>& rows,
                    const std::vector<std::size_t>& columns,
                    Eigen::MatrixXcd& buffer) {
  const auto n = static_cast<Eigen::Index>(rows.size());
  for (Eigen::Index r = 0; r < n; ++r)
    for (Eigen::Index c = 0; c < n; ++c)
      buffer(r, c) = matrix(static_cast<Eigen::Index>(rows[static_cast<std::size_t>(r)]),
                            static_cast<Eigen::Index>(columns[static_cast<std::size_t>(c)]));
  cd determinant(1.0, 0.0);
  for (Eigen::Index k = 0; k < n; ++k) {
    Eigen::Index pivot = k;
    double largest = std::abs(buffer(k, k));
    for (Eigen::Index r = k + 1; r < n; ++r) {
      const double magnitude = std::abs(buffer(r, k));
      if (magnitude > largest) {
        largest = magnitude;
        pivot = r;
      }
    }
    if (largest == 0.0) return cd(0.0, 0.0);
    if (pivot != k) {
      buffer.row(pivot).swap(buffer.row(k));
      determinant = -determinant;
    }
    determinant *= buffer(k, k);
    for (Eigen::Index r = k + 1; r < n; ++r) {
      const cd factor = buffer(r, k) / buffer(k, k);
      if (factor == cd(0.0, 0.0)) continue;
      for (Eigen::Index c = k + 1; c < n; ++c)
        buffer(r, c) -= factor * buffer(k, c);
    }
  }
  return determinant;
}

/// The Fock basis index of a pattern: mode i is bit i.
std::size_t patternIndex(const std::vector<std::size_t>& pattern) {
  std::size_t index = 0;
  for (const std::size_t mode : pattern) index |= std::size_t{1} << mode;
  return index;
}

}  // namespace

std::vector<std::vector<std::size_t>> SharpSpin::sectorPatterns(
    std::size_t modeCount, std::size_t particles,
    std::optional<std::size_t> patternLimit) {
  if (particles > modeCount) {
    std::ostringstream message;
    message << "SharpSpin::sectorPatterns: " << particles
            << " particles do not fit in " << modeCount << " modes.";
    throw std::invalid_argument(message.str());
  }
  // the number of patterns, C(M, n), compared with a declared limit before
  // anything is built
  if (patternLimit.has_value()) {
    double count = 1.0;
    for (std::size_t k = 1; k <= particles; ++k)
      count = count * static_cast<double>(modeCount - particles + k) /
              static_cast<double>(k);
    if (count > static_cast<double>(*patternLimit))
      declaredLimitReached("SharpSpin::sectorPatterns", "pattern",
                           *patternLimit, count);
  }
  std::vector<std::vector<std::size_t>> patterns;
  std::vector<std::size_t> current(particles);
  for (std::size_t k = 0; k < particles; ++k) current[k] = k;
  while (true) {
    patterns.push_back(current);
    if (particles == 0) break;
    // the next combination in lexicographic order
    std::size_t position = particles;
    while (position > 0 &&
           current[position - 1] == modeCount - particles + position - 1)
      --position;
    if (position == 0) break;
    ++current[position - 1];
    for (std::size_t k = position; k < particles; ++k)
      current[k] = current[k - 1] + 1;
  }
  return patterns;
}

Eigen::MatrixXcd SharpSpin::exteriorPowerMatrix(
    const Eigen::MatrixXcd& oneParticle, std::size_t particles,
    std::optional<std::size_t> patternLimit) {
  if (oneParticle.rows() != oneParticle.cols() || oneParticle.rows() <= 0) {
    std::ostringstream message;
    message << "SharpSpin::exteriorPowerMatrix: the one-particle map is "
            << oneParticle.rows() << "x" << oneParticle.cols()
            << "; a map of the mode space is square.";
    throw std::invalid_argument(message.str());
  }
  const auto modeCount = static_cast<std::size_t>(oneParticle.rows());
  const auto patterns = sectorPatterns(modeCount, particles, patternLimit);
  const auto dimension = static_cast<Eigen::Index>(patterns.size());
  Eigen::MatrixXcd out(dimension, dimension);
  Eigen::MatrixXcd buffer(static_cast<Eigen::Index>(particles),
                          static_cast<Eigen::Index>(particles));
  for (Eigen::Index column = 0; column < dimension; ++column)
    for (Eigen::Index row = 0; row < dimension; ++row)
      out(row, column) = particles == 0
                             ? cd(1.0, 0.0)
                             : minorDeterminant(
                                   oneParticle,
                                   patterns[static_cast<std::size_t>(row)],
                                   patterns[static_cast<std::size_t>(column)],
                                   buffer);
  return out;
}

Eigen::VectorXcd SharpSpin::sectorComponent(
    const Eigen::VectorXcd& state, std::size_t particles,
    std::optional<std::size_t> patternLimit) {
  const std::size_t modeCount = modeCountOf(state.size());
  if (modeCount == 0) {
    std::ostringstream message;
    message << "SharpSpin::sectorComponent: the state has dimension "
            << state.size()
            << ", which is not 2^M for any mode count M, so it is not a Fock "
               "vector.";
    throw std::invalid_argument(message.str());
  }
  const auto patterns = sectorPatterns(modeCount, particles, patternLimit);
  Eigen::VectorXcd out(static_cast<Eigen::Index>(patterns.size()));
  for (std::size_t k = 0; k < patterns.size(); ++k)
    out(static_cast<Eigen::Index>(k)) =
        state(static_cast<Eigen::Index>(patternIndex(patterns[k])));
  return out;
}

Eigen::VectorXcd SharpSpin::fockVector(
    const Eigen::VectorXcd& sector, std::size_t modeCount,
    std::size_t particles, std::optional<std::size_t> modeLimit,
    std::optional<std::size_t> patternLimit) {
  requireFockModes("SharpSpin::fockVector", modeCount, kIndexableModes,
                   modeLimit);
  const auto patterns = sectorPatterns(modeCount, particles, patternLimit);
  if (static_cast<std::size_t>(sector.size()) != patterns.size()) {
    std::ostringstream message;
    message << "SharpSpin::fockVector: the sector vector has "
            << sector.size() << " entries; the " << particles
            << "-particle sector of " << modeCount << " modes has "
            << patterns.size() << " patterns.";
    throw std::invalid_argument(message.str());
  }
  Eigen::VectorXcd out = Eigen::VectorXcd::Zero(
      static_cast<Eigen::Index>(std::size_t{1} << modeCount));
  for (std::size_t k = 0; k < patterns.size(); ++k)
    out(static_cast<Eigen::Index>(patternIndex(patterns[k]))) =
        sector(static_cast<Eigen::Index>(k));
  return out;
}

Eigen::MatrixXcd SharpSpin::isotypicProjector(
    const std::vector<Eigen::MatrixXcd>& maps,
    const std::vector<Complex>& characters, std::size_t dimension,
    std::size_t particles, std::optional<std::size_t> patternLimit) {
  if (maps.empty()) {
    throw std::invalid_argument(
        "SharpSpin::isotypicProjector: the group has no elements.");
  }
  if (maps.size() != characters.size()) {
    std::ostringstream message;
    message << "SharpSpin::isotypicProjector: " << maps.size()
            << " maps were supplied with " << characters.size()
            << " character values; one value per element is needed.";
    throw std::invalid_argument(message.str());
  }
  if (dimension == 0) {
    throw std::invalid_argument(
        "SharpSpin::isotypicProjector: the representation has dimension "
        "zero.");
  }
  const Eigen::Index modes = maps.front().rows();
  for (std::size_t g = 0; g < maps.size(); ++g) {
    if (maps[g].rows() != modes || maps[g].cols() != modes) {
      std::ostringstream message;
      message << "SharpSpin::isotypicProjector: map " << g << " is "
              << maps[g].rows() << "x" << maps[g].cols()
              << "; every map must be " << modes << "x" << modes << ".";
      throw std::invalid_argument(message.str());
    }
  }
  Eigen::MatrixXcd out;
  for (std::size_t g = 0; g < maps.size(); ++g) {
    const Eigen::MatrixXcd term =
        exteriorPowerMatrix(maps[g], particles, patternLimit);
    if (out.size() == 0) out = Eigen::MatrixXcd::Zero(term.rows(), term.cols());
    out += std::conj(characters[g]) * term;
  }
  out *= static_cast<double>(dimension) / static_cast<double>(maps.size());
  return out;
}

IsotypicRead SharpSpin::isotypicRead(const Eigen::MatrixXcd& projector,
                                     const Eigen::VectorXcd& rightState,
                                     const Eigen::VectorXcd& leftState,
                                     const std::string& type,
                                     double tolerance) {
  requirePositive(tolerance, "SharpSpin::isotypicRead tolerance");
  if (projector.rows() != projector.cols() ||
      rightState.size() != projector.rows() ||
      leftState.size() != projector.rows()) {
    std::ostringstream message;
    message << "SharpSpin::isotypicRead: the projector is " << projector.rows()
            << "x" << projector.cols() << ", the right state has dimension "
            << rightState.size() << " and the left state "
            << leftState.size() << "; all three live in one sector.";
    throw std::invalid_argument(message.str());
  }
  const double rightNorm = rightState.norm();
  const double leftNorm = leftState.norm();
  if (rightNorm == 0.0 || leftNorm == 0.0) {
    throw std::invalid_argument(
        "SharpSpin::isotypicRead: neither state may be the zero vector; a "
        "zero vector satisfies every projector equation vacuously.");
  }
  IsotypicRead read;
  read.type = type;
  read.rank = static_cast<std::size_t>(
      std::llround(std::max(0.0, projector.trace().real())));
  const Eigen::VectorXcd rightImage = projector * rightState;
  read.rightResidual = (rightState - rightImage).norm() / rightNorm;
  // The projector's idempotency, measured along the right state rather than
  // on the whole matrix: one more matrix-vector product instead of a matrix
  // square, which a read per pole cannot afford on an 816-dimensional sector.
  read.idempotencyResidual =
      (projector * rightImage - rightImage).norm() / rightNorm;
  // <Psi_L|(I - P) = 0 is the right equation for the transposed projector,
  // because the pairing is bilinear: (Psi_L^T P)^T = P^T Psi_L.
  const Eigen::VectorXcd leftImage = projector.transpose() * leftState;
  read.leftResidual = (leftState - leftImage).norm() / leftNorm;
  read.sharp =
      read.rightResidual <= tolerance && read.leftResidual <= tolerance;
  const cd pairing = (leftState.transpose() * rightState)(0, 0);
  if (std::abs(pairing) > 0.0)
    read.weight = (leftState.transpose() * rightImage)(0, 0) / pairing;
  const double residual = std::max(read.rightResidual, read.leftResidual);
  read.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, CertificateRegime::NonNormal, residual,
      tolerance);
  return read;
}

}  // namespace tessera::observables
