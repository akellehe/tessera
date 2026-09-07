// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/SimplicialQubit.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>

#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

namespace {

using Complex = std::complex<double>;
using chainhodge::Connection;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = std::numeric_limits<double>::epsilon();
/// Rounding tolerance of the pure-gauge certificate (spec section 16): a
/// gauge phi_e = g(target) - g(source) evaluated in doubles closes every loop
/// to a few ulps; flux or holonomy of physical size sits far above this.
constexpr double kGaugeTolerance = 1e-10;
/// The sign relating the ordered simplicial cup product (f_A ∪ f_B)[K] of the
/// period frame to the intersection number A · B of the marked cycles, fixed
/// so that the flat torus of spec §12 (counterclockwise faces, A along 1 and
/// B along tau: A · B = +1 by construction) reads +1 (`intersectionNumber`).
constexpr double kCupSign = 1.0;

/// rot90(x, y) = (-y, x): the +90 degree rotation of spec §7 / §8.
Eigen::Vector2d rot90(const Eigen::Vector2d &v) { return Eigen::Vector2d(-v(1), v(0)); }
Eigen::Vector2cd rot90c(const Eigen::Vector2cd &v) { return Eigen::Vector2cd(-v(1), v(0)); }

/// The transpose (bilinear) pairing of spec §16: a . b = a_1 b_1 + a_2 b_2,
/// never a conjugate (Eigen's dot() conjugates its first argument).
Complex pairT(const Eigen::Vector2cd &a, const Eigen::Vector2cd &b) { return a(0) * b(0) + a(1) * b(1); }

std::string edgeText(std::uint64_t u, std::uint64_t v) {
  return "(" + std::to_string(u) + "," + std::to_string(v) + ")";
}

std::string faceText(const SimplicialQubit::Face &f) {
  return "(" + std::to_string(f[0]) + "," + std::to_string(f[1]) + "," + std::to_string(f[2]) + ")";
}

std::string complexText(Complex z) {
  std::ostringstream s;
  s << z.real() << (z.imag() < 0 ? "-" : "+") << std::abs(z.imag()) << "i";
  return s.str();
}

/// The face's boundary traversal (spec §3): local edge slots 0, 1, 2 are
/// (i, j), (j, k), (k, i); slot s is opposite the local vertex (s + 2) mod 3.
std::pair<std::uint64_t, std::uint64_t> traversal(const SimplicialQubit::Face &f, int slot) {
  switch (slot) {
    case 0: return {f[0], f[1]};
    case 1: return {f[1], f[2]};
    default: return {f[2], f[0]};
  }
}

int oppositeVertexSlot(int edgeSlot) { return (edgeSlot + 2) % 3; }

/// Spec §4: the three angles of a triangle with a = l(jk), b = l(ki), c = l(ij)
/// by the law of cosines (the cosine is clamped to [-1, 1]; with the strict
/// triangle inequality it lies strictly inside, so the clamp only absorbs
/// rounding).
std::array<double, 3> anglesOf(double a, double b, double c) {
  auto angle = [](double cosine) { return std::acos(std::clamp(cosine, -1.0, 1.0)); };
  return {angle((b * b + c * c - a * a) / (2.0 * b * c)),
          angle((c * c + a * a - b * b) / (2.0 * c * a)),
          angle((a * a + b * b - c * c) / (2.0 * a * b))};
}

int numericalRank(const Eigen::MatrixXd &m) {
  if (m.cols() == 0 || m.rows() == 0) return 0;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(m);
  lu.setThreshold(1e-9);
  return static_cast<int>(lu.rank());
}

/// The principal square root with a real argument kept on the +0 side of the
/// cut (the `WhitneyMass` convention): a negative real squared length gives
/// +i times the root, never -i through a signed zero.
Complex principalSqrt(Complex z) {
  if (z.imag() == 0.0) z = Complex(z.real(), 0.0);
  return std::sqrt(z);
}

/// scipy.linalg.null_space over C: the right singular vectors of the singular
/// values at or below eps * max(M, N) * sigma_max (spec §6, §16).
Eigen::MatrixXcd complexNullSpace(const Eigen::MatrixXcd &S) {
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(S, Eigen::ComputeFullV);
  const Eigen::VectorXd sigma = svd.singularValues();
  const double tolerance = kEps * static_cast<double>(std::max(S.rows(), S.cols())) *
                           (sigma.size() > 0 ? sigma(0) : 0.0);
  Eigen::Index rank = 0;
  for (Eigen::Index n = 0; n < sigma.size(); ++n)
    if (sigma(n) > tolerance) ++rank;
  return svd.matrixV().rightCols(S.cols() - rank);
}

/// The Hermitian overlap |<a, b>| / (|a| |b|) of two lines (the tracking
/// criterion of spec §16's continuity rule; a numerical criterion, not a
/// pairing of the construction).
double lineOverlap(const Eigen::VectorXcd &a, const Eigen::VectorXcd &b) {
  const double scale = a.norm() * b.norm();
  return scale > 0.0 ? std::abs(a.dot(b)) / scale : 0.0;
}

}  // namespace

/// The §4–§8 quantities of the complex path at one point of the segment from
/// the real reference (spec §16).
struct SimplicialQubit::ComplexStage {
  Eigen::MatrixXcd angles;
  Eigen::VectorXcd areas;
  Eigen::MatrixXcd layout;
  Eigen::MatrixXcd gradients;
  Eigen::VectorXcd weights;
  Eigen::MatrixXcd H;
  Eigen::MatrixXcd Hdual;
  Eigen::MatrixXcd G;
  Eigen::MatrixXcd R;
  Eigen::MatrixXcd J;
  double jResidual{0.0};
};

// ============================================================================
// Constructors
// ============================================================================

SimplicialQubit::SimplicialQubit(std::vector<std::uint64_t> vertices, std::vector<EdgePair> edges,
                                 std::vector<Face> faces, std::vector<Complex> lengths, Cycle cycleA,
                                 Cycle cycleB, double degeneracyThreshold)
    : vertices_(std::move(vertices)),
      edges_(std::move(edges)),
      faces_(std::move(faces)),
      lengths_(std::move(lengths)),
      cycleA_(std::move(cycleA)),
      cycleB_(std::move(cycleB)),
      degeneracyThreshold_(degeneracyThreshold) {
  indexEdges();
  validateCombinatorics();
  // The container: the faces as cells, then the lengths onto its edges. The
  // container keeps its own (sorted) vertex order per face; the oriented faces
  // stay in faces_. The phases are zero: the trivial connection.
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(faces_.size());
  for (const Face &f : faces_) cells.push_back({f[0], f[1], f[2]});
  spacetime_ = Spacetime::fromCells(2, cells, 1.0, Complex(0.0, 0.0));
  for (mesh::Edge *edge : spacetime_->getEdgeList()->toVector()) {
    const std::uint64_t u = edge->getSource()->getId();
    const std::uint64_t v = edge->getTarget()->getId();
    edge->setLength(lengths_[edgeIndexOf(std::min(u, v), std::max(u, v))]);
    edge->setPhase(Complex(0.0, 0.0));
  }
  buildConnection({});
  initialize();
}

SimplicialQubit::SimplicialQubit(const std::shared_ptr<Spacetime> &spacetime, Cycle cycleA,
                                 Cycle cycleB, bool reversed, double degeneracyThreshold)
    : cycleA_(std::move(cycleA)),
      cycleB_(std::move(cycleB)),
      degeneracyThreshold_(degeneracyThreshold),
      spacetime_(spacetime) {
  if (!spacetime_) throw std::invalid_argument("SimplicialQubit: null spacetime");

  // Vertices: V = [0 .. nV-1] by ascending id (an order-preserving relabeling,
  // so the canonical edge order over ids and over indices coincide).
  std::vector<std::uint64_t> ids;
  for (const auto *vertex : spacetime_->getVertexList()->liveVector()) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  std::map<std::uint64_t, std::uint64_t> indexOfId;
  for (std::size_t n = 0; n < ids.size(); ++n) indexOfId[ids[n]] = n;
  vertices_.resize(ids.size());
  for (std::size_t n = 0; n < ids.size(); ++n) vertices_[n] = n;

  // Edges (i < j, ascending) with their lengths (validated in indexEdges).
  std::vector<std::pair<EdgePair, Complex>> edgeLengths;
  for (const mesh::Edge *edge : spacetime_->getEdgeList()->toVector()) {
    const std::uint64_t a = indexOfId.at(edge->getSource()->getId());
    const std::uint64_t b = indexOfId.at(edge->getTarget()->getId());
    edgeLengths.push_back({{std::min(a, b), std::max(a, b)}, edge->getLength()});
  }
  std::sort(edgeLengths.begin(), edgeLengths.end(),
            [](const auto &x, const auto &y) { return x.first < y.first; });
  for (const auto &[pair, length] : edgeLengths) {
    edges_.push_back(pair);
    lengths_.push_back(length);
  }

  // Faces: the triangles, consistently oriented by the fundamental class
  // (the container sorts vertex orders, so orientation is derived here).
  std::vector<std::vector<std::uint64_t>> cells, cellsById;
  for (const auto &simplex : spacetime_->getSimplices()) {
    if (simplex->size() != 3) continue;
    std::vector<std::uint64_t> cell, byId;
    for (const auto &vertex : simplex->getVertices()) {
      cell.push_back(indexOfId.at(vertex->getId()));
      byId.push_back(vertex->getId());
    }
    std::sort(cell.begin(), cell.end());
    cells.push_back(std::move(cell));
    cellsById.push_back(std::move(byId));
  }
  if (cells.empty()) throw std::invalid_argument("SimplicialQubit: the spacetime has no triangles");
  const cobordism::ChainComplex K = cobordism::ChainComplex::fromTopCells(cells);
  std::vector<int> epsilon;
  try {
    epsilon = K.fundamentalClass();
  } catch (const std::runtime_error &e) {
    throw std::invalid_argument(
        std::string("SimplicialQubit: the faces cannot be consistently oriented (") + e.what() + ")");
  }
  const auto canonical = K.kSimplexVertices(2);
  for (std::size_t t = 0; t < canonical.size(); ++t) {
    const auto &c = canonical[t];
    const bool counterclockwise = (epsilon[t] > 0) != reversed;
    faces_.push_back(counterclockwise ? Face{c[0], c[1], c[2]} : Face{c[0], c[2], c[1]});
  }

  indexEdges();
  validateCombinatorics();
  // The link phases (spec §16), by the Connection::fromSpacetime convention
  // over the ids; the relabeling is monotone, so the canonical order over ids
  // is the canonical order over indices.
  buildConnection(Connection::fromSpacetime(*spacetime_, cobordism::ChainComplex::fromTopCells(cellsById)).links());
  initialize();
}

void SimplicialQubit::buildConnection(std::vector<Complex> canonicalLinks) {
  std::vector<std::vector<std::uint64_t>> cells;
  cells.reserve(faces_.size());
  for (const Face &f : faces_) cells.push_back({f[0], f[1], f[2]});
  const cobordism::ChainComplex K = cobordism::ChainComplex::fromTopCells(cells);
  const auto canonical = K.kSimplexVertices(1);
  if (canonical.size() != edges_.size())
    throw std::logic_error("SimplicialQubit: the chain complex has " + std::to_string(canonical.size()) +
                           " edges for " + std::to_string(edges_.size()) + " of E");
  if (canonicalLinks.empty()) canonicalLinks.assign(canonical.size(), Complex(1.0, 0.0));
  if (canonicalLinks.size() != canonical.size())
    throw std::logic_error("SimplicialQubit: one link per canonical edge is required");
  canonicalOf_.assign(edges_.size(), 0);
  links_.assign(edges_.size(), Complex(1.0, 0.0));
  for (std::size_t j = 0; j < canonical.size(); ++j) {
    const std::size_t e = edgeIndexOf(canonical[j][0], canonical[j][1]);
    canonicalOf_[e] = j;
    links_[e] = canonicalLinks[j];
  }
  connection_ = std::make_shared<const Connection>(K, std::move(canonicalLinks));
  trivialConnection_ = std::all_of(links_.begin(), links_.end(),
                                   [](Complex u) { return u == Complex(1.0, 0.0); });
  real_ = trivialConnection_ &&
          std::all_of(lengths_.begin(), lengths_.end(), [](Complex l) { return l.imag() == 0.0; });
}

void SimplicialQubit::initialize() {
  buildIncidence();
  validateCycles();
  validatePureGauge();
  buildWalks();
  if (real_) {
    buildFaceGeometry();
    buildWeights();
    buildHarmonicSpace();
    buildComplexStructure();
  } else {
    // Spec §16: every formula of §4–§8 over C, on the twisted complex.
    ComplexStage stage = complexStageAt(lengths_);
    angles_ = std::move(stage.angles);
    areas_ = std::move(stage.areas);
    layout_ = std::move(stage.layout);
    gradients_ = std::move(stage.gradients);
    weights_ = std::move(stage.weights);
    H_ = std::move(stage.H);
    Hdual_ = std::move(stage.Hdual);
    G_ = std::move(stage.G);
    R_ = std::move(stage.R);
    J_ = std::move(stage.J);
    jResidual_ = stage.jResidual;
  }
  buildHolomorphicLine();
  buildPeriodFrame();
  diagnoseDegeneration();
}

// ============================================================================
// Spec section 2: input validation
// ============================================================================

void SimplicialQubit::indexEdges() {
  const std::size_t nV = vertices_.size();
  std::vector<std::uint64_t> sorted = vertices_;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t n = 0; n < nV; ++n)
    if (sorted[n] != n)
      throw std::invalid_argument("SimplicialQubit: vertices must be exactly 0 .. nV-1");
  edgeIndex_.clear();
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const auto [i, j] = edges_[e];
    if (!(i < j) || j >= nV)
      throw std::invalid_argument("SimplicialQubit: edge " + edgeText(i, j) +
                                  " must satisfy i < j < nV");
    if (!edgeIndex_.emplace(edges_[e], e).second)
      throw std::invalid_argument("SimplicialQubit: duplicate edge " + edgeText(i, j));
  }
  if (lengths_.size() != edges_.size())
    throw std::invalid_argument("SimplicialQubit: expected one length per edge (" +
                                std::to_string(edges_.size()) + "), got " +
                                std::to_string(lengths_.size()));
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const Complex length = lengths_[e];
    const std::string where = "SimplicialQubit: edge " + edgeText(edges_[e].first, edges_[e].second) +
                              " has length " + complexText(length);
    if (!std::isfinite(length.real()) || !std::isfinite(length.imag()) || length == Complex(0.0, 0.0))
      throw std::invalid_argument(where + "; lengths must be nonzero and finite");
    // Spec §2 on the real locus; §16 admits any nonzero complex length (only
    // its square enters the construction).
    if (length.imag() == 0.0 && !(length.real() > 0.0))
      throw std::invalid_argument(where + "; real lengths must be real and positive (spec section 2)");
  }
}

std::size_t SimplicialQubit::edgeIndexOf(std::uint64_t u, std::uint64_t v) const {
  const auto it = edgeIndex_.find({std::min(u, v), std::max(u, v)});
  if (it == edgeIndex_.end())
    throw std::invalid_argument("SimplicialQubit: " + edgeText(std::min(u, v), std::max(u, v)) +
                                " is not an edge of E");
  return it->second;
}

void SimplicialQubit::validateCombinatorics() {
  const std::size_t nV = vertices_.size();
  const std::size_t nE = edges_.size();
  const std::size_t nF = faces_.size();

  // Every face: three distinct vertices, three edges of E; record the
  // incidence (face, slot) and the traversal sign of each edge.
  edgeFaces_.assign(nE, {});
  std::vector<std::vector<int>> traversalSigns(nE);
  for (std::size_t t = 0; t < nF; ++t) {
    const Face &f = faces_[t];
    if (f[0] == f[1] || f[1] == f[2] || f[0] == f[2] || f[0] >= nV || f[1] >= nV || f[2] >= nV)
      throw std::invalid_argument("SimplicialQubit: face " + faceText(f) +
                                  " must have three distinct vertices below nV");
    for (int slot = 0; slot < 3; ++slot) {
      const auto [u, v] = traversal(f, slot);
      std::size_t e;
      try {
        e = edgeIndexOf(u, v);
      } catch (const std::invalid_argument &) {
        throw std::invalid_argument("SimplicialQubit: face " + faceText(f) + " uses " +
                                    edgeText(std::min(u, v), std::max(u, v)) + ", which is not in E");
      }
      edgeFaces_[e].push_back({t, slot});
      traversalSigns[e].push_back(u < v ? 1 : -1);
    }
  }

  // Every edge belongs to exactly 2 faces (closed surface), traversed in
  // opposite directions by the two (consistent orientation).
  for (std::size_t e = 0; e < nE; ++e) {
    if (edgeFaces_[e].size() != 2)
      throw std::invalid_argument("SimplicialQubit: edge " + edgeText(edges_[e].first, edges_[e].second) +
                                  " belongs to " + std::to_string(edgeFaces_[e].size()) +
                                  " faces; a closed surface needs exactly 2");
    if (traversalSigns[e][0] == traversalSigns[e][1])
      throw std::invalid_argument("SimplicialQubit: face orientations are inconsistent: faces " +
                                  faceText(faces_[edgeFaces_[e][0].first]) + " and " +
                                  faceText(faces_[edgeFaces_[e][1].first]) + " traverse edge " +
                                  edgeText(edges_[e].first, edges_[e].second) + " in the same direction");
  }

  // Euler characteristic zero.
  const long chi = static_cast<long>(nV) - static_cast<long>(nE) + static_cast<long>(nF);
  if (chi != 0)
    throw std::invalid_argument("SimplicialQubit: nV - nE + nF = " + std::to_string(chi) +
                                "; a torus needs Euler characteristic 0");

  // The strict triangle inequality on every face, on real lengths (§2); off
  // the real locus the admissibility of a face is the existence of its
  // continuation from the real reference (§16, buildFaceGeometry).
  const bool allReal = std::all_of(lengths_.begin(), lengths_.end(), [](Complex l) { return l.imag() == 0.0; });
  if (!allReal) return;
  for (const Face &f : faces_) {
    const double a = lengths_[edgeIndexOf(f[1], f[2])].real();
    const double b = lengths_[edgeIndexOf(f[2], f[0])].real();
    const double c = lengths_[edgeIndexOf(f[0], f[1])].real();
    if (!(a < b + c && b < c + a && c < a + b)) {
      std::ostringstream s;
      s << "SimplicialQubit: face " << faceText(f) << " violates the strict triangle inequality (a = "
        << a << ", b = " << b << ", c = " << c << ")";
      throw std::invalid_argument(s.str());
    }
  }
}

void SimplicialQubit::validateCycles() const {
  const std::size_t nV = vertices_.size();
  const std::size_t nE = edges_.size();
  auto chainOf = [&](const Cycle &cycle, const char *name) {
    if (cycle.empty())
      throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name + " is empty");
    Eigen::VectorXd chain = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nE));
    std::vector<long> net(nV, 0);
    for (const auto &[e, sign] : cycle) {
      if (e >= nE)
        throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                    " refers to edge index " + std::to_string(e) + " >= nE");
      if (sign != 1 && sign != -1)
        throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                    " has a sign that is not +1 or -1");
      chain(static_cast<Eigen::Index>(e)) += sign;
      const auto [i, j] = edges_[e];
      net[j] += sign;  // the edge runs i -> j; the step enters j and leaves i
      net[i] -= sign;
    }
    for (std::size_t v = 0; v < nV; ++v)
      if (net[v] != 0)
        throw std::invalid_argument(std::string("SimplicialQubit: cycle ") + name +
                                    " is not closed (vertex " + std::to_string(v) + " has net degree " +
                                    std::to_string(net[v]) + ")");
    return chain;
  };
  const Eigen::VectorXd cA = chainOf(cycleA_, "A");
  const Eigen::VectorXd cB = chainOf(cycleB_, "B");

  // Independent homology classes: neither cycle, nor a combination, is a
  // boundary — the rank of [d1^T | c_A | c_B] exceeds that of d1^T by two.
  const Eigen::MatrixXd boundaries = d1_.transpose();
  Eigen::MatrixXd augmented(boundaries.rows(), boundaries.cols() + 2);
  augmented << boundaries, cA, cB;
  if (numericalRank(augmented) != numericalRank(boundaries) + 2)
    throw std::invalid_argument(
        "SimplicialQubit: the homology classes of cycles A and B are not independent");
}

// ============================================================================
// Spec section 16: the link phases must be a pure gauge
// ============================================================================

void SimplicialQubit::validatePureGauge() const {
  if (trivialConnection_) return;
  // Flux: the curvature of every face must be 1.
  for (const Face &f : faces_) {
    std::array<std::uint64_t, 3> s = f;
    std::sort(s.begin(), s.end());
    const Complex F = connection_->curvature(s[0], s[1], s[2]);
    if (std::abs(F - Complex(1.0, 0.0)) > kGaugeTolerance)
      throw std::invalid_argument("SimplicialQubit: the link phases are not a pure gauge: face (" +
                                  std::to_string(s[0]) + "," + std::to_string(s[1]) + "," +
                                  std::to_string(s[2]) + ") carries flux F = U_rq U_qp U_pr = " +
                                  complexText(F) + " (|F - 1| = " +
                                  std::to_string(std::abs(F - Complex(1.0, 0.0))) +
                                  "); the twisted harmonic space of a torus with flux does not have "
                                  "dimension 2 (spec section 16)");
  }
  // Holonomy: a flat connection is a pure gauge U_xy = g_x^{-1} g_y iff the
  // gauge propagated along a spanning tree (g_y = g_x U_xy) closes every
  // non-tree edge; the defect of an edge is the holonomy around the cycle it
  // closes. The root is a choice; the verdict is not.
  const std::size_t nV = vertices_.size();
  std::vector<std::vector<std::pair<std::uint64_t, std::size_t>>> adjacency(nV);
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    adjacency[edges_[e].first].push_back({edges_[e].second, e});
    adjacency[edges_[e].second].push_back({edges_[e].first, e});
  }
  std::vector<Complex> g(nV, Complex(0.0, 0.0));
  std::vector<bool> treeEdge(edges_.size(), false);
  std::deque<std::uint64_t> queue;
  g[0] = Complex(1.0, 0.0);
  queue.push_back(0);
  while (!queue.empty()) {
    const std::uint64_t x = queue.front();
    queue.pop_front();
    for (const auto &[y, e] : adjacency[x]) {
      if (g[y] != Complex(0.0, 0.0)) continue;
      g[y] = g[x] * connection_->link(x, y);
      treeEdge[e] = true;
      queue.push_back(y);
    }
  }
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    if (treeEdge[e]) continue;
    const auto [x, y] = edges_[e];
    if (g[x] == Complex(0.0, 0.0) || g[y] == Complex(0.0, 0.0))
      throw std::invalid_argument("SimplicialQubit: the 1-skeleton is not connected");
    const Complex h = g[x] * connection_->link(x, y) / g[y];
    if (std::abs(h - Complex(1.0, 0.0)) > kGaugeTolerance)
      throw std::invalid_argument("SimplicialQubit: the link phases are not a pure gauge: the connection is flat "
                                  "but has holonomy h = " + complexText(h) + " around the cycle closed by edge " +
                                  edgeText(x, y) + " (|h - 1| = " +
                                  std::to_string(std::abs(h - Complex(1.0, 0.0))) +
                                  "); a flat connection with holonomy has no twisted harmonic space of "
                                  "dimension 2 (spec section 16)");
  }
}

// ============================================================================
// Spec section 16: the marked cycles as closed walks with a common base point
// ============================================================================

SimplicialQubit::Walk SimplicialQubit::walkOf(const Cycle &cycle, const char *name,
                                              std::string &obstruction) const {
  // The cycle's directed steps in the given order; `Connection::closedWalkOf`
  // (Hierholzer over their multigraph; the net degree of every vertex is zero,
  // validateCycles) starts the circuit at the first step's source and follows
  // the given order whenever it chains.
  Walk steps;
  steps.reserve(cycle.size());
  for (const auto &[e, sign] : cycle)
    steps.push_back(sign > 0 ? std::make_pair(edges_[e].first, edges_[e].second)
                             : std::make_pair(edges_[e].second, edges_[e].first));
  Walk walk = Connection::closedWalkOf(steps);
  if (walk.size() != steps.size()) {
    obstruction = std::string("cycle ") + name + " does not form one closed walk: its steps fall into more than "
                  "one connected component";
    return {};
  }
  return walk;
}

void SimplicialQubit::buildWalks() {
  walkObstruction_.clear();
  baseVertex_.reset();
  walkA_ = walkOf(cycleA_, "A", walkObstruction_);
  if (walkObstruction_.empty()) walkB_ = walkOf(cycleB_, "B", walkObstruction_);
  if (walkObstruction_.empty()) {
    // The common base point: the first vertex of A's walk that lies on B's;
    // both walks are rotated to start there (`Connection::commonBasePoint`,
    // the one rule every reader of a marking uses).
    std::vector<Walk> walks = {walkA_, walkB_};
    baseVertex_ = Connection::commonBasePoint(walks);
    if (baseVertex_) {
      walkA_ = std::move(walks[0]);
      walkB_ = std::move(walks[1]);
    } else {
      walkObstruction_ = "cycles A and B share no vertex, so their transported periods have no common base "
                         "point (A . B = +1 requires them to meet)";
    }
  }
  if (!walkObstruction_.empty() && !trivialConnection_)
    throw std::invalid_argument("SimplicialQubit: " + walkObstruction_ +
                                "; the periods under a nontrivial connection are taken with parallel transport "
                                "along the cycles (spec section 16)");
}

std::uint64_t SimplicialQubit::baseVertex() const {
  if (!baseVertex_) throw std::logic_error("SimplicialQubit::baseVertex: " + walkObstruction_);
  return *baseVertex_;
}

// ============================================================================
// Spec section 3: incidence matrices
// ============================================================================

void SimplicialQubit::buildIncidence() {
  const Eigen::Index nV = static_cast<Eigen::Index>(vertices_.size());
  const Eigen::Index nE = static_cast<Eigen::Index>(edges_.size());
  const Eigen::Index nF = static_cast<Eigen::Index>(faces_.size());
  d0_ = Eigen::MatrixXd::Zero(nE, nV);
  for (Eigen::Index e = 0; e < nE; ++e) {
    d0_(e, static_cast<Eigen::Index>(edges_[static_cast<std::size_t>(e)].first)) = -1.0;
    d0_(e, static_cast<Eigen::Index>(edges_[static_cast<std::size_t>(e)].second)) = 1.0;
  }
  d1_ = Eigen::MatrixXd::Zero(nF, nE);
  for (Eigen::Index t = 0; t < nF; ++t)
    for (int slot = 0; slot < 3; ++slot) {
      const auto [u, v] = traversal(faces_[static_cast<std::size_t>(t)], slot);
      d1_(t, static_cast<Eigen::Index>(edgeIndexOf(u, v))) = (u < v) ? 1.0 : -1.0;
    }
  // Check: d1 @ d0 == 0 exactly (integer arithmetic in doubles).
  if ((d1_ * d0_).cwiseAbs().maxCoeff() != 0.0)
    throw std::logic_error("SimplicialQubit: d1 d0 != 0");
}

// ============================================================================
// Spec section 4 (per-triangle geometry) and section 7 (barycentric gradients),
// the real locus
// ============================================================================

void SimplicialQubit::buildFaceGeometry() {
  const std::size_t nF = faces_.size();
  Eigen::MatrixXd angles(static_cast<Eigen::Index>(nF), 3);
  Eigen::VectorXd areas(static_cast<Eigen::Index>(nF));
  Eigen::MatrixXd layout(static_cast<Eigen::Index>(nF), 6);
  Eigen::MatrixXd gradients(static_cast<Eigen::Index>(nF), 6);
  double gradientScale = 0.0, gradientDefect = 0.0;
  for (std::size_t t = 0; t < nF; ++t) {
    const Face &f = faces_[t];
    const double a = lengths_[edgeIndexOf(f[1], f[2])].real();  // l(jk)
    const double b = lengths_[edgeIndexOf(f[2], f[0])].real();  // l(ki)
    const double c = lengths_[edgeIndexOf(f[0], f[1])].real();  // l(ij)
    const Eigen::Index row = static_cast<Eigen::Index>(t);

    const std::array<double, 3> alpha = anglesOf(a, b, c);
    angles(row, 0) = alpha[0];
    angles(row, 1) = alpha[1];
    angles(row, 2) = alpha[2];

    const double s = 0.5 * (a + b + c);
    const double area = std::sqrt(s * (s - a) * (s - b) * (s - c));
    areas(row) = area;

    const Eigen::Vector2d pi(0.0, 0.0);
    const Eigen::Vector2d pj(c, 0.0);
    const Eigen::Vector2d pk(b * std::cos(alpha[0]), b * std::sin(alpha[0]));
    layout.row(row) << pi(0), pi(1), pj(0), pj(1), pk(0), pk(1);

    const Eigen::Vector2d gi = rot90(pk - pj) / (2.0 * area);
    const Eigen::Vector2d gj = rot90(pi - pk) / (2.0 * area);
    const Eigen::Vector2d gk = rot90(pj - pi) / (2.0 * area);
    gradients.row(row) << gi(0), gi(1), gj(0), gj(1), gk(0), gk(1);
    gradientScale = std::max(gradientScale, gi.norm() + gj.norm() + gk.norm());
    gradientDefect = std::max(gradientDefect, (gi + gj + gk).norm());
  }
  // Verify grad_lambda_i + grad_lambda_j + grad_lambda_k == 0.
  if (gradientDefect > 1e-10 * std::max(gradientScale, 1.0))
    throw std::logic_error("SimplicialQubit: the barycentric gradients of a face do not sum to zero");
  angles_ = angles.cast<Complex>();
  areas_ = areas.cast<Complex>();
  layout_ = layout.cast<Complex>();
  realGradients_ = std::move(gradients);
  gradients_ = realGradients_.cast<Complex>();
}

// ============================================================================
// Spec section 5: cotangent weights, the real locus
// ============================================================================

void SimplicialQubit::buildWeights() {
  const std::size_t nE = edges_.size();
  Eigen::VectorXd weights(static_cast<Eigen::Index>(nE));
  negativeWeightEdges_.clear();
  nonDelaunayEdges_.clear();
  std::vector<double> angleSums(nE, 0.0);
  for (std::size_t e = 0; e < nE; ++e) {
    double cotangentSum = 0.0;
    for (const auto &[t, slot] : edgeFaces_[e]) {
      const double angle = angles_(static_cast<Eigen::Index>(t), oppositeVertexSlot(slot)).real();
      cotangentSum += std::cos(angle) / std::sin(angle);
      angleSums[e] += angle;
    }
    weights(static_cast<Eigen::Index>(e)) = 0.5 * cotangentSum;
  }
  // Flags at rounding tolerance: an edge with alpha_e + beta_e = pi exactly
  // (a co-circular quadrilateral, e.g. every diagonal of a square grid) has
  // weight zero, not a negative weight, however the last bits fall.
  const double scale = std::max(weights.cwiseAbs().maxCoeff(), 1.0);
  for (std::size_t e = 0; e < nE; ++e) {
    if (weights(static_cast<Eigen::Index>(e)) < -1e-12 * scale) negativeWeightEdges_.push_back(e);
    if (angleSums[e] > kPi + 1e-12) nonDelaunayEdges_.push_back(e);
  }
  if (!nonDelaunayEdges_.empty() || !negativeWeightEdges_.empty()) {
    std::ostringstream w;
    w << nonDelaunayEdges_.size() << " edge(s) violate the intrinsic Delaunay condition alpha_e + beta_e "
      << "<= pi and " << negativeWeightEdges_.size() << " cotangent weight(s) are negative; the "
      << "construction is numerically stable only when the condition holds on every edge "
      << "(apply the intrinsic Delaunay edge-flip pass)";
    warnings_.push_back(w.str());
  }
  weights_ = weights.cast<Complex>();
}

// ============================================================================
// Spec section 6: harmonic space, the real locus
// ============================================================================

void SimplicialQubit::buildHarmonicSpace() {
  const Eigen::Index nE = static_cast<Eigen::Index>(edges_.size());
  // S = vstack([d1, d0.T @ M1]); H = null_space(S).
  const Eigen::VectorXd weights = weights_.real();
  Eigen::MatrixXd S(d1_.rows() + d0_.cols(), nE);
  S.topRows(d1_.rows()) = d1_;
  S.bottomRows(d0_.cols()) = d0_.transpose() * weights.asDiagonal();
  Eigen::BDCSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullV);
  const Eigen::VectorXd sigma = svd.singularValues();
  // scipy.linalg.null_space: rcond = eps * max(M, N), tolerance rcond * sigma_max.
  const double tolerance = kEps * static_cast<double>(std::max(S.rows(), S.cols())) *
                           (sigma.size() > 0 ? sigma(0) : 0.0);
  Eigen::Index rank = 0;
  for (Eigen::Index n = 0; n < sigma.size(); ++n)
    if (sigma(n) > tolerance) ++rank;
  const Eigen::MatrixXd H = svd.matrixV().rightCols(nE - rank);
  if (H.cols() != 2)
    throw std::runtime_error("SimplicialQubit: dim H = " + std::to_string(H.cols()) +
                             " != 2; the input is not a torus or the weights are degenerate");
  H_ = H.cast<Complex>();
  Hdual_ = H_;
}

// ============================================================================
// Spec section 7 (Whitney interpolant, L2 inner product) and section 8 (J),
// the real locus
// ============================================================================

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> SimplicialQubit::whitneyAtBarycenter(
    std::size_t t, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &omega) const {
  // W_t(omega) = (1/3) sum over the three oriented edges (u, v) of t of
  // omega_uv (grad_lambda_v - grad_lambda_u), omega_uv signed relative to the
  // stored orientation of the edge.
  const Face &f = faces_[t];
  const Eigen::Index row = static_cast<Eigen::Index>(t);
  auto gradient = [&](int vertexSlot) {
    return Eigen::Vector2d(realGradients_(row, 2 * vertexSlot), realGradients_(row, 2 * vertexSlot + 1));
  };
  Eigen::Matrix<Scalar, 2, 1> w = Eigen::Matrix<Scalar, 2, 1>::Zero();
  for (int slot = 0; slot < 3; ++slot) {
    const int from = slot, to = (slot + 1) % 3;
    const auto [u, v] = traversal(f, slot);
    const Scalar value = (u < v ? Scalar(1.0) : Scalar(-1.0)) *
                         omega(static_cast<Eigen::Index>(edgeIndexOf(u, v)));
    w += value * (gradient(to) - gradient(from)).template cast<Scalar>();
  }
  return w / Scalar(3.0);
}

void SimplicialQubit::buildComplexStructure() {
  // The real path, kept in the real-length construction's own form (member
  // accumulators, the same expressions). Measured against origin/main's
  // build: bit-identical on every case but the square torus, where two
  // entries of G differ by 2 ulps through the compiler's floating-point
  // contraction of this accumulation (-O3 -march=native), which no source
  // form pins down.
  const std::size_t nF = faces_.size();
  const Eigen::MatrixXd H = H_.real();
  const Eigen::VectorXd areas = areas_.real();
  realG_ = Eigen::MatrixXd::Zero(2, 2);
  realR_ = Eigen::MatrixXd::Zero(2, 2);
  for (std::size_t t = 0; t < nF; ++t) {
    const double area = areas(static_cast<Eigen::Index>(t));
    const Eigen::Vector2d w0 = whitneyAtBarycenter<double>(t, H.col(0));
    const Eigen::Vector2d w1 = whitneyAtBarycenter<double>(t, H.col(1));
    const std::array<Eigen::Vector2d, 2> w = {w0, w1};
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        realG_(a, b) += area * w[a].dot(w[b]);
        realR_(a, b) += area * rot90(w[a]).dot(w[b]);
      }
  }
  // J = G^{-1} @ R.T; the residual is exposed, never symmetrized away.
  realJ_ = realG_.inverse() * realR_.transpose();
  jResidual_ = (realJ_ * realJ_ + Eigen::MatrixXd::Identity(2, 2)).norm();
  G_ = realG_.cast<Complex>();
  R_ = realR_.cast<Complex>();
  J_ = realJ_.cast<Complex>();
}

// ============================================================================
// Spec section 16: sections 4–8 over C on the twisted complex, at one point of
// the segment from the real reference
// ============================================================================

Eigen::Vector2cd SimplicialQubit::twistedWhitneyAtBarycenter(std::size_t t, const Eigen::MatrixXcd &gradients,
                                                             const Eigen::VectorXcd &omega, bool dual) const {
  // W_t^U(omega): each edge value carried to the face's base vertex by
  // U_{b(t) b(e)} (the inverse link for the dual), then the interpolant of §7.
  const Face &f = faces_[t];
  const Eigen::Index row = static_cast<Eigen::Index>(t);
  const std::uint64_t base = std::min({f[0], f[1], f[2]});
  auto gradient = [&](int vertexSlot) {
    return Eigen::Vector2cd(gradients(row, 2 * vertexSlot), gradients(row, 2 * vertexSlot + 1));
  };
  Eigen::Vector2cd w = Eigen::Vector2cd::Zero();
  for (int slot = 0; slot < 3; ++slot) {
    const int from = slot, to = (slot + 1) % 3;
    const auto [u, v] = traversal(f, slot);
    const std::uint64_t edgeBase = std::min(u, v);
    const Complex carry = trivialConnection_ ? Complex(1.0, 0.0)
                          : dual                ? connection_->link(edgeBase, base)
                                                : connection_->link(base, edgeBase);
    const Complex value = (u < v ? Complex(1.0, 0.0) : Complex(-1.0, 0.0)) * carry *
                          omega(static_cast<Eigen::Index>(edgeIndexOf(u, v)));
    w += value * (gradient(to) - gradient(from));
  }
  return w / Complex(3.0, 0.0);
}

SimplicialQubit::ComplexStage SimplicialQubit::complexStageAt(const std::vector<Complex> &lengths) const {
  const std::size_t nF = faces_.size();
  const std::size_t nE = edges_.size();
  const Eigen::Index nV = static_cast<Eigen::Index>(vertices_.size());
  ComplexStage stage;
  stage.angles.resize(static_cast<Eigen::Index>(nF), 3);
  stage.areas.resize(static_cast<Eigen::Index>(nF));
  stage.layout.resize(static_cast<Eigen::Index>(nF), 6);
  stage.gradients.resize(static_cast<Eigen::Index>(nF), 6);
  Eigen::MatrixXcd cotangents(static_cast<Eigen::Index>(nF), 3);
  double gradientScale = 0.0, gradientDefect = 0.0;

  // §4 over C. The face's Gram matrix in the WhitneyMass convention, local
  // vertices (0, 1, 2) = (i, j, k): s_01 = c^2, s_02 = b^2, s_12 = a^2.
  for (std::size_t t = 0; t < nF; ++t) {
    const Face &f = faces_[t];
    const Complex a = lengths[edgeIndexOf(f[1], f[2])];  // l(jk)
    const Complex b = lengths[edgeIndexOf(f[2], f[0])];  // l(ki)
    const Complex c = lengths[edgeIndexOf(f[0], f[1])];  // l(ij)
    const Complex sa = a * a, sb = b * b, sc = c * c;
    const Eigen::Index row = static_cast<Eigen::Index>(t);

    Eigen::MatrixXcd gram(2, 2);
    gram << sc, 0.5 * (sc + sb - sa), 0.5 * (sc + sb - sa), sb;
    bool ambiguous = false;
    // Heron on the continuation branch: sqrt(det g)/2 continued from the unit
    // equilateral reference along the straight segment in the squared lengths.
    const Complex area = chainhodge::WhitneyMass::volumeOnBranch(gram, chainhodge::Branch::Continuation,
                                                                 &ambiguous);
    if (ambiguous || area == Complex(0.0, 0.0) || !std::isfinite(area.real()) || !std::isfinite(area.imag()))
      throw std::invalid_argument("SimplicialQubit: face " + faceText(f) +
                                  ": the continuation of the Heron root from the real reference meets a root "
                                  "of det G on the segment in the squared lengths (a degenerate triangle on "
                                  "the way, or at the end); no continuous branch (spec section 16)");
    stage.areas(row) = area;

    // The principal branch of acos of the complex cosine of the law of cosines.
    const Complex cosI = (sb + sc - sa) / (2.0 * b * c);
    const Complex cosJ = (sc + sa - sb) / (2.0 * c * a);
    const Complex cosK = (sa + sb - sc) / (2.0 * a * b);
    stage.angles(row, 0) = std::acos(cosI);
    stage.angles(row, 1) = std::acos(cosJ);
    stage.angles(row, 2) = std::acos(cosK);

    // The layout with b sin(alpha_i) := 2 A_t / c on the continuation branch,
    // so that A_t = (1/2) b c sin(alpha_i) holds exactly; the cotangents on
    // the same branch: cot(alpha_v) = (adjacent^2 + adjacent^2 - opposite^2) / (4 A_t).
    const Eigen::Vector2cd pi(Complex(0.0, 0.0), Complex(0.0, 0.0));
    const Eigen::Vector2cd pj(c, Complex(0.0, 0.0));
    const Eigen::Vector2cd pk(b * cosI, 2.0 * area / c);
    stage.layout.row(row) << pi(0), pi(1), pj(0), pj(1), pk(0), pk(1);
    cotangents(row, 0) = (sb + sc - sa) / (4.0 * area);
    cotangents(row, 1) = (sc + sa - sb) / (4.0 * area);
    cotangents(row, 2) = (sa + sb - sc) / (4.0 * area);

    // §7: the barycentric gradients in the local frame.
    const Eigen::Vector2cd gi = rot90c(Eigen::Vector2cd(pk - pj)) / (2.0 * area);
    const Eigen::Vector2cd gj = rot90c(Eigen::Vector2cd(pi - pk)) / (2.0 * area);
    const Eigen::Vector2cd gk = rot90c(Eigen::Vector2cd(pj - pi)) / (2.0 * area);
    stage.gradients.row(row) << gi(0), gi(1), gj(0), gj(1), gk(0), gk(1);
    gradientScale = std::max(gradientScale, gi.norm() + gj.norm() + gk.norm());
    gradientDefect = std::max(gradientDefect, Eigen::Vector2cd(gi + gj + gk).norm());
  }
  if (gradientDefect > 1e-10 * std::max(gradientScale, 1.0))
    throw std::logic_error("SimplicialQubit: the barycentric gradients of a face do not sum to zero");

  // §5 over C: w_e = (cot alpha_e + cot beta_e) / 2 on the continuation branch.
  stage.weights.resize(static_cast<Eigen::Index>(nE));
  for (std::size_t e = 0; e < nE; ++e) {
    Complex cotangentSum(0.0, 0.0);
    for (const auto &[t, slot] : edgeFaces_[e])
      cotangentSum += cotangents(static_cast<Eigen::Index>(t), oppositeVertexSlot(slot));
    stage.weights(static_cast<Eigen::Index>(e)) = 0.5 * cotangentSum;
  }

  // §6 over C on the twisted complex: S^U = [d_1^U; ∂_1^U M_1] with
  // (d_1^U)_{te} = (d_1)_{te} U_{b(t) b(e)} and (∂_1^U M_1)_{ve} = (∂_1)_{ve} U_{v b(e)} w_e;
  // the dual kernel is the same construction under U^{-1}.
  auto stacked = [&](bool dual) {
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Zero(d1_.rows() + nV, static_cast<Eigen::Index>(nE));
    for (std::size_t t = 0; t < nF; ++t) {
      const Face &f = faces_[t];
      const std::uint64_t base = std::min({f[0], f[1], f[2]});
      for (int slot = 0; slot < 3; ++slot) {
        const auto [u, v] = traversal(f, slot);
        const std::size_t e = edgeIndexOf(u, v);
        const std::uint64_t edgeBase = std::min(u, v);
        const Complex carry = trivialConnection_ ? Complex(1.0, 0.0)
                              : dual                ? connection_->link(edgeBase, base)
                                                    : connection_->link(base, edgeBase);
        S(static_cast<Eigen::Index>(t), static_cast<Eigen::Index>(e)) =
            d1_(static_cast<Eigen::Index>(t), static_cast<Eigen::Index>(e)) * carry;
      }
    }
    for (std::size_t e = 0; e < nE; ++e) {
      const auto [x, y] = edges_[e];
      const Complex w = stage.weights(static_cast<Eigen::Index>(e));
      const Complex carryY = trivialConnection_ ? Complex(1.0, 0.0)
                             : dual                ? connection_->link(x, y)
                                                   : connection_->link(y, x);
      S(d1_.rows() + static_cast<Eigen::Index>(x), static_cast<Eigen::Index>(e)) = -w;
      S(d1_.rows() + static_cast<Eigen::Index>(y), static_cast<Eigen::Index>(e)) = carryY * w;
    }
    return S;
  };
  stage.H = complexNullSpace(stacked(false));
  if (stage.H.cols() != 2)
    throw std::runtime_error("SimplicialQubit: dim H = " + std::to_string(stage.H.cols()) +
                             " != 2; the input is not a torus or the weights are degenerate" +
                             (trivialConnection_ ? "" : " (twisted by the link phases)"));
  if (trivialConnection_) {
    stage.Hdual = stage.H;
  } else {
    stage.Hdual = complexNullSpace(stacked(true));
    if (stage.Hdual.cols() != 2)
      throw std::runtime_error("SimplicialQubit: dim H^vee = " + std::to_string(stage.Hdual.cols()) +
                               " != 2 for the kernel twisted by the inverse links");
  }

  // §7, §8 over C: the transpose pairing between the dual kernel and the
  // kernel; J = G^{-1} R^T, its residual exposed.
  stage.G = Eigen::MatrixXcd::Zero(2, 2);
  stage.R = Eigen::MatrixXcd::Zero(2, 2);
  for (std::size_t t = 0; t < nF; ++t) {
    const Complex area = stage.areas(static_cast<Eigen::Index>(t));
    std::array<Eigen::Vector2cd, 2> w, wd;
    for (int a = 0; a < 2; ++a) {
      w[static_cast<std::size_t>(a)] = twistedWhitneyAtBarycenter(t, stage.gradients, stage.H.col(a), false);
      wd[static_cast<std::size_t>(a)] = twistedWhitneyAtBarycenter(t, stage.gradients, stage.Hdual.col(a), true);
    }
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        stage.G(a, b) += area * pairT(wd[static_cast<std::size_t>(a)], w[static_cast<std::size_t>(b)]);
        stage.R(a, b) += area * pairT(rot90c(w[static_cast<std::size_t>(a)]), wd[static_cast<std::size_t>(b)]);
      }
  }
  stage.J = stage.G.inverse() * stage.R.transpose();
  stage.jResidual = (stage.J * stage.J + Eigen::MatrixXcd::Identity(2, 2)).norm();
  return stage;
}

// ============================================================================
// Spec section 9: holomorphic line and period ratio
// ============================================================================

Complex SimplicialQubit::periodOf(const Eigen::VectorXcd &omega, const Cycle &cycle, const Walk &walk) const {
  if (trivialConnection_) {
    // The plain signed sum of §9.
    Complex total(0.0, 0.0);
    for (const auto &[e, sign] : cycle) total += static_cast<double>(sign) * omega(static_cast<Eigen::Index>(e));
    return total;
  }
  // §16: with parallel transport along the cycle's walk from the base point,
  // the cochain handed over in the connection's canonical edge order.
  Eigen::VectorXcd canonical(omega.size());
  for (std::size_t e = 0; e < edges_.size(); ++e)
    canonical(static_cast<Eigen::Index>(canonicalOf_[e])) = omega(static_cast<Eigen::Index>(e));
  return connection_->transportedPeriod(canonical, walk);
}

void SimplicialQubit::buildHolomorphicLine() {
  // The periods of a candidate form over the marking, with the §9 rule for a
  // vanishing |P_A| (the marking (B, -A), reporting -1/tau).
  auto evaluate = [&](const Eigen::VectorXcd &omega, Complex &pA, Complex &pB, bool &swapped) {
    const Complex rawA = periodOf(omega, cycleA_, walkA_);
    const Complex rawB = periodOf(omega, cycleB_, walkB_);
    swapped = std::abs(rawA) <= 1e-12 * (std::abs(rawA) + std::abs(rawB));
    if (swapped) {
      pA = rawB;
      pB = -rawA;
    } else {
      pA = rawA;
      pB = rawB;
    }
    return pB / pA;
  };

  Eigen::VectorXcd omega;
  Complex pA, pB;
  bool swapped = false;
  Complex tau;

  if (real_) {
    // Complexify: the eigenvector of J for the eigenvalue closest to -i.
    Eigen::EigenSolver<Eigen::MatrixXd> eig(realJ_);
    const Eigen::VectorXcd lambda = eig.eigenvalues();
    int branch = 0;
    for (int b = 1; b < lambda.size(); ++b)
      if (std::abs(lambda(b) - Complex(0.0, -1.0)) < std::abs(lambda(branch) - Complex(0.0, -1.0)))
        branch = b;
    Eigen::VectorXcd c = eig.eigenvectors().col(branch);
    omega = H_ * c;  // omega = c[0] h1 + c[1] h2
    tau = evaluate(omega, pA, pB, swapped);
    // Require Im(tau) > 0; otherwise the surface orientation or the eigenvalue
    // branch is flipped: take the conjugate eigenvector and recompute.
    if (tau.imag() < 0.0) {
      c = c.conjugate();
      omega = H_ * c;
      tau = evaluate(omega, pA, pB, swapped);
      warnings_.push_back(
          "Im tau < 0 on the eigenvector nearest -i: the surface orientation or the eigenvalue branch is "
          "flipped; the conjugate eigenvector was taken");
    }
  } else {
    // Spec §16: the eigenline chosen by continuity from the real reference
    // along the straight segment s_e(t) = (1 - t) + t l_e^2 in the squared
    // lengths, with the marking and the links held fixed.
    const std::size_t nE = edges_.size();
    auto lengthsAt = [&](double t) {
      std::vector<Complex> at(nE);
      for (std::size_t e = 0; e < nE; ++e)
        at[e] = principalSqrt((1.0 - t) * Complex(1.0, 0.0) + t * lengths_[e] * lengths_[e]);
      return at;
    };
    struct Eigenlines {
      std::array<Eigen::VectorXcd, 2> coefficients;
      std::array<Eigen::VectorXcd, 2> forms;
      std::array<Complex, 2> eigenvalues;
    };
    auto eigenlinesOf = [&](const Eigen::MatrixXcd &J, const Eigen::MatrixXcd &H) {
      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eig(J);
      Eigenlines lines;
      for (int m = 0; m < 2; ++m) {
        lines.eigenvalues[static_cast<std::size_t>(m)] = eig.eigenvalues()(m);
        lines.coefficients[static_cast<std::size_t>(m)] = eig.eigenvectors().col(m);
        lines.forms[static_cast<std::size_t>(m)] = H * eig.eigenvectors().col(m);
      }
      return lines;
    };

    // At the real reference: the §9 rule selects the line.
    const ComplexStage reference = complexStageAt(std::vector<Complex>(nE, Complex(1.0, 0.0)));
    Eigenlines lines = eigenlinesOf(reference.J, reference.H);
    std::size_t pick = std::abs(lines.eigenvalues[1] - Complex(0.0, -1.0)) <
                               std::abs(lines.eigenvalues[0] - Complex(0.0, -1.0))
                           ? 1
                           : 0;
    {
      Complex rA, rB;
      bool rSwapped = false;
      const Complex tauReference = evaluate(lines.forms[pick], rA, rB, rSwapped);
      if (tauReference.imag() < 0.0) {
        pick = 1 - pick;
        warnings_.push_back(
            "Im tau < 0 at the real reference on the eigenvector nearest -i: the surface orientation or the "
            "eigenvalue branch is flipped; the other eigenline was taken and continued (spec section 16)");
      }
    }
    Eigen::VectorXcd tracked = lines.forms[pick];

    // Track the line by overlap along the segment, halving the step until the
    // overlap with the previous point exceeds 0.99.
    constexpr double kStep = 0.25;
    constexpr double kMinimumStep = 1e-6;
    constexpr double kOverlap = 0.99;
    double t = 0.0, step = kStep;
    while (t < 1.0) {
      const double next = std::min(1.0, t + step);
      const bool last = next >= 1.0;
      Eigenlines candidate = last ? eigenlinesOf(J_, H_) : [&] {
        const ComplexStage stage = complexStageAt(lengthsAt(next));
        return eigenlinesOf(stage.J, stage.H);
      }();
      const std::array<double, 2> overlap = {lineOverlap(tracked, candidate.forms[0]),
                                             lineOverlap(tracked, candidate.forms[1])};
      const std::size_t best = overlap[1] > overlap[0] ? 1 : 0;
      if (overlap[best] < kOverlap) {
        if (step / 2.0 < kMinimumStep)
          throw std::runtime_error(
              "SimplicialQubit: the eigenline cannot be continued from the real reference: at t = " +
              std::to_string(next) + " of the segment in the squared lengths the overlap with the tracked "
              "line is " + std::to_string(overlap[best]) + " below " + std::to_string(kOverlap) +
              " at the minimum step (spec section 16)");
        step /= 2.0;
        continue;
      }
      if (overlap[1 - best] >= overlap[best] - 1e-9)
        throw std::runtime_error(
            "SimplicialQubit: the eigenline cannot be continued from the real reference: at t = " +
            std::to_string(next) + " of the segment in the squared lengths the two eigenlines of J cannot be "
            "told apart (overlaps " + std::to_string(overlap[0]) + ", " + std::to_string(overlap[1]) +
            "); the complex structure is degenerate there (spec section 16)");
      tracked = candidate.forms[best];
      lines = candidate;
      pick = best;
      t = next;
      step = kStep;
    }
    omega = lines.forms[pick];
    tau = evaluate(omega, pA, pB, swapped);
    if (!(tau.imag() > 0.0) && std::isfinite(tau.imag()))
      warnings_.push_back("Im tau = " + std::to_string(tau.imag()) +
                          " <= 0 off the real locus: the state lies in the other hemisphere; the eigenline "
                          "is the continuation of the real reference's, not chosen by Im tau (spec section 16)");
  }

  if (!std::isfinite(tau.real()) || !std::isfinite(tau.imag()))
    throw std::runtime_error("SimplicialQubit: the period ratio is not finite (both periods vanish)");
  if (tau.imag() == 0.0)
    warnings_.push_back("Im tau = 0: the marking is degenerate for this metric (a pinched cycle)");
  if (swapped) warnings_.push_back("|P_A| vanished: the roles of A and B were swapped and -1/tau is reported");

  omega_ = omega;
  periodA_ = pA;
  periodB_ = pB;
  tau_ = tau;
  swapped_ = swapped;

  // Section 10's assertion: |r| == 1 to machine precision (an identity in
  // tau, on and off the real locus).
  const double blochNorm = bloch().norm();
  if (std::abs(blochNorm - 1.0) > 1e-12)
    throw std::logic_error("SimplicialQubit: the Bloch vector is not a unit vector (|r| = " +
                           std::to_string(blochNorm) + ")");
}

// ============================================================================
// Spec section 10: the qubit state
// ============================================================================

Eigen::VectorXcd SimplicialQubit::state() const {
  const double norm = std::sqrt(1.0 + std::norm(tau_));
  Eigen::VectorXcd psi(2);
  psi(0) = Complex(1.0, 0.0) / norm;
  psi(1) = tau_ / norm;
  return psi;
}

Eigen::VectorXd SimplicialQubit::bloch() const {
  const double N = 1.0 + std::norm(tau_);
  Eigen::VectorXd r(3);
  r(0) = 2.0 * tau_.real() / N;
  r(1) = 2.0 * tau_.imag() / N;
  r(2) = (1.0 - std::norm(tau_)) / N;
  return r;
}

Eigen::MatrixXcd SimplicialQubit::densityMatrix() const {
  const Eigen::VectorXd r = bloch();
  Eigen::MatrixXcd rho(2, 2);
  rho(0, 0) = 0.5 * (1.0 + r(2));
  rho(0, 1) = 0.5 * Complex(r(0), -r(1));
  rho(1, 0) = 0.5 * Complex(r(0), r(1));
  rho(1, 1) = 0.5 * (1.0 - r(2));
  return rho;
}

// ============================================================================
// Spec section 11: the two metrics on the state space
// ============================================================================

double SimplicialQubit::fubiniStudyDistance(const SimplicialQubit &q1, const SimplicialQubit &q2) {
  const Complex t1 = q1.tau(), t2 = q2.tau();
  const double overlap =
      std::abs(Complex(1.0, 0.0) + std::conj(t1) * t2) / std::sqrt((1.0 + std::norm(t1)) * (1.0 + std::norm(t2)));
  return std::acos(std::clamp(overlap, 0.0, 1.0));
}

double SimplicialQubit::weilPeterssonDistance(const SimplicialQubit &q1, const SimplicialQubit &q2) {
  const Complex t1 = q1.tau(), t2 = q2.tau();
  if (!(t1.imag() > 0.0) || !(t2.imag() > 0.0))
    throw std::invalid_argument(
        "SimplicialQubit::weilPeterssonDistance: both period ratios must lie in the upper half plane");
  return std::acosh(std::max(1.0 + std::norm(t1 - t2) / (2.0 * t1.imag() * t2.imag()), 1.0));
}

// ============================================================================
// Spec section 12: the flat torus C / (Z + tau Z)
// ============================================================================

SimplicialQubit SimplicialQubit::flatTorus(std::complex<double> tau, int nx, int ny) {
  if (!(tau.imag() > 0.0))
    throw std::invalid_argument("SimplicialQubit::flatTorus: tau must lie in the upper half plane");
  if (nx < 3 || ny < 3)
    throw std::invalid_argument(
        "SimplicialQubit::flatTorus: the grid needs at least 3 vertices per side to be a simplicial complex");

  // The unit cell spanned by 1 and tau; vertex (i, j) at i/nx + tau j/ny; every
  // square split by its diagonal; sides identified.
  const auto vid = [ny](int i, int j) {
    return static_cast<std::uint64_t>(i) * static_cast<std::uint64_t>(ny) + static_cast<std::uint64_t>(j);
  };
  const Complex e1(1.0 / nx, 0.0);
  const Complex e2 = tau / static_cast<double>(ny);
  std::vector<std::uint64_t> vertices(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
  for (std::size_t n = 0; n < vertices.size(); ++n) vertices[n] = n;

  std::map<EdgePair, double> lengthOf;
  auto addEdge = [&](std::uint64_t u, std::uint64_t v, Complex displacement) {
    lengthOf[{std::min(u, v), std::max(u, v)}] = std::abs(displacement);
  };
  std::vector<Face> faces;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j) {
      const int i1 = (i + 1) % nx, j1 = (j + 1) % ny;
      addEdge(vid(i, j), vid(i1, j), e1);
      addEdge(vid(i, j), vid(i, j1), e2);
      addEdge(vid(i, j), vid(i1, j1), e1 + e2);
      // Counterclockwise for Im tau > 0: (0, e1, e1 + e2) and (0, e1 + e2, e2).
      faces.push_back({vid(i, j), vid(i1, j), vid(i1, j1)});
      faces.push_back({vid(i, j), vid(i1, j1), vid(i, j1)});
    }
  std::vector<EdgePair> edges;
  std::vector<Complex> lengths;
  std::map<EdgePair, std::size_t> index;
  for (const auto &[pair, length] : lengthOf) {
    index[pair] = edges.size();
    edges.push_back(pair);
    lengths.push_back(Complex(length, 0.0));
  }

  // Marked cycles: A the row loop along 1 (j = 0), B the column loop along
  // tau (i = 0); a step's sign is +1 when it runs from the smaller id.
  auto step = [&](std::uint64_t from, std::uint64_t to) {
    return CycleStep{index.at({std::min(from, to), std::max(from, to)}), from < to ? 1 : -1};
  };
  Cycle cycleA, cycleB;
  for (int i = 0; i < nx; ++i) cycleA.push_back(step(vid(i, 0), vid((i + 1) % nx, 0)));
  for (int j = 0; j < ny; ++j) cycleB.push_back(step(vid(0, j), vid(0, (j + 1) % ny)));

  return SimplicialQubit(std::move(vertices), std::move(edges), std::move(faces), std::move(lengths),
                         std::move(cycleA), std::move(cycleB));
}

// ============================================================================
// Spec section 5: the optional intrinsic Delaunay edge-flip pass
// ============================================================================

SimplicialQubit SimplicialQubit::intrinsicDelaunay() const {
  if (!real_)
    throw std::invalid_argument(
        "SimplicialQubit::intrinsicDelaunay: the intrinsic Delaunay condition alpha_e + beta_e <= pi is an "
        "inequality on real angles; the pass is defined on the real locus only (real lengths, no phases)");
  std::vector<EdgePair> edges = edges_;
  std::vector<Face> faces = faces_;
  std::vector<double> lengths;
  lengths.reserve(lengths_.size());
  for (const Complex l : lengths_) lengths.push_back(l.real());
  Cycle cycleA = cycleA_, cycleB = cycleB_;
  std::map<EdgePair, std::size_t> index = edgeIndex_;
  std::vector<std::vector<std::pair<std::size_t, int>>> incidence = edgeFaces_;
  std::vector<std::string> skipped;

  auto lengthOf = [&](std::uint64_t u, std::uint64_t v) {
    return lengths[index.at({std::min(u, v), std::max(u, v)})];
  };
  auto faceAngles = [&](const Face &f) {
    return anglesOf(lengthOf(f[1], f[2]), lengthOf(f[2], f[0]), lengthOf(f[0], f[1]));
  };
  auto oppositeAngle = [&](std::size_t t, int slot) {
    return faceAngles(faces[t])[static_cast<std::size_t>(oppositeVertexSlot(slot))];
  };
  auto detach = [&](std::size_t e, std::size_t t) {
    auto &list = incidence[e];
    list.erase(std::remove_if(list.begin(), list.end(),
                              [t](const std::pair<std::size_t, int> &p) { return p.first == t; }),
               list.end());
  };
  auto attach = [&](std::size_t t) {
    for (int slot = 0; slot < 3; ++slot) {
      const auto [u, v] = traversal(faces[t], slot);
      incidence[index.at({std::min(u, v), std::max(u, v)})].push_back({t, slot});
    }
  };

  std::deque<std::size_t> queue;
  std::vector<bool> queued(edges.size(), true);
  for (std::size_t e = 0; e < edges.size(); ++e) queue.push_back(e);
  int flips = 0;
  const std::size_t cap = 100 * edges.size() + 100;
  std::size_t iterations = 0;
  while (!queue.empty()) {
    if (++iterations > cap)
      throw std::runtime_error("SimplicialQubit::intrinsicDelaunay: the flip pass did not terminate");
    const std::size_t e = queue.front();
    queue.pop_front();
    queued[e] = false;
    const auto &pair = incidence[e];
    const double alpha = oppositeAngle(pair[0].first, pair[0].second);
    const double beta = oppositeAngle(pair[1].first, pair[1].second);
    if (!(alpha + beta > kPi + 1e-12)) continue;

    // The two faces: t1 traverses the edge i -> j (its apex k), t2 traverses
    // j -> i (its apex l).
    const auto [u, v] = edges[e];
    std::size_t t1 = pair[0].first, t2 = pair[1].first;
    int slot1 = pair[0].second, slot2 = pair[1].second;
    if (traversal(faces[t1], slot1).first != u) {
      std::swap(t1, t2);
      std::swap(slot1, slot2);
    }
    const std::uint64_t i = u, j = v;
    const std::uint64_t k = faces[t1][static_cast<std::size_t>(oppositeVertexSlot(slot1))];
    const std::uint64_t l = faces[t2][static_cast<std::size_t>(oppositeVertexSlot(slot2))];
    if (k == l || index.count({std::min(k, l), std::max(k, l)}) != 0) {
      skipped.push_back("edge " + edgeText(i, j) + " violates the Delaunay condition but its flip would "
                        "duplicate edge " + edgeText(std::min(k, l), std::max(k, l)) + "; left in place");
      continue;
    }

    // Lay both triangles out in the frame of t1 (spec section 4): p_i = 0,
    // p_j = (c, 0), p_k above the edge, p_l below it; the new diagonal is the
    // distance between the two apexes.
    const double a = lengthOf(j, k), b = lengthOf(k, i), c = lengthOf(i, j);
    const double aPrime = lengthOf(j, l), bPrime = lengthOf(l, i);
    const double cosK = std::clamp((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0);
    const double cosL = std::clamp((bPrime * bPrime + c * c - aPrime * aPrime) / (2.0 * bPrime * c), -1.0, 1.0);
    const Eigen::Vector2d pk(b * cosK, b * std::sqrt(1.0 - cosK * cosK));
    const Eigen::Vector2d pl(bPrime * cosL, -bPrime * std::sqrt(1.0 - cosL * cosL));
    const double newLength = (pk - pl).norm();

    // Replace: faces (i,j,k), (j,i,l) -> (k,i,l), (l,j,k); edge (i,j) -> (k,l).
    for (int slot = 0; slot < 3; ++slot) {
      const auto [x, y] = traversal(faces[t1], slot);
      detach(index.at({std::min(x, y), std::max(x, y)}), t1);
      const auto [p, q] = traversal(faces[t2], slot);
      detach(index.at({std::min(p, q), std::max(p, q)}), t2);
    }
    index.erase({i, j});
    edges[e] = {std::min(k, l), std::max(k, l)};
    lengths[e] = newLength;
    index[edges[e]] = e;
    faces[t1] = {k, i, l};
    faces[t2] = {l, j, k};
    attach(t1);
    attach(t2);
    ++flips;

    // Reroute the marked cycles: a step across (i, j) becomes the path through k.
    auto reroute = [&](Cycle &cycle) {
      Cycle out;
      for (const auto &[edge, sign] : cycle) {
        if (edge != e) {
          out.push_back({edge, sign});
          continue;
        }
        const std::uint64_t from = sign > 0 ? i : j, to = sign > 0 ? j : i;
        out.push_back({index.at({std::min(from, k), std::max(from, k)}), from < k ? 1 : -1});
        out.push_back({index.at({std::min(k, to), std::max(k, to)}), k < to ? 1 : -1});
      }
      cycle = std::move(out);
    };
    reroute(cycleA);
    reroute(cycleB);

    // The four sides of the quadrilateral may have become non-Delaunay.
    for (const auto &[x, y] : {EdgePair{i, k}, EdgePair{k, j}, EdgePair{j, l}, EdgePair{l, i}}) {
      const std::size_t side = index.at({std::min(x, y), std::max(x, y)});
      if (!queued[side]) {
        queued[side] = true;
        queue.push_back(side);
      }
    }
  }

  std::vector<Complex> complexLengths;
  complexLengths.reserve(lengths.size());
  for (const double l : lengths) complexLengths.push_back(Complex(l, 0.0));
  SimplicialQubit result(vertices_, std::move(edges), std::move(faces), std::move(complexLengths),
                         std::move(cycleA), std::move(cycleB), degeneracyThreshold_);
  result.flips_ = flips;
  result.warnings_.insert(result.warnings_.end(), skipped.begin(), skipped.end());
  return result;
}

// ============================================================================
// The period frame (qubit cobordism spec D3) and spec section 13 diagnostics
// ============================================================================

void SimplicialQubit::buildPeriodFrame() {
  // Pi_{ca} = the period of h_a over cycle c, the cycles in force: (A, B), or
  // (B, -A) when section 9 swapped the marking (|P_A| vanished), so that the
  // frame is the basis tau() is a coordinate in. The period map of the
  // harmonic space over a homology basis is an isomorphism (section 2 checked
  // the cycles' independence), so Pi is invertible whenever dim H = 2.
  if (real_) {
    Eigen::Matrix2d Pi;
    for (Eigen::Index a = 0; a < 2; ++a) {
      const Eigen::VectorXcd h = H_.col(a);
      const double pA = periodOf(h, cycleA_, walkA_).real();
      const double pB = periodOf(h, cycleB_, walkB_).real();
      Pi(0, a) = swapped_ ? pB : pA;
      Pi(1, a) = swapped_ ? -pA : pB;
    }
    const double det = Pi.determinant();
    if (!std::isfinite(det) || det == 0.0)
      throw std::runtime_error("SimplicialQubit: the period matrix of the harmonic basis over the marking is "
                               "singular; the marked cycles do not span the torus's homology");
    // The real basis materialized first, so the product is the one the
    // real-length construction evaluates (bit-identical on the real locus).
    const Eigen::MatrixXd H = H_.real();
    const Eigen::MatrixXd F = H * Pi.inverse();
    F_ = F.cast<Complex>();
    return;
  }
  // §16: the transported periods (a common base point), complex.
  Eigen::Matrix2cd Pi;
  for (Eigen::Index a = 0; a < 2; ++a) {
    const Eigen::VectorXcd h = H_.col(a);
    const Complex pA = periodOf(h, cycleA_, walkA_);
    const Complex pB = periodOf(h, cycleB_, walkB_);
    Pi(0, a) = swapped_ ? pB : pA;
    Pi(1, a) = swapped_ ? -pA : pB;
  }
  const Complex det = Pi.determinant();
  if (!std::isfinite(det.real()) || !std::isfinite(det.imag()) || det == Complex(0.0, 0.0))
    throw std::runtime_error("SimplicialQubit: the period matrix of the harmonic basis over the marking is "
                             "singular; the marked cycles do not span the torus's homology");
  F_ = H_ * Pi.inverse();
}

void SimplicialQubit::diagnoseDegeneration() {
  const Eigen::VectorXd magnitudes = weights_.cwiseAbs();
  const double smallest = magnitudes.minCoeff();
  condM1_ = smallest > 0.0 ? magnitudes.maxCoeff() / smallest : std::numeric_limits<double>::infinity();
  Eigen::VectorXd sigma;
  if (real_) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(realG_);
    sigma = svd.singularValues();
  } else {
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(G_);
    sigma = svd.singularValues();
  }
  condG_ = sigma(1) > 0.0 ? sigma(0) / sigma(1) : std::numeric_limits<double>::infinity();
  nearDegenerate_ = condM1_ > degeneracyThreshold_ || condG_ > degeneracyThreshold_;
  if (nearDegenerate_) {
    const long zeroWeights = std::count_if(
        magnitudes.data(), magnitudes.data() + magnitudes.size(),
        [&](double w) { return w <= 1e-12 * std::max(magnitudes.maxCoeff(), 1.0); });
    std::ostringstream w;
    w << "near-degenerate: cond(M1) = " << condM1_ << ", cond(G) = " << condG_ << " against the threshold "
      << degeneracyThreshold_ << " (";
    if (zeroWeights > 0)
      w << zeroWeights << " cotangent weight(s) vanish, alpha_e + beta_e = pi on those edges";
    else
      w << "a cycle is pinching";
    w << "; the state remains valid)";
    warnings_.push_back(w.str());
  }
}

// ============================================================================
// Qubit cobordism spec D2: the derivative of tau and the intersection number
// ============================================================================

Eigen::VectorXcd SimplicialQubit::tauDerivative() const {
  using Matrix = Eigen::MatrixXcd;
  using Vector = Eigen::VectorXcd;
  const std::size_t nE = edges_.size(), nF = faces_.size(), nV = vertices_.size();
  const Eigen::Index NE = static_cast<Eigen::Index>(nE);
  const Eigen::Index NV = static_cast<Eigen::Index>(nV);
  const Complex one(1.0, 0.0), zero(0.0, 0.0);
  auto row = [](std::size_t n) { return static_cast<Eigen::Index>(n); };

  // ---- the links the twisted operators carry (§16): U for the primal
  // operators, U^{-1} for the dual ones; 1 on the trivial connection.
  auto carry = [&](std::uint64_t from, std::uint64_t to, bool dual) -> Complex {
    if (trivialConnection_) return one;
    return dual ? connection_->link(to, from) : connection_->link(from, to);
  };
  std::optional<Connection> inverseConnection;
  if (!trivialConnection_) inverseConnection = connection_->inverse();
  // The period of a cochain over a marked cycle: the plain signed sum on the
  // trivial connection; transported from the base point under U (primal) or
  // U^{-1} (dual) otherwise.
  auto period = [&](const Vector &omega, const Cycle &cycle, const Walk &walk, bool dual) -> Complex {
    if (trivialConnection_ || !dual) return periodOf(omega, cycle, walk);
    Vector canonical(omega.size());
    for (std::size_t e = 0; e < nE; ++e) canonical(row(canonicalOf_[e])) = omega(row(e));
    return inverseConnection->transportedPeriod(canonical, walk);
  };

  // ---- the period frames over the GIVEN marking (A, B): F = H Pi^{-1} on the
  // kernel, Fd = H^vee Pi_d^{-1} on the dual kernel. Both are canonical
  // functions of the lengths, which the null-space bases are not; J_F is
  // independent of the dual basis altogether, and so is its derivative once
  // the dual frame's derivative solves the same differentiated kernel
  // equation (a basis change K(z) of the dual frame drops out of dJ_F).
  auto frameOf = [&](const Matrix &basis, bool dual, const char *name) {
    Eigen::Matrix2cd Pi;
    for (Eigen::Index a = 0; a < 2; ++a) {
      const Vector h = basis.col(a);
      Pi(0, a) = period(h, cycleA_, walkA_, dual);
      Pi(1, a) = period(h, cycleB_, walkB_, dual);
    }
    const Complex det = Pi.determinant();
    if (!std::isfinite(det.real()) || !std::isfinite(det.imag()) || det == zero)
      throw std::runtime_error(std::string("SimplicialQubit::tauDerivative: the period matrix of the ") + name +
                               " over the marking is singular");
    return Matrix(basis * Pi.inverse());
  };
  const Matrix F = frameOf(H_, false, "kernel");
  const Matrix Fd = trivialConnection_ ? F : frameOf(Hdual_, true, "dual kernel");

  // ---- the twisted incidences of §3/§16 as dense matrices: B = ∂_1^U
  // (n_V x n_E, (∂_1^U)_{ve} = (∂_1)_{ve} U_{v b(e)}) and D = d_0^U
  // (n_E x n_V, (d_0^U φ)_e = U_{b(e) y} φ_y - φ_x for the edge x < y, whose
  // base is x), so that L_0^U = B M_1 D is the twisted weighted graph
  // Laplacian and d_1^U d_0^U = 0 under a pure gauge.
  auto incidences = [&](bool dual) {
    Matrix B = Matrix::Zero(NV, NE), D = Matrix::Zero(NE, NV);
    for (std::size_t e = 0; e < nE; ++e) {
      const auto [x, y] = edges_[e];
      B(row(x), row(e)) = -one;
      B(row(y), row(e)) = carry(y, x, dual);
      D(row(e), row(x)) = -one;
      D(row(e), row(y)) = carry(x, y, dual);
    }
    return std::make_pair(B, D);
  };
  const auto [Bp, Dp] = incidences(false);
  const auto [Bd, Dd] = trivialConnection_ ? std::make_pair(Bp, Dp) : incidences(true);
  const Vector weights = weights_;
  // The gauge kernel of L_0^U is one-dimensional (the twisted constants), so
  // pinning vertex 0 leaves a nonsingular system; the pinned solution differs
  // from any other by a twisted constant, which d_0^U annihilates.
  struct PinnedLaplacian {
    Matrix full;
    Eigen::FullPivLU<Matrix> lu;
  };
  auto pinned = [&](const Matrix &B, const Matrix &D, const char *name) {
    PinnedLaplacian L;
    L.full = B * weights.asDiagonal() * D;
    L.lu.compute(L.full.block(1, 1, NV - 1, NV - 1));
    if (!L.lu.isInvertible())
      throw std::runtime_error(std::string("SimplicialQubit::tauDerivative: the twisted Laplacian of the ") + name +
                               " is singular beyond its gauge kernel; the frame has no derivative");
    return L;
  };
  const PinnedLaplacian Lp = pinned(Bp, Dp, "kernel");
  const PinnedLaplacian Ld = trivialConnection_ ? pinned(Bp, Dp, "kernel") : pinned(Bd, Dd, "dual kernel");
  // dF_a = D φ with L_0^U φ = -B dM_1 F_a: the closed, zero-period motion of a
  // period-normalized frame is exact.
  auto frameDerivative = [&](const PinnedLaplacian &L, const Matrix &B, const Matrix &D, const Matrix &frame,
                             const std::vector<std::pair<std::size_t, Complex>> &dWeights) {
    Matrix out(NE, 2);
    for (Eigen::Index a = 0; a < 2; ++a) {
      Vector scaled = Vector::Zero(NE);
      for (const auto &[e, dw] : dWeights) scaled(row(e)) += dw * frame(row(e), a);
      const Vector rhs = -(B * scaled);
      Vector phi = Vector::Zero(NV);
      phi.tail(NV - 1) = L.lu.solve(rhs.tail(NV - 1));
      const double defect = (L.full * phi - rhs).norm();
      if (defect > 1e-8 * std::max(rhs.norm(), 1e-300) && rhs.norm() > 0.0)
        throw std::runtime_error("SimplicialQubit::tauDerivative: the differentiated harmonic condition is "
                                 "inconsistent (defect " + std::to_string(defect) + "): the connection is not "
                                 "a pure gauge on the pinned vertex");
      out.col(a) = D * phi;
    }
    return out;
  };

  // ---- §4, §5, §7 in closed form per face: the derivatives of the Heron
  // area, the layout, the barycentric gradients and the three cotangents
  // with respect to each of the face's three squared lengths, on the same
  // branch as the stored values (A_t is the continued root; c = l(ij) the
  // resident length, dc = dz_c / (2c)). Local index m: 0 = a = l(jk),
  // 1 = b = l(ki), 2 = c = l(ij); edge slot s of §3 (0 = (i,j), 1 = (j,k),
  // 2 = (k,i)) has length index m = (s + 2) % 3, the vertex slot opposite it.
  struct FaceDerivative {
    std::array<std::size_t, 3> edge{};
    std::array<Complex, 3> dA{};
    std::array<std::array<Eigen::Vector2cd, 3>, 3> dg{};    // dg[m][vertex slot]
    std::array<std::array<Complex, 3>, 3> dcot{};           // dcot[m][vertex slot]
  };
  std::vector<FaceDerivative> faceDerivatives(nF);
  for (std::size_t t = 0; t < nF; ++t) {
    const Face &f = faces_[t];
    FaceDerivative &d = faceDerivatives[t];
    d.edge = {edgeIndexOf(f[1], f[2]), edgeIndexOf(f[2], f[0]), edgeIndexOf(f[0], f[1])};
    const Complex a = lengths_[d.edge[0]], b = lengths_[d.edge[1]], c = lengths_[d.edge[2]];
    const Complex sa = a * a, sb = b * b, sc = c * c;
    const Complex A = areas_(row(t));
    const Complex xk = (sb + sc - sa) / (2.0 * c), yk = 2.0 * A / c;
    const Eigen::Vector2cd pj(c, zero), pk(xk, yk);
    const Eigen::Vector2cd gi = rot90c(Eigen::Vector2cd(pk - pj)) / (2.0 * A);
    const Eigen::Vector2cd gj = rot90c(Eigen::Vector2cd(-pk)) / (2.0 * A);
    const Eigen::Vector2cd gk = rot90c(pj) / (2.0 * A);
    const std::array<Complex, 3> cot = {(sb + sc - sa) / (4.0 * A), (sc + sa - sb) / (4.0 * A),
                                        (sa + sb - sc) / (4.0 * A)};
    // 16 A^2 = 2(s_a s_b + s_b s_c + s_c s_a) - (s_a^2 + s_b^2 + s_c^2).
    const std::array<Complex, 3> dA = {(sb + sc - sa) / (16.0 * A), (sa + sc - sb) / (16.0 * A),
                                       (sa + sb - sc) / (16.0 * A)};
    // The sign of each squared length in (adjacent + adjacent - opposite)
    // for the vertex slots i, j, k: opposite is a for i, b for j, c for k.
    const std::array<std::array<double, 3>, 3> sign = {{{-1.0, 1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, -1.0}}};
    for (int m = 0; m < 3; ++m) {
      const Complex dc = (m == 2) ? one / (2.0 * c) : zero;
      const Complex dN = sign[0][static_cast<std::size_t>(m)];  // d(s_b + s_c - s_a)
      const Complex dxk = dN / (2.0 * c) - xk / c * dc;
      const Complex dyk = 2.0 * dA[static_cast<std::size_t>(m)] / c - yk / c * dc;
      const Eigen::Vector2cd dpj(dc, zero), dpk(dxk, dyk);
      const Complex ratio = dA[static_cast<std::size_t>(m)] / A;
      d.dA[static_cast<std::size_t>(m)] = dA[static_cast<std::size_t>(m)];
      d.dg[static_cast<std::size_t>(m)][0] = rot90c(Eigen::Vector2cd(dpk - dpj)) / (2.0 * A) - gi * ratio;
      d.dg[static_cast<std::size_t>(m)][1] = rot90c(Eigen::Vector2cd(-dpk)) / (2.0 * A) - gj * ratio;
      d.dg[static_cast<std::size_t>(m)][2] = rot90c(dpj) / (2.0 * A) - gk * ratio;
      for (int v = 0; v < 3; ++v)
        d.dcot[static_cast<std::size_t>(m)][static_cast<std::size_t>(v)] =
            sign[static_cast<std::size_t>(v)][static_cast<std::size_t>(m)] / (4.0 * A) -
            cot[static_cast<std::size_t>(v)] * ratio;
    }
  }

  // ---- the Whitney interpolant of §7 at a face's barycenter for a given
  // gradient row (the stored one or its derivative), with the carries of §16.
  auto whitney = [&](std::size_t t, const std::array<Eigen::Vector2cd, 3> &g, const Vector &omega, bool dual) {
    const Face &f = faces_[t];
    const std::uint64_t base = std::min({f[0], f[1], f[2]});
    Eigen::Vector2cd w = Eigen::Vector2cd::Zero();
    for (int slot = 0; slot < 3; ++slot) {
      const int from = slot, to = (slot + 1) % 3;
      const auto [u, v] = traversal(f, slot);
      const std::uint64_t edgeBase = std::min(u, v);
      const Complex value = (u < v ? one : -one) * carry(base, edgeBase, dual) * omega(row(edgeIndexOf(u, v)));
      w += value * (g[static_cast<std::size_t>(to)] - g[static_cast<std::size_t>(from)]);
    }
    return Eigen::Vector2cd(w / Complex(3.0, 0.0));
  };
  auto storedGradients = [&](std::size_t t) {
    std::array<Eigen::Vector2cd, 3> g;
    for (int v = 0; v < 3; ++v)
      g[static_cast<std::size_t>(v)] = Eigen::Vector2cd(gradients_(row(t), 2 * v), gradients_(row(t), 2 * v + 1));
    return g;
  };

  // ---- §8 in the frames: G_F, R_F, J_F, and the quadratic tau solves.
  std::vector<std::array<Eigen::Vector2cd, 2>> Wp(nF), Wd(nF);
  Eigen::Matrix2cd G = Eigen::Matrix2cd::Zero(), R = Eigen::Matrix2cd::Zero();
  for (std::size_t t = 0; t < nF; ++t) {
    const auto g = storedGradients(t);
    for (int a = 0; a < 2; ++a) {
      Wp[t][static_cast<std::size_t>(a)] = whitney(t, g, F.col(a), false);
      Wd[t][static_cast<std::size_t>(a)] = whitney(t, g, Fd.col(a), true);
    }
    const Complex A = areas_(row(t));
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        G(a, b) += A * pairT(Wd[t][static_cast<std::size_t>(a)], Wp[t][static_cast<std::size_t>(b)]);
        R(a, b) += A * pairT(rot90c(Wp[t][static_cast<std::size_t>(a)]), Wd[t][static_cast<std::size_t>(b)]);
      }
  }
  const Eigen::Matrix2cd Ginv = G.inverse();
  const Eigen::Matrix2cd J = Ginv * R.transpose();
  // tau over the given marking: tau() is -1/tau_raw when §9 swapped it.
  const Complex tauRaw = swapped_ ? -one / tau_ : tau_;
  const bool sigmaChart = std::abs(tauRaw) > 1.0;
  const Complex sigma = one / tauRaw;
  {
    // (1, tau) is an eigenvector of J_F: the quadratic must vanish.
    const Complex residual = sigmaChart ? J(1, 0) * sigma * sigma + (J(1, 1) - J(0, 0)) * sigma - J(0, 1)
                                        : J(0, 1) * tauRaw * tauRaw + (J(0, 0) - J(1, 1)) * tauRaw - J(1, 0);
    const double scale = J.norm() * (sigmaChart ? (1.0 + std::norm(sigma)) : (1.0 + std::norm(tauRaw)));
    if (!(std::abs(residual) <= 1e-8 * std::max(scale, 1e-300)))
      throw std::logic_error("SimplicialQubit::tauDerivative: the period ratio is not a root of the frame's "
                             "complex structure (residual " + std::to_string(std::abs(residual)) + ")");
  }
  const Complex denominator = sigmaChart ? 2.0 * J(1, 0) * sigma + J(1, 1) - J(0, 0)
                                         : 2.0 * J(0, 1) * tauRaw + J(0, 0) - J(1, 1);
  if (!(std::abs(denominator) > 1e-12 * std::max(J.norm(), 1e-300)))
    throw std::runtime_error("SimplicialQubit::tauDerivative: the two eigenlines of J coincide; tau has no "
                             "derivative there");

  // ---- per edge: dM_1, the frames' motion, dG_F, dR_F, dJ_F, d tau.
  Vector derivative(NE);
  for (std::size_t e = 0; e < nE; ++e) {
    // The cotangent weights that move with z_e: those of the edges of the
    // faces containing e (§5), through dcot of the opposite vertex.
    std::vector<std::pair<std::size_t, Complex>> dWeights;
    struct Adjacent {
      std::size_t face;
      int m;
    };
    std::vector<Adjacent> adjacent;
    for (const auto &[t, slot] : edgeFaces_[e]) {
      const int m = oppositeVertexSlot(slot);  // the slot's length index: a for (j,k), b for (k,i), c for (i,j)
      adjacent.push_back({t, m});
      const FaceDerivative &d = faceDerivatives[t];
      for (int s = 0; s < 3; ++s) {
        const auto [u, v] = traversal(faces_[t], s);
        dWeights.emplace_back(edgeIndexOf(u, v),
                              0.5 * d.dcot[static_cast<std::size_t>(m)][static_cast<std::size_t>(oppositeVertexSlot(s))]);
      }
    }
    const Matrix dF = frameDerivative(Lp, Bp, Dp, F, dWeights);
    const Matrix dFd = trivialConnection_ ? dF : frameDerivative(Ld, Bd, Dd, Fd, dWeights);

    Eigen::Matrix2cd dG = Eigen::Matrix2cd::Zero(), dR = Eigen::Matrix2cd::Zero();
    for (std::size_t t = 0; t < nF; ++t) {
      const auto g = storedGradients(t);
      const Complex A = areas_(row(t));
      std::array<Eigen::Vector2cd, 2> WpdF, WddFd;
      for (int a = 0; a < 2; ++a) {
        WpdF[static_cast<std::size_t>(a)] = whitney(t, g, dF.col(a), false);
        WddFd[static_cast<std::size_t>(a)] = whitney(t, g, dFd.col(a), true);
      }
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) {
          const auto ia = static_cast<std::size_t>(a), ib = static_cast<std::size_t>(b);
          dG(a, b) += A * (pairT(WddFd[ia], Wp[t][ib]) + pairT(Wd[t][ia], WpdF[ib]));
          dR(a, b) += A * (pairT(rot90c(WpdF[ia]), Wd[t][ib]) + pairT(rot90c(Wp[t][ia]), WddFd[ib]));
        }
    }
    for (const Adjacent &adj : adjacent) {
      const std::size_t t = adj.face;
      const FaceDerivative &d = faceDerivatives[t];
      const std::size_t m = static_cast<std::size_t>(adj.m);
      const Complex A = areas_(row(t));
      const Complex dA = d.dA[m];
      std::array<Eigen::Vector2cd, 2> Wp1, Wd1;  // the interpolants through the moving gradients
      for (int a = 0; a < 2; ++a) {
        Wp1[static_cast<std::size_t>(a)] = whitney(t, d.dg[m], F.col(a), false);
        Wd1[static_cast<std::size_t>(a)] = whitney(t, d.dg[m], Fd.col(a), true);
      }
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b) {
          const auto ia = static_cast<std::size_t>(a), ib = static_cast<std::size_t>(b);
          dG(a, b) += dA * pairT(Wd[t][ia], Wp[t][ib]) + A * (pairT(Wd1[ia], Wp[t][ib]) + pairT(Wd[t][ia], Wp1[ib]));
          dR(a, b) += dA * pairT(rot90c(Wp[t][ia]), Wd[t][ib]) +
                      A * (pairT(rot90c(Wp1[ia]), Wd[t][ib]) + pairT(rot90c(Wp[t][ia]), Wd1[ib]));
        }
    }
    const Eigen::Matrix2cd dJ = Ginv * (dR.transpose() - dG * J);
    Complex dTauRaw;
    if (sigmaChart) {
      const Complex dSigma = -(sigma * sigma * dJ(1, 0) + sigma * (dJ(1, 1) - dJ(0, 0)) - dJ(0, 1)) / denominator;
      dTauRaw = -dSigma / (sigma * sigma);
    } else {
      dTauRaw = -(tauRaw * tauRaw * dJ(0, 1) + tauRaw * (dJ(0, 0) - dJ(1, 1)) - dJ(1, 0)) / denominator;
    }
    // tau() = -1/tau_raw when the marking was swapped: d tau = d tau_raw / tau_raw^2.
    derivative(row(e)) = swapped_ ? dTauRaw / (tauRaw * tauRaw) : dTauRaw;
  }
  return derivative;
}

double SimplicialQubit::intersectionNumber() const {
  // The untwisted reference of §16 (unit lengths, no links): the cotangent
  // weights of the unit equilateral triangle are all equal, so the harmonic
  // space is the null space of [d_1; d_0^T]; its period frame f_A, f_B has
  // periods (1, 0) and (0, 1), and the ordered cup product
  // (f_A ∪ f_B)(v_0 v_1 v_2) = f_A(v_0 v_1) f_B(v_1 v_2) on the sorted
  // vertices of every face, summed with the face's orientation sign, is the
  // cohomology pairing <[f_A] ∪ [f_B], [K]> = A · B: an integer, independent
  // of the metric and of the vertex order's cochain-level choices.
  const std::size_t nE = edges_.size(), nF = faces_.size();
  const Eigen::Index NE = static_cast<Eigen::Index>(nE);
  Eigen::MatrixXd S(d1_.rows() + d0_.cols(), NE);
  S.topRows(d1_.rows()) = d1_;
  S.bottomRows(d0_.cols()) = d0_.transpose();
  Eigen::BDCSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullV);
  const Eigen::VectorXd sigma = svd.singularValues();
  const double tolerance = kEps * static_cast<double>(std::max(S.rows(), S.cols())) *
                           (sigma.size() > 0 ? sigma(0) : 0.0);
  Eigen::Index rank = 0;
  for (Eigen::Index n = 0; n < sigma.size(); ++n)
    if (sigma(n) > tolerance) ++rank;
  const Eigen::MatrixXd H = svd.matrixV().rightCols(NE - rank);
  if (H.cols() != 2)
    throw std::runtime_error("SimplicialQubit::intersectionNumber: the reference harmonic space has dimension " +
                             std::to_string(H.cols()) + " != 2");
  auto plainPeriod = [&](const Eigen::VectorXd &h, const Cycle &cycle) {
    double total = 0.0;
    for (const auto &[e, sign] : cycle) total += static_cast<double>(sign) * h(static_cast<Eigen::Index>(e));
    return total;
  };
  Eigen::Matrix2d Pi;
  for (Eigen::Index a = 0; a < 2; ++a) {
    Pi(0, a) = plainPeriod(H.col(a), cycleA_);
    Pi(1, a) = plainPeriod(H.col(a), cycleB_);
  }
  if (Pi.determinant() == 0.0)
    throw std::runtime_error("SimplicialQubit::intersectionNumber: the reference period matrix is singular");
  const Eigen::MatrixXd F = H * Pi.inverse();
  double total = 0.0;
  for (std::size_t t = 0; t < nF; ++t) {
    const Face &f = faces_[t];
    std::array<std::uint64_t, 3> sorted = f;
    std::sort(sorted.begin(), sorted.end());
    // The face's stored cyclic order against the sorted one: a rotation of
    // (v0, v1, v2) is positively oriented, a rotation of (v0, v2, v1) is not.
    const bool positive = (f[0] == sorted[0] && f[1] == sorted[1]) || (f[0] == sorted[1] && f[1] == sorted[2]) ||
                          (f[0] == sorted[2] && f[1] == sorted[0]);
    const double epsilon = positive ? 1.0 : -1.0;
    const Eigen::Index e01 = static_cast<Eigen::Index>(edgeIndexOf(sorted[0], sorted[1]));
    const Eigen::Index e12 = static_cast<Eigen::Index>(edgeIndexOf(sorted[1], sorted[2]));
    total += epsilon * F(e01, 0) * F(e12, 1);
  }
  return kCupSign * total;
}

}  // namespace tessera::observables
