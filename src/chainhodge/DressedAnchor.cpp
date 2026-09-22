// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/DressedAnchor.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace tessera::chainhodge {

namespace {

using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

/// Every canonical edge of the complex, keyed by its ascending vertex pair and
/// valued by its canonical \f$ C_1 \f$ index.
std::map<EdgeKey, int> edgeIndexMap(const cobordism::ChainComplex &K) {
  std::map<EdgeKey, int> index;
  const auto edges = K.kSimplexVertices(1);
  for (std::size_t j = 0; j < edges.size(); ++j)
    index[{edges[j][0], edges[j][1]}] = static_cast<int>(j);
  return index;
}

/// The row subsets an exterior power of rank `r` selects from three rows, in
/// lexicographic order.
std::vector<std::vector<int>> rowSubsets(int r) {
  std::vector<std::vector<int>> out;
  for (int a = 0; a < 3; ++a) {
    if (r == 1) {
      out.push_back({a});
      continue;
    }
    for (int b = a + 1; b < 3; ++b) {
      if (r == 2) {
        out.push_back({a, b});
        continue;
      }
      for (int c = b + 1; c < 3; ++c) out.push_back({a, b, c});
    }
  }
  return out;
}

/// One step of the SplitMix64 sequence, the reproducible source of the
/// verification gauge.
std::uint64_t splitmix64(std::uint64_t &state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double unitInterval(std::uint64_t hashed) {
  return static_cast<double>(hashed >> 11) / 9007199254740992.0;
}

/// The numerical zero threshold a read judges its own coordinates against.
double zeroThreshold(const DressedAnchorRead &read) {
  return read.tolerance * (read.coordinateScale > 0.0 ? read.coordinateScale : 1.0);
}

void requireSameAtlas(const DressedAnchorRead &a, const DressedAnchorRead &b, const char *who) {
  if (a.faceIndices != b.faceIndices || a.bandRank != b.bandRank)
    throw std::invalid_argument(std::string(who) +
                                ": the two reads must cover the same atlas of faces at the same "
                                "band rank");
}

}  // namespace

// --------------------------------------------------------------------------
// DeclaredPaths
// --------------------------------------------------------------------------

DeclaredPaths DeclaredPaths::breadthFirst(const cobordism::ChainComplex &K,
                                          std::uint64_t basePoint,
                                          const std::vector<std::uint64_t> &support) {
  const auto edges = K.kSimplexVertices(1);
  if (edges.empty())
    throw std::invalid_argument("DeclaredPaths::breadthFirst: the complex has no edges, so no "
                                "walk can be declared");
  std::set<std::uint64_t> vertices;
  for (const auto &vertex : K.kSimplexVertices(0)) vertices.insert(vertex[0]);
  if (vertices.find(basePoint) == vertices.end())
    throw std::invalid_argument("DeclaredPaths::breadthFirst: the base vertex " +
                                std::to_string(basePoint) + " is not a vertex of the complex");
  std::set<std::uint64_t> allowed(support.begin(), support.end());
  if (!allowed.empty() && allowed.find(basePoint) == allowed.end())
    throw std::invalid_argument("DeclaredPaths::breadthFirst: the declared support does not "
                                "contain the base vertex " +
                                std::to_string(basePoint));
  const auto inSupport = [&](std::uint64_t v) {
    return allowed.empty() || allowed.find(v) != allowed.end();
  };

  std::map<std::uint64_t, std::set<std::uint64_t>> adjacency;
  for (const auto &edge : edges) {
    if (!inSupport(edge[0]) || !inSupport(edge[1])) continue;
    adjacency[edge[0]].insert(edge[1]);
    adjacency[edge[1]].insert(edge[0]);
  }

  DeclaredPaths paths;
  paths.basePoint_ = basePoint;
  paths.derivedFromWalks_ = true;
  paths.walks_[basePoint] = {basePoint};
  std::deque<std::uint64_t> queue{basePoint};
  while (!queue.empty()) {
    const std::uint64_t current = queue.front();
    queue.pop_front();
    const auto neighbours = adjacency.find(current);
    if (neighbours == adjacency.end()) continue;
    // The neighbours come out of a std::set in ascending id order, so the
    // parent a vertex is first reached from is fixed by the ids alone.
    for (const std::uint64_t next : neighbours->second) {
      if (paths.walks_.find(next) != paths.walks_.end()) continue;
      std::vector<std::uint64_t> walk = paths.walks_.at(current);
      walk.push_back(next);
      paths.walks_[next] = std::move(walk);
      queue.push_back(next);
    }
  }
  return paths;
}

DeclaredPaths DeclaredPaths::fromWalks(
    const cobordism::ChainComplex &K, std::uint64_t basePoint,
    const std::map<std::uint64_t, std::vector<std::uint64_t>> &walks) {
  const std::map<EdgeKey, int> index = edgeIndexMap(K);
  for (const auto &[vertex, walk] : walks) {
    if (walk.empty() || walk.front() != basePoint)
      throw std::invalid_argument("DeclaredPaths::fromWalks: the walk to vertex " +
                                  std::to_string(vertex) + " must start at the base vertex " +
                                  std::to_string(basePoint));
    if (walk.back() != vertex)
      throw std::invalid_argument("DeclaredPaths::fromWalks: the walk keyed by vertex " +
                                  std::to_string(vertex) + " must end at that vertex");
    for (std::size_t i = 0; i + 1 < walk.size(); ++i) {
      const EdgeKey key{std::min(walk[i], walk[i + 1]), std::max(walk[i], walk[i + 1])};
      if (walk[i] == walk[i + 1] || index.find(key) == index.end())
        throw std::invalid_argument("DeclaredPaths::fromWalks: the walk to vertex " +
                                    std::to_string(vertex) + " steps from " +
                                    std::to_string(walk[i]) + " to " +
                                    std::to_string(walk[i + 1]) +
                                    ", which is not an edge of the complex");
    }
  }
  DeclaredPaths paths;
  paths.basePoint_ = basePoint;
  paths.derivedFromWalks_ = true;
  paths.walks_ = walks;
  paths.walks_[basePoint] = {basePoint};
  return paths;
}

DeclaredPaths DeclaredPaths::declaredTransports(std::uint64_t basePoint,
                                                const std::map<std::uint64_t, Complex> &transports) {
  DeclaredPaths paths;
  paths.basePoint_ = basePoint;
  paths.derivedFromWalks_ = false;
  paths.transports_ = transports;
  return paths;
}

bool DeclaredPaths::reaches(std::uint64_t v) const {
  if (v == basePoint_) return true;
  if (derivedFromWalks_) return walks_.find(v) != walks_.end();
  return transports_.find(v) != transports_.end();
}

Complex DeclaredPaths::transport(const Connection &U, std::uint64_t v) const {
  if (derivedFromWalks_) {
    const auto it = walks_.find(v);
    if (it == walks_.end()) {
      if (v == basePoint_) return Complex(1.0, 0.0);
      throw std::invalid_argument("DeclaredPaths::transport: the path rule declares no walk to "
                                  "vertex " +
                                  std::to_string(v));
    }
    Complex product(1.0, 0.0);
    for (std::size_t i = 0; i + 1 < it->second.size(); ++i)
      product *= U.link(it->second[i], it->second[i + 1]);
    return product;
  }
  const auto it = transports_.find(v);
  if (it != transports_.end()) return it->second;
  if (v == basePoint_) return Complex(1.0, 0.0);
  throw std::invalid_argument("DeclaredPaths::transport: the declared transport table has no "
                              "entry for vertex " +
                              std::to_string(v));
}

// --------------------------------------------------------------------------
// The restriction
// --------------------------------------------------------------------------

FaceRestriction DressedAnchor::faceRestriction(const cobordism::ChainComplex &K,
                                               const Connection &U, const DeclaredPaths &paths,
                                               std::size_t faceIndex) {
  if (K.dimension() < 2)
    throw std::invalid_argument("DressedAnchor: the complex has no triangles");
  const auto triangles = K.kSimplexVertices(2);
  if (faceIndex >= triangles.size())
    throw std::invalid_argument("DressedAnchor: the triangle index " + std::to_string(faceIndex) +
                                " is out of range");
  const auto &t = triangles[faceIndex];
  const std::map<EdgeKey, int> index = edgeIndexMap(K);

  // The cyclic order of the oriented triangle (v0 -> v1 -> v2 -> v0), with the
  // incidence signs of d[v0,v1,v2] = [v1,v2] - [v0,v2] + [v0,v1].
  const std::vector<EdgeKey> ordered{{t[0], t[1]}, {t[1], t[2]}, {t[0], t[2]}};
  const std::vector<int> signs{+1, +1, -1};

  FaceRestriction out;
  out.faceIndex = faceIndex;
  out.incidenceSigns = signs;
  for (int i = 0; i < 3; ++i) {
    const auto it = index.find(ordered[static_cast<std::size_t>(i)]);
    if (it == index.end())
      throw std::invalid_argument("DressedAnchor: a triangle edge is missing from C_1");
    out.edgeIndices.push_back(it->second);
    const std::uint64_t base = ordered[static_cast<std::size_t>(i)].first;  // b(e) = min e
    out.basePoints.push_back(base);
    if (!paths.reaches(base))
      throw std::invalid_argument("DressedAnchor: the declared path rule does not reach vertex " +
                                  std::to_string(base) + ", a base vertex of triangle " +
                                  std::to_string(faceIndex) + ", so no restriction to it exists");
    const Complex transported = paths.transport(U, base);
    out.transports.push_back(transported);
    out.factors.push_back(static_cast<double>(signs[static_cast<std::size_t>(i)]) * transported);
  }
  return out;
}

Eigen::MatrixXcd DressedAnchor::restriction(const cobordism::ChainComplex &K, const Connection &U,
                                            const DeclaredPaths &paths, std::size_t faceIndex) {
  const FaceRestriction face = faceRestriction(K, U, paths, faceIndex);
  Eigen::MatrixXcd out =
      Eigen::MatrixXcd::Zero(3, static_cast<Eigen::Index>(K.numSimplices(1)));
  for (int i = 0; i < 3; ++i)
    out(i, face.edgeIndices[static_cast<std::size_t>(i)]) = face.factors[static_cast<std::size_t>(i)];
  return out;
}

Eigen::MatrixXcd DressedAnchor::restrictedFrame(const cobordism::ChainComplex &K,
                                                const Connection &U, const DeclaredPaths &paths,
                                                std::size_t faceIndex,
                                                const Eigen::MatrixXcd &Phi) {
  if (Phi.rows() != static_cast<Eigen::Index>(K.numSimplices(1)))
    throw std::invalid_argument("DressedAnchor: the chain frame must have one row per 1-simplex "
                                "of the complex");
  const FaceRestriction face = faceRestriction(K, U, paths, faceIndex);
  Eigen::MatrixXcd out(3, Phi.cols());
  for (int i = 0; i < 3; ++i)
    out.row(i) = face.factors[static_cast<std::size_t>(i)] *
                 Phi.row(face.edgeIndices[static_cast<std::size_t>(i)]);
  return out;
}

Eigen::Matrix3cd DressedAnchor::twistedCoboundaryBlock(const cobordism::ChainComplex &K,
                                                       const Connection &U,
                                                       std::size_t faceIndex) {
  if (K.dimension() < 2)
    throw std::invalid_argument("DressedAnchor: the complex has no triangles");
  const auto triangles = K.kSimplexVertices(2);
  if (faceIndex >= triangles.size())
    throw std::invalid_argument("DressedAnchor: the triangle index " + std::to_string(faceIndex) +
                                " is out of range");
  const auto &t = triangles[faceIndex];
  const std::uint64_t x = t[0];
  const std::uint64_t y = t[1];
  const std::uint64_t z = t[2];
  // Row e of the twisted coboundary of a vertex potential, in the convention
  // (delta_0^U f)_{ab} = U_{ab} f_b - f_a, read on the oriented edge with its
  // own tail: rows (x,y), (y,z), (z,x); columns x, y, z.
  Eigen::Matrix3cd out = Eigen::Matrix3cd::Zero();
  out(0, 0) = Complex(-1.0, 0.0);
  out(0, 1) = U.link(x, y);
  out(1, 1) = Complex(-1.0, 0.0);
  out(1, 2) = U.link(y, z);
  out(2, 2) = Complex(-1.0, 0.0);
  out(2, 0) = U.link(z, x);
  return out;
}

// --------------------------------------------------------------------------
// The coordinates
// --------------------------------------------------------------------------

std::vector<Complex> DressedAnchor::exteriorPower(const Eigen::MatrixXcd &A) {
  if (A.rows() != 3)
    throw std::invalid_argument("DressedAnchor::exteriorPower: the matrix must have three rows, "
                                "one per boundary edge of a triangle");
  const int r = static_cast<int>(A.cols());
  if (r < 1 || r > 3)
    throw std::invalid_argument("DressedAnchor::exteriorPower: the band's rank must be one, two "
                                "or three");
  std::vector<Complex> out;
  for (const auto &subset : rowSubsets(r)) {
    Eigen::MatrixXcd minor(r, r);
    for (int i = 0; i < r; ++i) minor.row(i) = A.row(subset[static_cast<std::size_t>(i)]);
    out.push_back(minor.determinant());
  }
  return out;
}

std::vector<Complex> DressedAnchor::dressedCoordinates(const cobordism::ChainComplex &K,
                                                       const Connection &U,
                                                       const DeclaredPaths &paths,
                                                       std::size_t faceIndex,
                                                       const Eigen::MatrixXcd &Phi) {
  return exteriorPower(restrictedFrame(K, U, paths, faceIndex, Phi));
}

Complex DressedAnchor::dressedCoordinate(const cobordism::ChainComplex &K, const Connection &U,
                                         const DeclaredPaths &paths, std::size_t faceIndex,
                                         const Eigen::MatrixXcd &Phi) {
  if (Phi.cols() != 3)
    throw std::invalid_argument("DressedAnchor::dressedCoordinate: Delta_tau is the determinant "
                                "of a rank-three band; use dressedCoordinates for a band of "
                                "lower rank");
  return dressedCoordinates(K, U, paths, faceIndex, Phi).front();
}

std::vector<std::size_t> DressedAnchor::anchorableFaces(const cobordism::ChainComplex &K,
                                                        const DeclaredPaths &paths) {
  std::vector<std::size_t> out;
  if (K.dimension() < 2) return out;
  const auto triangles = K.kSimplexVertices(2);
  for (std::size_t f = 0; f < triangles.size(); ++f) {
    const auto &t = triangles[f];
    // The base vertices of the triangle's three edges are v0, v1 and v0.
    if (paths.reaches(t[0]) && paths.reaches(t[1])) out.push_back(f);
  }
  return out;
}

// --------------------------------------------------------------------------
// The certificate
// --------------------------------------------------------------------------

std::map<std::uint64_t, Complex> DressedAnchor::verificationGauge(const cobordism::ChainComplex &K,
                                                                  std::uint64_t seed) {
  std::map<std::uint64_t, Complex> gauge;
  for (const auto &vertex : K.kSimplexVertices(0)) {
    std::uint64_t state = seed ^ (vertex[0] * 0x2545F4914F6CDD1DULL);
    const double re = 0.5 + unitInterval(splitmix64(state));
    const double im = 0.25 + unitInterval(splitmix64(state));
    gauge[vertex[0]] = Complex(re, im);
  }
  return gauge;
}

double DressedAnchor::covarianceResidual(const cobordism::ChainComplex &K, const Connection &U,
                                         const DeclaredPaths &paths,
                                         const std::vector<std::size_t> &faceIndices,
                                         const std::map<std::uint64_t, Complex> &gauge) {
  const Connection gauged = U.gauge(gauge);
  const std::uint64_t p = paths.basePoint();
  const auto at = [&](std::uint64_t v) {
    const auto it = gauge.find(v);
    if (it == gauge.end())
      throw std::invalid_argument("DressedAnchor::covarianceResidual: the verification gauge "
                                  "does not name vertex " +
                                  std::to_string(v));
    return it->second;
  };
  const Complex gp = at(p);
  double worst = 0.0;
  for (const std::size_t faceIndex : faceIndices) {
    const FaceRestriction plain = faceRestriction(K, U, paths, faceIndex);
    const FaceRestriction dressed = faceRestriction(K, gauged, paths, faceIndex);
    for (int i = 0; i < 3; ++i) {
      const std::size_t slot = static_cast<std::size_t>(i);
      // rho_1(g) multiplies the column of edge e by g_{b(e)}^{-1}.
      const Complex left = dressed.factors[slot] / at(plain.basePoints[slot]);
      const Complex right = plain.factors[slot] / gp;
      worst = std::max(worst, std::abs(left - right) / std::max(1.0, std::abs(right)));
    }
  }
  return worst;
}

DressedAnchorRead DressedAnchor::profile(const cobordism::ChainComplex &K, const Connection &U,
                                         const DeclaredPaths &paths,
                                         const std::vector<std::size_t> &faceIndices,
                                         const Eigen::MatrixXcd &Phi, double tolerance,
                                         std::uint64_t gaugeSeed) {
  if (Phi.rows() != static_cast<Eigen::Index>(K.numSimplices(1)))
    throw std::invalid_argument("DressedAnchor::profile: the chain frame must have one row per "
                                "1-simplex of the complex");
  const int rank = static_cast<int>(Phi.cols());
  if (rank < 1 || rank > 3)
    throw std::invalid_argument("DressedAnchor::profile: the base band's rank must be one, two "
                                "or three, since a triangle has three boundary edges");

  DressedAnchorRead read;
  read.basePoint = paths.basePoint();
  read.bandRank = rank;
  read.faceIndices = faceIndices;
  read.tolerance = tolerance;
  if (faceIndices.empty()) {
    read.failedCertificates.emplace_back("empty-anchor-atlas");
    return read;
  }

  const std::vector<std::vector<int>> subsets = rowSubsets(rank);
  std::vector<std::vector<double>> bounds;
  bounds.reserve(faceIndices.size());
  for (const std::size_t faceIndex : faceIndices) {
    const Eigen::MatrixXcd restricted = restrictedFrame(K, U, paths, faceIndex, Phi);
    read.coordinates.push_back(exteriorPower(restricted));
    std::vector<double> rowNorms(3, 0.0);
    for (int i = 0; i < 3; ++i) rowNorms[static_cast<std::size_t>(i)] = restricted.row(i).norm();
    std::vector<double> faceBounds;
    faceBounds.reserve(subsets.size());
    for (const auto &subset : subsets) {
      double bound = 1.0;
      for (const int row : subset) bound *= rowNorms[static_cast<std::size_t>(row)];
      faceBounds.push_back(bound);
      read.coordinateScale = std::max(read.coordinateScale, bound);
    }
    bounds.push_back(std::move(faceBounds));
  }

  const double threshold = zeroThreshold(read);
  for (const auto &faceCoordinates : read.coordinates) {
    bool anchors = false;
    for (const Complex &value : faceCoordinates)
      if (std::abs(value) > threshold) anchors = true;
    if (anchors) ++read.anchoringFaces;
  }
  if (read.anchoringFaces == 0) read.failedCertificates.emplace_back("anchor-profile-identically-zero");

  read.covarianceResidual =
      covarianceResidual(K, U, paths, faceIndices, verificationGauge(K, gaugeSeed));
  if (!(read.covarianceResidual <= tolerance))
    read.failedCertificates.emplace_back("connection-dressed-covariance");

  read.transitionCocycleResidual = transitionCocycleResidual(read, 0);
  if (!(read.transitionCocycleResidual <= tolerance))
    read.failedCertificates.emplace_back("transition-cocycle");

  read.anchored = read.failedCertificates.empty();
  return read;
}

DressedAnchorRead DressedAnchor::withInvariantCoordinates(const DressedAnchorRead &read,
                                                          const CovariantChainHodge &cov,
                                                          const Eigen::MatrixXcd &Zdual,
                                                          const Eigen::MatrixXcd &Z) {
  DressedAnchorRead out = read;
  out.invariantCoordinates.clear();
  out.invariantCoordinates.reserve(read.faceIndices.size());
  for (const std::size_t faceIndex : read.faceIndices)
    out.invariantCoordinates.push_back(FaceAnchor::anchorCoordinate(cov, faceIndex, Zdual, Z));
  return out;
}

// --------------------------------------------------------------------------
// The transitions
// --------------------------------------------------------------------------

Complex DressedAnchor::faceTransition(const DressedAnchorRead &read, std::size_t faceSlot,
                                      std::size_t otherFaceSlot, std::size_t coordinate) {
  if (faceSlot >= read.coordinates.size() || otherFaceSlot >= read.coordinates.size())
    throw std::invalid_argument("DressedAnchor::faceTransition: the face slot is out of range");
  if (coordinate >= read.coordinates[faceSlot].size())
    throw std::invalid_argument("DressedAnchor::faceTransition: the exterior-power coordinate "
                                "index is out of range");
  const Complex denominator = read.coordinates[otherFaceSlot][coordinate];
  if (std::abs(denominator) <= zeroThreshold(read))
    throw std::runtime_error("DressedAnchor::faceTransition: the second face's chart is empty at "
                             "that coordinate, so the two charts do not overlap there");
  return read.coordinates[faceSlot][coordinate] / denominator;
}

Complex DressedAnchor::basePointTransition(const DressedAnchorRead &read,
                                           const DressedAnchorRead &other, std::size_t faceSlot,
                                           std::size_t coordinate) {
  requireSameAtlas(read, other, "DressedAnchor::basePointTransition");
  if (faceSlot >= read.coordinates.size())
    throw std::invalid_argument("DressedAnchor::basePointTransition: the face slot is out of "
                                "range");
  if (coordinate >= read.coordinates[faceSlot].size())
    throw std::invalid_argument("DressedAnchor::basePointTransition: the exterior-power "
                                "coordinate index is out of range");
  const Complex denominator = other.coordinates[faceSlot][coordinate];
  if (std::abs(denominator) <= zeroThreshold(other))
    throw std::runtime_error("DressedAnchor::basePointTransition: the other base vertex's chart "
                             "is empty on that face");
  return read.coordinates[faceSlot][coordinate] / denominator;
}

double DressedAnchor::transitionCocycleResidual(const DressedAnchorRead &read,
                                                std::size_t coordinate) {
  const double threshold = zeroThreshold(read);
  std::vector<std::size_t> live;
  for (std::size_t slot = 0; slot < read.coordinates.size(); ++slot) {
    if (coordinate >= read.coordinates[slot].size()) continue;
    if (std::abs(read.coordinates[slot][coordinate]) > threshold) live.push_back(slot);
  }
  if (live.size() < 3) return 0.0;
  // The identity is algebraic in the ratios, so consecutive triples cover it:
  // testing every triple would be cubic in the size of the atlas and would
  // report the same number.
  double worst = 0.0;
  for (std::size_t i = 0; i + 2 < live.size(); ++i) {
    const Complex ab = read.coordinates[live[i]][coordinate] / read.coordinates[live[i + 1]][coordinate];
    const Complex bc =
        read.coordinates[live[i + 1]][coordinate] / read.coordinates[live[i + 2]][coordinate];
    const Complex ac = read.coordinates[live[i]][coordinate] / read.coordinates[live[i + 2]][coordinate];
    worst = std::max(worst, std::abs(ab * bc - ac) / std::max(1.0, std::abs(ac)));
  }
  return worst;
}

double DressedAnchor::projectiveDistance(const DressedAnchorRead &a, const DressedAnchorRead &b) {
  requireSameAtlas(a, b, "DressedAnchor::projectiveDistance");
  Complex inner(0.0, 0.0);
  double normA = 0.0;
  double normB = 0.0;
  for (std::size_t slot = 0; slot < a.coordinates.size(); ++slot)
    for (std::size_t k = 0; k < a.coordinates[slot].size(); ++k) {
      const Complex left = a.coordinates[slot][k];
      const Complex right = b.coordinates[slot][k];
      inner += std::conj(left) * right;
      normA += std::norm(left);
      normB += std::norm(right);
    }
  if (normA <= 0.0 || normB <= 0.0)
    throw std::runtime_error("DressedAnchor::projectiveDistance: an identically zero profile is "
                             "not a point of a projective space");
  const double cosine = std::norm(inner) / (normA * normB);
  return std::sqrt(std::max(0.0, 1.0 - cosine));
}

}  // namespace tessera::chainhodge
