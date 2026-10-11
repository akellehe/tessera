// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): the inverse
// eigenvector problem itself: construction, the captured edge order and
// boundary, the residual and Rayleigh quotient, and the edge weights and
// phases. One of the translation units that define the class's members by
// topic.

#include "cobordism/EigenstateSynthesis.h"
#include "quantum/ChoiJamiolkowski.h"

#include <numeric>
#include <limits>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"

#ifdef TESSERA_CUDA
#include "cuda/eigenstate_cuda.h"
#endif

namespace tessera::cobordism {

using cd = std::complex<double>;

EigenstateSynthesis::EigenstateSynthesis(std::shared_ptr<Spacetime> st, int k,
                                         HodgeLaplacian::MetricSource metricSource)
    : st_(st), k_(k), metricSource_(metricSource),
      laplacian_(std::move(st), HodgeLaplacian::defaultWeightConvention(), metricSource) {
  if (k_ < 0)
    throw std::runtime_error(
        "EigenstateSynthesis: degree k must be non-negative; got k=" +
        std::to_string(k_));
  if (!st_) return;
  capture();
  classifyBoundary();
}

void EigenstateSynthesis::capture() {
  order_ = 0;
  edges_.clear();
  cellOrdering_.clear();
  if (!st_) return;

  // The psi ordering: at k=0 the sorted-id vertex set (one component per vertex),
  // matching HodgeLaplacian's k=0 indexing; at k>=1 the canonical ChainComplex
  // k-cell column order, matching the metric L_k the operator assembles. order_
  // is the operator dimension N either way.
  std::unordered_set<std::uint64_t> idset;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) idset.insert(v->getId());

  if (k_ == 0) {
    std::vector<std::uint64_t> ids(idset.begin(), idset.end());
    std::sort(ids.begin(), ids.end());
    cellOrdering_.reserve(ids.size());
    for (const std::uint64_t id : ids) cellOrdering_.push_back({id});
  } else {
    cellOrdering_ = ChainComplex::fromSpacetime(*st_).kSimplexVertices(k_);
  }
  order_ = cellOrdering_.size();

  // Stable edge order: the tunable edges in EdgeList order — those carrying
  // weight in L = D - A (both endpoints present in the vertex set, not a
  // self-loop). This is HodgeLaplacian::assemble's edge filter, so the exposed
  // {w_ij, theta_ij} are the parameters the Laplacian reads.
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const auto s = e->getSource();
    const auto t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    if (s->getId() == t->getId()) continue;
    if (idset.find(s->getId()) == idset.end() ||
        idset.find(t->getId()) == idset.end())
      continue;
    edges_.push_back(e);
  }
}

void EigenstateSynthesis::classifyBoundary() {
  interiorEdgeIdx_.clear();
  boundaryEdgeIdx_.clear();
  boundaryVertexIdsSorted_.clear();
  interiorVertexCount_ = 0;
  if (!st_) return;

  // Top cells have d+1 vertices (d = metric dimension), matching the
  // Spacetime's own top-simplex bookkeeping (getRandomTopSimplex) so this
  // partition agrees with the pre-geometric Pachner moves growInterior reuses.
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;

  // ∂W: codim-1 faces (a top cell with one vertex dropped) belonging to exactly
  // one top cell. An edge sits on the boundary only once codim-1 faces are at
  // least edges themselves (topVerts >= 3); below that there is no boundary and
  // every tunable edge is interior.
  std::set<std::pair<std::uint64_t, std::uint64_t>> boundaryEdgeKeys;
  std::unordered_set<std::uint64_t> boundaryVertexIds;
  if (topVerts >= 3) {
    std::map<std::vector<std::uint64_t>, int> facetCount;
    for (const auto s : st_->getSimplices()) {
      if (s == nullptr) continue;
      if (s->size() != topVerts) continue;
      std::vector<std::uint64_t> ids;
      ids.reserve(topVerts);
      for (const auto v : s->getVertices())
        if (v != nullptr) ids.push_back(v->getId());
      if (ids.size() != topVerts) continue;
      std::sort(ids.begin(), ids.end());
      for (std::size_t skip = 0; skip < ids.size(); ++skip) {
        std::vector<std::uint64_t> facet;
        facet.reserve(ids.size() - 1);
        for (std::size_t i = 0; i < ids.size(); ++i)
          if (i != skip) facet.push_back(ids[i]);
        ++facetCount[facet];
      }
    }
    for (const auto &[facet, count] : facetCount) {
      if (count != 1) continue;  // interior facet (shared by two top cells)
      for (const std::uint64_t id : facet) boundaryVertexIds.insert(id);
      // facet is sorted ascending, so (facet[i], facet[j]) for i<j is ordered.
      for (std::size_t i = 0; i + 1 < facet.size(); ++i)
        for (std::size_t j = i + 1; j < facet.size(); ++j)
          boundaryEdgeKeys.insert({facet[i], facet[j]});
    }
  }

  // An edge is on ∂W iff some boundary facet contains both endpoints; otherwise
  // it is interior (free).
  for (std::size_t e = 0; e < edges_.size(); ++e) {
    const std::uint64_t a = edges_[e]->getSource()->getId();
    const std::uint64_t b = edges_[e]->getTarget()->getId();
    const std::pair<std::uint64_t, std::uint64_t> key =
        a < b ? std::make_pair(a, b) : std::make_pair(b, a);
    if (boundaryEdgeKeys.find(key) != boundaryEdgeKeys.end())
      boundaryEdgeIdx_.push_back(e);
    else
      interiorEdgeIdx_.push_back(e);
  }

  // Interior vertices: those on no boundary face (the coned-in apexes).
  std::unordered_set<std::uint64_t> seen;
  for (const auto v : st_->getVertexList()->toVector()) {
    if (v == nullptr) continue;
    if (!seen.insert(v->getId()).second) continue;
    if (boundaryVertexIds.find(v->getId()) == boundaryVertexIds.end())
      ++interiorVertexCount_;
  }

  // Persist the boundary vertex set (sorted) for boundaryVertexIds() — the
  // candidate pool a "boundary-star" connectivity search wires into.
  boundaryVertexIdsSorted_.assign(boundaryVertexIds.begin(),
                                  boundaryVertexIds.end());
  std::sort(boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
}

std::vector<cd> EigenstateSynthesis::apply(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::apply: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  std::vector<cd> out(N, cd(0.0, 0.0));
  if (N == 0) return out;
  // Reassembled from the live edges on each call (see readoutLaplacian): the
  // U(1) connection L = D - A at k = 0, the Hodge L_k at k >= 1.
  const std::vector<cd> L = readoutLaplacian();
  for (std::size_t i = 0; i < N; ++i) {
    cd acc(0.0, 0.0);
    for (std::size_t j = 0; j < N; ++j) acc += L[i * N + j] * psi[j];
    out[i] = acc;
  }
  return out;
}

double EigenstateSynthesis::residual(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::residual: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  // Normalize: r and the eigenvector condition are scale-invariant, and r is
  // defined for a unit target.
  double nrm2 = 0.0;
  for (const cd &c : psi) nrm2 += std::norm(c);
  if (nrm2 <= 0.0) return 0.0;
  const double inv = 1.0 / std::sqrt(nrm2);
  std::vector<cd> p(N);
  for (std::size_t i = 0; i < N; ++i) p[i] = psi[i] * inv;

  const std::vector<cd> Lp = apply(p);
  // lambda = p^dagger L p (real for Hermitian L; take the real part).
  cd lam(0.0, 0.0);
  for (std::size_t i = 0; i < N; ++i) lam += std::conj(p[i]) * Lp[i];
  const double lambda = lam.real();

  // r = || L p - lambda p ||^2 = ||(I - p p^dagger) L p||^2 (p unit).
  double r = 0.0;
  for (std::size_t i = 0; i < N; ++i) r += std::norm(Lp[i] - lambda * p[i]);
  return r;
}

double EigenstateSynthesis::rayleigh(const std::vector<cd> &psi) const {
  const std::size_t N = order_;
  if (psi.size() != N)
    throw std::runtime_error(
        "EigenstateSynthesis::rayleigh: psi has length " +
        std::to_string(psi.size()) + ", expected " + std::to_string(N));
  if (N == 0) return 0.0;

  const std::vector<cd> Lp = apply(psi);
  cd num(0.0, 0.0);
  double den = 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    num += std::conj(psi[i]) * Lp[i];
    den += std::norm(psi[i]);
  }
  if (den <= 0.0) return 0.0;
  return num.real() / den;
}

std::vector<std::complex<double>> EigenstateSynthesis::weights() const {
  std::vector<std::complex<double>> w;
  w.reserve(edges_.size());
  for (const auto e : edges_) w.push_back((e->getLength() * e->getLength()));
  return w;
}

std::vector<std::complex<double>> EigenstateSynthesis::phases() const {
  std::vector<std::complex<double>> th;
  th.reserve(edges_.size());
  for (const auto e : edges_) th.push_back(e->getPhase());
  return th;
}

void EigenstateSynthesis::setWeights(const std::vector<double> &w) {
  if (w.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setWeights: got " + std::to_string(w.size()) +
        " weights, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setLength(std::sqrt(std::complex<double>{w[i], 0.0}));
}

void EigenstateSynthesis::setPhases(
    const std::vector<std::complex<double>> &theta) {
  if (theta.size() != edges_.size())
    throw std::runtime_error(
        "EigenstateSynthesis::setPhases: got " + std::to_string(theta.size()) +
        " phases, expected " + std::to_string(edges_.size()));
  for (std::size_t i = 0; i < edges_.size(); ++i)
    edges_[i]->setPhase(theta[i]);
}

}  // namespace tessera::cobordism
