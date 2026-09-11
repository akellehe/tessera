// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/CovariantChainHodge.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <numeric>
#include <random>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::chainhodge {

// ---------------------------------------------------------------- Connection

Connection::Connection(const cobordism::ChainComplex &K, std::vector<Complex> links)
    : links_(std::move(links)) {
  const auto edges = K.kSimplexVertices(1);
  edges_.reserve(edges.size());
  if (links_.size() != edges.size())
    throw std::invalid_argument("Connection: one link per canonical edge is required (" +
                                std::to_string(edges.size()) + " edges, " +
                                std::to_string(links_.size()) + " links)");
  for (std::size_t j = 0; j < edges.size(); ++j) {
    if (links_[j] == Complex(0.0, 0.0))
      throw std::invalid_argument("Connection: a link must be nonzero (U_xy in C*)");
    index_[{edges[j][0], edges[j][1]}] = static_cast<int>(j);
    edges_.emplace_back(edges[j][0], edges[j][1]);
  }
}

Connection Connection::trivial(const cobordism::ChainComplex &K) {
  return Connection(K, std::vector<Complex>(K.numSimplices(1), Complex(1.0, 0.0)));
}

Connection Connection::fromSpacetime(const spacetime::Spacetime &st,
                                     const cobordism::ChainComplex &K) {
  std::map<std::pair<std::uint64_t, std::uint64_t>, Complex> byPair;
  if (st.getEdgeList()) {
    for (const auto &e : st.getEdgeList()->toVector()) {
      if (e == nullptr || e->getSource() == nullptr || e->getTarget() == nullptr) continue;
      const std::uint64_t a = e->getSource()->getId();
      const std::uint64_t b = e->getTarget()->getId();
      const Complex link = std::exp(Complex(0.0, 1.0) * e->getPhase());  // source -> target
      if (a < b)
        byPair[{a, b}] = link;
      else
        byPair[{b, a}] = Complex(1.0, 0.0) / link;
    }
  }
  std::vector<Complex> links;
  for (const auto &e : K.kSimplexVertices(1)) {
    const auto it = byPair.find({e[0], e[1]});
    if (it == byPair.end())
      throw std::invalid_argument("Connection::fromSpacetime: complex edge (" +
                                  std::to_string(e[0]) + "," + std::to_string(e[1]) +
                                  ") has no spacetime edge");
    links.push_back(it->second);
  }
  return Connection(K, std::move(links));
}

Complex Connection::link(std::uint64_t x, std::uint64_t y) const {
  if (x == y) return Complex(1.0, 0.0);
  const bool forward = x < y;
  const auto it = index_.find(forward ? std::make_pair(x, y) : std::make_pair(y, x));
  if (it == index_.end())
    throw std::invalid_argument("Connection::link: (" + std::to_string(x) + "," +
                                std::to_string(y) + ") is not an edge");
  const Complex u = links_[static_cast<std::size_t>(it->second)];
  return forward ? u : Complex(1.0, 0.0) / u;
}

Connection Connection::inverse() const {
  Connection out(*this);
  for (auto &u : out.links_) u = Complex(1.0, 0.0) / u;
  return out;
}

Connection Connection::gauge(const std::map<std::uint64_t, Complex> &g) const {
  Connection out(*this);
  for (std::size_t j = 0; j < edges_.size(); ++j) {
    const auto gx = g.find(edges_[j].first);
    const auto gy = g.find(edges_[j].second);
    if (gx == g.end() || gy == g.end() || gx->second == Complex(0.0, 0.0) || gy->second == Complex(0.0, 0.0))
      throw std::invalid_argument("Connection::gauge: every vertex needs a nonzero g_x");
    out.links_[j] = links_[j] / gx->second * gy->second;  // U_xy -> g_x^{-1} U_xy g_y
  }
  return out;
}

Complex Connection::curvature(std::uint64_t p, std::uint64_t q, std::uint64_t r) const {
  return link(r, q) * link(q, p) * link(p, r);
}

bool Connection::isUnitary(double tolerance) const {
  for (const auto &u : links_)
    if (std::abs(std::abs(u) - 1.0) > tolerance) return false;
  return true;
}

std::vector<std::pair<int, double>> Connection::checkWalk(const Walk &walk, const char *who) const {
  if (walk.empty()) throw std::invalid_argument(std::string(who) + ": the walk is empty");
  std::vector<std::pair<int, double>> steps;
  steps.reserve(walk.size());
  for (std::size_t k = 0; k < walk.size(); ++k) {
    const auto [u, v] = walk[k];
    if (u == v)
      throw std::invalid_argument(std::string(who) + ": step " + std::to_string(k) + " is a self-loop at vertex " +
                                  std::to_string(u));
    const auto it = index_.find(u < v ? std::make_pair(u, v) : std::make_pair(v, u));
    if (it == index_.end())
      throw std::invalid_argument(std::string(who) + ": step " + std::to_string(k) + " (" + std::to_string(u) +
                                  " -> " + std::to_string(v) + ") is not an edge");
    if (k + 1 < walk.size() && walk[k + 1].first != v)
      throw std::invalid_argument(std::string(who) + ": the walk does not chain: step " + std::to_string(k) +
                                  " ends at vertex " + std::to_string(v) + " but step " + std::to_string(k + 1) +
                                  " starts at vertex " + std::to_string(walk[k + 1].first));
    steps.emplace_back(it->second, u < v ? 1.0 : -1.0);
  }
  if (walk.back().second != walk.front().first)
    throw std::invalid_argument(std::string(who) + ": the walk is not closed: it starts at vertex " +
                                std::to_string(walk.front().first) + " and ends at vertex " +
                                std::to_string(walk.back().second));
  return steps;
}

Complex Connection::holonomy(const Walk &walk) const {
  static_cast<void>(checkWalk(walk, "Connection::holonomy"));
  Complex product(1.0, 0.0);
  for (const auto &[u, v] : walk) product *= link(u, v);
  return product;
}

Complex Connection::transportedPeriod(const Eigen::VectorXcd &cochain, const Walk &walk) const {
  if (cochain.size() != static_cast<Eigen::Index>(links_.size()))
    throw std::invalid_argument("Connection::transportedPeriod: the cochain has " +
                                std::to_string(cochain.size()) + " entries for " +
                                std::to_string(links_.size()) + " canonical edges");
  const auto steps = checkWalk(walk, "Connection::transportedPeriod");
  Complex total(0.0, 0.0);
  Complex toBase(1.0, 0.0);  // U_{v_0 v_1} ... U_{v_{k-1} v_k}: carries a value at v_k to v_0
  for (std::size_t k = 0; k < walk.size(); ++k) {
    const auto [u, v] = walk[k];
    const auto [index, sign] = steps[k];
    // The edge's value sits at its base vertex min(u, v); U_{u, min(u,v)}
    // carries it to the step's source u: the identity when u is the base,
    // the link U_{uv} of the step when the step runs against the orientation.
    const Complex toSource = (u < v) ? Complex(1.0, 0.0) : link(u, v);
    total += sign * toBase * toSource * cochain(index);
    toBase *= link(u, v);
  }
  return total;
}

Connection::Walk Connection::closedWalkOf(const Walk &steps) {
  if (steps.empty()) return {};
  // Balanced degrees at every vertex, else the steps cannot close into one
  // circuit; the outgoing steps of a vertex in the given order.
  std::map<std::uint64_t, std::vector<std::size_t>> outgoing;
  std::map<std::uint64_t, long> netDegree;
  for (std::size_t k = 0; k < steps.size(); ++k) {
    outgoing[steps[k].first].push_back(k);
    ++netDegree[steps[k].first];
    --netDegree[steps[k].second];
  }
  for (const auto &[vertex, degree] : netDegree)
    if (degree != 0) return {};
  std::map<std::uint64_t, std::size_t> next;
  std::vector<std::uint64_t> vertexStack = {steps.front().first};
  std::vector<std::size_t> stepStack, circuit;
  while (!vertexStack.empty()) {
    const std::uint64_t v = vertexStack.back();
    const auto out = outgoing.find(v);
    std::size_t &taken = next[v];
    if (out != outgoing.end() && taken < out->second.size()) {
      const std::size_t k = out->second[taken++];
      vertexStack.push_back(steps[k].second);
      stepStack.push_back(k);
    } else {
      vertexStack.pop_back();
      if (!stepStack.empty()) {
        circuit.push_back(stepStack.back());
        stepStack.pop_back();
      }
    }
  }
  std::reverse(circuit.begin(), circuit.end());
  if (circuit.size() != steps.size()) return {};  // more than one component
  Walk walk;
  walk.reserve(steps.size());
  for (const std::size_t k : circuit) walk.push_back(steps[k]);
  return walk;
}

std::optional<std::uint64_t> Connection::commonBasePoint(std::vector<Walk> &walks) {
  if (walks.empty() || walks.front().empty()) return std::nullopt;
  std::optional<std::uint64_t> base;
  for (const auto &step : walks.front()) {
    const std::uint64_t candidate = step.first;
    bool onEvery = true;
    for (std::size_t w = 1; w < walks.size() && onEvery; ++w)
      onEvery = std::any_of(walks[w].begin(), walks[w].end(),
                            [&](const std::pair<std::uint64_t, std::uint64_t> &s) { return s.first == candidate; });
    if (onEvery) {
      base = candidate;
      break;
    }
  }
  if (!base) return std::nullopt;
  for (auto &walk : walks) {
    std::size_t k = 0;
    while (walk[k].first != *base) ++k;
    std::rotate(walk.begin(), walk.begin() + static_cast<std::ptrdiff_t>(k), walk.end());
  }
  return base;
}

// ---------------------------------------------------------------- CovariantChainHodge

struct CovariantChainHodge::Factorization {
  Eigen::SparseLU<SparseMatrix> lu;
  bool ok{false};
};

SparseMatrix CovariantChainHodge::dress(const SparseMatrix &M,
                                        const std::vector<std::uint64_t> &baseRow,
                                        const std::vector<std::uint64_t> &baseCol,
                                        const Connection &U) {
  SparseMatrix out(M.rows(), M.cols());
  std::vector<Eigen::Triplet<Complex>> trip;
  trip.reserve(static_cast<std::size_t>(M.nonZeros()));
  for (int c = 0; c < M.outerSize(); ++c)
    for (SparseMatrix::InnerIterator it(M, c); it; ++it)
      trip.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()),
                        it.value() * U.link(baseRow[static_cast<std::size_t>(it.row())],
                                            baseCol[static_cast<std::size_t>(it.col())]));
  out.setFromTriplets(trip.begin(), trip.end());
  out.makeCompressed();
  return out;
}

CovariantChainHodge::CovariantChainHodge(const ChainHodge &base, Connection U, std::uint64_t gaugeSeed,
                                         bool measureCertificate)
    : base_(std::make_shared<ChainHodge>(base)), U_(std::move(U)), Uinv_(U_.inverse()) {
  const int d = base_->dimension();
  if (U_.edgeCount() != base_->complex().numSimplices(1))
    throw std::invalid_argument("CovariantChainHodge: the connection lives on a different edge set");
  base_vertex_.resize(static_cast<std::size_t>(d) + 1);
  for (int k = 0; k <= d; ++k)
    for (const auto &cell : base_->complex().kSimplexVertices(k))
      base_vertex_[static_cast<std::size_t>(k)].push_back(cell.front());  // b(σ) = min σ (sorted)
  for (int k = 0; k <= d; ++k) {
    const auto &bk = base_vertex_[static_cast<std::size_t>(k)];
    const SparseMatrix &sparse = (base_->preset() == Preset::L2) ? base_->Minv(k) : base_->chainMetricSparse(k);
    dressed_.push_back(dress(sparse, bk, bk, U_));
    dressedDual_.push_back(dress(sparse, bk, bk, Uinv_));
    if (k >= 1) {
      const auto &bkm1 = base_vertex_[static_cast<std::size_t>(k) - 1];
      twisted_.push_back(dress(base_->boundary(k), bkm1, bk, U_));      // (∂_k^U)_{τσ} = (∂_k)_{τσ} U_{b(τ)b(σ)}
      twistedDual_.push_back(dress(base_->boundary(k), bkm1, bk, Uinv_));
    } else {
      twisted_.push_back(base_->boundary(0));
      twistedDual_.push_back(base_->boundary(0));
    }
  }
  factor_.assign(static_cast<std::size_t>(d) + 1, nullptr);
  workspace_.assign(static_cast<std::size_t>(d) + 1, nullptr);
  cert_.gaugeSeed = gaugeSeed;
  if (measureCertificate) measureSparseIdentities(gaugeSeed);
}

// ---------------------------------------------------------------- derivatives

struct CovariantChainHodge::DerivativeWorkspace {
  // Whitney (L2) only: h = M_k A P B + C M_{k+1} D Q with A = (∂_k^{U^{-1}})^T,
  // B = ∂_k^U, C = ∂_{k+1}^U, D = (∂_{k+1}^{U^{-1}})^T, P = (M_{k-1}^U)^{-1}, Q = (M_k^U)^{-1}.
  Eigen::MatrixXcd PB;     // P B            (n_{k-1} x n_k)
  Eigen::MatrixXcd Q;      // (M_k^U)^{-1}   (n_k x n_k)
  Eigen::MatrixXcd DQ;     // D Q            (n_{k+1} x n_k)
  Eigen::MatrixXcd T1;     // M_k A          (n_k x n_{k-1})
  Eigen::MatrixXcd T2;     // C M_{k+1} D Q  (n_k x n_k)
  bool hasLower{false}, hasUpper{false};
};

const CovariantChainHodge::DerivativeWorkspace &CovariantChainHodge::derivativeWorkspace(int k) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge: operator derivatives are the Whitney preset's");
  const int n = base_->size(k);
  if (n >= base_->crossoverDimension())
    throw std::length_error("CovariantChainHodge: derivatives are formed densely below the crossover only");
  auto &slot = workspace_[static_cast<std::size_t>(k)];
  if (slot) return *slot;
  auto w = std::make_shared<DerivativeWorkspace>();
  const int d = dimension();
  const SparseMatrix &Mk = dressed_[static_cast<std::size_t>(k)];
  w->Q = solveDressed(k, Eigen::MatrixXcd::Identity(n, n));
  if (k >= 1) {
    w->hasLower = true;
    w->PB = solveDressed(k - 1, Eigen::MatrixXcd(twisted_[static_cast<std::size_t>(k)]));
    w->T1 = Eigen::MatrixXcd(Mk * SparseMatrix(twistedDual_[static_cast<std::size_t>(k)].transpose()));
  }
  if (k < d) {
    w->hasUpper = true;
    const SparseMatrix &C = twisted_[static_cast<std::size_t>(k) + 1];
    const SparseMatrix DT = SparseMatrix(twistedDual_[static_cast<std::size_t>(k) + 1].transpose());
    w->DQ = DT * w->Q;
    w->T2 = Eigen::MatrixXcd(C * dressed_[static_cast<std::size_t>(k) + 1]) * w->DQ;
  }
  slot = std::move(w);
  return *slot;
}

void CovariantChainHodge::warmDerivatives(int k) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  // Only the Whitney preset below the dense crossover has the derivative
  // workspace at all; anywhere else there is nothing to warm and asking for it
  // would throw, so the caller can warm unconditionally.
  if (preset() != Preset::L2) return;
  if (base_->size(k) >= base_->crossoverDimension()) return;
  // Fills workspace_[k], and through solveDressed the factorizations at k and
  // (for k >= 1) at k-1 -- every mutable slot the length and phase derivative
  // paths touch. `resolvent` factorizes its own bordered system per call and
  // caches nothing, so it needs no warming.
  (void)derivativeWorkspace(k);
}

Eigen::MatrixXcd CovariantChainHodge::assembleDerivative(
    int k, const SparseMatrix *dMkm1, const SparseMatrix *dMk, const SparseMatrix *dMkp1,
    const SparseMatrix *dBk, const SparseMatrix *dBkDual, const SparseMatrix *dBkp1,
    const SparseMatrix *dBkp1Dual) const {
  const DerivativeWorkspace &w = derivativeWorkspace(k);
  const int n = base_->size(k);
  Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(n, n);
  const SparseMatrix &Mk = dressed_[static_cast<std::size_t>(k)];
  if (w.hasLower) {
    const SparseMatrix &B = twisted_[static_cast<std::size_t>(k)];
    const SparseMatrix AT = SparseMatrix(twistedDual_[static_cast<std::size_t>(k)].transpose());
    // d(M_k) A P B
    if (dMk) out += Eigen::MatrixXcd(*dMk * AT) * w.PB;
    // M_k d(A) P B
    if (dBkDual) out += Eigen::MatrixXcd(Mk * SparseMatrix(dBkDual->transpose())) * w.PB;
    // M_k A d(P) B = -M_k A P dM_{k-1} P B
    if (dMkm1) out -= w.T1 * solveDressed(k - 1, Eigen::MatrixXcd(*dMkm1 * w.PB));
    // M_k A P d(B)
    if (dBk) out += w.T1 * solveDressed(k - 1, Eigen::MatrixXcd(*dBk));
  }
  if (w.hasUpper) {
    const SparseMatrix &C = twisted_[static_cast<std::size_t>(k) + 1];
    const SparseMatrix &Mk1 = dressed_[static_cast<std::size_t>(k) + 1];
    const SparseMatrix DT = SparseMatrix(twistedDual_[static_cast<std::size_t>(k) + 1].transpose());
    // d(C) M_{k+1} D Q
    if (dBkp1) out += Eigen::MatrixXcd(*dBkp1 * Mk1) * w.DQ;
    // C d(M_{k+1}) D Q
    if (dMkp1) out += Eigen::MatrixXcd(C * *dMkp1) * w.DQ;
    // C M_{k+1} d(D) Q
    if (dBkp1Dual) out += Eigen::MatrixXcd(C * Mk1 * SparseMatrix(dBkp1Dual->transpose())) * w.Q;
    // C M_{k+1} D d(Q) = -T2 dM_k Q
    if (dMk) out -= w.T2 * (Eigen::MatrixXcd(*dMk) * w.Q);
  }
  return out;
}

Eigen::MatrixXcd CovariantChainHodge::covariantOperatorDerivative(int k, std::size_t edgeIndex) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const int d = dimension();
  const auto &K = base_->complex();
  const auto &s = base_->squaredLengths();
  const Branch branch = base_->branch();
  SparseMatrix dMkm1, dMk, dMkp1;
  if (k >= 1) {
    const auto &b = base_vertex_[static_cast<std::size_t>(k) - 1];
    dMkm1 = dress(WhitneyMass::assembleDerivative(K, s, k - 1, edgeIndex, branch), b, b, U_);
  }
  {
    const auto &b = base_vertex_[static_cast<std::size_t>(k)];
    dMk = dress(WhitneyMass::assembleDerivative(K, s, k, edgeIndex, branch), b, b, U_);
  }
  if (k < d) {
    const auto &b = base_vertex_[static_cast<std::size_t>(k) + 1];
    dMkp1 = dress(WhitneyMass::assembleDerivative(K, s, k + 1, edgeIndex, branch), b, b, U_);
  }
  return assembleDerivative(k, k >= 1 ? &dMkm1 : nullptr, &dMk, k < d ? &dMkp1 : nullptr,
                            nullptr, nullptr, nullptr, nullptr);
}

SparseMatrix CovariantChainHodge::dressedDerivative(int k, std::size_t edgeIndex) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const auto &b = base_vertex_[static_cast<std::size_t>(k)];
  return dress(WhitneyMass::assembleDerivative(base_->complex(), base_->squaredLengths(), k, edgeIndex,
                                              base_->branch()), b, b, U_);
}

SparseMatrix CovariantChainHodge::dressedPhaseDerivative(int k, std::size_t edgeIndex) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const auto edges = base_->complex().kSimplexVertices(1);
  if (edgeIndex >= edges.size()) throw std::invalid_argument("CovariantChainHodge: edge index out of range");
  const auto &b = base_vertex_[static_cast<std::size_t>(k)];
  return phaseDerivative(dressed_[static_cast<std::size_t>(k)], b, b, edges[edgeIndex][0], edges[edgeIndex][1], false);
}

SparseMatrix CovariantChainHodge::phaseDerivative(const SparseMatrix &dressedM,
                                                  const std::vector<std::uint64_t> &baseRow,
                                                  const std::vector<std::uint64_t> &baseCol,
                                                  std::uint64_t x, std::uint64_t y, bool dual) {
  SparseMatrix out(dressedM.rows(), dressedM.cols());
  std::vector<Eigen::Triplet<Complex>> trip;
  const Complex plus = dual ? Complex(0.0, -1.0) : Complex(0.0, 1.0);
  for (int c = 0; c < dressedM.outerSize(); ++c)
    for (SparseMatrix::InnerIterator it(dressedM, c); it; ++it) {
      const std::uint64_t a = baseRow[static_cast<std::size_t>(it.row())];
      const std::uint64_t b = baseCol[static_cast<std::size_t>(it.col())];
      if (a == x && b == y)
        trip.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), plus * it.value());
      else if (a == y && b == x)
        trip.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), -plus * it.value());
    }
  out.setFromTriplets(trip.begin(), trip.end());
  out.makeCompressed();
  return out;
}

Eigen::MatrixXcd CovariantChainHodge::covariantOperatorPhaseDerivative(int k, std::size_t edgeIndex) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const int d = dimension();
  const auto edges = base_->complex().kSimplexVertices(1);
  if (edgeIndex >= edges.size()) throw std::invalid_argument("CovariantChainHodge: edge index out of range");
  const std::uint64_t x = edges[edgeIndex][0], y = edges[edgeIndex][1];
  const auto &bk = base_vertex_[static_cast<std::size_t>(k)];
  SparseMatrix dMkm1, dMk, dMkp1, dBk, dBkDual, dBkp1, dBkp1Dual;
  dMk = phaseDerivative(dressed_[static_cast<std::size_t>(k)], bk, bk, x, y, false);
  if (k >= 1) {
    const auto &bkm1 = base_vertex_[static_cast<std::size_t>(k) - 1];
    dMkm1 = phaseDerivative(dressed_[static_cast<std::size_t>(k) - 1], bkm1, bkm1, x, y, false);
    dBk = phaseDerivative(twisted_[static_cast<std::size_t>(k)], bkm1, bk, x, y, false);
    dBkDual = phaseDerivative(twistedDual_[static_cast<std::size_t>(k)], bkm1, bk, x, y, true);
  }
  if (k < d) {
    const auto &bkp1 = base_vertex_[static_cast<std::size_t>(k) + 1];
    dMkp1 = phaseDerivative(dressed_[static_cast<std::size_t>(k) + 1], bkp1, bkp1, x, y, false);
    dBkp1 = phaseDerivative(twisted_[static_cast<std::size_t>(k) + 1], bk, bkp1, x, y, false);
    dBkp1Dual = phaseDerivative(twistedDual_[static_cast<std::size_t>(k) + 1], bk, bkp1, x, y, true);
  }
  return assembleDerivative(k, k >= 1 ? &dMkm1 : nullptr, &dMk, k < d ? &dMkp1 : nullptr,
                            k >= 1 ? &dBk : nullptr, k >= 1 ? &dBkDual : nullptr,
                            k < d ? &dBkp1 : nullptr, k < d ? &dBkp1Dual : nullptr);
}

const SparseMatrix &CovariantChainHodge::Minv(int k) const {
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge::Minv: the GRASSMANN_ALL preset's dressed sparse "
                           "object is the chain metric (dressed)");
  return dressed(k);
}

const SparseMatrix &CovariantChainHodge::dressed(int k) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  return dressed_[static_cast<std::size_t>(k)];
}

const SparseMatrix &CovariantChainHodge::twistedBoundary(int k) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  return twisted_[static_cast<std::size_t>(k)];
}

const SparseMatrix &CovariantChainHodge::twistedBoundaryDual(int k) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  return twistedDual_[static_cast<std::size_t>(k)];
}

Eigen::VectorXcd CovariantChainHodge::rho(int k, const std::map<std::uint64_t, Complex> &g) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const auto &bk = base_vertex_[static_cast<std::size_t>(k)];
  Eigen::VectorXcd out(static_cast<Eigen::Index>(bk.size()));
  for (std::size_t j = 0; j < bk.size(); ++j) {
    const auto it = g.find(bk[j]);
    if (it == g.end() || it->second == Complex(0.0, 0.0))
      throw std::invalid_argument("CovariantChainHodge::rho: every base vertex needs a nonzero g");
    out(static_cast<Eigen::Index>(j)) = Complex(1.0, 0.0) / it->second;
  }
  return out;
}

Eigen::MatrixXcd CovariantChainHodge::solveDressed(int k, const Eigen::MatrixXcd &rhs) const {
  auto &slot = factor_[static_cast<std::size_t>(k)];
  if (!slot) {
    auto f = std::make_shared<Factorization>();
    f->lu.compute(dressed_[static_cast<std::size_t>(k)]);
    f->ok = (f->lu.info() == Eigen::Success);
    slot = std::move(f);
  }
  if (!slot->ok)
    throw std::runtime_error("CovariantChainHodge: the dressed sparse metric at degree " +
                             std::to_string(k) + " is singular");
  if (rhs.cols() == 0) return Eigen::MatrixXcd(rhs.rows(), 0);
  return slot->lu.solve(rhs);
}

Eigen::MatrixXcd CovariantChainHodge::applyG(int k, const Eigen::MatrixXcd &c) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  if (preset() == Preset::L2) return solveDressed(k, c);
  return dressed_[static_cast<std::size_t>(k)] * c;
}

Eigen::MatrixXcd CovariantChainHodge::applyMinv(int k, const Eigen::MatrixXcd &c) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  if (preset() == Preset::L2) return dressed_[static_cast<std::size_t>(k)] * c;
  return solveDressed(k, c);
}

Eigen::MatrixXcd CovariantChainHodge::applyH(int k, const Eigen::MatrixXcd &c) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const int d = dimension();
  const SparseMatrix &Mk = dressed_[static_cast<std::size_t>(k)];
  Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(c.rows(), c.cols());
  if (preset() == Preset::L2) {
    if (k >= 1) {
      // M_k^U (∂_k^{U^{-1}})^T (M_{k-1}^U)^{-1} ∂_k^U c
      const Eigen::MatrixXcd y = solveDressed(k - 1, Eigen::MatrixXcd(twisted_[static_cast<std::size_t>(k)] * c));
      out += Mk * (SparseMatrix(twistedDual_[static_cast<std::size_t>(k)].transpose()) * y);
    }
    if (k < d) {
      // ∂_{k+1}^U M_{k+1}^U (∂_{k+1}^{U^{-1}})^T (M_k^U)^{-1} c
      const Eigen::MatrixXcd w = solveDressed(k, c);
      out += twisted_[static_cast<std::size_t>(k) + 1] *
             (dressed_[static_cast<std::size_t>(k) + 1] *
              (SparseMatrix(twistedDual_[static_cast<std::size_t>(k) + 1].transpose()) * w));
    }
    return out;
  }
  // Grassmann: h = G^{-1} A on chains, A = (∂^{U^{-1}})^T G_{k-1} ∂^U + G_k ∂_{k+1}^U G_{k+1}^{-1} (∂_{k+1}^{U^{-1}})^T G_k
  return solveDressed(k, pencil(k).A * c);
}

Eigen::MatrixXcd CovariantChainHodge::covariantOperator(int k) const {
  const int n = base_->size(k);
  if (n >= base_->crossoverDimension())
    throw std::length_error("CovariantChainHodge::covariantOperator: at or above the dense crossover");
  return applyH(k, Eigen::MatrixXcd::Identity(n, n));
}

Pencil CovariantChainHodge::pencil(int k) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const int n = base_->size(k);
  if (n >= base_->crossoverDimension())
    throw std::length_error("CovariantChainHodge::pencil: at or above the dense crossover");
  const int d = dimension();
  Pencil P;
  P.degree = k;
  const SparseMatrix &Mk = dressed_[static_cast<std::size_t>(k)];
  P.B = Eigen::MatrixXcd(Mk);
  P.A = Eigen::MatrixXcd::Zero(n, n);
  if (preset() == Preset::L2) {
    P.variable = PencilVariable::GeometricImage;
    if (k >= 1) {
      // M_k^U (∂_k^{U^{-1}})^T (M_{k-1}^U)^{-1} ∂_k^U M_k^U
      // M_k^U (∂_k^{U^{-1}})^T (M_{k-1}^U)^{-1} ∂_k^U M_k^U. The dressed metric is
      // NOT symmetric ((M^U)^T = M^{U^{-1}}), so the left factor is formed as written.
      const Eigen::MatrixXcd X = Eigen::MatrixXcd(twisted_[static_cast<std::size_t>(k)] * Mk);
      const Eigen::MatrixXcd Y = solveDressed(k - 1, X);
      P.A += Eigen::MatrixXcd(Mk * SparseMatrix(twistedDual_[static_cast<std::size_t>(k)].transpose())) * Y;
    }
    if (k < d) {
      const SparseMatrix &B = twisted_[static_cast<std::size_t>(k) + 1];
      const SparseMatrix &BD = twistedDual_[static_cast<std::size_t>(k) + 1];
      P.A += Eigen::MatrixXcd(B * dressed_[static_cast<std::size_t>(k) + 1] * SparseMatrix(BD.transpose()));
    }
  } else {
    P.variable = PencilVariable::Chain;
    if (k >= 1) {
      const SparseMatrix &B = twisted_[static_cast<std::size_t>(k)];
      const SparseMatrix &BD = twistedDual_[static_cast<std::size_t>(k)];
      P.A += Eigen::MatrixXcd(SparseMatrix(BD.transpose()) * dressed_[static_cast<std::size_t>(k) - 1] * B);
    }
    if (k < d) {
      const SparseMatrix &B = twisted_[static_cast<std::size_t>(k) + 1];
      const SparseMatrix &BD = twistedDual_[static_cast<std::size_t>(k) + 1];
      const Eigen::MatrixXcd X = Eigen::MatrixXcd(SparseMatrix(BD.transpose()) * Mk);
      const Eigen::MatrixXcd Y = solveDressed(k + 1, X);
      P.A += Eigen::MatrixXcd(Mk * B) * Y;
    }
  }
  return P;
}

Eigen::MatrixXcd CovariantChainHodge::pencilAux(int k) const {
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge::pencilAux: Whitney preset only");
  return pencil(k).A;
}

SpectrumRead CovariantChainHodge::spectrum(int k) const {
  const Pencil P = pencil(k);
  const int n = static_cast<int>(P.A.rows());
  SpectrumRead read;
  read.degree = k;
  if (n == 0) return read;
  const Eigen::MatrixXcd C = solveDressed(k, P.A);
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(C, true);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("CovariantChainHodge::spectrum: eigensolver did not converge");
  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  const auto &ev = es.eigenvalues();
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (ev(a).real() != ev(b).real()) return ev(a).real() < ev(b).real();
    return ev(a).imag() < ev(b).imag();
  });
  read.vectors.resize(n, n);
  const double normA = P.A.norm();
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = order[static_cast<std::size_t>(i)];
    read.eigenvalues.push_back(ev(j));
    Eigen::VectorXcd x = es.eigenvectors().col(j);
    x /= x.norm();
    read.vectors.col(i) = x;
    const double res = (P.A * x - ev(j) * (P.B * x)).norm();
    worst = std::max(worst, normA > 0.0 ? res / normA : res);
  }
  read.residual = worst;
  return read;
}

PencilRegimeCertificate CovariantChainHodge::regimeCertificate(int k, double tolerance) const {
  PencilRegimeCertificate c;
  c.tolerance = tolerance;
  bool trivial = true;
  for (const auto &u : U_.links())
    if (std::abs(u - Complex(1.0, 0.0)) > 0.0) { trivial = false; break; }
  c.trivialConnection = trivial;
  const SparseMatrix &M = dressed_[static_cast<std::size_t>(k < 0 ? 0 : k)];
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  const SparseMatrix Mt = SparseMatrix(M.transpose());
  const double nM = M.norm();
  c.metricSymmetryDefect = (nM > 0.0)
      ? SparseMatrix(Mt - dressedDual_[static_cast<std::size_t>(k)]).norm() / nM
      : SparseMatrix(Mt - dressedDual_[static_cast<std::size_t>(k)]).norm();
  const Pencil P = pencil(k);
  const Pencil PD = trivial ? P : dual().pencil(k);
  const double nA = P.A.norm();
  const Eigen::MatrixXcd diff = P.A.transpose() - PD.A;
  c.symmetryDefect = (nA > 0.0) ? diff.norm() / nA : diff.norm();
  c.regime = (c.symmetryDefect <= tolerance && c.metricSymmetryDefect <= tolerance)
                 ? cobordism::CertificateRegime::ComplexSymmetricPencil
                 : cobordism::CertificateRegime::NonNormal;
  return c;
}

CovariantChainHodge CovariantChainHodge::dual() const {
  return CovariantChainHodge(*base_, Uinv_, cert_.gaugeSeed);
}

CovariantChainHodge CovariantChainHodge::gauged(const std::map<std::uint64_t, Complex> &g) const {
  return CovariantChainHodge(*base_, U_.gauge(g), cert_.gaugeSeed);
}

namespace {

double relativeSparseDiff(const SparseMatrix &A, const SparseMatrix &B) {
  const double n = A.norm();
  const SparseMatrix D = A - B;
  return n > 0.0 ? D.norm() / n : D.norm();
}

std::map<std::uint64_t, Complex> randomGauge(const cobordism::ChainComplex &K, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::map<std::uint64_t, Complex> g;
  for (const auto &v : K.kSimplexVertices(0)) {
    Complex z(nd(rng), nd(rng));
    while (std::abs(z) < 0.1) z = Complex(nd(rng), nd(rng));
    g[v[0]] = z;
  }
  return g;
}

}  // namespace

void CovariantChainHodge::measureSparseIdentities(std::uint64_t seed) {
  cert_.gaugeSeed = seed;
  const int d = dimension();
  // (ii) (M_k^U)^T = M_k^{U^{-1}}
  double worst = 0.0;
  for (int k = 0; k <= d; ++k)
    worst = std::max(worst, relativeSparseDiff(SparseMatrix(dressed_[static_cast<std::size_t>(k)].transpose()),
                                               dressedDual_[static_cast<std::size_t>(k)]));
  cert_.transposeMetric = worst;
  // (iii) M_k^{U^g} = ρ_k M_k^U ρ_k^{-1}, and (vi) invariance of the pairing.
  const auto g = randomGauge(base_->complex(), seed);
  const Connection Ug = U_.gauge(g);
  worst = 0.0;
  double worstPairing = 0.0;
  std::mt19937_64 rng(seed ^ 0x5bd1e995ULL);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int k = 0; k <= d; ++k) {
    const auto &bk = base_vertex_[static_cast<std::size_t>(k)];
    const SparseMatrix &sparse = (preset() == Preset::L2) ? base_->Minv(k) : base_->chainMetricSparse(k);
    const SparseMatrix dressedG = dress(sparse, bk, bk, Ug);
    const Eigen::VectorXcd r = rho(k, g);
    const SparseMatrix conj = r.asDiagonal() * dressed_[static_cast<std::size_t>(k)] * r.cwiseInverse().asDiagonal();
    worst = std::max(worst, relativeSparseDiff(conj, dressedG));
    // (vi): c~^T G^U c with c~ -> ρ^{-1} c~, c -> ρ c, G^{U^g} in place of G^U.
    const int n = static_cast<int>(bk.size());
    if (n == 0) continue;
    Eigen::VectorXcd ct(n), c(n);
    for (int i = 0; i < n; ++i) {
      ct(i) = Complex(nd(rng), nd(rng));
      c(i) = Complex(nd(rng), nd(rng));
    }
    const CovariantChainHodge *self = this;
    const Eigen::MatrixXcd Gc = self->applyG(k, c);
    const Complex before = (ct.transpose() * Gc)(0, 0);
    // G^{U^g} (ρ c) = (ρ G^U ρ^{-1}) ρ c = ρ G^U c  ⇒ (ρ^{-1} c~)^T ρ G^U c = c~^T G^U c
    Eigen::MatrixXcd GcG;
    if (preset() == Preset::L2) {
      Eigen::SparseLU<SparseMatrix> lu(dressedG);
      if (lu.info() != Eigen::Success) continue;
      GcG = lu.solve(Eigen::MatrixXcd(r.asDiagonal() * c));
    } else {
      GcG = dressedG * (r.asDiagonal() * c);
    }
    const Complex after = ((r.cwiseInverse().asDiagonal() * ct).transpose() * GcG)(0, 0);
    worstPairing = std::max(worstPairing, std::abs(after - before) / (std::abs(before) + 1.0));
  }
  cert_.covarianceMetric = worst;
  cert_.pairingInvariance = worstPairing;
  // (iv) ∂_1^U ∂_2^U t = U_rp (F_t − 1)[r] for every triangle t = [p<q<r].
  if (d >= 2) {
    const SparseMatrix C = twisted_[1] * twisted_[2];
    const auto tris = base_->complex().kSimplexVertices(2);
    const auto verts = base_->complex().kSimplexVertices(0);
    std::map<std::uint64_t, int> vindex;
    for (int i = 0; i < static_cast<int>(verts.size()); ++i) vindex[verts[static_cast<std::size_t>(i)][0]] = i;
    double worstCurv = 0.0, scale = 1.0;
    for (int t = 0; t < static_cast<int>(tris.size()); ++t) {
      const auto &tri = tris[static_cast<std::size_t>(t)];
      const std::uint64_t p = tri[0], q = tri[1], r = tri[2];
      const Complex expected = U_.link(r, p) * (U_.curvature(p, q, r) - Complex(1.0, 0.0));
      scale = std::max(scale, std::abs(expected));
      Eigen::VectorXcd col = Eigen::VectorXcd(C.col(t));
      const int ir = vindex.at(r);
      double err = std::abs(col(ir) - expected);
      col(ir) = 0.0;
      err = std::max(err, col.norm());
      worstCurv = std::max(worstCurv, err);
    }
    cert_.curvature = worstCurv / scale;
  } else {
    cert_.curvature = 0.0;
  }
}

CovarianceCertificate CovariantChainHodge::verify(int k) const {
  CovarianceCertificate c = cert_;
  c.checkedDegree = k;
  const int n = base_->size(k);
  if (n >= base_->crossoverDimension()) return c;
  // (ii) on the pencil: (Ã_k^U)^T = Ã_k^{U^{-1}}
  const Pencil P = pencil(k);
  const Pencil PD = dual().pencil(k);
  const double nA = P.A.norm();
  c.transposePencil = (nA > 0.0) ? (P.A.transpose() - PD.A).norm() / nA : (P.A.transpose() - PD.A).norm();
  // (iii) on the pencil under the certificate's random gauge
  const auto g = randomGauge(base_->complex(), cert_.gaugeSeed);
  const Pencil PG = gauged(g).pencil(k);
  const Eigen::VectorXcd r = rho(k, g);
  const Eigen::MatrixXcd conj = r.asDiagonal() * P.A * r.cwiseInverse().asDiagonal();
  c.covariancePencil = (nA > 0.0) ? (conj - PG.A).norm() / nA : (conj - PG.A).norm();
  // (i) h_k(s,1) = L_k when the connection is trivial
  bool trivial = true;
  for (const auto &u : U_.links())
    if (std::abs(u - Complex(1.0, 0.0)) > 0.0) { trivial = false; break; }
  if (trivial) {
    const Eigen::MatrixXcd L = base_->hodgeOperator(k);
    const Eigen::MatrixXcd h = covariantOperator(k);
    const double nL = L.norm();
    c.trivialReduction = (nL > 0.0) ? (h - L).norm() / nL : (h - L).norm();
  }
  // (v) pure gauge U = 1^g is isospectral to L_k
  const CovariantChainHodge pure(*base_, Connection::trivial(base_->complex()).gauge(g), cert_.gaugeSeed);
  const std::vector<Complex> a = pure.spectrum(k).eigenvalues;
  const std::vector<Complex> b = base_->spectrum(k).eigenvalues;
  double radius = 0.0, haus = 0.0;
  for (const auto &z : b) radius = std::max(radius, std::abs(z));
  auto oneSided = [](const std::vector<Complex> &x, const std::vector<Complex> &y) {
    double worst = 0.0;
    for (const auto &p : x) {
      double best = std::numeric_limits<double>::infinity();
      for (const auto &q : y) best = std::min(best, std::abs(p - q));
      worst = std::max(worst, best);
    }
    return worst;
  };
  haus = std::max(oneSided(a, b), oneSided(b, a));
  c.pureGaugeIsospectrality = radius > 0.0 ? haus / radius : haus;
  return c;
}

// ---------------------------------------------------------------- Riesz bands

Contour Contour::circle(Complex center, double radius, int nodeCount) {
  if (nodeCount < 3) throw std::invalid_argument("Contour::circle: at least three nodes");
  if (!(radius > 0.0)) throw std::invalid_argument("Contour::circle: radius must be positive");
  Contour c;
  c.nodes.reserve(static_cast<std::size_t>(nodeCount));
  c.weights.reserve(static_cast<std::size_t>(nodeCount));
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  for (int j = 0; j < nodeCount; ++j) {
    const double theta = kTwoPi * (static_cast<double>(j) + 0.5) / static_cast<double>(nodeCount);
    const Complex e = std::exp(Complex(0.0, theta));
    c.nodes.push_back(center + radius * e);
    // (1/2πi) ζ'(θ) Δθ = (1/2πi)(i r e^{iθ})(2π/N) = (r/N) e^{iθ}
    c.weights.push_back((radius / static_cast<double>(nodeCount)) * e);
  }
  c.description = "circle center=(" + std::to_string(center.real()) + "," + std::to_string(center.imag()) +
                  ") radius=" + std::to_string(radius) + " nodes=" + std::to_string(nodeCount);
  return c;
}

Eigen::MatrixXcd CovariantChainHodge::resolvent(int k, Complex zeta, const Eigen::MatrixXcd &c) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge::resolvent: the pencil resolvent is the Whitney preset's");
  const int d = dimension();
  const int n = base_->size(k);
  if (c.rows() != n) throw std::invalid_argument("CovariantChainHodge::resolvent: right-hand side has the wrong height");
  const SparseMatrix &Mk = dressed_[static_cast<std::size_t>(k)];
  // Upper block: ζ M_k − ∂_{k+1}^U M_{k+1}^U (∂_{k+1}^{U^{-1}})^T (sparse).
  SparseMatrix upper = zeta * Mk;
  if (k < d) {
    const SparseMatrix &C = twisted_[static_cast<std::size_t>(k) + 1];
    const SparseMatrix DT = SparseMatrix(twistedDual_[static_cast<std::size_t>(k) + 1].transpose());
    upper = upper - SparseMatrix(C * dressed_[static_cast<std::size_t>(k) + 1] * DT);
  }
  Eigen::MatrixXcd z;
  if (k >= 1) {
    const int m = base_->size(k - 1);
    const SparseMatrix Tlo = SparseMatrix(Mk * SparseMatrix(twistedDual_[static_cast<std::size_t>(k)].transpose()));  // n x m
    const SparseMatrix Slo = SparseMatrix(twisted_[static_cast<std::size_t>(k)] * Mk);                                // m x n
    const SparseMatrix &Mlo = dressed_[static_cast<std::size_t>(k) - 1];
    // Bordered system [[upper, -Tlo], [-Slo, Mlo]] [z; w] = [c; 0].
    std::vector<Eigen::Triplet<Complex>> trip;
    trip.reserve(static_cast<std::size_t>(upper.nonZeros() + Tlo.nonZeros() + Slo.nonZeros() + Mlo.nonZeros()));
    auto scatter = [&](const SparseMatrix &A, int r0, int c0, Complex scale) {
      for (int col = 0; col < A.outerSize(); ++col)
        for (SparseMatrix::InnerIterator it(A, col); it; ++it)
          trip.emplace_back(r0 + static_cast<int>(it.row()), c0 + static_cast<int>(it.col()), scale * it.value());
    };
    scatter(upper, 0, 0, Complex(1.0, 0.0));
    scatter(Tlo, 0, n, Complex(-1.0, 0.0));
    scatter(Slo, n, 0, Complex(-1.0, 0.0));
    scatter(Mlo, n, n, Complex(1.0, 0.0));
    SparseMatrix bordered(n + m, n + m);
    bordered.setFromTriplets(trip.begin(), trip.end());
    bordered.makeCompressed();
    Eigen::SparseLU<SparseMatrix> lu(bordered);
    if (lu.info() != Eigen::Success)
      throw std::runtime_error("CovariantChainHodge::resolvent: the bordered pencil system is singular at this zeta");
    Eigen::MatrixXcd rhs = Eigen::MatrixXcd::Zero(n + m, c.cols());
    rhs.topRows(n) = c;
    const Eigen::MatrixXcd sol = lu.solve(rhs);
    z = sol.topRows(n);
  } else {
    upper.makeCompressed();
    Eigen::SparseLU<SparseMatrix> lu(upper);
    if (lu.info() != Eigen::Success)
      throw std::runtime_error("CovariantChainHodge::resolvent: the pencil system is singular at this zeta");
    z = lu.solve(c);
  }
  // (ζI − h)^{-1} c = M_k^U (ζ M_k^U − Ã_k^U)^{-1} c
  return Mk * z;
}

CovariantChainHodge::ProjectorRead CovariantChainHodge::projectorOnContour(int k, const Contour &contour,
                                                                           double kappa) const {
  if (contour.nodes.size() != contour.weights.size() || contour.nodes.empty())
    throw std::invalid_argument("CovariantChainHodge::band: a contour needs matching nodes and weights");
  const int n = base_->size(k);
  if (n >= base_->crossoverDimension())
    throw std::length_error("CovariantChainHodge::band: the projector is formed densely below the crossover only");
  ProjectorRead read;
  read.projector = Eigen::MatrixXcd::Zero(n, n);
  const Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(n, n);
  double resolventMax = 0.0;
  for (std::size_t j = 0; j < contour.nodes.size(); ++j) {
    const Eigen::MatrixXcd R = resolvent(k, contour.nodes[j], I);
    read.projector += contour.weights[j] * R;
    Eigen::BDCSVD<Eigen::MatrixXcd> svd(R);
    resolventMax = std::max(resolventMax, svd.singularValues()(0));
  }
  read.certificate.contour = contour.description;
  read.certificate.nodeCount = static_cast<int>(contour.nodes.size());
  read.certificate.resolventMax = resolventMax;
  const double normP = read.projector.norm();
  read.certificate.idempotency =
      normP > 0.0 ? (read.projector * read.projector - read.projector).norm() / normP
                  : (read.projector * read.projector - read.projector).norm();
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(read.projector, Eigen::ComputeThinU);
  const Eigen::VectorXd sv = svd.singularValues();
  const double tol = (sv.size() > 0) ? kappa * static_cast<double>(n) *
                                           std::numeric_limits<double>::epsilon() * sv(0)
                                     : 0.0;
  int r = 0;
  for (int i = 0; i < sv.size(); ++i)
    if (sv(i) > tol) ++r;
  read.certificate.rank = r;
  read.certificate.rankTolerance = tol;
  read.certificate.singularGap = (r >= 1 && r < sv.size() && sv(r) > 0.0)
                                     ? sv(r - 1) / sv(r)
                                     : std::numeric_limits<double>::infinity();
  read.frame = svd.matrixU().leftCols(r);
  return read;
}

Band CovariantChainHodge::band(int k, const Contour &contour, double kappa, double isotropyTolerance) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge::band: Riesz bands on the pencil are the Whitney preset's");
  Band band;
  band.degree = k;
  band.contour = contour;
  ProjectorRead right = projectorOnContour(k, contour, kappa);
  band.projector = std::move(right.projector);
  band.frame = std::move(right.frame);
  band.certificate = right.certificate;
  const int r = static_cast<int>(band.frame.cols());
  // The same contour's band for the dual connection U^{-1}: Phi^vee.
  const CovariantChainHodge dualInstance = dual();
  const ProjectorRead left = dualInstance.projectorOnContour(k, contour, kappa);
  if (static_cast<int>(left.frame.cols()) != r)
    throw std::runtime_error("CovariantChainHodge::band: the dual connection's band has a different rank (" +
                             std::to_string(left.frame.cols()) + " vs " + std::to_string(r) +
                             ") on the same contour");
  band.dualFrame = left.frame;
  band.images = applyG(k, band.frame);                       // Z = G^U Phi
  // The contour's own scale: the largest node modulus, so that a zero band
  // (J = 0) is measured against something rather than against itself.
  double rho = 0.0;
  for (const auto &zeta : contour.nodes) rho = std::max(rho, std::abs(zeta));
  completeBand(band, dualInstance, rho, isotropyTolerance);
  return band;
}

void CovariantChainHodge::completeBand(Band &band, const CovariantChainHodge &dualInstance,
                                       double spectralScale, double isotropyTolerance) const {
  const int k = band.degree;
  const int r = static_cast<int>(band.frame.cols());
  band.pairing = band.dualFrame.transpose() * band.images;   // B_C = (Phi^vee)^T G^U Phi
  if (r > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXcd> bsvd(band.pairing);
    const Eigen::VectorXd bs = bsvd.singularValues();
    band.certificate.detB = band.pairing.determinant();
    band.certificate.condB = (bs(bs.size() - 1) > 0.0) ? bs(0) / bs(bs.size() - 1)
                                                       : std::numeric_limits<double>::infinity();
    // Isotropy is judged on the NORMALIZED pairing: the frames are orthonormal,
    // so sigma_min(B_C) relative to ||G^U Phi||_2 vanishes exactly when a
    // direction of the band is self-orthogonal (rank one: |u^vee^T G u| /
    // ||G u||). A condition number cannot see rank-one isotropy.
    Eigen::BDCSVD<Eigen::MatrixXcd> zsvd(band.images);
    const double imageScale = zsvd.singularValues()(0);
    band.certificate.pairingScale = imageScale > 0.0 ? bs(bs.size() - 1) / imageScale : 0.0;
    if (bs(bs.size() - 1) <= isotropyTolerance * imageScale) {
      band.certificate.leftFrameAvailable = false;
      band.certificate.leftFrameRefusal =
          "isotropic band: det B_C = 0 (the exceptional-point indicator); the canonical left frame "
          "G^{U^-1} Phi^vee B_C^{-T} does not exist and RSF's general biorthogonal machinery is required";
    } else {
      band.certificate.leftFrameAvailable = true;
      // Phi~ = G^{U^-1} Phi^vee B^{-T}: Phi~^T = B^{-1} (G^{U^-1} Phi^vee)^T
      const Eigen::MatrixXcd Y = dualInstance.applyG(k, band.dualFrame);
      band.leftFrame = (band.pairing.partialPivLu().solve(Y.transpose())).transpose();
      const Eigen::MatrixXcd hPhi = applyH(k, band.frame);
      band.reduced = band.leftFrame.transpose() * hPhi;          // J = Phi~^T h Phi
      band.covariance = band.frame * band.leftFrame.transpose();  // Gamma = Phi Phi~^T
      // Residuals relative to the band's own scale, max(||h Phi||, rho ||Phi||)
      // with rho the contour's radius scale: a zero band (J = 0) is not 0/0.
      const double scaleR = std::max(hPhi.norm(), spectralScale * band.frame.norm());
      band.certificate.rightResidual = scaleR > 0.0 ? (hPhi - band.frame * band.reduced).norm() / scaleR
                                                    : (hPhi - band.frame * band.reduced).norm();
      // Phi~^T h = (h^T Phi~)^T with h(U)^T = G^{U^-1} h(U^{-1}) (G^{U^-1})^{-1} (Prop. 5.1 ii).
      const Eigen::MatrixXcd hT_left =
          dualInstance.applyG(k, dualInstance.applyH(k, dualInstance.applyMinv(k, band.leftFrame)));
      const Eigen::MatrixXcd leftH = hT_left.transpose();  // Phi~^T h
      const double scaleL = std::max(leftH.norm(), spectralScale * band.leftFrame.norm());
      band.certificate.leftResidual = scaleL > 0.0 ? (leftH - band.reduced * band.leftFrame.transpose()).norm() / scaleL
                                                   : (leftH - band.reduced * band.leftFrame.transpose()).norm();
    }
  }
}

SparseMatrix CovariantChainHodge::stackedMatrix(int k) const {
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge::stackedMatrix: the stacked matrix of RSF Sec. 5 is "
                           "the Whitney preset's");
  const int d = dimension();
  const int n = base_->size(k);
  // S^U = [ (d_{k+1}^{U^-1})^T ; d_k^U M_k^U ]: the dressed form of
  // ChainHodge::stackedMatrix, whose kernel is G_k^U H_k.
  SparseMatrix top, bottom;
  if (k < d) top = SparseMatrix(twistedDual_[static_cast<std::size_t>(k) + 1].transpose());
  if (k >= 1)
    bottom = SparseMatrix(twisted_[static_cast<std::size_t>(k)] * dressed_[static_cast<std::size_t>(k)]);
  const int rows = static_cast<int>(top.rows() + bottom.rows());
  std::vector<Eigen::Triplet<Complex>> trip;
  trip.reserve(static_cast<std::size_t>(top.nonZeros() + bottom.nonZeros()));
  auto scatter = [&](const SparseMatrix &A, int r0) {
    for (int col = 0; col < A.outerSize(); ++col)
      for (SparseMatrix::InnerIterator it(A, col); it; ++it)
        trip.emplace_back(r0 + static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
  };
  scatter(top, 0);
  scatter(bottom, static_cast<int>(top.rows()));
  SparseMatrix S(rows, n);
  S.setFromTriplets(trip.begin(), trip.end());
  S.makeCompressed();
  return S;
}

HarmonicRead CovariantChainHodge::harmonicChains(int k, double kappa, bool forceSparse) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge::harmonicChains: H_k = M_k^U ker S^U is the Whitney "
                           "preset's");
  const int n = base_->size(k);
  HarmonicRead read;
  read.degree = k;
  const bool dense = !forceSparse && n < base_->crossoverDimension();
  read.dense = dense;
  const SparseMatrix S = stackedMatrix(k);
  Eigen::MatrixXcd kernel;
  if (S.rows() == 0) {
    kernel = Eigen::MatrixXcd::Identity(n, n);
    read.rank = 0;
    read.tolerance = 0.0;
    read.gap = std::numeric_limits<double>::infinity();
  } else if (dense) {
    // The same tolerance policy as ChainHodge::harmonicChains, on the dressed
    // matrix: kappa * max(m, n) * eps * sigma_max.
    const Eigen::MatrixXcd Sd(S);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(Sd, Eigen::ComputeFullV);
    const Eigen::VectorXd sv = svd.singularValues();
    const double tol = kappa * static_cast<double>(std::max(Sd.rows(), Sd.cols()))
                       * std::numeric_limits<double>::epsilon() * sv(0);
    int r = 0;
    for (int i = 0; i < sv.size(); ++i)
      if (sv(i) > tol) ++r;
    read.rank = r;
    read.tolerance = tol;
    kernel = svd.matrixV().rightCols(n - r);
    read.gap = (r < sv.size() && sv(r) > 0.0 && r >= 1)
                   ? sv(r - 1) / sv(r)
                   : std::numeric_limits<double>::infinity();
  } else {
    SparseMatrix ST = SparseMatrix(S.adjoint());  // ker S = range(S^H)^perp
    ST.makeCompressed();
    const Eigen::MatrixXcd Sd(S);
    double colNorm = 0.0;
    for (int c = 0; c < Sd.cols(); ++c) colNorm = std::max(colNorm, Sd.col(c).norm());
    const double tol = kappa * static_cast<double>(std::max(S.rows(), S.cols()))
                       * std::numeric_limits<double>::epsilon() * colNorm;
    Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
    qr.setPivotThreshold(tol);
    qr.compute(ST);
    if (qr.info() != Eigen::Success)
      throw std::runtime_error("CovariantChainHodge::harmonicChains: sparse QR of S^U failed");
    const int r = static_cast<int>(qr.rank());
    read.rank = r;
    read.tolerance = tol;
    read.gap = std::numeric_limits<double>::quiet_NaN();  // the sparse path measures none
    const Eigen::MatrixXcd Q = Eigen::MatrixXcd(qr.matrixQ());
    kernel = Q.rightCols(n - r);
  }
  read.nullity = static_cast<int>(kernel.cols());
  // For the Whitney preset the kernel vectors ARE the geometric images: the
  // chains are H_k = M_k^U ker S^U, and G_k^U H_k = ker S^U back again.
  read.images = kernel;
  read.chains = applyMinv(k, kernel);
  return read;
}

Band CovariantChainHodge::harmonicBand(int k, double kappa, double isotropyTolerance,
                                       bool forceSparse) const {
  if (k < 0 || k > dimension()) throw std::invalid_argument("CovariantChainHodge: degree out of range");
  if (preset() != Preset::L2)
    throw std::logic_error("CovariantChainHodge::harmonicBand: the harmonic band of the pencil is the "
                           "Whitney preset's");
  const HarmonicRead right = harmonicChains(k, kappa, forceSparse);
  const CovariantChainHodge dualInstance = dual();
  const HarmonicRead left = dualInstance.harmonicChains(k, kappa, forceSparse);
  if (left.nullity != right.nullity)
    throw std::runtime_error("CovariantChainHodge::harmonicBand: the dual connection's harmonic space has "
                             "a different dimension (" + std::to_string(left.nullity) + " vs " +
                             std::to_string(right.nullity) +
                             "); the rank conditions (R1)-(R4) of RSF Sec. 5 do not hold here, and the "
                             "lambda = 0 Riesz projector is not the projector onto H_k");
  Band band;
  band.degree = k;
  // The contour has degenerated to the origin: there is none, and a zero node
  // count says so rather than a plausible-looking circle.
  band.contour.description = "harmonic null space of S^U (no contour)";
  band.frame = right.chains;
  band.dualFrame = left.chains;
  band.images = right.images;
  band.certificate.contour = band.contour.description;
  band.certificate.nodeCount = 0;
  band.certificate.rank = right.nullity;
  band.certificate.rankTolerance = right.tolerance;
  band.certificate.singularGap = right.gap;
  // The projector is never formed: idempotency and the resolvent maximum are
  // certificates OF a contour quadrature and have no meaning without one.
  band.certificate.idempotency = std::numeric_limits<double>::quiet_NaN();
  band.certificate.resolventMax = std::numeric_limits<double>::quiet_NaN();
  completeBand(band, dualInstance, 0.0, isotropyTolerance);
  return band;
}

Eigen::MatrixXcd CovariantChainHodge::leftFrame(const Band &band, const CovariantChainHodge &dualInstance,
                                                double isotropyTolerance) {
  const int r = band.rank();
  if (r == 0) return Eigen::MatrixXcd(band.frame.rows(), 0);
  Eigen::JacobiSVD<Eigen::MatrixXcd> bsvd(band.pairing);
  const Eigen::VectorXd bs = bsvd.singularValues();
  Eigen::BDCSVD<Eigen::MatrixXcd> zsvd(band.images);
  const double imageScale = zsvd.singularValues()(0);
  if (bs(bs.size() - 1) <= isotropyTolerance * imageScale)
    throw std::runtime_error("CovariantChainHodge::leftFrame: isotropic band, det B_C = 0 "
                             "(exceptional-point indicator); no canonical left frame");
  const Eigen::MatrixXcd Y = dualInstance.applyG(band.degree, band.dualFrame);
  return (band.pairing.partialPivLu().solve(Y.transpose())).transpose();
}

}  // namespace tessera::chainhodge
