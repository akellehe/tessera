// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Undirected graph in compressed sparse row (CSR) form, with the symmetric
/// normalized Laplacian, modularity, and a heat-kernel spectral dimension.
/// Reference: Burioni & Cassi, "Random walks on graphs: ideas, techniques and
/// results", arXiv:cond-mat/0507142

#include "observables/SparseGraph.h"

#include "graph/CSRBuilder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

SparseGraph SparseGraph::fromCOO(
    const std::vector<std::uint32_t> &rows,
    const std::vector<std::uint32_t> &cols,
    std::uint32_t n) {
  // Dedupe edges with a (min, max) packed-pair key so each undirected edge is
  // represented once.
  std::vector<std::uint64_t> packed;
  packed.reserve(rows.size());
  for (std::size_t k = 0; k < rows.size(); ++k) {
    std::uint32_t a = rows[k], b = cols[k];
    if (a == b) continue;  // ignore self-loops
    std::uint32_t u = std::min(a, b), v = std::max(a, b);
    packed.push_back((static_cast<std::uint64_t>(u) << 32)
                     | static_cast<std::uint64_t>(v));
  }
  std::sort(packed.begin(), packed.end());
  packed.erase(std::unique(packed.begin(), packed.end()), packed.end());

  // Expand to the symmetric coordinate (COO) form the CSR builder expects: each
  // unique edge contributes both directions.
  std::vector<std::uint32_t> rowsSym, colsSym;
  rowsSym.reserve(packed.size() * 2);
  colsSym.reserve(packed.size() * 2);
  for (auto p : packed) {
    std::uint32_t u = static_cast<std::uint32_t>(p >> 32);
    std::uint32_t v = static_cast<std::uint32_t>(p & 0xFFFFFFFFu);
    rowsSym.push_back(u); colsSym.push_back(v);
    rowsSym.push_back(v); colsSym.push_back(u);
  }

  SparseGraph g;
  g.nNodes_ = n;
  g.nEdges_ = packed.size();
  ::tessera::graph::buildCSRFromCOO<std::uint32_t, std::int64_t>(
      static_cast<std::size_t>(n), rowsSym, colsSym, g.indptr_, g.indices_);

  // Precompute D^{-1/2} once; isolated nodes get 0.0, which collapses
  // applyLaplacian to the identity on those rows.
  g.invSqrtDeg_.assign(n, 0.0);
  for (std::uint32_t i = 0; i < n; ++i) {
    auto d = g.degree(i);
    if (d > 0) g.invSqrtDeg_[i] = 1.0 / std::sqrt(static_cast<double>(d));
  }
  return g;
}

void SparseGraph::applyLaplacian(std::vector<double> const &x,
                                    std::vector<double> &y) const {
  // y_i = x_i - (1/sqrt(d_i)) * sum_{j: (i,j) in E} (1/sqrt(d_j)) * x_j
  y.assign(nNodes_, 0.0);
  for (std::size_t i = 0; i < nNodes_; ++i) {
    double s = 0.0;
    auto p = indptr_[i];
    auto q = indptr_[i + 1];
    for (auto k = p; k < q; ++k) {
      std::uint32_t j = indices_[k];
      s += invSqrtDeg_[j] * x[j];
    }
    s *= invSqrtDeg_[i];
    y[i] = x[i] - s;
  }
}

bool SparseGraph::isBipartite() const {
  if (nNodes_ == 0 || nEdges_ == 0) return true;
  std::vector<int> color(nNodes_, -1);
  for (std::uint32_t s = 0; s < nNodes_; ++s) {
    if (color[s] != -1) continue;
    color[s] = 0;
    std::queue<std::uint32_t> q;
    q.push(s);
    while (!q.empty()) {
      auto u = q.front(); q.pop();
      auto p = indptr_[u];
      auto e = indptr_[u + 1];
      for (auto k = p; k < e; ++k) {
        auto v = indices_[k];
        if (color[v] == -1) {
          color[v] = 1 - color[u];
          q.push(v);
        } else if (color[v] == color[u]) {
          return false;
        }
      }
    }
  }
  return true;
}

double SparseGraph::modularity(const std::vector<int> &labels) const {
  if (labels.size() != nNodes_) {
    throw std::invalid_argument(
        "SparseGraph::modularity: labels length must equal nNodes()");
  }
  if (nNodes_ == 0 || nEdges_ == 0) return 0.0;

  // The sum of degrees is 2m (the CSR holds no self-loops), the modularity
  // denominator.
  const double twoM = 2.0 * static_cast<double>(nEdges_);

  // Intra-community edge contribution. The CSR stores each undirected edge in
  // both directions, so this directed scan counts 2·L_c summed over communities,
  // i.e. twoM · Σ_c (L_c/m).
  double intra = 0.0;
  for (std::size_t i = 0; i < nNodes_; ++i) {
    const int ci = labels[i];
    for (auto k = indptr_[i]; k < indptr_[i + 1]; ++k) {
      if (labels[indices_[k]] == ci) intra += 1.0;
    }
  }

  // Summed degree D_c per community.
  std::unordered_map<int, double> degByComm;
  for (std::size_t i = 0; i < nNodes_; ++i) {
    degByComm[labels[i]] += static_cast<double>(degree(static_cast<std::uint32_t>(i)));
  }

  double Q = intra / twoM;
  for (const auto &[comm, dc] : degByComm) {
    const double frac = dc / twoM;
    Q -= frac * frac;
  }
  return Q;
}

std::vector<double> SparseGraph::diagonalHeatKernel(
    const std::vector<std::uint32_t> &starts,
    const std::vector<double> &times,
    int krylovDim) const {
  const std::size_t nStarts = starts.size();
  const std::size_t nT = times.size();
  std::vector<double> out(nStarts * nT, 0.0);
  if (nNodes_ == 0 || nStarts == 0) return out;

  // Edgeless graph: report 1.0 for every (start, t) pair, i.e. no diffusion.
  // Nodes isolated inside an otherwise-connected graph are a different case:
  // there applyLaplacian collapses the symmetric Laplacian to the identity on
  // those rows, giving exp(-σ) diagonals.
  if (nEdges_ == 0) {
    std::fill(out.begin(), out.end(), 1.0);
    return out;
  }

  // Convert the starts to int for the base call; out-of-range entries are
  // marked with a sentinel and their rows are left zero.
  std::vector<int> startsInt;
  startsInt.reserve(nStarts);
  for (auto s : starts) {
    if (s < nNodes_) startsInt.push_back(static_cast<int>(s));
    else             startsInt.push_back(-1);  // sentinel; base skips
  }

  // Compress the sentinel rows out before calling the base, then expand back
  // into `out` so the row order still matches `starts`.
  std::vector<int> validIdx;
  std::vector<int> validStarts;
  validIdx.reserve(nStarts);
  validStarts.reserve(nStarts);
  for (std::size_t i = 0; i < startsInt.size(); ++i) {
    if (startsInt[i] >= 0) {
      validIdx.push_back(static_cast<int>(i));
      validStarts.push_back(startsInt[i]);
    }
  }
  if (validStarts.empty()) return out;

  auto flat = SpectralGraph::diagonalHeatKernel(validStarts, times, krylovDim);
  // Expand back: out[startIdx][j] = flat[validRow][j].
  for (std::size_t r = 0; r < validIdx.size(); ++r) {
    const auto destRow = static_cast<std::size_t>(validIdx[r]);
    for (std::size_t j = 0; j < nT; ++j) {
      out[destRow * nT + j] = flat[r * nT + j];
    }
  }
  return out;
}

std::pair<double, double> SparseGraph::spectralDimension(
    int nWalks, double maxSigma, std::mt19937 *rng,
    double tailFraction, int nTimes, double tMin, int krylovDim) const {
  const double NaN = std::numeric_limits<double>::quiet_NaN();
  if (nNodes_ == 0) return {NaN, NaN};

  int n = std::min<int>(nWalks, static_cast<int>(nNodes_));
  if (n <= 0) return {NaN, NaN};

  // Pick n random starts without replacement.
  std::vector<std::uint32_t> all(nNodes_);
  for (std::uint32_t i = 0; i < nNodes_; ++i) all[i] = i;
  std::shuffle(all.begin(), all.end(), *rng);
  std::vector<std::uint32_t> starts(all.begin(), all.begin() + n);

  // Log-spaced diffusion-time grid on [tMin, maxSigma].
  if (nTimes < 2) return {NaN, NaN};
  std::vector<double> times(nTimes);
  double logMin = std::log(tMin);
  double logMax = std::log(maxSigma);
  for (int j = 0; j < nTimes; ++j) {
    double f = static_cast<double>(j) / (nTimes - 1);
    times[j] = std::exp(logMin + f * (logMax - logMin));
  }

  auto K = diagonalHeatKernel(starts, times, krylovDim);

  // Average the return probability over the starts.
  std::vector<double> Kavg(nTimes, 0.0);
  for (int j = 0; j < nTimes; ++j) {
    double s = 0.0;
    for (int w = 0; w < n; ++w) s += K[static_cast<std::size_t>(w) * nTimes + j];
    Kavg[j] = s / n;
  }

  // Spectral dimension d_s = -2 d(log K) / d(log t) by centred finite
  // differences, skipping non-positive and non-finite samples.
  std::vector<double> logT, logK;
  logT.reserve(nTimes);
  logK.reserve(nTimes);
  for (int j = 0; j < nTimes; ++j) {
    if (Kavg[j] > 0.0 && std::isfinite(Kavg[j])) {
      logT.push_back(std::log(times[j]));
      logK.push_back(std::log(Kavg[j]));
    }
  }
  if (logT.size() < 2) return {NaN, NaN};

  std::vector<double> ds(logT.size());
  for (std::size_t i = 0; i + 1 < logT.size(); ++i) {
    if (i == 0) {
      ds[0] = (logK[1] - logK[0]) / (logT[1] - logT[0]);
    } else {
      ds[i] = (logK[i + 1] - logK[i - 1]) / (logT[i + 1] - logT[i - 1]);
    }
  }
  ds.back() = (logK[logT.size() - 1] - logK[logT.size() - 2])
            / (logT[logT.size() - 1] - logT[logT.size() - 2]);
  for (auto &d : ds) d *= -2.0;

  std::size_t nTail = std::max<std::size_t>(
      1, static_cast<std::size_t>(ds.size() * tailFraction));
  double small = 0.0, large = 0.0;
  for (std::size_t i = 0; i < nTail; ++i) {
    small += ds[i];
    large += ds[ds.size() - 1 - i];
  }
  return {small / nTail, large / nTail};
}

}  // namespace tessera::observables
