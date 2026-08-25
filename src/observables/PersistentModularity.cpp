// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/PersistentModularity.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::observables {

/// Sentinel for "this cell is not in the group currently being bisected",
/// stored in the position map the leading-eigenvector search reuses across
/// groups.  Named rather than spelled at each site: a mistyped literal here
/// would not fail to compile, it would silently include a foreign cell in
/// the modularity matrix.
inline constexpr std::uint32_t kNotInGroup = 0xFFFFFFFFu;

using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

namespace {

/// File-local deterministic hashing / RNG / summation utilities.
struct Mix {
  static std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
  }
  static std::uint64_t bits(double w) noexcept {
    return std::bit_cast<std::uint64_t>(w);
  }
};

/// Two-lane 128-bit accumulator; hex() emits 32 lowercase hex chars.
/// Order-sensitive: callers sort token streams where multiset semantics
/// (relabeling invariance) are required.
class Hash128 {
public:
  void mix(std::uint64_t v) noexcept {
    a_ = Mix::splitmix64(a_ ^ v);
    b_ = Mix::splitmix64(b_ ^ (v * 0xFF51AFD7ED558CCDULL + 0x2545F4914F6CDD1DULL));
  }
  void mixString(const std::string &s) noexcept {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char c : s) {
      h ^= c;
      h *= 0x100000001B3ULL;
    }
    mix(h);
  }
  std::uint64_t lane() const noexcept { return a_; }
  std::string hex() const {
    static const char *digits = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
      out[static_cast<std::size_t>(15 - i)] = digits[(a_ >> (4 * i)) & 0xF];
      out[static_cast<std::size_t>(31 - i)] = digits[(b_ >> (4 * i)) & 0xF];
    }
    return out;
  }

private:
  std::uint64_t a_ = 0x243F6A8885A308D3ULL;
  std::uint64_t b_ = 0x13198A2E03707344ULL;
};

/// Deterministic splitmix64-driven stream: uniform index + Fisher-Yates
/// shuffle.  Self-contained so determinism does not depend on any standard
/// library distribution implementation.
class SeedStream {
public:
  explicit SeedStream(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() noexcept {
    state_ += 0x9E3779B97F4A7C15ULL;
    return Mix::splitmix64(state_);
  }
  template <typename T>
  void shuffle(std::vector<T> &v) noexcept {
    for (std::size_t i = v.size(); i > 1; --i) {
      const std::size_t j = static_cast<std::size_t>(next() % i);
      std::swap(v[i - 1], v[j]);
    }
  }

private:
  std::uint64_t state_;
};

/// Kahan-compensated accumulator for the incremental delta-Q ledger.
class Kahan {
public:
  void add(double x) noexcept {
    const double y = x - c_;
    const double t = s_ + y;
    c_ = (t - s_) - y;
    s_ = t;
  }
  double value() const noexcept { return s_; }

private:
  double s_ = 0.0;
  double c_ = 0.0;
};

}  // namespace

// ───────────────────────── internal structures ──────────────────────────

/// Aggregated weighted graph at one multilevel step.  ``selfW[i]`` follows
/// the A_ii = Sigma_in convention (both ordered directions of the collapsed
/// community's internal weight), so strength[i] = selfW[i] + row sum.
struct PersistentModularity::LevelGraph {
  std::size_t n = 0;
  std::vector<std::int64_t> indptr;
  std::vector<std::uint32_t> indices;
  std::vector<std::complex<double>> weights;
  std::vector<std::complex<double>> selfW;
  // k_C, INHERITED BY SUMMATION when a level is aggregated (k_C = sum over
  // members of k_i) rather than recomputed from the coarse adjacency.  The
  // null model is a property of the level-0 degree sequence.
  std::vector<std::complex<double>> strength;
  std::vector<std::uint32_t> nodeRank;   // canonical visit rank per node
  std::vector<std::string> nodeHash;     // canonical hash per node
};

/// One deterministic restart's full multilevel outcome.
struct PersistentModularity::RunResult {
  std::uint64_t seed = 0;
  // ledger: Q0 + sum of accepted exact delta-Q, and the exact recompute from
  // the final labels.  Both COMPLEX; `objective` is the real scalar the
  // search actually maximized, derived from qCold.
  std::complex<double> qIncremental{0.0, 0.0};
  std::complex<double> qCold{0.0, 0.0};
  double objective = 0.0;
  ModularityObjective objectiveUsed = ModularityObjective::Score;
  std::size_t communities = 0;
  // levelAssign[k][cell] = community index at aggregation level k+1
  // (compact, ordered by (component hash, anchor rank)).
  std::vector<std::vector<std::uint32_t>> levelAssign;
  // levelHashes[k][c] = canonical hash of that community.
  std::vector<std::vector<std::string>> levelHashes;
  std::vector<std::string> sortedFinalHashes;  // equal-score tie-break key
};

// ───────────────────────── construction ─────────────────────────────────

PersistentModularity PersistentModularity::fromWeightedEdges(
    const std::vector<std::uint64_t> &src,
    const std::vector<std::uint64_t> &tgt,
    const std::vector<double> &weight,
    const std::vector<std::uint64_t> &isolatedCells) {
  std::vector<std::complex<double>> lifted;
  lifted.reserve(weight.size());
  for (const double w : weight) lifted.emplace_back(w, 0.0);
  return fromComplexWeightedEdges(src, tgt, lifted, isolatedCells);
}

PersistentModularity PersistentModularity::fromComplexWeightedEdges(
    const std::vector<std::uint64_t> &src,
    const std::vector<std::uint64_t> &tgt,
    const std::vector<std::complex<double>> &weight,
    const std::vector<std::uint64_t> &isolatedCells) {
  if (src.size() != tgt.size() || src.size() != weight.size()) {
    throw std::invalid_argument(
        "PersistentModularity::fromWeightedEdges: src/tgt/weight lengths "
        "differ");
  }
  PersistentModularity g;
  std::unordered_map<std::uint64_t, std::uint32_t> idToIdx;
  idToIdx.reserve(src.size() * 2 + isolatedCells.size());
  auto internIdx = [&](std::uint64_t id) -> std::uint32_t {
    auto it = idToIdx.find(id);
    if (it != idToIdx.end()) return it->second;
    const auto idx = static_cast<std::uint32_t>(g.cellIds_.size());
    idToIdx.emplace(id, idx);
    g.cellIds_.push_back(id);
    return idx;
  };

  // Consolidate parallel edges by weight summation; first-seen orientation
  // is kept.  Self-loops and zero-weight edges are ignored at level 0.
  const std::complex<double> zero(0.0, 0.0);
  std::vector<std::uint32_t> oSrc;
  std::vector<std::uint32_t> oTgt;
  std::vector<std::complex<double>> oW;
  std::unordered_map<std::uint64_t, std::size_t> pairPos;
  pairPos.reserve(src.size());
  for (std::size_t e = 0; e < src.size(); ++e) {
    const std::complex<double> w = weight[e];
    if (!std::isfinite(w.real()) || !std::isfinite(w.imag())) {
      throw std::invalid_argument(
          "PersistentModularity::fromWeightedEdges: weights must be finite in "
          "both components (the domain is the complex weighted graph; signed "
          "and non-real weights are permitted)");
    }
    const std::uint32_t u = internIdx(src[e]);
    const std::uint32_t v = internIdx(tgt[e]);
    if (u == v || w == zero) continue;
    const std::uint64_t key =
        (static_cast<std::uint64_t>(std::min(u, v)) << 32) | std::max(u, v);
    auto it = pairPos.find(key);
    if (it == pairPos.end()) {
      pairPos.emplace(key, oSrc.size());
      oSrc.push_back(u);
      oTgt.push_back(v);
      oW.push_back(w);
    } else {
      oW[it->second] += w;
    }
  }
  for (std::uint64_t id : isolatedCells) internIdx(id);

  // Drop pairs whose consolidated weight is exactly zero.  On a nonnegative
  // list this is unreachable (every summand was already filtered), so that
  // path is untouched; otherwise it removes a pair whose weights CANCELLED,
  // which is a measured absence of net similarity and not an edge we are
  // entitled to keep at weight zero.
  {
    std::vector<std::uint32_t> kSrc;
    std::vector<std::uint32_t> kTgt;
    std::vector<std::complex<double>> kW;
    kSrc.reserve(oSrc.size());
    kTgt.reserve(oTgt.size());
    kW.reserve(oW.size());
    for (std::size_t e = 0; e < oSrc.size(); ++e) {
      if (oW[e] == zero) continue;
      kSrc.push_back(oSrc[e]);
      kTgt.push_back(oTgt[e]);
      kW.push_back(oW[e]);
    }
    oSrc = std::move(kSrc);
    oTgt = std::move(kTgt);
    oW = std::move(kW);
  }

  g.nNodes_ = g.cellIds_.size();
  g.nEdges_ = oSrc.size();
  g.orientedSrc_ = std::move(oSrc);
  g.orientedTgt_ = std::move(oTgt);
  g.orientedW_ = std::move(oW);

  // CSR (both directions per undirected edge).  A is complex SYMMETRIC, so
  // the same value is stored in both directions -- no conjugation, because a
  // weight is a property of the edge rather than of a traversal of it.
  std::vector<std::uint32_t> deg(g.nNodes_, 0);
  for (std::size_t e = 0; e < g.nEdges_; ++e) {
    ++deg[g.orientedSrc_[e]];
    ++deg[g.orientedTgt_[e]];
  }
  g.indptr_.assign(g.nNodes_ + 1, 0);
  for (std::size_t i = 0; i < g.nNodes_; ++i) {
    g.indptr_[i + 1] = g.indptr_[i] + deg[i];
  }
  g.indices_.resize(static_cast<std::size_t>(g.indptr_[g.nNodes_]));
  g.weights_.resize(g.indices_.size());
  std::vector<std::int64_t> cursor(g.indptr_.begin(), g.indptr_.end() - 1);
  for (std::size_t e = 0; e < g.nEdges_; ++e) {
    const std::uint32_t u = g.orientedSrc_[e];
    const std::uint32_t v = g.orientedTgt_[e];
    const std::complex<double> w = g.orientedW_[e];
    g.indices_[static_cast<std::size_t>(cursor[u])] = v;
    g.weights_[static_cast<std::size_t>(cursor[u]++)] = w;
    g.indices_[static_cast<std::size_t>(cursor[v])] = u;
    g.weights_[static_cast<std::size_t>(cursor[v]++)] = w;
  }

  // Branch selectors: the GRAPH decides, never a caller flag.  A wholly
  // nonnegative real graph leaves both false and every scoring path takes the
  // arithmetic it has always taken.
  for (std::size_t e = 0; e < g.nEdges_; ++e) {
    if (g.orientedW_[e].imag() != 0.0) g.complex_ = true;
    if (g.orientedW_[e].imag() != 0.0 || g.orientedW_[e].real() < 0.0) {
      g.signed_ = true;
    }
  }

  g.strength_.assign(g.nNodes_, zero);
  if (!g.signed_) {
    // Verbatim incumbent accumulation: plain double summation in CSR order.
    for (std::size_t i = 0; i < g.nNodes_; ++i) {
      double s = 0.0;
      for (std::int64_t k = g.indptr_[i]; k < g.indptr_[i + 1]; ++k) {
        s += g.weights_[static_cast<std::size_t>(k)].real();
      }
      g.strength_[i] = std::complex<double>(s, 0.0);
    }
    double twoM = 0.0;
    for (const std::complex<double> &s : g.strength_) twoM += s.real();
    g.twoM_ = twoM;
    g.sumA_ = std::complex<double>(twoM, 0.0);
    return g;
  }

  for (std::size_t i = 0; i < g.nNodes_; ++i) {
    Kahan re;
    Kahan im;
    for (std::int64_t k = g.indptr_[i]; k < g.indptr_[i + 1]; ++k) {
      re.add(g.weights_[static_cast<std::size_t>(k)].real());
      im.add(g.weights_[static_cast<std::size_t>(k)].imag());
    }
    g.strength_[i] = std::complex<double>(re.value(), im.value());
  }
  {
    Kahan re;
    Kahan im;
    for (const std::complex<double> &s : g.strength_) {
      re.add(s.real());
      im.add(s.imag());
    }
    g.sumA_ = std::complex<double>(re.value(), im.value());
  }
  {
    // T = sum_ij |A_ij|, the real positive scale.  Accumulated over the CSR
    // (both directions), so it is the full double sum rather than the
    // upper-triangle one.
    Kahan t;
    for (const std::complex<double> &w : g.weights_) t.add(std::abs(w));
    g.twoM_ = t.value();
  }
  return g;
}

PersistentModularity::CausalWeightRead
PersistentModularity::causalWeightAvailability(const Spacetime &st) {
  CausalWeightRead read;
  const auto &edges = st.getEdgeList()->toVector();
  for (const auto &e : edges) {
    if (e->isDegenerate()) {
      ++read.degenerate;
    } else if (e->isSpacelike()) {
      ++read.spacelike;
    } else if (e->isTimelike()) {
      ++read.timelike;
    } else if (e->isNull()) {
      ++read.lightlike;
    } else {
      // Exactly one of Edge's five predicates holds, so whatever is left
      // after the four definite ones is mixed.  Written as the fallthrough
      // rather than as isMixed() so the census is total by construction: no
      // edge can escape being counted.  A mixed edge is an ORDINARY edge for
      // the complex weight map -- its argument is carried as it stands.
      ++read.mixed;
    }
  }
  // Only genuine ABSENCES make the map unreadable.  An indefinite argument is
  // not an absence: the complex weight carries it without classifying it.
  if (read.degenerate > 0) {
    read.available = false;
    read.reason = CausalWeightReason::kDegenerateEdgeLength;
    return read;
  }
  if (edges.empty()) {
    read.available = false;
    read.reason = CausalWeightReason::kNoScorableEdges;
    return read;
  }
  read.available = true;
  return read;
}

PersistentModularity PersistentModularity::fromSpacetime(const Spacetime &st,
                                                         WeightMap map) {
  if (map == WeightMap::CausalPhaseExpNegAbsLength) {
    const CausalWeightRead read = causalWeightAvailability(st);
    if (!read.available) {
      throw std::invalid_argument(
          "PersistentModularity::fromSpacetime: CausalPhaseExpNegAbsLength is "
          "unavailable on this complex (" + read.reason +
          "): an edge with no extent has no argument to carry, and arg(0) is "
          "not a reading of anything. An indefinite argument is NOT a reason "
          "-- the complex weight carries it as it stands. See "
          "causalWeightAvailability for the census.");
    }
  }
  std::vector<std::uint64_t> src;
  std::vector<std::uint64_t> tgt;
  std::vector<std::complex<double>> w;
  const auto &edges = st.getEdgeList()->toVector();
  src.reserve(edges.size());
  tgt.reserve(edges.size());
  w.reserve(edges.size());
  for (const auto &e : edges) {
    src.push_back(e->getSource()->getId());
    tgt.push_back(e->getTarget()->getId());
    switch (map) {
      case WeightMap::Unit:
        w.emplace_back(1.0, 0.0);
        break;
      case WeightMap::ExpNegAbsLength:
        w.emplace_back(std::exp(-std::abs(e->getLength())), 0.0);
        break;
      case WeightMap::CausalPhaseExpNegAbsLength: {
        // Magnitude exp(-|l|) as before; the causal character enters as the
        // ARGUMENT arg(l^2), the same measured quantity Edge classifies
        // dispositions by (#870).  Nothing is bucketed, so a generic
        // argument needs no special case: it lands where it lands.
        const double magnitude = std::exp(-std::abs(e->getLength()));
        // A DEFINITE disposition is placed on its axis exactly, from the same
        // predicate that decided it, rather than through cos/sin of an
        // argument that only rounds to the axis.  sin(pi) is 1.2e-16 in
        // double, not zero, so a timelike edge routed through the general
        // formula would carry a spurious imaginary part -- enough to make a
        // wholly real graph read as complex and take the wrong branch.  The
        // general formula is what MIXED edges need, and they are the ones
        // that get it.
        if (e->isSpacelike()) {
          w.emplace_back(magnitude, 0.0);
        } else if (e->isTimelike()) {
          w.emplace_back(-magnitude, 0.0);
        } else if (e->isNull()) {
          // arg(l^2) = +-pi/2: the sign of the measured argument says which,
          // and the two are conjugates rather than one convention.
          w.emplace_back(0.0, e->squaredArgument() >= 0.0 ? magnitude
                                                          : -magnitude);
        } else {
          const double argument = e->squaredArgument();
          w.emplace_back(magnitude * std::cos(argument),
                         magnitude * std::sin(argument));
        }
        break;
      }
    }
  }
  return fromComplexWeightedEdges(src, tgt, w);
}

std::complex<double> PersistentModularity::modularityGamma(
    const std::vector<int> &labels, double gamma) const {
  if (labels.size() != nNodes_) {
    throw std::invalid_argument(
        "PersistentModularity::modularityGamma: labels.size() != nCells()");
  }
  const std::complex<double> zero(0.0, 0.0);
  if (twoM_ <= 0.0) return zero;
  // Ordered map: community terms combine in ascending label order, which is
  // stable under any relabeling of the underlying cells.
  std::map<int, std::complex<double>> in;   // Sigma_in per label
  std::map<int, std::complex<double>> tot;  // S_c per label
  for (std::size_t e = 0; e < nEdges_; ++e) {
    const int a = labels[orientedSrc_[e]];
    const int b = labels[orientedTgt_[e]];
    if (a == b) in[a] += 2.0 * orientedW_[e];
  }
  for (std::size_t i = 0; i < nNodes_; ++i) tot[labels[i]] += strength_[i];

  if (!signed_) {
    // Verbatim incumbent arithmetic on a nonnegative real graph, so its score
    // is bit-identical rather than merely close.
    Kahan q;
    for (const auto &[label, s] : tot) {
      double lin = 0.0;
      auto it = in.find(label);
      if (it != in.end()) lin = it->second.real();
      const double frac = s.real() / twoM_;
      q.add(lin / twoM_ - gamma * frac * frac);
    }
    return std::complex<double>(q.value(), 0.0);
  }

  // Q = sum_c (Sigma_in(c) - gamma S_c^2 / SA) / T.  SA = 0 leaves the
  // configuration null model undefined -- there is no total weight for it to
  // redistribute -- so it is refused by name rather than silently treated as
  // a vanishing null term.
  if (sumA_ == zero) {
    throw std::invalid_argument(
        "PersistentModularity::modularityGamma: the total adjacency weight "
        "sum_ij A_ij vanishes, so the configuration null model has no weight "
        "to redistribute and Q_gamma is undefined. This is a property of the "
        "graph, not of the partition.");
  }
  const std::complex<double> invSumA = std::complex<double>(1.0, 0.0) / sumA_;
  Kahan re;
  Kahan im;
  for (const auto &[label, s] : tot) {
    std::complex<double> lin = zero;
    auto it = in.find(label);
    if (it != in.end()) lin = it->second;
    const std::complex<double> term = (lin - gamma * s * s * invSumA) / twoM_;
    re.add(term.real());
    im.add(term.imag());
  }
  return std::complex<double>(re.value(), im.value());
}

// ───────────────────────── canonical structure ──────────────────────────

void PersistentModularity::ensureCanonical() const {
  if (canonicalReady_) return;
  const std::size_t n = nNodes_;
  stableColor_.assign(n, 0);
  rank_.assign(n, 0);
  if (n == 0) {
    canonicalReady_ = true;
    return;
  }

  // Initial invariant color: strength summed in ascending weight order so
  // the double bits do not depend on the input edge order.  Complex weights
  // sort lexicographically by (real, imaginary), which is a total order on
  // the stored values and so serves the same purpose -- the ordering is only
  // ever used to fix a summation order, never read as a magnitude.
  std::vector<std::uint64_t> color(n, 0);
  {
    std::vector<std::complex<double>> row;
    for (std::size_t i = 0; i < n; ++i) {
      row.assign(weights_.begin() + static_cast<std::ptrdiff_t>(indptr_[i]),
                 weights_.begin() + static_cast<std::ptrdiff_t>(indptr_[i + 1]));
      std::sort(row.begin(), row.end(),
                [](const std::complex<double> &a,
                   const std::complex<double> &b) {
                  if (a.real() != b.real()) return a.real() < b.real();
                  return a.imag() < b.imag();
                });
      double re = 0.0;
      double im = 0.0;
      for (const std::complex<double> &x : row) {
        re += x.real();
        im += x.imag();
      }
      Hash128 h;
      h.mix(Mix::bits(re));
      h.mix(Mix::bits(im));
      color[i] = Mix::splitmix64(h.lane());
    }
  }

  auto countDistinct = [](const std::vector<std::uint64_t> &c) {
    std::vector<std::uint64_t> tmp(c);
    std::sort(tmp.begin(), tmp.end());
    return static_cast<std::size_t>(
        std::unique(tmp.begin(), tmp.end()) - tmp.begin());
  };

  // Capped iterated color refinement (weighted 1-WL).  Including the old
  // color in the new key means classes never merge, so the distinct count is
  // nondecreasing and stabilization is detectable.
  int maxRounds = 3;
  for (std::size_t t = 1; t < n; t <<= 1) maxRounds += 3;
  auto refine = [&]() {
    std::size_t distinct = countDistinct(color);
    std::vector<std::uint64_t> next(n);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> sig;
    for (int round = 0; round < maxRounds; ++round) {
      for (std::size_t i = 0; i < n; ++i) {
        sig.clear();
        for (std::int64_t k = indptr_[i]; k < indptr_[i + 1]; ++k) {
          Hash128 wh;
          wh.mix(Mix::bits(weights_[static_cast<std::size_t>(k)].real()));
          wh.mix(Mix::bits(weights_[static_cast<std::size_t>(k)].imag()));
          sig.emplace_back(color[indices_[static_cast<std::size_t>(k)]],
                           wh.lane());
        }
        std::sort(sig.begin(), sig.end());
        Hash128 h;
        h.mix(color[i]);
        for (const auto &[c, wb] : sig) {
          h.mix(c);
          h.mix(wb);
        }
        next[i] = h.lane();
      }
      color.swap(next);
      const std::size_t d = countDistinct(color);
      if (d == distinct) break;
      distinct = d;
    }
    return distinct;
  };

  std::size_t distinct = refine();
  stableColor_ = color;  // pre-individualization: pure invariant

  // Individualization-refinement: split remaining tied classes by injecting
  // a fresh color at one representative and mixing in BFS hop distances (a
  // global signal, so rings and other long-range-symmetric graphs resolve in
  // O(n + m) per step instead of O(n) local rounds).  The representative
  // within a structurally indistinguishable class is arbitrary but taken by
  // minimum cell id, so the whole discovery is a pure function of the
  // labeled graph (independent of edge input order); on graphs whose
  // refinement classes are automorphism orbits the resulting order is
  // canonical up to automorphism under relabeling.
  const int maxIndividualize = 64;
  std::vector<std::int64_t> dist(n);
  for (int iter = 0; iter < maxIndividualize && distinct < n; ++iter) {
    // Target class: smallest color value among classes with multiplicity>1
    // (invariant choice); representative: minimum internal index.
    std::unordered_map<std::uint64_t, std::uint32_t> count;
    count.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) ++count[color[i]];
    std::uint64_t targetColor = 0;
    bool found = false;
    for (std::size_t i = 0; i < n; ++i) {
      if (count[color[i]] > 1 && (!found || color[i] < targetColor)) {
        targetColor = color[i];
        found = true;
      }
    }
    if (!found) break;
    std::size_t rep = n;
    for (std::size_t i = 0; i < n; ++i) {
      if (color[i] == targetColor &&
          (rep == n || cellIds_[i] < cellIds_[rep])) {
        rep = i;
      }
    }
    color[rep] = Mix::splitmix64(color[rep] ^ (0xA5A5A5A5A5A5A5A5ULL +
                                               static_cast<std::uint64_t>(iter)));
    // BFS hop distances from the individualized vertex.
    std::fill(dist.begin(), dist.end(), -1);
    std::deque<std::uint32_t> queue;
    dist[rep] = 0;
    queue.push_back(static_cast<std::uint32_t>(rep));
    while (!queue.empty()) {
      const std::uint32_t u = queue.front();
      queue.pop_front();
      for (std::int64_t k = indptr_[u]; k < indptr_[u + 1]; ++k) {
        const std::uint32_t v = indices_[static_cast<std::size_t>(k)];
        if (dist[v] < 0) {
          dist[v] = dist[u] + 1;
          queue.push_back(v);
        }
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t d =
          dist[i] < 0 ? 0xFFFFFFFFFFFFFFFFULL
                      : static_cast<std::uint64_t>(dist[i]);
      color[i] = Mix::splitmix64(color[i] ^ Mix::splitmix64(d));
    }
    distinct = refine();
  }

  // Rank: order by (final color, cell id).  Remaining equal-color ties
  // (fully symmetric classes past the individualization cap, or exact
  // automorphic twins) fall back to the cell id — documented arbitrary
  // representative order, input-order independent.
  std::vector<std::uint32_t> order(n);
  for (std::size_t i = 0; i < n; ++i) order[i] = static_cast<std::uint32_t>(i);
  std::sort(order.begin(), order.end(),
            [&](std::uint32_t x, std::uint32_t y) {
              if (color[x] != color[y]) return color[x] < color[y];
              return cellIds_[x] < cellIds_[y];
            });
  for (std::size_t r = 0; r < n; ++r) rank_[order[r]] = static_cast<std::uint32_t>(r);
  canonicalReady_ = true;
}

// ───────────────────────── shared canonicalization ──────────────────────

void PersistentModularity::canonicalizeCommunities(
    const LevelGraph &g, const std::vector<std::uint32_t> &comm,
    const std::vector<std::vector<std::uint32_t>> &membersOf,
    std::vector<std::vector<std::uint64_t>> &tokens, std::size_t levelNumber,
    std::vector<std::uint32_t> *slotToCompact,
    std::vector<std::string> *hashes) const {
  (void)comm;
  struct CommRec {
    std::string hash;
    std::uint32_t anchorRank;
    std::uint32_t slot;
  };
  const std::size_t slots = membersOf.size();
  std::vector<CommRec> recs;
  for (std::uint32_t c = 0; c < slots; ++c) {
    if (membersOf[c].empty()) continue;
    std::vector<std::string> childHashes;
    childHashes.reserve(membersOf[c].size());
    std::uint32_t anchorRank = 0xFFFFFFFFu;
    for (const std::uint32_t i : membersOf[c]) {
      childHashes.push_back(g.nodeHash[i]);
      anchorRank = std::min(anchorRank, g.nodeRank[i]);
    }
    std::sort(childHashes.begin(), childHashes.end());
    std::sort(tokens[c].begin(), tokens[c].end());
    Hash128 h;
    h.mix(static_cast<std::uint64_t>(levelNumber));
    for (const auto &ch : childHashes) h.mixString(ch);
    h.mix(0x1CEB00DA1CEB00DAULL);  // separator between lineage / incidence
    for (const std::uint64_t t : tokens[c]) h.mix(t);
    recs.push_back(CommRec{h.hex(), anchorRank, c});
  }
  std::sort(recs.begin(), recs.end(), [](const CommRec &x, const CommRec &y) {
    if (x.hash != y.hash) return x.hash < y.hash;
    return x.anchorRank < y.anchorRank;
  });
  slotToCompact->assign(slots, 0xFFFFFFFFu);
  hashes->assign(recs.size(), std::string());
  for (std::size_t c = 0; c < recs.size(); ++c) {
    (*slotToCompact)[recs[c].slot] = static_cast<std::uint32_t>(c);
    (*hashes)[c] = recs[c].hash;
  }
}

// ───────────────────────── one deterministic restart ────────────────────

PersistentModularity::RunResult PersistentModularity::runOnce(
    double gamma, std::uint64_t seed,
    const PersistentModularityConfig &cfg) const {
  ensureCanonical();
  RunResult out;
  out.seed = seed;
  const std::size_t n0 = nNodes_;

  // Base level graph.
  LevelGraph g;
  g.n = n0;
  g.indptr = indptr_;
  g.indices = indices_;
  g.weights = weights_;
  g.selfW.assign(n0, std::complex<double>(0.0, 0.0));
  g.strength = strength_;
  g.nodeRank.resize(n0);
  g.nodeHash.resize(n0);
  for (std::size_t i = 0; i < n0; ++i) {
    g.nodeRank[i] = rank_[i];
    Hash128 h;
    h.mix(0);  // level 0
    h.mix(stableColor_[i]);
    g.nodeHash[i] = h.hex();
  }

  const double twoM = twoM_;
  const bool real = !signed_;
  if (!real && sumA_ == std::complex<double>(0.0, 0.0)) {
    throw std::invalid_argument(
        "PersistentModularity::discover: the total adjacency weight "
        "sum_ij A_ij vanishes, so the configuration null model has no weight "
        "to redistribute and Q_gamma is undefined on this graph.");
  }
  const std::complex<double> invSumA =
      real ? std::complex<double>(0.0, 0.0)
           : std::complex<double>(1.0, 0.0) / sumA_;
  // The real scalar the search maximizes.  `Score` is Q itself and is
  // available only where Q is real; on a complex graph the only ordered
  // choice is the magnitude, and `arg(Q)` carries what kind of structure was
  // found.  Selected here from the GRAPH, and reported on the slice.
  const bool useMagnitude = complex_ ||
                            cfg.objective == ModularityObjective::Magnitude;
  const auto score = [useMagnitude](std::complex<double> q) -> double {
    return useMagnitude ? std::abs(q) : q.real();
  };

  Kahan ledgerRe;
  Kahan ledgerIm;
  // Q of the all-singletons base partition (selfW = 0 at level 0).
  if (twoM > 0.0) {
    if (real) {
      for (std::size_t i = 0; i < n0; ++i) {
        const double frac = strength_[i].real() / twoM;
        ledgerRe.add(-gamma * frac * frac);
      }
    } else {
      for (std::size_t i = 0; i < n0; ++i) {
        const std::complex<double> k = strength_[i];
        const std::complex<double> term = -gamma * k * k * invSumA / twoM;
        ledgerRe.add(term.real());
        ledgerIm.add(term.imag());
      }
    }
  }
  const auto ledgerValue = [&]() {
    return std::complex<double>(ledgerRe.value(), ledgerIm.value());
  };
  const auto ledgerAdd = [&](std::complex<double> d) {
    ledgerRe.add(d.real());
    ledgerIm.add(d.imag());
  };

  // cellAssign[cell0] = node index at the current level.
  std::vector<std::uint32_t> cellAssign(n0);
  for (std::size_t i = 0; i < n0; ++i) {
    cellAssign[i] = static_cast<std::uint32_t>(i);
  }

  SeedStream levelSeeds(seed);
  const int maxLevels = 200;  // termination safety; Q strictly increases
  for (int level = 0; level < maxLevels; ++level) {
    const std::size_t n = g.n;
    // ── local-move sweeps with cached sufficient statistics ──────────────
    std::vector<std::uint32_t> comm(n);
    std::vector<std::complex<double>> S(n);   // community total strength
    std::vector<std::complex<double>> in(n);  // community Sigma_in
    std::vector<std::uint32_t> cnt(n, 1);
    std::vector<std::uint32_t> anchor(n);  // min member rank
    std::vector<bool> anchorDirty(n, false);
    for (std::size_t i = 0; i < n; ++i) {
      comm[i] = static_cast<std::uint32_t>(i);
      S[i] = g.strength[i];
      in[i] = g.selfW[i];
      anchor[i] = g.nodeRank[i];
    }
    std::vector<std::uint32_t> freeSlots;

    // Deterministic visit order: canonical ranks shuffled by the fixed
    // seed stream.
    std::vector<std::uint32_t> visit(n);
    for (std::size_t i = 0; i < n; ++i) visit[i] = static_cast<std::uint32_t>(i);
    std::sort(visit.begin(), visit.end(),
              [&](std::uint32_t x, std::uint32_t y) {
                return g.nodeRank[x] < g.nodeRank[y];
              });
    SeedStream sweepStream(levelSeeds.next());
    sweepStream.shuffle(visit);

    auto anchorOf = [&](std::uint32_t c) -> std::uint32_t {
      if (anchorDirty[c]) {
        std::uint32_t best = 0xFFFFFFFFu;
        for (std::size_t i = 0; i < n; ++i) {
          if (comm[i] == c) best = std::min(best, g.nodeRank[i]);
        }
        anchor[c] = best;
        anchorDirty[c] = false;
      }
      return anchor[c];
    };

    // Scatter buffers for O(deg v) neighbor-community gathering.
    const std::complex<double> zero(0.0, 0.0);
    std::vector<std::complex<double>> wTo(n, zero);
    std::vector<std::uint32_t> touched;
    bool movedAtLevel = false;
    if (twoM > 0.0) {
      for (int pass = 0; pass < cfg.maxSweepsPerLevel; ++pass) {
        bool movedThisPass = false;
        for (const std::uint32_t v : visit) {
          const std::uint32_t a = comm[v];
          touched.clear();
          for (std::int64_t k = g.indptr[v]; k < g.indptr[v + 1]; ++k) {
            const std::uint32_t u = g.indices[static_cast<std::size_t>(k)];
            const std::uint32_t c = comm[u];
            if (wTo[c] == zero) touched.push_back(c);
            wTo[c] += g.weights[static_cast<std::size_t>(k)];
          }
          const std::complex<double> wva = wTo[a];
          const std::complex<double> kv = g.strength[v];
          const std::complex<double> Sa = S[a];
          // Q as it stands, needed because the objective is a nonlinear
          // functional of it once the magnitude is what is maximized: the
          // gain of a move is |Q + dQ| - |Q|, not a function of dQ alone.
          // Reading the ledger is O(1) and it changes only on an accepted
          // move.
          const std::complex<double> qNow = ledgerValue();
          const double scoreNow = score(qNow);
          // Candidate deltas: exact closed form
          //   dQ(a->b) = [2 (w_vb - w_va) - 2 gamma k_v (k_v + S_b - S_a)/SA]
          //              / T,
          // which on a nonnegative real graph is the incumbent's
          //   2 (w_vb - w_va)/2m - 2 gamma k_v (k_v + S_b - S_a)/(2m)^2
          // evaluated in the identical order.
          const auto delta = [&](std::complex<double> wvb,
                                 std::complex<double> Sb) {
            if (real) {
              return std::complex<double>(
                  2.0 * (wvb.real() - wva.real()) / twoM -
                      2.0 * gamma * kv.real() *
                          (kv.real() + Sb.real() - Sa.real()) / (twoM * twoM),
                  0.0);
            }
            return (2.0 * (wvb - wva) -
                    2.0 * gamma * kv * (kv + Sb - Sa) * invSumA) /
                   twoM;
          };
          const auto gainOf = [&](std::complex<double> d) {
            // On the Score objective this is exactly Re(dQ), so the
            // incumbent's comparison is unchanged bit for bit.
            return useMagnitude ? score(qNow + d) - scoreNow : d.real();
          };
          double bestGain = 0.0;  // stay
          std::complex<double> bestDelta = zero;
          std::uint32_t bestTarget = a;
          bool bestIsIsolate = false;
          for (const std::uint32_t c : touched) {
            if (c == a) continue;
            const std::complex<double> d = delta(wTo[c], S[c]);
            const double gain = gainOf(d);
            if (gain > bestGain ||
                (gain == bestGain && bestTarget != a && !bestIsIsolate &&
                 gain > 0.0 && anchorOf(c) < anchorOf(bestTarget))) {
              bestGain = gain;
              bestDelta = d;
              bestTarget = c;
              bestIsIsolate = false;
            }
          }
          if (cnt[a] > 1) {
            // Isolation into a fresh community (S_b = 0, w_vb = 0).  Loses
            // exact ties against any real community and against staying.
            const std::complex<double> d = delta(zero, zero);
            const double gain = gainOf(d);
            if (gain > bestGain) {
              bestGain = gain;
              bestDelta = d;
              bestIsIsolate = true;
            }
          }
          if (bestGain > 0.0) {
            std::uint32_t b;
            if (bestIsIsolate) {
              if (!freeSlots.empty()) {
                b = freeSlots.back();
                freeSlots.pop_back();
              } else {
                b = static_cast<std::uint32_t>(S.size());
                S.push_back(zero);
                in.push_back(zero);
                cnt.push_back(0);
                anchor.push_back(0xFFFFFFFFu);
                anchorDirty.push_back(false);
                wTo.push_back(zero);
              }
              S[b] = zero;
              in[b] = zero;
              cnt[b] = 0;
              anchor[b] = 0xFFFFFFFFu;
              anchorDirty[b] = false;
            } else {
              b = bestTarget;
            }
            const std::complex<double> wvb = bestIsIsolate ? zero : wTo[b];
            in[a] -= 2.0 * wva + g.selfW[v];
            S[a] -= kv;
            cnt[a] -= 1;
            if (anchor[a] == g.nodeRank[v]) anchorDirty[a] = true;
            if (cnt[a] == 0) {
              anchorDirty[a] = false;
              anchor[a] = 0xFFFFFFFFu;
              freeSlots.push_back(a);
            }
            in[b] += 2.0 * wvb + g.selfW[v];
            S[b] += kv;
            cnt[b] += 1;
            if (g.nodeRank[v] < anchor[b] && !anchorDirty[b]) {
              anchor[b] = g.nodeRank[v];
            }
            comm[v] = b;
            ledgerAdd(bestDelta);
            movedThisPass = true;
            movedAtLevel = true;
          }
          for (const std::uint32_t c : touched) wTo[c] = zero;
        }
        if (!movedThisPass) break;
      }
    }

    // No change at a non-base level: the previous snapshot already captured
    // this partition — stop without appending a duplicate.
    if (!movedAtLevel && !out.levelAssign.empty()) break;

    // ── compact communities in (component hash, anchor rank) order ───────
    const std::size_t slots = S.size();
    std::vector<std::vector<std::uint32_t>> membersOf(slots);
    for (std::size_t i = 0; i < n; ++i) {
      membersOf[comm[i]].push_back(static_cast<std::uint32_t>(i));
    }
    // Internal-edge tokens per community for the incidence part of the hash.
    // Level 1 uses the stored (oriented) input incidence; aggregated levels
    // use unordered child-hash pairs (aggregated weights are label-order-
    // sensitive at the bit level and are excluded — the children already
    // carry the exact level-below weight structure).
    std::vector<std::vector<std::uint64_t>> tokens(slots);
    const bool oriented = out.levelAssign.empty();
    if (oriented) {
      for (std::size_t e = 0; e < nEdges_; ++e) {
        const std::uint32_t s = orientedSrc_[e];
        const std::uint32_t t = orientedTgt_[e];
        if (comm[s] != comm[t]) continue;
        Hash128 h;
        h.mixString(g.nodeHash[s]);
        h.mixString(g.nodeHash[t]);
        h.mix(Mix::bits(orientedW_[e].real()));
        h.mix(Mix::bits(orientedW_[e].imag()));
        tokens[comm[s]].push_back(h.lane());
      }
    } else {
      for (std::size_t u = 0; u < n; ++u) {
        for (std::int64_t k = g.indptr[u]; k < g.indptr[u + 1]; ++k) {
          const std::uint32_t v2 = g.indices[static_cast<std::size_t>(k)];
          if (v2 <= u || comm[u] != comm[v2]) continue;
          const std::string &ha = g.nodeHash[u];
          const std::string &hb = g.nodeHash[v2];
          Hash128 h;
          h.mixString(std::min(ha, hb));
          h.mixString(std::max(ha, hb));
          tokens[comm[u]].push_back(h.lane());
        }
      }
    }
    const std::size_t levelNumber = out.levelAssign.size() + 1;
    std::vector<std::uint32_t> slotToCompact;
    std::vector<std::string> hashes;
    canonicalizeCommunities(g, comm, membersOf, tokens, levelNumber,
                            &slotToCompact, &hashes);

    // Snapshot the level-0 assignment for this level.
    std::vector<std::uint32_t> snap(n0);
    for (std::size_t cell = 0; cell < n0; ++cell) {
      snap[cell] = slotToCompact[comm[cellAssign[cell]]];
    }
    out.levelAssign.push_back(snap);
    out.levelHashes.push_back(hashes);
    cellAssign = out.levelAssign.back();

    if (!movedAtLevel) break;
    const std::size_t nSuper = hashes.size();
    if (nSuper == n) {
      // Nothing merged (pure reshuffle at this granularity); the next level
      // would repeat the same graph.
      break;
    }

    // ── aggregate ─────────────────────────────────────────────────────────
    LevelGraph next;
    next.n = nSuper;
    next.selfW.assign(nSuper, zero);
    next.strength.assign(nSuper, zero);
    next.nodeRank.resize(nSuper);
    next.nodeHash.resize(nSuper);
    for (std::size_t c = 0; c < nSuper; ++c) {
      next.nodeRank[c] = static_cast<std::uint32_t>(c);
      next.nodeHash[c] = hashes[c];
    }
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
      if (slotToCompact[slot] == 0xFFFFFFFFu) continue;
      next.selfW[slotToCompact[slot]] = in[slot];
      next.strength[slotToCompact[slot]] = S[slot];
    }
    // Inter-community weights.
    std::unordered_map<std::uint64_t, std::complex<double>> inter;
    inter.reserve(g.indices.size() / 2 + 1);
    for (std::size_t u = 0; u < n; ++u) {
      const std::uint32_t cu = slotToCompact[comm[u]];
      for (std::int64_t k = g.indptr[u]; k < g.indptr[u + 1]; ++k) {
        const std::uint32_t v2 = g.indices[static_cast<std::size_t>(k)];
        if (v2 <= u) continue;
        const std::uint32_t cv = slotToCompact[comm[v2]];
        if (cu == cv) continue;
        const std::uint64_t key =
            (static_cast<std::uint64_t>(std::min(cu, cv)) << 32) |
            std::max(cu, cv);
        inter[key] += g.weights[static_cast<std::size_t>(k)];
      }
    }
    std::vector<std::uint32_t> sdeg(nSuper, 0);
    for (const auto &[key, w] : inter) {
      ++sdeg[static_cast<std::uint32_t>(key >> 32)];
      ++sdeg[static_cast<std::uint32_t>(key & 0xFFFFFFFFu)];
    }
    next.indptr.assign(nSuper + 1, 0);
    for (std::size_t i = 0; i < nSuper; ++i) {
      next.indptr[i + 1] = next.indptr[i] + sdeg[i];
    }
    next.indices.resize(static_cast<std::size_t>(next.indptr[nSuper]));
    next.weights.resize(next.indices.size());
    std::vector<std::int64_t> scursor(next.indptr.begin(),
                                      next.indptr.end() - 1);
    // Deterministic fill order: sorted keys (independent of hash-map order).
    std::vector<std::pair<std::uint64_t, std::complex<double>>> interSorted(
        inter.begin(), inter.end());
    std::sort(interSorted.begin(), interSorted.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    for (const auto &[key, w] : interSorted) {
      const auto cu = static_cast<std::uint32_t>(key >> 32);
      const auto cv = static_cast<std::uint32_t>(key & 0xFFFFFFFFu);
      next.indices[static_cast<std::size_t>(scursor[cu])] = cv;
      next.weights[static_cast<std::size_t>(scursor[cu]++)] = w;
      next.indices[static_cast<std::size_t>(scursor[cv])] = cu;
      next.weights[static_cast<std::size_t>(scursor[cv]++)] = w;
    }
    g = std::move(next);
  }

  out.qIncremental = ledgerValue();
  // Cold exact recompute from the final level-0 labels.
  std::vector<int> labels0(n0, 0);
  if (!out.levelAssign.empty()) {
    const auto &fin = out.levelAssign.back();
    for (std::size_t i = 0; i < n0; ++i) {
      labels0[i] = static_cast<int>(fin[i]);
    }
    out.communities = out.levelHashes.back().size();
    out.sortedFinalHashes = out.levelHashes.back();
    std::sort(out.sortedFinalHashes.begin(), out.sortedFinalHashes.end());
  } else {
    // n0 == 0 or degenerate: empty hierarchy.
    out.communities = 0;
  }
  out.qCold = modularityGamma(labels0, gamma);
  out.objective = score(out.qCold);
  out.objectiveUsed = useMagnitude ? ModularityObjective::Magnitude
                                   : ModularityObjective::Score;
  return out;
}

// ───────────────────────── slice assembly ───────────────────────────────

ResolutionSlice PersistentModularity::buildSlice(
    double gamma, const RunResult &winner,
    std::vector<RestartRead> restarts) const {
  ResolutionSlice slice;
  slice.gamma = gamma;
  slice.q = winner.qCold;
  slice.qIncremental = winner.qIncremental;
  slice.objectiveValue = winner.objective;
  slice.objective = winner.objectiveUsed;
  slice.levels = winner.levelAssign.size();
  slice.restarts = std::move(restarts);
  double qMin = std::numeric_limits<double>::infinity();
  double qMax = -std::numeric_limits<double>::infinity();
  for (const auto &r : slice.restarts) {
    qMin = std::min(qMin, r.objectiveValue);
    qMax = std::max(qMax, r.objectiveValue);
  }
  slice.restartSpread = slice.restarts.empty() ? 0.0 : qMax - qMin;

  const std::complex<double> zero(0.0, 0.0);
  const std::complex<double> invSumA =
      sumA_ == zero ? zero : std::complex<double>(1.0, 0.0) / sumA_;
  const std::size_t n0 = nNodes_;
  for (std::size_t k = 0; k < winner.levelAssign.size(); ++k) {
    const auto &assign = winner.levelAssign[k];
    const auto &hashes = winner.levelHashes[k];
    const std::size_t nc = hashes.size();
    std::vector<std::complex<double>> in(nc, zero);
    std::vector<std::complex<double>> tot(nc, zero);
    std::vector<std::vector<std::uint64_t>> support(nc);
    for (std::size_t e = 0; e < nEdges_; ++e) {
      if (assign[orientedSrc_[e]] == assign[orientedTgt_[e]]) {
        in[assign[orientedSrc_[e]]] += 2.0 * orientedW_[e];
      }
    }
    for (std::size_t i = 0; i < n0; ++i) {
      tot[assign[i]] += strength_[i];
      support[assign[i]].push_back(cellIds_[i]);
    }
    std::vector<ComponentRead> level;
    level.reserve(nc);
    for (std::size_t c = 0; c < nc; ++c) {
      ComponentRead comp;
      comp.id = ComponentId(hashes[c], k + 1);
      std::sort(support[c].begin(), support[c].end());
      comp.support = std::move(support[c]);
      comp.internalWeight = in[c];
      comp.strength = tot[c];
      if (twoM_ > 0.0) {
        if (!signed_) {
          const double cut = tot[c].real() - in[c].real();
          const double denom = std::min(tot[c].real(), twoM_ - tot[c].real());
          comp.conductance = denom > 0.0 ? cut / denom : 0.0;
          const double frac = tot[c].real() / twoM_;
          comp.modularityContribution = std::complex<double>(
              in[c].real() / twoM_ - gamma * frac * frac, 0.0);
        } else {
          // Conductance is a ratio of VOLUMES, and a graph whose weights
          // leave the nonnegative regime has none: the strength of a
          // community is then a signed or complex sum, so "half the total
          // volume" is not a quantity the cut can be compared against.  Left
          // unmeasured rather than computed by a formula that does not
          // apply -- see ComponentRead::conductance.
          comp.conductance = std::numeric_limits<double>::quiet_NaN();
          comp.modularityContribution =
              (in[c] - gamma * tot[c] * tot[c] * invSumA) / twoM_;
        }
      }
      level.push_back(std::move(comp));
    }
    slice.hierarchy.push_back(std::move(level));
  }
  if (!slice.hierarchy.empty()) slice.components = slice.hierarchy.back();
  return slice;
}

// ─────────────────── leading-eigenvector (Newman) search ────────────────

void PersistentModularity::applyGroupModularity(
    const std::vector<std::uint32_t> &group,
    const std::vector<std::uint32_t> &positionOf,
    const std::vector<std::complex<double>> &groupDegree,
    const GroupStrength &groupStrength, double gamma, ModularityPart part,
    const std::vector<double> &x, std::vector<double> *out) const {
  const std::size_t ng = group.size();
  out->assign(ng, 0.0);
  const NullModel nullModel(*this);
  // The null-model term needs its contraction of x over the group once.
  Kahan kxRe;
  Kahan kxIm;
  for (std::size_t p = 0; p < ng; ++p) {
    const std::complex<double> k = strength_[group[p]];
    kxRe.add(k.real() * x[p]);
    kxIm.add(k.imag() * x[p]);
  }
  const std::complex<double> kDotX(kxRe.value(), kxIm.value());
  for (std::size_t p = 0; p < ng; ++p) {
    const std::uint32_t i = group[p];
    Kahan rowRe;
    Kahan rowIm;
    for (std::int64_t k = indptr_[i]; k < indptr_[i + 1]; ++k) {
      const std::uint32_t j = indices_[static_cast<std::size_t>(k)];
      const std::uint32_t q = positionOf[j];
      if (q == kNotInGroup) continue;  // outside the group: excluded by B^g
      const std::complex<double> w = weights_[static_cast<std::size_t>(k)];
      rowRe.add(w.real() * x[q]);
      rowIm.add(w.imag() * x[q]);
    }
    const std::complex<double> row(rowRe.value(), rowIm.value());
    // - gamma (P x)_i   and the diagonal correction that makes B^g
    // annihilate the all-ones vector on the group.  The correction is a row
    // sum, so it is independent of the null model's shape.
    const std::complex<double> diag =
        groupDegree[p] - nullModel.coupling(gamma, i, groupStrength.total);
    const std::complex<double> value =
        row - nullModel.coupling(gamma, i, kDotX) - x[p] * diag;
    (*out)[p] = part == ModularityPart::Real ? value.real() : value.imag();
  }
}

bool PersistentModularity::denseLeadingPair(
    const std::vector<std::uint32_t> &group,
    const std::vector<std::uint32_t> &positionOf,
    const std::vector<std::complex<double>> &groupDegree,
    const GroupStrength &groupStrength, double gamma, ModularityPart part,
    double *first, double *second, std::vector<double> *firstVector,
    double *last, std::vector<double> *lastVector) const {
  const std::size_t ng = group.size();
  if (ng < 2) return false;
  const NullModel nullModel(*this);
  const auto pick = [part](std::complex<double> z) {
    return part == ModularityPart::Real ? z.real() : z.imag();
  };

  // Build the chosen real symmetric part of B^g densely.  Symmetric by
  // construction; the diagonal correction is what makes it annihilate the
  // all-ones vector on the group.
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(
      static_cast<Eigen::Index>(ng), static_cast<Eigen::Index>(ng));
  for (std::size_t p = 0; p < ng; ++p) {
    const std::uint32_t i = group[p];
    for (std::size_t q = 0; q < ng; ++q) {
      const std::uint32_t j = group[q];
      // -gamma P_ij: the null model contracted against column j's indicator.
      b(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)) =
          -pick(nullModel.coupling(gamma, i, strength_[j]));
    }
    for (std::int64_t k = indptr_[i]; k < indptr_[i + 1]; ++k) {
      const std::uint32_t j = indices_[static_cast<std::size_t>(k)];
      const std::uint32_t q = positionOf[j];
      if (q == kNotInGroup) continue;
      b(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)) +=
          pick(weights_[static_cast<std::size_t>(k)]);
    }
    b(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(p)) -=
        pick(groupDegree[p] -
             nullModel.coupling(gamma, i, groupStrength.total));
  }

  // Restrict to the complement of the all-ones vector, which B^g always
  // annihilates: an orthonormal basis of that complement via Householder,
  // so the trivial zero eigenvalue cannot masquerade as the leading one.
  Eigen::VectorXd ones =
      Eigen::VectorXd::Ones(static_cast<Eigen::Index>(ng)) /
      std::sqrt(static_cast<double>(ng));
  Eigen::MatrixXd basis(static_cast<Eigen::Index>(ng),
                        static_cast<Eigen::Index>(ng));
  basis.col(0) = ones;
  basis.rightCols(static_cast<Eigen::Index>(ng) - 1) =
      Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(ng),
                                static_cast<Eigen::Index>(ng))
          .rightCols(static_cast<Eigen::Index>(ng) - 1);
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
  Eigen::MatrixXd q = qr.householderQ();
  Eigen::MatrixXd perp = q.rightCols(static_cast<Eigen::Index>(ng) - 1);
  Eigen::MatrixXd reduced = perp.transpose() * b * perp;
  reduced = 0.5 * (reduced + reduced.transpose().eval());  // exact symmetry

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(reduced);
  if (solver.info() != Eigen::Success) return false;
  const Eigen::VectorXd &values = solver.eigenvalues();  // ascending
  const Eigen::Index lastIndex = values.size() - 1;
  *first = values(lastIndex);
  *second = lastIndex >= 1 ? values(lastIndex - 1)
                           : -std::numeric_limits<double>::infinity();
  const Eigen::VectorXd top = perp * solver.eigenvectors().col(lastIndex);
  firstVector->assign(ng, 0.0);
  for (std::size_t p = 0; p < ng; ++p) {
    (*firstVector)[p] = top(static_cast<Eigen::Index>(p));
  }
  // The MOST NEGATIVE eigenvector too: its sign pattern is the split that
  // most lowers the quadratic form, which is the anti-community candidate.
  // Returned as a candidate only -- what is accepted is decided by exact Q.
  if (last != nullptr && lastVector != nullptr) {
    *last = values(0);
    const Eigen::VectorXd bottom = perp * solver.eigenvectors().col(0);
    lastVector->assign(ng, 0.0);
    for (std::size_t p = 0; p < ng; ++p) {
      (*lastVector)[p] = bottom(static_cast<Eigen::Index>(p));
    }
  }
  return true;
}

bool PersistentModularity::leadingEigenpair(
    const std::vector<std::uint32_t> &group,
    const std::vector<std::uint32_t> &positionOf,
    const std::vector<std::complex<double>> &groupDegree,
    const GroupStrength &groupStrength, double gamma, ModularityPart part,
    const PersistentModularityConfig &cfg, const std::vector<double> *deflate,
    double *eigenvalue, std::vector<double> *eigenvector) const {
  const std::size_t ng = group.size();
  const NullModel nullModel(*this);
  const auto pick = [part](std::complex<double> z) {
    return part == ModularityPart::Real ? z.real() : z.imag();
  };

  // Gershgorin-style bound on the spectral radius of B^g, so B^g + beta I is
  // positive semidefinite and its dominant eigenpair is B^g's MOST POSITIVE
  // one.  Row i's absolute sum is bounded by sum_{j in g} |A_ij| plus the
  // null-model bound plus the diagonal magnitude.  The adjacency term must be
  // the ABSOLUTE row sum: once weights carry sign or phase the group degree
  // is a difference and can sit anywhere below it, so it bounds nothing.  On
  // a nonnegative graph the two coincide, which is why the incumbent could
  // use the group degree directly.
  double beta = 0.0;
  for (std::size_t p = 0; p < ng; ++p) {
    const std::uint32_t i = group[p];
    double rowAbs = groupDegree[p].real();
    if (signed_) {
      Kahan a;
      for (std::int64_t k = indptr_[i]; k < indptr_[i + 1]; ++k) {
        const std::uint32_t j = indices_[static_cast<std::size_t>(k)];
        if (positionOf[j] == kNotInGroup) continue;
        a.add(std::abs(weights_[static_cast<std::size_t>(k)]));
      }
      rowAbs = a.value();
    }
    const double nullBound =
        signed_ ? nullModel.couplingBound(gamma, i,
                                          std::abs(groupStrength.total))
                : nullModel.coupling(gamma, i, groupStrength.total).real();
    const double diag =
        pick(groupDegree[p] -
             nullModel.coupling(gamma, i, groupStrength.total));
    beta = std::max(beta, rowAbs + nullBound + std::abs(diag));
  }
  beta = beta > 0.0 ? beta * 1.0625 : 1.0;  // margin against round-off

  // Deterministic start: a fixed function of the canonical visit rank, so
  // the search carries no seed and no RNG.  The all-ones vector is ALWAYS
  // an exact eigenvector of B^g with eigenvalue zero, so it is projected
  // out at the start and after every step; B^g is symmetric and annihilates
  // it, hence its orthogonal complement is invariant and the projection is
  // exact rather than a correction.
  std::vector<double> v(ng);
  for (std::size_t p = 0; p < ng; ++p) {
    const std::uint64_t r = static_cast<std::uint64_t>(rank_[group[p]]);
    // splitmix64 of the canonical rank, mapped into [-1, 1): label-free and
    // reproducible, and generically not orthogonal to the leading vector.
    const std::uint64_t bits = Mix::splitmix64(r + 0x9E3779B97F4A7C15ULL);
    v[p] = static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0) * 2.0 -
           1.0;
  }
  const auto project = [&](std::vector<double> &x) {
    Kahan s;
    for (const double e : x) s.add(e);
    const double mean = s.value() / static_cast<double>(ng);
    for (double &e : x) e -= mean;
    if (deflate != nullptr) {
      Kahan d;
      for (std::size_t p = 0; p < ng; ++p) d.add(x[p] * (*deflate)[p]);
      const double dot = d.value();
      for (std::size_t p = 0; p < ng; ++p) x[p] -= dot * (*deflate)[p];
    }
  };
  const auto normalize = [&](std::vector<double> &x) {
    Kahan s;
    for (const double e : x) s.add(e * e);
    const double n = std::sqrt(s.value());
    if (n <= 0.0) return false;
    for (double &e : x) e /= n;
    return true;
  };
  project(v);
  if (!normalize(v)) {
    // The start vector collapsed into the projected-out subspace; fall back
    // to a deterministic alternating vector, which is orthogonal to ones for
    // even ng and is re-projected below.
    for (std::size_t p = 0; p < ng; ++p) v[p] = (p % 2 == 0) ? 1.0 : -1.0;
    project(v);
    if (!normalize(v)) return false;
  }

  std::vector<double> w;
  double rayleigh = 0.0;
  double previous = std::numeric_limits<double>::infinity();
  bool converged = false;
  const int maxIterations = std::max(1, cfg.maxPowerIterations);
  for (int it = 0; it < maxIterations; ++it) {
    applyGroupModularity(group, positionOf, groupDegree, groupStrength, gamma,
                         part, v, &w);
    Kahan rq;
    for (std::size_t p = 0; p < ng; ++p) rq.add(v[p] * w[p]);
    rayleigh = rq.value();
    for (std::size_t p = 0; p < ng; ++p) w[p] += beta * v[p];
    project(w);
    if (!normalize(w)) return false;
    v.swap(w);
    const double scale = std::max(1.0, std::abs(rayleigh));
    if (std::abs(rayleigh - previous) <= cfg.powerIterationTolerance * scale) {
      converged = true;
      break;
    }
    previous = rayleigh;
  }
  // One final Rayleigh quotient on the converged vector.
  applyGroupModularity(group, positionOf, groupDegree, groupStrength, gamma,
                       part, v, &w);
  Kahan rq;
  for (std::size_t p = 0; p < ng; ++p) rq.add(v[p] * w[p]);
  *eigenvalue = rq.value();
  *eigenvector = v;
  return converged;
}

void PersistentModularity::refineBisection(
    const std::vector<std::uint32_t> &group,
    const std::vector<std::uint32_t> &positionOf,
    const std::vector<std::complex<double>> &groupDegree,
    const GroupStrength &groupStrength, double gamma, ModularityPart part,
    bool maximize, std::vector<double> *signs) const {
  const std::size_t ng = group.size();
  if (ng < 3) return;
  // delta-Q of the split by s is (1/4m) s^T B^g s.  Flipping s_i changes the
  // quadratic form by -4 s_i (f_i - B_ii s_i) where f = B^g s, so each move
  // is O(1) once f is known; f is refreshed after each accepted move.
  std::vector<double> f;
  std::vector<bool> moved(ng, false);
  std::vector<double> best = *signs;
  double cumulative = 0.0;
  double bestGain = 0.0;
  const NullModel nullModel(*this);
  const auto pick = [part](std::complex<double> z) {
    return part == ModularityPart::Real ? z.real() : z.imag();
  };
  for (std::size_t pass = 0; pass < ng; ++pass) {
    applyGroupModularity(group, positionOf, groupDegree, groupStrength, gamma,
                         part, *signs, &f);
    std::size_t pick_ = ng;
    double pickGain = -std::numeric_limits<double>::infinity();
    for (std::size_t p = 0; p < ng; ++p) {
      if (moved[p]) continue;
      const std::uint32_t i = group[p];
      const double diag =
          pick(groupDegree[p] -
               nullModel.coupling(gamma, i, groupStrength.total));
      // B_ii within B^g: the A_ii term is zero (no self-loops at level 0),
      // minus the null-model diagonal, minus the group-diagonal correction.
      const double bii =
          -pick(nullModel.coupling(gamma, i, strength_[i])) - diag;
      const double raw = -4.0 * (*signs)[p] * (f[p] - bii * (*signs)[p]);
      // Refine in the direction the candidate was selected for: a trailing
      // (anti-community) split improves by LOWERING the quadratic form.
      const double gain = maximize ? raw : -raw;
      if (gain > pickGain) {
        pickGain = gain;
        pick_ = p;
      }
    }
    if (pick_ == ng) break;
    (*signs)[pick_] = -(*signs)[pick_];
    moved[pick_] = true;
    cumulative += pickGain;
    if (cumulative > bestGain) {
      bestGain = cumulative;
      best = *signs;
    }
  }
  *signs = best;  // rewind to the best cumulative point (possibly the input)
}

PersistentModularity::RunResult PersistentModularity::runLeadingEigenvector(
    double gamma, const PersistentModularityConfig &cfg,
    std::vector<SplitRead> *splits) const {
  ensureCanonical();
  RunResult out;
  out.seed = 0;  // no seed: the spectral search is not a sampled restart
  const std::size_t n0 = nNodes_;

  // Base level graph, identical to the aggregation search's, so component
  // identity is hashed by exactly the same rule.
  LevelGraph g;
  g.n = n0;
  g.indptr = indptr_;
  g.indices = indices_;
  g.weights = weights_;
  g.selfW.assign(n0, std::complex<double>(0.0, 0.0));
  g.strength = strength_;
  g.nodeRank.resize(n0);
  g.nodeHash.resize(n0);
  for (std::size_t i = 0; i < n0; ++i) {
    g.nodeRank[i] = rank_[i];
    Hash128 h;
    h.mix(0);
    h.mix(stableColor_[i]);
    g.nodeHash[i] = h.hex();
  }

  // Recursive spectral bisection.  labels[] is the running partition; the
  // queue holds groups still to examine.  Groups are examined in canonical
  // (minimum visit rank) order so the split sequence is label-free.
  std::vector<std::uint32_t> labels(n0, 0);
  std::vector<std::vector<std::uint32_t>> pending;
  {
    std::vector<std::uint32_t> all(n0);
    for (std::size_t i = 0; i < n0; ++i) all[i] = static_cast<std::uint32_t>(i);
    if (!all.empty()) pending.push_back(std::move(all));
  }
  std::vector<double> positionScratch;
  std::vector<std::uint32_t> positionOf(n0, kNotInGroup);
  std::uint32_t nextLabel = 1;

  while (!pending.empty()) {
    // Pop the canonically-first group: smallest member visit rank.
    std::size_t choose = 0;
    std::uint32_t bestRank = 0xFFFFFFFFu;
    for (std::size_t t = 0; t < pending.size(); ++t) {
      std::uint32_t r = 0xFFFFFFFFu;
      for (const std::uint32_t i : pending[t]) r = std::min(r, rank_[i]);
      if (r < bestRank) {
        bestRank = r;
        choose = t;
      }
    }
    std::vector<std::uint32_t> group = std::move(pending[choose]);
    pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(choose));

    SplitRead read;
    read.groupSize = group.size();
    if (group.size() < 2) {
      read.reason = SplitReason::kGroupTooSmall;
      read.resolved = true;
      splits->push_back(std::move(read));
      continue;
    }

    for (std::size_t p = 0; p < group.size(); ++p) {
      positionOf[group[p]] = static_cast<std::uint32_t>(p);
    }
    const auto clearPositions = [&]() {
      for (const std::uint32_t i : group) positionOf[i] = kNotInGroup;
    };

    // k^g and S_g for the generalized modularity matrix.
    std::vector<std::complex<double>> groupDegree(group.size(),
                                                  std::complex<double>(0.0, 0.0));
    Kahan strengthSumRe;
    Kahan strengthSumIm;
    for (std::size_t p = 0; p < group.size(); ++p) {
      const std::uint32_t i = group[p];
      strengthSumRe.add(strength_[i].real());
      strengthSumIm.add(strength_[i].imag());
      Kahan dRe;
      Kahan dIm;
      for (std::int64_t k = indptr_[i]; k < indptr_[i + 1]; ++k) {
        const std::uint32_t j = indices_[static_cast<std::size_t>(k)];
        if (positionOf[j] == kNotInGroup) continue;
        dRe.add(weights_[static_cast<std::size_t>(k)].real());
        dIm.add(weights_[static_cast<std::size_t>(k)].imag());
      }
      groupDegree[p] = std::complex<double>(dRe.value(), dIm.value());
    }
    GroupStrength groupStrength;
    groupStrength.total =
        std::complex<double>(strengthSumRe.value(), strengthSumIm.value());

    // Candidate bisection directions.  On a real graph under the Score
    // objective there is exactly ONE -- the leading eigenvector of Re(B) --
    // and the search is the incumbent's, unchanged.  Otherwise the most
    // NEGATIVE eigenvector is a candidate too (it is the anti-community
    // split, which the most positive one cannot propose), and so is each
    // extreme of Im(B), whose sign pattern separates cells by the ARGUMENT
    // of their coupling rather than its real part.  All are PROPOSALS;
    // acceptance below is by exact Q, so a useless candidate costs one
    // evaluation and nothing else.
    struct Candidate {
      ModularityPart part;
      bool trailing;
    };
    std::vector<Candidate> candidates;
    candidates.push_back({ModularityPart::Real, false});
    const bool wantMagnitude =
        complex_ || cfg.objective == ModularityObjective::Magnitude;
    if (wantMagnitude) candidates.push_back({ModularityPart::Real, true});
    if (complex_) {
      candidates.push_back({ModularityPart::Imaginary, false});
      candidates.push_back({ModularityPart::Imaginary, true});
    }

    const auto objectiveOf = [&](std::complex<double> q) {
      return wantMagnitude ? std::abs(q) : q.real();
    };
    std::vector<int> before(n0);
    for (std::size_t i = 0; i < n0; ++i) before[i] = static_cast<int>(labels[i]);
    const std::complex<double> qBefore = modularityGamma(before, gamma);

    double lambda1 = 0.0;
    double lambda2 = 0.0;
    std::vector<double> v1;
    bool anyPositive = false;
    bool anyDegenerate = false;
    bool notConverged = false;
    bool haveBest = false;
    double bestObjective = objectiveOf(qBefore);
    std::complex<double> bestQ = qBefore;
    std::vector<std::uint32_t> bestA;
    std::vector<std::uint32_t> bestB;

    for (const Candidate &candidate : candidates) {
      double first = 0.0;
      double second = 0.0;
      double lastValue = 0.0;
      std::vector<double> firstVector;
      std::vector<double> lastVector;
      bool haveBoth = false;
      if (group.size() <= cfg.denseEigenSolveMaxGroup) {
        // Exact: no convergence question, which matters precisely because
        // the near-degenerate case the gap adjudicates is where iteration is
        // slowest.  Deciding degeneracy with a method that converges only
        // when the pair is well separated would be circular.
        haveBoth = denseLeadingPair(group, positionOf, groupDegree,
                                    groupStrength, gamma, candidate.part,
                                    &first, &second, &firstVector, &lastValue,
                                    &lastVector);
      }
      if (!haveBoth) {
        if (candidate.trailing) continue;  // no iterative route to the tail
        const bool converged1 = leadingEigenpair(
            group, positionOf, groupDegree, groupStrength, gamma,
            candidate.part, cfg, nullptr, &first, &firstVector);
        if (!converged1) {
          notConverged = true;
          continue;
        }
        std::vector<double> v2;
        const bool converged2 = leadingEigenpair(
            group, positionOf, groupDegree, groupStrength, gamma,
            candidate.part, cfg, &firstVector, &second, &v2);
        if (!converged2) {
          notConverged = true;
          continue;
        }
      }
      const bool isPrimary = candidate.part == ModularityPart::Real &&
                             !candidate.trailing;
      if (isPrimary) {
        lambda1 = first;
        lambda2 = second;
        v1 = firstVector;
      }
      const double value = candidate.trailing ? -lastValue : first;
      if (value <= cfg.leadingEigenvalueTolerance) continue;
      anyPositive = true;
      if (isPrimary && first - second < cfg.minEigenvalueGap) {
        // The leading pair is (near-)degenerate: the eigenvector, and so the
        // sign pattern, is not determined.  Refuse rather than bisect on it.
        anyDegenerate = true;
        continue;
      }
      std::vector<double> signs(group.size());
      const std::vector<double> &source =
          candidate.trailing ? lastVector : firstVector;
      for (std::size_t p = 0; p < group.size(); ++p) {
        signs[p] = source[p] >= 0.0 ? 1.0 : -1.0;
      }
      if (cfg.kernighanLinRefinement) {
        refineBisection(group, positionOf, groupDegree, groupStrength, gamma,
                        candidate.part, !candidate.trailing, &signs);
      }
      std::vector<std::uint32_t> sideA;
      std::vector<std::uint32_t> sideB;
      for (std::size_t p = 0; p < group.size(); ++p) {
        (signs[p] >= 0.0 ? sideA : sideB).push_back(group[p]);
      }
      if (sideA.empty() || sideB.empty()) continue;
      // Accept only on an exact improvement of the SAME closed form the
      // incumbent is scored by, so the two strategies remain comparable.
      std::vector<int> after = before;
      for (const std::uint32_t i : sideB) after[i] = static_cast<int>(nextLabel);
      const std::complex<double> qAfter = modularityGamma(after, gamma);
      const double objectiveAfter = objectiveOf(qAfter);
      if (objectiveAfter > bestObjective) {
        bestObjective = objectiveAfter;
        bestQ = qAfter;
        bestA = std::move(sideA);
        bestB = std::move(sideB);
        haveBest = true;
      }
    }

    read.leadingEigenvalue =
        notConverged && v1.empty() ? std::numeric_limits<double>::quiet_NaN()
                                   : lambda1;
    if (notConverged && !haveBest) {
      read.reason = SplitReason::kPowerIterationNotConverged;
      read.resolved = false;
      splits->push_back(std::move(read));
      clearPositions();
      continue;
    }
    if (!anyPositive) {
      // Newman's stopping rule: no bisection of this group raises Q.
      read.reason = SplitReason::kNoPositiveEigenvalue;
      read.resolved = true;
      splits->push_back(std::move(read));
      clearPositions();
      continue;
    }
    read.secondEigenvalue = lambda2;
    read.eigenvalueGap = lambda1 - lambda2;
    if (!haveBest && anyDegenerate) {
      read.reason = SplitReason::kDegenerateLeadingPair;
      read.resolved = false;
      splits->push_back(std::move(read));
      clearPositions();
      continue;
    }
    if (!haveBest) {
      read.reason = SplitReason::kSplitLowersModularity;
      read.resolved = true;
      splits->push_back(std::move(read));
      clearPositions();
      continue;
    }
    read.deltaQ = bestObjective - objectiveOf(qBefore);
    std::vector<std::uint32_t> sideA = std::move(bestA);
    std::vector<std::uint32_t> sideB = std::move(bestB);

    for (const std::uint32_t i : sideB) labels[i] = nextLabel;
    ++nextLabel;
    read.reason = SplitReason::kSplitAccepted;
    read.resolved = true;
    read.accepted = true;
    read.sizeA = sideA.size();
    read.sizeB = sideB.size();
    splits->push_back(std::move(read));
    clearPositions();
    pending.push_back(std::move(sideA));
    pending.push_back(std::move(sideB));
  }
  (void)positionScratch;

  // Canonicalize the final partition through the SAME identity rule the
  // aggregation search uses, so components from either strategy are
  // comparable and matchable.
  const std::size_t slots = nextLabel;
  std::vector<std::vector<std::uint32_t>> membersOf(slots);
  for (std::size_t i = 0; i < n0; ++i) {
    membersOf[labels[i]].push_back(static_cast<std::uint32_t>(i));
  }
  std::vector<std::vector<std::uint64_t>> tokens(slots);
  for (std::size_t e = 0; e < nEdges_; ++e) {
    const std::uint32_t s = orientedSrc_[e];
    const std::uint32_t t = orientedTgt_[e];
    if (labels[s] != labels[t]) continue;
    Hash128 h;
    h.mixString(g.nodeHash[s]);
    h.mixString(g.nodeHash[t]);
    h.mix(Mix::bits(orientedW_[e].real()));
    h.mix(Mix::bits(orientedW_[e].imag()));
    tokens[labels[s]].push_back(h.lane());
  }
  std::vector<std::uint32_t> slotToCompact;
  std::vector<std::string> hashes;
  canonicalizeCommunities(g, labels, membersOf, tokens, 1, &slotToCompact,
                          &hashes);
  std::vector<std::uint32_t> snap(n0);
  for (std::size_t i = 0; i < n0; ++i) snap[i] = slotToCompact[labels[i]];
  out.levelAssign.push_back(std::move(snap));
  out.levelHashes.push_back(hashes);

  std::vector<int> finalLabels(n0);
  for (std::size_t i = 0; i < n0; ++i) {
    finalLabels[i] = static_cast<int>(out.levelAssign.back()[i]);
  }
  out.qCold = modularityGamma(finalLabels, gamma);
  // The spectral search accumulates no incremental ledger: every accepted
  // split is scored by the exact cold form above, so the two agree by
  // construction rather than by a separate accumulation.
  out.qIncremental = out.qCold;
  const bool usedMagnitude =
      complex_ || cfg.objective == ModularityObjective::Magnitude;
  out.objective = usedMagnitude ? std::abs(out.qCold) : out.qCold.real();
  out.objectiveUsed = usedMagnitude ? ModularityObjective::Magnitude
                                    : ModularityObjective::Score;
  out.communities = hashes.size();
  out.sortedFinalHashes = hashes;
  return out;
}

// ───────────────────────── discovery entry points ───────────────────────

ResolutionSlice PersistentModularity::discover(
    double gamma, const PersistentModularityConfig &cfg) const {
  if (cfg.strategy == DiscoveryStrategy::LeadingEigenvector) {
    std::vector<SplitRead> splits;
    const RunResult run = runLeadingEigenvector(gamma, cfg, &splits);
    ResolutionSlice slice = buildSlice(gamma, run, {});
    slice.strategy = DiscoveryStrategy::LeadingEigenvector;
    slice.splits = std::move(splits);
    // No seed, no restarts: there is no restart spread to report, and
    // unmeasured is never encoded as zero.
    slice.restartSpread = std::numeric_limits<double>::quiet_NaN();
    return slice;
  }
  const int restarts = std::max(1, cfg.restarts);
  std::vector<RunResult> runs;
  std::vector<RestartRead> reads;
  runs.reserve(static_cast<std::size_t>(restarts));
  for (int t = 0; t < restarts; ++t) {
    const std::uint64_t seed =
        Mix::splitmix64(cfg.baseSeed + static_cast<std::uint64_t>(t));
    runs.push_back(runOnce(gamma, seed, cfg));
    reads.push_back(RestartRead{seed, runs.back().qCold,
                                runs.back().objective,
                                runs.back().communities});
  }
  // Winner: best exact score; equal scores broken by the lexicographically
  // smaller sorted component hash list, then by seed order.
  // Winner by the objective actually maximized, not by the complex score --
  // a complex Q has no ordering, which is the whole reason `objective` is a
  // real functional of it.
  std::size_t best = 0;
  for (std::size_t t = 1; t < runs.size(); ++t) {
    if (runs[t].objective > runs[best].objective ||
        (runs[t].objective == runs[best].objective &&
         runs[t].sortedFinalHashes < runs[best].sortedFinalHashes)) {
      best = t;
    }
  }
  return buildSlice(gamma, runs[best], std::move(reads));
}

ScanReport PersistentModularity::scanResolutions(
    const PersistentModularityConfig &cfg) const {
  ScanReport report;
  for (const double gamma : cfg.resolutions) {
    report.slices.push_back(discover(gamma, cfg));
  }

  // Adjacent-slice best matches and persistence tracks (the same chaining
  // rule trackAcrossFrames applies over cobordism time).
  std::vector<const std::vector<ComponentRead> *> steps;
  steps.reserve(report.slices.size());
  for (const auto &slice : report.slices) steps.push_back(&slice.components);
  const std::vector<Chain> chains =
      chainTracks(steps, cfg.overlapThreshold, &report.matches);
  report.tracks.reserve(chains.size());
  for (const Chain &chain : chains) {
    PersistenceTrack t;
    t.members = chain.members;
    t.memberIndices = chain.memberIndices;
    t.firstSlice = chain.first;
    t.lastSlice = chain.last;
    t.gammaFirst = report.slices[chain.first].gamma;
    t.gammaLast = report.slices[chain.last].gamma;
    t.minAdjacentOverlap = chain.minAdjacentOverlap;
    report.tracks.push_back(std::move(t));
  }

  for (auto &t : report.tracks) {
    double c = 0.0;
    for (std::size_t i = 0; i < t.members.size(); ++i) {
      const std::size_t s = t.firstSlice + i;
      c += report.slices[s].components[t.memberIndices[i]].conductance;
    }
    t.meanConductance = t.members.empty()
                            ? 0.0
                            : c / static_cast<double>(t.members.size());
    // weightAwareStatus stays Null: the weight-aware gap / localization /
    // persistence certificates belong to later tickets; unknown is never
    // encoded as zero.
  }
  return report;
}

// ──────────────── tracking, matching, and invalidation ──────────────────

std::vector<PersistentModularity::Chain> PersistentModularity::chainTracks(
    const std::vector<const std::vector<ComponentRead> *> &steps,
    double overlapThreshold, std::vector<ComponentMatch> *matchesOut) const {
  std::vector<Chain> chains;
  struct Active {
    std::size_t chain;
    std::size_t index;  // component index in the current step
  };
  std::vector<Active> active;
  if (steps.empty()) return chains;
  for (std::size_t j = 0; j < steps[0]->size(); ++j) {
    Chain c;
    c.members = {(*steps[0])[j].id};
    c.memberIndices = {j};
    c.first = 0;
    c.last = 0;
    c.minAdjacentOverlap = 1.0;
    chains.push_back(std::move(c));
    active.push_back(Active{chains.size() - 1, j});
  }
  for (std::size_t r = 0; r + 1 < steps.size(); ++r) {
    const auto &a = *steps[r];
    const auto &b = *steps[r + 1];
    const std::vector<ComponentMatch> matches = matchComponents(a, b);
    if (matchesOut != nullptr)
      for (const auto &m : matches) matchesOut->push_back(m);

    // For each b-component pick the continuing chain: the a-side match with
    // the largest overlap >= threshold; ties by hash then index.
    std::vector<std::ptrdiff_t> chosenA(b.size(), -1);
    std::vector<double> chosenOverlap(b.size(), 0.0);
    for (const auto &m : matches) {
      if (m.supportOverlap < overlapThreshold) continue;
      const std::size_t j = m.toIndex;
      const bool better =
          m.supportOverlap > chosenOverlap[j] ||
          (m.supportOverlap == chosenOverlap[j] && chosenA[j] >= 0 &&
           (a[m.fromIndex].id.canonicalHash() <
                a[static_cast<std::size_t>(chosenA[j])].id.canonicalHash() ||
            (a[m.fromIndex].id.canonicalHash() ==
                 a[static_cast<std::size_t>(chosenA[j])].id.canonicalHash() &&
             m.fromIndex < static_cast<std::size_t>(chosenA[j]))));
      if (chosenA[j] < 0 || better) {
        chosenA[j] = static_cast<std::ptrdiff_t>(m.fromIndex);
        chosenOverlap[j] = m.supportOverlap;
      }
    }
    std::vector<Active> nextActive;
    for (std::size_t j = 0; j < b.size(); ++j) {
      bool continued = false;
      if (chosenA[j] >= 0) {
        for (const auto &act : active) {
          if (act.index == static_cast<std::size_t>(chosenA[j])) {
            Chain &c = chains[act.chain];
            c.members.push_back(b[j].id);
            c.memberIndices.push_back(j);
            c.last = r + 1;
            c.minAdjacentOverlap =
                std::min(c.minAdjacentOverlap, chosenOverlap[j]);
            nextActive.push_back(Active{act.chain, j});
            continued = true;
            break;
          }
        }
      }
      if (!continued) {
        Chain c;
        c.members = {b[j].id};
        c.memberIndices = {j};
        c.first = r + 1;
        c.last = r + 1;
        c.minAdjacentOverlap = 1.0;
        chains.push_back(std::move(c));
        nextActive.push_back(Active{chains.size() - 1, j});
      }
    }
    active = std::move(nextActive);
  }
  return chains;
}

std::vector<FrameTrack> PersistentModularity::trackAcrossFrames(
    const std::vector<std::vector<ComponentRead>> &frames,
    double overlapThreshold) const {
  std::vector<const std::vector<ComponentRead> *> steps;
  steps.reserve(frames.size());
  for (const auto &frame : frames) steps.push_back(&frame);
  const std::vector<Chain> chains =
      chainTracks(steps, overlapThreshold, /*matchesOut=*/nullptr);
  std::vector<FrameTrack> tracks;
  tracks.reserve(chains.size());
  for (const Chain &chain : chains) {
    FrameTrack t;
    t.members = chain.members;
    t.memberIndices = chain.memberIndices;
    t.firstFrame = chain.first;
    t.lastFrame = chain.last;
    t.minAdjacentOverlap = chain.minAdjacentOverlap;
    tracks.push_back(std::move(t));
  }
  return tracks;
}

std::vector<ComponentMatch> PersistentModularity::matchComponents(
    const std::vector<ComponentRead> &a,
    const std::vector<ComponentRead> &b) const {
  // Inverted index over b (b supports are disjoint in a partition slice; if
  // a cell appears in several b-components the last one wins — documented
  // partition requirement).
  std::unordered_map<std::uint64_t, std::size_t> cellToB;
  for (std::size_t j = 0; j < b.size(); ++j) {
    for (const std::uint64_t cell : b[j].support) cellToB[cell] = j;
  }
  std::vector<ComponentMatch> out;
  std::vector<double> hits(b.size(), 0.0);
  std::vector<std::size_t> touched;
  for (std::size_t i = 0; i < a.size(); ++i) {
    touched.clear();
    for (const std::uint64_t cell : a[i].support) {
      auto it = cellToB.find(cell);
      if (it == cellToB.end()) continue;
      if (hits[it->second] == 0.0) touched.push_back(it->second);
      hits[it->second] += 1.0;
    }
    std::ptrdiff_t bestJ = -1;
    double bestOverlap = 0.0;
    for (const std::size_t j : touched) {
      const double inter = hits[j];
      const double uni = static_cast<double>(a[i].support.size()) +
                         static_cast<double>(b[j].support.size()) - inter;
      const double overlap = uni > 0.0 ? inter / uni : 0.0;
      const bool better =
          overlap > bestOverlap ||
          (overlap == bestOverlap && bestJ >= 0 &&
           (b[j].id.canonicalHash() <
                b[static_cast<std::size_t>(bestJ)].id.canonicalHash() ||
            (b[j].id.canonicalHash() ==
                 b[static_cast<std::size_t>(bestJ)].id.canonicalHash() &&
             j < static_cast<std::size_t>(bestJ))));
      if (bestJ < 0 || better) {
        bestJ = static_cast<std::ptrdiff_t>(j);
        bestOverlap = overlap;
      }
    }
    for (const std::size_t j : touched) hits[j] = 0.0;
    if (bestJ >= 0 && bestOverlap > 0.0) {
      ComponentMatch m;
      m.from = a[i].id;
      m.to = b[static_cast<std::size_t>(bestJ)].id;
      m.fromIndex = i;
      m.toIndex = static_cast<std::size_t>(bestJ);
      m.supportOverlap = bestOverlap;
      if (projectorHook_) {
        m.projectorOverlap = projectorHook_(m.from, m.to);
      }
      out.push_back(std::move(m));
    }
  }
  return out;
}

InvalidationRead PersistentModularity::invalidatedAncestry(
    const ScanReport &report, const std::vector<std::uint64_t> &touchedCells) {
  InvalidationRead out;
  std::unordered_set<std::uint64_t> touched(touchedCells.begin(),
                                            touchedCells.end());
  auto intersects = [&](const std::vector<std::uint64_t> &support) {
    for (const std::uint64_t cell : support) {
      if (touched.count(cell)) return true;
    }
    return false;
  };
  std::vector<ComponentId> ids;
  for (std::size_t s = 0; s < report.slices.size(); ++s) {
    const auto &slice = report.slices[s];
    for (std::size_t k = 0; k < slice.hierarchy.size(); ++k) {
      for (std::size_t j = 0; j < slice.hierarchy[k].size(); ++j) {
        if (intersects(slice.hierarchy[k][j].support)) {
          out.positions.push_back({s, k, j});
          ids.push_back(slice.hierarchy[k][j].id);
        }
      }
    }
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  out.components = std::move(ids);
  for (std::size_t t = 0; t < report.tracks.size(); ++t) {
    const auto &track = report.tracks[t];
    for (std::size_t i = 0; i < track.memberIndices.size(); ++i) {
      const std::size_t s = track.firstSlice + i;
      const auto &comp =
          report.slices[s].components[track.memberIndices[i]];
      if (intersects(comp.support)) {
        out.tracks.push_back(t);
        break;
      }
    }
  }
  return out;
}

}  // namespace tessera
