// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): the charge
// sector: the E/B split of the field strength. One of the translation units
// that define the class's members by topic.

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
using ::tessera::mesh::ascendingIds;
using ::tessera::mesh::VertexIdPair;

// === Charge sector: the E/B split of the field strength F ∈ Ω² ===

std::vector<cd> EigenstateSynthesis::curvatureFromConnection(
    const std::vector<cd> &A) const {
  if (k_ != 2)
    throw std::runtime_error(
        "EigenstateSynthesis::curvatureFromConnection: requires a degree-2 "
        "instance (the field strength F is a 2-cochain); this instance is "
        "degree " +
        std::to_string(k_));
  // The connection A is a degree-1 cochain in the canonical ChainComplex 1-cell
  // order; map each sorted edge (u,v) to its index so the per-plaquette signed
  // edge sum can read A by edge.
  const auto edges1 = ChainComplex::fromSpacetime(*st_).kSimplexVertices(1);
  if (A.size() != edges1.size())
    throw std::runtime_error(
        "EigenstateSynthesis::curvatureFromConnection: the connection has " +
        std::to_string(A.size()) + " components; the complex has " +
        std::to_string(edges1.size()) + " 1-cells (edges)");
  std::map<std::vector<std::uint64_t>, std::size_t> edgeIdx;
  for (std::size_t i = 0; i < edges1.size(); ++i) edgeIdx[edges1[i]] = i;

  // F = dA on each sorted 2-cell (a,b,c): the induced-orientation signed edge sum
  // +A(a,b) + A(b,c) - A(a,c) — facet (drop v_j) carries (-1)^j, the same boundary
  // convention cyclePeriods uses (drop a -> +(b,c), drop b -> -(a,c), drop c ->
  // +(a,b)).
  std::vector<cd> F(order_, cd(0.0, 0.0));
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) {
    const auto &cell = cellOrdering_[i];  // sorted (a,b,c)
    if (cell.size() != 3)
      throw std::runtime_error(
          "EigenstateSynthesis::curvatureFromConnection: a degree-2 cell has " +
          std::to_string(cell.size()) + " vertices (expected 3)");
    const std::vector<std::vector<std::uint64_t>> facets = {
        {cell[0], cell[1]}, {cell[1], cell[2]}, {cell[0], cell[2]}};
    const double sign[3] = {1.0, 1.0, -1.0};  // +(a,b) +(b,c) -(a,c)
    for (std::size_t j = 0; j < 3; ++j) {
      const auto it = edgeIdx.find(facets[j]);
      if (it == edgeIdx.end())
        throw std::runtime_error(
            "EigenstateSynthesis::curvatureFromConnection: edge (" +
            std::to_string(facets[j][0]) + "," + std::to_string(facets[j][1]) +
            ") of a 2-cell is not a 1-cell of the complex");
      F[i] += sign[j] * A[it->second];
    }
  }
  return F;
}

EigenstateSynthesis::FieldStrengthSplit EigenstateSynthesis::fieldStrengthSplit(
    const std::vector<cd> &F) const {
  if (k_ != 2)
    throw std::runtime_error(
        "EigenstateSynthesis::fieldStrengthSplit: requires a degree-2 instance "
        "(the field strength F is a 2-cochain); this instance is degree " +
        std::to_string(k_));
  if (F.size() != order_)
    throw std::runtime_error(
        "EigenstateSynthesis::fieldStrengthSplit: F has " +
        std::to_string(F.size()) + " components; the degree-2 operator has " +
        std::to_string(order_) + " cells");

  // Map each sorted edge (u,v) to its live Edge* so each plaquette's causal
  // type is read off Edge::isTimelike(), the sanctioned causal test:
  // arg(l^2) ~ +/-pi.
  //
  // The split is binary: a cell is electric if any edge is timelike and magnetic
  // otherwise, so it cannot express a cell whose edges are mixed (off every
  // definite argument). Such a cell lands in `magnetic` rather than being
  // reported as indefinite. Widening the split is out of scope here; the gap is
  // recorded so it is not mistaken for a measurement.
  std::map<VertexIdPair, ::tessera::mesh::Edge *> em;
  for (auto *e : edges_) em[e->ascendingIds()] = e;

  FieldStrengthSplit split;
  split.electric.assign(order_, cd(0.0, 0.0));
  split.magnetic.assign(order_, cd(0.0, 0.0));
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) {
    const auto &cell = cellOrdering_[i];  // sorted (a,b,c)
    const std::pair<std::uint64_t, std::uint64_t> facets[3] = {
        {cell[0], cell[1]}, {cell[1], cell[2]}, {cell[0], cell[2]}};
    // Electric iff any leg is timelike (one temporal index, the discrete F_{0i});
    // else purely-spacelike = magnetic (F_{ij}).
    bool electric = false;
    for (const auto &f : facets) {
      const auto it = em.find(f);
      if (it == em.end())
        throw std::runtime_error(
            "EigenstateSynthesis::fieldStrengthSplit: edge (" +
            std::to_string(f.first) + "," + std::to_string(f.second) +
            ") of a 2-cell is not an edge of the complex");
      if (it->second->isTimelike()) {
        electric = true;
        break;
      }
    }
    if (electric) {
      split.electric[i] = F[i];
      split.electricCells.push_back(i);
    } else {
      split.magnetic[i] = F[i];
      split.magneticCells.push_back(i);
    }
  }
  return split;
}

cd EigenstateSynthesis::gaussLawCharge(
    const std::vector<cd> &F, const std::vector<std::uint64_t> &enclosedVertices,
    bool electricOnly) const {
  if (k_ != 2)
    throw std::runtime_error(
        "EigenstateSynthesis::gaussLawCharge: requires a degree-2 instance (the "
        "field strength F is a 2-cochain); this instance is degree " +
        std::to_string(k_));
  if (F.size() != order_)
    throw std::runtime_error(
        "EigenstateSynthesis::gaussLawCharge: F has " +
        std::to_string(F.size()) + " components; the degree-2 operator has " +
        std::to_string(order_) + " cells");

  // The electric (timelike-leg) plaquettes, the same E/B causal split, so Q
  // lives on the same temporal sector. The magnetic-only Q — the full flux on an
  // all-spacelike complex — is recovered with electricOnly = false.
  const FieldStrengthSplit split = fieldStrengthSplit(F);
  const std::set<std::size_t> electric(split.electricCells.begin(),
                                       split.electricCells.end());

  // Sorted 2-cell tuple -> its component index, so a boundary face of an
  // enclosed tetrahedron maps back to its F entry.
  std::map<std::vector<std::uint64_t>, std::size_t> faceIdx;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i)
    faceIdx[cellOrdering_[i]] = i;

  const std::set<std::uint64_t> enclosed(enclosedVertices.begin(),
                                         enclosedVertices.end());

  // V = the closed star of the enclosed vertices (every tetrahedron touching
  // one). S = boundary 2-chain dV: accumulate each enclosed tetrahedron's four
  // (-1)^j-signed faces; faces interior to V (shared by two V-cells with
  // opposite induced orientation) cancel, leaving the enclosing surface.
  const auto tets = ChainComplex::fromSpacetime(*st_).kSimplexVertices(3);
  std::map<std::vector<std::uint64_t>, double> boundary;
  for (const auto &tet : tets) {
    if (tet.size() != 4) continue;
    bool touches = false;
    for (const std::uint64_t v : tet)
      if (enclosed.count(v)) {
        touches = true;
        break;
      }
    if (!touches) continue;
    for (int j = 0; j < 4; ++j) {  // drop vertex j carries (-1)^j; tet sorted
      std::vector<std::uint64_t> face;
      face.reserve(3);
      for (int q = 0; q < 4; ++q)
        if (q != j) face.push_back(tet[static_cast<std::size_t>(q)]);
      boundary[face] += (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // Q = sum over the surviving surface plaquettes of the orientation-signed F,
  // restricted to the electric (temporal-sector) plaquettes when asked.
  cd Q(0.0, 0.0);
  for (const auto &[face, coeff] : boundary) {
    if (std::abs(coeff) < 1e-12) continue;  // interior face, cancelled
    const auto it = faceIdx.find(face);
    if (it == faceIdx.end()) continue;
    if (electricOnly && electric.find(it->second) == electric.end()) continue;
    Q += coeff * F[it->second];
  }
  return Q;
}

std::vector<cd> EigenstateSynthesis::carriedFromReadout(
    const RegisterReadout &ro, const std::vector<cd> &targetPeriods) const {
  const std::size_t n = order_;
  const std::size_t m = ro.leakColumns.size();
  if (n == 0) return {};
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::carriedFromReadout: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " cycles");

  // The minimum-norm least-squares projection onto the carried period rows.
  const std::vector<cd> c = lstsqOverReadout(ro, targetPeriods);

  // The carried representative plus the minimal leak: psi = sum_r c_r h_r,
  // then each cycle's uncarried remainder lands on its leak column, so the
  // cochain's periods are exactly the targets.
  std::vector<cd> psi(n, cd(0.0, 0.0));
  for (std::size_t r = 0; r < ro.dim; ++r) {
    const cd cr = c[r];
    for (std::size_t i = 0; i < n; ++i) psi[i] += cr * ro.H[r * n + i];
  }
  for (std::size_t q = 0; q < m; ++q) {
    cd carried(0.0, 0.0);
    for (std::size_t r = 0; r < ro.dim; ++r) carried += c[r] * ro.P[r * m + q];
    psi[ro.leakColumns[q]] += targetPeriods[q] - carried;
  }
  return psi;
}

std::vector<cd> EigenstateSynthesis::lstsqOverReadout(
    const RegisterReadout &ro, const std::vector<cd> &targetPeriods) const {
  // c = (P^T)^+ target (minimum-norm least squares, matching numpy.linalg.lstsq):
  // the SVD projection of the targets onto the carried period rows. Shared by
  // carriedFromReadout (r_U's leaked state) and periodGapForLoops (r_psi's
  // period gap), so the two terms fit onto the same carried space.
  const std::size_t m = ro.leakColumns.size();
  if (ro.dim == 0) return {};
  Eigen::VectorXcd t(static_cast<Eigen::Index>(m));
  for (std::size_t q = 0; q < m; ++q)
    t[static_cast<Eigen::Index>(q)] = targetPeriods[q];
  Eigen::MatrixXcd Pt(static_cast<Eigen::Index>(m),
                      static_cast<Eigen::Index>(ro.dim));
  for (std::size_t q = 0; q < m; ++q)
    for (std::size_t r = 0; r < ro.dim; ++r)
      Pt(static_cast<Eigen::Index>(q), static_cast<Eigen::Index>(r)) =
          ro.P[r * m + q];
  const Eigen::VectorXcd c =
      Pt.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(t);
  return std::vector<cd>(c.data(), c.data() + c.size());
}

EigenstateSynthesis::RegisterReadout EigenstateSynthesis::assembleReadoutOverLoops(
    const std::vector<EdgeLoop> &loops) const {
  RegisterReadout out;
  const std::size_t n = order_;
  out.H = readoutHarmonicMatrix();
  if (n == 0) {
    if (!loops.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleReadoutOverLoops: the complex has no "
          "k-cells to carry periods");
    return out;
  }
  if (out.H.size() % n != 0)
    throw std::runtime_error(
        "EigenstateSynthesis::assembleReadoutOverLoops: the harmonic matrix "
        "width " + std::to_string(out.H.size()) +
        " is not a multiple of the captured operator dimension " +
        std::to_string(n));
  out.dim = out.H.size() / n;
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t m = loops.size();
  out.P.assign(out.dim * m, cd(0.0, 0.0));
  out.leakColumns.reserve(m);
  for (std::size_t q = 0; q < m; ++q) {
    if (loops[q].empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleReadoutOverLoops: loop " +
          std::to_string(q) + " is empty");
    bool first = true;
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      const std::vector<std::uint64_t> e = {std::min(a, b), std::max(a, b)};
      const auto it = col.find(e);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::assembleReadoutOverLoops: loop edge (" +
            std::to_string(a) + "," + std::to_string(b) +
            ") is not a k-cell of the complex");
      if (first) {
        out.leakColumns.push_back(it->second);
        first = false;
      }
      for (std::size_t r = 0; r < out.dim; ++r)
        out.P[r * m + q] += s * out.H[r * n + it->second];
    });
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::cyclePeriodsOverLoops(
    const std::vector<EdgeLoop> &loops) const {
  return assembleReadoutOverLoops(loops).P;
}

double EigenstateSynthesis::residualForLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  const std::vector<cd> psi =
      carriedFromReadout(assembleReadoutOverLoops(loops), targetPeriods);
  if (psi.empty()) return 0.0;
  return residual(psi);
}

std::vector<cd> EigenstateSynthesis::carriedRepresentativeOverLoops(
    const std::vector<EdgeLoop> &loops, const std::vector<cd> &targetPeriods) const {
  return carriedFromReadout(assembleReadoutOverLoops(loops), targetPeriods);
}

std::vector<cd> EigenstateSynthesis::periodsOfCochainOverLoops(
    const std::vector<cd> &cochain, const std::vector<EdgeLoop> &loops) const {
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  std::vector<cd> out(loops.size(), cd(0.0, 0.0));
  for (std::size_t q = 0; q < loops.size(); ++q)
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      const std::vector<std::uint64_t> e = {std::min(a, b), std::max(a, b)};
      const auto it = col.find(e);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodsOfCochainOverLoops: loop edge (" +
            std::to_string(a) + "," + std::to_string(b) +
            ") is not a k-cell of the complex");
      if (it->second < cochain.size())
        out[q] += cd(s, 0.0) * cochain[it->second];
    });
  return out;
}

double EigenstateSynthesis::residualForPeriods(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  const std::vector<cd> psi = carriedRepresentative(holes, targetPeriods);
  if (psi.empty()) return 0.0;  // no k-cells to carry periods
  return residual(psi);
}

std::vector<EigenstateSynthesis::EdgeLoop> EigenstateSynthesis::holeLoops(
    const std::vector<std::vector<std::uint64_t>> &holes, const char *who) const {
  auto &vlist = *st_->getVertexList();
  auto edge = [&](std::uint64_t u, std::uint64_t v) {
    auto *vu = vlist.get(u), *vv = vlist.get(v);
    if (!vu || !vv)
      throw std::runtime_error(std::string(who) + ": hole vertex absent");
    return Edge(vu, vv, cd(1.0, 0.0));
  };
  std::vector<EdgeLoop> loops;
  loops.reserve(holes.size());
  for (const auto &h : holes) {
    if (h.size() != 3)
      throw std::runtime_error(std::string(who) + ": hole has " +
                               std::to_string(h.size()) +
                               " vertices, expected 3");
    std::vector<std::uint64_t> s(h);
    std::sort(s.begin(), s.end());
    loops.push_back({edge(s[0], s[1]), edge(s[1], s[2]), edge(s[2], s[0])});
  }
  return loops;
}

double EigenstateSynthesis::periodGapForLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  const std::size_t m = loops.size();
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoops: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  if (m == 0) return 0.0;
  const RegisterReadout ro = assembleReadoutOverLoops(loops);
  // The carried object stays a pure harmonic (no leak): least-squares-fit the
  // target onto the carried period rows (lstsqOverReadout, the same projection
  // r_U's carriedFromReadout uses, so the two terms share the realizable zero
  // set) and return the squared norm of the remainder no harmonic can reach,
  // ||carried - target||^2 = ||P^T c - target||^2. When ro.dim == 0 nothing is
  // carried and the gap is the whole target (c is empty -> carried = 0).
  const std::vector<cd> c = lstsqOverReadout(ro, targetPeriods);
  double gap = 0.0;
  for (std::size_t q = 0; q < m; ++q) {
    cd carried(0.0, 0.0);
    for (std::size_t r = 0; r < ro.dim; ++r) carried += c[r] * ro.P[r * m + q];
    gap += std::norm(carried - targetPeriods[q]);
  }
  return gap;
}

double EigenstateSynthesis::periodGapForPeriods(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  return periodGapForLoops(
      holeLoops(holes, "EigenstateSynthesis::periodGapForPeriods"),
      targetPeriods);
}

EigenstateSynthesis::PeriodGradientContext
EigenstateSynthesis::periodGradientContext() const {
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  PeriodGradientContext ctx;
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::size_t n1 = order_;
  const Index N = static_cast<Index>(n1);
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  ctx.tris = cc.kSimplexVertices(2);
  const std::size_t n2 = ctx.tris.size();
  const std::vector<long> &d1flat = cc.boundaryMatrix(1);  // n0 x n1
  const std::vector<long> &d2flat = cc.boundaryMatrix(2);  // n1 x n2
  const std::size_t n0 = n1 == 0 ? 0 : d1flat.size() / n1;
  ctx.n1 = n1;
  ctx.n2 = n2;

  const HodgeLaplacian hl(st_, HodgeLaplacian::defaultWeightConvention(),
                          metricSource_);
  const std::vector<cd> W1v = hl.weights(1);  // n1, signed complex
  ctx.W2v = hl.weights(2);  // n2, signed complex
  const std::vector<cd> &W2v = ctx.W2v;
  const std::vector<cd> Lflat = hl.laplacian(1, /*metric=*/true);

  ctx.M = MatrixXcd(N, N);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      ctx.M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n1 + j];
  ctx.W1 = VectorXcd(N);
  for (std::size_t i = 0; i < n1; ++i) ctx.W1[static_cast<Index>(i)] = W1v[i];
  MatrixXcd d1m(static_cast<Index>(n0), N);
  for (std::size_t v = 0; v < n0; ++v)
    for (std::size_t c = 0; c < n1; ++c)
      d1m(static_cast<Index>(v), static_cast<Index>(c)) =
          static_cast<double>(d1flat[v * n1 + c]);
  ctx.d2m = MatrixXcd(N, static_cast<Index>(n2));
  for (std::size_t c = 0; c < n1; ++c)
    for (std::size_t t = 0; t < n2; ++t)
      ctx.d2m(static_cast<Index>(c), static_cast<Index>(t)) =
          static_cast<double>(d2flat[c * n2 + t]);
  ctx.K1 = d1m.transpose() * d1m;  // n1 x n1
  VectorXcd W2inv(static_cast<Index>(n2));
  for (std::size_t t = 0; t < n2; ++t)
    W2inv[static_cast<Index>(t)] = 1.0 / W2v[t];
  ctx.K2 = ctx.d2m * W2inv.asDiagonal() * ctx.d2m.transpose();  // n1 x n1

  // ---- index maps: cell -> index, edge -> l^2, edge -> incident triangles ----
  for (std::size_t i = 0; i < n1; ++i)
    ctx.cidx1[ascendingIds(cells1[i][0], cells1[i][1])] = i;
  for (auto *e : edges_)
    ctx.l2map[e->ascendingIds()] =
        (e->getLength() * e->getLength());
  for (std::size_t ti = 0; ti < n2; ++ti)
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        ctx.trisOf[ascendingIds(ctx.tris[ti][i], ctx.tris[ti][j])].push_back(ti);
  return ctx;
}

std::vector<double> EigenstateSynthesis::periodGradientOverLoops(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  // The signed edge-loop machinery is degree-1 by construction: a loop period
  // reads an edge (1-cell) cochain, and everything below (M = L_1, the triangle
  // low-rank dM, the cell->index map over 2-vertex tuples) is the k = 1 layout.
  // Other degrees go through the hole entry points, which route by degree.
  if (k_ != 1)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientOverLoops: the edge-loop core is "
        "degree-1 machinery (loops are closed walks of 1-cells); this "
        "synthesis is degree " + std::to_string(k_) +
        " — use residualForPeriodsGradient, which routes holes by degree.");
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t n1 = order_;
  std::vector<double> grad(n1, 0.0);
  const std::size_t m = loops.size();
  if (n1 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientOverLoops: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n1);

  // The shared degree-1 setup: chain complex, M = L1, the boundary pieces and
  // the index maps. Both edge-loop gradients build the same one.
  const PeriodGradientContext ctx = periodGradientContext();
  const auto &tris = ctx.tris;
  const std::size_t n2 = ctx.n2;
  const Eigen::MatrixXcd &M = ctx.M;
  const Eigen::VectorXcd &W1 = ctx.W1;
  const Eigen::MatrixXcd &d2m = ctx.d2m;
  const Eigen::MatrixXcd &K1 = ctx.K1, &K2 = ctx.K2;
  const std::vector<cd> &W2v = ctx.W2v;
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  const auto &cidx1 = ctx.cidx1;
  const auto &trisOf = ctx.trisOf;
  const auto L2 = [&ctx](std::uint64_t a, std::uint64_t b) {
    return ctx.l2(a, b);
  };

  // ---- Q (signed edge-loop covector) + each cycle's leak column ----
  // Any closed walk of oriented edges, not just a removed triangle's boundary:
  // Q(q, edge) += +1 along the stored orientation, -1 against; the leak is the
  // loop's first edge.
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m);
  for (std::size_t q = 0; q < m; ++q) {
    const EdgeLoop &loop = loops[q];
    if (loop.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::periodGradientOverLoops: loop " +
          std::to_string(q) + " is empty");
    const Edge &fe = loop.front();
    leakCol[q] = cidx1.at(fe.ascendingIds());
    Edge::walkLoop(loop, [&](std::uint64_t a, std::uint64_t b, double s) {
      Q(static_cast<Index>(q),
        static_cast<Index>(cidx1.at(ascendingIds(a, b)))) += s;
    });
  }

  // ---- eigendecomposition of M; harmonic (null) / non-null split ----
  // The signed operator is generally non-self-adjoint (real but non-symmetric
  // on the real-l^2 manifold), so a general eigensolver is required: a
  // self-adjoint one reads a single triangle and silently symmetrizes, which
  // makes this gradient come out identically zero.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i) (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  // Left (dual) basis for the non-self-adjoint perturbation: rows of U^-1 are
  // the covectors v_m with v_m . u_l = delta_ml, which first-order eigenvector
  // perturbation of a non-symmetric M needs in place of U^T.
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);  // 1 / (0 - lambda_nn) for the eigenvector perturbation
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- carried representative psi (via Un / Q / pseudo-inverse), p, rho, r_U ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                            // m x nd
  const MatrixXcd AtAi = (A.transpose() * A).inverse();  // nd x nd
  const VectorXcd c = (AtAi * A.transpose()).cast<cd>() * target;  // nd
  VectorXcd psi = Un.cast<cd>() * c;                    // n1
  const VectorXcd carried = Q.cast<cd>() * psi;         // m
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M.cast<cd>() * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();

  // ---- per-edge analytic gradient d r_U / d l^2 (low-rank dM + perturbation) ----
  for (std::size_t je = 0; je < n1; ++je) {
    const Index j = static_cast<Index>(je);
    const auto ek = ascendingIds(cells1[je][0], cells1[je][1]);
    // dM for the signed operator M = W1^-1 K1 + K2 W1 (K2 = d2 W2^-1 d2^T),
    // under the V^2 weights (the HodgeLaplacian default): W1_j = l^2_j exactly,
    // so dW1_j/dl^2_j = 1 and, every piece rank one,
    //   dM = -(1/W1_j^2) e_j (K1 row j)                       [d(W1^-1) K1]
    //      + (K2 col j) e_j^T                                 [K2 W1 -> K2 dW1]
    //      + per triangle t on e:
    //        -(dW2_t/W2_t^2) (d2 col t)((W1 o d2 col t))^T    [d(W2^-1) term]
    // dM is generally non-symmetric, like M itself.
    VectorXcd dMp;
    MatrixXcd core;
    if (metricSource_ == HodgeLaplacian::MetricSource::WhitneyPencil) {
      // The pencil's exact dense dL_1/dl^2_e (HodgeLaplacian::laplacianGradient
      // under WhitneyPencil); the low-rank diagonal-weight form below does not
      // apply to a non-diagonal metric.
      const std::vector<cd> flat = laplacian_.laplacianGradient(1, cells1[je][0], cells1[je][1]);
      MatrixXcd dM = MatrixXcd::Zero(N, N);
      for (Index a = 0; a < N; ++a)
        for (Index b = 0; b < N; ++b)
          dM(a, b) = flat[static_cast<std::size_t>(a) * static_cast<std::size_t>(N) + static_cast<std::size_t>(b)];
      dMp = dM * p;
      core = Vnn * (dM * Un);
    } else {
      std::vector<VectorXcd> colsA, colsB;
      VectorXcd ev = VectorXcd::Zero(N);
      ev[j] = 1.0;
      colsA.push_back(ev);
      colsB.push_back((-1.0 / (W1[j] * W1[j])) * K1.row(j).transpose());
      colsA.push_back(K2.col(j));
      colsB.push_back(ev);
      const auto trisIt = trisOf.find(ek);
      for (std::size_t ti : (trisIt == trisOf.end()
                                 ? std::vector<std::size_t>{}
                                 : trisIt->second)) {
        const auto &t = tris[ti];
        Eigen::Matrix2cd G;
        for (int i = 0; i < 2; ++i)
          for (int jj = 0; jj < 2; ++jj)
            G(i, jj) = 0.5 * (L2(t[0], t[i + 1]) + L2(t[0], t[jj + 1]) - L2(t[i + 1], t[jj + 1]));
        const cd detG = G.determinant();
        const cd W2ti = W2v[ti];
        // W2 must be the V^2 weight detG/4 this derivation assumes.
        if (std::abs(detG / 4.0 - W2ti) > 1e-9 * std::max(1.0, std::abs(W2ti)) ||
            std::abs(detG) < 1e-12)
          continue;
        auto ind = [&](int pp, int qq) -> double {
          return (pp != qq && ascendingIds(t[pp], t[qq]) == ek) ? 1.0 : 0.0;
        };
        Eigen::Matrix2cd dG;
        for (int i = 0; i < 2; ++i)
          for (int jj = 0; jj < 2; ++jj)
            dG(i, jj) = 0.5 * (ind(0, i + 1) + ind(0, jj + 1) - ind(i + 1, jj + 1));
        // W2 = detG/4 => dW2 = W2 * tr(G^-1 dG), by Jacobi's formula.
        const cd dW2ti = W2ti * (G.inverse() * dG).trace();
        const VectorXcd dcol = d2m.col(static_cast<Index>(ti));
        colsA.push_back(dcol);
        colsB.push_back((-dW2ti / (W2ti * W2ti)) * W1.cwiseProduct(dcol));
      }
      const Index r = static_cast<Index>(colsA.size());
      MatrixXcd fa(N, r), fb(N, r);
      for (Index k = 0; k < r; ++k) {
        fa.col(k) = colsA[static_cast<std::size_t>(k)];
        fb.col(k) = colsB[static_cast<std::size_t>(k)];
      }
      // dM p, the eigenvector perturbation dUn, the pseudo-inverse perturbation, dpsi.
      dMp = fa.cast<cd>() * (fb.transpose().cast<cd>() * p);
      core = (Vnn * fa) * (fb.transpose() * Un);  // nnd x nd
    }
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);               // n1 x nd
    const MatrixXcd dA = Q * dUn;                                           // m x nd
    const MatrixXcd dAplus =
        -AtAi * (dA.transpose() * A + A.transpose() * dA) * AtAi * A.transpose() +
        AtAi * dA.transpose();                                             // nd x m
    const VectorXcd dc = dAplus.cast<cd>() * target;                       // nd
    VectorXcd dpsi = dUn.cast<cd>() * c + Un.cast<cd>() * dc;              // n1
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;                        // m
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];
    const VectorXcd Mdpsi = M.cast<cd>() * dpsi;
    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::periodGradientGeneral(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // k = 0 reads a different operator, not a different weight: the U(1)
  // connection L^U(1) = D - A is complex Hermitian (full l^2 and U(1) phases),
  // so it gets its own complex core rather than the laplacian(k).real()
  // projection below.
  if (k_ == 0) return periodGradientDegreeZero(holes, targetPeriods);
  // Arbitrary-degree exact d r_U / d l^2 over the removed-(k+1)-cell holes.
  // M = L_k, with the per-edge dL_k/dl^2 (HodgeLaplacian::laplacianGradient, on
  // Simplex::volumeGradient) through first-order eigenvector perturbation, and
  // the period covector plus leak from each hole's facet boundary (the
  // assembleRegisterReadout convention). Equals the k = 1 loop core
  // (periodGradientOverLoops) on triangle holes; certified by the Euler identity
  // Sum_e l^2_e d r_U/d l^2_e = -r_U.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t nk = order_;                 // # k-cells (rows/cols of L_k)
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> edges1 = cc.kSimplexVertices(1);
  std::vector<double> grad(edges1.size(), 0.0);  // d r_U / d l^2_e, 1-cell order
  const std::size_t m = holes.size();
  if (nk == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientGeneral: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " holes");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(nk);

  // ---- M = L_k, the signed operator, kept complex: a .real() here would
  // project it, and value and gradient must see the same M ----
  const std::vector<cd> Lflat = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_).laplacian(k_, /*metric=*/true);
  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < nk; ++i)
    for (std::size_t j = 0; j < nk; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * nk + j];

  // ---- Q (period covector, m x nk) + leak column, the assembleRegisterReadout
  // boundary convention: a hole is a removed (k+1)-cell; its facets (drop v_j,
  // sign (-1)^j) are k-cells; the leak is the first facet of the walk. ----
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t hv = static_cast<std::size_t>(k_) + 2;
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m, 0);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::periodGradientGeneral: hole has " +
          std::to_string(h.size()) + " vertices, expected " + std::to_string(hv));
    std::vector<std::size_t> walk(hv);
    for (std::size_t i = 0; i < hv; ++i) walk[i] = i;
    if (k_ == 1) std::rotate(walk.begin(), walk.end() - 1, walk.end());
    for (std::size_t w = 0; w < hv; ++w) {
      const std::size_t j = walk[w];
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodGradientGeneral: a hole facet is not a "
            "k-cell of the complex");
      if (w == 0) leakCol[q] = it->second;
      Q(static_cast<Index>(q), static_cast<Index>(it->second)) +=
          (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // ---- harmonic (null) / non-null eigensplit of M ----
  // The signed operator is generally non-self-adjoint (real but non-symmetric
  // on the real-l^2 manifold), so a general eigensolver is required: a
  // self-adjoint one reads a single triangle and silently symmetrizes, which
  // makes this gradient come out identically zero.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i)
    (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  // Left (dual) basis for the non-self-adjoint perturbation: rows of U^-1 are
  // the covectors v_m with v_m . u_l = delta_ml, which first-order eigenvector
  // perturbation of a non-symmetric M needs in place of U^T.
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- carried representative psi, p, rho, r_U (same as the loop core) ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                            // m x nd
  const MatrixXcd AtAi = (A.transpose() * A).inverse();  // nd x nd
  const VectorXcd c = (AtAi * A.transpose()).cast<cd>() * target;
  VectorXcd psi = Un.cast<cd>() * c;
  const VectorXcd carried = Q.cast<cd>() * psi;
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M.cast<cd>() * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();

  // ---- per-edge analytic gradient: dM_e = laplacianGradient, dense perturbation ----
  const HodgeLaplacian hl(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  for (std::size_t je = 0; je < edges1.size(); ++je) {
    const std::vector<cd> dMflat =
        hl.laplacianGradient(k_, edges1[je][0], edges1[je][1]);
    if (dMflat.empty()) continue;
    MatrixXcd dM(N, N);
    for (std::size_t i = 0; i < nk; ++i)
      for (std::size_t j = 0; j < nk; ++j)
        dM(static_cast<Index>(i), static_cast<Index>(j)) = dMflat[i * nk + j];
    const VectorXcd dMp = dM.cast<cd>() * p;
    // eigenvector perturbation of the harmonic block: dUn = Unn diag(invlam) Unn^T dM Un
    const MatrixXcd core = (Vnn * dM) * Un;                      // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);    // N x nd
    const MatrixXcd dA = Q * dUn;                                // m x nd
    const MatrixXcd dAplus =
        -AtAi * (dA.transpose() * A + A.transpose() * dA) * AtAi * A.transpose() +
        AtAi * dA.transpose();                                  // nd x m
    const VectorXcd dc = dAplus.cast<cd>() * target;
    VectorXcd dpsi = dUn.cast<cd>() * c + Un.cast<cd>() * dc;
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];
    const VectorXcd Mdpsi = M.cast<cd>() * dpsi;
    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::periodGradientDegreeZero(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Exact d r_U / d l^2 at k = 0, against the operator residualForPeriods
  // scores: the complex Hermitian vertex operator L^U(1) = D - A
  // (HodgeLaplacian::connectionLaplacian, D_ii = sum_e |l^2_e|,
  // A_ij = l^2_e e^{i phase_e}). That is the U(1) connection Laplacian, not the
  // Hodge L_0 = d_1 W_1^-1 d_1^T: the degree-zero register carries U(1) flux,
  // and ker L_0 is always b_0, so an L_0 readout would be identically gauge-flat
  // and carry nothing. readoutHarmonicMatrix() picks the same operator, so value
  // and gradient agree.
  // The structure mirrors periodGradientGeneral in complex arithmetic; the
  // product rule d||rho||^2 = 2 Re(rho^dagger d rho) is complex-safe as-is. Two
  // differences from the k >= 1 core, both forced by the operator:
  //   * dL^U(1) per edge has exactly four entries — dL_ii = dL_jj = d|w|/dw
  //     evaluated along the real axis (Re w / |w|; the manifold is real
  //     signed l^2), dL_ij = -e^{i phase}, dL_ji = -e^{-i phase} — and no
  //     volume weights.
  //   * The least-squares fit uses the SVD pseudo-inverse and its constant-rank
  //     derivative (Golub-Pereyra): at k = 0 a globally gauge-flat harmonic has
  //     zero period on every hole, so A = Q U_n is generically
  //     column-rank-deficient and the k >= 1 cores' (A^dagger A)^{-1} would be
  //     singular. The SVD fit is what the functional's lstsqOverReadout
  //     applies, so the gradient differentiates the value returned.
  // Euler identity: L^U(1)(s l^2) = s L^U(1)(l^2) for s > 0 (degree +1), so
  // Sum_e l^2_e d r_U/d l^2_e = +2 r_U (the k >= 1 metric L_k is degree -1,
  // giving -r_U there).
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t n0 = order_;  // # vertices (rows/cols of L^U(1))
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> edges1 = cc.kSimplexVertices(1);
  std::vector<double> grad(edges1.size(), 0.0);  // d r_U / d l^2_e, 1-cell order
  const std::size_t m = holes.size();
  if (n0 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGradientDegreeZero: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " holes");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n0);

  // ---- M = L^U(1) (Hermitian complex; the full l^2 and U(1) phases) ----
  const std::vector<cd> Lflat = readoutLaplacian();
  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t j = 0; j < n0; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * n0 + j];

  // ---- Q (period covector, m x n0) + leak column: a k = 0 hole is a removed
  // 1-cell (a vertex pair); its drop-v_j facets are the two vertices with sign
  // (-1)^j — the assembleRegisterReadout convention, leak on the first facet. ----
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t hv = 2;  // k + 2 vertices per hole at k = 0
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  std::vector<std::size_t> leakCol(m, 0);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::periodGradientDegreeZero: hole has " +
          std::to_string(h.size()) + " vertices, expected " +
          std::to_string(hv));
    for (std::size_t j = 0; j < hv; ++j) {
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodGradientDegreeZero: a hole vertex is "
            "not a 0-cell of the complex");
      if (j == 0) leakCol[q] = it->second;
      Q(static_cast<Index>(q), static_cast<Index>(it->second)) +=
          (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // ---- harmonic (null) / non-null eigensplit of M (Hermitian ⇒ real λ) ----
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i)
    (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  VectorXcd invlam(nnd);
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- the SVD pseudo-inverse fit (min-norm, as lstsqOverReadout) ----
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q.cast<cd>() * Un;  // m x nd
  Eigen::JacobiSVD<MatrixXcd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Index rank = svd.rank();
  MatrixXcd Aplus = MatrixXcd::Zero(nd, static_cast<Index>(m));
  if (rank > 0) {
    const MatrixXcd Ur = svd.matrixU().leftCols(rank);
    const MatrixXcd Vr = svd.matrixV().leftCols(rank);
    const VectorXcd sr = svd.singularValues().head(rank);
    Aplus = Vr * sr.cwiseInverse().asDiagonal() * Ur.adjoint();
  }
  const VectorXcd c = Aplus * target;

  // ---- carried representative psi, p, rho, r_U (as the k >= 1 cores) ----
  VectorXcd psi = Un * c;
  const VectorXcd carried = Q.cast<cd>() * psi;
  for (std::size_t q = 0; q < m; ++q)
    psi[static_cast<Index>(leakCol[q])] +=
        target[static_cast<Index>(q)] - carried[static_cast<Index>(q)];
  const double nrm = psi.norm();
  if (nrm <= 0.0) return grad;
  const VectorXcd p = psi / nrm;
  const VectorXcd Mp = M * p;
  const double lamR = (p.dot(Mp)).real();
  const VectorXcd rho = Mp - lamR * p;
  const double rU = rho.squaredNorm();

  // Constant-rank pseudo-inverse pieces: the range/row-space projectors and
  // the two Gram factors of the Golub–Pereyra derivative.
  const MatrixXcd Im_AAp =
      MatrixXcd::Identity(static_cast<Index>(m), static_cast<Index>(m)) - A * Aplus;
  const MatrixXcd Ind_ApA = MatrixXcd::Identity(nd, nd) - Aplus * A;
  const MatrixXcd ApApAdj = Aplus * Aplus.adjoint();   // nd x nd
  const MatrixXcd ApAdjAp = Aplus.adjoint() * Aplus;   // m x m

  // ---- per-edge index/value lookups ----
  std::map<VertexIdPair, const Edge *> edgeOf;
  for (const auto *e : edges_) edgeOf[e->ascendingIds()] = e;

  // ---- per-edge analytic gradient: the four-entry dL^U(1), dense perturbation ----
  for (std::size_t je = 0; je < edges1.size(); ++je) {
    const auto eIt = edgeOf.find(ascendingIds(edges1[je][0], edges1[je][1]));
    if (eIt == edgeOf.end()) continue;  // no live edge carries this 1-cell
    const Edge *edge = eIt->second;
    const auto srcIt = col.find({edge->getSource()->getId()});
    const auto tgtIt = col.find({edge->getTarget()->getId()});
    if (srcIt == col.end() || tgtIt == col.end()) continue;
    const Index is = static_cast<Index>(srcIt->second);
    const Index it = static_cast<Index>(tgtIt->second);
    const cd w = (edge->getLength() * edge->getLength());
    // d|w|/d(Re w): Re w / |w| — the real-axis directional derivative (sign w
    // for real w). At the |w| kink (w == 0) take the symmetric subgradient 0.
    const double dAbs = (std::abs(w) > 0.0) ? w.real() / std::abs(w) : 0.0;
    const cd zPhase = std::exp(cd(0.0, 1.0) * edge->getPhase());
    MatrixXcd dM = MatrixXcd::Zero(N, N);
    dM(is, is) += dAbs;                  // dD_ii
    dM(it, it) += dAbs;                  // dD_jj
    dM(is, it) -= zPhase;                // -dA_ij
    dM(it, is) -= std::conj(zPhase);     // -dA_ji (Hermitian)
    const VectorXcd dMp = dM * p;
    // Eigenvector perturbation of the harmonic block (complex Hermitian):
    // dUn = Unn diag(1/(0 - λ_nn)) Unn^dagger dM Un.
    const MatrixXcd core = (Unn.adjoint() * dM) * Un;         // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core); // N x nd
    const MatrixXcd dA = Q.cast<cd>() * dUn;                  // m x nd
    // Constant-rank derivative of the pseudo-inverse (Golub–Pereyra):
    // dA+ = -A+ dA A+ + (A+ A+^dagger) dA^dagger (I - A A+)
    //       + (I - A+ A) dA^dagger (A+^dagger A+).
    const MatrixXcd dAplus = -Aplus * dA * Aplus +
                             ApApAdj * dA.adjoint() * Im_AAp +
                             Ind_ApA * dA.adjoint() * ApAdjAp;
    const VectorXcd dc = dAplus * target;
    VectorXcd dpsi = dUn * c + Un * dc;
    const VectorXcd dcarried = Q.cast<cd>() * dpsi;
    for (std::size_t q = 0; q < m; ++q)
      dpsi[static_cast<Index>(leakCol[q])] += -dcarried[static_cast<Index>(q)];
    const VectorXcd Mdpsi = M * dpsi;
    grad[je] = 2.0 * (rho.dot(dMp)).real() +
               (2.0 / nrm) * (rho.dot(Mdpsi - lamR * dpsi)).real() -
               (2.0 * rU / nrm) * (p.dot(dpsi)).real();
  }
  return grad;
}

std::vector<double> EigenstateSynthesis::residualForPeriodsGradient(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Arbitrary-degree exact d r_U / d l^2, in ChainComplex 1-cell (edge) order.
  // At k = 1 (triangle holes) this routes through the fast low-rank edge-loop
  // core (periodGradientOverLoops), which builds the chain complex once and uses
  // a per-edge low-rank dM, so a relaxation loop stays affordable; it is
  // value-identical to the general path (verified to 1.7e-15). For k >= 2 the
  // degree-generic periodGradientGeneral is used (M = L_k, the per-edge analytic
  // dL_k/dl^2). Both satisfy the exact Euler identity
  // Sum_e l^2_e d r_U/d l^2_e = -2 r_U: with the V^2 weights L_k is homogeneous
  // of degree -1 in l^2 and r_U = ||(L - lambda)p||^2 of degree -2, measured as
  // r_U(s*l^2) = r_U/s^2 exactly.
  if (k_ == 1 && metricSource_ == HodgeLaplacian::MetricSource::DiagonalWeights)
    return periodGradientOverLoops(
        holeLoops(holes, "EigenstateSynthesis::residualForPeriodsGradient"),
        targetPeriods);
  // Under the Whitney pencil the loop core's low-rank dM is replaced by the
  // dense analytic dL_1/dl^2 inside periodGradientOverLoops; the general core
  // reads the same derivative through HodgeLaplacian::laplacianGradient.
  return periodGradientGeneral(holes, targetPeriods);
}

std::vector<cd> EigenstateSynthesis::periodGapForLoopsGradient(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  // Degree-1 machinery, exactly as periodGradientOverLoops (and as
  // the r_psi functional itself — periodGapForLoops reads loop periods of an
  // edge cochain, defined only at k = 1).
  if (k_ != 1)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoopsGradient: the edge-loop core "
        "is degree-1 machinery (loops are closed walks of 1-cells); this "
        "synthesis is degree " + std::to_string(k_) + ".");
  // The hard-pin sibling of periodGradientOverLoops (r_U): same first-order
  // eigenvector-perturbation setup (M = L1, harmonic split Un/Unn, the per-edge
  // low-rank dM, dUn), but the score is the period gap r_psi = ||A c - t||^2
  // with A = Q Un and c the least-squares fit, rather than the leaked state's
  // non-harmonicity. Least-squares optimality A^T r = 0 (envelope theorem) drops
  // the dc term, so d r_psi / d l^2 = 2 Re( r^H (Q dUn) c ): no leak, no dpsi
  // chain.
  //
  // The M / eigensplit / per-edge-dM machinery below duplicates
  // periodGradientOverLoops so that the r_U gradient, which is finite-difference
  // and GPU-mirror verified, need not be touched. Keep the two copies in sync;
  // test_period_gap_python.py finite-difference-guards this one.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t n1 = order_;
  std::vector<cd> grad(n1, cd(0.0, 0.0));
  const std::size_t m = loops.size();
  if (n1 == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForLoopsGradient: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " loops");
  static constexpr double kNullTol = 1e-7;
  const Index N = static_cast<Index>(n1);

  // The shared degree-1 setup: chain complex, M = L1, the boundary pieces and
  // the index maps. Both edge-loop gradients build the same one.
  const PeriodGradientContext ctx = periodGradientContext();
  const auto &tris = ctx.tris;
  const std::size_t n2 = ctx.n2;
  const Eigen::MatrixXcd &M = ctx.M;
  const Eigen::VectorXcd &W1 = ctx.W1;
  const Eigen::MatrixXcd &d2m = ctx.d2m;
  const Eigen::MatrixXcd &K1 = ctx.K1, &K2 = ctx.K2;
  const std::vector<cd> &W2v = ctx.W2v;
  const std::vector<std::vector<std::uint64_t>> &cells1 = cellSimplices();
  const auto &cidx1 = ctx.cidx1;
  const auto &trisOf = ctx.trisOf;
  const auto L2 = [&ctx](std::uint64_t a, std::uint64_t b) {
    return ctx.l2(a, b);
  };

  // ---- Q (signed edge-loop covector); the gap needs no leak column ----
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  for (std::size_t q = 0; q < m; ++q) {
    if (loops[q].empty())
      throw std::runtime_error(
          "EigenstateSynthesis::periodGapForLoopsGradient: loop " +
          std::to_string(q) + " is empty");
    Edge::walkLoop(loops[q], [&](std::uint64_t a, std::uint64_t b, double s) {
      Q(static_cast<Index>(q), static_cast<Index>(cidx1.at(ascendingIds(a, b)))) += s;
    });
  }

  // ---- eigendecomposition of M; harmonic (null) / non-null split ----
  // The signed operator is generally non-self-adjoint (real but non-symmetric
  // on the real-l^2 manifold), so a general eigensolver is required: a
  // self-adjoint one reads a single triangle and silently symmetrizes, which
  // makes this gradient come out identically zero.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i) (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;  // no harmonics -> nothing carried, gap is constant
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  // Left (dual) basis for the non-self-adjoint perturbation: rows of U^-1 are
  // the covectors v_m with v_m . u_l = delta_ml, which first-order eigenvector
  // perturbation of a non-symmetric M needs in place of U^T.
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);  // 1 / (0 - lambda_nn) for the eigenvector perturbation
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  // ---- the least-squares fit c and the period-gap residual r = A c - target ----
  // Rank-robust min-norm fit (Jacobi SVD), matching periodGapForLoops's value so
  // the gradient stays consistent with the function it differentiates, and so a
  // column-rank-deficient A (carried harmonic count nd > read-out cycle count m,
  // or dependent/zero-period harmonics) yields the min-norm solution rather than
  // the NaN a singular (A^T A)^{-1} would give. A^T r = 0 holds for any
  // least-squares solution, so the dropped dc term is unaffected.
  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                            // m x nd
  const MatrixXcd Ac = A.cast<cd>();
  const VectorXcd c =
      Ac.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(target);  // nd
  const VectorXcd r = Ac * c - target;                 // m; A^T r = 0 (optimality)

  // ---- per-edge analytic gradient d r_psi / d l^2 (low-rank dM + perturbation) ----
  for (std::size_t je = 0; je < n1; ++je) {
    const Index j = static_cast<Index>(je);
    const auto ek = ascendingIds(cells1[je][0], cells1[je][1]);
    // dM for the signed operator M = W1^-1 K1 + K2 W1 (K2 = d2 W2^-1 d2^T),
    // under the V^2 weights (the HodgeLaplacian default): W1_j = l^2_j exactly,
    // so dW1_j/dl^2_j = 1 and, every piece rank one,
    //   dM = -(1/W1_j^2) e_j (K1 row j)                       [d(W1^-1) K1]
    //      + (K2 col j) e_j^T                                 [K2 W1 -> K2 dW1]
    //      + per triangle t on e:
    //        -(dW2_t/W2_t^2) (d2 col t)((W1 o d2 col t))^T    [d(W2^-1) term]
    // dM is generally non-symmetric, like M itself.
    std::vector<VectorXcd> colsA, colsB;
    VectorXcd ev = VectorXcd::Zero(N);
    ev[j] = 1.0;
    colsA.push_back(ev);
    colsB.push_back((-1.0 / (W1[j] * W1[j])) * K1.row(j).transpose());
    colsA.push_back(K2.col(j));
    colsB.push_back(ev);
    const auto trisIt = trisOf.find(ek);
      for (std::size_t ti : (trisIt == trisOf.end()
                                 ? std::vector<std::size_t>{}
                                 : trisIt->second)) {
      const auto &t = tris[ti];
      Eigen::Matrix2cd G;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          G(i, jj) = 0.5 * (L2(t[0], t[i + 1]) + L2(t[0], t[jj + 1]) - L2(t[i + 1], t[jj + 1]));
      const cd detG = G.determinant();
      const cd W2ti = W2v[ti];
      // W2 must be the V^2 weight detG/4 this derivation assumes.
      if (std::abs(detG / 4.0 - W2ti) > 1e-9 * std::max(1.0, std::abs(W2ti)) ||
          std::abs(detG) < 1e-12)
        continue;
      auto ind = [&](int pp, int qq) -> double {
        return (pp != qq && ascendingIds(t[pp], t[qq]) == ek) ? 1.0 : 0.0;
      };
      Eigen::Matrix2cd dG;
      for (int i = 0; i < 2; ++i)
        for (int jj = 0; jj < 2; ++jj)
          dG(i, jj) = 0.5 * (ind(0, i + 1) + ind(0, jj + 1) - ind(i + 1, jj + 1));
      // W2 = detG/4 => dW2 = W2 * tr(G^-1 dG), by Jacobi's formula.
      const cd dW2ti = W2ti * (G.inverse() * dG).trace();
      const VectorXcd dcol = d2m.col(static_cast<Index>(ti));
      colsA.push_back(dcol);
      colsB.push_back((-dW2ti / (W2ti * W2ti)) * W1.cwiseProduct(dcol));
    }
    const Index rk = static_cast<Index>(colsA.size());
    MatrixXcd fa(N, rk), fb(N, rk);
    for (Index k = 0; k < rk; ++k) {
      fa.col(k) = colsA[static_cast<std::size_t>(k)];
      fb.col(k) = colsB[static_cast<std::size_t>(k)];
    }
    // The harmonic-subspace perturbation dUn, then dA = Q dUn; the envelope
    // theorem (A^T r = 0) leaves only 2 Re( r^H (dA c) ).
    const MatrixXcd core = (Vnn * fa) * (fb.transpose() * Un);  // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);               // n1 x nd
    const MatrixXcd dA = Q * dUn;                                           // m x nd
    // Complex gradient: r is holomorphic in l^2, so no .real() projection
    // belongs here. The value is
    //   g = dr/d(Re l^2) - i dr/d(Im l^2),
    // whose real part is the real-locus derivative. Discarding the imaginary
    // half would leave the register term unable to move in the plane the
    // descent direction steps in.
    grad[je] = 2.0 * r.dot(dA.cast<cd>() * c);
  }
  return grad;
}

std::vector<cd> EigenstateSynthesis::periodGapForPeriodsGradient(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Route by degree, exactly as residualForPeriodsGradient does: the fast
  // low-rank edge-loop core at k = 1, the degree-generic core at k >= 2. k = 0
  // reads a different operator — the complex Hermitian U(1) connection
  // L^U(1) = D - A, not a weight variant of L_k — so it has no period-gap core
  // at all, and that is stated here at the entry point rather than deeper in.
  if (k_ == 0)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapForPeriodsGradient: the period gap has "
        "no core at degree 0 — the U(1) connection L = D - A it reads is a "
        "different (complex Hermitian) operator, not a weight variant of "
        "L_k; this synthesis is degree " +
        std::to_string(k_) + ".");
  if (k_ == 1)
    return periodGapForLoopsGradient(
        holeLoops(holes, "EigenstateSynthesis::periodGapForPeriodsGradient"),
        targetPeriods);
  return periodGapGradientOverHoles(holes, targetPeriods);
}

std::vector<cd> EigenstateSynthesis::periodGapGradientOverHoles(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  // Arbitrary-degree exact d r_psi / d l^2 for the period gap
  // r_psi = ||A c - t||^2, with A = Q U_n the periods of the harmonic basis and
  // c the least-squares fit. The setup mirrors periodGradientGeneral term for
  // term — same M = L_k, same period covector Q, same non-self-adjoint
  // eigensplit — but the score differs, and so does the derivative:
  //
  // least-squares optimality gives A^dagger r = 0, so by the envelope theorem the
  // dc term drops out entirely and
  //     d r_psi / d l^2_e = 2 Re( r^dagger (Q dU_n) c ),
  // where dU_n is the first-order perturbation of the harmonic block. That is
  // the same identity the k = 1 loop core uses; this is its degree-generic form.
  //
  // The fit uses the SVD pseudo-inverse rather than normal equations: A can be
  // rank-deficient (more harmonics than holes, or degenerate periods), where
  // (A^T A)^-1 is singular. It matches lstsqOverReadout, which is what the value
  // uses, so gradient and value fit the same way.
  //
  // Certified by the Euler identity for a degree-0 functional:
  //     Sum_e l^2_e d r_psi / d l^2_e = 0.
  // L_k is homogeneous of degree -1 in l^2, so l^2 -> s l^2 sends L -> L/s,
  // which leaves the kernel — and hence the normalized harmonic basis, A, c and
  // the gap — unchanged. By contrast r_U is of degree -2, with Euler sum -2 r_U.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  const std::size_t nk = order_;
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const std::vector<std::vector<std::uint64_t>> edges1 = cc.kSimplexVertices(1);
  std::vector<cd> grad(edges1.size(), cd(0.0, 0.0));
  const std::size_t m = holes.size();
  if (nk == 0 || m == 0) return grad;
  if (targetPeriods.size() != m)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapGradientOverHoles: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(m) + " holes");
  if (k_ == 0)
    throw std::runtime_error(
        "EigenstateSynthesis::periodGapGradientOverHoles: no period-gap core at "
        "degree 0 — the U(1) connection L = D - A it reads is a different "
        "(complex Hermitian) operator, not a weight variant of L_k; this "
        "synthesis is degree " +
        std::to_string(k_) + ".");
  static constexpr double kNullTol = 1e-9;   // harmonicMatrix's tolerance: this
                                             // must differentiate the harmonic
                                             // set the value reads, not r_U's 1e-7
  const Index N = static_cast<Index>(nk);

  const std::vector<cd> Lflat = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_).laplacian(k_, /*metric=*/true);
  MatrixXcd M(N, N);
  for (std::size_t i = 0; i < nk; ++i)
    for (std::size_t j = 0; j < nk; ++j)
      M(static_cast<Index>(i), static_cast<Index>(j)) = Lflat[i * nk + j];

  // Period covector Q: a hole is a removed (k+1)-cell, its drop-one facets are
  // k-cells with sign (-1)^j — the assembleRegisterReadout convention.
  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;
  const std::size_t hv = static_cast<std::size_t>(k_) + 2;
  MatrixXcd Q = MatrixXcd::Zero(static_cast<Index>(m), N);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::periodGapGradientOverHoles: hole has " +
          std::to_string(h.size()) + " vertices, expected " + std::to_string(hv));
    for (std::size_t j = 0; j < hv; ++j) {
      std::vector<std::uint64_t> f;
      f.reserve(hv - 1);
      for (std::size_t i = 0; i < hv; ++i)
        if (i != j) f.push_back(h[i]);
      const auto it = col.find(f);
      if (it == col.end())
        throw std::runtime_error(
            "EigenstateSynthesis::periodGapGradientOverHoles: a hole facet is "
            "not a k-cell of the complex");
      Q(static_cast<Index>(q), static_cast<Index>(it->second)) +=
          (j % 2 == 0) ? 1.0 : -1.0;
    }
  }

  // Non-self-adjoint eigensplit: the signed operator is real but not symmetric,
  // so a self-adjoint solver would read one triangle and symmetrize.
  Eigen::ComplexEigenSolver<MatrixXcd> eig(M);
  const VectorXcd lam = eig.eigenvalues();
  const MatrixXcd U = eig.eigenvectors();
  std::vector<Index> nullIdx, nnIdx;
  for (Index i = 0; i < N; ++i)
    (std::abs(lam[i]) < kNullTol ? nullIdx : nnIdx).push_back(i);
  const Index nd = static_cast<Index>(nullIdx.size());
  const Index nnd = static_cast<Index>(nnIdx.size());
  if (nd == 0) return grad;      // no harmonics: nothing carried, gap constant
  MatrixXcd Un(N, nd), Unn(N, nnd);
  for (Index r = 0; r < nd; ++r) Un.col(r) = U.col(nullIdx[r]);
  for (Index r = 0; r < nnd; ++r) Unn.col(r) = U.col(nnIdx[r]);
  const MatrixXcd Uinv = U.inverse();
  MatrixXcd Vnn(nnd, N);
  for (Index r = 0; r < nnd; ++r) Vnn.row(r) = Uinv.row(nnIdx[r]);
  VectorXcd invlam(nnd);
  for (Index r = 0; r < nnd; ++r) invlam[r] = -1.0 / lam[nnIdx[r]];

  VectorXcd target(static_cast<Index>(m));
  for (std::size_t q = 0; q < m; ++q) target[static_cast<Index>(q)] = targetPeriods[q];
  const MatrixXcd A = Q * Un;                                   // m x nd
  Eigen::BDCSVD<MatrixXcd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const VectorXcd c = svd.solve(target);                        // min-norm fit
  const VectorXcd r = A * c - target;                           // the gap vector

  const HodgeLaplacian hl(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  for (std::size_t je = 0; je < edges1.size(); ++je) {
    const std::vector<cd> dMflat =
        hl.laplacianGradient(k_, edges1[je][0], edges1[je][1]);
    if (dMflat.empty()) continue;
    MatrixXcd dM(N, N);
    for (std::size_t i = 0; i < nk; ++i)
      for (std::size_t j = 0; j < nk; ++j)
        dM(static_cast<Index>(i), static_cast<Index>(j)) = dMflat[i * nk + j];
    const MatrixXcd core = (Vnn * dM) * Un;                     // nnd x nd
    const MatrixXcd dUn = Unn * (invlam.asDiagonal() * core);   // N x nd
    const MatrixXcd dA = Q * dUn;                               // m x nd
    // Envelope theorem, kept complex; see the k = 1 core.
    grad[je] = 2.0 * r.dot(dA * c);
  }
  return grad;
}


std::vector<double> EigenstateSynthesis::residualForLoopsGradient(
    const std::vector<EdgeLoop> &loops,
    const std::vector<cd> &targetPeriods) const {
  return periodGradientOverLoops(loops, targetPeriods);
}

}  // namespace tessera::cobordism
