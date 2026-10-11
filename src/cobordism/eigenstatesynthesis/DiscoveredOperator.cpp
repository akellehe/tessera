// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): the discovered
// operator, ker L1(W - dW), and the geometric operator read from it. One of the
// translation units that define the class's members by topic.

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

// === The discovered operator: ker L₁(W − ∂W) ===

std::vector<std::vector<std::uint64_t>>
EigenstateSynthesis::bulkMinusBoundaryCells() const {
  std::vector<std::vector<std::uint64_t>> out;
  if (!st_) return out;
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (auto &e : ChainComplex::fromSpacetime(*st_).kSimplexVertices(1)) {
    bool interior = true;
    for (const std::uint64_t v : e)
      if (bverts.count(v)) { interior = false; break; }
    if (interior) out.push_back(e);
  }
  return out;
}


std::vector<cd>
EigenstateSynthesis::bulkMinusBoundaryHarmonicMatrix(double tol,
                                                     bool metric) const {
  using Eigen::Index;
  using Eigen::MatrixXcd;
  using Eigen::VectorXcd;
  if (!st_)
    return {};

  // W − ∂W is the subcomplex induced on the interior vertices (those on no ∂W
  // face); a cell belongs to it iff all of its vertices are interior. Too few
  // interior vertices means no interior 1-cycle, so nothing is carried: three
  // interior vertices already carry nothing.
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  const auto interiorCell = [&](const std::vector<std::uint64_t> &c) {
    for (const std::uint64_t v : c)
      if (bverts.count(v))
        return false;
    return true;
  };

  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  const auto v0 = cc.kSimplexVertices(0);
  const auto v1 = cc.kSimplexVertices(1);
  const auto v2 = cc.kSimplexVertices(2);
  const std::size_t n0 = v0.size(), n1 = v1.size(), n2 = v2.size();
  if (n1 == 0)
    return {};

  // The interior-cell index lists into the full C_k orderings.
  std::vector<Index> i0, i1, i2;
  for (std::size_t i = 0; i < n0; ++i)
    if (interiorCell(v0[i]))
      i0.push_back(static_cast<Index>(i));
  for (std::size_t i = 0; i < n1; ++i)
    if (interiorCell(v1[i]))
      i1.push_back(static_cast<Index>(i));
  for (std::size_t i = 0; i < n2; ++i)
    if (interiorCell(v2[i]))
      i2.push_back(static_cast<Index>(i));
  const Index m1 = static_cast<Index>(i1.size());
  if (m1 == 0)
    return {};

  // Restrict the boundary maps to the interior rows and columns once. The
  // combinatorial and metric modes differ only in the weights wrapped around
  // these same incidence matrices.
  const std::vector<long> &d1 = cc.boundaryMatrix(1);
  const std::vector<long> &d2 = cc.boundaryMatrix(2);
  MatrixXcd D1 = MatrixXcd::Zero(static_cast<Index>(i0.size()), m1);
  for (Index r = 0; r < static_cast<Index>(i0.size()); ++r)
    for (Index c = 0; c < m1; ++c)
      D1(r, c) = static_cast<double>(d1[static_cast<std::size_t>(i0[r]) * n1 +
                                        static_cast<std::size_t>(i1[c])]);
  MatrixXcd D2 = MatrixXcd::Zero(m1, static_cast<Index>(i2.size()));
  for (Index r = 0; r < m1; ++r)
    for (Index c = 0; c < static_cast<Index>(i2.size()); ++c)
      D2(r, c) = static_cast<double>(d2[static_cast<std::size_t>(i1[r]) * n2 +
                                        static_cast<std::size_t>(i2[c])]);

  MatrixXcd L = D1.transpose() * D1 + D2 * D2.transpose();
  if (metric) {
    const HodgeLaplacian hodge(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
    const auto fullW0 = hodge.weights(0);
    const auto fullW1 = hodge.weights(1);
    const auto fullW2 = hodge.weights(2);
    if (fullW0.size() != n0 || fullW1.size() != n1 ||
        (!i2.empty() && fullW2.size() != n2))
      return {};

    const auto selectWeights = [](const std::vector<cd> &full,
                                  const std::vector<Index> &indices) {
      VectorXcd selected(static_cast<Index>(indices.size()));
      for (Index i = 0; i < selected.size(); ++i)
        selected[i] = full[static_cast<std::size_t>(
            indices[static_cast<std::size_t>(i)])];
      return selected;
    };
    const VectorXcd w0 = selectWeights(fullW0, i0);
    const VectorXcd w1 = selectWeights(fullW1, i1);
    const VectorXcd w2 = selectWeights(fullW2, i2);
    if (!w0.allFinite() || !w1.allFinite() || !w2.allFinite())
      return {};
    for (Index i = 0; i < w1.size(); ++i)
      if (std::abs(w1[i]) <= std::numeric_limits<double>::min())
        return {};
    for (Index i = 0; i < w2.size(); ++i)
      if (std::abs(w2[i]) <= std::numeric_limits<double>::min())
        return {};

    L = w1.cwiseInverse().asDiagonal() * D1.transpose() * w0.asDiagonal() * D1;
    if (!i2.empty())
      L += D2 * w2.cwiseInverse().asDiagonal() * D2.transpose() *
           w1.asDiagonal();
    if (!L.allFinite())
      return {};

    // The signed metric operator is generally non-normal. Its right nullspace
    // is the span of the right singular vectors with sigma < tol.
    Eigen::BDCSVD<MatrixXcd> svd(L, Eigen::ComputeFullV);
    const Eigen::VectorXd &sigma = svd.singularValues(); // descending
    const MatrixXcd &V = svd.matrixV();
    std::vector<cd> out;
    for (Index offset = 0; offset < sigma.size(); ++offset) {
      const Index j = sigma.size() - 1 - offset; // smallest first
      if (sigma[j] >= tol)
        continue;
      for (Index i = 0; i < m1; ++i)
        out.push_back(V(i, j));
    }
    return out;
  }

  // Combinatorial ker L₁ ≅ H₁ of the interior, in ascending eigenvalue order.
  Eigen::SelfAdjointEigenSolver<MatrixXcd> eig(L);
  const Eigen::VectorXd &lam = eig.eigenvalues();
  const MatrixXcd &V = eig.eigenvectors();
  std::vector<cd> out;
  for (Index j = 0; j < lam.size(); ++j) {
    if (std::abs(lam[j]) >= tol)
      continue;
    for (Index i = 0; i < m1; ++i)
      out.push_back(V(i, j));
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::readoutLaplacian() const {
  // k = 0 scores the U(1) connection operator, k >= 1 the Hodge L_k. Read
  // through the captured operator: its vertex/cell ordering is the one
  // cellOrdering_ was built from, and neither entry point consults a spectral
  // cache (both reassemble from the live edges on every call), so repeated
  // perturb-then-query reflects every perturbation.
  return k_ == 0 ? laplacian_.connectionLaplacian()
                 : laplacian_.laplacian(k_, /*metric=*/true);
}

std::vector<cd> EigenstateSynthesis::readoutHarmonicMatrix() const {
  // k = 0 reads the U(1) connection operator, k >= 1 the Hodge L_k. The
  // degree-zero register's content is the U(1) flux carried around a hole:
  // ker L_0 = b_0 at any weights, so L_0's harmonics can carry no flux and a
  // degree-zero readout taken from them would be identically gauge-flat. The
  // connection operator is indexed over the full sorted vertex order, which is
  // exactly cellOrdering_ at k = 0.
  const HodgeLaplacian hodge(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  return k_ == 0 ? hodge.connectionHarmonicMatrix(1e-9)
                 : hodge.harmonicMatrix(k_, 1e-9, /*metric=*/true);
}

EigenstateSynthesis::RegisterReadout EigenstateSynthesis::assembleRegisterReadout(
    const std::vector<std::vector<std::uint64_t>> &holes) const {
  const auto joinIds = [](const std::vector<std::uint64_t> &c) {
    std::string out = "(";
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(c[i]);
    }
    return out + ")";
  };

  RegisterReadout out;
  const std::size_t n = order_;
  // Harmonics fresh from the live complex: surgery between calls moves them,
  // and the operator's own spectral cache is keyed to construction time.
  out.H = readoutHarmonicMatrix();
  if (n == 0) {
    if (!holes.empty())
      throw std::runtime_error(
          "EigenstateSynthesis::assembleRegisterReadout: the complex has no "
          "k-cells to carry periods");
    return out;
  }
  if (out.H.size() % n != 0)
    throw std::runtime_error(
        "EigenstateSynthesis::assembleRegisterReadout: the harmonic matrix "
        "width " + std::to_string(out.H.size()) +
        " is not a multiple of the captured operator dimension " +
        std::to_string(n) + " (the complex changed behind the synthesizer)");
  out.dim = out.H.size() / n;

  std::map<std::vector<std::uint64_t>, std::size_t> col;
  for (std::size_t i = 0; i < cellOrdering_.size(); ++i) col[cellOrdering_[i]] = i;

  const std::size_t m = holes.size();
  const std::size_t hv = static_cast<std::size_t>(k_) + 2;
  out.P.assign(out.dim * m, cd(0.0, 0.0));
  out.leakColumns.reserve(m);
  for (std::size_t q = 0; q < m; ++q) {
    std::vector<std::uint64_t> h = holes[q];
    std::sort(h.begin(), h.end());
    if (h.size() != hv)
      throw std::runtime_error(
          "EigenstateSynthesis::assembleRegisterReadout: hole " + joinIds(h) +
          " has " + std::to_string(h.size()) + " vertices; a degree-" +
          std::to_string(k_) + " period needs a removed (k+1)-cell of " +
          std::to_string(hv));
    // Facets are visited in the degree's walk order: the (a,b),(b,c),(a,c) edge
    // walk of a circle at k = 1, so the period accumulates bit for bit in the
    // register layers' summation order; the canonical drop-v_j order otherwise.
    // The hole's leak facet is the first of the walk — the (a,b) edge, or the
    // drop-v_0 facet — which carries boundary sign +1 in both conventions, so
    // adding the leak there moves that hole's period by exactly the leak.
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
            "EigenstateSynthesis::assembleRegisterReadout: facet " +
            joinIds(f) + " of hole " + joinIds(h) +
            " is not a k-cell of the complex (not a boundary cycle here)");
      if (w == 0) out.leakColumns.push_back(it->second);
      // The induced-orientation boundary sign: facet j of the sorted hole
      // drops vertex v_j and carries (-1)^j.
      const double s = (j % 2 == 0) ? 1.0 : -1.0;
      for (std::size_t r = 0; r < out.dim; ++r)
        out.P[r * m + q] += s * out.H[r * n + it->second];
    }
  }
  return out;
}

std::vector<cd> EigenstateSynthesis::cyclePeriods(
    const std::vector<std::vector<std::uint64_t>> &holes) const {
  return assembleRegisterReadout(holes).P;
}

std::vector<cd> EigenstateSynthesis::carriedRepresentative(
    const std::vector<std::vector<std::uint64_t>> &holes,
    const std::vector<cd> &targetPeriods) const {
  if (targetPeriods.size() != holes.size())
    throw std::runtime_error(
        "EigenstateSynthesis::carriedRepresentative: " +
        std::to_string(targetPeriods.size()) + " target periods for " +
        std::to_string(holes.size()) + " holes");
  return carriedFromReadout(assembleRegisterReadout(holes), targetPeriods);
}


GeometricOperatorReadout EigenstateSynthesis::geometricOperator(
    const std::shared_ptr<Spacetime> &spacetime, int stateDimension,
    std::vector<std::vector<std::uint64_t>> frameCells, double tol, bool metric,
    HodgeLaplacian::MetricSource metricSource) {
  GeometricOperatorReadout result;
  result.stateDimension = stateDimension;
  result.metric = metric;
  const auto obstruct = [&](std::string reason) {
    result.identifiable = false;
    result.obstruction = std::move(reason);
    result.choiState.clear();
    result.operatorMatrix.clear();
    return result;
  };
  if (!spacetime)
    return obstruct("the cobordism has no spacetime");
  if (stateDimension <= 0)
    return obstruct("state dimension must be positive");
  if (!(tol > 0.0) || !std::isfinite(tol))
    return obstruct("kernel tolerance must be finite and positive");

  const std::size_t d = static_cast<std::size_t>(stateDimension);
  if (d > std::numeric_limits<std::size_t>::max() / d)
    return obstruct("state dimension overflows the Choi width");
  const std::size_t choiWidth = d * d;
  EigenstateSynthesis synthesis(spacetime, 1, metricSource);
  result.bulkCells = synthesis.bulkMinusBoundaryCells();
  result.bulkCellCount = result.bulkCells.size();
  if (result.bulkCells.empty())
    return obstruct("bulk-minus-boundary has no interior 1-cells");

  if (frameCells.empty()) {
    if (result.bulkCells.size() != choiWidth)
      return obstruct(
          "an ordered d^2 Choi frame is required when the bulk width differs");
    frameCells = result.bulkCells;
  }
  if (frameCells.size() != choiWidth)
    return obstruct("the Choi frame must contain exactly d^2 interior 1-cells");
  std::map<std::vector<std::uint64_t>, std::size_t> bulkIndex;
  for (std::size_t i = 0; i < result.bulkCells.size(); ++i)
    bulkIndex[result.bulkCells[i]] = i;
  std::set<std::vector<std::uint64_t>> usedFrameCells;
  std::vector<std::size_t> frameColumns;
  frameColumns.reserve(frameCells.size());
  for (auto &cell : frameCells) {
    std::sort(cell.begin(), cell.end());
    if (cell.size() != 2 || cell[0] == cell[1])
      return obstruct("a Choi frame entry is not an edge");
    const auto found = bulkIndex.find(cell);
    if (found == bulkIndex.end())
      return obstruct("a Choi frame edge is not in the bulk-minus-boundary");
    if (!usedFrameCells.insert(cell).second)
      return obstruct("the Choi frame repeats an interior edge");
    frameColumns.push_back(found->second);
  }
  result.frameCells = frameCells;

  const std::vector<cd> flat =
      synthesis.bulkMinusBoundaryHarmonicMatrix(tol, metric);
  if (flat.size() % result.bulkCells.size() != 0)
    return obstruct("bulk harmonic matrix has an inconsistent shape");
  result.kernelDimension = flat.size() / result.bulkCells.size();
  if (result.kernelDimension == 0)
    return obstruct("bulk-minus-boundary kernel is empty");

  Eigen::MatrixXcd restricted(static_cast<Eigen::Index>(result.kernelDimension),
                              static_cast<Eigen::Index>(choiWidth));
  for (std::size_t row = 0; row < result.kernelDimension; ++row)
    for (std::size_t column = 0; column < choiWidth; ++column)
      restricted(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(column)) =
          flat[row * result.bulkCells.size() + frameColumns[column]];
  if (!restricted.allFinite())
    return obstruct("the framed bulk kernel is non-finite");

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(restricted, Eigen::ComputeThinU |
                                                         Eigen::ComputeThinV);
  const Eigen::VectorXd &singularValues = svd.singularValues();
  result.frameSingularValues.reserve(
      static_cast<std::size_t>(singularValues.size()));
  for (Eigen::Index i = 0; i < singularValues.size(); ++i)
    result.frameSingularValues.push_back(singularValues[i]);
  const double leading = singularValues.size() == 0 ? 0.0 : singularValues[0];
  const double cutoff = tol * std::max(1.0, leading);
  for (Eigen::Index i = 0; i < singularValues.size(); ++i)
    if (singularValues[i] > cutoff)
      ++result.frameRank;
  if (result.frameRank == 0)
    return obstruct("the framed bulk kernel carries no Choi component");
  if (result.frameRank != 1)
    return obstruct("the framed bulk kernel has rank " +
                    std::to_string(result.frameRank) +
                    "; no unique Choi ray exists");

  // A = U Sigma V^H: the direct row-state spanning A is conj(V.col(0)). This
  // remains invariant under a change of basis among the bulk kernel rows.
  Eigen::VectorXcd choi = svd.matrixV().col(0).conjugate();
  for (Eigen::Index i = 0; i < choi.size(); ++i)
    if (std::abs(choi[i]) > cutoff) {
      choi *= std::conj(choi[i]) / std::abs(choi[i]);
      break;
    }
  result.choiState.assign(choi.data(), choi.data() + choi.size());

  result.operatorMatrix =
      ::tessera::quantum::ChoiJamiolkowski::operatorFromChoiState(
          result.choiState, stateDimension);
  Eigen::MatrixXcd U(stateDimension, stateDimension);
  for (int row = 0; row < stateDimension; ++row)
    for (int column = 0; column < stateDimension; ++column)
      U(row, column) = result.operatorMatrix[static_cast<std::size_t>(row) * d +
                                             static_cast<std::size_t>(column)];
  result.unitarityError =
      (U.adjoint() * U -
       Eigen::MatrixXcd::Identity(stateDimension, stateDimension))
          .norm();
  result.identifiable = true;
  result.obstruction.clear();
  return result;
}

}  // namespace tessera::cobordism
