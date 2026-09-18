// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/HodgeLaplacian.h"

#include "chainhodge/ChainHodge.h"
#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/WhitneyMass.h"

#include <Eigen/Dense>

#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cobordism/ChainComplex.h"
#include "cobordism/Cochain.h"
#include "cobordism/Spectrum.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

using cd = std::complex<double>;

namespace {

using Face = std::vector<std::uint64_t>;  // sorted vertex ids
using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

// Sorted vertex ids of a simplex — the homological reference ordering. Distinct
// simplices have distinct tuples, so sorting by it reproduces ChainComplex's
// column order.
Face sortedIds(const SimplexPtr &s) {
  Face ids;
  for (const auto &v : s->getVertices()) ids.push_back(v->getId());
  std::sort(ids.begin(), ids.end());
  return ids;
}

// Face-closure of the complex, bucketed by dimension and ordered within each
// dimension by sorted vertex ids, so faces[k][j] is the simplex behind column j
// of boundaryMatrix(k). Returns SimplexPtrs, for their volumes.
std::vector<std::vector<SimplexPtr>> orderedFaces(const Spacetime &K) {
  std::map<int, std::map<Face, SimplexPtr>> byDim;  // dim -> (sorted ids -> simplex)
  std::unordered_set<std::uint64_t> seen;
  std::vector<SimplexPtr> stack(K.getSimplices().begin(), K.getSimplices().end());
  while (!stack.empty()) {
    SimplexPtr s = stack.back();
    stack.pop_back();
    if (s == nullptr) continue;
    if (!seen.insert(s->fingerprint.fingerprint()).second) continue;
    const int k = static_cast<int>(s->size()) - 1;
    byDim[k][sortedIds(s)] = s;
    if (k >= 1)
      for (const auto &f : s->getFacets()) stack.push_back(f);
  }
  const int n = byDim.empty() ? -1 : byDim.rbegin()->first;
  std::vector<std::vector<SimplexPtr>> faces(static_cast<std::size_t>(n + 1));
  for (int k = 0; k <= n; ++k)
    for (const auto &[face, s] : byDim[k]) faces[static_cast<std::size_t>(k)].push_back(s);
  return faces;
}

// Diagonal weights W_k (length `count`) in ChainComplex column order: the signed
// per-k-simplex content Simplex::volume() under `convention`, or all ones for
// k <= 0, for k above the top dimension, and for `metric == false`. Timelike
// cells carry a negative or imaginary weight, so W_k is indefinite; a degenerate
// (zero) cell falls back to +1 to stay invertible.
std::vector<std::complex<double>> simplexWeights(
    const std::vector<std::vector<SimplexPtr>> &faces, int k, int count,
    bool metric, HodgeLaplacian::WeightConvention convention) {
  using cdw = std::complex<double>;
  std::vector<cdw> w(static_cast<std::size_t>(std::max(count, 0)), cdw{1.0, 0.0});
  if (!metric || k == 0 || k < 0 || k >= static_cast<int>(faces.size())) return w;
  const auto &fk = faces[static_cast<std::size_t>(k)];
  for (int j = 0; j < count && j < static_cast<int>(fk.size()); ++j) {
    // Both branches are signed and complex-valued; there is no |vol| mode.
    //  Content        W = V, the k-content; sqrt(l^2) for an edge, so a timelike
    //                 cell's weight is imaginary.
    //  SquaredContent W = V^2 = det G/(d!)^2, polynomial in the squared edge
    //                 lengths, so on real signed l^2 it is real and signed.
    const cdw vol = fk[static_cast<std::size_t>(j)]->volume();
    const cdw wt = (convention == HodgeLaplacian::WeightConvention::SquaredContent)
                       ? vol * vol
                       : vol;
    w[static_cast<std::size_t>(j)] = (std::abs(wt) > 0.0) ? wt : cdw{1.0, 0.0};
  }
  return w;
}

// Exact d(L_k)/d(l^2_(ea,eb)) for the signed-weight Laplacian
//   L_k = W_k^-1 d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k.
// Every W is diagonal and depends on l^2 only through the cell contents, so
//   dL = -W_k^-1 dW_k W_k^-1 d_k^T W_{k-1} d_k + W_k^-1 d_k^T dW_{k-1} d_k
//        -d_{k+1} W_{k+1}^-1 dW_{k+1} W_{k+1}^-1 d_{k+1}^T W_k
//        +d_{k+1} W_{k+1}^-1 d_{k+1}^T dW_k,
// with dW the signed volumeGradient verbatim. One workspace serves a fixed
// (spacetime revision, degree, weight convention): immutable ingredients are
// assembled once, simplex derivatives are indexed sparsely by edge, and each
// dL_e is a sum of local row/column scalings and rank-one updates.
class LaplacianDerivativeWorkspace {
public:
  LaplacianDerivativeWorkspace(const Spacetime &spacetime, int degree,
                               HodgeLaplacian::WeightConvention convention)
      : chainComplex_(ChainComplex::fromSpacetime(spacetime)),
        faces_(orderedFaces(spacetime)), degree_(degree),
        dimension_(chainComplex_.dimension()),
        degreeSize_(static_cast<int>(chainComplex_.numSimplices(degree))),
        convention_(convention),
        lowerBase_(Eigen::MatrixXcd::Zero(degreeSize_, degreeSize_)),
        upperBase_(Eigen::MatrixXcd::Zero(degreeSize_, degreeSize_)) {
    lowerWeights_ = buildWeightData(degree_ - 1);
    degreeWeights_ = buildWeightData(degree_);
    upperWeights_ = buildWeightData(degree_ + 1);

    lowerSize_ = degree_ >= 1
                     ? static_cast<int>(chainComplex_.numSimplices(degree_ - 1))
                     : 0;
    if (lowerSize_ > 0 && degreeSize_ > 0) {
      lowerBoundary_ = boundaryMatrix(degree_, lowerSize_, degreeSize_);
      lowerBase_.noalias() = lowerBoundary_.transpose() *
                             lowerWeights_.weights.matrix().asDiagonal() *
                             lowerBoundary_;
      lowerLeft_ = degreeWeights_.inverseWeights.matrix().asDiagonal() *
                   lowerBoundary_.transpose();
    }

    upperSize_ = degree_ + 1 <= dimension_
                     ? static_cast<int>(chainComplex_.numSimplices(degree_ + 1))
                     : 0;
    if (upperSize_ > 0 && degreeSize_ > 0) {
      upperBoundary_ = boundaryMatrix(degree_ + 1, degreeSize_, upperSize_);
      upperBase_.noalias() =
          upperBoundary_ * upperWeights_.inverseWeights.matrix().asDiagonal() *
          upperBoundary_.transpose();
      upperRight_ = upperBoundary_.transpose() *
                    degreeWeights_.weights.matrix().asDiagonal();
    }
  }

  [[nodiscard]] Eigen::MatrixXcd laplacian() const {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(degreeSize_, degreeSize_);
    if (degreeSize_ == 0)
      return result;
    result.noalias() =
        degreeWeights_.inverseWeights.matrix().asDiagonal() * lowerBase_;
    result.noalias() +=
        upperBase_ * degreeWeights_.weights.matrix().asDiagonal();
    return result;
  }

  [[nodiscard]] Eigen::MatrixXcd gradient(std::uint64_t edgeA,
                                          std::uint64_t edgeB) const {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(degreeSize_, degreeSize_);
    // At degree zero W_0 = I contributes no derivative and there is no lower
    // boundary block, leaving -d_1 W_1^-1 dW_1 W_1^-1 d_1^T. The lookups below
    // return nothing for the absent blocks, so no degree special case is
    // needed.
    if (degree_ < 0 || degreeSize_ == 0)
      return result;
    const EdgeKey edge{std::min(edgeA, edgeB), std::max(edgeA, edgeB)};

    if (const auto *entries = derivativesFor(degreeWeights_, edge)) {
      for (const auto &[index, derivative] : *entries) {
        const cd inverse = degreeWeights_.inverseWeights[index];
        result.row(index).noalias() +=
            (-inverse * inverse * derivative) * lowerBase_.row(index);
        result.col(index).noalias() += derivative * upperBase_.col(index);
      }
    }
    if (const auto *entries = derivativesFor(lowerWeights_, edge)) {
      for (const auto &[index, derivative] : *entries)
        result.noalias() +=
            derivative * lowerLeft_.col(index) * lowerBoundary_.row(index);
    }
    if (const auto *entries = derivativesFor(upperWeights_, edge)) {
      for (const auto &[index, derivative] : *entries) {
        const cd inverse = upperWeights_.inverseWeights[index];
        result.noalias() += (-inverse * inverse * derivative) *
                            upperBoundary_.col(index) * upperRight_.row(index);
      }
    }
    return result;
  }

  // What the second directional derivative needs that does not depend on the
  // differentiated edge: the weight-diagonal velocities Wdot_j, the edge-keyed
  // second weight derivatives, and the four base blocks differentiated once.
  // Built once per direction.
  struct DirectionData {
    Eigen::ArrayXcd lowerVelocity{};   // Wdot_{k-1}
    Eigen::ArrayXcd degreeVelocity{};  // Wdot_k
    Eigen::ArrayXcd upperVelocity{};   // Wdot_{k+1}
    std::map<EdgeKey, std::vector<std::pair<int, cd>>> lowerSecond{};
    std::map<EdgeKey, std::vector<std::pair<int, cd>>> degreeSecond{};
    std::map<EdgeKey, std::vector<std::pair<int, cd>>> upperSecond{};
    Eigen::MatrixXcd lowerBaseDot{};   // d_k^T Wdot_{k-1} d_k
    Eigen::MatrixXcd upperBaseDot{};   // -d_{k+1} W^-1 Wdot_{k+1} W^-1 d_{k+1}^T
    Eigen::MatrixXcd lowerLeftDot{};   // d/dt (W_k^-1 d_k^T)
    Eigen::MatrixXcd upperRightDot{};  // d_{k+1}^T Wdot_k
  };

  // The direction is edge-keyed like `Simplex::volumeGradient` and is handed
  // straight to the simplices.
  [[nodiscard]] DirectionData directionData(
      const std::map<EdgeKey, cd> &direction) const {
    DirectionData data;
    data.lowerVelocity = weightVelocity(lowerWeights_, degree_ - 1, direction);
    data.degreeVelocity = weightVelocity(degreeWeights_, degree_, direction);
    data.upperVelocity = weightVelocity(upperWeights_, degree_ + 1, direction);
    data.lowerSecond = weightSecond(lowerWeights_, degree_ - 1, direction);
    data.degreeSecond = weightSecond(degreeWeights_, degree_, direction);
    data.upperSecond = weightSecond(upperWeights_, degree_ + 1, direction);

    data.lowerBaseDot = Eigen::MatrixXcd::Zero(degreeSize_, degreeSize_);
    data.upperBaseDot = Eigen::MatrixXcd::Zero(degreeSize_, degreeSize_);
    if (lowerSize_ > 0 && degreeSize_ > 0) {
      data.lowerBaseDot.noalias() = lowerBoundary_.transpose() *
                                    data.lowerVelocity.matrix().asDiagonal() *
                                    lowerBoundary_;
      data.lowerLeftDot = lowerLeft_;
      for (int row = 0; row < degreeSize_; ++row)
        data.lowerLeftDot.row(row) *=
            -degreeWeights_.inverseWeights[row] * data.degreeVelocity[row];
    }
    if (upperSize_ > 0 && degreeSize_ > 0) {
      const Eigen::ArrayXcd inverseUpper = upperWeights_.inverseWeights;
      const Eigen::ArrayXcd scaled =
          -inverseUpper * data.upperVelocity * inverseUpper;
      data.upperBaseDot.noalias() = upperBoundary_ *
                                    scaled.matrix().asDiagonal() *
                                    upperBoundary_.transpose();
      data.upperRightDot = upperBoundary_.transpose() *
                           data.degreeVelocity.matrix().asDiagonal();
    }
    return data;
  }

  // Sum_f v_f d(dL/dz_e)/dz_f for one edge e: the exact second derivative,
  // contracted against the direction. Every term is a product rule on the four
  // blocks `gradient()` assembles.
  [[nodiscard]] Eigen::MatrixXcd gradientDirectionalDerivative(
      std::uint64_t edgeA, std::uint64_t edgeB,
      const DirectionData &data) const {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(degreeSize_, degreeSize_);
    if (degree_ < 0 || degreeSize_ == 0)
      return result;
    const EdgeKey edge{std::min(edgeA, edgeB), std::max(edgeA, edgeB)};

    if (const auto *entries = derivativesFor(degreeWeights_, edge)) {
      const auto *second = secondFor(data.degreeSecond, edge);
      for (const auto &[index, derivative] : *entries) {
        const cd inverse = degreeWeights_.inverseWeights[index];
        const cd velocity = data.degreeVelocity[index];
        const cd secondDerivative = lookup(second, index);
        // d/dt(-a^2 g) with a = 1/w, da/dt = -a^2 wdot.
        const cd rowFactor = 2.0 * inverse * inverse * inverse * velocity *
                                 derivative -
                             inverse * inverse * secondDerivative;
        result.row(index).noalias() += rowFactor * lowerBase_.row(index);
        result.row(index).noalias() +=
            (-inverse * inverse * derivative) * data.lowerBaseDot.row(index);
        result.col(index).noalias() += secondDerivative * upperBase_.col(index);
        result.col(index).noalias() +=
            derivative * data.upperBaseDot.col(index);
      }
    }
    if (const auto *entries = derivativesFor(lowerWeights_, edge)) {
      const auto *second = secondFor(data.lowerSecond, edge);
      for (const auto &[index, derivative] : *entries) {
        result.noalias() += lookup(second, index) * lowerLeft_.col(index) *
                            lowerBoundary_.row(index);
        result.noalias() += derivative * data.lowerLeftDot.col(index) *
                            lowerBoundary_.row(index);
      }
    }
    if (const auto *entries = derivativesFor(upperWeights_, edge)) {
      const auto *second = secondFor(data.upperSecond, edge);
      for (const auto &[index, derivative] : *entries) {
        const cd inverse = upperWeights_.inverseWeights[index];
        const cd velocity = data.upperVelocity[index];
        const cd secondDerivative = lookup(second, index);
        const cd factor = 2.0 * inverse * inverse * inverse * velocity *
                              derivative -
                          inverse * inverse * secondDerivative;
        result.noalias() +=
            factor * upperBoundary_.col(index) * upperRight_.row(index);
        result.noalias() += (-inverse * inverse * derivative) *
                            upperBoundary_.col(index) *
                            data.upperRightDot.row(index);
      }
    }
    return result;
  }

private:
  using IndexedDerivative = std::pair<int, cd>;

  struct WeightData {
    Eigen::ArrayXcd weights{};
    Eigen::ArrayXcd inverseWeights{};
    std::map<EdgeKey, std::vector<IndexedDerivative>> derivativesByEdge{};
  };

  [[nodiscard]] static const std::vector<IndexedDerivative> *secondFor(
      const std::map<EdgeKey, std::vector<IndexedDerivative>> &table,
      const EdgeKey &edge) {
    const auto found = table.find(edge);
    return found == table.end() ? nullptr : &found->second;
  }

  [[nodiscard]] static cd lookup(const std::vector<IndexedDerivative> *entries,
                                 int index) {
    if (entries == nullptr) return cd{0.0, 0.0};
    for (const auto &[candidate, value] : *entries)
      if (candidate == index) return value;
    return cd{0.0, 0.0};
  }

  // Wdot_j = sum_f v_f dW_j/dz_f, over the simplices `buildWeightData` admitted
  // (a pinned fallback weight has no derivative and no velocity).
  [[nodiscard]] Eigen::ArrayXcd weightVelocity(
      const WeightData &weights, int degree,
      const std::map<EdgeKey, cd> &direction) const {
    Eigen::ArrayXcd velocity =
        Eigen::ArrayXcd::Zero(weights.weights.size());
    if (degree < 1 || degree >= static_cast<int>(faces_.size()))
      return velocity;
    for (const auto &[edge, entries] : weights.derivativesByEdge) {
      const auto found = direction.find(edge);
      if (found == direction.end()) continue;
      for (const auto &[index, derivative] : entries)
        velocity[index] += found->second * derivative;
    }
    return velocity;
  }

  // The edge-keyed second weight derivative contracted against the direction,
  // from the simplex volume Hessian. The admission rule mirrors
  // `buildWeightData`, so the first- and second-derivative tables cover the
  // same (edge, index) pairs.
  [[nodiscard]] std::map<EdgeKey, std::vector<IndexedDerivative>> weightSecond(
      const WeightData &weights, int degree,
      const std::map<EdgeKey, cd> &direction) const {
    std::map<EdgeKey, std::vector<IndexedDerivative>> table;
    if (degree < 1 || degree >= static_cast<int>(faces_.size()))
      return table;
    const auto &degreeFaces = faces_[static_cast<std::size_t>(degree)];
    const int count = static_cast<int>(weights.weights.size());
    for (int index = 0;
         index < count && index < static_cast<int>(degreeFaces.size());
         ++index) {
      const auto &simplex = degreeFaces[static_cast<std::size_t>(index)];
      const cd volume = simplex->volume();
      const cd weight =
          convention_ == HodgeLaplacian::WeightConvention::SquaredContent
              ? volume * volume
              : volume;
      if (std::abs(weight) <= 0.0)
        continue;  // pinned to the constant fallback 1, as in buildWeightData
      const auto gradient = simplex->volumeGradient();
      const auto secondGradient =
          simplex->volumeGradientDirectionalDerivative(direction);
      cd volumeVelocity{0.0, 0.0};
      for (const auto &[edge, volumeDerivative] : gradient) {
        const auto found = direction.find(edge);
        if (found != direction.end())
          volumeVelocity += found->second * volumeDerivative;
      }
      for (const auto &[edge, volumeDerivative] : gradient) {
        const auto found = secondGradient.find(edge);
        const cd volumeSecond =
            found == secondGradient.end() ? cd{0.0, 0.0} : found->second;
        // w = V^2  ->  d(dw/dz_e) = 2 Vdot dV/dz_e + 2 V d(dV/dz_e)
        // w = V    ->  d(dw/dz_e) = d(dV/dz_e)
        const cd weightSecondDerivative =
            convention_ == HodgeLaplacian::WeightConvention::SquaredContent
                ? 2.0 * volumeVelocity * volumeDerivative +
                      2.0 * volume * volumeSecond
                : volumeSecond;
        // Mirror buildWeightData's admission test: an edge it dropped has no
        // gradient entry, so it gets no second one.
        const cd weightDerivative =
            convention_ == HodgeLaplacian::WeightConvention::SquaredContent
                ? 2.0 * volume * volumeDerivative
                : volumeDerivative;
        if (std::abs(weightDerivative) > 0.0)
          table[edge].emplace_back(index, weightSecondDerivative);
      }
    }
    return table;
  }

  [[nodiscard]] WeightData buildWeightData(int degree) const {
    WeightData data;
    const int count = degree >= 0 && degree <= dimension_
                          ? static_cast<int>(chainComplex_.numSimplices(degree))
                          : 0;
    data.weights = Eigen::ArrayXcd::Ones(std::max(count, 0));
    if (degree < 1 || degree >= static_cast<int>(faces_.size())) {
      data.inverseWeights = data.weights.inverse();
      return data;
    }

    const auto &degreeFaces = faces_[static_cast<std::size_t>(degree)];
    for (int index = 0;
         index < count && index < static_cast<int>(degreeFaces.size());
         ++index) {
      const auto &simplex = degreeFaces[static_cast<std::size_t>(index)];
      const cd volume = simplex->volume();
      const cd weight =
          convention_ == HodgeLaplacian::WeightConvention::SquaredContent
              ? volume * volume
              : volume;
      if (std::abs(weight) <= 0.0)
        continue; // pinned to the constant fallback 1
      data.weights[index] = weight;
      for (const auto &[edge, volumeDerivative] : simplex->volumeGradient()) {
        const cd weightDerivative =
            convention_ == HodgeLaplacian::WeightConvention::SquaredContent
                ? 2.0 * volume * volumeDerivative
                : volumeDerivative;
        if (std::abs(weightDerivative) > 0.0)
          data.derivativesByEdge[edge].emplace_back(index, weightDerivative);
      }
    }
    data.inverseWeights = data.weights.inverse();
    return data;
  }

  [[nodiscard]] Eigen::MatrixXcd boundaryMatrix(int degree, int rows,
                                                int columns) const {
    const std::vector<long> &flat = chainComplex_.boundaryMatrix(degree);
    Eigen::MatrixXcd boundary(rows, columns);
    for (int row = 0; row < rows; ++row)
      for (int column = 0; column < columns; ++column)
        boundary(row, column) = static_cast<double>(
            flat[static_cast<std::size_t>(row) * columns + column]);
    return boundary;
  }

  [[nodiscard]] static const std::vector<IndexedDerivative> *
  derivativesFor(const WeightData &data, const EdgeKey &edge) {
    const auto found = data.derivativesByEdge.find(edge);
    return found == data.derivativesByEdge.end() ? nullptr : &found->second;
  }

  ChainComplex chainComplex_;
  std::vector<std::vector<SimplexPtr>> faces_;
  int degree_{0};
  int dimension_{0};
  int degreeSize_{0};
  int lowerSize_{0};
  int upperSize_{0};
  HodgeLaplacian::WeightConvention convention_;
  WeightData lowerWeights_{};
  WeightData degreeWeights_{};
  WeightData upperWeights_{};
  Eigen::MatrixXcd lowerBoundary_{};
  Eigen::MatrixXcd upperBoundary_{};
  Eigen::MatrixXcd lowerBase_{};
  Eigen::MatrixXcd upperBase_{};
  Eigen::MatrixXcd lowerLeft_{};
  Eigen::MatrixXcd upperRight_{};
};

// Signed-weight (Lorentzian) metric Hodge Laplacian for k >= 0, assembled from
// the signed metric adjoint d_k* = W_k^{-1} d_k^T W_{k-1}:
//   L_k = W_k^{-1} d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^{-1} d_{k+1}^T W_k.
// At degree zero term 1 is absent and W_0 = I, leaving L_0 = d_1 W_1^{-1} d_1^T,
// whose row sums vanish identically. At positive weights
// W_k^{-1/2} L_k W_k^{1/2} = L_k^sym; with signed weights it is generally
// non-symmetric. Returns a |C_k| x |C_k| complex matrix, 0 x 0 if no k-cells.
// `metric == false` ⇒ unit weights (the positive combinatorial operator).
Eigen::MatrixXcd laplacianMatrix(const Spacetime &K, int k, bool metric,
                                 HodgeLaplacian::WeightConvention conv) {
  const ChainComplex cc = ChainComplex::fromSpacetime(K);
  const int n = cc.dimension();
  const int nk = static_cast<int>(cc.numSimplices(k));
  Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(nk, nk);
  if (nk == 0) return L;

  const std::vector<std::vector<SimplexPtr>> faces = orderedFaces(K);
  const auto weightArr = [&](int kk) {
    const std::vector<std::complex<double>> wv = simplexWeights(
        faces, kk, static_cast<int>(cc.numSimplices(kk)), metric, conv);
    Eigen::ArrayXcd a(static_cast<Eigen::Index>(wv.size()));
    for (std::size_t i = 0; i < wv.size(); ++i) a[static_cast<Eigen::Index>(i)] = wv[i];
    return a;
  };
  const auto boundary = [&](int kk, int rows, int cols) {
    const std::vector<long> &flat = cc.boundaryMatrix(kk);
    Eigen::MatrixXcd d(rows, cols);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        d(r, c) = static_cast<double>(flat[static_cast<std::size_t>(r) * cols + c]);
    return d;
  };

  const Eigen::ArrayXcd wk = weightArr(k);
  const Eigen::ArrayXcd invWk = wk.inverse();

  // Term 1: W_k^{-1} d_k^T W_{k-1} d_k (present once there are (k-1)-faces).
  const int rows = static_cast<int>(cc.numSimplices(k - 1));
  if (k >= 1 && rows > 0) {
    const Eigen::MatrixXcd dk = boundary(k, rows, nk);
    const Eigen::ArrayXcd wkm1 = weightArr(k - 1);
    L.noalias() += invWk.matrix().asDiagonal() * dk.transpose() *
                   wkm1.matrix().asDiagonal() * dk;
  }

  // Term 2: d_{k+1} W_{k+1}^{-1} d_{k+1}^T W_k (absent when there are no (k+1)-cells).
  const int cols = (k + 1 <= n) ? static_cast<int>(cc.numSimplices(k + 1)) : 0;
  if (cols > 0) {
    const Eigen::MatrixXcd dkp1 = boundary(k + 1, nk, cols);
    const Eigen::ArrayXcd invWkp1 = weightArr(k + 1).inverse();
    L.noalias() += dkp1 * invWkp1.matrix().asDiagonal() * dkp1.transpose() *
                   wk.matrix().asDiagonal();
  }
  return L;
}

struct SpectralEntropyData {
  Eigen::MatrixXcd laplacian{};
  Eigen::MatrixXcd entropyOperator{};
  Eigen::MatrixXcd entropyDerivative{};
  double entropy{0.0};
  bool zeroOperator{true};
  // Retained so the derivative of `entropyDerivative` can be formed without a
  // second eigensolve: the same decomposition, support mask and trace.
  Eigen::MatrixXcd eigenvectors{};
  Eigen::VectorXd eigenvalues{};
  std::vector<char> supported{};
  double trace{0.0};
};

// d(dS/dA)/dt on the fixed-rank stratum, given the positive operator's own
// velocity. With C = -(1/T)[log(A/T) + S P] and P the fixed support projector,
//   Cdot = -(Tdot/T) C - (1/T) P DLog(A/T)[Rdot] P - (Sdot/T) P,
// where DLog is the Frechet derivative of the matrix logarithm, the
// Daleckii-Krein divided differences of log in the eigenbasis.
Eigen::MatrixXcd entropyDerivativeVelocity(const SpectralEntropyData &data,
                                           const Eigen::MatrixXcd &velocity) {
  const Eigen::Index n = data.eigenvalues.size();
  if (n == 0 || data.zeroOperator || data.trace <= 0.0)
    return Eigen::MatrixXcd::Zero(velocity.rows(), velocity.cols());
  const double T = data.trace;
  // Hermitian part only, as the value path symmetrizes A the same way.
  const Eigen::MatrixXcd hermitianVelocity =
      0.5 * (velocity + velocity.adjoint());
  const Eigen::MatrixXcd inBasis =
      data.eigenvectors.adjoint() * hermitianVelocity * data.eigenvectors;
  const double traceVelocity = hermitianVelocity.trace().real();

  // pdot_i = (lambdadot_i - p_i Tdot)/T and Sdot = -sum_i log(p_i) pdot_i
  // (sum_i pdot_i = 0 exactly, so the +1 in d(-p log p) drops out).
  Eigen::VectorXd probability = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd probabilityVelocity = Eigen::VectorXd::Zero(n);
  double entropyVelocity = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    if (data.supported[static_cast<std::size_t>(i)] == 0) continue;
    probability[i] = std::max(data.eigenvalues[i], 0.0) / T;
    probabilityVelocity[i] =
        (inBasis(i, i).real() - probability[i] * traceVelocity) / T;
    entropyVelocity -= std::log(probability[i]) * probabilityVelocity[i];
  }

  // Rdot = (Adot - R Tdot)/T in the eigenbasis; R is diagonal there.
  Eigen::MatrixXcd logVelocity = Eigen::MatrixXcd::Zero(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    if (data.supported[static_cast<std::size_t>(i)] == 0) continue;
    for (Eigen::Index j = 0; j < n; ++j) {
      if (data.supported[static_cast<std::size_t>(j)] == 0) continue;
      const std::complex<double> rDot =
          (inBasis(i, j) -
           (i == j ? std::complex<double>{probability[i] * traceVelocity, 0.0}
                   : std::complex<double>{0.0, 0.0})) /
          T;
      // Divided difference of log; the coincident case is the derivative 1/p.
      const double gap = probability[i] - probability[j];
      const double dividedDifference =
          std::abs(gap) <= std::numeric_limits<double>::epsilon() *
                               std::max(probability[i], 1.0) * 64.0
              ? 1.0 / probability[i]
              : (std::log(probability[i]) - std::log(probability[j])) / gap;
      logVelocity(i, j) = dividedDifference * rDot;
    }
  }

  Eigen::MatrixXcd inBasisResult = -logVelocity / T;
  for (Eigen::Index i = 0; i < n; ++i) {
    if (data.supported[static_cast<std::size_t>(i)] == 0) continue;
    inBasisResult(i, i) -= std::complex<double>{entropyVelocity / T, 0.0};
  }
  Eigen::MatrixXcd result =
      data.eigenvectors * inBasisResult * data.eigenvectors.adjoint();
  result.noalias() -= (traceVelocity / T) * data.entropyDerivative;
  return result;
}

SpectralEntropyData
spectralEntropyData(Eigen::MatrixXcd laplacian,
                    HodgeLaplacian::EntropyPhaseMode phaseMode) {
  SpectralEntropyData data;
  data.laplacian = std::move(laplacian);
  if (data.laplacian.size() == 0)
    return data;
  if (data.laplacian.rows() != data.laplacian.cols())
    throw std::runtime_error(
        "HodgeLaplacian::spectralEntropy: Laplacian is not square");
  const auto n = static_cast<std::size_t>(data.laplacian.rows());
  if (!data.laplacian.allFinite())
    throw std::runtime_error(
        "HodgeLaplacian::spectralEntropy: non-finite Laplacian");

  data.entropyOperator = data.laplacian;
  if (phaseMode == HodgeLaplacian::EntropyPhaseMode::IgnoreComplexPhase)
    data.entropyOperator = data.laplacian.cwiseAbs().cast<cd>();

  Eigen::MatrixXcd positive =
      data.entropyOperator.adjoint() * data.entropyOperator;
  // Remove only roundoff-level anti-Hermitian noise before the self-adjoint solve.
  positive = 0.5 * (positive + positive.adjoint());
  const double trace = positive.trace().real();
  if (!std::isfinite(trace))
    throw std::runtime_error(
        "HodgeLaplacian::spectralEntropy: non-finite positive-operator trace");
  if (trace <= 0.0) {
    data.entropyDerivative = Eigen::MatrixXcd::Zero(
        static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    return data;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(positive);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error(
        "HodgeLaplacian::spectralEntropy: eigendecomposition failed");

  const Eigen::VectorXd eigenvalues = solver.eigenvalues();
  const double supportTolerance =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<std::size_t>(n, 1)) *
      std::max(trace, 1.0) * 64.0;
  data.eigenvalues = eigenvalues;
  data.eigenvectors = solver.eigenvectors();
  data.trace = trace;
  data.supported.assign(n, 0);
  Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(n));
  for (std::size_t index = 0; index < n; ++index) {
    const double eigenvalue =
        std::max(eigenvalues[static_cast<Eigen::Index>(index)], 0.0);
    if (eigenvalue <= supportTolerance) continue;
    data.supported[index] = 1;
    const double probability = eigenvalue / trace;
    data.entropy -= probability * std::log(probability);
  }
  for (std::size_t index = 0; index < n; ++index) {
    const double eigenvalue =
        std::max(eigenvalues[static_cast<Eigen::Index>(index)], 0.0);
    if (eigenvalue <= supportTolerance) continue;
    const double probability = eigenvalue / trace;
    coefficients[static_cast<Eigen::Index>(index)] =
        -(std::log(probability) + data.entropy) / trace;
  }
  data.entropyDerivative =
      solver.eigenvectors() * coefficients.asDiagonal() *
      solver.eigenvectors().adjoint();
  data.zeroOperator = false;
  return data;
}

// Row-major flat operator to a dense square matrix; `context` names the caller
// in the not-square message.
Eigen::MatrixXcd squareFromFlat(const std::vector<cd> &flat,
                                const char *context) {
  const auto n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  if (n * n != flat.size())
    throw std::runtime_error(std::string(context) +
                             ": Laplacian is not square");

  Eigen::MatrixXcd laplacian(static_cast<Eigen::Index>(n),
                             static_cast<Eigen::Index>(n));
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t column = 0; column < n; ++column)
      laplacian(static_cast<Eigen::Index>(row),
                static_cast<Eigen::Index>(column)) = flat[row * n + column];
  return laplacian;
}

SpectralEntropyData
spectralEntropyData(const std::vector<cd> &flat,
                    HodgeLaplacian::EntropyPhaseMode phaseMode) {
  if (flat.empty())
    return {};
  return spectralEntropyData(
      squareFromFlat(flat, "HodgeLaplacian::spectralEntropy"), phaseMode);
}

// Spectral data for the C* connection operator, read from its eigenvalues.
//
// The weights are |lambda|^2, not the eigenvalues of A = M^dag M: eigenvalues
// survive the non-unitary C* gauge similarity diag(g)^-1 (.) diag(g) and the
// singular values A depends on do not. Squaring makes the two agree in the
// Hermitian limit, where |lambda_i|^2 = sigma_i^2. See the header.
struct ConnectionEntropyData {
  Eigen::MatrixXcd laplacian{};
  Eigen::MatrixXcd eigenvectors{};
  Eigen::VectorXcd eigenvalues{};
  // dS/d(|lambda_k|^2) on the supported stratum, zero off it.
  Eigen::VectorXd squaredModulusDerivative{};
  double entropy{0.0};
  bool zeroOperator{true};
};

ConnectionEntropyData connectionEntropyData(Eigen::MatrixXcd laplacian) {
  ConnectionEntropyData data;
  data.laplacian = std::move(laplacian);
  if (data.laplacian.size() == 0)
    return data;
  if (data.laplacian.rows() != data.laplacian.cols())
    throw std::runtime_error(
        "HodgeLaplacian::connectionSpectralEntropy: Laplacian is not square");
  if (!data.laplacian.allFinite())
    throw std::runtime_error(
        "HodgeLaplacian::connectionSpectralEntropy: non-finite Laplacian");
  const Eigen::Index n = data.laplacian.rows();

  // General complex eigensolve: the operator is non-Hermitian as soon as the
  // phase or a weight is complex.
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(data.laplacian);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error(
        "HodgeLaplacian::connectionSpectralEntropy: eigendecomposition failed");
  data.eigenvalues = solver.eigenvalues();
  data.eigenvectors = solver.eigenvectors();
  data.squaredModulusDerivative = Eigen::VectorXd::Zero(n);

  const Eigen::VectorXd squaredModulus = data.eigenvalues.cwiseAbs2();
  const double total = squaredModulus.sum();
  if (!std::isfinite(total))
    throw std::runtime_error(
        "HodgeLaplacian::connectionSpectralEntropy: non-finite spectral sum");
  if (total <= 0.0)
    return data;

  // Same support convention the M^dag M path uses: p log p -> 0 at the floor.
  const double supportTolerance =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(n, 1)) *
      std::max(total, 1.0) * 64.0;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (squaredModulus[i] <= supportTolerance) continue;
    const double probability = squaredModulus[i] / total;
    data.entropy -= probability * std::log(probability);
  }
  for (Eigen::Index i = 0; i < n; ++i) {
    if (squaredModulus[i] <= supportTolerance) continue;
    // S = -T/A + log A with T = sum_i a_i log a_i and A = sum_i a_i gives
    // dS/da_k = T/A^2 - log(a_k)/A = -(S + log p_k)/A, for a_k = |lambda_k|^2.
    data.squaredModulusDerivative[i] =
        -(std::log(squaredModulus[i] / total) + data.entropy) / total;
  }
  data.zeroOperator = false;
  return data;
}

ConnectionEntropyData connectionEntropyData(const std::vector<cd> &flat) {
  if (flat.empty())
    return {};
  return connectionEntropyData(
      squareFromFlat(flat, "HodgeLaplacian::connectionSpectralEntropy"));
}

}  // namespace

HodgeLaplacian::WeightConvention HodgeLaplacian::defaultWeightConvention_ =
    HodgeLaplacian::WeightConvention::SquaredContent;

HodgeLaplacian::MetricSource HodgeLaplacian::defaultMetricSource_ =
    HodgeLaplacian::MetricSource::DiagonalWeights;

/// The Whitney pencil of the current geometry: chain complex, dressed operator
/// and orientation signs, rebuilt whenever the structural revision or any edge's
/// length or phase revision moves.
struct HodgeLaplacian::WhitneyState {
  std::uint64_t stamp{0};
  cobordism::ChainComplex complex;
  std::vector<std::vector<std::uint64_t>> edges;
  std::unordered_map<std::uint64_t, std::size_t> edgeIndex;  // key(a,b) -> canonical edge
  std::shared_ptr<chainhodge::CovariantChainHodge> op;
  std::shared_ptr<chainhodge::ChainHodge> base;
  chainhodge::InstanceCertificate certificate;
  // D_k: stored orientation relative to the reference orientation, per degree.
  // Every operator of the pencil is reported in the stored basis, D L^ref D.
  std::vector<Eigen::VectorXd> signs;
  [[nodiscard]] Eigen::MatrixXcd toStored(int k, const Eigen::MatrixXcd &ref) const {
    if (k < 0 || k >= static_cast<int>(signs.size()) || ref.rows() == 0) return ref;
    const Eigen::VectorXd &d = signs[static_cast<std::size_t>(k)];
    return d.asDiagonal() * ref * d.asDiagonal();
  }
};

namespace {
std::uint64_t geometryStamp(const Spacetime &st) {
  std::uint64_t stamp = st.structuralRevision() * 0x9e3779b97f4a7c15ULL;
  if (st.getEdgeList())
    for (const auto &e : st.getEdgeList()->toVector())
      if (e != nullptr) stamp += 3 * e->lengthRevision() + 7 * e->phaseRevision() + 11;
  return stamp;
}
std::uint64_t edgeKeyOf(std::uint64_t a, std::uint64_t b) {
  if (a > b) std::swap(a, b);
  return (a << 32) ^ b;
}
}  // namespace

const HodgeLaplacian::WhitneyState &HodgeLaplacian::whitneyState() const {
  if (!st_) throw std::runtime_error("HodgeLaplacian: no spacetime bound");
  const std::uint64_t stamp = geometryStamp(*st_);
  if (whitney_ && whitney_->stamp == stamp) return *whitney_;
  auto w = std::make_shared<WhitneyState>();
  w->stamp = stamp;
  w->complex = chainhodge::WhitneyMass::complexOf(*st_);
  // The pencil is in the reference orientation (ascending vertex id) while
  // consumers index cells by ChainComplex::fromSpacetime's basis, so the two
  // boundary maps must coincide. Refused by name otherwise.
  const ChainComplex stored = ChainComplex::fromSpacetime(*st_);
  if (stored.dimension() != w->complex.dimension())
    throw std::runtime_error("HodgeLaplacian: WhitneyPencil — the spacetime's chain complex and the "
                             "reference-oriented complex differ in dimension");
  for (int k = 0; k <= stored.dimension(); ++k)
    if (stored.kSimplexVertices(k) != w->complex.kSimplexVertices(k))
      throw std::runtime_error(
          "HodgeLaplacian: WhitneyPencil — the spacetime's cell order differs from the canonical "
          "order at degree " + std::to_string(k));
  // The stored orientations may differ from the reference (ascending id) ones
  // by a sign per cell, cells created by surgery keeping their creation order.
  // orientationSigns derives and verifies those signs from the stored maps.
  const auto signs = stored.orientationSigns();
  w->signs.reserve(signs.size());
  for (const auto &sk : signs) {
    Eigen::VectorXd d(static_cast<Eigen::Index>(sk.size()));
    for (std::size_t j = 0; j < sk.size(); ++j) d(static_cast<Eigen::Index>(j)) = static_cast<double>(sk[j]);
    w->signs.push_back(std::move(d));
  }
  w->edges = w->complex.kSimplexVertices(1);
  for (std::size_t j = 0; j < w->edges.size(); ++j)
    w->edgeIndex[edgeKeyOf(w->edges[j][0], w->edges[j][1])] = j;
  const chainhodge::SquaredLengths s = chainhodge::WhitneyMass::squaredLengthsOf(*st_, w->complex);
  w->base = std::make_shared<chainhodge::ChainHodge>(
      w->complex, s, chainhodge::Preset::L2, chainhodge::Branch::Continuation,
      std::numeric_limits<int>::max());
  w->certificate = w->base->certificate();
  const chainhodge::Connection U = chainhodge::Connection::fromSpacetime(*st_, w->complex);
  w->op = std::make_shared<chainhodge::CovariantChainHodge>(*w->base, U, 7, /*measureCertificate=*/false);
  whitney_ = std::move(w);
  return *whitney_;
}

Eigen::MatrixXcd HodgeLaplacian::operatorMatrix(int k, bool metric) const {
  if (metricSource_ == MetricSource::WhitneyPencil && metric) {
    const WhitneyState &w = whitneyState();
    if (k > w.complex.dimension()) return Eigen::MatrixXcd();
    // The operator on geometric images, L_z = (M^U)^{-1} h M^U; its kernel
    // vectors are the images z = G h.
    const Eigen::MatrixXcd h = w.op->covariantOperator(k);
    const Eigen::MatrixXcd hM = h * w.op->dressed(k);
    return w.toStored(k, w.op->applyG(k, hM));
  }
  return laplacianMatrix(*st_, k, metric, weightConvention_);
}

double HodgeLaplacian::kontsevichSegalMargin(const Spacetime &st) {
  const ChainComplex K = chainhodge::WhitneyMass::complexOf(st);
  if (K.dimension() < 0) return std::numeric_limits<double>::infinity();
  return chainhodge::WhitneyMass::allowabilityMargin(
      K, chainhodge::WhitneyMass::squaredLengthsOf(st, K));
}

HodgeLaplacian::HodgeLaplacian(std::shared_ptr<Spacetime> st,
                               WeightConvention weights, MetricSource source)
    : st_(std::move(st)), weightConvention_(weights), metricSource_(source) {
  if (!st_) {
    sharedSpectra_ = std::make_shared<SharedSpectrumMap>();
    return;
  }
  // Adopt the spacetime's shared spectrum map when its revision stamp is
  // current; otherwise start a fresh one and stamp it. This instance keeps the
  // map it captured across later geometry changes.
  if (auto slot = std::static_pointer_cast<SharedSpectrumMap>(
          st_->cachedSpectralSlot())) {
    sharedSpectra_ = std::move(slot);
  } else {
    sharedSpectra_ = std::make_shared<SharedSpectrumMap>();
    st_->storeSpectralSlot(sharedSpectra_);
  }
  // Stable vertex order: sort by id, then id -> 0..N-1.
  const auto &verts = st_->getVertexList()->toVector();
  ids_.reserve(verts.size());
  for (const auto v : verts)
    if (v != nullptr) ids_.push_back(v->getId());
  std::sort(ids_.begin(), ids_.end());
  order_ = ids_.size();
  idToIndex_.reserve(order_);
  for (std::size_t i = 0; i < order_; ++i) idToIndex_[ids_[i]] = i;
}

void HodgeLaplacian::requireNonNegativeDegree(int k) {
  if (k < 0)
    throw std::runtime_error(
        "HodgeLaplacian: degree k must be non-negative; requested k=" +
        std::to_string(k));
}

std::vector<std::vector<std::uint64_t>> HodgeLaplacian::cochainOrdering(
    int k, bool useVertexSet) const {
  std::vector<std::vector<std::uint64_t>> ord;
  if (k < 0 || !st_) return ord;
  if (useVertexSet && k == 0) {
    // The Hermitian U(1) connection operator is indexed over the full sorted-id
    // vertex set, including lone vertices ChainComplex omits.
    ord.reserve(ids_.size());
    for (const std::uint64_t id : ids_) ord.push_back({id});
    return ord;
  }
  // Eigenvector components are indexed in the canonical ChainComplex k-simplex
  // column order, kSimplexVertices(k), whose count matches numSimplices(k).
  return ChainComplex::fromSpacetime(*st_).kSimplexVertices(k);
}

Spectrum HodgeLaplacian::makeSpectrum(
    int degree, std::vector<std::vector<std::uint64_t>> ordering,
    const std::vector<cd> &evals, const std::vector<cd> &evecsFlat, int dim,
    bool hermitian) {
  if (dim <= 0) return Spectrum();
  if (ordering.size() != static_cast<std::size_t>(dim))
    throw std::runtime_error(
        "HodgeLaplacian::makeSpectrum: k-simplex ordering size " +
        std::to_string(ordering.size()) + " != operator dimension " +
        std::to_string(dim));
  Eigen::VectorXcd eigenvalues(dim);
  for (int j = 0; j < dim; ++j)
    eigenvalues[j] = evals[static_cast<std::size_t>(j)];
  std::vector<Cochain> eigenvectors;
  eigenvectors.reserve(static_cast<std::size_t>(dim));
  for (int j = 0; j < dim; ++j) {
    Eigen::VectorXcd col(dim);
    for (int i = 0; i < dim; ++i)
      col[i] = evecsFlat[static_cast<std::size_t>(i) * dim + j];
    eigenvectors.emplace_back(degree, ordering, std::move(col));
  }
  return Spectrum(std::move(eigenvalues), std::move(eigenvectors), hermitian);
}

void HodgeLaplacian::assemble(std::vector<cd> &A, std::vector<double> &D) const {
  const std::size_t N = order_;
  A.assign(N * N, cd(0.0, 0.0));
  D.assign(N, 0.0);
  if (N == 0 || !st_) return;

  for (const EdgePtr e : st_->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const VertexPtr s = e->getSource();
    const VertexPtr t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    const auto is = idToIndex_.find(s->getId());
    const auto it = idToIndex_.find(t->getId());
    if (is == idToIndex_.end() || it == idToIndex_.end()) continue;
    const std::size_t i = is->second;
    const std::size_t j = it->second;
    if (i == j) continue;  // a simplicial complex carries no self-loops

    const cd w = (e->getLength() * e->getLength());  // exact complex squared length l^2
    const cd phase = e->getPhase();  // C* connection on src->tgt

    // Degree uses the magnitude convention D_ii = sum |squaredLength| over
    // incident edges: phase-independent, and Hermitian with unitary e^{-iLt} in
    // the real-phase, positive-weight case.
    D[i] += std::abs(w);
    D[j] += std::abs(w);

    // The link variable U = e^{i*phase} in C*; the reverse orientation carries
    // its inverse U^{-1} = e^{-i*phase}, never its conjugate, since
    // conj(g) != g^{-1} once the gauge function leaves U(1). A gauge
    // transformation U_ij -> g_i^{-1} U_ij g_j then acts on A by the similarity
    // diag(g)^{-1} A diag(g), so the spectrum is gauge-invariant exactly. Only
    // the link is inverted; the geometry keeps its conjugate, so a real phase
    // reproduces the Hermitian magnetic operator entry for entry.
    const cd link = std::exp(cd(0.0, 1.0) * phase);
    const cd linkInverse = std::exp(cd(0.0, -1.0) * phase);
    A[i * N + j] += w * link;
    A[j * N + i] += std::conj(w) * linkInverse;
  }
}

std::vector<cd> HodgeLaplacian::adjacency() const {
  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  return A;
}

std::vector<double> HodgeLaplacian::degree() const {
  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  return D;
}

std::vector<cd> HodgeLaplacian::laplacian(int k, bool metric) const {
  requireNonNegativeDegree(k);
  // The signed-weight d'Alembertian, complex and generally non-symmetric:
  // L_k = W_k^-1 d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^-1 d_{k+1}^T W_k.
  // The Hermitian U(1) graph Laplacian D - A is connectionLaplacian().
  if (!st_) return {};
  const Eigen::MatrixXcd L = operatorMatrix(k, metric);
  const int nk = static_cast<int>(L.rows());
  std::vector<cd> out(static_cast<std::size_t>(nk) * nk, cd(0.0, 0.0));
  for (int i = 0; i < nk; ++i)
    for (int j = 0; j < nk; ++j)
      out[static_cast<std::size_t>(i) * nk + j] = L(i, j);
  return out;
}

std::vector<cd> HodgeLaplacian::connectionLaplacian() const {
  // L^U(1) = D - A. Off-diagonal entries are -A_ij; the diagonal carries D_ii
  // (the adjacency has no diagonal without self-loops). Not L_0: the diagonal
  // is the magnitude sum and the off-diagonal the signed complex weight, so the
  // row sums do not vanish once a squared length is negative or complex.
  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  const std::size_t N = order_;
  std::vector<cd> L(N * N, cd(0.0, 0.0));
  for (std::size_t idx = 0; idx < A.size(); ++idx) L[idx] = -A[idx];
  for (std::size_t i = 0; i < N; ++i) L[i * N + i] += D[i];
  return L;
}

std::vector<std::complex<double>> HodgeLaplacian::weights(int k) const {
  if (k < 0 || !st_) return {};
  const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
  if (k > cc.dimension()) return {};
  const int m = static_cast<int>(cc.numSimplices(k));
  // W_0 = I, the unit weight on 0-chains, is what makes the row sums of
  // L_0 = d_1 W_1^-1 d_1^T W_0 vanish identically.
  if (k == 0)
    return std::vector<std::complex<double>>(static_cast<std::size_t>(m),
                                             std::complex<double>{1.0, 0.0});
  return simplexWeights(orderedFaces(*st_), k, m, /*metric=*/true, weightConvention_);
}

std::vector<std::complex<double>> HodgeLaplacian::laplacianGradient(
    int k, std::uint64_t ea, std::uint64_t eb) const {
  if (k < 0 || !st_) return {};
  Eigen::MatrixXcd dL;
  if (metricSource_ == MetricSource::WhitneyPencil) {
    const WhitneyState &w = whitneyState();
    const auto it = w.edgeIndex.find(edgeKeyOf(ea, eb));
    if (it == w.edgeIndex.end() || k > w.complex.dimension()) {
      const int nk = static_cast<int>(w.complex.numSimplices(k));
      return std::vector<std::complex<double>>(static_cast<std::size_t>(nk) * nk,
                                               std::complex<double>{0.0, 0.0});
    }
    // d L_z = M^{-1} [ -dM L_z + dh M + h dM ] with L_z = M^{-1} h M.
    const Eigen::MatrixXcd h = w.op->covariantOperator(k);
    const chainhodge::SparseMatrix &M = w.op->dressed(k);
    const Eigen::MatrixXcd Lz = w.op->applyG(k, Eigen::MatrixXcd(h * M));
    const chainhodge::SparseMatrix dM = w.op->dressedDerivative(k, it->second);
    const Eigen::MatrixXcd dh = w.op->covariantOperatorDerivative(k, it->second);
    const Eigen::MatrixXcd inner = -Eigen::MatrixXcd(dM * Lz) + Eigen::MatrixXcd(dh * M) + Eigen::MatrixXcd(h * dM);
    dL = w.toStored(k, w.op->applyG(k, inner));
  } else {
    const LaplacianDerivativeWorkspace workspace(*st_, k, weightConvention_);
    dL = workspace.gradient(ea, eb);
  }
  const int nk = static_cast<int>(dL.rows());
  std::vector<std::complex<double>> out(static_cast<std::size_t>(nk) * nk,
                                        std::complex<double>{0.0, 0.0});
  for (int i = 0; i < nk; ++i)
    for (int j = 0; j < nk; ++j)
      out[static_cast<std::size_t>(i) * nk + j] = dL(i, j);
  return out;
}

std::vector<std::complex<double>> HodgeLaplacian::laplacianPhaseGradient(
    int k, std::uint64_t ea, std::uint64_t eb) const {
  if (k < 0 || !st_) return {};
  if (metricSource_ != MetricSource::WhitneyPencil) {
    const ChainComplex cc = ChainComplex::fromSpacetime(*st_);
    const int nk = (k <= cc.dimension()) ? static_cast<int>(cc.numSimplices(k)) : 0;
    return std::vector<std::complex<double>>(static_cast<std::size_t>(nk) * nk,
                                             std::complex<double>{0.0, 0.0});
  }
  const WhitneyState &w = whitneyState();
  const int nk = (k <= w.complex.dimension()) ? static_cast<int>(w.complex.numSimplices(k)) : 0;
  std::vector<std::complex<double>> out(static_cast<std::size_t>(nk) * nk,
                                        std::complex<double>{0.0, 0.0});
  const auto it = w.edgeIndex.find(edgeKeyOf(ea, eb));
  if (it == w.edgeIndex.end() || nk == 0) return out;
  const Eigen::MatrixXcd h = w.op->covariantOperator(k);
  const chainhodge::SparseMatrix &M = w.op->dressed(k);
  const Eigen::MatrixXcd Lz = w.op->applyG(k, Eigen::MatrixXcd(h * M));
  const chainhodge::SparseMatrix dM = w.op->dressedPhaseDerivative(k, it->second);
  const Eigen::MatrixXcd dh = w.op->covariantOperatorPhaseDerivative(k, it->second);
  const Eigen::MatrixXcd inner = -Eigen::MatrixXcd(dM * Lz) + Eigen::MatrixXcd(dh * M) + Eigen::MatrixXcd(h * dM);
  const Eigen::MatrixXcd dL = w.toStored(k, w.op->applyG(k, inner));
  for (int i = 0; i < nk; ++i)
    for (int j = 0; j < nk; ++j)
      out[static_cast<std::size_t>(i) * nk + j] = dL(i, j);
  return out;
}

double HodgeLaplacian::spectralEntropy(int k,
                                      EntropyPhaseMode phaseMode) const {
  requireNonNegativeDegree(k);
  return spectralEntropyData(laplacian(k, /*metric=*/true), phaseMode).entropy;
}

double HodgeLaplacian::connectionSpectralEntropy() const {
  // -sum p log p over the normalized eigenvalue moduli. No EntropyPhaseMode
  // here: the entrywise |.| ablation would erase the dependence being measured.
  // See `connectionEntropyData`.
  return connectionEntropyData(connectionLaplacian()).entropy;
}

std::vector<std::complex<double>>
HodgeLaplacian::connectionSpectralEntropyPhaseGradient() const {
  const auto edges = st_ && st_->getEdgeList()
                         ? st_->getEdgeList()->toVector()
                         : std::vector<EdgePtr>{};
  std::vector<cd> gradient(edges.size(), cd{0.0, 0.0});
  if (!st_ || order_ == 0) return gradient;

  const ConnectionEntropyData data =
      connectionEntropyData(connectionLaplacian());
  if (data.laplacian.size() == 0 || data.zeroOperator) return gradient;

  // Each simple eigenvalue moves holomorphically,
  //   dlambda_k = u_k^dag (dL) v_k / (u_k^dag v_k),
  // with normalization 1 since V^-1 supplies the left eigenvectors. The squared
  // modulus is the only non-holomorphic step: with a_k = lambda_k conj(lambda_k)
  // and u = conj(lambda_k) dlambda_k,
  //   da_k/dx = 2 Re(u),   da_k/dy = -2 Im(u),
  // so in the h = S_x - i S_y convention h = sum_k beta_k dlambda_k with
  // beta_k = 2 (dS/da_k) conj(lambda_k). Contracting the sum over k once into
  // P = V diag(beta) V^-1 leaves O(1) work per edge.
  const Eigen::Index n = data.eigenvalues.size();
  Eigen::VectorXcd beta = Eigen::VectorXcd::Zero(n);
  for (Eigen::Index index = 0; index < n; ++index)
    beta[index] = 2.0 * data.squaredModulusDerivative[index] *
                  std::conj(data.eigenvalues[index]);
  const Eigen::MatrixXcd contraction =
      data.eigenvectors * beta.asDiagonal() * data.eigenvectors.inverse();
  // The perturbation formula assumes simple eigenvalues. A defective operator
  // has no eigenbasis to invert and `inverse()` reports that as non-finite;
  // throw rather than return NaN gradients.
  if (!contraction.allFinite())
    throw std::runtime_error(
        "HodgeLaplacian::connectionSpectralEntropyPhaseGradient: the "
        "connection operator is not diagonalizable at this geometry");
  const cd imaginaryUnit{0.0, 1.0};

  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto *edge = edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const auto is = idToIndex_.find(edge->getSource()->getId());
    const auto it = idToIndex_.find(edge->getTarget()->getId());
    if (is == idToIndex_.end() || it == idToIndex_.end()) continue;
    const auto i = static_cast<Eigen::Index>(is->second);
    const auto j = static_cast<Eigen::Index>(it->second);
    if (i == j) continue;  // no self-loops in a simplicial complex

    // L = D - A. The diagonal is the magnitude sum and carries no phase, so
    // only the two off-diagonal entries move:
    //   L_ij = -w e^{i phi}      => dL_ij/dphi =  i L_ij
    //   L_ji = -conj(w) e^{-i phi} => dL_ji/dphi = -i L_ji
    // Both are holomorphic; no conj(phi) appears. Contracting dL/dphi against P
    // collapses to these two terms.
    const cd lowerLeft = data.laplacian(j, i);
    const cd upperRight = data.laplacian(i, j);
    gradient[edgeIndex] = contraction(j, i) * (imaginaryUnit * upperRight) +
                          contraction(i, j) * (-imaginaryUnit * lowerLeft);
  }
  return gradient;
}

double HodgeLaplacian::connectionSpectralEntropyPhaseGradientNorm() const {
  double total = 0.0;
  for (const cd &component : connectionSpectralEntropyPhaseGradient())
    total += std::norm(component);
  return total;
}

std::vector<std::complex<double>> HodgeLaplacian::spectralEntropyGradient(
    int k, EntropyPhaseMode phaseMode) const {
  requireNonNegativeDegree(k);
  // At degree zero L_0 = d_1 W_1^-1 d_1^T is holomorphic in z = l^2, so the
  // same workspace derivative applies.
  const auto edges = st_ && st_->getEdgeList()
                         ? st_->getEdgeList()->toVector()
                         : std::vector<EdgePtr>{};
  std::vector<cd> gradient(edges.size(), cd{0.0, 0.0});
  if (!st_)
    return gradient;
  const LaplacianDerivativeWorkspace workspace(*st_, k, weightConvention_);
  const SpectralEntropyData data =
      spectralEntropyData(workspace.laplacian(), phaseMode);
  if (data.laplacian.size() == 0 || data.zeroOperator) return gradient;
  const Eigen::Index n = data.laplacian.rows();

  Eigen::MatrixXcd fullPhaseLeft;
  Eigen::MatrixXd magnitudeDerivative;
  if (phaseMode == EntropyPhaseMode::IncludeComplexPhase) {
    fullPhaseLeft = data.entropyDerivative * data.laplacian.adjoint();
  } else {
    // M=|L| is real entrywise. For A=M^T M, dS/dM=2 M C.
    magnitudeDerivative =
        2.0 * data.entropyOperator.real() * data.entropyDerivative.real();
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
  for (std::int64_t edgeIndex = 0;
       edgeIndex < static_cast<std::int64_t>(edges.size()); ++edgeIndex) {
    const auto *edge = edges[static_cast<std::size_t>(edgeIndex)];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const Eigen::MatrixXcd derivative = workspace.gradient(
        edge->getSource()->getId(), edge->getTarget()->getId());

    if (phaseMode == EntropyPhaseMode::IncludeComplexPhase) {
      // dS = 2 Re Tr(C L^dagger dL), so in the h = S_x - i S_y convention
      // h = 2 Tr(C L^dagger dL/dz).
      gradient[static_cast<std::size_t>(edgeIndex)] =
          2.0 * (fullPhaseLeft.array() * derivative.transpose().array()).sum();
    } else {
      // d|L_ij| = Re(conj(L_ij)/|L_ij| dL_ij), contracted with dS/dM; a zero
      // entry contributes zero on this fixed support stratum.
      cd component{0.0, 0.0};
      for (Eigen::Index row = 0; row < n; ++row)
        for (Eigen::Index column = 0; column < n; ++column) {
          const cd value = data.laplacian(row, column);
          const double magnitude = std::abs(value);
          if (magnitude <= 0.0) continue;
          component += magnitudeDerivative(row, column) *
                       (std::conj(value) / magnitude) *
                       derivative(row, column);
        }
      gradient[static_cast<std::size_t>(edgeIndex)] = component;
    }
  }
  return gradient;
}

std::vector<std::complex<double>>
HodgeLaplacian::spectralEntropyGradientDirectionalDerivative(
    int k, const std::vector<std::complex<double>> &direction,
    EntropyPhaseMode phaseMode) const {
  requireNonNegativeDegree(k);
  const auto edges = st_ && st_->getEdgeList()
                         ? st_->getEdgeList()->toVector()
                         : std::vector<EdgePtr>{};
  if (direction.size() != edges.size())
    throw std::runtime_error(
        "HodgeLaplacian::spectralEntropyGradientDirectionalDerivative: "
        "direction has " +
        std::to_string(direction.size()) + " entries, expected " +
        std::to_string(edges.size()));
  std::vector<cd> velocityOfGradient(edges.size(), cd{0.0, 0.0});
  if (!st_)
    return velocityOfGradient;

  // The direction, keyed the way the simplices key their own gradients.
  std::map<std::pair<std::uint64_t, std::uint64_t>, cd> keyedDirection;
  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto *edge = edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const std::uint64_t a = edge->getSource()->getId();
    const std::uint64_t b = edge->getTarget()->getId();
    keyedDirection[{std::min(a, b), std::max(a, b)}] += direction[edgeIndex];
  }

  const LaplacianDerivativeWorkspace workspace(*st_, k, weightConvention_);
  const SpectralEntropyData data =
      spectralEntropyData(workspace.laplacian(), phaseMode);
  if (data.laplacian.size() == 0 || data.zeroOperator)
    return velocityOfGradient;
  const Eigen::Index n = data.laplacian.rows();
  const auto directionData = workspace.directionData(keyedDirection);

  // Ldot = sum_f v_f dL/dz_f, one pass over the per-edge derivatives.
  Eigen::MatrixXcd laplacianVelocity = Eigen::MatrixXcd::Zero(n, n);
  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto *edge = edges[edgeIndex];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr || direction[edgeIndex] == cd{0.0, 0.0})
      continue;
    laplacianVelocity.noalias() +=
        direction[edgeIndex] * workspace.gradient(edge->getSource()->getId(),
                                                  edge->getTarget()->getId());
  }

  Eigen::MatrixXcd fullPhaseLeft;
  Eigen::MatrixXcd fullPhaseLeftVelocity;
  Eigen::MatrixXd magnitudeDerivative;
  Eigen::MatrixXd magnitudeDerivativeVelocity;
  Eigen::MatrixXcd phaseFactor;
  Eigen::MatrixXcd phaseFactorVelocity;
  if (phaseMode == EntropyPhaseMode::IncludeComplexPhase) {
    // A = L^dagger L, so Adot = Ldot^dagger L + L^dagger Ldot.
    const Eigen::MatrixXcd positiveVelocity =
        laplacianVelocity.adjoint() * data.laplacian +
        data.laplacian.adjoint() * laplacianVelocity;
    const Eigen::MatrixXcd entropyDerivativeVelocityMatrix =
        entropyDerivativeVelocity(data, positiveVelocity);
    fullPhaseLeft = data.entropyDerivative * data.laplacian.adjoint();
    fullPhaseLeftVelocity =
        entropyDerivativeVelocityMatrix * data.laplacian.adjoint() +
        data.entropyDerivative * laplacianVelocity.adjoint();
  } else {
    // M = |L| entrywise, d|L_ij| = Re(conj(L_ij)/|L_ij| dL_ij): the unit phase
    // factor and the magnitude both carry a velocity.
    const Eigen::MatrixXcd &L = data.laplacian;
    phaseFactor = Eigen::MatrixXcd::Zero(n, n);
    phaseFactorVelocity = Eigen::MatrixXcd::Zero(n, n);
    Eigen::MatrixXd magnitudeVelocity = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index row = 0; row < n; ++row)
      for (Eigen::Index column = 0; column < n; ++column) {
        const cd value = L(row, column);
        const double magnitude = std::abs(value);
        if (magnitude <= 0.0) continue;  // fixed support stratum
        const cd unit = std::conj(value) / magnitude;
        phaseFactor(row, column) = unit;
        const double magnitudeDot =
            (unit * laplacianVelocity(row, column)).real();
        magnitudeVelocity(row, column) = magnitudeDot;
        phaseFactorVelocity(row, column) =
            (std::conj(laplacianVelocity(row, column)) - unit * magnitudeDot) /
            magnitude;
      }
    const Eigen::MatrixXd magnitudeOperator = data.entropyOperator.real();
    const Eigen::MatrixXd positiveVelocityReal =
        magnitudeVelocity.transpose() * magnitudeOperator +
        magnitudeOperator.transpose() * magnitudeVelocity;
    const Eigen::MatrixXcd entropyDerivativeVelocityMatrix =
        entropyDerivativeVelocity(data, positiveVelocityReal.cast<cd>());
    magnitudeDerivative =
        2.0 * magnitudeOperator * data.entropyDerivative.real();
    magnitudeDerivativeVelocity =
        2.0 * (magnitudeVelocity * data.entropyDerivative.real() +
               magnitudeOperator * entropyDerivativeVelocityMatrix.real());
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
  for (std::int64_t edgeIndex = 0;
       edgeIndex < static_cast<std::int64_t>(edges.size()); ++edgeIndex) {
    const auto *edge = edges[static_cast<std::size_t>(edgeIndex)];
    if (edge == nullptr || edge->getSource() == nullptr ||
        edge->getTarget() == nullptr)
      continue;
    const std::uint64_t source = edge->getSource()->getId();
    const std::uint64_t target = edge->getTarget()->getId();
    const Eigen::MatrixXcd derivative = workspace.gradient(source, target);
    const Eigen::MatrixXcd derivativeVelocity =
        workspace.gradientDirectionalDerivative(source, target, directionData);

    if (phaseMode == EntropyPhaseMode::IncludeComplexPhase) {
      // d/dt [2 Tr(C L^dagger dL/dz_e)] by the product rule.
      velocityOfGradient[static_cast<std::size_t>(edgeIndex)] =
          2.0 * ((fullPhaseLeftVelocity.array() *
                  derivative.transpose().array())
                     .sum() +
                 (fullPhaseLeft.array() *
                  derivativeVelocity.transpose().array())
                     .sum());
    } else {
      cd component{0.0, 0.0};
      for (Eigen::Index row = 0; row < n; ++row)
        for (Eigen::Index column = 0; column < n; ++column) {
          const cd unit = phaseFactor(row, column);
          if (unit == cd{0.0, 0.0}) continue;
          component += magnitudeDerivativeVelocity(row, column) * unit *
                           derivative(row, column) +
                       magnitudeDerivative(row, column) *
                           phaseFactorVelocity(row, column) *
                           derivative(row, column) +
                       magnitudeDerivative(row, column) * unit *
                           derivativeVelocity(row, column);
        }
      velocityOfGradient[static_cast<std::size_t>(edgeIndex)] = component;
    }
  }
  return velocityOfGradient;
}

double HodgeLaplacian::spectralEntropyGradientNorm(
    int k, EntropyPhaseMode phaseMode) const {
  double normSquared = 0.0;
  for (const cd component : spectralEntropyGradient(k, phaseMode))
    normSquared += std::norm(component);
  return normSquared;
}

const HodgeLaplacian::SpectrumCache &HodgeLaplacian::ensureSpectrum(
    int k, bool metric) const {
  // Key is (k, metric, weight convention); the map is shared across instances
  // whose conventions may differ.
  const long long key =
      ((static_cast<long long>(k) * 2 + (metric ? 1 : 0)) * 2 +
       (weightConvention_ == WeightConvention::SquaredContent ? 1 : 0)) * 2 +
      (metricSource_ == MetricSource::WhitneyPencil ? 1 : 0);
  auto &spectrumCache_ = sharedSpectra_->map;
  const auto cached = spectrumCache_.find(key);
  if (cached != spectrumCache_.end()) return cached->second;

  SpectrumCache sp;
  if (st_) {
    const Eigen::MatrixXcd L = operatorMatrix(k, metric);
    const int nk = static_cast<int>(L.rows());
    sp.dim = nk;
    sp.evals.assign(static_cast<std::size_t>(nk), cd(0.0, 0.0));
    sp.evecs.assign(static_cast<std::size_t>(nk) * nk, cd(0.0, 0.0));
    sp.wk = weights(k);
    if (nk > 0) {
      // Indefinite metric ⇒ non-self-adjoint, so a general solver is needed;
      // eigenvalues may be negative or complex-conjugate pairs.
      Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(L);
      const Eigen::VectorXcd lam = es.eigenvalues();
      const Eigen::MatrixXcd V = es.eigenvectors();

      // Stable, reproducible order: ascending by (Re, Im) of the eigenvalue.
      std::vector<int> order(static_cast<std::size_t>(nk));
      for (int i = 0; i < nk; ++i) order[static_cast<std::size_t>(i)] = i;
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (lam[a].real() != lam[b].real()) return lam[a].real() < lam[b].real();
        return lam[a].imag() < lam[b].imag();
      });
      for (int c = 0; c < nk; ++c) {
        const int src = order[static_cast<std::size_t>(c)];
        sp.evals[static_cast<std::size_t>(c)] = lam[src];
        for (int r = 0; r < nk; ++r)
          sp.evecs[static_cast<std::size_t>(r) * nk + c] = V(r, src);
      }
    }
  }
  return spectrumCache_.emplace(key, std::move(sp)).first->second;
}

void HodgeLaplacian::ensureDecomposition() const {
  // The U(1) connection Laplacian's Hermitian eigendecomposition (not L_0).
  if (decomposed_) return;
  const int N = static_cast<int>(order_);
  evals_.assign(static_cast<std::size_t>(N), 0.0);
  evecs_.assign(static_cast<std::size_t>(N) * N, cd(0.0, 0.0));
  if (N == 0) {
    decomposed_ = true;
    return;
  }

  std::vector<cd> A;
  std::vector<double> D;
  assemble(A, D);
  Eigen::MatrixXcd L(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      L(i, j) = -A[static_cast<std::size_t>(i) * N + j];
  for (int i = 0; i < N; ++i) L(i, i) += D[static_cast<std::size_t>(i)];

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(L);
  // Eigen yields real eigenvalues in ascending order and orthonormal columns.
  const Eigen::VectorXd &lam = es.eigenvalues();
  const Eigen::MatrixXcd &V = es.eigenvectors();
  for (int i = 0; i < N; ++i) evals_[static_cast<std::size_t>(i)] = lam[i];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      evecs_[static_cast<std::size_t>(i) * N + j] = V(i, j);
  decomposed_ = true;
}

bool HodgeLaplacian::isHermitian(double tol) const {
  const int N = static_cast<int>(order_);
  if (N == 0) return true;
  const std::vector<cd> Lflat = connectionLaplacian();
  Eigen::MatrixXcd L(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      L(i, j) = Lflat[static_cast<std::size_t>(i) * N + j];
  return (L - L.adjoint()).norm() <= tol;
}

double HodgeLaplacian::unitarityResidual(double t) const {
  ensureDecomposition();
  const int N = static_cast<int>(order_);
  if (N == 0) return 0.0;

  Eigen::MatrixXcd V(N, N);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      V(i, j) = evecs_[static_cast<std::size_t>(i) * N + j];

  Eigen::VectorXcd phases(N);
  for (int k = 0; k < N; ++k)
    phases[k] = std::exp(cd(0.0, -evals_[static_cast<std::size_t>(k)] * t));

  // U = e^{-iLt} = V diag(e^{-i lambda t}) V^dagger.
  const Eigen::MatrixXcd U = V * phases.asDiagonal() * V.adjoint();
  const Eigen::MatrixXcd resid =
      U * U.adjoint() - Eigen::MatrixXcd::Identity(N, N);
  return resid.norm();
}

Spectrum HodgeLaplacian::spectrum(int k, bool metric) const {
  requireNonNegativeDegree(k);
  const SpectrumCache &sp = ensureSpectrum(k, metric);
  // L_k is complex and generally non-self-adjoint, so the spectrum is not
  // flagged Hermitian. Components are indexed over the canonical ChainComplex
  // k-cell order.
  return makeSpectrum(k, cochainOrdering(k, /*useVertexSet=*/false), sp.evals,
                      sp.evecs, sp.dim, /*hermitian=*/false);
}

std::vector<std::complex<double>> HodgeLaplacian::eigenvalues(int k, bool metric) const {
  requireNonNegativeDegree(k);
  return ensureSpectrum(k, metric).evals;
}

std::vector<cd> HodgeLaplacian::eigenvectors(int k, bool metric) const {
  requireNonNegativeDegree(k);
  return ensureSpectrum(k, metric).evecs;
}

Spectrum HodgeLaplacian::connectionSpectrum() const {
  ensureDecomposition();
  std::vector<cd> evalsC(evals_.size());
  for (std::size_t i = 0; i < evals_.size(); ++i) evalsC[i] = cd(evals_[i], 0.0);
  return makeSpectrum(0, cochainOrdering(0, /*useVertexSet=*/true), evalsC,
                      evecs_, static_cast<int>(order_), /*hermitian=*/true);
}

std::vector<cd> HodgeLaplacian::connectionEigenvalues() const {
  // Hermitian, so the eigenvalues are real and ascending; widened to complex
  // for type parity with the L_k family.
  ensureDecomposition();
  return std::vector<cd>(evals_.begin(), evals_.end());
}

std::vector<cd> HodgeLaplacian::connectionEigenvectors() const {
  ensureDecomposition();
  return evecs_;
}

std::vector<Cochain> HodgeLaplacian::connectionHarmonics(double tol) const {
  return connectionSpectrum().harmonics(tol);
}

std::vector<cd> HodgeLaplacian::connectionHarmonicMatrix(double tol) const {
  ensureDecomposition();
  const int dim = static_cast<int>(order_);
  std::vector<cd> rows;
  for (int j = 0; j < dim; ++j) {
    if (std::abs(evals_[static_cast<std::size_t>(j)]) >= tol) continue;
    for (int i = 0; i < dim; ++i)
      rows.push_back(evecs_[static_cast<std::size_t>(i) * dim + j]);
  }
  return rows;
}

std::vector<Cochain> HodgeLaplacian::harmonics(int k, double tol,
                                               bool metric) const {
  // ker L_k as Cochains: the eigenvectors with (near-)zero eigenvalue, a basis
  // for H_k. requireNonNegativeDegree runs inside spectrum().
  return spectrum(k, metric).harmonics(tol);
}

std::vector<cd> HodgeLaplacian::harmonicMatrix(int k, double tol,
                                               bool metric) const {
  requireNonNegativeDegree(k);
  // The cached eigendecompositions harmonics() reads, emitted column by
  // selected column so no Cochain objects are materialized.
  const SpectrumCache &sp = ensureSpectrum(k, metric);
  const std::vector<cd> *evals = &sp.evals;
  const std::vector<cd> *evecs = &sp.evecs;
  const int dim = sp.dim;
  std::vector<cd> rows;
  for (int j = 0; j < dim; ++j) {
    if (std::abs((*evals)[static_cast<std::size_t>(j)]) >= tol) continue;
    for (int i = 0; i < dim; ++i)
      rows.push_back((*evecs)[static_cast<std::size_t>(i) * dim + j]);
  }
  return rows;
}

std::vector<std::complex<double>> HodgeLaplacian::nullNorms(int k, double tol,
                                                        bool metric) const {
  requireNonNegativeDegree(k);
  const SpectrumCache &sp = ensureSpectrum(k, metric);
  const std::size_t N = static_cast<std::size_t>(sp.dim);
  if (N == 0) return {};

  std::vector<cd> norms;
  for (std::size_t j = 0; j < N; ++j) {
    if (std::abs(sp.evals[j]) >= tol) continue;
    // Indefinite W-norm <h,h>_W = sum_i W_{k,i} |h_i|^2 with signed W_k. W_k is
    // complex once a Lorentzian cell's signed content is imaginary, so the norm
    // is returned complex: its sign says whether the direction is spacelike- or
    // timelike-dominated, and ~0 marks a null (lightlike) harmonic.
    cd nrm{0.0, 0.0};
    for (std::size_t i = 0; i < N; ++i) {
      const cd hi = sp.evecs[i * N + j];
      nrm += sp.wk[i] * std::norm(hi);  // std::norm = |hi|^2
    }
    norms.push_back(nrm);
  }
  return norms;
}

}  // namespace tessera::cobordism
