// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/KuennethProduct.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "cobordism/HodgeLaplacian.h"
#include "cobordism/SpacetimeComposition.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

namespace {

using cd = std::complex<double>;

// The sorted-ascending vertex-id order HodgeLaplacian indexes its k=0
// operator over, as id -> row/column index.
std::unordered_map<std::uint64_t, int> sortedIdIndex(const Spacetime &st) {
  std::vector<std::uint64_t> ids;
  for (const auto &vertex : st.getVertexList()->toVector())
    if (vertex != nullptr)
      ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  std::unordered_map<std::uint64_t, int> index;
  index.reserve(ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    index[ids[i]] = static_cast<int>(i);
  return index;
}

void requireSquare(const std::vector<cd> &matrix, int dim, const char *name) {
  if (dim < 0 || matrix.size() != static_cast<std::size_t>(dim) *
                                      static_cast<std::size_t>(dim))
    throw std::invalid_argument(std::string(name) +
                                ": flat size does not match dimension");
}

} // namespace

std::vector<cd> KuennethProduct::pairwiseSpectrum(
    const std::vector<cd> &spectrumA, const std::vector<cd> &spectrumB) {
  std::vector<cd> sums;
  sums.reserve(spectrumA.size() * spectrumB.size());
  for (const cd &a : spectrumA)
    for (const cd &b : spectrumB)
      sums.push_back(a + b);
  std::sort(sums.begin(), sums.end(), [](const cd &x, const cd &y) {
    if (x.real() != y.real())
      return x.real() < y.real();
    return x.imag() < y.imag();
  });
  return sums;
}

Certificate KuennethProduct::productCertificate(
    const std::shared_ptr<Spacetime> &product,
    const std::shared_ptr<Spacetime> &factorA,
    const std::shared_ptr<Spacetime> &factorB,
    const std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>>
        &pairing,
    double tolerance) {
  if (!product || !factorA || !factorB)
    throw std::invalid_argument("productCertificate: null spacetime");

  const auto productIndex = sortedIdIndex(*product);
  const auto indexA = sortedIdIndex(*factorA);
  const auto indexB = sortedIdIndex(*factorB);
  const int dimA = static_cast<int>(indexA.size());
  const int dimB = static_cast<int>(indexB.size());
  const std::size_t dim =
      static_cast<std::size_t>(dimA) * static_cast<std::size_t>(dimB);
  if (pairing.size() != dim || productIndex.size() != dim)
    throw std::invalid_argument(
        "productCertificate: pairing must list every product vertex exactly "
        "once as |V(A)| * |V(B)| triples");

  // Product row index (sorted product ids) -> Kronecker index iA*dimB + iB.
  std::vector<int> toKronecker(dim, -1);
  std::vector<bool> pairSeen(dim, false);
  for (const auto &[productId, aId, bId] : pairing) {
    const auto p = productIndex.find(productId);
    const auto a = indexA.find(aId);
    const auto b = indexB.find(bId);
    if (p == productIndex.end() || a == indexA.end() || b == indexB.end())
      throw std::invalid_argument(
          "productCertificate: pairing names an unknown vertex identifier");
    const int kron = a->second * dimB + b->second;
    if (toKronecker[static_cast<std::size_t>(p->second)] != -1 ||
        pairSeen[static_cast<std::size_t>(kron)])
      throw std::invalid_argument(
          "productCertificate: duplicate product vertex or factor pair");
    toKronecker[static_cast<std::size_t>(p->second)] = kron;
    pairSeen[static_cast<std::size_t>(kron)] = true;
  }

  // The U(1) connection graph Laplacians D - A, indexed over the full sorted
  // vertex order the pairing is expressed in -- not the Hodge L_0. The rule's
  // domain is a weighted 1-skeleton that is the Cartesian product of the
  // factors', with the factor edges' complex weights and phases.
  const HodgeLaplacian hodgeProduct(product);
  const HodgeLaplacian hodgeA(factorA);
  const HodgeLaplacian hodgeB(factorB);
  const std::vector<cd> laplacianProduct = hodgeProduct.connectionLaplacian();
  const std::vector<cd> sum = SpacetimeComposition::kroneckerSum(
      hodgeA.connectionLaplacian(), dimA, hodgeB.connectionLaplacian(), dimB);

  double scale = 0.0;
  for (const cd &value : laplacianProduct)
    scale = std::max(scale, std::abs(value));
  for (const cd &value : sum)
    scale = std::max(scale, std::abs(value));
  double residual = 0.0;
  for (std::size_t i = 0; i < dim; ++i)
    for (std::size_t j = 0; j < dim; ++j) {
      const std::size_t ki = static_cast<std::size_t>(toKronecker[i]);
      const std::size_t kj = static_cast<std::size_t>(toKronecker[j]);
      residual = std::max(
          residual, std::abs(laplacianProduct[i * dim + j] - sum[ki * dim + kj]));
    }
  if (scale > 0.0)
    residual /= scale;

  // The U(1) connection operator is Hermitian and diagonally dominant with
  // |A_ij|-magnitude degrees, hence positive semidefinite by Gershgorin, for any
  // complex or signed edge weight. The Hodge L_0 carries no such bound.
  return Certificate::algebraicallyExact(CertificateDomain::Static,
                                         CertificateRegime::PositiveSemidefinite,
                                         residual, tolerance);
}

} // namespace tessera::cobordism
