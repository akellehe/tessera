// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_KUENNETHPRODUCT_H
#define TESSERA_COBORDISM_KUENNETHPRODUCT_H

#include <complex>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "cobordism/Certificate.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # KuennethProduct
///
/// The exact Kronecker-sum / Kuenneth rule for product complexes:
///
/// \f[ L_{A\times B} \;=\; L_A \otimes I \;+\; I \otimes L_B . \f]
///
/// **Identity and domain.** The Kronecker sum is exact for any two square
/// operators, and since \f$ L_A\otimes I \f$ and \f$ I\otimes L_B \f$ commute
/// its spectrum is the pairwise sums \f$ \{\lambda_i+\mu_j\} \f$. The
/// degree-zero operator here is the U(1) connection graph Laplacian
/// `HodgeLaplacian::connectionLaplacian`, not the Hodge \f$ L_0 \f$. As a
/// statement about a complex it needs a product cell structure with product
/// weights: at degree zero, a weighted 1-skeleton that is the Cartesian product
/// of the factors' (product vertices \f$ (u,v) \f$; edges
/// \f$ (u,v)\!-\!(u',v) \f$ with \f$ A \f$'s weight, \f$ (u,v)\!-\!(u,v') \f$
/// with \f$ B \f$'s). A staircase-subdivided `SimplicialProduct` falls outside
/// that domain, and `productCertificate` reports `holds() == false` for it.
/// Weights may be complex or signed; positive definiteness is not assumed.
///
/// This class works at the spectrum/matrix level; the derived many-body spectra
/// are `OccupationSpectra`'s job.
class KuennethProduct {
  public:
    /// The Kronecker sum \f$ L_A\otimes I_{n_B} + I_{n_A}\otimes L_B \f$ as a
    /// flat row-major \f$ (n_An_B)\times(n_An_B) \f$ matrix. Product index
    /// \f$ (i_A, i_B) \mapsto i_A\,n_B + i_B \f$. The assembly is exact
    /// (additions only).
    /// @throws std::invalid_argument on dimension mismatch.
    [[nodiscard]] static std::vector<std::complex<double>> kroneckerSum(
        const std::vector<std::complex<double>> &laplacianA, int dimA,
        const std::vector<std::complex<double>> &laplacianB, int dimB);

    /// The exact spectrum of the Kronecker sum from the factor spectra: all
    /// pairwise sums \f$ \lambda_i + \mu_j \f$, sorted ascending by
    /// \f$ (\mathrm{Re}, \mathrm{Im}) \f$ (the `Spectrum` convention).
    /// Costs \f$ O(n_An_B\log(n_An_B)) \f$; the product operator is never
    /// diagonalized.
    [[nodiscard]] static std::vector<std::complex<double>> pairwiseSpectrum(
        const std::vector<std::complex<double>> &spectrumA,
        const std::vector<std::complex<double>> &spectrumB);

    /// Certify that `product` is a product complex of the two factors at degree
    /// zero: its U(1) connection graph Laplacian
    /// (`HodgeLaplacian::connectionLaplacian`, \f$ D - A \f$ over the sorted
    /// vertex order) equals the Kronecker sum of the factors' under the declared
    /// vertex `pairing`, entrywise, to relative `tolerance`. This is not a
    /// statement about the Hodge \f$ L_0 \f$.
    ///
    /// `pairing` lists (product vertex id, factor-A vertex id, factor-B vertex
    /// id): the product structure is supplied by the caller, not discovered.
    /// Vertices are matched by identifier, not by an imposed sort.
    ///
    /// The certificate carries the relative residual
    /// \f$ \max_{ij}|L_{\text{prod}} - (L_A\otimes I + I\otimes L_B)|_{ij} \f$
    /// divided by the largest entry magnitude in either matrix. `holds()` is
    /// false for a complex outside the domain.
    ///
    /// @throws std::invalid_argument when the pairing is malformed: wrong
    ///   size, duplicate or unknown identifiers, or a missing factor pair.
    [[nodiscard]] static Certificate productCertificate(
        const std::shared_ptr<Spacetime> &product,
        const std::shared_ptr<Spacetime> &factorA,
        const std::shared_ptr<Spacetime> &factorB,
        const std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>>
            &pairing,
        double tolerance = 1e-12);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_KUENNETHPRODUCT_H
