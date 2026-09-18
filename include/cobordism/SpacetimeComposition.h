// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_SPACETIMECOMPOSITION_H
#define TESSERA_COBORDISM_SPACETIMECOMPOSITION_H

#include <complex>
#include <memory>
#include <vector>

#include "cobordism/HodgeLaplacian.h"

namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {

using ::tessera::spacetime::Spacetime;

/// # SpacetimeComposition
///
/// Combining two complexes' operators: the Kronecker sum, the Kronecker
/// product, and the direct sum.
///
/// Each appears twice, at two levels. The space-level entry points take two
/// complexes and a degree, assemble both Hodge Laplacians, and combine them.
/// The matrix-level ones take flat matrices, for a caller that already has the
/// operators or wants to combine something other than a Laplacian.
///
/// ## Which composition corresponds to which product of complexes
///
/// The space-level names are the standard graph-product names, and the
/// correspondence to the matrix operation is exact:
///
/// | product of complexes | operator |
/// | --- | --- |
/// | Cartesian, \f$ A \,\square\, B \f$ | Kronecker sum \f$ L_A \otimes I + I \otimes L_B \f$ |
/// | tensor, \f$ A \times B \f$ | Kronecker product \f$ L_A \otimes L_B \f$ |
/// | disjoint union, \f$ A \sqcup B \f$ | direct sum \f$ L_A \oplus L_B \f$ |
///
/// The Cartesian case is the one with a spectral shortcut: the two terms
/// commute, so the spectrum is the pairwise sums \f$ \{\lambda_i + \mu_j\} \f$
/// and the product operator never has to be diagonalized. That is
/// `KuennethProduct::pairwiseSpectrum`, and certifying that a given complex
/// really is the Cartesian product of two factors is
/// `KuennethProduct::productCertificate`. Those stay there because they are
/// statements of the Kuenneth theorem; the assembly below is linear algebra and
/// makes no claim about homology.
///
/// ## Index conventions
///
/// A product index \f$ (i_A, i_B) \f$ maps to \f$ i_A n_B + i_B \f$, so factor
/// A is the most significant. The direct sum is block diagonal with A first, so
/// its dimension is \f$ n_A + n_B \f$ where the products' is
/// \f$ n_A n_B \f$. Every assembly is exact: additions and multiplications of
/// the inputs, no solve and no decomposition.
///
/// Weights may be complex or signed; positive definiteness is not assumed, and
/// nothing here requires either factor to be self-adjoint.
class SpacetimeComposition {
  public:
    // ---- space level: two complexes and a degree --------------------------

    /// \f$ L_k \f$ of the Cartesian product \f$ A \,\square\, B \f$, as a flat
    /// row-major \f$ (n_An_B) \times (n_An_B) \f$ matrix.
    ///
    /// Assembles \f$ L_k \f$ of each factor and returns their Kronecker sum.
    /// @throws std::invalid_argument for a negative degree, or a null complex.
    [[nodiscard]] static std::vector<std::complex<double>> cartesianProduct(
        const std::shared_ptr<Spacetime> &factorA,
        const std::shared_ptr<Spacetime> &factorB, int degree = 0,
        bool metric = true);

    /// \f$ L_k \f$ of the tensor product \f$ A \times B \f$: the Kronecker
    /// product of the factors' operators, flat row-major.
    /// @throws std::invalid_argument for a negative degree, or a null complex.
    [[nodiscard]] static std::vector<std::complex<double>> tensorProduct(
        const std::shared_ptr<Spacetime> &factorA,
        const std::shared_ptr<Spacetime> &factorB, int degree = 0,
        bool metric = true);

    /// \f$ L_k \f$ of the disjoint union \f$ A \sqcup B \f$: the block-diagonal
    /// direct sum, flat row-major and \f$ (n_A + n_B) \f$ square.
    /// @throws std::invalid_argument for a negative degree, or a null complex.
    [[nodiscard]] static std::vector<std::complex<double>> directSum(
        const std::shared_ptr<Spacetime> &factorA,
        const std::shared_ptr<Spacetime> &factorB, int degree = 0,
        bool metric = true);

    // ---- matrix level: flat operators -------------------------------------

    /// \f$ A \otimes I_{n_B} + I_{n_A} \otimes B \f$, flat row-major.
    /// @throws std::invalid_argument when a matrix is not its stated square.
    [[nodiscard]] static std::vector<std::complex<double>> kroneckerSum(
        const std::vector<std::complex<double>> &operatorA, int dimA,
        const std::vector<std::complex<double>> &operatorB, int dimB);

    /// \f$ A \otimes B \f$, flat row-major:
    /// \f$ (A \otimes B)_{(i_A i_B),(j_A j_B)} = A_{i_Aj_A} B_{i_Bj_B} \f$.
    /// @throws std::invalid_argument when a matrix is not its stated square.
    [[nodiscard]] static std::vector<std::complex<double>> kroneckerProduct(
        const std::vector<std::complex<double>> &operatorA, int dimA,
        const std::vector<std::complex<double>> &operatorB, int dimB);

    /// \f$ A \oplus B \f$: block diagonal, A first, zero off the blocks.
    /// @throws std::invalid_argument when a matrix is not its stated square.
    [[nodiscard]] static std::vector<std::complex<double>> directSum(
        const std::vector<std::complex<double>> &operatorA, int dimA,
        const std::vector<std::complex<double>> &operatorB, int dimB);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SPACETIMECOMPOSITION_H
