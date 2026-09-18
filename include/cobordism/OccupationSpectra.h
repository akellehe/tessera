// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_OCCUPATIONSPECTRA_H
#define TESSERA_COBORDISM_OCCUPATIONSPECTRA_H

#include <complex>
#include <cstddef>
#include <vector>

namespace tessera::cobordism {

/// # OccupationSpectra
///
/// Fermionic second quantization at the spectrum/matrix level: the free
/// many-body spectra and one-particle block assemblies that follow from a
/// one-particle operator, with no Fock-space operator materialized. Everything
/// here follows from the exact identities
///
/// \f[ \mathrm{spec}\, d\Gamma(h)\big|_{\Lambda^N} =
///     \Big\{ \textstyle\sum_{i\in S}\lambda_i : S\subseteq\mathrm{spec}\,h,
///     |S| = N \Big\}, \qquad
///     d\Gamma(h_A\oplus h_B) = d\Gamma(h_A)\hat\otimes I +
///     I\hat\otimes d\Gamma(h_B), \qquad
///     F_-(h_A\oplus h_B)\cong F_-(h_A)\hat\otimes F_-(h_B), \f]
///
/// read at the spectrum level: occupation subset sums, and the direct-sum
/// spectrum identity that the subset sums of \f$ h_A\oplus h_B \f$ are the
/// merged pairwise sums over particle splits \f$ N = N_A + N_B \f$.
///
/// Exact for any square one-particle operator: eigenvalues may be complex and
/// nothing assumes a Hermitian or positive-definite operator. Returned spectra
/// are sorted ascending by \f$ (\mathrm{Re}, \mathrm{Im}) \f$, the `Spectrum`
/// convention, so equal multisets compare elementwise. An enumeration whose
/// output would exceed `maxTerms` throws.
class OccupationSpectra {
  public:
    /// Refusal threshold for enumerated outputs (number of terms).
    static constexpr std::size_t kDefaultMaxTerms = std::size_t{1} << 22;

    /// The \f$ \binom{n}{N} \f$ fermionic occupation subset sums of a
    /// one-particle spectrum: the exact free \f$ N \f$-particle spectrum of
    /// \f$ d\Gamma(h) \f$ on \f$ \Lambda^N \mathfrak h \f$. `particles = 0`
    /// yields the vacuum \f$ \{0\} \f$; `particles > n` yields empty (Pauli).
    /// @throws std::invalid_argument for negative `particles`;
    ///   std::length_error when the output would exceed `maxTerms`.
    [[nodiscard]] static std::vector<std::complex<double>> subsetSums(
        const std::vector<std::complex<double>> &oneParticle, int particles,
        std::size_t maxTerms = kDefaultMaxTerms);

    /// All \f$ 2^n \f$ subset sums: the free fermionic Fock spectrum of
    /// \f$ d\Gamma(h) \f$ across every particle number.
    /// @throws std::length_error when \f$ 2^n \f$ would exceed `maxTerms`.
    [[nodiscard]] static std::vector<std::complex<double>> fockSums(
        const std::vector<std::complex<double>> &oneParticle,
        std::size_t maxTerms = kDefaultMaxTerms);

    /// The free \f$ N \f$-particle spectrum of \f$ h_A \oplus h_B \f$ computed
    /// from the factors — the \f$ F_-(h_A)\hat\otimes F_-(h_B) \f$ identity at
    /// the spectrum level: merge over \f$ N_A + N_B = N \f$ of the pairwise sums
    /// of the factors' subset sums. Equals `subsetSums(concat(A, B), N)`.
    /// @throws as `subsetSums`.
    [[nodiscard]] static std::vector<std::complex<double>> directSumSubsetSums(
        const std::vector<std::complex<double>> &factorA,
        const std::vector<std::complex<double>> &factorB, int particles,
        std::size_t maxTerms = kDefaultMaxTerms);

    /// The one-particle direct sum \f$ h_A \oplus h_B \f$ as a flat
    /// row-major \f$ (n_A+n_B)^2 \f$ matrix (zero coupling blocks).
    ///
    /// The assembly is `SpacetimeComposition::directSum`; this is the same
    /// matrix under the name the Fock-level callers reach for. Use
    /// `hoppingBlock` when the blocks are coupled.
    /// @throws std::invalid_argument on dimension mismatch.
    [[nodiscard]] static std::vector<std::complex<double>> directSum(
        const std::vector<std::complex<double>> &blockA, int dimA,
        const std::vector<std::complex<double>> &blockB, int dimB);

    /// The one-particle hopping-block assembly
    /// \f$ h = \begin{pmatrix} h_A & C \\ C' & h_B \end{pmatrix} \f$, where
    /// `coupling` is the \f$ n_A\times n_B \f$ block \f$ C \f$ and
    /// `couplingReverse` the \f$ n_B\times n_A \f$ block \f$ C' \f$. An empty
    /// `couplingReverse` selects \f$ C' = C^\dagger \f$, the Hermitian hopping
    /// term \f$ \sum c_i^\dagger C_{ij} c_j + \text{h.c.} \f$ at the
    /// one-particle level; in the non-normal regime the reverse block is
    /// independent data and must be passed explicitly.
    /// @throws std::invalid_argument on dimension mismatch.
    [[nodiscard]] static std::vector<std::complex<double>> hoppingBlock(
        const std::vector<std::complex<double>> &blockA, int dimA,
        const std::vector<std::complex<double>> &blockB, int dimB,
        const std::vector<std::complex<double>> &coupling,
        const std::vector<std::complex<double>> &couplingReverse = {});
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_OCCUPATIONSPECTRA_H
