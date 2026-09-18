// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_COCHAIN_H
#define TESSERA_COBORDISM_COCHAIN_H

#include <Eigen/Core>

#include <complex>
#include <cstdint>
#include <vector>

namespace tessera::cobordism {

/// # Cochain
///
/// A \f$ k \f$-cochain value object: an Eigen-backed vector of complex
/// amplitudes (`Eigen::VectorXcd`) with the degree \f$ k \f$ and the
/// \f$ k \f$-simplex ordering it is indexed over. The ordering is the
/// `HodgeLaplacian` / `ChainComplex` column order: component \f$ i \f$ is the
/// amplitude on the \f$ i \f$-th \f$ k \f$-cell, recorded as its sorted
/// vertex-id tuple (`simplices()[i]`); at \f$ k = 0 \f$, a single vertex id.
///
/// The Hermitian inner product follows the numpy convention
/// \f$ \langle a, b\rangle = \sum_i \overline{a_i}\,b_i = \texttt{np.vdot(a, b)} \f$
/// (conjugate-linear in the first argument).
class Cochain {
  public:
    /// The empty 0-cochain (degree 0, no cells).
    Cochain() = default;

    /// A degree-`degree` cochain over `simplices` (sorted vertex-id tuples, in
    /// the indexing order) carrying `coeffs`. @throws std::invalid_argument if
    /// `simplices.size() != coeffs.size()`.
    Cochain(int degree, std::vector<std::vector<std::uint64_t>> simplices,
            Eigen::VectorXcd coeffs);

    /// The cochain degree \f$ k \f$.
    [[nodiscard]] int degree() const noexcept { return degree_; }

    /// The number of \f$ k \f$-cells (= `coeffs().size()` = `simplices().size()`).
    [[nodiscard]] std::size_t size() const noexcept {
      return static_cast<std::size_t>(coeffs_.size());
    }

    /// The amplitude vector; pybind exposes it as a 1-D complex
    /// `numpy.ndarray`.
    [[nodiscard]] const Eigen::VectorXcd &coeffs() const noexcept { return coeffs_; }

    /// The \f$ k \f$-simplex ordering: `simplices()[i]` is the sorted vertex-id
    /// tuple of the cell carrying `coeffs()[i]`.
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &simplices()
        const noexcept {
      return simplices_;
    }

    /// Amplitude on the \f$ i \f$-th \f$ k \f$-cell (by index).
    /// @throws std::out_of_range if `index >= size()`.
    [[nodiscard]] std::complex<double> amplitude(std::size_t index) const;

    /// Amplitude on the \f$ k \f$-cell identified by its sorted vertex-id tuple
    /// (e.g. `{vertexId}` at \f$ k = 0 \f$). @throws std::out_of_range if the
    /// simplex is not in this cochain's ordering.
    [[nodiscard]] std::complex<double> amplitudeFor(
        const std::vector<std::uint64_t> &simplex) const;

    /// The Hermitian inner product \f$ \langle \text{this}, \text{other}\rangle
    /// = \sum_i \overline{\text{this}_i}\,\text{other}_i \f$ (= `np.vdot`).
    /// @throws std::invalid_argument if the degrees or orderings differ.
    [[nodiscard]] std::complex<double> innerProduct(const Cochain &other) const;

    /// The Euclidean norm \f$ \sqrt{\sum_i |c_i|^2} \f$.
    [[nodiscard]] double norm() const;

    /// A copy scaled to unit norm; the cochain itself if its norm is ~0.
    [[nodiscard]] Cochain normalized() const;

  private:
    int degree_{0};
    std::vector<std::vector<std::uint64_t>> simplices_{};
    Eigen::VectorXcd coeffs_{};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_COCHAIN_H
