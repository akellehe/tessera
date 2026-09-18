// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_INTEGERLINALG_H
#define TESSERA_COBORDISM_INTEGERLINALG_H

#include <cstdint>
#include <vector>

// Linear algebra for simplicial homology: exact-integer (Smith normal form over
// ℤ) and GF(2) routines, plus Sylvester inertia via Eigen's symmetric
// eigensolver. Matrices are flat row-major and dense.
namespace tessera::cobordism {

/// Smith normal form data for an integer matrix: its rank and the positive
/// invariant factors d_1 | d_2 | ... | d_rank (each > 0, each dividing the next).
struct SmithNormalForm {
  int rank{0};
  std::vector<long> invariantFactors{};
};

/// Smith normal form of an integer matrix M (rows x cols, flat row-major).
/// Returns the rank and invariant factors. M is taken by value (mutated
/// internally).
[[nodiscard]] SmithNormalForm smithNormalForm(std::vector<long> M, int rows, int cols);

/// Rank over Q of an integer matrix (the number of Smith normal form pivots).
[[nodiscard]] int integerRank(const std::vector<long> &M, int rows, int cols);

/// Rank over GF(2) of a 0/1 matrix (flat row-major). Entries are read mod 2.
[[nodiscard]] int gf2Rank(std::vector<int> M, int rows, int cols);

/// Basis of the GF(2) kernel (nullspace) of a 0/1 matrix M (rows x cols, flat
/// row-major; entries read mod 2). Each returned vector has length `cols` and
/// satisfies M·x ≡ 0 (mod 2); there are nullity = cols - gf2Rank(M) of them.
/// Empty when M has full column rank.
///
/// For M = ∂₂ᵀ this is the cocycle space Z¹ of flat ℤ₂ gauge fields; pair with
/// gf2Span to enumerate the connections.
[[nodiscard]] std::vector<std::vector<int>> gf2Nullspace(std::vector<int> M,
                                                         int rows, int cols);

/// Basis of the rational kernel of an integer matrix M (rows x cols, flat
/// row-major), returned as integer vectors by exact Gauss-Jordan over Q. Each
/// has length `cols`, satisfies M·x = 0 exactly over Z and has coprime entries;
/// there are nullity = cols - rank(M) of them. Pivoting is deterministic (first
/// nonzero per column).
/// @throws std::invalid_argument if `M.size() != rows * cols`.
/// @throws std::overflow_error when the exact rational elimination would
///   overflow 64-bit intermediates.
[[nodiscard]] std::vector<std::vector<long>> integerNullspace(
    const std::vector<long> &M, int rows, int cols);

/// All 2^k GF(2) linear combinations of a `basis` of k length-`cols` vectors,
/// each combination a length-`cols` vector (entries read mod 2). The first
/// element is the zero vector; `cols` is explicit so an empty basis still yields
/// one zero vector of the right length. On a gf2Nullspace cocycle basis this
/// enumerates the 2^nullity flat ℤ₂ connections.
/// @throws std::invalid_argument if k > 24, where the result no longer fits in
///   memory.
[[nodiscard]] std::vector<std::vector<int>> gf2Span(
    const std::vector<std::vector<int>> &basis, int cols);

/// Inertia of a symmetric integer matrix Q (n x n): the counts of positive,
/// negative, and zero eigenvalues by Sylvester's law of inertia. The signature
/// is nPos - nNeg.
struct Inertia {
  int nPos{0};
  int nNeg{0};
  int nZero{0};
  [[nodiscard]] int signature() const noexcept { return nPos - nNeg; }
};

/// Compute the inertia of a symmetric integer matrix Q (n x n, flat row-major)
/// via Eigen's self-adjoint eigensolver, counting eigenvalue signs. An
/// asymmetric Q is first replaced by (Q + Qᵀ)/2. Eigenvalues within `tol` of
/// zero count as zero.
[[nodiscard]] Inertia symmetricInertia(std::vector<long> Q, int n,
                                       double tol = 1e-9);

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_INTEGERLINALG_H
