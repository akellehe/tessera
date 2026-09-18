// Choi–Jamiołkowski map–state duality ("bending") over dense complex matrices.
//
// The Choi–Jamiołkowski isomorphism bends an operator into a state: a linear
// map / matrix U on H_A → H_B becomes a vector |vec(U)⟩ in H_A ⊗ H_B. This
// class is the dense-linear-algebra realisation of that bend — Eigen only,
// no ITensor, no matrix product states. It serves as the reference oracle
// for the tensor-network path: the bent operation is checked against its
// transition amplitude.
//
// ─── Conventions ──────────────────────────────────────────────────
//
// Matrices are passed flat, ROW-MAJOR: a dA×dB matrix U has
//   U_{ij} = U[i*dB + j],   i ∈ [0, dA),  j ∈ [0, dB).
//
// Vectorisation (row-major flatten = tensor index i*dB + j):
//
//   vec(U) = Σ_{ij} U_{ij} |i⟩_A ⊗ |j⟩_B.
//
// On a rank-one outer product this gives the separable state
//
//   vec(|a⟩⟨b|) = a ⊗ conj(b),
//
// which has Schmidt rank 1. More generally the Schmidt rank of vec(U) equals
// the number of nonzero singular values of U (the SVD of U is the Schmidt
// decomposition of vec(U)).
//
// The transition operator of two states is the rank-one map
//
//   U_T = |psiA⟩⟨psiB|        ((U_T)_{ij} = psiA_i · conj(psiB_j)),
//
// and the central map–state (Hilbert–Schmidt) duality identity is
//
//   ⟨psiA|U|psiB⟩ = ⟨vec(U_T)|vec(U)⟩ = Tr(U_T^H · U)
//                 = Σ_{ij} conj(psiA_i) · U_{ij} · psiB_j.
//
// ─── References ─────────────────────────────────────────────────────────────
//   Choi, *Completely positive linear maps on complex matrices*,
//     Linear Algebra Appl. 10, 285 (1975).
//   Jamiołkowski, *Linear transformations which preserve trace and positive
//     semidefiniteness of operators*, Rep. Math. Phys. 3, 275 (1972).

#pragma once

#include <complex>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;

/// Dense Choi–Jamiołkowski map–state duality ("bending").
///
/// Stateless and static-only: not instantiable, every operation is a static
/// method. All matrices and vectors are flat, row-major
/// `std::vector<std::complex<double>>` (see the file header for the
/// conventions); Eigen is used internally for the SVD.
class ChoiJamiolkowski {
  public:
    ChoiJamiolkowski() = delete;

    /// Vectorise a dA×dB operator: vec(U) = Σ_{ij} U_{ij} |i⟩_A ⊗ |j⟩_B.
    /// With the row-major convention the tensor index is i*dB + j, so vec(U) is
    /// exactly the row-major flatten of U (length dA·dB). Throws
    /// std::invalid_argument if U.size() != dA·dB or a dimension is non-positive.
    [[nodiscard]] static std::vector<std::complex<double>> vectorize(
        const std::vector<std::complex<double>> &U, int dA, int dB);

    /// Un-vectorise — the inverse of `vectorize`: reshape a length-(dA·dB) state
    /// |v⟩ = Σ_{ij} v_{ij} |i⟩_A ⊗ |j⟩_B into the dA×dB operator U with
    /// U_{ij} = v_{i·dB + j}. With the row-major convention the returned buffer
    /// equals the input, now read as a matrix, so
    /// `unvectorize(vectorize(U), dA, dB) == U`. Throws std::invalid_argument
    /// if v.size() != dA·dB or a dimension is non-positive.
    [[nodiscard]] static std::vector<std::complex<double>> unvectorize(
        const std::vector<std::complex<double>> &v, int dA, int dB);

    /// Singular values of the dA×dB operator U (Eigen JacobiSVD), in descending
    /// order; min(dA, dB) of them. These are the Schmidt coefficients of vec(U).
    [[nodiscard]] static std::vector<double> singularValues(
        const std::vector<std::complex<double>> &U, int dA, int dB);

    /// Schmidt rank of vec(U): the number of singular values of U that exceed
    /// tol·σ_max (σ_max the largest singular value). Equals the operator rank of
    /// U and the bipartite Schmidt rank of the state vec(U).
    [[nodiscard]] static int schmidtRank(
        const std::vector<std::complex<double>> &U, int dA, int dB,
        double tol = 1e-10);

    /// Transition operator U_T = |psiA⟩⟨psiB| (a rank-one dA×dB matrix,
    /// (U_T)_{ij} = psiA_i · conj(psiB_j)), returned flat row-major. Throws
    /// std::invalid_argument if psiA.size() != dA or psiB.size() != dB.
    [[nodiscard]] static std::vector<std::complex<double>> transitionOperator(
        const std::vector<std::complex<double>> &psiA,
        const std::vector<std::complex<double>> &psiB, int dA, int dB);

    /// Transition amplitude ⟨psiA|U|psiB⟩ = Σ_{ij} conj(psiA_i)·U_{ij}·psiB_j.
    /// By the duality identity this equals ⟨vec(U_T)|vec(U)⟩ = Tr(U_T^H·U) for
    /// U_T = |psiA⟩⟨psiB|. Throws std::invalid_argument on a dimension mismatch.
    [[nodiscard]] static std::complex<double> transitionAmplitude(
        const std::vector<std::complex<double>> &psiA,
        const std::vector<std::complex<double>> &U,
        const std::vector<std::complex<double>> &psiB, int dA, int dB);

    /// Choi–Jamiołkowski *state* of a square d×d operator U:
    /// |Φ_U⟩ = (U ⊗ I)|Φ⁺⟩ with |Φ⁺⟩ = (1/√d) Σ_k |k⟩|k⟩, which in the
    /// row-major convention is (1/√d)·vec(U) (length d·d). For unitary U this
    /// is a unit vector. Throws std::invalid_argument if U.size() != d·d or d
    /// is non-positive.
    [[nodiscard]] static std::vector<std::complex<double>> choiState(
        const std::vector<std::complex<double>> &U, int d);

    /// Recover the operator U from its Choi–Jamiołkowski *state* — the inverse of
    /// `choiState`. Since |Φ_U⟩ = (1/√d)·vec(U), U = √d · unvectorise(state),
    /// returned flat row-major (length d·d). For a unit Choi state of a unitary
    /// this is that unitary up to the global phase the state carries (so compare
    /// up to phase). Throws std::invalid_argument if state.size() != d·d or d is
    /// non-positive.
    [[nodiscard]] static std::vector<std::complex<double>> operatorFromChoiState(
        const std::vector<std::complex<double>> &state, int d);

    /// Choi *matrix* J(U) = |Φ_U⟩⟨Φ_U| of a square d×d operator U: the
    /// (d·d)×(d·d) density matrix of choiState(U, d), returned flat row-major
    /// (J[i·d² + j] = state_i · conj(state_j)). This is the standard
    /// (U ⊗ I)|Φ⁺⟩⟨Φ⁺|(U ⊗ I)^H Choi matrix; Tr J = 1 for unitary U. Throws
    /// std::invalid_argument as choiState.
    [[nodiscard]] static std::vector<std::complex<double>> choiMatrix(
        const std::vector<std::complex<double>> &U, int d);
};

}  // namespace tessera::quantum
