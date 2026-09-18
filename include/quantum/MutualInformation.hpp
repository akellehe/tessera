// Mutual information on a Schwinger matrix product state (MPS).
//
// Two-site reduced density matrices for arbitrary site pairs (i, j), the
// von Neumann entropy of a small Hermitian matrix, and the resulting
// site-site mutual information I(i:j) = S(ρ_i) + S(ρ_j) - S(ρ_{ij}).
//
// The implementation uses ITensor canonical form: with the orthogonality
// center at site i, sites 1..i-1 are left-canonical and sites j+1..N are
// right-canonical, so the partial trace over everything outside {i, j}
// reduces to the contraction ρ_{ij} = T · dag(T) over the block
// T = A_i ⊗ A_{i+1} ⊗ … ⊗ A_j, with the site indices at i and j primed
// on the bra side to keep them open.
//
// Reference: Hauschild & Pollmann, "Efficient numerical simulations with
// Tensor Networks", SciPost Phys. Lect. Notes 5 (2018).

#pragma once

#include <Eigen/Dense>
#include <itensor/all.h>
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

// Static utility class. Not instantiable.
class MutualInformation {
public:
    MutualInformation() = delete;
    MutualInformation(MutualInformation const&) = delete;
    MutualInformation& operator=(MutualInformation const&) = delete;

    // Single-site reduced density matrix at 1-based site i, returned as
    // a 2x2 Hermitian complex matrix. Bringing the orthogonality center
    // to i and contracting with dag yields ρ_i in one step.
    [[nodiscard]] static Eigen::Matrix2cd
    oneSiteReducedDensity(itensor::MPS const& psi, int i);

    // Two-site reduced density matrix on 1-based sites (i, j) with i < j.
    // Returned as a 4x4 Hermitian complex matrix in the basis order
    // (|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩): row index = 2*s_i + s_j with
    // |↑⟩ ↔ 0, |↓⟩ ↔ 1 (matching ITensor SpinHalf indexing).
    //
    // Throws std::invalid_argument when i or j falls outside [1, N] or
    // when i ≥ j; use oneSiteReducedDensity for i == j.
    [[nodiscard]] static Eigen::Matrix4cd
    twoSiteReducedDensity(itensor::MPS const& psi, int i, int j);

    // Von Neumann entropy of a Hermitian density matrix, in nats.
    // Eigenvalues below ``tol`` (default 1e-12) contribute zero, avoiding
    // the log(0) singularity. The fixed-size overloads give C++ callers
    // an allocation-free path; the dynamic-size one carries the Python
    // binding.
    [[nodiscard]] static double
    vonNeumannEntropy(Eigen::Matrix2cd const& rho, double tol = 1e-12);
    [[nodiscard]] static double
    vonNeumannEntropy(Eigen::Matrix4cd const& rho, double tol = 1e-12);
    [[nodiscard]] static double
    vonNeumannEntropy(Eigen::MatrixXcd const& rho, double tol = 1e-12);

    // Von Neumann / Shannon entropy directly from an already-diagonal
    // spectrum (e.g. a Schmidt spectrum or density-matrix eigenvalues),
    // in nats: -Σ pᵢ log pᵢ over pᵢ > tol. Same convention as the matrix
    // overloads (the spectrum is assumed trace-normalised); avoids
    // re-diagonalising a diagonal input.
    [[nodiscard]] static double
    vonNeumannEntropy(std::vector<double> const& eigenvalues,
                      double tol = 1e-12);

    // Site-site mutual information I({i} : {j}) on the MPS, in nats.
    // = S(ρ_i) + S(ρ_j) - S(ρ_{ij}).
    [[nodiscard]] static double
    siteSite(itensor::MPS const& psi, int i, int j);

    // All-pairs site-site mutual information as a symmetric N×N matrix
    // (zero diagonal). The single-site entropies are computed once and
    // reused; each pair then costs one two-site reduced density matrix,
    // O(N χ³).
    [[nodiscard]] static Eigen::MatrixXd
    allPairs(itensor::MPS const& psi);

    // Edge length ℓ = -log(I) with infinity floor at -log(epsilon).
    // Returns +inf when I < epsilon. Used by EmergentGraph to convert
    // mutual information to a metric weight.
    [[nodiscard]] static double
    edgeLength(double I, double epsilon = 1e-10) noexcept;

    // Elementwise edge length on a matrix of mutual-information values:
    // ℓ_{ij} = -log(I_{ij}), with +inf where I_{ij} < epsilon. Same map
    // as the scalar overload, vectorised so callers (e.g. a B×B bond-MI
    // matrix) need not loop in Python.
    [[nodiscard]] static Eigen::MatrixXd
    edgeLength(Eigen::MatrixXd const& I, double epsilon = 1e-10);

    // ── Dual / bond-cut observables (van Raamsdonk graph) ──────────────
    //
    // Contiguous-interval entropy and bond-cut tripartite information
    // drive the bond-cut spectral-dimension pipeline, which uses chain
    // bipartitions as graph vertices (one per bond, one per snapshot).

    // Bipartite entanglement entropy in nats at bond ``k`` (1-based,
    // between sites k and k+1, 1 ≤ k ≤ N-1), from the Schmidt spectrum
    // of the bipartition [1..k] | [k+1..N].
    [[nodiscard]] static double
    bondEntropy(itensor::MPS const& psi, int k);

    // Entropy in nats of the reduced density matrix on the contiguous
    // interval [i, j] (1-based, inclusive, 1 ≤ i ≤ j ≤ N). A χ⁴
    // transfer-matrix sweep; ρ is never materialised in the (d^L, d^L)
    // basis, so memory stays at O(χ⁴) and runtime at O((j-i) χ⁵).
    [[nodiscard]] static double
    regionEntropy(itensor::MPS const& psi, int i, int j);

    // Tripartite information between bonds n < m (both 1-based,
    // 1 ≤ n < m ≤ N-1):
    //   I(A : C) = S(A) + S(C) - S(B), with
    //     A = [1..n], B = [n+1..m], C = [m+1..N].
    // For a pure state |ψ⟩, S(A ∪ C) = S(B), so this is the mutual
    // information between the two outer regions: how strongly the two
    // cuts are entangled through the middle region. In the van
    // Raamsdonk picture a bond is a minimal surface and the tripartite
    // information measures mutual connectivity.
    [[nodiscard]] static double
    tripartiteInformation(itensor::MPS const& psi, int n, int m);

    // All-pairs bond-cut tripartite information as a symmetric
    // (N-1) × (N-1) matrix with zero diagonal. Each pair costs three
    // ``regionEntropy`` sweeps, O(N χ⁵).
    [[nodiscard]] static Eigen::MatrixXd
    allBondPairs(itensor::MPS const& psi);
};

} // namespace tessera::quantum
