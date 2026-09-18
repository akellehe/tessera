// Dense Choi–Jamiołkowski map–state duality ("bending"). See
// include/quantum/ChoiJamiolkowski.h for the conventions and the
// algebra.

#include "quantum/ChoiJamiolkowski.h"

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <string>
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

namespace {

// A complex matrix laid out the way callers pass them: row-major, so that the
// flat index i*cols + j is the (i, j) entry — i.e. the vec(·) tensor index.
using RowMajorXcd =
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::RowMajor>;

void requirePositiveDims(int dA, int dB, const char *who) {
    if (dA <= 0 || dB <= 0) {
        throw std::invalid_argument(std::string(who) +
                                    ": dimensions dA, dB must be positive");
    }
}

// Validate length and view a flat vector as an Eigen column vector (no copy).
Eigen::Map<const Eigen::VectorXcd> asVector(
    const std::vector<std::complex<double>> &v, int n, const char *who,
    const char *name) {
    if (static_cast<long long>(v.size()) != static_cast<long long>(n)) {
        throw std::invalid_argument(std::string(who) + ": " + name +
                                    " must have length " + std::to_string(n) +
                                    " (got " + std::to_string(v.size()) + ")");
    }
    return Eigen::Map<const Eigen::VectorXcd>(v.data(), n);
}

// Validate length and view a flat row-major buffer as a dA×dB Eigen matrix.
Eigen::Map<const RowMajorXcd> asMatrix(
    const std::vector<std::complex<double>> &m, int dA, int dB,
    const char *who) {
    const long long expected = static_cast<long long>(dA) * dB;
    if (static_cast<long long>(m.size()) != expected) {
        throw std::invalid_argument(
            std::string(who) + ": operator must have dA*dB = " +
            std::to_string(dA) + "*" + std::to_string(dB) + " = " +
            std::to_string(expected) + " entries row-major (got " +
            std::to_string(m.size()) + ")");
    }
    return Eigen::Map<const RowMajorXcd>(m.data(), dA, dB);
}

}  // namespace

std::vector<std::complex<double>> ChoiJamiolkowski::vectorize(
    const std::vector<std::complex<double>> &U, int dA, int dB) {
    requirePositiveDims(dA, dB, "ChoiJamiolkowski::vectorize");
    // vec(U) = Σ_{ij} U_{ij} |i⟩_A ⊗ |j⟩_B. With the row-major layout the
    // tensor index i*dB + j is U's own flat index, so the vectorisation
    // is U's buffer, validated and returned as the length-(dA·dB) state
    // vector.
    asMatrix(U, dA, dB, "ChoiJamiolkowski::vectorize");
    return U;
}

std::vector<std::complex<double>> ChoiJamiolkowski::unvectorize(
    const std::vector<std::complex<double>> &v, int dA, int dB) {
    requirePositiveDims(dA, dB, "ChoiJamiolkowski::unvectorize");
    // The inverse of vectorize: |v⟩ = Σ_{ij} v_{ij} |i⟩_A ⊗ |j⟩_B reshapes
    // to the operator U_{ij} = v_{i·dB + j}. With the row-major
    // convention the matrix buffer is the vector buffer, so the length
    // (dA·dB) is validated and the buffer returned, read as a dA×dB
    // operator.
    asMatrix(v, dA, dB, "ChoiJamiolkowski::unvectorize");
    return v;
}

std::vector<double> ChoiJamiolkowski::singularValues(
    const std::vector<std::complex<double>> &U, int dA, int dB) {
    requirePositiveDims(dA, dB, "ChoiJamiolkowski::singularValues");
    const Eigen::MatrixXcd M =
        asMatrix(U, dA, dB, "ChoiJamiolkowski::singularValues");
    // No U/V requested: JacobiSVD computes the singular values alone, already
    // sorted in descending order.
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M);
    const Eigen::VectorXd &sv = svd.singularValues();
    return std::vector<double>(sv.data(), sv.data() + sv.size());
}

int ChoiJamiolkowski::schmidtRank(const std::vector<std::complex<double>> &U,
                                  int dA, int dB, double tol) {
    const std::vector<double> sv = singularValues(U, dA, dB);
    if (sv.empty()) return 0;
    const double sigmaMax = sv.front();  // descending ⇒ first is the largest
    if (sigmaMax <= 0.0) return 0;       // zero operator ⇒ rank 0
    const double threshold = tol * sigmaMax;
    int rank = 0;
    for (const double s : sv) {
        if (s > threshold) ++rank;
    }
    return rank;
}

std::vector<std::complex<double>> ChoiJamiolkowski::transitionOperator(
    const std::vector<std::complex<double>> &psiA,
    const std::vector<std::complex<double>> &psiB, int dA, int dB) {
    requirePositiveDims(dA, dB, "ChoiJamiolkowski::transitionOperator");
    const auto a =
        asVector(psiA, dA, "ChoiJamiolkowski::transitionOperator", "psiA");
    const auto b =
        asVector(psiB, dB, "ChoiJamiolkowski::transitionOperator", "psiB");
    // U_T = |psiA⟩⟨psiB|, so (U_T)_{ij} = psiA_i · conj(psiB_j); written out
    // row-major into the returned flat buffer.
    std::vector<std::complex<double>> out(static_cast<std::size_t>(dA) *
                                          static_cast<std::size_t>(dB));
    Eigen::Map<RowMajorXcd>(out.data(), dA, dB) = a * b.adjoint();
    return out;
}

std::complex<double> ChoiJamiolkowski::transitionAmplitude(
    const std::vector<std::complex<double>> &psiA,
    const std::vector<std::complex<double>> &U,
    const std::vector<std::complex<double>> &psiB, int dA, int dB) {
    requirePositiveDims(dA, dB, "ChoiJamiolkowski::transitionAmplitude");
    const auto a =
        asVector(psiA, dA, "ChoiJamiolkowski::transitionAmplitude", "psiA");
    const auto b =
        asVector(psiB, dB, "ChoiJamiolkowski::transitionAmplitude", "psiB");
    const auto M = asMatrix(U, dA, dB, "ChoiJamiolkowski::transitionAmplitude");
    // ⟨psiA|U|psiB⟩ = Σ_{ij} conj(psiA_i)·U_{ij}·psiB_j. Eigen's complex dot()
    // conjugates its first argument: a.dot(M*b) = Σ_i conj(a_i) (M b)_i.
    return a.dot(M * b);
}

std::vector<std::complex<double>> ChoiJamiolkowski::choiState(
    const std::vector<std::complex<double>> &U, int d) {
    requirePositiveDims(d, d, "ChoiJamiolkowski::choiState");
    asMatrix(U, d, d, "ChoiJamiolkowski::choiState");  // validates U.size() == d*d
    // |Φ_U⟩ = (U ⊗ I)|Φ⁺⟩ with |Φ⁺⟩ = (1/√d) Σ_k |k⟩|k⟩  ==  (1/√d) · vec(U).
    const double inv = 1.0 / std::sqrt(static_cast<double>(d));
    std::vector<std::complex<double>> state = U;
    for (auto &z : state) z *= inv;
    return state;
}

std::vector<std::complex<double>> ChoiJamiolkowski::operatorFromChoiState(
    const std::vector<std::complex<double>> &state, int d) {
    requirePositiveDims(d, d, "ChoiJamiolkowski::operatorFromChoiState");
    asMatrix(state, d, d, "ChoiJamiolkowski::operatorFromChoiState");
    // choiState gives (1/√d)·vec(U); invert by U = √d · unvectorise(state).
    const double scale = std::sqrt(static_cast<double>(d));
    std::vector<std::complex<double>> U = state;
    for (auto &z : U) z *= scale;
    return U;
}

std::vector<std::complex<double>> ChoiJamiolkowski::choiMatrix(
    const std::vector<std::complex<double>> &U, int d) {
    const std::vector<std::complex<double>> v = choiState(U, d);
    const std::size_t n = v.size();  // d*d
    // J = |v⟩⟨v|, row-major: J[i*n + j] = v_i · conj(v_j).
    std::vector<std::complex<double>> J(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            J[i * n + j] = v[i] * std::conj(v[j]);
    return J;
}

}  // namespace tessera::quantum
