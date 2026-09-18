// Implementation of MutualInformation — see
// include/quantum/MutualInformation.hpp for the canonical-form
// derivation.

#include "quantum/MutualInformation.hpp"
#include "quantum/Schmidt.hpp"

#include <itensor/all.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

// SpinHalf basis convention: ITensor's "Up" index value is 1 and "Dn"
// is 2 (1-based). We pack (s_i, s_j) into a 4×4 matrix row index
// using row = 2*(s_i - 1) + (s_j - 1) so the basis order matches the
// header docstring: (|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩) at rows 0..3.
inline int packTwoSite(int s_i, int s_j) noexcept {
    return 2 * (s_i - 1) + (s_j - 1);
}

// Symmetrise a candidate Hermitian matrix by averaging with its
// adjoint. Numerical noise breaks exact Hermiticity by O(ε); the fix
// is cheap and keeps SelfAdjointEigenSolver happy downstream.
template <typename MatT>
MatT symmetrise(MatT const& m) {
    return 0.5 * (m + m.adjoint());
}

double entropyFromEigenvalues(Eigen::ArrayXd const& eigs, double tol) {
    double S = 0.0;
    for (Eigen::Index k = 0; k < eigs.size(); ++k) {
        const double p = eigs[k];
        if (p > tol) S -= p * std::log(p);
    }
    return S;
}

} // namespace

Eigen::Matrix2cd
MutualInformation::oneSiteReducedDensity(itensor::MPS const& psi_in, int i) {
    using namespace itensor;
    const int N = length(psi_in);
    if (i < 1 || i > N) {
        throw std::invalid_argument(
            "MutualInformation::oneSiteReducedDensity: i out of range [1, N]");
    }

    MPS psi = psi_in;
    psi.position(i);

    // Contract psi(i) with dag(psi(i)) but with the site index primed
    // on the bra side. The orthogonality center handles both bond
    // sums implicitly — left bond is trivially a δ from the canonical
    // chain to its left, and right bond is δ from the right-canonical
    // chain to its right.
    auto site = siteIndex(psi, i);
    auto bra = dag(psi(i));
    bra.prime(site);

    ITensor rho = psi(i) * bra;
    // rho now has indices (site_unprimed, site_primed) — a rank-2
    // tensor that IS the 2×2 reduced density matrix.

    Eigen::Matrix2cd out;
    for (int a = 1; a <= 2; ++a) {
        for (int b = 1; b <= 2; ++b) {
            // a is the unprimed site value (ket); b is the primed
            // value (bra) — so out(a-1, b-1) = ⟨a | ρ | b⟩.
            out(a - 1, b - 1) = eltC(rho, site = a, prime(site) = b);
        }
    }
    return symmetrise(out);
}

Eigen::Matrix4cd
MutualInformation::twoSiteReducedDensity(itensor::MPS const& psi_in,
                                          int i, int j) {
    using namespace itensor;
    const int N = length(psi_in);
    if (i < 1 || j < 1 || i > N || j > N) {
        throw std::invalid_argument(
            "MutualInformation::twoSiteReducedDensity: site index out of [1, N]");
    }
    if (i > j) {
        throw std::invalid_argument(
            "MutualInformation::twoSiteReducedDensity: require i <= j");
    }
    if (i == j) {
        throw std::invalid_argument(
            "MutualInformation::twoSiteReducedDensity: require i < j; "
            "use oneSiteReducedDensity for the i == j case");
    }

    MPS psi = psi_in;
    psi.position(i);

    auto site_i = siteIndex(psi, i);
    auto site_j = siteIndex(psi, j);

    // Transfer-matrix sweep from site i to site j. At every step the
    // intermediate tensor T carries four open indices — (site_i,
    // site_i', current_right_bond_ket, current_right_bond_bra) — so its
    // size is O(χ²) regardless of (j - i). Contracting sites i..j into a
    // single dense tensor first would instead accumulate 2^(j-i+1)
    // physical elements, which is intractable on long doubled chains
    // such as the Choi state's (in_1, out_N) pair.
    //
    // Interior site indices are unprimed on the bra side and auto-trace
    // against the ket. The left bond at site i and the right bond at
    // site j auto-contract via the canonical conditions
    // (psi.position(i) puts sites 1..i-1 in left-canonical and sites
    // i+1..N in right-canonical form), giving identity environments at
    // both boundaries.

    auto primeBondAt = [&](ITensor& tensor, int bond) {
        // bond ∈ [1, N-1]: link index between sites `bond` and bond+1.
        if (bond < 1 || bond > N - 1) return;
        auto link = commonIndex(psi(bond), psi(bond + 1));
        if (link) tensor.prime(link);
    };

    // Site i: prime site_i on the bra (keep it open) and the right
    // bond on the bra (keep ket/bra bonds distinct as we sweep).
    ITensor bra = dag(psi(i));
    bra.prime(site_i);
    primeBondAt(bra, i);
    ITensor T = psi(i);
    T *= bra;

    // Interior sites i+1 .. j-1: prime left+right bonds on the bra so
    // its chain stays distinct; site_k is unprimed → auto-traces.
    for (int k = i + 1; k <= j - 1; ++k) {
        ITensor braK = dag(psi(k));
        primeBondAt(braK, k - 1);
        primeBondAt(braK, k);
        T *= psi(k);
        T *= braK;
    }

    // Site j: prime site_j on the bra and the left bond on the bra
    // (continuing the primed chain from j-1). The right bond is left
    // unprimed so it auto-traces against the right-canonical
    // environment past j.
    ITensor braJ = dag(psi(j));
    braJ.prime(site_j);
    primeBondAt(braJ, j - 1);
    T *= psi(j);
    T *= braJ;

    // T now has only (site_i, site_i', site_j, site_j') open.
    Eigen::Matrix4cd out;
    out.setZero();
    for (int a = 1; a <= 2; ++a) {
        for (int b = 1; b <= 2; ++b) {
            for (int c = 1; c <= 2; ++c) {
                for (int d = 1; d <= 2; ++d) {
                    const auto v = eltC(T,
                                          site_i = a, prime(site_i) = c,
                                          site_j = b, prime(site_j) = d);
                    out(packTwoSite(a, b), packTwoSite(c, d)) = v;
                }
            }
        }
    }
    return symmetrise(out);
}

double
MutualInformation::vonNeumannEntropy(Eigen::Matrix2cd const& rho, double tol) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> es(rho);
    return entropyFromEigenvalues(es.eigenvalues().array(), tol);
}

double
MutualInformation::vonNeumannEntropy(Eigen::Matrix4cd const& rho, double tol) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> es(rho);
    return entropyFromEigenvalues(es.eigenvalues().array(), tol);
}

double
MutualInformation::vonNeumannEntropy(Eigen::MatrixXcd const& rho, double tol) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho);
    return entropyFromEigenvalues(es.eigenvalues().array(), tol);
}

double
MutualInformation::vonNeumannEntropy(std::vector<double> const& eigenvalues,
                                     double tol) {
    // Already-diagonal input: entropy is -Σ pᵢ log pᵢ directly, matching
    // entropyFromEigenvalues (no re-diagonalisation).
    double S = 0.0;
    for (double p : eigenvalues)
        if (p > tol) S -= p * std::log(p);
    return S;
}

double
MutualInformation::siteSite(itensor::MPS const& psi, int i, int j) {
    if (i == j) return 0.0;
    if (i > j) std::swap(i, j);

    auto rho_i  = oneSiteReducedDensity(psi, i);
    auto rho_j  = oneSiteReducedDensity(psi, j);
    auto rho_ij = twoSiteReducedDensity(psi, i, j);

    const double Si  = vonNeumannEntropy(rho_i);
    const double Sj  = vonNeumannEntropy(rho_j);
    const double Sij = vonNeumannEntropy(rho_ij);
    return Si + Sj - Sij;
}

Eigen::MatrixXd
MutualInformation::allPairs(itensor::MPS const& psi) {
    using namespace itensor;
    const int N = length(psi);
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(N, N);
    if (N < 2) return out;

    // Cache the single-site entropies; otherwise each pair recomputes
    // S(ρ_i) and S(ρ_j).
    std::vector<double> S_single(N + 1, 0.0);  // 1-based
    for (int i = 1; i <= N; ++i) {
        S_single[i] = vonNeumannEntropy(oneSiteReducedDensity(psi, i));
    }

    for (int i = 1; i <= N; ++i) {
        for (int j = i + 1; j <= N; ++j) {
            const auto rho_ij = twoSiteReducedDensity(psi, i, j);
            const double Sij  = vonNeumannEntropy(rho_ij);
            const double I_ij = S_single[i] + S_single[j] - Sij;
            // Symmetric, 0-based output matrix.
            out(i - 1, j - 1) = I_ij;
            out(j - 1, i - 1) = I_ij;
        }
    }
    return out;
}

double
MutualInformation::edgeLength(double I, double epsilon) noexcept {
    if (I < epsilon) return std::numeric_limits<double>::infinity();
    return -std::log(I);
}

Eigen::MatrixXd
MutualInformation::edgeLength(Eigen::MatrixXd const& I, double epsilon) {
    return I.unaryExpr([epsilon](double v) {
        return v < epsilon ? std::numeric_limits<double>::infinity()
                           : -std::log(v);
    });
}

// ─── Dual / bond-cut observables ─────────────────────────────────────

namespace {

// Shannon entropy in nats from a list of non-negative probabilities.
// Renormalises against round-off; entries below `tol` contribute 0.
double shannonEntropy(std::vector<double> const& probs,
                        double tol = 1e-12) {
    double total = 0.0;
    for (auto p : probs) total += p;
    if (total <= 0.0) return 0.0;
    double H = 0.0;
    for (auto p : probs) {
        const double pn = p / total;
        if (pn > tol) H -= pn * std::log(pn);
    }
    return H;
}

} // namespace

double
MutualInformation::bondEntropy(itensor::MPS const& psi, int k) {
    const int N = itensor::length(psi);
    if (k < 1 || k > N - 1) {
        throw std::invalid_argument(
            "MutualInformation::bondEntropy: bond k must be in [1, N-1]");
    }
    // The bipartition [1..k] | [k+1..N] is what ``Schmidt::of(psi, 1, k)``
    // returns the spectrum of: eigenvalues of the reduced density matrix
    // on the left half. Shannon entropy of that spectrum is the bond
    // entanglement entropy.
    auto eigs = Schmidt::of(psi, 1, k);
    return shannonEntropy(eigs);
}

double
MutualInformation::regionEntropy(itensor::MPS const& psi_in, int i, int j) {
    using namespace itensor;
    const int N = length(psi_in);
    if (i > j) std::swap(i, j);
    if (i < 1 || j > N) {
        throw std::invalid_argument(
            "MutualInformation::regionEntropy: require 1 <= i <= j <= N");
    }
    // Trivial bipartition (whole chain | empty): pure state ⇒ S = 0.
    if (i == 1 && j == N) return 0.0;

    // χ⁴ transfer-matrix sweep: contract sites i..j with their
    // conjugates, tracing the physical index at each site and keeping
    // the boundary bond pair (l, l', r, r') open. The resulting tensor K
    // has eigenvalues equal to the non-zero spectrum of ρ_{[i, j]}, and
    // its memory footprint is bounded by χ⁴ regardless of |j - i|,
    // avoiding the d^L blow-up of a dense contraction.
    //
    // Boundary cases (left of i is left-canonical, right of j is right-
    // canonical when psi.position(i) has been called):
    //   • i = 1:  K starts as a scalar 1. The left-bond axes of K are
    //             absent throughout the sweep; the final K has rank 2
    //             on (r_j, r_j').
    //   • j = N:  the rightmost MPS tensor has a trivial right bond;
    //             K's final rank is on (l_{i-1}, l_{i-1}').
    //   • interior: K has rank 4 on (l_{i-1}, l_{i-1}', r_j, r_j').

    MPS psi = psi_in;
    psi.position(i);

    // χ⁴ K-matrix sweep. K starts as scalar 1; after the first site
    // multiplication the boundary bond pair (l_{i-1}, prime(l_{i-1}))
    // and the running bond pair (l_i, prime(l_i)) become open
    // simultaneously and the boundary pair persists through every
    // subsequent site. Bra has all link indices primed (so ket / bra
    // bonds stay distinct); site indices stay unprimed on both so
    // they auto-trace.
    ITensor K(1.0);
    for (int k = i; k <= j; ++k) {
        ITensor const& A = psi(k);
        ITensor bra = dag(A);
        if (k > 1) {
            auto leftL = commonIndex(psi(k - 1), psi(k));
            bra.prime(leftL);
        }
        if (k < N) {
            auto rightL = commonIndex(psi(k), psi(k + 1));
            bra.prime(rightL);
        }
        K *= A;
        K *= bra;
    }

    // Extract K's open indices — exactly the un-traced bond pairs at the
    // interval boundaries — split them into ``ket`` (unprimed) and
    // ``bra`` (primed) sides, and pack them into a Hermitian Eigen
    // matrix for the eigendecomposition.
    std::vector<Index> ketSide;
    std::vector<Index> braSide;
    for (auto const& ind : K.inds()) {
        if (ind.primeLevel() == 0) ketSide.push_back(ind);
        else                        braSide.push_back(ind);
    }
    // Each ket-side index should pair with a primed sibling on the bra
    // side; the pairs come in the order links were primed.
    int dimK = 1;
    for (auto const& ind : ketSide) dimK *= dim(ind);
    int dimB = 1;
    for (auto const& ind : braSide) dimB *= dim(ind);
    if (dimK != dimB) {
        throw std::runtime_error(
            "regionEntropy: ket / bra side dimension mismatch — "
            "indicates a priming bug in the transfer sweep");
    }

    Eigen::MatrixXcd M(dimK, dimK);
    auto [C_ket, rowIdx] = combiner(IndexSet(ketSide));
    auto [C_bra, colIdx] = combiner(IndexSet(braSide));
    auto Kflat = K * C_ket * C_bra;
    for (int a = 1; a <= dimK; ++a) {
        for (int b = 1; b <= dimK; ++b) {
            M(a - 1, b - 1) =
                eltC(Kflat, rowIdx = a, colIdx = b);
        }
    }
    // Symmetrise against numerical noise.
    M = 0.5 * (M + M.adjoint());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "regionEntropy: SelfAdjointEigenSolver failed");
    }
    auto evs = es.eigenvalues();
    std::vector<double> probs;
    probs.reserve(static_cast<std::size_t>(evs.size()));
    for (int k = 0; k < evs.size(); ++k) {
        probs.push_back(std::max(0.0, evs[k]));
    }
    return shannonEntropy(probs);
}

double
MutualInformation::tripartiteInformation(itensor::MPS const& psi,
                                            int n, int m) {
    const int N = itensor::length(psi);
    if (n < 1 || m < 1 || n >= N || m >= N) {
        throw std::invalid_argument(
            "MutualInformation::tripartiteInformation: bonds must be "
            "in [1, N-1]");
    }
    if (n == m) return 0.0;
    if (n > m) std::swap(n, m);
    // A = [1..n], B = [n+1..m], C = [m+1..N].
    const double S_A = regionEntropy(psi, 1, n);
    const double S_C = regionEntropy(psi, m + 1, N);
    const double S_B = regionEntropy(psi, n + 1, m);
    return S_A + S_C - S_B;
}

Eigen::MatrixXd
MutualInformation::allBondPairs(itensor::MPS const& psi) {
    const int N = itensor::length(psi);
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(N - 1, N - 1);
    if (N < 3) return out;
    for (int n = 1; n <= N - 1; ++n) {
        for (int m = n + 1; m <= N - 1; ++m) {
            const double I = tripartiteInformation(psi, n, m);
            out(n - 1, m - 1) = I;
            out(m - 1, n - 1) = I;
        }
    }
    return out;
}

} // namespace tessera::quantum
