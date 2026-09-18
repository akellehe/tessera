// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Shared Lanczos-based spectral-graph machinery.
//
// Implements ``SpectralGraph::diagonalHeatKernel``,
// ``::returnProbability``, ``::spectralDimension``, and
// ``::spectralDimensionSmoothed`` for any subclass that provides an
// ``applyLaplacian`` matvec.
//
// The Padé-13 scaling-and-squaring dense matrix exponential and the
// normal-equations least-squares solver live in the anonymous namespace
// below — no Eigen dependency, so this file is safe to link into
// ``tessera_core`` (whose header surface promises no Eigen).

#include "graph/SpectralGraph.hpp"

#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

namespace {

// Small dense k×k matrix stored row-major. Used only for the projected
// tridiagonal exponential and the local polynomial fit; k is bounded by
// the caller's ``krylovDim`` and ``windowSize`` arguments (typically
// ≤ 30) so the O(k^3) operations here are cheap.
class DenseMatrixK {
public:
    std::size_t k;
    std::vector<double> data;
    explicit DenseMatrixK(std::size_t k_) : k(k_), data(k_ * k_, 0.0) {}
    double& at(std::size_t i, std::size_t j)       { return data[i * k + j]; }
    double  at(std::size_t i, std::size_t j) const { return data[i * k + j]; }

    double oneNorm() const {
        double m = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < k; ++i) s += std::abs(at(i, j));
            if (s > m) m = s;
        }
        return m;
    }
};

DenseMatrixK matMul(DenseMatrixK const& A, DenseMatrixK const& B) {
    DenseMatrixK C(A.k);
    for (std::size_t i = 0; i < A.k; ++i) {
        for (std::size_t l = 0; l < A.k; ++l) {
            double a = A.at(i, l);
            if (a == 0.0) continue;
            for (std::size_t j = 0; j < A.k; ++j) {
                C.at(i, j) += a * B.at(l, j);
            }
        }
    }
    return C;
}

void axpy(DenseMatrixK& A, double s, DenseMatrixK const& B) {
    for (std::size_t i = 0; i < A.data.size(); ++i) A.data[i] += s * B.data[i];
}

DenseMatrixK identityK(std::size_t k) {
    DenseMatrixK I(k);
    for (std::size_t i = 0; i < k; ++i) I.at(i, i) = 1.0;
    return I;
}

// Solve A · X = B via Gauss-Jordan with partial pivoting. A is consumed.
// Returns false on numerical singularity. B is overwritten with X.
bool solveLinearSystem(DenseMatrixK A, DenseMatrixK& B) {
    const std::size_t k = A.k;
    for (std::size_t i = 0; i < k; ++i) {
        std::size_t piv = i;
        double pivVal = std::abs(A.at(i, i));
        for (std::size_t r = i + 1; r < k; ++r) {
            double v = std::abs(A.at(r, i));
            if (v > pivVal) { pivVal = v; piv = r; }
        }
        if (pivVal < 1e-300) return false;
        if (piv != i) {
            for (std::size_t c = 0; c < k; ++c) {
                std::swap(A.at(i, c), A.at(piv, c));
                std::swap(B.at(i, c), B.at(piv, c));
            }
        }
        double inv = 1.0 / A.at(i, i);
        for (std::size_t c = 0; c < k; ++c) {
            A.at(i, c) *= inv;
            B.at(i, c) *= inv;
        }
        for (std::size_t r = 0; r < k; ++r) {
            if (r == i) continue;
            double f = A.at(r, i);
            if (f == 0.0) continue;
            for (std::size_t c = 0; c < k; ++c) {
                A.at(r, c) -= f * A.at(i, c);
                B.at(r, c) -= f * B.at(i, c);
            }
        }
    }
    return true;
}

// Padé-13 scaling-and-squaring (Higham 2010 Alg. 2.3). For symmetric
// tridiagonal inputs (which we always pass) the eigenvector-based
// alternative is faster, but this path is dependency-free and works for
// arbitrary small dense matrices.
DenseMatrixK matExpPade13(DenseMatrixK A) {
    const std::size_t k = A.k;
    static const double b[14] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0,  129060195264000.0,    10559470521600.0,
        670442572800.0,      33522128640.0,        1323241920.0,
        40840800.0,          960960.0,             16380.0,
        182.0,               1.0
    };
    const double theta13 = 5.371920351148152;
    double normA = A.oneNorm();
    int s = 0;
    if (normA > theta13) s = static_cast<int>(std::ceil(std::log2(normA / theta13)));
    if (s > 0) {
        double scale = std::ldexp(1.0, -s);
        for (auto& v : A.data) v *= scale;
    }
    DenseMatrixK A2 = matMul(A, A);
    DenseMatrixK A4 = matMul(A2, A2);
    DenseMatrixK A6 = matMul(A4, A2);
    DenseMatrixK Ik = identityK(k);
    DenseMatrixK innerU(k);
    axpy(innerU, b[13], A6);
    axpy(innerU, b[11], A4);
    axpy(innerU, b[9],  A2);
    DenseMatrixK A6_innerU = matMul(A6, innerU);
    DenseMatrixK U_inside(k);
    axpy(U_inside, b[7], A6);
    axpy(U_inside, b[5], A4);
    axpy(U_inside, b[3], A2);
    axpy(U_inside, b[1], Ik);
    for (std::size_t i = 0; i < U_inside.data.size(); ++i)
        U_inside.data[i] += A6_innerU.data[i];
    DenseMatrixK U = matMul(A, U_inside);
    DenseMatrixK innerV(k);
    axpy(innerV, b[12], A6);
    axpy(innerV, b[10], A4);
    axpy(innerV, b[8],  A2);
    DenseMatrixK A6_innerV = matMul(A6, innerV);
    DenseMatrixK V(k);
    axpy(V, b[6], A6);
    axpy(V, b[4], A4);
    axpy(V, b[2], A2);
    axpy(V, b[0], Ik);
    for (std::size_t i = 0; i < V.data.size(); ++i)
        V.data[i] += A6_innerV.data[i];
    DenseMatrixK numer(k), denom(k);
    for (std::size_t i = 0; i < numer.data.size(); ++i) {
        numer.data[i] = V.data[i] + U.data[i];
        denom.data[i] = V.data[i] - U.data[i];
    }
    if (!solveLinearSystem(denom, numer)) return identityK(k);
    DenseMatrixK R = numer;
    for (int i = 0; i < s; ++i) R = matMul(R, R);
    return R;
}

// Solve the normal equations V^T V a = V^T y for the
// least-squares polynomial fit. V is (m × k) row-major in
// ``vandermonde``; y is length m. Returns the (k-vector) ``coeffs`` or
// nullopt on singularity. Hand-rolled so this TU stays Eigen-free.
bool normalEquationsSolve(std::vector<double> const& vandermonde,
                            int m, int k,
                            std::vector<double> const& y,
                            std::vector<double>& coeffs) {
    DenseMatrixK A(static_cast<std::size_t>(k));
    DenseMatrixK b(static_cast<std::size_t>(k));
    // A = V^T V, b = V^T y (stored as k×k with rhs in column 0).
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            double s = 0.0;
            for (int r = 0; r < m; ++r) {
                s += vandermonde[static_cast<std::size_t>(r * k + i)] *
                     vandermonde[static_cast<std::size_t>(r * k + j)];
            }
            A.at(static_cast<std::size_t>(i),
                   static_cast<std::size_t>(j)) = s;
        }
        double s = 0.0;
        for (int r = 0; r < m; ++r) {
            s += vandermonde[static_cast<std::size_t>(r * k + i)] *
                 y[static_cast<std::size_t>(r)];
        }
        b.at(static_cast<std::size_t>(i), 0) = s;
    }
    if (!solveLinearSystem(A, b)) return false;
    coeffs.resize(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        coeffs[static_cast<std::size_t>(i)] =
            b.at(static_cast<std::size_t>(i), 0);
    }
    return true;
}

} // anonymous namespace

std::vector<double>
SpectralGraph::diagonalHeatKernel(std::vector<int> const& starts,
                                     std::vector<double> const& sigmas,
                                     int krylovDim) const {
    const int n     = nVertices();
    const int nSt   = static_cast<int>(starts.size());
    const int nSig  = static_cast<int>(sigmas.size());
    std::vector<double> out(static_cast<std::size_t>(nSt) * nSig, 0.0);
    if (n == 0 || nSt == 0 || nSig == 0) return out;

    // The outer loop over `starts` is embarrassingly parallel: each
    // iteration builds its own Krylov subspace from a distinct unit
    // vector, calls applyLaplacian (const, thread-safe), and writes
    // only to its own contiguous slice of `out` (positions
    // s*nSig .. (s+1)*nSig). Scratch buffers (v, w, prev, V, alpha,
    // beta) are per-iteration and therefore per-thread under OpenMP.
    //
    // `schedule(dynamic)` because per-start cost varies (sparser
    // neighborhoods break out of the kMax loop earlier when normW <
    // 1e-12). Bit-identical to the serial version — same arithmetic
    // in the same order within each iteration, and inter-iteration
    // independence means thread ordering doesn't affect output.
    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < nSt; ++s) {
        const int start = starts[static_cast<std::size_t>(s)];
        if (start < 0 || start >= n) continue;

        std::vector<double> v(static_cast<std::size_t>(n));
        std::vector<double> w(static_cast<std::size_t>(n));
        std::vector<double> prev(static_cast<std::size_t>(n));
        std::vector<std::vector<double>> V;
        std::vector<double> alpha, beta;

        v[static_cast<std::size_t>(start)] = 1.0;
        V.assign(1, v);

        const int kMax = std::min(krylovDim, n);
        for (int j = 0; j < kMax; ++j) {
            applyLaplacian(v, w);
            double a = 0.0;
            for (int i = 0; i < n; ++i)
                a += v[static_cast<std::size_t>(i)] * w[static_cast<std::size_t>(i)];
            alpha.push_back(a);
            if (j + 1 == kMax) break;

            for (int i = 0; i < n; ++i) {
                w[static_cast<std::size_t>(i)] -= a * v[static_cast<std::size_t>(i)];
                if (j > 0) {
                    w[static_cast<std::size_t>(i)] -=
                        beta.back() * prev[static_cast<std::size_t>(i)];
                }
            }
            // Full Gram-Schmidt against existing basis (numerical stability).
            for (auto const& u : V) {
                double dot = 0.0;
                for (int i = 0; i < n; ++i)
                    dot += u[static_cast<std::size_t>(i)] *
                            w[static_cast<std::size_t>(i)];
                for (int i = 0; i < n; ++i)
                    w[static_cast<std::size_t>(i)] -=
                        dot * u[static_cast<std::size_t>(i)];
            }
            double normW = 0.0;
            for (int i = 0; i < n; ++i)
                normW += w[static_cast<std::size_t>(i)] *
                          w[static_cast<std::size_t>(i)];
            normW = std::sqrt(normW);
            if (normW < 1e-12) break;
            beta.push_back(normW);
            prev = v;
            const double inv = 1.0 / normW;
            for (int i = 0; i < n; ++i)
                v[static_cast<std::size_t>(i)] =
                    w[static_cast<std::size_t>(i)] * inv;
            V.push_back(v);
        }

        const int actualK = static_cast<int>(alpha.size());
        if (actualK == 0) {
            for (int j = 0; j < nSig; ++j)
                out[static_cast<std::size_t>(s) * nSig + j] = 1.0;
            continue;
        }

        for (int j = 0; j < nSig; ++j) {
            const double sigma = sigmas[static_cast<std::size_t>(j)];
            DenseMatrixK T(static_cast<std::size_t>(actualK));
            for (int i = 0; i < actualK; ++i)
                T.at(i, i) = -sigma * alpha[static_cast<std::size_t>(i)];
            for (int i = 0; i + 1 < actualK; ++i) {
                T.at(static_cast<std::size_t>(i),
                       static_cast<std::size_t>(i + 1)) =
                    -sigma * beta[static_cast<std::size_t>(i)];
                T.at(static_cast<std::size_t>(i + 1),
                       static_cast<std::size_t>(i)) =
                    -sigma * beta[static_cast<std::size_t>(i)];
            }
            DenseMatrixK eT = matExpPade13(T);
            out[static_cast<std::size_t>(s) * nSig + j] = eT.at(0, 0);
        }
    }
    return out;
}

std::vector<double>
SpectralGraph::returnProbability(std::vector<double> const& sigmas,
                                    int krylovDim,
                                    int m,
                                    std::uint64_t seed) const {
    const int n = nVertices();
    std::vector<double> P(sigmas.size(), 0.0);
    if (n == 0 || sigmas.empty()) return P;

    // Subsample a random m-subset of starting vertices.
    // m <= 0 → default `min(n, 3000)`. m >= n → all vertices (exact).
    int sampleSize;
    if (m <= 0) {
        constexpr int kDefaultM = 3000;
        sampleSize = std::min(n, kDefaultM);
    } else {
        sampleSize = std::min(n, m);
    }

    std::vector<int> starts;
    starts.reserve(static_cast<std::size_t>(sampleSize));
    if (sampleSize == n) {
        for (int i = 0; i < n; ++i) starts.push_back(i);
    } else {
        // Reservoir sampling: deterministic, without replacement.
        starts.resize(static_cast<std::size_t>(sampleSize));
        for (int i = 0; i < sampleSize; ++i) starts[static_cast<std::size_t>(i)] = i;
        std::mt19937_64 rng(seed);
        for (int i = sampleSize; i < n; ++i) {
            std::uniform_int_distribution<int> dist(0, i);
            const int j = dist(rng);
            if (j < sampleSize) starts[static_cast<std::size_t>(j)] = i;
        }
    }

    auto diag = diagonalHeatKernel(starts, sigmas, krylovDim);
    const int nSig = static_cast<int>(sigmas.size());
    for (int i = 0; i < sampleSize; ++i) {
        for (int j = 0; j < nSig; ++j) {
            P[static_cast<std::size_t>(j)] +=
                diag[static_cast<std::size_t>(i) * nSig + j];
        }
    }
    // Estimator: P(σ) ≈ (1/m) Σ_{s ∈ sample} [e^{-σL}]_{s,s}.
    // For an exact trace estimator the prefactor would carry an extra
    // (n/m) factor, but the convention everywhere in this codebase is
    // P(σ) = (1/n) Tr e^{-σL} ≈ mean of diagonals — which the m-subset
    // estimator returns directly. ``spectralDimension`` is a log-derivative
    // of P so a constant prefactor cancels regardless.
    const double invM = 1.0 / static_cast<double>(sampleSize);
    for (auto& p : P) p *= invM;
    return P;
}

std::vector<double>
SpectralGraph::spectralDimension(std::vector<double> const& sigmas,
                                    std::vector<double> const& P) {
    const int n = static_cast<int>(sigmas.size());
    std::vector<double> dS(static_cast<std::size_t>(n),
                            std::numeric_limits<double>::quiet_NaN());
    if (n < 2 || static_cast<int>(P.size()) != n) return dS;

    std::vector<double> logSig(static_cast<std::size_t>(n));
    std::vector<double> logP(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double s = sigmas[static_cast<std::size_t>(i)];
        const double p = P[static_cast<std::size_t>(i)];
        if (s <= 0.0 || p <= 0.0) {
            logSig[static_cast<std::size_t>(i)] =
                std::numeric_limits<double>::quiet_NaN();
            logP[static_cast<std::size_t>(i)] =
                std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        logSig[static_cast<std::size_t>(i)] = std::log(s);
        logP[static_cast<std::size_t>(i)]   = std::log(p);
    }

    auto finite = [](double x) { return std::isfinite(x); };
    for (int i = 0; i < n; ++i) {
        double slope;
        if (i == 0) {
            const double a = logP[1] - logP[0];
            const double b = logSig[1] - logSig[0];
            slope = (finite(a) && finite(b) && b != 0.0)
                        ? a / b : std::nan("");
        } else if (i == n - 1) {
            const double a = logP[static_cast<std::size_t>(n - 1)] -
                              logP[static_cast<std::size_t>(n - 2)];
            const double b = logSig[static_cast<std::size_t>(n - 1)] -
                              logSig[static_cast<std::size_t>(n - 2)];
            slope = (finite(a) && finite(b) && b != 0.0)
                        ? a / b : std::nan("");
        } else {
            const double a = logP[static_cast<std::size_t>(i + 1)] -
                              logP[static_cast<std::size_t>(i - 1)];
            const double b = logSig[static_cast<std::size_t>(i + 1)] -
                              logSig[static_cast<std::size_t>(i - 1)];
            slope = (finite(a) && finite(b) && b != 0.0)
                        ? a / b : std::nan("");
        }
        dS[static_cast<std::size_t>(i)] = -2.0 * slope;
    }
    return dS;
}

std::vector<double>
SpectralGraph::spectralDimensionSmoothed(std::vector<double> const& sigmas,
                                            std::vector<double> const& P,
                                            int windowSize,
                                            int polyOrder) {
    const int n = static_cast<int>(sigmas.size());
    std::vector<double> dS(static_cast<std::size_t>(n),
                            std::numeric_limits<double>::quiet_NaN());
    if (n < 2 || static_cast<int>(P.size()) != n) return dS;
    if (windowSize < 3 || (windowSize % 2) == 0 ||
        polyOrder < 1 || polyOrder + 1 > windowSize) {
        throw std::invalid_argument(
            "SpectralGraph::spectralDimensionSmoothed: require windowSize "
            "odd >= 3, polyOrder >= 1, polyOrder + 1 <= windowSize");
    }
    const int half = windowSize / 2;

    std::vector<double> logSig(static_cast<std::size_t>(n));
    std::vector<double> logP(static_cast<std::size_t>(n));
    std::vector<bool>   ok(static_cast<std::size_t>(n), false);
    for (int i = 0; i < n; ++i) {
        const double s = sigmas[static_cast<std::size_t>(i)];
        const double p = P[static_cast<std::size_t>(i)];
        if (s > 0.0 && p > 0.0) {
            logSig[static_cast<std::size_t>(i)] = std::log(s);
            logP[static_cast<std::size_t>(i)]   = std::log(p);
            ok[static_cast<std::size_t>(i)]     = true;
        }
    }

    for (int i = 0; i < n; ++i) {
        if (!ok[static_cast<std::size_t>(i)]) continue;
        int lo = std::max(0, i - half);
        int hi = std::min(n - 1, i + half);
        if (hi - lo + 1 < polyOrder + 1) {
            // Fallback to finite difference for short effective windows.
            double slope = 0.0;
            if (i == 0 || i == n - 1) {
                int a = (i == 0) ? 0 : n - 2;
                int b = a + 1;
                if (ok[static_cast<std::size_t>(a)] &&
                    ok[static_cast<std::size_t>(b)] &&
                    logSig[static_cast<std::size_t>(b)] !=
                        logSig[static_cast<std::size_t>(a)]) {
                    slope = (logP[static_cast<std::size_t>(b)] -
                              logP[static_cast<std::size_t>(a)]) /
                            (logSig[static_cast<std::size_t>(b)] -
                              logSig[static_cast<std::size_t>(a)]);
                }
            } else {
                if (ok[static_cast<std::size_t>(i - 1)] &&
                    ok[static_cast<std::size_t>(i + 1)] &&
                    logSig[static_cast<std::size_t>(i + 1)] !=
                        logSig[static_cast<std::size_t>(i - 1)]) {
                    slope = (logP[static_cast<std::size_t>(i + 1)] -
                              logP[static_cast<std::size_t>(i - 1)]) /
                            (logSig[static_cast<std::size_t>(i + 1)] -
                              logSig[static_cast<std::size_t>(i - 1)]);
                }
            }
            dS[static_cast<std::size_t>(i)] = -2.0 * slope;
            continue;
        }

        const int K = polyOrder + 1;
        std::vector<double> vandermonde;
        std::vector<double> y;
        vandermonde.reserve(static_cast<std::size_t>((hi - lo + 1) * K));
        y.reserve(static_cast<std::size_t>(hi - lo + 1));
        const double xCenter = logSig[static_cast<std::size_t>(i)];
        int m = 0;
        for (int k = lo; k <= hi; ++k) {
            if (!ok[static_cast<std::size_t>(k)]) continue;
            const double dx = logSig[static_cast<std::size_t>(k)] - xCenter;
            double pw = 1.0;
            for (int p = 0; p < K; ++p) {
                vandermonde.push_back(pw);
                pw *= dx;
            }
            y.push_back(logP[static_cast<std::size_t>(k)]);
            ++m;
        }
        if (m < K) continue;

        std::vector<double> coeffs;
        if (!normalEquationsSolve(vandermonde, m, K, y, coeffs)) continue;
        // coeffs[1] is the local slope at xCenter.
        dS[static_cast<std::size_t>(i)] = -2.0 * coeffs[1];
    }
    return dS;
}

} // namespace tessera::graph
