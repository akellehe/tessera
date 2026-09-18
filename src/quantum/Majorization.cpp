// Concrete implementations of the `MajorizationPredicate` hierarchy and
// the variant-agnostic `Majorization` façade. See
// include/quantum/Majorization.hpp for the abstract contract, the
// partial-order axioms each variant satisfies, and the full
// bibliography; the short keys below point back to it.
//
//   {N1999}  = Nielsen (1999), Phys. Rev. Lett. 83, 436,
//              arXiv:quant-ph/9811053.
//   {B2015}  = Brändén (2015), arXiv:1410.6601.
//   {AN2008} = Aubrun & Nechita (2008), arXiv:0707.0211.

#include "quantum/Majorization.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

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

// Sort `v` non-increasingly and zero-pad to length `n`. Returns a fresh
// vector so the caller's input is unmodified.
std::vector<double> sortedPadded(std::vector<double> const& v,
                                  std::size_t n) {
    std::vector<double> out = v;
    std::sort(out.begin(), out.end(), std::greater<double>{});
    if (out.size() < n) out.resize(n, 0.0);
    return out;
}

// Sort descending and strip trailing zeros (entries below `tol`). Used
// by the log-concavity check, which is only meaningful on the support.
std::vector<double> sortedTrimmed(std::vector<double> const& v, double tol) {
    std::vector<double> out = v;
    std::sort(out.begin(), out.end(), std::greater<double>{});
    while (!out.empty() && out.back() <= tol) out.pop_back();
    return out;
}

} // namespace

// ─── StandardMajorization ────────────────────────────────────────────────

StandardMajorization::StandardMajorization(double tol) noexcept
    : tol_(tol) {}

bool StandardMajorization::majorizes(std::vector<double> const& mu,
                                       std::vector<double> const& lambda) const {
    const std::size_t n = std::max(mu.size(), lambda.size());
    if (n == 0) return true;

    auto mus  = sortedPadded(mu,     n);
    auto lams = sortedPadded(lambda, n);

    // {N1999}: at every k,
    //     sum(mus[0..k])  ≥  sum(lams[0..k])
    // with equality at k = n (the total-mass condition).
    double sumMu = 0.0;
    double sumLa = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        sumMu += mus[k];
        sumLa += lams[k];
        if (sumMu + tol_ < sumLa) return false;
    }
    return std::abs(sumMu - sumLa) <= tol_;
}

std::string StandardMajorization::name() const {
    return "standard";
}

// ─── LogConcaveMajorization ──────────────────────────────────────────────

LogConcaveMajorization::LogConcaveMajorization(double tol) noexcept
    : StandardMajorization(tol) {}

bool LogConcaveMajorization::majorizes(std::vector<double> const& mu,
                                         std::vector<double> const& lambda) const {
    if (!isLogConcave(mu, tol_) || !isLogConcave(lambda, tol_)) return false;
    return StandardMajorization::majorizes(mu, lambda);
}

std::string LogConcaveMajorization::name() const {
    return "log-concave";
}

bool LogConcaveMajorization::isLogConcave(std::vector<double> const& v,
                                            double tol) {
    auto s = sortedTrimmed(v, tol);
    if (s.size() < 3) return true;  // length 0/1/2: trivially log-concave
    // {B2015}: s_i² ≥ s_{i-1} · s_{i+1}, with `tol` slack on the right.
    for (std::size_t i = 1; i + 1 < s.size(); ++i) {
        if (s[i] * s[i] + tol < s[i - 1] * s[i + 1]) return false;
    }
    return true;
}

// ─── PeakRadialMajorization ──────────────────────────────────────────────

PeakRadialMajorization::PeakRadialMajorization(double tol) noexcept
    : tol_(tol) {}

bool PeakRadialMajorization::majorizes(std::vector<double> const& mu,
                                         std::vector<double> const& lambda) const {
    const std::size_t n = std::max(mu.size(), lambda.size());
    if (n == 0) return true;

    auto mus  = sortedPadded(mu,     n);
    auto lams = sortedPadded(lambda, n);

    const double peakMu = mus[0];
    const double peakLa = lams[0];

    if (peakMu <= tol_) return peakLa <= tol_;
    if (peakLa <= tol_) return true;

    // Cross-multiplied form: μᵢ · λ₁ ≤ λᵢ · μ₁ ⟺ μᵢ/μ₁ ≤ λᵢ/λ₁.
    for (std::size_t i = 0; i < n; ++i) {
        if (mus[i] * peakLa > lams[i] * peakMu + tol_) return false;
    }
    return true;
}

std::string PeakRadialMajorization::name() const {
    return "peak-radial";
}

// ─── Majorization façade ─────────────────────────────────────────────────

Poset Majorization::posetOf(std::vector<std::vector<double>> const& spectra,
                              MajorizationPredicate const& predicate) {
    const int N = static_cast<int>(spectra.size());
    Poset out(N);
    if (N == 0) return out;

    // Pre-compute the strict adjacency: O(N²) predicate calls here make
    // the transitive-reduction loop below O(N³) boolean operations
    // instead of O(N³) sort-and-compare operations.
    std::vector<std::vector<char>> strict(
        static_cast<std::size_t>(N),
        std::vector<char>(static_cast<std::size_t>(N), 0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (predicate.strictlyMajorizes(
                    spectra[static_cast<std::size_t>(i)],
                    spectra[static_cast<std::size_t>(j)])) {
                strict[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(j)] = 1;
            }
        }
    }

    // Transitive reduction: an edge (i, j) in the strict graph survives
    // iff there is no third node k with i ≻ k ≻ j. Correctness depends
    // on transitivity of the underlying predicate.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!strict[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(j)]) continue;
            bool hasIntermediate = false;
            for (int k = 0; k < N; ++k) {
                if (k == i || k == j) continue;
                if (strict[static_cast<std::size_t>(i)]
                          [static_cast<std::size_t>(k)] &&
                    strict[static_cast<std::size_t>(k)]
                          [static_cast<std::size_t>(j)]) {
                    hasIntermediate = true;
                    break;
                }
            }
            if (!hasIntermediate) {
                out.addCover(i, j);
            }
        }
    }
    return out;
}

Poset Majorization::posetOf(std::vector<std::vector<double>> const& spectra,
                              double tol) {
    return posetOf(spectra, StandardMajorization{tol});
}

OrderAgreement Majorization::agreement(Poset const& a, Poset const& b,
                                         int nLabels) {
    return ::tessera::compareOrders(a, b, nLabels);
}

} // namespace tessera::quantum
