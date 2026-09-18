// Majorization partial order on probability distributions, plus the
// Hasse-diagram construction used to build the majorization poset of the
// Schmidt spectra of a matrix product state (MPS).
//
// The Poset / OrderAgreement types live at the top of tessera
// (`include/Poset.h`) so non-quantum analyses can share them; this header
// re-exports them under `tessera::quantum::` and adds the
// quantum-specific predicate hierarchy plus the `Majorization` façade.
//
// ─── Bibliographic keys used throughout this file ────────────────────
//
// {N1999}    Nielsen, "Conditions for a class of entanglement
//            transformations", Phys. Rev. Lett. 83, 436 (1999).
//            arXiv:quant-ph/9811053. PDF in
//            docs/source/resources/quantum/Nielsen1999_LOCC_majorization.pdf.
//            Gives the cumulative-sum-dominance definition of
//            majorization, and the theorem that |α⟩ → |β⟩ is achievable
//            by deterministic LOCC iff λ_α ≺ λ_β (Schmidt-spectrum
//            majorization).
//
// {AN2008}   Aubrun & Nechita, "Stochastic domination for iterated
//            convolutions and catalytic majorization", Comm. Math. Phys.
//            278, 133 (2008). arXiv:0707.0211. PDF in docs/source/
//            resources/quantum/AubrunNechita2008_CatalyticMajorization.pdf.
//            Gives the L^p-norm-dominance characterization of asymptotic
//            and catalytic majorization.
//
// {B2015}    Brändén, "Unimodality, log-concavity, real-rootedness and
//            beyond", Handbook of Enumerative Combinatorics (2015).
//            arXiv:1410.6601. PDF in docs/source/resources/quantum/
//            Branden2015_Unimodality_LogConcavity.pdf. Log-concavity of a
//            sequence: a_i² ≥ a_{i-1} · a_{i+1}.
//
// {B1997}    Bhatia, "Matrix Analysis", Springer GTM 169 (1997), chapter
//            "Majorisation" — the textbook account, and the principal
//            majorization reference cited by {N1999}.
//
// {MOA2011}  Marshall, Olkin & Arnold, "Inequalities: Theory of
//            Majorization and Its Applications", Springer, 2nd ed. (2011)
//            — the encyclopedic reference.
//
// ─── Majorization ──────────────────────────────────────────────
//
// Given finite non-negative sequences μ and λ normalised to the same
// total mass, μ majorizes λ (written μ ≻ λ) iff
//
//   sum_{i=1..k} μ_i^↓  ≥  sum_{i=1..k} λ_i^↓     for every k = 1, 2, …
//
// with x_i^↓ the entries of x sorted non-increasingly and the shorter
// vector zero-padded to the longer's length. μ is then "more
// concentrated" than λ. For probability distributions the total-mass
// equality at k = d is automatic.

#pragma once

#include "Poset.h"  // top-level tessera::Poset / OrderAgreement / compareOrders

#include <cstddef>
#include <string>
#include <utility>
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

// Aliases keeping the quantum-side code paths working unchanged. The
// canonical types live in `tessera::` so non-quantum analyses can use
// them too.
using Poset          = ::tessera::Poset;
using OrderAgreement = ::tessera::OrderAgreement;

// ─── Variant contract ────────────────────────────────────────────────────

// Abstract base class: every concrete variant of "μ majorizes λ" honours
// this single contract. Pass instances by `const&` everywhere; ownership
// stays with the caller.
//
// Subclasses must satisfy the partial-order axioms on the simplex:
//   • reflexivity:    majorizes(x, x) is true for every probability x;
//   • antisymmetry:   majorizes(x, y) and majorizes(y, x) implies x and
//                     y are equivalent (sorted-padded equal, or whatever
//                     equivalence the variant uses);
//   • transitivity:   majorizes(x, y) and majorizes(y, z) implies
//                     majorizes(x, z).
// A subclass that violates any of these breaks `Majorization::posetOf`
// (the transitive-reduction would fail to converge to a Hasse diagram).
//
// Reference: {N1999} — the classical case this interface generalises.
class MajorizationPredicate {
public:
    virtual ~MajorizationPredicate() = default;

    [[nodiscard]] virtual bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const = 0;

    // μ strictly majorizes λ iff μ ≻ λ but not λ ≻ μ. Default
    // implementation is two `majorizes` calls; subclasses with a
    // cheaper strict path can override.
    [[nodiscard]] virtual bool
    strictlyMajorizes(std::vector<double> const& mu,
                       std::vector<double> const& lambda) const {
        return majorizes(mu, lambda) && !majorizes(lambda, mu);
    }

    // Short identifier used in diagnostics and the Python repr
    // ("standard", "log-concave", "peak-radial", …).
    [[nodiscard]] virtual std::string name() const = 0;

protected:
    MajorizationPredicate() = default;
    MajorizationPredicate(MajorizationPredicate const&) = default;
    MajorizationPredicate(MajorizationPredicate&&) noexcept = default;
    MajorizationPredicate& operator=(MajorizationPredicate const&) = default;
    MajorizationPredicate& operator=(MajorizationPredicate&&) noexcept = default;
};

// ─── Concrete variants ───────────────────────────────────────────────────

// Classical majorization, as in {N1999}:
//
//     μ ≻ λ   ⟺   ∑_{i=1..k} μ_i^↓  ≥  ∑_{i=1..k} λ_i^↓   ∀ k = 1..d
//                                       ∧  ∑ μ_i  =  ∑ λ_i .
//
// References: {N1999}; {B1997}; {MOA2011}.
class StandardMajorization : public MajorizationPredicate {
public:
    explicit StandardMajorization(double tol = 1e-12) noexcept;

    [[nodiscard]] bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const override;

    [[nodiscard]] std::string name() const override;

    [[nodiscard]] double tol() const noexcept { return tol_; }

protected:
    double tol_;
};

// Standard majorization, restricted to spectra that are log-concave on
// their support: a_i² ≥ a_{i-1} · a_{i+1}. Pairs where either spectrum
// fails log-concavity are declared incomparable, so this is a strict
// sub-relation of `StandardMajorization`.
//
// Reference: {B2015} — log-concavity and its structural consequences.
class LogConcaveMajorization : public StandardMajorization {
public:
    explicit LogConcaveMajorization(double tol = 1e-12) noexcept;

    [[nodiscard]] bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const override;

    [[nodiscard]] std::string name() const override;

    // Predicate for log-concavity of a single spectrum. After sorting
    // descending and stripping trailing zeros, requires
    // s_i² ≥ s_{i-1} · s_{i+1} for every interior i. Spectra of length
    // ≤ 2 are trivially log-concave.
    [[nodiscard]] static bool
    isLogConcave(std::vector<double> const& v, double tol = 1e-12);
};

// Peak-radial dominance: μ ≻ λ iff, after sorting both descending and
// zero-padding, μ's normalised profile decays no slower than λ's:
//
//     μᵢ / μ₁  ≤  λᵢ / λ₁     for every i.
//
// Cross-multiplied form (used in the implementation for stability):
//     μᵢ · λ₁  ≤  λᵢ · μ₁     for every i.
//
// Strictly stronger than classical majorization.
//
// Reference: {AN2008} — the closest published analog: ratio / L^p
// dominance, in a different direction.
class PeakRadialMajorization : public MajorizationPredicate {
public:
    explicit PeakRadialMajorization(double tol = 1e-12) noexcept;

    [[nodiscard]] bool
    majorizes(std::vector<double> const& mu,
              std::vector<double> const& lambda) const override;

    [[nodiscard]] std::string name() const override;

    [[nodiscard]] double tol() const noexcept { return tol_; }

private:
    double tol_;
};

// ─── Coarse-grained façade ───────────────────────────────────────────────

// Static utility class for majorization-poset construction and pairwise
// order-agreement statistics. Stateless — not instantiable.
//
// `posetOf` builds the Hasse-cover poset on a list of spectra under a
// chosen majorization variant. `agreement` reports pairwise statistics
// (Kendall-τ, discordant fraction, Hasse edit distance) between two
// posets on a shared label set.
class Majorization {
public:
    Majorization() = delete;
    Majorization(Majorization const&) = delete;
    Majorization& operator=(Majorization const&) = delete;

    // Build the Hasse-cover poset of the strict-majorization order under
    // an explicit predicate variant.
    //
    // `spectra[k]` becomes node k; the resulting Poset stores Hasse cover
    // edges only (transitive closure is implicit, recover with the usual
    // reachability traversal).
    //
    // Complexity: O(M³) for M = spectra.size(), dominated by the
    // transitive-reduction pass. Each predicate call is O(L log L) on
    // the spectrum lengths L.
    //
    // Reference: {N1999} — the partial order this poset Hasse-encodes
    // for the `StandardMajorization` predicate.
    [[nodiscard]] static Poset posetOf(
        std::vector<std::vector<double>> const& spectra,
        MajorizationPredicate const& predicate);

    // Build the poset under the classical {N1999} majorization at the
    // given numerical tolerance.
    [[nodiscard]] static Poset posetOf(
        std::vector<std::vector<double>> const& spectra,
        double tol = 1e-12);

    // Pairwise agreement statistics between two posets on the same
    // label set of size nLabels. Delegates to ::tessera::compareOrders;
    // exists here so quantum callers don't need to reach into the
    // top-level tessera namespace.
    //
    // Complexity: O(nLabels^3) for the Floyd-Warshall transitive
    // closures, then O(nLabels^2) to count pairs.
    [[nodiscard]] static OrderAgreement agreement(
        Poset const& a, Poset const& b, int nLabels);
};

} // namespace tessera::quantum
