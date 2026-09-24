// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_QUARKCONDITIONS_H
#define TESSERA_OBSERVABLES_QUARKCONDITIONS_H

// The quark verdict of the whitepaper's Section 10 in its v16 form: seven
// derived conditions, each reported by name as passed, failed or not
// evaluable, from the certificates a caller measured.
//
// ## Why a separate verdict
//
// `ParticleClusters::classifyQuark` implements the earlier (v15) reading: it
// requires a rank-three colour band on the cluster itself, gates on a
// modulus-based anchor score, and folds flavour and Gauss-flux charge into its
// confidence. The v16 conditions differ on each of those points: the colour
// fibre is a per-sheet base band tensored with the sheet space
// (E-bar (x) C^3), the anchor is a projective profile with no modulus in its
// definition, and flavour and charge are not quark conditions at all. This
// class states the v16 conditions and nothing else, and it decides nothing a
// caller did not measure: a condition whose required certificate is absent is
// "not evaluable", never "passed".
//
// ## The seven conditions (WP v16 Section 10)
//
//   1 persistent-cluster      Q is a persistent cluster certified as in
//                             Section 5, however its support was proposed.
//   2 color-spin-fiber        Its colour-spin fibre is E-bar_Q (x) C^3: a
//                             per-sheet base band (the coexact triplet at flat
//                             connection, or the j = 1/2 doublet of an
//                             odd-monopole support) on a three-sheeted
//                             support.
//   3 anchor-atlas            Its anchor atlas (projective profile, invariant
//                             coordinates, determinant transitions) is stable.
//   4 odd-occupation          It is a single occupied mode of that fibre: its
//                             occupation parity is odd.
//   5 color-transport         Its colour transport S in GL(3, C) stays full
//                             rank, and its base transport has bounded
//                             leakage, over its lifetime.
//   6 lineage                 Its oriented cluster lineage has intersection
//                             number N_Q = +1 with a separating cut (and,
//                             where the determinant family closes, winding
//                             nu = +1).
//   7 fingerprint             Its total spectral fingerprint is stable under
//                             refinement and vertex relabeling.
//
// Everything here is a pure function of caller-supplied data: no solver call,
// no Spacetime mutation, and nothing enters the emergence objective.

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace tessera::observables {

/// The outcome of one quark condition.
///
/// * `Passed` — every certificate the condition requires was measured and
///   held.
/// * `Failed` — some measured certificate of the condition did not hold.
/// * `NotEvaluable` — no measured certificate failed, but at least one
///   required certificate was not measured (for instance a lifetime across
///   several cobordism frames on a single-level run).
enum class QuarkConditionStatus { Passed, Failed, NotEvaluable };

/// One measured (or unmeasured) certificate offered as evidence for a
/// condition.
struct QuarkConditionEvidence {
    /// The certificate's name, one of `requiredEvidence(condition)` or an
    /// additional name the caller reports beside them.
    std::string name{};
    /// Whether it held; empty when it was not measured.
    std::optional<bool> held{};
    /// The measured value or the reason it was not measured, as text, so the
    /// verdict carries the number it rests on.
    std::string detail{};
};

/// The read of one condition.
struct QuarkConditionRead {
    /// The condition's number, 1 to 7, in the whitepaper's order.
    int number{0};
    /// Its short name (see the file comment).
    std::string name{};
    /// The condition as the whitepaper states it.
    std::string statement{};
    /// The outcome.
    QuarkConditionStatus status{QuarkConditionStatus::NotEvaluable};
    /// Every evidence item the caller supplied for it, in the caller's order,
    /// followed by an unmeasured item for each required certificate the caller
    /// did not supply.
    std::vector<QuarkConditionEvidence> evidence{};
    /// The names of the required certificates that were not measured.
    std::vector<std::string> missing{};
    /// The names of the supplied certificates that did not hold.
    std::vector<std::string> failing{};
};

/// The v16 verdict over all seven conditions.
struct QuarkVerdict {
    /// The seven reads, in the whitepaper's order.
    std::vector<QuarkConditionRead> conditions{};
    /// Whether every condition passed: the candidate is a certified quark.
    bool certified{false};
    /// The names of the conditions that failed.
    std::vector<std::string> failed{};
    /// The names of the conditions that could not be evaluated.
    std::vector<std::string> notEvaluable{};
};

/// # QuarkConditions
///
/// The Section 10 quark conditions of the whitepaper, v16, as a verdict over
/// caller-measured certificates.
///
/// Each condition names the certificates it requires. A caller supplies, per
/// condition, the certificates it measured, with whether each held; the
/// verdict is `Failed` when any supplied certificate did not hold, `Passed`
/// when every required certificate was supplied and held, and `NotEvaluable`
/// otherwise. A supplied certificate outside the required list (an optional
/// one, such as the determinant winding of condition 6, which the whitepaper
/// asks for only "where the determinant family closes") can fail a condition
/// but its absence never blocks one.
class QuarkConditions {
  public:
    QuarkConditions() = delete;  // static kernel — no instances.

    /// The number of conditions, seven.
    static constexpr int kConditionCount = 7;

    /// The short names of the seven conditions, in order.
    [[nodiscard]] static std::vector<std::string> conditionNames();

    /// The whitepaper's statement of condition `number` (1 to 7).
    /// @throws std::invalid_argument when `number` is out of range.
    [[nodiscard]] static std::string statement(int number);

    /// The certificates condition `number` requires:
    ///
    ///   1: persistent-support, localized-projector-rank, contour-separation,
    ///      successor-overlap, multi-frame-lifetime, external-leakage;
    ///   2: three-sheeted-support, sheet-isomorphism, protected-base-band,
    ///      base-band-sector, fibre-lift;
    ///   3: anchor-profile-nonzero, anchor-covariance, anchor-transitions,
    ///      anchor-stable-across-frames;
    ///   4: odd-occupation-parity;
    ///   5: color-transport-full-rank, base-transport-leakage,
    ///      transport-over-lifetime;
    ///   6: lineage-intersection;
    ///   7: refinement-stability, relabeling-stability.
    ///
    /// @throws std::invalid_argument when `number` is out of range.
    [[nodiscard]] static std::vector<std::string> requiredEvidence(int number);

    /// The read of one condition from the evidence supplied for it.
    /// @throws std::invalid_argument when `number` is out of range or an
    ///         evidence item has an empty name.
    [[nodiscard]] static QuarkConditionRead evaluateCondition(
        int number, const std::vector<QuarkConditionEvidence>& evidence);

    /// The verdict over all seven conditions; `evidence[i]` is the evidence
    /// for condition i + 1.
    /// @throws std::invalid_argument as `evaluateCondition` does.
    [[nodiscard]] static QuarkVerdict evaluate(
        const std::array<std::vector<QuarkConditionEvidence>, 7>& evidence);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_QUARKCONDITIONS_H
