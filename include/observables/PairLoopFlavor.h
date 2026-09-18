// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_PAIR_LOOP_FLAVOR_H
#define TESSERA_OBSERVABLES_PAIR_LOOP_FLAVOR_H

#include <array>
#include <complex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # PairLoopFlavor
///
/// Pair-loop dual-basis flavor read over three emergent holes, on a structure
/// whose periods over those holes carry the color singlet `[1, ω, ω²]`:
///
///   1. A single correlated multi-hole read: the joint carried representative
///      `ψ = EigenstateSynthesis.carriedRepresentative(holes, σ·target)`, not
///      three independent per-hole extractions. The per-hole weight is
///      \f$ w_h = \sigma_h \oint_h \psi \f$ over the five
///      \f$ (-1)^j \f$-signed tetrahedral facets of the removed 4-cell, and the
///      per-hole Dirac-Kähler charge is
///      \f$ q_h = \sum_{c \in \partial h} W_c |\psi_c|^2 \f$.
///   2. Pair loops \f$ \gamma_{ij} \f$ are homologous to `[i]+[j]`, so their
///      period is `w_i + w_j` — arithmetic on the per-hole weights, with no new
///      geometry. The singlet gives the duality
///      \f$ [\gamma_{ij}] = -[k] \f$.
///   3. Criterion (a): 2:1 multiplicity via `rho`. Criterion (b): odd-one-out
///      against the recorded diquark pair, supplied to the constructor and
///      evaluated only when the build history provides it.
///
/// The headline (`compute`) is `rho`; accessors expose the joint read and the
/// verdict. Oriented periods are reported divided by the unit phase of `w0`, so
/// every record leaf is GAUGE- and RELABEL-invariant. Requires 3 holes on a
/// 4-complex.
class PairLoopFlavor : public RegisterObservable {
  public:
    /// The three pair loops as (i, j) hole-index pairs; `γ_ij` encircles i, j.
    static constexpr std::array<std::pair<int, int>, 3> PAIR_LOOPS = {
        {{0, 1}, {0, 2}, {1, 2}}};
    /// Criterion (a): the closest pair's spread over its separation from the odd
    /// one must stay below this for a 2:1 (u:u:d) verdict.
    static constexpr double RHO_MAX = 0.5;
    /// Criterion-(b) status sentinels.
    static constexpr std::string_view kOddDiquarkEvaluated = "evaluated";
    static constexpr std::string_view kOddDiquarkNotEvaluable =
        "not_evaluable(no_provenance)";
    static constexpr std::string_view kRecordKey = "pair_loop_flavor";

    /// The single correlated multi-hole read (the joint read).
    struct JointRead {
      std::vector<int> sigma;                    ///< induced-orientation signs
      double rU = 0.0;                           ///< residualForPeriods of the pin
      std::vector<std::complex<double>> w;       ///< oriented per-hole weights
      std::vector<double> q;                     ///< per-hole Dirac-Kähler charges
      std::vector<std::complex<double>> loopW;   ///< pair-loop periods (w_i+w_j)
      std::vector<double> loopQ;                 ///< pair-loop charges
      std::vector<double> dualResidual;          ///< |w_i+w_j+w_k| per loop
    };

    /// The pre-registered criteria on a finished joint read.
    struct Verdict {
      std::pair<int, int> oddLoop;               ///< the charge-odd pair loop
      int dualHole = 0;                          ///< its complementary hole
      double rho = 0.0;
      bool multiplicity21 = false;
      std::optional<bool> oddIsDiquarkLoop;      ///< empty ⇒ not evaluable
    };

    /// Read over three emergent holes with no recorded diquark pair, leaving
    /// criterion (b) not evaluable.
    PairLoopFlavor() = default;
    /// Read over three emergent holes with the diquark's hole-index pair taken
    /// from the specimen's build history, making criterion (b) decidable.
    explicit PairLoopFlavor(std::pair<int, int> diquarkPair)
        : diquarkPair_(diquarkPair) {}

    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    /// The clustering ratio `rho` divides two small charge differences, which
    /// amplifies eigensolver roundoff to around 1e-13. One tolerance covers
    /// every leaf; raw residuals are reported alongside.
    [[nodiscard]] double gateTol() const override { return 1e-9; }
    [[nodiscard]] int minHoles() const override { return 3; }
    [[nodiscard]] int requiredDimensions() const override { return 4; }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

    /// The joint read (typed).
    [[nodiscard]] JointRead jointRead(const RegisterContext &ctx) const;
    /// The verdict on a finished joint read.
    [[nodiscard]] Verdict evaluateCriteria(const JointRead &read) const;

    /// (odd loop index, rho): the loop whose charge sits farthest from the mean
    /// of the other two, and rho = |spread of the other two| / |that separation|.
    [[nodiscard]] static std::pair<int, double> oddOneOut(
        const std::vector<double> &loopQ);
    /// The hole index dual to the pair loop `γ_ij`: the third index (`3-i-j`).
    [[nodiscard]] static int complementHole(const std::pair<int, int> &pair) {
      return 3 - pair.first - pair.second;
    }

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;

  private:
    std::optional<std::pair<int, int>> diquarkPair_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_PAIR_LOOP_FLAVOR_H
