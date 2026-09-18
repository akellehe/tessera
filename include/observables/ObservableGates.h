// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_OBSERVABLE_GATES_H
#define TESSERA_OBSERVABLES_OBSERVABLE_GATES_H

#include <cstdint>
#include <string>

#include "observables/Record.h"
#include "observables/RegisterContext.h"
#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # ObservableGates
///
/// Gate harness for the GAUGE and RELABEL invariance checks. Post-hoc
/// validation, never a loop condition. Each gate re-measures an observable on a
/// transformed context and reports the maximum absolute delta over every numeric
/// leaf of its record (`Record::reportDelta`); a record channel that is not
/// gauge- and relabel-invariant is flagged as a leak.
///
///   * GAUGE — re-measure on `ctx.gauged(GAUGE_THETA)`: the same live complex,
///     with the register target rotated by the surviving global U(1) phase. No
///     reconstruction.
///   * RELABEL — re-measure on a relabeled rebuild. The rebuild lives in the
///     loader (`LiveComplex::relabel`), not in a reader: this harness loads the
///     relabeled live complex, wraps it in a read-only `RegisterContext` with the
///     register's images matched by permuted vertex set, and maps any
///     vertex-id-bearing provenance through the permutation
///     (`RegisterObservable::recordRelabeled`).
///
/// `selfTest` checks that the harness actually compares: a deliberately
/// label-dependent probe must be flagged by RELABEL and a deliberately
/// gauge-dependent probe by GAUGE.
class ObservableGates {
  public:
    /// GAUGE-gate angle: an incommensurate fraction of 2π, so the rotated target
    /// never lands on a symmetry of the singlet by accident.
    static constexpr double GAUGE_THETA =
        2.0 * 3.14159265358979323846 * 0.371;
    /// RELABEL-gate permutation seed.
    static constexpr std::uint64_t GATE_SEED = 3;

    /// One observable's gate verdicts.
    struct GateResult {
      double gaugeDelta = 0.0;
      double relabelDelta = 0.0;
      double gateTol = 0.0;
      bool gaugeOk = false;
      bool relabelOk = false;
    };

    /// The GAUGE residual: `reportDelta(record(ctx), record(ctx.gauged(θ)))`.
    [[nodiscard]] static double gaugeDelta(const RegisterObservable &observable,
                                           const RegisterContext &ctx);
    /// The RELABEL residual: `reportDelta(record(ctx),
    /// recordRelabeled(relabeled ctx, perm))`.
    [[nodiscard]] static double relabelDelta(
        const RegisterObservable &observable, const RegisterContext &ctx);
    /// Both gates + the `*_ok` verdicts against the observable's `gateTol`.
    [[nodiscard]] static GateResult evaluate(
        const RegisterObservable &observable, const RegisterContext &ctx);

    /// The harness self-test: the label-dependent probe must be RELABEL-flagged
    /// and the gauge-dependent probe GAUGE-flagged (both deltas > 0). Returns
    /// true iff both are flagged.
    [[nodiscard]] static bool selfTest(const RegisterContext &ctx);
};

/// Deliberately label-dependent probe used by the harness self-test: its record
/// leaks the sum of the selected holes' vertex ids, so the RELABEL gate must
/// flag it.
class LabelLeakProbe : public RegisterObservable {
  public:
    static constexpr std::string_view kRecordKey = "label_leak_probe";
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

/// Deliberately gauge-dependent probe: its record leaks the raw register
/// target's first-component phase, so the GAUGE gate must flag it.
class GaugeLeakProbe : public RegisterObservable {
  public:
    static constexpr std::string_view kRecordKey = "gauge_leak_probe";
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_OBSERVABLE_GATES_H
