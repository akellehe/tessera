// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SINGLET_RESIDUAL_H
#define TESSERA_OBSERVABLES_SINGLET_RESIDUAL_H

#include <string>

#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # SingletResidual
///
/// Whole-complex singlet diagnostic: the relabeling-invariant singlet residual
/// `r_state` of `ProtonSynthesis::singlet()` against the whole complex's \f$ L_k \f$
/// harmonic form (`≈ 0` means the singlet is carried), plus the hole/Betti
/// census with the `holes_vs_b3_divergent` flag. Diagnostic only; it never
/// steers the simulation.
///
/// The headline (`compute`) is the singlet residual. `conjugateResidual` is the
/// companion read against the conjugate singlet `[1, ω̄, ω̄²]` (the antibaryon
/// channel). The record scores the true singlet regardless of the register
/// target, so the GAUGE gate (which rotates the target) is trivially
/// satisfied.
class SingletResidual : public RegisterObservable {
  public:
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

    /// `r_state` of the conjugate singlet `[1, ω̄, ω̄²]` against the whole.
    [[nodiscard]] double conjugateResidual(const RegisterContext &ctx) const;

    static constexpr std::string_view kRecordKey = "singlet_residual";

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SINGLET_RESIDUAL_H
