// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_EMERGENT_MASS_H
#define TESSERA_OBSERVABLES_EMERGENT_MASS_H

#include <string>

#include "observables/InteriorHinges.h"
#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # EmergentMass
///
/// Mass half of the mass/radius battery on the relaxed 4D interior. Shares the
/// `InteriorHinges` core (via `RegisterContext::interiorHinges`) with
/// `EmergentRadius`, so both read exactly one hinge selection.
///
///   * headline (`compute`) = `m_shell`, the intensive shell mass;
///   * `masses()`: `m_shell` / `m_sum` / `m_action`, the per-shell means, and the
///     \f$ |\mathrm{Im}\,\varepsilon| \f$ boost accounting;
///     `localization()`: the curvature localization;
///   * `record()`: the interior census, the three masses, the localization, and
///     the r·m table. r·m is definition-sensitive, so its spread across
///     definitions is stated before any single value.
class EmergentMass : public RegisterObservable {
  public:
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    /// Geometric aggregates (sums over hundreds of hinges, r·m products) are
    /// re-summed in a different container order on the relabeled rebuild. The
    /// resulting rounding noise scales with the magnitudes, so this tolerance is
    /// loose in absolute terms; the raw residuals are still reported.
    [[nodiscard]] double gateTol() const override { return 1e-6; }
    [[nodiscard]] int requiredDimensions() const override { return 4; }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

    /// The three mass readings + shell means + Im accounting.
    [[nodiscard]] InteriorHinges::Masses masses(
        const RegisterContext &ctx) const;
    /// The curvature localization (participation ratio, shell profile).
    [[nodiscard]] InteriorHinges::Localization localization(
        const RegisterContext &ctx) const;

    static constexpr std::string_view kRecordKey = "emergent_mass";

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_EMERGENT_MASS_H
