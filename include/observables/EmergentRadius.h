// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_EMERGENT_RADIUS_H
#define TESSERA_OBSERVABLES_EMERGENT_RADIUS_H

#include <string>

#include "observables/InteriorHinges.h"
#include "observables/RegisterObservable.h"

namespace tessera::observables {

/// # EmergentRadius
///
/// Radius half of the mass/radius battery on the relaxed 4D interior. Shares the
/// `InteriorHinges` core (via `RegisterContext::interiorHinges`) with
/// `EmergentMass`.
///
///   * headline (`compute`) = \f$ r_{\mathrm{dual}} = V_{\mathrm{dual}}^{1/4}
///     \f$, the dimension-correct dual-volume radius on a 4-complex;
///   * `radii()`: `V_dual` / `V_primal`, the primal cross-check `r_primal`, and
///     the strictly-interior-vertex count;
///   * `record()`: the radius block (dual and primal) and the hole count.
class EmergentRadius : public RegisterObservable {
  public:
    [[nodiscard]] std::string recordKey() const override {
      return std::string(kRecordKey);
    }
    /// Volume aggregates re-summed in a relabeled container order carry
    /// rounding noise of a few units in the last place, scaling with the
    /// magnitudes (see `EmergentMass`).
    [[nodiscard]] double gateTol() const override { return 1e-6; }
    [[nodiscard]] int requiredDimensions() const override { return 4; }
    [[nodiscard]] Record record(const RegisterContext &ctx) const override;

    /// The dual/primal size of the interior + interior-vertex count.
    [[nodiscard]] InteriorHinges::Radii radii(const RegisterContext &ctx) const;

    static constexpr std::string_view kRecordKey = "emergent_radius";

  protected:
    [[nodiscard]] double computeHeadline(
        const RegisterContext &ctx) const override;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_EMERGENT_RADIUS_H
