// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_INTERIOR_HINGES_H
#define TESSERA_OBSERVABLES_INTERIOR_HINGES_H

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::spacetime;

/// # InteriorHinges
///
/// Shared 4D mass/radius reader core, composed by `EmergentMass` and
/// `EmergentRadius`. Everything here is a post-hoc reader; nothing shapes the
/// lattice.
///
///   * Hinges are triangles. On a `d = 4` complex the Regge hinges are the
///     `(d-2) = 2`-simplices, and curvature is the complex Lorentzian deficit
///     angle.
///   * Closed fans only. A triangle carries curvature only if every tetrahedron
///     of its coface fan is shared by exactly two 4-cells. Open-fan triangles
///     (the \f$ \partial W \f$ boundary, the register-hole walls) are boundary
///     artefacts near \f$ 2\pi \f$; they are excluded and counted in the census.
///     The fan test is combinatorial over the current top cells, so
///     Pachner-orphaned sub-simplices cannot pollute the selection. The readings
///     come off the canonical registered triangle `Simplex`, which requires the
///     skeleton the `RegisterContext` constructor materializes.
///   * Signature-aware readings. Dual volumes are the circumcentric signed
///     `Simplex::dualVolume`; the deficit is the complex `Simplex::deficitAngle`.
///     Masses use \f$ \mathrm{Re}\,\varepsilon \f$;
///     \f$ |\mathrm{Im}\,\varepsilon| \f$ (boost content) is always reported.
///   * Dimension-correct radius. \f$ r = V^{1/4} \f$ on a 4-complex; the root
///     tracks the top dimension.
///
/// Constructed once per (spacetime, holes). The constructor throws
/// `std::invalid_argument` if the complex is not genuinely 4D (5-vertex top
/// cells): on a `d`-complex the hinge dimension and the radius root both track
/// `d`, so a mismatched reader refuses rather than reading the wrong
/// quantity.
class InteriorHinges {
  public:
    /// One interior (closed-fan) triangle hinge and its curvature.
    struct Hinge {
      std::vector<std::uint64_t> vids;  ///< the 3 sorted vertex ids
      double re = 0.0;                  ///< Re of the complex deficit
      double im = 0.0;                  ///< Im of the complex deficit (boost)
      double dv = 0.0;                  ///< signed circumcentric dual content
      std::optional<int> shell;         ///< breadth-first distance from the
                                        ///< holes (empty when no holes given)
    };

    /// The interior/boundary hinge census (reported with every reading).
    struct Census {
      int nTops = 0;
      int nTets = 0;
      int nHingesTotal = 0;
      int nHingesInterior = 0;
      int nHingesBoundary = 0;
      int nBoundaryTets = 0;
      int nHoleVertices = 0;
      std::vector<std::vector<std::uint64_t>> boundaryTets;  ///< vertex-id sets
    };

    /// Three mass readings — one intensive, two extensive — plus the per-shell
    /// means and the imaginary-part accounting.
    struct Masses {
      double mShell = 0.0;   ///< intensive: sum over breadth-first shells of the
                             ///< shell-mean real deficit (plain mean with no
                             ///< holes)
      double mSum = 0.0;     ///< extensive: Σ Re ε
      double mAction = 0.0;  ///< extensive: Σ |★h|·Re ε
      /// per-shell mean real deficit, ordered shell-ascending with the unshelled
      /// bin last (`std::nullopt`).
      std::vector<std::pair<std::optional<int>, double>> shellMeans;
      double maxAbsIm = 0.0;
      int nImNonzero = 0;
      bool empty = true;  ///< no interior hinges — every scalar is NaN
    };

    /// The emergent size, dual and primal.
    struct Radii {
      double vDual = 0.0;    ///< Σ |★v| over strictly interior vertices
      double vPrimal = 0.0;  ///< Σ |V₄| over all top 4-cells
      int nInteriorVertices = 0;
      double rDual = 0.0;    ///< V_dual^{1/4} (NaN when V_dual ≤ 0)
      double rPrimal = 0.0;  ///< V_primal^{1/4} (NaN when V_primal ≤ 0)
    };

    /// Per-shell curvature profile entry.
    struct ShellProfile {
      int n = 0;
      double meanRe = 0.0;
      double weightShare = 0.0;
    };

    /// Is the curvature a localized lump or spread out?
    struct Localization {
      double pr = 0.0;             ///< participation ratio of |Re ε·★h| in (0,1]
      double concentration = 0.0;  ///< 1/PR
      double meanRe = 0.0;
      double stdRe = 0.0;
      double stdOverMean = 0.0;
      /// per breadth-first shell; empty unless every hinge is shelled.
      std::vector<std::pair<int, ShellProfile>> shellProfile;
      double rmsShellRadius = 0.0;
      double fracWithinShell1 = 0.0;
      bool empty = true;
    };

    /// One r·m combination (`"{r_name} x {m_name}"` -> product).
    struct RmTable {
      std::vector<std::pair<std::string, double>> combos;  ///< 6 entries
      double spreadMin = 0.0;
      double spreadMax = 0.0;
      double physical = 0.0;  ///< the physical anchor m_p·r_p/ħc ≈ 4.0
    };

    /// Physical anchor: m_p·r_p/ħc = 938 MeV · 0.84 fm / 197 MeV·fm ≈ 4.0.
    static constexpr double PHYSICAL_RM = 938.0 * 0.84 / 197.0;
    /// |Im ε| above this counts as genuinely complex (boost content).
    static constexpr double IM_TOL = 1e-12;

    /// Select the interior closed-fan triangle hinges of the 4-complex and read
    /// their curvature. `holes` are the emergent holes' vertex-id tuples, used as
    /// the breadth-first shell seeds; if empty, every hinge reports no shell.
    ///
    /// The spacetime is held `const`, so the compiler enforces that this reader
    /// cannot mutate build or skeleton state: it touches only the `const` query
    /// surface (`getTopSimplices`, `getBoundary`, `getSimplices` and the `const`
    /// geometry methods on the simplices).
    ///
    /// @throws std::invalid_argument if the complex has no top cells or its top
    ///   cells are not all 5-vertex (genuinely 4D).
    /// @throws std::runtime_error if an interior triangle has no registered
    ///   `Simplex`, i.e. the C++ skeleton was not materialized.
    InteriorHinges(std::shared_ptr<const Spacetime> spacetime,
                   std::vector<std::vector<std::uint64_t>> holes);

    [[nodiscard]] const std::vector<Hinge> &hinges() const noexcept {
      return hinges_;
    }
    [[nodiscard]] const Census &census() const noexcept { return census_; }

    /// The three mass readings over the interior hinges.
    [[nodiscard]] Masses masses() const;
    /// The dual/primal size of the interior.
    [[nodiscard]] Radii radii() const;
    /// The curvature localization.
    [[nodiscard]] Localization localization() const;
    /// Every r·m combination (3 masses by 2 radii). r·m is definition-sensitive,
    /// so the spread across definitions is stated before any single value.
    [[nodiscard]] RmTable rmTable(const Masses &mass, const Radii &rad) const;

  private:
    std::shared_ptr<const Spacetime> spacetime_;
    std::vector<std::vector<std::uint64_t>> holes_;
    std::vector<Hinge> hinges_;
    Census census_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_INTERIOR_HINGES_H
