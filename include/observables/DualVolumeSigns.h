// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_DUALVOLUMESIGNS_H
#define TESSERA_OBSERVABLES_DUALVOLUMESIGNS_H

#include <memory>
#include <vector>

#include "observables/Observable.h"

// === cross-subsystem fwd-decls ===
namespace tessera::mesh { class Simplex; }
namespace tessera::spacetime { class Spacetime; }

namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;

/// # DualVolumeSigns
///
/// Read-only audit of the sign of the diagonal discrete exterior calculus (DEC)
/// Hodge star on a triangulation. It measures only: it changes no geometry, adds
/// no dynamics, and enforces nothing.
///
/// Reference: Desbrun, Hirani, Leok, Marsden, "Discrete Exterior Calculus",
/// arXiv:math/0508341.
///
/// ## The measured quantity
///
/// The diagonal DEC Hodge star assigns each \f$ k \f$-simplex
/// \f$ \sigma \f$ the scalar ratio \f$ |\star\sigma| / |\sigma| \f$, where
/// \f$ |\sigma| \f$ is the simplex's own signed content (``Simplex::volume``)
/// and \f$ |\star\sigma| \f$ its signed circumcentric dual cell content
/// (``Simplex::dualVolume``). It is diagonal — one number per simplex rather
/// than a dense matrix — only for the circumcentric (Voronoi) dual; a
/// barycentric dual yields the dense Whitney (Galerkin) star instead.
///
/// A Maxwell-type or gauge term discretised with DEC carries its whole metric
/// dependence in that ratio, as \f$ \sum_\sigma (|\star\sigma| /
/// |\sigma|)\,F_\sigma^2 \f$. A negative ratio costs positive-definiteness of
/// the associated Hodge Laplacian and breaks the sign structure that a
/// self-dual / anti-self-dual split of a 2-cochain relies on, the discrete
/// analogue of \f$ \star^2 = -1 \f$ on 2-forms, which is what makes a helicity
/// or circular-polarisation readout meaningful. Whether the ratio does go
/// negative on the complexes this project grows is an empirical question, and
/// this class answers it.
///
/// ## Two reasons the ratio can go negative
///
/// The audit separates them, because they have opposite implications.
///
///   * Mesh quality. A simplex is well-centered when its circumcenter lies in
///     its interior. ``Simplex::circumcenterBarycentric`` returns the
///     circumcenter in barycentric coordinates, and a coordinate is negative
///     exactly when the circumcenter falls outside the simplex on that vertex's
///     side. This is the Riemannian failure mode — a well-centeredness
///     violation — and indicates badly shaped cells from surgery or relaxation.
///   * Signature. This project relaxes along the real signed-\f$ \ell^2 \f$
///     manifold, so an edge may be spacelike, timelike, or null, and
///     ``Simplex::circumradiusSquared`` goes negative when the
///     circumcenter-to-vertex displacement is timelike. Lorentzian signature has
///     no positive-definite circumradius, so "circumcenter inside the simplex"
///     is not the criterion it is in Riemannian signature. A negative ratio on a
///     cell carrying timelike or null edges is signature-driven, not a defect.
///
/// Every count is therefore also broken out by whether the simplex is
/// all-spacelike (every edge has \f$ \ell^2 > 0 \f$) or mixed-signature (at
/// least one timelike or null edge). Negatives concentrated in the
/// mixed-signature population mean the diagonal star behaves as Lorentzian
/// signature requires; negatives among all-spacelike cells mean mesh
/// degradation.
///
/// ## Scope
///
/// Only simplices that are genuine faces of the current complex, i.e. those for
/// which ``Simplex::hasTopCoface`` is true. A Pachner move that removes a cell
/// can leave a lazily materialised sub-face registered with no surviving top
/// coface. Such an orphan is not part of the complex and does not enter the
/// statistics.
class DualVolumeSigns : public Observable {
  public:
    /// Per-dimension audit counts. Every ``n*`` field is a simplex count; the
    /// fractions are left to the caller so the raw counts stay inspectable.
    struct DimensionReport {
      /// Simplex dimension \f$ k \f$ these counts describe.
      int dimension{0};
      /// Non-orphan \f$ k \f$-simplices audited.
      int nSimplices{0};

      /// Simplices whose signed dual cell content is strictly negative.
      int nNegativeDualVolume{0};
      /// Simplices whose own signed content is within ``tolerance`` of zero. The
      /// star ratio is undefined for these, so they are excluded from the ratio
      /// statistics and from ``nNegativeStar``.
      int nDegenerateVolume{0};
      /// Simplices whose circumcenter falls outside the simplex, detected as a
      /// negative barycentric coordinate. The Riemannian well-centeredness
      /// violation.
      int nCircumcenterOutside{0};
      /// Simplices with negative signed circumradius squared: a timelike
      /// circumcenter displacement, possible only in Lorentzian signature.
      int nNegativeCircumradius{0};
      /// Simplices whose diagonal Hodge star ratio is strictly negative. This
      /// count decides whether the star is indefinite.
      int nNegativeStar{0};

      /// Simplices all of whose edges are spacelike.
      int nAllSpacelike{0};
      /// Of the all-spacelike simplices, those with a negative star ratio. A
      /// nonzero value here indicates genuine mesh degradation.
      int nNegativeStarAllSpacelike{0};
      /// Simplices carrying at least one timelike or null edge.
      int nMixedSignature{0};
      /// Of the mixed-signature simplices, those with a negative star ratio. These
      /// are expected on signature grounds.
      int nNegativeStarMixedSignature{0};

      /// Extremes and mean of the star ratio over the non-degenerate simplices of
      /// this dimension. Left at zero when no such simplex exists.
      double minStarRatio{0.0};
      double maxStarRatio{0.0};
      double meanStarRatio{0.0};
    };

    /// The full audit: one entry per simplex dimension present in the complex,
    /// ordered by increasing dimension.
    struct Report {
      std::vector<DimensionReport> dimensions;
      /// Combined non-orphan, non-degenerate simplex count across dimensions.
      int nSimplices{0};
      /// Combined negative-star count across dimensions.
      int nNegativeStar{0};
    };

    /// @param tolerance Magnitude below which a signed content counts as
    ///        degenerate rather than signed. Degenerate simplices are reported
    ///        separately and excluded from the ratio statistics, so the mean is
    ///        never contaminated by a division by an almost-zero volume.
    explicit DualVolumeSigns(double tolerance = 1e-12) : tolerance_(tolerance) {}

    /// The full per-dimension audit.
    [[nodiscard]] Report analyze(
        const std::shared_ptr<Spacetime> &spacetime) const;

    /// Headline scalar: the fraction of audited, non-degenerate simplices whose
    /// diagonal Hodge star ratio is negative, across all dimensions. Zero means
    /// the diagonal star is positive everywhere on this complex and is safe to
    /// use as-is; any positive value means it is indefinite. Returns zero for an
    /// empty complex.
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;

  private:
    /// True when every edge of `simplex` is spacelike. A simplex with no
    /// registered edges (a vertex) is treated as all-spacelike, since it carries
    /// no signature information of its own.
    [[nodiscard]] static bool isAllSpacelike(const Simplex &simplex);

    double tolerance_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_DUALVOLUMESIGNS_H
