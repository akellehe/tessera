// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.
//
// The register carried by a certified cluster: a recursive spectral fiber.
//
// Within component C, choose an isolated localized spectral band and, in the
// positive self-adjoint regime, a weighted orthonormal frame
// Phi_C = (phi_1, ..., phi_r) with Phi_C^dagger W_C Phi_C = I_r. The derived
// fiber is E_C = Ran Phi_C.
//
// The fiber need not be a harmonic space and so need not be supported by a
// hole. It does require a spectral gap, localization and persistence: a
// candidate component is accepted only if all of the following remain stable
// across a stated range of scales.
//
//   - a persistent connected cluster support, however proposed;
//   - a localized spectral projector with stable rank;
//   - a nonzero band gap separating it from discarded modes;
//   - overlap with its predecessor and successor components;
//   - lifetime across multiple cobordism frames; and
//   - small external transport leakage.
//
// This file assembles that from the existing observables and derives no
// spectrum, transport or clustering of its own. Bands and their gap,
// localization and regime certificates come from `SpectralFiber`; the support
// and its frame lifetime and overlap come from `PersistentModularity`; the
// leakage comes from `FiberConnection`.
//
// Nothing here is target-conditioned: there is no target vector, residual or
// objective term. The register is read from a relaxed geometry after the fact,
// is not reachable from an emergence objective, and does not tell the geometry
// what to become.
//
// No hole is required or consulted. A hole names the boundary cycle of a
// removed top cell, and the removal is what makes that cycle non-bounding. A
// cluster inside a filled complex supplies only cycles that bound, and a
// harmonic form has vanishing period over a bounding cycle, so a period readout
// would report zero on a cluster regardless of the geometry. That is why the
// fiber is a frame range rather than a period. The hole machinery is untouched
// by this file and is not consulted as a fallback.

#ifndef TESSERA_OBSERVABLES_CLUSTERREGISTER_H
#define TESSERA_OBSERVABLES_CLUSTERREGISTER_H

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cobordism/Certificate.h"
#include "observables/FiberConnection.h"
#include "observables/PersistentModularity.h"
#include "observables/Record.h"
#include "observables/SpectralFiber.h"

namespace tessera::spacetime {
  class Spacetime;
}

namespace tessera::observables {

/// The six fiber-acceptance conjuncts, named.
///
/// Named constants rather than literals at each site: a conjunct name is
/// written where it is decided and compared where it is consumed, and a
/// misspelling in either place would compile but produce a name no consumer
/// matches. Bound to Python so a caller references the constant instead of
/// retyping the string.
struct RegisterConjunct {
  /// Persistent connected cluster support: the support is non-empty and its
  /// induced one-skeleton is connected. How the support was proposed is never
  /// consulted; any proposer is admissible and none may veto.
  static constexpr const char *kClusterSupport = "cluster-support";
  /// Localized spectral projector with stable rank: the band's localization
  /// excess is within the detector's declared cap. Decided by `SpectralFiber`'s
  /// own certificate and re-read here rather than re-derived.
  static constexpr const char *kLocalizedProjector = "localized-projector";
  /// Nonzero band gap separating the band from the discarded modes: its
  /// separation from the nearest discarded eigenvalue in the complex plane.
  static constexpr const char *kBandGap = "band-gap";
  /// Overlap with predecessor and successor components: the smallest
  /// adjacent-frame support overlap along the component's frame track.
  static constexpr const char *kNeighbourOverlap = "neighbour-overlap";
  /// Lifetime across multiple cobordism frames: the number of consecutive
  /// frames the support was tracked through.
  static constexpr const char *kFrameLifetime = "frame-lifetime";
  /// Small external transport leakage: the largest leakage over the supplied
  /// external transports.
  static constexpr const char *kTransportLeakage = "transport-leakage";
};

/// Why a conjunct could not be decided, as distinct from being decided against.
///
/// A conjunct that was measured and fell short is a failure; a conjunct with no
/// evidence behind it is unmeasured. Both block acceptance and the two are never
/// merged: an unmeasured quantity is not a failed one, and neither is encoded as
/// a zero that would claim a measurement nobody made.
struct RegisterUnmeasured {
  /// No band was supplied, so nothing spectral could be decided.
  static constexpr const char *kNoBand = "no-band";
  /// The band carries no localization measurement (a NaN excess).
  static constexpr const char *kLocalizationUnmeasured = "localization-unmeasured";
  /// The band's separation from the discarded modes is unknown: the truncated
  /// sparse top leaves one side uncovered.
  static constexpr const char *kBandGapUnknown = "band-gap-unknown";
  /// No frame track was supplied, so the lifetime was never measured. The
  /// certificate fails by name rather than falling back to a single-frame test
  /// that would pass vacuously.
  static constexpr const char *kNoFrameTrack = "no-frame-track";
  /// No external transport was supplied, so leakage was never measured. Absence
  /// of transports is not evidence of small leakage.
  static constexpr const char *kNoTransport = "no-transport";
  /// The complex could not be read to decide support connectivity.
  static constexpr const char *kSupportUnreadable = "support-unreadable";
};

/// Thresholds the register is accepted under. Each is an analysis parameter
/// recorded on the read; none of them selects which bands or clusters exist.
struct ClusterRegisterConfig {
  /// Floor on the smallest adjacent-frame support overlap
  /// (`FrameTrack::minAdjacentOverlap`).
  double minNeighbourOverlap = 0.5;
  /// Floor on the cobordism-frame lifetime (`FrameTrack::frames`). The default
  /// is two, since one frame is not a lifetime.
  std::size_t minFrameLifetime = 2;
  /// Cap on the largest external transport leakage.
  double maxTransportLeakage = 1e-6;
};

/// What is reported of the band's metric regime.
///
/// In a Hermitian indefinite regime, the inertia of Phi_C^dagger W_C Phi_C is
/// recorded and normalized to a signature matrix J_C = diag(I_p, -I_q). In a
/// non-normal regime, matched right and left frames Phi_C, Psi_C with
/// Psi_C^dagger W_C Phi_C = I are used, and both residuals and the frame
/// condition number are reported.
///
/// Quantities the regime does not define are NaN, never zero.
struct RegisterRegimeReport {
  /// The verified regime of the band's solve; never assumed.
  cobordism::CertificateRegime regime =
      cobordism::CertificateRegime::NonNormal;
  /// The weighted Gram (signature) defect ||Phi^dagger W Phi - J||. In the
  /// positive regime J = I, and this is the orthonormality condition
  /// Phi_C^dagger W_C Phi_C = I_r.
  double gramDefect = std::numeric_limits<double>::quiet_NaN();
  /// Krein inertia (p, q) of Phi^dagger W Phi, so the normalized signature is
  /// J_C = diag(I_p, -I_q). Reported in every regime (p = rank, q = 0 in the
  /// positive one). A negative signature is a certificate, not an
  /// identification with an antiparticle.
  int positiveSignature = 0;
  int negativeSignature = 0;
  /// Neutral directions rank - p - q, nonzero only when the W-Gram is singular.
  int neutralSignature = 0;
  /// Whether the inertia is normalizable, i.e. has no neutral directions, so
  /// J_C = diag(I_p, -I_q) exists.
  bool signatureNormalizable = false;
  /// Right-frame residual ||L Phi - Phi Lambda|| / ||L||.
  double eigenResidual = std::numeric_limits<double>::quiet_NaN();
  /// Left-frame residual. Equal to `eigenResidual` on the self-adjoint path,
  /// where the frames coincide.
  double leftResidual = std::numeric_limits<double>::quiet_NaN();
  /// The frame condition number, reported in the non-normal regime.
  double frameConditionNumber = std::numeric_limits<double>::quiet_NaN();
};

/// # ClusterRegisterRead
///
/// One register read: the cluster it is carried by, the fiber it is, the six
/// conjuncts as measured, and the verdict.
///
/// The fiber is `E_C = Ran Phi_C`, carried here as the band whose right frame
/// is `Phi_C` (`band.rightFrame()`). The range is represented by the band
/// projector rather than by any individual eigenvector, since a choice of
/// in-band basis is a gauge choice and does not determine an identity.
///
/// Unmeasured values are NaN and unmeasured conjuncts are named; nothing is
/// zero-filled and no worst-case value stands in for a measurement.
struct ClusterRegisterRead {
  /// The cluster's label-free identity, when the caller supplied one.
  ComponentId component{};
  /// The cluster's level-0 vertex support, as supplied. How it was proposed is
  /// not recorded: it plays no part in acceptance.
  std::vector<std::uint64_t> support{};
  /// Form degree of the band.
  int degree = 0;
  /// Band rank r, the fiber's dimension. Reported as measured; this read never
  /// requests or requires a particular rank.
  std::size_t rank = 0;
  /// The band carrying the fiber (its right frame is `Phi_C`).
  SpectralFiber band{};

  // ── the six conjuncts, as measured ────────────────────────────────
  /// Whether the support is non-empty and its induced one-skeleton is
  /// connected.
  bool supportConnected = false;
  /// Number of connected pieces the support induces (1 when connected).
  std::size_t supportPieces = 0;
  /// The band's localization excess (0 = as concentrated as the rank
  /// permits, 1 = perfectly delocalized).
  double localizationExcess = std::numeric_limits<double>::quiet_NaN();
  /// The band's separation from the nearest discarded eigenvalue.
  double bandGap = std::numeric_limits<double>::quiet_NaN();
  /// Smallest adjacent-frame support overlap along the frame track.
  double neighbourOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Consecutive cobordism frames the support was tracked through.
  double frameLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Largest leakage over the supplied external transports.
  double transportLeakage = std::numeric_limits<double>::quiet_NaN();

  /// The regime report.
  RegisterRegimeReport regime{};

  /// Conjuncts that were measured and fell short, by name.
  std::vector<std::string> failedConjuncts{};
  /// Conjuncts that could not be measured at all, by name (see
  /// :class:`RegisterUnmeasured`). Distinct from a failure: absence of evidence
  /// is not evidence of failure, and neither is a zero.
  std::vector<std::string> unmeasured{};

  /// Accepted exactly when all six conjuncts were measured and met, and the
  /// band's own certificate holds.
  bool accepted = false;

  /// The graded claim: `CertifiedNumerical` when accepted, otherwise the
  /// uncertified `HeuristicDiscovery`, which never `holds()`.
  cobordism::Certificate certificate{};

  /// The thresholds this read was decided under.
  ClusterRegisterConfig thresholds{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static ClusterRegisterRead fromRecord(const Record &record);
};

/// # ClusterRegister
///
/// Reads the recursive spectral fiber register: the fiber `E_C = Ran Phi_C` of
/// an isolated localized band on a persistent cluster, accepted under the
/// six-conjunct list.
///
/// Reference: Lim, "Hodge Laplacians on graphs", arXiv:1507.05379.
///
///   * Assembles, never re-derives. Bands, their gap and localization measures
///     and their regime certificates come from `SpectralFiber`; the frame track
///     comes from `PersistentModularity::trackAcrossFrames`; leakage comes from
///     `FiberConnection`. This class decides the conjunction and reports it.
///   * However proposed. The support arrives as a plain vertex-id list and its
///     provenance is never consulted. A modularity read cannot veto a fiber that
///     meets the six conjuncts, because no proposer is an input to the decision.
///   * No color claim. The anchoring certificate is required only when a color
///     interpretation is claimed; a register is not such a claim, so no anchor
///     is required or consulted here. A caller wanting a color reading takes it
///     from the quark classifier, which gates on the anchor.
///   * Read-only. Never solves on, and never mutates, the spacetime. Nothing
///     here enters any emergence objective.
class ClusterRegister {
  public:
    /// Bind the acceptance thresholds.
    explicit ClusterRegister(ClusterRegisterConfig cfg = {});

    [[nodiscard]] const ClusterRegisterConfig &config() const noexcept {
      return cfg_;
    }

    /// Read the register of one cluster.
    ///
    /// `support` is the cluster's vertex-id set, however proposed. `band` is
    /// the isolated localized band whose right frame is `Phi_C`. `track`
    /// supplies the cobordism-frame lifetime and the adjacent-frame overlap; if
    /// absent, those two conjuncts are unmeasured rather than satisfied.
    /// `externalTransports` are the transports leaving the cluster; an empty
    /// list likewise leaves leakage unmeasured rather than small.
    ///
    /// `st` is read only to decide support connectivity.
    [[nodiscard]] ClusterRegisterRead read(
        const std::shared_ptr<Spacetime> &st,
        const std::vector<std::uint64_t> &support, const SpectralFiber &band,
        const std::optional<FrameTrack> &track,
        const std::vector<FiberTransportRead> &externalTransports,
        ComponentId component = {}) const;

    /// Whether the induced one-skeleton on `support` is connected, and in how
    /// many pieces. Returns `{false, 0}` when the support is empty or the
    /// complex cannot be read.
    [[nodiscard]] static std::pair<bool, std::size_t> supportConnectivity(
        const std::shared_ptr<Spacetime> &st,
        const std::vector<std::uint64_t> &support);

  private:
    ClusterRegisterConfig cfg_{};
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_CLUSTERREGISTER_H
