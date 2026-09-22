// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_PARTICLECLUSTERS_H
#define TESSERA_OBSERVABLES_PARTICLECLUSTERS_H

// Particle classification from persistent modular spectral components.
//
// References: Newman, "Modularity and community structure in networks",
// arXiv:physics/0602124; Greensite, "The Confinement Problem in Lattice Gauge
// Theory", arXiv:hep-lat/0301023.
//
// ─── Contents ────────────────────────────────────────────────────────────
//
//   • ParticleClusters       — the classifier: quarks and antiquarks, the
//                              even-parity sector (octet bilinear read,
//                              gluon/meson/diquark candidates), and the
//                              three-cluster sector (bound-supercomponent
//                              search, color singlet, proton certificate).
//   • QuarkCandidateEvidence / QuarkRead
//                            — the per-candidate evidence bundle and the
//                              quark/antiquark verdict.
//   • FlavorDoubletRead      — the emergent transported two-state spectral
//                              subclass, searched without a requested
//                              dimension.
//   • GaussFluxRead          — electric flux consistency across nested
//                              enclosing surfaces.
//   • ConjugatePairRead      — pair conservation of a conjugate creation
//                              homotopy.
//   • OctetBilinearRead      — the quasi-free 1 ⊕ 8 read of three declared
//                              color modes of a covariance.
//   • GluonCandidateEvidence / GluonRead
//                            — the octet-excitation verdict.  "Candidate" is
//                              the strongest claim emitted; no even octet
//                              excitation is classified a physical gluon.
//   • CompositeCandidateEvidence / MesonRead / DiquarkRead
//                            — the two-cluster composite verdicts.
//   • BoundCandidateEvidence / BoundSupercomponentRead
//                            — the bound-supercomponent search.
//   • ScaleProfileSample / ScaleProfileRead
//                            — the refinement-window scale certificate.
//                              Nothing here is a form factor; see below.
//   • BaryonCandidateEvidence / BaryonRead
//                            — the three-cluster verdict and the proton
//                              certificate.
//
// Every field is a read produced by an upstream kernel; nothing is
// recomputed here.  Unknown or uncertified values are null throughout (an
// empty optional, NaN, or a 0 sign for the sign-valued ints), never
// zero-filled, and every gap is named in the read's `failedCertificates`.
//
// ─── Exact identities and their domains ──────────────────────────────────
//
//   • The verdict is an exact boolean combination of consumed certificates
//     (a StructureExact claim; the reported residual is their maximum).
//   • Exterior parity and total occupation are Wick reads on the candidate's
//     carried quasi-free state: ⟨(−1)^N⟩ = det(I − 2Γ) and ⟨N⟩ = tr Γ,
//     algebraically exact on the covariance.  Domain: the caller's carried
//     CovarianceState for the component's modes.
//   • Baryon flux B = ν/3, with ν the certified determinant-line winding of
//     a closed full-rank gapped family, or of an open cobordism segment
//     closed by its recorded matched-reference or boundary-register
//     trivialization (`DeterminantWindingRead::windingClosure` travels on
//     the QuarkRead).  A raw endpoint phase is not evidence: with no
//     certified winding the flux is unknown (null), never zero.  The
//     quark/antiquark verdict also requires ν = ±1; a certified ν = 0 is a
//     certified zero flux (used by the gluon sector), not a quark.
//   • Quark vs antiquark is determinant-line orientation (the sign of ν;
//     reversing the world-tube reverses it and transports in the dual color
//     representation), never the color representation alone: the Λ²C³
//     anti-triplet of two quarks is excluded by its even parity and total
//     occupation two.
//   • The Gauss-flux electric read applies
//     `cobordism::EigenstateSynthesis::gaussLawCharge` to nested enclosing
//     surfaces (closed stars of nested vertex sets); the flux sum is
//     algebraically exact in the supplied field-strength 2-cochain, and the
//     consistency claim is the measured max deviation across surfaces.
//   • Q = I3 + B/2, the Gell-Mann–Nishijima relation on the accepted doublet
//     hypothesis, is tested rather than asserted: only when baryon flux, the
//     emergent flavor doublet and the Gauss read are independently certified,
//     and a pass is recorded as the proposed u/d identification
//     (`udIdentificationProposed`), not a charge definition.
//
// ─── Three-cluster (baryon / proton) identities ──────────────────────────
//
//   • Invariant color volume S_ABC = det[c_A c_B c_C] = ε_ijk c_A^i c_B^j
//     c_C^k (`ColorFiber::colorWedge`), invariant under a common g ∈ SU(3)
//     because det(gC) = det(g) det(C) = det(C).  Its squared magnitude is
//     the singlet Gram certificate |S_ABC|² = det(C†C) ∈ [0, 1]
//     (`ColorFiber::singletGram`): exactly one for an orthonormal anchored
//     triad, exactly zero for duplicated color modes.  Domain: three
//     normalized color columns.
//   • The wedge is built once.  Its antisymmetry is the ε tensor inside the
//     determinant, so no extra fermion permutation sign is multiplied onto
//     it.  Transposing two color columns flips `colorWedge` (det C) and
//     leaves det(C†C) = |det C|² invariant; the composite's fermionic
//     statistics come from the graded product of the three constituent
//     parities.
//   • Net color flux: the octet (traceless) Frobenius weight
//     ‖M − (Tr M/3) I‖_F² of the bound object's color bilinear
//     M_ij = ⟨a_i†a_j⟩ under the 1 ⊕ 8 split, from `octetBilinearRead`.
//     Zero octet weight means no net color polarization.  An independent
//     diagnostic on a finite complex, not by itself a proof of confinement.
//   • Composite baryon flux ν = ν_A + ν_B + ν_C over the three certified
//     determinant windings, each carrying its own closure specification,
//     with B = ν/3 — a certified proton reads ν = 3, B = +1.  An uncertified
//     leg leaves the total unknown (null), never zero; only integers already
//     carrying a closure specification are summed.
//   • Composite exterior parity: the graded product of the three certified
//     constituent parities (parity adds mod 2), odd for three odd clusters.
//     Unknown when any constituent parity is uncertified.
//   • Flavor and charge: the constituent reads are consumed verbatim.  The
//     `uud` pattern is the certified-isospin multiset {+1/2, +1/2, −1/2}
//     under each constituent's recorded doublet orientation, and the
//     electric flux is the sum of the three certified Gauss-consistent
//     constituent fluxes (2/3 + 2/3 − 1/3 = +1 for a proton).  No u/d label
//     is inserted.
//   • Sharp total-space spin: ⟨J²⟩ = Σ_α ⟨dΓ(J_α)²⟩ and
//     Var(J²) = ⟨(J²)²⟩ − ⟨J²⟩², both exact finite Wick sums on the
//     covariance (`CovarianceState::wickSpinSquaredExpectation` /
//     `wickSpinSquaredVariance`) — the total-space operator, not a product
//     of per-hole or per-edge spinors.  A candidate carried as an explicit
//     composite state instead supplies `ExchangeHolonomy::totalJSquared`,
//     which certifies ⟨J²⟩ and leaves Var(J²) unknown: expectation alone is
//     not a sharp-spin certificate.
//   • The reference-normalized 2π character is the `rotationCharacter` read
//     (channel `PhysicalRotation`), required at characterSign = −1.  The
//     SO(d) → Spin(d) lift is required only when the caller declares a
//     continuum spin claim.  The particle-exchange channel reports only:
//     `exchangeCharacter` and the doubly cancelled spin-statistics ratio
//     χ̂(exchange)·χ̂(2π)^{-1} travel on the read but gate nothing.  A read
//     tagged with the wrong channel is refused.
//   • Verdicts: "no-baryon" when a structural gate fails; "certified-proton"
//     when every certificate holds; "quasi-free-sharp-spin-obstruction" when
//     the only failure is `sharp-spin`, the candidate's own Var(J²) was
//     measured (an absent variance is unknown, not an obstruction), and the
//     accepted covariance-only class was swept with its Var(J²) floor above
//     tolerance; "baryon-candidate" otherwise.  The obstruction is a branch
//     point calling for an explicit non-Gaussian mechanism, not a refutation
//     of the geometry, and nothing here adds such a mechanism.
//
// ─── Form factors ────────────────────────────────────────────────────────
//
// Two different observables bear on a finite radius and stable
// crossing-mass readouts; they are not interchangeable.
//
// `CrossingReadouts` supplies the crossing readouts: the crossing mass
// m_x = kappa_m sum |pi_perp|, the coherent one-third baryon sum, and the
// spectral charge-power profile, each a difference against M0.  They reach
// the baryon verdict through `BaryonCandidateEvidence::crossingMass` /
// `crossingBaryon` and the `crossing-readouts` gate, applicable-gated like
// `spin-lift`: absent evidence passes vacuously, supplied evidence is
// enforced.  The electromagnetic form factor G_E stays unavailable — it
// needs a certified conserved U(1) current and certified momentum-transfer
// states, neither of which exists in this tree, so
// `ElectromagneticFormFactorRead` names those missing certificates and the
// charge radius is refused rather than inferred.  The spectral charge-power
// profile is an incoherent structure factor, never a substitute for G_E.
//
// The refinement certificate below consumes the mass-radius battery instead
// (`InteriorHinges` through `RegisterContext::interiorHinges`, exactly what
// `EmergentRadius` / `EmergentMass` read): the emergent radius
// r = V_dual^{1/4} (dimensionful, lattice units — only its finiteness is
// certified); the mean interior deficit angle m_shell (an angle, so
// dimensionless in lattice units) as the spectral-mass channel; the
// participation ratio of the curvature weight; and `radialWeightProfile`,
// the share of the |Re ε · ★h| curvature weight per breadth-first shell — a
// normalized radial curvature-weight density.
//
// `radialWeightProfile` is not a form factor: no charge radius is extracted
// from a slope at q² → 0, and the profiled density is curvature weight
// rather than electromagnetic charge.  Its certificate, `profile-stability`,
// certifies only a refinement-stable dimensionless radial profile.
//
// Any dimensionful mass stays unknown: `ScaleProfileRead::physicalMass` and
// `BaryonRead::physicalMass` are always empty, because converting a
// deficit-angle reading into a physical mass needs a physical scale this
// program has not independently established.  The crossing mass is likewise
// uncalibrated by default (`CrossingReadoutsConfig::kappaMass` ships at 1.0
// with `massCalibrated` false), so `BaryonRead::crossingMassValue` is a
// ratio-only quantity and not a physical mass.
//
// ─── Thresholds and boundaries ───────────────────────────────────────────
//
// "Exact" is exact where algebraic; every acceptance threshold is an
// analysis parameter (`ParticleClustersConfig`), echoed verbatim on every
// read (`QuarkRead::thresholds`) so a checkpoint carries the configuration
// that produced its verdicts.
//
// Read-only observable: it consumes caller-assembled reads, never calls a
// solver and never mutates the spacetime (the Gauss adapter constructs a
// degree-2 EigenstateSynthesis solely for its read-only charge and curvature
// entry points).  No "quark = hole", no hard-coded u/d labels, no baryon
// number without determinant-winding evidence, and no rank-three band is
// called a quark without its anchor, parity, persistence and leakage
// certificates.  No output, and no quantity derived from one, enters any
// emergence objective.

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "observables/ColorFiber.h"
#include "observables/CrossingReadouts.h"
#include "observables/EffectiveTopology.h"
#include "observables/ExchangeHolonomy.h"
#include "observables/FiberConnection.h"
#include "observables/PersistentModularity.h"
#include "observables/Record.h"
#include "observables/SpectralFiber.h"
#include "quantum/CovarianceState.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::cobordism {
  class AnalyticCache;
}
namespace tessera::observables {

class RegisterContext;  // the validated read context (RegisterContext.h)

/// Analysis thresholds of the particle classification.  Every value selects
/// which reads are certified, never which value is reported, and the whole
/// configuration is echoed on every read (`QuarkRead::thresholds`).
struct ParticleClustersConfig {
  /// |⟨(−1)^N⟩ ∓ 1| cap for a definite exterior-parity sign.  The Wick
  /// parity is exact on the covariance; this absorbs rounding.
  double parityTolerance = 1e-9;
  /// |⟨N⟩ − 1| cap for the single-fermion occupation certificate (the
  /// total-occupation channel distinguishing an antiquark from the
  /// two-quark anti-triplet).
  double occupationTolerance = 1e-9;
  /// Calibrated triangle-anchor atlas-score floor (a² ∈ [0, 1]).
  double minAnchorScore = 0.5;
  /// Determinant-phase coherence floor of the anchor profile (∈ [0, 1]).
  double minPhaseCoherence = 0.5;
  /// Cap on the worst lifetime transport leakage (the regime-appropriate
  /// isometry defect); mirrors `FiberConnectionConfig::leakageTolerance`.
  double maxTransportLeakage = 1e-6;
  /// Minimum persistence lifetime: the number of cobordism frames the
  /// candidate was tracked through
  /// (`QuarkCandidateEvidence::frameLifetime`, from
  /// `PersistentModularity::trackAcrossFrames`).  The modularity
  /// resolution-slice lifetime is a different quantity, reported beside
  /// this one and never gating: a modularity read may not veto an
  /// otherwise certified fiber.
  double minPersistenceLifetime = 2.0;
  /// Minimum adjacent-frame support overlap along the cobordism-frame
  /// track: overlap with the predecessor and successor components.
  double minPersistenceOverlap = 0.5;
  /// Band-localization floor: the inverse participation ratio of the color
  /// band, ∈ [1/n, 1/rank].  Localization is already enforced in fiber
  /// acceptance (`SpectralFiberConfig::maxLocalizationExcess`), so the
  /// default 0 adds nothing; an unmeasured NaN still fails.
  double minLocalization = 0.0;
  /// Minimum band subspace overlap across a refinement for the
  /// refinement-stability certificate.
  double minRefinementOverlap = 0.9;
  /// Minimum number of frames a stable quark condition must hold at.  Rank
  /// stability and anchor/coherence stability are across-frame statements,
  /// so a single frame can never establish them.
  std::size_t minStabilityFrames = 2;
  /// Subspace-overlap threshold of the flavor-doublet tracking (passed to
  /// `SpectralFiberTracker::matchFibers`).
  double doubletOverlapThreshold = 0.5;
  /// Minimum number of frames a flavor subclass must persist through.
  std::size_t minDoubletFrames = 2;
  /// |I3 ∓ 1/2| cap for a definite doublet-member occupancy.
  double isospinTolerance = 1e-9;
  /// Max deviation across nested enclosing surfaces (and |Im| leakage)
  /// for a consistent Gauss flux.
  double gaussTolerance = 1e-9;
  /// Minimum number of nested enclosing surfaces for a consistency claim.
  std::size_t minEnclosingSurfaces = 2;
  /// |Q_gauss − (I3 + B/2)| cap for the proposed u/d identification.
  double udTolerance = 1e-9;

  // ── even-sector thresholds ─────────────────────────────────────────────

  /// Floor on the octet Frobenius weight ‖M − (Tr M/3) I‖_F² of a gluon
  /// candidate's bilinear: a nonzero color polarization.  The vacuum reads
  /// exactly zero and fails by name.
  double minOctetWeight = 1e-9;
  /// Cap on the octet-projector residual ‖(I₉ − P₈) vec(M₈)‖ / ‖M₈‖_F of
  /// the excitation.  The traceless bilinear lies in the 8 exactly, so the
  /// residual is rounding and this is a machine-precision certificate.
  double octetPurityTolerance = 1e-9;
  /// Cap on the octet fraction octet/(octet + singlet) of a meson's pair
  /// color bilinear — the color-singlet certificate of the q-q̄ composite.
  double compositeOctetTolerance = 1e-9;
  /// Floor on the certified Λ²C³ anti-triplet wedge occupation
  /// det(C†ΓC) ∈ [0, 1] of a diquark candidate: 1 for two orthonormal
  /// columns, 0 for a duplicated color mode.
  double minAntiTripletWeight = 0.5;

  // ── baryon / proton thresholds ─────────────────────────────────────────

  /// |det(C†C) − 1| cap of the color-singlet certificate of the three
  /// normalized anchored color columns (`ColorFiber::singletGram`).  The
  /// value is 1 for an orthonormal triad and 0 for duplicate color modes,
  /// so this only absorbs rounding.
  double colorGramTolerance = 1e-9;
  /// Cap on the composite's net color flux: the octet (traceless)
  /// Frobenius weight of the bound object's color bilinear.
  double colorFluxTolerance = 1e-9;
  /// |⟨J²⟩ − 3/4| cap of the total-space spin expectation.
  double spinExpectationTolerance = 1e-9;
  /// |Var(J²)| cap of the sharp-spin certificate, the quantity separating
  /// a proton certificate from an accidental expectation value.
  double spinVarianceTolerance = 1e-9;
  /// Minimum fraction of a constituent's level-0 support that must lie
  /// inside a candidate supercomponent (1.0 = full containment).
  double minSupportContainment = 1.0;
  /// Minimum number of shared persistence slices across the three
  /// constituents' lifetime windows.
  double minLifetimeOverlap = 1.0;
  /// Strict floor a finite emergent radius must exceed.
  double minRadius = 0.0;
  /// Cap on the deviation of every dimensionless scale channel across the
  /// configured refinement window (relative spread for the scalars,
  /// max absolute per-shell deviation for the radial weight profile).
  double maxProfileDeviation = 1e-6;
};

/// # GaussFluxRead
///
/// The electric Gauss-flux consistency read over nested enclosing surfaces.
/// Each per-surface flux is a
/// `cobordism::EigenstateSynthesis::gaussLawCharge` value: an
/// orientation-signed sum of the supplied field-strength 2-cochain over the
/// closed star boundary of one enclosed vertex set, restricted to the
/// electric (timelike-leg) plaquettes when `electricOnly`.  Charge is
/// certified only when the surfaces agree; an inconsistent or
/// single-surface read reports an unknown flux.
struct GaussFluxRead {
  /// The per-surface complex fluxes, in the nested-surface input order.
  std::vector<std::complex<double>> fluxes{};
  /// Number of enclosed vertices of each surface (the nesting witness).
  std::vector<std::size_t> surfaceVertexCounts{};
  /// Whether only electric (timelike-leg) plaquettes were summed.
  bool electricOnly = true;
  /// Max |flux_i − flux_j| over all surface pairs (0 for < 2 surfaces).
  double maxDeviation = std::numeric_limits<double>::quiet_NaN();
  /// Max |Im flux_i|: the imaginary leakage of the real-by-construction
  /// electric charge, never silently discarded.
  double imagLeakage = std::numeric_limits<double>::quiet_NaN();
  /// Whether the read met `gaussTolerance` across at least
  /// `minEnclosingSurfaces` surfaces.
  bool consistent = false;
  /// The consistent electric flux: Re(mean of the per-surface values) when
  /// `consistent`; empty otherwise.
  std::optional<double> electricFlux{};
  /// Names of the failed consistency certificates ("gauss-consistency";
  /// empty when consistent).
  std::vector<std::string> failedCertificates{};
  /// AlgebraicallyExact (the flux sums are exact signed sums of the
  /// supplied cochain) with residual = max(maxDeviation, imagLeakage)
  /// against `gaussTolerance`; HeuristicDiscovery when inconsistent.
  cobordism::Certificate certificate{};
};

/// # FlavorDoubletRead
///
/// The emergent, unlabeled, transported two-state spectral subclass that
/// could carry isospin.  The search runs without a requested dimension:
/// every stable transported subclass of the candidate's band enumeration is
/// followed, and "two-state" is an outcome (`stableSubclassRanks` reports
/// every stable rank found).  The stored first-frame fiber is the recorded
/// member trivialization, a convention like a declared anchor weighting and
/// not a physical u/d label.
struct FlavorDoubletRead {
  /// Whether exactly one stable transported two-state subclass emerged.
  bool found = false;
  /// Form degree of the doublet band (meaningful when `found`).
  int degree = 0;
  /// Rank of the accepted subclass (2 when `found`; never requested).
  std::size_t rank = 0;
  /// Number of frames the winning subclass persisted through.
  std::size_t framesTracked = 0;
  /// Smallest certified continuation overlap along the winning track.
  double minContinuationOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Worst band isolation min(lowerGap, upperGap) along the winning track.
  double minIsolation = std::numeric_limits<double>::quiet_NaN();
  /// Ranks of all stable full-length certified subclasses found, in
  /// first-frame band order — the no-requested-dimension witness.
  std::vector<std::size_t> stableSubclassRanks{};
  /// Number of stable two-state subclasses.  `found` requires exactly one;
  /// two or more is an ambiguous doublet hypothesis and stays uncertified.
  std::size_t twoStateCount = 0;
  /// The winning subclass's first-frame fiber: the trivialization isospin
  /// occupancy is measured against.  Default-constructed when not `found`.
  SpectralFiber doublet{};
  /// Names of the failed certificates ("flavor-doublet" when the doublet
  /// hypothesis is uncertified).
  std::vector<std::string> failedCertificates{};
  /// Why the doublet is uncertified, when it is ("" when found):
  /// "insufficient-frames", "no-stable-two-state-subclass", or
  /// "ambiguous-two-state-subclasses".
  std::string invalidationReason{};
  /// CertifiedNumerical (residual = 1 − minContinuationOverlap against
  /// 1 − doubletOverlapThreshold) when found; HeuristicDiscovery
  /// otherwise.
  cobordism::Certificate certificate{};
};

/// # QuarkCandidateEvidence
///
/// The assembled evidence bundle of one candidate.  Unsupplied evidence (a
/// default-constructed, NaN or empty field) fails its certificate by name.
struct QuarkCandidateEvidence {
  /// The candidate's label-free identity.
  ComponentId component{};
  /// The candidate's selected color band.  The classifier reads only its
  /// rank, acceptance and localization; rank three is required, never
  /// requested from the detector.  Rank stability is decided on
  /// `colorBandFrames`.
  SpectralFiber colorBand{};
  /// The candidate's color band at each cobordism frame of its tracked
  /// lifetime, in frame order (`colorBand` is conventionally the first).
  /// Stable rank three is decided here: every frame's band accepted at rank
  /// three, consecutive frames certified continuations
  /// (`SpectralFiberTracker::matchFibers`: equal rank, both accepted,
  /// subspace overlap at or above `doubletOverlapThreshold`), over at least
  /// `minStabilityFrames` frames.  With fewer frames the stability was
  /// never measured and the certificate fails by name.
  std::vector<SpectralFiber> colorBandFrames{};
  /// The calibrated oriented-triangle anchor profile of `colorBand`
  /// (`ColorAnchor::evaluate` output with its pre-declared weighting).
  /// The reported profile; its stability is decided on `anchorFrames`.
  AnchorProfile anchor{};
  /// The candidate's anchor profile at each cobordism frame, in frame
  /// order (`anchor` is conventionally the first).  Anchor and coherence
  /// stability are decided here: every frame's profile certified, with its
  /// score and its overlap-restricted determinant-phase coherence above
  /// their floors, over at least `minStabilityFrames` frames.  The measured
  /// across-frame spreads travel on the read as diagnostics.
  std::vector<AnchorProfile> anchorFrames{};
  /// The candidate's lifetime transports (the world-tube family): every
  /// link must be accepted with leakage under the configured cap.
  std::vector<FiberTransportRead> lifetimeTransports{};
  /// The determinant-line winding of the lifetime family: a closed
  /// full-rank family, or an open cobordism segment under its recorded
  /// closure specification.  Invalidated or unclosed leaves the baryon
  /// flux unknown.
  DeterminantWindingRead winding{};
  /// The Wick parity ⟨(−1)^N⟩ of the candidate's carried quasi-free state
  /// (`CovarianceState::wickParity`).
  quantum::WickCertificateRead parityRead{};
  /// The Wick total occupation ⟨N⟩ (`CovarianceState::wickTotalNumber`).
  quantum::WickCertificateRead occupationRead{};
  /// Modularity resolution-slice lifetime (covered slices; NaN = missing).
  /// Report-only: a resolution-scan count states how stable a modularity
  /// proposal is under the resolution parameter, not how long the candidate
  /// lived, and modularity may not veto a certified fiber.
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Smallest adjacent-slice support overlap along the resolution track.
  /// Report-only, for the same reason.
  double persistenceMinOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Cobordism-frame lifetime: consecutive frames the candidate was
  /// tracked through (`FrameTrack::frames` from
  /// `PersistentModularity::trackAcrossFrames`; NaN = missing).  This is
  /// the gated persistence quantity.
  double frameLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Smallest adjacent-frame support overlap along that track
  /// (`FrameTrack::minAdjacentOverlap`; NaN = missing).
  double frameMinOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Band subspace overlap across a refinement
  /// (`SpectralFiber::overlap(...).subspaceOverlap` between the band and
  /// its refined re-extraction; NaN = missing).
  double refinementOverlap = std::numeric_limits<double>::quiet_NaN();
  /// The flavor-doublet search result (`flavorDoubletSearch`); absent =
  /// no doublet evidence, flavor unknown.
  std::optional<FlavorDoubletRead> flavor{};
  /// The candidate's amplitudes on the two doublet members, in the
  /// recorded trivialization (the doublet fiber's stored column order);
  /// absent = occupancy unknown.
  std::optional<Eigen::Vector2cd> doubletOccupancy{};
  /// The declared doublet orientation s ∈ {+1, −1}: which member carries
  /// I3 = +1/2 under the proposed identification.  A recorded convention,
  /// like a declared anchor weighting, not a hidden label.
  int doubletOrientation = +1;
  /// The nested-surface Gauss-flux read (`gaussFluxOnSurfaces`); absent =
  /// no charge evidence, charge unknown.
  std::optional<GaussFluxRead> charge{};
};

/// # QuarkRead
///
/// The quark/antiquark particle read: the verdict, the evidence summary the
/// classification consumed, the recorded thresholds, and the certificate.
struct QuarkRead {
  /// The candidate's label-free identity.
  ComponentId component{};
  /// Certified exterior parity: −1 (odd) / +1 (even) / 0 = unknown (an
  /// uncertified parity read never emits a sign).
  int exteriorParity = 0;
  /// Rank of the supplied color band (0 when none was supplied).
  int colorRank = 0;
  /// The calibrated anchor atlas score a² (NaN when no anchor evidence).
  double triangleAnchorScore = std::numeric_limits<double>::quiet_NaN();
  /// max_τ |det A_τ|² of the anchor profile.
  double triangleAnchorMaxTerm = std::numeric_limits<double>::quiet_NaN();
  /// Participation ratio of the anchor term distribution.
  double triangleAnchorParticipation =
      std::numeric_limits<double>::quiet_NaN();
  /// Determinant-phase dispersion (1 − coherence) of the anchor profile.
  double anchorPhaseDispersion = std::numeric_limits<double>::quiet_NaN();
  /// The pre-declared anchor weighting rule ("uniform" / "declared").
  std::string anchorWeightingId{};
  /// The certified determinant-line winding ν; empty when the family was
  /// invalidated or no closure was declared.
  std::optional<int> determinantWinding{};
  /// The recorded winding closure specification: "closed-family",
  /// "matched-reference", "endpoint-trivialization", or "none".
  std::string windingClosure{"none"};
  /// The caller's closure reference identifier ("" when none).
  std::string windingReferenceId{};
  /// B = ν/3 under a certified winding (a certified ν = 0 is a certified
  /// zero flux); empty without the winding certificate.
  std::optional<double> baryonFlux{};
  /// I3 = ±1/2 under the certified doublet hypothesis and the declared
  /// orientation; empty when the doublet is missing or unstable, or the
  /// occupancy is not a definite member.
  std::optional<double> isospin{};
  /// The Gauss-consistent electric flux; empty unless the nested-surface
  /// Gauss read is consistent and the flavor doublet is certified, since a
  /// missing doublet leaves both flavor and charge unknown.
  std::optional<double> electricFlux{};
  /// Passed-fraction of the twelve core quark certificates; 1.0 exactly
  /// when the candidate is a certified quark or antiquark.
  double confidence = 0.0;
  /// Names of every failed or missing certificate, core order first then
  /// flavor/charge order.  Empty for a fully certified, u/d-identified
  /// quark.
  std::vector<std::string> failedCertificates{};

  // ── evidence summary ─────────────────────────────────────────────────

  /// "quark" (ν = +1), "antiquark" (ν = −1), or "none".
  std::string classification{"none"};
  /// ⟨N⟩ of the carried state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// Determinant-phase coherence of the anchor profile (1 − dispersion).
  double anchorPhaseCoherence = std::numeric_limits<double>::quiet_NaN();
  /// Number of lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst lifetime transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Modularity resolution-slice lifetime consumed (NaN = missing).
  /// Reported, never gated (see `QuarkCandidateEvidence`).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Minimum adjacent-slice overlap consumed.  Reported, never gated.
  double persistenceMinOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Cobordism-frame lifetime consumed — the gated persistence quantity
  /// (NaN = missing).
  double frameLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Smallest adjacent-frame overlap consumed — the gated overlap
  /// quantity (NaN = missing).
  double frameMinOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Number of frames the color band / anchor stability was measured over
  /// (0 = never measured; the stability certificates then fail by name).
  std::size_t stabilityFrames = 0;
  /// Measured across-frame spread (max − min) of the anchor atlas score
  /// and of its overlap-restricted determinant-phase coherence; NaN when
  /// stability was never measured.  Diagnostics only: the stability
  /// certificate requires each frame's value to clear its floor, not the
  /// spread to clear a cap.
  double anchorScoreSpread = std::numeric_limits<double>::quiet_NaN();
  double anchorCoherenceSpread = std::numeric_limits<double>::quiet_NaN();
  /// Smallest certified-continuation subspace overlap between consecutive
  /// color-band frames (NaN when fewer than two frames were supplied).
  double bandContinuationOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Band localization consumed (from the color-band certificate).
  double localization = std::numeric_limits<double>::quiet_NaN();
  /// The band's effective support fraction consumed (the quantity the
  /// upstream localization condition uses; NaN = missing).
  double localizationSupportFraction =
      std::numeric_limits<double>::quiet_NaN();
  /// Refinement subspace overlap consumed (NaN = missing).
  double refinementOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Whether Q = I3 + B/2 was tested and held — the proposed u/d
  /// identification, not a general charge definition.
  bool udIdentificationProposed = false;
  /// The declared doublet orientation the isospin was reported under
  /// (0 when no isospin was reported).
  int doubletOrientation = 0;
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (an exact boolean combination given the consumed held
  /// certificates; residual = their maximum residual) for a certified
  /// quark/antiquark; HeuristicDiscovery (never holds) otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary (classification, ν, B, parity,
  /// failed certificates).
  [[nodiscard]] std::string describe() const;

  /// Checkpoint serialization (`particles.quarks`): every field, the
  /// evidence summary, the failed-certificate names and the threshold echo
  /// travel together.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate from `toRecord()` output; rejects an unknown
  /// `schema_version` (std::invalid_argument).
  [[nodiscard]] static QuarkRead fromRecord(const Record &record);
};

/// # ConjugatePairRead
///
/// Pair-conservation verification of a conjugate quark-antiquark creation
/// path, valid only for a certified conjugate homotopy.  A gap-preserving
/// path has certified windings on both legs and they cancel exactly; a
/// singular (rank- or gap-closing) leg leaves the total flux unknown.
struct ConjugatePairRead {
  /// ν_a + ν_b when both windings are certified; empty otherwise.
  std::optional<int> totalWinding{};
  /// B_a + B_b when both baryon fluxes are known; empty otherwise — the
  /// channel on which a singular path returns an unknown flux.
  std::optional<double> totalBaryonFlux{};
  /// Product of the two certified parities (+1 even / −1 odd / 0 =
  /// unknown).
  int totalParity = 0;
  /// Whether the certified total parity is even (+1).
  bool parityEven = false;
  /// Certified conservation: both windings certified, total winding 0,
  /// and even total parity.
  bool conserved = false;
  /// Names of the failed certificates: "winding-first", "winding-second",
  /// "parity-first", "parity-second", "winding-conservation",
  /// "parity-even".
  std::vector<std::string> failedCertificates{};
  /// StructureExact (integer sums are exact given certified integer
  /// windings and parities) when conserved; HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};
};

/// # OctetBilinearRead
///
/// The quasi-free traceless-bilinear (octet) read of three declared color
/// modes of a carried `quantum::CovarianceState`, evaluated on the
/// covariance layer: polynomial in the mode count, with no Fock vector.
/// The lazy engine is the dense oracle in tests and the carrier of
/// explicitly non-Gaussian boundary data.
///
/// Exact identities, all delegated and tested to rounding:
///   • M_ij = ⟨a_i†a_j⟩ = Γ_{ji} on the declared modes — `bilinear` is the
///     transposed principal submatrix of Γ.
///   • 1 ⊕ 8 resolution: `singletWeight`/`octetWeight` are exactly
///     `ColorFiber::octetRead(bilinear)`; `octetComponent` is exactly
///     `ColorFiber::tracelessPart(bilinear)`, which lies in the octet by
///     the algebra, so `octetProjectorResidual` measures
///     ‖(I₉ − P₈) vec(M₈)‖/‖M₈‖_F ≈ 0 (rounding only).
///   • `casimir` = `ColorFiber::adjointCasimir(octetComponent)` = 3 for a
///     nonzero excitation (C = 3 P₈).
///   • `casimirExpectation` = ⟨Σ_a dΓ(λ_a/2)²⟩ by quartic Wick sums
///     (`wickBilinearMoment` with the embedded Gell-Mann halves): exactly 0
///     on the vacuum and the fully occupied singlet, exactly C₂ = 4/3 on
///     the fundamental (N = 1) and anti-triplet (N = 2) Slater states.
///   • `gellMannComponents[a-1]` = Tr(λ_a M)/2 — the octet coordinates:
///     M = (Tr M/3) I + Σ_a comp_a λ_a exactly.
///
/// Adding microscopic modes (vacuum embedding via `embedInVacuum` or a
/// zero-block covariance extension) leaves this read unchanged: arbitrarily
/// many collective excitations are represented without changing any
/// two-dimensional edge-mode factor.
struct OctetBilinearRead {
  /// The three declared color modes, in the caller's recorded order: the
  /// color trivialization.
  std::vector<std::size_t> colorModes{};
  /// Certified subset occupation ⟨N_S⟩ = tr Γ_S (NaN when the constituent
  /// Wick reads did not certify).
  double occupation = std::numeric_limits<double>::quiet_NaN();
  /// Certified subset parity sign of ⟨(−1)^{N_S}⟩: +1 / −1 / 0 = unknown
  /// or indefinite.  Never a forced sign.
  int subsetParity = 0;
  /// The bilinear matrix M_ij = ⟨a_i†a_j⟩ on the declared modes.
  Eigen::Matrix3cd bilinear = Eigen::Matrix3cd::Zero();
  /// The traceless octet component `ColorFiber::tracelessPart(bilinear)`.
  Eigen::Matrix3cd octetComponent = Eigen::Matrix3cd::Zero();
  /// ‖M − (Tr M/3) I‖_F² — the octet Frobenius weight
  /// (`ColorFiber::octetRead`).
  double octetWeight = std::numeric_limits<double>::quiet_NaN();
  /// |Tr M|²/3 — the singlet (trace/number) Frobenius weight.
  double singletWeight = std::numeric_limits<double>::quiet_NaN();
  /// ‖(I₉ − P₈) vec(octetComponent)‖ / ‖octetComponent‖_F, rounding-level
  /// for any state since the excitation is octet by the algebra; NaN when
  /// the excitation vanishes, an undefined residual being unknown.
  double octetProjectorResidual = std::numeric_limits<double>::quiet_NaN();
  /// `ColorFiber::adjointCasimir(octetComponent)` ∈ [0, 3]; 3 exactly for
  /// a nonzero excitation; NaN when it vanishes.
  double casimir = std::numeric_limits<double>::quiet_NaN();
  /// ⟨Σ_a dΓ(λ_a/2)²⟩ — the quartic-Wick color Casimir expectation of the
  /// carried state on the declared modes (NaN when uncertified).
  double casimirExpectation = std::numeric_limits<double>::quiet_NaN();
  /// Tr(λ_a M)/2 for a = 1..8 — the octet coordinates of the bilinear.
  std::vector<std::complex<double>> gellMannComponents{};
  /// Max residual of the consumed Wick reads (their certificates all
  /// travel through `certificate`).
  double residual = std::numeric_limits<double>::quiet_NaN();
  /// AlgebraicallyExact / Static in the covariance's verified regime
  /// (residual = the consumed Wick residual maximum against the read
  /// tolerance): the read is a finite exact Wick sum on the covariance.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization; complex leaves split into
  /// `{name}_re`/`{name}_im`.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static OctetBilinearRead fromRecord(const Record &record);
};

/// # GluonCandidateEvidence
///
/// The assembled evidence bundle of one gluon candidate: the carried-state
/// Wick parity and occupation, the quasi-free octet bilinear read, the
/// lifetime transports and determinant winding, and the persistence
/// diagnostics.  Missing evidence fails its certificate by name.
struct GluonCandidateEvidence {
  /// The excitation's label-free identity.
  ComponentId component{};
  /// The component the excitation is bound to, reported verbatim.
  ComponentId bindingComponent{};
  /// The quasi-free octet bilinear read of the carried state
  /// (`octetBilinearRead`).
  OctetBilinearRead octet{};
  /// The Wick parity ⟨(−1)^N⟩ of the whole carried state
  /// (`CovarianceState::wickParity`) — the even-parity gate.
  quantum::WickCertificateRead parityRead{};
  /// The Wick total occupation ⟨N⟩ (`wickTotalNumber`; report-only).
  quantum::WickCertificateRead occupationRead{};
  /// The candidate's lifetime transports: every link must be accepted,
  /// rank three, with leakage under the configured cap.  The adjoint action
  /// on the octet is exact given the accepted rank-three factor
  /// (`FiberConnection::adjointRepresentation`).
  std::vector<FiberTransportRead> lifetimeTransports{};
  /// The determinant-line winding of the lifetime family.  The gluon gate
  /// requires a certified ν = 0; an unknown winding leaves the flux
  /// unknown.
  DeterminantWindingRead winding{};
  /// Modularity resolution-slice lifetime (NaN = missing).  Report-only,
  /// exactly as on `QuarkCandidateEvidence`.
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Cobordism-frame lifetime (NaN = missing) — the gated persistence
  /// quantity.
  double frameLifetime = std::numeric_limits<double>::quiet_NaN();
};

/// # GluonRead
///
/// The gluon-candidate read: a persistent transported octet excitation with
/// certified even parity and certified zero total determinant winding and
/// baryon flux.  `classification` is "gluon-candidate" or "none", never
/// "gluon": no even octet excitation is claimed to be a physical gluon.
///
/// Certificate names, in fixed order: "parity-even", "octet-excitation",
/// "octet-purity", "octet-transport", "winding-zero", "persistence".
struct GluonRead {
  /// The excitation's label-free identity.
  ComponentId component{};
  /// The reported binding component.
  ComponentId bindingComponent{};
  /// "gluon-candidate" or "none".
  std::string classification{"none"};
  /// Certified carried-state exterior parity: +1 (even) / −1 (odd) / 0 =
  /// unknown.
  int exteriorParity = 0;
  /// ⟨N⟩ of the whole carried state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// Consumed scalars of the octet evidence: the adjoint Casimir, the
  /// quartic-Wick color Casimir expectation, the octet-projector residual,
  /// and the 1 ⊕ 8 Frobenius weights (NaN when missing).  The full
  /// `OctetBilinearRead` travels on the evidence and serializes itself.
  double casimir = std::numeric_limits<double>::quiet_NaN();
  double casimirExpectation = std::numeric_limits<double>::quiet_NaN();
  double octetProjectorResidual = std::numeric_limits<double>::quiet_NaN();
  double octetWeight = std::numeric_limits<double>::quiet_NaN();
  double singletWeight = std::numeric_limits<double>::quiet_NaN();
  /// The certified determinant winding (0 for a certified gluon
  /// candidate); empty when invalidated or unclosed.
  std::optional<int> determinantWinding{};
  /// The recorded winding closure specification.
  std::string windingClosure{"none"};
  /// The caller's closure reference identifier ("" when none).
  std::string windingReferenceId{};
  /// B = ν/3 under a certified winding; 0.0 is a certified zero flux.
  /// Empty without the winding certificate.
  std::optional<double> baryonFlux{};
  /// Number of lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst lifetime transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Modularity resolution-slice lifetime consumed (report-only; NaN =
  /// missing).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Cobordism-frame lifetime consumed — the gated persistence quantity.
  double frameLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the six gluon certificates; 1.0 exactly for a
  /// certified gluon candidate.
  double confidence = 0.0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (an exact boolean combination given the consumed held
  /// certificates) for a certified candidate; HeuristicDiscovery
  /// otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization (`particles.gluons`).
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static GluonRead fromRecord(const Record &record);
};

/// # CompositeCandidateEvidence
///
/// The assembled evidence bundle of one two-cluster composite, meson or
/// diquark; the three-cluster bundle is `BaryonCandidateEvidence` below.
/// The constituents are `QuarkRead`s consumed verbatim, and the composite
/// channels (carried-state occupation, the pair color bilinear, the
/// anti-triplet wedge read) are caller-assembled upstream reads.
struct CompositeCandidateEvidence {
  /// The component binding the two clusters, reported verbatim.
  ComponentId bindingComponent{};
  /// The first constituent's read.
  QuarkRead first{};
  /// The second constituent's read.
  QuarkRead second{};
  /// The Wick total occupation ⟨N⟩ of the carried composite state
  /// (report-only; NaN or uncertified = unknown).
  quantum::WickCertificateRead occupationRead{};
  /// Meson channel: the pair color bilinear M_ij pairing constituent color
  /// i with conjugate color j (3 ⊗ 3̄); a singlet composite has M ∝ I.
  /// Absent means no pairing evidence, and the color-singlet certificate
  /// fails by name.
  std::optional<Eigen::Matrix3cd> colorPairing{};
  /// Diquark channel: the certified Λ²C³ wedge occupation det(C†ΓC) of the
  /// two constituent color columns on the carried state
  /// (`wickGramDeterminant` / `wickColorWedgeSquared`), exactly zero for
  /// duplicated color modes by the Pauli principle.  Default-constructed
  /// means missing evidence.
  quantum::WickCertificateRead antiTripletRead{};
  /// The composite's lifetime transports (report-only for the two-cluster
  /// reads — the max leakage travels on the read).
  std::vector<FiberTransportRead> lifetimeTransports{};
  /// Persistence-track lifetime of the composite (NaN = missing).
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
};

/// # MesonRead
///
/// The meson-candidate read: one certified quark plus one certified
/// antiquark (order-insensitive), even composite parity — the graded
/// product of the constituent parities, since parity adds mod 2 — a
/// color-singlet pair bilinear under the 1 ⊕ 8 split, and zero total
/// certified winding and baryon flux.
///
/// Certificate names, in fixed order: "constituent-quark",
/// "constituent-antiquark", "parity-even", "color-singlet", "flux-zero".
struct MesonRead {
  /// The reported binding component.
  ComponentId bindingComponent{};
  /// The two constituents' label-free identities, in evidence order.
  ComponentId firstConstituent{};
  ComponentId secondConstituent{};
  /// "meson-candidate" or "none".
  std::string classification{"none"};
  /// The composite exterior parity: the product of the two certified
  /// constituent parities (+1 even / −1 odd / 0 = unknown).
  int exteriorParity = 0;
  /// ⟨N⟩ of the carried composite state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// The 1 ⊕ 8 Frobenius weights of the pair color bilinear
  /// (`ColorFiber::octetRead`; NaN when no pairing evidence).
  double pairingSingletWeight = std::numeric_limits<double>::quiet_NaN();
  double pairingOctetWeight = std::numeric_limits<double>::quiet_NaN();
  /// octet/(octet + singlet) of the pairing (NaN when missing or zero).
  double pairingOctetFraction = std::numeric_limits<double>::quiet_NaN();
  /// ν₁ + ν₂ when both constituent windings are certified; empty
  /// otherwise (unknown, never zero).
  std::optional<int> totalWinding{};
  /// B₁ + B₂ when both baryon fluxes are known; empty otherwise.
  std::optional<double> totalBaryonFlux{};
  /// Number of composite lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst composite transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Composite persistence lifetime.
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the five meson certificates.
  double confidence = 0.0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact given the consumed held certificates when certified;
  /// HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static MesonRead fromRecord(const Record &record);
};

/// # DiquarkRead
///
/// The diquark-candidate read: two certified quarks (ν = +1 each), even
/// composite parity (the graded product), a certified Λ²C³ anti-triplet
/// wedge occupation, and the preserved constituent baryon flux
/// B = B₁ + B₂ = 2/3.  A diquark is not an antiquark: the color
/// representation alone (3̄) coincides, but the recorded distinction
/// channels are total occupation two, even parity and B = +2/3, against an
/// antiquark's occupation one, odd parity and B = −1/3.
///
/// Certificate names, in fixed order: "constituent-quarks", "parity-even",
/// "anti-triplet", "baryon-flux-two-thirds".
struct DiquarkRead {
  /// The reported binding component.
  ComponentId bindingComponent{};
  /// The two constituents' label-free identities, in evidence order.
  ComponentId firstConstituent{};
  ComponentId secondConstituent{};
  /// "diquark-candidate" or "none".
  std::string classification{"none"};
  /// The composite exterior parity: the constituent product (0 =
  /// unknown).
  int exteriorParity = 0;
  /// ⟨N⟩ of the carried composite state (NaN when unmeasured).
  double occupationTotal = std::numeric_limits<double>::quiet_NaN();
  /// The certified anti-triplet wedge occupation det(C†ΓC) (NaN when the
  /// wedge read is missing/uncertified).
  double antiTripletWeight = std::numeric_limits<double>::quiet_NaN();
  /// ν₁ + ν₂ when both constituent windings are certified (2 for a
  /// certified diquark); empty otherwise.
  std::optional<int> totalWinding{};
  /// B₁ + B₂, which is 2/3 for a certified diquark: the constituent flux is
  /// preserved, never re-derived.  Empty when unknown.
  std::optional<double> totalBaryonFlux{};
  /// Number of composite lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst composite transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Composite persistence lifetime.
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the four diquark certificates.
  double confidence = 0.0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact given the consumed held certificates when certified;
  /// HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate; rejects an unknown `schema_version`.
  [[nodiscard]] static DiquarkRead fromRecord(const Record &record);
};

/// # BoundCandidateEvidence
///
/// One constituent's datum for the bound-supercomponent search: the quark
/// verdict, the level-0 support and persistence window, and the transports
/// to the other constituents, each consumed verbatim.
struct BoundCandidateEvidence {
  /// The candidate's quark read; only a certified "quark" verdict counts
  /// toward the three-quark census.
  QuarkRead quark{};
  /// The candidate's level-0 cell support (`ComponentRead::support`), the
  /// set tested for containment in the supercomponent.  Empty means
  /// missing evidence and the containment certificate fails by name.
  std::vector<std::uint64_t> support{};
  /// The candidate's persistence window {`PersistenceTrack::firstSlice`,
  /// `PersistenceTrack::lastSlice`}, inclusive slice indices.  Empty means
  /// no lifetime evidence, and the overlap certificate fails by name.
  std::optional<std::pair<std::size_t, std::size_t>> lifetime{};
  /// The transports between this constituent and the others.  Every
  /// supplied link must be accepted with leakage under
  /// `maxTransportLeakage`, so that mutual transport remains inside the
  /// supercomponent; a leaking transfer is the tracked subspace turning
  /// away from its successor.  Empty means missing evidence.
  std::vector<FiberTransportRead> mutualTransports{};
};

/// # BoundSupercomponentRead
///
/// One next-modular-level component examined by the bound-supercomponent
/// search: which certified quark candidates it contains, their shared
/// lifetime window, and the containment and transport certificates.
///
/// Certificate names, in fixed order: "supercomponent-level",
/// "quark-count", "support-containment", "lifetime-overlap",
/// "transport-containment".
struct BoundSupercomponentRead {
  /// The examined next-level component's label-free identity.
  ComponentId boundComponent{};
  /// The contained certified quark candidates' identities, in input order
  /// (three when `found`).
  std::vector<ComponentId> quarks{};
  /// Indices of the contained candidates in the input candidate list.
  std::vector<std::size_t> quarkIndices{};
  /// Whether this component is a certified bound supercomponent of
  /// exactly three lifetime-overlapping certified quark candidates.
  bool found = false;
  /// The shared lifetime window [first, last] of the contained
  /// candidates; empty when any lifetime is missing or the windows are
  /// disjoint.
  std::optional<std::pair<std::size_t, std::size_t>> lifetimeWindow{};
  /// Number of shared persistence slices (0 when disjoint or unknown).
  double lifetimeOverlap = 0.0;
  /// Smallest per-constituent support-containment fraction (NaN when a
  /// constituent supplied no support).
  double minContainment = std::numeric_limits<double>::quiet_NaN();
  /// Worst mutual-transport leakage over the contained candidates (NaN
  /// when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Number of mutual transports consumed.
  std::size_t transportCount = 0;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (the containment and overlap decisions are exact set
  /// and integer-interval statements given the consumed certified
  /// constituent reads; residual = their maximum) when `found`;
  /// HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
};

/// # ScaleProfileSample
///
/// One refinement-window sample of the mass-radius battery
/// (`InteriorHinges`).  See the file banner section "Form factors":
/// `radialWeightProfile` is a radial curvature-weight density, not a
/// momentum-transfer form factor.
struct ScaleProfileSample {
  /// The emergent radius r = V_dual^{1/4} (`InteriorHinges::Radii::rDual`),
  /// dimensionful in lattice units; only its finiteness is certified.
  double radius = std::numeric_limits<double>::quiet_NaN();
  /// The primal cross-check radius V_primal^{1/4}
  /// (`InteriorHinges::Radii::rPrimal`), dimensionful; its ratio to
  /// `radius` is the dimensionless channel.
  double radiusCrossCheck = std::numeric_limits<double>::quiet_NaN();
  /// The spectral-mass channel: the intensive shell mass
  /// (`InteriorHinges::Masses::mShell`), a mean interior deficit angle and
  /// therefore dimensionless in lattice units.
  double spectralMass = std::numeric_limits<double>::quiet_NaN();
  /// The curvature-weight participation ratio
  /// (`InteriorHinges::Localization::pr` ∈ (0, 1]), dimensionless.
  double localization = std::numeric_limits<double>::quiet_NaN();
  /// The share of the |Re ε · ★h| curvature weight per breadth-first shell
  /// (`InteriorHinges::Localization::shellProfile[k].weightShare`), shell
  /// ascending, dimensionless.  Empty when the interior carried no shell
  /// seeds, and the stability certificate then fails by name.
  std::vector<double> radialWeightProfile{};

  // ── the remaining dimensionless certificate channels ─────────────────
  //
  // Every dimensionless certificate must be stable under refinement, not
  // the mass-radius battery alone.  These channels carry the candidate's
  // own certificate values at this refinement, filled by the caller from
  // the same constituent reads `classifyBaryon` consumes.  An unfilled
  // channel is NaN or a 0 sign, and its stability certificate then fails
  // by name rather than passing vacuously.

  /// det(C†C) of the three normalized color columns at this refinement.
  double colorGramDeterminant = std::numeric_limits<double>::quiet_NaN();
  /// The Berry-cancelled 2π rotation character at this refinement.
  std::complex<double> rotationCharacter{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// B = ν/3 over the certified constituent windings at this refinement.
  double baryonFlux = std::numeric_limits<double>::quiet_NaN();
  /// The summed certified constituent electric Gauss flux at this
  /// refinement.
  double electricFlux = std::numeric_limits<double>::quiet_NaN();
  /// The composite exterior parity at this refinement: −1 (odd) / +1
  /// (even) / 0 = unknown.  An integer channel: stability is exact
  /// equality across the window, never a tolerance.
  int compositeParity = 0;
  /// The anchor atlas score a² carried by the composite at this
  /// refinement: the minimum over the three constituents' certified
  /// scores, so the worst leg governs the composite's stability.
  double anchorScore = std::numeric_limits<double>::quiet_NaN();
};

/// # ScaleProfileRead
///
/// The refinement-window certificate over `ScaleProfileSample`s: a finite
/// emergent radius plus the refinement stability of every dimensionless
/// channel.  `physicalMass` is always empty, since no physical scale has
/// been independently established.
///
/// Certificate names, in fixed order: "refinement-window", "finite-radius",
/// "radius-ratio-stability", "spectral-mass-stability",
/// "localization-stability", "profile-stability", "color-gram-stability",
/// "rotation-character-stability", "baryon-flux-stability",
/// "electric-flux-stability", "composite-parity-stability",
/// "anchor-score-stability".
struct ScaleProfileRead {
  /// Number of refinement samples consumed.
  std::size_t sampleCount = 0;
  /// The reported emergent radius (the first sample's `radius`).
  double radius = std::numeric_limits<double>::quiet_NaN();
  /// Whether every sample's radius is finite and above `minRadius`.
  bool radiusFinite = false;
  /// The dimensionless radius ratio r_dual / r_primal of the first sample,
  /// and its normalized spread (max − min)/max(|mean|, 1) over the window:
  /// relative for O(1)-and-larger channels, absolute for channels near
  /// zero, so a near-zero channel is never reported as infinitely unstable.
  /// Every spread below uses the same normalization.
  double radiusRatio = std::numeric_limits<double>::quiet_NaN();
  double radiusRatioSpread = std::numeric_limits<double>::quiet_NaN();
  /// The dimensionless spectral-mass channel of the first sample and its
  /// relative spread over the window.
  double spectralMass = std::numeric_limits<double>::quiet_NaN();
  double spectralMassSpread = std::numeric_limits<double>::quiet_NaN();
  /// The localization channel of the first sample and its relative
  /// spread over the window.
  double localization = std::numeric_limits<double>::quiet_NaN();
  double localizationSpread = std::numeric_limits<double>::quiet_NaN();
  /// Max absolute per-shell deviation of the radial weight profiles across
  /// the window (NaN when a profile was missing or the shell counts
  /// disagree).
  double profileMaxDeviation = std::numeric_limits<double>::quiet_NaN();
  /// Number of shells the profiles share (0 when unavailable).
  std::size_t profileShells = 0;
  /// The remaining dimensionless certificate channels: the first sample's
  /// value and its spread over the refinement window, under the same
  /// normalization as the scalars above.  NaN means the channel was never
  /// supplied, and its certificate fails by name.
  double colorGramDeterminant = std::numeric_limits<double>::quiet_NaN();
  double colorGramSpread = std::numeric_limits<double>::quiet_NaN();
  std::complex<double> rotationCharacter{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// max |χ_i − χ_j| over the window (a complex-plane deviation).
  double rotationCharacterSpread = std::numeric_limits<double>::quiet_NaN();
  double baryonFlux = std::numeric_limits<double>::quiet_NaN();
  double baryonFluxSpread = std::numeric_limits<double>::quiet_NaN();
  double electricFlux = std::numeric_limits<double>::quiet_NaN();
  double electricFluxSpread = std::numeric_limits<double>::quiet_NaN();
  /// The composite parity of the first sample (0 = unknown) and whether
  /// every sample carried the same definite parity (exact integer
  /// equality, no tolerance).
  int compositeParity = 0;
  bool compositeParityStable = false;
  double anchorScore = std::numeric_limits<double>::quiet_NaN();
  double anchorScoreSpread = std::numeric_limits<double>::quiet_NaN();
  /// The dimensionful mass: always empty, unknown until a physical scale is
  /// independently established.  Never zero, and never a lattice number
  /// relabeled as a mass.
  std::optional<double> physicalMass{};
  /// Whether every listed certificate held.
  bool stable = false;
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// CertifiedNumerical (the spreads are measured deviations of finite
  /// sums; residual = the worst measured deviation against
  /// `maxProfileDeviation`) when `stable`; HeuristicDiscovery otherwise.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
};

/// # BaryonCandidateEvidence
///
/// The assembled evidence bundle of one three-cluster candidate: the
/// constituent verdicts, the bound-supercomponent search and octet bilinear
/// read, the color columns, the rotation character and spin lift, the Wick
/// spin reads, and the mass-radius samples.  Missing evidence fails its
/// certificate by name.
struct BaryonCandidateEvidence {
  /// The bound supercomponent's label-free identity, reported verbatim.
  ComponentId boundComponent{};
  /// The three constituents' quark reads, consumed verbatim.
  std::array<QuarkRead, 3> quarks{};
  /// The bound-supercomponent search result
  /// (`boundSupercomponentSearch`).  Its contained-candidate set must be
  /// exactly these three constituents' identities — one persistent bound
  /// supercluster containing them; an incoherent bundle fails
  /// `bound-supercomponent` by name.
  BoundSupercomponentRead binding{};
  /// The three normalized anchored color columns C = [c_A c_B c_C], in
  /// constituent order.  The three-mode wedge is built once from this
  /// matrix; no extra fermion sign is multiplied onto the color ε.
  Eigen::Matrix3cd colorColumns = Eigen::Matrix3cd::Zero();
  /// The bound object's octet bilinear read (`octetBilinearRead` on the
  /// composite's carried state and its three declared color modes) — the
  /// independent net-color-flux diagnostic.  Default-constructed means
  /// missing evidence.
  OctetBilinearRead colorFlux{};
  /// The Berry-cancelled physical-rotation character of the closed
  /// total-space 2π cluster-frame cycle against its matched co-moving
  /// non-rotating reference (`ExchangeHolonomy::rotationCharacter`).
  HolonomyCharacterRead rotation{};
  /// The Berry-cancelled particle-exchange character
  /// (`ExchangeHolonomy::exchangeCharacter`), when the caller ran the
  /// exchange experiment.  Report-only: it gates nothing and only fills
  /// `BaryonRead::exchangeCharacter` and the doubly cancelled
  /// spin-statistics ratio.  A read tagged with the wrong channel is
  /// refused.
  std::optional<HolonomyCharacterRead> exchange{};
  /// Whether the caller is making a continuum spin claim.  When true the
  /// SO(d) → Spin(d) lift is required, and a missing or obstructed lift
  /// fails by name; when false the lift is not applicable and is never
  /// demanded.
  bool continuumSpinClaim = false;
  /// The `ExchangeHolonomy::spinLift` decision, when one was made.
  std::optional<SpinLiftRead> spinLift{};
  /// The total-space ⟨J²⟩ of the carried quasi-free state
  /// (`CovarianceState::wickSpinSquaredExpectation`).
  quantum::WickCertificateRead spinSquaredRead{};
  /// The Var(J²) of the carried quasi-free state
  /// (`CovarianceState::wickSpinSquaredVariance`) — the sharp-spin
  /// certificate.
  quantum::WickCertificateRead spinVarianceRead{};
  /// The accepted covariance-only class: the Var(J²) read of every
  /// quasi-free candidate the obstruction verdict quantifies over,
  /// including the candidate's own read.  Empty or partially uncertified
  /// means the class was not swept, so a variance failure is unknown rather
  /// than an obstruction.
  std::vector<quantum::WickCertificateRead> classVarianceReads{};
  /// The dense total-space ⟨J²⟩ oracle
  /// (`ExchangeHolonomy::totalJSquared`) for a candidate carried as an
  /// explicit composite state rather than a covariance.  Consulted only
  /// when the Wick expectation is absent or uncertified; it supplies no
  /// variance, and expectation alone is not a sharp-spin certificate.
  std::optional<double> totalSpaceJ2{};
  /// The refinement-window samples of the mass-radius battery
  /// (`scaleProfileSample`).
  std::vector<ScaleProfileSample> scaleSamples{};
  /// Persistence-track lifetime of the bound component (NaN = missing).
  /// Report-only: the persistence requirement lives in the supercomponent
  /// search's lifetime-overlap certificate.
  double persistenceLifetime = std::numeric_limits<double>::quiet_NaN();
  /// The composite's lifetime transports (report-only — the max leakage
  /// travels on the read).
  std::vector<FiberTransportRead> lifetimeTransports{};
  /// The world-tube crossing mass for this candidate,
  /// `m_x = kappa_m sum |pi_perp|`, a difference against M0.  Empty when
  /// the caller supplied none, and the `crossing-readouts` gate then passes
  /// vacuously, as `spin-lift` does.  When supplied it is enforced: the
  /// crossing mass must be finite and the coherent one-third sum must read
  /// B = +1 with no determinant-line sign defect.
  std::optional<CrossingMassRead> crossingMass{};
  /// The coherent one-third baryon sum for the same candidate and level.
  /// Travels with `crossingMass`; supplying one without the other fails the
  /// gate by name rather than grading half a certificate.
  std::optional<BaryonCrossingRead> crossingBaryon{};
};

/// # BaryonRead
///
/// The three-quark baryon read and proton certificate, with the evidence
/// summary the verdict consumed, the recorded thresholds, and the
/// certificate.  `classification` is one of "no-baryon",
/// "baryon-candidate", "certified-proton", or
/// "quasi-free-sharp-spin-obstruction".
///
/// Certificate names: the two structural gates first, a failure of either
/// giving "no-baryon" — "constituent-quarks", "bound-supercomponent" — then
/// the proton-certificate gates "color-singlet", "color-flux-zero",
/// "baryon-flux-unit", "composite-parity-odd", "flavor-uud",
/// "electric-flux-unit", "spin-expectation", "sharp-spin",
/// "rotation-character", "spin-lift", "finite-radius", "profile-stability",
/// "crossing-readouts".
struct BaryonRead {
  /// The three constituents' label-free identities, in evidence order.
  std::array<ComponentId, 3> quarks{};
  /// The bound supercomponent's label-free identity.
  ComponentId boundComponent{};
  /// det(C†C) = |det C|² of the three normalized color columns (NaN when
  /// no color evidence was supplied).
  double colorGramDeterminant = std::numeric_limits<double>::quiet_NaN();
  /// The net-color-flux diagnostic: the octet Frobenius weight of the
  /// bound object's color bilinear (NaN when the octet read is missing or
  /// uncertified).  Not on its own a proof of confinement.
  double colorFlux = std::numeric_limits<double>::quiet_NaN();
  /// B = ν/3 with ν the sum of the three certified constituent windings
  /// (+1 for a certified proton); empty when any leg is uncertified.
  std::optional<double> baryonFlux{};
  /// The summed certified constituent electric Gauss fluxes (+1 for a
  /// certified proton); empty when any constituent's charge is unknown.
  std::optional<double> electricFlux{};
  /// The certified total-space ⟨J²⟩ (3/4 for a proton, 15/4 for a Δ);
  /// empty when neither the Wick read nor the dense oracle certified it.
  std::optional<double> totalJ2{};
  /// The certified Var(J²) (≈ 0 for a sharp spin); empty when the
  /// quasi-free variance read is missing or uncertified — unknown, never
  /// zero, and never inferred from the expectation.
  std::optional<double> totalJ2Variance{};
  /// The Berry-cancelled 2π character; empty when the rotation read did
  /// not certify.
  std::optional<std::complex<double>> rotationCharacter{};
  /// "no-baryon", "baryon-candidate", "certified-proton", or
  /// "quasi-free-sharp-spin-obstruction".
  std::string classification{"no-baryon"};
  /// The bound component's persistence lifetime (NaN = missing).
  double persistence = std::numeric_limits<double>::quiet_NaN();
  /// Names of every failed/missing certificate, in the fixed order above.
  std::vector<std::string> failedCertificates{};

  // ── evidence summary ─────────────────────────────────────────────────

  /// The invariant color volume S_ABC = det[c_A c_B c_C], built once (NaN
  /// when no color evidence).  Transposing two constituents flips this sign
  /// and leaves `colorGramDeterminant` invariant.
  std::complex<double> colorWedge{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
  /// ν = ν_A + ν_B + ν_C over the certified constituent windings (3 for a
  /// certified proton); empty when any leg is uncertified.
  std::optional<int> totalWinding{};
  /// The composite exterior parity: the graded product of the three
  /// certified constituent parities (−1 odd / +1 even / 0 = unknown).
  int exteriorParity = 0;
  /// The certified flavor occupation pattern under the constituents'
  /// recorded doublet orientations ("uud", "uuu", "udd", "ddd"); empty
  /// when any constituent's isospin is unknown.
  std::string flavorPattern{};
  /// The summed certified constituent I3 (+1/2 for uud); empty when any
  /// constituent's isospin is unknown.
  std::optional<double> totalIsospin{};
  /// −1 / +1 when the rotation read emitted a sign; 0 otherwise.
  int rotationCharacterSign = 0;
  /// The Berry-cancelled particle-exchange character, when the caller
  /// supplied a certified exchange read; empty otherwise.  Report-only.
  std::optional<std::complex<double>> exchangeCharacter{};
  /// The doubly cancelled spin-statistics ratio
  /// χ̂(exchange) · χ̂(2π)^{-1} (`ExchangeHolonomy::doublyCancelledRatio`;
  /// +1 on a spin-½ fixture, each factor separately near −1), filled only
  /// when both channels are certified and correctly tagged.  Report-only.
  std::optional<std::complex<double>> spinStatisticsRatio{};
  /// Whether a continuum spin claim was declared, so the lift was
  /// demanded, and, when demanded, whether it was accepted.
  bool spinLiftApplicable = false;
  bool spinLiftAccepted = false;
  /// Whether the sharp-spin certificate (certified Var(J²) within
  /// `spinVarianceTolerance`) held.
  bool sharpSpin = false;
  /// Whether the accepted covariance-only class was swept (every class
  /// variance read supplied and certified) — the premise the obstruction
  /// verdict quantifies over.
  bool quasiFreeClassSwept = false;
  /// min over the swept class of |Var(J²)| (NaN when not swept): the
  /// floor the variance failed to converge below.
  double classVarianceFloor = std::numeric_limits<double>::quiet_NaN();
  /// The consumed scale-profile scalars: the emergent radius, its
  /// finiteness, the dimensionless spectral-mass channel and radius ratio,
  /// the worst dimensionless deviation across the refinement window, and
  /// whether every dimensionless channel was stable.  `scaleProfile`
  /// recomputes the full `ScaleProfileRead` from the same evidence.
  double radius = std::numeric_limits<double>::quiet_NaN();
  bool radiusFinite = false;
  double spectralMass = std::numeric_limits<double>::quiet_NaN();
  double radiusRatio = std::numeric_limits<double>::quiet_NaN();
  double profileMaxDeviation = std::numeric_limits<double>::quiet_NaN();
  bool profileStable = false;
  /// The dimensionful mass: always empty (see the file banner).
  std::optional<double> physicalMass{};
  /// The world-tube crossing readouts, when the caller supplied them.
  /// `crossingMassApplicable` is false when none were supplied and the
  /// `crossing-readouts` gate then passed vacuously, leaving the value NaN
  /// and the baryon number empty.  The crossing mass is uncalibrated unless
  /// the producing configuration declared otherwise, so it is not a
  /// dimensionful physical mass.
  bool crossingMassApplicable = false;
  double crossingMassValue = std::numeric_limits<double>::quiet_NaN();
  std::optional<double> crossingBaryonNumber{};
  /// Tubes whose crossing sign disagreed with their determinant-line
  /// winding — a defect signal, carried verbatim onto the verdict.
  std::vector<std::string> crossingSignDefects{};
  /// Number of shared persistence slices of the three constituents.
  double lifetimeOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Number of composite lifetime transports supplied.
  std::size_t transportCount = 0;
  /// Worst composite transport leakage (NaN when none supplied).
  double transportLeakageMax = std::numeric_limits<double>::quiet_NaN();
  /// Passed-fraction of the fifteen certificates listed above; 1.0
  /// exactly for a certified proton.
  double confidence = 0.0;
  /// The thresholds that produced this read (echoed configuration).
  ParticleClustersConfig thresholds{};
  /// StructureExact (an exact boolean combination given the consumed held
  /// certificates; residual = their maximum) for a certified proton;
  /// HeuristicDiscovery (never holds) otherwise, including for the
  /// obstruction verdict, which is a reported branch point rather than a
  /// held claim.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
  /// Checkpoint serialization (`particles.baryons`): every field, the
  /// evidence summary, the failed-certificate names and the threshold echo
  /// travel together.
  [[nodiscard]] Record toRecord() const;
  /// Rehydrate from `toRecord()` output; rejects an unknown
  /// `schema_version` (std::invalid_argument).
  [[nodiscard]] static BaryonRead fromRecord(const Record &record);
};

/// One proposed cluster support and the proposers that offered it
/// (`ParticleClusters::proposeSupports`).
///
/// A support is a set of level-0 cell ids on which the acceptance
/// certificates are then evaluated.  Two proposers offer supports:
/// Newman–Girvan modularity on the combinatorial one-skeleton, which does not
/// see the complex Hodge weights, and the degree-zero band of the covariant
/// operator, which is nothing but those weights
/// (`EffectiveTopology::components`).  A proposal carries which of the two
/// offered it, so a support the metric finds and modularity never proposes is
/// visible as such rather than silently absent.
struct ClusterSupportProposal {
  /// The value of `modularityIndex` or `bandIndex` when that proposer did not
  /// offer this support.
  static constexpr std::size_t kNoProposer = static_cast<std::size_t>(-1);

  /// The proposed support: level-0 cell ids, ascending and deduplicated.
  std::vector<std::uint64_t> support;
  /// Newman–Girvan modularity proposed this support.
  bool modularity = false;
  /// The degree-zero band of the covariant operator proposed this support.
  bool band = false;
  /// The index of the modularity component that proposed it, in the input
  /// order of `proposeSupports`; `kNoProposer` when modularity did not.
  std::size_t modularityIndex = kNoProposer;
  /// The index of the effective component that proposed it, in the input
  /// order of `proposeSupports`; `kNoProposer` when the band did not.
  std::size_t bandIndex = kNoProposer;
  /// The largest Jaccard index \f$ |A \cap B| / |A \cup B| \f$ between this
  /// support and any support the other proposer offered: one when both
  /// proposers offered exactly this set, zero when the other proposer offered
  /// nothing that overlaps it.  A near-agreement reads as a value just below
  /// one, and is reported rather than merged, because the two proposers are
  /// independent and their supports are not interchangeable.
  double crossProposerOverlap = 0.0;
};

/// # ParticleClusters
///
/// The quark/antiquark classifier over persistent modular spectral
/// components.  See the file banner for the identities implemented, their
/// domains, and the certificate names.
///
/// **Composition, not recomputation.**  Every certificate consumed here is
/// produced by an upstream kernel: persistence diagnostics, band
/// certificates and tracking, anchor profiles, transports and determinant
/// windings, parity via the Wick reads, and the Gauss-flux read
/// (`EigenstateSynthesis::gaussLawCharge`).  The classifier's own claim is
/// the exact boolean combination.
///
/// **Certificate names** (`failedCertificates`): the twelve core gates, in
/// order — "persistence", "localization", "parity-odd", "occupation-one",
/// "color-rank-three", "color-rank-stability", "anchor",
/// "anchor-stability", "transport-leakage", "winding", "winding-unit",
/// "refinement-stability" — then the flavor and charge gates
/// "flavor-doublet", "isospin", "gauss-consistency", "ud-identification",
/// which never veto quark-ness and only leave their own fields unknown.
///
/// **Read-only observable.**  Never calls a solver, never mutates the
/// spacetime, and no output may enter any emergence objective.
class ParticleClusters {
  public:
    /// `AnalyticCache` kind string of a cached QuarkRead.
    static constexpr const char *kCacheKind = "quark-read";

    /// Bind the classification thresholds (echoed on every read).
    explicit ParticleClusters(ParticleClustersConfig cfg = {});

    /// The threshold configuration this instance classifies with.
    [[nodiscard]] const ParticleClustersConfig &config() const noexcept {
      return cfg_;
    }

    // ── the quark classifier ────────────────────────────────────────────

    /// Classify one candidate from its assembled evidence: evaluate the
    /// core certificates, derive quark vs antiquark from the
    /// determinant-line orientation, attach the provisional baryon flux
    /// under the certified winding, and fill isospin and charge only from
    /// their own certificates.  Never throws; missing evidence is a named
    /// failed certificate.
    [[nodiscard]] QuarkRead classifyQuark(
        const QuarkCandidateEvidence &evidence) const;

    /// `classifyQuark` over a candidate stream, in input order.
    [[nodiscard]] std::vector<QuarkRead> classifyQuarks(
        const std::vector<QuarkCandidateEvidence> &candidates) const;

    /// `classifyQuark` through the `AnalyticCache` contract: keyed by the
    /// color band's cell-vertex set (kind `kCacheKind`, parameter =
    /// `evidenceFingerprint`), served while the band's star is untouched
    /// and the evidence fingerprint matches, recomputed and re-stored
    /// otherwise.  Cached results equal cold recomputation.
    [[nodiscard]] QuarkRead classifyQuarkCached(
        cobordism::AnalyticCache &cache,
        const QuarkCandidateEvidence &evidence) const;

    /// Order-sensitive content fingerprint of the decision-relevant
    /// evidence and the threshold configuration (the cache parameter): a
    /// change in either recomputes rather than serving a stale verdict.
    [[nodiscard]] std::uint64_t evidenceFingerprint(
        const QuarkCandidateEvidence &evidence) const;

    // ── conjugate-pair conservation ─────────────────────────────────────

    /// Verify pair conservation of a conjugate creation path from the two
    /// endpoint reads: total certified winding, total baryon flux, and
    /// total parity (see `ConjugatePairRead`).  A singular leg (unknown
    /// winding) leaves the totals unknown.
    [[nodiscard]] ConjugatePairRead conjugatePair(const QuarkRead &first,
                                                  const QuarkRead &second) const;

    // ── even sectors: octet bilinear read + gluon/meson/diquark ─────────

    /// `AnalyticCache` kind string of a cached octet bilinear read.
    static constexpr const char *kOctetCacheKind = "octet-bilinear-read";

    /// The quasi-free traceless-bilinear (octet) read of three declared
    /// color modes of a carried covariance (see `OctetBilinearRead` for the
    /// exact identities).  Every quantity is a finite exact Wick sum
    /// evaluated through the public `CovarianceState` reads; dense Fock
    /// objects never appear on this path, and adding vacuum-embedded
    /// microscopic modes leaves the read unchanged.
    /// @throws std::invalid_argument unless `colorModes` names exactly
    ///   three distinct in-range modes.
    [[nodiscard]] OctetBilinearRead octetBilinearRead(
        const quantum::CovarianceState &state,
        const std::vector<std::size_t> &colorModes) const;

    /// `octetBilinearRead` through the `AnalyticCache` contract: keyed by
    /// the caller's component vertex set (kind `kOctetCacheKind`,
    /// parameter = `octetFingerprint` — the covariance hash, the declared
    /// modes, and the sign-extraction threshold), served while the
    /// component's star is untouched and the fingerprint matches (a Γ
    /// change is a state change), recomputed and re-stored otherwise.
    /// Cached results equal cold recomputation.
    [[nodiscard]] OctetBilinearRead octetBilinearReadCached(
        cobordism::AnalyticCache &cache,
        const std::vector<std::uint64_t> &componentVertexIds,
        const quantum::CovarianceState &state,
        const std::vector<std::size_t> &colorModes) const;

    /// Order-sensitive content fingerprint of an octet-read request: the
    /// exact covariance hash, the declared color modes, and the decision
    /// thresholds (the cache parameter of `octetBilinearReadCached`).
    [[nodiscard]] std::uint64_t octetFingerprint(
        const quantum::CovarianceState &state,
        const std::vector<std::size_t> &colorModes) const;

    /// Classify one gluon candidate from its assembled evidence: certified
    /// even parity, a nonzero certified octet excitation with machine-level
    /// octet purity, accepted rank-three lifetime transports under the
    /// leakage cap, a certified zero total determinant winding, and
    /// sufficient persistence.  Never throws; missing evidence is a named
    /// failed certificate.
    [[nodiscard]] GluonRead classifyGluon(
        const GluonCandidateEvidence &evidence) const;

    /// Classify one meson candidate: a certified quark + certified
    /// antiquark pair (order-insensitive), even composite parity (exact
    /// constituent product), color-singlet pairing, and zero total
    /// certified winding/flux (see `MesonRead`).
    [[nodiscard]] MesonRead classifyMeson(
        const CompositeCandidateEvidence &evidence) const;

    /// Classify one diquark candidate: two certified quarks, even
    /// composite parity, a certified anti-triplet wedge occupation, and
    /// the preserved constituent baryon flux B = 2/3 (see `DiquarkRead`).
    [[nodiscard]] DiquarkRead classifyDiquark(
        const CompositeCandidateEvidence &evidence) const;

    // ── three-cluster sector: baryons and the proton certificate ────────

    /// The bound-supercomponent search: for each next-modular-level
    /// component, report which certified quark candidates it contains,
    /// their shared lifetime window, and the containment and
    /// mutual-transport certificates.  One read is emitted per component
    /// containing at least one candidate, in input order.  `found` requires
    /// the component to sit at a strictly higher modular level than every
    /// constituent, to contain exactly three certified quark candidates,
    /// and for their lifetimes to overlap and their mutual transports to
    /// stay bounded.  Never throws; missing evidence is a named failed
    /// certificate.
    [[nodiscard]] std::vector<BoundSupercomponentRead>
    boundSupercomponentSearch(
        const std::vector<ComponentRead> &nextLevelComponents,
        const std::vector<BoundCandidateEvidence> &candidates) const;

    /// One refinement sample of the mass-radius battery, read through the
    /// validated context exactly as `EmergentRadius` / `EmergentMass` do
    /// (`RegisterContext::interiorHinges`): the dual and primal radii, the
    /// intensive shell mass, the curvature-weight participation ratio, and
    /// the per-shell curvature-weight profile.  Read-only; nothing is
    /// recomputed and no solver is called.
    [[nodiscard]] static ScaleProfileSample scaleProfileSample(
        const RegisterContext &ctx);

    /// The refinement-window certificate over `ScaleProfileSample`s: a
    /// finite emergent radius plus the refinement stability of every
    /// dimensionless channel (see `ScaleProfileRead`, and the file banner
    /// section "Form factors" — nothing here is a form factor and no
    /// dimensionful mass is ever emitted).
    [[nodiscard]] ScaleProfileRead scaleProfile(
        const std::vector<ScaleProfileSample> &samples) const;

    /// Classify one three-cluster candidate and evaluate the proton
    /// certificate: the two structural gates, the color volume and Gram
    /// determinant with the wedge built once, the net-color-flux
    /// diagnostic, the summed certified winding and graded parity, the
    /// constituent flavor and charge reads, the 2π character and (when a
    /// continuum spin claim is declared) the spin lift, the sharp
    /// total-space spin certificate, and the refinement-window scale reads.
    /// Returns "no-baryon", "baryon-candidate", "certified-proton", or
    /// "quasi-free-sharp-spin-obstruction" with every failed or unknown
    /// certificate named.  Never throws.
    [[nodiscard]] BaryonRead classifyBaryon(
        const BaryonCandidateEvidence &evidence) const;

    /// `classifyBaryon` over the `boundSupercomponentSearch` result: for
    /// every binding that grouped exactly three certified constituents,
    /// assemble the evidence bundle from the constituent verdicts the
    /// caller already produced and classify it.  One `BaryonRead` per such
    /// binding, in `bindings` order.  A binding that grouped a different
    /// number of certified constituents emits nothing; the missing legs are
    /// never padded with default reads, which would report a structural gap
    /// the geometry did not have.
    ///
    /// Only the evidence the caller has travels: the binding, the three
    /// `QuarkRead`s (`BoundSupercomponentRead::quarkIndices` indexes
    /// `constituentReads`), and the bound component's persistence lifetime.
    /// The color columns, the octet flux read, the rotation character, the
    /// quasi-free spin reads, the swept covariance-only class and the
    /// refinement-window samples are left absent, so `classifyBaryon` names
    /// each gap rather than presuming it.  A caller holding any of them
    /// classifies through `classifyBaryon` directly.
    ///
    /// `boundLifetimes` is the bound component's persistence lifetime per
    /// binding, in `bindings` order; empty leaves every one unknown (NaN).
    /// Read-only, and never throws on missing evidence.
    /// @throws std::invalid_argument when `boundLifetimes` is neither empty
    ///   nor exactly as long as `bindings`, or when a binding indexes a
    ///   constituent outside `constituentReads` (a caller error, never a
    ///   silently dropped verdict).
    [[nodiscard]] std::vector<BaryonRead> classifyBoundSupercomponents(
        const std::vector<BoundSupercomponentRead> &bindings,
        const std::vector<QuarkRead> &constituentReads,
        const std::vector<double> &boundLifetimes = {}) const;

    // ── the emergent flavor doublet (no requested dimension) ────────────

    /// Search the candidate's band enumeration across frames for a stable
    /// transported two-state subclass: follow every band through
    /// consecutive frames via `SpectralFiberTracker::matchFibers`
    /// (certified continuations only, and unambiguous — two chains merging
    /// onto one band invalidate each other), report every stable subclass
    /// rank, and accept exactly one full-length rank-two subclass.
    /// `frames[t]` is the candidate's `ComponentBandRead` at time or scale
    /// sample t, one degree per call.  No dimension is requested.
    [[nodiscard]] FlavorDoubletRead flavorDoubletSearch(
        const std::vector<ComponentBandRead> &frames) const;

    // ── the reused Gauss-flux electric read ─────────────────────────────

    /// The Gauss-flux read on nested enclosing surfaces: for each enclosed
    /// vertex set, `EigenstateSynthesis(st, 2).gaussLawCharge(F, set,
    /// electricOnly)` — the closed-star boundary flux of the supplied
    /// field-strength 2-cochain `F` in canonical degree-2 cell order —
    /// then the consistency combination `gaussFluxConsistency`.  Read-only
    /// on the spacetime.
    /// @throws std::invalid_argument on an empty surface list;
    ///   std::runtime_error from the underlying read on a malformed `F`.
    [[nodiscard]] GaussFluxRead gaussFluxOnSurfaces(
        const std::shared_ptr<Spacetime> &st,
        const std::vector<std::complex<double>> &fieldStrength,
        const std::vector<std::vector<std::uint64_t>> &enclosedVertexSets,
        bool electricOnly = true) const;

    /// The pure consistency combination over precomputed per-surface
    /// fluxes (the spacetime path delegates here): consistent when at
    /// least `minEnclosingSurfaces` surfaces agree within
    /// `gaussTolerance` (max pairwise deviation and |Im| leakage).
    [[nodiscard]] GaussFluxRead gaussFluxConsistency(
        const std::vector<std::complex<double>> &fluxes,
        const std::vector<std::size_t> &surfaceVertexCounts = {},
        bool electricOnly = true) const;

    /// Nested enclosing vertex sets by breadth-first shell growth: returns
    /// exactly `shells` sets, with sets[0] the seed ids (deduplicated and
    /// restricted to vertices present in the complex) and sets[k] = sets[k−1]
    /// plus every vertex sharing an edge with it.  Nested by construction,
    /// since growth saturates at the whole one-skeleton component.
    /// Read-only.
    /// @throws std::invalid_argument on an empty seed, no seed vertex in
    ///   the complex, or shells < 1.
    [[nodiscard]] static std::vector<std::vector<std::uint64_t>>
    nestedEnclosures(const std::shared_ptr<Spacetime> &st,
                     const std::vector<std::uint64_t> &seedVertexIds,
                     std::size_t shells);

    // ── candidate tracking across scale/time ────────────────────────────

    // ── proposing cluster supports ──────────────────────────────────────

    /// The cluster supports both proposers offer, merged.
    ///
    /// Newman–Girvan modularity on the combinatorial one-skeleton is a
    /// heuristic proposal generator that does not see the complex Hodge
    /// weights and is subject to the modularity resolution limit.  Because
    /// that proposer is metric-blind while acceptance is metric-aware,
    /// supports the metric would find but modularity never proposes would
    /// never be tested at all.  The degree-zero band of the covariant
    /// operator is the weight-aware second proposer: its supports are the
    /// committors of the metastable decomposition the operator itself sees
    /// (`EffectiveTopology::components`), so they are proposed from the
    /// squared lengths and the connection and from nothing else.
    ///
    /// Both proposers only propose.  Neither may veto: a support offered by
    /// one and not the other is a proposal like any other, and acceptance
    /// stays conditioned on the independent separation, localization,
    /// leakage, persistence and refinement certificates.
    ///
    /// Two supports are one proposal when their cell-id sets are equal.  The
    /// result lists the modularity components in their input order first,
    /// then every band component no modularity component matched, in its
    /// input order; every proposal carries the proposers that offered it and
    /// its largest Jaccard overlap with the other proposer's supports.  An
    /// empty support is dropped, since there is nothing on it to certify.
    ///
    /// Read-only and pure: it consumes two caller-produced reads, calls no
    /// solver and touches no spacetime.
    [[nodiscard]] static std::vector<ClusterSupportProposal> proposeSupports(
        const std::vector<ComponentRead> &modularityComponents,
        const EffectiveComponentPartition &bandComponents);

    /// Track candidates across frames by their color bands, delegating to
    /// `SpectralFiberTracker::matchFibers` on the evidence bands with an
    /// `overlapThreshold` independent of `doubletOverlapThreshold`, under
    /// certified-continuation semantics.  Entry (fromIndex, toIndex)
    /// indexes the input evidence lists.
    [[nodiscard]] static std::vector<FiberMatchRead> trackCandidates(
        const std::vector<QuarkCandidateEvidence> &from,
        const std::vector<QuarkCandidateEvidence> &to,
        double overlapThreshold = 0.5);

  private:
    ParticleClustersConfig cfg_{};

    /// One core certificate evaluation: append `name` to `failed` when
    /// `passed` is false; returns `passed` (shared bookkeeping).
    static bool gate(bool passed, const char *name,
                     std::vector<std::string> &failed);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_PARTICLECLUSTERS_H
