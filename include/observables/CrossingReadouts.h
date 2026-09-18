// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_CROSSING_READOUTS_H
#define TESSERA_OBSERVABLES_CROSSING_READOUTS_H

// # World-tube crossing readouts
//
// Four observables, all read from one object — a band world tube crossing a
// surface — and differing only in how the crossings are summed.
//
// Reference: Ambjorn, Goerlich, Jurkiewicz, Loll, "Nonperturbative Quantum
// Gravity", arXiv:1203.3591.
//
// ## The surfaces
//
// A cobordism supplies its own reference surface. `tau(x)` is the Lorentzian
// distance from the incoming boundary M0, complex like the geometry that
// defines it, and the surfaces are the level sets
// `Sigma_t = {x : Re tau(x) = t}`. The slicing is geometrically selected: tau
// is fixed by the cobordism and its incoming boundary, never by a coordinate,
// so `temporalFunction` reads the 1-skeleton and the stored complex edge
// lengths rather than `Vertex::getTime`.
//
// The assumptions are certificates. `Re tau` must be a certified temporal
// function — strictly increasing along every future-directed causal path — and
// a causal cycle, a nonregular level, a null normal, or a failed transversality
// makes the readout refuse with the failure named
// (`TemporalFunctionRead::failedCertificates`,
// `TubeCrossingRead::failedCertificates`). Nothing here labels a crossing it
// could not certify.
//
// ## The crossing decomposition
//
// Where a tube `c` crosses `Sigma_t`, its crossing set `C(c)` is the set of
// edges whose endpoints the level separates. The perpendicular projection is
// the band-weighted increment of the temporal function,
//
//     pi_perp(c) = sum_{e in C(c)} mu_c(e) * dtau(e),   dtau(e) = tau(e+) - tau(e-)
//
// complex, with `e-` and `e+` the past and future endpoints. `dtau` is built
// from the squared lengths alone and never contains the connection. The weight
// `mu_c(e)` is the band density on `e`, formed bilinearly from the band's
// matched left and right frames (the projector diagonal `P = Phi Psi^dagger W`)
// and normalized to sum to one over the crossing set. It is gauge-invariant: a
// local C* gauge factor acts on right frames by `g^-1` and on left frames by
// `g`, cancelling in the bilinear product. `pi_perp` is unchanged by relabeling
// the level values, since only increments of tau enter.
//
// A crossing is admissible only if the band passes the positivity certificate
// (`positiveSignature == rank && negativeSignature == 0`; a band that fails it
// supplies no covariance and no particle reading) and only if the crossing is
// timelike and transversal, i.e. `Re pi_perp` is nonvanishing with a single
// sign across the crossing set. A spacelike, null, or grazing crossing has no
// particle reading. On an admissible crossing
//
//     sgn pi_perp := sgn Re pi_perp  in  {+1, -1}
//
// is canonical rather than conventional: a bare complex number has no preferred
// sign, but the time orientation and the temporal function select the real part
// of the increment as the causal component. A future-directed crossing carries
// +1 and a past-directed crossing -1.
//
// ## The two sums
//
//     m_x(Sigma_t) = kappa_m * sum_c |pi_perp(c)|          (incoherent)
//     B(Sigma_t)   = (1/3) * sum_c sgn pi_perp(c)          (coherent)
//
// the second over certified quark tubes. Mass takes moduli and nothing cancels;
// baryon number takes signs and opposites cancel. Three forward quark tubes
// give B = 1; a quark and an antiquark tube give B = 0 while carrying twice the
// crossing mass of one constituent. The one-third per quark tube makes the
// coherent sum consistent with the independent determinant-line proposal
// B = nu/3, so the quantization of B follows from counting signed thirds rather
// than being assumed.
//
// `kappa_m` is one declared calibration with dimensions of mass per length. It
// ships as 1.0 with `massCalibrated == false` and units labeled "uncalibrated":
// before calibration only ratios are meaningful, and this header never emits a
// dimensionful physical mass. `m_x` is the crossing-mass functional; additivity
// over crossings and strict positivity per admissible crossing are properties
// of that functional by construction, not asserted laws of physical mass.
// Massless content lies outside its domain: a null crossing is refused, not
// counted at zero.
//
// The crossing sign and the determinant-line winding must agree on every
// certified tube. A tube on which they disagree is a defect signal, reported in
// `BaryonCrossingRead::signDefects` and never silently resolved.
//
// ## The reference
//
// M0 supplies both the reference surface and the reference values: every
// readout is the difference between its value on `Sigma_t` and the same sum
// evaluated at M0. M0 is a boundary hypersurface carrying boundary data, not a
// quantum state, so a readout evaluated at M0 itself is zero by construction.
//
// ## Localizing charge: two distinct observables
//
// Each admissible crossing contributes its signed unit at its position on
// `Sigma_t`, so the crossings define a charge density `rho` on the surface.
//
//  1. Unconditional — the spectral charge-power profile
//
//         S(lambda) = <rho, P_lambda rho> / <rho, P_0 rho>
//
//     with `P_lambda` the eigenspace projectors of the slice Laplacian, so
//     degeneracies are handled and no eigenvector phase enters: S is basis- and
//     phase-invariant. The slice Laplacian is the discrete -grad^2, so lambda
//     is the squared momentum transfer rather than an analogy for it. S is an
//     incoherent power, the analogue of a structure factor. It is not the
//     electromagnetic form factor, because a squared overlap loses the sign and
//     the relative phase of the coherent matrix element. For a neutral system
//     the normalizing monopole vanishes, the normalized profile refuses, and
//     the unnormalized values are reported.
//
//  2. Conditional — the electromagnetic form factor `G_E(Q^2)`, a normalized
//     matrix element of the conserved U(1) current's charge density between
//     states of certified momentum transfer, with
//
//         <r^2> = -6 dG_E/dQ^2 |_{Q^2 = 0}
//
//     read only in a certified three-spatial-dimensional refinement regime: a
//     finite spectrum has no literal derivative at zero, so the slope needs a
//     small-Q^2 refinement extrapolation with stability and normalization
//     certificates. Neither the conserved-current certificate nor certified
//     momentum-transfer states exist here, so `ElectromagneticFormFactorRead`
//     reports the radius unavailable with the reason named. The spectral power
//     is never substituted for G_E.
//
// ## Boundaries
//
// Read-only observable: it reads a relaxed spacetime and caller-assembled band
// reads, never calls a solver, never materializes facets, never rebuilds a
// complex, and nothing here enters any emergence objective. Unmeasured values
// are NaN or an empty optional with the reason named, never zero.
//
// This is a different definition of mass and radius from the static
// `EmergentMass` and `EmergentRadius` battery, which reads hinge curvature and
// dual volume on one relaxed interior. These readouts are swept by a world tube
// across a level set and are differences against M0; the two are not
// interchangeable and neither is derived from the other.

#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "observables/Record.h"
#include "observables/SpectralFiber.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
class Spacetime;
}

namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// # CrossingReadoutsConfig
///
/// Every analysis parameter of the crossing readouts, echoed on each read so a
/// checkpoint carries the configuration that produced it.
struct CrossingReadoutsConfig {
  /// The declared mass calibration `kappa_m`, in mass per length. Defaults to
  /// 1.0; see `massCalibrated`.
  double kappaMass = 1.0;
  /// False until `kappaMass` is fixed against a physical input. While false the
  /// crossing mass is reported in uncalibrated units and only ratios are
  /// meaningful.
  bool massCalibrated = false;
  /// `|Re pi_perp|` must exceed this for a crossing to count as timelike and
  /// transversal; at or below it the crossing is grazing and refused.
  double signTolerance = 1e-12;
  /// Two slice-Laplacian eigenvalues within this absolute distance share an
  /// eigenspace projector `P_lambda` (degeneracy grouping).
  double degeneracyTolerance = 1e-9;
  /// `|<rho, P_0 rho>|` must exceed this for the profile to normalize; at
  /// or below it the system is neutral and the normalized profile refuses.
  double monopoleTolerance = 1e-12;

  /// Record echo of every threshold above.
  [[nodiscard]] Record toRecord() const;
};

/// # TemporalFunctionRead
///
/// The complex Lorentzian distance `tau` from the incoming boundary M0, with
/// the temporal-function certificate. `tau` is intrinsic: it reads the
/// 1-skeleton and the stored complex edge lengths, not a vertex coordinate.
struct TemporalFunctionRead {
  /// Vertex ids in ascending order.
  std::vector<std::uint64_t> vertices{};
  /// `tau` per vertex, parallel to `vertices`. A vertex the causal relation
  /// does not reach from M0 carries NaN.
  std::vector<std::complex<double>> tau{};
  /// Combinatorial layer, the hop distance from M0 in the 1-skeleton, parallel
  /// to `vertices`. This is the time orientation M0 induces: an edge from a
  /// lower to a higher layer points to the future.
  std::vector<std::size_t> layer{};
  /// Whether `Re tau` is a certified temporal function.
  bool certified = false;
  /// Named failures: "empty-boundary", "unreachable-vertices",
  /// "causal-cycle", "nonmonotone-temporal-function", "null-causal-edge".
  std::vector<std::string> failedCertificates{};
  /// Smallest `Re dtau` over the future-directed causal edges; NaN when none
  /// exist. Positive exactly when the monotonicity certificate holds.
  double minCausalIncrement = std::numeric_limits<double>::quiet_NaN();
  /// Number of future-directed causal (timelike or null) edges examined.
  std::size_t causalEdgeCount = 0;
  /// Number of vertices the causal relation never reached from M0.
  std::size_t unreachableCount = 0;

  /// Lookup helper: `tau` of one vertex, or NaN when unknown.
  [[nodiscard]] std::complex<double> at(std::uint64_t vertex) const;
  /// Checkpoint serialization. Complex leaves split into `_re` and `_im`;
  /// unknowns serialize as null or NaN, never zero.
  [[nodiscard]] Record toRecord() const;
};

/// # WorldTubeInput
///
/// One persistent band tracked across cobordism frames, as the crossing
/// readouts consume it. The band is a `SpectralFiber` (matched frames plus
/// certificate) and the winding is the determinant-line integer a `QuarkRead`
/// carries, so nothing here re-derives a band or a transport.
struct WorldTubeInput {
  /// Caller's stable label for the tube (a component hash, typically).
  std::string tubeId{};
  /// The tracked band: matched left/right frames, weights, certificate.
  SpectralFiber band{};
  /// The tube's traversal direction against the temporal function: +1 for a
  /// future-directed tube, -1 for the reversed tube. This is what makes a
  /// crossing past-directed, since
  /// `dtau(e) = orientation * (tau(future) - tau(past))`; reversing a tube
  /// flips `sgn pi_perp` and sends `B = +1/3` to `B = -1/3`. The geometry alone
  /// cannot supply this sign: `Re tau` increases toward the future everywhere,
  /// so a past-directed reading is a property of the tube, not the surface.
  int orientation = +1;
  /// The certified determinant-line winding `nu`; empty when no closure was
  /// certified. Present values are cross-checked against the crossing sign.
  std::optional<int> determinantWinding{};
  /// Whether the caller certified this tube as a quark tube. Only certified
  /// quark tubes enter the baryon sum; every admissible crossing enters the
  /// crossing mass.
  bool certifiedQuarkTube = false;
};

/// # TubeCrossingRead
///
/// One tube's crossing of one level set, with its admissibility certificate.
struct TubeCrossingRead {
  /// The tube's label.
  std::string tubeId{};
  /// The level `t` this crossing was read at.
  double level = std::numeric_limits<double>::quiet_NaN();
  /// The crossing set `C(c)`: each entry is the two endpoint vertex ids of a
  /// separated edge, past endpoint first.
  std::vector<std::array<std::uint64_t, 2>> crossingEdges{};
  /// The band density `mu_c(e)` on each crossing edge, parallel to
  /// `crossingEdges` and summing to one; empty when the crossing set is empty.
  std::vector<double> density{};
  /// `pi_perp(c)`, complex. NaN when the crossing could not be formed.
  std::complex<double> perpendicular{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// `sgn Re pi_perp` on an admissible crossing; 0 means unknown, since an
  /// inadmissible crossing has no sign.
  int sign = 0;
  /// Whether every admissibility certificate held.
  bool admissible = false;
  /// Named failures: "band-positivity", "band-unaccepted", "empty-crossing",
  /// "grazing-crossing", "mixed-sign-crossing", "null-crossing",
  /// "degree-zero-band", "uncertified-temporal-function".
  std::vector<std::string> failedCertificates{};

  /// Checkpoint serialization.
  [[nodiscard]] Record toRecord() const;
};

/// # CrossingMassRead
///
/// The crossing-mass functional `m_x` on one level, as a difference against M0.
/// Not a dimensionful physical mass while `calibrated` is false.
struct CrossingMassRead {
  /// The level `t`.
  double level = std::numeric_limits<double>::quiet_NaN();
  /// `m_x(Sigma_t) - m_x(M0)`, the reported readout. NaN when no admissible
  /// crossing was found on either surface.
  double crossingMass = std::numeric_limits<double>::quiet_NaN();
  /// The raw sum on `Sigma_t` before the M0 subtraction.
  double levelSum = std::numeric_limits<double>::quiet_NaN();
  /// The same sum evaluated at M0 (the reference value subtracted).
  double referenceSum = std::numeric_limits<double>::quiet_NaN();
  /// The declared `kappa_m` this reading used.
  double kappaMass = 1.0;
  /// Whether `kappa_m` was calibrated against a physical input.
  bool calibrated = false;
  /// "uncalibrated" while `calibrated` is false; the caller's declared unit
  /// string afterwards.
  std::string units{"uncalibrated"};
  /// Number of admissible crossings summed on `Sigma_t`.
  std::size_t admissibleCrossings = 0;
  /// Number of tubes that were refused, with their reasons preserved on the
  /// per-tube reads.
  std::size_t refusedCrossings = 0;

  [[nodiscard]] Record toRecord() const;
};

/// # BaryonCrossingRead
///
/// The coherent one-third sum over certified quark tubes, with the
/// determinant-line cross-check.
struct BaryonCrossingRead {
  /// The level `t`.
  double level = std::numeric_limits<double>::quiet_NaN();
  /// `B(Sigma_t) - B(M0)`; empty when no certified quark tube supplied an
  /// admissible crossing, meaning unknown rather than zero.
  std::optional<double> baryonNumber{};
  /// The raw one-third sum on `Sigma_t` before the M0 subtraction.
  std::optional<double> levelSum{};
  /// The same sum at M0.
  std::optional<double> referenceSum{};
  /// Number of certified quark tubes contributing a sign.
  std::size_t quarkTubes = 0;
  /// Tubes whose crossing sign disagrees with their determinant-line winding
  /// sign, by name. A defect signal: reported, never resolved, and never
  /// silently dropped from the sum.
  std::vector<std::string> signDefects{};
  /// Tubes whose winding was certified and agreed with the crossing sign.
  std::size_t windingAgreements = 0;

  [[nodiscard]] Record toRecord() const;
};

/// # ChargePowerProfileRead
///
/// The spectral charge-power profile `S(lambda)` on one level: an incoherent
/// power, the analogue of a structure factor, and never the electromagnetic
/// form factor.
struct ChargePowerProfileRead {
  /// The level `t`.
  double level = std::numeric_limits<double>::quiet_NaN();
  /// Distinct slice-Laplacian eigenvalues, ascending, grouped at
  /// `degeneracyTolerance`.
  std::vector<double> eigenvalues{};
  /// `<rho, P_lambda rho>` per eigenvalue, parallel to `eigenvalues`.
  std::vector<double> power{};
  /// `S(lambda) = power / <rho, P_0 rho>`, parallel to `eigenvalues`; empty
  /// when the monopole vanished and the normalization refused.
  std::vector<double> normalizedPower{};
  /// `<rho, P_0 rho>`, the normalizing monopole term.
  double monopole = std::numeric_limits<double>::quiet_NaN();
  /// Whether the normalized profile was produced.
  bool normalized = false;
  /// Named failure when `normalized` is false: "neutral-system", meaning the
  /// monopole vanished, or "empty-slice".
  std::vector<std::string> failedCertificates{};
  /// Number of slice nodes, i.e. crossing edges carrying charge.
  std::size_t sliceNodes = 0;

  [[nodiscard]] Record toRecord() const;
};

/// # ElectromagneticFormFactorRead
///
/// The conditional electromagnetic form factor and the charge radius. Neither
/// a conserved U(1) current nor momentum-transfer states are certified here, so
/// this read refuses: `available` is false and the radius is unavailable with
/// the reason named. The spectral charge-power profile is never substituted for
/// `G_E`.
struct ElectromagneticFormFactorRead {
  /// Whether a certified `G_E` was produced. Currently always false.
  bool available = false;
  /// `<r^2> = -6 dG_E/dQ^2|_0` under the certificates; empty otherwise.
  std::optional<double> chargeRadiusSquared{};
  /// The certificates that are missing, named:
  /// "no-certified-conserved-current", "no-certified-momentum-states",
  /// "no-refinement-extrapolation".
  std::vector<std::string> failedCertificates{};
  /// Restates on every read that the unconditional spectral power is a
  /// structure factor and is not this observable.
  std::string note{"spectral charge-power profile is an incoherent structure factor and is never substituted for G_E"};

  [[nodiscard]] Record toRecord() const;
};

/// # CrossingReadouts
///
/// The world-tube crossing readouts, as static reads over a relaxed spacetime
/// and caller-assembled band reads.
class CrossingReadouts {
  public:
    /// Record key of the assembled overlay block.
    static constexpr std::string_view kRecordKey = "crossing_readouts";
    /// Serialization schema of the overlay block.
    static constexpr int kSchemaVersion = 1;

    /// The complex Lorentzian distance `tau` from the incoming boundary
    /// `m0Vertices`, with its temporal-function certificate. Reads the
    /// 1-skeleton and the stored complex edge lengths only.
    ///
    /// The time orientation is the one M0 induces: the combinatorial layer (hop
    /// distance from M0) orders every edge, and `tau` accumulates the proper
    /// time `sqrt(-z)` of future-directed causal edges along the path maximizing
    /// `Re tau`, i.e. the discrete Lorentzian distance from a hypersurface as a
    /// supremum over causal curves. A causal edge inside one layer, a causal
    /// cycle, or a non-increasing `Re tau` fails the certificate and is
    /// named.
    /// @throws std::invalid_argument when `spacetime` is null.
    [[nodiscard]] static TemporalFunctionRead temporalFunction(
        const std::shared_ptr<Spacetime> &spacetime,
        const std::vector<std::uint64_t> &m0Vertices,
        const CrossingReadoutsConfig &cfg = {});

    /// The band density `mu_c(e)` induced on the 1-skeleton by one band: the
    /// projector diagonal `|P_ii|` of `P = Phi Psi^dagger W` carried to edges. A
    /// degree-1 band supplies it directly; a degree-k band spreads each k-cell's
    /// density uniformly over that cell's boundary edges. Keys are the endpoint
    /// pair in ascending vertex order. Empty for a degree-zero band, which
    /// carries no edge density.
    [[nodiscard]] static std::map<std::array<std::uint64_t, 2>, double>
    bandEdgeDensity(const SpectralFiber &band);

    /// One tube's crossing of the level `Re tau = level`.
    [[nodiscard]] static TubeCrossingRead crossing(
        const WorldTubeInput &tube, const TemporalFunctionRead &temporal,
        double level, const CrossingReadoutsConfig &cfg = {});

    /// The crossing-mass functional on `level`, as the difference against
    /// the reference level `m0Level` (the M0 surface).
    ///
    /// `m0Level` is the level at which the reference sum is evaluated. The M0
    /// surface itself sits at `Re tau = 0`, which passes through M0's vertices
    /// and is therefore nonregular: no crossing is admissible there and the
    /// reference sum is zero. A tube that has not yet crossed contributes
    /// nothing, which is why reading `level == m0Level` returns exactly zero.
    [[nodiscard]] static CrossingMassRead crossingMass(
        const std::vector<WorldTubeInput> &tubes,
        const TemporalFunctionRead &temporal, double level, double m0Level,
        const CrossingReadoutsConfig &cfg = {});

    /// The coherent one-third baryon sum on `level`, as the difference
    /// against `m0Level`, with the determinant-line cross-check.
    [[nodiscard]] static BaryonCrossingRead baryonNumber(
        const std::vector<WorldTubeInput> &tubes,
        const TemporalFunctionRead &temporal, double level, double m0Level,
        const CrossingReadoutsConfig &cfg = {});

    /// The spectral charge-power profile on `level`.
    [[nodiscard]] static ChargePowerProfileRead chargePowerProfile(
        const std::vector<WorldTubeInput> &tubes,
        const TemporalFunctionRead &temporal, double level,
        const CrossingReadoutsConfig &cfg = {});

    /// The conditional electromagnetic form factor. It refuses, naming the
    /// certificates that are not available.
    [[nodiscard]] static ElectromagneticFormFactorRead formFactor(
        const ChargePowerProfileRead &profile,
        const CrossingReadoutsConfig &cfg = {});

    /// Every readout on one level, assembled as the overlay block.
    [[nodiscard]] static Record overlayRecord(
        const std::vector<WorldTubeInput> &tubes,
        const TemporalFunctionRead &temporal, double level, double m0Level,
        const CrossingReadoutsConfig &cfg = {});
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_CROSSING_READOUTS_H
