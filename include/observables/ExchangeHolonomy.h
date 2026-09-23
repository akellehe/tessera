// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_EXCHANGEHOLONOMY_H
#define TESSERA_OBSERVABLES_EXCHANGEHOLONOMY_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cobordism/Certificate.h"
#include "observables/SpectralFiber.h"

namespace tessera::observables {

/// Analysis parameters of the exchange and rotation holonomy reads. Every
/// threshold selects which reads are certified, never which value is reported:
/// a failed threshold yields an uncertified read, never a different sign.
struct ExchangeHolonomyConfig {
  /// A certified transport step needs every singular value of its overlap
  /// matrix at least this large. A leaking transfer — the tracked subspace
  /// turning away from its successor — invalidates the read before polar
  /// normalization.
  double leakFloor = 1e-6;
  /// A certified transport step needs overlap conditioning
  /// sigma_max/sigma_min at most this large.
  double conditionCap = 1e8;
  /// Certificate tolerance on the composed loop's unitarity residual
  /// ||U^dagger U - I||_F / sqrt(r) and on | |chi| - 1 | of a character.
  double unitaryTolerance = 1e-9;
  /// A definite characterSign (+1/-1) is reported only when the normalized
  /// character sits within this distance of +1 or -1 (and the certificate
  /// holds); otherwise characterSign = 0 and only the complex value speaks.
  double signTolerance = 1e-6;
  /// Minimum subspace overlap for a certified block continuation in
  /// :func:`ExchangeHolonomy::blockPermutation`. Matches
  /// `SpectralFiberConfig::trackOverlapThreshold`; the matching itself is
  /// delegated to `SpectralFiberTracker::matchFibers`.
  double blockMatchThreshold = 0.5;
  /// Incremental steps of a lifted SO(d) loop must keep their rotation angle at
  /// least this far below pi, the double-cover branch cut. A larger step makes
  /// the principal lift ambiguous and leaves the loop character uncertified
  /// rather than guessed.
  double liftAngleMargin = 1e-6;
  /// Cap on the verified SO(d) cocycle residual
  /// max_t ||g_ij g_jk g_ki - I||_F of :func:`ExchangeHolonomy::spinLift`: the
  /// premise under which the GF(2) obstruction decision is exact.
  double cocycleTolerance = 1e-9;
};

/// Which physical question a Berry-cancelled character answers. Particle
/// exchange and physical rotation are separate channels in the interface and
/// the report: the same interferometric machinery runs both, a read is always
/// tagged with the channel it certifies, and
/// :func:`ExchangeHolonomy::doublyCancelledRatio` refuses mislabeled inputs.
enum class HolonomyChannel {
  /// A configuration-space loop that permutes identical clusters.
  ParticleExchange,
  /// The constructed total-space spin-holonomy cycle: one global rotation
  /// of the whole carried cluster frame (never a per-hole Bloch product).
  PhysicalRotation,
};

/// One overlap-transport step of a closed loop: the singular-value data of the
/// r x r frame overlap before polar normalization, and whether the step met the
/// leak and conditioning thresholds.
struct TransportStepRead {
  /// Loop positions transported from / to (toIndex = (fromIndex+1) mod T).
  std::size_t fromIndex = 0;
  std::size_t toIndex = 0;
  /// Extreme singular values of the overlap matrix Phi_{t+1}^dagger W_t
  /// Phi_t (restricted to shared cells on the fiber path).
  double minSingularValue = std::numeric_limits<double>::quiet_NaN();
  double maxSingularValue = std::numeric_limits<double>::quiet_NaN();
  /// sigma_max / sigma_min (infinity when sigma_min = 0).
  double conditioning = std::numeric_limits<double>::quiet_NaN();
  /// Whether the step met `leakFloor` and `conditionCap`.
  bool certified = false;
};

/// # LoopHolonomyRead
///
/// The certified overlap transport of one tracked frame around one closed
/// configuration-space loop:
///
///   R_t = polar(Phi_{t+1 mod T}^dagger W_t Phi_t),
///   U_gamma = R_{T-1} ... R_1 R_0 .
///
/// The transport is cyclic: the last step closes the loop back onto the t = 0
/// frame, so U_gamma maps the base frame's gauge to itself and det U_gamma is
/// invariant under every in-band frame rotation Phi_t -> Phi_t g_t, since polar
/// decomposition is unitarily equivariant and the g's cancel around the cycle
/// up to conjugation.
///
/// `determinant` is the raw loop determinant chi_raw = det U_gamma. It contains
/// the ordinary path-dependent Berry phase of the reference motion and is not
/// an exchange sign by itself; only the interferometric ratio of
/// :func:`ExchangeHolonomy::exchangeCharacter` (against a matched
/// non-exchanging reference loop) or
/// :func:`ExchangeHolonomy::rotationCharacter` (against a matched co-moving
/// non-rotating reference) is.
struct LoopHolonomyRead {
  /// The composed loop holonomy U_gamma (rank x rank).  Empty when the
  /// read was structurally invalidated (rank change along a fiber track).
  Eigen::MatrixXcd holonomy{};
  /// chi_raw = det U_gamma, Berry phase included; not a sign by itself.
  std::complex<double> determinant{std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::quiet_NaN()};
  /// Number of loop frames T (= number of cyclic transport steps).
  std::size_t steps = 0;
  /// Tracked band rank r.
  std::size_t rank = 0;
  /// Per-step overlap certificates.
  std::vector<TransportStepRead> stepReads{};
  /// ||U^dagger U - I||_F / sqrt(r) of the composed holonomy.
  double unitarityResidual = std::numeric_limits<double>::quiet_NaN();
  /// min over steps of the overlap's smallest singular value (the loop's
  /// worst leak) and max over steps of the overlap conditioning.
  double minStepSingularValue = std::numeric_limits<double>::quiet_NaN();
  double conditioning = std::numeric_limits<double>::quiet_NaN();
  /// True when a fiber on the loop carried an uncertified band certificate,
  /// i.e. its isolating gap closed. The read is then reported but not
  /// certified: a closing gap invalidates rather than flipping a sign.
  bool uncertifiedBand = false;
  /// Certificate: `CertifiedNumerical` on the verified regime when every step
  /// met the thresholds and no band was uncertified; `HeuristicDiscovery`,
  /// which never holds, otherwise.
  cobordism::Certificate certificate{};
};

/// # HolonomyCharacterRead
///
/// The interferometric (Berry-cancelled) character of one loop against its
/// matched reference loop:
///
///   chi_hat_F = det U_loop / det U_reference
///
/// The report keeps the three phase channels separate: `rawLoopDeterminant`
/// (exchange or rotation motion plus Berry), `referenceDeterminant` (the Berry
/// reference motion alone), and `character` (the cancelled ratio, the dynamical
/// certificate).
struct HolonomyCharacterRead {
  /// Which physical question this character answers. Exchange and rotation are
  /// separate channels by construction.
  HolonomyChannel channel = HolonomyChannel::ParticleExchange;
  /// det U of the exchange or rotation loop, Berry phase included.
  std::complex<double> rawLoopDeterminant{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// det U of the matched non-exchanging or co-moving non-rotating reference
  /// loop: the Berry reference channel on its own.
  std::complex<double> referenceDeterminant{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// chi_hat = rawLoopDeterminant / referenceDeterminant.
  std::complex<double> character{std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()};
  /// -1 or +1 when the certificate holds and |character -+ 1| is within
  /// `signTolerance`; 0 otherwise. An uncertified read never emits a sign.
  int characterSign = 0;
  /// |character - characterSign| when a sign was emitted (NaN otherwise).
  double signResidual = std::numeric_limits<double>::quiet_NaN();
  /// Whether the two loops had the same step count (the timing premise of
  /// the cancellation identity).  A mismatch is reported and uncertified.
  bool timingMatched = false;
  /// Whether the two loops had equal tracked rank.
  bool ranksMatched = false;
  /// Certificate: holds only when both loop certificates hold, timing and
  /// ranks match, and | |character| - 1 | is within tolerance.
  cobordism::Certificate certificate{};
};

/// The occupancy declaration of one tracked cluster block: how much of the
/// block's fibre the many-body state actually occupies, and how the block's
/// support is built.
///
/// The exchange statistic is occupation parity, so the number a block
/// contributes to it is the number of occupied one-particle modes it carries,
/// not the rank of its fibre. A quark is one occupied mode of a colour-spin
/// fibre of rank six, and the two numbers give opposite signs.
struct ClusterOccupancy {
  /// n_b, the number of one-particle modes of this block's fibre that the
  /// state occupies. It is the number that enters the graded interchange law
  /// tau(a (x) b) = (-1)^{F_a F_b} b (x) a, so a quark or antiquark declares
  /// one, a meson or a diquark two, and a baryon three.
  std::size_t occupation = 1;
  /// k, the number of sheets the block's support carries: one for an
  /// unsheeted support and three for the adopted quark support. It is what
  /// decides whether the rank-parity cross-check applies, because a sheeted
  /// fibre has even rank and exchanging whole frames of even rank gives +1
  /// whatever the occupations are.
  std::size_t sheetCount = 1;
};

/// # BlockPermutationRead
///
/// The structural channel of the exchange experiment: persistent-component
/// matching of localized odd blocks around the loop, the extracted permutation,
/// its exact parities (the algebraic wedge sign from the exterior-algebra
/// grading), and the residual in-block motion left after the matched reference
/// loop is cancelled. Kept separate from the interferometric determinant
/// channel.
///
/// ## Which parity is the statistic
///
/// `occupationParity` is the exchange statistic. It is the sign of the
/// permutation the exchange induces on the OCCUPIED one-particle modes, which
/// is the graded interchange law of the exterior Fock functor applied to the
/// fermion numbers the clusters actually carry.
///
/// `rankParity` is the odd-rank determinant cross-check and is a separate,
/// independent number: it is the determinant of exchanging the whole fibre
/// frames, det pi_AB = (-1)^{r_A r_B}, which is exact as an identity about
/// frames but is only a hypothesis about particle statistics. The two minus
/// signs are never multiplied together, and `rankParityRetired` marks the case
/// the whitepaper retires the cross-check in: a sheeted support, where the
/// colour-spin fibre has even rank so the frame exchange gives +1 while the
/// occupation parity of a single occupied mode is -1.
struct BlockPermutationRead {
  /// blockPermutation[b] = index (in the t = 0 block list) the block at
  /// position b arrives at after one full loop.  Empty when uncertified.
  std::vector<std::size_t> blockPermutation{};
  /// The tracked blocks' ranks at t = 0.
  std::vector<std::size_t> blockRanks{};
  /// The declared occupation n_b of each block, in block order: the number of
  /// occupied one-particle modes the block carries.
  std::vector<std::size_t> blockOccupations{};
  /// The declared sheet count of each block's support, in block order.
  std::vector<std::size_t> blockSheetCounts{};
  /// Sign of `blockPermutation` as a permutation of block labels: +1 or -1, 0
  /// when uncertified. A combinatorial datum, not the exchange statistic: a
  /// rank-1 to rank-2 block swap has blockParity -1 but graded sign +1.
  int blockParity = 0;
  /// The exchange statistic: the graded sign the exterior Fock functor's
  /// interchange law attaches to the reordering, the product of
  /// (-1)^{n_a n_b} over the inversions of `blockPermutation` — the pairs
  /// a < b it sends to pi(a) > pi(b) — with n_b the block's declared
  /// occupation. For blocks of equal occupation it is the parity of the
  /// permutation induced on the occupied one-particle modes. +1 or -1; 0 when
  /// uncertified.
  int occupationParity = 0;
  /// The odd-rank determinant cross-check: the same product over the same
  /// inversions with the fibre ranks in place of the occupations, which is the
  /// determinant of exchanging the whole fibre frames,
  /// prod det pi_AB = prod (-1)^{r_a r_b}. Reported as an independent number
  /// and never multiplied into `occupationParity`. +1 or -1; 0 when
  /// uncertified or retired.
  int rankParity = 0;
  /// Whether the rank-parity cross-check was retired rather than reported: it
  /// is retired when any block declares a sheeted support, where the fibre has
  /// even rank and the frame exchange gives +1 whatever the occupations are.
  /// `rankParity` is then 0.
  bool rankParityRetired = false;
  /// Whether the reported `rankParity` agrees with `occupationParity`. False
  /// whenever the cross-check was retired or uncertified, and false when the
  /// two genuinely disagree, which is the case the whitepaper's falsifiable
  /// hypothesis fails in.
  bool rankParityAgrees = false;
  /// Optional composite-level view (when `composites` was supplied):
  /// compositePermutation[c] = composite that composite c's blocks landed
  /// in; empty when no grouping was supplied or when the blocks of some
  /// composite scattered over several targets (then also uncertified).
  std::vector<std::size_t> compositePermutation{};
  /// Sign of `compositePermutation` (+1/-1; 0 when absent/uncertified).
  int compositeParity = 0;
  /// Smallest matched subspace overlap used anywhere in the tracking.
  double minMatchOverlap = std::numeric_limits<double>::quiet_NaN();
  /// Residual in-block motion after reference cancellation:
  /// max over permutation cycles c of
  /// || U_cycle(c) * (U_ref(b_{L-1}) ... U_ref(b_0))^{-1} - I ||_F /
  /// sqrt(rank), where U_cycle composes the block transports along the
  /// cycle's full track and U_ref(b) is block b's closed reference-loop
  /// transport, multiplied in the cycle's visit order.  NaN when no
  /// reference was supplied (`cobordism::Certificate::kUnmeasured`).
  double residualInBlockMotion = std::numeric_limits<double>::quiet_NaN();
  /// Certificate: `StructureExact`, since parities are exact integers given the
  /// verified premise that every step's block matching is a certified bijection
  /// of equal-rank accepted bands; `HeuristicDiscovery` when that premise
  /// failed through gap closure, rank change, or ambiguous matching.
  cobordism::Certificate certificate{};
};

/// The Z2 character of a closed SO(d) loop lifted step-by-step to Spin(d):
/// +1 for a contractible loop, -1 for the nontrivial pi_1(SO(d)) class
/// (d >= 3).  The frame-level counterpart of the interferometric 2 pi
/// rotation certificate.
struct LoopLiftRead {
  /// +1 / -1 when certified; 0 when a step approached the pi branch cut
  /// or the lifted product failed to close on +-I.
  int character = 0;
  /// Largest incremental rotation angle encountered (branch-safety margin).
  double maxStepAngle = std::numeric_limits<double>::quiet_NaN();
  /// || S -+ I ||_F of the closed lifted product against the reported sign.
  double closureResidual = std::numeric_limits<double>::quiet_NaN();
  /// `StructureExact` when every step stayed below the branch margin and
  /// the product closed on +-I within tolerance; `HeuristicDiscovery`
  /// otherwise (never a guessed sign).
  cobordism::Certificate certificate{};
};

/// # SpinLiftRead
///
/// The SO(d) -> Spin(d) lift decision over Cech transition data on a
/// triangulated 2-complex, with the second Stiefel-Whitney obstruction:
/// per-triangle signs w_t = sign of lift(g_ij) lift(g_jk) lift(g_ki), the
/// exact GF(2) coboundary decision (does an edge-sign choice make every
/// triangle +1?), and — when the lift exists — one such choice.  This is
/// CONDITIONAL machinery for continuum spin claims on emergent
/// manifold-like regimes only; the abstract canonical
/// anticommutation-relation (CAR) Fock exchange algebra
/// requires no spin structure and no Kasteleyn orientation.
struct SpinLiftRead {
  /// Whether a consistent sign choice exists (w2 cohomologically trivial).
  bool liftExists = false;
  /// The negation of `liftExists` once certified. Kept explicit so an
  /// uncertified read can report neither.
  bool obstructed = false;
  /// Per-triangle cocycle signs (+1 / -1) in input triangle order.
  std::vector<int> triangleSigns{};
  /// A per-edge sign choice (+1 / -1, input edge order) under which every
  /// triangle's lifted product is +I.  Empty when obstructed/uncertified.
  std::vector<int> edgeSigns{};
  /// Verified SO cocycle residual max_t ||g_ij g_jk g_ki - I||_F (the
  /// structural premise) and the worst lifted-product deviation from +-I.
  double maxCocycleResidual = std::numeric_limits<double>::quiet_NaN();
  double maxLiftResidual = std::numeric_limits<double>::quiet_NaN();
  /// `StructureExact` (the GF(2) decision is exact given the verified
  /// cocycle premise and the documented pi-branch convention);
  /// `HeuristicDiscovery` when the premise failed.
  cobordism::Certificate certificate{};

  /// One-line human-readable summary.
  [[nodiscard]] std::string describe() const;
};

/// # ExchangeHolonomy
///
/// Berry-cancelled exchange statistics, the constructed total-space spin
/// holonomy cycle, and the conditional SO(d) -> Spin(d) lift.
///
/// ## Identities implemented
///
///  1. Certified cyclic overlap transport of an isolated tracked subspace:
///     `R_t = polar(Phi_{t+1 mod T}^dagger W_t Phi_t)` and
///     `U_gamma = R_{T-1} ... R_0`. It composes `SpectralFiber` frames, with
///     cells matched by sorted vertex-id tuple, or explicit frame paths; no
///     second subspace tracker is built here. det U_gamma is exactly invariant
///     under in-band frame rotations, vertex relabeling, and simplex
///     reorientation (a common row sign flip).
///  2. The interferometric exchange character
///     `chi_hat_F = det U_exchange / det U_reference` against a matched
///     non-exchanging reference loop with the same step count and rank. The raw
///     determinant contains an ordinary Berry phase and is never the exchange
///     sign; only the cancelled ratio is the dynamical certificate. Valid for
///     equal-step, equal-rank certified loops.
///  3. The structural permutation of persistent localized odd blocks, with
///     matching delegated to `SpectralFiberTracker::matchFibers`, its exact
///     parity through the exterior-algebra grading
///     (`quantum::OccupationBitset::permutationParity`, the algebraic wedge
///     sign, an exact integer), and the residual in-block motion after
///     reference cancellation. The algebraic and dynamical channels are
///     reported separately. The statistic is the parity of the permutation
///     induced on the OCCUPIED one-particle modes, declared per block by
///     `ClusterOccupancy`; the parity of the fibre ranks is the independent
///     odd-rank determinant cross-check and is reported beside it, retired on
///     a sheeted support, and never multiplied into it.
///  4. The total-space spin holonomy cycle as the canonical physical rotation
///     path: the Euclidean gamma layer, spin generators
///     `Sigma_ab = [gamma_a, gamma_b]/4`, the closed-form plane rotation
///     `exp(theta Sigma_ab) = cos(theta/2) I + sin(theta/2) gamma_a gamma_b`,
///     and the closed 2 pi cluster-frame loop with its matched co-moving
///     non-rotating reference. The rotation acts on the whole carried frame at
///     once, never as a product of per-hole or per-edge Bloch vectors. The
///     transverse frame makes the double cover interferometrically visible: a
///     frame polarized along the rotation axis does not precess and shows no
///     relative phase.
///  5. The total-space spin read `J^2 = sum_a (sum_i S_a^(i))^2` on
///     `(C^2)^(tensor n)`, whose oracle values are pinned: the proton
///     eigenstate `2|uud> - |udu> - |duu>` gives 3/4, the Delta `|uuu>` gives
///     15/4, and the product state `|uud>` gives 7/4.
///  6. The conditional Spin(d) lift: the principal rotation logarithm via the
///     real Schur plane decomposition, the closed-form plane-product lift
///     SO(d) -> Spin(d) for d = 3, 4, the Z2 character of a closed SO(d) loop,
///     and the second Stiefel-Whitney obstruction of Cech transition data with
///     the exact GF(2) coboundary decision (`cobordism::gf2Rank`).
///
/// ## Channel separation
///
/// Five channels are kept distinct in the interface and the reports.
///
///  (i)   Simplex reorientation is a common row sign flip, `reorientedFrames`,
///        under which every read is exactly invariant.
///  (ii)  Compilation ordering (mode order, vertex labels) is a compilation
///        artifact. Reads here match cells by vertex tuple and permute rows, so
///        they are exactly invariant; any bookkeeping parity is supplied by
///        `EdgeModeRegistry` or `OccupationBitset`, never by this class.
///  (iii) Particle exchange is `HolonomyChannel::ParticleExchange` plus the
///        structural `occupationParity`.
///  (iv)  Berry reference motion is the `referenceDeterminant` channel,
///        reported raw and cancelled, and never interpreted alone.
///  (v)   Physical rotation is `HolonomyChannel::PhysicalRotation`, with its
///        own loop builder and co-moving reference.
///
/// `doublyCancelledRatio` enforces the channel tags.
///
/// ## What is exact and what is certified
///
/// Parities and wedge (graded) signs are algebraically exact integers.
/// Transported characters are `CertifiedNumerical` with reported residuals and
/// conditioning. Gap closure, leaks, ill-conditioning, rank changes and
/// ambiguous matchings return uncertified reads, never a sign.
///
/// ## Read-only
///
/// Stateless: it never calls a solver, never mutates anything it reads, and
/// nothing here may enter any emergence objective. No Kasteleyn orientation is
/// required: the abstract exterior algebra is order-independent, and a
/// Kasteleyn gadget is only a possible surface-dimer implementation detail, not
/// the general spin certificate.
class ExchangeHolonomy {
  public:
    ExchangeHolonomy() = delete;  // static-only utility class

    // ---- certified overlap transport --------------------------------------

    /// The unitary polar factor of a (square) matrix M = U Sigma V^dagger
    /// -> U V^dagger.  Exposed because the polar step is the normative
    /// transport primitive of the holonomy reads.
    [[nodiscard]] static Eigen::MatrixXcd polarUnitary(
        const Eigen::MatrixXcd &overlap);

    /// Closed-loop holonomy of an explicit frame path: `frames[t]` is the
    /// (cells x rank) frame at loop position t (t = 0..T-1, cyclically
    /// closed back to t = 0), `weights` the constant diagonal metric W.
    /// All frames must share the row count and column count
    /// @throws std::invalid_argument otherwise — an explicit path with a
    /// shape mismatch is a structural error, unlike the fiber path where a
    /// rank change is a physical invalidation.
    [[nodiscard]] static LoopHolonomyRead loopHolonomy(
        const std::vector<Eigen::MatrixXcd> &frames,
        const Eigen::VectorXcd &weights,
        const ExchangeHolonomyConfig &cfg = {});

    /// As above with per-step metrics: `stepWeights[t]` is W_t used in the
    /// transport step t -> t+1 (T entries; the last is the closure step).
    [[nodiscard]] static LoopHolonomyRead loopHolonomyPerStep(
        const std::vector<Eigen::MatrixXcd> &frames,
        const std::vector<Eigen::VectorXcd> &stepWeights,
        const ExchangeHolonomyConfig &cfg = {});

    /// Closed-loop holonomy of a fiber track: consecutive fibers' frames are
    /// restricted to their shared cells, matched by sorted vertex-id tuple
    /// (gauge- and relabeling-invariant, the `SpectralFiber::overlap`
    /// convention), with W_t the departing fiber's weight diagonal on the
    /// shared cells. An uncertified band anywhere on the loop — a closed gap —
    /// or a rank change yields an uncertified read, never a sign.
    [[nodiscard]] static LoopHolonomyRead fiberLoopHolonomy(
        const std::vector<SpectralFiber> &loop,
        const ExchangeHolonomyConfig &cfg = {});

    // ---- interferometric (Berry-cancelled) characters --------------------

    /// chi_hat_F = det U_exchange / det U_reference for a particle-exchange
    /// loop against its matched non-exchanging reference loop (same
    /// geometric footprint and timing by construction of the caller; the
    /// step count and rank premises are verified here).  Channel:
    /// `ParticleExchange`.
    [[nodiscard]] static HolonomyCharacterRead exchangeCharacter(
        const LoopHolonomyRead &exchangeLoop,
        const LoopHolonomyRead &referenceLoop,
        const ExchangeHolonomyConfig &cfg = {});

    /// chi_hat(2 pi) = det U_rotation / det U_reference for the physical
    /// rotation loop against its matched CO-MOVING non-rotating reference.
    /// Channel: `PhysicalRotation`.  -1 on a clean spin-1/2 cycle, +1 on a
    /// vector cycle.
    [[nodiscard]] static HolonomyCharacterRead rotationCharacter(
        const LoopHolonomyRead &rotationLoop,
        const LoopHolonomyRead &referenceLoop,
        const ExchangeHolonomyConfig &cfg = {});

    /// The doubly cancelled spin-statistics ratio
    /// chi_hat(exchange) * chi_hat(2 pi rotation)^{-1} (+1 on the spin-1/2
    /// fixture, each factor separately near -1).
    /// @throws std::invalid_argument unless `exchange` is tagged
    /// `ParticleExchange` and `rotation` is tagged `PhysicalRotation` (the
    /// channels are never interchangeable).
    [[nodiscard]] static std::complex<double> doublyCancelledRatio(
        const HolonomyCharacterRead &exchange,
        const HolonomyCharacterRead &rotation);

    // ---- structural permutation channel ----------------------------------

    /// Persistent-block tracking around a closed loop.  `steps[t]` lists
    /// the localized block fibers at loop position t (every step the same
    /// block count; consecutive steps matched by
    /// `SpectralFiberTracker::matchFibers` with `blockMatchThreshold`, and
    /// every match must be a certified continuation — accepted equal-rank
    /// bands — for the read to certify).  `referenceSteps` is the matched
    /// non-exchanging reference tracking (same T and block count, identity
    /// full-loop permutation) used for the in-block-motion cancellation;
    /// pass empty to skip (residual reported unmeasured).  `composites`
    /// optionally groups block indices into clusters for the composite-
    /// level permutation view (e.g. one odd block + one even 2-mode
    /// composite).
    ///
    /// `occupancies` declares, per block and in block order, how many
    /// one-particle modes the state occupies there and how many sheets the
    /// block's support carries. That declaration is what the exchange
    /// statistic is computed from: the statistic is occupation parity, and the
    /// rank of a block's fibre enters only the separately reported
    /// cross-check. Passing an empty list declares the default cluster of the
    /// construction — one occupied mode on an unsheeted support — for every
    /// block, which is the quark, the antiquark and any other single occupied
    /// mode.
    ///
    /// Parities are exact integers; everything else carries residuals.
    /// @throws std::invalid_argument when `occupancies` is neither empty nor
    ///   one entry per tracked block, when a declared occupation exceeds the
    ///   block's rank — a state cannot occupy more modes than the fibre has —
    ///   or when a declared sheet count is zero.
    [[nodiscard]] static BlockPermutationRead blockPermutation(
        const std::vector<std::vector<SpectralFiber>> &steps,
        const std::vector<std::vector<SpectralFiber>> &referenceSteps = {},
        const std::vector<std::vector<std::size_t>> &composites = {},
        const std::vector<ClusterOccupancy> &occupancies = {},
        const ExchangeHolonomyConfig &cfg = {});

    /// The determinant of exchanging two complete fibre frames of ranks
    /// `rankA` and `rankB`: det pi_AB = (-1)^{r_A r_B}, the action of the
    /// interchange on the orientation line of E_A (+) E_B.
    ///
    /// The identity is exact. Its physical use is not: promoting it to
    /// particle statistics is the falsifiable hypothesis that cluster parity
    /// is fibre rank modulo two, which the construction does not adopt. It is
    /// offered as the independent cross-check that
    /// `BlockPermutationRead::rankParity` reports, and it is never multiplied
    /// into an occupation parity.
    [[nodiscard]] static int frameExchangeDeterminant(std::size_t rankA,
                                                      std::size_t rankB);

    // ---- the constructed total-space spin holonomy cycle ------------------

    /// The spinor representation dimension carried at spatial dimension d:
    /// 2 at d = 3 (Pauli), 4 at d = 4 (the documented Euclidean Dirac
    /// layer).  Other d are not implemented and throw
    /// @throws std::invalid_argument
    [[nodiscard]] static int spinorDimension(int d);

    /// Euclidean gamma matrix gamma_a (0-based axis a < d) with
    /// {gamma_a, gamma_b} = 2 delta_ab: the Pauli triple at d = 3; at
    /// d = 4 the documented layer gamma_0 = sigma_1 x sigma_1,
    /// gamma_1 = sigma_1 x sigma_2, gamma_2 = sigma_1 x sigma_3,
    /// gamma_3 = sigma_2 x I.
    [[nodiscard]] static Eigen::MatrixXcd gamma(int a, int d);

    /// Spin generator Sigma_ab = [gamma_a, gamma_b] / 4 = gamma_a gamma_b/2
    /// (a != b), eigenvalues +-i/2 — the documented half-angle generator.
    [[nodiscard]] static Eigen::MatrixXcd spinGenerator(int a, int b, int d);

    /// The plane rotation's spinor holonomy exp(theta Sigma_ab) =
    /// cos(theta/2) I + sin(theta/2) gamma_a gamma_b (closed form; theta =
    /// 2 pi gives exactly -I — the double cover).
    [[nodiscard]] static Eigen::MatrixXcd spinorRotation(double theta, int a,
                                                         int b, int d);

    /// The canonical transverse rank-1 spinor frame for the (a, b) plane:
    /// the equal-weight superposition of one +i/2 and one -i/2 eigenvector
    /// of Sigma_ab (deterministic eigenvector and phase convention).  Under
    /// the 2 pi cycle this line precesses a full great circle, making the
    /// spinor double cover interferometrically visible; a Sigma_ab
    /// EIGENvector is stationary and shows no relative phase (polarization
    /// along the rotation axis — documented, not guarded).
    [[nodiscard]] static Eigen::MatrixXcd transverseSpinorFrame(int a, int b,
                                                                int d);

    /// The constructed total-space spin holonomy cycle as an explicit closed
    /// frame path: Phi_t = exp(theta_t Sigma_ab) Phi_0 with theta_t =
    /// 2 pi turns t / steps, t = 0..steps-1, cyclically closed. One global
    /// rotation of the whole carried frame `frame0` (spinorDimension(d) rows),
    /// never a per-hole product.
    [[nodiscard]] static std::vector<Eigen::MatrixXcd> rotationLoopFrames(
        const Eigen::MatrixXcd &frame0, int a, int b, int d, int turns,
        int steps);

    /// The matched co-moving NON-rotating reference of the cycle: the same
    /// base frame held for the same number of steps (same timing, same
    /// metric, no rotation) — the Berry reference channel of the rotation
    /// experiment.
    [[nodiscard]] static std::vector<Eigen::MatrixXcd> referenceLoopFrames(
        const Eigen::MatrixXcd &frame0, int steps);

    /// The vector-representation counterpart of `rotationLoopFrames` for
    /// the +1 control: v_t = R_ab(theta_t) v_0 with the SO(d) plane
    /// rotation acting on a d-row frame.
    [[nodiscard]] static std::vector<Eigen::MatrixXcd> vectorLoopFrames(
        const Eigen::MatrixXcd &frame0, int a, int b, int d, int turns,
        int steps);

    // ---- the total-space spin read (the existing measuring stick) --------

    /// The total-spin Casimir J^2 = sum_a (sum_i S_a^(i))^2 on
    /// (C^2)^(tensor n) as a dense matrix with S_a = Pauli/2: the total-space
    /// operator, acting on the whole composite state at once.
    /// @throws std::invalid_argument for constituents < 1 or > 10 (the
    /// dense 2^n matrix cap; the read is a fixture-scale measuring stick).
    [[nodiscard]] static Eigen::MatrixXcd totalJSquaredOperator(
        int constituents);

    /// <J^2> of a composite state in (C^2)^(tensor n) (n inferred from the
    /// state size; the state is normalized internally).  Exact oracles:
    /// proton eigenstate 2|uud> - |udu> - |duu> -> 3/4, Delta |uuu> ->
    /// 15/4, product |uud> -> 7/4.
    /// @throws std::invalid_argument when the size is not a power of two
    /// or the state has zero norm.
    [[nodiscard]] static double totalJSquared(const Eigen::VectorXcd &state);

    // ---- SO(d) -> Spin(d) lift (continuum-claim machinery) ---------------

    /// The principal antisymmetric logarithm of a rotation matrix via the
    /// real Schur plane decomposition (every plane angle in (-pi, pi]; the
    /// angle-pi branch fixed by a deterministic axis-sign rule).  The
    /// returned A satisfies exp(A) = R block-exactly.
    /// @throws std::invalid_argument when R is not orthogonal with
    /// determinant +1 (tolerance 1e-9).
    [[nodiscard]] static Eigen::MatrixXd rotationLog(
        const Eigen::MatrixXd &rotation);

    /// The principal Spin(d) lift of an SO(d) rotation (d = 3, 4): the
    /// plane decomposition of `rotationLog` mapped through the closed-form
    /// factor cos(theta/2) I - sin(theta/2) gamma(u) gamma(v) per rotation
    /// plane (factors commute across orthogonal planes).  The sign selects
    /// the COVERING-HOMOMORPHISM orientation — the defining identity is
    /// S gamma(x) S^{-1} = gamma(R x), so lifts compose projectively:
    /// rotationToSpin(R1 R2) = +- rotationToSpin(R1) rotationToSpin(R2).
    /// Orientation note: with Sigma_ab = gamma_a gamma_b / 2 this makes
    /// rotationToSpin(R_ab(theta)) = spinorRotation(-theta, a, b, d), the
    /// two documented conventions related by the plane orientation.  A
    /// rotation by theta lifts with the half angle theta/2; the two lifts
    /// +-S differ by the center, and this function returns the principal one,
    /// with plane angles in (-pi, pi] and the pi branch fixed by the documented
    /// axis rule.
    [[nodiscard]] static Eigen::MatrixXcd rotationToSpin(
        const Eigen::MatrixXd &rotation, int d);

    /// The Z2 character of a closed discretized SO(d) loop `loop[t]`,
    /// t = 0..T-1, cyclic: incremental rotations R_{t+1} R_t^T are lifted
    /// principally and composed. The closed product is +-I and the sign is the
    /// pi_1(SO(d)) class: +1 contractible, -1 the double-cover generator, such
    /// as a 2 pi plane rotation. A step at or beyond pi - liftAngleMargin makes
    /// the branch ambiguous, leaving the read uncertified with no sign.
    [[nodiscard]] static LoopLiftRead loopLiftCharacter(
        const std::vector<Eigen::MatrixXd> &loop, int d,
        const ExchangeHolonomyConfig &cfg = {});

    /// The SO(d) -> Spin(d) lift decision over Cech transition data:
    /// `edges[e] = (i, j)` carries `edgeRotations[e]` = g_ij (g_ji is its
    /// transpose, lifted independently by the same principal rule);
    /// `triangles` lists vertex triples (i, j, k) traversed as given.  The
    /// SO cocycle g_ij g_jk g_ki = I is verified per triangle, which is the
    /// structural premise. Per-triangle lift signs w_t are computed, and the
    /// exact GF(2) coboundary decision via `cobordism::gf2Rank` either accepts
    /// the lift, returning a consistent per-edge sign choice, or rejects it as
    /// the w2 obstruction. The class is independent of the
    /// pi-branch convention (an edge's branch flip toggles exactly its
    /// adjacent triangles).  This concerns only continuum spinor claims —
    /// never the abstract CAR/Fock algebra, and no Kasteleyn orientation
    /// is involved.
    /// @throws std::invalid_argument on malformed input (edge/rotation
    /// count mismatch, a triangle edge missing from `edges`, non-3
    /// triangle, d not in {3, 4}).
    [[nodiscard]] static SpinLiftRead spinLift(
        const std::vector<std::pair<std::uint64_t, std::uint64_t>> &edges,
        const std::vector<Eigen::MatrixXd> &edgeRotations,
        const std::vector<std::vector<std::uint64_t>> &triangles, int d,
        const ExchangeHolonomyConfig &cfg = {});

    // ---- channel-separation gauge actions (for tests and reports) --------

    /// The simplex-reorientation gauge on a frame path: row r of every frame
    /// multiplied by cellSigns[r] (+-1). Reversing a k-cell's orientation flips
    /// its cochain component on every frame alike, and every read of this class
    /// is exactly invariant, since the diagonal sign conjugates away in
    /// Phi^dagger W Phi when W is diagonal.
    /// @throws std::invalid_argument on a sign not in {-1, +1} or a size
    /// mismatch.
    [[nodiscard]] static std::vector<Eigen::MatrixXcd> reorientedFrames(
        const std::vector<Eigen::MatrixXcd> &frames,
        const std::vector<int> &cellSigns);

    /// The compilation-ordering gauge on a frame path: every frame's rows, and
    /// the caller's weights separately, permuted by `rowPermutation` so that new
    /// row r is old row rowPermutation[r]. A vertex relabeling or cell
    /// reordering under which every read is exactly invariant.
    /// @throws std::invalid_argument unless `rowPermutation` is a
    /// bijection of the row count.
    [[nodiscard]] static std::vector<Eigen::MatrixXcd> permutedCellFrames(
        const std::vector<Eigen::MatrixXcd> &frames,
        const std::vector<std::size_t> &rowPermutation);

  private:
    struct RestrictedPair;  // consecutive fibers restricted to shared cells

    // Assemble a LoopHolonomyRead (residuals + certificate) from the
    // composed holonomy and the per-step reads — shared by the frame and
    // fiber paths.
    [[nodiscard]] static LoopHolonomyRead finalizeLoop(
        Eigen::MatrixXcd holonomy, std::vector<TransportStepRead> stepReads,
        std::size_t steps, std::size_t rank,
        cobordism::CertificateRegime regime, bool uncertifiedBand,
        const ExchangeHolonomyConfig &cfg);

    // One transport step: overlap, singular data, polar factor.
    [[nodiscard]] static Eigen::MatrixXcd transportStep(
        const Eigen::MatrixXcd &from, const Eigen::MatrixXcd &to,
        const Eigen::VectorXcd &weights, TransportStepRead &read,
        const ExchangeHolonomyConfig &cfg);

    // Restrict two consecutive fibers to their shared cells.
    [[nodiscard]] static RestrictedPair restrictToSharedCells(
        const SpectralFiber &from, const SpectralFiber &to);

    // The verified regime of a plain weight vector.
    [[nodiscard]] static cobordism::CertificateRegime weightsRegime(
        const Eigen::VectorXcd &weights);

    // Shared character core (exchangeCharacter / rotationCharacter).
    [[nodiscard]] static HolonomyCharacterRead characterAgainstReference(
        const LoopHolonomyRead &loop, const LoopHolonomyRead &reference,
        HolonomyChannel channel, const ExchangeHolonomyConfig &cfg);

    // Exact parity of a permutation via the exterior-algebra grading.
    [[nodiscard]] static int permutationSign(
        const std::vector<std::size_t> &permutation);

    // The SO(d) plane decomposition backing rotationLog / rotationToSpin:
    // orthonormal plane pairs (u_k, v_k) with angles theta_k in (-pi, pi].
    struct PlaneDecomposition;
    [[nodiscard]] static PlaneDecomposition planeDecomposition(
        const Eigen::MatrixXd &rotation);

    // Solve D s = w over GF(2) (row-major dense), returning a particular
    // solution when consistent.
    [[nodiscard]] static bool gf2Solve(std::vector<int> matrix, int rows,
                                       int cols, std::vector<int> rhs,
                                       std::vector<int> &solution);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_EXCHANGEHOLONOMY_H
