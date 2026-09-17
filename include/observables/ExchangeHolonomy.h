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

/// Analysis parameters of the exchange/rotation holonomy reads (ticket
/// #772).  Every threshold selects which
/// reads are CERTIFIED, never which value is reported: a failed threshold
/// yields an UNCERTIFIED read (the #769 gap-closure semantics), never a
/// different sign.
struct ExchangeHolonomyConfig {
  /// A certified transport step needs every singular value of its overlap
  /// matrix at least this large (a leaking transfer — the tracked subspace
  /// turning away from its successor — invalidates the read BEFORE polar
  /// normalization).
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
  /// :func:`ExchangeHolonomy::blockPermutation` (mirrors the #769
  /// `SpectralFiberConfig::trackOverlapThreshold`; the matching itself is
  /// delegated to `SpectralFiberTracker::matchFibers`).
  double blockMatchThreshold = 0.5;
  /// Incremental steps of a lifted SO(d) loop must keep their rotation
  /// angle at least this far below pi (the double-cover branch cut); a
  /// larger step makes the principal lift ambiguous and the loop character
  /// UNCERTIFIED instead of a guessed sign.
  double liftAngleMargin = 1e-6;
  /// Cap on the verified SO(d) cocycle residual
  /// max_t ||g_ij g_jk g_ki - I||_F of :func:`ExchangeHolonomy::spinLift`
  /// (the structural premise the GF(2) obstruction decision is exact under).
  double cocycleTolerance = 1e-9;
};

/// Which physical question a Berry-cancelled character answers.  The ticket
/// requires particle exchange and physical rotation to be SEPARATE channels
/// in the API and the report: the same interferometric machinery runs both,
/// but a read is always tagged with the channel it certifies, and
/// :func:`ExchangeHolonomy::doublyCancelledRatio` refuses mislabeled inputs.
enum class HolonomyChannel {
  /// A configuration-space loop that permutes identical clusters.
  ParticleExchange,
  /// The constructed total-space spin-holonomy cycle: one global rotation
  /// of the whole carried cluster frame (never a per-hole Bloch product).
  PhysicalRotation,
};

/// One overlap-transport step of a closed loop: the singular-value data of
/// the r x r frame overlap BEFORE polar normalization, and whether the step
/// met the leak/conditioning thresholds.
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
/// The certified overlap transport of one tracked frame around one CLOSED
/// configuration-space loop:
///
///   R_t = polar(Phi_{t+1 mod T}^dagger W_t Phi_t),
///   U_gamma = R_{T-1} ... R_1 R_0 .
///
/// The transport is cyclic — the last step closes the loop back onto the
/// t = 0 frame, so U_gamma maps the base frame's gauge to itself and
/// det U_gamma is invariant under EVERY in-band frame rotation
/// Phi_t -> Phi_t g_t (polar is unitarily equivariant, so the g's cancel
/// around the cycle up to conjugation).
///
/// `determinant` is the RAW loop determinant chi_raw = det U_gamma.  It
/// contains the ordinary path-dependent Berry phase of the reference motion
/// and IS NEVER an exchange sign by itself — only the interferometric ratio
/// of :func:`ExchangeHolonomy::exchangeCharacter` (a matched non-exchanging
/// reference loop) or :func:`ExchangeHolonomy::rotationCharacter` (a matched
/// co-moving non-rotating reference) is.
struct LoopHolonomyRead {
  /// The composed loop holonomy U_gamma (rank x rank).  Empty when the
  /// read was structurally invalidated (rank change along a fiber track).
  Eigen::MatrixXcd holonomy{};
  /// chi_raw = det U_gamma — Berry phase INCLUDED, never a sign by itself.
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
  /// True when a fiber on the loop carried an UNCERTIFIED band certificate
  /// (its isolating gap closed): the read is reported but never certified —
  /// the #769 semantics, a closing gap invalidates instead of flipping.
  bool uncertifiedBand = false;
  /// #764 certificate: `CertifiedNumerical` on the verified regime when
  /// every step met the thresholds and no band was uncertified;
  /// `HeuristicDiscovery` (never holds) otherwise.
  cobordism::Certificate certificate{};
};

/// # HolonomyCharacterRead
///
/// The interferometric (Berry-cancelled) character of one loop against its
/// matched reference loop:
///
///   chi_hat_F = det U_loop / det U_reference
///
/// (whitepaper "Fermion statistics from simplicial orientation").  The
/// report keeps the three phase channels SEPARATE:
/// `rawLoopDeterminant` (exchange or rotation motion + Berry),
/// `referenceDeterminant` (the Berry reference motion alone), and
/// `character` (the cancelled ratio — the dynamical certificate).
struct HolonomyCharacterRead {
  /// Which physical question this character answers (exchange vs rotation
  /// are separate channels by construction — the ticket's API separation).
  HolonomyChannel channel = HolonomyChannel::ParticleExchange;
  /// det U of the exchange/rotation loop — Berry phase INCLUDED.
  std::complex<double> rawLoopDeterminant{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// det U of the matched non-exchanging / co-moving non-rotating
  /// reference loop — the Berry reference channel on its own.
  std::complex<double> referenceDeterminant{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
  /// chi_hat = rawLoopDeterminant / referenceDeterminant.
  std::complex<double> character{std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()};
  /// -1 or +1 when the certificate holds and |character -+ 1| is within
  /// `signTolerance`; 0 otherwise (an uncertified read NEVER emits a sign).
  int characterSign = 0;
  /// |character - characterSign| when a sign was emitted (NaN otherwise).
  double signResidual = std::numeric_limits<double>::quiet_NaN();
  /// Whether the two loops had the same step count (the timing premise of
  /// the cancellation identity).  A mismatch is reported and uncertified.
  bool timingMatched = false;
  /// Whether the two loops had equal tracked rank.
  bool ranksMatched = false;
  /// #764 certificate: holds only when both loop certificates hold, timing
  /// and ranks match, and | |character| - 1 | is within tolerance.
  cobordism::Certificate certificate{};
};

/// # BlockPermutationRead
///
/// The structural channel of the exchange experiment: persistent-component
/// matching of localized odd blocks around the loop, the extracted
/// permutation, its EXACT parities (the algebraic wedge sign, delegated to
/// the #766 grading), and the residual in-block motion left after the
/// matched reference loop is cancelled — kept strictly separate from the
/// interferometric determinant channel.
struct BlockPermutationRead {
  /// blockPermutation[b] = index (in the t = 0 block list) the block at
  /// position b arrives at after one full loop.  Empty when uncertified.
  std::vector<std::size_t> blockPermutation{};
  /// The tracked blocks' ranks at t = 0.
  std::vector<std::size_t> blockRanks{};
  /// Sign of `blockPermutation` as a permutation of block LABELS
  /// (+1/-1; 0 when uncertified).  A combinatorial datum — NOT the
  /// exchange statistic (a rank-1 <-> rank-2 block swap has blockParity -1
  /// but graded sign +1).
  int blockParity = 0;
  /// The EXCHANGE STATISTIC: the sign of the induced MODE permutation
  /// (blocks expanded to their `blockRanks` modes, in-block order carried),
  /// computed by the exact #766 rule
  /// `quantum::OccupationBitset::permutationParity` — equal to the graded
  /// sign prod (-1)^{n_a n_b} over exchanged cluster pairs.  +1/-1; 0 when
  /// uncertified.
  int modeParity = 0;
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
  /// #764 certificate: `StructureExact` (parities are exact integers GIVEN
  /// the verified premise that every step's block matching is a certified
  /// bijection of equal-rank accepted bands); `HeuristicDiscovery` when the
  /// premise failed (gap closure, rank change, ambiguous matching).
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
/// manifold-like regimes only; the abstract CAR/Fock exchange algebra
/// requires no spin structure and no Kasteleyn orientation.
struct SpinLiftRead {
  /// Whether a consistent sign choice exists (w2 cohomologically trivial).
  bool liftExists = false;
  /// The negation of `liftExists` once certified (kept explicit so an
  /// UNCERTIFIED read can report neither).
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
/// holonomy cycle, and the conditional SO(d) -> Spin(d) lift (ticket #772,
/// Wave 2 of #763; whitepaper section "Fermion statistics from simplicial
/// orientation").
///
/// **Identities implemented.**
///
///  1. Certified cyclic overlap transport of an isolated tracked subspace:
///     `R_t = polar(Phi_{t+1 mod T}^dagger W_t Phi_t)`,
///     `U_gamma = R_{T-1} ... R_0` — composes #769 `SpectralFiber` frames
///     (cells matched by sorted vertex-id tuple; no second subspace tracker
///     is built here) or explicit frame paths.  det U_gamma is invariant
///     under in-band frame rotations, vertex relabeling, and simplex
///     reorientation (a common row sign flip), exactly.
///  2. The interferometric exchange character
///     `chi_hat_F = det U_exchange / det U_reference` against a matched
///     non-exchanging reference loop with the same timing (step count) and
///     rank.  The RAW determinant contains an ordinary Berry phase and is
///     never the exchange sign; only the cancelled ratio is the dynamical
///     certificate.  Domain: equal-step, equal-rank certified loops.
///  3. The structural permutation of persistent localized odd blocks
///     (matching delegated to `SpectralFiberTracker::matchFibers`), its
///     exact parity through the #766 grading
///     (`quantum::OccupationBitset::permutationParity` — the algebraic
///     wedge sign, an integer, exact), and the residual in-block motion
///     after reference cancellation.  Algebraic and dynamical channels are
///     reported separately and never conflated.
///  4. The total-space spin holonomy cycle, constructed HERE as the
///     canonical physical rotation path (no prior document supplies a
///     cycle, a closed loop, or a reference normalization): the
///     Euclidean gamma layer, spin generators `Sigma_ab = [gamma_a,
///     gamma_b]/4`, the closed-form plane rotation
///     `exp(theta Sigma_ab) = cos(theta/2) I + sin(theta/2) gamma_a
///     gamma_b`, and the closed 2 pi cluster-frame loop with its matched
///     co-moving non-rotating reference.  The rotation acts on the WHOLE
///     carried frame at once — never a product of per-hole or per-edge
///     Bloch vectors.  The transverse frame makes the double cover
///     interferometrically visible (a frame polarized along the rotation
///     axis does not precess and shows no relative phase).
///  5. The total-space spin read `J^2 = sum_a (sum_i S_a^(i))^2` on
///     `(C^2)^(tensor n)` — the exact measuring stick whose oracle values
///     are pinned: proton eigenstate `2|uud> - |udu> - |duu>` -> 3/4,
///     Delta `|uuu>` -> 15/4, product `|uud>` -> 7/4.  The operator and
///     these oracle values are what `joint_proton_spin_findings.md`
///     supplies; the rotation cycle of item 4 is not from that document.
///  6. The conditional Spin(d) lift: the principal rotation logarithm via
///     the real Schur plane decomposition, the closed-form plane-product
///     lift SO(d) -> Spin(d) (d = 3, 4), the Z2 character of a closed
///     SO(d) loop, and the second Stiefel-Whitney obstruction of Cech
///     transition data with the exact GF(2) coboundary decision
///     (`cobordism::gf2Rank`).  Continuum-claim machinery only.
///
/// **Channel separation (ticket requirement).**  Five channels are kept
/// distinct in the API and reports: (i) simplex REORIENTATION is a common
/// row sign flip `reorientedFrames` under which every read is exactly
/// invariant; (ii) COMPILATION ORDERING (mode order, vertex labels) is a
/// #766 compilation artifact — reads here match cells by vertex tuple /
/// permute rows and are exactly invariant, with any bookkeeping parity
/// supplied by `EdgeModeRegistry`/`OccupationBitset`, never by this class;
/// (iii) PARTICLE EXCHANGE is `HolonomyChannel::ParticleExchange` plus the
/// structural `modeParity`; (iv) BERRY REFERENCE MOTION is the
/// `referenceDeterminant` channel, reported raw and cancelled, never
/// interpreted alone; (v) PHYSICAL ROTATION is
/// `HolonomyChannel::PhysicalRotation` with its own loop builder and
/// co-moving reference.  `doublyCancelledRatio` enforces the channel tags.
///
/// **What is exact and what is certified.**  Parities and wedge/graded
/// signs are algebraically exact integers (#766).  Transported characters
/// are `CertifiedNumerical` with reported residuals and conditioning (#764
/// vocabulary).  Gap closure, leaks, ill-conditioning, rank changes, and
/// ambiguous matchings return UNCERTIFIED reads — never a sign.
///
/// **Read-only observable.**  Stateless; never calls a solver, never
/// mutates anything it reads, and nothing here may enter any emergence
/// objective.  No Kasteleyn orientation is required anywhere: the abstract
/// exterior algebra is order-independent (#766), and a Kasteleyn gadget is
/// only a possible surface-dimer IMPLEMENTATION detail, never the general
/// spin certificate.
class ExchangeHolonomy {
  public:
    ExchangeHolonomy() = delete;  // static-only utility class

    // ---- certified overlap transport (composing #769 frames) ------------

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

    /// Closed-loop holonomy of a #769 fiber track: consecutive fibers'
    /// frames are restricted to their SHARED cells (matched by sorted
    /// vertex-id tuple — gauge- and relabeling-invariant, the
    /// `SpectralFiber::overlap` convention), with W_t the departing fiber's
    /// weight diagonal on the shared cells.  An uncertified band anywhere
    /// on the loop (a closed gap) or a rank change yields an UNCERTIFIED
    /// read, never a sign (#769 semantics).
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
    /// composite).  Parities are exact #766 integers; everything else
    /// carries residuals.
    [[nodiscard]] static BlockPermutationRead blockPermutation(
        const std::vector<std::vector<SpectralFiber>> &steps,
        const std::vector<std::vector<SpectralFiber>> &referenceSteps = {},
        const std::vector<std::vector<std::size_t>> &composites = {},
        const ExchangeHolonomyConfig &cfg = {});

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
    /// 2 pi turns t / steps, t = 0..steps-1 (cyclically closed).  ONE
    /// global rotation of the whole carried frame `frame0`
    /// (spinorDimension(d) rows) — never a per-hole product.  The rotation
    /// path is never left abstract: this IS the executable path.
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
    /// (C^2)^(tensor n) as a dense matrix (S_a = Pauli/2) — the TOTAL-SPACE
    /// operator acting on the whole composite state at once.
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
    /// +-S differ by the center, and THIS function returns the principal
    /// one (plane angles in (-pi, pi], the pi branch by the documented
    /// axis rule).
    [[nodiscard]] static Eigen::MatrixXcd rotationToSpin(
        const Eigen::MatrixXd &rotation, int d);

    /// The Z2 character of a CLOSED discretized SO(d) loop `loop[t]`
    /// (t = 0..T-1, cyclic): incremental rotations R_{t+1} R_t^T are
    /// lifted principally and composed; the closed product is +-I and the
    /// sign is the pi_1(SO(d)) class (+1 contractible, -1 the double-cover
    /// generator, e.g. a 2 pi plane rotation).  A step at or beyond
    /// pi - liftAngleMargin makes the branch ambiguous: UNCERTIFIED, no
    /// sign.
    [[nodiscard]] static LoopLiftRead loopLiftCharacter(
        const std::vector<Eigen::MatrixXd> &loop, int d,
        const ExchangeHolonomyConfig &cfg = {});

    /// The SO(d) -> Spin(d) lift decision over Cech transition data:
    /// `edges[e] = (i, j)` carries `edgeRotations[e]` = g_ij (g_ji is its
    /// transpose, lifted independently by the same principal rule);
    /// `triangles` lists vertex triples (i, j, k) traversed as given.  The
    /// SO cocycle g_ij g_jk g_ki = I is VERIFIED per triangle (the
    /// structural premise); per-triangle lift signs w_t are computed, and
    /// the exact GF(2) coboundary decision (via `cobordism::gf2Rank`)
    /// accepts (returning a consistent per-edge sign choice) or rejects
    /// (the w2 obstruction) the lift.  The CLASS is independent of the
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

    /// The simplex-reorientation gauge on a frame path: row r of every
    /// frame multiplied by cellSigns[r] (+-1) — reversing a k-cell's
    /// orientation flips its cochain component on every frame alike, and
    /// every read of this class is EXACTLY invariant (the diagonal sign
    /// conjugates away in Phi^dagger W Phi since W is diagonal).
    /// @throws std::invalid_argument on a sign not in {-1, +1} or a size
    /// mismatch.
    [[nodiscard]] static std::vector<Eigen::MatrixXcd> reorientedFrames(
        const std::vector<Eigen::MatrixXcd> &frames,
        const std::vector<int> &cellSigns);

    /// The compilation-ordering gauge on a frame path: every frame's rows
    /// (and the caller's weights, separately) permuted by `rowPermutation`
    /// (new row r = old row rowPermutation[r]) — a vertex relabeling /
    /// cell reordering under which every read is EXACTLY invariant.
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

    // Exact parity of a permutation via the #766 grading.
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
