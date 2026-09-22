// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_SYNTHESIS_H
#define TESSERA_COBORDISM_PROTON_SYNTHESIS_H

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

class MultiCobordism;  // returned (seeded, not run) by the node factories below

/// # ProtonSynthesis
///
/// Controlled synthesis of a proton: a labelled
/// `MultiCobordism::SimulationMode::Synthesis` experiment that composes
/// `MultiCobordism` without modifying it. The synthesis pins colour targets —
/// the output is driven toward the singlet `{1, ω, ω²}` — and accepts an
/// attempt only if it carries that singlet on at least `minEmergentHoles`
/// holes. A target of this kind is permitted only in explicitly labelled
/// controlled synthesis, never in emergence mode, so every node this class
/// builds is stamped `SimulationMode::Synthesis` (the mode recorded on a
/// checkpoint as `"synthesis"`), and `driveNode`, `build()` and
/// `buildDirect()` refuse a node in any other mode. The result is an existence
/// and obstruction experiment, not a proton found by the emergence protocol,
/// which pins no target.
///
/// A proton is three quarks in a colourless bound state, so the synthesis runs
/// in two steps; a single `MultiCobordism` merge would be physically invalid.
///
///   * Step A — recombination, one co-optimized 2→2 node: two neutral q-q̄
///     pairs `{1,-1,0}` ⊔ `{1,0,-1}` → a diquark `{1,ω}` ⊔ an antidiquark
///     `{1,ω²}`. A diquark is coloured (an SU(3) `3̄`), so its target is a
///     2-vector rather than the singlet.
///   * Step B — formation, a separate 2→1 node: the diquark `{1,ω}` plus the
///     third quark `{ω²}` → the proton `{1,ω,ω²}`, the colourless 3-vector
///     colour singlet. Target dimensions may differ between blocks; each
///     boundary block's `r_state` is fitted against its own.
///
/// Step B's output block is the synthesized proton at a point in time; its
/// spatial slice, with the relaxed metric copied in, is what the observable
/// readers of `block()` consume.
///
/// `build()` grows each step from a single Δ⁴ simplex seed — one pentatope,
/// with all further topology grown through stage 1's F-lowering candidate
/// draw — runs A then B, and restarts across distinct seeds until step B's
/// whole cobordism carries the singlet on at least `minEmergentHoles`
/// (default 3) holes. The two-step build converges less often than a single
/// merge, hence the restarts. Carrying the singlet is the acceptance
/// criterion; the hole count is a topological precondition of the period
/// readout, not a quark count. The accessors run `build()` lazily on first
/// use.
class ProtonSynthesis {
 public:
  /// ω, the primitive cube root of unity `(−1 + i√3)/2` — the unit
  /// colour-charge phase.
  [[nodiscard]] static std::complex<double> omega();
  /// The proton colour singlet `{1, ω, ω²}`, the colourless 3-vector step B
  /// drives the proton block to carry.
  [[nodiscard]] static std::vector<std::complex<double>> singlet();

  /// Configure a proton synthesis. The physics — the targets, the two-step
  /// structure, the single Δ⁴ simplex seed — is fixed; only the optimization
  /// knobs are exposed.
  ///   * `seed`           — base RNG seed; restart `i` uses A-seed `seed+2i`
  ///                        and B-seed `seed+2i+1`.
  ///   * `registerDegree` — the colour register degree `k` (3 on a
  ///                        4-manifold, where the register is `ker L_{d-1}`, a
  ///                        spectral subspace). The periods are read over the
  ///                        holes of the grown complex, which are not the
  ///                        register.
  ///   * `gamma`          — Γ in `F = ‖∇S_Regge‖² + Γ·r_U`, chosen so Γ·r_U
  ///                        sits on the same order as ‖∇S‖²; otherwise ∇S
  ///                        dominates and the register is never driven to
  ///                        carry.
  ///   * `inputWeight`    — weight on the input residuals, so the diquark and
  ///                        quark inputs are driven to carry rather than
  ///                        dissolve.
  ///   * `precone`        — pre-grow each step's seed by this many gated
  ///                        cone-in moves before optimization (forwarded to
  ///                        every node's `MultiCobordism` constructor), giving
  ///                        surgery room to act without prebuilding a host.
  ///                        Default 0.
  ///   * `shouldUseDirectedSurgery` — augment each step's drive with the
  ///                        directed cone-out and cone-in probes
  ///                        (`MultiCobordism::directedConeOut` and
  ///                        `directedConeIn`), a gated score-guided search for
  ///                        the cells to remove and the boundary facets to
  ///                        cap, in place of `runStage1`'s random cone draws.
  ///                        Default false.
  ///   * `preconeTimelike` — draw every precone cone-in as the timelike
  ///                        disposition.
  ///   * `preconeAlternate` — alternate the precone cone-ins timelike and
  ///                        spacelike for balanced causal content at one
  ///                        uniform edge-length magnitude. Takes precedence
  ///                        over `preconeTimelike`.
  ///   * `balancedEdges`  — wire the seed and every node with the balanced
  ///                        edge form (`MultiCobordism::seedSimplex`, where
  ///                        balanced wiring gives `ℓ = √(1/2)(1+i)`).
  ///   * `singularValueRatio` — score the whole-complex term of `r_U` with the
  ///                        scale-invariant singular-value half-sum ratio
  ///                        (`MultiCobordism::singularValueHalfSumRatio`)
  ///                        instead of the singlet period residual plus
  ///                        near-kernel pair. The singlet stays the
  ///                        after-the-fact readout.
  ///   * `einsteinHilbert` — keep the discrete Einstein-Hilbert term
  ///                        `‖∇S_Regge‖²` in every node's objective. False
  ///                        optimizes `gamma * r_U` alone; see the
  ///                        `MultiCobordism` constructor for what that means
  ///                        for stage 2's descent direction.
  explicit ProtonSynthesis(std::uint64_t seed = 0, int registerDegree = 3,
                           double gamma = 50.0, double inputWeight = 20.0,
                           int precone = 0, bool shouldUseDirectedSurgery = false,
                           bool preconeTimelike = false,
                           bool preconeAlternate = false,
                           bool balancedEdges = false,
                           bool singularValueRatio = false,
                           bool einsteinHilbert = true);

  /// Run the synthesis, restarting across seeds until step B's whole
  /// cobordism carries the singlet on at least `minEmergentHoles` holes. When
  /// `maxRestarts` is exhausted the best attempt is kept and `converged()` is
  /// false. Each step runs an initialization pass (`initSteps`,
  /// `grow_boundaries=true`, which establishes the carrying input regions),
  /// then an evolution pass (`evolveSteps`, `grow_boundaries=false`, ∂W
  /// frozen), then `runStage2`, all in `SimulationMode::Synthesis`.
  /// Idempotent.
  void build(int maxRestarts = 16, int initSteps = 180,
             int evolveSteps = 60, int stage1CandidateMoves = 8,
             double stage2Beta = 1.0, int stage2MaxIters = 10,
             double colorTolerance = 0.5, int minEmergentHoles = 3);

  /// One-step synthesis: drive `directNode` — three q-q̄ pairs in, the singlet
  /// out, in a single `MultiCobordism` — with `MultiCobordism::run`, which
  /// interleaves the stage-1 surgery update and the stage-2 geometric
  /// relaxation in one loop: an initialization pass (`initSteps`,
  /// `grow_boundaries=true`) then an evolution pass (`evolveSteps`,
  /// `grow_boundaries=false`, ∂W frozen). Restart `i` uses seed `seed + i`;
  /// restarts run until the whole cobordism carries the singlet on at least
  /// `minEmergentHoles` holes, or until `maxRestarts` is exhausted, keeping
  /// the best attempt. Populates the same accessors as `build()`, with
  /// `diquarkResidual()` left at 0 because there is no step A. Runs in
  /// `SimulationMode::Synthesis`. Idempotent, and shares build state with
  /// `build()`: whichever runs first claims it, so call this before any
  /// accessor triggers the lazy two-step `build()`.
  void buildDirect(int maxRestarts = 16, int initSteps = 180, int evolveSteps = 60,
                   int stage1CandidateMoves = 8, double stage2Beta = 1.0,
                   double colorTolerance = 0.5, int minEmergentHoles = 3);

  /// A fresh, seeded but not-yet-run step-A node (recombination, 2→2): two
  /// neutral q-q̄ pairs `{1,-1,0}` ⊔ `{1,0,-1}` → a diquark `{1,ω}` ⊔
  /// antidiquark `{1,ω²}`, on a single Δ⁴ seed (inputs at v0,v1; outputs at
  /// v2,v3; input weight set), in `SimulationMode::Synthesis`. `build()`
  /// drives this setup through `driveNode`.
  [[nodiscard]] std::shared_ptr<MultiCobordism> recombinationNode(std::uint64_t seed) const;
  /// A fresh, seeded but not-yet-run step-B node (formation, 2→1): the diquark
  /// `{1,ω}` plus the third quark `{ω²}` → the proton singlet `{1,ω,ω²}`, on a
  /// single Δ⁴ seed, in `SimulationMode::Synthesis`. Inputs at v0,v1; the
  /// single output is read off the whole cobordism, so there is no
  /// `seedOutputs`.
  [[nodiscard]] std::shared_ptr<MultiCobordism> formationNode(std::uint64_t seed) const;
  /// A fresh, seeded but not-yet-run one-step node (6→1) in
  /// `SimulationMode::Synthesis`: the three bare quarks `{1}`, `{ω}`, `{ω²}`
  /// and their three anti-quarks `{1}`, `{ω̄}`, `{ω̄²}` — the elementwise
  /// conjugates, this construction's antiparticle convention, under which
  /// antidiquark = conj(diquark) — as inputs on a single Δ⁴ seed, so the
  /// prepared content is three q-q̄ pairs. The proton singlet `{1,ω,ω²}` is
  /// the single output, read off the whole cobordism (no `seedOutputs`); the
  /// anti-baryon partner is not pinned. An experimental single-merge
  /// alternative to the two-step synthesis.
  [[nodiscard]] std::shared_ptr<MultiCobordism> directNode(std::uint64_t seed) const;

  /// True iff the whole step-B cobordism carries the singlet (`colorResidual()
  /// < colorTolerance`) on at least `minEmergentHoles` holes. Triggers
  /// `build()`.
  [[nodiscard]] bool converged();
  /// The base seed of the converged (or best) attempt. Triggers `build()`.
  [[nodiscard]] std::uint64_t seed();
  /// The full relaxed complex of step B (proton formation), grown from the
  /// single Δ⁴ seed. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime();
  /// The synthesized proton: the relaxed step-B cobordism as a whole. The
  /// single output is the whole's harmonic — the inputs are held by their
  /// residual and the bulk is driven to carry the singlet — so there is no
  /// sub-block. This is what the observable readers consume. Triggers
  /// `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> block();
  /// The holes (`(k+2)`-vertex tuples, `MultiCobordism::emergentHoles`) of the
  /// synthesized proton, over which the singlet periods are read; at least
  /// `minEmergentHoles` of them when converged. A topological observable of
  /// this construction, not a quark count. Triggers `build()`.
  [[nodiscard]] std::vector<std::vector<std::uint64_t>> emergentHoles();
  /// The proton singlet residual: the relabeling-invariant, zero-filled
  /// `r_state` of `singlet()` against the whole cobordism's `L_k` harmonic
  /// (`≈0` means carried). Triggers `build()`.
  [[nodiscard]] double colorResidual();
  /// Step A's realizability residual `r_U`. Small means the diquark
  /// recombination converged, a separate claim from the proton's formation.
  /// Triggers `build()`.
  [[nodiscard]] double diquarkResidual();

  /// The per-node drive schedule: one initialization pass with the boundary
  /// free to grow, an optional directed cone-out, one evolution pass with the
  /// boundary frozen, an optional directed cone-in, then the geometric
  /// relaxation.
  struct NodeDrive {
    int initSteps{180};
    int evolveSteps{60};
    int stage1CandidateMoves{8};
    double stage2Beta{1.0};
    int stage2MaxIters{10};
    /// Remove cells and cap facets between the two stage-1 passes.
    bool directedSurgery{false};

    /// Run this schedule on `node` without consulting its simulation mode.
    /// The synthesis drives its nodes through `driveNode`, which checks the
    /// mode first; `ProtonIngredients` runs its step B, which pins no output
    /// target, through this directly, so both arms run the identical
    /// schedule.
    void run(MultiCobordism &node) const;
  };

  /// Drive one synthesis node through `schedule`.
  /// @throws std::invalid_argument when `node` is not in
  ///         `SimulationMode::Synthesis` (in particular, in either emergence
  ///         sub-mode); nothing runs in that case.
  static void driveNode(MultiCobordism &node, const NodeDrive &schedule);

  /// Refuse a node that is not in `SimulationMode::Synthesis`: the synthesis
  /// pins targets, and targets are permitted only in the labelled
  /// controlled-synthesis mode.
  /// @throws std::invalid_argument naming the node's mode otherwise.
  static void requireSynthesisMode(const MultiCobordism &node);

 private:
  /// Lazily run `build()` with default parameters on first accessor use.
  void ensureBuilt();
  /// The minimal seed: a single `Δ⁴` simplex (one pentatope — 5 vertices, 1 top
  /// cell, Betti `[1,0,0,0,0]`, a contractible 4-ball) with a uniform metric.
  /// All topology grows from this one simplex through stage 1's F-lowering
  /// candidate draw and all geometry from the relaxation; nothing is
  /// pre-built.
  [[nodiscard]] static std::shared_ptr<Spacetime> buildMinimalSeed(
      bool balancedEdges = false);


  // ---- configuration ----
  std::uint64_t baseSeed_;
  int registerDegree_;
  double gamma_;
  bool balancedEdges_{false};
  bool singularValueRatio_{false};  // forwarded to every node (see the ctor)
  bool einsteinHilbert_{true};      // false optimizes gamma*rU alone
  double inputResidualWeight_;
  int precone_;  // gated cone-ins pre-grown into each node's seed (ctor → nodes)
  bool shouldUseDirectedSurgery_;  // build() uses the directed cone-out/cone-in probes
  bool preconeTimelike_;  // precone cone-ins drawn as the timelike disposition
  bool preconeAlternate_;  // precone cone-ins alternate timelike/spacelike

  // ---- build state (populated by build()) ----
  bool attempted_ = false;
  bool converged_ = false;
  std::uint64_t convergedSeed_ = 0;
  std::shared_ptr<Spacetime> spacetime_;  // step B's full relaxed complex
  std::shared_ptr<Spacetime> block_;      // proton sub-complex, relaxed metric
  std::vector<std::vector<std::uint64_t>> emergentHoles_;
  double colorResidual_ = 0.0;
  double diquarkResidual_ = 0.0;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PROTON_SYNTHESIS_H
