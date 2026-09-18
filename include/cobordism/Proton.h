// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PROTON_H
#define TESSERA_COBORDISM_PROTON_H

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

class MultiCobordism;  // returned (seeded, not run) by the node factories below

/// # Proton
///
/// Builder for the emergent proton, composing `MultiCobordism` without
/// modifying it. A proton is three quarks in a colorless bound state, so it is
/// built in two steps; a single `MultiCobordism` merge would be physically
/// invalid.
///
///   * Step A — recombination, one co-optimized 2→2 node: two neutral q-q̄
///     pairs `{1,-1,0}` ⊔ `{1,0,-1}` → a diquark `{1,ω}` ⊔ an antidiquark
///     `{1,ω²}`. A diquark is colored (an SU(3) `3̄`), so its target is a
///     2-vector rather than the singlet.
///   * Step B — formation, a separate 2→1 node: the diquark `{1,ω}` plus the
///     third quark `{ω²}` → the proton `{1,ω,ω²}`, the colorless 3-vector
///     color singlet. Target dimensions may differ between blocks; each
///     boundary block's `r_state` is fitted against its own.
///
/// Step B's output block is the proton at a point in time; its spatial slice,
/// with the relaxed metric copied in, is what the observable readers of
/// `block()` consume.
///
/// `build()` grows each step from a single Δ⁴ simplex seed — one pentatope,
/// with all further topology emerging through stage 1's F-lowering candidate
/// draw — runs A then B, and restarts across distinct seeds until step B's
/// whole cobordism carries the singlet on at least `minEmergentHoles`
/// (default 3) emergent holes. The two-step build converges less often than a
/// single merge, hence the restarts. Carrying the singlet is the physical
/// criterion; the hole count is a topological precondition of the period
/// readout, not a quark count. The accessors run `build()` lazily on first
/// use.
class Proton {
 public:
  /// ω, the primitive cube root of unity `(−1 + i√3)/2` — the unit
  /// color-charge phase.
  [[nodiscard]] static std::complex<double> omega();
  /// The proton color singlet `{1, ω, ω²}`, the colorless 3-vector step B
  /// drives the proton block to carry.
  [[nodiscard]] static std::vector<std::complex<double>> singlet();

  /// Configure a proton build. The physics — the targets, the two-step
  /// structure, the single Δ⁴ simplex seed — is fixed; only the optimization
  /// knobs are exposed.
  ///   * `seed`           — base RNG seed; restart `i` uses A-seed `seed+2i`
  ///                        and B-seed `seed+2i+1`.
  ///   * `registerDegree` — the color register degree `k` (3 on a 4-manifold,
  ///                        where the register is `ker L_{d-1}`, a spectral
  ///                        subspace). The periods are read over the emergent
  ///                        holes, which are not the register.
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
  explicit Proton(std::uint64_t seed = 0, int registerDegree = 3,
                  double gamma = 50.0, double inputWeight = 20.0,
                  int precone = 0, bool shouldUseDirectedSurgery = false,
                  bool preconeTimelike = false, bool preconeAlternate = false,
                  bool balancedEdges = false, bool singularValueRatio = false,
                  bool einsteinHilbert = true);

  /// Build the proton, restarting across seeds until step B's whole cobordism
  /// carries the singlet on at least `minEmergentHoles` emergent holes. When
  /// `maxRestarts` is exhausted the best attempt is kept and `converged()` is
  /// false. Each step runs an initialization pass (`initSteps`,
  /// `grow_boundaries=true`, which establishes the carrying input regions),
  /// then an evolution pass (`evolveSteps`, `grow_boundaries=false`, ∂W
  /// frozen), then `runStage2`. Idempotent.
  void build(int maxRestarts = 16, int initSteps = 180,
             int evolveSteps = 60, int stage1CandidateMoves = 8,
             double stage2Beta = 1.0, int stage2MaxIters = 10,
             double colorTolerance = 0.5, int minEmergentHoles = 3);

  /// One-step build: drive `directNode` — three q-q̄ pairs in, the singlet out,
  /// in a single `MultiCobordism` — with `MultiCobordism::run`, which
  /// interleaves the stage-1 surgery update and the stage-2 geometric
  /// relaxation in one loop: an initialization pass (`initSteps`,
  /// `grow_boundaries=true`) then an evolution pass (`evolveSteps`,
  /// `grow_boundaries=false`, ∂W frozen). Restart `i` uses seed `seed + i`;
  /// restarts run until the whole cobordism carries the singlet on at least
  /// `minEmergentHoles` emergent holes, or until `maxRestarts` is exhausted,
  /// keeping the best attempt. Populates the same accessors as `build()`, with
  /// `diquarkResidual()` left at 0 because there is no step A. Idempotent, and
  /// shares build state with `build()`: whichever runs first claims it, so call
  /// this before any accessor triggers the lazy two-step `build()`.
  void buildDirect(int maxRestarts = 16, int initSteps = 180, int evolveSteps = 60,
                   int stage1CandidateMoves = 8, double stage2Beta = 1.0,
                   double colorTolerance = 0.5, int minEmergentHoles = 3);

  /// A fresh, seeded but not-yet-run step-A node (recombination, 2→2): two
  /// neutral q-q̄ pairs `{1,-1,0}` ⊔ `{1,0,-1}` → a diquark `{1,ω}` ⊔
  /// antidiquark `{1,ω²}`, on a single Δ⁴ seed (inputs at v0,v1; outputs at
  /// v2,v3; input weight set). `build()` and the animation both drive this
  /// setup via `runStage1`/`runStage2`.
  [[nodiscard]] std::shared_ptr<MultiCobordism> recombinationNode(std::uint64_t seed) const;
  /// A fresh, seeded but not-yet-run step-B node (formation, 2→1): the diquark
  /// `{1,ω}` plus the third quark `{ω²}` → the proton singlet `{1,ω,ω²}`, on a
  /// single Δ⁴ seed. Inputs at v0,v1; the single output is read off the whole
  /// cobordism, so there is no `seedOutputs`.
  [[nodiscard]] std::shared_ptr<MultiCobordism> formationNode(std::uint64_t seed) const;
  /// A fresh, seeded but not-yet-run one-step node (6→1): the three bare quarks
  /// `{1}`, `{ω}`, `{ω²}` and their three anti-quarks `{1}`, `{ω̄}`, `{ω̄²}` —
  /// the elementwise conjugates, this construction's antiparticle convention,
  /// under which antidiquark = conj(diquark) — as inputs on a single Δ⁴ seed,
  /// so the prepared content is three q-q̄ pairs. The proton singlet `{1,ω,ω²}`
  /// is the single output, read off the whole cobordism (no `seedOutputs`); the
  /// anti-baryon partner is left to emerge unpinned. An experimental
  /// single-merge alternative to the two-step build.
  [[nodiscard]] std::shared_ptr<MultiCobordism> directNode(std::uint64_t seed) const;

  /// True iff the whole step-B cobordism carries the singlet (`colorResidual()
  /// < colorTolerance`) on at least `minEmergentHoles` emergent holes. Triggers
  /// `build()`.
  [[nodiscard]] bool converged();
  /// The base seed of the converged (or best) attempt. Triggers `build()`.
  [[nodiscard]] std::uint64_t seed();
  /// The full relaxed emergent complex of step B (proton formation), grown from
  /// the single Δ⁴ seed. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> spacetime();
  /// The proton itself: the relaxed step-B cobordism as a whole. The single
  /// output is the whole's harmonic — the inputs are held by their residual and
  /// the bulk evolves to carry the singlet — so there is no sub-block. This is
  /// what the observable readers consume. Triggers `build()`.
  [[nodiscard]] std::shared_ptr<Spacetime> block();
  /// The emergent holes (`(k+2)`-vertex tuples) on the proton, over which the
  /// singlet periods are read; at least `minEmergentHoles` of them when
  /// converged. A topological observable of this construction, not a quark
  /// count. Triggers `build()`.
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
  };

  /// Drive one node through `schedule`. Shared by the canonical arm and the
  /// ingredients arm, which run the identical schedule.
  static void driveNode(MultiCobordism &node, const NodeDrive &schedule);

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

#endif  // TESSERA_COBORDISM_PROTON_H
