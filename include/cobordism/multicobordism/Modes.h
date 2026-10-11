// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// The simulation modes, the particle-independent refinement indicators and the
// analysis overlay's configuration.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_MODES_H
#define TESSERA_COBORDISM_MULTICOBORDISM_MODES_H

#include <string>
#include <vector>

namespace tessera::cobordism::multicobordism {

/// The three top-level simulation modes.
enum class SimulationMode {
  /// The production scientific mode: only the base geometric objective (plus
  /// the one permitted state-energy term) drives optimization; every particle
  /// and gauge quantity is a post-hoc observable.
  Emergence,
  /// A pinned carrier / spectral sector, to establish existence, measure a
  /// residual floor, or build an oracle fixture. Never counted as emergence.
  Synthesis,
  /// Recompute every derived hierarchy and certificate from a checkpoint and
  /// verify that no cached or serialized choice changed the result.
  Replay
};

/// The two labeled, Gaussian-closed emergence sub-modes. Recorded in
/// provenance on every checkpoint; both carry a covariance purity
/// certificate.
enum class EmergenceSubmode {
  /// The carried state does not act back on the geometry.
  Strict,
  /// The carried state's energy density may enter the joint stationarity
  /// objective through \f$ h = h(\Gamma, g) \f$, while no component, fiber,
  /// transport, amplitude, color, particle, charge, flavor, exchange or spin
  /// certificate may influence a geometry move.
  CertificatesBlindMeanField
};

/// The geometric and numerical indicators emergence-mode refinement may
/// consult. Every member is a quantity of the base problem: none is a
/// coarse-response residual, band gap, modularity, transport leakage, Wilson
/// or center read, exchange read, anchor score, amplitude Gram defect, or
/// particle score.
struct RefinementIndicators {
  /// \f$ \lVert\nabla_z S_{\rm Regge}\rVert^2 \f$ on the current complex.
  double reggeStationarityResidual = 0.0;
  /// \f$ \sum_k\lVert\nabla_z S_{\rm Hodge,k}\rVert^2 \f$.
  double hodgeStationarityResidual = 0.0;
  /// Curvature concentration: \f$ \max_h|\varepsilon_h| \f$ over the mean
  /// \f$ |\varepsilon_h| \f$ across the \f$(d-2)\f$-hinges (1 = flat spread,
  /// large = curvature piling onto one hinge). 0 when no hinge carries any.
  double curvatureConcentration = 0.0;
  /// Mesh quality: \f$ \min_\sigma|{\rm vol}\,\sigma| /
  /// \max_\sigma|{\rm vol}\,\sigma| \f$ over the top cells, in `[0,1]`;
  /// 0 means a degenerate cell. 1 with no cells.
  double meshQuality = 1.0;
  /// Solver discretization error: the magnitude of the last accepted stage-2
  /// objective improvement; 0 at a stationary point.
  double solverError = 0.0;
};

/// Whether to refine, and which indicator asked.
struct RefinementDecision {
  /// True when some indicator crossed its threshold.
  bool refine = false;
  /// The indicator that asked: one of `refinementIndicatorNames()`, or empty.
  std::string trigger{};
  /// The measured indicators the decision was taken on.
  RefinementIndicators indicators{};
};

/// Analysis-overlay configuration. Disabled by default: with `enabled` false
/// no part of the overlay runs.
struct AnalysisConfig {
  /// Master switch. False means the overlay never runs.
  bool enabled = false;
  /// Run one pass after every \p cadence accepted combinatorial moves.
  int cadence = 1;
  /// Hodge degrees the spectral bands are enumerated at.
  std::vector<int> degrees{1};
  /// The modularity resolution sequence scanned per pass.
  std::vector<double> resolutions{1.0};
  /// How many cobordism frames the overlay retains. One analysis pass is one
  /// frame, so the retained frames are the last \p frameHistory passes, each
  /// keeping its components, its candidate bands and their anchors. The
  /// candidate's lifetime, its smallest adjacent-frame overlap, its
  /// per-frame band and anchor families and its lifetime transports are
  /// measured across them; with `frameHistory` 1 there is no history, the
  /// pass sees a single frame, and the lifetime certificates fail by name
  /// rather than passing vacuously. The default 4 clears the classifier's
  /// two-frame stability floor with room to lose a frame.
  /// @throws std::invalid_argument (from `setAnalysisConfig`) below 1.
  int frameHistory = 4;
  /// How the determinant line of a lifetime transport family is closed.
  ///
  ///  - `"none"` (the default): the family is an open cobordism segment with
  ///    no declared closure. Its phase is reported and the winding is left
  ///    unknown — an open path has no integer winding, and inventing one
  ///    would be a measurement nobody made.
  ///  - `"closed-family"`: the caller declares the candidate's world tube
  ///    closed, so the family is read cyclically (the closing step returns to
  ///    the first sample) and the closed-family determinant winding is the
  ///    winding reported. It is a declaration about the run, like a causal
  ///    type, and it is recorded on every read it produces. The overlay
  ///    consumes that winding; nothing here computes it.
  ///
  /// @throws std::invalid_argument (from `setAnalysisConfig`) on any other
  ///   value: an unrecognized closure is refused, never silently ignored.
  std::string lifetimeWindingClosure{"none"};
  /// Build the lazy Fock expression. For oracle and explicit non-Gaussian
  /// boundary data only, never the quasi-free production representation.
  bool fockOracle = false;
  /// Serve nothing from the `AnalyticCache`, so the cold path can be compared
  /// against the incremental one.
  bool coldCaches = false;
};

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_MODES_H
