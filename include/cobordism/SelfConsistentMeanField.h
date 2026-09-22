// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_SELFCONSISTENTMEANFIELD_H
#define TESSERA_COBORDISM_SELFCONSISTENTMEANFIELD_H

#include <complex>
#include <cstddef>
#include <vector>

#include "cobordism/HolomorphicRelaxation.h"
#include "cobordism/JointAction.h"

namespace tessera::cobordism {

/// # OccupationOrder
///
/// Which modes of the carrier operator the covariance projects onto.
///
/// The whitepaper calls the filled modes "the occupied modes" and fixes no
/// order for a genuinely complex spectrum, where "lowest" is not defined by the
/// eigenvalues alone. The rule is therefore declared by the caller and recorded
/// with the run rather than assumed.
///
/// * `AscendingRealPart` — the modes of smallest real part, which is the
///   lowest-energy reading and reduces to the usual one on a Hermitian
///   specialization with real spectrum.
/// * `AscendingModulus` — the modes of smallest modulus, which is the reading
///   that orders by distance from the origin of the complex plane and is the
///   one a resolvent contour around zero selects.
enum class OccupationOrder { AscendingRealPart, AscendingModulus };

/// # SelfConsistentMeanFieldDeclaration
///
/// The configuration of a self-consistent backreaction solve.
struct SelfConsistentMeanFieldDeclaration {
  /// How many modes of the carrier operator are filled. The covariance at every
  /// step is the spectral projector onto exactly these.
  std::size_t occupiedModes = 1;

  /// Which modes those are.
  OccupationOrder occupationOrder = OccupationOrder::AscendingRealPart;

  /// The largest number of outer iterations — geometry relaxation followed by
  /// re-occupation — taken before the solve reports what it reached.
  std::size_t maximumIterations = 24;

  /// The Euclidean norm of the stationarity force, and the Frobenius norm of
  /// the change in the covariance, at or below which the pair
  /// \f$ (z^{*},\Gamma^{*}) \f$ is declared self-consistent. Both conditions
  /// must hold: a geometry that is stationary for a covariance that is still
  /// moving is not a fixed point, and neither is a settled covariance on a
  /// geometry that still carries a force.
  double tolerance = 1e-9;

  /// The fraction of the newly computed projector mixed into the covariance at
  /// each step, \f$ \Gamma \leftarrow (1-m)\Gamma + m\,P(h(z)) \f$.
  ///
  /// One — the default — is the plain re-occupation the whitepaper describes,
  /// and it is the only value for which the covariance is a projector at every
  /// step. A smaller value damps the outer iteration at the cost of leaving the
  /// intermediate covariance off the idempotent manifold, which
  /// `SelfConsistentMeanFieldStep::purityDefect` then measures; the fixed point
  /// is the same either way, because at a fixed point the old and the new
  /// projector coincide.
  double mixing = 1.0;

  /// The inner holomorphic relaxation that makes the geometry stationary
  /// against the current covariance. Its own tolerance and iteration count are
  /// independent of the outer ones.
  HolomorphicRelaxationDeclaration geometry;
};

/// # SelfConsistentMeanFieldStep
///
/// One outer iteration, recorded so a run can be read back.
struct SelfConsistentMeanFieldStep {
  /// The iteration index, counting from zero.
  std::size_t iteration = 0;
  /// The Euclidean norm of the joint stationarity force after the geometry was
  /// relaxed against the covariance this step started from.
  double forceNorm = 0.0;
  /// \f$ \lVert\Gamma_{n+1}-\Gamma_n\rVert_F \f$, the movement of the
  /// covariance under re-occupation.
  double covarianceChange = 0.0;
  /// \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$ of the covariance this step produced.
  double purityDefect = 0.0;
  /// The complex action at the end of this step.
  std::complex<double> action{0.0, 0.0};
  /// \f$ \operatorname{tr}(\Gamma h) \f$, the occupied band's energy, which for
  /// a spectral projector is the sum of the occupied eigenvalues.
  std::complex<double> occupiedEnergy{0.0, 0.0};
  /// The occupied eigenvalues themselves, in the declared order.
  std::vector<std::complex<double>> occupiedEigenvalues;
  /// \f$ |\lambda_{\rm first\ empty}-\lambda_{\rm last\ occupied}| \f$, the gap
  /// that isolates the occupied band. Quiet NaN when every mode is occupied or
  /// none is.
  double spectralGap = 0.0;
  /// Whether the inner geometry relaxation reached its own tolerance.
  bool geometryConverged = false;
  /// The inner relaxation's residual norm when it stopped.
  double geometryResidualNorm = 0.0;
};

/// # SelfConsistentMeanFieldReport
///
/// What a self-consistent solve reached.
struct SelfConsistentMeanFieldReport {
  /// Every outer iteration, in order.
  std::vector<SelfConsistentMeanFieldStep> steps;
  /// Whether both fixed-point conditions held at the declared tolerance.
  bool converged = false;
  /// The stationarity force norm at the point the solve stopped at.
  double forceNorm = 0.0;
  /// The last movement of the covariance.
  double covarianceChange = 0.0;
  /// \f$ \lVert\Gamma^2-\Gamma\rVert_F \f$ at the point the solve stopped at.
  double purityDefect = 0.0;
  /// \f$ \Gamma^{*} \f$, flat row-major over the carrier degree's cells.
  std::vector<std::complex<double>> covariance;
  /// The occupied eigenvalues of \f$ h(z^{*}) \f$.
  std::vector<std::complex<double>> occupiedEigenvalues;
  /// \f$ \operatorname{tr}(\Gamma^{*}h(z^{*})) \f$.
  std::complex<double> occupiedEnergy{0.0, 0.0};
  /// The gap above the occupied band at \f$ z^{*} \f$.
  double spectralGap = 0.0;
  /// The complex action at \f$ (z^{*},\Gamma^{*}) \f$.
  std::complex<double> action{0.0, 0.0};
};

/// # SelfConsistentMeanField
///
/// The certificates-blind mean-field backreaction of Section 7 of the
/// whitepaper, solved to self-consistency.
///
/// Reference: Landau, "Über die Bewegung der Elektronen im Kristallgitter",
/// Physikalische Zeitschrift der Sowjetunion 3, 664 (1933); Pekar, "Local
/// quantum states of electrons in an ideal ion crystal", Zhurnal
/// Eksperimentalnoi i Teoreticheskoi Fiziki 16, 341 (1946) — the self-consistent
/// polaron this construction is the complex-first form of.
/// Reference: Bach, Lieb and Solovej, "Generalized Hartree-Fock theory and the
/// Hubbard model", Journal of Statistical Physics 76, 3 (1994), for the
/// quasi-free closure of a covariance coupled to its own mean field.
///
/// The only channel from the carried state to the geometry is the bilinear
/// action density \f$ \tilde\psi^{\mathsf T}h(z,U)\psi \f$ evaluated on
/// \f$ \Gamma \f$ by Wick reduction, so the force on a geometric coordinate is
/// the Hellmann-Feynman term
/// \f$ \operatorname{tr}(\Gamma\,\partial h/\partial z_e) \f$ and the force on a
/// link is \f$ \operatorname{tr}(\Gamma\,U_e\,\partial h/\partial U_e) \f$.
/// Both are complex and neither is projected onto a real part. No cluster,
/// fiber, colour, exchange or baryon observable enters, which is the firewall
/// the sub-mode's name records.
///
/// The solve alternates the two halves of the fixed-point condition:
///
/// 1. with \f$ \Gamma \f$ held, the geometry is made stationary against the
///    whole joint action by `HolomorphicRelaxation`, so that
///    \f$ \partial S/\partial z=0 \f$ and \f$ U\,\partial S/\partial U=0 \f$;
/// 2. with the geometry held, \f$ \Gamma \f$ is replaced by the spectral
///    projector onto the occupied modes of \f$ h(z,U) \f$ at that geometry.
///
/// A fixed point of the pair is the self-consistent polaron the whitepaper
/// names: \f$ \Gamma^{*} \f$ is a projector onto modes of \f$ h(z^{*}) \f$, and
/// the state's force balances the geometric action edge by edge. It is a
/// stationary point of a complex action rather than a minimum of a real one,
/// and the report carries the residuals that certify it as such.
///
/// The iteration stays inside the Gaussian class throughout: the covariance is
/// idempotent at every step by construction, and `purityDefect` measures that
/// closure rather than assuming it.
class SelfConsistentMeanField {
 public:
  /// Build a solve over an action.
  ///
  /// @param action The joint action. Its declared covariance is the starting
  ///   \f$ \Gamma \f$; when it is empty the solve starts from the spectral
  ///   projector of the carrier operator at the initial geometry, which is the
  ///   uniform-seeded start. The complex the action refers to is the object the
  ///   solve writes.
  /// @param declaration The occupation rule and the convergence controls.
  /// @throws std::invalid_argument when the mixing is outside \f$ (0,1] \f$, or
  ///   when no mode is declared occupied, which leaves the matter term
  ///   identically zero and the self-consistency empty.
  SelfConsistentMeanField(JointAction action,
                          SelfConsistentMeanFieldDeclaration declaration);

  /// Run the solve, writing the relaxed geometry into the complex as it goes.
  [[nodiscard]] SelfConsistentMeanFieldReport solve();

  /// The action, carrying the covariance and the multipliers as the solve left
  /// them.
  [[nodiscard]] const JointAction &action() const noexcept { return action_; }

 private:
  JointAction action_;
  SelfConsistentMeanFieldDeclaration declaration_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_SELFCONSISTENTMEANFIELD_H
