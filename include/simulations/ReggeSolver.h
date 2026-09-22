// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include "mesh/ForwardDeclarations.h"
#include "matter/MatterConfiguration.h"
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/SparseCore>


// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;


/// Regge equation solver.
///
/// Given a triangulated spacetime and a matter configuration, adjusts the
/// edge lengths so that the deficit angles at every hinge satisfy the
/// discretized Einstein equations:
///
/// \f[
///   \frac{\partial S}{\partial \ell^2_e} = 0
/// \f]
///
/// The total action is:
/// \f[
///   S = \underbrace{\sum_h A_h\,\varepsilon_h}_{S_{\text{grav}}}
///     \underbrace{- M \sum_{e \in W} \sqrt{-\ell^2_e}}_{S_{\text{matter}}}
/// \f]
/// (Timelike edges have \f$\ell^2 < 0\f$; spacelike edges have \f$\ell^2 > 0\f$.)
///
/// ## Algorithm
///
/// 1. Compute the Gram matrix \f$G\f$ of each top-simplex from its edge
///    lengths.
/// 2. Derive dihedral angles from the cofactors of \f$G\f$.
/// 3. Sum dihedral angles at each hinge → actual deficit angles.
/// 4. Minimize \f$F = \|\nabla S\|^2 = \sum_e (\partial S/\partial \ell^2_e)^2\f$
///    by gradient descent.  \f$F = 0\f$ exactly at a stationary point of
///    the action, i.e. when the Regge equations are satisfied.
///    (Minimizing \f$S\f$ directly diverges because it is unbounded below.)
///
/// Reference: Regge, "General relativity without coordinates", Nuovo Cimento
/// 19, 558 (1961).
///
class ReggeSolver {
  public:
    ReggeSolver(std::shared_ptr<Spacetime> spacetime,
                MatterConfiguration matter);

    // ==================== Geometry queries ====================

    /// Dihedral angle at hinge \a h within top-simplex \a sigma.
    [[nodiscard]] std::complex<double> dihedralAngle(SimplexPtr sigma,
                                        SimplexPtr hinge) const;

    /// Deficit angle at a hinge: \f$\varepsilon_h = 2\pi - \sum_\sigma \theta_h^{(\sigma)}\f$.
    [[nodiscard]] std::complex<double> deficitAngle(SimplexPtr hinge) const;

    /// Area of a triangular hinge (for the Regge action weighting).
    [[nodiscard]] static std::complex<double> hingeArea(SimplexPtr hinge);

    /// Gravitational Regge action \f$S_{\text{grav}} = \sum_h A_h\,\varepsilon_h\f$,
    /// with \f$A_h\f$ the signed Lorentzian hinge area and \f$\varepsilon_h\f$
    /// the complex Lorentzian deficit angle.
    [[nodiscard]] std::complex<double> reggeAction() const;

    /// Dual Lorentzian Regge action on \f$W^*\f$:
    /// \f$S_{\text{Regge}}(W^*) = \sum_h |\!\star\! h|\,\varepsilon_h\f$. Each
    /// \f$(d\!-\!2)\f$-hinge contributes its circumcentric dual volume
    /// (``Simplex::dualVolume``) weighted by its complex Lorentzian deficit angle
    /// (``Simplex::deficitAngle``). In the returned ``std::complex`` the real part
    /// is the angle-defect curvature and the imaginary part the boost (rapidity)
    /// content of hinges with a timelike normal plane. Pure gravity:
    /// matter-independent. References: Regge (1961); Ambjorn, Jurkiewicz & Loll,
    /// arXiv:hep-th/0105267; Sorkin, Lorentzian angles, arXiv:1908.10022;
    /// Asante, Dittrich & Padua-Arguelles, arXiv:2104.00485.
    [[nodiscard]] std::complex<double> dualReggeAction() const;

    /// The \f$(d\!-\!2)\f$ hinge vertex-tuples that are faces of the given top
    /// \f$d\f$-cells — the hinges whose dual-action contribution a move over those
    /// cells can change. Each input cell is a vertex-id tuple; the result is the
    /// dedup'd set of \f$(d\!-\!1)\f$-vertex sub-tuples (sorted ids). Pure
    /// topology, no geometry. Build it once from a move's touched cells (created,
    /// removed, and the top cofaces of a perturbed edge), then use the same set
    /// for the before and after legs of ``dualReggeActionOverHinges``.
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> hingeFacesOfCells(
        const std::vector<std::vector<std::uint64_t>> &cells) const;

    /// The dual (Sorkin) Regge action
    /// \f$\sum_{h} |\!\star\! h|\,\varepsilon_h\f$ restricted to the given
    /// \f$(d\!-\!2)\f$ hinge tuples that are genuine in the live complex (a
    /// registered simplex with a top coface; orphans contribute 0, as in
    /// ``dualReggeAction``). Each tuple is resolved by vertex id
    /// (order-independent, ``findSimplexByVerts``) and the per-term measure is the
    /// circumcentric ``dualVolume`` used by ``dualReggeAction``. Evaluated over a
    /// fixed affected-hinge set across a move,
    /// \f$\Delta S = S_{\text{after}} - S_{\text{before}}\f$ is exact: every
    /// hinge outside the set is unchanged and cancels.
    [[nodiscard]] std::complex<double> dualReggeActionOverHinges(
        const std::vector<std::vector<std::uint64_t>> &hinges) const;

    /// The edges whose per-edge action gradient `∂S/∂ℓ²_e` a move over the given
    /// top cells can change — the affected-edge index for the incremental
    /// `Δ‖∇S_Regge‖²`. A move changes `∂S/∂ℓ²_e` only when an affected hinge
    /// (`hingeFacesOfCells(cells)`) contributes to `e`, i.e. only for edges that
    /// share a top cell with an affected hinge. Returns those edges as sorted
    /// `(a,b)` id pairs (deduped). Cells are both added and removed by a move, so
    /// take the union of this set evaluated before and after it, then use that
    /// fixed set for both legs of `gradientNorm2OverEdges`.
    [[nodiscard]] std::vector<std::pair<std::uint64_t, std::uint64_t>>
    affectedEdgesOfCells(
        const std::vector<std::vector<std::uint64_t>> &cells) const;

    /// The squared gradient norm `Σ_{e∈edges} |∂S/∂ℓ²_e|²` of the dual (Sorkin)
    /// Regge action restricted to the given edges — the geometry term of the
    /// optimizer objective `F = ‖∇S_Regge‖² + Γ·r_U`, zero at a stationary action.
    /// Each `∂S/∂ℓ²_e` is the full per-edge gradient
    /// `Σ_{h∋e}[∂|★h|·ε_h + |★h|·∂ε_h]` over all hinges incident to `e` (its
    /// star), built from the same per-hinge analytic gradients as
    /// `actionGradientExact`; the complex modulus keeps Re and Im together. Over a
    /// fixed affected-edge set across a move, `Δ‖∇S_Regge‖² = after − before` is
    /// exact: every edge outside the set keeps its gradient and cancels. Over all
    /// edges it equals `Σ_e |actionGradientExact()_e|²`.
    [[nodiscard]] double gradientNorm2OverEdges(
        const std::vector<std::pair<std::uint64_t, std::uint64_t>> &edges) const;

    /// Point-particle matter action:
    /// \f$S_{\text{matter}} = -M \sum_{e \in W,\ e\ \text{timelike}} \sqrt{-\mathrm{Re}\,\ell^2_e}\f$.
    /// Causal character is the canonical ``Edge::isTimelike()``; null edges
    /// contribute 0. The resident \f$\ell^2\f$ is real and signed
    /// (``Edge::setSquaredLength``), so
    /// \f$\sqrt{-\mathrm{Re}\,\ell^2} = \sqrt{-\ell^2}\f$. Real by
    /// construction.
    [[nodiscard]] double matterAction() const;

    /// Total action: \f$S = S_{\text{grav}} + S_{\text{matter}}\f$.
    /// The Regge equations are \f$\partial S/\partial \ell^2_e = 0\f$.
    [[nodiscard]] std::complex<double> totalAction() const;

    /// Squared gradient norm: \f$F = \sum_e (\partial S/\partial \ell^2_e)^2\f$.
    /// Non-negative; zero exactly when the Regge equations are satisfied.
    [[nodiscard]] double actionGradientNorm() const;

    /// Exact analytic gradient of the complex dual (Sorkin) Regge action
    /// ``dualReggeAction`` = Σ_h |★h|·ε_h: ∂S/∂ℓ²_e for each edge, in
    /// ``getEdgeList()`` order (matching ``actionGradient``). Assembled by the
    /// product rule Σ_h [∂|★h|·ε_h + |★h|·∂ε_h] from the per-hinge analytic
    /// gradients ``Simplex::dualVolumeGradient`` and ``deficitAngleGradient``,
    /// with no finite differences. Complex: Re S and Im S together. Matches a
    /// central difference of ``dualReggeAction`` to machine precision, in one
    /// pass rather than 2·|E| action evaluations. Finite where the circumcentres
    /// of the dual coincide, as on a Kuhn torus (``Simplex::dualVolumeGradient``).
    [[nodiscard]] std::vector<std::complex<double>> actionGradientExact() const;

    /// Exact analytic Hessian ∂²S/∂ℓ²_e∂ℓ²_f of the dual Lorentzian Regge action,
    /// as a dense |E|×|E| complex matrix in ``getEdgeList`` order:
    /// Σ_h [∂²|★h|·ε_h + ∂|★h|_e·∂ε_h_f + ∂|★h|_f·∂ε_h_e + |★h|·∂²ε_h], assembled
    /// from the per-hinge ``dualVolumeHessian`` / ``deficitAngleHessian`` and
    /// their gradients, with no finite differences. Supports exact Newton /
    /// Gauss-Newton steps in the stationary-action relaxation. Finite where the
    /// circumcentres of the dual coincide, as on a Kuhn torus.
    [[nodiscard]] std::vector<std::vector<std::complex<double>>>
    actionHessianExact() const;

    /// Sparse assembly of the exact analytic Hessian ``actionHessianExact``.
    /// ∂²S/∂ℓ²_e∂ℓ²_f is nonzero only when edges e,f share a hinge (local
    /// coupling), so the Hessian is assembled directly as an Eigen
    /// ``SparseMatrix`` at O(nnz) memory instead of O(|E|²). Same per-hinge
    /// product-rule terms as the dense ``actionHessianExact``
    /// (``hingeHessianEntries``), equal to it to machine precision on the nonzero
    /// pattern. ``getEdgeList`` order; column-major.
    [[nodiscard]] Eigen::SparseMatrix<std::complex<double>>
    actionHessianExactSparse() const;


    // ==================== Accessors ====================

    [[nodiscard]] const std::shared_ptr<Spacetime> &getSpacetime() const noexcept {
        return spacetime_;
    }

    [[nodiscard]] const MatterConfiguration& getMatter() const noexcept {
        return matter_;
    }

  private:
    std::shared_ptr<Spacetime> spacetime_;
    MatterConfiguration matter_;

    /// Collect all (d-2)-simplices (hinges) in the complex.
    [[nodiscard]] std::vector<SimplexPtr> collectHinges() const;

    /// All (edgeI, edgeJ, ∂²S term) contributions of a single hinge to the action
    /// Hessian, with edge indices resolved via @p eidx. The per-hinge
    /// product-rule kernel shared by the dense ``actionHessianExact`` and the
    /// sparse ``actionHessianExactSparse`` assemblies.
    [[nodiscard]] std::vector<
        std::tuple<std::size_t, std::size_t, std::complex<double>>>
    hingeHessianEntries(
        const SimplexPtr &hinge,
        const std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>
            &eidx) const;

    /// Compute the gradient of the total action: ∂S/∂ℓ²_e for each edge.
    [[nodiscard]] std::vector<std::complex<double>> actionGradient() const;


};

} // namespace tessera::simulations
