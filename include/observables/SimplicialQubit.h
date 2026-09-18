// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H
#define TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::chainhodge { class Connection; }

namespace tessera::observables {

using namespace ::tessera::spacetime;

/// # SimplicialQubit
///
/// A single qubit state encoded as the holomorphic line in the harmonic space
/// of the metric Hodge Laplacian on a triangulated torus. The input is
/// intrinsic geometry — a simplicial complex \f$ K \cong T^2 \f$ with edge
/// lengths (real or complex) and a marked cycle pair \f$ (A, B) \f$ with
/// \f$ A \cdot B = +1 \f$ — together with the pure-gauge link phases of its
/// edges read from a `Spacetime`. The output is a point of
/// \f$ \mathbb{CP}^1 \f$.
///
/// Reference: Mercat, "Discrete Riemann Surfaces and the Ising model",
/// arXiv:0909.3600.
///
/// ## Construction
///
///  - The input is validated on load: every edge in exactly two faces,
///    \f$ n_V - n_E + n_F = 0 \f$, consistent face orientations, the strict
///    triangle inequality on every face (on the real locus), and closed,
///    homologically independent marked cycles.
///  - The incidence matrices \f$ d_0 \f$ (\f$ n_E \times n_V \f$) and
///    \f$ d_1 \f$ (\f$ n_F \times n_E \f$) satisfy \f$ d_1 d_0 = 0 \f$ exactly.
///  - Per triangle: the angles by the law of cosines, the area by Heron's
///    formula, and the intrinsic planar layout \f$ p_i = (0,0),\ p_j = (c,0),\
///    p_k = (b\cos\alpha_i, b\sin\alpha_i) \f$ that every per-face vector lives
///    in. Frames of different faces are never compared.
///  - The cotangent weights
///    \f$ w_e = \tfrac12(\cot\alpha_e + \cot\beta_e) \f$ give
///    \f$ M_1 = \mathrm{diag}(w_e) \f$, with negative weights and intrinsic
///    Delaunay violations \f$ \alpha_e + \beta_e > \pi \f$ flagged. An optional
///    intrinsic Delaunay edge-flip pass is `intrinsicDelaunay()`.
///  - The harmonic space \f$ H = \ker[d_1;\ d_0^T M_1] \f$ of closed and
///    co-closed 1-cochains has dimension 2, the topological input, and is
///    computed by a dense singular value decomposition null space.
///  - The \f$ L^2 \f$ inner product of 1-cochains goes through Whitney
///    1-forms: barycentric gradients per face in the local frame, the
///    interpolant \f$ W_t(\omega) \f$ at the barycenter, and
///    \f$ \langle\omega,\eta\rangle = \sum_t A_t\, W_t(\omega)\cdot
///    W_t(\eta) \f$.
///  - The complex structure comes from rotate-then-project: the Gram matrix
///    \f$ G \f$, the rotation pairing \f$ R_{ab} = \sum_t A_t\,
///    \mathrm{rot}_{90}(W_t(h_a))\cdot W_t(h_b) \f$, and
///    \f$ J = G^{-1}R^T \f$, the metric input. The residual
///    \f$ \|J^2 + I\|_F \f$ is exposed and never symmetrized away.
///  - The holomorphic line is the eigenvector of \f$ J \f$ for the eigenvalue
///    nearest \f$ -i \f$ (convention \f$ \star dz = -i\,dz \f$). Its periods
///    \f$ P_A, P_B \f$ over the marking give \f$ \tau = P_B/P_A \f$ in the
///    upper half plane; the conjugate branch is taken when
///    \f$ \mathrm{Im}\,\tau < 0 \f$, and the marking \f$ (B, -A) \f$ with
///    \f$ -1/\tau \f$ when \f$ |P_A| \f$ vanishes.
///  - \f$ |\psi\rangle = (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$,
///    with its Bloch vector (unit, asserted) and density matrix.
///  - The Fubini-Study and Weil-Petersson distances are two separate
///    functions.
///  - Degeneration diagnostics: \f$ \mathrm{cond}(M_1) \f$ and
///    \f$ \mathrm{cond}(G) \f$ against a configurable threshold, warning
///    rather than failing.
///
/// The dimension of \f$ H \f$ is topological (\f$ b_1 = 2 \f$) and the line
/// inside it is metric, so the qubit is exactly the metric's contribution: the
/// conformal structure of the torus, one point of the upper half plane once a
/// marking is fixed. Both halves are computed from the lengths and the marking
/// alone, which is what lets a geometric process reach a state by moving
/// lengths. The construction is exact on flat tori and first-order accurate in
/// the mesh size otherwise, which is why the flat torus
/// \f$ \mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z}) \f$ of `flatTorus` is the
/// reference: it returns \f$ \tau \f$ to rounding at every resolution.
///
/// The marking is stored with the complex because \f$ \tau \f$ depends on
/// \f$ (A, B) \f$ up to \f$ SL(2,\mathbb{Z}) \f$, so a canonical state needs
/// the marking fixed and carried along. Coverage is one open hemisphere of the
/// Bloch sphere; the other requires the opposite surface orientation, which is
/// the faces given with the opposite cyclic order, or the `reversed` flag of
/// the `Spacetime` constructor, whose container stores no orientation. There is
/// no action of \f$ SU(2) \f$ on edge lengths: gates act on `state()` as
/// \f$ 2\times 2 \f$ matrices and are never pulled back.
///
/// A `Spacetime` sits underneath because it is this repository's container for
/// vertices, edges, lengths and link phases, so a qubit built from the raw data
/// structures also exists as a `Spacetime` (`spacetime()`), and a torus that
/// already lives as a `Spacetime` can be read directly. The container sorts the
/// vertices of each face, so the consistently oriented faces are held here.
///
/// ## Complex geometry
///
/// The real locus is the set of inputs with every length real
/// (\f$ \mathrm{Im}\,\ell_e = 0 \f$) and every link exactly 1
/// (`onRealLocus()`). On it the construction above runs over real numbers,
/// bit-identical to the construction over real lengths. Off it every formula
/// above is taken over \f$ \mathbb{C} \f$, under these rules.
///
/// The real reference of a torus is the same complex with every squared edge
/// length equal to 1: the unit equilateral triangle on every face, which is
/// `chainhodge::WhitneyMass::Branch::Continuation`'s reference simplex
/// \f$ g_{\rm ref} = \tfrac12(1 + \delta_{ij}) \f$, carrying the same marking
/// and links. Every root and branch choice off the real locus is continued from
/// it along the straight segment
/// \f$ s_e(t) = (1 - t) + t\,\ell_e^2 \f$, \f$ t \in [0, 1] \f$, in the squared
/// lengths:
///  - the angles are the principal branch of \f$ \arccos \f$ of the complex
///    cosine of the law of cosines. The Heron area is
///    \f$ \sqrt{\det g_t}/2 \f$ on the continuation branch,
///    `WhitneyMass::volumeOnBranch` of the face's Gram matrix
///    \f$ (g_t)_{ij} = \tfrac12(s_{0i} + s_{0j} - s_{ij}) \f$, with the
///    argument of \f$ \det g_t(t) \f$ tracked through its roots; a root on the
///    segment leaves no continuous branch and is refused by name. The layout
///    takes \f$ b\sin\alpha_i := 2A_t/c \f$ on that same branch, so the
///    identity \f$ A_t = \tfrac12 bc\sin\alpha_i \f$ holds exactly and the
///    barycentric gradients satisfy
///    \f$ \nabla\lambda_v\cdot(p_v - p_u) = 1 \f$. The cotangents are
///    \f$ \cot\alpha_v = (\text{adjacent}^2 + \text{adjacent}^2 -
///    \text{opposite}^2)/(4A_t) \f$ on the same branch;
///  - the harmonic space is the complex null space (complex singular value
///    decomposition) of the stacked matrix, whose incidences are twisted by
///    the links (below);
///  - the pairings are the transpose (bilinear) pairing
///    \f$ a\cdot b = a_1 b_1 + a_2 b_2 \f$, never a conjugate: \f$ G \f$ is
///    complex symmetric on the trivial connection and \f$ J = G^{-1}R^T \f$ is
///    complex;
///  - the eigenline is chosen by continuity from the real reference. At the
///    reference the rule above (eigenvalue nearest \f$ -i \f$, the other
///    eigenline when \f$ \mathrm{Im}\,\tau < 0 \f$) selects the line, and the
///    holomorphic form is then tracked along the segment by the largest overlap
///    between the eigenlines of consecutive points, halving the step until the
///    overlap exceeds 0.99; two eigenlines that cannot be told apart are
///    refused by name. \f$ \mathrm{Im}\,\tau > 0 \f$ is not a criterion off the
///    real locus: \f$ \tau \f$ may land anywhere in \f$ \mathbb{C} \f$ and the
///    state may lie in either hemisphere;
///  - the state and distance formulas in \f$ \tau \f$ are unchanged;
///    \f$ |\vec r| = 1 \f$ is an algebraic identity in \f$ \tau \f$ and still
///    holds; the Weil-Petersson distance keeps its upper-half-plane domain.
///
/// Link phases are a pure gauge \f$ U = 1^g \f$,
/// \f$ U_{xy} = g_x^{-1}g_y \f$, read from the `Spacetime` edges by the
/// `chainhodge::Connection::fromSpacetime` convention
/// (\f$ U_{xy} = e^{i\varphi} \f$ when the source is \f$ x < y \f$,
/// \f$ e^{-i\varphi} \f$ otherwise). A connection that is not a pure gauge —
/// flux through a face (\f$ \mathcal F_t \ne 1 \f$), or a flat connection with
/// holonomy around a cycle of the 1-skeleton — is refused by name, since its
/// twisted kernel does not have dimension 2. Under a pure gauge the phases
/// enter as in `chainhodge::CovariantChainHodge`, with the base vertex
/// \f$ b(\sigma) = \min\sigma \f$ of every cell:
///  - the incidences of the stacked matrix are twisted,
///    \f$ (d_1^U)_{te} = (d_1)_{te}\,U_{b(t)b(e)} \f$, carrying each edge value
///    to the face's base, and \f$ (\partial_1^U M_1)_{ve} = (\partial_1)_{ve}\,
///    U_{v\,b(e)}\,w_e \f$, carrying each weighted edge value to the vertex.
///    The kernel is \f$ H^U = \rho_1 H \f$ with
///    \f$ \rho_1 = \mathrm{diag}(g_{b(e)}^{-1}) \f$, a twisted section, and the
///    dual kernel \f$ H^\vee = \rho_1^{-1}H \f$ is the same construction under
///    \f$ U^{-1} \f$ (`dualHarmonicBasis()`);
///  - the Whitney interpolant carries each edge value to the face's base
///    (\f$ W_t^U \f$ with \f$ U_{b(t)b(e)} \f$, \f$ W_t^{U^{-1}} \f$ with the
///    inverse) and the pairings are between the kernel and the dual kernel,
///    \f$ G_{ab} = \sum_t A_t\, W_t^{U^{-1}}(h^\vee_a)\cdot W_t^U(h_b) \f$ and
///    \f$ R_{ab} = \sum_t A_t\, \mathrm{rot}_{90}(W_t^U(h_a))\cdot
///    W_t^{U^{-1}}(h^\vee_b) \f$. This transpose pairing is the only pairing
///    invariant under a gauge, so \f$ J = G^{-1}R^T \f$ is the matrix of the
///    same operator on \f$ H^U \f$ and its eigenline is \f$ \rho_1\omega \f$;
///  - the periods are taken with parallel transport along the marked cycles
///    (`chainhodge::Connection::transportedPeriod`), both from one base point:
///    the first vertex of \f$ A \f$ that lies on \f$ B \f$ (`baseVertex()`).
///    That makes each period \f$ g_{v_0}^{-1} \f$ times the untwisted one, and
///    makes \f$ \tau \f$, the state and the coefficient pairs in the period
///    frame exactly gauge invariant. On the trivial connection the periods are
///    plain signed sums and the marked cycles need not chain in the given
///    order; with a nontrivial connection each marked cycle is walked as one
///    closed walk, its steps ordered by Hierholzer's algorithm and the given
///    order kept when it already chains. A cycle whose steps do not form one
///    closed walk, or two cycles without a common vertex, are refused by name.
///
/// The weight flags and `intrinsicDelaunay()` are inequalities on real angles
/// and are evaluated on the real locus only; the pass refuses off it.
class SimplicialQubit {
  public:
    using Complex = std::complex<double>;
    /// An edge \f$ (i, j) \f$ with \f$ i < j \f$, oriented \f$ i \to j \f$.
    using EdgePair = std::pair<std::uint64_t, std::uint64_t>;
    /// A face \f$ (i, j, k) \f$ in its counterclockwise order.
    using Face = std::array<std::uint64_t, 3>;
    /// One step of a marked cycle: (edge index into `edges()`, sign \f$ \pm 1 \f$)
    /// — \f$ +1 \f$ traverses the edge along its stored orientation.
    using CycleStep = std::pair<std::size_t, int>;
    using Cycle = std::vector<CycleStep>;
    /// A marked cycle as a closed walk of directed vertex steps
    /// (`chainhodge::Connection::Walk`), in the torus's vertex indices.
    using Walk = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

    /// Construct from the raw data structures.
    /// @param vertices \f$ V = [0 .. n_V - 1] \f$.
    /// @param edges \f$ E = [(i, j)] \f$, \f$ i < j \f$.
    /// @param faces \f$ F = [(i, j, k)] \f$, consistently oriented.
    /// @param lengths \f$ \ell : E \to \mathbb{C} \f$, one per edge, in edge
    ///   order: real and positive on the real locus, and nonzero and finite
    ///   off it, where only \f$ \ell^2 \f$ enters the construction. The link
    ///   phases are zero, i.e. the trivial connection.
    /// @param cycleA, cycleB The marked cycles as (edge index, sign) lists,
    ///   closed loops with \f$ A \cdot B = +1 \f$.
    /// @param degeneracyThreshold The condition-number level above which a
    ///   warning is recorded.
    /// @throws std::invalid_argument when an input validation fails, or a
    ///   branch cannot be continued from the real reference.
    /// @throws std::runtime_error when \f$ \dim H \ne 2 \f$: not a torus, or
    ///   degenerate weights.
    SimplicialQubit(std::vector<std::uint64_t> vertices, std::vector<EdgePair> edges,
                    std::vector<Face> faces, std::vector<Complex> lengths, Cycle cycleA,
                    Cycle cycleB, double degeneracyThreshold = 1e8);

    /// The same qubit read from a `Spacetime` of dimension 2: its vertices
    /// (indexed by ascending id), its edges (ascending \f$ (i, j) \f$ order,
    /// which is the edge order the cycles refer to), its triangles, its edge
    /// lengths (real positive or complex) and its edge phases as the
    /// pure-gauge link connection. The container stores no face
    /// orientation, so the consistent orientation is the one making the top
    /// chain a cycle (`cobordism::ChainComplex::fundamentalClass`), reversed
    /// when \p reversed is set, which selects the other hemisphere.
    /// @throws std::invalid_argument when a length is zero, non-finite, or
    ///   real and non-positive, the surface is not closed-orientable, the
    ///   phases are not a pure gauge (flux or holonomy, named), or an input
    ///   validation fails.
    SimplicialQubit(const std::shared_ptr<Spacetime> &spacetime, Cycle cycleA, Cycle cycleB,
                    bool reversed = false, double degeneracyThreshold = 1e8);

    // ----- readouts -----------------------------------------------------------

    /// \f$ H \f$: \f$ n_E \times 2 \f$, a unitary-orthonormal basis of the null
    /// space of the twisted stacked matrix
    /// \f$ [d_1^U;\ \partial_1^U M_1] \f$; real on the real locus.
    [[nodiscard]] const Eigen::MatrixXcd &harmonicBasis() const noexcept { return H_; }
    /// \f$ H^\vee \f$: the null space of the stacked matrix twisted by
    /// \f$ U^{-1} \f$: the dual kernel the pairings are taken against. The same
    /// matrix as `harmonicBasis()` on the trivial connection.
    [[nodiscard]] const Eigen::MatrixXcd &dualHarmonicBasis() const noexcept { return Hdual_; }
    /// \f$ J = G^{-1}R^T \f$ in the basis \f$ \{h_1, h_2\} \f$,
    /// \f$ 2 \times 2 \f$; real on the real locus.
    [[nodiscard]] const Eigen::MatrixXcd &complexStructure() const noexcept { return J_; }
    /// \f$ \|J^2 + I\|_F \f$: the discretization-error diagnostic.
    [[nodiscard]] double jResidual() const noexcept { return jResidual_; }
    /// \f$ \omega = c_0 h_1 + c_1 h_2 \f$, the complex 1-cochain spanning the
    /// holomorphic line, after the branch and marking rules; a twisted
    /// section (values in the frame at each edge's base vertex) under a
    /// nontrivial connection.
    [[nodiscard]] const Eigen::VectorXcd &holomorphicForm() const noexcept { return omega_; }
    /// \f$ (P_A, P_B) \f$ of the holomorphic form over the marking in force,
    /// taken with parallel transport from `baseVertex()` under a nontrivial
    /// connection.
    [[nodiscard]] std::pair<std::complex<double>, std::complex<double>> periods() const noexcept {
      return {periodA_, periodB_};
    }
    /// \f$ \tau = P_B / P_A \f$.
    [[nodiscard]] std::complex<double> tau() const noexcept { return tau_; }
    /// The period frame of the torus: the basis \f$ (f_A, f_B) \f$ of its
    /// harmonic space with periods \f$ (1, 0) \f$ and \f$ (0, 1) \f$ over the
    /// marking in force. An \f$ n_E \times 2 \f$ matrix in the torus's edge
    /// order, equal to `harmonicBasis()` times the inverse of the period
    /// matrix, \f$ F = H\,\Pi^{-1} \f$ with
    /// \f$ \Pi_{ca} = \oint_c h_a \f$ (rows the cycles \f$ A, B \f$, columns
    /// the basis elements \f$ h_1, h_2 \f$), so that
    /// \f$ \oint_A f_A = 1,\ \oint_B f_A = 0,\ \oint_A f_B = 0,\
    /// \oint_B f_B = 1 \f$.
    ///
    /// The frame supplies the coordinates a state is written in. Every harmonic
    /// form is \f$ \omega = P_A f_A + P_B f_B \f$ with its own periods as the
    /// coefficients, so the holomorphic line is the column combination
    /// \f$ (1, \tau) \f$ of the frame: `holomorphicForm()` equals
    /// `periodFrame()` \f$ \cdot (P_A, P_B)^T = P_A\,F (1, \tau)^T \f$, the
    /// qubit \f$ |0\rangle + \tau|1\rangle \f$ read as a 1-form with
    /// \f$ f_A \leftrightarrow |0\rangle \f$ and
    /// \f$ f_B \leftrightarrow |1\rangle \f$. A two-body target
    /// \f$ \chi \f$ is written in the \f$ |0\rangle, |1\rangle \f$ bases of
    /// two qubits, so the transfer of a cobordism between two tori compares
    /// with it when read in the period frames of the two boundary tori
    /// (`cobordism::MultiCobordism::setInputFrame`), where it is
    /// \f$ 2 \times 2 \f$.
    ///
    /// \f$ \tau \f$ is reported over the marking in force — \f$ (B, -A) \f$
    /// when \f$ |P_A| \f$ vanishes, see `markingSwapped()` — and the frame is
    /// the basis \f$ \tau \f$ is a coordinate in, so `periods()` are always
    /// the coefficients of `holomorphicForm()` in it. The frame is real on the
    /// real locus and complex off it, invariant under a common scale of the
    /// lengths, independent of the orthonormal basis the null-space solve
    /// happens to return, and always defined, since the period map of the
    /// harmonic space over a homology basis is an isomorphism and the
    /// independence check guarantees it. Under a nontrivial connection the
    /// periods are the transported ones from `baseVertex()`: the frame is then
    /// \f$ g_{v_0}\,\rho_1 F \f$ of the untwisted torus, and the coefficient
    /// pair of any twisted section in it is \f$ g_{v_0}^{-1} \f$ times the
    /// untwisted pair, hence invariant as a point of
    /// \f$ \mathbb{CP}^1 \f$.
    [[nodiscard]] const Eigen::MatrixXcd &periodFrame() const noexcept { return F_; }
    /// \f$ (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$.
    [[nodiscard]] Eigen::VectorXcd state() const;
    /// \f$ (2\,\mathrm{Re}\,\tau,\ 2\,\mathrm{Im}\,\tau,\
    /// 1 - |\tau|^2)/(1+|\tau|^2) \f$: a unit vector for every finite
    /// \f$ \tau \f$, on and off the real locus.
    [[nodiscard]] Eigen::VectorXd bloch() const;
    /// \f$ \rho = \tfrac12(I + \vec r\cdot\vec\sigma) \f$.
    [[nodiscard]] Eigen::MatrixXcd densityMatrix() const;
    /// The flat torus \f$ \mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z}) \f$: the
    /// unit cell spanned by \f$ 1 \f$ and \f$ \tau \f$ as an \f$ n_x \times
    /// n_y \f$ grid, every square split by its diagonal, sides identified;
    /// edge lengths are the Euclidean lengths of the lattice displacements;
    /// \f$ A \f$ is the row loop along \f$ 1 \f$ and \f$ B \f$ the column loop
    /// along \f$ \tau \f$. Exact: the read returns \f$ \tau \f$ to rounding.
    /// @throws std::invalid_argument unless \f$ \mathrm{Im}\,\tau > 0 \f$ and
    ///   \f$ n_x, n_y \ge 3 \f$ (below 3 the grid is not a simplicial complex).
    [[nodiscard]] static SimplicialQubit flatTorus(std::complex<double> tau, int nx, int ny);
    /// \f$ d_{FS} = \arccos\big(|1 + \bar\tau_1\tau_2| / \sqrt{(1+|\tau_1|^2)(1+|\tau_2|^2)}\big) \f$:
    /// distinguishability of the two states; curvature \f$ +4 \f$.
    [[nodiscard]] static double fubiniStudyDistance(const SimplicialQubit &q1,
                                                    const SimplicialQubit &q2);
    /// \f$ d_{WP} = \operatorname{arccosh}\big(1 + |\tau_1 - \tau_2|^2 /
    /// (2\,\mathrm{Im}\,\tau_1\,\mathrm{Im}\,\tau_2)\big) \f$: the moduli
    /// distance between the two shapes; curvature \f$ -1 \f$. Kept separate
    /// from \f$ d_{FS} \f$: the two are conformally equivalent, not
    /// isometric.
    [[nodiscard]] static double weilPeterssonDistance(const SimplicialQubit &q1,
                                                      const SimplicialQubit &q2);

    // ----- the derivative of tau ----------------------------------------------

    /// \f$ \partial\tau/\partial z_e \f$ for every edge, \f$ z_e = \ell_e^2 \f$,
    /// one entry per edge in edge order — the derivative of `tau()` (the
    /// ratio over the marking in force) with respect to each squared edge
    /// length, holomorphic in \f$ z \f$ on and off the real locus: every root
    /// sits on its continuation branch, so the derivative is the derivative of
    /// that branch.
    ///
    /// In the period frame \f$ F = H\Pi^{-1} \f$ the holomorphic form has
    /// coefficients \f$ (1, \tau) \f$ and is an eigenvector of
    /// \f$ J_F = G_F^{-1}R_F^T \f$, so \f$ \tau \f$ is a root of
    /// \f$ J_{12}\tau^2 + (J_{11} - J_{22})\tau - J_{21} = 0 \f$, using the
    /// chart \f$ \sigma = 1/\tau \f$ when \f$ |\tau| > 1 \f$, and
    /// \f$ d\tau \f$ follows from \f$ dJ_F \f$ without differentiating an
    /// eigendecomposition or the particular null-space basis returned.
    /// \f$ dJ_F \f$ comes from the pairings in the period frame and its dual —
    /// the dual kernel normalized by the periods transported under
    /// \f$ U^{-1} \f$ — with the per-face Heron areas, layouts, barycentric
    /// gradients and cotangent weights in closed form, and the frames through
    /// the harmonic condition: a period-normalized frame moves by an exact
    /// twisted 1-cochain \f$ dF = d_0^U\varphi \f$ with
    /// \f$ \partial_1^U M_1 d_0^U\varphi = -\partial_1^U\,dM_1\,F \f$, one
    /// weighted graph-Laplacian solve per edge and column. The derivative is
    /// analytic rather than a finite difference. It is invariant under a pure
    /// gauge and homogeneous of degree zero,
    /// \f$ \sum_e z_e\,\partial\tau/\partial z_e = 0 \f$, since
    /// \f$ \tau \f$ is scale-free.
    /// @throws std::runtime_error when the two eigenlines of \f$ J \f$
    ///   coincide (the quadratic's roots are equal: no derivative), or when
    ///   the twisted Laplacian of a frame cannot be solved beyond its gauge
    ///   kernel.
    [[nodiscard]] Eigen::VectorXcd tauDerivative() const;

    /// The intersection number \f$ A \cdot B \f$ of the marked cycles on the
    /// surface as oriented by the faces: \f$ \langle f_A \cup f_B, [K]\rangle
    /// \f$, the ordered simplicial cup product of the period frame of the
    /// unit-length, untwisted reference (a topological invariant, so the
    /// reference suffices) evaluated on the fundamental cycle, which is
    /// \f$ \pm 1 \f$ to rounding. The input requires \f$ A \cdot B = +1 \f$:
    /// the marking fixes the orientation, and a `Spacetime`, which stores none,
    /// is read in the orientation that gives \f$ +1 \f$. `flatTorus` with its
    /// counterclockwise faces returns \f$ +1 \f$ and the `reversed` read
    /// \f$ -1 \f$. It is exposed because a torus that lives as the boundary of
    /// a cobordism has no faces of its own to orient it, and \f$ \tau \f$ of
    /// the other orientation is the other hemisphere.
    [[nodiscard]] double intersectionNumber() const;

    // ----- optional intrinsic Delaunay preprocessing pass ---------------------

    /// Flip every edge violating the intrinsic Delaunay condition
    /// \f$ \alpha_e + \beta_e \le \pi \f$ until none remains (the two triangles
    /// are laid out in one plane, the diagonal is replaced by the other one
    /// with its intrinsic length, the marked cycles are rerouted around the
    /// quadrilateral), and return the qubit of the flipped triangulation. The
    /// intrinsic geometry is unchanged, so on a flat torus \f$ \tau \f$ is
    /// unchanged; the cotangent weights become non-negative and the
    /// construction numerically stable. An edge whose flip would duplicate an
    /// existing edge is left alone and reported in `warnings()`.
    /// @throws std::invalid_argument off the real locus (the condition is an
    ///   inequality on real angles).
    [[nodiscard]] SimplicialQubit intrinsicDelaunay() const;
    /// Number of flips the pass that produced this qubit performed (0 unless
    /// it came from `intrinsicDelaunay()`).
    [[nodiscard]] int delaunayFlipCount() const noexcept { return flips_; }

    // ----- inputs -------------------------------------------------------------

    [[nodiscard]] const std::vector<std::uint64_t> &vertices() const noexcept { return vertices_; }
    [[nodiscard]] const std::vector<EdgePair> &edges() const noexcept { return edges_; }
    [[nodiscard]] const std::vector<Face> &faces() const noexcept { return faces_; }
    /// The edge lengths in edge order (real on the real locus).
    [[nodiscard]] const std::vector<Complex> &lengths() const noexcept { return lengths_; }
    /// The links \f$ U_{ij} \f$ of the connection in edge order (all 1 on the
    /// trivial connection).
    [[nodiscard]] const std::vector<Complex> &links() const noexcept { return links_; }
    /// The connection over the torus's `cobordism::ChainComplex` (canonical
    /// edge order, `canonicalEdgeIndex()`).
    [[nodiscard]] const chainhodge::Connection &connection() const noexcept { return *connection_; }
    /// The canonical (`ChainComplex`, lexicographic) index of edge \p e of `edges()`.
    [[nodiscard]] std::size_t canonicalEdgeIndex(std::size_t e) const { return canonicalOf_.at(e); }
    [[nodiscard]] const Cycle &cycleA() const noexcept { return cycleA_; }
    [[nodiscard]] const Cycle &cycleB() const noexcept { return cycleB_; }
    [[nodiscard]] double degeneracyThreshold() const noexcept { return degeneracyThreshold_; }
    /// The `Spacetime` holding the vertices, edges, lengths and phases.
    [[nodiscard]] const std::shared_ptr<Spacetime> &spacetime() const noexcept { return spacetime_; }
    /// True when every length is real and every link is exactly 1: the
    /// construction ran over real numbers, bit-identical to the real-length
    /// construction.
    [[nodiscard]] bool onRealLocus() const noexcept { return real_; }
    /// True when every link is exactly 1 (no phases).
    [[nodiscard]] bool trivialConnection() const noexcept { return trivialConnection_; }

    // ----- intermediate quantities --------------------------------------------

    /// \f$ d_0 \f$ (\f$ n_E \times n_V \f$) and \f$ d_1 \f$ (\f$ n_F \times n_E \f$),
    /// untwisted (integer entries).
    [[nodiscard]] const Eigen::MatrixXd &d0() const noexcept { return d0_; }
    [[nodiscard]] const Eigen::MatrixXd &d1() const noexcept { return d1_; }
    /// Per face \f$ (\alpha_i, \alpha_j, \alpha_k) \f$, \f$ n_F \times 3 \f$;
    /// the principal \f$ \arccos \f$ off the real locus.
    [[nodiscard]] const Eigen::MatrixXcd &angles() const noexcept { return angles_; }
    /// Per face the Heron area \f$ A_t \f$; the continuation branch off
    /// the real locus.
    [[nodiscard]] const Eigen::VectorXcd &areas() const noexcept { return areas_; }
    /// Per face the local layout \f$ (p_i, p_j, p_k) \f$, \f$ n_F \times 6 \f$.
    [[nodiscard]] const Eigen::MatrixXcd &layout() const noexcept { return layout_; }
    /// Per face \f$ (\nabla\lambda_i, \nabla\lambda_j, \nabla\lambda_k) \f$ in
    /// the local frame, \f$ n_F \times 6 \f$.
    [[nodiscard]] const Eigen::MatrixXcd &barycentricGradients() const noexcept { return gradients_; }
    /// The cotangent weights \f$ w_e \f$.
    [[nodiscard]] const Eigen::VectorXcd &weights() const noexcept { return weights_; }
    /// Edges with \f$ w_e < 0 \f$ and edges with \f$ \alpha_e + \beta_e > \pi \f$;
    /// evaluated on the real locus only.
    [[nodiscard]] const std::vector<std::size_t> &negativeWeightEdges() const noexcept {
      return negativeWeightEdges_;
    }
    [[nodiscard]] const std::vector<std::size_t> &nonDelaunayEdges() const noexcept {
      return nonDelaunayEdges_;
    }
    /// \f$ G_{ab} = \langle h_a, h_b\rangle \f$ and the rotation pairing \f$ R_{ab} \f$,
    /// the transpose pairing between the dual kernel and the kernel under a
    /// nontrivial connection.
    [[nodiscard]] const Eigen::MatrixXcd &gram() const noexcept { return G_; }
    [[nodiscard]] const Eigen::MatrixXcd &rotationPairing() const noexcept { return R_; }
    /// True when \f$ |P_A| \f$ vanished and the marking \f$ (B, -A) \f$ is in
    /// force, i.e. `tau()` is \f$ -1/\tau \f$ of the given marking.
    [[nodiscard]] bool markingSwapped() const noexcept { return swapped_; }
    /// The marked cycles as closed walks, in the order they are walked;
    /// under a nontrivial connection both start at `baseVertex()`. Empty when
    /// a cycle's steps do not form one closed walk (only refused when the
    /// connection is nontrivial).
    [[nodiscard]] const Walk &walkA() const noexcept { return walkA_; }
    [[nodiscard]] const Walk &walkB() const noexcept { return walkB_; }
    /// The common base point of the transported periods: the first vertex of
    /// \f$ A \f$'s walk that lies on \f$ B \f$'s.
    /// @throws std::logic_error when the cycles share no vertex (only refused
    ///   on construction when the connection is nontrivial).
    [[nodiscard]] std::uint64_t baseVertex() const;
    /// \f$ \mathrm{cond}(M_1) = \max|w_e| / \min|w_e| \f$ and the condition
    /// number of the \f$ 2 \times 2 \f$ Gram matrix.
    [[nodiscard]] double conditionM1() const noexcept { return condM1_; }
    [[nodiscard]] double conditionG() const noexcept { return condG_; }
    /// True when either condition number exceeds the threshold.
    [[nodiscard]] bool nearDegenerate() const noexcept { return nearDegenerate_; }
    /// Every warning the construction raised: weight flags, near-degeneracy,
    /// and branch notes. Empty when there is nothing to report.
    [[nodiscard]] const std::vector<std::string> &warnings() const noexcept { return warnings_; }

  private:
    // Inputs.
    std::vector<std::uint64_t> vertices_;
    std::vector<EdgePair> edges_;
    std::vector<Face> faces_;
    std::vector<Complex> lengths_;
    std::vector<Complex> links_;
    Cycle cycleA_;
    Cycle cycleB_;
    double degeneracyThreshold_;
    std::shared_ptr<Spacetime> spacetime_;
    std::shared_ptr<const chainhodge::Connection> connection_;
    std::vector<std::size_t> canonicalOf_;   // edge index -> canonical (ChainComplex) edge index
    bool real_{true};
    bool trivialConnection_{true};
    int flips_{0};

    // incidence matrices
    Eigen::MatrixXd d0_;
    Eigen::MatrixXd d1_;
    // per-face geometry and Whitney gradients
    Eigen::MatrixXcd angles_;
    Eigen::VectorXcd areas_;
    Eigen::MatrixXcd layout_;
    Eigen::MatrixXcd gradients_;
    /// The real-locus gradients in the real-length construction's own type
    /// (bit-identity under floating-point contraction); empty off it.
    Eigen::MatrixXd realGradients_;
    // cotangent weights
    Eigen::VectorXcd weights_;
    std::vector<std::size_t> negativeWeightEdges_;
    std::vector<std::size_t> nonDelaunayEdges_;
    // harmonic space
    Eigen::MatrixXcd H_;
    Eigen::MatrixXcd Hdual_;
    // Gram, rotation pairing and complex structure
    Eigen::MatrixXcd G_;
    Eigen::MatrixXcd R_;
    Eigen::MatrixXcd J_;
    /// The real-locus pairing accumulators, in the real-length construction's
    /// own form (bit-identity under floating-point contraction); empty off it.
    Eigen::MatrixXd realG_;
    Eigen::MatrixXd realR_;
    Eigen::MatrixXd realJ_;
    double jResidual_{0.0};
    // holomorphic line, periods and period frame
    Eigen::VectorXcd omega_;
    std::complex<double> periodA_{0.0, 0.0};
    std::complex<double> periodB_{0.0, 0.0};
    std::complex<double> tau_{0.0, 0.0};
    bool swapped_{false};
    Walk walkA_;
    Walk walkB_;
    std::optional<std::uint64_t> baseVertex_;
    std::string walkObstruction_;
    /// The period frame \f$ F = H\,\Pi^{-1} \f$ over the marking in force (`periodFrame`).
    Eigen::MatrixXcd F_;
    // degeneration diagnostics
    double condM1_{0.0};
    double condG_{0.0};
    bool nearDegenerate_{false};
    std::vector<std::string> warnings_;

    // Edge lookup by (min, max) vertex pair, and per edge its two
    // (face, local edge slot) incidences: slot 0 is the face's edge (i, j),
    // slot 1 is (j, k), slot 2 is (k, i).
    std::map<EdgePair, std::size_t> edgeIndex_;
    std::vector<std::vector<std::pair<std::size_t, int>>> edgeFaces_;

    /// The per-face geometry, weights, harmonic space and pairings of the
    /// complex path at one point of the segment from the real reference.
    struct ComplexStage;

    // The shared body of both constructors: validation through diagnostics.
    void initialize();
    void indexEdges();                          // E well-formed
    void validateCombinatorics();               // incidence, chi, orientation, triangles
    void buildConnection(std::vector<Complex> canonicalLinks);  // links, the real locus
    void validatePureGauge() const;             // flux and holonomy refused by name
    void buildIncidence();                      // d_0 and d_1
    void validateCycles() const;                // closed, independent
    void buildWalks();                          // cycles as closed walks, the base point
    void buildFaceGeometry();                   // angles, areas, layouts, gradients
    void buildWeights();                        // cotangent weights
    void buildHarmonicSpace();                  // the null space H
    void buildComplexStructure();               // Whitney pairings, G, R, J
    void buildHolomorphicLine();                // the eigenline, periods, tau
    void buildPeriodFrame();                    // F = H Pi^{-1}
    void diagnoseDegeneration();                // condition numbers
    [[nodiscard]] std::size_t edgeIndexOf(std::uint64_t u, std::uint64_t v) const;
    /// The period of \p omega over a marked cycle: the plain signed sum on the
    /// trivial connection, the transported period over the cycle's walk from
    /// the base point otherwise.
    [[nodiscard]] std::complex<double> periodOf(const Eigen::VectorXcd &omega, const Cycle &cycle,
                                                const Walk &walk) const;
    /// The cycle's steps as one closed walk (Hierholzer), starting at the
    /// first step's source; empty with \p obstruction set when the steps do
    /// not form one closed walk.
    [[nodiscard]] Walk walkOf(const Cycle &cycle, const char *name, std::string &obstruction) const;
    /// \f$ W_t(\omega) \f$ at the barycenter of face \p t, real or complex,
    /// on the untwisted (real-locus) gradients.
    template <typename Scalar>
    [[nodiscard]] Eigen::Matrix<Scalar, 2, 1> whitneyAtBarycenter(
        std::size_t t, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &omega) const;
    [[nodiscard]] ComplexStage complexStageAt(const std::vector<Complex> &lengths) const;
    [[nodiscard]] Eigen::Vector2cd twistedWhitneyAtBarycenter(std::size_t t, const Eigen::MatrixXcd &gradients,
                                                              const Eigen::VectorXcd &omega, bool dual) const;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SIMPLICIALQUBIT_H
