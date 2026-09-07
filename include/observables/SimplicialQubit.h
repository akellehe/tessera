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
/// of the metric Hodge Laplacian on a triangulated torus
/// (`docs/design/simplicial_qubit_spec.md`, followed section by section). The
/// input is intrinsic geometry — a simplicial complex \f$ K \cong T^2 \f$ with
/// edge lengths (real, or complex per §16) and a marked cycle pair
/// \f$ (A, B) \f$, \f$ A \cdot B = +1 \f$ — and, read from a `Spacetime`, the
/// pure-gauge link phases of its edges; the output is a point of
/// \f$ \mathbb{CP}^1 \f$.
///
/// What is computed, and nothing is assumed (spec §1):
///  - §2 the input is validated on load: every edge in exactly two faces,
///    \f$ n_V - n_E + n_F = 0 \f$, consistent face orientations, the strict
///    triangle inequality on every face (on the real locus), closed and
///    homologically independent marked cycles;
///  - §3 the incidence matrices \f$ d_0 \f$ (\f$ n_E \times n_V \f$) and
///    \f$ d_1 \f$ (\f$ n_F \times n_E \f$) with \f$ d_1 d_0 = 0 \f$ exactly;
///  - §4 per triangle: the angles by the law of cosines, the area by Heron's
///    formula, and the intrinsic planar layout \f$ p_i = (0,0),\ p_j = (c,0),\
///    p_k = (b\cos\alpha_i, b\sin\alpha_i) \f$ that every per-face vector lives
///    in (frames of different faces are never compared);
///  - §5 the cotangent weights \f$ w_e = \tfrac12(\cot\alpha_e + \cot\beta_e) \f$,
///    \f$ M_1 = \mathrm{diag}(w_e) \f$, with negative weights and intrinsic
///    Delaunay violations \f$ \alpha_e + \beta_e > \pi \f$ flagged, and the
///    optional intrinsic Delaunay edge-flip pass `intrinsicDelaunay()`;
///  - §6 the harmonic space \f$ H = \ker[d_1;\ d_0^T M_1] \f$ (closed and
///    co-closed 1-cochains), of dimension 2 — the topological input — by a
///    dense SVD null space;
///  - §7 the \f$ L^2 \f$ inner product of 1-cochains through Whitney 1-forms:
///    barycentric gradients per face in the local frame, the interpolant
///    \f$ W_t(\omega) \f$ at the barycenter, \f$ \langle\omega,\eta\rangle =
///    \sum_t A_t\, W_t(\omega)\cdot W_t(\eta) \f$;
///  - §8 the complex structure by rotate-then-project: the Gram matrix
///    \f$ G \f$, the rotation pairing \f$ R_{ab} = \sum_t A_t\,
///    \mathrm{rot}_{90}(W_t(h_a))\cdot W_t(h_b) \f$ and \f$ J = G^{-1}R^T \f$
///    — the metric input — with the residual \f$ \|J^2 + I\|_F \f$ exposed and
///    never symmetrized away;
///  - §9 the holomorphic line: the eigenvector of \f$ J \f$ for the eigenvalue
///    nearest \f$ -i \f$ (convention \f$ \star dz = -i\,dz \f$), its periods
///    \f$ P_A, P_B \f$ over the marking, \f$ \tau = P_B/P_A \f$ in the upper
///    half plane (conjugate branch when \f$ \mathrm{Im}\,\tau < 0 \f$; the
///    marking \f$ (B, -A) \f$ and \f$ -1/\tau \f$ when \f$ |P_A| \f$ vanishes);
///  - §10 \f$ |\psi\rangle = (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$,
///    its Bloch vector (unit, asserted) and density matrix;
///  - §11 the Fubini–Study and Weil–Petersson distances, as two separate
///    functions;
///  - §13 degeneration diagnostics: \f$ \mathrm{cond}(M_1) \f$ and
///    \f$ \mathrm{cond}(G) \f$ against a configurable threshold, warning
///    rather than failing;
///  - §16 complex geometry: see below.
///
/// WHY the state is read from the geometry this way: the dimension of
/// \f$ H \f$ is topological (\f$ b_1 = 2 \f$) and the line inside it is
/// metric, so the qubit is exactly the metric's contribution — the conformal
/// structure of the torus, one point of the upper half plane once a marking
/// is fixed. Both halves are computed from the lengths and the marking and
/// nothing else, which is what lets a geometric process reach a state by
/// moving lengths. The construction is exact on flat tori and first-order
/// accurate in the mesh size otherwise (spec §15), which is why the flat torus
/// \f$ \mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z}) \f$ of `flatTorus` is the
/// reference: it returns \f$ \tau \f$ to rounding at every resolution.
///
/// WHY the marking is stored with the complex: \f$ \tau \f$ depends on
/// \f$ (A, B) \f$ up to \f$ SL(2,\mathbb{Z}) \f$, so a canonical state needs the
/// marking fixed and carried along (spec §15). Coverage is one open hemisphere
/// of the Bloch sphere; the other requires the opposite surface orientation,
/// which is the faces given with the opposite cyclic order (or the `reversed`
/// flag of the `Spacetime` constructor, whose container stores no
/// orientation). There is no action of \f$ SU(2) \f$ on edge lengths: gates
/// act on `state()` as \f$ 2\times 2 \f$ matrices and are never pulled back.
///
/// WHY a `Spacetime` sits underneath: it is the repository's container for
/// vertices, edges, lengths and link phases, so a qubit built from the raw
/// data structures of spec §2 also exists as a `Spacetime` (`spacetime()`),
/// and a torus that already lives as a `Spacetime` can be read directly. The
/// container sorts the vertices of each face, so the consistently oriented
/// faces of §2 are held here, not there.
///
/// ## §16: complex geometry
///
/// THE REAL LOCUS is the set of inputs with every length real
/// (\f$ \mathrm{Im}\,\ell_e = 0 \f$) and every link exactly 1 (`onRealLocus()`).
/// On it the construction above runs over real numbers, unchanged and
/// bit-identical to the construction over real lengths. Off it every formula
/// of §4–§9 is taken over \f$ \mathbb{C} \f$, with these rules:
///
/// THE REAL REFERENCE of a torus is the same complex with every squared edge
/// length equal to 1 — the unit equilateral triangle on every face, which is
/// `chainhodge::WhitneyMass::Branch::Continuation`'s reference simplex
/// \f$ g_{\rm ref} = \tfrac12(1 + \delta_{ij}) \f$ — carrying the same marking
/// and the same links. Every root and every branch choice off the real locus
/// is continued from it along the straight segment
/// \f$ s_e(t) = (1 - t) + t\,\ell_e^2 \f$, \f$ t \in [0, 1] \f$, in the squared
/// lengths:
///  - §4 the angles are the principal branch of \f$ \arccos \f$ of the complex
///    cosine of the law of cosines; the Heron area is \f$ \sqrt{\det g_t}/2 \f$
///    on the continuation branch, `WhitneyMass::volumeOnBranch` of the face's
///    Gram matrix \f$ (g_t)_{ij} = \tfrac12(s_{0i} + s_{0j} - s_{ij}) \f$ (the
///    argument of \f$ \det g_t(t) \f$ tracked through its roots; a root ON the
///    segment leaves no continuous branch and is refused by name). The layout
///    takes \f$ b\sin\alpha_i := 2A_t/c \f$ on that same branch, so the
///    identity \f$ A_t = \tfrac12 bc\sin\alpha_i \f$ holds exactly and the
///    barycentric gradients satisfy \f$ \nabla\lambda_v\cdot(p_v - p_u) = 1 \f$;
///    the cotangents of §5 are \f$ \cot\alpha_v = (\text{adjacent}^2 +
///    \text{adjacent}^2 - \text{opposite}^2)/(4A_t) \f$ on the same branch;
///  - §6 the harmonic space is the complex null space (complex SVD) of the
///    stacked matrix, whose incidences are twisted by the links (below);
///  - §7, §8 the pairings are the transpose (bilinear) pairing
///    \f$ a\cdot b = a_1 b_1 + a_2 b_2 \f$, never a conjugate: \f$ G \f$ is
///    complex symmetric on the trivial connection, \f$ J = G^{-1}R^T \f$ is
///    complex;
///  - §9 the eigenline is chosen by CONTINUITY from the real reference: at
///    the reference the §9 rule (eigenvalue nearest \f$ -i \f$, the other
///    eigenline when \f$ \mathrm{Im}\,\tau < 0 \f$) selects the line, and the
///    holomorphic form is then tracked along the segment by the largest
///    overlap between the eigenlines of consecutive points (the step is halved
///    until the overlap exceeds 0.99; two eigenlines that cannot be told apart
///    are refused by name). \f$ \mathrm{Im}\,\tau > 0 \f$ is not a criterion
///    off the real locus: \f$ \tau \f$ may land anywhere in \f$ \mathbb{C} \f$,
///    and the state may lie in either hemisphere;
///  - §10, §11 are the same formulas of \f$ \tau \f$; \f$ |\vec r| = 1 \f$ is
///    an algebraic identity in \f$ \tau \f$ and still holds; the Weil–Petersson
///    distance keeps its upper-half-plane domain.
///
/// LINK PHASES are a pure gauge \f$ U = 1^g \f$, \f$ U_{xy} = g_x^{-1}g_y \f$,
/// read from the `Spacetime` edges by the `chainhodge::Connection::fromSpacetime`
/// convention (\f$ U_{xy} = e^{i\varphi} \f$ when the source is \f$ x < y \f$,
/// \f$ e^{-i\varphi} \f$ otherwise). A connection that is not a pure gauge —
/// flux through a face (\f$ \mathcal F_t \ne 1 \f$) or a flat connection with
/// holonomy around a cycle of the 1-skeleton — is refused by name: its twisted
/// kernel does not have dimension 2. Under a pure gauge the phases enter as
/// in `chainhodge::CovariantChainHodge`, with the base vertex
/// \f$ b(\sigma) = \min\sigma \f$ of every cell:
///  - §3/§6 the incidences of the stacked matrix are twisted,
///    \f$ (d_1^U)_{te} = (d_1)_{te}\,U_{b(t)b(e)} \f$ (each edge value carried to
///    the face's base) and \f$ (\partial_1^U M_1)_{ve} = (\partial_1)_{ve}\,
///    U_{v\,b(e)}\,w_e \f$ (each weighted edge value carried to the vertex);
///    the kernel is \f$ H^U = \rho_1 H \f$, \f$ \rho_1 = \mathrm{diag}(g_{b(e)}^{-1}) \f$,
///    a twisted section, and the dual kernel \f$ H^\vee = \rho_1^{-1}H \f$ is
///    the same construction under \f$ U^{-1} \f$ (`dualHarmonicBasis()`);
///  - §7/§8 the Whitney interpolant carries each edge value to the face's base
///    (\f$ W_t^U \f$ with \f$ U_{b(t)b(e)} \f$, \f$ W_t^{U^{-1}} \f$ with the
///    inverse) and the pairings are between the kernel and the dual kernel,
///    \f$ G_{ab} = \sum_t A_t\, W_t^{U^{-1}}(h^\vee_a)\cdot W_t^U(h_b) \f$,
///    \f$ R_{ab} = \sum_t A_t\, \mathrm{rot}_{90}(W_t^U(h_a))\cdot
///    W_t^{U^{-1}}(h^\vee_b) \f$ — the transpose pairing of Prop. 5.1(vi), the
///    only pairing invariant under a gauge — so \f$ J = G^{-1}R^T \f$ is the
///    matrix of the same operator on \f$ H^U \f$ and its eigenline is
///    \f$ \rho_1\omega \f$;
///  - §9 the periods are taken with parallel transport along the marked cycles
///    (`chainhodge::Connection::transportedPeriod`), both from ONE base point,
///    the first vertex of \f$ A \f$ that lies on \f$ B \f$ (`baseVertex()`),
///    which makes each period \f$ g_{v_0}^{-1} \f$ times the untwisted one and
///    \f$ \tau \f$, the state and the coefficient pairs in the period frame
///    exactly gauge invariant. On the trivial connection the periods are the
///    plain signed sums of §9 (the marked cycles need not chain in the given
///    order); with a nontrivial connection each marked cycle is walked as one
///    closed walk (its steps ordered by Hierholzer's algorithm, the given
///    order kept when it already chains; a cycle whose steps do not form one
///    closed walk, or two cycles without a common vertex, are refused by name).
///
/// The §5 flags and `intrinsicDelaunay()` are inequalities on real angles;
/// they are evaluated on the real locus only (the pass refuses off it).
class SimplicialQubit {
  public:
    using Complex = std::complex<double>;
    /// An edge \f$ (i, j) \f$ with \f$ i < j \f$, oriented \f$ i \to j \f$ (spec §2).
    using EdgePair = std::pair<std::uint64_t, std::uint64_t>;
    /// A face \f$ (i, j, k) \f$ in its counterclockwise order (spec §2).
    using Face = std::array<std::uint64_t, 3>;
    /// One step of a marked cycle: (edge index into `edges()`, sign \f$ \pm 1 \f$)
    /// — \f$ +1 \f$ traverses the edge along its stored orientation.
    using CycleStep = std::pair<std::size_t, int>;
    using Cycle = std::vector<CycleStep>;
    /// A marked cycle as a closed walk of directed vertex steps
    /// (`chainhodge::Connection::Walk`), in the torus's vertex indices.
    using Walk = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

    /// The §2 / §14 constructor from the raw data structures.
    /// @param vertices \f$ V = [0 .. n_V - 1] \f$.
    /// @param edges \f$ E = [(i, j)] \f$, \f$ i < j \f$.
    /// @param faces \f$ F = [(i, j, k)] \f$, consistently oriented.
    /// @param lengths \f$ \ell : E \to \mathbb{C} \f$, one per edge, in edge
    ///   order: real and positive on the real locus (§2); nonzero and finite
    ///   off it (§16, only \f$ \ell^2 \f$ enters the construction). The link
    ///   phases are zero (the trivial connection).
    /// @param cycleA, cycleB The marked cycles as (edge index, sign) lists,
    ///   closed loops with \f$ A \cdot B = +1 \f$.
    /// @param degeneracyThreshold The condition-number level of spec §13 above
    ///   which a warning is recorded.
    /// @throws std::invalid_argument when a §2 validation fails or a §16 branch
    ///   cannot be continued from the real reference; std::runtime_error when
    ///   \f$ \dim H \ne 2 \f$ (§6: not a torus, or degenerate weights).
    SimplicialQubit(std::vector<std::uint64_t> vertices, std::vector<EdgePair> edges,
                    std::vector<Face> faces, std::vector<Complex> lengths, Cycle cycleA,
                    Cycle cycleB, double degeneracyThreshold = 1e8);

    /// The same qubit read from a `Spacetime` of dimension 2: its vertices
    /// (indexed by ascending id), its edges (ascending \f$ (i, j) \f$ order,
    /// which is the edge order the cycles refer to), its triangles, its edge
    /// lengths (real positive, or complex per §16) and its edge phases as the
    /// pure-gauge link connection of §16. The container stores no face
    /// orientation, so the consistent orientation is the one making the top
    /// chain a cycle (`cobordism::ChainComplex::fundamentalClass`), reversed
    /// when \p reversed is set — the "separate boolean" of spec §15 for the
    /// other hemisphere.
    /// @throws std::invalid_argument when a length is zero, non-finite, or
    ///   real and non-positive, the surface is not closed-orientable, the
    ///   phases are not a pure gauge (flux or holonomy, named), or a §2 / §16
    ///   validation fails.
    SimplicialQubit(const std::shared_ptr<Spacetime> &spacetime, Cycle cycleA, Cycle cycleB,
                    bool reversed = false, double degeneracyThreshold = 1e8);

    // ----- spec §14 -----------------------------------------------------------

    /// \f$ H \f$: \f$ n_E \times 2 \f$, a unitary-orthonormal basis of the null
    /// space of the (twisted) stacked matrix \f$ [d_1^U;\ \partial_1^U M_1] \f$
    /// (§6); real on the real locus.
    [[nodiscard]] const Eigen::MatrixXcd &harmonicBasis() const noexcept { return H_; }
    /// \f$ H^\vee \f$: the null space of the stacked matrix twisted by
    /// \f$ U^{-1} \f$, the dual kernel the §7/§8 pairings are taken against
    /// (§16); the same matrix as `harmonicBasis()` on the trivial connection.
    [[nodiscard]] const Eigen::MatrixXcd &dualHarmonicBasis() const noexcept { return Hdual_; }
    /// \f$ J = G^{-1}R^T \f$ in the basis \f$ \{h_1, h_2\} \f$, \f$ 2 \times 2 \f$ (§8);
    /// real on the real locus.
    [[nodiscard]] const Eigen::MatrixXcd &complexStructure() const noexcept { return J_; }
    /// \f$ \|J^2 + I\|_F \f$: the discretization-error diagnostic (§8).
    [[nodiscard]] double jResidual() const noexcept { return jResidual_; }
    /// \f$ \omega = c_0 h_1 + c_1 h_2 \f$, the complex 1-cochain spanning the
    /// holomorphic line (§9), after the branch and marking rules; a twisted
    /// section (values in the frame at each edge's base vertex) under a
    /// nontrivial connection.
    [[nodiscard]] const Eigen::VectorXcd &holomorphicForm() const noexcept { return omega_; }
    /// \f$ (P_A, P_B) \f$ of the holomorphic form over the marking in force (§9),
    /// taken with parallel transport from `baseVertex()` under a nontrivial
    /// connection (§16).
    [[nodiscard]] std::pair<std::complex<double>, std::complex<double>> periods() const noexcept {
      return {periodA_, periodB_};
    }
    /// \f$ \tau = P_B / P_A \f$ (§9).
    [[nodiscard]] std::complex<double> tau() const noexcept { return tau_; }
    /// The PERIOD FRAME of the torus (qubit cobordism spec D3): the basis
    /// \f$ (f_A, f_B) \f$ of its harmonic space with periods \f$ (1, 0) \f$
    /// and \f$ (0, 1) \f$ over the marking in force, an \f$ n_E \times 2 \f$
    /// matrix in the torus's edge order — `harmonicBasis()` times the
    /// inverse of the period matrix, \f$ F = H\,\Pi^{-1} \f$ with
    /// \f$ \Pi_{ca} = \oint_c h_a \f$ (rows = the cycles \f$ A, B \f$,
    /// columns = the basis elements \f$ h_1, h_2 \f$), so that
    /// \f$ \oint_A f_A = 1,\ \oint_B f_A = 0,\ \oint_A f_B = 0,\ \oint_B f_B = 1 \f$.
    ///
    /// WHAT it is for: the coordinates a state is written in. Every harmonic
    /// form is \f$ \omega = P_A f_A + P_B f_B \f$ with its own periods as the
    /// coefficients, so the holomorphic line is the column combination
    /// \f$ (1, \tau) \f$ of the frame: `holomorphicForm()`
    /// \f$ = \f$ `periodFrame()` \f$ \cdot (P_A, P_B)^T = P_A\,F (1, \tau)^T \f$,
    /// the qubit \f$ |0\rangle + \tau|1\rangle \f$ of §10 read as a 1-form
    /// with \f$ f_A \leftrightarrow |0\rangle \f$ and
    /// \f$ f_B \leftrightarrow |1\rangle \f$. A two-body target \f$ \chi \f$
    /// is written in the \f$ |0\rangle, |1\rangle \f$ bases of two qubits,
    /// so the transfer of a cobordism between two tori compares with it when
    /// it is read in the period frames of the two boundary tori
    /// (`cobordism::MultiCobordism::setInputFrame`), where it is
    /// \f$ 2 \times 2 \f$. WHY over the marking in force: \f$ \tau \f$ is
    /// reported over the marking in force (§9: \f$ (B, -A) \f$ when
    /// \f$ |P_A| \f$ vanishes, `markingSwapped()`), and the frame is the
    /// basis \f$ \tau \f$ is a coordinate in, so `periods()` are always the
    /// coefficients of `holomorphicForm()` in it. The frame is real on the
    /// real locus (the harmonic basis and the periods are) and complex off it
    /// (§16), invariant under a common scale of the lengths (the harmonic
    /// space and the periods are), independent of the orthonormal basis §6
    /// happens to return, and always defined: the period map of the harmonic
    /// space over a homology basis is an isomorphism, which §2's
    /// independence check guarantees. Under a nontrivial connection the
    /// periods are the transported ones from `baseVertex()`: the frame is
    /// then \f$ g_{v_0}\,\rho_1 F \f$ of the untwisted torus, and the
    /// coefficient pair of any twisted section in it is \f$ g_{v_0}^{-1} \f$
    /// times the untwisted pair — invariant as a point of \f$ \mathbb{CP}^1 \f$.
    [[nodiscard]] const Eigen::MatrixXcd &periodFrame() const noexcept { return F_; }
    /// \f$ (|0\rangle + \tau|1\rangle)/\sqrt{1+|\tau|^2} \f$ (§10).
    [[nodiscard]] Eigen::VectorXcd state() const;
    /// \f$ (2\,\mathrm{Re}\,\tau,\ 2\,\mathrm{Im}\,\tau,\ 1 - |\tau|^2)/(1+|\tau|^2) \f$ (§10);
    /// a unit vector for every finite \f$ \tau \f$, on and off the real locus.
    [[nodiscard]] Eigen::VectorXd bloch() const;
    /// \f$ \rho = \tfrac12(I + \vec r\cdot\vec\sigma) \f$ (§10).
    [[nodiscard]] Eigen::MatrixXcd densityMatrix() const;
    /// The flat torus \f$ \mathbb{C}/(\mathbb{Z} + \tau\mathbb{Z}) \f$ (§12): the
    /// unit cell spanned by \f$ 1 \f$ and \f$ \tau \f$ as an \f$ n_x \times
    /// n_y \f$ grid, every square split by its diagonal, sides identified;
    /// edge lengths are the Euclidean lengths of the lattice displacements;
    /// \f$ A \f$ is the row loop along \f$ 1 \f$ and \f$ B \f$ the column loop
    /// along \f$ \tau \f$. Exact: the read returns \f$ \tau \f$ to rounding.
    /// @throws std::invalid_argument unless \f$ \mathrm{Im}\,\tau > 0 \f$ and
    ///   \f$ n_x, n_y \ge 3 \f$ (below 3 the grid is not a simplicial complex).
    [[nodiscard]] static SimplicialQubit flatTorus(std::complex<double> tau, int nx, int ny);
    /// \f$ d_{FS} = \arccos\big(|1 + \bar\tau_1\tau_2| / \sqrt{(1+|\tau_1|^2)(1+|\tau_2|^2)}\big) \f$:
    /// distinguishability of the two states (§11); curvature \f$ +4 \f$.
    [[nodiscard]] static double fubiniStudyDistance(const SimplicialQubit &q1,
                                                    const SimplicialQubit &q2);
    /// \f$ d_{WP} = \operatorname{arccosh}\big(1 + |\tau_1 - \tau_2|^2 /
    /// (2\,\mathrm{Im}\,\tau_1\,\mathrm{Im}\,\tau_2)\big) \f$: the moduli
    /// distance between the two shapes (§11); curvature \f$ -1 \f$. Kept
    /// separate from \f$ d_{FS} \f$ on purpose: conformally equivalent, not
    /// isometric.
    [[nodiscard]] static double weilPeterssonDistance(const SimplicialQubit &q1,
                                                      const SimplicialQubit &q2);

    // ----- the derivative of tau (qubit cobordism spec D2) ---------------------

    /// \f$ \partial\tau/\partial z_e \f$ for every edge, \f$ z_e = \ell_e^2 \f$,
    /// one entry per edge in edge order — the derivative of `tau()` (the
    /// ratio over the marking in force) with respect to each squared edge
    /// length, holomorphic in \f$ z \f$ on and off the real locus (§16: every
    /// root sits on its continuation branch, so the derivative is the
    /// derivative of that branch). WHAT is differentiated: in the period
    /// frame \f$ F = H\Pi^{-1} \f$ the holomorphic form has coefficients
    /// \f$ (1, \tau) \f$ and is an eigenvector of \f$ J_F = G_F^{-1}R_F^T \f$,
    /// so \f$ \tau \f$ is a root of \f$ J_{12}\tau^2 + (J_{11} - J_{22})\tau -
    /// J_{21} = 0 \f$ (the chart \f$ \sigma = 1/\tau \f$ when \f$ |\tau| > 1 \f$)
    /// and \f$ d\tau \f$ follows from \f$ dJ_F \f$ without differentiating an
    /// eigen-decomposition or the null-space basis §6 happens to return.
    /// \f$ dJ_F \f$ comes from the pairings of §7–§8 in the period frame and
    /// its dual (the dual kernel of §16 normalized by the periods transported
    /// under \f$ U^{-1} \f$): the per-face Heron areas, layouts and
    /// barycentric gradients of §4/§7 and the cotangent weights of §5 in
    /// closed form, and the frames through the harmonic condition — a
    /// period-normalized frame moves by an exact twisted 1-cochain
    /// \f$ dF = d_0^U\varphi \f$ with \f$ \partial_1^U M_1 d_0^U\varphi =
    /// -\partial_1^U\,dM_1\,F \f$ (one weighted graph-Laplacian solve per
    /// edge and column). WHY: the qubit cobordism spec D2 holds a torus's own
    /// state at its input by a residual in \f$ \tau \f$ and descends it
    /// analytically; a finite difference is not a derivative of this
    /// construction. Invariant under a pure gauge (\f$ \tau \f$ is) and
    /// homogeneous of degree zero (\f$ \sum_e z_e\,\partial\tau/\partial z_e
    /// = 0 \f$: \f$ \tau \f$ is scale-free), which the tests assert.
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
    /// \f$ \pm 1 \f$ to rounding. Spec §2 requires \f$ A \cdot B = +1 \f$:
    /// the marking fixes the orientation, and a `Spacetime` (which stores
    /// none) is read in the orientation that gives \f$ +1 \f$ — the flat torus
    /// of §12 with its counterclockwise faces returns \f$ +1 \f$, the
    /// `reversed` read \f$ -1 \f$. WHY it is exposed: a torus that lives as
    /// the boundary of a cobordism has no faces of its own to orient it, and
    /// \f$ \tau \f$ of the other orientation is the other hemisphere.
    [[nodiscard]] double intersectionNumber() const;

    // ----- spec §5: the optional intrinsic Delaunay preprocessing pass --------

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

    // ----- inputs (§2, §16) ---------------------------------------------------

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
    /// True when every length is real and every link is exactly 1 (§16): the
    /// construction ran over real numbers, bit-identical to the real-length
    /// construction.
    [[nodiscard]] bool onRealLocus() const noexcept { return real_; }
    /// True when every link is exactly 1 (no phases).
    [[nodiscard]] bool trivialConnection() const noexcept { return trivialConnection_; }

    // ----- intermediate quantities (§3–§9, §13, §16) -------------------------

    /// \f$ d_0 \f$ (\f$ n_E \times n_V \f$) and \f$ d_1 \f$ (\f$ n_F \times n_E \f$) (§3),
    /// untwisted (integer entries).
    [[nodiscard]] const Eigen::MatrixXd &d0() const noexcept { return d0_; }
    [[nodiscard]] const Eigen::MatrixXd &d1() const noexcept { return d1_; }
    /// Per face \f$ (\alpha_i, \alpha_j, \alpha_k) \f$, \f$ n_F \times 3 \f$ (§4);
    /// the principal \f$ \arccos \f$ off the real locus.
    [[nodiscard]] const Eigen::MatrixXcd &angles() const noexcept { return angles_; }
    /// Per face the Heron area \f$ A_t \f$ (§4); the continuation branch off
    /// the real locus.
    [[nodiscard]] const Eigen::VectorXcd &areas() const noexcept { return areas_; }
    /// Per face the local layout \f$ (p_i, p_j, p_k) \f$, \f$ n_F \times 6 \f$ (§4).
    [[nodiscard]] const Eigen::MatrixXcd &layout() const noexcept { return layout_; }
    /// Per face \f$ (\nabla\lambda_i, \nabla\lambda_j, \nabla\lambda_k) \f$ in
    /// the local frame, \f$ n_F \times 6 \f$ (§7).
    [[nodiscard]] const Eigen::MatrixXcd &barycentricGradients() const noexcept { return gradients_; }
    /// The cotangent weights \f$ w_e \f$ (§5).
    [[nodiscard]] const Eigen::VectorXcd &weights() const noexcept { return weights_; }
    /// Edges with \f$ w_e < 0 \f$ and edges with \f$ \alpha_e + \beta_e > \pi \f$ (§5);
    /// evaluated on the real locus only.
    [[nodiscard]] const std::vector<std::size_t> &negativeWeightEdges() const noexcept {
      return negativeWeightEdges_;
    }
    [[nodiscard]] const std::vector<std::size_t> &nonDelaunayEdges() const noexcept {
      return nonDelaunayEdges_;
    }
    /// \f$ G_{ab} = \langle h_a, h_b\rangle \f$ and the rotation pairing \f$ R_{ab} \f$ (§8),
    /// the transpose pairing between the dual kernel and the kernel under a
    /// nontrivial connection (§16).
    [[nodiscard]] const Eigen::MatrixXcd &gram() const noexcept { return G_; }
    [[nodiscard]] const Eigen::MatrixXcd &rotationPairing() const noexcept { return R_; }
    /// True when \f$ |P_A| \f$ vanished and the marking \f$ (B, -A) \f$ is in
    /// force, i.e. `tau()` is \f$ -1/\tau \f$ of the given marking (§9).
    [[nodiscard]] bool markingSwapped() const noexcept { return swapped_; }
    /// The marked cycles as closed walks (§16), in the order they are walked;
    /// under a nontrivial connection both start at `baseVertex()`. Empty when
    /// a cycle's steps do not form one closed walk (only refused when the
    /// connection is nontrivial).
    [[nodiscard]] const Walk &walkA() const noexcept { return walkA_; }
    [[nodiscard]] const Walk &walkB() const noexcept { return walkB_; }
    /// The common base point of the transported periods: the first vertex of
    /// \f$ A \f$'s walk that lies on \f$ B \f$'s (§16).
    /// @throws std::logic_error when the cycles share no vertex (only refused
    ///   on construction when the connection is nontrivial).
    [[nodiscard]] std::uint64_t baseVertex() const;
    /// \f$ \mathrm{cond}(M_1) = \max|w_e| / \min|w_e| \f$ and the condition
    /// number of the \f$ 2 \times 2 \f$ Gram matrix (§13).
    [[nodiscard]] double conditionM1() const noexcept { return condM1_; }
    [[nodiscard]] double conditionG() const noexcept { return condG_; }
    /// True when either condition number exceeds the threshold (§13).
    [[nodiscard]] bool nearDegenerate() const noexcept { return nearDegenerate_; }
    /// Every warning the construction raised (§5 flags, §13 near-degeneracy,
    /// §9 branch notes); empty when there is nothing to report.
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

    // §3
    Eigen::MatrixXd d0_;
    Eigen::MatrixXd d1_;
    // §4, §7
    Eigen::MatrixXcd angles_;
    Eigen::VectorXcd areas_;
    Eigen::MatrixXcd layout_;
    Eigen::MatrixXcd gradients_;
    /// The real-locus gradients in the real-length construction's own type
    /// (bit-identity under floating-point contraction); empty off it.
    Eigen::MatrixXd realGradients_;
    // §5
    Eigen::VectorXcd weights_;
    std::vector<std::size_t> negativeWeightEdges_;
    std::vector<std::size_t> nonDelaunayEdges_;
    // §6
    Eigen::MatrixXcd H_;
    Eigen::MatrixXcd Hdual_;
    // §8
    Eigen::MatrixXcd G_;
    Eigen::MatrixXcd R_;
    Eigen::MatrixXcd J_;
    /// The real-locus accumulators of §8, in the real-length construction's
    /// own form (bit-identity under floating-point contraction); empty off it.
    Eigen::MatrixXd realG_;
    Eigen::MatrixXd realR_;
    Eigen::MatrixXd realJ_;
    double jResidual_{0.0};
    // §9
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
    // §13
    double condM1_{0.0};
    double condG_{0.0};
    bool nearDegenerate_{false};
    std::vector<std::string> warnings_;

    // Edge lookup by (min, max) vertex pair, and per edge its two
    // (face, local edge slot) incidences: slot 0 is the face's edge (i, j),
    // slot 1 is (j, k), slot 2 is (k, i).
    std::map<EdgePair, std::size_t> edgeIndex_;
    std::vector<std::vector<std::pair<std::size_t, int>>> edgeFaces_;

    /// The §4–§8 quantities of the complex path at one point of the segment
    /// from the real reference (§16).
    struct ComplexStage;

    // The shared body of both constructors: §2 validation through §13.
    void initialize();
    void indexEdges();                          // §2: E well-formed
    void validateCombinatorics();               // §2: incidence, chi, orientation, triangles
    void buildConnection(std::vector<Complex> canonicalLinks);  // §16: the links, the real locus
    void validatePureGauge() const;             // §16: flux and holonomy refused by name
    void buildIncidence();                      // §3
    void validateCycles() const;                // §2: closed, independent
    void buildWalks();                          // §16: the cycles as closed walks, the base point
    void buildFaceGeometry();                   // §4, §7 (gradients)
    void buildWeights();                        // §5
    void buildHarmonicSpace();                  // §6
    void buildComplexStructure();               // §7, §8
    void buildHolomorphicLine();                // §9
    void buildPeriodFrame();                    // qubit cobordism spec D3
    void diagnoseDegeneration();                // §13
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
    /// \f$ W_t(\omega) \f$ at the barycenter of face \p t (§7), real or complex,
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
