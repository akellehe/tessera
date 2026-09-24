// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_MONOPOLESPIN_H
#define TESSERA_OBSERVABLES_MONOPOLESPIN_H

// Spin from an odd Dirac monopole: the monopole number of the connection
// through a closed cut, the projective representation D_k(g) of the cluster's
// rotation group with its cocycle, the spinor bands those data protect, and
// the sharp-spin eigen-equations on a superposition of determinants.
//
// ## Contents
//
//   • MonopoleSupport — a symmetric cluster bounded by a closed cut, carried
//                       as its vertices, its oriented edges, its
//                       outward-oriented faces and one U(1) connection value
//                       per edge. It reads the monopole number from the
//                       outward face holonomies, builds the twisted
//                       coboundary and the twisted Laplacians on vertex and
//                       edge cochains, solves for the compensating gauge
//                       transformation u_g of each rotation, assembles
//                       D_0(g) and D_1(g), measures the cocycle varpi and
//                       decides its cohomology class, and reads the
//                       symmetry-protected bands of a rotation-invariant
//                       operator.
//   • SharpSpin       — the sharpness read of the total-space spin: the right
//                       and left eigen-equations (J^2 - j(j+1) I)|Psi_R> = 0
//                       and <Psi_L|(J^2 - j(j+1) I) = 0 evaluated on a bounded
//                       superposition of determinants, together with the
//                       biorthogonal expectation and complex variance that
//                       the whitepaper calls insufficient on their own.
//
// ## Why a rotation cannot supply the sign
//
// The one-particle operator h_k(z, U) depends on the embedding only through
// squared lengths and connection values, both invariant under a rigid motion,
// so along a rigid 2 pi rotation every band is constant, the transport is the
// identity, and the rotation character is +1 exactly. What supplies
// half-integer spin instead is the connection's topological charge. Let a
// symmetric cluster be bounded by a closed cut through which the U(1) part of
// U carries total outward flux 2 pi mu, a Dirac monopole of charge mu in Z;
// for the tetrahedron the outward face holonomies are exp(2 pi i mu / 4). The
// configuration is symmetric up to gauge, so every rotation g of the cluster
// is implemented on k-cochains by
//
//     D_k(g) = rho_k(u_g) P_g,
//
// the geometric signed permutation P_g dressed by a compensating gauge
// transformation u_g. These satisfy
//
//     D_k(g) D_k(h) = varpi(g, h) D_k(gh)
//
// with a cocycle varpi whose class in H^2(G; U(1)) is nontrivial exactly when
// mu is odd. In that case the modes decompose into spinor representations of
// the double cover, the element covering the 2 pi rotation acts as -1, and
// every symmetry-protected band has even rank. This is Dirac quantization in
// discrete form, j_min = |mu| / 2.
//
// ## Conventions fixed by this file
//
// Vertex cochains are complex functions on the vertices. Edge cochains are
// complex functions on the stored oriented edges, each stored with its
// smaller vertex index first. The twisted coboundary is
//
//     (delta_0^U f)[x, y] = U_{xy} f(y) - f(x),
//
// so the vertex Laplacian (delta_0^U)^dagger delta_0^U is deg(x) on the
// diagonal and -U_{xy} off it. A face (x, y, z) has holonomy
// U_{xy} U_{yz} U_{zx}, evaluated in the declared outward order. The twisted
// face coboundary on a face stored with a < b < c is
//
//     (delta_1^U w)[a, b, c] = U_{ab} w[b, c] - w[a, c] + w[a, b],
//
// and the edge Laplacian is delta_0^U (delta_0^U)^dagger +
// (delta_1^U)^dagger delta_1^U. At zero flux on the tetrahedron that operator
// is exactly 4 I.
//
// A rotation is a permutation of the vertices that carries edges to edges and
// faces to faces. The gauge compensation solves
// (g U)_{xy} = u_g(x) U_{xy} u_g(y)^{-1} on a spanning tree rooted at the
// lowest vertex, with u_g(root) = 1, and the residual on the remaining edges
// is reported: it vanishes exactly when the pushed-forward connection is
// gauge-equivalent to the original one, which is what "symmetric up to gauge"
// means.
//
// Everything here is a pure function of caller-supplied data: no solver call,
// no Spacetime mutation, and nothing enters the emergence objective.

#include "cobordism/Certificate.h"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace tessera::observables {

/// The monopole number of the connection through a closed cut, read from the
/// outward face holonomies.
///
/// Each face contributes the principal argument of its outward holonomy, on
/// the half-open interval (-pi, pi], so a holonomy on the negative real axis
/// carries +pi. The total flux is their sum and the monopole number is the
/// nearest integer to that total over 2 pi.
///
/// Two limits of that recipe are measured rather than hidden. A face carrying
/// more than half a turn of flux is not resolved, because the principal
/// argument cannot see past pi; `branchMargin` reports how close the closest
/// face came to that limit and `onBranchCut` says whether one reached it, as
/// every face does at even monopole number on a four-face cut. And a total
/// flux that is not an integer multiple of 2 pi is not a bundle at all;
/// `integralityResidual` measures how far it sits from one.
struct MonopoleNumberRead {
    /// The outward holonomy of each declared face, in declaration order.
    std::vector<std::complex<double>> faceHolonomies{};
    /// The principal argument of each outward face holonomy, on (-pi, pi].
    std::vector<double> faceFluxes{};
    /// The total outward flux, the sum of `faceFluxes`.
    double totalFlux{0.0};
    /// The monopole number mu: the nearest integer to totalFlux / (2 pi).
    int monopoleNumber{0};
    /// |totalFlux / (2 pi) - monopoleNumber| — how far the total flux sits
    /// from an integer multiple of 2 pi.
    double integralityResidual{0.0};
    /// Whether every face holonomy has unit modulus and the total flux is an
    /// integer multiple of 2 pi at the declared tolerance. A total flux that
    /// is not such a multiple is not a bundle and breaks the symmetry.
    bool bundle{false};
    /// Whether the monopole number is odd — the condition under which the
    /// projective class is nontrivial and the modes carry spinor
    /// representations.
    bool odd{false};
    /// min over faces of (pi - |face flux|): how far the closest face flux
    /// stayed from the ends of the principal interval. Zero means a face
    /// holonomy sits exactly on the negative real axis, where the flux is
    /// assigned +pi by the stated convention.
    double branchMargin{0.0};
    /// Whether some face flux reached the ends of the principal interval, so
    /// that the reported integer rests on the stated convention rather than
    /// on a strictly interior argument. Every face of a four-face cut does so
    /// at even monopole number, where the holonomies are all -1.
    bool onBranchCut{false};
    /// max over edges of | |U_e| - 1 |: how far the connection departs from
    /// the U(1) values this read is defined on. The constructor refuses a
    /// connection outside that domain, so this is the verified premise, not a
    /// free parameter.
    double unitModulusResidual{0.0};
    /// The record grading the read: AlgebraicallyExact and Static, because
    /// the flux is a sum of arguments of stored numbers. The graded residual
    /// is the larger of `integralityResidual` and `unitModulusResidual`
    /// against the declared tolerance.
    ::tessera::cobordism::Certificate certificate{};
};

/// The compensating gauge transformation of one rotation and the residual
/// that decides whether the configuration really is symmetric up to gauge.
struct GaugeCompensationRead {
    /// u_g on the vertices, indexed by vertex. Normalized by u_g(root) = 1 on
    /// the spanning tree's root, which is the lowest vertex index; the
    /// remaining global phase freedom is exactly what makes varpi a cocycle
    /// rather than a coboundary.
    Eigen::VectorXcd gauge{};
    /// max over edges of |(g U)_{xy} - u_g(x) U_{xy} u_g(y)^{-1}| — zero
    /// exactly when the pushed-forward connection is gauge-equivalent to the
    /// original.
    double residual{0.0};
    /// Whether the residual met the declared tolerance.
    bool symmetric{false};
};

/// The cocycle of the projective representation and its cohomology class.
///
/// For every ordered pair (g, h) the product D(g) D(h) equals a scalar
/// multiple of D(gh); that scalar is varpi(g, h). The class of varpi in
/// H^2(G; U(1)) is nontrivial exactly when some pair of COMMUTING elements
/// has a commutator phase different from one, because the commutator phase
/// varpi(g, h) / varpi(h, g) of a commuting pair is invariant under every
/// rescaling of the u_g and is therefore a property of the class alone.
struct CocycleRead {
    /// How many group elements were scanned.
    std::size_t groupOrder{0};
    /// max over ordered pairs of the deviation of D(g) D(h) D(gh)^{-1} from a
    /// scalar multiple of the identity. Zero up to rounding; a nonzero value
    /// means the supplied permutations do not act projectively, so nothing
    /// below is meaningful.
    double scalarResidual{0.0};
    /// The distinct cocycle values encountered, sorted by real then
    /// imaginary part and deduplicated at the declared tolerance.
    std::vector<std::complex<double>> values{};
    /// max over COMMUTING pairs (g, h) of |varpi(g,h)/varpi(h,g) - 1|.
    double maxCommutatorDeviation{0.0};
    /// The commutator phase of the witnessing commuting pair: the value of
    /// varpi(g,h)/varpi(h,g) that achieved `maxCommutatorDeviation`. On an
    /// odd-monopole tetrahedral support this is -1, the quaternion relation
    /// of the binary tetrahedral group.
    std::complex<double> commutatorPhase{1.0, 0.0};
    /// The witnessing commuting pair, as indices into the supplied group.
    std::array<std::size_t, 2> commutatorPair{0, 0};
    /// Whether the class is nontrivial: some commuting pair anticommutes (or
    /// more generally acquires a phase) beyond the declared tolerance.
    bool nontrivial{false};
    /// The record grading the read: AlgebraicallyExact and Static, with
    /// `scalarResidual` graded against the declared tolerance. The class
    /// verdict is meaningful only when this certificate holds.
    ::tessera::cobordism::Certificate certificate{};
};

/// One symmetry-protected band of a rotation-invariant operator, with the
/// data that decide whether it is a spinor doublet.
struct SpinorBandRead {
    /// The eigenvalue the band sits at.
    double eigenvalue{0.0};
    /// The band's rank. Under a nontrivial projective class every
    /// symmetry-protected band has even rank.
    std::size_t dimension{0};
    /// max over group elements of ||D(g) P_W - P_W D(g)||_max, where P_W is
    /// the orthogonal projector onto the band: the measured invariance of the
    /// band under the projective action.
    double invarianceResidual{0.0};
    /// (1 / |G|) sum_g |tr D(g)|_W|^2. Exactly one for an irreducible
    /// projective representation with this cocycle, and larger when the band
    /// splits, so it is the irreducibility certificate.
    double irreducibilityScore{1.0};
    /// max over the band's basis of ||(I - P_coexact) v||, where P_coexact
    /// projects onto ker((delta_0^U)^dagger): zero exactly when the band lies
    /// in the coexact sector, which on an odd-monopole tetrahedron is the
    /// genuine j = 1/2 doublet, the restriction of the SU(2) fundamental.
    double coexactResidual{0.0};
    /// Whether this band is a spinor doublet: rank two, invariant,
    /// irreducible, and carried by a nontrivial projective class.
    bool spinorDoublet{false};
    /// Whether the band lies in the coexact sector at the declared tolerance.
    bool coexact{false};
};

/// What the classifier needs from an odd-monopole support in one record.
struct MonopoleSpinRead {
    /// The monopole number read of the bounding cut.
    MonopoleNumberRead monopole{};
    /// The cocycle read of the rotation group's projective action.
    CocycleRead cocycle{};
    /// Every symmetry-protected band of the rotation-averaged edge Laplacian,
    /// in ascending eigenvalue order.
    std::vector<SpinorBandRead> bands{};
    /// The index in `bands` of the j = 1/2 doublet: the rank-two, invariant,
    /// irreducible, spinorial band lying in the coexact sector. Equal to
    /// `bands.size()` when there is none.
    std::size_t doubletIndex{0};
    /// Whether such a doublet was found.
    bool halfIntegerDoublet{false};
    /// The record grading the whole read: it holds only when the monopole
    /// read and the cocycle read both hold. The graded residual is the larger
    /// of theirs.
    ::tessera::cobordism::Certificate certificate{};
};

/// # MonopoleSupport
///
/// A symmetric cluster bounded by a closed cut, carried as a simplicial
/// skeleton with one U(1) connection value per edge.
///
/// The class owns the skeleton and the connection and derives everything else
/// on demand. It never mutates a Spacetime and never runs a solver: the
/// twisted operators are assembled directly from the stored values and the
/// eigen-decompositions are dense self-adjoint ones on the cluster's own
/// cochain spaces.
class MonopoleSupport {
  public:
    /// Double-precision complex scalar of every connection value and operator
    /// entry.
    using Complex = std::complex<double>;
    /// One permutation of the vertices: `perm[x]` is the image of vertex x.
    using Permutation = std::vector<std::size_t>;

    /// A support on `vertexCount` vertices.
    ///
    /// `edges` are the oriented edges, each a pair of distinct vertex indices
    /// with the smaller first; duplicates are refused. `faces` are the
    /// two-simplices of the closed cut in their OUTWARD orientation, each a
    /// triple of distinct vertices whose three boundary pairs must all be
    /// declared edges. `connection` holds U_e on the stored orientation of
    /// each edge, in edge order; the reverse value U_{yx} is 1 / U_{xy} and is
    /// never stored separately. Every value must have unit modulus: the
    /// monopole number is a charge of the U(1) part of the connection, and the
    /// unitarity of D_k(g), the rotation average and the self-adjoint band
    /// decomposition all rest on that part alone. An unrestricted GL(1, C)
    /// connection enters through `u1Part`, which the caller applies
    /// explicitly.
    /// @throws std::invalid_argument on a size mismatch, a repeated or
    ///         self-incident edge, an undeclared boundary edge of a face, a
    ///         vertex index out of range, a zero connection value, or a value
    ///         whose modulus is not one.
    MonopoleSupport(std::size_t vertexCount,
                    std::vector<std::array<std::size_t, 2>> edges,
                    std::vector<std::array<std::size_t, 3>> faces,
                    std::vector<Complex> connection);

    /// The tetrahedron on vertices 0, 1, 2, 3 with its six edges, its four
    /// outward-oriented faces and the symmetric monopole connection whose
    /// every outward face holonomy is exp(2 pi i mu / 4).
    ///
    /// The connection is gauge-fixed on the spanning tree from vertex 0, so
    /// U_{0j} = 1 for j = 1, 2, 3, and the three remaining values carry the
    /// whole flux. This is the fixture the whitepaper's tetrahedral
    /// statements are made about.
    [[nodiscard]] static MonopoleSupport tetrahedron(int monopoleNumber);

    /// The twelve rotations of the tetrahedron as vertex permutations: the
    /// even permutations of four letters, the group T isomorphic to A_4. The
    /// identity is first.
    [[nodiscard]] static std::vector<Permutation> tetrahedralRotations();

    /// The U(1) part U_e / |U_e| of a general connection — the explicit way
    /// to bring a GL(1, C) connection into this kernel's domain. Applying it
    /// is the caller's declared choice; nothing here does it silently.
    /// @throws std::invalid_argument when any value is zero.
    [[nodiscard]] static std::vector<Complex> u1Part(
        const std::vector<Complex>& connection);

    /// The number of vertices.
    [[nodiscard]] std::size_t vertexCount() const noexcept {
        return vertexCount_;
    }
    /// The stored oriented edges.
    [[nodiscard]] const std::vector<std::array<std::size_t, 2>>& edges()
        const noexcept {
        return edges_;
    }
    /// The outward-oriented faces of the closed cut.
    [[nodiscard]] const std::vector<std::array<std::size_t, 3>>& faces()
        const noexcept {
        return faces_;
    }
    /// U_e on the stored orientation of each edge.
    [[nodiscard]] const std::vector<Complex>& connection() const noexcept {
        return connection_;
    }

    /// U_{xy} for any ordered vertex pair joined by a declared edge: the
    /// stored value when x < y and its reciprocal when x > y.
    /// @throws std::invalid_argument when no edge joins x and y.
    [[nodiscard]] Complex transport(std::size_t x, std::size_t y) const;

    /// The monopole number of the bounding cut.
    /// @throws std::invalid_argument when the tolerance is not positive.
    [[nodiscard]] MonopoleNumberRead monopoleNumber(
        double tolerance = 1e-9) const;

    /// The twisted coboundary delta_0^U as an edges x vertices matrix.
    [[nodiscard]] Eigen::MatrixXcd twistedCoboundary() const;

    /// The twisted face coboundary delta_1^U as a faces x edges matrix, each
    /// face taken in its sorted vertex order (the overall sign of a face does
    /// not reach the Laplacian below).
    [[nodiscard]] Eigen::MatrixXcd twistedFaceCoboundary() const;

    /// The twisted vertex Laplacian (delta_0^U)^dagger delta_0^U.
    [[nodiscard]] Eigen::MatrixXcd vertexLaplacian() const;

    /// The twisted edge Laplacian delta_0^U (delta_0^U)^dagger +
    /// (delta_1^U)^dagger delta_1^U.
    [[nodiscard]] Eigen::MatrixXcd edgeLaplacian() const;

    /// The orthogonal projector onto the coexact sector
    /// ker((delta_0^U)^dagger) of the edge cochains.
    /// @throws std::invalid_argument when the tolerance is not positive.
    [[nodiscard]] Eigen::MatrixXcd coexactProjector(
        double tolerance = 1e-9) const;

    /// The compensating gauge transformation u_g of a rotation.
    /// @throws std::invalid_argument when the permutation is not a
    ///         permutation of the vertices, does not carry edges to edges, or
    ///         the support is disconnected so that no spanning tree reaches
    ///         every vertex.
    [[nodiscard]] GaugeCompensationRead gaugeCompensation(
        const Permutation& rotation, double tolerance = 1e-9) const;

    /// D_0(g) on vertex cochains: (D_0(g) f)(g x) = u_g(g x)^{-1} f(x).
    /// @throws std::invalid_argument on the same conditions as
    ///         `gaugeCompensation`.
    [[nodiscard]] Eigen::MatrixXcd vertexRepresentation(
        const Permutation& rotation) const;

    /// D_1(g) on edge cochains, canonical because an edge's orientation
    /// reversal is transported along the edge itself:
    /// (D_1(g) w)[g x, g y] = u_g(g x)^{-1} w[x, y] when g x < g y, and
    /// (D_1(g) w)[g y, g x] = -U_{g y, g x} u_g(g x)^{-1} w[x, y] otherwise.
    /// D_1 intertwines delta_0^U with D_0 and carries the same cocycle.
    /// @throws std::invalid_argument on the same conditions as
    ///         `gaugeCompensation`.
    [[nodiscard]] Eigen::MatrixXcd edgeRepresentation(
        const Permutation& rotation) const;

    /// max over vertex cochains of
    /// ||delta_0^U D_0(g) - D_1(g) delta_0^U||_max: the measured intertwining
    /// of the two representations by the twisted coboundary.
    [[nodiscard]] double intertwiningResidual(
        const Permutation& rotation) const;

    /// The cocycle read of a rotation group supplied as vertex permutations.
    ///
    /// Closure under composition is checked, because varpi is defined by
    /// comparing D(g) D(h) with D(gh) and a missing product has no D(gh) to
    /// compare against. Closure is the whole condition here: a finite set of
    /// permutations closed under composition is already a group, so the
    /// identity and the inverses come with it.
    /// @throws std::invalid_argument when the supplied set is not closed, a
    ///         member is not a rotation of this support, the cochain degree
    ///         is outside {0, 1}, or the tolerance is not positive.
    [[nodiscard]] CocycleRead cocycle(
        const std::vector<Permutation>& group, int cochainDegree = 1,
        double tolerance = 1e-9) const;

    /// The rotation average (1 / |G|) sum_g D_1(g) L D_1(g)^dagger of an edge
    /// operator: the rotation-invariant operator whose bands the projective
    /// class protects. Applied to `edgeLaplacian()` on the unit-monopole
    /// tetrahedron it has the spectrum
    /// {4 - 2/sqrt(3) (twice), 4 (twice), 4 + 2/sqrt(3) (twice)}, with the
    /// j = 1/2 doublet at 4; at zero flux it is exactly 4 I.
    /// @throws std::invalid_argument on an empty group or a shape mismatch.
    [[nodiscard]] Eigen::MatrixXcd rotationAveragedEdgeOperator(
        const Eigen::MatrixXcd& edgeOperator,
        const std::vector<Permutation>& group) const;

    /// The symmetry-protected bands of a rotation-invariant edge operator.
    ///
    /// `operatorMatrix` must be Hermitian and edge-sized; its eigenvalues are
    /// grouped at `degeneracyTolerance` and each group is one band.
    /// `nontrivialClass` says whether the projective class is nontrivial, and
    /// only then can a band be a spinor doublet.
    /// @throws std::invalid_argument on a shape mismatch, a non-Hermitian
    ///         operator, an empty group, or a non-positive tolerance.
    [[nodiscard]] std::vector<SpinorBandRead> spinorBands(
        const Eigen::MatrixXcd& operatorMatrix,
        const std::vector<Permutation>& group, bool nontrivialClass,
        double degeneracyTolerance = 1e-7, double tolerance = 1e-9) const;

    /// The whole spin read of this support against a rotation group: the
    /// monopole number, the cocycle, the bands of the rotation-averaged edge
    /// Laplacian and the j = 1/2 doublet among them.
    /// @throws std::invalid_argument on the same conditions as the parts.
    [[nodiscard]] MonopoleSpinRead spinRead(
        const std::vector<Permutation>& group,
        double degeneracyTolerance = 1e-7, double tolerance = 1e-9) const;

  private:
    /// Index of the edge joining x and y, or the edge count when there is
    /// none.
    [[nodiscard]] std::size_t edgeIndex(std::size_t x, std::size_t y) const;
    void validateRotation(const Permutation& rotation) const;

    std::size_t vertexCount_{0};
    std::vector<std::array<std::size_t, 2>> edges_{};
    std::vector<std::array<std::size_t, 3>> faces_{};
    std::vector<Complex> connection_{};
};

/// The sharpness read of a total-space spin: two eigen-equations, plus the
/// moments the whitepaper calls insufficient on their own.
///
/// In a complex bilinear theory <J^2> = 3/4 is only a matrix-element
/// identity, and even a vanishing complex variance can result from isotropic
/// cancellation without a sharp eigenstate. The right and left residual
/// vectors must vanish algebraically; their coordinate norms are reported
/// only as numerical certificates.
struct SharpSpinRead {
    /// The eigenvalue the state is tested against: j(j+1), which is 3/4 for
    /// the j = 1/2 readout of the proton certificate.
    double targetEigenvalue{0.75};
    /// ||(J^2 - target I)|Psi_R>|| — the right residual norm.
    double rightResidual{std::numeric_limits<double>::quiet_NaN()};
    /// ||<Psi_L|(J^2 - target I)|| — the left residual norm.
    double leftResidual{std::numeric_limits<double>::quiet_NaN()};
    /// Whether BOTH residuals met the declared tolerance, relative to the
    /// norms of the two states. This is the sharp-spin certificate.
    bool sharp{false};
    /// The biorthogonal expectation <Psi_L|J^2|Psi_R> / <Psi_L|Psi_R>, formed
    /// with the bilinear left-right pairing and never with a conjugate
    /// transpose. NaN when the pairing vanishes.
    std::complex<double> expectation{std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::quiet_NaN()};
    /// The complex variance <(J^2)^2> - <J^2>^2 in the same pairing. It is
    /// reported, never gated on: it can vanish on a state that is not an
    /// eigenstate, which is exactly the case the two eigen-equations decide.
    std::complex<double> variance{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
    /// Whether the variance test alone would have called this state sharp:
    /// the expectation sits at the target and the variance vanishes, both at
    /// the declared tolerance. When this is true and `sharp` is false, the
    /// variance test has been decided wrongly by isotropic cancellation.
    bool varianceWouldAccept{false};
    /// How many occupation-basis determinants the right state occupies: the
    /// number of nonzero amplitudes in its expansion. One means a single
    /// Slater determinant, which a covariance carries; more than one means a
    /// genuine superposition, which no quasi-free covariance represents, so
    /// the covariance-based variance test has nothing to evaluate there.
    std::size_t determinantCount{0};
    /// The record grading the read: AlgebraicallyExact and NonNormal, because
    /// the residuals are exact matrix-vector products of supplied data. The
    /// graded residual is the larger of the two relative residual norms
    /// against the declared tolerance.
    ::tessera::cobordism::Certificate certificate{};
};

/// # SharpSpin
///
/// The sharp total-space spin readout of the proton certificate.
///
/// The two eigen-equations
///
///     (J^2 - 3/4 I)|Psi_R> = 0,      <Psi_L|(J^2 - 3/4 I) = 0
///
/// are essential and are evaluated on the bounded superposition of
/// determinants selected by the isolating interaction, because by the theorem
/// of Section 7 no covariance-only state satisfies them together with a
/// nonzero colour wedge. J^2 is polynomial in the exterior generators, so its
/// action is applied mode-pair by mode-pair rather than by materializing the
/// full Fock matrix; `totalSpinSquaredMatrix` materializes it anyway for
/// fixtures and cross-checks, and refuses mode counts at which that is not
/// affordable.
class SharpSpin {
  public:
    /// Double-precision complex scalar of every state and operator entry.
    using Complex = std::complex<double>;

    SharpSpin() = delete;  // static kernel — no instances.

    /// Largest mode count for which `totalSpinSquaredMatrix` materializes the
    /// dense 2^M x 2^M operator. The applied form has no such limit.
    static constexpr std::size_t kMaxDenseModes = 16;

    /// Largest mode count for which a Fock state vector (2^M entries) is
    /// built by `determinant` and `determinantSuperposition`, and for which
    /// `doubletSpinMatrices` builds its one-particle matrices. It is the
    /// exterior algebra's own matrix-layer limit: a state vector is one column
    /// of the space, not the dense operator, so it is affordable well past
    /// `kMaxDenseModes`. Eighteen modes, the edge modes of a three-sheeted
    /// tetrahedron, is a 2^18-entry vector.
    static constexpr std::size_t kMaxStateModes = 24;

    /// The Slater determinant of the listed occupied modes as a Fock vector
    /// over `modeCount` modes: the modes are wedged in the listed order, so a
    /// reordering changes the vector by the permutation sign and nothing
    /// else.
    /// @throws std::invalid_argument on a repeated or out-of-range mode, or a
    ///         mode count above `kMaxStateModes`.
    [[nodiscard]] static Eigen::VectorXcd determinant(
        const std::vector<std::size_t>& occupiedModes, std::size_t modeCount);

    /// The bounded superposition sum_d a_d |D_d> of determinants: one
    /// occupation list and one complex amplitude per determinant.
    /// @throws std::invalid_argument on mismatched lengths, an empty
    ///         superposition, or any condition `determinant` refuses.
    [[nodiscard]] static Eigen::VectorXcd determinantSuperposition(
        const std::vector<std::vector<std::size_t>>& occupations,
        const std::vector<Complex>& amplitudes, std::size_t modeCount);

    /// J^2 |Psi> = sum_a dGamma(J_a) dGamma(J_a) |Psi> applied without
    /// materializing the Fock matrix: each dGamma(J_a) is applied as the sum
    /// over mode pairs of J_a(i, j) a_i^dagger a_j.
    ///
    /// `spinMatrices` are the three one-particle spin matrices J_x, J_y, J_z,
    /// each modeCount x modeCount, where modeCount is read from the state
    /// dimension.
    /// @throws std::invalid_argument when three matrices were not supplied,
    ///         a matrix is misshaped, or the state dimension is not a power
    ///         of two.
    [[nodiscard]] static Eigen::VectorXcd applyTotalSpinSquared(
        const std::array<Eigen::MatrixXcd, 3>& spinMatrices,
        const Eigen::VectorXcd& state);

    /// The dense total-space J^2 on the 2^M-dimensional Fock space, for
    /// fixtures and cross-checks.
    /// @throws std::invalid_argument on a misshaped matrix or a mode count
    ///         above `kMaxDenseModes`.
    [[nodiscard]] static Eigen::MatrixXcd totalSpinSquaredMatrix(
        const std::array<Eigen::MatrixXcd, 3>& spinMatrices);

    /// The sharpness read of a right and a left state against the target
    /// eigenvalue.
    ///
    /// `leftState` is the left vector of the biorthogonal pair, paired with
    /// the right one bilinearly: the expectation is
    /// leftState^T J^2 rightState / (leftState^T rightState), with no
    /// conjugate transpose anywhere, because the theory is complex bilinear.
    /// Supplying the same vector twice reduces the read to the ordinary
    /// bilinear one.
    ///
    /// `tolerance` grades the two residual norms relative to the norms of the
    /// states, so the verdict does not move when a state is rescaled.
    /// @throws std::invalid_argument on a dimension mismatch, a zero state,
    ///         or a non-positive tolerance.
    [[nodiscard]] static SharpSpinRead read(
        const std::array<Eigen::MatrixXcd, 3>& spinMatrices,
        const Eigen::VectorXcd& rightState, const Eigen::VectorXcd& leftState,
        double targetEigenvalue = 0.75, double tolerance = 1e-9);

    /// The one-particle spin matrices of `carrierCount` distinguishable
    /// spin-one-half carriers: J_a = I_carrierCount (x) sigma_a / 2 in the
    /// mode order (carrier, spin), so mode 2 c + s is spin state s of
    /// carrier c. On the tetrahedron the three carriers are the doublets
    /// 2, 2' and 2'' distinguished by the Z_3 = 2T / Q_8 character.
    /// @throws std::invalid_argument when `carrierCount` is zero or the
    ///         resulting mode count exceeds `kMaxStateModes`.
    [[nodiscard]] static std::array<Eigen::MatrixXcd, 3> doubletSpinMatrices(
        std::size_t carrierCount);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_MONOPOLESPIN_H
