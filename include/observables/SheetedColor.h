// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_SHEETEDCOLOR_H
#define TESSERA_OBSERVABLES_SHEETEDCOLOR_H

// Colour as sheet multiplicity: the k-sheeted support, the attachment matrix
// of connecting simplices, the GL(k, C) sheet-frame freedom, and the
// determinant-wedge singlet amplitude read after transport.
//
// ## Contents
//
//   • SheetedSupport   — a cluster support carried as k isomorphic copies of
//                        one base complex K-bar. It certifies that the copies
//                        really are isomorphic (equal stored squared lengths
//                        and equal connection values on corresponding cells),
//                        lifts a base operator h to the free sheeted operator
//                        h (x) I_k, lifts a base Riesz band E-bar to the
//                        colour-spin fibre E-bar (x) C^k, and builds the two
//                        commuting actions I (x) g (sheet relabeling) and
//                        D (x) I (a base symmetry).
//   • SheetFock        — the exterior algebra of the k sheets of ONE base
//                        mode, Lambda^bullet E with E = C^k: the occupation
//                        sectors, their dimensions and fermion parities, the
//                        exterior creation and contraction matrices, and the
//                        gl(E) bilinears E^i_j = epsilon_i iota^j. For k = 3
//                        the sectors are the 1 + 3 + 3-bar + 1 of
//                        `ColorFiber`, to which this class delegates rather
//                        than rebuilding them.
//   • SheetAttachment  — the attachment matrix S_AB of the connecting
//                        simplices between two sheeted supports, its frame
//                        law, the factorized coupling block
//                        C_AB = C-bar_AB (x) S_AB, the composition of colour
//                        transport along a declared path, and the closed
//                        holonomy with its conjugacy-invariant data.
//   • ColorSinglet     — the common-frame colour amplitude
//                        S_ABC = Omega_p(c-hat_A ^ c-hat_B ^ c-hat_C) of three
//                        quark colour vectors transported to a common base
//                        cluster p. No normalization and no modulus square is
//                        imposed: the reported content is the complex
//                        amplitude and whether it vanishes.
//
// ## The adopted sheet convention
//
// A cluster support may be k-sheeted: k isomorphic copies of a base complex
// K-bar with equal squared lengths and equal connection values on
// corresponding cells. Nothing is added to the field content — a sheet is
// more cells, and each edge still carries its complex squared length z_e, its
// connection value U_e and one two-level mode. On a three-sheeted support the
// one-particle space is C^1(K-bar) (x) C^3 and the free operator is h (x) I_3,
// so every Riesz band E-bar of the base becomes E-bar (x) C^3. The operators
// I (x) g for g in GL(3, C) commute with h (x) I_3 and with any base symmetry
// action D (x) I_3, which is what makes colour an exact, degenerate, internal
// symmetry of the free theory.
//
// ## Index convention
//
// A sheeted mode is the pair (base cell b, sheet s) and is stored at the flat
// index b * k + s. That is the Kronecker convention kron(base, sheet): the
// base factor is the slow index and the sheet factor the fast one, so
// `freeOperator(h)` is kron(h, I_k) and `sheetFrameOperator(g)` is
// kron(I_base, g) with no re-indexing anywhere.
//
// ## Frame law
//
// An independent choice of sheet basis at each end is the frame freedom
// g_A, g_B in GL(k, C). Colour data transform exactly as
//
//     S_AB  |-->  g_A^{-1} S_AB g_B,
//
// so a closed holonomy transforms by conjugation, H |--> g^{-1} H g, and its
// characteristic polynomial, traces, determinant and conjugacy class are
// frame-free observables. No polar factor is taken, no cube root of the
// determinant is chosen, and no compact real form is imposed: the retained
// datum is the GL(k, C) element together with its determinant line.
//
// Everything here is a pure function of caller-supplied data: no solver call,
// no Spacetime mutation, and nothing enters the emergence objective.

#include "cobordism/Certificate.h"
#include "observables/ColorFiber.h"

#include <Eigen/Dense>

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace tessera::observables {

/// The certificate that k supports really are k sheets of one base complex:
/// the measured disagreement between corresponding cells of different sheets.
///
/// Two quantities are measured separately because they fail for different
/// physical reasons: `squaredLengthResidual` is a disagreement of the
/// geometry (the stored complex squared lengths z_e), and
/// `connectionResidual` is a disagreement of the declared connection (the
/// values U_e). Both are absolute maxima over corresponding cells and over
/// sheet pairs; an empty support measures zero on both.
struct SheetIsomorphismRead {
    /// How many sheets were compared (k).
    std::size_t sheetCount{0};
    /// How many cells the base complex K-bar carries.
    std::size_t baseCellCount{0};
    /// max over cells b and sheet pairs (s, t) of |z_{b,s} - z_{b,t}|.
    double squaredLengthResidual{0.0};
    /// max over cells b and sheet pairs (s, t) of |U_{b,s} - U_{b,t}|.
    double connectionResidual{0.0};
    /// Whether both residuals met the declared tolerance.
    bool isomorphic{false};
    /// The record grading the comparison: AlgebraicallyExact and Static,
    /// because the residual is a difference of stored numbers and its only
    /// error source is rounding. The graded residual is the larger of the
    /// two above and the tolerance is the one the caller declared. A
    /// default-constructed read carries the never-holding HeuristicDiscovery
    /// default, so an unmeasured support is never mistaken for a certified
    /// one.
    ::tessera::cobordism::Certificate certificate{};
};

/// One occupation sector of the exterior algebra of the sheet space.
struct SheetSector {
    /// The occupation number N of the sector (0 <= N <= k).
    std::size_t occupation{0};
    /// dim Lambda^N E = binomial(k, N).
    std::size_t dimension{0};
    /// The fermion parity (-1)^N of the sector: +1 even, -1 odd.
    int fermionParity{+1};
    /// The complex representation the sector carries, named for k = 3 as the
    /// whitepaper names it ("scalar vacuum", "fundamental E",
    /// "det E (x) E-dual", "determinant line") and as "exterior power N" for
    /// every other sheet count, where those four names do not apply.
    std::string representation{};
};

/// # SheetedSupport
///
/// A cluster support carried as k isomorphic copies ("sheets") of one base
/// complex K-bar.
///
/// The class holds only the two integers that fix the index layout — the
/// sheet count k and the base cell count — because a sheet adds cells and
/// nothing else. Every method is a pure function of its arguments and of
/// those two integers.
///
/// ## What the class certifies and what it constructs
///
/// `certifyIsomorphism` grades a candidate sheeting: it is the check that the
/// k copies carry equal stored squared lengths and equal connection values on
/// corresponding cells, which is what makes the shared geometry a stationary
/// sector of the sheet-permuting group S_k.
///
/// `freeOperator`, `liftBand`, `sheetFrameOperator` and `baseSymmetryOperator`
/// construct the four objects the sheet convention names:
/// h (x) I_k, E-bar (x) C^k, I (x) g and D (x) I_k. The two commutation
/// identities [h (x) I, I (x) g] = 0 and [D (x) I, I (x) g] = 0 are exact and
/// are measured by `sheetCommutatorResidual`.
class SheetedSupport {
  public:
    /// Double-precision complex scalar of every operator and frame entry.
    using Complex = std::complex<double>;

    /// A support of `sheetCount` sheets over a base complex of
    /// `baseCellCount` cells.
    /// @throws std::invalid_argument when `sheetCount` is zero.
    SheetedSupport(std::size_t sheetCount, std::size_t baseCellCount);

    /// The sheet number k. For quarks the adopted value is three.
    [[nodiscard]] std::size_t sheetCount() const noexcept { return sheetCount_; }

    /// The number of cells of the base complex K-bar.
    [[nodiscard]] std::size_t baseCellCount() const noexcept {
        return baseCellCount_;
    }

    /// The number of sheeted cells, k times the base cell count.
    [[nodiscard]] std::size_t cellCount() const noexcept {
        return sheetCount_ * baseCellCount_;
    }

    /// The flat index b * k + s of base cell `baseCell` on sheet `sheet`.
    /// @throws std::invalid_argument when either index is out of range.
    [[nodiscard]] std::size_t modeIndex(std::size_t baseCell,
                                        std::size_t sheet) const;

    /// Certify that `sheetSquaredLengths[s]` and `sheetConnections[s]`, each
    /// a vector over the base cells, agree across the k sheets.
    ///
    /// Both outer vectors must have length k and every inner vector must have
    /// the base cell count; the connection vector may be empty for every
    /// sheet, which declares a support with no connection values to compare
    /// and reports a zero connection residual.
    /// @throws std::invalid_argument on a size mismatch or a non-positive
    ///         tolerance.
    [[nodiscard]] SheetIsomorphismRead certifyIsomorphism(
        const std::vector<Eigen::VectorXcd>& sheetSquaredLengths,
        const std::vector<Eigen::VectorXcd>& sheetConnections,
        double tolerance = 1e-12) const;

    /// The free sheeted operator h (x) I_k of a base one-particle operator
    /// `baseOperator` (which must be square with the base cell count).
    /// @throws std::invalid_argument on a shape mismatch.
    [[nodiscard]] Eigen::MatrixXcd freeOperator(
        const Eigen::MatrixXcd& baseOperator) const;

    /// The colour-spin fibre E-bar (x) C^k of a base Riesz band: the base
    /// band `baseBand` has the base cell count as its row count and its rank
    /// r as its column count, and the lift is kron(baseBand, I_k), of shape
    /// (baseCellCount * k) x (r * k). Column j * k + s is the base mode j on
    /// sheet s, which is one fermion carrying both its base index and its
    /// sheet index.
    /// @throws std::invalid_argument on a row-count mismatch.
    [[nodiscard]] Eigen::MatrixXcd liftBand(
        const Eigen::MatrixXcd& baseBand) const;

    /// The sheet relabeling I (x) g for a k x k frame `g` in GL(k, C).
    /// @throws std::invalid_argument when `g` is not k x k.
    [[nodiscard]] Eigen::MatrixXcd sheetFrameOperator(
        const Eigen::MatrixXcd& g) const;

    /// A base symmetry action D (x) I_k for a base-sized `baseAction`, the
    /// rotation action of the cluster carried to the sheeted support.
    /// @throws std::invalid_argument on a shape mismatch.
    [[nodiscard]] Eigen::MatrixXcd baseSymmetryOperator(
        const Eigen::MatrixXcd& baseAction) const;

    /// The exact commutator defect ||[A (x) I_k, I (x) g]||_max of a
    /// base-sized operator `baseOperator` against a sheet frame `g`. It is
    /// zero up to rounding for every base operator and every frame, which is
    /// the statement that colour commutes with the base dynamics and with the
    /// base symmetries.
    /// @throws std::invalid_argument on a shape mismatch.
    [[nodiscard]] double sheetCommutatorResidual(
        const Eigen::MatrixXcd& baseOperator,
        const Eigen::MatrixXcd& g) const;

  private:
    std::size_t sheetCount_{1};
    std::size_t baseCellCount_{0};
};

/// # SheetFock
///
/// The exterior algebra Lambda^bullet E of the sheet space E = C^k of ONE
/// base mode: the graded Fock space of k sheet copies of a single base mode.
///
/// Interpreting |1> as an occupied sheet copy of one base mode, the graded
/// tensor product of the k copies is Lambda^bullet E, which for k = 3 is
///
///     (C^2)^{(x)3} = C + E + (det E (x) E-dual) + det E,
///
/// the scalar vacuum, the fundamental, its determinant-twisted dual and the
/// determinant line, with fermion parities even, odd, even, odd. No ordering
/// of the sheets is geometric: a choice of sheet basis is exactly the frame
/// freedom g in GL(k, C), and the wedge sign is fixed by the exterior algebra
/// itself.
///
/// The algebra is not rebuilt here. The sector projectors and the canonical
/// anticommutation-relation matrices are delegated to
/// `quantum::ExteriorAlgebra`, and for k = 3 every sector projector is
/// bit-identical to the corresponding `ColorFiber` projector, which this
/// class states as an identity and `sectorAgreementResidual` measures.
class SheetFock {
  public:
    /// Double-precision complex scalar of every operator entry.
    using Complex = std::complex<double>;

    /// The exterior algebra of `sheetCount` sheets of one base mode.
    /// @throws std::invalid_argument when `sheetCount` is zero or exceeds
    ///         `quantum::ExteriorAlgebra::kMaxMatrixModes`.
    explicit SheetFock(std::size_t sheetCount);

    /// The sheet number k.
    [[nodiscard]] std::size_t sheetCount() const noexcept {
        return sheetCount_;
    }

    /// dim Lambda^bullet E = 2^k.
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }

    /// The k + 1 occupation sectors with their dimensions, fermion parities
    /// and representation names, in ascending occupation order. The
    /// dimensions sum to 2^k exactly.
    [[nodiscard]] std::vector<SheetSector> sectors() const;

    /// The 2^k x 2^k projector onto total occupation `occupation`; the zero
    /// matrix when `occupation` exceeds k.
    [[nodiscard]] Eigen::MatrixXcd sectorProjector(
        std::size_t occupation) const;

    /// Exterior creation epsilon_i by the sheet copy e_i as a dense
    /// 2^k x 2^k matrix.
    /// @throws std::invalid_argument when `sheet` is out of range.
    [[nodiscard]] Eigen::MatrixXcd exteriorCreation(std::size_t sheet) const;

    /// Contraction iota^j by the dual basis vector e^j, the adjoint of
    /// `exteriorCreation`.
    /// @throws std::invalid_argument when `sheet` is out of range.
    [[nodiscard]] Eigen::MatrixXcd contraction(std::size_t sheet) const;

    /// The gl(E) bilinear E^i_j = epsilon_i iota^j on the whole exterior
    /// algebra. These satisfy [E^i_j, E^k_l] = delta^k_j E^i_l -
    /// delta^i_l E^k_j exactly, the standard gl(k, C) action; their traceless
    /// combinations are sl(k, C).
    /// @throws std::invalid_argument when either index is out of range.
    [[nodiscard]] Eigen::MatrixXcd sheetBilinear(std::size_t i,
                                                 std::size_t j) const;

    /// The maximum absolute deviation over the gl(k, C) commutator table
    /// [E^i_j, E^k_l] - (delta^k_j E^i_l - delta^i_l E^k_j). Zero up to
    /// rounding.
    [[nodiscard]] double commutatorResidual() const;

    /// The maximum absolute deviation, over all four occupation sectors,
    /// between this algebra's sector projectors and `ColorFiber`'s. Defined
    /// only at k = 3, where the two describe the same Lambda^bullet C^3.
    /// @throws std::logic_error when the sheet count is not three.
    [[nodiscard]] double sectorAgreementResidual() const;

  private:
    void validateSheet(std::size_t sheet) const;

    std::size_t sheetCount_{1};
    std::size_t dimension_{2};
};

/// One connecting simplex between two sheeted supports A and B.
///
/// A connecting simplex is attached to a definite sheet of each end, so it
/// contributes its weight to one entry of the attachment matrix. The default
/// attachment rule joins sheet to sheet, which makes the attachment matrix
/// diagonal and the lone interaction colour-abelian; a cross-sheet attachment
/// produced by the rule is the nonabelian part of the colour transport.
struct ConnectingSimplex {
    /// The sheet index i of the target support A the simplex is attached to.
    std::size_t sheetA{0};
    /// The sheet index j of the source support B the simplex is attached to.
    std::size_t sheetB{0};
    /// The simplex weight contributed to entry (i, j). A plain count uses
    /// weight one; a weighted gluing supplies the weight the gluing rule
    /// assigns. Weights accumulate additively over simplices attached to the
    /// same sheet pair.
    std::complex<double> weight{1.0, 0.0};
};

/// The attachment matrix and the data that decide whether it is usable as a
/// colour transport.
struct AttachmentRead {
    /// S_AB itself, k x k, whose (i, j) entry is the accumulated weight of
    /// the connecting simplices joining sheet i of A to sheet j of B.
    Eigen::MatrixXcd matrix{};
    /// det S_AB, the canonical determinant-line datum. It is not projected,
    /// normalized or cube-rooted.
    std::complex<double> determinant{0.0, 0.0};
    /// The condition number sigma_max / sigma_min of S_AB; infinity when the
    /// matrix is singular, so a rank-dropping attachment is reported rather
    /// than silently inverted.
    double conditioning{0.0};
    /// The smallest singular value of S_AB — the distance to the
    /// rank-dropping configuration that the determinant winding encircles.
    double minSingularValue{0.0};
    /// Whether every connecting simplex joined a sheet to the sheet of the
    /// same index, so that S_AB is diagonal and this lone interaction is
    /// colour-abelian.
    bool sheetDiagonal{true};
    /// How many connecting simplices were accumulated.
    std::size_t simplexCount{0};
    /// The record grading the read: StructureExact and NonNormal, because
    /// S_AB is an exact accumulation of the declared weights given the
    /// verified premise that every connecting simplex named a sheet at each
    /// end. The graded residual is the declared full-rank tolerance minus
    /// `minSingularValue`, clamped below at zero, so a comfortably full-rank
    /// attachment grades at residual zero and a singular one never holds.
    ::tessera::cobordism::Certificate certificate{};
};

/// The conjugacy-invariant data of a closed colour holonomy.
struct HolonomyInvariants {
    /// H(theta) = M_{A0 A(n-1)} ... M_{A2 A1} M_{A1 A0}, the ordered product
    /// around the closed sequence. It transforms by conjugation under a
    /// frame change at the base point.
    Eigen::MatrixXcd holonomy{};
    /// tr H^j for j = 1, ..., k — the power traces, which together determine
    /// the characteristic polynomial and are frame-free.
    std::vector<std::complex<double>> powerTraces{};
    /// The coefficients of the characteristic polynomial
    /// det(lambda I - H), in ascending powers of lambda, obtained from the
    /// power traces by the Newton identities. The leading coefficient is one.
    std::vector<std::complex<double>> characteristicPolynomial{};
    /// det H, equal to the last characteristic coefficient up to the sign
    /// (-1)^k and computed independently as a cross-check.
    std::complex<double> determinant{0.0, 0.0};
    /// How many links were composed.
    std::size_t linkCount{0};
};

/// # SheetAttachment
///
/// The colour transport of the sheet convention: the attachment matrix S_AB
/// reconstructed from the connecting simplices between two sheeted supports,
/// its frame law, the coupling block it factorizes, its composition along a
/// declared path, and its closed holonomy.
///
/// The whitepaper's rank-r fibre map M_AB is the matched-frame chain-level
/// transfer; on sheeted supports the part of M_AB that commutes with the base
/// symmetry is I (x) S_AB, and the colour transport is the S_AB factor alone.
/// The base part is spectral transport of the geometric band and is reported
/// separately by `FiberConnection`; this class is the sheet factor and
/// nothing else. That separation is what makes the determinant a colour
/// quantity: det M_AB mixes det S_AB with the base determinant and is defined
/// only up to the scalar ambiguity of the tensor factorization, whereas
/// det S_AB is canonical.
///
/// Every member is static: the transport is a pure function of the declared
/// gluing and of caller-supplied frames.
class SheetAttachment {
  public:
    /// Double-precision complex scalar of every matrix entry.
    using Complex = std::complex<double>;

    SheetAttachment() = delete;  // static kernel — no instances.

    /// The attachment matrix of `simplices` between two k-sheeted supports.
    ///
    /// Entry (i, j) is the sum of the weights of the connecting simplices
    /// joining sheet i of A to sheet j of B. A sheet pair named by no simplex
    /// has entry zero.
    ///
    /// `fullRankTolerance` is the smallest singular value at which the matrix
    /// still counts as a full-rank element of GL(k, C). The certificate holds
    /// exactly when the measured smallest singular value meets it; the graded
    /// residual is the declared tolerance minus that singular value, clamped
    /// below at zero, so a comfortably full-rank matrix grades at residual
    /// zero and a singular one never holds.
    /// @throws std::invalid_argument when `sheetCount` is zero, a simplex
    ///         names a sheet outside the support, or the tolerance is not
    ///         positive.
    [[nodiscard]] static AttachmentRead attachmentMatrix(
        std::size_t sheetCount,
        const std::vector<ConnectingSimplex>& simplices,
        double fullRankTolerance = 1e-12);

    /// The frame law S_AB |--> g_A^{-1} S_AB g_B under independent sheet
    /// relabelings at the two ends. This is the identity the whole
    /// construction rests on, written once.
    /// @throws std::invalid_argument on a shape mismatch or a singular g_A.
    [[nodiscard]] static Eigen::MatrixXcd frameChanged(
        const Eigen::MatrixXcd& attachment, const Eigen::MatrixXcd& frameA,
        const Eigen::MatrixXcd& frameB);

    /// The factorized coupling block C_AB = C-bar_AB (x) S_AB of the glued
    /// operator, with `baseCoupling` the shared-geometry base block C-bar_AB.
    /// The Kronecker convention is kron(base, sheet), matching
    /// `SheetedSupport`'s index layout.
    /// @throws std::invalid_argument when either factor is empty.
    [[nodiscard]] static Eigen::MatrixXcd couplingBlock(
        const Eigen::MatrixXcd& baseCoupling,
        const Eigen::MatrixXcd& attachment);

    /// The composition of colour transport along a declared path, applied
    /// right to left: `compose({S_1, S_2, ..., S_n})` returns
    /// S_n ... S_2 S_1, the transport of the path that traverses S_1 first.
    /// An empty path returns the k x k identity, for which `sheetCount` must
    /// be supplied.
    /// @throws std::invalid_argument on an empty path with a zero sheet
    ///         count, a non-square factor, or a shape mismatch between
    ///         consecutive factors.
    [[nodiscard]] static Eigen::MatrixXcd compose(
        const std::vector<Eigen::MatrixXcd>& path, std::size_t sheetCount = 0);

    /// The closed holonomy of a link sequence and its conjugacy invariants.
    /// The links are supplied in traversal order, so `links.front()` is
    /// applied first, exactly as in `compose`.
    /// @throws std::invalid_argument on an empty sequence or a shape
    ///         mismatch.
    [[nodiscard]] static HolonomyInvariants holonomy(
        const std::vector<Eigen::MatrixXcd>& links);

  private:
    static void validateSquare(const Eigen::MatrixXcd& m, const char* what);
};

/// The common-frame colour amplitude of three quark colour vectors.
///
/// Every field is reported; none is normalized. The physical singlet
/// condition is a nonzero, refinement-stable, covariantly trivial determinant
/// wedge, so the state-independent content is the determinant ray and its
/// vanishing or non-vanishing, not a modulus square driven to one.
struct ColorSingletRead {
    /// S_ABC = Omega_p(c-hat_A ^ c-hat_B ^ c-hat_C), the complex amplitude.
    std::complex<double> amplitude{0.0, 0.0};
    /// The determinant det[c-hat_A c-hat_B c-hat_C] of the transported
    /// representatives, before the determinant-line trivialization Omega_p is
    /// applied. The amplitude is this times the trivialization value.
    std::complex<double> transportedWedge{0.0, 0.0};
    /// The three transported representatives c-hat_X = S_pX c_X as the
    /// columns of a k x 3 matrix, in the order A, B, C.
    Eigen::MatrixXcd transportedColumns{};
    /// |S_ABC|, reported so a caller can compare amplitudes without
    /// re-deriving them. It is a reported magnitude, never a gate on one.
    double magnitude{0.0};
    /// Whether the amplitude is nonzero at the declared tolerance — the
    /// singlet condition the whitepaper states.
    bool nonvanishing{false};
    /// The smallest singular value of the transported column matrix: how far
    /// the three colour rays are from a degenerate (coinciding) triple, which
    /// is the configuration that annihilates the wedge.
    double minSingularValue{0.0};
    /// The record grading the read: AlgebraicallyExact and NonNormal, because
    /// the amplitude is a determinant of supplied numbers and its only error
    /// source is rounding. The graded residual is the declared tolerance
    /// minus |S_ABC|, clamped below at zero, so a comfortably nonzero wedge
    /// grades at residual zero and a vanishing one never holds.
    ::tessera::cobordism::Certificate certificate{};
};

/// # ColorSinglet
///
/// The singlet readout of the sheet convention, as the whitepaper defines it.
///
/// Three quark modes A, B, C carry projective colour rays [c_A], [c_B], [c_C]
/// in their sheet spaces — in one three-sheeted cluster, where the sheet space
/// is common, or in three clusters whose sheet frames are compared through the
/// colour transport S along declared paths to a common base cluster p in the
/// bound supercluster. With representatives supplied by the many-body state,
///
///     c-hat_A = S_pA c_A,   c-hat_B = S_pB c_B,   c-hat_C = S_pC c_C,
///
/// with S_pA the identity when A and p are the same cluster. With
/// Omega_p in (det E_p)-dual the dual determinant trivialization transported
/// from the boundary reference, the common-frame colour amplitude is
///
///     S_ABC = Omega_p(c-hat_A ^ c-hat_B ^ c-hat_C).
///
/// It is invariant under all local frame changes when Omega_p is transformed
/// dually: c_X |--> g_X^{-1} c_X and S_pX |--> g_p^{-1} S_pX g_X send
/// c-hat_X |--> g_p^{-1} c-hat_X, the wedge by det(g_p)^{-1}, and
/// Omega_p |--> det(g_p) Omega_p, so the amplitude does not move.
/// `frameCovarianceResidual` measures that invariance on supplied frames
/// rather than asserting it.
///
/// Rescaling arbitrary representatives of the three rays rescales S_ABC, so
/// the state-independent content is the determinant ray and its vanishing or
/// non-vanishing; scalar amplitudes and ratios use the representatives fixed
/// by the boundary or Fock state. No normalization to one and no modulus
/// square is imposed.
class ColorSinglet {
  public:
    /// Double-precision complex scalar of every vector and matrix entry.
    using Complex = std::complex<double>;

    ColorSinglet() = delete;  // static kernel — no instances.

    /// The transported determinant wedge of three colour vectors.
    ///
    /// `trivialization` is the value Omega_p takes on the ordered basis wedge
    /// e_1 ^ ... ^ e_k of det E_p — one complex number, because det E_p is a
    /// line. Supplying one declares the determinant-line trivialization
    /// transported from the boundary reference; supplying one is required,
    /// and the natural choice on an already-trivialized line is one.
    ///
    /// `transports` are the three colour transports S_pA, S_pB, S_pC in the
    /// order A, B, C; each must be k x k, and the identity is supplied for a
    /// cluster that is the base cluster p itself.
    ///
    /// `colors` are the three representatives c_A, c_B, c_C, each of length
    /// k. The sheet count k is three on the adopted quark support, and the
    /// wedge of exactly three vectors is defined only there, so the routine
    /// refuses any other sheet count by name rather than returning a
    /// degenerate value.
    ///
    /// `tolerance` is the magnitude below which the amplitude counts as
    /// vanishing.
    /// @throws std::invalid_argument when the sheet count is not three, a
    ///         transport or colour vector is misshaped, or the tolerance is
    ///         not positive.
    [[nodiscard]] static ColorSingletRead amplitude(
        Complex trivialization,
        const std::vector<Eigen::MatrixXcd>& transports,
        const std::vector<Eigen::VectorXcd>& colors,
        double tolerance = 1e-12);

    /// The measured invariance of the amplitude under local frame changes:
    /// |S_ABC(frame-changed data) - S_ABC(original data)|.
    ///
    /// `baseFrame` is g_p at the common base cluster and `frames` are the
    /// three g_A, g_B, g_C. The frame-changed data are
    /// c_X |--> g_X^{-1} c_X, S_pX |--> g_p^{-1} S_pX g_X and
    /// Omega_p |--> det(g_p) Omega_p.
    /// @throws std::invalid_argument on the same conditions as `amplitude`,
    ///         or when a supplied frame is singular.
    [[nodiscard]] static double frameCovarianceResidual(
        Complex trivialization,
        const std::vector<Eigen::MatrixXcd>& transports,
        const std::vector<Eigen::VectorXcd>& colors,
        const Eigen::MatrixXcd& baseFrame,
        const std::vector<Eigen::MatrixXcd>& frames);
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_SHEETEDCOLOR_H
