// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_ISOSPINDOUBLET_H
#define TESSERA_OBSERVABLES_ISOSPINDOUBLET_H

// The observation of flavour as the whitepaper (v16, Section 10) prescribes it:
//
//   "Flavor and electric charge are not assumed as hidden labels. The
//    conservative hypothesis is that two stable subclasses of the same cluster
//    fiber provide an isospin doublet. ... it succeeds only if an unlabeled
//    two-dimensional spectral band emerges, is transported coherently, and its
//    flavor-derived current agrees with the microscopic Ward-current flux at
//    those values."
//
// Nothing here declares a flavour. The detector reads the Riesz bands of the
// operator a caller supplies (the covariant h_k(z, U) of the cluster, or the
// rotation-averaged operator the spin read uses), removes from every band the
// two multiplicities the whitepaper has already assigned, and reports what is
// left.
//
// ## The two multiplicities that are not flavour
//
//   * Colour is sheet multiplicity (Section 8): on a k-sheeted support the free
//     operator is h (x) I_k, and every band is E-bar (x) C^k. The sheet matrix
//     units I (x) e_st act on such a band.
//   * Spin is the irreducible content under the support's symmetry group
//     (Sections 3 and 11.1): the three doublets 2, 2', 2'' of the unit-monopole
//     tetrahedron are spinor irreducibles of the binary tetrahedral group and
//     are spin content, not colour and not flavour.
//
// ## What a flavour band is, algebraically
//
// Let A be the algebra generated on a band E by the symmetry action D(g) and
// by the sheet matrix units, whenever the band is invariant under them. A is
// semisimple (it is the image of a finite group algebra tensored with M_k), so
//
//     E = sum_rho  V_rho (x) C^{f_rho},
//
// with V_rho the irreducible modules of A (each an irreducible of the symmetry
// group times the k sheets) and C^{f_rho} the multiplicity spaces. The
// commutant A' of A on E is sum_rho M_{f_rho}(C): exactly the operators on E
// that commute with the rotations and with colour. A band with one isotype of
// multiplicity f = 2 carries a two-dimensional space on which neither the
// rotations nor the sheets act: two copies of one spin-and-colour irreducible,
// that is "two stable subclasses of the same cluster fiber", and the
// commutant M_2 is the GL(2, C) a flavour frame change lives in. That is a
// flavour-doublet candidate. The dimensions are computed, never requested: the
// commutant is the null space of the commutator map, its centre fixes the
// isotypes, and f_rho^2 is the dimension of each isotypic block of the
// commutant.
//
// ## Falsifier 10
//
// "A robust degeneracy that is not an irreducible dimension of the support's
// symmetry group ... is neither predicted by the stated one-particle operator
// nor promoted to a flavor mechanism until it is explained." A band that is not
// one irreducible of A (more than one isotype, or a multiplicity above one) is
// reported as an unexplained multiplicity. A flavour-doublet candidate is by
// construction such a band, so it carries that finding too, and the report
// says so rather than suppressing either reading. Robustness under refinement
// is part of the falsifier and is reported as unmeasured when no refinement
// was supplied.
//
// ## The three conditions
//
//   1 emergence            A stable two-dimensional flavour band exists: a
//                          candidate as above, selected by a closed Riesz
//                          contour with a certified gap, persisting across the
//                          supplied frames (tracked by
//                          `SpectralFiberTracker::matchFibers`) and across the
//                          supplied resolutions (Section 5).
//   2 coherent-transport   Its transport between consecutive frames
//                          (`ComplexTransport::transport`) is full rank with
//                          bounded relative leakage, it intertwines the
//                          rotation and colour actions (so it factors as
//                          I (x) m with m in GL(2, C)), and the composed
//                          transport over the lifetime is full rank.
//   3 current-agreement    Its flavour-derived current agrees with the Ward
//                          flux. Reported not evaluable: the Ward current of
//                          Section 13.4 is deferred, and that section itself
//                          records that the Ward flux counts fermions and that
//                          a flavour-dependent charge carries no Ward current in
//                          the declared fields.
//
// Each condition is Passed, Failed or NotEvaluable by the rule of
// `QuarkConditions`: a measured certificate that did not hold fails the
// condition; a condition passes only when every required certificate was
// measured and held.
//
// ## Isospin and charge
//
// When conditions 1 and 2 pass, the doublet's two members are the two
// eigenspaces of a traceless element H of its commutant M_2, and I_3 is
// +1/2 on one and -1/2 on the other. H is the in-band operator when that
// splits the members; otherwise a caller-supplied splitting operator; otherwise
// a declared trivialization (the cell-order seed below). Which member is called
// +1/2 is the declared convention that the larger real part of H's eigenvalue
// is +1/2; nothing physical labels the two. The baryon number of a quark is
// B = n_Q N_Q / 3 from its certified lineage (`ClusterLineage::read`), and the
// charges are Q = I_3 + B/2, which gives +2/3 and -1/3 for N_Q = +1. The
// occupation pattern of a three-quark state is read from its one-body density
// as the occupations tr(gamma Pi_+) and tr(gamma Pi_-) of the two members.
//
// Everything here is a pure function of caller-supplied data: no solver call on
// a spacetime, no mutation, and nothing enters any emergence objective.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "observables/ClusterLineage.h"
#include "observables/ComplexTransport.h"
#include "observables/QuarkConditions.h"
#include "observables/SpectralFiber.h"

namespace tessera::observables {

/// Thresholds of the isospin-doublet detector. Each selects which reads are
/// certified, never which value is reported.
struct IsospinDoubletConfig {
  /// Relative width within which eigenvalues belong to one band: two
  /// eigenvalues are in one band when a chain of pairwise distances at most
  /// `groupingTolerance * scale` joins them, `scale` the largest eigenvalue
  /// modulus (1 for the zero operator).
  double groupingTolerance = 1e-8;
  /// A band is isolated when its distance to the nearest eigenvalue outside it
  /// is at least `minRelativeGap * scale` and the circle drawn around it
  /// separates it from every other eigenvalue.
  double minRelativeGap = 1e-6;
  /// Trapezoidal node count of each band's circular Riesz contour.
  int contourNodes = 64;
  /// Cap on the Riesz projector's relative idempotency defect.
  double projectorTolerance = 1e-9;
  /// Cap on the relative commutator ||[P, X]||_F / (||P||_F ||X||_F) below
  /// which a band is invariant under a symmetry element or a sheet matrix unit.
  double invarianceTolerance = 1e-8;
  /// Relative eigenvalue cut deciding the null space of the commutator map,
  /// and the rank of the isotypic blocks of the commutant.
  double commutantTolerance = 1e-9;
  /// Minimum subspace overlap for a certified continuation between frames and
  /// between resolutions.
  double trackOverlapThreshold = 0.5;
  /// Minimum number of frames the lifetime must span for the persistence and
  /// lifetime certificates to be measured at all.
  std::size_t minFrames = 2;
  /// Cap on the relative leakage of one frame-to-frame transport: the part of
  /// the transfer of the band that lands outside the next frame's band.
  double transportLeakageTolerance = 1e-1;
  /// Cap on the relative intertwining residual of a transport against the
  /// rotation and colour actions.
  double intertwiningTolerance = 1e-6;
  /// Cap on the condition number of a certified transport and on each band's
  /// projector norm.
  double conditionNumberCap = 1e8;
};

/// One frame of the cluster's lifetime: the operator the fiber is read on at
/// that frame, and the chain-level transfer from the previous frame's cells.
struct IsospinFrame {
  /// The name the frame is reported under.
  std::string label{};
  /// The operator on the cluster's cells at this frame (square, one row per
  /// entry of `IsospinDoubletDeclaration::cells`).
  Eigen::MatrixXcd operatorMatrix{};
  /// T from the previous frame's cells to this frame's; empty means the
  /// identity, the transfer between two frames of one cell set. Ignored on the
  /// first frame.
  Eigen::MatrixXcd transferFromPrevious{};
};

/// One resolution of the cluster: the operator at that resolution and the
/// prolongation that carries the reference resolution's cochains into it.
struct IsospinResolution {
  /// The name the resolution is reported under.
  std::string label{};
  /// The operator at this resolution.
  Eigen::MatrixXcd operatorMatrix{};
  /// Rows: this resolution's cells; columns: the reference cells.
  Eigen::MatrixXcd prolongation{};
  /// The sheet of each of this resolution's cells (empty: one sheet).
  std::vector<std::size_t> sheetOfCell{};
  /// The base cell of each of this resolution's cells (empty: the cell itself).
  std::vector<std::size_t> baseCellOfCell{};
  /// The symmetry action at this resolution, in the order of the declaration's
  /// `symmetry` (empty: none declared at this resolution).
  std::vector<Eigen::MatrixXcd> symmetry{};
};

/// What the detector is asked to read.
struct IsospinDoubletDeclaration {
  /// The operator's name, for the report ("covariant h_1(z, U)", ...).
  std::string operatorName{};
  /// Form degree of the cells.
  int degree = 1;
  /// The cells as sorted vertex-id tuples, in the operator's row order. Empty
  /// means the cells are labelled by their row index.
  std::vector<std::vector<std::uint64_t>> cells{};
  /// The sheet of each cell, from the certified sheeting (`SheetedSupport`).
  /// Empty: one sheet.
  std::vector<std::size_t> sheetOfCell{};
  /// The base cell of each cell: cells on different sheets with equal base
  /// cell correspond. Empty: every cell is its own base cell.
  std::vector<std::size_t> baseCellOfCell{};
  /// The symmetry action D(g) of the support's group on the cells, one matrix
  /// per group element, projective actions allowed. Empty means no symmetry
  /// was declared, and then no band can be told apart from a spin irreducible.
  /// A declared trivial group is the single identity matrix.
  std::vector<Eigen::MatrixXcd> symmetry{};
  /// The group's name, for the report.
  std::string symmetryName{};
  /// Whether the action's projective class is nontrivial
  /// (`MonopoleSupport::cocycle`): an irreducible of dimension two is then a
  /// spinor doublet, spin content.
  bool spinorial = false;
  /// The frames of the cluster's lifetime, first to last (at least one).
  std::vector<IsospinFrame> frames{};
  /// Further resolutions of the first frame (Section 5 persistence under
  /// refinement). Empty: refinement persistence is unmeasured.
  std::vector<IsospinResolution> resolutions{};
  /// An operator on the cells whose compression splits a doublet's two
  /// members, used when the in-band operator does not. Empty: none.
  Eigen::MatrixXcd memberSplitting{};
  /// The certified lineage reading of the cluster (`ClusterLineage::read`),
  /// the source of the baryon number. Empty: B is unmeasured.
  std::optional<LineageNumberRead> lineage{};
  /// The one-body density gamma of a three-quark state on the first frame's
  /// cells, gamma_ij = <a_j^dagger a_i>, so that a mode projector Pi has the
  /// occupation tr(gamma Pi). Empty: the occupation pattern is unmeasured.
  Eigen::MatrixXcd threeQuarkDensity{};
};

/// The representation content of one band of one frame.
struct IsospinBandRead {
  /// Position in the frame's band list (ascending real part, then imaginary).
  std::size_t index = 0;
  /// The eigenvalues in the band.
  std::vector<std::complex<double>> eigenvalues{};
  /// Their mean.
  std::complex<double> center{0.0, 0.0};
  /// The band rank (trace of the Riesz projector, rounded).
  std::size_t rank = 0;
  /// Distance in the complex plane to the nearest eigenvalue outside the band
  /// (infinite when there is none).
  double gap = 0.0;
  /// The contour the projector was integrated on.
  std::complex<double> contourCenter{0.0, 0.0};
  double contourRadius = 0.0;
  /// ||P^2 - P||_F / max(1, ||P||_F).
  double projectorResidual = 0.0;
  /// ||P||_2.
  double projectorNorm = 0.0;
  /// max over the contour nodes of ||(zeta - h)^{-1}||_2.
  double resolventMax = 0.0;
  /// Whether the band met the isolation, idempotency and conditioning caps.
  bool isolated = false;

  /// The number of sheets declared, and whether the band is invariant under
  /// the sheet matrix units (then colour acts on it and is divided out).
  std::size_t sheetCount = 1;
  double sheetInvarianceResidual = 0.0;
  bool colourActs = false;
  /// Whether a symmetry was declared, the band's invariance under it, and
  /// whether it acts on the band.
  bool symmetryDeclared = false;
  double symmetryInvarianceResidual = 0.0;
  bool symmetryActs = false;

  /// dim of the commutant of the acting algebra on the band.
  std::size_t commutantDimension = 0;
  /// Number of isotypes (the dimension of the commutant's centre).
  std::size_t isotypeCount = 0;
  /// Per isotype: the dimension of the irreducible of the symmetry group on
  /// one sheet (the whole irreducible when colour does not act), and the
  /// multiplicity f of that irreducible per sheet.
  std::vector<std::size_t> irreducibleDimensions{};
  std::vector<std::size_t> multiplicities{};
  /// "d x k sheets x f" per isotype, joined by " + ".
  std::string content{};
  /// Whether the band is a single irreducible of dimension two of a
  /// spinorial action: a spinor doublet, spin content.
  bool spinDoublet = false;
  /// Whether the band is one isotype of multiplicity two: two copies of one
  /// spin-and-colour irreducible, the flavour-doublet candidate.
  bool doubletCandidate = false;
  /// Whether the band is anything other than one irreducible of the acting
  /// algebra (falsifier 10 at this resolution).
  bool unexplainedMultiplicity = false;
  /// One sentence naming what the band is.
  std::string classification{};

  /// The band as a library fiber (right frame, transpose dual, certificate).
  SpectralFiber fiber{};
};

/// The bands of one frame.
struct IsospinFrameRead {
  std::string label{};
  /// Every eigenvalue, ascending real part then imaginary part.
  std::vector<std::complex<double>> spectrum{};
  std::vector<IsospinBandRead> bands{};
};

/// One frame-to-frame transport of a doublet candidate.
struct IsospinTransportStep {
  /// The frames joined, as positions in the declaration's frame list.
  std::size_t fromFrame = 0;
  std::size_t toFrame = 0;
  /// The library transport of the whole band (rank d k 2).
  GeneralLinearTransportRead transport{};
  /// max over the acting generators of
  /// ||M A_from - A_to M||_F / (||M||_F ||A||_F): zero when the transport
  /// intertwines the rotation and colour actions and so factors as I (x) m.
  double intertwiningResidual = 0.0;
  /// The two singular values of the flavour factor m, read as the two
  /// distinct singular values of the band transport (each with multiplicity
  /// d k); empty when the singular values do not come in two such groups.
  std::vector<double> flavourSingularValues{};
};

/// Isospin, charge and occupation of an observed doublet.
struct IsospinChargeRead {
  /// Where the member splitting came from: "in-band operator",
  /// "declared splitting operator" or "declared trivialization (cell-order
  /// seed)".
  std::string memberSource{};
  /// I_3 of the two members, +1/2 first.
  std::vector<double> isospin{};
  /// The member projectors on the first frame's cells, +1/2 first.
  std::vector<Eigen::MatrixXcd> memberProjectors{};
  /// B = n_Q N_Q / 3 from the certified lineage; empty when unmeasured.
  std::optional<double> baryonNumber{};
  /// Q = I_3 + B/2 of the two members; empty when B is unmeasured.
  std::vector<double> charges{};
  /// Occupations tr(gamma Pi_+) and tr(gamma Pi_-) of the three-quark state;
  /// empty when no density was supplied.
  std::vector<std::complex<double>> memberOccupations{};
  /// "uud", "udd", or a description of an occupation that is neither; empty
  /// when no density was supplied. "u" names the I_3 = +1/2 member.
  std::string occupationPattern{};
  /// Why a quantity above is missing, when one is.
  std::vector<std::string> notes{};
};

/// The read of one flavour-doublet candidate.
struct IsospinCandidateRead {
  /// The candidate's band position in the first frame.
  std::size_t bandIndex = 0;
  /// Its band position in every frame it was tracked through.
  std::vector<std::size_t> trackedBands{};
  /// Smallest subspace overlap along the track.
  double minTrackOverlap = 0.0;
  /// Per resolution: whether a candidate overlapping the prolonged band was
  /// found, and the overlap.
  std::vector<bool> resolutionFound{};
  std::vector<double> resolutionOverlap{};
  /// The transports between consecutive frames.
  std::vector<IsospinTransportStep> transports{};
  /// The singular values of the composed lifetime transport, descending.
  std::vector<double> lifetimeSingularValues{};
  /// The three conditions, in order.
  std::vector<QuarkConditionRead> conditions{};
  /// Whether conditions 1 and 2 passed.
  bool observed = false;
  /// Isospin and charges; present when `observed`.
  std::optional<IsospinChargeRead> charges{};
};

/// The whole read.
struct IsospinDoubletRead {
  std::string operatorName{};
  std::string symmetryName{};
  /// The bands of every frame.
  std::vector<IsospinFrameRead> frames{};
  /// The bands of every resolution.
  std::vector<IsospinFrameRead> resolutions{};
  /// Every flavour-doublet candidate of the first frame.
  std::vector<IsospinCandidateRead> candidates{};
  /// The three conditions as they stand for the whole read: those of the
  /// first observed candidate, else of the first candidate, else the
  /// no-candidate read (condition 1 failed by name).
  std::vector<QuarkConditionRead> conditions{};
  /// Whether some candidate passed conditions 1 and 2.
  bool doubletObserved = false;
  /// Falsifier 8, "No isospin doublet": true when no candidate passed
  /// conditions 1 and 2.
  bool noIsospinDoublet = true;
  /// Falsifier 10 at the supplied resolutions: every band, in every frame,
  /// that is not one irreducible of the acting algebra, as "frame f band b:
  /// content".
  std::vector<std::string> unexplainedMultiplicities{};
  /// Whether robustness under refinement was measured for those.
  bool multiplicityRefinementMeasured = false;
  /// One paragraph stating what was observed.
  std::string summary{};
};

/// # IsospinDoublet
///
/// The detector. See the file banner for what it reads, what it divides out,
/// and how each condition is decided.
class IsospinDoublet {
 public:
  IsospinDoublet() = delete;  // static kernel — no instances.

  /// The number of conditions, three.
  static constexpr int kConditionCount = 3;

  /// The short names of the three conditions, in order.
  [[nodiscard]] static std::vector<std::string> conditionNames();

  /// The whitepaper's statement of condition `number` (1 to 3).
  /// @throws std::invalid_argument when `number` is out of range.
  [[nodiscard]] static std::string statement(int number);

  /// The bands of one operator with their representation content.
  /// @throws std::invalid_argument on a non-square operator or a sheet, base
  ///         cell or symmetry declaration whose size does not match it.
  [[nodiscard]] static IsospinFrameRead bands(
      const Eigen::MatrixXcd& operatorMatrix,
      const std::vector<std::vector<std::uint64_t>>& cells,
      const std::vector<std::size_t>& sheetOfCell,
      const std::vector<std::size_t>& baseCellOfCell,
      const std::vector<Eigen::MatrixXcd>& symmetry, bool spinorial,
      int degree = 1, const IsospinDoubletConfig& cfg = {});

  /// The whole read.
  /// @throws std::invalid_argument on an empty frame list or on any size
  ///         mismatch in the declaration.
  [[nodiscard]] static IsospinDoubletRead observe(
      const IsospinDoubletDeclaration& declaration,
      const IsospinDoubletConfig& cfg = {});
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_ISOSPINDOUBLET_H
