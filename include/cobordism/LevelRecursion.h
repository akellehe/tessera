// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_LEVELRECURSION_H
#define TESSERA_COBORDISM_LEVELRECURSION_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "cobordism/Certificate.h"
#include "cobordism/JointAction.h"
#include "cobordism/RecursiveQuotient.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # RecursionBandSelection
///
/// How the band of a response vertex, the set of eigenvalues of its block whose
/// invariant subspace is the fiber, is selected.
///
/// The whitepaper selects a band by a closed contour \f$ \gamma_v \f$ in the
/// complex spectral plane, and the band is the set of eigenvalues the contour
/// encloses. The contour is recorded as the selection, as a centre and a
/// radius; the projector onto the band's invariant subspace is then formed
/// exactly from the block's eigendecomposition (`LevelRecursion::readBand`),
/// and the two options below differ only in who writes the contour down.
///
/// * `LowestModes` — the contour is derived from a declared band rank: the
///   block's eigenvalues are put in the declared `OccupationOrder`, the first
///   `bandRank` of them are the band, the centre is their mean and the radius
///   sits halfway between the farthest band member and the nearest excluded
///   eigenvalue, both measured from that centre. When no eigenvalue is
///   excluded the selection encloses the whole spectrum, which the read
///   records as such with an infinite radius. Two eigenvalues equal at the
///   declared tolerance are one eigenvalue with multiplicity. A band rank
///   that separates them is read like any other, from the order the exact
///   keys give, and the read reports it: its isolation gap is at or below
///   the declared tolerance times the block's Frobenius norm and the fiber
///   is not accepted (`RecursionBandRead::accepted`).
/// * `DeclaredContours` — the caller supplies one centre and one radius per
///   component, and nothing is sorted at all. The band is the set of
///   eigenvalues strictly inside the circle: an eigenvalue \f$ \lambda \f$
///   belongs to it exactly when \f$ |\lambda-c|<r \f$. The distance from the
///   circle to the nearest eigenvalue is reported
///   (`RecursionBandRead::contourGap`), and a circle that passes through an
///   eigenvalue at the declared tolerance yields a fiber that is not
///   accepted. A circle enclosing no eigenvalue yields the band of rank
///   zero, whose projector is zero and whose frames have no column, and
///   which is not accepted.
///
/// Under either option one selection has no projector at all: one that takes
/// an eigenvalue into the band and leaves an eigenvalue exactly equal to it
/// out. The projector onto a part of a multiple eigenvalue's invariant
/// subspace is not defined, the Sylvester equation that would produce it is
/// singular, and `LevelRecursion::readBand` says so by name.
enum class RecursionBandSelection { LowestModes, DeclaredContours };

/// # RecursionBandDeclaration
///
/// How every level's fibers are selected.
struct RecursionBandDeclaration {
  /// Which rule fixes the contours (see `RecursionBandSelection`).
  RecursionBandSelection selection = RecursionBandSelection::LowestModes;

  /// \f$ r_v \f$, the number of eigenvalues the contour of each response vertex
  /// encloses under `LowestModes`. A block with fewer coordinates than this
  /// encloses all of them, and the band reports the rank it actually reached.
  std::size_t bandRank = 1;

  /// Which eigenvalues those are under `LowestModes`. `AscendingRealPart`
  /// orders by real part, then by imaginary part; `AscendingModulus` orders by
  /// modulus, then by real part, then by imaginary part. Every key is compared
  /// at the declared tolerance, relative to the block's Frobenius norm, so that
  /// an order is never decided by rounding: two eigenvalues whose keys agree
  /// at that tolerance are ordered by the next key, and two whose every key
  /// agrees are one eigenvalue with multiplicity.
  OccupationOrder order = OccupationOrder::AscendingRealPart;

  /// The contour centres under `DeclaredContours`, one per component of the
  /// level's partition, in component order.
  std::vector<std::complex<double>> contourCentres;

  /// The contour radii under `DeclaredContours`, matching `contourCentres`.
  std::vector<double> contourRadii;
};

/// # LevelRecursionDeclaration
///
/// Everything that fixes how a recursion is driven. None of it changes which
/// identities hold; it fixes which partition, which contour and which reference
/// point a level is built at.
struct LevelRecursionDeclaration {
  /// The modularity resolutions \f$ \gamma \f$ the partition of every level is
  /// swept over, in scan order. A single entry is a single-resolution
  /// partition; several entries make the partition a persistence statement, and
  /// the components that survive the most adjacent resolutions are the ones the
  /// level carries.
  std::vector<double> resolutions{1.0};

  /// How many restarts the modularity search takes at each resolution.
  int modularityRestarts = 4;

  /// The seed the modularity search runs from, so a level's partition is
  /// reproducible.
  std::uint64_t modularitySeed = 0;

  /// The support overlap two components of adjacent resolutions must share to
  /// be matched into one persistence track.
  double persistenceOverlap = 0.5;

  /// \f$ \lambda_{\rm ref} \f$, the spectral parameter each level's partition
  /// and fibers are read at.
  ///
  /// The partition and the fibers are declared structure of a level, not
  /// functions of the spectral parameter: the whitepaper's box discovers
  /// \f$ P_\ell \f$ from \f$ \mathcal R_\ell \f$ and then applies the Feshbach
  /// map to \f$ \mathcal R_\ell(\lambda) \f$ at every \f$ \lambda \f$. The
  /// energy dependence is therefore carried exactly by `responsePencil`, which
  /// re-derives the whole chain at whatever \f$ \lambda \f$ it is asked for,
  /// while the structure stays where it was declared.
  std::complex<double> referenceLambda{0.0, 0.0};

  /// How the fibers of every level are selected.
  RecursionBandDeclaration bands;

  /// The relative tolerance the certificates of every level hold against and
  /// the rank decisions of the band reads are made at: which eigenvalues of a
  /// block are equal, and whether the declared selection separates them.
  double tolerance = 1e-15;

  /// The relative threshold of the rank decisions of the quotient's interior
  /// solves (`RecursiveQuotient::Options::rankTolerance`): a pivot or a
  /// singular value of an interior block at or below this fraction of the
  /// block's largest counts as zero, which decides the interior nullity and
  /// so the harmonic modes a level retains.
  double rankTolerance = 1e-15;

  /// A limit the caller may declare on the size of a level: the dimension at
  /// and above which this class refuses to build or advance a level. The
  /// response pencil is evaluated densely, so the cost of a level grows as the
  /// cube of its dimension. No limit is declared by default, and then a level
  /// of any dimension is built and every interior solve of the quotient takes
  /// its dense path.
  std::optional<int> denseCrossover{};
};

/// # RecursionBandRead
///
/// One certified fiber \f$ E_v^{\ell+1}=\operatorname{Ran}P_v^{\ell+1} \f$,
/// the selection that named its band, and the certificates of the exact
/// projector.
///
/// The projector is formed from the complex Schur form \f$ h_v=QTQ^H \f$ of
/// the component's block: the selected eigenvalues are reordered to the leading
/// diagonal block \f$ T_{11} \f$ by unitary swaps, the Sylvester equation
/// \f$ T_{11}Y-YT_{22}=T_{12} \f$ is solved by back substitution, and
/// \f$ P_v=Q\begin{pmatrix}I&Y\\0&0\end{pmatrix}Q^H=\Phi_v\tilde\Phi_v^{\mathsf T} \f$
/// with \f$ \Phi_v \f$ the leading Schur vectors and
/// \f$ \tilde\Phi_v^{\mathsf T}=(I\;\;Y)\,Q^H \f$. For a diagonalizable block
/// this is \f$ V_B(V^{-1})_B \f$ over the selected eigenvalues; for a
/// non-diagonalizable one it is the same spectral projector, formed without
/// inverting an eigenvector matrix. A selection that encloses every eigenvalue
/// of the block has the whole coordinate space as its invariant subspace:
/// its projector is the identity and its two frames are the canonical basis,
/// with nothing to reorder and no equation to solve, so its certificates are
/// zero exactly. A selection that encloses no eigenvalue has the zero
/// subspace as its invariant subspace: its projector is zero, its two frames
/// have no column, and its residuals are zero exactly.
///
/// A read is made whatever the isolation of its band. The preconditions of a
/// certified fiber are reported beside the result, in `isolationGap`,
/// `contourGap` and `accepted`, and never withhold it.
struct RecursionBandRead {
  /// The component of the level's partition this fiber belongs to: the response
  /// vertex it becomes at the next level.
  int component = 0;

  /// \f$ r_v \f$, the number of selected eigenvalues, counted with
  /// multiplicity, and the rank of the projector: \f$ P_v=\Phi_v\tilde\Phi_v^{\mathsf T} \f$
  /// through \f$ r_v \f$ columns bounds the rank above, and
  /// \f$ \tilde\Phi_v^{\mathsf T}\Phi_v=I_{r_v} \f$ bounds it below.
  std::size_t rank = 0;

  /// The centre of \f$ \gamma_v \f$, the recorded selection: the mean of the
  /// selected eigenvalues under `LowestModes`, the declared centre under
  /// `DeclaredContours`.
  std::complex<double> contourCentre{0.0, 0.0};
  /// The radius of \f$ \gamma_v \f$: halfway between the farthest selected and
  /// the nearest excluded eigenvalue from the centre under `LowestModes`, the
  /// declared radius under `DeclaredContours`. Infinite when the selection
  /// under `LowestModes` excludes no eigenvalue, where there is no nearest
  /// excluded eigenvalue to measure to.
  double contourRadius = 0.0;
  /// Whether the selection excludes no eigenvalue of the block, so that the
  /// fiber is the whole of the component.
  bool enclosesEverything = false;

  /// The eigenvalues of the component's block that the selection encloses, in
  /// the declared order under `LowestModes` and ascending by real then
  /// imaginary part under `DeclaredContours`.
  std::vector<std::complex<double>> eigenvalues;

  /// The smallest distance between a selected and an excluded eigenvalue of
  /// the block: the isolation of the band, which is what makes its invariant
  /// subspace a spectral one and the Sylvester equation solvable. Infinite when
  /// the selection excludes no eigenvalue or selects none. A gap at or below
  /// the declared tolerance times the block's Frobenius norm says that the
  /// selection separates two eigenvalues equal at that tolerance; the fiber
  /// is then read from the order the exact keys give and is not accepted.
  /// The gap is positive whenever the read returns, because a gap of exactly
  /// zero leaves the projector without a value.
  double isolationGap = 0.0;

  /// The smallest distance between the declared circle and an eigenvalue of
  /// the block under `DeclaredContours`,
  /// \f$ \min_i\bigl||\lambda_i-c|-r\bigr| \f$: how far the nearest
  /// eigenvalue is from changing sides of the circle. An eigenvalue is on the
  /// circle at the declared tolerance when this distance is at or below the
  /// tolerance times the larger of the radius and the eigenvalue's distance
  /// from the centre; its membership is then the strict comparison all the
  /// same, and the fiber is not accepted. Infinite under `LowestModes`, where
  /// the recorded circle is derived from the selection and decides no
  /// membership.
  double contourGap = 0.0;

  /// \f$ \lVert P_v^2-P_v\rVert_F/\lVert P_v\rVert_F \f$: the rounding
  /// residual of the exact projector's idempotency.
  double projectorIdempotency = 0.0;

  /// \f$ \lVert\tilde\Phi_v^{\mathsf T}\Phi_v-I_{r_v}\rVert_F \f$: how far the
  /// two frames are from the bilinear pairing the whitepaper asks of them.
  double pairingDefect = 0.0;

  /// \f$ \lVert h_v\Phi_v-\Phi_v(\tilde\Phi_v^{\mathsf T}h_v\Phi_v)\rVert_F/
  /// \lVert h_v\rVert_F \f$: the residual of the invariant subspace, zero for
  /// a zero block, whose numerator vanishes identically.
  double invariantSubspaceResidual = 0.0;

  /// \f$ \Phi_v \f$, the fiber's right frame over the level's coordinates, flat
  /// row-major (level dimension by \f$ r_v \f$). Its columns are supported on
  /// the component's coordinates and are zero elsewhere.
  std::vector<std::complex<double>> frame;

  /// \f$ \tilde\Phi_v^{\mathsf T} \f$, the fiber's left frame, flat row-major
  /// (\f$ r_v \f$ by level dimension), the algebraic dual of `frame` with no
  /// conjugation in the pairing.
  std::vector<std::complex<double>> leftFrame;

  /// Whether the fiber is certified at the declared tolerance. That needs the
  /// preconditions of the read to hold: the band has at least one eigenvalue,
  /// the selection separates no two eigenvalues equal at the tolerance
  /// (`isolationGap` above the tolerance times the block's Frobenius norm),
  /// and no eigenvalue is on a declared circle at the tolerance
  /// (`contourGap`). It also needs the projector to have come out idempotent,
  /// the two frames to pair to the identity and the frame to span an
  /// invariant subspace, each residual a finite number at or below the
  /// tolerance. An unaccepted fiber is still carried and reported; it makes
  /// the level's certificate fail to hold rather than disappearing.
  bool accepted = false;

  /// The fiber's certificate, which holds exactly when `accepted` is true.
  /// Its residual is the worst of the idempotency, the pairing defect and the
  /// invariant-subspace residual when the preconditions of the read hold, and
  /// infinite when one of them does not.
  Certificate certificate{};
};

/// # LevelTransport
///
/// One block \f$ M_{vw}^{\ell+1}=\tilde\Phi_v^{\ell+1\,\mathsf T}
/// T_{vw}^{\ell}\Phi_w^{\ell+1}\in\operatorname{Hom}(E_w^{\ell+1},
/// E_v^{\ell+1}) \f$: the level's coupling between two response vertices, read
/// in their own fibers.
struct LevelTransport {
  /// The component the block maps from, which is \f$ w \f$.
  int from = 0;
  /// The component the block maps to, which is \f$ v \f$.
  int to = 0;
  /// The block, flat row-major (\f$ r_v \f$ by \f$ r_w \f$).
  std::vector<std::complex<double>> block;
};

/// # RecursionLevelRead
///
/// One turn of the whitepaper's box: the partition, the fibers, the labeled
/// sum, and the response pencil of the level above.
struct RecursionLevelRead {
  /// \f$ \ell \f$, the level this step reduced. The step produces level
  /// \f$ \ell+1 \f$.
  std::size_t level = 0;

  /// The number of coordinates of \f$ \mathcal R_\ell \f$, which is what the
  /// partition ran on.
  int dimension = 0;

  /// \f$ P_\ell \f$: the coordinates of each component, ascending, covering
  /// every coordinate exactly once.
  std::vector<std::vector<int>> partition;

  /// The resolutions the partition was swept over, in scan order.
  std::vector<double> resolutions;

  /// The resolution the carried partition came from.
  double selectedResolution = 0.0;

  /// How many adjacent resolutions each carried component survived, in
  /// component order. One means the component exists at the selected resolution
  /// and at no neighbour of it.
  std::vector<double> componentPersistence;

  /// The weakest adjacent-resolution support overlap over the carried
  /// components' persistence tracks. Quiet NaN when the sweep has one
  /// resolution, where there is no adjacent slice to overlap with.
  double worstPersistenceOverlap = 0.0;

  /// The certified fibers, one per component of `partition`, in component
  /// order.
  std::vector<RecursionBandRead> bands;

  /// \f$ Y_{\ell+1} \f$, the embedding of the labeled sum into the level's
  /// coordinates, flat row-major (`dimension` by `modes`): the fibers' right
  /// frames side by side.
  std::vector<std::complex<double>> embedding;

  /// \f$ \tilde Y_{\ell+1}^{\mathsf T} \f$, the matching left embedding, flat
  /// row-major (`modes` by `dimension`).
  std::vector<std::complex<double>> dualEmbedding;

  /// \f$ \mathcal G_{\ell+1}=\tilde Y_{\ell+1}^{\mathsf T}Y_{\ell+1} \f$, flat
  /// row-major (`modes` by `modes`). The pairing is the transpose one, so this
  /// is the bilinear overlap of the labeled sum and not a Gram matrix of inner
  /// products. Its diagonal blocks are the identity by construction; an
  /// off-diagonal block is nonzero exactly when two fibers' supports meet,
  /// which is why the internal sum is never asserted to be direct.
  std::vector<std::complex<double>> gram;

  /// \f$ \lVert\mathcal G_{\ell+1}-I\rVert_2 \f$.
  double gramDefect = 0.0;

  /// \f$ M=\dim\mathfrak h^{\ell+1}=\sum_v r_v \f$, the labeled sum's rank.
  std::size_t modes = 0;

  /// \f$ \mathfrak h^{\ell+1}=\tilde Y_{\ell+1}^{\mathsf T}\,
  /// \mathcal R_\ell(\lambda_{\rm ref})\,Y_{\ell+1} \f$, flat row-major
  /// (`modes` by `modes`): the level's operator read in the fibers. Its
  /// \f$ (v,w) \f$ block is the transport \f$ M_{vw}^{\ell+1} \f$.
  std::vector<std::complex<double>> fiberOperator;

  /// The eigenvalues of `fiberOperator`, ascending by real then imaginary part.
  std::vector<std::complex<double>> fiberSpectrum;

  /// The blocks of `fiberOperator` named by the pair of components they run
  /// between, for every pair whose block is nonzero.
  std::vector<LevelTransport> transports;

  /// \f$ \dim F_-(\mathfrak h^{\ell+1})=2^M \f$ as a double, exact through
  /// \f$ 2^{53} \f$ and infinite beyond: the Fock stage the level's one-particle
  /// space carries. The vector is never allocated.
  double fockStageDimension = 0.0;

  /// The modes the interaction stage adds to \f$ \mathfrak h^{\ell+1} \f$
  /// beyond the ones the reduction retained, which the state reaches by the
  /// vacuum embedding \f$ \iota:\psi\mapsto\psi\wedge|0\rangle_{\rm new} \f$ of
  /// Section 12. It is zero until interactions are attached, because the
  /// reduction alone grows nothing.
  std::size_t vacuumEmbeddedModes = 0;

  /// The number of coordinates of \f$ \mathcal R_{\ell+1} \f$: the kept
  /// coordinates of the Feshbach map at \f$ \lambda_{\rm ref} \f$.
  int responseDimension = 0;

  /// \f$ |\det\mathcal R_\ell-\det(\mathcal R_\ell)_{II}
  ///     \det\mathcal R_{\ell+1}|/|\det\mathcal R_\ell| \f$ at
  /// \f$ \lambda_{\rm ref} \f$: the determinant factorization that makes the
  /// step exact, measured. Quiet NaN when the reduction retained an interior
  /// mode, where the interior block is singular and the factorization is not
  /// the statement to check.
  double determinantResidual = 0.0;

  /// The producing Feshbach step's own certificate, carried verbatim.
  Certificate reductionCertificate{};

  /// The level's certificate: `StructureExact`, holding when every fiber was
  /// accepted and the determinant factorization met the declared tolerance.
  Certificate certificate{};
};

/// # LevelRecursion
///
/// The master recursive construction of Section 15 of the whitepaper, driven
/// level by level with the energy dependence kept exact.
///
/// Reference: Feshbach, "Unified theory of nuclear reactions", Annals of
/// Physics 5, 357 (1958) — the energy-dependent effective operator on a
/// retained subspace.
/// Reference: Kato, "Perturbation Theory for Linear Operators", Springer (1966),
/// II.1.4 — the Riesz projector of a spectral set enclosed by a contour.
/// Reference: Newman and Girvan, "Finding and evaluating community structure in
/// networks", arXiv:cond-mat/0308217 — the modularity the partition is swept
/// over.
///
/// ## The box
///
/// With \f$ \mathcal R_0(\lambda)=h(z,U)-\lambda M \f$ the microscopic response
/// pencil, every scale runs
/// \f[
///   P_\ell=\mathrm{PersistentPartition}(\mathcal R_\ell),\qquad
///   P_v^{\ell+1}=\frac{1}{2\pi i}\oint_{\gamma_v}(\zeta I-h_v^\ell)^{-1}d\zeta,
///   \qquad E_v^{\ell+1}=\operatorname{Ran}P_v^{\ell+1},
/// \f]
/// where the Riesz projector \f$ P_v^{\ell+1} \f$ of the band the contour
/// \f$ \gamma_v \f$ encloses is formed exactly, as the spectral projector of
/// those eigenvalues from the complex Schur form of the block
/// (`readBand`), the contour being recorded as the selection.
/// \f[
///   \mathcal R_{\ell+1}(\lambda)=\mathrm{Feshbach}_{P_\ell}
///     (\mathcal R_\ell(\lambda)),\qquad
///   M_{vw}^{\ell+1}=\tilde\Phi_v^{\ell+1\,\mathsf T}T_{vw}^\ell\Phi_w^{\ell+1},
/// \f]
/// \f[
///   \mathfrak h^{\ell+1}=\boxplus_v E_v^{\ell+1},\qquad
///   \mathcal G_{\ell+1}=\tilde Y_{\ell+1}^{\mathsf T}Y_{\ell+1},\qquad
///   \mathcal H^{\ell+1}=F_-(\mathfrak h^{\ell+1}).
/// \f]
/// `advance` performs one turn of it and records a `RecursionLevelRead`.
///
/// ## The energy dependence is exact
///
/// \f$ \mathcal R_\ell \f$ is a function of \f$ \lambda \f$, never a matrix
/// frozen at one value of it. `responsePencil(level, lambda)` re-derives the
/// whole chain from the microscopic pencil: it forms
/// \f$ \mathcal R_0(\lambda)=A-\lambda M \f$ and then applies the declared
/// partition of each level in turn as a plain supported block elimination, with
/// no further shift, because the spectral parameter is already inside the
/// matrix being eliminated. Shifting again at a later level would subtract
/// \f$ \lambda \f$ twice and is the one thing the exactness of the step rests
/// on not happening.
///
/// The step is exact because of the determinant factorization
/// \f$ \det\mathcal R_\ell(\lambda)=\det(\mathcal R_\ell(\lambda))_{II}\,
///     \det\mathcal R_{\ell+1}(\lambda) \f$,
/// so a \f$ \lambda \f$ is an eigenvalue of the microscopic pencil exactly when
/// it is a zero of \f$ \det\mathcal R_\ell \f$ or of one of the interior
/// determinants the chain eliminated along the way. Every level's spectrum is
/// therefore reproduced from the level below it, and
/// `determinantFactorizationResidual` measures the identity at any
/// \f$ \lambda \f$ the caller names. Nothing is linearized and no square root
/// or polar projection enters the recursion: the ordering the band
/// declaration names is a rule for selecting a band, and the projector is the
/// exact spectral projector of the band it selects.
///
/// ## What the reduction does not supply
///
/// The reduction determines no incidence maps of its own. The cells of
/// \f$ K^{\ell+1} \f$ come from the interactions among the response vertices,
/// and the tick that carries one level into the next is the mapping cylinder
/// \f$ W^\ell \f$ of `MappingCylinder`, built over the reduction map this class
/// reports as `componentOfCoordinate`. A level here is the change of scale; it
/// is the pair of the reduction and the interaction stage that is a tick of
/// time.
class LevelRecursion {
 public:
  /// Build over an explicit pencil \f$ (A, M) \f$.
  ///
  /// @param pencil \f$ A \f$, flat row-major \p dimension by \p dimension.
  /// @param metric \f$ M \f$, flat row-major and of the same shape, or empty
  ///   for the identity, in which case the microscopic pencil is
  ///   \f$ A-\lambda I \f$.
  /// @param dimension The number of microscopic coordinates.
  /// @param declaration How the recursion is driven.
  /// @throws std::invalid_argument when the dimension is not positive, when a
  ///   matrix has the wrong size, when the declared resolutions are empty, or
  ///   when the declared band rank is zero; std::length_error at or above a
  ///   declared dense crossover.
  [[nodiscard]] static LevelRecursion overPencil(
      const std::vector<std::complex<double>> &pencil,
      const std::vector<std::complex<double>> &metric, int dimension,
      LevelRecursionDeclaration declaration);

  /// Build over a triangulation's Hodge operator at \p degree, which is the
  /// whitepaper's microscopic edge-mode response pencil at \p degree one. The
  /// metric is the identity, so the microscopic pencil is
  /// \f$ h_k(z,U)-\lambda I \f$.
  /// @param spacetime The complex the operator is read from.
  /// @param degree The simplicial degree of the carrier.
  /// @param metricSource Whether the operator's metric is the diagonal
  ///   per-simplex weights or the Whitney Hodge pencil.
  /// @param declaration How the recursion is driven.
  /// @throws std::invalid_argument when \p spacetime is null or carries no cell
  ///   of the degree, and as `overPencil` otherwise.
  [[nodiscard]] static LevelRecursion overSpacetime(
      const std::shared_ptr<Spacetime> &spacetime, int degree,
      HodgeLaplacian::MetricSource metricSource,
      LevelRecursionDeclaration declaration);

  /// The declaration this recursion is driven by.
  [[nodiscard]] const LevelRecursionDeclaration &declaration() const noexcept {
    return declaration_;
  }

  /// The exact band read of one block: the selection the declaration names
  /// for component \p component, the Riesz projector onto the invariant
  /// subspace of the selected eigenvalues from the block's complex Schur form,
  /// its frames and its certificates (see `RecursionBandRead`). `advance`
  /// makes this read for every component of a level; on its own it reads a
  /// block as a level of one component, so the frames are over the block's
  /// own coordinates.
  ///
  /// The read is made whatever the isolation of the band: a selection that
  /// separates two eigenvalues equal at \p tolerance, a declared circle that
  /// passes through an eigenvalue at \p tolerance and a declared circle that
  /// encloses no eigenvalue each return their band, with `accepted` false
  /// and the measured gap beside it (see `RecursionBandRead`).
  /// @param block \f$ h_v \f$, flat row-major \p order by \p order.
  /// @param order The number of coordinates of the block.
  /// @param bands How the band is selected.
  /// @param component Which declared contour applies under
  ///   `DeclaredContours`; recorded on the read either way.
  /// @param tolerance The relative tolerance the rank decisions are made at
  ///   and the certificates hold against.
  /// @throws std::invalid_argument when the block is not square of the given
  ///   order, when the declared band rank is zero, or when the declared
  ///   contours do not name \p component; std::domain_error when the
  ///   selection takes an eigenvalue into the band and leaves an eigenvalue
  ///   exactly equal to it out, where the Sylvester equation of the projector
  ///   is singular and the projector has no value; std::runtime_error when
  ///   the Schur decomposition of the block does not converge.
  [[nodiscard]] static RecursionBandRead readBand(
      const std::vector<std::complex<double>> &block, int order,
      const RecursionBandDeclaration &bands, std::size_t component,
      double tolerance);

  /// The number of coordinates of the microscopic level.
  [[nodiscard]] int baseDimension() const noexcept { return dimension_; }

  /// How many turns of the box have been taken. Level \p index of
  /// `responsePencil` is available for every `index` up to and including this.
  [[nodiscard]] std::size_t levelCount() const noexcept {
    return levels_.size();
  }

  /// Take one turn of the box. A fiber whose band read is not accepted is
  /// carried into the level and makes the level's certificate fail to hold; a
  /// band of rank zero contributes no mode and no transport.
  /// @throws std::length_error when the level to be reduced is at or above a
  ///   declared dense crossover, or when a level has been reduced to nothing
  ///   and there is no further response pencil to partition;
  ///   std::domain_error when the band read of a component has no value (see
  ///   `readBand`), in which case no level is recorded.
  void advance();

  /// Take turns until `levelCount()` reaches \p levels.
  void advanceTo(std::size_t levels);

  /// One completed turn of the box.
  /// @throws std::out_of_range when \p index names no completed level.
  [[nodiscard]] const RecursionLevelRead &level(std::size_t index) const;

  /// The component of \f$ P_\ell \f$ each coordinate of level \p index belongs
  /// to, in coordinate order: the reduction map, as a map on coordinates. It is
  /// the map `MappingCylinder` builds \f$ W^\ell \f$ over when the level's
  /// coordinates are the vertices of \f$ K^\ell \f$.
  /// @throws std::out_of_range when \p index names no completed level.
  [[nodiscard]] std::vector<int> componentOfCoordinate(std::size_t index) const;

  /// \f$ \mathcal R_\ell(\lambda) \f$, flat row-major over the level's
  /// coordinates: the exact energy-dependent response pencil of level
  /// \p level, re-derived from the microscopic pencil at this \f$ \lambda \f$.
  ///
  /// Level zero is \f$ A-\lambda M \f$. Every level above it is the supported
  /// block elimination of the level below at the same \f$ \lambda \f$, taken
  /// with no further shift.
  /// @throws std::out_of_range when \p level exceeds `levelCount()`.
  [[nodiscard]] std::vector<std::complex<double>> responsePencil(
      std::size_t level, std::complex<double> lambda) const;

  /// The number of coordinates of \f$ \mathcal R_\ell \f$.
  /// @throws std::out_of_range when \p level exceeds `levelCount()`.
  [[nodiscard]] int responseDimension(std::size_t level) const;

  /// \f$ \det\mathcal R_\ell(\lambda) \f$: the function whose zeros, together
  /// with the interior determinants the chain eliminated, are the microscopic
  /// spectrum.
  /// @throws as `responsePencil`.
  [[nodiscard]] std::complex<double> responseDeterminant(
      std::size_t level, std::complex<double> lambda) const;

  /// \f$ \det(\mathcal R_\ell(\lambda))_{II} \f$, the determinant of the
  /// interior block the step from level \p level eliminates: the product over
  /// the components of \f$ P_\ell \f$ of their interior blocks' determinants.
  /// @throws std::out_of_range when \p level names no completed level.
  [[nodiscard]] std::complex<double> interiorDeterminant(
      std::size_t level, std::complex<double> lambda) const;

  /// \f$ |\det\mathcal R_\ell-\det(\mathcal R_\ell)_{II}\,
  ///      \det\mathcal R_{\ell+1}|/|\det\mathcal R_\ell| \f$ at \p lambda: the
  /// identity that makes the step exact, measured at a spectral parameter of
  /// the caller's choosing rather than at the one the level was built at.
  /// @throws std::out_of_range when \p level names no completed level.
  [[nodiscard]] double determinantFactorizationResidual(
      std::size_t level, std::complex<double> lambda) const;

  /// Record that the interaction stage of Section 6 attached \p modes new
  /// one-particle modes to level \p index, which the carried state reaches by
  /// the vacuum embedding. The reduction alone grows nothing, so this is the
  /// only way the count moves.
  /// @throws std::out_of_range when \p index names no completed level.
  void recordVacuumEmbeddedModes(std::size_t index, std::size_t modes);

 private:
  LevelRecursion() = default;

  [[nodiscard]] RecursiveQuotient::Options quotientOptions() const;
  [[nodiscard]] RecursiveQuotient quotientAt(std::size_t level,
                                             std::complex<double> lambda) const;

  std::vector<std::complex<double>> pencil_;
  std::vector<std::complex<double>> metric_;
  int dimension_ = 0;
  LevelRecursionDeclaration declaration_;
  std::vector<RecursionLevelRead> levels_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_LEVELRECURSION_H
