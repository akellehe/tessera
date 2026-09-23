// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_LEVELRECURSION_H
#define TESSERA_COBORDISM_LEVELRECURSION_H

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cobordism/Certificate.h"
#include "cobordism/JointAction.h"
#include "cobordism/RecursiveQuotient.h"

namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;

/// # RecursionBandSelection
///
/// How the closed contour \f$ \gamma_v \f$ of a response vertex is fixed.
///
/// The whitepaper selects a band by a closed contour in the complex spectral
/// plane and never by sorting real parts or imaginary parts, so the contour is
/// the declared object. The two options below differ only in who writes the
/// contour down.
///
/// * `LowestModes` — the contour is derived from a declared band rank: the
///   block's eigenvalues are put in the declared `OccupationOrder`, the first
///   `bandRank` of them are the band, the centre is their mean and the radius
///   sits halfway between the farthest band member and the nearest excluded
///   eigenvalue, both measured from that centre. The ordering is a rule for
///   writing down a contour and never a substitute for one: the projector is
///   still the contour integral, and the isolation gap it reports is the
///   distance that makes the contour legitimate.
/// * `DeclaredContours` — the caller supplies one centre and one radius per
///   component, and nothing is sorted at all.
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

  /// Which eigenvalues those are under `LowestModes`.
  OccupationOrder order = OccupationOrder::AscendingRealPart;

  /// The number of quadrature nodes on each contour. The Riesz projector is
  /// \f$ P_v=\frac{1}{2\pi i}\oint_{\gamma_v}(\zeta I-h_v)^{-1}d\zeta \f$,
  /// evaluated by the trapezoidal rule on the circle, which converges
  /// geometrically in the node count for a contour that separates the spectrum.
  int contourNodes = 64;

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

  /// The relative tolerance the certificates of every level hold against.
  double tolerance = 1e-9;

  /// The dimension at and above which the dense paths of this class refuse. The
  /// response pencil is evaluated densely, so this is the size of the largest
  /// level it will build.
  int denseCrossover = 512;
};

/// # RecursionBandRead
///
/// One certified fiber \f$ E_v^{\ell+1}=\operatorname{Ran}P_v^{\ell+1} \f$ and
/// the contour that produced it.
struct RecursionBandRead {
  /// The component of the level's partition this fiber belongs to: the response
  /// vertex it becomes at the next level.
  int component = 0;

  /// \f$ r_v \f$, the rank the projector actually reached.
  std::size_t rank = 0;

  /// The centre of \f$ \gamma_v \f$.
  std::complex<double> contourCentre{0.0, 0.0};
  /// The radius of \f$ \gamma_v \f$.
  double contourRadius = 0.0;
  /// The number of quadrature nodes on \f$ \gamma_v \f$.
  int contourNodes = 0;

  /// The eigenvalues of the component's block that the contour encloses.
  std::vector<std::complex<double>> eigenvalues;

  /// The distance from \f$ \gamma_v \f$ to the nearest eigenvalue of the
  /// block, inside or outside: the isolation the contour rests on. Positive for
  /// a contour that separates the spectrum, and quiet NaN when the block has no
  /// eigenvalue off the contour to measure against.
  double isolationGap = 0.0;

  /// \f$ \lVert P_v^2-P_v\rVert_F \f$ of the quadrature's projector: the
  /// measure of how well the node count resolved the contour integral.
  double projectorIdempotency = 0.0;

  /// \f$ \lVert\tilde\Phi_v^{\mathsf T}\Phi_v-I_{r_v}\rVert_F \f$: how far the
  /// two frames are from the bilinear pairing the whitepaper asks of them.
  double pairingDefect = 0.0;

  /// \f$ \Phi_v \f$, the fiber's right frame over the level's coordinates, flat
  /// row-major (level dimension by \f$ r_v \f$). Its columns are supported on
  /// the component's coordinates and are zero elsewhere.
  std::vector<std::complex<double>> frame;

  /// \f$ \tilde\Phi_v^{\mathsf T} \f$, the fiber's left frame, flat row-major
  /// (\f$ r_v \f$ by level dimension), the algebraic dual of `frame` with no
  /// conjugation in the pairing.
  std::vector<std::complex<double>> leftFrame;

  /// Whether the contour separated the spectrum, the projector came out
  /// idempotent and the two frames paired to the identity, all at the declared
  /// tolerance. An unaccepted fiber is still carried and reported; it makes the
  /// level's certificate fail to hold rather than disappearing.
  bool accepted = false;

  /// The fiber's certificate, whose residual is the worst of the idempotency
  /// and the pairing defect.
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
/// \f$ \lambda \f$ the caller names. Nothing is linearized and no square root,
/// polar projection or eigenvalue ordering enters the recursion: the ordering
/// the band declaration names is a rule for writing down a contour, and the
/// projector is still the contour integral.
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
  ///   matrix has the wrong size, when the declared resolutions are empty, when
  ///   the declared band rank is zero, or when the contour node count is below
  ///   three; std::length_error at or above the declared dense crossover.
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

  /// The number of coordinates of the microscopic level.
  [[nodiscard]] int baseDimension() const noexcept { return dimension_; }

  /// How many turns of the box have been taken. Level \p index of
  /// `responsePencil` is available for every `index` up to and including this.
  [[nodiscard]] std::size_t levelCount() const noexcept {
    return levels_.size();
  }

  /// Take one turn of the box.
  /// @throws std::length_error when the level to be reduced is at or above the
  ///   declared dense crossover, or when a level has been reduced to nothing
  ///   and there is no further response pencil to partition.
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
