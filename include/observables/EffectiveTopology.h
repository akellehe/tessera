// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_EFFECTIVETOPOLOGY_H
#define TESSERA_OBSERVABLES_EFFECTIVETOPOLOGY_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/SparsePencilSolver.h"

namespace tessera::observables {

/// The gap an effective read is held to unless the caller declares another:
/// the enclosed band must sit at least an order of magnitude below the rest of
/// the spectrum, \f$ \lambda_{\mathrm{out}} / \lambda_{\mathrm{in}} \ge 10 \f$.
/// The whitepaper requires a gap and fixes no number; this is a declared
/// policy, recorded on every read as `EffectiveBettiNumber::minimumGap`.
inline constexpr double kDefaultMinimumGap = 10.0;

/// The effective Betti number of one degree of a declared operator at a scale.
struct EffectiveBettiNumber {
  /// How the count was obtained.
  enum class Method {
    /// The dense spectrum of the degree's pencil (below the crossover).
    DenseSpectrum,
    /// `chainhodge::SparsePencilSolver::effectiveBetti` on the sparse degree-zero pencil.
    SparsePencil,
    /// Not measured: the degree has no sparse pencil and is at or above the
    /// crossover, or the sparse pencil is not Hermitian. `reason` says which.
    Unmeasured,
  };

  /// The chain degree \f$ k \f$.
  int degree{0};
  /// \f$ \beta_k^{\mathrm{eff}}(\epsilon) \f$: the number of eigenvalues with
  /// \f$ |\lambda| \le \epsilon \f$, with multiplicity; \f$ -1 \f$ when
  /// unmeasured.
  int rank{-1};
  /// The largest modulus inside the window; quiet NaN when the band is empty.
  double lastInside{std::numeric_limits<double>::quiet_NaN()};
  /// The smallest modulus outside the window; quiet NaN when there is none.
  double firstOutside{std::numeric_limits<double>::quiet_NaN()};
  /// `firstOutside / lastInside`: \f$ +\infty \f$ for an empty band or a band
  /// at zero, quiet NaN when nothing lies outside the window.
  double gap{std::numeric_limits<double>::quiet_NaN()};
  /// The gap the count was held to.
  double minimumGap{kDefaultMinimumGap};
  /// How the count was obtained.
  Method method{Method::Unmeasured};
  /// The dense spectrum's residual, or the sparse read's own certificate, met
  /// the tolerance.
  bool converged{false};
  /// The enclosed band is isolated from the rest of the spectrum:
  /// `gap >= minimumGap`. False when the gap is unmeasured, since a band with
  /// nothing outside it is separated from nothing.
  bool separated{false};
  /// `converged` and `separated`: the count is certified by its residual and
  /// by its gap, as the whitepaper certifies every band.
  bool certified{false};
  /// Why the degree is unmeasured; empty otherwise.
  std::string reason{};
};

/// The exact/coexact split of one degree's effective band
/// (`EffectiveTopology::split`).
///
/// The near-kernel of \f$ L_k \f$ is not all holes. Its exact part is
/// \f$ \partial_k^\sharp \f$ of the degree \f$ k-1 \f$ band: at \f$ k = 1 \f$
/// the bridge flows of the effective components, with the eigenvalues of the
/// bottlenecks. Its coexact part is the near-cycles, the band's intersection
/// with \f$ \ker \partial_k \f$: the harmonic chains and the circulations
/// around effectively transparent cells. The effective holes (\f$ k = 1 \f$)
/// and the effective voids (\f$ k = 2 \f$ in three dimensions) are the coexact
/// part,
/// \f[
///   \ker_\epsilon L_k = \partial_k^\sharp \ker_\epsilon L_{k-1} \oplus
///   \{\text{coexact near-cycles}\}.
/// \f]
/// Here "coexact" is the whitepaper's term for everything in the band that is
/// not exact; it includes the harmonic chains.
struct EffectiveHodgeSplit {
  /// The degree's count, with its gap and its certificate.
  EffectiveBettiNumber band{};
  /// The dimension of the exact part, \f$ -1 \f$ when unmeasured.
  int exact{-1};
  /// The dimension of the coexact part, \f$ -1 \f$ when unmeasured. At
  /// \f$ k = 1 \f$ this is the number of effective holes; at \f$ k = 2 \f$ on
  /// a complex of dimension three, the number of effective voids.
  int coexact{-1};
  /// An orthonormal basis of the exact part, as chains (\f$ n_k \times \f$ `exact`).
  Eigen::MatrixXcd exactFrame{};
  /// An orthonormal basis of the coexact part, as chains
  /// (\f$ n_k \times \f$ `coexact`): the band's null space under
  /// \f$ \partial_k^U \f$.
  Eigen::MatrixXcd coexactFrame{};
  /// The singular values of \f$ \partial_k^U Q \f$, descending, with \f$ Q \f$
  /// an orthonormal basis of the band's chains: `exact` of them lie above
  /// `rankTolerance`.
  std::vector<double> boundarySingularValues{};
  /// \f$ \kappa\,\max(n_{k-1}, r)\,\epsilon_m\,\|\partial_k^U\| \f$, the
  /// tolerance that decided the exact rank.
  double rankTolerance{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \varsigma_e / \varsigma_{e+1} \f$ at the exact rank \f$ e \f$:
  /// \f$ +\infty \f$ when either part is empty.
  double splitGap{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \|(I - QQ^\dagger) E\|_2 \f$ for the orthonormal exact frame
  /// \f$ E \f$: how far \f$ \partial_k^\sharp \partial_k \f$ of the band
  /// leaves the band. Zero in exact arithmetic for a flat connection; a
  /// curved one breaks \f$ \partial\partial = 0 \f$ and with it the split.
  double closure{std::numeric_limits<double>::quiet_NaN()};
  /// The relative residual \f$ \|A Z - B Z T\|_F / \|A\|_F \f$ of the band's
  /// invariant subspace \f$ Z \f$ in the pencil \f$ (A, B) \f$.
  double frameResidual{std::numeric_limits<double>::quiet_NaN()};
  /// The band is certified, its frame meets the tolerance, the two parts are
  /// separated by at least the band's minimum gap, and the closure meets the
  /// tolerance.
  bool certified{false};
  /// Why the split is unmeasured; empty otherwise.
  std::string reason{};
};

/// One effective component: the support of one direction of a degree-zero
/// band (`EffectiveTopology::components`).
struct EffectiveComponentSupport {
  /// The vertex that pins this component: its committor is one here and zero
  /// at every other component's pivot.
  std::uint64_t pivot{0};
  /// The vertex ids assigned to this component, ascending.
  std::vector<std::uint64_t> support{};
  /// The committor \f$ q_j \f$ on every vertex, in the canonical vertex order.
  Eigen::VectorXcd committor{};
};

/// The partition of a complex into effective components, read from the
/// degree-zero band (`EffectiveTopology::components`).
struct EffectiveComponentPartition {
  /// The degree-zero count, with its gap and its certificate. The gap is what
  /// certifies each support's isolation from its surroundings.
  EffectiveBettiNumber band{};
  /// One support per direction of the band.
  std::vector<EffectiveComponentSupport> components{};
  /// The 2-norm condition number of the band's rows at the pivots, the
  /// matrix inverted to form the committors.
  double pivotConditioning{std::numeric_limits<double>::quiet_NaN()};
  /// \f$ \max_v |\sum_j q_j(v) - 1| \f$: zero when the constant lies in the
  /// band (the trivial connection), so that the committors are a partition of
  /// unity.
  double partitionDefect{std::numeric_limits<double>::quiet_NaN()};
  /// The band is certified and its pivot rows are invertible.
  bool certified{false};
  /// Why the partition is unmeasured; empty otherwise.
  std::string reason{};
};

/// The share of a region's enclosing surface that must lie in the certified
/// coexact part of the degree-two band before the surface counts as a coexact
/// near-cycle of \f$ L_2 \f$: the enclosing surface, normalized, must keep at
/// least this much of itself inside that subspace
/// (`AntiClusterCertificate::voidContent`). The whitepaper requires the
/// enclosing surface to be a certified coexact near-cycle and fixes no number;
/// this is a declared policy, recorded on every certificate as
/// `AntiClusterCertificate::minimumVoidContent`.
inline constexpr double kDefaultMinimumVoidContent = 0.1;

/// Which way the enclosing surface of a region is cooriented, as the
/// whitepaper's anti-cluster identification requires it to be read
/// (`AntiClusterCertificate::coorientation`).
enum class EnclosingCoorientation {
  /// The coorientation points from the material into the region: the region is
  /// a bubble, which is what the whitepaper requires of an anti-cluster.
  Inward,
  /// The coorientation points from the region out into the material: the
  /// region is a blob, an ordinary cluster.
  Outward,
  /// Neither: the supplied reference is orthogonal to the enclosing surface,
  /// so no direction is read.
  Undeclared,
};

/// Where `AntiClusterCertificate::coorientation` was read from.
enum class CoorientationSource {
  /// From a reference chain the caller supplied, such as the cooriented cut of
  /// a lineage: the coorientation is inward when the reference and the
  /// region's outward enclosing surface point in opposite directions.
  Reference,
  /// From the interior spectrum. With no reference chain the enclosing surface
  /// carries no direction of its own, because the band fixes a complex
  /// near-cycle only up to a nonzero complex scale, and the coorientation is
  /// then read as inward exactly when the enclosed side is the one the
  /// operator's band does not reach.
  InteriorSpectrum,
};

/// The whitepaper's proposed identification of an anti-cluster with an
/// effective void, evaluated on one declared region
/// (`EffectiveTopology::antiCluster`).
///
/// The proposal (Section 10, and falsifier 9) is that an anti-cluster is "a
/// region whose enclosing surface is a certified coexact near-cycle of
/// \f$ L_2 \f$ (of \f$ L_1 \f$ on the dual), whose interior spectrum is nearly
/// empty, and whose enclosing coorientation is inward". Each of those three
/// clauses is one group of fields below, measured on the region the caller
/// declares by its vertices: the enclosing surface, and how much of it lies in
/// the certified coexact part of the degree-two band; the spectrum of the
/// operator restricted to the region; and the coorientation of the enclosing
/// surface.
///
/// A bubble and a blob differ in the last two. A bubble encloses a side the
/// operator's band does not reach, and its surface is cooriented into that
/// side; a blob encloses the band itself, and its surface is cooriented out of
/// it. The certificate holds for a bubble alone.
struct AntiClusterCertificate {
  /// The declared region, as vertex ids, ascending and deduplicated.
  std::vector<std::uint64_t> region{};
  /// The \f$ d \f$-cells of the region: those all of whose vertices lie in
  /// `region`, by canonical index. A region declared around a cavity has none,
  /// since the cavity's cells are not in the complex.
  std::vector<std::size_t> interiorCells{};
  /// The enclosing surface: the \f$ (d-1) \f$-cells all of whose vertices lie
  /// in `region` that are not shared by two of `interiorCells`, by canonical
  /// index.
  std::vector<std::size_t> surface{};
  /// The enclosing surface as a chain, cooriented out of the region:
  /// \f$ \partial_d^U \f$ of the region's cells when it has any, and otherwise
  /// the part of the complex's own boundary that the region's faces carry.
  /// Indexed like \f$ C_{d-1} \f$, and zero off `surface`.
  Eigen::VectorXcd enclosingSurface{};
  /// The degree-two band the void is read from, with its gap and its
  /// certificate.
  EffectiveBettiNumber band{};
  /// The number of effective voids at the scale: the dimension of the coexact
  /// part of the degree-two band (`EffectiveTopology::voids`), \f$ -1 \f$ when
  /// unmeasured.
  int voids{-1};
  /// \f$ \|P\hat\sigma\|_2 \in [0, 1] \f$ for the normalized enclosing surface
  /// \f$ \hat\sigma \f$ and the orthogonal projector \f$ P \f$ onto the
  /// certified coexact part of the degree-two band: how much of the enclosing
  /// surface is a coexact near-cycle of \f$ L_2 \f$. The enclosing surface of
  /// a region of cells is a boundary, and a boundary reaches the coexact
  /// near-kernel only through a void it encloses, so a region that encloses
  /// nothing reads zero here.
  double voidContent{std::numeric_limits<double>::quiet_NaN()};
  /// The share required of `voidContent`.
  double minimumVoidContent{kDefaultMinimumVoidContent};
  /// \f$ \|\partial_{d-1}^U\hat\sigma\|_2 \f$: the enclosing surface is
  /// closed. Zero in exact arithmetic for a flat connection, since
  /// \f$ \partial\partial = 0 \f$; a curved connection breaks it.
  double cycleResidual{std::numeric_limits<double>::quiet_NaN()};
  /// The degree the interior spectrum was read at.
  int interiorDegree{0};
  /// The moduli of the eigenvalues of the operator restricted to the region,
  /// ascending: the pencil of `interiorDegree` on the cells of that degree all
  /// of whose vertices lie in `region`, which is the Dirichlet problem of the
  /// region. Empty when the region carries no cell of that degree.
  std::vector<double> interiorSpectrum{};
  /// How many of `interiorSpectrum` lie inside the window
  /// \f$ |\lambda| \le \epsilon \f$: the number of the operator's own modes
  /// the region holds, \f$ -1 \f$ when unmeasured.
  int interiorRank{-1};
  /// The smallest modulus of `interiorSpectrum`: how far the region's own
  /// spectrum stands above zero. Quiet NaN when the region carries no cell of
  /// the degree.
  double interiorFloor{std::numeric_limits<double>::quiet_NaN()};
  /// `interiorRank` is zero: the region holds none of the operator's modes at
  /// the scale, which is what "nearly empty" asks of a bubble.
  bool interiorEmpty{false};
  /// The coorientation of the enclosing surface.
  EnclosingCoorientation coorientation{EnclosingCoorientation::Undeclared};
  /// Where the coorientation was read from.
  CoorientationSource coorientationSource{CoorientationSource::InteriorSpectrum};
  /// \f$ \mathrm{Re}\,\langle\hat r, \hat\sigma\rangle \f$ for a supplied
  /// reference chain \f$ r \f$ and the normalized enclosing surface: negative
  /// for an inward coorientation. Quiet NaN when no reference was supplied.
  double coorientationOverlap{std::numeric_limits<double>::quiet_NaN()};
  /// The band is certified, the enclosing surface is closed within the
  /// tolerance and keeps at least `minimumVoidContent` of itself in the
  /// certified coexact part of the degree-two band, the interior spectrum is
  /// empty at the scale, and the coorientation is inward.
  bool certified{false};
  /// Why the certificate is unmeasured or does not hold; empty when it holds.
  std::string reason{};
};

/// What `EffectiveTopology::antiCluster` is allowed to assume, beyond the
/// operator, the region and the scale.
struct AntiClusterOptions {
  /// The dense residual, or the sparse certificate's residual, the degree-two
  /// band must meet to be converged, and the tolerance the enclosing surface's
  /// cycle residual must meet.
  double tolerance{1e-10};
  /// The gap the degree-two band must reach to be separated.
  double minimumGap{kDefaultMinimumGap};
  /// The share of the enclosing surface required in the certified coexact part
  /// of the degree-two band.
  double minimumVoidContent{kDefaultMinimumVoidContent};
  /// The degree whose restriction to the region is the interior spectrum.
  /// Degree zero, the default, is the degree whose band carries the effective
  /// components, so a region that holds one of the operator's degree-zero
  /// modes is a cluster rather than a void.
  int interiorDegree{0};
  /// A chain of degree \f$ d - 1 \f$ carrying a coorientation the caller has
  /// established independently, such as the cooriented cut of a lineage. When
  /// it is empty the coorientation is read from the interior spectrum instead;
  /// see `CoorientationSource`.
  Eigen::VectorXcd coorientationReference{};
  /// The modulus below which an overlap with `coorientationReference` reads no
  /// direction at all and the coorientation is `Undeclared`.
  double coorientationTolerance{1e-12};
};

/// # EffectiveTopology
///
/// What a declared operator sees at a scale, as opposed to what the complex is.
///
/// A complex has an actual topology, fixed by its incidence: the Betti numbers
/// of `cobordism::ChainComplex::bettiNumbers`, which the `spacetime::Topology`
/// classes construct by design (`Toroid`, `Sphere`, `PeriodicKuhnGrid`) and
/// which change only when cells are attached. An operator on that complex has
/// an effective topology: the effective Betti number
/// \f[
///   \beta_k^{\mathrm{eff}}(\epsilon) = \mathrm{rank}\, P_{[0,\epsilon]}\bigl(h_k(s, U)\bigr),
/// \f]
/// the rank of the spectral band of the covariant operator of degree
/// \f$ k \f$ near zero, certified by the gap between the enclosed band and the
/// rest of the spectrum. It depends on the squared lengths and on the
/// connection, it can emerge and disappear under relaxation, and it is the
/// count the dynamics sees. The two agree only in a limit: at the trivial
/// connection, under the rank conditions of `chainhodge::ChainHodge`, and as
/// \f$ \epsilon \to 0 \f$. Away from it they part in both directions. Two
/// regions joined through a bottleneck are one component by incidence and two
/// effective components at any scale above the bridge level; a torus carrying
/// a connection with nontrivial holonomy keeps its incidence Betti numbers
/// \f$ (1, 3, 3, 1) \f$ while every effective Betti number of the covariant
/// operator is zero.
///
/// A count is certified only when the band is separated: the smallest modulus
/// outside the window over the largest inside it must reach the declared
/// minimum gap. A window that cuts through the spectrum returns its count
/// uncertified. Persistence across a stated range of scales is the separate
/// certificate of `EffectivePersistence`.
///
/// This class is the effective read and nothing else: it never consults the
/// incidence ranks, so a caller cannot obtain one kind of number where the
/// other was meant. Each degree is counted from the dense complex Schur form of
/// its pencil below the crossover. At or above the crossover only degree zero
/// is available, from the sparse pencil by
/// `chainhodge::SparsePencilSolver::effectiveBetti` with the shift at
/// \f$ -\epsilon \f$ (whose Cholesky factorization certifies that nothing lies
/// below the window); the pencils of higher degrees contain an inverse metric
/// and are reported as unmeasured rather than estimated.
///
/// Beyond the counts it reads three structures of the bands: the split of a
/// degree's band into its exact and coexact parts (`split`), the effective
/// voids (`voids`), and the supports of the effective components
/// (`components`).
class EffectiveTopology {
 public:
  /// Read every degree of \p cov at scale \p epsilon.
  /// @param cov The covariant operator to read.
  /// @param epsilon The scale: eigenvalues with \f$ |\lambda| \le \epsilon \f$
  ///   are counted.
  /// @param tolerance The dense residual, or the sparse certificate's
  ///   residual, a degree must meet to be converged.
  /// @param minimumGap The gap a degree must reach to be separated.
  /// @throws std::invalid_argument when \p epsilon is not positive or
  ///   \p minimumGap is below one.
  [[nodiscard]] static EffectiveTopology read(const chainhodge::CovariantChainHodge &cov, double epsilon,
                                              double tolerance = 1e-10,
                                              double minimumGap = kDefaultMinimumGap);

  /// The exact/coexact split of the degree-\p k band at scale \p epsilon (see
  /// `EffectiveHodgeSplit`). The band is the invariant subspace of the dense
  /// pencil for the eigenvalues with \f$ |\lambda| \le \epsilon \f$, from the
  /// reordered complex Schur form, so that a degenerate or non-normal band
  /// keeps an orthonormal basis. The coexact part is the numerical null space
  /// of \f$ \partial_k^U \f$ on the band's chains; the exact part is
  /// \f$ \partial_k^\sharp \partial_k^U \f$ of the band, with
  /// \f$ \partial_k^\sharp = M_k^U (\partial_k^{U^{-1}})^T (M_{k-1}^U)^{-1} \f$,
  /// which is \f$ \partial_k^\sharp \f$ of the part of the degree
  /// \f$ k-1 \f$ band that is not in its kernel. At degree zero, which has no
  /// boundary, the whole band is coexact, and the degree is read from the
  /// sparse pencil at or above the crossover. Higher degrees at or above the
  /// crossover are unmeasured.
  /// @throws std::invalid_argument for a degree outside \f$ [0, d] \f$, and as `read`.
  [[nodiscard]] static EffectiveHodgeSplit split(const chainhodge::CovariantChainHodge &cov, int degree,
                                                 double epsilon, double tolerance = 1e-10,
                                                 double minimumGap = kDefaultMinimumGap);

  /// The effective voids of a three-dimensional complex: the coexact part of
  /// the degree-two band, the near-cycles of \f$ L_2 \f$ that enclose a
  /// region (equivalently the near-kernel of \f$ L_1 \f$ on the dual
  /// complex). `split(cov, 2, ...)`, with `coexact` the number of voids.
  ///
  /// The whitepaper proposes that an anti-cluster is an effective void; this
  /// reads the void, and says nothing about its lineage.
  /// @throws std::invalid_argument when the complex is not of dimension
  ///   three, and as `read`.
  [[nodiscard]] static EffectiveHodgeSplit voids(const chainhodge::CovariantChainHodge &cov, double epsilon,
                                                 double tolerance = 1e-10,
                                                 double minimumGap = kDefaultMinimumGap);

  /// The effective components at scale \p epsilon: the supports of the
  /// degree-zero band. By the higher-order Cheeger inequality the band has
  /// rank \f$ r \f$ exactly when the complex splits into \f$ r \f$ pieces of
  /// small conductance, and its vectors span the committors of that
  /// metastable decomposition. The committors are recovered from an
  /// orthonormal basis \f$ Q \f$ of the band (as functions: the geometric
  /// images of the degree-zero pencil) by the column-pivoted QR of
  /// \f$ Q^T \f$: its first \f$ r \f$ pivots are the vertices whose rows are
  /// most independent, and \f$ C = Q\,(Q_{\text{pivots},:})^{-1} \f$ is one at
  /// its own pivot and zero at the others. Each vertex belongs to the
  /// component whose committor has the largest modulus there. Read from the
  /// dense Schur form below the crossover and from the sparse pencil at or
  /// above it.
  ///
  /// Reference: Damle, Minden and Ying, "Simple, direct and efficient
  /// multi-way spectral clustering", Information and Inference 8, 2019.
  /// @throws as `read`.
  [[nodiscard]] static EffectiveComponentPartition components(const chainhodge::CovariantChainHodge &cov,
                                                              double epsilon, double tolerance = 1e-10,
                                                              double minimumGap = kDefaultMinimumGap);

  /// The anti-cluster certificate on the region \p regionVertices at scale
  /// \p epsilon: the whitepaper's proposal that an anti-cluster is an
  /// effective void, evaluated clause by clause (see
  /// `AntiClusterCertificate`).
  ///
  /// The region is declared by its vertices, which is how every other support
  /// in this subsystem travels — the support of an effective component, of a
  /// modularity community, of a lineage. From the vertices the read takes the
  /// region's cells (those the vertices contain), its enclosing surface (the
  /// faces the vertices contain that two of those cells do not share) and that
  /// surface as a chain cooriented out of the region. It then measures how
  /// much of the enclosing surface lies in the certified coexact part of the
  /// degree-two band, the spectrum of the operator restricted to the region,
  /// and the coorientation.
  ///
  /// A region declared around a cavity contains no cells, because the cavity's
  /// cells are not in the complex; its enclosing surface is the part of the
  /// complex's own boundary that its faces carry. A region declared on a block
  /// of cells contains those cells and its enclosing surface is their
  /// boundary, which is exact and therefore reaches the coexact near-kernel
  /// only through a void it encloses.
  /// @throws std::invalid_argument when the complex is not of dimension three,
  ///   when the region names a vertex the complex does not have, when
  ///   `AntiClusterOptions::interiorDegree` is outside \f$ [0, d] \f$, when
  ///   `AntiClusterOptions::coorientationReference` is neither empty nor as
  ///   long as \f$ C_{d-1} \f$, and as `read`.
  [[nodiscard]] static AntiClusterCertificate antiCluster(const chainhodge::CovariantChainHodge &cov,
                                                          const std::vector<std::uint64_t> &regionVertices,
                                                          double epsilon,
                                                          const AntiClusterOptions &options = {});

  /// The scale the read was taken at.
  [[nodiscard]] double epsilon() const noexcept { return epsilon_; }
  /// The dimension \f$ d \f$ of the complex.
  [[nodiscard]] int dimension() const noexcept { return static_cast<int>(degrees_.size()) - 1; }
  /// One read per degree, \f$ k = 0 \ldots d \f$.
  [[nodiscard]] const std::vector<EffectiveBettiNumber> &degrees() const noexcept { return degrees_; }
  /// \f$ (\beta_0^{\mathrm{eff}}, \ldots, \beta_d^{\mathrm{eff}}) \f$ with
  /// \f$ -1 \f$ for an unmeasured degree.
  [[nodiscard]] std::vector<int> betti() const;
  /// True when every degree was measured and certified.
  [[nodiscard]] bool certified() const noexcept;

 private:
  friend class EffectivePersistence;
  // Read every degree at each scale, computing each dense degree's Schur form once.
  [[nodiscard]] static std::vector<EffectiveTopology> readScales(const chainhodge::CovariantChainHodge &cov,
                                                                 const std::vector<double> &epsilons,
                                                                 double tolerance, double minimumGap);
  double epsilon_{0.0};
  std::vector<EffectiveBettiNumber> degrees_{};
};

/// A run of consecutive reads over which one degree keeps one certified
/// count (`EffectivePersistence::plateaus`).
struct EffectivePlateau {
  /// The chain degree.
  int degree{0};
  /// The count that persists.
  int rank{-1};
  /// The indices of the first and last reads of the run.
  std::size_t first{0};
  std::size_t last{0};
  /// The scales of the first and last reads.
  double firstEpsilon{std::numeric_limits<double>::quiet_NaN()};
  double lastEpsilon{std::numeric_limits<double>::quiet_NaN()};
  /// The smallest gap over the run.
  double gap{std::numeric_limits<double>::quiet_NaN()};
  /// The number of reads in the run.
  [[nodiscard]] std::size_t length() const noexcept { return last - first + 1; }
};

/// # EffectivePersistence
///
/// An effective count that persists across a stated range of scales.
///
/// The whitepaper certifies an effective component, like every fiber, by the
/// gap between its band and the rest of the spectrum and by persistence: the
/// count must stay the same, certified, across the range the caller states.
/// This class holds a sequence of effective reads — one operator swept over
/// scales \f$ \epsilon \f$, or a sequence of operators (a refinement or a
/// relaxation) read at one scale — and reports, per degree, the runs of
/// consecutive certified reads that share one count. A degree persists when
/// the whole sequence is one such run.
class EffectivePersistence {
 public:
  /// Read \p cov at every scale of \p epsilons, in the order given. Each
  /// degree's dense spectrum is computed once and read at every scale.
  /// @throws std::invalid_argument for an empty sweep, and as
  ///   `EffectiveTopology::read` for any scale.
  [[nodiscard]] static EffectivePersistence sweep(const chainhodge::CovariantChainHodge &cov,
                                                  const std::vector<double> &epsilons,
                                                  double tolerance = 1e-10,
                                                  double minimumGap = kDefaultMinimumGap);

  /// A sequence of reads taken by the caller.
  /// @throws std::invalid_argument for an empty sequence or reads of
  ///   different dimensions.
  explicit EffectivePersistence(std::vector<EffectiveTopology> reads);

  /// The reads, in sequence order.
  [[nodiscard]] const std::vector<EffectiveTopology> &reads() const noexcept { return reads_; }
  /// The dimension of the complex.
  [[nodiscard]] int dimension() const noexcept { return reads_.front().dimension(); }
  /// The maximal runs of consecutive reads that are certified at degree \p k
  /// and share one count, in sequence order.
  /// @throws std::invalid_argument for a degree outside \f$ [0, d] \f$.
  [[nodiscard]] std::vector<EffectivePlateau> plateaus(int k) const;
  /// The count that persists at degree \p k across every read, \f$ -1 \f$
  /// when none does.
  [[nodiscard]] int persistentRank(int k) const;
  /// The persistent count of every degree, \f$ -1 \f$ where none persists.
  [[nodiscard]] std::vector<int> betti() const;
  /// True when every degree persists.
  [[nodiscard]] bool certified() const;

 private:
  std::vector<EffectiveTopology> reads_;
};

/// The verdict of an `EffectiveSignature` on an `EffectiveTopology`, or on an
/// `EffectivePersistence` across its range of scales.
struct EffectiveSignatureCertificate {
  /// The signature's name, such as "effective 3-torus".
  std::string signature{};
  /// The scale of the read that was judged; the first scale of a persistence
  /// range.
  double epsilon{0.0};
  /// Every scale the verdict covers, in sequence order.
  std::vector<double> scales{};
  /// The Betti numbers the signature requires, \f$ -1 \f$ where it requires
  /// nothing.
  std::vector<int> expected{};
  /// The effective Betti numbers that were read, \f$ -1 \f$ where unmeasured
  /// (or, across a range, where no count persists).
  std::vector<int> measured{};
  /// Every required degree was measured and equals its required value.
  bool matches{false};
  /// Every required degree's read was certified, residual and gap, at every
  /// scale covered.
  bool certified{false};
  /// The smallest gap among the required degrees over the scales covered:
  /// how far the recognized bands stand from the rest of the spectrum.
  double gap{std::numeric_limits<double>::quiet_NaN()};

  /// `matches` and `certified` together.
  [[nodiscard]] bool holds() const noexcept { return matches && certified; }
};

/// # EffectiveSignature
///
/// A named pattern of effective Betti numbers, recognized in an operator
/// rather than built into a complex: the effective counterpart of a
/// `spacetime::Topology`. `spacetime::Toroid` constructs a complex whose
/// incidence is a torus; `EffectiveTorus` certifies that an operator, on
/// whatever complex, has the near-kernel of one.
///
/// The claim is one of rank. Betti numbers do not determine a space up to
/// homeomorphism, and an effective signature says nothing about incidence, so
/// a certificate that holds means that the operator's spectral bands near zero
/// have the ranks of the named space at the stated scale, with the stated gap,
/// and nothing more.
class EffectiveSignature {
 public:
  virtual ~EffectiveSignature() = default;
  /// The signature's name, as it appears on its certificates.
  [[nodiscard]] virtual std::string name() const = 0;
  /// The required \f$ (\beta_0, \ldots, \beta_d) \f$, \f$ -1 \f$ where nothing
  /// is required.
  [[nodiscard]] virtual std::vector<int> betti() const = 0;

  /// Compare an effective read with this signature.
  /// @throws std::invalid_argument when the read has a different dimension.
  [[nodiscard]] EffectiveSignatureCertificate certify(const EffectiveTopology &topology) const;
  /// `certify(EffectiveTopology::read(cov, epsilon, tolerance, minimumGap))`.
  [[nodiscard]] EffectiveSignatureCertificate certify(const chainhodge::CovariantChainHodge &cov, double epsilon,
                                                      double tolerance = 1e-10,
                                                      double minimumGap = kDefaultMinimumGap) const;
  /// Compare a persistence range with this signature: the required counts
  /// must persist, certified, across every read.
  /// @throws std::invalid_argument when the reads have a different dimension.
  [[nodiscard]] EffectiveSignatureCertificate certify(const EffectivePersistence &persistence) const;
};

/// The effective \f$ d \f$-torus: \f$ \beta_k^{\mathrm{eff}} = \binom{d}{k} \f$.
class EffectiveTorus : public EffectiveSignature {
 public:
  /// @throws std::invalid_argument for \f$ d < 1 \f$.
  explicit EffectiveTorus(int dimension);
  /// See `EffectiveSignature::name`.
  [[nodiscard]] std::string name() const override;
  /// See `EffectiveSignature::betti`.
  [[nodiscard]] std::vector<int> betti() const override;

 private:
  int dimension_;
};

/// The effective \f$ d \f$-sphere: \f$ \beta_0^{\mathrm{eff}} =
/// \beta_d^{\mathrm{eff}} = 1 \f$ and zero between.
class EffectiveSphere : public EffectiveSignature {
 public:
  /// @throws std::invalid_argument for \f$ d < 1 \f$.
  explicit EffectiveSphere(int dimension);
  /// See `EffectiveSignature::name`.
  [[nodiscard]] std::string name() const override;
  /// See `EffectiveSignature::betti`.
  [[nodiscard]] std::vector<int> betti() const override;

 private:
  int dimension_;
};

/// \f$ n \f$ effective components: \f$ \beta_0^{\mathrm{eff}} = n \f$ on a
/// complex of dimension \f$ d \f$, with no requirement on the higher degrees.
/// This is the signature a mesh above the dense crossover can be held to,
/// since degree zero is the one read from the sparse pencil.
class EffectiveComponents : public EffectiveSignature {
 public:
  /// @throws std::invalid_argument for \f$ n < 0 \f$ or \f$ d < 0 \f$.
  EffectiveComponents(int count, int dimension);
  /// See `EffectiveSignature::name`.
  [[nodiscard]] std::string name() const override;
  /// See `EffectiveSignature::betti`.
  [[nodiscard]] std::vector<int> betti() const override;

 private:
  int count_;
  int dimension_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_EFFECTIVETOPOLOGY_H
