// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_PENCILLAYER_H
#define TESSERA_COBORDISM_PENCILLAYER_H

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/PencilSchur.h"
#include "chainhodge/RieszBand.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }

namespace tessera::cobordism {
using ::tessera::spacetime::Spacetime;
using chainhodge::Complex;

/// A retained fiber on a block's degree-\f$ k \f$ cells, read from a certified
/// Riesz band of the cobordism that carries it. It is the interface coordinates
/// the next level retains.
struct BoundaryFiber {
  int degree{1};
  /// The block's degree-\f$ k \f$ cells (sorted vertex tuples), the fiber's rows.
  std::vector<std::vector<std::uint64_t>> cells{};
  /// \f$ Z_B \f$: the band's geometric images restricted to `cells`
  /// (\f$ |cells| \times r \f$).
  Eigen::MatrixXcd images{};
  /// \f$ Z_B^\vee \f$: the dual band's images on the same cells (equals
  /// `images` at \f$ U = 1 \f$).
  Eigen::MatrixXcd dualImages{};
  /// \f$ \mathcal G = Z_B^T M_{BB} Z_B \f$: the chain metric on the fiber's own
  /// cells.
  Eigen::MatrixXcd gram{};
  /// \f$ Z^T M Z \f$ of the whole-complex band the fiber was cut from.
  Eigen::MatrixXcd fullGram{};
  /// The band's eigenvalue (mean of the reduced operator's eigenvalues).
  Complex eigenvalue{0.0, 0.0};
  chainhodge::Contour contour{};
  chainhodge::BandCertificate certificate{};
  /// The Lorentzian-protocol rotation the geometry was computed at (NaN unset).
  double epsilon{std::numeric_limits<double>::quiet_NaN()};
  [[nodiscard]] int rank() const noexcept { return static_cast<int>(images.cols()); }
};

/// A glued pencil: the union of several cobordisms' top cells under one
/// geometry, assembled per top simplex.
struct AssembledPencil {
  /// The union complex, owned through a stable pointer that the connection and
  /// the dressed operators refer to, so the record may be copied or moved.
  std::shared_ptr<const ChainComplex> complexPtr{};
  chainhodge::SquaredLengths lengths{};
  std::shared_ptr<chainhodge::ChainHodge> base{};
  std::shared_ptr<chainhodge::CovariantChainHodge> op{};
  std::shared_ptr<chainhodge::CovariantChainHodge> dual{};
  /// The one rotation shared by every piece (NaN when none declared one).
  double epsilon{std::numeric_limits<double>::quiet_NaN()};
  /// Each piece's top cells (sorted tuples) in the order supplied.
  std::vector<std::vector<std::vector<std::uint64_t>>> pieces{};
  /// Cells (per degree) belonging to more than one piece: the shared interface
  /// on which the pieces' contributions add.
  std::vector<std::vector<std::vector<std::uint64_t>>> sharedCells{};
  [[nodiscard]] const ChainComplex &complex() const { return *complexPtr; }
  [[nodiscard]] int dimension() const noexcept { return complexPtr ? complexPtr->dimension() : -1; }
  /// Canonical index of a degree-\p k cell, or -1 when absent.
  [[nodiscard]] int cellIndex(int k, const std::vector<std::uint64_t> &cell) const;
};

/// The bordered form of the degree-\f$ k \f$ pencil at a shift (the system
/// `CovariantChainHodge::resolvent` factorizes): coordinates are the
/// degree-\f$ k \f$ cells followed by the degree-\f$ (k-1) \f$ cells, and the
/// Schur complement over the lower block is \f$ \lambda M_k^U - \tilde A_k^U \f$.
/// Assembled per top simplex, so on shared cells a glued complex's bordered
/// form is the sum of the pieces'; the dense \f$ \tilde A_k \f$ is not, since it
/// contains \f$ M_{k-1}^{-1} \f$.
struct BorderedPencil {
  int degree{1};
  Complex lambda{0.0, 0.0};
  int upperCount{0};   // n_k
  int lowerCount{0};   // n_{k-1}
  Eigen::MatrixXcd matrix{};
};

/// A pencil level whose interface coordinates are retained fibers: the Feshbach
/// reduction of the assembled pencil onto the fibers' cells, restricted to the
/// fibers.
struct FiberLevel {
  int degree{1};
  Complex lambda{0.0, 0.0};
  /// Canonical indices of the interface cells (the retained fibers' cells) and
  /// of the eliminated interior (the further cobordism's bulk).
  std::vector<int> interfaceCells{};
  std::vector<int> interiorCells{};
  /// \f$ F_B(\lambda) \f$ over the interface cells with its certificates.
  chainhodge::FeshbachResult response{};
  /// \f$ J \f$: the fibers placed on their cells (\f$ |B| \times R \f$), \f$ R = \sum_a r_a \f$.
  Eigen::MatrixXcd J{};
  /// \f$ \tilde J = Z \f$: the dual fibers on their cells.
  Eigen::MatrixXcd Jdual{};
  /// \f$ (\hat A_{\ell+1}, \mathcal G_{\ell+1}) = (J^T F_B J,\ J^T M_{BB} J) \f$:
  /// the level pencil and chain metric on the fibers' cells. The Gram's
  /// off-diagonal blocks vanish unless two fibers' cells share a top simplex.
  chainhodge::FiberRestriction restriction{};
  /// \f$ J^T (T^T M T) J \f$: the Gram through the Feshbach constraint modes,
  /// which couples fibers through the eliminated bulk.
  Eigen::MatrixXcd constraintGram{};
  std::vector<int> blockOffsets{};
  std::vector<int> blockRanks{};
  /// Whether every pair of distinct fibers neither shares a cell nor has cells
  /// in a common top simplex; then `restriction.gram` is exactly
  /// block-diagonal. Overlapping fibers are carried as a labeled sum, with
  /// \f$ J \f$ and \f$ \mathcal G \f$ exact.
  bool fibersDisjoint{true};
};

/// # PencilLayer
///
/// Continuation of a relaxed cobordism's boundary fibers into the next pencil
/// level, composed from existing primitives:
///
/// * **Assembly.** `assemble` glues cobordisms by taking the union of their top
///   cells. `WhitneyMass` assembles every \f$ M_k \f$ per top simplex, so the
///   glued pencil is the sum of the pieces' on shared cells and the direct sum
///   elsewhere.
/// * **Boundary response.** `boundaryResponse` is the Feshbach complement onto
///   a set of cells; `composeResponses` is the star product of two responses
///   along shared cells.
/// * **Fibers.** `readBoundaryFiber` restricts a certified Riesz band to a
///   block's cells.
/// * **Levels.** `level` is the Feshbach reduction onto the retained fibers'
///   cells, restricted to the fibers, carrying \f$ J \f$,
///   \f$ \tilde J = Z \f$ and \f$ \mathcal G = J^T M_{BB} J \f$ exactly.
/// * **Transfer.** `transfer` between two retained fibers.
///
/// Every pairing is the transpose; no conjugation enters. Dense below the
/// crossover of the underlying `ChainHodge`.
/// Which band of a pencil a fiber is fitted in when its residual, or the
/// residual's gradient, is read on that pencil.
/// A marking of a boundary surface, in host vertex ids: cycles, each a closed
/// walk of directed steps \f$ (u \to v) \f$ over edges of the host. A step
/// contributes \f$ +h(u,v) \f$ to a period when \f$ u < v \f$ (the
/// ascending-id reference orientation of `ChainComplex`) and \f$ -h(u,v) \f$
/// otherwise — the `Edge::walkLoop` convention. A `SimplicialQubit` cycle
/// (edge index, sign) becomes a step by taking the edge's \f$ (i, j) \f$ as
/// \f$ (i \to j) \f$ for \f$ +1 \f$ and \f$ (j \to i) \f$ for \f$ -1 \f$,
/// mapped through `SurfaceSeed::vertexIds`.
using Marking = std::vector<chainhodge::Connection::Walk>;

enum class FiberBand {
  /// The contour stored on the fiber when it names one, else the lowest
  /// band above the zero mode (`PencilLayer::bandContour(…, 1)`): the read
  /// of a whole-complex target and of an ordinary block's sub-complex.
  AsStored,
  /// The zero mode of the pencil the fiber is read on — the harmonic contour
  /// of that pencil (`PencilLayer::harmonicContour`), whatever contour the
  /// fiber stores. This is a surface block's own Laplacian.
  ZeroMode,
};

class PencilLayer {
 public:
  using Cell = std::vector<std::uint64_t>;

  /// Glue \p pieces (each with its declared \p epsilons entry; NaN = none).
  /// @throws std::invalid_argument on a shared edge whose squared length or
  ///   link differs between pieces, or on differing \f$ \varepsilon \f$.
  [[nodiscard]] static AssembledPencil assemble(
      const std::vector<std::shared_ptr<Spacetime>> &pieces,
      const std::vector<double> &epsilons = {},
      chainhodge::Branch branch = chainhodge::Branch::Continuation,
      int crossoverDimension = std::numeric_limits<int>::max());

  /// \f$ \max |M_k(\text{union}) - \sum_i \text{scatter}(M_k(\text{piece}_i))| \f$:
  /// the assembly identity, exactly zero up to round-off.
  [[nodiscard]] static double assemblyResidual(const AssembledPencil &assembled, int k,
                                               chainhodge::Branch branch =
                                                   chainhodge::Branch::Continuation);

  /// Canonical indices of the degree-\p k cells lying inside a vertex set.
  [[nodiscard]] static std::vector<int> cellsWithin(const AssembledPencil &assembled, int k,
                                                    const std::vector<std::uint64_t> &vertices);
  /// Canonical indices of explicit cells.
  [[nodiscard]] static std::vector<int> indicesOf(const AssembledPencil &assembled, int k,
                                                  const std::vector<Cell> &cells);

  /// \f$ F_B(\lambda) \f$ of the assembled pencil onto \p interface (canonical indices).
  [[nodiscard]] static chainhodge::FeshbachResult boundaryResponse(
      const AssembledPencil &assembled, int k, const std::vector<int> &interface,
      Complex lambda);

  /// The bordered pencil at \p lambda (see `BorderedPencil`); \p k must be
  /// at least one.
  [[nodiscard]] static BorderedPencil borderedPencil(const AssembledPencil &assembled, int k,
                                                     Complex lambda);
  /// The Feshbach complement of the bordered pencil onto boundary cells of
  /// degree \p k (\p upperInterface) and \p k−1 (\p lowerInterface), both as
  /// canonical indices of their degree. Result coordinates are the bordered ones
  /// (degree-\p k index, then \p upperCount + degree-\p (k−1) index), ascending.
  /// Two glued cobordisms' bordered responses compose under `composeResponses`
  /// to the assembled complex's.
  [[nodiscard]] static chainhodge::FeshbachResult borderedResponse(
      const AssembledPencil &assembled, int k, const std::vector<int> &upperInterface,
      const std::vector<int> &lowerInterface, Complex lambda);
  /// Eliminate the lower (degree-\p k−1) coordinates of a bordered response,
  /// leaving the response over its degree-\p k cells: equals
  /// \f$ -F_B(\lambda) \f$ of `boundaryResponse` on the same cells.
  [[nodiscard]] static Eigen::MatrixXcd upperResponse(const chainhodge::FeshbachResult &bordered,
                                                      int upperCount);

  /// The star product of two boundary responses along their shared cells:
  /// \p left is over \p leftCells and \p right over \p rightCells, both
  /// canonical indices in one assembled complex; the cells in both lists are
  /// summed and eliminated. Returns the response over the remaining cells in
  /// ascending canonical order.
  [[nodiscard]] static Eigen::MatrixXcd composeResponses(const Eigen::MatrixXcd &left,
                                                         const std::vector<int> &leftCells,
                                                         const Eigen::MatrixXcd &right,
                                                         const std::vector<int> &rightCells);

  /// A circle around the \p bandIndex-th distinct eigenvalue cluster of the
  /// degree-\p k pencil, clusters ordered by modulus (index 0 is the smallest,
  /// the harmonic cluster when zero is an eigenvalue). The radius is a quarter
  /// of the distance to the nearest other cluster, so the quadrature's leak from
  /// it is \f$ 3^{-n} \f$. Two eigenvalues belong to one cluster when they
  /// differ by less than \f$ 10^{-9} \f$ of the spectral scale.
  /// @throws std::invalid_argument when the index exceeds the cluster count.
  [[nodiscard]] static chainhodge::Contour bandContour(const AssembledPencil &assembled, int k,
                                                       int bandIndex, int nodeCount = 64);

  /// A circle around zero of radius half the smallest nonzero eigenvalue
  /// modulus of the degree-\p k pencil.
  [[nodiscard]] static chainhodge::Contour harmonicContour(const AssembledPencil &assembled,
                                                           int k, int nodeCount = 64);

  /// The monodromy of a drawn cobordism between two marked surfaces. The whole
  /// complex's degree-1 zero mode — the harmonic band of the chain-level
  /// Whitney pencil on `PencilLayer::harmonicContour`, bulk and boundary edges
  /// in one operator — is read on the edges of both markings, giving periods
  /// \f$ P_A \f$ (\f$ |A| \times r \f$) and \f$ P_B \f$ and the matrix
  /// \f$ M \f$ with \f$ P_B = M P_A \f$: \f$ M = P_B P_A^{-1} \f$ for a rank-2
  /// zero mode, the least-squares fit otherwise.
  ///
  /// That restriction is topological, so \f$ M \f$ is rational, and integral
  /// when the restriction to \f$ T_A \f$ is unimodular — an element of
  /// \f$ SL(2,\mathbb Z) \f$ when \f$ W \f$ is an \f$ I \f$-bundle. A read
  /// that cannot be made (no zero mode, a marking edge absent from the whole, a
  /// rank-deficient \f$ P_A \f$, or a pencil that refuses the geometry) reports
  /// an `obstruction` instead.
  struct MonodromyRead {
    /// Betti numbers of the whole complex.
    std::vector<int> betti;
    /// \f$ r = \dim \ker L_1 \f$ of the whole, the zero mode's rank.
    int harmonicRank{0};
    /// \f$ P_A \f$, \f$ |A| \times r \f$.
    Eigen::MatrixXcd periodsA;
    /// \f$ P_B \f$, \f$ |B| \times r \f$.
    Eigen::MatrixXcd periodsB;
    /// \f$ M \f$ (\f$ |B| \times |A| \f$); empty when obstructed before the fit.
    Eigen::MatrixXcd monodromy;
    /// \f$ \operatorname{round}(\operatorname{Re} M) \f$, row-major.
    std::vector<std::vector<long>> rounded;
    /// \f$ \max_{ij} |M_{ij} - \operatorname{round}(M_{ij})| \f$ (NaN when obstructed).
    double roundingResidual{std::numeric_limits<double>::quiet_NaN()};
    /// \f$ \|M P_A - P_B\|_F / \|P_B\|_F \f$ (zero to rounding when the relation is exact).
    double fitResidual{std::numeric_limits<double>::quiet_NaN()};
    /// Empty when the read succeeded; otherwise why it could not be made.
    std::string obstruction;
  };
  /// The monodromy read of \p spacetime between \p markingA and \p markingB.
  /// Read-only on the geometry; the metric source is the Whitney pencil.
  [[nodiscard]] static MonodromyRead monodromy(const std::shared_ptr<Spacetime> &spacetime,
                                               const Marking &markingA, const Marking &markingB);

  /// The whole complex's degree-1 zero mode read on every marking at once: one
  /// harmonic basis \f$ Z \f$ and its periods over each marking, all from that
  /// same basis. `monodromy` is the two-marking case with the fit
  /// \f$ P_B = M P_A \f$ added. Comparisons across markings need one basis,
  /// since two separate reads may return the zero mode in different ones.
  struct RestrictionRead {
    std::vector<int> betti;
    /// \f$ r = \dim \ker L_1 \f$ of the whole, the zero mode's rank.
    int harmonicRank{0};
    /// \f$ Z \f$, \f$ n_1 \times r \f$, in the assembled pencil's degree-1 cell
    /// order (`AssembledPencil::cellIndex(1, edge)` maps a sorted edge to its row).
    Eigen::MatrixXcd images;
    /// \f$ P_i \f$ (\f$ |\text{marking}_i| \times r \f$): the transported periods
    /// of `images` over marking \f$ i \f$, in the order the markings were given.
    std::vector<Eigen::MatrixXcd> periods;
    /// \f$ \Phi \f$, the band's frame on chains, \f$ n_1 \times r \f$, with
    /// `images` \f$ = G_1^U \Phi \f$. Co-closedness is a statement about
    /// \f$ \Phi \f$ (\f$ \partial_1^U \Phi = 0 \f$) and closedness one about
    /// `images` (\f$ (\partial_2^{U^{-1}})^T Z = 0 \f$), so certifying the band
    /// as harmonic needs both representations.
    Eigen::MatrixXcd frame;
    /// The band's certificate: contour node count, projector idempotency, rank
    /// and its tolerance, singular gap, resolvent bound. Rank alone does not
    /// make a spectral band the harmonic kernel.
    chainhodge::BandCertificate certificate;
    /// Empty when the read succeeded; otherwise why it could not be made.
    std::string obstruction;
  };
  /// The restriction read of \p spacetime over \p markings (see
  /// `RestrictionRead`). Read-only on the geometry; the Whitney pencil is the
  /// metric source by construction. Every marking edge must be an edge of the
  /// whole, and every marking must order into closed walks from one base
  /// point, or the read names the obstruction and returns nothing else.
  [[nodiscard]] static RestrictionRead restriction(const std::shared_ptr<Spacetime> &spacetime,
                                                   const std::vector<Marking> &markings);

  /// The least-squares leak of \p target's images in the band read on
  /// \p spacetime's pencil at the target's cells, relative to the target's
  /// norm: 0 when the band carries the target, 1 when nothing does. The band is
  /// chosen by \p band (`fiberContour`). A geometry the pencil refuses (a
  /// singular dressed metric, a branch or allowability failure, a cell outside
  /// the complex), or no complex at all, scores as the full leak 1.
  /// @throws std::logic_error for a target with no images or a zero target.
  [[nodiscard]] static double fiberResidual(const std::shared_ptr<Spacetime> &spacetime,
                                            const BoundaryFiber &target,
                                            FiberBand band = FiberBand::AsStored);
  /// The contour a fiber is read at on \p assembled under \p band: the
  /// harmonic contour of \p assembled for `ZeroMode`; the fiber's stored
  /// contour, else the lowest band above the zero mode, for `AsStored`.
  [[nodiscard]] static chainhodge::Contour fiberContour(const AssembledPencil &assembled,
                                                          const BoundaryFiber &fiber, FiberBand band);
  /// The dual of a frame under the transpose pairing of \p complex's own
  /// chain-level Whitney pencil at degree \p degree. With \f$ M_k \f$ the
  /// pencil's chain-metric inverse (`CovariantChainHodge::Minv`, the Whitney
  /// mass matrix) restricted to \p cells, and \f$ B = Z^T M_k Z \f$ the frame's
  /// pairing, the dual is \f$ Z^\vee = Z\,B^{-T} \f$, so that
  /// \f$ (Z^\vee)^T M_k Z = I \f$ as `BlockFrame` requires.
  ///
  /// `chainhodge::PencilSchur::transfer` normalizes nothing, so a frame paired
  /// with its plain images gives the bilinear form \f$ Z_A^T \tilde A Z_B \f$,
  /// which transforms by \f$ g_A^T T g_B \f$. Under this dual it instead gives
  /// the matrix of the operator block in the frames' coordinates,
  /// \f$ g_A^{-1} T g_B \f$, which is what a target written in those
  /// coordinates compares with. Read-only on \p complex.
  /// @throws std::invalid_argument on a null complex, a degree above its
  ///   dimension, a frame without columns, a row count other than the cell
  ///   count, or a cell absent from the complex at that degree
  ///   (`PencilLayer::indicesOf`, by name); std::runtime_error when the
  ///   pairing is singular (an isotropic frame).
  [[nodiscard]] static Eigen::MatrixXcd dualFrame(const std::shared_ptr<Spacetime> &complex, int degree,
                                                  const std::vector<std::vector<std::uint64_t>> &cells,
                                                  const Eigen::MatrixXcd &images);
  /// Read the fiber form of a block's target: the band of \p contour on the
  /// assembled pencil, restricted to \p cells.
  [[nodiscard]] static BoundaryFiber readBoundaryFiber(const AssembledPencil &assembled, int k,
                                                       const chainhodge::Contour &contour,
                                                       const std::vector<Cell> &cells,
                                                       double kappa = 10.0);

  /// The next pencil level over \p assembled with \p retained fibers as the
  /// interface coordinates at shift \p lambda. Fibers may overlap on cells (the
  /// labeled sum); the Gram then carries the overlap block exactly.
  /// @throws std::invalid_argument when a fiber's cell is not a cell of the
  ///   assembled complex.
  [[nodiscard]] static FiberLevel level(const AssembledPencil &assembled, int k,
                                        const std::vector<BoundaryFiber> &retained,
                                        Complex lambda);

  /// The transfer between two retained fibers on the assembled pencil, with the
  /// reversal identity asserted (`PencilSchur::transfer`).
  [[nodiscard]] static chainhodge::TransferResult transfer(const AssembledPencil &assembled, int k,
                                                           const BoundaryFiber &A,
                                                           const BoundaryFiber &B,
                                                           double tolerance = 1e-8);

  /// The pencil \f$ (\tilde A_k^U, M_k^U) \f$ of the assembled complex, dense.
  [[nodiscard]] static chainhodge::Pencil pencil(const AssembledPencil &assembled, int k);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_PENCILLAYER_H
