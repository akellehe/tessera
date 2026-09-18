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
