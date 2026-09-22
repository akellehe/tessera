// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_MESH_REGGECONTINUATION_H
#define TESSERA_MESH_REGGECONTINUATION_H

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "mesh/RiemannSheet.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }

namespace tessera::mesh {

/// A cell, hinge or edge named by its sorted vertex ids — a label that survives
/// a cache refill and a re-read of the triangulation, which a `Simplex*` does
/// not.
using SimplexKey = std::vector<std::uint64_t>;

/// # ReggeContinuation
///
/// The declared Riemann sheet of every root a complex Regge evaluation cannot
/// avoid, carried along a path of geometries with its monodromy.
///
/// A complex Regge action is a function of the complex squared edge lengths
/// \f$ s_e \f$. Written through squared volumes it is single-valued and
/// polynomial; written as an action it is not, because three operations on the
/// way from \f$ s \f$ to \f$ S \f$ are branched:
///
/// - the content of a cell, \f$ |T| = \sqrt{\det G_T}/d! \f$, branched where
///   \f$ \det G_T = 0 \f$;
/// - the two roots in the dihedral cosine
///   \f$ \cos\theta = -C_{ij}/(\sqrt{C_{ii}}\sqrt{C_{jj}}) \f$, branched where a
///   Cayley-Menger cofactor vanishes;
/// - the inverse cosine \f$ \theta = \arccos(\cos\theta) \f$, branched at
///   \f$ \cos\theta = \pm 1 \f$ and at infinity.
///
/// Evaluated at a single geometry each of those is given a principal value and
/// the result is a number. Evaluated along a path — a relaxation, a Lorentzian
/// rotation, a continuation in a squared length — the principal value is
/// discontinuous wherever the path crosses a cut, and the action jumps at a
/// place where nothing happened to the geometry. This class holds, instead, a
/// declared sheet per root: `SheetedSqrt` for the contents and the cofactor
/// roots, `SheetedAcos` for the angles. `advance` moves every label by
/// continuity from the geometry the labels currently describe to the geometry
/// the mesh now holds, and `action` reads the action off the declared sheets.
/// The labels are the monodromy: after a loop of squared lengths around a
/// branch point the geometry is the one it started at and the labels are not,
/// which is the statement that the root came back on the other sheet.
///
/// The edge roots are not held here. An edge owns which of \f$ \pm\sqrt{s_e} \f$
/// it is, and carries that sheet itself through `Edge::continueLength`; this
/// class carries the sheets of the quantities built from the edges. The usual
/// step of a path is therefore: `continueLength` every edge that moves, then
/// `advance` this.
///
/// Declared over a fixed triangulation. A Pachner move invalidates it — the
/// cells and hinges it holds labels for are gone, and a label for a cell that
/// no longer exists is not a continuation of anything — so a caller that
/// changes the topology declares a new continuation after the move.
class ReggeContinuation {
 public:
  /// Declare every root of \a spacetime's current geometry on its principal
  /// sheet. At this point `volume`, `dihedralAngle`, `deficitAngle` and
  /// `action` reproduce the sheet-blind `Simplex` values; they part company
  /// only once a path has been walked.
  explicit ReggeContinuation(std::shared_ptr<spacetime::Spacetime> spacetime);

  /// Continue every label from the geometry it describes to the geometry the
  /// mesh now holds.
  ///
  /// Each step must turn each radicand by less than half a turn about its branch
  /// point, since a rotation by \f$ \pi + \delta \f$ and one by
  /// \f$ \delta - \pi \f$ have the same endpoint and no continuation can tell
  /// them apart. `maxRadicandTurn` reports the largest turn the last step made,
  /// so a caller refines its sampling on evidence rather than on faith.
  void advance();

  /// The top cells carrying labels, as sorted vertex-id tuples.
  [[nodiscard]] std::vector<SimplexKey> cells() const;
  /// The hinges carrying labels: the \f$ (d-2) \f$-faces with at least one top
  /// coface, the same set the Regge action sums over.
  [[nodiscard]] std::vector<SimplexKey> hinges() const;

  /// \f$ (-1)^w \sqrt{\det G_T}/d! \f$ — the content of a top cell on its
  /// declared sheet.
  [[nodiscard]] std::complex<double> volume(const SimplexKey &cell) const;
  /// \f$ w \bmod 2 \f$ for that content: 0 principal, 1 the other sheet.
  [[nodiscard]] int volumeSheet(const SimplexKey &cell) const;
  /// The accumulated monodromy \f$ w \f$ of \f$ \det G_T \f$ about zero.
  [[nodiscard]] int volumeWinding(const SimplexKey &cell) const;

  /// The content of a hinge on its declared sheet: \f$ (-1)^w \sqrt{\det G_h}/k! \f$
  /// for a \f$ k \f$-dimensional hinge, and 1 for a hinge that is a point, whose
  /// content is a count and carries no root.
  [[nodiscard]] std::complex<double> hingeContent(const SimplexKey &hinge) const;
  /// \f$ w \bmod 2 \f$ for that content; 0 for a point hinge.
  [[nodiscard]] int hingeContentSheet(const SimplexKey &hinge) const;

  /// The dihedral angle at \a hinge within \a cell on its declared sheets — both
  /// cofactor roots and the inverse cosine.
  [[nodiscard]] std::complex<double> dihedralAngle(const SimplexKey &cell,
                                                   const SimplexKey &hinge) const;
  /// The integer \f$ k \f$ of the angle's declared arccosine sheet.
  [[nodiscard]] int angleBranchIndex(const SimplexKey &cell,
                                     const SimplexKey &hinge) const;
  /// The sign \f$ \varepsilon \f$ of the angle's declared arccosine sheet.
  [[nodiscard]] int angleOrientation(const SimplexKey &cell,
                                     const SimplexKey &hinge) const;
  /// The sheets of the two cofactor roots \f$ \sqrt{C_{ii}} \f$,
  /// \f$ \sqrt{C_{jj}} \f$ the angle's cosine is divided by.
  [[nodiscard]] std::pair<int, int> angleCofactorSheets(const SimplexKey &cell,
                                                        const SimplexKey &hinge) const;

  /// \f$ 2\pi - \sum_T \theta_T \f$ over the top cells at \a hinge, every angle on
  /// its declared sheet.
  [[nodiscard]] std::complex<double> deficitAngle(const SimplexKey &hinge) const;

  /// \f$ S = \sum_h |h|\,\varepsilon_h \f$ on the declared sheets: the Regge action
  /// continued along the path walked so far. Continuous along any path the
  /// sampling resolves, including one that crosses a principal cut.
  ///
  /// The primal form, with \f$ |h| \f$ the hinge's own content — in four
  /// dimensions the hinge area, equal to `simulations::ReggeSolver::reggeAction`.
  /// The circumcentric dual form `ReggeSolver::dualReggeAction` weights each
  /// hinge by \f$ |{\star}h| \f$ instead, which is built from a further family of
  /// roots (the circumcentric heights \f$ \lambda_v \sqrt{\det G_{\rm coface} /
  /// \det G_{\rm face}} \f$), and those carry no labels here: continuing the dual
  /// action means declaring a sheet per height as well, which this class does not
  /// do.
  [[nodiscard]] std::complex<double> action() const;

  /// The same sum with every root and inverse cosine taken principal — the
  /// sheet-blind value, carried so that a caller can see the jump the
  /// declaration removes rather than be told about it.
  [[nodiscard]] std::complex<double> principalAction() const;

  /// The largest turn, in radians, any radicand made about its branch point on
  /// the last `advance`. Approaching \f$ \pi \f$ means the path was sampled too
  /// coarsely for the continuation to be trusted.
  [[nodiscard]] double maxRadicandTurn() const { return maxRadicandTurn_; }
  /// The largest distance any angle moved in the complex plane on the last
  /// `advance`.
  [[nodiscard]] double maxAngleStep() const { return maxAngleStep_; }
  /// True when some radicand landed exactly on its branch point, where the sheet
  /// is held rather than continued because no continuation exists there.
  [[nodiscard]] bool touchedBranchPoint() const;

  /// The spatial dimension the labels were declared in.
  [[nodiscard]] int dimension() const { return dimension_; }

 private:
  struct CellState {
    SimplexPtr cell{nullptr};
    SheetedSqrt content{};
    double factorial{1.0};
  };
  struct HingeState {
    SimplexPtr hinge{nullptr};
    /// Absent for a point hinge, whose content is 1 and carries no root.
    bool hasContentRoot{false};
    SheetedSqrt content{};
    double factorial{1.0};
    std::vector<SimplexKey> cofaces{};
  };
  struct AngleState {
    SheetedSqrt rootII{};
    SheetedSqrt rootJJ{};
    SheetedAcos angle{};
  };
  using AngleKey = std::pair<SimplexKey, SimplexKey>;

  [[nodiscard]] const CellState &cellState(const SimplexKey &cell) const;
  [[nodiscard]] const HingeState &hingeState(const SimplexKey &hinge) const;
  [[nodiscard]] const AngleState &angleState(const AngleKey &key) const;
  /// The dihedral cosine built from the two cofactor roots on their declared
  /// sheets, with a real ratio pinned to the \f$ +0 \f$ side of the cut exactly as
  /// `Simplex::dihedralAngle` pins it.
  [[nodiscard]] static std::complex<double> cosineOn(
      std::complex<double> Cij, std::complex<double> rootII,
      std::complex<double> rootJJ);

  std::shared_ptr<spacetime::Spacetime> spacetime_{};
  int dimension_{0};
  std::map<SimplexKey, CellState> cells_{};
  std::map<SimplexKey, HingeState> hinges_{};
  std::map<AngleKey, AngleState> angles_{};
  double maxRadicandTurn_{0.0};
  double maxAngleStep_{0.0};
};

}  // namespace tessera::mesh

#endif  // TESSERA_MESH_REGGECONTINUATION_H
