// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// The boundary blocks a MultiCobordism carries: their markings, frames and
// surfaces.

#ifndef TESSERA_COBORDISM_MULTICOBORDISM_BLOCKS_H
#define TESSERA_COBORDISM_MULTICOBORDISM_BLOCKS_H

#include <complex>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cobordism/PencilLayer.h"

namespace tessera::cobordism::multicobordism {

/// A frame on a block's attached cells: the coordinates the two-body transfer
/// is read in. The transfer is
/// \f$ T_{AB} = (Z_A^\vee)^T (\tilde A)_{AB} Z_B \f$
/// (`chainhodge::PencilSchur::transfer`), the matrix of the whole complex's
/// pencil block in the frames' coordinates; a change of frame
/// \f$ Z_A \mapsto Z_A g_A \f$, \f$ Z_B \mapsto Z_B g_B \f$ gives
/// \f$ g_A^{-1} T g_B \f$. For two qubit tori the frames are their period
/// frames (`observables::SimplicialQubit::periodFrame`) and \f$ T \f$ is
/// \f$ 2 \times 2 \f$ in the \f$ |0\rangle, |1\rangle \f$ basis a two-body
/// target is written in. The frame is never scored; the fiber it expresses
/// is.
struct BlockFrame {
  /// The block's attached fiber cells (sorted vertex tuples), one per row.
  std::vector<std::vector<std::uint64_t>> cells;
  /// \f$ Z \f$: one column per basis element on `cells`
  /// (\f$ |cells| \times r \f$, rows in the fiber's attachment order).
  Eigen::MatrixXcd images;
  /// \f$ Z^\vee \f$: the left partner under the transpose pairing, with
  /// \f$ (Z^\vee)^T M_k Z = I_r \f$ for \f$ M_k \f$ the inverse chain metric
  /// (the Whitney mass matrix) of the block's own pencil on `cells`. Built by
  /// `dualFrame`, or `inputFrameDual` on the block's own complex.
  Eigen::MatrixXcd dualImages;
  /// The frame's rank \f$ r \f$.
  [[nodiscard]] int rank() const noexcept { return static_cast<int>(images.cols()); }
};

/// A block's marking together with its input coefficients: the cycles
/// \f$ A, B \f$ of the torus the block carries, as closed walks of directed
/// steps in host vertex ids (`Marking`), and the coefficients \f$ (a, b) \f$
/// of the state the block represents, one per cycle —
/// \f$ (1, \tau_{in}) \f$ for an input torus.
///
/// The engine uses a marking in two ways. It scores the block's own state
/// against the coefficients: the block's live surface is read as the
/// simplicial qubit over the marking (`blockQubit`), and the transported
/// periods of its holomorphic form over \f$ (A, B) \f$ must equal the
/// coefficients as a point of \f$ \mathbb{CP}^1 \f$ (`ownStateResidualOn`).
/// It also derives the block's frame at every read (`deriveFrame`), in which
/// the zero mode of the whole cobordism is reported as coefficients
/// (`readInputState`), as is the two-body transfer.
///
/// Unlike a frame stated at attachment (`setInputFrame`), a marking is
/// combinatorial data that every engine move preserves, so the frame it
/// normalizes stays the block's live kernel as its lengths move. The marking
/// itself is never scored: it fixes the coordinates, and by
/// \f$ A \cdot B = +1 \f$ the orientation, that the state is read in.
struct BlockMarking {
  /// The cycles, each one closed walk starting at `baseVertex`.
  Marking cycles;
  /// \f$ (a, b, \dots) \f$: one coefficient per cycle, the state at the block.
  Eigen::VectorXcd coefficients;
  /// The common base point of the transported periods: the first vertex of
  /// the first cycle's walk that lies on every other cycle's walk (the
  /// `observables::SimplicialQubit::baseVertex` rule).
  std::uint64_t baseVertex{0};
  [[nodiscard]] int rank() const noexcept { return static_cast<int>(cycles.size()); }
};

/// A boundary block of the cobordism, either an input or an output. A block
/// is not itself a complex: it stores the vertex set it occupies plus the
/// target period vector its own \f$ L_k \f$ sub-complex must carry. The
/// sub-complex is recovered on demand from `vertices` by
/// `getRegion`, the ambient complex's top cells whose
/// vertices all lie in the set.
struct BoundaryBlock {
  /// Host vertex ids the block occupies.
  std::set<std::uint64_t> vertices;
  /// Target period vector the block's own \f$ L_k \f$ sub-complex must carry.
  std::vector<std::complex<double>> target;
  /// The fiber form of the target: a retained fiber on the block's degree-k
  /// cells with its Gram matrix, band eigenvalue, contour and certificate,
  /// alongside the period vector. Set on an input block by `setInputFiber`
  /// (pinned as boundary data by `pinInputFibers`); read on an output block
  /// by `readOutputFiber` after relaxation.
  std::optional<BoundaryFiber> fiber;
  /// True when the block is an input surface of the host: seeded by the
  /// region form of `seedInputs` on a `fromSurfaces` host, its
  /// \f$ (d-1) \f$-faces are cells of the host that no top cell need cover,
  /// and the bulk is drawn onto them by bridges. A block seeded from a
  /// simplex is never a surface.
  bool surface{false};
  /// A surface block's own faces (sorted vertex-id tuples): the input
  /// surface's \f$ (d-1) \f$-simplices, recorded when the block is seeded
  /// and carried by the block thereafter, so that the block's own complex
  /// (`blockSurface`, `blockSurfaceWithGeometry`) is the surface whatever the
  /// host currently registers. The host registers a surface face only while a
  /// top cell covers it or a read has materialized it — a cone-out dent
  /// uncovers a face and the host's orphan prune then drops it — whereas the
  /// surface's vertices and edges survive every stage-1 move. The faces are
  /// therefore the one part of the surface the vertex set alone cannot
  /// recover. Empty for an ordinary block, whose own complex is the host's
  /// top cells inside its vertex set. Neither pinned nor scored.
  std::vector<std::vector<std::uint64_t>> faces;
  /// The block's frame for the two-body transfer, set by `setInputFrame` on
  /// the cells of its attached fiber. The engine holds it constant; the
  /// block's own residual keeps it valid as the lengths move. Absent on an
  /// unframed block; when either input block lacks one, the transfer is read
  /// in identity frames on the cells. Ignored on a block that carries a
  /// `marking`, where the frame is derived live instead.
  std::optional<BlockFrame> frame;
  /// The block's marking with its input coefficients, set by
  /// `setInputMarking`. Absent on an unmarked block.
  std::optional<BlockMarking> marking;
};

/// The frame derived for a marked block from its live own complex.
///
/// `frame.cells` are the block's own edges, in the own pencil's canonical
/// order. `frame.images` is \f$ F = Z\,\Pi^{-1} \f$, with \f$ Z \f$ the
/// images of the zero mode of the block's own covariant Whitney pencil
/// (`PencilLayer::harmonicContour` on `blockSurfaceWithGeometry`) and
/// \f$ \Pi_{ca} = \oint_c z_a \f$ its transported periods over the marking's
/// cycles from the common base point
/// (`chainhodge::Connection::transportedPeriod`), so column \f$ b \f$ has
/// period \f$ \delta_{cb} \f$ over cycle \f$ c \f$. On the collar seed this
/// is `observables::SimplicialQubit::periodFrame` of the input torus through
/// the id map.
///
/// `frame.dualImages` is
/// \f$ F^\vee = \tilde Z\,(\tilde Z^T M_1^U F)^{-T} \f$, with
/// \f$ \tilde Z \f$ the dual kernel's images (the same zero mode under the
/// inverse links, `AssembledPencil::dual`) and \f$ M_1^U \f$ the dressed
/// Whitney mass matrix of the block's own pencil, giving the `BlockFrame`
/// normalization \f$ (F^\vee)^T M_1^U F = I \f$ (`PencilLayer::dualFrame`). That pairing
/// is the gauge-covariant one; at zero phases the dual kernel is the kernel.
///
/// An `obstruction` is reported when the block's own kernel does not have the
/// marking's rank (a torn surface, or phases that are not a pure gauge), a
/// marking edge is absent from the own complex, the periods are singular
/// (cycles that do not span the block's homology), or the pairing is
/// isotropic. Such a block reads the full leak.
struct DerivedFrame {
  /// The derived frame on the block's own edges.
  BlockFrame frame;
  /// \f$ \Pi \f$, the transported periods over the marking's cycles
  /// (rank × rank).
  Eigen::MatrixXcd periods;
  /// Rank of the block's own zero mode.
  int kernelRank{0};
  /// Empty when the frame was derived; otherwise why it could not be.
  std::string obstruction;
  /// True when the frame was derived without obstruction.
  [[nodiscard]] bool derived() const noexcept { return obstruction.empty(); }
};

/// A block's own surface: its \f$ (d-1) \f$-faces and its edges inside its
/// vertex set, as sorted vertex-id tuples. A face is inside the block when
/// the block carries it (`BoundaryBlock::faces`), when the host registers it,
/// or when it is a facet of a top cell with all \f$ d \f$ of its vertices in
/// the block. All three are enumerated from vertex tuples rather than from
/// the lazily materialized facet links, so the answer does not depend on what
/// an earlier read materialized. For a surface block this is the input
/// surface itself, as long as no chord — an edge or face inside the block
/// that is not part of its surface — has been created.
struct BlockSurface {
  /// The block's \f$ (d-1) \f$-faces, as sorted vertex-id tuples.
  std::vector<std::vector<std::uint64_t>> faces;
  /// The block's edges, as sorted vertex-id pairs.
  std::vector<std::vector<std::uint64_t>> edges;
};

}  // namespace tessera::cobordism::multicobordism

#endif  // TESSERA_COBORDISM_MULTICOBORDISM_BLOCKS_H
