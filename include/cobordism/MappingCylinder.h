// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_MAPPINGCYLINDER_H
#define TESSERA_COBORDISM_MAPPINGCYLINDER_H

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "cobordism/Certificate.h"

namespace tessera::cobordism {

/// # MappingCylinderDeclaration
///
/// The three pieces of data one tick of the recursion is built from, in the
/// convention of Section 3 of the whitepaper.
struct MappingCylinderDeclaration {
  /// The top cells of the incoming complex \f$ K^\ell \f$, each a tuple of
  /// vertex identifiers. The order inside a tuple is irrelevant; the cells are
  /// read on their ascending-vertex-identifier reference orientation. The list
  /// must be pure: every cell carries the same number of distinct vertices.
  std::vector<std::vector<std::uint64_t>> incomingTopCells;

  /// The reduction map \f$ r \f$ on vertices: for every vertex of
  /// \f$ K^\ell \f$, the identifier of the response vertex of
  /// \f$ \mathcal R^{\ell+1} \f$ it is carried to. It is a map on vertices
  /// alone, which is all a simplicial map needs, and its identifiers must be
  /// disjoint from the incoming ones because the cylinder holds both ends at
  /// once.
  std::map<std::uint64_t, std::uint64_t> reductionMap;

  /// The top cells of the outgoing complex \f$ K^{\ell+1} \f$, each a tuple of
  /// response-vertex identifiers. These are the cells the interactions of
  /// Section 6 attach among the response vertices; the reduction supplies none
  /// of them. An empty list declares that no interaction has been attached yet,
  /// and then the outgoing end of the cylinder is the image complex alone.
  std::vector<std::vector<std::uint64_t>> outgoingTopCells;
};

/// # MappingCylinderRead
///
/// The interaction cobordism \f$ W^\ell \f$ and the two complexes it runs
/// between, with the checks that identify its boundary.
struct MappingCylinderRead {
  /// The dimension of \f$ K^\ell \f$: one less than the number of vertices of
  /// each of its top cells.
  int incomingDimension = -1;

  /// The dimension of \f$ W^\ell \f$, which is `incomingDimension` plus one:
  /// the fibering direction is the extra simplex dimension, and it is the only
  /// reason the cylinder has one.
  int cylinderDimension = -1;

  /// The dimension of \f$ K^{\ell+1} \f$, or \f$ -1 \f$ when no outgoing cell
  /// is declared.
  int outgoingDimension = -1;

  /// The vertices of \f$ K^\ell \f$, ascending.
  std::vector<std::uint64_t> incomingVertices;

  /// The response vertices the reduction map lands on, ascending. They are the
  /// vertices of \f$ \mathcal R^{\ell+1} \f$.
  std::vector<std::uint64_t> responseVertices;

  /// The top cells of \f$ W^\ell \f$, each a tuple of
  /// `cylinderDimension` + 1 distinct vertices, sorted within the tuple and
  /// ascending over the list.
  ///
  /// For an incoming top cell \f$ [v_0<\dots<v_d] \f$ with images
  /// \f$ r_i=r(v_i) \f$, these are the staircase simplices
  /// \f$ [v_0,\dots,v_i,r_i,\dots,r_d] \f$ for \f$ i=0,\dots,d \f$, which is
  /// the standard triangulation of the prism \f$ \sigma\times[0,1] \f$ with the
  /// top face identified along \f$ r \f$. A staircase simplex whose response
  /// vertices repeat is degenerate and is dropped; when \f$ r \f$ carries the
  /// whole cell to one response vertex, exactly one survives and it is the cone
  /// \f$ \sigma * r \f$, which is the mapping cylinder of a constant map.
  std::vector<std::vector<std::uint64_t>> cylinderTopCells;

  /// The fiber edges \f$ (v, r(v)) \f$, one per vertex of \f$ K^\ell \f$: the
  /// edges joining each certified cluster's cells to its response vertex. They
  /// are the only timelike edges of the history, and the Lorentzian rotation
  /// protocol applies to them alone.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> fiberEdges;

  /// Every edge of \f$ W^\ell \f$ with one endpoint at each end, ascending. It
  /// contains the fiber edges and, where the reduction map is not constant on a
  /// cell, the diagonals of that cell's prism, which join a vertex to a
  /// response vertex other than its own.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> crossEdges;

  /// The maximal image cells \f$ r(\sigma) \f$ over the incoming top cells,
  /// with any cell contained in another removed: the image complex the
  /// outgoing end of the cylinder carries before \f$ K^{\ell+1} \f$ is
  /// attached. A cell the reduction map collapses appears here with fewer
  /// vertices than it had.
  std::vector<std::vector<std::uint64_t>> imageTopCells;

  /// The facets of \f$ W^\ell \f$ carried by exactly one top cell, split by
  /// which end they sit at: every vertex in \f$ K^\ell \f$, every vertex a
  /// response vertex, or one of each.
  std::vector<std::vector<std::uint64_t>> incomingFreeFacets;
  std::vector<std::vector<std::uint64_t>> outgoingFreeFacets;
  std::vector<std::vector<std::uint64_t>> sideFreeFacets;

  /// Whether the free facets that lie wholly in \f$ K^\ell \f$ are exactly the
  /// top cells of \f$ K^\ell \f$: the incoming end of \f$ \partial W^\ell \f$
  /// is the incoming complex.
  bool incomingBoundaryIsTheIncomingComplex = false;

  /// Whether every maximal image cell is a face of some top cell of
  /// \f$ K^{\ell+1} \f$: the outgoing end of the cylinder is attached inside
  /// the next level's complex rather than beside it. Reported false, with the
  /// offending cells countable from `boundaryResidual`, when no outgoing cell
  /// is declared.
  bool outgoingComplexContainsTheImage = false;

  /// Whether no free facet mixes the two ends. A closed \f$ K^\ell \f$ has no
  /// such facet; one with boundary has the cylinder over that boundary as its
  /// side wall, and then \f$ \partial W^\ell \f$ is not the disjoint union of
  /// the two ends alone.
  bool hasNoSideWall = false;

  /// \f$ \partial W^\ell = K^\ell \sqcup K^{\ell+1} \f$: the three conditions
  /// above together.
  bool boundaryIsTheDisjointUnion = false;

  /// The fraction of the checked cells that failed: free facets that are not
  /// top cells of \f$ K^\ell \f$, image cells that no outgoing cell carries,
  /// and free facets that mix the two ends, over the number of cells checked.
  /// Zero exactly when `boundaryIsTheDisjointUnion`.
  double boundaryResidual = 1.0;

  /// The construction's certificate. The staircase triangulation is a closed
  /// combinatorial identity, so the grade is `AlgebraicallyExact` and
  /// `boundaryResidual` is the whole measured error.
  Certificate certificate{};
};

/// # MappingCylinder
///
/// The interaction cobordism \f$ W^\ell \f$ of Section 3 of the whitepaper: one
/// tick of time, realized as the mapping cylinder of the reduction map from
/// \f$ K^\ell \f$ onto the vertex set of \f$ \mathcal R^{\ell+1} \f$, with the
/// cells of \f$ K^{\ell+1} \f$ attached on its outgoing end.
///
/// Reference: Hatcher, "Algebraic Topology", Cambridge University Press (2002),
/// Chapter 0, for the mapping cylinder and its prism (staircase) triangulation.
///
/// The whitepaper fixes the convention once: a level is a spatial complex, the
/// fibering direction is the passage of a certified cluster at level
/// \f$ \ell \f$ to a response vertex at level \f$ \ell+1 \f$, that direction is
/// time, and one tick is \f$ W^\ell \f$. The cylinder contains \f$ K^\ell \f$,
/// the fiber edges joining each certified cluster's cells to its response
/// vertex, and \f$ K^{\ell+1} \f$, with
/// \f$ \partial W^\ell = K^\ell \sqcup K^{\ell+1} \f$. Time is therefore not a
/// simplex dimension of any level; it is the fourth simplex dimension of
/// \f$ W^\ell \f$ only because \f$ W^\ell \f$ is built to realize that
/// direction geometrically, and that is exactly why `cylinderDimension` is
/// `incomingDimension` plus one.
///
/// The reduction determines no incidence maps of its own. The cells of
/// \f$ K^{\ell+1} \f$ are supplied entirely by the interactions among the
/// response vertices, so they are declared here rather than derived, and the
/// class checks that the image of the reduction lands inside them instead of
/// inventing them.
class MappingCylinder {
 public:
  /// Build the cylinder.
  ///
  /// @param declaration The incoming complex, the reduction map on its
  ///   vertices, and the outgoing complex.
  /// @throws std::invalid_argument when the incoming cells are empty, when they
  ///   are not pure, when a cell repeats a vertex, when a vertex of an incoming
  ///   cell has no image under the reduction map, when a response vertex
  ///   identifier is also an incoming vertex identifier, when the outgoing
  ///   cells are not pure, or when an outgoing cell names a vertex that is not
  ///   a response vertex.
  explicit MappingCylinder(MappingCylinderDeclaration declaration);

  /// The declaration this instance was built from.
  [[nodiscard]] const MappingCylinderDeclaration &declaration() const noexcept {
    return declaration_;
  }

  /// The cylinder, its two ends and the boundary checks.
  [[nodiscard]] const MappingCylinderRead &read() const noexcept {
    return read_;
  }

 private:
  MappingCylinderDeclaration declaration_;
  MappingCylinderRead read_;
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_MAPPINGCYLINDER_H
