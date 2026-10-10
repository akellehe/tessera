// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the move machinery:
// snapshots, edge geometry, move specifications and their application, the
// local change of the objective, and one step. One of the translation units
// that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "MultiCobordismInternal.h"

namespace tessera::cobordism {

MultiCobordism::Snapshot MultiCobordism::snapshotOf(
    const Spacetime &spacetime) const {
  std::vector<std::vector<std::uint64_t>> cellVertexTuples;
  for (const auto &topSimplex : spacetime.getTopSimplices())
    cellVertexTuples.push_back(topSimplex->topTuple());
  // A surface input's faces that no top cell covers are cells of the complex
  // in their own right (the bulk is still being drawn onto them), and a
  // rebuild from top cells alone would erase them. They follow the top cells,
  // so the top-cell order — the dual node indexing — is unchanged, and
  // `Spacetime::fromVertexTuples` registers a (d-1)-vertex cell as the non-top
  // simplex it is. Absent on every node without surface inputs.
  for (const auto &face : uncoveredInputFacesOn(spacetime))
    cellVertexTuples.push_back(face);
  // Every edge's length and phase, verbatim and branch-exact: the phase is a
  // live field of the geometry (the tori's pure-gauge links), and a record
  // that dropped it would rebuild a different operator.
  std::map<std::pair<std::uint64_t, std::uint64_t>, EdgeGeometry> geometryByEdge;
  for (const auto *edge : spacetime.getEdgeList()->toVector())
    geometryByEdge[edgeKey(edge)] = edgeGeometryOf(*edge);
  return {std::move(cellVertexTuples), std::move(geometryByEdge)};
}

MultiCobordism::Snapshot MultiCobordism::snapshot() const {
  return snapshotOf(*spacetime_);
}

complexd MultiCobordism::inverseLinkPhase(complexd phase) noexcept {
  // 0 - phi, not -phi: IEEE negation of +0 is -0, and a -0 phase would make
  // an untouched all-zero connection differ bit for bit from the rebuilt one
  // (a signed zero survives exp and prints as such); 0 - x equals -x exactly
  // for every nonzero x and is +0 for either zero.
  return complexd(0.0, 0.0) - phase;
}

MultiCobordism::BoundaryRecord MultiCobordism::boundaryRecordOf(
    const Spacetime &spacetime) {
  BoundaryRecord record;
  record.facets = boundaryFacetsOf(spacetime);
  if (record.facets.empty() || spacetime.getEdgeList() == nullptr) return record;
  // Every vertex pair of every boundary facet: an edge is on the boundary when
  // some boundary facet contains both its endpoints, which is the same
  // incidence the facet set is read from, so the two halves cannot disagree
  // about what the boundary is.
  std::set<std::pair<std::uint64_t, std::uint64_t>> boundaryEdges;
  for (const auto &facet : record.facets)
    for (std::size_t first = 0; first + 1 < facet.size(); ++first)
      for (std::size_t second = first + 1; second < facet.size(); ++second)
        boundaryEdges.emplace(std::min(facet[first], facet[second]),
                              std::max(facet[first], facet[second]));
  for (const auto &endpoints : boundaryEdges) {
    const ::tessera::mesh::EdgeKey key(endpoints.first, endpoints.second);
    if (const auto *edge =
            spacetime.getEdgeList()->get(key.fingerprint.fingerprint()))
      record.edges.emplace(endpoints, edgeGeometryOf(*edge));
  }
  return record;
}

MultiCobordism::EdgeGeometry MultiCobordism::edgeGeometryOf(
    const ::tessera::mesh::Edge &edge) {
  const auto sourceVertexId = edge.getSource()->getId();
  const auto targetVertexId = edge.getTarget()->getId();
  return {edge.getLength(), sourceVertexId < targetVertexId
                                ? edge.getPhase()
                                : inverseLinkPhase(edge.getPhase())};
}

void MultiCobordism::restoreEdgeGeometry(::tessera::mesh::Edge &edge,
                                         const EdgeGeometry &geometry) {
  const auto sourceVertexId = edge.getSource()->getId();
  const auto targetVertexId = edge.getTarget()->getId();
  edge.setLength(geometry.length);  // verbatim, branch-exact
  edge.setPhase(sourceVertexId < targetVertexId
                    ? geometry.phase
                    : inverseLinkPhase(geometry.phase));
}

std::shared_ptr<Spacetime> MultiCobordism::rebuild(
    int dimensions, const Snapshot &complexSnapshot) {
  auto rebuiltSpacetime =
      Spacetime::fromVertexTuples(dimensions, complexSnapshot.first, 1.0,
                           complexd(0.0, 0.0));
  for (auto *edge : rebuiltSpacetime->getEdgeList()->toVector()) {
    const auto savedEntry = complexSnapshot.second.find(edgeKey(edge));
    if (savedEntry != complexSnapshot.second.end())
      restoreEdgeGeometry(*edge, savedEntry->second);
  }
  return rebuiltSpacetime;
}

std::shared_ptr<Spacetime> MultiCobordism::build(
    const Snapshot &complexSnapshot) const {
  auto rebuiltSpacetime = rebuild(spacetime_->getDimensions(), complexSnapshot);
  // Candidate clones inherit the wiring mode so combinatorial moves scored on
  // them wire their new edges under the same convention.
  rebuiltSpacetime->setBalancedEdgeWiring(balancedEdgeWiring_);
  return rebuiltSpacetime;
}

MultiCobordism::MoveSpec MultiCobordism::drawRandomMoveSpecification(
    const Spacetime &spacetime) {
  // With emergent dispositions enabled the draw also offers a timelike cone-in
  // and a disposition flip on an existing edge. Both are ordinary candidate moves:
  // Proposed at random, scored by deltaF, committed only if they lower F. Nothing
  // prescribes causal structure -- the objective decides whether it wants any.
  //
  // These discrete proposals remain useful for jumping directly between causal
  // sectors. Complex-z stage 2 can also rotate continuously around z=0; neither
  // path prescribes which causal structure the objective should prefer.
  // The four Pachner kinds always; the surgical kinds and the disposition
  // kinds as declared, in this order, so the draw is the same stream for the
  // same declaration.
  std::vector<const char *> moveKinds = {kAddMove, kRemoveMove, kFlipMove,
                                         kIFlipMove};
  if (shouldProposeSurgery_) {
    moveKinds.push_back(kConeOut);
    moveKinds.push_back(kConeIn);
  }
  if (shouldProposeDispositions_) {
    if (shouldProposeSurgery_) moveKinds.push_back(kConeInTimelike);
    moveKinds.push_back(kFlipDisposition);
  }
  const std::size_t nMoveKinds = moveKinds.size();
  // The bridge kind joins the draw only on a node with surface inputs whose
  // bridge phase is incomplete: its candidates are the cells adjacent to the
  // drawing's frontier, split across two surface blocks. Any other node draws
  // the remaining kinds from the same generator stream.
  std::vector<std::vector<std::uint64_t>> bridgeCandidates;
  if (hasSurfaceInputs()) bridgeCandidates = bridgeCandidatesOn(spacetime);
  const std::size_t nKindsOffered =
      nMoveKinds + (bridgeCandidates.empty() ? 0u : 1u);
  const std::size_t kindIndex = randomNumberGenerator_() % nKindsOffered;
  if (kindIndex == nMoveKinds)
    return {kBridge,
            bridgeCandidates[randomNumberGenerator_() % bridgeCandidates.size()]};
  const std::string moveKind = moveKinds[kindIndex];

  // Flip the disposition of one existing edge, chosen uniformly. The payload is
  // the edge's two vertex ids.
  if (moveKind == kFlipDisposition) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> edgeEndpoints;
    if (spacetime.getEdgeList())
      for (const auto *edge : spacetime.getEdgeList()->toVector())
        if (edge != nullptr && edge->getSource() != nullptr &&
            edge->getTarget() != nullptr)
          edgeEndpoints.emplace_back(edge->getSource()->getId(),
                                     edge->getTarget()->getId());
    if (edgeEndpoints.empty()) return {kNoop, {}};
    const auto &chosen =
        edgeEndpoints[randomNumberGenerator_() % edgeEndpoints.size()];
    return {kFlipDisposition, {chosen.first, chosen.second}};
  }
  if (moveKind == kAddMove || moveKind == kRemoveMove ||
      moveKind == kFlipMove || moveKind == kIFlipMove)
    return {moveKind,
            {static_cast<std::uint64_t>(randomNumberGenerator_() % (1u << 31))}};
  if (moveKind == kConeOut) {
    std::vector<std::vector<std::uint64_t>> topCellTuples;
    for (const auto &topSimplex : spacetime.getTopSimplices())
      topCellTuples.push_back(topSimplex->topTuple());
    if (topCellTuples.empty()) return {kNoop, {}};
    return {kConeOut,
            topCellTuples[randomNumberGenerator_() % topCellTuples.size()]};
  }
  // Cone_in and cone_in_timelike share a payload (the facet to cone onto); only
  // the apex-edge disposition differs when applied. Only a boundary facet (one
  // coface) can accept a cone — an interior facet already has two cofaces, so
  // coning it would be non-manifold and the gate rejects it after a full
  // build+apply+deltaF evaluation. Drawing from getBoundary() directly spends
  // the batch on the coneable set only (measured on a 13-cell build frame: the
  // old cell×dropped-vertex draw had 65 outcomes aliasing onto 41 facets, 17
  // coneable, with interior duds drawn twice as often as valid facets). The
  // facet's stored vertex order is passed through verbatim.
  auto boundaryFacets = spacetime.getBoundary();
  if (boundaryFacets.empty()) return {kNoop, {}};  // closed: nothing coneable
  return {moveKind,
          boundaryFacets[randomNumberGenerator_() % boundaryFacets.size()]};
}

std::vector<MultiCobordism::MoveSpec> MultiCobordism::enumerateMoveSpecifications(
    const std::shared_ptr<Spacetime> &spacetime, bool withDispositions,
    bool withSurgery) {
  std::vector<MoveSpec> specifications;
  if (!spacetime) return specifications;
  // The four Pachner kinds, each site produced by the move class that will act
  // on it -- asked of the move rather than re-derived here, so a site this
  // returns is one that class recognizes.
  for (auto &site : ::tessera::spacetime::AddMove::sitesOn(*spacetime))
    specifications.emplace_back(kAddAt, std::move(site));
  for (auto &site : ::tessera::spacetime::RemoveMove::sitesOn(*spacetime))
    specifications.emplace_back(kRemoveAt, std::move(site));
  for (auto &site : ::tessera::spacetime::FlipMove::sitesOn(*spacetime))
    specifications.emplace_back(kFlipAt, std::move(site));
  for (auto &site : ::tessera::spacetime::IFlipMove::sitesOn(*spacetime))
    specifications.emplace_back(kIFlipAt, std::move(site));
  // The surgical kinds already name their sites, so they are enumerated in the
  // same encoding the draw uses -- no second spelling of a cone's payload.
  if (withSurgery) {
    for (const auto &topSimplex : spacetime->getTopSimplices())
      if (topSimplex)
        specifications.emplace_back(kConeOut, topSimplex->topTuple());
    for (const auto &facet : spacetime->getBoundary()) {
      specifications.emplace_back(kConeIn, facet);
      if (withDispositions) specifications.emplace_back(kConeInTimelike, facet);
    }
  }
  if (withDispositions && spacetime->getEdgeList())
    for (const auto *edge : spacetime->getEdgeList()->toVector())
      if (edge != nullptr && edge->getSource() != nullptr &&
          edge->getTarget() != nullptr)
        specifications.emplace_back(
            kFlipDisposition,
            std::vector<std::uint64_t>{edge->getSource()->getId(),
                                       edge->getTarget()->getId()});
  return specifications;
}

bool MultiCobordism::applyMoveSpecification(
    const std::shared_ptr<Spacetime> &spacetime,
    const MoveSpec &moveSpecification) {
  const auto &moveKind = moveSpecification.first;
  CLOG(INFO_LEVEL, "Applying a ", moveKind, " move.");
  if (moveKind == kNoop) return false;
  // The boundary before the move (setBoundaryMayExtend). A cone-in buries the
  // facet it stands on and exposes the new cell's others, a cone-out exposes
  // every facet of the cell it removes, so either can hand the boundary faces
  // it did not have; and a disposition flip on an edge of a boundary facet
  // leaves the facet set alone while changing the metric of ∂W underneath it.
  // The record is therefore taken for every move kind and carries the
  // boundary's geometry as well as its facets.
  const bool gateBoundary = !boundaryMayExtend_ && hasFixedBoundary();
  const BoundaryRecord boundaryBefore =
      gateBoundary ? boundaryRecordOf(*spacetime) : BoundaryRecord{};
  bool moveWasApplied = false;
  // The site-addressed Pachner kinds: the same four moves, proposed at
  // the site the payload names instead of one drawn from a seed. Everything
  // after the proposal -- apply, the gates below, the rollback -- is the path
  // the drawn kinds take, because `proposeAt` leaves the move in exactly the
  // state a successful `propose()` does.
  if (moveKind == kAddAt || moveKind == kRemoveAt || moveKind == kFlipAt ||
      moveKind == kIFlipAt) {
    using ::tessera::spacetime::PachnerMode;
    // Unused by these kinds -- the site is named, not drawn -- but the move
    // classes take a generator, so give each its own rather than share one.
    std::mt19937 unusedEngine(0u);
    const auto &site = moveSpecification.second;
    if (moveKind == kAddAt) {
      ::tessera::spacetime::AddMove pachnerMove(
          spacetime.get(), &unusedEngine, false, PachnerMode::PreGeometric,
          false);
      moveWasApplied = pachnerMove.proposeAt(site) && pachnerMove.apply();
    } else if (moveKind == kRemoveAt) {
      ::tessera::spacetime::RemoveMove pachnerMove(
          spacetime.get(), &unusedEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.proposeAt(site) && pachnerMove.apply();
    } else if (moveKind == kFlipAt) {
      ::tessera::spacetime::FlipMove pachnerMove(
          spacetime.get(), &unusedEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.proposeAt(site) && pachnerMove.apply();
    } else {
      ::tessera::spacetime::IFlipMove pachnerMove(
          spacetime.get(), &unusedEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.proposeAt(site) && pachnerMove.apply();
    }
  } else if (moveKind == kAddMove || moveKind == kRemoveMove ||
      moveKind == kFlipMove || moveKind == kIFlipMove) {
    std::mt19937 moveRandomEngine(
        static_cast<std::uint32_t>(moveSpecification.second[0]));
    using ::tessera::spacetime::PachnerMode;
    if (moveKind == kAddMove) {
      ::tessera::spacetime::AddMove pachnerMove(
          spacetime.get(), &moveRandomEngine, false, PachnerMode::PreGeometric,
          false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == kRemoveMove) {
      ::tessera::spacetime::RemoveMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else if (moveKind == kFlipMove) {
      ::tessera::spacetime::FlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    } else {
      ::tessera::spacetime::IFlipMove pachnerMove(
          spacetime.get(), &moveRandomEngine, PachnerMode::PreGeometric, false);
      moveWasApplied = pachnerMove.propose() && pachnerMove.apply();
    }
  } else if (moveKind == kConeOut) {
    moveWasApplied =
        SurgicalCone(spacetime.get()).coneOut(moveSpecification.second).first;
  } else if (moveKind == kBridge) {
    moveWasApplied =
        SurgicalCone(spacetime.get()).bridge(moveSpecification.second).first;
  } else if (moveKind == kFlipDisposition) {
    // Negate one edge's squared length, carrying it across the light cone.
    // Spacelike <-> timelike is a discrete step stage 2 cannot take (it would have
    // to pass through the singular l^2 = 0), which is why it is a move. Whether it
    // lowers F is left to deltaF and step()'s acceptance test, exactly as for
    // every other move; whether it is a member of the configuration space at all
    // is settled by the boundary record taken above, which refuses it on an edge
    // of a fixed boundary.
    if (payloadNamesAnEdge(moveSpecification.second) &&
        spacetime->getEdgeList()) {
      // O(1) via the EdgeList's fingerprint -> slot map, not an O(|E|) scan:
      // EdgeKey canonicalizes the endpoint pair, so orientation does not matter.
      const ::tessera::mesh::EdgeKey key(moveSpecification.second[0],
                                         moveSpecification.second[1]);
      if (auto *edge =
              spacetime->getEdgeList()->get(key.fingerprint.fingerprint())) {
        edge->setLength(std::sqrt(-(edge->getLength() * edge->getLength())));
        moveWasApplied = true;
      }
    }
  } else {
    moveWasApplied = SurgicalCone(spacetime.get())
                         .coneIn(moveSpecification.second,
                                 /*timelike=*/moveKind == kConeInTimelike)
                         .first;
  }
  if (!moveWasApplied) return false;
  if (gateBoundary) {
    // Refused, not repaired: a complex whose boundary gained a face, or whose
    // boundary is the same faces carrying different lengths or phases, is not
    // the cobordism of the declared states. It is not a member of the
    // configuration space and the candidate is dropped like any other gate
    // failure. The boundary is the input state: a trial that rewrites it is
    // answering a different question, however much it lowers F.
    const BoundaryRecord boundaryAfter = boundaryRecordOf(*spacetime);
    for (const auto &facet : boundaryAfter.facets)
      if (!boundaryBefore.facets.count(facet)) return false;
    for (const auto &[endpoints, geometry] : boundaryBefore.edges) {
      const auto found = boundaryAfter.edges.find(endpoints);
      if (found == boundaryAfter.edges.end() ||
          found->second.length != geometry.length ||
          found->second.phase != geometry.phase)
        return false;
    }
  }
  // Manifold validity is the whole gate. A move that removes a pinned vertex is
  // accepted when what it leaves is a valid manifold in its own right: pinning
  // constrains the geometry, it does not veto a topology change.
  if (!EigenstateSynthesis(spacetime, dualComplexGateDegree_, metricSource_)
           .dualComplexValid()
           .first)
    return false;
  // Under the Whitney pencil the configuration space is the closure of the
  // Kontsevich–Segal allowable domain; a proposal outside it is not a member.
  return geometryAdmissible(spacetime);
}

bool MultiCobordism::geometryAdmissible(const std::shared_ptr<Spacetime> &spacetime) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil) return true;
  if (!spacetime) return false;
  constexpr double kBoundaryTolerance = 1e-12;
  return HodgeLaplacian::kontsevichSegalMargin(*spacetime) >= -kBoundaryTolerance;
}

double MultiCobordism::deltaF(
    const std::shared_ptr<Spacetime> &candidateSpacetime, double baseObjective,
    double baseResidualU,
    const std::set<std::vector<std::uint64_t>> &baseCellSet) const {
  // An objective that declares a localized exact delta is differenced over the
  // cells the move touches, below. One that does not depends on global spectra
  // or action magnitudes, so its full scalar difference is the only valid
  // score. It costs more, but it prevents stage 1 from optimizing a surrogate
  // different from the objective it reports.
  if (!compositeSupportsLocalizedDelta()) {
    // A candidate whose operator the metric source cannot assemble carries no
    // objective value: under the Whitney pencil a geometry on the closure of
    // the allowable domain can have a singular dressed metric, whose
    // factorization refuses by name. It scores as the worst case, so it is
    // never committed, exactly as a residual term evaluates such a geometry.
    try {
      return objectiveFor(candidateSpacetime) - baseObjective;
    } catch (const std::runtime_error &) {
      return std::numeric_limits<double>::infinity();
    }
  }

  std::set<std::vector<std::uint64_t>> candidateCellSet;
  for (const auto &topSimplex : candidateSpacetime->getTopSimplices())
    candidateCellSet.insert(topSimplex->topTuple());
  std::vector<std::vector<std::uint64_t>> touchedCells;
  for (const auto &cell : baseCellSet)
    if (!candidateCellSet.count(cell)) touchedCells.push_back(cell);
  for (const auto &cell : candidateCellSet)
    if (!baseCellSet.count(cell)) touchedCells.push_back(cell);

  // The touched-cell diff alone is not the whole affected set: flip_disposition
  // changes an edge's l^2 sign and no cells, so the diff comes back empty even
  // though flipping an edge between spacelike and timelike changes the deficit
  // angle of every hinge on it. Diff the edge l^2 values too and pull in the
  // top cells incident to any edge that moved. This is move-agnostic, so it
  // also covers cone_in_timelike's apex edges.
  //
  // Widening is safe: delta||grad S||^2 = after - before is exact over any
  // fixed superset of the affected edges, because every edge outside the set
  // keeps its gradient and cancels.
  std::map<std::pair<std::uint64_t, std::uint64_t>, complexd> baseLengths;
  for (const auto *edge : spacetime_->getEdgeList()->toVector())
    baseLengths[edgeKey(edge)] = edge->getLength();
  std::set<std::pair<std::uint64_t, std::uint64_t>> movedEdges;
  for (const auto *edge : candidateSpacetime->getEdgeList()->toVector()) {
    const auto key = edgeKey(edge);
    const auto found = baseLengths.find(key);
    if (found == baseLengths.end() ||
        found->second != edge->getLength())
      movedEdges.insert(key);
    if (found != baseLengths.end()) baseLengths.erase(found);
  }
  for (const auto &leftover : baseLengths)  // in base, absent from candidate
    movedEdges.insert(leftover.first);
  if (!movedEdges.empty()) {
    std::set<std::vector<std::uint64_t>> incidentCells;
    const auto collectIncident = [&](const Spacetime &spacetime) {
      for (const auto &topSimplex : spacetime.getTopSimplices()) {
        auto cellVertexIds = topSimplex->topTuple();
        for (const auto &moved : movedEdges) {
          const bool cellHoldsBothEndpoints =
              std::find(cellVertexIds.begin(), cellVertexIds.end(),
                        moved.first) != cellVertexIds.end() &&
              std::find(cellVertexIds.begin(), cellVertexIds.end(),
                        moved.second) != cellVertexIds.end();
          if (cellHoldsBothEndpoints) {
            incidentCells.insert(std::move(cellVertexIds));
            break;
          }
        }
      }
    };
    collectIncident(*spacetime_);
    collectIncident(*candidateSpacetime);
    for (const auto &cell : incidentCells)
      if (std::find(touchedCells.begin(), touchedCells.end(), cell) ==
          touchedCells.end())
        touchedCells.push_back(cell);
  }

  ReggeSolver baseReggeSolver(spacetime_, MatterConfiguration());
  ReggeSolver candidateReggeSolver(candidateSpacetime, MatterConfiguration());
  std::set<std::pair<std::uint64_t, std::uint64_t>> affectedEdgeSet;
  for (const auto &edgeEndpoints :
       baseReggeSolver.affectedEdgesOfCells(touchedCells))
    affectedEdgeSet.insert(edgeEndpoints);
  for (const auto &edgeEndpoints :
       candidateReggeSolver.affectedEdgesOfCells(touchedCells))
    affectedEdgeSet.insert(edgeEndpoints);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> affectedEdges(
      affectedEdgeSet.begin(), affectedEdgeSet.end());
  // Skipped entirely when the Einstein-Hilbert term is off: scoring a
  // move by a term the objective does not contain would make stage 1 disagree
  // with `objective()` about which moves lower F.
  const double gradientDelta =
      einsteinHilbert_
          ? candidateReggeSolver.gradientNorm2OverEdges(affectedEdges) -
                baseReggeSolver.gradientNorm2OverEdges(affectedEdges)
          : 0.0;
  const double residualUDelta = rU(candidateSpacetime) - baseResidualU;
  return reggeWeight_ * gradientDelta + gamma_ * residualUDelta;
}

std::pair<double, MultiCobordism::Snapshot> MultiCobordism::bestComposition(
    const Snapshot &fromSnapshot, int remainingMoves, double baseObjective,
    double baseResidualU,
    const std::set<std::vector<std::uint64_t>> &baseCellSet) {
  auto fromSpacetime = build(fromSnapshot);
  // Nothing left to apply: the caller asked what the complex it holds is
  // worth, which is the composition's score.
  if (remainingMoves <= 0)
    return {deltaF(fromSpacetime, baseObjective, baseResidualU, baseCellSet),
            fromSnapshot};
  std::pair<double, Snapshot> best{std::numeric_limits<double>::infinity(),
                                   Snapshot{}};
  // Enumerated here, against this level's complex, not against the base one: a
  // move the first move created a site for is a legitimate second move, and a
  // site the first move destroyed is not one.
  for (const auto &specification : enumerateMoveSpecifications(
           fromSpacetime, shouldProposeDispositions_, shouldProposeSurgery_)) {
    auto candidateSpacetime = build(fromSnapshot);
    if (!applyMoveSpecification(candidateSpacetime, specification)) continue;
    if (remainingMoves == 1) {
      // The leaf, scored in place: the finished composition diffed against the
      // base complex, exactly as a single move is. Snapshotting only on an
      // improvement keeps the walk's memory at one recorded complex per level.
      const double objectiveDelta = deltaF(candidateSpacetime, baseObjective,
                                           baseResidualU, baseCellSet);
      if (objectiveDelta < best.first)
        best = {objectiveDelta, snapshotOf(*candidateSpacetime)};
      continue;
    }
    auto reached =
        bestComposition(snapshotOf(*candidateSpacetime), remainingMoves - 1,
                        baseObjective, baseResidualU, baseCellSet);
    if (reached.first < best.first) best = std::move(reached);
  }
  return best;
}

double MultiCobordism::step(int nCandidateMoves, int lookaheadDepth,
                            double baseObjective) {
  // The candidate loop below constructs one `ReggeSolver` on the live
  // spacetime per candidate (`deltaF`), and that constructor materializes the
  // facet lattice lazily — a mutation of the shared object. On a live complex
  // whose facets are not yet materialized (a fresh seed, or the complex built
  // from the snapshot of a committed move) two OpenMP threads then race in
  // `Simplex::getFacets` and corrupt the simplex deque (measured on the Δ³
  // fiber drive: SIGSEGV in `Spacetime::createSimplex` from two
  // candidate threads). Reach the fixpoint once, here, so every thread's
  // construction is read-only.
  spacetime_->materializeFacets();
  const auto currentSnapshot = snapshot();
  const double baseResidualU =
      compositeSupportsLocalizedDelta() ? rU(spacetime_) : 0.0;
  std::set<std::vector<std::uint64_t>> baseCellSet;
  for (const auto &topSimplex : spacetime_->getTopSimplices())
    baseCellSet.insert(topSimplex->topTuple());
  // One scoring rule at every depth: the localized, unrelaxed deltaF.
  // The two stages have separate jobs — the combinatorial moves exist to leave
  // a local minimum, the geometric update to descend to the minimum of the
  // region the complex then sits in — and scoring a candidate through a
  // relaxation mixed them, asking where a move would land after stage 2 rather
  // than whether the move itself improves the state. Relaxation now happens
  // only after a move is committed, bounded by the caller's relaxBudgetPerMove.
  double bestObjectiveDelta = -convergenceTolerance_;
  bool foundImprovingMove = false;
  Snapshot bestSnapshot;
  if (lookaheadDepth <= 1) {
    // Depth 1: every candidate starts from the same base complex, so the specs
    // are pre-drawn serially, in the same RNG order as a serial loop, and the
    // batch is scored in parallel. applyMoveSpecification is deterministic
    // given its spec (it seeds a local engine from the payload), build()
    // constructs an independent complex, and deltaF is const over it. The inner
    // OpenMP region of the action gradient serializes inside each worker
    // (nesting off). The reduction is the lexicographic (delta, index) min, so
    // the earliest candidate among equals wins.
    std::vector<MoveSpec> specifications;
    // A non-positive count means every candidate rather than a sample of them.
    // Read this way round because "how many to draw" and "draw them all" are
    // the same question, and a caller that asks for none of them means
    // something no drive can do.
    if (nCandidateMoves <= 0) {
      specifications = enumerateMoveSpecifications(
          spacetime_, shouldProposeDispositions_, shouldProposeSurgery_);
    } else {
      specifications.reserve(static_cast<std::size_t>(nCandidateMoves));
      for (int candidateIndex = 0; candidateIndex < nCandidateMoves;
           ++candidateIndex)
        specifications.push_back(drawRandomMoveSpecification(*spacetime_));
    }
    // Deduplicate exact (kind, payload) repeats before evaluating: the batch
    // samples with replacement, and on a small complex the same spec recurs
    // (the cone-in space can be a dozen-odd facets). Duplicates carry
    // identical deltas, so dropping every copy after the first cannot change
    // the lexicographic (delta, index) winner; only the wasted
    // build+apply+deltaF evaluations go away.
    // The RNG stream is untouched (all nCandidateMoves draws happen above).
    // Pachner specs carry RNG-seed payloads, so only exact seed repeats
    // collapse there; the cone/disposition kinds dedup by actual site.
    {
      std::set<MoveSpec> seenSpecifications;
      std::vector<MoveSpec> distinctSpecifications;
      distinctSpecifications.reserve(specifications.size());
      for (auto &specification : specifications)
        if (seenSpecifications.insert(specification).second)
          distinctSpecifications.push_back(std::move(specification));
      specifications = std::move(distinctSpecifications);
    }
    const int distinctCount = static_cast<int>(specifications.size());
    std::vector<double> deltas(static_cast<std::size_t>(distinctCount),
                               std::numeric_limits<double>::infinity());
    std::vector<Snapshot> snapshots(static_cast<std::size_t>(distinctCount));
    std::exception_ptr pending = nullptr;
#pragma omp parallel for schedule(dynamic)
    for (int candidateIndex = 0; candidateIndex < distinctCount;
         ++candidateIndex) {
      try {
        auto candidateSpacetime = build(currentSnapshot);
        if (!applyMoveSpecification(
                candidateSpacetime,
                specifications[static_cast<std::size_t>(candidateIndex)]))
          continue;  // failed the gate: stays at +inf
        const double objectiveDelta =
            deltaF(candidateSpacetime, baseObjective, baseResidualU, baseCellSet);
        deltas[static_cast<std::size_t>(candidateIndex)] = objectiveDelta;
        if (objectiveDelta < -convergenceTolerance_)
          snapshots[static_cast<std::size_t>(candidateIndex)] =
              snapshotOf(*candidateSpacetime);
      } catch (...) {
        // An exception may not leave an OpenMP region: capture the first and
        // rethrow after the join, so it reaches the caller instead of
        // terminating the process.
#pragma omp critical(tessera_stage1_candidate_eptr)
        if (!pending) pending = std::current_exception();
      }
    }
    if (pending) std::rethrow_exception(pending);
    for (int candidateIndex = 0; candidateIndex < distinctCount;
         ++candidateIndex) {
      const double objectiveDelta =
          deltas[static_cast<std::size_t>(candidateIndex)];
      if (objectiveDelta < bestObjectiveDelta) {
        bestObjectiveDelta = objectiveDelta;
        bestSnapshot =
            std::move(snapshots[static_cast<std::size_t>(candidateIndex)]);
        foundImprovingMove = true;
      }
    }
  } else if (nCandidateMoves <= 0) {
    // Exhaustive at depth > 1: every gated composition of `lookaheadDepth`
    // moves, each level enumerated against the complex the previous level
    // left. The first move is the parallel level — one independent subtree per
    // candidate, so the work divides evenly and nothing below it is shared —
    // and each subtree is walked serially by `bestComposition`. The reduction
    // is the same lexicographic (delta, index) min the depth-1 batch uses, so
    // the earliest first move among equals wins here too.
    //
    // The cost is the move space raised to the depth, and is paid only when a
    // caller passes the exhaustive sentinel.
    const auto firstMoves = enumerateMoveSpecifications(
        spacetime_, shouldProposeDispositions_, shouldProposeSurgery_);
    const int firstMoveCount = static_cast<int>(firstMoves.size());
    std::vector<double> deltas(static_cast<std::size_t>(firstMoveCount),
                               std::numeric_limits<double>::infinity());
    std::vector<Snapshot> snapshots(static_cast<std::size_t>(firstMoveCount));
    std::exception_ptr pending = nullptr;
#pragma omp parallel for schedule(dynamic)
    for (int candidateIndex = 0; candidateIndex < firstMoveCount;
         ++candidateIndex) {
      try {
        auto candidateSpacetime = build(currentSnapshot);
        if (!applyMoveSpecification(
                candidateSpacetime,
                firstMoves[static_cast<std::size_t>(candidateIndex)]))
          continue;  // failed the gate: the whole subtree stays at +inf
        auto reached =
            bestComposition(snapshotOf(*candidateSpacetime), lookaheadDepth - 1,
                            baseObjective, baseResidualU, baseCellSet);
        deltas[static_cast<std::size_t>(candidateIndex)] = reached.first;
        if (reached.first < -convergenceTolerance_)
          snapshots[static_cast<std::size_t>(candidateIndex)] =
              std::move(reached.second);
      } catch (...) {
        // An exception may not leave an OpenMP region: capture the first and
        // rethrow after the join, so it reaches the caller instead of
        // terminating the process.
#pragma omp critical(tessera_stage1_candidate_eptr)
        if (!pending) pending = std::current_exception();
      }
    }
    if (pending) std::rethrow_exception(pending);
    for (int candidateIndex = 0; candidateIndex < firstMoveCount;
         ++candidateIndex) {
      const double objectiveDelta =
          deltas[static_cast<std::size_t>(candidateIndex)];
      if (objectiveDelta < bestObjectiveDelta) {
        bestObjectiveDelta = objectiveDelta;
        bestSnapshot =
            std::move(snapshots[static_cast<std::size_t>(candidateIndex)]);
        foundImprovingMove = true;
      }
    }
  } else
  for (int candidateIndex = 0; candidateIndex < nCandidateMoves; ++candidateIndex) {
    // One candidate = `lookaheadDepth` gated random moves applied in sequence,
    // each drawn against the evolving candidate complex. The sequence is scored
    // — and, if best, committed — as a whole, so an F-lowering pair whose first
    // move alone raises F is still a descent step. This deepened path
    // stays serial: each draw is made against the candidate the previous move
    // left, so the sequence cannot be pre-drawn the way a depth-1 batch is.
    auto candidateSpacetime = build(currentSnapshot);
    bool wholeSequenceApplied = true;
    for (int moveIndex = 0; moveIndex < lookaheadDepth; ++moveIndex) {
      const auto moveSpecification =
          drawRandomMoveSpecification(*candidateSpacetime);
      if (!applyMoveSpecification(candidateSpacetime, moveSpecification)) {
        wholeSequenceApplied = false;  // one link failed the gate: discard the candidate
        break;
      }
    }
    if (!wholeSequenceApplied) continue;
    // Scored exactly as a depth-1 candidate is: the finished sequence diffed
    // against the base complex. deltaF is exact over any fixed superset of the
    // affected edges, so a multi-move candidate needs no special treatment.
    const double objectiveDelta =
        deltaF(candidateSpacetime, baseObjective, baseResidualU, baseCellSet);
    if (objectiveDelta < bestObjectiveDelta) {
      bestObjectiveDelta = objectiveDelta;
      // The snapshot carries the sequence's as-built geometry: nothing was
      // relaxed to earn the score, so nothing is being banked here either.
      bestSnapshot = snapshotOf(*candidateSpacetime);
      foundImprovingMove = true;
    }
  }
  if (foundImprovingMove) {
    spacetime_ = build(bestSnapshot);
    // The first committed move is what starts linking the bulk, so block
    // regions are settled from here on.
    bulkConnected_ = true;
    // The move is already committed — `bestObjectiveDelta` is fixed and
    // `spacetime_` already replaced — before the analysis overlay is offered
    // the chance to look at it. The overlay is post-hoc by construction: there
    // is no path from here back to the acceptance test above.
    noteAcceptedMove();
    return bestObjectiveDelta;
  }
  return 0.0;
}

void MultiCobordism::preconeCells(int count, bool timelike, bool alternate) {
  // Each cone-in cones a fresh apex onto a random codimension-1 facet (a top
  // cell with one vertex dropped) and is committed only through
  // applyMoveSpecification's dualComplexValid gate. On the single-Δ⁴ seed a
  // cone-in over a boundary facet is valid, so this enlarges the 4-ball; a draw
  // onto an already-saturated interior facet is rejected and retried.
  // `timelike` draws every cone-in as the timelike disposition (apex edges
  // ℓ² = −1); `alternate` interleaves timelike and spacelike for balanced
  // causal content. Either way every edge sits at |ℓ²| = 1; the default is the
  // all-spacelike precone.
  constexpr int kAttemptsPerCone = 20;  // gated tries before giving up on one cone
  for (int conedSoFar = 0; conedSoFar < count; ++conedSoFar) {
    std::vector<std::vector<std::uint64_t>> topCellTuples;
    for (const auto &topSimplex : spacetime_->getTopSimplices())
      topCellTuples.push_back(topSimplex->topTuple());
    if (topCellTuples.empty()) return;  // nothing to cone onto
    bool coned = false;
    for (int attempt = 0; attempt < kAttemptsPerCone && !coned; ++attempt) {
      const auto &chosenCell =
          topCellTuples[randomNumberGenerator_() % topCellTuples.size()];
      const std::size_t droppedVertexIndex =
          randomNumberGenerator_() % chosenCell.size();
      std::vector<std::uint64_t> coneInFace;  // a codim-1 facet: drop one vertex
      for (std::size_t vertexIndex = 0; vertexIndex < chosenCell.size();
           ++vertexIndex)
        if (vertexIndex != droppedVertexIndex)
          coneInFace.push_back(chosenCell[vertexIndex]);
      auto candidateSpacetime = build(snapshot());
      const bool coneTimelike =
          alternate ? (conedSoFar % 2 == 0) : timelike;
      if (applyMoveSpecification(
              candidateSpacetime,
              {coneTimelike ? kConeInTimelike : kConeIn, coneInFace})) {
        spacetime_ = build(snapshotOf(*candidateSpacetime));
        coned = true;
      }
    }
    if (!coned) return;  // no valid cone-in found for this cell; stop early
  }
}

}  // namespace tessera::cobordism
