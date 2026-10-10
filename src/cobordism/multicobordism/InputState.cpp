// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): markings, derived
// frames, the input-state and own-state residuals and the block qubit. One
// of the translation units that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

// ---- the marking, the derived frame and the state at a block ----

bool MultiCobordism::orderMarking(Marking &cycles, std::uint64_t &baseVertex, std::string &obstruction) {
  for (std::size_t c = 0; c < cycles.size(); ++c) {
    if (cycles[c].empty()) {
      obstruction = "cycle " + std::to_string(c) + " has no steps";
      return false;
    }
    chainhodge::Connection::Walk walk = chainhodge::Connection::closedWalkOf(cycles[c]);
    if (walk.size() != cycles[c].size()) {
      obstruction = "cycle " + std::to_string(c) +
                    " does not form one closed walk (a vertex with unequal in- and out-degrees, or steps in "
                    "more than one connected component)";
      return false;
    }
    cycles[c] = std::move(walk);
  }
  const std::optional<std::uint64_t> base = chainhodge::Connection::commonBasePoint(cycles);
  if (!base) {
    obstruction = "the cycles share no vertex, so their transported periods have no common base point";
    return false;
  }
  baseVertex = *base;
  return true;
}

void MultiCobordism::setInputMarking(std::size_t index, Marking cycles,
                                     std::vector<std::complex<double>> coefficients) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::setInputMarking: input block index out of range");
  BoundaryBlock &block = inputBlocks_[index];
  if (!block.fiber || block.fiber->images.cols() == 0)
    throw std::logic_error("MultiCobordism::setInputMarking: input block " + std::to_string(index) +
                           " carries no attached fiber; attach the state fiber first (attachInputFiber)");
  if (block.fiber->degree != 1)
    throw std::invalid_argument("MultiCobordism::setInputMarking: a marking is cycles of edges, read at degree 1, "
                                "but input block " + std::to_string(index) + "'s fiber is at degree " +
                                std::to_string(block.fiber->degree));
  if (cycles.empty()) throw std::invalid_argument("MultiCobordism::setInputMarking: a marking needs at least one cycle");
  if (coefficients.size() != cycles.size())
    throw std::invalid_argument("MultiCobordism::setInputMarking: one coefficient per cycle (" +
                                std::to_string(cycles.size()) + " cycles, " + std::to_string(coefficients.size()) +
                                " coefficients)");
  double magnitude = 0.0;
  for (const auto &z : coefficients) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
      throw std::invalid_argument("MultiCobordism::setInputMarking: a coefficient is not finite");
    magnitude += std::norm(z);
  }
  if (!(magnitude > 0.0))
    throw std::invalid_argument("MultiCobordism::setInputMarking: the coefficients are all zero: no state");
  // Every step an edge of the live complex inside the block's vertex set.
  std::set<std::vector<std::uint64_t>> live;
  for (auto edge : ChainComplex::fromSpacetime(*spacetime_).kSimplexVertices(1)) live.insert(std::move(edge));
  for (std::size_t c = 0; c < cycles.size(); ++c)
    for (const auto &[u, v] : cycles[c]) {
      if (u == v)
        throw std::invalid_argument("MultiCobordism::setInputMarking: cycle " + std::to_string(c) +
                                    " steps on a self-loop at vertex " + std::to_string(u));
      if (!live.count({std::min(u, v), std::max(u, v)}))
        throw std::invalid_argument("MultiCobordism::setInputMarking: cycle " + std::to_string(c) + " steps (" +
                                    std::to_string(u) + " -> " + std::to_string(v) +
                                    ") across a pair that is not an edge of the live complex");
      if (!block.vertices.count(u) || !block.vertices.count(v))
        throw std::invalid_argument("MultiCobordism::setInputMarking: cycle " + std::to_string(c) + " steps (" +
                                    std::to_string(u) + " -> " + std::to_string(v) + ") outside input block " +
                                    std::to_string(index) + "'s vertex set");
    }
  std::uint64_t base = 0;
  std::string why;
  if (!orderMarking(cycles, base, why)) throw std::invalid_argument("MultiCobordism::setInputMarking: " + why);
  BlockMarking marking;
  marking.cycles = std::move(cycles);
  marking.coefficients = Eigen::Map<const Eigen::VectorXcd>(coefficients.data(),
                                                            static_cast<Eigen::Index>(coefficients.size()));
  marking.baseVertex = base;
  block.marking = std::move(marking);
}

const std::optional<MultiCobordism::BlockMarking> &MultiCobordism::inputMarking(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::inputMarking: input block index out of range");
  return inputBlocks_[index].marking;
}

void MultiCobordism::clearInputMarking(std::size_t index) {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::clearInputMarking: input block index out of range");
  inputBlocks_[index].marking.reset();
}

MultiCobordism::DerivedFrame MultiCobordism::deriveFrame(const BoundaryBlock &block,
                                                         const std::shared_ptr<Spacetime> &spacetime) {
  if (!block.marking) throw std::logic_error("MultiCobordism::deriveFrame: the block carries no marking");
  const BlockMarking &marking = *block.marking;
  DerivedFrame out;
  const auto own = blockComplexWithGeometry(block, spacetime);
  if (!own) {
    out.obstruction = "the block has no own complex (a torn surface carries no frame)";
    return out;
  }
  try {
    const AssembledPencil assembled = PencilLayer::assemble({own});
    if (assembled.dimension() < 1) {
      out.obstruction = "the block's own complex has no edges";
      return out;
    }
    // The zero mode of the block's own covariant pencil: its harmonic contour,
    // recomputed here as the lengths move, with the phases entering through
    // the dressed pencil.
    const chainhodge::Contour contour = PencilLayer::harmonicContour(assembled, 1);
    const chainhodge::Band band = assembled.op->band(1, contour);
    out.kernelRank = band.rank();
    if (band.rank() != marking.rank()) {
      out.obstruction = "the block's own kernel has rank " + std::to_string(band.rank()) + " for a marking of " +
                        std::to_string(marking.rank()) +
                        " cycles (a torn surface, or link phases that are not a pure gauge)";
      return out;
    }
    const auto rank = static_cast<Eigen::Index>(marking.rank());
    // Pi_{ca}: the transported period of zero-mode column a over cycle c from
    // the common base point (a marking step absent from the own complex is
    // refused by name by the connection, and read as an obstruction here).
    Eigen::MatrixXcd periods(rank, rank);
    for (Eigen::Index c = 0; c < rank; ++c)
      for (Eigen::Index a = 0; a < rank; ++a)
        periods(c, a) = assembled.op->connection().transportedPeriod(
            band.images.col(a), marking.cycles[static_cast<std::size_t>(c)]);
    Eigen::FullPivLU<Eigen::MatrixXcd> lu(periods);
    if (!lu.isInvertible()) {
      out.obstruction = "the periods of the block's own kernel over the marking are singular: the cycles do not "
                        "span the block's homology";
      return out;
    }
    // F = Z Pi^{-1}: column b has period delta_{cb} over cycle c.
    const Eigen::MatrixXcd F = band.images * lu.inverse();
    // F^vee = Z~ (Z~^T M_1^U F)^{-T} with Z~ the dual kernel's images (the
    // zero mode under the inverse links) and M_1^U the dressed Whitney mass
    // matrix: (F^vee)^T M_1^U F = I, the BlockFrame contract paired between
    // the kernel and the dual kernel.
    const Eigen::MatrixXcd dualImages = assembled.dual->applyG(1, band.dualFrame);
    const Eigen::MatrixXcd M(assembled.op->Minv(1));
    const Eigen::MatrixXcd pairing = dualImages.transpose() * M * F;
    Eigen::FullPivLU<Eigen::MatrixXcd> pairingLu(pairing);
    if (!pairingLu.isInvertible()) {
      out.obstruction = "the pairing of the block's dual kernel against its frame is singular (an isotropic frame)";
      return out;
    }
    out.frame.cells = assembled.complex().kSimplexVertices(1);
    out.frame.images = F;
    out.frame.dualImages = pairingLu.solve(dualImages.transpose()).transpose();
    out.periods = periods;
  } catch (const std::runtime_error &e) {
    out.obstruction = std::string("the block's own pencil refused the read: ") + e.what();
  } catch (const std::invalid_argument &e) {
    out.obstruction = std::string("the block's own pencil refused the read: ") + e.what();
  }
  return out;
}

MultiCobordism::DerivedFrame MultiCobordism::deriveInputFrame(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::deriveInputFrame: input block index out of range");
  if (!inputBlocks_[index].marking)
    throw std::logic_error("MultiCobordism::deriveInputFrame: input block " + std::to_string(index) +
                           " carries no marking (setInputMarking)");
  return deriveFrame(inputBlocks_[index], spacetime_);
}

BoundaryFiber MultiCobordism::stateTargetOf(const BlockMarking &marking, const BlockFrame &frame) {
  BoundaryFiber target;
  target.degree = 1;
  target.cells = frame.cells;
  target.images = frame.images * marking.coefficients;
  return target;
}

double MultiCobordism::inputStateResidualOn(const BoundaryBlock &block,
                                            const std::shared_ptr<Spacetime> &spacetime) const {
  if (!block.marking)
    throw std::logic_error("MultiCobordism::inputStateResidualOn: the block carries no marking (setInputMarking)");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::inputStateResidualOn: the block residual is read on the chain-level "
                           "Whitney pencil; this node uses the diagonal-weight metric");
  if (!spacetime) return 1.0;
  const DerivedFrame derived = deriveFrame(block, spacetime);
  if (!derived.derived()) return 1.0;  // no frame: the state leaks in full
  // The whole's zero mode at the whole's harmonic contour (R7), on the
  // block's edges; the leak of the target there.
  return fiberResidualOn(spacetime, stateTargetOf(*block.marking, derived.frame), FiberBand::ZeroMode);
}

double MultiCobordism::inputStateResidual(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::inputStateResidual: input block index out of range");
  if (!inputBlocks_[index].marking)
    throw std::logic_error("MultiCobordism::inputStateResidual: input block " + std::to_string(index) +
                           " carries no marking (setInputMarking)");
  return inputStateResidualOn(inputBlocks_[index], spacetime_);
}

observables::SimplicialQubit MultiCobordism::blockQubit(const BoundaryBlock &block,
                                                        const std::shared_ptr<Spacetime> &spacetime) {
  if (!block.marking) throw std::logic_error("MultiCobordism::blockQubit: the block carries no marking (setInputMarking)");
  if (block.marking->rank() != 2)
    throw std::invalid_argument("MultiCobordism::blockQubit: a qubit torus is marked by two cycles; this marking has " +
                                std::to_string(block.marking->rank()));
  if (!spacetime) throw std::invalid_argument("MultiCobordism::blockQubit: null spacetime");
  const std::shared_ptr<Spacetime> surface = blockSurfaceWithGeometry(block, spacetime);
  if (!surface)
    throw std::runtime_error("MultiCobordism::blockQubit: the block has no surface (a face of the torus lost an "
                             "edge, so it carries no state)");
  // The surface as SimplicialQubit's Spacetime constructor indexes it:
  // Vertices by ascending id, edges by ascending index pair. The marking's
  // steps (host ids, u -> v) become (edge index, sign) with the sign of the
  // step against the edge's stored orientation (min -> max).
  std::vector<std::uint64_t> ids;
  for (const auto *vertex : surface->getVertexList()->liveVector()) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  std::map<std::uint64_t, std::uint64_t> indexOfId;
  for (std::size_t n = 0; n < ids.size(); ++n) indexOfId[ids[n]] = n;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> pairs;
  for (const mesh::Edge *edge : surface->getEdgeList()->toVector()) {
    const std::uint64_t a = indexOfId.at(edge->getSource()->getId());
    const std::uint64_t b = indexOfId.at(edge->getTarget()->getId());
    pairs.emplace_back(std::min(a, b), std::max(a, b));
  }
  std::sort(pairs.begin(), pairs.end());
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> indexOfPair;
  for (std::size_t n = 0; n < pairs.size(); ++n) indexOfPair[pairs[n]] = n;
  auto cycleOf = [&](const std::vector<std::pair<std::uint64_t, std::uint64_t>> &steps, const char *name) {
    observables::SimplicialQubit::Cycle cycle;
    for (const auto &[u, v] : steps) {
      const auto iu = indexOfId.find(u), iv = indexOfId.find(v);
      if (iu == indexOfId.end() || iv == indexOfId.end())
        throw std::runtime_error(std::string("MultiCobordism::blockQubit: a step of cycle ") + name + " (" +
                                 std::to_string(u) + " -> " + std::to_string(v) +
                                 ") leaves the block's live surface");
      const auto found = indexOfPair.find({std::min(iu->second, iv->second), std::max(iu->second, iv->second)});
      if (found == indexOfPair.end())
        throw std::runtime_error(std::string("MultiCobordism::blockQubit: the marking edge (") + std::to_string(u) +
                                 ", " + std::to_string(v) + ") of cycle " + name +
                                 " is not an edge of the block's live surface");
      cycle.emplace_back(found->second, iu->second < iv->second ? 1 : -1);
    }
    return cycle;
  };
  const observables::SimplicialQubit::Cycle cycleA = cycleOf(block.marking->cycles[0], "A");
  const observables::SimplicialQubit::Cycle cycleB = cycleOf(block.marking->cycles[1], "B");
  // The orientation the marking fixes, A . B = +1. The
  // container stores none; the fundamental class picks one of the two, and
  // the intersection number says whether it is the marking's.
  observables::SimplicialQubit read(surface, cycleA, cycleB, /*reversed=*/false);
  if (read.intersectionNumber() < 0.0) return observables::SimplicialQubit(surface, cycleA, cycleB, /*reversed=*/true);
  return read;
}

observables::SimplicialQubit MultiCobordism::blockQubit(std::size_t index) const {
  if (index >= inputBlocks_.size()) throw std::out_of_range("MultiCobordism::blockQubit: input block index out of range");
  return blockQubit(inputBlocks_[index], spacetime_);
}

double MultiCobordism::ownStateLeakOf(std::complex<double> periodA, std::complex<double> periodB,
                                      const Eigen::VectorXcd &coefficients) {
  if (coefficients.size() != 2)
    throw std::invalid_argument("MultiCobordism::ownStateLeakOf: a qubit state has two coefficients");
  const Eigen::Vector2cd p(periodA, periodB);
  const Eigen::Vector2cd c(coefficients(0), coefficients(1));
  const double pp = p.squaredNorm(), cc = c.squaredNorm();
  if (!(pp > 0.0) || !(cc > 0.0)) return 1.0;
  const std::complex<double> overlap = c.dot(p);  // c^H p: Eigen's dot conjugates its first argument
  return std::max(0.0, 1.0 - std::norm(overlap) / (cc * pp));
}

double MultiCobordism::ownStateResidualOn(const BoundaryBlock &block,
                                          const std::shared_ptr<Spacetime> &spacetime) const {
  if (!block.marking)
    throw std::logic_error("MultiCobordism::ownStateResidualOn: the block carries no marking (setInputMarking)");
  if (!spacetime) return 1.0;
  std::optional<observables::SimplicialQubit> read;
  try {
    read.emplace(blockQubit(block, spacetime));
  } catch (const std::runtime_error &) {
    return 1.0;  // no surface, or a refused read: the state leaks in full
  } catch (const std::invalid_argument &) {
    return 1.0;  // the surface refuses the qubit's validation (a degenerate triangle)
  }
  // The periods over the given marking: the qubit read reports (B, -A) when
  // |P_A| vanishes, and the leak is written on the raw pair.
  auto [pA, pB] = read->periods();
  if (read->markingSwapped()) {
    const std::complex<double> rawA = -pB, rawB = pA;
    pA = rawA;
    pB = rawB;
  }
  return ownStateLeakOf(pA, pB, block.marking->coefficients);
}

double MultiCobordism::ownStateResidual(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::ownStateResidual: input block index out of range");
  if (!inputBlocks_[index].marking)
    throw std::logic_error("MultiCobordism::ownStateResidual: input block " + std::to_string(index) +
                           " carries no marking (setInputMarking)");
  return ownStateResidualOn(inputBlocks_[index], spacetime_);
}

MultiCobordism::ResidualGradient MultiCobordism::ownStateResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime, const BoundaryBlock &block) const {
  if (!spacetime) throw std::invalid_argument("MultiCobordism::ownStateResidualGradientOn: null spacetime");
  if (!block.marking)
    throw std::logic_error("MultiCobordism::ownStateResidualGradientOn: the block carries no marking (setInputMarking)");
  const auto parentEdges = spacetime->getEdgeList()->toVector();
  ResidualGradient gradient;
  gradient.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(parentEdges.size()));
  gradient.phases = Eigen::VectorXcd();  // tau is invariant under the pure gauge the surface carries
  std::optional<observables::SimplicialQubit> read;
  try {
    read.emplace(blockQubit(block, spacetime));
  } catch (const std::runtime_error &) {
    return gradient;  // the full leak has no direction
  } catch (const std::invalid_argument &) {
    return gradient;
  }
  using Complex = std::complex<double>;
  const Eigen::VectorXcd &c = block.marking->coefficients;
  if (c.size() != 2) throw std::logic_error("MultiCobordism::ownStateResidualGradientOn: a qubit state has two coefficients");
  const Complex a = c(0), b = c(1);
  const double N = std::norm(a) + std::norm(b);
  // dr = 2 Re(d_tau r . d tau): the Wirtinger derivative of the real leak
  // with respect to the reported tau(), in the chart tau = P_B/P_A, or
  // sigma = P_A/P_B = -tau() when the marking is swapped.
  const Complex tau = read->tau();
  Complex dr;
  if (!read->markingSwapped()) {
    const Complex u = std::conj(a) + std::conj(b) * tau;
    const double D = 1.0 + std::norm(tau);
    dr = -(std::conj(b) * std::conj(u) * D - std::norm(u) * std::conj(tau)) / (N * D * D);
  } else {
    const Complex sigma = -tau;
    const Complex u = std::conj(a) * sigma + std::conj(b);
    const double D = 1.0 + std::norm(sigma);
    const Complex drdsigma = -(std::conj(a) * std::conj(u) * D - std::norm(u) * std::conj(sigma)) / (N * D * D);
    dr = -drdsigma;  // tau() = -sigma
  }
  const Eigen::VectorXcd dtau = read->tauDerivative();
  // The torus's edges (index pairs in the surface's ascending-id order) onto
  // the parent's edges by vertex pair.
  std::vector<std::uint64_t> ids;
  for (const auto *vertex : read->spacetime()->getVertexList()->liveVector()) ids.push_back(vertex->getId());
  std::sort(ids.begin(), ids.end());
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> parentIndex;
  for (std::size_t e = 0; e < parentEdges.size(); ++e) parentIndex[edgeKey(parentEdges[e])] = e;
  const auto &edges = read->edges();
  for (std::size_t e = 0; e < edges.size(); ++e) {
    const std::uint64_t u = ids.at(edges[e].first), v = ids.at(edges[e].second);
    const auto found = parentIndex.find({std::min(u, v), std::max(u, v)});
    if (found == parentIndex.end())
      throw std::logic_error("MultiCobordism::ownStateResidualGradientOn: an edge of the block's own surface is not "
                             "an edge of the parent complex");
    // (d/dRe z, d/dIm z) of the real residual packed as a complex number:
    // dz real moves tau by tau_z dz, dz imaginary by i tau_z dz.
    const Complex w = dr * dtau(static_cast<Eigen::Index>(e));
    gradient.lengths[static_cast<Eigen::Index>(found->second)] += Complex(2.0 * w.real(), -2.0 * w.imag());
  }
  return gradient;
}

MultiCobordism::ResidualGradient MultiCobordism::ownStateResidualGradient(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::ownStateResidualGradient: input block index out of range");
  return ownStateResidualGradientOn(spacetime_, inputBlocks_[index]);
}

MultiCobordism::InputStateRead MultiCobordism::readInputState(std::size_t index) const {
  if (index >= inputBlocks_.size())
    throw std::out_of_range("MultiCobordism::readInputState: input block index out of range");
  const BoundaryBlock &block = inputBlocks_[index];
  if (!block.marking)
    throw std::logic_error("MultiCobordism::readInputState: input block " + std::to_string(index) +
                           " carries no marking (setInputMarking)");
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::readInputState: the state at a block is read on the chain-level "
                           "Whitney pencil; this node uses the diagonal-weight metric");
  const BlockMarking &marking = *block.marking;
  InputStateRead read;
  read.block = index;
  read.input = marking.coefficients;
  read.coefficients = Eigen::VectorXcd::Zero(marking.coefficients.size());
  read.weight = inputResidualWeight_;
  read.baseVertex = marking.baseVertex;
  read.residual = 1.0;
  const DerivedFrame derived = deriveFrame(block, spacetime_);
  read.frameRank = derived.kernelRank;
  if (!derived.derived()) {
    read.obstruction = derived.obstruction;
    return read;
  }
  const BoundaryFiber target = stateTargetOf(marking, derived.frame);
  try {
    const AssembledPencil assembled = PencilLayer::assemble({spacetime_});
    const std::vector<int> idx = PencilLayer::indicesOf(assembled, 1, target.cells);
    const chainhodge::Contour contour = PencilLayer::harmonicContour(assembled, 1);
    const chainhodge::Band band = assembled.op->band(1, contour);
    read.harmonicRank = band.rank();
    if (band.rank() == 0) {
      read.obstruction = "the whole has no degree-1 zero mode";
      return read;
    }
    // The same least-squares fit as `fiberResidualOn`, so the residual here is
    // the scored one to the bit; the fitted combination is a kernel vector of
    // the whole, and its transported periods over the cycles are its
    // coordinates in the block's frame.
    Eigen::MatrixXcd ZT(static_cast<Eigen::Index>(idx.size()), band.rank());
    for (std::size_t i = 0; i < idx.size(); ++i) ZT.row(static_cast<Eigen::Index>(i)) = band.images.row(idx[i]);
    const Eigen::MatrixXcd c = ZT.colPivHouseholderQr().solve(target.images);
    read.residual = (ZT * c - target.images).squaredNorm() / target.images.squaredNorm();
    const Eigen::MatrixXcd fitted = band.images * c;  // one column: the rank-one state
    for (Eigen::Index cycle = 0; cycle < read.coefficients.size(); ++cycle)
      read.coefficients(cycle) = assembled.op->connection().transportedPeriod(
          fitted.col(0), marking.cycles[static_cast<std::size_t>(cycle)]);
  } catch (const std::runtime_error &e) {
    read.obstruction = std::string("the whole refused the read: ") + e.what();
    read.residual = 1.0;
  } catch (const std::invalid_argument &e) {
    read.obstruction = std::string("the whole refused the read: ") + e.what();
    read.residual = 1.0;
  }
  return read;
}

}  // namespace tessera::cobordism
