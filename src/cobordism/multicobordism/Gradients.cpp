// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): analytic gradients of
// the fiber-mode, input-state, whole-harmonic and two-body residuals, and
// the fiber-mode ascent. One of the translation units that define the
// class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

namespace {

// ---- analytic gradients of the fiber-mode residuals ----


/// (∂/∂Re, ∂/∂Im) of a real objective whose holomorphic sensitivity to the
/// coordinate is `dF` (that is, dObjective = 2 Re(dF · dcoordinate)).
complexd packHolomorphic(complexd dF) {
  return complexd{2.0 * dF.real(), -2.0 * dF.imag()};
}

/// Canonical C_1 index of every live edge (EdgeList order), by vertex pair.
std::vector<std::size_t> canonicalEdgeIndices(const Spacetime &spacetime, const ChainComplex &K) {
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> canonical;
  const auto cells = K.kSimplexVertices(1);
  for (std::size_t j = 0; j < cells.size(); ++j) canonical[{cells[j][0], cells[j][1]}] = j;
  std::vector<std::size_t> out;
  for (const auto *edge : spacetime.getEdgeList()->toVector()) {
    const auto found = canonical.find(edgeKey(edge));
    if (found == canonical.end())
      throw std::logic_error("MultiCobordism: a live edge is absent from the chain complex");
    out.push_back(found->second);
  }
  return out;
}

}  // namespace

MultiCobordism::ResidualGradient MultiCobordism::fiberResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime, const BoundaryFiber &target, FiberBand band) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::fiberResidualGradientOn: read on the chain-level Whitney pencil");
  if (!spacetime) throw std::invalid_argument("MultiCobordism::fiberResidualGradientOn: null spacetime");
  const auto edges = spacetime->getEdgeList()->toVector();
  ResidualGradient gradient;
  gradient.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  if (target.degree == 0) gradient.phases = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  const std::vector<int> idx = PencilLayer::indicesOf(assembled, target.degree, target.cells);
  const chainhodge::Contour contour = fiberContourOn(assembled, target, band);
  const chainhodge::Band riesz = assembled.op->band(target.degree, contour);
  if (riesz.rank() == 0) return gradient;  // full leak everywhere: no descent direction
  // The least-squares fit on the cells: u = psi - Z_T c, r = |u|^2 / |psi|^2.
  Eigen::MatrixXcd ZT(static_cast<Eigen::Index>(idx.size()), riesz.rank());
  for (std::size_t i = 0; i < idx.size(); ++i) ZT.row(static_cast<Eigen::Index>(i)) = riesz.images.row(idx[i]);
  const Eigen::MatrixXcd c = ZT.colPivHouseholderQr().solve(target.images);
  const Eigen::MatrixXcd u = target.images - ZT * c;
  const double norm = target.images.squaredNorm();
  const chainhodge::BandDerivative::ResolventFrames frames =
      chainhodge::BandDerivative::resolventFrames(*assembled.op, target.degree, contour, riesz.frame);
  const std::vector<std::size_t> canonical = canonicalEdgeIndices(*spacetime, assembled.complex());
  // Holomorphic sensitivity: d r = 2 Re(dF · dcoord) with dF = -tr(u^H dZ_T c)/|psi|^2.
  const auto sensitivity = [&](const Eigen::MatrixXcd &dZ) {
    complexd dF(0.0, 0.0);
    for (std::size_t i = 0; i < idx.size(); ++i) {
      const Eigen::RowVectorXcd row = dZ.row(idx[i]) * c;  // 1 x columns of psi
      dF += (u.row(static_cast<Eigen::Index>(i)).conjugate().cwiseProduct(row)).sum();
    }
    return -dF / norm;
  };
  // One edge's dZ is an independent dense solve against the same band, and each
  // writes its own slot, so the sweep is the natural parallel level: no
  // reduction, no shared accumulator, and the result is bit-identical to the
  // serial sweep. The caches the derivative path fills lazily are warmed first
  // (see `warmDerivatives`); nesting is left off so a call from inside stage
  // 1's candidate batch stays on the outer level.
  assembled.op->warmDerivatives(target.degree);
  std::exception_ptr pending = nullptr;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
  for (std::int64_t e = 0; e < static_cast<std::int64_t>(edges.size()); ++e) {
    try {
      const std::size_t index = static_cast<std::size_t>(e);
      gradient.lengths[static_cast<Eigen::Index>(index)] = packHolomorphic(sensitivity(
          chainhodge::BandDerivative::imagesLengthDerivative(*assembled.op, frames, riesz.images,
                                                             canonical[index])));
      if (target.degree == 0)
        gradient.phases[static_cast<Eigen::Index>(index)] = packHolomorphic(sensitivity(
            chainhodge::BandDerivative::imagesPhaseDerivative(*assembled.op, frames, riesz.images,
                                                              canonical[index])));
    } catch (...) {
      // An exception may not leave an OpenMP region: capture the first and
      // rethrow after the join, so a singular pencil still reaches the callers
      // that catch it rather than aborting the process.
#pragma omp critical(tessera_fiber_gradient_eptr)
      if (!pending) pending = std::current_exception();
    }
  }
  if (pending) std::rethrow_exception(pending);
  return gradient;
}

std::vector<MultiCobordism::ResidualGradient> MultiCobordism::inputStateResidualGradientsOn(
    const std::shared_ptr<Spacetime> &spacetime, const std::vector<const BoundaryBlock *> &blocks) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism::inputStateResidualGradientsOn: read on the chain-level Whitney pencil");
  if (!spacetime) throw std::invalid_argument("MultiCobordism::inputStateResidualGradientsOn: null spacetime");
  const auto edges = spacetime->getEdgeList()->toVector();
  const auto edgeCount = static_cast<Eigen::Index>(edges.size());
  std::vector<ResidualGradient> gradients(blocks.size());
  for (auto &gradient : gradients) gradient.lengths = Eigen::VectorXcd::Zero(edgeCount);
  if (blocks.empty()) return gradients;
  for (const BoundaryBlock *block : blocks)
    if (!block || !block->marking)
      throw std::logic_error("MultiCobordism::inputStateResidualGradientsOn: a block carries no marking");
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> parentIndex;
  for (std::size_t e = 0; e < edges.size(); ++e) parentIndex[edgeKey(edges[e])] = e;
  // The whole: its zero mode and the resolvent frames of its band derivative,
  // computed once and shared by every block (the expensive part is the
  // per-edge derivative of the whole's images, which does not depend on the
  // block).
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  const chainhodge::Contour contour = PencilLayer::harmonicContour(assembled, 1);
  const chainhodge::Band riesz = assembled.op->band(1, contour);
  if (riesz.rank() == 0) return gradients;  // full leak everywhere: no direction
  const chainhodge::BandDerivative::ResolventFrames frames =
      chainhodge::BandDerivative::resolventFrames(*assembled.op, 1, contour, riesz.frame);
  const std::vector<std::size_t> canonical = canonicalEdgeIndices(*spacetime, assembled.complex());
  // Per block: the target through its live frame, the least-squares fit of
  // the target in the whole's zero mode on its edges (as `fiberResidualOn`),
  // and the target's motion dt on its own edges.
  struct BlockTerms {
    bool active{false};
    std::vector<int> idx;
    Eigen::VectorXcd t, c, u;
    double norm{0.0}, leak{0.0};
    std::vector<std::optional<Eigen::VectorXcd>> targetMotion;
  };
  std::vector<BlockTerms> terms(blocks.size());
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const BoundaryBlock &block = *blocks[b];
    const BlockMarking &marking = *block.marking;
    const DerivedFrame derived = deriveFrame(block, spacetime);
    if (!derived.derived()) continue;  // the full leak has no direction
    BlockTerms &term = terms[b];
    const BoundaryFiber target = stateTargetOf(marking, derived.frame);
    term.idx = PencilLayer::indicesOf(assembled, 1, target.cells);
    Eigen::MatrixXcd ZT(static_cast<Eigen::Index>(term.idx.size()), riesz.rank());
    for (std::size_t i = 0; i < term.idx.size(); ++i)
      ZT.row(static_cast<Eigen::Index>(i)) = riesz.images.row(term.idx[i]);
    term.t = target.images.col(0);
    term.c = ZT.colPivHouseholderQr().solve(term.t);
    term.u = term.t - ZT * term.c;
    term.norm = term.t.squaredNorm();
    term.leak = term.u.squaredNorm() / term.norm;
    // dt = (dZ Pi^{-1} - F dPi Pi^{-1}) (a, b)^T on the block's own pencil, one
    // column per own edge (the frame's rows, the target's rows), mapped to the
    // parent's edges by vertex pair (a surface's edges are host edges); zero
    // on every bulk edge.
    const auto own = blockComplexWithGeometry(block, spacetime);
    const AssembledPencil ownAssembled = PencilLayer::assemble({own});
    const chainhodge::Contour ownContour = PencilLayer::harmonicContour(ownAssembled, 1);
    const chainhodge::Band ownBand = ownAssembled.op->band(1, ownContour);
    const chainhodge::BandDerivative::ResolventFrames ownFrames =
        chainhodge::BandDerivative::resolventFrames(*ownAssembled.op, 1, ownContour, ownBand.frame);
    const Eigen::FullPivLU<Eigen::MatrixXcd> periodsLu(derived.periods);
    const Eigen::MatrixXcd periodsInverse = periodsLu.inverse();
    const Eigen::MatrixXcd &F = derived.frame.images;
    term.targetMotion.resize(edges.size());
    const auto ownEdges = ownAssembled.complex().kSimplexVertices(1);
    const auto rank = static_cast<Eigen::Index>(marking.rank());
    // The block's own sweep: one independent dZ per own edge, each landing in
    // its own parent slot (the parent index is injective on own edges), so the
    // loop parallelizes without touching the ordering of anything it writes.
    ownAssembled.op->warmDerivatives(1);
    std::exception_ptr ownPending = nullptr;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
    for (std::int64_t je = 0; je < static_cast<std::int64_t>(ownEdges.size()); ++je) {
      try {
        const std::size_t j = static_cast<std::size_t>(je);
        const auto parent = parentIndex.find({ownEdges[j][0], ownEdges[j][1]});
        if (parent == parentIndex.end())
          throw std::logic_error("MultiCobordism::inputStateResidualGradientsOn: an edge of the block's own complex "
                                 "is absent from the parent");
        const Eigen::MatrixXcd dZ =
            chainhodge::BandDerivative::imagesLengthDerivative(*ownAssembled.op, ownFrames, ownBand.images, j);
        Eigen::MatrixXcd dPeriods(rank, rank);
        for (Eigen::Index cycle = 0; cycle < rank; ++cycle)
          for (Eigen::Index a = 0; a < rank; ++a)
            dPeriods(cycle, a) = ownAssembled.op->connection().transportedPeriod(
                dZ.col(a), marking.cycles[static_cast<std::size_t>(cycle)]);
        const Eigen::MatrixXcd dF = dZ * periodsInverse - F * (dPeriods * periodsInverse);
        term.targetMotion[parent->second] = Eigen::VectorXcd(dF * marking.coefficients);
      } catch (...) {
#pragma omp critical(tessera_input_state_gradient_eptr)
        if (!ownPending) ownPending = std::current_exception();
      }
    }
    if (ownPending) std::rethrow_exception(ownPending);
    term.active = true;
  }
  if (std::none_of(terms.begin(), terms.end(), [](const BlockTerms &term) { return term.active; })) return gradients;
  // dr = 2 Re(dF . Dcoord), dF = [u^H (dt - dZ_T c) - r t^H dt] / |t|^2, the
  // whole's dZ on this edge shared by every block.
  // The whole's sweep, and the expensive one: dZ on this edge is shared by
  // every block, and each block writes only its own (b, e) slot, so edges
  // distribute across threads with no interaction. Bit-identical to the serial
  // sweep -- nothing here accumulates across iterations.
  assembled.op->warmDerivatives(1);
  std::exception_ptr pending = nullptr;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
  for (std::int64_t ee = 0; ee < static_cast<std::int64_t>(edges.size()); ++ee) {
    try {
      const std::size_t e = static_cast<std::size_t>(ee);
      const Eigen::MatrixXcd dZ =
          chainhodge::BandDerivative::imagesLengthDerivative(*assembled.op, frames, riesz.images, canonical[e]);
      for (std::size_t b = 0; b < blocks.size(); ++b) {
        const BlockTerms &term = terms[b];
        if (!term.active) continue;
        Eigen::VectorXcd dZTc(static_cast<Eigen::Index>(term.idx.size()));
        for (std::size_t i = 0; i < term.idx.size(); ++i)
          dZTc(static_cast<Eigen::Index>(i)) = dZ.row(term.idx[i]) * term.c;
        complexd dF = -term.u.dot(dZTc);
        if (term.targetMotion[e])
          dF += term.u.dot(*term.targetMotion[e]) - term.leak * term.t.dot(*term.targetMotion[e]);
        gradients[b].lengths[static_cast<Eigen::Index>(e)] = packHolomorphic(dF / term.norm);
      }
    } catch (...) {
#pragma omp critical(tessera_input_state_gradient_eptr)
      if (!pending) pending = std::current_exception();
    }
  }
  if (pending) std::rethrow_exception(pending);
  return gradients;
}

MultiCobordism::ResidualGradient MultiCobordism::inputStateResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime, const BoundaryBlock &block) const {
  return inputStateResidualGradientsOn(spacetime, {&block}).front();
}

MultiCobordism::ResidualGradient MultiCobordism::wholeHarmonicResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime, const TwoBodyTarget &target) const {
  const auto edges = spacetime ? spacetime->getEdgeList()->toVector()
                               : std::vector<::tessera::mesh::Edge *>{};
  ResidualGradient gradient;
  gradient.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  // The zero gradient wherever the residual itself refuses: a reading that
  // cannot name a state has no direction either, and a direction invented for
  // it would be worse than none.
  if (!spacetime || metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil) return gradient;
  const Eigen::VectorXcd wanted =
      outputStateTarget_ ? *outputStateTarget_
                         : Eigen::Map<const Eigen::VectorXcd>(target.chi.data(), target.chi.size());
  const double ww = wanted.squaredNorm();
  if (!(ww > 0.0)) return gradient;
  AssembledPencil assembled;
  chainhodge::Contour contour;
  chainhodge::Band band;
  try {
    assembled = PencilLayer::assemble({spacetime});
    if (assembled.dimension() < 1) return gradient;
    contour = PencilLayer::harmonicContour(assembled, 1);
    band = assembled.op->band(1, contour);
  } catch (const std::runtime_error &) {
    return gradient;
  } catch (const std::invalid_argument &) {
    return gradient;
  }
  const auto rank = static_cast<Eigen::Index>(band.rank());
  if (rank != wanted.size()) return gradient;
  std::vector<const BlockMarking *> markings;
  std::vector<complexd> coefficients;
  for (const auto &block : inputBlocks_) {
    if (!block.marking) continue;
    markings.push_back(&*block.marking);
    for (const auto &value : block.marking->coefficients) coefficients.push_back(value);
  }
  if (markings.empty()) return gradient;
  Eigen::Index cycleCount = 0;
  for (const auto *marking : markings) cycleCount += static_cast<Eigen::Index>(marking->rank());
  const bool hasCoefficientOverride = target.inputCoefficients.size() != 0;
  if (!hasCoefficientOverride &&
      cycleCount != static_cast<Eigen::Index>(coefficients.size()))
    return gradient;
  if (hasCoefficientOverride &&
      (target.inputCoefficients.size() != cycleCount ||
       !target.inputCoefficients.allFinite()))
    return gradient;
  // Pi and the inputs, exactly as the residual reads them.
  const auto periodsOf = [&](const Eigen::MatrixXcd &images) {
    Eigen::MatrixXcd out(cycleCount, rank);
    Eigen::Index row = 0;
    for (std::size_t m = 0; m < markings.size(); ++m)
      for (std::size_t cy = 0; cy < markings[m]->rank(); ++cy, ++row)
        for (Eigen::Index a = 0; a < rank; ++a)
          out(row, a) = assembled.op->connection().transportedPeriod(
              images.col(a), markings[m]->cycles[cy]);
    return out;
  };
  Eigen::MatrixXcd Pi;
  Eigen::VectorXcd inputs(cycleCount);
  try {
    Pi = periodsOf(band.images);
  } catch (const std::runtime_error &) {
    return gradient;
  }
  for (Eigen::Index row = 0; row < cycleCount; ++row)
    inputs(row) = hasCoefficientOverride
                      ? target.inputCoefficients(row)
                      : coefficients[static_cast<std::size_t>(row)];
  std::vector<Eigen::Index> markingRanks;
  markingRanks.reserve(markings.size());
  for (const auto *marking : markings)
    markingRanks.push_back(static_cast<Eigen::Index>(marking->rank()));
  const PeriodFrameSelection periodFrame =
      selectPeriodFrame(Pi, markingRanks);
  if (!periodFrame.obstruction.empty()) return gradient;
  const Eigen::MatrixXcd gram = Pi.adjoint() * Pi;
  Eigen::FullPivLU<Eigen::MatrixXcd> gramLu(gram);
  // A singular Gram means the marked cycles do not pin the harmonic form, so
  // the coefficient vector is not a function of the geometry here and has no
  // derivative. The residual still reads (its solve takes the minimum-norm
  // answer); the direction is what is undefined, and it is left at zero.
  if (!gramLu.isInvertible()) return gradient;
  const Eigen::VectorXcd bandState = gramLu.solve(Pi.adjoint() * inputs);
  const Eigen::VectorXcd state = periodFrame.block * bandState;
  const double ss = state.squaredNorm();
  if (!(ss > 0.0)) return gradient;
  const Eigen::VectorXcd fit =
      inputs - Pi * bandState;  // what the periods cannot reach
  const complexd overlap = state.dot(wanted);  // <B c, w>
  // r = 1 - |<B c,w>|^2 / (|B c|^2 |w|^2), the same
  // period-frame expression the residual takes.
  // The derivative along a direction, taken twice -- once along ds = 1 and once
  // along ds = i -- because the packed gradient is the pair
  // (dr/d(Re s), dr/d(Im s)).
  //
  // The (2 Re dF, -2 Im dF) shortcut off a single holomorphic dF is valid only
  // while everything between the coordinate and the residual is holomorphic,
  // and c is not: c = G^-1 Pi* p depends on s-bar through Pi*. Along a real
  // direction both terms are present:
  //   dc = -G^-1 (Pi* dPi) c + G^-1 dPi* (p - Pi c).
  const auto along = [&](const Eigen::MatrixXcd &dZ) {
    const Eigen::MatrixXcd dPi = periodsOf(dZ);
    const Eigen::VectorXcd dBandState = gramLu.solve(
        dPi.adjoint() * fit - (Pi.adjoint() * dPi) * bandState);
    Eigen::MatrixXcd dFrame(rank, rank);
    for (Eigen::Index frameRow = 0; frameRow < rank; ++frameRow)
      dFrame.row(frameRow) =
          dPi.row(periodFrame.rows[static_cast<std::size_t>(frameRow)]);
    const Eigen::VectorXcd dstate =
        dFrame * bandState + periodFrame.block * dBandState;
    const complexd s1 = wanted.dot(dstate);   // <w, dc>
    const complexd s2 = state.dot(dstate);    // <c, dc>
    return 2.0 * (-(overlap * s1 * ss - std::norm(overlap) * s2) /
                  (ss * ss * ww)).real();
  };
  const auto sensitivity = [&](const Eigen::MatrixXcd &dZ) {
    return complexd(along(dZ), along(complexd(0.0, 1.0) * dZ));
  };
  const std::vector<std::size_t> canonical = canonicalEdgeIndices(*spacetime, assembled.complex());
  const chainhodge::BandDerivative::ResolventFrames frames =
      chainhodge::BandDerivative::resolventFrames(*assembled.op, 1, contour, band.frame);
  assembled.op->warmDerivatives(1);
  // The same two-phase, windowed shape the transfer gradient uses: the dense
  // per-edge dZ parallelizes, the cheap sensitivity stays in a plain serial
  // loop so its arithmetic is bit-for-bit the serial sweep's.
  const std::size_t window = 32;
  std::vector<Eigen::MatrixXcd> imageDerivatives(window);
  for (std::size_t base = 0; base < edges.size(); base += window) {
    const std::size_t upper = std::min(base + window, edges.size());
    const auto span = static_cast<std::int64_t>(upper - base);
    std::exception_ptr pending = nullptr;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
    for (std::int64_t i = 0; i < span; ++i) {
      try {
        imageDerivatives[static_cast<std::size_t>(i)] =
            chainhodge::BandDerivative::imagesLengthDerivative(
                *assembled.op, frames, band.images, canonical[base + static_cast<std::size_t>(i)]);
      } catch (...) {
#pragma omp critical(tessera_whole_harmonic_gradient_eptr)
        if (!pending) pending = std::current_exception();
      }
    }
    if (pending) std::rethrow_exception(pending);
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(span); ++slot)
      // Already the packed pair (dr/d(Re s), dr/d(Im s)): `packHolomorphic`
      // would be the shortcut this cannot take.
      gradient.lengths[static_cast<Eigen::Index>(base + slot)] =
          sensitivity(imageDerivatives[slot]);
  }
  return gradient;
}

MultiCobordism::ResidualGradient MultiCobordism::twoBodyResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime, const TwoBodyTarget &target) const {
  const auto [blockA, blockB] = attachedInputBlocks();
  const BoundaryFiber *A = &*blockA->fiber, *B = &*blockB->fiber;
  if (!spacetime) throw std::invalid_argument("MultiCobordism::twoBodyResidualGradientOn: null spacetime");
  const double targetNorm = target.chi.squaredNorm();
  if (!target.chi.allFinite() || !std::isfinite(targetNorm) ||
      !(targetNorm > 0.0))
    throw std::invalid_argument(
        "MultiCobordism::twoBodyResidualGradientOn: the target must be "
        "finite and nonzero");
  const bool framedA = blockA->marking || blockA->frame, framedB = blockB->marking || blockB->frame;
  if (framedA != framedB)
    throw std::logic_error("MultiCobordism: the two-body transfer is read in the blocks' frames only when both "
                           "input blocks carry one (a marking by setInputMarking or a frame by setInputFrame, "
                           "on both or on neither)");
  const bool framed = framedA;
  const auto edges = spacetime->getEdgeList()->toVector();
  ResidualGradient gradient;
  gradient.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  if (A->degree == 0) gradient.phases = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  // The frames in effect (derived live from a marking, or supplied) with the
  // cells they are placed on; a marked block whose frame cannot be derived
  // has no transfer and no direction (its residual is the full leak).
  const TransferOperand operandA = transferOperand(*blockA, spacetime);
  const TransferOperand operandB = transferOperand(*blockB, spacetime);
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  const std::vector<int> ia = PencilLayer::indicesOf(assembled, A->degree, operandA.cells);
  const std::vector<int> ib = PencilLayer::indicesOf(assembled, B->degree, operandB.cells);
  const Eigen::MatrixXcd Atilde = PencilLayer::pencil(assembled, A->degree).A;
  // The (A, B) block of an operator on the whole, in the blocks' frames when
  // both carry one: (Z_A^vee)^T X_AB Z_B, the frames held constant at the
  // evaluation point (the transfer's derivative differentiates the pencil
  // operator; a derived frame's own motion is carried by the block residual's
  // gradient, not here), so d T = (Z_A^vee)^T dA~_AB Z_B.
  const auto block = [&](const Eigen::MatrixXcd &full) {
    Eigen::MatrixXcd out(static_cast<Eigen::Index>(ia.size()), static_cast<Eigen::Index>(ib.size()));
    for (std::size_t i = 0; i < ia.size(); ++i)
      for (std::size_t j = 0; j < ib.size(); ++j)
        out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = full(ia[i], ib[j]);
    if (framed) return Eigen::MatrixXcd(operandA.frame->dualImages.transpose() * out * operandB.frame->images);
    return out;
  };
  const Eigen::MatrixXcd T = block(Atilde);
  if (T.rows() != target.chi.rows() || T.cols() != target.chi.cols())
    throw std::logic_error(
        "MultiCobordism::twoBodyResidualGradientOn: the two-body target is " +
        std::to_string(target.chi.rows()) + "x" +
        std::to_string(target.chi.cols()) +
        " but the attached frames give " + std::to_string(T.rows()) + "x" +
        std::to_string(T.cols()));
  const double tt = T.squaredNorm();
  if (!(tt > 0.0)) return gradient;
  const double cc = targetNorm;
  const complexd overlap = (T.conjugate().cwiseProduct(target.chi)).sum();  // <T, chi>
  const std::vector<std::size_t> canonical = canonicalEdgeIndices(*spacetime, assembled.complex());
  // r = 1 - |<T,chi>|^2 / (|T|^2 |chi|^2). For a holomorphic dT:
  //   d|<T,chi>|^2 = 2 Re( <T,chi> <chi,dT> ),  d|T|^2 = 2 Re( <T,dT> ),
  // so dr = 2 Re(dF · dcoord) with
  //   dF = -( <T,chi> <chi,dT> |T|^2 - |<T,chi>|^2 <T,dT> ) / (|T|^4 |chi|^2).
  const auto sensitivity = [&](const Eigen::MatrixXcd &dA) {
    const Eigen::MatrixXcd dT = block(dA);
    const complexd s1 = (target.chi.conjugate().cwiseProduct(dT)).sum();  // <chi, dT>
    const complexd s2 = (T.conjugate().cwiseProduct(dT)).sum();           // <T, dT>
    return -(overlap * s1 * tt - std::norm(overlap) * s2) / (tt * tt * cc);
  };
  // Two phases. The expensive half is the dense operator derivative per edge,
  // which parallelizes. `sensitivity` is the cheap half and the FP-sensitive
  // one: a Release build links with LTO, so pulling it into an OpenMP-outlined
  // body changes the inliner's choices and with them whether `a*b+c` contracts
  // to an FMA. Keeping it in a serial loop over the edges in order keeps that
  // arithmetic bit-for-bit what the serial sweep produced.
  //
  // The derivatives are held for a bounded window rather than all at once:
  // they are dense n_k x n_k, so one per edge would be O(E n^2) live at the
  // peak.
  assembled.op->warmDerivatives(A->degree);
  const std::size_t window = 32;
  std::vector<Eigen::MatrixXcd> lengthDerivatives(window);
  std::vector<Eigen::MatrixXcd> phaseDerivatives(A->degree == 0 ? window : 0);
  for (std::size_t base = 0; base < edges.size(); base += window) {
    const std::size_t upper = std::min(base + window, edges.size());
    const auto span = static_cast<std::int64_t>(upper - base);
    std::exception_ptr pending = nullptr;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
    for (std::int64_t i = 0; i < span; ++i) {
      try {
        const std::size_t slot = static_cast<std::size_t>(i);
        const std::size_t e = base + slot;
        lengthDerivatives[slot] = chainhodge::BandDerivative::pencilOperatorLengthDerivative(
            *assembled.op, A->degree, canonical[e]);
        if (A->degree == 0)
          phaseDerivatives[slot] = chainhodge::BandDerivative::pencilOperatorPhaseDerivative(
              *assembled.op, A->degree, canonical[e]);
      } catch (...) {
#pragma omp critical(tessera_two_body_gradient_eptr)
        if (!pending) pending = std::current_exception();
      }
    }
    if (pending) std::rethrow_exception(pending);
    for (std::size_t slot = 0; slot < static_cast<std::size_t>(span); ++slot) {
      const std::size_t e = base + slot;
      gradient.lengths[static_cast<Eigen::Index>(e)] =
          packHolomorphic(sensitivity(lengthDerivatives[slot]));
      if (A->degree == 0)
        gradient.phases[static_cast<Eigen::Index>(e)] =
            packHolomorphic(sensitivity(phaseDerivatives[slot]));
    }
  }
  return gradient;
}

MultiCobordism::ResidualGradient
MultiCobordism::selectedTwoBodyResidualGradientOn(
    const std::shared_ptr<Spacetime> &spacetime,
    const TwoBodyTarget &target) const {
  if (!spacetime)
    throw std::invalid_argument(
        "MultiCobordism::selectedTwoBodyResidualGradientOn: null spacetime");
  const auto edgeCount =
      static_cast<Eigen::Index>(spacetime->getEdgeList()->toVector().size());
  ResidualGradient total;
  total.lengths = Eigen::VectorXcd::Zero(edgeCount);
  const auto accumulate = [&](const ResidualGradient &part) {
    if (part.lengths.size() != edgeCount)
      throw std::logic_error(
          "MultiCobordism::selectedTwoBodyResidualGradientOn: a selected reading "
          "returned a gradient with the wrong edge count");
    total.lengths += part.lengths;
    if (part.phases.size() != 0) {
      if (part.phases.size() != edgeCount)
        throw std::logic_error(
            "MultiCobordism::selectedTwoBodyResidualGradientOn: a selected reading "
            "returned a phase gradient with the wrong edge count");
      if (total.phases.size() == 0)
        total.phases = Eigen::VectorXcd::Zero(edgeCount);
      total.phases += part.phases;
    }
  };
  for (const ReadoutMode mode : readoutModes_) {
    if (mode == ReadoutMode::Transfer) {
      try {
        accumulate(twoBodyResidualGradientOn(spacetime, target));
      } catch (const std::runtime_error &) {
        // The transfer value maps a refused geometry to the constant full
        // leak. Its zero direction must not discard another selected read's.
      } catch (const std::invalid_argument &) {
      }
    } else if (mode == ReadoutMode::Whole) {
      try {
        accumulate(wholeHarmonicResidualGradientOn(spacetime, target));
      } catch (const std::runtime_error &) {
        // A read whose derivative refuses contributes no direction; other
        // selected readings remain independently differentiable.
      } catch (const std::invalid_argument &) {
      }
    } else {
      throw std::logic_error(
          "MultiCobordism::selectedTwoBodyResidualGradientOn: the selected bulk and "
          "paired-operator readings have no analytic gradient; use the "
          "numerical objective ascent");
    }
  }
  return total;
}

bool MultiCobordism::readoutsHaveAnalyticGradient() const noexcept {
  // Transfer and whole-harmonic reads have their own derivatives. Anything
  // else falls back to the numerical ascent of the selected objective rather
  // than borrowing another reading's direction.
  bool usesTransfer = false;
  for (const ReadoutMode mode : readoutModes_)
    if (mode == ReadoutMode::Transfer)
      usesTransfer = true;
    else if (mode != ReadoutMode::Whole)
      return false;
  // The analytic transfer derivative below is defined for exactly two input
  // blocks. Four attached blocks use the paired-frame read and must take the
  // objective's numerical fallback rather than entering that two-block path.
  if (!usesTransfer || (!twoBodyTarget_ && twoBodyCases_.empty())) return true;
  std::size_t attached = 0;
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0) ++attached;
  return attached == 2;
}

MultiCobordism::ResidualGradient MultiCobordism::fiberModeAscent() const {
  if ((twoBodyTarget_ || !twoBodyCases_.empty()) &&
      !readoutsHaveAnalyticGradient())
    throw std::logic_error(
        "MultiCobordism::fiberModeAscent: a selected two-body reading has no "
        "analytic gradient; use the numerical objective ascent");
  const auto edges = spacetime_->getEdgeList()->toVector();
  ResidualGradient total;
  total.lengths = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  total.phases = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(edges.size()));
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> parentIndex;
  for (std::size_t e = 0; e < edges.size(); ++e) parentIndex[edgeKey(edges[e])] = e;
  const auto accumulate = [&](const ResidualGradient &g, const Spacetime &on, double weight) {
    const auto onEdges = on.getEdgeList()->toVector();
    for (std::size_t e = 0; e < onEdges.size(); ++e) {
      const auto found = parentIndex.find(edgeKey(onEdges[e]));
      if (found == parentIndex.end()) continue;
      total.lengths[static_cast<Eigen::Index>(found->second)] += weight * g.lengths[static_cast<Eigen::Index>(e)];
      if (g.phases.size() == static_cast<Eigen::Index>(onEdges.size()))
        total.phases[static_cast<Eigen::Index>(found->second)] += weight * g.phases[static_cast<Eigen::Index>(e)];
    }
  };
  if (wholeFiberTarget_) accumulate(fiberResidualGradientOn(spacetime_, *wholeFiberTarget_), *spacetime_, 1.0);
  // A marked block's term is its own-state residual: the analytic
  // tau derivative of the holomorphic form of its own Laplacian, supported
  // on the block's own edges (already in the host's edge order; a refused
  // read has no direction and comes back as the zero gradient).
  for (const auto &block : inputBlocks_) {
    if (!block.marking) continue;
    accumulate(ownStateResidualGradientOn(spacetime_, block), *spacetime_, inputResidualWeight_);
  }
  // The whole's leak for every marked block at once: `inputStateResidualGradientsOn`
  // computes the whole's band derivative — the per-edge cost, and the reason
  // this is not a loop over the single-block call — once and shares it.
  if (scoreWholeComplexLeak_) {
    std::vector<const BoundaryBlock *> marked;
    for (const auto &block : inputBlocks_)
      if (block.marking) marked.push_back(&block);
    if (!marked.empty()) {
      try {
        auto leaks = inputStateResidualGradientsOn(spacetime_, marked);
        for (auto &leak : leaks)
          accumulate(leak, *spacetime_, inputResidualWeight_);
      } catch (const std::runtime_error &) {
        // a refused geometry has no descent direction (its leak is the full 1.0)
      } catch (const std::invalid_argument &) {
      } catch (const std::logic_error &) {
      }
    }
  }
  for (const auto &block : inputBlocks_) {
    if (block.marking) continue;
    if (!block.fiber || block.fiber->images.cols() == 0) continue;
    auto sub = blockComplexWithGeometry(block, spacetime_);
    if (!sub) continue;
    try {
      accumulate(fiberResidualGradientOn(sub, *block.fiber, fiberBandFor(block)), *sub, inputResidualWeight_);
    } catch (const std::runtime_error &) {
      // a refused geometry has no descent direction (its residual is the full leak)
    } catch (const std::invalid_argument &) {
    }
  }
  if (!twoBodyCases_.empty()) {
    // One gradient per case, each taken with that case's boundary written.
    // Summed because the objective is a sum; the boundary components are
    // zeroed by pinning either way, and the bulk components are what differ
    // between cases -- which is the whole point.
    for (const auto &boundaryCase : twoBodyCases_) {
      const auto previous = writeCaseBoundary(boundaryCase, spacetime_);
      try {
        accumulate(selectedTwoBodyResidualGradientOn(
                       spacetime_, TwoBodyTarget{boundaryCase.chi,
                                                 boundaryCase.choiDecomposed,
                                                 boundaryCase.twoStateVector,
                                                 boundaryCase.inputCoefficients}),
                   *spacetime_, 1.0);
      } catch (const std::runtime_error &) {
      } catch (const std::invalid_argument &) {
      } catch (...) {
        restoreCaseBoundary(previous, spacetime_);
        throw;
      }
      restoreCaseBoundary(previous, spacetime_);
    }
  } else if (twoBodyTarget_) {
    try {
      accumulate(selectedTwoBodyResidualGradientOn(spacetime_, *twoBodyTarget_),
                 *spacetime_, 1.0);
    } catch (const std::runtime_error &) {
    } catch (const std::invalid_argument &) {
    }
  }
  return total;
}

}  // namespace tessera::cobordism
