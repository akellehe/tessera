// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the two-body
// cobordism map and the readouts: transfer shapes and operands, frame
// transfers and the whole-harmonic residual. One of the translation units
// that define the class's members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "MultiCobordismInternal.h"

namespace tessera::cobordism {

std::optional<MultiCobordism::TransferShape> MultiCobordism::transferShape() const {
  std::vector<const BoundaryBlock *> attached;
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0) attached.push_back(&block);
  if (attached.size() != 2) return std::nullopt;
  const BoundaryBlock &A = *attached[0], &B = *attached[1];
  // The frame in effect: a marking's derived frame has the marking's rank,
  // a supplied frame its column count.
  const auto rankOf = [](const BoundaryBlock &block) -> std::optional<Eigen::Index> {
    if (block.marking) return static_cast<Eigen::Index>(block.marking->rank());
    if (block.frame) return block.frame->images.cols();
    return std::nullopt;
  };
  const auto rA = rankOf(A), rB = rankOf(B);
  if (rA && rB) return TransferShape{*rA, *rB, true};
  if (!rA && !rB)
    return TransferShape{static_cast<Eigen::Index>(A.fiber->cells.size()),
                         static_cast<Eigen::Index>(B.fiber->cells.size()), false};
  return std::nullopt;
}

std::optional<MultiCobordism::TransferShape>
MultiCobordism::pairedTransferShape() const {
  std::vector<const BoundaryBlock *> attached;
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0)
      attached.push_back(&block);
  if (attached.size() < 2 || attached.size() % 2 != 0)
    return std::nullopt;
  const auto framed = [](const BoundaryBlock *block) {
    return block->marking.has_value() || block->frame.has_value();
  };
  const bool anyFramed =
      std::any_of(attached.begin(), attached.end(), framed);
  const bool allFramed =
      std::all_of(attached.begin(), attached.end(), framed);
  if (anyFramed != allFramed) return std::nullopt;
  const auto rank = [allFramed](const BoundaryBlock *block) {
    if (!allFramed)
      return static_cast<Eigen::Index>(block->fiber->cells.size());
    if (block->marking)
      return static_cast<Eigen::Index>(block->marking->rank());
    return block->frame->images.cols();
  };
  const std::size_t half = attached.size() / 2;
  Eigen::Index rows = 0, cols = 0;
  for (std::size_t index = 0; index < half; ++index)
    rows += rank(attached[index]);
  for (std::size_t index = half; index < attached.size(); ++index)
    cols += rank(attached[index]);
  return TransferShape{rows, cols, allFramed};
}

void MultiCobordism::setTwoBodyTarget(Eigen::MatrixXcd chi, bool choiDecomposed) {
  const double norm = chi.squaredNorm();
  if (chi.rows() == 0 || chi.cols() == 0 || !chi.allFinite() ||
      !std::isfinite(norm) || !(norm > 0.0))
    throw std::invalid_argument(
        "MultiCobordism::setTwoBodyTarget: the target must be a finite, "
        "nonzero matrix");
  if (const auto shape = transferShape(); shape && (chi.rows() != shape->rows || chi.cols() != shape->cols))
    throw std::invalid_argument("MultiCobordism::setTwoBodyTarget: the target is " + std::to_string(chi.rows()) + "x" +
                                std::to_string(chi.cols()) + " but the attached " +
                                (shape->framed ? "frames give " : "cells give ") + std::to_string(shape->rows) +
                                "x" + std::to_string(shape->cols));
  for (std::size_t index = 0; index < twoBodyCases_.size(); ++index)
    if (twoBodyCases_[index].chi.rows() != chi.rows() ||
        twoBodyCases_[index].chi.cols() != chi.cols())
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyTarget: case " +
          std::to_string(index) + " target is " +
          std::to_string(twoBodyCases_[index].chi.rows()) + "x" +
          std::to_string(twoBodyCases_[index].chi.cols()) +
          " but the single target is " + std::to_string(chi.rows()) + "x" +
          std::to_string(chi.cols()));
  twoBodyTarget_ = TwoBodyTarget{std::move(chi), choiDecomposed};
}

void MultiCobordism::setTwoBodyCases(std::vector<TwoBodyCase> cases) {
  Eigen::Index coefficientCount = 0;
  for (const auto &block : inputBlocks_)
    if (block.marking)
      coefficientCount += static_cast<Eigen::Index>(block.marking->rank());
  const auto pairedShape = pairedTransferShape();
  std::optional<std::pair<Eigen::Index, Eigen::Index>> casePairedShape;
  std::size_t firstPairedCase = 0;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto &boundaryCase = cases[index];
    if (boundaryCase.chi.size() == 0)
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyCases: case " +
          std::to_string(index) + " has an empty target");
    const double targetNorm = boundaryCase.chi.squaredNorm();
    if (!boundaryCase.chi.allFinite() || !std::isfinite(targetNorm) ||
        !(targetNorm > 0.0))
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyCases: case " +
          std::to_string(index) +
          " must have a finite, nonzero target");
    if (twoBodyTarget_ &&
        (boundaryCase.chi.rows() != twoBodyTarget_->chi.rows() ||
         boundaryCase.chi.cols() != twoBodyTarget_->chi.cols()))
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyCases: case " +
          std::to_string(index) + " target is " +
          std::to_string(boundaryCase.chi.rows()) + "x" +
          std::to_string(boundaryCase.chi.cols()) +
          " but the single target is " +
          std::to_string(twoBodyTarget_->chi.rows()) + "x" +
          std::to_string(twoBodyTarget_->chi.cols()));
    if (index > 0 &&
        (boundaryCase.chi.rows() != cases.front().chi.rows() ||
         boundaryCase.chi.cols() != cases.front().chi.cols()))
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyCases: case " +
          std::to_string(index) + " target is " +
          std::to_string(boundaryCase.chi.rows()) + "x" +
          std::to_string(boundaryCase.chi.cols()) +
          " but case 0 target is " +
          std::to_string(cases.front().chi.rows()) + "x" +
          std::to_string(cases.front().chi.cols()));
    if (const auto shape = transferShape();
        shape && (boundaryCase.chi.rows() != shape->rows ||
                  boundaryCase.chi.cols() != shape->cols))
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyCases: case " +
          std::to_string(index) + " target is " +
          std::to_string(boundaryCase.chi.rows()) + "x" +
          std::to_string(boundaryCase.chi.cols()) +
          " but the attached transfer frames give " +
          std::to_string(shape->rows) + "x" +
          std::to_string(shape->cols));
    if (boundaryCase.twoStateVector.size() != 0) {
      const double stateNorm = boundaryCase.twoStateVector.squaredNorm();
      if (!boundaryCase.twoStateVector.allFinite() ||
          !std::isfinite(stateNorm) || !(stateNorm > 0.0))
        throw std::invalid_argument(
            "MultiCobordism::setTwoBodyCases: case " +
            std::to_string(index) +
            " must have a finite, nonzero paired direct-sum target");
      if (pairedShape &&
          (boundaryCase.twoStateVector.rows() != pairedShape->rows ||
           boundaryCase.twoStateVector.cols() != pairedShape->cols))
        throw std::invalid_argument(
            "MultiCobordism::setTwoBodyCases: case " +
            std::to_string(index) + " paired direct-sum target is " +
            std::to_string(boundaryCase.twoStateVector.rows()) + "x" +
            std::to_string(boundaryCase.twoStateVector.cols()) +
            " but the attached paired frames give " +
            std::to_string(pairedShape->rows) + "x" +
            std::to_string(pairedShape->cols));
      const auto shape = std::make_pair(boundaryCase.twoStateVector.rows(),
                                        boundaryCase.twoStateVector.cols());
      if (casePairedShape && shape != *casePairedShape)
        throw std::invalid_argument(
            "MultiCobordism::setTwoBodyCases: case " +
            std::to_string(index) + " paired direct-sum target is " +
            std::to_string(shape.first) + "x" +
            std::to_string(shape.second) + " but case " +
            std::to_string(firstPairedCase) + " is " +
            std::to_string(casePairedShape->first) + "x" +
            std::to_string(casePairedShape->second));
      if (!casePairedShape) {
        casePairedShape = shape;
        firstPairedCase = index;
      }
    }
    std::set<std::pair<std::uint64_t, std::uint64_t>> boundaryEdges;
    for (const auto &[endpoints, squaredLength] : boundaryCase.boundary) {
      if (!std::isfinite(squaredLength.real()) ||
          !std::isfinite(squaredLength.imag()))
        throw std::invalid_argument(
            "MultiCobordism::setTwoBodyCases: case " +
            std::to_string(index) + " gives boundary edge (" +
            std::to_string(endpoints.first) + ", " +
            std::to_string(endpoints.second) +
            ") a non-finite squared length");
      const auto canonical = std::minmax(endpoints.first, endpoints.second);
      if (!boundaryEdges.emplace(canonical.first, canonical.second).second)
        throw std::invalid_argument(
            "MultiCobordism::setTwoBodyCases: case " +
            std::to_string(index) + " repeats boundary edge (" +
            std::to_string(canonical.first) + ", " +
            std::to_string(canonical.second) + ")");
    }
    if (boundaryCase.inputCoefficients.size() != 0 &&
        boundaryCase.inputCoefficients.size() != coefficientCount)
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyCases: case " + std::to_string(index) +
          " has " + std::to_string(boundaryCase.inputCoefficients.size()) +
          " input coefficients, but the marked input blocks have " +
          std::to_string(coefficientCount) + " cycles");
    if (!boundaryCase.inputCoefficients.allFinite())
      throw std::invalid_argument(
          "MultiCobordism::setTwoBodyCases: case " + std::to_string(index) +
          " has a non-finite input coefficient");
    if (boundaryCase.inputCoefficients.size() != 0) {
      const double inputNorm = boundaryCase.inputCoefficients.squaredNorm();
      if (!std::isfinite(inputNorm) || !(inputNorm > 0.0))
        throw std::invalid_argument(
            "MultiCobordism::setTwoBodyCases: case " +
            std::to_string(index) +
            " has all-zero input coefficients: no state");
    }
  }
  twoBodyCases_ = std::move(cases);
}

std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>>
MultiCobordism::writeCaseBoundary(const TwoBodyCase &boundaryCase,
                                  const std::shared_ptr<Spacetime> &spacetime) const {
  std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>>> previous;
  if (!spacetime || !spacetime->getEdgeList()) return previous;
  previous.reserve(boundaryCase.boundary.size());
  for (const auto &[endpoints, squaredLength] : boundaryCase.boundary) {
    const ::tessera::mesh::EdgeKey key(endpoints.first, endpoints.second);
    auto *edge = spacetime->getEdgeList()->get(key.fingerprint.fingerprint());
    // A case naming an edge the complex does not have is skipped, not an
    // error: stage 1 rebuilds the complex between evaluations, so a boundary
    // edge is always present but a stale case would otherwise abort a drive.
    if (edge == nullptr) continue;
    const auto length = edge->getLength();
    previous.emplace_back(endpoints, length);
    edge->setLength(continuousSquareRoot(squaredLength, length));
  }
  return previous;
}

void MultiCobordism::restoreCaseBoundary(
    const std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>,
                                std::complex<double>>> &lengths,
    const std::shared_ptr<Spacetime> &spacetime) const {
  if (!spacetime || !spacetime->getEdgeList()) return;
  for (const auto &[endpoints, length] : lengths) {
    const ::tessera::mesh::EdgeKey key(endpoints.first, endpoints.second);
    if (auto *edge = spacetime->getEdgeList()->get(
            key.fingerprint.fingerprint()))
      edge->setLength(length);
  }
}

double MultiCobordism::twoBodyResidualOverCasesOn(
    const std::shared_ptr<Spacetime> &spacetime) const {
  if (twoBodyCases_.empty())
    return twoBodyTarget_ ? twoBodyResidualOn(spacetime, *twoBodyTarget_) : 0.0;
  // The sum of the per-case reads, so the number the drive minimises and the
  // numbers a reader inspects cannot disagree.
  double total = 0.0;
  for (const double residual : twoBodyResidualsPerCaseOn(spacetime))
    total += residual;
  return total;
}

std::vector<double> MultiCobordism::twoBodyResidualsPerCaseOn(
    const std::shared_ptr<Spacetime> &spacetime) const {
  std::vector<double> residuals;
  residuals.reserve(twoBodyCases_.size());
  for (const auto &boundaryCase : twoBodyCases_) {
    const auto previous = writeCaseBoundary(boundaryCase, spacetime);
    // Restored even when a read throws: a half-written boundary would be
    // scored by every later case and by whatever the caller does next.
    try {
      residuals.push_back(twoBodyResidualOn(
          spacetime, TwoBodyTarget{boundaryCase.chi, boundaryCase.choiDecomposed,
                                   boundaryCase.twoStateVector,
                                   boundaryCase.inputCoefficients}));
    } catch (...) {
      restoreCaseBoundary(previous, spacetime);
      throw;
    }
    restoreCaseBoundary(previous, spacetime);
  }
  return residuals;
}

std::pair<const MultiCobordism::BoundaryBlock *, const MultiCobordism::BoundaryBlock *>
MultiCobordism::attachedInputBlocks() const {
  std::vector<const BoundaryBlock *> attached;
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0) attached.push_back(&block);
  if (attached.size() != 2)
    throw std::logic_error("MultiCobordism: the two-body map needs exactly two attached input fibers; " +
                           std::to_string(attached.size()) + " found");
  return {attached[0], attached[1]};
}

chainhodge::TransferResult MultiCobordism::frameTransferOn(const std::shared_ptr<Spacetime> &spacetime,
                                                            const BoundaryBlock &A,
                                                            const BoundaryBlock &B) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism: the two-body map is read on the chain-level Whitney pencil; "
                           "this node uses the diagonal-weight metric");
  if (A.fiber->degree != B.fiber->degree)
    throw std::logic_error("MultiCobordism: the two attached fibers are at different degrees");
  const bool framedA = A.marking || A.frame, framedB = B.marking || B.frame;
  if (framedA != framedB)
    throw std::logic_error("MultiCobordism: the two-body transfer is read in the blocks' frames only when both "
                           "input blocks carry one (a marking by setInputMarking or a frame by setInputFrame, "
                           "on both or on neither)");
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  // Without frames, the full frames on the attached cells: unit images (and
  // unit dual images), so the transfer is the coupling block of the whole
  // between the two cell sets. With frames, the blocks' images and dual images
  // on the frames' cells (BlockFrame: derived live from a marking, or
  // supplied), so the transfer is that block in the frames' coordinates.
  auto frame = [&](const BoundaryBlock &block) {
    const TransferOperand operand = transferOperand(block, spacetime);
    BoundaryFiber out;
    out.degree = block.fiber->degree;
    out.cells = operand.cells;
    if (operand.frame) {
      if (static_cast<std::size_t>(operand.frame->images.rows()) != operand.cells.size())
        throw std::logic_error("MultiCobordism: a block's frame has a row count other than its cells");
      out.images = operand.frame->images;
      out.dualImages = operand.frame->dualImages;
      return out;
    }
    const Eigen::Index r = static_cast<Eigen::Index>(operand.cells.size());
    out.images = Eigen::MatrixXcd::Identity(r, r);
    out.dualImages = Eigen::MatrixXcd::Identity(r, r);
    return out;
  };
  return PencilLayer::transfer(assembled, A.fiber->degree, frame(A), frame(B));
}

chainhodge::TransferResult MultiCobordism::pairedFrameTransferOn(
    const std::shared_ptr<Spacetime> &spacetime,
    const std::vector<const BoundaryBlock *> &sideA,
    const std::vector<const BoundaryBlock *> &sideB) const {
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    throw std::logic_error("MultiCobordism: the paired-frame transfer is read on the chain-level Whitney "
                           "pencil; this node uses the diagonal-weight metric");
  if (sideA.empty() || sideB.empty())
    throw std::logic_error("MultiCobordism: a paired frame needs at least one block a side");
  const auto framed = [](const BoundaryBlock *block) {
    return block->marking.has_value() || block->frame.has_value();
  };
  const bool anyFramed =
      std::any_of(sideA.begin(), sideA.end(), framed) ||
      std::any_of(sideB.begin(), sideB.end(), framed);
  const bool allFramed =
      std::all_of(sideA.begin(), sideA.end(), framed) &&
      std::all_of(sideB.begin(), sideB.end(), framed);
  if (anyFramed != allFramed)
    throw std::logic_error(
        "MultiCobordism: the paired transfer is read in block frames only "
        "when every attached block carries one; partial framing is ambiguous");
  const AssembledPencil assembled = PencilLayer::assemble({spacetime});
  // One side's frame is its blocks' frames stacked: cells concatenated, images
  // and dual images block-diagonal. A torus and its conjugate carry disjoint
  // cells and their own kernels, so the combined frame is their direct sum --
  // nothing is mixed here that the geometry does not already separate.
  auto side = [&](const std::vector<const BoundaryBlock *> &blocks) {
    BoundaryFiber out;
    out.degree = blocks.front()->fiber->degree;
    Eigen::Index rows = 0, cols = 0;
    std::vector<BlockFrame> frames;
    for (const auto *block : blocks) {
      if (block->fiber->degree != out.degree)
        throw std::logic_error("MultiCobordism: a paired frame's blocks are at different degrees");
      const TransferOperand operand = transferOperand(*block, spacetime);
      BlockFrame frame;
      if (operand.frame) {
        frame = *operand.frame;
      } else {
        const auto r = static_cast<Eigen::Index>(operand.cells.size());
        frame.cells = operand.cells;
        frame.images = Eigen::MatrixXcd::Identity(r, r);
        frame.dualImages = Eigen::MatrixXcd::Identity(r, r);
      }
      if (static_cast<std::size_t>(frame.images.rows()) != operand.cells.size())
        throw std::logic_error("MultiCobordism: a block's frame has a row count other than its cells");
      out.cells.insert(out.cells.end(), operand.cells.begin(), operand.cells.end());
      rows += frame.images.rows();
      cols += frame.images.cols();
      frames.push_back(std::move(frame));
    }
    out.images = Eigen::MatrixXcd::Zero(rows, cols);
    out.dualImages = Eigen::MatrixXcd::Zero(rows, cols);
    Eigen::Index row = 0, col = 0;
    for (const auto &frame : frames) {
      out.images.block(row, col, frame.images.rows(), frame.images.cols()) = frame.images;
      out.dualImages.block(row, col, frame.dualImages.rows(), frame.dualImages.cols()) = frame.dualImages;
      row += frame.images.rows();
      col += frame.images.cols();
    }
    return out;
  };
  return PencilLayer::transfer(assembled, sideA.front()->fiber->degree, side(sideA), side(sideB));
}

MultiCobordism::TransferOperand MultiCobordism::transferOperand(const BoundaryBlock &block,
                                                                const std::shared_ptr<Spacetime> &spacetime) const {
  TransferOperand operand;
  if (block.marking) {
    DerivedFrame derived = deriveFrame(block, spacetime);
    if (!derived.derived())
      throw std::runtime_error("MultiCobordism: the frame of a marked block could not be derived: " +
                               derived.obstruction);
    operand.cells = derived.frame.cells;
    operand.frame = std::move(derived.frame);
    operand.derived = true;
    return operand;
  }
  operand.cells = block.fiber->cells;
  if (block.frame) operand.frame = block.frame;
  return operand;
}

double MultiCobordism::bulkOperatorResidualOn(
    const std::shared_ptr<Spacetime> &spacetime,
    const TwoBodyTarget &target) const {
  if (!spacetime) return 1.0;
  const double targetNorm = target.chi.squaredNorm();
  if (!target.chi.allFinite() || !std::isfinite(targetNorm) ||
      !(targetNorm > 0.0))
    throw std::invalid_argument(
        "MultiCobordism::bulkOperatorResidualOn: the target must be finite "
        "and nonzero");
  const Eigen::Index dimension = target.chi.rows();
  if (dimension < 1 || target.chi.cols() != dimension)
    throw std::logic_error("MultiCobordism::bulkOperatorResidualOn: the target is " +
                           std::to_string(target.chi.rows()) + "x" + std::to_string(target.chi.cols()) +
                           ", not square");
  // The operator the bulk names, read through a Choi frame of d^2 interior
  // edges in canonical order. A complex whose framed kernel is not rank one
  // names no operator: it scores the full leak, exactly as a refused geometry
  // does under the transfer reading, rather than a number standing in for one.
  GeometricOperatorReadout readout;
  try {
    // Metric=true: the live signed Hodge weights, so the reading retains what
    // relaxation did. The combinatorial unit-weight mode is topology-only and
    // would score the same for every geometry with the same cells, making the
    // term a constant under stage 2.
    readout = geometricOperatorOn(spacetime, static_cast<int>(dimension), {},
                                  /*tol=*/1e-9, /*metric=*/true);
  } catch (const std::runtime_error &) {
    return 1.0;
  } catch (const std::invalid_argument &) {
    return 1.0;
  }
  if (!readout.identifiable ||
      readout.choiState.size() != static_cast<std::size_t>(target.chi.size()))
    return 1.0;
  const Eigen::Map<const Eigen::VectorXcd> choi(readout.choiState.data(),
                                                static_cast<Eigen::Index>(readout.choiState.size()));
  const Eigen::Map<const Eigen::VectorXcd> chi(target.chi.data(), target.chi.size());
  const double cc = choi.squaredNorm();
  if (!(cc > 0.0)) return 1.0;
  // The same projective Frobenius leak the transfer reading takes, so the two
  // readings are on one scale and a selected readout set may sum them.
  const complexd overlap = choi.dot(chi);
  const double leak = chi.squaredNorm() - std::norm(overlap) / cc;
  return std::max(0.0, leak / chi.squaredNorm());
}

void MultiCobordism::setWholePairing(WholePairing pairing) { wholePairing_ = pairing; }

void MultiCobordism::setReadoutModes(std::vector<ReadoutMode> modes) {
  if (modes.empty())
    throw std::invalid_argument(
        "MultiCobordism::setReadoutModes: at least one reading is required; a "
        "two-body term scored against nothing is not a term");
  std::vector<ReadoutMode> unique;
  unique.reserve(modes.size());
  for (const ReadoutMode mode : modes)
    if (std::find(unique.begin(), unique.end(), mode) == unique.end())
      unique.push_back(mode);
  readoutModes_ = std::move(unique);
}

double MultiCobordism::wholeHarmonicResidualOn(
    const std::shared_ptr<Spacetime> &spacetime,
    const TwoBodyTarget &target) const {
  wholeHarmonicObstruction_.clear();
  const auto refuse = [&](std::string reason) {
    wholeHarmonicObstruction_ = std::move(reason);
    return 1.0;
  };
  if (!spacetime) return refuse("no complex to read");
  // The same guard the transfer and paired readings take: the harmonic band is
  // the chain-level Whitney pencil's, so a node configured for diagonal
  // weights would otherwise be read through a metric it did not ask for.
  if (metricSource_ != HodgeLaplacian::MetricSource::WhitneyPencil)
    return refuse("the whole-complex harmonic is read on the chain-level Whitney pencil; this node uses "
                  "the diagonal-weight metric");
  // The declared output state when one is set, otherwise the two-body target.
  // A rank-2 harmonic space carries a 2-dimensional state and a rank-4 one a
  // 4-dimensional state; forcing the 4-dimensional chi on a two-torus host was
  // asking the wrong question of it.
  const Eigen::VectorXcd wanted =
      outputStateTarget_
          ? *outputStateTarget_
          : Eigen::Map<const Eigen::VectorXcd>(target.chi.data(), target.chi.size());
  const double wantedNorm = wanted.squaredNorm();
  if (!wanted.allFinite() || !std::isfinite(wantedNorm) ||
      !(wantedNorm > 0.0))
    throw std::invalid_argument(
        "MultiCobordism::wholeHarmonicResidualOn: the target state must be "
        "finite and nonzero");
  const Eigen::Index dimension = wanted.size();
  chainhodge::Band band;
  AssembledPencil assembled;
  try {
    assembled = PencilLayer::assemble({spacetime});
    if (assembled.dimension() < 1) return refuse("the complex has no edges");
    // The lambda = 0 band as a null space rather than a contour integral around
    // it: the two are one subspace under the rank conditions, and
    // `harmonicBand` refuses when the dual connection disagrees about its
    // dimension, which is how those conditions fail. The null-space route is
    // far cheaper.
    //
    // This substitution is safe only because the reading below is taken in the
    // period frame: a Riesz band and a null-space band return different bases
    // of the same space, so read in the band's own basis the two disagree.
    band = assembled.op->harmonicBand(1);
  } catch (const std::runtime_error &error) {
    return refuse(error.what());
  } catch (const std::invalid_argument &error) {
    return refuse(error.what());
  }
  const auto rank = static_cast<Eigen::Index>(band.rank());
  // The harmonic space is a space. Its rank is what can be carried, and a
  // target of another dimension is not something this geometry has a state
  // for -- said rather than fitted. Two boundary tori give b_1 = 2 against a
  // 4-dimensional target; four give b_1(dW) = 8, hence rank 4.
  if (rank != dimension)
    return refuse("the whole complex's degree-1 harmonic space has rank " +
                  std::to_string(rank) + " for a target of dimension " +
                  std::to_string(dimension));
  // Pi_{ca}: the transported period of harmonic column a over marked cycle c,
  // every marking every input block carries. The markings are the only input:
  // no basis is named here that the geometry does not already carry.
  std::vector<const BlockMarking *> markings;
  std::vector<complexd> coefficients;
  // The marking's own coefficients, which are the block's input state (1, tau)
  // -- `readInputState` reads the same field. `block.target` is the register
  // target and is a different quantity.
  for (const auto &block : inputBlocks_) {
    if (!block.marking) continue;
    markings.push_back(&*block.marking);
    for (const auto &value : block.marking->coefficients)
      coefficients.push_back(value);
  }
  if (markings.empty()) return refuse("no input block carries a marking");
  Eigen::Index cycleCount = 0;
  for (const auto *marking : markings)
    cycleCount += static_cast<Eigen::Index>(marking->rank());
  const bool hasCoefficientOverride = target.inputCoefficients.size() != 0;
  if (!hasCoefficientOverride &&
      cycleCount != static_cast<Eigen::Index>(coefficients.size()))
    return refuse("the marked cycles and the input coefficients differ in count");
  if (hasCoefficientOverride &&
      target.inputCoefficients.size() != cycleCount)
    return refuse("the case has " +
                  std::to_string(target.inputCoefficients.size()) +
                  " input coefficients for " + std::to_string(cycleCount) +
                  " marked cycles");
  if (hasCoefficientOverride && !target.inputCoefficients.allFinite())
    return refuse("the case has a non-finite input coefficient");
  Eigen::MatrixXcd periods(cycleCount, rank);
  Eigen::VectorXcd inputs(cycleCount);
  Eigen::Index row = 0;
  // Under gram the harmonic columns are contracted against the block's live
  // frame through the chain metric rather than integrated over its cycles.
  // M_1 Z is the same for every block, so it is applied once.
  Eigen::MatrixXcd metricImages;
  if (wholePairing_ == WholePairing::Gram)
    metricImages = assembled.op->applyMinv(1, band.images);   // M_1 Z
  try {
    row = 0;
    std::size_t markingIndex = 0;
    for (const auto &block : inputBlocks_) {
      if (!block.marking) continue;
      const BlockMarking &marking = *markings[markingIndex++];
      if (wholePairing_ == WholePairing::Periods) {
        for (std::size_t c = 0; c < marking.rank(); ++c, ++row) {
          for (Eigen::Index a = 0; a < rank; ++a)
            periods(row, a) = assembled.op->connection().transportedPeriod(
                band.images.col(a), marking.cycles[c]);
          inputs(row) = hasCoefficientOverride
                            ? target.inputCoefficients(row)
                            : coefficients[static_cast<std::size_t>(row)];
        }
        continue;
      }
      // The block's live frame, on the host's edges. Its columns are the
      // probes f_c; the pairing is f_c^T M_1 Z_a, the transpose pairing the
      // harmonic Gram Z^T M_1 Z already uses.
      const DerivedFrame derived = deriveFrame(block, spacetime);
      if (!derived.derived())
        return refuse("the gram pairing needs the block's live frame: " + derived.obstruction);
      const auto width = static_cast<Eigen::Index>(derived.frame.rank());
      if (width != static_cast<Eigen::Index>(marking.rank()))
        return refuse("the block's live frame has rank " + std::to_string(width) +
                      " against a marking of rank " + std::to_string(marking.rank()) +
                      ", so the gram pairing has no square block to fit");
      const std::vector<int> idx = PencilLayer::indicesOf(assembled, 1, derived.frame.cells);
      Eigen::MatrixXcd probes = Eigen::MatrixXcd::Zero(band.images.rows(), width);
      for (std::size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0) return refuse("a block's frame cell is no edge of the whole complex");
        probes.row(idx[i]) = derived.frame.images.row(static_cast<Eigen::Index>(i));
      }
      const Eigen::MatrixXcd paired = probes.transpose() * metricImages;   // f^T M_1 Z
      for (Eigen::Index c = 0; c < width; ++c, ++row) {
        periods.row(row) = paired.row(c);
        inputs(row) = hasCoefficientOverride
                          ? target.inputCoefficients(row)
                          : coefficients[static_cast<std::size_t>(row)];
      }
    }
  } catch (const std::runtime_error &error) {
    return refuse(std::string("a marked cycle is not a walk on the whole complex: ") + error.what());
  }
  // The form the inputs determine: the coefficient vector minimizing
  // ||Pi c - p||. The form is basis-free, but this c is its coordinates in
  // whatever basis the band returned, which the geometry does not carry.
  //
  // The frame is the period frame (`SimplicialQubit::periodFrame`): the band
  // normalized so a marking's cycles read periods (1, 0) and (0, 1), in which
  // the coefficients are the periods of the form. Writing B for a marking's
  // square block of Pi, that frame is Z B^-1, whose period matrix is Pi B^-1,
  // so the coefficients there are simply B c, and the band is never rebuilt.
  //
  // Every marking of the band's rank induces such a frame, and they are one
  // frame exactly when the monodromy B_y B_x^-1 between them is the identity.
  // A monodromy that is not the identity means the markings disagree about the
  // frame of the whole, which is reported rather than settled by taking the
  // first one.
  Eigen::VectorXcd state = periods.completeOrthogonalDecomposition().solve(inputs);
  // A group of markings whose cycles number exactly the harmonic rank frames
  // the whole: its square block of Pi is the period matrix, and the band
  // normalized by its inverse reads periods (1,0,...), (0,1,...) there. One
  // torus frames a rank-2 space by itself; a rank-4 space needs two.
  //
  // Which two is not a free choice: a torus and its orientation reversal carry
  // dependent periods, so a conjugate pair frames nothing. The groups are
  // therefore enumerated and the singular ones drop out by conditioning.
  std::vector<Eigen::Index> markingRanks;
  markingRanks.reserve(markings.size());
  for (const auto *marking : markings)
    markingRanks.push_back(static_cast<Eigen::Index>(marking->rank()));
  PeriodFrameSelection periodFrame =
      selectPeriodFrame(periods, markingRanks,
                        wholePairing_ == WholePairing::Periods);
  if (!periodFrame.obstruction.empty())
    return refuse(std::move(periodFrame.obstruction));
  state = periodFrame.block * state;  // the coordinates in the period frame
  const double ss = state.squaredNorm();
  if (!(ss > 0.0)) return refuse("the inputs determine the zero harmonic form");
  // The same projective Frobenius leak the other readings take, so all are on
  // one scale and a set of them may be summed.
  const complexd overlap = state.dot(wanted);
  const double leak = wanted.squaredNorm() - std::norm(overlap) / ss;
  return std::max(0.0, leak / wanted.squaredNorm());
}

double MultiCobordism::twoBodyResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                         const TwoBodyTarget &target) const {
  // The sum over the selected readings. Each is the same projective leak on
  // the same scale, and each scores the full 1.0 when it cannot name a state,
  // so summing them is well defined however many are chosen.
  double total = 0.0;
  for (const ReadoutMode mode : readoutModes_) {
    if (mode == ReadoutMode::Operator)
      total += operatorResidualOn(spacetime, target);
    else if (mode == ReadoutMode::Whole)
      total += wholeHarmonicResidualOn(spacetime, target);
    else if (mode == ReadoutMode::Bulk)
      total += bulkOperatorResidualOn(spacetime, target);
    else
      total += transferResidualOn(spacetime, target);
  }
  return total;
}

void MultiCobordism::setOutputStateTarget(Eigen::VectorXcd state) {
  if (state.size() < 1)
    throw std::invalid_argument("MultiCobordism::setOutputStateTarget: the state is empty");
  const double norm = state.squaredNorm();
  if (!state.allFinite() || !std::isfinite(norm) || !(norm > 0.0))
    throw std::invalid_argument(
        "MultiCobordism::setOutputStateTarget: the state must be finite and "
        "nonzero");
  outputStateTarget_ = std::move(state);
}

double MultiCobordism::operatorResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                          const TwoBodyTarget &target) const {
  // No backward wavefunction, no two-state vector. A host without conjugate
  // tori carries none, and one is not invented to fill the term.
  if (target.twoStateVector.size() == 0) return 1.0;
  const double targetNorm = target.twoStateVector.squaredNorm();
  if (!target.twoStateVector.allFinite() || !std::isfinite(targetNorm) ||
      !(targetNorm > 0.0))
    throw std::invalid_argument(
        "MultiCobordism::operatorResidualOn: the paired-frame target must "
        "be finite and nonzero");
  if (!spacetime) return 1.0;
  // The two sides are the two pairs, in block order: a state and its conjugate
  // are one side, so nothing is left out of either.
  std::vector<const BoundaryBlock *> attached;
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0) attached.push_back(&block);
  if (attached.size() % 2 != 0 || attached.size() < 2)
    throw std::logic_error("MultiCobordism::operatorResidualOn: the operator reading pairs the attached input "
                           "blocks two to a side; " + std::to_string(attached.size()) + " are attached");
  const std::size_t half = attached.size() / 2;
  const std::vector<const BoundaryBlock *> sideA(attached.begin(), attached.begin() + half);
  const std::vector<const BoundaryBlock *> sideB(attached.begin() + half, attached.end());
  Eigen::MatrixXcd T;
  try {
    T = pairedFrameTransferOn(spacetime, sideA, sideB).forward;
  } catch (const std::runtime_error &) {
    return 1.0;  // a refused geometry carries no operator: full leak
  } catch (const std::invalid_argument &) {
    return 1.0;
  }
  const Eigen::MatrixXcd &wanted = target.twoStateVector;
  if (T.rows() != wanted.rows() || T.cols() != wanted.cols())
    throw std::logic_error("MultiCobordism::operatorResidualOn: the paired direct-sum target is " +
                           std::to_string(wanted.rows()) + "x" + std::to_string(wanted.cols()) +
                           " but the paired frames give " + std::to_string(T.rows()) + "x" +
                           std::to_string(T.cols()));
  const double tt = T.squaredNorm();
  if (!(tt > 0.0)) return 1.0;
  // The same projective Frobenius leak every other reading takes, so all are
  // on one scale and a set of them may be summed.
  const complexd overlap = (T.conjugate().cwiseProduct(wanted)).sum();
  const double leak = wanted.squaredNorm() - std::norm(overlap) / tt;
  return std::max(0.0, leak / wanted.squaredNorm());
}

double MultiCobordism::transferResidualOn(const std::shared_ptr<Spacetime> &spacetime,
                                          const TwoBodyTarget &target) const {
  const auto [A, B] = attachedInputBlocks();
  if (!spacetime) return 1.0;
  const double targetNorm = target.chi.squaredNorm();
  if (!target.chi.allFinite() || !std::isfinite(targetNorm) ||
      !(targetNorm > 0.0))
    throw std::invalid_argument(
        "MultiCobordism::transferResidualOn: the target must be finite and "
        "nonzero");
  Eigen::MatrixXcd T;
  try {
    T = frameTransferOn(spacetime, *A, *B).forward;
  } catch (const std::runtime_error &) {
    return 1.0;  // a refused geometry cannot carry the map: full leak
  } catch (const std::invalid_argument &) {
    return 1.0;
  }
  if (T.rows() != target.chi.rows() || T.cols() != target.chi.cols())
    throw std::logic_error("MultiCobordism: the two-body target is " + std::to_string(target.chi.rows()) + "x" +
                           std::to_string(target.chi.cols()) + " but the attached frames give " +
                           std::to_string(T.rows()) + "x" + std::to_string(T.cols()));
  // Projective Frobenius leak: min_c ||c T - chi||^2 / ||chi||^2 (the same in
  // the state and operator readings, vec being linear).
  const double tt = T.squaredNorm();
  if (!(tt > 0.0)) return 1.0;
  const complexd overlap = (T.conjugate().cwiseProduct(target.chi)).sum();
  const double leak = target.chi.squaredNorm() - std::norm(overlap) / tt;
  return std::max(0.0, leak / target.chi.squaredNorm());
}

double MultiCobordism::twoBodyResidual() const {
  if (!twoBodyTarget_) throw std::logic_error("MultiCobordism::twoBodyResidual: no two-body target");
  return twoBodyResidualOn(spacetime_, *twoBodyTarget_);
}

MultiCobordism::TwoBodyRead MultiCobordism::readTwoBody() const {
  std::vector<const BoundaryBlock *> attached;
  for (const auto &block : inputBlocks_)
    if (block.fiber && block.fiber->images.cols() > 0)
      attached.push_back(&block);
  if (attached.size() != 2 && attached.size() != 4)
    throw std::logic_error(
        "MultiCobordism::readTwoBody: the read needs two attached input "
        "fibers, or four split into two paired frames; " +
        std::to_string(attached.size()) + " found");
  const std::size_t half = attached.size() / 2;
  const std::vector<const BoundaryBlock *> sideA(
      attached.begin(), attached.begin() + half);
  const std::vector<const BoundaryBlock *> sideB(
      attached.begin() + half, attached.end());
  const chainhodge::TransferResult transfer =
      attached.size() == 2
          ? frameTransferOn(spacetime_, *attached[0], *attached[1])
          : pairedFrameTransferOn(spacetime_, sideA, sideB);
  TwoBodyRead read;
  read.choiDecomposed =
      twoBodyTarget_ ? twoBodyTarget_->choiDecomposed
                     : (!twoBodyCases_.empty()
                            ? twoBodyCases_.front().choiDecomposed
                            : true);
  read.transfer = transfer.forward;
  read.inFrames = std::all_of(
      attached.begin(), attached.end(), [](const BoundaryBlock *block) {
        return block->marking.has_value() || block->frame.has_value();
      });
  read.derivedFrames = std::all_of(
      attached.begin(), attached.end(), [](const BoundaryBlock *block) {
        return block->marking.has_value();
      });
  read.choiState = Eigen::Map<const Eigen::VectorXcd>(transfer.forward.data(), transfer.forward.size());
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(transfer.forward);
  const Eigen::VectorXd sv = svd.singularValues();
  read.singularValues.assign(sv.data(), sv.data() + sv.size());
  read.schmidtRank = 0;
  for (Eigen::Index i = 0; i < sv.size(); ++i)
    if (sv.size() > 0 && sv(i) > 1e-10 * sv(0)) ++read.schmidtRank;
  read.reversalResidual = transfer.reversalResidual;
  const bool canScore =
      attached.size() == 2 && (twoBodyTarget_ || !twoBodyCases_.empty());
  read.residual = canScore ? twoBodyResidualOverCasesOn(spacetime_)
                           : std::numeric_limits<double>::quiet_NaN();
  for (std::size_t index = 0; index < inputBlocks_.size(); ++index) {
    const BoundaryBlock &block = inputBlocks_[index];
    if (block.fiber && block.fiber->images.cols() > 0)
      read.inputFiberResiduals.push_back(fiberResidualForBoundaryBlock(block, spacetime_));
    if (block.marking) read.inputStates.push_back(readInputState(index));
  }
  const auto appendCells = [&](const std::vector<const BoundaryBlock *> &side,
                               std::vector<std::vector<std::uint64_t>> &cells) {
    for (const auto *block : side) {
      const auto operand = transferOperand(*block, spacetime_);
      cells.insert(cells.end(), operand.cells.begin(), operand.cells.end());
    }
  };
  appendCells(sideA, read.cellsA);
  appendCells(sideB, read.cellsB);
  return read;
}

}  // namespace tessera::cobordism
