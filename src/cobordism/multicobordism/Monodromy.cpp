// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the restriction and
// monodromy reads. One of the translation units that define the class's
// members by responsibility
// (https://github.com/akellehe/tessera/issues/1481).

#include "Internal.h"

namespace tessera::cobordism {

MultiCobordism::RestrictionRead MultiCobordism::restriction(const std::shared_ptr<Spacetime> &spacetime,
                                                            const std::vector<Marking> &markings) {
  RestrictionRead read;
  if (!spacetime) {
    read.obstruction = "no spacetime";
    return read;
  }
  read.betti = betti(*spacetime);
  // The edges every marking walks, as sorted tuples: each must be an edge of
  // the whole before any operator is built.
  std::set<std::vector<std::uint64_t>> seen;
  std::vector<std::vector<std::uint64_t>> cells;
  for (const Marking &marking : markings)
    for (const auto &cycle : marking)
      for (const auto &[u, v] : cycle) {
        if (u == v) {
          read.obstruction = "a marking step is a self-loop";
          return read;
        }
        std::vector<std::uint64_t> edge = {std::min(u, v), std::max(u, v)};
        if (seen.insert(edge).second) cells.push_back(std::move(edge));
      }
  if (cells.empty()) {
    read.obstruction = "empty markings";
    return read;
  }
  // A marking is stated on the surfaces' edges, which every engine move
  // keeps; a step off the whole is named before any operator is built.
  {
    std::set<std::vector<std::uint64_t>> edges;
    for (auto edge : ChainComplex::fromSpacetime(*spacetime).kSimplexVertices(1)) edges.insert(std::move(edge));
    for (const auto &cell : cells)
      if (!edges.count(cell)) {
        read.obstruction = "marking edge (" + std::to_string(cell[0]) + "," + std::to_string(cell[1]) +
                           ") is not an edge of the whole";
        return read;
      }
  }
  // Each marking's cycles as closed walks from that marking's common base
  // point (`Connection::commonBasePoint`), so that
  // the periods are taken with parallel transport: on zero phases the
  // transported period is the plain signed sum in the walk's order.
  std::vector<Marking> walks;
  walks.reserve(markings.size());
  for (std::size_t i = 0; i < markings.size(); ++i) {
    Marking ordered = markings[i];
    std::uint64_t base = 0;
    std::string why;
    if (!orderMarking(ordered, base, why)) {
      read.obstruction = "marking " + std::to_string(i) + ": " + why;
      return read;
    }
    walks.push_back(std::move(ordered));
  }
  chainhodge::Band zeroMode;
  std::shared_ptr<const chainhodge::CovariantChainHodge> op;
  try {
    const AssembledPencil assembled = PencilLayer::assemble({spacetime});
    if (assembled.dimension() < 1) {
      read.obstruction = "the whole has no edges";
      return read;
    }
    zeroMode = assembled.op->band(1, PencilLayer::harmonicContour(assembled, 1));
    op = assembled.op;
  } catch (const std::exception &e) {
    read.obstruction = std::string("the pencil refused the whole: ") + e.what();
    return read;
  }
  read.harmonicRank = zeroMode.rank();
  if (read.harmonicRank == 0) {
    read.obstruction = "the whole has no degree-1 zero mode";
    return read;
  }
  read.images = zeroMode.images;
  read.frame = zeroMode.frame;
  read.certificate = zeroMode.certificate;
  for (const Marking &walk : walks) {
    Eigen::MatrixXcd periods = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(walk.size()),
                                                      zeroMode.images.cols());
    for (std::size_t c = 0; c < walk.size(); ++c)
      for (Eigen::Index a = 0; a < zeroMode.images.cols(); ++a)
        periods(static_cast<Eigen::Index>(c), a) =
            op->connection().transportedPeriod(zeroMode.images.col(a), walk[c]);
    if (periods.rows() == 0) {
      read.obstruction = "a marking has no cycles";
      return read;
    }
    read.periods.push_back(std::move(periods));
  }
  return read;
}

MultiCobordism::MonodromyRead MultiCobordism::monodromy(const std::shared_ptr<Spacetime> &spacetime,
                                                        const Marking &markingA,
                                                        const Marking &markingB) {
  MonodromyRead read;
  const RestrictionRead restricted = restriction(spacetime, {markingA, markingB});
  read.betti = restricted.betti;
  read.harmonicRank = restricted.harmonicRank;
  if (!restricted.obstruction.empty()) {
    read.obstruction = restricted.obstruction;
    return read;
  }
  read.periodsA = restricted.periods[0];
  read.periodsB = restricted.periods[1];
  // P_B = M P_A, least squares over the zero mode's columns (exact for a rank-2
  // zero mode with invertible P_A); M is |B| x |A|.
  const Eigen::JacobiSVD<Eigen::MatrixXcd> svd(read.periodsA);
  const double tolerance = 1e-9 * std::max(1.0, svd.singularValues()(0));
  Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i)
    if (svd.singularValues()(i) > tolerance) ++rank;
  const Eigen::MatrixXcd transposed =
      read.periodsA.transpose().colPivHouseholderQr().solve(read.periodsB.transpose());
  read.monodromy = transposed.transpose();
  const double normB = read.periodsB.norm();
  read.fitResidual = normB > 0.0 ? (read.monodromy * read.periodsA - read.periodsB).norm() / normB
                                 : std::numeric_limits<double>::quiet_NaN();
  read.rounded.assign(static_cast<std::size_t>(read.monodromy.rows()),
                      std::vector<long>(static_cast<std::size_t>(read.monodromy.cols()), 0));
  double roundingResidual = 0.0;
  for (Eigen::Index i = 0; i < read.monodromy.rows(); ++i)
    for (Eigen::Index j = 0; j < read.monodromy.cols(); ++j) {
      const long nearest = std::lround(read.monodromy(i, j).real());
      read.rounded[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = nearest;
      roundingResidual = std::max(
          roundingResidual, std::abs(read.monodromy(i, j) - complexd(static_cast<double>(nearest), 0.0)));
    }
  read.roundingResidual = roundingResidual;
  if (rank < read.periodsA.rows())
    read.obstruction = "the zero mode's periods over marking A have rank " + std::to_string(rank) +
                       " below the marking's " + std::to_string(read.periodsA.rows()) +
                       " cycles: the whole does not carry that marking independently";
  return read;
}

}  // namespace tessera::cobordism
