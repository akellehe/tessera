// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/PencilLayer.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <Eigen/Dense>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

namespace {

using Cell = std::vector<std::uint64_t>;

std::string cellName(const Cell &c) {
  std::string s = "(";
  for (std::size_t i = 0; i < c.size(); ++i) s += (i ? "," : "") + std::to_string(c[i]);
  return s + ")";
}

bool sameComplex(Complex a, Complex b, double tol) {
  return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

Eigen::MatrixXcd rows(const Eigen::MatrixXcd &A, const std::vector<int> &idx) {
  Eigen::MatrixXcd out(static_cast<Eigen::Index>(idx.size()), A.cols());
  for (std::size_t i = 0; i < idx.size(); ++i) out.row(static_cast<Eigen::Index>(i)) = A.row(idx[i]);
  return out;
}

Eigen::MatrixXcd block(const Eigen::MatrixXcd &A, const std::vector<int> &r, const std::vector<int> &c) {
  Eigen::MatrixXcd out(static_cast<Eigen::Index>(r.size()), static_cast<Eigen::Index>(c.size()));
  for (std::size_t i = 0; i < r.size(); ++i)
    for (std::size_t j = 0; j < c.size(); ++j) out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = A(r[i], c[j]);
  return out;
}

}  // namespace

int AssembledPencil::cellIndex(int k, const Cell &cell) const {
  if (!complexPtr || k < 0 || k > complexPtr->dimension()) return -1;
  Cell sorted(cell);
  std::sort(sorted.begin(), sorted.end());
  const auto cells = complexPtr->kSimplexVertices(k);
  const auto it = std::lower_bound(cells.begin(), cells.end(), sorted);
  if (it == cells.end() || *it != sorted) return -1;
  return static_cast<int>(it - cells.begin());
}

AssembledPencil PencilLayer::assemble(const std::vector<std::shared_ptr<Spacetime>> &pieces,
                                      const std::vector<double> &epsilons,
                                      chainhodge::Branch branch, int crossoverDimension) {
  if (pieces.empty()) throw std::invalid_argument("PencilLayer::assemble: no pieces");
  if (!epsilons.empty() && epsilons.size() != pieces.size())
    throw std::invalid_argument("PencilLayer::assemble: one epsilon per piece (or none)");
  // One epsilon per connected assembly.
  double epsilon = std::numeric_limits<double>::quiet_NaN();
  int epsilonOwner = -1;
  for (std::size_t i = 0; i < epsilons.size(); ++i) {
    if (std::isnan(epsilons[i])) continue;
    if (epsilonOwner < 0) {
      epsilon = epsilons[i];
      epsilonOwner = static_cast<int>(i);
    } else if (std::abs(epsilons[i] - epsilon) > 1e-15 * std::max(1.0, std::abs(epsilon))) {
      throw std::invalid_argument(
          "PencilLayer::assemble: piece " + std::to_string(epsilonOwner) + " was computed at epsilon=" +
          std::to_string(epsilon) + " and piece " + std::to_string(i) + " at epsilon=" +
          std::to_string(epsilons[i]) + "; one epsilon per connected assembly, so the shared-cell "
          "contributions agree");
    }
  }

  AssembledPencil out;
  out.epsilon = epsilon;
  std::map<Cell, int> topOwner;  // top cell -> first piece
  std::vector<Cell> unionTop;
  std::unordered_map<std::uint64_t, std::pair<Complex, int>> lengthByEdge;  // key -> (s, piece)
  std::unordered_map<std::uint64_t, std::pair<Complex, int>> linkByEdge;
  std::vector<std::vector<std::set<Cell>>> faceSets(pieces.size());
  auto key = [](std::uint64_t a, std::uint64_t b) { if (a > b) std::swap(a, b); return (a << 32) ^ b; };
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    if (!pieces[i]) throw std::invalid_argument("PencilLayer::assemble: null piece");
    const ChainComplex Ki = chainhodge::WhitneyMass::complexOf(*pieces[i]);
    const chainhodge::SquaredLengths si = chainhodge::WhitneyMass::squaredLengthsOf(*pieces[i], Ki);
    const chainhodge::Connection Ui = chainhodge::Connection::fromSpacetime(*pieces[i], Ki);
    std::vector<Cell> top = Ki.orientedTopSimplices();
    out.pieces.push_back(top);
    for (auto &t : top)
      if (topOwner.emplace(t, static_cast<int>(i)).second) unionTop.push_back(t);
    faceSets[i].resize(static_cast<std::size_t>(Ki.dimension()) + 1);
    for (int k = 0; k <= Ki.dimension(); ++k)
      for (const auto &c : Ki.kSimplexVertices(k)) faceSets[i][static_cast<std::size_t>(k)].insert(c);
    const auto edges = Ki.kSimplexVertices(1);
    for (std::size_t j = 0; j < edges.size(); ++j) {
      const std::uint64_t ek = key(edges[j][0], edges[j][1]);
      const Complex s = si[j];
      const Complex u = Ui.links()[j];
      const auto seen = lengthByEdge.find(ek);
      if (seen == lengthByEdge.end()) {
        lengthByEdge[ek] = {s, static_cast<int>(i)};
        linkByEdge[ek] = {u, static_cast<int>(i)};
      } else {
        if (!sameComplex(seen->second.first, s, 1e-12))
          throw std::invalid_argument(
              "PencilLayer::assemble: shared edge " + cellName(edges[j]) + " has squared length " +
              std::to_string(seen->second.first.real()) + "+" + std::to_string(seen->second.first.imag()) +
              "i in piece " + std::to_string(seen->second.second) + " and " + std::to_string(s.real()) + "+" +
              std::to_string(s.imag()) + "i in piece " + std::to_string(i) +
              "; the pieces' geometries must agree on shared cells");
        const auto lk = linkByEdge.find(ek);
        if (!sameComplex(lk->second.first, u, 1e-12))
          throw std::invalid_argument("PencilLayer::assemble: shared edge " + cellName(edges[j]) +
                                      " carries different links in pieces " +
                                      std::to_string(lk->second.second) + " and " + std::to_string(i));
      }
    }
  }
  out.complexPtr = std::make_shared<const ChainComplex>(ChainComplex::fromTopCells(unionTop));
  const ChainComplex &K = *out.complexPtr;
  const int d = K.dimension();
  const auto unionEdges = K.kSimplexVertices(1);
  out.lengths.reserve(unionEdges.size());
  std::vector<Complex> links;
  links.reserve(unionEdges.size());
  for (const auto &e : unionEdges) {
    out.lengths.push_back(lengthByEdge.at(key(e[0], e[1])).first);
    links.push_back(linkByEdge.at(key(e[0], e[1])).first);
  }
  out.sharedCells.assign(static_cast<std::size_t>(d) + 1, {});
  for (int k = 0; k <= d; ++k)
    for (const auto &c : K.kSimplexVertices(k)) {
      int owners = 0;
      for (const auto &fs : faceSets)
        if (k < static_cast<int>(fs.size()) && fs[static_cast<std::size_t>(k)].count(c)) ++owners;
      if (owners > 1) out.sharedCells[static_cast<std::size_t>(k)].push_back(c);
    }
  out.base = std::make_shared<chainhodge::ChainHodge>(K, out.lengths, chainhodge::Preset::L2, branch,
                                                      crossoverDimension, epsilon);
  chainhodge::Connection U(K, links);
  out.op = std::make_shared<chainhodge::CovariantChainHodge>(*out.base, U, 7, /*measureCertificate=*/false);
  out.dual = std::make_shared<chainhodge::CovariantChainHodge>(*out.base, U.inverse(), 7, false);
  return out;
}

double PencilLayer::assemblyResidual(const AssembledPencil &assembled, int k, chainhodge::Branch branch) {
  const Eigen::MatrixXcd whole(chainhodge::WhitneyMass::assemble(assembled.complex(), assembled.lengths, k, branch));
  Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(whole.rows(), whole.cols());
  const auto unionEdges = assembled.complex().kSimplexVertices(1);
  std::map<Cell, Complex> lengthOf;
  for (std::size_t j = 0; j < unionEdges.size(); ++j) lengthOf[unionEdges[j]] = assembled.lengths[j];
  for (const auto &top : assembled.pieces) {
    const ChainComplex Ki = ChainComplex::fromTopCells(top);
    chainhodge::SquaredLengths si;
    for (const auto &e : Ki.kSimplexVertices(1)) si.push_back(lengthOf.at(e));
    const Eigen::MatrixXcd Mi(chainhodge::WhitneyMass::assemble(Ki, si, k, branch));
    const auto cells = Ki.kSimplexVertices(k);
    std::vector<int> idx;
    for (const auto &c : cells) idx.push_back(assembled.cellIndex(k, c));
    for (Eigen::Index a = 0; a < Mi.rows(); ++a)
      for (Eigen::Index b = 0; b < Mi.cols(); ++b) sum(idx[static_cast<std::size_t>(a)], idx[static_cast<std::size_t>(b)]) += Mi(a, b);
  }
  return (whole - sum).cwiseAbs().maxCoeff();
}

std::vector<int> PencilLayer::cellsWithin(const AssembledPencil &assembled, int k,
                                          const std::vector<std::uint64_t> &vertices) {
  const std::set<std::uint64_t> region(vertices.begin(), vertices.end());
  std::vector<int> out;
  const auto cells = assembled.complex().kSimplexVertices(k);
  for (int j = 0; j < static_cast<int>(cells.size()); ++j) {
    bool inside = true;
    for (const auto v : cells[static_cast<std::size_t>(j)])
      if (!region.count(v)) { inside = false; break; }
    if (inside) out.push_back(j);
  }
  return out;
}

std::vector<int> PencilLayer::indicesOf(const AssembledPencil &assembled, int k, const std::vector<Cell> &cells) {
  std::vector<int> out;
  out.reserve(cells.size());
  for (const auto &c : cells) {
    const int j = assembled.cellIndex(k, c);
    if (j < 0)
      throw std::invalid_argument("PencilLayer: cell " + cellName(c) + " is not a degree-" +
                                  std::to_string(k) + " cell of the assembled complex");
    out.push_back(j);
  }
  return out;
}

chainhodge::Pencil PencilLayer::pencil(const AssembledPencil &assembled, int k) {
  return assembled.op->pencil(k);
}

chainhodge::FeshbachResult PencilLayer::boundaryResponse(const AssembledPencil &assembled, int k,
                                                          const std::vector<int> &interface, Complex lambda) {
  const chainhodge::Pencil P = pencil(assembled, k);
  return chainhodge::PencilSchur::feshbach(P.A, P.B, lambda, interface);
}

BorderedPencil PencilLayer::borderedPencil(const AssembledPencil &assembled, int k, Complex lambda) {
  if (k < 1) throw std::invalid_argument("PencilLayer::borderedPencil: the bordered form needs k >= 1");
  const auto &op = *assembled.op;
  const int d = assembled.dimension();
  const Eigen::MatrixXcd Mk(op.dressed(k));
  const Eigen::MatrixXcd Mkm1(op.dressed(k - 1));
  const Eigen::MatrixXcd Bk(op.twistedBoundary(k));          // n_{k-1} x n_k
  const Eigen::MatrixXcd BkD(op.twistedBoundaryDual(k));
  const int nk = static_cast<int>(Mk.rows()), nl = static_cast<int>(Mkm1.rows());
  BorderedPencil B;
  B.degree = k;
  B.lambda = lambda;
  B.upperCount = nk;
  B.lowerCount = nl;
  B.matrix = Eigen::MatrixXcd::Zero(nk + nl, nk + nl);
  Eigen::MatrixXcd upper = lambda * Mk;
  if (k < d) {
    const Eigen::MatrixXcd Bk1(op.twistedBoundary(k + 1));
    const Eigen::MatrixXcd Bk1D(op.twistedBoundaryDual(k + 1));
    const Eigen::MatrixXcd Mk1(op.dressed(k + 1));
    upper -= Bk1 * Mk1 * Bk1D.transpose();
  }
  B.matrix.topLeftCorner(nk, nk) = upper;
  B.matrix.topRightCorner(nk, nl) = -Mk * BkD.transpose();
  B.matrix.bottomLeftCorner(nl, nk) = -Bk * Mk;
  B.matrix.bottomRightCorner(nl, nl) = Mkm1;
  return B;
}

chainhodge::FeshbachResult PencilLayer::borderedResponse(const AssembledPencil &assembled, int k,
                                                          const std::vector<int> &upperInterface,
                                                          const std::vector<int> &lowerInterface,
                                                          Complex lambda) {
  const BorderedPencil B = borderedPencil(assembled, k, lambda);
  std::vector<int> interface(upperInterface);
  for (const int j : lowerInterface) interface.push_back(B.upperCount + j);
  std::sort(interface.begin(), interface.end());
  interface.erase(std::unique(interface.begin(), interface.end()), interface.end());
  const Eigen::MatrixXcd zero = Eigen::MatrixXcd::Zero(B.matrix.rows(), B.matrix.cols());
  // P = A - 0 * M with A the bordered matrix at lambda: the plain Schur complement.
  return chainhodge::PencilSchur::feshbach(B.matrix, zero, Complex(0.0, 0.0), interface);
}

Eigen::MatrixXcd PencilLayer::upperResponse(const chainhodge::FeshbachResult &bordered, int upperCount) {
  std::vector<int> up, low;
  for (std::size_t i = 0; i < bordered.interface.size(); ++i)
    (bordered.interface[i] < upperCount ? up : low).push_back(static_cast<int>(i));
  if (low.empty()) return bordered.response;
  const Eigen::MatrixXcd Ruu = block(bordered.response, up, up), Rul = block(bordered.response, up, low),
                         Rlu = block(bordered.response, low, up), Rll = block(bordered.response, low, low);
  return Ruu - Rul * Rll.partialPivLu().solve(Rlu);
}

Eigen::MatrixXcd PencilLayer::composeResponses(const Eigen::MatrixXcd &left, const std::vector<int> &leftCells,
                                               const Eigen::MatrixXcd &right, const std::vector<int> &rightCells) {
  if (left.rows() != static_cast<Eigen::Index>(leftCells.size()) || right.rows() != static_cast<Eigen::Index>(rightCells.size()))
    throw std::invalid_argument("PencilLayer::composeResponses: response and cell list sizes differ");
  std::vector<int> all(leftCells);
  all.insert(all.end(), rightCells.begin(), rightCells.end());
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());
  std::map<int, int> pos;
  for (std::size_t i = 0; i < all.size(); ++i) pos[all[i]] = static_cast<int>(i);
  Eigen::MatrixXcd R = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(all.size()), static_cast<Eigen::Index>(all.size()));
  for (std::size_t a = 0; a < leftCells.size(); ++a)
    for (std::size_t b = 0; b < leftCells.size(); ++b) R(pos[leftCells[a]], pos[leftCells[b]]) += left(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b));
  for (std::size_t a = 0; a < rightCells.size(); ++a)
    for (std::size_t b = 0; b < rightCells.size(); ++b) R(pos[rightCells[a]], pos[rightCells[b]]) += right(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b));
  const std::set<int> L(leftCells.begin(), leftCells.end()), Rt(rightCells.begin(), rightCells.end());
  std::vector<int> shared, kept;
  for (const int c : all) (L.count(c) && Rt.count(c) ? shared : kept).push_back(pos[c]);
  if (shared.empty()) return R;
  const Eigen::MatrixXcd Rkk = block(R, kept, kept), Rks = block(R, kept, shared), Rsk = block(R, shared, kept),
                         Rss = block(R, shared, shared);
  return Rkk - Rks * Rss.partialPivLu().solve(Rsk);
}

chainhodge::Contour PencilLayer::bandContour(const AssembledPencil &assembled, int k, int bandIndex,
                                             int nodeCount) {
  if (bandIndex < 0) throw std::invalid_argument("PencilLayer::bandContour: negative band index");
  const auto ev = assembled.op->spectrum(k).eigenvalues;
  double scale = 0.0;
  for (const auto &z : ev) scale = std::max(scale, std::abs(z));
  const double tolerance = 1e-9 * std::max(scale, 1.0);
  std::vector<Complex> sorted(ev.begin(), ev.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const Complex &a, const Complex &b) { return std::abs(a) < std::abs(b); });
  // Distinct clusters in order of modulus: a value joins the last cluster when
  // it lies within the tolerance of that cluster's mean.
  std::vector<Complex> centers;
  std::vector<int> counts;
  for (const auto &z : sorted) {
    if (!centers.empty() && std::abs(z - centers.back()) <= tolerance) {
      const int n = ++counts.back();
      centers.back() += (z - centers.back()) / static_cast<double>(n);
    } else {
      centers.push_back(z);
      counts.push_back(1);
    }
  }
  if (bandIndex >= static_cast<int>(centers.size()))
    throw std::invalid_argument("PencilLayer::bandContour: band index " + std::to_string(bandIndex) +
                                " exceeds the " + std::to_string(centers.size()) +
                                " distinct eigenvalue clusters of the degree-" + std::to_string(k) +
                                " pencil");
  const Complex center = centers[static_cast<std::size_t>(bandIndex)];
  double nearest = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < centers.size(); ++j)
    if (static_cast<int>(j) != bandIndex) nearest = std::min(nearest, std::abs(centers[j] - center));
  // A quarter of the gap: the nearest other cluster then lies three radii from
  // the circle, so the trapezoidal resolvent quadrature leaks (1/3)^nodeCount of
  // it into the band.
  const double radius = std::isfinite(nearest) ? 0.25 * nearest : 1.0;
  return chainhodge::Contour::circle(center, radius, nodeCount);
}

chainhodge::Contour PencilLayer::harmonicContour(const AssembledPencil &assembled, int k, int nodeCount) {
  const auto ev = assembled.op->spectrum(k).eigenvalues;
  double scale = 0.0;
  for (const auto &z : ev) scale = std::max(scale, std::abs(z));
  double smallest = std::numeric_limits<double>::infinity();
  for (const auto &z : ev)
    if (std::abs(z) > 1e-9 * std::max(scale, 1.0)) smallest = std::min(smallest, std::abs(z));
  const double radius = std::isfinite(smallest) ? 0.5 * smallest : 1.0;
  return chainhodge::Contour::circle(Complex(0.0, 0.0), radius, nodeCount);
}

BoundaryFiber PencilLayer::readBoundaryFiber(const AssembledPencil &assembled, int k,
                                             const chainhodge::Contour &contour,
                                             const std::vector<Cell> &cells, double kappa) {
  const chainhodge::Band band = assembled.op->band(k, contour, kappa);
  const std::vector<int> idx = indicesOf(assembled, k, cells);
  const Eigen::MatrixXcd dualImages = assembled.dual->applyG(k, band.dualFrame);
  const Eigen::MatrixXcd M(assembled.op->Minv(k));
  BoundaryFiber f;
  f.degree = k;
  f.cells = cells;
  for (auto &c : f.cells) std::sort(c.begin(), c.end());
  f.images = rows(band.images, idx);
  f.dualImages = rows(dualImages, idx);
  f.gram = f.images.transpose() * block(M, idx, idx) * f.images;
  f.fullGram = band.images.transpose() * M * band.images;
  f.eigenvalue = (band.reduced.rows() > 0) ? band.reduced.trace() / static_cast<double>(band.reduced.rows())
                                           : Complex(0.0, 0.0);
  if (band.reduced.rows() == 0 && !contour.nodes.empty()) {
    Complex c(0.0, 0.0);
    for (const auto &z : contour.nodes) c += z;
    f.eigenvalue = c / static_cast<double>(contour.nodes.size());
  }
  f.contour = contour;
  f.certificate = band.certificate;
  f.epsilon = assembled.epsilon;
  return f;
}

FiberLevel PencilLayer::level(const AssembledPencil &assembled, int k,
                              const std::vector<BoundaryFiber> &retained, Complex lambda) {
  if (retained.empty()) throw std::invalid_argument("PencilLayer::level: no retained fibers");
  std::vector<std::vector<int>> supports;
  std::set<int> claimed;
  for (const auto &f : retained) {
    if (f.degree != k)
      throw std::invalid_argument("PencilLayer::level: a retained fiber is at degree " + std::to_string(f.degree) +
                                  ", the level is at degree " + std::to_string(k));
    std::vector<int> idx = indicesOf(assembled, k, f.cells);
    claimed.insert(idx.begin(), idx.end());  // overlaps are the labeled sum
    supports.push_back(std::move(idx));
  }
  FiberLevel L;
  L.degree = k;
  L.lambda = lambda;
  L.interfaceCells.assign(claimed.begin(), claimed.end());
  const chainhodge::Pencil P = pencil(assembled, k);
  L.response = chainhodge::PencilSchur::feshbach(P.A, P.B, lambda, L.interfaceCells);
  L.interiorCells = L.response.interior;
  std::map<int, int> pos;
  for (std::size_t i = 0; i < L.response.interface.size(); ++i) pos[L.response.interface[i]] = static_cast<int>(i);
  const int nB = static_cast<int>(L.response.interface.size());
  std::vector<Eigen::MatrixXcd> placed, placedDual;
  int R = 0;
  for (std::size_t a = 0; a < retained.size(); ++a) {
    const auto &f = retained[a];
    Eigen::MatrixXcd Z = Eigen::MatrixXcd::Zero(nB, f.rank()), Zd = Eigen::MatrixXcd::Zero(nB, f.rank());
    for (std::size_t i = 0; i < supports[a].size(); ++i) {
      Z.row(pos[supports[a][i]]) = f.images.row(static_cast<Eigen::Index>(i));
      if (f.dualImages.rows() == f.images.rows()) Zd.row(pos[supports[a][i]]) = f.dualImages.row(static_cast<Eigen::Index>(i));
      else Zd.row(pos[supports[a][i]]) = f.images.row(static_cast<Eigen::Index>(i));
    }
    placed.push_back(Z);
    placedDual.push_back(Zd);
    R += f.rank();
  }
  L.J.resize(nB, R);
  L.Jdual.resize(nB, R);
  int off = 0;
  for (std::size_t a = 0; a < placed.size(); ++a) {
    L.J.middleCols(off, placed[a].cols()) = placed[a];
    L.Jdual.middleCols(off, placed[a].cols()) = placedDual[a];
    off += static_cast<int>(placed[a].cols());
  }
  const Eigen::MatrixXcd MBB = block(P.B, L.response.interface, L.response.interface);
  if (L.response.interiorSingular)
    throw std::runtime_error("PencilLayer::level: the further cobordism's bulk is resonant at the shift "
                             "(interior singular); the Feshbach complement is not defined there");
  L.restriction = chainhodge::PencilSchur::restrictToFibers(L.response.response, MBB, placed);
  L.blockOffsets = L.restriction.blockOffsets;
  L.blockRanks = L.restriction.blockRanks;
  const Eigen::MatrixXcd &T = L.response.constraintModes;
  const Eigen::MatrixXcd MT = T.transpose() * P.B * T;
  L.constraintGram = L.J.transpose() * MT * L.J;
  L.fibersDisjoint = true;
  for (std::size_t a = 0; a < supports.size() && L.fibersDisjoint; ++a)
    for (std::size_t b = a + 1; b < supports.size(); ++b) {
      const std::set<int> sa(supports[a].begin(), supports[a].end());
      const bool overlap = std::any_of(supports[b].begin(), supports[b].end(), [&](int j) { return sa.count(j) > 0; });
      if (overlap || chainhodge::PencilSchur::supportsShareTopSimplex(assembled.complex(), k, supports[a], supports[b])) {
        L.fibersDisjoint = false;
        break;
      }
    }
  return L;
}

chainhodge::TransferResult PencilLayer::transfer(const AssembledPencil &assembled, int k,
                                                 const BoundaryFiber &A, const BoundaryFiber &B,
                                                 double tolerance) {
  const chainhodge::Pencil P = pencil(assembled, k);
  const chainhodge::Pencil Pd = assembled.dual->pencil(k);
  const int n = static_cast<int>(P.A.rows());
  auto place = [&](const BoundaryFiber &f, bool dual) {
    const std::vector<int> idx = indicesOf(assembled, k, f.cells);
    const Eigen::MatrixXcd &src = (dual && f.dualImages.rows() == f.images.rows()) ? f.dualImages : f.images;
    Eigen::MatrixXcd Z = Eigen::MatrixXcd::Zero(n, f.rank());
    for (std::size_t i = 0; i < idx.size(); ++i) Z.row(idx[i]) = src.row(static_cast<Eigen::Index>(i));
    return Z;
  };
  return chainhodge::PencilSchur::transfer(P.A, Pd.A, place(A, false), place(A, true), place(B, false),
                                           place(B, true), tolerance);
}

double PencilLayer::fiberResidual(const std::shared_ptr<Spacetime> &spacetime,
                                  const BoundaryFiber &target, FiberBand band) {
  if (target.images.cols() == 0)
    throw std::logic_error("PencilLayer::fiberResidual: the fiber target has no images");
  const double targetNorm = target.images.squaredNorm();
  if (!(targetNorm > 0.0))
    throw std::logic_error("PencilLayer::fiberResidual: the fiber target is zero");
  if (!spacetime) return 1.0;  // no complex to read: the target leaks in full
  // A candidate geometry the pencil refuses (a singular dressed metric, a
  // branch or allowability failure, a cell outside the complex) cannot carry
  // the state: it scores as the full leak, exactly as a block with no emerged
  // register does under the period residual. Contract errors still propagate.
  BoundaryFiber read;
  try {
    const AssembledPencil assembled = assemble({spacetime});
    if (assembled.dimension() < target.degree) return 1.0;
    const std::vector<int> idx = indicesOf(assembled, target.degree, target.cells);
    if (idx.size() != target.cells.size()) return 1.0;
    const chainhodge::Contour contour = fiberContour(assembled, target, band);
    read = readBoundaryFiber(assembled, target.degree, contour, target.cells);
  } catch (const std::runtime_error &) {
    return 1.0;
  } catch (const std::invalid_argument &) {
    return 1.0;
  }
  if (read.images.cols() == 0 || read.images.rows() != target.images.rows()) return 1.0;
  // Least-squares fit of the target images in the band's images on the cells.
  const Eigen::MatrixXcd coefficients = read.images.colPivHouseholderQr().solve(target.images);
  const double leak = (read.images * coefficients - target.images).squaredNorm();
  return leak / targetNorm;
}

chainhodge::Contour PencilLayer::fiberContour(const AssembledPencil &assembled,
                                              const BoundaryFiber &target, FiberBand band) {
  switch (band) {
    case FiberBand::ZeroMode:
      // The zero mode of this pencil: the block's own Laplacian's harmonic
      // space, whatever contour the fiber stores (see `MultiCobordism::fiberBandFor`).
      return harmonicContour(assembled, target.degree);
    case FiberBand::AsStored:
      break;
  }
  return target.contour.nodes.empty() ? bandContour(assembled, target.degree, 1)
                                      : target.contour;
}

Eigen::MatrixXcd PencilLayer::dualFrame(const std::shared_ptr<Spacetime> &complex, int degree,
                                        const std::vector<std::vector<std::uint64_t>> &cells,
                                        const Eigen::MatrixXcd &images) {
  if (!complex) throw std::invalid_argument("PencilLayer::dualFrame: null complex");
  if (images.cols() == 0) throw std::invalid_argument("PencilLayer::dualFrame: the frame has no columns");
  if (static_cast<std::size_t>(images.rows()) != cells.size())
    throw std::invalid_argument("PencilLayer::dualFrame: one image row per cell (" +
                                std::to_string(images.rows()) + " rows, " + std::to_string(cells.size()) + " cells)");
  const AssembledPencil assembled = assemble({complex});
  if (degree < 0 || degree > assembled.dimension())
    throw std::invalid_argument("PencilLayer::dualFrame: degree " + std::to_string(degree) +
                                " is outside the complex's dimension " + std::to_string(assembled.dimension()));
  std::vector<std::vector<std::uint64_t>> sorted = cells;
  for (auto &c : sorted) std::sort(c.begin(), c.end());
  // A cell absent from the complex at this degree is refused by name here.
  const std::vector<int> idx = indicesOf(assembled, degree, sorted);
  // B = Z^T M_k Z on the cells: the frame's pairing under the transpose pairing
  // of the complex's own pencil (M_k the Whitney mass matrix, the chain
  // metric's inverse, as PencilLayer::readBoundaryFiber's Gram uses it).
  const Eigen::MatrixXcd M(assembled.op->Minv(degree));
  const auto n = static_cast<Eigen::Index>(idx.size());
  Eigen::MatrixXcd MBB(n, n);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < n; ++j)
      MBB(i, j) = M(idx[static_cast<std::size_t>(i)], idx[static_cast<std::size_t>(j)]);
  const Eigen::MatrixXcd pairing = images.transpose() * MBB * images;
  Eigen::FullPivLU<Eigen::MatrixXcd> lu(pairing);
  if (!lu.isInvertible())
    throw std::runtime_error("PencilLayer::dualFrame: the frame's pairing Z^T M Z is singular (an isotropic "
                             "frame); no dual under the transpose pairing exists");
  // Z^vee = Z B^{-T}, i.e. (Z^vee)^T = B^{-1} Z^T, so that (Z^vee)^T M Z = B^{-1} B = I.
  return lu.solve(images.transpose()).transpose();
}

PencilLayer::RestrictionRead PencilLayer::restriction(const std::shared_ptr<Spacetime> &spacetime,
                                                            const std::vector<Marking> &markings) {
  RestrictionRead read;
  if (!spacetime) {
    read.obstruction = "no spacetime";
    return read;
  }
  read.betti = spacetime->getBettiNumbers();
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
    if (!chainhodge::Connection::orderWalks(ordered, base, why)) {
      read.obstruction = "marking " + std::to_string(i) + ": " + why;
      return read;
    }
    walks.push_back(std::move(ordered));
  }
  chainhodge::Band zeroMode;
  std::shared_ptr<const chainhodge::CovariantChainHodge> op;
  try {
    const AssembledPencil assembled = assemble({spacetime});
    if (assembled.dimension() < 1) {
      read.obstruction = "the whole has no edges";
      return read;
    }
    zeroMode = assembled.op->band(1, harmonicContour(assembled, 1));
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

PencilLayer::MonodromyRead PencilLayer::monodromy(const std::shared_ptr<Spacetime> &spacetime,
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
          roundingResidual, std::abs(read.monodromy(i, j) - std::complex<double>(static_cast<double>(nearest), 0.0)));
    }
  read.roundingResidual = roundingResidual;
  if (rank < read.periodsA.rows())
    read.obstruction = "the zero mode's periods over marking A have rank " + std::to_string(rank) +
                       " below the marking's " + std::to_string(read.periodsA.rows()) +
                       " cycles: the whole does not carry that marking independently";
  return read;
}

}  // namespace tessera::cobordism
