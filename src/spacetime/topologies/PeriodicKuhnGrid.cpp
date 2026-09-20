// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/topologies/PeriodicKuhnGrid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::spacetime {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

int wrap(int index, int modulus) {
  const int r = index % modulus;
  return r < 0 ? r + modulus : r;
}

// The representative of a residue difference in {-1, 0, +1}; `ok` is cleared
// when the two indices are not within one step of each other around the axis.
int unwrapStep(int from, int to, int modulus, bool &ok) {
  const int r = wrap(to - from, modulus);
  if (r == 0) return 0;
  if (r == 1) return 1;
  if (r == modulus - 1) return -1;
  ok = false;
  return 0;
}

}  // namespace

PeriodicKuhnGrid::PeriodicKuhnGrid(int n1, int n2, int n3, const Gram &latticeGram)
    : n_{{n1, n2, n3}}, gram_(latticeGram) {
  for (int a = 0; a < 3; ++a)
    if (n_[static_cast<std::size_t>(a)] < 3)
      throw std::invalid_argument(
          "PeriodicKuhnGrid: every division must be at least 3 so that two vertices "
          "share at most one edge (got " +
          std::to_string(n_[static_cast<std::size_t>(a)]) + " on axis " + std::to_string(a) + ")");
  const auto &A = gram_;
  double scale = 0.0;
  for (const auto &row : A)
    for (double v : row) scale = std::max(scale, std::abs(v));
  for (std::size_t a = 0; a < 3; ++a)
    for (std::size_t b = a + 1; b < 3; ++b)
      if (std::abs(A[a][b] - A[b][a]) > 1e-12 * scale)
        throw std::invalid_argument("PeriodicKuhnGrid: the lattice Gram matrix must be symmetric");
  // Sylvester's criterion on the leading principal minors.
  const double m1 = A[0][0];
  const double m2 = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const double m3 = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                    A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                    A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  if (!(m1 > 0.0) || !(m2 > 0.0) || !(m3 > 0.0))
    throw std::invalid_argument(
        "PeriodicKuhnGrid: the lattice Gram matrix must be positive definite (three "
        "linearly independent lattice vectors)");
}

PeriodicKuhnGrid PeriodicKuhnGrid::cubic(int n, double a) {
  const double a2 = a * a;
  return PeriodicKuhnGrid(n, n, n, Gram{{{{a2, 0.0, 0.0}}, {{0.0, a2, 0.0}}, {{0.0, 0.0, a2}}}});
}

PeriodicKuhnGrid::Gram PeriodicKuhnGrid::stepMetric() const {
  Gram g{};
  for (std::size_t a = 0; a < 3; ++a)
    for (std::size_t b = 0; b < 3; ++b)
      g[a][b] = gram_[a][b] / (static_cast<double>(n_[a]) * static_cast<double>(n_[b]));
  return g;
}

std::size_t PeriodicKuhnGrid::vertexCount() const noexcept {
  return static_cast<std::size_t>(n_[0]) * static_cast<std::size_t>(n_[1]) *
         static_cast<std::size_t>(n_[2]);
}

std::uint64_t PeriodicKuhnGrid::vertexId(int i, int j, int l) const {
  const auto wi = static_cast<std::uint64_t>(wrap(i, n_[0]));
  const auto wj = static_cast<std::uint64_t>(wrap(j, n_[1]));
  const auto wl = static_cast<std::uint64_t>(wrap(l, n_[2]));
  return (wi * static_cast<std::uint64_t>(n_[1]) + wj) * static_cast<std::uint64_t>(n_[2]) + wl;
}

std::array<int, 3> PeriodicKuhnGrid::gridIndex(std::uint64_t id) const {
  if (id >= vertexCount()) throw std::out_of_range("PeriodicKuhnGrid: vertex id out of range");
  const auto n2 = static_cast<std::uint64_t>(n_[1]);
  const auto n3 = static_cast<std::uint64_t>(n_[2]);
  return {{static_cast<int>(id / (n2 * n3)), static_cast<int>((id / n3) % n2),
           static_cast<int>(id % n3)}};
}

std::array<double, 3> PeriodicKuhnGrid::fractionalCoordinates(std::uint64_t id) const {
  const auto index = gridIndex(id);
  return {{static_cast<double>(index[0]) / n_[0], static_cast<double>(index[1]) / n_[1],
           static_cast<double>(index[2]) / n_[2]}};
}

std::vector<PeriodicKuhnGrid::Cell> PeriodicKuhnGrid::cells() const {
  std::vector<Cell> out;
  out.reserve(6 * vertexCount());
  std::array<int, 3> axes{{0, 1, 2}};
  for (int i = 0; i < n_[0]; ++i)
    for (int j = 0; j < n_[1]; ++j)
      for (int l = 0; l < n_[2]; ++l) {
        std::sort(axes.begin(), axes.end());
        do {
          std::array<int, 3> p{{i, j, l}};
          Cell cell;
          cell.reserve(4);
          cell.push_back(vertexId(p[0], p[1], p[2]));
          for (int axis : axes) {
            ++p[static_cast<std::size_t>(axis)];
            cell.push_back(vertexId(p[0], p[1], p[2]));
          }
          std::sort(cell.begin(), cell.end());
          out.push_back(std::move(cell));
        } while (std::next_permutation(axes.begin(), axes.end()));
      }
  std::sort(out.begin(), out.end());
  return out;
}

std::array<int, 3> PeriodicKuhnGrid::displacement(std::uint64_t x, std::uint64_t y) const {
  const auto a = gridIndex(x);
  const auto b = gridIndex(y);
  bool ok = true;
  std::array<int, 3> n{};
  for (std::size_t c = 0; c < 3; ++c) n[c] = unwrapStep(a[c], b[c], n_[c], ok);
  // A Kuhn edge steps the same way along every axis it moves on: the three
  // axis steps, the three face diagonals e_a + e_b and the body diagonal, or
  // their negatives. A mixed-sign vector such as e_1 - e_2 is the other face
  // diagonal, which the triangulation does not contain.
  const bool anyPositive = n[0] > 0 || n[1] > 0 || n[2] > 0;
  const bool anyNegative = n[0] < 0 || n[1] < 0 || n[2] < 0;
  if (!ok || anyPositive == anyNegative)
    throw std::invalid_argument("PeriodicKuhnGrid: vertices " + std::to_string(x) + " and " +
                                std::to_string(y) + " are not joined by an edge of the grid");
  return n;
}

double PeriodicKuhnGrid::squaredLength(std::uint64_t x, std::uint64_t y) const {
  const auto n = displacement(x, y);
  const Gram g = stepMetric();
  double s = 0.0;
  for (std::size_t a = 0; a < 3; ++a)
    for (std::size_t b = 0; b < 3; ++b) s += n[a] * g[a][b] * n[b];
  return s;
}

std::vector<std::complex<double>> PeriodicKuhnGrid::squaredLengths(
    const std::vector<Cell> &edges) const {
  std::vector<std::complex<double>> out;
  out.reserve(edges.size());
  for (const auto &e : edges) {
    if (e.size() != 2)
      throw std::invalid_argument("PeriodicKuhnGrid::squaredLengths: an edge is a pair of vertex ids");
    out.emplace_back(squaredLength(e[0], e[1]), 0.0);
  }
  return out;
}

double PeriodicKuhnGrid::blochPhase(std::uint64_t x, std::uint64_t y,
                                    const std::array<double, 3> &kappa) const {
  const auto n = displacement(x, y);
  double phase = 0.0;
  for (std::size_t a = 0; a < 3; ++a) phase += kappa[a] * n[a] / static_cast<double>(n_[a]);
  return kTwoPi * phase;
}

std::vector<std::complex<double>> PeriodicKuhnGrid::blochLinks(
    const std::vector<Cell> &edges, const std::array<double, 3> &kappa) const {
  std::vector<std::complex<double>> out;
  out.reserve(edges.size());
  for (const auto &e : edges) {
    if (e.size() != 2)
      throw std::invalid_argument("PeriodicKuhnGrid::blochLinks: an edge is a pair of vertex ids");
    out.push_back(std::polar(1.0, blochPhase(e[0], e[1], kappa)));
  }
  return out;
}

PeriodicKuhnGrid::Walk PeriodicKuhnGrid::fundamentalCycle(int axis, std::uint64_t base) const {
  if (axis < 0 || axis > 2)
    throw std::invalid_argument("PeriodicKuhnGrid::fundamentalCycle: the axis is 0, 1 or 2");
  auto p = gridIndex(base);
  Walk walk;
  walk.reserve(static_cast<std::size_t>(n_[static_cast<std::size_t>(axis)]));
  for (int step = 0; step < n_[static_cast<std::size_t>(axis)]; ++step) {
    const std::uint64_t from = vertexId(p[0], p[1], p[2]);
    ++p[static_cast<std::size_t>(axis)];
    walk.emplace_back(from, vertexId(p[0], p[1], p[2]));
  }
  return walk;
}

void PeriodicKuhnGrid::build(Spacetime *spacetime, int /*numSimplices*/) {
  buildExplicit(spacetime, vertexCount(), cells());
  for (const auto &edge : spacetime->getEdgeList()->toVector())
    edge->setLength(std::sqrt(squaredLength(edge->getSource()->getId(), edge->getTarget()->getId())));
}

} // namespace tessera::spacetime
