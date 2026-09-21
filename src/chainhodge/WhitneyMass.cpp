// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "chainhodge/WhitneyMass.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::chainhodge {

namespace {

using Cell = std::vector<std::uint64_t>;
using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey &e) const noexcept {
    return std::hash<std::uint64_t>{}(e.first * 1000003ULL ^ (e.second + 0x9e3779b97f4a7c15ULL));
  }
};

constexpr double kPi = 3.14159265358979323846;

double factorial(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= static_cast<double>(i);
  return f;
}

// Principal square root with a negative real axis resolved to +i: a negative
// real argument carrying a signed zero imaginary part (-0.0) would otherwise
// land on the -i side. The real Lorentzian convention is uniformly +i, the
// e^{-2 i eps} side of the cut.
Complex principalSqrt(Complex z) {
  if (z.imag() == 0.0) z = Complex(z.real(), 0.0);
  return std::sqrt(z);
}

double principalArg(Complex z) {
  if (z.imag() == 0.0) z = Complex(z.real(), 0.0);
  return std::arg(z);
}

// Index maps from sorted vertex tuples to canonical C_k indices.
struct CellIndex {
  std::vector<std::map<Cell, int>> byDegree;   // byDegree[k][tuple] = index
  std::unordered_map<EdgeKey, int, EdgeKeyHash> edges;

  explicit CellIndex(const cobordism::ChainComplex &K) {
    const int d = K.dimension();
    byDegree.resize(static_cast<std::size_t>(std::max(d + 1, 0)));
    for (int k = 0; k <= d; ++k) {
      const auto cells = K.kSimplexVertices(k);
      for (int j = 0; j < static_cast<int>(cells.size()); ++j)
        byDegree[static_cast<std::size_t>(k)][cells[static_cast<std::size_t>(j)]] = j;
    }
    if (d >= 1) {
      const auto es = K.kSimplexVertices(1);
      for (int j = 0; j < static_cast<int>(es.size()); ++j)
        edges[EdgeKey{es[static_cast<std::size_t>(j)][0], es[static_cast<std::size_t>(j)][1]}] = j;
    }
  }

  [[nodiscard]] int edge(std::uint64_t a, std::uint64_t b) const {
    if (a > b) std::swap(a, b);
    const auto it = edges.find(EdgeKey{a, b});
    if (it == edges.end())
      throw std::invalid_argument("WhitneyMass: vertex pair (" + std::to_string(a) + "," +
                                  std::to_string(b) + ") is not an edge of the complex");
    return it->second;
  }

  [[nodiscard]] int cell(int k, const Cell &c) const {
    const auto &m = byDegree.at(static_cast<std::size_t>(k));
    const auto it = m.find(c);
    if (it == m.end())
      throw std::invalid_argument("WhitneyMass: a face of a top simplex is missing from C_k");
    return it->second;
  }
};

void checkInputs(const cobordism::ChainComplex &K, const SquaredLengths &s, int k) {
  const int d = K.dimension();
  if (d < 0) throw std::invalid_argument("WhitneyMass: empty complex");
  if (k < 0 || k > d)
    throw std::invalid_argument("WhitneyMass: degree k=" + std::to_string(k) +
                                " outside [0," + std::to_string(d) + "]");
  if (s.size() != K.numSimplices(1))
    throw std::invalid_argument("WhitneyMass: expected one squared length per edge (" +
                                std::to_string(K.numSimplices(1)) + "), got " +
                                std::to_string(s.size()));
}

// All (k+1)-subsets of {0..d} in lexicographic order, as local index tuples.
std::vector<std::vector<int>> subsets(int d, int k) {
  std::vector<std::vector<int>> out;
  std::vector<int> cur;
  std::function<void(int)> rec = [&](int start) {
    if (static_cast<int>(cur.size()) == k + 1) {
      out.push_back(cur);
      return;
    }
    for (int v = start; v <= d; ++v) {
      cur.push_back(v);
      rec(v + 1);
      cur.pop_back();
    }
  };
  rec(0);
  return out;
}

// The Gram matrix (d x d) of one top simplex from its local squared lengths
// s(i,j), i<j in 0..d, in local-edge order (0,1),(0,2),...,(d-1,d).
struct LocalGeometry {
  int d{0};
  std::vector<std::pair<int, int>> localEdges;  // (a,b) a<b
  Eigen::MatrixXcd gram;                        // d x d
  std::vector<Eigen::MatrixXcd> dGram;          // per local edge

  static int localEdgeIndex(int d, int a, int b) {
    // position of (a,b), a<b, in the lexicographic list of pairs
    int idx = 0;
    for (int i = 0; i < a; ++i) idx += d - i;
    return idx + (b - a - 1);
  }

  LocalGeometry(int dim, const std::vector<Complex> &sLocal) : d(dim) {
    for (int a = 0; a <= d; ++a)
      for (int b = a + 1; b <= d; ++b) localEdges.emplace_back(a, b);
    auto S = [&](int a, int b) -> Complex {
      if (a == b) return Complex(0.0, 0.0);
      if (a > b) std::swap(a, b);
      return sLocal[static_cast<std::size_t>(localEdgeIndex(d, a, b))];
    };
    gram = Eigen::MatrixXcd::Zero(d, d);
    for (int i = 1; i <= d; ++i)
      for (int j = 1; j <= d; ++j)
        gram(i - 1, j - 1) = 0.5 * (S(0, i) + S(0, j) - S(i, j));
    dGram.resize(localEdges.size());
    for (std::size_t m = 0; m < localEdges.size(); ++m) {
      const auto [a, b] = localEdges[m];
      Eigen::MatrixXcd dg = Eigen::MatrixXcd::Zero(d, d);
      if (a == 0) {
        // g_ij = 1/2 (s_0i + s_0j - s_ij): s_0b enters g_ib and g_bj with 1/2 each.
        for (int i = 1; i <= d; ++i) {
          dg(i - 1, b - 1) += 0.5;
          dg(b - 1, i - 1) += 0.5;
        }
      } else {
        dg(a - 1, b - 1) -= 0.5;
        dg(b - 1, a - 1) -= 0.5;
      }
      dGram[m] = dg;
    }
  }
};

// Extended Gamma ((d+1)x(d+1)) from the d x d inverse Gram block.
Eigen::MatrixXcd extendGamma(const Eigen::MatrixXcd &ginvBlock) {
  const int d = static_cast<int>(ginvBlock.rows());
  Eigen::MatrixXcd G = Eigen::MatrixXcd::Zero(d + 1, d + 1);
  G.block(1, 1, d, d) = ginvBlock;
  for (int j = 1; j <= d; ++j) {
    Complex col = 0.0;
    for (int i = 1; i <= d; ++i) col += ginvBlock(i - 1, j - 1);
    G(0, j) = -col;
    G(j, 0) = -col;
  }
  Complex all = 0.0;
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) all += ginvBlock(i, j);
  G(0, 0) = all;
  return G;
}

// det[Gamma_{c_p e_q}] for two k-subsets c, e of local vertices (k x k).
Complex minorDet(const Eigen::MatrixXcd &Gamma, const std::vector<int> &c,
                 const std::vector<int> &e) {
  const int k = static_cast<int>(c.size());
  if (k == 0) return Complex(1.0, 0.0);
  Eigen::MatrixXcd A(k, k);
  for (int p = 0; p < k; ++p)
    for (int q = 0; q < k; ++q) A(p, q) = Gamma(c[static_cast<std::size_t>(p)], e[static_cast<std::size_t>(q)]);
  return A.determinant();
}

// Jacobi: d det[A] = sum_p det[A with row p replaced by dA row p].
Complex minorDetDerivative(const Eigen::MatrixXcd &Gamma, const Eigen::MatrixXcd &dGamma,
                           const std::vector<int> &c, const std::vector<int> &e) {
  const int k = static_cast<int>(c.size());
  if (k == 0) return Complex(0.0, 0.0);
  Eigen::MatrixXcd A(k, k), dA(k, k);
  for (int p = 0; p < k; ++p)
    for (int q = 0; q < k; ++q) {
      A(p, q) = Gamma(c[static_cast<std::size_t>(p)], e[static_cast<std::size_t>(q)]);
      dA(p, q) = dGamma(c[static_cast<std::size_t>(p)], e[static_cast<std::size_t>(q)]);
    }
  Complex total = 0.0;
  for (int p = 0; p < k; ++p) {
    Eigen::MatrixXcd B = A;
    B.row(p) = dA.row(p);
    total += B.determinant();
  }
  return total;
}

std::vector<int> without(const std::vector<int> &sigma, int i) {
  std::vector<int> out;
  out.reserve(sigma.size() - 1);
  for (int p = 0; p < static_cast<int>(sigma.size()); ++p)
    if (p != i) out.push_back(sigma[static_cast<std::size_t>(p)]);
  return out;
}

// The local Whitney block of one top simplex at degree k, with derivatives.
struct LocalBlock {
  Eigen::MatrixXcd block;
  std::vector<Eigen::MatrixXcd> derivative;  // per local edge (empty unless requested)
  std::vector<std::vector<int>> faces;       // local (k+1)-tuples, block order
  Complex volume;
  bool ambiguous{false};
};

LocalBlock localWhitneyBlock(const LocalGeometry &geo, int k, Branch branch, bool withDerivative) {
  const int d = geo.d;
  LocalBlock out;
  out.faces = subsets(d, k);
  const int nf = static_cast<int>(out.faces.size());

  const Eigen::MatrixXcd ginv = geo.gram.inverse();
  const Eigen::MatrixXcd Gamma = extendGamma(ginv);
  out.volume = WhitneyMass::volumeOnBranch(geo.gram, branch, &out.ambiguous);
  const double kfac2 = factorial(k) * factorial(k);
  const double lam = 1.0 / static_cast<double>((d + 1) * (d + 2));

  auto entry = [&](const std::vector<int> &sig, const std::vector<int> &tau,
                   const Eigen::MatrixXcd &G, Complex vol) -> Complex {
    Complex acc = 0.0;
    for (int i = 0; i <= k; ++i) {
      const std::vector<int> c = without(sig, i);
      for (int j = 0; j <= k; ++j) {
        const std::vector<int> e = without(tau, j);
        const double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        const double delta = (sig[static_cast<std::size_t>(i)] == tau[static_cast<std::size_t>(j)]) ? 2.0 : 1.0;
        acc += sign * (vol * delta * lam) * minorDet(G, c, e);
      }
    }
    return kfac2 * acc;
  };
  // Product rule: d(I * D) = dI * D + I * dD with I = vol*delta*lam.
  auto entryDerivative = [&](const std::vector<int> &sig, const std::vector<int> &tau,
                             const Eigen::MatrixXcd &G, const Eigen::MatrixXcd &dG,
                             Complex vol, Complex dvol) -> Complex {
    Complex acc = 0.0;
    for (int i = 0; i <= k; ++i) {
      const std::vector<int> c = without(sig, i);
      for (int j = 0; j <= k; ++j) {
        const std::vector<int> e = without(tau, j);
        const double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        const double delta = (sig[static_cast<std::size_t>(i)] == tau[static_cast<std::size_t>(j)]) ? 2.0 : 1.0;
        acc += sign * ((dvol * delta * lam) * minorDet(G, c, e) +
                       (vol * delta * lam) * minorDetDerivative(G, dG, c, e));
      }
    }
    return kfac2 * acc;
  };

  out.block = Eigen::MatrixXcd::Zero(nf, nf);
  for (int p = 0; p < nf; ++p)
    for (int q = p; q < nf; ++q) {
      const Complex v = entry(out.faces[static_cast<std::size_t>(p)], out.faces[static_cast<std::size_t>(q)], Gamma, out.volume);
      out.block(p, q) = v;
      out.block(q, p) = v;
    }

  if (withDerivative) {
    out.derivative.resize(geo.localEdges.size());
    for (std::size_t m = 0; m < geo.localEdges.size(); ++m) {
      const Eigen::MatrixXcd dginv = -ginv * geo.dGram[m] * ginv;
      const Eigen::MatrixXcd dGamma = extendGamma(dginv);
      const Complex dvol = 0.5 * out.volume * (ginv * geo.dGram[m]).trace();
      Eigen::MatrixXcd dB = Eigen::MatrixXcd::Zero(nf, nf);
      for (int p = 0; p < nf; ++p)
        for (int q = p; q < nf; ++q) {
          const Complex v = entryDerivative(out.faces[static_cast<std::size_t>(p)], out.faces[static_cast<std::size_t>(q)],
                                            Gamma, dGamma, out.volume, dvol);
          dB(p, q) = v;
          dB(q, p) = v;
        }
      out.derivative[m] = std::move(dB);
    }
  }
  return out;
}

// Gather one top simplex's local squared lengths, cell indices, and edge indices.
struct TopSimplexContext {
  Cell verts;
  std::vector<Complex> sLocal;
  std::vector<int> edgeIndices;
};

TopSimplexContext topContext(const Cell &T, const SquaredLengths &s, const CellIndex &index) {
  TopSimplexContext ctx;
  ctx.verts = T;
  const int d = static_cast<int>(T.size()) - 1;
  for (int a = 0; a <= d; ++a)
    for (int b = a + 1; b <= d; ++b) {
      const int e = index.edge(T[static_cast<std::size_t>(a)], T[static_cast<std::size_t>(b)]);
      ctx.edgeIndices.push_back(e);
      ctx.sLocal.push_back(s[static_cast<std::size_t>(e)]);
    }
  return ctx;
}

std::vector<int> faceIndices(const Cell &T, const std::vector<std::vector<int>> &faces, int k,
                             const CellIndex &index) {
  std::vector<int> out;
  out.reserve(faces.size());
  for (const auto &f : faces) {
    Cell c;
    c.reserve(f.size());
    for (int v : f) c.push_back(T[static_cast<std::size_t>(v)]);
    out.push_back(index.cell(k, c));
  }
  return out;
}

// Roots of a polynomial with complex coefficients c[0] + c[1] t + ... + c[n] t^n
// via the companion matrix; trailing (leading-power) coefficients below the
// relative tolerance drop the degree.
std::vector<Complex> polynomialRoots(std::vector<Complex> c) {
  double scale = 0.0;
  for (const auto &v : c) scale = std::max(scale, std::abs(v));
  while (c.size() > 1 && std::abs(c.back()) <= 1e-14 * scale) c.pop_back();
  const int n = static_cast<int>(c.size()) - 1;
  if (n < 1) return {};
  Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(n, n);
  for (int i = 1; i < n; ++i) C(i, i - 1) = 1.0;
  for (int i = 0; i < n; ++i) C(i, n - 1) = -c[static_cast<std::size_t>(i)] / c[static_cast<std::size_t>(n)];
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(C, false);
  std::vector<Complex> roots;
  for (int i = 0; i < n; ++i) roots.push_back(es.eigenvalues()(i));
  return roots;
}

// Coefficients of the degree-<=d polynomial p(t) = det((1-t) gref + t g) by
// exact interpolation at the nodes t = 0, 1/d, ..., 1.
std::vector<Complex> gramSegmentPolynomial(const Eigen::MatrixXcd &gref, const Eigen::MatrixXcd &g) {
  const int d = static_cast<int>(g.rows());
  const int n = d + 1;
  Eigen::MatrixXcd V(n, n);
  Eigen::VectorXcd y(n);
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(d);
    const Eigen::MatrixXcd gt = (1.0 - t) * gref + t * g;
    y(i) = gt.determinant();
    Complex tp = 1.0;
    for (int j = 0; j < n; ++j) {
      V(i, j) = tp;
      tp *= t;
    }
  }
  const Eigen::VectorXcd coef = V.fullPivLu().solve(y);
  std::vector<Complex> c(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) c[static_cast<std::size_t>(j)] = coef(j);
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------

cobordism::ChainComplex WhitneyMass::complexOf(const spacetime::Spacetime &st) {
  std::size_t topSize = 0;
  for (const auto &sp : st.getSimplices())
    if (sp != nullptr) topSize = std::max(topSize, static_cast<std::size_t>(sp->size()));
  std::vector<Cell> cells;
  for (const auto &sp : st.getSimplices()) {
    if (sp == nullptr || static_cast<std::size_t>(sp->size()) != topSize) continue;
    Cell c;
    for (const auto &v : sp->getVertices()) c.push_back(v->getId());
    std::sort(c.begin(), c.end());
    cells.push_back(std::move(c));
  }
  return cobordism::ChainComplex::fromTopCells(cells);
}

SquaredLengths WhitneyMass::squaredLengthsOf(const spacetime::Spacetime &st,
                                             const cobordism::ChainComplex &K) {
  std::unordered_map<EdgeKey, Complex, EdgeKeyHash> byPair;
  if (st.getEdgeList()) {
    for (const auto &e : st.getEdgeList()->toVector()) {
      if (e == nullptr) continue;
      std::uint64_t a = e->getSource()->getId();
      std::uint64_t b = e->getTarget()->getId();
      if (a > b) std::swap(a, b);
      const Complex l = e->getLength();
      byPair[EdgeKey{a, b}] = l * l;
    }
  }
  SquaredLengths s;
  const auto edges = K.kSimplexVertices(1);
  s.reserve(edges.size());
  for (const auto &e : edges) {
    const auto it = byPair.find(EdgeKey{e[0], e[1]});
    if (it == byPair.end())
      throw std::invalid_argument("WhitneyMass::squaredLengthsOf: complex edge (" +
                                  std::to_string(e[0]) + "," + std::to_string(e[1]) +
                                  ") has no spacetime edge");
    s.push_back(it->second);
  }
  return s;
}

Complex WhitneyMass::volumeOnBranch(const Eigen::MatrixXcd &gram, Branch branch, bool *ambiguous) {
  const int d = static_cast<int>(gram.rows());
  if (ambiguous != nullptr) *ambiguous = false;
  if (d == 0) return Complex(1.0, 0.0);

  auto kontsevichSegal = [&]() -> Complex {
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(gram, false);
    Complex root = 1.0;
    for (int i = 0; i < d; ++i) root *= principalSqrt(es.eigenvalues()(i));
    return root;
  };

  Complex sqrtDet;
  if (branch == Branch::KontsevichSegal) {
    sqrtDet = kontsevichSegal();
  } else {
    // Unit Euclidean reference simplex: g_ref = 1/2 (1 + delta_ij).
    Eigen::MatrixXcd gref = Eigen::MatrixXcd::Constant(d, d, Complex(0.5, 0.0));
    for (int i = 0; i < d; ++i) gref(i, i) = 1.0;
    const std::vector<Complex> coef = gramSegmentPolynomial(gref, gram);
    const std::vector<Complex> roots = polynomialRoots(coef);
    bool onSegment = false;
    double dtheta = 0.0;
    for (const auto &r : roots) {
      const double tol = 1e-12 * (1.0 + std::abs(r));
      if (std::abs(r.imag()) <= tol && r.real() >= -tol && r.real() <= 1.0 + tol) {
        onSegment = true;
        break;
      }
      dtheta += principalArg(Complex(1.0, 0.0) - r) - principalArg(-r);
    }
    const Complex det1 = gram.determinant();
    if (onSegment || roots.empty()) {
      // A root on the reference segment (or a degree-0 polynomial, i.e. a
      // degenerate reference) leaves no continuous continuation: report and
      // take the Kontsevich-Segal value.
      if (ambiguous != nullptr) *ambiguous = onSegment;
      sqrtDet = kontsevichSegal();
    } else {
      // The continuation decides which of the two square roots of det g is
      // meant; the value is then that root itself. Building it from the
      // tracked argument instead would carry the error of the polynomial
      // roots (a multiple root is only accurate to a fractional power of the
      // rounding level), which on real positive data shows up as a spurious
      // imaginary part of the volume.
      const Complex root = std::sqrt(det1);
      const Complex tracked = std::exp(Complex(0.0, 0.5 * dtheta));
      sqrtDet = (root * std::conj(tracked)).real() >= 0.0 ? root : -root;
    }
  }
  return sqrtDet / factorial(d);
}

double WhitneyMass::marginOf(const Eigen::MatrixXcd &gram) {
  const int d = static_cast<int>(gram.rows());
  if (d == 0) return kPi;
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(gram, false);
  double sum = 0.0;
  for (int i = 0; i < d; ++i) sum += std::abs(principalArg(es.eigenvalues()(i)));
  return kPi - sum;
}

std::vector<TopSimplexBlock> WhitneyMass::topSimplexBlocks(const cobordism::ChainComplex &K,
                                                           const SquaredLengths &s, int k,
                                                           Branch branch, bool withDerivative) {
  checkInputs(K, s, k);
  const CellIndex index(K);
  const auto tops = K.orientedTopSimplices();
  std::vector<TopSimplexBlock> out;
  out.reserve(tops.size());
  for (std::size_t t = 0; t < tops.size(); ++t) {
    const TopSimplexContext ctx = topContext(tops[t], s, index);
    const LocalGeometry geo(static_cast<int>(tops[t].size()) - 1, ctx.sLocal);
    LocalBlock lb = localWhitneyBlock(geo, k, branch, withDerivative);
    TopSimplexBlock b;
    b.topIndex = t;
    b.cellIndices = faceIndices(tops[t], lb.faces, k, index);
    b.edgeIndices = ctx.edgeIndices;
    b.block = std::move(lb.block);
    b.derivative = std::move(lb.derivative);
    out.push_back(std::move(b));
  }
  return out;
}

SparseMatrix WhitneyMass::assemble(const cobordism::ChainComplex &K, const SquaredLengths &s,
                                   int k, Branch branch) {
  checkInputs(K, s, k);
  const int n = static_cast<int>(K.numSimplices(k));
  std::vector<Eigen::Triplet<Complex>> trip;
  for (const auto &b : topSimplexBlocks(K, s, k, branch, false)) {
    const int nf = static_cast<int>(b.cellIndices.size());
    for (int p = 0; p < nf; ++p)
      for (int q = 0; q < nf; ++q)
        trip.emplace_back(b.cellIndices[static_cast<std::size_t>(p)],
                          b.cellIndices[static_cast<std::size_t>(q)], b.block(p, q));
  }
  SparseMatrix M(n, n);
  M.setFromTriplets(trip.begin(), trip.end());
  M.makeCompressed();
  return M;
}

SparseMatrix WhitneyMass::assemble(const cobordism::ChainComplex &K, const SquaredLengths &s,
                                   int k, Preset preset, Branch branch) {
  return preset == Preset::GRASSMANN_ALL ? assembleGrassmann(K, s, k) : assemble(K, s, k, branch);
}

SparseMatrix WhitneyMass::assembleGrassmann(const cobordism::ChainComplex &K,
                                            const SquaredLengths &s, int k) {
  checkInputs(K, s, k);
  const CellIndex index(K);
  const int d = K.dimension();
  const int n = static_cast<int>(K.numSimplices(k));
  auto S = [&](std::uint64_t a, std::uint64_t b) -> Complex {
    if (a == b) return Complex(0.0, 0.0);
    return s[static_cast<std::size_t>(index.edge(a, b))];
  };
  // <u_ab, u_cd> by polarization.
  auto dot = [&](std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t e) -> Complex {
    return 0.5 * (S(b, c) + S(a, e) - S(b, e) - S(a, c));
  };
  auto blade = [&](const Cell &sig, const Cell &tau) -> Complex {
    if (k == 0) return Complex(1.0, 0.0);
    Eigen::MatrixXcd A(k, k);
    for (int i = 1; i <= k; ++i)
      for (int j = 1; j <= k; ++j)
        A(i - 1, j - 1) = dot(sig[0], sig[static_cast<std::size_t>(i)], tau[0], tau[static_cast<std::size_t>(j)]);
    return A.determinant() / (factorial(k) * factorial(k));
  };
  std::map<std::pair<int, int>, Complex> value;   // blade pairing, computed once
  std::map<std::pair<int, int>, double> multiplicity;
  for (int kk = k; kk <= d; ++kk) {
    const auto faces = subsets(kk, k);
    for (const auto &rho : K.kSimplexVertices(kk)) {
      std::vector<std::pair<int, Cell>> sub;
      sub.reserve(faces.size());
      for (const auto &f : faces) {
        Cell c;
        for (int v : f) c.push_back(rho[static_cast<std::size_t>(v)]);
        sub.emplace_back(index.cell(k, c), std::move(c));
      }
      for (const auto &[i, a] : sub)
        for (const auto &[j, b] : sub) {
          multiplicity[{i, j}] += 1.0;
          if (value.find({i, j}) == value.end()) value[{i, j}] = blade(a, b);
        }
    }
  }
  std::vector<Eigen::Triplet<Complex>> trip;
  trip.reserve(value.size());
  for (const auto &[ij, v] : value) trip.emplace_back(ij.first, ij.second, multiplicity[ij] * v);
  SparseMatrix G(n, n);
  G.setFromTriplets(trip.begin(), trip.end());
  G.makeCompressed();
  return G;
}

double WhitneyMass::allowabilityMargin(const cobordism::ChainComplex &K, const SquaredLengths &s) {
  return certificate(K, s, Branch::KontsevichSegal).margin;
}

InstanceCertificate WhitneyMass::certificate(const cobordism::ChainComplex &K,
                                             const SquaredLengths &s, Branch branch) {
  const int d = K.dimension();
  if (d < 0) throw std::invalid_argument("WhitneyMass: empty complex");
  checkInputs(K, s, 0);
  const CellIndex index(K);
  InstanceCertificate cert;
  cert.branch = branch;
  const auto tops = K.orientedTopSimplices();
  cert.margins.reserve(tops.size());
  cert.volumes.reserve(tops.size());
  cert.gramDeterminants.reserve(tops.size());
  double minMargin = std::numeric_limits<double>::infinity();
  for (std::size_t t = 0; t < tops.size(); ++t) {
    const TopSimplexContext ctx = topContext(tops[t], s, index);
    const LocalGeometry geo(d, ctx.sLocal);
    const double m = marginOf(geo.gram);
    bool amb = false;
    const Complex vol = volumeOnBranch(geo.gram, branch, &amb);
    cert.margins.push_back(m);
    cert.volumes.push_back(vol);
    cert.gramDeterminants.push_back(geo.gram.determinant());
    if (amb) {
      cert.continuationAmbiguous = true;
      cert.ambiguousTopSimplices.push_back(t);
    }
    minMargin = std::min(minMargin, m);
  }
  cert.margin = minMargin;
  cert.allowable = !tops.empty() && minMargin > 0.0;
  return cert;
}

SparseMatrix WhitneyMass::assembleDerivative(const cobordism::ChainComplex &K,
                                             const SquaredLengths &s, int k,
                                             std::size_t edgeIndex, Branch branch) {
  checkInputs(K, s, k);
  if (edgeIndex >= K.numSimplices(1))
    throw std::invalid_argument("WhitneyMass::assembleDerivative: edge index out of range");
  const int n = static_cast<int>(K.numSimplices(k));
  std::vector<Eigen::Triplet<Complex>> trip;
  for (const auto &b : topSimplexBlocks(K, s, k, branch, true)) {
    for (std::size_t m = 0; m < b.edgeIndices.size(); ++m) {
      if (static_cast<std::size_t>(b.edgeIndices[m]) != edgeIndex) continue;
      const int nf = static_cast<int>(b.cellIndices.size());
      for (int p = 0; p < nf; ++p)
        for (int q = 0; q < nf; ++q)
          trip.emplace_back(b.cellIndices[static_cast<std::size_t>(p)],
                            b.cellIndices[static_cast<std::size_t>(q)], b.derivative[m](p, q));
    }
  }
  SparseMatrix D(n, n);
  D.setFromTriplets(trip.begin(), trip.end());
  D.makeCompressed();
  return D;
}

std::vector<Complex> WhitneyMass::derivativeContraction(const cobordism::ChainComplex &K,
                                                        const SquaredLengths &s, int k,
                                                        const Eigen::MatrixXcd &X,
                                                        const Eigen::MatrixXcd &Y,
                                                        Branch branch) {
  checkInputs(K, s, k);
  const int n = static_cast<int>(K.numSimplices(k));
  if (X.rows() != n || Y.rows() != n || X.cols() != Y.cols())
    throw std::invalid_argument("WhitneyMass::derivativeContraction: X and Y must be n_k x m");
  std::vector<Complex> out(K.numSimplices(1), Complex(0.0, 0.0));
  for (const auto &b : topSimplexBlocks(K, s, k, branch, true)) {
    const int nf = static_cast<int>(b.cellIndices.size());
    Eigen::MatrixXcd XT(nf, X.cols()), YT(nf, Y.cols());
    for (int p = 0; p < nf; ++p) {
      XT.row(p) = X.row(b.cellIndices[static_cast<std::size_t>(p)]);
      YT.row(p) = Y.row(b.cellIndices[static_cast<std::size_t>(p)]);
    }
    for (std::size_t m = 0; m < b.edgeIndices.size(); ++m) {
      // tr(X_T^T dM Y_T): the transpose pairing, never the conjugate.
      const Complex c = (XT.transpose() * b.derivative[m] * YT).trace();
      out[static_cast<std::size_t>(b.edgeIndices[m])] += c;
    }
  }
  return out;
}

namespace {

// |T| d! prod_v m_v! / (d + m)! without the volume: the multiplicities of the
// local vertices 0..d among the m factors.
double productWeight(int d, const std::vector<int> &multiplicity) {
  int m = 0;
  double numerator = factorial(d);
  for (int mv : multiplicity) {
    m += mv;
    numerator *= factorial(mv);
  }
  return numerator / factorial(d + m);
}

// The (d+1)^3 weights of the three-factor integral on one top simplex.
std::vector<double> tripleWeights(int d) {
  const int nv = d + 1;
  std::vector<double> w(static_cast<std::size_t>(nv) * nv * nv);
  std::vector<int> multiplicity(static_cast<std::size_t>(nv));
  for (int a = 0; a < nv; ++a)
    for (int b = 0; b < nv; ++b)
      for (int c = 0; c < nv; ++c) {
        std::fill(multiplicity.begin(), multiplicity.end(), 0);
        ++multiplicity[static_cast<std::size_t>(a)];
        ++multiplicity[static_cast<std::size_t>(b)];
        ++multiplicity[static_cast<std::size_t>(c)];
        w[(static_cast<std::size_t>(a) * nv + b) * nv + c] = productWeight(d, multiplicity);
      }
  return w;
}

Complex topVolume(const Cell &T, const SquaredLengths &s, const CellIndex &index, Branch branch) {
  const TopSimplexContext ctx = topContext(T, s, index);
  const LocalGeometry geo(static_cast<int>(T.size()) - 1, ctx.sLocal);
  return WhitneyMass::volumeOnBranch(geo.gram, branch);
}

}  // namespace

Complex WhitneyMass::vertexProductIntegral(const cobordism::ChainComplex &K, const SquaredLengths &s,
                                           std::size_t topIndex,
                                           const std::vector<std::uint64_t> &vertices, Branch branch) {
  checkInputs(K, s, 0);
  const auto tops = K.orientedTopSimplices();
  if (topIndex >= tops.size())
    throw std::invalid_argument("WhitneyMass::vertexProductIntegral: top simplex index out of range");
  const Cell &T = tops[topIndex];
  std::vector<int> multiplicity(T.size(), 0);
  for (std::uint64_t v : vertices) {
    const auto it = std::find(T.begin(), T.end(), v);
    if (it == T.end())
      throw std::invalid_argument("WhitneyMass::vertexProductIntegral: vertex " + std::to_string(v) +
                                  " is not a vertex of the top simplex");
    ++multiplicity[static_cast<std::size_t>(it - T.begin())];
  }
  const CellIndex index(K);
  return topVolume(T, s, index, branch) * productWeight(static_cast<int>(T.size()) - 1, multiplicity);
}

SparseMatrix WhitneyMass::assembleVertexPotential(const cobordism::ChainComplex &K,
                                                  const SquaredLengths &s,
                                                  const std::vector<Complex> &potential, Branch branch) {
  checkInputs(K, s, 0);
  const int n = static_cast<int>(K.numSimplices(0));
  if (static_cast<int>(potential.size()) != n)
    throw std::invalid_argument("WhitneyMass::assembleVertexPotential: expected one value per vertex (" +
                                std::to_string(n) + "), got " + std::to_string(potential.size()));
  const CellIndex index(K);
  const int d = K.dimension();
  const int nv = d + 1;
  const std::vector<double> weight = tripleWeights(d);
  std::vector<Eigen::Triplet<Complex>> trip;
  const auto tops = K.orientedTopSimplices();
  trip.reserve(tops.size() * static_cast<std::size_t>(nv) * nv);
  std::vector<int> local(static_cast<std::size_t>(nv));
  for (const auto &T : tops) {
    const Complex volume = topVolume(T, s, index, branch);
    for (int a = 0; a < nv; ++a) local[static_cast<std::size_t>(a)] = index.cell(0, Cell{T[static_cast<std::size_t>(a)]});
    for (int a = 0; a < nv; ++a)
      for (int b = 0; b < nv; ++b) {
        Complex sum(0.0, 0.0);
        for (int c = 0; c < nv; ++c)
          sum += potential[static_cast<std::size_t>(local[static_cast<std::size_t>(c)])] *
                 weight[(static_cast<std::size_t>(a) * nv + b) * nv + c];
        trip.emplace_back(local[static_cast<std::size_t>(a)], local[static_cast<std::size_t>(b)], volume * sum);
      }
  }
  SparseMatrix M(n, n);
  M.setFromTriplets(trip.begin(), trip.end());
  M.makeCompressed();
  return M;
}

std::vector<Complex> WhitneyMass::vertexDensityContraction(const cobordism::ChainComplex &K,
                                                           const SquaredLengths &s,
                                                           const Eigen::MatrixXcd &X,
                                                           const Eigen::MatrixXcd &Y, Branch branch) {
  checkInputs(K, s, 0);
  const auto n = static_cast<Eigen::Index>(K.numSimplices(0));
  if (X.rows() != n || Y.rows() != n || X.cols() != Y.cols())
    throw std::invalid_argument(
        "WhitneyMass::vertexDensityContraction: X and Y must be n_0 x m with the same m");
  const CellIndex index(K);
  const int d = K.dimension();
  const int nv = d + 1;
  const std::vector<double> weight = tripleWeights(d);
  std::vector<Complex> rho(static_cast<std::size_t>(n), Complex(0.0, 0.0));
  std::vector<int> local(static_cast<std::size_t>(nv));
  Eigen::MatrixXcd pairing(nv, nv);
  for (const auto &T : K.orientedTopSimplices()) {
    const Complex volume = topVolume(T, s, index, branch);
    for (int a = 0; a < nv; ++a) local[static_cast<std::size_t>(a)] = index.cell(0, Cell{T[static_cast<std::size_t>(a)]});
    for (int a = 0; a < nv; ++a)
      for (int b = 0; b < nv; ++b)
        pairing(a, b) = (X.row(local[static_cast<std::size_t>(a)]).array() *
                         Y.row(local[static_cast<std::size_t>(b)]).array()).sum();
    for (int c = 0; c < nv; ++c) {
      Complex sum(0.0, 0.0);
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
          sum += pairing(a, b) * weight[(static_cast<std::size_t>(a) * nv + b) * nv + c];
      rho[static_cast<std::size_t>(local[static_cast<std::size_t>(c)])] += volume * sum;
    }
  }
  return rho;
}

}  // namespace tessera::chainhodge
