// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/IsospinDoublet.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "chainhodge/RieszBand.h"
#include "cobordism/Certificate.h"

namespace tessera::observables {

namespace {

using cd = std::complex<double>;
using Mat = Eigen::MatrixXcd;
using cobordism::Certificate;
using cobordism::CertificateDomain;
using cobordism::CertificateRegime;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// The largest band rank whose commutant is decomposed densely. The commutator
/// map of a rank-R band acts on R^2 coordinates.
constexpr Eigen::Index kMaxDecomposedRank = 24;
/// The largest commutant whose centre is decomposed.
constexpr Eigen::Index kMaxDecomposedCommutant = 100;

double spectralNorm(const Mat& a) {
  if (a.size() == 0) return 0.0;
  Eigen::JacobiSVD<Mat> svd(a);
  return svd.singularValues()(0);
}

std::string fmt(double x) {
  std::ostringstream s;
  s.precision(4);
  s << x;
  return s.str();
}

std::string fmt(cd x) {
  std::ostringstream s;
  s.precision(6);
  s << "(" << x.real() << (x.imag() < 0 ? "" : "+") << x.imag() << "j)";
  return s.str();
}

bool lessComplex(const cd& a, const cd& b) {
  if (a.real() != b.real()) return a.real() < b.real();
  return a.imag() < b.imag();
}

/// Single-linkage clusters of complex numbers at an absolute width.
std::vector<std::vector<Eigen::Index>> clusters(const std::vector<cd>& values,
                                                double width) {
  const auto n = static_cast<Eigen::Index>(values.size());
  std::vector<Eigen::Index> parent(static_cast<std::size_t>(n));
  std::iota(parent.begin(), parent.end(), Eigen::Index{0});
  const auto find = [&](Eigen::Index x) {
    while (parent[static_cast<std::size_t>(x)] != x)
      x = parent[static_cast<std::size_t>(x)] =
          parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    return x;
  };
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = i + 1; j < n; ++j)
      if (std::abs(values[static_cast<std::size_t>(i)] -
                   values[static_cast<std::size_t>(j)]) <= width)
        parent[static_cast<std::size_t>(find(i))] = find(j);
  std::map<Eigen::Index, std::vector<Eigen::Index>> groups;
  for (Eigen::Index i = 0; i < n; ++i) groups[find(i)].push_back(i);
  std::vector<std::vector<Eigen::Index>> out;
  for (auto& [root, members] : groups) out.push_back(std::move(members));
  const auto centre = [&](const std::vector<Eigen::Index>& g) {
    cd sum(0.0, 0.0);
    for (const Eigen::Index i : g) sum += values[static_cast<std::size_t>(i)];
    return sum / static_cast<double>(g.size());
  };
  std::sort(out.begin(), out.end(),
            [&](const auto& a, const auto& b) {
              return lessComplex(centre(a), centre(b));
            });
  return out;
}

/// Orthonormal basis of the column span of `a`, at a relative cut.
Mat orthonormalSpan(const Mat& a, double relativeCut) {
  if (a.cols() == 0 || a.rows() == 0) return Mat(a.rows(), 0);
  Eigen::JacobiSVD<Mat> svd(a, Eigen::ComputeThinU);
  const Eigen::VectorXd& s = svd.singularValues();
  Eigen::Index r = 0;
  for (Eigen::Index i = 0; i < s.size(); ++i)
    if (s(i) > relativeCut * std::max(s(0), 1e-300)) ++r;
  return svd.matrixU().leftCols(r);
}

/// Subspace overlap sum cos^2 / max(ranks) of two column spans.
double subspaceOverlap(const Mat& a, const Mat& b) {
  const Mat qa = orthonormalSpan(a, 1e-10);
  const Mat qb = orthonormalSpan(b, 1e-10);
  if (qa.cols() == 0 || qb.cols() == 0) return 0.0;
  Eigen::JacobiSVD<Mat> svd(qa.adjoint() * qb);
  const Eigen::VectorXd s = svd.singularValues();
  return s.squaredNorm() / static_cast<double>(std::max(qa.cols(), qb.cols()));
}

/// The sheet matrix units I (x) e_st on the declared cells: cell (b, t) is sent
/// to cell (b, s).
std::vector<Mat> sheetUnits(Eigen::Index n,
                            const std::vector<std::size_t>& sheetOfCell,
                            const std::vector<std::size_t>& baseCellOfCell,
                            std::size_t& sheetCount) {
  sheetCount = 1;
  if (sheetOfCell.empty()) return {};
  sheetCount = *std::max_element(sheetOfCell.begin(), sheetOfCell.end()) + 1;
  std::map<std::pair<std::size_t, std::size_t>, Eigen::Index> cellOf;
  std::map<std::size_t, std::size_t> sheetsOfBase;
  for (Eigen::Index i = 0; i < n; ++i) {
    const std::size_t s = sheetOfCell[static_cast<std::size_t>(i)];
    const std::size_t b = baseCellOfCell.empty()
                              ? static_cast<std::size_t>(i)
                              : baseCellOfCell[static_cast<std::size_t>(i)];
    if (!cellOf.emplace(std::make_pair(b, s), i).second)
      throw std::invalid_argument(
          "IsospinDoublet: two cells share one base cell and one sheet; the "
          "sheet declaration does not describe copies of one base complex");
    ++sheetsOfBase[b];
  }
  for (const auto& [b, count] : sheetsOfBase)
    if (count != sheetCount)
      throw std::invalid_argument(
          "IsospinDoublet: base cell " + std::to_string(b) + " appears on " +
          std::to_string(count) + " of " + std::to_string(sheetCount) +
          " sheets; every sheet must carry a copy of every base cell");
  if (sheetCount < 2) return {};
  std::vector<Mat> units;
  for (std::size_t s = 0; s < sheetCount; ++s)
    for (std::size_t t = 0; t < sheetCount; ++t) {
      Mat x = Mat::Zero(n, n);
      for (const auto& [b, count] : sheetsOfBase)
        x(cellOf.at({b, s}), cellOf.at({b, t})) = 1.0;
      units.push_back(std::move(x));
    }
  return units;
}

double invarianceResidual(const Mat& p, const std::vector<Mat>& generators) {
  double worst = 0.0;
  const double pn = std::max(p.norm(), 1e-300);
  for (const Mat& x : generators)
    worst = std::max(worst, (p * x - x * p).norm() /
                                (pn * std::max(x.norm(), 1e-300)));
  return worst;
}

/// Orthonormal (Frobenius) basis of the commutant of `generators` on C^R, as
/// R x R matrices. With no generator it is the whole matrix algebra.
std::vector<Mat> commutant(const std::vector<Mat>& generators, Eigen::Index r,
                           double tolerance) {
  const Eigen::Index r2 = r * r;
  Mat gram = Mat::Zero(r2, r2);
  const Mat id = Mat::Identity(r, r);
  for (const Mat& a : generators) {
    // vec(A X - X A) = (I (x) A - A^T (x) I) vec X, column-major vec.
    Mat k = Mat::Zero(r2, r2);
    for (Eigen::Index i = 0; i < r; ++i)
      for (Eigen::Index j = 0; j < r; ++j) {
        // block (i, j) of I (x) A is delta_ij A; of A^T (x) I is A(j, i) I.
        if (i == j) k.block(i * r, j * r, r, r) += a;
        k.block(i * r, j * r, r, r) -= a(j, i) * id;
      }
    gram.noalias() += k.adjoint() * k;
  }
  Eigen::SelfAdjointEigenSolver<Mat> eig(gram);
  const Eigen::VectorXd& w = eig.eigenvalues();
  const double scale = std::max(1.0, w.size() > 0 ? w(w.size() - 1) : 0.0);
  std::vector<Mat> basis;
  for (Eigen::Index c = 0; c < r2; ++c)
    if (w(c) <= tolerance * scale) {
      Mat x(r, r);
      for (Eigen::Index j = 0; j < r; ++j)
        x.col(j) = eig.eigenvectors().col(c).segment(j * r, r);
      basis.push_back(std::move(x));
    }
  return basis;
}

struct Content {
  std::size_t commutantDimension = 0;
  std::size_t isotypeCount = 0;
  std::vector<std::size_t> irreducibleDimensions{};
  std::vector<std::size_t> multiplicities{};
  bool decomposed = false;
};

/// The isotypic decomposition of a band under the acting algebra, from the
/// commutant basis: the centre fixes the isotypes, and f^2 is the dimension of
/// each isotypic block of the commutant.
Content decompose(const std::vector<Mat>& basis, Eigen::Index r,
                  std::size_t colourFactor, double tolerance) {
  Content out;
  out.commutantDimension = basis.size();
  const auto m = static_cast<Eigen::Index>(basis.size());
  if (m == 0) return out;
  if (m > kMaxDecomposedCommutant) return out;
  // The centre: coefficients c with sum_a c_a [B_a, B_j] = 0 for every j.
  Mat gram = Mat::Zero(m, m);
  std::vector<std::vector<Mat>> comm(static_cast<std::size_t>(m));
  for (Eigen::Index a = 0; a < m; ++a)
    for (Eigen::Index j = 0; j < m; ++j)
      comm[static_cast<std::size_t>(a)].push_back(
          basis[static_cast<std::size_t>(a)] * basis[static_cast<std::size_t>(j)] -
          basis[static_cast<std::size_t>(j)] * basis[static_cast<std::size_t>(a)]);
  for (Eigen::Index a = 0; a < m; ++a)
    for (Eigen::Index b = a; b < m; ++b) {
      cd sum(0.0, 0.0);
      for (Eigen::Index j = 0; j < m; ++j)
        sum += (comm[static_cast<std::size_t>(a)][static_cast<std::size_t>(j)]
                    .adjoint() *
                comm[static_cast<std::size_t>(b)][static_cast<std::size_t>(j)])
                   .trace();
      gram(a, b) = sum;
      gram(b, a) = std::conj(sum);
    }
  Eigen::SelfAdjointEigenSolver<Mat> eig(gram);
  const Eigen::VectorXd& w = eig.eigenvalues();
  const double scale = std::max(1.0, w(w.size() - 1));
  std::vector<Mat> centre;
  for (Eigen::Index c = 0; c < m; ++c)
    if (w(c) <= tolerance * scale) {
      Mat z = Mat::Zero(r, r);
      for (Eigen::Index a = 0; a < m; ++a)
        z += eig.eigenvectors()(a, c) * basis[static_cast<std::size_t>(a)];
      centre.push_back(std::move(z));
    }
  out.isotypeCount = centre.size();
  // A generic central element with fixed deterministic coefficients; its
  // eigenspaces are the isotypic components.
  Mat generic = Mat::Zero(r, r);
  for (std::size_t c = 0; c < centre.size(); ++c)
    generic += cd(1.0 / (static_cast<double>(c) + std::sqrt(2.0)),
                  1.0 / (static_cast<double>(c) + std::sqrt(3.0))) *
               centre[c];
  Eigen::ComplexEigenSolver<Mat> ces(generic);
  std::vector<cd> values(ces.eigenvalues().data(),
                         ces.eigenvalues().data() + r);
  double valueScale = 0.0;
  for (const cd& v : values) valueScale = std::max(valueScale, std::abs(v));
  const auto groups = clusters(values, 1e-6 * std::max(valueScale, 1e-300));
  const Mat v = ces.eigenvectors();
  const Mat vinv = v.inverse();
  for (const auto& g : groups) {
    Mat e = Mat::Zero(r, r);
    for (const Eigen::Index i : g) e += v.col(i) * vinv.row(i);
    // f^2 = dim e A' e.
    Mat stack(r * r, m);
    for (Eigen::Index a = 0; a < m; ++a) {
      const Mat block = e * basis[static_cast<std::size_t>(a)] * e;
      stack.col(a) = Eigen::Map<const Eigen::VectorXcd>(block.data(), r * r);
    }
    Eigen::JacobiSVD<Mat> svd(stack);
    const Eigen::VectorXd& s = svd.singularValues();
    std::size_t rank = 0;
    for (Eigen::Index i = 0; i < s.size(); ++i)
      if (s(i) > 1e-6 * std::max(s(0), 1e-300)) ++rank;
    const auto f = static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(rank))));
    const std::size_t n = g.size();
    out.multiplicities.push_back(f);
    out.irreducibleDimensions.push_back(
        f == 0 ? 0 : n / (f * std::max<std::size_t>(colourFactor, 1)));
  }
  out.decomposed = true;
  return out;
}

QuarkConditionRead combine(int number, const std::vector<QuarkConditionEvidence>& ev) {
  QuarkConditionRead read;
  read.number = number;
  read.name = IsospinDoublet::conditionNames()[static_cast<std::size_t>(number - 1)];
  read.statement = IsospinDoublet::statement(number);
  read.evidence = ev;
  for (const QuarkConditionEvidence& e : ev) {
    if (!e.held.has_value())
      read.missing.push_back(e.name);
    else if (!*e.held)
      read.failing.push_back(e.name);
  }
  if (!read.failing.empty())
    read.status = QuarkConditionStatus::Failed;
  else if (!read.missing.empty() || ev.empty())
    read.status = QuarkConditionStatus::NotEvaluable;
  else
    read.status = QuarkConditionStatus::Passed;
  return read;
}

QuarkConditionEvidence evidence(std::string name, std::optional<bool> held,
                                std::string detail) {
  return QuarkConditionEvidence{std::move(name), held, std::move(detail)};
}

QuarkConditionRead wardCondition() {
  return combine(
      3,
      {evidence("ward-flux-agreement", std::nullopt,
                "not evaluable: the Ward current of WP v16 Section 13.4 is "
                "deferred by the user; that section also records that the Ward "
                "flux counts fermions (it equals N_q) and that a "
                "flavour-dependent charge carries no Ward current in the "
                "declared fields")});
}

void requireNumber(int number, const char* where) {
  if (number < 1 || number > IsospinDoublet::kConditionCount)
    throw std::invalid_argument(std::string(where) +
                                ": the isospin-doublet conditions are numbered "
                                "1 to 3; got " +
                                std::to_string(number));
}

/// The restriction of each generator to a band, Phi~^T A Phi.
std::vector<Mat> restricted(const std::vector<Mat>& generators,
                            const SpectralFiber& band) {
  std::vector<Mat> out;
  const Mat dual = band.dualFrame();
  const Mat right = band.rightFrame();
  for (const Mat& a : generators) out.push_back(dual.transpose() * a * right);
  return out;
}

/// The traceless part of the commutant projection of an R x R matrix.
Mat tracelessCommutantPart(const Mat& x, const std::vector<Mat>& basis) {
  const Eigen::Index r = x.rows();
  Mat p = Mat::Zero(r, r);
  for (const Mat& b : basis) p += (b.adjoint() * x).trace() * b;
  p -= (p.trace() / static_cast<double>(r)) * Mat::Identity(r, r);
  return p;
}

}  // namespace

std::vector<std::string> IsospinDoublet::conditionNames() {
  return {"emergence", "coherent-transport", "current-agreement"};
}

std::string IsospinDoublet::statement(int number) {
  requireNumber(number, "IsospinDoublet::statement");
  switch (number) {
    case 1:
      return "An unlabeled two-dimensional spectral band emerges: two stable "
             "subclasses of the same cluster fiber, neither the sheet (colour) "
             "multiplicity nor a spin irreducible of the support's symmetry "
             "group, selected by a Riesz contour with a certified gap and "
             "persistent across frames and resolutions (WP v16 Sections 5, "
             "10).";
    case 2:
      return "The band is transported coherently: full-rank GL(2, C) transport "
             "with bounded leakage over the cluster's lifetime (WP v16 "
             "Sections 9, 10).";
    default:
      return "Its flavor-derived current agrees with the microscopic "
             "Ward-current flux at Q_u = +2/3 and Q_d = -1/3 (WP v16 Sections "
             "10, 13.4).";
  }
}

IsospinFrameRead IsospinDoublet::bands(
    const Mat& h, const std::vector<std::vector<std::uint64_t>>& cellsIn,
    const std::vector<std::size_t>& sheetOfCell,
    const std::vector<std::size_t>& baseCellOfCell,
    const std::vector<Mat>& symmetry, bool spinorial, int degree,
    const IsospinDoubletConfig& cfg) {
  const Eigen::Index n = h.rows();
  if (h.cols() != n || n == 0)
    throw std::invalid_argument(
        "IsospinDoublet::bands: the operator must be square and nonempty");
  if (!cellsIn.empty() && static_cast<Eigen::Index>(cellsIn.size()) != n)
    throw std::invalid_argument(
        "IsospinDoublet::bands: one cell tuple per operator row is required");
  if (!sheetOfCell.empty() && static_cast<Eigen::Index>(sheetOfCell.size()) != n)
    throw std::invalid_argument(
        "IsospinDoublet::bands: one sheet per operator row is required");
  if (!baseCellOfCell.empty() &&
      static_cast<Eigen::Index>(baseCellOfCell.size()) != n)
    throw std::invalid_argument(
        "IsospinDoublet::bands: one base cell per operator row is required");
  for (const Mat& d : symmetry)
    if (d.rows() != n || d.cols() != n)
      throw std::invalid_argument(
          "IsospinDoublet::bands: every symmetry matrix must match the "
          "operator's size");

  std::vector<std::vector<std::uint64_t>> cells = cellsIn;
  if (cells.empty())
    for (Eigen::Index i = 0; i < n; ++i)
      cells.push_back({static_cast<std::uint64_t>(i)});

  std::size_t sheetCount = 1;
  const std::vector<Mat> units =
      sheetUnits(n, sheetOfCell, baseCellOfCell, sheetCount);

  IsospinFrameRead read;
  Eigen::ComplexEigenSolver<Mat> ces(h, false);
  std::vector<cd> values(ces.eigenvalues().data(),
                         ces.eigenvalues().data() + n);
  double scale = 0.0;
  for (const cd& v : values) scale = std::max(scale, std::abs(v));
  if (scale == 0.0) scale = 1.0;
  read.spectrum = values;
  std::sort(read.spectrum.begin(), read.spectrum.end(), lessComplex);

  const bool hermitian = (h - h.adjoint()).norm() <= 1e-12 * std::max(h.norm(), 1.0);
  const double hn = std::max(h.norm(), 1e-300);
  const auto groups = clusters(values, cfg.groupingTolerance * scale);
  for (std::size_t gi = 0; gi < groups.size(); ++gi) {
    const auto& g = groups[gi];
    IsospinBandRead band;
    band.index = gi;
    std::vector<bool> inBand(static_cast<std::size_t>(n), false);
    for (const Eigen::Index i : g) {
      band.eigenvalues.push_back(values[static_cast<std::size_t>(i)]);
      inBand[static_cast<std::size_t>(i)] = true;
    }
    std::sort(band.eigenvalues.begin(), band.eigenvalues.end(), lessComplex);
    cd centre(0.0, 0.0);
    for (const cd& v : band.eigenvalues) centre += v;
    centre /= static_cast<double>(band.eigenvalues.size());
    band.center = centre;
    double inner = 0.0;
    for (const cd& v : band.eigenvalues) inner = std::max(inner, std::abs(v - centre));
    double outer = kInf;
    double gap = kInf;
    for (Eigen::Index j = 0; j < n; ++j) {
      if (inBand[static_cast<std::size_t>(j)]) continue;
      outer = std::min(outer, std::abs(values[static_cast<std::size_t>(j)] - centre));
      for (const cd& v : band.eigenvalues)
        gap = std::min(gap, std::abs(values[static_cast<std::size_t>(j)] - v));
    }
    band.gap = gap;
    const bool separable = outer > inner;
    const double radius =
        std::isinf(outer) ? inner + 0.5 * scale : 0.5 * (inner + outer);
    band.contourCenter = centre;
    band.contourRadius = separable ? radius : kNaN;

    Mat p = Mat::Zero(n, n);
    double resolventMax = kNaN;
    std::string contourText;
    if (separable && radius > 0.0) {
      const chainhodge::Contour contour =
          chainhodge::Contour::circle(centre, radius, cfg.contourNodes);
      contourText = contour.description;
      resolventMax = 0.0;
      for (std::size_t j = 0; j < contour.nodes.size(); ++j) {
        const Mat shifted = contour.nodes[j] * Mat::Identity(n, n) - h;
        const Mat resolvent = shifted.partialPivLu().inverse();
        resolventMax = std::max(resolventMax, spectralNorm(resolvent));
        p += contour.weights[j] * resolvent;
      }
    } else {
      // Not separable by a circle: the eigenvector projector, reported as an
      // uncertified band.
      Eigen::ComplexEigenSolver<Mat> full(h, true);
      const Mat v = full.eigenvectors();
      const Mat vinv = v.inverse();
      for (const Eigen::Index i : g) p += v.col(i) * vinv.row(i);
    }
    band.resolventMax = resolventMax;
    band.projectorResidual = (p * p - p).norm() / std::max(1.0, p.norm());
    band.projectorNorm = spectralNorm(p);
    band.rank = static_cast<std::size_t>(std::max<long long>(
        0, std::llround(p.trace().real())));
    band.isolated = separable && gap >= cfg.minRelativeGap * scale &&
                    band.projectorResidual <= cfg.projectorTolerance &&
                    band.projectorNorm <= cfg.conditionNumberCap;

    // Right frame: an orthonormal basis of Ran P; transpose dual
    // Phi~^T = Phi^dagger P, so that P = Phi Phi~^T and Phi~^T Phi = I.
    Eigen::JacobiSVD<Mat> svd(p, Eigen::ComputeThinU);
    const Mat right = svd.matrixU().leftCols(static_cast<Eigen::Index>(band.rank));
    const Mat dual = (right.adjoint() * p).transpose();
    const Mat reduced = dual.transpose() * h * right;

    SpectralBandCertificate cert;
    cert.degree = degree;
    cert.rank = band.rank;
    cert.lowerGap = gap;
    cert.upperGap = gap;
    cert.nearestDiscardedSeparation = gap;
    cert.projectorResidual = band.projectorResidual;
    cert.eigenResidual = (h * right - right * reduced).norm() / hn;
    cert.leftResidual = (dual.transpose() * h - reduced * dual.transpose()).norm() / hn;
    cert.gramDefect =
        (dual.transpose() * right - Mat::Identity(right.cols(), right.cols())).norm();
    cert.projectorNorm = band.projectorNorm;
    cert.frameConditionNumber = 1.0;
    cert.contour = contourText;
    cert.contourNodeCount = separable ? cfg.contourNodes : 0;
    cert.contourCenter = centre;
    cert.contourRadius = band.contourRadius;
    cert.resolventMax = resolventMax;
    cert.resolventBound = separable ? radius * resolventMax : kNaN;
    cert.bilinearLeftFrame = true;
    cert.selfAdjoint = hermitian;
    double lo = kInf, hi = -kInf;
    for (const cd& v : band.eigenvalues) {
      lo = std::min(lo, v.real());
      hi = std::max(hi, v.real());
    }
    cert.frequencyLower = lo;
    cert.frequencyUpper = hi;
    cert.accepted = band.isolated;
    const CertificateRegime regime =
        hermitian ? CertificateRegime::HermitianIndefinite : CertificateRegime::NonNormal;
    cert.certificate =
        band.isolated
            ? Certificate::certifiedNumerical(
                  CertificateDomain::BandWindow, regime,
                  std::max(band.projectorResidual, cert.eigenResidual),
                  band.projectorNorm, cfg.projectorTolerance)
            : Certificate::heuristicDiscovery(CertificateDomain::BandWindow, regime);
    band.fiber = SpectralFiber(cells, band.eigenvalues, right, dual,
                               Eigen::VectorXcd::Ones(n), cert);

    // ── representation content ───────────────────────────────────────────
    band.sheetCount = sheetCount;
    band.sheetInvarianceResidual = units.empty() ? 0.0 : invarianceResidual(p, units);
    band.colourActs = !units.empty() &&
                      band.sheetInvarianceResidual <= cfg.invarianceTolerance;
    band.symmetryDeclared = !symmetry.empty();
    band.symmetryInvarianceResidual =
        symmetry.empty() ? kNaN : invarianceResidual(p, symmetry);
    band.symmetryActs = band.symmetryDeclared &&
                        band.symmetryInvarianceResidual <= cfg.invarianceTolerance;

    std::vector<Mat> generators;
    if (band.colourActs)
      for (const Mat& x : restricted(units, band.fiber)) generators.push_back(x);
    if (band.symmetryActs)
      for (const Mat& x : restricted(symmetry, band.fiber)) generators.push_back(x);
    const auto r = static_cast<Eigen::Index>(band.rank);
    const std::size_t colourFactor = band.colourActs ? sheetCount : 1;
    std::ostringstream content;
    if (r == 0) {
      band.classification = "an empty band";
    } else if (r > kMaxDecomposedRank) {
      band.classification =
          "a band of rank " + std::to_string(r) +
          ", above the rank whose commutant is decomposed densely; its content "
          "is not read";
      band.unexplainedMultiplicity = false;
    } else {
      const std::vector<Mat> basis = commutant(generators, r, cfg.commutantTolerance);
      const Content c = decompose(basis, r, colourFactor, cfg.commutantTolerance);
      band.commutantDimension = c.commutantDimension;
      band.isotypeCount = c.isotypeCount;
      band.irreducibleDimensions = c.irreducibleDimensions;
      band.multiplicities = c.multiplicities;
      for (std::size_t i = 0; i < c.multiplicities.size(); ++i) {
        if (i > 0) content << " + ";
        content << c.irreducibleDimensions[i] << " x " << colourFactor
                << (colourFactor == 1 ? " sheet" : " sheets") << " x "
                << c.multiplicities[i];
      }
      if (!c.decomposed) content << "commutant of dimension " << c.commutantDimension;
      band.content = content.str();
      const bool irreducible = c.commutantDimension == 1;
      band.unexplainedMultiplicity = !irreducible;
      band.doubletCandidate = c.decomposed && c.isotypeCount == 1 &&
                              c.multiplicities.size() == 1 &&
                              c.multiplicities[0] == 2;
      band.spinDoublet = irreducible && band.symmetryActs && spinorial &&
                         !c.irreducibleDimensions.empty() &&
                         c.irreducibleDimensions[0] == 2;
      std::ostringstream text;
      const std::string colour =
          band.colourActs ? " times the " + std::to_string(sheetCount) +
                                " sheets (colour)"
                          : std::string{};
      if (band.spinDoublet) {
        text << "a spinor doublet of the declared group" << colour
             << ": spin content, not flavour";
      } else if (irreducible) {
        text << "one irreducible of dimension "
             << (c.irreducibleDimensions.empty() ? 0 : c.irreducibleDimensions[0])
             << (band.symmetryActs ? " of the declared group" : "") << colour;
      } else if (band.doubletCandidate) {
        text << "two copies of one irreducible of dimension "
             << c.irreducibleDimensions[0] << colour
             << ": a two-dimensional multiplicity space on which neither the "
                "declared symmetry nor the sheets act, a flavour-doublet "
                "candidate";
      } else {
        text << "reducible (" << band.content
             << "): a multiplicity that no declared symmetry explains";
      }
      if (!band.symmetryDeclared)
        text << "; no symmetry was declared, so spin content is not read";
      else if (!band.symmetryActs)
        text << "; the declared symmetry does not act on this band (relative "
                "invariance residual "
             << fmt(band.symmetryInvarianceResidual) << ")";
      if (!units.empty() && !band.colourActs)
        text << "; the sheets do not act on this band (relative invariance "
                "residual "
             << fmt(band.sheetInvarianceResidual) << ")";
      band.classification = text.str();
    }
    read.bands.push_back(std::move(band));
  }
  return read;
}

IsospinDoubletRead IsospinDoublet::observe(const IsospinDoubletDeclaration& decl,
                                           const IsospinDoubletConfig& cfg) {
  if (decl.frames.empty())
    throw std::invalid_argument("IsospinDoublet::observe: at least one frame is required");
  IsospinDoubletRead out;
  out.operatorName = decl.operatorName;
  out.symmetryName = decl.symmetryName;

  for (std::size_t f = 0; f < decl.frames.size(); ++f) {
    IsospinFrameRead fr = bands(decl.frames[f].operatorMatrix, decl.cells, decl.sheetOfCell,
                                decl.baseCellOfCell, decl.symmetry, decl.spinorial,
                                decl.degree, cfg);
    fr.label = decl.frames[f].label.empty() ? "frame " + std::to_string(f)
                                            : decl.frames[f].label;
    out.frames.push_back(std::move(fr));
  }
  for (std::size_t q = 0; q < decl.resolutions.size(); ++q) {
    const IsospinResolution& res = decl.resolutions[q];
    if (res.prolongation.rows() != res.operatorMatrix.rows() ||
        res.prolongation.cols() != decl.frames[0].operatorMatrix.rows())
      throw std::invalid_argument(
          "IsospinDoublet::observe: a prolongation must map the first frame's "
          "cells to the resolution's cells");
    IsospinFrameRead fr = bands(res.operatorMatrix, {}, res.sheetOfCell, res.baseCellOfCell,
                                res.symmetry, decl.spinorial, decl.degree, cfg);
    fr.label = res.label.empty() ? "resolution " + std::to_string(q) : res.label;
    out.resolutions.push_back(std::move(fr));
  }
  for (std::size_t f = 0; f < out.frames.size(); ++f)
    for (const IsospinBandRead& b : out.frames[f].bands)
      if (b.unexplainedMultiplicity)
        out.unexplainedMultiplicities.push_back(
            out.frames[f].label + " band " + std::to_string(b.index) + " at " +
            fmt(b.center) + ": " + b.content);
  out.multiplicityRefinementMeasured = !decl.resolutions.empty();

  const IsospinFrameRead& first = out.frames[0];
  const std::size_t frameCount = out.frames.size();
  const bool lifetimeMeasured = frameCount >= cfg.minFrames && frameCount >= 2;

  // Symmetry and sheet generators on the cells, for the intertwining read.
  std::size_t sheetCount = 1;
  const Eigen::Index n = decl.frames[0].operatorMatrix.rows();
  const std::vector<Mat> units = sheetUnits(n, decl.sheetOfCell, decl.baseCellOfCell, sheetCount);

  for (const IsospinBandRead& band : first.bands) {
    if (!band.doubletCandidate) continue;
    IsospinCandidateRead cand;
    cand.bandIndex = band.index;
    cand.trackedBands.push_back(band.index);
    cand.minTrackOverlap = 1.0;

    // ── persistence across frames ────────────────────────────────────────
    bool chainComplete = true;
    std::string chainBreak;
    for (std::size_t t = 0; t + 1 < frameCount && chainComplete; ++t) {
      std::vector<SpectralFiber> from, to;
      for (const IsospinBandRead& b : out.frames[t].bands) from.push_back(b.fiber);
      for (const IsospinBandRead& b : out.frames[t + 1].bands) to.push_back(b.fiber);
      const bool sameCells = decl.frames[t + 1].transferFromPrevious.size() == 0;
      std::size_t next = 0;
      double overlap = 0.0;
      bool certified = false;
      if (sameCells) {
        const std::vector<FiberMatchRead> matches =
            SpectralFiberTracker::matchFibers(from, to, cfg.trackOverlapThreshold);
        for (const FiberMatchRead& m : matches)
          if (m.fromIndex == cand.trackedBands.back()) {
            next = m.toIndex;
            overlap = m.overlap.subspaceOverlap;
            certified = m.certifiedContinuation;
          }
      } else {
        const Mat& transfer = decl.frames[t + 1].transferFromPrevious;
        const Mat carried =
            transfer * out.frames[t].bands[cand.trackedBands.back()].fiber.rightFrame();
        for (const IsospinBandRead& b : out.frames[t + 1].bands) {
          const double o = subspaceOverlap(carried, b.fiber.rightFrame());
          if (o > overlap) {
            overlap = o;
            next = b.index;
          }
        }
        certified = overlap >= cfg.trackOverlapThreshold &&
                    out.frames[t + 1].bands[next].isolated &&
                    out.frames[t + 1].bands[next].rank == band.rank;
      }
      if (!certified || !out.frames[t + 1].bands[next].doubletCandidate) {
        chainComplete = false;
        chainBreak = "the track ends between " + out.frames[t].label + " and " +
                     out.frames[t + 1].label + " (best overlap " + fmt(overlap) +
                     (certified ? ", but the continuation is not a doublet candidate)"
                                : ", no certified continuation)");
        break;
      }
      cand.trackedBands.push_back(next);
      cand.minTrackOverlap = std::min(cand.minTrackOverlap, overlap);
    }

    // ── persistence across resolutions ───────────────────────────────────
    for (std::size_t q = 0; q < out.resolutions.size(); ++q) {
      const Mat carried = decl.resolutions[q].prolongation * band.fiber.rightFrame();
      double best = 0.0;
      for (const IsospinBandRead& b : out.resolutions[q].bands)
        if (b.doubletCandidate && b.isolated)
          best = std::max(best, subspaceOverlap(carried, b.fiber.rightFrame()));
      cand.resolutionOverlap.push_back(best);
      cand.resolutionFound.push_back(best >= cfg.trackOverlapThreshold);
    }

    std::vector<QuarkConditionEvidence> ev1;
    ev1.push_back(evidence("two-dimensional-flavour-band", true,
                           "band " + std::to_string(band.index) + " at " +
                               fmt(band.center) + ": " + band.classification));
    ev1.push_back(evidence(
        "not-colour", band.colourActs || band.sheetCount == 1,
        band.colourActs
            ? "the " + std::to_string(band.sheetCount) +
                  " sheets act on the band and are divided out; the "
                  "multiplicity space commutes with them"
            : (band.sheetCount == 1
                   ? std::string("one sheet: no colour multiplicity to divide out")
                   : "the sheets do not act on the band (relative residual " +
                         fmt(band.sheetInvarianceResidual) +
                         "), so colour cannot be divided out")));
    if (!band.symmetryDeclared)
      ev1.push_back(evidence("not-spin-irreducible", std::nullopt,
                             "no symmetry was declared, so the band cannot be "
                             "told apart from a spin irreducible"));
    else if (!band.symmetryActs)
      ev1.push_back(evidence("not-spin-irreducible", std::nullopt,
                             "the declared symmetry does not act on the band "
                             "(relative residual " +
                                 fmt(band.symmetryInvarianceResidual) +
                                 "), so its spin content is not read"));
    else
      ev1.push_back(evidence("not-spin-irreducible", true,
                             "the band is two copies of one irreducible of " +
                                 (decl.symmetryName.empty() ? std::string("the declared group")
                                                            : decl.symmetryName) +
                                 ", not one irreducible"));
    ev1.push_back(evidence("riesz-contour-and-gap", band.isolated,
                           "gap " + fmt(band.gap) + ", idempotency " +
                               fmt(band.projectorResidual) + ", resolvent max " +
                               fmt(band.resolventMax) + ", projector norm " +
                               fmt(band.projectorNorm)));
    if (!lifetimeMeasured)
      ev1.push_back(evidence("frame-persistence", std::nullopt,
                             std::to_string(frameCount) +
                                 " frame(s) supplied; persistence needs at least " +
                                 std::to_string(std::max<std::size_t>(cfg.minFrames, 2))));
    else
      ev1.push_back(evidence("frame-persistence", chainComplete,
                             chainComplete ? "tracked through " + std::to_string(frameCount) +
                                                 " frames, least overlap " +
                                                 fmt(cand.minTrackOverlap)
                                           : chainBreak));
    if (out.resolutions.empty())
      ev1.push_back(evidence("refinement-persistence", std::nullopt,
                             "no further resolution was supplied"));
    else {
      bool all = true;
      std::string detail = "overlaps";
      for (std::size_t q = 0; q < cand.resolutionFound.size(); ++q) {
        all = all && cand.resolutionFound[q];
        detail += " " + out.resolutions[q].label + ": " + fmt(cand.resolutionOverlap[q]);
      }
      ev1.push_back(evidence("refinement-persistence", all, detail));
    }
    cand.conditions.push_back(combine(1, ev1));

    // ── coherent transport ───────────────────────────────────────────────
    std::vector<QuarkConditionEvidence> ev2;
    if (!lifetimeMeasured) {
      const std::string why = std::to_string(frameCount) +
                              " frame(s) supplied: a single frame has no transport";
      ev2.push_back(evidence("transport-full-rank", std::nullopt, why));
      ev2.push_back(evidence("transport-leakage", std::nullopt, why));
      ev2.push_back(evidence("flavour-structure-preserved", std::nullopt, why));
      ev2.push_back(evidence("transport-over-lifetime", std::nullopt, why));
    } else {
      ComplexTransportConfig tcfg;
      tcfg.leakageTolerance = cfg.transportLeakageTolerance;
      tcfg.conditionNumberCap = cfg.conditionNumberCap;
      bool fullRank = true;
      double worstLeak = 0.0;
      double worstIntertwining = 0.0;
      bool intertwiningMeasured = true;
      std::vector<Mat> maps;
      for (std::size_t t = 0; t + 1 < cand.trackedBands.size(); ++t) {
        const IsospinBandRead& a = out.frames[t].bands[cand.trackedBands[t]];
        const IsospinBandRead& b = out.frames[t + 1].bands[cand.trackedBands[t + 1]];
        const Mat& given = decl.frames[t + 1].transferFromPrevious;
        const Mat transfer = given.size() == 0 ? Mat::Identity(n, n) : given;
        IsospinTransportStep step;
        step.fromFrame = t;
        step.toFrame = t + 1;
        step.transport = ComplexTransport::transport(b.fiber, a.fiber, transfer, tcfg);
        fullRank = fullRank && step.transport.invertible;
        worstLeak = std::max(worstLeak, step.transport.relativeLeakage);
        const Mat& m = step.transport.map;
        if (given.size() == 0) {
          std::vector<Mat> gens;
          if (a.colourActs && b.colourActs)
            for (const Mat& x : units) gens.push_back(x);
          if (a.symmetryActs && b.symmetryActs)
            for (const Mat& x : decl.symmetry) gens.push_back(x);
          const std::vector<Mat> ga = restricted(gens, a.fiber);
          const std::vector<Mat> gb = restricted(gens, b.fiber);
          double worst = 0.0;
          for (std::size_t i = 0; i < gens.size(); ++i)
            worst = std::max(worst, (m * ga[i] - gb[i] * m).norm() /
                                        (std::max(m.norm(), 1e-300) *
                                         std::max(ga[i].norm(), 1e-300)));
          step.intertwiningResidual = worst;
          worstIntertwining = std::max(worstIntertwining, worst);
        } else {
          step.intertwiningResidual = kNaN;
          intertwiningMeasured = false;
        }
        // The flavour factor's singular values: those of the band transport,
        // which come in two groups of multiplicity R/2 when M = I (x) m.
        const std::vector<double>& s = step.transport.singularValues;
        if (!s.empty()) {
          std::vector<cd> sv(s.begin(), s.end());
          const auto groups = clusters(sv, 1e-6 * std::max(s.front(), 1e-300));
          const std::size_t half = s.size() / 2;
          if (groups.size() == 1)
            step.flavourSingularValues = {s.front(), s.front()};
          else if (groups.size() == 2 && groups[0].size() == half &&
                   groups[1].size() == half)
            step.flavourSingularValues = {s.front(), s.back()};
        }
        maps.push_back(m);
        cand.transports.push_back(std::move(step));
      }
      if (!maps.empty()) {
        const Mat composed = ComplexTransport::compose(maps);
        Eigen::JacobiSVD<Mat> svd(composed);
        const Eigen::VectorXd& s = svd.singularValues();
        cand.lifetimeSingularValues.assign(s.data(), s.data() + s.size());
        fullRank = fullRank && s.size() > 0 && s(s.size() - 1) > 1e-9 * s(0) &&
                   s(0) / s(s.size() - 1) <= cfg.conditionNumberCap;
      }
      if (!chainComplete) {
        ev2.push_back(evidence("transport-full-rank", maps.empty() ? std::optional<bool>{} : std::optional<bool>{fullRank},
                               "over the tracked part of the lifetime"));
        ev2.push_back(evidence("transport-leakage",
                               maps.empty() ? std::optional<bool>{}
                                            : std::optional<bool>{worstLeak <= cfg.transportLeakageTolerance},
                               "worst relative leakage " + fmt(worstLeak)));
        ev2.push_back(evidence("flavour-structure-preserved", std::nullopt,
                               "the track does not span the lifetime"));
        ev2.push_back(evidence("transport-over-lifetime", false, chainBreak));
      } else {
        std::ostringstream sv;
        for (double x : cand.lifetimeSingularValues) sv << " " << fmt(x);
        ev2.push_back(evidence("transport-full-rank", fullRank,
                               "composed lifetime transport singular values" + sv.str()));
        ev2.push_back(evidence("transport-leakage", worstLeak <= cfg.transportLeakageTolerance,
                               "worst relative leakage " + fmt(worstLeak) + " against " +
                                   fmt(cfg.transportLeakageTolerance)));
        ev2.push_back(evidence(
            "flavour-structure-preserved",
            intertwiningMeasured ? std::optional<bool>{worstIntertwining <= cfg.intertwiningTolerance}
                                 : std::optional<bool>{},
            intertwiningMeasured
                ? "worst relative intertwining residual " + fmt(worstIntertwining) +
                      ": the transport factors as I (x) m with m in GL(2, C)"
                : std::string("a transfer between different cell sets carries no "
                              "common symmetry action to intertwine")));
        ev2.push_back(evidence("transport-over-lifetime", true,
                               "transported through " + std::to_string(frameCount) + " frames"));
      }
    }
    cand.conditions.push_back(combine(2, ev2));
    cand.conditions.push_back(wardCondition());
    cand.observed = cand.conditions[0].status == QuarkConditionStatus::Passed &&
                    cand.conditions[1].status == QuarkConditionStatus::Passed;

    // ── isospin, charge and occupation ───────────────────────────────────
    if (cand.observed) {
      IsospinChargeRead charges;
      const SpectralFiber& fiber = band.fiber;
      const Mat right = fiber.rightFrame();
      const Mat dual = fiber.dualFrame();
      const auto r = static_cast<Eigen::Index>(band.rank);
      std::vector<Mat> gens;
      if (band.colourActs)
        for (const Mat& x : restricted(units, fiber)) gens.push_back(x);
      if (band.symmetryActs)
        for (const Mat& x : restricted(decl.symmetry, fiber)) gens.push_back(x);
      const std::vector<Mat> basis = commutant(gens, r, cfg.commutantTolerance);
      const Mat& h0 = decl.frames[0].operatorMatrix;
      const double floor = 1e-8;
      Mat split = tracelessCommutantPart(dual.transpose() * h0 * right, basis);
      charges.memberSource = "in-band operator";
      if (split.norm() <= floor * std::max(1.0, h0.norm())) {
        split = Mat();
        if (decl.memberSplitting.size() != 0) {
          if (decl.memberSplitting.rows() != n || decl.memberSplitting.cols() != n)
            throw std::invalid_argument(
                "IsospinDoublet::observe: the member-splitting operator must "
                "match the first frame's cells");
          const Mat s = tracelessCommutantPart(
              dual.transpose() * decl.memberSplitting * right, basis);
          if (s.norm() > floor * std::max(1.0, decl.memberSplitting.norm())) {
            split = s;
            charges.memberSource = "declared splitting operator";
          }
        }
        if (split.size() == 0) {
          Mat seed = Mat::Zero(n, n);
          for (Eigen::Index i = 0; i < n; ++i) seed(i, i) = static_cast<double>(i + 1);
          split = tracelessCommutantPart(dual.transpose() * seed * right, basis);
          charges.memberSource = "declared trivialization (cell-order seed)";
        }
      }
      Eigen::ComplexEigenSolver<Mat> ces(split);
      std::vector<std::pair<cd, Eigen::Index>> order;
      for (Eigen::Index i = 0; i < r; ++i) order.emplace_back(ces.eigenvalues()(i), i);
      std::sort(order.begin(), order.end(),
                [](const auto& a, const auto& b) { return lessComplex(b.first, a.first); });
      const Mat v = ces.eigenvectors();
      const Mat vinv = v.inverse();
      Mat plus = Mat::Zero(r, r), minus = Mat::Zero(r, r);
      for (Eigen::Index k = 0; k < r; ++k) {
        const Eigen::Index i = order[static_cast<std::size_t>(k)].second;
        (k < r / 2 ? plus : minus) += v.col(i) * vinv.row(i);
      }
      charges.isospin = {0.5, -0.5};
      charges.memberProjectors = {right * plus * dual.transpose(),
                                  right * minus * dual.transpose()};
      if (!decl.lineage.has_value()) {
        charges.notes.push_back("no lineage was supplied, so B and Q are unmeasured");
      } else if (!decl.lineage->failedCertificates.empty()) {
        std::string why;
        for (const std::string& s : decl.lineage->failedCertificates) why += " " + s;
        charges.notes.push_back("the lineage reading is uncertified:" + why);
      } else {
        const double b = static_cast<double>(decl.lineage->number) / 3.0;
        charges.baryonNumber = b;
        charges.charges = {0.5 + b / 2.0, -0.5 + b / 2.0};
        charges.notes.push_back(
            "B = N_Q / 3 per occupied mode of the fiber, N_Q = " +
            std::to_string(decl.lineage->number) + " (n_Q = " +
            std::to_string(decl.lineage->fermionNumber) + " on the lineage)");
      }
      if (decl.threeQuarkDensity.size() == 0) {
        charges.notes.push_back(
            "no three-quark one-body density was supplied, so the occupation "
            "pattern is unmeasured");
      } else {
        if (decl.threeQuarkDensity.rows() != n || decl.threeQuarkDensity.cols() != n)
          throw std::invalid_argument(
              "IsospinDoublet::observe: the one-body density must match the "
              "first frame's cells");
        const cd np = (decl.threeQuarkDensity * charges.memberProjectors[0]).trace();
        const cd nm = (decl.threeQuarkDensity * charges.memberProjectors[1]).trace();
        charges.memberOccupations = {np, nm};
        const auto integer = [](cd x, long long& k) {
          k = std::llround(x.real());
          return std::abs(x - cd(static_cast<double>(k), 0.0)) <= 1e-6;
        };
        long long kp = 0, km = 0;
        if (integer(np, kp) && integer(nm, km) && kp == 2 && km == 1)
          charges.occupationPattern = "uud";
        else if (integer(np, kp) && integer(nm, km) && kp == 1 && km == 2)
          charges.occupationPattern = "udd";
        else
          charges.occupationPattern = "neither uud nor udd: occupations " + fmt(np) +
                                      " (I_3 = +1/2) and " + fmt(nm) + " (I_3 = -1/2)";
      }
      charges.notes.push_back(
          "which member is I_3 = +1/2 is the declared convention (larger real "
          "part of the splitting element), not a physical label");
      cand.charges = std::move(charges);
    }
    out.candidates.push_back(std::move(cand));
  }

  // ── the read as a whole ─────────────────────────────────────────────────
  const IsospinCandidateRead* chosen = nullptr;
  for (const IsospinCandidateRead& c : out.candidates)
    if (c.observed) {
      chosen = &c;
      break;
    }
  if (chosen == nullptr && !out.candidates.empty()) chosen = &out.candidates.front();
  if (chosen != nullptr) {
    out.conditions = chosen->conditions;
  } else {
    std::ostringstream detail;
    detail << "no band of " << first.label << " is two copies of one irreducible; bands:";
    for (const IsospinBandRead& b : first.bands)
      detail << " [" << b.index << " at " << fmt(b.center) << ", rank " << b.rank << ": "
             << b.classification << "]";
    out.conditions.push_back(
        combine(1, {evidence("two-dimensional-flavour-band", false, detail.str())}));
    out.conditions.push_back(combine(
        2, {evidence("transport-full-rank", std::nullopt, "no candidate band to transport"),
            evidence("transport-leakage", std::nullopt, "no candidate band to transport"),
            evidence("flavour-structure-preserved", std::nullopt,
                     "no candidate band to transport"),
            evidence("transport-over-lifetime", std::nullopt,
                     "no candidate band to transport")}));
    out.conditions.push_back(wardCondition());
  }
  out.doubletObserved = chosen != nullptr && chosen->observed;
  out.noIsospinDoublet = !out.doubletObserved;

  std::ostringstream summary;
  summary << "On " << (decl.operatorName.empty() ? "the supplied operator" : decl.operatorName)
          << " (" << frameCount << " frame(s), " << out.resolutions.size()
          << " further resolution(s)): " << out.candidates.size()
          << " flavour-doublet candidate(s). ";
  if (out.doubletObserved)
    summary << "A doublet emerged and was transported coherently (conditions 1 "
               "and 2 passed); agreement with the Ward flux is not evaluable, so "
               "the identification is incomplete. ";
  else if (out.candidates.empty())
    summary << "No unlabeled two-dimensional band emerged (falsifier 8, \"No "
               "isospin doublet\", holds on this read). ";
  else
    summary << "No candidate passed both conditions 1 and 2 (falsifier 8 holds "
               "on this read). ";
  summary << out.unexplainedMultiplicities.size()
          << " band(s) carry a multiplicity that no declared symmetry explains"
          << (out.multiplicityRefinementMeasured ? "." : "; refinement robustness is unmeasured.");
  out.summary = summary.str();
  return out;
}

}  // namespace tessera::observables
