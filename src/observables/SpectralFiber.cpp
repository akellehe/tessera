// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Extraction and tracking of isolated spectral bands of the
/// component-restricted Hodge Laplacian, with their frames, projectors and
/// certificates.
/// Reference: Lim, "Hodge Laplacians on graphs", arXiv:1507.05379

#include "observables/SpectralFiber.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SVD>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/RieszBand.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/AnalyticCache.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/DenseReference.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

using cd = std::complex<double>;
using cobordism::Certificate;
using cobordism::CertificateDomain;
using cobordism::CertificateGrade;
using cobordism::CertificateRegime;
using cobordism::gradeName;
using cobordism::domainName;
using cobordism::regimeName;
using cobordism::domainFromName;
using cobordism::regimeFromName;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// Schema 2 carries the acceptance quantities `nearest_discarded_separation`,
// `localization_support_fraction`, `localization_excess`, `projector_norm`
// (named `condition_number` in schema 1) and `frame_condition_number`.
// Schema 1 remains readable: its projector norm is carried over verbatim and
// quantities it never measured read back as NaN, never zero.
constexpr int kSchemaVersion = 2;
constexpr int kOldestReadableSchema = 1;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

std::vector<std::uint64_t> normalizedSupport(
    const std::vector<std::uint64_t> &support) {
  std::vector<std::uint64_t> out(support);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

bool lessReIm(const cd &a, const cd &b) {
  if (a.real() != b.real()) return a.real() < b.real();
  return a.imag() < b.imag();
}






// A double leaf that older schemas may not carry: absent reads back as NaN
// (unknown), never zero.
double optionalDouble(const Record::Map &m, const char *key) {
  const auto it = m.find(key);
  return it == m.end() ? kNaN : it->second.asDouble();
}

Record certificateToRecord(const Certificate &cert) {
  Record::Map m;
  m["grade"] = Record(gradeName(cert.grade()));
  m["domain"] = Record(domainName(cert.domain()));
  m["regime"] = Record(regimeName(cert.regime()));
  m["residual"] = Record(cert.residual());
  m["conditioning"] = Record(cert.conditioning());
  m["dense_reference_error"] = Record(cert.denseReferenceError());
  m["tolerance"] = Record(cert.tolerance());
  return Record(std::move(m));
}

Certificate certificateFromRecord(const Record &record) {
  const auto &m = record.asMap();
  const std::string grade = m.at("grade").asString();
  const CertificateDomain domain = domainFromName(m.at("domain").asString());
  const CertificateRegime regime = regimeFromName(m.at("regime").asString());
  const double residual = m.at("residual").asDouble();
  const double conditioning = m.at("conditioning").asDouble();
  const double tolerance = m.at("tolerance").asDouble();
  Certificate cert;
  if (grade == "algebraically-exact") {
    cert = Certificate::algebraicallyExact(domain, regime, residual, tolerance);
  } else if (grade == "structure-exact") {
    cert = Certificate::structureExact(domain, regime, residual, conditioning,
                                       tolerance);
  } else if (grade == "certified-numerical") {
    cert = Certificate::certifiedNumerical(domain, regime, residual,
                                           conditioning, tolerance);
  } else if (grade == "heuristic-discovery") {
    cert = Certificate::heuristicDiscovery(domain, regime);
  } else {
    throw std::invalid_argument("SpectralFiber: unknown certificate grade '" +
                                grade + "'");
  }
  cert.setDenseReferenceError(m.at("dense_reference_error").asDouble());
  return cert;
}

Record bandCertificateToRecord(const SpectralBandCertificate &c) {
  Record::Map m;
  m["degree"] = Record(c.degree);
  m["rank"] = Record(static_cast<std::int64_t>(c.rank));
  m["lower_gap"] = Record(c.lowerGap);
  m["upper_gap"] = Record(c.upperGap);
  m["nearest_discarded_separation"] = Record(c.nearestDiscardedSeparation);
  m["localization"] = Record(c.localization);
  m["localization_support_fraction"] = Record(c.localizationSupportFraction);
  m["localization_excess"] = Record(c.localizationExcess);
  m["projector_residual"] = Record(c.projectorResidual);
  m["eigen_residual"] = Record(c.eigenResidual);
  m["left_residual"] = Record(c.leftResidual);
  m["gram_defect"] = Record(c.gramDefect);
  m["projector_norm"] = Record(c.projectorNorm);
  m["frame_condition_number"] = Record(c.frameConditionNumber);
  m["positive_signature"] = Record(c.positiveSignature);
  m["negative_signature"] = Record(c.negativeSignature);
  m["frequency_lower"] = Record(c.frequencyLower);
  m["frequency_upper"] = Record(c.frequencyUpper);
  m["self_adjoint"] = Record(c.selfAdjoint);
  m["pairing_determinant_re"] = Record(c.pairingDeterminant.real());
  m["pairing_determinant_im"] = Record(c.pairingDeterminant.imag());
  m["pairing_condition"] = Record(c.pairingCondition);
  m["pairing_scale"] = Record(c.pairingScale);
  m["isotropic"] = Record(c.isotropic);
  m["left_frame_refusal"] = Record(c.leftFrameRefusal);
  m["metric_symmetry_defect"] = Record(c.metricSymmetryDefect);
  m["bilinear_left_frame"] = Record(c.bilinearLeftFrame);
  m["accepted"] = Record(c.accepted);
  m["certificate"] = certificateToRecord(c.certificate);
  return Record(std::move(m));
}

SpectralBandCertificate bandCertificateFromRecord(const Record &record) {
  const auto &m = record.asMap();
  SpectralBandCertificate c;
  c.degree = static_cast<int>(m.at("degree").asInt());
  c.rank = static_cast<std::size_t>(m.at("rank").asInt());
  c.lowerGap = m.at("lower_gap").asDouble();
  c.upperGap = m.at("upper_gap").asDouble();
  c.nearestDiscardedSeparation =
      optionalDouble(m, "nearest_discarded_separation");
  c.localization = m.at("localization").asDouble();
  c.localizationSupportFraction =
      optionalDouble(m, "localization_support_fraction");
  c.localizationExcess = optionalDouble(m, "localization_excess");
  c.projectorResidual = m.at("projector_residual").asDouble();
  c.eigenResidual = m.at("eigen_residual").asDouble();
  c.leftResidual = m.at("left_residual").asDouble();
  c.gramDefect = m.at("gram_defect").asDouble();
  // Schema 1 named the projector norm `condition_number`.
  c.projectorNorm = m.count("projector_norm")
                        ? m.at("projector_norm").asDouble()
                        : optionalDouble(m, "condition_number");
  c.frameConditionNumber = optionalDouble(m, "frame_condition_number");
  c.positiveSignature = static_cast<int>(m.at("positive_signature").asInt());
  c.negativeSignature = static_cast<int>(m.at("negative_signature").asInt());
  c.frequencyLower = m.at("frequency_lower").asDouble();
  c.frequencyUpper = m.at("frequency_upper").asDouble();
  c.selfAdjoint = m.at("self_adjoint").asBool();
  c.pairingDeterminant = cd(optionalDouble(m, "pairing_determinant_re"),
                            optionalDouble(m, "pairing_determinant_im"));
  c.pairingCondition = optionalDouble(m, "pairing_condition");
  c.pairingScale = optionalDouble(m, "pairing_scale");
  c.isotropic = m.count("isotropic") ? m.at("isotropic").asBool() : false;
  c.leftFrameRefusal = m.count("left_frame_refusal") ? m.at("left_frame_refusal").asString() : std::string{};
  c.metricSymmetryDefect = optionalDouble(m, "metric_symmetry_defect");
  c.accepted = m.at("accepted").asBool();
  c.certificate = certificateFromRecord(m.at("certificate"));
  // Records written before the flag existed stored the bilinear left frame
  // exactly on the complex-symmetric pencil regime.
  c.bilinearLeftFrame =
      m.count("bilinear_left_frame")
          ? m.at("bilinear_left_frame").asBool()
          : c.certificate.regime() == CertificateRegime::ComplexSymmetricPencil;
  return c;
}

Record cellsToRecord(const std::vector<std::vector<std::uint64_t>> &cells) {
  Record::List out;
  out.reserve(cells.size());
  for (const auto &cell : cells) {
    Record::List tuple;
    tuple.reserve(cell.size());
    for (const std::uint64_t v : cell)
      tuple.emplace_back(static_cast<std::int64_t>(v));
    out.emplace_back(std::move(tuple));
  }
  return Record(std::move(out));
}

std::vector<std::vector<std::uint64_t>> cellsFromRecord(const Record &record) {
  std::vector<std::vector<std::uint64_t>> cells;
  for (const Record &tuple : record.asList()) {
    std::vector<std::uint64_t> cell;
    cell.reserve(tuple.asList().size());
    for (const Record &v : tuple.asList())
      cell.push_back(static_cast<std::uint64_t>(v.asInt()));
    cells.push_back(std::move(cell));
  }
  return cells;
}

std::vector<cd> matrixToFlat(const Eigen::MatrixXcd &m) {
  std::vector<cd> flat(static_cast<std::size_t>(m.rows()) *
                       static_cast<std::size_t>(m.cols()));
  for (Eigen::Index r = 0; r < m.rows(); ++r)
    for (Eigen::Index c = 0; c < m.cols(); ++c)
      flat[static_cast<std::size_t>(r) * static_cast<std::size_t>(m.cols()) +
           static_cast<std::size_t>(c)] = m(r, c);
  return flat;
}

Eigen::MatrixXcd matrixFromFlat(const std::vector<cd> &flat, std::size_t rows,
                                std::size_t cols) {
  if (flat.size() != rows * cols)
    throw std::invalid_argument("SpectralFiber: matrix payload size mismatch");
  Eigen::MatrixXcd m(static_cast<Eigen::Index>(rows),
                     static_cast<Eigen::Index>(cols));
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c)
      m(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
          flat[r * cols + c];
  return m;
}

std::vector<cd> complexListFromRecord(const Record::Map &m,
                                      const std::string &name) {
  const auto &re = m.at(name + "_re").asList();
  const auto &im = m.at(name + "_im").asList();
  if (re.size() != im.size())
    throw std::invalid_argument("SpectralFiber: complex list length mismatch");
  std::vector<cd> out(re.size());
  for (std::size_t i = 0; i < re.size(); ++i)
    out[i] = cd(re[i].asDouble(), im[i].asDouble());
  return out;
}

void requireSchema(const Record::Map &m, const char *type) {
  const auto version = m.find("schema_version");
  if (version == m.end() ||
      version->second.asInt() <
          static_cast<std::int64_t>(kOldestReadableSchema) ||
      version->second.asInt() > static_cast<std::int64_t>(kSchemaVersion))
    throw std::invalid_argument(
        "SpectralFiber: unknown schema_version (reader rejects unknown "
        "checkpoint schemas)");
  const auto rt = m.find("record_type");
  if (rt == m.end() || rt->second.asString() != type)
    throw std::invalid_argument(std::string("SpectralFiber: expected a '") +
                                type + "' record");
}

// ||A M B^dagger||_F computed through the r x r trace identity
// tr(M^dagger (A^dagger A) M (B^dagger B)) — never materializes the n x n
// product.
double productFrobenius(const Eigen::MatrixXcd &A, const Eigen::MatrixXcd &M,
                        const Eigen::MatrixXcd &B) {
  const Eigen::MatrixXcd AtA = A.adjoint() * A;
  const Eigen::MatrixXcd BtB = B.adjoint() * B;
  const cd trace = (M.adjoint() * AtA * M * BtB).trace();
  return std::sqrt(std::max(0.0, trace.real()));
}

// ||A B^dagger||_2 = sqrt(lambda_max((A^dagger A)(B^dagger B))) — the band
// projector spectral norm from r x r blocks.
double productSpectralNorm(const Eigen::MatrixXcd &A,
                           const Eigen::MatrixXcd &B) {
  const Eigen::MatrixXcd prod = (A.adjoint() * A) * (B.adjoint() * B);
  if (prod.rows() == 0) return 0.0;
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(prod, false);
  double best = 0.0;
  for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i)
    best = std::max(best, es.eigenvalues()[i].real());
  return std::sqrt(std::max(0.0, best));
}

// Riesz condition number of one frame in the |W| metric:
// sqrt(lambda_max / lambda_min) of X^dagger |W| X. Exactly 1 for a
// |W|-orthonormal frame (the self-adjoint path), +infinity for a
// |W|-degenerate one, NaN when there is no frame to condition. A property of
// the frame, not of its range: an in-band basis change moves it.
double frameCondition(const Eigen::MatrixXcd &X, const Eigen::VectorXcd &W) {
  if (X.rows() == 0 || X.cols() == 0) return kNaN;
  Eigen::VectorXcd absW(X.rows());
  for (Eigen::Index i = 0; i < X.rows(); ++i) absW[i] = cd(std::abs(W[i]), 0.0);
  Eigen::MatrixXcd G = X.adjoint() * (absW.asDiagonal() * X);
  G = (0.5 * (G + G.adjoint())).eval();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(G);
  if (es.info() != Eigen::Success) return kNaN;
  const double lo = es.eigenvalues().minCoeff();
  const double hi = es.eigenvalues().maxCoeff();
  if (!(hi > 0.0)) return kNaN;
  if (!(lo > 0.0)) return kInf;
  return std::sqrt(hi / lo);
}

// Thin orthonormal basis of the column span (columns with relative singular
// value above tol are kept) — the gauge-invariant subspace representative.
Eigen::MatrixXcd thinOrthonormal(const Eigen::MatrixXcd &A,
                                 double tol = 1e-12) {
  if (A.rows() == 0 || A.cols() == 0) return Eigen::MatrixXcd(A.rows(), 0);
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A, Eigen::ComputeThinU);
  const auto &sv = svd.singularValues();
  const double cutoff = (sv.size() > 0 ? sv[0] : 0.0) * tol;
  Eigen::Index keep = 0;
  for (Eigen::Index i = 0; i < sv.size(); ++i)
    if (sv[i] > cutoff) ++keep;
  return svd.matrixU().leftCols(keep);
}

}  // namespace

// ---------------------------------------------------------------------------
// SpectralBandCertificate
// ---------------------------------------------------------------------------

std::string SpectralBandCertificate::describe() const {
  std::ostringstream out;
  out << (accepted ? "certified" : "uncertified") << " band: degree " << degree
      << ", rank " << rank;
  if (rank >= 2)
    out << " (degenerate; multiplicity reported without interpretation)";
  out << ", window [" << frequencyLower << ", " << frequencyUpper << "]"
      << ", sort-order gaps (" << lowerGap << ", " << upperGap << ")"
      << ", nearest-discarded separation " << nearestDiscardedSeparation
      << ", signature (+" << positiveSignature << ", -" << negativeSignature
      << ")"
      << ", localization " << localization << " (support fraction "
      << localizationSupportFraction << ", excess " << localizationExcess
      << "), residuals (eig " << eigenResidual
      << ", left " << leftResidual << ", proj " << projectorResidual
      << ", gram " << gramDefect << "), projector norm " << projectorNorm
      << ", frame condition " << frameConditionNumber << ", "
      << (selfAdjoint ? "self-adjoint" : "general") << " path";
  if (certificate.regime() == CertificateRegime::ComplexSymmetricPencil)
    out << "; complex-symmetric pencil: det B_C " << pairingDeterminant
        << ", cond B_C " << pairingCondition << ", pairing scale "
        << pairingScale << (isotropic ? ", ISOTROPIC (" + leftFrameRefusal + ")" : "")
        << ", metric symmetry defect " << metricSymmetryDefect;
  return out.str();
}

// ---------------------------------------------------------------------------
// SpectralFiber
// ---------------------------------------------------------------------------

SpectralFiber::SpectralFiber(std::vector<std::vector<std::uint64_t>> cells,
                             std::vector<cd> eigenvalues,
                             Eigen::MatrixXcd rightFrame,
                             Eigen::MatrixXcd leftFrame,
                             Eigen::VectorXcd weights,
                             SpectralBandCertificate certificate)
    : cells_(std::move(cells)), eigenvalues_(std::move(eigenvalues)),
      right_(std::move(rightFrame)), left_(std::move(leftFrame)),
      weights_(std::move(weights)), certificate_(std::move(certificate)) {}

Eigen::MatrixXcd SpectralFiber::dualFrame() const {
  // The chain-level pencil path stores the canonical bilinear left frame
  // Phi~ itself (the weight diagonal is the identity placeholder there), in
  // either verified regime.
  if (certificate_.bilinearLeftFrame) return left_;
  // Elsewhere the solver's normalization is Psi^dagger W Phi = I, and
  // (W conj(Psi))^T Phi = Psi^dagger W Phi: the transpose dual is W conj(Psi).
  if (left_.size() == 0) return left_;
  return weights_.asDiagonal() * left_.conjugate();
}

Eigen::MatrixXcd SpectralFiber::projector() const {
  if (right_.rows() == 0 || right_.cols() == 0)
    return Eigen::MatrixXcd::Zero(right_.rows(), right_.rows());
  // One pairing in every regime: the Riesz projector is Phi Phi~^T.
  return right_ * dualFrame().transpose();
}

std::complex<double> SpectralFiber::bandCenter() const {
  if (eigenvalues_.empty()) return cd(kNaN, kNaN);
  cd sum(0.0, 0.0);
  for (const cd &v : eigenvalues_) sum += v;
  return sum / static_cast<double>(eigenvalues_.size());
}

FiberOverlapRead SpectralFiber::overlap(const SpectralFiber &a,
                                        const SpectralFiber &b) {
  FiberOverlapRead read;
  std::map<std::vector<std::uint64_t>, std::size_t> rowOfA;
  for (std::size_t i = 0; i < a.cells_.size(); ++i) rowOfA[a.cells_[i]] = i;
  std::vector<std::size_t> ia;
  std::vector<std::size_t> ib;
  for (std::size_t j = 0; j < b.cells_.size(); ++j) {
    const auto it = rowOfA.find(b.cells_[j]);
    if (it != rowOfA.end()) {
      ia.push_back(it->second);
      ib.push_back(j);
    }
  }
  read.sharedCells = ia.size();
  const std::size_t unionCells =
      a.cells_.size() + b.cells_.size() - ia.size();
  read.supportOverlap =
      unionCells == 0 ? 0.0
                      : static_cast<double>(ia.size()) /
                            static_cast<double>(unionCells);
  if (ia.empty() || a.rank() == 0 || b.rank() == 0) return read;

  Eigen::MatrixXcd ra(static_cast<Eigen::Index>(ia.size()), a.right_.cols());
  Eigen::MatrixXcd rb(static_cast<Eigen::Index>(ib.size()), b.right_.cols());
  for (std::size_t r = 0; r < ia.size(); ++r) {
    ra.row(static_cast<Eigen::Index>(r)) =
        a.right_.row(static_cast<Eigen::Index>(ia[r]));
    rb.row(static_cast<Eigen::Index>(r)) =
        b.right_.row(static_cast<Eigen::Index>(ib[r]));
  }
  const Eigen::MatrixXcd qa = thinOrthonormal(ra);
  const Eigen::MatrixXcd qb = thinOrthonormal(rb);
  if (qa.cols() == 0 || qb.cols() == 0) return read;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(qa.adjoint() * qb);
  double sumSq = 0.0;
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
    const double c = std::min(1.0, std::max(0.0, svd.singularValues()[i]));
    read.principalAngles.push_back(std::acos(c));
    sumSq += c * c;
  }
  std::sort(read.principalAngles.begin(), read.principalAngles.end());
  read.subspaceOverlap =
      sumSq / static_cast<double>(std::max(a.rank(), b.rank()));
  return read;
}

Record SpectralFiber::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kSchemaVersion);
  m["record_type"] = Record("spectral_fiber");
  m["cells"] = cellsToRecord(cells_);
  m["rows"] = Record(static_cast<std::int64_t>(right_.rows()));
  m["rank"] = Record(static_cast<std::int64_t>(right_.cols()));
  Record::splitComplex(m, "eigenvalues", eigenvalues_);
  Record::splitComplex(m, "right_frame", matrixToFlat(right_));
  Record::splitComplex(m, "left_frame", matrixToFlat(left_));
  std::vector<cd> w(static_cast<std::size_t>(weights_.size()));
  for (Eigen::Index i = 0; i < weights_.size(); ++i)
    w[static_cast<std::size_t>(i)] = weights_[i];
  Record::splitComplex(m, "weights", w);
  m["certificate"] = bandCertificateToRecord(certificate_);
  return Record(std::move(m));
}

SpectralFiber SpectralFiber::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "spectral_fiber");
  const auto rows = static_cast<std::size_t>(m.at("rows").asInt());
  const auto rank = static_cast<std::size_t>(m.at("rank").asInt());
  SpectralFiber fiber;
  fiber.cells_ = cellsFromRecord(m.at("cells"));
  if (fiber.cells_.size() != rows)
    throw std::invalid_argument("SpectralFiber: cell/row count mismatch");
  fiber.eigenvalues_ = complexListFromRecord(m, "eigenvalues");
  fiber.right_ = matrixFromFlat(complexListFromRecord(m, "right_frame"), rows,
                                rank);
  fiber.left_ =
      matrixFromFlat(complexListFromRecord(m, "left_frame"), rows, rank);
  const std::vector<cd> w = complexListFromRecord(m, "weights");
  if (w.size() != rows)
    throw std::invalid_argument("SpectralFiber: weight length mismatch");
  fiber.weights_ = Eigen::VectorXcd(static_cast<Eigen::Index>(rows));
  for (std::size_t i = 0; i < rows; ++i)
    fiber.weights_[static_cast<Eigen::Index>(i)] = w[i];
  fiber.certificate_ = bandCertificateFromRecord(m.at("certificate"));
  return fiber;
}

// ---------------------------------------------------------------------------
// ComponentBandRead serialization
// ---------------------------------------------------------------------------

Record ComponentBandRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kSchemaVersion);
  m["record_type"] = Record("spectral_band_read");
  Record::List supportList;
  supportList.reserve(support.size());
  for (const std::uint64_t v : support)
    supportList.emplace_back(static_cast<std::int64_t>(v));
  m["support"] = Record(std::move(supportList));
  m["degree"] = Record(degree);
  m["dimension"] = Record(static_cast<std::int64_t>(dimension));
  m["cell_vertices"] = cellsToRecord(cellVertices);
  m["regime"] = Record(regimeName(regime));
  m["solver_path"] = Record(solverPath);
  m["truncated"] = Record(truncated);
  Record::splitComplex(m, "covered_eigenvalues", coveredEigenvalues);
  Record::List fiberList;
  fiberList.reserve(fibers.size());
  for (const SpectralFiber &f : fibers) fiberList.push_back(f.toRecord());
  m["fibers"] = Record(std::move(fiberList));
  m["solve_certificate"] = certificateToRecord(solveCertificate);
  return Record(std::move(m));
}

ComponentBandRead ComponentBandRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "spectral_band_read");
  ComponentBandRead read;
  for (const Record &v : m.at("support").asList())
    read.support.push_back(static_cast<std::uint64_t>(v.asInt()));
  read.degree = static_cast<int>(m.at("degree").asInt());
  read.dimension = static_cast<std::size_t>(m.at("dimension").asInt());
  read.cellVertices = cellsFromRecord(m.at("cell_vertices"));
  read.regime = regimeFromName(m.at("regime").asString());
  read.solverPath = m.at("solver_path").asString();
  read.truncated = m.at("truncated").asBool();
  read.coveredEigenvalues = complexListFromRecord(m, "covered_eigenvalues");
  for (const Record &f : m.at("fibers").asList())
    read.fibers.push_back(SpectralFiber::fromRecord(f));
  read.solveCertificate = certificateFromRecord(m.at("solve_certificate"));
  return read;
}

// ---------------------------------------------------------------------------
// restricted operator assembly
// ---------------------------------------------------------------------------

/// The component-restricted Hodge data: the induced-subcomplex operator's
/// cells, weights, and (regime-dependent) dense/sparse representations.
struct SpectralFiberTracker::RestrictedOperator {
  int degree = 0;
  std::vector<std::uint64_t> support{};   // sorted ascending (reporting)
  std::vector<std::vector<std::uint64_t>> cells{};
  Eigen::VectorXcd wk{};                  // restricted W_k diagonal
  Eigen::MatrixXcd L{};                   // dense operator (general paths; and
                                          // dense positive path via symmetrize)
  Eigen::SparseMatrix<cd> S{};            // symmetric representation (positive)
  CertificateRegime regime = CertificateRegime::NonNormal;
  bool positive = false;                  // verified positive regime
  double opScale = 0.0;                   // Frobenius norm of the solved matrix
  // Chain-level pencil path: the dressed pencil of the induced subcomplex and
  // the regime's verification residual (M L = (M L)^T).
  std::shared_ptr<const chainhodge::CovariantChainHodge> pencil{};
  double metricSymmetryDefect = kNaN;

  [[nodiscard]] std::size_t dim() const { return cells.size(); }
};

SpectralFiberTracker::SpectralFiberTracker(
    std::shared_ptr<Spacetime> st, SpectralFiberConfig cfg,
    cobordism::HodgeLaplacian::WeightConvention weights)
    : SpectralFiberTracker(std::move(st), std::move(cfg), weights,
                           cobordism::HodgeLaplacian::defaultMetricSource()) {}

SpectralFiberTracker::SpectralFiberTracker(
    std::shared_ptr<Spacetime> st, SpectralFiberConfig cfg,
    cobordism::HodgeLaplacian::MetricSource source)
    : SpectralFiberTracker(std::move(st), std::move(cfg),
                           cobordism::HodgeLaplacian::defaultWeightConvention(),
                           source) {}

SpectralFiberTracker::SpectralFiberTracker(
    std::shared_ptr<Spacetime> st, SpectralFiberConfig cfg,
    cobordism::HodgeLaplacian::WeightConvention weights,
    cobordism::HodgeLaplacian::MetricSource source)
    : st_(std::move(st)), cfg_(std::move(cfg)), weights_(weights),
      metricSource_(source) {
  if (!st_)
    throw std::invalid_argument("SpectralFiberTracker: null spacetime");
}

SpectralFiberTracker::RestrictedOperator
SpectralFiberTracker::assembleRestricted(
    const std::vector<std::uint64_t> &support, int degree) const {
  if (degree < 0)
    throw std::invalid_argument("SpectralFiberTracker: negative degree");
  RestrictedOperator op;
  op.degree = degree;
  op.support = normalizedSupport(support);
  const std::unordered_set<std::uint64_t> members(op.support.begin(),
                                                  op.support.end());

  if (degree == 0) {
    // Induced-subgraph Hermitian U(1) connection Laplacian, in
    // HodgeLaplacian::connectionLaplacian's conventions: A_ij = sum l^2
    // e^{i phase} (stored source->target carries +phase), D_ii = sum |l^2|,
    // L = D - A. This is not the Hodge laplacian(0) = d_1 W_1^-1 d_1^T: a
    // degree-0 spectral band tracks Aharonov-Bohm structure, which only the
    // connection operator carries.
    std::vector<std::uint64_t> ids;
    for (const auto &v : st_->getVertexList()->toVector()) {
      if (v == nullptr) continue;
      if (members.count(v->getId())) ids.push_back(v->getId());
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::unordered_map<std::uint64_t, Eigen::Index> index;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      index[ids[i]] = static_cast<Eigen::Index>(i);
      op.cells.push_back({ids[i]});
    }
    const auto n = static_cast<Eigen::Index>(ids.size());
    op.wk = Eigen::VectorXcd::Ones(n);
    Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(n, n);
    Eigen::VectorXd D = Eigen::VectorXd::Zero(n);
    for (const auto &e : st_->getEdgeList()->toVector()) {
      if (e == nullptr || e->getSource() == nullptr ||
          e->getTarget() == nullptr)
        continue;
      const auto is = index.find(e->getSource()->getId());
      const auto it = index.find(e->getTarget()->getId());
      if (is == index.end() || it == index.end()) continue;
      if (is->second == it->second) continue;  // no self-loops
      const cd w = e->getLength() * e->getLength();
      // The reverse orientation carries the inverse link e^{-i*phase}, not its
      // conjugate; the two agree only for real phase. Mirrors
      // HodgeLaplacian::assemble, the operator this reproduces.
      const cd phase = e->getPhase();
      A(is->second, it->second) += w * std::exp(cd(0.0, 1.0) * phase);
      A(it->second, is->second) += std::conj(w) * std::exp(cd(0.0, -1.0) * phase);
      D[is->second] += std::abs(w);
      D[it->second] += std::abs(w);
    }
    op.L = Eigen::MatrixXcd(D.cast<cd>().asDiagonal());
    op.L -= A;
    const double norm = op.L.norm();
    const double hermDefect = (op.L - op.L.adjoint()).norm();
    if (hermDefect <= 1e-12 * std::max(1.0, norm)) {
      // Positive semidefinite by Gershgorin, for every complex or signed edge
      // weight: the diagonal is sum_e |l^2_e| over the same induced edge set
      // the off-diagonals sum over, and |sum_e z_e e^{i theta}| <= sum_e |z_e|,
      // so the operator is Hermitian and diagonally dominant with a
      // non-negative diagonal. This holds for the connection operator only; it
      // does not transfer to the Hodge L_0.
      op.positive = true;
      op.regime = CertificateRegime::PositiveSemidefinite;
      op.S = op.L.sparseView();
    } else {
      op.regime = CertificateRegime::NonNormal;
    }
    op.opScale = norm;
    return op;
  }
  if (metricSource_ == cobordism::HodgeLaplacian::MetricSource::WhitneyPencil) {
    // The chain-level Whitney pencil of the induced subcomplex on the support:
    // its top cells are the spacetime's top simplices with every vertex in the
    // support, in the reference (ascending id) orientation; squared lengths and
    // links are read from the spacetime's edges.
    std::size_t topSize = 0;
    for (const auto &sp : st_->getSimplices())
      if (sp != nullptr) topSize = std::max(topSize, static_cast<std::size_t>(sp->size()));
    std::vector<std::vector<std::uint64_t>> topCells;
    for (const auto &sp : st_->getSimplices()) {
      if (sp == nullptr || static_cast<std::size_t>(sp->size()) != topSize) continue;
      std::vector<std::uint64_t> cell;
      bool inside = true;
      for (const auto &v : sp->getVertices()) {
        if (!members.count(v->getId())) { inside = false; break; }
        cell.push_back(v->getId());
      }
      if (!inside) continue;
      std::sort(cell.begin(), cell.end());
      topCells.push_back(std::move(cell));
    }
    if (topCells.empty()) return op;  // no cell of the support: empty read
    const cobordism::ChainComplex K = cobordism::ChainComplex::fromTopCells(topCells);
    if (degree > K.dimension()) return op;
    const chainhodge::SquaredLengths s = chainhodge::WhitneyMass::squaredLengthsOf(*st_, K);
    const chainhodge::Connection U = chainhodge::Connection::fromSpacetime(*st_, K);
    const chainhodge::ChainHodge base(K, s, chainhodge::Preset::L2,
                                      chainhodge::Branch::Continuation,
                                      std::numeric_limits<int>::max());
    auto cov = std::make_shared<const chainhodge::CovariantChainHodge>(
        base, U, 7, /*measureCertificate=*/false);
    op.cells = K.kSimplexVertices(degree);
    const auto n = static_cast<Eigen::Index>(op.cells.size());
    op.wk = Eigen::VectorXcd::Ones(n);  // identity placeholder: the pairing is bilinear
    op.L = cov->covariantOperator(degree);
    op.opScale = op.L.norm();
    const chainhodge::PencilRegimeCertificate regime = cov->regimeCertificate(degree);
    op.regime = regime.regime;
    op.metricSymmetryDefect = std::max(regime.symmetryDefect, regime.metricSymmetryDefect);
    op.positive = false;
    op.pencil = cov;
    return op;
  }

  // degree >= 1: restrict the canonical ChainComplex boundary maps and the
  // HodgeLaplacian inner-product weights (identical conventions, consumed
  // read-only) to the cells fully inside the support.
  const cobordism::ChainComplex cc =
      cobordism::ChainComplex::fromSpacetime(*st_);
  const cobordism::HodgeLaplacian hodge(
      st_, weights_, cobordism::HodgeLaplacian::MetricSource::DiagonalWeights);
  const auto insideIndices =
      [&](int k) -> std::pair<std::vector<std::size_t>,
                              std::vector<std::vector<std::uint64_t>>> {
    std::vector<std::size_t> idx;
    std::vector<std::vector<std::uint64_t>> kept;
    const auto cells = cc.kSimplexVertices(k);
    for (std::size_t i = 0; i < cells.size(); ++i) {
      bool inside = true;
      for (const std::uint64_t v : cells[i])
        if (!members.count(v)) {
          inside = false;
          break;
        }
      if (inside) {
        idx.push_back(i);
        kept.push_back(cells[i]);
      }
    }
    return {std::move(idx), std::move(kept)};
  };

  auto [idxK, cellsK] = insideIndices(degree);
  op.cells = std::move(cellsK);
  const auto nk = static_cast<Eigen::Index>(idxK.size());
  if (nk == 0) return op;

  const auto restrictedWeights = [&](int k, const std::vector<std::size_t> &idx)
      -> Eigen::VectorXcd {
    const std::vector<cd> full = hodge.weights(k);
    Eigen::VectorXcd w(static_cast<Eigen::Index>(idx.size()));
    for (std::size_t i = 0; i < idx.size(); ++i)
      w[static_cast<Eigen::Index>(i)] = full[idx[i]];
    return w;
  };
  op.wk = restrictedWeights(degree, idxK);

  auto [idxLower, cellsLower] = insideIndices(degree - 1);
  auto [idxUpper, cellsUpper] = insideIndices(degree + 1);
  const auto nLower = static_cast<Eigen::Index>(idxLower.size());
  const auto nUpper = static_cast<Eigen::Index>(idxUpper.size());

  // Restricted boundary matrices (sparse). d_k columns for cells inside the
  // support only touch faces inside the support (a face of an inside cell is
  // inside), so the row restriction drops nothing; a (k+1)-cell with a vertex
  // outside is dropped — the induced subcomplex is read as a complex in its
  // own right.
  const auto restrictedBoundary =
      [&](int k, const std::vector<std::size_t> &rowIdx,
          const std::vector<std::size_t> &colIdx) -> Eigen::SparseMatrix<cd> {
    const std::vector<long> &flat = cc.boundaryMatrix(k);
    const auto fullRows = cc.numSimplices(k - 1);
    std::unordered_map<std::size_t, Eigen::Index> rowOf;
    for (std::size_t i = 0; i < rowIdx.size(); ++i)
      rowOf[rowIdx[i]] = static_cast<Eigen::Index>(i);
    std::vector<Eigen::Triplet<cd>> trips;
    const std::size_t fullCols = cc.numSimplices(k);
    for (std::size_t c = 0; c < colIdx.size(); ++c) {
      const std::size_t col = colIdx[c];
      for (std::size_t r = 0; r < fullRows; ++r) {
        const long entry = flat[r * fullCols + col];
        if (entry == 0) continue;
        const auto rr = rowOf.find(r);
        if (rr == rowOf.end()) continue;  // face outside (cannot happen for
                                          // an induced subcomplex; kept as a
                                          // guard)
        trips.emplace_back(rr->second, static_cast<Eigen::Index>(c),
                           cd(static_cast<double>(entry), 0.0));
      }
    }
    Eigen::SparseMatrix<cd> d(static_cast<Eigen::Index>(rowIdx.size()),
                              static_cast<Eigen::Index>(colIdx.size()));
    d.setFromTriplets(trips.begin(), trips.end());
    return d;
  };

  Eigen::SparseMatrix<cd> dk;
  Eigen::SparseMatrix<cd> dk1;
  Eigen::VectorXcd wLower;
  Eigen::VectorXcd wUpper;
  if (nLower > 0) {
    dk = restrictedBoundary(degree, idxLower, idxK);
    wLower = restrictedWeights(degree - 1, idxLower);
  }
  if (nUpper > 0) {
    dk1 = restrictedBoundary(degree + 1, idxK, idxUpper);
    wUpper = restrictedWeights(degree + 1, idxUpper);
  }

  // Regime classification from the participating weights.
  bool complexWeights = false;
  bool negativeWeights = false;
  const auto classify = [&](const Eigen::VectorXcd &w) {
    for (Eigen::Index i = 0; i < w.size(); ++i) {
      if (std::abs(w[i].imag()) > 1e-14 * std::max(1.0, std::abs(w[i])))
        complexWeights = true;
      else if (w[i].real() < 0.0)
        negativeWeights = true;
    }
  };
  classify(op.wk);
  if (nLower > 0) classify(wLower);
  if (nUpper > 0) classify(wUpper);

  if (!complexWeights && !negativeWeights) {
    // Positive regime: assemble the symmetric W-orthonormal representation
    // S = B_k^T B_k + B_{k+1} B_{k+1}^T, B_k = W_{k-1}^{1/2} d_k W_k^{-1/2}.
    op.positive = true;
    op.regime = CertificateRegime::PositiveSemidefinite;
    Eigen::VectorXcd sqrtWk(nk);
    Eigen::VectorXcd invSqrtWk(nk);
    for (Eigen::Index i = 0; i < nk; ++i) {
      const double w = op.wk[i].real();
      sqrtWk[i] = cd(std::sqrt(w), 0.0);
      invSqrtWk[i] = cd(1.0 / std::sqrt(w), 0.0);
    }
    Eigen::SparseMatrix<cd> S(nk, nk);
    if (nLower > 0) {
      Eigen::VectorXcd sqrtLower(nLower);
      for (Eigen::Index i = 0; i < nLower; ++i)
        sqrtLower[i] = cd(std::sqrt(wLower[i].real()), 0.0);
      const Eigen::SparseMatrix<cd> B =
          sqrtLower.asDiagonal() * dk * invSqrtWk.asDiagonal();
      S += Eigen::SparseMatrix<cd>(B.adjoint()) * B;
    }
    if (nUpper > 0) {
      Eigen::VectorXcd invSqrtUpper(nUpper);
      for (Eigen::Index i = 0; i < nUpper; ++i)
        invSqrtUpper[i] = cd(1.0 / std::sqrt(wUpper[i].real()), 0.0);
      const Eigen::SparseMatrix<cd> B2 =
          sqrtWk.asDiagonal() * dk1 * invSqrtUpper.asDiagonal();
      S += B2 * Eigen::SparseMatrix<cd>(B2.adjoint());
    }
    op.S = S;
    op.opScale = S.norm();
    return op;
  }

  // Signed / complex weights: assemble the direct (generally non-symmetric)
  // operator L = W_k^{-1} d_k^T W_{k-1} d_k + d_{k+1} W_{k+1}^{-1} d_{k+1}^T
  // W_k in the cochain coordinates.
  Eigen::MatrixXcd L = Eigen::MatrixXcd::Zero(nk, nk);
  Eigen::VectorXcd invWk(nk);
  for (Eigen::Index i = 0; i < nk; ++i) invWk[i] = cd(1.0, 0.0) / op.wk[i];
  if (nLower > 0) {
    const Eigen::MatrixXcd dkDense = Eigen::MatrixXcd(dk);
    L += invWk.asDiagonal() * dkDense.transpose() * wLower.asDiagonal() *
         dkDense;
  }
  if (nUpper > 0) {
    Eigen::VectorXcd invWUpper(nUpper);
    for (Eigen::Index i = 0; i < nUpper; ++i)
      invWUpper[i] = cd(1.0, 0.0) / wUpper[i];
    const Eigen::MatrixXcd dk1Dense = Eigen::MatrixXcd(dk1);
    L += dk1Dense * invWUpper.asDiagonal() * dk1Dense.transpose() *
         op.wk.asDiagonal();
  }
  op.L = std::move(L);
  op.opScale = op.L.norm();

  if (complexWeights) {
    op.regime = CertificateRegime::NonNormal;
  } else {
    // Real signed weights: verify W-self-adjointness (W L symmetric) before
    // claiming the Hermitian-indefinite (Krein) regime.
    const Eigen::MatrixXcd WL = op.wk.asDiagonal() * op.L;
    const double defect = (WL - WL.transpose()).norm();
    op.regime = defect <= 1e-10 * std::max(1.0, WL.norm())
                    ? CertificateRegime::HermitianIndefinite
                    : CertificateRegime::NonNormal;
  }
  return op;
}

// ---------------------------------------------------------------------------
// solve paths
// ---------------------------------------------------------------------------

/// One solve path's output, in cochain coordinates: eigen-paired right vectors
/// Phi, Euclidean left vectors Y (Y^dagger Phi ~ I on the covered pairs),
/// (Re, Im)-sorted eigenvalues, per-pair absolute residual norms, and the
/// optional truncation shield (the first uncovered Ritz value).
struct SpectralFiberTracker::SolveOutput {
  Eigen::MatrixXcd right{};
  Eigen::MatrixXcd left{};
  std::vector<cd> eigenvalues{};
  std::vector<double> rightResidual{};
  std::vector<double> leftResidual{};
  bool selfAdjoint = false;
  double shield = kNaN;
};

void SpectralFiberTracker::solveDenseSelfAdjoint(const RestrictedOperator &op,
                                                 ComponentBandRead &read) const {
  const auto n = static_cast<Eigen::Index>(op.dim());
  const Eigen::MatrixXcd Sd(op.S);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Sd);
  if (es.info() != Eigen::Success)
    throw std::runtime_error(
        "SpectralFiberTracker: dense self-adjoint eigensolve failed");
  SolveOutput out;
  out.selfAdjoint = true;
  const Eigen::VectorXd theta = es.eigenvalues();
  const Eigen::MatrixXcd U = es.eigenvectors();

  // Map the W-orthonormal symmetric eigenvectors to cochain coordinates:
  // Phi = W^{-1/2} U (right), Y = W^{1/2} U (Euclidean left), Y^dagger Phi = I.
  Eigen::VectorXcd sqrtW(n);
  Eigen::VectorXcd invSqrtW(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double w = op.wk[i].real();
    sqrtW[i] = cd(std::sqrt(w), 0.0);
    invSqrtW[i] = cd(1.0 / std::sqrt(w), 0.0);
  }
  out.right = invSqrtW.asDiagonal() * U;
  out.left = sqrtW.asDiagonal() * U;
  out.eigenvalues.resize(static_cast<std::size_t>(n));
  out.rightResidual.resize(static_cast<std::size_t>(n));
  out.leftResidual.resize(static_cast<std::size_t>(n));
  double maxResidual = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    out.eigenvalues[static_cast<std::size_t>(i)] = cd(theta[i], 0.0);
    // Residual measured in the coordinates actually solved (the symmetric
    // W-orthonormal representation).
    const double r = (Sd * U.col(i) - theta[i] * U.col(i)).norm();
    out.rightResidual[static_cast<std::size_t>(i)] = r;
    out.leftResidual[static_cast<std::size_t>(i)] = r;
    maxResidual = std::max(maxResidual, r);
  }
  read.solverPath = "dense-self-adjoint";
  read.truncated = false;
  const double relResidual =
      op.opScale > 0.0 ? maxResidual / op.opScale : maxResidual;
  read.solveCertificate = Certificate::certifiedNumerical(
      CertificateDomain::Static, CertificateRegime::PositiveSemidefinite,
      relResidual, 1.0, cfg_.residualTolerance);

  if (cfg_.crossValidateDense && n >= 1 && n < cfg_.denseCrossover) {
    const cobordism::DenseReference reference(cfg_.denseCrossover);
    const auto certified = reference.spectrum(matrixToFlat(Sd),
                                              static_cast<int>(n), true);
    double dev = 0.0;
    const double scale = std::max(
        1.0, std::max(std::abs(theta[0]), std::abs(theta[n - 1])));
    for (Eigen::Index i = 0; i < n; ++i)
      dev = std::max(dev, std::abs(certified.values[static_cast<std::size_t>(
                                       i)].real() -
                                   theta[i]) /
                              scale);
    read.solveCertificate.setDenseReferenceError(dev);
  }

  read.coveredEigenvalues = out.eigenvalues;
  buildFibers(op, out, read);
}

void SpectralFiberTracker::solveSparseSelfAdjoint(const RestrictedOperator &op,
                                                  ComponentBandRead &read) const {
  const auto n = static_cast<Eigen::Index>(op.dim());
  const int m = std::min<int>(std::max(1, cfg_.requestedEigenpairs),
                              static_cast<int>(n));
  const int b = std::min<int>(m + std::max(1, cfg_.oversample),
                              static_cast<int>(n));

  // Deterministic shift-invert block subspace iteration: factor S + sigma I
  // once (S is positive semidefinite here, so the shift keeps it positive
  // definite), amplify the lowest eigenspace, Rayleigh-Ritz on S.
  const double meanEig =
      n > 0 ? std::abs(Eigen::VectorXcd(op.S.diagonal()).sum().real()) /
                  static_cast<double>(n)
            : 0.0;
  const double sigma = std::max(1e-8 * std::max(meanEig, 1.0), 1e-300);
  Eigen::SparseMatrix<cd> shifted = op.S;
  for (Eigen::Index i = 0; i < n; ++i) shifted.coeffRef(i, i) += cd(sigma, 0.0);
  shifted.makeCompressed();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<cd>> ldlt;
  ldlt.compute(shifted);
  if (ldlt.info() != Eigen::Success)
    throw std::runtime_error(
        "SpectralFiberTracker: sparse shift factorization failed");

  std::mt19937_64 rng(cfg_.solverSeed + 0x9e3779b97f4a7c15ULL);
  std::normal_distribution<double> gauss(0.0, 1.0);
  Eigen::MatrixXcd X(n, b);
  for (Eigen::Index j = 0; j < b; ++j)
    for (Eigen::Index i = 0; i < n; ++i) X(i, j) = cd(gauss(rng), gauss(rng));

  Eigen::VectorXd theta = Eigen::VectorXd::Zero(b);
  Eigen::MatrixXcd V(n, b);
  Eigen::VectorXd residuals = Eigen::VectorXd::Constant(b, kInf);
  const int watch = std::min<int>(m + 1, b);
  const double target = cfg_.solverTolerance * std::max(op.opScale, 1e-300);
  for (int iter = 0; iter < std::max(1, cfg_.maxSolverIterations); ++iter) {
    X = ldlt.solve(X);
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(X);
    const Eigen::MatrixXcd Q =
        qr.householderQ() * Eigen::MatrixXcd::Identity(n, b);
    const Eigen::MatrixXcd SQ = op.S * Q;
    Eigen::MatrixXcd T = Q.adjoint() * SQ;
    T = 0.5 * (T + T.adjoint()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> small(T);
    theta = small.eigenvalues();
    V = Q * small.eigenvectors();
    const Eigen::MatrixXcd SV = SQ * small.eigenvectors();
    double worst = 0.0;
    for (int j = 0; j < b; ++j) {
      residuals[j] = (SV.col(j) - theta[j] * V.col(j)).norm();
      if (j < watch) worst = std::max(worst, residuals[j]);
    }
    X = V;
    if (worst <= target) break;
  }

  // Truncation-safe coverage: cover the first c <= m pairs where a certified
  // relative gap separates the covered block from the first uncovered Ritz
  // value, so a band is not split at the truncation edge.
  int covered = static_cast<int>(n);
  double shield = kNaN;
  bool truncated = false;
  if (b < static_cast<int>(n)) {
    truncated = true;
    double scale = 0.0;
    for (int j = 0; j < b; ++j) scale = std::max(scale, std::abs(theta[j]));
    if (scale == 0.0) scale = 1.0;
    covered = 0;
    for (int c = m; c >= 1; --c) {
      if (theta[c] - theta[c - 1] >= cfg_.minRelativeGap * scale) {
        covered = c;
        break;
      }
    }
    if (covered > 0) shield = theta[covered];
  } else {
    covered = b;  // full coverage: b == n
  }

  SolveOutput out;
  out.selfAdjoint = true;
  out.shield = shield;
  Eigen::VectorXcd sqrtW(n);
  Eigen::VectorXcd invSqrtW(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double w = op.wk[i].real();
    sqrtW[i] = cd(std::sqrt(w), 0.0);
    invSqrtW[i] = cd(1.0 / std::sqrt(w), 0.0);
  }
  out.right = invSqrtW.asDiagonal() * V.leftCols(covered);
  out.left = sqrtW.asDiagonal() * V.leftCols(covered);
  double maxResidual = 0.0;
  for (int j = 0; j < covered; ++j) {
    out.eigenvalues.emplace_back(theta[j], 0.0);
    out.rightResidual.push_back(residuals[j]);
    out.leftResidual.push_back(residuals[j]);
    maxResidual = std::max(maxResidual, residuals[j]);
  }
  read.solverPath = "sparse-block-self-adjoint";
  read.truncated = truncated;
  const double relResidual =
      op.opScale > 0.0 ? maxResidual / op.opScale : maxResidual;
  read.solveCertificate = Certificate::certifiedNumerical(
      CertificateDomain::Static, CertificateRegime::PositiveSemidefinite,
      relResidual, 1.0, std::max(cfg_.solverTolerance, cfg_.residualTolerance));
  read.coveredEigenvalues = out.eigenvalues;
  buildFibers(op, out, read);
}

void SpectralFiberTracker::solveDenseGeneral(const RestrictedOperator &op,
                                             ComponentBandRead &read) const {
  const auto n = static_cast<Eigen::Index>(op.dim());
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(op.L, true);
  if (es.info() != Eigen::Success)
    throw std::runtime_error(
        "SpectralFiberTracker: dense general eigensolve failed");
  // Sort eigenpairs by (Re, Im) and permute right/left coherently.
  std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
  for (Eigen::Index i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) {
    return lessReIm(es.eigenvalues()[a], es.eigenvalues()[b]);
  });
  Eigen::MatrixXcd V(n, n);
  std::vector<cd> lambda(static_cast<std::size_t>(n));
  for (Eigen::Index j = 0; j < n; ++j) {
    V.col(j) = es.eigenvectors().col(order[static_cast<std::size_t>(j)]);
    lambda[static_cast<std::size_t>(j)] =
        es.eigenvalues()[order[static_cast<std::size_t>(j)]];
  }
  // Euclidean left vectors: Y^dagger = V^{-1} row block, so Y^dagger V = I.
  const Eigen::MatrixXcd Vinv = V.fullPivLu().inverse();
  const Eigen::MatrixXcd Y = Vinv.adjoint();

  SolveOutput out;
  out.selfAdjoint = false;
  out.right = V;
  out.left = Y;
  out.eigenvalues = lambda;
  out.rightResidual.resize(static_cast<std::size_t>(n));
  out.leftResidual.resize(static_cast<std::size_t>(n));
  const Eigen::MatrixXcd Ladj = op.L.adjoint();
  double maxResidual = 0.0;
  for (Eigen::Index j = 0; j < n; ++j) {
    const double rr =
        (op.L * V.col(j) - lambda[static_cast<std::size_t>(j)] * V.col(j))
            .norm();
    const double rl = (Ladj * Y.col(j) -
                       std::conj(lambda[static_cast<std::size_t>(j)]) *
                           Y.col(j))
                          .norm();
    out.rightResidual[static_cast<std::size_t>(j)] = rr;
    out.leftResidual[static_cast<std::size_t>(j)] = rl;
    maxResidual = std::max(maxResidual, std::max(rr, rl));
  }
  // Eigenvector-matrix conditioning (the general-eigensolver conditioning
  // convention of DenseReference).
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(V);
  const double smin = svd.singularValues()[n - 1];
  const double kappa =
      smin > 0.0 ? svd.singularValues()[0] / smin : kInf;

  read.solverPath = "dense-general";
  read.truncated = false;
  const double relResidual =
      op.opScale > 0.0 ? maxResidual / op.opScale : maxResidual;
  read.solveCertificate = Certificate::certifiedNumerical(
      CertificateDomain::Static, op.regime, relResidual, kappa,
      cfg_.residualTolerance);

  if (cfg_.crossValidateDense && n >= 1 && n < cfg_.denseCrossover) {
    const cobordism::DenseReference reference(cfg_.denseCrossover);
    const auto certified =
        reference.spectrum(matrixToFlat(op.L), static_cast<int>(n), false);
    double scale = 1.0;
    for (const cd &v : lambda) scale = std::max(scale, std::abs(v));
    double dev = 0.0;
    for (Eigen::Index i = 0; i < n; ++i)
      dev = std::max(dev,
                     std::abs(certified.values[static_cast<std::size_t>(i)] -
                              lambda[static_cast<std::size_t>(i)]) /
                         scale);
    read.solveCertificate.setDenseReferenceError(dev);
  }

  read.coveredEigenvalues = out.eigenvalues;
  buildFibers(op, out, read);
}

// ---------------------------------------------------------------------------
// band grouping, measurement, certification
// ---------------------------------------------------------------------------

void SpectralFiberTracker::solvePencilBands(const RestrictedOperator &op,
                                            ComponentBandRead &read) const {
  // Dense eigenvalues of h_k(s,U) locate the bands (the gap rule of
  // SpectralFiberConfig, sorted by (Re, Im)); every band is then the Riesz
  // projector of a circular contour drawn around its group, computed on the
  // pencil resolvent. Nothing here reads a sign or an inertia from the
  // pairing.
  const auto n = static_cast<Eigen::Index>(op.dim());
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(op.L, false);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("SpectralFiberTracker: dense pencil eigensolve failed");
  std::vector<cd> lambda(static_cast<std::size_t>(n));
  for (Eigen::Index i = 0; i < n; ++i) lambda[static_cast<std::size_t>(i)] = es.eigenvalues()[i];
  std::sort(lambda.begin(), lambda.end(), lessReIm);
  read.solverPath = "pencil-riesz";
  read.truncated = false;
  read.coveredEigenvalues = lambda;
  double scale = 0.0;
  for (const cd &v : lambda) scale = std::max(scale, std::abs(v));
  if (scale == 0.0) scale = 1.0;
  std::vector<std::size_t> starts{0};
  for (std::size_t i = 1; i < lambda.size(); ++i)
    if (std::abs(lambda[i] - lambda[i - 1]) > cfg_.groupingTolerance * scale)
      starts.push_back(i);
  starts.push_back(lambda.size());
  double worstResidual = 0.0;
  double worstProjectorNorm = 1.0;
  for (std::size_t bandIdx = 0; bandIdx + 1 < starts.size(); ++bandIdx) {
    const std::size_t a = starts[bandIdx];
    const std::size_t b = starts[bandIdx + 1];
    std::vector<cd> bandEigs(lambda.begin() + static_cast<std::ptrdiff_t>(a),
                             lambda.begin() + static_cast<std::ptrdiff_t>(b));
    SpectralBandCertificate cert;
    cert.degree = op.degree;
    cert.selfAdjoint = false;
    cert.metricSymmetryDefect = op.metricSymmetryDefect;
    cert.frequencyLower = kInf;
    cert.frequencyUpper = -kInf;
    cd center(0.0, 0.0);
    for (const cd &v : bandEigs) {
      cert.frequencyLower = std::min(cert.frequencyLower, v.real());
      cert.frequencyUpper = std::max(cert.frequencyUpper, v.real());
      center += v;
    }
    center /= static_cast<double>(bandEigs.size());
    double spread = 0.0;
    double radiusInner = 0.0;
    for (std::size_t i = a; i < b; ++i) {
      radiusInner = std::max(radiusInner, std::abs(lambda[i] - center));
      for (std::size_t j = i + 1; j < b; ++j)
        spread = std::max(spread, std::abs(lambda[i] - lambda[j]));
    }
    cert.lowerGap = a > 0 ? std::abs(lambda[a] - lambda[a - 1]) : kInf;
    cert.upperGap = b < lambda.size() ? std::abs(lambda[b] - lambda[b - 1]) : kInf;
    double separation = kInf;
    for (std::size_t i = a; i < b; ++i)
      for (std::size_t j = 0; j < lambda.size(); ++j) {
        if (j >= a && j < b) continue;
        separation = std::min(separation, std::abs(lambda[i] - lambda[j]));
      }
    cert.nearestDiscardedSeparation = separation;
    // The contour: centred on the band, enclosing every member with half the
    // separation to the nearest discarded eigenvalue as clearance (a radius
    // of the spectral scale when nothing is discarded).
    const double clearance = std::isfinite(separation) ? 0.5 * separation : scale;
    const double radius = radiusInner + std::max(clearance, 1e-12 * scale);
    const chainhodge::Contour contour =
        chainhodge::Contour::circle(center, radius, cfg_.contourNodes);
    const chainhodge::Band band =
        op.pencil->band(op.degree, contour, 10.0, cfg_.isotropyTolerance);
    const auto rank = static_cast<std::size_t>(band.rank());
    cert.rank = rank;
    cert.eigenResidual = band.certificate.rightResidual;
    cert.leftResidual = band.certificate.leftResidual;
    cert.projectorResidual = band.certificate.idempotency;
    cert.pairingDeterminant = band.certificate.detB;
    cert.pairingCondition = band.certificate.condB;
    cert.pairingScale = band.certificate.pairingScale;
    cert.isotropic = !band.certificate.leftFrameAvailable;
    cert.leftFrameRefusal = band.certificate.leftFrameRefusal;
    cert.bilinearLeftFrame = true;  // Psi below is Phi~ itself
    cert.positiveSignature = 0;  // no inertia in the bilinear regime
    cert.negativeSignature = 0;
    Eigen::MatrixXcd Phi = band.frame;
    Eigen::MatrixXcd Psi = band.leftFrame;  // Phi~ (empty when refused)
    if (band.certificate.leftFrameAvailable && Psi.cols() == Phi.cols()) {
      cert.gramDefect =
          (Psi.transpose() * Phi -
           Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(rank),
                                      static_cast<Eigen::Index>(rank)))
              .norm();
    } else {
      cert.gramDefect = kNaN;  // refused left frame: unmeasured, never zero
      Psi = Eigen::MatrixXcd::Zero(Phi.rows(), Phi.cols());
    }
    Eigen::JacobiSVD<Eigen::MatrixXcd> psvd(band.projector);
    cert.projectorNorm =
        psvd.singularValues().size() > 0 ? psvd.singularValues()[0] : 0.0;
    cert.frameConditionNumber =
        std::max(frameCondition(Phi, op.wk), frameCondition(Psi, op.wk));
    // Localization of the Riesz projector's diagonal, as on the other paths.
    double diagSum = 0.0;
    std::vector<double> diagAbs(static_cast<std::size_t>(n), 0.0);
    for (Eigen::Index i = 0; i < n; ++i) {
      diagAbs[static_cast<std::size_t>(i)] = std::abs(band.projector(i, i));
      diagSum += diagAbs[static_cast<std::size_t>(i)];
    }
    if (diagSum > 0.0) {
      double diagSq = 0.0;
      for (const double v : diagAbs) {
        const double pi = v / diagSum;
        diagSq += pi * pi;
      }
      cert.localization = diagSq;
      const double effectiveSupport = 1.0 / diagSq;
      cert.localizationSupportFraction =
          effectiveSupport / static_cast<double>(n);
      const double room = static_cast<double>(n) - static_cast<double>(rank);
      cert.localizationExcess =
          room > 0.0
              ? std::min(1.0, std::max(0.0, (effectiveSupport -
                                             static_cast<double>(rank)) / room))
              : 0.0;
    }
    const auto separationOk = [&](double gap) {
      if (std::isnan(gap)) return false;
      if (!std::isfinite(gap)) return true;
      return gap >= cfg_.minRelativeGap * scale && gap >= cfg_.gapDominance * spread;
    };
    const bool residualsOk = cert.eigenResidual <= cfg_.residualTolerance &&
                             cert.leftResidual <= cfg_.residualTolerance &&
                             cert.projectorResidual <= cfg_.residualTolerance;
    const bool localizedOk = std::isfinite(cert.localizationExcess) &&
                             cert.localizationExcess <= cfg_.maxLocalizationExcess;
    cert.accepted = !cert.isotropic &&
                    separationOk(cert.nearestDiscardedSeparation) && localizedOk &&
                    residualsOk && cert.gramDefect <= cfg_.gramDefectTolerance &&
                    cert.projectorNorm <= cfg_.projectorNormCap &&
                    rank == bandEigs.size();
    if (cert.accepted) {
      cert.certificate = Certificate::certifiedNumerical(
          CertificateDomain::BandWindow, op.regime, cert.eigenResidual,
          cert.projectorNorm, cfg_.residualTolerance);
    } else {
      cert.certificate =
          Certificate::heuristicDiscovery(CertificateDomain::BandWindow, op.regime);
    }
    worstResidual = std::max(
        worstResidual, std::isfinite(cert.eigenResidual) ? cert.eigenResidual : 0.0);
    worstProjectorNorm = std::max(
        worstProjectorNorm,
        std::isfinite(cert.projectorNorm) ? cert.projectorNorm : 1.0);
    read.fibers.emplace_back(op.cells, std::move(bandEigs), std::move(Phi),
                             std::move(Psi), Eigen::VectorXcd(op.wk),
                             std::move(cert));
  }
  read.solveCertificate = Certificate::certifiedNumerical(
      CertificateDomain::Static, op.regime, worstResidual, worstProjectorNorm,
      cfg_.residualTolerance);
}

void SpectralFiberTracker::buildFibers(const RestrictedOperator &op,
                                       const SolveOutput &out,
                                       ComponentBandRead &read) const {
  const std::size_t covered = out.eigenvalues.size();
  if (covered == 0) return;

  double scale = 0.0;
  for (const cd &v : out.eigenvalues) scale = std::max(scale, std::abs(v));
  if (std::isfinite(out.shield)) scale = std::max(scale, std::abs(out.shield));
  if (scale == 0.0) scale = 1.0;

  // Relative gap grouping over the (Re, Im)-sorted covered eigenvalues.
  std::vector<std::size_t> starts{0};
  for (std::size_t i = 1; i < covered; ++i) {
    if (std::abs(out.eigenvalues[i] - out.eigenvalues[i - 1]) >
        cfg_.groupingTolerance * scale)
      starts.push_back(i);
  }
  starts.push_back(covered);

  const Eigen::VectorXcd &W = op.wk;
  const auto n = static_cast<Eigen::Index>(op.dim());
  for (std::size_t bandIdx = 0; bandIdx + 1 < starts.size(); ++bandIdx) {
    const std::size_t a = starts[bandIdx];
    const std::size_t b = starts[bandIdx + 1];
    const std::size_t rank = b - a;

    SpectralBandCertificate cert;
    cert.degree = op.degree;
    cert.rank = rank;
    cert.selfAdjoint = out.selfAdjoint;

    std::vector<cd> bandEigs(out.eigenvalues.begin() +
                                 static_cast<std::ptrdiff_t>(a),
                             out.eigenvalues.begin() +
                                 static_cast<std::ptrdiff_t>(b));
    cert.frequencyLower = kInf;
    cert.frequencyUpper = -kInf;
    for (const cd &v : bandEigs) {
      cert.frequencyLower = std::min(cert.frequencyLower, v.real());
      cert.frequencyUpper = std::max(cert.frequencyUpper, v.real());
    }
    double spread = 0.0;
    for (std::size_t i = a; i < b; ++i)
      for (std::size_t j = i + 1; j < b; ++j)
        spread = std::max(spread,
                          std::abs(out.eigenvalues[i] - out.eigenvalues[j]));

    // Sort-order neighbour gaps, reported as diagnostics. The (Re, Im) sort
    // supplies the band grouping, not the isolation: with a genuinely complex
    // spectrum the sorted neighbour need not be the nearest eigenvalue in the
    // plane.
    cert.lowerGap = a > 0
                        ? std::abs(out.eigenvalues[a] - out.eigenvalues[a - 1])
                        : kInf;
    if (b < covered) {
      cert.upperGap = std::abs(out.eigenvalues[b] - out.eigenvalues[b - 1]);
    } else if (std::isfinite(out.shield)) {
      cert.upperGap = out.shield - out.eigenvalues[b - 1].real();
    } else {
      cert.upperGap = read.truncated ? kNaN : kInf;
    }

    // Band gap: the distance in the complex plane to the nearest discarded
    // eigenvalue, over every discarded mode on either side. On a truncated
    // sparse read the uncovered top is bounded by the shield value; without a
    // shield that side is NaN (unknown), not +infinity.
    double separation = kInf;
    for (std::size_t i = a; i < b; ++i) {
      for (std::size_t j = 0; j < covered; ++j) {
        if (j >= a && j < b) continue;
        separation =
            std::min(separation, std::abs(out.eigenvalues[i] -
                                          out.eigenvalues[j]));
      }
    }
    if (b == covered) {
      if (std::isfinite(out.shield))
        separation = std::min(separation,
                              out.shield - out.eigenvalues[b - 1].real());
      else if (read.truncated)
        separation = kNaN;
    }
    cert.nearestDiscardedSeparation = separation;

    // Frames.
    Eigen::MatrixXcd Phi = out.right.middleCols(static_cast<Eigen::Index>(a),
                                                static_cast<Eigen::Index>(rank));
    Eigen::MatrixXcd Yband = out.left.middleCols(static_cast<Eigen::Index>(a),
                                                 static_cast<Eigen::Index>(rank));

    // Residuals (Frobenius over the band's eigen-paired columns).
    double sumR = 0.0;
    double sumL = 0.0;
    for (std::size_t i = a; i < b; ++i) {
      sumR += out.rightResidual[i] * out.rightResidual[i];
      sumL += out.leftResidual[i] * out.leftResidual[i];
    }
    const double denom = std::max(op.opScale, 1e-300);
    cert.eigenResidual = std::sqrt(sumR) / denom;
    cert.leftResidual = std::sqrt(sumL) / denom;

    // Gram / signature and the left-frame normalization Psi^dagger W Phi = I.
    // Biorthogonal fallback shared by the non-normal and neutral-Krein
    // branches: Psi = W^{-dagger} Y (Y^dagger Phi)^{-dagger}, so that
    // Psi^dagger W Phi = (Y^dagger Phi)^{-1} Y^dagger Phi = I; the defect is
    // measured, not assumed.
    const auto biorthogonalPsi = [&](Eigen::MatrixXcd &psi, double &defect) {
      const Eigen::MatrixXcd M = Yband.adjoint() * Phi;
      const Eigen::MatrixXcd Minv = M.fullPivLu().inverse();
      Eigen::VectorXcd invWconj(n);
      for (Eigen::Index i = 0; i < n; ++i)
        invWconj[i] = cd(1.0, 0.0) / std::conj(W[i]);
      psi = invWconj.asDiagonal() * (Yband * Minv.adjoint());
      defect = (psi.adjoint() * (W.asDiagonal() * Phi) -
                Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(rank),
                                           static_cast<Eigen::Index>(rank)))
                   .norm();
    };
    Eigen::MatrixXcd Psi;
    if (out.selfAdjoint) {
      const Eigen::MatrixXcd G =
          Phi.adjoint() * (W.asDiagonal() * Phi);
      cert.gramDefect =
          (G - Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(rank),
                                          static_cast<Eigen::Index>(rank)))
              .norm();
      cert.positiveSignature = static_cast<int>(rank);
      cert.negativeSignature = 0;
      Psi = Phi;
    } else if (op.regime == CertificateRegime::HermitianIndefinite) {
      Eigen::MatrixXcd G = Phi.adjoint() * (W.asDiagonal() * Phi);
      G = 0.5 * (G + G.adjoint()).eval();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ges(G);
      const Eigen::VectorXd d = ges.eigenvalues();
      double dmax = 0.0;
      for (Eigen::Index i = 0; i < d.size(); ++i)
        dmax = std::max(dmax, std::abs(d[i]));
      const double zeroTol = std::max(dmax, 1.0) * 1e-10;
      int p = 0;
      int q = 0;
      for (Eigen::Index i = 0; i < d.size(); ++i) {
        if (d[i] > zeroTol) ++p;
        else if (d[i] < -zeroTol) ++q;
      }
      cert.positiveSignature = p;
      cert.negativeSignature = q;
      if (p + q == static_cast<int>(rank)) {
        // Nonsingular W-Gram: normalize Phi' = Phi V |d|^{-1/2}, positives
        // first, so Phi'^dagger W Phi' = J = diag(I_p, -I_q) and Psi = Phi' J.
        std::vector<Eigen::Index> orderCols;
        for (Eigen::Index i = d.size() - 1; i >= 0; --i)
          if (d[i] > zeroTol) orderCols.push_back(i);  // descending positives
        for (Eigen::Index i = 0; i < d.size(); ++i)
          if (d[i] < -zeroTol) orderCols.push_back(i);
        Eigen::MatrixXcd VJ(static_cast<Eigen::Index>(rank),
                            static_cast<Eigen::Index>(rank));
        Eigen::VectorXd jdiag(static_cast<Eigen::Index>(rank));
        for (std::size_t cix = 0; cix < orderCols.size(); ++cix) {
          const Eigen::Index src = orderCols[cix];
          VJ.col(static_cast<Eigen::Index>(cix)) =
              ges.eigenvectors().col(src) / std::sqrt(std::abs(d[src]));
          jdiag[static_cast<Eigen::Index>(cix)] = d[src] > 0.0 ? 1.0 : -1.0;
        }
        Phi = (Phi * VJ).eval();
        const Eigen::MatrixXcd Gn = Phi.adjoint() * (W.asDiagonal() * Phi);
        Eigen::MatrixXcd J = Eigen::MatrixXcd::Zero(
            static_cast<Eigen::Index>(rank), static_cast<Eigen::Index>(rank));
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(rank); ++i)
          J(i, i) = cd(jdiag[i], 0.0);
        cert.gramDefect = (Gn - J).norm();
        Psi = Phi * J;
      } else {
        // Neutral directions (e.g. complex-eigenvalue Krein bands): keep the
        // matched biorthogonal frames; the inertia is reported as measured.
        biorthogonalPsi(Psi, cert.gramDefect);
      }
    } else {
      // Non-normal: matched right/left subspaces, Psi^dagger W Phi = I.
      // Signature counts are populated only in the self-adjoint /
      // Krein-normalizable regimes.
      biorthogonalPsi(Psi, cert.gramDefect);
      cert.positiveSignature = 0;
      cert.negativeSignature = 0;
    }

    // Projector measurements through r x r identities (P = Phi Psi^dagger W;
    // B = W^dagger Psi so that P = Phi B^dagger).
    const Eigen::MatrixXcd B =
        W.conjugate().asDiagonal() * Psi;
    const Eigen::MatrixXcd E =
        (Psi.adjoint() * (W.asDiagonal() * Phi)) -
        Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(rank),
                                   static_cast<Eigen::Index>(rank));
    const double pNorm = productFrobenius(
        Phi, Eigen::MatrixXcd::Identity(static_cast<Eigen::Index>(rank),
                                        static_cast<Eigen::Index>(rank)),
        B);
    const double idemDefect = productFrobenius(Phi, E, B);
    cert.projectorResidual = idemDefect / std::max(1.0, pNorm);
    // Two conditioning quantities, reported separately: the gauge-invariant
    // projector norm ||P||_2 (Kato), and the frame condition number used in
    // the non-normal regime — the Riesz condition of the reported matched
    // frames in the |W| metric.
    cert.projectorNorm = productSpectralNorm(Phi, B);
    cert.frameConditionNumber =
        std::max(frameCondition(Phi, W), frameCondition(Psi, W));

    // Localization: inverse participation ratio (IPR) of the projector's
    // diagonal density (gauge- and relabeling-invariant; not an eigenvector
    // read). Rowwise: P = Phi B^dagger, so P_ii = sum_r Phi_ir conj(B_ir).
    std::vector<double> diagAbs(static_cast<std::size_t>(n), 0.0);
    double diagSum = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
      const cd pii = B.row(i).dot(Phi.row(i));
      diagAbs[static_cast<std::size_t>(i)] = std::abs(pii);
      diagSum += diagAbs[static_cast<std::size_t>(i)];
    }
    if (diagSum > 0.0) {
      double diagSq = 0.0;
      for (const double v : diagAbs) {
        const double pi = v / diagSum;
        diagSq += pi * pi;
      }
      cert.localization = diagSq;
      // n_eff = 1 / IPR is the effective number of cells the band lives on:
      // `rank` at maximal concentration, n when perfectly spread.
      const double effectiveSupport = 1.0 / diagSq;
      cert.localizationSupportFraction =
          effectiveSupport / static_cast<double>(n);
      const double room = static_cast<double>(n) - static_cast<double>(rank);
      cert.localizationExcess =
          room > 0.0
              ? std::min(1.0, std::max(0.0,
                                       (effectiveSupport -
                                        static_cast<double>(rank)) / room))
              : 0.0;
    }

    // Certification: isolation from the nearest discarded eigenvalue,
    // localization, residuals, Gram defect, and the gauge-invariant projector
    // conditioning.
    const auto separationOk = [&](double gap) {
      if (std::isnan(gap)) return false;
      if (!std::isfinite(gap)) return true;  // nothing was discarded
      return gap >= cfg_.minRelativeGap * scale &&
             gap >= cfg_.gapDominance * spread;
    };
    const bool residualsOk =
        cert.eigenResidual <= cfg_.residualTolerance &&
        cert.leftResidual <= cfg_.residualTolerance &&
        cert.projectorResidual <= cfg_.residualTolerance;
    const bool localizedOk =
        std::isfinite(cert.localizationExcess) &&
        cert.localizationExcess <= cfg_.maxLocalizationExcess;
    cert.accepted = separationOk(cert.nearestDiscardedSeparation) &&
                    localizedOk && residualsOk &&
                    cert.gramDefect <= cfg_.gramDefectTolerance &&
                    cert.projectorNorm <= cfg_.projectorNormCap;
    if (cert.accepted) {
      cert.certificate = Certificate::certifiedNumerical(
          CertificateDomain::BandWindow, op.regime, cert.eigenResidual,
          cert.projectorNorm, cfg_.residualTolerance);
      cert.certificate.setDenseReferenceError(
          read.solveCertificate.denseReferenceError());
    } else {
      cert.certificate = Certificate::heuristicDiscovery(
          CertificateDomain::BandWindow, op.regime);
    }

    read.fibers.emplace_back(op.cells, std::move(bandEigs), std::move(Phi),
                             std::move(Psi),
                             Eigen::VectorXcd(W), std::move(cert));
  }
}

// ---------------------------------------------------------------------------
// public enumeration / tracking / cache entry points
// ---------------------------------------------------------------------------

ComponentBandRead SpectralFiberTracker::enumerateBands(
    const std::vector<std::uint64_t> &support, int degree) const {
  RestrictedOperator op = assembleRestricted(support, degree);
  ComponentBandRead read;
  read.support = op.support;
  read.degree = degree;
  read.dimension = op.dim();
  read.cellVertices = op.cells;
  read.regime = op.regime;
  if (op.dim() == 0) {
    read.solverPath = "empty";
    read.solveCertificate = Certificate::algebraicallyExact(
        CertificateDomain::Static, op.regime, 0.0, cfg_.residualTolerance);
    return read;
  }
  if (op.pencil) {
    solvePencilBands(op, read);
  } else if (op.positive) {
    if (static_cast<int>(op.dim()) < cfg_.denseCrossover)
      solveDenseSelfAdjoint(op, read);
    else
      solveSparseSelfAdjoint(op, read);
  } else {
    solveDenseGeneral(op, read);
  }
  return read;
}

std::vector<ComponentBandRead> SpectralFiberTracker::enumerateOnComponents(
    const std::vector<ComponentRead> &components) const {
  std::vector<ComponentBandRead> reads;
  reads.reserve(components.size() * cfg_.degrees.size());
  for (const ComponentRead &component : components)
    for (const int degree : cfg_.degrees)
      reads.push_back(enumerateBands(component.support, degree));
  return reads;
}

ComponentBandRead SpectralFiberTracker::enumerateBandsCached(
    cobordism::AnalyticCache &cache, const std::vector<std::uint64_t> &support,
    int degree) const {
  const std::vector<std::uint64_t> key = normalizedSupport(support);
  if (const auto cached = cache.fetch(key, kCacheKind, degree)) {
    return *std::static_pointer_cast<const ComponentBandRead>(cached);
  }
  ComponentBandRead read = enumerateBands(key, degree);
  cache.store(key, kCacheKind, degree,
              std::make_shared<ComponentBandRead>(read),
              read.solveCertificate);
  return read;
}

std::vector<SpectralBandWindow> SpectralFiberTracker::acceptedWindows(
    const std::vector<ComponentBandRead> &reads) {
  std::vector<SpectralBandWindow> windows;
  for (const ComponentBandRead &read : reads) {
    for (const SpectralFiber &fiber : read.fibers) {
      if (!fiber.accepted()) continue;
      SpectralBandWindow window;
      window.degree = fiber.degree();
      window.rank = fiber.rank();
      window.frequencyLower = fiber.certificate().frequencyLower;
      window.frequencyUpper = fiber.certificate().frequencyUpper;
      window.certificate = fiber.certificate();
      windows.push_back(std::move(window));
    }
  }
  return windows;
}

std::vector<FiberMatchRead> SpectralFiberTracker::matchFibers(
    const std::vector<SpectralFiber> &from, const std::vector<SpectralFiber> &to,
    double overlapThreshold) {
  std::vector<FiberMatchRead> matches;
  for (std::size_t i = 0; i < from.size(); ++i) {
    bool found = false;
    FiberMatchRead best;
    for (std::size_t j = 0; j < to.size(); ++j) {
      if (to[j].degree() != from[i].degree()) continue;
      const FiberOverlapRead o = SpectralFiber::overlap(from[i], to[j]);
      if (o.subspaceOverlap <= 0.0 && o.supportOverlap <= 0.0) continue;
      const bool better =
          !found || o.subspaceOverlap > best.overlap.subspaceOverlap ||
          (o.subspaceOverlap == best.overlap.subspaceOverlap &&
           o.supportOverlap > best.overlap.supportOverlap);
      if (better) {
        best.fromIndex = i;
        best.toIndex = j;
        best.degree = from[i].degree();
        best.overlap = o;
        best.ranksEqual = from[i].rank() == to[j].rank();
        best.certifiedContinuation =
            from[i].accepted() && to[j].accepted() && best.ranksEqual &&
            o.subspaceOverlap >= overlapThreshold;
        found = true;
      }
    }
    if (found) matches.push_back(std::move(best));
  }
  return matches;
}

}  // namespace tessera::observables
