// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// Derived U(r) transport between spectral fibers, the Wilson holonomy of a
/// loop of such transports, and the determinant winding of a closed family.
/// Reference: Greensite, "The Confinement Problem in Lattice Gauge Theory",
/// arXiv:hep-lat/0301023

#include "observables/FiberConnection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "cobordism/AnalyticCache.h"
#include "cobordism/ChainComplex.h"
#include "mesh/Fingerprint.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "observables/ColorFiber.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

/// Shorthand for the complex scalar type used throughout this file.
using cd = std::complex<double>;
using cobordism::Certificate;
using cobordism::CertificateDomain;
using cobordism::CertificateRegime;
using cobordism::gradeName;
using cobordism::domainName;
using cobordism::regimeName;
using cobordism::domainFromName;
using cobordism::regimeFromName;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kTwoPi = 2.0 * kPi;

/// Spectral norm ||A||_2 of a small dense matrix (largest singular value).
double spectralNorm(const Eigen::MatrixXcd &a) {
  if (a.size() == 0) return 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(a);
  return svd.singularValues().size() > 0 ? svd.singularValues()[0] : 0.0;
}

/// The signature matrix J = diag(I_p, -I_q) of a band certificate.
/// Krein-normalizable frames store the positive directions first
/// (Phi^dagger W Phi = J), so the diagonal order is structural.
Eigen::MatrixXcd signatureMatrix(int p, int q) {
  Eigen::MatrixXcd j = Eigen::MatrixXcd::Zero(p + q, p + q);
  for (int i = 0; i < p; ++i) j(i, i) = cd(1.0, 0.0);
  for (int i = 0; i < q; ++i) j(p + i, p + i) = cd(-1.0, 0.0);
  return j;
}

/// The band's isolation: its separation from the nearest discarded
/// eigenvalue in the complex plane, not the sort-order neighbour distance.
/// When that separation is unmeasured, falls back to min(lowerGap, upperGap)
/// with NaN propagation, so an unknown side stays unknown rather than being
/// reported as infinitely isolated.
double isolationGap(const SpectralBandCertificate &c) {
  if (!std::isnan(c.nearestDiscardedSeparation))
    return c.nearestDiscardedSeparation;
  if (std::isnan(c.lowerGap) || std::isnan(c.upperGap)) return kNaN;
  return std::min(c.lowerGap, c.upperGap);
}

/// The paired transport regime of two endpoint bands: NonNormal dominates,
/// then HermitianIndefinite, else the shared positive regime.
CertificateRegime pairedRegime(CertificateRegime a, CertificateRegime b) {
  if (a == CertificateRegime::NonNormal || b == CertificateRegime::NonNormal)
    return CertificateRegime::NonNormal;
  // The pencil regime pairs bilinearly (no J-isometry, no inertia); it
  // dominates the Hermitian regimes and is named, never folded into NonNormal.
  if (a == CertificateRegime::ComplexSymmetricPencil ||
      b == CertificateRegime::ComplexSymmetricPencil)
    return CertificateRegime::ComplexSymmetricPencil;
  if (a == CertificateRegime::HermitianIndefinite ||
      b == CertificateRegime::HermitianIndefinite)
    return CertificateRegime::HermitianIndefinite;
  return CertificateRegime::PositiveSemidefinite;
}

/// Principal phase step Arg(to / from) in (-pi, pi] between unit complexes.
double principalStep(cd fromUnit, cd toUnit) {
  return std::arg(toUnit / fromUnit);
}

/// The unit determinant of one accepted transport: det of the emitted
/// factor when present, the raw GL determinant phase otherwise.
cd unitDeterminant(const FiberTransportRead &link, bool *ok) {
  const Eigen::MatrixXcd &m =
      link.unitaryMap.size() > 0 ? link.unitaryMap : link.rawMap;
  if (m.rows() != m.cols() || m.size() == 0) {
    *ok = false;
    return cd(0.0, 0.0);
  }
  const cd det = m.determinant();
  if (!(std::abs(det) > 0.0)) {
    *ok = false;
    return cd(0.0, 0.0);
  }
  *ok = true;
  return det / std::abs(det);
}

/// The matrix a winding/holonomy consumer composes for one link.
const Eigen::MatrixXcd &linkMatrix(const FiberTransportRead &link) {
  return link.unitaryMap.size() > 0 ? link.unitaryMap : link.rawMap;
}

/// Principal square root of a small matrix through its eigendecomposition.
/// Defined only when no eigenvalue sits on the closed negative real axis;
/// `ok` reports that domain check.
Eigen::MatrixXcd principalSqrt(const Eigen::MatrixXcd &k, bool *ok) {
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(k);
  if (es.info() != Eigen::Success) {
    *ok = false;
    return Eigen::MatrixXcd();
  }
  const Eigen::VectorXcd &lambda = es.eigenvalues();
  for (Eigen::Index i = 0; i < lambda.size(); ++i) {
    const cd l = lambda[i];
    const double mag = std::abs(l);
    if (mag == 0.0 ||
        (l.real() <= 0.0 && std::abs(l.imag()) <= 1e-14 * std::max(1.0, mag))) {
      *ok = false;
      return Eigen::MatrixXcd();
    }
  }
  Eigen::VectorXcd roots(lambda.size());
  for (Eigen::Index i = 0; i < lambda.size(); ++i)
    roots[i] = std::sqrt(lambda[i]);  // principal branch
  const Eigen::MatrixXcd v = es.eigenvectors();
  *ok = true;
  return v * roots.asDiagonal() * v.inverse();
}

/// Order-sensitive mix64 chain over the pieces of a cache parameter.  The
/// component key is the order-independent part; the parameter must
/// distinguish direction and loop order, so it chains.
std::int64_t chainedParameter(int degree, int convention,
                              const std::vector<std::uint64_t> &keysInOrder) {
  std::uint64_t h = mesh::Fingerprint::mix64(
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(degree)) << 8) ^
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(convention)));
  for (const std::uint64_t k : keysInOrder)
    h = mesh::Fingerprint::mix64(h ^ k);
  return static_cast<std::int64_t>(h);
}

/// NaN-ignoring running max (std::fmax semantics) for certificate rollups.
double fmaxAccumulate(double acc, double value) { return std::fmax(acc, value); }

// --- Record serialization helpers ------------------------------------------
// The grade/domain/regime name tables and the Certificate sub-record follow
// the shared checkpoint conventions, so `transports` entries stay uniform.

// Schema 2 names the endpoint leaves `*_projector_norm` (they carry ||P||_2)
// and fills `frame_condition_number` with the endpoints' frame conditioning.
// Schema 1 is still readable: its endpoint leaves use `*_condition_number`.
constexpr int kRecordSchemaVersion = 2;
constexpr int kOldestReadableRecordSchema = 1;






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
    throw std::invalid_argument(
        "FiberConnection: unknown certificate grade '" + grade + "'");
  }
  cert.setDenseReferenceError(m.at("dense_reference_error").asDouble());
  return cert;
}

void matrixToRecord(Record::Map &m, const std::string &name,
                    const Eigen::MatrixXcd &matrix) {
  m[name + "_rows"] = Record(static_cast<std::int64_t>(matrix.rows()));
  m[name + "_cols"] = Record(static_cast<std::int64_t>(matrix.cols()));
  std::vector<cd> flat(static_cast<std::size_t>(matrix.size()));
  for (Eigen::Index r = 0; r < matrix.rows(); ++r)
    for (Eigen::Index c = 0; c < matrix.cols(); ++c)
      flat[static_cast<std::size_t>(r * matrix.cols() + c)] = matrix(r, c);
  Record::splitComplex(m, name, flat);
}

Eigen::MatrixXcd matrixFromRecord(const Record::Map &m,
                                  const std::string &name) {
  const auto rows = m.at(name + "_rows").asInt();
  const auto cols = m.at(name + "_cols").asInt();
  const auto &re = m.at(name + "_re").asList();
  const auto &im = m.at(name + "_im").asList();
  if (re.size() != im.size() ||
      re.size() != static_cast<std::size_t>(rows * cols))
    throw std::invalid_argument(
        "FiberConnection: matrix record payload size mismatch");
  Eigen::MatrixXcd matrix(rows, cols);
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t c = 0; c < cols; ++c) {
      const auto i = static_cast<std::size_t>(r * cols + c);
      matrix(r, c) = cd(re[i].asDouble(), im[i].asDouble());
    }
  return matrix;
}

void requireSchema(const Record::Map &m, const char *type) {
  const auto version = m.find("schema_version");
  if (version == m.end() ||
      version->second.asInt() <
          static_cast<std::int64_t>(kOldestReadableRecordSchema) ||
      version->second.asInt() >
          static_cast<std::int64_t>(kRecordSchemaVersion))
    throw std::invalid_argument(
        "FiberConnection: unknown schema_version (reader rejects unknown "
        "checkpoint schemas)");
  const auto rt = m.find("record_type");
  if (rt == m.end() || rt->second.asString() != type)
    throw std::invalid_argument(std::string("FiberConnection: expected a '") +
                                type + "' record");
}

}  // namespace

// ---------------------------------------------------------------------------
// FiberTransportRead
// ---------------------------------------------------------------------------

std::string FiberTransportRead::describe() const {
  std::ostringstream out;
  out << "FiberTransport[deg " << degree << ", rank " << rank << ", regime ";
  switch (regime) {
    case CertificateRegime::PositiveSemidefinite: out << "positive"; break;
    case CertificateRegime::HermitianIndefinite: out << "krein"; break;
    case CertificateRegime::NonNormal: out << "non-normal"; break;
    case CertificateRegime::ComplexSymmetricPencil: out << "complex-symmetric-pencil"; break;
  }
  out << "] numerical rank " << numericalRank << ", leakage " << leakage
      << ", overlap cond " << overlapConditionNumber << ", gaps ("
      << toGap << ", " << fromGap << "), signatures (" << toPositiveSignature
      << "," << toNegativeSignature << ")/(" << fromPositiveSignature << ","
      << fromNegativeSignature << ")";
  if (accepted) {
    out << (unitaryMap.size() > 0 ? "; reduced (polar residual "
                                  : "; certified GL transport (polar residual ")
        << polarResidual << ")";
    if (projectiveOnly) out << " [projective-only]";
  } else {
    out << "; REJECTED: " << rejectionReason;
  }
  return out.str();
}

Record FiberTransportRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("fiber_transport");
  m["to_key"] = Record(static_cast<std::int64_t>(toKey));
  m["from_key"] = Record(static_cast<std::int64_t>(fromKey));
  m["degree"] = Record(degree);
  m["rank"] = Record(rank);
  matrixToRecord(m, "raw_map", rawMap);
  Record::List sigma;
  sigma.reserve(singularValues.size());
  for (const double s : singularValues) sigma.emplace_back(s);
  m["singular_values"] = Record(std::move(sigma));
  m["numerical_rank"] = Record(numericalRank);
  m["leakage"] = Record(leakage);
  m["overlap_condition_number"] = Record(overlapConditionNumber);
  m["to_gap"] = Record(toGap);
  m["from_gap"] = Record(fromGap);
  m["to_positive_signature"] = Record(toPositiveSignature);
  m["to_negative_signature"] = Record(toNegativeSignature);
  m["from_positive_signature"] = Record(fromPositiveSignature);
  m["from_negative_signature"] = Record(fromNegativeSignature);
  m["to_projector_norm"] = Record(toProjectorNorm);
  m["from_projector_norm"] = Record(fromProjectorNorm);
  m["frame_condition_number"] = Record(frameConditionNumber);
  m["regime"] = Record(regimeName(regime));
  matrixToRecord(m, "unitary_map", unitaryMap);
  Record::splitComplex(m, "determinant_phase", determinantPhase);
  m["polar_residual"] = Record(polarResidual);
  m["determinant_residual"] = Record(determinantResidual);
  m["projective_only"] = Record(projectiveOnly);
  m["accepted"] = Record(accepted);
  m["rejection_reason"] = Record(rejectionReason);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

FiberTransportRead FiberTransportRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "fiber_transport");
  FiberTransportRead read;
  read.toKey = static_cast<std::uint64_t>(m.at("to_key").asInt());
  read.fromKey = static_cast<std::uint64_t>(m.at("from_key").asInt());
  read.degree = static_cast<int>(m.at("degree").asInt());
  read.rank = static_cast<int>(m.at("rank").asInt());
  read.rawMap = matrixFromRecord(m, "raw_map");
  for (const Record &s : m.at("singular_values").asList())
    read.singularValues.push_back(s.asDouble());
  read.numericalRank = static_cast<int>(m.at("numerical_rank").asInt());
  read.leakage = m.at("leakage").asDouble();
  read.overlapConditionNumber = m.at("overlap_condition_number").asDouble();
  read.toGap = m.at("to_gap").asDouble();
  read.fromGap = m.at("from_gap").asDouble();
  read.toPositiveSignature =
      static_cast<int>(m.at("to_positive_signature").asInt());
  read.toNegativeSignature =
      static_cast<int>(m.at("to_negative_signature").asInt());
  read.fromPositiveSignature =
      static_cast<int>(m.at("from_positive_signature").asInt());
  read.fromNegativeSignature =
      static_cast<int>(m.at("from_negative_signature").asInt());
  read.toProjectorNorm = m.count("to_projector_norm")
                             ? m.at("to_projector_norm").asDouble()
                             : m.at("to_condition_number").asDouble();
  read.fromProjectorNorm = m.count("from_projector_norm")
                               ? m.at("from_projector_norm").asDouble()
                               : m.at("from_condition_number").asDouble();
  read.frameConditionNumber = m.at("frame_condition_number").asDouble();
  read.regime = regimeFromName(m.at("regime").asString());
  read.unitaryMap = matrixFromRecord(m, "unitary_map");
  read.determinantPhase = cd(m.at("determinant_phase_re").asDouble(),
                             m.at("determinant_phase_im").asDouble());
  read.polarResidual = m.at("polar_residual").asDouble();
  read.determinantResidual = m.at("determinant_residual").asDouble();
  read.projectiveOnly = m.at("projective_only").asBool();
  read.accepted = m.at("accepted").asBool();
  read.rejectionReason = m.at("rejection_reason").asString();
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

Record FundamentalLiftRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("fundamental_lift");
  m["rank"] = Record(rank);
  m["base_branch"] = Record(baseBranch);
  matrixToRecord(m, "lift", lift);
  Record::splitComplex(m, "lift_trace", liftTrace);
  m["center_sector"] = Record(centerSector);
  m["accumulated_determinant_phase"] = Record(accumulatedDeterminantPhase);
  m["max_determinant_phase_step"] = Record(maxDeterminantPhaseStep);
  m["det_residual"] = Record(detResidual);
  m["valid"] = Record(valid);
  m["invalid_reason"] = Record(invalidReason);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

FundamentalLiftRead FundamentalLiftRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "fundamental_lift");
  FundamentalLiftRead read;
  read.rank = static_cast<int>(m.at("rank").asInt());
  read.baseBranch = static_cast<int>(m.at("base_branch").asInt());
  read.lift = matrixFromRecord(m, "lift");
  read.liftTrace = cd(m.at("lift_trace_re").asDouble(),
                      m.at("lift_trace_im").asDouble());
  read.centerSector = static_cast<int>(m.at("center_sector").asInt());
  read.accumulatedDeterminantPhase =
      m.at("accumulated_determinant_phase").asDouble();
  read.maxDeterminantPhaseStep =
      m.at("max_determinant_phase_step").asDouble();
  read.detResidual = m.at("det_residual").asDouble();
  read.valid = m.at("valid").asBool();
  read.invalidReason = m.at("invalid_reason").asString();
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

Record DeterminantWindingRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("determinant_winding");
  // An unknown winding serializes as unknown, not as zero.
  m["winding_known"] = Record(winding.has_value());
  m["winding"] = winding.has_value() ? Record(*winding) : Record();
  m["winding_closure"] = Record(windingClosure);
  m["winding_reference_id"] = Record(windingReferenceId);
  m["accumulated_phase"] = Record(accumulatedPhase);
  m["max_phase_step"] = Record(maxPhaseStep);
  m["phase_step_margin"] = Record(phaseStepMargin);
  m["closure_defect"] = Record(closureDefect);
  m["invalidation_reason"] = Record(invalidationReason);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

DeterminantWindingRead DeterminantWindingRead::fromRecord(
    const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "determinant_winding");
  DeterminantWindingRead read;
  if (m.at("winding_known").asBool())
    read.winding = static_cast<int>(m.at("winding").asInt());
  read.windingClosure = m.at("winding_closure").asString();
  read.windingReferenceId = m.at("winding_reference_id").asString();
  read.accumulatedPhase = m.at("accumulated_phase").asDouble();
  read.maxPhaseStep = m.at("max_phase_step").asDouble();
  read.phaseStepMargin = m.at("phase_step_margin").asDouble();
  read.closureDefect = m.at("closure_defect").asDouble();
  read.invalidationReason = m.at("invalidation_reason").asString();
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

// ---------------------------------------------------------------------------
// construction / keys
// ---------------------------------------------------------------------------

FiberConnection::FiberConnection(FiberConnectionConfig cfg) : cfg_(cfg) {}

std::uint64_t FiberConnection::fiberKey(const SpectralFiber &fiber) {
  std::set<std::uint64_t> ids;
  for (const auto &cell : fiber.cellVertices())
    ids.insert(cell.begin(), cell.end());
  return mesh::Fingerprint::fingerprintOf(ids);
}

std::uint64_t FiberConnection::bandFingerprint(const SpectralFiber &fiber) {
  std::uint64_t hash = mesh::Fingerprint::mix64(
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(fiber.degree()))
       << 32) ^
      static_cast<std::uint64_t>(fiber.rank()));
  for (const auto &eigenvalue : fiber.eigenvalues()) {
    std::uint64_t bits = 0;
    const double real = eigenvalue.real();
    std::memcpy(&bits, &real, sizeof(bits));
    hash = mesh::Fingerprint::mix64(hash ^ bits);
    const double imaginary = eigenvalue.imag();
    std::memcpy(&bits, &imaginary, sizeof(bits));
    hash = mesh::Fingerprint::mix64(hash ^ bits);
  }
  return hash;
}

std::vector<std::uint64_t> FiberConnection::unionVertexIds(
    const std::vector<const SpectralFiber *> &fibers) {
  std::set<std::uint64_t> ids;
  for (const SpectralFiber *fiber : fibers)
    for (const auto &cell : fiber->cellVertices())
      ids.insert(cell.begin(), cell.end());
  return {ids.begin(), ids.end()};
}

// ---------------------------------------------------------------------------
// chain-transfer sources (wrappers over existing machinery)
// ---------------------------------------------------------------------------

Eigen::MatrixXcd FiberConnection::chainTransfer(
    const std::shared_ptr<Spacetime> &st, int degree,
    const std::vector<std::vector<std::uint64_t>> &toCells,
    const std::vector<std::vector<std::uint64_t>> &fromVertexTuples,
    cobordism::HodgeLaplacian::WeightConvention weights) {
  if (st == nullptr)
    throw std::invalid_argument("FiberConnection::chainTransfer: null spacetime");
  if (degree < 0)
    throw std::invalid_argument("FiberConnection::chainTransfer: negative degree");

  // Canonical whole-complex cell order: sorted vertex ids at degree 0, the
  // ChainComplex column order at degree >= 1 (the laplacian(k) alignment).
  std::vector<std::vector<std::uint64_t>> cells;
  if (degree == 0) {
    std::vector<std::uint64_t> ids;
    for (const auto &v : st->getVertexList()->toVector()) {
      if (v == nullptr) continue;
      ids.push_back(v->getId());
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    cells.reserve(ids.size());
    for (const std::uint64_t id : ids) cells.push_back({id});
  } else {
    const cobordism::ChainComplex cc =
        cobordism::ChainComplex::fromSpacetime(*st);
    cells = cc.kSimplexVertices(degree);
  }
  const std::size_t n = cells.size();
  if (n == 0)
    throw std::invalid_argument(
        "FiberConnection::chainTransfer: no cells at this degree");

  std::map<std::vector<std::uint64_t>, std::size_t> indexOf;
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::uint64_t> key = cells[i];
    std::sort(key.begin(), key.end());
    indexOf.emplace(std::move(key), i);
  }
  const auto lookup =
      [&](const std::vector<std::uint64_t> &cell) -> std::size_t {
    std::vector<std::uint64_t> key = cell;
    std::sort(key.begin(), key.end());
    const auto it = indexOf.find(key);
    if (it == indexOf.end()) {
      std::ostringstream out;
      out << "FiberConnection::chainTransfer: unknown cell (";
      for (std::size_t i = 0; i < cell.size(); ++i)
        out << (i ? "," : "") << cell[i];
      out << ") at degree " << degree;
      throw std::invalid_argument(out.str());
    }
    return it->second;
  };

  const cobordism::HodgeLaplacian hodge(st, weights);
  // Degree 0 reads the U(1) connection Laplacian D - A, not the Hodge L_0:
  // the degree-zero entry is the oriented U(1) link value -l^2 e^{i phase},
  // matching the Wilson-loop convention this transport is compared against.
  // L_0 = d_1 W_1^-1 d_1^T carries no link phase; its off-diagonal is
  // -1/W_1(e).
  const std::vector<cd> flat =
      degree == 0 ? hodge.connectionLaplacian() : hodge.laplacian(degree);
  if (flat.size() != n * n)
    throw std::invalid_argument(
        "FiberConnection::chainTransfer: operator/cell count mismatch");

  Eigen::MatrixXcd block(static_cast<Eigen::Index>(toCells.size()),
                         static_cast<Eigen::Index>(fromVertexTuples.size()));
  std::vector<std::size_t> rows;
  rows.reserve(toCells.size());
  for (const auto &cell : toCells) rows.push_back(lookup(cell));
  std::vector<std::size_t> cols;
  cols.reserve(fromVertexTuples.size());
  for (const auto &cell : fromVertexTuples) cols.push_back(lookup(cell));
  for (std::size_t r = 0; r < rows.size(); ++r)
    for (std::size_t c = 0; c < cols.size(); ++c)
      block(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
          flat[rows[r] * n + cols[c]];
  return block;
}

Eigen::MatrixXcd FiberConnection::responseTransfer(
    const cobordism::RecursiveQuotient::ResponseNetworkRead &network,
    int toComponent, int fromComponent) {
  const int count = static_cast<int>(network.stalkDimensions.size());
  if (toComponent < 0 || toComponent >= count || fromComponent < 0 ||
      fromComponent >= count)
    throw std::out_of_range(
        "FiberConnection::responseTransfer: component index out of range");
  const int rows = network.stalkDimensions[static_cast<std::size_t>(toComponent)];
  const int cols =
      network.stalkDimensions[static_cast<std::size_t>(fromComponent)];
  Eigen::MatrixXcd block = Eigen::MatrixXcd::Zero(rows, cols);
  for (const auto &edge : network.edges) {
    if (edge.from != toComponent || edge.to != fromComponent) continue;
    if (edge.block.size() != static_cast<std::size_t>(rows) *
                                 static_cast<std::size_t>(cols))
      throw std::invalid_argument(
          "FiberConnection::responseTransfer: malformed edge block");
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        block(r, c) = edge.block[static_cast<std::size_t>(r) *
                                     static_cast<std::size_t>(cols) +
                                 static_cast<std::size_t>(c)];
    return block;
  }
  return block;  // no such edge: the zero transfer of the right shape
}

// ---------------------------------------------------------------------------
// the derived transport
// ---------------------------------------------------------------------------

FiberTransportRead FiberConnection::transport(
    const SpectralFiber &to, const SpectralFiber &from,
    const Eigen::MatrixXcd &transfer) const {
  return deriveTransport(to, from, transfer);
}

FiberTransportRead FiberConnection::transportReverse(
    const SpectralFiber &to, const SpectralFiber &from,
    const Eigen::MatrixXcd &transfer) const {
  // W-adjoint reverse block T_BA = W_B^{-1} T_AB^dagger W_A: the exact
  // reverse chain transfer whenever W L is (anti)symmetric, that is, in the
  // W-self-adjoint regimes.
  const Eigen::VectorXcd wTo = to.weightDiagonal();
  const Eigen::VectorXcd wFrom = from.weightDiagonal();
  if (transfer.rows() != wTo.size() || transfer.cols() != wFrom.size())
    throw std::invalid_argument(
        "FiberConnection::transportReverse: transfer shape mismatch");
  Eigen::VectorXcd invWFrom(wFrom.size());
  for (Eigen::Index i = 0; i < wFrom.size(); ++i) {
    if (!(std::abs(wFrom[i]) > 0.0))
      throw std::invalid_argument(
          "FiberConnection::transportReverse: singular source weight");
    invWFrom[i] = cd(1.0, 0.0) / wFrom[i];
  }
  const Eigen::MatrixXcd reversed =
      invWFrom.asDiagonal() * transfer.adjoint() * wTo.asDiagonal();
  return deriveTransport(from, to, reversed);
}

FiberTransportRead FiberConnection::deriveTransport(
    const SpectralFiber &to, const SpectralFiber &from,
    const Eigen::MatrixXcd &transfer) const {
  const SpectralBandCertificate &certTo = to.certificate();
  const SpectralBandCertificate &certFrom = from.certificate();
  if (certTo.degree != certFrom.degree)
    throw std::invalid_argument(
        "FiberConnection::transport: the two bands live at different degrees");
  const auto rowsNeeded = static_cast<Eigen::Index>(to.cellVertices().size());
  const auto colsNeeded = static_cast<Eigen::Index>(from.cellVertices().size());
  if (transfer.rows() != rowsNeeded || transfer.cols() != colsNeeded)
    throw std::invalid_argument(
        "FiberConnection::transport: transfer shape mismatch (rows = "
        "destination cells, cols = source cells)");

  FiberTransportRead read;
  read.toKey = fiberKey(to);
  read.fromKey = fiberKey(from);
  read.degree = certTo.degree;
  read.rank = static_cast<int>(to.rank());
  read.toGap = isolationGap(certTo);
  read.fromGap = isolationGap(certFrom);
  read.toPositiveSignature = certTo.positiveSignature;
  read.toNegativeSignature = certTo.negativeSignature;
  read.fromPositiveSignature = certFrom.positiveSignature;
  read.fromNegativeSignature = certFrom.negativeSignature;
  read.toProjectorNorm = certTo.projectorNorm;
  read.fromProjectorNorm = certFrom.projectorNorm;
  // The frame condition number is the endpoints' own frame conditioning, a
  // different quantity from the projector norms above.
  read.frameConditionNumber =
      std::fmax(certTo.frameConditionNumber, certFrom.frameConditionNumber);
  read.regime = pairedRegime(certTo.certificate.regime(),
                             certFrom.certificate.regime());
  // The biorthogonal (left-frame) path serves both the non-normal regime and
  // the complex-symmetric pencil, whose pairing is bilinear; the certificate
  // keeps the pencil's own name.
  const bool nonNormal = read.regime == CertificateRegime::NonNormal ||
                         read.regime == CertificateRegime::ComplexSymmetricPencil;

  // Overlap: M = Phi_A^dagger W_A T Phi_B in the self-adjoint regimes; on the
  // biorthogonal path the transpose dual, M = Phi~_A^T T Phi_B, which is
  // Psi_A^dagger W_A T Phi_B off the pencil path and pairs by the bilinear
  // left frame itself (never its conjugate) on it.
  const Eigen::MatrixXcd m =
      nonNormal ? Eigen::MatrixXcd(to.dualFrame().transpose() * transfer *
                                   from.rightFrame())
                : Eigen::MatrixXcd(to.rightFrame().adjoint() *
                                   to.weightDiagonal().asDiagonal() *
                                   transfer * from.rightFrame());
  read.rawMap = m;

  // Pre-normalization diagnostics: rank, singular values, conditioning.
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(m, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
  const Eigen::VectorXd &sigma = svd.singularValues();
  read.singularValues.assign(sigma.data(), sigma.data() + sigma.size());
  const double sigmaMax = sigma.size() > 0 ? sigma[0] : 0.0;
  const double sigmaMin = sigma.size() > 0 ? sigma[sigma.size() - 1] : 0.0;
  int numericalRank = 0;
  for (Eigen::Index i = 0; i < sigma.size(); ++i)
    if (sigma[i] > cfg_.rankTolerance * sigmaMax) ++numericalRank;
  if (sigmaMax == 0.0) numericalRank = 0;
  read.numericalRank = numericalRank;
  read.overlapConditionNumber =
      sigma.size() == 0 ? 0.0 : (sigmaMin > 0.0 ? sigmaMax / sigmaMin : kInf);

  // Regime-appropriate leakage.  The self-adjoint regimes use the Krein form
  // ||M^dagger J_A M - J_B||, with J = I reproducing ||M^dagger M - I||; the
  // biorthogonal path reports the Euclidean unitarity defect without gating
  // the GL transport on it.
  const int rankTo = static_cast<int>(to.rank());
  const int rankFrom = static_cast<int>(from.rank());
  const bool ranksMatch = rankTo == rankFrom;
  const bool toSignatureComplete =
      certTo.positiveSignature + certTo.negativeSignature == rankTo;
  const bool fromSignatureComplete =
      certFrom.positiveSignature + certFrom.negativeSignature == rankFrom;
  Eigen::MatrixXcd jTo;
  Eigen::MatrixXcd jFrom;
  if (!nonNormal && toSignatureComplete && fromSignatureComplete) {
    jTo = signatureMatrix(certTo.positiveSignature, certTo.negativeSignature);
    jFrom =
        signatureMatrix(certFrom.positiveSignature, certFrom.negativeSignature);
    read.leakage = spectralNorm(m.adjoint() * jTo * m - jFrom);
  } else {
    read.leakage = spectralNorm(
        m.adjoint() * m -
        Eigen::MatrixXcd::Identity(m.cols(), m.cols()));
  }

  // Threshold gates: reject before any polar or pseudo-unitary reduction.
  const auto reject = [&](const std::string &reason) {
    read.accepted = false;
    read.rejectionReason = reason;
    read.certificate =
        Certificate::heuristicDiscovery(CertificateDomain::BandWindow,
                                        read.regime);
    if (read.numericalRank > 0 && m.rows() == m.cols() && m.size() > 0) {
      const cd det = m.determinant();
      if (std::abs(det) > 0.0) read.determinantPhase = det / std::abs(det);
    }
    return read;
  };

  if (!ranksMatch) return reject("band rank mismatch");
  if (cfg_.requireCertifiedFibers && !(certTo.accepted && certFrom.accepted))
    return reject("uncertified endpoint band (gap closed or residuals failed)");
  if (cfg_.minEndpointGap > 0.0 &&
      !(read.toGap >= cfg_.minEndpointGap &&
        read.fromGap >= cfg_.minEndpointGap))
    return reject("endpoint band gap below the configured floor");
  if (!(certTo.projectorNorm <= cfg_.conditionNumberCap) ||
      !(certFrom.projectorNorm <= cfg_.conditionNumberCap))
    return reject("endpoint projector conditioning above the cap");
  if (read.numericalRank < read.rank) return reject("rank-deficient overlap");
  if (!(read.overlapConditionNumber <= cfg_.conditionNumberCap))
    return reject("ill-conditioned overlap");

  if (nonNormal) {
    // Certified GL(r, C) transport: rawMap is the observable; no U(r) or
    // SU(3) value is emitted outside the positive-metric domain.
    read.accepted = true;
    const cd det = m.determinant();
    if (std::abs(det) > 0.0) read.determinantPhase = det / std::abs(det);
    const double gramResidual =
        fmaxAccumulate(certTo.gramDefect, certFrom.gramDefect);
    read.certificate = Certificate::certifiedNumerical(
        CertificateDomain::BandWindow, read.regime, gramResidual,
        read.overlapConditionNumber, cfg_.certificateTolerance);
    return read;
  }

  if (!toSignatureComplete || !fromSignatureComplete)
    return reject("neutral Krein directions (singular band Gram)");
  if (certTo.positiveSignature != certFrom.positiveSignature ||
      certTo.negativeSignature != certFrom.negativeSignature)
    return reject("Krein signature mismatch between the endpoint bands");
  if (!(read.leakage <= cfg_.leakageTolerance))
    return reject("leaking transport (isometry defect above tolerance)");

  const bool positivePair =
      read.regime == CertificateRegime::PositiveSemidefinite;
  if (positivePair) {
    // Polar factor V = M (M^dagger M)^{-1/2} = U V^dagger from the singular
    // value decomposition (SVD); exactly unitary-equivariant under local
    // frame changes.
    read.unitaryMap = svd.matrixU() * svd.matrixV().adjoint();
    read.polarResidual = spectralNorm(
        read.unitaryMap.adjoint() * read.unitaryMap -
        Eigen::MatrixXcd::Identity(read.rank, read.rank));
  } else {
    // Pseudo-unitary reduction on matching signatures:
    // V = M K^{-1/2}, K = J_B M^dagger J_A M (J_B-self-adjoint; principal
    // square root well defined for the near-J-isometric maps the leakage
    // gate admits), giving V^dagger J_A V = J_B.
    const Eigen::MatrixXcd k = jFrom * m.adjoint() * jTo * m;
    bool ok = false;
    const Eigen::MatrixXcd kSqrt = principalSqrt(k, &ok);
    if (!ok)
      return reject("pseudo-unitary square root undefined "
                    "(spectrum met the negative real axis)");
    read.unitaryMap = m * kSqrt.inverse();
    read.polarResidual =
        spectralNorm(read.unitaryMap.adjoint() * jTo * read.unitaryMap - jFrom);
  }

  const cd det = read.unitaryMap.determinant();
  read.determinantPhase = det;
  read.determinantResidual = std::abs(std::abs(det) - 1.0);
  read.projectiveOnly =
      read.rank == 3 && read.determinantResidual > cfg_.certificateTolerance;
  read.accepted = true;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, read.regime, read.polarResidual,
      read.overlapConditionNumber, cfg_.certificateTolerance);
  return read;
}

FiberTransportRead FiberConnection::transportOnSpacetime(
    const std::shared_ptr<Spacetime> &st, const SpectralFiber &to,
    const SpectralFiber &from,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  if (to.degree() != from.degree())
    throw std::invalid_argument(
        "FiberConnection::transportOnSpacetime: degree mismatch");
  const Eigen::MatrixXcd transfer = chainTransfer(
      st, to.degree(), to.cellVertices(), from.cellVertices(), weights);
  return deriveTransport(to, from, transfer);
}

FiberTransportRead FiberConnection::transportOnSpacetimeCached(
    cobordism::AnalyticCache &cache, const std::shared_ptr<Spacetime> &st,
    const SpectralFiber &to, const SpectralFiber &from,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  const std::vector<std::uint64_t> ids = unionVertexIds({&to, &from});
  // The band fingerprints join the component keys here: every band of one
  // component restricts to the same cells, so component keys alone collide
  // across a component pair's bands and the cache would serve one band's
  // transport for all of them.
  const std::int64_t parameter =
      chainedParameter(to.degree(), static_cast<int>(weights),
                       {fiberKey(to), bandFingerprint(to), fiberKey(from),
                        bandFingerprint(from)});
  if (const auto payload = cache.fetch(ids, kTransportCacheKind, parameter))
    return *std::static_pointer_cast<FiberTransportRead>(payload);
  FiberTransportRead read = transportOnSpacetime(st, to, from, weights);
  cache.store(ids, kTransportCacheKind, parameter,
              std::make_shared<FiberTransportRead>(read), read.certificate);
  return read;
}

// ---------------------------------------------------------------------------
// Wilson observables
// ---------------------------------------------------------------------------

WilsonHolonomyRead FiberConnection::holonomy(
    const std::vector<FiberTransportRead> &links) const {
  if (links.empty())
    throw std::invalid_argument("FiberConnection::holonomy: empty chain");
  const int rank = links.front().rank;
  bool unitary = true;
  for (std::size_t i = 0; i < links.size(); ++i) {
    if (!links[i].accepted)
      throw std::invalid_argument(
          "FiberConnection::holonomy: only ACCEPTED maps are multiplied "
          "(link " + std::to_string(i) + " was rejected: " +
          links[i].rejectionReason + ")");
    if (links[i].rank != rank)
      throw std::invalid_argument(
          "FiberConnection::holonomy: rank mismatch along the chain");
    if (links[i].unitaryMap.size() == 0) unitary = false;
  }

  WilsonHolonomyRead read;
  read.rank = rank;
  read.loopLength = links.size();
  read.baseKey = links.front().toKey;
  read.unitary = unitary;

  bool closed = true;
  for (std::size_t i = 0; i + 1 < links.size(); ++i)
    if (links[i].fromKey != links[i + 1].toKey) closed = false;
  if (links.back().fromKey != links.front().toKey) closed = false;
  read.closed = closed;

  Eigen::MatrixXcd h = Eigen::MatrixXcd::Identity(rank, rank);
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  double residual = kNaN;
  double conditioning = kNaN;
  // Polar normalization must not conceal a bad fiber assignment, so the
  // pre-normalization diagnostics of every constituent link travel with the
  // loop: worst leakage, worst endpoint isolation, worst frame conditioning,
  // and the singular-value and rank evidence.
  double maxLeakage = kNaN;
  double minEndpointGap = kNaN;
  double maxFrameCondition = kNaN;
  double minSingularValue = kNaN;
  int minNumericalRank = rank;
  for (const FiberTransportRead &link : links) {
    // Never a mixture: the unitary product uses every emitted factor, the
    // GL product the raw maps throughout.
    h = h * (unitary ? link.unitaryMap : link.rawMap);
    regime = pairedRegime(regime, link.regime);
    residual = fmaxAccumulate(residual, unitary ? link.polarResidual
                                                : link.certificate.residual());
    conditioning = fmaxAccumulate(conditioning, link.certificate.conditioning());
    maxLeakage = fmaxAccumulate(maxLeakage, link.leakage);
    maxFrameCondition =
        fmaxAccumulate(maxFrameCondition, link.frameConditionNumber);
    const double endpointGap = std::fmin(link.toGap, link.fromGap);
    minEndpointGap = std::isnan(minEndpointGap)
                         ? endpointGap
                         : std::fmin(minEndpointGap, endpointGap);
    if (!link.singularValues.empty()) {
      const double smallest = *std::min_element(link.singularValues.begin(),
                                                link.singularValues.end());
      minSingularValue = std::isnan(minSingularValue)
                             ? smallest
                             : std::fmin(minSingularValue, smallest);
    }
    minNumericalRank = std::min(minNumericalRank, link.numericalRank);
  }
  read.maxLeakage = maxLeakage;
  read.minEndpointGap = minEndpointGap;
  read.maxFrameConditionNumber = maxFrameCondition;
  read.minSingularValue = minSingularValue;
  read.minNumericalRank = minNumericalRank;
  read.holonomy = h;
  read.normalizedTrace = h.trace() / static_cast<double>(rank);
  read.determinant = h.determinant();
  const double traceAbs = std::abs(h.trace());
  read.adjointTrace = cd(traceAbs * traceAbs - 1.0, 0.0);
  // Metric-appropriate isometry defect: ||H^dagger H - I|| in the positive
  // regime, the base-point J-isometry defect ||H^dagger J H - J|| on a Krein
  // loop, where the pseudo-unitary product is exactly J-unitary.
  Eigen::MatrixXcd jBase = Eigen::MatrixXcd::Identity(rank, rank);
  if (regime == CertificateRegime::HermitianIndefinite &&
      links.front().toPositiveSignature + links.front().toNegativeSignature ==
          rank)
    jBase = signatureMatrix(links.front().toPositiveSignature,
                            links.front().toNegativeSignature);
  read.unitarityResidual = spectralNorm(h.adjoint() * jBase * h - jBase);
  if (rank == 3) read.adjointMatrix = adjointRepresentation(h);
  if (unitary) residual = fmaxAccumulate(residual, read.unitarityResidual);
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime, residual, conditioning,
      cfg_.certificateTolerance);
  return read;
}

WilsonHolonomyRead FiberConnection::holonomyOnSpacetime(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<SpectralFiber> &fibers,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  if (fibers.size() < 2)
    throw std::invalid_argument(
        "FiberConnection::holonomyOnSpacetime: need at least two fibers");
  std::vector<FiberTransportRead> links;
  links.reserve(fibers.size());
  for (std::size_t i = 0; i < fibers.size(); ++i)
    links.push_back(transportOnSpacetime(
        st, fibers[i], fibers[(i + 1) % fibers.size()], weights));
  return holonomy(links);
}

WilsonHolonomyRead FiberConnection::holonomyOnSpacetimeCached(
    cobordism::AnalyticCache &cache, const std::shared_ptr<Spacetime> &st,
    const std::vector<SpectralFiber> &fibers,
    cobordism::HodgeLaplacian::WeightConvention weights) const {
  if (fibers.size() < 2)
    throw std::invalid_argument(
        "FiberConnection::holonomyOnSpacetimeCached: need at least two fibers");
  std::vector<const SpectralFiber *> pointers;
  std::vector<std::uint64_t> orderedKeys;
  pointers.reserve(fibers.size());
  orderedKeys.reserve(fibers.size());
  for (const SpectralFiber &fiber : fibers) {
    pointers.push_back(&fiber);
    // Both the component key and the band fingerprint, for the same reason
    // `transportOnSpacetimeCached` needs both: two loops over the same
    // components through different bands are different loops.
    orderedKeys.push_back(fiberKey(fiber));
    orderedKeys.push_back(bandFingerprint(fiber));
  }
  const std::vector<std::uint64_t> ids = unionVertexIds(pointers);
  const std::int64_t parameter = chainedParameter(
      fibers.front().degree(), static_cast<int>(weights), orderedKeys);
  if (const auto payload = cache.fetch(ids, kHolonomyCacheKind, parameter))
    return *std::static_pointer_cast<WilsonHolonomyRead>(payload);
  std::vector<FiberTransportRead> links;
  links.reserve(fibers.size());
  for (std::size_t i = 0; i < fibers.size(); ++i)
    links.push_back(transportOnSpacetimeCached(
        cache, st, fibers[i], fibers[(i + 1) % fibers.size()], weights));
  WilsonHolonomyRead read = holonomy(links);
  cache.store(ids, kHolonomyCacheKind, parameter,
              std::make_shared<WilsonHolonomyRead>(read), read.certificate);
  return read;
}

// ---------------------------------------------------------------------------
// rank-three center structure
// ---------------------------------------------------------------------------

Eigen::MatrixXcd FiberConnection::projectiveRepresentative(
    const Eigen::MatrixXcd &unitary, const AnchorGate &gate) {
  // Every colour-specific kernel is gated on the triangle-anchor
  // certificate: a rank-three band that was never anchored gets no 3x3
  // colour determinant.
  if (!gate.accepted)
    throw std::invalid_argument(
        "FiberConnection::projectiveRepresentative: " + gate.refusalReason);
  if (unitary.rows() != 3 || unitary.cols() != 3)
    throw std::invalid_argument(
        "FiberConnection::projectiveRepresentative: expected a 3x3 matrix");
  const cd det = unitary.determinant();
  if (!(std::abs(det) > 0.0))
    throw std::invalid_argument(
        "FiberConnection::projectiveRepresentative: singular matrix");
  // Principal cube root of the determinant phase; the modulus is left to
  // the caller's unitarity certificate.
  const cd root = std::exp(cd(0.0, std::arg(det) / 3.0));
  return unitary / root;
}

Eigen::MatrixXcd FiberConnection::adjointRepresentation(
    const Eigen::MatrixXcd &unitary) {
  if (unitary.rows() != 3 || unitary.cols() != 3)
    throw std::invalid_argument(
        "FiberConnection::adjointRepresentation: expected a 3x3 matrix");
  // vec(U M U^dagger) = (conj(U) ⊗ U) vec(M), column-major vec index i + 3j,
  // the ColorFiber::adjointOctetProjector convention.  That projector
  // restricts to the traceless octet and is centre-blind by construction:
  // Ad(zU) = Ad(U) for a central phase z.
  Eigen::MatrixXcd kron(9, 9);
  for (int a = 0; a < 3; ++a)
    for (int c = 0; c < 3; ++c)
      kron.block<3, 3>(3 * a, 3 * c) = std::conj(unitary(a, c)) * unitary;
  const Eigen::MatrixXcd p8 = ColorFiber::adjointOctetProjector();
  return p8 * kron * p8;
}

FundamentalLiftRead FiberConnection::fundamentalLift(
    const std::vector<FiberTransportRead> &links, const AnchorGate &gate,
    int baseBranch) const {
  if (links.empty())
    throw std::invalid_argument("FiberConnection::fundamentalLift: empty path");
  if (baseBranch < 0 || baseBranch > 2)
    throw std::invalid_argument(
        "FiberConnection::fundamentalLift: base branch must be 0, 1, or 2");

  FundamentalLiftRead read;
  read.rank = links.front().rank;
  read.baseBranch = baseBranch;

  const auto invalid = [&](const std::string &reason) {
    read.valid = false;
    read.invalidReason = reason;
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::PositiveSemidefinite);
    return read;
  };

  // The anchor gate is checked first: rank three and an accepted transport
  // do not by themselves license an SU(3) value.
  if (!gate.accepted)
    return invalid(gate.refusalReason);
  if (read.rank != 3)
    return invalid("fundamental SU(3) lift requested at rank " +
                   std::to_string(read.rank) +
                   " (never hard-coded at generic rank)");
  Eigen::MatrixXcd h = Eigen::MatrixXcd::Identity(3, 3);
  double theta = 0.0;
  double maxStep = 0.0;
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  for (std::size_t i = 0; i < links.size(); ++i) {
    const FiberTransportRead &link = links[i];
    if (!link.accepted)
      return invalid("link " + std::to_string(i) + " rejected: " +
                     link.rejectionReason);
    if (link.rank != 3)
      return invalid("rank mismatch along the path");
    if (link.unitaryMap.size() == 0)
      return invalid("link " + std::to_string(i) +
                     " carries only a GL transport (no unitary factor)");
    if (link.regime != CertificateRegime::PositiveSemidefinite)
      return invalid("link " + std::to_string(i) +
                     " is outside the positive regime (an SU(3) lift is "
                     "never emitted from a pseudo-unitary factor)");
    const cd det = link.unitaryMap.determinant();
    if (!(std::abs(det) > 0.0))
      return invalid("vanishing link determinant");
    const double step = std::arg(det);  // principal, (-pi, pi]
    theta += step;
    maxStep = std::max(maxStep, std::abs(step));
    h = h * link.unitaryMap;
    regime = pairedRegime(regime, link.regime);
  }

  // Continued branch: lift = H e^{-i Theta / 3} omega^{-s0}, with omega the
  // algebraic cube root of unity so the determinant cancels exactly.
  const cd omega = ColorFiber::omega();
  const std::array<cd, 3> centerPower{cd(1.0, 0.0), omega, omega * omega};
  const cd omegaInverseS0 = centerPower[static_cast<std::size_t>(
      (3 - (baseBranch % 3)) % 3)];
  read.lift = h * std::exp(cd(0.0, -theta / 3.0)) * omegaInverseS0;
  read.liftTrace = read.lift.trace();
  read.accumulatedDeterminantPhase = theta;
  read.maxDeterminantPhaseStep = maxStep;

  // Accumulated center sector: the sheet count of Theta relative to its
  // principal value, mod 3 — branch-independent by construction.
  const double principal = std::arg(std::exp(cd(0.0, theta)));
  const long sheets = std::lround((theta - principal) / kTwoPi);
  read.centerSector = static_cast<int>(((sheets % 3) + 3) % 3);

  const cd liftDet = read.lift.determinant();
  const double unitarityResidual = spectralNorm(
      read.lift.adjoint() * read.lift - Eigen::MatrixXcd::Identity(3, 3));
  read.detResidual = std::abs(liftDet - cd(1.0, 0.0));
  read.valid = true;
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime,
      std::max(read.detResidual, unitarityResidual), 1.0,
      cfg_.certificateTolerance);
  return read;
}

// ---------------------------------------------------------------------------
// determinant winding
// ---------------------------------------------------------------------------

DeterminantWindingRead FiberConnection::windingRead(
    const std::vector<FiberTransportRead> &family, bool cyclic,
    const WindingClosureSpec *closure) const {
  if (family.empty())
    throw std::invalid_argument("FiberConnection: empty transport family");

  DeterminantWindingRead read;
  if (cyclic) {
    read.windingClosure = "closed-family";
  } else {
    switch (closure->mode) {
      case WindingClosureSpec::Mode::None:
        read.windingClosure = "none";
        break;
      case WindingClosureSpec::Mode::MatchedReference:
        read.windingClosure = "matched-reference";
        break;
      case WindingClosureSpec::Mode::EndpointTrivialization:
        read.windingClosure = "endpoint-trivialization";
        break;
    }
    read.windingReferenceId = closure->referenceId;
  }
  const auto invalidate = [&](const std::string &reason) {
    read.winding.reset();
    read.invalidationReason = reason;
    read.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::PositiveSemidefinite);
    return read;
  };

  // Family gates: an integer winding exists only for a continuous,
  // full-rank, gapped family of accepted transports of one rank.
  const int rank = family.front().rank;
  std::vector<cd> units;
  units.reserve(family.size());
  CertificateRegime regime = CertificateRegime::PositiveSemidefinite;
  for (std::size_t i = 0; i < family.size(); ++i) {
    const FiberTransportRead &sample = family[i];
    if (!sample.accepted)
      return invalidate("sample " + std::to_string(i) +
                        " is not an accepted transport (gap or rank closed): " +
                        sample.rejectionReason);
    if (sample.rank != rank)
      return invalidate("rank changed along the family");
    bool ok = false;
    const cd unit = unitDeterminant(sample, &ok);
    if (!ok) return invalidate("vanishing determinant along the family");
    units.push_back(unit);
    regime = pairedRegime(regime, sample.regime);
  }

  // The determinant phase path: principal legs of the declared composite.
  std::vector<double> legs;
  double closureDefect = 0.0;
  if (cyclic) {
    for (std::size_t k = 0; k + 1 < units.size(); ++k)
      legs.push_back(principalStep(units[k], units[k + 1]));
    legs.push_back(principalStep(units.back(), units.front()));
  } else {
    const WindingClosureSpec &spec = *closure;
    switch (spec.mode) {
      case WindingClosureSpec::Mode::None: {
        for (std::size_t k = 0; k + 1 < units.size(); ++k)
          legs.push_back(principalStep(units[k], units[k + 1]));
        double theta = 0.0;
        double maxStep = 0.0;
        for (const double leg : legs) {
          theta += leg;
          maxStep = std::max(maxStep, std::abs(leg));
        }
        read.accumulatedPhase = theta;  // raw open-path phase, not an
        read.maxPhaseStep = maxStep;    // integer winding claim
        read.phaseStepMargin = maxStep / kPi;
        return invalidate(
            "no closure declared (a raw endpoint phase difference is not a "
            "winding certificate)");
      }
      case WindingClosureSpec::Mode::MatchedReference: {
        if (spec.referenceTransports.size() != family.size())
          throw std::invalid_argument(
              "FiberConnection::openSegmentWinding: the matched reference "
              "must supply one transport per segment sample");
        std::vector<cd> refUnits;
        refUnits.reserve(spec.referenceTransports.size());
        for (const Eigen::MatrixXcd &r : spec.referenceTransports) {
          if (r.rows() != rank || r.cols() != rank)
            throw std::invalid_argument(
                "FiberConnection::openSegmentWinding: reference transport "
                "shape mismatch");
          const cd det = r.determinant();
          if (!(std::abs(det) > 0.0))
            throw std::invalid_argument(
                "FiberConnection::openSegmentWinding: singular reference "
                "transport");
          refUnits.push_back(det / std::abs(det));
        }
        // Segment forward, cross to the reference, reference backward,
        // cross back: a closed cycle of principal legs.
        for (std::size_t k = 0; k + 1 < units.size(); ++k)
          legs.push_back(principalStep(units[k], units[k + 1]));
        legs.push_back(principalStep(units.back(), refUnits.back()));
        for (std::size_t k = refUnits.size(); k-- > 1;)
          legs.push_back(principalStep(refUnits[k], refUnits[k - 1]));
        legs.push_back(principalStep(refUnits.front(), units.front()));
        const auto relativeMismatch = [](const Eigen::MatrixXcd &a,
                                         const Eigen::MatrixXcd &b) {
          const double scale = std::max(1.0, spectralNorm(b));
          return spectralNorm(a - b) / scale;
        };
        closureDefect = std::max(
            relativeMismatch(spec.referenceTransports.front(),
                             linkMatrix(family.front())),
            relativeMismatch(spec.referenceTransports.back(),
                             linkMatrix(family.back())));
        break;
      }
      case WindingClosureSpec::Mode::EndpointTrivialization: {
        if (spec.startTrivialization.rows() != rank ||
            spec.startTrivialization.cols() != rank ||
            spec.endTrivialization.rows() != rank ||
            spec.endTrivialization.cols() != rank)
          throw std::invalid_argument(
              "FiberConnection::openSegmentWinding: trivialization shape "
              "mismatch");
        const cd det0 = spec.startTrivialization.determinant();
        const cd det1 = spec.endTrivialization.determinant();
        if (!(std::abs(det0) > 0.0) || !(std::abs(det1) > 0.0))
          throw std::invalid_argument(
              "FiberConnection::openSegmentWinding: singular trivialization");
        const cd tau0 = det0 / std::abs(det0);
        const cd tau1 = det1 / std::abs(det1);
        // Four principal legs close the determinant path exactly:
        // tau0 -> V(0) -> ... -> V(n-1) -> tau1 -> tau0.
        legs.push_back(principalStep(tau0, units.front()));
        for (std::size_t k = 0; k + 1 < units.size(); ++k)
          legs.push_back(principalStep(units[k], units[k + 1]));
        legs.push_back(principalStep(units.back(), tau1));
        legs.push_back(principalStep(tau1, tau0));
        const auto unitarityDefect = [rank](const Eigen::MatrixXcd &t) {
          return spectralNorm(t.adjoint() * t -
                              Eigen::MatrixXcd::Identity(rank, rank));
        };
        closureDefect = std::max(unitarityDefect(spec.startTrivialization),
                                 unitarityDefect(spec.endTrivialization));
        break;
      }
    }
  }

  double theta = 0.0;
  double maxStep = 0.0;
  for (const double leg : legs) {
    theta += leg;
    maxStep = std::max(maxStep, std::abs(leg));
  }
  read.accumulatedPhase = theta;
  read.maxPhaseStep = maxStep;
  read.phaseStepMargin = maxStep / kPi;
  read.closureDefect = closureDefect;
  if (maxStep >= kPi * (1.0 - 1e-12))
    return invalidate("phase step aliasing (a leg reached pi)");

  const long nu = std::lround(theta / kTwoPi);
  read.winding = static_cast<int>(nu);
  read.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, regime, closureDefect,
      read.phaseStepMargin < 1.0 ? 1.0 / (1.0 - read.phaseStepMargin) : kInf,
      cfg_.closureTolerance);
  return read;
}

DeterminantWindingRead FiberConnection::closedFamilyWinding(
    const std::vector<FiberTransportRead> &family) const {
  return windingRead(family, /*cyclic=*/true, nullptr);
}

DeterminantWindingRead FiberConnection::openSegmentWinding(
    const std::vector<FiberTransportRead> &segment,
    const WindingClosureSpec &closure) const {
  return windingRead(segment, /*cyclic=*/false, &closure);
}

}  // namespace tessera::observables
