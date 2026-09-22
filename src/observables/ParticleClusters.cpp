// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/ParticleClusters.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "cobordism/AnalyticCache.h"
#include "cobordism/EigenstateSynthesis.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "observables/InteriorHinges.h"
#include "observables/RegisterContext.h"
#include "spacetime/Spacetime.h"

/// \file
/// Certified classification of quark, gluon, meson, diquark and baryon
/// candidates from spectral fiber, holonomy and quasi-free covariance
/// evidence.  The color-singlet and net-color-flux gates are confinement
/// diagnostics on a finite complex.
/// Reference: Greensite, "The Confinement Problem in Lattice Gauge Theory",
/// arXiv:hep-lat/0301023

namespace tessera::observables {

using cobordism::Certificate;
using cobordism::CertificateDomain;
using cobordism::CertificateGrade;
using cobordism::CertificateRegime;
using cobordism::gradeName;
using cobordism::domainName;
using cobordism::regimeName;
using cobordism::domainFromName;
using cobordism::regimeFromName;
using cd = std::complex<double>;

namespace {

// Checkpoint record schema.  A reader accepts any version in
// [kOldestReadableRecordSchema, kRecordSchemaVersion]; a leaf the record does
// not carry reads back as unknown (NaN, empty or false) rather than as a
// zero-filled claim or a missing-key throw.
constexpr std::int64_t kRecordSchemaVersion = 3;
constexpr std::int64_t kOldestReadableRecordSchema = 1;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// certificate <-> record helpers
// ---------------------------------------------------------------------------






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
        "ParticleClusters: unknown certificate grade '" + grade + "'");
  }
  cert.setDenseReferenceError(m.at("dense_reference_error").asDouble());
  return cert;
}

/// Reads an optional double leaf.  An absent key is unknown (NaN), not zero.
double optionalLeaf(const Record::Map &m, const char *key) {
  const auto it = m.find(key);
  return it == m.end() ? std::numeric_limits<double>::quiet_NaN()
                       : it->second.asDouble();
}

void requireSchema(const Record::Map &m, const char *type) {
  const auto version = m.find("schema_version");
  if (version == m.end() ||
      version->second.asInt() < kOldestReadableRecordSchema ||
      version->second.asInt() > kRecordSchemaVersion)
    throw std::invalid_argument(
        "ParticleClusters: unknown schema_version (reader rejects unknown "
        "checkpoint schemas)");
  const auto rt = m.find("record_type");
  if (rt == m.end() || rt->second.asString() != type)
    throw std::invalid_argument(std::string("ParticleClusters: expected a '") +
                                type + "' record");
}

Record optionalDouble(const std::optional<double> &value) {
  return value.has_value() ? Record(*value) : Record();
}

Record optionalInt(const std::optional<int> &value) {
  return value.has_value() ? Record(*value) : Record();
}

std::optional<double> optionalDoubleFrom(const Record &r) {
  if (r.isNull()) return std::nullopt;
  return r.asDouble();
}

std::optional<int> optionalIntFrom(const Record &r) {
  if (r.isNull()) return std::nullopt;
  return static_cast<int>(r.asInt());
}

// ---------------------------------------------------------------------------
// fingerprint helpers
// ---------------------------------------------------------------------------

std::uint64_t chainHash(std::uint64_t seed, std::uint64_t value) {
  return mesh::Fingerprint::mix64(seed ^ value);
}

std::uint64_t hashDouble(std::uint64_t seed, double value) {
  return chainHash(seed, std::bit_cast<std::uint64_t>(value));
}

std::uint64_t hashComplex(std::uint64_t seed, cd value) {
  return hashDouble(hashDouble(seed, value.real()), value.imag());
}

std::uint64_t hashString(std::uint64_t seed, const std::string &s) {
  std::uint64_t h = chainHash(seed, s.size());
  for (const char c : s)
    h = chainHash(h, static_cast<std::uint64_t>(
                         static_cast<unsigned char>(c)));
  return h;
}

std::uint64_t hashCertificateHolds(std::uint64_t seed,
                                   const Certificate &cert) {
  std::uint64_t h = chainHash(seed, cert.holds() ? 1u : 0u);
  h = hashDouble(h, cert.residual());
  return hashDouble(h, cert.tolerance());
}

/// Deduplicated, sorted cell-vertex ids of a fiber: the analytic-cache key
/// material.
std::vector<std::uint64_t> fiberVertexIds(const SpectralFiber &fiber) {
  std::set<std::uint64_t> ids;
  for (const auto &cell : fiber.cellVertices())
    ids.insert(cell.begin(), cell.end());
  return {ids.begin(), ids.end()};
}

/// Running minimum over the finite candidates: a non-finite candidate never
/// lowers the minimum, and an all-unmeasured set stays NaN.  Note the
/// asymmetry with `maxFinite`, which only skips NaN.
double minFinite(double current, double candidate) {
  if (!std::isfinite(candidate)) return current;
  if (!std::isfinite(current)) return candidate;
  return std::min(current, candidate);
}

/// Running maximum of `current` and `candidate`, skipping NaN candidates.
double maxFinite(double current, double candidate) {
  if (std::isnan(candidate)) return current;
  if (std::isnan(current)) return candidate;
  return std::max(current, candidate);
}

/// Writes a fixed-shape 3x3 complex matrix to a record, row-major, split into
/// real and imaginary lists.
void matrix3ToRecord(Record::Map &m, const std::string &name,
                     const Eigen::Matrix3cd &matrix) {
  std::vector<cd> flat(9);
  for (Eigen::Index r = 0; r < 3; ++r)
    for (Eigen::Index c = 0; c < 3; ++c)
      flat[static_cast<std::size_t>(r * 3 + c)] = matrix(r, c);
  Record::splitComplex(m, name, flat);
}

Eigen::Matrix3cd matrix3FromRecord(const Record::Map &m,
                                   const std::string &name) {
  const auto &re = m.at(name + "_re").asList();
  const auto &im = m.at(name + "_im").asList();
  if (re.size() != 9 || im.size() != 9)
    throw std::invalid_argument(
        "ParticleClusters: 3x3 matrix record payload size mismatch");
  Eigen::Matrix3cd matrix;
  for (Eigen::Index r = 0; r < 3; ++r)
    for (Eigen::Index c = 0; c < 3; ++c) {
      const auto i = static_cast<std::size_t>(r * 3 + c);
      matrix(r, c) = cd(re[i].asDouble(), im[i].asDouble());
    }
  return matrix;
}

/// Component identity <-> record key pair.
void componentToRecord(Record::Map &m, const std::string &prefix,
                       const ComponentId &component) {
  m[prefix + "_hash"] = Record(component.canonicalHash());
  m[prefix + "_level"] =
      Record(static_cast<std::int64_t>(component.level()));
}

ComponentId componentFromRecord(const Record::Map &m,
                                const std::string &prefix) {
  return ComponentId(
      m.at(prefix + "_hash").asString(),
      static_cast<std::size_t>(m.at(prefix + "_level").asInt()));
}

/// Sign of a certified \f$ \pm 1 \f$-valued Wick read: +1 or −1 when the
/// value lies within `tolerance` of that sign and the certificate holds,
/// 0 (unknown) otherwise.  An uncertified read never emits a sign.
int certifiedSign(const quantum::WickCertificateRead &read, double tolerance) {
  if (!read.certificate.holds()) return 0;
  if (std::abs(read.value - cd(1.0, 0.0)) <= tolerance) return +1;
  if (std::abs(read.value - cd(-1.0, 0.0)) <= tolerance) return -1;
  return 0;
}

/// Accumulates residual and tolerance over the certificates a verdict
/// consumes; shared by every classifier.  An accepted verdict is
/// StructureExact given those held certificates, with residual and tolerance
/// their maxima and regime and conditioning taken from a designated donor
/// certificate; a refused verdict is HeuristicDiscovery in the donor regime.
class ConsumedCertificates {
 public:
  void consume(const Certificate &cert) {
    residual_ = maxFinite(residual_, cert.residual());
    tolerance_ = std::max(tolerance_, cert.tolerance());
  }
  /// Fold an extra measured residual channel (e.g. |Im| leakage of a
  /// real-by-construction sum).
  void consumeResidual(double value) {
    residual_ = maxFinite(residual_, value);
  }
  [[nodiscard]] double residual() const { return residual_; }
  /// The residual with an unmeasured (NaN) residual collapsed to 0, for
  /// AlgebraicallyExact assemblies whose consumed reads always measure.
  [[nodiscard]] double residualOrZero() const {
    return std::isnan(residual_) ? 0.0 : residual_;
  }
  [[nodiscard]] double tolerance() const { return tolerance_; }
  [[nodiscard]] Certificate verdict(bool accepted,
                                    const Certificate &donor) const {
    if (!accepted) {
      return Certificate::heuristicDiscovery(CertificateDomain::Static,
                                             donor.regime());
    }
    return Certificate::structureExact(CertificateDomain::Static,
                                       donor.regime(), residual_,
                                       donor.conditioning(), tolerance_);
  }

 private:
  double residual_ = std::numeric_limits<double>::quiet_NaN();
  double tolerance_ = 0.0;
};

/// Composite parity of two constituent reads: the product of the certified
/// constituent parities (parity adds mod 2), 0 (unknown) when either is
/// uncertified.
int compositeParity(const QuarkRead &first, const QuarkRead &second) {
  if (first.exteriorParity == 0 || second.exteriorParity == 0) return 0;
  return first.exteriorParity * second.exteriorParity;
}

/// A constituent QuarkRead with the given certified verdict.
bool certifiedConstituent(const QuarkRead &read, const char *verdict) {
  return read.classification == verdict && read.certificate.holds();
}

/// Shared composite report channels (occupation, transports, lifetime).
template <typename ReadT>
void fillCompositeReports(ReadT &read,
                          const CompositeCandidateEvidence &evidence) {
  read.bindingComponent = evidence.bindingComponent;
  read.firstConstituent = evidence.first.component;
  read.secondConstituent = evidence.second.component;
  read.occupationTotal = evidence.occupationRead.certificate.holds()
                             ? evidence.occupationRead.value.real()
                             : kNaN;
  read.transportCount = evidence.lifetimeTransports.size();
  double maxLeakage = kNaN;
  for (const FiberTransportRead &transport : evidence.lifetimeTransports)
    maxLeakage = maxFinite(maxLeakage, transport.leakage);
  read.transportLeakageMax = maxLeakage;
  read.persistenceLifetime = evidence.persistenceLifetime;
}

}  // namespace

// ---------------------------------------------------------------------------
// QuarkRead
// ---------------------------------------------------------------------------

std::string QuarkRead::describe() const {
  std::ostringstream out;
  out << "QuarkRead[" << classification;
  if (determinantWinding.has_value())
    out << ", nu=" << *determinantWinding << " (" << windingClosure << ")";
  else
    out << ", nu=unknown";
  if (baryonFlux.has_value())
    out << ", B=" << *baryonFlux;
  else
    out << ", B=unknown";
  out << ", parity=" << exteriorParity << ", color rank " << colorRank
      << ", confidence " << confidence << "]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

namespace {

Record thresholdsToRecord(const ParticleClustersConfig &cfg) {
  Record::Map m;
  m["parity_tolerance"] = Record(cfg.parityTolerance);
  m["occupation_tolerance"] = Record(cfg.occupationTolerance);
  m["min_anchor_score"] = Record(cfg.minAnchorScore);
  m["min_phase_coherence"] = Record(cfg.minPhaseCoherence);
  m["max_transport_leakage"] = Record(cfg.maxTransportLeakage);
  m["min_persistence_lifetime"] = Record(cfg.minPersistenceLifetime);
  m["min_persistence_overlap"] = Record(cfg.minPersistenceOverlap);
  m["min_localization"] = Record(cfg.minLocalization);
  m["min_refinement_overlap"] = Record(cfg.minRefinementOverlap);
  m["min_stability_frames"] =
      Record(static_cast<std::int64_t>(cfg.minStabilityFrames));
  m["doublet_overlap_threshold"] = Record(cfg.doubletOverlapThreshold);
  m["min_doublet_frames"] =
      Record(static_cast<std::int64_t>(cfg.minDoubletFrames));
  m["isospin_tolerance"] = Record(cfg.isospinTolerance);
  m["gauss_tolerance"] = Record(cfg.gaussTolerance);
  m["min_enclosing_surfaces"] =
      Record(static_cast<std::int64_t>(cfg.minEnclosingSurfaces));
  m["ud_tolerance"] = Record(cfg.udTolerance);
  m["min_octet_weight"] = Record(cfg.minOctetWeight);
  m["octet_purity_tolerance"] = Record(cfg.octetPurityTolerance);
  m["composite_octet_tolerance"] = Record(cfg.compositeOctetTolerance);
  m["min_anti_triplet_weight"] = Record(cfg.minAntiTripletWeight);
  m["color_gram_tolerance"] = Record(cfg.colorGramTolerance);
  m["color_flux_tolerance"] = Record(cfg.colorFluxTolerance);
  m["spin_expectation_tolerance"] = Record(cfg.spinExpectationTolerance);
  m["spin_variance_tolerance"] = Record(cfg.spinVarianceTolerance);
  m["min_support_containment"] = Record(cfg.minSupportContainment);
  m["min_lifetime_overlap"] = Record(cfg.minLifetimeOverlap);
  m["min_radius"] = Record(cfg.minRadius);
  m["max_profile_deviation"] = Record(cfg.maxProfileDeviation);
  return Record(std::move(m));
}

ParticleClustersConfig thresholdsFromRecord(const Record &record) {
  const auto &m = record.asMap();
  ParticleClustersConfig cfg;
  cfg.parityTolerance = m.at("parity_tolerance").asDouble();
  cfg.occupationTolerance = m.at("occupation_tolerance").asDouble();
  cfg.minAnchorScore = m.at("min_anchor_score").asDouble();
  cfg.minPhaseCoherence = m.at("min_phase_coherence").asDouble();
  cfg.maxTransportLeakage = m.at("max_transport_leakage").asDouble();
  cfg.minPersistenceLifetime = m.at("min_persistence_lifetime").asDouble();
  cfg.minPersistenceOverlap = m.at("min_persistence_overlap").asDouble();
  cfg.minLocalization = m.at("min_localization").asDouble();
  cfg.minRefinementOverlap = m.at("min_refinement_overlap").asDouble();
  if (m.count("min_stability_frames"))
    cfg.minStabilityFrames =
        static_cast<std::size_t>(m.at("min_stability_frames").asInt());
  cfg.doubletOverlapThreshold = m.at("doublet_overlap_threshold").asDouble();
  cfg.minDoubletFrames =
      static_cast<std::size_t>(m.at("min_doublet_frames").asInt());
  cfg.isospinTolerance = m.at("isospin_tolerance").asDouble();
  cfg.gaussTolerance = m.at("gauss_tolerance").asDouble();
  cfg.minEnclosingSurfaces =
      static_cast<std::size_t>(m.at("min_enclosing_surfaces").asInt());
  cfg.udTolerance = m.at("ud_tolerance").asDouble();
  // Optional keys: read with a default so a record without them rehydrates.
  const auto readOr = [&m](const char *key, double fallback) {
    const auto it = m.find(key);
    return it == m.end() ? fallback : it->second.asDouble();
  };
  cfg.minOctetWeight = readOr("min_octet_weight", cfg.minOctetWeight);
  cfg.octetPurityTolerance =
      readOr("octet_purity_tolerance", cfg.octetPurityTolerance);
  cfg.compositeOctetTolerance =
      readOr("composite_octet_tolerance", cfg.compositeOctetTolerance);
  cfg.minAntiTripletWeight =
      readOr("min_anti_triplet_weight", cfg.minAntiTripletWeight);
  // Same default-fallback contract for these keys.
  cfg.colorGramTolerance =
      readOr("color_gram_tolerance", cfg.colorGramTolerance);
  cfg.colorFluxTolerance =
      readOr("color_flux_tolerance", cfg.colorFluxTolerance);
  cfg.spinExpectationTolerance =
      readOr("spin_expectation_tolerance", cfg.spinExpectationTolerance);
  cfg.spinVarianceTolerance =
      readOr("spin_variance_tolerance", cfg.spinVarianceTolerance);
  cfg.minSupportContainment =
      readOr("min_support_containment", cfg.minSupportContainment);
  cfg.minLifetimeOverlap =
      readOr("min_lifetime_overlap", cfg.minLifetimeOverlap);
  cfg.minRadius = readOr("min_radius", cfg.minRadius);
  cfg.maxProfileDeviation =
      readOr("max_profile_deviation", cfg.maxProfileDeviation);
  return cfg;
}

}  // namespace

Record QuarkRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("quark_read");
  m["component_hash"] = Record(component.canonicalHash());
  m["component_level"] =
      Record(static_cast<std::int64_t>(component.level()));
  m["exterior_parity"] = Record(exteriorParity);
  m["color_rank"] = Record(colorRank);
  m["triangle_anchor_score"] = Record(triangleAnchorScore);
  m["triangle_anchor_max_term"] = Record(triangleAnchorMaxTerm);
  m["triangle_anchor_participation"] = Record(triangleAnchorParticipation);
  m["anchor_phase_dispersion"] = Record(anchorPhaseDispersion);
  m["anchor_phase_coherence"] = Record(anchorPhaseCoherence);
  m["anchor_weighting_id"] = Record(anchorWeightingId);
  m["determinant_winding"] = optionalInt(determinantWinding);
  m["winding_closure"] = Record(windingClosure);
  m["winding_reference_id"] = Record(windingReferenceId);
  m["baryon_flux"] = optionalDouble(baryonFlux);
  m["isospin"] = optionalDouble(isospin);
  m["electric_flux"] = optionalDouble(electricFlux);
  m["confidence"] = Record(confidence);
  Record::List failed;
  failed.reserve(failedCertificates.size());
  for (const auto &name : failedCertificates) failed.emplace_back(name);
  m["failed_certificates"] = Record(std::move(failed));
  m["classification"] = Record(classification);
  m["occupation_total"] = Record(occupationTotal);
  m["transport_count"] = Record(static_cast<std::int64_t>(transportCount));
  m["transport_leakage_max"] = Record(transportLeakageMax);
  m["persistence_lifetime"] = Record(persistenceLifetime);
  m["persistence_min_overlap"] = Record(persistenceMinOverlap);
  m["frame_lifetime"] = Record(frameLifetime);
  m["frame_min_overlap"] = Record(frameMinOverlap);
  m["stability_frames"] = Record(static_cast<std::int64_t>(stabilityFrames));
  m["anchor_score_spread"] = Record(anchorScoreSpread);
  m["anchor_coherence_spread"] = Record(anchorCoherenceSpread);
  m["band_continuation_overlap"] = Record(bandContinuationOverlap);
  m["localization"] = Record(localization);
  m["localization_support_fraction"] = Record(localizationSupportFraction);
  m["refinement_overlap"] = Record(refinementOverlap);
  m["ud_identification_proposed"] = Record(udIdentificationProposed);
  m["doublet_orientation"] = Record(doubletOrientation);
  m["thresholds"] = thresholdsToRecord(thresholds);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

QuarkRead QuarkRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "quark_read");
  QuarkRead read;
  read.component = ComponentId(
      m.at("component_hash").asString(),
      static_cast<std::size_t>(m.at("component_level").asInt()));
  read.exteriorParity = static_cast<int>(m.at("exterior_parity").asInt());
  read.colorRank = static_cast<int>(m.at("color_rank").asInt());
  read.triangleAnchorScore = m.at("triangle_anchor_score").asDouble();
  read.triangleAnchorMaxTerm = m.at("triangle_anchor_max_term").asDouble();
  read.triangleAnchorParticipation =
      m.at("triangle_anchor_participation").asDouble();
  read.anchorPhaseDispersion = m.at("anchor_phase_dispersion").asDouble();
  read.anchorPhaseCoherence = m.at("anchor_phase_coherence").asDouble();
  read.anchorWeightingId = m.at("anchor_weighting_id").asString();
  read.determinantWinding = optionalIntFrom(m.at("determinant_winding"));
  read.windingClosure = m.at("winding_closure").asString();
  read.windingReferenceId = m.at("winding_reference_id").asString();
  read.baryonFlux = optionalDoubleFrom(m.at("baryon_flux"));
  read.isospin = optionalDoubleFrom(m.at("isospin"));
  read.electricFlux = optionalDoubleFrom(m.at("electric_flux"));
  read.confidence = m.at("confidence").asDouble();
  for (const auto &entry : m.at("failed_certificates").asList())
    read.failedCertificates.push_back(entry.asString());
  read.classification = m.at("classification").asString();
  read.occupationTotal = m.at("occupation_total").asDouble();
  read.transportCount =
      static_cast<std::size_t>(m.at("transport_count").asInt());
  read.transportLeakageMax = m.at("transport_leakage_max").asDouble();
  read.persistenceLifetime = m.at("persistence_lifetime").asDouble();
  read.persistenceMinOverlap = m.at("persistence_min_overlap").asDouble();
  read.frameLifetime = optionalLeaf(m, "frame_lifetime");
  read.frameMinOverlap = optionalLeaf(m, "frame_min_overlap");
  read.stabilityFrames =
      m.count("stability_frames")
          ? static_cast<std::size_t>(m.at("stability_frames").asInt())
          : 0;
  read.anchorScoreSpread = optionalLeaf(m, "anchor_score_spread");
  read.anchorCoherenceSpread = optionalLeaf(m, "anchor_coherence_spread");
  read.bandContinuationOverlap = optionalLeaf(m, "band_continuation_overlap");
  read.localization = m.at("localization").asDouble();
  read.localizationSupportFraction =
      optionalLeaf(m, "localization_support_fraction");
  read.refinementOverlap = m.at("refinement_overlap").asDouble();
  read.udIdentificationProposed =
      m.at("ud_identification_proposed").asBool();
  read.doubletOrientation =
      static_cast<int>(m.at("doublet_orientation").asInt());
  read.thresholds = thresholdsFromRecord(m.at("thresholds"));
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

// ---------------------------------------------------------------------------
// ParticleClusters
// ---------------------------------------------------------------------------

ParticleClusters::ParticleClusters(ParticleClustersConfig cfg) : cfg_(cfg) {}

bool ParticleClusters::gate(bool passed, const char *name,
                            std::vector<std::string> &failed) {
  if (!passed) failed.emplace_back(name);
  return passed;
}

QuarkRead ParticleClusters::classifyQuark(
    const QuarkCandidateEvidence &evidence) const {
  QuarkRead read;
  read.component = evidence.component;
  read.thresholds = cfg_;

  std::vector<std::string> failed;
  int passedCore = 0;
  constexpr int kCoreCertificates = 12;

  // 1. persistence: lifetime across multiple cobordism frames.  The
  //    modularity resolution-slice lifetime travels beside it as a report
  //    only; a modularity read never vetoes a certified fiber.  NaN means
  //    missing evidence.
  read.persistenceLifetime = evidence.persistenceLifetime;
  read.persistenceMinOverlap = evidence.persistenceMinOverlap;
  read.frameLifetime = evidence.frameLifetime;
  read.frameMinOverlap = evidence.frameMinOverlap;
  const bool persistenceOk =
      std::isfinite(evidence.frameLifetime) &&
      evidence.frameLifetime >= cfg_.minPersistenceLifetime &&
      std::isfinite(evidence.frameMinOverlap) &&
      evidence.frameMinOverlap >= cfg_.minPersistenceOverlap;
  passedCore += gate(persistenceOk, "persistence", failed);

  // 2. localization, from the color band's own certificate.  Fiber
  //    acceptance (SpectralFiberConfig::maxLocalizationExcess) already
  //    rejects a delocalized band upstream; this gate applies the
  //    classifier's own floor.
  const SpectralBandCertificate &band = evidence.colorBand.certificate();
  read.localization = band.localization;
  read.localizationSupportFraction = band.localizationSupportFraction;
  const bool localizationOk = std::isfinite(band.localization) &&
                              band.localization >= cfg_.minLocalization;
  passedCore += gate(localizationOk, "localization", failed);

  // 3. odd exterior parity from the Wick parity read; an uncertified read
  //    emits no sign.
  const auto &parity = evidence.parityRead;
  const bool parityCertified = parity.certificate.holds();
  if (parityCertified &&
      std::abs(parity.value - cd(-1.0, 0.0)) <= cfg_.parityTolerance) {
    read.exteriorParity = -1;
  } else if (parityCertified &&
             std::abs(parity.value - cd(1.0, 0.0)) <= cfg_.parityTolerance) {
    read.exteriorParity = +1;
  } else {
    read.exteriorParity = 0;
  }
  passedCore += gate(read.exteriorParity == -1, "parity-odd", failed);

  // 4. single-fermion occupation from the Wick total-number read: the
  //    total-occupation channel excluding the two-quark anti-triplet.
  const auto &occupation = evidence.occupationRead;
  const bool occupationCertified = occupation.certificate.holds();
  read.occupationTotal =
      occupationCertified ? occupation.value.real() : kNaN;
  const bool occupationOneOk =
      occupationCertified &&
      std::abs(occupation.value - cd(1.0, 0.0)) <= cfg_.occupationTolerance;
  passedCore += gate(occupationOneOk, "occupation-one", failed);

  // 5. accepted rank-three color band; the rank is read, never requested
  //    from the detector.
  read.colorRank = static_cast<int>(evidence.colorBand.rank());
  const bool rankThreeOk =
      evidence.colorBand.accepted() && evidence.colorBand.rank() == 3;
  passedCore += gate(rankThreeOk, "color-rank-three", failed);

  // 6. stable rank three: rank three accepted at every supplied cobordism
  //    frame, with consecutive frames linked by certified continuations.
  //    One frame cannot establish stability, so a window shorter than
  //    minStabilityFrames fails by name.
  const std::vector<SpectralFiber> &bandFrames = evidence.colorBandFrames;
  read.stabilityFrames = bandFrames.size();
  bool rankStableOk = bandFrames.size() >= cfg_.minStabilityFrames;
  for (const SpectralFiber &frame : bandFrames)
    rankStableOk = rankStableOk && frame.accepted() && frame.rank() == 3;
  double continuationOverlap = kNaN;
  for (std::size_t t = 0; t + 1 < bandFrames.size(); ++t) {
    const std::vector<FiberMatchRead> links = SpectralFiberTracker::matchFibers(
        {bandFrames[t]}, {bandFrames[t + 1]}, cfg_.doubletOverlapThreshold);
    // No link at all means the two frames share no support: the measured
    // continuation overlap is zero, not unknown.
    const double measured =
        links.empty() ? 0.0 : links.front().overlap.subspaceOverlap;
    continuationOverlap = minFinite(continuationOverlap, measured);
    if (links.empty() || !links.front().certifiedContinuation)
      rankStableOk = false;
  }
  read.bandContinuationOverlap = continuationOverlap;
  passedCore += gate(rankStableOk, "color-rank-stability", failed);

  // 7. calibrated oriented-triangle anchor.  A default-constructed profile
  //    (no weighting declared) is missing evidence: the anchor fields stay
  //    unknown.
  const AnchorProfile &anchor = evidence.anchor;
  const bool anchorSupplied = !anchor.weightingId.empty();
  if (anchorSupplied) {
    read.triangleAnchorScore = anchor.score;
    read.triangleAnchorMaxTerm = anchor.maxTerm;
    read.triangleAnchorParticipation = anchor.participationRatio;
    read.anchorPhaseDispersion = anchor.phaseDispersion;
    read.anchorPhaseCoherence = anchor.phaseCoherence;
    read.anchorWeightingId = anchor.weightingId;
  }
  // `ColorAnchor::accepts` is the single acceptance predicate, shared with the
  // color kernels gated on this same certificate, so interpretation and
  // kernels cannot drift apart.  This lambda only binds the configured floors
  // for the two call sites below: the verdict here and the per-frame
  // conjunction at item 8.
  const auto anchorHolds = [&](const AnchorProfile &profile) {
    return ColorAnchor::accepts(profile, cfg_.minAnchorScore,
                                cfg_.minPhaseCoherence);
  };
  passedCore += gate(anchorHolds(anchor), "anchor", failed);

  // 8. stable anchor profile and determinant-line coherence: both hold at
  //    every supplied frame, over a window of at least minStabilityFrames.
  //    The across-frame spreads are measured and reported; the certificate
  //    is the conjunction, not a spread cap.
  const std::vector<AnchorProfile> &anchorFrames = evidence.anchorFrames;
  bool anchorStableOk = anchorFrames.size() >= cfg_.minStabilityFrames;
  double scoreLo = kNaN;
  double scoreHi = kNaN;
  double coherenceLo = kNaN;
  double coherenceHi = kNaN;
  for (const AnchorProfile &profile : anchorFrames) {
    anchorStableOk = anchorStableOk && anchorHolds(profile);
    scoreLo = minFinite(scoreLo, profile.score);
    scoreHi = maxFinite(scoreHi, profile.score);
    coherenceLo = minFinite(coherenceLo, profile.phaseCoherence);
    coherenceHi = maxFinite(coherenceHi, profile.phaseCoherence);
  }
  if (anchorFrames.size() >= 2) {
    read.anchorScoreSpread = scoreHi - scoreLo;
    read.anchorCoherenceSpread = coherenceHi - coherenceLo;
  }
  passedCore += gate(anchorStableOk, "anchor-stability", failed);

  // 9. bounded transport leakage over the lifetime.
  read.transportCount = evidence.lifetimeTransports.size();
  bool allTransportsAccepted = !evidence.lifetimeTransports.empty();
  double maxLeakage = kNaN;
  for (const FiberTransportRead &transport : evidence.lifetimeTransports) {
    allTransportsAccepted = allTransportsAccepted && transport.accepted;
    maxLeakage = maxFinite(maxLeakage, transport.leakage);
  }
  read.transportLeakageMax = maxLeakage;
  const bool leakageOk = allTransportsAccepted &&
                         std::isfinite(maxLeakage) &&
                         maxLeakage <= cfg_.maxTransportLeakage;
  passedCore += gate(leakageOk, "transport-leakage", failed);

  // 10/11. certified determinant-line winding and unit magnitude.  The
  //      closure convention travels with the read; B = nu/3 exists exactly
  //      when the winding certificate does (a certified nu = 0 is a
  //      certified zero flux), and quark-ness additionally needs
  //      |nu| = 1.
  const DeterminantWindingRead &winding = evidence.winding;
  read.windingClosure = winding.windingClosure;
  read.windingReferenceId = winding.windingReferenceId;
  const bool windingOk =
      winding.winding.has_value() && winding.certificate.holds();
  if (windingOk) {
    read.determinantWinding = winding.winding;
    read.baryonFlux = static_cast<double>(*winding.winding) / 3.0;
  }
  passedCore += gate(windingOk, "winding", failed);
  const bool windingUnitOk =
      windingOk && std::abs(*winding.winding) == 1;
  passedCore += gate(windingUnitOk, "winding-unit", failed);

  // 12. refinement stability (band subspace overlap across a refinement).
  read.refinementOverlap = evidence.refinementOverlap;
  const bool refinementOk =
      std::isfinite(evidence.refinementOverlap) &&
      evidence.refinementOverlap >= cfg_.minRefinementOverlap;
  passedCore += gate(refinementOk, "refinement-stability", failed);

  read.confidence =
      static_cast<double>(passedCore) / static_cast<double>(kCoreCertificates);

  // Verdict: quark vs antiquark is the determinant-line orientation of the
  // certified winding, not the color representation alone.
  if (passedCore == kCoreCertificates) {
    read.classification = (*winding.winding == +1) ? "quark" : "antiquark";
  } else {
    read.classification = "none";
  }

  // Flavor: only an emergent certified two-state subclass reports isospin.
  const bool flavorOk = evidence.flavor.has_value() &&
                        evidence.flavor->found &&
                        evidence.flavor->certificate.holds();
  if (!flavorOk) {
    failed.emplace_back("flavor-doublet");
  } else {
    bool isospinDefinite = false;
    if (evidence.doubletOccupancy.has_value() &&
        (evidence.doubletOrientation == +1 ||
         evidence.doubletOrientation == -1)) {
      const Eigen::Vector2cd &f = *evidence.doubletOccupancy;
      const double norm2 = std::norm(f(0)) + std::norm(f(1));
      if (norm2 > 0.0) {
        const double i3Raw =
            (std::norm(f(0)) - std::norm(f(1))) / (2.0 * norm2);
        if (std::abs(i3Raw - 0.5) <= cfg_.isospinTolerance) {
          read.isospin = 0.5 * evidence.doubletOrientation;
          isospinDefinite = true;
        } else if (std::abs(i3Raw + 0.5) <= cfg_.isospinTolerance) {
          read.isospin = -0.5 * evidence.doubletOrientation;
          isospinDefinite = true;
        }
      }
    }
    if (isospinDefinite) {
      read.doubletOrientation = evidence.doubletOrientation;
    } else {
      failed.emplace_back("isospin");
    }
  }

  // Charge: the Gauss read must be consistent across nested enclosing
  // surfaces and the doublet must be certified.  A missing or unstable flavor
  // doublet leaves both flavor and charge unknown.
  const bool gaussOk = evidence.charge.has_value() &&
                       evidence.charge->consistent &&
                       evidence.charge->certificate.holds() &&
                       evidence.charge->electricFlux.has_value();
  if (!gaussOk) failed.emplace_back("gauss-consistency");
  if (gaussOk && flavorOk) read.electricFlux = evidence.charge->electricFlux;

  // The proposed u/d identification Q = I3 + B/2 is tested only when
  // baryon flux, isospin and the Gauss-consistent charge all exist.
  if (read.baryonFlux.has_value() && read.isospin.has_value() &&
      read.electricFlux.has_value()) {
    const double predicted = *read.isospin + *read.baryonFlux / 2.0;
    if (std::abs(*read.electricFlux - predicted) <= cfg_.udTolerance) {
      read.udIdentificationProposed = true;
    } else {
      failed.emplace_back("ud-identification");
    }
  }

  read.failedCertificates = std::move(failed);

  // An accepted verdict is an exact boolean combination of the consumed held
  // certificates (StructureExact); residual and tolerance are their maxima,
  // so holds() follows from theirs.
  ConsumedCertificates consumed;
  consumed.consume(band.certificate);
  consumed.consume(anchor.certificate);
  for (const FiberTransportRead &transport : evidence.lifetimeTransports)
    consumed.consume(transport.certificate);
  consumed.consume(winding.certificate);
  consumed.consume(parity.certificate);
  consumed.consume(occupation.certificate);
  read.certificate =
      consumed.verdict(read.classification != "none", band.certificate);
  return read;
}

std::vector<QuarkRead> ParticleClusters::classifyQuarks(
    const std::vector<QuarkCandidateEvidence> &candidates) const {
  std::vector<QuarkRead> reads;
  reads.reserve(candidates.size());
  for (const QuarkCandidateEvidence &evidence : candidates)
    reads.push_back(classifyQuark(evidence));
  return reads;
}

QuarkRead ParticleClusters::classifyQuarkCached(
    cobordism::AnalyticCache &cache,
    const QuarkCandidateEvidence &evidence) const {
  const std::vector<std::uint64_t> ids = fiberVertexIds(evidence.colorBand);
  const auto parameter =
      static_cast<std::int64_t>(evidenceFingerprint(evidence));
  if (const auto payload = cache.fetch(ids, kCacheKind, parameter))
    return *std::static_pointer_cast<const QuarkRead>(payload);
  QuarkRead read = classifyQuark(evidence);
  cache.store(ids, kCacheKind, parameter, std::make_shared<QuarkRead>(read),
              read.certificate);
  return read;
}

std::uint64_t ParticleClusters::evidenceFingerprint(
    const QuarkCandidateEvidence &evidence) const {
  std::uint64_t h = 0x9e3779b97f4a7c15ull;
  // The thresholds are part of the verdict: a different configuration must
  // never serve another configuration's cached read.
  h = hashDouble(h, cfg_.parityTolerance);
  h = hashDouble(h, cfg_.occupationTolerance);
  h = hashDouble(h, cfg_.minAnchorScore);
  h = hashDouble(h, cfg_.minPhaseCoherence);
  h = hashDouble(h, cfg_.maxTransportLeakage);
  h = hashDouble(h, cfg_.minPersistenceLifetime);
  h = hashDouble(h, cfg_.minPersistenceOverlap);
  h = hashDouble(h, cfg_.minLocalization);
  h = hashDouble(h, cfg_.minRefinementOverlap);
  h = chainHash(h, cfg_.minStabilityFrames);
  h = hashDouble(h, cfg_.doubletOverlapThreshold);
  h = chainHash(h, cfg_.minDoubletFrames);
  h = hashDouble(h, cfg_.isospinTolerance);
  h = hashDouble(h, cfg_.gaussTolerance);
  h = chainHash(h, cfg_.minEnclosingSurfaces);
  h = hashDouble(h, cfg_.udTolerance);
  h = hashDouble(h, cfg_.minOctetWeight);
  h = hashDouble(h, cfg_.octetPurityTolerance);
  h = hashDouble(h, cfg_.compositeOctetTolerance);
  h = hashDouble(h, cfg_.minAntiTripletWeight);
  h = hashDouble(h, cfg_.colorGramTolerance);
  h = hashDouble(h, cfg_.colorFluxTolerance);
  h = hashDouble(h, cfg_.spinExpectationTolerance);
  h = hashDouble(h, cfg_.spinVarianceTolerance);
  h = hashDouble(h, cfg_.minSupportContainment);
  h = hashDouble(h, cfg_.minLifetimeOverlap);
  h = hashDouble(h, cfg_.minRadius);
  h = hashDouble(h, cfg_.maxProfileDeviation);

  h = hashString(h, evidence.component.canonicalHash());
  h = chainHash(h, evidence.component.level());

  // Color band: cells, eigenvalues, rank, acceptance, localization.
  for (const auto &cell : evidence.colorBand.cellVertices()) {
    h = chainHash(h, cell.size());
    for (const std::uint64_t id : cell) h = chainHash(h, id);
  }
  for (const cd &lambda : evidence.colorBand.eigenvalues())
    h = hashComplex(h, lambda);
  h = chainHash(h, evidence.colorBand.rank());
  h = chainHash(h, evidence.colorBand.accepted() ? 1u : 0u);
  h = hashDouble(h, evidence.colorBand.certificate().localization);

  // The across-frame stability window: every frame's decision channels
  // (rank, acceptance, cells) is part of the verdict.
  h = chainHash(h, evidence.colorBandFrames.size());
  for (const SpectralFiber &frame : evidence.colorBandFrames) {
    h = chainHash(h, frame.rank());
    h = chainHash(h, frame.accepted() ? 1u : 0u);
    for (const auto &cell : frame.cellVertices()) {
      h = chainHash(h, cell.size());
      for (const std::uint64_t id : cell) h = chainHash(h, id);
    }
    for (const cd &lambda : frame.eigenvalues()) h = hashComplex(h, lambda);
  }
  h = chainHash(h, evidence.anchorFrames.size());
  for (const AnchorProfile &profile : evidence.anchorFrames) {
    h = hashDouble(h, profile.score);
    h = hashDouble(h, profile.phaseCoherence);
    h = hashString(h, profile.weightingId);
    h = hashCertificateHolds(h, profile.certificate);
  }

  // Anchor profile (decision channels + declared weighting).
  h = hashDouble(h, evidence.anchor.score);
  h = hashDouble(h, evidence.anchor.maxTerm);
  h = hashDouble(h, evidence.anchor.participationRatio);
  h = hashDouble(h, evidence.anchor.phaseCoherence);
  h = hashDouble(h, evidence.anchor.phaseDispersion);
  h = hashString(h, evidence.anchor.weightingId);
  h = hashCertificateHolds(h, evidence.anchor.certificate);

  // Lifetime transports.
  h = chainHash(h, evidence.lifetimeTransports.size());
  for (const FiberTransportRead &transport : evidence.lifetimeTransports) {
    h = chainHash(h, transport.accepted ? 1u : 0u);
    h = hashDouble(h, transport.leakage);
    h = hashComplex(h, transport.determinantPhase);
  }

  // Winding read with its recorded closure convention.
  h = chainHash(h, evidence.winding.winding.has_value() ? 1u : 0u);
  if (evidence.winding.winding.has_value())
    h = chainHash(h, static_cast<std::uint64_t>(
                         static_cast<std::int64_t>(*evidence.winding.winding)));
  h = hashString(h, evidence.winding.windingClosure);
  h = hashString(h, evidence.winding.windingReferenceId);
  h = hashDouble(h, evidence.winding.accumulatedPhase);
  h = hashCertificateHolds(h, evidence.winding.certificate);

  // Quasi-free parity/occupation reads (covariance hashes included).
  h = hashComplex(h, evidence.parityRead.value);
  h = hashString(h, evidence.parityRead.covarianceHash);
  h = hashCertificateHolds(h, evidence.parityRead.certificate);
  h = hashComplex(h, evidence.occupationRead.value);
  h = hashString(h, evidence.occupationRead.covarianceHash);
  h = hashCertificateHolds(h, evidence.occupationRead.certificate);

  h = hashDouble(h, evidence.persistenceLifetime);
  h = hashDouble(h, evidence.persistenceMinOverlap);
  h = hashDouble(h, evidence.frameLifetime);
  h = hashDouble(h, evidence.frameMinOverlap);
  h = hashDouble(h, evidence.refinementOverlap);

  h = chainHash(h, evidence.flavor.has_value() ? 1u : 0u);
  if (evidence.flavor.has_value()) {
    h = chainHash(h, evidence.flavor->found ? 1u : 0u);
    h = chainHash(h, evidence.flavor->rank);
    h = hashDouble(h, evidence.flavor->minContinuationOverlap);
    h = hashCertificateHolds(h, evidence.flavor->certificate);
  }
  h = chainHash(h, evidence.doubletOccupancy.has_value() ? 1u : 0u);
  if (evidence.doubletOccupancy.has_value()) {
    h = hashComplex(h, (*evidence.doubletOccupancy)(0));
    h = hashComplex(h, (*evidence.doubletOccupancy)(1));
  }
  h = chainHash(h, static_cast<std::uint64_t>(
                       static_cast<std::int64_t>(evidence.doubletOrientation)));
  h = chainHash(h, evidence.charge.has_value() ? 1u : 0u);
  if (evidence.charge.has_value()) {
    h = chainHash(h, evidence.charge->consistent ? 1u : 0u);
    h = chainHash(h, evidence.charge->electricFlux.has_value() ? 1u : 0u);
    if (evidence.charge->electricFlux.has_value())
      h = hashDouble(h, *evidence.charge->electricFlux);
    for (const cd &flux : evidence.charge->fluxes) h = hashComplex(h, flux);
  }
  return h;
}

// ---------------------------------------------------------------------------
// conjugate-pair conservation
// ---------------------------------------------------------------------------

ConjugatePairRead ParticleClusters::conjugatePair(
    const QuarkRead &first, const QuarkRead &second) const {
  ConjugatePairRead out;

  const bool windingFirst = first.determinantWinding.has_value();
  const bool windingSecond = second.determinantWinding.has_value();
  gate(windingFirst, "winding-first", out.failedCertificates);
  gate(windingSecond, "winding-second", out.failedCertificates);
  if (windingFirst && windingSecond)
    out.totalWinding = *first.determinantWinding + *second.determinantWinding;
  if (first.baryonFlux.has_value() && second.baryonFlux.has_value())
    out.totalBaryonFlux = *first.baryonFlux + *second.baryonFlux;

  const bool parityFirst = first.exteriorParity != 0;
  const bool paritySecond = second.exteriorParity != 0;
  gate(parityFirst, "parity-first", out.failedCertificates);
  gate(paritySecond, "parity-second", out.failedCertificates);
  if (parityFirst && paritySecond) {
    out.totalParity = first.exteriorParity * second.exteriorParity;
    out.parityEven = (out.totalParity == +1);
  }

  const bool windingConserved =
      out.totalWinding.has_value() && *out.totalWinding == 0;
  gate(windingConserved, "winding-conservation", out.failedCertificates);
  gate(out.parityEven, "parity-even", out.failedCertificates);

  out.conserved = windingConserved && out.parityEven;
  if (out.conserved) {
    // Integer sums of certified integers: exact given the premises.
    out.certificate = Certificate::structureExact(
        CertificateDomain::Static, CertificateRegime::NonNormal,
        /*residual=*/0.0, /*conditioning=*/kNaN, /*tolerance=*/1e-15);
  } else {
    out.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::Static, CertificateRegime::NonNormal);
  }
  return out;
}

// ---------------------------------------------------------------------------
// the emergent flavor doublet
// ---------------------------------------------------------------------------

FlavorDoubletRead ParticleClusters::flavorDoubletSearch(
    const std::vector<ComponentBandRead> &frames) const {
  FlavorDoubletRead out;
  const auto fail = [&](const char *reason) {
    out.found = false;
    out.invalidationReason = reason;
    out.failedCertificates.emplace_back("flavor-doublet");
    out.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::BandWindow, CertificateRegime::NonNormal);
    return out;
  };

  if (frames.size() < cfg_.minDoubletFrames || frames.size() < 2)
    return fail("insufficient-frames");

  // One chain per frame-0 fiber, extended across consecutive frames through
  // certified continuations only (SpectralFiberTracker::matchFibers).
  struct Chain {
    std::vector<std::size_t> positions;  // fiber index per covered frame
    double minOverlap = 1.0;
    bool alive = true;
  };
  std::vector<Chain> chains(frames[0].fibers.size());
  for (std::size_t i = 0; i < chains.size(); ++i)
    chains[i].positions = {i};

  for (std::size_t t = 0; t + 1 < frames.size(); ++t) {
    const std::vector<FiberMatchRead> matches =
        SpectralFiberTracker::matchFibers(frames[t].fibers,
                                          frames[t + 1].fibers,
                                          cfg_.doubletOverlapThreshold);
    std::unordered_map<std::size_t, const FiberMatchRead *> bestByFrom;
    for (const FiberMatchRead &match : matches)
      bestByFrom[match.fromIndex] = &match;

    // Tentative targets; two chains merging onto one band invalidate each
    // other (an ambiguous continuation is no continuation).
    std::unordered_map<std::size_t, std::size_t> claims;
    for (const Chain &chain : chains) {
      if (!chain.alive) continue;
      const auto it = bestByFrom.find(chain.positions.back());
      if (it != bestByFrom.end() && it->second->certifiedContinuation)
        ++claims[it->second->toIndex];
    }
    for (Chain &chain : chains) {
      if (!chain.alive) continue;
      const auto it = bestByFrom.find(chain.positions.back());
      if (it == bestByFrom.end() || !it->second->certifiedContinuation ||
          claims[it->second->toIndex] > 1) {
        chain.alive = false;
        continue;
      }
      chain.positions.push_back(it->second->toIndex);
      chain.minOverlap =
          std::min(chain.minOverlap, it->second->overlap.subspaceOverlap);
    }
  }

  // Stable subclasses are the full-length chains; ranks are an outcome.
  std::vector<std::size_t> stableChains;
  for (std::size_t c = 0; c < chains.size(); ++c)
    if (chains[c].alive && chains[c].positions.size() == frames.size())
      stableChains.push_back(c);
  for (const std::size_t c : stableChains)
    out.stableSubclassRanks.push_back(frames[0].fibers[c].rank());

  std::vector<std::size_t> twoState;
  for (const std::size_t c : stableChains)
    if (frames[0].fibers[c].rank() == 2) twoState.push_back(c);
  out.twoStateCount = twoState.size();

  if (twoState.empty()) return fail("no-stable-two-state-subclass");
  if (twoState.size() > 1) return fail("ambiguous-two-state-subclasses");

  const Chain &winner = chains[twoState.front()];
  const SpectralFiber &doublet = frames[0].fibers[twoState.front()];
  out.found = true;
  out.degree = doublet.degree();
  out.rank = doublet.rank();
  out.framesTracked = frames.size();
  out.minContinuationOverlap = winner.minOverlap;
  double minIsolation = std::numeric_limits<double>::infinity();
  for (std::size_t t = 0; t < frames.size(); ++t) {
    const SpectralBandCertificate &cert =
        frames[t].fibers[winner.positions[t]].certificate();
    minIsolation = std::min(minIsolation, cert.nearestDiscardedSeparation);
  }
  out.minIsolation = minIsolation;
  out.doublet = doublet;
  out.certificate = Certificate::certifiedNumerical(
      CertificateDomain::BandWindow, doublet.certificate().certificate.regime(),
      /*residual=*/1.0 - winner.minOverlap,
      /*conditioning=*/doublet.certificate().projectorNorm,
      /*tolerance=*/1.0 - cfg_.doubletOverlapThreshold);
  return out;
}

// ---------------------------------------------------------------------------
// the reused Gauss-flux electric read
// ---------------------------------------------------------------------------

GaussFluxRead ParticleClusters::gaussFluxOnSurfaces(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<std::complex<double>> &fieldStrength,
    const std::vector<std::vector<std::uint64_t>> &enclosedVertexSets,
    bool electricOnly) const {
  if (enclosedVertexSets.empty())
    throw std::invalid_argument(
        "ParticleClusters::gaussFluxOnSurfaces: at least one enclosing "
        "surface is required");
  // The degree-2 Gauss read is consumed verbatim; constructing the reader
  // mutates nothing.
  cobordism::EigenstateSynthesis reader(st, 2);
  std::vector<cd> fluxes;
  std::vector<std::size_t> counts;
  fluxes.reserve(enclosedVertexSets.size());
  counts.reserve(enclosedVertexSets.size());
  for (const auto &enclosed : enclosedVertexSets) {
    fluxes.push_back(
        reader.gaussLawCharge(fieldStrength, enclosed, electricOnly));
    const std::set<std::uint64_t> unique(enclosed.begin(), enclosed.end());
    counts.push_back(unique.size());
  }
  return gaussFluxConsistency(fluxes, counts, electricOnly);
}

GaussFluxRead ParticleClusters::gaussFluxConsistency(
    const std::vector<std::complex<double>> &fluxes,
    const std::vector<std::size_t> &surfaceVertexCounts,
    bool electricOnly) const {
  if (fluxes.empty())
    throw std::invalid_argument(
        "ParticleClusters::gaussFluxConsistency: at least one flux is "
        "required");
  GaussFluxRead out;
  out.fluxes = fluxes;
  out.surfaceVertexCounts = surfaceVertexCounts;
  out.electricOnly = electricOnly;

  double maxDeviation = 0.0;
  double imagLeakage = 0.0;
  for (std::size_t i = 0; i < fluxes.size(); ++i) {
    imagLeakage = std::max(imagLeakage, std::abs(fluxes[i].imag()));
    for (std::size_t j = i + 1; j < fluxes.size(); ++j)
      maxDeviation = std::max(maxDeviation, std::abs(fluxes[i] - fluxes[j]));
  }
  out.maxDeviation = maxDeviation;
  out.imagLeakage = imagLeakage;

  out.consistent = fluxes.size() >= cfg_.minEnclosingSurfaces &&
                   maxDeviation <= cfg_.gaussTolerance &&
                   imagLeakage <= cfg_.gaussTolerance;
  if (out.consistent) {
    double mean = 0.0;
    for (const cd &flux : fluxes) mean += flux.real();
    out.electricFlux = mean / static_cast<double>(fluxes.size());
    // The per-surface sums are exact signed sums of the supplied cochain;
    // the consistency residual is what is graded.
    out.certificate = Certificate::algebraicallyExact(
        CertificateDomain::Static, CertificateRegime::NonNormal,
        std::max(maxDeviation, imagLeakage), cfg_.gaussTolerance);
  } else {
    out.failedCertificates.emplace_back("gauss-consistency");
    out.certificate = Certificate::heuristicDiscovery(
        CertificateDomain::Static, CertificateRegime::NonNormal);
  }
  return out;
}

std::vector<std::vector<std::uint64_t>> ParticleClusters::nestedEnclosures(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<std::uint64_t> &seedVertexIds, std::size_t shells) {
  if (shells < 1)
    throw std::invalid_argument(
        "ParticleClusters::nestedEnclosures: shells must be >= 1");
  if (seedVertexIds.empty())
    throw std::invalid_argument(
        "ParticleClusters::nestedEnclosures: empty seed");

  std::unordered_set<std::uint64_t> present;
  for (const auto v : st->getVertexList()->toVector())
    if (v != nullptr) present.insert(v->getId());

  std::set<std::uint64_t> current;
  for (const std::uint64_t id : seedVertexIds)
    if (present.count(id)) current.insert(id);
  if (current.empty())
    throw std::invalid_argument(
        "ParticleClusters::nestedEnclosures: no seed vertex exists in the "
        "complex");

  // Undirected one-skeleton adjacency (read-only edge sweep).
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adjacency;
  for (const auto e : st->getEdgeList()->toVector()) {
    if (e == nullptr) continue;
    const auto s = e->getSource();
    const auto t = e->getTarget();
    if (s == nullptr || t == nullptr) continue;
    if (s->getId() == t->getId()) continue;
    adjacency[s->getId()].push_back(t->getId());
    adjacency[t->getId()].push_back(s->getId());
  }

  std::vector<std::vector<std::uint64_t>> sets;
  sets.reserve(shells);
  sets.emplace_back(current.begin(), current.end());
  for (std::size_t k = 1; k < shells; ++k) {
    std::set<std::uint64_t> next = current;
    for (const std::uint64_t id : current) {
      const auto it = adjacency.find(id);
      if (it == adjacency.end()) continue;
      next.insert(it->second.begin(), it->second.end());
    }
    current = std::move(next);
    sets.emplace_back(current.begin(), current.end());
  }
  return sets;
}

// ---------------------------------------------------------------------------
// candidate tracking across scale/time
// ---------------------------------------------------------------------------

std::vector<FiberMatchRead> ParticleClusters::trackCandidates(
    const std::vector<QuarkCandidateEvidence> &from,
    const std::vector<QuarkCandidateEvidence> &to, double overlapThreshold) {
  std::vector<SpectralFiber> fromBands;
  std::vector<SpectralFiber> toBands;
  fromBands.reserve(from.size());
  toBands.reserve(to.size());
  for (const QuarkCandidateEvidence &evidence : from)
    fromBands.push_back(evidence.colorBand);
  for (const QuarkCandidateEvidence &evidence : to)
    toBands.push_back(evidence.colorBand);
  return SpectralFiberTracker::matchFibers(fromBands, toBands,
                                           overlapThreshold);
}

// ---------------------------------------------------------------------------
// proposing cluster supports: modularity and the degree-zero band
// ---------------------------------------------------------------------------

namespace {

// A support as an ascending, deduplicated set of cell ids.
std::vector<std::uint64_t> canonicalSupport(
    const std::vector<std::uint64_t> &support) {
  std::vector<std::uint64_t> out(support);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// |A n B| / |A u B| for two ascending, deduplicated sets; 0 for two empty
// sets, which share nothing to agree on.
double jaccard(const std::vector<std::uint64_t> &a,
               const std::vector<std::uint64_t> &b) {
  if (a.empty() || b.empty()) return 0.0;
  std::vector<std::uint64_t> shared;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(shared));
  const std::size_t united = a.size() + b.size() - shared.size();
  return static_cast<double>(shared.size()) / static_cast<double>(united);
}

}  // namespace

std::vector<ClusterSupportProposal> ParticleClusters::proposeSupports(
    const std::vector<ComponentRead> &modularityComponents,
    const EffectiveComponentPartition &bandComponents) {
  std::vector<std::vector<std::uint64_t>> fromModularity;
  std::vector<std::size_t> modularityIndices;
  for (std::size_t i = 0; i < modularityComponents.size(); ++i) {
    std::vector<std::uint64_t> support =
        canonicalSupport(modularityComponents[i].support);
    if (support.empty()) continue;
    fromModularity.push_back(std::move(support));
    modularityIndices.push_back(i);
  }
  std::vector<std::vector<std::uint64_t>> fromBand;
  std::vector<std::size_t> bandIndices;
  for (std::size_t i = 0; i < bandComponents.components.size(); ++i) {
    std::vector<std::uint64_t> support =
        canonicalSupport(bandComponents.components[i].support);
    if (support.empty()) continue;
    fromBand.push_back(std::move(support));
    bandIndices.push_back(i);
  }

  std::vector<ClusterSupportProposal> proposals;
  std::vector<bool> bandMatched(fromBand.size(), false);
  for (std::size_t i = 0; i < fromModularity.size(); ++i) {
    ClusterSupportProposal proposal;
    proposal.support = fromModularity[i];
    proposal.modularity = true;
    proposal.modularityIndex = modularityIndices[i];
    for (std::size_t j = 0; j < fromBand.size(); ++j) {
      proposal.crossProposerOverlap = std::max(
          proposal.crossProposerOverlap, jaccard(proposal.support, fromBand[j]));
      if (proposal.band || fromBand[j] != proposal.support) continue;
      proposal.band = true;
      proposal.bandIndex = bandIndices[j];
      bandMatched[j] = true;
    }
    proposals.push_back(std::move(proposal));
  }
  for (std::size_t j = 0; j < fromBand.size(); ++j) {
    if (bandMatched[j]) continue;
    ClusterSupportProposal proposal;
    proposal.support = fromBand[j];
    proposal.band = true;
    proposal.bandIndex = bandIndices[j];
    for (const std::vector<std::uint64_t> &other : fromModularity)
      proposal.crossProposerOverlap =
          std::max(proposal.crossProposerOverlap, jaccard(proposal.support, other));
    proposals.push_back(std::move(proposal));
  }
  return proposals;
}

// ===========================================================================
// Even sectors: the quasi-free octet bilinear read and the gluon, meson and
// diquark candidate classifiers.  File-local helpers live in the single
// anonymous namespace at the top of this file.
// ===========================================================================

// ---------------------------------------------------------------------------
// OctetBilinearRead
// ---------------------------------------------------------------------------

std::string OctetBilinearRead::describe() const {
  std::ostringstream out;
  out << "OctetBilinearRead[modes (";
  for (std::size_t i = 0; i < colorModes.size(); ++i)
    out << (i ? "," : "") << colorModes[i];
  out << "), N=" << occupation << ", parity=" << subsetParity
      << ", octet " << octetWeight << ", singlet " << singletWeight
      << ", casimir " << casimir << "]";
  return out.str();
}

Record OctetBilinearRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("octet_bilinear_read");
  Record::List modes;
  modes.reserve(colorModes.size());
  for (const std::size_t mode : colorModes)
    modes.emplace_back(static_cast<std::int64_t>(mode));
  m["color_modes"] = Record(std::move(modes));
  m["occupation"] = Record(occupation);
  m["subset_parity"] = Record(subsetParity);
  matrix3ToRecord(m, "bilinear", bilinear);
  matrix3ToRecord(m, "octet_component", octetComponent);
  m["octet_weight"] = Record(octetWeight);
  m["singlet_weight"] = Record(singletWeight);
  m["octet_projector_residual"] = Record(octetProjectorResidual);
  m["casimir"] = Record(casimir);
  m["casimir_expectation"] = Record(casimirExpectation);
  Record::splitComplex(m, "gell_mann_components", gellMannComponents);
  m["residual"] = Record(residual);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

OctetBilinearRead OctetBilinearRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "octet_bilinear_read");
  OctetBilinearRead read;
  for (const auto &entry : m.at("color_modes").asList())
    read.colorModes.push_back(static_cast<std::size_t>(entry.asInt()));
  read.occupation = m.at("occupation").asDouble();
  read.subsetParity = static_cast<int>(m.at("subset_parity").asInt());
  read.bilinear = matrix3FromRecord(m, "bilinear");
  read.octetComponent = matrix3FromRecord(m, "octet_component");
  read.octetWeight = m.at("octet_weight").asDouble();
  read.singletWeight = m.at("singlet_weight").asDouble();
  read.octetProjectorResidual = m.at("octet_projector_residual").asDouble();
  read.casimir = m.at("casimir").asDouble();
  read.casimirExpectation = m.at("casimir_expectation").asDouble();
  {
    const auto &re = m.at("gell_mann_components_re").asList();
    const auto &im = m.at("gell_mann_components_im").asList();
    if (re.size() != im.size())
      throw std::invalid_argument(
          "ParticleClusters: gell_mann_components payload size mismatch");
    for (std::size_t i = 0; i < re.size(); ++i)
      read.gellMannComponents.emplace_back(re[i].asDouble(),
                                           im[i].asDouble());
  }
  read.residual = m.at("residual").asDouble();
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

OctetBilinearRead ParticleClusters::octetBilinearRead(
    const quantum::CovarianceState &state,
    const std::vector<std::size_t> &colorModes) const {
  if (colorModes.size() != 3)
    throw std::invalid_argument(
        "ParticleClusters::octetBilinearRead: exactly three color modes "
        "are required, got " +
        std::to_string(colorModes.size()));
  const std::set<std::size_t> unique(colorModes.begin(), colorModes.end());
  if (unique.size() != 3)
    throw std::invalid_argument(
        "ParticleClusters::octetBilinearRead: color modes must be "
        "distinct");
  for (const std::size_t mode : colorModes) {
    if (mode >= state.modeCount())
      throw std::invalid_argument(
          "ParticleClusters::octetBilinearRead: color mode " +
          std::to_string(mode) + " is out of range for " +
          std::to_string(state.modeCount()) + " modes");
  }

  OctetBilinearRead read;
  read.colorModes = colorModes;

  // Residual/tolerance accumulation over every consumed Wick read.
  ConsumedCertificates consumed;

  // The bilinear matrix M_ij = ⟨a_i†a_j⟩ = Γ_{m_j m_i} on the declared
  // modes; the covariance stores Γ_ij = ⟨a_j†a_i⟩.
  const Eigen::MatrixXcd &gamma = state.gamma();
  for (Eigen::Index i = 0; i < 3; ++i)
    for (Eigen::Index j = 0; j < 3; ++j)
      read.bilinear(i, j) =
          gamma(static_cast<Eigen::Index>(
                    colorModes[static_cast<std::size_t>(j)]),
                static_cast<Eigen::Index>(
                    colorModes[static_cast<std::size_t>(i)]));

  // Subset occupation ⟨N_S⟩: the sum of the three certified per-mode
  // occupation reads (each an exact Wick value).
  {
    cd total(0.0, 0.0);
    bool certified = true;
    for (const std::size_t mode : colorModes) {
      const quantum::WickCertificateRead occ = state.wickOccupation(mode);
      certified = certified && occ.certificate.holds();
      total += occ.value;
      consumed.consume(occ.certificate);
    }
    consumed.consumeResidual(std::abs(total.imag()));
    read.occupation = certified ? total.real() : kNaN;
  }

  // Subset parity ⟨(−1)^{N_S}⟩ = det(I_S − 2Γ_S).
  const quantum::WickCertificateRead parity =
      state.wickSubsetParity(colorModes);
  consumed.consume(parity.certificate);
  read.subsetParity = certifiedSign(parity, cfg_.parityTolerance);

  // The exact 1 ⊕ 8 resolution, delegated to the ColorFiber kernel.
  read.octetComponent = ColorFiber::tracelessPart(read.bilinear);
  const ColorFiber::OctetRead weights = ColorFiber::octetRead(read.bilinear);
  read.octetWeight = weights.octet;
  read.singletWeight = weights.singlet;

  const double octetNorm = read.octetComponent.norm();
  if (octetNorm > 0.0) {
    const Eigen::VectorXcd vec =
        Eigen::Map<const Eigen::VectorXcd>(read.octetComponent.data(), 9);
    // The singlet complement is the ColorFiber projector, so the bitwise
    // P₁ + P₈ = I₉ identity stays single-sourced.
    read.octetProjectorResidual =
        (ColorFiber::adjointSingletProjector() * vec).norm() / octetNorm;
    read.casimir = ColorFiber::adjointCasimir(read.octetComponent);
  }

  // The quartic-Wick color Casimir ⟨Σ_a dΓ(λ_a/2)²⟩: the Gell-Mann halves are
  // embedded on the declared modes of the full mode space, leaving every other
  // mode untouched, so adding microscopic modes never changes this read.
  {
    const std::size_t modeCount = state.modeCount();
    cd total(0.0, 0.0);
    bool certified = true;
    for (int a = 1; a <= 8; ++a) {
      const Eigen::Matrix3cd half = 0.5 * ColorFiber::gellMann(a);
      Eigen::MatrixXcd embedded = Eigen::MatrixXcd::Zero(
          static_cast<Eigen::Index>(modeCount),
          static_cast<Eigen::Index>(modeCount));
      for (Eigen::Index i = 0; i < 3; ++i)
        for (Eigen::Index j = 0; j < 3; ++j)
          embedded(static_cast<Eigen::Index>(
                       colorModes[static_cast<std::size_t>(i)]),
                   static_cast<Eigen::Index>(
                       colorModes[static_cast<std::size_t>(j)])) =
              half(i, j);
      const quantum::WickCertificateRead moment =
          state.wickBilinearMoment({embedded, embedded});
      certified = certified && moment.certificate.holds();
      total += moment.value;
      consumed.consume(moment.certificate);
    }
    consumed.consumeResidual(std::abs(total.imag()));
    read.casimirExpectation = certified ? total.real() : kNaN;
  }

  // The octet coordinates Tr(λ_a M)/2 (Tr λ_aλ_b = 2δ_ab):
  // M = (Tr M/3) I + Σ_a comp_a λ_a exactly.
  read.gellMannComponents.reserve(8);
  for (int a = 1; a <= 8; ++a)
    read.gellMannComponents.push_back(
        0.5 * (ColorFiber::gellMann(a) * read.bilinear).trace());

  read.residual = consumed.residual();
  // Exact Wick algebra on the covariance: AlgebraicallyExact in the verified
  // regime of the consumed reads, graded by the measured residual against the
  // read tolerance.
  read.certificate = Certificate::algebraicallyExact(
      CertificateDomain::Static, parity.certificate.regime(),
      consumed.residualOrZero(), consumed.tolerance());
  return read;
}

OctetBilinearRead ParticleClusters::octetBilinearReadCached(
    cobordism::AnalyticCache &cache,
    const std::vector<std::uint64_t> &componentVertexIds,
    const quantum::CovarianceState &state,
    const std::vector<std::size_t> &colorModes) const {
  const auto parameter =
      static_cast<std::int64_t>(octetFingerprint(state, colorModes));
  if (const auto payload =
          cache.fetch(componentVertexIds, kOctetCacheKind, parameter))
    return *std::static_pointer_cast<const OctetBilinearRead>(payload);
  OctetBilinearRead read = octetBilinearRead(state, colorModes);
  cache.store(componentVertexIds, kOctetCacheKind, parameter,
              std::make_shared<OctetBilinearRead>(read), read.certificate);
  return read;
}

std::uint64_t ParticleClusters::octetFingerprint(
    const quantum::CovarianceState &state,
    const std::vector<std::size_t> &colorModes) const {
  std::uint64_t h = 0x9e3779b97f4a7c15ull;
  // A Γ change is a state change: the exact bit-pattern hash travels in
  // the parameter, so a stale-Γ payload can only cause recomputation.
  h = hashString(h, state.covarianceHash());
  h = chainHash(h, colorModes.size());
  for (const std::size_t mode : colorModes) h = chainHash(h, mode);
  // The decision thresholds the read applies (sign extraction).
  h = hashDouble(h, cfg_.parityTolerance);
  return h;
}

// ---------------------------------------------------------------------------
// GluonRead
// ---------------------------------------------------------------------------

std::string GluonRead::describe() const {
  std::ostringstream out;
  out << "GluonRead[" << classification;
  if (determinantWinding.has_value())
    out << ", nu=" << *determinantWinding << " (" << windingClosure << ")";
  else
    out << ", nu=unknown";
  if (baryonFlux.has_value())
    out << ", B=" << *baryonFlux;
  else
    out << ", B=unknown";
  out << ", parity=" << exteriorParity << ", octet " << octetWeight
      << ", casimir " << casimir << ", confidence " << confidence << "]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

Record GluonRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("gluon_read");
  componentToRecord(m, "component", component);
  componentToRecord(m, "binding_component", bindingComponent);
  m["classification"] = Record(classification);
  m["exterior_parity"] = Record(exteriorParity);
  m["occupation_total"] = Record(occupationTotal);
  m["casimir"] = Record(casimir);
  m["casimir_expectation"] = Record(casimirExpectation);
  m["octet_projector_residual"] = Record(octetProjectorResidual);
  m["octet_weight"] = Record(octetWeight);
  m["singlet_weight"] = Record(singletWeight);
  m["determinant_winding"] = optionalInt(determinantWinding);
  m["winding_closure"] = Record(windingClosure);
  m["winding_reference_id"] = Record(windingReferenceId);
  m["baryon_flux"] = optionalDouble(baryonFlux);
  m["transport_count"] = Record(static_cast<std::int64_t>(transportCount));
  m["transport_leakage_max"] = Record(transportLeakageMax);
  m["persistence_lifetime"] = Record(persistenceLifetime);
  m["frame_lifetime"] = Record(frameLifetime);
  m["confidence"] = Record(confidence);
  Record::List failed;
  failed.reserve(failedCertificates.size());
  for (const auto &name : failedCertificates) failed.emplace_back(name);
  m["failed_certificates"] = Record(std::move(failed));
  m["thresholds"] = thresholdsToRecord(thresholds);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

GluonRead GluonRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "gluon_read");
  GluonRead read;
  read.component = componentFromRecord(m, "component");
  read.bindingComponent = componentFromRecord(m, "binding_component");
  read.classification = m.at("classification").asString();
  read.exteriorParity = static_cast<int>(m.at("exterior_parity").asInt());
  read.occupationTotal = m.at("occupation_total").asDouble();
  read.casimir = m.at("casimir").asDouble();
  read.casimirExpectation = m.at("casimir_expectation").asDouble();
  read.octetProjectorResidual =
      m.at("octet_projector_residual").asDouble();
  read.octetWeight = m.at("octet_weight").asDouble();
  read.singletWeight = m.at("singlet_weight").asDouble();
  read.determinantWinding = optionalIntFrom(m.at("determinant_winding"));
  read.windingClosure = m.at("winding_closure").asString();
  read.windingReferenceId = m.at("winding_reference_id").asString();
  read.baryonFlux = optionalDoubleFrom(m.at("baryon_flux"));
  read.transportCount =
      static_cast<std::size_t>(m.at("transport_count").asInt());
  read.transportLeakageMax = m.at("transport_leakage_max").asDouble();
  read.persistenceLifetime = m.at("persistence_lifetime").asDouble();
  read.frameLifetime = optionalLeaf(m, "frame_lifetime");
  read.confidence = m.at("confidence").asDouble();
  for (const auto &entry : m.at("failed_certificates").asList())
    read.failedCertificates.push_back(entry.asString());
  read.thresholds = thresholdsFromRecord(m.at("thresholds"));
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

GluonRead ParticleClusters::classifyGluon(
    const GluonCandidateEvidence &evidence) const {
  GluonRead read;
  read.component = evidence.component;
  read.bindingComponent = evidence.bindingComponent;
  read.thresholds = cfg_;
  // Flat scalar summaries of the octet evidence; the full OctetBilinearRead
  // lives on the evidence and serializes itself.
  read.casimir = evidence.octet.casimir;
  read.casimirExpectation = evidence.octet.casimirExpectation;
  read.octetProjectorResidual = evidence.octet.octetProjectorResidual;
  read.octetWeight = evidence.octet.octetWeight;
  read.singletWeight = evidence.octet.singletWeight;

  std::vector<std::string> failed;
  int passed = 0;
  constexpr int kGates = 6;

  // 1. even carried-state parity from the Wick parity read; an uncertified
  //    read emits no sign.
  read.exteriorParity =
      certifiedSign(evidence.parityRead, cfg_.parityTolerance);
  passed += gate(read.exteriorParity == +1, "parity-even", failed);

  // ⟨N⟩ is reported, never gated.
  read.occupationTotal = evidence.occupationRead.certificate.holds()
                             ? evidence.occupationRead.value.real()
                             : kNaN;

  // 2. a certified, genuinely nonzero octet excitation.
  const bool excitationOk = evidence.octet.certificate.holds() &&
                            std::isfinite(evidence.octet.octetWeight) &&
                            evidence.octet.octetWeight >= cfg_.minOctetWeight;
  passed += gate(excitationOk, "octet-excitation", failed);

  // 3. machine-level octet purity of the excitation: the traceless bilinear
  //    lies in the 8 exactly, so the residual is rounding.
  const bool purityOk =
      std::isfinite(evidence.octet.octetProjectorResidual) &&
      evidence.octet.octetProjectorResidual <= cfg_.octetPurityTolerance;
  passed += gate(purityOk, "octet-purity", failed);

  // 4. accepted rank-three transports under the leakage cap; the adjoint
  //    (octet) action is exact given the accepted fundamental factor.
  read.transportCount = evidence.lifetimeTransports.size();
  bool transportsOk = !evidence.lifetimeTransports.empty();
  double maxLeakage = kNaN;
  for (const FiberTransportRead &transport : evidence.lifetimeTransports) {
    transportsOk = transportsOk && transport.accepted && transport.rank == 3;
    maxLeakage = maxFinite(maxLeakage, transport.leakage);
  }
  read.transportLeakageMax = maxLeakage;
  transportsOk = transportsOk && std::isfinite(maxLeakage) &&
                 maxLeakage <= cfg_.maxTransportLeakage;
  passed += gate(transportsOk, "octet-transport", failed);

  // 5. certified zero determinant winding.  A certified ν = 0 is evidence of
  //    zero baryon flux, never a default; an unknown winding leaves the flux
  //    unknown.
  const DeterminantWindingRead &winding = evidence.winding;
  read.windingClosure = winding.windingClosure;
  read.windingReferenceId = winding.windingReferenceId;
  const bool windingCertified =
      winding.winding.has_value() && winding.certificate.holds();
  if (windingCertified) {
    read.determinantWinding = winding.winding;
    read.baryonFlux = static_cast<double>(*winding.winding) / 3.0;
  }
  passed += gate(windingCertified && *winding.winding == 0, "winding-zero",
                 failed);

  // 6. persistence, gated on the cobordism-frame lifetime as in the quark
  //    classifier; the modularity resolution-slice count is reported.
  read.persistenceLifetime = evidence.persistenceLifetime;
  read.frameLifetime = evidence.frameLifetime;
  const bool persistenceOk =
      std::isfinite(evidence.frameLifetime) &&
      evidence.frameLifetime >= cfg_.minPersistenceLifetime;
  passed += gate(persistenceOk, "persistence", failed);

  read.confidence = static_cast<double>(passed) / kGates;
  read.classification = (passed == kGates) ? "gluon-candidate" : "none";
  read.failedCertificates = std::move(failed);

  ConsumedCertificates consumed;
  consumed.consume(evidence.parityRead.certificate);
  consumed.consume(evidence.octet.certificate);
  for (const FiberTransportRead &transport : evidence.lifetimeTransports)
    consumed.consume(transport.certificate);
  consumed.consume(winding.certificate);
  read.certificate = consumed.verdict(read.classification != "none",
                                      evidence.octet.certificate);
  return read;
}

// ---------------------------------------------------------------------------
// MesonRead / DiquarkRead — the two-cluster even composites
// ---------------------------------------------------------------------------

std::string MesonRead::describe() const {
  std::ostringstream out;
  out << "MesonRead[" << classification << ", parity=" << exteriorParity;
  if (totalWinding.has_value())
    out << ", nu_total=" << *totalWinding;
  else
    out << ", nu_total=unknown";
  if (totalBaryonFlux.has_value())
    out << ", B_total=" << *totalBaryonFlux;
  else
    out << ", B_total=unknown";
  out << ", octet fraction " << pairingOctetFraction << ", confidence "
      << confidence << "]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

Record MesonRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("meson_read");
  componentToRecord(m, "binding_component", bindingComponent);
  componentToRecord(m, "first_constituent", firstConstituent);
  componentToRecord(m, "second_constituent", secondConstituent);
  m["classification"] = Record(classification);
  m["exterior_parity"] = Record(exteriorParity);
  m["occupation_total"] = Record(occupationTotal);
  m["pairing_singlet_weight"] = Record(pairingSingletWeight);
  m["pairing_octet_weight"] = Record(pairingOctetWeight);
  m["pairing_octet_fraction"] = Record(pairingOctetFraction);
  m["total_winding"] = optionalInt(totalWinding);
  m["total_baryon_flux"] = optionalDouble(totalBaryonFlux);
  m["transport_count"] = Record(static_cast<std::int64_t>(transportCount));
  m["transport_leakage_max"] = Record(transportLeakageMax);
  m["persistence_lifetime"] = Record(persistenceLifetime);
  m["confidence"] = Record(confidence);
  Record::List failed;
  failed.reserve(failedCertificates.size());
  for (const auto &name : failedCertificates) failed.emplace_back(name);
  m["failed_certificates"] = Record(std::move(failed));
  m["thresholds"] = thresholdsToRecord(thresholds);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

MesonRead MesonRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "meson_read");
  MesonRead read;
  read.bindingComponent = componentFromRecord(m, "binding_component");
  read.firstConstituent = componentFromRecord(m, "first_constituent");
  read.secondConstituent = componentFromRecord(m, "second_constituent");
  read.classification = m.at("classification").asString();
  read.exteriorParity = static_cast<int>(m.at("exterior_parity").asInt());
  read.occupationTotal = m.at("occupation_total").asDouble();
  read.pairingSingletWeight = m.at("pairing_singlet_weight").asDouble();
  read.pairingOctetWeight = m.at("pairing_octet_weight").asDouble();
  read.pairingOctetFraction = m.at("pairing_octet_fraction").asDouble();
  read.totalWinding = optionalIntFrom(m.at("total_winding"));
  read.totalBaryonFlux = optionalDoubleFrom(m.at("total_baryon_flux"));
  read.transportCount =
      static_cast<std::size_t>(m.at("transport_count").asInt());
  read.transportLeakageMax = m.at("transport_leakage_max").asDouble();
  read.persistenceLifetime = m.at("persistence_lifetime").asDouble();
  read.confidence = m.at("confidence").asDouble();
  for (const auto &entry : m.at("failed_certificates").asList())
    read.failedCertificates.push_back(entry.asString());
  read.thresholds = thresholdsFromRecord(m.at("thresholds"));
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

std::string DiquarkRead::describe() const {
  std::ostringstream out;
  out << "DiquarkRead[" << classification << ", parity=" << exteriorParity;
  if (totalWinding.has_value())
    out << ", nu_total=" << *totalWinding;
  else
    out << ", nu_total=unknown";
  if (totalBaryonFlux.has_value())
    out << ", B_total=" << *totalBaryonFlux;
  else
    out << ", B_total=unknown";
  out << ", anti-triplet " << antiTripletWeight << ", confidence "
      << confidence << "]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

Record DiquarkRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("diquark_read");
  componentToRecord(m, "binding_component", bindingComponent);
  componentToRecord(m, "first_constituent", firstConstituent);
  componentToRecord(m, "second_constituent", secondConstituent);
  m["classification"] = Record(classification);
  m["exterior_parity"] = Record(exteriorParity);
  m["occupation_total"] = Record(occupationTotal);
  m["anti_triplet_weight"] = Record(antiTripletWeight);
  m["total_winding"] = optionalInt(totalWinding);
  m["total_baryon_flux"] = optionalDouble(totalBaryonFlux);
  m["transport_count"] = Record(static_cast<std::int64_t>(transportCount));
  m["transport_leakage_max"] = Record(transportLeakageMax);
  m["persistence_lifetime"] = Record(persistenceLifetime);
  m["confidence"] = Record(confidence);
  Record::List failed;
  failed.reserve(failedCertificates.size());
  for (const auto &name : failedCertificates) failed.emplace_back(name);
  m["failed_certificates"] = Record(std::move(failed));
  m["thresholds"] = thresholdsToRecord(thresholds);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

DiquarkRead DiquarkRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "diquark_read");
  DiquarkRead read;
  read.bindingComponent = componentFromRecord(m, "binding_component");
  read.firstConstituent = componentFromRecord(m, "first_constituent");
  read.secondConstituent = componentFromRecord(m, "second_constituent");
  read.classification = m.at("classification").asString();
  read.exteriorParity = static_cast<int>(m.at("exterior_parity").asInt());
  read.occupationTotal = m.at("occupation_total").asDouble();
  read.antiTripletWeight = m.at("anti_triplet_weight").asDouble();
  read.totalWinding = optionalIntFrom(m.at("total_winding"));
  read.totalBaryonFlux = optionalDoubleFrom(m.at("total_baryon_flux"));
  read.transportCount =
      static_cast<std::size_t>(m.at("transport_count").asInt());
  read.transportLeakageMax = m.at("transport_leakage_max").asDouble();
  read.persistenceLifetime = m.at("persistence_lifetime").asDouble();
  read.confidence = m.at("confidence").asDouble();
  for (const auto &entry : m.at("failed_certificates").asList())
    read.failedCertificates.push_back(entry.asString());
  read.thresholds = thresholdsFromRecord(m.at("thresholds"));
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

MesonRead ParticleClusters::classifyMeson(
    const CompositeCandidateEvidence &evidence) const {
  MesonRead read;
  read.thresholds = cfg_;
  fillCompositeReports(read, evidence);

  std::vector<std::string> failed;
  int passed = 0;
  constexpr int kGates = 5;

  // 1/2. one certified quark and one certified antiquark; the constituent
  //      verdicts are consumed verbatim and the order does not matter.
  const bool hasQuark = certifiedConstituent(evidence.first, "quark") ||
                        certifiedConstituent(evidence.second, "quark");
  const bool hasAntiquark =
      certifiedConstituent(evidence.first, "antiquark") ||
      certifiedConstituent(evidence.second, "antiquark");
  passed += gate(hasQuark, "constituent-quark", failed);
  passed += gate(hasAntiquark, "constituent-antiquark", failed);

  // 3. even composite parity: the graded product of the certified
  //    constituent parities (parity adds mod 2).
  read.exteriorParity = compositeParity(evidence.first, evidence.second);
  passed += gate(read.exteriorParity == +1, "parity-even", failed);

  // 4. color singlet: the exact 1 ⊕ 8 split of the pair bilinear
  //    (ColorFiber::octetRead).
  bool singletOk = false;
  if (evidence.colorPairing.has_value()) {
    const ColorFiber::OctetRead weights =
        ColorFiber::octetRead(*evidence.colorPairing);
    read.pairingSingletWeight = weights.singlet;
    read.pairingOctetWeight = weights.octet;
    const double total = weights.singlet + weights.octet;
    if (total > 0.0) {
      read.pairingOctetFraction = weights.octet / total;
      singletOk =
          read.pairingOctetFraction <= cfg_.compositeOctetTolerance;
    }
  }
  passed += gate(singletOk, "color-singlet", failed);

  // 5. zero total certified winding and baryon flux, composed from the
  //    conjugate-pair integer sums rather than recomputed.
  const ConjugatePairRead pair =
      conjugatePair(evidence.first, evidence.second);
  read.totalWinding = pair.totalWinding;
  read.totalBaryonFlux = pair.totalBaryonFlux;
  passed += gate(pair.totalWinding.has_value() && *pair.totalWinding == 0,
                 "flux-zero", failed);

  read.confidence = static_cast<double>(passed) / kGates;
  read.classification = (passed == kGates) ? "meson-candidate" : "none";
  read.failedCertificates = std::move(failed);

  ConsumedCertificates consumed;
  consumed.consume(evidence.first.certificate);
  consumed.consume(evidence.second.certificate);
  read.certificate = consumed.verdict(read.classification != "none",
                                      evidence.first.certificate);
  return read;
}

DiquarkRead ParticleClusters::classifyDiquark(
    const CompositeCandidateEvidence &evidence) const {
  DiquarkRead read;
  read.thresholds = cfg_;
  fillCompositeReports(read, evidence);

  std::vector<std::string> failed;
  int passed = 0;
  constexpr int kGates = 4;

  // 1. two certified quarks (ν = +1 each).
  const bool quarksOk = certifiedConstituent(evidence.first, "quark") &&
                        certifiedConstituent(evidence.second, "quark");
  passed += gate(quarksOk, "constituent-quarks", failed);

  // 2. even composite parity (exact graded product; two odd clusters
  //    compose even).
  read.exteriorParity = compositeParity(evidence.first, evidence.second);
  passed += gate(read.exteriorParity == +1, "parity-even", failed);

  // 3. the certified Λ²C³ anti-triplet wedge occupation (a Gram
  //    determinant): exactly zero for duplicated color modes (Pauli).
  const quantum::WickCertificateRead &wedge = evidence.antiTripletRead;
  const bool wedgeCertified = wedge.certificate.holds();
  if (wedgeCertified) read.antiTripletWeight = wedge.value.real();
  passed += gate(wedgeCertified &&
                     wedge.value.real() >= cfg_.minAntiTripletWeight,
                 "anti-triplet", failed);

  // 4. the preserved constituent baryon flux: ν₁ + ν₂ = 2 ⇒ B = 2/3, summed
  //    from the constituents' certified fluxes.  Together with occupation
  //    two and even parity this is what distinguishes a diquark from an
  //    antiquark (B = −1/3).
  const ConjugatePairRead pair =
      conjugatePair(evidence.first, evidence.second);
  read.totalWinding = pair.totalWinding;
  read.totalBaryonFlux = pair.totalBaryonFlux;
  passed += gate(pair.totalWinding.has_value() && *pair.totalWinding == 2,
                 "baryon-flux-two-thirds", failed);

  read.confidence = static_cast<double>(passed) / kGates;
  read.classification = (passed == kGates) ? "diquark-candidate" : "none";
  read.failedCertificates = std::move(failed);

  ConsumedCertificates consumed;
  consumed.consume(evidence.first.certificate);
  consumed.consume(evidence.second.certificate);
  consumed.consume(wedge.certificate);
  read.certificate = consumed.verdict(read.classification != "none",
                                      evidence.first.certificate);
  return read;
}

// ---------------------------------------------------------------------------
// bound supercomponent, color singlet, and the proton certificate
// ---------------------------------------------------------------------------

namespace {

/// Spread (max − min) of a sampled channel, normalized by max(|mean|, 1):
/// relative for channels of order one and larger, absolute for channels near
/// zero, so a near-zero channel is never reported as infinitely unstable.
/// Returns NaN (unmeasured, never zero) for fewer than two samples or when
/// any sample is not finite.
double normalizedSpread(const std::vector<double> &values) {
  if (values.size() < 2) return kNaN;
  double lo = values.front();
  double hi = values.front();
  double sum = 0.0;
  for (const double v : values) {
    if (!std::isfinite(v)) return kNaN;
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    sum += v;
  }
  const double mean = sum / static_cast<double>(values.size());
  return (hi - lo) / std::max(std::abs(mean), 1.0);
}

/// Totals over a set of constituent quark reads: the summed certified
/// determinant windings (with B = ν/3), the summed certified baryon fluxes
/// and the graded parity product.  Any uncertified leg leaves its total
/// unknown rather than zero.  Shared by `conjugatePair` and
/// `classifyBaryon`.
struct ConstituentTotals {
  std::optional<int> winding{};
  std::optional<double> baryonFlux{};
  int parity = 0;
  std::optional<double> isospin{};
  std::optional<double> electricFlux{};
  /// The certified isospin occupation pattern in canonical order (every 'u'
  /// before every 'd'), so a constituent permutation cannot change it; empty
  /// when any constituent's isospin is unknown.
  std::string flavorPattern{};
};

ConstituentTotals constituentTotals(const std::vector<const QuarkRead *> &reads,
                                    double isospinTolerance) {
  ConstituentTotals out;
  if (reads.empty()) return out;

  int windingSum = 0;
  double fluxSum = 0.0;
  double isospinSum = 0.0;
  double electricSum = 0.0;
  int parityProduct = 1;
  bool windingOk = true;
  bool fluxOk = true;
  bool parityOk = true;
  bool isospinOk = true;
  bool electricOk = true;
  std::size_t ups = 0;
  std::size_t downs = 0;
  for (const QuarkRead *read : reads) {
    windingOk = windingOk && read->determinantWinding.has_value();
    if (read->determinantWinding.has_value())
      windingSum += *read->determinantWinding;
    fluxOk = fluxOk && read->baryonFlux.has_value();
    if (read->baryonFlux.has_value()) fluxSum += *read->baryonFlux;
    parityOk = parityOk && read->exteriorParity != 0;
    parityProduct *= read->exteriorParity;
    electricOk = electricOk && read->electricFlux.has_value();
    if (read->electricFlux.has_value()) electricSum += *read->electricFlux;
    if (!read->isospin.has_value()) {
      isospinOk = false;
      continue;
    }
    isospinSum += *read->isospin;
    if (std::abs(*read->isospin - 0.5) <= isospinTolerance)
      ++ups;
    else if (std::abs(*read->isospin + 0.5) <= isospinTolerance)
      ++downs;
    else
      isospinOk = false;
  }
  if (windingOk) out.winding = windingSum;
  if (fluxOk) out.baryonFlux = fluxSum;
  if (parityOk) out.parity = parityProduct;
  if (electricOk) out.electricFlux = electricSum;
  if (isospinOk) {
    out.isospin = isospinSum;
    out.flavorPattern = std::string(ups, 'u') + std::string(downs, 'd');
  }
  return out;
}

}  // namespace

std::string BoundSupercomponentRead::describe() const {
  std::ostringstream out;
  out << "BoundSupercomponentRead[" << (found ? "bound" : "unbound") << ", "
      << quarks.size() << " quark(s), overlap " << lifetimeOverlap
      << " slice(s), containment " << minContainment << "]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

std::string ScaleProfileRead::describe() const {
  std::ostringstream out;
  out << "ScaleProfileRead[" << (stable ? "stable" : "unstable") << ", "
      << sampleCount << " sample(s), r=" << radius
      << (radiusFinite ? " (finite)" : " (not finite)")
      << ", m_shell=" << spectralMass << ", worst dimensionless deviation "
      << profileMaxDeviation << ", physical mass unknown]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

std::string BaryonRead::describe() const {
  std::ostringstream out;
  out << "BaryonRead[" << classification;
  out << ", det(C^dag C)=" << colorGramDeterminant
      << ", color flux=" << colorFlux;
  if (baryonFlux.has_value())
    out << ", B=" << *baryonFlux;
  else
    out << ", B=unknown";
  if (electricFlux.has_value())
    out << ", Q=" << *electricFlux;
  else
    out << ", Q=unknown";
  out << ", flavor=" << (flavorPattern.empty() ? "unknown" : flavorPattern);
  if (totalJ2.has_value())
    out << ", J2=" << *totalJ2;
  else
    out << ", J2=unknown";
  if (totalJ2Variance.has_value())
    out << ", Var(J2)=" << *totalJ2Variance;
  else
    out << ", Var(J2)=unknown";
  out << ", parity=" << exteriorParity << ", confidence " << confidence << "]";
  if (!failedCertificates.empty()) {
    out << " failed:";
    for (const auto &name : failedCertificates) out << " " << name;
  }
  return out.str();
}

Record BaryonRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kRecordSchemaVersion);
  m["record_type"] = Record("baryon_read");
  for (std::size_t i = 0; i < quarks.size(); ++i)
    componentToRecord(m, "quark" + std::to_string(i), quarks[i]);
  componentToRecord(m, "bound_component", boundComponent);
  m["color_gram_determinant"] = Record(colorGramDeterminant);
  m["color_flux"] = Record(colorFlux);
  m["baryon_flux"] = optionalDouble(baryonFlux);
  m["electric_flux"] = optionalDouble(electricFlux);
  m["total_j2"] = optionalDouble(totalJ2);
  m["total_j2_variance"] = optionalDouble(totalJ2Variance);
  m["rotation_character_re"] =
      rotationCharacter.has_value() ? Record(rotationCharacter->real())
                                    : Record();
  m["rotation_character_im"] =
      rotationCharacter.has_value() ? Record(rotationCharacter->imag())
                                    : Record();
  m["classification"] = Record(classification);
  m["persistence"] = Record(persistence);
  Record::List failed;
  failed.reserve(failedCertificates.size());
  for (const auto &name : failedCertificates) failed.emplace_back(name);
  m["failed_certificates"] = Record(std::move(failed));
  m["color_wedge_re"] = Record(colorWedge.real());
  m["color_wedge_im"] = Record(colorWedge.imag());
  m["total_winding"] = optionalInt(totalWinding);
  m["exterior_parity"] = Record(exteriorParity);
  m["flavor_pattern"] = Record(flavorPattern);
  m["total_isospin"] = optionalDouble(totalIsospin);
  m["rotation_character_sign"] = Record(rotationCharacterSign);
  m["exchange_character_re"] =
      exchangeCharacter.has_value() ? Record(exchangeCharacter->real())
                                    : Record();
  m["exchange_character_im"] =
      exchangeCharacter.has_value() ? Record(exchangeCharacter->imag())
                                    : Record();
  m["spin_statistics_ratio_re"] =
      spinStatisticsRatio.has_value() ? Record(spinStatisticsRatio->real())
                                      : Record();
  m["spin_statistics_ratio_im"] =
      spinStatisticsRatio.has_value() ? Record(spinStatisticsRatio->imag())
                                      : Record();
  m["monopole_number"] = optionalInt(monopoleNumber);
  m["odd_monopole"] = Record(oddMonopole);
  m["projective_cocycle_nontrivial"] = Record(projectiveCocycleNontrivial);
  m["sharp_spin_right_residual"] = Record(sharpSpinRightResidual);
  m["sharp_spin_left_residual"] = Record(sharpSpinLeftResidual);
  m["variance_would_accept"] = Record(varianceWouldAccept);
  m["spin_lift_applicable"] = Record(spinLiftApplicable);
  m["spin_lift_accepted"] = Record(spinLiftAccepted);
  m["sharp_spin"] = Record(sharpSpin);
  m["quasi_free_class_swept"] = Record(quasiFreeClassSwept);
  m["class_variance_floor"] = Record(classVarianceFloor);
  m["radius"] = Record(radius);
  m["radius_finite"] = Record(radiusFinite);
  m["spectral_mass"] = Record(spectralMass);
  m["radius_ratio"] = Record(radiusRatio);
  m["profile_max_deviation"] = Record(profileMaxDeviation);
  m["profile_stable"] = Record(profileStable);
  m["physical_mass"] = optionalDouble(physicalMass);
  m["crossing_mass_applicable"] = Record(crossingMassApplicable);
  m["crossing_mass"] = Record(crossingMassValue);
  m["crossing_baryon_number"] = optionalDouble(crossingBaryonNumber);
  {
    Record::List defects;
    defects.reserve(crossingSignDefects.size());
    for (const auto &name : crossingSignDefects) defects.emplace_back(name);
    m["crossing_sign_defects"] = Record(std::move(defects));
  }
  m["lifetime_overlap"] = Record(lifetimeOverlap);
  m["transport_count"] = Record(static_cast<std::int64_t>(transportCount));
  m["transport_leakage_max"] = Record(transportLeakageMax);
  m["confidence"] = Record(confidence);
  m["thresholds"] = thresholdsToRecord(thresholds);
  m["certificate"] = certificateToRecord(certificate);
  return Record(std::move(m));
}

BaryonRead BaryonRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  requireSchema(m, "baryon_read");
  BaryonRead read;
  for (std::size_t i = 0; i < read.quarks.size(); ++i)
    read.quarks[i] = componentFromRecord(m, "quark" + std::to_string(i));
  read.boundComponent = componentFromRecord(m, "bound_component");
  read.colorGramDeterminant = m.at("color_gram_determinant").asDouble();
  read.colorFlux = m.at("color_flux").asDouble();
  read.baryonFlux = optionalDoubleFrom(m.at("baryon_flux"));
  read.electricFlux = optionalDoubleFrom(m.at("electric_flux"));
  read.totalJ2 = optionalDoubleFrom(m.at("total_j2"));
  read.totalJ2Variance = optionalDoubleFrom(m.at("total_j2_variance"));
  {
    const Record &re = m.at("rotation_character_re");
    const Record &im = m.at("rotation_character_im");
    if (!re.isNull() && !im.isNull())
      read.rotationCharacter = cd(re.asDouble(), im.asDouble());
  }
  read.classification = m.at("classification").asString();
  read.persistence = m.at("persistence").asDouble();
  for (const auto &entry : m.at("failed_certificates").asList())
    read.failedCertificates.push_back(entry.asString());
  read.colorWedge = cd(m.at("color_wedge_re").asDouble(),
                       m.at("color_wedge_im").asDouble());
  read.totalWinding = optionalIntFrom(m.at("total_winding"));
  read.exteriorParity = static_cast<int>(m.at("exterior_parity").asInt());
  read.flavorPattern = m.at("flavor_pattern").asString();
  read.totalIsospin = optionalDoubleFrom(m.at("total_isospin"));
  read.rotationCharacterSign =
      static_cast<int>(m.at("rotation_character_sign").asInt());
  {
    const Record &re = m.at("exchange_character_re");
    const Record &im = m.at("exchange_character_im");
    if (!re.isNull() && !im.isNull())
      read.exchangeCharacter = cd(re.asDouble(), im.asDouble());
  }
  {
    const Record &re = m.at("spin_statistics_ratio_re");
    const Record &im = m.at("spin_statistics_ratio_im");
    if (!re.isNull() && !im.isNull())
      read.spinStatisticsRatio = cd(re.asDouble(), im.asDouble());
  }
  read.monopoleNumber = optionalIntFrom(m.at("monopole_number"));
  read.oddMonopole = m.at("odd_monopole").asBool();
  read.projectiveCocycleNontrivial =
      m.at("projective_cocycle_nontrivial").asBool();
  read.sharpSpinRightResidual = m.at("sharp_spin_right_residual").asDouble();
  read.sharpSpinLeftResidual = m.at("sharp_spin_left_residual").asDouble();
  read.varianceWouldAccept = m.at("variance_would_accept").asBool();
  read.spinLiftApplicable = m.at("spin_lift_applicable").asBool();
  read.spinLiftAccepted = m.at("spin_lift_accepted").asBool();
  read.sharpSpin = m.at("sharp_spin").asBool();
  read.quasiFreeClassSwept = m.at("quasi_free_class_swept").asBool();
  read.classVarianceFloor = m.at("class_variance_floor").asDouble();
  read.radius = m.at("radius").asDouble();
  read.radiusFinite = m.at("radius_finite").asBool();
  read.spectralMass = m.at("spectral_mass").asDouble();
  read.radiusRatio = m.at("radius_ratio").asDouble();
  read.profileMaxDeviation = m.at("profile_max_deviation").asDouble();
  read.profileStable = m.at("profile_stable").asBool();
  read.physicalMass = optionalDoubleFrom(m.at("physical_mass"));
  // The crossing leaves are optional: a record without them reads back as not
  // applicable with the value unknown, never as a zero-filled claim and never
  // as a missing-key throw.
  read.crossingMassApplicable = m.count("crossing_mass_applicable") &&
                                m.at("crossing_mass_applicable").asBool();
  read.crossingMassValue = optionalLeaf(m, "crossing_mass");
  read.crossingBaryonNumber =
      m.count("crossing_baryon_number")
          ? optionalDoubleFrom(m.at("crossing_baryon_number"))
          : std::optional<double>{};
  read.crossingSignDefects.clear();
  if (m.count("crossing_sign_defects")) {
    for (const auto &entry : m.at("crossing_sign_defects").asList()) {
      read.crossingSignDefects.push_back(entry.asString());
    }
  }
  read.lifetimeOverlap = m.at("lifetime_overlap").asDouble();
  read.transportCount =
      static_cast<std::size_t>(m.at("transport_count").asInt());
  read.transportLeakageMax = m.at("transport_leakage_max").asDouble();
  read.confidence = m.at("confidence").asDouble();
  read.thresholds = thresholdsFromRecord(m.at("thresholds"));
  read.certificate = certificateFromRecord(m.at("certificate"));
  return read;
}

std::vector<BoundSupercomponentRead>
ParticleClusters::boundSupercomponentSearch(
    const std::vector<ComponentRead> &nextLevelComponents,
    const std::vector<BoundCandidateEvidence> &candidates) const {
  std::vector<BoundSupercomponentRead> out;

  for (const ComponentRead &component : nextLevelComponents) {
    const std::unordered_set<std::uint64_t> support(component.support.begin(),
                                                    component.support.end());

    // Membership: a certified quark candidate whose level-0 support meets
    // this component.  An uncertified candidate is never counted.
    std::vector<std::size_t> members;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const BoundCandidateEvidence &cand = candidates[i];
      if (!certifiedConstituent(cand.quark, "quark")) continue;
      if (cand.support.empty()) continue;
      const bool meets =
          std::any_of(cand.support.begin(), cand.support.end(),
                      [&support](std::uint64_t id) {
                        return support.find(id) != support.end();
                      });
      if (meets) members.push_back(i);
    }
    if (members.empty()) continue;

    BoundSupercomponentRead read;
    read.boundComponent = component.id;
    read.thresholds = cfg_;
    read.quarkIndices = members;
    for (const std::size_t i : members)
      read.quarks.push_back(candidates[i].quark.component);

    std::vector<std::string> failed;
    int passed = 0;
    constexpr int kGates = 5;

    // 1. the next modular level: strictly above every constituent's level.
    bool levelOk = true;
    for (const std::size_t i : members)
      levelOk = levelOk &&
                component.id.level() > candidates[i].quark.component.level();
    passed += gate(levelOk, "supercomponent-level", failed);

    // 2. exactly three certified quark candidates.
    passed += gate(members.size() == 3, "quark-count", failed);

    // 3. support containment: every member's level-0 support lies inside
    //    the supercomponent.
    double minContainment = kNaN;
    for (const std::size_t i : members) {
      const auto &cand = candidates[i];
      std::size_t inside = 0;
      for (const std::uint64_t id : cand.support)
        if (support.find(id) != support.end()) ++inside;
      const double fraction = static_cast<double>(inside) /
                              static_cast<double>(cand.support.size());
      minContainment =
          std::isnan(minContainment) ? fraction
                                     : std::min(minContainment, fraction);
    }
    read.minContainment = minContainment;
    passed += gate(std::isfinite(minContainment) &&
                       minContainment >= cfg_.minSupportContainment,
                   "support-containment", failed);

    // 4. overlapping lifetimes: the intersection of the members' persistence
    //    windows.  A missing window is missing evidence.
    bool lifetimesKnown = true;
    std::size_t first = 0;
    std::size_t last = std::numeric_limits<std::size_t>::max();
    for (const std::size_t i : members) {
      const auto &window = candidates[i].lifetime;
      if (!window.has_value()) {
        lifetimesKnown = false;
        break;
      }
      first = std::max(first, window->first);
      last = std::min(last, window->second);
    }
    double overlap = 0.0;
    if (lifetimesKnown && last >= first) {
      overlap = static_cast<double>(last - first + 1);
      read.lifetimeWindow = std::make_pair(first, last);
    }
    read.lifetimeOverlap = overlap;
    passed += gate(lifetimesKnown && overlap >= cfg_.minLifetimeOverlap,
                   "lifetime-overlap", failed);

    // 5. mutual transport stays inside: every member supplies at least one
    //    transport to its partners and every supplied link is accepted with
    //    leakage under the cap.  Leakage is the tracked subspace turning
    //    away from its successor.
    bool transportsOk = true;
    double maxLeakage = kNaN;
    std::size_t transportCount = 0;
    for (const std::size_t i : members) {
      const auto &links = candidates[i].mutualTransports;
      transportsOk = transportsOk && !links.empty();
      transportCount += links.size();
      for (const FiberTransportRead &link : links) {
        transportsOk = transportsOk && link.accepted;
        maxLeakage = maxFinite(maxLeakage, link.leakage);
      }
    }
    read.transportCount = transportCount;
    read.transportLeakageMax = maxLeakage;
    transportsOk = transportsOk && std::isfinite(maxLeakage) &&
                   maxLeakage <= cfg_.maxTransportLeakage;
    passed += gate(transportsOk, "transport-containment", failed);

    read.found = (passed == kGates);
    read.failedCertificates = std::move(failed);

    ConsumedCertificates consumed;
    for (const std::size_t i : members)
      consumed.consume(candidates[i].quark.certificate);
    read.certificate = consumed.verdict(
        read.found, candidates[members.front()].quark.certificate);
    out.push_back(std::move(read));
  }
  return out;
}

ScaleProfileSample ParticleClusters::scaleProfileSample(
    const RegisterContext &ctx) {
  const std::shared_ptr<InteriorHinges> &hinges = ctx.interiorHinges();
  ScaleProfileSample sample;
  if (!hinges) return sample;

  // The interior-hinge battery, read exactly as EmergentRadius and
  // EmergentMass read it: nothing recomputed, no solver called.
  const InteriorHinges::Radii radii = hinges->radii();
  const InteriorHinges::Masses masses = hinges->masses();
  const InteriorHinges::Localization localization = hinges->localization();

  sample.radius = radii.rDual;
  sample.radiusCrossCheck = radii.rPrimal;
  sample.spectralMass = masses.empty ? kNaN : masses.mShell;
  sample.localization = localization.empty ? kNaN : localization.pr;
  sample.radialWeightProfile.reserve(localization.shellProfile.size());
  for (const auto &entry : localization.shellProfile)
    sample.radialWeightProfile.push_back(entry.second.weightShare);
  return sample;
}

ScaleProfileRead ParticleClusters::scaleProfile(
    const std::vector<ScaleProfileSample> &samples) const {
  ScaleProfileRead read;
  read.thresholds = cfg_;
  read.sampleCount = samples.size();

  std::vector<std::string> failed;
  int passed = 0;
  constexpr int kGates = 12;

  // 1. a refinement window: stability is unmeasurable from one sample.
  passed += gate(samples.size() >= 2, "refinement-window", failed);

  // 2. a finite emergent radius in every sample.  The radius is
  //    dimensionful, so only its finiteness is certified.
  bool radiusOk = !samples.empty();
  for (const ScaleProfileSample &sample : samples)
    radiusOk = radiusOk && std::isfinite(sample.radius) &&
               sample.radius > cfg_.minRadius;
  if (!samples.empty()) read.radius = samples.front().radius;
  read.radiusFinite = radiusOk;
  passed += gate(radiusOk, "finite-radius", failed);

  // 3-5. the dimensionless scalar channels and their refinement spreads.
  std::vector<double> ratios;
  std::vector<double> masses;
  std::vector<double> localizations;
  ratios.reserve(samples.size());
  masses.reserve(samples.size());
  localizations.reserve(samples.size());
  for (const ScaleProfileSample &sample : samples) {
    ratios.push_back(sample.radiusCrossCheck == 0.0
                         ? kNaN
                         : sample.radius / sample.radiusCrossCheck);
    masses.push_back(sample.spectralMass);
    localizations.push_back(sample.localization);
  }
  if (!samples.empty()) {
    read.radiusRatio = ratios.front();
    read.spectralMass = masses.front();
    read.localization = localizations.front();
  }
  read.radiusRatioSpread = normalizedSpread(ratios);
  read.spectralMassSpread = normalizedSpread(masses);
  read.localizationSpread = normalizedSpread(localizations);
  passed += gate(std::isfinite(read.radiusRatioSpread) &&
                     read.radiusRatioSpread <= cfg_.maxProfileDeviation,
                 "radius-ratio-stability", failed);
  passed += gate(std::isfinite(read.spectralMassSpread) &&
                     read.spectralMassSpread <= cfg_.maxProfileDeviation,
                 "spectral-mass-stability", failed);
  passed += gate(std::isfinite(read.localizationSpread) &&
                     read.localizationSpread <= cfg_.maxProfileDeviation,
                 "localization-stability", failed);

  // 6. the dimensionless radial weight profile, a radial curvature-weight
  //    density rather than a form factor: present in every sample, with the
  //    same shell count, and stable per shell.
  bool profileShapeOk = samples.size() >= 2;
  std::size_t shells = 0;
  if (!samples.empty()) {
    shells = samples.front().radialWeightProfile.size();
    profileShapeOk = profileShapeOk && shells > 0;
    for (const ScaleProfileSample &sample : samples)
      profileShapeOk =
          profileShapeOk && sample.radialWeightProfile.size() == shells;
  }
  double profileDeviation = kNaN;
  if (profileShapeOk) {
    profileDeviation = 0.0;
    for (std::size_t k = 0; k < shells; ++k) {
      double lo = samples.front().radialWeightProfile[k];
      double hi = lo;
      for (const ScaleProfileSample &sample : samples) {
        const double v = sample.radialWeightProfile[k];
        if (!std::isfinite(v)) {
          profileDeviation = kNaN;
          break;
        }
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      if (std::isnan(profileDeviation)) break;
      profileDeviation = std::max(profileDeviation, hi - lo);
    }
    read.profileShells = shells;
  }
  read.profileMaxDeviation = profileDeviation;
  passed += gate(std::isfinite(profileDeviation) &&
                     profileDeviation <= cfg_.maxProfileDeviation,
                 "profile-stability", failed);

  // 7-12. the remaining dimensionless certificates under refinement; the
  //       mass-radius battery is not the whole list.  A channel the caller
  //       never filled is unknown, so its spread is NaN and the certificate
  //       fails by name.
  std::vector<double> colorGrams;
  std::vector<double> baryonFluxes;
  std::vector<double> electricFluxes;
  std::vector<double> anchorScores;
  colorGrams.reserve(samples.size());
  baryonFluxes.reserve(samples.size());
  electricFluxes.reserve(samples.size());
  anchorScores.reserve(samples.size());
  for (const ScaleProfileSample &sample : samples) {
    colorGrams.push_back(sample.colorGramDeterminant);
    baryonFluxes.push_back(sample.baryonFlux);
    electricFluxes.push_back(sample.electricFlux);
    anchorScores.push_back(sample.anchorScore);
  }
  if (!samples.empty()) {
    read.colorGramDeterminant = samples.front().colorGramDeterminant;
    read.rotationCharacter = samples.front().rotationCharacter;
    read.baryonFlux = samples.front().baryonFlux;
    read.electricFlux = samples.front().electricFlux;
    read.compositeParity = samples.front().compositeParity;
    read.anchorScore = samples.front().anchorScore;
  }
  read.colorGramSpread = normalizedSpread(colorGrams);
  read.baryonFluxSpread = normalizedSpread(baryonFluxes);
  read.electricFluxSpread = normalizedSpread(electricFluxes);
  read.anchorScoreSpread = normalizedSpread(anchorScores);
  // The 2π character is complex: its deviation is the maximum pairwise
  // distance in the plane, never a real-part comparison.
  double rotationSpread = samples.size() >= 2 ? 0.0 : kNaN;
  for (std::size_t i = 0; i < samples.size() && samples.size() >= 2; ++i) {
    const cd a = samples[i].rotationCharacter;
    if (!std::isfinite(a.real()) || !std::isfinite(a.imag())) {
      rotationSpread = kNaN;
      break;
    }
    for (std::size_t j = i + 1; j < samples.size(); ++j)
      rotationSpread = std::max(rotationSpread,
                                std::abs(a - samples[j].rotationCharacter));
  }
  read.rotationCharacterSpread = rotationSpread;
  // Composite parity is an integer channel: stability is exact equality of a
  // definite sign across the window, never a tolerance.
  bool parityStable = samples.size() >= 2;
  for (const ScaleProfileSample &sample : samples)
    parityStable = parityStable && sample.compositeParity != 0 &&
                   sample.compositeParity == samples.front().compositeParity;
  read.compositeParityStable = parityStable;

  passed += gate(std::isfinite(read.colorGramSpread) &&
                     read.colorGramSpread <= cfg_.maxProfileDeviation,
                 "color-gram-stability", failed);
  passed += gate(std::isfinite(read.rotationCharacterSpread) &&
                     read.rotationCharacterSpread <= cfg_.maxProfileDeviation,
                 "rotation-character-stability", failed);
  passed += gate(std::isfinite(read.baryonFluxSpread) &&
                     read.baryonFluxSpread <= cfg_.maxProfileDeviation,
                 "baryon-flux-stability", failed);
  passed += gate(std::isfinite(read.electricFluxSpread) &&
                     read.electricFluxSpread <= cfg_.maxProfileDeviation,
                 "electric-flux-stability", failed);
  passed += gate(parityStable, "composite-parity-stability", failed);
  passed += gate(std::isfinite(read.anchorScoreSpread) &&
                     read.anchorScoreSpread <= cfg_.maxProfileDeviation,
                 "anchor-score-stability", failed);

  read.stable = (passed == kGates);
  read.failedCertificates = std::move(failed);

  // The measured deviations of finite sums: CertifiedNumerical against the
  // configured refinement cap.  No dimensionful mass is emitted.
  double residual = kNaN;
  for (const double channel :
       {read.radiusRatioSpread, read.spectralMassSpread,
        read.localizationSpread, read.profileMaxDeviation,
        read.colorGramSpread, read.rotationCharacterSpread,
        read.baryonFluxSpread, read.electricFluxSpread,
        read.anchorScoreSpread})
    residual = maxFinite(residual, channel);
  read.certificate =
      read.stable
          ? Certificate::certifiedNumerical(
                CertificateDomain::Static, CertificateRegime::NonNormal,
                residual, /*conditioning=*/kNaN, cfg_.maxProfileDeviation)
          : Certificate::heuristicDiscovery(CertificateDomain::Static,
                                            CertificateRegime::NonNormal);
  return read;
}

BaryonRead ParticleClusters::classifyBaryon(
    const BaryonCandidateEvidence &evidence) const {
  BaryonRead read;
  read.thresholds = cfg_;
  read.boundComponent = evidence.boundComponent;
  for (std::size_t i = 0; i < read.quarks.size(); ++i)
    read.quarks[i] = evidence.quarks[i].component;
  read.persistence = evidence.persistenceLifetime;
  read.lifetimeOverlap = evidence.binding.lifetimeOverlap;
  read.transportCount = evidence.lifetimeTransports.size();
  double compositeLeakage = kNaN;
  for (const FiberTransportRead &transport : evidence.lifetimeTransports)
    compositeLeakage = maxFinite(compositeLeakage, transport.leakage);
  read.transportLeakageMax = compositeLeakage;

  std::vector<const QuarkRead *> constituents;
  constituents.reserve(evidence.quarks.size());
  for (const QuarkRead &quark : evidence.quarks) constituents.push_back(&quark);
  const ConstituentTotals totals =
      constituentTotals(constituents, cfg_.isospinTolerance);

  std::vector<std::string> failed;
  int passed = 0;
  constexpr int kGates = 16;

  // ── structural gates (a failure of either is "no baryon") ────────────

  // 1. three certified quark constituents, consumed verbatim; each already
  //    carries its accepted oriented-triangle anchor and its
  //    determinant-winding certificate.
  bool constituentsOk = true;
  for (const QuarkRead &quark : evidence.quarks)
    constituentsOk = constituentsOk && certifiedConstituent(quark, "quark");
  const bool structuralQuarks =
      gate(constituentsOk, "constituent-quarks", failed);
  passed += structuralQuarks;

  // 2. one persistent bound supercomponent containing these three
  //    constituents: the supercomponent search result must hold and its
  //    contained-candidate set must equal the three constituents' label-free
  //    identities (an order-insensitive set comparison, so an incoherent
  //    bundle never certifies).
  std::vector<ComponentId> boundIds = evidence.binding.quarks;
  std::vector<ComponentId> constituentIds;
  constituentIds.reserve(evidence.quarks.size());
  for (const QuarkRead &quark : evidence.quarks)
    constituentIds.push_back(quark.component);
  std::sort(boundIds.begin(), boundIds.end());
  std::sort(constituentIds.begin(), constituentIds.end());
  const bool bindingOk = evidence.binding.found &&
                         evidence.binding.certificate.holds() &&
                         boundIds == constituentIds;
  const bool structuralBinding =
      gate(bindingOk, "bound-supercomponent", failed);
  passed += structuralBinding;

  // ── the proton certificate ──────────────────────────────────────────

  // 3. the color singlet.  The three color columns are normalized once and
  //    the three-mode wedge S_ABC = det[c_A c_B c_C] is built once
  //    (ColorFiber::colorWedge); the Gram certificate is its squared
  //    magnitude, the ColorFiber::singletGram identity read off the same
  //    wedge rather than a second determinant, with no extra fermion sign on
  //    the color epsilon.
  bool singletOk = false;
  if (evidence.colorColumns.norm() > 0.0) {
    Eigen::Matrix3cd columns = evidence.colorColumns;
    for (Eigen::Index j = 0; j < 3; ++j) {
      const double norm = columns.col(j).norm();
      if (norm > 0.0) columns.col(j) /= norm;
    }
    read.colorWedge = ColorFiber::colorWedge(columns);
    read.colorGramDeterminant = std::norm(read.colorWedge);
    singletOk = std::abs(read.colorGramDeterminant - 1.0) <=
                cfg_.colorGramTolerance;
  }
  passed += gate(singletOk, "color-singlet", failed);

  // 4. the independent vanishing net-color-flux diagnostic: the octet
  //    (traceless) weight of the bound object's color bilinear under the
  //    exact 1 ⊕ 8 split (octetBilinearRead).  On a finite complex this is a
  //    diagnostic, not by itself a proof of confinement.
  const bool colorFluxCertified = evidence.colorFlux.certificate.holds() &&
                                  std::isfinite(evidence.colorFlux.octetWeight);
  if (colorFluxCertified) read.colorFlux = evidence.colorFlux.octetWeight;
  passed += gate(colorFluxCertified &&
                     evidence.colorFlux.octetWeight <= cfg_.colorFluxTolerance,
                 "color-flux-zero", failed);

  // 5. summed certified determinant winding: ν = 3 ⇒ B = ν/3 = +1.
  read.totalWinding = totals.winding;
  read.baryonFlux = totals.baryonFlux;
  passed += gate(totals.winding.has_value() && *totals.winding == 3,
                 "baryon-flux-unit", failed);

  // 6. odd composite exterior parity (the graded product).
  read.exteriorParity = totals.parity;
  passed += gate(read.exteriorParity == -1, "composite-parity-odd", failed);

  // 7. the constituent flavor read: the `uud` occupation pattern.
  read.flavorPattern = totals.flavorPattern;
  read.totalIsospin = totals.isospin;
  passed += gate(read.flavorPattern == "uud", "flavor-uud", failed);

  // 8. the constituent charge read: summed certified Gauss fluxes = +1.
  read.electricFlux = totals.electricFlux;
  const bool electricOk =
      totals.electricFlux.has_value() &&
      std::abs(*totals.electricFlux - 1.0) <= cfg_.gaussTolerance;
  passed += gate(electricOk, "electric-flux-unit", failed);

  // 9. the total-space ⟨J²⟩ = 3/4.  The Wick expectation is the quasi-free
  //    path; a candidate carried as an explicit composite state supplies the
  //    dense oracle instead, which carries no variance.  This is never a
  //    product of per-hole or per-edge spinors.
  if (evidence.spinSquaredRead.certificate.holds())
    read.totalJ2 = evidence.spinSquaredRead.value.real();
  else if (evidence.totalSpaceJ2.has_value())
    read.totalJ2 = *evidence.totalSpaceJ2;
  passed += gate(read.totalJ2.has_value() &&
                     std::abs(*read.totalJ2 - 0.75) <=
                         cfg_.spinExpectationTolerance,
                 "spin-expectation", failed);

  // 10. sharp spin: BOTH eigen-equations (J² − ¾I)|Ψ_R⟩ = 0 and
  //     ⟨Ψ_L|(J² − ¾I) = 0 on the bounded superposition of determinants.
  //     The complex variance is read alongside them and reported, because a
  //     vanishing complex variance can come from isotropic cancellation on a
  //     state that is not an eigenstate; an absent eigen read is unknown, not
  //     sharp, and an absent variance is unknown, not zero.
  if (evidence.spinVarianceRead.certificate.holds())
    read.totalJ2Variance = evidence.spinVarianceRead.value.real();
  if (evidence.sharpSpinEigen.has_value()) {
    read.sharpSpinRightResidual = evidence.sharpSpinEigen->rightResidual;
    read.sharpSpinLeftResidual = evidence.sharpSpinEigen->leftResidual;
    read.varianceWouldAccept = evidence.sharpSpinEigen->varianceWouldAccept;
    read.sharpSpin = evidence.sharpSpinEigen->sharp &&
                     evidence.sharpSpinEigen->certificate.holds();
  }
  passed += gate(read.sharpSpin, "sharp-spin", failed);

  // 11. half-integer spin from the connection's topological charge: an odd
  //     monopole number of the U(1) part of U through the cluster's bounding
  //     cut, and a cohomologically nontrivial cocycle of the rotation group's
  //     projective action.  The 2π character is recorded here and gates
  //     nothing: a rigid rotation leaves every band constant, so it is +1
  //     along any rigid cycle whatever the spin.
  const HolonomyCharacterRead &rotation = evidence.rotation;
  const bool rotationCertified =
      rotation.certificate.holds() &&
      rotation.channel == HolonomyChannel::PhysicalRotation;
  if (rotationCertified) {
    read.rotationCharacter = rotation.character;
    read.rotationCharacterSign = rotation.characterSign;
  }
  const bool monopoleCertified = evidence.monopoleSpin.has_value() &&
                                 evidence.monopoleSpin->certificate.holds();
  if (monopoleCertified) {
    read.monopoleNumber = evidence.monopoleSpin->monopole.monopoleNumber;
    read.oddMonopole = evidence.monopoleSpin->monopole.odd &&
                       evidence.monopoleSpin->monopole.bundle;
    read.projectiveCocycleNontrivial =
        evidence.monopoleSpin->cocycle.nontrivial;
  }
  passed += gate(monopoleCertified && read.oddMonopole, "odd-monopole",
                 failed);
  passed += gate(monopoleCertified && read.projectiveCocycleNontrivial,
                 "projective-cocycle", failed);

  //     The particle-exchange channel is report-only: the exchange character
  //     and the doubly cancelled spin-statistics ratio
  //     chi(exchange)·chi(2π)^{-1} are recorded but never gate.  A mislabeled
  //     channel is refused, not reinterpreted.
  if (evidence.exchange.has_value() &&
      evidence.exchange->certificate.holds() &&
      evidence.exchange->channel == HolonomyChannel::ParticleExchange) {
    read.exchangeCharacter = evidence.exchange->character;
    if (rotationCertified)
      read.spinStatisticsRatio = ExchangeHolonomy::doublyCancelledRatio(
          *evidence.exchange, rotation);
  }

  // 12. the SO(d) → Spin(d) lift, demanded only when the caller declares a
  //     continuum spin claim.
  read.spinLiftApplicable = evidence.continuumSpinClaim;
  read.spinLiftAccepted = evidence.spinLift.has_value() &&
                          evidence.spinLift->certificate.holds() &&
                          evidence.spinLift->liftExists;
  passed += gate(!evidence.continuumSpinClaim || read.spinLiftAccepted,
                 "spin-lift", failed);

  // 13/14. the mass-radius battery over the refinement window: a finite
  //        radius and refinement-stable dimensionless profiles.  The
  //        dimensionful mass stays unknown (physicalMass is always empty).
  const ScaleProfileRead scale = scaleProfile(evidence.scaleSamples);
  read.radius = scale.radius;
  read.radiusFinite = scale.radiusFinite;
  read.spectralMass = scale.spectralMass;
  read.radiusRatio = scale.radiusRatio;
  read.profileMaxDeviation = scale.profileMaxDeviation;
  read.profileStable = scale.stable;
  passed += gate(scale.radiusFinite, "finite-radius", failed);
  passed += gate(scale.stable, "profile-stability", failed);

  // 15. the world-tube crossing readouts, applicable-gated like `spin-lift`:
  //     a candidate assembled without crossing evidence passes vacuously,
  //     while supplied evidence is enforced — a finite crossing mass, the
  //     coherent one-third sum reading B = +1, and no determinant-line sign
  //     defect on any certified tube.  A half bundle (mass without baryon
  //     sum, or the reverse) fails by name rather than grading half a
  //     certificate.
  const bool crossingApplicable = evidence.crossingMass.has_value() ||
                                  evidence.crossingBaryon.has_value();
  bool crossingAccepted = false;
  if (crossingApplicable && evidence.crossingMass.has_value() &&
      evidence.crossingBaryon.has_value()) {
    const CrossingMassRead &mass = *evidence.crossingMass;
    const BaryonCrossingRead &baryon = *evidence.crossingBaryon;
    crossingAccepted =
        std::isfinite(mass.crossingMass) && mass.admissibleCrossings != 0 &&
        baryon.signDefects.empty() && baryon.baryonNumber.has_value() &&
        std::abs(*baryon.baryonNumber - 1.0) <= cfg_.isospinTolerance;
  }
  read.crossingMassApplicable = crossingApplicable;
  read.crossingMassValue =
      evidence.crossingMass.has_value() ? evidence.crossingMass->crossingMass
                                        : kNaN;
  read.crossingBaryonNumber =
      evidence.crossingBaryon.has_value()
          ? evidence.crossingBaryon->baryonNumber
          : std::optional<double>{};
  read.crossingSignDefects =
      evidence.crossingBaryon.has_value()
          ? evidence.crossingBaryon->signDefects
          : std::vector<std::string>{};
  passed += gate(!crossingApplicable || crossingAccepted, "crossing-readouts",
                 failed);

  // ── the accepted covariance-only class (the obstruction premise) ─────

  bool classSwept = !evidence.classVarianceReads.empty();
  double varianceFloor = kNaN;
  for (const quantum::WickCertificateRead &variance :
       evidence.classVarianceReads) {
    classSwept = classSwept && variance.certificate.holds();
    const double magnitude = std::abs(variance.value);
    varianceFloor = std::isnan(varianceFloor)
                        ? magnitude
                        : std::min(varianceFloor, magnitude);
  }
  read.quasiFreeClassSwept = classSwept;
  read.classVarianceFloor = classSwept ? varianceFloor : kNaN;

  // ── the four-way verdict ─────────────────────────────────────────────

  read.confidence = static_cast<double>(passed) / kGates;
  if (!structuralQuarks || !structuralBinding) {
    read.classification = "no-baryon";
  } else if (passed == kGates) {
    read.classification = "certified-proton";
  } else if (failed.size() == 1 && failed.front() == "sharp-spin" &&
             read.totalJ2Variance.has_value() && read.quasiFreeClassSwept &&
             read.classVarianceFloor > cfg_.spinVarianceTolerance) {
    // Every other certificate passes and Var(J²) fails to converge to zero
    // across the accepted covariance-only class.  Resolving this requires an
    // explicit non-Gaussian mechanism, which this classifier does not
    // supply; it is not a refutation of the geometry.
    read.classification = "quasi-free-sharp-spin-obstruction";
  } else {
    read.classification = "baryon-candidate";
  }
  read.failedCertificates = std::move(failed);

  ConsumedCertificates consumed;
  for (const QuarkRead &quark : evidence.quarks)
    consumed.consume(quark.certificate);
  consumed.consume(evidence.binding.certificate);
  consumed.consume(evidence.colorFlux.certificate);
  consumed.consume(evidence.spinSquaredRead.certificate);
  consumed.consume(evidence.spinVarianceRead.certificate);
  if (evidence.monopoleSpin.has_value())
    consumed.consume(evidence.monopoleSpin->certificate);
  if (evidence.sharpSpinEigen.has_value())
    consumed.consume(evidence.sharpSpinEigen->certificate);
  consumed.consume(scale.certificate);
  if (evidence.spinLift.has_value())
    consumed.consume(evidence.spinLift->certificate);
  read.certificate = consumed.verdict(read.classification == "certified-proton",
                                      evidence.quarks.front().certificate);
  return read;
}

std::vector<BaryonRead> ParticleClusters::classifyBoundSupercomponents(
    const std::vector<BoundSupercomponentRead> &bindings,
    const std::vector<QuarkRead> &constituentReads,
    const std::vector<double> &boundLifetimes) const {
  if (!boundLifetimes.empty() && boundLifetimes.size() != bindings.size())
    throw std::invalid_argument(
        "ParticleClusters::classifyBoundSupercomponents: boundLifetimes must "
        "be empty or one entry per binding");
  std::vector<BaryonRead> out;
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    const BoundSupercomponentRead &binding = bindings[index];
    // Exactly three certified constituents.  `quarkIndices` lists only the
    // certified contained candidates, so this is the three-cluster condition
    // itself.
    if (binding.quarkIndices.size() != 3) continue;
    BaryonCandidateEvidence evidence;
    evidence.boundComponent = binding.boundComponent;
    evidence.binding = binding;
    for (std::size_t leg = 0; leg < 3; ++leg) {
      const std::size_t candidate = binding.quarkIndices[leg];
      if (candidate >= constituentReads.size())
        throw std::invalid_argument(
            "ParticleClusters::classifyBoundSupercomponents: a binding "
            "indexes a constituent outside the supplied read list");
      evidence.quarks[leg] = constituentReads[candidate];
    }
    evidence.persistenceLifetime =
        boundLifetimes.empty() ? kNaN : boundLifetimes[index];
    // Everything else stays default-constructed: absent evidence, which
    // `classifyBaryon` reports as a named failed certificate.  Nothing is
    // filled in on the caller's behalf.
    out.push_back(classifyBaryon(evidence));
  }
  return out;
}

}  // namespace tessera::observables
