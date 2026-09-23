// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/ClusterRegister.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mesh/Edge.h"
#include "mesh/Vertex.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

using ::tessera::cobordism::Certificate;
using ::tessera::cobordism::CertificateDomain;
using ::tessera::cobordism::CertificateRegime;
using ::tessera::cobordism::regimeName;
using ::tessera::cobordism::regimeFromName;

namespace {

// Schema 2 carries the contour certificate (`contour`, `contour_node_count`,
// `resolvent_bound`) and the Kontsevich-Segal allowability of the instance the
// band was read on (`allowability_margin`, `lorentzian_epsilon`), plus the
// thresholds deciding them. Schema 1 remains readable: what it never measured
// reads back as unmeasured, never as zero.
constexpr int kSchemaVersion = 2;
constexpr int kOldestReadableSchema = 1;

/// Whether a value was measured; NaN means "not measured".
bool measured(double value) { return !std::isnan(value); }

/// A double leaf an older schema may not carry: absent reads back as NaN.
double optionalDouble(const Record::Map &m, const char *key) {
  const auto it = m.find(key);
  return it == m.end() ? std::numeric_limits<double>::quiet_NaN()
                       : it->second.asDouble();
}

Record stringsToRecord(const std::vector<std::string> &names) {
  Record::List list;
  list.reserve(names.size());
  for (const auto &name : names) list.push_back(Record(name));
  return Record(std::move(list));
}

std::vector<std::string> stringsFromRecord(const Record &record) {
  std::vector<std::string> out;
  for (const auto &entry : record.asList()) out.push_back(entry.asString());
  return out;
}

Record supportToRecord(const std::vector<std::uint64_t> &support) {
  Record::List list;
  list.reserve(support.size());
  for (const auto id : support)
    list.push_back(Record(static_cast<std::int64_t>(id)));
  return Record(std::move(list));
}

std::vector<std::uint64_t> supportFromRecord(const Record &record) {
  std::vector<std::uint64_t> out;
  for (const auto &entry : record.asList())
    out.push_back(static_cast<std::uint64_t>(entry.asInt()));
  return out;
}



}  // namespace

// ─────────────────────────────────────────────────────────────────────
// ClusterRegisterRead
// ─────────────────────────────────────────────────────────────────────

std::string ClusterRegisterRead::describe() const {
  std::ostringstream out;
  out << "register degree " << degree << " rank " << rank << " on "
      << support.size() << " vertices, regime " << regimeName(regime.regime);
  if (!contour.empty())
    out << ", contour " << contour << " (" << contourNodeCount
        << " nodes, resolvent bound " << resolventBound << ")";
  out << (accepted ? " — ACCEPTED" : " — not accepted");
  if (!failedConjuncts.empty()) {
    out << "; failed:";
    for (const auto &name : failedConjuncts) out << " " << name;
  }
  if (!unmeasured.empty()) {
    out << "; unmeasured:";
    for (const auto &name : unmeasured) out << " " << name;
  }
  return out.str();
}

Record ClusterRegisterRead::toRecord() const {
  Record::Map m;
  m["schema_version"] = Record(kSchemaVersion);
  m["record_type"] = Record("cluster_register");
  m["component_hash"] = Record(component.canonicalHash());
  m["component_level"] = Record(static_cast<std::int64_t>(component.level()));
  m["support"] = supportToRecord(support);
  m["degree"] = Record(degree);
  m["rank"] = Record(static_cast<std::int64_t>(rank));
  m["band"] = band.toRecord();

  m["support_connected"] = Record(supportConnected);
  m["support_pieces"] = Record(static_cast<std::int64_t>(supportPieces));
  m["localization_excess"] = Record(localizationExcess);
  m["band_gap"] = Record(bandGap);
  m["contour"] = Record(contour);
  m["contour_node_count"] = Record(contourNodeCount);
  m["resolvent_bound"] = Record(resolventBound);
  m["allowability_margin"] = Record(allowabilityMargin);
  m["lorentzian_epsilon"] = Record(lorentzianEpsilon);
  m["neighbour_overlap"] = Record(neighbourOverlap);
  m["frame_lifetime"] = Record(frameLifetime);
  m["transport_leakage"] = Record(transportLeakage);

  Record::Map r;
  r["regime"] = Record(regimeName(regime.regime));
  r["gram_defect"] = Record(regime.gramDefect);
  r["positive_signature"] = Record(regime.positiveSignature);
  r["negative_signature"] = Record(regime.negativeSignature);
  r["neutral_signature"] = Record(regime.neutralSignature);
  r["signature_normalizable"] = Record(regime.signatureNormalizable);
  r["eigen_residual"] = Record(regime.eigenResidual);
  r["left_residual"] = Record(regime.leftResidual);
  r["frame_condition_number"] = Record(regime.frameConditionNumber);
  m["regime_report"] = Record(std::move(r));

  m["failed_conjuncts"] = stringsToRecord(failedConjuncts);
  m["unmeasured"] = stringsToRecord(unmeasured);
  m["accepted"] = Record(accepted);

  Record::Map t;
  t["min_neighbour_overlap"] = Record(thresholds.minNeighbourOverlap);
  t["min_frame_lifetime"] =
      Record(static_cast<std::int64_t>(thresholds.minFrameLifetime));
  t["max_transport_leakage"] = Record(thresholds.maxTransportLeakage);
  t["max_resolvent_bound"] = Record(thresholds.maxResolventBound);
  t["require_contour"] = Record(thresholds.requireContour);
  t["lorentzian"] = Record(thresholds.lorentzian);
  t["min_allowability_margin"] = Record(thresholds.minAllowabilityMargin);
  m["thresholds"] = Record(std::move(t));

  return Record(std::move(m));
}

ClusterRegisterRead ClusterRegisterRead::fromRecord(const Record &record) {
  const auto &m = record.asMap();
  const auto version = m.find("schema_version");
  if (version == m.end() ||
      version->second.asInt() < static_cast<std::int64_t>(kOldestReadableSchema) ||
      version->second.asInt() > static_cast<std::int64_t>(kSchemaVersion))
    throw std::invalid_argument(
        "ClusterRegister: unknown schema_version (reader rejects unknown "
        "versions rather than guessing)");
  const auto type = m.find("record_type");
  if (type == m.end() || type->second.asString() != "cluster_register")
    throw std::invalid_argument("ClusterRegister: not a cluster_register record");

  ClusterRegisterRead read;
  read.component = ComponentId(
      m.at("component_hash").asString(),
      static_cast<std::size_t>(m.at("component_level").asInt()));
  read.support = supportFromRecord(m.at("support"));
  read.degree = static_cast<int>(m.at("degree").asInt());
  read.rank = static_cast<std::size_t>(m.at("rank").asInt());
  read.band = SpectralFiber::fromRecord(m.at("band"));

  read.supportConnected = m.at("support_connected").asBool();
  read.supportPieces = static_cast<std::size_t>(m.at("support_pieces").asInt());
  read.localizationExcess = m.at("localization_excess").asDouble();
  read.bandGap = m.at("band_gap").asDouble();
  // Schema 1 carries no contour certificate: absent stays unmeasured.
  read.contour = m.count("contour") ? m.at("contour").asString() : std::string{};
  read.contourNodeCount = m.count("contour_node_count")
                              ? static_cast<int>(m.at("contour_node_count").asInt())
                              : 0;
  read.resolventBound = optionalDouble(m, "resolvent_bound");
  read.allowabilityMargin = optionalDouble(m, "allowability_margin");
  read.lorentzianEpsilon = optionalDouble(m, "lorentzian_epsilon");
  read.neighbourOverlap = m.at("neighbour_overlap").asDouble();
  read.frameLifetime = m.at("frame_lifetime").asDouble();
  read.transportLeakage = m.at("transport_leakage").asDouble();

  const auto &r = m.at("regime_report").asMap();
  read.regime.regime = regimeFromName(r.at("regime").asString());
  read.regime.gramDefect = r.at("gram_defect").asDouble();
  read.regime.positiveSignature =
      static_cast<int>(r.at("positive_signature").asInt());
  read.regime.negativeSignature =
      static_cast<int>(r.at("negative_signature").asInt());
  read.regime.neutralSignature =
      static_cast<int>(r.at("neutral_signature").asInt());
  read.regime.signatureNormalizable = r.at("signature_normalizable").asBool();
  read.regime.eigenResidual = r.at("eigen_residual").asDouble();
  read.regime.leftResidual = r.at("left_residual").asDouble();
  read.regime.frameConditionNumber = r.at("frame_condition_number").asDouble();

  read.failedConjuncts = stringsFromRecord(m.at("failed_conjuncts"));
  read.unmeasured = stringsFromRecord(m.at("unmeasured"));
  read.accepted = m.at("accepted").asBool();

  const auto &t = m.at("thresholds").asMap();
  read.thresholds.minNeighbourOverlap = t.at("min_neighbour_overlap").asDouble();
  read.thresholds.minFrameLifetime =
      static_cast<std::size_t>(t.at("min_frame_lifetime").asInt());
  read.thresholds.maxTransportLeakage =
      t.at("max_transport_leakage").asDouble();
  if (t.count("max_resolvent_bound"))
    read.thresholds.maxResolventBound = t.at("max_resolvent_bound").asDouble();
  if (t.count("require_contour"))
    read.thresholds.requireContour = t.at("require_contour").asBool();
  if (t.count("lorentzian"))
    read.thresholds.lorentzian = t.at("lorentzian").asBool();
  if (t.count("min_allowability_margin"))
    read.thresholds.minAllowabilityMargin =
        t.at("min_allowability_margin").asDouble();
  return read;
}

// ─────────────────────────────────────────────────────────────────────
// ClusterRegister
// ─────────────────────────────────────────────────────────────────────

ClusterRegister::ClusterRegister(ClusterRegisterConfig cfg)
    : cfg_(std::move(cfg)) {}

std::pair<bool, std::size_t> ClusterRegister::supportConnectivity(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<std::uint64_t> &support) {
  if (support.empty() || st == nullptr) return {false, 0};

  const std::unordered_set<std::uint64_t> inside(support.begin(),
                                                 support.end());
  // Adjacency of the induced one-skeleton, built from the complex's edges.
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> adjacency;
  adjacency.reserve(inside.size());
  for (const auto id : inside) adjacency[id];

  const auto &edges = st->getEdgeList();
  if (!edges) return {false, 0};
  for (const auto *edge : edges->toVector()) {
    if (edge == nullptr) continue;
    const auto *source = edge->getSource();
    const auto *target = edge->getTarget();
    if (source == nullptr || target == nullptr) continue;
    const auto a = source->getId();
    const auto b = target->getId();
    if (a == b) continue;
    if (inside.count(a) == 0 || inside.count(b) == 0) continue;
    adjacency[a].push_back(b);
    adjacency[b].push_back(a);
  }

  // Count connected pieces by flood fill.
  std::unordered_set<std::uint64_t> seen;
  seen.reserve(inside.size());
  std::size_t pieces = 0;
  for (const auto start : inside) {
    if (seen.count(start) != 0) continue;
    ++pieces;
    std::vector<std::uint64_t> frontier{start};
    seen.insert(start);
    while (!frontier.empty()) {
      const auto current = frontier.back();
      frontier.pop_back();
      for (const auto next : adjacency[current]) {
        if (seen.insert(next).second) frontier.push_back(next);
      }
    }
  }
  return {pieces == 1, pieces};
}

ClusterRegisterRead ClusterRegister::read(
    const std::shared_ptr<Spacetime> &st,
    const std::vector<std::uint64_t> &support, const SpectralFiber &band,
    const std::optional<FrameTrack> &track,
    const std::vector<FiberTransportRead> &externalTransports,
    ComponentId component) const {
  ClusterRegisterRead read;
  read.component = std::move(component);
  read.support = support;
  read.band = band;
  read.degree = band.degree();
  read.rank = band.rank();
  read.thresholds = cfg_;

  const SpectralBandCertificate &cert = band.certificate();

  // Classify the band's operator regime from its certificate.
  read.regime.regime =
      cert.certificate.regime() == CertificateRegime::ComplexSymmetricPencil
          ? CertificateRegime::ComplexSymmetricPencil  // bilinear: no inertia
          : cert.selfAdjoint
                ? (cert.negativeSignature > 0
                       ? CertificateRegime::HermitianIndefinite
                       : CertificateRegime::PositiveSemidefinite)
                : CertificateRegime::NonNormal;
  read.regime.gramDefect = cert.gramDefect;
  read.regime.positiveSignature = cert.positiveSignature;
  read.regime.negativeSignature = cert.negativeSignature;
  read.regime.neutralSignature =
      static_cast<int>(cert.rank) - cert.positiveSignature -
      cert.negativeSignature;
  read.regime.signatureNormalizable = read.regime.neutralSignature == 0;
  read.regime.eigenResidual = cert.eigenResidual;
  read.regime.leftResidual = cert.leftResidual;
  read.regime.frameConditionNumber = cert.frameConditionNumber;

  // conjunct 1: connected cluster support
  if (st == nullptr) {
    read.unmeasured.emplace_back(RegisterUnmeasured::kSupportUnreadable);
  } else {
    const auto [connected, pieces] = supportConnectivity(st, support);
    read.supportConnected = connected;
    read.supportPieces = pieces;
    if (pieces == 0)
      read.unmeasured.emplace_back(RegisterUnmeasured::kSupportUnreadable);
    else if (!connected)
      read.failedConjuncts.emplace_back(RegisterConjunct::kClusterSupport);
  }

  const bool haveBand = band.rank() > 0;
  if (!haveBand) read.unmeasured.emplace_back(RegisterUnmeasured::kNoBand);

  // conjunct 2: a localized spectral projector with stable rank
  read.localizationExcess = cert.localizationExcess;
  if (haveBand) {
    if (!measured(read.localizationExcess))
      read.unmeasured.emplace_back(RegisterUnmeasured::kLocalizationUnmeasured);
    else if (!cert.accepted)
      // The localization cap is enforced by the detector; an uncertified band
      // fails this conjunct by name.
      read.failedConjuncts.emplace_back(RegisterConjunct::kLocalizedProjector);
  }

  // conjunct 3: a closed complex-plane contour with nonzero separation and a
  // controlled resolvent separating the band from the discarded modes, taken
  // at the reported rotation eps_L > 0 for a Lorentzian complex. Its three
  // measurements are decided, and named, separately.
  read.bandGap = cert.nearestDiscardedSeparation;
  read.contour = cert.contour;
  read.contourNodeCount = cert.contourNodeCount;
  read.resolventBound = cert.resolventBound;
  read.allowabilityMargin = cert.allowabilityMargin;
  read.lorentzianEpsilon = cert.lorentzianEpsilon;
  if (haveBand) {
    if (!measured(read.bandGap))
      read.unmeasured.emplace_back(RegisterUnmeasured::kBandGapUnknown);
    else if (read.bandGap <= 0.0)
      read.failedConjuncts.emplace_back(RegisterConjunct::kBandGap);

    // The contour half. A band the detector grouped by the sort-and-gap rule
    // carries no contour and claims none: it is unmeasured here only when a
    // contour was required.
    const bool contourDrawn = !cert.contour.empty() && cert.contourNodeCount > 0;
    if (!contourDrawn) {
      if (cfg_.requireContour)
        read.unmeasured.emplace_back(RegisterUnmeasured::kNoContour);
    } else if (!measured(read.resolventBound)) {
      read.unmeasured.emplace_back(RegisterUnmeasured::kResolventUnmeasured);
    } else if (!(read.resolventBound <= cfg_.maxResolventBound)) {
      read.failedConjuncts.emplace_back(RegisterConjunct::kContourResolvent);
    }

    // The rotation half, decided only for a complex the caller declared
    // Lorentzian. Nothing here infers the declaration from the band.
    if (cfg_.lorentzian) {
      if (!measured(read.lorentzianEpsilon) ||
          !measured(read.allowabilityMargin)) {
        read.unmeasured.emplace_back(RegisterUnmeasured::kRotationUnmeasured);
      } else if (!(read.lorentzianEpsilon > 0.0) || !cert.allowable ||
                 !(read.allowabilityMargin > cfg_.minAllowabilityMargin)) {
        read.failedConjuncts.emplace_back(RegisterConjunct::kLorentzianRotation);
      }
    }
  }

  // conjuncts 4 and 5: neighbour overlap and cobordism-frame lifetime
  if (!track.has_value()) {
    read.unmeasured.emplace_back(RegisterUnmeasured::kNoFrameTrack);
  } else {
    read.neighbourOverlap = track->minAdjacentOverlap;
    read.frameLifetime = static_cast<double>(track->frames());
    if (read.neighbourOverlap < cfg_.minNeighbourOverlap)
      read.failedConjuncts.emplace_back(RegisterConjunct::kNeighbourOverlap);
    if (track->frames() < cfg_.minFrameLifetime)
      read.failedConjuncts.emplace_back(RegisterConjunct::kFrameLifetime);
  }

  // conjunct 6: small external transport leakage
  if (externalTransports.empty()) {
    read.unmeasured.emplace_back(RegisterUnmeasured::kNoTransport);
  } else {
    double worst = 0.0;
    bool any = false;
    for (const auto &transport : externalTransports) {
      if (!measured(transport.leakage)) continue;
      worst = std::max(worst, transport.leakage);
      any = true;
    }
    if (!any) {
      read.unmeasured.emplace_back(RegisterUnmeasured::kNoTransport);
    } else {
      read.transportLeakage = worst;
      if (worst > cfg_.maxTransportLeakage)
        read.failedConjuncts.emplace_back(RegisterConjunct::kTransportLeakage);
    }
  }

  read.accepted = read.failedConjuncts.empty() && read.unmeasured.empty() &&
                  haveBand && cert.accepted;

  read.certificate =
      read.accepted
          ? Certificate::certifiedNumerical(
                CertificateDomain::BandWindow, read.regime.regime,
                measured(cert.gramDefect) ? cert.gramDefect
                                          : Certificate::kUnmeasured,
                cert.projectorNorm, Certificate::kUnmeasured)
          : Certificate::heuristicDiscovery(CertificateDomain::BandWindow,
                                            read.regime.regime);
  return read;
}

}  // namespace tessera::observables
