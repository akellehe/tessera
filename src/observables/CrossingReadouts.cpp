// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

/// \file
/// World-tube crossing readouts on level sets of the Lorentzian distance from
/// the incoming boundary: crossing mass, baryon number, and the spectral
/// charge-power profile.
/// Reference: Ambjorn, Goerlich, Jurkiewicz, Loll, "Nonperturbative Quantum
/// Gravity", arXiv:1203.3591

#include "observables/CrossingReadouts.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Eigenvalues>

#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kUnreached = static_cast<std::size_t>(-1);

using EdgeKeyPair = std::array<std::uint64_t, 2>;

/// The endpoint pair in ascending vertex order — the one key convention for
/// an undirected edge in this file.
EdgeKeyPair edgeKey(std::uint64_t a, std::uint64_t b) {
  return a <= b ? EdgeKeyPair{a, b} : EdgeKeyPair{b, a};
}

/// The future-directed proper time of a causal edge with complex squared
/// length `z`: `sqrt(-z)` on the branch whose real part is non-negative.
/// The branch is fixed here rather than left to the principal root, so the
/// sign convention is explicit.
std::complex<double> properTime(std::complex<double> z) {
  std::complex<double> root = std::sqrt(-z);
  if (root.real() < 0.0 || (root.real() == 0.0 && root.imag() < 0.0)) {
    root = -root;
  }
  return root;
}

/// Append `name` to `into` when it is not already present, so a repeated
/// certificate failure is named once.
void nameFailure(std::vector<std::string> &into, const std::string &name) {
  if (std::find(into.begin(), into.end(), name) == into.end()) {
    into.push_back(name);
  }
}

Record::List stringList(const std::vector<std::string> &values) {
  Record::List list;
  list.reserve(values.size());
  for (const auto &value : values) list.emplace_back(value);
  return list;
}

Record::List doubleList(const std::vector<double> &values) {
  Record::List list;
  list.reserve(values.size());
  for (double value : values) list.emplace_back(value);
  return list;
}

Record optionalDouble(const std::optional<double> &value) {
  return value.has_value() ? Record(*value) : Record(nullptr);
}

}  // namespace

// ─── configuration ───────────────────────────────────────────────────────

Record CrossingReadoutsConfig::toRecord() const {
  Record::Map map;
  map["kappa_mass"] = Record(kappaMass);
  map["mass_calibrated"] = Record(massCalibrated);
  map["sign_tolerance"] = Record(signTolerance);
  map["degeneracy_tolerance"] = Record(degeneracyTolerance);
  map["monopole_tolerance"] = Record(monopoleTolerance);
  return Record(std::move(map));
}

// ─── the temporal function ───────────────────────────────────────────────

std::complex<double> TemporalFunctionRead::at(std::uint64_t vertex) const {
  const auto it = std::lower_bound(vertices.begin(), vertices.end(), vertex);
  if (it == vertices.end() || *it != vertex) return {kNaN, kNaN};
  return tau[static_cast<std::size_t>(it - vertices.begin())];
}

Record TemporalFunctionRead::toRecord() const {
  Record::Map map;
  Record::List vertexList;
  vertexList.reserve(vertices.size());
  for (std::uint64_t id : vertices) {
    vertexList.emplace_back(static_cast<std::int64_t>(id));
  }
  map["vertices"] = Record(std::move(vertexList));
  Record::splitComplex(map, "tau", tau);
  Record::List layerList;
  layerList.reserve(layer.size());
  for (std::size_t value : layer) {
    layerList.emplace_back(value == kUnreached
                               ? Record(nullptr)
                               : Record(static_cast<std::int64_t>(value)));
  }
  map["layer"] = Record(std::move(layerList));
  map["certified"] = Record(certified);
  map["failed_certificates"] = Record(stringList(failedCertificates));
  map["min_causal_increment"] = Record(minCausalIncrement);
  map["causal_edge_count"] =
      Record(static_cast<std::int64_t>(causalEdgeCount));
  map["unreachable_count"] =
      Record(static_cast<std::int64_t>(unreachableCount));
  return Record(std::move(map));
}

TemporalFunctionRead CrossingReadouts::temporalFunction(
    const std::shared_ptr<Spacetime> &spacetime,
    const std::vector<std::uint64_t> &m0Vertices,
    const CrossingReadoutsConfig &cfg) {
  (void)cfg;
  if (!spacetime) {
    throw std::invalid_argument(
        "CrossingReadouts::temporalFunction: null spacetime");
  }

  TemporalFunctionRead read;

  // Every vertex id, ascending — the read's index order.
  const auto &edges = spacetime->getEdgeList()->toVector();
  std::set<std::uint64_t> vertexSet;
  for (const auto &edge : edges) {
    if (!edge || !edge->getSource() || !edge->getTarget()) continue;
    vertexSet.insert(edge->getSource()->getId());
    vertexSet.insert(edge->getTarget()->getId());
  }
  read.vertices.assign(vertexSet.begin(), vertexSet.end());
  const std::size_t n = read.vertices.size();
  read.tau.assign(n, {kNaN, kNaN});
  read.layer.assign(n, kUnreached);

  std::unordered_map<std::uint64_t, std::size_t> index;
  index.reserve(n * 2);
  for (std::size_t i = 0; i < n; ++i) index.emplace(read.vertices[i], i);

  // Adjacency over the 1-skeleton, carrying each edge's complex length.
  struct Incidence {
    std::size_t other;
    std::complex<double> length;
    bool causal;
    bool null;
  };
  std::vector<std::vector<Incidence>> adjacency(n);
  for (const auto &edge : edges) {
    if (!edge || !edge->getSource() || !edge->getTarget()) continue;
    const auto sourceIt = index.find(edge->getSource()->getId());
    const auto targetIt = index.find(edge->getTarget()->getId());
    if (sourceIt == index.end() || targetIt == index.end()) continue;
    if (sourceIt->second == targetIt->second) continue;
    const std::complex<double> length = edge->getLength();
    const bool isNull = edge->isNull();
    const bool causal = edge->isTimelike() || isNull;
    adjacency[sourceIt->second].push_back(
        {targetIt->second, length, causal, isNull});
    adjacency[targetIt->second].push_back(
        {sourceIt->second, length, causal, isNull});
  }

  std::vector<std::size_t> seeds;
  for (std::uint64_t id : m0Vertices) {
    const auto it = index.find(id);
    if (it != index.end()) seeds.push_back(it->second);
  }
  if (seeds.empty()) {
    nameFailure(read.failedCertificates, "empty-boundary");
    read.unreachableCount = n;
    return read;
  }

  // The time orientation induced by M0: the combinatorial layer is the
  // breadth-first-search (BFS) hop distance from M0 in the 1-skeleton, and an
  // edge from a lower to a higher layer points to the future.  This is
  // intrinsic to the cobordism and its incoming boundary; no vertex
  // coordinate is read.
  std::deque<std::size_t> queue;
  for (std::size_t seed : seeds) {
    if (read.layer[seed] != 0) {
      read.layer[seed] = 0;
      queue.push_back(seed);
    }
  }
  while (!queue.empty()) {
    const std::size_t current = queue.front();
    queue.pop_front();
    for (const auto &incidence : adjacency[current]) {
      if (read.layer[incidence.other] == kUnreached) {
        read.layer[incidence.other] = read.layer[current] + 1;
        queue.push_back(incidence.other);
      }
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (read.layer[i] == kUnreached) ++read.unreachableCount;
  }
  if (read.unreachableCount != 0) {
    nameFailure(read.failedCertificates, "unreachable-vertices");
  }

  // tau accumulates the proper time of future-directed causal edges along the
  // path maximizing Re tau: the discrete Lorentzian distance from a
  // hypersurface, a supremum over causal curves.  Layer order is a
  // topological order of the induced orientation, so one sweep suffices.
  std::vector<std::size_t> order;
  order.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (read.layer[i] != kUnreached) order.push_back(i);
  }
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) {
              if (read.layer[a] != read.layer[b]) {
                return read.layer[a] < read.layer[b];
              }
              return read.vertices[a] < read.vertices[b];
            });

  for (std::size_t seed : seeds) read.tau[seed] = {0.0, 0.0};

  std::size_t causalWithinLayer = 0;
  std::size_t nullCausal = 0;
  for (std::size_t current : order) {
    if (read.layer[current] == 0) continue;
    bool found = false;
    std::complex<double> best{kNaN, kNaN};
    for (const auto &incidence : adjacency[current]) {
      if (!incidence.causal) continue;
      const std::size_t other = incidence.other;
      if (read.layer[other] == kUnreached) continue;
      if (read.layer[other] >= read.layer[current]) continue;
      if (std::isnan(read.tau[other].real())) continue;
      if (incidence.null) ++nullCausal;
      const std::complex<double> candidate =
          read.tau[other] + properTime(incidence.length * incidence.length);
      if (!found || candidate.real() > best.real()) {
        best = candidate;
        found = true;
      }
    }
    if (found) read.tau[current] = best;
  }

  // Certificates: Re tau strictly increasing along every future-directed
  // causal edge, no causal edge inside one layer, no null causal edge.
  double minIncrement = std::numeric_limits<double>::infinity();
  std::size_t causalEdges = 0;
  bool nonmonotone = false;
  bool sawUnreachedTau = false;
  for (std::size_t i = 0; i < n; ++i) {
    for (const auto &incidence : adjacency[i]) {
      if (!incidence.causal) continue;
      const std::size_t other = incidence.other;
      if (i > other) continue;  // each undirected edge once
      const std::size_t layerA = read.layer[i];
      const std::size_t layerB = read.layer[other];
      if (layerA == kUnreached || layerB == kUnreached) continue;
      if (layerA == layerB) {
        ++causalWithinLayer;
        continue;
      }
      const std::size_t past = layerA < layerB ? i : other;
      const std::size_t future = layerA < layerB ? other : i;
      if (std::isnan(read.tau[past].real()) ||
          std::isnan(read.tau[future].real())) {
        sawUnreachedTau = true;
        continue;
      }
      ++causalEdges;
      const double increment = read.tau[future].real() - read.tau[past].real();
      minIncrement = std::min(minIncrement, increment);
      if (!(increment > 0.0)) nonmonotone = true;
    }
  }
  read.causalEdgeCount = causalEdges;
  read.minCausalIncrement =
      causalEdges == 0 ? kNaN : minIncrement;

  if (causalWithinLayer != 0) {
    // A causal edge joining two vertices of the same layer cannot be
    // ordered by the induced time orientation: the causal relation is not
    // acyclic with respect to it.
    nameFailure(read.failedCertificates, "causal-cycle");
  }
  if (nullCausal != 0) nameFailure(read.failedCertificates, "null-causal-edge");
  if (nonmonotone) {
    nameFailure(read.failedCertificates, "nonmonotone-temporal-function");
  }
  if (sawUnreachedTau) {
    nameFailure(read.failedCertificates, "unreachable-vertices");
  }
  if (causalEdges == 0) {
    nameFailure(read.failedCertificates, "no-causal-edges");
  }

  read.certified = read.failedCertificates.empty();
  return read;
}

// ─── the band density on the 1-skeleton ──────────────────────────────────

std::map<EdgeKeyPair, double> CrossingReadouts::bandEdgeDensity(
    const SpectralFiber &band) {
  std::map<EdgeKeyPair, double> density;
  if (band.degree() <= 0 || band.rank() == 0) return density;

  const auto &cells = band.cellVertices();
  if (cells.empty()) return density;
  const Eigen::MatrixXcd projector = band.projector();
  if (projector.rows() != static_cast<Eigen::Index>(cells.size())) {
    return density;
  }

  for (std::size_t i = 0; i < cells.size(); ++i) {
    const double weight = std::abs(projector(static_cast<Eigen::Index>(i),
                                             static_cast<Eigen::Index>(i)));
    if (!(weight > 0.0)) continue;
    const auto &cell = cells[i];
    if (cell.size() < 2) continue;
    if (cell.size() == 2) {
      density[edgeKey(cell[0], cell[1])] += weight;
      continue;
    }
    // A degree-k band (k >= 2) spreads each cell's density uniformly over
    // that cell's boundary edges, so the density lives on the 1-skeleton the
    // crossing set is defined on.
    const std::size_t pairs = cell.size() * (cell.size() - 1) / 2;
    const double share = weight / static_cast<double>(pairs);
    for (std::size_t a = 0; a + 1 < cell.size(); ++a) {
      for (std::size_t b = a + 1; b < cell.size(); ++b) {
        density[edgeKey(cell[a], cell[b])] += share;
      }
    }
  }
  return density;
}

// ─── one tube's crossing of one level ────────────────────────────────────

Record TubeCrossingRead::toRecord() const {
  Record::Map map;
  map["tube_id"] = Record(tubeId);
  map["level"] = Record(level);
  Record::List edgeList;
  edgeList.reserve(crossingEdges.size());
  for (const auto &pair : crossingEdges) {
    Record::List entry;
    entry.emplace_back(static_cast<std::int64_t>(pair[0]));
    entry.emplace_back(static_cast<std::int64_t>(pair[1]));
    edgeList.emplace_back(std::move(entry));
  }
  map["crossing_edges"] = Record(std::move(edgeList));
  map["density"] = Record(doubleList(density));
  Record::splitComplex(map, "perpendicular", perpendicular);
  map["sign"] = sign == 0 ? Record(nullptr) : Record(sign);
  map["admissible"] = Record(admissible);
  map["failed_certificates"] = Record(stringList(failedCertificates));
  return Record(std::move(map));
}

TubeCrossingRead CrossingReadouts::crossing(
    const WorldTubeInput &tube, const TemporalFunctionRead &temporal,
    double level, const CrossingReadoutsConfig &cfg) {
  TubeCrossingRead read;
  read.tubeId = tube.tubeId;
  read.level = level;

  if (!temporal.certified) {
    nameFailure(read.failedCertificates, "uncertified-temporal-function");
  }
  const auto &certificate = tube.band.certificate();
  if (!certificate.accepted) {
    nameFailure(read.failedCertificates, "band-unaccepted");
  }
  // The positivity certificate: a band whose restricted metric is not
  // positive supplies no covariance and therefore no particle reading.
  const bool positive =
      certificate.rank != 0 &&
      certificate.positiveSignature == static_cast<int>(certificate.rank) &&
      certificate.negativeSignature == 0;
  if (!positive) nameFailure(read.failedCertificates, "band-positivity");
  if (tube.band.degree() <= 0) {
    nameFailure(read.failedCertificates, "degree-zero-band");
  }
  if (!read.failedCertificates.empty()) return read;

  const auto density = bandEdgeDensity(tube.band);
  const int orientation = tube.orientation >= 0 ? 1 : -1;

  std::vector<double> weights;
  std::vector<std::complex<double>> increments;
  bool nonregular = false;
  for (const auto &entry : density) {
    const std::uint64_t a = entry.first[0];
    const std::uint64_t b = entry.first[1];
    const std::complex<double> tauA = temporal.at(a);
    const std::complex<double> tauB = temporal.at(b);
    if (std::isnan(tauA.real()) || std::isnan(tauB.real())) continue;
    // A level passing exactly through a vertex is a nonregular level: the
    // crossing is not transversal and the readout refuses.
    if (tauA.real() == level || tauB.real() == level) {
      nonregular = true;
      continue;
    }
    const bool aIsPast = tauA.real() < tauB.real();
    const double lower = aIsPast ? tauA.real() : tauB.real();
    const double upper = aIsPast ? tauB.real() : tauA.real();
    if (!(lower < level && level < upper)) continue;

    const std::complex<double> tauPast = aIsPast ? tauA : tauB;
    const std::complex<double> tauFuture = aIsPast ? tauB : tauA;
    read.crossingEdges.push_back({aIsPast ? a : b, aIsPast ? b : a});
    weights.push_back(entry.second);
    increments.push_back(static_cast<double>(orientation) *
                         (tauFuture - tauPast));
  }

  if (nonregular) nameFailure(read.failedCertificates, "nonregular-level");
  if (read.crossingEdges.empty()) {
    nameFailure(read.failedCertificates, "empty-crossing");
    return read;
  }

  double total = 0.0;
  for (double weight : weights) total += weight;
  if (!(total > 0.0)) {
    nameFailure(read.failedCertificates, "empty-crossing");
    return read;
  }
  read.density.reserve(weights.size());
  for (double weight : weights) read.density.push_back(weight / total);

  // Timelike and transversal: every contribution nonvanishing with a single
  // sign across the crossing set.
  int commonSign = 0;
  for (std::size_t i = 0; i < increments.size(); ++i) {
    const double part = increments[i].real();
    if (std::abs(part) <= cfg.signTolerance) {
      nameFailure(read.failedCertificates, "grazing-crossing");
      continue;
    }
    const int partSign = part > 0.0 ? 1 : -1;
    if (commonSign == 0) {
      commonSign = partSign;
    } else if (commonSign != partSign) {
      nameFailure(read.failedCertificates, "mixed-sign-crossing");
    }
  }

  std::complex<double> perpendicular{0.0, 0.0};
  for (std::size_t i = 0; i < increments.size(); ++i) {
    perpendicular += read.density[i] * increments[i];
  }
  read.perpendicular = perpendicular;

  if (std::abs(perpendicular.real()) <= cfg.signTolerance) {
    nameFailure(read.failedCertificates, "grazing-crossing");
  }
  if (!read.failedCertificates.empty()) {
    read.sign = 0;
    return read;
  }

  read.sign = perpendicular.real() > 0.0 ? 1 : -1;
  read.admissible = true;
  return read;
}

// ─── the two sums ────────────────────────────────────────────────────────

Record CrossingMassRead::toRecord() const {
  Record::Map map;
  map["level"] = Record(level);
  map["crossing_mass"] = Record(crossingMass);
  map["level_sum"] = Record(levelSum);
  map["reference_sum"] = Record(referenceSum);
  map["kappa_mass"] = Record(kappaMass);
  map["calibrated"] = Record(calibrated);
  map["units"] = Record(units);
  map["admissible_crossings"] =
      Record(static_cast<std::int64_t>(admissibleCrossings));
  map["refused_crossings"] =
      Record(static_cast<std::int64_t>(refusedCrossings));
  return Record(std::move(map));
}

namespace {

/// The incoherent modulus sum on one level, plus the admissible / refused
/// counts.  Returns NaN when no crossing was admissible.
struct ModulusSum {
  double value = kNaN;
  std::size_t admissible = 0;
  std::size_t refused = 0;
};

ModulusSum modulusSumAt(const std::vector<WorldTubeInput> &tubes,
                        const TemporalFunctionRead &temporal, double level,
                        const CrossingReadoutsConfig &cfg) {
  ModulusSum sum;
  double total = 0.0;
  for (const auto &tube : tubes) {
    const TubeCrossingRead read =
        CrossingReadouts::crossing(tube, temporal, level, cfg);
    if (!read.admissible) {
      ++sum.refused;
      continue;
    }
    total += std::abs(read.perpendicular);
    ++sum.admissible;
  }
  if (sum.admissible != 0) sum.value = total;
  return sum;
}

}  // namespace

CrossingMassRead CrossingReadouts::crossingMass(
    const std::vector<WorldTubeInput> &tubes,
    const TemporalFunctionRead &temporal, double level, double m0Level,
    const CrossingReadoutsConfig &cfg) {
  CrossingMassRead read;
  read.level = level;
  read.kappaMass = cfg.kappaMass;
  read.calibrated = cfg.massCalibrated;
  read.units = cfg.massCalibrated ? std::string("calibrated")
                                  : std::string("uncalibrated");

  const ModulusSum atLevel = modulusSumAt(tubes, temporal, level, cfg);
  const ModulusSum atReference = modulusSumAt(tubes, temporal, m0Level, cfg);
  read.admissibleCrossings = atLevel.admissible;
  read.refusedCrossings = atLevel.refused;
  read.levelSum =
      std::isnan(atLevel.value) ? kNaN : cfg.kappaMass * atLevel.value;
  read.referenceSum =
      std::isnan(atReference.value) ? kNaN : cfg.kappaMass * atReference.value;

  // Every readout is the difference against the same sum evaluated at M0.
  // An absent reference sum is a known zero contribution only when the
  // reference surface genuinely carries no admissible crossing.
  const double levelValue = std::isnan(read.levelSum) ? 0.0 : read.levelSum;
  const double referenceValue =
      std::isnan(read.referenceSum) ? 0.0 : read.referenceSum;
  if (std::isnan(read.levelSum) && std::isnan(read.referenceSum)) {
    read.crossingMass = kNaN;
  } else {
    read.crossingMass = levelValue - referenceValue;
  }
  return read;
}

Record BaryonCrossingRead::toRecord() const {
  Record::Map map;
  map["level"] = Record(level);
  map["baryon_number"] = optionalDouble(baryonNumber);
  map["level_sum"] = optionalDouble(levelSum);
  map["reference_sum"] = optionalDouble(referenceSum);
  map["quark_tubes"] = Record(static_cast<std::int64_t>(quarkTubes));
  map["sign_defects"] = Record(stringList(signDefects));
  map["winding_agreements"] =
      Record(static_cast<std::int64_t>(windingAgreements));
  return Record(std::move(map));
}

BaryonCrossingRead CrossingReadouts::baryonNumber(
    const std::vector<WorldTubeInput> &tubes,
    const TemporalFunctionRead &temporal, double level, double m0Level,
    const CrossingReadoutsConfig &cfg) {
  BaryonCrossingRead read;
  read.level = level;

  const auto coherentSum =
      [&](double at, bool collectDefects) -> std::optional<double> {
    int signTotal = 0;
    std::size_t counted = 0;
    for (const auto &tube : tubes) {
      if (!tube.certifiedQuarkTube) continue;
      const TubeCrossingRead crossingRead =
          CrossingReadouts::crossing(tube, temporal, at, cfg);
      if (!crossingRead.admissible) continue;
      signTotal += crossingRead.sign;
      ++counted;
      if (!collectDefects) continue;
      ++read.quarkTubes;
      // The crossing sign and the determinant-line winding must agree on
      // every certified tube.  Disagreement is reported as a defect; the tube
      // is neither resolved nor dropped from the sum.
      if (tube.determinantWinding.has_value() &&
          *tube.determinantWinding != 0) {
        const int windingSign = *tube.determinantWinding > 0 ? 1 : -1;
        if (windingSign != crossingRead.sign) {
          read.signDefects.push_back(tube.tubeId);
        } else {
          ++read.windingAgreements;
        }
      }
    }
    if (counted == 0) return std::nullopt;
    return static_cast<double>(signTotal) / 3.0;
  };

  read.levelSum = coherentSum(level, true);
  read.referenceSum = coherentSum(m0Level, false);
  if (read.levelSum.has_value() || read.referenceSum.has_value()) {
    read.baryonNumber = read.levelSum.value_or(0.0) -
                        read.referenceSum.value_or(0.0);
  }
  return read;
}

// ─── the spectral charge-power profile ───────────────────────────────────

Record ChargePowerProfileRead::toRecord() const {
  Record::Map map;
  map["level"] = Record(level);
  map["eigenvalues"] = Record(doubleList(eigenvalues));
  map["power"] = Record(doubleList(power));
  map["normalized_power"] = Record(doubleList(normalizedPower));
  map["monopole"] = Record(monopole);
  map["normalized"] = Record(normalized);
  map["failed_certificates"] = Record(stringList(failedCertificates));
  map["slice_nodes"] = Record(static_cast<std::int64_t>(sliceNodes));
  return Record(std::move(map));
}

ChargePowerProfileRead CrossingReadouts::chargePowerProfile(
    const std::vector<WorldTubeInput> &tubes,
    const TemporalFunctionRead &temporal, double level,
    const CrossingReadoutsConfig &cfg) {
  ChargePowerProfileRead read;
  read.level = level;

  // The slice: every crossing edge of every admissible tube is one node,
  // carrying that tube's signed unit spread by the band density.
  std::map<EdgeKeyPair, double> charge;
  for (const auto &tube : tubes) {
    const TubeCrossingRead crossingRead =
        CrossingReadouts::crossing(tube, temporal, level, cfg);
    if (!crossingRead.admissible) continue;
    for (std::size_t i = 0; i < crossingRead.crossingEdges.size(); ++i) {
      const auto &pair = crossingRead.crossingEdges[i];
      charge[edgeKey(pair[0], pair[1])] +=
          static_cast<double>(crossingRead.sign) * crossingRead.density[i];
    }
  }

  read.sliceNodes = charge.size();
  if (charge.empty()) {
    nameFailure(read.failedCertificates, "empty-slice");
    return read;
  }

  std::vector<EdgeKeyPair> nodes;
  Eigen::VectorXd rho(static_cast<Eigen::Index>(charge.size()));
  nodes.reserve(charge.size());
  for (const auto &entry : charge) {
    rho(static_cast<Eigen::Index>(nodes.size())) = entry.second;
    nodes.push_back(entry.first);
  }

  // The slice Laplacian is the graph Laplacian on the crossing set: two
  // crossing edges are adjacent when they share a vertex.
  const Eigen::Index size = static_cast<Eigen::Index>(nodes.size());
  Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(size, size);
  for (Eigen::Index i = 0; i < size; ++i) {
    for (Eigen::Index j = i + 1; j < size; ++j) {
      const bool shared =
          nodes[static_cast<std::size_t>(i)][0] ==
              nodes[static_cast<std::size_t>(j)][0] ||
          nodes[static_cast<std::size_t>(i)][0] ==
              nodes[static_cast<std::size_t>(j)][1] ||
          nodes[static_cast<std::size_t>(i)][1] ==
              nodes[static_cast<std::size_t>(j)][0] ||
          nodes[static_cast<std::size_t>(i)][1] ==
              nodes[static_cast<std::size_t>(j)][1];
      if (!shared) continue;
      laplacian(i, j) -= 1.0;
      laplacian(j, i) -= 1.0;
      laplacian(i, i) += 1.0;
      laplacian(j, j) += 1.0;
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(laplacian);
  if (solver.info() != Eigen::Success) {
    nameFailure(read.failedCertificates, "slice-eigensolve-failed");
    return read;
  }
  const Eigen::VectorXd values = solver.eigenvalues();
  const Eigen::MatrixXd vectors = solver.eigenvectors();
  const Eigen::VectorXd overlaps = vectors.transpose() * rho;

  // Eigenvalues within `degeneracyTolerance` share one eigenspace projector,
  // so `<rho, P_lambda rho>` is a sum of squared overlaps over the whole
  // eigenspace: basis- and phase-invariant, degeneracies handled.
  Eigen::Index start = 0;
  while (start < values.size()) {
    Eigen::Index stop = start + 1;
    while (stop < values.size() &&
           std::abs(values(stop) - values(start)) <= cfg.degeneracyTolerance) {
      ++stop;
    }
    double power = 0.0;
    double centre = 0.0;
    for (Eigen::Index i = start; i < stop; ++i) {
      power += overlaps(i) * overlaps(i);
      centre += values(i);
    }
    read.eigenvalues.push_back(centre / static_cast<double>(stop - start));
    read.power.push_back(power);
    start = stop;
  }

  // The normalizing monopole is the power of the eigenspace at lambda = 0.
  double monopole = kNaN;
  for (std::size_t i = 0; i < read.eigenvalues.size(); ++i) {
    if (std::abs(read.eigenvalues[i]) <= cfg.degeneracyTolerance) {
      monopole = read.power[i];
      break;
    }
  }
  read.monopole = monopole;
  if (std::isnan(monopole) || std::abs(monopole) <= cfg.monopoleTolerance) {
    // A neutral system: the normalizing monopole vanishes, the normalized
    // profile refuses, and the unnormalized values above remain reported.
    nameFailure(read.failedCertificates, "neutral-system");
    return read;
  }
  read.normalizedPower.reserve(read.power.size());
  for (double value : read.power) read.normalizedPower.push_back(value / monopole);
  read.normalized = true;
  return read;
}

// ─── the conditional form factor ─────────────────────────────────────────

Record ElectromagneticFormFactorRead::toRecord() const {
  Record::Map map;
  map["available"] = Record(available);
  map["charge_radius_squared"] = optionalDouble(chargeRadiusSquared);
  map["failed_certificates"] = Record(stringList(failedCertificates));
  map["note"] = Record(note);
  return Record(std::move(map));
}

ElectromagneticFormFactorRead CrossingReadouts::formFactor(
    const ChargePowerProfileRead &profile, const CrossingReadoutsConfig &cfg) {
  (void)profile;
  (void)cfg;
  // The electric form factor G_E needs a certified conserved U(1) current,
  // certified momentum-transfer states, and a small-Q^2 refinement
  // extrapolation.  None of the three is available here, so the charge radius
  // is reported as unavailable with each missing certificate named.  The
  // spectral charge-power profile is an incoherent structure factor and is
  // not substituted for it.
  ElectromagneticFormFactorRead read;
  read.available = false;
  read.failedCertificates = {"no-certified-conserved-current",
                             "no-certified-momentum-states",
                             "no-refinement-extrapolation"};
  return read;
}

// ─── the overlay block ───────────────────────────────────────────────────

Record CrossingReadouts::overlayRecord(
    const std::vector<WorldTubeInput> &tubes,
    const TemporalFunctionRead &temporal, double level, double m0Level,
    const CrossingReadoutsConfig &cfg) {
  Record::Map map;
  map["schema_version"] = Record(kSchemaVersion);
  map["level"] = Record(level);
  map["reference_level"] = Record(m0Level);
  map["thresholds"] = cfg.toRecord();
  map["temporal_function"] = temporal.toRecord();

  Record::List crossingList;
  crossingList.reserve(tubes.size());
  for (const auto &tube : tubes) {
    crossingList.emplace_back(crossing(tube, temporal, level, cfg).toRecord());
  }
  map["crossings"] = Record(std::move(crossingList));

  map["crossing_mass"] =
      crossingMass(tubes, temporal, level, m0Level, cfg).toRecord();
  map["baryon_number"] =
      baryonNumber(tubes, temporal, level, m0Level, cfg).toRecord();
  const ChargePowerProfileRead profile =
      chargePowerProfile(tubes, temporal, level, cfg);
  map["charge_power_profile"] = profile.toRecord();
  map["form_factor"] = formFactor(profile, cfg).toRecord();
  return Record(std::move(map));
}

}  // namespace tessera::observables
