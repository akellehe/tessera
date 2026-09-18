// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/ObservableGates.h"

#include <complex>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

#include "cobordism/MultiCobordism.h"
#include "observables/LiveComplex.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::cobordism::MultiCobordism;

double ObservableGates::gaugeDelta(const RegisterObservable &observable,
                                   const RegisterContext &ctx) {
  const Record base = observable.record(ctx);
  const auto gauged = ctx.gauged(GAUGE_THETA);
  return Record::reportDelta(base, observable.record(*gauged));
}

double ObservableGates::relabelDelta(const RegisterObservable &observable,
                                     const RegisterContext &ctx) {
  const Record base = observable.record(ctx);

  // The relabeled complex is rebuilt by the loader.
  const LiveComplex::Relabeled rel =
      LiveComplex::relabel(*ctx.spacetime(), GATE_SEED);

  // Match this register's holes to the relabeled complex's emergent holes by
  // permuted vertex set; a missing image throws, since the gate has to compare
  // like with like.
  const std::vector<std::vector<std::uint64_t>> rederived =
      MultiCobordism::emergentHoles(*rel.spacetime, ctx.degree());
  std::map<std::set<std::uint64_t>, std::vector<std::uint64_t>> found;
  for (const auto &h : rederived) {
    found[std::set<std::uint64_t>(h.begin(), h.end())] = h;
  }
  std::vector<std::vector<std::uint64_t>> holes2;
  holes2.reserve(ctx.holes().size());
  for (const auto &h : ctx.holes()) {
    std::set<std::uint64_t> image;
    for (std::uint64_t v : h) image.insert(rel.vertexMap.at(v));
    auto it = found.find(image);
    if (it == found.end()) {
      throw std::runtime_error(
          "ObservableGates: the relabeled complex's emergent holes are missing "
          "the image of a selected hole (the gate cannot compare like with "
          "like)");
    }
    holes2.push_back(it->second);
  }

  const RegisterContext relabeledCtx(rel.spacetime, holes2,
                                     static_cast<int>(holes2.size()),
                                     ctx.degree(), ctx.target());
  const Record relabeled =
      observable.recordRelabeled(relabeledCtx, rel.vertexMap);
  return Record::reportDelta(base, relabeled);
}

ObservableGates::GateResult ObservableGates::evaluate(
    const RegisterObservable &observable, const RegisterContext &ctx) {
  GateResult result;
  result.gaugeDelta = gaugeDelta(observable, ctx);
  result.relabelDelta = relabelDelta(observable, ctx);
  result.gateTol = observable.gateTol();
  result.gaugeOk = result.gaugeDelta <= result.gateTol;
  result.relabelOk = result.relabelDelta <= result.gateTol;
  return result;
}

bool ObservableGates::selfTest(const RegisterContext &ctx) {
  const LabelLeakProbe labelProbe;
  const GaugeLeakProbe gaugeProbe;
  return relabelDelta(labelProbe, ctx) > 0.0 &&
         gaugeDelta(gaugeProbe, ctx) > 0.0;
}

// ---- Self-test probes ----

Record LabelLeakProbe::record(const RegisterContext &ctx) const {
  double sum = 0.0;
  for (const auto &hole : ctx.holes()) {
    for (std::uint64_t v : hole) sum += static_cast<double>(v);
  }
  Record::Map m;
  m["vertex_id_sum"] = sum;
  return Record(std::move(m));
}

double LabelLeakProbe::computeHeadline(const RegisterContext &ctx) const {
  double sum = 0.0;
  for (const auto &hole : ctx.holes()) {
    for (std::uint64_t v : hole) sum += static_cast<double>(v);
  }
  return sum;
}

Record GaugeLeakProbe::record(const RegisterContext &ctx) const {
  Record::Map m;
  Record::splitComplex(m, "raw_target0", ctx.target().at(0));
  return Record(std::move(m));
}

double GaugeLeakProbe::computeHeadline(const RegisterContext &ctx) const {
  return std::abs(ctx.target().at(0));
}

}  // namespace tessera::observables
