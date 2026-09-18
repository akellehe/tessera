// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/PairLoopFlavor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include "cobordism/EigenstateSynthesis.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::cobordism::EigenstateSynthesis;

namespace {

// A hole's boundary facets as (cochain index, sign) pairs: facet j of the
// sorted hole drops vertex j and carries (-1)^j, the boundary-operator
// convention used throughout. Facets are matched by vertex set, never by an
// imposed order; the physical orientation comes separately from the
// induced-orientation signs.
std::vector<std::pair<std::size_t, int>> facetIndices(
    const std::map<std::vector<std::uint64_t>, std::size_t> &cellIndex,
    const std::vector<std::uint64_t> &hole) {
  std::vector<std::uint64_t> hs = hole;
  std::sort(hs.begin(), hs.end());
  std::vector<std::pair<std::size_t, int>> out;
  out.reserve(hs.size());
  for (std::size_t j = 0; j < hs.size(); ++j) {
    std::vector<std::uint64_t> facet;
    facet.reserve(hs.size() - 1);
    for (std::size_t i = 0; i < hs.size(); ++i) {
      if (i != j) facet.push_back(hs[i]);
    }
    auto it = cellIndex.find(facet);
    if (it == cellIndex.end()) {
      throw std::runtime_error(
          "PairLoopFlavor: a hole facet is not a registered k-cell — the "
          "register degree does not match the hole dimension");
    }
    out.emplace_back(it->second, (j % 2 == 0) ? 1 : -1);
  }
  return out;
}

}  // namespace

std::pair<int, double> PairLoopFlavor::oddOneOut(
    const std::vector<double> &loopQ) {
  std::array<double, 3> separations{};
  for (int k = 0; k < 3; ++k) {
    std::array<double, 2> others{};
    int n = 0;
    for (int i = 0; i < 3; ++i) {
      if (i != k) others[n++] = loopQ[i];
    }
    separations[k] = std::fabs(loopQ[k] - (others[0] + others[1]) / 2.0);
  }
  // Ties break to the first maximal index.
  int odd = 0;
  for (int k = 1; k < 3; ++k) {
    if (separations[k] > separations[odd]) odd = k;
  }
  std::array<double, 2> others{};
  int n = 0;
  for (int i = 0; i < 3; ++i) {
    if (i != odd) others[n++] = loopQ[i];
  }
  const double rho = separations[odd] > 0.0
                         ? std::fabs(others[0] - others[1]) / separations[odd]
                         : std::numeric_limits<double>::infinity();
  return {odd, rho};
}

PairLoopFlavor::JointRead PairLoopFlavor::jointRead(
    const RegisterContext &ctx) const {
  const auto &holes = ctx.holes();
  if (holes.size() != 3 || ctx.target().size() != 3) {
    throw std::invalid_argument(
        "PairLoopFlavor: the pair-loop read is over exactly 3 holes");
  }
  EigenstateSynthesis &es = ctx.synthesis();
  const auto &cellIndex = ctx.cellIndex();
  const std::vector<double> &weights = ctx.hodgeWeights();
  const std::vector<int> &sigma = ctx.epsilonSigns();
  const auto &target = ctx.target();

  std::vector<std::complex<double>> rawTarget(3);
  for (int h = 0; h < 3; ++h) {
    rawTarget[h] = static_cast<double>(sigma[h]) * target[h];
  }
  const std::vector<std::vector<std::uint64_t>> holeLists(holes.begin(),
                                                          holes.end());
  const std::vector<std::complex<double>> psi =
      es.carriedRepresentative(holeLists, rawTarget);

  JointRead read;
  read.sigma = sigma;
  read.rU = es.residualForPeriods(holeLists, rawTarget);

  std::array<std::vector<std::pair<std::size_t, int>>, 3> facets;
  for (int h = 0; h < 3; ++h) facets[h] = facetIndices(cellIndex, holes[h]);

  read.w.resize(3);
  read.q.resize(3);
  for (int h = 0; h < 3; ++h) {
    std::complex<double> w(0.0, 0.0);
    double q = 0.0;
    for (const auto &cs : facets[h]) {
      w += static_cast<double>(cs.second) * psi[cs.first];
      q += weights[cs.first] * std::norm(psi[cs.first]);
    }
    read.w[h] = static_cast<double>(sigma[h]) * w;
    read.q[h] = q;
  }

  for (const auto &pair : PAIR_LOOPS) {
    const int i = pair.first;
    const int j = pair.second;
    read.loopW.push_back(read.w[i] + read.w[j]);
    std::set<std::size_t> support;
    for (const auto &cs : facets[i]) support.insert(cs.first);
    for (const auto &cs : facets[j]) support.insert(cs.first);
    double loopQ = 0.0;
    for (std::size_t c : support) loopQ += weights[c] * std::norm(psi[c]);
    read.loopQ.push_back(loopQ);
    const int k = complementHole(pair);
    read.dualResidual.push_back(std::abs(read.w[i] + read.w[j] + read.w[k]));
  }
  return read;
}

PairLoopFlavor::Verdict PairLoopFlavor::evaluateCriteria(
    const JointRead &read) const {
  const auto [odd, rho] = oddOneOut(read.loopQ);
  Verdict verdict;
  verdict.oddLoop = PAIR_LOOPS[odd];
  verdict.dualHole = complementHole(PAIR_LOOPS[odd]);
  verdict.rho = rho;
  verdict.multiplicity21 = rho < RHO_MAX;
  if (diquarkPair_) {
    std::pair<int, int> sorted = *diquarkPair_;
    if (sorted.first > sorted.second) std::swap(sorted.first, sorted.second);
    verdict.oddIsDiquarkLoop = (sorted == PAIR_LOOPS[odd]);
  }
  return verdict;
}

double PairLoopFlavor::computeHeadline(const RegisterContext &ctx) const {
  return evaluateCriteria(jointRead(ctx)).rho;
}

Record PairLoopFlavor::record(const RegisterContext &ctx) const {
  const JointRead read = jointRead(ctx);
  const Verdict verdict = evaluateCriteria(read);

  // The oriented periods are U(1)-covariant (w -> e^{i theta} w) and flip with
  // the endSignCovector propagation root under relabelling. Dividing out the
  // unit phase of w[0] fixes that convention, so every reported value is
  // invariant.
  const std::complex<double> phase0 =
      std::abs(read.w[0]) > 0.0 ? read.w[0] / std::abs(read.w[0])
                                : std::complex<double>(1.0, 0.0);
  std::vector<std::complex<double>> wFixed;
  for (const auto &wi : read.w) wFixed.push_back(wi / phase0);
  std::vector<std::complex<double>> loopWFixed;
  for (const auto &wi : read.loopW) loopWFixed.push_back(wi / phase0);

  Record::List q;
  for (double x : read.q) q.emplace_back(x);
  Record::List loopQ;
  for (double x : read.loopQ) loopQ.emplace_back(x);
  Record::List dualResidual;
  for (double x : read.dualResidual) dualResidual.emplace_back(x);
  Record::List pairLoops;
  for (const auto &pl : PAIR_LOOPS) {
    pairLoops.emplace_back(Record::List{pl.first, pl.second});
  }

  Record::Map m;
  m["r_u"] = read.rU;
  m["q"] = std::move(q);
  m["loop_q"] = std::move(loopQ);
  m["dual_residual"] = std::move(dualResidual);
  m["pair_loops"] = std::move(pairLoops);
  m["odd_loop"] = Record::List{verdict.oddLoop.first, verdict.oddLoop.second};
  m["dual_hole"] = verdict.dualHole;
  m["rho"] = verdict.rho;
  m["rho_max"] = RHO_MAX;
  m["multiplicity_2_1"] = verdict.multiplicity21;
  if (verdict.oddIsDiquarkLoop) {
    m["odd_is_diquark_loop"] = *verdict.oddIsDiquarkLoop;
    m["odd_is_diquark_loop_status"] = std::string(kOddDiquarkEvaluated);
  } else {
    m["odd_is_diquark_loop"] = Record();  // null
    m["odd_is_diquark_loop_status"] = std::string(kOddDiquarkNotEvaluable);
  }
  Record::splitComplex(m, "w", wFixed);
  Record::splitComplex(m, "loop_w", loopWFixed);
  return Record(std::move(m));
}

}  // namespace tessera::observables
