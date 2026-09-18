// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/EmergentMass.h"

#include <optional>
#include <string>

namespace tessera::observables {

namespace {

// Shell-bin key: a shelled bin is keyed by its integer distance, the bin with
// no reachable hole by "unshelled".
std::string shellKey(const std::optional<int> &shell) {
  return shell ? std::to_string(*shell) : std::string("unshelled");
}

Record::Map censusRecord(const InteriorHinges::Census &c) {
  // The `boundary_tets` list carries vertex ids, so it is label-bound and left
  // out of the record; the counts stay.
  Record::Map m;
  m["n_tops"] = c.nTops;
  m["n_tets"] = c.nTets;
  m["n_hinges_total"] = c.nHingesTotal;
  m["n_hinges_interior"] = c.nHingesInterior;
  m["n_hinges_boundary"] = c.nHingesBoundary;
  m["n_boundary_tets"] = c.nBoundaryTets;
  m["n_hole_vertices"] = c.nHoleVertices;
  return m;
}

Record::Map massRecord(const InteriorHinges::Masses &mass) {
  Record::Map shellMeans;
  for (const auto &kv : mass.shellMeans) shellMeans[shellKey(kv.first)] = kv.second;
  Record::Map m;
  m["m_shell"] = mass.mShell;
  m["m_sum"] = mass.mSum;
  m["m_action"] = mass.mAction;
  m["shell_means"] = std::move(shellMeans);
  m["max_abs_im"] = mass.maxAbsIm;
  m["n_im_nonzero"] = mass.nImNonzero;
  return m;
}

Record::Map localizationRecord(const InteriorHinges::Localization &loc) {
  Record::Map profile;
  for (const auto &kv : loc.shellProfile) {
    Record::Map p;
    p["n"] = kv.second.n;
    p["mean_re"] = kv.second.meanRe;
    p["weight_share"] = kv.second.weightShare;
    profile[std::to_string(kv.first)] = std::move(p);
  }
  Record::Map m;
  m["PR"] = loc.pr;
  m["concentration"] = loc.concentration;
  m["mean_re"] = loc.meanRe;
  m["std_re"] = loc.stdRe;
  m["std_over_mean"] = loc.stdOverMean;
  m["shell_profile"] = std::move(profile);
  m["rms_shell_radius"] = loc.rmsShellRadius;
  m["frac_within_shell1"] = loc.fracWithinShell1;
  return m;
}

Record::Map rmRecord(const InteriorHinges::RmTable &rm) {
  Record::Map combos;
  for (const auto &kv : rm.combos) combos[kv.first] = kv.second;
  Record::Map m;
  m["spread_min"] = rm.spreadMin;
  m["spread_max"] = rm.spreadMax;
  m["combos"] = std::move(combos);
  m["physical"] = rm.physical;
  return m;
}

}  // namespace

InteriorHinges::Masses EmergentMass::masses(const RegisterContext &ctx) const {
  return ctx.interiorHinges()->masses();
}

InteriorHinges::Localization EmergentMass::localization(
    const RegisterContext &ctx) const {
  return ctx.interiorHinges()->localization();
}

double EmergentMass::computeHeadline(const RegisterContext &ctx) const {
  return ctx.interiorHinges()->masses().mShell;
}

Record EmergentMass::record(const RegisterContext &ctx) const {
  const auto &hinges = *ctx.interiorHinges();
  const InteriorHinges::Masses mass = hinges.masses();
  const InteriorHinges::Radii rad = hinges.radii();
  const InteriorHinges::Localization loc = hinges.localization();
  const InteriorHinges::RmTable rm = hinges.rmTable(mass, rad);

  Record::Map m;
  m["census"] = censusRecord(hinges.census());
  m["mass"] = massRecord(mass);
  m["localization"] = localizationRecord(loc);
  m["rm"] = rmRecord(rm);
  m["n_holes"] = ctx.holesUsed();
  return Record(std::move(m));
}

}  // namespace tessera::observables
