// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/SingletResidual.h"

#include <complex>
#include <vector>

#include "cobordism/MultiCobordism.h"
#include "cobordism/ProtonSynthesis.h"
#include "spacetime/Spacetime.h"

namespace tessera::observables {
using ::tessera::cobordism::MultiCobordism;
using ::tessera::cobordism::ProtonSynthesis;

namespace {

std::vector<std::complex<double>> conjugateSinglet() {
  std::vector<std::complex<double>> conj;
  for (const auto &c : ProtonSynthesis::singlet()) conj.push_back(std::conj(c));
  return conj;
}

}  // namespace

double SingletResidual::computeHeadline(const RegisterContext &ctx) const {
  return MultiCobordism::residualOfTargetStateAgainstHarmonic(
      ctx.spacetime(), ctx.degree(), ProtonSynthesis::singlet());
}

double SingletResidual::conjugateResidual(const RegisterContext &ctx) const {
  return MultiCobordism::residualOfTargetStateAgainstHarmonic(
      ctx.spacetime(), ctx.degree(), conjugateSinglet());
}

Record SingletResidual::record(const RegisterContext &ctx) const {
  const std::vector<std::complex<double>> singlet = ProtonSynthesis::singlet();
  Record::Map m;
  m["singlet_residual"] = MultiCobordism::residualOfTargetStateAgainstHarmonic(
      ctx.spacetime(), ctx.degree(), singlet);
  m["holes_used"] = ctx.holesUsed();
  m["holes_total"] = ctx.holesTotal();
  m["b3"] = ctx.bK();
  Record::List betti;
  for (int b : ctx.betti()) betti.emplace_back(b);
  m["betti"] = std::move(betti);
  m["holes_vs_b3_divergent"] = ctx.holesVsBettiDivergent();
  Record::splitComplex(m, "target", singlet);
  return Record(std::move(m));
}

}  // namespace tessera::observables
