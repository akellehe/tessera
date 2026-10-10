// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// MultiCobordism (include/cobordism/MultiCobordism.h): the objective: rU,
// the near-kernel residual, the Hodge-entropy terms, the objective
// specification and its context, pinned regions and register constraints.
// One of the translation units that define the class's members by
// responsibility (https://github.com/akellehe/tessera/issues/1481).

#include "MultiCobordismInternal.h"

namespace tessera::cobordism {

namespace {

/// The operator of a metric source, totalized. A configuration extreme enough
/// that the operator cannot be assembled at all -- an overflowed geometry makes
/// the Whitney pencil's dressed metric singular, and its factorization refuses
/// by name -- is as unusable as one whose entries leave double range, and the
/// residual terms below evaluate both to their worst case rather than raising
/// out of a line-search trial.
std::optional<std::vector<complexd>> totalizedLaplacian(
    const cobordism::HodgeLaplacian &laplacian, int degree) {
  try {
    return laplacian.laplacian(degree, /*metric=*/true);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

}  // namespace

double MultiCobordism::rU(const std::shared_ptr<Spacetime> &spacetime) const {
  // The cobordism residual. Inputs are localized boundary sub-complexes (built near
  // a seed, held representable by these terms, not pinned) — each read off its own
  // region and weighted by inputResidualWeight_ so they are not out-competed by the
  // whole/output term.
  //
  // one claim set spans the whole evaluation: every register here is scored by the
  // same min-over-relabelings, so without it they all pick the same argmin matching
  // and the sum is smallest when the registers carry identical weights. The set
  // records each register's winning matching and withholds it from the ones after.
  std::set<std::vector<int>> claimedMatchings;
  double totalResidual = 0.0;
  // Explicit constraints are already framed: their hole and target ordering is
  // fixed by the caller, so they bypass emergent-hole relabeling entirely.
  for (const auto &constraint : registerConstraints_)
    totalResidual += EigenstateSynthesis(spacetime, constraint.degree, metricSource_)
                         .residualForPeriods(constraint.holes,
                                             constraint.target);
  for (const auto &inputBlock : inputBlocks_)
    totalResidual += inputResidualWeight_ *
                     residualForBoundaryBlockWithDistinctMatchings(inputBlock, spacetime,
                                                   claimedMatchings);
  if (outputTargets_.size() == 1) {
    // A single output is the whole cobordism's output boundary: as in the Python
    // reference it is "the harmonic of the entire structure", never a pinned
    // region. Read it off the whole complex so the bulk loop drives the whole to
    // carry it (the output emerges; it is not frozen by seedOutputs).
    // In the singularValueRatio mode this period read is part of the
    // whole-complex term the ratio below replaces, so it is skipped — the
    // output target then names an expectation for the after-the-fact readout
    // (and sizes expectedRegisterCount), never a scored prescription.
    if (!singularValueRatio_)
      for (int registerDegree : registerDegrees_)
        totalResidual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
            spacetime, registerDegree, outputTargets_.front(), claimedMatchings);
  } else {
    // Multiple outputs (e.g. a 2->2 recombination → diquark ⊔ antidiquark) live in
    // distinct regions: read each off its own constructed block. Empty outputTargets
    // is the supported nothing-pinned-downstream shape: no output term at
    // all — rU is the weighted input residuals alone, and the whole's final state
    // emerges (read after the fact, e.g. protonIngredients' singlet diagnostic).
    for (const auto &outputBlock : outputBlocks_)
      totalResidual +=
          residualForBoundaryBlockWithDistinctMatchings(outputBlock, spacetime,
                                                        claimedMatchings);
    if (inputBlocks_.empty() && outputBlocks_.empty())  // bare objective, nothing built yet
      for (int registerDegree : registerDegrees_)
        for (const auto &outputTarget : outputTargets_)
          totalResidual += residualOfTargetStateAgainstHarmonicWithDistinctMatching(
              spacetime, registerDegree, outputTarget, claimedMatchings);
  }
  if (singularValueRatio_) {
    // The whole-complex term in the ratio mode: one scale-invariant
    // spectral-shape term per degree covers both regimes the two terms below
    // split between — it reads the full spectrum, so it presses from the bare
    // seed (no topological threshold) and keeps pressing after the holes open
    // (the lower half keeps collapsing past the exact kernel).
    for (int registerDegree : registerDegrees_)
      totalResidual += singularValueHalfSumRatio(spacetime, registerDegree, metricSource_);
    return totalResidual;
  }
  // The pre-topological register signal: the period residuals above are
  // step functions in the topology — exactly flat until a register exists — so
  // they carry no register-seeking gradient at a seed. The near-kernel residual
  // is the same functional continued below the topological threshold, and it
  // saturates at 0 the moment b_k reaches the expected count (see the header).
  // Fiber-form targets declare no register count and no hole: the
  // whole-complex fiber residual replaces the near-kernel hole-forcing term.
  if (useFiberResiduals_) {
    if (wholeFiberTarget_) totalResidual += fiberResidualOn(spacetime, *wholeFiberTarget_);
    // The cases, when set, are the two-body term: one bulk scored
    // against several input pairs at once, so every move is priced against all
    // of them. With none set this is the single target, unchanged.
    // On whatever complex is being scored, never only the live one: stage 1
    // prices each candidate on a complex rebuilt from a snapshot, so a
    // live-only test would rank moves by the first case while stage 2
    // optimized the sum.
    if (!twoBodyCases_.empty())
      totalResidual += twoBodyResidualOverCasesOn(spacetime);
    else if (twoBodyTarget_) totalResidual += twoBodyResidualOn(spacetime, *twoBodyTarget_);
    return totalResidual;
  }
  const std::size_t expectedRegisters = expectedRegisterCount();
  if (expectedRegisters > 0)
    for (int registerDegree : registerDegrees_)
      totalResidual += nearKernelResidual(spacetime, registerDegree,
                                          expectedRegisters, metricSource_);
  return totalResidual;
}

std::size_t MultiCobordism::expectedRegisterCount() const {
  std::size_t expected = 0;
  for (const auto &target : inputTargets_)
    expected = std::max(expected, target.size());
  for (const auto &target : outputTargets_)
    expected = std::max(expected, target.size());
  return expected;
}

double MultiCobordism::nearKernelResidual(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    std::size_t expectedRegisterCount, HodgeLaplacian::MetricSource metricSource) {
  if (expectedRegisterCount == 0) return 0.0;
  cobordism::HodgeLaplacian laplacian(spacetime, HodgeLaplacian::defaultWeightConvention(), metricSource);
  // Metric operator, deliberately: the term must feel the continuously-valued
  // edge lengths, so stage 2 can tune the causal structure toward null
  // directions and open near-kernels with no holes at all — that channel is
  // the point, not a loophole (measured: a build driven this way ends with
  // most edges timelike and spectral near-kernels but zero topological holes).
  // Whether such causal near-kernels can carry a register is the next level of
  // exploration; the semantics for reading them out are not implemented here.
  // Stage-1 surgery remains the other route to the same descent: a genuine
  // hole zeroes the same singular values exactly.
  const std::optional<std::vector<std::complex<double>>> assembled =
      totalizedLaplacian(laplacian, registerDegree);
  if (!assembled) return std::numeric_limits<double>::infinity();
  const std::vector<std::complex<double>> &flat = *assembled;
  const std::size_t n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  // No k-cells at all: every expected register is missing — the worst case on
  // the normalized scale, 1 per missing dimension.
  if (n == 0) return static_cast<double>(expectedRegisterCount);
  Eigen::MatrixXcd L(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      L(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * n + j];
  // Total over every configuration: a non-finite operator evaluates to
  // +inf (see bestRelabelingOfTarget) rather than reaching BDCSVD.
  if (!L.allFinite()) return std::numeric_limits<double>::infinity();
  // Singular values of the non-normal signed operator: the smooth surrogate for
  // the eigenvalue magnitudes (they share the kernel exactly).
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(L);
  const Eigen::VectorXd sigma = svd.singularValues();  // descending
  double total = 0.0;
  for (Eigen::Index i = 0; i < sigma.size(); ++i) total += sigma[i] * sigma[i];
  // L identically zero: every mode is kernel — nothing left to open.
  if (total <= 0.0) return 0.0;
  const std::size_t m = std::min(expectedRegisterCount, n);
  double smallest = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double s = sigma[static_cast<Eigen::Index>(n - 1 - i)];
    smallest += s * s;
  }
  // n * (smallest m) / (all): scale-invariant (L is degree −1 in l^2, so a raw
  // spectral sum is a conformal-inflation descent channel; the ratio is degree
  // 0). Missing dimensions count 1 each — the generic-mode value.
  return static_cast<double>(n) * smallest / total +
         static_cast<double>(expectedRegisterCount - m);
}

std::vector<std::complex<double>> MultiCobordism::nearKernelResidualGradient(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    std::size_t expectedRegisterCount, HodgeLaplacian::MetricSource metricSource) {
  // d/dl^2 of  r = n * (sum of the m smallest sigma^2) / (sum of all sigma^2).
  //
  // The sigma^2 are the eigenvalues of the Hermitian H = L^dagger L, so first-order
  // perturbation gives d(sigma_i^2) = w_i^dagger (dL^dagger L + L^dagger dL) w_i for the
  // normalized eigenvector w_i — no singular-vector pair needed, and the
  // expression stays valid for the non-normal signed operator. The denominator
  // is tr(H), whose derivative is the trace of the same perturbation. Quotient
  // rule over the two, times n.
  //
  // complex throughout: laplacianGradient is already complex, and the
  // return follows the same convention as the period-gap family,
  //   g = dr/d(Re l^2) - i dr/d(Im l^2),
  // so Re(g) and -Im(g) are the two directional derivatives.
  using Eigen::Index;
  using Eigen::MatrixXcd;
  const ChainComplex chain = ChainComplex::fromSpacetime(*spacetime);
  const std::vector<std::vector<std::uint64_t>> oneCells = chain.kSimplexVertices(1);
  std::vector<complexd> gradient(oneCells.size(), complexd(0.0, 0.0));
  if (expectedRegisterCount == 0) return gradient;   // value is the constant 0

  cobordism::HodgeLaplacian laplacian(spacetime, HodgeLaplacian::defaultWeightConvention(), metricSource);
  const std::vector<complexd> flat =
      laplacian.laplacian(registerDegree, /*metric=*/true);
  const std::size_t n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  if (n == 0) return gradient;      // value is the constant expectedRegisterCount
  const Index N = static_cast<Index>(n);
  MatrixXcd L(N, N);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      L(static_cast<Index>(i), static_cast<Index>(j)) = flat[i * n + j];
  if (!L.allFinite()) return gradient;   // the value is +inf here; no slope

  // H = L^dagger L is Hermitian positive semi-definite; its eigenvalues are the
  // sigma^2 the value reads, ascending here, and its eigenvectors give the exact
  // first-order response of each one.
  const MatrixXcd H = L.adjoint() * L;
  Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(H);
  const Eigen::VectorXd eigenvalues = solver.eigenvalues();      // ascending
  const MatrixXcd eigenvectors = solver.eigenvectors();
  double total = 0.0;
  for (Index i = 0; i < eigenvalues.size(); ++i) total += eigenvalues[i];
  if (total <= 0.0) return gradient;    // L identically zero: value is 0, flat
  const std::size_t m = std::min(expectedRegisterCount, n);
  double smallest = 0.0;
  for (std::size_t i = 0; i < m; ++i)
    smallest += eigenvalues[static_cast<Index>(i)];              // m smallest

  for (std::size_t edgeIndex = 0; edgeIndex < oneCells.size(); ++edgeIndex) {
    const std::vector<complexd> derivativeFlat = laplacian.laplacianGradient(
        registerDegree, oneCells[edgeIndex][0], oneCells[edgeIndex][1]);
    if (derivativeFlat.empty()) continue;
    MatrixXcd dL(N, N);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        dL(static_cast<Index>(i), static_cast<Index>(j)) =
            derivativeFlat[i * n + j];
    // L is holomorphic in l^2 (the weights are polynomial in it), so the
    // complex derivative of H = L^dagger L is 2 L^dagger dL — not
    // dL^dagger L + L^dagger dL, the Hermitian combination, and therefore the
    // real-direction derivative alone. The Hermitian form makes w^dagger dH w
    // real by construction and the gradient's imaginary part identically zero.
    // Re(2 L^dagger dL) reproduces the Hermitian value exactly, so the real
    // direction is unchanged.
    const MatrixXcd dH = 2.0 * (L.adjoint() * dL);
    // d(sum of the m smallest) and d(trace) from the same perturbation.
    complexd dSmallest(0.0, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      const auto w = eigenvectors.col(static_cast<Index>(i));
      dSmallest += w.dot(dH * w);      // w^dagger dH w, as a scalar
    }
    const complexd dTotal = dH.trace();
    gradient[edgeIndex] = static_cast<double>(n) *
                          (dSmallest * total - smallest * dTotal) /
                          (total * total);
  }
  return gradient;
}

double MultiCobordism::singularValueHalfSumRatio(
    const std::shared_ptr<Spacetime> &spacetime, int registerDegree,
    HodgeLaplacian::MetricSource metricSource) {
  cobordism::HodgeLaplacian laplacian(spacetime, HodgeLaplacian::defaultWeightConvention(), metricSource);
  // The same operator nearKernelResidual reads (metric, signed, generally
  // non-normal — see its comment); the two terms are alternatives for the one
  // whole-complex slot in rU, so they must see the same spectrum.
  const std::optional<std::vector<std::complex<double>>> assembled =
      totalizedLaplacian(laplacian, registerDegree);
  if (!assembled) return std::numeric_limits<double>::infinity();
  const std::vector<std::complex<double>> &flat = *assembled;
  const std::size_t n = static_cast<std::size_t>(
      std::llround(std::sqrt(static_cast<double>(flat.size()))));
  // No k-cells: the worst case on the [0, 1] scale. Returning the perfect 0
  // here would reward deleting every k-cell over collapsing the spectrum.
  if (n == 0) return 1.0;
  const std::size_t h = n / 2;
  if (h == 0) return 0.0;  // a single mode: no pair of halves to compare
  Eigen::MatrixXcd L(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      L(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
          flat[i * n + j];
  // Total over every configuration: +inf, as nearKernelResidual.
  if (!L.allFinite()) return std::numeric_limits<double>::infinity();
  Eigen::BDCSVD<Eigen::MatrixXcd> svd(L);
  const Eigen::VectorXd sigma = svd.singularValues();  // descending
  double upperHalfSum = 0.0;
  double lowerHalfSum = 0.0;
  for (std::size_t i = 0; i < h; ++i) {
    upperHalfSum += sigma[static_cast<Eigen::Index>(i)];
    lowerHalfSum += sigma[static_cast<Eigen::Index>(n - h + i)];
  }
  // L identically zero: every mode is kernel — nothing left to collapse.
  if (upperHalfSum <= 0.0) return 0.0;
  // Each lower-half value is bounded by its upper-half counterpart (descending
  // order), so the ratio lives in [0, 1]; and L is homogeneous of degree −1 in
  // l^2, so a uniform rescale scales every sigma alike and cancels — degree 0,
  // the same closed conformal-inflation channel as nearKernelResidual.
  return lowerHalfSum / upperHalfSum;
}

double MultiCobordism::hodgeEntropy() const {
  // Over the hodge degrees, which are what the entropy is taken at. Reported
  // unweighted: the per-degree weights balance the stationarity residuals
  // against each other in the objective, and applying them to entropy values
  // would report a number that is not any degree's entropy.
  double entropy = 0.0;
  for (int degree : hodgeDegrees_)
    entropy += HodgeLaplacian(spacetime_, HodgeLaplacian::defaultWeightConvention(), metricSource_).spectralEntropy(
        degree, hodgeEntropyPhaseMode_);
  return entropy;
}

double MultiCobordism::hodgeEntropyStationarity() const {
  // The entropy half of the objective, so it must read the same degrees and
  // the same weights the objective does. Reading the register degrees here
  // while the term reads the Hodge degrees would let an observation disagree
  // silently with the quantity being descended.
  //
  // Accumulated in the term's order — weighted norms summed, the entropy
  // weight applied by the caller — so `hodgeEntropyWeight() * this` reproduces
  // `ObjectiveTerms::hodgeStationarity` exactly.
  double residual = 0.0;
  for (std::size_t index = 0; index < hodgeDegrees_.size(); ++index) {
    const double weight = index < hodgeDegreeWeights_.size()
                              ? hodgeDegreeWeights_[index]
                              : 1.0;
    residual += weight * HodgeLaplacian(spacetime_, HodgeLaplacian::defaultWeightConvention(), metricSource_).spectralEntropyGradientNorm(
                             hodgeDegrees_[index], hodgeEntropyPhaseMode_);
  }
  return residual;
}

void MultiCobordism::requireObjectiveAcceptable(
    const std::shared_ptr<CobordismObjective> &objective) const {
  // The objective's own declared domain, enforced here so the restriction
  // travels with the objective that declares it rather than living in the
  // engine as a special case. It is a declaration, not a capability limit.
  const int minimumDegree = objective->minimumRegisterDegree();
  for (int degree : registerDegrees_)
    if (degree < minimumDegree)
      throw std::invalid_argument(
          "MultiCobordism: objective '" + objective->name() +
          "' is declared over degrees >= " + std::to_string(minimumDegree) +
          "; got degree " + std::to_string(degree));
  // A scope can only carry a handle minted by `regionHandle`, which refuses an
  // undeclared name, so a mis-spelling cannot reach this point. Re-check
  // anyway: regions can be cleared after a handle was minted, and an objective
  // pointing at a region that no longer exists must fail rather than score
  // nothing.
  const auto scope = objective->scope();
  if (scope.isWholeCobordism()) return;
  for (const auto &region : pinnedRegions_)
    if (region.name == scope.region.name()) return;
  throw std::invalid_argument(
      "MultiCobordism: the injected objective is scoped to pinned region "
      "'" + scope.region.name() + "', which is not declared");
}

void MultiCobordism::setObjective(
    std::shared_ptr<CobordismObjective> objective) {
  if (!objective)
    throw std::invalid_argument(
        "MultiCobordism: the injected objective must not be null");
  requireObjectiveAcceptable(objective);
  objectiveSpec_ = std::move(objective);
}

void MultiCobordism::setPinnedObjective(
    std::shared_ptr<CobordismObjective> objective) {
  if (!objective)
    throw std::invalid_argument(
        "MultiCobordism: the pinned-region objective must not be null");
  requireObjectiveAcceptable(objective);
  pinnedObjectiveSpec_ = std::move(objective);
}

RegionHandle MultiCobordism::regionHandle(const std::string &name) const {
  for (const auto &region : pinnedRegions_)
    if (region.name == name) return RegionHandle(name);
  throw std::invalid_argument(
      "MultiCobordism: no pinned region named '" + name + "' is declared");
}

std::string MultiCobordism::objectiveName() const {
  return objectiveSpec_->name();
}

bool MultiCobordism::compositeSupportsLocalizedDelta() const {
  // A localized delta differences the objective over the cells a move touches.
  // That shortcut is valid only while the scalar being reported is the one
  // being differenced, and with a pinned objective in force the reported scalar
  // is the sum of two functionals over two different scopes. Differencing the
  // bulk alone would optimize a surrogate that is not the objective, so any
  // pinned objective drops
  // the whole node back to global re-evaluation. Global is always correct,
  // merely more expensive.
  if (pinnedObjectiveSpec_) return false;
  return objectiveSpec_->supportsLocalizedDelta();
}

bool MultiCobordism::objectiveIsTargetConditioned() const {
  // The disjunction, not the bulk objective's answer. A search policy asks this
  // to find out whether the run it is driving is unforced, and a run whose
  // pinned region is held to a declared state is target-conditioned however
  // geometric the bulk objective is. Reporting the bulk alone would let a
  // policy believe it was unforced while a target steered part of the complex.
  if (pinnedObjectiveSpec_ && pinnedObjectiveSpec_->isTargetConditioned())
    return true;
  return objectiveSpec_->isTargetConditioned();
}

ObjectiveContext MultiCobordism::objectiveContextFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  return objectiveContextFor(spacetime, objectiveSpec_);
}

std::vector<std::size_t> MultiCobordism::scopedEdgeIndices(
    const std::shared_ptr<Spacetime> &spacetime,
    const std::set<std::uint64_t> &region,
    bool includesStraddlingEdges) const {
  std::vector<std::size_t> scored;
  if (!spacetime || region.empty()) return scored;
  const auto edges = spacetime->getEdgeList()->toVector();
  for (std::size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
    const auto key = edges[edgeIndex]->getKey();
    const int endpointsInside = static_cast<int>(region.count(key.first)) +
                                static_cast<int>(region.count(key.second));
    if (endpointsInside == 0) continue;          // wholly bulk
    if (endpointsInside == 2) {                  // interior to the region
      scored.push_back(edgeIndex);
      continue;
    }
    // Exactly one endpoint inside: a straddling edge, in only where the
    // objective declared it so.
    if (includesStraddlingEdges) scored.push_back(edgeIndex);
  }
  return scored;
}

ObjectiveContext MultiCobordism::objectiveContextFor(
    const std::shared_ptr<Spacetime> &spacetime,
    const std::shared_ptr<CobordismObjective> &objective) const {
  // The firewall. Everything an objective can see is assembled here and nowhere
  // else, and every field of it is plain data: the complex, the region and its
  // declared targets, the configured weights, and two precomputed geometric
  // scalars. No `std::function`, because a bound callable would capture `this`
  // and hand the objective a route back into the node. An objective therefore
  // cannot consult a component, fiber, transport, amplitude, colour, particle,
  // charge, flavour, exchange or spin certificate.
  ObjectiveContext context;
  context.spacetime = spacetime;
  // The objective's declared scope, resolved here. Declaring nothing means the
  // whole cobordism, which leaves `region` and `scoredEdges` empty and every
  // sum running over every coordinate exactly as it did before scopes existed.
  const ObjectiveScope scope = objective ? objective->scope() : ObjectiveScope{};
  context.region = {};
  if (!scope.isWholeCobordism()) {
    for (const auto &region : pinnedRegions_)
      if (region.name == scope.region.name()) {
        context.region = region.vertices;
        break;
      }
    // Present, even when empty: a region with no scored coordinate scores
    // nothing, which is not the whole cobordism.
    context.scoredEdges = scopedEdgeIndices(spacetime, context.region,
                                            scope.includesStraddlingEdges);
  }
  for (const auto &block : inputBlocks_) context.regionTargets.push_back(block.target);
  for (const auto &block : outputBlocks_) context.regionTargets.push_back(block.target);
  context.registerDegrees = registerDegrees_;
  // Declared independently and never inherited from the register degrees: the
  // degrees a register is constructed at and the degrees whose entropy should
  // be stationary are different questions, and defaulting one to the other
  // would reinstate that coupling in the implementation.
  context.hodgeDegrees = hodgeDegrees_;
  context.hodgeDegreeWeights = hodgeDegreeWeights_;
  context.reggeWeight = reggeWeight_;
  context.hodgeEntropyWeight = hodgeEntropyWeight_;
  context.connectionEntropyWeight = connectionEntropyWeight_;
  context.gamma = gamma_;
  context.carriedStateEnergyWeight = carriedStateEnergyWeight_;
  context.momentStiffnessWeight = momentStiffnessWeight_;
  context.momentStiffnessDegrees = momentStiffnessDegrees_;
  context.momentStiffnessCoefficients = momentStiffnessCoefficients_;
  context.momentStiffnessReference = momentStiffnessReference_;
  context.einsteinHilbert = einsteinHilbert_;
  context.fiberResiduals = useFiberResiduals_;
  context.hodgeEntropyPhaseMode = hodgeEntropyPhaseMode_;
  // Computed only where the objective declares it reads them, so a purely
  // geometric objective never pays for a target-conditioned quantity. Left NaN
  // rather than zero where not computed: unmeasured is not "measured zero".
  if (spacetime && objective && objective->needsRegisterResidual())
    context.registerResidual = rU(spacetime);
  if (spacetime && carriedStateEnergyWeight_ != 0.0)
    context.carriedStateEnergy = carriedStateEnergy(spacetime);
  return context;
}

void MultiCobordism::setMomentStiffness(double weight, const std::vector<int> &degrees,
                                        const std::vector<double> &coefficients) {
  if (!std::isfinite(weight) || weight < 0.0)
    throw std::invalid_argument("MultiCobordism: moment stiffness weight must be finite and non-negative");
  for (const double coefficient : coefficients)
    if (!std::isfinite(coefficient) || coefficient < 0.0)
      throw std::invalid_argument("MultiCobordism: moment stiffness coefficients must be finite and non-negative");
  momentStiffnessReference_.clear();
  if (weight == 0.0) {
    momentStiffnessWeight_ = 0.0;
    momentStiffnessDegrees_.clear();
    momentStiffnessCoefficients_.clear();
    return;
  }
  if (coefficients.empty() || degrees.empty())
    throw std::invalid_argument("MultiCobordism: a moment stiffness needs at least one degree and one order");
  if (!spacetime_)
    throw std::logic_error("MultiCobordism: the moment stiffness is declared about an existing geometry");
  // The carrier is the geometry as it is now; its local moments are the reference.
  const HodgeLaplacian hodge(spacetime_);
  for (const int degree : degrees)
    momentStiffnessReference_.push_back(hodge.localSpectralMoments(degree, static_cast<int>(coefficients.size())));
  momentStiffnessWeight_ = weight;
  momentStiffnessDegrees_ = degrees;
  momentStiffnessCoefficients_ = coefficients;
}

void MultiCobordism::setHodgeEntropyWeight(double weight) {
  if (!std::isfinite(weight) || weight < 0.0)
    throw std::invalid_argument(
        "MultiCobordism: Hodge entropy weight must be finite and non-negative");
  hodgeEntropyWeight_ = weight;
}

void MultiCobordism::setConnectionEntropyWeight(double weight) {
  if (!std::isfinite(weight) || weight < 0.0)
    throw std::invalid_argument(
        "MultiCobordism: connection entropy weight must be finite and "
        "non-negative");
  connectionEntropyWeight_ = weight;
}

void MultiCobordism::setReggeWeight(double weight) {
  if (!std::isfinite(weight) || weight < 0.0)
    throw std::invalid_argument(
        "MultiCobordism: Regge weight must be finite and non-negative");
  reggeWeight_ = weight;
}

void MultiCobordism::setHodgeDegrees(std::vector<int> degrees,
                                     std::vector<double> weights) {
  if (degrees.empty())
    throw std::invalid_argument(
        "MultiCobordism: the Hodge degree list must name at least one degree; "
        "an empty list would silently score no entropy at all");
  for (int degree : degrees)
    if (degree < 0)
      throw std::invalid_argument(
          "MultiCobordism: Hodge degree " + std::to_string(degree) +
          " is negative; a degree indexes a Laplacian L_k with k >= 0");
  // A repeat would double-count that degree's share while reading as a list of
  // distinct degrees, so it is refused rather than silently summed twice.
  for (std::size_t outer = 0; outer < degrees.size(); ++outer)
    for (std::size_t inner = outer + 1; inner < degrees.size(); ++inner)
      if (degrees[outer] == degrees[inner])
        throw std::invalid_argument(
            "MultiCobordism: Hodge degree " + std::to_string(degrees[outer]) +
            " is listed more than once");
  if (!weights.empty()) {
    if (weights.size() != degrees.size())
      throw std::invalid_argument(
          "MultiCobordism: " + std::to_string(weights.size()) +
          " Hodge degree weights were given for " +
          std::to_string(degrees.size()) +
          " degrees; supply one weight per degree, or none for uniform");
    for (double weight : weights)
      if (!std::isfinite(weight) || weight < 0.0)
        throw std::invalid_argument(
            "MultiCobordism: a Hodge degree weight must be finite and "
            "non-negative");
  }
  hodgeDegrees_ = std::move(degrees);
  hodgeDegreeWeights_ = std::move(weights);
}

std::vector<HodgeDegreeContribution>
MultiCobordism::hodgeDegreeContributionsFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  // Read from the objective that owns the term, so a reported share is the one
  // that was descended rather than a second computation that could disagree.
  if (!objectiveSpec_) return {};
  return objectiveSpec_->hodgeDegreeContributions(
      objectiveContextFor(spacetime, objectiveSpec_));
}

std::vector<HodgeDegreeContribution> MultiCobordism::hodgeDegreeContributions()
    const {
  return hodgeDegreeContributionsFor(spacetime_);
}

std::vector<std::string> MultiCobordism::objectiveTermNames() {
  // The declaration order of `ObjectiveTerms`. Enumerated as data so the
  // no-feedback firewall is checkable rather than asserted in a comment: a
  // test reads this list and confirms no particle, fiber, transport, or
  // amplitude quantity is on it. Every objective records into these same
  // slots, so a record stays comparable across objectives.
  return CobordismObjective::declaredTermNames();
}

double MultiCobordism::objectiveOf(const ObjectiveTerms &terms) {
  // Static: no `this`, so the scalar the optimizer descends provably depends
  // on nothing but the declared terms.
  return CobordismObjective::total(terms);
}

std::vector<MultiCobordism::ObjectiveContribution>
MultiCobordism::objectiveContributionsFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  // One contribution per objective, in evaluation order, so a reader can tell
  // whether descent came from the bulk or from the pinned region. Summing them
  // reproduces `objectiveTermsFor` exactly.
  std::vector<ObjectiveContribution> contributions;
  const auto record = [&](const std::shared_ptr<CobordismObjective> &objective) {
    if (!objective) return;
    contributions.push_back(
        {objective->name(), objective->scope().region.name(),
         objective->terms(objectiveContextFor(spacetime, objective))});
  };
  record(objectiveSpec_);
  record(pinnedObjectiveSpec_);
  return contributions;
}

std::vector<MultiCobordism::ObjectiveContribution>
MultiCobordism::objectiveContributions() const {
  return objectiveContributionsFor(spacetime_);
}

MultiCobordism::ObjectiveTerms MultiCobordism::objectiveTermsFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  // The engine does not know which functional it is scoring: it assembles the
  // firewalled context and the injected objective decomposes itself.
  ObjectiveTerms terms = objectiveSpec_->terms(objectiveContextFor(spacetime));
  // A pinned-region objective adds its terms on top. The bulk objective keeps
  // scoring the entire cobordism including the pinned interior, so a
  // boundary-interior edge contributes to both; the scoring is additive rather
  // than double-counted, since the bulk sees one coherent cobordism and this is
  // an additional hold on part of it.
  if (pinnedObjectiveSpec_) {
    const auto pinned = pinnedObjectiveSpec_->terms(
        objectiveContextFor(spacetime, pinnedObjectiveSpec_));
    terms.reggeStationarity += pinned.reggeStationarity;
    terms.hodgeStationarity += pinned.hodgeStationarity;
    terms.registerResidual += pinned.registerResidual;
    terms.actionMagnitude += pinned.actionMagnitude;
    terms.carriedStateEnergy += pinned.carriedStateEnergy;
  }
  return terms;
}

MultiCobordism::ObjectiveTerms MultiCobordism::objectiveTerms() const {
  return objectiveTermsFor(spacetime_);
}

double MultiCobordism::objectiveFor(
    const std::shared_ptr<Spacetime> &spacetime) const {
  if (!spacetime) return std::numeric_limits<double>::infinity();
  return objectiveOf(objectiveTermsFor(spacetime));
}

double MultiCobordism::objective() const { return objectiveFor(spacetime_); }

void MultiCobordism::declarePinnedRegion(PinnedRegion region) {
  for (auto &existing : pinnedRegions_)
    if (existing.name == region.name) {
      existing = std::move(region);
      return;
    }
  pinnedRegions_.push_back(std::move(region));
}

void MultiCobordism::clearPinnedRegions() { pinnedRegions_.clear(); }

void MultiCobordism::declareRegisterConstraint(RegisterConstraint constraint) {
  if (constraint.name.empty())
    throw std::invalid_argument(
        "MultiCobordism::declareRegisterConstraint: name is empty");
  if (constraint.degree < 0)
    throw std::invalid_argument(
        "MultiCobordism::declareRegisterConstraint: degree is negative");
  if (constraint.holes.size() != constraint.target.size())
    throw std::invalid_argument(
        "MultiCobordism::declareRegisterConstraint: hole/target size mismatch");
  const std::size_t expectedWidth =
      static_cast<std::size_t>(constraint.degree) + 2;
  for (auto &hole : constraint.holes) {
    if (hole.size() != expectedWidth)
      throw std::invalid_argument(
          "MultiCobordism::declareRegisterConstraint: malformed hole");
    std::sort(hole.begin(), hole.end());
    if (std::adjacent_find(hole.begin(), hole.end()) != hole.end())
      throw std::invalid_argument(
          "MultiCobordism::declareRegisterConstraint: repeated hole vertex");
  }
  for (auto &existing : registerConstraints_)
    if (existing.name == constraint.name) {
      existing = std::move(constraint);
      return;
    }
  registerConstraints_.push_back(std::move(constraint));
}

void MultiCobordism::clearRegisterConstraints() {
  registerConstraints_.clear();
}

std::set<std::uint64_t> MultiCobordism::pinnedVertices() const {
  std::set<std::uint64_t> united;
  for (const auto &region : pinnedRegions_)
    united.insert(region.vertices.begin(), region.vertices.end());
  return united;
}

bool MultiCobordism::edgeIsPinned(std::uint64_t a, std::uint64_t b) const {
  // Both endpoints within one region. One pinned endpoint leaves the edge free to
  // relax, and two regions that each hold one endpoint do not pin the edge that
  // spans between them — that edge is bulk.
  for (const auto &region : pinnedRegions_)
    if (region.vertices.count(a) != 0 && region.vertices.count(b) != 0)
      return true;
  return false;
}

}  // namespace tessera::cobordism
