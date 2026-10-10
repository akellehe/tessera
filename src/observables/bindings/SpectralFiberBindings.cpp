// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: spectral fibers and their
// tracker. One of the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_observables_spectral_fiber(py::module_ &m) {
  // ========================================
  // SpectralFiber: localized spectral bands and their certificates
  // ========================================
  py::class_<SpectralFiberConfig>(m, "SpectralFiberConfig",
      "Configuration of the spectral-band detector and tracker. "
      "Thresholds select which bands are certified, never "
      "which eigenvalues exist; no threshold is a Betti-number "
      "oracle and no rank is ever requested.")
      .def(py::init<>())
      .def_readwrite("degrees", &SpectralFiberConfig::degrees,
                     "Form degrees enumerated by enumerateOnComponents.")
      .def_readwrite("groupingTolerance",
                     &SpectralFiberConfig::groupingTolerance,
                     "Relative band-grouping width (fraction of the "
                     "spectral scale).")
      .def_readwrite("minRelativeGap", &SpectralFiberConfig::minRelativeGap,
                     "Isolation floor: certified bands need both gaps >= "
                     "minRelativeGap * scale (a closing gap returns an "
                     "uncertified band).")
      .def_readwrite("gapDominance", &SpectralFiberConfig::gapDominance,
                     "Certified gaps must exceed this multiple of the "
                     "in-band spread.")
      .def_readwrite("residualTolerance",
                     &SpectralFiberConfig::residualTolerance,
                     "Cap on the relative eigen/left/projector residuals.")
      .def_readwrite("gramDefectTolerance",
                     &SpectralFiberConfig::gramDefectTolerance,
                     "Cap on ||Phi^dagger W Phi - J||.")
      .def_readwrite("projectorNormCap",
                     &SpectralFiberConfig::projectorNormCap,
                     "Cap on the band projector norm ||P||_2.")
      .def_readwrite("maxLocalizationExcess",
                     &SpectralFiberConfig::maxLocalizationExcess,
                     "Cap on the band's rank-normalized localization excess "
                     "-- the localization acceptance conjunct.  1.0 accepts "
                     "any measured localization.")
      .def_readwrite("denseCrossover", &SpectralFiberConfig::denseCrossover,
                     "Dimension at/above which the self-adjoint path goes "
                     "sparse.")
      .def_readwrite("requestedEigenpairs",
                     &SpectralFiberConfig::requestedEigenpairs,
                     "Lowest eigenpairs the sparse block path computes.")
      .def_readwrite("oversample", &SpectralFiberConfig::oversample,
                     "Extra Ritz vectors beyond requestedEigenpairs.")
      .def_readwrite("maxSolverIterations",
                     &SpectralFiberConfig::maxSolverIterations)
      .def_readwrite("solverTolerance", &SpectralFiberConfig::solverTolerance)
      .def_readwrite("solverSeed", &SpectralFiberConfig::solverSeed,
                     "Seed of the deterministic sparse start block.")
      .def_readwrite("trackOverlapThreshold",
                     &SpectralFiberConfig::trackOverlapThreshold,
                     "Minimum subspace overlap for a certified track "
                     "continuation.")
      .def_readwrite("contourNodes", &SpectralFiberConfig::contourNodes,
                     "Whitney pencil path: trapezoidal node count of each band's Riesz contour.")
      .def_readwrite("isotropyTolerance", &SpectralFiberConfig::isotropyTolerance,
                     "Whitney pencil path: relative tolerance declaring a band's pairing isotropic.")
      .def_readwrite("resolventBoundCap", &SpectralFiberConfig::resolventBoundCap,
                     "Cap on the contour resolvent bound r * max_j "
                     "||(zeta_j I - h)^-1||_2 -- the 'controlled resolvent' "
                     "acceptance conjunct, enforced wherever a contour is drawn.")
      .def_readwrite("minAllowabilityMargin",
                     &SpectralFiberConfig::minAllowabilityMargin,
                     "Floor on the Kontsevich-Segal allowability margin of the "
                     "instance the band was read on; the default 0 asks for a "
                     "strictly positive margin.")
      .def_readwrite("lorentzianEpsilon", &SpectralFiberConfig::lorentzianEpsilon,
                     "The DECLARED Lorentzian-protocol rotation epsilon_L the "
                     "complex was rotated by (chainhodge.LorentzianFamily.rotate) "
                     "before it was handed to the tracker; NaN means the complex "
                     "was not declared Lorentzian. Never inferred from a squared "
                     "length. Acceptance requires epsilon_L > 0 where it is declared.")
      .def_readwrite("crossValidateDense",
                     &SpectralFiberConfig::crossValidateDense,
                     "Cross-check solves below the crossover against the "
                     "independent DenseReference kernel and record the "
                     "deviation on the certificate.");

  py::class_<SpectralBandCertificate>(m, "SpectralBandCertificate",
      R"doc(Certification record of one whole spectral band: degree, rank,
lower/upper gap, localization (projector-diagonal inverse participation
ratio, IPR), projector/eigen/left residuals, weighted Gram/signature defect
||Phi^dagger W Phi - J||, band condition number ||P||_2, Krein inertia
(p, q), frequency window, self-adjointness flag, and the graded Certificate
(BandWindow domain; an uncertified band carries HeuristicDiscovery, which
never holds).

A degenerate band is one object of rank >= 2; an unexplained multiplicity
is reported as its rank and never labeled.  Negative signature is a
certificate, never an automatic antiparticle identification.  Unmeasured
quantities are NaN, never zero.)doc")
      .def_readonly("degree", &SpectralBandCertificate::degree)
      .def_readonly("rank", &SpectralBandCertificate::rank)
      .def_readonly("lowerGap", &SpectralBandCertificate::lowerGap)
      .def_readonly("upperGap", &SpectralBandCertificate::upperGap)
      .def_readonly("nearestDiscardedSeparation",
                    &SpectralBandCertificate::nearestDiscardedSeparation,
                    "Distance in the complex plane to the nearest discarded "
                    "eigenvalue -- the isolation acceptance conjunct.")
      .def_readonly("localization", &SpectralBandCertificate::localization)
      .def_readonly("localizationSupportFraction",
                    &SpectralBandCertificate::localizationSupportFraction,
                    "Effective support fraction n_eff/n in [rank/n, 1]; 1 "
                    "exactly for a perfectly delocalized band.")
      .def_readonly("localizationExcess",
                    &SpectralBandCertificate::localizationExcess,
                    "(n_eff - rank)/(n - rank) in [0, 1] -- the gated "
                    "localization datum; 0 = as concentrated as the rank "
                    "permits, 1 = perfectly delocalized.")
      .def_readonly("projectorResidual",
                    &SpectralBandCertificate::projectorResidual)
      .def_readonly("eigenResidual", &SpectralBandCertificate::eigenResidual)
      .def_readonly("leftResidual", &SpectralBandCertificate::leftResidual)
      .def_readonly("gramDefect", &SpectralBandCertificate::gramDefect)
      .def_readonly("projectorNorm", &SpectralBandCertificate::projectorNorm,
                    "||P||_2, Kato's condition number of the spectral "
                    "projector (gauge-invariant).")
      .def_readonly("frameConditionNumber",
                    &SpectralBandCertificate::frameConditionNumber,
                    "The frame condition number: max Riesz conditioning of "
                    "the reported matched frames in the |W| metric.")
      .def_readonly("positiveSignature",
                    &SpectralBandCertificate::positiveSignature)
      .def_readonly("negativeSignature",
                    &SpectralBandCertificate::negativeSignature)
      .def_readonly("pairingDeterminant", &SpectralBandCertificate::pairingDeterminant,
                    "det B_C of the bilinear pairing (complex-symmetric pencil regime only).")
      .def_readonly("pairingCondition", &SpectralBandCertificate::pairingCondition)
      .def_readonly("pairingScale", &SpectralBandCertificate::pairingScale)
      .def_readonly("isotropic", &SpectralBandCertificate::isotropic,
                    "det B_C = 0: the exceptional-point indicator; no left frame.")
      .def_readonly("leftFrameRefusal", &SpectralBandCertificate::leftFrameRefusal)
      .def_readonly("metricSymmetryDefect", &SpectralBandCertificate::metricSymmetryDefect,
                    "The regime's verification residual, M L = (M L)^T.")
      .def_readonly("contour", &SpectralBandCertificate::contour,
                    "Description of the closed contour gamma_C the band's Riesz "
                    "projector was computed on; empty when no contour was drawn.")
      .def_readonly("contourNodeCount",
                    &SpectralBandCertificate::contourNodeCount,
                    "Quadrature node count of gamma_C (0 = no contour).")
      .def_readonly("contourCenter", &SpectralBandCertificate::contourCenter)
      .def_readonly("contourRadius", &SpectralBandCertificate::contourRadius)
      .def_readonly("resolventMax", &SpectralBandCertificate::resolventMax,
                    "max_j ||(zeta_j I - h_C)^-1||_2 over the contour nodes.")
      .def_readonly("resolventBound", &SpectralBandCertificate::resolventBound,
                    "The Riesz bound ||P_C|| <= (|gamma_C|/2 pi) max ||R||, i.e. "
                    "radius * resolventMax -- the gated contour quantity.")
      .def_readonly("allowable", &SpectralBandCertificate::allowable,
                    "Kontsevich-Segal allowability of the instance the band was "
                    "read on (every top simplex of strictly positive margin).")
      .def_readonly("allowabilityMargin",
                    &SpectralBandCertificate::allowabilityMargin,
                    "min_T (pi - sum_i |arg lambda_i(g_T)|): pi for a Euclidean "
                    "instance, 0 for a real Lorentzian one.")
      .def_readonly("lorentzianEpsilon",
                    &SpectralBandCertificate::lorentzianEpsilon,
                    "The declared rotation epsilon_L the band was read at; NaN "
                    "when the complex was not declared Lorentzian.")
      .def_readonly("bilinearLeftFrame", &SpectralBandCertificate::bilinearLeftFrame,
                    "Whether the stored left frame is the transpose dual Phi~ "
                    "itself (the chain-level pencil path) rather than Psi with "
                    "Psi^dagger W Phi = I.")
      .def_readonly("frequencyLower",
                    &SpectralBandCertificate::frequencyLower)
      .def_readonly("frequencyUpper",
                    &SpectralBandCertificate::frequencyUpper)
      .def_readonly("selfAdjoint", &SpectralBandCertificate::selfAdjoint)
      .def_readonly("accepted", &SpectralBandCertificate::accepted)
      .def_readonly("certificate", &SpectralBandCertificate::certificate)
      .def("describe", &SpectralBandCertificate::describe)
      .def("__repr__", &SpectralBandCertificate::describe);

  py::class_<FiberOverlapRead>(m, "FiberOverlapRead",
      "Principal-angle / support comparison of two fibers: cells matched "
      "by sorted vertex-id tuple (gauge- and relabeling-invariant).")
      .def_readonly("supportOverlap", &FiberOverlapRead::supportOverlap)
      .def_readonly("sharedCells", &FiberOverlapRead::sharedCells)
      .def_readonly("principalAngles", &FiberOverlapRead::principalAngles)
      .def_readonly("subspaceOverlap", &FiberOverlapRead::subspaceOverlap);

  py::class_<SpectralFiber>(m, "SpectralFiber",
      R"doc(One whole isolated spectral band of a component-restricted Hodge
operator: right/left frames, the transpose dual Phi~ (Phi~^T Phi = I, the
one pairing across regimes), band projector P = Phi Phi~^T (= Phi Psi^dagger
W off the pencil path), eigenvalues, and the SpectralBandCertificate.  The
band is represented by its projector;
individual eigenvectors are a gauge choice and never determine an identity
or a downstream observable.)doc")
      .def("degree", &SpectralFiber::degree)
      .def("rank", &SpectralFiber::rank)
      .def("accepted", &SpectralFiber::accepted)
      .def("rightFrame", &SpectralFiber::rightFrame,
           "Right frame Phi (cells x rank).")
      .def("leftFrame", &SpectralFiber::leftFrame,
           "Left frame as the regime's solver produced it: Psi with "
           "Psi^dagger W Phi = I, or Phi~ itself on the pencil path.")
      .def("dualFrame", &SpectralFiber::dualFrame,
           R"doc(The algebraic (transpose) dual Phi~ of the right frame,
Phi~^T Phi = I, in every regime: the stored left frame on the pencil path and
W conj(Psi) elsewhere. CovarianceState.fromBiorthogonalFrames(rightFrame(),
dualFrame()) is the band's biorthogonal Slater covariance.)doc")
      .def("projector", &SpectralFiber::projector,
           "The band projector P = Phi Phi~^T (cells x cells).")
      .def("weightDiagonal", &SpectralFiber::weightDiagonal,
           "Diagonal inner-product weights W restricted to the band's "
           "cells.")
      .def("eigenvalues", &SpectralFiber::eigenvalues,
           "Band eigenvalues (with multiplicity), sorted by (Re, Im).")
      .def("bandCenter", &SpectralFiber::bandCenter)
      .def("cellVertices", &SpectralFiber::cellVertices,
           "The k-cells carrying the band, as sorted vertex-id tuples in "
           "frame row order.")
      .def("certificate", &SpectralFiber::certificate,
           py::return_value_policy::copy)
      .def_static("overlap", &SpectralFiber::overlap, py::arg("a"),
                  py::arg("b"),
                  "Principal-angle / support comparison (cells matched by "
                  "vertex-id tuple).")
      .def("toRecord",
           [](const SpectralFiber &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the JSON-able record of the fiber "
           "(schema-versioned; complex leaves split _re/_im).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return SpectralFiber::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate from toRecord() output; rejects an unknown "
                  "schema_version (ValueError).");

  py::class_<SpectralBandWindow>(m, "SpectralBandWindow",
      "An accepted band's frequency window as plain data for the response "
      "consumer: lower and upper frequency bounds plus the band certificate. "
      "Carries no operator, frame, or quotient reference.")
      .def_readonly("degree", &SpectralBandWindow::degree)
      .def_readonly("rank", &SpectralBandWindow::rank)
      .def_readonly("frequencyLower", &SpectralBandWindow::frequencyLower)
      .def_readonly("frequencyUpper", &SpectralBandWindow::frequencyUpper)
      .def_readonly("certificate", &SpectralBandWindow::certificate);

  py::class_<FiberMatchRead>(m, "FiberMatchRead",
      "One matched fiber pair across frames or resolutions.  A certified "
      "continuation needs both endpoint bands accepted, equal ranks, and "
      "subspace overlap above the threshold — an endpoint whose gap closed "
      "is reported but never certified (no discontinuous identity flip).")
      .def_readonly("fromIndex", &FiberMatchRead::fromIndex)
      .def_readonly("toIndex", &FiberMatchRead::toIndex)
      .def_readonly("degree", &FiberMatchRead::degree)
      .def_readonly("overlap", &FiberMatchRead::overlap)
      .def_readonly("ranksEqual", &FiberMatchRead::ranksEqual)
      .def_readonly("certifiedContinuation",
                    &FiberMatchRead::certifiedContinuation);

  py::class_<ComponentBandRead>(m, "ComponentBandRead",
      "The band enumeration of one (component, degree) pair: the restricted "
      "operator's cells, verified regime, solver path, covered eigenvalues, "
      "every enumerated band (certified or not), and the solve certificate.")
      .def_readonly("support", &ComponentBandRead::support)
      .def_readonly("degree", &ComponentBandRead::degree)
      .def_readonly("dimension", &ComponentBandRead::dimension)
      .def_readonly("cellVertices", &ComponentBandRead::cellVertices)
      .def_readonly("regime", &ComponentBandRead::regime)
      .def_readonly("solverPath", &ComponentBandRead::solverPath)
      .def_readonly("truncated", &ComponentBandRead::truncated)
      .def_readonly("coveredEigenvalues",
                    &ComponentBandRead::coveredEigenvalues)
      .def_readonly("fibers", &ComponentBandRead::fibers)
      .def_readonly("solveCertificate", &ComponentBandRead::solveCertificate)
      .def("toRecord",
           [](const ComponentBandRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization of the whole read (fibers included).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return ComponentBandRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version "
                  "(ValueError).");

  py::class_<SpectralFiberTracker>(m, "SpectralFiberTracker",
      R"doc(Extraction and tracking of whole isolated localized Hodge bands
on persistent components.

For a component support S the tracker assembles the Hodge operator of the full
induced subcomplex on S under its metric source (the process-wide
HodgeLaplacian.defaultMetricSource() unless named).  Under the default
WhitneyPencil every degree k >= 1 is the covariant operator h_k(s, U) of the
subcomplex's own chain-level Whitney pencil, read in the complex-symmetric
pencil regime with Riesz bands and bilinear pairing certificates.  Under
DiagonalWeights it uses the same boundary maps, canonical cell order and
diagonal inner-product weights as the whole-complex HodgeLaplacian, so
support = all vertices reproduces a DiagonalWeights
HodgeLaplacian.laplacian(k) entry for entry.  Regimes are verified, never
assumed: positive -> self-adjoint solves (exact dense below the crossover,
deterministic sparse block shift-invert at or above it); real signed
weights -> W-self-adjointness verified, Krein inertia of Phi^dagger W Phi
recorded and normalized to diag(I_p, -I_q); complex weights -> matched
biorthogonal right/left subspaces with Psi^dagger W Phi = I.  Bands are
grouped by a relative gap rule and every band is reported with its projector
and certificate; a closing gap yields an uncertified band, never a different
identity.  No rank is requested and no eigenvalue threshold is a Betti
oracle.

Read-only: never calls a solver on the spacetime, never mutates it, and
nothing here enters any emergence objective.)doc")
      .def(py::init([](std::shared_ptr<Spacetime> st, const SpectralFiberConfig &cfg,
                       cobordism::HodgeLaplacian::MetricSource source) {
             return SpectralFiberTracker(std::move(st), cfg, source);
           }),
           py::arg("spacetime"), py::arg("config"), py::arg("metric_source"),
           "Bind with an explicit metric source; WhitneyPencil reads every degree >= 1 on the "
           "chain-level Whitney pencil in the complex-symmetric-pencil regime.")
      .def(py::init([](std::shared_ptr<Spacetime> st,
                       const SpectralFiberConfig &cfg,
                       const py::object &weights,
                       const py::object &metricSource) {
             const auto convention =
                 weights.is_none()
                     ? tessera::cobordism::HodgeLaplacian::
                           defaultWeightConvention()
                     : weights
                           .cast<tessera::cobordism::HodgeLaplacian::
                                     WeightConvention>();
             const auto source =
                 metricSource.is_none()
                     ? tessera::cobordism::HodgeLaplacian::defaultMetricSource()
                     : metricSource.cast<tessera::cobordism::HodgeLaplacian::MetricSource>();
             return SpectralFiberTracker(std::move(st), cfg, convention, source);
           }),
           py::arg("spacetime"), py::arg("config") = SpectralFiberConfig{},
           py::arg("weights") = py::none(),
           py::arg("metric_source") = py::none(),
           "Bind to the spacetime to read; weights=None follows the "
           "process-wide HodgeLaplacian.defaultWeightConvention() and "
           "metric_source=None the process-wide "
           "HodgeLaplacian.defaultMetricSource() (the Whitney pencil unless "
           "changed), both at call time. The weight convention is read only "
           "under DiagonalWeights.")
      .def("metricSource", &SpectralFiberTracker::metricSource,
           "Where this tracker's operators take their metric from.")
      .def("config", &SpectralFiberTracker::config,
           py::return_value_policy::copy)
      .def("weightConvention", &SpectralFiberTracker::weightConvention)
      .def("enumerateBands",
           [](const SpectralFiberTracker &self,
              const std::vector<std::uint64_t> &support, int degree) {
             py::gil_scoped_release release;
             return self.enumerateBands(support, degree);
           },
           py::arg("support"), py::arg("degree"),
           "Enumerate the bands of one component (vertex-id support) at "
           "one form degree.")
      .def("enumerateOnComponents",
           [](const SpectralFiberTracker &self,
              const std::vector<ComponentRead> &components) {
             py::gil_scoped_release release;
             return self.enumerateOnComponents(components);
           },
           py::arg("components"),
           "Enumerate every configured degree on every component.")
      .def("enumerateBandsCached",
           [](const SpectralFiberTracker &self,
              tessera::cobordism::AnalyticCache &cache,
              const std::vector<std::uint64_t> &support, int degree) {
             py::gil_scoped_release release;
             return self.enumerateBandsCached(cache, support, degree);
           },
           py::arg("cache"), py::arg("support"), py::arg("degree"),
           "enumerateBands through the AnalyticCache contract "
           "(touched-star invalidation; served while the component is "
           "untouched).")
      .def_static("acceptedWindows", &SpectralFiberTracker::acceptedWindows,
                  py::arg("reads"),
                  "The accepted bands' frequency windows as plain data for "
                  "the response consumer.")
      .def_static("matchFibers", &SpectralFiberTracker::matchFibers,
                  py::arg("fromFibers"), py::arg("toFibers"),
                  py::arg("overlapThreshold") = 0.5,
                  "Track fibers across frames/resolutions by principal "
                  "angles and component overlap.")
      .def_readonly_static("CACHE_KIND", &SpectralFiberTracker::kCacheKind);
}
