// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: the fiber connection and
// winding reads. One of the translation units that Bindings.cpp registers in
// order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_fiber_connection(py::module_ &m) {
  // ========================================
  // FiberConnection: derived U(r) fiber transport, Wilson
  // observables, rank-three center structure, determinant winding
  // ========================================
  py::class_<FiberConnectionConfig>(m, "FiberConnectionConfig",
      R"doc(Threshold configuration of the derived-transport gates.  Every
gate fires before polar / pseudo-unitary reduction; a failed gate yields a
rejected read that still reports its raw map and diagnostics, so polar
normalization never conceals a bad assignment.)doc")
      .def(py::init<>())
      .def_readwrite("rankTolerance", &FiberConnectionConfig::rankTolerance,
                     "Relative singular-value cut for the numerical rank.")
      .def_readwrite("leakageTolerance",
                     &FiberConnectionConfig::leakageTolerance,
                     "Cap on the isometry leakage before a unitary factor "
                     "may be emitted.")
      .def_readwrite("conditionNumberCap",
                     &FiberConnectionConfig::conditionNumberCap,
                     "Cap on endpoint frame and overlap conditioning.")
      .def_readwrite("minEndpointGap", &FiberConnectionConfig::minEndpointGap,
                     "Absolute floor on each endpoint band's isolation "
                     "min(lowerGap, upperGap); 0 = rely on band "
                     "certification.")
      .def_readwrite("requireCertifiedFibers",
                     &FiberConnectionConfig::requireCertifiedFibers,
                     "Require both endpoint bands accepted (a closing gap "
                     "rejects the transport).")
      .def_readwrite("certificateTolerance",
                     &FiberConnectionConfig::certificateTolerance,
                     "Tolerance the emitted certificates hold against.")
      .def_readwrite("closureTolerance",
                     &FiberConnectionConfig::closureTolerance,
                     "Relative endpoint-mismatch cap for certified winding "
                     "closures.");

  py::class_<FiberTransportRead>(m, "FiberTransportRead",
      R"doc(One derived fiber transport A <- B: the raw overlap
M_AB = Phi_A^dagger W_A T_AB Phi_B (Psi_A^dagger on the biorthogonal path),
every pre-normalization diagnostic (rank, singular values, leakage,
endpoint gaps and signatures, frame conditioning), the normalized U(r) or
pseudo-unitary factor when its gates passed, the determinant-line datum,
and the graded certificate.  A rejected read still carries the raw map and
diagnostics.  Per-transport winding and center fields live on the dedicated
family reads (DeterminantWindingRead / FundamentalLiftRead): an integer
winding exists only for a declared family and closure, a center sector only
for a declared lift path.)doc")
      .def_readonly("toKey", &FiberTransportRead::toKey,
                    "Order-independent key of the destination fiber A.")
      .def_readonly("fromKey", &FiberTransportRead::fromKey,
                    "Order-independent key of the source fiber B.")
      .def_readonly("degree", &FiberTransportRead::degree)
      .def_readonly("rank", &FiberTransportRead::rank)
      .def_readonly("rawMap", &FiberTransportRead::rawMap,
                    "M_AB before any normalization.")
      .def_readonly("singularValues", &FiberTransportRead::singularValues,
                    "Singular values of rawMap, descending.")
      .def_readonly("numericalRank", &FiberTransportRead::numericalRank)
      .def_readonly("leakage", &FiberTransportRead::leakage,
                    "Regime-appropriate isometry defect.")
      .def_readonly("overlapConditionNumber",
                    &FiberTransportRead::overlapConditionNumber)
      .def_readonly("toGap", &FiberTransportRead::toGap)
      .def_readonly("fromGap", &FiberTransportRead::fromGap)
      .def_readonly("toPositiveSignature",
                    &FiberTransportRead::toPositiveSignature)
      .def_readonly("toNegativeSignature",
                    &FiberTransportRead::toNegativeSignature)
      .def_readonly("fromPositiveSignature",
                    &FiberTransportRead::fromPositiveSignature)
      .def_readonly("fromNegativeSignature",
                    &FiberTransportRead::fromNegativeSignature)
      .def_readonly("toProjectorNorm", &FiberTransportRead::toProjectorNorm)
      .def_readonly("fromProjectorNorm",
                    &FiberTransportRead::fromProjectorNorm)
      .def_readonly("frameConditionNumber",
                    &FiberTransportRead::frameConditionNumber,
                    "max of the endpoints' frame condition numbers, "
                    "distinct from the projector norms.")
      .def_readonly("regime", &FiberTransportRead::regime)
      .def_readonly("unitaryMap", &FiberTransportRead::unitaryMap,
                    "The emitted U(r)/pseudo-unitary factor; empty when "
                    "rejected or on the certified GL(r,C) non-normal path.")
      .def_readonly("determinantPhase", &FiberTransportRead::determinantPhase,
                    "det of the emitted factor (U(1)); the raw determinant "
                    "phase on the GL path -- never discarded.")
      .def_readonly("polarResidual", &FiberTransportRead::polarResidual)
      .def_readonly("determinantResidual",
                    &FiberTransportRead::determinantResidual)
      .def_readonly("projectiveOnly", &FiberTransportRead::projectiveOnly)
      .def_readonly("accepted", &FiberTransportRead::accepted)
      .def_readonly("rejectionReason", &FiberTransportRead::rejectionReason)
      .def_readonly("certificate", &FiberTransportRead::certificate)
      .def("describe", &FiberTransportRead::describe)
      .def("__repr__", &FiberTransportRead::describe)
      .def("toRecord",
           [](const FiberTransportRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization (`transports`): "
           "at rank three the full U(3) factor, det V, and thereby the "
           "PU(3) class travel.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return FiberTransportRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<WilsonHolonomyRead>(m, "WilsonHolonomyRead",
      R"doc(The product of accepted transports around a loop: full U(r)
holonomy (or the certified GL(r,C) product),
normalized trace Tr H / r, determinant line det H, and the center-blind
adjoint reads.  Under independent local frame changes a closed holonomy is
conjugated at its base component, so the normalized trace is invariant.)doc")
      .def_readonly("rank", &WilsonHolonomyRead::rank)
      .def_readonly("loopLength", &WilsonHolonomyRead::loopLength)
      .def_readonly("closed", &WilsonHolonomyRead::closed)
      .def_readonly("baseKey", &WilsonHolonomyRead::baseKey)
      .def_readonly("holonomy", &WilsonHolonomyRead::holonomy)
      .def_readonly("normalizedTrace", &WilsonHolonomyRead::normalizedTrace)
      .def_readonly("determinant", &WilsonHolonomyRead::determinant)
      .def_readonly("adjointTrace", &WilsonHolonomyRead::adjointTrace,
                    "|Tr H|^2 - 1 -- center-blind.")
      .def_readonly("adjointMatrix", &WilsonHolonomyRead::adjointMatrix,
                    "Rank 3 only: the faithful PU(3) image on the traceless "
                    "octet (ColorFiber::adjointOctetProjector conventions).")
      .def_readonly("unitarityResidual",
                    &WilsonHolonomyRead::unitarityResidual)
      .def_readonly("unitary", &WilsonHolonomyRead::unitary)
      .def_readonly("maxLeakage", &WilsonHolonomyRead::maxLeakage,
                    "Worst pre-normalization isometry defect over the links.")
      .def_readonly("minEndpointGap", &WilsonHolonomyRead::minEndpointGap,
                    "Worst endpoint band isolation over the links.")
      .def_readonly("maxFrameConditionNumber",
                    &WilsonHolonomyRead::maxFrameConditionNumber,
                    "Worst endpoint frame conditioning over the links.")
      .def_readonly("minSingularValue",
                    &WilsonHolonomyRead::minSingularValue,
                    "Smallest singular value over the links' raw overlaps.")
      .def_readonly("minNumericalRank",
                    &WilsonHolonomyRead::minNumericalRank,
                    "Smallest numerical rank over the links.")
      .def_readonly("certificate", &WilsonHolonomyRead::certificate);

  py::class_<FundamentalLiftRead>(m, "FundamentalLiftRead",
      R"doc(The explicitly lifted SU(3) fundamental holonomy: a cube-root
branch continued from a declared base
branch s0, lift = H exp(-i Theta/3) omega^{-s0} with Theta the accumulated
per-link principal determinant phase, and the accumulated Z3 center sector
recorded (branch-independent; the lift shifts by omega^{-s0} across
branches while every projective/adjoint read of it is branch-independent).
Rank three only -- SU(3) is never hard-coded at generic rank.)doc")
      .def_readonly("rank", &FundamentalLiftRead::rank)
      .def_readonly("baseBranch", &FundamentalLiftRead::baseBranch)
      .def_readonly("lift", &FundamentalLiftRead::lift)
      .def_readonly("liftTrace", &FundamentalLiftRead::liftTrace)
      .def_readonly("centerSector", &FundamentalLiftRead::centerSector)
      .def_readonly("accumulatedDeterminantPhase",
                    &FundamentalLiftRead::accumulatedDeterminantPhase)
      .def_readonly("maxDeterminantPhaseStep",
                    &FundamentalLiftRead::maxDeterminantPhaseStep)
      .def_readonly("detResidual", &FundamentalLiftRead::detResidual)
      .def_readonly("valid", &FundamentalLiftRead::valid)
      .def_readonly("invalidReason", &FundamentalLiftRead::invalidReason)
      .def_readonly("certificate", &FundamentalLiftRead::certificate)
      .def("toRecord",
           [](const FundamentalLiftRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the lift and its accumulated center "
           "sector travel together.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return FundamentalLiftRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<WindingClosureSpec> windingClosure(m, "WindingClosureSpec",
      R"doc(The declared closure of an open-segment determinant winding: how
the open composite is closed is part of the certificate.  Mode.NONE leaves
the winding unknown; a raw endpoint phase difference is never promoted to
an integer.)doc");
  py::enum_<WindingClosureSpec::Mode>(windingClosure, "Mode")
      .value("NONE", WindingClosureSpec::Mode::None)
      .value("MATCHED_REFERENCE", WindingClosureSpec::Mode::MatchedReference)
      .value("ENDPOINT_TRIVIALIZATION",
             WindingClosureSpec::Mode::EndpointTrivialization);
  windingClosure.def(py::init<>())
      .def_readwrite("mode", &WindingClosureSpec::mode)
      .def_readwrite("referenceId", &WindingClosureSpec::referenceId,
                     "Caller-supplied reference specification id, recorded "
                     "verbatim on the read.")
      .def_readwrite("referenceTransports",
                     &WindingClosureSpec::referenceTransports,
                     "MATCHED_REFERENCE: one reference transport per "
                     "segment sample (same orientation; traversed "
                     "backwards by the closure).")
      .def_readwrite("startTrivialization",
                     &WindingClosureSpec::startTrivialization)
      .def_readwrite("endTrivialization",
                     &WindingClosureSpec::endTrivialization);

  py::class_<DeterminantWindingRead>(m, "DeterminantWindingRead",
      R"doc(The integer determinant winding of a closed full-rank transport
family, or the relative winding of an open segment under a recorded
closure.  `winding` is None when
invalidated (closed gap / lost rank / aliasing step) or when no closure
was declared -- never a silently wrong integer.)doc")
      .def_readonly("winding", &DeterminantWindingRead::winding)
      .def_readonly("windingClosure", &DeterminantWindingRead::windingClosure)
      .def_readonly("windingReferenceId",
                    &DeterminantWindingRead::windingReferenceId)
      .def_readonly("accumulatedPhase",
                    &DeterminantWindingRead::accumulatedPhase)
      .def_readonly("maxPhaseStep", &DeterminantWindingRead::maxPhaseStep)
      .def_readonly("phaseStepMargin",
                    &DeterminantWindingRead::phaseStepMargin)
      .def_readonly("closureDefect", &DeterminantWindingRead::closureDefect)
      .def_readonly("invalidationReason",
                    &DeterminantWindingRead::invalidationReason)
      .def_readonly("certificate", &DeterminantWindingRead::certificate)
      .def("toRecord",
           [](const DeterminantWindingRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the closure specification travels "
           "with the integer; an unknown winding serializes as unknown, "
           "never as zero.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return DeterminantWindingRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<FiberConnection>(m, "FiberConnection",
      R"doc(Derived spectral-frame transport and Wilson observables.  Wraps
the induced-transfer machinery (the whole-complex Hodge d'Alembertian's
intercomponent block and RecursiveQuotient response-network blocks), forms
the overlap M_AB = Phi_A^dagger W_A T_AB Phi_B (Psi_A^dagger on the
biorthogonal path), reports every diagnostic before normalization, gates,
and only then reduces to the polar U(r) / pseudo-unitary factor.  Composes
accepted maps into full U(r), determinant-line, projective/adjoint, and
explicitly lifted fundamental holonomies; certifies closed-family and
declared open-segment determinant windings.

Read-only observable: consumes accepted SpectralFibers, mutates nothing,
and none of its outputs enters any emergence objective.  The link matrix is
always reconstructed from neighboring Hodge frames with a leakage
certificate, never sampled independently.)doc")
      .def(py::init<FiberConnectionConfig>(),
           py::arg("config") = FiberConnectionConfig{})
      .def("config", &FiberConnection::config,
           py::return_value_policy::reference_internal)
      .def_static("chainTransfer",
                  [](const std::shared_ptr<Spacetime> &st, int degree,
                     const std::vector<std::vector<std::uint64_t>> &toCells,
                     const std::vector<std::vector<std::uint64_t>> &fromVertexTuples,
                     std::optional<cobordism::HodgeLaplacian::WeightConvention>
                         weights,
                     std::optional<cobordism::HodgeLaplacian::MetricSource> source) {
                    return FiberConnection::chainTransfer(
                        st, degree, toCells, fromVertexTuples,
                        weights.value_or(cobordism::HodgeLaplacian::
                                             defaultWeightConvention()),
                        source.value_or(cobordism::HodgeLaplacian::defaultMetricSource()));
                  },
                  py::arg("st"), py::arg("degree"), py::arg("to_cells"),
                  py::arg("from_cells"), py::arg("weights") = py::none(),
                  py::arg("metric_source") = py::none(),
                  "The chain transfer T_AB induced by the connecting "
                  "simplices: the off-diagonal block L_k[cells(to), "
                  "cells(from)] of the whole-complex Hodge operator, cells "
                  "matched by sorted vertex-id tuple.  Under WhitneyPencil "
                  "(degree >= 1) the operator is h_k(s, U) on chains in the "
                  "reference orientation, the basis of the tracker's Whitney "
                  "bands.  weights = None and metric_source = None follow the "
                  "process-wide defaults at call time.")
      .def_static("responseTransfer", &FiberConnection::responseTransfer,
                  py::arg("network"), py::arg("to_component"),
                  py::arg("from_component"),
                  "The effective response block of a supplied "
                  "response network (rows = to's stalk, cols = from's "
                  "stalk; zero block when the network carries no such "
                  "edge).")
      .def("transport", &FiberConnection::transport, py::arg("to_fiber"),
           py::arg("from_fiber"), py::arg("transfer"),
           "Derive the transport A <- B from an explicit transfer block "
           "(rows = A's cells, cols = B's cells): overlap, full "
           "diagnostics, gates, then reduction, in that order.")
      .def("transportReverse", &FiberConnection::transportReverse,
           py::arg("to_fiber"), py::arg("from_fiber"), py::arg("transfer"),
           "The reverse-direction transport B <- A through the W-adjoint "
           "reverse block T_BA = W_B^{-1} T_AB^dagger W_A (exact in the "
           "W-self-adjoint regimes, where it returns the adjoint/inverse "
           "factor).")
      .def("transportOnSpacetime",
           [](const FiberConnection &self, const std::shared_ptr<Spacetime> &st,
              const SpectralFiber &to, const SpectralFiber &from,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w,
              std::optional<cobordism::HodgeLaplacian::MetricSource> source) {
             return self.transportOnSpacetime(
                 st, to, from,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()),
                 source.value_or(cobordism::HodgeLaplacian::defaultMetricSource()));
           },
           py::arg("st"), py::arg("to_fiber"), py::arg("from_fiber"),
           py::arg("weights") = py::none(), py::arg("metric_source") = py::none(),
           "Derive the transport on a spacetime: assembles the chain "
           "transfer from the Hodge operator, then transport().")
      .def("transportOnSpacetimeCached",
           [](const FiberConnection &self, cobordism::AnalyticCache &cache,
              const std::shared_ptr<Spacetime> &st, const SpectralFiber &to,
              const SpectralFiber &from,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w,
              std::optional<cobordism::HodgeLaplacian::MetricSource> source) {
             return self.transportOnSpacetimeCached(
                 cache, st, to, from,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()),
                 source.value_or(cobordism::HodgeLaplacian::defaultMetricSource()));
           },
           py::arg("cache"), py::arg("st"), py::arg("to_fiber"),
           py::arg("from_fiber"), py::arg("weights") = py::none(),
           py::arg("metric_source") = py::none(),
           "transportOnSpacetime through the AnalyticCache contract "
           "(key: the union of the two fibers' cell-vertex sets; cached "
           "equals cold).")
      .def("holonomy", &FiberConnection::holonomy, py::arg("links"),
           "Multiply accepted transports along a chain; reports the full "
           "holonomy, normalized trace, determinant line, and adjoint "
           "reads (closed = the keys chain into a loop).")
      .def("holonomyOnSpacetime",
           [](const FiberConnection &self, const std::shared_ptr<Spacetime> &st,
              const std::vector<SpectralFiber> &fibers,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w,
              std::optional<cobordism::HodgeLaplacian::MetricSource> source) {
             return self.holonomyOnSpacetime(
                 st, fibers,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()),
                 source.value_or(cobordism::HodgeLaplacian::defaultMetricSource()));
           },
           py::arg("st"), py::arg("fibers"), py::arg("weights") = py::none(),
           py::arg("metric_source") = py::none(),
           "Wilson loop over an ordered cycle of fibers: links "
           "fibers[i] <- fibers[i+1] (wrapping), then the product.")
      .def("holonomyOnSpacetimeCached",
           [](const FiberConnection &self, cobordism::AnalyticCache &cache,
              const std::shared_ptr<Spacetime> &st,
              const std::vector<SpectralFiber> &fibers,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w,
              std::optional<cobordism::HodgeLaplacian::MetricSource> source) {
             return self.holonomyOnSpacetimeCached(
                 cache, st, fibers,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()),
                 source.value_or(cobordism::HodgeLaplacian::defaultMetricSource()));
           },
           py::arg("cache"), py::arg("st"), py::arg("fibers"),
           py::arg("weights") = py::none(), py::arg("metric_source") = py::none(),
           "holonomyOnSpacetime through the AnalyticCache: per-link caching "
           "plus the loop product keyed by all participating fibers, so a "
           "published TouchedStar invalidates only the loops touching the "
           "changed star.")
      .def_static("projectiveRepresentative",
                  &FiberConnection::projectiveRepresentative,
                  py::arg("unitary"), py::arg("gate"),
                  "A canonical PU(3) class representative: V / (det "
                  "V)^{1/3} with the principal cube root (the class {U, "
                  "omega U, omega^2 U} is the faithful datum). Gated on the "
                  "triangle-anchor certificate: a closed AnchorGate raises, "
                  "because rank three plus an accepted transport is not a "
                  "licence to emit a colour datum.")
      .def_static("adjointRepresentation",
                  &FiberConnection::adjointRepresentation, py::arg("unitary"),
                  "The faithful PU(3) image of a 3x3 unitary on the "
                  "traceless octet (ColorFiber conventions; center-blind).")
      .def("fundamentalLift", &FiberConnection::fundamentalLift,
           py::arg("links"), py::arg("gate"), py::arg("base_branch") = 0,
           "Continue a cube-root branch along the links from the declared "
           "base branch and record the accumulated Z3 center sector. Gated "
           "on the triangle-anchor certificate: a closed AnchorGate reports "
           "valid=False carrying the gate's own refusal reason.")
      .def("closedFamilyWinding", &FiberConnection::closedFamilyWinding,
           py::arg("family"),
           "Integer determinant winding of a closed transport family "
           "(cyclic samples); invalidated when a gap/rank closes or a "
           "phase step reaches pi.")
      .def("openSegmentWinding", &FiberConnection::openSegmentWinding,
           py::arg("segment"), py::arg("closure"),
           "Relative determinant winding of an open cobordism segment "
           "under the declared closure (matched-reference or endpoint "
           "trivializations), with the specification recorded; unknown "
           "when no closure is declared.")
      .def_static("fiberKey", &FiberConnection::fiberKey, py::arg("fiber"),
                  "Order-independent key of a fiber (Fingerprint over its "
                  "deduplicated cell-vertex-id set).");
}
