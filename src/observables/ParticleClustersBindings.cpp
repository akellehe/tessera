// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: particle clusters: quarks,
// gluons, mesons, diquarks and baryons. One of the translation units that
// Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_particle_clusters(py::module_ &m) {
  // ---- ParticleClusters: quark/antiquark classification ---------

  py::class_<ParticleClustersConfig>(m, "ParticleClustersConfig",
      R"doc(Analysis thresholds of the particle classification.
Every value selects which reads are certified, never which value is
reported, and the whole configuration is echoed on every read
(QuarkRead.thresholds).)doc")
      .def(py::init<>())
      .def_readwrite("parity_tolerance",
                     &ParticleClustersConfig::parityTolerance,
                     "|<(-1)^N> -+ 1| cap for a definite parity sign.")
      .def_readwrite("occupation_tolerance",
                     &ParticleClustersConfig::occupationTolerance,
                     "|<N> - 1| cap for the single-fermion occupation.")
      .def_readwrite("min_anchor_score",
                     &ParticleClustersConfig::minAnchorScore,
                     "Calibrated anchor atlas-score floor (a^2 in [0,1]).")
      .def_readwrite("min_phase_coherence",
                     &ParticleClustersConfig::minPhaseCoherence,
                     "Determinant-phase coherence floor of the anchor.")
      .def_readwrite("max_transport_leakage",
                     &ParticleClustersConfig::maxTransportLeakage,
                     "Cap on the worst lifetime transport leakage.")
      .def_readwrite("min_persistence_lifetime",
                     &ParticleClustersConfig::minPersistenceLifetime,
                     "Minimum lifetime across cobordism frames; the "
                     "modularity resolution-slice count never gates the "
                     "classification of a candidate.")
      .def_readwrite("min_persistence_overlap",
                     &ParticleClustersConfig::minPersistenceOverlap,
                     "Minimum adjacent-frame track overlap.")
      .def_readwrite("min_localization",
                     &ParticleClustersConfig::minLocalization,
                     "Band-localization floor (0 accepts any measured "
                     "localization; NaN still fails).")
      .def_readwrite("min_refinement_overlap",
                     &ParticleClustersConfig::minRefinementOverlap,
                     "Minimum band subspace overlap across a refinement.")
      .def_readwrite("min_stability_frames",
                     &ParticleClustersConfig::minStabilityFrames,
                     "Frames a 'stable' quark condition must hold at; "
                     "the stability conditions are across-frame "
                     "statements about the track.")
      .def_readwrite("doublet_overlap_threshold",
                     &ParticleClustersConfig::doubletOverlapThreshold,
                     "Subspace-overlap threshold of the doublet tracking.")
      .def_readwrite("min_doublet_frames",
                     &ParticleClustersConfig::minDoubletFrames,
                     "Minimum frames a flavor subclass must persist.")
      .def_readwrite("isospin_tolerance",
                     &ParticleClustersConfig::isospinTolerance,
                     "|I3 -+ 1/2| cap for a definite doublet member.")
      .def_readwrite("gauss_tolerance",
                     &ParticleClustersConfig::gaussTolerance,
                     "Max nested-surface deviation (and |Im| leakage) for "
                     "a consistent Gauss flux.")
      .def_readwrite("min_enclosing_surfaces",
                     &ParticleClustersConfig::minEnclosingSurfaces,
                     "Minimum nested surfaces for a consistency claim.")
      .def_readwrite("ud_tolerance", &ParticleClustersConfig::udTolerance,
                     "|Q_gauss - (I3 + B/2)| cap for the proposed u/d "
                     "identification.")
      .def_readwrite("min_octet_weight",
                     &ParticleClustersConfig::minOctetWeight,
                     "Floor on a gluon candidate's octet Frobenius "
                     "weight (a genuinely nonzero color polarization).")
      .def_readwrite("octet_purity_tolerance",
                     &ParticleClustersConfig::octetPurityTolerance,
                     "Cap on the (I9 - P8) residual of the excitation "
                     "(machine-level: the traceless bilinear is octet "
                     "exactly).")
      .def_readwrite("composite_octet_tolerance",
                     &ParticleClustersConfig::compositeOctetTolerance,
                     "Cap on the octet fraction of a meson's pair "
                     "color bilinear (the color-singlet certificate).")
      .def_readwrite("min_anti_triplet_weight",
                     &ParticleClustersConfig::minAntiTripletWeight,
                     "Floor on the certified anti-triplet wedge "
                     "occupation det(C^dag Gamma C) of a diquark.")
      .def_readwrite("color_gram_tolerance",
                     &ParticleClustersConfig::colorGramTolerance,
                     "|det(C^dag C) - 1| cap of the color-singlet "
                     "certificate (exactly 1 for an orthonormal triad, "
                     "exactly 0 for duplicate color modes).")
      .def_readwrite("color_flux_tolerance",
                     &ParticleClustersConfig::colorFluxTolerance,
                     "Cap on the net color flux: the octet weight "
                     "of the bound object's color bilinear.  An "
                     "independent finite-complex diagnostic, never on its "
                     "own a proof of confinement.")
      .def_readwrite("spin_expectation_tolerance",
                     &ParticleClustersConfig::spinExpectationTolerance,
                     "|<J^2> - 3/4| cap of the total-space spin "
                     "expectation.")
      .def_readwrite("spin_variance_tolerance",
                     &ParticleClustersConfig::spinVarianceTolerance,
                     "|Var(J^2)| cap the reported complex variance is "
                     "graded against; the sharp-spin certificate is the "
                     "pair of eigen-equations, not this cap.")
      .def_readwrite("min_support_containment",
                     &ParticleClustersConfig::minSupportContainment,
                     "Minimum fraction of a constituent's level-0 "
                     "support inside the supercomponent (1.0 = full).")
      .def_readwrite("min_lifetime_overlap",
                     &ParticleClustersConfig::minLifetimeOverlap,
                     "Minimum number of shared persistence slices "
                     "across the three constituents' lifetimes.")
      .def_readwrite("min_radius", &ParticleClustersConfig::minRadius,
                     "Strict floor a finite emergent radius must "
                     "exceed.")
      .def_readwrite("max_profile_deviation",
                     &ParticleClustersConfig::maxProfileDeviation,
                     "Cap on the deviation of every dimensionless "
                     "scale channel across the refinement window.");

  py::class_<GaussFluxRead>(m, "GaussFluxRead",
      R"doc(The electric Gauss-flux consistency read over nested enclosing
surfaces.  Each per-surface flux is the EigenstateSynthesis.gauss_law_charge
value: an exact signed sum of the supplied field-strength 2-cochain over
the closed-star boundary, restricted to electric (timelike-leg) plaquettes
when electricOnly.
Charge is certified only when consistent across at least
minEnclosingSurfaces surfaces; otherwise electricFlux is None (unknown),
never zero.  No metric regime is verified by the sum, so the certificate
carries the non-normal (no self-adjointness claimed) regime tag.)doc")
      .def(py::init<>())
      .def_readonly("fluxes", &GaussFluxRead::fluxes,
                    "Per-surface complex fluxes, in input surface order.")
      .def_readonly("surface_vertex_counts",
                    &GaussFluxRead::surfaceVertexCounts,
                    "Distinct enclosed vertices per surface (nesting "
                    "witness).")
      .def_readonly("electric_only", &GaussFluxRead::electricOnly)
      .def_readonly("max_deviation", &GaussFluxRead::maxDeviation,
                    "Max |flux_i - flux_j| over surface pairs.")
      .def_readonly("imag_leakage", &GaussFluxRead::imagLeakage,
                    "Max |Im flux_i| (never silently discarded).")
      .def_readonly("consistent", &GaussFluxRead::consistent)
      .def_readonly("electric_flux", &GaussFluxRead::electricFlux,
                    "Re(mean) of the agreeing surfaces; None = unknown.")
      .def_readonly("failed_certificates",
                    &GaussFluxRead::failedCertificates)
      .def_readonly("certificate", &GaussFluxRead::certificate);

  py::class_<FlavorDoubletRead>(m, "FlavorDoubletRead",
      R"doc(The emergent, unlabeled, transported two-state spectral
subclass that could carry isospin.  The search runs without a requested
dimension: stableSubclassRanks reports every stable rank found, and
"two-state" is an outcome.  The stored first-frame doublet fiber is the
recorded member trivialization, a compilation convention and never a
physical u/d label.)doc")
      .def(py::init<>())
      .def_readonly("found", &FlavorDoubletRead::found,
                    "Exactly one stable two-state subclass emerged.")
      .def_readonly("degree", &FlavorDoubletRead::degree)
      .def_readonly("rank", &FlavorDoubletRead::rank,
                    "2 when found; never requested.")
      .def_readonly("frames_tracked", &FlavorDoubletRead::framesTracked)
      .def_readonly("min_continuation_overlap",
                    &FlavorDoubletRead::minContinuationOverlap,
                    "Smallest certified continuation overlap on the track.")
      .def_readonly("min_isolation", &FlavorDoubletRead::minIsolation,
                    "Worst band isolation min(lowerGap, upperGap) along "
                    "the track.")
      .def_readonly("stable_subclass_ranks",
                    &FlavorDoubletRead::stableSubclassRanks,
                    "Ranks of all stable subclasses (the no-requested-"
                    "dimension witness).")
      .def_readonly("two_state_count", &FlavorDoubletRead::twoStateCount,
                    "Stable two-state subclasses (found needs exactly 1).")
      .def_readonly("doublet", &FlavorDoubletRead::doublet,
                    "First-frame fiber of the winning subclass (the "
                    "recorded trivialization).")
      .def_readonly("failed_certificates",
                    &FlavorDoubletRead::failedCertificates)
      .def_readonly("invalidation_reason",
                    &FlavorDoubletRead::invalidationReason)
      .def_readonly("certificate", &FlavorDoubletRead::certificate);

  py::class_<QuarkCandidateEvidence>(m, "QuarkCandidateEvidence",
      R"doc(The assembled evidence bundle of one candidate.  Every field is
a read produced by an upstream kernel (persistence, spectral bands,
anchors, transports and windings, Wick reads, the Gauss read); the
classifier never recomputes any of them.  Unsupplied evidence is missing
evidence: the corresponding certificate fails by name, never presumed to
pass.)doc")
      .def(py::init<>())
      .def_readwrite("component", &QuarkCandidateEvidence::component,
                     "Label-free component identity.")
      .def_readwrite("color_band", &QuarkCandidateEvidence::colorBand,
                     "The selected band (rank is read, never "
                     "requested).")
      .def_readwrite("color_band_frames",
                     &QuarkCandidateEvidence::colorBandFrames,
                     "The band at each cobordism frame; the quark "
                     "condition 'stable rank three' is decided here.")
      .def_readwrite("anchor", &QuarkCandidateEvidence::anchor,
                     "Calibrated anchor profile of the band.")
      .def_readwrite("anchor_frames", &QuarkCandidateEvidence::anchorFrames,
                     "The anchor profile at each cobordism frame; the "
                     "quark condition of a stable profile with "
                     "determinant-line coherence is decided here.")
      .def_readwrite("dressed_anchor", &QuarkCandidateEvidence::dressedAnchor,
                     "The Section 10 anchor certificate of the band by the "
                     "dressed coordinate (chainhodge.DressedAnchor.profile). "
                     "A supplied read that refuses names the 'dressed-anchor' "
                     "certificate; an absent one leaves the channel "
                     "unmeasured. Reported, never gating.")
      .def_readwrite("lifetime_transports",
                     &QuarkCandidateEvidence::lifetimeTransports,
                     "World-tube transports (all must be accepted).")
      .def_readwrite("winding", &QuarkCandidateEvidence::winding,
                     "Determinant-line winding with its recorded "
                     "closure specification.")
      .def_readwrite("parity_read", &QuarkCandidateEvidence::parityRead,
                     "CovarianceState.wick_parity of the carried "
                     "state.")
      .def_readwrite("occupation_read",
                     &QuarkCandidateEvidence::occupationRead,
                     "CovarianceState.wick_total_number of the state.")
      .def_readwrite("persistence_lifetime",
                     &QuarkCandidateEvidence::persistenceLifetime,
                     "Modularity resolution-slice lifetime "
                     "(report-only; NaN = missing).")
      .def_readwrite("persistence_min_overlap",
                     &QuarkCandidateEvidence::persistenceMinOverlap,
                     "Smallest adjacent-slice overlap (report-only).")
      .def_readwrite("frame_lifetime",
                     &QuarkCandidateEvidence::frameLifetime,
                     "Cobordism-frame lifetime "
                     "(PersistentModularity.track_across_frames) -- the gated "
                     "persistence quantity.")
      .def_readwrite("frame_min_overlap",
                     &QuarkCandidateEvidence::frameMinOverlap,
                     "Smallest adjacent-frame support overlap -- the gated "
                     "predecessor/successor overlap.")
      .def_readwrite("refinement_overlap",
                     &QuarkCandidateEvidence::refinementOverlap,
                     "Band subspace overlap across a refinement "
                     "(SpectralFiber.overlap).")
      .def_readwrite("flavor", &QuarkCandidateEvidence::flavor,
                     "flavorDoubletSearch result; None = flavor unknown.")
      .def_readwrite("doublet_occupancy",
                     &QuarkCandidateEvidence::doubletOccupancy,
                     "Amplitudes on the two doublet members in the "
                     "recorded trivialization; None = unknown.")
      .def_readwrite("doublet_orientation",
                     &QuarkCandidateEvidence::doubletOrientation,
                     "Declared orientation s in {+1,-1}: which member "
                     "carries I3=+1/2 under the proposed identification "
                     "(a recorded convention, never a hidden label).")
      .def_readwrite("charge", &QuarkCandidateEvidence::charge,
                     "gaussFluxOnSurfaces result; None = charge unknown.");

  py::class_<QuarkRead>(m, "QuarkRead",
      R"doc(The quark/antiquark particle read, plus the evidence summary
the classification consumed, the recorded thresholds, and the
certificate.  Unknown or
uncertified values are None/NaN/0-sign, never zero-filled, and every gap
is named in failedCertificates.  B = nu/3 exists exactly when the winding
certificate does; quark-ness additionally needs |nu| = 1.)doc")
      .def(py::init<>())
      .def_readonly("component", &QuarkRead::component)
      .def_readonly("exterior_parity", &QuarkRead::exteriorParity,
                    "-1 odd / +1 even / 0 unknown (an uncertified parity "
                    "read never emits a sign).")
      .def_readonly("color_rank", &QuarkRead::colorRank)
      .def_readonly("triangle_anchor_score", &QuarkRead::triangleAnchorScore)
      .def_readonly("triangle_anchor_max_term",
                    &QuarkRead::triangleAnchorMaxTerm)
      .def_readonly("triangle_anchor_participation",
                    &QuarkRead::triangleAnchorParticipation)
      .def_readonly("anchor_phase_dispersion",
                    &QuarkRead::anchorPhaseDispersion)
      .def_readonly("anchor_phase_coherence",
                    &QuarkRead::anchorPhaseCoherence)
      .def_readonly("anchor_weighting_id", &QuarkRead::anchorWeightingId)
      .def_readonly("determinant_winding", &QuarkRead::determinantWinding,
                    "Certified nu; None when invalidated/unclosed.")
      .def_readonly("winding_closure", &QuarkRead::windingClosure,
                    "The recorded closure specification.")
      .def_readonly("winding_reference_id", &QuarkRead::windingReferenceId)
      .def_readonly("baryon_flux", &QuarkRead::baryonFlux,
                    "B = nu/3 under a certified winding; None = unknown, "
                    "never inserted.")
      .def_readonly("isospin", &QuarkRead::isospin,
                    "I3 = +-1/2 under the certified doublet hypothesis; "
                    "None = unknown.")
      .def_readonly("electric_flux", &QuarkRead::electricFlux,
                    "Gauss-consistent charge; None unless both the Gauss "
                    "read and the flavor doublet are certified.")
      .def_readonly("confidence", &QuarkRead::confidence,
                    "Passed fraction of the ten core certificates.")
      .def_readonly("failed_certificates", &QuarkRead::failedCertificates,
                    "Every failed/missing certificate, by name.")
      .def_readonly("classification", &QuarkRead::classification,
                    "'quark' (nu=+1) / 'antiquark' (nu=-1) / 'none'.")
      .def_readonly("occupation_total", &QuarkRead::occupationTotal)
      .def_readonly("transport_count", &QuarkRead::transportCount)
      .def_readonly("transport_leakage_max", &QuarkRead::transportLeakageMax)
      .def_readonly("persistence_lifetime", &QuarkRead::persistenceLifetime,
                    "Modularity resolution-slice lifetime (reported).")
      .def_readonly("persistence_min_overlap",
                    &QuarkRead::persistenceMinOverlap)
      .def_readonly("frame_lifetime", &QuarkRead::frameLifetime,
                    "Cobordism-frame lifetime (the gated quantity).")
      .def_readonly("frame_min_overlap", &QuarkRead::frameMinOverlap)
      .def_readonly("stability_frames", &QuarkRead::stabilityFrames,
                    "Frames the stability certificates were measured over.")
      .def_readonly("anchor_score_spread", &QuarkRead::anchorScoreSpread)
      .def_readonly("anchor_coherence_spread",
                    &QuarkRead::anchorCoherenceSpread)
      .def_readonly("band_continuation_overlap",
                    &QuarkRead::bandContinuationOverlap)
      .def_readonly("localization", &QuarkRead::localization)
      .def_readonly("localization_support_fraction",
                    &QuarkRead::localizationSupportFraction)
      .def_readonly("refinement_overlap", &QuarkRead::refinementOverlap)
      .def_readonly("ud_identification_proposed",
                    &QuarkRead::udIdentificationProposed,
                    "Q = I3 + B/2 was tested and held (the proposed u/d "
                    "identification, never a charge definition).")
      .def_readonly("doublet_orientation", &QuarkRead::doubletOrientation)
      .def_readonly("thresholds", &QuarkRead::thresholds,
                    "The configuration that produced this read.")
      .def_readonly("certificate", &QuarkRead::certificate)
      .def("describe", &QuarkRead::describe)
      .def("__repr__", &QuarkRead::describe)
      .def("to_record",
           [](const QuarkRead &self) { return recordToPython(self.toRecord()); },
           "Checkpoint serialization (particles.quarks): fields, "
           "evidence summary, failed "
           "certificates, and the threshold echo; unknown values are "
           "null, never zero.")
      .def_static("from_record",
                  [](const py::handle &record) {
                    return QuarkRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<ConjugatePairRead>(m, "ConjugatePairRead",
      R"doc(Pair-conservation verification of a conjugate quark-antiquark
creation path: total certified winding, total baryon flux, and total
parity.  A singular (gap/rank-closing) leg leaves the totals unknown
(None) -- never zero by assumption.)doc")
      .def(py::init<>())
      .def_readonly("total_winding", &ConjugatePairRead::totalWinding,
                    "nu_a + nu_b when both certified; None otherwise.")
      .def_readonly("total_baryon_flux", &ConjugatePairRead::totalBaryonFlux,
                    "B_a + B_b when both known; None = unknown flux.")
      .def_readonly("total_parity", &ConjugatePairRead::totalParity,
                    "Product of certified parities; 0 = unknown.")
      .def_readonly("parity_even", &ConjugatePairRead::parityEven)
      .def_readonly("conserved", &ConjugatePairRead::conserved,
                    "Both windings certified, total 0, even parity.")
      .def_readonly("failed_certificates",
                    &ConjugatePairRead::failedCertificates)
      .def_readonly("certificate", &ConjugatePairRead::certificate);

  // ---- even sectors: octet bilinear + gluon/meson/diquark ----------

  py::class_<OctetBilinearRead>(m, "OctetBilinearRead",
      R"doc(The quasi-free traceless-bilinear (octet) read of three declared
color modes of a carried CovarianceState: the bilinear matrix
M_ij = <a_i^dag a_j> (the transposed principal submatrix of Gamma), its
exact 1+8 split (delegated to ColorFiber), the adjoint Casimir (= 3 for a
nonzero excitation, since C = 3 P8), the quartic-Wick color Casimir
expectation <sum_a d_gamma(lambda_a/2)^2> (exactly 4/3 on the fundamental and
anti-triplet Slater states, 0 on the vacuum and full singlet), the octet
coordinates Tr(lambda_a M)/2, and the certified subset occupation and
parity.  Evaluated on the covariance (polynomial in the mode count, no Fock
vector), so adding vacuum-embedded microscopic modes leaves the read
unchanged.  Unknown values are NaN / 0-sign, never zero.)doc")
      .def(py::init<>())
      .def_readwrite("color_modes", &OctetBilinearRead::colorModes,
                     "The three declared color modes (the recorded color "
                     "trivialization order).")
      .def_readonly("occupation", &OctetBilinearRead::occupation,
                    "Certified subset occupation <N_S>; NaN = unknown.")
      .def_readonly("subset_parity", &OctetBilinearRead::subsetParity,
                    "+1 / -1 / 0 = unknown or indefinite.")
      .def_readonly("bilinear", &OctetBilinearRead::bilinear,
                    "M_ij = <a_i^dag a_j> on the declared modes.")
      .def_readonly("octet_component", &OctetBilinearRead::octetComponent,
                    "ColorFiber.traceless_part(bilinear) -- the excitation.")
      .def_readonly("octet_weight", &OctetBilinearRead::octetWeight,
                    "||M - (tr M/3) I||_F^2 (ColorFiber.octet_read).")
      .def_readonly("singlet_weight", &OctetBilinearRead::singletWeight,
                    "|tr M|^2 / 3.")
      .def_readonly("octet_projector_residual",
                    &OctetBilinearRead::octetProjectorResidual,
                    "||(I9 - P8) vec(M8)|| / ||M8||_F -- rounding-level; "
                    "NaN when the excitation vanishes.")
      .def_readonly("casimir", &OctetBilinearRead::casimir,
                    "ColorFiber.adjoint_casimir(octetComponent) in [0, 3].")
      .def_readonly("casimir_expectation",
                    &OctetBilinearRead::casimirExpectation,
                    "<sum_a d_gamma(lambda_a/2)^2> by quartic Wick sums.")
      .def_readonly("gell_mann_components",
                    &OctetBilinearRead::gellMannComponents,
                    "Tr(lambda_a M)/2 for a = 1..8.")
      .def_readonly("residual", &OctetBilinearRead::residual,
                    "Max residual of the consumed Wick reads.")
      .def_readonly("certificate", &OctetBilinearRead::certificate)
      .def("describe", &OctetBilinearRead::describe)
      .def("__repr__", &OctetBilinearRead::describe)
      .def("to_record",
           [](const OctetBilinearRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization (complex leaves split _re/_im).")
      .def_static("from_record",
                  [](const py::handle &record) {
                    return OctetBilinearRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<GluonCandidateEvidence>(m, "GluonCandidateEvidence",
      R"doc(The assembled evidence bundle of one gluon candidate: the
quasi-free octet bilinear read of the carried state, the carried-state
Wick parity and occupation, the lifetime transports and determinant
winding, and the persistence lifetime.  Missing evidence fails its
certificate by name.)doc")
      .def(py::init<>())
      .def_readwrite("component", &GluonCandidateEvidence::component,
                     "Label-free component identity of the excitation.")
      .def_readwrite("binding_component",
                     &GluonCandidateEvidence::bindingComponent,
                     "The component the excitation is bound to (reported "
                     "verbatim as the binding component).")
      .def_readwrite("octet", &GluonCandidateEvidence::octet,
                     "octetBilinearRead output of the carried state.")
      .def_readwrite("parity_read", &GluonCandidateEvidence::parityRead,
                     "CovarianceState.wick_parity of the whole carried "
                     "state (the even-parity gate).")
      .def_readwrite("occupation_read",
                     &GluonCandidateEvidence::occupationRead,
                     "wick_total_number (report-only).")
      .def_readwrite("lifetime_transports",
                     &GluonCandidateEvidence::lifetimeTransports,
                     "Transports: accepted, rank three, leakage under "
                     "the cap (the accepted-octet-transport gate).")
      .def_readwrite("winding", &GluonCandidateEvidence::winding,
                     "Determinant winding; a certified nu = 0 is the "
                     "zero-baryon-flux evidence.")
      .def_readwrite("persistence_lifetime",
                     &GluonCandidateEvidence::persistenceLifetime,
                     "Modularity resolution-slice lifetime (report-only).")
      .def_readwrite("frame_lifetime",
                     &GluonCandidateEvidence::frameLifetime,
                     "Cobordism-frame lifetime; the gated quantity.");

  py::class_<GluonRead>(m, "GluonRead",
      R"doc(The gluon-candidate read: a persistent transported octet
excitation with certified even parity and certified zero total
determinant winding / baryon flux.  classification is "gluon-candidate"
or "none" -- never "gluon": no even octet excitation is claimed to be a
physical gluon.  Unknown values are None/NaN/0-sign, never zero-filled;
every gap is named in failed_certificates ("parity-even",
"octet-excitation", "octet-purity", "octet-transport", "winding-zero",
"persistence").)doc")
      .def(py::init<>())
      .def_readonly("component", &GluonRead::component)
      .def_readonly("binding_component", &GluonRead::bindingComponent)
      .def_readonly("classification", &GluonRead::classification,
                    "'gluon-candidate' or 'none'.")
      .def_readonly("exterior_parity", &GluonRead::exteriorParity,
                    "+1 even / -1 odd / 0 unknown.")
      .def_readonly("occupation_total", &GluonRead::occupationTotal)
      .def_readonly("casimir", &GluonRead::casimir,
                    "Flat consumed-scalar summary of the octet evidence "
                    "(one source of truth: the full OctetBilinearRead "
                    "travels on the evidence).")
      .def_readonly("casimir_expectation", &GluonRead::casimirExpectation,
                    "The quartic-Wick color Casimir expectation consumed.")
      .def_readonly("octet_projector_residual",
                    &GluonRead::octetProjectorResidual)
      .def_readonly("octet_weight", &GluonRead::octetWeight)
      .def_readonly("singlet_weight", &GluonRead::singletWeight)
      .def_readonly("determinant_winding", &GluonRead::determinantWinding,
                    "Certified nu (0 for a candidate); None = unknown.")
      .def_readonly("winding_closure", &GluonRead::windingClosure)
      .def_readonly("winding_reference_id", &GluonRead::windingReferenceId)
      .def_readonly("baryon_flux", &GluonRead::baryonFlux,
                    "0.0 is a certified zero flux; None = unknown, never "
                    "zero by default.")
      .def_readonly("transport_count", &GluonRead::transportCount)
      .def_readonly("transport_leakage_max", &GluonRead::transportLeakageMax)
      .def_readonly("persistence_lifetime", &GluonRead::persistenceLifetime,
                    "Modularity resolution-slice lifetime (reported).")
      .def_readonly("frame_lifetime", &GluonRead::frameLifetime,
                    "Cobordism-frame lifetime (the gated quantity).")
      .def_readonly("confidence", &GluonRead::confidence,
                    "Passed fraction of the six gluon certificates.")
      .def_readonly("failed_certificates", &GluonRead::failedCertificates)
      .def_readonly("thresholds", &GluonRead::thresholds)
      .def_readonly("certificate", &GluonRead::certificate)
      .def("describe", &GluonRead::describe)
      .def("__repr__", &GluonRead::describe)
      .def("to_record",
           [](const GluonRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization (particles.gluons).")
      .def_static("from_record",
                  [](const py::handle &record) {
                    return GluonRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<CompositeCandidateEvidence>(m, "CompositeCandidateEvidence",
      R"doc(The assembled evidence bundle of one two-cluster composite
(meson or diquark; three-cluster composites are handled by BaryonRead):
the two constituent QuarkReads consumed verbatim, the carried composite
occupation, the meson-channel pair color bilinear, the diquark-channel
certified anti-triplet wedge read, composite transports, and the composite
persistence lifetime.)doc")
      .def(py::init<>())
      .def_readwrite("binding_component",
                     &CompositeCandidateEvidence::bindingComponent,
                     "The component binding the two clusters.")
      .def_readwrite("first", &CompositeCandidateEvidence::first,
                     "First constituent's QuarkRead.")
      .def_readwrite("second", &CompositeCandidateEvidence::second,
                     "Second constituent's QuarkRead.")
      .def_readwrite("occupation_read",
                     &CompositeCandidateEvidence::occupationRead,
                     "wickTotalNumber of the carried composite state "
                     "(report-only).")
      .def_readwrite("color_pairing",
                     &CompositeCandidateEvidence::colorPairing,
                     "Meson channel: the 3x3 pair color bilinear in "
                     "3 x 3bar (singlet composite: M ~ I); None = missing "
                     "-- the color-singlet certificate fails by name.")
      .def_readwrite("anti_triplet_read",
                     &CompositeCandidateEvidence::antiTripletRead,
                     "Diquark channel: the certified Lambda^2 C^3 wedge "
                     "occupation det(C^dag Gamma C) ("
                     "wickGramDeterminant) -- exactly zero for duplicated "
                     "color modes (Pauli).")
      .def_readwrite("lifetime_transports",
                     &CompositeCandidateEvidence::lifetimeTransports,
                     "Composite transports (report-only for the "
                     "two-cluster reads).")
      .def_readwrite("persistence_lifetime",
                     &CompositeCandidateEvidence::persistenceLifetime,
                     "Composite track lifetime (NaN = missing).");

  py::class_<MesonRead>(m, "MesonRead",
      R"doc(The meson-candidate read: one certified quark plus one
certified antiquark (order-insensitive), even composite parity (the
exact graded product of the certified constituent parities -- the
standard parity table), a color-singlet pair bilinear under the exact
1+8 split, and zero total certified winding and baryon flux (the
conjugate-pair integer sums).  failedCertificates vocabulary:
"constituent-quark", "constituent-antiquark", "parity-even",
"color-singlet", "flux-zero".)doc")
      .def(py::init<>())
      .def_readonly("binding_component", &MesonRead::bindingComponent)
      .def_readonly("first_constituent", &MesonRead::firstConstituent)
      .def_readonly("second_constituent", &MesonRead::secondConstituent)
      .def_readonly("classification", &MesonRead::classification,
                    "'meson-candidate' or 'none'.")
      .def_readonly("exterior_parity", &MesonRead::exteriorParity,
                    "Exact constituent-parity product; 0 = unknown.")
      .def_readonly("occupation_total", &MesonRead::occupationTotal)
      .def_readonly("pairing_singlet_weight",
                    &MesonRead::pairingSingletWeight)
      .def_readonly("pairing_octet_weight", &MesonRead::pairingOctetWeight)
      .def_readonly("pairing_octet_fraction",
                    &MesonRead::pairingOctetFraction,
                    "octet/(octet+singlet) of the pairing; NaN = missing.")
      .def_readonly("total_winding", &MesonRead::totalWinding,
                    "nu1 + nu2 when both certified; None = unknown.")
      .def_readonly("total_baryon_flux", &MesonRead::totalBaryonFlux,
                    "B1 + B2 when both known; None = unknown, never zero.")
      .def_readonly("transport_count", &MesonRead::transportCount)
      .def_readonly("transport_leakage_max", &MesonRead::transportLeakageMax)
      .def_readonly("persistence_lifetime", &MesonRead::persistenceLifetime)
      .def_readonly("confidence", &MesonRead::confidence)
      .def_readonly("failed_certificates", &MesonRead::failedCertificates)
      .def_readonly("thresholds", &MesonRead::thresholds)
      .def_readonly("certificate", &MesonRead::certificate)
      .def("describe", &MesonRead::describe)
      .def("__repr__", &MesonRead::describe)
      .def("to_record",
           [](const MesonRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization.")
      .def_static("from_record",
                  [](const py::handle &record) {
                    return MesonRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<DiquarkRead>(m, "DiquarkRead",
      R"doc(The diquark-candidate read: two certified quarks
(nu = +1 each), even composite parity, a certified anti-triplet wedge
occupation, and the preserved constituent baryon flux B = 2/3.
Explicitly not an antiquark: the 3bar color representation coincides,
but occupation two, even parity, and B = +2/3 (vs one/odd/-1/3) are the
recorded distinction channels.  failedCertificates vocabulary:
"constituent-quarks", "parity-even", "anti-triplet",
"baryon-flux-two-thirds".)doc")
      .def(py::init<>())
      .def_readonly("binding_component", &DiquarkRead::bindingComponent)
      .def_readonly("first_constituent", &DiquarkRead::firstConstituent)
      .def_readonly("second_constituent", &DiquarkRead::secondConstituent)
      .def_readonly("classification", &DiquarkRead::classification,
                    "'diquark-candidate' or 'none'.")
      .def_readonly("exterior_parity", &DiquarkRead::exteriorParity,
                    "Exact constituent-parity product; 0 = unknown.")
      .def_readonly("occupation_total", &DiquarkRead::occupationTotal)
      .def_readonly("anti_triplet_weight", &DiquarkRead::antiTripletWeight,
                    "Certified wedge occupation; NaN = unknown.")
      .def_readonly("total_winding", &DiquarkRead::totalWinding,
                    "nu1 + nu2 when both certified (2 for a candidate).")
      .def_readonly("total_baryon_flux", &DiquarkRead::totalBaryonFlux,
                    "B1 + B2 (2/3 for a candidate); None = unknown.")
      .def_readonly("transport_count", &DiquarkRead::transportCount)
      .def_readonly("transport_leakage_max",
                    &DiquarkRead::transportLeakageMax)
      .def_readonly("persistence_lifetime",
                    &DiquarkRead::persistenceLifetime)
      .def_readonly("confidence", &DiquarkRead::confidence)
      .def_readonly("failed_certificates", &DiquarkRead::failedCertificates)
      .def_readonly("thresholds", &DiquarkRead::thresholds)
      .def_readonly("certificate", &DiquarkRead::certificate)
      .def("describe", &DiquarkRead::describe)
      .def("__repr__", &DiquarkRead::describe)
      .def("to_record",
           [](const DiquarkRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization.")
      .def_static("from_record",
                  [](const py::handle &record) {
                    return DiquarkRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<BoundCandidateEvidence>(m, "BoundCandidateEvidence",
      R"doc(One constituent's datum for the bound-supercomponent search:
the quark verdict, the level-0 support, the persistence window
(first, last) -- None means no lifetime evidence, and the overlap
certificate then fails by name -- and the mutual transports to the other
constituents.)doc")
      .def(py::init<>())
      .def_readwrite("quark", &BoundCandidateEvidence::quark,
                     "The candidate's QuarkRead (only a certified "
                     "'quark' verdict counts toward the three-quark "
                     "census).")
      .def_readwrite("support", &BoundCandidateEvidence::support,
                     "Level-0 cell support (ComponentRead.support); empty "
                     "= missing evidence.")
      .def_readwrite("lifetime", &BoundCandidateEvidence::lifetime,
                     "(firstSlice, lastSlice) of the PersistenceTrack "
                     "window, inclusive; None = unknown.")
      .def_readwrite("mutual_transports",
                     &BoundCandidateEvidence::mutualTransports,
                     "Transports to the other constituents; every "
                     "supplied link must be accepted under the leakage "
                     "cap.");

  py::class_<BoundSupercomponentRead>(m, "BoundSupercomponentRead",
      R"doc(One next-modular-level component examined by the
bound-supercomponent search: the contained certified quark candidates,
their shared lifetime window, and the containment/transport
certificates.  failedCertificates vocabulary: "supercomponent-level",
"quark-count", "support-containment", "lifetime-overlap",
"transport-containment".)doc")
      .def(py::init<>())
      .def_readonly("bound_component",
                    &BoundSupercomponentRead::boundComponent)
      .def_readonly("quarks", &BoundSupercomponentRead::quarks,
                    "Contained certified quark candidates' ids.")
      .def_readonly("quark_indices", &BoundSupercomponentRead::quarkIndices,
                    "Their indices in the input candidate list.")
      .def_readonly("found", &BoundSupercomponentRead::found,
                    "A certified bound supercomponent of exactly three "
                    "lifetime-overlapping certified quark candidates.")
      .def_readonly("lifetime_window",
                    &BoundSupercomponentRead::lifetimeWindow,
                    "Shared (first, last) window; None = disjoint/unknown.")
      .def_readonly("lifetime_overlap",
                    &BoundSupercomponentRead::lifetimeOverlap,
                    "Number of shared persistence slices.")
      .def_readonly("min_containment",
                    &BoundSupercomponentRead::minContainment,
                    "Smallest per-constituent support-containment "
                    "fraction; NaN = unknown.")
      .def_readonly("transport_leakage_max",
                    &BoundSupercomponentRead::transportLeakageMax)
      .def_readonly("transport_count",
                    &BoundSupercomponentRead::transportCount)
      .def_readonly("failed_certificates",
                    &BoundSupercomponentRead::failedCertificates)
      .def_readonly("thresholds", &BoundSupercomponentRead::thresholds)
      .def_readonly("certificate", &BoundSupercomponentRead::certificate)
      .def("describe", &BoundSupercomponentRead::describe)
      .def("__repr__", &BoundSupercomponentRead::describe);

  py::class_<ScaleProfileSample>(m, "ScaleProfileSample",
      R"doc(One refinement-window sample of the mass-radius battery
(InteriorHinges).  radialWeightProfile is the share of the
|Re eps * star h| curvature weight per breadth-first-search shell: a radial
curvature-weight density, not a momentum-transfer form factor.  No Fourier
transform of a charge density is computed anywhere in this tree.)doc")
      .def(py::init<>())
      .def_readwrite("radius", &ScaleProfileSample::radius,
                     "r = V_dual^(1/4) (InteriorHinges.Radii.rDual) -- "
                     "dimensionful; only its finiteness is certified.")
      .def_readwrite("radius_cross_check",
                     &ScaleProfileSample::radiusCrossCheck,
                     "r = V_primal^(1/4); its ratio to radius is the "
                     "dimensionless channel.")
      .def_readwrite("spectral_mass", &ScaleProfileSample::spectralMass,
                     "The intensive shell mass m_shell -- a mean interior "
                     "deficit angle, dimensionless in lattice units.")
      .def_readwrite("localization", &ScaleProfileSample::localization,
                     "Curvature-weight participation ratio (dimensionless).")
      .def_readwrite("radial_weight_profile",
                     &ScaleProfileSample::radialWeightProfile,
                     "Per-shell curvature-weight shares, shell ascending "
                     "(dimensionless); empty = no shell seeds, profile "
                     "unknown.")
      .def_readwrite("color_gram_determinant",
                     &ScaleProfileSample::colorGramDeterminant,
                     "det(C^dag C) at this refinement.")
      .def_readwrite("rotation_character",
                     &ScaleProfileSample::rotationCharacter,
                     "The 2pi rotation character at this refinement.")
      .def_readwrite("baryon_flux", &ScaleProfileSample::baryonFlux,
                     "B = nu/3 at this refinement.")
      .def_readwrite("electric_flux", &ScaleProfileSample::electricFlux,
                     "Summed certified Gauss flux at this refinement.")
      .def_readwrite("composite_parity",
                     &ScaleProfileSample::compositeParity,
                     "-1 odd / +1 even / 0 unknown at this refinement "
                     "(an integer channel: stability is exact equality).")
      .def_readwrite("anchor_score", &ScaleProfileSample::anchorScore,
                     "Worst constituent anchor score at this refinement.");

  py::class_<ScaleProfileRead>(m, "ScaleProfileRead",
      R"doc(The refinement-window certificate: a finite emergent
radius plus the refinement stability of every dimensionless channel.
physicalMass is always None -- a dimensionful mass stays unknown until a
physical scale is independently established.  failedCertificates
vocabulary: "refinement-window", "finite-radius",
"radius-ratio-stability", "spectral-mass-stability",
"localization-stability", "profile-stability".)doc")
      .def(py::init<>())
      .def_readonly("sample_count", &ScaleProfileRead::sampleCount)
      .def_readonly("radius", &ScaleProfileRead::radius)
      .def_readonly("radius_finite", &ScaleProfileRead::radiusFinite)
      .def_readonly("radius_ratio", &ScaleProfileRead::radiusRatio)
      .def_readonly("radius_ratio_spread",
                    &ScaleProfileRead::radiusRatioSpread)
      .def_readonly("spectral_mass", &ScaleProfileRead::spectralMass)
      .def_readonly("spectral_mass_spread",
                    &ScaleProfileRead::spectralMassSpread)
      .def_readonly("localization", &ScaleProfileRead::localization)
      .def_readonly("localization_spread",
                    &ScaleProfileRead::localizationSpread)
      .def_readonly("profile_max_deviation",
                    &ScaleProfileRead::profileMaxDeviation,
                    "Max absolute per-shell deviation across the window; "
                    "NaN = unknown, never zero.")
      .def_readonly("profile_shells", &ScaleProfileRead::profileShells)
      .def_readonly("color_gram_determinant",
                    &ScaleProfileRead::colorGramDeterminant)
      .def_readonly("color_gram_spread", &ScaleProfileRead::colorGramSpread)
      .def_readonly("rotation_character",
                    &ScaleProfileRead::rotationCharacter)
      .def_readonly("rotation_character_spread",
                    &ScaleProfileRead::rotationCharacterSpread)
      .def_readonly("baryon_flux", &ScaleProfileRead::baryonFlux)
      .def_readonly("baryon_flux_spread", &ScaleProfileRead::baryonFluxSpread)
      .def_readonly("electric_flux", &ScaleProfileRead::electricFlux)
      .def_readonly("electric_flux_spread",
                    &ScaleProfileRead::electricFluxSpread)
      .def_readonly("composite_parity", &ScaleProfileRead::compositeParity)
      .def_readonly("composite_parity_stable",
                    &ScaleProfileRead::compositeParityStable)
      .def_readonly("anchor_score", &ScaleProfileRead::anchorScore)
      .def_readonly("anchor_score_spread",
                    &ScaleProfileRead::anchorScoreSpread)
      .def_readonly("physical_mass", &ScaleProfileRead::physicalMass,
                    "Always None: unknown until a physical scale is "
                    "independently established.")
      .def_readonly("stable", &ScaleProfileRead::stable)
      .def_readonly("failed_certificates",
                    &ScaleProfileRead::failedCertificates)
      .def_readonly("thresholds", &ScaleProfileRead::thresholds)
      .def_readonly("certificate", &ScaleProfileRead::certificate)
      .def("describe", &ScaleProfileRead::describe)
      .def("__repr__", &ScaleProfileRead::describe);

  py::class_<BaryonCandidateEvidence>(m, "BaryonCandidateEvidence",
      R"doc(The assembled evidence bundle of one three-cluster candidate:
the three constituent verdicts consumed verbatim, the bound-supercomponent
search result, the three normalized anchored color columns (the wedge is
built once from them), the octet bilinear read of the bound object (the
independent net-color-flux diagnostic), the Berry-cancelled 2pi rotation
character and optional Spin(d) lift, the Wick <J^2> and Var(J^2), the
accepted covariance-only class's variance reads, and the refinement-window
mass-radius samples.)doc")
      .def(py::init<>())
      .def_readwrite("bound_component",
                     &BaryonCandidateEvidence::boundComponent)
      .def_readwrite("quarks", &BaryonCandidateEvidence::quarks,
                     "The three constituents' QuarkReads.  Assign the "
                     "whole list (ev.quarks = [a, b, c]): like every "
                     "std::array/std::vector binding, reading it yields a "
                     "copy, so item assignment does not stick.")
      .def_readwrite("binding", &BaryonCandidateEvidence::binding,
                     "The boundSupercomponentSearch result.")
      .def_readwrite("color_columns", &BaryonCandidateEvidence::colorColumns,
                     "The 3x3 matrix of normalized anchored color columns "
                     "C = [c_A c_B c_C]; the three-mode wedge is built "
                     "once from it -- no extra fermion sign is multiplied "
                     "onto the color epsilon.")
      .def_readwrite("color_flux", &BaryonCandidateEvidence::colorFlux,
                     "The bound object's OctetBilinearRead -- the "
                     "independent net-color-flux diagnostic.")
      .def_readwrite("rotation", &BaryonCandidateEvidence::rotation,
                     "PhysicalRotation character of the closed 2pi "
                     "total-space cluster-frame cycle.  Report-only: a "
                     "rigid rotation leaves every band constant, so this "
                     "character is +1 along any rigid cycle whatever the "
                     "spin, and it gates nothing.")
      .def_readwrite("monopole_spin", &BaryonCandidateEvidence::monopoleSpin,
                     "MonopoleSupport.spin_read of the cluster's bounding "
                     "cut: the monopole number of the U(1) part of the "
                     "connection, the cocycle of the rotation group's "
                     "projective action, and the j = 1/2 doublet it "
                     "protects.  None fails 'odd-monopole' and "
                     "'projective-cocycle' by name.")
      .def_readwrite("sharp_spin_eigen",
                     &BaryonCandidateEvidence::sharpSpinEigen,
                     "SharpSpin.read of the two eigen-equations on the "
                     "bounded superposition of determinants -- the "
                     "sharp-spin certificate.  None fails 'sharp-spin' by "
                     "name; it is never inferred from the expectation or "
                     "from the variance.")
      .def_readwrite("exchange", &BaryonCandidateEvidence::exchange,
                     "The particle-exchange character, when the "
                     "exchange experiment was run.  Report-only: the "
                     "proton certificate has no exchange row, so this "
                     "read gates nothing.")
      .def_readwrite("continuum_spin_claim",
                     &BaryonCandidateEvidence::continuumSpinClaim,
                     "When True the SO(d)->Spin(d) lift is required; when "
                     "False it is never demanded.")
      .def_readwrite("spin_lift", &BaryonCandidateEvidence::spinLift,
                     "spinLift decision; None = none made.")
      .def_readwrite("spin_squared_read",
                     &BaryonCandidateEvidence::spinSquaredRead,
                     "wickSpinSquaredExpectation of the carried "
                     "quasi-free state.")
      .def_readwrite("spin_variance_read",
                     &BaryonCandidateEvidence::spinVarianceRead,
                     "wickSpinSquaredVariance.  Report-only: a vanishing "
                     "complex variance can come from isotropic "
                     "cancellation on a state that is not an eigenstate, "
                     "so it fills totalJ2Variance and supplies the "
                     "obstruction premise, but sharpSpinEigen is the "
                     "certificate.")
      .def_readwrite("class_variance_reads",
                     &BaryonCandidateEvidence::classVarianceReads,
                     "Var(J^2) of every candidate of the accepted "
                     "covariance-only class; empty/uncertified = the class "
                     "was not swept, so a variance failure is an unknown, "
                     "never an obstruction.")
      .def_readwrite("total_space_j2", &BaryonCandidateEvidence::totalSpaceJ2,
                     "The dense ExchangeHolonomy.total_j_squared "
                     "oracle, consulted only when the Wick "
                     "expectation is absent; it never supplies a variance.")
      .def_readwrite("scale_samples", &BaryonCandidateEvidence::scaleSamples,
                     "Refinement-window ScaleProfileSamples.")
      .def_readwrite("persistence_lifetime",
                     &BaryonCandidateEvidence::persistenceLifetime,
                     "Lifetime of the bound component (report-only).")
      .def_readwrite("lifetime_transports",
                     &BaryonCandidateEvidence::lifetimeTransports,
                     "Composite transports (report-only).")
      .def_readwrite("crossing_mass", &BaryonCandidateEvidence::crossingMass,
                     "The world-tube crossing mass for this "
                     "candidate.  None = the crossing-readouts gate passes "
                     "vacuously (applicable-gated like spin-lift); supplied, "
                     "it is enforced together with crossingBaryon.")
      .def_readwrite("crossing_baryon",
                     &BaryonCandidateEvidence::crossingBaryon,
                     "The coherent one-third baryon sum for the same "
                     "candidate and level.  Must travel with crossingMass: a "
                     "half bundle fails the gate by name rather than grading "
                     "half a certificate.");

  py::class_<BaryonRead>(m, "BaryonRead",
      R"doc(The three-quark baryon read and complete proton certificate.
classification is one of "no-baryon", "baryon-candidate",
"certified-proton", or "quasi-free-sharp-spin-obstruction".

failedCertificates vocabulary, the two structural gates first (a
failure of either is "no-baryon"): "constituent-quarks",
"bound-supercomponent"; then the proton gates: "color-singlet",
"color-flux-zero", "baryon-flux-unit", "composite-parity-odd",
"flavor-uud", "electric-flux-unit", "spin-expectation", "sharp-spin",
"odd-monopole", "projective-cocycle", "spin-lift", "finite-radius",
"profile-stability", "crossing-readouts".

Unknown values are None/NaN/0-sign, never zero-filled; physicalMass is
always None.)doc")
      .def(py::init<>())
      .def_readonly("quarks", &BaryonRead::quarks,
                    "The three constituents' component ids, in evidence order.")
      .def_readonly("bound_component", &BaryonRead::boundComponent)
      .def_readonly("color_gram_determinant",
                    &BaryonRead::colorGramDeterminant,
                    "det(C^dag C) = |det C|^2; NaN = no color evidence.")
      .def_readonly("color_flux", &BaryonRead::colorFlux,
                    "The net color flux diagnostic (octet weight of the "
                    "bound object's color bilinear); NaN = unknown.  An "
                    "independent finite-complex diagnostic -- never on its "
                    "own a proof of confinement.")
      .def_readonly("baryon_flux", &BaryonRead::baryonFlux,
                    "B = nu/3 over the three certified windings (+1 for a "
                    "proton); None = unknown, never zero.")
      .def_readonly("electric_flux", &BaryonRead::electricFlux,
                    "Summed certified constituent Gauss fluxes (+1 for a "
                    "proton); None = unknown.")
      .def_readonly("total_j2", &BaryonRead::totalJ2,
                    "Certified total-space <J^2> (3/4 proton, 15/4 Delta); "
                    "None = unknown.")
      .def_readonly("total_j2_variance", &BaryonRead::totalJ2Variance,
                    "Certified Var(J^2); None = unknown, never zero and "
                    "never inferred from the expectation.")
      .def_readonly("rotation_character", &BaryonRead::rotationCharacter,
                    "The Berry-cancelled 2pi character; None = "
                    "uncertified.  Report-only.")
      .def_readonly("monopole_number", &BaryonRead::monopoleNumber,
                    "The monopole number of the U(1) part of the "
                    "connection through the bounding cut; None = "
                    "uncertified.")
      .def_readonly("classification", &BaryonRead::classification)
      .def_readonly("persistence", &BaryonRead::persistence)
      .def_readonly("failed_certificates", &BaryonRead::failedCertificates)
      .def_readonly("color_wedge", &BaryonRead::colorWedge,
                    "S_ABC = det[c_A c_B c_C], built once.  A constituent "
                    "transposition flips this sign and leaves "
                    "colorGramDeterminant invariant.")
      .def_readonly("total_winding", &BaryonRead::totalWinding,
                    "nu = nu_A + nu_B + nu_C (3 for a proton); None = "
                    "unknown.")
      .def_readonly("exterior_parity", &BaryonRead::exteriorParity,
                    "Exact graded product of the constituent parities; "
                    "0 = unknown.")
      .def_readonly("flavor_pattern", &BaryonRead::flavorPattern,
                    "Certified isospin occupation pattern in canonical "
                    "order ('uud', ...); '' = unknown.")
      .def_readonly("total_isospin", &BaryonRead::totalIsospin)
      .def_readonly("rotation_character_sign",
                    &BaryonRead::rotationCharacterSign)
      .def_readonly("exchange_character", &BaryonRead::exchangeCharacter,
                    "The Berry-cancelled exchange character; None "
                    "unless a certified, correctly tagged exchange read "
                    "was supplied.  Report-only.")
      .def_readonly("spin_statistics_ratio", &BaryonRead::spinStatisticsRatio,
                    "chi(exchange) * chi(2pi)^-1 (+1 on a spin-1/2 "
                    "fixture, each factor separately near -1); None unless "
                    "both channels certified.  Report-only.")
      .def_readonly("spin_lift_applicable", &BaryonRead::spinLiftApplicable)
      .def_readonly("spin_lift_accepted", &BaryonRead::spinLiftAccepted)
      .def_readonly("odd_monopole", &BaryonRead::oddMonopole,
                    "Whether the monopole number through the bounding cut "
                    "is odd.")
      .def_readonly("projective_cocycle_nontrivial",
                    &BaryonRead::projectiveCocycleNontrivial,
                    "Whether the cocycle of the rotation group's "
                    "projective action is cohomologically nontrivial, so "
                    "the modes carry spinor representations of the double "
                    "cover.")
      .def_readonly("sharp_spin_right_residual",
                    &BaryonRead::sharpSpinRightResidual,
                    "||(J^2 - 3/4 I)|Psi_R>|| relative to the state norm; "
                    "NaN = no eigen read.")
      .def_readonly("sharp_spin_left_residual",
                    &BaryonRead::sharpSpinLeftResidual,
                    "||<Psi_L|(J^2 - 3/4 I)|| relative to the state norm; "
                    "NaN = no eigen read.")
      .def_readonly("variance_would_accept", &BaryonRead::varianceWouldAccept,
                    "Whether the complex variance alone would have "
                    "accepted the state.  Report-only: true here with "
                    "sharpSpin false means the variance was cancelled "
                    "isotropically on a state that is not an eigenstate.")
      .def_readonly("sharp_spin", &BaryonRead::sharpSpin,
                    "Whether BOTH eigen-equations held on the supplied "
                    "superposition of determinants.")
      .def_readonly("quasi_free_class_swept",
                    &BaryonRead::quasiFreeClassSwept,
                    "Whether the accepted covariance-only class was swept "
                    "-- the premise the obstruction verdict quantifies "
                    "over.")
      .def_readonly("class_variance_floor", &BaryonRead::classVarianceFloor,
                    "min |Var(J^2)| over the swept class; NaN = not swept.")
      .def_readonly("radius", &BaryonRead::radius)
      .def_readonly("radius_finite", &BaryonRead::radiusFinite)
      .def_readonly("spectral_mass", &BaryonRead::spectralMass)
      .def_readonly("radius_ratio", &BaryonRead::radiusRatio)
      .def_readonly("profile_max_deviation",
                    &BaryonRead::profileMaxDeviation)
      .def_readonly("profile_stable", &BaryonRead::profileStable)
      .def_readonly("physical_mass", &BaryonRead::physicalMass,
                    "Always None (see ScaleProfileRead.physical_mass).")
      .def_readonly("crossing_mass_applicable",
                    &BaryonRead::crossingMassApplicable,
                    "False when the caller supplied no world-tube crossing "
                    "evidence; the crossing-readouts gate then passed "
                    "vacuously, exactly like spin-lift.")
      .def_readonly("crossing_mass_value", &BaryonRead::crossingMassValue,
                    "The crossing mass m_x as a difference "
                    "against M0.  Uncalibrated by default: ratio-only, never "
                    "a physical mass.  NaN without crossing evidence.")
      .def_readonly("crossing_baryon_number", &BaryonRead::crossingBaryonNumber,
                    "The coherent one-third crossing sum; None when no "
                    "crossing evidence was supplied (unknown, never zero).")
      .def_readonly("crossing_sign_defects", &BaryonRead::crossingSignDefects,
                    "Tubes whose crossing sign disagreed with their "
                    "determinant-line winding -- a defect signal.")
      .def_readonly("lifetime_overlap", &BaryonRead::lifetimeOverlap)
      .def_readonly("transport_count", &BaryonRead::transportCount)
      .def_readonly("transport_leakage_max",
                    &BaryonRead::transportLeakageMax)
      .def_readonly("confidence", &BaryonRead::confidence,
                    "Passed-fraction of the fifteen certificates; 1.0 "
                    "exactly for a certified proton.")
      .def_readonly("thresholds", &BaryonRead::thresholds)
      .def_readonly("certificate", &BaryonRead::certificate)
      .def("describe", &BaryonRead::describe)
      .def("__repr__", &BaryonRead::describe)
      .def("to_record",
           [](const BaryonRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization.")
      .def_static("from_record",
                  [](const py::handle &record) {
                    return BaryonRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<ClusterSupportProposal> clusterSupportProposal(m, "ClusterSupportProposal",
      R"doc(One proposed cluster support and the proposers that offered it: the support as
level-0 cell ids, whether Newman-Girvan modularity proposed it, whether the degree-zero
band of the covariant operator proposed it, each proposer's index into its own input
list (NO_PROPOSER when it did not propose this support), and the largest Jaccard index
between this support and any support the other proposer offered.)doc");
  clusterSupportProposal.attr("NO_PROPOSER") = ClusterSupportProposal::kNoProposer;
  clusterSupportProposal
      .def_readonly("support", &ClusterSupportProposal::support)
      .def_readonly("modularity", &ClusterSupportProposal::modularity)
      .def_readonly("band", &ClusterSupportProposal::band)
      .def_readonly("modularity_index", &ClusterSupportProposal::modularityIndex)
      .def_readonly("band_index", &ClusterSupportProposal::bandIndex)
      .def_readonly("cross_proposer_overlap", &ClusterSupportProposal::crossProposerOverlap);

  py::class_<ParticleClusters>(m, "ParticleClusters",
      R"doc(The quark/antiquark classifier over persistent modular
spectral components.  Composes the upstream certificates (persistence,
bands and tracking, anchors, transports and determinant windings with
recorded closures, Wick parity and occupation, and the Gauss-flux read)
into QuarkReads; its own claim is the exact boolean combination
(StructureExact given the consumed held certificates).

Certificate name vocabulary (failedCertificates): "persistence",
"localization", "parity-odd", "occupation-one", "color-rank-three",
"anchor", "transport-leakage", "winding", "winding-unit",
"refinement-stability" (the ten core gates), then "flavor-doublet",
"isospin", "gauss-consistency", "ud-identification" (flavor/charge gates
that never veto quark-ness -- they only leave their own fields unknown).

Read-only observable: never calls a solver, never mutates the spacetime,
and no output enters any emergence objective.  No "quark = hole", no
hard-coded u/d labels, no baryon number without determinant-winding
evidence.)doc")
      .def(py::init<ParticleClustersConfig>(),
           py::arg("config") = ParticleClustersConfig{})
      .def("config", &ParticleClusters::config,
           py::return_value_policy::reference_internal)
      .def("classify_quark", &ParticleClusters::classifyQuark,
           py::arg("evidence"),
           "Classify one candidate from its assembled evidence: the ten "
           "core certificates, quark vs antiquark from the determinant-"
           "line orientation, B = nu/3 under the certified winding, and "
           "isospin/charge from their own independent certificates.  "
           "Missing evidence is a named failed certificate, never an "
           "error.")
      .def("classify_quarks", &ParticleClusters::classifyQuarks,
           py::arg("candidates"),
           "classifyQuark over a candidate stream, in input order.")
      .def("classify_quark_cached", &ParticleClusters::classifyQuarkCached,
           py::arg("cache"), py::arg("evidence"),
           "classifyQuark through the AnalyticCache contract (key: "
           "the color band's cell-vertex set; parameter: the evidence "
           "fingerprint).  Cached equals cold.")
      .def("evidence_fingerprint", &ParticleClusters::evidenceFingerprint,
           py::arg("evidence"),
           "Content fingerprint of the decision-relevant evidence and the "
           "thresholds (the cache parameter).")
      .def("conjugate_pair", &ParticleClusters::conjugatePair,
           py::arg("first"), py::arg("second"),
           "Verify pair conservation of a conjugate creation path from "
           "the two endpoint reads; a singular leg leaves the totals "
           "unknown.")
      .def("flavor_doublet_search", &ParticleClusters::flavorDoubletSearch,
           py::arg("frames"),
           "Search the candidate's band enumeration across frames for a "
           "stable transported two-state subclass (certified "
           "continuations, unambiguous, full length).  No dimension is "
           "ever requested; every stable rank is reported.")
      .def("gauss_flux_on_surfaces", &ParticleClusters::gaussFluxOnSurfaces,
           py::arg("st"), py::arg("field_strength"),
           py::arg("enclosed_vertex_sets"), py::arg("electric_only") = true,
           "The Gauss-flux read "
           "(EigenstateSynthesis.gauss_law_charge) on nested enclosing "
           "surfaces, then the consistency combination.  Read-only on the "
           "spacetime.")
      .def("gauss_flux_consistency", &ParticleClusters::gaussFluxConsistency,
           py::arg("fluxes"),
           py::arg("surface_vertex_counts") = std::vector<std::size_t>{},
           py::arg("electric_only") = true,
           "Pure consistency combination over precomputed per-surface "
           "fluxes (the spacetime path delegates here).")
      .def_static("nested_enclosures", &ParticleClusters::nestedEnclosures,
                  py::arg("st"), py::arg("seed_vertex_ids"),
                  py::arg("shells"),
                  "Nested enclosing vertex sets by breadth-first shell "
                  "growth (returns exactly `shells` sets; sets[0] = the "
                  "seed).")
      .def_static("track_candidates", &ParticleClusters::trackCandidates,
                  py::arg("from_candidates"), py::arg("to_candidates"),
                  py::arg("overlap_threshold") = 0.5,
                  "Track candidates across scale/time by their color "
                  "bands (matchFibers delegation).")
      .def_static("propose_supports", &ParticleClusters::proposeSupports,
                  py::arg("modularity_components"), py::arg("band_components"),
                  "The cluster supports both proposers offer, merged: "
                  "Newman-Girvan modularity on the combinatorial "
                  "one-skeleton, which does not see the complex Hodge "
                  "weights, and the degree-zero band of the covariant "
                  "operator, which is nothing but those weights. Two "
                  "supports are one proposal when their cell-id sets are "
                  "equal; the modularity components come first in input "
                  "order, then every band component none of them matched. "
                  "Both proposers only propose, and neither may veto.")
      // ---- even sectors ------------------------------------------
      .def("octet_bilinear_read", &ParticleClusters::octetBilinearRead,
           py::arg("state"), py::arg("color_modes"),
           "The quasi-free traceless-bilinear (octet) read of three "
           "declared color modes of a carried covariance: exact Wick "
           "sums on the covariance layer (no Fock vector); the 1+8 split "
           "is delegated to ColorFiber.  Throws unless exactly three "
           "distinct in-range modes are named.")
      .def("octet_bilinear_read_cached",
           &ParticleClusters::octetBilinearReadCached,
           py::arg("cache"), py::arg("component_vertex_ids"),
           py::arg("state"), py::arg("color_modes"),
           "octetBilinearRead through the AnalyticCache contract "
           "(key: the caller's component vertex set; parameter: the "
           "covariance hash + declared modes + thresholds).  Cached "
           "equals cold; a Gamma change recomputes.")
      .def("octet_fingerprint", &ParticleClusters::octetFingerprint,
           py::arg("state"), py::arg("color_modes"),
           "Content fingerprint of an octet-read request (the cache "
           "parameter).")
      .def("classify_gluon", &ParticleClusters::classifyGluon,
           py::arg("evidence"),
           "Classify one gluon candidate: "
           "certified even parity, a nonzero certified octet excitation "
           "with machine-level octet purity, accepted rank-three "
           "transports, a certified zero total determinant winding (zero "
           "baryon flux as evidence), and persistence.  Missing evidence "
           "is a named failed certificate.")
      .def("classify_meson", &ParticleClusters::classifyMeson,
           py::arg("evidence"),
           "Classify one meson candidate: certified quark + antiquark "
           "(order-insensitive), even composite parity (exact constituent "
           "product), color-singlet pairing, zero total certified "
           "winding/flux.")
      .def("classify_diquark", &ParticleClusters::classifyDiquark,
           py::arg("evidence"),
           "Classify one diquark candidate: two certified quarks, even "
           "composite parity, a certified anti-triplet wedge occupation, "
           "and the preserved constituent baryon flux B = 2/3 (not an "
           "antiquark).")
      .def("bound_supercomponent_search",
           &ParticleClusters::boundSupercomponentSearch,
           py::arg("next_level_components"), py::arg("candidates"),
           "The bound-supercomponent search: one "
           "read per next-level component containing at least one "
           "certified quark candidate; found requires a strictly higher "
           "modular level, exactly three contained certified quark "
           "candidates, full support containment, overlapping "
           "lifetimes, and bounded mutual transports.")
      .def_static("scale_profile_sample",
                  &ParticleClusters::scaleProfileSample, py::arg("ctx"),
                  "One refinement sample of the "
                  "mass-radius battery, read through the context "
                  "exactly as EmergentRadius/EmergentMass read it "
                  "(RegisterContext.interiorHinges).  Read-only.")
      .def("scale_profile", &ParticleClusters::scaleProfile,
           py::arg("samples"),
           "The refinement-window certificate: a finite emergent "
           "radius plus the refinement stability of every dimensionless "
           "channel.  Nothing here is a form factor and no dimensionful "
           "mass is ever emitted.")
      .def("classify_baryon", &ParticleClusters::classifyBaryon,
           py::arg("evidence"),
           "Classify one three-cluster candidate and evaluate the "
           "complete proton certificate.  Returns "
           "'no-baryon', 'baryon-candidate', 'certified-proton', or "
           "'quasi-free-sharp-spin-obstruction' with every failed or "
           "unknown certificate named.")
      .def("classify_bound_supercomponents",
           &ParticleClusters::classifyBoundSupercomponents,
           py::arg("bindings"), py::arg("constituent_reads"),
           py::arg("bound_lifetimes") = std::vector<double>{},
           "classifyBaryon over the boundSupercomponentSearch result: one "
           "BaryonRead per binding that grouped exactly three certified "
           "constituents, in bindings order.  A binding that grouped a "
           "different number emits nothing -- a three-cluster verdict is "
           "never assembled by padding the missing legs.  Only the "
           "binding, the three QuarkReads (quarkIndices indexes "
           "constituentReads) and the bound component's persistence "
           "lifetime travel; the colour columns, the octet flux, the "
           "rotation character, the spin reads, the swept "
           "covariance-only class and the refinement window are left "
           "absent, so each gap is named rather than presumed.");
}
