// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: crossing readouts, the
// cluster register and the cluster lineage. One of the translation units
// that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_crossing_register_lineage(py::module_ &m) {
  // ========================================
  // CrossingReadouts: world-tube crossing readouts
  // ========================================
  py::class_<CrossingReadoutsConfig>(m, "CrossingReadoutsConfig",
      "Analysis parameters of the world-tube crossing readouts, echoed "
      "verbatim on every read.  kappaMass is the one declared mass "
      "calibration; while massCalibrated is False the crossing mass is "
      "reported in uncalibrated units and only ratios are meaningful.")
      .def(py::init<>())
      .def_readwrite("kappa_mass", &CrossingReadoutsConfig::kappaMass)
      .def_readwrite("mass_calibrated", &CrossingReadoutsConfig::massCalibrated)
      .def_readwrite("sign_tolerance", &CrossingReadoutsConfig::signTolerance)
      .def_readwrite("degeneracy_tolerance",
                     &CrossingReadoutsConfig::degeneracyTolerance)
      .def_readwrite("monopole_tolerance",
                     &CrossingReadoutsConfig::monopoleTolerance)
      .def("to_record", [](const CrossingReadoutsConfig &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<TemporalFunctionRead>(m, "TemporalFunctionRead",
      "The complex Lorentzian distance tau from the incoming boundary M0, "
      "with its temporal-function certificate.  tau is intrinsic: it reads "
      "the 1-skeleton and the stored complex edge lengths, never a vertex "
      "coordinate.  `certified` is True only when Re tau strictly increases "
      "along every future-directed causal edge; otherwise every failure is "
      "named in failedCertificates.")
      .def(py::init<>())
      .def_readonly("vertices", &TemporalFunctionRead::vertices)
      .def_readonly("tau", &TemporalFunctionRead::tau)
      .def_readonly("layer", &TemporalFunctionRead::layer)
      .def_readonly("certified", &TemporalFunctionRead::certified)
      .def_readonly("failed_certificates",
                    &TemporalFunctionRead::failedCertificates)
      .def_readonly("min_causal_increment",
                    &TemporalFunctionRead::minCausalIncrement)
      .def_readonly("causal_edge_count", &TemporalFunctionRead::causalEdgeCount)
      .def_readonly("unreachable_count",
                    &TemporalFunctionRead::unreachableCount)
      .def("at", &TemporalFunctionRead::at, py::arg("vertex"),
           "tau of one vertex, or NaN when unknown.")
      .def("to_record", [](const TemporalFunctionRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<WorldTubeInput>(m, "WorldTubeInput",
      "One persistent band tracked across cobordism frames, as the crossing "
      "readouts consume it.  `orientation` is the tube's traversal direction "
      "(+1 future-directed, -1 the reversed tube): reversing it flips "
      "sgn(pi_perp) and sends B = +1/3 to B = -1/3.  Only certified quark "
      "tubes enter the baryon sum; every admissible crossing enters the "
      "crossing mass.")
      .def(py::init<>())
      .def_readwrite("tube_id", &WorldTubeInput::tubeId)
      .def_readwrite("band", &WorldTubeInput::band)
      .def_readwrite("orientation", &WorldTubeInput::orientation)
      .def_readwrite("determinant_winding", &WorldTubeInput::determinantWinding)
      .def_readwrite("certified_quark_tube",
                     &WorldTubeInput::certifiedQuarkTube);

  py::class_<TubeCrossingRead>(m, "TubeCrossingRead",
      "One tube's crossing of one level set.  `perpendicular` is the complex "
      "pi_perp; `sign` is sgn(Re pi_perp) on an admissible crossing and 0 "
      "when unknown (an inadmissible crossing has no sign at all, never a "
      "silent zero).")
      .def(py::init<>())
      .def_readonly("tube_id", &TubeCrossingRead::tubeId)
      .def_readonly("level", &TubeCrossingRead::level)
      .def_readonly("crossing_edges", &TubeCrossingRead::crossingEdges)
      .def_readonly("density", &TubeCrossingRead::density)
      .def_readonly("perpendicular", &TubeCrossingRead::perpendicular)
      .def_readonly("sign", &TubeCrossingRead::sign)
      .def_readonly("admissible", &TubeCrossingRead::admissible)
      .def_readonly("failed_certificates",
                    &TubeCrossingRead::failedCertificates)
      .def("to_record", [](const TubeCrossingRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<CrossingMassRead>(m, "CrossingMassRead",
      "The crossing-mass functional m_x on one level, as the difference "
      "against the same sum at M0.  Never a dimensionful physical mass while "
      "`calibrated` is False.")
      .def(py::init<>())
      .def_readonly("level", &CrossingMassRead::level)
      .def_readonly("crossing_mass", &CrossingMassRead::crossingMass)
      .def_readonly("level_sum", &CrossingMassRead::levelSum)
      .def_readonly("reference_sum", &CrossingMassRead::referenceSum)
      .def_readonly("kappa_mass", &CrossingMassRead::kappaMass)
      .def_readonly("calibrated", &CrossingMassRead::calibrated)
      .def_readonly("units", &CrossingMassRead::units)
      .def_readonly("admissible_crossings",
                    &CrossingMassRead::admissibleCrossings)
      .def_readonly("refused_crossings", &CrossingMassRead::refusedCrossings)
      .def("to_record", [](const CrossingMassRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<BaryonCrossingRead>(m, "BaryonCrossingRead",
      "The coherent one-third sum over certified quark tubes, with the "
      "determinant-line cross-check.  A tube whose crossing sign disagrees "
      "with its certified winding sign is named in signDefects: a defect "
      "signal, reported and never silently resolved.")
      .def(py::init<>())
      .def_readonly("level", &BaryonCrossingRead::level)
      .def_readonly("baryon_number", &BaryonCrossingRead::baryonNumber)
      .def_readonly("level_sum", &BaryonCrossingRead::levelSum)
      .def_readonly("reference_sum", &BaryonCrossingRead::referenceSum)
      .def_readonly("quark_tubes", &BaryonCrossingRead::quarkTubes)
      .def_readonly("sign_defects", &BaryonCrossingRead::signDefects)
      .def_readonly("winding_agreements",
                    &BaryonCrossingRead::windingAgreements)
      .def("to_record", [](const BaryonCrossingRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<ChargePowerProfileRead>(m, "ChargePowerProfileRead",
      "The spectral charge-power profile S(lambda) built from the eigenspace "
      "projectors of the slice Laplacian (basis- and phase-invariant, "
      "degeneracies handled).  An incoherent power -- the analogue of a "
      "structure factor -- and never the electromagnetic form factor.  For a "
      "neutral system the monopole vanishes, the normalized profile refuses "
      "('neutral-system') and the unnormalized power stays reported.")
      .def(py::init<>())
      .def_readonly("level", &ChargePowerProfileRead::level)
      .def_readonly("eigenvalues", &ChargePowerProfileRead::eigenvalues)
      .def_readonly("power", &ChargePowerProfileRead::power)
      .def_readonly("normalized_power", &ChargePowerProfileRead::normalizedPower)
      .def_readonly("monopole", &ChargePowerProfileRead::monopole)
      .def_readonly("normalized", &ChargePowerProfileRead::normalized)
      .def_readonly("failed_certificates",
                    &ChargePowerProfileRead::failedCertificates)
      .def_readonly("slice_nodes", &ChargePowerProfileRead::sliceNodes)
      .def("to_record", [](const ChargePowerProfileRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<ElectromagneticFormFactorRead>(m,
      "ElectromagneticFormFactorRead",
      "The conditional electromagnetic form factor G_E and the charge "
      "radius.  This tree certifies neither a conserved U(1) current nor "
      "momentum-transfer states, so this is a refusal scaffold: `available` "
      "is False and the radius is unavailable with each missing certificate "
      "named.  The spectral charge-power profile is never substituted.")
      .def(py::init<>())
      .def_readonly("available", &ElectromagneticFormFactorRead::available)
      .def_readonly("charge_radius_squared",
                    &ElectromagneticFormFactorRead::chargeRadiusSquared)
      .def_readonly("failed_certificates",
                    &ElectromagneticFormFactorRead::failedCertificates)
      .def_readonly("note", &ElectromagneticFormFactorRead::note)
      .def("to_record", [](const ElectromagneticFormFactorRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<CrossingReadouts>(m, "CrossingReadouts",
      "World-tube crossing readouts: mass, charge, and form factor from "
      "world-tube crossings.  Read-only: no "
      "solver, no facet materialization, no complex rebuild, and nothing "
      "here enters any emergence objective.")
      .def(py::init<>())
      .def_readonly_static("k_schema_version", &CrossingReadouts::kSchemaVersion)
      .def_static("temporal_function", &CrossingReadouts::temporalFunction,
                  py::arg("spacetime"), py::arg("m0_vertices"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "The complex Lorentzian distance tau from M0 with its "
                  "temporal-function certificate.")
      .def_static("band_edge_density",
                  [](const SpectralFiber &band) {
                    py::dict out;
                    for (const auto &entry :
                         CrossingReadouts::bandEdgeDensity(band)) {
                      out[py::make_tuple(entry.first[0], entry.first[1])] =
                          entry.second;
                    }
                    return out;
                  },
                  py::arg("band"),
                  "The band density mu on the 1-skeleton, keyed by the "
                  "endpoint pair in ascending vertex order: the projector "
                  "diagonal of P = Phi Psi^dagger W carried to edges "
                  "(gauge-invariant by left/right cancellation).")
      .def_static("crossing", &CrossingReadouts::crossing, py::arg("tube"),
                  py::arg("temporal"), py::arg("level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "One tube's crossing of the level Re tau = level.")
      .def_static("crossing_mass", &CrossingReadouts::crossingMass,
                  py::arg("tubes"), py::arg("temporal"), py::arg("level"),
                  py::arg("m0_level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "m_x on `level` as the difference against `m0_level`.")
      .def_static("baryon_number", &CrossingReadouts::baryonNumber,
                  py::arg("tubes"), py::arg("temporal"), py::arg("level"),
                  py::arg("m0_level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "B = (1/3) sum sgn(pi_perp) over certified quark tubes, as "
                  "the difference against `m0_level`.")
      .def_static("charge_power_profile",
                  &CrossingReadouts::chargePowerProfile, py::arg("tubes"),
                  py::arg("temporal"), py::arg("level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "The spectral charge-power profile on `level`.")
      .def_static("form_factor", &CrossingReadouts::formFactor,
                  py::arg("profile"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "The conditional electromagnetic form factor: a refusal "
                  "scaffold naming the certificates this tree lacks.")
      .def_static("overlay_record",
                  [](const std::vector<WorldTubeInput> &tubes,
                     const TemporalFunctionRead &temporal, double level,
                     double m0Level, const CrossingReadoutsConfig &cfg) {
                    return recordToPython(CrossingReadouts::overlayRecord(
                        tubes, temporal, level, m0Level, cfg));
                  },
                  py::arg("tubes"), py::arg("temporal"), py::arg("level"),
                  py::arg("m0_level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "Every readout on one level as the versioned overlay "
                  "block.");

  // ── the register carried by a certified cluster ───────────────
  py::class_<RegisterConjunct>(m, "RegisterConjunct",
      "The fiber-acceptance conjuncts, named.  Six in the whitepaper's list; "
      "eight names here, because the contour conjunct is decided on three "
      "distinct measurements (band gap, contour resolvent, Lorentzian "
      "rotation).  Reference these constants rather than retyping the "
      "strings: a mis-spelled literal produces a name no consumer matches.")
      .def_property_readonly_static("CLUSTER_SUPPORT",
          [](py::object) { return RegisterConjunct::kClusterSupport; })
      .def_property_readonly_static("LOCALIZED_PROJECTOR",
          [](py::object) { return RegisterConjunct::kLocalizedProjector; })
      .def_property_readonly_static("BAND_GAP",
          [](py::object) { return RegisterConjunct::kBandGap; })
      .def_property_readonly_static("NEIGHBOUR_OVERLAP",
          [](py::object) { return RegisterConjunct::kNeighbourOverlap; })
      .def_property_readonly_static("FRAME_LIFETIME",
          [](py::object) { return RegisterConjunct::kFrameLifetime; })
      .def_property_readonly_static("TRANSPORT_LEAKAGE",
          [](py::object) { return RegisterConjunct::kTransportLeakage; })
      .def_property_readonly_static("CONTOUR_RESOLVENT",
          [](py::object) { return RegisterConjunct::kContourResolvent; })
      .def_property_readonly_static("LORENTZIAN_ROTATION",
          [](py::object) { return RegisterConjunct::kLorentzianRotation; });

  py::class_<RegisterUnmeasured>(m, "RegisterUnmeasured",
      "Why a conjunct could not be decided, as distinct from being decided "
      "against.  An unmeasured quantity is not a failed one, and neither is "
      "ever encoded as a zero.")
      .def_property_readonly_static("NO_BAND",
          [](py::object) { return RegisterUnmeasured::kNoBand; })
      .def_property_readonly_static("LOCALIZATION_UNMEASURED",
          [](py::object) { return RegisterUnmeasured::kLocalizationUnmeasured; })
      .def_property_readonly_static("BAND_GAP_UNKNOWN",
          [](py::object) { return RegisterUnmeasured::kBandGapUnknown; })
      .def_property_readonly_static("NO_FRAME_TRACK",
          [](py::object) { return RegisterUnmeasured::kNoFrameTrack; })
      .def_property_readonly_static("NO_TRANSPORT",
          [](py::object) { return RegisterUnmeasured::kNoTransport; })
      .def_property_readonly_static("SUPPORT_UNREADABLE",
          [](py::object) { return RegisterUnmeasured::kSupportUnreadable; })
      .def_property_readonly_static("NO_CONTOUR",
          [](py::object) { return RegisterUnmeasured::kNoContour; })
      .def_property_readonly_static("RESOLVENT_UNMEASURED",
          [](py::object) { return RegisterUnmeasured::kResolventUnmeasured; })
      .def_property_readonly_static("ROTATION_UNMEASURED",
          [](py::object) { return RegisterUnmeasured::kRotationUnmeasured; });

  py::class_<ClusterRegisterConfig>(m, "ClusterRegisterConfig",
      "Thresholds the register is accepted under.  Analysis parameters "
      "only: none of them selects which bands or clusters exist.")
      .def(py::init<>())
      .def_readwrite("min_neighbour_overlap",
                     &ClusterRegisterConfig::minNeighbourOverlap)
      .def_readwrite("min_frame_lifetime",
                     &ClusterRegisterConfig::minFrameLifetime)
      .def_readwrite("max_transport_leakage",
                     &ClusterRegisterConfig::maxTransportLeakage)
      .def_readwrite("max_resolvent_bound",
                     &ClusterRegisterConfig::maxResolventBound,
                     "Cap on the Riesz resolvent bound of the contour the band "
                     "was selected by; decided only where a contour was drawn.")
      .def_readwrite("require_contour", &ClusterRegisterConfig::requireContour,
                     "Require a closed complex-plane contour of the band; a "
                     "band with none is then unmeasured, never failed.")
      .def_readwrite("lorentzian", &ClusterRegisterConfig::lorentzian,
                     "DECLARE the complex Lorentzian: the band must then carry "
                     "a reported rotation epsilon_L > 0 on the allowable side. "
                     "Never inferred from a squared length.")
      .def_readwrite("min_allowability_margin",
                     &ClusterRegisterConfig::minAllowabilityMargin);

  py::class_<RegisterRegimeReport>(m, "RegisterRegimeReport",
      "What is reported of the band's metric regime.  A negative signature "
      "is a certificate, never an automatic antiparticle identification.  "
      "Unmeasured values are NaN.")
      .def_readonly("regime", &RegisterRegimeReport::regime)
      .def_readonly("gram_defect", &RegisterRegimeReport::gramDefect)
      .def_readonly("positive_signature",
                    &RegisterRegimeReport::positiveSignature)
      .def_readonly("negative_signature",
                    &RegisterRegimeReport::negativeSignature)
      .def_readonly("neutral_signature",
                    &RegisterRegimeReport::neutralSignature)
      .def_readonly("signature_normalizable",
                    &RegisterRegimeReport::signatureNormalizable)
      .def_readonly("eigen_residual", &RegisterRegimeReport::eigenResidual)
      .def_readonly("left_residual", &RegisterRegimeReport::leftResidual)
      .def_readonly("frame_condition_number",
                    &RegisterRegimeReport::frameConditionNumber);

  py::class_<ClusterRegisterRead>(m, "ClusterRegisterRead",
      "One register read: the cluster it is carried by, the fiber "
      "E_C = Ran Phi_C it is, the six conjuncts as measured, and the "
      "verdict.  Unmeasured values are NaN and unmeasured conjuncts are "
      "named; nothing is zero-filled.")
      .def_readonly("component", &ClusterRegisterRead::component)
      .def_readonly("support", &ClusterRegisterRead::support)
      .def_readonly("degree", &ClusterRegisterRead::degree)
      .def_readonly("rank", &ClusterRegisterRead::rank)
      .def_readonly("band", &ClusterRegisterRead::band)
      .def_readonly("support_connected", &ClusterRegisterRead::supportConnected)
      .def_readonly("support_pieces", &ClusterRegisterRead::supportPieces)
      .def_readonly("localization_excess",
                    &ClusterRegisterRead::localizationExcess)
      .def_readonly("band_gap", &ClusterRegisterRead::bandGap)
      .def_readonly("contour", &ClusterRegisterRead::contour,
                    "The closed contour the band was selected by; empty when "
                    "no contour was drawn.")
      .def_readonly("contour_node_count",
                    &ClusterRegisterRead::contourNodeCount)
      .def_readonly("resolvent_bound", &ClusterRegisterRead::resolventBound,
                    "The Riesz resolvent bound on that contour -- the "
                    "'controlled resolvent' measurement.")
      .def_readonly("allowability_margin",
                    &ClusterRegisterRead::allowabilityMargin,
                    "Kontsevich-Segal allowability margin of the instance the "
                    "band was read on.")
      .def_readonly("lorentzian_epsilon",
                    &ClusterRegisterRead::lorentzianEpsilon,
                    "The reported rotation epsilon_L the band was read at.")
      .def_readonly("neighbour_overlap", &ClusterRegisterRead::neighbourOverlap)
      .def_readonly("frame_lifetime", &ClusterRegisterRead::frameLifetime)
      .def_readonly("transport_leakage",
                    &ClusterRegisterRead::transportLeakage)
      .def_readonly("regime", &ClusterRegisterRead::regime)
      .def_readonly("failed_conjuncts", &ClusterRegisterRead::failedConjuncts)
      .def_readonly("unmeasured", &ClusterRegisterRead::unmeasured)
      .def_readonly("accepted", &ClusterRegisterRead::accepted)
      .def_readonly("certificate", &ClusterRegisterRead::certificate)
      .def_readonly("thresholds", &ClusterRegisterRead::thresholds)
      .def("describe", &ClusterRegisterRead::describe)
      .def("to_record",
           [](const ClusterRegisterRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the JSON-able record of the read "
           "(schema-versioned; unmeasured channels stay NaN).")
      .def_static("from_record",
                  [](const py::handle &record) {
                    return ClusterRegisterRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate from to_record() output; rejects an unknown "
                  "schema_version (ValueError).");

  py::class_<ClusterRegister>(m, "ClusterRegister",
      "Reads the recursive spectral-fiber register: the fiber "
      "E_C = Ran Phi_C of an isolated localized "
      "band on a persistent cluster, accepted under the six-conjunct list.  "
      "Assembles the other observables and derives no spectrum, "
      "transport or clustering of its own.  The support's provenance is "
      "never consulted, so no proposer can veto a certified fiber.  "
      "Read-only; nothing here enters any emergence objective and no hole "
      "is required or consulted.")
      .def(py::init<ClusterRegisterConfig>(),
           py::arg("cfg") = ClusterRegisterConfig{})
      .def_property_readonly("config", &ClusterRegister::config)
      .def("read", &ClusterRegister::read, py::arg("st"), py::arg("support"),
           py::arg("band"), py::arg("track"), py::arg("external_transports"),
           py::arg("component") = ComponentId{},
           "Read the register of one cluster.  An absent track leaves the "
           "lifetime and overlap conjuncts unmeasured, never satisfied; an "
           "empty transport list likewise leaves leakage unmeasured rather "
           "than small.")
      .def_static("support_connectivity", &ClusterRegister::supportConnectivity,
                  py::arg("st"), py::arg("support"),
                  "Whether the induced one-skeleton on the support is "
                  "connected, and in how many pieces.");

  // ---- the lineage number on the mapping cylinder with a cooriented cut ----

  py::class_<LevelComplex>(m, "LevelComplex",
      "One level of an interaction history: the declared cells of K_l in the "
      "level's own vertex numbering, and the level's vertex count.")
      .def(py::init<>())
      .def(py::init([](std::vector<std::vector<std::uint64_t>> cells, std::size_t vertices) {
             LevelComplex level;
             level.cells = std::move(cells);
             level.vertices = vertices;
             return level;
           }),
           py::arg("cells"), py::arg("vertices"))
      .def_readwrite("cells", &LevelComplex::cells)
      .def_readwrite("vertices", &LevelComplex::vertices);

  py::class_<InteractionCobordism>(m, "InteractionCobordism",
      "The interaction cobordism W: the mapping cylinder of the reduction map "
      "from each level onto the next level's response vertices, with the next "
      "level's cells attached on the outgoing end, concatenated over the "
      "history's steps.")
      .def_readonly("complex", &InteractionCobordism::complex)
      .def_readonly("cells", &InteractionCobordism::cells)
      .def_readonly("vertex_offsets", &InteractionCobordism::vertexOffsets)
      .def_readonly("levels", &InteractionCobordism::levels)
      .def_readonly("level_of", &InteractionCobordism::levelOf)
      .def_readonly("response_of", &InteractionCobordism::responseOf)
      .def_readonly("edges", &InteractionCobordism::edges)
      .def("incoming_vertices", &InteractionCobordism::incomingVertices,
           "The vertices of the incoming boundary: every vertex of level 0.")
      .def("outgoing_vertices", &InteractionCobordism::outgoingVertices,
           "The vertices of the outgoing boundary: every vertex of the last level.")
      .def("edge_index", &InteractionCobordism::edgeIndex, py::arg("a"), py::arg("b"),
           "The canonical C_1(W) index of the edge on the two vertices, or -1 when the pair is "
           "not an edge of W.")
      .def("fiber_edges", &InteractionCobordism::fiberEdges,
           "The canonical C_1(W) indices of the fiber edges, the only timelike edges of W.")
      .def("to_record", [](const InteractionCobordism &self) {
             return recordToPython(self.toRecord());
           });

  py::class_<CoorientedCut>(m, "CoorientedCut",
      "A cooriented separating cut Sigma, carried by the 0-cochain that is 0 on "
      "the incoming side and 1 on the outgoing side. Sigma is the "
      "codimension-one cycle dual to that cochain's coboundary, closed because "
      "a coboundary is a cocycle and separating because every edge with "
      "endpoints on opposite sides is a crossing edge.")
      .def_readonly("side", &CoorientedCut::side)
      .def_readonly("crossing_edges", &CoorientedCut::crossingEdges)
      .def_readonly("crossing_signs", &CoorientedCut::crossingSigns)
      .def_readonly("separates", &CoorientedCut::separates)
      .def_readonly("failed_certificates", &CoorientedCut::failedCertificates)
      .def("to_record",
           [](const CoorientedCut &self) { return recordToPython(self.toRecord()); });

  py::class_<Lineage>(m, "Lineage",
      "An oriented cluster lineage: an integral one-chain of W relative to its "
      "boundary, one coefficient per 1-simplex of W, together with the fermion "
      "number n_Q the lineage carries.")
      .def_readonly("cluster_id", &Lineage::clusterId)
      .def_readonly("coefficients", &Lineage::coefficients)
      .def_readwrite("fermion_number", &Lineage::fermionNumber)
      .def("to_record", [](const Lineage &self) { return recordToPython(self.toRecord()); });

  py::class_<LineageNumberRead>(m, "LineageNumberRead",
      "The reading of one lineage against one cut: N_Q, whether the cut "
      "separates, whether the lineage is a relative cycle, and the interior "
      "vertices at which it has a source.")
      .def_readonly("cluster_id", &LineageNumberRead::clusterId)
      .def_readonly("number", &LineageNumberRead::number)
      .def_readonly("fermion_number", &LineageNumberRead::fermionNumber)
      .def_readonly("relative_cycle", &LineageNumberRead::relativeCycle)
      .def_readonly("interior_sources", &LineageNumberRead::interiorSources)
      .def_readonly("cut_separates", &LineageNumberRead::cutSeparates)
      .def_readonly("failed_certificates", &LineageNumberRead::failedCertificates)
      .def("to_record",
           [](const LineageNumberRead &self) { return recordToPython(self.toRecord()); });

  py::class_<TotalLineageRead>(m, "TotalLineageRead",
      "The reading of a collection of lineages against one cut: "
      "N_q = sum_Q n_Q c_Q . Sigma and B = N_q / 3, the factor 1/3 being an "
      "explicit physical calibration and not a topological theorem.")
      .def_readonly("fermion_number", &TotalLineageRead::fermionNumber)
      .def_readonly("baryon_number", &TotalLineageRead::baryonNumber)
      .def_readonly("per_lineage", &TotalLineageRead::perLineage)
      .def_readonly("failed_certificates", &TotalLineageRead::failedCertificates)
      .def("to_record",
           [](const TotalLineageRead &self) { return recordToPython(self.toRecord()); });

  py::class_<ClusterLineage>(m, "ClusterLineage",
      "The oriented integer of a cluster's history: N_Q = c_Q . Sigma, the "
      "simplicial intersection pairing of an integral one-chain with a "
      "cooriented cut on the interaction cobordism. No sign is taken from a "
      "spectral coordinate, from the connection, from an eigenvalue or from a "
      "density, and no level set of a real part is used.")
      .def_readonly_static("k_schema_version", &ClusterLineage::kSchemaVersion)
      .def_static("history", &ClusterLineage::history, py::arg("levels"), py::arg("reductions"),
                  "The concatenated interaction cobordism of a history of levels.")
      .def_static("mapping_cylinder", &ClusterLineage::mappingCylinder, py::arg("incoming"),
                  py::arg("reduction"), py::arg("outgoing"),
                  "One interaction step's cobordism.")
      .def_static("level_cut", &ClusterLineage::levelCut, py::arg("W"), py::arg("after_level"),
                  "The cut placed between one level and the next.")
      .def_static("cut_from_sides", &ClusterLineage::cutFromSides, py::arg("W"), py::arg("side"),
                  "The cut carried by a declared side per vertex, validated.")
      .def_static("from_fiber_path", &ClusterLineage::fromFiberPath, py::arg("W"),
                  py::arg("start_vertex"), py::arg("fermion_number") = 1,
                  py::arg("cluster_id") = std::string{},
                  "The lineage through the fiber edges out of one starting vertex.")
      .def_static("from_tracked_supports", &ClusterLineage::fromTrackedSupports, py::arg("W"),
                  py::arg("first_level"), py::arg("supports"), py::arg("fermion_number") = 1,
                  py::arg("cluster_id") = std::string{},
                  "The lineage of a cluster whose support is tracked across levels.")
      .def_static("from_vertex_path", &ClusterLineage::fromVertexPath, py::arg("W"), py::arg("path"),
                  py::arg("fermion_number") = 1, py::arg("cluster_id") = std::string{},
                  "The lineage along a declared vertex path of W.")
      .def_static("reversed", &ClusterLineage::reversed, py::arg("lineage"),
                  "The same cluster history traversed in the opposite direction.")
      .def_static("pair_surface_boundary", &ClusterLineage::pairSurfaceBoundary, py::arg("W"),
                  py::arg("surface"), py::arg("fermion_number") = 1,
                  py::arg("cluster_id") = std::string{},
                  "The boundary of an oriented pair surface, as a lineage.")
      .def_static("relative_boundary", &ClusterLineage::relativeBoundary, py::arg("W"),
                  py::arg("lineage"), "The lineage's boundary, one integer per vertex of W.")
      .def_static("intersection_number", &ClusterLineage::intersectionNumber, py::arg("W"),
                  py::arg("cut"), py::arg("lineage"), "N_Q = c_Q . Sigma, the bare integer.")
      .def_static("read", &ClusterLineage::read, py::arg("W"), py::arg("cut"), py::arg("lineage"),
                  "N_Q with its certificates.")
      .def_static("totals", &ClusterLineage::totals, py::arg("W"), py::arg("cut"),
                  py::arg("lineages"), "N_q and B over a collection of lineages.")
      .def_static("order_key", &ClusterLineage::orderKey, py::arg("read"),
                  "The deterministic compilation-order key of one cluster's "
                  "oriented lineage, which quantum.EdgeModeRegistry sorts the "
                  "one-particle modes on.  Lexicographic order on the keys is "
                  "the numeric order of (N_Q, n_Q, clusterId).  An "
                  "uncertified reading raises ValueError, because a lineage "
                  "number another cut would change cannot fix a compilation "
                  "order.");
}
