// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: complex transport, monopole
// and sharp spin reads. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_complex_transport_monopole(py::module_ &m) {
  // ==========================================================================
  // ComplexTransport (#1199): the complex fibre transport of matched Riesz
  // frames.  The general-linear map M_AB = Phi~_A^T T_AB Phi_B retained
  // unprojected, the leakage ||(I - P_A) T_AB P_B|| measured before the
  // restriction to the bands, the reversal by transposition, the Kato parallel
  // transport of an isolated band, and the exchange character
  // chi_F = det(H_ex H_ref^-1) with its modulus reported rather than required.
  // ==========================================================================
  py::class_<ComplexTransportConfig>(m, "ComplexTransportConfig",
      R"doc(Threshold configuration of the complex-transport reads.  A
threshold selects which reads are certified, never which value is reported:
a failed threshold yields an uncertified read carrying the same numbers.)doc")
      .def(py::init<>())
      .def_readwrite("rankTolerance", &ComplexTransportConfig::rankTolerance)
      .def_readwrite("leakageTolerance",
                     &ComplexTransportConfig::leakageTolerance,
                     "Cap on the RELATIVE leakage leak_AB / ||T_AB P_B||_2.")
      .def_readwrite("conditionNumberCap",
                     &ComplexTransportConfig::conditionNumberCap)
      .def_readwrite("isolationFloor", &ComplexTransportConfig::isolationFloor)
      .def_readwrite("requireCertifiedFibers",
                     &ComplexTransportConfig::requireCertifiedFibers)
      .def_readwrite("certificateTolerance",
                     &ComplexTransportConfig::certificateTolerance);

  py::class_<GeneralLinearTransportRead>(m, "GeneralLinearTransportRead",
      R"doc(One complex fibre transport A <- B, reported unprojected: the map
M_AB in GL(r, C), its determinant, its singular data and conditioning, the
leakage measured on the transfer before the restriction to the bands, the
endpoint resolvent bounds and the endpoints' left and right frame residuals.
No polar factor, compact real form or determinant root is taken anywhere.)doc")
      .def(py::init<>())
      .def_readonly("degree", &GeneralLinearTransportRead::degree)
      .def_readonly("rank", &GeneralLinearTransportRead::rank)
      .def_readonly("map", &GeneralLinearTransportRead::map,
                    "M_AB = Phi~_A^T T_AB Phi_B, unprojected.")
      .def_readonly("determinant", &GeneralLinearTransportRead::determinant)
      .def_readonly("singularValues",
                    &GeneralLinearTransportRead::singularValues)
      .def_readonly("numericalRank",
                    &GeneralLinearTransportRead::numericalRank)
      .def_readonly("conditionNumber",
                    &GeneralLinearTransportRead::conditionNumber)
      .def_readonly("minSingularValue",
                    &GeneralLinearTransportRead::minSingularValue)
      .def_readonly("leakage", &GeneralLinearTransportRead::leakage,
                    "||(I - P_A) T_AB P_B||_2.")
      .def_readonly("relativeLeakage",
                    &GeneralLinearTransportRead::relativeLeakage)
      .def_readonly("toIsolation", &GeneralLinearTransportRead::toIsolation)
      .def_readonly("fromIsolation",
                    &GeneralLinearTransportRead::fromIsolation)
      .def_readonly("toResolventBound",
                    &GeneralLinearTransportRead::toResolventBound)
      .def_readonly("fromResolventBound",
                    &GeneralLinearTransportRead::fromResolventBound)
      .def_readonly("toRightFrameResidual",
                    &GeneralLinearTransportRead::toRightFrameResidual)
      .def_readonly("toLeftFrameResidual",
                    &GeneralLinearTransportRead::toLeftFrameResidual)
      .def_readonly("fromRightFrameResidual",
                    &GeneralLinearTransportRead::fromRightFrameResidual)
      .def_readonly("fromLeftFrameResidual",
                    &GeneralLinearTransportRead::fromLeftFrameResidual)
      .def_readonly("toProjectorNorm",
                    &GeneralLinearTransportRead::toProjectorNorm)
      .def_readonly("fromProjectorNorm",
                    &GeneralLinearTransportRead::fromProjectorNorm)
      .def_readonly("regime", &GeneralLinearTransportRead::regime)
      .def_readonly("invertible", &GeneralLinearTransportRead::invertible)
      .def_readonly("accepted", &GeneralLinearTransportRead::accepted)
      .def_readonly("rejectionReason",
                    &GeneralLinearTransportRead::rejectionReason)
      .def_readonly("certificate", &GeneralLinearTransportRead::certificate)
      .def("describe", &GeneralLinearTransportRead::describe);

  py::enum_<KatoScheme>(m, "KatoScheme",
      R"doc(Which step carries the band from one sampled Riesz projector to the
next, and what that step is exact for.  Naming the scheme is required rather
than inferred, so that no approximation is ever applied silently.)doc")
      .value("DirectRotation", KatoScheme::DirectRotation,
             "The direct rotation: the exact solution of the Kato equation "
             "along the geodesic joining the two projectors, for orthogonal "
             "projectors closer than one in operator norm.")
      .value("Intertwiner", KatoScheme::Intertwiner,
             "R = P1 P0 + (I - P1)(I - P0): exact intertwining for any pair "
             "of idempotents, orthogonal or oblique, and unnormalized.")
      .value("ExponentialGenerator", KatoScheme::ExponentialGenerator,
             "exp([P1, P0]): the Kato equation's own exponential step, whose "
             "intertwining residual is third order in the step and is "
             "reported.");

  py::class_<KatoTransportRead>(m, "KatoTransportRead",
      R"doc(The parallel transport of an isolated band along a sampled path of
Riesz projectors: the composed transport, the per-step transports, and the
residuals that certify them -- the intertwining residual K_t P_t = P_{t+1} K_t,
the idempotency of the supplied projectors, the rank held along the path and
the coarseness of the sampling.)doc")
      .def(py::init<>())
      .def_readonly("transport", &KatoTransportRead::transport)
      .def_readonly("stepTransports", &KatoTransportRead::stepTransports)
      .def_readonly("scheme", &KatoTransportRead::scheme)
      .def_readonly("steps", &KatoTransportRead::steps)
      .def_readonly("dimension", &KatoTransportRead::dimension)
      .def_readonly("rank", &KatoTransportRead::rank)
      .def_readonly("rankDefect", &KatoTransportRead::rankDefect)
      .def_readonly("idempotencyResidual",
                    &KatoTransportRead::idempotencyResidual)
      .def_readonly("intertwiningResidual",
                    &KatoTransportRead::intertwiningResidual)
      .def_readonly("composedIntertwiningResidual",
                    &KatoTransportRead::composedIntertwiningResidual)
      .def_readonly("maxProjectorStep", &KatoTransportRead::maxProjectorStep)
      .def_readonly("complete", &KatoTransportRead::complete)
      .def_readonly("invalidReason", &KatoTransportRead::invalidReason)
      .def_readonly("certificate", &KatoTransportRead::certificate);

  py::class_<ExchangeCharacterRead>(m, "ExchangeCharacterRead",
      R"doc(chi_F = det(H_ex H_ref^-1), the interferometric exchange character
of a general-linear exchange holonomy against a matched non-exchanging
reference holonomy.  The complex number is the whole result: its modulus is
reported because a reader wants to see it, not because anything is required of
it, and no phase angle, component sign or unit-modulus projection is taken.)doc")
      .def(py::init<>())
      .def_readonly("character", &ExchangeCharacterRead::character)
      .def_readonly("determinantRatio",
                    &ExchangeCharacterRead::determinantRatio)
      .def_readonly("routeAgreementResidual",
                    &ExchangeCharacterRead::routeAgreementResidual)
      .def_readonly("exchangeDeterminant",
                    &ExchangeCharacterRead::exchangeDeterminant)
      .def_readonly("referenceDeterminant",
                    &ExchangeCharacterRead::referenceDeterminant)
      .def_readonly("modulus", &ExchangeCharacterRead::modulus,
                    "|chi_F|, reported and never required.")
      .def_readonly("distanceToMinusOne",
                    &ExchangeCharacterRead::distanceToMinusOne)
      .def_readonly("distanceToPlusOne",
                    &ExchangeCharacterRead::distanceToPlusOne)
      .def_readonly("rank", &ExchangeCharacterRead::rank)
      .def_readonly("referenceConditionNumber",
                    &ExchangeCharacterRead::referenceConditionNumber)
      .def_readonly("pathLeakage", &ExchangeCharacterRead::pathLeakage)
      .def_readonly("referenceInvertible",
                    &ExchangeCharacterRead::referenceInvertible)
      .def_readonly("certificate", &ExchangeCharacterRead::certificate);

  py::class_<ComplexTransport>(m, "ComplexTransport",
      R"doc(The complex fibre transport of matched Riesz frames: the
general-linear map M_AB, its leakage certificate, its reversal by
transposition, the Kato parallel transport of an isolated band, and the
exchange character of two general-linear holonomies.

The retained observable is the complex matrix itself.  Under independent frame
changes at the two ends it transforms as M_AB -> g_A^-1 M_AB g_B, so a closed
holonomy transforms by conjugation and its power traces, characteristic
polynomial, determinant and conjugacy class are frame-free.  That covariance is
the reason nothing is normalized: a polar factor, a determinant root or a
modulus is a choice of representative and destroys either the determinant
transport or the frame law.)doc")
      .def_static("fiberMap", &ComplexTransport::fiberMap,
                  py::arg("dualFrameTo"), py::arg("transfer"),
                  py::arg("rightFrameFrom"),
                  "M_AB = Phi~_A^T T_AB Phi_B, the bilinear pairing.")
      .def_static("leakage", &ComplexTransport::leakage,
                  py::arg("projectorTo"), py::arg("transfer"),
                  py::arg("projectorFrom"),
                  "||(I - P_A) T_AB P_B||_2, measured before the restriction "
                  "to the bands.")
      .def_static("transport", &ComplexTransport::transport,
                  py::arg("to_fiber"), py::arg("from_fiber"),
                  py::arg("transfer"),
                  py::arg("config") = ComplexTransportConfig{},
                  "The complete transport A <- B of a transfer between two "
                  "bands.")
      .def_static("frameChanged", &ComplexTransport::frameChanged,
                  py::arg("map"), py::arg("frameTo"), py::arg("frameFrom"),
                  "M_AB -> g_A^-1 M_AB g_B.")
      .def_static("reversedTransfer", &ComplexTransport::reversedTransfer,
                  py::arg("transfer"),
                  "T_BA = T_AB^T, the transposition reversal of a "
                  "hopping-defined transfer.")
      .def_static("dualTransport", &ComplexTransport::dualTransport,
                  py::arg("map"),
                  "M^v_AB = M_AB^-T, the branch-free dual transport of an "
                  "anti-cluster, whose determinant is (det M_AB)^-1.")
      .def_static("compose", &ComplexTransport::compose, py::arg("path"),
                  py::arg("rank") = std::size_t{0},
                  "The path transport, first factor applied first.")
      .def_static("holonomy", &ComplexTransport::holonomy, py::arg("links"),
                  "The closed holonomy and its conjugacy invariants.")
      .def_static("katoGenerator", &ComplexTransport::katoGenerator,
                  py::arg("projectorRate"), py::arg("projector"),
                  "[P', P], the right-hand side of the Kato equation.")
      .def_static("katoStep", &ComplexTransport::katoStep,
                  py::arg("fromProjector"), py::arg("toProjector"),
                  py::arg("scheme") = KatoScheme::DirectRotation,
                  "One Kato step between two sampled projectors.")
      .def_static("katoTransport", &ComplexTransport::katoTransport,
                  py::arg("projectors"),
                  py::arg("scheme") = KatoScheme::DirectRotation,
                  py::arg("config") = ComplexTransportConfig{},
                  "The Kato parallel transport of an isolated band along a "
                  "sampled path of Riesz projectors.")
      .def_static("katoTransportOnFibers",
                  &ComplexTransport::katoTransportOnFibers, py::arg("loop"),
                  py::arg("scheme") = KatoScheme::DirectRotation,
                  py::arg("config") = ComplexTransportConfig{},
                  "The same, with the projectors read from a path of bands "
                  "that all carry the same cells.")
      .def_static("bandTransport", &ComplexTransport::bandTransport,
                  py::arg("transport"), py::arg("dualFrameEnd"),
                  py::arg("rightFrameStart"),
                  "k = Phi~_end^T K Phi_start, the general-linear link a "
                  "Kato-transported band contributes to a holonomy.")
      .def_static("exchangeCharacter", &ComplexTransport::exchangeCharacter,
                  py::arg("exchangeHolonomy"), py::arg("referenceHolonomy"),
                  py::arg("pathLeakage"),
                  py::arg("config") = ComplexTransportConfig{},
                  "chi_F = det(H_ex H_ref^-1).")
      .def_static("exchangeCharacterOfPaths",
                  &ComplexTransport::exchangeCharacterOfPaths,
                  py::arg("exchangePath"), py::arg("referencePath"),
                  py::arg("pathLeakage"),
                  py::arg("config") = ComplexTransportConfig{},
                  "chi_F of the two paths' composed holonomies.");

  // ==========================================================================
  // MonopoleSpin (#1196): spin from an odd Dirac monopole.  The monopole
  // number through a closed cut, the projective representation D_k(g) with
  // its cocycle, the spinor bands it protects, and the sharp-spin
  // eigen-equations on a superposition of determinants.
  // ==========================================================================
  py::class_<MonopoleNumberRead>(m, "MonopoleNumberRead",
      R"doc(The monopole number of the connection through a closed cut, read
from the outward face holonomies.  Each face contributes the principal
argument of its holonomy, so a face carrying more than half a turn is not
resolved; integrality_residual measures how far the total sits from an integer
multiple of 2 pi, and a total that is not such a multiple is not a bundle at
all.)doc")
      .def(py::init<>())
      .def_readonly("face_holonomies", &MonopoleNumberRead::faceHolonomies)
      .def_readonly("face_fluxes", &MonopoleNumberRead::faceFluxes)
      .def_readonly("total_flux", &MonopoleNumberRead::totalFlux)
      .def_readonly("monopole_number", &MonopoleNumberRead::monopoleNumber)
      .def_readonly("integrality_residual",
                    &MonopoleNumberRead::integralityResidual)
      .def_readonly("bundle", &MonopoleNumberRead::bundle)
      .def_readonly("odd", &MonopoleNumberRead::odd,
                    "Whether the monopole number is odd -- the condition "
                    "under which the projective class is nontrivial.")
      .def_readonly("branch_margin", &MonopoleNumberRead::branchMargin,
                    "min over faces of (pi - |face flux|).")
      .def_readonly("on_branch_cut", &MonopoleNumberRead::onBranchCut,
                    "Whether some face flux reached the ends of the "
                    "principal interval, where the reported integer rests "
                    "on the stated convention.")
      .def_readonly("unit_modulus_residual",
                    &MonopoleNumberRead::unitModulusResidual)
      .def_readonly("certificate", &MonopoleNumberRead::certificate);

  py::class_<GaugeCompensationRead>(m, "GaugeCompensationRead",
      R"doc(The compensating gauge transformation u_g of one rotation,
normalized to one at the spanning tree's root, and the residual that decides
whether the configuration really is symmetric up to gauge.)doc")
      .def(py::init<>())
      .def_readonly("gauge", &GaugeCompensationRead::gauge)
      .def_readonly("residual", &GaugeCompensationRead::residual)
      .def_readonly("symmetric", &GaugeCompensationRead::symmetric);

  py::class_<CocycleRead>(m, "CocycleRead",
      R"doc(The cocycle varpi of the projective representation and its
cohomology class.  The class is nontrivial exactly when some COMMUTING pair
has a commutator phase varpi(g,h)/varpi(h,g) different from one, that phase
being invariant under every rescaling of the compensating gauges.  On an
odd-monopole tetrahedral support it is -1, the quaternion relation of the
binary tetrahedral group.)doc")
      .def(py::init<>())
      .def_readonly("group_order", &CocycleRead::groupOrder)
      .def_readonly("scalar_residual", &CocycleRead::scalarResidual)
      .def_readonly("values", &CocycleRead::values)
      .def_readonly("max_commutator_deviation",
                    &CocycleRead::maxCommutatorDeviation)
      .def_readonly("commutator_phase", &CocycleRead::commutatorPhase)
      .def_readonly("commutator_pair", &CocycleRead::commutatorPair)
      .def_readonly("nontrivial", &CocycleRead::nontrivial)
      .def_readonly("certificate", &CocycleRead::certificate);

  py::class_<SpinorBandRead>(m, "SpinorBandRead",
      R"doc(One symmetry-protected band of a rotation-invariant operator: its
eigenvalue and rank, the measured invariance under the projective action, the
irreducibility score (exactly one for an irreducible projective
representation with this cocycle), the distance from the coexact sector, and
whether it is a spinor doublet.)doc")
      .def(py::init<>())
      .def_readonly("eigenvalue", &SpinorBandRead::eigenvalue)
      .def_readonly("dimension", &SpinorBandRead::dimension)
      .def_readonly("invariance_residual",
                    &SpinorBandRead::invarianceResidual)
      .def_readonly("irreducibility_score",
                    &SpinorBandRead::irreducibilityScore)
      .def_readonly("coexact_residual", &SpinorBandRead::coexactResidual)
      .def_readonly("spinor_doublet", &SpinorBandRead::spinorDoublet)
      .def_readonly("coexact", &SpinorBandRead::coexact);

  py::class_<MonopoleSpinRead>(m, "MonopoleSpinRead",
      R"doc(What a classifier needs from an odd-monopole support in one
record: the monopole number of the bounding cut, the cocycle of the rotation
group's projective action, the symmetry-protected bands of the
rotation-averaged edge Laplacian, and the j = 1/2 doublet among them -- the
rank-two, invariant, irreducible, spinorial band lying in the coexact
sector.)doc")
      .def(py::init<>())
      .def_readonly("monopole", &MonopoleSpinRead::monopole)
      .def_readonly("cocycle", &MonopoleSpinRead::cocycle)
      .def_readonly("bands", &MonopoleSpinRead::bands)
      .def_readonly("doublet_index", &MonopoleSpinRead::doubletIndex)
      .def_readonly("half_integer_doublet",
                    &MonopoleSpinRead::halfIntegerDoublet)
      .def_readonly("certificate", &MonopoleSpinRead::certificate);

  py::class_<MonopoleSupport>(m, "MonopoleSupport",
      R"doc(A symmetric cluster bounded by a closed cut, carried as its
vertices, its oriented edges (smaller vertex first), its outward-oriented
faces and one U(1) connection value per edge.  It reads the monopole number
from the outward face holonomies, builds the twisted coboundary and
Laplacians, solves for the compensating gauge transformation of each rotation,
assembles D_0(g) and D_1(g), measures the cocycle and decides its class, and
reads the symmetry-protected bands of a rotation-invariant operator.)doc")
      .def(py::init<std::size_t,
                    std::vector<std::array<std::size_t, 2>>,
                    std::vector<std::array<std::size_t, 3>>,
                    std::vector<std::complex<double>>>(),
           py::arg("vertexCount"), py::arg("edges"), py::arg("faces"),
           py::arg("connection"))
      .def_static("tetrahedron", &MonopoleSupport::tetrahedron,
                  py::arg("monopoleNumber"),
                  "The tetrahedron with the symmetric monopole connection "
                  "whose every outward face holonomy is exp(2 pi i mu / 4).")
      .def_static("tetrahedralRotations",
                  &MonopoleSupport::tetrahedralRotations,
                  "The twelve rotations of the tetrahedron (T = A_4), the "
                  "identity first.")
      .def_static("u1Part", &MonopoleSupport::u1Part, py::arg("connection"),
                  "U_e / |U_e| -- the explicit way to bring an unrestricted "
                  "connection into this kernel's domain.")
      .def_property_readonly("vertex_count", &MonopoleSupport::vertexCount)
      .def_property_readonly("edges", &MonopoleSupport::edges)
      .def_property_readonly("faces", &MonopoleSupport::faces)
      .def_property_readonly("connection", &MonopoleSupport::connection)
      .def("transport", &MonopoleSupport::transport, py::arg("x"),
           py::arg("y"))
      .def("monopoleNumber", &MonopoleSupport::monopoleNumber,
           py::arg("tolerance") = 1e-9)
      .def("twistedCoboundary", &MonopoleSupport::twistedCoboundary)
      .def("twistedFaceCoboundary",
           &MonopoleSupport::twistedFaceCoboundary)
      .def("vertexLaplacian", &MonopoleSupport::vertexLaplacian)
      .def("edgeLaplacian", &MonopoleSupport::edgeLaplacian)
      .def("coexactProjector", &MonopoleSupport::coexactProjector,
           py::arg("tolerance") = 1e-9)
      .def("gaugeCompensation", &MonopoleSupport::gaugeCompensation,
           py::arg("rotation"), py::arg("tolerance") = 1e-9)
      .def("vertexRepresentation", &MonopoleSupport::vertexRepresentation,
           py::arg("rotation"), "D_0(g) on vertex cochains.")
      .def("edgeRepresentation", &MonopoleSupport::edgeRepresentation,
           py::arg("rotation"), "D_1(g) on edge cochains.")
      .def("intertwiningResidual", &MonopoleSupport::intertwiningResidual,
           py::arg("rotation"),
           "||delta_0^U D_0(g) - D_1(g) delta_0^U||_max.")
      .def("cocycle", &MonopoleSupport::cocycle, py::arg("group"),
           py::arg("cochainDegree") = 1, py::arg("tolerance") = 1e-9)
      .def("rotationAveragedEdgeOperator",
           &MonopoleSupport::rotationAveragedEdgeOperator,
           py::arg("edgeOperator"), py::arg("group"))
      .def("spinorBands", &MonopoleSupport::spinorBands,
           py::arg("operatorMatrix"), py::arg("group"),
           py::arg("nontrivialClass"),
           py::arg("degeneracyTolerance") = 1e-7,
           py::arg("tolerance") = 1e-9)
      .def("spinRead", &MonopoleSupport::spinRead, py::arg("group"),
           py::arg("degeneracyTolerance") = 1e-7,
           py::arg("tolerance") = 1e-9,
           "The whole spin read: monopole number, cocycle, bands and the "
           "j = 1/2 doublet among them.");

  py::class_<SharpSpinRead>(m, "SharpSpinRead",
      R"doc(The spin-lift read: the right and left J^2 eigen-equation residuals
under a declared SU(2) action on the modes, the verdict they decide, and the
biorthogonal expectation and complex variance that the whitepaper calls
insufficient on their own.  A finite cluster has no J^2 of its own (WP v18
Section 11.1); the sharp spinor certificate is IsotypicRead, and this read
states the continuum value j an accepted spin lift supplies.
variance_would_accept is true when the variance test alone would have called
the state sharp; when it is true and sharp is false, the variance test has
been decided wrongly by isotropic cancellation.)doc")
      .def(py::init<>())
      .def_readonly("target_eigenvalue", &SharpSpinRead::targetEigenvalue)
      .def_readonly("right_residual", &SharpSpinRead::rightResidual)
      .def_readonly("left_residual", &SharpSpinRead::leftResidual)
      .def_readonly("sharp", &SharpSpinRead::sharp)
      .def_readonly("expectation", &SharpSpinRead::expectation)
      .def_readonly("variance", &SharpSpinRead::variance)
      .def_readonly("variance_would_accept",
                    &SharpSpinRead::varianceWouldAccept)
      .def_readonly("determinant_count", &SharpSpinRead::determinantCount)
      .def_readonly("certificate", &SharpSpinRead::certificate);

  py::enum_<QuarkConditionStatus>(m, "QuarkConditionStatus",
      "The outcome of one v16 quark condition: Passed (every required "
      "certificate measured and held), Failed (some measured certificate did "
      "not hold), NotEvaluable (none failed, but a required certificate was "
      "not measured).")
      .value("Passed", QuarkConditionStatus::Passed)
      .value("Failed", QuarkConditionStatus::Failed)
      .value("NotEvaluable", QuarkConditionStatus::NotEvaluable);

  py::class_<QuarkConditionEvidence>(m, "QuarkConditionEvidence",
      "One certificate offered as evidence for a quark condition: its name, "
      "whether it held (None when not measured), and the measured value or the "
      "reason it was not measured.")
      .def(py::init([](std::string name, std::optional<bool> held,
                       std::string detail) {
             return QuarkConditionEvidence{std::move(name), held,
                                           std::move(detail)};
           }),
           py::arg("name"), py::arg("held") = std::nullopt,
           py::arg("detail") = std::string{})
      .def_readwrite("name", &QuarkConditionEvidence::name)
      .def_readwrite("held", &QuarkConditionEvidence::held)
      .def_readwrite("detail", &QuarkConditionEvidence::detail);

  py::class_<QuarkConditionRead>(m, "QuarkConditionRead",
      "The read of one v16 quark condition.")
      .def_readonly("number", &QuarkConditionRead::number)
      .def_readonly("name", &QuarkConditionRead::name)
      .def_readonly("statement", &QuarkConditionRead::statement)
      .def_readonly("status", &QuarkConditionRead::status)
      .def_readonly("evidence", &QuarkConditionRead::evidence)
      .def_readonly("missing", &QuarkConditionRead::missing)
      .def_readonly("failing", &QuarkConditionRead::failing);

  py::class_<QuarkVerdict>(m, "QuarkVerdict",
      "The v16 quark verdict over all seven Section 10 conditions.")
      .def_readonly("conditions", &QuarkVerdict::conditions)
      .def_readonly("certified", &QuarkVerdict::certified)
      .def_readonly("failed", &QuarkVerdict::failed)
      .def_readonly("not_evaluable", &QuarkVerdict::notEvaluable);

  py::class_<QuarkConditions>(m, "QuarkConditions",
      "The seven quark conditions of Section 10 of the whitepaper, v16, as a "
      "verdict over caller-measured certificates. A condition fails when any "
      "supplied certificate did not hold, passes when every required one was "
      "supplied and held, and is otherwise not evaluable. It implements the "
      "v16 reading (a per-sheet base band tensored with the sheet space, a "
      "projective anchor, no flavour or charge condition), which "
      "ParticleClusters.classifyQuark, the v15 reading, does not.")
      .def_property_readonly_static("kConditionCount", [](py::object) {
        return QuarkConditions::kConditionCount;
      })
      .def_static("condition_names", &QuarkConditions::conditionNames)
      .def_static("statement", &QuarkConditions::statement, py::arg("number"))
      .def_static("required_evidence", &QuarkConditions::requiredEvidence,
                  py::arg("number"))
      .def_static("evaluate_condition", &QuarkConditions::evaluateCondition,
                  py::arg("number"), py::arg("evidence"))
      .def_static("evaluate", &QuarkConditions::evaluate, py::arg("evidence"),
                  "The verdict; evidence[i] is the list for condition i + 1.");

  py::class_<IsotypicRead>(m, "IsotypicRead",
      R"doc(The sharpness read of a spinor readout against one isotypic
component of the cluster's symmetry action (WP v18 Sections 11.1 and 14):
the residuals of (I - P)|Psi_R> = 0 and <Psi_L|(I - P) = 0, the verdict they
decide, the projector's rank and measured idempotency, and the bilinear
weight <Psi_L|P|Psi_R> / <Psi_L|Psi_R>, which is only a matrix-element
identity and gates nothing.)doc")
      .def(py::init<>())
      .def_readonly("type", &IsotypicRead::type)
      .def_readonly("rank", &IsotypicRead::rank)
      .def_readonly("idempotency_residual", &IsotypicRead::idempotencyResidual)
      .def_readonly("right_residual", &IsotypicRead::rightResidual)
      .def_readonly("left_residual", &IsotypicRead::leftResidual)
      .def_readonly("sharp", &IsotypicRead::sharp)
      .def_readonly("weight", &IsotypicRead::weight)
      .def_readonly("certificate", &IsotypicRead::certificate);

  py::class_<SharpSpin>(m, "SharpSpin",
      R"doc(The sharp spinor readout of WP v18 (Sections 11.1 and 14): the
isotypic projector P_rho of the cluster's symmetry action on an n-particle
sector and the projector equations (I - P_rho)|Psi_R> = 0 and
<Psi_L|(I - P_rho) = 0 on the bounded superposition of determinants selected
by the isolating interaction; beside it the spin-lift read, the J^2
eigen-equations under a declared SU(2) action on the modes.  J^2 is
polynomial in the exterior generators, so its action is applied mode-pair by
mode-pair; the dense Fock matrix is materialized only for fixtures and only
below the declared mode limit.)doc")
      .def_property_readonly_static("kMaxDenseModes",
          [](py::object) { return SharpSpin::kMaxDenseModes; })
      .def_property_readonly_static("kMaxStateModes",
          [](py::object) { return SharpSpin::kMaxStateModes; })
      .def_static("determinant", &SharpSpin::determinant,
                  py::arg("occupiedModes"), py::arg("modeCount"))
      .def_static("determinantSuperposition",
                  &SharpSpin::determinantSuperposition,
                  py::arg("occupations"), py::arg("amplitudes"),
                  py::arg("modeCount"))
      .def_static("applyTotalSpinSquared", &SharpSpin::applyTotalSpinSquared,
                  py::arg("spinMatrices"), py::arg("state"))
      .def_static("totalSpinSquaredMatrix",
                  &SharpSpin::totalSpinSquaredMatrix,
                  py::arg("spinMatrices"))
      .def_static("read", &SharpSpin::read, py::arg("spinMatrices"),
                  py::arg("rightState"), py::arg("leftState"),
                  py::arg("targetEigenvalue") = 0.75,
                  py::arg("tolerance") = 1e-9)
      .def_static("doubletSpinMatrices", &SharpSpin::doubletSpinMatrices,
                  py::arg("carrierCount"),
                  "J_a = I (x) sigma_a / 2 on carrierCount distinguishable "
                  "spin-one-half carriers, mode 2c + s being spin state s of "
                  "carrier c.")
      .def_property_readonly_static("kMaxSectorPatterns",
          [](py::object) { return SharpSpin::kMaxSectorPatterns; })
      .def_static("sectorPatterns", &SharpSpin::sectorPatterns,
                  py::arg("modeCount"), py::arg("particles"),
                  "The n-particle occupation patterns: ascending mode tuples "
                  "in lexicographic order, the basis of the sector matrices.")
      .def_static("exteriorPowerMatrix", &SharpSpin::exteriorPowerMatrix,
                  py::arg("oneParticle"), py::arg("particles"),
                  "Lambda^n D on the n-particle sector: entry (J, I) is the "
                  "minor det D[J, I] over the sector patterns.")
      .def_static("sectorComponent", &SharpSpin::sectorComponent,
                  py::arg("state"), py::arg("particles"),
                  "The n-particle component of a Fock vector over the sector "
                  "patterns.")
      .def_static("fockVector", &SharpSpin::fockVector, py::arg("sector"),
                  py::arg("modeCount"), py::arg("particles"),
                  "A vector over the sector patterns as a Fock vector.")
      .def_static("isotypicProjector", &SharpSpin::isotypicProjector,
                  py::arg("maps"), py::arg("characters"), py::arg("dimension"),
                  py::arg("particles"),
                  "P_rho = (dim rho / |G|) sum_g conj(chi_rho(g)) Lambda^n "
                  "D(g) on the n-particle sector, from every element's "
                  "one-particle map and rho's character on it.")
      .def_static("isotypicRead", &SharpSpin::isotypicRead,
                  py::arg("projector"), py::arg("rightState"),
                  py::arg("leftState"), py::arg("type"),
                  py::arg("tolerance") = 1e-9,
                  "The sharp spinor certificate: (I - P)|Psi_R> = 0 and "
                  "<Psi_L|(I - P) = 0 against the declared tolerance.");
}
