// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the Ward flux, intrinsic
// response and bound-state pole. One of the translation units that
// Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_cobordism_ward_flux_bound_state(py::module_ &m) {
  // ── Section 13.4/13.5: the Ward flux and the intrinsic response ────────

  py::class_<WardFluxConfig>(m, "WardFluxConfig",
      "Every threshold of the Ward-flux read.")
      .def(py::init<>())
      .def_readwrite("divergence_tolerance", &WardFluxConfig::divergenceTolerance,
                     "|(d j)_x| at or below this is a vanishing divergence.")
      .def_readwrite("integrality_tolerance",
                     &WardFluxConfig::integralityTolerance,
                     "|phi_j(Sigma) - phi_j^ref(Sigma) - n| at or below this "
                     "lets the excess of the flux over the reference be read "
                     "as the integer quark number N_q.")
      .def_readwrite("imaginary_tolerance", &WardFluxConfig::imaginaryTolerance,
                     "|Im(phi_j(Sigma) - phi_j^ref(Sigma))| must be at or "
                     "below this for the excess to be read as an integer.")
      .def_readwrite("reference_flux", &WardFluxConfig::referenceFlux,
                     "phi_j^ref(Sigma): the flux of the matched reference "
                     "state on d_in W through the same cut. The quark number "
                     "is the excess of the flux over it (WP v18 Section "
                     "13.4). None, the default, is the empty reference, whose "
                     "flux is zero.")
      .def_readwrite("charge_tolerance", &WardFluxConfig::chargeTolerance,
                     "|phi_j(Sigma) - Q_in| at or below this counts as the "
                     "flux agreeing with the incoming boundary charge.");

  py::class_<WardFluxRead>(m, "WardFluxRead",
      "The flux phi_j(Sigma) of the complex Ward current through one "
      "cooriented cut of the interaction cobordism W, with every certificate "
      "Section 13.4 attaches to it.")
      .def_readonly("crossing_edges", &WardFluxRead::crossingEdges,
                    "The canonical C_1(W) indices of the edges the cut "
                    "crosses.")
      .def_readonly("crossing_signs", &WardFluxRead::crossingSigns,
                    "The coorientation u(b) - u(a) of each crossing edge.")
      .def_readonly("crossing_current", &WardFluxRead::crossingCurrent,
                    "The Ward current on each crossing edge.")
      .def_readonly("flux", &WardFluxRead::flux,
                    "phi_j(Sigma) = <delta u, j>, the flux: the fermion "
                    "number enclosed by the cut, every occupied mode counted. "
                    "Complex, never projected onto a real part.")
      .def_readonly("reference_flux", &WardFluxRead::referenceFlux,
                    "phi_j^ref(Sigma), the declared flux of the matched "
                    "reference state through the same cut; zero for the "
                    "empty reference.")
      .def_readonly("reference_declared", &WardFluxRead::referenceDeclared,
                    "Whether a reference flux was declared; False means the "
                    "empty reference and excess_flux equals flux.")
      .def_readonly("excess_flux", &WardFluxRead::excessFlux,
                    "phi_j(Sigma) - phi_j^ref(Sigma): the coherent excess of "
                    "the flux over the matched reference's, the quark number "
                    "of WP v18 Section 13.4 as the complex number it is.")
      .def_readonly("incoming_side_divergence",
                    &WardFluxRead::incomingSideDivergence,
                    "The divergence summed over the cut's incoming side, "
                    "which the divergence theorem makes minus the flux.")
      .def_readonly("divergence_theorem_residual",
                    &WardFluxRead::divergenceTheoremResidual,
                    "The residual of that identity.")
      .def_readonly("incoming_boundary_divergence",
                    &WardFluxRead::incomingBoundaryDivergence,
                    "Minus the divergence summed over the incoming boundary: "
                    "the charge the current carries in through d_in W.")
      .def_readonly("bulk_divergence_max", &WardFluxRead::bulkDivergenceMax,
                    "max |(d j)_x| over the interior vertices of W: the Ward "
                    "identity, measured. NaN when W has no interior vertex.")
      .def_readonly("bulk_vertices", &WardFluxRead::bulkVertices)
      .def_readonly("bulk_divergence_vertex",
                    &WardFluxRead::bulkDivergenceVertex,
                    "The interior vertex the divergence was largest at.")
      .def_readonly("incoming_boundary_charge",
                    &WardFluxRead::incomingBoundaryCharge,
                    "Q_in: the fermion number the declared covariance places "
                    "on the carrier cells of d_in W, read independently of the "
                    "current. None when no covariance is declared.")
      .def_readonly("incoming_boundary_cells",
                    &WardFluxRead::incomingBoundaryCells)
      .def_readonly("outgoing_boundary_charge",
                    &WardFluxRead::outgoingBoundaryCharge,
                    "Q_out: the same reading on d_out W.")
      .def_readonly("outgoing_boundary_cells",
                    &WardFluxRead::outgoingBoundaryCells)
      .def_readonly("boundary_charge_residual",
                    &WardFluxRead::boundaryChargeResidual,
                    "|flux - Q_in|.")
      .def_readonly("quark_number", &WardFluxRead::quarkNumber,
                    "N_q, the integer reading of excess_flux when it is "
                    "integral and real to the declared tolerances; None "
                    "otherwise.")
      .def_readonly("quark_number_defect", &WardFluxRead::quarkNumberDefect)
      .def_readonly("baryon_number", &WardFluxRead::baryonNumber,
                    "B(Sigma) = N_q / 3, the whitepaper's one explicit "
                    "physical calibration.")
      .def_readonly("cut_separates", &WardFluxRead::cutSeparates)
      .def_readonly("failed_certificates", &WardFluxRead::failedCertificates);

  py::class_<WardSlabRead>(m, "WardSlabRead",
      "Two cuts of one cobordism compared through the slab between them.")
      .def_readonly("first", &WardSlabRead::first)
      .def_readonly("second", &WardSlabRead::second)
      .def_readonly("slab_vertices", &WardSlabRead::slabVertices,
                    "The vertices on which the two cuts disagree.")
      .def_readonly("flux_difference", &WardSlabRead::fluxDifference,
                    "phi_j(u) - phi_j(u').")
      .def_readonly("slab_divergence", &WardSlabRead::slabDivergence,
                    "sum_x (u(x) - u'(x)) (d j)_x, the signed divergence the "
                    "slab carries.")
      .def_readonly("slab_identity_residual",
                    &WardSlabRead::slabIdentityResidual,
                    "The residual of the slab identity.");

  py::class_<WardHomologyRead>(m, "WardHomologyRead",
      "Several cuts of one cobordism read together.")
      .def_readonly("cuts", &WardHomologyRead::cuts)
      .def_readonly("slabs", &WardHomologyRead::slabs)
      .def_readonly("max_flux_deviation", &WardHomologyRead::maxFluxDeviation,
                    "The largest pairwise difference of the fluxes.")
      .def_readonly("max_slab_divergence", &WardHomologyRead::maxSlabDivergence,
                    "The divergence carried by the slabs between the cuts, "
                    "which is the source content the invariance statement "
                    "excludes.")
      .def_readonly("invariant", &WardHomologyRead::invariant);

  py::class_<IntrinsicResponseConfig>(m, "IntrinsicResponseConfig",
      "The declared parameters of the intrinsic spectral response.")
      .def(py::init<>())
      .def_readwrite("left_current", &IntrinsicResponseConfig::leftCurrent,
                     "The current the left restriction is cut from, in "
                     "canonical degree-one cell order. Empty means the right "
                     "current paired with itself through the transpose.")
      .def_readwrite("degeneracy_tolerance",
                     &IntrinsicResponseConfig::degeneracyTolerance,
                     "Two eigenvalues this close are one degenerate band and "
                     "share one Riesz projector.")
      .def_readwrite("pole_tolerance", &IntrinsicResponseConfig::poleTolerance,
                     "A sample this close to a pole is reported unavailable "
                     "rather than as a large finite number.")
      .def_readwrite("contour_nodes", &IntrinsicResponseConfig::contourNodes,
                     "Quadrature nodes of the Riesz contour each band's "
                     "residue is read on.");

  py::class_<IntrinsicResponseRead>(m, "IntrinsicResponseRead",
      "The intrinsic spectral response Upsilon_Q(lambda) of Section 13.5, "
      "read on one cooriented cut. lambda is an eigenvalue of the slice "
      "operator and is never relabelled as a momentum transfer.")
      .def_readonly("slice_cells", &IntrinsicResponseRead::sliceCells)
      .def_readonly("rho_right", &IntrinsicResponseRead::rhoRight,
                    "The right restriction of the Ward current to the cut.")
      .def_readonly("rho_left", &IntrinsicResponseRead::rhoLeft,
                    "The left restriction.")
      .def_readonly("slice_operator", &IntrinsicResponseRead::sliceOperator,
                    "L_Sigma, flat row-major over the cut's cells.")
      .def_readonly("poles", &IntrinsicResponseRead::poles,
                    "The distinct eigenvalues of L_Sigma.")
      .def_readonly("pole_multiplicity",
                    &IntrinsicResponseRead::poleMultiplicity)
      .def_readonly("residues", &IntrinsicResponseRead::residues,
                    "The residue of Upsilon_Q at each pole, taken on a Riesz "
                    "contour around the whole band.")
      .def_readonly("samples", &IntrinsicResponseRead::samples)
      .def_readonly("response", &IntrinsicResponseRead::response,
                    "Upsilon_Q at each sample.")
      .def_readonly("slope", &IntrinsicResponseRead::slope,
                    "dUpsilon_Q/dlambda at each sample, taken exactly from the "
                    "square of the resolvent.")
      .def_readonly("failed_certificates",
                    &IntrinsicResponseRead::failedCertificates);

  py::class_<WardFlux>(m, "WardFlux",
      "The flux of the complex Ward current through a cooriented cut of the "
      "interaction cobordism W (Section 13.4) and the intrinsic spectral "
      "response it carries (Section 13.5).\n\n"
      "The current is the joint action's link stationarity vector, "
      "j_xy = U_xy dS/dU_xy; nothing here re-derives it and nothing here "
      "supplies a field of its own. The flux is not electric charge: every "
      "edge mode carries charge one under the C* group, so the flux counts "
      "fermions, and a flavor-dependent electric charge is not a gauge charge "
      "of the declared fields and carries no Ward current.")
      .def_static("flux", &WardFlux::flux, py::arg("action"),
                  py::arg("cobordism"), py::arg("cut"),
                  py::arg("cfg") = WardFluxConfig{},
                  "The flux of the action's Ward current through one cut of "
                  "the interaction cobordism W.")
      .def_static("homologous_fluxes", &WardFlux::homologousFluxes,
                  py::arg("action"), py::arg("cobordism"), py::arg("cuts"),
                  py::arg("cfg") = WardFluxConfig{},
                  "Several cuts of W read together, with the flux difference "
                  "and the signed slab divergence of every pair.")
      .def_static("difference", &WardFlux::difference, py::arg("state"),
                  py::arg("matched"), py::arg("cfg") = WardFluxConfig{},
                  "The coherent background removal of Section 13.5: the "
                  "complex difference of two flux reads on one cut, with no "
                  "modulus taken on either side.")
      .def_static("intrinsic_response", &WardFlux::intrinsicResponse,
                  py::arg("action"), py::arg("cobordism"), py::arg("cut"),
                  py::arg("samples"),
                  py::arg("cfg") = IntrinsicResponseConfig{},
                  "Upsilon_Q on one cut of W, evaluated at the declared "
                  "samples.");

  // ── Section 13.3: mass is a complex bound-state pole ───────────────────

  py::class_<BoundStatePoleConfig>(m, "BoundStatePoleConfig",
      "Every declared parameter of the pole read.")
      .def(py::init<>())
      .def_readwrite("rank_tolerance", &BoundStatePoleConfig::rankTolerance,
                     "The relative tolerance of every decision of the read: "
                     "the rank of each residue, the ranks of the powers of "
                     "each cluster's nilpotent part, the invertibility of "
                     "the metric, and the distance, as a fraction of the "
                     "block's largest singular value, at or below which two "
                     "eigenvalues form one pole.")
      .def_readwrite("free_threshold", &BoundStatePoleConfig::freeThreshold,
                     "The complex spectral value the binding shift is measured "
                     "against. None leaves the binding shift unreported.");

  py::class_<BoundStatePoleRead>(m, "BoundStatePoleRead",
      "The zeros of D_C(s) = det F_C(s), read exactly from the spectrum of "
      "the pencil, with the certificates Section 13.3 attaches to a "
      "bound-state pole. Every per-pole vector is parallel to `poles`.")
      .def_readonly("scale", &BoundStatePoleRead::scale,
                    "The largest singular value of the block T = M^-1 A, the "
                    "reference of every rank decision and of the clustering.")
      .def_readonly("poles", &BoundStatePoleRead::poles,
                    "The distinct zeros s_C of D_C: the clusters of "
                    "eigenvalues of the pencil the interior block does not "
                    "carry, ascending by (real part, imaginary part).")
      .def_readonly("multiplicity", &BoundStatePoleRead::multiplicity,
                    "The algebraic multiplicity of each pole: the size of its "
                    "cluster, the dimension of its generalized eigenspace.")
      .def_readonly("geometric_multiplicity",
                    &BoundStatePoleRead::geometricMultiplicity,
                    "The number of Jordan blocks of each pole.")
      .def_readonly("jordan_blocks", &BoundStatePoleRead::jordanBlocks,
                    "The sizes of the Jordan blocks of each pole, descending, "
                    "read from the ranks of the powers of the cluster's "
                    "nilpotent part at the rank tolerance.")
      .def_readonly("cluster_spread", &BoundStatePoleRead::clusterSpread,
                    "The largest distance between two eigenvalues of each "
                    "pole's cluster; zero for an exactly repeated eigenvalue.")
      .def_readonly("simple", &BoundStatePoleRead::simple,
                    "Whether the pole met the simple-isolated specification: "
                    "algebraic multiplicity one.")
      .def_readonly("separation", &BoundStatePoleRead::separation,
                    "The distance from each pole to the nearest other "
                    "reported pole; infinite when it is the only one.")
      .def_readonly("subspace_residual",
                    &BoundStatePoleRead::subspaceResidual,
                    "The Frobenius norm of T V - V U_11 for each pole, with V "
                    "the orthonormal Schur basis of its generalized "
                    "eigenspace and U_11 the block of T on it.")
      .def_readonly("residue", &BoundStatePoleRead::residue,
                    "The residue of the supported resolvent F_C^-1 at each "
                    "pole, flat row-major over the interface coordinates: "
                    "minus the interface block of Pi M^-1, with Pi the "
                    "spectral projector onto the pole's generalized "
                    "eigenspace.")
      .def_readonly("residue_norm", &BoundStatePoleRead::residueNorm)
      .def_readonly("residue_rank", &BoundStatePoleRead::residueRank,
                    "The rank of each residue at the rank tolerance.")
      .def_readonly("binding_shift", &BoundStatePoleRead::bindingShift,
                    "s_C minus the declared free threshold.")
      .def_readonly("interior_poles", &BoundStatePoleRead::interiorPoles,
                    "The distinct eigenvalues of the interior pencil, the "
                    "poles of F_C, which the domain of the read excludes.")
      .def_readonly("interior_multiplicity",
                    &BoundStatePoleRead::interiorMultiplicity,
                    "The algebraic multiplicity of each interior pole.")
      .def_readonly("failed_certificates",
                    &BoundStatePoleRead::failedCertificates,
                    "Named failures: 'empty-interface', "
                    "'eigenvalue-at-interior-pole' and "
                    "'jordan-structure-unresolved'.");

  py::class_<BoundStatePole>(m, "BoundStatePole",
      "Mass as the complex bound-state pole of Section 13.3: the zeros of "
      "D_C(s) = det F_C(s), with F_C the exact meromorphic Feshbach response "
      "pencil of a persistent bound cluster, continued in the complex "
      "spectral parameter s. The zeros are read exactly from the spectrum of "
      "the pencil: the eigenvalues of M^-1 A the interior block does not "
      "carry, clustered at the declared rank tolerance, each with the "
      "spectral projector onto its generalized eigenspace as its residue.\n\n"
      "Mass is not defined here by an incoherent sum of moduli, and nothing "
      "here converts s_C into a mass: the theory carries s_C and takes no "
      "square root of it.")
      .def_static("response", &BoundStatePole::response, py::arg("A"),
                  py::arg("M"), py::arg("interface"), py::arg("s"),
                  py::arg("rank_tolerance") = 1e-12,
                  "F_C(s), as the framework's own Schur complement supplies "
                  "it.")
      .def_static("determinant", &BoundStatePole::determinant, py::arg("A"),
                  py::arg("M"), py::arg("interface"), py::arg("s"),
                  "D_C(s) = det F_C(s).")
      .def_static("response_derivative", &BoundStatePole::responseDerivative,
                  py::arg("A"), py::arg("M"), py::arg("interface"),
                  py::arg("s"),
                  "F_C'(s), the exact analytic derivative of the response.")
      .def_static("poles", &BoundStatePole::poles, py::arg("A"), py::arg("M"),
                  py::arg("interface"),
                  py::arg("cfg") = BoundStatePoleConfig{},
                  "The zeros of D_C with their multiplicities, Jordan "
                  "structure, residues, separations and subspace residuals, "
                  "read exactly from the spectrum of the pencil.")
      .def_static("cluster_poles", &BoundStatePole::clusterPoles,
                  py::arg("assembled"), py::arg("k"), py::arg("cluster_cells"),
                  py::arg("cfg") = BoundStatePoleConfig{},
                  "The same read on an assembled pencil's degree-k block.");
}
