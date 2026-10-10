// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: action term gradients,
// spectral moment constraints, the Villain series and the joint action. One
// of the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_joint_action(py::module_ &m) {
  // ================================================================
  // The holomorphic joint action S(z, U, Gamma) and its solves
  // ================================================================

  py::class_<ActionTermGradient>(
      m, "ActionTermGradient",
      "One term of the joint action at the action's point: its value and its "
      "stationarity, dS_term/dz_e on the lengths and U_e dS_term/dU_e on the "
      "links, one entry per edge each; summed over the terms these are "
      "length_stationarity and link_stationarity exactly.")
      .def(py::init<>())
      .def_readwrite("name", &ActionTermGradient::name,
                     "regge, stiffness, holonomy, matter, or constraint j "
                     "(j from one, in declaration order).")
      .def_readwrite("label", &ActionTermGradient::label,
                     "The term as it stands in the action: (1/kappa) "
                     "S_Regge, beta S_hol, w_m tr(Gamma h_1), "
                     "or xi_j (c_j - c_j*) with what c_j is.")
      .def_readwrite("weight", &ActionTermGradient::weight,
                     "The coefficient in front of the term: 1/kappa, "
                     "beta, w_m, or the multiplier xi_j.")
      .def_readwrite("bare", &ActionTermGradient::bare,
                     "What the weight multiplies (S_Regge, "
                     "tr(Gamma h_1), or the constraint residual c_j - c_j* in "
                     "the declared unit), so value == weight * bare when "
                     "factored; the value itself for the holonomy term.")
      .def_readwrite("factored", &ActionTermGradient::factored,
                     "Whether value == weight * bare by construction.")
      .def_readwrite("value", &ActionTermGradient::value,
                     "The term's value; NaN for a holonomy term that refuses "
                     "to evaluate.")
      .def_readwrite("length_stationarity",
                     &ActionTermGradient::lengthStationarity)
      .def_readwrite("link_stationarity",
                     &ActionTermGradient::linkStationarity);

  py::enum_<SpectralConstraintForm>(
      m, "SpectralConstraintForm",
      "What one SpectralMomentConstraint pins. PowerSum: p_j(h) = tr(h^j) of "
      "the carrier, or of its compression to the declared fiber. BandMean: "
      "the mean eigenvalue lambda_b = tr(P_b h) / r_b of one band, P_b its "
      "Riesz projector (moment_band_projectors) and r_b = tr P_b its rank, "
      "the band's eigenvalue itself when the band is degenerate; its "
      "derivative at fixed P_b is the Hellmann-Feynman tr(P_b dh) / r_b, the "
      "whole derivative when P_b is a spectral projector of h.")
      .value("PowerSum", SpectralConstraintForm::PowerSum)
      .value("BandMean", SpectralConstraintForm::BandMean);

  py::class_<SpectralMomentConstraint>(m, "SpectralMomentConstraint",
      "One holomorphic spectral constraint of the targeted action S_spec: the "
      "power sum p_j(h) = tr(h^j) of the complex edge-mode operator is "
      "required to equal a prescribed complex number p_j*, imposed by an "
      "independent complex Lagrange multiplier xi_j rather than by a penalty. "
      "Stationarity in xi_j is the FULL complex equation p_j(h) - p_j* = 0, "
      "both parts, with no residual norm minimized in its place. These target "
      "terms belong to explicitly labelled controlled synthesis and are absent "
      "in emergence mode.")
      .def(py::init<>())
      .def(py::init([](int order, std::complex<double> target,
                       std::complex<double> multiplier) {
             SpectralMomentConstraint constraint;
             constraint.order = order;
             constraint.target = target;
             constraint.multiplier = multiplier;
             return constraint;
           }),
           py::arg("order"), py::arg("target"),
           py::arg("multiplier") = std::complex<double>{0.0, 0.0})
      .def_readwrite("form", &SpectralMomentConstraint::form,
                     "What the constraint pins: a power sum (the default) or "
                     "a band mean.")
      .def_readwrite("order", &SpectralMomentConstraint::order,
                     "The moment index j >= 1 of a power sum, the power h is "
                     "raised to. It is not a simplicial degree. Unused by a "
                     "band mean.")
      .def_readwrite("band", &SpectralMomentConstraint::band,
                     "For a band mean, the index of the band's projector in "
                     "the declaration's moment_band_projectors. Unused by a "
                     "power sum.")
      .def_readwrite("target", &SpectralMomentConstraint::target,
                     "p_j* = sum_a (lambda_a*)^j, the same power sum of the "
                     "prescribed eigenvalue multiset; or the band mean's "
                     "lambda_b*.")
      .def_readwrite("multiplier", &SpectralMomentConstraint::multiplier,
                     "The complex Lagrange multiplier xi_j at the current "
                     "point of a solve. A variable, not a configured weight.");

  py::enum_<ReggeForm>(m, "ReggeForm",
      "Which discretization the Regge term of JointAction is. Primal is "
      "Regge's own sum_h |h| eps_h (sum_e l_e eps_e in three dimensions), the "
      "form the whitepaper's Section 7 computation takes the second variation "
      "of; Dual is the circumcentric (Sorkin) sum_h |*h| eps_h.")
      .value("Primal", ReggeForm::Primal)
      .value("Dual", ReggeForm::Dual);

  py::enum_<ReggeHinges>(m, "ReggeHinges",
      "Which hinges the primal Regge sum runs over. Interior keeps only the "
      "hinges whose link closes (every (d-1)-face through the hinge shared by "
      "exactly two top cells), where deficit angles are defined; on a complex "
      "with no interior hinge the primal term is zero. All keeps every hinge of "
      "a top cell with the library's 2 pi - sum(theta) deficit at the boundary "
      "too, the sum ReggeSolver.regge_action evaluates.")
      .value("Interior", ReggeHinges::Interior)
      .value("All", ReggeHinges::All);

  py::enum_<ReggeBranch>(m, "ReggeBranch",
      "Which Riemann sheet the primal Regge term and its derivatives are read "
      "on. Continued (the default) declares every cofactor root, inverse "
      "cosine and hinge-content root on its principal sheet at the real "
      "projection of the starting geometry, the Euclidean reference of "
      "specification section 4.2, and continues it along the straight segment "
      "to the starting geometry and from there to the current one. Principal "
      "takes every root principal, which is discontinuous where a Euclidean "
      "face cofactor sits on the cut of the square root.")
      .value("Continued", ReggeBranch::Continued)
      .value("Principal", ReggeBranch::Principal);

  py::class_<VillainSeries>(m, "VillainSeries",
      "The truncated Laurent series W, F W' and (F d/dF)^2 W at one face "
      "holonomy, the largest |m| kept, and certified bounds on the modulus of "
      "each omitted tail.")
      .def_readonly("value", &VillainSeries::value)
      .def_readonly("first", &VillainSeries::first)
      .def_readonly("second", &VillainSeries::second)
      .def_readonly("term_count", &VillainSeries::termCount)
      .def_readonly("value_tail", &VillainSeries::valueTail)
      .def_readonly("first_tail", &VillainSeries::firstTail)
      .def_readonly("second_tail", &VillainSeries::secondTail)
      .def_readonly("magnitude", &VillainSeries::magnitude,
                    "sum of |q^(m^2) F^m| over the kept terms: the rounding "
                    "scale of W.");

  py::class_<VillainCharacter>(m, "VillainCharacter",
      "The Villain weight W(F) = sum_m exp(-m^2/(2 beta)) F^m of one face and "
      "its potential phi(F) = -beta_V log W(F), beta_V = beta / <m^2>_beta, "
      "which matches the Wilson form's curvature beta at trivial holonomy. "
      "Derivatives use W'/W and W''/W only; the logarithm is the branch real "
      "on the unit circle, continued radially.")
      .def(py::init<double, double>(), py::arg("beta"),
           py::arg("tolerance") = 1e-18)
      .def_property_readonly("beta", &VillainCharacter::beta)
      .def_property_readonly("tolerance", &VillainCharacter::tolerance)
      .def_property_readonly("declared_term_count",
                             &VillainCharacter::declaredTermCount,
                             "M_0: the least m with exp(-m^2/(2 beta)) below "
                             "the tolerance.")
      .def_property_readonly("second_moment", &VillainCharacter::secondMoment,
                             "<m^2>_beta at trivial holonomy.")
      .def_property_readonly("matched_weight",
                             &VillainCharacter::matchedWeight,
                             "beta_V = beta / <m^2>_beta.")
      .def("series", &VillainCharacter::series, py::arg("holonomy"))
      .def("logarithm", &VillainCharacter::logarithm, py::arg("holonomy"),
           "log W(F) on the branch real on the unit circle, continued "
           "radially from F/|F|. The start on the unit circle is accepted when "
           "|Im W| <= reality_margin() (tail bound + machine epsilon times the "
           "sum of the moduli of the kept terms) and Re W exceeds the nonzero "
           "margin times the same uncertainty; raises ValueError when W is "
           "not certified nonzero on the path, its start included.")
      .def_static("reality_margin", &VillainCharacter::realityMargin,
                  "c_R, the declared multiple of the series' uncertainty (its "
                  "tail bound plus its rounding scale) within which logarithm "
                  "reads the imaginary part of W on the unit circle as "
                  "rounding.")
      .def("potential", &VillainCharacter::potential, py::arg("holonomy"))
      .def("first_derivative", &VillainCharacter::firstDerivative,
           py::arg("holonomy"), "F dphi/dF = -beta_V F W'/W.")
      .def("second_derivative", &VillainCharacter::secondDerivative,
           py::arg("holonomy"), "(F d/dF)^2 phi.");

  py::class_<HolonomyTruncation>(m, "HolonomyTruncation",
      "The Villain truncation over every face at the current connection: the "
      "declared tolerance and term count, the largest term count any face "
      "needed, and the largest certified tail bounds relative to |W|.")
      .def_readonly("tolerance", &HolonomyTruncation::tolerance)
      .def_readonly("declared_term_count",
                    &HolonomyTruncation::declaredTermCount)
      .def_readonly("maximum_term_count",
                    &HolonomyTruncation::maximumTermCount)
      .def_readonly("relative_value_tail",
                    &HolonomyTruncation::relativeValueTail)
      .def_readonly("relative_first_tail",
                    &HolonomyTruncation::relativeFirstTail)
      .def_readonly("relative_second_tail",
                    &HolonomyTruncation::relativeSecondTail);

  py::class_<JointActionDeclaration>(m, "JointActionDeclaration",
      "Everything that fixes which action S(z, U, Gamma) a JointAction is: the "
      "carrier degree, the three coefficients, the carried covariance, the "
      "spectral constraints and the metric source. Plain data.")
      .def(py::init<>())
      .def_readwrite("carrier_degree", &JointActionDeclaration::carrierDegree,
                     "The simplicial degree k of the one-particle carrier. The "
                     "operator of the action is h = h_k(z, U) and Gamma is a "
                     "matrix over the k-cells.")
      .def_readwrite("gravitational_weight",
                     &JointActionDeclaration::gravitationalWeight,
                     "w_R, the coefficient on the Regge action in the "
                     "declared regge_form. The whitepaper's Section 7 "
                     "identification is 1/(8 pi G) in lattice units, so a "
                     "caller working in the backreaction coupling "
                     "kappa = 8 pi G sets 1/kappa.")
      .def_readwrite("regge_form", &JointActionDeclaration::reggeForm,
                     "Which Regge discretization S_Regge is; Primal by "
                     "default.")
      .def_readwrite("regge_hinges", &JointActionDeclaration::reggeHinges,
                     "Which hinges the primal sum runs over; Interior by "
                     "default.")
      .def_readwrite("regge_branch", &JointActionDeclaration::reggeBranch,
                     "Which sheet the primal Regge term is read on; Continued "
                     "by default.")
      .def_readwrite("regge_start_squared_lengths",
                     &JointActionDeclaration::reggeStartSquaredLengths,
                     "The starting geometry the continued Regge sheets are "
                     "continued from, one squared length per edge in "
                     "get_edge_list() order; empty means the squared lengths "
                     "the mesh holds when the JointAction is constructed.")
      .def_readwrite("holonomy_weight",
                     &JointActionDeclaration::holonomyWeight,
                     "beta, the heat-kernel coupling of the face-holonomy "
                     "term, the Villain action in character form, whose "
                     "coefficient is beta_V = beta / <m^2>_beta; the bare "
                     "stiffness at trivial holonomy is beta L_1^up. Zero "
                     "leaves the term out.")
      .def_readwrite("villain_tolerance",
                     &JointActionDeclaration::villainTolerance,
                     "The relative tolerance below which a Villain coefficient "
                     "exp(-m^2/(2 beta)) is left out of the series.")
      .def_readwrite("matter_weight", &JointActionDeclaration::matterWeight,
                     "w_M, the coefficient on tr(Gamma h(z, U)), the carried "
                     "state's bilinear action density. Zero is strict "
                     "emergence: the geometry is blind to the carried state.")
      .def_readwrite("covariance", &JointActionDeclaration::covariance,
                     "Gamma = Phi PhiTilde^T, flat row-major over the k-cells "
                     "in canonical ChainComplex order. Complex bilinear: no "
                     "adjoint is taken of it and it need not be Hermitian.")
      .def_readwrite("moment_constraints",
                     &JointActionDeclaration::momentConstraints,
                     "The declared holomorphic spectral constraints. Empty in "
                     "emergence mode.")
      .def_readwrite("moment_scale", &JointActionDeclaration::momentScale,
                     "s > 0, the unit the power sums are measured in: the "
                     "constraints are p_j(h / s) (or of h_C / s), with "
                     "targets and multipliers in that unit. The same "
                     "constraints for every s; one by default.")
      .def_readwrite("moment_projector",
                     &JointActionDeclaration::momentProjector,
                     "The Riesz projector P_C of the fiber the constraints are "
                     "imposed on, flat row-major over the k-cells. Empty: the "
                     "power sums of the whole carrier, tr(h^j). Declared: "
                     "those of the compression h_C = P_C h P_C (WP v17 §3.4), "
                     "tr((P_C h P_C)^j), differentiated at fixed P_C.")
      .def_readwrite("moment_band_projectors",
                     &JointActionDeclaration::momentBandProjectors,
                     "The Riesz projectors P_b of the bands the BandMean "
                     "constraints refer to (a constraint's band indexes this "
                     "list), each flat row-major over the k-cells, "
                     "differentiated at fixed P_b; a self-consistent solve "
                     "rebuilds them at every point. The unit s applies: the "
                     "constraint is lambda_b / s = lambda_b* / s.")
      .def_readwrite("metric_source", &JointActionDeclaration::metricSource,
                     "Where the carrier operator's metric comes from. "
                     "WhitneyPencil is the whitepaper's W_k = M_k^-1 and gives "
                     "the covariant h_k(z, U); under DiagonalWeights the "
                     "operator is blind to U, so only the face-holonomy term "
                     "then depends on the connection.");

  py::class_<ReportedActionValue>(m, "ReportedActionValue",
      "The joint action's value as a solver records it: the value when it can "
      "be evaluated, and otherwise the reason it cannot, by name.")
      .def(py::init<>())
      .def_readonly("available", &ReportedActionValue::available)
      .def_readonly("value", &ReportedActionValue::value,
                    "S(z, U, Gamma); NaN in both parts when unavailable.")
      .def_readonly("unavailable", &ReportedActionValue::unavailable,
                    "Why the value is unavailable; empty when it is "
                    "available.");

  py::class_<JointAction>(m, "JointAction",
      "The gauge-invariant joint action S(z, U, Gamma) of Sections 3 and 13 of "
      "the whitepaper, and its exact holomorphic stationarity equations.\n\n"
      "S = w_R S_Regge(z) + S_hol(U) + w_M tr(Gamma "
      "h(z, U)) + sum_j xi_j (c_j - c_j*), with S_hol the branch-free sum of "
      "the Villain per-face potential over the face holonomies "
      "F_tau = prod_e U_e^eps. The stationarity conditions are "
      "the complex equations dS/dz_e = 0, U_e dS/dU_e = 0 and p_j(h) = p_j*, "
      "never the minimization of a selected real projection.\n\n"
      "Every per-edge vector is in get_edge_list() order and every per-cell "
      "quantity in canonical ChainComplex order. The link stationarity of an "
      "edge is reported on its STORED source-to-target orientation, which is "
      "the orientation the Ward current's sign convention refers to.")
      .def(py::init<std::shared_ptr<Spacetime>, JointActionDeclaration>(),
           py::arg("spacetime"), py::arg("declaration"))
      .def_property_readonly("declaration", &JointAction::declaration,
                             "The declaration this instance was built from.")
      .def_property_readonly("spacetime", &JointAction::spacetime,
                             "The complex the action is defined over.")
      .def("set_multipliers", &JointAction::setMultipliers,
           py::arg("multipliers"),
           "Replace xi_j for every constraint, in declaration order. This is "
           "how a solver advances the multipliers, which are variables of the "
           "stationarity system rather than configuration.")
      .def("multipliers", &JointAction::multipliers,
           "The current xi_j, in declaration order.")
      .def("set_covariance", &JointAction::setCovariance,
           py::arg("covariance"),
           "Replace the carried covariance Gamma (flat row-major over the "
           "k-cells). Nothing else of the declaration changes; the Riemann "
           "sheets of a continued Regge term stay those fixed at construction.")
      .def("set_moment_projector", &JointAction::setMomentProjector,
           py::arg("projector"),
           "Replace the fiber the spectral constraints are imposed on, in "
           "place, as set_covariance replaces Gamma.")
      .def("set_moment_band_projectors", &JointAction::setMomentBandProjectors,
           py::arg("projectors"),
           "Replace the band projectors the BandMean constraints refer to, "
           "one per declared entry, in place, as set_covariance replaces "
           "Gamma.")
      .def("carrier_operator", &JointAction::carrierOperator,
           "h_k(z, U), flat row-major over the k-cells.")
      .def("face_holonomies", &JointAction::faceHolonomies,
           "F_tau per triangle, the ordered product of the links over the "
           "incidences of the boundary map. No sum of phases and no logarithm "
           "is formed.")
      .def("constraint_values", &JointAction::constraintValues,
           "The value of each declared constraint in the declared unit: "
           "p_j(h / s) = tr((h / s)^j) for a power sum, by repeated "
           "multiplication, so a defective operator needs no "
           "eigendecomposition and no eigenvalue ordering; tr(P_b h) / "
           "(r_b s) for a band mean.")
      .def("power_sums", &JointAction::powerSums,
           "constraint_values under the name of the power-sum form; a "
           "band-mean constraint's entry is its band mean.")
      .def("moment_residuals", &JointAction::momentResiduals,
           "Each declared constraint's value less its target: the exact "
           "complex stationarity equation in xi_j.")
      .def("regge_term", &JointAction::reggeTerm,
           "w_R S_Regge(z) in the declared form and hinge set.")
      .def("regge_hinge_count", &JointAction::reggeHingeCount,
           "The number of hinges the primal Regge sum runs over under the "
           "declared hinge rule.")
      .def("regge_structurally_zero", &JointAction::reggeStructurallyZero,
           "True when a declared primal Regge term has no hinge on this "
           "complex under the declared hinge rule, so the term and its "
           "gradient are identically zero for every geometry.")
      .def("regge_off_principal_angles", &JointAction::reggeOffPrincipalAngles,
           "Under ReggeBranch.Continued, the number of dihedral angles whose "
           "continued sheet differs from the principal one at the current "
           "geometry.")
      .def("holonomy_term", &JointAction::holonomyTerm,
           "S_hol(U) in the declared form, weight included. For the Villain "
           "form it is the one quantity that needs log W, taken on the branch "
           "real on the unit circle; it raises for a face holonomy at or "
           "beyond a zero of W on the negative real axis.")
      .def("matter_term", &JointAction::matterTerm,
           "w_M tr(Gamma h(z, U)).")
      .def("spectral_term", &JointAction::spectralTerm,
           "sum_j xi_j (p_j(h) - p_j*).")
      .def("value", &JointAction::value,
           "S(z, U, Gamma), the sum of the five terms.")
      .def("reported_value", &JointAction::reportedValue,
           "The value as a solver reports it: available with the value, or "
           "unavailable with the refusal's message (for example log W refused "
           "at a face holonomy). Only the holonomy term needs log W, and a "
           "solver reads the value only to report it.")
      .def("length_stationarity", &JointAction::lengthStationarity,
           "dS/dz_e per edge, assembled from the framework's exact analytic "
           "gradients. No finite difference and no discarded imaginary part.")
      .def("link_stationarity", &JointAction::linkStationarity,
           "U_e dS/dU_e per edge on its stored orientation, from the closed "
           "form of the holonomy term and the identity U d/dU = -i d/dphi "
           "applied to the exact analytic operator phase gradient.")
      .def("ward_current", &JointAction::wardCurrent,
           "j_xy = U_xy dS/dU_xy of Section 13.4, which is link_stationarity "
           "under its other name. Odd under reversing an edge.")
      .def("canonical_ward_current", &JointAction::canonicalWardCurrent,
           "The Ward current on the canonical degree-one cells, in the chain "
           "complex's cell order and on each cell's ascending-vertex "
           "orientation: ward_current reordered and re-signed to the indexing "
           "every chain-level consumer of it uses.")
      .def("ward_current_divergence", &JointAction::wardCurrentDivergence,
           "(d j)_x per vertex. It vanishes identically for every "
           "gauge-invariant term; a fixed Gamma held while the connection "
           "varies breaks the identity, and this measures by how much.")
      .def("holonomy_hessian", &JointAction::holonomyHessian,
           "sum_tau eps_tau,e eps_tau,e' (F d/dF)^2 phi(F_tau), flat |E| x |E| "
           "in get_edge_list() order on stored orientations: the holonomy term's "
           "exact contribution to the link block of the Jacobian in the "
           "multiplicative coordinate U -> U e^delta. Minus it is the Hessian "
           "in the real angles.")
      .def("holonomy_truncation", &JointAction::holonomyTruncation,
           "The Villain series truncation over the current face holonomies, "
           "with certified relative tail bounds.")
      .def("hellmann_feynman_length_force",
           &JointAction::hellmannFeynmanLengthForce,
           "tr(Gamma dh/dz_e) per edge, the carried state's whole "
           "contribution to the length equation, without w_M and without the "
           "geometric terms.")
      .def("hellmann_feynman_link_force",
           &JointAction::hellmannFeynmanLinkForce,
           "tr(Gamma U_e dh/dU_e) per edge, the link counterpart. Identically "
           "zero when the operator is blind to the connection.")
      .def("occupation_numbers", &JointAction::occupationNumbers,
           "n_c = Gamma_cc per k-cell, the derived readout. Complex in "
           "general, because Gamma is a complex bilinear covariance.")
      .def("stationarity_residual", &JointAction::stationarityResidual,
           "The whole residual (dS/dz; U dS/dU; p_j - p_j*) in that block "
           "order, the vector a holomorphic root find drives to zero.")
      .def("stationarity_residual_norm",
           &JointAction::stationarityResidualNorm,
           "The Euclidean norm of stationarity_residual, a convergence "
           "certificate rather than a functional minimized in place of the "
           "equations.")
      .def("term_gradients", &JointAction::termGradients,
           "Every term of the action with its value and its stationarity "
           "(ActionTermGradient): regge, stiffness, holonomy, matter, then one "
           "per declared constraint; a term of zero weight is listed with "
           "zeros. For records and traces, not the inner loop of a solve.")
      .def("moment_gradient", &JointAction::momentGradient, py::arg("index"),
           "(dp_j/dz; U dp_j/dU) for one declared constraint, the exact "
           "analytic Jacobian column its multiplier contributes.")
      .def("edge_count", &JointAction::edgeCount)
      .def("constraint_count", &JointAction::constraintCount)
      .def("carrier_eigenvalues", &JointAction::carrierEigenvalues,
           "The eigenvalues of h_k(z, U), unordered.")
      .def("ordered_carrier_eigenvalues",
           &JointAction::orderedCarrierEigenvalues,
           py::arg("ascending_real_part") = true,
           "The eigenvalues in the order occupation_projector fills them.")
      .def("occupation_projector", &JointAction::occupationProjector,
           py::arg("occupied"), py::arg("ascending_real_part") = true,
           "The spectral (Riesz) projector onto the occupied modes: Gamma = V "
           "diag(chi) V^-1, which is exactly Gamma = Phi PhiTilde^T for the "
           "matched left/right frame pair. Idempotent by construction, with no "
           "adjoint anywhere, and Hermitian only when h is normal.")
      .def_static("term_names", &JointAction::termNames,
                  "The five declared terms, in the order value sums them.");
}
