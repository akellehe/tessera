// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the self-consistent mean
// field. One of the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_self_consistent_mean_field(py::module_ &m) {
  py::enum_<OccupationOrder>(m, "OccupationOrder",
      "Which modes of the carrier operator the covariance projects onto. The "
      "whitepaper names the filled modes 'the occupied modes' and fixes no "
      "order for a genuinely complex spectrum, so the rule is declared and "
      "recorded rather than assumed.")
      .value("AscendingRealPart", OccupationOrder::AscendingRealPart)
      .value("AscendingModulus", OccupationOrder::AscendingModulus);

  py::enum_<CovarianceRule>(m, "CovarianceRule",
      "How SelfConsistentMeanField builds the carried covariance. "
      "OccupiedProjector is the spectral projector onto the declared number of "
      "occupied modes, a Slater determinant. BandFilling groups the ordered "
      "spectrum into degenerate bands and spreads a declared occupation n_b "
      "evenly over band b, Gamma = sum_b (n_b / r_b) P_b: the one-body density "
      "of a state invariant under the symmetry that protects the bands. Which "
      "bands carry the occupations is fixed by BandSelection.")
      .value("OccupiedProjector", CovarianceRule::OccupiedProjector)
      .value("BandFilling", CovarianceRule::BandFilling);

  py::enum_<BandSelection>(m, "BandSelection",
      "Where the occupied bands are chosen. Continuation (the default): once, "
      "at the starting point, by the declared order, and then each band is "
      "followed from point to point by maximal overlap with its previous "
      "Riesz projector (WP v17 line 151: a band is selected by a contour, not "
      "by sorting real parts), with every band's overlap and any crossing "
      "reported. SortEveryIterate: re-selected at every point by the declared "
      "order, so crossing bands exchange their occupations.")
      .value("Continuation", BandSelection::Continuation)
      .value("SortEveryIterate", BandSelection::SortEveryIterate);

  py::enum_<FiberConstraintForm>(
      m, "FiberConstraintForm",
      "What the fiber constraints of a self-consistent solve pin. PowerSums: "
      "p_j(h_C), j = 1..m_c, of the occupied fiber, the constraints of WP v17 "
      "§3.4 as written. BandEigenvalues: the eigenvalue lambda_b = tr(P_b h) / "
      "r_b of each occupied band, one constraint per band in the declared "
      "order, the first m_c of them, each band's projector rebuilt at every "
      "point; on a sheeted host the r power sums carry only as many "
      "independent constraints as there are occupied bands, and this states "
      "those constraints without the dependent rows.")
      .value("PowerSums", FiberConstraintForm::PowerSums)
      .value("BandEigenvalues", FiberConstraintForm::BandEigenvalues);

  py::class_<SelfConsistentMeanFieldDeclaration>(
      m, "SelfConsistentMeanFieldDeclaration",
      "The configuration of a self-consistent backreaction solve.")
      .def(py::init<>())
      .def_readwrite("occupied_modes",
                     &SelfConsistentMeanFieldDeclaration::occupiedModes,
                     "How many modes of the carrier operator are filled under "
                     "the occupied-projector rule.")
      .def_readwrite("occupation_order",
                     &SelfConsistentMeanFieldDeclaration::occupationOrder,
                     "The order modes and bands are counted in where the "
                     "bands are chosen.")
      .def_readwrite("covariance_rule",
                     &SelfConsistentMeanFieldDeclaration::covarianceRule,
                     "How the covariance is built: the occupied projector, or "
                     "the band filling sum_b (n_b / r_b) P_b.")
      .def_readwrite("band_occupations",
                     &SelfConsistentMeanFieldDeclaration::bandOccupations,
                     "n_b, the number of particles each band holds under the "
                     "band-filling rule, in the declared order.")
      .def_readwrite("band_tolerance",
                     &SelfConsistentMeanFieldDeclaration::bandTolerance,
                     "The relative separation at or below which consecutive "
                     "ordered eigenvalues belong to one band.")
      .def_readwrite("band_symmetry",
                     &SelfConsistentMeanFieldDeclaration::bandSymmetry,
                     "The operators D(g) of a declared finite symmetry, each "
                     "flat row-major over the carrier's cells. When present "
                     "the bands are read on the group average "
                     "|G|^-1 sum_g D(g)^-1 h D(g), and Gamma then does not "
                     "commute with h in general. Empty, the default, reads "
                     "the bands of h itself, so Gamma commutes with h (WP v17 "
                     "Section 7: Gamma* a projector onto modes of h(z*)).")
      .def_readwrite("band_selection",
                     &SelfConsistentMeanFieldDeclaration::bandSelection,
                     "Where the occupied bands are chosen: Continuation (the "
                     "default) or SortEveryIterate.")
      .def_readwrite("fiber_moments",
                     &SelfConsistentMeanFieldDeclaration::fiberMoments,
                     "m_c, the number of power sums p_j(h_C), j = 1..m_c, of "
                     "the occupied fiber (the occupied bands, followed and "
                     "rebuilt as the covariance's are) pinned by the "
                     "holomorphic spectral constraints of WP v17 §3.4, with "
                     "their complex multipliers solved for beside the "
                     "geometry. Zero (the default) pins nothing; at most the "
                     "fiber's rank r.")
      .def_readwrite("fiber_moment_targets",
                     &SelfConsistentMeanFieldDeclaration::fiberMomentTargets,
                     "The targets p_j*, one per pinned moment, in the "
                     "operator's own unit; empty (the default) takes the "
                     "fiber's own values at the starting point, which pins "
                     "the carrier as declared.")
      .def_readwrite("fiber_moment_scale",
                     &SelfConsistentMeanFieldDeclaration::fiberMomentScale,
                     "The unit s the pinned power sums are solved in, "
                     "p_j(h_C / s) = p_j* s^-j (the same constraints, with "
                     "commensurate gradients); zero (the default) takes the "
                     "fiber's spectral radius at the starting point.")
      .def_readwrite("fiber_constraint_form",
                     &SelfConsistentMeanFieldDeclaration::fiberConstraintForm,
                     "What the pinned constraints are: the fiber's power sums "
                     "(the default) or its bands' eigenvalues, under which "
                     "fiber_moments is the number of occupied bands pinned "
                     "and fiber_moment_targets are their eigenvalues.")
      .def_readwrite("maximum_iterations",
                     &SelfConsistentMeanFieldDeclaration::maximumIterations,
                     "The largest number of iterations of the declared "
                     "method: Newton steps of the joint system, or outer "
                     "iterations of the alternation. Zero reads the starting "
                     "point only.")
      .def_readwrite("tolerance",
                     &SelfConsistentMeanFieldDeclaration::tolerance,
                     "The force norm at or below which the pair is "
                     "self-consistent. Under the alternation the covariance "
                     "change must also sit at or below it, since there the "
                     "covariance lags the geometry by one iteration.")
      .def_readwrite("geometry",
                     &SelfConsistentMeanFieldDeclaration::geometry,
                     "The Newton solve of the geometry: the joint solve's "
                     "step control (its maximum_iterations is not read).");

  py::class_<OccupiedBand>(m, "OccupiedBand",
      "One occupied band at one point of a solve: where it was chosen, its "
      "occupation and rank, its eigenvalues and places in the declared order "
      "here, its overlap tr(P P_prev) / r with its projector at the previous "
      "point, whether it crossed another band, and whether it splits a "
      "degenerate group.")
      .def(py::init<>())
      .def_readonly("declared_index", &OccupiedBand::declaredIndex)
      .def_readonly("occupation", &OccupiedBand::occupation)
      .def_readonly("rank", &OccupiedBand::rank)
      .def_readonly("eigenvalues", &OccupiedBand::eigenvalues)
      .def_readonly("positions", &OccupiedBand::positions)
      .def_readonly("declared_positions", &OccupiedBand::declaredPositions)
      .def_readonly("overlap", &OccupiedBand::overlap)
      .def_readonly("crossed", &OccupiedBand::crossed)
      .def_readonly("ambiguous", &OccupiedBand::ambiguous)
      .def_readonly("projector", &OccupiedBand::projector);

  py::class_<BandRead>(m, "BandRead",
      "What BandFollower.read builds from one operator: the covariance, the "
      "band ranks and eigenvalues in the declared order, the occupied bands, "
      "the occupied span and its gap in the declared order, the isolation of "
      "the occupied bands, and whether any crossed.")
      .def(py::init<>())
      .def_readonly("covariance", &BandRead::covariance)
      .def_readonly("ranks", &BandRead::ranks)
      .def_readonly("ordered", &BandRead::ordered)
      .def_readonly("bands", &BandRead::bands)
      .def_readonly("occupied_eigenvalues", &BandRead::occupiedEigenvalues)
      .def_readonly("spectral_gap", &BandRead::spectralGap)
      .def_readonly("band_isolation", &BandRead::bandIsolation)
      .def_readonly("crossing", &BandRead::crossing)
      .def_readonly("lowest_overlap", &BandRead::lowestOverlap);

  py::class_<BandFollower>(m, "BandFollower",
      "The declared covariance rule with its band selection: read(h) builds "
      "Gamma from an operator, choosing the occupied bands by the declared "
      "order when no reference is set (or under SortEveryIterate) and "
      "following them from the reference otherwise; follow(read) makes a "
      "read's bands the reference.")
      .def(py::init<const SelfConsistentMeanFieldDeclaration &>(),
           py::arg("declaration"))
      .def("read", &BandFollower::read, py::arg("operator"),
           "The band read of a flat row-major operator.")
      .def("follow", &BandFollower::follow, py::arg("read"),
           "Make the read's bands the reference the next read follows.")
      .def_property_readonly("following", &BandFollower::following);

  py::class_<SelfConsistentMeanFieldStep>(m, "SelfConsistentMeanFieldStep",
      "One iterate of a self-consistent solve; iterate zero is the starting "
      "point.")
      .def(py::init<>())
      .def_readwrite("iteration", &SelfConsistentMeanFieldStep::iteration)
      .def_readwrite("force_norm", &SelfConsistentMeanFieldStep::forceNorm)
      .def_readwrite("terms", &SelfConsistentMeanFieldStep::terms,
                     "Every term of the action at this iterate "
                     "(ActionTermRecord), when the geometry declaration "
                     "records terms; empty otherwise.")
      .def_readwrite("covariance_change",
                     &SelfConsistentMeanFieldStep::covarianceChange)
      .def_readwrite("purity_defect",
                     &SelfConsistentMeanFieldStep::purityDefect,
                     "||Gamma^2 - Gamma||_F, the Gaussianity certificate.")
      .def_readwrite("action", &SelfConsistentMeanFieldStep::action)
      .def_readwrite("action_available",
                     &SelfConsistentMeanFieldStep::actionAvailable)
      .def_readwrite("action_unavailable",
                     &SelfConsistentMeanFieldStep::actionUnavailable)
      .def_readwrite("occupied_energy",
                     &SelfConsistentMeanFieldStep::occupiedEnergy,
                     "tr(Gamma h), which for a spectral projector is the sum "
                     "of the occupied eigenvalues.")
      .def_readwrite("occupied_eigenvalues",
                     &SelfConsistentMeanFieldStep::occupiedEigenvalues,
                     "The eigenvalues of the occupied span in the declared "
                     "order.")
      .def_readwrite("spectral_gap",
                     &SelfConsistentMeanFieldStep::spectralGap,
                     "The gap at the end of the occupied span in the declared "
                     "order.")
      .def_readwrite("band_ranks", &SelfConsistentMeanFieldStep::bandRanks,
                     "The band ranks under the band-filling rule.")
      .def_readwrite("bands", &SelfConsistentMeanFieldStep::bands,
                     "The occupied bands with their overlaps and crossings.")
      .def_readwrite("band_isolation",
                     &SelfConsistentMeanFieldStep::bandIsolation)
      .def_readwrite("band_crossing",
                     &SelfConsistentMeanFieldStep::bandCrossing)
      .def_readwrite("multipliers", &SelfConsistentMeanFieldStep::multipliers,
                     "The pinned fiber moments' multipliers xi_j, in the "
                     "operator's own unit.")
      .def_readwrite("moment_residual_norm",
                     &SelfConsistentMeanFieldStep::momentResidualNorm,
                     "The norm of the pinned constraints' residuals in the "
                     "unit they are solved in; zero when none is pinned.")
      .def_readwrite("geometry_converged",
                     &SelfConsistentMeanFieldStep::geometryConverged)
      .def_readwrite("geometry_residual_norm",
                     &SelfConsistentMeanFieldStep::geometryResidualNorm)
      .def_readwrite("geometry_stop_reason",
                     &SelfConsistentMeanFieldStep::geometryStopReason)
      .def_readwrite("geometry_stop_detail",
                     &SelfConsistentMeanFieldStep::geometryStopDetail)
      .def_readwrite("newton_iterated",
                     &SelfConsistentMeanFieldStep::newtonIterated)
      .def_readwrite("newton", &SelfConsistentMeanFieldStep::newton);

  py::class_<SelfConsistentMeanFieldReport>(m, "SelfConsistentMeanFieldReport",
      "What a self-consistent solve reached, which method and band selection "
      "ran, and why it stopped.")
      .def(py::init<>())
      .def_readwrite("band_selection",
                     &SelfConsistentMeanFieldReport::bandSelection)
      .def_readwrite("steps", &SelfConsistentMeanFieldReport::steps)
      .def_readwrite("iterations", &SelfConsistentMeanFieldReport::iterations,
                     "Accepted Newton steps of the joint system, or outer "
                     "iterations of the alternation.")
      .def_readwrite("converged", &SelfConsistentMeanFieldReport::converged)
      .def_readwrite("stop_reason", &SelfConsistentMeanFieldReport::stopReason)
      .def_readwrite("stop_detail", &SelfConsistentMeanFieldReport::stopDetail)
      .def_readwrite("force_norm", &SelfConsistentMeanFieldReport::forceNorm)
      .def_readwrite("covariance_change",
                     &SelfConsistentMeanFieldReport::covarianceChange)
      .def_readwrite("purity_defect",
                     &SelfConsistentMeanFieldReport::purityDefect)
      .def_readwrite("covariance", &SelfConsistentMeanFieldReport::covariance)
      .def_readwrite("occupied_eigenvalues",
                     &SelfConsistentMeanFieldReport::occupiedEigenvalues)
      .def_readwrite("occupied_energy",
                     &SelfConsistentMeanFieldReport::occupiedEnergy)
      .def_readwrite("spectral_gap",
                     &SelfConsistentMeanFieldReport::spectralGap)
      .def_readwrite("band_ranks", &SelfConsistentMeanFieldReport::bandRanks)
      .def_readwrite("bands", &SelfConsistentMeanFieldReport::bands)
      .def_readwrite("band_isolation",
                     &SelfConsistentMeanFieldReport::bandIsolation)
      .def_readwrite("band_crossing_iterates",
                     &SelfConsistentMeanFieldReport::bandCrossingIterates)
      .def_readwrite("lowest_band_overlap",
                     &SelfConsistentMeanFieldReport::lowestBandOverlap)
      .def_readwrite("action", &SelfConsistentMeanFieldReport::action)
      .def_readwrite("action_available",
                     &SelfConsistentMeanFieldReport::actionAvailable)
      .def_readwrite("action_unavailable",
                     &SelfConsistentMeanFieldReport::actionUnavailable)
      .def_readwrite("jacobian_size",
                     &SelfConsistentMeanFieldReport::jacobianSize)
      .def_readwrite("jacobian_rank",
                     &SelfConsistentMeanFieldReport::jacobianRank,
                     "The rank of the joint Jacobian at the end point.")
      .def_readwrite("largest_singular_value",
                     &SelfConsistentMeanFieldReport::largestSingularValue)
      .def_readwrite(
          "smallest_retained_singular_value",
          &SelfConsistentMeanFieldReport::smallestRetainedSingularValue)
      .def_readwrite(
          "largest_discarded_singular_value",
          &SelfConsistentMeanFieldReport::largestDiscardedSingularValue)
      .def_readwrite("rank_gap", &SelfConsistentMeanFieldReport::rankGap,
                     "The joint Jacobian's rank gap at the end point.")
      .def_readwrite("kontsevich_segal_margin",
                     &SelfConsistentMeanFieldReport::kontsevichSegalMargin,
                     "Positive on an allowable geometry, negative otherwise.")
      .def_readwrite("fiber_rank", &SelfConsistentMeanFieldReport::fiberRank,
                     "r, the rank of the occupied fiber where the bands were "
                     "chosen.")
      .def_readwrite("moment_scale",
                     &SelfConsistentMeanFieldReport::momentScale,
                     "The unit s the pinned constraints were solved in.")
      .def_readwrite("fiber_constraint_form",
                     &SelfConsistentMeanFieldReport::fiberConstraintForm,
                     "What was pinned: the fiber's power sums or its bands' "
                     "eigenvalues.")
      .def_readwrite("moment_targets",
                     &SelfConsistentMeanFieldReport::momentTargets,
                     "The pinned fiber moments' targets p_j*, in the "
                     "operator's own unit.")
      .def_readwrite("multipliers", &SelfConsistentMeanFieldReport::multipliers,
                     "The multipliers xi_j of xi_j (p_j(h_C) - p_j*) at the "
                     "end point, in the operator's own unit.")
      .def_readwrite("moment_residuals",
                     &SelfConsistentMeanFieldReport::momentResiduals,
                     "p_j(h_C) - p_j* at the end point, in the operator's own "
                     "unit.")
      .def_readwrite(
          "hellmann_feynman_force_norm",
          &SelfConsistentMeanFieldReport::hellmannFeynmanForceNorm,
          "||tr(Gamma dh)|| on the relaxed geometric coordinates at the end "
          "point, without the geometric terms and the constraints.")
      .def_readwrite("force_hessian",
                     &SelfConsistentMeanFieldReport::forceHessian,
                     "The moment-constrained action's Hessian on the range of "
                     "the Hellmann-Feynman force (WP v17 line 265): the "
                     "Rayleigh quotient f^T H f / f^T f of the joint "
                     "Jacobian's geometric block along the Hellmann-Feynman "
                     "force projected onto the pinned constraints' tangent "
                     "space, in the coordinates (z, theta), delta = i theta. "
                     "Real on the real slice, where its sign is the "
                     "condition's reading.")
      .def_readwrite("force_hessian_scale",
                     &SelfConsistentMeanFieldReport::forceHessianScale,
                     "The Frobenius norm of that Hessian's geometric block in "
                     "(z, theta), against which the quotient's imaginary part "
                     "is read.")
      .def_readwrite("largest_length_ratio",
                     &SelfConsistentMeanFieldReport::largestLengthRatio,
                     "The largest |z_e| at the end over its value at the "
                     "start.");

  py::class_<SelfConsistentMeanField>(m, "SelfConsistentMeanField",
      "The certificates-blind mean-field backreaction of Section 7, solved to "
      "self-consistency.\n\n"
      "The only channel from the state to the geometry is the bilinear action "
      "density, so the force on an edge is tr(Gamma dh/dz_e) and the force on "
      "a link is tr(Gamma U_e dh/dU_e); both are complex and neither is "
      "projected onto a real part. A fixed point is the self-consistent "
      "polaron: Gamma* is the declared rule's density of the modes of h(z*) "
      "and the state's force balances the geometric action edge by edge. It "
      "is a stationary point of a complex action, not a minimum of a real "
      "one, solved for by Newton's method on the joint system; the declared "
      "band selection (Continuation by default) changes no equation. A solve "
      "that finds no fixed point reports why, by name.")
      .def(py::init<JointAction, SelfConsistentMeanFieldDeclaration>(),
           py::arg("action"), py::arg("declaration"))
      .def("solve", &SelfConsistentMeanField::solve,
           py::call_guard<py::gil_scoped_release>(),
           "Run the solve, writing the relaxed geometry into the complex. The "
           "interpreter lock is released for its duration, so a caller's "
           "other threads (a live display) keep running.")
      .def_property_readonly("action", &SelfConsistentMeanField::action,
                             "The action, carrying the covariance and the "
                             "multipliers as the solve left them.");
}
