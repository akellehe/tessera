// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the holomorphic relaxation.
// One of the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_holomorphic_relaxation(py::module_ &m) {
  py::enum_<HolomorphicJacobianMode>(m, "HolomorphicJacobianMode",
      "How the Jacobian of the stationarity system is formed.\n\n"
      "ContourDerivative is the Cauchy derivative on a small circle, which for "
      "a residual analytic on the whole disc converges geometrically in the "
      "node count and is exact to rounding at the default eight nodes.\n\n"
      "RealAxisDifference is the two-node rule with both nodes placed exactly "
      "on the real axis. It is the rule for a residual analytic on each side of "
      "a cut along the real axis but not across it, which is what the dual "
      "Regge action's exact gradient is: the deficit angle is taken on the "
      "principal branch with no Riemann-sheet label, so an arbitrarily small "
      "positive imaginary part in a squared length shifts a hinge's deficit by "
      "2 pi. Its truncation is O(radius^2), so a caller declaring it usually "
      "declares a smaller radius with it.")
      .value("ContourDerivative", HolomorphicJacobianMode::ContourDerivative)
      .value("RealAxisDifference",
             HolomorphicJacobianMode::RealAxisDifference);

  py::enum_<RelaxationStop>(m, "RelaxationStop",
      "Why a solve stopped, reported by name. Converged; IterationBudget (the "
      "declared iterations ran out); NoDescent (no damped step reduced the "
      "residual); SectorBoundary (no stationary point in the declared "
      "monopole sector: the smallest damped step changed a held monopole "
      "number, a held face holonomy driven across -1); DomainBoundary (the "
      "smallest damped step left the domain of the action); HeldFloor (with "
      "held sectors, the residual is at its floor on the held "
      "set: the constrained step cannot reduce it by more than the "
      "tolerance); LengthRunaway (a squared length overflowed the double); "
      "Continued (not a stop: a per-iterate trace entry the solve stepped on "
      "from).")
      .value("Converged", RelaxationStop::Converged)
      .value("IterationBudget", RelaxationStop::IterationBudget)
      .value("NoDescent", RelaxationStop::NoDescent)
      .value("SectorBoundary", RelaxationStop::SectorBoundary)
      .value("DomainBoundary", RelaxationStop::DomainBoundary)
      .value("HeldFloor", RelaxationStop::HeldFloor)
      .value("LengthRunaway", RelaxationStop::LengthRunaway)
      .value("Continued", RelaxationStop::Continued);

  m.def("relaxation_stop_name", &relaxationStopName, py::arg("reason"),
        "The name a report prints for a stop reason, for example 'no "
        "stationary point in the declared monopole sector'.");

  py::class_<HeldMonopoleSector>(m, "HeldMonopoleSector",
      "A declared cluster whose monopole sector is boundary data of a "
      "relaxation: the outward-oriented faces of its bounding cut (three "
      "vertex ids each) and its monopole number.")
      .def(py::init<>())
      .def_readwrite("faces", &HeldMonopoleSector::faces)
      .def_readwrite("monopole_number", &HeldMonopoleSector::monopoleNumber);

  py::class_<HolomorphicRelaxationDeclaration>(
      m, "HolomorphicRelaxationDeclaration",
      "The numerical controls of a holomorphic Newton solve. None of them "
      "changes which equations are solved.")
      .def(py::init<>())
      .def_readwrite("relax_lengths",
                     &HolomorphicRelaxationDeclaration::relaxLengths,
                     "Whether the squared lengths are variables. When false "
                     "they are held and their equations are dropped, because "
                     "requiring an equation of a frozen variable would "
                     "overdetermine the system.")
      .def_readwrite("relax_links",
                     &HolomorphicRelaxationDeclaration::relaxLinks,
                     "Whether the links are variables, under the same rule.")
      .def_readwrite("relax_multipliers",
                     &HolomorphicRelaxationDeclaration::relaxMultipliers,
                     "Whether the multipliers are variables, under the same "
                     "rule.")
      .def_readwrite("maximum_iterations",
                     &HolomorphicRelaxationDeclaration::maximumIterations)
      .def_readwrite("tolerance", &HolomorphicRelaxationDeclaration::tolerance,
                     "The residual norm at or below which the solve is "
                     "declared converged.")
      .def_readwrite("contour_nodes",
                     &HolomorphicRelaxationDeclaration::contourNodes,
                     "The number of nodes on the contour. At least five.")
      .def_readwrite("contour_radius",
                     &HolomorphicRelaxationDeclaration::contourRadius,
                     "The contour radius, relative to the magnitude of the "
                     "coordinate being differentiated and floored at one.")
      .def_readwrite("maximum_dampings",
                     &HolomorphicRelaxationDeclaration::maximumDampings,
                     "The largest number of step halvings tried when a full "
                     "Newton step does not reduce the residual norm.")
      .def_readwrite("jacobian_mode",
                     &HolomorphicRelaxationDeclaration::jacobianMode)
      .def_readwrite("rank_tolerance",
                     &HolomorphicRelaxationDeclaration::rankTolerance,
                     "The relative threshold below which a singular value of "
                     "the Jacobian, over the largest, counts as zero in the "
                     "minimum-norm solve.")
      .def_readwrite("record_terms",
                     &HolomorphicRelaxationDeclaration::recordTerms,
                     "Whether every recorded step (and the starting point) "
                     "carries every term of the action with its value and "
                     "gradient norm (ActionTermRecord). Off by default: it "
                     "costs a few stationarity evaluations per step.")
      .def_readwrite("edge_classes",
                     &HolomorphicRelaxationDeclaration::edgeClasses,
                     "Coordinates shared by several edges, one class index per "
                     "edge in get_edge_list() order (0 to K - 1, every index "
                     "used); empty makes every edge its own coordinate. The "
                     "edges of a class carry one squared length and one link, "
                     "and the equation of a class is the sum of its edges' "
                     "equations: a k-sheeted support relaxed as one base "
                     "field (WP v17 §8).")
      .def_readwrite("edge_class_orientations",
                     &HolomorphicRelaxationDeclaration::edgeClassOrientations,
                     "+1 when an edge's stored link is its class's link, -1 "
                     "when it is the inverse; empty means +1 throughout.")
      .def_readwrite("held_sectors",
                     &HolomorphicRelaxationDeclaration::heldSectors,
                     "Monopole sectors held as boundary data: the moduli of "
                     "their cut faces' holonomies and their monopole numbers "
                     "are kept, every other connection degree of freedom "
                     "relaxes. The step is the constrained Newton step, the "
                     "minimum-norm least-squares solution of the linearized "
                     "equations over the tangent space of the held set.");

  py::class_<ActionTermRecord>(
      m, "ActionTermRecord",
      "One term of the joint action at a recorded point: its value and the "
      "Euclidean norm of its stationarity gradient on the coordinates the "
      "relaxation relaxes, reduced onto the declared edge classes as the "
      "residual is. The multiplier equations are left out, as they are of "
      "the force norm.")
      .def(py::init<>())
      .def_readwrite("name", &ActionTermRecord::name,
                     "As ActionTermGradient.name, plus constraints (the sum "
                     "of the constraint terms) and action (the whole "
                     "action).")
      .def_readwrite("label", &ActionTermRecord::label)
      .def_readwrite("weight", &ActionTermRecord::weight)
      .def_readwrite("bare", &ActionTermRecord::bare)
      .def_readwrite("factored", &ActionTermRecord::factored)
      .def_readwrite("value", &ActionTermRecord::value)
      .def_readwrite("gradient_norm", &ActionTermRecord::gradientNorm,
                     "The norm of the term's stationarity on the relaxed "
                     "coordinates; for action, the force norm the solve "
                     "reports.");

  m.def("action_term_records", &actionTermRecords, py::arg("action"),
        py::arg("declaration"),
        "Every term of the action with its value and gradient norm on the "
        "coordinates the declaration relaxes (ActionTermRecord).");

  py::class_<HolomorphicStep>(m, "HolomorphicStep",
      "One Newton iteration, recorded so a run can be read back rather than "
      "only its outcome.")
      .def(py::init<>())
      .def_readwrite("iteration", &HolomorphicStep::iteration)
      .def_readwrite("residual_norm", &HolomorphicStep::residualNorm)
      .def_readwrite("step_norm", &HolomorphicStep::stepNorm)
      .def_readwrite("damping", &HolomorphicStep::damping,
                     "One for a full Newton step, a negative power of two "
                     "otherwise.")
      .def_readwrite("jacobian_rank", &HolomorphicStep::jacobianRank,
                     "The number of singular values above rank_tolerance "
                     "times the largest. Below the variable count whenever the "
                     "connection is relaxed, because the action is gauge "
                     "invariant.")
      .def_readwrite("rank_tolerance", &HolomorphicStep::rankTolerance,
                     "The relative tolerance the rank was decided at.")
      .def_readwrite("largest_singular_value",
                     &HolomorphicStep::largestSingularValue)
      .def_readwrite("smallest_retained_singular_value",
                     &HolomorphicStep::smallestRetainedSingularValue,
                     "The smallest singular value counted in the rank.")
      .def_readwrite("largest_discarded_singular_value",
                     &HolomorphicStep::largestDiscardedSingularValue,
                     "The largest singular value counted as zero; 0 when "
                     "none is.")
      .def_readwrite("rank_gap", &HolomorphicStep::rankGap,
                     "The smallest retained over the largest discarded "
                     "singular value; inf when none is discarded.")
      .def_readwrite("action", &HolomorphicStep::action)
      .def_readwrite("sector_guard_dampings",
                     &HolomorphicStep::sectorGuardDampings,
                     "The halvings of this step the held monopole sectors "
                     "forced.")
      .def_readwrite("domain_guard_dampings",
                     &HolomorphicStep::domainGuardDampings,
                     "The halvings of this step forced by trial points at "
                     "which the action refuses to evaluate or its residual "
                     "is not finite.")
      .def_readwrite("residual_test_dampings",
                     &HolomorphicStep::residualTestDampings,
                     "The halvings of this step the residual test forced.")
      .def_readwrite("accepted", &HolomorphicStep::accepted,
                     "Whether a damped step was accepted; when not, damping "
                     "and step_norm are zero and the solve stopped here.")
      .def_readwrite("terms", &HolomorphicStep::terms,
                     "Every term of the action at the point the step ended "
                     "at (ActionTermRecord), when the declaration records "
                     "terms; empty otherwise.")
      .def_readwrite("linear_residual", &HolomorphicStep::linearResidual,
                     "||F + J d|| / ||F|| for the full Newton step d: what the "
                     "linearized equations leave.")
      .def_readwrite("constrained_step", &HolomorphicStep::constrainedStep,
                     "Whether the step was solved on the tangent space of the "
                     "held sectors.")
      .def_readwrite("constrained_rank", &HolomorphicStep::constrainedRank,
                     "The rank of the real least-squares system of a "
                     "constrained step; zero otherwise.")
      .def_readwrite("constrained_rank_gap",
                     &HolomorphicStep::constrainedRankGap,
                     "That system's rank gap; NaN for an unconstrained step.")
      .def_readwrite("action_available", &HolomorphicStep::actionAvailable)
      .def_readwrite("action_unavailable",
                     &HolomorphicStep::actionUnavailable);

  py::class_<HolomorphicRelaxationReport>(m, "HolomorphicRelaxationReport",
      "What a solve reached, and the trace of how it got there.")
      .def(py::init<>())
      .def_readwrite("steps", &HolomorphicRelaxationReport::steps)
      .def_readwrite("converged", &HolomorphicRelaxationReport::converged)
      .def_readwrite("initial_terms",
                     &HolomorphicRelaxationReport::initialTerms,
                     "Every term of the action at the starting point "
                     "(ActionTermRecord), when the declaration records terms.")
      .def_readwrite("initial_residual_norm",
                     &HolomorphicRelaxationReport::initialResidualNorm)
      .def_readwrite("residual_norm",
                     &HolomorphicRelaxationReport::residualNorm)
      .def_readwrite("action", &HolomorphicRelaxationReport::action)
      .def_readwrite("multipliers", &HolomorphicRelaxationReport::multipliers)
      .def_readwrite("moment_residuals",
                     &HolomorphicRelaxationReport::momentResiduals)
      .def_readwrite("sector_guard_damped_steps",
                     &HolomorphicRelaxationReport::sectorGuardDampedSteps)
      .def_readwrite("sector_monopole_numbers",
                     &HolomorphicRelaxationReport::sectorMonopoleNumbers)
      .def_readwrite("held_modulus_drift",
                     &HolomorphicRelaxationReport::heldModulusDrift)
      .def_readwrite("regge_hinge_count",
                     &HolomorphicRelaxationReport::reggeHingeCount,
                     "The number of hinges the primal Regge sum runs over.")
      .def_readwrite("regge_structurally_zero",
                     &HolomorphicRelaxationReport::reggeStructurallyZero,
                     "True when a declared primal Regge term has no hinge on "
                     "this complex under the declared hinge rule, so it and "
                     "its gradient were identically zero throughout the "
                     "solve.")
      .def_readwrite("regge_off_principal_angles",
                     &HolomorphicRelaxationReport::reggeOffPrincipalAngles,
                     "The number of dihedral angles whose continued sheet "
                     "differs from the principal one at the end point.")
      .def_readwrite("stop_reason", &HolomorphicRelaxationReport::stopReason,
                     "Why the solve stopped (RelaxationStop).")
      .def_readwrite("stop_detail", &HolomorphicRelaxationReport::stopDetail,
                     "The stop reason in words, with the numbers that decided "
                     "it.")
      .def_readwrite("largest_length_ratio",
                     &HolomorphicRelaxationReport::largestLengthRatio,
                     "The largest |z_e| at the end over its value at the "
                     "start.")
      .def_readwrite("action_available",
                     &HolomorphicRelaxationReport::actionAvailable)
      .def_readwrite("action_unavailable",
                     &HolomorphicRelaxationReport::actionUnavailable);

  py::class_<HolomorphicRelaxation>(m, "HolomorphicRelaxation",
      "A Newton root find on the holomorphic stationarity equations of a "
      "JointAction: dS/dz = 0, U dS/dU = 0 and p_j(h) = p_j*.\n\n"
      "It solves the equations themselves. It does not minimize the residual "
      "norm, a real part, or any other real projection: the norm appears only "
      "as the quantity the damping compares and as the convergence "
      "certificate. The connection is updated MULTIPLICATIVELY, U -> U e^delta, "
      "which on the stored phase is the exact increment phi -> phi - i delta "
      "and selects no logarithm branch. A new squared length is written back "
      "through the square root taken by continuation from the edge's current "
      "length, so a relaxation path never jumps between the two sheets.\n\n"
      "The Newton system is solved in the minimum-norm sense, because the "
      "action is gauge invariant and its connection block is therefore "
      "singular along every pure-gauge direction; the minimum-norm solution is "
      "the one orthogonal to the gauge orbit.")
      .def(py::init<JointAction, HolomorphicRelaxationDeclaration>(),
           py::arg("action"), py::arg("declaration"))
      .def("solve", &HolomorphicRelaxation::solve,
           "Run the solve, writing the relaxed fields into the complex.")
      .def_property_readonly("action", &HolomorphicRelaxation::action,
                             "The action, carrying the multipliers as the "
                             "solve left them.")
      .def("jacobian", &HolomorphicRelaxation::jacobian,
           "The Jacobian at the current point, flat row-major. Forming it "
           "restores the complex exactly, so the geometry is unchanged.")
      .def("residual", &HolomorphicRelaxation::residual,
           "The residual of the equations in scope at the current point, in "
           "the block order of the Jacobian's rows.")
      .def("equation_count", &HolomorphicRelaxation::equationCount)
      .def("variable_count", &HolomorphicRelaxation::variableCount);
}
