// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the dressed fluctuation and
// the mapping cylinder. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_dressed_fluctuation(py::module_ &m) {
  py::class_<DressedFluctuationDeclaration>(m, "DressedFluctuationDeclaration",
      "Everything that fixes which fluctuation problem a DressedFluctuation "
      "is: the carrier h0, the couplings O_a = dh/df_a, the second "
      "derivatives the diamagnetic term is built from, the bare stiffness A of "
      "the fluctuation's own action, and the occupation rule. Plain data.")
      .def(py::init<>())
      .def_readwrite("carrier_dimension",
                     &DressedFluctuationDeclaration::carrierDimension,
                     "n, the dimension of the one-particle carrier space.")
      .def_readwrite("carrier", &DressedFluctuationDeclaration::carrier,
                     "h0, the carrier operator at the stationary "
                     "configuration, flat row-major n by n. It is never "
                     "symmetrized and no adjoint of it is taken.")
      .def_readwrite("couplings", &DressedFluctuationDeclaration::couplings,
                     "O_a = dh/df_a for each retained fluctuation, each flat "
                     "row-major n by n. Their number is R.")
      .def_readwrite("second_derivatives",
                     &DressedFluctuationDeclaration::secondDerivatives,
                     "d^2h/df_a df_b for the pairs a <= b in row-major "
                     "upper-triangular order, each flat row-major n by n. "
                     "Empty declares a carrier that depends on the "
                     "fluctuations linearly, so the diamagnetic term is zero.")
      .def_readwrite("bare_stiffness",
                     &DressedFluctuationDeclaration::bareStiffness,
                     "A, the bare stiffness of the fluctuation's own action, "
                     "flat row-major R by R and complex symmetric. Empty "
                     "declares a zero bare stiffness.")
      .def_readwrite("occupied_modes",
                     &DressedFluctuationDeclaration::occupiedModes,
                     "How many modes of h0 the quasi-free state occupies.")
      .def_readwrite("occupation_order",
                     &DressedFluctuationDeclaration::occupationOrder,
                     "Which modes those are.")
      .def_readwrite("continuum_broadening",
                     &DressedFluctuationDeclaration::continuumBroadening,
                     "eta >= 0, a declared finite lifetime given to every "
                     "particle-hole excitation, replacing each excitation "
                     "energy by Delta - i eta. Zero is the whitepaper's "
                     "formula exactly; a positive value is the one "
                     "approximation of this class and is never applied unless "
                     "asked for.")
      .def_readwrite("tolerance", &DressedFluctuationDeclaration::tolerance,
                     "The relative tolerance the certificates hold against.");

  py::class_<CollectiveMode>(m, "CollectiveMode",
      "One pole of the dressed fluctuation propagator A_eff(w)^-1: a frequency "
      "at which the dressed stiffness is singular, with the fluctuation "
      "direction that is soft there. These are the gauge quanta of Section 7.")
      .def(py::init<>())
      .def_readwrite("frequency", &CollectiveMode::frequency,
                     "w, the frequency at which det A_eff(w) = 0. The dressed "
                     "stiffness depends on w only through w^2, so the poles "
                     "come in pairs and both members are reported.")
      .def_readwrite("polarization", &CollectiveMode::polarization,
                     "The unit null direction of A_eff(w), length R: the "
                     "combination of retained fluctuations that is soft here.")
      .def_readwrite("radiation_rate", &CollectiveMode::radiationRate,
                     "Gamma = -2 Im w, the rate at which the mode's "
                     "occupation relaxes into the particle-hole continuum. "
                     "Zero for a real frequency, which is a mode that does not "
                     "radiate.")
      .def_readwrite("continuum_distance", &CollectiveMode::continuumDistance,
                     "The distance from the nearest bare particle-hole "
                     "excitation energy or its negative.")
      .def_readwrite("inside_particle_hole_continuum",
                     &CollectiveMode::insideParticleHoleContinuum,
                     "Whether |Re w| lies between the smallest and the largest "
                     "Re Delta: the mode is then degenerate with the "
                     "particle-hole excitations rather than bound outside "
                     "them.")
      .def_readwrite("residual", &CollectiveMode::residual,
                     "The relative residual of the null-vector equation this "
                     "mode solves.")
      .def_readwrite("certificate", &CollectiveMode::certificate);

  py::class_<ManyBodySpaceRead>(m, "ManyBodySpaceRead",
      "The effective action of the exact elimination, evaluated on the "
      "N-particle space of a cluster. The basis is the ascending N-element "
      "subsets of the fiber's modes in lexicographic order.")
      .def(py::init<>())
      .def_readwrite("particles", &ManyBodySpaceRead::particles)
      .def_readwrite("fiber_rank", &ManyBodySpaceRead::fiberRank,
                     "r, the rank of the cluster's fiber.")
      .def_readwrite("dimension", &ManyBodySpaceRead::dimension,
                     "r choose N, the order of every matrix below.")
      .def_readwrite("basis", &ManyBodySpaceRead::basis,
                     "The occupation basis: one ascending N-tuple of fiber "
                     "mode indices per dimension, in lexicographic order.")
      .def_readwrite("one_body", &ManyBodySpaceRead::oneBody,
                     "psiTilde^T h0 psi on this space, flat row-major.")
      .def_readwrite("quartic", &ManyBodySpaceRead::quartic,
                     "-1/2 J^T A^-1 J on this space, flat row-major: the whole "
                     "second term of the exact elimination.")
      .def_readwrite("induced_one_body", &ManyBodySpaceRead::inducedOneBody,
                     "The part of the quartic that is a one-body operator, "
                     "-1/2 dGamma(sum_ab (A^-1)_ab O_a O_b), which reordering "
                     "the product of two currents produces.")
      .def_readwrite("normal_ordered_quartic",
                     &ManyBodySpaceRead::normalOrderedQuartic,
                     "The quartic minus its one-body part: the strictly "
                     "quartic, normal-ordered interaction, which is the term "
                     "that takes the state outside the Gaussian class.")
      .def_readwrite("effective_action", &ManyBodySpaceRead::effectiveAction,
                     "S_eff = psiTilde^T h0 psi - 1/2 J^T A^-1 J on this "
                     "space, flat row-major.")
      .def_readwrite("frame_pairing_defect",
                     &ManyBodySpaceRead::framePairingDefect,
                     "||PhiTilde^T Phi - I|| of the declared cluster frames.")
      .def_readwrite("stiffness_asymmetry",
                     &ManyBodySpaceRead::stiffnessAsymmetry,
                     "||A - A^T||_F / ||A||_F of the bare stiffness.")
      .def_readwrite("stiffness_conditioning",
                     &ManyBodySpaceRead::stiffnessConditioning)
      .def_readwrite("certificate", &ManyBodySpaceRead::certificate);

  py::class_<DressedFluctuation>(m, "DressedFluctuation",
      "The dressed fluctuation propagator of Section 7, its poles, and the "
      "exact elimination of the fluctuation.\n\n"
      "A retained fluctuation f in C^R of the geometry or of the connection "
      "couples linearly to the fermion bilinear currents J_a = psiTilde^T O_a "
      "psi with O_a = dh/df_a. In a quasi-free state its stiffness is dressed "
      "to A_eff(w) = A + D - Pi(w), with D the diamagnetic term and Pi the "
      "paramagnetic polarization. Every bracket is the complex bilinear one "
      "taken between the left and right modes of h0, so no adjoint appears "
      "anywhere and a non-normal carrier is carried as it stands. At w = 0 the "
      "induced stiffness D - Pi(0) is the Hessian of the occupied energy and "
      "vanishes on every pure-gauge direction, which is the Ward identity. The "
      "poles of A_eff(w)^-1 are the collective modes the whitepaper names the "
      "gauge quanta, and they are found exactly by a linear pencil rather than "
      "by sampling w. Eliminating f at its saddle gives S_eff = psiTilde^T h0 "
      "psi - 1/2 J^T A^-1 J, whose second term is evaluated on the "
      "three-particle space of a cluster.")
      .def(py::init<DressedFluctuationDeclaration>(), py::arg("declaration"))
      .def_property_readonly("declaration", &DressedFluctuation::declaration,
                             "The declaration this instance was built from.")
      .def("carrier_dimension", &DressedFluctuation::carrierDimension)
      .def("fluctuation_count", &DressedFluctuation::fluctuationCount,
           "R, the number of retained fluctuations.")
      .def("particle_hole_pair_count",
           &DressedFluctuation::particleHolePairCount,
           "P, the occupied mode count times the empty mode count.")
      .def("carrier_eigenvalues", &DressedFluctuation::carrierEigenvalues,
           "The eigenvalues of h0 in the declared occupation order, so the "
           "first occupied_modes entries are the occupied ones.")
      .def("particle_hole_energies", &DressedFluctuation::particleHoleEnergies,
           "The bare excitation energies Delta = lambda_n - lambda_m, one per "
           "pair of an occupied m with an empty n, occupied-major.")
      .def("mode_currents", &DressedFluctuation::modeCurrents, py::arg("index"),
           "V^-1 O_a V, the current matrix of one fluctuation in the carrier's "
           "mode basis, flat row-major in the declared occupation order.")
      .def("diamagnetic", &DressedFluctuation::diamagnetic,
           "D, the diamagnetic term, flat row-major R by R.")
      .def("paramagnetic", &DressedFluctuation::paramagnetic,
           py::arg("frequency") = std::complex<double>{0.0, 0.0},
           "Pi(w), the paramagnetic polarization, flat row-major R by R.")
      .def("dressed_stiffness", &DressedFluctuation::dressedStiffness,
           py::arg("frequency") = std::complex<double>{0.0, 0.0},
           "A_eff(w) = A + D - Pi(w), flat row-major R by R.")
      .def("induced_stiffness", &DressedFluctuation::inducedStiffness,
           "D - Pi(0), the whole fermion contribution to the dressed "
           "stiffness, which is the Hessian of the occupied energy.")
      .def("ward_residual", &DressedFluctuation::wardResidual,
           py::arg("gauge_direction"),
           "||(D - Pi(0)) g|| / (||D - Pi(0)|| ||g||) on one declared "
           "pure-gauge direction.")
      .def("ward_certificate", &DressedFluctuation::wardCertificate,
           py::arg("gauge_directions"),
           "The Ward identity over the declared pure-gauge directions: the "
           "induced stiffness vanishes on every one of them because the "
           "carrier moves along one by a similarity transformation, which "
           "leaves its eigenvalues where they were.")
      .def("collective_modes", &DressedFluctuation::collectiveModes,
           "The poles of A_eff(w)^-1, ascending by (Re w, Im w). A candidate "
           "whose geometric component vanishes is an uncoupled particle-hole "
           "excitation rather than a pole and is not reported.")
      .def("effective_action", &DressedFluctuation::effectiveAction,
           py::arg("cluster_frame"), py::arg("cluster_dual_frame"),
           py::arg("particles") = std::size_t{3},
           py::arg("dimension_cap") =
               DressedFluctuation::kDefaultManyBodyDimensionCap,
           "S_eff on the N-particle space of a cluster fiber. The frames are "
           "the fiber's right frame (n by r) and its algebraic dual (r by n); "
           "empty frames declare the whole carrier space.");

  py::class_<MappingCylinderDeclaration>(m, "MappingCylinderDeclaration",
      "The three pieces of data one tick of the recursion is built from: the "
      "top cells of K^l, the reduction map on its vertices, and the top cells "
      "of K^{l+1}. Plain data.")
      .def(py::init<>())
      .def_readwrite("incoming_top_cells",
                     &MappingCylinderDeclaration::incomingTopCells,
                     "The top cells of K^l, each a tuple of vertex "
                     "identifiers. The list must be pure.")
      .def_readwrite("reduction_map",
                     &MappingCylinderDeclaration::reductionMap,
                     "r: for every vertex of K^l, the response vertex of "
                     "R^{l+1} it is carried to. Its identifiers must be "
                     "disjoint from the incoming ones.")
      .def_readwrite("outgoing_top_cells",
                     &MappingCylinderDeclaration::outgoingTopCells,
                     "The top cells of K^{l+1}, each a tuple of response "
                     "vertices. The interactions of Section 6 supply these; "
                     "the reduction supplies none of them.");

  py::class_<MappingCylinderRead>(m, "MappingCylinderRead",
      "The interaction cobordism W^l and the two complexes it runs between, "
      "with the checks that identify its boundary.")
      .def(py::init<>())
      .def_readwrite("incoming_dimension",
                     &MappingCylinderRead::incomingDimension)
      .def_readwrite("cylinder_dimension",
                     &MappingCylinderRead::cylinderDimension,
                     "One more than the incoming dimension: the fibering "
                     "direction is the extra simplex dimension and the only "
                     "reason the cylinder has one.")
      .def_readwrite("outgoing_dimension",
                     &MappingCylinderRead::outgoingDimension)
      .def_readwrite("incoming_vertices",
                     &MappingCylinderRead::incomingVertices)
      .def_readwrite("response_vertices",
                     &MappingCylinderRead::responseVertices)
      .def_readwrite("cylinder_top_cells",
                     &MappingCylinderRead::cylinderTopCells,
                     "The staircase simplices of the prism over each incoming "
                     "top cell, with the degenerate ones dropped.")
      .def_readwrite("fiber_edges", &MappingCylinderRead::fiberEdges,
                     "The edges (v, r(v)), one per vertex of K^l. They are the "
                     "only timelike edges of the history.")
      .def_readwrite("cross_edges", &MappingCylinderRead::crossEdges,
                     "Every edge of W^l with one endpoint at each end: the "
                     "fiber edges plus the prism diagonals.")
      .def_readwrite("image_top_cells", &MappingCylinderRead::imageTopCells,
                     "The maximal image cells r(sigma): the image complex the "
                     "outgoing end carries before K^{l+1} is attached.")
      .def_readwrite("incoming_free_facets",
                     &MappingCylinderRead::incomingFreeFacets)
      .def_readwrite("outgoing_free_facets",
                     &MappingCylinderRead::outgoingFreeFacets)
      .def_readwrite("side_free_facets",
                     &MappingCylinderRead::sideFreeFacets)
      .def_readwrite("incoming_boundary_is_the_incoming_complex",
                     &MappingCylinderRead::incomingBoundaryIsTheIncomingComplex)
      .def_readwrite("outgoing_complex_contains_the_image",
                     &MappingCylinderRead::outgoingComplexContainsTheImage)
      .def_readwrite("has_no_side_wall", &MappingCylinderRead::hasNoSideWall)
      .def_readwrite("boundary_is_the_disjoint_union",
                     &MappingCylinderRead::boundaryIsTheDisjointUnion,
                     "dW^l = K^l disjoint union K^{l+1}.")
      .def_readwrite("boundary_residual",
                     &MappingCylinderRead::boundaryResidual)
      .def_readwrite("certificate", &MappingCylinderRead::certificate);

  py::class_<MappingCylinder>(m, "MappingCylinder",
      "The interaction cobordism W^l of Section 3: one tick of time, realized "
      "as the mapping cylinder of the reduction map from K^l onto the vertex "
      "set of R^{l+1}, with the cells of K^{l+1} attached on its outgoing "
      "end.\n\n"
      "The cylinder contains K^l, the fiber edges joining each certified "
      "cluster's cells to its response vertex, and K^{l+1}, with dW^l = K^l "
      "disjoint union K^{l+1}. Time is not a simplex dimension of any level; "
      "it is the fourth simplex dimension of W^l only because W^l is built to "
      "realize the fibering direction geometrically. The reduction determines "
      "no incidence maps of its own, so K^{l+1} is declared rather than "
      "derived and the class checks that the image of the reduction lands "
      "inside it.")
      .def(py::init<MappingCylinderDeclaration>(), py::arg("declaration"))
      .def_property_readonly("declaration", &MappingCylinder::declaration)
      .def("read", &MappingCylinder::read,
           py::return_value_policy::reference_internal,
           "The cylinder, its two ends and the boundary checks.");
}
