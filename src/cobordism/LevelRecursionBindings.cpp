// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the level recursion. One of
// the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_level_recursion(py::module_ &m) {
  py::enum_<RecursionBandSelection>(m, "RecursionBandSelection",
      "How the band of a response vertex, the set of eigenvalues of its block "
      "whose invariant subspace is the fiber, is selected. The contour is "
      "recorded as the selection, as a centre and a radius, and the projector "
      "onto the band is formed exactly from the block's Schur form. "
      "LowestModes derives the selection from a declared band rank and the "
      "declared occupation order; DeclaredContours takes a centre and a radius "
      "per component and sorts nothing at all.")
      .value("LowestModes", RecursionBandSelection::LowestModes)
      .value("DeclaredContours", RecursionBandSelection::DeclaredContours);

  py::class_<RecursionBandDeclaration>(m, "RecursionBandDeclaration",
      "How every level's fibers are selected.")
      .def(py::init<>())
      .def_readwrite("selection", &RecursionBandDeclaration::selection)
      .def_readwrite("band_rank", &RecursionBandDeclaration::bandRank,
                     "r_v, the number of eigenvalues each contour encloses "
                     "under LowestModes.")
      .def_readwrite("order", &RecursionBandDeclaration::order,
                     "Which eigenvalues the band rank counts under "
                     "LowestModes: ascending real part (then imaginary part) "
                     "or ascending modulus (then real, then imaginary part), "
                     "every key compared at the declared tolerance.")
      .def_readwrite("contour_centres",
                     &RecursionBandDeclaration::contourCentres)
      .def_readwrite("contour_radii", &RecursionBandDeclaration::contourRadii);

  py::class_<LevelRecursionDeclaration>(m, "LevelRecursionDeclaration",
      "Everything that fixes how a recursion is driven. None of it changes "
      "which identities hold.")
      .def(py::init<>())
      .def_readwrite("resolutions", &LevelRecursionDeclaration::resolutions,
                     "The modularity resolutions every level's partition is "
                     "swept over, in scan order.")
      .def_readwrite("modularity_restarts",
                     &LevelRecursionDeclaration::modularityRestarts)
      .def_readwrite("modularity_seed",
                     &LevelRecursionDeclaration::modularitySeed)
      .def_readwrite("persistence_overlap",
                     &LevelRecursionDeclaration::persistenceOverlap)
      .def_readwrite("reference_lambda",
                     &LevelRecursionDeclaration::referenceLambda,
                     "The spectral parameter each level's partition and fibers "
                     "are read at. The partition and the fibers are declared "
                     "structure of a level, not functions of lambda; the "
                     "energy dependence is carried by response_pencil, which "
                     "re-derives the whole chain at whatever lambda it is "
                     "asked for.")
      .def_readwrite("bands", &LevelRecursionDeclaration::bands)
      .def_readwrite("tolerance", &LevelRecursionDeclaration::tolerance,
                     "The relative tolerance the certificates of every level "
                     "hold against and the rank decisions of the band reads "
                     "are made at.")
      .def_readwrite("dense_crossover",
                     &LevelRecursionDeclaration::denseCrossover);

  py::class_<RecursionBandRead>(m, "RecursionBandRead",
      "One certified fiber E_v = Ran P_v, the selection that named its band, "
      "and the certificates of the exact projector. P_v is the spectral "
      "projector onto the invariant subspace of the selected eigenvalues, "
      "formed from the complex Schur form of the component's block: the "
      "selected eigenvalues are reordered to the leading block, the Sylvester "
      "equation for the invariant subspace is solved, and P_v = Phi_v "
      "PhiTilde_v^T with Phi_v the leading Schur vectors. For a diagonalizable "
      "block this is V_B (V^-1)_B over the selected eigenvalues. A selection "
      "that encloses every eigenvalue has the identity as its projector and "
      "the canonical basis as both frames, exactly.")
      .def(py::init<>())
      .def_readwrite("component", &RecursionBandRead::component)
      .def_readwrite("rank", &RecursionBandRead::rank,
                     "r_v, the number of selected eigenvalues counted with "
                     "multiplicity, and the rank of the projector.")
      .def_readwrite("contour_centre", &RecursionBandRead::contourCentre,
                     "The centre of the recorded selection: the mean of the "
                     "selected eigenvalues under LowestModes, the declared "
                     "centre under DeclaredContours.")
      .def_readwrite("contour_radius", &RecursionBandRead::contourRadius,
                     "The radius of the recorded selection: halfway between "
                     "the farthest selected and the nearest excluded "
                     "eigenvalue under LowestModes, infinite when nothing is "
                     "excluded; the declared radius under DeclaredContours.")
      .def_readwrite("encloses_everything",
                     &RecursionBandRead::enclosesEverything,
                     "Whether the selection excludes no eigenvalue of the "
                     "block, so that the fiber is the whole of the component.")
      .def_readwrite("eigenvalues", &RecursionBandRead::eigenvalues,
                     "The eigenvalues of the component's block the selection "
                     "encloses.")
      .def_readwrite("isolation_gap", &RecursionBandRead::isolationGap,
                     "The smallest distance between a selected and an excluded "
                     "eigenvalue of the block; infinite when nothing is "
                     "excluded. A selection separating two eigenvalues equal "
                     "at the declared tolerance is refused by name.")
      .def_readwrite("projector_idempotency",
                     &RecursionBandRead::projectorIdempotency,
                     "||P_v^2 - P_v||_F / ||P_v||_F, the rounding residual of "
                     "the exact projector's idempotency.")
      .def_readwrite("pairing_defect", &RecursionBandRead::pairingDefect,
                     "||PhiTilde^T Phi - I||, the bilinear pairing of the two "
                     "frames.")
      .def_readwrite("invariant_subspace_residual",
                     &RecursionBandRead::invariantSubspaceResidual,
                     "||h_v Phi_v - Phi_v (PhiTilde_v^T h_v Phi_v)||_F / "
                     "||h_v||_F, the residual of the invariant subspace.")
      .def_readwrite("frame", &RecursionBandRead::frame,
                     "Phi_v over the level's coordinates, flat row-major.")
      .def_readwrite("left_frame", &RecursionBandRead::leftFrame,
                     "PhiTilde_v^T, flat row-major.")
      .def_readwrite("accepted", &RecursionBandRead::accepted)
      .def_readwrite("certificate", &RecursionBandRead::certificate);

  py::class_<LevelTransport>(m, "LevelTransport",
      "One block M_vw = PhiTilde_v^T T_vw Phi_w: the level's coupling between "
      "two response vertices, read in their own fibers.")
      .def(py::init<>())
      .def_readwrite("from_component", &LevelTransport::from)
      .def_readwrite("to_component", &LevelTransport::to)
      .def_readwrite("block", &LevelTransport::block);

  py::class_<RecursionLevelRead>(m, "RecursionLevelRead",
      "One turn of the whitepaper's box: the partition, the fibers, the "
      "labeled sum and the response pencil of the level above.")
      .def(py::init<>())
      .def_readwrite("level", &RecursionLevelRead::level)
      .def_readwrite("dimension", &RecursionLevelRead::dimension)
      .def_readwrite("partition", &RecursionLevelRead::partition)
      .def_readwrite("resolutions", &RecursionLevelRead::resolutions)
      .def_readwrite("selected_resolution",
                     &RecursionLevelRead::selectedResolution)
      .def_readwrite("component_persistence",
                     &RecursionLevelRead::componentPersistence)
      .def_readwrite("worst_persistence_overlap",
                     &RecursionLevelRead::worstPersistenceOverlap)
      .def_readwrite("bands", &RecursionLevelRead::bands)
      .def_readwrite("embedding", &RecursionLevelRead::embedding,
                     "Y_{l+1}, the fibers' right frames side by side.")
      .def_readwrite("dual_embedding", &RecursionLevelRead::dualEmbedding)
      .def_readwrite("gram", &RecursionLevelRead::gram,
                     "G_{l+1} = YTilde^T Y, the bilinear overlap of the "
                     "labeled sum. An off-diagonal block is nonzero exactly "
                     "when two fibers' supports meet, which is why the "
                     "internal sum is never asserted to be direct.")
      .def_readwrite("gram_defect", &RecursionLevelRead::gramDefect)
      .def_readwrite("modes", &RecursionLevelRead::modes)
      .def_readwrite("fiber_operator", &RecursionLevelRead::fiberOperator,
                     "h^{l+1} = YTilde^T R_l(lambda_ref) Y.")
      .def_readwrite("fiber_spectrum", &RecursionLevelRead::fiberSpectrum)
      .def_readwrite("transports", &RecursionLevelRead::transports)
      .def_readwrite("fock_stage_dimension",
                     &RecursionLevelRead::fockStageDimension,
                     "2^M, the Fock stage the level's one-particle space "
                     "carries. The vector is never allocated.")
      .def_readwrite("vacuum_embedded_modes",
                     &RecursionLevelRead::vacuumEmbeddedModes,
                     "The modes the interaction stage adds beyond the ones the "
                     "reduction retained, which the state reaches by the "
                     "vacuum embedding of Section 12.")
      .def_readwrite("response_dimension",
                     &RecursionLevelRead::responseDimension)
      .def_readwrite("determinant_residual",
                     &RecursionLevelRead::determinantResidual,
                     "The determinant factorization that makes the step exact, "
                     "measured at the reference lambda.")
      .def_readwrite("reduction_certificate",
                     &RecursionLevelRead::reductionCertificate)
      .def_readwrite("certificate", &RecursionLevelRead::certificate);

  py::class_<LevelRecursion>(m, "LevelRecursion",
      "The master recursive construction of Section 15, driven level by level "
      "with the energy dependence kept exact.\n\n"
      "Every scale runs the whitepaper's box: the partition is discovered over "
      "a declared sweep of modularity resolutions, each component's fiber is "
      "the range of the exact Riesz projector of its own block onto the band "
      "its declared selection names, the next level's response pencil is the "
      "exact Feshbach map of the one below, and the fibers are summed into "
      "the labeled sum with its bilinear overlap carried exactly.\n\n"
      "R_l is a function of lambda and never a matrix frozen at one value of "
      "it: response_pencil re-derives the whole chain from the microscopic "
      "pencil A - lambda M, applying each level's declared partition in turn "
      "as a plain supported block elimination with no further shift, because "
      "the spectral parameter is already inside the matrix being eliminated. "
      "The step is exact by the determinant factorization det R_l = det(R_l)_II "
      "det R_{l+1}, so every level's spectrum is reproduced from the level "
      "below it. Nothing is linearized, and no square root or polar "
      "projection enters the recursion.")
      .def_static("read_band", &LevelRecursion::readBand, py::arg("block"),
                  py::arg("order"), py::arg("bands"), py::arg("component"),
                  py::arg("tolerance"),
                  "The exact band read of one block, flat row-major of the "
                  "given order: the selection the declaration names for the "
                  "component, the Riesz projector onto the invariant subspace "
                  "of the selected eigenvalues from the block's complex Schur "
                  "form, its frames over the block's own coordinates, and its "
                  "certificates. Rank decisions are made at the tolerance, and "
                  "a selection that names no invariant subspace is refused by "
                  "name.")
      .def_static("over_pencil", &LevelRecursion::overPencil, py::arg("pencil"),
                  py::arg("metric"), py::arg("dimension"),
                  py::arg("declaration"),
                  "Build over an explicit pencil (A, M), flat row-major. An "
                  "empty metric is the identity, so the microscopic pencil is "
                  "A - lambda I.")
      .def_static("over_spacetime", &LevelRecursion::overSpacetime,
                  py::arg("spacetime"), py::arg("degree"),
                  py::arg("metric_source"), py::arg("declaration"),
                  "Build over a triangulation's Hodge operator at a degree, "
                  "which at degree one is the whitepaper's microscopic "
                  "edge-mode response pencil.")
      .def_property_readonly("declaration", &LevelRecursion::declaration)
      .def("base_dimension", &LevelRecursion::baseDimension)
      .def("level_count", &LevelRecursion::levelCount,
           "How many turns of the box have been taken.")
      .def("advance", &LevelRecursion::advance, "Take one turn of the box.")
      .def("advance_to", &LevelRecursion::advanceTo, py::arg("levels"),
           "Take turns until level_count reaches the named number.")
      .def("level", &LevelRecursion::level, py::arg("index"),
           py::return_value_policy::reference_internal,
           "One completed turn of the box.")
      .def("component_of_coordinate", &LevelRecursion::componentOfCoordinate,
           py::arg("index"),
           "The component of P_l each coordinate of the level belongs to: the "
           "reduction map, as a map on coordinates. It is the map "
           "MappingCylinder builds W^l over when the level's coordinates are "
           "the vertices of K^l.")
      .def("response_pencil", &LevelRecursion::responsePencil,
           py::arg("level"), py::arg("lambda_"),
           "R_l(lambda), flat row-major: the exact energy-dependent response "
           "pencil of the level, re-derived from the microscopic pencil at "
           "this lambda.")
      .def("response_dimension", &LevelRecursion::responseDimension,
           py::arg("level"))
      .def("response_determinant", &LevelRecursion::responseDeterminant,
           py::arg("level"), py::arg("lambda_"),
           "det R_l(lambda): the function whose zeros, together with the "
           "interior determinants the chain eliminated, are the microscopic "
           "spectrum.")
      .def("interior_determinant", &LevelRecursion::interiorDeterminant,
           py::arg("level"), py::arg("lambda_"),
           "det (R_l(lambda))_II, the determinant of the interior block the "
           "step from this level eliminates.")
      .def("determinant_factorization_residual",
           &LevelRecursion::determinantFactorizationResidual, py::arg("level"),
           py::arg("lambda_"),
           "The identity that makes the step exact, measured at a spectral "
           "parameter of the caller's choosing.")
      .def("record_vacuum_embedded_modes",
           &LevelRecursion::recordVacuumEmbeddedModes, py::arg("index"),
           py::arg("modes"),
           "Record that the interaction stage attached new one-particle modes "
           "to the level, which the carried state reaches by the vacuum "
           "embedding. The reduction alone grows nothing.");
}
