// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: boundary fibers and pencil
// layers. One of the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_pencil(py::module_ &m) {
  // === MultiCobordism: the C++ fully-emergent merge optimizer, emergent
  // topology at a user-defined degree k. ===

  py::class_<BoundaryFiber>(m, "BoundaryFiber",
      R"doc(The fiber form of a boundary block's target: a retained fiber on the
block's degree-k cells (images Z_B, dual images, Gram Z_B^T M_BB Z_B, the band's eigenvalue,
contour, certificate, and the Lorentzian rotation epsilon).)doc")
      .def(py::init<>())
      .def_readwrite("degree", &BoundaryFiber::degree)
      .def_readwrite("cells", &BoundaryFiber::cells)
      .def_readwrite("images", &BoundaryFiber::images)
      .def_readwrite("dual_images", &BoundaryFiber::dualImages)
      .def_readwrite("gram", &BoundaryFiber::gram)
      .def_readwrite("full_gram", &BoundaryFiber::fullGram)
      .def_readwrite("eigenvalue", &BoundaryFiber::eigenvalue)
      .def_readwrite("contour", &BoundaryFiber::contour)
      .def_readwrite("certificate", &BoundaryFiber::certificate)
      .def_readwrite("epsilon", &BoundaryFiber::epsilon)
      .def("rank", &BoundaryFiber::rank);

  py::class_<AssembledPencil>(m, "AssembledPencil",
      "A glued pencil: the union of cobordisms' top cells with one geometry, "
      "assembled per top simplex, with the shared cells and the one epsilon recorded.")
      .def_property_readonly("complex", [](const AssembledPencil &a) { return a.complex(); })
      .def_property_readonly("lengths", [](const AssembledPencil &a) { return a.lengths; })
      .def_property_readonly("epsilon", [](const AssembledPencil &a) { return a.epsilon; })
      .def_property_readonly("pieces", [](const AssembledPencil &a) { return a.pieces; })
      .def_property_readonly("shared_cells", [](const AssembledPencil &a) { return a.sharedCells; })
      .def_property_readonly("op", [](const AssembledPencil &a) { return *a.op; })
      .def_property_readonly("dual", [](const AssembledPencil &a) { return *a.dual; })
      .def("dimension", &AssembledPencil::dimension)
      .def("cell_index", &AssembledPencil::cellIndex, py::arg("k"), py::arg("cell"));

  py::class_<BorderedPencil>(m, "BorderedPencil",
      "The bordered form of the degree-k pencil at a shift: degree-k cells then degree-(k-1) "
      "cells; its Schur complement over the lower block is lambda M_k - A~_k. Assembled per top "
      "simplex, so it composes exactly across glued cobordisms.")
      .def_readonly("degree", &BorderedPencil::degree)
      .def_readonly("lambda_", &BorderedPencil::lambda)
      .def_readonly("upper_count", &BorderedPencil::upperCount)
      .def_readonly("lower_count", &BorderedPencil::lowerCount)
      .def_readonly("matrix", &BorderedPencil::matrix);

  py::class_<FiberLevel>(m, "FiberLevel",
      "A pencil level whose interface coordinates are retained fibers: the Feshbach "
      "reduction onto the fibers' cells restricted to the fibers, with J, J~ = Z, and the Gram.")
      .def_readonly("degree", &FiberLevel::degree)
      .def_readonly("lambda_", &FiberLevel::lambda)
      .def_readonly("interface_cells", &FiberLevel::interfaceCells)
      .def_readonly("interior_cells", &FiberLevel::interiorCells)
      .def_readonly("response", &FiberLevel::response)
      .def_readonly("J", &FiberLevel::J)
      .def_readonly("j_dual", &FiberLevel::Jdual)
      .def_readonly("restriction", &FiberLevel::restriction)
      .def_readonly("constraint_gram", &FiberLevel::constraintGram)
      .def_readonly("block_offsets", &FiberLevel::blockOffsets)
      .def_readonly("block_ranks", &FiberLevel::blockRanks)
      .def_readonly("fibers_disjoint", &FiberLevel::fibersDisjoint);

  py::class_<PencilLayer::RestrictionRead>(m, "RestrictionRead",
      "The whole's degree-1 zero mode read on every marking at once: ONE harmonic basis "
      "`images` (n_1 x rank, in the assembled pencil's degree-1 cell order) and its periods "
      "over each marking (`periods[i]`, |marking_i| x rank) from that same basis. `monodromy` "
      "is the two-marking case with the fit added. Betti numbers of the whole come with it; "
      "a read that cannot be made names its `obstruction` instead of guessing.")
      .def_readonly("betti", &PencilLayer::RestrictionRead::betti)
      .def_readonly("harmonic_rank", &PencilLayer::RestrictionRead::harmonicRank)
      .def_readonly("images", &PencilLayer::RestrictionRead::images)
      .def_readonly("periods", &PencilLayer::RestrictionRead::periods)
      .def_readonly("frame", &PencilLayer::RestrictionRead::frame,
                    "The band's frame on chains, n_1 x rank; `images` = G_1^U frame.")
      .def_readonly("certificate", &PencilLayer::RestrictionRead::certificate,
                    "The band's BandCertificate: node count, idempotency, rank, gap, resolvent bound.")
      .def_readonly("obstruction", &PencilLayer::RestrictionRead::obstruction);
  py::class_<PencilLayer::MonodromyRead>(m, "MonodromyRead",
      "The monodromy of a drawn cobordism between two marked surfaces: the "
      "whole's degree-1 ZERO MODE (the harmonic contour of the Whitney pencil, bulk and "
      "boundary edges in one operator) read on both markings' edges, its periods "
      "periods_a (|A| x rank) and periods_b, and the matrix M with P_B = M P_A "
      "(M = P_B P_A^-1 for a rank-2 zero mode, least squares otherwise), with its integer "
      "rounding and rounding residual; Betti numbers of the whole come with it. A read that "
      "cannot be made names its `obstruction` instead of guessing.")
      .def_readonly("betti", &PencilLayer::MonodromyRead::betti)
      .def_readonly("harmonic_rank", &PencilLayer::MonodromyRead::harmonicRank)
      .def_readonly("periods_a", &PencilLayer::MonodromyRead::periodsA)
      .def_readonly("periods_b", &PencilLayer::MonodromyRead::periodsB)
      .def_readonly("monodromy", &PencilLayer::MonodromyRead::monodromy)
      .def_readonly("rounded", &PencilLayer::MonodromyRead::rounded)
      .def_readonly("rounding_residual", &PencilLayer::MonodromyRead::roundingResidual)
      .def_readonly("fit_residual", &PencilLayer::MonodromyRead::fitResidual)
      .def_readonly("obstruction", &PencilLayer::MonodromyRead::obstruction);

  py::class_<PencilLayer>(m, "PencilLayer",
      R"doc(Continuation of a relaxed cobordism's boundary fibers into the next pencil level: exact assembly of cobordisms along shared cells (one epsilon per assembly), boundary
responses and their star-product composition, fiber reads from certified Riesz bands, the
next level with the Gram carried exactly, and fiber-to-fiber transfer with the reversal
assertion. Every pairing is the transpose.)doc")
      // The chainhodge enums are registered after this submodule, so their
      // defaults are resolved at call time through std::optional.
      .def_static("assemble",
           [](const std::vector<std::shared_ptr<Spacetime>> &pieces, const std::vector<double> &epsilons,
              std::optional<chainhodge::Branch> branch, int crossover) {
             return PencilLayer::assemble(pieces, epsilons,
                                          branch.value_or(chainhodge::Branch::Continuation), crossover);
           },
           py::arg("pieces"), py::arg("epsilons") = std::vector<double>{},
           py::arg("branch") = py::none(),
           py::arg("crossover_dimension") = std::numeric_limits<int>::max())
      .def_static("assembly_residual",
           [](const AssembledPencil &a, int k, std::optional<chainhodge::Branch> branch) {
             return PencilLayer::assemblyResidual(a, k, branch.value_or(chainhodge::Branch::Continuation));
           },
           py::arg("assembled"), py::arg("k"), py::arg("branch") = py::none())
      .def_static("cells_within", &PencilLayer::cellsWithin, py::arg("assembled"), py::arg("k"), py::arg("vertices"))
      .def_static("indices_of", &PencilLayer::indicesOf, py::arg("assembled"), py::arg("k"), py::arg("cells"))
      .def_static("boundary_response", &PencilLayer::boundaryResponse, py::arg("assembled"), py::arg("k"),
           py::arg("interface"), py::arg("lambda_"))
      .def_static("bordered_pencil", &PencilLayer::borderedPencil, py::arg("assembled"), py::arg("k"), py::arg("lambda_"))
      .def_static("bordered_response", &PencilLayer::borderedResponse, py::arg("assembled"), py::arg("k"),
           py::arg("upper_interface"), py::arg("lower_interface"), py::arg("lambda_"))
      .def_static("upper_response", &PencilLayer::upperResponse, py::arg("bordered"), py::arg("upper_count"))
      .def_static("compose_responses", &PencilLayer::composeResponses, py::arg("left"), py::arg("left_cells"),
           py::arg("right"), py::arg("right_cells"))
      .def_static("harmonic_contour", &PencilLayer::harmonicContour, py::arg("assembled"), py::arg("k"),
           py::arg("node_count") = 64)
      .def_static("band_contour", &PencilLayer::bandContour, py::arg("assembled"), py::arg("k"),
           py::arg("band_index"), py::arg("node_count") = 64,
           "A circle around the band_index-th distinct eigenvalue cluster (by modulus; 0 is the harmonic "
           "cluster when zero is an eigenvalue, 1 the lowest band above it), radius a quarter of the gap.")
      .def_static("monodromy", &PencilLayer::monodromy, py::arg("spacetime"), py::arg("marking_a"),
           py::arg("marking_b"), py::call_guard<py::gil_scoped_release>(),
           "The MonodromyRead of `spacetime` between two markings given in host vertex ids as "
           "cycles of directed steps (u, v): a step contributes +h(u,v) when u < v and -h(u,v) "
           "otherwise. Read-only; the Whitney pencil's harmonic contour is the zero mode.")
      .def_static("restriction", &PencilLayer::restriction, py::arg("spacetime"), py::arg("markings"),
           py::call_guard<py::gil_scoped_release>(),
           "The RestrictionRead of `spacetime` over a list of markings, each given in host vertex "
           "ids as cycles of directed steps (u, v) with the same step convention as `monodromy`: "
           "one harmonic basis and every marking's periods from it. Read-only.")
      .def_static("dual_frame", &PencilLayer::dualFrame, py::arg("complex"), py::arg("degree"),
           py::arg("cells"), py::arg("images"),
           "The dual of a frame under the transpose pairing of `complex`'s own chain-level Whitney "
           "pencil at `degree`: with M_k the pencil's Whitney mass matrix (CovariantChainHodge.m_inv, "
           "PencilLayer.pencil(...).B) restricted to `cells` and B = Z.T @ M_k @ Z the frame's pairing, "
           "Z^vee = Z @ inv(B).T, so that Z^vee.T @ M_k @ Z == I (the BlockFrame contract; the "
           "canonical left frame at U = 1). Read-only on the complex. Refuses by name a "
           "null complex, a degree above its dimension, a frame without columns, a row count other than "
           "the cell count, a cell absent at that degree, and a singular pairing (an isotropic frame).")
      .def_static("read_boundary_fiber", &PencilLayer::readBoundaryFiber, py::arg("assembled"), py::arg("k"),
           py::arg("contour"), py::arg("cells"), py::arg("kappa") = 10.0)
      .def_static("level", &PencilLayer::level, py::arg("assembled"), py::arg("k"), py::arg("retained"),
           py::arg("lambda_"))
      .def_static("transfer", &PencilLayer::transfer, py::arg("assembled"), py::arg("k"), py::arg("A"),
           py::arg("B"), py::arg("tolerance") = 1e-8)
      .def_static("pencil", &PencilLayer::pencil, py::arg("assembled"), py::arg("k"));
}
