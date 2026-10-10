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
      .def_readwrite("dualImages", &BoundaryFiber::dualImages)
      .def_readwrite("gram", &BoundaryFiber::gram)
      .def_readwrite("fullGram", &BoundaryFiber::fullGram)
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
      .def_property_readonly("sharedCells", [](const AssembledPencil &a) { return a.sharedCells; })
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
      .def_readonly("upperCount", &BorderedPencil::upperCount)
      .def_readonly("lowerCount", &BorderedPencil::lowerCount)
      .def_readonly("matrix", &BorderedPencil::matrix);

  py::class_<FiberLevel>(m, "FiberLevel",
      "A pencil level whose interface coordinates are retained fibers: the Feshbach "
      "reduction onto the fibers' cells restricted to the fibers, with J, J~ = Z, and the Gram.")
      .def_readonly("degree", &FiberLevel::degree)
      .def_readonly("lambda_", &FiberLevel::lambda)
      .def_readonly("interfaceCells", &FiberLevel::interfaceCells)
      .def_readonly("interiorCells", &FiberLevel::interiorCells)
      .def_readonly("response", &FiberLevel::response)
      .def_readonly("J", &FiberLevel::J)
      .def_readonly("Jdual", &FiberLevel::Jdual)
      .def_readonly("restriction", &FiberLevel::restriction)
      .def_readonly("constraintGram", &FiberLevel::constraintGram)
      .def_readonly("blockOffsets", &FiberLevel::blockOffsets)
      .def_readonly("blockRanks", &FiberLevel::blockRanks)
      .def_readonly("fibersDisjoint", &FiberLevel::fibersDisjoint);

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
      .def_static("read_boundary_fiber", &PencilLayer::readBoundaryFiber, py::arg("assembled"), py::arg("k"),
           py::arg("contour"), py::arg("cells"), py::arg("kappa") = 10.0)
      .def_static("level", &PencilLayer::level, py::arg("assembled"), py::arg("k"), py::arg("retained"),
           py::arg("lambda_"))
      .def_static("transfer", &PencilLayer::transfer, py::arg("assembled"), py::arg("k"), py::arg("A"),
           py::arg("B"), py::arg("tolerance") = 1e-8)
      .def_static("pencil", &PencilLayer::pencil, py::arg("assembled"), py::arg("k"));
}
