// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of SurfaceSeed (include/cobordism/SurfaceSeed.h): the hosts
// seeded from boundary surfaces, and the tube of tubed collars. Registered by
// src/cobordism/Bindings.cpp before MultiCobordism, whose seed_inputs takes them.

#include "Bindings.h"

#include "cobordism/SurfaceSeed.h"

void register_cobordism_surface_seed(py::module_ &m) {
  py::class_<TubeSpec>(m, "TubeSpec",
      "The tube of SurfaceSeed.tubed_collars: a prism over one attachment face of each far surface "
      "(`face_a`, `face_b` in the far surfaces' OWN vertex ids), `layers` prism layers long, "
      "consecutive rings `length` apart, the interior rings scaled about their centroid by "
      "`waist`; `reflect` (default) matches the last two vertices of face_b in reversed order, "
      "the orientation-consistent connected sum of the two far surfaces.")
      .def(py::init<>())
      .def_readwrite("layers", &TubeSpec::layers)
      .def_readwrite("length", &TubeSpec::length)
      .def_readwrite("waist", &TubeSpec::waist)
      .def_readwrite("face_a", &TubeSpec::faceA)
      .def_readwrite("face_b", &TubeSpec::faceB)
      .def_readwrite("reflect", &TubeSpec::reflect);
  py::class_<SurfaceSeed>(m, "SurfaceSeed",
      "A host seeded from boundary surfaces (qubit cobordism spec S2): `host` is ONE "
      "d-dimensional Spacetime holding every surface as its own simplices (triangles, "
      "edges, vertices with the surface's lengths and zero phases) on disjoint vertex id "
      "ranges and NO d-cell; `vertex_ids[s]` maps surface s's vertex ids to the host's.")
      .def_readonly("host", &SurfaceSeed::host)
      .def_readonly("vertex_ids", &SurfaceSeed::vertexIds)
      .def_readonly("tube_rings", &SurfaceSeed::tubeRings,
                    "The vertex rings of a tube (tubed_collars), ring 0 on the first collar's far "
                    "surface, the last on the second's; empty on every other seed.")
      .def_static("from_surfaces", &SurfaceSeed::fromSurfaces, py::arg("surfaces"),
           "The SurfaceSeed of closed (d-1)-dimensional Spacetimes (e.g. "
           "SimplicialQubit.flat_torus(tau, n, n).spacetime()): one d-dimensional host holding "
           "every surface as its own simplices with its lengths and zero phases on disjoint "
           "vertex id ranges, and no d-cell -- the bulk is drawn afterwards by gated bridges.")
      .def_static("joined_collars", &SurfaceSeed::joinedCollars,
                  py::arg("surfaces"), py::arg("layers") = 3,
                  py::arg("twist") = std::vector<std::uint64_t>{},
                  "Two collars joined along a removed tetrahedron: FOUR boundary surfaces on one "
                  "connected manifold. Gluing along a sphere is a connected sum, which adds no first "
                  "homology, so b_1 = 2 + 2 = 4 with nothing dying on the boundary -- the dimension a "
                  "4-dimensional target needs, since rank(H^1(W) -> H^1(dW)) = b_1(dW)/2. layers must "
                  "be at least three: a prism cell spans two adjacent layers, so an all-interior cell "
                  "exists only with two interior layers.")
      .def_static("tubed_collars", &SurfaceSeed::tubedCollars,
                  py::arg("surfaces"), py::arg("layers") = 1,
                  py::arg("twist") = std::vector<std::uint64_t>{},
                  py::arg("tube") = TubeSpec{},
                  "Two collars joined by a TUBE (a 1-handle) between their far surfaces: the prism over "
                  "one attachment face of each (TubeSpec), so the far boundary is ONE genus-two surface, "
                  "the connected sum of the two far tori through the tube, and dW = T^2 + T^2 + Sigma_2. "
                  "The tube joins two components and adds no loop: b_1(W) = 4 = b_1(dW)/2 with no class "
                  "invisible to the boundary. Euclidean tube geometry declared from the attachment faces' "
                  "own lengths, the waist and the length (the knobs of the neck); gated once as a whole "
                  "by dualComplexIsValid and refused by name.")
      .def_static("collar", &SurfaceSeed::collar, py::arg("surface_a"), py::arg("surface_b"),
           py::arg("layers") = 1, py::arg("twist") = std::vector<std::uint64_t>{},
           "The SurfaceSeed of the COLLAR between two surfaces of identical combinatorics: "
           "T^2 x I over their shared triangulation (Spacetime.prism_cells, `layers` product layers), "
           "layer 0 = surface A, last layer = surface B, the surfaces' lengths verbatim, the auto-wired "
           "length on every other edge, zero phases, gated once as a whole by dualComplexIsValid and "
           "refused by name; a combinatorial mismatch is refused by name. Seed both id sets as input "
           "blocks with seed_inputs; bridge_phase_complete() then holds by construction.");
}
