// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the read records of
// MultiCobordism. One of the translation units that Bindings.cpp registers
// in order (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_cobordism_multi_cobordism_read(py::module_ &m) {
  py::class_<MultiCobordism::BlockFrame>(m, "BlockFrame",
      "A frame on a block's attached cells (qubit cobordism spec D3): the coordinates the two-body "
      "transfer is read in. `images` Z is one column per basis element on the block's fiber cells "
      "(rows in the attachment order); `dual_images` Z^vee is the frame's left partner under the "
      "transpose pairing, the contract Z^vee.T @ M_k @ Z == I with M_k the Whitney mass matrix of "
      "the block's OWN pencil on the cells (MultiCobordism.dual_frame / input_frame_dual). With both "
      "input blocks framed the transfer is T_AB = Z_A^vee.T @ A~_AB @ Z_B, the whole's pencil-operator "
      "block in the frames' coordinates (a change of frame g_A, g_B gives g_A^-1 T g_B); for two "
      "qubit tori the frames are their period frames (SimplicialQubit.period_frame) and T is 2x2 in "
      "the |0>, |1> bases a two-body target chi is written in. Separate from the fiber: the fiber is "
      "the rank-one state the block keeps representing (scored), the frame is the basis that state "
      "is a coordinate vector in (never scored).")
      .def_readonly("cells", &MultiCobordism::BlockFrame::cells)
      .def_readonly("images", &MultiCobordism::BlockFrame::images)
      .def_readonly("dual_images", &MultiCobordism::BlockFrame::dualImages)
      .def("rank", &MultiCobordism::BlockFrame::rank);
  py::class_<MultiCobordism::BlockMarking>(m, "BlockMarking",
      "A block's MARKING with its input coefficients (qubit cobordism spec D2/D3 as revised): "
      "`cycles`, each one closed walk of directed host steps (u, v) starting at `base_vertex` (the "
      "first vertex of the first cycle's walk that lies on every other cycle's walk), and "
      "`coefficients` (a, b), one per cycle -- (1, tau_in) for an input torus. The engine derives the "
      "block's frame from it live (MultiCobordism.derive_input_frame) and scores the leak of the "
      "coefficients, written through that frame, in the WHOLE's zero mode on the block's edges "
      "(input_state_residual). Nothing about the marking is scored.")
      .def_readonly("cycles", &MultiCobordism::BlockMarking::cycles)
      .def_readonly("coefficients", &MultiCobordism::BlockMarking::coefficients)
      .def_readonly("base_vertex", &MultiCobordism::BlockMarking::baseVertex)
      .def("rank", &MultiCobordism::BlockMarking::rank);
  py::class_<MultiCobordism::DerivedFrame>(m, "DerivedFrame",
      "The frame the engine DERIVES for a marked block from its live own complex: `frame` (BlockFrame on the block's own edges) with images F = Z inv(Pi), Z "
      "the images of the zero mode of the block's own covariant Whitney pencil and Pi its transported "
      "periods over the marking's cycles from the common base point, so column b has period delta_cb; "
      "dual images from the DUAL kernel (the zero mode under the inverse links) normalized so that "
      "F^vee.T @ M_1^U @ F == I; `periods` = Pi; `kernel_rank` = the own zero mode's rank. An obstructed "
      "derivation (kernel rank other than the marking's, a marking edge absent, singular periods, an "
      "isotropic pairing) names `obstruction` and `derived()` is False; the block then reads the full leak.")
      .def_readonly("frame", &MultiCobordism::DerivedFrame::frame)
      .def_readonly("periods", &MultiCobordism::DerivedFrame::periods)
      .def_readonly("kernel_rank", &MultiCobordism::DerivedFrame::kernelRank)
      .def_readonly("obstruction", &MultiCobordism::DerivedFrame::obstruction)
      .def("derived", &MultiCobordism::DerivedFrame::derived);
  py::class_<MultiCobordism::InputStateRead>(m, "InputStateRead",
      "The state at a marked block: `coefficients` = the "
      "coefficients of the WHOLE's zero mode in the block's live frame -- the transported periods over "
      "the marking's cycles of the least-squares combination of the whole's zero mode fitting the "
      "block's target edge values (its `input` coefficients through the live frame) on the block's "
      "edges -- next to `input`; `residual` = that fit's leak, the block residual of D2 scored at "
      "`weight` in r_U; `harmonic_rank` of the whole, `frame_rank` of the own kernel, `base_vertex`; an "
      "obstructed read names `obstruction` and reads the full leak 1.0.")
      .def_readonly("block", &MultiCobordism::InputStateRead::block)
      .def_readonly("input", &MultiCobordism::InputStateRead::input)
      .def_readonly("coefficients", &MultiCobordism::InputStateRead::coefficients)
      .def_readonly("residual", &MultiCobordism::InputStateRead::residual)
      .def_readonly("weight", &MultiCobordism::InputStateRead::weight)
      .def_readonly("harmonic_rank", &MultiCobordism::InputStateRead::harmonicRank)
      .def_readonly("frame_rank", &MultiCobordism::InputStateRead::frameRank)
      .def_readonly("base_vertex", &MultiCobordism::InputStateRead::baseVertex)
      .def_readonly("obstruction", &MultiCobordism::InputStateRead::obstruction);

  py::class_<MultiCobordism::BoundaryBlock>(m, "MultiCobordismBlock",
      "An emergent boundary block of a MultiCobordism (an input or output): the "
      "vertex set whose own sub-complex carries the block, and its target period "
      "vector. Read the block's sub-complex with Spacetime.fromVertexTuples over the "
      "cells inside `vertices`, then its holes with MultiCobordism.emergent_holes.")
      .def_property_readonly(
          "vertices",
          [](const MultiCobordism::BoundaryBlock &block) {
            return std::vector<std::uint64_t>(block.vertices.begin(),
                                              block.vertices.end());
          })
      .def_property_readonly("target",
                             [](const MultiCobordism::BoundaryBlock &block) {
                               return block.target;
                             })
      .def_property_readonly("fiber",
                             [](const MultiCobordism::BoundaryBlock &block) {
                               return block.fiber;
                             },
                             "The fiber form of the target, or None.")
      .def_property_readonly("surface",
                             [](const MultiCobordism::BoundaryBlock &block) {
                               return block.surface;
                             },
                             "True for an input SURFACE block (the region form of "
                             "seed_inputs on a seed_from_surfaces host): its faces are "
                             "cells of the host the bulk is drawn onto by bridges.")
      .def_property_readonly("faces",
                             [](const MultiCobordism::BoundaryBlock &block) {
                               return block.faces;
                             },
                             "A surface block's own faces as seeded (sorted vertex tuples), "
                             "carried by the block so its own complex is the surface whatever "
                             "the host registers; empty for an ordinary block.")
      .def_property_readonly("frame",
                             [](const MultiCobordism::BoundaryBlock &block) {
                               return block.frame;
                             },
                             "The block's frame for the two-body transfer (BlockFrame, spec D3), "
                             "set by set_input_frame, or None; never read on a block that carries "
                             "a marking (its frame is derived live).")
      .def_property_readonly("marking",
                             [](const MultiCobordism::BoundaryBlock &block) {
                               return block.marking;
                             },
                             "The block's marking with its input coefficients (BlockMarking, spec "
                             "D2/D3 as revised), set by set_input_marking, or None.");
  py::class_<MultiCobordism::FixedBoundaryEigenstateResult>(
      m, "FixedBoundaryEigenstateResult",
      "Witness from the historical fixed-boundary Rayleigh-residual "
      "relaxation. This is direct eigenstate synthesis, not a period r_U "
      "constraint or target-free operator readout.")
      .def_readonly(
          "converged",
          &MultiCobordism::FixedBoundaryEigenstateResult::converged)
      .def_readonly(
          "residual",
          &MultiCobordism::FixedBoundaryEigenstateResult::residual)
      .def_readonly(
          "eigenvalue",
          &MultiCobordism::FixedBoundaryEigenstateResult::eigenvalue)
      .def_readonly("degree",
                    &MultiCobordism::FixedBoundaryEigenstateResult::degree)
      .def_readonly(
          "growth_steps",
          &MultiCobordism::FixedBoundaryEigenstateResult::growthSteps)
      .def_readonly(
          "interior_vertex_count",
          &MultiCobordism::FixedBoundaryEigenstateResult::interiorVertexCount)
      .def_readonly(
          "interior_edge_count",
          &MultiCobordism::FixedBoundaryEigenstateResult::interiorEdgeCount)
      .def_readonly(
          "auxiliary_cell_count",
          &MultiCobordism::FixedBoundaryEigenstateResult::auxiliaryCellCount)
      .def_readonly(
          "support_cells",
          &MultiCobordism::FixedBoundaryEigenstateResult::supportCells)
      .def_readonly("target",
                    &MultiCobordism::FixedBoundaryEigenstateResult::target)
      .def_readonly("state",
                    &MultiCobordism::FixedBoundaryEigenstateResult::state);
  py::class_<MultiCobordism::BoundaryStateTransferResult>(
      m, "BoundaryStateTransferResult",
      "Coupled full-cobordism eigenstate witnesses with exact input/output "
      "restrictions on two independently prepared, geometrically pinned "
      "boundary components.")
      .def_readonly(
          "converged",
          &MultiCobordism::BoundaryStateTransferResult::converged)
      .def_readonly(
          "common_eigenvalue",
          &MultiCobordism::BoundaryStateTransferResult::commonEigenvalue)
      .def_readonly(
          "residual",
          &MultiCobordism::BoundaryStateTransferResult::residual)
      .def_readonly(
          "eigenvalue",
          &MultiCobordism::BoundaryStateTransferResult::eigenvalue)
      .def_readonly("degree",
                    &MultiCobordism::BoundaryStateTransferResult::degree)
      .def_readonly(
          "growth_steps",
          &MultiCobordism::BoundaryStateTransferResult::growthSteps)
      .def_readonly(
          "free_edge_count",
          &MultiCobordism::BoundaryStateTransferResult::freeEdgeCount)
      .def_readonly(
          "auxiliary_cell_count",
          &MultiCobordism::BoundaryStateTransferResult::auxiliaryCellCount)
      .def_readonly(
          "input_region",
          &MultiCobordism::BoundaryStateTransferResult::inputRegion)
      .def_readonly(
          "output_region",
          &MultiCobordism::BoundaryStateTransferResult::outputRegion)
      .def_readonly(
          "input_cells",
          &MultiCobordism::BoundaryStateTransferResult::inputCells)
      .def_readonly(
          "output_cells",
          &MultiCobordism::BoundaryStateTransferResult::outputCells)
      .def_readonly(
          "input_states",
          &MultiCobordism::BoundaryStateTransferResult::inputStates)
      .def_readonly(
          "output_states",
          &MultiCobordism::BoundaryStateTransferResult::outputStates)
      .def_readonly(
          "states",
          &MultiCobordism::BoundaryStateTransferResult::states)
      .def_readonly(
          "state_residuals",
          &MultiCobordism::BoundaryStateTransferResult::stateResiduals)
      .def_readonly(
          "state_eigenvalues",
          &MultiCobordism::BoundaryStateTransferResult::stateEigenvalues)
      .def_readonly(
          "input_boundary_residuals",
          &MultiCobordism::BoundaryStateTransferResult::
              inputBoundaryResiduals)
      .def_readonly(
          "output_boundary_residuals",
          &MultiCobordism::BoundaryStateTransferResult::
              outputBoundaryResiduals)
      .def_readonly(
          "residual_trace",
          &MultiCobordism::BoundaryStateTransferResult::residualTrace);
  py::class_<MultiCobordism::TwoBodyTarget>(m, "TwoBodyTarget",
      "chi on the pair of attached frames and the reading flag.")
      .def(py::init<>())
      .def_readwrite("chi", &MultiCobordism::TwoBodyTarget::chi)
      .def_readwrite("choi_decomposed", &MultiCobordism::TwoBodyTarget::choiDecomposed)
      .def_readwrite("paired_direct_sum_target",
                     &MultiCobordism::TwoBodyTarget::twoStateVector,
                     "Target expressed in the paired direct-sum frame.")
      .def_readwrite("two_state_vector",
                     &MultiCobordism::TwoBodyTarget::twoStateVector,
                     "Deprecated legacy name for paired_direct_sum_target; "
                     "this field is not in tensor-product coordinates.")
      .def_readwrite("input_coefficients", &MultiCobordism::TwoBodyTarget::inputCoefficients,
                     "Optional case-specific coefficients in marked-block/cycle order.");
  py::class_<MultiCobordism::SurfaceSeed>(m, "SurfaceSeed",
      "A host seeded from boundary surfaces (qubit cobordism spec S2): `host` is ONE "
      "d-dimensional Spacetime holding every surface as its own simplices (triangles, "
      "edges, vertices with the surface's lengths and zero phases) on disjoint vertex id "
      "ranges and NO d-cell; `vertex_ids[s]` maps surface s's vertex ids to the host's.")
      .def_readonly("host", &MultiCobordism::SurfaceSeed::host)
      .def_readonly("vertex_ids", &MultiCobordism::SurfaceSeed::vertexIds)
      .def_readonly("tube_rings", &MultiCobordism::SurfaceSeed::tubeRings,
                    "The vertex rings of a tube (seed_tubed_collars), ring 0 on the first collar's far "
                    "surface, the last on the second's; empty on every other seed.");
  py::class_<MultiCobordism::TubeSpec>(m, "TubeSpec",
      "The tube of seed_tubed_collars: a prism over one attachment face of each far surface "
      "(`face_a`, `face_b` in the far surfaces' OWN vertex ids), `layers` prism layers long, "
      "consecutive rings `length` apart, the interior rings scaled about their centroid by "
      "`waist`; `reflect` (default) matches the last two vertices of face_b in reversed order, "
      "the orientation-consistent connected sum of the two far surfaces.")
      .def(py::init<>())
      .def_readwrite("layers", &MultiCobordism::TubeSpec::layers)
      .def_readwrite("length", &MultiCobordism::TubeSpec::length)
      .def_readwrite("waist", &MultiCobordism::TubeSpec::waist)
      .def_readwrite("face_a", &MultiCobordism::TubeSpec::faceA)
      .def_readwrite("face_b", &MultiCobordism::TubeSpec::faceB)
      .def_readwrite("reflect", &MultiCobordism::TubeSpec::reflect);
  py::class_<MultiCobordism::BlockSurface>(m, "BlockSurface",
      "A block's own surface: its (d-1)-faces and its "
      "edges inside its vertex set, as sorted vertex-id tuples.")
      .def_readonly("faces", &MultiCobordism::BlockSurface::faces)
      .def_readonly("edges", &MultiCobordism::BlockSurface::edges);
  py::class_<MultiCobordism::MonodromyRead>(m, "MonodromyRead",
      "The monodromy of a drawn cobordism between two marked surfaces: the "
      "whole's degree-1 ZERO MODE (the harmonic contour of the Whitney pencil, bulk and "
      "boundary edges in one operator) read on both markings' edges, its periods "
      "periods_a (|A| x rank) and periods_b, and the matrix M with P_B = M P_A "
      "(M = P_B P_A^-1 for a rank-2 zero mode, least squares otherwise), with its integer "
      "rounding and rounding residual; Betti numbers of the whole come with it. A read that "
      "cannot be made names its `obstruction` instead of guessing.")
      .def_readonly("betti", &MultiCobordism::MonodromyRead::betti)
      .def_readonly("harmonic_rank", &MultiCobordism::MonodromyRead::harmonicRank)
      .def_readonly("periods_a", &MultiCobordism::MonodromyRead::periodsA)
      .def_readonly("periods_b", &MultiCobordism::MonodromyRead::periodsB)
      .def_readonly("monodromy", &MultiCobordism::MonodromyRead::monodromy)
      .def_readonly("rounded", &MultiCobordism::MonodromyRead::rounded)
      .def_readonly("rounding_residual", &MultiCobordism::MonodromyRead::roundingResidual)
      .def_readonly("fit_residual", &MultiCobordism::MonodromyRead::fitResidual)
      .def_readonly("obstruction", &MultiCobordism::MonodromyRead::obstruction);
  py::class_<MultiCobordism::RestrictionRead>(m, "RestrictionRead",
      "The whole's degree-1 zero mode read on every marking at once: ONE harmonic basis "
      "`images` (n_1 x rank, in the assembled pencil's degree-1 cell order) and its periods "
      "over each marking (`periods[i]`, |marking_i| x rank) from that same basis. `monodromy` "
      "is the two-marking case with the fit added. Betti numbers of the whole come with it; "
      "a read that cannot be made names its `obstruction` instead of guessing.")
      .def_readonly("betti", &MultiCobordism::RestrictionRead::betti)
      .def_readonly("harmonic_rank", &MultiCobordism::RestrictionRead::harmonicRank)
      .def_readonly("images", &MultiCobordism::RestrictionRead::images)
      .def_readonly("periods", &MultiCobordism::RestrictionRead::periods)
      .def_readonly("frame", &MultiCobordism::RestrictionRead::frame,
                    "The band's frame on chains, n_1 x rank; `images` = G_1^U frame.")
      .def_readonly("certificate", &MultiCobordism::RestrictionRead::certificate,
                    "The band's BandCertificate: node count, idempotency, rank, gap, resolvent bound.")
      .def_readonly("obstruction", &MultiCobordism::RestrictionRead::obstruction);
  py::class_<MultiCobordism::TwoBodyRead>(m, "TwoBodyRead",
      "The reading of the bulk between two attached input frames: the frame transfer "
      "T_AB (operator reading), vec(T_AB) (Choi-decomposed state reading), its Schmidt spectrum "
      "and rank, the reversal residual, the fit residual for two inputs (NaN for the four-input "
      "paired diagnostic), and the input blocks' fiber residuals.")
      .def_readonly("choi_decomposed", &MultiCobordism::TwoBodyRead::choiDecomposed)
      .def_readonly("transfer", &MultiCobordism::TwoBodyRead::transfer)
      .def_readonly("in_frames", &MultiCobordism::TwoBodyRead::inFrames,
                    "True when `transfer` was read in the blocks' frames (a marking by set_input_marking "
                    "or a frame by set_input_frame, on both), false when in identity frames on the cells.")
      .def_readonly("derived_frames", &MultiCobordism::TwoBodyRead::derivedFrames,
                    "True when both frames were derived live from the blocks' markings.")
      .def_readonly("input_states", &MultiCobordism::TwoBodyRead::inputStates,
                    "The state at every marked input block (InputStateRead, block order).")
      .def_readonly("choi_state", &MultiCobordism::TwoBodyRead::choiState)
      .def_readonly("singular_values", &MultiCobordism::TwoBodyRead::singularValues)
      .def_readonly("schmidt_rank", &MultiCobordism::TwoBodyRead::schmidtRank)
      .def_readonly("reversal_residual", &MultiCobordism::TwoBodyRead::reversalResidual)
      .def_readonly("residual", &MultiCobordism::TwoBodyRead::residual)
      .def_readonly("input_fiber_residuals", &MultiCobordism::TwoBodyRead::inputFiberResiduals,
                    "The leak of each attached input fiber in its block's OWN kernel (T2): a geometric "
                    "diagnostic on a marked block, whose scored residual is in input_states.")
      .def_readonly("cells_a", &MultiCobordism::TwoBodyRead::cellsA)
      .def_readonly("cells_b", &MultiCobordism::TwoBodyRead::cellsB);
  py::class_<MultiCobordism::WholeComplexReadoutResult>(
      m, "WholeComplexReadoutResult",
      "Coupled full-cobordism eigenstate witnesses whose boundary "
      "restrictions on BOTH components are fixed inputs and whose "
      "whole-complex readouts are fixed exactly to the prescribed outputs.")
      .def_readonly("converged",
                    &MultiCobordism::WholeComplexReadoutResult::converged)
      .def_readonly("common_eigenvalue",
                    &MultiCobordism::WholeComplexReadoutResult::commonEigenvalue)
      .def_readonly("residual",
                    &MultiCobordism::WholeComplexReadoutResult::residual)
      .def_readonly("eigenvalue",
                    &MultiCobordism::WholeComplexReadoutResult::eigenvalue)
      .def_readonly("degree", &MultiCobordism::WholeComplexReadoutResult::degree)
      .def_readonly("growth_steps",
                    &MultiCobordism::WholeComplexReadoutResult::growthSteps)
      .def_readonly("free_edge_count",
                    &MultiCobordism::WholeComplexReadoutResult::freeEdgeCount)
      .def_readonly("auxiliary_cell_count",
                    &MultiCobordism::WholeComplexReadoutResult::auxiliaryCellCount)
      .def_readonly("readout_rank",
                    &MultiCobordism::WholeComplexReadoutResult::readoutRank)
      .def_readonly("region_a", &MultiCobordism::WholeComplexReadoutResult::regionA)
      .def_readonly("region_b", &MultiCobordism::WholeComplexReadoutResult::regionB)
      .def_readonly("cells_a", &MultiCobordism::WholeComplexReadoutResult::cellsA)
      .def_readonly("cells_b", &MultiCobordism::WholeComplexReadoutResult::cellsB)
      .def_readonly("states_a", &MultiCobordism::WholeComplexReadoutResult::statesA)
      .def_readonly("states_b", &MultiCobordism::WholeComplexReadoutResult::statesB)
      .def_readonly("targets", &MultiCobordism::WholeComplexReadoutResult::targets)
      .def_readonly("readouts", &MultiCobordism::WholeComplexReadoutResult::readouts)
      .def_readonly("readout_deviation",
                    &MultiCobordism::WholeComplexReadoutResult::readoutDeviation)
      .def_readonly("states", &MultiCobordism::WholeComplexReadoutResult::states)
      .def_readonly("state_residuals",
                    &MultiCobordism::WholeComplexReadoutResult::stateResiduals)
      .def_readonly("state_eigenvalues",
                    &MultiCobordism::WholeComplexReadoutResult::stateEigenvalues)
      .def_readonly("boundary_residuals_a",
                    &MultiCobordism::WholeComplexReadoutResult::boundaryResidualsA)
      .def_readonly("boundary_residuals_b",
                    &MultiCobordism::WholeComplexReadoutResult::boundaryResidualsB)
      .def_readonly("residual_trace",
                    &MultiCobordism::WholeComplexReadoutResult::residualTrace);
}
