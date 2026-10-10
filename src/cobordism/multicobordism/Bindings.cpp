// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of MultiCobordism (include/cobordism/MultiCobordism.h): the
// class, its nested types and enums, and the read records it returns.
// Registered by src/cobordism/Bindings.cpp after the objective family, whose
// types its methods take (https://github.com/akellehe/tessera/issues/1453).

#include "../Bindings.h"

void register_cobordism_multicobordism(py::module_ &m) {
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

  py::class_<MultiCobordism::GeometricOperatorReadout>(
      m, "GeometricOperatorReadout",
      "Target-free promotion of a framed bulk-minus-boundary kernel. "
      "identifiable is false unless the framed kernel has rank one.")
      .def_readonly("identifiable",
                    &MultiCobordism::GeometricOperatorReadout::identifiable)
      .def_readonly("obstruction",
                    &MultiCobordism::GeometricOperatorReadout::obstruction)
      .def_readonly("state_dimension",
                    &MultiCobordism::GeometricOperatorReadout::stateDimension)
      .def_readonly("metric", &MultiCobordism::GeometricOperatorReadout::metric)
      .def_readonly("bulk_cell_count",
                    &MultiCobordism::GeometricOperatorReadout::bulkCellCount)
      .def_readonly("kernel_dimension",
                    &MultiCobordism::GeometricOperatorReadout::kernelDimension)
      .def_readonly("frame_rank",
                    &MultiCobordism::GeometricOperatorReadout::frameRank)
      .def_readonly("unitarity_error",
                    &MultiCobordism::GeometricOperatorReadout::unitarityError)
      .def_readonly(
          "frame_singular_values",
          &MultiCobordism::GeometricOperatorReadout::frameSingularValues)
      .def_readonly("bulk_cells",
                    &MultiCobordism::GeometricOperatorReadout::bulkCells)
      .def_readonly("frame_cells",
                    &MultiCobordism::GeometricOperatorReadout::frameCells)
      .def_readonly("choi_state",
                    &MultiCobordism::GeometricOperatorReadout::choiState)
      .def_readonly("operator_matrix",
                    &MultiCobordism::GeometricOperatorReadout::operatorMatrix);
  auto multiCobordismClass =
      py::class_<MultiCobordism, std::shared_ptr<MultiCobordism>>(m, "MultiCobordism",
      "The fully-emergent MultiCobordism merge optimizer: merge as a "
      "fully emergent optimization. From a bare host it grows the register by "
      "gated surgical moves under an explicitly selected Legacy, joint "
      "Regge-Hodge-stationarity, or mediated-correspondence objective at a "
      "USER-DEFINED degree k (degrees), reading holes "
      "dynamically off getBoundary. Two stages: run_stage1 (combinatorial), "
      "run_stage2 (geometric) -- or run(), which interleaves both updates in one "
      "loop. An EMPTY output_targets list is supported: "
      "nothing is pinned downstream, r_u sums only the input blocks, and the "
      "whole's final state emerges (read after the fact).")
      .def(py::init<std::shared_ptr<Spacetime>,
                    std::vector<std::vector<std::complex<double>>>,
                    std::vector<std::vector<std::complex<double>>>,
                    std::vector<int>, double, std::uint64_t, int, bool, bool,
                    bool, bool, bool, bool, bool>(),
           py::arg("host"), py::arg("input_targets"), py::arg("output_targets"),
           py::arg("degrees") = std::vector<int>{3}, py::arg("gamma") = 1.0,
           py::arg("seed") = 0, py::arg("precone") = 0,
           py::arg("should_propose_dispositions") = true,
           py::arg("precone_timelike") = false,
           py::arg("precone_alternate") = false,
           py::arg("balanced_edge_wiring") = false,
           py::arg("singular_value_ratio") = false,
           py::arg("einstein_hilbert") = true,
           py::arg("real_squared_lengths_only") = false)
      // Explicit metric source: a pybind11 default would capture the
      // process-wide HodgeLaplacian.defaultMetricSource() at import rather than
      // reading it at call time.
      .def(py::init<std::shared_ptr<Spacetime>,
                    std::vector<std::vector<std::complex<double>>>,
                    std::vector<std::vector<std::complex<double>>>,
                    std::vector<int>, double, std::uint64_t, int, bool, bool,
                    bool, bool, bool, bool, bool, HodgeLaplacian::MetricSource>(),
           py::arg("host"), py::arg("input_targets"), py::arg("output_targets"),
           py::arg("degrees") = std::vector<int>{3}, py::arg("gamma") = 1.0,
           py::arg("seed") = 0, py::arg("precone") = 0,
           py::arg("should_propose_dispositions") = true,
           py::arg("precone_timelike") = false,
           py::arg("precone_alternate") = false,
           py::arg("balanced_edge_wiring") = false,
           py::arg("singular_value_ratio") = false,
           py::arg("einstein_hilbert") = true,
           py::arg("real_squared_lengths_only") = false,
           py::arg("metric_source"))
      .def("metricSource", &MultiCobordism::metricSource,
           "Where every Hodge operator this node scores takes its metric from.")
      .def("geometryAdmissible", &MultiCobordism::geometryAdmissible, py::arg("spacetime"),
           "Whitney pencil: whether the geometry lies in the closure of the Kontsevich-Segal "
           "allowable domain (margin >= 0); always true under DiagonalWeights.")
      .def_static("betti", &MultiCobordism::betti, py::arg("st"))
      .def_static("emergent_holes", &MultiCobordism::emergentHoles,
                  py::arg("st"), py::arg("k"))
      .def_static("regge_action_gradient", &MultiCobordism::reggeActionGradient, py::arg("st"))
      .def_static("nearKernelResidualGradient",
           [](const std::shared_ptr<Spacetime> &st, int k, std::size_t n,
              std::optional<HodgeLaplacian::MetricSource> source) {
             return MultiCobordism::nearKernelResidualGradient(
                 st, k, n, source.value_or(HodgeLaplacian::defaultMetricSource()));
           },
           py::arg("st"), py::arg("register_degree"),
           py::arg("expected_register_count"),
           py::arg("metric_source") = py::none(),
           "Exact COMPLEX gradient of nearKernelResidual per edge, in "
           "ChainComplex 1-cell order: g = dr/d(Re l^2) - i dr/d(Im l^2), so "
           "Re(g) and -Im(g) are the two directional derivatives and conj(g) is "
           "the steepest-ascent direction. Certified by the scale-invariance "
           "Euler identity sum l^2 g = 0 in both parts.")
      .def_static("nearKernelResidual",
           [](const std::shared_ptr<Spacetime> &st, int k, std::size_t n,
              std::optional<HodgeLaplacian::MetricSource> source) {
             return MultiCobordism::nearKernelResidual(
                 st, k, n, source.value_or(HodgeLaplacian::defaultMetricSource()));
           },
           py::arg("st"), py::arg("register_degree"),
           py::arg("expected_register_count"),
           py::arg("metric_source") = py::none(),
           "The pre-topological register signal: the normalized sum of the "
           "expected_register_count smallest squared SINGULAR values of the "
           "METRIC L_k (n * sum_m sigma^2 / sum_all sigma^2; range [0, m]; "
           "scale-invariant, so no conformal-inflation channel). Metric by "
           "DESIGN: it descends both by stage-1 surgery (a hole zeroes the "
           "sigma exactly) and by stage-2 tuning the causal structure toward "
           "null directions — near-kernels with no holes are the intended "
           "exploration, not a loophole. "
           "Saturates at exactly 0 once b_k reaches the expected count; before "
           "any register exists it is the objective's only register-seeking "
           "gradient (the period residual is a step function in the topology). "
           "The count comes from the TARGETS (one register per target "
           "component), never a constant.")
      .def_static("singularValueHalfSumRatio",
           [](const std::shared_ptr<Spacetime> &st, int k,
              std::optional<HodgeLaplacian::MetricSource> source) {
             return MultiCobordism::singularValueHalfSumRatio(
                 st, k, source.value_or(HodgeLaplacian::defaultMetricSource()));
           },
           py::arg("st"), py::arg("register_degree"),
           py::arg("metric_source") = py::none(),
           "The scale-invariant spectral-shape term the singular_value_ratio "
           "mode scores as rU's whole-complex contribution, replacing both the "
           "single-output period residual and nearKernelResidual: the sum of "
           "the lower half of the singular values of the METRIC L_k over the "
           "sum of the upper half (odd counts leave the median out of both). "
           "Range [0, 1]; degree 0 in l^2, so no conformal-inflation channel; "
           "no target enters — what the register carries is read afterwards. "
           "An empty degree (no k-cells) scores the worst case 1; a single "
           "mode or an identically-zero L_k scores 0.")
      .def("expectedRegisterCount", &MultiCobordism::expectedRegisterCount,
           "The number of registers the targets ask for: the largest component "
           "count over every input and output target vector.")
      .def_static("r_state",
                  [](const std::shared_ptr<Spacetime> &st, int k,
                     const std::vector<std::complex<double>> &target,
                     std::optional<HodgeLaplacian::MetricSource> source) {
                    return MultiCobordism::residualOfTargetStateAgainstHarmonic(
                        st, k, target, source.value_or(HodgeLaplacian::defaultMetricSource()));
                  },
                  py::arg("st"), py::arg("k"), py::arg("target"),
                  py::arg("metric_source") = py::none())
      .def("r_u", &MultiCobordism::rU, py::arg("st"))
      .def("declare_register_constraint",
           [](MultiCobordism &self, const std::string &name, int degree,
              const std::vector<std::vector<std::uint64_t>> &holes,
              const std::vector<std::complex<double>> &target) {
             self.declareRegisterConstraint({name, degree, holes, target});
           },
           py::arg("name"), py::arg("degree"), py::arg("holes"),
           py::arg("target"),
           "Declare an ordered exact-period r_U constraint. target[i] is "
           "matched to holes[i] with no component permutation; reusing a name "
           "replaces that constraint.")
      .def("register_constraints",
           [](const MultiCobordism &self) {
             py::list records;
             for (const auto &constraint : self.registerConstraints()) {
               py::dict record;
               record["name"] = constraint.name;
               record["degree"] = constraint.degree;
               record["holes"] = constraint.holes;
               record["target"] = constraint.target;
               records.append(std::move(record));
             }
             return records;
           },
           "The ordered explicit register constraints as dictionaries.")
      .def("clear_register_constraints",
           &MultiCobordism::clearRegisterConstraints,
           "Remove explicit register constraints without changing emergent "
           "targets or geometric pins.")
      .def("relax_fixed_boundary_eigenstate",
           &MultiCobordism::relaxFixedBoundaryEigenstate,
           py::arg("degree"), py::arg("support_cells"), py::arg("target"),
           py::arg("epsilon") = 1e-10, py::arg("restarts") = 64,
           py::arg("max_growth") = 4, py::arg("seed") = 0,
           py::arg("max_iterations") = 200,
           py::call_guard<py::gil_scoped_release>(),
           "Historical fixed-boundary inverse-eigenvector relaxation. "
           "Pins the relative amplitudes of the named cochain block, frees "
           "every other amplitude, varies only interior geometry, and "
           "minimizes the Rayleigh "
           "residual. It does not run the node's period or Regge objective.")
      .def("set_boundary_may_extend", &MultiCobordism::setBoundaryMayExtend, py::arg("allowed"),
           "Whether a CONE move may extend a FIXED boundary (default False: it may not). False "
           "refuses a cone_out, cone_in or cone_in_timelike candidate whose complex has a boundary "
           "facet the complex before the move lacked, beside the manifold gate and on the same "
           "footing. It applies only where a boundary is DECLARED fixed (has_fixed_boundary: a "
           "surface input or output block); a node with neither is unrestricted either "
           "way. The Pachner moves and bridge keep their own gates.")
      .def_property_readonly("boundary_may_extend", &MultiCobordism::boundaryMayExtend)
      .def_property_readonly("has_fixed_boundary", &MultiCobordism::hasFixedBoundary,
                             "Whether this node declares a boundary it holds fixed: any surface "
                             "input or output block.")
      .def("set_two_body_cases",
           [](MultiCobordism &self, const py::list &cases) {
             std::vector<MultiCobordism::TwoBodyCase> out;
             out.reserve(cases.size());
             for (const auto &item : cases) {
               auto entry = item.cast<py::tuple>();
               if (entry.size() < 2 || entry.size() > 5)
                 throw std::invalid_argument(
                     "each case is (boundary, chi[, choi_decomposed[, paired_direct_sum_target"
                     "[, input_coefficients]]])");
               MultiCobordism::TwoBodyCase one;
               for (const auto &edge : entry[0].cast<py::list>()) {
                 auto quad = edge.cast<py::tuple>();
                 if (quad.size() != 3)
                   throw std::invalid_argument(
                       "each boundary entry is (source_id, target_id, squared_length)");
                 one.boundary.emplace_back(
                     std::make_pair(quad[0].cast<std::uint64_t>(), quad[1].cast<std::uint64_t>()),
                     quad[2].cast<std::complex<double>>());
               }
               one.chi = entry[1].cast<Eigen::MatrixXcd>();
               one.choiDecomposed = entry.size() >= 3 ? entry[2].cast<bool>() : true;
               if (entry.size() >= 4 && !entry[3].is_none())
                 one.twoStateVector = entry[3].cast<Eigen::MatrixXcd>();
               if (entry.size() == 5 && !entry[4].is_none())
                 one.inputCoefficients = entry[4].cast<Eigen::VectorXcd>();
               out.push_back(std::move(one));
             }
             self.setTwoBodyCases(std::move(out));
           }, py::arg("cases"),
           "Fit ONE bulk to several input pairs at once. Each case is "
           "(boundary, chi[, choi_decomposed[, paired_direct_sum_target[, "
           "input_coefficients]]]), where boundary is a list of "
           "(source_id, target_id, squared_length) for the BOUNDARY edges and "
           "chi is the gate's image of that pair. The legacy fourth tuple "
           "slot is a target explicitly expressed in the paired direct-sum "
           "frame; it is not a tensor-product two-state vector. Input "
           "coefficients are "
           "flattened in marked-block/cycle order; empty uses the live markings. "
           "Under the transfer reading, a bulk fitted to one pair reproduces "
           "that pair and nothing else: one 2x2 transfer is six real constraints "
           "against some eighty free bulk coordinates. Other selected readings "
           "add their own residual for the same case. The per-case sum lives in "
           "the objective, so stage 1 prices every candidate move and stage 2 "
           "accepts every step against ALL states, with neither stage needing "
           "to know there is more than one. Boundary metrics, targets and "
           "input coefficients may differ between cases; all share one "
           "triangulation, one gluing and one bulk. Expects a "
           "pinned boundary. Empty restores the single-target behaviour.")
      .def("two_body_case_count",
           [](const MultiCobordism &self) { return self.twoBodyCases().size(); })
      .def("two_body_residuals_per_case", &MultiCobordism::twoBodyResidualsPerCase,
           py::call_guard<py::gil_scoped_release>(),
           "One residual PER CASE on the live complex, in the order the cases "
           "were set; empty when none are. The sum is what the drive minimises, "
           "but a sum cannot show a step that improves one state at another's "
           "expense -- the interesting failure mode when fitting a map.")
      .def("two_body_residual_over_cases", &MultiCobordism::twoBodyResidualOverCases,
           py::call_guard<py::gil_scoped_release>(),
           "The two-body residual summed over the cases, or the single target's "
           "residual when none are set.")
      .def_static("enumerate_move_specifications",
                  [](const std::shared_ptr<Spacetime> &spacetime,
                     bool with_dispositions, bool with_surgery) {
                    return MultiCobordism::enumerateMoveSpecifications(
                        spacetime, with_dispositions, with_surgery);
                  },
                  py::arg("spacetime"), py::arg("with_dispositions") = false,
                  py::arg("with_surgery") = true,
                  "EVERY candidate move on `spacetime` as (kind, site) pairs, "
                  "rather than a sample of them. The random draw picks a KIND "
                  "uniformly and only then a site, so n draws is n/6 samples "
                  "per kind against site sets of order the cell count; and "
                  "three of the four Pachner kinds draw their site from "
                  "Spacetime::rng, which no seed controls. Enumeration is "
                  "complete and reproducible. The Pachner kinds come back as "
                  "add_at / remove_at / flip_at / iflip_at, whose payload names "
                  "the site; the cone and disposition kinds already name theirs "
                  "and come back unchanged. Every entry is a candidate to "
                  "SCORE, not a promise that it applies -- the gates still "
                  "refuse what they always refused.")
      .def_static("boundary_facets",
                  [](const Spacetime &spacetime) {
                    // A list, not a set: a Python set cannot hold an
                    // unhashable list, so the std::set's order (ascending
                    // vertex tuples) is handed over as a sequence.
                    const auto facets = MultiCobordism::boundaryFacetsOf(spacetime);
                    return std::vector<std::vector<std::uint64_t>>(facets.begin(), facets.end());
                  },
                  py::arg("spacetime"), py::call_guard<py::gil_scoped_release>(),
                  "The boundary of `spacetime` as sorted vertex tuples in ascending order: every "
                  "codimension-one face carried by exactly one top cell.")
      .def("use_fiber_residuals", &MultiCobordism::useFiberResiduals, py::arg("enabled"),
           "Score blocks carrying a fiber-form target by the fiber residual (least-squares leak of the "
           "target images in the band read on the block's own pencil, restricted to the fiber's cells) "
           "instead of the period residual; folded into r_U so both stages descend it. A surface block "
           "(seed_inputs by region) is read on its own surface (block_surface_subcomplex) in the ZERO MODE "
           "of that pencil, whatever contour its fiber stores; an ordinary block on its sub-complex at the "
           "fiber's contour (default: the lowest band above the zero mode). Off by default.")
      .def("uses_fiber_residuals", &MultiCobordism::usesFiberResiduals)
      .def("score_whole_complex_leak", &MultiCobordism::scoreWholeComplexLeak, py::arg("enabled"),
           "Also score a MARKED block by the leak of its input state in the zero mode of the WHOLE "
           "cobordism (input_state_residual), ADDED to the block's own-Laplacian residual under "
           "use_fiber_residuals and carried at the same input residual weight. The two answer different "
           "questions: own_state_residual builds the Laplacian of ONE torus in isolation and asks whether "
           "that torus, judged alone, still represents the state it was handed, while input_state_residual "
           "builds the Laplacian of the entire cobordism and asks how much of the input coefficients fails "
           "to lie in its zero mode. Scoring only the first leaves the second unconstrained and it drifts "
           "upward. Decisive when the boundary is HELD: a block whose edges cannot move cannot change "
           "shape, so its own residual is identically zero and the weight multiplies nothing. Enabling "
           "this holds nothing fixed -- the leak is a term the relaxation trades against, not a "
           "constraint. Off by default.")
      .def("scores_whole_complex_leak", &MultiCobordism::scoresWholeComplexLeak)
      .def("set_fiber_phase_descent", &MultiCobordism::setFiberPhaseDescent, py::arg("enabled"),
           "Also descend the degree-0 link phases through the analytic fiber gradient (off by default).")
      .def("fiber_phase_descent", &MultiCobordism::fiberPhaseDescent)
      .def("fiber_residual_for_input_block", &MultiCobordism::fiberResidualForInputBlock, py::arg("index"),
           py::call_guard<py::gil_scoped_release>(),
           "The fiber residual of one input block on the live complex: a surface block's own Laplacian "
           "(its surface with the live lengths) at the zero mode, an ordinary block's sub-complex at the "
           "fiber's contour.")
      .def("fiber_residual_gradient",
           [](const MultiCobordism &self, const BoundaryFiber &fiber) {
             py::gil_scoped_release release;
             const auto g = self.fiberResidualGradientOn(self.spacetime(), fiber);
             return std::make_pair(g.lengths, g.phases);
           }, py::arg("fiber"),
           "Analytic gradient of the fiber residual of `fiber` on the live complex: (lengths, phases) "
           "in EdgeList order, each entry (d/dRe, d/dIm) packed as a complex number; phases empty above degree 0.")
      .def("two_body_residual_gradient",
           [](const MultiCobordism &self) {
             if (!self.twoBodyTarget()) throw std::logic_error("no two-body target");
             py::gil_scoped_release release;
             const auto g = self.twoBodyResidualGradientOn(self.spacetime(), *self.twoBodyTarget());
             return std::make_pair(g.lengths, g.phases);
           }, "Legacy analytic gradient of the frame-transfer two-body residual on the live "
              "complex, independent of the selected readout. fiber_mode_ascent dispatches "
              "the objective's selected readings.")
      .def("fiber_mode_ascent",
           [](const MultiCobordism &self) {
             py::gil_scoped_release release;
             const auto g = self.fiberModeAscent();
             return std::make_pair(g.lengths, g.phases);
           }, "The analytic ascent of every fiber-mode term of r_U on the live complex. "
              "Refuses when a selected two-body reading has no analytic gradient.")
      .def("attach_input_fiber", &MultiCobordism::attachInputFiber, py::arg("index"), py::arg("fiber"),
           py::arg("cells"),
           "Attach a piped input fiber to THIS complex's cells (one per fiber row, in the attachment "
           "order = the attachment permutation); refuses overlaps with other attached input fibers. "
           "Clears the block's frame (set_input_frame), whose rows referred to the previous attachment.")
      .def("set_input_frame", &MultiCobordism::setInputFrame, py::arg("index"), py::arg("cells"),
           py::arg("images"), py::arg("dual_images"),
           "Set the frame (BlockFrame, spec D3) the two-body transfer is read in on input block `index`: "
           "images and dual images on `cells`, which must be the block's attached fiber cells in the "
           "attachment order. Stated by the caller at attachment (a torus's period frame through the "
           "collar's id map, its dual from input_frame_dual) and held constant by the engine. With both "
           "input blocks framed, read_two_body, two_body_residual and the two-body gradient read the "
           "transfer in the frames (r_A x r_B; the gradient differentiates the pencil operator only). "
           "Refuses by name: no attached fiber, cells that are not the fiber's in its order, row counts "
           "other than the cell count, differing or zero column counts, non-finite entries.")
      .def("input_frame", [](const MultiCobordism &self, std::size_t i) { return self.inputFrame(i); },
           py::arg("index"), "The frame of input block `index` (BlockFrame), or None.")
      .def("clear_input_frame", &MultiCobordism::clearInputFrame, py::arg("index"),
           "Drop the frame of input block `index`: the transfer returns to identity frames on the cells.")
      .def("set_input_marking", &MultiCobordism::setInputMarking, py::arg("index"), py::arg("cycles"),
           py::arg("coefficients"),
           "Set the MARKING of input block `index` with its input coefficients (BlockMarking, spec D2/D3 "
           "as revised): `cycles` in host vertex ids as lists of directed steps (u, v) (the monodromy "
           "convention: +h(u,v) when u < v, -h(u,v) otherwise), `coefficients` one per cycle -- (1, tau_in) "
           "for an input torus. Each cycle is ordered into one closed walk and rotated to the common base "
           "point. From then on the block's term of r_U is input_state_residual (the leak of the "
           "coefficients written through the LIVE frame in the whole's zero mode on the block's edges, in "
           "place of the own-kernel leak), its stage-2 direction carries the frame's motion, and the "
           "two-body transfer is read in the derived frame. Refuses by name: no attached fiber, a fiber "
           "not at degree 1, no cycle, a coefficient count other than the cycle count, non-finite or "
           "all-zero coefficients, a self-loop, a step that is not a live edge inside the block, a cycle "
           "that is not one closed walk, cycles sharing no vertex.")
      .def("input_marking", [](const MultiCobordism &self, std::size_t i) { return self.inputMarking(i); },
           py::arg("index"), "The marking of input block `index` (BlockMarking), or None.")
      .def("clear_input_marking", &MultiCobordism::clearInputMarking, py::arg("index"),
           "Drop the marking of input block `index`: its scoring and transfer return to what they were.")
      .def_static("derive_frame", &MultiCobordism::deriveFrame, py::arg("block"), py::arg("spacetime"),
           py::call_guard<py::gil_scoped_release>(),
           "The live frame of a marked block on `spacetime` (DerivedFrame); read-only on the geometry.")
      .def("derive_input_frame", &MultiCobordism::deriveInputFrame, py::arg("index"),
           py::call_guard<py::gil_scoped_release>(),
           "derive_frame of input block `index` on the live complex.")
      .def("input_state_residual", &MultiCobordism::inputStateResidual, py::arg("index"),
           py::call_guard<py::gil_scoped_release>(),
           "The leak of the OUTPUT-state read at input block `index`: the leak of its "
           "input coefficients, written on its edges through its live frame, in the zero mode of the "
           "ENTIRE cobordism at the whole's harmonic contour restricted to those edges; 1.0 when the "
           "frame cannot be derived. Reported, not scored: r_U scores own_state_residual.")
      .def("own_state_residual", &MultiCobordism::ownStateResidual, py::arg("index"),
           py::call_guard<py::gil_scoped_release>(),
           "The block residual of input block `index` (qubit cobordism spec D2): 1 - |<psi(tau_in)|psi(tau_hat)>|^2 "
           "with tau_hat the ratio of the transported periods of the holomorphic form of the block's OWN "
           "Laplacian on its live surface (block_qubit) over the marking, and (1, tau_in) the block's input "
           "coefficients; zero exactly when the torus represents its input state on its own; 1.0 when the "
           "block has no surface or the read is refused. What r_U scores for the block at the input residual "
           "weight.")
      .def("own_state_residual_gradient",
           [](const MultiCobordism &self, std::size_t index) {
             if (index >= self.inputs().size()) throw std::out_of_range("input block index out of range");
             py::gil_scoped_release release;
             const auto g = self.ownStateResidualGradientOn(self.spacetime(), self.inputs()[index]);
             return std::make_pair(g.lengths, g.phases);
           }, py::arg("index"),
           "Analytic gradient of own_state_residual(index) on the live complex: (lengths, phases) in "
           "EdgeList order, each entry (d/dRe, d/dIm) packed as a complex number, through the qubit's own "
           "analytic tau derivative (SimplicialQubit.tau_derivative) on the block's edges; zero on bulk "
           "edges; phases empty (tau is gauge invariant). The stage-2 direction of a marked block.")
      .def("block_qubit", static_cast<observables::SimplicialQubit (MultiCobordism::*)(std::size_t) const>(
                              &MultiCobordism::blockQubit),
           py::arg("index"), py::call_guard<py::gil_scoped_release>(),
           "The live surface of input block `index` read as the SimplicialQubit over its marking, in the "
           "orientation with A . B = +1: the object own_state_residual scores and the qubit "
           "read reports, so the two cannot disagree.")
      .def("read_input_state", &MultiCobordism::readInputState, py::arg("index"),
           py::call_guard<py::gil_scoped_release>(),
           "The state at input block `index` (InputStateRead): the coefficients of the whole's zero mode "
           "in the block's live frame, next to its input coefficients and its residual.")
      .def("input_state_residual_gradient",
           [](const MultiCobordism &self, std::size_t index) {
             if (index >= self.inputs().size()) throw std::out_of_range("input block index out of range");
             py::gil_scoped_release release;
             const auto g = self.inputStateResidualGradientOn(self.spacetime(), self.inputs()[index]);
             return std::make_pair(g.lengths, g.phases);
           }, py::arg("index"),
           "Analytic gradient of input_state_residual(index) on the live complex: (lengths, phases) in "
           "EdgeList order, each entry (d/dRe, d/dIm) packed as a complex number, the whole's band "
           "derivative with the MOVING target (the frame's own derivative on the block's edges); phases "
           "empty at degree 1.")
      .def_static("dual_frame", &MultiCobordism::dualFrame, py::arg("complex"), py::arg("degree"),
           py::arg("cells"), py::arg("images"),
           "The dual of a frame under the transpose pairing of `complex`'s own chain-level Whitney "
           "pencil at `degree`: with M_k the pencil's Whitney mass matrix (CovariantChainHodge.Minv, "
           "PencilLayer.pencil(...).B) restricted to `cells` and B = Z.T @ M_k @ Z the frame's pairing, "
           "Z^vee = Z @ inv(B).T, so that Z^vee.T @ M_k @ Z == I (the BlockFrame contract; the "
           "canonical left frame at U = 1). Read-only on the complex. Refuses by name a "
           "null complex, a degree above its dimension, a frame without columns, a row count other than "
           "the cell count, a cell absent at that degree, and a singular pairing (an isotropic frame).")
      .def("input_frame_dual", &MultiCobordism::inputFrameDual, py::arg("index"), py::arg("images"),
           "dual_frame of `images` on input block `index`'s OWN pencil - its own complex with the live "
           "lengths (block_surface_subcomplex for a surface block, the sub-complex for an ordinary one) "
           "on its attached fiber cells at the fiber's degree. Call it at attachment.")
      .def("set_two_body_target", &MultiCobordism::setTwoBodyTarget, py::arg("chi"),
           py::arg("choi_decomposed") = true,
           "The two-body target chi on the pair of attached frames; choi_decomposed selects the reading "
           "(vec(T_AB) as a state on the pair space, or the operator T_AB). Scored inside r_U under "
           "use_fiber_residuals once two input fibers are attached. With two fibers attached, a shape "
           "other than the transfer's (the frames' ranks r_A x r_B when both blocks carry a frame, the "
           "cell counts otherwise) is refused by name.")
      .def("two_body_target", &MultiCobordism::twoBodyTarget)
      .def("two_body_residual", &MultiCobordism::twoBodyResidual, py::call_guard<py::gil_scoped_release>(),
           "The sum of the selected projective two-body readout leaks on the live complex.")
      .def("read_two_body", &MultiCobordism::readTwoBody, py::call_guard<py::gil_scoped_release>(),
           "The bulk between two attached frames, or between two paired frames when four inputs are "
           "attached: T_AB, vec(T_AB), Schmidt spectrum and rank, the reversal residual, and every "
           "input block's fiber residual. The four-input read is a direct-sum geometric diagnostic, "
           "so its fit residual is NaN rather than an inferred tensor-product score.")
      .def("set_output_state_target", &MultiCobordism::setOutputStateTarget, py::arg("state"),
           "The state the WHOLE complex's harmonic form is meant to be (ReadoutMode.WHOLE). Its "
           "dimension is the claim: a rank-2 harmonic space carries a 2-dimensional state, a "
           "rank-4 one a 4-dimensional state, and the reading refuses a target of any other size "
           "rather than fitting it. Unset, the two-body target is used.")
      .def_property_readonly("output_state_target", &MultiCobordism::outputStateTarget)
      .def("operator_residual",
           [](const MultiCobordism &self, const Eigen::MatrixXcd &twoStateVector) {
             return self.operatorResidualOn(
                 self.spacetime(),
                 MultiCobordism::TwoBodyTarget{Eigen::MatrixXcd(), true, twoStateVector});
           },
           py::arg("two_state_vector"), py::call_guard<py::gil_scoped_release>(),
           "The projective leak of a target expressed in the paired direct-sum frames. "
           "Equal dimensions do not identify this with a tensor-product two-qubit operator.")
      .def("set_whole_pairing", &MultiCobordism::setWholePairing, py::arg("pairing"),
           "How the whole-complex reading pairs harmonic columns against the input blocks. "
           "PERIODS integrates each column over the marked cycles: how a state is defined on a "
           "boundary torus, but entirely topological on the three-dimensional bulk, where a change "
           "of metric moves the harmonic representative by exactly a coboundary whose period "
           "around a closed cycle telescopes to zero. GRAM contracts through the chain metric "
           "instead, f_c^T M_1 Z_a, the transpose pairing the harmonic Gram already uses. Same "
           "shape, so the fit and the projective leak downstream are unchanged. PERIODS by default.")
      .def_property_readonly("whole_pairing", &MultiCobordism::wholePairing)
      .def("set_readout_modes", &MultiCobordism::setReadoutModes, py::arg("modes"),
           "The readings SUMMED into the two-body residual. TRANSFER is the "
           "coupling block between the two boundary frames; BULK is "
           "ker L1(W - dW), the boundary REMOVED; WHOLE is ker L1(W), the "
           "boundary INCLUDED. Part of the OBJECTIVE, not the reporting: this "
           "residual is a term in r_U, so it prices stage-1 moves and drives "
           "stage-2 descent. Repeated values are normalized to their first "
           "occurrence; empty is refused.")
      .def_property_readonly("readout_modes", &MultiCobordism::readoutModes)
      .def_property_readonly("readouts_have_analytic_gradient",
                             &MultiCobordism::readoutsHaveAnalyticGradient,
                             "Whether every SELECTED reading has an analytic gradient, so the stage-2 "
                             "ascent is the direction of the objective actually being minimised. "
                             "Transfer and whole-harmonic reads have one; bulk and paired-operator "
                             "reads fall back to the numerical ascent of r_U.")
      .def("whole_harmonic_residual_gradient",
           [](const MultiCobordism &self, const Eigen::MatrixXcd &chi) {
             const auto g = self.wholeHarmonicResidualGradientOn(
                 self.spacetime(), MultiCobordism::TwoBodyTarget{chi, true, Eigen::MatrixXcd()});
             return g.lengths;
           },
           py::arg("chi"), py::call_guard<py::gil_scoped_release>(),
           "The analytic gradient of whole_harmonic_residual with respect to each edge's squared "
           "length, in the live complex's edge order. Zero where the reading refuses, and zero "
           "where the marked cycles do not pin the harmonic form.")
      .def("whole_harmonic_residual",
           [](const MultiCobordism &self, const Eigen::MatrixXcd &chi, bool choiDecomposed) {
             return self.wholeHarmonicResidualOn(
                 self.spacetime(), MultiCobordism::TwoBodyTarget{chi, choiDecomposed});
           },
           py::arg("chi"), py::arg("choi_decomposed") = true,
           py::call_guard<py::gil_scoped_release>(),
           "The projective leak of chi against the WHOLE cobordism's degree-1 harmonic form, "
           "the form the input blocks' markings and coefficients determine. 1.0 when the "
           "harmonic space cannot carry a target of that dimension; see "
           "whole_harmonic_obstruction for the reason.")
      .def_property_readonly("whole_harmonic_obstruction",
                             &MultiCobordism::wholeHarmonicObstruction,
                             "Why the last harmonic readout could not name a state, or empty.")
      .def("bulk_operator_residual",
           [](const MultiCobordism &self, const Eigen::MatrixXcd &chi, bool choiDecomposed) {
             return self.bulkOperatorResidualOn(
                 self.spacetime(), MultiCobordism::TwoBodyTarget{chi, choiDecomposed});
           },
           py::arg("chi"), py::arg("choi_decomposed") = true,
           py::call_guard<py::gil_scoped_release>(),
           "The projective leak of chi against the operator promoted from "
           "ker L1(W - dW) through a Choi frame. 1.0 when the framed kernel is "
           "not rank one, so no operator is identifiable.")
      .def("set_whole_complex_fiber_target", &MultiCobordism::setWholeComplexFiberTarget, py::arg("fiber"),
           "A fiber-form target carried by the WHOLE complex on the fiber's cells and contour (default: the "
           "lowest band above the flat zero mode); scored inside r_U under use_fiber_residuals.")
      .def("whole_complex_fiber_target", &MultiCobordism::wholeComplexFiberTarget)
      .def("whole_complex_fiber_residual", &MultiCobordism::wholeComplexFiberResidual,
           py::call_guard<py::gil_scoped_release>())
      .def("read_whole_complex_fiber",
           [](const MultiCobordism &self, std::optional<chainhodge::Contour> contour, double kappa) {
             py::gil_scoped_release release;
             return self.readWholeComplexFiber(contour ? &*contour : nullptr, kappa);
           },
           py::arg("contour") = std::nullopt, py::arg("kappa") = 10.0,
           "The fiber the whole complex carries on the target's cells: what a downstream node is piped.")
      .def_static("seed_simplex", &MultiCobordism::seedSimplex, py::arg("dimension"),
           py::arg("balanced_edges") = false,
           "A single dimension-simplex host with a uniform Lorentzian metric: the canonical seed.")
      .def_static("seed_from_surfaces", &MultiCobordism::seedFromSurfaces, py::arg("surfaces"),
           "The SurfaceSeed of closed (d-1)-dimensional Spacetimes (e.g. "
           "SimplicialQubit.flat_torus(tau, n, n).spacetime()): one d-dimensional host holding "
           "every surface as its own simplices with its lengths and zero phases on disjoint "
           "vertex id ranges, and no d-cell -- the bulk is drawn afterwards by gated bridges.")
      .def_static("seed_joined_collars", &MultiCobordism::seedJoinedCollars,
                  py::arg("surfaces"), py::arg("layers") = 3,
                  py::arg("twist") = std::vector<std::uint64_t>{},
                  "Two collars joined along a removed tetrahedron: FOUR boundary surfaces on one "
                  "connected manifold. Gluing along a sphere is a connected sum, which adds no first "
                  "homology, so b_1 = 2 + 2 = 4 with nothing dying on the boundary -- the dimension a "
                  "4-dimensional target needs, since rank(H^1(W) -> H^1(dW)) = b_1(dW)/2. layers must "
                  "be at least three: a prism cell spans two adjacent layers, so an all-interior cell "
                  "exists only with two interior layers.")
      .def_static("seed_tubed_collars", &MultiCobordism::seedTubedCollars,
                  py::arg("surfaces"), py::arg("layers") = 1,
                  py::arg("twist") = std::vector<std::uint64_t>{},
                  py::arg("tube") = MultiCobordism::TubeSpec{},
                  "Two collars joined by a TUBE (a 1-handle) between their far surfaces: the prism over "
                  "one attachment face of each (TubeSpec), so the far boundary is ONE genus-two surface, "
                  "the connected sum of the two far tori through the tube, and dW = T^2 + T^2 + Sigma_2. "
                  "The tube joins two components and adds no loop: b_1(W) = 4 = b_1(dW)/2 with no class "
                  "invisible to the boundary. Euclidean tube geometry declared from the attachment faces' "
                  "own lengths, the waist and the length (the knobs of the neck); gated once as a whole "
                  "by dualComplexIsValid and refused by name.")
      .def_static("seed_collar", &MultiCobordism::seedCollar, py::arg("surface_a"), py::arg("surface_b"),
           py::arg("layers") = 1, py::arg("twist") = std::vector<std::uint64_t>{},
           "The SurfaceSeed of the COLLAR between two surfaces of identical combinatorics: "
           "T^2 x I over their shared triangulation (Spacetime.prismCells, `layers` product layers), "
           "layer 0 = surface A, last layer = surface B, the surfaces' lengths verbatim, the auto-wired "
           "length on every other edge, zero phases, gated once as a whole by dualComplexIsValid and "
           "refused by name; a combinatorial mismatch is refused by name. Seed both id sets as input "
           "blocks with seed_inputs; bridge_phase_complete() then holds by construction.")
      .def_static("block_surface", &MultiCobordism::blockSurface, py::arg("block"), py::arg("spacetime"),
           "The block's own surface: its (d-1)-faces (registered ones and facets of top cells "
           "inside the block) and its edges inside its vertex set, as sorted vertex tuples.")
      .def_static("block_surface_subcomplex", &MultiCobordism::blockSurfaceWithGeometry, py::arg("block"),
           py::arg("spacetime"),
           "A surface block's OWN complex with the host's geometry: block_surface's faces as "
           "the top cells of a fresh (d-1)-dimensional Spacetime keeping the host's vertex ids, every "
           "edge with the host's current length and phase. Its Laplacian is the block's own Laplacian: "
           "the block's fiber residual is read in its zero mode and the residual's gradient is taken on "
           "it (mapped back to the host's edges by vertex pair). None when the block has no face.")
      .def_static("monodromy", &MultiCobordism::monodromy, py::arg("spacetime"), py::arg("marking_a"),
           py::arg("marking_b"), py::call_guard<py::gil_scoped_release>(),
           "The MonodromyRead of `spacetime` between two markings given in host vertex ids as "
           "cycles of directed steps (u, v): a step contributes +h(u,v) when u < v and -h(u,v) "
           "otherwise. Read-only; the Whitney pencil's harmonic contour is the zero mode.")
      .def_static("restriction", &MultiCobordism::restriction, py::arg("spacetime"), py::arg("markings"),
           py::call_guard<py::gil_scoped_release>(),
           "The RestrictionRead of `spacetime` over a list of markings, each given in host vertex "
           "ids as cycles of directed steps (u, v) with the same step convention as `monodromy`: "
           "one harmonic basis and every marking's periods from it. Read-only.")
      .def("set_input_fiber", &MultiCobordism::setInputFiber, py::arg("index"), py::arg("fiber"),
           "Attach the fiber form of an input block's target.")
      .def("set_output_fiber", &MultiCobordism::setOutputFiber, py::arg("index"), py::arg("fiber"))
      .def("input_fiber", [](const MultiCobordism &self, std::size_t i) { return self.inputFiber(i); },
           py::arg("index"))
      .def("output_fiber", [](const MultiCobordism &self, std::size_t i) { return self.outputFiber(i); },
           py::arg("index"))
      .def("pin_input_fibers", &MultiCobordism::pinInputFibers, py::arg("degree"), py::arg("epsilon") = 1e-10,
           py::arg("restarts") = 64, py::arg("max_growth") = 4, py::arg("seed") = 0,
           py::arg("max_iterations") = 200,
           "Pin the two input blocks' rank-one fibers as boundary data on the union of their cells and "
           "relax the bulk through relax_fixed_boundary_eigenstate.")
      .def("read_output_fiber",
           [](MultiCobordism &self, std::size_t index, int degree,
              std::optional<chainhodge::Contour> contour, double kappa) {
             return self.readOutputFiber(index, degree, contour ? &*contour : nullptr, kappa);
           },
           py::arg("index"), py::arg("degree"), py::arg("contour") = py::none(), py::arg("kappa") = 10.0,
           "Read the fiber form of an output block's target from the live complex (harmonic contour "
           "by default) and store it on the block.")
      .def("relax_boundary_state_pairs",
           &MultiCobordism::relaxBoundaryStatePairs,
           py::arg("degree"), py::arg("input_region"),
           py::arg("input_cells"), py::arg("input_states"),
           py::arg("output_region"), py::arg("output_cells"),
           py::arg("output_states"), py::arg("common_eigenvalue") = true,
           py::arg("epsilon") = 1e-10,
           py::arg("boundary_epsilon") = 1e-10,
           py::arg("restarts") = 64, py::arg("max_growth") = 4,
           py::arg("seed") = 0, py::arg("max_iterations") = 200,
           py::call_guard<py::gil_scoped_release>(),
           "Fit one shared bulk geometry to complete input/output state pairs "
           "on two independently prepared boundary components. Boundary "
           "states and all declared pinned geometry remain exact. The "
           "full-W Rayleigh residual is minimized jointly; by default every "
           "witness is constrained to one common eigenvalue so their span "
           "extends linearly to unseen input combinations.")
      .def("relax_whole_complex_readout_targets",
           &MultiCobordism::relaxWholeComplexReadoutTargets,
           py::arg("degree"), py::arg("region_a"), py::arg("cells_a"),
           py::arg("states_a"), py::arg("region_b"), py::arg("cells_b"),
           py::arg("states_b"), py::arg("readouts"), py::arg("targets"),
           py::arg("common_eigenvalue") = true, py::arg("epsilon") = 1e-10,
           py::arg("boundary_epsilon") = 1e-10, py::arg("restarts") = 64,
           py::arg("max_growth") = 4, py::arg("seed") = 0,
           py::arg("max_iterations") = 200,
           py::call_guard<py::gil_scoped_release>(),
           "Fit one shared bulk geometry so the whole complex carries a "
           "spanning set of common-eigenvalue eigenstates whose restrictions "
           "to BOTH boundary components are the prepared inputs and whose "
           "readouts over the given chains (lists of (cell, coefficient) "
           "pairs) equal the prescribed outputs EXACTLY, by parametrizing "
           "each witness's free amplitudes on the affine solution set of its "
           "readout system. Boundary geometry and amplitudes remain exact; "
           "the residual is the whole-complex Rayleigh residual alone.")
      .def("geometric_operator", &MultiCobordism::geometricOperator,
           py::arg("state_dimension"),
           py::arg("frame_cells") =
               std::vector<std::vector<std::uint64_t>>{},
           py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Target-free Choi promotion of ker L1(W-dW). The ordered frame must "
           "contain d^2 interior edges (or may be omitted when the bulk has "
           "exactly d^2 edges). Promotion succeeds only for a rank-one framed "
           "kernel; otherwise the result carries a specific obstruction.")
      .def_property_readonly("einstein_hilbert_enabled",
           &MultiCobordism::einsteinHilbertEnabled)
      .def_property_readonly("real_squared_lengths_only",
           &MultiCobordism::realSquaredLengthsOnly)
      .def("spacetime", &MultiCobordism::spacetime,
           "The node's LIVE complex. Stage 1 REPLACES the node's spacetime "
           "whenever it commits a move, so a caller holding the shared_ptr it "
           "passed to the constructor keeps the ORIGINAL complex, frozen from "
           "the first committed move onward. Read this — not the constructor "
           "argument — whenever the complex is inspected after a drive: the "
           "two diverge silently, and a readout taken from the stale one "
           "describes a complex the node stopped using.")
      .def("objective", &MultiCobordism::objective)
      .def("hodge_entropy", &MultiCobordism::hodgeEntropy,
           "Sum of normalized positive-operator Hodge entropies over the "
           "configured degrees. Observed but not directly minimized by the "
           "joint objective.")
      .def("hodge_entropy_stationarity",
           &MultiCobordism::hodgeEntropyStationarity,
           "Sum_k ||grad_z S_Hodge,k||^2, the entropy half of the joint "
           "stationarity objective.")
      .def("set_objective", &MultiCobordism::setObjective,
           py::arg("objective"), py::keep_alive<1, 2>(),
           "Inject the functional this node descends. The engine calls through "
           "it and knows nothing about which objective it holds.\n\n"
           "The node keeps the objective alive for as long as it holds it, so "
           "a Python-defined objective survives the caller dropping its own "
           "last reference -- otherwise the Python half of a subclass could be "
           "collected while the engine still descends through it.")
      .def_property_readonly("objective_spec", &MultiCobordism::objectiveSpec,
           "The injected functional. Never null: construction installs a "
           "default.")
      .def_property_readonly("objective_name", &MultiCobordism::objectiveName,
           "The injected objective's stable identifier, as stamped on records.")
      .def("set_pinned_objective", &MultiCobordism::setPinnedObjective,
           py::arg("objective"), py::keep_alive<1, 2>(),
           "Inject an ADDITIONAL objective holding a pinned region, alongside "
           "the bulk objective. Optional: with none supplied the pinned "
           "region's objective IS the bulk objective and the run is identical "
           "to a single-objective one. The region is not named here -- the "
           "objective declares its own scope, whose RegionHandle can only come "
           "from region_handle, so a mis-spelling cannot reach this call.")
      .def_property_readonly("pinned_objective",
           &MultiCobordism::pinnedObjective,
           "The additional pinned-region objective, or None where none is "
           "supplied.")
      .def("clear_pinned_objective", &MultiCobordism::clearPinnedObjective,
           "Drop the pinned-region objective, returning the node to a single "
           "objective scoring the whole cobordism.")
      .def_property_readonly("objective_contributions",
           &MultiCobordism::objectiveContributions,
           "Every objective's decomposition, in evaluation order: the bulk "
           "objective first, then the pinned-region objective where one is "
           "supplied. Summing the terms reproduces objective_terms exactly, so "
           "a reader can tell whether descent came from the bulk or from the "
           "pinned region.")
      .def("region_handle", &MultiCobordism::regionHandle, py::arg("name"),
           "Mint a RegionHandle for a DECLARED region. The only way to obtain "
           "a non-empty handle; an undeclared name raises BY NAME rather than "
           "producing a scope that silently matches nothing.")
      .def_property_readonly("objective_is_target_conditioned",
           &MultiCobordism::objectiveIsTargetConditioned,
           "Whether the injected objective's value depends on prescribed "
           "boundary targets rather than on the geometry alone.")
      .def("set_hodge_entropy_phase_mode",
           &MultiCobordism::setHodgeEntropyPhaseMode, py::arg("mode"),
           "Choose full complex L or entrywise |L| for entropy only; this never "
           "projects the live complex edge geometry.")
      .def_property_readonly("hodge_entropy_phase_mode",
                             &MultiCobordism::hodgeEntropyPhaseMode)
      .def("set_connection_entropy_weight",
           &MultiCobordism::setConnectionEntropyWeight, py::arg("weight"),
           "Declare the weight on the connection-entropy stationarity term -- "
           "the only term with a gradient in the connection phase. Zero by "
           "default, so a node acquires phase dynamics only when asked.")
      .def_property_readonly("connection_entropy_weight",
                             &MultiCobordism::connectionEntropyWeight)
      .def("set_hodge_entropy_weight", &MultiCobordism::setHodgeEntropyWeight,
           py::arg("weight"))
      .def_property_readonly("hodge_entropy_weight",
                             &MultiCobordism::hodgeEntropyWeight)
      .def("set_hodge_degrees", &MultiCobordism::setHodgeDegrees,
           py::arg("degrees"), py::arg("weights") = std::vector<double>{},
           "Declare the Laplacian degrees k the Hodge entropy term is summed "
           "over, and optionally a weight per degree. These are configured "
           "HERE and read from nowhere else -- the register degrees, which "
           "answer the unrelated question of where a register is constructed, "
           "never supply them, not even as a fallback. The default is [0]. An "
           "empty weights list means uniform 1; a non-empty one must match the "
           "degree list in length. Raises on an empty degree list, a negative "
           "or repeated degree, or a mismatched weight list.")
      .def_property_readonly("hodge_degrees", &MultiCobordism::hodgeDegrees,
           "The declared Hodge degrees, in declaration order.")
      .def_property_readonly("hodge_degree_weights",
                             &MultiCobordism::hodgeDegreeWeights,
           "The declared per-degree weights, or empty for uniform.")
      .def_property_readonly("hodge_degree_contributions",
                             &MultiCobordism::hodgeDegreeContributions,
           "The Hodge stationarity term broken down by declared degree, so a "
           "reader can tell WHICH degree the descent came from rather than "
           "only the total. Empty for an objective with no Hodge term.")
      .def("set_regge_weight", &MultiCobordism::setReggeWeight,
           py::arg("weight"))
      .def_property_readonly("regge_weight", &MultiCobordism::reggeWeight)
      .def("set_input_residual_weight", &MultiCobordism::setInputResidualWeight,
           py::arg("weight"))
      .def("seed_inputs",
           py::overload_cast<const std::vector<std::uint64_t> &>(&MultiCobordism::seedInputs),
           py::arg("seeds"),
           "Seed one INPUT block per seed vertex (region = the seed's cell neighbourhood).")
      .def("seed_inputs",
           py::overload_cast<const std::vector<std::vector<std::uint64_t>> &>(
               &MultiCobordism::seedInputs),
           py::arg("regions"),
           "Seed one INPUT block per explicit vertex region -- the surface inputs of a "
           "seed_from_surfaces host (each block is marked `surface`).")
      .def("has_surface_inputs", &MultiCobordism::hasSurfaceInputs,
           "At least two input blocks are surface blocks; only then does stage 1 offer bridges.")
      .def("uncovered_input_faces", &MultiCobordism::uncoveredInputFaces,
           "The faces of the surface input blocks that no top cell covers.")
      .def("bridge_phase_complete", &MultiCobordism::bridgePhaseComplete,
           "Every surface face has exactly one top cell on it and getBoundary() is exactly the "
           "union of the surface faces: the boundary of W is the surfaces. True by construction "
           "on a collar seed; false without surface inputs.")
      .def("seed_outputs", &MultiCobordism::seedOutputs, py::arg("seeds"))
      // Long pure-C++ compute: release the GIL so a background thread can drive
      // a pass without blocking the main thread.
      .def_static("depth_schedule", &MultiCobordism::depthSchedule,
                  py::arg("max_lookahead"), py::arg("combinatorial_breadth"),
                  "The depth ladder one stage-1 update walks, in the order it "
                  "walks it: ascending 1..max_lookahead by default, or "
                  "descending combinatorial_breadth..1 when a breadth is "
                  "named.")
      .def("run_stage1", &MultiCobordism::runStage1, py::arg("max_steps") = 200,
           py::arg("n_candidate_moves") = 12, py::arg("grow_boundaries") = false,
           py::arg("max_lookahead") = 1,
           py::arg("combinatorial_breadth") = 0,
           py::call_guard<py::gil_scoped_release>(),
           "max_lookahead: when a batch of single moves finds no improvement, "
           "the search deepens iteratively -- 2-move sequences, then 3, up to "
           "this many moves -- committing an F-lowering sequence as a whole "
           "(1 = single moves only). "
           "combinatorial_breadth: non-zero runs the depth ladder the other "
           "way round -- sequences of exactly that many moves are searched "
           "FIRST, and the search backs off one move at a time only when "
           "nothing at the current breadth lowers F, down to single moves. "
           "It asks whether a composition of that length improves a complex "
           "no shorter one improves. 0 (the default) leaves the ascending "
           "max_lookahead schedule in place. With n_candidate_moves <= 0 the "
           "search at every breadth is exhaustive, which costs the move "
           "space raised to the breadth."
           )
      .def("run_stage2", &MultiCobordism::runStage2, py::arg("beta") = 1.0,
           py::arg("max_iters") = 200, py::arg("alpha0") = 0.05,
           py::arg("tolerance") = 1e-12,
           py::call_guard<py::gil_scoped_release>(),
           "Stage 2 (geometric): relax the full complex squared edge coordinates "
           "z=l^2 under the selected objective. Derivatives are subtracted from "
           "z itself, then written to Edge's stored l on the nearest square-root "
           "branch; no imaginary component or phase is projected away. A real "
           "backtracking scale accepts only exact objective decreases of at least "
           "the absolute tolerance. Read last_stage2_stationary to distinguish "
           "line-search stationarity from the max_iters budget. Returns F trace.")
      .def("run", &MultiCobordism::run, py::arg("max_iters") = 200,
           py::arg("n_candidate_moves") = 12,
           py::arg("grow_boundaries") = false, py::arg("beta") = 1.0,
           py::arg("alpha0") = 0.05, py::arg("tolerance") = 10e-9,
           py::arg("max_lookahead") = 1,
           py::arg("relax_budget_per_move") = 10,
           py::arg("combinatorial_breadth") = 0,
           py::call_guard<py::gil_scoped_release>(),
           "The combined drive: each iteration takes ONE combinatorial stage-1 "
           "update (a best-dF move, deepening to max_lookahead-move sequences "
           "on a stall) then relaxes the geometry FULLY -- stage-2 updates "
           "repeat until the absolute-improvement test at tolerance (default "
           "10e-9) reports diminishing returns -- so every move is proposed "
           "from, and leaves behind, relaxed geometry. Exit: once the register "
           "is carried + stationary, or the moves have had no effect for a few "
           "consecutive iterations, the LAST relaxation re-runs at the tight "
           "1e-12; if it still finds descent the exit was premature and the "
           "loop continues -- only a state stationary at 1e-12 exits. max_iters "
           "is the hard budget cap. n_candidate_moves/grow_boundaries/"
           "max_lookahead parameterize the combinatorial half exactly as in "
           "run_stage1; beta/alpha0/tolerance the geometric half exactly as in "
           "run_stage2. beta is stored before either half, so the F trace is "
           "coherent at every non-negative value. "
           "relax_budget_per_move caps the stage-2 updates after each "
           "committed move (and the tight exit re-check); the stationarity "
           "test is the real terminator, the cap only bounds slow descent "
           "tails of threshold-sized line-search micro-steps. "
           "last_stage2_stationary reports the LAST geometric update's outcome. "
           "Returns the combined F trace. "
           "combinatorial_breadth: non-zero runs the depth ladder the other "
           "way round -- sequences of exactly that many moves are searched "
           "FIRST, and the search backs off one move at a time only when "
           "nothing at the current breadth lowers F, down to single moves. "
           "It asks whether a composition of that length improves a complex "
           "no shorter one improves. 0 (the default) leaves the ascending "
           "max_lookahead schedule in place. With n_candidate_moves <= 0 the "
           "search at every breadth is exhaustive, which costs the move "
           "space raised to the breadth."
           )
      .def_property("should_propose_surgery",
                    &MultiCobordism::shouldProposeSurgery,
                    &MultiCobordism::setShouldProposeSurgery,
                    "Whether the stage-1 move draw and the exhaustive "
                    "enumeration offer the surgical kinds (cone-out, cone-in, "
                    "the timelike cone-in). False leaves the four Pachner "
                    "kinds alone (and the disposition flip, when dispositions "
                    "are proposed): the moves that keep the complex the "
                    "manifold it is, with the boundary it has. True by "
                    "default.")
      .def_property_readonly("should_propose_dispositions",
                             &MultiCobordism::shouldProposeDispositions,
           "Whether the stage-1 move draw also proposes CAUSAL DISPOSITIONS "
           " -- a timelike cone-in and a disposition flip on an existing "
           "edge. Both are ordinary candidate moves: drawn at random, scored by "
           "deltaF, committed only when they lower F. Nothing prescribes causal "
           "structure; the objective decides whether it wants any.\n\n"
           "They remain useful discrete proposals across causal sectors. The "
           "complex-z Stage 2 may also rotate around z=0 continuously; it does "
           "not project the imaginary component away.\n\n"
           "Enabled by default in the constructor.")
      .def_property_readonly("st", &MultiCobordism::spacetime,
          R"doc(The node's CURRENT complex. Re-read it after every drive call.

Do not cache this handle across a drive. run_stage1 commits an accepted move by
REPLACING the node's complex (spacetime_ = build(bestSnapshot)) rather than
mutating it in place, so a handle taken before the call keeps referring to the
old complex while the node moves on. run_stage2, build_step and the directed
cone probes reach the same reassignment.

A stale handle fails SILENTLY: every read succeeds and returns self-consistent
values -- for a complex the node no longer holds. Note that objective() and
r_u(st) are not interchangeable here. objective() reads the node's live complex
internally, while r_u(st) reads whichever complex you hand it, so mixing the two
against a cached handle yields figures that cannot be reconciled with each
other.

Wrong -- st describes the pre-drive complex, so every later read is stale:

    st = node.st
    node.run_stage1(180, 8, 15, True)
    holes = len(MultiCobordism.emergent_holes(st, 3))   # the OLD complex

Right -- re-read after each drive call:

    node.run_stage1(180, 8, 15, True)
    st = node.st
    holes = len(MultiCobordism.emergent_holes(st, 3))   # the node's complex
)doc")
      .def_property_readonly("inputs", &MultiCobordism::inputs,
                             py::return_value_policy::reference_internal,
                             "The emergent input blocks (each a MultiCobordismBlock).")
      .def_property_readonly("outputs", &MultiCobordism::outputs,
                             py::return_value_policy::reference_internal,
                             "The emergent output blocks (each a MultiCobordismBlock).")
      .def_property_readonly("last_stage1_lookahead",
                             &MultiCobordism::lastStage1Lookahead,
                             "Lookahead depth of the LAST stage-1 update's committed "
                             "sequence: 1 = ordinary single move, >1 = the single-move "
                             "batch stalled and an F-lowering multi-move sequence was "
                             "found at this depth, 0 = nothing found at any depth up "
                             "to max_lookahead (a stage-1 stall).")
      .def_property_readonly("last_stage2_stationary",
                             &MultiCobordism::lastStage2Stationary,
                             "True iff no complex-z line-search trial lowered the "
                             "selected objective by the absolute tolerance; False "
                             "if run_stage2 hit its max_iters budget.");
  py::enum_<MultiCobordism::ReadoutMode>(multiCobordismClass, "ReadoutMode",
      "The space a reading takes the state from, named for the space so a name cannot suggest "
      "the wrong one. TRANSFER is (Z_A^v)^T A~_1 Z_B: the whole complex's degree-1 operator read "
      "as the coupling block between the two boundary frames, factorized across them, which is "
      "what lets it carry an entangled target. BULK is ker L1(W - dW), the Laplacian on interior "
      "cells with the boundary REMOVED, read through a Choi frame. WHOLE is ker L1(W), the "
      "boundary INCLUDED, read through the blocks' markings; its rank is b_1(W). Selected as a "
      "SET and summed (set_readout_modes). Part of the OBJECTIVE: this residual is a term in r_U.")
      .value("TRANSFER", MultiCobordism::ReadoutMode::Transfer)
      .value("BULK", MultiCobordism::ReadoutMode::Bulk)
      .value("WHOLE", MultiCobordism::ReadoutMode::Whole)
      .value("OPERATOR", MultiCobordism::ReadoutMode::Operator);

  py::enum_<MultiCobordism::WholePairing>(multiCobordismClass, "WholePairing",
      "How the whole-complex reading pairs harmonic columns against input blocks. PERIODS "
      "integrates over the marked cycles and is topological on a 3-manifold bulk; GRAM contracts "
      "through the chain metric and responds to it.")
      .value("PERIODS", MultiCobordism::WholePairing::Periods)
      .value("GRAM", MultiCobordism::WholePairing::Gram);
  py::enum_<MultiCobordism::BuildAction>(multiCobordismClass, "BuildAction",
      "One canonical solve action a search policy (ProtonSynthesis's build restart loop, a greedy "
      "driver, or the RL agent) composes, so the solve runs through the engine rather than "
      "being re-implemented by each consumer.")
      .value("GROW", MultiCobordism::BuildAction::Grow)
      .value("EVOLVE", MultiCobordism::BuildAction::Evolve)
      .value("RELAX", MultiCobordism::BuildAction::Relax)
      .value("CONE_OUT", MultiCobordism::BuildAction::ConeOut)
      .value("CONE_IN", MultiCobordism::BuildAction::ConeIn);
  py::enum_<MultiCobordism::HolePlacementStrategy>(multiCobordismClass,
      "HolePlacementStrategy",
      "Secondary ordering for the directed cone-out probe (both interior-first): "
      "ADJACENT_HOLES_LAST sends cells sharing vertices with existing holes to the back "
      "(separated register), ADJACENT_HOLES_FIRST to the front (clustered).")
      .value("ADJACENT_HOLES_FIRST", MultiCobordism::HolePlacementStrategy::AdjacentHolesFirst)
      .value("ADJACENT_HOLES_LAST", MultiCobordism::HolePlacementStrategy::AdjacentHolesLast);
  multiCobordismClass
      .def("build_step", &MultiCobordism::buildStep, py::arg("action"),
           py::arg("max_steps") = 30, py::arg("n_candidate_moves") = 8,
           py::arg("stage2_beta") = 1.0,
           py::arg("stage2_max_iters") = 10, py::arg("stage2_alpha0") = 0.05,
           py::arg("hole_placement_strategy") =
               MultiCobordism::HolePlacementStrategy::AdjacentHolesLast,
           // Composes run_stage1/run_stage2 internally (C++ -> C++, so no nested
           // guard); release the GIL here too.
           py::call_guard<py::gil_scoped_release>(),
           "Apply one BuildAction to this node in place (GROW/EVOLVE = run_stage1 with "
           "grow_boundaries true/false; RELAX = run_stage2; CONE_OUT/CONE_IN = the directed "
           "probes) -- the canonical solve step a policy (build, greedy, or RL) composes.")
      .def("directed_cone_out", &MultiCobordism::directedConeOut,
           py::arg("strategy") = MultiCobordism::HolePlacementStrategy::AdjacentHolesLast,
           py::arg("max_open") = 6,
           "Directed gated cone-out: deliberately remove top cells, keeping the opener "
           "that most lowers this node's rU (which absorbs r_state). Returns #holes opened.")
      .def("random_cone_out", &MultiCobordism::randomConeOut, py::arg("count"),
           "Cone out `count` top cells chosen UNIFORMLY AT RANDOM among those valid to remove -- "
           "the BACKSTEP, a deliberate perturbation rather than an improvement. directed_cone_out "
           "keeps the candidate that most lowers rU, which is the greedy rule that walks a run into "
           "a local minimum; nothing here is priced and no candidate is preferred. Valid means the "
           "manifold gate accepts the removal AND the cell touches no declared pinned region, so a "
           "held boundary is not eaten by a perturbation. Draws from the node's seeded generator. "
           "The objective will generally be HIGHER afterwards, which is the point. Returns how many "
           "were removed, fewer than `count` when the valid candidates run out.")
      .def("directed_cone_in", &MultiCobordism::directedConeIn, py::arg("max_close") = 6,
           "Directed gated cone-in: select the register by capping the hole whose removal "
           "most lowers rU. Returns #holes capped.")

      // === pinning: a plain geometric constraint ===
      .def("declare_pinned_region",
           [](MultiCobordism &self, const std::string &name,
              const std::set<std::uint64_t> &vertices) {
             self.declarePinnedRegion({name, vertices});
           },
           py::arg("name"), py::arg("vertices"),
           "Declare a pinned region: a named vertex set held fixed while the rest of the "
           "complex relaxes around it. Re-declaring an existing name replaces it. The "
           "region carries no target and no objective -- it says WHICH cells are held, "
           "never what they are held to.")
      .def("pinned_regions",
           [](const MultiCobordism &self) {
             std::vector<std::pair<std::string, std::set<std::uint64_t>>> regions;
             regions.reserve(self.pinnedRegions().size());
             for (const auto &region : self.pinnedRegions())
               regions.emplace_back(region.name, region.vertices);
             return regions;
           },
           "Every declared pinned region as (name, vertices), in declaration order.")
      .def("clear_pinned_regions", &MultiCobordism::clearPinnedRegions,
           "Drop every declared region, leaving the whole complex free to relax.")
      .def("pinned_vertices", &MultiCobordism::pinnedVertices,
           "The union of every region's vertices.")
      .def("edge_is_pinned", &MultiCobordism::edgeIsPinned, py::arg("a"), py::arg("b"),
           "Whether the edge between a and b is held fixed: true iff some ONE region "
           "contains both endpoints. An edge spanning two distinct regions is bulk.");

  // === modes, the enumerable objective, refinement, and the overlay ===
  py::enum_<MultiCobordism::SimulationMode>(multiCobordismClass, "SimulationMode",
      "The three top-level simulation modes.")
      .value("EMERGENCE", MultiCobordism::SimulationMode::Emergence,
             "Only the base geometric objective (plus the one permitted "
             "state-energy term) drives optimization; every particle and gauge "
             "quantity is a post-hoc observable.")
      .value("SYNTHESIS", MultiCobordism::SimulationMode::Synthesis,
             "A pinned carrier or spectral sector. Never counted as emergence.")
      .value("REPLAY", MultiCobordism::SimulationMode::Replay,
             "Recompute every derived hierarchy and certificate from a "
             "checkpoint and verify that nothing cached changed the result.");
  py::enum_<MultiCobordism::EmergenceSubmode>(multiCobordismClass, "EmergenceSubmode",
      "The two labeled, Gaussian-closed emergence sub-modes.")
      .value("STRICT", MultiCobordism::EmergenceSubmode::Strict,
             "The carried state does not act back on the geometry at all.")
      .value("CERTIFICATES_BLIND_MEAN_FIELD",
             MultiCobordism::EmergenceSubmode::CertificatesBlindMeanField,
             "Only the carried state's energy density may enter the objective; "
             "every particle certificate stays firewalled from it.");

  py::class_<MultiCobordism::ObjectiveContribution>(multiCobordismClass,
      "ObjectiveContribution",
      "One objective's decomposition, labelled by the objective that produced "
      "it and the region it was scored over. The record carries a contribution "
      "per objective rather than one summed record, so a reader can tell "
      "whether descent came from the bulk or from the pinned region.")
      .def(py::init<>())
      .def_readwrite("objective_name",
                     &MultiCobordism::ObjectiveContribution::objectiveName,
                     "The objective's stable identifier.")
      .def_readwrite("region_name",
                     &MultiCobordism::ObjectiveContribution::regionName,
                     "The declared region, or empty for the whole cobordism.")
      .def_readwrite("terms",
                     &MultiCobordism::ObjectiveContribution::terms,
                     "That objective's terms over its own scope.");

  py::class_<MultiCobordism::RefinementIndicators>(multiCobordismClass,
      "RefinementIndicators",
      "The particle-independent geometric/numerical indicators emergence-mode "
      "refinement is allowed to consult.")
      .def(py::init<>())
      .def_readwrite("regge_stationarity_residual",
                     &MultiCobordism::RefinementIndicators::reggeStationarityResidual)
      .def_readwrite("hodge_stationarity_residual",
                     &MultiCobordism::RefinementIndicators::hodgeStationarityResidual)
      .def_readwrite("curvature_concentration",
                     &MultiCobordism::RefinementIndicators::curvatureConcentration)
      .def_readwrite("mesh_quality",
                     &MultiCobordism::RefinementIndicators::meshQuality)
      .def_readwrite("solver_error",
                     &MultiCobordism::RefinementIndicators::solverError);

  py::class_<MultiCobordism::RefinementDecision>(multiCobordismClass,
      "RefinementDecision", "Whether to refine, and which indicator asked.")
      .def_readonly("refine", &MultiCobordism::RefinementDecision::refine)
      .def_readonly("trigger", &MultiCobordism::RefinementDecision::trigger)
      .def_readonly("indicators", &MultiCobordism::RefinementDecision::indicators);

  py::class_<MultiCobordism::AnalysisConfig>(multiCobordismClass, "AnalysisConfig",
      "Analysis-overlay configuration. DISABLED by default: with enabled False "
      "not one line of the overlay runs.")
      .def(py::init<>())
      .def_readwrite("enabled", &MultiCobordism::AnalysisConfig::enabled)
      .def_readwrite("cadence", &MultiCobordism::AnalysisConfig::cadence)
      .def_readwrite("degrees", &MultiCobordism::AnalysisConfig::degrees)
      .def_readwrite("resolutions", &MultiCobordism::AnalysisConfig::resolutions)
      .def_readwrite("frame_history",
                     &MultiCobordism::AnalysisConfig::frameHistory,
                     "Cobordism frames the overlay retains (one pass is one "
                     "frame): what makes a candidate's lifetime, its "
                     "adjacent-frame overlap, its per-frame band and anchor "
                     "families and its lifetime transports measurable rather "
                     "than assumed. 1 means no history.")
      .def_readwrite("lifetime_winding_closure",
                     &MultiCobordism::AnalysisConfig::lifetimeWindingClosure,
                     "\"none\" (an open cobordism segment: the phase is "
                     "reported and the winding stays unknown) or "
                     "\"closed-family\" (the caller DECLARES the world tube "
                     "closed and the winding is read cyclically).")
      .def_readwrite("fock_oracle", &MultiCobordism::AnalysisConfig::fockOracle)
      .def_readwrite("cold_caches", &MultiCobordism::AnalysisConfig::coldCaches);

  multiCobordismClass
      .def_static("objective_term_names", &MultiCobordism::objectiveTermNames,
           "The names of ObjectiveTerms' members, in declaration order -- the "
           "firewall list a structural test asserts against.")
      .def_static("objective_of", &MultiCobordism::objectiveOf, py::arg("terms"),
           "The scalar objective: the plain sum of the declared terms. STATIC "
           "by design -- it cannot reach any analysis state.")
      .def("objective_terms", &MultiCobordism::objectiveTerms,
           "Decompose this node's objective into its declared terms.")
      .def("objective_terms_for", &MultiCobordism::objectiveTermsFor, py::arg("st"),
           "Decompose the objective on an explicit complex.")
      .def("set_simulation_mode", &MultiCobordism::setSimulationMode,
           py::arg("mode"),
           py::arg("submode") = MultiCobordism::EmergenceSubmode::Strict,
           "Select the simulation mode and (for emergence) its labeled "
           "sub-mode. Anything but CERTIFICATES_BLIND_MEAN_FIELD zeroes the "
           "carried-state energy coupling.")
      .def_property_readonly("simulation_mode", &MultiCobordism::simulationMode)
      .def_property_readonly("emergence_submode", &MultiCobordism::emergenceSubmode)
      .def_static("mode_name", &MultiCobordism::modeName, py::arg("mode"))
      .def_static("submode_name", &MultiCobordism::submodeName, py::arg("submode"))
      .def("set_carried_state", &MultiCobordism::setCarriedState,
           py::arg("mode_cells"), py::arg("degree"), py::arg("covariance"),
           "Adopt the carried quasi-free state: the covariance Gamma (flat "
           "row-major) over modes each NAMED by the degree-cell it occupies.")
      .def("clear_carried_state", &MultiCobordism::clearCarriedState)
      .def_property_readonly("has_carried_state", &MultiCobordism::hasCarriedState)
      .def_property_readonly("carried_state_degree",
                             &MultiCobordism::carriedStateDegree)
      .def_property_readonly("carried_state_mode_cells",
                             &MultiCobordism::carriedStateModeCells)
      .def_property_readonly("carried_state_covariance",
                             &MultiCobordism::carriedStateCovariance)
      .def("set_carried_state_energy_weight",
           &MultiCobordism::setCarriedStateEnergyWeight, py::arg("weight"),
           "The mean-field coefficient beta_E. Nonzero requires the "
           "CERTIFICATES_BLIND_MEAN_FIELD emergence sub-mode.")
      .def_property_readonly("carried_state_energy_weight",
                             &MultiCobordism::carriedStateEnergyWeight)
      .def("set_moment_stiffness", &MultiCobordism::setMomentStiffness, py::arg("weight"),
           py::arg("degrees"), py::arg("coefficients"),
           "Declare the spectral-moment stiffness of the geometric action about the CURRENT geometry, the "
           "carrier: its local spectral moments at `degrees` are recorded as the reference, and the objective "
           "gains beta_M sum_k Re S_M,k (HodgeLaplacian.spectralMomentStiffness). Weight 0 removes it.")
      .def_property_readonly("moment_stiffness_weight", &MultiCobordism::momentStiffnessWeight)
      .def_property_readonly("moment_stiffness_degrees", &MultiCobordism::momentStiffnessDegrees)
      .def_property_readonly("moment_stiffness_coefficients", &MultiCobordism::momentStiffnessCoefficients)
      .def("carried_state_energy", &MultiCobordism::carriedStateEnergy, py::arg("st"),
           "E_carried(Gamma, g) = Re tr(Gamma_S h_S(g)) with h_S the Hermitian "
           "part of the metric Hodge operator at the carried degree, restricted "
           "to the carried modes' cells.")
      .def("carried_state_energy_gradient",
           &MultiCobordism::carriedStateEnergyGradient, py::arg("st"),
           "Exact analytic dE/dz per edge in getEdgeList() order.")
      .def("carried_state_purity_defect",
           &MultiCobordism::carriedStatePurityDefect,
           "The purity defect ||Gamma^2 - Gamma||_F of the carried "
           "covariance (NaN with no carried state).")
      .def("carried_state_purity_holds",
           &MultiCobordism::carriedStatePurityHolds, py::arg("tolerance") = 1e-9,
           "Whether the purity certificate HOLDS at the tolerance.")
      .def("set_mean_field_schedule", &MultiCobordism::setMeanFieldSchedule,
           py::arg("dt"), py::arg("steps"),
           "The checkpointed mean-field update schedule.")
      .def_property_readonly("mean_field_step_size",
                             &MultiCobordism::meanFieldStepSize)
      .def_property_readonly("mean_field_steps", &MultiCobordism::meanFieldSteps)
      .def("advance_carried_state", &MultiCobordism::advanceCarriedState,
           py::call_guard<py::gil_scoped_release>(),
           "Advance the carried covariance through meanFieldEvolve under "
           "the SAME generator the energy term uses. Returns the worst purity "
           "defect measured across the steps.")
      .def_static("refinement_indicator_names",
           &MultiCobordism::refinementIndicatorNames,
           "The names of RefinementIndicators' members, in declaration order.")
      .def("refinement_indicators", &MultiCobordism::refinementIndicators,
           "Measure the indicators on this node's current complex.")
      .def("set_refinement_thresholds", &MultiCobordism::setRefinementThresholds,
           py::arg("thresholds"))
      .def_property_readonly("refinement_thresholds",
                             &MultiCobordism::refinementThresholds,
                             py::return_value_policy::copy)
      .def_static("refinement_decision_of", &MultiCobordism::refinementDecisionOf,
           py::arg("indicators"), py::arg("thresholds"),
           "The refinement rule. STATIC over the indicator record by design: it "
           "cannot reach a certificate, fiber, transport, or particle read.")
      .def("refinement_decision", &MultiCobordism::refinementDecision)
      .def("refine_geometry", &MultiCobordism::refineGeometry, py::arg("max_cells") = 1,
           py::call_guard<py::gil_scoped_release>(),
           "Apply geometry refinement when -- and only when -- "
           "refinement_decision() asks, through the EXISTING gated cone-in "
           "surgery. Returns the number of refinement cells committed.")
      .def("set_analysis_config", &MultiCobordism::setAnalysisConfig, py::arg("config"))
      .def_property_readonly("analysis_config", &MultiCobordism::analysisConfig,
                             py::return_value_policy::copy)
      .def("set_provenance", &MultiCobordism::setProvenance,
           py::arg("config_hash"), py::arg("commit"),
           "Deterministic provenance stamped on every checkpoint.")
      .def_property_readonly("provenance_config_hash",
                             &MultiCobordism::provenanceConfigHash)
      .def_property_readonly("provenance_commit", &MultiCobordism::provenanceCommit)
      .def_property_readonly("accepted_move_count", &MultiCobordism::acceptedMoveCount)
      .def_property_readonly("analysis_pass_count", &MultiCobordism::analysisPassCount)
      .def("run_recursive_analysis", &MultiCobordism::runRecursiveAnalysis,
           py::call_guard<py::gil_scoped_release>(),
           "Run ONE post-hoc analysis pass over the CURRENT accepted geometry "
           "in firewall order. Read-only on the geometry.")
      .def_property_readonly("checkpoint_json", &MultiCobordism::checkpointJson,
                             "The versioned checkpoint document of the last "
                             "pass (schema 4). Schema 4 splits the "
                             "bound-supercomponent search records and the "
                             "three-cluster verdicts "
                             "into two blocks; unknown values are null, "
                             "never zero.")
      .def_static("checkpoint_schema_version",
                  &MultiCobordism::checkpointSchemaVersion)
      .def_static("checkpoint_version_of", &MultiCobordism::checkpointVersionOf,
                  py::arg("checkpoint"))
      .def_static("replay_checkpoint", &MultiCobordism::replayCheckpoint,
                  py::arg("checkpoint"),
                  py::call_guard<py::gil_scoped_release>(),
                  "Replay mode: rebuild the raw complex, disable every cache, "
                  "recompute every derived hierarchy and certificate, and "
                  "return the freshly written checkpoint. Raises on an unknown "
                  "schema_version.");
}
