// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the recursive static and
// shifted response reduction. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_recursive_quotient(py::module_ &m) {
  // ----- Recursive static/shifted response reduction -----
  py::enum_<FiberEmbeddingPolicy>(m, "FiberEmbeddingPolicy",
      "The declared labeled-sum Gram treatment: carry G exactly, certify "
      "||G - I|| <= epsilon, or quotient ker G and restate the ranks. Exactly "
      "one per run; an internal direct sum is never assumed.")
      .value("CarryGramExactly", FiberEmbeddingPolicy::CarryGramExactly)
      .value("CertifiedNearIsometry", FiberEmbeddingPolicy::CertifiedNearIsometry)
      .value("QuotientKernel", FiberEmbeddingPolicy::QuotientKernel);

  py::enum_<LevelOrigin>(m, "LevelOrigin",
      "How a recursion level was produced from its parent: Base, the static "
      "lambda = 0 Schur complement, the exact energy-dependent BandPencil at "
      "a declared lambda, or a certified linear AMLS Surrogate.")
      .value("Base", LevelOrigin::Base)
      .value("StaticResponse", LevelOrigin::StaticResponse)
      .value("BandPencil", LevelOrigin::BandPencil)
      .value("Surrogate", LevelOrigin::Surrogate);

  py::enum_<RetainedCoordinateKind>(m, "RetainedCoordinateKind",
      "Why a reduced coordinate was kept: Interface cell, interior Harmonic "
      "zero mode, Resonant shifted kernel mode, or caller-Selected cell. "
      "Retained coordinates are never silently deleted.")
      .value("Interface", RetainedCoordinateKind::Interface)
      .value("Harmonic", RetainedCoordinateKind::Harmonic)
      .value("Resonant", RetainedCoordinateKind::Resonant)
      .value("Selected", RetainedCoordinateKind::Selected);

  py::class_<RecursiveQuotient> recursiveQuotient(m, "RecursiveQuotient",
      R"doc(Recursive static and shifted response reduction over a declared cell
partition. Static: the exact supported response
L_eff = L_BB - L_BI L_II^+ L_IB by sparse/rank-revealing factor solves
(minimization certificate in the positive self-adjoint regime, stationarity
in the Hermitian-indefinite regime, certified block elimination with the
left-kernel compatibility check in the non-normal regime; interior kernels
are RETAINED as explicit stalk coordinates, never regularized away). Band:
the exact Feshbach-Schur pencil F_B(lambda) = L_BB - lambda I -
L_BI (L_II - lambda I)^{-1} L_IB over caller-supplied windows with the exact
determinant factorization det(L - lambda) = det(L_II - lambda) det F_B(lambda),
algebraic (the eigenvalues of L inside a declared disc, counted from the
Schur forms of L and L_II, the interior count reported separately) vs
geometric (dim ker F_B) multiplicities, and a certified Craig-Bampton/AMLS
linear surrogate. The next level is the
abstract labeled sum of retained fibers with embedding J and Gram G = J^dag W J
(one declared policy per run), an operator-valued response network, and a
cellular-sheaf realization emitted ONLY when restriction maps reproduce the
blocks. Nested quotients carry lineage; per-component contributions reuse the
AnalyticCache so a published TouchedStar recomputes only the affected
ancestry. Read-only: nothing here enters the emergence objective.)doc");

  py::class_<RecursiveQuotient::Options>(recursiveQuotient, "Options",
      "Reduction options: certificate tolerance, rank-revealing threshold, "
      "dense crossover, the declared FiberEmbeddingPolicy (+ epsilon), and "
      "caller-selected interior cells to retain.")
      .def(py::init<>())
      .def_readwrite("tolerance", &RecursiveQuotient::Options::tolerance)
      .def_readwrite("rank_tolerance", &RecursiveQuotient::Options::rankTolerance)
      .def_readwrite("dense_crossover", &RecursiveQuotient::Options::denseCrossover)
      .def_readwrite("embedding_policy",
                     &RecursiveQuotient::Options::embeddingPolicy)
      .def_readwrite("near_isometry_epsilon",
                     &RecursiveQuotient::Options::nearIsometryEpsilon)
      .def_readwrite("selected_interior_indices",
                     &RecursiveQuotient::Options::selectedInteriorIndices)
      .def_readwrite("selected_interior_cells",
                     &RecursiveQuotient::Options::selectedInteriorCells);

  py::class_<RecursiveQuotient::RetainedCoordinate>(recursiveQuotient,
      "RetainedCoordinate",
      "One reduced coordinate: kind, owning component, fine index (cells) or "
      "-1 (modes), the fine-space embedding column, and provenance.")
      .def_readonly("kind", &RecursiveQuotient::RetainedCoordinate::kind)
      .def_readonly("component",
                    &RecursiveQuotient::RetainedCoordinate::component)
      .def_readonly("fine_index",
                    &RecursiveQuotient::RetainedCoordinate::fineIndex)
      .def_readonly("embedding",
                    &RecursiveQuotient::RetainedCoordinate::embedding)
      .def_readonly("provenance",
                    &RecursiveQuotient::RetainedCoordinate::provenance);

  py::class_<RecursiveQuotient::InteriorNullspaceRead>(recursiveQuotient,
      "InteriorNullspaceRead",
      "Interior nullspace of one component: exact integer topological basis "
      "(spacetime path), numerical right/left kernels, measured residual.")
      .def_readonly("component",
                    &RecursiveQuotient::InteriorNullspaceRead::component)
      .def_readonly("nullity", &RecursiveQuotient::InteriorNullspaceRead::nullity)
      .def_readonly("integer_nullity",
                    &RecursiveQuotient::InteriorNullspaceRead::integerNullity)
      .def_readonly(
          "integer_nullity_measured",
          &RecursiveQuotient::InteriorNullspaceRead::integerNullityMeasured,
          "Whether the exact integer nullity was computed at all (False on the "
          "matrix path and on integer-kernel overflow, where integerNullity == "
          "0 means 'not measured').")
      .def_readonly(
          "nullity_discrepancy",
          &RecursiveQuotient::InteriorNullspaceRead::nullityDiscrepancy,
          "nullity - integerNullity: the recorded disagreement between the "
          "numerical kernel of the weighted interior block and the exact "
          "integer topological nullity. 0 means they agree; NaN means no "
          "integer nullity was measured (never 0, which would claim an "
          "agreement that was never made).")
      .def_readonly("integer_basis",
                    &RecursiveQuotient::InteriorNullspaceRead::integerBasis)
      .def_readonly("kernel_basis",
                    &RecursiveQuotient::InteriorNullspaceRead::kernelBasis)
      .def_readonly("left_kernel_basis",
                    &RecursiveQuotient::InteriorNullspaceRead::leftKernelBasis)
      .def_readonly("certificate",
                    &RecursiveQuotient::InteriorNullspaceRead::certificate);

  py::class_<RecursiveQuotient::StaticReductionRead>(recursiveQuotient,
      "StaticReductionRead",
      "The exact supported static reduction: kept-cell indices, all reduced "
      "coordinates with provenance, the effective operator (leading kept "
      "block = L_BB - L_BI L_II^+ L_IB), residuals, certificate.")
      .def_readonly("interface_indices",
                    &RecursiveQuotient::StaticReductionRead::interfaceIndices)
      .def_readonly("coordinates",
                    &RecursiveQuotient::StaticReductionRead::coordinates)
      .def_readonly("effective_operator",
                    &RecursiveQuotient::StaticReductionRead::effectiveOperator)
      .def_readonly("solve_residual",
                    &RecursiveQuotient::StaticReductionRead::solveResidual)
      .def_readonly(
          "compatibility_residual",
          &RecursiveQuotient::StaticReductionRead::compatibilityResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::StaticReductionRead::certificate);

  py::class_<RecursiveQuotient::FeshbachRead>(recursiveQuotient, "FeshbachRead",
      "One exact Feshbach-Schur response evaluation over a declared window, "
      "with resonance retention, compatibility check, and the measured "
      "determinant-factorization residual (below the dense crossover).")
      .def_readonly("lam", &RecursiveQuotient::FeshbachRead::lambda)
      .def_readonly("window_lower", &RecursiveQuotient::FeshbachRead::windowLower)
      .def_readonly("window_upper", &RecursiveQuotient::FeshbachRead::windowUpper)
      .def_readonly("response", &RecursiveQuotient::FeshbachRead::response)
      .def_readonly("coordinates", &RecursiveQuotient::FeshbachRead::coordinates)
      .def_readonly("resonant", &RecursiveQuotient::FeshbachRead::resonant)
      .def_readonly("solve_residual",
                    &RecursiveQuotient::FeshbachRead::solveResidual)
      .def_readonly("compatibility_residual",
                    &RecursiveQuotient::FeshbachRead::compatibilityResidual)
      .def_readonly("determinant_residual",
                    &RecursiveQuotient::FeshbachRead::determinantResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::FeshbachRead::certificate);

  py::class_<RecursiveQuotient::MultiplicityRead>(recursiveQuotient,
      "MultiplicityRead",
      "Multiplicity report at a candidate eigenvalue: algebraic = the number "
      "of eigenvalues of L inside the declared counting disc, counted with "
      "multiplicity from the complex Schur form of L; interiorWinding = the "
      "number of eigenvalues of the interior block L_II inside it, reported "
      "separately; responseWinding = their difference, the winding number of "
      "det F_B about the circle; geometric = dim ker F_B(lambda). Algebraic "
      "and geometric agree only in the self-adjoint/semisimple setting.")
      .def_readonly("lam", &RecursiveQuotient::MultiplicityRead::lambda)
      .def_readonly("contour_radius",
                    &RecursiveQuotient::MultiplicityRead::contourRadius,
                    "The radius of the counting disc, the declared selection.")
      .def_readonly("response_winding",
                    &RecursiveQuotient::MultiplicityRead::responseWinding)
      .def_readonly("interior_winding",
                    &RecursiveQuotient::MultiplicityRead::interiorWinding)
      .def_readonly("algebraic", &RecursiveQuotient::MultiplicityRead::algebraic)
      .def_readonly("geometric", &RecursiveQuotient::MultiplicityRead::geometric)
      .def_readonly("semisimple",
                    &RecursiveQuotient::MultiplicityRead::semisimple)
      .def_readonly("isolation_gap",
                    &RecursiveQuotient::MultiplicityRead::isolationGap,
                    "The distance from the circle to the nearest eigenvalue of "
                    "L or of L_II, inside or outside.")
      .def_readonly("decomposition_residual",
                    &RecursiveQuotient::MultiplicityRead::decompositionResidual,
                    "The largest relative backward error of the Schur "
                    "decompositions the counts were read from.")
      .def_readonly("certificate",
                    &RecursiveQuotient::MultiplicityRead::certificate);

  py::class_<RecursiveQuotient::CraigBamptonRead>(recursiveQuotient,
      "CraigBamptonRead",
      "Craig-Bampton/AMLS retained-mode surrogate: the declared window disc "
      "(centre, radius) and retention radius with their real extents, retained "
      "fixed-interface modes per component, basis, reduced (stiffness, mass) "
      "pencil, discarded-mode gap, the claimed spectrum inside the disc, and "
      "fine-space eigenresiduals.")
      .def_readonly("window_centre",
                    &RecursiveQuotient::CraigBamptonRead::windowCentre)
      .def_readonly("window_radius",
                    &RecursiveQuotient::CraigBamptonRead::windowRadius)
      .def_readonly("retention_radius",
                    &RecursiveQuotient::CraigBamptonRead::retentionRadius)
      .def_readonly("window_lower",
                    &RecursiveQuotient::CraigBamptonRead::windowLower)
      .def_readonly("window_upper",
                    &RecursiveQuotient::CraigBamptonRead::windowUpper)
      .def_readonly("mode_cutoff",
                    &RecursiveQuotient::CraigBamptonRead::modeCutoff)
      .def_readonly("retained_modes",
                    &RecursiveQuotient::CraigBamptonRead::retainedModes)
      .def_readonly("basis", &RecursiveQuotient::CraigBamptonRead::basis)
      .def_readonly("reduced_stiffness",
                    &RecursiveQuotient::CraigBamptonRead::reducedStiffness)
      .def_readonly("reduced_mass",
                    &RecursiveQuotient::CraigBamptonRead::reducedMass)
      .def_readonly("discarded_mode_gap",
                    &RecursiveQuotient::CraigBamptonRead::discardedModeGap)
      .def_readonly("window_spectrum",
                    &RecursiveQuotient::CraigBamptonRead::windowSpectrum)
      .def_readonly("window_eigenvalues",
                    &RecursiveQuotient::CraigBamptonRead::windowEigenvalues)
      .def_readonly("eigen_residuals",
                    &RecursiveQuotient::CraigBamptonRead::eigenResiduals)
      .def_readonly("certificate",
                    &RecursiveQuotient::CraigBamptonRead::certificate);

  py::class_<RecursiveQuotient::LabeledFiberSumRead>(recursiveQuotient,
      "LabeledFiberSumRead",
      "The abstract labeled sum of retained fibers: embedding J (= Y) into "
      "the chain space, the explicit left embedding Y~ when the bands carry "
      "their left Riesz frames, the overlap Gram (G = Y~^T Y against the "
      "left embedding, else J^dag W J on an operator level and J^T M J on a "
      "pencil level), the declared policy, gram defect, kernel nullity, and "
      "nominal vs effective ranks. Adjacent fibers may overlap on shared "
      "interface cells; a direct sum is never asserted.")
      .def_readonly("summand_components",
                    &RecursiveQuotient::LabeledFiberSumRead::summandComponents)
      .def_readonly("summand_ranks",
                    &RecursiveQuotient::LabeledFiberSumRead::summandRanks)
      .def_readonly("embedding",
                    &RecursiveQuotient::LabeledFiberSumRead::embedding)
      .def_readonly("left_embedding",
                    &RecursiveQuotient::LabeledFiberSumRead::leftEmbedding,
                    "The explicit left embedding Y~ (flat, fineDim x "
                    "totalRank), fixed before the overlap test; empty when "
                    "the level's metric dual is the left embedding.")
      .def_readonly("gram", &RecursiveQuotient::LabeledFiberSumRead::gram)
      .def_readonly("policy", &RecursiveQuotient::LabeledFiberSumRead::policy)
      .def_readonly("gram_defect",
                    &RecursiveQuotient::LabeledFiberSumRead::gramDefect)
      .def_readonly("quotient_nullity",
                    &RecursiveQuotient::LabeledFiberSumRead::quotientNullity)
      .def_readonly("nominal_rank",
                    &RecursiveQuotient::LabeledFiberSumRead::nominalRank)
      .def_readonly("effective_rank",
                    &RecursiveQuotient::LabeledFiberSumRead::effectiveRank)
      .def_readonly("quotient_basis",
                    &RecursiveQuotient::LabeledFiberSumRead::quotientBasis)
      .def_readonly("left_quotient_basis",
                    &RecursiveQuotient::LabeledFiberSumRead::leftQuotientBasis,
                    "The left partner L_q of quotientBasis R_q: the quotient "
                    "of a one-particle matrix is L_q^T X R_q.")
      .def_readonly(
          "from_certified_bands",
          &RecursiveQuotient::LabeledFiberSumRead::fromCertifiedBands)
      .def_readonly(
          "summand_certificates",
          &RecursiveQuotient::LabeledFiberSumRead::summandCertificates)
      .def_readonly(
          "worst_isolation_gap",
          &RecursiveQuotient::LabeledFiberSumRead::worstIsolationGap)
      .def_readonly("all_bands_accepted",
                    &RecursiveQuotient::LabeledFiberSumRead::allBandsAccepted)
      .def_readonly("certificate",
                    &RecursiveQuotient::LabeledFiberSumRead::certificate);

  py::class_<RecursiveQuotient::CertifiedBand>(recursiveQuotient,
      "CertifiedBand",
      "One certified isolated band handed to certifiedFiberSum as the summand "
      "E_v of the master recursion: its right frame over this level's fine "
      "coordinates, optionally its local left Riesz frame (Phi~^T Phi = I, "
      "the left embedding of the overlap certificate), its rank, its "
      "isolation gaps and frequency window, whether its producing "
      "configuration accepted it, and its certificate.")
      .def(py::init<>())
      .def_readwrite("component", &RecursiveQuotient::CertifiedBand::component)
      .def_readwrite("frame", &RecursiveQuotient::CertifiedBand::frame)
      .def_readwrite("left_frame", &RecursiveQuotient::CertifiedBand::leftFrame,
                     "The band's local left Riesz frame Phi~ (flat, "
                     "dimension x rank), paired with frame by the plain "
                     "transpose; empty: the level's metric dual stands in.")
      .def_readwrite("rank", &RecursiveQuotient::CertifiedBand::rank)
      .def_readwrite("lower_gap", &RecursiveQuotient::CertifiedBand::lowerGap)
      .def_readwrite("upper_gap", &RecursiveQuotient::CertifiedBand::upperGap)
      .def_readwrite("frequency_lower",
                     &RecursiveQuotient::CertifiedBand::frequencyLower)
      .def_readwrite("frequency_upper",
                     &RecursiveQuotient::CertifiedBand::frequencyUpper)
      .def_readwrite("window_centre",
                     &RecursiveQuotient::CertifiedBand::windowCentre,
                     "Centre of the band's window disc in the complex plane (NaN when the "
                     "band was declared by its real extent alone).")
      .def_readwrite("window_radius",
                     &RecursiveQuotient::CertifiedBand::windowRadius,
                     "Radius of the band's window disc (NaN when declared by real extent alone).")
      .def_readwrite("accepted", &RecursiveQuotient::CertifiedBand::accepted)
      .def_readwrite("certificate",
                     &RecursiveQuotient::CertifiedBand::certificate);

  py::class_<RecursiveQuotient::CertifiedFiberSummand>(recursiveQuotient,
      "CertifiedFiberSummand",
      "The certificate data of one summand of a certified labeled sum, "
      "carried verbatim from its producing band.")
      .def_readonly("component",
                    &RecursiveQuotient::CertifiedFiberSummand::component)
      .def_readonly("rank", &RecursiveQuotient::CertifiedFiberSummand::rank)
      .def_readonly("lower_gap",
                    &RecursiveQuotient::CertifiedFiberSummand::lowerGap)
      .def_readonly("upper_gap",
                    &RecursiveQuotient::CertifiedFiberSummand::upperGap)
      .def_readonly("frequency_lower",
                    &RecursiveQuotient::CertifiedFiberSummand::frequencyLower)
      .def_readonly("frequency_upper",
                    &RecursiveQuotient::CertifiedFiberSummand::frequencyUpper)
      .def_readonly("window_centre",
                    &RecursiveQuotient::CertifiedFiberSummand::windowCentre)
      .def_readonly("window_radius",
                    &RecursiveQuotient::CertifiedFiberSummand::windowRadius)
      .def_readonly("accepted",
                    &RecursiveQuotient::CertifiedFiberSummand::accepted)
      .def_readonly("certificate",
                    &RecursiveQuotient::CertifiedFiberSummand::certificate);

  py::class_<RecursiveQuotient::LevelProvenanceRead>(recursiveQuotient,
      "LevelProvenanceRead",
      "How a level was produced from its parent: the origin, the declared "
      "lambda and window, the producing step's solve/compatibility residuals, "
      "the surrogate residual and discarded-mode gap, the resonance flag, and "
      "the producing certificate carried verbatim. Unmeasured fields are NaN, "
      "never zero.")
      .def_readonly("origin", &RecursiveQuotient::LevelProvenanceRead::origin)
      .def_readonly("lambda_", &RecursiveQuotient::LevelProvenanceRead::lambda)
      .def_readonly("window_lower",
                    &RecursiveQuotient::LevelProvenanceRead::windowLower)
      .def_readonly("window_upper",
                    &RecursiveQuotient::LevelProvenanceRead::windowUpper)
      .def_readonly("solve_residual",
                    &RecursiveQuotient::LevelProvenanceRead::solveResidual)
      .def_readonly(
          "compatibility_residual",
          &RecursiveQuotient::LevelProvenanceRead::compatibilityResidual)
      .def_readonly("surrogate_residual",
                    &RecursiveQuotient::LevelProvenanceRead::surrogateResidual)
      .def_readonly("discarded_mode_gap",
                    &RecursiveQuotient::LevelProvenanceRead::discardedModeGap)
      .def_readonly("resonant",
                    &RecursiveQuotient::LevelProvenanceRead::resonant)
      .def_readonly("certificate",
                    &RecursiveQuotient::LevelProvenanceRead::certificate);

  py::class_<RecursiveQuotient::FockStageRead>(recursiveQuotient,
      "FockStageRead",
      "The Fock stage over a labeled sum: the one-particle compression in "
      "the pairing its Gram was built in (h = Y~^T L Y, J^T A~ J on a pencil "
      "level, J^dag W L J on an operator level; named by `pairing`), the "
      "Gram on the same basis, the pencil spectrum, "
      "2^M as fockDimension, and the exact free many-body spectrum as "
      "occupation subset sums. The 2^M space is never materialized and the "
      "spectrum refuses past the declared term budget.")
      .def_readonly("modes", &RecursiveQuotient::FockStageRead::modes)
      .def_readonly("policy", &RecursiveQuotient::FockStageRead::policy)
      .def_readonly("gram_defect",
                    &RecursiveQuotient::FockStageRead::gramDefect)
      .def_readonly("one_particle",
                    &RecursiveQuotient::FockStageRead::oneParticle)
      .def_readonly("gram", &RecursiveQuotient::FockStageRead::gram)
      .def_readonly("pairing", &RecursiveQuotient::FockStageRead::pairing,
                    "The one pairing h and G were compressed in: "
                    "'metric-hermitian', 'metric-transpose' or "
                    "'left-embedding'.")
      .def_readonly("one_particle_spectrum",
                    &RecursiveQuotient::FockStageRead::oneParticleSpectrum)
      .def_readonly("fock_dimension",
                    &RecursiveQuotient::FockStageRead::fockDimension)
      .def_readonly("spectrum_materialized",
                    &RecursiveQuotient::FockStageRead::spectrumMaterialized)
      .def_readonly("fock_spectrum",
                    &RecursiveQuotient::FockStageRead::fockSpectrum)
      .def_readonly("certificate",
                    &RecursiveQuotient::FockStageRead::certificate);

  py::class_<RecursiveQuotient::ResponseEdge>(recursiveQuotient, "ResponseEdge",
      "One operator-valued link: from/to component and the effective block.")
      .def_readonly("from_component", &RecursiveQuotient::ResponseEdge::from)
      .def_readonly("to_component", &RecursiveQuotient::ResponseEdge::to)
      .def_readonly("block", &RecursiveQuotient::ResponseEdge::block);

  py::class_<RecursiveQuotient::ResponseNetworkRead>(recursiveQuotient,
      "ResponseNetworkRead",
      "The next-level operator-valued response network: per-component stalks "
      "(shared interface cells appear in every claiming stalk), vertex and "
      "edge blocks of the reduced operator, and the coverage residual.")
      .def_readonly("stalk_dimensions",
                    &RecursiveQuotient::ResponseNetworkRead::stalkDimensions)
      .def_readonly("stalk_coordinates",
                    &RecursiveQuotient::ResponseNetworkRead::stalkCoordinates)
      .def_readonly("vertex_blocks",
                    &RecursiveQuotient::ResponseNetworkRead::vertexBlocks)
      .def_readonly("edges", &RecursiveQuotient::ResponseNetworkRead::edges)
      .def_readonly("coverage_residual",
                    &RecursiveQuotient::ResponseNetworkRead::coverageResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::ResponseNetworkRead::certificate);

  py::class_<RecursiveQuotient::ResolvedPartitionRead>(recursiveQuotient,
      "ResolvedPartitionRead",
      "The window form of persistentPartition, with the persistence of each "
      "carried component reported beside the partition: how many resolutions "
      "of the window it stood at, the resolution its support was read at, and "
      "the weakest adjacent-resolution support overlap of its track.")
      .def_readonly("components",
                    &RecursiveQuotient::ResolvedPartitionRead::components)
      .def_readonly("resolutions",
                    &RecursiveQuotient::ResolvedPartitionRead::resolutions)
      .def_readonly("selected_resolution",
                    &RecursiveQuotient::ResolvedPartitionRead::selectedResolution)
      .def_readonly(
          "component_persistence",
          &RecursiveQuotient::ResolvedPartitionRead::componentPersistence)
      .def_readonly("worst_overlap",
                    &RecursiveQuotient::ResolvedPartitionRead::worstOverlap);

  py::class_<RecursiveQuotient::SheafRealizationRead>(recursiveQuotient,
      "SheafRealizationRead",
      "Cellular-sheaf/simplicial realization attempt: emitted ONLY when the "
      "restriction maps reproduce the network blocks (certified); otherwise "
      "the general response network is retained and nothing is invented.")
      .def_readonly("emitted",
                    &RecursiveQuotient::SheafRealizationRead::emitted)
      .def_readonly("simplicial",
                    &RecursiveQuotient::SheafRealizationRead::simplicial)
      .def_readonly(
          "edge_stalk_dimensions",
          &RecursiveQuotient::SheafRealizationRead::edgeStalkDimensions)
      .def_readonly("restriction_maps",
                    &RecursiveQuotient::SheafRealizationRead::restrictionMaps)
      .def_readonly(
          "reconstruction_residual",
          &RecursiveQuotient::SheafRealizationRead::reconstructionResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::SheafRealizationRead::certificate);

  recursiveQuotient
      .def_static("over_matrix", &RecursiveQuotient::overMatrix, py::arg("op"),
                  py::arg("dim"), py::arg("weights"), py::arg("components"),
                  py::arg("options") = RecursiveQuotient::Options(),
                  "Build over an explicit operator (flat row-major) with a "
                  "diagonal chain metric (empty = identity) and 0-based, "
                  "possibly overlapping component index sets covering every "
                  "index. Fixtures and next-level recursion.")
      .def_static("over_pencil", &RecursiveQuotient::overPencil, py::arg("A"),
                  py::arg("M"), py::arg("dim"), py::arg("components"),
                  py::arg("options") = RecursiveQuotient::Options(),
                  "Build over a symmetric PENCIL (A~, M) on geometric images (flat "
                  "row-major both): every shifted elimination is taken on "
                  "P(lambda) = A~ - lambda M, and a child level carries the Gram "
                  "T^T M T of its constraint modes.")
      .def("is_pencil", &RecursiveQuotient::isPencil, "Whether this level is a pencil level.")
      .def("pencil_metric", &RecursiveQuotient::pencilMetric,
           "The pencil's metric M (base) or carried Gram (child), flat row-major; empty otherwise.")
      .def_static(
          "over_cells",
          [](std::shared_ptr<Spacetime> st, int degree,
             const std::vector<std::vector<std::vector<std::uint64_t>>> &cells,
             const RecursiveQuotient::Options &options, AnalyticCache *cache,
             std::optional<HodgeLaplacian::MetricSource> source) {
            std::shared_ptr<AnalyticCache> held;
            if (cache) held = std::shared_ptr<AnalyticCache>(cache, [](AnalyticCache *) {});
            return RecursiveQuotient::overCells(
                std::move(st), degree, cells, options, std::move(held),
                source.value_or(HodgeLaplacian::defaultMetricSource()));
          },
          py::arg("spacetime"), py::arg("degree"), py::arg("component_cells"),
          py::arg("options") = RecursiveQuotient::Options(),
          py::arg("cache") = nullptr, py::arg("metric_source") = py::none(),
          // the non-owning cache pointer must outlive the quotient
          py::keep_alive<0, 5>(),
          "Build over the spacetime's Hodge operator at `degree` with "
          "components as explicit k-cell sets (vertex-id tuples, matched by "
          "vertex SET). An AnalyticCache bound to the same spacetime enables "
          "per-component reuse across accepted moves. The operator and its "
          "metric come from one metric_source (None: the process-wide "
          "HodgeLaplacian.default_metric_source() at call time): WhitneyPencil "
          "builds a pencil level over (A~_k^U, M_k^U) of HodgeLaplacian.pencil; "
          "DiagonalWeights an operator level over laplacian(k) with weights(k).")
      .def_static(
          "over_vertex_supports",
          [](std::shared_ptr<Spacetime> st, int degree,
             const std::vector<std::vector<std::uint64_t>> &supports,
             const RecursiveQuotient::Options &options, AnalyticCache *cache,
             std::optional<HodgeLaplacian::MetricSource> source) {
            std::shared_ptr<AnalyticCache> held;
            if (cache) held = std::shared_ptr<AnalyticCache>(cache, [](AnalyticCache *) {});
            return RecursiveQuotient::overVertexSupports(
                std::move(st), degree, supports, options, std::move(held),
                source.value_or(HodgeLaplacian::defaultMetricSource()));
          },
          py::arg("spacetime"), py::arg("degree"), py::arg("vertex_supports"),
          py::arg("options") = RecursiveQuotient::Options(),
          py::arg("cache") = nullptr, py::arg("metric_source") = py::none(),
          py::keep_alive<0, 5>(),
          "Build with components as vertex supports (the PersistentModularity "
          "convention): a k-cell belongs to a component when ALL its vertices "
          "lie in the support; unclaimed cells form one residual component. "
          "metric_source as in overCells.")
      .def("metric_source", &RecursiveQuotient::metricSource,
          "The metric source a spacetime-backed level was built on; None on the "
          "matrix and pencil paths and on child levels.")
      .def_property_readonly("dimension", &RecursiveQuotient::dimension)
      .def_property_readonly("component_count",
                             &RecursiveQuotient::componentCount)
      .def_property_readonly("degree", &RecursiveQuotient::degree)
      .def_property_readonly("level", &RecursiveQuotient::level)
      .def_property_readonly("regime", &RecursiveQuotient::regime)
      .def_property_readonly("interface_indices",
                             &RecursiveQuotient::interfaceIndices)
      .def("interior_indices", &RecursiveQuotient::interiorIndices,
           py::arg("component"))
      .def_property_readonly("coordinate_provenance",
                             &RecursiveQuotient::coordinateProvenance)
      .def("interior_nullspace", &RecursiveQuotient::interiorNullspace,
           py::arg("component"),
           "Exact integer topological zero modes (spacetime path) + the "
           "numerical right/left kernels of the interior block.")
      .def("static_reduction", &RecursiveQuotient::staticReduction,
           py::return_value_policy::reference_internal,
           "The exact supported static reduction (memoized; per-component "
           "contributions served from the bound AnalyticCache when fresh).")
      .def("static_probe_certificate", &RecursiveQuotient::staticProbeCertificate,
           py::arg("probe"),
           "Regime-appropriate static certificate on one kept-cell probe: "
           "minimum (positive), stationarity (Hermitian-indefinite), or "
           "certified block elimination + left-kernel compatibility "
           "(non-normal).")
      .def("verify_static", &RecursiveQuotient::verifyStatic,
           "Worst static probe certificate over every kept basis vector and "
           "the all-ones probe.")
      .def("feshbach", &RecursiveQuotient::feshbach, py::arg("lam"),
           py::arg("window_lower"), py::arg("window_upper"),
           "Exact Feshbach-Schur response F_B(lambda) over a caller-supplied "
           "window, with resonance retention + compatibility checks and the "
           "determinant-factorization residual below the dense crossover.")
      .def("multiplicity", &RecursiveQuotient::multiplicity, py::arg("lam"),
           py::arg("radius"),
           "The algebraic multiplicity inside the disc of the given radius "
           "about lam: the eigenvalues of L in the disc, counted with "
           "multiplicity from the Schur form of L, with the interior block's "
           "count reported separately; the geometric multiplicity is "
           "dim ker F_B(lam). Whether an eigenvalue is inside is decided at the "
           "rank tolerance, and a circle through an eigenvalue is refused by "
           "name.")
      .def("craig_bampton",
           py::overload_cast<double, double, double, double>(
               &RecursiveQuotient::craigBampton, py::const_),
           py::arg("window_lower"), py::arg("window_upper"),
           py::arg("mode_cutoff"), py::arg("residual_tolerance") = -1.0,
           "Craig-Bampton retained-mode basis + reduced (K, M) pencil declared by a "
           "real window: [a, b] is the disc with centre (a+b)/2 and radius (b-a)/2, "
           "and mode_cutoff is the retention disc's real upper edge, so the retention "
           "radius is mode_cutoff - (a+b)/2. Certified approximation: the certificate "
           "holds against the caller-declared residual_tolerance; negative selects the "
           "strict Options.tolerance. It runs in every regime: the adjoint pairing "
           "against the positive diagonal chain metric in the two Hermitian regimes, "
           "the transpose pairing against the level's own metric in the non-normal and "
           "complex-symmetric-pencil ones. An indefinite chain metric is refused in a "
           "Hermitian regime, a singular interior or reduced metric in a bilinear one.")
      .def("craig_bampton",
           py::overload_cast<std::complex<double>, double, double, double>(
               &RecursiveQuotient::craigBampton, py::const_),
           py::arg("window_centre"), py::arg("window_radius"),
           py::arg("retention_radius"), py::arg("residual_tolerance") = -1.0,
           "The same surrogate over a declared window disc |theta - centre| <= radius "
           "in the complex spectral plane: a fixed-interface mode is retained when its "
           "eigenvalue is within retention_radius of the centre and a reduced "
           "eigenvalue is claimed when it lies in the disc, by complex distance in "
           "every regime. retention_radius must cover window_radius.")
      .def("labeled_fiber_sum", &RecursiveQuotient::labeledFiberSum,
           "The abstract labeled sum of retained fibers with embedding J and "
           "Gram G = J^dag W J under the run's declared policy.")
      .def_static("compose_near_isometry_budget",
                  &RecursiveQuotient::composeNearIsometryBudget,
                  py::arg("epsilon_a"), py::arg("epsilon_b"),
                  "Composable amplitude budget of the CertifiedNearIsometry "
                  "policy: eps_AB <= eps_A + eps_B + eps_A * eps_B.")
      .def("response_network", &RecursiveQuotient::responseNetwork,
           "The next-level operator-valued response network (stalks + "
           "effective blocks).")
      .def("sheaf_realization", &RecursiveQuotient::sheafRealization,
           "Cellular-sheaf/simplicial realization, emitted only when "
           "certified; otherwise the general network is retained.")
      .def("next_level",
           py::overload_cast<const std::vector<std::vector<int>> &,
                             const RecursiveQuotient::Options &>(
               &RecursiveQuotient::nextLevel, py::const_),
           py::arg("components"), py::arg("options"),
           "Reduce again over this level's reduced operator; the child "
           "carries provenance-prefixed lineage and level + 1.")
      .def("next_level",
           py::overload_cast<const std::vector<std::vector<int>> &>(
               &RecursiveQuotient::nextLevel, py::const_),
           py::arg("components"))
      .def("certified_fiber_sum", &RecursiveQuotient::certifiedFiberSum,
           py::arg("bands"),
           "The labeled sum over CERTIFIED ISOLATED BANDS (the master "
           "recursion's E_v), carrying each band's isolation gap and "
           "certificate onto its summand. An uncertified band is summed and "
           "reported, never dropped, and makes the sum's certificate fail to "
           "hold.")
      .def("fock_stage", &RecursiveQuotient::fockStage, py::arg("sum"),
           py::arg("max_terms") = std::size_t{1} << 22,
           "The Fock stage over a labeled sum: the one-particle compression, "
           "the pencil spectrum, and the exact free many-body spectrum as "
           "occupation subset sums (refusing past max_terms).")
      .def_static("persistent_partition",
                  py::overload_cast<const std::vector<std::complex<double>> &,
                                    int, double, int, std::uint64_t>(
                      &RecursiveQuotient::persistentPartition),
                  py::arg("op"),
                  py::arg("dim"), py::arg("gamma") = 1.0,
                  py::arg("restarts") = 4, py::arg("base_seed") = 0,
                  "P = PersistentPartition(R): partition a response "
                  "network's coordinates by persistent modularity over its "
                  "symmetrized off-diagonal magnitude graph. Covers every "
                  "index exactly once; isolated coordinates come back as "
                  "singletons.")
      .def_static("persistent_partition",
                  py::overload_cast<const std::vector<std::complex<double>> &,
                                    int, const std::vector<double> &, int,
                                    std::uint64_t>(
                      &RecursiveQuotient::persistentPartition),
                  py::arg("op"), py::arg("dim"), py::arg("gammas"),
                  py::arg("restarts") = 4, py::arg("base_seed") = 0,
                  "persistentPartition over a declared window of "
                  "resolutions: the resolution parameter is a free knob of "
                  "the proposer, so only the components whose persistence "
                  "track covers the whole window are kept, each proposing "
                  "its support at the first resolution of the window. Every "
                  "coordinate no such component claimed comes back as a "
                  "singleton.")
      .def_static("persistent_partition_over_resolutions",
                  &RecursiveQuotient::persistentPartitionOverResolutions,
                  py::arg("op"), py::arg("dim"), py::arg("gammas"),
                  py::arg("restarts") = 4, py::arg("base_seed") = 0,
                  py::arg("overlap_threshold") = 0.5,
                  "The window form of persistentPartition, with the "
                  "persistence of every carried component reported beside the "
                  "partition: the same scan, the same rule and the same "
                  "result. A recursion that records why a level was "
                  "partitioned as it was reads this form instead.")
      .def("child_persistent_partition",
           py::overload_cast<double, int, std::uint64_t>(
               &RecursiveQuotient::childPersistentPartition, py::const_),
           py::arg("gamma") = 1.0, py::arg("restarts") = 4,
           py::arg("base_seed") = 0,
           "persistentPartition of this level's reduced operator — the "
           "partition P_l to hand straight to nextLevel.")
      .def("child_persistent_partition",
           py::overload_cast<const std::vector<double> &, int, std::uint64_t>(
               &RecursiveQuotient::childPersistentPartition, py::const_),
           py::arg("gammas"), py::arg("restarts") = 4,
           py::arg("base_seed") = 0,
           "childPersistentPartition over a declared window of resolutions: "
           "the components of this level's reduced operator that persist "
           "across the whole window.")
      .def("next_level_at_lambda",
           py::overload_cast<const std::vector<std::vector<int>> &,
                             std::complex<double>, double, double,
                             const RecursiveQuotient::Options &>(
               &RecursiveQuotient::nextLevelAtLambda, py::const_),
           py::arg("components"), py::arg("lambda_"), py::arg("window_lower"),
           py::arg("window_upper"), py::arg("options"),
           "Reduce again ON THE PENCIL: the child's operator is the exact "
           "energy-dependent response F_B(lambda), and it carries the window, "
           "residuals, resonance flag and producing certificate.")
      .def("next_level_at_lambda",
           py::overload_cast<const std::vector<std::vector<int>> &,
                             std::complex<double>, double, double>(
               &RecursiveQuotient::nextLevelAtLambda, py::const_),
           py::arg("components"), py::arg("lambda_"), py::arg("window_lower"),
           py::arg("window_upper"))
      .def("next_level_from_surrogate",
           py::overload_cast<const std::vector<std::vector<int>> &, double,
                             double, double, double,
                             const RecursiveQuotient::Options &>(
               &RecursiveQuotient::nextLevelFromSurrogate, py::const_),
           py::arg("components"), py::arg("window_lower"),
           py::arg("window_upper"), py::arg("mode_cutoff"),
           py::arg("residual_tolerance"), py::arg("options"),
           "Reduce again through the certified linear AMLS surrogate, on the "
           "M-orthonormalized basis (a spectrum-preserving congruence). The "
           "child carries the surrogate's certified-approximation "
           "certificate.")
      .def("next_level_from_surrogate",
           py::overload_cast<const std::vector<std::vector<int>> &, double,
                             double, double, double>(
               &RecursiveQuotient::nextLevelFromSurrogate, py::const_),
           py::arg("components"), py::arg("window_lower"),
           py::arg("window_upper"), py::arg("mode_cutoff"),
           py::arg("residual_tolerance") = -1.0)
      .def_property_readonly("level_provenance",
                             &RecursiveQuotient::levelProvenance,
                             "How this level was produced from its parent.")
      .def_property_readonly("cell_vertices", &RecursiveQuotient::cellVertices,
                             "The k-cell vertex tuples of this level's fine "
                             "coordinates, in coordinate order (spacetime "
                             "paths only; empty on the matrix path and on "
                             "child levels). Match a band's cells against "
                             "these by vertex SET to build a CertifiedBand.")
      .def("invalidate", &RecursiveQuotient::invalidate,
           "Drop memoized results and re-read the operator values for the "
           "same cell complex (call after an accepted metric move).")
      .def_property_readonly("options", &RecursiveQuotient::options);
}
