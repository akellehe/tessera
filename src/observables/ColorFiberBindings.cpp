// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: oriented triangles, anchors
// and the colour fiber. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_color_fiber(py::module_ &m) {
  // ==========================================================================
  // ColorFiber / ColorAnchor: the exact three-edge SU(3) color kernel
  // and the calibrated weighted oriented-triangle anchor.  Pure reads over
  // caller-supplied data; nothing enters the emergence objective.
  // ==========================================================================
  py::class_<OrientedTriangle>(m, "OrientedTriangle",
      R"doc(One oriented 2-simplex descriptor for the anchoring kernel: the
three boundary edge indices in the cyclic order induced by the triangle's
orientation (rows of the caller's frame), with their incidence signs (+1 /
-1).  det A_tau is invariant under cyclic rotation of (edges, signs) and
negates under an odd permutation (the opposite orientation).)doc")
      .def(py::init([](std::array<Eigen::Index, 3> edges,
                       std::array<int, 3> signs) {
             return OrientedTriangle{edges, signs};
           }),
           py::arg("edges"),
           py::arg("signs") = std::array<int, 3>{+1, +1, +1})
      .def_readwrite("edges", &OrientedTriangle::edges)
      .def_readwrite("signs", &OrientedTriangle::signs);

  py::class_<AnchorProfile>(m, "AnchorProfile",
      R"doc(The full anchor profile, not only the score: the calibrated atlas
score a^2 = sum_tau w_tau |det A_tau|^2, the per-triangle terms, the maximal
term, the participation ratio of the term distribution, the determinant
phases with their circular coherence and dispersion on overlapping oriented
triangles (NaN when no determinant is nonzero; unknown is never encoded as
zero), the per-triangle Krein signatures (n+, n0, n-) of the restricted
weight blocks, the frame-normalization residual, the checked calibration
margin, and the pre-declared convex weighting that produced the score.)doc")
      .def(py::init<>())
      .def_readonly("score", &AnchorProfile::score)
      .def_readonly("terms", &AnchorProfile::terms)
      .def_readonly("max_term", &AnchorProfile::maxTerm)
      .def_readonly("max_term_index", &AnchorProfile::maxTermIndex)
      .def_readonly("participation_ratio", &AnchorProfile::participationRatio)
      .def_readonly("det_phases", &AnchorProfile::detPhases)
      .def_readonly("phase_coherence", &AnchorProfile::phaseCoherence,
                    "Determinant-phase coherence on overlapping triangles "
                    "(NaN on a disjoint atlas: no overlap content).")
      .def_readonly("phase_dispersion", &AnchorProfile::phaseDispersion)
      .def_readonly("overlapping_triangles",
                    &AnchorProfile::overlappingTriangles,
                    "Declared triangles that share a boundary edge with "
                    "another declared triangle -- the coherence support.")
      .def_readonly("overlap_relation", &AnchorProfile::overlapRelation,
                    "The sharing relation used: 'shared-edge'.")
      .def_readonly("krein_signatures", &AnchorProfile::kreinSignatures)
      .def_readonly("positive_regime", &AnchorProfile::positiveRegime)
      .def_readonly("frame_gram_residual", &AnchorProfile::frameGramResidual)
      .def_readonly("calibration_margin", &AnchorProfile::calibrationMargin)
      .def_readonly("weighting_id", &AnchorProfile::weightingId)
      .def_readonly("weights", &AnchorProfile::weights)
      .def_readonly("certificate", &AnchorProfile::certificate,
          "The tessera.cobordism.Certificate grading the calibrated "
          "score: StructureExact on the diagonal (decoupled) weight path, "
          "CertifiedNumerical on the general Hermitian-matrix path; regime "
          "PositiveSemidefinite / HermitianIndefinite per the Krein read; "
          "residual = max(frame_gram_residual, max(0, calibration_margin)) "
          "against the evaluate gram tolerance.");

  py::class_<AnchorGate>(m, "AnchorGate",
      "The triangle-anchor gate required before any colour-specific "
      "kernel runs. A default-constructed gate is closed, so a caller "
      "that supplies nothing is refused rather than admitted; the "
      "only way to open one is ColorAnchor.gate_for, which applies "
      "the same acceptance predicate the quark verdict uses.")
      .def(py::init<>())
      .def_readonly("accepted", &AnchorGate::accepted)
      .def_readonly("score", &AnchorGate::score)
      .def_readonly("phase_coherence", &AnchorGate::phaseCoherence)
      .def_readonly("weighting_id", &AnchorGate::weightingId)
      .def_readonly("refusal_reason", &AnchorGate::refusalReason,
          "Why the gate is closed ('' when accepted).");

  py::class_<ColorFiber::SectorWeights>(m, "SectorWeights",
      "Occupation-sector weights ||P_N psi||^2 of an 8-dimensional Fock "
      "vector over the three edge modes: vacuum (N=0), quark / fundamental "
      "triplet (N=1), anti-triplet / diquark (N=2), top-wedge color singlet "
      "(N=3).  Sector reads only; never a particle classification.")
      .def_readonly("vacuum", &ColorFiber::SectorWeights::vacuum)
      .def_readonly("quark", &ColorFiber::SectorWeights::quark)
      .def_readonly("anti_triplet", &ColorFiber::SectorWeights::antiTriplet)
      .def_readonly("singlet", &ColorFiber::SectorWeights::singlet);

  py::class_<ColorFiber::OctetRead>(m, "OctetRead",
      "Frobenius split of a 3x3 bilinear under 3 (x) 3bar = 1 (+) 8: "
      "octet = ||M - (tr M / 3) I||_F^2, singlet = |tr M|^2 / 3; their sum "
      "is ||M||_F^2 exactly.")
      .def_readonly("octet", &ColorFiber::OctetRead::octet)
      .def_readonly("singlet", &ColorFiber::OctetRead::singlet);

  py::class_<ColorFiber>(m, "ColorFiber",
      R"doc(The exact three-edge SU(3) color kernel: the constant
color-sector algebra of three oriented edge modes,
Lambda* C^3 = 1 (+) 3 (+) 3bar (+) 1, layered over the exterior-algebra
primitives; sector projectors and canonical anticommutation relation (CAR)
matrices are delegated to tessera.quantum.ExteriorAlgebra.

All members are static; Fock operators are dense 8x8 matrices on the
occupation basis n(b) = sum_i b_i 2^i, and the one-occupation (triplet)
sector is spanned by Fock indices (1, 2, 4).  Exact identities (tested to
double round-off): F3^dag F3 = I, |det F3| = 1; lambda_a Hermitian,
traceless, Tr(lambda_a lambda_b) = 2 delta_ab; [E_ij, E_kl] = delta_jk E_il
- delta_il E_kj on both representations; det(gC) = det(C) for g in SU(3);
||v1 ^ v2 ^ v3||^2 = det[<v_i, v_j>].  Pure constants and reads -- no
solver call, no mutation, nothing enters the emergence objective.)doc")
      .def_static("sector_projector", &ColorFiber::sectorProjector,
                  py::arg("occupation"),
                  "The 8x8 projector onto total occupation N (0..3; zero "
                  "matrix above 3).  Delegates to "
                  "quantum.ExteriorAlgebra.sector_projector on three modes.")
      .def_static("vacuum_projector", &ColorFiber::vacuumProjector,
                  "Lambda^0: the even vacuum singlet (N=0).")
      .def_static("triplet_projector", &ColorFiber::tripletProjector,
                  "Lambda^1: the odd fundamental color triplet 3 (N=1).")
      .def_static("anti_triplet_projector", &ColorFiber::antiTripletProjector,
                  "Lambda^2: the even antisymmetric anti-triplet 3bar (N=2).")
      .def_static("singlet_projector", &ColorFiber::singletProjector,
                  "Lambda^3: the odd top-wedge color singlet (N=3).")
      .def_static("creation_matrix", &ColorFiber::creationMatrix,
                  py::arg("mode"), "The 8x8 creation matrix a_i^dag.")
      .def_static("annihilation_matrix", &ColorFiber::annihilationMatrix,
                  py::arg("mode"), "The 8x8 annihilation matrix a_i.")
      .def_static("hopping_matrix", &ColorFiber::hoppingMatrix,
                  py::arg("i"), py::arg("j"),
                  "The 8x8 bilinear E_ij = a_i^dag a_j (exact gl(3) "
                  "commutation relations on the whole Fock space).")
      .def_static("triplet_basis_indices", &ColorFiber::tripletBasisIndices,
                  "The Fock indices (1, 2, 4) identifying the N=1 sector "
                  "with C^3.")
      .def_static("restrict_to_triplet", &ColorFiber::restrictToTriplet,
                  py::arg("op"),
                  "Restrict an 8x8 Fock operator to the one-occupation "
                  "sector as a 3x3 matrix; restrict_to_triplet(d_gamma(M)) = M "
                  "exactly.")
      .def_static("matrix_unit", &ColorFiber::matrixUnit,
                  py::arg("i"), py::arg("j"),
                  "The 3x3 matrix unit E_ij on the one-occupation sector.")
      .def_static("d_gamma", &ColorFiber::dGamma, py::arg("m"),
                  "Second quantization d_gamma(M) = sum_ij M_ij a_i^dag a_j "
                  "of a 3x3 one-particle matrix (8x8).")
      .def_static("gell_mann", &ColorFiber::gellMann, py::arg("a"),
                  "lambda_a for a in 1..8, assembled from the matrix units "
                  "(lambda_3 = E11-E22, lambda_8 = (E11+E22-2E33)/sqrt(3)).")
      .def_static("adjoint_octet_projector", &ColorFiber::adjointOctetProjector,
                  "The 9x9 orthogonal projector onto the traceless "
                  "(adjoint-octet) part of a 3x3 bilinear, acting on "
                  "column-major vec(M).")
      .def_static("traceless_part", &ColorFiber::tracelessPart, py::arg("m"),
                  "M - (tr M / 3) I: the octet component of a bilinear.")
      .def_static("adjoint_singlet_projector",
                  &ColorFiber::adjointSingletProjector,
                  "The 9x9 projector vec(I)vec(I)^dag/3 onto the trace "
                  "(singlet) part -- implemented literally as I9 - "
                  "adjoint_octet_projector(), so P1 + P8 = I9 resolves "
                  "3 x 3bar = 1 + 8 exactly.")
      .def_static("octet_bilinear", &ColorFiber::octetBilinear,
                  py::arg("i"), py::arg("j"),
                  "The 8x8 traceless even bilinear T_ij = "
                  "a_i^dag a_j - (delta_ij/3) N on Fock space "
                  "(= d_gamma(traceless_part(matrix_unit(i, j)))): "
                  "conserves N (even fermion parity), and "
                  "the nine T_ij span the octet.")
      .def_static("adjoint_casimir_matrix", &ColorFiber::adjointCasimirMatrix,
                  "The 9x9 quadratic Casimir of the adjoint action, "
                  "C = sum_a K_a^2 with K_a vec(M) = vec([lambda_a/2, M]); "
                  "exactly C = 3 P8.")
      .def_static("adjoint_casimir", &ColorFiber::adjointCasimir,
                  py::arg("m"),
                  "The adjoint-Casimir Rayleigh quotient in [0, 3]: exactly "
                  "3 for traceless M, 0 for M ~ I, NaN for M = 0.")
      .def_static("omega", &ColorFiber::omega,
                  "The primitive cube root of unity as its algebraic value "
                  "(-1 + i sqrt(3))/2 (never exp), so 1 + omega + omega^2 "
                  "cancels exactly in floating point.")
      .def_static("fourier_frame", &ColorFiber::fourierFrame,
                  "The exact unitary Fourier frame F3 with entries "
                  "omega^{jk}/sqrt(3), assembled from the algebraic table "
                  "{1, omega, omega^2} by exponent jk mod 3.")
      .def_static("fourier_basis_vector", &ColorFiber::fourierBasisVector,
                  py::arg("k"),
                  "Column k of F3: the Z3 character vector "
                  "(1, omega^k, omega^{2k})/sqrt(3).")
      .def_static("omega_phase_state", &ColorFiber::omegaPhaseState,
                  "The phase pattern (1, omega, omega^2)/sqrt(3), "
                  "identified as one color basis vector "
                  "(fourier_basis_vector(1)); its cyclic orbit under pointwise "
                  "Z3 powers is the exact orthonormal triad = the columns of "
                  "F3.")
      .def_static("perimeter", &ColorFiber::perimeter, py::arg("z"),
                  "The triangle perimeter sum_i |z_i|^{1/2} of three stored "
                  "complex squared lengths (the L1 geometric datum).")
      .def_static("perimeter_normalized", &ColorFiber::perimeterNormalized,
                  py::arg("z"),
                  "Rescale the squared lengths so the perimeter is one -- a "
                  "geometric scale gauge (L1), never a state normalization.")
      .def_static("hilbert_norm", &ColorFiber::hilbertNorm, py::arg("z"),
                  "The Hilbert L2 norm ||z||_2.")
      .def_static("hilbert_normalized", &ColorFiber::hilbertNormalized,
                  py::arg("z"),
                  "z / ||z||_2 with <c|c> = 1 -- the state normalization, "
                  "distinct from the perimeter gauge.")
      .def_static("color_vector", &ColorFiber::colorVector, py::arg("z"),
                  "The color vector from the stored complex squared "
                  "lengths: c = z / ||z||_2.")
      .def_static("color_wedge",
                  py::overload_cast<const Eigen::Matrix3cd&>(
                      &ColorFiber::colorWedge),
                  py::arg("c"),
                  "The color-wedge (singlet) amplitude det C = eps_ijk C_i1 "
                  "C_j2 C_k3; det(gC) = det(C) for g in SU(3).")
      .def_static("color_wedge_columns",
                  py::overload_cast<const Eigen::Vector3cd&,
                                    const Eigen::Vector3cd&,
                                    const Eigen::Vector3cd&>(
                      &ColorFiber::colorWedge),
                  py::arg("a"), py::arg("b"), py::arg("c"),
                  "colorWedge of three explicit color columns.")
      .def_static("singlet_gram", &ColorFiber::singletGram, py::arg("c"),
                  "det(C^dag C) = |det C|^2 = ||c1 ^ c2 ^ c3||^2: exactly "
                  "zero for duplicate color modes, exactly one for an "
                  "orthonormal triad.")
      .def_static("is_special_unitary", &ColorFiber::isSpecialUnitary,
                  py::arg("g"), py::arg("tol") = 1e-12,
                  "Certify g in SU(3): ||g^dag g - I||_max <= tol and "
                  "|det g - 1| <= tol.")
      .def_static("sector_weights", &ColorFiber::sectorWeights,
                  py::arg("state"),
                  "The four occupation-sector weights of an 8-dimensional "
                  "Fock vector (their sum is ||psi||^2 exactly).")
      .def_static("octet_read", &ColorFiber::octetRead, py::arg("m"),
                  "The octet/singlet Frobenius weights of a 3x3 bilinear.")
      .def_static("verify_constant_algebra", &ColorFiber::verifyConstantAlgebra,
                  "Re-derive every constant-algebra identity and return the "
                  "maximum absolute residual (run at startup in debug "
                  "builds; callable in every build).")
      .def_static("constant_algebra_certificate",
                  &ColorFiber::constantAlgebraCertificate,
                  "The AlgebraicallyExact certificate of the constant "
                  "algebra (measured verifyConstantAlgebra residual against "
                  "the startup tolerance 1e-12).");

  py::class_<ColorAnchor>(m, "ColorAnchor",
      R"doc(The calibrated weighted oriented-triangle anchoring kernel for an
abstract rank-three band: A_tau = |W_tau|^{1/2} R_tau Phi per declared
oriented triangle, atlas score a^2 = sum_tau w_tau |det A_tau|^2 with the
convex weighting declared before the data are examined (post-hoc
re-weighting raises).

Exact identity and domain: with the frame |W|-orthonormal (verified per
evaluate and reported as frame_gram_residual) and |W| triangle-decoupled
(any diagonal per-edge metric, the discrete exterior calculus / Hodge
case), each |det A_tau|^2 = det(A_tau^dag A_tau) <= 1 because
R_tau^dag |W_tau| R_tau is dominated by |W|, so the score is calibrated to
[0, 1], with value one exactly at full concentration on the weighted edge
span.  A single literal triangle is the exact oracle.  For a general
Hermitian (coupled) weight the <= 1 bound is checked (calibration_margin),
never assumed.  Signed sectors restrict with |W_tau|^{1/2} and report each
restricted block's Krein signature separately.

Operates only on caller-supplied inputs (frame over oriented edges, edge
weight data, oriented-triangle descriptors); mutates nothing; never enters
the emergence objective; contains no transport code.)doc")
      .def(py::init<std::vector<OrientedTriangle>>(), py::arg("triangles"),
           "Declare the atlas with the uniform convex weighting 1/T.")
      .def(py::init<std::vector<OrientedTriangle>, std::vector<double>>(),
           py::arg("triangles"), py::arg("weights"),
           "Declare the atlas with an explicit convex weighting (each >= 0, "
           "summing to one within 1e-12).")
      .def("triangles", &ColorAnchor::triangles,
           "The declared oriented triangles (immutable).")
      .def("weights", &ColorAnchor::weights, "The declared convex weights.")
      .def("weighting_id", &ColorAnchor::weightingId,
           "'uniform' or 'declared'.")
      .def("overlaps_another", &ColorAnchor::overlapsAnother, py::arg("index"),
           "Whether declared triangle `index` shares a boundary edge with "
           "another declared triangle: the overlap relation the "
           "determinant-phase coherence is recorded on.")
      .def("overlapping_triangle_count", &ColorAnchor::overlappingTriangleCount,
           "How many declared triangles overlap another (0 on a disjoint "
           "atlas, where the coherence is unknown).")
      .def("sealed", &ColorAnchor::sealed,
           "True once any data have been evaluated (weighting sealed).")
      .def("declare_weights", &ColorAnchor::declareWeights, py::arg("weights"),
           "Replace the declared convex weighting; allowed only before the "
           "first evaluate().  Afterwards post-hoc weight selection is "
           "rejected (raises).")
      .def_static("accepts", &ColorAnchor::accepts, py::arg("profile"),
                  py::arg("min_score") = ColorAnchor::kDefaultMinScore,
                  py::arg("min_phase_coherence") =
                      ColorAnchor::kDefaultMinPhaseCoherence,
                  "The triangle-anchor acceptance predicate, shared "
                  "by the quark verdict and by the colour kernels gated "
                  "on it. A profile passes when a weighting was actually "
                  "declared (an empty weighting_id is missing evidence, "
                  "not a zero score), its calibration certificate holds, "
                  "and both the atlas score and the determinant-phase "
                  "coherence meet their floors.")
      .def_static("gate_for", &ColorAnchor::gateFor, py::arg("profile"),
                  py::arg("min_score") = ColorAnchor::kDefaultMinScore,
                  py::arg("min_phase_coherence") =
                      ColorAnchor::kDefaultMinPhaseCoherence,
                  "The AnchorGate for a profile: accepts() plus the "
                  "provenance a refusal needs to name what failed.")
      .def("evaluate",
           py::overload_cast<const Eigen::MatrixXcd&, const Eigen::VectorXd&,
                             double>(&ColorAnchor::evaluate),
           py::arg("frame"), py::arg("edge_weights"),
           py::arg("gram_tolerance") = 1e-9,
           "Evaluate against a diagonal (possibly signed) per-edge weight "
           "vector, the domain where the [0,1] calibration bound is "
           "exact.  The frame must be |W|-orthonormal within "
           "gram_tolerance (use orthonormalizeFrame).")
      .def("evaluate_matrix",
           py::overload_cast<const Eigen::MatrixXcd&, const Eigen::MatrixXcd&,
                             double>(&ColorAnchor::evaluate),
           py::arg("frame"), py::arg("weight"),
           py::arg("gram_tolerance") = 1e-9,
           "Evaluate against a general Hermitian ExE weight matrix; the "
           "calibration bound is checked (calibration_margin), not "
           "assumed.")
      .def_static("anchor_matrix", &ColorAnchor::anchorMatrix,
                  py::arg("frame"), py::arg("edge_weights"), py::arg("tri"),
                  "The raw 3x3 weighted anchor matrix A_tau = |W_tau|^{1/2} "
                  "R_tau Phi of one triangle (diagonal weights; no "
                  "normalization check).")
      .def_static("orthonormalize_frame",
                  py::overload_cast<const Eigen::MatrixXcd&,
                                    const Eigen::VectorXd&>(
                      &ColorAnchor::orthonormalizeFrame),
                  py::arg("frame"), py::arg("edge_weights"),
                  "The |W|-orthonormalized frame phi (Phi^dag |W| "
                  "Phi)^{-1/2} for a diagonal per-edge weight vector.")
      .def_static("orthonormalize_frame_matrix",
                  py::overload_cast<const Eigen::MatrixXcd&,
                                    const Eigen::MatrixXcd&>(
                      &ColorAnchor::orthonormalizeFrame),
                  py::arg("frame"), py::arg("weight"),
                  "Matrix-weight overload of orthonormalize_frame (Hermitian "
                  "W; uses the eigen-modulus |W|).");
}
