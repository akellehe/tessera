// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <limits>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chainhodge/ChainHodge.h"
#include "chainhodge/CovariantChainHodge.h"
#include "chainhodge/BandDerivative.h"
#include "chainhodge/FaceAnchor.h"
#include "chainhodge/DressedAnchor.h"
#include "chainhodge/RieszBand.h"
#include "chainhodge/LorentzianFamily.h"
#include "chainhodge/PencilSchur.h"
#include "chainhodge/SparsePencil.h"
#include "chainhodge/SparsePencilSolver.h"
#include "chainhodge/SparseRank.h"
#include "chainhodge/WhitneyMass.h"
#include "cobordism/ChainComplex.h"
#include "spacetime/Spacetime.h"

namespace py = pybind11;
using namespace tessera::chainhodge;
using tessera::cobordism::ChainComplex;

void register_chainhodge(py::module_ m) {
  py::enum_<Preset>(m, "Preset",
      "Chain metric family: L2 (Whitney, default) or GRASSMANN_ALL (the retained "
      "Grassmann projection metric with its documented deviation).")
      .value("L2", Preset::L2)
      .value("GRASSMANN_ALL", Preset::GRASSMANN_ALL);

  py::enum_<Branch>(m, "Branch",
      "How sqrt(det g_T) is fixed per top simplex: continuation from the unit "
      "Euclidean reference along the straight segment, or the Kontsevich-Segal "
      "rule (principal eigenvalue roots, cut resolved to +i).")
      .value("Continuation", Branch::Continuation)
      .value("KontsevichSegal", Branch::KontsevichSegal);

  py::class_<InstanceCertificate>(m, "InstanceCertificate",
      R"doc(The instance certificate: Kontsevich-Segal allowability of every top
simplex, the minimal margin pi - sum_i |arg lambda_i(g_T)|, the volumes on
the declared branch, the Gram determinants, continuation ambiguity, and the Lorentzian
protocol rotation epsilon (NaN until set).)doc")
      .def_readonly("branch", &InstanceCertificate::branch)
      .def_readonly("allowable", &InstanceCertificate::allowable)
      .def_readonly("margin", &InstanceCertificate::margin)
      .def_readonly("margins", &InstanceCertificate::margins)
      .def_readonly("volumes", &InstanceCertificate::volumes)
      .def_readonly("gramDeterminants", &InstanceCertificate::gramDeterminants)
      .def_readonly("volumeWindings", &InstanceCertificate::volumeWindings,
          "The Riemann-sheet label of each volume root: the signed number of "
          "turns det g_T makes about zero along the continuation that fixed it, "
          "so volumes[t] is (-1)**volumeWindings[t] times the principal root "
          "over d!. Carried because the root, not the squared volume, is where "
          "the branch is: two instances with the same gramDeterminants and "
          "different volumeWindings are on different sheets of the same "
          "geometry, and only the label says so. Zero throughout on the "
          "Kontsevich-Segal branch, which declares its sheet eigenvalue by "
          "eigenvalue rather than along a path.")
      .def_readonly("continuationAmbiguous", &InstanceCertificate::continuationAmbiguous)
      .def_readonly("ambiguousTopSimplices", &InstanceCertificate::ambiguousTopSimplices)
      .def_readwrite("epsilon", &InstanceCertificate::epsilon);

  py::class_<TopSimplexBlock>(m, "TopSimplexBlock",
      "One top simplex's local block of M_k over its k-faces, with the canonical "
      "cell and edge indices and (when requested) the derivative per local edge.")
      .def_readonly("topIndex", &TopSimplexBlock::topIndex)
      .def_readonly("cellIndices", &TopSimplexBlock::cellIndices)
      .def_readonly("edgeIndices", &TopSimplexBlock::edgeIndices)
      .def_readonly("block", &TopSimplexBlock::block)
      .def_readonly("derivative", &TopSimplexBlock::derivative);

  py::class_<WhitneyMass>(m, "WhitneyMass",
      R"doc(Sparse complex symmetric inverse chain metrics M_k of the chain-level Whitney
Hodge pencil, assembled per top simplex from the complex squared edge lengths alone.
Reference orientation is ascending vertex id (ChainComplex.fromTopCells). Sparse
results are scipy CSC matrices.

Reference: Whitney, "Geometric Integration Theory", 1957.)doc")
      .def_static("complexOf", &WhitneyMass::complexOf, py::arg("spacetime"),
           "ChainComplex.fromTopCells over the spacetime's top simplices (sorted vertex ids).")
      .def_static("squaredLengthsOf", &WhitneyMass::squaredLengthsOf,
           py::arg("spacetime"), py::arg("complex"),
           "Complex squared edge lengths l^2 in the complex's canonical edge order.")
      .def_static("assemble",
           py::overload_cast<const ChainComplex &, const SquaredLengths &, int, Branch>(
               &WhitneyMass::assemble),
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           py::arg("branch") = Branch::Continuation,
           "The Whitney inverse chain metric M_k (Preset.L2) on the declared branch.")
      .def_static("assemblePreset",
           py::overload_cast<const ChainComplex &, const SquaredLengths &, int, Preset, Branch>(
               &WhitneyMass::assemble),
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"), py::arg("preset"),
           py::arg("branch") = Branch::Continuation,
           "Preset dispatch: L2 -> M_k (inverse chain metric); GRASSMANN_ALL -> G_k (chain metric).")
      .def_static("assembleGrassmann", &WhitneyMass::assembleGrassmann,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           "The Grassmann projection chain metric G_k = multiplicity o blade pairing.")
      .def_static("allowabilityMargin", &WhitneyMass::allowabilityMargin,
           py::arg("complex"), py::arg("squared_lengths"),
           "min over top simplices of pi - sum_i |arg lambda_i(g_T)|.")
      .def_static("certificate", &WhitneyMass::certificate,
           py::arg("complex"), py::arg("squared_lengths"),
           py::arg("branch") = Branch::Continuation,
           "The full instance certificate.")
      .def_static("volumeOnBranch",
           [](const Eigen::MatrixXcd &gram, Branch branch) {
             bool ambiguous = false;
             const Complex v = WhitneyMass::volumeOnBranch(gram, branch, &ambiguous);
             return py::make_tuple(v, ambiguous);
           },
           py::arg("gram"), py::arg("branch") = Branch::Continuation,
           "(sqrt(det g)/d!, ambiguous) for one Gram matrix on the declared branch.")
      .def_static("volumeWindingOnBranch",
           [](const Eigen::MatrixXcd &gram, Branch branch) {
             bool ambiguous = false;
             int winding = 0;
             WhitneyMass::volumeOnBranch(gram, branch, &ambiguous, &winding);
             return winding;
           },
           py::arg("gram"), py::arg("branch") = Branch::Continuation,
           "The Riemann-sheet label of the root volumeOnBranch takes: the signed "
           "number of turns det g makes about zero along the continuation from "
           "the unit Euclidean reference, so that the volume is (-1)**winding "
           "times the principal root over d!. Zero on the Kontsevich-Segal "
           "branch, which declares its sheet eigenvalue by eigenvalue rather "
           "than along a path.")
      .def_static("volumeContinuedFrom",
           [](const Eigen::MatrixXcd &gramFrom, int windingFrom,
              const Eigen::MatrixXcd &gramTo) {
             bool ambiguous = false;
             int windingTo = 0;
             const Complex v = WhitneyMass::volumeContinuedFrom(
                 gramFrom, windingFrom, gramTo, &windingTo, &ambiguous);
             return py::make_tuple(v, ambiguous, windingTo);
           },
           py::arg("gram_from"), py::arg("winding_from"), py::arg("gram_to"),
           "(sqrt(det g_to)/d!, ambiguous, winding) continued from a declared "
           "previous geometry and its sheet instead of from the fixed Euclidean "
           "reference. Starting from the geometry an instance actually came "
           "from is what makes a family of instances one continued state rather "
           "than a sequence of independent principal-value choices: the sheet "
           "composes along the path, so a loop of squared lengths about a zero "
           "of det g returns the winding one higher and the volume negated.")
      .def_static("marginOf", &WhitneyMass::marginOf, py::arg("gram"),
           "pi - sum_i |arg lambda_i(g)| for one Gram matrix.")
      .def_static("topSimplexBlocks", &WhitneyMass::topSimplexBlocks,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           py::arg("branch") = Branch::Continuation, py::arg("with_derivative") = false,
           "Per-top-simplex local blocks of M_k (and derivatives when requested).")
      .def_static("assembleDerivative", &WhitneyMass::assembleDerivative,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"), py::arg("edge_index"),
           py::arg("branch") = Branch::Continuation,
           "dM_k/ds_e for the edge at the given canonical index, sparse.")
      .def_static("assembleDirectionalDerivative", &WhitneyMass::assembleDirectionalDerivative,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"), py::arg("direction"),
           py::arg("branch") = Branch::Continuation,
           "D_v M_k = sum_e v_e dM_k/ds_e along a squared-length direction (one entry per edge, "
           "canonical order), sparse.")
      .def_static("assembleSecondDerivatives", &WhitneyMass::assembleSecondDerivatives,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"), py::arg("direction"),
           py::arg("branch") = Branch::Continuation,
           "D_v dM_k/ds_e for every edge e along a squared-length direction v, a list indexed by "
           "canonical edge, sparse.")
      .def_static("derivativeContraction", &WhitneyMass::derivativeContraction,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           py::arg("X"), py::arg("Y"), py::arg("branch") = Branch::Continuation,
           "Per-edge tr(X^T (dM_k/ds_e) Y) from the local blocks (transpose pairing).")
      .def_static("vertexProductIntegral", &WhitneyMass::vertexProductIntegral,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("top_index"),
           py::arg("vertices"), py::arg("branch") = Branch::Continuation,
           "The integral over one top simplex of the product of the barycentric coordinate "
           "functions of the listed vertex ids (with repetition): |T| d! prod_v m_v! / (d + m)!.")
      .def_static("assembleVertexPotential", &WhitneyMass::assembleVertexPotential,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("potential"),
           py::arg("branch") = Branch::Continuation,
           "M_0[V]: the mass matrix weighted by a function given by its vertex values (canonical "
           "C_0 order), sparse on the pattern of M_0. M_0[1] = M_0.")
      .def_static("pairLoads", &WhitneyMass::pairLoads, py::arg("complex"), py::arg("squared_lengths"),
           py::arg("links_x"), py::arg("links_y"), py::arg("x"), py::arg("Y"),
           py::arg("branch") = Branch::Continuation,
           "The loads int phi_c x y of the product of a section x of the connection links_x with every "
           "column of Y, sections of links_y: the three factors carried to the first vertex of every top "
           "simplex and the load carried back.")
      .def_static("vertexDensityContraction", &WhitneyMass::vertexDensityContraction,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("X"), py::arg("Y"),
           py::arg("branch") = Branch::Continuation,
           "rho_c = d/dV_c tr(X^T M_0[V] Y) per vertex, from the local integrals without forming "
           "the covariance (transpose pairing).");
  py::enum_<PencilVariable>(m, "PencilVariable",
      "Which vector a pencil eigenproblem A x = lambda B x is written in: geometric "
      "images (Whitney) or chains (Grassmann).")
      .value("GeometricImage", PencilVariable::GeometricImage)
      .value("Chain", PencilVariable::Chain);

  py::class_<PairLoads>(m, "PairLoads",
                        "WhitneyMass.pairLoads for many calls on one complex: the volumes of the top simplices "
                        "and the edges that carry a value to the first vertex of each are found once.")
      .def(py::init<const ChainComplex &, const SquaredLengths &, Branch>(), py::arg("complex"),
           py::arg("squared_lengths"), py::arg("branch") = Branch::Continuation)
      .def("loads", &PairLoads::loads, py::arg("links_x"), py::arg("links_y"), py::arg("x"), py::arg("Y"),
           py::call_guard<py::gil_scoped_release>(),
           "The loads int phi_c x y of the product of a section x of the connection links_x with every column "
           "of Y, sections of links_y.")
      .def("loadsPhaseDerivativeAlong", &PairLoads::loadsPhaseDerivativeAlong, py::arg("links_x"),
           py::arg("links_y"), py::arg("x"), py::arg("Y"), py::arg("edge_weights"),
           py::call_guard<py::gil_scoped_release>(),
           "The derivative of loads along the change U_e -> U_e exp(i t w_e) of the links of Y, one weight per "
           "edge in the canonical edge order.")
      .def_property_readonly("numVertices", &PairLoads::numVertices)
      .def_property_readonly("numEdges", &PairLoads::numEdges);

  py::class_<Pencil>(m, "Pencil", "A complex symmetric pencil A - lambda B at one degree, dense.")
      .def_readonly("degree", &Pencil::degree)
      .def_readonly("variable", &Pencil::variable)
      .def_readonly("A", &Pencil::A)
      .def_readonly("B", &Pencil::B);

  py::class_<SingularSplit>(m, "SingularSplit",
      "The singular values on either side of one rank split: sigmaAt = sigma_at, "
      "sigmaNext = sigma_{at+1}, their gap, the numerical rank and its tolerance.")
      .def_readonly("at", &SingularSplit::at)
      .def_readonly("rank", &SingularSplit::rank)
      .def_readonly("tolerance", &SingularSplit::tolerance)
      .def_readonly("largest", &SingularSplit::largest)
      .def_readonly("sigmaAt", &SingularSplit::sigmaAt)
      .def_readonly("sigmaNext", &SingularSplit::sigmaNext)
      .def_readonly("gap", &SingularSplit::gap)
      .def_readonly("dense", &SingularSplit::dense);

  py::class_<SparseKernel>(m, "SparseKernel",
      "The kernel of a sparse matrix (orthonormal basis) with the split at its numerical rank.")
      .def_readonly("split", &SparseKernel::split)
      .def_readonly("basis", &SparseKernel::basis);

  py::class_<SparseRank>(m, "SparseRank",
      R"doc(Numerical ranks of sparse matrices with the singular values on either side of
the decision, without forming a dense matrix of the size of the problem: kernel(A)
by a thresholded sparse QR of A (the kernel by back-substitution, the last kept
singular value by inverse subspace iteration on the triangular factor, the first
discarded one as ||A N||); congruence(C, X, inverse, rho) for P = C^T X^{+-1} C with
C an integer matrix of exact rank rho, reduced to the rho x rho matrix W X^{+-1} W^T
by a sparse QR of C^T.

Reference: Foster & Davis, "Algorithm 933: Reliable calculation of numerical rank,
null space bases, pseudoinverse solutions, and basic solutions using SuiteSparseQR",
ACM TOMS 40 (2013).)doc")
      .def_static("kernel", &SparseRank::kernel, py::arg("A"), py::arg("kappa") = 10.0,
           "The kernel of sparse A with the split at its numerical rank.")
      .def_static("congruence", &SparseRank::congruence, py::arg("C"), py::arg("X"),
           py::arg("inverse"), py::arg("structural_rank"), py::arg("kappa") = 10.0,
           "The split of P = C^T X C (or C^T X^{-1} C) at the exact rank of C.")
      .def_static("fromSingularValues", &SparseRank::fromSingularValues, py::arg("singular_values"),
           py::arg("rows"), py::arg("cols"), py::arg("kappa") = 10.0, py::arg("at") = -1,
           "A dense SVD's singular values as a split (at the numerical rank when at < 0).");

  py::class_<HarmonicRead>(m, "HarmonicRead",
      "Harmonic chains H_k, their geometric images, and the kernel's rank certificate.")
      .def_readonly("degree", &HarmonicRead::degree)
      .def_readonly("chains", &HarmonicRead::chains)
      .def_readonly("images", &HarmonicRead::images)
      .def_readonly("nullity", &HarmonicRead::nullity)
      .def_readonly("rank", &HarmonicRead::rank)
      .def_readonly("tolerance", &HarmonicRead::tolerance)
      .def_readonly("gap", &HarmonicRead::gap)
      .def_readonly("largestSingular", &HarmonicRead::largestSingular)
      .def_readonly("lastKept", &HarmonicRead::lastKept)
      .def_readonly("firstDiscarded", &HarmonicRead::firstDiscarded)
      .def_readonly("dense", &HarmonicRead::dense);

  py::class_<SparseCostReport>(m, "SparseCostReport",
      "What one operation of the sparse production path cost: wall time, memory and fill-in, "
      "measured on the operation itself. fillIn is the stored entries of the factors over those "
      "of the matrix they factorize; factorMegabytes is the factors' own memory, computed from "
      "their stored entries; residentMegabytes is the change in the process's resident set size "
      "across the operation, NaN where the operating system does not publish it.")
      .def_readonly("operation", &SparseCostReport::operation)
      .def_readonly("degree", &SparseCostReport::degree)
      .def_readonly("dimension", &SparseCostReport::dimension)
      .def_readonly("systemRows", &SparseCostReport::systemRows)
      .def_readonly("systemNonZeros", &SparseCostReport::systemNonZeros)
      .def_readonly("factorNonZeros", &SparseCostReport::factorNonZeros)
      .def_readonly("fillIn", &SparseCostReport::fillIn)
      .def_readonly("wallSeconds", &SparseCostReport::wallSeconds)
      .def_readonly("factorMegabytes", &SparseCostReport::factorMegabytes)
      .def_readonly("residentMegabytes", &SparseCostReport::residentMegabytes)
      .def_readonly("rightHandSides", &SparseCostReport::rightHandSides);

  py::class_<SparseKernelRead>(m, "SparseKernelRead",
      "The null space of a sparse matrix from SparseRank::kernel, with the singular values on "
      "either side of the rank decision and the cost of the read. The factorization is "
      "SparseRank's and does not publish its stored entries, so the cost's fill-in and factor "
      "memory are NaN while its wall time and process memory are measured.")
      .def_readonly("kernel", &SparseKernelRead::kernel)
      .def_readonly("split", &SparseKernelRead::split)
      .def_readonly("rank", &SparseKernelRead::rank)
      .def_readonly("tolerance", &SparseKernelRead::tolerance)
      .def_readonly("cost", &SparseKernelRead::cost);

  py::class_<RankReport>(m, "RankReport",
      "The rank conditions (R1)-(R4) at one degree.")
      .def_readonly("degree", &RankReport::degree)
      .def_property_readonly("measured", [](const RankReport &r) {
        return std::vector<int>(r.measured.begin(), r.measured.end()); })
      .def_property_readonly("expected", [](const RankReport &r) {
        return std::vector<int>(r.expected.begin(), r.expected.end()); })
      .def_property_readonly("holds", [](const RankReport &r) {
        return std::vector<bool>(r.holds.begin(), r.holds.end()); })
      .def_property_readonly("splits", [](const RankReport &r) {
        return std::vector<SingularSplit>(r.splits.begin(), r.splits.end()); },
        "Per condition, the singular values at and beyond its exact rank.")
      .def_readonly("dense", &RankReport::dense)
      .def_readonly("decompositionHolds", &RankReport::decompositionHolds)
      .def_readonly("kernelIsHarmonic", &RankReport::kernelIsHarmonic)
      .def_readonly("kappa", &RankReport::kappa);

  py::class_<SpectrumRead>(m, "SpectrumRead", "Dense spectrum of one degree's pencil.")
      .def_readonly("degree", &SpectrumRead::degree)
      .def_readonly("eigenvalues", &SpectrumRead::eigenvalues)
      .def_readonly("residual", &SpectrumRead::residual)
      .def_readonly("vectors", &SpectrumRead::vectors);

  py::class_<ChainHodge>(m, "ChainHodge",
      R"doc(The chain-level Hodge pencil of a complexified simplicial complex: sparse
inverse chain metrics, geometric images by solves, the symmetric pencil and its
auxiliary form, harmonic chains H_k = M_k ker S, rank conditions R1-R4, exact Betti
numbers, and the dense spectrum below the crossover. The adjoint is the transpose;
no conjugation enters any operator.

Reference: Eckmann, "Harmonische Funktionen und Randwertaufgaben in einem Komplex",
1944, for the combinatorial Hodge Laplacian.)doc")
      .def(py::init<ChainComplex, SquaredLengths, Preset, Branch, int, double>(),
           py::arg("complex"), py::arg("squared_lengths"), py::arg("preset") = Preset::L2,
           py::arg("branch") = Branch::Continuation,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           py::arg("epsilon") = std::numeric_limits<double>::quiet_NaN())
      .def("complex", &ChainHodge::complex, py::return_value_policy::reference_internal)
      .def("squaredLengths", &ChainHodge::squaredLengths)
      .def("dimension", &ChainHodge::dimension)
      .def("preset", &ChainHodge::preset)
      .def("branch", &ChainHodge::branch)
      .def("crossoverDimension", &ChainHodge::crossoverDimension)
      .def("certificate", &ChainHodge::certificate)
      .def("size", &ChainHodge::size, py::arg("k"))
      .def("Minv", [](const ChainHodge &c, int k) { return SparseMatrix(c.Minv(k)); }, py::arg("k"),
           "The sparse inverse chain metric M_k (Whitney preset).")
      .def("chainMetricSparse", [](const ChainHodge &c, int k) { return SparseMatrix(c.chainMetricSparse(k)); },
           py::arg("k"), "The sparse chain metric G_k (Grassmann preset).")
      .def("boundary", [](const ChainHodge &c, int k) { return SparseMatrix(c.boundary(k)); }, py::arg("k"),
           "The sparse boundary map d_k.")
      .def("applyG", &ChainHodge::applyG, py::arg("k"), py::arg("c"), "G_k c: the geometric image, by solve.")
      .def("applyMinv", &ChainHodge::applyMinv, py::arg("k"), py::arg("c"), "M_k c.")
      .def("pencil", &ChainHodge::pencil, py::arg("k"), "The dense pencil at degree k.")
      .def("pencilAux", &ChainHodge::pencilAux, py::arg("k"), "A~_k = M_k A_k M_k (Whitney), dense.")
      .def("applyPencilOperator", &ChainHodge::applyPencilOperator, py::arg("k"), py::arg("Z"),
           "A~_k Z (Whitney) or A_k Z (Grassmann) by sparse products and sparse solves: the "
           "production path's pencil operator, defined at any size.")
      .def("stackedMatrix", [](const ChainHodge &c, int k) { return SparseMatrix(c.stackedMatrix(k)); },
           py::arg("k"),
           "The sparse stacked cochain matrix S of degree k, whose kernel is the harmonic space.")
      .def_static("sparseNullSpace",
           [](const SparseMatrix &S, double kappa) { return ChainHodge::sparseNullSpace(S, kappa); },
           py::arg("S"), py::arg("kappa") = 10.0,
           "ker S by rank-revealing sparse QR of S^H, with the rank, the threshold and the cost; "
           "neither S nor the orthogonal factor is densified.")
      .def("hodgeOperator", &ChainHodge::hodgeOperator, py::arg("k"), "The dense L_k on chains.")
      .def("harmonicChains", &ChainHodge::harmonicChains, py::arg("k"), py::arg("kappa") = 10.0,
           py::arg("force_sparse") = false, "H_k = M_k ker S with the kernel's rank certificate.")
      .def("geometricImage", &ChainHodge::geometricImage, py::arg("k"), py::arg("H"), "G_k H.")
      .def("harmonicGram", &ChainHodge::harmonicGram, py::arg("read"), "Phi^T G_k Phi = Z^T M_k Z.")
      .def("rankConditions", &ChainHodge::rankConditions, py::arg("k"), py::arg("kappa") = 10.0,
           py::arg("force_sparse") = false,
           "The rank conditions (R1)-(R4) at degree k, each with the singular values at and "
           "beyond its exact rank; sparse at or above the crossover.")
      .def("betti", &ChainHodge::betti, "Betti numbers over Q, exact.")
      .def("spectrum", &ChainHodge::spectrum, py::arg("k"), "Dense spectrum of the degree-k pencil.");
  py::class_<LorentzianRead>(m, "LorentzianRead",
      "One member of the epsilon family: allowability, margin, the harmonic read with its "
      "gap, and the dense spectrum when requested.")
      .def_readonly("epsilon", &LorentzianRead::epsilon)
      .def_readonly("allowable", &LorentzianRead::allowable)
      .def_readonly("margin", &LorentzianRead::margin)
      .def_readonly("degree", &LorentzianRead::degree)
      .def_readonly("harmonic", &LorentzianRead::harmonic)
      .def_readonly("eigenvalues", &LorentzianRead::eigenvalues);

  py::class_<LorentzianExtrapolation>(m, "LorentzianExtrapolation",
      "A labeled least-squares extrapolation of reads at epsilon > 0 to epsilon -> 0.")
      .def_readonly("epsilons", &LorentzianExtrapolation::epsilons)
      .def_readonly("values", &LorentzianExtrapolation::values)
      .def_readonly("order", &LorentzianExtrapolation::order)
      .def_readonly("extrapolated", &LorentzianExtrapolation::extrapolated)
      .def_readonly("residual", &LorentzianExtrapolation::residual)
      .def_readonly("label", &LorentzianExtrapolation::label);

  py::class_<LorentzianFamily>(m, "LorentzianFamily",
      R"doc(The Lorentzian protocol (integration specification, Requirement 2): the family
s_e(epsilon) with the timelike part of every squared length rotated by e^{-2 i epsilon},
at reported epsilon > 0; reads at epsilon = 0 exist only inside a family and carry their
gap; extrapolation to epsilon -> 0 is a separate, labeled step.

Every squared length s_e is declared with its timelike part tau_e, the contribution of
the edge's temporal displacement (-e^{2 phi} dt^2 for a metric e^{2 phi}(-dt^2 + dx^2));
the family is s_e(epsilon) = (s_e - tau_e) + e^{-2 i epsilon} tau_e. The split is an
input and is never inferred from s_e.

Reference: Kontsevich & Segal, "Wick rotation and the positivity of energy in quantum
field theory", arXiv:2105.10161.)doc")
      .def_static("rotate", &LorentzianFamily::rotate, py::arg("squared_lengths"),
           py::arg("timelike_parts"), py::arg("epsilon"),
           "s_e + (e^{-2 i epsilon} - 1) tau_e: every timelike part rotated, the spacelike "
           "part s_e - tau_e unchanged. Raises ValueError on mismatched lengths, a "
           "non-finite timelike part, or epsilon < 0.")
      .def_static("instance", &LorentzianFamily::instance, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("timelike_parts"), py::arg("epsilon"),
           py::arg("preset") = Preset::L2, py::arg("branch") = Branch::Continuation,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           "The ChainHodge at epsilon, with epsilon on its certificate.")
      .def_static("sweep", &LorentzianFamily::sweep, py::arg("complex"), py::arg("squared_lengths"),
           py::arg("timelike_parts"), py::arg("epsilons"), py::arg("degree"),
           py::arg("preset") = Preset::L2, py::arg("branch") = Branch::Continuation,
           py::arg("kappa") = 10.0, py::arg("with_spectrum") = false,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           "Reads at every epsilon of the family at one degree. Raises ValueError when "
           "no epsilon > 0 is given (a read at epsilon = 0 is never reported alone) or "
           "any epsilon is negative.")
      .def_static("extrapolateToZero", &LorentzianFamily::extrapolateToZero, py::arg("epsilons"),
           py::arg("values"), py::arg("order") = 2,
           "Labeled polynomial extrapolation of reads at epsilon > 0 to epsilon -> 0.");
  py::class_<Connection>(m, "Connection",
      R"doc(A C* connection on the canonical edges x < y: U_xy per edge, U_yx = 1/U_xy exactly,
U_xx = 1. Gauge: U_xy -> g_x^{-1} U_xy g_y. Links are never
normalized or conjugated.)doc")
      .def(py::init<const ChainComplex &, std::vector<Complex>>(), py::arg("complex"), py::arg("links"))
      .def_static("trivial", &Connection::trivial, py::arg("complex"))
      .def_static("fromSpacetime", &Connection::fromSpacetime, py::arg("spacetime"), py::arg("complex"),
           "U_xy = exp(i phase) on the stored source->target orientation, inverted when the source is the larger id.")
      .def("links", &Connection::links)
      .def("edgeCount", &Connection::edgeCount)
      .def("link", &Connection::link, py::arg("x"), py::arg("y"))
      .def("inverse", &Connection::inverse)
      .def("gauge", &Connection::gauge, py::arg("g"))
      .def("curvature", &Connection::curvature, py::arg("p"), py::arg("q"), py::arg("r"))
      .def("holonomy", &Connection::holonomy, py::arg("walk"),
           "The ordered product of the links along a closed walk of directed steps (u, v): "
           "U_{v0 v1} U_{v1 v2} ... U_{v_{n-1} v0}, the Wilson loop in the walk's direction; exactly 1 on "
           "every closed walk iff the connection is a pure gauge.")
      .def("transportedPeriod", &Connection::transportedPeriod, py::arg("cochain"), py::arg("walk"),
           "The period of a 1-cochain (indexed like links(), each edge's value in the frame at its base "
           "vertex min(x, y)) over a closed walk of directed steps (u, v), taken with parallel transport to "
           "the walk's first vertex: sum_k s_k (U_{v0 v1} ... U_{v_{k-1} v_k}) U_{v_k, min(v_k, v_{k+1})} "
           "omega_k with s_k = +1 when v_k < v_{k+1} and -1 otherwise (the Edge.walkLoop convention). "
           "The plain signed sum on the trivial connection; g_{v0}^{-1} times the untwisted period under "
           "a pure gauge U_xy = g_x^{-1} g_y on a twisted-kernel cochain, so ratios of periods over walks "
           "with a common base point are gauge invariant.")
      .def("isUnitary", &Connection::isUnitary, py::arg("tolerance") = 1e-12);

  py::class_<CovarianceCertificate>(m, "CovarianceCertificate",
      "Residuals of the exact properties (i)-(vi) of CovariantChainHodge on an instance, "
      "measured and asserted on construction at tolerance 10 n eps cond; NaN means unmeasured.")
      .def_readonly("transposeMetric", &CovarianceCertificate::transposeMetric)
      .def_readonly("transposePencil", &CovarianceCertificate::transposePencil)
      .def_readonly("covarianceMetric", &CovarianceCertificate::covarianceMetric)
      .def_readonly("covariancePencil", &CovarianceCertificate::covariancePencil)
      .def_readonly("curvature", &CovarianceCertificate::curvature)
      .def_readonly("pairingInvariance", &CovarianceCertificate::pairingInvariance)
      .def_readonly("trivialReduction", &CovarianceCertificate::trivialReduction)
      .def_readonly("pureGaugeIsospectrality", &CovarianceCertificate::pureGaugeIsospectrality)
      .def_readonly("trivialReductionProbe", &CovarianceCertificate::trivialReductionProbe)
      .def_readonly("transposePencilProbe", &CovarianceCertificate::transposePencilProbe)
      .def_readonly("transposeOperatorProbe", &CovarianceCertificate::transposeOperatorProbe)
      .def_readonly("covariancePencilProbe", &CovarianceCertificate::covariancePencilProbe)
      .def_readonly("covarianceOperatorProbe", &CovarianceCertificate::covarianceOperatorProbe)
      .def_readonly("pureGaugeSimilarityProbe", &CovarianceCertificate::pureGaugeSimilarityProbe)
      .def_readonly("conditionEstimate", &CovarianceCertificate::conditionEstimate)
      .def_readonly("tolerance", &CovarianceCertificate::tolerance)
      .def_readonly("holds", &CovarianceCertificate::holds)
      .def_readonly("gaugeSeed", &CovarianceCertificate::gaugeSeed)
      .def_readonly("checkedDegree", &CovarianceCertificate::checkedDegree);

  py::class_<Contour>(m, "Contour",
      "A closed positively oriented contour as quadrature nodes and weights with "
      "(1/2 pi i) oint f = sum_j w_j f(zeta_j); circle(center, radius, nodes) is the trapezoidal rule.")
      .def_static("circle", &Contour::circle, py::arg("center"), py::arg("radius"), py::arg("nodes") = 32)
      .def_readonly("nodes", &Contour::nodes)
      .def_readonly("weights", &Contour::weights)
      .def_readonly("description", &Contour::description)
      .def("nodeCount", &Contour::nodeCount);

  py::class_<BandCertificate>(m, "BandCertificate",
      "Certificates of one Riesz band; NaN means unmeasured; no sign or "
      "inertia is extracted from B_C.")
      .def_readonly("contour", &BandCertificate::contour)
      .def_readonly("nodeCount", &BandCertificate::nodeCount)
      .def_readonly("idempotency", &BandCertificate::idempotency)
      .def_readonly("rank", &BandCertificate::rank)
      .def_readonly("rankTolerance", &BandCertificate::rankTolerance)
      .def_readonly("singularGap", &BandCertificate::singularGap)
      .def_readonly("resolventMax", &BandCertificate::resolventMax)
      .def_readonly("resolventProbeMax", &BandCertificate::resolventProbeMax)
      .def_readonly("detB", &BandCertificate::detB)
      .def_readonly("condB", &BandCertificate::condB)
      .def_readonly("pairingScale", &BandCertificate::pairingScale)
      .def_readonly("leftFrameAvailable", &BandCertificate::leftFrameAvailable)
      .def_readonly("leftFrameRefusal", &BandCertificate::leftFrameRefusal)
      .def_readonly("rightResidual", &BandCertificate::rightResidual)
      .def_readonly("leftResidual", &BandCertificate::leftResidual);

  py::class_<Band>(m, "Band",
      R"doc(One Riesz band of h_k(s,U): projector P on chains, right frame Phi,
the dual connection's frame Phi^vee on the same contour, images Z = G^U Phi, pairing
B_C = (Phi^vee)^T G^U Phi, the canonical left frame Phi~ = G^{U^-1} Phi^vee B_C^{-T} (empty
when refused), the reduced operator J = Phi~^T h Phi, the covariance Gamma = Phi Phi~^T, and
the certificates.)doc")
      .def_readonly("degree", &Band::degree)
      .def_readonly("contour", &Band::contour)
      .def_readonly("projector", &Band::projector)
      .def_readonly("frame", &Band::frame)
      .def_readonly("dualFrame", &Band::dualFrame)
      .def_readonly("images", &Band::images)
      .def_readonly("pairing", &Band::pairing)
      .def_readonly("leftFrame", &Band::leftFrame)
      .def_readonly("reduced", &Band::reduced)
      .def_readonly("covariance", &Band::covariance)
      .def_readonly("certificate", &Band::certificate)
      .def("rank", &Band::rank)
      .def("occupations", &Band::occupations);

  py::class_<PencilRegimeCertificate>(m, "PencilRegimeCertificate",
      "The pencil's measured metric regime: ComplexSymmetricPencil when the transpose identities "
      "(A~^U)^T = A~^{U^-1} and (M^U)^T = M^{U^-1} hold to tolerance, else NonNormal.")
      .def_readonly("regime", &PencilRegimeCertificate::regime)
      .def_readonly("symmetryDefect", &PencilRegimeCertificate::symmetryDefect)
      .def_readonly("metricSymmetryDefect", &PencilRegimeCertificate::metricSymmetryDefect)
      .def_readonly("trivialConnection", &PencilRegimeCertificate::trivialConnection)
      .def_readonly("tolerance", &PencilRegimeCertificate::tolerance);

  py::class_<CovariantChainHodge>(m, "CovariantChainHodge",
      R"doc(The covariant one-particle operator h_k(s,U): the sparse inverse chain metric
dressed by U_{b(sigma) b(tau)} and the incidences twisted by U_{b(tau) b(sigma)},
b(sigma) = min sigma, with the dressed pencil (A~_k^U, M_k^U) on images and the exact
properties (i)-(vi) measured on every instance.)doc")
      .def(py::init<const ChainHodge &, Connection, std::uint64_t, bool>(), py::arg("base"),
           py::arg("connection"), py::arg("gauge_seed") = 7, py::arg("measure_certificate") = true)
      .def("base", &CovariantChainHodge::base, py::return_value_policy::reference_internal)
      .def("connection", &CovariantChainHodge::connection, py::return_value_policy::reference_internal)
      .def("dimension", &CovariantChainHodge::dimension)
      .def("preset", &CovariantChainHodge::preset)
      .def("certificate", &CovariantChainHodge::certificate)
      .def("Minv", [](const CovariantChainHodge &c, int k) { return SparseMatrix(c.Minv(k)); }, py::arg("k"))
      .def("dressed", [](const CovariantChainHodge &c, int k) { return SparseMatrix(c.dressed(k)); }, py::arg("k"))
      .def("twistedBoundary", [](const CovariantChainHodge &c, int k) { return SparseMatrix(c.twistedBoundary(k)); }, py::arg("k"))
      .def("twistedBoundaryDual", [](const CovariantChainHodge &c, int k) { return SparseMatrix(c.twistedBoundaryDual(k)); }, py::arg("k"))
      .def("rho", &CovariantChainHodge::rho, py::arg("k"), py::arg("g"))
      .def("applyG", &CovariantChainHodge::applyG, py::arg("k"), py::arg("c"))
      .def("applyMinv", &CovariantChainHodge::applyMinv, py::arg("k"), py::arg("c"))
      .def("applyH", &CovariantChainHodge::applyH, py::arg("k"), py::arg("c"))
      .def("covariantOperator", &CovariantChainHodge::covariantOperator, py::arg("k"))
      .def("dressedDerivative", [](const CovariantChainHodge &self, int k, std::size_t e) {
             return Eigen::MatrixXcd(self.dressedDerivative(k, e)); }, py::arg("k"), py::arg("edge_index"),
           "dM_k^U/ds_e, dense.")
      .def("dressedPhaseDerivative", [](const CovariantChainHodge &self, int k, std::size_t e) {
             return Eigen::MatrixXcd(self.dressedPhaseDerivative(k, e)); }, py::arg("k"), py::arg("edge_index"))
      .def("covariantOperatorDerivative", &CovariantChainHodge::covariantOperatorDerivative,
           py::arg("k"), py::arg("edge_index"), "dh_k/ds_e for the canonical edge index, dense.")
      .def("covariantOperatorDirectionalDerivative",
           [](const CovariantChainHodge &self, int k, const std::vector<std::complex<double>> &direction) {
             return self.lengthDirection(k, direction).operatorDirectional;
           },
           py::arg("k"), py::arg("direction"),
           "D_v h_k along a squared-length direction v (one entry per edge, canonical order), dense.")
      .def("covariantOperatorSecondDerivative",
           [](const CovariantChainHodge &self, int k, std::size_t e,
              const std::vector<std::complex<double>> &direction) {
             return self.covariantOperatorSecondDerivative(self.lengthDirection(k, direction), e);
           },
           py::arg("k"), py::arg("edge_index"), py::arg("direction"),
           "D_v dh_k/ds_e for the canonical edge index along a squared-length direction v, dense.")
      .def("covariantOperatorPhaseDerivative", &CovariantChainHodge::covariantOperatorPhaseDerivative,
           py::arg("k"), py::arg("edge_index"),
           "dh_k/dphi_e for the multiplicative link variation U_e = e^{i phi_e}, dense.")
      .def("covariantOperatorPhaseHessian", &CovariantChainHodge::covariantOperatorPhaseHessian,
           py::arg("k"), py::arg("edge_a"), py::arg("edge_b"),
           "d^2 h_k / dphi_a dphi_b for the multiplicative link variations U_e = e^{i phi_e} at two "
           "canonical edge indices, dense.")
      .def("sparsePencil", &CovariantChainHodge::sparsePencil, py::arg("k") = 0,
           "The sparse degree-zero pencil (d_1^U M_1^U (d_1^{U^-1})^T, M_0^U), available at any size.")
      .def("dressedVertexPotential", &CovariantChainHodge::dressedVertexPotential,
           py::arg("potential"),
           "M_0^U[V]: the potential-weighted mass matrix dressed by the connection like M_0.")
      .def("sparsePencilPhaseDerivativeAlong", &CovariantChainHodge::sparsePencilPhaseDerivativeAlong,
           py::arg("edge_weights"),
           "The derivative of sparsePencil(0) along phi_e -> phi_e + t w_e (one weight per edge): the "
           "current operator of a uniform connection when w_e = q . dx_e.")
      .def("dressedVertexPotentialPhaseDerivativeAlong",
           &CovariantChainHodge::dressedVertexPotentialPhaseDerivativeAlong, py::arg("potential"),
           py::arg("edge_weights"), "The same derivative of dressedVertexPotential(potential).")
      .def("pencil", &CovariantChainHodge::pencil, py::arg("k"))
      .def("pencilAux", &CovariantChainHodge::pencilAux, py::arg("k"))
      .def("spectrum", &CovariantChainHodge::spectrum, py::arg("k"))
      .def("dual", &CovariantChainHodge::dual)
      .def("gauged", &CovariantChainHodge::gauged, py::arg("g"))
      .def("verify", &CovariantChainHodge::verify, py::arg("k") = 1)
      .def("regimeCertificate", &CovariantChainHodge::regimeCertificate, py::arg("k"),
           py::arg("tolerance") = 1e-10, "The measured metric regime of the pencil at degree k.")
      .def("resolvent", &CovariantChainHodge::resolvent, py::arg("k"), py::arg("zeta"), py::arg("c"),
           "(zeta I - h_k)^{-1} c = M^U (zeta M^U - A~^U)^{-1} c by one sparse bordered factorization.")
      .def("harmonicChains", &CovariantChainHodge::harmonicChains, py::arg("k"),
           py::arg("kappa") = 10.0, py::arg("force_sparse") = false,
           "H_k = M_k^U ker S^U with S^U = [(d_{k+1}^{U^-1})^T; d_k^U M_k^U]: the "
           "harmonic chains read as a null space, dense SVD below the crossover and sparse "
           "rank-revealing QR at or above it.")
      .def("harmonicBand", &CovariantChainHodge::harmonicBand, py::arg("k"), py::arg("kappa") = 10.0,
           py::arg("isotropy_tolerance") = 1e-10, py::arg("force_sparse") = false,
           "The lambda = 0 band from harmonicChains, carrying everything band carries. Under the "
           "rank conditions (R1)-(R4) this is the same subspace the contour reading returns; a "
           "nullity that differs between U and U^-1 is refused by name.")
      .def("band", &CovariantChainHodge::band, py::arg("k"), py::arg("contour"), py::arg("kappa") = 10.0,
           py::arg("isotropy_tolerance") = 1e-10,
           "The Riesz band of the contour: P, Phi, Phi^vee, Z, B_C, Phi~, J, Gamma, certificates.")
      .def("applyPencilOperator", &CovariantChainHodge::applyPencilOperator, py::arg("k"), py::arg("Z"),
           "A~_k^U Z by sparse products and one sparse factorization of M_{k-1}^U: the production "
           "path's pencil operator, which never forms the dense A~_k^U and is defined at any size.")
      .def("borderedSystem",
           [](const CovariantChainHodge &self, int k, std::complex<double> zeta) {
             return self.borderedSystem(k, zeta); },
           py::arg("k"), py::arg("zeta"),
           "The sparse bordered system of the shifted pencil, whose Schur complement is "
           "zeta M_k^U - A~_k^U.")
      .def("shiftedSolve",
           [](const CovariantChainHodge &self, int k, std::complex<double> zeta,
              const Eigen::MatrixXcd &B) {
             SparseCostReport report;
             Eigen::MatrixXcd X = self.shiftedSolve(k, zeta, B, &report);
             return std::make_pair(std::move(X), report); },
           py::arg("k"), py::arg("zeta"), py::arg("B"),
           "((zeta M_k^U - A~_k^U)^{-1} B, cost): the production path's shifted solve through one "
           "sparse LU of the bordered system, with the factorization's cost report.")
      .def("sparseBand",
           [](const CovariantChainHodge &self, int k, const Contour &contour, int probeCount,
              double kappa, double isotropyTolerance, std::uint64_t seed) {
             SparseCostReport report;
             Band band = self.sparseBand(k, contour, probeCount, kappa, isotropyTolerance, seed,
                                         &report);
             return std::make_pair(std::move(band), report); },
           py::arg("k"), py::arg("contour"), py::arg("probe_count"), py::arg("kappa") = 10.0,
           py::arg("isotropy_tolerance") = 1e-10, py::arg("seed") = std::uint64_t{20260922},
           "(band, cost): the Riesz band of the contour read on the sparse production path, the "
           "quadrature applied to a probe block so that the n x n projector is never formed.")
      .def_static("leftFrame", &CovariantChainHodge::leftFrame, py::arg("band"), py::arg("dual_instance"),
           py::arg("isotropy_tolerance") = 1e-10,
           "G^{U^-1} Phi^vee B_C^{-T} from the band's dual frame and pairing; raises on an isotropic band.");
  py::class_<SparsePencil>(m, "SparsePencil",
      "A pencil A - lambda M with both matrices sparse (scipy CSC).")
      .def(py::init([](int degree, const SparseMatrix &A, const SparseMatrix &M) {
             SparsePencil p;
             p.degree = degree;
             p.A = A;
             p.M = M;
             return p;
           }),
           py::arg("degree"), py::arg("A"), py::arg("M"))
      .def_readonly("degree", &SparsePencil::degree)
      .def_readonly("A", &SparsePencil::A)
      .def_readonly("M", &SparsePencil::M);

  py::class_<SparsePencilComposition>(m, "SparsePencilComposition",
      R"doc(Block assembly of two sparse pencils, acting on both matrices at once: the sparse
counterpart of OccupationSpectra.directSum and hoppingBlock. A coupling enters the
left-hand matrix only and is written in the pencil's variable, so a constant coupling
Delta between two identical sheets is C = Delta * M.)doc")
      .def_static("directSum", &SparsePencilComposition::directSum, py::arg("a"), py::arg("b"),
           "(A_a + A_b, M_a + M_b) as block-diagonal direct sums, sheet a first.")
      .def_static("hoppingBlock", &SparsePencilComposition::hoppingBlock, py::arg("a"), py::arg("b"),
           py::arg("coupling"), py::arg("coupling_reverse") = SparseMatrix(),
           "A = [[A_a, C], [C', A_b]] with M block diagonal; an empty reverse block selects "
           "C' = C^dagger.");

  py::class_<SparsePencilRead>(m, "SparsePencilRead",
      "The lowest eigenpairs of a sparse Hermitian pencil with their certificate and the "
      "solver's diagnostics.")
      .def_readonly("eigenvalues", &SparsePencilRead::eigenvalues)
      .def_readonly("vectors", &SparsePencilRead::vectors)
      .def_readonly("residuals", &SparsePencilRead::residuals)
      .def_readonly("orthonormalityDefect", &SparsePencilRead::orthonormalityDefect)
      .def_readonly("hermitianDefectA", &SparsePencilRead::hermitianDefectA)
      .def_readonly("hermitianDefectM", &SparsePencilRead::hermitianDefectM)
      .def_readonly("shiftBelowSpectrum", &SparsePencilRead::shiftBelowSpectrum)
      .def_readonly("converged", &SparsePencilRead::converged)
      .def_readonly("blockSize", &SparsePencilRead::blockSize)
      .def_readonly("iterations", &SparsePencilRead::iterations)
      .def_readonly("restarts", &SparsePencilRead::restarts)
      .def_readonly("solves", &SparsePencilRead::solves);

  py::class_<EffectiveBettiRead>(m, "EffectiveBettiRead",
      "The effective Betti number of a sparse pencil at a scale: the number of eigenvalues in the "
      "window, the eigenvalues that bracket its edge, their gap, and the eigenpairs it was read from.")
      .def_readonly("rank", &EffectiveBettiRead::rank)
      .def_readonly("epsilon", &EffectiveBettiRead::epsilon)
      .def_readonly("lastInside", &EffectiveBettiRead::lastInside)
      .def_readonly("firstOutside", &EffectiveBettiRead::firstOutside)
      .def_readonly("gap", &EffectiveBettiRead::gap)
      .def_readonly("complete", &EffectiveBettiRead::complete)
      .def_readonly("certified", &EffectiveBettiRead::certified)
      .def_readonly("band", &EffectiveBettiRead::band);

  py::class_<SparsePencilSolver>(m, "SparsePencilSolver",
      R"doc(The lowest eigenpairs of a sparse Hermitian positive-definite pencil A z = lambda M z by
block shift-invert Lanczos in the M inner product, with one sparse factorization of
A - sigma M (Cholesky when positive definite, which certifies that sigma lies below the
spectrum; LU otherwise). Hermiticity and the positivity of M are measured and refused by
name when they fail.

Reference: Ericsson & Ruhe, Mathematics of Computation 35, 1980.)doc")
      .def_static("lowest",
           [](const SparseMatrix &A, const SparseMatrix &M, int count, double sigma, double tolerance,
              int blockSize, int maxBasisSize, int maxIterations, double hermitianTolerance,
              std::uint64_t seed) {
             SparsePencilOptions options;
             options.tolerance = tolerance;
             options.blockSize = blockSize;
             options.maxBasisSize = maxBasisSize;
             options.maxIterations = maxIterations;
             options.hermitianTolerance = hermitianTolerance;
             options.seed = seed;
             return SparsePencilSolver::lowest(A, M, count, sigma, options);
           },
           py::arg("A"), py::arg("M"), py::arg("count"), py::arg("sigma"),
           py::arg("tolerance") = SparsePencilOptions{}.tolerance,
           py::arg("block_size") = 0, py::arg("max_basis_size") = 0,
           py::arg("max_iterations") = SparsePencilOptions{}.maxIterations,
           py::arg("hermitian_tolerance") = SparsePencilOptions{}.hermitianTolerance,
           py::arg("seed") = SparsePencilOptions{}.seed,
           "The count lowest eigenpairs above the shift sigma. The certificate's residual is "
           "max_i |A z_i - lambda_i M z_i| / |(A - sigma M) z_i| and its conditioning the 1-norm "
           "condition estimate of A - sigma M. Zero block_size / max_basis_size select the "
           "automatic values.")
      .def_static("lowestWithLowRank",
           [](const SparseMatrix &A, const SparseMatrix &M, const Eigen::MatrixXcd &left,
              const Eigen::MatrixXcd &core, int count, double sigma, double tolerance, int blockSize,
              int maxBasisSize, int maxIterations, std::uint64_t seed) {
             SparsePencilOptions options;
             options.tolerance = tolerance;
             options.blockSize = blockSize;
             options.maxBasisSize = maxBasisSize;
             options.maxIterations = maxIterations;
             options.seed = seed;
             return SparsePencilSolver::lowest(A, M, LowRankTerm{left, core}, count, sigma, options);
           },
           py::arg("A"), py::arg("M"), py::arg("left"), py::arg("core"), py::arg("count"), py::arg("sigma"),
           py::arg("tolerance") = SparsePencilOptions{}.tolerance, py::arg("block_size") = 0,
           py::arg("max_basis_size") = 0, py::arg("max_iterations") = SparsePencilOptions{}.maxIterations,
           py::arg("seed") = SparsePencilOptions{}.seed,
           "lowest for the pencil (A + P D P^dagger, M) with the low-rank term in factored form "
           "(left = P, n x r; core = D, r x r Hermitian): Woodbury inside the shift-invert solve, and "
           "the shift certified below the spectrum by inertia.")
      .def_static("effectiveBetti",
           [](const SparseMatrix &A, const SparseMatrix &M, double epsilon, double sigma,
              double tolerance, int initialCount, std::uint64_t seed) {
             SparsePencilOptions options;
             options.tolerance = tolerance;
             options.seed = seed;
             return SparsePencilSolver::effectiveBetti(A, M, epsilon, sigma, options, initialCount);
           },
           py::arg("A"), py::arg("M"), py::arg("epsilon"), py::arg("sigma"),
           py::arg("tolerance") = SparsePencilOptions{}.tolerance, py::arg("initial_count") = 8,
           py::arg("seed") = SparsePencilOptions{}.seed,
           "The effective Betti number at scale epsilon: the rank of the pencil's spectral band in "
           "[0, epsilon], a property of the declared operator and not of the incidence of the "
           "complex. Certified by the first eigenvalue found outside the window and the gap it leaves.");

  py::class_<FaceBlock>(m, "FaceBlock",
      "One triangle's 3x3 face block over its three edges (canonical edge indices in "
      "local order (v0v1),(v0v2),(v1v2)), its numerical rank, and the preset.")
      .def_readonly("faceIndex", &FaceBlock::faceIndex)
      .def_readonly("edgeIndices", &FaceBlock::edgeIndices)
      .def_readonly("block", &FaceBlock::block)
      .def_readonly("rank", &FaceBlock::rank)
      .def_readonly("preset", &FaceBlock::preset);

  py::class_<TetrahedronBlock>(m, "TetrahedronBlock",
      "One tetrahedron's 4x4 degree-0 block over its four vertices (canonical vertex indices in "
      "ascending order), its numerical rank, and the preset.")
      .def_readonly("tetrahedronIndex", &TetrahedronBlock::tetrahedronIndex)
      .def_readonly("vertexIndices", &TetrahedronBlock::vertexIndices)
      .def_readonly("block", &TetrahedronBlock::block)
      .def_readonly("rank", &TetrahedronBlock::rank)
      .def_readonly("preset", &TetrahedronBlock::preset);

  py::class_<BandDerivative::ResolventFrames>(m, "ResolventFrames")
      .def_readonly("degree", &BandDerivative::ResolventFrames::degree)
      .def_readonly("frame", &BandDerivative::ResolventFrames::frame);
  py::class_<BandDerivative>(m, "BandDerivative",
      "Analytic derivatives of a Riesz band's geometric images: dZ = G^U(-dM^U Z + dP Phi) with "
      "dP Phi = sum_j w_j R(zeta_j) dh R(zeta_j) Phi; and d(A~^U) = dM^U h + M^U dh.")
      .def_static("resolventFrames", &BandDerivative::resolventFrames, py::arg("covariant"), py::arg("k"),
           py::arg("contour"), py::arg("frame"))
      .def_static("imagesLengthDerivative", &BandDerivative::imagesLengthDerivative, py::arg("covariant"),
           py::arg("frames"), py::arg("images"), py::arg("edge_index"))
      .def_static("imagesPhaseDerivative", &BandDerivative::imagesPhaseDerivative, py::arg("covariant"),
           py::arg("frames"), py::arg("images"), py::arg("edge_index"))
      .def_static("pencilOperatorLengthDerivative", &BandDerivative::pencilOperatorLengthDerivative,
           py::arg("covariant"), py::arg("k"), py::arg("edge_index"))
      .def_static("pencilOperatorPhaseDerivative", &BandDerivative::pencilOperatorPhaseDerivative,
           py::arg("covariant"), py::arg("k"), py::arg("edge_index"));
  py::class_<FaceAnchor>(m, "FaceAnchor",
      R"doc(The face anchor with the Whitney metric: the per-triangle Whitney block
M_1^{(t)} (rank 3), its connection dressing by U_{b(e)b(e')},
the face endomorphism Pi_tau(U) = G_1^U M_1^{(tau)U} G_1^U applied by solves, and the
invariant anchor coordinate alpha_tau = det((Z^vee)^T M_1^{(tau)U} Z) of a fiber paired
through its geometric images. The Grassmann per-face blade block has rank 2 and makes
alpha_tau vanish identically. Transpose pairing throughout.)doc")
      .def_static("whitneyFaceBlock", &FaceAnchor::whitneyFaceBlock, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("face_index"), py::arg("branch") = Branch::Continuation)
      .def_static("whitneyFaceBlocks", &FaceAnchor::whitneyFaceBlocks, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("branch") = Branch::Continuation)
      .def_static("grassmannFaceBlock", &FaceAnchor::grassmannFaceBlock, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("face_index"))
      .def_static("faceBlock", &FaceAnchor::faceBlock, py::arg("hodge"), py::arg("face_index"),
           "The face block of the instance's own preset.")
      .def_static("dressedFaceBlock", &FaceAnchor::dressedFaceBlock, py::arg("block"), py::arg("complex"),
           py::arg("connection"), "M_1^{(tau)U}: the block dressed entrywise by U_{b(e)b(e')}.")
      .def_static("applyFaceEndomorphism", &FaceAnchor::applyFaceEndomorphism, py::arg("covariant"),
           py::arg("face_index"), py::arg("c"), "Pi_tau(U) c by solves.")
      .def_static("anchorCoordinate", &FaceAnchor::anchorCoordinate, py::arg("covariant"),
           py::arg("face_index"), py::arg("Z_dual"), py::arg("Z"),
           "alpha_tau = det((Z^vee)^T M_1^{(tau)U} Z) from the fiber's images.")
      .def_static("anchorCoordinateFromChains", &FaceAnchor::anchorCoordinateFromChains,
           py::arg("covariant"), py::arg("face_index"), py::arg("Phi_dual"), py::arg("Phi"),
           "alpha_tau = det((Phi^vee)^T Pi_tau(U) Phi) from the chain frames (images by solves).")
      .def_static("anchorCoordinates", &FaceAnchor::anchorCoordinates, py::arg("covariant"),
           py::arg("Z_dual"), py::arg("Z"), "alpha_tau for every triangle.")
      .def_static("numericalRank", &FaceAnchor::numericalRank, py::arg("A"), py::arg("kappa") = 10.0)
      // ---- tetrahedral anchor at degree 0 ----
      .def_static("whitneyTetrahedronBlock", &FaceAnchor::whitneyTetrahedronBlock, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("tetrahedron_index"),
           py::arg("branch") = Branch::Continuation,
           "M_0^{(T)}: the per-tetrahedron Whitney block over its four vertices (rank 4).")
      .def_static("whitneyTetrahedronBlocks", &FaceAnchor::whitneyTetrahedronBlocks, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("branch") = Branch::Continuation)
      .def_static("tetrahedronBlock", &FaceAnchor::tetrahedronBlock, py::arg("hodge"),
           py::arg("tetrahedron_index"),
           "The tetrahedron block of the instance's preset; refuses the Grassmann preset by name.")
      .def_static("dressedTetrahedronBlock", &FaceAnchor::dressedTetrahedronBlock, py::arg("block"),
           py::arg("complex"), py::arg("connection"), "M_0^{(T)U}: the block dressed entrywise by U_{vw}.")
      .def_static("applyTetrahedronEndomorphism", &FaceAnchor::applyTetrahedronEndomorphism,
           py::arg("covariant"), py::arg("tetrahedron_index"), py::arg("c"),
           "Pi_T(U) c = G_0^U M_0^{(T)U} G_0^U c by solves.")
      .def_static("tetrahedronAnchorCoordinate", &FaceAnchor::tetrahedronAnchorCoordinate,
           py::arg("covariant"), py::arg("tetrahedron_index"), py::arg("Z_dual"), py::arg("Z"),
           "alpha_T = det((Z^vee)^T M_0^{(T)U} Z) from the fiber's degree-0 images.")
      .def_static("tetrahedronAnchorCoordinateFromChains", &FaceAnchor::tetrahedronAnchorCoordinateFromChains,
           py::arg("covariant"), py::arg("tetrahedron_index"), py::arg("Phi_dual"), py::arg("Phi"))
      .def_static("tetrahedronAnchorCoordinates", &FaceAnchor::tetrahedronAnchorCoordinates,
           py::arg("covariant"), py::arg("Z_dual"), py::arg("Z"), "alpha_T for every tetrahedron.")
      .def_static("flatZeroModeOverlap", &FaceAnchor::flatZeroModeOverlap, py::arg("covariant"),
           py::arg("Z_dual"), py::arg("Z"),
           "Certificate: |(P z_0)^T M_0^U (P z_0)| / |z_0^T M_0^U z_0| for the Riesz projection P of the "
           "constant image z_0 (the vertex-volume chain, the degree-0 flat zero mode) onto the band: 1 on "
           "the trivial-holonomy harmonic band, 0 on a band of other eigenvalues.");
  py::class_<FeshbachResult>(m, "FeshbachResult",
      "One Feshbach complement F_B(lambda) = P_BB - P_BI P_II^{-1} P_IB of a symmetric pencil "
      "P = A - lambda M, with det P = det P_II det F_B and the constraint modes T = [I_B; -P_II^{-1} P_IB]. "
      "At an interior resonance the inverse is the Drazin inverse P_II^D = (P_II + Pi0)^{-1} (I - Pi0) "
      "built on the Riesz projector Pi0 onto the generalized eigenspace of the eigenvalues inside the "
      "resonance disc; nullProjector is Pi0 and rangeProjector is I - Pi0, both oblique.")
      .def_readonly("lambda_", &FeshbachResult::lambda)
      .def_readonly("interface", &FeshbachResult::interface)
      .def_readonly("interior", &FeshbachResult::interior)
      .def_readonly("response", &FeshbachResult::response)
      .def_readonly("constraintModes", &FeshbachResult::constraintModes)
      .def_readonly("interiorDeterminant", &FeshbachResult::interiorDeterminant)
      .def_readonly("responseDeterminant", &FeshbachResult::responseDeterminant)
      .def_readonly("pencilDeterminant", &FeshbachResult::pencilDeterminant)
      .def_readonly("determinantResidual", &FeshbachResult::determinantResidual)
      .def_readonly("pencilLogDeterminant", &FeshbachResult::pencilLogDeterminant)
      .def_readonly("interiorLogDeterminant", &FeshbachResult::interiorLogDeterminant)
      .def_readonly("responseLogDeterminant", &FeshbachResult::responseLogDeterminant)
      .def_readonly("logModulusResidual", &FeshbachResult::logModulusResidual)
      .def_readonly("logPhaseResidual", &FeshbachResult::logPhaseResidual)
      .def_readonly("solveResidual", &FeshbachResult::solveResidual)
      .def_readonly("interiorSingular", &FeshbachResult::interiorSingular)
      .def_readonly("resonanceRadius", &FeshbachResult::resonanceRadius)
      .def_readonly("resonanceEnclosure", &FeshbachResult::resonanceEnclosure)
      .def_readonly("resonanceSeparation", &FeshbachResult::resonanceSeparation)
      .def_readonly("interiorRank", &FeshbachResult::interiorRank)
      .def_readonly("interiorInverse", &FeshbachResult::interiorInverse)
      .def_readonly("resonantSpace", &FeshbachResult::resonantSpace)
      .def_readonly("resonantLeftSpace", &FeshbachResult::resonantLeftSpace)
      .def_readonly("resonantModes", &FeshbachResult::resonantModes)
      .def_readonly("rangeProjector", &FeshbachResult::rangeProjector)
      .def_readonly("nullProjector", &FeshbachResult::nullProjector)
      .def_readonly("projectorIdempotency", &FeshbachResult::projectorIdempotency)
      .def_readonly("projectorTrace", &FeshbachResult::projectorTrace)
      .def_readonly("compatibilityResidual", &FeshbachResult::compatibilityResidual)
      .def_readonly("compatible", &FeshbachResult::compatible)
      .def_readonly("independenceResidual", &FeshbachResult::independenceResidual)
      .def_readonly("responseIndependent", &FeshbachResult::responseIndependent)
      .def_readonly("resonantResponse", &FeshbachResult::resonantResponse)
      .def_readonly("reductionResidual", &FeshbachResult::reductionResidual)
      .def_readonly("liftResidual", &FeshbachResult::liftResidual);

  py::class_<CongruenceResult>(m, "CongruenceResult",
      "A congruence (T^T A T, T^T M T) with the symmetry defects of the reduced pair and the "
      "inverse condition number of the basis.")
      .def_readonly("A", &CongruenceResult::A)
      .def_readonly("M", &CongruenceResult::M)
      .def_readonly("symmetryDefect", &CongruenceResult::symmetryDefect)
      .def_readonly("metricSymmetryDefect", &CongruenceResult::metricSymmetryDefect)
      .def_readonly("basisConditionInverse", &CongruenceResult::basisConditionInverse);

  py::class_<SurrogateResult>(m, "SurrogateResult",
      "A certified Craig-Bampton/AMLS surrogate of a pencil over a window disc in the complex "
      "spectral plane, with every claimed eigenvalue held to the exact Feshbach map: "
      "feshbachDefects is ||F_B(theta) x_B|| / (||F_B|| ||x_B||) and feshbachBounds is "
      "(1 + ||P_BI P_II^{-1}||) ||P(theta) x|| / (||F_B|| ||x_B||), the bound the defect is "
      "measured against pair by pair.")
      .def_readonly("interface", &SurrogateResult::interface)
      .def_readonly("interior", &SurrogateResult::interior)
      .def_readonly("shift", &SurrogateResult::shift)
      .def_readonly("windowCentre", &SurrogateResult::windowCentre)
      .def_readonly("windowRadius", &SurrogateResult::windowRadius)
      .def_readonly("retentionRadius", &SurrogateResult::retentionRadius)
      .def_readonly("basis", &SurrogateResult::basis)
      .def_readonly("reduced", &SurrogateResult::reduced)
      .def_readonly("interiorEigenvalues", &SurrogateResult::interiorEigenvalues)
      .def_readonly("retainedModes", &SurrogateResult::retainedModes)
      .def_readonly("discardedModeSeparation", &SurrogateResult::discardedModeSeparation)
      .def_readonly("eigenvalues", &SurrogateResult::eigenvalues)
      .def_readonly("vectors", &SurrogateResult::vectors)
      .def_readonly("windowIndices", &SurrogateResult::windowIndices)
      .def_readonly("residuals", &SurrogateResult::residuals)
      .def_readonly("feshbachDefects", &SurrogateResult::feshbachDefects)
      .def_readonly("feshbachBounds", &SurrogateResult::feshbachBounds)
      .def_readonly("feshbachHolds", &SurrogateResult::feshbachHolds)
      .def_readonly("resonantAtEigenvalue", &SurrogateResult::resonantAtEigenvalue)
      .def_readonly("tolerance", &SurrogateResult::tolerance)
      .def_readonly("certified", &SurrogateResult::certified)
      .def_readonly("refusal", &SurrogateResult::refusal);

  py::class_<FiberRestriction>(m, "FiberRestriction",
      "The coarse pencil and chain metric restricted to retained fibers: (Z^T A~ Z, Z^T M Z).")
      .def_readonly("A", &FiberRestriction::A)
      .def_readonly("gram", &FiberRestriction::gram)
      .def_readonly("blockOffsets", &FiberRestriction::blockOffsets)
      .def_readonly("blockRanks", &FiberRestriction::blockRanks);

  py::class_<TransferResult>(m, "TransferResult",
      "A transfer T_AB(U) = (Z_A^vee)^T (A~^U)_AB Z_B with its reversal certificate "
      "T_BA(U^-1) = T_AB(U)^T and the groupoid hypothesis T_BA = T_AB^-1.")
      .def_readonly("forward", &TransferResult::forward)
      .def_readonly("reverse", &TransferResult::reverse)
      .def_readonly("reversalResidual", &TransferResult::reversalResidual)
      .def_readonly("tolerance", &TransferResult::tolerance)
      .def_readonly("groupoidHolds", &TransferResult::groupoidHolds)
      .def_readonly("groupoidResidual", &TransferResult::groupoidResidual)
      .def_readonly("dualTransfer", &TransferResult::dualTransfer);

  py::class_<PencilSchur>(m, "PencilSchur",
      R"doc(The recursion on the symmetric pencil P(lambda) = A~ - lambda M on geometric
images: the Feshbach complement with its determinant factorization, the Craig-Bampton
congruence, the restriction of pencil and chain metric to retained fibers, and the
transfer between fibers with the reversal identity asserted at runtime. Every pairing
is the transpose.)doc")
      .def_static("logDeterminant", &PencilSchur::logDeterminant, py::arg("A"),
           "log det A = log|det A| + i arg det A from a partial-pivoting LU, arg in (-pi, pi].")
      .def_static("feshbach", &PencilSchur::feshbach, py::arg("A"), py::arg("M"), py::arg("lambda_"),
           py::arg("interface"), py::arg("rank_tolerance") = 1e-12, py::arg("resonance_radius") = 1e-12,
           "The Feshbach complement at lambda with the interior block's Riesz projectors and, at an "
           "interior resonance -- an eigenvalue of P_II inside the disc about zero of radius "
           "resonance_radius times its spectral radius -- the Drazin inverse, the compatibility and "
           "independence residuals, the retained resonant modes (a basis of the generalized "
           "eigenspace), the resonant reduction and the two residuals that certify it. "
           "rank_tolerance governs only the rank of the reduction's null space and the two "
           "verdict thresholds.")
      .def_static("sparseFeshbach",
           [](const SparseMatrix &A, const SparseMatrix &M, std::complex<double> lambda,
              const std::vector<int> &interface, double solveTolerance) {
             SparseCostReport report;
             FeshbachResult result =
                 PencilSchur::sparseFeshbach(A, M, lambda, interface, solveTolerance, &report);
             return std::make_pair(std::move(result), report); },
           py::arg("A"), py::arg("M"), py::arg("lambda_"), py::arg("interface"),
           py::arg("solve_tolerance") = 1e-8,
           "(result, cost): the same complement on the sparse production path, the interior block "
           "factorized by sparse LU and no n x n matrix formed. An interior resonance is refused "
           "by name; the dense feshbach resolves it.")
      .def_static("craigBampton",
           py::overload_cast<const Eigen::MatrixXcd &, const Eigen::MatrixXcd &,
                             const Eigen::MatrixXcd &>(&PencilSchur::craigBampton),
           py::arg("A"), py::arg("M"), py::arg("T"),
           "The congruence (T^T A T, T^T M T) of an explicit basis, with the symmetry defects of "
           "the reduced pair and the inverse condition number of the basis.")
      .def_static("craigBamptonSurrogate",
           py::overload_cast<const Eigen::MatrixXcd &, const Eigen::MatrixXcd &,
                             const std::vector<int> &, std::complex<double>, double, double,
                             std::complex<double>, double, double>(&PencilSchur::craigBampton),
           py::arg("A"), py::arg("M"), py::arg("interface"), py::arg("window_centre"),
           py::arg("window_radius"), py::arg("retention_radius"),
           py::arg("shift") = std::complex<double>(0.0, 0.0), py::arg("tolerance") = 1e-8,
           py::arg("rank_tolerance") = 1e-12,
           "The certified Craig-Bampton/AMLS surrogate over the window disc |theta - centre| <= "
           "radius: interface constraint modes at the shift plus the fixed-interface modes of "
           "(A_II, M_II) inside the retention radius, with every claimed eigenvalue held to the "
           "exact Feshbach map. It runs in every pencil regime.")
      .def_static("restrictToFibers",
           py::overload_cast<const Eigen::MatrixXcd &, const Eigen::MatrixXcd &, const Eigen::MatrixXcd &>(
               &PencilSchur::restrictToFibers),
           py::arg("A"), py::arg("M"), py::arg("Z"))
      .def_static("restrictToFiberBlocks",
           py::overload_cast<const Eigen::MatrixXcd &, const Eigen::MatrixXcd &,
                             const std::vector<Eigen::MatrixXcd> &>(&PencilSchur::restrictToFibers),
           py::arg("A"), py::arg("M"), py::arg("fibers"))
      .def_static("gramBlock", &PencilSchur::gramBlock, py::arg("M"), py::arg("ZA"), py::arg("ZB"))
      .def_static("supportsShareTopSimplex", &PencilSchur::supportsShareTopSimplex,
           py::arg("complex"), py::arg("k"), py::arg("support_a"), py::arg("support_b"))
      .def_static("support", &PencilSchur::support, py::arg("Z"), py::arg("threshold") = 1e-12)
      .def_static("transfer", &PencilSchur::transfer, py::arg("AtildeU"), py::arg("AtildeUinv"),
           py::arg("ZA"), py::arg("ZAdual"), py::arg("ZB"), py::arg("ZBdual"),
           py::arg("tolerance") = 1e-8);

  // ---- the anchor by the dressed coordinate ----

  py::class_<DeclaredPaths>(m, "DeclaredPaths",
      R"doc(The declared path rule of an anchor: one base vertex p and one walk in the
1-skeleton from p to each vertex the rule reaches. The transport of a coefficient at v
to p is the ordered product of the links along the walk, which telescopes under a vertex
gauge to g_p^{-1} T_p(v) g_v. The same base vertex and the same rule are used for every
face of the atlas.)doc")
      .def_static("breadthFirst", &DeclaredPaths::breadthFirst, py::arg("complex"),
           py::arg("base_point"), py::arg("support") = std::vector<std::uint64_t>{},
           "Shortest walks from the base vertex, ties broken by ascending vertex id; an "
           "empty support means the whole complex.")
      .def_static("fromWalks", &DeclaredPaths::fromWalks, py::arg("complex"),
           py::arg("base_point"), py::arg("walks"),
           "A caller-declared rule: one vertex sequence per reached vertex, validated "
           "against the complex's edges.")
      .def_static("declaredTransports", &DeclaredPaths::declaredTransports, py::arg("base_point"),
           py::arg("transports"),
           "A transport table declared as bare numbers with no walk behind it. It does not "
           "transform under a vertex gauge, so the anchor refuses it; it exists so that the "
           "refusal is reachable.")
      .def("basePoint", &DeclaredPaths::basePoint)
      .def("derivedFromWalks", &DeclaredPaths::derivedFromWalks)
      .def("reaches", &DeclaredPaths::reaches, py::arg("v"))
      .def("walks", &DeclaredPaths::walks)
      .def("transport", &DeclaredPaths::transport, py::arg("connection"), py::arg("v"),
           "T_p(v), the transport of a coefficient at v to the base vertex.");

  py::class_<FaceRestriction>(m, "FaceRestriction",
      "One oriented triangle's dressed restriction: its three ordered boundary edges in the "
      "cyclic order (v0v1), (v1v2), (v0v2), their incidence signs (+1, +1, -1), their base "
      "vertices b(e) = min e, the transports of those base vertices to p, and the three "
      "nonzero entries of res_{tau->p}(U).")
      .def_readonly("faceIndex", &FaceRestriction::faceIndex)
      .def_readonly("edgeIndices", &FaceRestriction::edgeIndices)
      .def_readonly("incidenceSigns", &FaceRestriction::incidenceSigns)
      .def_readonly("basePoints", &FaceRestriction::basePoints)
      .def_readonly("transports", &FaceRestriction::transports)
      .def_readonly("factors", &FaceRestriction::factors);

  py::class_<DressedAnchorRead>(m, "DressedAnchorRead",
      "The anchor certificate of one base band on one atlas of faces: the exterior-power "
      "coordinates of every face, the invariant coordinates alpha_tau when attached, the "
      "covariance and transition-cocycle residuals, and the named refusals.")
      .def_readonly("basePoint", &DressedAnchorRead::basePoint)
      .def_readonly("bandRank", &DressedAnchorRead::bandRank)
      .def_readonly("faceIndices", &DressedAnchorRead::faceIndices)
      .def_readonly("coordinates", &DressedAnchorRead::coordinates)
      .def_readonly("invariantCoordinates", &DressedAnchorRead::invariantCoordinates)
      .def_readonly("anchoringFaces", &DressedAnchorRead::anchoringFaces)
      .def_readonly("coordinateScale", &DressedAnchorRead::coordinateScale)
      .def_readonly("covarianceResidual", &DressedAnchorRead::covarianceResidual)
      .def_readonly("transitionCocycleResidual", &DressedAnchorRead::transitionCocycleResidual)
      .def_readonly("tolerance", &DressedAnchorRead::tolerance)
      .def_readonly("anchored", &DressedAnchorRead::anchored)
      .def_readonly("failedCertificates", &DressedAnchorRead::failedCertificates);

  py::class_<DressedAnchor>(m, "DressedAnchor",
      R"doc(The anchor of a base band to oriented two-simplices by the dressed coordinate:
Delta_tau = det(res_{tau->p}(U) Phi_Q) for a rank-three band, its Lambda^r variant for a
band of rank below three, the profile those coordinates form as a point of a projective
space, and the determinant-line transition functions on overlaps. No modulus, square
root, free face weight or real-valued score enters the physical definition. The anchor
refuses rather than reporting a gauge-dependent raw restriction when the
connection-dressed covariance cannot be verified, and refuses an identically zero
profile, which is what an exact band at flat connection produces.)doc")
      .def_static("faceRestriction", &DressedAnchor::faceRestriction, py::arg("complex"),
           py::arg("connection"), py::arg("paths"), py::arg("face_index"),
           "The triangle's ordered boundary edges, incidence signs, base vertices and transports.")
      .def_static("restriction", &DressedAnchor::restriction, py::arg("complex"),
           py::arg("connection"), py::arg("paths"), py::arg("face_index"),
           "res_{tau->p}(U) as a dense 3 x n_1 matrix, for verifying the covariance law as "
           "written.")
      .def_static("restrictedFrame", &DressedAnchor::restrictedFrame, py::arg("complex"),
           py::arg("connection"), py::arg("paths"), py::arg("face_index"), py::arg("Phi"),
           "res_{tau->p}(U) Phi_Q, a 3 x r matrix, without forming any n_1-wide object.")
      .def_static("twistedCoboundaryBlock", &DressedAnchor::twistedCoboundaryBlock,
           py::arg("complex"), py::arg("connection"), py::arg("face_index"),
           "The restriction to tau's three ordered boundary edges of the twisted coboundary "
           "of a vertex potential; its determinant is F_tau - 1.")
      .def_static("exteriorPower", &DressedAnchor::exteriorPower, py::arg("A"),
           "Lambda^r of a 3 x r matrix: its maximal minors in lexicographic order of the row "
           "subset.")
      .def_static("dressedCoordinates", &DressedAnchor::dressedCoordinates, py::arg("complex"),
           py::arg("connection"), py::arg("paths"), py::arg("face_index"), py::arg("Phi"),
           "The Lambda^r coordinates of one face.")
      .def_static("dressedCoordinate", &DressedAnchor::dressedCoordinate, py::arg("complex"),
           py::arg("connection"), py::arg("paths"), py::arg("face_index"), py::arg("Phi"),
           "Delta_tau for a rank-three band.")
      .def_static("anchorableFaces", &DressedAnchor::anchorableFaces, py::arg("complex"),
           py::arg("paths"), "The triangles every one of whose edge base vertices the rule reaches.")
      .def_static("profile", &DressedAnchor::profile, py::arg("complex"), py::arg("connection"),
           py::arg("paths"), py::arg("face_indices"), py::arg("Phi"),
           py::arg("tolerance") = 1e-9, py::arg("gauge_seed") = 7,
           "The anchor certificate of the band on the atlas.")
      .def_static("withInvariantCoordinates", &DressedAnchor::withInvariantCoordinates,
           py::arg("read"), py::arg("covariant"), py::arg("Z_dual"), py::arg("Z"),
           "The same read with the invariant coordinates alpha_tau of its atlas attached.")
      .def_static("covarianceResidual", &DressedAnchor::covarianceResidual, py::arg("complex"),
           py::arg("connection"), py::arg("paths"), py::arg("face_indices"), py::arg("gauge"),
           "The residual of res(U^g) rho_1(g) = g_p^{-1} res(U) over the atlas.")
      .def_static("verificationGauge", &DressedAnchor::verificationGauge, py::arg("complex"),
           py::arg("seed"), "The deterministic verification gauge of a seed.")
      .def_static("faceTransition", &DressedAnchor::faceTransition, py::arg("read"),
           py::arg("face_slot"), py::arg("other_face_slot"), py::arg("coordinate") = 0,
           "The determinant-line transition on the overlap of two face charts.")
      .def_static("basePointTransition", &DressedAnchor::basePointTransition, py::arg("read"),
           py::arg("other"), py::arg("face_slot"), py::arg("coordinate") = 0,
           "The transition between the charts of two base vertices on one face.")
      .def_static("transitionCocycleResidual", &DressedAnchor::transitionCocycleResidual,
           py::arg("read"), py::arg("coordinate") = 0,
           "The residual of t_ab t_bc = t_ac over the read's anchoring faces.")
      .def_static("projectiveDistance", &DressedAnchor::projectiveDistance, py::arg("a"),
           py::arg("b"),
           "The chordal Fubini-Study distance between two profiles, a reported numerical "
           "stability certificate and never part of a physical statement.");
}
