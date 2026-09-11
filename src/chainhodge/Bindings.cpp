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
#include "chainhodge/RieszBand.h"
#include "chainhodge/LorentzianFamily.h"
#include "chainhodge/PencilSchur.h"
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
      R"doc(The instance certificate of specification §4.2: Kontsevich-Segal allowability
of every top simplex, the minimal margin pi - sum_i |arg lambda_i(g_T)|, the volumes on
the declared branch, the Gram determinants, continuation ambiguity, and the Lorentzian
protocol rotation epsilon (NaN until set).)doc")
      .def_readonly("branch", &InstanceCertificate::branch)
      .def_readonly("allowable", &InstanceCertificate::allowable)
      .def_readonly("margin", &InstanceCertificate::margin)
      .def_readonly("margins", &InstanceCertificate::margins)
      .def_readonly("volumes", &InstanceCertificate::volumes)
      .def_readonly("gramDeterminants", &InstanceCertificate::gramDeterminants)
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
Hodge pencil, assembled per top simplex from the complex squared edge lengths alone
(specification §4.1, §4.2). Reference orientation is ascending vertex id
(ChainComplex.fromTopCells). Sparse results are scipy CSC matrices.)doc")
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
           "The Grassmann projection chain metric G_k = multiplicity o blade pairing (CH §6).")
      .def_static("allowabilityMargin", &WhitneyMass::allowabilityMargin,
           py::arg("complex"), py::arg("squared_lengths"),
           "min over top simplices of pi - sum_i |arg lambda_i(g_T)|.")
      .def_static("certificate", &WhitneyMass::certificate,
           py::arg("complex"), py::arg("squared_lengths"),
           py::arg("branch") = Branch::Continuation,
           "The full instance certificate (§4.2).")
      .def_static("volumeOnBranch",
           [](const Eigen::MatrixXcd &gram, Branch branch) {
             bool ambiguous = false;
             const Complex v = WhitneyMass::volumeOnBranch(gram, branch, &ambiguous);
             return py::make_tuple(v, ambiguous);
           },
           py::arg("gram"), py::arg("branch") = Branch::Continuation,
           "(sqrt(det g)/d!, ambiguous) for one Gram matrix on the declared branch.")
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
      .def_static("derivativeContraction", &WhitneyMass::derivativeContraction,
           py::arg("complex"), py::arg("squared_lengths"), py::arg("k"),
           py::arg("X"), py::arg("Y"), py::arg("branch") = Branch::Continuation,
           "Per-edge tr(X^T (dM_k/ds_e) Y) from the local blocks (transpose pairing).");
  py::enum_<PencilVariable>(m, "PencilVariable",
      "Which vector a pencil eigenproblem A x = lambda B x is written in: geometric "
      "images (Whitney) or chains (Grassmann).")
      .value("GeometricImage", PencilVariable::GeometricImage)
      .value("Chain", PencilVariable::Chain);

  py::class_<Pencil>(m, "Pencil", "A complex symmetric pencil A - lambda B at one degree, dense.")
      .def_readonly("degree", &Pencil::degree)
      .def_readonly("variable", &Pencil::variable)
      .def_readonly("A", &Pencil::A)
      .def_readonly("B", &Pencil::B);

  py::class_<HarmonicRead>(m, "HarmonicRead",
      "Harmonic chains H_k, their geometric images, and the kernel's rank certificate.")
      .def_readonly("degree", &HarmonicRead::degree)
      .def_readonly("chains", &HarmonicRead::chains)
      .def_readonly("images", &HarmonicRead::images)
      .def_readonly("nullity", &HarmonicRead::nullity)
      .def_readonly("rank", &HarmonicRead::rank)
      .def_readonly("tolerance", &HarmonicRead::tolerance)
      .def_readonly("gap", &HarmonicRead::gap)
      .def_readonly("dense", &HarmonicRead::dense);

  py::class_<RankReport>(m, "RankReport",
      "The rank conditions (R1)-(R4) of specification Prop. 4.2 at one degree.")
      .def_readonly("degree", &RankReport::degree)
      .def_property_readonly("measured", [](const RankReport &r) {
        return std::vector<int>(r.measured.begin(), r.measured.end()); })
      .def_property_readonly("expected", [](const RankReport &r) {
        return std::vector<int>(r.expected.begin(), r.expected.end()); })
      .def_property_readonly("holds", [](const RankReport &r) {
        return std::vector<bool>(r.holds.begin(), r.holds.end()); })
      .def_readonly("decompositionHolds", &RankReport::decompositionHolds)
      .def_readonly("kernelIsHarmonic", &RankReport::kernelIsHarmonic)
      .def_readonly("kappa", &RankReport::kappa);

  py::class_<SpectrumRead>(m, "SpectrumRead", "Dense spectrum of one degree's pencil.")
      .def_readonly("degree", &SpectrumRead::degree)
      .def_readonly("eigenvalues", &SpectrumRead::eigenvalues)
      .def_readonly("residual", &SpectrumRead::residual)
      .def_readonly("vectors", &SpectrumRead::vectors);

  py::class_<ChainHodge>(m, "ChainHodge",
      R"doc(The chain-level Hodge pencil of a complexified simplicial complex (specification
§4.3, §9, §13): sparse inverse chain metrics, geometric images by solves, the symmetric
pencil and its auxiliary form, harmonic chains H_k = M_k ker S, rank conditions R1-R4,
exact Betti numbers, and the dense spectrum below the crossover. The adjoint is the
transpose; no conjugation enters any operator.)doc")
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
      .def("hodgeOperator", &ChainHodge::hodgeOperator, py::arg("k"), "The dense L_k on chains.")
      .def("harmonicChains", &ChainHodge::harmonicChains, py::arg("k"), py::arg("kappa") = 10.0,
           py::arg("force_sparse") = false, "H_k = M_k ker S with the kernel's rank certificate.")
      .def("geometricImage", &ChainHodge::geometricImage, py::arg("k"), py::arg("H"), "G_k H.")
      .def("harmonicGram", &ChainHodge::harmonicGram, py::arg("read"), "Phi^T G_k Phi = Z^T M_k Z.")
      .def("rankConditions", &ChainHodge::rankConditions, py::arg("k"), py::arg("kappa") = 10.0,
           "The rank conditions (R1)-(R4) at degree k.")
      .def("betti", &ChainHodge::betti, "Betti numbers over Q, exact.")
      .def("spectrum", &ChainHodge::spectrum, py::arg("k"), "Dense spectrum of the degree-k pencil.");
  py::enum_<CausalType>(m, "CausalType",
      "Declared causal type of an edge (an input, never inferred from a squared length).")
      .value("Spacelike", CausalType::Spacelike)
      .value("Timelike", CausalType::Timelike)
      .value("Null", CausalType::Null);

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
      R"doc(The Lorentzian protocol (specification §10): the family s_e(epsilon) with the
timelike squared lengths rotated by e^{-2 i epsilon} at reported epsilon > 0; reads at
epsilon = 0 exist only inside a family and carry their gap; extrapolation to
epsilon -> 0 is a separate, labeled step.)doc")
      .def_static("rotate", &LorentzianFamily::rotate, py::arg("squared_lengths"),
           py::arg("causal_types"), py::arg("epsilon"),
           "Timelike entries times e^{-2 i epsilon}; others unchanged.")
      .def_static("instance", &LorentzianFamily::instance, py::arg("complex"),
           py::arg("squared_lengths"), py::arg("causal_types"), py::arg("epsilon"),
           py::arg("preset") = Preset::L2, py::arg("branch") = Branch::Continuation,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           "The ChainHodge at epsilon, with epsilon on its certificate.")
      .def_static("sweep", &LorentzianFamily::sweep, py::arg("complex"), py::arg("squared_lengths"),
           py::arg("causal_types"), py::arg("epsilons"), py::arg("degree"),
           py::arg("preset") = Preset::L2, py::arg("branch") = Branch::Continuation,
           py::arg("kappa") = 10.0, py::arg("with_spectrum") = false,
           py::arg("crossover_dimension") = ChainHodge::kDefaultCrossoverDimension,
           "Reads at every epsilon of the family at one degree.")
      .def_static("extrapolateToZero", &LorentzianFamily::extrapolateToZero, py::arg("epsilons"),
           py::arg("values"), py::arg("order") = 2,
           "Labeled polynomial extrapolation of reads at epsilon > 0 to epsilon -> 0.");
  py::class_<Connection>(m, "Connection",
      R"doc(A C* connection on the canonical edges x < y: U_xy per edge, U_yx = 1/U_xy exactly,
U_xx = 1 (specification Def. 5.1). Gauge: U_xy -> g_x^{-1} U_xy g_y. Links are never
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
      "Residuals of specification Prop. 5.1 (i)-(vi) on an instance; NaN means unmeasured.")
      .def_readonly("transposeMetric", &CovarianceCertificate::transposeMetric)
      .def_readonly("transposePencil", &CovarianceCertificate::transposePencil)
      .def_readonly("covarianceMetric", &CovarianceCertificate::covarianceMetric)
      .def_readonly("covariancePencil", &CovarianceCertificate::covariancePencil)
      .def_readonly("curvature", &CovarianceCertificate::curvature)
      .def_readonly("pairingInvariance", &CovarianceCertificate::pairingInvariance)
      .def_readonly("trivialReduction", &CovarianceCertificate::trivialReduction)
      .def_readonly("pureGaugeIsospectrality", &CovarianceCertificate::pureGaugeIsospectrality)
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
      "Certificates of one Riesz band (specification §6); NaN means unmeasured; no sign or "
      "inertia is extracted from B_C.")
      .def_readonly("contour", &BandCertificate::contour)
      .def_readonly("nodeCount", &BandCertificate::nodeCount)
      .def_readonly("idempotency", &BandCertificate::idempotency)
      .def_readonly("rank", &BandCertificate::rank)
      .def_readonly("rankTolerance", &BandCertificate::rankTolerance)
      .def_readonly("singularGap", &BandCertificate::singularGap)
      .def_readonly("resolventMax", &BandCertificate::resolventMax)
      .def_readonly("detB", &BandCertificate::detB)
      .def_readonly("condB", &BandCertificate::condB)
      .def_readonly("pairingScale", &BandCertificate::pairingScale)
      .def_readonly("leftFrameAvailable", &BandCertificate::leftFrameAvailable)
      .def_readonly("leftFrameRefusal", &BandCertificate::leftFrameRefusal)
      .def_readonly("rightResidual", &BandCertificate::rightResidual)
      .def_readonly("leftResidual", &BandCertificate::leftResidual);

  py::class_<Band>(m, "Band",
      R"doc(One Riesz band of h_k(s,U) (specification §6): projector P on chains, right frame Phi,
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
      R"doc(The covariant one-particle operator h_k(s,U) of specification §5: the sparse
inverse chain metric dressed by U_{b(sigma) b(tau)} and the incidences twisted by
U_{b(tau) b(sigma)}, b(sigma) = min sigma, with the dressed pencil (A~_k^U, M_k^U) on
images and the exact properties of Prop. 5.1 measured on every instance.)doc")
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
      .def("covariantOperatorPhaseDerivative", &CovariantChainHodge::covariantOperatorPhaseDerivative,
           py::arg("k"), py::arg("edge_index"),
           "dh_k/dphi_e for the multiplicative link variation U_e = e^{i phi_e}, dense.")
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
           "H_k = M_k^U ker S^U with S^U = [(d_{k+1}^{U^-1})^T; d_k^U M_k^U] (RSF Sec. 5): the "
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
      .def_static("leftFrame", &CovariantChainHodge::leftFrame, py::arg("band"), py::arg("dual_instance"),
           py::arg("isotropy_tolerance") = 1e-10,
           "G^{U^-1} Phi^vee B_C^{-T} from the band's dual frame and pairing; raises on an isotropic band.");
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
      "Analytic derivatives of a Riesz band's geometric images (#947): dZ = G^U(-dM^U Z + dP Phi) with "
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
      R"doc(The face anchor of specification §8 with the Whitney metric: the per-triangle
Whitney block M_1^{(t)} (rank 3, Prop. 8.1), its connection dressing by U_{b(e)b(e')},
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
      // ---- tetrahedral anchor at degree 0 (#939) ----
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
      "P = A - lambda M, with det P = det P_II det F_B and the constraint modes T = [I_B; -P_II^{-1} P_IB].")
      .def_readonly("lambda_", &FeshbachResult::lambda)
      .def_readonly("interface", &FeshbachResult::interface)
      .def_readonly("interior", &FeshbachResult::interior)
      .def_readonly("response", &FeshbachResult::response)
      .def_readonly("constraintModes", &FeshbachResult::constraintModes)
      .def_readonly("interiorDeterminant", &FeshbachResult::interiorDeterminant)
      .def_readonly("responseDeterminant", &FeshbachResult::responseDeterminant)
      .def_readonly("pencilDeterminant", &FeshbachResult::pencilDeterminant)
      .def_readonly("determinantResidual", &FeshbachResult::determinantResidual)
      .def_readonly("solveResidual", &FeshbachResult::solveResidual)
      .def_readonly("interiorSingular", &FeshbachResult::interiorSingular);

  py::class_<CongruenceResult>(m, "CongruenceResult", "A congruence (T^T A T, T^T M T).")
      .def_readonly("A", &CongruenceResult::A)
      .def_readonly("M", &CongruenceResult::M);

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
      R"doc(The recursion on the symmetric pencil P(lambda) = A~ - lambda M on geometric images
(specification §7, §13): the Feshbach complement with its determinant factorization, the
Craig-Bampton congruence, the restriction of pencil and chain metric to retained fibers, and
the transfer between fibers with the reversal identity asserted at runtime. Every pairing is
the transpose.)doc")
      .def_static("feshbach", &PencilSchur::feshbach, py::arg("A"), py::arg("M"), py::arg("lambda_"),
           py::arg("interface"), py::arg("rank_tolerance") = 1e-12)
      .def_static("craigBampton", &PencilSchur::craigBampton, py::arg("A"), py::arg("M"), py::arg("T"))
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
}
