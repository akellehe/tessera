// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the Hodge Laplacian and
// eigenstate synthesis. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_cobordism_hodge(py::module_ &m) {
  // ----- Hodge Laplacian: k=0 Hermitian graph, k>=1 metric Hodge -----
  py::enum_<HodgeLaplacian::MetricSource>(m, "HodgeMetricSource",
      R"doc(Where a Hodge operator's metric comes from: WhitneyPencil (the chain-level
Whitney Hodge pencil of tessera.chainhodge, W_k = M_k^-1 dressed at every degree by
the edge-phase links, so the operator is h_k(s, U) and moves with the connection;
the process default) or DiagonalWeights (the per-simplex diagonal weights of
HodgeWeightConvention, whose operator ignores the connection).)doc")
      .value("DiagonalWeights", HodgeLaplacian::MetricSource::DiagonalWeights)
      .value("WhitneyPencil", HodgeLaplacian::MetricSource::WhitneyPencil);

  py::enum_<HodgeLaplacian::WeightConvention>(m, "HodgeWeightConvention",
      R"doc(Which quantity the diagonal inner-product weight W_k is built from.

BOTH are fully Lorentzian and complex-valued. This is a choice of inner product,
not of signature; neither reintroduces a Euclidean path.)doc")
      .value("Content", HodgeLaplacian::WeightConvention::Content,
             "W_k = V, the k-content (the textbook diagonal DEC star). For an "
             "edge that is sqrt(l^2), so a TIMELIKE cell's weight is imaginary. "
             "Spacelike and timelike contributions to <h,h>_W are then 90 degrees "
             "apart and cannot cancel, so no null kernel direction exists.")
      .value("SquaredContent", HodgeLaplacian::WeightConvention::SquaredContent,
             "W_k = V^2, the squared k-content; for an edge exactly l^2. It is "
             "det G/(d!)^2, a polynomial in the squared edge lengths, so on real "
             "signed l^2 it is real and SIGNED -- timelike cells carry a negative "
             "weight and genuine null kernel directions survive.")
      .export_values();

  py::enum_<HodgeLaplacian::EntropyPhaseMode>(m, "HodgeEntropyPhaseMode",
      "Whether Hodge spectral entropy retains complex operator entries or "
      "uses the entrywise-magnitude phase-blind ablation.")
      .value("IncludeComplexPhase",
             HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase,
             "Use M=L_k, retaining every complex phase.")
      .value("IgnoreComplexPhase",
             HodgeLaplacian::EntropyPhaseMode::IgnoreComplexPhase,
             "Use M_ij=|L_k,ij| for this entropy observable only; live complex "
             "edge lengths and z=l^2 are unchanged.")
      .export_values();

  py::class_<HodgeLaplacian>(m, "HodgeLaplacian",
      R"doc(Hodge Laplacian on a Spacetime, degree-parameterized by int k.

METRIC SOURCE. The metric is chosen by HodgeMetricSource, read from the
process-wide defaultMetricSource() at construction unless named. The default,
WhitneyPencil, is the chain-level Whitney pencil: W_k = M_k^-1 with M_k the sparse
Whitney mass matrix of the complex squared lengths, dressed by the connection U of
the edge phases. laplacian(k) is then the operator on geometric images
L_z = (M_k^U)^-1 A~_k^U, similar to h_k(s, U), and pencil(k) returns the pair
(A~_k^U, M_k^U). The DiagonalWeights operator described next ignores U.

DIAGONAL WEIGHTS. ONE definition at every degree: with the integer boundary maps
d_k (ChainComplex), the diagonal metric weight W_k (weights(k); W_0 = I) and the
weighted adjoint d_k* = W_k^-1 d_k^dagger W_{k-1},

    L_k = d_{k+1} d_{k+1}* + d_k* d_k
        = d_{k+1} W_{k+1}^-1 d_{k+1}^dagger W_k + W_k^-1 d_k^dagger W_{k-1} d_k

for every k >= 0. At k = 0 the second term is absent (no (-1)-chains), so
L_0 = d_1 W_1^-1 d_1^dagger: the graph Laplacian with conductance 1/W_1(e) on the
1-cell e. Its row sums vanish identically, so the constant 0-cochain is harmonic
at ANY weights and dim ker L_0 = b_0 = the number of connected components,
independently of the geometry.

W_k is the SIGNED Simplex.volume under the active HodgeWeightConvention (timelike
cells negative or imaginary), so the inner product is indefinite and L_k is
assembled directly and is generally non-self-adjoint at every degree, degree zero
included: eigenvalues may be negative or complex (ComplexEigenSolver, sorted by
(Re, Im)). ker L_k ~= H_k likewise degrades away from positive weights --
'harmonic' becomes the small-|lambda| near-kernel and a representative h can be
null (<h,h>_W = sum_i W_{k,i}|h_i|^2 ~= 0, see nullNorms) -- but L_0 @ 1 = 0
survives every weight. metric=False selects unit weights (the combinatorial
d_{k+1} d_{k+1}^T + d_k^T d_k) at every degree. k-cells follow the canonical
ChainComplex column order at every degree, so the matrices align with
boundaryMatrix(k) and weights(k). Negative k raises; k above the top dimension
yields empty results. Spectra are computed lazily and cached. This is the
operator only -- fluxes, cycle bases, and Betti numbers belong to WilsonLoop /
ChainComplex.

The U(1) CONNECTION Laplacian is a DIFFERENT operator. connectionLaplacian
(with adjacency, degree, connectionSpectrum and friends) is the Hermitian
L = D - A on the 1-skeleton assembled from each edge's complex weight
squaredLength * exp(i*phase): adjacency Hermitian (the reverse orientation
negates the phase), degree D_ii = sum |squaredLength| (the MAGNITUDE convention).
On a Lorentzian complex a timelike edge has l^2 < 0, so its magnitude diagonal
and signed off-diagonal disagree, its row sums do not vanish, and it is not
d_1 W_1^-1 d_1^dagger for any W. It carries the Aharonov-Bohm content that L_0
cannot -- a nonzero flux lifts its zero mode, whereas dim ker L_0 is always b_0 --
and it is indexed over the FULL sorted vertex-id order, including any lone vertex
ChainComplex omits.)doc")
      .def(py::init([](std::shared_ptr<Spacetime> st) {
             // No-weights overload: read the process default at call time; a
             // pybind default argument would bake it in at import.
             return new HodgeLaplacian(std::move(st));
           }),
           py::arg("spacetime"))
      .def(py::init<std::shared_ptr<Spacetime>, HodgeLaplacian::WeightConvention>(),
           py::arg("spacetime"),
           py::arg("weights"),
           "Build the Hodge Laplacian operator over a triangulation. `weights` "
           "selects which quantity the diagonal W_k is built from -- the "
           "k-content or its square (the default). See HodgeWeightConvention.")
      .def(py::init<std::shared_ptr<Spacetime>, HodgeLaplacian::WeightConvention,
                    HodgeLaplacian::MetricSource>(),
           py::arg("spacetime"), py::arg("weights"), py::arg("metric_source"),
           "Build with an explicit metric source (see HodgeMetricSource).")
      .def_static("defaultMetricSource", &HodgeLaplacian::defaultMetricSource,
           "The process-wide default HodgeMetricSource (ships as WhitneyPencil).")
      .def_static("setDefaultMetricSource", &HodgeLaplacian::setDefaultMetricSource,
           py::arg("source"), "Flip the process-wide default metric source ONCE at startup.")
      .def("metricSource", &HodgeLaplacian::metricSource, "This operator's metric source.")
      .def("pencil",
           [](const HodgeLaplacian &hodge, int k) {
             HodgeLaplacian::MetricPencil pencil = hodge.pencil(k);
             return py::make_tuple(std::move(pencil.op), std::move(pencil.metric));
           },
           py::arg("k"),
           "Whitney pencil only: (A~_k^U, M_k^U), both flat row-major |C_k| x |C_k| in the "
           "cell order and stored orientation of laplacian(k), which is M^-1 A~. Raises "
           "under DiagonalWeights, whose metric is weights(k).")
      .def("laplacianPhaseGradient", &HodgeLaplacian::laplacianPhaseGradient,
           py::arg("k"), py::arg("ea"), py::arg("eb"),
           "Whitney pencil: the analytic dL_k/dphi_e of the link on edge (ea, eb), flat "
           "row-major; identically zero under DiagonalWeights.")
      .def_static("kontsevichSegalMargin", &HodgeLaplacian::kontsevichSegalMargin,
           py::arg("spacetime"),
           "min over top simplices of pi - sum_i |arg lambda_i(g_T)| of the current geometry.")
      .def_static("defaultWeightConvention",
           &HodgeLaplacian::defaultWeightConvention,
           "The process-wide default HodgeWeightConvention, read by every "
           "internally-constructed operator (r_U terms, the near-kernel "
           "residual, the register readout). See setDefaultWeightConvention.")
      .def_static("defaultWeightConvention",
           &HodgeLaplacian::defaultWeightConvention,
           "The process-global default weight convention new HodgeLaplacian "
           "instances adopt — capture it before setDefaultWeightConvention "
           "to restore the prior state exactly.")
      .def_static("setDefaultWeightConvention",
           &HodgeLaplacian::setDefaultWeightConvention, py::arg("convention"),
           "Set the process-wide default HodgeWeightConvention. Flip it ONCE "
           "at startup (e.g. the animation's --hodge-weights flag); flipping "
           "mid-run mixes conventions across cached spectra.")
      .def("adjacency", &HodgeLaplacian::adjacency,
           "Weighted adjacency A of the U(1) CONNECTION operator as a flat "
           "row-major N*N complex array over the sorted vertex order "
           "(Hermitian; A_ij = sum squaredLength * exp(i*phase)). Not part of "
           "L_0.")
      .def("degree", &HodgeLaplacian::degree,
           "Degree vector of the U(1) CONNECTION operator (length N, real): "
           "D_ii = sum |squaredLength| over incident edges (magnitude "
           "convention). Not part of L_0.")
      .def("connectionLaplacian", &HodgeLaplacian::connectionLaplacian,
           "The Hermitian U(1) connection graph Laplacian L = D - A as a flat "
           "row-major N*N complex array over the FULL sorted vertex-id order "
           "(every vertex, including any carried by no simplex). NOT the "
           "degree-zero Hodge Laplacian: its off-diagonal is the signed complex "
           "weight while its diagonal is the magnitude, so its row sums do not "
           "vanish on a Lorentzian complex. It is the Aharonov-Bohm operator -- "
           "Hermitian, PSD by Gershgorin, unitary under exp(-iLt), zero mode "
           "lifted by flux. Use laplacian(0) for the Hodge operator.")
      .def("laplacian", &HodgeLaplacian::laplacian, py::arg("k") = 0,
           py::arg("metric") = true,
           "Laplacian L_k as a flat row-major |C_k|*|C_k| complex array in the "
           "canonical ChainComplex column order: the signed-weight "
           "d'Alembertian (complex, generally non-symmetric) at EVERY degree, "
           "degree zero included (L_0 = d_1 W_1^-1 d_1^dagger, row sums zero). "
           "metric=False uses unit weights (combinatorial) at every degree. "
           "Raises for k<0; empty above the top dimension.")
      .def("weights", &HodgeLaplacian::weights, py::arg("k"),
           "Diagonal inner-product weights W_k (length |C_k|) in ChainComplex "
           "column order: the per-k-simplex SIGNED complex volume (W_0 = I, "
           "which is what makes the L_0 row sums vanish). A Lorentzian cell's "
           "content is negative or imaginary. Empty for k<0 or k above the top "
           "dimension.")
      .def("laplacianGradient", &HodgeLaplacian::laplacianGradient, py::arg("k"),
           py::arg("edgeA"), py::arg("edgeB"),
           "Exact analytic dL_k/dl^2_e w.r.t. one edge's squared length, flat "
           "|C_k|x|C_k| row-major, at every degree k>=0. Only the weights W_j "
           "depend on l^2 (dW_j = Simplex.volumeGradient); at k=0, where "
           "W_0 = I is constant, the only surviving term is "
           "-d_1 W_1^-1 (dW_1) W_1^-1 d_1^dagger. Empty for k<0 or an absent "
           "edge.")
      .def("spectralEntropy", &HodgeLaplacian::spectralEntropy, py::arg("k"),
           py::arg("phase_mode") =
               HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase,
           "Von Neumann entropy of rho=A/Tr(A), A=M^dagger M. M=L_k when "
           "complex phase is included and M_ij=|L_k,ij| in the phase-blind "
           "ablation. Empty/zero operators return zero.")
      .def("spectralEntropyGradient",
           &HodgeLaplacian::spectralEntropyGradient, py::arg("k"),
           py::arg("phase_mode") =
               HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase,
           "Complex-z gradient h=dS/dRe(z)-i*dS/dIm(z), in EdgeList order, "
           "for z=l^2. conj(h) is the steepest-ascent displacement. Available "
           "at every degree k>=0: L_k is holomorphic in z at all of them.")
      .def("spectralEntropyGradientNorm",
           &HodgeLaplacian::spectralEntropyGradientNorm, py::arg("k"),
           py::arg("phase_mode") =
               HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase,
           "Entropy-stationarity residual sum_e |dS/dz_e|^2.")
      .def("connectionSpectralEntropy",
           &HodgeLaplacian::connectionSpectralEntropy,
           "-sum p log p over the normalized SQUARED EIGENVALUE MODULI of the "
           "C* CONNECTION operator, p_i = |lambda_i|^2 / sum_j |lambda_j|^2. "
           "Read from the EIGENvalues, NOT from the A = M^dag M form "
           "spectralEntropy uses on L_k: that one is a functional of the "
           "SINGULAR values, which only UNITARY similarity preserves, and the "
           "C* gauge action is non-unitary whenever g has a modulus. "
           "Eigenvalues survive the full similarity, so gauge invariance here "
           "is structural. The SQUARE is what makes it reduce to the Hodge "
           "term's own functional in the Hermitian limit, where |lambda|^2 = "
           "sigma^2 are exactly the eigenvalues of A. The entropy of the 1-skeleton "
           "operator; spectralEntropy of the default h_k(s, U) sees the connection "
           "as well, at every degree.")
      .def("connectionSpectralEntropyPhaseGradient",
           &HodgeLaplacian::connectionSpectralEntropyPhaseGradient,
           "dS/dphi_e of connectionSpectralEntropy, EdgeList order, in the "
           "h = S_x - i S_y convention. Each eigenvalue is holomorphic in phi "
           "-- the reverse orientation carries the INVERSE link, never the "
           "conjugate -- and the squared modulus supplies the only "
           "non-holomorphic step, in closed form, so this is exact rather than "
           "a real-parameter approximation. Both components are differentiated.")
      .def("connectionSpectralEntropyPhaseGradientNorm",
           &HodgeLaplacian::connectionSpectralEntropyPhaseGradientNorm,
           "Connection-entropy stationarity residual sum_e |dS/dphi_e|^2.")
      .def("spectralEntropyGradientDirectionalDerivative",
           &HodgeLaplacian::spectralEntropyGradientDirectionalDerivative,
           py::arg("k"), py::arg("direction"),
           py::arg("phase_mode") =
               HodgeLaplacian::EntropyPhaseMode::IncludeComplexPhase,
           "EXACT analytic Hessian-vector product: the directional derivative "
           "of spectralEntropyGradient along `direction` (EdgeList order) for a "
           "REAL parameter, which is what the descent direction of "
           "||grad_z S||^2 needs. Closed form -- the simplex volume Hessian, "
           "the second derivative of L_k, and the Daleckii-Krein derivative of "
           "dS/dA on the fixed-rank stratum. S is invariant under complex "
           "rescaling of z, so h is homogeneous of degree -1 and the exact "
           "Euler check is: direction = z reproduces -h.")
      .def("localSpectralMoments", &HodgeLaplacian::localSpectralMoments, py::arg("k"), py::arg("orders"),
           "The local spectral moments mu_j(x) = (L_k^j)_xx, j = 1..orders, flat row-major |C_k| x orders: the "
           "local parts of the power sums tr(L_k^j), holomorphic in z. L_k is the one spectralEntropy uses.")
      .def("spectralMomentStiffness", &HodgeLaplacian::spectralMomentStiffness, py::arg("k"),
           py::arg("reference"), py::arg("coefficients"),
           "S_M = 1/2 sum_j beta_j sum_x (mu_j(x) - mu_j^0(x))^2 about the carrier whose localSpectralMoments(k, m) "
           "are `reference`, beta_j = coefficients[j-1]: extensive, zero with its gradient at the carrier, and with "
           "the Hessian sum_j beta_j sum_x grad mu_j grad mu_j^T there. Holomorphic in z.")
      .def("spectralMomentStiffnessGradient", &HodgeLaplacian::spectralMomentStiffnessGradient, py::arg("k"),
           py::arg("reference"), py::arg("coefficients"),
           "dS_M/dz_e in EdgeList order, the holomorphic derivative; for Re S_M it is also the h of "
           "spectralEntropyGradient.")
      .def("spectralMomentStiffnessHessianProduct", &HodgeLaplacian::spectralMomentStiffnessHessianProduct,
           py::arg("k"), py::arg("reference"), py::arg("coefficients"), py::arg("direction"),
           "EXACT Hessian-vector product sum_f d^2 S_M / dz_e dz_f v_f in EdgeList order: the product rule on "
           "the moments and the exact second derivative of L_k along v.")
      .def("isHermitian", &HodgeLaplacian::isHermitian, py::arg("tol") = 1e-12,
           "True iff ||L - L^dagger|| <= tol (Frobenius) for the U(1) CONNECTION "
           "Laplacian. True by construction; it says nothing about L_0, which is "
           "complex symmetric as soon as a weight is complex.")
      .def("unitarityResidual", &HodgeLaplacian::unitarityResidual,
           py::arg("t") = 1.0,
           "Residual ||U U^dagger - I|| of U = e^{-iLt} for the U(1) CONNECTION "
           "Laplacian, formed from its eigendecomposition (~0, that operator "
           "being Hermitian).")
      .def("connectionSpectrum", &HodgeLaplacian::connectionSpectrum,
           "The U(1) CONNECTION Laplacian's eigendecomposition as a Spectrum "
           "(real ascending eigenvalues + eigenvectors as degree-0 Cochains; "
           "isHermitian()==True), over the full sorted vertex order.")
      .def("connectionEigenvalues", &HodgeLaplacian::connectionEigenvalues,
           "Eigenvalues of the U(1) CONNECTION Laplacian (real, ascending), "
           "complex-typed for parity with the L_k family.")
      .def("connectionEigenvectors", &HodgeLaplacian::connectionEigenvectors,
           "Eigenvectors of the U(1) CONNECTION Laplacian as a flat row-major "
           "N*N complex array (column j is the eigenvector for the j-th "
           "ascending eigenvalue).")
      .def("connectionHarmonics", &HodgeLaplacian::connectionHarmonics,
           py::arg("tol") = 1e-9,
           "Harmonic representatives of the U(1) CONNECTION Laplacian "
           "(|lambda| < tol) as degree-0 Cochains. NOT b_0: a nonzero U(1) flux "
           "lifts this zero mode.")
      .def("connectionHarmonicMatrix",
           &HodgeLaplacian::connectionHarmonicMatrix, py::arg("tol") = 1e-9,
           "The U(1) CONNECTION harmonic amplitude matrix: connectionHarmonics "
           "stacked as the ROWS of a flat row-major (dim ker) x N complex "
           "array, columns in the sorted vertex-id order.")
      .def("spectrum", &HodgeLaplacian::spectrum, py::arg("k") = 0,
           py::arg("metric") = true,
           "The eigendecomposition of L_k as a Spectrum. L_k is the "
           "signed-weight d'Alembertian at every degree, generally "
           "non-self-adjoint, so eigenvalues are complex, sorted by (Re, Im), "
           "and isHermitian()==False. metric selects signed-content vs. unit "
           "weights. Raises for k<0; empty above the top dimension.")
      .def("eigenvalues", &HodgeLaplacian::eigenvalues, py::arg("k") = 0,
           py::arg("metric") = true,
           "Eigenvalues of L_k (complex, sorted by (Re, Im)), a flat view "
           "consistent with spectrum(k, metric). metric selects signed-content "
           "vs. unit weights. Raises for k<0; empty above the top dimension.")
      .def("eigenvectors", &HodgeLaplacian::eigenvectors, py::arg("k") = 0,
           py::arg("metric") = true,
           "Eigenvectors of L_k as a flat row-major |C_k|*|C_k| complex array "
           "(column j is the eigenvector for the j-th eigenvalue), a flat view "
           "consistent with spectrum(k, metric).eigenvectors(). metric selects "
           "signed-content vs. unit weights. Raises for k<0; empty above the "
           "top dimension.")
      .def("harmonics", &HodgeLaplacian::harmonics, py::arg("k") = 0,
           py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Harmonic representatives (eigenvectors with |lambda| < tol) as a list "
           "of Cochains spanning ker L_k ~= H_k (the count is b_k at positive "
           "weights; at k=0 the constant is always among them, so dim ker L_0 = "
           "b_0 always). metric selects signed-content vs. unit weights. Raises "
           "for k<0; empty above the top dimension.")
      .def("harmonicMatrix", &HodgeLaplacian::harmonicMatrix, py::arg("k") = 0,
           py::arg("tol") = 1e-9, py::arg("metric") = true,
           "The harmonic amplitude matrix: the harmonics(k, tol, metric) "
           "representatives stacked as the ROWS of a flat row-major "
           "(dim ker L_k) x |C_k| complex array, "
           "columns in the canonical cell order (cellSimplices / "
           "kSimplexVertices). Entry [r*|C_k| + c] equals "
           "harmonics(k)[r].amplitude(c) exactly -- one call instead of one "
           "amplitudeFor round-trip per cell per harmonic. Raises for k<0; "
           "empty when the kernel is empty or k is above the top dimension.")
      // ----- indefinite W-norms of the near-kernel -----
      .def("nullNorms", &HodgeLaplacian::nullNorms,
           py::arg("k"), py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Indefinite norms of the near-kernel representatives in the metric that "
           "produced them, one per column of harmonics (same order): h^dagger M_k^U h "
           "with the dressed Whitney mass under WhitneyPencil, sum_i W_{k,i} |h_i|^2 "
           "with the signed diagonal weights under DiagonalWeights. A value ~0 flags a "
           "NULL (lightlike) harmonic; all positive on an all-spacelike complex.");

  // ----- eigenstate synthesis: residual + parameter access -----
  auto eigenstateSynthesis = py::class_<EigenstateSynthesis>(m, "EigenstateSynthesis",
      R"doc(Inverse eigenvector problem on a fixed complex, degree-k.

Scores how close the complex's current Hermitian edge weights make a target
state psi to being an eigenvector of the degree-k Hodge Laplacian L_k (via
HodgeLaplacian), and reads/writes those weights so a search can perturb them.
At k=0 the scored operator is the U(1) CONNECTION graph Laplacian D - A
(connectionLaplacian, the magnitude convention), NOT the Hodge L_0: a
degree-zero register carries U(1) flux and dim ker L_0 is always b_0, so an L_0
readout would be identically gauge-flat. psi is then a vertex vector
(|V|, sorted-id order). At k>=1 L_k is the metric Hodge Laplacian
on k-forms (|C_k|, ChainComplex k-cell order); the tunable parameters stay the
edge squared-lengths, which feed the volume weights W_k of L_k via Simplex.volume
(phases enter only k=0). cellSimplices() gives each psi component's vertex tuple,
so a caller can pin the boundary k-cells to a target form (k=1
3-manifold boundary-harmonic synthesis). The non-convex, multi-restart search
itself (e.g.
scipy.optimize.minimize L-BFGS-B over the flat {w_ij} + {theta_ij} vector) lives
in the driver and calls residual() here; the cone-and-retry growth loop is a
separate stage (this class is fixed-complex only).

Residual: for a unit target, r(psi) = ||(I - psi psi^dagger) L psi||^2 =
||L psi - lambda psi||^2 with lambda = psi^dagger L psi, so r = 0 iff
L psi || psi (psi is an eigenvector) and the realized eigenvalue is the Rayleigh
quotient lambda. A non-unit psi is normalized internally. L is reassembled from
the live edge weights/phases on every call, so residual() tracks setWeights /
setPhases in place. psi is indexed in the same sorted-vertex-id order as
HodgeLaplacian (k=0).

Parameters: the per-edge SIGNED real squared lengths {w_ij} = Re l^2
(Edge.setSquaredLength; weights() reads Re, not a magnitude) and U(1)
phases {theta_ij} (Edge.setPhase), in a stable edge order fixed at
construction (the weight-carrying edges: both endpoints present, no self-loops).

Fixed-boundary interior fill: the tunable edges split into a boundary set
dW (edges on a codim-1 face in exactly one top cell — held fixed) and an interior
set (free). interiorWeights / interiorPhases + setInteriorWeights /
setInteriorPhases read/write only the interior edges, so a search drives r -> 0
for a target output eigenvector while dW stays byte-identical (boundaryEdges()
exposes that fixed set). growInterior() cones a fresh interior vertex via the
boundary-fixed pre-geometric Pachner add, enriching the interior with dW
untouched; interiorVertexCount / numInteriorEdges report the interior complexity
reached. On a 1-complex there is no boundary — every edge is interior.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, int, HodgeLaplacian::MetricSource>(),
           py::arg("spacetime"), py::arg("k"), py::arg("metric_source"),
           "Build with an explicit metric source (HodgeMetricSource).")
      .def(py::init<std::shared_ptr<Spacetime>, int>(), py::arg("spacetime"),
           py::arg("k") = 0,
           "Build the synthesizer over a fixed triangulation at Hodge degree k "
           "(default 0, the vertex graph Laplacian; k=1 is the metric Hodge "
           "Laplacian on edge 1-forms for the 3-manifold boundary-harmonic "
           "synthesis). Raises if k < 0.")
      .def("degree", &EigenstateSynthesis::degree,
           "The cochain degree k of L_k this synthesizer scores against.")
      .def("metricSource", &EigenstateSynthesis::metricSource,
           "Where this synthesizer's operator takes its metric from (HodgeMetricSource).")
      .def("order", &EigenstateSynthesis::order,
           "Operator dimension N — the required length of any psi (|V| at k=0, "
           "else |C_k|, the number of k-cells).")
      .def("cellSimplices", &EigenstateSynthesis::cellSimplices,
           "The sorted vertex-id tuple of each psi component, in operator order "
           "(a single-vertex tuple per component at k=0, else the k-cell tuples "
           "in canonical ChainComplex column order) — used to pin the boundary "
           "k-cells to a target form and leave the interior free.")
      .def("numEdges", &EigenstateSynthesis::numEdges,
           "Number of tunable edges — the length of weights() / phases().")
      .def("residual", &EigenstateSynthesis::residual, py::arg("psi"),
           "Eigenvalue-agnostic residual r(psi) = ||(I - psi psi^dagger) L psi||^2 "
           "against the current edge weights/phases (psi normalized internally). "
           "r = 0 iff L psi || psi. Raises if len(psi) != order().")
      .def("rayleigh", &EigenstateSynthesis::rayleigh, py::arg("psi"),
           "Rayleigh quotient lambda = psi^dagger L psi / psi^dagger psi (real; L "
           "Hermitian) — the realized eigenvalue when r = 0. Raises if "
           "len(psi) != order().")
      .def("apply", &EigenstateSynthesis::apply, py::arg("psi"),
           "L psi against the current edge weights/phases (no normalization), for "
           "direct L psi || psi cross-checks. Raises if len(psi) != order().")
      .def("weights", &EigenstateSynthesis::weights,
           "The SIGNED real parts {Re l^2_ij} of the edge squared lengths, in "
           "the stable edge order — not magnitudes (a timelike edge reads "
           "negative), and any resident Im l^2 is not reported.")
      .def("phases", &EigenstateSynthesis::phases,
           "Edge phases {theta_ij} (radians) in the stable edge order.")
      .def("setWeights", &EigenstateSynthesis::setWeights, py::arg("w"),
           "Write the edge squared lengths in place as REAL signed values "
           "(l^2 = w + 0i, zeroing any resident Im — the ordinary-Lorentzian "
           "convention). Raises if len(w) != numEdges().")
      .def("setPhases", &EigenstateSynthesis::setPhases, py::arg("theta"),
           "Write the edge phases in place. Raises if len(theta) != numEdges().")
      // ----- Fixed-boundary interior fill -----
      .def("numInteriorEdges", &EigenstateSynthesis::numInteriorEdges,
           "Number of interior tunable edges (not on dW) — the length of "
           "interiorWeights() / interiorPhases() and the free parameters a "
           "fixed-boundary search varies.")
      .def("numBoundaryEdges", &EigenstateSynthesis::numBoundaryEdges,
           "Number of boundary tunable edges (on dW, held fixed).")
      .def("interiorVertexCount", &EigenstateSynthesis::interiorVertexCount,
           "Number of interior vertices (on no boundary face) — the coned-in "
           "apexes; the interior complexity the synthesis grows / reports.")
      .def("interiorWeights", &EigenstateSynthesis::interiorWeights,
           "Interior edge SIGNED real squared lengths {Re l^2_ij} in "
           "interior-edge order (Re, not magnitudes).")
      .def("interiorPhases", &EigenstateSynthesis::interiorPhases,
           "Interior edge phases {theta_ij} (radians) in interior-edge order.")
      .def("setInteriorWeights", &EigenstateSynthesis::setInteriorWeights,
           py::arg("w"),
           "Write the interior edge squared lengths in place as REAL signed "
           "values (l^2 = w + 0i, zeroing any resident Im); the boundary "
           "edges are left untouched. Raises if len(w) != numInteriorEdges().")
      .def("setInteriorPhases", &EigenstateSynthesis::setInteriorPhases,
           py::arg("theta"),
           "Write the interior edge phases in place; the boundary edges are left "
           "untouched. Raises if len(theta) != numInteriorEdges().")
      .def("boundaryEdges", &EigenstateSynthesis::boundaryEdges,
           "The boundary tunable edges as sorted (min_id, max_id) endpoint "
           "tuples — the fixed dW edge set, for asserting it is untouched through "
           "an interior fill / growth sweep.")
      .def("interiorEdges", &EigenstateSynthesis::interiorEdges,
           "The interior tunable edges as sorted (min_id, max_id) endpoint tuples "
           "(the complement of boundaryEdges()).")
      .def("growInterior", &EigenstateSynthesis::growInterior, py::arg("seed"),
           "Cone a fresh interior vertex into a top cell via the boundary-fixed "
           "pre-geometric Pachner add: a 1->(d+1) stellar subdivision that "
           "leaves dW exactly fixed while enriching the interior. Re-captures the "
           "vertex order and interior/boundary partition, so order() grows by one "
           "(extend psi on the new apex, appended last in sorted-id order) and "
           "numInteriorEdges() grows. Returns False if no top cell can be "
           "subdivided (e.g. a 1-complex), leaving the complex unchanged.")
      // ----- Free interior connectivity (general growth primitive) -----
      .def("attachInteriorVertex", &EigenstateSynthesis::attachInteriorVertex,
           py::arg("incident_simplices"),
           "Add a fresh interior vertex with an arbitrary specified set of "
           "incident simplices — the cone-free generalization of growInterior. "
           "incident_simplices is a list of vertex-id lists; the new vertex + each "
           "such set forms one new simplex (its full 1-skeleton is materialized), "
           "so a singleton [u] wires the new vertex to u by an edge and the d "
           "facets of a top cell reproduce coning. The new vertex takes the "
           "largest id. Validates ONLY (a) a valid downward-closed complex and "
           "(b) the pinned boundary dW bit-exact; no manifold/topology constraint. "
           "Returns False, leaving the complex unchanged, on an invalid spec "
           "(missing/repeated vertex, empty) or any perturbation of dW.")
      .def("detachLastInteriorVertex",
           &EigenstateSynthesis::detachLastInteriorVertex,
           "Undo the most recent attachInteriorVertex (LIFO): remove its created "
           "simplices/edges and the interior vertex, restoring the complex bit-"
           "exactly, and re-capture. Returns False if there is no attach to undo. "
           "Lets a search try a candidate connectivity, score it, and roll back.")
      .def("vertexIds", &EigenstateSynthesis::vertexIds,
           "All vertex ids, sorted — the candidate pool a connectivity search "
           "wires a fresh interior vertex into.")
      .def("boundaryVertexIds", &EigenstateSynthesis::boundaryVertexIds,
           "The boundary (dW) vertex ids, sorted — the vertices on a codim-one "
           "face of exactly one top cell (a 'boundary-star' candidate).")
      .def("topCells", &EigenstateSynthesis::topCells,
           "The top cells as sorted vertex-id tuples (the d+1-vertex simplices); "
           "wiring the new vertex to one reproduces growInterior's 1-skeleton.")
      .def("dualComplexValid", &EigenstateSynthesis::dualComplexValid,
           "(ok, reason): ChainComplex.dualComplexIsValid for the CURRENT "
           "complex -- top cells from the surgery state, with the k-cell "
           "universe checked for dangling facets when k = n-1 (the register "
           "layers). Accept topology moves only while this stays true.")
      // ----- The carried register read-outs -----
      .def("cyclePeriods", &EigenstateSynthesis::cyclePeriods, py::arg("holes"),
           "The period matrix of the current harmonics over the boundary "
           "cycles of the given (removed) cells: flat row-major "
           "(dim ker L_k) x len(holes), complex. Entry [r*m + q] sums "
           "harmonic r over hole q's facets with the boundary operator's "
           "induced-orientation signs (facet j of the sorted hole drops v_j, "
           "sign (-1)^j) -- degree-general: circles at k=1, spheres at k=2. "
           "Harmonics are read fresh from the live complex, rows ascending "
           "by eigenvalue. Raises if a hole is not a (k+2)-vertex tuple "
           "whose facets are all current k-cells.")
      .def("carriedRepresentative", &EigenstateSynthesis::carriedRepresentative,
           py::arg("holes"), py::arg("target_periods"),
           "The carried representative psi that residualForPeriods scores, as a "
           "cochain in its own right (it builds this internally but does not "
           "return it). Least-squares-projects target_periods onto the carried "
           "period rows (minimum-norm, as numpy.linalg.lstsq), forms the harmonic "
           "combination psi = sum_r c_r h_r, and attaches each hole's uncarried "
           "remainder (the minimal leak) to the hole's first walk-order facet so "
           "psi's periods are exactly target_periods. A full order()-length cell "
           "vector; residual(psi) is residualForPeriods. Raises on a hole/target "
           "length mismatch or a malformed hole.")
      .def("residualForPeriods", &EigenstateSynthesis::residualForPeriods,
           py::arg("holes"), py::arg("target_periods"),
           "The verdict primitive in one call: the genuine residual of the "
           "carried representative of target_periods over the holes' cycles. "
           "Least-squares-projects the targets onto the carried period rows "
           "(minimum-norm, as numpy.linalg.lstsq), forms the harmonic "
           "combination, attaches each hole's uncarried remainder (the "
           "minimal leak) to the hole's first walk-order facet (the (a,b) "
           "edge of a circle at k=1, the drop-v0 facet otherwise; boundary "
           "sign +1), and returns residual(psi): -> 0 iff the targets lie "
           "in the carried register, floored otherwise. Raises on a "
           "hole/target length mismatch or a malformed hole.")
      .def("residualForPeriodsGradient",
           &EigenstateSynthesis::residualForPeriodsGradient,
           py::arg("holes"), py::arg("target_periods"),
           "Arbitrary-degree exact analytic gradient d r_U / d l^2 of "
           "residualForPeriods w.r.t. each edge's squared length, in ChainComplex "
           "1-cell (edge) order. M = L_k, the per-edge dL_k/dl^2 = HodgeLaplacian."
           "laplacianGradient (built on Simplex.volumeGradient), through eigenvector-"
           "perturbation theory; period covector + leak from each removed-(k+1)-cell "
           "hole's facets. Reproduces the k=1 edge-loop core on triangle holes. At "
           "k=0 the core runs against the genuinely COMPLEX Hermitian U(1) "
           "CONNECTION operator D - A (full l^2 + U(1) phases; holes are removed "
           "1-cells, i.e. vertex pairs) with the SVD pseudo-inverse fit — "
           "the k=0 Euler identity is Σ l² ∂r_U = +2 r_U (that operator is "
           "degree +1 in l²). At k>=1, certified by the exact Euler identity Σ l² ∂r_U = −r_U (FD does not "
           "converge). Raises on a hole/target length mismatch.")
      .def("periodGapForPeriods", &EigenstateSynthesis::periodGapForPeriods,
           py::arg("holes"), py::arg("target_periods"),
           "The hard period-pin r_psi over the holes' cycles: r_psi = "
           "||P^T c - target||^2, where the columns of P^T are the live "
           "harmonics' periods over the holes and c is their least-squares fit "
           "-- the squared norm of the part of target_periods no pure harmonic "
           "can carry. Unlike residualForPeriods (r_U), the carried object stays "
           "a pure harmonic (NO leak). -> 0 iff the targets lie in the carried "
           "period span (the same realizable set as r_U), floored otherwise. "
           "Raises on a hole/target length mismatch or a malformed hole.")
      .def("periodGapForPeriodsGradient",
           &EigenstateSynthesis::periodGapForPeriodsGradient,
           py::arg("holes"), py::arg("target_periods"),
           "The exact analytic gradient d r_psi / d l^2 of periodGapForPeriods, "
           "in cellSimplices() (k=1 cell) order. By least-squares optimality "
           "(A^T r = 0, the envelope theorem) only the harmonic-subspace "
           "perturbation enters: d r_psi = 2 Re( r^H (Q dUn) c ) -- no leak, no "
           "dpsi chain. Raises on a hole/target length mismatch or a malformed "
           "hole.")
      // ----- The discovered operator: ker L1(W - dW) -----
      .def("bulkMinusBoundaryCells",
           &EigenstateSynthesis::bulkMinusBoundaryCells,
           "The interior 1-cells of W - dW (edges both of whose endpoints are "
           "interior vertices, on no dW face), as sorted (u,v) tuples in "
           "canonical ChainComplex C_1 order -- the column ordering of "
           "bulkMinusBoundaryHarmonicMatrix. Empty for a bare (un-grown) "
           "cobordism (all boundary, no interior bulk).")
      .def("bulkMinusBoundaryHarmonicMatrix",
           &EigenstateSynthesis::bulkMinusBoundaryHarmonicMatrix,
           py::arg("tol") = 1e-9, py::arg("metric") = false,
           "ker L1(W - dW) after deleting the full boundary subcomplex. "
           "metric=False preserves the combinatorial unit-weight operator. "
           "metric=True restricts the live signed Hodge weights and takes the "
           "right nullspace of the generally non-normal Lorentzian operator. "
           "The null vectors are stacked as rows of a flat row-major "
           "(dim ker L1) x len(bulkMinusBoundaryCells()) complex array. "
           "Use metric=True for relaxed-geometry claims; metric=False is "
           "topology-only. Read fresh from the live complex.")
      // ----- Surgery: the topology-changing interior remove move -----
      .def("interiorTopCells", &EigenstateSynthesis::interiorTopCells,
           "The interior top cells (all-interior vertices, on no dW face) as "
           "sorted vertex-id tuples — the surgery removal candidates. Removing one "
           "(removeInteriorCell) cannot touch dW, so it is the boundary-fixed "
           "TOPOLOGY-CHANGING move that can open a hole/handle and MOVE b_k, unlike "
           "growInterior's subdivision and the additive attach.")
      .def("removeInteriorCell", &EigenstateSynthesis::removeInteriorCell,
           py::arg("cell"),
           "Surgery: remove the interior top cell `cell` (a tuple from "
           "interiorTopCells()) and any edges it leaves orphaned, keeping a valid "
           "downward-closed complex. Topology-CHANGING: b_k moves (a filled disk "
           "b_1=0 becomes an annulus b_1=1). dW is held bit-exact — the cell has no "
           "boundary vertex, and the move is rejected if a dW edge would vanish; "
           "the EXPOSED interior boundary (the opened hole) is allowed. Records the "
           "removal for restoreLastRemoval. Returns False, complex unchanged, if "
           "`cell` is not an interior top cell or the removal would touch dW.")
      .def("restoreLastRemoval", &EigenstateSynthesis::restoreLastRemoval,
           "Undo the most recent removeInteriorCell (LIFO): re-create the removed "
           "top cell and the edges it orphaned, restoring their weights/phases bit-"
           "exactly, and re-capture. Returns False if there is no removal to undo. "
           "Lets a surgery search try a removal, score it, and roll back.")
      // ----- Gated moves: the checked cut and the composed stellar move -----
      .def("removeInteriorCellChecked",
           &EigenstateSynthesis::removeInteriorCellChecked, py::arg("cell"),
           "(ok, reason): the gated surgery cut — removeInteriorCell(cell), then "
           "the dual-validity gate (dualComplexValid), rolled back via "
           "restoreLastRemoval when the cut violates the dual. (True, 'ok') means "
           "the cut is applied and the dual complex stayed valid; (False, reason) "
           "means the complex is unchanged — the cell was not a removable interior "
           "top cell, or the reason names the dual violation. The gate is rigorous "
           "for n <= 3; dimension-4 callers use explicit constructions, not gated "
           "moves.")
      .def("stellarSubdivideInterior",
           &EigenstateSynthesis::stellarSubdivideInterior, py::arg("cell"),
           "(ok, reason): the composed gated stellar move — attach a fresh "
           "interior vertex onto `cell`'s facet fan (attachInteriorVertex with "
           "the d+1 codim-one facets; dW untouched), remove the subdivided parent "
           "(removeInteriorCell; its facets keep two cofaces, so dW stays "
           "bit-exact), gate on dualComplexValid, and roll back BOTH in LIFO "
           "order (restoreLastRemoval, then detachLastInteriorVertex) on "
           "violation. Each accepted move adds exactly ONE interior vertex and "
           "preserves ker L_k (the fan is homotopic to the cell it replaces). On "
           "acceptance the bulk's edges are re-pinned uniform (squaredLength 1, "
           "phase 0) — the unit cochain metric the register/fill seeds are built "
           "with, held by construction rather than by the createSimplexTracked "
           "time-rule coincidence on all-same-time seeds.")
      // ----- Charge sector: the E/B split of F in Omega^2 -----
      .def("curvatureFromConnection",
           &EigenstateSynthesis::curvatureFromConnection, py::arg("A"),
           "The curvature 2-cochain F = dA from a U(1) connection 1-cochain A by "
           "discrete coboundary: on each sorted degree-2 cell (a,b,c), F = "
           "A(a,b) + A(b,c) - A(a,c), the induced-orientation signed edge sum the "
           "period read-out uses (cyclePeriods). A is a degree-1 cochain in the "
           "canonical ChainComplex 1-cell order (length = the number of edges, "
           "i.e. EigenstateSynthesis(st, 1).order()); this instance is degree 2 "
           "and returns an order()-length 2-cochain. Gauge-invariant (d.d = 0): a "
           "pure gauge A -> A + d chi leaves F unchanged. Raises if degree() != 2, "
           "if len(A) is not the number of 1-cells, or if a 2-cell edge is "
           "missing.")
      .def("fieldStrengthSplit", &EigenstateSynthesis::fieldStrengthSplit,
           py::arg("F"),
           "The electric/magnetic split of a field-strength 2-cochain F by the "
           "causal type of each plaquette: electric = F on plaquettes carrying a "
           "timelike edge (one temporal leg, the discrete F_{0i}); magnetic = F "
           "on purely-spacelike plaquettes (F_{ij}). Returns a FieldStrengthSplit "
           "whose electric/magnetic are order()-length cochains (agreeing with F "
           "on their own support, zero elsewhere, so electric + magnetic == F) "
           "and whose electricCells/magneticCells are the disjoint, complete "
           "index lists into cellSimplices(). A plaquette is electric iff any of "
           "its three edges is Edge.isTimelike() on the live complex. Raises if "
           "degree() != 2, if len(F) != order(), or if a plaquette edge is "
           "missing.")
      .def("gaussLawCharge", &EigenstateSynthesis::gaussLawCharge, py::arg("F"),
           py::arg("enclosedVertices"), py::arg("electricOnly") = true,
           "The discrete Gauss-law charge Q = oint_S E: the temporal-sector "
           "flux of a field-strength 2-cochain F through the closed surface S = dV "
           "bounding the worldtube V (the closed star of enclosedVertices, the quark "
           "windows). Sums F over S's plaquettes with their induced (-1)^j "
           "orientation (interior faces of V cancel), restricted to the ELECTRIC "
           "(timelike-leg, F_{0i}) plaquettes when electricOnly, else the full flux. "
           "For an exact F = d psi the full flux is <psi, d^2 V> = 0 to round-off -- "
           "the topological protection that makes Q a metric-robust gauged-U(1) "
           "holonomy (unlike a hand-weighted flavor covector). On an all-spacelike "
           "(Riemannian) complex no plaquette is electric, so the electric Q is "
           "exactly 0 (the neutral total of the reduced color-only sector). Raises "
           "if degree() != 2 or len(F) != order().");

  // The result of fieldStrengthSplit: the E/B partition of F in Omega^2.
  py::class_<EigenstateSynthesis::FieldStrengthSplit>(
      eigenstateSynthesis, "FieldStrengthSplit",
      "The E/B split of a field-strength 2-cochain F by plaquette causal type "
      "(EigenstateSynthesis.fieldStrengthSplit): electric (timelike-leg "
      "plaquettes), magnetic (purely-spacelike plaquettes), and their disjoint "
      "index lists into cellSimplices(). electric + magnetic == F.")
      .def_readonly("electric",
                    &EigenstateSynthesis::FieldStrengthSplit::electric,
                    "F on plaquettes with a timelike leg (zero elsewhere); an "
                    "order()-length 2-cochain.")
      .def_readonly("magnetic",
                    &EigenstateSynthesis::FieldStrengthSplit::magnetic,
                    "F on purely-spacelike plaquettes (zero elsewhere); an "
                    "order()-length 2-cochain.")
      .def_readonly("electricCells",
                    &EigenstateSynthesis::FieldStrengthSplit::electricCells,
                    "Indices into cellSimplices() of the electric (timelike-leg) "
                    "plaquettes.")
      .def_readonly("magneticCells",
                    &EigenstateSynthesis::FieldStrengthSplit::magneticCells,
                    "Indices into cellSimplices() of the magnetic "
                    "(purely-spacelike) plaquettes.");
}
