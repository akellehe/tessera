// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Pybind11 bindings for the cobordism subsystem. Lives outside tessera_core
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is always added to
// _tessera's sources (see CMakeLists.txt, TESSERA_PYBIND_SOURCES).

#include <limits>
#include <optional>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/AnalyticCache.h"
#include "cobordism/Certificate.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/Characteristic.h"
#include "cobordism/DenseReference.h"
#include "cobordism/KuennethProduct.h"
#include "cobordism/LowRankUpdate.h"
#include "cobordism/OccupationSpectra.h"
#include "cobordism/Cochain.h"
#include "cobordism/CombinatorialDimension.h"
#include "cobordism/CobordismDAG.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/MultiCobordism.h"
#include "cobordism/PencilLayer.h"
#include "cobordism/Proton.h"
#include "cobordism/ProtonIngredients.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/IntegerLinalg.h"
#include "cobordism/RecursiveQuotient.h"
#include "cobordism/SurgicalCone.h"
#include "cobordism/Spectrum.h"
#include "spacetime/Spacetime.h"  // complete type required by pybind (typeid)

namespace py = pybind11;

using namespace tessera;
using namespace tessera::cobordism;

/// # PyCobordismObjective
///
/// The trampoline that lets a Python subclass supply the functional
/// `MultiCobordism` descends. Each override forwards to the Python method of
/// the corresponding snake_case name; the three with C++ defaults fall back to
/// the base implementation when a subclass does not define them.
///
/// The firewall survives the crossing unchanged, and for the same reason it
/// held in C++: a Python objective is handed an `ObjectiveContext`, which is
/// plain data carrying geometry, a region, that region's targets and scalar
/// configuration. It receives no node, no callable that closes over one, and
/// therefore no route to a component, fiber, transport, colour, charge,
/// flavour, exchange, spin certificate or verdict. Subclassing widens who may
/// write an objective; it does not widen what one can read.
///
/// The override macros acquire the GIL themselves, so an engine entry point
/// that released it — every long-running one does — re-enters Python safely.
class PyCobordismObjective : public CobordismObjective {
 public:
  using CobordismObjective::CobordismObjective;

  [[nodiscard]] std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, CobordismObjective, name);
  }

  [[nodiscard]] std::vector<std::string> termNames() const override {
    PYBIND11_OVERRIDE_PURE_NAME(std::vector<std::string>, CobordismObjective,
                                "term_names", termNames);
  }

  [[nodiscard]] ObjectiveTerms terms(
      const ObjectiveContext &context) const override {
    PYBIND11_OVERRIDE_PURE(ObjectiveTerms, CobordismObjective, terms, context);
  }

  [[nodiscard]] ObjectiveDirection direction(
      const ObjectiveDirectionContext &context) const override {
    PYBIND11_OVERRIDE_PURE(ObjectiveDirection, CobordismObjective, direction,
                           context);
  }

  [[nodiscard]] bool isTargetConditioned() const override {
    PYBIND11_OVERRIDE_PURE_NAME(bool, CobordismObjective,
                                "is_target_conditioned", isTargetConditioned);
  }

  [[nodiscard]] ObjectiveScope scope() const override {
    PYBIND11_OVERRIDE(ObjectiveScope, CobordismObjective, scope);
  }

  [[nodiscard]] bool needsRegisterResidual() const override {
    PYBIND11_OVERRIDE_NAME(bool, CobordismObjective, "needs_register_residual",
                           needsRegisterResidual);
  }

  [[nodiscard]] double numericalRegisterResidualWeight(
      const ObjectiveContext &context) const override {
    PYBIND11_OVERRIDE_NAME(double, CobordismObjective,
                           "numerical_register_residual_weight",
                           numericalRegisterResidualWeight, context);
  }

  [[nodiscard]] std::vector<HodgeDegreeContribution> hodgeDegreeContributions(
      const ObjectiveContext &context) const override {
    PYBIND11_OVERRIDE_NAME(std::vector<HodgeDegreeContribution>,
                           CobordismObjective, "hodge_degree_contributions",
                           hodgeDegreeContributions, context);
  }
};

void register_cobordism(py::module_ m) {
  // Smoke hook: lets tests assert the subsystem loaded before any
  // mathematical capability (issues #63–#70) is implemented. Single leading
  // underscore (not double) to avoid Python name-mangling inside test classes.
  m.def("_cobordism_smoke", [] { return true; },
        "Returns True; confirms the cobordism subsystem is built and importable.");

  // Per-complex scalar measurements are Observables (tessera's convention),
  // not methods on Spacetime or a bespoke wrapper. The characteristic-number
  // capabilities (Euler characteristic, signature, …) follow the same pattern;
  // multi-complex / structural operations (cobordism verification,
  // reconstruction, Pachner search) will be static-only classes taking a
  // Spacetime.
  py::class_<CombinatorialDimension, std::shared_ptr<CombinatorialDimension>>(
      m, "CombinatorialDimension",
      R"doc(Observable: combinatorial dimension of a triangulation.

The largest k with a k-simplex present (max simplex size - 1), or -1 if empty.
A purely combinatorial/topological integer (= n for a PL n-manifold), distinct
from the spectral dimension (a real-valued diffusion quantity) and from the
Spacetime's declared metric dimension.)doc")
      .def(py::init<>())
      .def("compute", &CombinatorialDimension::compute, py::arg("spacetime"),
           "Return the combinatorial dimension of the given Spacetime as a double.");

  // ----- Homology backbone (#64): chain complex + exact linear algebra -----

  py::class_<ChainComplex>(m, "ChainComplex",
      R"doc(Simplicial chain complex of a triangulation.

Boundary maps ∂_k over ℤ plus the homology invariants derived from them — Betti
numbers (over ℚ and GF(2)), torsion coefficients, Euler characteristic, and the
∂²=0 sanity check. Purely combinatorial (built from vertex sets; no geometry).)doc")
      .def("orientationSigns", &ChainComplex::orientationSigns,
           "Per-degree +/-1 signs relating stored cell orientations to the reference (ascending id) orientation.")
      .def_static("fromTopCells", &ChainComplex::fromTopCells, py::arg("top_cells"),
           "Build from top cells (vertex-id tuples) alone, oriented by ascending vertex id; no geometry.")
      .def_static("fromSpacetime", &ChainComplex::fromSpacetime, py::arg("spacetime"),
                  "Build the chain complex from a triangulation (a Spacetime).")
      .def("dimension", &ChainComplex::dimension)
      .def("numSimplices", &ChainComplex::numSimplices, py::arg("k"))
      .def("fVector", &ChainComplex::fVector)
      .def("eulerCharacteristic", &ChainComplex::eulerCharacteristic)
      .def("boundaryMatrix", &ChainComplex::boundaryMatrix, py::arg("k"),
           "Flat row-major ∂_k (rows=|C_{k-1}|, cols=|C_k|), entries in {-1,0,1}.")
      .def("boundaryComposesToZero", &ChainComplex::boundaryComposesToZero,
           "True iff ∂_{k-1}∘∂_k = 0 for all k.")
      .def_static(
          "dualComplexIsValid", &ChainComplex::dualComplexIsValid,
          py::arg("top_cells"), py::arg("dim"),
          py::arg("facet_cells") = std::vector<std::vector<std::uint64_t>>{},
          "(ok, reason): is the dual block decomposition of this pure "
          "n-complex a valid cell complex -- equivalently, is the primal a "
          "combinatorial manifold with boundary? Facet coface counts in "
          "{1,2}; no dangling facets against the optional (n-1)-cell "
          "universe; ridge links single paths/cycles; at n=3, vertex links "
          "2-spheres or disks. Pure combinatorics on sorted vertex-id "
          "tuples; rigorous for n <= 3. Accept topology moves only while "
          "this holds: validity in the DUAL space, not merely scoreability "
          "on the primal lattice.")
      .def("bettiNumbers", &ChainComplex::bettiNumbers, "Betti numbers b_0..b_n over Q.")
      .def("bettiNumbersGF2", &ChainComplex::bettiNumbersGF2, "Betti numbers over GF(2).")
      .def("torsion", &ChainComplex::torsion, py::arg("k"),
           "Torsion coefficients of H_k (invariant factors > 1 of d_{k+1}).")
      .def("kSimplexVertices", &ChainComplex::kSimplexVertices, py::arg("k"),
           "k-simplices as sorted vertex-id tuples in C_k order (the column "
           "order of d_{k+1} / row order of d_k). k=1 gives the edge ordering "
           "the rows of boundaryMatrix(2) refer to; k=dimension() equals "
           "orientedTopSimplices(). Empty when k is out of range.")
      .def("orientedTopSimplices", &ChainComplex::orientedTopSimplices,
           "Top simplices as sorted vertex-id tuples, in the canonical column "
           "order of the top boundary matrix d_d (d = dimension()); the order "
           "the fundamentalClass() signs refer to. Empty for the empty complex.")
      .def("fundamentalClass", &ChainComplex::fundamentalClass,
           "Fundamental class [W] in H_d: the per top-simplex orientation signs "
           "eps_t = +/-1 (the +/-1 generator of ker d_d) making the top chain a "
           "cycle (d_d applied to the signed top chain is 0). Sign-normalized so "
           "the first nonzero entry is +1. Raises if the complex is not a closed "
           "connected oriented manifold (dim ker d_d != 1) or dimension < 1.")
      .def_static(
          "endSignCovector", &ChainComplex::endSignCovector,
          py::arg("surface_cells"), py::arg("holes"),
          "The end sign covector sigma in {+/-1}^len(holes): the induced-"
          "orientation charge pattern of an end surface, from its fundamental "
          "chain. surface_cells are the end's top cells, holes the removed "
          "cells whose boundary cycles carry the periods; the union is "
          "oriented by sign propagation (component roots = lex-smallest "
          "cells, +1) and sigma_k is the orientation coefficient of "
          "holes[k], so every closed form's signed periods obey "
          "sum_k sigma_k p_k = 0 end by end. Deterministic -- a property of "
          "the end surface, not of any fill or spectrum -- and equivariant "
          "under order-preserving relabelings (e.g. a layer shift). Raises "
          "on mixed-dimension cells, a facet with > 2 cofaces, or a "
          "non-orientable surface.")
      .def_static(
          "orientationCovector", &ChainComplex::orientationCovector,
          py::arg("top_cells"),
          "The induced-orientation covector eps in {+/-1}^len(top_cells): the "
          "per-cell sign from orienting a whole top-cell complex by facet-"
          "sharing propagation (component roots = lex-smallest cells, +1; "
          "across an interior facet the two induced signs cancel). The result "
          "aligns to the sorted-unique (canonical C_d) order of the cells. "
          "Unlike fundamentalClass() it does NOT require closedness (boundary "
          "facets impose nothing), so it reads the orientation of an open "
          "refinement region (a stellar cone star, a CDT slab). Determined "
          "combinatorially, independent of geometry, vertex labels, and input "
          "order. Raises on mixed-dimension cells, a facet with > 2 cofaces, "
          "or a non-orientable propagation contradiction.")
      .def("intersectionForm", &ChainComplex::intersectionForm,
           "Symmetric intersection form on free H^2 (flat b2 x b2), for a closed "
           "oriented 4-manifold; empty if n != 4 or b2 == 0.")
      .def("signature", &ChainComplex::signature,
           "Signature b+ - b- of the intersection form (0 if n != 4 or b2 == 0).")
      .def("stiefelWhitneyNumbers", &ChainComplex::stiefelWhitneyNumbers,
           "Mod-2 Stiefel-Whitney numbers <w_{i1}..w_{ir}, [K]> keyed by "
           "monomial (e.g. 'w4', 'w2^2'); empty for the empty complex. Raises "
           "if a class needs a deferred higher Steenrod cup-i product (#65).");

  // ----- Eigen-backed value objects for the Hodge spectrum (#183) -----
  py::class_<Cochain>(m, "Cochain",
      R"doc(A k-cochain: complex amplitudes over a k-simplex ordering.

An Eigen-backed vector of complex amplitudes together with the degree k and the
k-simplex ordering it indexes (the same HodgeLaplacian / ChainComplex column
order), so its indices are meaningful. simplices()[i] is the sorted vertex-id
tuple of the cell carrying coeffs()[i]; at k=0 each tuple is a single vertex id
(the sorted-id vertex order). Eigen-backed and iTensor-free. The inner product is
Hermitian, np.vdot convention: <a, b> = sum conj(a_i) b_i.)doc")
      .def(py::init<int, std::vector<std::vector<std::uint64_t>>,
                    Eigen::VectorXcd>(),
           py::arg("degree"), py::arg("simplices"), py::arg("coeffs"),
           "A degree-k cochain over simplices (sorted vertex-id tuples, in the "
           "indexing order) carrying coeffs (a 1-D complex array). Raises if "
           "len(simplices) != len(coeffs).")
      .def("degree", &Cochain::degree, "The cochain degree k.")
      .def("size", &Cochain::size,
           "Number of k-cells (= len(coeffs()) = len(simplices())).")
      .def("__len__", &Cochain::size)
      .def("coeffs", &Cochain::coeffs,
           "The complex amplitudes as a 1-D numpy.ndarray (Eigen-backed).")
      .def("simplices", &Cochain::simplices,
           "The k-simplex ordering: simplices()[i] is the sorted vertex-id tuple "
           "of the cell carrying coeffs()[i].")
      .def("amplitude", &Cochain::amplitude, py::arg("index"),
           "Amplitude on the index-th k-cell. Raises IndexError if out of range.")
      .def("__getitem__", &Cochain::amplitude)
      .def("amplitudeFor", &Cochain::amplitudeFor, py::arg("simplex"),
           "Amplitude on the k-cell identified by its sorted vertex-id tuple "
           "(e.g. (vertexId,) at k=0). Raises IndexError if absent.")
      .def("innerProduct", &Cochain::innerProduct, py::arg("other"),
           "The Hermitian inner product <self, other> = sum conj(self_i) other_i "
           "(= np.vdot). Raises if the degrees or orderings differ.")
      .def("norm", &Cochain::norm, "The Euclidean norm sqrt(sum |c_i|^2).")
      .def("normalized", &Cochain::normalized,
           "A copy scaled to unit norm (the cochain itself if its norm is ~0).");

  py::class_<Spectrum>(m, "Spectrum",
      R"doc(The eigendecomposition of a Hodge Laplacian L_k as a value object.

Eigenvalues paired with their eigenvectors-as-Cochains, in matching order
(eigenvalues()[i] is the eigenvalue of eigenvectors()[i]). Eigenvalues are stored
complex to cover both regimes uniformly: in the Hermitian/metric case
(isHermitian() == True) they are real (imag 0) and ascending; in the Lorentzian
(signed-weight d'Alembertian) case they may be negative or complex-conjugate
pairs, sorted by (Re, Im). harmonics(tol) is the kernel subset |lambda| < tol =
ker L_k as Cochains. Supports len() and indexing (spectrum[i] is the i-th
eigenvector Cochain).)doc")
      .def("eigenvalues", &Spectrum::eigenvalues,
           "The eigenvalues as a 1-D complex numpy.ndarray (ascending real, "
           "imag 0, in the Hermitian regime; sorted by (Re, Im) in the Lorentzian "
           "one).")
      .def("eigenvectors", &Spectrum::eigenvectors,
           "The eigenvectors as a list of Cochains, one per eigenvalue.")
      .def("harmonics", &Spectrum::harmonics, py::arg("tol") = 1e-9,
           "The harmonic subset: eigenvectors with |lambda| < tol (a basis for "
           "ker L_k), as a list of Cochains.")
      .def("size", &Spectrum::size, "The number of modes.")
      .def("__len__", &Spectrum::size)
      .def("isHermitian", &Spectrum::isHermitian,
           "Whether the eigenvalues are guaranteed real and ascending (the "
           "metric/self-adjoint regime) vs. the indefinite Lorentzian one.")
      .def("eigenvalue", &Spectrum::eigenvalue, py::arg("i"),
           "The i-th eigenvalue. Raises IndexError if out of range.")
      .def("__getitem__",
           [](const Spectrum &s, std::size_t i) -> const Cochain & { return s[i]; },
           py::return_value_policy::reference_internal,
           "The i-th eigenvector Cochain. Raises IndexError if out of range.");

  // ----- Hodge Laplacian: k=0 Hermitian graph (#90), k>=1 metric Hodge (#104) -----
  py::enum_<HodgeLaplacian::MetricSource>(m, "HodgeMetricSource",
      R"doc(Where a Hodge operator's metric comes from: DiagonalWeights (the historical
per-simplex diagonal weights of HodgeWeightConvention, the process default) or
WhitneyPencil (the chain-level Whitney Hodge pencil of tessera.chainhodge, dressed at
every degree by the edge-phase links; MultiCobordism's default).)doc")
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

ONE definition at every degree, the whitepaper's: with the integer boundary maps
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

The U(1) CONNECTION Laplacian is a DIFFERENT operator (#805). connectionLaplacian
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
             // No-weights overload: read the PROCESS default at call time (the
             // pybind default-argument form would bake it in at import).
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
           "The process-wide default HodgeMetricSource (ships as DiagonalWeights).")
      .def_static("setDefaultMetricSource", &HodgeLaplacian::setDefaultMetricSource,
           py::arg("source"), "Flip the process-wide default metric source ONCE at startup.")
      .def("metricSource", &HodgeLaplacian::metricSource, "This operator's metric source.")
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
           "sigma^2 are exactly the eigenvalues of A. This is the entropy that "
           "can SEE the connection; every L_k is blind to phi.")
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
      // ----- indefinite W-norms of the near-kernel (spec §5.6) -----
      .def("nullNorms", &HodgeLaplacian::nullNorms,
           py::arg("k"), py::arg("tol") = 1e-9, py::arg("metric") = true,
           "Indefinite W-norms <h,h>_W = sum_i W_{k,i} |h_i|^2 of the near-kernel "
           "representatives, one per column of harmonics (same order). "
           "A value ~0 flags a NULL (lightlike) harmonic; all positive on an "
           "all-spacelike complex.");

  // ----- §4b eigenstate synthesis: residual + parameter access (#133) -----
  auto eigenstateSynthesis = py::class_<EigenstateSynthesis>(m, "EigenstateSynthesis",
      R"doc(§4b inverse eigenvector problem on a fixed complex, degree-k.

Scores how close the complex's current Hermitian edge weights make a target
state psi to being an eigenvector of the degree-k Hodge Laplacian L_k (via
HodgeLaplacian), and reads/writes those weights so a search can perturb them.
At k=0 the scored operator is the U(1) CONNECTION graph Laplacian D - A
(connectionLaplacian, the magnitude convention), NOT the Hodge L_0: a
degree-zero register carries U(1) flux and dim ker L_0 is always b_0, so an L_0
readout would be identically gauge-flat (#805). psi is then a vertex vector
(|V|, sorted-id order). At k>=1 L_k is the metric Hodge Laplacian
on k-forms (|C_k|, ChainComplex k-cell order); the tunable parameters stay the
edge squared-lengths, which feed the volume weights W_k of L_k via Simplex.volume
(phases enter only k=0). cellSimplices() gives each psi component's vertex tuple,
so a caller can pin the boundary k-cells to a target form (the #176 k=1
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
(Edge.setSquaredLength; weights() reads Re, not a magnitude — #581) and U(1)
phases {theta_ij} (Edge.setPhase), in a stable edge order fixed at
construction (the weight-carrying edges: both endpoints present, no self-loops).

Fixed-boundary interior fill (§5.0): the tunable edges split into a boundary set
dW (edges on a codim-1 face in exactly one top cell — held fixed) and an interior
set (free). interiorWeights / interiorPhases + setInteriorWeights /
setInteriorPhases read/write only the interior edges, so a search drives r -> 0
for a target output eigenvector while dW stays byte-identical (boundaryEdges()
exposes that fixed set). growInterior() cones a fresh interior vertex via the
boundary-fixed pre-geometric Pachner add (#112), enriching the interior with dW
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
           "negative), and any resident Im l^2 is not reported (#581).")
      .def("phases", &EigenstateSynthesis::phases,
           "Edge phases {theta_ij} (radians) in the stable edge order.")
      .def("setWeights", &EigenstateSynthesis::setWeights, py::arg("w"),
           "Write the edge squared lengths in place as REAL signed values "
           "(l^2 = w + 0i, zeroing any resident Im — the ordinary-Lorentzian "
           "convention, #581). Raises if len(w) != numEdges().")
      .def("setPhases", &EigenstateSynthesis::setPhases, py::arg("theta"),
           "Write the edge phases in place. Raises if len(theta) != numEdges().")
      // ----- Fixed-boundary interior fill (§5.0, #147) -----
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
           "interior-edge order (Re, not magnitudes — #581).")
      .def("interiorPhases", &EigenstateSynthesis::interiorPhases,
           "Interior edge phases {theta_ij} (radians) in interior-edge order.")
      .def("setInteriorWeights", &EigenstateSynthesis::setInteriorWeights,
           py::arg("w"),
           "Write the interior edge squared lengths in place as REAL signed "
           "values (l^2 = w + 0i, zeroing any resident Im — #581); the boundary "
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
           "pre-geometric Pachner add (#112): a 1->(d+1) stellar subdivision that "
           "leaves dW exactly fixed while enriching the interior. Re-captures the "
           "vertex order and interior/boundary partition, so order() grows by one "
           "(extend psi on the new apex, appended last in sorted-id order) and "
           "numInteriorEdges() grows. Returns False if no top cell can be "
           "subdivided (e.g. a 1-complex), leaving the complex unchanged.")
      // ----- Free interior connectivity (general growth primitive, #200) -----
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
      // ----- The carried register read-outs (#286) -----
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
           "1-cells, i.e. vertex pairs) with the SVD pseudo-inverse fit (#589) — "
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
      // ----- The discovered operator: ker L1(W - dW) (#363) -----
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
      // ----- Surgery: the topology-changing interior remove move (#196) -----
      .def("interiorTopCells", &EigenstateSynthesis::interiorTopCells,
           "The interior top cells (all-interior vertices, on no dW face) as "
           "sorted vertex-id tuples — the surgery removal candidates. Removing one "
           "(removeInteriorCell) cannot touch dW, so it is the boundary-fixed "
           "TOPOLOGY-CHANGING move that can open a hole/handle and MOVE b_k, unlike "
           "growInterior's subdivision and the additive attach.")
      .def("removeInteriorCell", &EigenstateSynthesis::removeInteriorCell,
           py::arg("cell"),
           "Surgery (#196): remove the interior top cell `cell` (a tuple from "
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
      // ----- Charge sector: the E/B split of F in Omega^2 (#417) -----
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
           "The discrete Gauss-law charge Q = oint_S E (#411): the temporal-sector "
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

  // The result of fieldStrengthSplit (#417): the E/B partition of F in Omega^2.
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

  // Exact integer / GF(2) / inertia primitives (also exposed for direct
  // testing). Matrices are passed flat row-major with explicit dims.
  py::class_<SmithNormalForm>(m, "SmithNormalForm")
      .def_readonly("rank", &SmithNormalForm::rank)
      .def_readonly("invariant_factors", &SmithNormalForm::invariantFactors);
  m.def("smith_normal_form", &smithNormalForm, py::arg("matrix"), py::arg("rows"),
        py::arg("cols"), "Smith Normal Form (rank + invariant factors) over Z.");
  m.def("integer_rank", &integerRank, py::arg("matrix"), py::arg("rows"), py::arg("cols"),
        "Rank over Q of an integer matrix.");
  m.def("gf2_rank", &gf2Rank, py::arg("matrix"), py::arg("rows"), py::arg("cols"),
        "Rank over GF(2) of a 0/1 matrix.");
  m.def("gf2_nullspace", &gf2Nullspace, py::arg("matrix"), py::arg("rows"),
        py::arg("cols"),
        "Basis of the GF(2) kernel of a 0/1 matrix, as a list of nullity "
        "length-cols vectors (each x with matrix·x == 0 mod 2; independent over "
        "GF(2); nullity == cols - gf2_rank). The cocycles Z1 = ker(d2^T mod 2).");
  m.def("gf2_span", &gf2Span, py::arg("basis"), py::arg("cols"),
        "All 2^k GF(2) combinations of a basis of k length-cols vectors (first "
        "is the zero vector). For a gf2_nullspace basis, the flat Z2 connections.");
  m.def("integer_nullspace", &integerNullspace, py::arg("matrix"),
        py::arg("rows"), py::arg("cols"),
        "Basis of the rational kernel of an integer matrix as INTEGER vectors "
        "(exact Gauss-Jordan over Q; coprime entries; nullity == cols - rank). "
        "Raises OverflowError when the exact elimination would overflow 64-bit "
        "intermediates -- never rounded.");

  py::class_<Inertia>(m, "Inertia")
      .def_readonly("n_pos", &Inertia::nPos)
      .def_readonly("n_neg", &Inertia::nNeg)
      .def_readonly("n_zero", &Inertia::nZero)
      .def("signature", &Inertia::signature);
  m.def("symmetric_inertia", &symmetricInertia, py::arg("matrix"), py::arg("n"),
        py::arg("tol") = 1e-9,
        "Inertia (#pos,#neg,#zero eigenvalues) of a symmetric integer matrix.");

  // ----- Capability A (#65): characteristic numbers -----
  // Scalar invariants are Observables; families come from CharacteristicNumbers.
  py::class_<EulerCharacteristic, std::shared_ptr<EulerCharacteristic>>(
      m, "EulerCharacteristic",
      "Observable: Euler characteristic chi = sum_k (-1)^k |C_k|.")
      .def(py::init<>())
      .def("compute", &EulerCharacteristic::compute, py::arg("spacetime"));
  // Qualified: tessera::spacetime::Signature (metric signature) is also in
  // scope via the using-directives.
  py::class_<cobordism::Signature, std::shared_ptr<cobordism::Signature>>(
      m, "Signature",
      "Observable: signature b+ - b- of the H_2 intersection form (closed "
      "oriented 4-manifold; 0 if n != 4 or b2 == 0).")
      .def(py::init<>())
      .def("compute", &cobordism::Signature::compute, py::arg("spacetime"));

  py::class_<CharacteristicNumbers>(m, "CharacteristicNumbers",
      "Topological invariants of a closed PL n-manifold: Euler characteristic, "
      "signature (4-manifolds), Stiefel-Whitney numbers (pending), and "
      "Pontryagin numbers (4-manifolds: p1 = 3*signature).")
      .def_readonly("euler", &CharacteristicNumbers::euler,
                    "Euler characteristic (alternating count of cells).")
      .def_readonly("signature", &CharacteristicNumbers::signature,
                    "Signature of the intersection form; None unless an "
                    "orientable 4-manifold.")
      .def_readonly("stiefel_whitney_numbers",
                    &CharacteristicNumbers::stiefelWhitneyNumbers,
                    "Mod-2 Stiefel-Whitney numbers, keyed by monomial (pending; "
                    "currently empty).")
      .def_readonly("pontryagin_numbers",
                    &CharacteristicNumbers::pontryaginNumbers,
                    "Pontryagin numbers (4-manifolds: {'p1': 3*signature}).")
      .def_static("of", &CharacteristicNumbers::of, py::arg("spacetime"),
                  py::arg("oriented") = true,
                  "Compute the characteristic numbers of the given manifold.");


  // === MultiCobordism (#491): the C++ source-of-truth fully-emergent merge
  // optimizer — emergent topology at a user-defined degree k. ===

  py::class_<BoundaryFiber>(m, "BoundaryFiber",
      R"doc(The fiber form of a boundary block's target (#916): a retained fiber on the
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
      "A glued pencil (#916): the union of cobordisms' top cells with one geometry, "
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
      "A pencil level whose interface coordinates are retained fibers (#916): the Feshbach "
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
      R"doc(Continuation of a relaxed cobordism's boundary fibers into the next pencil level
(#916): exact assembly of cobordisms along shared cells (one epsilon per assembly), boundary
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

  py::class_<MultiCobordism::BoundaryBlock>(m, "MultiCobordismBlock",
      "An emergent boundary block of a MultiCobordism (an input or output): the "
      "vertex set whose own sub-complex carries the block, and its target period "
      "vector. Read the block's sub-complex with Spacetime.fromCells over the "
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
                             "The fiber form of the target (#916), or None.");
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
      "chi on the pair of attached frames and the reading flag (#941).")
      .def(py::init<>())
      .def_readwrite("chi", &MultiCobordism::TwoBodyTarget::chi)
      .def_readwrite("choi_decomposed", &MultiCobordism::TwoBodyTarget::choiDecomposed);
  py::class_<MultiCobordism::TwoBodyRead>(m, "TwoBodyRead",
      "The reading of the bulk between two attached input frames (#941): the frame transfer "
      "T_AB (operator reading), vec(T_AB) (Choi-decomposed state reading), its Schmidt spectrum "
      "and rank, the reversal residual, the fit residual, and the input blocks' fiber residuals.")
      .def_readonly("choi_decomposed", &MultiCobordism::TwoBodyRead::choiDecomposed)
      .def_readonly("transfer", &MultiCobordism::TwoBodyRead::transfer)
      .def_readonly("choi_state", &MultiCobordism::TwoBodyRead::choiState)
      .def_readonly("singular_values", &MultiCobordism::TwoBodyRead::singularValues)
      .def_readonly("schmidt_rank", &MultiCobordism::TwoBodyRead::schmidtRank)
      .def_readonly("reversal_residual", &MultiCobordism::TwoBodyRead::reversalResidual)
      .def_readonly("residual", &MultiCobordism::TwoBodyRead::residual)
      .def_readonly("input_fiber_residuals", &MultiCobordism::TwoBodyRead::inputFiberResiduals)
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
      "The fully-emergent MultiCobordism merge optimizer (#491): merge as a "
      "fully emergent optimization. From a bare host it grows the register by "
      "gated surgical moves under an explicitly selected Legacy, joint "
      "Regge-Hodge-stationarity, or mediated-correspondence objective at a "
      "USER-DEFINED degree k (degrees), reading holes "
      "dynamically off getBoundary. Two stages: run_stage1 (combinatorial), "
      "run_stage2 (geometric) -- or run(), which interleaves both updates in one "
      "loop. An EMPTY output_targets list is supported (#555): "
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
      // Explicit metric source. Without it the C++ default argument reads the
      // process-wide HodgeLaplacian.defaultMetricSource() at CALL time (a
      // pybind11 default would capture it at import).
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
      .def("use_fiber_residuals", &MultiCobordism::useFiberResiduals, py::arg("enabled"),
           "Score blocks carrying a fiber-form target by the fiber residual (least-squares leak of the "
           "target images in the band read on the block's own pencil, restricted to the fiber's cells) "
           "instead of the period residual; folded into r_U so both stages descend it. Off by default.")
      .def("uses_fiber_residuals", &MultiCobordism::usesFiberResiduals)
      .def("set_fiber_phase_descent", &MultiCobordism::setFiberPhaseDescent, py::arg("enabled"),
           "Also descend the degree-0 link phases through the analytic fiber gradient (off by default).")
      .def("fiber_phase_descent", &MultiCobordism::fiberPhaseDescent)
      .def("fiber_residual_for_input_block", &MultiCobordism::fiberResidualForInputBlock, py::arg("index"),
           py::call_guard<py::gil_scoped_release>(),
           "The fiber residual of one input block on the live complex.")
      .def("fiber_residual_gradient",
           [](const MultiCobordism &self, const BoundaryFiber &fiber) {
             py::gil_scoped_release release;
             const auto g = self.fiberResidualGradientOn(self.spacetime(), fiber);
             return std::make_pair(g.lengths, g.phases);
           }, py::arg("fiber"),
           "Analytic gradient of the fiber residual of `fiber` on the live complex (#947): (lengths, phases) "
           "in EdgeList order, each entry (d/dRe, d/dIm) packed as a complex number; phases empty above degree 0.")
      .def("two_body_residual_gradient",
           [](const MultiCobordism &self) {
             if (!self.twoBodyTarget()) throw std::logic_error("no two-body target");
             py::gil_scoped_release release;
             const auto g = self.twoBodyResidualGradientOn(self.spacetime(), *self.twoBodyTarget());
             return std::make_pair(g.lengths, g.phases);
           }, "Analytic gradient of the two-body residual on the live complex (#947).")
      .def("fiber_mode_ascent",
           [](const MultiCobordism &self) {
             py::gil_scoped_release release;
             const auto g = self.fiberModeAscent();
             return std::make_pair(g.lengths, g.phases);
           }, "The analytic ascent of every fiber-mode term of r_U on the live complex (#947).")
      .def("set_input_block_region", &MultiCobordism::setInputBlockRegion, py::arg("index"), py::arg("vertices"),
           "Set an input block's region explicitly (the attached fiber's cells must lie inside).")
      .def("attach_input_fiber", &MultiCobordism::attachInputFiber, py::arg("index"), py::arg("fiber"),
           py::arg("cells"),
           "Attach a piped input fiber to THIS complex's cells (one per fiber row, in the attachment "
           "order = the attachment permutation); refuses overlaps with other attached input fibers.")
      .def("set_two_body_target", &MultiCobordism::setTwoBodyTarget, py::arg("chi"),
           py::arg("choi_decomposed") = true,
           "The two-body target chi on the pair of attached frames; choi_decomposed selects the reading "
           "(vec(T_AB) as a state on the pair space, or the operator T_AB). Scored inside r_U under "
           "use_fiber_residuals once two input fibers are attached.")
      .def("two_body_target", &MultiCobordism::twoBodyTarget)
      .def("two_body_residual", &MultiCobordism::twoBodyResidual, py::call_guard<py::gil_scoped_release>(),
           "The projective Frobenius leak of chi against the frame transfer T_AB on the live complex.")
      .def("read_two_body", &MultiCobordism::readTwoBody, py::call_guard<py::gil_scoped_release>(),
           "The bulk between the two attached frames: T_AB, vec(T_AB), Schmidt spectrum and rank, the "
           "reversal residual, the fit residual, and both input blocks' fiber residuals.")
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
      .def("set_input_fiber", &MultiCobordism::setInputFiber, py::arg("index"), py::arg("fiber"),
           "Attach the fiber form of an input block's target (#916).")
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
      .def("seed_inputs", &MultiCobordism::seedInputs, py::arg("seeds"))
      .def("seed_outputs", &MultiCobordism::seedOutputs, py::arg("seeds"))
      // Long pure-C++ compute: release the GIL for the duration so a background thread can
      // drive a pass (a single call, per the register-growth constraint) without blocking the
      // main thread -- e.g. multicobordism_animation.py --live keeps its GUI responsive.
      .def("run_stage1", &MultiCobordism::runStage1, py::arg("max_steps") = 200,
           py::arg("n_candidate_moves") = 12, py::arg("grow_boundaries") = false,
           py::arg("max_lookahead") = 1,
           py::call_guard<py::gil_scoped_release>(),
           "max_lookahead: when a batch of single moves finds no improvement, "
           "the search deepens iteratively -- 2-move sequences, then 3, up to "
           "this many moves -- committing an F-lowering sequence as a whole "
           "(1 = single moves only).")
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
           "Returns the combined F trace.")
      .def_property_readonly("should_propose_dispositions",
                             &MultiCobordism::shouldProposeDispositions,
           "Whether the stage-1 move draw also proposes CAUSAL DISPOSITIONS "
           "(#613) -- a timelike cone-in and a disposition flip on an existing "
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
  py::enum_<MultiCobordism::BuildAction>(multiCobordismClass, "BuildAction",
      "One canonical solve action a search policy (Proton's build restart loop, a greedy "
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
           // Composes run_stage1/run_stage2 internally (C++ -> C++, so no nested guard); release
           // the GIL here too so a background thread driving the build stays off the main thread.
           py::call_guard<py::gil_scoped_release>(),
           "Apply one BuildAction to this node in place (GROW/EVOLVE = run_stage1 with "
           "grow_boundaries true/false; RELAX = run_stage2; CONE_OUT/CONE_IN = the directed "
           "probes) -- the canonical solve step a policy (build, greedy, or RL) composes.")
      .def("directed_cone_out", &MultiCobordism::directedConeOut,
           py::arg("strategy") = MultiCobordism::HolePlacementStrategy::AdjacentHolesLast,
           py::arg("max_open") = 6,
           "Directed gated cone-out: deliberately remove top cells, keeping the opener "
           "that most lowers this node's rU (which absorbs r_state). Returns #holes opened.")
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

  // === #776: modes, the enumerable objective, refinement, and the overlay ===
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

  py::class_<MultiCobordism::ObjectiveTerms>(multiCobordismClass, "ObjectiveTerms",
      "The COMPLETE, enumerable term list the scalar objective is the sum of. "
      "MultiCobordism.objective_of is static over this record, so the objective "
      "provably reads nothing else -- the structural half of the no-feedback "
      "firewall.")
      .def(py::init<>())
      .def_readwrite("regge_stationarity",
                     &MultiCobordism::ObjectiveTerms::reggeStationarity)
      .def_readwrite("hodge_stationarity",
                     &MultiCobordism::ObjectiveTerms::hodgeStationarity)
      .def_readwrite("connection_stationarity",
                     &MultiCobordism::ObjectiveTerms::connectionStationarity,
                     "eta_C ||grad_phi S||^2 of the C* connection operator -- "
                     "the ONLY term with a gradient in the connection phase. "
                     "Every L_k is blind to phi, so without this the phase is "
                     "a declared field no update can move.")
      .def_readwrite("register_residual",
                     &MultiCobordism::ObjectiveTerms::registerResidual)
      .def_readwrite("action_magnitude",
                     &MultiCobordism::ObjectiveTerms::actionMagnitude)
      .def_readwrite("carried_state_energy",
                     &MultiCobordism::ObjectiveTerms::carriedStateEnergy);

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

  py::class_<HodgeDegreeContribution>(m, "HodgeDegreeContribution",
      "One degree's share of the Hodge stationarity term, so a reader can tell "
      "WHICH degree the descent came from rather than only the total.")
      .def(py::init<>())
      .def_readwrite("degree", &HodgeDegreeContribution::degree,
                     "The Laplacian degree k.")
      .def_readwrite("weight", &HodgeDegreeContribution::weight,
                     "The declared weight on this degree.")
      .def_readwrite("gradient_norm_squared",
                     &HodgeDegreeContribution::gradientNormSquared,
                     "||grad_z S_k||^2 over the edges in scope, UNWEIGHTED, so "
                     "the raw spread across degrees is visible rather than "
                     "folded into the weighting.")
      .def_readwrite("contribution", &HodgeDegreeContribution::contribution,
                     "This degree's share of hodge_stationarity: the entropy "
                     "weight times the degree weight times the norm. Summing "
                     "this over the contributions reproduces the term to "
                     "double round-off -- the term applies the entropy weight "
                     "once to the accumulated weighted norms, whereas each "
                     "share here carries its own multiply.");

  py::class_<ObjectiveScope>(m, "ObjectiveScope",
      "What an objective DECLARES that it references: a named pinned region, "
      "or -- by declaring nothing -- the whole cobordism. Independent of "
      "whether that region's coordinates are frozen; a pinned edge does not "
      "vary but is still scored.")
      .def(py::init<>())
      .def_readwrite("region", &ObjectiveScope::region,
                     "The region referenced, as a handle obtainable only from "
                     "MultiCobordism.region_handle. Default means the whole "
                     "cobordism.")
      .def_readwrite("includes_straddling_edges",
                     &ObjectiveScope::includesStraddlingEdges,
                     "Whether edges with a single endpoint in the region enter "
                     "the score. Meaningless for a whole-cobordism scope.")
      .def("is_whole_cobordism", &ObjectiveScope::isWholeCobordism,
           "Whether nothing was declared, i.e. the scope is everything.");

  py::class_<RegionHandle>(m, "RegionHandle",
      "A reference to a DECLARED pinned region. A caller cannot fabricate one: "
      "the only non-empty handle comes from MultiCobordism.region_handle, "
      "which throws BY NAME on an undeclared region rather than silently "
      "matching nothing.")
      .def(py::init<>())
      .def("is_whole_cobordism", &RegionHandle::isWholeCobordism)
      .def("name", &RegionHandle::name)
      .def("__eq__", &RegionHandle::operator==, py::is_operator());

  py::class_<ObjectiveName>(m, "ObjectiveName",
      "The identifiers objectives are known by, as named constants rather "
      "than literals repeated at each site.")
      .def_readonly_static("JOINT_STATIONARITY",
                           &ObjectiveName::kJointStationarity)
      .def_readonly_static("LEGACY", &ObjectiveName::kLegacy)
      .def_readonly_static("MEDIATED_CORRESPONDENCE",
                           &ObjectiveName::kMediatedCorrespondence);

  py::class_<ObjectiveTermName>(m, "ObjectiveTermName",
      "The declared term slots, named so the list and the constants cannot "
      "drift apart.")
      .def_readonly_static("REGGE_STATIONARITY",
                           &ObjectiveTermName::kReggeStationarity)
      .def_readonly_static("HODGE_STATIONARITY",
                           &ObjectiveTermName::kHodgeStationarity)
      .def_readonly_static("CONNECTION_STATIONARITY",
                           &ObjectiveTermName::kConnectionStationarity)
      .def_readonly_static("REGISTER_RESIDUAL",
                           &ObjectiveTermName::kRegisterResidual)
      .def_readonly_static("ACTION_MAGNITUDE",
                           &ObjectiveTermName::kActionMagnitude)
      .def_readonly_static("CARRIED_STATE_ENERGY",
                           &ObjectiveTermName::kCarriedStateEnergy);

  py::class_<ObjectiveContext>(m, "ObjectiveContext",
      "The COMPLETE set of inputs an objective may read -- the no-feedback "
      "firewall restated as an input type. Plain data: geometry, a region, "
      "that region's declared targets, configured weights, and precomputed "
      "geometric scalars. No MultiCobordism reference and deliberately no "
      "callable, since a bound callable would capture the node and smuggle "
      "back the reachability the former static objective_of denied.")
      .def(py::init<>())
      // Every field is readable, and every one is data. A Python objective
      // must be able to read what it scores; what it must NOT be able to read
      // is an analysis product, and none is here to read.
      .def_readwrite("spacetime", &ObjectiveContext::spacetime,
                     "The complex being scored.")
      .def_readwrite("region", &ObjectiveContext::region,
                     "The vertex set this objective is scored over. EMPTY "
                     "means the whole complex.")
      .def_readwrite("scored_edges", &ObjectiveContext::scoredEdges,
                     "The edge INDICES this objective's sums run over, "
                     "resolved by the engine from the objective's declared "
                     "scope. None means every edge -- the whole cobordism. An "
                     "empty list is a different thing: it means score nothing, "
                     "which is what a region with no interior edge and the "
                     "straddling edges declared out comes to. Conflating the "
                     "two would silently promote such a region to scoring the "
                     "entire complex.")
      .def_readwrite("region_targets", &ObjectiveContext::regionTargets,
                     "The target states the region is scored against. Empty "
                     "for a purely geometric objective.")
      .def_readwrite("register_degrees", &ObjectiveContext::registerDegrees,
                     "The register degrees the objective is declared over.")
      .def_readwrite("hodge_degrees", &ObjectiveContext::hodgeDegrees,
                     "The Laplacian degrees k the Hodge entropy term is summed "
                     "over. NOT a register concept: each entry selects which "
                     "L_k the entropy is taken of, and nothing else. Never "
                     "read from register_degrees, not even as a fallback. "
                     "Defaults to [0].")
      .def_readwrite("hodge_degree_weights",
                     &ObjectiveContext::hodgeDegreeWeights,
                     "The weight on each entry of hodge_degrees, positionally. "
                     "Empty means uniform 1.")
      .def_readwrite("regge_weight", &ObjectiveContext::reggeWeight)
      .def_readwrite("hodge_entropy_weight",
                     &ObjectiveContext::hodgeEntropyWeight)
      .def_readwrite("connection_entropy_weight",
                     &ObjectiveContext::connectionEntropyWeight,
                     "eta_C, the connection-entropy stationarity weight. Zero "
                     "by default: an objective acquires a phi gradient only "
                     "when a caller declares one.")
      .def_readwrite("gamma", &ObjectiveContext::gamma)
      .def_readwrite("carried_state_energy_weight",
                     &ObjectiveContext::carriedStateEnergyWeight)
      .def_readwrite("einstein_hilbert", &ObjectiveContext::einsteinHilbert)
      .def_readwrite("hodge_entropy_phase_mode",
                     &ObjectiveContext::hodgeEntropyPhaseMode,
                     "Which entropy the Hodge term reads: the complex "
                     "operator or its phase-blind entrywise ablation.")
      .def_readwrite("register_residual", &ObjectiveContext::registerResidual,
                     "r_U on this region, precomputed by the engine and passed "
                     "as a NUMBER rather than a callable, so no node is "
                     "reachable from here. NaN when the objective did not ask "
                     "for it -- never a silent zero.")
      .def_readwrite("carried_state_energy",
                     &ObjectiveContext::carriedStateEnergy,
                     "E_carried(Gamma, g), likewise a precomputed number.")
      .def_static("input_names", &ObjectiveContext::inputNames,
                  "Every field of the context, in declaration order -- the "
                  "firewall list a structural test asserts against.");

  py::class_<ObjectiveDirection>(m, "ObjectiveDirection",
      "A stage-2 search direction together with the exact objective value at "
      "the point it was taken from.")
      .def(py::init<>())
      .def_readwrite("ascent", &ObjectiveDirection::ascent,
                     "The ascent displacement. Stage 2 subtracts a scaled "
                     "multiple of it.")
      .def_readwrite("phase_ascent", &ObjectiveDirection::phaseAscent,
                     "The ascent displacement in the CONNECTION PHASE, same "
                     "edge order. Empty when the objective has no phi "
                     "dependence, which is every functional of L_k alone. "
                     "Stage 2 subtracts it from the stored phases under the "
                     "same line search and step scale that move z.")
      .def_readwrite("baseline", &ObjectiveDirection::baseline,
                     "The exact objective at the current point, when the "
                     "direction's assembly already produced it.")
      .def_readwrite("baseline_computed", &ObjectiveDirection::baselineComputed,
                     "Whether `baseline` is meaningful. False makes the engine "
                     "evaluate the scalar itself rather than trust an "
                     "accumulated trace.");

  py::class_<ObjectiveDirectionContext>(m, "ObjectiveDirectionContext",
      "ObjectiveContext plus the extra data a stage-2 direction needs. Plain "
      "data for the same reason, so the direction path cannot reach a node "
      "either.")
      .def(py::init<>())
      .def_readwrite("scalar", &ObjectiveDirectionContext::scalar,
                     "The scalar inputs, unchanged.")
      .def_readwrite("edge_count", &ObjectiveDirectionContext::edgeCount,
                     "The number of edge coordinates the direction is taken "
                     "over.")
      .def_readwrite("carried_state_energy_gradient",
                     &ObjectiveDirectionContext::carriedStateEnergyGradient,
                     "dE_carried/dz, exact and analytic, computed by the "
                     "engine. Empty where the carried-state weight is zero.");

  py::class_<CobordismObjective, PyCobordismObjective,
             std::shared_ptr<CobordismObjective>>(
      m, "CobordismObjective",
      "The functional MultiCobordism descends, as an injected specification "
      "rather than a value of a closed enum. An objective is scored over a "
      "REGION rather than implicitly over a whole node, so more than one may "
      "coexist on one complex.\n\n"
      "Subclass it in Python to descend a functional of your own: override "
      "name, term_names, terms, direction and is_target_conditioned; scope, "
      "needs_register_residual and numerical_register_residual_weight have "
      "defaults. A subclass reads only the ObjectiveContext it is handed, "
      "which is the same firewall a C++ objective sits behind.")
      .def(py::init<>())
      .def("name", &CobordismObjective::name)
      .def("term_names", &CobordismObjective::termNames)
      .def("terms", &CobordismObjective::terms, py::arg("context"))
      .def("direction", &CobordismObjective::direction, py::arg("context"),
           "The stage-2 search direction over the context's region.")
      .def("is_target_conditioned", &CobordismObjective::isTargetConditioned)
      .def("needs_register_residual",
           &CobordismObjective::needsRegisterResidual)
      .def("numerical_register_residual_weight",
           &CobordismObjective::numericalRegisterResidualWeight,
           py::arg("context"),
           "The weight this objective puts on a NUMERICALLY differentiated "
           "register-residual direction. A weight rather than a callable on "
           "purpose: handing an objective something that could difference the "
           "scalar would mean handing it a closure over the node.")
      .def("hodge_degree_contributions",
           &CobordismObjective::hodgeDegreeContributions, py::arg("context"),
           "This objective's Hodge stationarity term broken down by degree, or "
           "an empty list for an objective with no such term. Reported "
           "alongside the term record rather than inside it: ObjectiveTerms is "
           "a fixed record of scalars that `total` is static over, and a "
           "per-degree breakdown decomposes one of those scalars rather than "
           "adding a term.")
      .def("scope", &CobordismObjective::scope)
      .def("set_scope", &CobordismObjective::setScope, py::arg("scope"),
           "Declare what this objective references. Scope is a property of the "
           "INSTANCE, so an existing objective can be pointed at a region "
           "without writing a new type. Default-constructed means the whole "
           "cobordism.")
      .def_static("total", &CobordismObjective::total, py::arg("terms"),
                  "The scalar: the plain sum of the declared terms. STATIC by "
                  "design -- no `this`, so it cannot reach any state at all.")
      .def_static("declared_term_names",
                  &CobordismObjective::declaredTermNames);

  py::class_<JointStationarityObjective, CobordismObjective,
             std::shared_ptr<JointStationarityObjective>>(
      m, "JointStationarityObjective",
      "beta_R ||grad_z S_Regge||^2 + eta_H sum_k ||grad_z S_Hodge,k||^2 -- the "
      "objective the whitepaper describes, and the only built-in that is not "
      "target-conditioned.")
      .def(py::init<>());

  py::class_<LegacyObjective, CobordismObjective,
             std::shared_ptr<LegacyObjective>>(m, "LegacyObjective",
      "beta_R ||grad_z S_Regge||^2 + gamma r_U -- the compatibility objective. "
      "Target-conditioned through r_U.")
      .def(py::init<>());

  py::class_<MediatedCorrespondenceObjective, CobordismObjective,
             std::shared_ptr<MediatedCorrespondenceObjective>>(
      m, "MediatedCorrespondenceObjective",
      "r_U + beta |S_Regge(W*)| -- the historical operator-cobordism "
      "experiment. Target-conditioned through r_U.")
      .def(py::init<>());

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
      .def("carried_state_energy", &MultiCobordism::carriedStateEnergy, py::arg("st"),
           "E_carried(Gamma, g) = Re tr(Gamma_S h_S(g)) with h_S the Hermitian "
           "part of the metric Hodge operator at the carried degree, restricted "
           "to the carried modes' cells.")
      .def("carried_state_energy_gradient",
           &MultiCobordism::carriedStateEnergyGradient, py::arg("st"),
           "Exact analytic dE/dz per edge in getEdgeList() order.")
      .def("carried_state_purity_defect",
           &MultiCobordism::carriedStatePurityDefect,
           "The #780 purity defect ||Gamma^2 - Gamma||_F of the carried "
           "covariance (NaN with no carried state).")
      .def("carried_state_purity_holds",
           &MultiCobordism::carriedStatePurityHolds, py::arg("tolerance") = 1e-9,
           "Whether the #780 purity certificate HOLDS at the tolerance.")
      .def("set_mean_field_schedule", &MultiCobordism::setMeanFieldSchedule,
           py::arg("dt"), py::arg("steps"),
           "The checkpointed mean-field update schedule.")
      .def_property_readonly("mean_field_step_size",
                             &MultiCobordism::meanFieldStepSize)
      .def_property_readonly("mean_field_steps", &MultiCobordism::meanFieldSteps)
      .def("advance_carried_state", &MultiCobordism::advanceCarriedState,
           py::call_guard<py::gil_scoped_release>(),
           "Advance the carried covariance through #780's meanFieldEvolve under "
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

  // === CobordismDAG (#491): chain emergent merges, output -> input ===
  py::class_<CobordismDAG>(m, "CobordismDAG",
      "Chain emergent merges (MultiCobordism) into a DAG: the output of one "
      "cobordism is an input to the next (the proton_merge_sequence compose, "
      "generalized). add_node returns a node id; edges pipe upstream outputs into "
      "downstream input slots; run() executes in topological order, recording each "
      "node's output (its verified output_target) and realizability residual r_U.")
      .def(py::init<>())
      .def("add_node", &CobordismDAG::addNode, py::arg("host"),
           py::arg("literal_inputs"), py::arg("upstream"),
           py::arg("output_targets"), py::arg("degrees") = std::vector<int>{3},
           py::arg("gamma") = 1.0, py::arg("seed") = 0,
           "Add a node (one co-optimized MultiCobordism system): a bare host, "
           "literal input targets, `upstream` as (node_id, output_index) tuples "
           "whose outputs feed further inputs, and `output_targets` (one for a "
           "merge, two for a 2->2 recombination). Returns the node id.")
      .def("run", &CobordismDAG::run, py::arg("stage1_max_steps") = 30,
           py::arg("stage1_candidate_moves") = 8,
           py::arg("stage2_beta") = 1.0, py::arg("stage2_max_iters") = 40,
           "Run all nodes in topological order (raises on a cycle).")
      .def("output", &CobordismDAG::output, py::arg("node"),
           py::arg("output_index") = 0)
      .def("num_outputs", &CobordismDAG::numOutputs, py::arg("node"))
      .def("residual", &CobordismDAG::residual, py::arg("node"))
      .def("set_fiber_piping", &CobordismDAG::setFiberPiping, py::arg("enabled"), py::arg("degree") = 1,
           py::arg("score_blocks_by_fiber") = false,
           "Pipe each node's output fibers (#916) into the downstream input blocks the edges name.")
      .def("fiber_piping", &CobordismDAG::fiberPiping)
      .def("scores_blocks_by_fiber", &CobordismDAG::scoresBlocksByFiber)
      .def("set_input_attachment", &CobordismDAG::setInputAttachment, py::arg("node"), py::arg("slot"),
           py::arg("cells"), "Cells of the node's own complex that the piped fiber of input slot attaches to.")
      .def("set_two_body_target", &CobordismDAG::setTwoBodyTarget, py::arg("node"), py::arg("chi"),
           py::arg("choi_decomposed") = true)
      .def("two_body_read", &CobordismDAG::twoBodyRead, py::arg("node"), py::return_value_policy::copy)
      .def("has_two_body_read", &CobordismDAG::hasTwoBodyRead, py::arg("node"))
      .def("output_fiber", &CobordismDAG::outputFiber, py::arg("node"), py::arg("output_index"),
           py::return_value_policy::copy)
      .def("has_output_fiber", &CobordismDAG::hasOutputFiber, py::arg("node"), py::arg("output_index"))
      .def("fiber_refusal", &CobordismDAG::fiberRefusal, py::arg("node"))
      .def("piped_input_count", &CobordismDAG::pipedInputCount, py::arg("node"))
      .def("__len__", &CobordismDAG::size);

  // === Proton (#503): the canonical two-step MultiCobordism proton build ===
  auto protonClass = py::class_<Proton>(m, "Proton",
      R"doc(The canonical, footgun-free proton builder, composing MultiCobordism.

A proton is THREE quarks in a colorless bound state, so it is built in TWO steps
(a single merge would be physically invalid). omega = exp(2*pi*i/3).
  * Step A (recombination, one 2->2 node): two neutral q-qbar pairs {1,-1,0},
    {1,0,-1} -> a colored diquark {1,w} + antidiquark {1,w*w} (2-vectors).
  * Step B (formation, a separate 2->1 node): the diquark {1,w} + the third
    quark {w*w} -> the proton {1,w,w*w} (the 3-vector color singlet).
build() builds the closed-S^4 hosts internally and restarts across distinct
seeds until step B's proton block carries the singlet on >=3 emergent holes. The
accessors lazily trigger build() on first use, so `Proton().block()` just works.
Observable readers (charge/mass/radius/spin) read OFF block() in their own
tickets.)doc");
  protonClass
      .def(py::init<std::uint64_t, int, double, double, int, bool, bool, bool,
                    bool, bool, bool>(),
           py::arg("seed") = 0,
           py::arg("register_degree") = 3, py::arg("gamma") = 50.0,
           py::arg("input_weight") = 20.0, py::arg("precone") = 0,
           py::arg("should_use_directed_surgery") = false,
           py::arg("precone_timelike") = false,
           py::arg("precone_alternate") = false,
           py::arg("balanced_edges") = false,
           py::arg("singular_value_ratio") = false,
           py::arg("einstein_hilbert") = true)
      .def_static("omega", &Proton::omega, "omega = exp(2*pi*i/3).")
      .def_static("singlet", &Proton::singlet,
                  "The proton color singlet {1, w, w*w}.")
      .def("build", &Proton::build, py::arg("max_restarts") = 16,
           py::arg("init_steps") = 180,
           py::arg("evolve_steps") = 60, py::arg("stage1_candidate_moves") = 8,
           py::arg("stage2_beta") = 1.0,
           py::arg("stage2_max_iters") = 10, py::arg("color_tolerance") = 0.5,
           py::arg("min_emergent_holes") = 3,
           "Restart across seeds until the whole step-B cobordism carries the singlet "
           "on >= min_emergent_holes emergent holes. Each step runs an init pass (grow the "
           "boundary until it carries) then an evolution pass (boundary frozen).")
      .def("recombination_node", &Proton::recombinationNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) Step A node: two neutral q-qbar pairs -> a "
           "diquark {1,w} + antidiquark {1,w*w}, on a single Delta^4 seed. Drive it with "
           "run_stage1/run_stage2 -- the exact node build() uses for recombination.")
      .def("formation_node", &Proton::formationNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) Step B node: the diquark {1,w} + the third "
           "quark {w*w} -> the proton singlet, on a single Delta^4 seed (output read off "
           "the whole). Drive it with run_stage1/run_stage2.")
      .def("direct_node", &Proton::directNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) ONE-STEP node (6->1): the three bare quarks "
           "{1}, {w}, {w*w} and their three anti-quarks (the elementwise conjugates -- "
           "three q-qbar pairs) as inputs, and the proton singlet as the single output, "
           "read off the WHOLE cobordism (the anti-baryon partner emerges unpinned), on "
           "a single Delta^4 seed -- the experimental single-merge alternative to the "
           "two-step build. Drive it with run().")
      .def("build_direct", &Proton::buildDirect, py::arg("max_restarts") = 16,
           py::arg("init_steps") = 180, py::arg("evolve_steps") = 60,
           py::arg("stage1_candidate_moves") = 8, py::arg("stage2_beta") = 1.0,
           py::arg("color_tolerance") = 0.5, py::arg("min_emergent_holes") = 3,
           py::call_guard<py::gil_scoped_release>(),
           "EXPERIMENTAL one-step build: drive direct_node (three q-qbar pairs in, the "
           "singlet out) with the combined run() drive -- stage-1 surgery and stage-2 "
           "relaxation interleaved in one loop -- as an init pass then an evolution "
           "pass, restarting across seeds. Populates the same accessors as build() "
           "(diquark_residual stays 0 -- no step A). Shares build()'s once-only latch: "
           "call it BEFORE any accessor triggers the lazy two-step build().")
      .def("converged", &Proton::converged,
           "True iff step B's proton block carries the singlet on enough emergent holes.")
      .def("seed", &Proton::seed, "Base seed of the converged (or best) attempt.")
      .def("spacetime", &Proton::spacetime,
           "Step B's full relaxed closed-S^4 complex.")
      .def("block", &Proton::block,
           "Step B's proton sub-complex, with the relaxed metric copied in.")
      .def("emergent_holes", &Proton::emergentHoles,
           "The emergent holes on the proton block over which the singlet periods are "
           "read (>=3 when converged). A topological observable, not a quark count.")
      .def("color_residual", &Proton::colorResidual,
           "Step B's proton singlet r_state (~0 => carried).")
      .def("diquark_residual", &Proton::diquarkResidual,
           "Step A's r_U (small => the diquark recombination converged).");

  // === ProtonIngredients (#555): the emergent arm — nothing pinned downstream ===
  py::class_<ProtonIngredients>(m, "ProtonIngredients",
      R"doc(The emergent arm of the proton build (#555). Proton is the canonical line
in the sand and is composed here unchanged; ProtonIngredients prepares the same
ingredients through the same two-step drive EXCEPT that the final state is never
pinned: step B's output-target list is EMPTY, so the objective is
F = ||grad S||^2 + gamma * sum_i r_U(input_i) and whatever the whole cobordism
comes to carry is READ afterwards, never driven. Exactly one variable differs
from Proton.build() (the singlet output target), so the two classes form a clean
A/B experiment. The seed stays uniform and all-spacelike by design: at
initialization no time has passed — causal structure marks sequences of events
and may only emerge. Convergence carries no answer-shaped gate: an attempt
converges iff it is STATIONARY (stage 2 stopped on its stationarity test) and
PERSISTENT (a continued evolve+relax pass leaves holes, b_k, and F stable).
Everything physical is a post-hoc observable, including the singlet residual —
a diagnostic for comparing against the canonical build's carried level.)doc")
      .def(py::init<std::uint64_t, int, double, double, int, bool>(),
           py::arg("seed") = 0, py::arg("register_degree") = 3,
           py::arg("gamma") = 50.0, py::arg("input_weight") = 20.0,
           py::arg("precone") = 0, py::arg("should_use_directed_surgery") = false)
      .def("build", &ProtonIngredients::build, py::arg("max_restarts") = 16,
           py::arg("init_steps") = 180, py::arg("evolve_steps") = 60,
           py::arg("stage1_candidate_moves") = 8,
           py::arg("stage2_beta") = 1.0, py::arg("stage2_max_iters") = 10,
           py::arg("persist_tolerance") = 0.05,
           "Restart across seeds until an attempt is stationary AND persistent (no "
           "color tolerance, no minimum hole count); otherwise keep the lowest-F "
           "attempt. Same drive per node as Proton.build().")
      .def("recombination_node", &ProtonIngredients::recombinationNode,
           py::arg("seed"),
           "Step A verbatim: the composed canonical Proton's recombination_node.")
      .def("formation_node", &ProtonIngredients::formationNode, py::arg("seed"),
           "Step B with nothing pinned: the same ideal diquark {1,w} + third quark "
           "{w*w} inputs on the same single Delta^4 seed as Proton.formation_node, "
           "but with an EMPTY output-target list — the final state emerges.")
      .def("joint_node", &ProtonIngredients::jointNode, py::arg("seed"),
           "The joint inputs-only node: ONE MultiCobordism whose inputs are the three "
           "Z3-symmetric neutral q-qbar pairs {1,-1,0} | {0,1,-1} | {-1,0,1} (each "
           "Sigma = 0 — the only prepared content, fixed for the whole build) and "
           "whose output-target list is EMPTY. No diquark, no bare quark, no "
           "intermediate imposed; the pre-registered expectation (a baryon with a "
           "conjugate partner) is READ off the relaxed whole afterwards — singlet and "
           "conjugate-singlet residuals as diagnostics, never drives. The two-step "
           "nodes remain the reference oracle. NOT run (the caller drives it).")
      .def("converged", &ProtonIngredients::converged,
           "True iff the kept attempt was stationary AND persistent — never a "
           "statement about the singlet or the hole count.")
      .def("stationary", &ProtonIngredients::stationary,
           "Whether the kept attempt's final run_stage2 stopped on stationarity.")
      .def("persistent", &ProtonIngredients::persistent,
           "Whether continued evolve+relax left holes, b_k, and F stable.")
      .def("seed", &ProtonIngredients::seed, "Base seed of the kept attempt.")
      .def("spacetime", &ProtonIngredients::spacetime,
           "The full relaxed emergent step-B complex of the KEPT attempt.\n\n"
           "Safe to hold: it is set once when build() finishes and is not "
           "reassigned afterwards, unlike MultiCobordism.st, which a drive call "
           "replaces (see its docstring). Read it AFTER build(); before that it "
           "is the not-yet-driven complex.")
      .def("block", &ProtonIngredients::block,
           "The emergent object IS the whole step-B cobordism (parity with "
           "Proton.block).")
      .def("emergent_holes", &ProtonIngredients::emergentHoles,
           "The emergent holes on the whole — an observable, not a gate; "
           "may be any count, including zero.")
      .def("singlet_residual", &ProtonIngredients::singletResidual,
           "DIAGNOSTIC only: the singlet r_state of Proton.singlet() against the "
           "whole, read after the fact for comparison with the canonical build. It "
           "never steers or gates this build.")
      .def("input_residual", &ProtonIngredients::inputResidual,
           "Step B's inputs-only r_U — the whole matter term of the emergent arm.")
      .def("final_objective", &ProtonIngredients::finalObjective,
           "The kept attempt's final objective F.")
      .def("diquark_residual", &ProtonIngredients::diquarkResidual,
           "Step A's r_U — reported exactly as Proton reports it.");

  // ----- Gated surgical cone-out/cone-in (topology change, #460) -----
  py::class_<SurgicalCone>(m, "SurgicalCone",
      R"doc(Gated surgical cone-out/cone-in: the topology-CHANGING move (#460, T3).

The genuine b_k-hole creator of the Emergent Color Topology epic (#457). Pachner
moves and the orientation-safe stellar refinement cone (T1/T2) are topology-
PRESERVING; this is not. coneOut removes one top cell (its orphaned edges, then
any isolated vertex) -- on a closed manifold this opens a manifold-with-boundary
and, for a cell disjoint from an existing hole, raises b_{d-1} by 1 (on S^3, the
color register's b_2). coneIn adds one top cell on a fresh vertex joined to d
existing vertices, lowering b_{d-1} by 1 when it caps a hole. EVERY move is gated
on ChainComplex.dualComplexIsValid (a valid manifold-with-boundary; the #429
n>=4 recursive check) -- surgery is allowed BECAUSE it is gated; bypassing the
gate is what broke the #353 weld. Rejected moves roll back bit-identically.
Accepted moves stack; rollback() undoes the last LIFO, restoring every edge
length and phase so a round trip leaves the dual Regge action (Re AND Im)
invariant.)doc")
      .def(py::init<Spacetime *>(), py::arg("spacetime"), py::keep_alive<1, 2>(),
           "Bind the cone to a spacetime (does not mutate it).")
      .def("coneOut", &SurgicalCone::coneOut, py::arg("cell"),
           "(ok, reason): gated surgical cone-out -- remove the top cell whose "
           "sorted vertex ids equal `cell` (plus orphaned edges and any vertex "
           "thereby isolated). Accepts only a valid manifold-with-boundary; "
           "otherwise restores the cell and names the reason. Rejects removing "
           "the last top cell.")
      .def("coneIn", &SurgicalCone::coneIn, py::arg("target_verts"),
           py::arg("timelike") = false,
           "(ok, reason): gated surgical cone-in -- create a fresh vertex, join "
           "it to the d `target_verts` to form a new top cell. Accepts only a "
           "valid manifold-with-boundary; otherwise undoes the additions.")
      .def("rollback", &SurgicalCone::rollback,
           "Undo the last accepted move (LIFO), restoring the complex bit-for-"
           "bit (edge lengths and phases). False if nothing is applied.")
      .def("rollbackAll", &SurgicalCone::rollbackAll,
           "Roll every accepted move back; returns the number undone.")
      .def_property_readonly("depth", &SurgicalCone::depth,
           "Number of accepted, not-yet-rolled-back moves on the stack.")
      .def_property_readonly("isApplied", &SurgicalCone::isApplied,
           "True iff at least one move is accepted and not yet rolled back.")
      .def("bettiNumbers", &SurgicalCone::bettiNumbers,
           "Betti numbers b_0..b_n (over Q) of the CURRENT complex -- the read-"
           "out the b_k-delta tests assert a surgical move shifts by one.")
      .def("validate", &SurgicalCone::validate,
           "(ok, reason): the manifold-with-boundary verdict on the CURRENT "
           "complex -- the same gate coneOut / coneIn apply.");

  // ----- Analytic-first kernel and cache contract (#764) -----

  py::enum_<CertificateGrade>(m, "CertificateGrade",
      "How a result was obtained: algebraically exact (closed-form identity, "
      "rounding only), structure-exact (exact given a verified structural "
      "premise), certified numerical (iterative/truncated with residual + "
      "conditioning), or heuristic discovery (uncertified proposal).")
      .value("AlgebraicallyExact", CertificateGrade::AlgebraicallyExact)
      .value("StructureExact", CertificateGrade::StructureExact)
      .value("CertifiedNumerical", CertificateGrade::CertifiedNumerical)
      .value("HeuristicDiscovery", CertificateGrade::HeuristicDiscovery);

  py::enum_<CertificateDomain>(m, "CertificateDomain",
      "The spectral domain a certificate speaks for: the static/whole-"
      "operator statement, or an explicit frequency band window.")
      .value("Static", CertificateDomain::Static)
      .value("BandWindow", CertificateDomain::BandWindow);

  py::enum_<CertificateRegime>(m, "CertificateRegime",
      "The metric regime the producing kernel verified: positive-"
      "semidefinite, Hermitian indefinite, or non-normal (the general "
      "Lorentzian d'Alembertian regime).")
      .value("PositiveSemidefinite", CertificateRegime::PositiveSemidefinite)
      .value("HermitianIndefinite", CertificateRegime::HermitianIndefinite)
      .value("NonNormal", CertificateRegime::NonNormal)
      .value("ComplexSymmetricPencil", CertificateRegime::ComplexSymmetricPencil);

  py::class_<Certificate>(m, "Certificate",
      R"doc(Certification record attached to every analytic-first kernel result (#764).

Grade (claim class) + domain + regime + measured relative residual, the
conditioning of the computation, the dense-reference error where one was
measured on a crossover fixture, and the declared tolerance. Unmeasured
quantities are NaN, never zero. holds() = a certified grade whose residual met
the tolerance; HeuristicDiscovery never holds.)doc")
      .def(py::init<>())
      .def_static("algebraicallyExact", &Certificate::algebraicallyExact,
                  py::arg("domain"), py::arg("regime"), py::arg("residual"),
                  py::arg("tolerance"))
      .def_static("structureExact", &Certificate::structureExact,
                  py::arg("domain"), py::arg("regime"), py::arg("residual"),
                  py::arg("conditioning"), py::arg("tolerance"))
      .def_static("certifiedNumerical", &Certificate::certifiedNumerical,
                  py::arg("domain"), py::arg("regime"), py::arg("residual"),
                  py::arg("conditioning"), py::arg("tolerance"))
      .def_static("heuristicDiscovery", &Certificate::heuristicDiscovery,
                  py::arg("domain"), py::arg("regime"))
      .def_property_readonly("grade", &Certificate::grade)
      .def_property_readonly("domain", &Certificate::domain)
      .def_property_readonly("regime", &Certificate::regime)
      .def_property_readonly("residual", &Certificate::residual)
      .def_property_readonly("conditioning", &Certificate::conditioning)
      .def_property_readonly("denseReferenceError",
                             &Certificate::denseReferenceError)
      .def("setDenseReferenceError", &Certificate::setDenseReferenceError,
           py::arg("error"),
           "Record the relative error measured against the dense reference on "
           "a crossover fixture.")
      .def_property_readonly("tolerance", &Certificate::tolerance)
      .def("holds", &Certificate::holds)
      .def("describe", &Certificate::describe)
      .def("__repr__", &Certificate::describe);

  py::class_<CertifiedVector>(m, "CertifiedVector",
      "A vector-valued kernel result (solution / eigenvalue list / spectrum) "
      "with its attached Certificate; no result travels without its "
      "certification.")
      .def_readonly("values", &CertifiedVector::values)
      .def_readonly("certificate", &CertifiedVector::certificate);

  py::class_<TouchedStar>(m, "TouchedStar",
      R"doc(Publication record of one accepted move: touched
simplices, changed edges, created/deleted cells, all named by vertex
identifiers. AnalyticCache.publish drops entries whose component vertex set
meets this star; disjoint siblings survive.)doc")
      .def(py::init<>())
      .def("addTouchedSimplex", &TouchedStar::addTouchedSimplex,
           py::arg("vertex_ids"),
           "Record a simplex whose geometry or incidence changed.")
      .def("addChangedEdge", &TouchedStar::addChangedEdge, py::arg("vertex_a"),
           py::arg("vertex_b"),
           "Record an edge whose complex length or phase changed.")
      .def("addCreatedCell", &TouchedStar::addCreatedCell, py::arg("vertex_ids"),
           "Record a created cell (a combinatorial change).")
      .def("addDeletedCell", &TouchedStar::addDeletedCell, py::arg("vertex_ids"),
           "Record a deleted cell (a combinatorial change).")
      .def_property_readonly("vertices",
           [](const TouchedStar &star) {
             return std::vector<std::uint64_t>(star.vertices().begin(),
                                               star.vertices().end());
           },
           "The union of recorded vertex identifiers (unordered).")
      .def_property_readonly("structuralChange", &TouchedStar::structuralChange)
      .def_property_readonly("empty", &TouchedStar::empty);

  py::class_<AnalyticCache>(m, "AnalyticCache",
      R"doc(Revision- and touched-star-keyed cache for per-component analytic payloads
(#764): Hodge blocks, component factorizations, spectral projectors,
transports, covariance blocks, Wick contraction plans.

Entries are keyed by the order-independent component vertex-set fingerprint
(the MultiCobordism block convention), a kind string, and an integer
parameter, and stamped with Spacetime.metricRevisionKey(). An entry is served
only while nothing changed anywhere, or while every change since its stamp
was published (publish) and the entry survived every star-intersection test.
An unpublished revision drift serves NOTHING until the next publish/store --
fail-safe. Replay mode disables the cache (setEnabled) and compares against
the incremental path.)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"),
           "Bind to the spacetime whose geometry revisions gate this cache.")
      .def_static("componentKey", &AnalyticCache::componentKey,
                  py::arg("vertex_ids"),
                  "Order-independent fingerprint of a vertex-identifier set "
                  "(any permutation yields the same key).")
      .def("geometryRevision", &AnalyticCache::geometryRevision,
           "Spacetime.metricRevisionKey(): moves on any combinatorial change, "
           "setLength, or setPhase.")
      .def("structuralRevision", &AnalyticCache::structuralRevision,
           "Spacetime.structuralRevision(): the combinatorial revision.")
      .def("store",
           [](AnalyticCache &cache,
              const std::vector<std::uint64_t> &componentVertexIds,
              const std::string &kind, std::int64_t parameter,
              const py::object &payload, const Certificate &certificate) {
             cache.store(componentVertexIds, kind, parameter,
                         std::make_shared<py::object>(payload), certificate);
           },
           py::arg("component_vertex_ids"), py::arg("kind"),
           py::arg("parameter"), py::arg("payload"), py::arg("certificate"),
           "Store payload + certificate for (component vertex set, kind, "
           "parameter), stamped at the CURRENT metric revision.")
      .def("fetch",
           [](const AnalyticCache &cache,
              const std::vector<std::uint64_t> &componentVertexIds,
              const std::string &kind, std::int64_t parameter) -> py::object {
             const auto payload =
                 cache.fetch(componentVertexIds, kind, parameter);
             if (!payload)
               return py::none();
             return *std::static_pointer_cast<py::object>(payload);
           },
           py::arg("component_vertex_ids"), py::arg("kind"),
           py::arg("parameter"),
           "The cached payload, or None when absent, disabled, or stale.")
      .def("fetchCertificate",
           [](const AnalyticCache &cache,
              const std::vector<std::uint64_t> &componentVertexIds,
              const std::string &kind, std::int64_t parameter) -> py::object {
             const Certificate *certificate =
                 cache.fetchCertificate(componentVertexIds, kind, parameter);
             if (certificate == nullptr)
               return py::none();
             return py::cast(*certificate);
           },
           py::arg("component_vertex_ids"), py::arg("kind"),
           py::arg("parameter"),
           "The certificate stored beside a payload, or None exactly when "
           "fetch would return None.")
      .def("publish", &AnalyticCache::publish, py::arg("star"),
           "Publish one accepted move AFTER mutating: drop entries meeting "
           "the star, then mark the cache synchronized to the current "
           "revision. Disjoint siblings survive.")
      .def_property_readonly("size", &AnalyticCache::size)
      .def("clear", &AnalyticCache::clear)
      .def("setEnabled", &AnalyticCache::setEnabled, py::arg("enabled"),
           "Replay-mode switch: a disabled cache serves nothing but keeps "
           "accepting stores.")
      .def_property_readonly("enabled", &AnalyticCache::enabled)
      .def_property_readonly("hits", &AnalyticCache::hits)
      .def_property_readonly("misses", &AnalyticCache::misses)
      .def_property_readonly("invalidations", &AnalyticCache::invalidations);

  py::class_<KuennethProduct>(m, "KuennethProduct",
      R"doc(The exact Kronecker-sum/Kuenneth rule L_{AxB} = L_A (x) I + I (x) L_B (#764).

Algebraically exact as a matrix identity; as a statement about a complex it
holds only for an actual product cell structure with product weights, which
productCertificate verifies at degree zero (a staircase SimplicialProduct is
refused: holds() == False). The degree-zero operator there is the U(1)
CONNECTION graph Laplacian connectionLaplacian, not the Hodge L_0 (#805). The
spectrum of the Kronecker sum is exactly the pairwise sums of the factor spectra
-- no product eigensolve.)doc")
      .def_static("kroneckerSum", &KuennethProduct::kroneckerSum,
                  py::arg("laplacian_a"), py::arg("dim_a"),
                  py::arg("laplacian_b"), py::arg("dim_b"),
                  "L_A (x) I + I (x) L_B, flat row-major (dimA*dimB)^2; "
                  "product index (iA, iB) -> iA*dimB + iB.")
      .def_static("pairwiseSpectrum", &KuennethProduct::pairwiseSpectrum,
                  py::arg("spectrum_a"), py::arg("spectrum_b"),
                  "All pairwise sums, ascending by (Re, Im): the exact "
                  "Kronecker-sum spectrum.")
      .def_static("productCertificate", &KuennethProduct::productCertificate,
                  py::arg("product"), py::arg("factor_a"), py::arg("factor_b"),
                  py::arg("pairing"), py::arg("tolerance") = 1e-12,
                  "Certify that `product`'s U(1) CONNECTION graph Laplacian "
                  "(connectionLaplacian, D - A over the sorted vertex order) "
                  "equals the Kronecker sum of the factors' under the declared "
                  "(product_id, a_id, b_id) vertex pairing. holds() grants the "
                  "Kuenneth rule for this complex. Not a statement about the "
                  "Hodge L_0.");

  py::class_<OccupationSpectra>(m, "OccupationSpectra",
      R"doc(Fermionic second quantization at the SPECTRUM/MATRIX level (#764): free
many-body spectra as occupation subset sums of a one-particle spectrum, the
direct-sum identity at the spectrum level, and one-particle direct-sum /
hopping-block assembly. Exact for any square one-particle operator (complex
eigenvalues allowed; nothing assumes Hermitian or positive-definite). Fock
OPERATOR structure (creation/annihilation, wedge, dGamma as an operator) is
the exterior-algebra track's, not built here.)doc")
      .def_static("subsetSums", &OccupationSpectra::subsetSums,
                  py::arg("one_particle"), py::arg("particles"),
                  py::arg("max_terms") = OccupationSpectra::kDefaultMaxTerms,
                  "The C(n, N) fermionic occupation subset sums: the exact "
                  "free N-particle spectrum of dGamma(h). Ascending (Re, Im).")
      .def_static("fockSums", &OccupationSpectra::fockSums,
                  py::arg("one_particle"),
                  py::arg("max_terms") = OccupationSpectra::kDefaultMaxTerms,
                  "All 2^n subset sums: the full free fermionic Fock spectrum "
                  "across every particle number.")
      .def_static("directSumSubsetSums", &OccupationSpectra::directSumSubsetSums,
                  py::arg("factor_a"), py::arg("factor_b"), py::arg("particles"),
                  py::arg("max_terms") = OccupationSpectra::kDefaultMaxTerms,
                  "The N-particle spectrum of h_A + h_B (direct sum) computed "
                  "FROM THE FACTORS: merged pairwise sums over particle splits "
                  "N_A + N_B = N -- the F_-(A (+) B) ~ F_-(A) (x) F_-(B) "
                  "identity at the spectrum level.")
      .def_static("directSum", &OccupationSpectra::directSum,
                  py::arg("block_a"), py::arg("dim_a"), py::arg("block_b"),
                  py::arg("dim_b"),
                  "The one-particle direct sum [[A, 0], [0, B]], flat "
                  "row-major (dimA+dimB)^2.")
      .def_static("hoppingBlock", &OccupationSpectra::hoppingBlock,
                  py::arg("block_a"), py::arg("dim_a"), py::arg("block_b"),
                  py::arg("dim_b"), py::arg("coupling"),
                  py::arg("coupling_reverse") =
                      std::vector<std::complex<double>>{},
                  "The one-particle hopping assembly [[A, C], [C', B]]. An "
                  "empty coupling_reverse selects C' = C^dagger (the Hermitian "
                  "hopping term); pass it explicitly in the non-normal "
                  "regime.");

  py::class_<LowRankUpdate> lowRankUpdate(m, "LowRankUpdate",
      R"doc(Structure-exact Woodbury / secular update helpers for genuinely low-rank
local operator changes (#764). The base operator is LU-factored once (general
complex square; no Hermitian or positive-definite assumption); a registered
change Delta = U W is solved through the Woodbury identity by factor solves
only -- no explicit inverse. Results are exact GIVEN the verified premise that
U W spans the FULL affected change: factorsFromTouched builds spanning factors
from a declared touched row/column set and reports leakage, spansAffectedChange
re-verifies a claimed factorization, refactor is the cold-recompute fallback.
Every solve reports its measured residual and conditioning.

rankOneEigenvalues is the secular rank-one HERMITIAN eigenvalue update
(interlacing bisection, certified numerical); the non-normal regime is refused
-- a self-adjoint method is never applied to a non-self-adjoint operator.)doc");

  py::class_<LowRankUpdate::TouchedFactors>(lowRankUpdate, "TouchedFactors",
      "Factors of a touched-star operator change: Delta = left * right, rank "
      "<= 2 * |active touched indices|. spansChange == False means the delta "
      "leaked outside the declared touched rows/columns -- the factors are "
      "NOT exact and the caller must cold-recompute.")
      .def_readonly("spansChange", &LowRankUpdate::TouchedFactors::spansChange)
      .def_readonly("rank", &LowRankUpdate::TouchedFactors::rank)
      .def_readonly("left", &LowRankUpdate::TouchedFactors::left)
      .def_readonly("right", &LowRankUpdate::TouchedFactors::right);

  lowRankUpdate
      .def(py::init<const std::vector<std::complex<double>> &, int>(),
           py::arg("base"), py::arg("dim"),
           "Factor the base operator (flat row-major dim x dim, partial-pivot "
           "LU).")
      .def_property_readonly("dimension", &LowRankUpdate::dimension)
      .def_property_readonly("updateRank", &LowRankUpdate::updateRank)
      .def("setUpdate", &LowRankUpdate::setUpdate, py::arg("left"),
           py::arg("right"), py::arg("rank"),
           "Register the pending change Delta = left(dim x rank) * "
           "right(rank x dim), replacing any previous one.")
      .def("clearUpdate", &LowRankUpdate::clearUpdate)
      .def("solve", &LowRankUpdate::solve, py::arg("rhs"),
           py::arg("tolerance") = 1e-12,
           "Woodbury solve of (A + U W) x = b by factor solves; certificate "
           "carries the measured relative residual and the LU/capacitance "
           "condition estimates.")
      .def("apply", &LowRankUpdate::apply, py::arg("x"),
           "y = (A + U W) x, for external residual checks and benchmarks.")
      .def("spansAffectedChange", &LowRankUpdate::spansAffectedChange,
           py::arg("updated"), py::arg("tolerance") = 1e-12,
           "Exactness check: ||(updated - A) - U W||_F <= tolerance * "
           "||updated||_F. False = the low-rank path may NOT be called exact; "
           "cold-recompute instead.")
      .def_static("factorsFromTouched", &LowRankUpdate::factorsFromTouched,
                  py::arg("base"), py::arg("updated"), py::arg("dim"),
                  py::arg("touched"),
                  "Exact factors of the change from the declared touched "
                  "row/column set, with the spans-the-change verdict.")
      .def("refactor", &LowRankUpdate::refactor, py::arg("base"), py::arg("dim"),
           "Cold-recompute fallback: refactor `base` as the new base operator "
           "and clear any registered update.")
      .def_static("rankOneEigenvalues", &LowRankUpdate::rankOneEigenvalues,
                  py::arg("eigenvalues"), py::arg("z"), py::arg("rho"),
                  py::arg("tolerance") = 1e-10,
                  "Secular rank-one Hermitian eigenvalue update: ascending "
                  "eigenvalues of diag(d) + rho z z^dagger, z in the "
                  "eigenbasis. Certified numerical (bracket width + exact "
                  "trace identity + deflation bound). Hermitian domain only.");

  py::class_<DenseReference>(m, "DenseReference",
      R"doc(Dense reference kernels used ONLY below a configurable dimension crossover
(#764): on small fixtures they supply the independent answer a structured path
is compared against; at or above the crossover they refuse (throw) -- a dense
global solve is the prohibited default at scale, never a silent fallback.
solve is an LU factor solve (never an explicit inverse); spectrum honors a
self-adjoint request only after verifying Hermiticity; fockSpectrum is the
dense-Fock oracle at the SPECTRUM level (dense one-particle eigensolve +
explicit occupation subset-sum enumeration) used to validate structured subset
sums and quasi-free Wick reads on crossover fixtures.)doc")
      .def(py::init<int>(),
           py::arg("crossover_dimension") =
               DenseReference::kDefaultCrossoverDimension)
      .def_property_readonly("crossoverDimension",
                             &DenseReference::crossoverDimension)
      .def("setCrossoverDimension", &DenseReference::setCrossoverDimension,
           py::arg("crossover_dimension"))
      .def("belowCrossover", &DenseReference::belowCrossover, py::arg("dim"))
      .def("solve", &DenseReference::solve, py::arg("matrix"), py::arg("dim"),
           py::arg("rhs"), py::arg("tolerance") = 1e-12,
           "Dense LU factor solve with measured residual + conditioning.")
      .def("spectrum", &DenseReference::spectrum, py::arg("matrix"),
           py::arg("dim"), py::arg("self_adjoint"),
           py::arg("tolerance") = 1e-10,
           "Dense eigenvalues ascending by (Re, Im); the self-adjoint solver "
           "runs only after Hermiticity is verified, else the general solver "
           "with a NonNormal certificate.")
      .def("fockSpectrum", &DenseReference::fockSpectrum,
           py::arg("one_particle"), py::arg("dim"), py::arg("particles"),
           py::arg("self_adjoint"), py::arg("tolerance") = 1e-10,
           "Dense-Fock oracle at the spectrum level: dense eigensolve + exact "
           "occupation subset sums for the N-particle sector.");

  // ----- Recursive static/shifted response reduction (#768) -----
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
partition (#768). Static: the exact supported response
L_eff = L_BB - L_BI L_II^+ L_IB by sparse/rank-revealing factor solves
(minimization certificate in the positive self-adjoint regime, stationarity
in the Hermitian-indefinite regime, certified block elimination with the
left-kernel compatibility check in the non-normal regime; interior kernels
are RETAINED as explicit stalk coordinates, never regularized away). Band:
the exact Feshbach-Schur pencil F_B(lambda) = L_BB - lambda I -
L_BI (L_II - lambda I)^{-1} L_IB over caller-supplied windows with the exact
determinant factorization det(L - lambda) = det(L_II - lambda) det F_B(lambda),
honest algebraic (det winding) vs geometric (dim ker F_B) multiplicities, and
a certified Craig-Bampton/AMLS linear surrogate. The next level is the
abstract labeled sum of retained fibers with embedding J and Gram G = J^dag W J
(one declared policy per run), an operator-valued response network, and a
cellular-sheaf realization emitted ONLY when restriction maps reproduce the
blocks. Nested quotients carry lineage; per-component contributions reuse the
#764 AnalyticCache so a published TouchedStar recomputes only the affected
ancestry. Read-only: nothing here enters the emergence objective.)doc");

  py::class_<RecursiveQuotient::Options>(recursiveQuotient, "Options",
      "Reduction options: certificate tolerance, rank-revealing threshold, "
      "dense crossover, the declared FiberEmbeddingPolicy (+ epsilon), and "
      "caller-selected interior cells to retain.")
      .def(py::init<>())
      .def_readwrite("tolerance", &RecursiveQuotient::Options::tolerance)
      .def_readwrite("rankTolerance", &RecursiveQuotient::Options::rankTolerance)
      .def_readwrite("denseCrossover", &RecursiveQuotient::Options::denseCrossover)
      .def_readwrite("embeddingPolicy",
                     &RecursiveQuotient::Options::embeddingPolicy)
      .def_readwrite("nearIsometryEpsilon",
                     &RecursiveQuotient::Options::nearIsometryEpsilon)
      .def_readwrite("selectedInteriorIndices",
                     &RecursiveQuotient::Options::selectedInteriorIndices)
      .def_readwrite("selectedInteriorCells",
                     &RecursiveQuotient::Options::selectedInteriorCells);

  py::class_<RecursiveQuotient::RetainedCoordinate>(recursiveQuotient,
      "RetainedCoordinate",
      "One reduced coordinate: kind, owning component, fine index (cells) or "
      "-1 (modes), the fine-space embedding column, and provenance.")
      .def_readonly("kind", &RecursiveQuotient::RetainedCoordinate::kind)
      .def_readonly("component",
                    &RecursiveQuotient::RetainedCoordinate::component)
      .def_readonly("fineIndex",
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
      .def_readonly("integerNullity",
                    &RecursiveQuotient::InteriorNullspaceRead::integerNullity)
      .def_readonly(
          "integerNullityMeasured",
          &RecursiveQuotient::InteriorNullspaceRead::integerNullityMeasured,
          "Whether the exact integer nullity was computed at all (False on the "
          "matrix path and on integer-kernel overflow, where integerNullity == "
          "0 means 'not measured').")
      .def_readonly(
          "nullityDiscrepancy",
          &RecursiveQuotient::InteriorNullspaceRead::nullityDiscrepancy,
          "nullity - integerNullity: the recorded disagreement between the "
          "numerical kernel of the weighted interior block and the exact "
          "integer topological nullity. 0 means they agree; NaN means no "
          "integer nullity was measured (never 0, which would claim an "
          "agreement that was never made).")
      .def_readonly("integerBasis",
                    &RecursiveQuotient::InteriorNullspaceRead::integerBasis)
      .def_readonly("kernelBasis",
                    &RecursiveQuotient::InteriorNullspaceRead::kernelBasis)
      .def_readonly("leftKernelBasis",
                    &RecursiveQuotient::InteriorNullspaceRead::leftKernelBasis)
      .def_readonly("certificate",
                    &RecursiveQuotient::InteriorNullspaceRead::certificate);

  py::class_<RecursiveQuotient::StaticReductionRead>(recursiveQuotient,
      "StaticReductionRead",
      "The exact supported static reduction: kept-cell indices, all reduced "
      "coordinates with provenance, the effective operator (leading kept "
      "block = L_BB - L_BI L_II^+ L_IB), residuals, certificate.")
      .def_readonly("interfaceIndices",
                    &RecursiveQuotient::StaticReductionRead::interfaceIndices)
      .def_readonly("coordinates",
                    &RecursiveQuotient::StaticReductionRead::coordinates)
      .def_readonly("effectiveOperator",
                    &RecursiveQuotient::StaticReductionRead::effectiveOperator)
      .def_readonly("solveResidual",
                    &RecursiveQuotient::StaticReductionRead::solveResidual)
      .def_readonly(
          "compatibilityResidual",
          &RecursiveQuotient::StaticReductionRead::compatibilityResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::StaticReductionRead::certificate);

  py::class_<RecursiveQuotient::FeshbachRead>(recursiveQuotient, "FeshbachRead",
      "One exact Feshbach-Schur response evaluation over a declared window, "
      "with resonance retention, compatibility check, and the measured "
      "determinant-factorization residual (below the dense crossover).")
      .def_readonly("lam", &RecursiveQuotient::FeshbachRead::lambda)
      .def_readonly("windowLower", &RecursiveQuotient::FeshbachRead::windowLower)
      .def_readonly("windowUpper", &RecursiveQuotient::FeshbachRead::windowUpper)
      .def_readonly("response", &RecursiveQuotient::FeshbachRead::response)
      .def_readonly("coordinates", &RecursiveQuotient::FeshbachRead::coordinates)
      .def_readonly("resonant", &RecursiveQuotient::FeshbachRead::resonant)
      .def_readonly("solveResidual",
                    &RecursiveQuotient::FeshbachRead::solveResidual)
      .def_readonly("compatibilityResidual",
                    &RecursiveQuotient::FeshbachRead::compatibilityResidual)
      .def_readonly("determinantResidual",
                    &RecursiveQuotient::FeshbachRead::determinantResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::FeshbachRead::certificate);

  py::class_<RecursiveQuotient::MultiplicityRead>(recursiveQuotient,
      "MultiplicityRead",
      "Honest multiplicity report: algebraic = winding of det F_B plus the "
      "separately-reported interior winding; geometric = dim ker F_B(lambda). "
      "They agree only in the self-adjoint/semisimple setting.")
      .def_readonly("lam", &RecursiveQuotient::MultiplicityRead::lambda)
      .def_readonly("contourRadius",
                    &RecursiveQuotient::MultiplicityRead::contourRadius)
      .def_readonly("nodes", &RecursiveQuotient::MultiplicityRead::nodes)
      .def_readonly("responseWinding",
                    &RecursiveQuotient::MultiplicityRead::responseWinding)
      .def_readonly("interiorWinding",
                    &RecursiveQuotient::MultiplicityRead::interiorWinding)
      .def_readonly("algebraic", &RecursiveQuotient::MultiplicityRead::algebraic)
      .def_readonly("geometric", &RecursiveQuotient::MultiplicityRead::geometric)
      .def_readonly("semisimple",
                    &RecursiveQuotient::MultiplicityRead::semisimple)
      .def_readonly("phaseStepMargin",
                    &RecursiveQuotient::MultiplicityRead::phaseStepMargin)
      .def_readonly("certificate",
                    &RecursiveQuotient::MultiplicityRead::certificate);

  py::class_<RecursiveQuotient::CraigBamptonRead>(recursiveQuotient,
      "CraigBamptonRead",
      "Craig-Bampton/AMLS retained-mode surrogate: declared window, retained "
      "fixed-interface modes per component, basis, reduced (stiffness, mass) "
      "pencil, discarded-mode gap, and fine-space eigenresiduals.")
      .def_readonly("windowLower",
                    &RecursiveQuotient::CraigBamptonRead::windowLower)
      .def_readonly("windowUpper",
                    &RecursiveQuotient::CraigBamptonRead::windowUpper)
      .def_readonly("modeCutoff",
                    &RecursiveQuotient::CraigBamptonRead::modeCutoff)
      .def_readonly("retainedModes",
                    &RecursiveQuotient::CraigBamptonRead::retainedModes)
      .def_readonly("basis", &RecursiveQuotient::CraigBamptonRead::basis)
      .def_readonly("reducedStiffness",
                    &RecursiveQuotient::CraigBamptonRead::reducedStiffness)
      .def_readonly("reducedMass",
                    &RecursiveQuotient::CraigBamptonRead::reducedMass)
      .def_readonly("discardedModeGap",
                    &RecursiveQuotient::CraigBamptonRead::discardedModeGap)
      .def_readonly("windowEigenvalues",
                    &RecursiveQuotient::CraigBamptonRead::windowEigenvalues)
      .def_readonly("eigenResiduals",
                    &RecursiveQuotient::CraigBamptonRead::eigenResiduals)
      .def_readonly("certificate",
                    &RecursiveQuotient::CraigBamptonRead::certificate);

  py::class_<RecursiveQuotient::LabeledFiberSumRead>(recursiveQuotient,
      "LabeledFiberSumRead",
      "The abstract labeled sum of retained fibers: embedding J into the "
      "chain space, Gram G = J^dag W J, the declared policy, gram defect, "
      "kernel nullity, and nominal vs effective ranks. Adjacent fibers may "
      "overlap on shared interface cells; a direct sum is never asserted.")
      .def_readonly("summandComponents",
                    &RecursiveQuotient::LabeledFiberSumRead::summandComponents)
      .def_readonly("summandRanks",
                    &RecursiveQuotient::LabeledFiberSumRead::summandRanks)
      .def_readonly("embedding",
                    &RecursiveQuotient::LabeledFiberSumRead::embedding)
      .def_readonly("gram", &RecursiveQuotient::LabeledFiberSumRead::gram)
      .def_readonly("policy", &RecursiveQuotient::LabeledFiberSumRead::policy)
      .def_readonly("gramDefect",
                    &RecursiveQuotient::LabeledFiberSumRead::gramDefect)
      .def_readonly("quotientNullity",
                    &RecursiveQuotient::LabeledFiberSumRead::quotientNullity)
      .def_readonly("nominalRank",
                    &RecursiveQuotient::LabeledFiberSumRead::nominalRank)
      .def_readonly("effectiveRank",
                    &RecursiveQuotient::LabeledFiberSumRead::effectiveRank)
      .def_readonly("quotientBasis",
                    &RecursiveQuotient::LabeledFiberSumRead::quotientBasis)
      .def_readonly(
          "fromCertifiedBands",
          &RecursiveQuotient::LabeledFiberSumRead::fromCertifiedBands)
      .def_readonly(
          "summandCertificates",
          &RecursiveQuotient::LabeledFiberSumRead::summandCertificates)
      .def_readonly(
          "worstIsolationGap",
          &RecursiveQuotient::LabeledFiberSumRead::worstIsolationGap)
      .def_readonly("allBandsAccepted",
                    &RecursiveQuotient::LabeledFiberSumRead::allBandsAccepted)
      .def_readonly("certificate",
                    &RecursiveQuotient::LabeledFiberSumRead::certificate);

  py::class_<RecursiveQuotient::CertifiedBand>(recursiveQuotient,
      "CertifiedBand",
      "One certified isolated band handed to certifiedFiberSum as the summand "
      "E_v of the master recursion: its right frame over this level's fine "
      "coordinates, its rank, its isolation gaps and frequency window, "
      "whether its producing configuration accepted it, and its certificate.")
      .def(py::init<>())
      .def_readwrite("component", &RecursiveQuotient::CertifiedBand::component)
      .def_readwrite("frame", &RecursiveQuotient::CertifiedBand::frame)
      .def_readwrite("rank", &RecursiveQuotient::CertifiedBand::rank)
      .def_readwrite("lowerGap", &RecursiveQuotient::CertifiedBand::lowerGap)
      .def_readwrite("upperGap", &RecursiveQuotient::CertifiedBand::upperGap)
      .def_readwrite("frequencyLower",
                     &RecursiveQuotient::CertifiedBand::frequencyLower)
      .def_readwrite("frequencyUpper",
                     &RecursiveQuotient::CertifiedBand::frequencyUpper)
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
      .def_readonly("lowerGap",
                    &RecursiveQuotient::CertifiedFiberSummand::lowerGap)
      .def_readonly("upperGap",
                    &RecursiveQuotient::CertifiedFiberSummand::upperGap)
      .def_readonly("frequencyLower",
                    &RecursiveQuotient::CertifiedFiberSummand::frequencyLower)
      .def_readonly("frequencyUpper",
                    &RecursiveQuotient::CertifiedFiberSummand::frequencyUpper)
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
      .def_readonly("windowLower",
                    &RecursiveQuotient::LevelProvenanceRead::windowLower)
      .def_readonly("windowUpper",
                    &RecursiveQuotient::LevelProvenanceRead::windowUpper)
      .def_readonly("solveResidual",
                    &RecursiveQuotient::LevelProvenanceRead::solveResidual)
      .def_readonly(
          "compatibilityResidual",
          &RecursiveQuotient::LevelProvenanceRead::compatibilityResidual)
      .def_readonly("surrogateResidual",
                    &RecursiveQuotient::LevelProvenanceRead::surrogateResidual)
      .def_readonly("discardedModeGap",
                    &RecursiveQuotient::LevelProvenanceRead::discardedModeGap)
      .def_readonly("resonant",
                    &RecursiveQuotient::LevelProvenanceRead::resonant)
      .def_readonly("certificate",
                    &RecursiveQuotient::LevelProvenanceRead::certificate);

  py::class_<RecursiveQuotient::FockStageRead>(recursiveQuotient,
      "FockStageRead",
      "The Fock stage over a labeled sum: the one-particle compression "
      "h = J^dag W L J, the Gram on the same basis, the pencil spectrum, "
      "2^M as fockDimension, and the exact free many-body spectrum as "
      "occupation subset sums. The 2^M space is never materialized and the "
      "spectrum refuses past the declared term budget.")
      .def_readonly("modes", &RecursiveQuotient::FockStageRead::modes)
      .def_readonly("policy", &RecursiveQuotient::FockStageRead::policy)
      .def_readonly("gramDefect",
                    &RecursiveQuotient::FockStageRead::gramDefect)
      .def_readonly("oneParticle",
                    &RecursiveQuotient::FockStageRead::oneParticle)
      .def_readonly("gram", &RecursiveQuotient::FockStageRead::gram)
      .def_readonly("oneParticleSpectrum",
                    &RecursiveQuotient::FockStageRead::oneParticleSpectrum)
      .def_readonly("fockDimension",
                    &RecursiveQuotient::FockStageRead::fockDimension)
      .def_readonly("spectrumMaterialized",
                    &RecursiveQuotient::FockStageRead::spectrumMaterialized)
      .def_readonly("fockSpectrum",
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
      .def_readonly("stalkDimensions",
                    &RecursiveQuotient::ResponseNetworkRead::stalkDimensions)
      .def_readonly("stalkCoordinates",
                    &RecursiveQuotient::ResponseNetworkRead::stalkCoordinates)
      .def_readonly("vertexBlocks",
                    &RecursiveQuotient::ResponseNetworkRead::vertexBlocks)
      .def_readonly("edges", &RecursiveQuotient::ResponseNetworkRead::edges)
      .def_readonly("coverageResidual",
                    &RecursiveQuotient::ResponseNetworkRead::coverageResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::ResponseNetworkRead::certificate);

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
          "edgeStalkDimensions",
          &RecursiveQuotient::SheafRealizationRead::edgeStalkDimensions)
      .def_readonly("restrictionMaps",
                    &RecursiveQuotient::SheafRealizationRead::restrictionMaps)
      .def_readonly(
          "reconstructionResidual",
          &RecursiveQuotient::SheafRealizationRead::reconstructionResidual)
      .def_readonly("certificate",
                    &RecursiveQuotient::SheafRealizationRead::certificate);

  recursiveQuotient
      .def_static("overMatrix", &RecursiveQuotient::overMatrix, py::arg("op"),
                  py::arg("dim"), py::arg("weights"), py::arg("components"),
                  py::arg("options") = RecursiveQuotient::Options(),
                  "Build over an explicit operator (flat row-major) with a "
                  "diagonal chain metric (empty = identity) and 0-based, "
                  "possibly overlapping component index sets covering every "
                  "index. Fixtures and next-level recursion.")
      .def_static("overPencil", &RecursiveQuotient::overPencil, py::arg("A"),
                  py::arg("M"), py::arg("dim"), py::arg("components"),
                  py::arg("options") = RecursiveQuotient::Options(),
                  "Build over a symmetric PENCIL (A~, M) on geometric images (flat "
                  "row-major both): every shifted elimination is taken on "
                  "P(lambda) = A~ - lambda M, and a child level carries the Gram "
                  "T^T M T of its constraint modes (specification §7).")
      .def("isPencil", &RecursiveQuotient::isPencil, "Whether this level is a pencil level.")
      .def("pencilMetric", &RecursiveQuotient::pencilMetric,
           "The pencil's metric M (base) or carried Gram (child), flat row-major; empty otherwise.")
      .def_static(
          "overCells",
          [](std::shared_ptr<Spacetime> st, int degree,
             const std::vector<std::vector<std::vector<std::uint64_t>>> &cells,
             const RecursiveQuotient::Options &options, AnalyticCache *cache) {
            std::shared_ptr<AnalyticCache> held;
            if (cache) held = std::shared_ptr<AnalyticCache>(cache, [](AnalyticCache *) {});
            return RecursiveQuotient::overCells(std::move(st), degree, cells,
                                                options, std::move(held));
          },
          py::arg("spacetime"), py::arg("degree"), py::arg("component_cells"),
          py::arg("options") = RecursiveQuotient::Options(),
          py::arg("cache") = nullptr,
          // the non-owning cache pointer must outlive the quotient
          py::keep_alive<0, 5>(),
          "Build over the spacetime's Hodge operator at `degree` with "
          "components as explicit k-cell sets (vertex-id tuples, matched by "
          "vertex SET). An AnalyticCache bound to the same spacetime enables "
          "per-component reuse across accepted moves.")
      .def_static(
          "overVertexSupports",
          [](std::shared_ptr<Spacetime> st, int degree,
             const std::vector<std::vector<std::uint64_t>> &supports,
             const RecursiveQuotient::Options &options, AnalyticCache *cache) {
            std::shared_ptr<AnalyticCache> held;
            if (cache) held = std::shared_ptr<AnalyticCache>(cache, [](AnalyticCache *) {});
            return RecursiveQuotient::overVertexSupports(
                std::move(st), degree, supports, options, std::move(held));
          },
          py::arg("spacetime"), py::arg("degree"), py::arg("vertex_supports"),
          py::arg("options") = RecursiveQuotient::Options(),
          py::arg("cache") = nullptr, py::keep_alive<0, 5>(),
          "Build with components as vertex supports (the PersistentModularity "
          "convention): a k-cell belongs to a component when ALL its vertices "
          "lie in the support; unclaimed cells form one residual component.")
      .def_property_readonly("dimension", &RecursiveQuotient::dimension)
      .def_property_readonly("componentCount",
                             &RecursiveQuotient::componentCount)
      .def_property_readonly("degree", &RecursiveQuotient::degree)
      .def_property_readonly("level", &RecursiveQuotient::level)
      .def_property_readonly("regime", &RecursiveQuotient::regime)
      .def_property_readonly("interfaceIndices",
                             &RecursiveQuotient::interfaceIndices)
      .def("interiorIndices", &RecursiveQuotient::interiorIndices,
           py::arg("component"))
      .def_property_readonly("coordinateProvenance",
                             &RecursiveQuotient::coordinateProvenance)
      .def("interiorNullspace", &RecursiveQuotient::interiorNullspace,
           py::arg("component"),
           "Exact integer topological zero modes (spacetime path) + the "
           "numerical right/left kernels of the interior block.")
      .def("staticReduction", &RecursiveQuotient::staticReduction,
           py::return_value_policy::reference_internal,
           "The exact supported static reduction (memoized; per-component "
           "contributions served from the bound AnalyticCache when fresh).")
      .def("staticProbeCertificate", &RecursiveQuotient::staticProbeCertificate,
           py::arg("probe"),
           "Regime-appropriate static certificate on one kept-cell probe: "
           "minimum (positive), stationarity (Hermitian-indefinite), or "
           "certified block elimination + left-kernel compatibility "
           "(non-normal).")
      .def("verifyStatic", &RecursiveQuotient::verifyStatic,
           "Worst static probe certificate over every kept basis vector and "
           "the all-ones probe.")
      .def("feshbach", &RecursiveQuotient::feshbach, py::arg("lam"),
           py::arg("window_lower"), py::arg("window_upper"),
           "Exact Feshbach-Schur response F_B(lambda) over a caller-supplied "
           "window, with resonance retention + compatibility checks and the "
           "determinant-factorization residual below the dense crossover.")
      .def("multiplicity", &RecursiveQuotient::multiplicity, py::arg("lam"),
           py::arg("radius"), py::arg("nodes") = 64,
           "Algebraic multiplicity from the unwrapped det-phase windings "
           "(response + interior, reported separately), geometric from "
           "dim ker F_B(lambda); node count doubles until stable.")
      .def("craigBampton", &RecursiveQuotient::craigBampton,
           py::arg("window_lower"), py::arg("window_upper"),
           py::arg("mode_cutoff"), py::arg("residual_tolerance") = -1.0,
           "Craig-Bampton retained-mode basis + reduced (K, M) pencil over "
           "the declared window (certified approximation: the certificate "
           "holds against the caller-declared residual_tolerance; negative "
           "selects the strict Options.tolerance). Refuses the non-normal "
           "regime and indefinite chain metrics.")
      .def("labeledFiberSum", &RecursiveQuotient::labeledFiberSum,
           "The abstract labeled sum of retained fibers with embedding J and "
           "Gram G = J^dag W J under the run's declared policy.")
      .def_static("composeNearIsometryBudget",
                  &RecursiveQuotient::composeNearIsometryBudget,
                  py::arg("epsilon_a"), py::arg("epsilon_b"),
                  "Composable amplitude budget of the CertifiedNearIsometry "
                  "policy: eps_AB <= eps_A + eps_B + eps_A * eps_B.")
      .def("responseNetwork", &RecursiveQuotient::responseNetwork,
           "The next-level operator-valued response network (stalks + "
           "effective blocks).")
      .def("sheafRealization", &RecursiveQuotient::sheafRealization,
           "Cellular-sheaf/simplicial realization, emitted only when "
           "certified; otherwise the general network is retained.")
      .def("nextLevel",
           py::overload_cast<const std::vector<std::vector<int>> &,
                             const RecursiveQuotient::Options &>(
               &RecursiveQuotient::nextLevel, py::const_),
           py::arg("components"), py::arg("options"),
           "Reduce again over this level's reduced operator; the child "
           "carries provenance-prefixed lineage and level + 1.")
      .def("nextLevel",
           py::overload_cast<const std::vector<std::vector<int>> &>(
               &RecursiveQuotient::nextLevel, py::const_),
           py::arg("components"))
      .def("certifiedFiberSum", &RecursiveQuotient::certifiedFiberSum,
           py::arg("bands"),
           "The labeled sum over CERTIFIED ISOLATED BANDS (the master "
           "recursion's E_v), carrying each band's isolation gap and "
           "certificate onto its summand. An uncertified band is summed and "
           "reported, never dropped, and makes the sum's certificate fail to "
           "hold.")
      .def("fockStage", &RecursiveQuotient::fockStage, py::arg("sum"),
           py::arg("max_terms") = std::size_t{1} << 22,
           "The Fock stage over a labeled sum: the one-particle compression, "
           "the pencil spectrum, and the exact free many-body spectrum as "
           "occupation subset sums (refusing past max_terms).")
      .def_static("persistentPartition",
                  &RecursiveQuotient::persistentPartition, py::arg("op"),
                  py::arg("dim"), py::arg("gamma") = 1.0,
                  py::arg("restarts") = 4, py::arg("base_seed") = 0,
                  "P = PersistentPartition(R): partition a response "
                  "network's coordinates by persistent modularity over its "
                  "symmetrized off-diagonal magnitude graph. Covers every "
                  "index exactly once; isolated coordinates come back as "
                  "singletons.")
      .def("childPersistentPartition",
           &RecursiveQuotient::childPersistentPartition,
           py::arg("gamma") = 1.0, py::arg("restarts") = 4,
           py::arg("base_seed") = 0,
           "persistentPartition of this level's reduced operator — the "
           "partition P_l to hand straight to nextLevel.")
      .def("nextLevelAtLambda",
           py::overload_cast<const std::vector<std::vector<int>> &,
                             std::complex<double>, double, double,
                             const RecursiveQuotient::Options &>(
               &RecursiveQuotient::nextLevelAtLambda, py::const_),
           py::arg("components"), py::arg("lambda_"), py::arg("window_lower"),
           py::arg("window_upper"), py::arg("options"),
           "Reduce again ON THE PENCIL: the child's operator is the exact "
           "energy-dependent response F_B(lambda), and it carries the window, "
           "residuals, resonance flag and producing certificate.")
      .def("nextLevelAtLambda",
           py::overload_cast<const std::vector<std::vector<int>> &,
                             std::complex<double>, double, double>(
               &RecursiveQuotient::nextLevelAtLambda, py::const_),
           py::arg("components"), py::arg("lambda_"), py::arg("window_lower"),
           py::arg("window_upper"))
      .def("nextLevelFromSurrogate",
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
      .def("nextLevelFromSurrogate",
           py::overload_cast<const std::vector<std::vector<int>> &, double,
                             double, double, double>(
               &RecursiveQuotient::nextLevelFromSurrogate, py::const_),
           py::arg("components"), py::arg("window_lower"),
           py::arg("window_upper"), py::arg("mode_cutoff"),
           py::arg("residual_tolerance") = -1.0)
      .def_property_readonly("levelProvenance",
                             &RecursiveQuotient::levelProvenance,
                             "How this level was produced from its parent.")
      .def_property_readonly("cellVertices", &RecursiveQuotient::cellVertices,
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
