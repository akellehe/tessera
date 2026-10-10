// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: chain complexes, cochains and
// spectra. One of the translation units that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_cobordism_chain_complex(py::module_ &m) {
  // Per-complex scalar measurements are Observables, as are the characteristic
  // numbers (Euler characteristic, signature, …). Multi-complex and structural
  // operations (cobordism verification, reconstruction, Pachner search) are
  // static-only classes taking a Spacetime.
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

  // ----- Homology backbone: chain complex + exact linear algebra -----

  py::class_<ChainComplex>(m, "ChainComplex",
      R"doc(Simplicial chain complex of a triangulation.

Boundary maps ∂_k over ℤ plus the homology invariants derived from them — Betti
numbers (over ℚ and GF(2)), torsion coefficients, Euler characteristic, and the
∂²=0 sanity check. Purely combinatorial (built from vertex sets; no geometry).)doc")
      .def("orientationSigns", &ChainComplex::orientationSigns,
           "Per-degree +/-1 signs relating stored cell orientations to the reference (ascending id) orientation.")
      .def_static("fromTopCells", &ChainComplex::fromTopCells, py::arg("top_cells"),
           "Build from top cells (vertex-id tuples) alone, oriented by ascending vertex id; no geometry.")
      .def_static("fromCells", &ChainComplex::fromCells, py::arg("cells"),
           "Build from declared cells (vertex-id tuples) of any dimensions, oriented by ascending "
           "vertex id; the complex need not be pure.")
      .def_static("fromSpacetime", &ChainComplex::fromSpacetime, py::arg("spacetime"),
                  "Build the chain complex from a triangulation (a Spacetime).")
      .def("dimension", &ChainComplex::dimension)
      .def("numSimplices", &ChainComplex::numSimplices, py::arg("k"))
      .def("fVector", &ChainComplex::fVector)
      .def("eulerCharacteristic", &ChainComplex::eulerCharacteristic)
      .def("boundaryMatrix", &ChainComplex::boundaryMatrix, py::arg("k"),
           "Flat row-major ∂_k (rows=|C_{k-1}|, cols=|C_k|), entries in {-1,0,1}. "
           "Dense: materialized from boundaryEntries on the first request and cached.")
      .def("boundaryEntries",
           [](const ChainComplex &K, int k) {
             std::vector<std::tuple<int, int, int>> out;
             const auto &entries = K.boundaryEntries(k);
             out.reserve(entries.size());
             for (const auto &e : entries) out.emplace_back(e.row, e.column, e.value);
             return out;
           },
           py::arg("k"),
           "The nonzero entries of ∂_k as (row, column, value) with value in {-1, +1}, "
           "grouped by ascending column. This is the stored form, available at any "
           "complex size.")
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
           "if a class needs a deferred higher Steenrod cup-i product.");

  // ----- Eigen-backed value objects for the Hodge spectrum -----
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
}
