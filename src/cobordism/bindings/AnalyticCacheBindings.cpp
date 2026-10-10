// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the analytic cache, spacetime
// composition, Kuenneth product, occupation spectra, low-rank update and
// dense reference. One of the translation units that Bindings.cpp registers
// in order (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_cobordism_analytic_cache(py::module_ &m) {
  py::class_<AnalyticCache>(m, "AnalyticCache",
      R"doc(Revision- and touched-star-keyed cache for per-component analytic payloads: Hodge blocks, component factorizations, spectral projectors,
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

  py::class_<SpacetimeComposition>(m, "SpacetimeComposition",
      R"doc(Combining two complexes' operators.

Each operation appears twice. The space-level entry points take two complexes
and a degree, assemble both Hodge Laplacians, and combine them. The matrix-level
ones take flat matrices, for a caller that already has the operators.

    product of complexes          operator
    Cartesian   A [] B            Kronecker sum      L_A (x) I + I (x) L_B
    tensor      A  x B            Kronecker product  L_A (x) L_B
    disjoint    A  u B            direct sum         L_A (+) L_B

A product index (iA, iB) maps to iA*dimB + iB, so factor A is most significant.
The direct sum is block diagonal with A first, so it is (dimA + dimB) square
where the products are (dimA*dimB) square. Every assembly is exact.

The Cartesian case has a spectral shortcut, since its two terms commute: see
KuennethProduct.pairwiseSpectrum, which gives the product spectrum without
diagonalizing anything, and KuennethProduct.productCertificate, which certifies
that a complex really is the Cartesian product of two factors.)doc")
      .def_static("cartesianProduct", &SpacetimeComposition::cartesianProduct,
                  py::arg("factor_a"), py::arg("factor_b"),
                  py::arg("degree") = 0, py::arg("metric") = true,
                  "L_k of the Cartesian product A [] B: the Kronecker sum of "
                  "the factors' L_k, flat row-major (dimA*dimB)^2.")
      .def_static("tensorProduct", &SpacetimeComposition::tensorProduct,
                  py::arg("factor_a"), py::arg("factor_b"),
                  py::arg("degree") = 0, py::arg("metric") = true,
                  "L_k of the tensor product A x B: the Kronecker product of "
                  "the factors' L_k, flat row-major (dimA*dimB)^2.")
      .def_static("directSum",
                  py::overload_cast<const std::shared_ptr<Spacetime> &,
                                    const std::shared_ptr<Spacetime> &, int,
                                    bool>(&SpacetimeComposition::directSum),
                  py::arg("factor_a"), py::arg("factor_b"),
                  py::arg("degree") = 0, py::arg("metric") = true,
                  "L_k of the disjoint union A u B: the block-diagonal direct "
                  "sum of the factors' L_k, flat row-major (dimA+dimB)^2.")
      .def_static("kroneckerSum", &SpacetimeComposition::kroneckerSum,
                  py::arg("operator_a"), py::arg("dim_a"),
                  py::arg("operator_b"), py::arg("dim_b"),
                  "A (x) I + I (x) B, flat row-major (dimA*dimB)^2.")
      .def_static("kroneckerProduct", &SpacetimeComposition::kroneckerProduct,
                  py::arg("operator_a"), py::arg("dim_a"),
                  py::arg("operator_b"), py::arg("dim_b"),
                  "A (x) B, flat row-major (dimA*dimB)^2.")
      .def_static("directSum",
                  py::overload_cast<const std::vector<std::complex<double>> &,
                                    int,
                                    const std::vector<std::complex<double>> &,
                                    int>(&SpacetimeComposition::directSum),
                  py::arg("operator_a"), py::arg("dim_a"),
                  py::arg("operator_b"), py::arg("dim_b"),
                  "A (+) B: block diagonal, A first, flat row-major "
                  "(dimA+dimB)^2.");

  py::class_<KuennethProduct>(m, "KuennethProduct",
      R"doc(The exact Kronecker-sum/Kuenneth rule L_{AxB} = L_A (x) I + I (x) L_B.

Algebraically exact as a matrix identity; as a statement about a complex it
holds only for an actual product cell structure with product weights, which
productCertificate verifies at degree zero (a staircase SimplicialProduct is
refused: holds() == False). The degree-zero operator there is the U(1)
CONNECTION graph Laplacian connectionLaplacian, not the Hodge L_0. The
spectrum of the Kronecker sum is exactly the pairwise sums of the factor spectra
-- no product eigensolve.)doc")
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
      R"doc(Fermionic second quantization at the SPECTRUM/MATRIX level: free
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
local operator changes. The base operator is LU-factored once (general
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
      R"doc(Dense reference kernels used ONLY below a configurable dimension crossover: on small fixtures they supply the independent answer a structured path
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
}
