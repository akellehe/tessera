// Pybind11 bindings for the quantum subsystem. Lives outside libtessera_quantum
// (which is pybind-free) so the static library can be reused without pulling
// in the Python dependency. This translation unit is always added to
// _tessera's sources (the quantum subsystem is unconditional).
//
// Surface area is deliberately minimal: scalars in, scalars out. No matrix
// product state (MPS) / MPO / ITensor type crosses the Python boundary.
//
// Every Python-visible operation is a method on a coarse-grained class; there
// are no free functions in tessera.quantum. The user-facing classes are
// SchwingerModel, SchwingerQuench, Majorization and Causet, plus the data
// classes (QuantumConfig, GroundStateResult, SchmidtSpectra, …) and the
// MajorizationPredicate hierarchy. The module docstring in
// register_quantum() below summarizes each of them.

// pybind11 must be included *before* any quantum header that transitively pulls
// in <itensor/all.h> (ChoiState.hpp, MutualInformation.hpp, Schmidt.hpp).
// ITensor's itensor/global.h defines NDEBUG unconditionally to silence its own
// asserts. pybind11 enables PYBIND11_DETAILED_ERROR_MESSAGES only while NDEBUG
// is undefined (detail/common.h), and that flag adds a `std::string type`
// member to pybind11::arg_v. Every other binding TU is ITensor-free and so sees
// the larger arg_v; letting ITensor's NDEBUG reach pybind11 here first would
// give this one TU the smaller layout. The two layouts are an ODR violation:
// arg_v's vague-linkage destructor is merged across TUs, and the size-mismatched
// copy over-reads the `type` member and frees a borrowed string-literal pointer
// at import (`free(): invalid pointer`), aborting every Debug build. Release
// defines NDEBUG uniformly, so both sides agree and the bug stays latent.
// Parsing pybind11 first fixes arg_v's layout to match the rest of the module,
// because the member's `#if` is resolved when cast.h is parsed.
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/AnalyticCache.h"
#include "observables/Record.h"
#include "quantum/CausalCompare.hpp"
#include "quantum/CausetChain.hpp"
#include "quantum/ChoiJamiolkowski.h"
#include "quantum/ChoiState.hpp"
#include "quantum/CovarianceState.h"
#include "quantum/DMRGRunner.hpp"
#include "quantum/GradedFock.h"
#include "quantum/Holography.hpp"
#include "quantum/LazyFock.h"
#include "simulations/InteractionSimulation.h"
#include "quantum/Majorization.hpp"
#include "quantum/MutualInformation.hpp"
#include "quantum/KoashiImoto.hpp"
#include "quantum/QuantumSimplex.hpp"
#include "quantum/QuantumVertex.hpp"
#include "quantum/Schmidt.hpp"
#include "quantum/TDVPRunner.hpp"
#include "spacetime/Spacetime.h"  // full type needed for py::cast<Spacetime*>()

namespace py = pybind11;

namespace {
// Coordinate-format (COO) conversion for sparse operators crossing the
// Python boundary. The pybind11/eigen.h sparse binding produces an empty
// CSC under LTO (see EmergentGraph::laplacianCOO), so every
// sparse-returning GradedFock method ships plain (rows, cols, values, n)
// arrays instead.
// Wrap in scipy on the Python side:
//     rows, cols, vals, n = alg.creationMatrixCOO(i)
//     a = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
py::tuple sparseOpToCoo(
    const Eigen::SparseMatrix<std::complex<double>>& op) {
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> cols;
    std::vector<std::complex<double>> vals;
    rows.reserve(static_cast<std::size_t>(op.nonZeros()));
    cols.reserve(static_cast<std::size_t>(op.nonZeros()));
    vals.reserve(static_cast<std::size_t>(op.nonZeros()));
    for (int k = 0; k < op.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(op, k);
             it; ++it) {
            rows.push_back(static_cast<std::int64_t>(it.row()));
            cols.push_back(static_cast<std::int64_t>(it.col()));
            vals.push_back(it.value());
        }
    }
    return py::make_tuple(rows, cols, vals,
                          static_cast<std::int64_t>(op.rows()));
}

// A JSON-able observable Record -> a native Python object (dict/list/scalar)
// and back — the same conversion the observables bindings use for their
// checkpoint records (internal linkage per TU).
py::object quantumRecordToPython(const tessera::observables::Record& r) {
    using Record = tessera::observables::Record;
    switch (r.type()) {
        case Record::Type::Null:
            return py::none();
        case Record::Type::Bool:
            return py::bool_(r.asBool());
        case Record::Type::Int:
            return py::int_(r.asInt());
        case Record::Type::Double:
            return py::float_(r.asDouble());
        case Record::Type::String:
            return py::str(r.asString());
        case Record::Type::List: {
            py::list out;
            for (const auto& e : r.asList()) out.append(quantumRecordToPython(e));
            return out;
        }
        case Record::Type::Map: {
            py::dict out;
            for (const auto& [key, value] : r.asMap())
                out[py::str(key)] = quantumRecordToPython(value);
            return out;
        }
    }
    return py::none();
}

tessera::observables::Record quantumPythonToRecord(const py::handle& o) {
    using Record = tessera::observables::Record;
    if (o.is_none()) return Record();
    if (py::isinstance<py::bool_>(o)) return Record(o.cast<bool>());
    if (py::isinstance<py::int_>(o))
        return Record(static_cast<std::int64_t>(o.cast<long long>()));
    if (py::isinstance<py::float_>(o)) return Record(o.cast<double>());
    if (py::isinstance<py::str>(o)) return Record(o.cast<std::string>());
    if (py::isinstance<py::dict>(o)) {
        Record::Map m;
        for (const auto item : o.cast<py::dict>())
            m[item.first.cast<std::string>()] = quantumPythonToRecord(item.second);
        return Record(std::move(m));
    }
    if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
        Record::List l;
        for (const auto& e : o) l.push_back(quantumPythonToRecord(e));
        return Record(std::move(l));
    }
    throw std::invalid_argument(
        "CovarianceState record: unsupported Python leaf type");
}
}  // namespace

void register_quantum(py::module_ m) {
    using namespace tessera::quantum;
    // InteractionSimulation lives in the top-level tessera namespace; its
    // Python module path stays tessera.quantum.InteractionSimulation.
    using ::tessera::InitialChargeMode;
    using ::tessera::InteractionConfig;
    using ::tessera::InteractionSimulation;

    m.doc() = R"doc(
Schwinger model, density-matrix renormalization group (DMRG), time-dependent
variational principle (TDVP), and causal-order analysis.

The user-facing API is exclusively class-based:

* :class:`SchwingerModel`  — DMRG ground-state pipeline.
* :class:`SchwingerQuench` — q-qbar quench + TDVP + causal-order comparison.
* :class:`Majorization`    — static utility: poset construction and
                              pairwise order-agreement statistics.
* :class:`Causet`          — static utility: tessera.Spacetime → causet
                              adapter.

Plus the data classes (QuantumConfig, GroundStateResult, SchmidtSpectra,
TDVPConfig, …) and the MajorizationPredicate hierarchy
(StandardMajorization, LogConcaveMajorization, PeakRadialMajorization)
that callers configure the workflow with.

The Hamiltonian is the Kogut-Susskind (staggered) Schwinger model after the
Jordan-Wigner transformation and Gauss's-law elimination, expressed as a
spin chain:

.. math::

    H = H_\text{hop} + H_m + H_E

    H_\text{hop} = \frac{1}{4a} \sum_{n=1}^{N-1}
                   (X_n X_{n+1} + Y_n Y_{n+1})

    H_m   = \frac{m}{2} \sum_{n=1}^{N} (-1)^n \sigma^z_n

    H_E   = \frac{g^2 a}{2} \sum_{n=1}^{N-1} L_n^2,
            \quad L_n = L_0 + \sum_{k=1}^{n}\bigl[(1-\sigma^z_k)/2 - (1-(-1)^k)/2\bigr]

References
----------
* Bañuls, Cichy, Jansen, Cirac, arXiv:1305.3765 — the Hamiltonian above and
  the matrix-product-state benchmarks.
* Schollwoeck, arXiv:1008.3477 — DMRG and matrix product states.
* Haegeman et al., arXiv:1103.0936 and arXiv:1408.5056 — TDVP.
* Kogut & Susskind, Phys. Rev. D 11, 395 (1975) — staggered fermions.
* Jordan & Wigner (1928) — the Jordan-Wigner transformation.
* Schwinger, Phys. Rev. 128, 2425 (1962) — the original gauge theory.
* Coleman, Ann. Phys. 101, 239 (1976) — the massive Schwinger model.
)doc";

    // ─── Ground-state config + result ──────────────────────────────────
    py::class_<QuantumConfig>(m, "QuantumConfig",
            R"doc(Configuration for a Schwinger-model DMRG ground-state run.

Bundles the dimensional Hamiltonian parameters and the DMRG sweep
settings into one struct. Default-constructed instances have N = 0 and
must be filled in before passing to :class:`SchwingerModel`.

Attributes
----------
N : int
    Staggered sites, 1-based indexing. Must be ≥ 2; even N is required
    if you want the global GS to live in the Sz = 0 sector.
a : float
    Lattice spacing. Must be positive. Default 1.0.
m : float
    Bare staggered-fermion mass.
g : float
    Gauge coupling. May be 0 (free-Dirac limit, gauge field decouples).
L0 : float
    Background electric field on the link to the left of site 1.
maxBondDim : int
    Cap on the MPS bond dimension during DMRG sweeps.
nSweeps : int
    Total number of DMRG sweeps.
cutoff : float
    SVD truncation threshold per local solve.
krylovDim : int
    Lanczos / Krylov dimension per local 2-site solve.
quiet : bool
    If True (default), suppress ITensor's per-sweep diagnostic prints.
conserveQns : bool
    If True (default), enforce U(1) total-Sz conservation on the SiteSet.
dt : float
    TDVP real-time step size; used by SchwingerQuench, not by SchwingerModel.
T : float
    Total TDVP evolution time; used by SchwingerQuench, not by SchwingerModel.

Examples
--------
>>> from tessera.quantum import QuantumConfig, SchwingerModel
>>> cfg = QuantumConfig()
>>> cfg.N = 20
>>> cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
>>> cfg.maxBondDim = 100; cfg.nSweeps = 12
>>> result = SchwingerModel(cfg).solve()
)doc")
        .def(py::init<>())
        .def_readwrite("N",            &QuantumConfig::N)
        .def_readwrite("a",            &QuantumConfig::a)
        .def_readwrite("m",            &QuantumConfig::m)
        .def_readwrite("g",            &QuantumConfig::g)
        .def_readwrite("L0",           &QuantumConfig::L0)
        .def_readwrite("maxBondDim",   &QuantumConfig::maxBondDim)
        .def_readwrite("nSweeps",      &QuantumConfig::nSweeps)
        .def_readwrite("cutoff",       &QuantumConfig::cutoff)
        .def_readwrite("krylovDim",    &QuantumConfig::krylovDim)
        .def_readwrite("quiet",        &QuantumConfig::quiet)
        .def_readwrite("conserveQns",  &QuantumConfig::conserveQns)
        .def_readwrite("dt",           &QuantumConfig::dt)
        .def_readwrite("T",            &QuantumConfig::T)
        .def("__repr__", [](QuantumConfig const& c) {
            return "QuantumConfig(N=" + std::to_string(c.N) +
                   ", a=" + std::to_string(c.a) +
                   ", m=" + std::to_string(c.m) +
                   ", g=" + std::to_string(c.g) +
                   ", L0=" + std::to_string(c.L0) +
                   ", maxBondDim=" + std::to_string(c.maxBondDim) +
                   ", nSweeps=" + std::to_string(c.nSweeps) + ")";
        });

    py::class_<GroundStateResult>(m, "GroundStateResult",
            R"doc(Result of :meth:`SchwingerModel.solve`.

Attributes
----------
energy : float
    Full physical energy ⟨H⟩ + constant.
operatorEnergy : float
    ⟨H⟩ alone, as returned by ITensor's dmrg().
constant : float
    The c-number shift (g²a/2) Σ_n (c_n² + n/4) pulled out of L_n²
    when AutoMPO-encoding the electric term.
bondDim : int
    Largest bond dimension of the optimized MPS.
truncationErr : float
    Conservative upper bound on the SVD truncation error.
)doc")
        .def_readonly("energy",          &GroundStateResult::energy)
        .def_readonly("operatorEnergy",  &GroundStateResult::operatorEnergy)
        .def_readonly("constant",        &GroundStateResult::constant)
        .def_readonly("bondDim",         &GroundStateResult::bondDim)
        .def_readonly("truncationErr",   &GroundStateResult::truncationErr)
        .def("__repr__", [](GroundStateResult const& r) {
            return "GroundStateResult(energy=" + std::to_string(r.energy) +
                   ", bondDim=" + std::to_string(r.bondDim) +
                   ", truncationErr=" + std::to_string(r.truncationErr) + ")";
        });

    // ─── Schmidt / majorization data classes ──────────────────────────
    py::class_<Interval>(m, "Interval",
            R"doc(1-based contiguous interval [i, j] on the spin chain.)doc")
        .def(py::init<>())
        .def_readwrite("i", &Interval::i)
        .def_readwrite("j", &Interval::j)
        .def("__repr__", [](Interval const& iv) {
            return "Interval(i=" + std::to_string(iv.i) +
                   ", j=" + std::to_string(iv.j) + ")";
        });

    py::class_<SchmidtSpectra>(m, "SchmidtSpectra",
            R"doc(All contiguous-cut Schmidt spectra of an MPS.)doc")
        .def_readonly("N",         &SchmidtSpectra::N)
        .def_readonly("intervals", &SchmidtSpectra::intervals)
        .def_readonly("spectra",   &SchmidtSpectra::spectra);

    py::class_<Poset>(m, "Poset",
            R"doc(Hasse / cover representation of a finite partial order.

Nodes are integers ``0 .. getNodeCount - 1``. ``covers`` lists the cover
edges: each entry ``(a, b)`` means ``a`` strictly precedes ``b`` with no
intermediate node. The full strict order is the transitive closure of
the covers; see :func:`compareOrders` for pairwise statistics derived
from that closure.

Construct empty (``Poset()``) and resize via the ``getNodeCount``
setter, or pass an integer to pre-populate node count
(``Poset(4)``). Mutate via :meth:`addCover` (single edge) or the
``covers`` setter (whole list). The class makes no internal
consistency checks; callers are responsible for transitivity and
acyclicity.

See ``docs/source/causal_sets.md`` for the conceptual background.
)doc")
        .def(py::init<>())
        .def(py::init<int>(), py::arg("nodeCount"),
            "Construct with the node count pre-set to ``nodeCount``.")
        .def("addCover", &Poset::addCover, py::arg("a"), py::arg("b"),
            R"doc(Add the cover edge ``a -> b`` (a strictly precedes b, no intermediate).

Both endpoints must already exist (call the int constructor or the
``getNodeCount`` setter first). No deduplication is performed — adding
the same cover twice creates two parallel edges. Covers normally come
from a transitive reduction, where duplicates cannot arise.
)doc")
        .def_property("getNodeCount",
            [](Poset const& p) { return p.getNodeCount(); },
            [](Poset& p, int n) { p.setNodeCount(n); },
            "Number of nodes. Setting grows the node set; nodes are not "
            "removed if you set a smaller value, and cover edges are "
            "preserved across resizes.")
        .def("getCoverCount", &Poset::getCoverCount,
            "Number of cover edges currently registered.")
        .def_property("covers",
            [](Poset const& p) { return p.covers(); },
            [](Poset& p, std::vector<std::pair<int, int>> const& covers) {
                p.setCovers(covers);
            },
            "Cover edges as a list of ``(a, b)`` pairs. Setting replaces "
            "the entire cover list in one pass.")
        .def("toDot", &Poset::toDot,
            "Graphviz DOT representation of the Hasse diagram. "
            "Nodes labelled by their integer id; one directed edge per "
            "cover. Suitable for ``dot -Tsvg`` rendering.")
        .def_static("fromSpacetime",
            [](py::object spacetime_obj) {
                auto const* st = spacetime_obj.cast<tessera::spacetime::Spacetime const*>();
                return tessera::Poset::fromSpacetime(*st);
            }, py::arg("spacetime"),
            R"doc(Build the causet partial order on a Spacetime's vertices.

Reads the directed-edge / timelike-edge subgraph as the strict
``precedes`` relation, then takes the transitive reduction to recover
cover edges. The result has one node per Spacetime vertex (in
ascending ID order); cover edges are between strictly comparable
vertices with no intermediate.
)doc")
        .def("__repr__", [](Poset const& p) {
            return "Poset(getNodeCount=" + std::to_string(p.getNodeCount()) +
                   ", covers=" + std::to_string(p.getCoverCount()) + " edges)";
        });

    py::class_<GroundStateMajorizationResult>(m, "GroundStateMajorizationResult",
            R"doc(Result of :meth:`SchwingerModel.solveWithMajorization`.

Attributes
----------
groundState : GroundStateResult
    Same diagnostics returned by :meth:`SchwingerModel.solve`.
spectra : SchmidtSpectra
    All contiguous-cut Schmidt spectra of the ground-state MPS.
poset : Poset
    Hasse cover edges of the strict-majorization order on
    ``spectra.spectra``.
)doc")
        .def_readonly("groundState", &GroundStateMajorizationResult::groundState)
        .def_readonly("spectra",     &GroundStateMajorizationResult::spectra)
        .def_readonly("poset",       &GroundStateMajorizationResult::poset);

    // ─── MajorizationPredicate hierarchy ────────────────────────────────
    py::class_<MajorizationPredicate>(m, "MajorizationPredicate",
            R"doc(Abstract base class for variants of the majorization predicate.

Concrete subclasses:

* :class:`StandardMajorization` -- classical majorization
  (Nielsen, arXiv:quant-ph/9811053).
* :class:`LogConcaveMajorization` -- gated on log-concave spectra
  (Brändén, arXiv:1410.6601).
* :class:`PeakRadialMajorization` -- peak-relative entrywise dominance.

Methods that take a predicate -- :meth:`Majorization.posetOf` and
:meth:`SchwingerQuench.compareCausalOrders` -- accept any concrete
subclass instance.
)doc")
        .def("majorizes",
             &MajorizationPredicate::majorizes,
             py::arg("mu"), py::arg("lambda_"),
             "True iff μ majorizes λ under this variant.")
        .def("strictlyMajorizes",
             &MajorizationPredicate::strictlyMajorizes,
             py::arg("mu"), py::arg("lambda_"),
             "True iff μ strictly majorizes λ under this variant.")
        .def_property_readonly("name",
             &MajorizationPredicate::name,
             "Short identifier of the variant.");

    py::class_<StandardMajorization, MajorizationPredicate>(
            m, "StandardMajorization",
            R"doc(Classical Nielsen majorization.

μ ≻ λ iff for every k, sum of the top-k of μ ≥ sum of the top-k of λ,
with equality at the total-mass step (Nielsen, arXiv:quant-ph/9811053).
)doc")
        .def(py::init<double>(), py::arg("tol") = 1e-12)
        .def_property_readonly("tol", &StandardMajorization::tol);

    py::class_<LogConcaveMajorization, StandardMajorization>(
            m, "LogConcaveMajorization",
            R"doc(Standard majorization, restricted to log-concave spectra
(Brändén, arXiv:1410.6601).
)doc")
        .def(py::init<double>(), py::arg("tol") = 1e-12)
        .def_static("isLogConcave",
                    &LogConcaveMajorization::isLogConcave,
                    py::arg("v"), py::arg("tol") = 1e-12,
                    "True iff the spectrum is log-concave on its support.");

    py::class_<PeakRadialMajorization, MajorizationPredicate>(
            m, "PeakRadialMajorization",
            R"doc(Peak-radial dominance: relative-to-peak entrywise majorization.
Strictly stronger than classical majorization.
)doc")
        .def(py::init<double>(), py::arg("tol") = 1e-12)
        .def_property_readonly("tol", &PeakRadialMajorization::tol);

    // ─── TDVP quench config + snapshot ─────────────────────────────────
    py::class_<TDVPConfig>(m, "TDVPConfig",
            R"doc(Configuration for the q-qbar quench + TDVP run.

Bundles the Hamiltonian parameters, the DMRG ground-state setup, the
quench location / separation, and the real-time-evolution schedule.
``i0`` and ``d`` describe the q-qbar pair: σ⁻ acts at site ``i0``,
σ⁺ at site ``i0 + d``. With ``quenchEnforceParity = True`` (default)
``i0`` must be odd and ``d`` must be odd.

The pipeline runs through :meth:`SchwingerQuench.evolve` (or
:meth:`SchwingerQuench.compareCausalOrders` for the causal-order
extension).
)doc")
        .def(py::init<>())
        .def_readwrite("N",                       &TDVPConfig::N)
        .def_readwrite("a",                       &TDVPConfig::a)
        .def_readwrite("m",                       &TDVPConfig::m)
        .def_readwrite("g",                       &TDVPConfig::g)
        .def_readwrite("L0",                      &TDVPConfig::L0)
        .def_readwrite("dmrgMaxBondDim",          &TDVPConfig::dmrgMaxBondDim)
        .def_readwrite("dmrgNSweeps",             &TDVPConfig::dmrgNSweeps)
        .def_readwrite("dmrgKrylovDim",           &TDVPConfig::dmrgKrylovDim)
        .def_readwrite("dmrgCutoff",              &TDVPConfig::dmrgCutoff)
        .def_readwrite("i0",                      &TDVPConfig::i0)
        .def_readwrite("d",                       &TDVPConfig::d)
        .def_readwrite("quenchEnforceParity",     &TDVPConfig::quenchEnforceParity)
        .def_readwrite("dt",                      &TDVPConfig::dt)
        .def_readwrite("T",                       &TDVPConfig::T)
        .def_readwrite("maxBondDim",              &TDVPConfig::maxBondDim)
        .def_readwrite("krylovDim",               &TDVPConfig::krylovDim)
        .def_readwrite("cutoff",                  &TDVPConfig::cutoff)
        .def_readwrite("snapshotEvery",           &TDVPConfig::snapshotEvery)
        .def_readwrite("quiet",                   &TDVPConfig::quiet)
        .def_readwrite("conserveQns",             &TDVPConfig::conserveQns)
        .def_readwrite("recordSpectra",           &TDVPConfig::recordSpectra)
        .def_readwrite("recordPoset",             &TDVPConfig::recordPoset)
        .def_readwrite("recordMutualInformation", &TDVPConfig::recordMutualInformation)
        .def_readwrite("recordBondMutualInformation", &TDVPConfig::recordBondMutualInformation)
        .def_readwrite("hoppingPairs",            &TDVPConfig::hoppingPairs,
            "Optional custom hopping graph as a list of (i, j) "
            "0-based flat-lattice pairs. Empty = default 1D NN chain. "
            "Sourced from tessera.quantum.Causet.chainFrom(spacetime).");

    py::class_<TDVPSnapshot>(m, "TDVPSnapshot",
            R"doc(Per-step diagnostics recorded during a TDVP run.

Always-populated fields: time, energy, bondDim, zProfile, lProfile.
Optional fields (populated only if the corresponding TDVPConfig flag
is set): spectra, poset, mutualInformation.
)doc")
        .def_readonly("time",      &TDVPSnapshot::time)
        .def_readonly("energy",    &TDVPSnapshot::energy)
        .def_readonly("bondDim",   &TDVPSnapshot::bondDim)
        .def_readonly("zProfile",  &TDVPSnapshot::zProfile)
        .def_readonly("lProfile",  &TDVPSnapshot::lProfile)
        .def_readonly("spectra",   &TDVPSnapshot::spectra)
        .def_readonly("poset",     &TDVPSnapshot::poset)
        .def_readonly("bondMutualInformation",
                       &TDVPSnapshot::bondMutualInformation,
                       R"doc(Symmetric (N-1)×(N-1) bond-cut tripartite-information matrix in nats,
flattened row-major. Zero diagonal.
Populated iff TDVPConfig.recordBondMutualInformation = True.)doc")
        .def_readonly("mutualInformation", &TDVPSnapshot::mutualInformation,
                       "Flat row-major N×N matrix of site-site MI in nats. "
                       "Populated iff TDVPConfig.recordMutualInformation = True.")
        .def("__repr__", [](TDVPSnapshot const& s) {
            return "TDVPSnapshot(time=" + std::to_string(s.time) +
                   ", energy=" + std::to_string(s.energy) +
                   ", bondDim=" + std::to_string(s.bondDim) + ")";
        });

    py::class_<QuenchResult>(m, "QuenchResult",
            R"doc(Result of :meth:`SchwingerQuench.evolve`.

Attributes
----------
groundState : GroundStateResult
    DMRG ground-state diagnostics for the pre-quench state.
snapshots : list[TDVPSnapshot]
    Per-step diagnostics; ``snapshots[0]`` is the post-quench state.
)doc")
        .def_readonly("groundState", &QuenchResult::groundState)
        .def_readonly("snapshots",   &QuenchResult::snapshots);

    // ─── Causal-order comparison data classes ─────────────────────────
    py::class_<LabelSpacetime>(m, "LabelSpacetime",
            R"doc(One label in a (cut, time) spacetime.)doc")
        .def_readonly("cutIdx",    &LabelSpacetime::cutIdx)
        .def_readonly("tIdx",      &LabelSpacetime::tIdx)
        .def_readonly("intervalI", &LabelSpacetime::intervalI)
        .def_readonly("intervalJ", &LabelSpacetime::intervalJ)
        .def_readonly("time",      &LabelSpacetime::time);

    py::class_<CausalOrders>(m, "CausalOrders",
            R"doc(Three Hasse-cover posets on a shared (cut, time) label set.

Build via :meth:`CausalOrders.fromSnapshots` (which delegates to the
underlying SchwingerQuench pipeline; most users go through
:meth:`SchwingerQuench.compareCausalOrders` instead).
)doc")
        .def_readonly("labels", &CausalOrders::labels)
        .def_readonly("maj",    &CausalOrders::maj)
        .def_readonly("lr",     &CausalOrders::lr)
        .def_readonly("cs",     &CausalOrders::cs)
        .def_static("fromSnapshots",
            [](std::vector<TDVPSnapshot> const& snapshots,
               double vLr,
               MajorizationPredicate const* predicate) {
                return CausalOrders::fromSnapshots(snapshots, vLr, predicate);
            },
            py::arg("snapshots"), py::arg("vLr"), py::arg("predicate") = nullptr,
            R"doc(Build the three orders from a list of TDVP snapshots.)doc");

    py::class_<OrderAgreement>(m, "OrderAgreement",
            R"doc(Pairwise agreement statistics between two posets.

Counted over unordered pairs (i, j) with i < j:
* concordant — both orders relate the pair, in the same direction.
* discordant — both orders relate the pair, in opposite directions.
* only_a / only_b — exactly one order relates the pair.

Build via :meth:`Majorization.agreement(a, b, n_labels)`.
)doc")
        .def_readonly("kendallTau",         &OrderAgreement::kendallTau)
        .def_readonly("discordantFraction", &OrderAgreement::discordantFraction)
        .def_readonly("hasseEditDistance",  &OrderAgreement::hasseEditDistance)
        .def_readonly("nConcordant",        &OrderAgreement::nConcordant)
        .def_readonly("nDiscordant",        &OrderAgreement::nDiscordant)
        .def_readonly("nComparableBoth",    &OrderAgreement::nComparableBoth)
        .def_readonly("nOnlyA",             &OrderAgreement::nOnlyA)
        .def_readonly("nOnlyB",             &OrderAgreement::nOnlyB);

    m.def("compareOrders", &tessera::compareOrders,
        py::arg("a"), py::arg("b"), py::arg("nLabels"),
        R"doc(Pairwise agreement statistics between two posets on a shared label set.

Counts unordered pairs (i, j) with i < j in five disjoint buckets via
Floyd–Warshall transitive closures of `a` and `b`:

* concordant   — both orders relate the pair the same way
* discordant   — both orders relate the pair, opposite ways
* only-a       — `a` relates the pair, `b` does not
* only-b       — `b` relates the pair, `a` does not
* neither      — neither order relates the pair

Returns an :class:`OrderAgreement` with ``kendallTau``,
``discordantFraction``, ``hasseEditDistance``, and the five counts.

Complexity: O(nLabels^3) for the transitive closure, O(nLabels^2) for
the pair counts. Practical up to a few thousand labels.

See ``docs/source/causal_sets.md`` for the methodology context.
)doc");

    py::class_<CausalComparisonReport>(m, "CausalComparisonReport",
            R"doc(Pairwise agreement statistics across the three causal orders (≼_maj, ≼_LR, ≼_cs).

Build via :meth:`SchwingerQuench.compareCausalOrders`.
)doc")
        .def_readonly("majVsLr",    &CausalComparisonReport::majVsLr)
        .def_readonly("majVsCs",    &CausalComparisonReport::majVsCs)
        .def_readonly("lrVsCs",     &CausalComparisonReport::lrVsCs)
        .def_readonly("nLabels",    &CausalComparisonReport::nLabels)
        .def_readonly("nSnapshots", &CausalComparisonReport::nSnapshots)
        .def_readonly("vLr",        &CausalComparisonReport::vLr)
        .def_readonly("majKind",    &CausalComparisonReport::majKind);

    // ─── Causet adapter data classes ───────────────────────────────────
    py::class_<CausetChain>(m, "CausetChain",
            R"doc(Spacetime-derived chain layout for the Schwinger MPO.

Build via :meth:`Causet.chainFrom`.

Attributes
----------
nSites : int
    Total number of lattice sites.
times : list[int]
    Sorted ascending list of integer time slices.
antichains : list[list[int]]
    ``antichains[s]`` is the ascending-ID list of vertex IDs at
    ``times[s]``.
vertexIds : list[int]
    Flat lattice site → Spacetime vertex ID.
hoppingPairs : list[tuple[int, int]]
    Pairs ``(i, j)`` of flat lattice sites coupled by adjacent-slice
    timelike causet edges.
partialOrder : Poset
    Hasse cover Poset on the nSites label set.
)doc")
        .def_readonly("nSites",       &CausetChain::nSites)
        .def_readonly("times",        &CausetChain::times)
        .def_readonly("antichains",   &CausetChain::antichains)
        .def_readonly("vertexIds",    &CausetChain::vertexIds)
        .def_readonly("hoppingPairs", &CausetChain::hoppingPairs)
        .def_readonly("partialOrder", &CausetChain::partialOrder)
        .def("__repr__", [](CausetChain const& c) {
            return "CausetChain(nSites=" + std::to_string(c.nSites) +
                   ", times=" + std::to_string(c.times.size()) +
                   ", hops=" + std::to_string(c.hoppingPairs.size()) + ")";
        });

    // ─── Coarse-grained workflow classes ────────────────────────────────

    py::class_<SchwingerModel>(m, "SchwingerModel",
            R"doc(Schwinger-model DMRG ground-state pipeline.

Coarse-grained interface for the ground-state workflow:
bundle a :class:`QuantumConfig` and call :meth:`solve` for the bare
DMRG diagnostics or :meth:`solveWithMajorization` for the full Schmidt
+ majorization-poset bundle.

The model is stateless beyond its config — every method runs the
underlying ITensor pipeline from scratch.

Examples
--------
>>> from tessera.quantum import QuantumConfig, SchwingerModel
>>> cfg = QuantumConfig()
>>> cfg.N = 4; cfg.a = 1.0; cfg.g = 1.0; cfg.m = 0.0; cfg.L0 = 0.0
>>> cfg.maxBondDim = 32; cfg.nSweeps = 8
>>> r = SchwingerModel(cfg).solve()
>>> abs(r.operatorEnergy - (-1.738676174)) < 1e-8
True
)doc")
        .def(py::init<QuantumConfig>(), py::arg("config"))
        .def_property_readonly("config",
            [](SchwingerModel const& m) { return m.config(); },
            "The bound QuantumConfig (read-only).")
        .def("solve", &SchwingerModel::solve,
            R"doc(Run DMRG to the Schwinger ground state.

Returns
-------
GroundStateResult
    Energy, bond-dim, and truncation diagnostics.

Raises
------
RuntimeError or ValueError
    If config.N < 2 or config.a <= 0.
)doc")
        .def("solveWithMajorization",
            &SchwingerModel::solveWithMajorization,
            py::arg("tol") = 1e-12,
            R"doc(Run DMRG, then extract Schmidt spectra and majorization poset.

Single-shot pipeline:
1. DMRG ground state of the Schwinger MPO.
2. Schmidt spectrum of every contiguous bipartition (excluding the
   trivial full-chain cut).
3. Majorization poset on those spectra (Hasse cover edges only).

Parameters
----------
tol : float, optional
    Slack for the majorization comparisons. Default 1e-12.

Returns
-------
GroundStateMajorizationResult
)doc");

    py::class_<SchwingerQuench>(m, "SchwingerQuench",
            R"doc(Schwinger-model q-qbar quench + TDVP pipeline.

Coarse-grained interface for the dynamics workflow: bundle a
:class:`TDVPConfig` and call :meth:`evolve` for the snapshot
trajectory or :meth:`compareCausalOrders` for the causal-order
comparison. The model is stateless beyond its config.

Examples
--------
>>> from tessera.quantum import TDVPConfig, SchwingerQuench
>>> cfg = TDVPConfig()
>>> cfg.N = 14; cfg.m = 20.0; cfg.g = 1.0
>>> cfg.i0 = 5; cfg.d = 5
>>> cfg.dt = 0.05; cfg.T = 5.0; cfg.snapshotEvery = 5
>>> r = SchwingerQuench(cfg).evolve()  # doctest: +SKIP
>>> r.snapshots[0].lProfile[:3]        # doctest: +SKIP
[-1.0, -0.0, -0.0]
)doc")
        .def(py::init<TDVPConfig>(), py::arg("config"))
        .def_property_readonly("config",
            [](SchwingerQuench const& q) { return q.config(); },
            "The bound TDVPConfig (read-only).")
        .def("evolve", &SchwingerQuench::evolve,
            R"doc(Run the full DMRG → quench → TDVP pipeline.

Returns
-------
QuenchResult
    Ground-state diagnostics + snapshot list.
)doc")
        .def("compareCausalOrders",
            [](SchwingerQuench const& q,
               double vLr,
               MajorizationPredicate const* predicate) {
                return q.compareCausalOrders(vLr, predicate);
            },
            py::arg("vLr") = 1.0,
            py::arg("predicate") = nullptr,
            R"doc(End-to-end causal-order comparison.

Runs :meth:`evolve` (forcing recordSpectra=True), builds three
partial orders on the (cut, time) label set:

* ≼_maj — strict-majorization on Schmidt spectra (across cuts AND times).
* ≼_LR  — Lieb-Robinson cone: dist(A, B) ≤ vLr · (t_B - t_A).
* ≼_cs  — causet (time-only on regular chain).

Pairwise agreement is reported as Kendall-τ, the discordant-pair
fraction, and the Hasse-graph edit distance.

Parameters
----------
vLr : float, optional
    Lieb-Robinson velocity in lattice units. Default 1.0.
predicate : MajorizationPredicate, optional
    Majorization variant for ≼_maj. None = StandardMajorization (default).

Returns
-------
CausalComparisonReport
)doc");

    py::class_<Majorization>(m, "Majorization",
            R"doc(Static utility for majorization-poset construction and pairwise
order-agreement statistics. Not instantiable; call methods on the class.

>>> from tessera.quantum import Majorization, StandardMajorization
>>> p = Majorization.posetOf([[1.0], [0.5, 0.5], [1/3]*3])
>>> sorted(p.covers)
[(0, 1), (1, 2)]
)doc")
        // Two static overloads of `posetOf` (one with predicate, one with
        // tol). Spelled out explicitly because pybind11 needs help selecting
        // between them.
        .def_static("posetOf",
            [](std::vector<std::vector<double>> const& spectra,
               MajorizationPredicate const& predicate) {
                return Majorization::posetOf(spectra, predicate);
            },
            py::arg("spectra"), py::arg("predicate"),
            R"doc(Build the majorization poset under an explicit predicate.)doc")
        .def_static("posetOf",
            [](std::vector<std::vector<double>> const& spectra, double tol) {
                return Majorization::posetOf(spectra, tol);
            },
            py::arg("spectra"), py::arg("tol") = 1e-12,
            R"doc(Build the classical Nielsen majorization poset at the given tolerance.)doc")
        .def_static("agreement",
            &Majorization::agreement,
            py::arg("a"), py::arg("b"), py::arg("nLabels"),
            R"doc(Pairwise agreement statistics between two Posets on the same
label set of size nLabels. Returns an :class:`OrderAgreement`.
)doc");

    py::class_<Causet>(m, "Causet",
            R"doc(Static utility for tessera.Spacetime → causet adapters.
Not instantiable; call methods on the class.
)doc")
        .def_static("chainFrom",
            [](py::object spacetime_obj) {
                auto const* st = spacetime_obj.cast<tessera::spacetime::Spacetime const*>();
                return Causet::chainFrom(*st);
            }, py::arg("spacetime"),
            R"doc(Extract a chain-of-antichains adapter from a Spacetime.

Walks the Spacetime's vertex list, groups vertices by integer time
slice, and packages the antichain layering, the flat-lattice ↔
Spacetime ID mapping, the adjacent-slice timelike-edge hopping pairs,
and the inherited Hasse cover :class:`Poset`.

Parameters
----------
spacetime : tessera.Spacetime
    Source spacetime.

Returns
-------
CausetChain
)doc");

    // ─── MutualInformation utility ─────────────────────────────────────
    py::class_<MutualInformation>(m, "MutualInformation",
            R"doc(Static utility for site-site mutual information on a Schwinger MPS.

Not instantiable. The full computation pipeline goes through
SchwingerQuench(cfg).evolve() with TDVPConfig.recordMutualInformation
= True; this class is exposed so callers can run the pure-math
operations (vonNeumannEntropy, edgeLength) on numpy data without
needing an MPS in hand.
)doc")
        .def_static("vonNeumannEntropy",
            static_cast<double(*)(std::vector<double> const&, double)>(
                &MutualInformation::vonNeumannEntropy),
            py::arg("eigenvalues"), py::arg("tol") = 1e-12,
            R"doc(Von Neumann / Shannon entropy of an already-diagonal spectrum
(e.g. a Schmidt spectrum), in nats: -sum p_i log p_i over p_i > tol.

Pass the eigenvalue / probability list directly — no need to wrap it in a
diagonal density matrix (which the matrix overload would re-diagonalise).
)doc")
        .def_static("vonNeumannEntropy",
            [](py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast> rho,
                double tol) -> double {
                auto buf = rho.unchecked<2>();
                const int r = static_cast<int>(buf.shape(0));
                const int c = static_cast<int>(buf.shape(1));
                if (r != c) {
                    throw std::invalid_argument(
                        "MutualInformation.vonNeumannEntropy: rho must be square");
                }
                Eigen::MatrixXcd m(r, c);
                for (int i = 0; i < r; ++i) {
                    for (int j = 0; j < c; ++j) {
                        m(i, j) = buf(i, j);
                    }
                }
                return MutualInformation::vonNeumannEntropy(m, tol);
            },
            py::arg("rho"), py::arg("tol") = 1e-12,
            R"doc(Von Neumann entropy of a Hermitian density matrix, in nats.

Accepts any square dimension; the holography pipeline uses 2×2
(single-site marginals) and 4×4 (two-site joint reduced density
matrices).
)doc")
        .def_static("edgeLength",
            static_cast<double(*)(double, double) noexcept>(
                &MutualInformation::edgeLength),
            py::arg("I"), py::arg("epsilon") = 1e-10,
            R"doc(ℓ = -log(I) with infinity floor at -log(epsilon).)doc")
        .def_static("edgeLength",
            static_cast<Eigen::MatrixXd(*)(Eigen::MatrixXd const&, double)>(
                &MutualInformation::edgeLength),
            py::arg("I"), py::arg("epsilon") = 1e-10,
            R"doc(Elementwise edge length on a matrix of mutual-information values:
ell_{ij} = -log(I_{ij}), with +inf where I_{ij} < epsilon. Vectorised form of
the scalar overload, so a B x B bond-MI matrix maps in one call.)doc");

    // ─── ChoiJamiolkowski: dense map–state duality ("bending") ─────────
    py::class_<ChoiJamiolkowski>(m, "ChoiJamiolkowski",
            R"doc(Static utility for the dense Choi–Jamiołkowski map–state
duality ("bending"). Not instantiable; call the methods on the class.

Operators and states are flat, row-major lists of complex numbers: a dA×dB
operator U has ``U[i*dB + j] = U_{ij}``. Conventions:

* ``vec(U) = Σ_{ij} U_{ij} |i⟩_A ⊗ |j⟩_B`` (the row-major flatten);
* ``vec(|a⟩⟨b|) = a ⊗ conj(b)`` (separable, Schmidt rank 1);
* ``⟨psiA|U|psiB⟩ = ⟨vec(U_T)|vec(U)⟩ = Tr(U_T^H·U)`` with
  ``U_T = |psiA⟩⟨psiB|``;
* Schmidt rank of ``vec(U)`` = number of nonzero singular values of U.
)doc")
        .def_static("vectorize", &ChoiJamiolkowski::vectorize,
            py::arg("U"), py::arg("dA"), py::arg("dB"),
            R"doc(Vectorise a dA×dB operator: vec(U) = Σ_{ij} U_{ij} |i⟩⊗|j⟩,
the length-(dA·dB) row-major flatten.)doc")
        .def_static("unvectorize", &ChoiJamiolkowski::unvectorize,
            py::arg("v"), py::arg("dA"), py::arg("dB"),
            R"doc(Un-vectorise (the inverse of vectorize): reshape a
length-(dA·dB) state back into the dA×dB operator U_{ij} = v[i·dB + j],
so unvectorize(vectorize(U)) == U.)doc")
        .def_static("singularValues", &ChoiJamiolkowski::singularValues,
            py::arg("U"), py::arg("dA"), py::arg("dB"),
            R"doc(Singular values of the dA×dB operator U (descending); the
Schmidt coefficients of vec(U).)doc")
        .def_static("schmidtRank", &ChoiJamiolkowski::schmidtRank,
            py::arg("U"), py::arg("dA"), py::arg("dB"), py::arg("tol") = 1e-10,
            R"doc(Schmidt rank of vec(U): the number of singular values of U
exceeding tol·σ_max.)doc")
        .def_static("transitionOperator", &ChoiJamiolkowski::transitionOperator,
            py::arg("psiA"), py::arg("psiB"), py::arg("dA"), py::arg("dB"),
            R"doc(Transition operator U_T = |psiA⟩⟨psiB| (rank one), returned
flat row-major.)doc")
        .def_static("transitionAmplitude", &ChoiJamiolkowski::transitionAmplitude,
            py::arg("psiA"), py::arg("U"), py::arg("psiB"), py::arg("dA"), py::arg("dB"),
            R"doc(Transition amplitude ⟨psiA|U|psiB⟩ = Σ_{ij}
conj(psiA_i)·U_{ij}·psiB_j.)doc")
        .def_static("choiState", &ChoiJamiolkowski::choiState,
            py::arg("U"), py::arg("d"),
            R"doc(Choi–Jamiołkowski state (U⊗I)|Φ⁺⟩ = (1/√d)·vec(U) of a square
d×d operator (length d·d, row-major; a unit vector for unitary U).)doc")
        .def_static("operatorFromChoiState", &ChoiJamiolkowski::operatorFromChoiState,
            py::arg("state"), py::arg("d"),
            R"doc(Recover U from its Choi state (the inverse of choiState):
U = √d · unvectorize(state), flat row-major (length d·d); the original
operator up to the global phase the state carries.)doc")
        .def_static("choiMatrix", &ChoiJamiolkowski::choiMatrix,
            py::arg("U"), py::arg("d"),
            R"doc(Choi matrix J(U) = |state⟩⟨state| of a square d×d operator
(flat row-major (d·d)×(d·d); Tr J = 1 for unitary U).)doc");

    // ─── InteractionSimulation: interaction-history Monte Carlo ────────
    // See docs/source/interaction-history-monte-carlo.md.
    py::class_<InteractionConfig>(m, "InteractionConfig",
        R"doc(Configuration for an interaction-history Monte Carlo run.

nSystems randomized correlated mixed-state systems on a Poisson-Delaunay
initial layer (delaunayEdges is the connectivity, supplied by the
caller); the Schwinger two-site unitary exp(-i H_XY dt) drives each
interaction; beta is the inverse temperature in e^{-beta S}.
)doc")
        .def(py::init<>())
        .def_readwrite("nSystems",           &InteractionConfig::nSystems)
        .def_readwrite("a",                  &InteractionConfig::a)
        .def_readwrite("g",                  &InteractionConfig::g)
        .def_readwrite("m",                  &InteractionConfig::m)
        .def_readwrite("dt",                 &InteractionConfig::dt)
        .def_readwrite("beta",               &InteractionConfig::beta)
        .def_readwrite("epsilonI",           &InteractionConfig::epsilonI)
        .def_readwrite("targetInteractions",
                       &InteractionConfig::targetInteractions)
        .def_readwrite("delaunayEdges",      &InteractionConfig::delaunayEdges)
        .def_readwrite("useCharges",         &InteractionConfig::useCharges)
        .def_readwrite("featureCharges",
                       &InteractionConfig::featureCharges)
        .def_readwrite("featureDeactivateOnAnnihilate",
                       &InteractionConfig::featureDeactivateOnAnnihilate)
        .def_readwrite("featurePhotonOnAnnihilate",
                       &InteractionConfig::featurePhotonOnAnnihilate)
        .def_readwrite("featureQuditBasis",
                       &InteractionConfig::featureQuditBasis)
        .def_readwrite("featureChoiSigmaAB",
                       &InteractionConfig::featureChoiSigmaAB)
        .def_readwrite("j_chargeCharge",
                       &InteractionConfig::j_chargeCharge)
        .def_readwrite("j_spinSpin",
                       &InteractionConfig::j_spinSpin)
        .def_readwrite("massShift",
                       &InteractionConfig::massShift)
        .def_readwrite("gammaCpViolation",
                       &InteractionConfig::gammaCpViolation)
        .def_readwrite("dtPair",
                       &InteractionConfig::dtPair)
        .def_readwrite("cpBias",             &InteractionConfig::cpBias)
        .def_readwrite("initialChargeMode",
                       &InteractionConfig::initialChargeMode)
        .def_readwrite("seed",               &InteractionConfig::seed)
        .def_readwrite("quiet",              &InteractionConfig::quiet);

    py::enum_<tessera::simulations::InitialChargeMode>(m, "InitialChargeMode")
        .value("ALTERNATING",
               tessera::simulations::InitialChargeMode::ALTERNATING)
        .value("RANDOM",
               tessera::simulations::InitialChargeMode::RANDOM);

    py::class_<InteractionSimulation>(m, "InteractionSimulation",
        R"doc(Metropolis Monte Carlo over interaction histories, weighted by
the geometric Regge action on the dual lattice.

Mirrors tessera.CDT: the move primitives interact() / unInteract(), the
driving loop sweep() / thermalize() / tune(), and the diagnostics
computeAction() / getSpectralDimension() / getAcceptanceRates(). The
object of the search is the beta at which the emergent spectral
dimension reaches 4.
)doc")
        .def(py::init<InteractionConfig>(), py::arg("config"))
        .def("interact",   &InteractionSimulation::interact,
             R"doc(Propose + Metropolis-accept one interaction. Returns acceptance.)doc")
        .def("unInteract", &InteractionSimulation::unInteract,
             R"doc(Propose + Metropolis-accept one un-interaction. Returns acceptance.)doc")
        .def("sweep",      &InteractionSimulation::sweep,
             R"doc(One Monte Carlo sweep; returns the number of accepted moves.)doc")
        .def("thermalize", &InteractionSimulation::thermalize,
             R"doc(Tune to the target volume, then sweep to equilibrium.)doc")
        .def("tune",       &InteractionSimulation::tune,
             py::arg("progress") = nullptr,
             R"doc(Grow the complex toward targetInteractions.)doc")
        .def("computeAction", &InteractionSimulation::computeAction,
             R"doc(The geometric Regge action S = sum_h A_h eps_h.)doc")
        .def("getSpectralDimension",
             &InteractionSimulation::getSpectralDimension,
             py::arg("sigmas"), py::arg("krylovDim") = 30,
             R"doc(Heat-kernel spectral dimension D_S(sigma) of the MI-weighted complex.)doc")
        .def("getDeficitAngleDistribution",
             &InteractionSimulation::getDeficitAngleDistribution,
             R"doc(Deficit angles over the interior hinges.)doc")
        .def("getVolumeProfile", &InteractionSimulation::getVolumeProfile,
             R"doc(Interaction-count profile by time slice.)doc")
        .def("getAcceptanceRates",
             &InteractionSimulation::getAcceptanceRates,
             R"doc(Accepted / attempted ratio per move type.)doc")
        .def("annihilate", &InteractionSimulation::annihilate,
             R"doc(Spontaneous partial annihilation of a (+, -) frontier pair.)doc")
        .def("pairCreate", &InteractionSimulation::pairCreate,
             R"doc(Spontaneous (+, -) pair creation with a Bell joint.)doc")
        .def("getGlobalCharge", &InteractionSimulation::getGlobalCharge,
             R"doc(Total signed charge across the complex.)doc")
        .def("getChargeProfile", &InteractionSimulation::getChargeProfile,
             R"doc(Per-time-slice (n_+, n_0, n_-, sum_q).)doc")
        .def("getChargeCorrelation",
             &InteractionSimulation::getChargeCorrelation,
             py::arg("maxDist"),
             R"doc(<q_v . q_w> as a function of graph distance.)doc")
        .def("quditChargeOf", &InteractionSimulation::quditChargeOf,
             py::arg("vertex"),
             R"doc(A single vertex's continuous charge via Tr[ρ · Q̂].

Q̂ = diag(+1, +1, -1, -1) on the {|+0⟩, |+1⟩, |−0⟩, |−1⟩} basis.
For an integer-charge eigenstate this returns ±1; for the maximally-mixed
I/4 proxy it returns 0; for an arbitrary mixed state, the value sits in
[−1, +1]. Requires ``featureQuditBasis = True``. Returns 0.0 for vertices
the simulation has no qudit state for.)doc")
        .def("quditStateOf",
             [](const InteractionSimulation &self, tessera::mesh::VertexPtr v)
                 -> py::object {
               const auto &m = self.quditStateOfMap();
               auto it = m.find(v);
               if (it == m.end()) return py::none();
               return py::cast(it->second);
             },
             py::arg("vertex"),
             R"doc(A single vertex's 4×4 qudit density matrix, or ``None`` if
no qudit state is stored. Exposes per-vertex purity, charge content and
basis populations directly, rather than through the projected
``Tr[ρ · Q̂]`` accessor. Requires ``featureQuditBasis = True``.)doc")
        .def("quditJointStateFor",
             &InteractionSimulation::quditJointStateFor,
             py::arg("x"), py::arg("y"),
             R"doc(16×16 joint qudit state ρ_XY for a pair.

Returns the stored correlated joint when (x, y) share an interaction
history or are initial-layer Delaunay neighbours; otherwise the
uncorrelated product ρ_x ⊗ ρ_y.)doc")
        .def("getSpacetime", &InteractionSimulation::getSpacetime,
             R"doc(The interaction-history simplicial complex (the primal).)doc")
        .def_property_readonly("interactionCount",
             &InteractionSimulation::interactionCount)
        .def_property_readonly("frontierSize",
             &InteractionSimulation::frontierSize)
        .def_property("beta", &InteractionSimulation::getBeta,
             &InteractionSimulation::setBeta)
        .def("setSeed", &InteractionSimulation::setSeed, py::arg("seed"));

    // ─── Holography submodule: emergent spectral dimension ─────────────
    auto holo = m.def_submodule("holography",
        R"doc(Emergent spectral dimension from the Schwinger TDVP state.

The pipeline runs through one workflow class:

>>> from tessera.quantum import TDVPConfig
>>> from tessera.quantum.holography import HolographyConfig, EmergentSpectralDimension
>>> cfg = HolographyConfig()
>>> cfg.tdvp.N = 10; cfg.tdvp.m = 0.5; cfg.tdvp.g = 1.0
>>> # ... fill in TDVP fields ...
>>> result = EmergentSpectralDimension(cfg).compute()
)doc");

    py::class_<HolographyConfig>(holo, "HolographyConfig",
            R"doc(Configuration for an emergent-spectral-dimension run.

Wraps a TDVPConfig and adds the σ-grid and mutual-information cutoff.
Validated at construction inside EmergentSpectralDimension; invalid
configs raise ValueError before any TDVP work is done.
)doc")
        .def(py::init<>())
        .def_readwrite("tdvp",              &HolographyConfig::tdvp)
        .def_readwrite("sigmaMin",          &HolographyConfig::sigmaMin)
        .def_readwrite("sigmaMax",          &HolographyConfig::sigmaMax)
        .def_readwrite("sigmaCount",        &HolographyConfig::sigmaCount)
        .def_readwrite("epsilonI",          &HolographyConfig::epsilonI)
        .def_readwrite("includeTemporal",   &HolographyConfig::includeTemporal)
        .def_readwrite("maxTemporalStride", &HolographyConfig::maxTemporalStride)
        .def_readwrite("krylovDim",         &HolographyConfig::krylovDim)
        .def_readwrite("seed",              &HolographyConfig::seed)
        .def_readwrite("vertexIds",         &HolographyConfig::vertexIds,
            "Optional spacetime-vertex labels for the site axis. "
            "Sourced from tessera.quantum.Causet.chainFrom(spacetime). "
            "Empty = use flat-site indices 0..N-1 as labels.")
        .def("validate",                    &HolographyConfig::validate);

    py::class_<MutualInformationProfile>(holo, "MutualInformationProfile",
            R"doc(Symmetric site×snapshot mutual-information matrix.

Built from a list of TDVPSnapshots that have ``mutualInformation``
recorded. Provides indexed access via at(site_v, snap_v, site_w, snap_w)
and a COO weighted-adjacency export.
)doc")
        .def(py::init<std::vector<TDVPSnapshot> const&,
                       HolographyConfig const&>(),
             py::arg("snapshots"), py::arg("config"))
        .def_property_readonly("nSites",     &MutualInformationProfile::nSites)
        .def_property_readonly("nSnapshots", &MutualInformationProfile::nSnapshots)
        .def_property_readonly("nLabels",    &MutualInformationProfile::nLabels)
        .def("at",     &MutualInformationProfile::at,
             py::arg("siteV"), py::arg("snapV"),
             py::arg("siteW"), py::arg("snapW"))
        .def("atFlat", &MutualInformationProfile::atFlat,
             py::arg("v"), py::arg("w"))
        .def("siteOf",     &MutualInformationProfile::siteOf,     py::arg("label"))
        .def("snapshotOf", &MutualInformationProfile::snapshotOf, py::arg("label"))
        .def("vertexId",   &MutualInformationProfile::vertexId,   py::arg("site"),
            "Spacetime-vertex ID for a flat-site index. Returns the "
            "flat index itself when the profile was built without "
            "CausetChain labels.")
        .def("weightedAdjacency",
            [](MutualInformationProfile const& p) {
                auto coo = p.weightedAdjacency();
                return py::make_tuple(coo.rows, coo.cols, coo.weights, coo.n);
            },
            R"doc(COO arrays (rows, cols, weights, nVertices) of edges with I > epsilonI.

Each undirected edge appears twice (v→w and w→v).
)doc");

    py::class_<EmergentGraph>(holo, "EmergentGraph",
            R"doc(Weighted graph (V_G, E_G, ℓ_G) on the (site × snapshot) label set.

Edge weights are mutual-information values; the Laplacian is L = D - W.
)doc")
        .def(py::init<MutualInformationProfile const&>(), py::arg("profile"))
        .def_property_readonly("nVertices", &EmergentGraph::nVertices)
        .def_property_readonly("nEdges",    &EmergentGraph::nEdges)
        .def("laplacianCOO",
            [](EmergentGraph const& g) {
                // Return the Laplacian as plain COO arrays (rows, cols,
                // values, n). pybind11/eigen.h's sparse-matrix binding
                // currently produces an empty CSC under LTO; the COO
                // path is small and stable.
                auto L = g.laplacian();
                std::vector<int>    rows;
                std::vector<int>    cols;
                std::vector<double> vals;
                rows.reserve(static_cast<std::size_t>(L.nonZeros()));
                cols.reserve(static_cast<std::size_t>(L.nonZeros()));
                vals.reserve(static_cast<std::size_t>(L.nonZeros()));
                for (int k = 0; k < L.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<double>::InnerIterator
                            it(L, k); it; ++it) {
                        rows.push_back(static_cast<int>(it.row()));
                        cols.push_back(static_cast<int>(it.col()));
                        vals.push_back(it.value());
                    }
                }
                return py::make_tuple(rows, cols, vals, g.nVertices());
            },
            R"doc(Weighted Laplacian as a COO tuple (rows, cols, values, n).

Wrap in scipy.sparse for downstream use::

    >>> import scipy.sparse as sp
    >>> rows, cols, vals, n = graph.laplacianCOO()
    >>> L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
)doc")
        .def("returnProbability", &EmergentGraph::returnProbability,
             py::arg("sigmas"), py::arg("krylovDim") = 30,
             py::arg("m") = 0, py::arg("seed") = 0,
             R"doc(P(σ) = (1/|V|) Tr exp(-σ L) via Krylov-Lanczos diagonal estimation.

``m`` is the Hutchinson-style subsample size. 0 (default) uses
``min(n, 3000)`` start vertices, trading a small variance penalty for a
large speedup at big n. Set ``m = n`` for the exact sum.
``seed`` controls the subset RNG for reproducibility.)doc")
        .def_static("spectralDimension", &EmergentGraph::spectralDimension,
             py::arg("sigmas"), py::arg("P"),
             R"doc(D_S(σ) = -2 d log P / d log σ via centered finite differences.)doc")
        .def_static("spectralDimensionSmoothed",
             &EmergentGraph::spectralDimensionSmoothed,
             py::arg("sigmas"), py::arg("P"),
             py::arg("windowSize") = 5, py::arg("polyOrder") = 2,
             R"doc(D_S(σ) via local-polynomial fit on (log σ, log P).

Savitzky-Golay-style smoothing: for each grid point, fit a
degree-`polyOrder` polynomial in (log σ, log P) over a centered window
of size `windowSize`, then read the slope at that point.
)doc")
        .def("toDot", &EmergentGraph::toDot,
             R"doc(Graphviz DOT export. Mirrors Poset.toDot().)doc")
        .def("toGraphML", &EmergentGraph::toGraphML,
             R"doc(GraphML export string; suitable for Gephi / yEd.

Mirrors `tessera.Spacetime.save("*.graphml")`. Edge weights are
exported under the `weight` attribute.
)doc")
        .def_static("fromWeightedEdges",
            [](int n, std::vector<std::tuple<int, int, double>> const& edges) {
                return EmergentGraph::fromWeightedEdges(n, edges);
            },
            py::arg("n"), py::arg("edges"),
            R"doc(Construct an EmergentGraph from a weighted edge list.

`edges` is a list of (u, v, weight) tuples; each undirected edge
should appear once.
)doc");

    py::class_<AmbjornLollFit::Result>(holo, "AmbjornLollFitResult")
        .def_readonly("dInfinity",  &AmbjornLollFit::Result::dInfinity)
        .def_readonly("C",          &AmbjornLollFit::Result::C)
        .def_readonly("B",          &AmbjornLollFit::Result::B)
        .def_readonly("chiSquared", &AmbjornLollFit::Result::chiSquared);

    py::class_<AmbjornLollFit>(holo, "AmbjornLollFit",
            R"doc(Static utility: three-parameter D_S(σ) = D_∞ − C / (B + σ) fit.

Stateless; not instantiable. Mirrors the form used by
examples/spectral_dimension.py for CDT comparisons.
)doc")
        .def_static("fit", &AmbjornLollFit::fit,
             py::arg("sigmas"), py::arg("dS"),
             py::arg("sigmaFitMin") = -1.0,
             py::arg("sigmaFitMax") = -1.0);

    py::class_<SpectralDimensionResult>(holo, "SpectralDimensionResult",
            R"doc(Result bundle from EmergentSpectralDimension.compute().

A plain data container; no MPS/MPO state crosses the boundary.
)doc")
        .def_readonly("sigmas",           &SpectralDimensionResult::sigmas)
        .def_readonly("P",                &SpectralDimensionResult::P)
        .def_readonly("dS",               &SpectralDimensionResult::dS)
        .def_readonly("dSSmoothed",       &SpectralDimensionResult::dSSmoothed)
        .def_readonly("dInfinity",        &SpectralDimensionResult::dInfinity)
        .def_readonly("C",                &SpectralDimensionResult::C)
        .def_readonly("B",                &SpectralDimensionResult::B)
        .def_readonly("fitChiSquared",    &SpectralDimensionResult::fitChiSquared)
        .def_readonly("graphNVertices",   &SpectralDimensionResult::graphNVertices)
        .def_readonly("graphNEdges",      &SpectralDimensionResult::graphNEdges)
        .def_readonly("snapshotTimes",    &SpectralDimensionResult::snapshotTimes)
        .def_readonly("snapshotBondDims", &SpectralDimensionResult::snapshotBondDims)
        .def_readonly("snapshotEnergies", &SpectralDimensionResult::snapshotEnergies)
        .def("toJson",
            [](SpectralDimensionResult const& r, HolographyConfig const& cfg) {
                return r.toJson(cfg);
            },
            py::arg("config"),
            R"doc(Single JSON record of the run.

Includes the bound config, the TDVP summary, the graph diagnostics,
both raw and smoothed D_S(σ), the Ambjorn-Loll fit, and a provenance
block. Suitable for archiving alongside the experiment results.
)doc");

    // ─── ChoiPropagator (temporal MI engine) ──────────────────────────
    //
    // Exposed for unit-testing the identity-channel and single-qubit-unitary
    // checks. The full pipeline reaches the same code via
    // MutualInformationProfile / HolographyConfig.includeTemporal.
    py::class_<ChoiPropagator::TDVPSettings>(holo, "ChoiTDVPSettings",
            R"doc(Sweep settings for the Choi-state TDVP evolution.)doc")
        .def(py::init<>())
        .def_readwrite("dt",         &ChoiPropagator::TDVPSettings::dt)
        .def_readwrite("maxBondDim", &ChoiPropagator::TDVPSettings::maxBondDim)
        .def_readwrite("krylovDim",  &ChoiPropagator::TDVPSettings::krylovDim)
        .def_readwrite("cutoff",     &ChoiPropagator::TDVPSettings::cutoff)
        .def_readwrite("quiet",      &ChoiPropagator::TDVPSettings::quiet);

    py::class_<ChoiPropagator>(holo, "ChoiPropagator",
            R"doc(Static utility for the Choi-state temporal MI.

For a unitary U on N qubits, the Choi state |U⟩ = (U ⊗ I)|Φ+⟩^{⊗N}
encodes the temporal mutual information between (site i at the input
time) and (site j at the output time) as the 2-site reduced-density-
matrix MI on (in_i, out_j). This class exposes the C++ helpers that
build the Choi state under the Schwinger Hamiltonian (acting only on
the output register of an interleaved doubled chain) and extract the
N×N temporal MI matrix.

Not instantiable; call methods on the class.
)doc")
        .def_static("temporalMutualInformation",
            [](SchwingerParams const& p, double duration,
                ChoiPropagator::TDVPSettings const& settings) {
                auto choi = ChoiPropagator::choiState(p, duration, settings);
                return ChoiPropagator::temporalMutualInformation(choi, p.N);
            },
            py::arg("params"), py::arg("duration"), py::arg("settings"),
            R"doc(Temporal MI matrix for the Schwinger propagator over `duration`.

Returns an N×N numpy array; entry (i-1, j-1) is I({in_i} : {out_j})
in nats. At duration = 0 the Choi state is |Φ+⟩^{⊗N} and the matrix
equals 2·ln(2) on the diagonal, zero elsewhere (the identity channel).
)doc");

    py::class_<SchwingerParams>(holo, "SchwingerParams",
            R"doc(Bare dimensional parameters of the Schwinger Hamiltonian.

Exposed here so the holography test suite can drive ChoiPropagator
directly without going through TDVPConfig.
)doc")
        .def(py::init<>())
        .def_readwrite("N",  &SchwingerParams::N)
        .def_readwrite("a",  &SchwingerParams::a)
        .def_readwrite("m",  &SchwingerParams::m)
        .def_readwrite("g",  &SchwingerParams::g)
        .def_readwrite("L0", &SchwingerParams::L0);

    py::class_<EmergentSpectralDimension>(holo, "EmergentSpectralDimension",
            R"doc(Workflow class: bind a HolographyConfig, run the full pipeline.

DMRG ground state → q-qbar quench → TDVP loop with per-snapshot MI →
weighted (site, time) graph → heat-kernel trace → D_S(σ) →
Ambjorn-Loll fit. Mirrors the SchwingerModel(cfg).solve() and
SchwingerQuench(cfg).evolve() patterns in tessera.quantum.

The constructor forces recordMutualInformation on in the underlying
TDVPConfig.
)doc")
        .def(py::init<HolographyConfig>(), py::arg("config"))
        .def_property_readonly("config",
            [](EmergentSpectralDimension const& m) { return m.config(); })
        .def("compute", &EmergentSpectralDimension::compute,
             R"doc(Run the full pipeline; returns SpectralDimensionResult.)doc")
        .def("computeFromSnapshots",
             &EmergentSpectralDimension::computeFromSnapshots,
             py::arg("quench"),
             R"doc(Reuse a single TDVP run across multiple σ-grids or ε_I values.)doc");

    // ─── Koashi-Imoto + QuantumSimplex ─────────────────────────────────
    m.def("partialTraceA", &::tessera::quantum::partialTraceA,
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          R"doc(Partial trace over the A factor of a bipartite ρ_AB.

rhoAB is a (dimA * dimB) x (dimA * dimB) matrix in (A ⊗ B) ordering
(row index = a * dimB + b). Returns a dimB x dimB density matrix.)doc");

    m.def("partialTraceB", &::tessera::quantum::partialTraceB,
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          R"doc(Partial trace over the B factor of a bipartite ρ_AB.

Returns a dimA x dimA density matrix.)doc");

    m.def("mutualInformation",
          py::overload_cast<const Eigen::MatrixXcd&, int, int>(
              &::tessera::quantum::mutualInformation),
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          R"doc(Mutual information I(A:B) = S(A) + S(B) - S(AB) in nats.

Floors at 0. Marginals are computed by partial trace from ρ_AB.)doc");

    m.def("mutualInformation",
          py::overload_cast<const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&>(
              &::tessera::quantum::mutualInformation),
          py::arg("rhoAB"), py::arg("rhoA"), py::arg("rhoB"),
          R"doc(Mutual information I(A:B) = S(A) + S(B) - S(AB) with
explicit marginals. Use this when ρ_A and ρ_B are already on hand
(e.g. on QuantumVertex objects) — it skips the partial-trace step.)doc");

    py::class_<::tessera::quantum::KoashiImotoTolerances>(m,
            "KoashiImotoTolerances",
            R"doc(Numerical tolerances for the symmetric KI decomposition.

Each defaults to 1e-10. Tightening or relaxing affects the block /
cond-state clustering and so the resolved L/R structure.)doc")
        .def(py::init<>())
        .def(py::init<double, double, double>(),
             py::arg("epsKiEigen"), py::arg("epsKiCondState"),
             py::arg("epsKiSvd"))
        .def_property("epsKiEigen",
                      &::tessera::quantum::KoashiImotoTolerances::getEpsKiEigen,
                      &::tessera::quantum::KoashiImotoTolerances::setEpsKiEigen)
        .def_property("epsKiCondState",
                      &::tessera::quantum::KoashiImotoTolerances::getEpsKiCondState,
                      &::tessera::quantum::KoashiImotoTolerances::setEpsKiCondState)
        .def_property("epsKiSvd",
                      &::tessera::quantum::KoashiImotoTolerances::getEpsKiSvd,
                      &::tessera::quantum::KoashiImotoTolerances::setEpsKiSvd);

    py::class_<::tessera::quantum::KoashiImotoBlock>(m, "KoashiImotoBlock",
            R"doc(A single j-block of the symmetric KI decomposition.

Outputs of ``koashiImotoDecompose``; immutable.)doc")
        .def_property_readonly("weight",
                      &::tessera::quantum::KoashiImotoBlock::getWeight)
        .def_property_readonly("coreState",
                      &::tessera::quantum::KoashiImotoBlock::getCoreState)
        .def_property_readonly("tailA",
                      &::tessera::quantum::KoashiImotoBlock::getTailA)
        .def_property_readonly("tailB",
                      &::tessera::quantum::KoashiImotoBlock::getTailB)
        .def_property_readonly("dimLeftA",
                      &::tessera::quantum::KoashiImotoBlock::getDimLeftA)
        .def_property_readonly("dimLeftB",
                      &::tessera::quantum::KoashiImotoBlock::getDimLeftB)
        .def_property_readonly("dimRightA",
                      &::tessera::quantum::KoashiImotoBlock::getDimRightA)
        .def_property_readonly("dimRightB",
                      &::tessera::quantum::KoashiImotoBlock::getDimRightB);

    py::class_<::tessera::quantum::KoashiImotoResult>(m,
            "KoashiImotoResult",
            R"doc(Result of the symmetric Koashi-Imoto decomposition.

Holds the three child matrices (sigma = the joint core; aPrime, bPrime
= the uncorrelated tails) and the per-block breakdown.)doc")
        .def_property_readonly("sigma",
                      &::tessera::quantum::KoashiImotoResult::getSigma)
        .def_property_readonly("aPrime",
                      &::tessera::quantum::KoashiImotoResult::getAPrime)
        .def_property_readonly("bPrime",
                      &::tessera::quantum::KoashiImotoResult::getBPrime)
        .def_property_readonly("blocks",
                      &::tessera::quantum::KoashiImotoResult::getBlocks);

    m.def("koashiImotoDecompose",
          py::overload_cast<const Eigen::MatrixXcd&, int, int,
                            const ::tessera::quantum::KoashiImotoTolerances&>(
              &::tessera::quantum::koashiImotoDecompose),
          py::arg("rhoAB"), py::arg("dimA"), py::arg("dimB"),
          py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
          R"doc(Symmetric Koashi-Imoto decomposition of a bipartite ρ_AB.

Returns a KoashiImotoResult with the three child matrices and the
per-block breakdown. Marginals are extracted from ρ_AB by partial
trace.)doc");

    m.def("koashiImotoDecompose",
          py::overload_cast<const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&,
                            const Eigen::MatrixXcd&,
                            const ::tessera::quantum::KoashiImotoTolerances&>(
              &::tessera::quantum::koashiImotoDecompose),
          py::arg("rhoAB"), py::arg("rhoA"), py::arg("rhoB"),
          py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
          R"doc(Symmetric Koashi-Imoto decomposition with explicit
marginals ρ_A and ρ_B (preferred when the marginals are already on
hand — avoids the partial-trace step's numerical drift).)doc");

    m.def("createQuantumVertex",
          [](::tessera::spacetime::Spacetime& st,
             Eigen::MatrixXcd                 state) {
              const auto id = st.reserveVertexId();
              auto vlist = st.getVertexList();
              return vlist->template addAs<::tessera::quantum::QuantumVertex>(
                  id, id, std::move(state));
          },
          py::arg("spacetime"), py::arg("state"),
          py::return_value_policy::reference,
          R"doc(Allocate a new QuantumVertex in the spacetime's vertex
list, carrying the given density matrix. The vertex id is assigned
from the spacetime's counter. The returned pointer is owned by the
spacetime.)doc");

    py::class_<::tessera::quantum::QuantumVertex,
               ::tessera::mesh::Vertex,
               std::unique_ptr<::tessera::quantum::QuantumVertex,
                               py::nodelete>>(m, "QuantumVertex",
            R"doc(A mesh.Vertex carrying a density matrix.

QuantumVertex extends mesh.Vertex with an Eigen-typed density
matrix in its local Hilbert space. The matrix dimension is fixed
at construction time and may differ across QuantumVertex objects
in the same VertexList (the KI factories use this to give A, B, Σ,
A', B' their own per-block dimensions).

Construct via ``createQuantumVertex(spacetime, state)`` (which
allocates one inside the spacetime's vertex list) or directly via
``QuantumVertex(id, state)`` for free-standing use.)doc")
        .def(py::init<std::uint64_t, Eigen::MatrixXcd>(),
             py::arg("id"), py::arg("state"))
        .def("getState",
             &::tessera::quantum::QuantumVertex::getState,
             py::return_value_policy::reference_internal,
             R"doc(Return the density matrix ρ on this vertex.)doc")
        .def("setState",
             &::tessera::quantum::QuantumVertex::setState,
             py::arg("state"),
             R"doc(Replace the density matrix.)doc")
        .def("stateDim",
             &::tessera::quantum::QuantumVertex::stateDim,
             R"doc(Return the Hilbert-space dimension of ρ.)doc")
        ;

    py::class_<::tessera::quantum::QuantumSimplex>(m, "QuantumSimplex",
            R"doc(Static-only utility: KI factories for a 5-vertex
KI-interaction cell.

QuantumSimplex is not a separate runtime type — it is a namespace
for the four KI factory entry points that build a regular
mesh.Simplex (five vertices, ten edges) inside a Spacetime from
two pre-existing QuantumVertex inputs. The returned mesh.Simplex
is owned by the Spacetime; the per-vertex ρ lives on the
QuantumVertex objects in the vertex list; the per-edge d_VR is
stored as the edge length.

iMax is global to the simulation and is passed to each factory
call — it is not stored on the simplex.)doc")
        .def_static("fromSchmidtPurification",
            &::tessera::quantum::QuantumSimplex::fromSchmidtPurification,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Build ρ_AB = |ψ⟩⟨ψ| with
|ψ⟩ = Σ_i √λ_i |a_i⟩|b_i⟩ from matched marginal spectra of
ρ_A on qva and ρ_B on qvb, then construct the cell.)doc")
        .def_static("fromClassicalCorrelation",
            &::tessera::quantum::QuantumSimplex::fromClassicalCorrelation,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Build the perfectly-correlated classical joint
ρ_AB = Σ_i λ_i |a_i⟩⟨a_i| ⊗ |b_i⟩⟨b_i| in matched eigenbases.)doc")
        .def_static("fromExplicitJoint",
            &::tessera::quantum::QuantumSimplex::fromExplicitJoint,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("rhoAB"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Build the cell from a caller-supplied joint
ρ_AB. The partial traces of ρ_AB must agree with the marginals
on qva, qvb.)doc")
        .def_static("fromTargetMutualInformation",
            &::tessera::quantum::QuantumSimplex::fromTargetMutualInformation,
            py::arg("spacetime"),
            py::arg("qva"),
            py::arg("qvb"),
            py::arg("targetMI"),
            py::arg("iMax"),
            py::arg("tol") = ::tessera::quantum::KoashiImotoTolerances{},
            py::return_value_policy::reference,
            R"doc(Binary-search α in
ρ_AB(α) = (1-α)·(ρ_A ⊗ ρ_B) + α·ρ_AB^Schmidt
to hit ``targetMI``. Requires matched spectra of ρ_A, ρ_B; throws
if ``targetMI`` lies outside [0, 2·H(λ)].)doc");

    py::enum_<::tessera::quantum::QuantumSimplex::Position>(
            m, "QuantumSimplexPosition")
        .value("A",      ::tessera::quantum::QuantumSimplex::A)
        .value("B",      ::tessera::quantum::QuantumSimplex::B)
        .value("Sigma",  ::tessera::quantum::QuantumSimplex::Sigma)
        .value("APrime", ::tessera::quantum::QuantumSimplex::APrime)
        .value("BPrime", ::tessera::quantum::QuantumSimplex::BPrime);

    // ── Exterior-algebra / graded-tensor primitives ────────────────────
    // Sparse operators cross the boundary as COO tuples (see sparseOpToCoo).

    py::class_<OccupationBitset>(m, "OccupationBitset",
        R"doc(An exterior basis state of Λ*C^M as a chunked occupation bitset.

One 64-bit word up to the machine-word threshold, ceil(M/64) words above
it, with the exact prefix-popcount creation sign
(-1)^popcount(b & ((1 << i) - 1)) in both regimes. The mode order behind
bit positions is a compilation artifact of the order-independent abstract
exterior algebra (no Kasteleyn orientation is required); permuted /
permutationParity give the exact signed action of a mode relabeling.)doc")
        .def(py::init<std::size_t>(), py::arg("modeCount"))
        .def_static("fromOccupiedModes", &OccupationBitset::fromOccupiedModes,
             py::arg("modeCount"), py::arg("modes"),
             "Bitset with exactly the listed modes occupied.")
        .def_static("fromIndex", &OccupationBitset::fromIndex,
             py::arg("modeCount"), py::arg("index"),
             "Bitset from a Fock basis index (modeCount <= 64).")
        .def("modeCount", &OccupationBitset::modeCount)
        .def("chunkCount", &OccupationBitset::chunkCount,
             "Number of 64-bit storage chunks, ceil(M/64).")
        .def("chunks", &OccupationBitset::chunks,
             "Raw storage words, least-significant chunk first.")
        .def("test", &OccupationBitset::test, py::arg("mode"))
        .def("set", &OccupationBitset::set, py::arg("mode"))
        .def("reset", &OccupationBitset::reset, py::arg("mode"))
        .def("count", &OccupationBitset::count,
             "Total occupation number N.")
        .def("parity", &OccupationBitset::parity,
             "Fermion parity (-1)^N as +1/-1.")
        .def("prefixPopcount", &OccupationBitset::prefixPopcount,
             py::arg("mode"),
             "Number of occupied modes strictly below `mode`.")
        .def("applyCreation", &OccupationBitset::applyCreation, py::arg("mode"),
             R"doc(Apply a_i^dagger at the bit level: 0 if the mode was occupied
(Pauli exclusion, state unchanged), else the exact sign
(-1)^prefixPopcount(mode) with the mode now occupied.)doc")
        .def("applyAnnihilation", &OccupationBitset::applyAnnihilation,
             py::arg("mode"),
             "Apply a_i at the bit level: 0 if empty, else the exact sign.")
        .def("toIndex", &OccupationBitset::toIndex,
             "Fock basis index sum_i b_i 2^i (modeCount <= 64).")
        .def("occupiedModes", &OccupationBitset::occupiedModes,
             "Occupied modes in ascending order (the wedge word).")
        .def("permuted", &OccupationBitset::permuted, py::arg("perm"),
             "The relabeled bitset: bit perm[i] of the result = bit i.")
        .def("permutationParity", &OccupationBitset::permutationParity,
             py::arg("perm"),
             R"doc(The exact +1/-1 a basis state picks up under the mode relabeling
`perm`: the inversion parity of the images of its occupied modes. Physical
amplitudes are invariant under relabeling once this parity is applied.)doc")
        .def("__eq__",
             [](const OccupationBitset& a, const OccupationBitset& b) {
                 return a == b;
             }, py::is_operator())
        .def("__repr__", &OccupationBitset::str)
        .def("__str__", &OccupationBitset::str);

    py::class_<ExteriorAlgebra>(m, "ExteriorAlgebra",
        R"doc(The exterior algebra Λ*C^M with its CAR operator layer.

Fock basis |b> at index n(b) = sum_i b_i 2^i (mode 0 = least-significant
bit). Exact identities carried (tested to double round-off): the CAR
{a_i, a_j} = {a_i+, a_j+} = 0, {a_i, a_j+} = delta_ij; dim Λ*C^M = 2^M;
||v_1 ^ ... ^ v_n||^2 = det(<v_i, v_j>); duplicate complete one-particle
modes wedge to exactly zero. Sparse operators are returned as COO tuples
(rows, cols, values, n) — wrap with scipy.sparse.csr_matrix.)doc")
        .def(py::init<std::size_t>(), py::arg("modeCount"))
        .def("modeCount", &ExteriorAlgebra::modeCount)
        .def("fockDimension", &ExteriorAlgebra::fockDimension,
             "dim Λ*C^M = 2^M.")
        .def("creationMatrixCOO",
             [](const ExteriorAlgebra& a, std::size_t mode) {
                 return sparseOpToCoo(a.creationMatrix(mode));
             }, py::arg("mode"),
             "a_i^dagger with the prefix-popcount sign rule, as COO.")
        .def("annihilationMatrixCOO",
             [](const ExteriorAlgebra& a, std::size_t mode) {
                 return sparseOpToCoo(a.annihilationMatrix(mode));
             }, py::arg("mode"), "a_i (adjoint of creationMatrixCOO), as COO.")
        .def("numberMatrixCOO",
             [](const ExteriorAlgebra& a, std::size_t mode) {
                 return sparseOpToCoo(a.numberMatrix(mode));
             }, py::arg("mode"), "n_i = a_i^dagger a_i, as COO.")
        .def("totalNumberMatrixCOO",
             [](const ExteriorAlgebra& a) {
                 return sparseOpToCoo(a.totalNumberMatrix());
             }, "N = sum_i n_i, as COO.")
        .def("parityMatrixCOO",
             [](const ExteriorAlgebra& a) {
                 return sparseOpToCoo(a.parityMatrix());
             }, "Fermion parity (-1)^N, as COO.")
        .def("sectorProjectorCOO",
             [](const ExteriorAlgebra& a, std::size_t occupation) {
                 return sparseOpToCoo(a.sectorProjector(occupation));
             }, py::arg("occupation"),
             "Projector onto total occupation N = `occupation`, as COO.")
        .def("subsetSectorProjectorCOO",
             [](const ExteriorAlgebra& a, const std::vector<std::size_t>& modes,
                std::size_t occupation) {
                 return sparseOpToCoo(a.subsetSectorProjector(modes, occupation));
             }, py::arg("modes"), py::arg("occupation"),
             R"doc(Projector onto occupation `occupation` restricted to the mode
subset `modes`, as COO. With a three-mode subset these are the exact
Lambda^0, Lambda^1, Lambda^2, Lambda^3 occupation-number projectors of
that factor.)doc")
        .def("vacuumState", &ExteriorAlgebra::vacuumState,
             "The vacuum |0...0> as a dense Fock vector.")
        .def("basisState", &ExteriorAlgebra::basisState, py::arg("bitset"),
             "The Fock basis vector |b> of an occupation bitset.")
        .def("creationOperatorCOO",
             [](const ExteriorAlgebra& a, const Eigen::VectorXcd& v) {
                 return sparseOpToCoo(a.creationOperator(v));
             }, py::arg("v"),
             "Smeared creation a^dagger(v) = sum_i v_i a_i^dagger, as COO.")
        .def("annihilationOperatorCOO",
             [](const ExteriorAlgebra& a, const Eigen::VectorXcd& v) {
                 return sparseOpToCoo(a.annihilationOperator(v));
             }, py::arg("v"),
             R"doc(Smeared annihilation a(v) = sum_i conj(v_i) a_i (antilinear in v),
so {a(v), a^dagger(w)} = <v, w> exactly, as COO.)doc")
        .def("wedge", &ExteriorAlgebra::wedge, py::arg("vectors"),
             R"doc(v_1 ^ ... ^ v_n = a^dagger(v_1)...a^dagger(v_n)|vacuum> as a dense
Fock vector (rightmost factor applied first).
||v_1 ^ ... ^ v_n||^2 = det(<v_i, v_j>) exactly; a repeated complete
one-particle mode gives exactly zero.)doc")
        .def("contract", &ExteriorAlgebra::contract, py::arg("w"),
             py::arg("state"),
             "Interior product iota_w state = a(w) state (odd antiderivation).")
        .def("dGammaCOO",
             [](const ExteriorAlgebra& a, const Eigen::MatrixXcd& L) {
                 return sparseOpToCoo(a.dGamma(L));
             }, py::arg("oneParticle"),
             R"doc(Second quantization dGamma(L) = sum_ij L_ij a_i^dagger a_j of an
MxM one-particle block matrix — the number-preserving quadratic/hopping
operator. For Hermitian L the spectrum is exactly the occupation subset
sums of the one-particle spectrum. Returned as COO.)doc")
        .def("modePermutationMatrixCOO",
             [](const ExteriorAlgebra& a, const std::vector<std::size_t>& perm) {
                 return sparseOpToCoo(a.modePermutationMatrix(perm));
             }, py::arg("perm"),
             R"doc(The signed permutation unitary U_pi with
U_pi |b> = permutationParity(b) |pi(b)>, so
U_pi a_i^dagger U_pi^dagger = a_pi(i)^dagger. Returned as COO.)doc");

    py::class_<GradedTensorComplex>(m, "GradedTensorComplex",
        R"doc(The graded tensor product of two finite chain complexes.

C_n = sum_{p+q=n} A_p x B_q with the graded Leibniz differential
d(a x b) = da x b + (-1)^deg(a) a x db — the chain complex of an actual
product cell complex (cubical/CW products satisfy C(XxY) = C(X) x C(Y)
on the nose). Exact consequences of the sign rule: d o d = 0, and the
Hodge Laplacian is blockwise Delta_A x 1 + 1 x Delta_B, so the degree-n
Hodge spectrum is the multiset of pairwise sums (Kunneth at the Hodge
level, identity metrics).

Conventions: diff[k] is the boundary C_{k+1} -> C_k of shape
dims[k] x dims[k+1]; product blocks are ordered by ascending p; within a
block the index is i_a * dimB_q + i_b (kron(A-side, B-side)).)doc")
        .def(py::init<std::vector<std::size_t>, std::vector<Eigen::MatrixXcd>,
                      std::vector<std::size_t>, std::vector<Eigen::MatrixXcd>,
                      double>(),
             py::arg("dimsA"), py::arg("diffA"), py::arg("dimsB"),
             py::arg("diffB"), py::arg("boundaryTolerance") = 0.0)
        .def("maxDegree", &GradedTensorComplex::maxDegree)
        .def("chainDimension", &GradedTensorComplex::chainDimension,
             py::arg("degree"))
        .def("blocks", &GradedTensorComplex::blocks, py::arg("degree"),
             "The (p, q) block labels of C_degree in storage order.")
        .def("differential", &GradedTensorComplex::differential,
             py::arg("degree"),
             "The graded Leibniz differential C_degree -> C_{degree-1}.")
        .def("laplacian", &GradedTensorComplex::laplacian, py::arg("degree"),
             "Hodge Laplacian d+ d + d d+ of the product at `degree`.")
        .def("factorLaplacianA", &GradedTensorComplex::factorLaplacianA,
             py::arg("degree"), "Hodge Laplacian of factor A alone.")
        .def("factorLaplacianB", &GradedTensorComplex::factorLaplacianB,
             py::arg("degree"), "Hodge Laplacian of factor B alone.");

    py::class_<FockDirectSum>(m, "FockDirectSum",
        R"doc(The Fock direct-sum functor F(h_A + h_B) = F(h_A) x F(h_B).

Compiled with A modes first: |b> <-> |b_A> x |b_B> at joint index
i_A + 2^M_A i_B, sign-free. Even operators lift as X x 1 and 1 x Y; odd
right-factor operators acquire the parity twist (-1)^N_A x Y (the Koszul
sign, i.e. the Jordan-Wigner string over A), making the joint CAR
generators exactly the lifted factor generators — direct sums become
graded tensor products, and the two directed coupling blocks C_AB and C_BA
of a one-particle operator become hopping terms, with no Hermitian-conjugate
relation between them assumed. The
graded swap S(x b y) = (-1)^{|x||y|} y x x has odd/odd sign -1 and +1 on
every other elementary parity combination. Operator arguments are dense;
sparse results are COO tuples.)doc")
        .def(py::init<std::size_t, std::size_t>(), py::arg("modesA"),
             py::arg("modesB"))
        .def("modesA", &FockDirectSum::modesA)
        .def("modesB", &FockDirectSum::modesB)
        .def("jointAlgebra", &FockDirectSum::jointAlgebra,
             "The joint ExteriorAlgebra over M_A + M_B modes.")
        .def("leftAlgebra", &FockDirectSum::leftAlgebra)
        .def("rightAlgebra", &FockDirectSum::rightAlgebra)
        .def("liftLeftCOO",
             [](const FockDirectSum& f, const Eigen::MatrixXcd& opA) {
                 return sparseOpToCoo(f.liftLeft(opA.sparseView()));
             }, py::arg("opA"),
             R"doc(X_A -> X_A x 1_FB, exact for odd and even operators (A modes
precede all B modes, so the Jordan-Wigner string is empty). COO result.)doc")
        .def("liftRightCOO",
             [](const FockDirectSum& f, const Eigen::MatrixXcd& opB,
                bool oddOperator) {
                 return sparseOpToCoo(f.liftRight(opB.sparseView(),
                                                  oddOperator));
             }, py::arg("opB"), py::arg("oddOperator"),
             R"doc(Y_B -> 1 x Y_B (even) or (-1)^N_A x Y_B (odd; the Koszul /
Jordan-Wigner parity twist that preserves the joint CAR). COO result.)doc")
        .def("gradedSwapMatrixCOO",
             [](const FockDirectSum& f) {
                 return sparseOpToCoo(f.gradedSwapMatrix());
             },
             R"doc(The graded swap S: F_A x F_B -> F_B x F_A,
S(x b y) = (-1)^{|x||y|} y x x: odd/odd exchange is -1, every other
elementary parity combination is +1. COO result.)doc")
        .def_static("assembleBlockOneParticle",
             py::overload_cast<const Eigen::MatrixXcd&, const Eigen::MatrixXcd&,
                               const Eigen::MatrixXcd&, const Eigen::MatrixXcd&>(
                 &FockDirectSum::assembleBlockOneParticle),
             py::arg("blockA"), py::arg("blockB"), py::arg("couplingAB"),
             py::arg("couplingBA"),
             R"doc(The block one-particle matrix [[L_A, C_AB], [C_BA, L_B]] (dense)
from both directed couplings: C_AB (M_A x M_B, hopping B -> A) and C_BA
(M_B x M_A, hopping A -> B), each placed as given. No Hermitian-conjugate
relation between them is assumed.)doc")
        .def_static("assembleBlockOneParticle",
             py::overload_cast<const Eigen::MatrixXcd&, const Eigen::MatrixXcd&,
                               const Eigen::MatrixXcd&>(
                 &FockDirectSum::assembleBlockOneParticle),
             py::arg("blockA"), py::arg("blockB"), py::arg("coupling"),
             "The *-structure special case [[L_A, C], [C+, L_B]]: the "
             "directed form with C_BA = C^dagger certified by the caller.")
        .def("dGammaBlockCOO",
             [](const FockDirectSum& f, const Eigen::MatrixXcd& blockA,
                const Eigen::MatrixXcd& blockB,
                const Eigen::MatrixXcd& couplingAB,
                const Eigen::MatrixXcd& couplingBA) {
                 return sparseOpToCoo(
                     f.dGammaBlock(blockA, blockB, couplingAB, couplingBA));
             }, py::arg("blockA"), py::arg("blockB"), py::arg("couplingAB"),
             py::arg("couplingBA"),
             R"doc(dGamma([[L_A, C_AB], [C_BA, L_B]]) on the joint Fock space:
liftLeft(dGamma(L_A)) + liftRight(dGamma(L_B), even) plus the directed
hopping terms sum_{i in A, j in B} (C_AB)_ij a_i^dagger a_j +
sum_{i in B, j in A} (C_BA)_ij a_i^dagger a_j. No relation between the two
blocks is assumed. COO result.)doc")
        .def("dGammaBlockCOO",
             [](const FockDirectSum& f, const Eigen::MatrixXcd& blockA,
                const Eigen::MatrixXcd& blockB,
                const Eigen::MatrixXcd& coupling) {
                 return sparseOpToCoo(f.dGammaBlock(blockA, blockB, coupling));
             }, py::arg("blockA"), py::arg("blockB"), py::arg("coupling"),
             R"doc(dGamma([[L_A, C], [C+, L_B]]): the *-structure special case of
the directed form (C_BA = C^dagger, certified by the caller). COO result.)doc");

    py::class_<EdgeModeRecord>(m, "EdgeModeRecord",
        R"doc(One edge-mode record: oriented incidence + mode identity.

The edge indexes one two-level mode factor span{|0>, |1>} (modeId) inside
the global exterior Fock space. No per-edge state vector is stored, and
the Edge's single complex length stays on the Edge; a per-edge occupation
is a derived marginal of the global state, not a stored product state.)doc")
        .def_readonly("vertexA", &EdgeModeRecord::vertexA)
        .def_readonly("vertexB", &EdgeModeRecord::vertexB)
        .def_readonly("orientationSign", &EdgeModeRecord::orientationSign)
        .def_readonly("modeId", &EdgeModeRecord::modeId)
        .def_readonly("lineageKey", &EdgeModeRecord::lineageKey);

    py::class_<EdgeModeRegistry>(m, "EdgeModeRegistry",
        R"doc(Edge-mode basis bookkeeping + the deterministic compilation order.

canonicalModeOrder sorts modes by (lineageKey, min vertex, max vertex):
oriented component lineage first, the unordered vertex pair as the
deterministic tie-break. The order is a compilation artifact of the
order-independent abstract exterior algebra — no Kasteleyn orientation is
required. A vertex relabeling rebuilds the order; orderPermutation +
OccupationBitset.permutationParity / ExteriorAlgebra.modePermutationMatrixCOO
give the exact parity map under which all physical amplitudes are
invariant.

Reorientation convention: reverseStoredDirection swaps endpoints and flips
orientationSign (a pure storage change — nothing observable moves);
flipOrientation flips only the sign (physical reversal — the mode's
one-particle embedding vector is multiplied by -1, i.e. a_e -> -a_e,
a_e+ -> -a_e+, with the two-level factor fixed pointwise and nothing
conjugated; CAR and occupation observables are preserved).)doc")
        .def(py::init<>())
        .def_static("fromSpacetime", &EdgeModeRegistry::fromSpacetime,
             py::arg("spacetime"), py::arg("lineageKey") = "K1",
             R"doc(Register one two-level mode per edge of the spacetime.

The carrier is F(h_K) with h_K = span{|e> : e in K1}; any per-band carrier is
a derived view of these per-edge modes. Each edge is registered on its stored
source -> target direction with orientationSign +1, so canonicalOrientationSign
reports how that orientation sits against the canonical min -> max direction.
Every mode gets lineageKey, reducing the canonical order to the deterministic
endpoint sort.

Reads incidence only -- never a length, never a connection phase.

Raises:
    ValueError: on a self-loop or a duplicated unordered vertex pair.)doc")
        .def("addEdge", &EdgeModeRegistry::addEdge, py::arg("vertexA"),
             py::arg("vertexB"), py::arg("orientationSign"),
             py::arg("lineageKey"),
             "Register an edge mode; returns the assigned modeId.")
        .def("modeCount", &EdgeModeRegistry::modeCount)
        .def("record", &EdgeModeRegistry::record, py::arg("modeId"),
             py::return_value_policy::copy)
        .def("records", &EdgeModeRegistry::records,
             py::return_value_policy::copy)
        .def("reverseStoredDirection", &EdgeModeRegistry::reverseStoredDirection,
             py::arg("modeId"),
             "(a, b, s) -> (b, a, -s): storage convention only; invariant.")
        .def("flipOrientation", &EdgeModeRegistry::flipOrientation,
             py::arg("modeId"),
             "s -> -s: physically reverses the oriented edge.")
        .def("canonicalOrientationSign",
             &EdgeModeRegistry::canonicalOrientationSign, py::arg("modeId"),
             R"doc(Orientation relative to the canonical (min -> max vertex)
direction; invariant under reverseStoredDirection, flips under
flipOrientation.)doc")
        .def("canonicalModeOrder", &EdgeModeRegistry::canonicalModeOrder,
             "modeIds sorted by (lineageKey, min vertex, max vertex).")
        .def("compilationPositions", &EdgeModeRegistry::compilationPositions,
             "positions[modeId] = index in canonicalModeOrder().")
        .def("relabeled", &EdgeModeRegistry::relabeled, py::arg("vertexMap"),
             R"doc(The registry after a vertex relabeling (dict old -> new; must
cover all used vertices, injectively). modeIds, signs and lineage keys are
preserved; the canonical order is rebuilt from the new ids.)doc")
        .def_static("orderPermutation", &EdgeModeRegistry::orderPermutation,
             py::arg("before"), py::arg("after"),
             R"doc(perm[i] = position in `after`'s canonical order of the mode at
position i of `before`'s canonical order (matched by modeId). Feed to
OccupationBitset.permutationParity / ExteriorAlgebra.
modePermutationMatrixCOO for the exact parity map.)doc");

    // ── Lazy graded Fock oracle and boundary carrier ──────────────────
    // Dense Eigen crossings only; the LocalMap COO route uses plain
    // (rows, cols, values) arrays per the repository convention.

    py::enum_<LazyNodeKind>(m, "LazyNodeKind",
        "Node vocabulary of the lazy Fock expression DAG.")
        .value("Vacuum", LazyNodeKind::Vacuum)
        .value("Occupation", LazyNodeKind::Occupation)
        .value("GradedTensor", LazyNodeKind::GradedTensor)
        .value("LocalMap", LazyNodeKind::LocalMap)
        .value("SectorSum", LazyNodeKind::SectorSum)
        .value("Wedge", LazyNodeKind::Wedge);

    py::enum_<LazySectorKind>(m, "LazySectorKind",
        "Conserved functional labeling a sector direct sum.")
        .value("Occupation", LazySectorKind::Occupation)
        .value("Parity", LazySectorKind::Parity);

    py::class_<LazyFockState>(m, "LazyFockState",
        R"doc(A value handle on one lazy-Fock expression-DAG root.

Carries the accumulated discarded-norm bound D (exactly 0.0 in exact
certification mode; in truncation mode an upper bound on the l2 error of
the represented state, reported in every scalar read) and the optional
boundary-fixture label, set only by boundaryProductFixture. Per-edge
occupations are derived marginals of the global state; the handle never
stores per-mode state vectors.)doc")
        .def("valid", &LazyFockState::valid)
        .def("rootNodeId", &LazyFockState::rootNodeId,
             "Process-unique root node id (sharing / negative-control "
             "tests).")
        .def("contentHash", &LazyFockState::contentHash,
             "Replay-stable content hash of the root subexpression.")
        .def("kind", &LazyFockState::kind, "Root node kind.")
        .def("modes", &LazyFockState::modes,
             "Sorted global mode indices the state covers.")
        .def("childNodeIds", &LazyFockState::childNodeIds)
        .def("childContentHashes", &LazyFockState::childContentHashes)
        .def("nodeCount", &LazyFockState::nodeCount,
             "Distinct DAG nodes (a shared subexpression counts once).")
        .def("discardedNorm", &LazyFockState::discardedNorm,
             "Accumulated discarded-norm bound D (0.0 = exact).")
        .def("definiteOccupation", &LazyFockState::definiteOccupation,
             "Definite total occupation, or -1 when indefinite.")
        .def("definiteParity", &LazyFockState::definiteParity,
             "Definite fermion parity +1/-1, or 0 when indefinite.")
        .def("boundaryFixtureLabel", &LazyFockState::boundaryFixtureLabel,
             py::return_value_policy::copy)
        .def("isBoundaryFixture", &LazyFockState::isBoundaryFixture);

    py::class_<LazyScalarRead>(m, "LazyScalarRead",
        R"doc(A scalar read (amplitude / inner product / squared norm) with its
accumulated discarded norm (0.0 in exact certification mode) and its
Certificate: AlgebraicallyExact when the discarded norm is exactly zero,
CertifiedNumerical with residual = the absolute discarded-norm bound
otherwise. Amplitude reads satisfy |value - exact| <= discardedNorm.)doc")
        .def_readonly("value", &LazyScalarRead::value)
        .def_readonly("discardedNorm", &LazyScalarRead::discardedNorm)
        .def_readonly("certificate", &LazyScalarRead::certificate);

    py::class_<LazySlaterReference>(m, "LazySlaterReference",
        R"doc(The optional quasi-free/Slater reference from a spectral projector:
covariance Gamma_ef = P_ef exactly (StructureExact given the verified
premise P^2 = P = P^dagger, residual on the certificate). CovarianceState
is the primary representation of the quasi-free sector; this engine is its
dense reference.)doc")
        .def_readonly("state", &LazySlaterReference::state)
        .def_readonly("rank", &LazySlaterReference::rank)
        .def_readonly("projectorResidual",
                      &LazySlaterReference::projectorResidual)
        .def_readonly("certificate", &LazySlaterReference::certificate);

    py::class_<LazyCovarianceRead>(m, "LazyCovarianceRead",
        R"doc(Covariance read Gamma_ef = <a_f^dagger a_e>/<psi|psi> over the full
mode universe (vacuum rows/columns outside the state's support).
Certificate residual = measured Hermiticity defect (general path) or the
trace defect |tr Gamma - n| (closed-form Slater path).)doc")
        .def_readonly("matrix", &LazyCovarianceRead::matrix)
        .def_readonly("discardedNorm", &LazyCovarianceRead::discardedNorm)
        .def_readonly("certificate", &LazyCovarianceRead::certificate);

    py::class_<LazyCompatibilityRead>(m, "LazyCompatibilityRead",
        R"doc(Inductive compatibility read for the vacuum embedding:
epsilon = ||iota_M U_M - U_{M+1} iota_M|| on the active carried subspace
(top singular value of the column-stacked defect).)doc")
        .def_readonly("epsilon", &LazyCompatibilityRead::epsilon)
        .def_readonly("activeDimension",
                      &LazyCompatibilityRead::activeDimension)
        .def_readonly("certificate", &LazyCompatibilityRead::certificate);

    py::class_<FockRefinementStage>(m, "FockRefinementStage",
        R"doc(One stage M of a refinement sequence: the modes H_M it carries, the
support of the map V_M on them, and V_M dense over the 2^|support| support Fock
basis. The stages are nested, because the step from one to the next is the vacuum
embedding, which adds modes and changes no amplitude.)doc")
        .def(py::init<>())
        .def(py::init([](std::vector<std::size_t> modes,
                         std::vector<std::size_t> support,
                         Eigen::MatrixXcd map) {
                 return FockRefinementStage{std::move(modes),
                                            std::move(support),
                                            std::move(map)};
             }),
             py::arg("modes"), py::arg("support"), py::arg("map"))
        .def_readwrite("modes", &FockRefinementStage::modes)
        .def_readwrite("support", &FockRefinementStage::support)
        .def_readwrite("map", &FockRefinementStage::map);

    py::class_<LazyInductiveLimitRead>(m, "LazyInductiveLimitRead",
        R"doc(The inductive limit read over a refinement sequence: one compatibility
defect per adjacent pair of stages, all on one active carried subspace, the last
defect, the worst step-to-step ratio, and whether the sequence falls. The
whitepaper's consistency condition ||iota_M V_M - V_{M+1} iota_M|| -> 0 is a
statement about a sequence, which one pair of stages cannot establish.)doc")
        .def_readonly("defects", &LazyInductiveLimitRead::defects)
        .def_readonly("steps", &LazyInductiveLimitRead::steps)
        .def_readonly("lastDefect", &LazyInductiveLimitRead::lastDefect)
        .def_readonly("largestRatio", &LazyInductiveLimitRead::largestRatio)
        .def_readonly("falls", &LazyInductiveLimitRead::falls)
        .def_readonly("activeDimension",
                      &LazyInductiveLimitRead::activeDimension)
        .def_readonly("certificate", &LazyInductiveLimitRead::certificate);

    py::class_<LazyFockEngine>(m, "LazyFockEngine",
        R"doc(The lazy graded Fock oracle and boundary carrier.

One-particle mode space h = span{|e>} (one two-level mode per edge, in the
EdgeModeRegistry compilation order); global carrier F_-(h) = Lambda* h.
States are expression DAGs — vacuum, sparse occupation blocks, graded
tensor products, local unitary/cobordism maps, direct sums by conserved
occupation/parity sector, antisymmetrized wedges — evaluated lazily: a
graded tensor partition is expanded only when an applied operation crosses
it; exact subexpressions are memoized by content hash; block sparsity by
occupation/parity short-circuits out-of-sector reads.

Exact identities (stated with their domains in the C++ header): the Koszul
graded-tensor amplitude rule (strictly associative, so parenthesizations
agree exactly); even/odd local-operator action through the two Koszul
signs; Slater-determinant amplitudes, det-Gram norms, and
Gamma = V (V^dagger V)^{-1} V^dagger covariance; bit-level dGamma with the
CAR sign rule (direct sums become graded tensor products, coupling blocks
become hopping terms); free subset-sum spectra delegated to
cobordism::OccupationSpectra; the vacuum embedding iota psi = psi (x) |0>,
which preserves every preexisting amplitude, with its compatibility read.

Exact certification mode by default (algebraically lossless rewrites
only); optional truncation mode accumulates and reports its discarded norm
in every scalar read. CovarianceState is the primary representation of the
quasi-free sector; this engine is its dense reference and the carrier for
explicitly non-Gaussian boundary data.)doc")
        .def(py::init<std::size_t>(), py::arg("modeCount"))
        .def_static("fromRegistry", &LazyFockEngine::fromRegistry,
             py::arg("registry"),
             "Engine over the registry's canonical compilation order.")
        .def("modeCount", &LazyFockEngine::modeCount)
        .def_static("stageDimension", &LazyFockEngine::stageDimension,
             py::arg("stageModeCount"),
             "dim Lambda* C^m = 2^m (enumerable stages, m <= 63).")
        .def("setTruncationThreshold",
             &LazyFockEngine::setTruncationThreshold, py::arg("threshold"),
             py::arg("normTolerance"),
             "Enter truncation mode with a stated threshold and declared "
             "discarded-norm budget.")
        .def("clearTruncation", &LazyFockEngine::clearTruncation,
             "Return to exact certification mode.")
        .def("exactMode", &LazyFockEngine::exactMode)
        .def("truncationThreshold", &LazyFockEngine::truncationThreshold)
        .def("truncationNormTolerance",
             &LazyFockEngine::truncationNormTolerance)
        .def("setMaxExpansionTerms", &LazyFockEngine::setMaxExpansionTerms,
             py::arg("maxTerms"))
        .def("maxExpansionTerms", &LazyFockEngine::maxExpansionTerms)
        .def("vacuum", &LazyFockEngine::vacuum,
             "The vacuum over the full mode universe.")
        .def("vacuumOn", &LazyFockEngine::vacuumOn, py::arg("modes"))
        .def("occupationState", &LazyFockEngine::occupationState,
             py::arg("modes"), py::arg("occupations"), py::arg("amplitudes"),
             "Sparse occupation block from (occupied-mode list, amplitude) "
             "terms.")
        .def("wedgeState", &LazyFockEngine::wedgeState, py::arg("modes"),
             py::arg("orbitals"),
             "v_1 ^ ... ^ v_n (orbitals |modes| x n; n = 1 is a general "
             "one-particle state).")
        .def("slaterFromProjector", &LazyFockEngine::slaterFromProjector,
             py::arg("modes"), py::arg("projector"), py::arg("tolerance"),
             "Quasi-free/Slater reference with covariance Gamma = P "
             "exactly.")
        .def("gradedTensor", &LazyFockEngine::gradedTensor, py::arg("a"),
             py::arg("b"),
             "a (x) b over disjoint (arbitrarily interleaved) mode sets.")
        .def("sectorSum", &LazyFockEngine::sectorSum, py::arg("children"),
             py::arg("kind"),
             "Direct sum of states in distinct conserved sectors.")
        .def("boundaryProductFixture",
             &LazyFockEngine::boundaryProductFixture, py::arg("modes"),
             py::arg("emptyAmplitudes"), py::arg("occupiedAmplitudes"),
             py::arg("label"),
             "The labeled optional product boundary fixture.")
        .def("embedInVacuum", &LazyFockEngine::embedInVacuum,
             py::arg("state"), py::arg("newModes"),
             "iota: psi -> psi (x) |0>; preserves every preexisting "
             "amplitude exactly.")
        .def("applyLocalMapDense", &LazyFockEngine::applyLocalMapDense,
             py::arg("state"), py::arg("supportModes"), py::arg("op"),
             "Apply a local map given dense over the 2^{|support|} support "
             "Fock basis.")
        .def("applyLocalMapCOO", &LazyFockEngine::applyLocalMapCOO,
             py::arg("state"), py::arg("supportModes"), py::arg("rows"),
             py::arg("cols"), py::arg("values"),
             "Apply a local map given as COO triplets over the support "
             "Fock basis.")
        .def("applyCreation", &LazyFockEngine::applyCreation,
             py::arg("state"), py::arg("mode"))
        .def("applyAnnihilation", &LazyFockEngine::applyAnnihilation,
             py::arg("state"), py::arg("mode"))
        .def("applyDGamma", &LazyFockEngine::applyDGamma, py::arg("state"),
             py::arg("supportModes"), py::arg("oneParticle"),
             "Apply dGamma(L) at the bit level (arbitrary support size; no "
             "2^{|S|} operator is formed).")
        .def("materialize", &LazyFockEngine::materialize, py::arg("state"),
             "Expand to one sparse occupation node (lossless in exact "
             "mode; drops and accounts |a| <= threshold in truncation "
             "mode).")
        .def("permuteModes", &LazyFockEngine::permuteModes, py::arg("state"),
             py::arg("perm"),
             "Exact signed mode relabeling (permutationParity per term).")
        .def("amplitude", &LazyFockEngine::amplitude, py::arg("state"),
             py::arg("occupiedModes"),
             "<b|psi> for the basis state occupying exactly occupiedModes.")
        .def("innerProduct", &LazyFockEngine::innerProduct, py::arg("a"),
             py::arg("b"), "<a|b> (antilinear in a).")
        .def("normSquared", &LazyFockEngine::normSquared, py::arg("state"))
        .def("covarianceMatrix", &LazyFockEngine::covarianceMatrix,
             py::arg("state"),
             "Gamma_ef = <a_f^dagger a_e>/norm^2 over the full universe.")
        .def("denseVector", &LazyFockEngine::denseVector, py::arg("state"),
             "Full dense Fock vector (n(b) = sum_i b_i 2^i; capped at "
             "kMaxDenseModes).")
        .def("freeSpectrum",
             [](const LazyFockEngine& e, const Eigen::MatrixXcd& l,
                int particles) {
                 auto out = e.freeSpectrum(l, particles);
                 return py::make_tuple(out.values, out.certificate);
             }, py::arg("oneParticle"), py::arg("particles"),
             "(values, certificate): free N-particle spectrum of dGamma(L) "
             "via OccupationSpectra subset sums.")
        .def("freeSpectrumFromEigenvalues",
             [](const LazyFockEngine& e,
                const std::vector<std::complex<double>>& spec,
                int particles) {
                 auto out = e.freeSpectrumFromEigenvalues(spec, particles);
                 return py::make_tuple(out.values, out.certificate);
             }, py::arg("oneParticleSpectrum"), py::arg("particles"),
             "(values, certificate): OccupationSpectra delegation with an "
             "independent-path residual.")
        .def("inductiveCompatibility",
             &LazyFockEngine::inductiveCompatibility, py::arg("stageModes"),
             py::arg("extendedModes"), py::arg("stageSupport"),
             py::arg("stageOp"), py::arg("extendedSupport"),
             py::arg("extendedOp"), py::arg("activeBasis"),
             "epsilon = ||iota U_M - U_{M+1} iota|| on the active carried "
             "subspace.")
        .def("inductiveLimit", &LazyFockEngine::inductiveLimit,
             py::arg("stages"), py::arg("activeBasis"),
             "The inductive limit read over a refinement sequence: one "
             "compatibility defect per adjacent pair of stages, all on the "
             "same active carried subspace, with the defect sequence and "
             "whether it falls. The active basis states must lie inside the "
             "first stage's modes, so that one and the same subspace is "
             "carried through every embedding and the defects are "
             "commensurable.")
        .def("expansionCount", &LazyFockEngine::expansionCount,
             "Partition crossings that forced a tensor expansion.")
        .def("memoHits", &LazyFockEngine::memoHits)
        .def("memoMisses", &LazyFockEngine::memoMisses)
        .def("memoSize", &LazyFockEngine::memoSize)
        .def("clearMemo", &LazyFockEngine::clearMemo,
             "Drop memoized expansions (cold-vs-memoized comparisons).")
        .def("serialize", &LazyFockEngine::serialize, py::arg("state"),
             "Strict-JSON DAG checkpoint: shared nodes once (no "
             "flattening), content hash per node, bit-exact amplitudes.")
        .def("deserialize", &LazyFockEngine::deserialize, py::arg("json"),
             "Rebuild a checkpoint; verifies schema, universe, and every "
             "recomputed content hash.");

    // ── The quasi-free covariance layer ────────────────────────────────────

    py::class_<WickCertificateRead>(m, "WickCertificateRead",
        R"doc(One Wick-evaluated polynomial certificate: the value, the measured
residual (covariance Hermiticity defect, maximized with the imaginary
rounding leakage for real-by-construction observables), the normal-ordered
observable / contraction-plan identifier, the covariance fingerprint the
value was read from, and the Certificate grading the claim
(AlgebraicallyExact / Static; the regime is verified on the covariance).)doc")
        .def(py::init<>())
        .def_readonly("value", &WickCertificateRead::value)
        .def_readonly("residual", &WickCertificateRead::residual)
        .def_readonly("polynomialId", &WickCertificateRead::polynomialId)
        .def_readonly("covarianceHash", &WickCertificateRead::covarianceHash)
        .def_readonly("certificate", &WickCertificateRead::certificate);

    py::class_<MeanFieldStepRead>(m, "MeanFieldStepRead",
        R"doc(The per-iteration record of the mean-field self-consistency loop:
the measured generator/covariance Hermiticity defects, the purity defect
||Gamma^2 - Gamma||_F (pure-Slater path), the covariance-spectrum constraint
(mixed path), and the purity/Gaussianity certificate of the step.)doc")
        .def_readonly("step", &MeanFieldStepRead::step)
        .def_readonly("time", &MeanFieldStepRead::time)
        .def_readonly("generatorHermiticityDefect",
                      &MeanFieldStepRead::generatorHermiticityDefect)
        .def_readonly("hermiticityDefect", &MeanFieldStepRead::hermiticityDefect)
        .def_readonly("purityDefect", &MeanFieldStepRead::purityDefect)
        .def_readonly("occupationSpectrumDefect",
                      &MeanFieldStepRead::occupationSpectrumDefect)
        .def_readonly("dualityDefect", &MeanFieldStepRead::dualityDefect,
                      "Transpose path: ||Phi~^T Phi - I||_F after the step "
                      "(NaN on the Hermitian-adjoint path).")
        .def_readonly("certificate", &MeanFieldStepRead::certificate);

    py::enum_<CovarianceDual>(m, "CovarianceDual",
        R"doc(The dual a CovarianceState is paired against.

HermitianAdjoint: Gamma_ij = <Psi|a_j^dagger a_i|Psi> with the Hermitian
adjoint of one state, the special case of a certified *-structure (Hermitian
generator, U Gamma U^dagger transports). Transpose: Gamma = Phi Phi~^T over
a matched right/left Slater pair with Phi~^T Phi = I, the biorthogonal
Slater covariance of the complex-bilinear formulation; both frames are
carried and evolve under any complex generator, with no h^dagger.)doc")
        .value("HermitianAdjoint", CovarianceDual::HermitianAdjoint)
        .value("Transpose", CovarianceDual::Transpose);

    py::class_<CovarianceState>(m, "CovarianceState",
        R"doc(The number-conserving quasi-free state, stored exactly as its
covariance matrix Gamma_ij = <a_j^dagger a_i>.

Every polynomial observable is a finite exact Wick sum over Gamma —
occupations, parities, Gram/Pauli determinants, the color wedge |S_ABC|^2,
<J^2> and Var(J^2) (quartic and octic Wick sums) — at polynomial cost in
the mode count; no code path here allocates a 2^M object. Dense Fock
constructions (ExteriorAlgebra, Jordan-Wigner chains) are test references,
and LazyFockEngine is the oracle layer and the carrier for explicitly
non-Gaussian boundary data. The mean-field loop takes h from the caller.

Propagation (both entry points): evolve(h, dt) applies the exact solution
Gamma <- exp(-i h dt) Gamma exp(+i h dt) of i dGamma/dt = [h, Gamma];
applyTransport(U) conjugates by the one-particle transport of a cobordism
step. Quadratic evolution — including the mean-field self-consistency —
never leaves the Gaussian manifold; purity is a measured certificate
||Gamma^2 - Gamma||_F, never an assumption. The API is Nambu-shaped
(numberConserving / pairing / nambuCovariance) with the pairing block
identically zero.)doc")
        .def(py::init<Eigen::MatrixXcd>(), py::arg("gamma"),
             "Adopt an explicit covariance matrix (no symmetrization, no "
             "clamping: defects are measured and reported, never repaired).")
        .def_static("fromBandProjector", &CovarianceState::fromBandProjector,
             py::arg("projector"),
             R"doc(Gamma = P from an accepted band projector (SpectralFiber.
projector() output consumed as a plain matrix). Self-adjoint-path
projectors are orthogonal, hence pure Slater covariances; an oblique
projector is adopted verbatim and its Hermiticity defect reported.)doc")
        .def_static("fromOccupations", &CovarianceState::fromOccupations,
             py::arg("occupations"),
             "Diagonal Gamma = diag(n) from boundary-register occupation "
             "data.")
        .def_static("fromSlaterFrame", &CovarianceState::fromSlaterFrame,
             py::arg("orbitals"), py::arg("rankTolerance") = 1e-12,
             R"doc(Pure Slater covariance Gamma = Phi (Phi+ Phi)^-1 Phi+ from an
M x N frame of occupied one-particle orbitals (boundary-register state
vectors enter as occupied columns; the frame need not be orthonormal).)doc")
        .def_static("fromBiorthogonalFrames",
             &CovarianceState::fromBiorthogonalFrames, py::arg("rightFrame"),
             py::arg("leftFrame"),
             R"doc(The biorthogonal Slater covariance Gamma = Phi Phi~^T of a
matched right/left pair, paired by the transpose (Phi~^T Phi = I, no
conjugation): |Xi_R> = phi_1 ^ ... ^ phi_N, <Xi_L| = phi~_1 ^ ... ^ phi~_N,
Gamma_ij = <Xi_L|a_j^dagger a_i|Xi_R>. The state is on the Transpose dual
and carries both frames; the pairing defect is measured (dualityDefect),
never repaired. A SpectralFiber's rightFrame()/dualFrame() or a Riesz band's
frame/leftFrame enter as they are.)doc")
        .def("dual", &CovarianceState::dual,
             "The dual the covariance is paired against (CovarianceDual).")
        .def("rightFrame", &CovarianceState::rightFrame,
             py::return_value_policy::copy,
             "The right frame Phi of a Transpose state (0 x 0 otherwise).")
        .def("leftFrame", &CovarianceState::leftFrame,
             py::return_value_policy::copy,
             "The left frame Phi~ of a Transpose state, Gamma = Phi Phi~^T "
             "(0 x 0 otherwise).")
        .def("dualityDefect", &CovarianceState::dualityDefect,
             "||Phi~^T Phi - I||_F of a Transpose state — the premise of the "
             "biorthogonal Wick theorem; NaN on the Hermitian-adjoint path.")
        .def("modeCount", &CovarianceState::modeCount)
        .def("gamma", &CovarianceState::gamma,
             py::return_value_policy::copy,
             "The covariance matrix Gamma_ij = <a_j^dagger a_i>.")
        .def("numberConserving", &CovarianceState::numberConserving,
             "True in this implementation; a pairing extension reports "
             "False and populates pairing().")
        .def("pairing", &CovarianceState::pairing,
             "The anomalous block F_ij = <a_j a_i> — identically zero in "
             "the number-conserving sector (Nambu-shaped API).")
        .def("nambuCovariance", &CovarianceState::nambuCovariance,
             R"doc(The 2M x 2M Nambu covariance over alpha = (a, a+): blocks
[[Gamma, F], [-conj(F), I - Gamma^T]] (number-conserving: F = 0);
idempotent exactly when Gamma is.)doc")
        .def("occupation", &CovarianceState::occupation, py::arg("mode"),
             "<n_mode> = Gamma_mm (complex diagonal entry — real up to the "
             "Hermiticity defect).")
        .def("occupations", &CovarianceState::occupations,
             "The diagonal <n_i> for every mode.")
        .def("particleNumber", &CovarianceState::particleNumber,
             "<N> = tr Gamma.")
        .def("hermiticityDefect", &CovarianceState::hermiticityDefect,
             "||Gamma - Gamma+||_F / max(1, ||Gamma||_F).")
        .def("purityDefect", &CovarianceState::purityDefect,
             "||Gamma^2 - Gamma||_F — exactly zero for a pure Slater state; "
             "an O(1) value is a mixed covariance reporting itself.")
        .def("occupationSpectrumDefect",
             &CovarianceState::occupationSpectrumDefect,
             "max_i dist(lambda_i, [0, 1]) of the Hermitian part — the "
             "mixed-state covariance-spectrum constraint.")
        .def("purityCertificate", &CovarianceState::purityCertificate,
             py::arg("tolerance") = 1e-9,
             "The purity certificate of the pure-Slater path; a mixed state "
             "does not hold() it.")
        .def("covarianceHash", &CovarianceState::covarianceHash,
             "Order-sensitive fingerprint of the exact double bit patterns "
             "of Gamma (16 hex digits) — replay-stable.")
        .def("evolve", &CovarianceState::evolve, py::arg("h"), py::arg("dt"),
             py::arg("hermitianTolerance") = 1e-9,
             R"doc(Gamma <- exp(-i h dt) Gamma exp(+i h dt): the exact solution of
i dGamma/dt = [h, Gamma] (no step-size error).

Hermitian-adjoint path: h must be Hermitian (throws otherwise); Hermiticity,
spectrum, and purity are preserved to round-off. Transpose path: h is any
complex matrix (e.g. complex-symmetric) and no h^dagger is formed; the
frames evolve by i dPhi/dt = h Phi and -i dPhi~^T/dt = Phi~^T h, i.e.
Phi <- exp(-i h dt) Phi and Phi~ <- exp(+i h dt)^T Phi~, and Gamma is
rebuilt as Phi Phi~^T; the pairing and Gamma^2 = Gamma are preserved.)doc")
        .def_static("propagator", &CovarianceState::propagator, py::arg("h"),
             py::arg("dt"), py::arg("hermitianTolerance") = 1e-9,
             "The Hermitian-path propagator exp(-i h dt) evolve conjugates "
             "by: evolve(h, dt) == applyTransport(propagator(h, dt)).")
        .def_static("complexPropagator", &CovarianceState::complexPropagator,
             py::arg("h"), py::arg("dt"),
             "exp(-i h dt) for an arbitrary complex generator (Pade scaling "
             "and squaring; no eigendecomposition, no h^dagger).")
        .def("applyTransport", &CovarianceState::applyTransport,
             py::arg("transport"),
             R"doc(Carry the state through the one-particle transport U of a
cobordism step. Hermitian-adjoint path: Gamma <- U Gamma U^dagger (a leaky
transport's effect shows up in the defect reads and is never repaired).
Transpose path: Phi <- U Phi and Phi~ <- U^{-T} Phi~, so Gamma <- U Gamma
U^{-1} and the pairing is preserved; a singular U is refused.)doc")
        .def("meanFieldEvolve", &CovarianceState::meanFieldEvolve,
             py::arg("hamiltonian"), py::arg("dt"), py::arg("steps"),
             py::arg("hermitianTolerance") = 1e-9,
             py::arg("purityTolerance") = 1e-9,
             R"doc(The certificates-blind mean-field loop: each iteration obtains
h = hamiltonian(Gamma) from the caller (classical geometry is closed over
by the caller), advances by dt via evolve, and records the
purity/Gaussianity certificate. Generalized Hartree-Fock: nonlinear in
Gamma but Gaussian-closed, and that closure is measured every step.)doc")
        .def("wickOccupation", &CovarianceState::wickOccupation,
             py::arg("mode"), "<n_mode> as a certified Wick read.")
        .def("wickTotalNumber", &CovarianceState::wickTotalNumber,
             "<N> = tr Gamma as a certified Wick read.")
        .def("wickParity", &CovarianceState::wickParity,
             "Fermion parity <(-1)^N> = det(I - 2 Gamma).")
        .def("wickSubsetParity", &CovarianceState::wickSubsetParity,
             py::arg("modes"),
             "Subset parity <(-1)^{N_S}> = det(I_S - 2 Gamma_S) on the "
             "principal submatrix.")
        .def("wickNormalOrdered", &CovarianceState::wickNormalOrdered,
             py::arg("creators"), py::arg("annihilators"),
             R"doc(<a+_{c1}...a+_{cp} a_{ap}...a_{a1}> = det[Gamma_{a_l c_k}] in
paired slot order (annihilators applied in reversed list order, so equal
distinct lists give the joint occupation <n_{c1}...n_{cp}>). Mismatched
lengths are exactly zero on a number-conserving state; duplicate creators
give a repeated determinant row — an exact Pauli zero.)doc")
        .def("wickGramDeterminant", &CovarianceState::wickGramDeterminant,
             py::arg("creatorFrame"), py::arg("annihilatorFrame"),
             R"doc(The smeared Gram/Pauli determinant
<a+(v_1)...a+(v_p) a(w_p)...a(w_1)> = det(W+ Gamma V) with the
ExteriorAlgebra smearing conventions (columns of V create, columns of W
annihilate).)doc")
        .def("wickTransposeGramDeterminant",
             &CovarianceState::wickTransposeGramDeterminant,
             py::arg("creatorFrame"), py::arg("annihilatorFrame"),
             R"doc(The transpose-paired smeared Gram determinant
<a+(v_1)...a+(v_p) a~(w_p)...a~(w_1)> = det(W^T Gamma V), with the
annihilator smeared by the transpose pairing a~(w) = sum_i w_i a_i (linear,
never conjugated). On a Transpose state it is the transition amplitude of
the left/right pair.)doc")
        .def("wickColorWedgeSquared", &CovarianceState::wickColorWedgeSquared,
             py::arg("colorColumns"),
             R"doc(|S_ABC|^2 = det(C+ Gamma C) of three color columns. When Gamma
is the Slater projector onto colspan(C) this equals det(C+ C) = |det C|^2,
i.e. ColorFiber.singletGram / |ColorFiber.colorWedge|^2.)doc")
        .def("wickBilinearMoment", &CovarianceState::wickBilinearMoment,
             py::arg("oneParticleFactors"),
             R"doc(The ordered bilinear moment <dGamma(A_1)...dGamma(A_n)> —
the general quartic (n = 2) and octic (n = 4) Wick engine, evaluated by
the exact set-partition/ordered-composition trace expansion of
det(I + (prod_k(I + s_k A_k) - I) Gamma). Polynomial in the mode count.)doc")
        .def("wickSpinSquaredExpectation",
             &CovarianceState::wickSpinSquaredExpectation, py::arg("jx"),
             py::arg("jy"), py::arg("jz"),
             "<J^2> = sum_alpha <dGamma(J_alpha)^2> from CALLER-SUPPLIED "
             "one-particle spin matrices (quartic Wick sums).")
        .def("wickSpinSquaredVariance",
             &CovarianceState::wickSpinSquaredVariance, py::arg("jx"),
             py::arg("jy"), py::arg("jz"),
             "Var(J^2) = <(J^2)^2> - <J^2>^2 (octic Wick sums); exactly "
             "zero on a J^2 eigenstate.")
        .def("wickReadCached", &CovarianceState::wickReadCached,
             py::arg("cache"), py::arg("componentVertexIds"),
             py::arg("polynomialId"), py::arg("compute"),
             R"doc(Fetch-or-compute one Wick read through the AnalyticCache
contract (kind "wick-read"; parameter = a mixed fingerprint of the
polynomialId and the current covarianceHash). A hit is served only when
the cache's geometry-freshness contract holds and the stored polynomialId
and covarianceHash both match, so a Gamma change can only cause
recomputation, never a wrong serve.)doc")
        .def("toRecord",
             [](const CovarianceState& self) {
                 return quantumRecordToPython(self.toRecord());
             },
             "Checkpoint serialization of Gamma (schema-versioned; complex "
             "leaves split gamma_re / gamma_im).")
        .def_static("fromRecord",
             [](const py::handle& record) {
                 return CovarianceState::fromRecord(
                     quantumPythonToRecord(record));
             },
             py::arg("record"),
             "Rehydrate from toRecord() output; rejects an unknown "
             "schema_version.");
}
