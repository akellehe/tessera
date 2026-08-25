// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "spacetime/topologies/Topology.h"
#include "spacetime/topologies/Cylinder.h"
#include "spacetime/topologies/Sphere.h"
#include "spacetime/topologies/Toroid.h"
#include "simulations/CDT.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/pachner/AddMove.h"
#include "spacetime/pachner/FlipMove.h"
#include "spacetime/pachner/IFlipMove.h"
#include "spacetime/pachner/RemoveMove.h"
#include "spacetime/pachner/ShiftMove.h"
#include "simulations/ReggeSolver.h"
#include "matter/MatterConfiguration.h"
#include "mesh/SimplexFilter.h"
#include "observables/ModularityOptimizer.h"
#include "observables/PersistentModularity.h"
#include "observables/SpectralFiber.h"
#include "observables/SparseGraph.h"
#include "cobordism/AnalyticCache.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
#include "observables/Spectral.h"
#include "observables/SimplicialQubit.h"
#include "chainhodge/CovariantChainHodge.h"
#include "observables/Record.h"
#include "observables/ClusterRegister.h"
#include "observables/RegisterContext.h"
#include "observables/RegisterObservable.h"
#include "observables/InteriorHinges.h"
#include "observables/LiveComplex.h"
#include "observables/SingletResidual.h"
#include "observables/BlockResiduals.h"
#include "observables/EmergentMass.h"
#include "observables/EmergentRadius.h"
#include "observables/PairLoopFlavor.h"
#include "observables/ObservableGates.h"
#include "observables/DualVolumeSigns.h"
#include "observables/ColorFiber.h"
#include "observables/CrossingReadouts.h"
#include "observables/ExchangeHolonomy.h"
#include "observables/FiberConnection.h"
#include "observables/ParticleClusters.h"
#include "cobordism/Proton.h"
#include "spacetime/Spacetime.h"
#include "ForceLayout.h"
#include "mesh/VertexList.h"
#include "mesh/EdgeList.h"
#include "spacetime/Signature.h"
#include "mesh/Vertex.h"
#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "spacetime/Metric.h"
#include "Renderer.h"

#include <vector>
#include <algorithm>


namespace py = pybind11;
using namespace tessera;
using namespace tessera::observables;

namespace {

// A JSON-able observable Record -> a native Python object (dict/list/scalar).
py::object recordToPython(const Record &r) {
  switch (r.type()) {
    case Record::Type::Null:
      return py::none();
    case Record::Type::Bool:
      return py::bool_(r.asBool());
    case Record::Type::Int:
      return py::int_(static_cast<long long>(r.asInt()));
    case Record::Type::Double:
      return py::float_(r.asDouble());
    case Record::Type::String:
      return py::str(r.asString());
    case Record::Type::List: {
      py::list out;
      for (const auto &e : r.asList()) out.append(recordToPython(e));
      return out;
    }
    case Record::Type::Map: {
      py::dict out;
      for (const auto &kv : r.asMap()) {
        out[py::str(kv.first)] = recordToPython(kv.second);
      }
      return out;
    }
  }
  return py::none();
}

// A native Python object -> a Record (for report_delta testing). bool is checked
// before int (Python bool is an int subtype).
Record pythonToRecord(const py::handle &o) {
  if (o.is_none()) return Record();
  if (py::isinstance<py::bool_>(o)) return Record(o.cast<bool>());
  if (py::isinstance<py::int_>(o)) {
    return Record(static_cast<std::int64_t>(o.cast<long long>()));
  }
  if (py::isinstance<py::float_>(o)) return Record(o.cast<double>());
  if (py::isinstance<py::str>(o)) return Record(o.cast<std::string>());
  if (py::isinstance<py::dict>(o)) {
    Record::Map m;
    for (const auto &item : o.cast<py::dict>()) {
      m[item.first.cast<std::string>()] = pythonToRecord(item.second);
    }
    return Record(std::move(m));
  }
  if (py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
    Record::List l;
    for (const auto &e : o) l.push_back(pythonToRecord(e));
    return Record(std::move(l));
  }
  throw std::runtime_error(
      "report_delta: record leaves must be dict/list/str/float/int/bool/None");
}

// Emit the RegisterContext's surplus-selection warning as a Python UserWarning
// (the header's documented binding behavior).
void emitSelectionWarning(const RegisterContext &ctx) {
  if (!ctx.selectionWarning().empty()) {
    auto warnings = py::module_::import("warnings");
    warnings.attr("warn")(ctx.selectionWarning());
  }
}

}  // namespace

// Registers all tessera::observables classes into the `m` submodule
// (i.e. `tessera.observables`). Called from src/bindings.cpp's
// PYBIND11_MODULE entry point.
void register_observables(py::module_ m) {
  // ========================================
  // SparseGraph (for modularity / spectral dimension)
  // ========================================
  py::class_<SparseGraph>(m, "SparseGraph",
      R"doc(Undirected sparse graph in CSR form.

Built from the COO output of ``Spacetime.getDualAdjacency`` (or
similar).  Used by the modularity sweep to compute spectral
dimension on the dual graph.)doc")
      .def_static("fromCOO", &SparseGraph::fromCOO,
                  py::arg("rows"), py::arg("cols"), py::arg("n"),
                  "Construct from COO arrays + node count.")
      .def("nNodes", &SparseGraph::nNodes,
           "Number of nodes.")
      .def("nEdges", &SparseGraph::nEdges,
           "Number of undirected edges.")
      .def("degree", &SparseGraph::degree, py::arg("i"),
           "Degree of node ``i`` (number of incident undirected edges).")
      .def("isBipartite", &SparseGraph::isBipartite,
           "True iff the graph is 2-colorable (no odd cycle).")
      .def("modularity", &SparseGraph::modularity, py::arg("labels"),
           R"doc(Newman-Girvan modularity Q for a node partition.

Q = sum_c [L_c/m - (D_c/2m)^2] over communities c, where L_c is the
intra-community edge count, D_c the summed degree, and m the edge count.
``labels`` has one community id per node (length nNodes()); distinct
values are distinct communities and need not be dense. Returns 0 for an
empty / edgeless graph; raises ValueError if len(labels) != nNodes().)doc")
      .def("diagonalHeatKernel",
           [](const SparseGraph &self,
              const std::vector<std::uint32_t> &starts,
              const std::vector<double> &times,
              int krylovDim) {
             auto flat = self.diagonalHeatKernel(starts, times, krylovDim);
             // Reshape to list-of-lists for Python convenience.
             std::vector<std::vector<double>> out(starts.size(),
                                                   std::vector<double>(times.size()));
             for (std::size_t s = 0; s < starts.size(); ++s) {
               for (std::size_t j = 0; j < times.size(); ++j) {
                 out[s][j] = flat[s * times.size() + j];
               }
             }
             return out;
           },
           py::arg("starts"), py::arg("times"), py::arg("krylovDim") = 30,
           R"doc(Diagonal of the heat kernel ``e^{-t L_sym}`` for each
(start, t) pair.  Returns a list of lists with shape
(len(starts), len(times)).)doc")
      .def("spectralDimension",
           [](const SparseGraph &self, int nWalks, double maxSigma,
              std::uint64_t seed, double tailFraction, int nTimes,
              double tMin, int krylovDim) {
             std::mt19937 rng(seed);
             return self.spectralDimension(nWalks, maxSigma, &rng,
                                           tailFraction, nTimes, tMin,
                                           krylovDim);
           },
           py::arg("nWalks"), py::arg("maxSigma"), py::arg("seed") = 0,
           py::arg("tailFraction") = 0.2, py::arg("nTimes") = 40,
           py::arg("tMin") = 0.5, py::arg("krylovDim") = 30,
           R"doc(Estimate spectral dimension at small / large diffusion times.

Returns ``(D_S_small, D_S_large)``.  Mirrors the Python implementation
in ``examples/modularity.py:Graph.spectral_dimension``.)doc")
      // ── Per-sigma return probability + spectral-dimension curve ──────────
      // Inherited from SpectralGraph (SparseGraph::applyLaplacian installs
      // the symmetric-normalised Laplacian L_sym).  Exposed here so the
      // examples can read the full D_S(sigma) curve straight off the dual
      // graph instead of re-implementing dual-graph diffusion + finite
      // differences in NumPy/SciPy.  Mirrors the EmergentGraph bindings.
      .def("returnProbability",
           &::tessera::graph::SpectralGraph::returnProbability,
           py::arg("sigmas"), py::arg("krylovDim") = 30,
           py::arg("m") = 0, py::arg("seed") = 0,
           R"doc(P(sigma) = (1/|V|) Tr exp(-sigma L_sym) via Krylov-Lanczos
diagonal estimation, evaluated at each diffusion time in ``sigmas``.

``m`` is the Hutchinson-style subsample of start vertices: 0 (the default)
uses ``min(nNodes(), 3000)``; pass ``m = nNodes()`` for the exact trace.
``seed`` controls the subset RNG for reproducibility.

This is the continuous-diffusion counterpart of the discrete random-walk
return probability the modularity / spectral-dimension examples used to
build by hand; feed the result to ``spectralDimensionCurve`` (or
``spectralDimensionSmoothed``) to extract D_S(sigma).)doc")
      .def_static("spectralDimensionCurve",
                  &::tessera::graph::SpectralGraph::spectralDimension,
                  py::arg("sigmas"), py::arg("P"),
                  R"doc(D_S(sigma) = -2 d log P / d log sigma via centered finite
differences (one-sided at the endpoints); NaN where P <= 0 or non-finite.

The full per-sigma curve, aligned with ``sigmas``.  Distinct from the
``spectralDimension(nWalks, maxSigma, ...)`` instance method above, which
random-walk samples and returns only the (small, large) summary pair.)doc")
      .def_static("spectralDimensionSmoothed",
                  &::tessera::graph::SpectralGraph::spectralDimensionSmoothed,
                  py::arg("sigmas"), py::arg("P"),
                  py::arg("windowSize") = 5, py::arg("polyOrder") = 2,
                  R"doc(Savitzky-Golay-smoothed D_S(sigma): a local polynomial of
order ``polyOrder`` is fit over a centered ``windowSize`` window in
(log sigma, log P) and its slope read off at each point.  ``windowSize``
must be odd and >= ``polyOrder + 1``.)doc");
  // ========================================
  // PersistentModularity (#765): label-free persistent component discovery
  // ========================================
  py::class_<ComponentId>(m, "ComponentId",
      R"doc(Stable label-free component identity: a canonical hash derived
from oriented incidence structure and parent lineage (never raw vertex
numbers) plus the multilevel-aggregation level.  Used for persistence
matching and deterministic tie-breaking, never as a physical observable.
Structurally identical (automorphic) components share a hash.)doc")
      .def(py::init<>())
      .def(py::init<std::string, std::size_t>(), py::arg("hash"),
           py::arg("level"),
           "Assemble an identity from its parts (replay/synthetic-fixture "
           "route; discovery normally mints these).")
      .def("canonicalHash", &ComponentId::canonicalHash,
           "The canonical structural hash (32 lowercase hex chars).")
      .def("level", &ComponentId::level,
           "Multilevel-aggregation depth at which the component formed.")
      .def("__eq__", [](const ComponentId &a, const ComponentId &b) {
             return a == b;
           }, py::is_operator())
      .def("__lt__", [](const ComponentId &a, const ComponentId &b) {
             return a < b;
           }, py::is_operator())
      .def("__hash__", [](const ComponentId &a) {
             return py::hash(py::make_tuple(a.canonicalHash(), a.level()));
           })
      .def("__repr__", [](const ComponentId &a) {
             return "ComponentId(" + a.canonicalHash() + ", level=" +
                    std::to_string(a.level()) + ")";
           });

  py::enum_<DiscoveryStrategy>(m, "DiscoveryStrategy",
      "Which search proposes the communities.  Both score the SAME exact "
      "Q_gamma closed form, so their slices are directly comparable; they "
      "differ only in how a partition is searched for.  An enum rather than "
      "a name string, so a mis-spelling is an error instead of a value that "
      "silently selects a default.")
      .value("MultilevelAggregation", DiscoveryStrategy::MultilevelAggregation,
             "Multilevel aggregation from a fixed restart seed sequence, "
             "keeping the best exact score and reporting the restart spread.")
      .value("LeadingEigenvector", DiscoveryStrategy::LeadingEigenvector,
             "Newman's leading-eigenvector bisection of B_gamma = A - gamma "
             "k k^T / 2m, recursed until no group has a positive leading "
             "eigenvalue.  The community COUNT is fixed by the spectrum "
             "rather than by a parameter, and the search carries no seed.");

  py::class_<SplitReason>(m, "SplitReason",
      "The named outcomes of one attempted leading-eigenvector bisection.  "
      "Reference these constants rather than retyping the strings: a "
      "mis-spelled literal produces a reason no consumer matches.")
      .def_property_readonly_static("SPLIT_ACCEPTED",
          [](py::object) { return SplitReason::kSplitAccepted; })
      .def_property_readonly_static("NO_POSITIVE_EIGENVALUE",
          [](py::object) { return SplitReason::kNoPositiveEigenvalue; })
      .def_property_readonly_static("DEGENERATE_LEADING_PAIR",
          [](py::object) { return SplitReason::kDegenerateLeadingPair; })
      .def_property_readonly_static("GROUP_TOO_SMALL",
          [](py::object) { return SplitReason::kGroupTooSmall; })
      .def_property_readonly_static("EMPTY_SIDE",
          [](py::object) { return SplitReason::kEmptySide; })
      .def_property_readonly_static("SPLIT_LOWERS_MODULARITY",
          [](py::object) { return SplitReason::kSplitLowersModularity; })
      .def_property_readonly_static("POWER_ITERATION_NOT_CONVERGED",
          [](py::object) { return SplitReason::kPowerIterationNotConverged; });

  py::class_<SplitRead>(m, "SplitRead",
      "One attempted bisection: the spectrum that decided it, and what was "
      "decided.  Unmeasured quantities are NaN, never zero.")
      .def_readonly("groupSize", &SplitRead::groupSize)
      .def_readonly("leadingEigenvalue", &SplitRead::leadingEigenvalue,
                    "Most positive eigenvalue of B_gamma on the group, over "
                    "the complement of the all-ones vector.  NaN if not "
                    "computed.")
      .def_readonly("secondEigenvalue", &SplitRead::secondEigenvalue,
                    "Second most positive eigenvalue, by deflation.  NaN if "
                    "not computed.")
      .def_readonly("eigenvalueGap", &SplitRead::eigenvalueGap,
                    "leadingEigenvalue - secondEigenvalue: how well "
                    "determined the bisection is.  NaN if either is "
                    "unmeasured.")
      .def_readonly("deltaQ", &SplitRead::deltaQ,
                    "Exact change in total Q_gamma this split would produce, "
                    "from the class's own closed form.  NaN if no split was "
                    "evaluated.")
      .def_readonly("accepted", &SplitRead::accepted,
                    "Whether the group was actually bisected.")
      .def_readonly("resolved", &SplitRead::resolved,
                    "Whether the spectrum determined the outcome.  False "
                    "means the split was REFUSED as not well determined, "
                    "which is distinct from a determined 'do not split'.")
      .def_readonly("reason", &SplitRead::reason,
                    "One of the SplitReason constants.")
      .def_readonly("sizeA", &SplitRead::sizeA)
      .def_readonly("sizeB", &SplitRead::sizeB);

  py::class_<PersistentModularityConfig>(m, "PersistentModularityConfig",
      "Configuration for the label-free multiscale component discovery.")
      .def(py::init<>())
      .def_readwrite("strategy", &PersistentModularityConfig::strategy,
                     "Which search proposes the communities.  Default keeps "
                     "the incumbent multilevel aggregation.")
      .def_readwrite("objective", &PersistentModularityConfig::objective,
                     R"doc(Which real functional of Q the search maximizes.
Default Score keeps the incumbent's behaviour on a real graph; a COMPLEX
graph selects Magnitude regardless, since Score is not an ordering there.

Setting Magnitude on a real graph is how ANTI-community structure is
pursued: it has Q < 0, so maximizing Q passes it over in favour of the
one-community partition while maximizing |Q| finds it.)doc")
      .def_readwrite("leadingEigenvalueTolerance",
                     &PersistentModularityConfig::leadingEigenvalueTolerance,
                     "LeadingEigenvector: a group is indivisible when its "
                     "leading eigenvalue does not exceed this.")
      .def_readwrite("minEigenvalueGap",
                     &PersistentModularityConfig::minEigenvalueGap,
                     "LeadingEigenvector: minimum leading-to-second gap for "
                     "a bisection to count as well determined.  Below it the "
                     "split is refused with a named reason.")
      .def_readwrite("denseEigenSolveMaxGroup",
                     &PersistentModularityConfig::denseEigenSolveMaxGroup,
                     "LeadingEigenvector: groups of at most this many cells "
                     "get an EXACT dense symmetric eigendecomposition; larger "
                     "groups fall back to shifted power iteration.  The dense "
                     "path exists because iteration is slowest exactly where "
                     "the pair is near-degenerate, which is the case the gap "
                     "certificate has to adjudicate.")
      .def_readwrite("maxPowerIterations",
                     &PersistentModularityConfig::maxPowerIterations,
                     "LeadingEigenvector: hard cap on power-iteration steps "
                     "per eigenpair above denseEigenSolveMaxGroup.  "
                     "Non-convergence is reported.")
      .def_readwrite("powerIterationTolerance",
                     &PersistentModularityConfig::powerIterationTolerance,
                     "LeadingEigenvector: relative convergence tolerance of "
                     "the Rayleigh quotient.")
      .def_readwrite("kernighanLinRefinement",
                     &PersistentModularityConfig::kernighanLinRefinement,
                     "LeadingEigenvector: run a Kernighan-Lin local "
                     "refinement after each sign bisection.")
      .def_readwrite("resolutions", &PersistentModularityConfig::resolutions,
                     "Resolution parameters gamma, in scan order.")
      .def_readwrite("baseSeed", &PersistentModularityConfig::baseSeed,
                     "Base of the fixed restart seed sequence "
                     "(restart t uses splitmix64(baseSeed + t)).")
      .def_readwrite("restarts", &PersistentModularityConfig::restarts,
                     "Deterministic restarts per resolution; best exact "
                     "score kept, spread reported.")
      .def_readwrite("maxSweepsPerLevel",
                     &PersistentModularityConfig::maxSweepsPerLevel,
                     "Hard cap on local-move sweeps per aggregation level.")
      .def_readwrite("overlapThreshold",
                     &PersistentModularityConfig::overlapThreshold,
                     "Minimum support overlap for a persistence track to "
                     "continue across adjacent resolutions.");

  py::class_<ComponentRead>(m, "ComponentRead",
      "One discovered component: canonical id, level-0 cell support, cached "
      "sufficient statistics, and the exact per-component scores.")
      .def_readonly("id", &ComponentRead::id)
      .def_readonly("support", &ComponentRead::support,
                    "Level-0 member cell ids (ascending; a set — the order "
                    "carries no convention).")
      .def_readonly("internalWeight", &ComponentRead::internalWeight,
                    "Sigma_in: internal weight counting both directions.")
      .def_readonly("strength", &ComponentRead::strength,
                    "S_C: summed member strength.")
      .def_readonly("conductance", &ComponentRead::conductance,
                    R"doc(cut(C)/min(vol C, vol V\C); 0 when the denominator
vanishes.  NaN on a signed graph, where a community's strength is a
difference and there is no volume for the cut to be a fraction of -- left
unmeasured rather than computed by a formula that does not apply.)doc")
      .def_readonly("modularityContribution",
                    &ComponentRead::modularityContribution,
                    R"doc(This community's exact additive Q_gamma term, under
whichever null model scores the graph.  These sum over a level to that
level's exact Q_gamma either way.)doc");

  py::class_<RestartRead>(m, "RestartRead",
      "One deterministic restart: seed and exact best score.")
      .def_readonly("seed", &RestartRead::seed)
      .def_readonly("q", &RestartRead::q,
                    "Exact Q_gamma of this restart, complex and unreduced.")
      .def_readonly("objectiveValue", &RestartRead::objectiveValue,
                    "The real scalar this restart was ranked on.")
      .def_readonly("communities", &RestartRead::communities);

  py::class_<ResolutionSlice>(m, "ResolutionSlice",
      R"doc(Discovery result at one resolution gamma.  ``q`` is the exact
Q_gamma of the winning partition (cold recompute) — the best score across
deterministic restarts, a heuristic proposal, never the NP-hard global
optimum.  ``qIncremental`` is the accepted-delta-Q ledger and must agree
with ``q`` to double round-off.

``q`` is COMPLEX and unreduced: ``abs(q)`` is how much structure the
partition has, ``cmath.phase(q)`` is what KIND — 0 a community, pi an
ANTI-community, +-pi/2 lightlike cohesion, anything else mixed.  It is
exactly real on a real graph.  ``objectiveValue`` is the real scalar the
search actually maximized and ``objective`` says which functional that
was.)doc")
      .def_readonly("gamma", &ResolutionSlice::gamma)
      .def_readonly("q", &ResolutionSlice::q)
      .def_readonly("qIncremental", &ResolutionSlice::qIncremental)
      .def_readonly("objectiveValue", &ResolutionSlice::objectiveValue,
                    "The real scalar the search maximized: q.real under "
                    "Score, abs(q) under Magnitude.")
      .def_readonly("objective", &ResolutionSlice::objective,
                    "Which functional that was.  Reported rather than "
                    "assumed: a complex graph selects Magnitude whatever "
                    "the config asked for, since Score is not an ordering "
                    "there.")
      .def_readonly("levels", &ResolutionSlice::levels)
      .def_readonly("components", &ResolutionSlice::components,
                    "Final-level components, ordered by canonical hash.")
      .def_readonly("hierarchy", &ResolutionSlice::hierarchy,
                    "hierarchy[k] = communities at aggregation level k+1.")
      .def_readonly("restarts", &ResolutionSlice::restarts)
      .def_readonly("restartSpread", &ResolutionSlice::restartSpread,
                    "max - min of the restart scores (honest heuristic "
                    "uncertainty).  NaN under LeadingEigenvector, which has "
                    "no restarts — unmeasured is never encoded as zero.")
      .def_readonly("strategy", &ResolutionSlice::strategy,
                    "Which search produced this slice.")
      .def_readonly("splits", &ResolutionSlice::splits,
                    "LeadingEigenvector: one SplitRead per attempted "
                    "bisection, in the order attempted — the strategy's "
                    "spectral certificate.  Empty for MultilevelAggregation.");

  py::class_<ComponentMatch>(m, "ComponentMatch",
      R"doc(Matched component pair across adjacent resolutions or cobordism
time.  ``projectorOverlap`` is the documented spectral-projector hook: None
(unknown) until a later ticket supplies projectors and a hook is installed;
unknown is never encoded as zero.)doc")
      .def_readonly("fromId", &ComponentMatch::from)
      .def_readonly("toId", &ComponentMatch::to)
      .def_readonly("fromIndex", &ComponentMatch::fromIndex)
      .def_readonly("toIndex", &ComponentMatch::toIndex)
      .def_readonly("supportOverlap", &ComponentMatch::supportOverlap,
                    "Jaccard overlap of level-0 cell supports.")
      .def_readonly("projectorOverlap", &ComponentMatch::projectorOverlap);

  py::class_<PersistenceTrack>(m, "PersistenceTrack",
      R"doc(A component followed across the resolution scan by maximum
support overlap.  Lifetime/overlap/conductance are proposal diagnostics
only: they neither accept nor veto a fiber.  ``weightAwareStatus`` is the
downstream weight-aware gap/localization/persistence status — None until
the later weight-aware certificate tickets populate it (unknown is never
encoded as zero).)doc")
      .def_readonly("members", &PersistenceTrack::members)
      .def_readonly("memberIndices", &PersistenceTrack::memberIndices)
      .def_readonly("firstSlice", &PersistenceTrack::firstSlice)
      .def_readonly("lastSlice", &PersistenceTrack::lastSlice)
      .def_readonly("gammaFirst", &PersistenceTrack::gammaFirst)
      .def_readonly("gammaLast", &PersistenceTrack::gammaLast)
      .def_readonly("minAdjacentOverlap",
                    &PersistenceTrack::minAdjacentOverlap)
      .def_readonly("meanConductance", &PersistenceTrack::meanConductance)
      .def_property_readonly("weightAwareStatus",
           [](const PersistenceTrack &t) {
             return recordToPython(t.weightAwareStatus);
           });

  py::class_<FrameTrack>(m, "FrameTrack",
      R"doc(A component followed across COBORDISM FRAMES by maximum support
overlap.  ``frames`` is the lifetime the whitepaper's fiber-acceptance
conjunct names ("lifetime across multiple cobordism frames") -- a different
quantity from :class:`PersistenceTrack`, which counts MODULARITY RESOLUTION
SLICES of a single frame.)doc")
      .def_readonly("members", &FrameTrack::members)
      .def_readonly("memberIndices", &FrameTrack::memberIndices)
      .def_readonly("firstFrame", &FrameTrack::firstFrame)
      .def_readonly("lastFrame", &FrameTrack::lastFrame)
      .def_readonly("minAdjacentOverlap", &FrameTrack::minAdjacentOverlap)
      .def_property_readonly("frames", &FrameTrack::frames,
                             "Consecutive cobordism frames covered.");

  py::class_<ScanReport>(m, "ScanReport",
      "The full resolution-scan report: slices, adjacent-slice matches, and "
      "persistence tracks.")
      .def_readonly("slices", &ScanReport::slices)
      .def_readonly("matches", &ScanReport::matches)
      .def_readonly("tracks", &ScanReport::tracks);

  py::class_<InvalidationRead>(m, "InvalidationRead",
      "Components and tracks invalidated by a local change; positions "
      "(slice, level index, index in level) disambiguate automorphic twins "
      "that share a hash.")
      .def_readonly("components", &InvalidationRead::components)
      .def_readonly("positions", &InvalidationRead::positions)
      .def_readonly("tracks", &InvalidationRead::tracks);

  py::class_<PersistentModularity> pm(m, "PersistentModularity",
      R"doc(Label-free discovery of modular components that persist across
resolution and cobordism time (ticket #765).

Exact identities on the nonnegative weighted undirected similarity graph:
generalized modularity Q_gamma(P) = (1/2m) sum_ij (A_ij - gamma k_i k_j/2m)
[c_i = c_j], evaluated from per-community sufficient statistics, and the
exact O(deg v) cached local move gain
dQ(v: a->b) = (w_vb - w_va)/m - gamma k_v (k_v + S_b - S_a)/(2 m^2), so one
sparse sweep is near O(|E|).  Incremental accumulations are tested against
cold recomputation at double round-off.

Heuristic status: global modularity maximization is NP-hard; discovery is a
deterministic multilevel aggregation from a fixed seed sequence with the
restart spread reported honestly.  The score is blind to signed/complex
Hodge weights.  Modularity is a heuristic proposal generator only: it never
enters the emergence objective and may not veto an otherwise certified
fiber (acceptance belongs to the independent weight-aware certificates,
which later tickets supply; unknown is reported as None, never zero).

Read-only: never calls a solver, never mutates the spacetime it reads.)doc");

  py::enum_<PersistentModularity::WeightMap>(pm, "WeightMap",
      "Documented monotone map from complex edge magnitude to similarity.")
      .value("Unit", PersistentModularity::WeightMap::Unit,
             "w = 1: the combinatorial one-skeleton, exactly the legacy "
             "Newman-Girvan graph.")
      .value("ExpNegAbsLength",
             PersistentModularity::WeightMap::ExpNegAbsLength,
             "w = exp(-|l|): monotone decreasing in the complex edge "
             "magnitude (the mutual-information convention l = -log I).  "
             "Causally blind: a timelike and a spacelike edge of equal "
             "magnitude give the identical weight.")
      .value("CausalPhaseExpNegAbsLength",
             PersistentModularity::WeightMap::CausalPhaseExpNegAbsLength,
             "w = exp(-|l|) exp(i arg(l^2)): the same similarity MAGNITUDE "
             "as ExpNegAbsLength, carrying the edge's causal character as "
             "its ARGUMENT.  Spacelike lands on the positive real axis, "
             "timelike on the negative, lightlike on +-i, and a generic "
             "argument stays where it is -- nothing is bucketed, so there "
             "is no indefinite case to refuse.");

  py::enum_<ModularityObjective>(m, "ModularityObjective",
      R"doc(Which real functional of the complex Q the search maximizes.
Both readings are always REPORTED; this chooses only what is pursued.)doc")
      .value("Score", ModularityObjective::Score,
             "Maximize Q itself.  Available only where Q is real; the "
             "default, so existing behaviour does not move.  Finds "
             "community structure and passes over anti-communities, which "
             "score below the one-community partition.")
      .value("Magnitude", ModularityObjective::Magnitude,
             "Maximize |Q|.  Always available, and the only ordered choice "
             "once A is genuinely complex.  Finds community AND "
             "anti-community structure, with arg(Q) saying which.");

  py::class_<PersistentModularity::CausalWeightRead>(pm, "CausalWeightRead",
      R"doc(Whether the causal weight map can be read off a spacetime, with
the census that decides it.  `available` is False exactly when some edge has
no definite causal character, or when no edge carries a nonzero Lorentzian
interval; `reason` then names which, and is empty when available.)doc")
      .def_readonly("available",
                    &PersistentModularity::CausalWeightRead::available)
      .def_readonly("reason", &PersistentModularity::CausalWeightRead::reason)
      .def_readonly("spacelike",
                    &PersistentModularity::CausalWeightRead::spacelike)
      .def_readonly("timelike",
                    &PersistentModularity::CausalWeightRead::timelike)
      .def_readonly("lightlike",
                    &PersistentModularity::CausalWeightRead::lightlike)
      .def_readonly("mixed", &PersistentModularity::CausalWeightRead::mixed,
                    "Edges with a generic arg(l^2).  ORDINARY edges for the "
                    "complex weight map, which carries their argument as it "
                    "stands; a diagnostic, not a gate.")
      .def_readonly("degenerate",
                    &PersistentModularity::CausalWeightRead::degenerate);

  pm.def_static("causalWeightAvailability",
                [](const std::shared_ptr<Spacetime> &st) {
                  return PersistentModularity::causalWeightAvailability(*st);
                },
                py::arg("spacetime"),
                R"doc(Census of the one-skeleton's causal characters and
whether CausalExpNegAbsLength can be read from it.  Read-only; asks the
same Edge.disposition() classifier the weight map uses, so a True here is
exactly the condition under which fromSpacetime will not raise.)doc")
      .def_static("fromWeightedEdges",
                &PersistentModularity::fromWeightedEdges,
                py::arg("src"), py::arg("tgt"), py::arg("weight"),
                py::arg("isolatedCells") = std::vector<std::uint64_t>{},
                R"doc(Build from an explicit REAL weighted edge list, signed
or not (cells are arbitrary 64-bit ids; parallel edges consolidate by weight
summation; self-loops are ignored, as are edges whose consolidated weight is
zero -- a measured absence of net similarity).  Raises ValueError on
non-finite weights or mismatched lengths.

A negative weight anywhere switches the score to the signed null model; a
wholly nonnegative edge list scores bit-identically to before that branch
existed.)doc")
      .def_static("fromComplexWeightedEdges",
                &PersistentModularity::fromComplexWeightedEdges,
                py::arg("src"), py::arg("tgt"), py::arg("weight"),
                py::arg("isolatedCells") = std::vector<std::uint64_t>{},
                R"doc(Build from an explicit COMPLEX weighted edge list.
Consolidation, self-loops and the cancel-to-zero convention are as for
fromWeightedEdges; both components must be finite.  A list that happens to
be real takes the real path and scores exactly as fromWeightedEdges would.

The adjacency is complex SYMMETRIC -- A_ij = A_ji, no conjugation -- because
a weight is a property of the EDGE, and its magnitude and argument do not
depend on which end you read it from.)doc")
      .def_static("fromSpacetime",
                  [](const std::shared_ptr<Spacetime> &st,
                     PersistentModularity::WeightMap map) {
                    return PersistentModularity::fromSpacetime(*st, map);
                  },
                  py::arg("spacetime"),
                  py::arg("map") =
                      PersistentModularity::WeightMap::ExpNegAbsLength,
                  R"doc(Build the similarity graph from the spacetime
one-skeleton (read-only).  With CausalExpNegAbsLength this raises
ValueError, naming the reason, when causalWeightAvailability() reports the
map unreadable -- an unreadable causal character is refused, never
substituted with a magnitude.)doc")
      .def("nCells", &PersistentModularity::nCells)
      .def("nEdges", &PersistentModularity::nEdges)
      .def("isComplex", &PersistentModularity::isComplex,
           R"doc(True when some edge weight has a nonzero imaginary part, so
Q is genuinely complex and Score is not an ordering.  A property of the
graph, not a setting.)doc")
      .def("isSigned", &PersistentModularity::isSigned,
           R"doc(True when some edge weight is negative, so the score uses
the signed (rank-two) null model.  A property of the graph, not a setting.)doc")
      .def("totalWeight2", &PersistentModularity::totalWeight2,
           R"doc(Total adjacency weight: sum_ij A_ij on a nonnegative graph,
and T = 2m+ + 2m- (the total ABSOLUTE weight, which is the signed branch's
normalizer) when isSigned().)doc")
      .def("totalWeightSum", &PersistentModularity::totalWeightSum,
           R"doc(SA = sum_ij A_ij, the COMPLEX total the configuration null
model redistributes.  Equal to totalWeight2() on a nonnegative graph.  A
vanishing SA leaves the null model undefined and is refused by name.)doc")
      .def("cellIds", &PersistentModularity::cellIds,
           "Cell ids in internal storage order (no convention).")
      .def("modularityGamma", &PersistentModularity::modularityGamma,
           py::arg("labels"), py::arg("gamma"),
           R"doc(Exact generalized modularity Q_gamma of a fixed partition
(labels[i] labels cellIds()[i]).  The fixed-partition entry point: at
gamma = 1 on a Unit-weight graph this is exactly the Newman-Girvan score.)doc")
      .def("discover",
           [](const PersistentModularity &self, double gamma,
              const PersistentModularityConfig &cfg) {
             py::gil_scoped_release release;
             return self.discover(gamma, cfg);
           },
           py::arg("gamma"), py::arg("config"),
           "Deterministic label-free discovery at one resolution.")
      .def("scanResolutions",
           [](const PersistentModularity &self,
              const PersistentModularityConfig &cfg) {
             py::gil_scoped_release release;
             return self.scanResolutions(cfg);
           },
           py::arg("config"),
           "The configurable resolution-sequence scan with persistence "
           "tracks.")
      .def("matchComponents", &PersistentModularity::matchComponents,
           py::arg("a"), py::arg("b"),
           R"doc(Match components across resolution or cobordism time by
simplex-support overlap (Jaccard on level-0 cell ids over a common cell-id
universe).  When a projector-overlap hook is installed its value is
reported per match; matching decisions remain support-based until a later
ticket supplies the projectors.)doc")
      .def("trackAcrossFrames",
           [](const PersistentModularity &self,
              const std::vector<std::vector<ComponentRead>> &frames,
              double overlapThreshold) {
             py::gil_scoped_release release;
             return self.trackAcrossFrames(frames, overlapThreshold);
           },
           py::arg("frames"), py::arg("overlapThreshold") = 0.5,
           R"doc(Follow components across COBORDISM FRAMES: frames[t] is the
component list read from frame t over a common cell-id universe.  Chains
consecutive frames with matchComponents by best support overlap, exactly the
rule scanResolutions chains resolution slices with.  This is the supplier of
the whitepaper's "lifetime across multiple cobordism frames"; a component
seen in one frame gets a one-frame track, which is a measured fact and not a
structural artifact of reading a single resolution.)doc")
      .def("setProjectorOverlapHook",
           [](PersistentModularity &self, py::object hook) {
             if (hook.is_none()) {
               self.setProjectorOverlapHook(nullptr);
               return;
             }
             self.setProjectorOverlapHook(
                 [hook](const ComponentId &a, const ComponentId &b) {
                   py::gil_scoped_acquire acquire;
                   return hook(a, b).cast<double>();
                 });
           },
           py::arg("hook"),
           "Install (or clear with None) the documented spectral-projector "
           "overlap hook: hook(fromId, toId) -> float in [0, 1].  This "
           "ticket only plumbs the hook; a later ticket supplies the "
           "projectors.")
      .def_static("invalidatedAncestry",
                  &PersistentModularity::invalidatedAncestry,
                  py::arg("report"), py::arg("touchedCells"),
                  R"doc(Components (at every hierarchy level of every slice)
whose support intersects the touched level-0 cells, plus the affected
tracks.  Siblings with disjoint support remain valid.  Pure bookkeeping —
no recomputation.)doc");

  // ========================================
  // SpectralFiber (#769): localized spectral bands and their certificates
  // ========================================
  py::class_<SpectralFiberConfig>(m, "SpectralFiberConfig",
      "Configuration of the spectral-band detector/tracker (ticket #769). "
      "Thresholds select which bands are "
      "CERTIFIED, never which eigenvalues exist; no threshold is a "
      "Betti-number oracle and no rank is ever requested.")
      .def(py::init<>())
      .def_readwrite("degrees", &SpectralFiberConfig::degrees,
                     "Form degrees enumerated by enumerateOnComponents.")
      .def_readwrite("groupingTolerance",
                     &SpectralFiberConfig::groupingTolerance,
                     "Relative band-grouping width (fraction of the "
                     "spectral scale).")
      .def_readwrite("minRelativeGap", &SpectralFiberConfig::minRelativeGap,
                     "Isolation floor: certified bands need both gaps >= "
                     "minRelativeGap * scale (a closing gap returns an "
                     "uncertified band).")
      .def_readwrite("gapDominance", &SpectralFiberConfig::gapDominance,
                     "Certified gaps must exceed this multiple of the "
                     "in-band spread.")
      .def_readwrite("residualTolerance",
                     &SpectralFiberConfig::residualTolerance,
                     "Cap on the relative eigen/left/projector residuals.")
      .def_readwrite("gramDefectTolerance",
                     &SpectralFiberConfig::gramDefectTolerance,
                     "Cap on ||Phi^dagger W Phi - J||.")
      .def_readwrite("projectorNormCap",
                     &SpectralFiberConfig::projectorNormCap,
                     "Cap on the band projector norm ||P||_2.")
      .def_readwrite("maxLocalizationExcess",
                     &SpectralFiberConfig::maxLocalizationExcess,
                     "Cap on the band's rank-normalized localization excess "
                     "-- the localization acceptance conjunct.  1.0 accepts "
                     "any measured localization.")
      .def_readwrite("denseCrossover", &SpectralFiberConfig::denseCrossover,
                     "Dimension at/above which the self-adjoint path goes "
                     "sparse.")
      .def_readwrite("requestedEigenpairs",
                     &SpectralFiberConfig::requestedEigenpairs,
                     "Lowest eigenpairs the sparse block path computes.")
      .def_readwrite("oversample", &SpectralFiberConfig::oversample,
                     "Extra Ritz vectors beyond requestedEigenpairs.")
      .def_readwrite("maxSolverIterations",
                     &SpectralFiberConfig::maxSolverIterations)
      .def_readwrite("solverTolerance", &SpectralFiberConfig::solverTolerance)
      .def_readwrite("solverSeed", &SpectralFiberConfig::solverSeed,
                     "Seed of the deterministic sparse start block.")
      .def_readwrite("trackOverlapThreshold",
                     &SpectralFiberConfig::trackOverlapThreshold,
                     "Minimum subspace overlap for a certified track "
                     "continuation.")
      .def_readwrite("contourNodes", &SpectralFiberConfig::contourNodes,
                     "Whitney pencil path: trapezoidal node count of each band's Riesz contour.")
      .def_readwrite("isotropyTolerance", &SpectralFiberConfig::isotropyTolerance,
                     "Whitney pencil path: relative tolerance declaring a band's pairing isotropic.")
      .def_readwrite("crossValidateDense",
                     &SpectralFiberConfig::crossValidateDense,
                     "Cross-check solves below the crossover against the "
                     "independent DenseReference kernel and record the "
                     "deviation on the certificate.");

  py::class_<SpectralBandCertificate>(m, "SpectralBandCertificate",
      R"doc(Certification record of one whole spectral band: degree, rank,
lower/upper gap, localization (projector-diagonal inverse participation
ratio), projector/eigen/left residuals,
weighted Gram/signature defect ||Phi^dagger W Phi - J||, band condition
number ||P||_2, Krein inertia (p, q), frequency window, self-adjointness
flag, and the graded #764 Certificate (BandWindow domain; an uncertified
band carries HeuristicDiscovery, which never holds).

A degenerate band is one object of rank >= 2; an unexplained multiplicity
is reported exactly as its rank and never labeled.  Negative signature is
a certificate, never an automatic antiparticle identification.  Unmeasured
quantities are NaN, never zero.)doc")
      .def_readonly("degree", &SpectralBandCertificate::degree)
      .def_readonly("rank", &SpectralBandCertificate::rank)
      .def_readonly("lowerGap", &SpectralBandCertificate::lowerGap)
      .def_readonly("upperGap", &SpectralBandCertificate::upperGap)
      .def_readonly("nearestDiscardedSeparation",
                    &SpectralBandCertificate::nearestDiscardedSeparation,
                    "Distance in the complex plane to the nearest DISCARDED "
                    "eigenvalue -- the isolation acceptance conjunct.")
      .def_readonly("localization", &SpectralBandCertificate::localization)
      .def_readonly("localizationSupportFraction",
                    &SpectralBandCertificate::localizationSupportFraction,
                    "Effective support fraction n_eff/n in [rank/n, 1]; 1 "
                    "exactly for a perfectly delocalized band.")
      .def_readonly("localizationExcess",
                    &SpectralBandCertificate::localizationExcess,
                    "(n_eff - rank)/(n - rank) in [0, 1] -- the GATED "
                    "localization datum; 0 = as concentrated as the rank "
                    "permits, 1 = perfectly delocalized.")
      .def_readonly("projectorResidual",
                    &SpectralBandCertificate::projectorResidual)
      .def_readonly("eigenResidual", &SpectralBandCertificate::eigenResidual)
      .def_readonly("leftResidual", &SpectralBandCertificate::leftResidual)
      .def_readonly("gramDefect", &SpectralBandCertificate::gramDefect)
      .def_readonly("projectorNorm", &SpectralBandCertificate::projectorNorm,
                    "||P||_2, Kato's condition number of the spectral "
                    "projector (gauge-invariant).")
      .def_readonly("frameConditionNumber",
                    &SpectralBandCertificate::frameConditionNumber,
                    "The FRAME condition number: max Riesz conditioning of "
                    "the reported matched frames in the |W| metric.")
      .def_readonly("positiveSignature",
                    &SpectralBandCertificate::positiveSignature)
      .def_readonly("negativeSignature",
                    &SpectralBandCertificate::negativeSignature)
      .def_readonly("pairingDeterminant", &SpectralBandCertificate::pairingDeterminant,
                    "det B_C of the bilinear pairing (complex-symmetric pencil regime only).")
      .def_readonly("pairingCondition", &SpectralBandCertificate::pairingCondition)
      .def_readonly("pairingScale", &SpectralBandCertificate::pairingScale)
      .def_readonly("isotropic", &SpectralBandCertificate::isotropic,
                    "det B_C = 0: the exceptional-point indicator; no left frame.")
      .def_readonly("leftFrameRefusal", &SpectralBandCertificate::leftFrameRefusal)
      .def_readonly("metricSymmetryDefect", &SpectralBandCertificate::metricSymmetryDefect,
                    "The regime's verification residual, M L = (M L)^T.")
      .def_readonly("frequencyLower",
                    &SpectralBandCertificate::frequencyLower)
      .def_readonly("frequencyUpper",
                    &SpectralBandCertificate::frequencyUpper)
      .def_readonly("selfAdjoint", &SpectralBandCertificate::selfAdjoint)
      .def_readonly("accepted", &SpectralBandCertificate::accepted)
      .def_readonly("certificate", &SpectralBandCertificate::certificate)
      .def("describe", &SpectralBandCertificate::describe)
      .def("__repr__", &SpectralBandCertificate::describe);

  py::class_<FiberOverlapRead>(m, "FiberOverlapRead",
      "Principal-angle / support comparison of two fibers: cells matched "
      "by sorted vertex-id tuple (gauge- and relabeling-invariant).")
      .def_readonly("supportOverlap", &FiberOverlapRead::supportOverlap)
      .def_readonly("sharedCells", &FiberOverlapRead::sharedCells)
      .def_readonly("principalAngles", &FiberOverlapRead::principalAngles)
      .def_readonly("subspaceOverlap", &FiberOverlapRead::subspaceOverlap);

  py::class_<SpectralFiber>(m, "SpectralFiber",
      R"doc(One whole isolated spectral band of a component-restricted Hodge
operator: right/left frames, band projector
P = Phi Psi^dagger W with Psi^dagger W Phi = I, eigenvalues, and the
SpectralBandCertificate.  The band is represented by its PROJECTOR —
individual eigenvectors are a gauge choice and never determine an identity
or a downstream observable.)doc")
      .def("degree", &SpectralFiber::degree)
      .def("rank", &SpectralFiber::rank)
      .def("accepted", &SpectralFiber::accepted)
      .def("rightFrame", &SpectralFiber::rightFrame,
           "Right frame Phi (cells x rank).")
      .def("leftFrame", &SpectralFiber::leftFrame,
           "Left frame Psi (cells x rank), Psi^dagger W Phi = I.")
      .def("projector", &SpectralFiber::projector,
           "The band projector P = Phi Psi^dagger W (cells x cells).")
      .def("weightDiagonal", &SpectralFiber::weightDiagonal,
           "Diagonal inner-product weights W restricted to the band's "
           "cells.")
      .def("eigenvalues", &SpectralFiber::eigenvalues,
           "Band eigenvalues (with multiplicity), sorted by (Re, Im).")
      .def("bandCenter", &SpectralFiber::bandCenter)
      .def("cellVertices", &SpectralFiber::cellVertices,
           "The k-cells carrying the band, as sorted vertex-id tuples in "
           "frame row order.")
      .def("certificate", &SpectralFiber::certificate,
           py::return_value_policy::copy)
      .def_static("overlap", &SpectralFiber::overlap, py::arg("a"),
                  py::arg("b"),
                  "Principal-angle / support comparison (cells matched by "
                  "vertex-id tuple).")
      .def("toRecord",
           [](const SpectralFiber &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the JSON-able record of the fiber "
           "(schema-versioned; complex leaves split _re/_im).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return SpectralFiber::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate from toRecord() output; rejects an unknown "
                  "schema_version (ValueError).");

  py::class_<SpectralBandWindow>(m, "SpectralBandWindow",
      "An accepted band's frequency window as PLAIN DATA for the response "
      "consumer (#768): lower/upper frequency bounds plus the band "
      "certificate.  Carries no operator, frame, or quotient reference.")
      .def_readonly("degree", &SpectralBandWindow::degree)
      .def_readonly("rank", &SpectralBandWindow::rank)
      .def_readonly("frequencyLower", &SpectralBandWindow::frequencyLower)
      .def_readonly("frequencyUpper", &SpectralBandWindow::frequencyUpper)
      .def_readonly("certificate", &SpectralBandWindow::certificate);

  py::class_<FiberMatchRead>(m, "FiberMatchRead",
      "One matched fiber pair across frames or resolutions.  A certified "
      "continuation needs both endpoint bands accepted, equal ranks, and "
      "subspace overlap above the threshold — an endpoint whose gap closed "
      "is reported but never certified (no discontinuous identity flip).")
      .def_readonly("fromIndex", &FiberMatchRead::fromIndex)
      .def_readonly("toIndex", &FiberMatchRead::toIndex)
      .def_readonly("degree", &FiberMatchRead::degree)
      .def_readonly("overlap", &FiberMatchRead::overlap)
      .def_readonly("ranksEqual", &FiberMatchRead::ranksEqual)
      .def_readonly("certifiedContinuation",
                    &FiberMatchRead::certifiedContinuation);

  py::class_<ComponentBandRead>(m, "ComponentBandRead",
      "The band enumeration of one (component, degree) pair: the restricted "
      "operator's cells, verified regime, solver path, covered eigenvalues, "
      "every enumerated band (certified or not), and the solve certificate.")
      .def_readonly("support", &ComponentBandRead::support)
      .def_readonly("degree", &ComponentBandRead::degree)
      .def_readonly("dimension", &ComponentBandRead::dimension)
      .def_readonly("cellVertices", &ComponentBandRead::cellVertices)
      .def_readonly("regime", &ComponentBandRead::regime)
      .def_readonly("solverPath", &ComponentBandRead::solverPath)
      .def_readonly("truncated", &ComponentBandRead::truncated)
      .def_readonly("coveredEigenvalues",
                    &ComponentBandRead::coveredEigenvalues)
      .def_readonly("fibers", &ComponentBandRead::fibers)
      .def_readonly("solveCertificate", &ComponentBandRead::solveCertificate)
      .def("toRecord",
           [](const ComponentBandRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization of the whole read (fibers included).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return ComponentBandRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version "
                  "(ValueError).");

  py::class_<SpectralFiberTracker>(m, "SpectralFiberTracker",
      R"doc(Extraction and tracking of whole isolated localized Hodge bands
on persistent components (ticket #769).

Identity: for a component support S the tracker assembles the weighted
Hodge operator of the full induced subcomplex on S (the same boundary maps,
canonical cell order, and diagonal inner-product weights as the
whole-complex HodgeLaplacian, consumed read-only), so support = all
vertices reproduces HodgeLaplacian.laplacian(k) entry for entry.  Regimes
are VERIFIED, never assumed: positive -> self-adjoint solves (exact dense
below the crossover, deterministic sparse block shift-invert at/above);
real signed weights -> W-self-adjointness verified, Krein inertia of
Phi^dagger W Phi recorded and normalized to diag(I_p, -I_q); complex
weights -> matched biorthogonal right/left subspaces with
Psi^dagger W Phi = I.  Bands are grouped by a relative gap rule and every
band is reported with its projector and certificate; a closing gap yields
an uncertified band, never a different identity.  The detector never
requests rank three and no eigenvalue threshold is a Betti oracle.

Read-only observable: never calls a solver on the spacetime, never mutates
it, and nothing here enters any emergence objective.)doc")
      .def(py::init([](std::shared_ptr<Spacetime> st, const SpectralFiberConfig &cfg,
                       cobordism::HodgeLaplacian::MetricSource source) {
             return SpectralFiberTracker(std::move(st), cfg, source);
           }),
           py::arg("spacetime"), py::arg("config"), py::arg("metric_source"),
           "Bind with an explicit metric source; WhitneyPencil reads every degree >= 1 on the "
           "chain-level Whitney pencil in the complex-symmetric-pencil regime.")
      .def(py::init([](std::shared_ptr<Spacetime> st,
                       const SpectralFiberConfig &cfg,
                       const py::object &weights) {
             const auto convention =
                 weights.is_none()
                     ? tessera::cobordism::HodgeLaplacian::
                           defaultWeightConvention()
                     : weights
                           .cast<tessera::cobordism::HodgeLaplacian::
                                     WeightConvention>();
             return SpectralFiberTracker(std::move(st), cfg, convention);
           }),
           py::arg("spacetime"), py::arg("config") = SpectralFiberConfig{},
           py::arg("weights") = py::none(),
           "Bind to the spacetime to read; weights=None follows the "
           "process-wide HodgeLaplacian.defaultWeightConvention() at call "
           "time.")
      .def("metricSource", &SpectralFiberTracker::metricSource,
           "Where this tracker's operators take their metric from.")
      .def("config", &SpectralFiberTracker::config,
           py::return_value_policy::copy)
      .def("weightConvention", &SpectralFiberTracker::weightConvention)
      .def("enumerateBands",
           [](const SpectralFiberTracker &self,
              const std::vector<std::uint64_t> &support, int degree) {
             py::gil_scoped_release release;
             return self.enumerateBands(support, degree);
           },
           py::arg("support"), py::arg("degree"),
           "Enumerate the bands of one component (vertex-id support) at "
           "one form degree.")
      .def("enumerateOnComponents",
           [](const SpectralFiberTracker &self,
              const std::vector<ComponentRead> &components) {
             py::gil_scoped_release release;
             return self.enumerateOnComponents(components);
           },
           py::arg("components"),
           "Enumerate every configured degree on every #765 component.")
      .def("enumerateBandsCached",
           [](const SpectralFiberTracker &self,
              tessera::cobordism::AnalyticCache &cache,
              const std::vector<std::uint64_t> &support, int degree) {
             py::gil_scoped_release release;
             return self.enumerateBandsCached(cache, support, degree);
           },
           py::arg("cache"), py::arg("support"), py::arg("degree"),
           "enumerateBands through the #764 AnalyticCache contract "
           "(touched-star invalidation; served while the component is "
           "untouched).")
      .def_static("acceptedWindows", &SpectralFiberTracker::acceptedWindows,
                  py::arg("reads"),
                  "The accepted bands' frequency windows as plain data for "
                  "the response consumer (#768).")
      .def_static("matchFibers", &SpectralFiberTracker::matchFibers,
                  py::arg("fromFibers"), py::arg("toFibers"),
                  py::arg("overlapThreshold") = 0.5,
                  "Track fibers across frames/resolutions by principal "
                  "angles and component overlap.")
      .def_readonly_static("CACHE_KIND", &SpectralFiberTracker::kCacheKind);

  // ========================================
  // ModularityOptimizer
  // ========================================
  py::class_<ModularityMeasurement>(m, "ModularityMeasurement",
      R"doc(One recorded point on the (Q, D_S) trajectory.

Mirrors examples/modularity.py:Measurement.)doc")
      .def_readonly("Q", &ModularityMeasurement::Q)
      .def_readonly("dsSmall", &ModularityMeasurement::dsSmall)
      .def_readonly("dsLarge", &ModularityMeasurement::dsLarge)
      .def_readonly("nVertices", &ModularityMeasurement::nVertices)
      .def_readonly("nEdges", &ModularityMeasurement::nEdges)
      .def_readonly("nSimplices", &ModularityMeasurement::nSimplices)
      .def_readonly("iter", &ModularityMeasurement::iter)
      .def_readonly("direction", &ModularityMeasurement::direction);

  py::class_<ModularityOptimizerConfig>(m, "ModularityOptimizerConfig")
      .def(py::init<>())
      .def_readwrite("targetDq", &ModularityOptimizerConfig::targetDq)
      .def_readwrite("maxIterations",
                     &ModularityOptimizerConfig::maxIterations)
      .def_readwrite("nDiffusionWalks",
                     &ModularityOptimizerConfig::nDiffusionWalks)
      .def_readwrite("maxSigma", &ModularityOptimizerConfig::maxSigma)
      .def_readwrite("negativeRetryMax",
                     &ModularityOptimizerConfig::negativeRetryMax)
      .def_readwrite("epsilonQMax",
                     &ModularityOptimizerConfig::epsilonQMax)
      .def_readwrite("krylovDim", &ModularityOptimizerConfig::krylovDim)
      .def_readwrite("targetNModules",
                     &ModularityOptimizerConfig::targetNModules);

  py::class_<ModularityOptimizer>(m, "ModularityOptimizer",
      R"doc(Modularity sweep on a CDT spacetime, driven by transactional
Pachner moves with Q-direction acceptance.

Each iteration:
  1. Picks a random move type from {add, remove, flip, iflip, shift}.
  2. Calls cdt.proposeXxx() — if no eligible target, tries another type.
  3. Snapshots Q on the spacetime 1-skeleton.
  4. Calls move.apply() to commit the move.
  5. Computes new Q.  If direction matches, keeps the move; else
     calls move.rollback().
  6. If Q crossed the next target_dq threshold, builds the dual graph
     and measures D_S.

See docs/source/modularity-plan.md for the design rationale.)doc")
      .def(py::init<ModularityOptimizerConfig, std::uint64_t>(),
           py::arg("config"), py::arg("seed") = 0)
      .def("sweep",
           [](ModularityOptimizer &self, CDT &cdt,
              const std::string &direction, py::object progress) {
             ModularityOptimizer::ProgressCallback cb = nullptr;
             if (!progress.is_none()) {
               cb = [&progress](int it, int maxIt, double q,
                                std::size_t n) {
                 py::gil_scoped_acquire acquire;
                 progress(it, maxIt, q, n);
               };
             }
             py::gil_scoped_release release;
             return self.sweep(cdt, direction, cb);
           },
           py::arg("cdt"), py::arg("direction"),
           py::arg("progress") = py::none(),
           "Drive `cdt` to walk Q in direction ('up' or 'down'). "
           "Returns list of ModularityMeasurement.")
      .def("getNAccepted", &ModularityOptimizer::getNAccepted,
           "Number of moves applied + kept (since the last sweep).")
      .def("getNRolledBack", &ModularityOptimizer::getNRolledBack,
           "Number of moves applied then rolled back.")
      .def("getNNoMove", &ModularityOptimizer::getNNoMove,
           "Number of iterations with no eligible move.")
      .def("getNMeasurements", &ModularityOptimizer::getNMeasurements,
           "Number of D_S measurements taken.")
      .def("discoverComponents",
           [](const ModularityOptimizer &self,
              const std::shared_ptr<Spacetime> &st,
              const PersistentModularityConfig &cfg,
              PersistentModularity::WeightMap map) {
             py::gil_scoped_release release;
             return self.discoverComponents(*st, cfg, map);
           },
           py::arg("spacetime"), py::arg("config"),
           py::arg("map") = PersistentModularity::WeightMap::ExpNegAbsLength,
           R"doc(Label-free discovery of persistent modular components on the
CURRENT spacetime one-skeleton (ticket #765).  Read-only: never mutates the
spacetime and never proposes moves.  Builds the nonnegative similarity graph
under ``map`` and runs PersistentModularity.scanResolutions(config).  A
heuristic proposal generator — blind to signed/complex Hodge weights, never
part of the emergence objective, and never a veto over a certified fiber.)doc");
  // ========================================
  // VolumeProfile
  // ========================================
  py::class_<VolumeProfile, std::shared_ptr<VolumeProfile> >(m, "VolumeProfile",
      R"doc(Observable measuring the spatial volume profile N(t).

Counts top simplices per time slice.  Can accumulate measurements over
multiple configurations for averaging.)doc")
      .def(py::init<>())
      .def("compute", &VolumeProfile::compute, py::arg("spacetime"),
           "Compute the volume profile for the current configuration.")
      .def("getProfile", &VolumeProfile::getProfile,
           "Return the most recent volume profile as a list.")
      .def("getAverageProfile", &VolumeProfile::getAverageProfile,
           "Return the time-averaged volume profile (over all measure() calls).")
      .def("getCenteredAverageProfile",
           &VolumeProfile::getCenteredAverageProfile,
           py::arg("subtractStalk") = false,
           py::arg("normalizePeak") = false,
           "Peak-centered average of all measure() calls (see centeredAverage).")
      .def_static("centeredAverage", &VolumeProfile::centeredAverage,
           py::arg("profiles"),
           py::arg("subtractStalk") = false,
           py::arg("normalizePeak") = false,
           R"doc(Peak-centered average of a set of volume profiles.

Each profile is zero-padded to the longest length and circularly rolled so
its peak sits at T//2 before the bin-wise mean is taken, preventing the de
Sitter blob from smearing when its position fluctuates along the chain
(Ambjorn, Jurkiewicz, Loll, 2005).

Args:
    profiles: Per-configuration volume profiles (lists of counts).
    subtractStalk: Subtract each padded profile's minimum before centering.
    normalizePeak: Rescale the result so its peak equals 1.)doc")
      .def("measure", &VolumeProfile::measure, py::arg("spacetime"),
           "Compute and accumulate a volume profile measurement for averaging.")
      .def("reset", &VolumeProfile::reset,
           "Reset the accumulated measurements.");
  // ========================================
  // Simplicial qubit (#955): docs/design/simplicial_qubit_spec.md, section 14 API;
  // complex geometry and pure-gauge link phases (#976): section 16
  // ========================================
  {
    auto emitWarnings = [](const SimplicialQubit &q) {
      if (q.warnings().empty()) return;
      std::string text;
      for (const auto &w : q.warnings()) {
        if (!text.empty()) text += " | ";
        text += w;
      }
      py::module_::import("warnings").attr("warn")(text);
    };
    // On the real locus (real lengths, no phases) every stored matrix is real
    // and the construction is bit-identical to the real-length one: hand the
    // real part back (exact); off it the complex matrix.
    auto realOr = [](const SimplicialQubit &q, const Eigen::MatrixXcd &m) -> py::object {
      if (q.onRealLocus()) return py::cast(Eigen::MatrixXd(m.real()));
      return py::cast(Eigen::MatrixXcd(m));
    };
    auto realOrVector = [](const SimplicialQubit &q, const Eigen::VectorXcd &v) -> py::object {
      if (q.onRealLocus()) return py::cast(Eigen::VectorXd(v.real()));
      return py::cast(Eigen::VectorXcd(v));
    };
    auto realOrList = [](const SimplicialQubit &q, const std::vector<std::complex<double>> &v) -> py::object {
      py::list out;
      for (const auto &z : v) {
        if (q.onRealLocus())
          out.append(z.real());
        else
          out.append(z);
      }
      return out;
    };
    py::class_<SimplicialQubit>(m, "SimplicialQubit",
        "A single qubit state encoded as the holomorphic line in the harmonic space of the "
        "metric Hodge Laplacian on a triangulated torus (docs/design/simplicial_qubit_spec.md). "
        "Input: vertices 0..nV-1, edges (i, j) with i < j, consistently oriented faces (i, j, k), "
        "edge lengths (real positive, or complex per section 16), and marked cycles A, B as "
        "(edge_index, sign) lists with A.B = +1. Computed on load: the section-2 validations; d0, d1; "
        "per-face angles, Heron areas and local layouts; cotangent weights M1 = diag(w_e) (negative "
        "weights and Delaunay violations flagged on the real locus); the harmonic space "
        "H = null_space([d1; d0.T M1]) of dimension 2; the Whitney-form L2 inner product at barycenters; "
        "the complex structure J = G^{-1} R.T by rotate-then-project with its residual ||J J + I||_F; the "
        "holomorphic line, periods, tau = P_B / P_A (in the upper half plane on the real locus); the "
        "state, Bloch vector and density matrix; and the section-13 degeneration warnings. The optional "
        "intrinsic Delaunay edge-flip pass is intrinsic_delaunay(). Alternatively read a Spacetime of "
        "dimension 2 directly (its faces are oriented by the fundamental class; reversed=True selects "
        "the other hemisphere; its edge phases are the pure-gauge link connection of section 16).\n\n"
        "Section 16 (complex geometry): off the REAL LOCUS (real lengths, no phases) every formula of "
        "sections 4-9 runs over C. The real reference is the same complex with every squared length 1 "
        "(the unit equilateral reference simplex of chainhodge.WhitneyMass); angles are the principal "
        "acos, Heron areas the continuation branch of sqrt(det G)/2 (WhitneyMass.volumeOnBranch) along "
        "the straight segment in the squared lengths from the reference, pairings are the transpose "
        "(bilinear) pairing, the harmonic space a complex null space, and the eigenline of section 9 is "
        "chosen by continuity from the reference (the Im tau > 0 rule holds on the real locus only). Link "
        "phases must be a pure gauge (flux or holonomy is refused by name): they twist the incidences and "
        "the Whitney pairing (against the kernel of the inverse links, dual_harmonic_basis()) and the "
        "periods are taken with parallel transport along the marked cycles from one base point "
        "(base_vertex()), which leaves tau, the state and the coefficient pairs in the period frame "
        "invariant. On the real locus the matrix-valued reads return real arrays, bit-identical to the "
        "real-length construction; off it complex arrays.")
        .def(py::init([emitWarnings](std::vector<std::uint64_t> vertices,
                                     std::vector<SimplicialQubit::EdgePair> edges,
                                     std::vector<SimplicialQubit::Face> faces,
                                     std::vector<std::complex<double>> lengths, SimplicialQubit::Cycle cycle_A,
                                     SimplicialQubit::Cycle cycle_B, double degeneracy_threshold) {
               SimplicialQubit q(std::move(vertices), std::move(edges), std::move(faces),
                                 std::move(lengths), std::move(cycle_A), std::move(cycle_B),
                                 degeneracy_threshold);
               emitWarnings(q);
               return q;
             }),
             py::arg("vertices"), py::arg("edges"), py::arg("faces"), py::arg("lengths"),
             py::arg("cycle_A"), py::arg("cycle_B"), py::arg("degeneracy_threshold") = 1e8,
             "The section-2 / section-14 constructor; lengths real positive or complex (section 16), the "
             "trivial connection. Raises ValueError when a validation of section 2 or a continuation of "
             "section 16 fails and RuntimeError when dim H != 2 (section 6).")
        .def(py::init([emitWarnings](const std::shared_ptr<Spacetime> &spacetime,
                                     SimplicialQubit::Cycle cycle_A, SimplicialQubit::Cycle cycle_B,
                                     bool reversed, double degeneracy_threshold) {
               SimplicialQubit q(spacetime, std::move(cycle_A), std::move(cycle_B), reversed,
                                 degeneracy_threshold);
               emitWarnings(q);
               return q;
             }),
             py::arg("spacetime"), py::arg("cycle_A"), py::arg("cycle_B"), py::arg("reversed") = false,
             py::arg("degeneracy_threshold") = 1e8,
             "Read a Spacetime of dimension 2: vertices by ascending id, edges in ascending (i, j) "
             "order (the order the cycles index), faces oriented by the fundamental class "
             "(reversed flips them), lengths real positive or complex, edge phases as the pure-gauge "
             "link connection (flux or holonomy refused by name).")
        // ---- section 14
        .def("harmonic_basis", [realOr](const SimplicialQubit &q) { return realOr(q, q.harmonicBasis()); },
             "H: nE x 2 (section 6); real on the real locus.")
        .def("dual_harmonic_basis",
             [realOr](const SimplicialQubit &q) { return realOr(q, q.dualHarmonicBasis()); },
             "H^vee: nE x 2, the kernel twisted by the inverse links that the section-7/8 pairings are "
             "taken against (section 16); harmonic_basis() itself on the trivial connection.")
        .def("complex_structure", [realOr](const SimplicialQubit &q) { return realOr(q, q.complexStructure()); },
             "J: 2 x 2 (section 8); real on the real locus.")
        .def("j_residual", &SimplicialQubit::jResidual, "||J @ J + I||_F (section 8).")
        .def("holomorphic_form", &SimplicialQubit::holomorphicForm, "omega: nE complex (section 9).")
        .def("periods", &SimplicialQubit::periods,
             "(P_A, P_B) (section 9); transported from base_vertex() under a nontrivial connection.")
        .def("tau", &SimplicialQubit::tau, "P_B / P_A (section 9).")
        .def("tau_derivative", &SimplicialQubit::tauDerivative,
             "d tau / d z_e for every edge in edge order, z_e = l_e^2 (qubit cobordism spec D2): the "
             "holomorphic derivative of tau() with respect to each squared edge length, from the "
             "complex structure in the period frame and its dual (no eigen-decomposition and no "
             "null-space basis is differentiated); scale-free (sum_e z_e d tau/d z_e = 0) and gauge "
             "invariant. Raises RuntimeError when the two eigenlines of J coincide.")
        .def("intersection_number", &SimplicialQubit::intersectionNumber,
             "A . B of the marked cycles on the surface as oriented by the faces: the ordered cup "
             "product of the reference period frame on the fundamental cycle, +1 or -1 to rounding "
             "(+1 on flat_torus, -1 under reversed=True).")
        .def("period_frame", [realOr](const SimplicialQubit &q) { return realOr(q, q.periodFrame()); },
             "F: nE x 2 (real on the real locus), the period frame (qubit cobordism spec D3) - the basis "
             "(f_A, f_B) of the harmonic space with periods (1, 0) and (0, 1) over the marking in force, "
             "harmonic_basis() times the inverse period matrix, in the torus's edge order. The coordinates a "
             "state is written in: holomorphic_form() == period_frame() @ (P_A, P_B), i.e. P_A * F @ (1, tau), "
             "the qubit |0> + tau|1> read as a 1-form (f_A <-> |0>, f_B <-> |1>); the frame a two-body "
             "target chi compares with the transfer in (MultiCobordism.set_input_frame). Under a nontrivial "
             "connection the periods are the transported ones from base_vertex().")
        .def("state", &SimplicialQubit::state, "(|0> + tau|1>) / sqrt(1 + |tau|^2) (section 10).")
        .def("bloch", &SimplicialQubit::bloch, "The Bloch vector (section 10), a unit vector for every tau.")
        .def("density_matrix", &SimplicialQubit::densityMatrix, "rho = (I + r . sigma) / 2 (section 10).")
        .def_static("flat_torus",
                    [emitWarnings](std::complex<double> tau, int nx, int ny) {
                      SimplicialQubit q = SimplicialQubit::flatTorus(tau, nx, ny);
                      emitWarnings(q);
                      return q;
                    },
                    py::arg("tau"), py::arg("nx"), py::arg("ny"),
                    "The flat torus C / (Z + tau Z): unit cell spanned by 1 and tau as an nx x ny "
                    "grid split by its diagonals, sides identified, marked by the row loop A and "
                    "the column loop B (section 12). Exact: returns tau to rounding.")
        // ---- section 5: the optional preprocessing pass
        .def("intrinsic_delaunay",
             [emitWarnings](const SimplicialQubit &q) {
               SimplicialQubit r = q.intrinsicDelaunay();
               emitWarnings(r);
               return r;
             },
             "Flip every edge with alpha_e + beta_e > pi until none remains (marked cycles "
             "rerouted); returns the qubit of the flipped triangulation (section 5). Real locus only.")
        .def("delaunay_flip_count", &SimplicialQubit::delaunayFlipCount)
        // ---- inputs
        .def("vertices", &SimplicialQubit::vertices)
        .def("edges", &SimplicialQubit::edges, "E = [(i, j)], the order the cycles index.")
        .def("faces", &SimplicialQubit::faces, "F = [(i, j, k)], consistently oriented.")
        .def("lengths", [realOrList](const SimplicialQubit &q) { return realOrList(q, q.lengths()); },
             "The edge lengths in edge order (floats on the real locus, complex off it).")
        .def("links", &SimplicialQubit::links,
             "The links U_ij of the connection in edge order (all 1 on the trivial connection).")
        .def("canonical_edge_index", &SimplicialQubit::canonicalEdgeIndex, py::arg("e"),
             "The canonical (ChainComplex, lexicographic) index of edge e of edges(), the order "
             "connection().links() and Connection.transportedPeriod use.")
        .def("connection", &SimplicialQubit::connection, py::return_value_policy::reference_internal,
             "The chainhodge.Connection over the torus's chain complex (canonical edge order).")
        .def("cycle_A", &SimplicialQubit::cycleA)
        .def("cycle_B", &SimplicialQubit::cycleB)
        .def("walk_A", &SimplicialQubit::walkA,
             "Cycle A as a closed walk of directed vertex steps (section 16); starts at base_vertex() "
             "under a nontrivial connection.")
        .def("walk_B", &SimplicialQubit::walkB, "Cycle B as a closed walk (section 16).")
        .def("base_vertex", &SimplicialQubit::baseVertex,
             "The common base point of the transported periods: the first vertex of A's walk on B's.")
        .def("degeneracy_threshold", &SimplicialQubit::degeneracyThreshold)
        .def("spacetime", &SimplicialQubit::spacetime, "The Spacetime holding vertices, edges, lengths, phases.")
        .def("on_real_locus", &SimplicialQubit::onRealLocus,
             "True when every length is real and every link is 1: the real-number construction ran, "
             "bit-identical to the real-length one (section 16).")
        .def("trivial_connection", &SimplicialQubit::trivialConnection, "True when every link is 1.")
        // ---- intermediate quantities
        .def("d0", &SimplicialQubit::d0, "nE x nV (section 3).")
        .def("d1", &SimplicialQubit::d1, "nF x nE (section 3).")
        .def("angles", [realOr](const SimplicialQubit &q) { return realOr(q, q.angles()); },
             "Per face (alpha_i, alpha_j, alpha_k) (section 4); the principal acos off the real locus.")
        .def("areas", [realOrVector](const SimplicialQubit &q) { return realOrVector(q, q.areas()); },
             "Per face the Heron area (section 4); the continuation branch off the real locus.")
        .def("layout", [realOr](const SimplicialQubit &q) { return realOr(q, q.layout()); },
             "Per face (p_i, p_j, p_k) in its local frame (section 4).")
        .def("barycentric_gradients",
             [realOr](const SimplicialQubit &q) { return realOr(q, q.barycentricGradients()); },
             "Per face (grad lambda_i, grad lambda_j, grad lambda_k) (section 7).")
        .def("weights", [realOrVector](const SimplicialQubit &q) { return realOrVector(q, q.weights()); },
             "The cotangent weights w_e (section 5).")
        .def("negative_weight_edges", &SimplicialQubit::negativeWeightEdges)
        .def("non_delaunay_edges", &SimplicialQubit::nonDelaunayEdges,
             "Edges with alpha_e + beta_e > pi (section 5); evaluated on the real locus only.")
        .def("gram", [realOr](const SimplicialQubit &q) { return realOr(q, q.gram()); },
             "G[a][b] = <h_a, h_b> (section 8): the transpose pairing, between the dual kernel and the "
             "kernel under a nontrivial connection (section 16).")
        .def("rotation_pairing", [realOr](const SimplicialQubit &q) { return realOr(q, q.rotationPairing()); },
             "R[a][b] = sum_t A_t rot90(W_t(h_a)) . W_t(h_b) (section 8).")
        .def("marking_swapped", &SimplicialQubit::markingSwapped,
             "True when |P_A| vanished and -1/tau of the given marking is reported (section 9).")
        .def("condition_m1", &SimplicialQubit::conditionM1, "cond(M1) (section 13).")
        .def("condition_g", &SimplicialQubit::conditionG, "cond(G) (section 13).")
        .def("near_degenerate", &SimplicialQubit::nearDegenerate)
        .def("warnings", &SimplicialQubit::warnings, "Every warning the construction raised.");
    m.def("fubini_study_distance", &SimplicialQubit::fubiniStudyDistance, py::arg("q1"), py::arg("q2"),
          "arccos(|1 + conj(tau1) tau2| / sqrt((1+|tau1|^2)(1+|tau2|^2))): distinguishability, "
          "curvature +4 (section 11).");
    m.def("weil_petersson_distance", &SimplicialQubit::weilPeterssonDistance, py::arg("q1"), py::arg("q2"),
          "arccosh(1 + |tau1 - tau2|^2 / (2 Im tau1 Im tau2)): the moduli distance between shapes, "
          "curvature -1 (section 11).");
  }
  // ========================================
  // Hodge spectral Observables (#95): scalars over cobordism::HodgeLaplacian
  // ========================================
  py::class_<SpectralGap, std::shared_ptr<SpectralGap>>(m, "SpectralGap",
      "Observable: first spectral gap lambda_1 - lambda_0 of the Hermitian-"
      "weighted Hodge Laplacian (cobordism::HodgeLaplacian, k=0). The gauge-"
      "invariant interference signature: on the triangle it collapses from 3 to "
      "0 at flux Phi=pi.")
      .def(py::init<>())
      .def("compute", &SpectralGap::compute, py::arg("spacetime"));
  py::class_<HarmonicDimension, std::shared_ptr<HarmonicDimension>>(m, "HarmonicDimension",
      "Observable: dim ker L_0 (harmonic zero-mode count) of the Hermitian-"
      "weighted Hodge Laplacian. Equals b_0 at zero flux; a nonzero U(1) flux "
      "lifts the zero-mode, dropping it below the topological ChainComplex count.")
      .def(py::init<>())
      .def("compute", &HarmonicDimension::compute, py::arg("spacetime"));
  // ========================================
  // WilsonLoop
  // ========================================
  py::enum_<WilsonMode>(m, "WilsonMode",
      "Evaluation mode for Wilson loops.")
      .value("COMBINATORIAL", WilsonMode::COMBINATORIAL,
             "Dual-graph topology only (loop length, enclosed hinges).")
      .value("DEFICIT_ANGLE", WilsonMode::DEFICIT_ANGLE,
             "Deficit-angle based: W = ((d-2)+2cos(epsilon))/d.")
      .value("CAUSAL", WilsonMode::CAUSAL,
             "CDT causal orientation changes around the loop.")
      .value("U1_CONNECTION", WilsonMode::U1_CONNECTION,
             "U(1) connection holonomy: oriented sum of Edge.phase around a "
             "1-skeleton vertex cycle, reduced mod 2*pi. The Wilson-loop view "
             "of the Stage-1 cobordism.HodgeLaplacian cycle flux.");

  py::enum_<LoopType>(m, "LoopType",
      "Which loop-shape generator to use.")
      .value("HINGE", LoopType::HINGE,
             "Elementary loop around a (d-2)-simplex.")
      .value("DUAL_LATTICE", LoopType::DUAL_LATTICE,
             "BFS-discovered loop of a target size.")
      .value("GEODESIC", LoopType::GEODESIC,
             "Shortest cycle through a start simplex.");

  py::class_<LoopPath>(m, "LoopPath",
      "A closed path through the dual graph (sequence of top-simplices).")
      .def_readonly("simplices", &LoopPath::simplices,
                    py::return_value_policy::reference)
      .def_readonly("facets", &LoopPath::facets,
                    py::return_value_policy::reference)
      .def("__len__", [](const LoopPath &lp) { return lp.simplices.size(); });

  py::class_<WilsonResult>(m, "WilsonResult",
      "Result of evaluating a Wilson loop.")
      .def_readonly("value", &WilsonResult::value,
                    "Primary scalar value. In U1_CONNECTION mode this is a "
                    "DERIVED view of connectionAccumulation (its "
                    "residualPhase()), not a second datum.")
      .def_readonly("connectionAccumulation",
                    &WilsonResult::connectionAccumulation,
                    "U1_CONNECTION mode: the complete gauge-invariant datum -- "
                    "the UNREDUCED complex accumulation of the oriented edge "
                    "phase around the cycle. Both components are carried: "
                    "around a closed loop a gauge transformation telescopes to "
                    "zero, so the whole complex sum is gauge-invariant. Only Re "
                    "quantizes, which makes e^{-Im} a gauge-invariant real "
                    "rather than a quantum number. Never reduced mod 2*pi -- "
                    "reducing would destroy the winding. NaN in other modes: "
                    "unmeasured, never zero.")
      .def("holonomy", &WilsonResult::holonomy,
           "Derived: the holonomy exp(i * connectionAccumulation).")
      .def("holonomyModulus", &WilsonResult::holonomyModulus,
           "Derived: |H| = exp(-Im accumulation). Exactly 1 for a purely "
           "compact connection -- a cancellation to be observed, not imposed.")
      .def("residualPhase", &WilsonResult::residualPhase,
           "Derived: Re(accumulation) mod 2*pi, in (-pi, pi].")
      .def("windingNumber", &WilsonResult::windingNumber,
           "Derived: whole 2*pi turns in Re(accumulation). Recoverable only "
           "because the accumulation is stored unreduced.")
      .def_readonly("loopSize", &WilsonResult::loopSize,
                    "Number of simplices in the loop.")
      .def_readonly("enclosedHinges", &WilsonResult::enclosedHinges,
                    "Hinges enclosed by the loop.")
      .def_readonly("contractible", &WilsonResult::contractible,
                    "Whether the loop is contractible.")
      .def_readonly("causalWindingNumber", &WilsonResult::causalWindingNumber,
                    "Net time-orientation changes (causal mode).");

  py::class_<WilsonLoop, std::shared_ptr<WilsonLoop>>(m, "WilsonLoop",
      R"doc(Wilson loop observable on a triangulated spacetime.

A Wilson loop is the trace of a parallel-transport operator around a
closed path. On a curved triangulation without an explicit gauge field
tessera computes the Levi-Civita holonomy analogue: closed walks on the
dual graph (top-simplices as nodes, shared facets as edges), with the
loop value determined by the deficit angles of enclosed hinges.

Four evaluation modes:

* ``COMBINATORIAL``  — dual-graph topology only. ``value`` is the loop
  length; ``enclosedHinges`` counts hinges contained in every loop
  simplex; ``contractible`` is True iff ``enclosedHinges == 0``.
* ``DEFICIT_ANGLE``  — Regge-curvature holonomy. For a hinge loop
  enclosing one hinge h:
      W = ((d-2) + 2 cos(eps_h)) / d
  For multi-hinge loops the U(1) approximation is used:
      W = product_{h in enclosed} cos(eps_h).
  W = 1 corresponds to a flat loop; deviation from 1 measures local
  curvature.
* ``CAUSAL``  — CDT causal-orientation winding. ``causalWindingNumber``
  is the signed net change in foliation index around the loop; non-
  zero values mark loops that cross a CDT slice boundary.
* ``U1_CONNECTION``  — U(1) connection holonomy. The oriented sum of the
  ``Edge.phase`` carried on the primal 1-skeleton around a closed vertex
  cycle (``+phase`` along the stored source->target orientation,
  ``-phase`` reversed), reduced mod 2*pi. Evaluated via
  ``evaluateU1Connection(cycle)`` (the connection is a primal-edge, not a
  dual-graph, quantity); ``value`` carries the holonomy. This is the
  Wilson-loop view of the Stage-1 ``cobordism.HodgeLaplacian`` cycle flux.

Three loop-shape generators:

* ``hingeLoop(h)``        — cyclically ordered loop of top-simplices
                            around the (d-2)-simplex ``h``. Encloses
                            exactly one hinge (``h``), so the
                            DEFICIT_ANGLE formula above is exact.
* ``dualLatticeLoop(start, L)``  — BFS-discovered loop of approximately
                            ``L`` simplices through ``start``. Suitable
                            for population sweeps at fixed loop scale.
* ``geodesicLoop(start)``  — shortest cycle through ``start`` in the
                            dual graph (girth at that simplex).

Measurement bookkeeping: ``measure()`` and ``measureAllHinges()`` append
``WilsonResult`` entries to an internal list. ``getMeasurements()``
returns the accumulated list; ``getAverageBySize()`` aggregates by loop
length (the standard form for Creutz-ratio-style analyses);
``reset()`` clears.

See ``docs/source/wilson_loops.md`` for an end-to-end tutorial with
curvature-scan, contractibility-statistics, and causal-winding
examples.
)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"),
           "Construct a Wilson-loop calculator bound to a Spacetime.")
      .def("evaluate", &WilsonLoop::evaluate,
           py::arg("loop"), py::arg("mode"),
           R"doc(Evaluate the Wilson loop in the given mode.

Dispatches to ``evaluateCombinatorial``, ``evaluateDeficitAngle``, or
``evaluateCausal`` depending on ``mode``.
)doc")
      .def("evaluateCombinatorial", &WilsonLoop::evaluateCombinatorial,
           py::arg("loop"),
           R"doc(Evaluate using dual-graph topology only.

Returns a ``WilsonResult`` with ``value = loopSize``,
``enclosedHinges`` = count of hinges shared by every loop simplex, and
``contractible`` = True iff no hinge is enclosed.
)doc")
      .def("evaluateDeficitAngle", &WilsonLoop::evaluateDeficitAngle,
           py::arg("loop"),
           R"doc(Evaluate using Regge deficit angles.

For a hinge loop (exactly one enclosed hinge h):
    W = ((d - 2) + 2 cos(eps_h)) / d.
For multi-hinge loops the U(1) approximation:
    W = product_{h in enclosed} cos(eps_h).
``value`` carries W; ``enclosedHinges`` carries the count.
)doc")
      .def("evaluateCausal", &WilsonLoop::evaluateCausal,
           py::arg("loop"),
           R"doc(Evaluate using CDT causal-orientation changes.

Walks the loop and accumulates a signed winding count from the
final-time stamps of consecutive simplices. ``causalWindingNumber`` is
the net winding; ``value`` carries the same number as a double.
)doc")
      .def("evaluateU1Connection", &WilsonLoop::evaluateU1Connection,
           py::arg("cycle"),
           R"doc(U(1) connection holonomy around a closed vertex cycle.

``cycle`` is an ordered list of vertices on the primal 1-skeleton whose
consecutive pairs (with wrap-around) are joined by edges. Accumulates each
edge's ``phase`` along its stored source->target orientation (``+phase``
forward, ``-phase`` reversed) and returns the total reduced into the
principal interval ``(-pi, pi]`` in ``value``; ``loopSize`` is the number of
edges. Returns an empty result (``loopSize == 0``) for a degenerate (fewer
than two vertices) or open (a consecutive pair with no joining edge) cycle.

This is the Wilson-loop counterpart of the Stage-1 cycle flux carried by the
Hermitian-weighted ``cobordism.HodgeLaplacian`` — the same oriented phase
sum. Restricted to phases in ``{0, pi}`` the holonomy lands in ``{0, pi}``
and reproduces the Z2 flux.
)doc")
      .def("hingeLoop", &WilsonLoop::hingeLoop,
           py::arg("hinge"),
           R"doc(Loop of top-simplices around a hinge, ordered cyclically.

Encloses exactly one hinge (the input). This is the natural loop for
``DEFICIT_ANGLE`` mode and is what ``measureAllHinges`` uses internally.
)doc")
      .def("dualLatticeLoop", &WilsonLoop::dualLatticeLoop,
           py::arg("start"), py::arg("targetLength"),
           R"doc(BFS-discovered loop of approximately ``targetLength`` simplices.

Not guaranteed to be exactly ``targetLength`` — the BFS may overshoot
or return a shorter loop if local connectivity doesn't permit closing
at the target size. Suitable for population-level scans at a fixed
loop scale (analogous to specifying Wilson-loop side length in lattice
gauge theory).
)doc")
      .def("geodesicLoop", &WilsonLoop::geodesicLoop,
           py::arg("start"),
           R"doc(Shortest cycle through ``start`` in the dual graph.

The cycle length is the local girth of the dual graph at ``start``.
Useful when you want the natural shortest loop without specifying a
target size.
)doc")
      .def("measure", &WilsonLoop::measure,
           py::arg("loop"), py::arg("mode"),
           "Evaluate ``loop`` in ``mode`` and append the result to the "
           "internal measurement list.")
      .def("measureAllHinges", &WilsonLoop::measureAllHinges,
           py::arg("mode"),
           R"doc(Walk every (d-2)-simplex of the spacetime, generate its hinge
loop, and record the evaluation in ``mode``. Skips degenerate hinges
whose loop has fewer than 2 distinct simplices. Bulk shortcut for a
curvature scan.
)doc")
      .def("reset", &WilsonLoop::reset,
           "Clear all accumulated measurements.")
      .def("getMeasurements", &WilsonLoop::getMeasurements,
           "Return the full list of accumulated ``WilsonResult`` entries.")
      .def("getAverageBySize", &WilsonLoop::getAverageBySize,
           R"doc(Mean ``value`` grouped by loop size, as a ``{size: mean}`` dict.

The standard form for Creutz-ratio-style analyses: fix loop size L,
read off the population-averaged Wilson value at that scale.
)doc");

  // ==========================================================================
  // Emergent-proton readout battery (#593): pure readers over a live complex,
  // the loader/transform layer, and the GAUGE/RELABEL gates.
  // ==========================================================================

  // ---- LiveComplex: the loader / transform layer (outside the readers) ----
  py::class_<LiveComplex::Relabeled>(m, "Relabeled",
      "A relabeled rebuild: the live relabeled complex + the vertex-id "
      "permutation (original id -> relabeled id).")
      .def_readonly("spacetime", &LiveComplex::Relabeled::spacetime)
      .def_readonly("vertex_map", &LiveComplex::Relabeled::vertexMap);

  py::class_<LiveComplex>(m, "LiveComplex",
      R"doc(The loader / transform layer OUTSIDE the pure readers: LOAD a saved
combinatorial + metric description back into a live, skeleton-complete Spacetime,
and produce a relabeled copy for the RELABEL gate. Never builds a spacetime of
its own or re-runs the emergent dynamics (those live in Proton / ProtonIngredients
/ MultiCobordism) — only reads a recorded geometry back through the canonical
``Spacetime.fromCells``, completing the facet skeleton with ``materializeFacets``.)doc")
      .def_static("load", &LiveComplex::load, py::arg("cells"),
                  py::arg("squared_lengths"), py::arg("vertex_times"),
                  py::arg("dimensions"),
                  "Load a live skeleton-complete complex from cells + per-edge "
                  "complex squared lengths + vertex times (schema-1 dump "
                  "rehydration).")
      .def_static("subcomplex", &LiveComplex::subcomplex, py::arg("cells"),
                  py::arg("dimensions"),
                  "Load a uniform-metric sub-complex from already-selected "
                  "ambient cells (the block-residual carry diagnostic).")
      .def_static("relabel", &LiveComplex::relabel, py::arg("spacetime"),
                  py::arg("seed"),
                  "A relabeled rebuild under a deterministic vertex-id "
                  "permutation (the RELABEL-gate transform).");

  // ---- RegisterContext: the validated read context (pure reader) ----
  py::class_<RegisterContext, std::shared_ptr<RegisterContext>>(
      m, "RegisterContext",
      R"doc(The one validated read context every emergent-proton observable
measures: a LIVE, already-built complex, its emergent holes, the
induced-orientation signs, and the shared per-complex caches. A pure reader — it
never builds, solves, or materializes anything.)doc")
      .def(py::init([](std::shared_ptr<Spacetime> st, int count, int degree,
                       std::vector<std::complex<double>> target) {
             auto ctx = std::make_shared<RegisterContext>(
                 std::move(st), count, degree, std::move(target));
             emitSelectionWarning(*ctx);
             return ctx;
           }),
           py::arg("spacetime"), py::arg("count") = 3, py::arg("degree") = 3,
           py::arg("target") = ::tessera::cobordism::Proton::singlet())
      .def(py::init([](std::shared_ptr<Spacetime> st,
                       std::vector<std::vector<std::uint64_t>> holes, int count,
                       int degree, std::vector<std::complex<double>> target) {
             auto ctx = std::make_shared<RegisterContext>(
                 std::move(st), holes, count, degree, std::move(target));
             emitSelectionWarning(*ctx);
             return ctx;
           }),
           py::arg("spacetime"), py::arg("holes"), py::arg("count"),
           py::arg("degree"), py::arg("target"))
      .def("spacetime", &RegisterContext::spacetime)
      .def("degree", &RegisterContext::degree)
      .def("target", &RegisterContext::target)
      .def("holes", &RegisterContext::holes)
      .def("dropped_holes", &RegisterContext::droppedHoles)
      .def("holes_used", &RegisterContext::holesUsed)
      .def("holes_total", &RegisterContext::holesTotal)
      .def("bK", &RegisterContext::bK)
      .def("betti", &RegisterContext::betti)
      .def("holes_vs_betti_divergent",
           &RegisterContext::holesVsBettiDivergent)
      .def("dimensions", &RegisterContext::dimensions)
      .def("top_cell_count", &RegisterContext::topCellCount)
      .def("causal_content", &RegisterContext::causalContent)
      .def("selection_warning", &RegisterContext::selectionWarning)
      .def("gauged", &RegisterContext::gauged, py::arg("theta"))
      .def("summary", [](const RegisterContext &ctx) {
        py::dict d;
        d["degree"] = ctx.degree();
        d["dimensions"] = ctx.dimensions();
        d["n_top_cells"] = ctx.topCellCount();
        d["holes_used"] = ctx.holesUsed();
        d["holes_total"] = ctx.holesTotal();
        py::list dropped;
        for (const auto &h : ctx.droppedHoles()) {
          dropped.append(py::cast(h));
        }
        d["dropped_holes"] = dropped;
        d["b3"] = ctx.bK();
        d["betti"] = py::cast(ctx.betti());
        d["holes_vs_b3_divergent"] = ctx.holesVsBettiDivergent();
        d["causal_content"] = ctx.causalContent();
        return d;
      });

  // ---- the observable base (pure reader) ----
  py::class_<RegisterObservable, std::shared_ptr<RegisterObservable>>(
      m, "RegisterObservable",
      "Base for the emergent-proton readouts: a pure post-hoc reader over a "
      "RegisterContext.")
      .def("record_key", &RegisterObservable::recordKey)
      .def("gate_tol", &RegisterObservable::gateTol)
      .def("min_holes", &RegisterObservable::minHoles)
      .def("required_dimensions", &RegisterObservable::requiredDimensions)
      .def("needs_provenance", &RegisterObservable::needsProvenance)
      .def("has_provenance", &RegisterObservable::hasProvenance)
      .def("needs_causal_content", &RegisterObservable::needsCausalContent)
      .def("skip_reason", &RegisterObservable::skipReason, py::arg("ctx"))
      .def(
          "record",
          [](const RegisterObservable &o, const RegisterContext &ctx) {
            return recordToPython(o.record(ctx));
          },
          py::arg("ctx"))
      .def(
          "compute",
          [](const RegisterObservable &o, const RegisterContext &ctx) {
            return o.compute(ctx);
          },
          py::arg("ctx"));

  py::class_<SingletResidual, RegisterObservable,
             std::shared_ptr<SingletResidual>>(m, "SingletResidual",
      "The #574 whole-complex singlet diagnostic (headline = singlet r_state).")
      .def(py::init<>())
      .def("conjugate_residual", &SingletResidual::conjugateResidual,
           py::arg("ctx"));

  py::class_<BlockResiduals::Block>(m, "Block",
      "One provenance block: a label, its vertex region, and its register target.")
      .def(py::init([](std::string label, std::vector<std::uint64_t> vertices,
                       std::vector<std::complex<double>> target) {
             return BlockResiduals::Block{std::move(label), std::move(vertices),
                                          std::move(target)};
           }),
           py::arg("label"), py::arg("vertices"), py::arg("target"))
      .def_readwrite("label", &BlockResiduals::Block::label)
      .def_readwrite("vertices", &BlockResiduals::Block::vertices)
      .def_readwrite("target", &BlockResiduals::Block::target);

  py::class_<BlockResiduals, RegisterObservable,
             std::shared_ptr<BlockResiduals>>(m, "BlockResiduals",
      "The #574 per-output-block carry residuals (blocks are ctor provenance).")
      .def(py::init<std::vector<BlockResiduals::Block>>(), py::arg("blocks"));

  // The mass/radius reader structs (typed accessors).
  py::class_<InteriorHinges::Masses>(m, "EmergentMasses")
      .def_readonly("m_shell", &InteriorHinges::Masses::mShell)
      .def_readonly("m_sum", &InteriorHinges::Masses::mSum)
      .def_readonly("m_action", &InteriorHinges::Masses::mAction)
      .def_readonly("max_abs_im", &InteriorHinges::Masses::maxAbsIm)
      .def_readonly("n_im_nonzero", &InteriorHinges::Masses::nImNonzero)
      .def_readonly("empty", &InteriorHinges::Masses::empty);
  py::class_<InteriorHinges::Radii>(m, "EmergentRadii")
      .def_readonly("v_dual", &InteriorHinges::Radii::vDual)
      .def_readonly("v_primal", &InteriorHinges::Radii::vPrimal)
      .def_readonly("n_interior_vertices",
                    &InteriorHinges::Radii::nInteriorVertices)
      .def_readonly("r_dual", &InteriorHinges::Radii::rDual)
      .def_readonly("r_primal", &InteriorHinges::Radii::rPrimal);
  py::class_<InteriorHinges::Localization>(m, "EmergentLocalization")
      .def_readonly("pr", &InteriorHinges::Localization::pr)
      .def_readonly("concentration", &InteriorHinges::Localization::concentration)
      .def_readonly("mean_re", &InteriorHinges::Localization::meanRe)
      .def_readonly("std_re", &InteriorHinges::Localization::stdRe)
      .def_readonly("std_over_mean",
                    &InteriorHinges::Localization::stdOverMean)
      .def_readonly("rms_shell_radius",
                    &InteriorHinges::Localization::rmsShellRadius)
      .def_readonly("frac_within_shell1",
                    &InteriorHinges::Localization::fracWithinShell1)
      .def_readonly("empty", &InteriorHinges::Localization::empty);

  py::class_<EmergentMass, RegisterObservable, std::shared_ptr<EmergentMass>>(
      m, "EmergentMass",
      "The #575 mass half on the relaxed 4D interior (headline = m_shell).")
      .def(py::init<>())
      .def("masses", &EmergentMass::masses, py::arg("ctx"))
      .def("localization", &EmergentMass::localization, py::arg("ctx"));

  py::class_<EmergentRadius, RegisterObservable,
             std::shared_ptr<EmergentRadius>>(m, "EmergentRadius",
      "The #575 radius half on the relaxed 4D interior (headline = r_dual).")
      .def(py::init<>())
      .def("radii", &EmergentRadius::radii, py::arg("ctx"));

  py::class_<PairLoopFlavor::JointRead>(m, "PairLoopJointRead")
      .def_readonly("sigma", &PairLoopFlavor::JointRead::sigma)
      .def_readonly("r_u", &PairLoopFlavor::JointRead::rU)
      .def_readonly("w", &PairLoopFlavor::JointRead::w)
      .def_readonly("q", &PairLoopFlavor::JointRead::q)
      .def_readonly("loop_w", &PairLoopFlavor::JointRead::loopW)
      .def_readonly("loop_q", &PairLoopFlavor::JointRead::loopQ)
      .def_readonly("dual_residual", &PairLoopFlavor::JointRead::dualResidual);
  py::class_<PairLoopFlavor::Verdict>(m, "PairLoopVerdict")
      .def_readonly("odd_loop", &PairLoopFlavor::Verdict::oddLoop)
      .def_readonly("dual_hole", &PairLoopFlavor::Verdict::dualHole)
      .def_readonly("rho", &PairLoopFlavor::Verdict::rho)
      .def_readonly("multiplicity_2_1", &PairLoopFlavor::Verdict::multiplicity21)
      .def_readonly("odd_is_diquark_loop",
                    &PairLoopFlavor::Verdict::oddIsDiquarkLoop);

  py::class_<PairLoopFlavor, RegisterObservable,
             std::shared_ptr<PairLoopFlavor>>(m, "PairLoopFlavor",
      "The #561/#576 pair-loop dual-basis flavor read (headline = rho).")
      .def(py::init<>())
      .def(py::init([](std::pair<int, int> diquark) {
             return std::make_shared<PairLoopFlavor>(diquark);
           }),
           py::arg("diquark_pair"))
      .def("joint_read", &PairLoopFlavor::jointRead, py::arg("ctx"))
      .def("evaluate_criteria", &PairLoopFlavor::evaluateCriteria,
           py::arg("read"))
      .def_static("odd_one_out", &PairLoopFlavor::oddOneOut, py::arg("loop_q"))
      .def_static("complement_hole", &PairLoopFlavor::complementHole,
                  py::arg("pair"));
  m.attr("PairLoopFlavor").attr("RHO_MAX") = PairLoopFlavor::RHO_MAX;

  // ---- the self-test probes ----
  py::class_<LabelLeakProbe, RegisterObservable,
             std::shared_ptr<LabelLeakProbe>>(m, "LabelLeakProbe",
      "A deliberately label-dependent probe (RELABEL must flag it).")
      .def(py::init<>());
  py::class_<GaugeLeakProbe, RegisterObservable,
             std::shared_ptr<GaugeLeakProbe>>(m, "GaugeLeakProbe",
      "A deliberately gauge-dependent probe (GAUGE must flag it).")
      .def(py::init<>());

  // ---- the GAUGE/RELABEL gate harness ----
  py::class_<ObservableGates::GateResult>(m, "GateResult")
      .def_readonly("gauge_delta", &ObservableGates::GateResult::gaugeDelta)
      .def_readonly("relabel_delta", &ObservableGates::GateResult::relabelDelta)
      .def_readonly("gate_tol", &ObservableGates::GateResult::gateTol)
      .def_readonly("gauge_ok", &ObservableGates::GateResult::gaugeOk)
      .def_readonly("relabel_ok", &ObservableGates::GateResult::relabelOk);

  py::class_<ObservableGates>(m, "ObservableGates",
      "The GAUGE/RELABEL gate harness — post-hoc validation, never a loop "
      "condition.")
      .def_static("gauge_delta", &ObservableGates::gaugeDelta,
                  py::arg("observable"), py::arg("ctx"))
      .def_static("relabel_delta", &ObservableGates::relabelDelta,
                  py::arg("observable"), py::arg("ctx"))
      .def_static("evaluate", &ObservableGates::evaluate, py::arg("observable"),
                  py::arg("ctx"))
      .def_static("self_test", &ObservableGates::selfTest, py::arg("ctx"))
      .def_static(
          "report_delta",
          [](const py::object &a, const py::object &b) {
            return Record::reportDelta(pythonToRecord(a), pythonToRecord(b));
          },
          py::arg("a"), py::arg("b"),
          "The max-abs delta over every numeric leaf of two records (the gate "
          "metric).");
  m.attr("ObservableGates").attr("GAUGE_THETA") = ObservableGates::GAUGE_THETA;
  m.attr("ObservableGates").attr("GATE_SEED") =
      py::int_(ObservableGates::GATE_SEED);

  // ========================================
  // DualVolumeSigns (#605)
  // ========================================
  py::class_<DualVolumeSigns::DimensionReport>(m, "DualVolumeDimensionReport",
      "Per-dimension counts from the diagonal DEC Hodge star sign audit.")
      .def_readonly("dimension", &DualVolumeSigns::DimensionReport::dimension)
      .def_readonly("n_simplices", &DualVolumeSigns::DimensionReport::nSimplices)
      .def_readonly("n_negative_dual_volume",
                    &DualVolumeSigns::DimensionReport::nNegativeDualVolume)
      .def_readonly("n_degenerate_volume",
                    &DualVolumeSigns::DimensionReport::nDegenerateVolume)
      .def_readonly("n_circumcenter_outside",
                    &DualVolumeSigns::DimensionReport::nCircumcenterOutside)
      .def_readonly("n_negative_circumradius",
                    &DualVolumeSigns::DimensionReport::nNegativeCircumradius)
      .def_readonly("n_negative_star",
                    &DualVolumeSigns::DimensionReport::nNegativeStar)
      .def_readonly("n_all_spacelike",
                    &DualVolumeSigns::DimensionReport::nAllSpacelike)
      .def_readonly("n_negative_star_all_spacelike",
                    &DualVolumeSigns::DimensionReport::nNegativeStarAllSpacelike)
      .def_readonly("n_mixed_signature",
                    &DualVolumeSigns::DimensionReport::nMixedSignature)
      .def_readonly(
          "n_negative_star_mixed_signature",
          &DualVolumeSigns::DimensionReport::nNegativeStarMixedSignature)
      .def_readonly("min_star_ratio",
                    &DualVolumeSigns::DimensionReport::minStarRatio)
      .def_readonly("max_star_ratio",
                    &DualVolumeSigns::DimensionReport::maxStarRatio)
      .def_readonly("mean_star_ratio",
                    &DualVolumeSigns::DimensionReport::meanStarRatio);

  py::class_<DualVolumeSigns::Report>(m, "DualVolumeReport",
      "The full diagonal DEC Hodge star sign audit, one entry per simplex "
      "dimension.")
      .def_readonly("dimensions", &DualVolumeSigns::Report::dimensions)
      .def_readonly("n_simplices", &DualVolumeSigns::Report::nSimplices)
      .def_readonly("n_negative_star", &DualVolumeSigns::Report::nNegativeStar);

  py::class_<DualVolumeSigns, std::shared_ptr<DualVolumeSigns> >(
      m, "DualVolumeSigns",
      R"doc(Read-only audit of the sign of the diagonal DEC Hodge star.

The diagonal Discrete Exterior Calculus Hodge star assigns each k-simplex the
scalar ratio |*sigma| / |sigma|, the signed circumcentric dual cell content over
the simplex's own signed content. A Maxwell-type or gauge term discretised with
DEC carries its whole metric dependence in that ratio, so a negative entry costs
positive-definiteness of the Hodge Laplacian and breaks the sign structure a
self-dual / anti-self-dual split of a 2-cochain relies on.

The audit separates the two causes of a negative ratio. A circumcenter falling
outside its simplex (a negative barycentric coordinate) is the Riemannian
well-centeredness violation and indicates badly shaped cells. A timelike
circumcenter displacement (negative signed circumradius squared) is reachable
only in Lorentzian signature and is expected rather than defective. Counts are
therefore broken out by all-spacelike versus mixed-signature cells.

Measures only: changes no geometry and enforces nothing.)doc")
      .def(py::init<double>(), py::arg("tolerance") = 1e-12)
      .def("analyze", &DualVolumeSigns::analyze, py::arg("spacetime"),
           "The full per-dimension audit.")
      .def("compute", &DualVolumeSigns::compute, py::arg("spacetime"),
           "Fraction of audited, non-degenerate simplices whose star ratio is "
           "negative. Zero means the diagonal star is positive everywhere.");

  // ==========================================================================
  // ColorFiber / ColorAnchor (#767): the exact three-edge SU(3) color kernel
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
      R"doc(The reported anchor datum -- the PROFILE, not only the score:
the calibrated atlas score a^2 = sum_tau w_tau |det A_tau|^2, the per-
triangle terms, the maximal term, the participation ratio of the term
distribution, the determinant phases with their circular coherence /
dispersion on overlapping oriented triangles (NaN when no determinant is
nonzero -- unknown is never encoded as zero), the per-triangle Krein
signatures (n+, n0, n-) of the restricted weight blocks (reported
separately from the |W_tau|-restricted score), the frame-normalization
residual, the numerically-checked calibration margin, and the pre-declared
convex weighting that produced the score.)doc")
      .def(py::init<>())
      .def_readonly("score", &AnchorProfile::score)
      .def_readonly("terms", &AnchorProfile::terms)
      .def_readonly("max_term", &AnchorProfile::maxTerm)
      .def_readonly("max_term_index", &AnchorProfile::maxTermIndex)
      .def_readonly("participation_ratio", &AnchorProfile::participationRatio)
      .def_readonly("det_phases", &AnchorProfile::detPhases)
      .def_readonly("phase_coherence", &AnchorProfile::phaseCoherence,
                    "Determinant-phase coherence on OVERLAPPING triangles "
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
          "The #764 tessera.cobordism.Certificate grading the calibrated "
          "score: StructureExact on the diagonal (decoupled) weight path, "
          "CertifiedNumerical on the general Hermitian-matrix path; regime "
          "PositiveSemidefinite / HermitianIndefinite per the Krein read; "
          "residual = max(frame_gram_residual, max(0, calibration_margin)) "
          "against the evaluate gram tolerance.");

  py::class_<AnchorGate>(m, "AnchorGate",
      "The triangle-anchor gate the exactness contract requires before any "
      "colour-specific kernel runs. DEFAULT-CONSTRUCTED IS CLOSED, so a "
      "caller that supplies nothing is refused rather than admitted; the "
      "only way to open one is ColorAnchor.gateFor, which applies the same "
      "acceptance predicate the quark verdict uses.")
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
      "(N=3).  Sector READS only -- never a particle classification.")
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
      R"doc(The exact three-edge SU(3) color kernel (#767): the constant
color-sector algebra of three
oriented edge modes, Lambda* C^3 = 1 (+) 3 (+) 3bar (+) 1, layered over the
#766 exterior-algebra primitives (sector projectors and CAR matrices are
delegated to tessera.quantum.ExteriorAlgebra, never reimplemented).

All members are static; Fock operators are dense 8x8 matrices on the
occupation basis n(b) = sum_i b_i 2^i, and the one-occupation (triplet)
sector is spanned by Fock indices (1, 2, 4).  Exact identities (tested to
double round-off): F3^dag F3 = I, |det F3| = 1; lambda_a Hermitian,
traceless, Tr(lambda_a lambda_b) = 2 delta_ab; [E_ij, E_kl] = delta_jk E_il
- delta_il E_kj on both representations; det(gC) = det(C) for g in SU(3);
||v1 ^ v2 ^ v3||^2 = det[<v_i, v_j>].  Pure constants and reads -- no
solver call, no mutation, nothing enters the emergence objective.)doc")
      .def_static("sectorProjector", &ColorFiber::sectorProjector,
                  py::arg("occupation"),
                  "The 8x8 projector onto total occupation N (0..3; zero "
                  "matrix above 3).  Delegates to "
                  "quantum.ExteriorAlgebra.sectorProjector on three modes.")
      .def_static("vacuumProjector", &ColorFiber::vacuumProjector,
                  "Lambda^0: the even vacuum singlet (N=0).")
      .def_static("tripletProjector", &ColorFiber::tripletProjector,
                  "Lambda^1: the odd fundamental color triplet 3 (N=1).")
      .def_static("antiTripletProjector", &ColorFiber::antiTripletProjector,
                  "Lambda^2: the even antisymmetric anti-triplet 3bar (N=2).")
      .def_static("singletProjector", &ColorFiber::singletProjector,
                  "Lambda^3: the odd top-wedge color singlet (N=3).")
      .def_static("creationMatrix", &ColorFiber::creationMatrix,
                  py::arg("mode"), "The 8x8 creation matrix a_i^dag.")
      .def_static("annihilationMatrix", &ColorFiber::annihilationMatrix,
                  py::arg("mode"), "The 8x8 annihilation matrix a_i.")
      .def_static("hoppingMatrix", &ColorFiber::hoppingMatrix,
                  py::arg("i"), py::arg("j"),
                  "The 8x8 bilinear E_ij = a_i^dag a_j (exact gl(3) "
                  "commutation relations on the whole Fock space).")
      .def_static("tripletBasisIndices", &ColorFiber::tripletBasisIndices,
                  "The Fock indices (1, 2, 4) identifying the N=1 sector "
                  "with C^3.")
      .def_static("restrictToTriplet", &ColorFiber::restrictToTriplet,
                  py::arg("op"),
                  "Restrict an 8x8 Fock operator to the one-occupation "
                  "sector as a 3x3 matrix; restrictToTriplet(dGamma(M)) = M "
                  "exactly.")
      .def_static("matrixUnit", &ColorFiber::matrixUnit,
                  py::arg("i"), py::arg("j"),
                  "The 3x3 matrix unit E_ij on the one-occupation sector.")
      .def_static("dGamma", &ColorFiber::dGamma, py::arg("m"),
                  "Second quantization dGamma(M) = sum_ij M_ij a_i^dag a_j "
                  "of a 3x3 one-particle matrix (8x8).")
      .def_static("gellMann", &ColorFiber::gellMann, py::arg("a"),
                  "lambda_a for a in 1..8, assembled from the matrix units "
                  "(lambda_3 = E11-E22, lambda_8 = (E11+E22-2E33)/sqrt(3)).")
      .def_static("adjointOctetProjector", &ColorFiber::adjointOctetProjector,
                  "The 9x9 orthogonal projector onto the traceless "
                  "(adjoint-octet) part of a 3x3 bilinear, acting on "
                  "column-major vec(M).")
      .def_static("tracelessPart", &ColorFiber::tracelessPart, py::arg("m"),
                  "M - (tr M / 3) I: the octet component of a bilinear.")
      .def_static("adjointSingletProjector",
                  &ColorFiber::adjointSingletProjector,
                  "The 9x9 projector vec(I)vec(I)^dag/3 onto the trace "
                  "(singlet) part -- implemented literally as I9 - "
                  "adjointOctetProjector(), so P1 + P8 = I9 resolves "
                  "3 x 3bar = 1 + 8 exactly (#774).")
      .def_static("octetBilinear", &ColorFiber::octetBilinear,
                  py::arg("i"), py::arg("j"),
                  "The 8x8 traceless even bilinear "
                  "T_ij = a_i^dag a_j - (delta_ij/3) N on Fock space "
                  "(= dGamma(tracelessPart(matrixUnit(i, j)))): conserves N "
                  "(even fermion parity) and the nine T_ij span the octet "
                  "(#774).")
      .def_static("adjointCasimirMatrix", &ColorFiber::adjointCasimirMatrix,
                  "The 9x9 quadratic Casimir of the adjoint action, "
                  "C = sum_a K_a^2 with K_a vec(M) = vec([lambda_a/2, M]); "
                  "exactly C = 3 P8 (#774).")
      .def_static("adjointCasimir", &ColorFiber::adjointCasimir,
                  py::arg("m"),
                  "The adjoint-Casimir Rayleigh quotient in [0, 3]: exactly "
                  "3 for traceless M, 0 for M ~ I, NaN for M = 0 (#774).")
      .def_static("omega", &ColorFiber::omega,
                  "The primitive cube root of unity as its algebraic value "
                  "(-1 + i sqrt(3))/2 (never exp), so 1 + omega + omega^2 "
                  "cancels exactly in floating point.")
      .def_static("fourierFrame", &ColorFiber::fourierFrame,
                  "The exact unitary Fourier frame F3 with entries "
                  "omega^{jk}/sqrt(3), assembled from the algebraic table "
                  "{1, omega, omega^2} by exponent jk mod 3.")
      .def_static("fourierBasisVector", &ColorFiber::fourierBasisVector,
                  py::arg("k"),
                  "Column k of F3: the Z3 character vector "
                  "(1, omega^k, omega^{2k})/sqrt(3).")
      .def_static("omegaPhaseState", &ColorFiber::omegaPhaseState,
                  "The existing phase pattern (1, omega, omega^2)/sqrt(3), "
                  "identified as ONE color basis vector "
                  "(fourierBasisVector(1)); its cyclic orbit under pointwise "
                  "Z3 powers is the exact orthonormal triad = the columns of "
                  "F3.")
      .def_static("perimeter", &ColorFiber::perimeter, py::arg("z"),
                  "The triangle perimeter sum_i |z_i|^{1/2} of three stored "
                  "complex SQUARED lengths (the L1 geometric datum).")
      .def_static("perimeterNormalized", &ColorFiber::perimeterNormalized,
                  py::arg("z"),
                  "Rescale the squared lengths so the perimeter is one -- a "
                  "GEOMETRIC SCALE GAUGE (L1), never a state normalization.")
      .def_static("hilbertNorm", &ColorFiber::hilbertNorm, py::arg("z"),
                  "The Hilbert L2 norm ||z||_2.")
      .def_static("hilbertNormalized", &ColorFiber::hilbertNormalized,
                  py::arg("z"),
                  "z / ||z||_2 with <c|c> = 1 -- the STATE normalization, "
                  "distinct from the perimeter gauge.")
      .def_static("colorVector", &ColorFiber::colorVector, py::arg("z"),
                  "The color vector from the stored complex squared "
                  "lengths: c = z / ||z||_2.")
      .def_static("colorWedge",
                  py::overload_cast<const Eigen::Matrix3cd&>(
                      &ColorFiber::colorWedge),
                  py::arg("c"),
                  "The color-wedge (singlet) amplitude det C = eps_ijk C_i1 "
                  "C_j2 C_k3; det(gC) = det(C) for g in SU(3).")
      .def_static("colorWedgeColumns",
                  py::overload_cast<const Eigen::Vector3cd&,
                                    const Eigen::Vector3cd&,
                                    const Eigen::Vector3cd&>(
                      &ColorFiber::colorWedge),
                  py::arg("a"), py::arg("b"), py::arg("c"),
                  "colorWedge of three explicit color columns.")
      .def_static("singletGram", &ColorFiber::singletGram, py::arg("c"),
                  "det(C^dag C) = |det C|^2 = ||c1 ^ c2 ^ c3||^2: exactly "
                  "zero for duplicate color modes, exactly one for an "
                  "orthonormal triad.")
      .def_static("isSpecialUnitary", &ColorFiber::isSpecialUnitary,
                  py::arg("g"), py::arg("tol") = 1e-12,
                  "Certify g in SU(3): ||g^dag g - I||_max <= tol and "
                  "|det g - 1| <= tol.")
      .def_static("sectorWeights", &ColorFiber::sectorWeights,
                  py::arg("state"),
                  "The four occupation-sector weights of an 8-dimensional "
                  "Fock vector (their sum is ||psi||^2 exactly).")
      .def_static("octetRead", &ColorFiber::octetRead, py::arg("m"),
                  "The octet/singlet Frobenius weights of a 3x3 bilinear.")
      .def_static("verifyConstantAlgebra", &ColorFiber::verifyConstantAlgebra,
                  "Re-derive every constant-algebra identity and return the "
                  "maximum absolute residual (run at startup in debug "
                  "builds; callable in every build).")
      .def_static("constantAlgebraCertificate",
                  &ColorFiber::constantAlgebraCertificate,
                  "The #764 AlgebraicallyExact certificate of the constant "
                  "algebra (measured verifyConstantAlgebra residual against "
                  "the startup tolerance 1e-12).");

  py::class_<ColorAnchor>(m, "ColorAnchor",
      R"doc(The calibrated weighted oriented-triangle anchoring kernel for
an abstract rank-three band (#767; whitepaper "Quarks as modular
clusters"): A_tau = |W_tau|^{1/2} R_tau Phi per declared oriented triangle,
atlas score a^2 = sum_tau w_tau |det A_tau|^2 with the convex weighting
DECLARED BEFORE the data are examined (post-hoc re-weighting raises).

Exact identity and domain: with the frame |W|-orthonormal (verified per
evaluate and reported as frame_gram_residual) and |W| triangle-decoupled
(any diagonal per-edge metric -- the production DEC/Hodge case), each
|det A_tau|^2 = det(A_tau^dag A_tau) <= 1 because R_tau^dag |W_tau| R_tau
is dominated by |W|, so the score is calibrated to [0, 1] with value one
exactly at full concentration on the weighted edge span.  A single literal
triangle is the exact oracle; an extended anchored fiber is the production
case.  For a general Hermitian (coupled) weight the <= 1 bound is CHECKED
(calibration_margin), never assumed.  Signed sectors restrict with
|W_tau|^{1/2} and report each restricted block's Krein signature
separately.

Operates only on caller-supplied inputs (frame over oriented edges, edge
weight data, oriented-triangle descriptors); mutates nothing; never enters
the emergence objective; contains no transport code.)doc")
      .def(py::init<std::vector<OrientedTriangle>>(), py::arg("triangles"),
           "Declare the atlas with the UNIFORM convex weighting 1/T.")
      .def(py::init<std::vector<OrientedTriangle>, std::vector<double>>(),
           py::arg("triangles"), py::arg("weights"),
           "Declare the atlas with an EXPLICIT convex weighting (each >= 0, "
           "summing to one within 1e-12).")
      .def("triangles", &ColorAnchor::triangles,
           "The declared oriented triangles (immutable).")
      .def("weights", &ColorAnchor::weights, "The declared convex weights.")
      .def("weightingId", &ColorAnchor::weightingId,
           "'uniform' or 'declared'.")
      .def("overlapsAnother", &ColorAnchor::overlapsAnother, py::arg("index"),
           "Whether declared triangle `index` shares a boundary EDGE with "
           "another declared triangle -- the overlap relation the "
           "determinant-phase coherence is recorded on.")
      .def("overlappingTriangleCount", &ColorAnchor::overlappingTriangleCount,
           "How many declared triangles overlap another (0 on a disjoint "
           "atlas, where the coherence is UNKNOWN).")
      .def("sealed", &ColorAnchor::sealed,
           "True once any data have been evaluated (weighting sealed).")
      .def("declareWeights", &ColorAnchor::declareWeights, py::arg("weights"),
           "Replace the declared convex weighting -- allowed ONLY before "
           "the first evaluate(); afterwards post-hoc weight selection is "
           "rejected (raises).")
      .def_static("accepts", &ColorAnchor::accepts, py::arg("profile"),
                  py::arg("min_score") = ColorAnchor::kDefaultMinScore,
                  py::arg("min_phase_coherence") =
                      ColorAnchor::kDefaultMinPhaseCoherence,
                  "THE triangle-anchor acceptance predicate -- one "
                  "definition, shared by the quark verdict and by the colour "
                  "kernels the exactness contract gates on it. A profile "
                  "passes when a weighting was actually declared (an empty "
                  "weighting_id is MISSING evidence, not a zero score), its "
                  "calibration certificate holds, and both the atlas score "
                  "and the determinant-phase coherence meet their floors.")
      .def_static("gateFor", &ColorAnchor::gateFor, py::arg("profile"),
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
           "Evaluate against a DIAGONAL (possibly signed) per-edge weight "
           "vector -- the domain where the [0,1] calibration bound is "
           "exact.  The frame must be |W|-orthonormal within "
           "gram_tolerance (use orthonormalizeFrame).")
      .def("evaluateMatrix",
           py::overload_cast<const Eigen::MatrixXcd&, const Eigen::MatrixXcd&,
                             double>(&ColorAnchor::evaluate),
           py::arg("frame"), py::arg("weight"),
           py::arg("gram_tolerance") = 1e-9,
           "Evaluate against a general Hermitian ExE weight matrix; the "
           "calibration bound is checked (calibration_margin), not "
           "assumed.")
      .def_static("anchorMatrix", &ColorAnchor::anchorMatrix,
                  py::arg("frame"), py::arg("edge_weights"), py::arg("tri"),
                  "The raw 3x3 weighted anchor matrix A_tau = |W_tau|^{1/2} "
                  "R_tau Phi of one triangle (diagonal weights; no "
                  "normalization check).")
      .def_static("orthonormalizeFrame",
                  py::overload_cast<const Eigen::MatrixXcd&,
                                    const Eigen::VectorXd&>(
                      &ColorAnchor::orthonormalizeFrame),
                  py::arg("frame"), py::arg("edge_weights"),
                  "The |W|-orthonormalized frame Phi (Phi^dag |W| "
                  "Phi)^{-1/2} for a diagonal per-edge weight vector.")
      .def_static("orthonormalizeFrameMatrix",
                  py::overload_cast<const Eigen::MatrixXcd&,
                                    const Eigen::MatrixXcd&>(
                      &ColorAnchor::orthonormalizeFrame),
                  py::arg("frame"), py::arg("weight"),
                  "Matrix-weight overload of orthonormalizeFrame (Hermitian "
                  "W; uses the eigen-modulus |W|).");

  // ==========================================================================
  // ExchangeHolonomy (#772): Berry-cancelled exchange statistics, the
  // constructed total-space spin holonomy cycle, and the conditional
  // SO(d) -> Spin(d) lift.  Read-only; nothing enters any emergence
  // objective; no Kasteleyn orientation is required anywhere.
  // ==========================================================================
  py::class_<ExchangeHolonomyConfig>(m, "ExchangeHolonomyConfig",
      "Analysis parameters of the exchange/rotation holonomy reads "
      "(ticket #772).  Thresholds select which reads are CERTIFIED — a "
      "failed threshold yields an UNCERTIFIED read, never a different "
      "sign.")
      .def(py::init<>())
      .def_readwrite("leakFloor", &ExchangeHolonomyConfig::leakFloor,
                     "Minimum overlap singular value of a certified step "
                     "(a leaking transfer is rejected before polar "
                     "normalization).")
      .def_readwrite("conditionCap", &ExchangeHolonomyConfig::conditionCap,
                     "Maximum overlap conditioning of a certified step.")
      .def_readwrite("unitaryTolerance",
                     &ExchangeHolonomyConfig::unitaryTolerance,
                     "Certificate tolerance on loop unitarity / character "
                     "modulus.")
      .def_readwrite("signTolerance",
                     &ExchangeHolonomyConfig::signTolerance,
                     "Distance from +-1 within which a definite "
                     "characterSign is reported.")
      .def_readwrite("blockMatchThreshold",
                     &ExchangeHolonomyConfig::blockMatchThreshold,
                     "Minimum subspace overlap of a certified block "
                     "continuation (mirrors the #769 tracker threshold).")
      .def_readwrite("liftAngleMargin",
                     &ExchangeHolonomyConfig::liftAngleMargin,
                     "Lifted loop steps must stay this far below the pi "
                     "branch cut.")
      .def_readwrite("cocycleTolerance",
                     &ExchangeHolonomyConfig::cocycleTolerance,
                     "Cap on the verified SO(d) cocycle residual of "
                     "spinLift.");

  py::enum_<HolonomyChannel>(m, "HolonomyChannel",
      "Which physical question a Berry-cancelled character answers; "
      "particle exchange and physical rotation are separate channels by "
      "construction and doublyCancelledRatio refuses mislabeled inputs.")
      .value("ParticleExchange", HolonomyChannel::ParticleExchange)
      .value("PhysicalRotation", HolonomyChannel::PhysicalRotation);

  py::class_<TransportStepRead>(m, "TransportStepRead",
      "One overlap-transport step: singular-value data of the r x r frame "
      "overlap BEFORE polar normalization, and whether the step met the "
      "leak/conditioning thresholds.")
      .def_readonly("fromIndex", &TransportStepRead::fromIndex)
      .def_readonly("toIndex", &TransportStepRead::toIndex)
      .def_readonly("minSingularValue", &TransportStepRead::minSingularValue)
      .def_readonly("maxSingularValue", &TransportStepRead::maxSingularValue)
      .def_readonly("conditioning", &TransportStepRead::conditioning)
      .def_readonly("certified", &TransportStepRead::certified);

  py::class_<LoopHolonomyRead>(m, "LoopHolonomyRead",
      R"doc(Certified cyclic overlap transport of one tracked frame around a
CLOSED loop: R_t = polar(Phi_{t+1 mod T}^dagger W_t Phi_t), U_gamma =
R_{T-1} ... R_0.  `determinant` is the RAW chi_raw = det U_gamma — it
contains the ordinary Berry phase of the reference motion and is NEVER an
exchange sign by itself; only the interferometric ratio against a matched
reference loop is the dynamical certificate.  An uncertified band on the
loop (gap closure), a leak, or ill-conditioning yields an UNCERTIFIED
read, never a sign.)doc")
      .def_readonly("holonomy", &LoopHolonomyRead::holonomy)
      .def_readonly("determinant", &LoopHolonomyRead::determinant)
      .def_readonly("steps", &LoopHolonomyRead::steps)
      .def_readonly("rank", &LoopHolonomyRead::rank)
      .def_readonly("stepReads", &LoopHolonomyRead::stepReads)
      .def_readonly("unitarityResidual",
                    &LoopHolonomyRead::unitarityResidual)
      .def_readonly("minStepSingularValue",
                    &LoopHolonomyRead::minStepSingularValue)
      .def_readonly("conditioning", &LoopHolonomyRead::conditioning)
      .def_readonly("uncertifiedBand", &LoopHolonomyRead::uncertifiedBand)
      .def_readonly("certificate", &LoopHolonomyRead::certificate);

  py::class_<HolonomyCharacterRead>(m, "HolonomyCharacterRead",
      R"doc(The interferometric (Berry-cancelled) character chi_hat =
det U_loop / det U_reference, with the phase channels kept SEPARATE:
rawLoopDeterminant (exchange/rotation + Berry), referenceDeterminant (the
Berry reference motion alone), character (the cancelled ratio).
characterSign is -1/+1 only when the certificate holds and the character
sits within signTolerance of -+1; an uncertified read never emits a
sign.)doc")
      .def_readonly("channel", &HolonomyCharacterRead::channel)
      .def_readonly("rawLoopDeterminant",
                    &HolonomyCharacterRead::rawLoopDeterminant)
      .def_readonly("referenceDeterminant",
                    &HolonomyCharacterRead::referenceDeterminant)
      .def_readonly("character", &HolonomyCharacterRead::character)
      .def_readonly("characterSign", &HolonomyCharacterRead::characterSign)
      .def_readonly("signResidual", &HolonomyCharacterRead::signResidual)
      .def_readonly("timingMatched", &HolonomyCharacterRead::timingMatched)
      .def_readonly("ranksMatched", &HolonomyCharacterRead::ranksMatched)
      .def_readonly("certificate", &HolonomyCharacterRead::certificate);

  py::class_<BlockPermutationRead>(m, "BlockPermutationRead",
      R"doc(The structural exchange channel: the permutation of persistent
localized blocks around the loop (matching delegated to
SpectralFiberTracker.matchFibers), its EXACT parities through the #766
grading (modeParity = the graded exchange statistic; blockParity = the
block-label sign; compositeParity = the optional composite-level sign),
and the residual in-block motion after reference cancellation.  Parities
are exact integers GIVEN the verified matching premise; a failed premise
(gap closure, rank change, ambiguous matching) yields an UNCERTIFIED read
with no parities.)doc")
      .def_readonly("blockPermutation",
                    &BlockPermutationRead::blockPermutation)
      .def_readonly("blockRanks", &BlockPermutationRead::blockRanks)
      .def_readonly("blockParity", &BlockPermutationRead::blockParity)
      .def_readonly("modeParity", &BlockPermutationRead::modeParity)
      .def_readonly("compositePermutation",
                    &BlockPermutationRead::compositePermutation)
      .def_readonly("compositeParity",
                    &BlockPermutationRead::compositeParity)
      .def_readonly("minMatchOverlap",
                    &BlockPermutationRead::minMatchOverlap)
      .def_readonly("residualInBlockMotion",
                    &BlockPermutationRead::residualInBlockMotion)
      .def_readonly("certificate", &BlockPermutationRead::certificate);

  py::class_<LoopLiftRead>(m, "LoopLiftRead",
      "The Z2 character of a closed SO(d) loop lifted step-by-step to "
      "Spin(d): +1 contractible, -1 the double-cover generator; 0 "
      "(UNCERTIFIED) when a step approached the pi branch cut or the "
      "lifted product failed to close on +-I.")
      .def_readonly("character", &LoopLiftRead::character)
      .def_readonly("maxStepAngle", &LoopLiftRead::maxStepAngle)
      .def_readonly("closureResidual", &LoopLiftRead::closureResidual)
      .def_readonly("certificate", &LoopLiftRead::certificate);

  py::class_<SpinLiftRead>(m, "SpinLiftRead",
      R"doc(The SO(d) -> Spin(d) lift decision over Cech transition data
with the second Stiefel-Whitney obstruction: per-triangle lift signs, the
exact GF(2) coboundary decision, and (when the lift exists) a consistent
per-edge sign choice.  Continuum-claim machinery only — the abstract
CAR/Fock algebra needs no spin structure and no Kasteleyn orientation.)doc")
      .def_readonly("liftExists", &SpinLiftRead::liftExists)
      .def_readonly("obstructed", &SpinLiftRead::obstructed)
      .def_readonly("triangleSigns", &SpinLiftRead::triangleSigns)
      .def_readonly("edgeSigns", &SpinLiftRead::edgeSigns)
      .def_readonly("maxCocycleResidual",
                    &SpinLiftRead::maxCocycleResidual)
      .def_readonly("maxLiftResidual", &SpinLiftRead::maxLiftResidual)
      .def_readonly("certificate", &SpinLiftRead::certificate)
      .def("describe", &SpinLiftRead::describe)
      .def("__repr__", &SpinLiftRead::describe);

  py::class_<ExchangeHolonomy>(m, "ExchangeHolonomy",
      R"doc(Berry-cancelled exchange statistics, the constructed total-space
spin holonomy cycle, and the conditional SO(d) -> Spin(d) lift (ticket
#772).

Identities: (1) certified cyclic overlap transport R_t =
polar(Phi_{t+1}^dagger W_t Phi_t), U_gamma = R_{T-1}...R_0, composing #769
SpectralFiber frames; (2) the interferometric exchange character chi_hat =
det U_exchange / det U_reference (the raw determinant contains Berry phase
and is never the sign); (3) the structural block permutation with exact
#766 parities and the reference-cancelled in-block residual; (4) the
constructed total-space spin holonomy cycle as the canonical physical
rotation path with its co-moving reference (one global rotation of the
whole carried frame — never per-hole Bloch products); (5) the exact
total-space J^2 measuring stick (proton eigenstate -> 3/4, Delta -> 15/4);
(6) the principal rotation logarithm, the Spin(d) lift, the Z2 loop
character, and the w2 obstruction over Cech data.

Channels kept separate in API and report: simplex reorientation
(reorientedFrames — exactly invariant), compilation ordering
(permutedCellFrames / cell-tuple matching — exactly invariant), particle
exchange, Berry reference motion, physical rotation.

Read-only and stateless: never calls a solver, never mutates what it
reads, and nothing here may enter any emergence objective.)doc")
      .def_static("polarUnitary", &ExchangeHolonomy::polarUnitary,
                  py::arg("overlap"),
                  "The unitary polar factor U V^dagger of an overlap "
                  "matrix (the normative transport primitive).")
      .def_static("loopHolonomy", &ExchangeHolonomy::loopHolonomy,
                  py::arg("frames"), py::arg("weights"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Closed-loop holonomy of an explicit frame path under a "
                  "constant diagonal metric.")
      .def_static("loopHolonomyPerStep",
                  &ExchangeHolonomy::loopHolonomyPerStep,
                  py::arg("frames"), py::arg("stepWeights"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Closed-loop holonomy with per-step diagonal metrics "
                  "W_t.")
      .def_static("fiberLoopHolonomy",
                  &ExchangeHolonomy::fiberLoopHolonomy, py::arg("loop"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Closed-loop holonomy of a #769 fiber track (shared "
                  "cells matched by vertex tuple; an uncertified band or "
                  "rank change yields an UNCERTIFIED read).")
      .def_static("exchangeCharacter",
                  &ExchangeHolonomy::exchangeCharacter,
                  py::arg("exchangeLoop"), py::arg("referenceLoop"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "chi_hat_F = det U_exchange / det U_reference "
                  "(ParticleExchange channel).")
      .def_static("rotationCharacter",
                  &ExchangeHolonomy::rotationCharacter,
                  py::arg("rotationLoop"), py::arg("referenceLoop"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "chi_hat(2 pi) against the matched co-moving "
                  "non-rotating reference (PhysicalRotation channel).")
      .def_static("doublyCancelledRatio",
                  &ExchangeHolonomy::doublyCancelledRatio,
                  py::arg("exchange"), py::arg("rotation"),
                  "chi_hat(exchange) * chi_hat(2 pi)^{-1}; requires the "
                  "correct channel tags (ValueError otherwise).")
      .def_static("blockPermutation", &ExchangeHolonomy::blockPermutation,
                  py::arg("steps"),
                  py::arg("referenceSteps") =
                      std::vector<std::vector<SpectralFiber>>{},
                  py::arg("composites") =
                      std::vector<std::vector<std::size_t>>{},
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "Structural block tracking around the loop: permutation, "
                  "exact #766 parities, reference-cancelled in-block "
                  "residual.")
      .def_static("spinorDimension", &ExchangeHolonomy::spinorDimension,
                  py::arg("d"))
      .def_static("gamma", &ExchangeHolonomy::gamma, py::arg("a"),
                  py::arg("d"),
                  "Euclidean gamma_a with {gamma_a, gamma_b} = 2 delta_ab "
                  "(Pauli at d = 3; the documented Dirac layer at d = 4).")
      .def_static("spinGenerator", &ExchangeHolonomy::spinGenerator,
                  py::arg("a"), py::arg("b"), py::arg("d"),
                  "Sigma_ab = [gamma_a, gamma_b]/4, eigenvalues -+i/2.")
      .def_static("spinorRotation", &ExchangeHolonomy::spinorRotation,
                  py::arg("theta"), py::arg("a"), py::arg("b"),
                  py::arg("d"),
                  "exp(theta Sigma_ab) in closed form; theta = 2 pi gives "
                  "exactly -I (the double cover).")
      .def_static("transverseSpinorFrame",
                  &ExchangeHolonomy::transverseSpinorFrame, py::arg("a"),
                  py::arg("b"), py::arg("d"),
                  "The canonical transverse rank-1 spinor frame of the "
                  "(a, b) plane (deterministic conventions).")
      .def_static("rotationLoopFrames",
                  &ExchangeHolonomy::rotationLoopFrames, py::arg("frame0"),
                  py::arg("a"), py::arg("b"), py::arg("d"),
                  py::arg("turns"), py::arg("steps"),
                  "The constructed total-space spin holonomy cycle as an "
                  "explicit closed frame path (one global rotation of the "
                  "whole carried frame).")
      .def_static("referenceLoopFrames",
                  &ExchangeHolonomy::referenceLoopFrames, py::arg("frame0"),
                  py::arg("steps"),
                  "The matched co-moving NON-rotating reference (same "
                  "timing, no rotation).")
      .def_static("vectorLoopFrames", &ExchangeHolonomy::vectorLoopFrames,
                  py::arg("frame0"), py::arg("a"), py::arg("b"),
                  py::arg("d"), py::arg("turns"), py::arg("steps"),
                  "The vector-representation rotation loop (the +1 "
                  "control).")
      .def_static("totalJSquaredOperator",
                  &ExchangeHolonomy::totalJSquaredOperator,
                  py::arg("constituents"),
                  "J^2 = sum_a (sum_i S_a^(i))^2 on (C^2)^(tensor n) — the "
                  "total-space operator on the whole composite state.")
      .def_static("totalJSquared", &ExchangeHolonomy::totalJSquared,
                  py::arg("state"),
                  "<J^2> of a composite state (exact oracles: proton "
                  "eigenstate 3/4, Delta 15/4, product |uud> 7/4).")
      .def_static("rotationLog", &ExchangeHolonomy::rotationLog,
                  py::arg("rotation"),
                  "The principal antisymmetric logarithm via the real "
                  "Schur plane decomposition (pi branch by the documented "
                  "axis rule).")
      .def_static("rotationToSpin", &ExchangeHolonomy::rotationToSpin,
                  py::arg("rotation"), py::arg("d"),
                  "The principal Spin(d) lift of an SO(d) rotation "
                  "(half-angle plane factors; d = 3, 4).")
      .def_static("loopLiftCharacter",
                  &ExchangeHolonomy::loopLiftCharacter, py::arg("loop"),
                  py::arg("d"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "The Z2 character of a closed SO(d) loop by incremental "
                  "principal lifts (UNCERTIFIED near the pi branch cut).")
      .def_static("spinLift", &ExchangeHolonomy::spinLift,
                  py::arg("edges"), py::arg("edgeRotations"),
                  py::arg("triangles"), py::arg("d"),
                  py::arg("config") = ExchangeHolonomyConfig{},
                  "The SO(d) -> Spin(d) lift decision over Cech data with "
                  "the w2 obstruction (exact GF(2) coboundary decision "
                  "given the verified cocycle premise).")
      .def_static("reorientedFrames", &ExchangeHolonomy::reorientedFrames,
                  py::arg("frames"), py::arg("cellSigns"),
                  "The simplex-reorientation gauge (common row sign "
                  "flips) — every read is exactly invariant.")
      .def_static("permutedCellFrames",
                  &ExchangeHolonomy::permutedCellFrames, py::arg("frames"),
                  py::arg("rowPermutation"),
                  "The compilation-ordering gauge (common row "
                  "permutation) — every read is exactly invariant.");
  // ========================================
  // FiberConnection (#770): derived U(r) fiber transport, Wilson
  // observables, rank-three center structure, determinant winding
  // ========================================
  py::class_<FiberConnectionConfig>(m, "FiberConnectionConfig",
      R"doc(Threshold configuration of the derived-transport gates (#770).
Every gate fires BEFORE polar/pseudo-unitary reduction; a failed gate
yields a rejected read that still reports its raw map and diagnostics --
polar normalization never conceals a bad assignment.)doc")
      .def(py::init<>())
      .def_readwrite("rankTolerance", &FiberConnectionConfig::rankTolerance,
                     "Relative singular-value cut for the numerical rank.")
      .def_readwrite("leakageTolerance",
                     &FiberConnectionConfig::leakageTolerance,
                     "Cap on the isometry leakage before a unitary factor "
                     "may be emitted.")
      .def_readwrite("conditionNumberCap",
                     &FiberConnectionConfig::conditionNumberCap,
                     "Cap on endpoint frame and overlap conditioning.")
      .def_readwrite("minEndpointGap", &FiberConnectionConfig::minEndpointGap,
                     "Absolute floor on each endpoint band's isolation "
                     "min(lowerGap, upperGap); 0 = rely on band "
                     "certification.")
      .def_readwrite("requireCertifiedFibers",
                     &FiberConnectionConfig::requireCertifiedFibers,
                     "Require both endpoint bands accepted (a closing gap "
                     "rejects the transport).")
      .def_readwrite("certificateTolerance",
                     &FiberConnectionConfig::certificateTolerance,
                     "Tolerance the emitted #764 certificates hold against.")
      .def_readwrite("closureTolerance",
                     &FiberConnectionConfig::closureTolerance,
                     "Relative endpoint-mismatch cap for certified winding "
                     "closures.");

  py::class_<FiberTransportRead>(m, "FiberTransportRead",
      R"doc(One derived fiber transport A <- B: the raw overlap
M_AB = Phi_A^dagger W_A T_AB Phi_B (Psi_A^dagger on the
biorthogonal path), EVERY pre-normalization diagnostic (rank, singular
values, leakage, endpoint gaps/signatures, frame conditioning), the
normalized U(r)/pseudo-unitary factor when its gates passed, the
determinant-line datum, and the graded #764 certificate.  A rejected read
still carries the raw map and diagnostics.  The spec's per-transport
winding/center fields materialize on the dedicated family reads
(DeterminantWindingRead / FundamentalLiftRead): an integer winding exists
only for a declared family/closure, a center sector only for a declared
lift path.)doc")
      .def_readonly("toKey", &FiberTransportRead::toKey,
                    "Order-independent key of the destination fiber A.")
      .def_readonly("fromKey", &FiberTransportRead::fromKey,
                    "Order-independent key of the source fiber B.")
      .def_readonly("degree", &FiberTransportRead::degree)
      .def_readonly("rank", &FiberTransportRead::rank)
      .def_readonly("rawMap", &FiberTransportRead::rawMap,
                    "M_AB before any normalization.")
      .def_readonly("singularValues", &FiberTransportRead::singularValues,
                    "Singular values of rawMap, descending.")
      .def_readonly("numericalRank", &FiberTransportRead::numericalRank)
      .def_readonly("leakage", &FiberTransportRead::leakage,
                    "Regime-appropriate isometry defect (spec 5.5).")
      .def_readonly("overlapConditionNumber",
                    &FiberTransportRead::overlapConditionNumber)
      .def_readonly("toGap", &FiberTransportRead::toGap)
      .def_readonly("fromGap", &FiberTransportRead::fromGap)
      .def_readonly("toPositiveSignature",
                    &FiberTransportRead::toPositiveSignature)
      .def_readonly("toNegativeSignature",
                    &FiberTransportRead::toNegativeSignature)
      .def_readonly("fromPositiveSignature",
                    &FiberTransportRead::fromPositiveSignature)
      .def_readonly("fromNegativeSignature",
                    &FiberTransportRead::fromNegativeSignature)
      .def_readonly("toProjectorNorm", &FiberTransportRead::toProjectorNorm)
      .def_readonly("fromProjectorNorm",
                    &FiberTransportRead::fromProjectorNorm)
      .def_readonly("frameConditionNumber",
                    &FiberTransportRead::frameConditionNumber,
                    "max of the endpoints' FRAME condition numbers "
                    "(spec 6.6) -- distinct from the projector norms.")
      .def_readonly("regime", &FiberTransportRead::regime)
      .def_readonly("unitaryMap", &FiberTransportRead::unitaryMap,
                    "The emitted U(r)/pseudo-unitary factor; EMPTY when "
                    "rejected or on the certified GL(r,C) non-normal path.")
      .def_readonly("determinantPhase", &FiberTransportRead::determinantPhase,
                    "det of the emitted factor (U(1)); the raw determinant "
                    "phase on the GL path -- never discarded.")
      .def_readonly("polarResidual", &FiberTransportRead::polarResidual)
      .def_readonly("determinantResidual",
                    &FiberTransportRead::determinantResidual)
      .def_readonly("projectiveOnly", &FiberTransportRead::projectiveOnly)
      .def_readonly("accepted", &FiberTransportRead::accepted)
      .def_readonly("rejectionReason", &FiberTransportRead::rejectionReason)
      .def_readonly("certificate", &FiberTransportRead::certificate)
      .def("describe", &FiberTransportRead::describe)
      .def("__repr__", &FiberTransportRead::describe)
      .def("toRecord",
           [](const FiberTransportRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization (`transports`): "
           "at rank three the full U(3) factor, det V, and thereby the "
           "PU(3) class travel.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return FiberTransportRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<WilsonHolonomyRead>(m, "WilsonHolonomyRead",
      R"doc(The product of accepted transports around a loop: full U(r)
holonomy (or the certified GL(r,C) product),
normalized trace Tr H / r, determinant line det H, and the center-blind
adjoint reads.  Under independent local frame changes a CLOSED holonomy is
conjugated at its base component, so the normalized trace is invariant.)doc")
      .def_readonly("rank", &WilsonHolonomyRead::rank)
      .def_readonly("loopLength", &WilsonHolonomyRead::loopLength)
      .def_readonly("closed", &WilsonHolonomyRead::closed)
      .def_readonly("baseKey", &WilsonHolonomyRead::baseKey)
      .def_readonly("holonomy", &WilsonHolonomyRead::holonomy)
      .def_readonly("normalizedTrace", &WilsonHolonomyRead::normalizedTrace)
      .def_readonly("determinant", &WilsonHolonomyRead::determinant)
      .def_readonly("adjointTrace", &WilsonHolonomyRead::adjointTrace,
                    "|Tr H|^2 - 1 -- center-blind.")
      .def_readonly("adjointMatrix", &WilsonHolonomyRead::adjointMatrix,
                    "Rank 3 only: the faithful PU(3) image on the traceless "
                    "octet (ColorFiber::adjointOctetProjector conventions).")
      .def_readonly("unitarityResidual",
                    &WilsonHolonomyRead::unitarityResidual)
      .def_readonly("unitary", &WilsonHolonomyRead::unitary)
      .def_readonly("maxLeakage", &WilsonHolonomyRead::maxLeakage,
                    "Worst pre-normalization isometry defect over the links.")
      .def_readonly("minEndpointGap", &WilsonHolonomyRead::minEndpointGap,
                    "Worst endpoint band isolation over the links.")
      .def_readonly("maxFrameConditionNumber",
                    &WilsonHolonomyRead::maxFrameConditionNumber,
                    "Worst endpoint frame conditioning over the links.")
      .def_readonly("minSingularValue",
                    &WilsonHolonomyRead::minSingularValue,
                    "Smallest singular value over the links' raw overlaps.")
      .def_readonly("minNumericalRank",
                    &WilsonHolonomyRead::minNumericalRank,
                    "Smallest numerical rank over the links.")
      .def_readonly("certificate", &WilsonHolonomyRead::certificate);

  py::class_<FundamentalLiftRead>(m, "FundamentalLiftRead",
      R"doc(The explicitly lifted SU(3) fundamental holonomy: a cube-root
branch continued from a declared base
branch s0, lift = H exp(-i Theta/3) omega^{-s0} with Theta the accumulated
per-link principal determinant phase, and the accumulated Z3 center sector
RECORDED (branch-independent; the lift shifts by omega^{-s0} across
branches while every projective/adjoint read of it is branch-independent).
Rank three only -- SU(3) is never hard-coded at generic rank.)doc")
      .def_readonly("rank", &FundamentalLiftRead::rank)
      .def_readonly("baseBranch", &FundamentalLiftRead::baseBranch)
      .def_readonly("lift", &FundamentalLiftRead::lift)
      .def_readonly("liftTrace", &FundamentalLiftRead::liftTrace)
      .def_readonly("centerSector", &FundamentalLiftRead::centerSector)
      .def_readonly("accumulatedDeterminantPhase",
                    &FundamentalLiftRead::accumulatedDeterminantPhase)
      .def_readonly("maxDeterminantPhaseStep",
                    &FundamentalLiftRead::maxDeterminantPhaseStep)
      .def_readonly("detResidual", &FundamentalLiftRead::detResidual)
      .def_readonly("valid", &FundamentalLiftRead::valid)
      .def_readonly("invalidReason", &FundamentalLiftRead::invalidReason)
      .def_readonly("certificate", &FundamentalLiftRead::certificate)
      .def("toRecord",
           [](const FundamentalLiftRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the lift and its accumulated center "
           "sector travel together.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return FundamentalLiftRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<WindingClosureSpec> windingClosure(m, "WindingClosureSpec",
      R"doc(The declared closure of an open-segment determinant winding
HOW the open composite is closed is part of
the certificate.  Mode.NONE leaves the winding unknown -- a raw endpoint
phase difference is never promoted to an integer.)doc");
  py::enum_<WindingClosureSpec::Mode>(windingClosure, "Mode")
      .value("NONE", WindingClosureSpec::Mode::None)
      .value("MATCHED_REFERENCE", WindingClosureSpec::Mode::MatchedReference)
      .value("ENDPOINT_TRIVIALIZATION",
             WindingClosureSpec::Mode::EndpointTrivialization);
  windingClosure.def(py::init<>())
      .def_readwrite("mode", &WindingClosureSpec::mode)
      .def_readwrite("referenceId", &WindingClosureSpec::referenceId,
                     "Caller-supplied reference specification id, recorded "
                     "verbatim on the read.")
      .def_readwrite("referenceTransports",
                     &WindingClosureSpec::referenceTransports,
                     "MATCHED_REFERENCE: one reference transport per "
                     "segment sample (same orientation; traversed "
                     "backwards by the closure).")
      .def_readwrite("startTrivialization",
                     &WindingClosureSpec::startTrivialization)
      .def_readwrite("endTrivialization",
                     &WindingClosureSpec::endTrivialization);

  py::class_<DeterminantWindingRead>(m, "DeterminantWindingRead",
      R"doc(The integer determinant winding of a closed full-rank transport
family, or the RELATIVE winding of an open segment under a recorded
closure.  `winding` is None when
invalidated (closed gap / lost rank / aliasing step) or when no closure
was declared -- never a silently wrong integer.)doc")
      .def_readonly("winding", &DeterminantWindingRead::winding)
      .def_readonly("windingClosure", &DeterminantWindingRead::windingClosure)
      .def_readonly("windingReferenceId",
                    &DeterminantWindingRead::windingReferenceId)
      .def_readonly("accumulatedPhase",
                    &DeterminantWindingRead::accumulatedPhase)
      .def_readonly("maxPhaseStep", &DeterminantWindingRead::maxPhaseStep)
      .def_readonly("phaseStepMargin",
                    &DeterminantWindingRead::phaseStepMargin)
      .def_readonly("closureDefect", &DeterminantWindingRead::closureDefect)
      .def_readonly("invalidationReason",
                    &DeterminantWindingRead::invalidationReason)
      .def_readonly("certificate", &DeterminantWindingRead::certificate)
      .def("toRecord",
           [](const DeterminantWindingRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the closure specification travels "
           "with the integer; an unknown winding serializes as unknown, "
           "never as zero.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return DeterminantWindingRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<FiberConnection>(m, "FiberConnection",
      R"doc(Derived spectral-frame transport and Wilson observables (#770).
Wraps EXISTING induced-transfer
machinery -- the whole-complex Hodge d'Alembertian's intercomponent block
and RecursiveQuotient response-network blocks -- forms the overlap
M_AB = Phi_A^dagger W_A T_AB Phi_B (Psi_A^dagger on the biorthogonal
path), reports every diagnostic BEFORE normalization, gates, and only then
reduces to the polar U(r) / pseudo-unitary factor.  Composes accepted maps
into full U(r), determinant-line, projective/adjoint, and explicitly
lifted fundamental holonomies; certifies closed-family and declared
open-segment determinant windings.

Read-only observable: consumes accepted #769 SpectralFibers, mutates
nothing, and none of its outputs enters any emergence objective; the link
matrix is always reconstructed from neighboring Hodge frames with a
leakage certificate, never sampled independently.)doc")
      .def(py::init<FiberConnectionConfig>(),
           py::arg("config") = FiberConnectionConfig{})
      .def("config", &FiberConnection::config,
           py::return_value_policy::reference_internal)
      .def_static("chainTransfer",
                  [](const std::shared_ptr<Spacetime> &st, int degree,
                     const std::vector<std::vector<std::uint64_t>> &toCells,
                     const std::vector<std::vector<std::uint64_t>> &fromCells,
                     std::optional<cobordism::HodgeLaplacian::WeightConvention>
                         weights) {
                    return FiberConnection::chainTransfer(
                        st, degree, toCells, fromCells,
                        weights.value_or(cobordism::HodgeLaplacian::
                                             defaultWeightConvention()));
                  },
                  py::arg("st"), py::arg("degree"), py::arg("to_cells"),
                  py::arg("from_cells"), py::arg("weights") = py::none(),
                  "The chain transfer T_AB induced by the connecting "
                  "simplices: the off-diagonal block L_k[cells(to), "
                  "cells(from)] of the whole-complex weighted Hodge "
                  "operator, cells matched by sorted vertex-id tuple.  "
                  "weights = None follows the process-wide "
                  "HodgeWeightConvention at CALL time.")
      .def_static("responseTransfer", &FiberConnection::responseTransfer,
                  py::arg("network"), py::arg("to_component"),
                  py::arg("from_component"),
                  "The effective response block of an existing #768 "
                  "response network (rows = to's stalk, cols = from's "
                  "stalk; zero block when the network carries no such "
                  "edge).")
      .def("transport", &FiberConnection::transport, py::arg("to_fiber"),
           py::arg("from_fiber"), py::arg("transfer"),
           "Derive the transport A <- B from an explicit transfer block "
           "(rows = A's cells, cols = B's cells): overlap, full "
           "diagnostics, gates, then reduction, in that order.")
      .def("transportReverse", &FiberConnection::transportReverse,
           py::arg("to_fiber"), py::arg("from_fiber"), py::arg("transfer"),
           "The reverse-direction transport B <- A through the W-adjoint "
           "reverse block T_BA = W_B^{-1} T_AB^dagger W_A (exact in the "
           "W-self-adjoint regimes, where it returns the adjoint/inverse "
           "factor).")
      .def("transportOnSpacetime",
           [](const FiberConnection &self, const std::shared_ptr<Spacetime> &st,
              const SpectralFiber &to, const SpectralFiber &from,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w) {
             return self.transportOnSpacetime(
                 st, to, from,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()));
           },
           py::arg("st"), py::arg("to_fiber"), py::arg("from_fiber"),
           py::arg("weights") = py::none(),
           "Derive the transport on a spacetime: assembles the chain "
           "transfer from the Hodge operator, then transport().")
      .def("transportOnSpacetimeCached",
           [](const FiberConnection &self, cobordism::AnalyticCache &cache,
              const std::shared_ptr<Spacetime> &st, const SpectralFiber &to,
              const SpectralFiber &from,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w) {
             return self.transportOnSpacetimeCached(
                 cache, st, to, from,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()));
           },
           py::arg("cache"), py::arg("st"), py::arg("to_fiber"),
           py::arg("from_fiber"), py::arg("weights") = py::none(),
           "transportOnSpacetime through the #764 AnalyticCache contract "
           "(key: the union of the two fibers' cell-vertex sets; cached "
           "equals cold).")
      .def("holonomy", &FiberConnection::holonomy, py::arg("links"),
           "Multiply ACCEPTED transports along a chain; reports the full "
           "holonomy, normalized trace, determinant line, and adjoint "
           "reads (closed = the keys chain into a loop).")
      .def("holonomyOnSpacetime",
           [](const FiberConnection &self, const std::shared_ptr<Spacetime> &st,
              const std::vector<SpectralFiber> &fibers,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w) {
             return self.holonomyOnSpacetime(
                 st, fibers,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()));
           },
           py::arg("st"), py::arg("fibers"), py::arg("weights") = py::none(),
           "Wilson loop over an ordered cycle of fibers: links "
           "fibers[i] <- fibers[i+1] (wrapping), then the product.")
      .def("holonomyOnSpacetimeCached",
           [](const FiberConnection &self, cobordism::AnalyticCache &cache,
              const std::shared_ptr<Spacetime> &st,
              const std::vector<SpectralFiber> &fibers,
              std::optional<cobordism::HodgeLaplacian::WeightConvention> w) {
             return self.holonomyOnSpacetimeCached(
                 cache, st, fibers,
                 w.value_or(
                     cobordism::HodgeLaplacian::defaultWeightConvention()));
           },
           py::arg("cache"), py::arg("st"), py::arg("fibers"),
           py::arg("weights") = py::none(),
           "holonomyOnSpacetime through the AnalyticCache: per-link caching "
           "plus the loop product keyed by ALL participating fibers, so a "
           "published TouchedStar invalidates only the loops touching the "
           "changed star.")
      .def_static("projectiveRepresentative",
                  &FiberConnection::projectiveRepresentative,
                  py::arg("unitary"), py::arg("gate"),
                  "A canonical PU(3) class representative: V / (det "
                  "V)^{1/3} with the PRINCIPAL cube root (the class {U, "
                  "omega U, omega^2 U} is the faithful datum). GATED on the "
                  "triangle-anchor certificate: a closed AnchorGate raises, "
                  "because rank three plus an accepted transport is not a "
                  "licence to emit a colour datum.")
      .def_static("adjointRepresentation",
                  &FiberConnection::adjointRepresentation, py::arg("unitary"),
                  "The faithful PU(3) image of a 3x3 unitary on the "
                  "traceless octet (ColorFiber conventions; center-blind).")
      .def("fundamentalLift", &FiberConnection::fundamentalLift,
           py::arg("links"), py::arg("gate"), py::arg("base_branch") = 0,
           "Continue a cube-root branch along the links from the declared "
           "base branch and RECORD the accumulated Z3 center sector. GATED "
           "on the triangle-anchor certificate: a closed AnchorGate reports "
           "valid=False carrying the gate's own refusal reason.")
      .def("closedFamilyWinding", &FiberConnection::closedFamilyWinding,
           py::arg("family"),
           "Integer determinant winding of a CLOSED transport family "
           "(cyclic samples); invalidated when a gap/rank closes or a "
           "phase step reaches pi.")
      .def("openSegmentWinding", &FiberConnection::openSegmentWinding,
           py::arg("segment"), py::arg("closure"),
           "RELATIVE determinant winding of an OPEN cobordism segment "
           "under the declared closure (matched-reference or endpoint "
           "trivializations), with the specification recorded; unknown "
           "when no closure is declared.")
      .def_static("fiberKey", &FiberConnection::fiberKey, py::arg("fiber"),
                  "Order-independent key of a fiber (Fingerprint over its "
                  "deduplicated cell-vertex-id set).");

  // ---- ParticleClusters (#773): quark/antiquark classification ---------

  py::class_<ParticleClustersConfig>(m, "ParticleClustersConfig",
      R"doc(Analysis thresholds of the particle classification (#773).
Every value selects which reads are CERTIFIED, never which value is
reported, and the whole configuration is echoed on every read
(QuarkRead.thresholds).)doc")
      .def(py::init<>())
      .def_readwrite("parityTolerance",
                     &ParticleClustersConfig::parityTolerance,
                     "|<(-1)^N> -+ 1| cap for a definite parity sign.")
      .def_readwrite("occupationTolerance",
                     &ParticleClustersConfig::occupationTolerance,
                     "|<N> - 1| cap for the single-fermion occupation.")
      .def_readwrite("minAnchorScore",
                     &ParticleClustersConfig::minAnchorScore,
                     "Calibrated anchor atlas-score floor (a^2 in [0,1]).")
      .def_readwrite("minPhaseCoherence",
                     &ParticleClustersConfig::minPhaseCoherence,
                     "Determinant-phase coherence floor of the anchor.")
      .def_readwrite("maxTransportLeakage",
                     &ParticleClustersConfig::maxTransportLeakage,
                     "Cap on the worst lifetime transport leakage (#770).")
      .def_readwrite("minPersistenceLifetime",
                     &ParticleClustersConfig::minPersistenceLifetime,
                     "Minimum COBORDISM-FRAME lifetime (the whitepaper's "
                     "'lifetime across multiple cobordism frames'); the "
                     "modularity resolution-slice count never gates.")
      .def_readwrite("minPersistenceOverlap",
                     &ParticleClustersConfig::minPersistenceOverlap,
                     "Minimum adjacent-FRAME track overlap.")
      .def_readwrite("minLocalization",
                     &ParticleClustersConfig::minLocalization,
                     "Band-localization floor (0 accepts any MEASURED "
                     "localization; NaN still fails).")
      .def_readwrite("minRefinementOverlap",
                     &ParticleClustersConfig::minRefinementOverlap,
                     "Minimum band subspace overlap across a refinement.")
      .def_readwrite("minStabilityFrames",
                     &ParticleClustersConfig::minStabilityFrames,
                     "Frames a 'stable' quark condition must hold at (the "
                     "whitepaper's conditions two and three are "
                     "across-frame statements).")
      .def_readwrite("doubletOverlapThreshold",
                     &ParticleClustersConfig::doubletOverlapThreshold,
                     "Subspace-overlap threshold of the doublet tracking.")
      .def_readwrite("minDoubletFrames",
                     &ParticleClustersConfig::minDoubletFrames,
                     "Minimum frames a flavor subclass must persist.")
      .def_readwrite("isospinTolerance",
                     &ParticleClustersConfig::isospinTolerance,
                     "|I3 -+ 1/2| cap for a definite doublet member.")
      .def_readwrite("gaussTolerance",
                     &ParticleClustersConfig::gaussTolerance,
                     "Max nested-surface deviation (and |Im| leakage) for "
                     "a consistent Gauss flux.")
      .def_readwrite("minEnclosingSurfaces",
                     &ParticleClustersConfig::minEnclosingSurfaces,
                     "Minimum nested surfaces for a consistency claim.")
      .def_readwrite("udTolerance", &ParticleClustersConfig::udTolerance,
                     "|Q_gauss - (I3 + B/2)| cap for the proposed u/d "
                     "identification.")
      .def_readwrite("minOctetWeight",
                     &ParticleClustersConfig::minOctetWeight,
                     "#774: floor on a gluon candidate's octet Frobenius "
                     "weight (a genuinely nonzero color polarization).")
      .def_readwrite("octetPurityTolerance",
                     &ParticleClustersConfig::octetPurityTolerance,
                     "#774: cap on the (I9 - P8) residual of the excitation "
                     "(machine-level: the traceless bilinear is octet "
                     "exactly).")
      .def_readwrite("compositeOctetTolerance",
                     &ParticleClustersConfig::compositeOctetTolerance,
                     "#774: cap on the octet fraction of a meson's pair "
                     "color bilinear (the color-singlet certificate).")
      .def_readwrite("minAntiTripletWeight",
                     &ParticleClustersConfig::minAntiTripletWeight,
                     "#774: floor on the certified anti-triplet wedge "
                     "occupation det(C^dag Gamma C) of a diquark.")
      .def_readwrite("colorGramTolerance",
                     &ParticleClustersConfig::colorGramTolerance,
                     "#775: |det(C^dag C) - 1| cap of the color-singlet "
                     "certificate (exactly 1 for an orthonormal triad, "
                     "exactly 0 for duplicate color modes).")
      .def_readwrite("colorFluxTolerance",
                     &ParticleClustersConfig::colorFluxTolerance,
                     "#775: cap on the NET COLOR FLUX -- the octet weight "
                     "of the bound object's color bilinear.  An "
                     "INDEPENDENT finite-complex diagnostic, never on its "
                     "own a proof of confinement.")
      .def_readwrite("spinExpectationTolerance",
                     &ParticleClustersConfig::spinExpectationTolerance,
                     "#775: |<J^2> - 3/4| cap of the total-space spin "
                     "expectation.")
      .def_readwrite("spinVarianceTolerance",
                     &ParticleClustersConfig::spinVarianceTolerance,
                     "#775: |Var(J^2)| cap of the SHARP-spin certificate "
                     "value.")
      .def_readwrite("minSupportContainment",
                     &ParticleClustersConfig::minSupportContainment,
                     "#775: minimum fraction of a constituent's level-0 "
                     "support inside the supercomponent (1.0 = full).")
      .def_readwrite("minLifetimeOverlap",
                     &ParticleClustersConfig::minLifetimeOverlap,
                     "#775: minimum number of SHARED persistence slices "
                     "across the three constituents' lifetimes.")
      .def_readwrite("minRadius", &ParticleClustersConfig::minRadius,
                     "#775: strict floor a finite emergent radius must "
                     "exceed.")
      .def_readwrite("maxProfileDeviation",
                     &ParticleClustersConfig::maxProfileDeviation,
                     "#775: cap on the deviation of every DIMENSIONLESS "
                     "scale channel across the refinement window.");

  py::class_<GaussFluxRead>(m, "GaussFluxRead",
      R"doc(The electric Gauss-flux consistency read over nested enclosing
surfaces: each per-surface flux is the EXISTING
EigenstateSynthesis.gaussLawCharge value (an exact signed sum of the
supplied field-strength 2-cochain over the closed-star boundary,
restricted to electric/timelike-leg plaquettes when electricOnly).
Charge is certified only when consistent across at least
minEnclosingSurfaces surfaces; otherwise electricFlux is None (unknown),
never zero.  No metric regime is verified by the sum, so the certificate
carries the non-normal (no self-adjointness claimed) regime tag.)doc")
      .def(py::init<>())
      .def_readonly("fluxes", &GaussFluxRead::fluxes,
                    "Per-surface complex fluxes, in input surface order.")
      .def_readonly("surfaceVertexCounts",
                    &GaussFluxRead::surfaceVertexCounts,
                    "Distinct enclosed vertices per surface (nesting "
                    "witness).")
      .def_readonly("electricOnly", &GaussFluxRead::electricOnly)
      .def_readonly("maxDeviation", &GaussFluxRead::maxDeviation,
                    "Max |flux_i - flux_j| over surface pairs.")
      .def_readonly("imagLeakage", &GaussFluxRead::imagLeakage,
                    "Max |Im flux_i| (never silently discarded).")
      .def_readonly("consistent", &GaussFluxRead::consistent)
      .def_readonly("electricFlux", &GaussFluxRead::electricFlux,
                    "Re(mean) of the agreeing surfaces; None = unknown.")
      .def_readonly("failedCertificates",
                    &GaussFluxRead::failedCertificates)
      .def_readonly("certificate", &GaussFluxRead::certificate);

  py::class_<FlavorDoubletRead>(m, "FlavorDoubletRead",
      R"doc(The emergent, unlabeled, transported two-state spectral
subclass that could carry isospin.  The search
runs WITHOUT a requested dimension: stableSubclassRanks reports every
stable rank found, and "two-state" is an outcome.  The stored first-frame
doublet fiber is the RECORDED member trivialization -- a compilation
convention, never a physical u/d label.)doc")
      .def(py::init<>())
      .def_readonly("found", &FlavorDoubletRead::found,
                    "Exactly one stable two-state subclass emerged.")
      .def_readonly("degree", &FlavorDoubletRead::degree)
      .def_readonly("rank", &FlavorDoubletRead::rank,
                    "2 when found; never requested.")
      .def_readonly("framesTracked", &FlavorDoubletRead::framesTracked)
      .def_readonly("minContinuationOverlap",
                    &FlavorDoubletRead::minContinuationOverlap,
                    "Smallest certified continuation overlap on the track.")
      .def_readonly("minIsolation", &FlavorDoubletRead::minIsolation,
                    "Worst band isolation min(lowerGap, upperGap) along "
                    "the track.")
      .def_readonly("stableSubclassRanks",
                    &FlavorDoubletRead::stableSubclassRanks,
                    "Ranks of ALL stable subclasses (the no-requested-"
                    "dimension witness).")
      .def_readonly("twoStateCount", &FlavorDoubletRead::twoStateCount,
                    "Stable two-state subclasses (found needs exactly 1).")
      .def_readonly("doublet", &FlavorDoubletRead::doublet,
                    "First-frame fiber of the winning subclass (the "
                    "recorded trivialization).")
      .def_readonly("failedCertificates",
                    &FlavorDoubletRead::failedCertificates)
      .def_readonly("invalidationReason",
                    &FlavorDoubletRead::invalidationReason)
      .def_readonly("certificate", &FlavorDoubletRead::certificate);

  py::class_<QuarkCandidateEvidence>(m, "QuarkCandidateEvidence",
      R"doc(The assembled evidence bundle of one candidate -- every field
is a read PRODUCED BY the merged upstream kernels (#765 persistence, #769
bands, #767 anchors, #770 transports/windings, #780 Wick reads, the
existing Gauss read); the classifier never recomputes any of them.
Unsupplied evidence is MISSING evidence: the corresponding certificate
fails by name, never presumed to pass.)doc")
      .def(py::init<>())
      .def_readwrite("component", &QuarkCandidateEvidence::component,
                     "#765 label-free identity.")
      .def_readwrite("colorBand", &QuarkCandidateEvidence::colorBand,
                     "The selected #769 band (rank is read, never "
                     "requested).")
      .def_readwrite("colorBandFrames",
                     &QuarkCandidateEvidence::colorBandFrames,
                     "The band AT EACH cobordism frame -- whitepaper quark "
                     "condition two ('STABLE rank three') is decided here.")
      .def_readwrite("anchor", &QuarkCandidateEvidence::anchor,
                     "#767 calibrated anchor profile of the band.")
      .def_readwrite("anchorFrames", &QuarkCandidateEvidence::anchorFrames,
                     "The anchor profile AT EACH cobordism frame -- "
                     "whitepaper quark condition three (a STABLE profile "
                     "and determinant-line coherence) is decided here.")
      .def_readwrite("lifetimeTransports",
                     &QuarkCandidateEvidence::lifetimeTransports,
                     "#770 world-tube transports (all must be accepted).")
      .def_readwrite("winding", &QuarkCandidateEvidence::winding,
                     "#770 determinant-line winding with its RECORDED "
                     "closure specification.")
      .def_readwrite("parityRead", &QuarkCandidateEvidence::parityRead,
                     "#780 CovarianceState.wickParity of the carried "
                     "state.")
      .def_readwrite("occupationRead",
                     &QuarkCandidateEvidence::occupationRead,
                     "#780 CovarianceState.wickTotalNumber.")
      .def_readwrite("persistenceLifetime",
                     &QuarkCandidateEvidence::persistenceLifetime,
                     "#765 modularity RESOLUTION-slice lifetime "
                     "(REPORT-ONLY; NaN = missing).")
      .def_readwrite("persistenceMinOverlap",
                     &QuarkCandidateEvidence::persistenceMinOverlap,
                     "#765 smallest adjacent-SLICE overlap (REPORT-ONLY).")
      .def_readwrite("frameLifetime",
                     &QuarkCandidateEvidence::frameLifetime,
                     "COBORDISM-FRAME lifetime "
                     "(PersistentModularity.trackAcrossFrames) -- THE gated "
                     "persistence quantity.")
      .def_readwrite("frameMinOverlap",
                     &QuarkCandidateEvidence::frameMinOverlap,
                     "Smallest adjacent-FRAME support overlap -- the gated "
                     "predecessor/successor overlap.")
      .def_readwrite("refinementOverlap",
                     &QuarkCandidateEvidence::refinementOverlap,
                     "Band subspace overlap across a refinement "
                     "(SpectralFiber.overlap).")
      .def_readwrite("flavor", &QuarkCandidateEvidence::flavor,
                     "flavorDoubletSearch result; None = flavor unknown.")
      .def_readwrite("doubletOccupancy",
                     &QuarkCandidateEvidence::doubletOccupancy,
                     "Amplitudes on the two doublet members in the "
                     "recorded trivialization; None = unknown.")
      .def_readwrite("doubletOrientation",
                     &QuarkCandidateEvidence::doubletOrientation,
                     "Declared orientation s in {+1,-1}: which member "
                     "carries I3=+1/2 under the PROPOSED identification "
                     "(a recorded convention, never a hidden label).")
      .def_readwrite("charge", &QuarkCandidateEvidence::charge,
                     "gaussFluxOnSurfaces result; None = charge unknown.");

  py::class_<QuarkRead>(m, "QuarkRead",
      R"doc(The quark/antiquark particle read, plus the evidence summary the
classification
consumed, the recorded thresholds, and the #764 certificate.  Unknown or
uncertified values are None/NaN/0-sign, never zero-filled, and every gap
is NAMED in failedCertificates.  B = nu/3 exists exactly when the winding
certificate does; quark-ness additionally needs |nu| = 1.)doc")
      .def(py::init<>())
      .def_readonly("component", &QuarkRead::component)
      .def_readonly("exteriorParity", &QuarkRead::exteriorParity,
                    "-1 odd / +1 even / 0 unknown (an uncertified parity "
                    "read never emits a sign).")
      .def_readonly("colorRank", &QuarkRead::colorRank)
      .def_readonly("triangleAnchorScore", &QuarkRead::triangleAnchorScore)
      .def_readonly("triangleAnchorMaxTerm",
                    &QuarkRead::triangleAnchorMaxTerm)
      .def_readonly("triangleAnchorParticipation",
                    &QuarkRead::triangleAnchorParticipation)
      .def_readonly("anchorPhaseDispersion",
                    &QuarkRead::anchorPhaseDispersion)
      .def_readonly("anchorPhaseCoherence",
                    &QuarkRead::anchorPhaseCoherence)
      .def_readonly("anchorWeightingId", &QuarkRead::anchorWeightingId)
      .def_readonly("determinantWinding", &QuarkRead::determinantWinding,
                    "Certified nu; None when invalidated/unclosed.")
      .def_readonly("windingClosure", &QuarkRead::windingClosure,
                    "The recorded closure specification.")
      .def_readonly("windingReferenceId", &QuarkRead::windingReferenceId)
      .def_readonly("baryonFlux", &QuarkRead::baryonFlux,
                    "B = nu/3 under a certified winding; None = unknown, "
                    "never inserted.")
      .def_readonly("isospin", &QuarkRead::isospin,
                    "I3 = +-1/2 under the certified doublet hypothesis; "
                    "None = unknown.")
      .def_readonly("electricFlux", &QuarkRead::electricFlux,
                    "Gauss-consistent charge; None unless BOTH the Gauss "
                    "read and the flavor doublet are certified.")
      .def_readonly("confidence", &QuarkRead::confidence,
                    "Passed fraction of the ten core certificates.")
      .def_readonly("failedCertificates", &QuarkRead::failedCertificates,
                    "Every failed/missing certificate, by name.")
      .def_readonly("classification", &QuarkRead::classification,
                    "'quark' (nu=+1) / 'antiquark' (nu=-1) / 'none'.")
      .def_readonly("occupationTotal", &QuarkRead::occupationTotal)
      .def_readonly("transportCount", &QuarkRead::transportCount)
      .def_readonly("transportLeakageMax", &QuarkRead::transportLeakageMax)
      .def_readonly("persistenceLifetime", &QuarkRead::persistenceLifetime,
                    "Modularity RESOLUTION-slice lifetime (reported).")
      .def_readonly("persistenceMinOverlap",
                    &QuarkRead::persistenceMinOverlap)
      .def_readonly("frameLifetime", &QuarkRead::frameLifetime,
                    "COBORDISM-FRAME lifetime (the gated quantity).")
      .def_readonly("frameMinOverlap", &QuarkRead::frameMinOverlap)
      .def_readonly("stabilityFrames", &QuarkRead::stabilityFrames,
                    "Frames the stability certificates were measured over.")
      .def_readonly("anchorScoreSpread", &QuarkRead::anchorScoreSpread)
      .def_readonly("anchorCoherenceSpread",
                    &QuarkRead::anchorCoherenceSpread)
      .def_readonly("bandContinuationOverlap",
                    &QuarkRead::bandContinuationOverlap)
      .def_readonly("localization", &QuarkRead::localization)
      .def_readonly("localizationSupportFraction",
                    &QuarkRead::localizationSupportFraction)
      .def_readonly("refinementOverlap", &QuarkRead::refinementOverlap)
      .def_readonly("udIdentificationProposed",
                    &QuarkRead::udIdentificationProposed,
                    "Q = I3 + B/2 was tested AND held (the proposed u/d "
                    "identification, never a charge definition).")
      .def_readonly("doubletOrientation", &QuarkRead::doubletOrientation)
      .def_readonly("thresholds", &QuarkRead::thresholds,
                    "The configuration that produced this read.")
      .def_readonly("certificate", &QuarkRead::certificate)
      .def("describe", &QuarkRead::describe)
      .def("__repr__", &QuarkRead::describe)
      .def("toRecord",
           [](const QuarkRead &self) { return recordToPython(self.toRecord()); },
           "Checkpoint serialization (particles.quarks): fields, "
           "evidence summary, failed "
           "certificates, and the threshold echo; unknown values are "
           "null, never zero.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return QuarkRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<ConjugatePairRead>(m, "ConjugatePairRead",
      R"doc(Pair-conservation verification of a conjugate quark-antiquark
creation path: total certified winding, total baryon flux, and total
parity.  A singular (gap/rank-closing) leg leaves the totals UNKNOWN
(None) -- never zero by assumption.)doc")
      .def(py::init<>())
      .def_readonly("totalWinding", &ConjugatePairRead::totalWinding,
                    "nu_a + nu_b when both certified; None otherwise.")
      .def_readonly("totalBaryonFlux", &ConjugatePairRead::totalBaryonFlux,
                    "B_a + B_b when both known; None = unknown flux.")
      .def_readonly("totalParity", &ConjugatePairRead::totalParity,
                    "Product of certified parities; 0 = unknown.")
      .def_readonly("parityEven", &ConjugatePairRead::parityEven)
      .def_readonly("conserved", &ConjugatePairRead::conserved,
                    "Both windings certified, total 0, even parity.")
      .def_readonly("failedCertificates",
                    &ConjugatePairRead::failedCertificates)
      .def_readonly("certificate", &ConjugatePairRead::certificate);

  // ---- #774 even sectors: octet bilinear + gluon/meson/diquark ----------

  py::class_<OctetBilinearRead>(m, "OctetBilinearRead",
      R"doc(The #774 quasi-free traceless-bilinear (octet) read of three
declared color modes of a carried #780 CovarianceState: the bilinear
matrix M_ij = <a_i^dag a_j> (the transposed principal submatrix of
Gamma), its EXACT 1+8 split (delegated to ColorFiber.octetRead /
tracelessPart / adjointOctetProjector), the adjoint Casimir (= 3 for a
nonzero excitation, by C = 3 P8), the quartic-Wick color Casimir
expectation <sum_a dGamma(lambda_a/2)^2> (exactly 4/3 on the fundamental
and anti-triplet Slater states, 0 on the vacuum and full singlet), the
octet coordinates Tr(lambda_a M)/2, and the certified subset
occupation/parity.  Evaluated ON THE COVARIANCE (polynomial in the mode
count, no Fock vector); adding vacuum-embedded microscopic modes leaves
the read unchanged.  Unknown values are NaN / 0-sign, never zero.)doc")
      .def(py::init<>())
      .def_readwrite("colorModes", &OctetBilinearRead::colorModes,
                     "The three declared color modes (the recorded color "
                     "trivialization order).")
      .def_readonly("occupation", &OctetBilinearRead::occupation,
                    "Certified subset occupation <N_S>; NaN = unknown.")
      .def_readonly("subsetParity", &OctetBilinearRead::subsetParity,
                    "+1 / -1 / 0 = unknown or indefinite.")
      .def_readonly("bilinear", &OctetBilinearRead::bilinear,
                    "M_ij = <a_i^dag a_j> on the declared modes.")
      .def_readonly("octetComponent", &OctetBilinearRead::octetComponent,
                    "ColorFiber.tracelessPart(bilinear) -- the excitation.")
      .def_readonly("octetWeight", &OctetBilinearRead::octetWeight,
                    "||M - (tr M/3) I||_F^2 (ColorFiber.octetRead).")
      .def_readonly("singletWeight", &OctetBilinearRead::singletWeight,
                    "|tr M|^2 / 3.")
      .def_readonly("octetProjectorResidual",
                    &OctetBilinearRead::octetProjectorResidual,
                    "||(I9 - P8) vec(M8)|| / ||M8||_F -- rounding-level; "
                    "NaN when the excitation vanishes.")
      .def_readonly("casimir", &OctetBilinearRead::casimir,
                    "ColorFiber.adjointCasimir(octetComponent) in [0, 3].")
      .def_readonly("casimirExpectation",
                    &OctetBilinearRead::casimirExpectation,
                    "<sum_a dGamma(lambda_a/2)^2> by quartic Wick sums.")
      .def_readonly("gellMannComponents",
                    &OctetBilinearRead::gellMannComponents,
                    "Tr(lambda_a M)/2 for a = 1..8.")
      .def_readonly("residual", &OctetBilinearRead::residual,
                    "Max residual of the consumed #780 Wick reads.")
      .def_readonly("certificate", &OctetBilinearRead::certificate)
      .def("describe", &OctetBilinearRead::describe)
      .def("__repr__", &OctetBilinearRead::describe)
      .def("toRecord",
           [](const OctetBilinearRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization (complex leaves split _re/_im).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return OctetBilinearRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<GluonCandidateEvidence>(m, "GluonCandidateEvidence",
      R"doc(The assembled evidence bundle of one #774 gluon candidate
the quasi-free octet bilinear read of the
carried state, the #780 carried-state Wick parity/occupation, the #770
lifetime transports and determinant winding, and the #765 persistence
lifetime.  Missing evidence fails its certificate BY NAME.)doc")
      .def(py::init<>())
      .def_readwrite("component", &GluonCandidateEvidence::component,
                     "#765 label-free identity of the excitation.")
      .def_readwrite("bindingComponent",
                     &GluonCandidateEvidence::bindingComponent,
                     "The component the excitation is bound to (reported "
                     "verbatim -- the ticket's binding component).")
      .def_readwrite("octet", &GluonCandidateEvidence::octet,
                     "octetBilinearRead output of the carried state.")
      .def_readwrite("parityRead", &GluonCandidateEvidence::parityRead,
                     "#780 CovarianceState.wickParity of the WHOLE carried "
                     "state (the even-parity gate).")
      .def_readwrite("occupationRead",
                     &GluonCandidateEvidence::occupationRead,
                     "#780 wickTotalNumber (report-only).")
      .def_readwrite("lifetimeTransports",
                     &GluonCandidateEvidence::lifetimeTransports,
                     "#770 transports: accepted, rank three, leakage under "
                     "the cap (the accepted-octet-transport gate).")
      .def_readwrite("winding", &GluonCandidateEvidence::winding,
                     "#770 determinant winding -- a certified nu = 0 is the "
                     "zero-baryon-flux evidence.")
      .def_readwrite("persistenceLifetime",
                     &GluonCandidateEvidence::persistenceLifetime,
                     "Modularity RESOLUTION-slice lifetime (REPORT-ONLY).")
      .def_readwrite("frameLifetime",
                     &GluonCandidateEvidence::frameLifetime,
                     "COBORDISM-FRAME lifetime -- the gated quantity.");

  py::class_<GluonRead>(m, "GluonRead",
      R"doc(The #774 gluon-candidate read: a persistent transported octet
excitation with certified even parity and certified ZERO total
determinant winding / baryon flux.  classification is "gluon-candidate"
or "none" -- NEVER "gluon": no even octet excitation is claimed to be a
physical gluon.  Unknown values are None/NaN/0-sign, never zero-filled;
every gap is NAMED in failedCertificates ("parity-even",
"octet-excitation", "octet-purity", "octet-transport", "winding-zero",
"persistence").)doc")
      .def(py::init<>())
      .def_readonly("component", &GluonRead::component)
      .def_readonly("bindingComponent", &GluonRead::bindingComponent)
      .def_readonly("classification", &GluonRead::classification,
                    "'gluon-candidate' or 'none'.")
      .def_readonly("exteriorParity", &GluonRead::exteriorParity,
                    "+1 even / -1 odd / 0 unknown.")
      .def_readonly("occupationTotal", &GluonRead::occupationTotal)
      .def_readonly("casimir", &GluonRead::casimir,
                    "Flat consumed-scalar summary of the octet evidence "
                    "(one source of truth: the full OctetBilinearRead "
                    "travels on the evidence).")
      .def_readonly("casimirExpectation", &GluonRead::casimirExpectation,
                    "The quartic-Wick color Casimir expectation consumed.")
      .def_readonly("octetProjectorResidual",
                    &GluonRead::octetProjectorResidual)
      .def_readonly("octetWeight", &GluonRead::octetWeight)
      .def_readonly("singletWeight", &GluonRead::singletWeight)
      .def_readonly("determinantWinding", &GluonRead::determinantWinding,
                    "Certified nu (0 for a candidate); None = unknown.")
      .def_readonly("windingClosure", &GluonRead::windingClosure)
      .def_readonly("windingReferenceId", &GluonRead::windingReferenceId)
      .def_readonly("baryonFlux", &GluonRead::baryonFlux,
                    "0.0 is a CERTIFIED zero flux; None = unknown, never "
                    "zero by default.")
      .def_readonly("transportCount", &GluonRead::transportCount)
      .def_readonly("transportLeakageMax", &GluonRead::transportLeakageMax)
      .def_readonly("persistenceLifetime", &GluonRead::persistenceLifetime,
                    "Modularity RESOLUTION-slice lifetime (reported).")
      .def_readonly("frameLifetime", &GluonRead::frameLifetime,
                    "COBORDISM-FRAME lifetime (the gated quantity).")
      .def_readonly("confidence", &GluonRead::confidence,
                    "Passed fraction of the six gluon certificates.")
      .def_readonly("failedCertificates", &GluonRead::failedCertificates)
      .def_readonly("thresholds", &GluonRead::thresholds)
      .def_readonly("certificate", &GluonRead::certificate)
      .def("describe", &GluonRead::describe)
      .def("__repr__", &GluonRead::describe)
      .def("toRecord",
           [](const GluonRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization (particles.gluons).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return GluonRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<CompositeCandidateEvidence>(m, "CompositeCandidateEvidence",
      R"doc(The assembled evidence bundle of one #774 TWO-cluster
composite (meson/diquark; three-cluster composites belong to #775): the
two constituent #773 QuarkReads consumed VERBATIM, the carried composite
occupation, the meson-channel pair color bilinear, the diquark-channel
certified anti-triplet wedge read, composite transports, and the
composite persistence lifetime.)doc")
      .def(py::init<>())
      .def_readwrite("bindingComponent",
                     &CompositeCandidateEvidence::bindingComponent,
                     "The component binding the two clusters.")
      .def_readwrite("first", &CompositeCandidateEvidence::first,
                     "First constituent's #773 QuarkRead.")
      .def_readwrite("second", &CompositeCandidateEvidence::second,
                     "Second constituent's #773 QuarkRead.")
      .def_readwrite("occupationRead",
                     &CompositeCandidateEvidence::occupationRead,
                     "#780 wickTotalNumber of the carried composite state "
                     "(report-only).")
      .def_readwrite("colorPairing",
                     &CompositeCandidateEvidence::colorPairing,
                     "Meson channel: the 3x3 pair color bilinear in "
                     "3 x 3bar (singlet composite: M ~ I); None = missing "
                     "-- the color-singlet certificate fails by name.")
      .def_readwrite("antiTripletRead",
                     &CompositeCandidateEvidence::antiTripletRead,
                     "Diquark channel: the certified Lambda^2 C^3 wedge "
                     "occupation det(C^dag Gamma C) (#780 "
                     "wickGramDeterminant) -- exactly zero for duplicated "
                     "color modes (Pauli).")
      .def_readwrite("lifetimeTransports",
                     &CompositeCandidateEvidence::lifetimeTransports,
                     "#770 composite transports (report-only for the "
                     "two-cluster reads).")
      .def_readwrite("persistenceLifetime",
                     &CompositeCandidateEvidence::persistenceLifetime,
                     "#765 composite track lifetime (NaN = missing).");

  py::class_<MesonRead>(m, "MesonRead",
      R"doc(The #774 meson-candidate read: one certified quark plus one
certified antiquark (order-insensitive), EVEN composite parity (the
exact graded product of the certified constituent parities -- the
whitepaper parity table), a color-SINGLET pair bilinear under the exact
1+8 split, and zero total certified winding/baryon flux (the #773
conjugate-pair integer sums).  failedCertificates vocabulary:
"constituent-quark", "constituent-antiquark", "parity-even",
"color-singlet", "flux-zero".)doc")
      .def(py::init<>())
      .def_readonly("bindingComponent", &MesonRead::bindingComponent)
      .def_readonly("firstConstituent", &MesonRead::firstConstituent)
      .def_readonly("secondConstituent", &MesonRead::secondConstituent)
      .def_readonly("classification", &MesonRead::classification,
                    "'meson-candidate' or 'none'.")
      .def_readonly("exteriorParity", &MesonRead::exteriorParity,
                    "Exact constituent-parity product; 0 = unknown.")
      .def_readonly("occupationTotal", &MesonRead::occupationTotal)
      .def_readonly("pairingSingletWeight",
                    &MesonRead::pairingSingletWeight)
      .def_readonly("pairingOctetWeight", &MesonRead::pairingOctetWeight)
      .def_readonly("pairingOctetFraction",
                    &MesonRead::pairingOctetFraction,
                    "octet/(octet+singlet) of the pairing; NaN = missing.")
      .def_readonly("totalWinding", &MesonRead::totalWinding,
                    "nu1 + nu2 when both certified; None = unknown.")
      .def_readonly("totalBaryonFlux", &MesonRead::totalBaryonFlux,
                    "B1 + B2 when both known; None = unknown, never zero.")
      .def_readonly("transportCount", &MesonRead::transportCount)
      .def_readonly("transportLeakageMax", &MesonRead::transportLeakageMax)
      .def_readonly("persistenceLifetime", &MesonRead::persistenceLifetime)
      .def_readonly("confidence", &MesonRead::confidence)
      .def_readonly("failedCertificates", &MesonRead::failedCertificates)
      .def_readonly("thresholds", &MesonRead::thresholds)
      .def_readonly("certificate", &MesonRead::certificate)
      .def("describe", &MesonRead::describe)
      .def("__repr__", &MesonRead::describe)
      .def("toRecord",
           [](const MesonRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return MesonRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<DiquarkRead>(m, "DiquarkRead",
      R"doc(The #774 diquark-candidate read: TWO certified quarks
(nu = +1 each), EVEN composite parity, a certified anti-triplet wedge
occupation, and the PRESERVED constituent baryon flux B = 2/3.
Explicitly NOT an antiquark: the 3bar color representation coincides,
but occupation TWO, EVEN parity, and B = +2/3 (vs one/odd/-1/3) are the
recorded distinction channels.  failedCertificates vocabulary:
"constituent-quarks", "parity-even", "anti-triplet",
"baryon-flux-two-thirds".)doc")
      .def(py::init<>())
      .def_readonly("bindingComponent", &DiquarkRead::bindingComponent)
      .def_readonly("firstConstituent", &DiquarkRead::firstConstituent)
      .def_readonly("secondConstituent", &DiquarkRead::secondConstituent)
      .def_readonly("classification", &DiquarkRead::classification,
                    "'diquark-candidate' or 'none'.")
      .def_readonly("exteriorParity", &DiquarkRead::exteriorParity,
                    "Exact constituent-parity product; 0 = unknown.")
      .def_readonly("occupationTotal", &DiquarkRead::occupationTotal)
      .def_readonly("antiTripletWeight", &DiquarkRead::antiTripletWeight,
                    "Certified wedge occupation; NaN = unknown.")
      .def_readonly("totalWinding", &DiquarkRead::totalWinding,
                    "nu1 + nu2 when both certified (2 for a candidate).")
      .def_readonly("totalBaryonFlux", &DiquarkRead::totalBaryonFlux,
                    "B1 + B2 (2/3 for a candidate); None = unknown.")
      .def_readonly("transportCount", &DiquarkRead::transportCount)
      .def_readonly("transportLeakageMax",
                    &DiquarkRead::transportLeakageMax)
      .def_readonly("persistenceLifetime",
                    &DiquarkRead::persistenceLifetime)
      .def_readonly("confidence", &DiquarkRead::confidence)
      .def_readonly("failedCertificates", &DiquarkRead::failedCertificates)
      .def_readonly("thresholds", &DiquarkRead::thresholds)
      .def_readonly("certificate", &DiquarkRead::certificate)
      .def("describe", &DiquarkRead::describe)
      .def("__repr__", &DiquarkRead::describe)
      .def("toRecord",
           [](const DiquarkRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return DiquarkRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<BoundCandidateEvidence>(m, "BoundCandidateEvidence",
      R"doc(One constituent's datum for the #775 bound-supercomponent
search: the #773 quark verdict, the #765 level-0
support, the #765 persistence window (first, last) -- None = no lifetime
evidence, the overlap certificate then fails by name -- and the #770
mutual transports to the other constituents.)doc")
      .def(py::init<>())
      .def_readwrite("quark", &BoundCandidateEvidence::quark,
                     "The candidate's #773 QuarkRead (only a CERTIFIED "
                     "'quark' verdict counts toward the three-quark "
                     "census).")
      .def_readwrite("support", &BoundCandidateEvidence::support,
                     "Level-0 cell support (ComponentRead.support); empty "
                     "= missing evidence.")
      .def_readwrite("lifetime", &BoundCandidateEvidence::lifetime,
                     "(firstSlice, lastSlice) of the #765 PersistenceTrack "
                     "window, inclusive; None = unknown.")
      .def_readwrite("mutualTransports",
                     &BoundCandidateEvidence::mutualTransports,
                     "#770 transports to the other constituents; every "
                     "supplied link must be accepted under the leakage "
                     "cap.");

  py::class_<BoundSupercomponentRead>(m, "BoundSupercomponentRead",
      R"doc(One next-modular-level component examined by the #775
bound-supercomponent search: the contained certified quark candidates,
their shared #765 lifetime window, and the containment/transport
certificates.  failedCertificates vocabulary: "supercomponent-level",
"quark-count", "support-containment", "lifetime-overlap",
"transport-containment".)doc")
      .def(py::init<>())
      .def_readonly("boundComponent",
                    &BoundSupercomponentRead::boundComponent)
      .def_readonly("quarks", &BoundSupercomponentRead::quarks,
                    "Contained certified quark candidates' #765 ids.")
      .def_readonly("quarkIndices", &BoundSupercomponentRead::quarkIndices,
                    "Their indices in the input candidate list.")
      .def_readonly("found", &BoundSupercomponentRead::found,
                    "A certified bound supercomponent of exactly three "
                    "lifetime-overlapping certified quark candidates.")
      .def_readonly("lifetimeWindow",
                    &BoundSupercomponentRead::lifetimeWindow,
                    "Shared (first, last) window; None = disjoint/unknown.")
      .def_readonly("lifetimeOverlap",
                    &BoundSupercomponentRead::lifetimeOverlap,
                    "Number of SHARED persistence slices.")
      .def_readonly("minContainment",
                    &BoundSupercomponentRead::minContainment,
                    "Smallest per-constituent support-containment "
                    "fraction; NaN = unknown.")
      .def_readonly("transportLeakageMax",
                    &BoundSupercomponentRead::transportLeakageMax)
      .def_readonly("transportCount",
                    &BoundSupercomponentRead::transportCount)
      .def_readonly("failedCertificates",
                    &BoundSupercomponentRead::failedCertificates)
      .def_readonly("thresholds", &BoundSupercomponentRead::thresholds)
      .def_readonly("certificate", &BoundSupercomponentRead::certificate)
      .def("describe", &BoundSupercomponentRead::describe)
      .def("__repr__", &BoundSupercomponentRead::describe);

  py::class_<ScaleProfileSample>(m, "ScaleProfileSample",
      R"doc(One refinement-window sample of the EXISTING #575/#566/#593
mass-radius battery (InteriorHinges).  radialWeightProfile is the
per-BFS-shell share of the |Re eps * star h| curvature weight -- a radial
CURVATURE-WEIGHT density, NOT a momentum-transfer form factor: no Fourier
transform of a charge density is computed anywhere in this tree.)doc")
      .def(py::init<>())
      .def_readwrite("radius", &ScaleProfileSample::radius,
                     "r = V_dual^(1/4) (InteriorHinges.Radii.rDual) -- "
                     "DIMENSIONFUL; only its finiteness is certified.")
      .def_readwrite("radiusCrossCheck",
                     &ScaleProfileSample::radiusCrossCheck,
                     "r = V_primal^(1/4); its RATIO to radius is the "
                     "dimensionless channel.")
      .def_readwrite("spectralMass", &ScaleProfileSample::spectralMass,
                     "The intensive shell mass m_shell -- a mean interior "
                     "deficit ANGLE, dimensionless in lattice units.")
      .def_readwrite("localization", &ScaleProfileSample::localization,
                     "Curvature-weight participation ratio (dimensionless).")
      .def_readwrite("radialWeightProfile",
                     &ScaleProfileSample::radialWeightProfile,
                     "Per-shell curvature-weight shares, shell ascending "
                     "(dimensionless); empty = no shell seeds, profile "
                     "UNKNOWN.")
      .def_readwrite("colorGramDeterminant",
                     &ScaleProfileSample::colorGramDeterminant,
                     "det(C^dag C) at this refinement.")
      .def_readwrite("rotationCharacter",
                     &ScaleProfileSample::rotationCharacter,
                     "The #772 2pi rotation character at this refinement.")
      .def_readwrite("baryonFlux", &ScaleProfileSample::baryonFlux,
                     "B = nu/3 at this refinement.")
      .def_readwrite("electricFlux", &ScaleProfileSample::electricFlux,
                     "Summed certified Gauss flux at this refinement.")
      .def_readwrite("compositeParity",
                     &ScaleProfileSample::compositeParity,
                     "-1 odd / +1 even / 0 unknown at this refinement "
                     "(an INTEGER channel: stability is exact equality).")
      .def_readwrite("anchorScore", &ScaleProfileSample::anchorScore,
                     "Worst constituent anchor score at this refinement.");

  py::class_<ScaleProfileRead>(m, "ScaleProfileRead",
      R"doc(The #775 refinement-window certificate: a FINITE emergent
radius plus the refinement stability of every DIMENSIONLESS channel.
physicalMass is ALWAYS None -- a dimensionful mass stays unknown until a
physical scale is independently established.  failedCertificates
vocabulary: "refinement-window", "finite-radius",
"radius-ratio-stability", "spectral-mass-stability",
"localization-stability", "profile-stability".)doc")
      .def(py::init<>())
      .def_readonly("sampleCount", &ScaleProfileRead::sampleCount)
      .def_readonly("radius", &ScaleProfileRead::radius)
      .def_readonly("radiusFinite", &ScaleProfileRead::radiusFinite)
      .def_readonly("radiusRatio", &ScaleProfileRead::radiusRatio)
      .def_readonly("radiusRatioSpread",
                    &ScaleProfileRead::radiusRatioSpread)
      .def_readonly("spectralMass", &ScaleProfileRead::spectralMass)
      .def_readonly("spectralMassSpread",
                    &ScaleProfileRead::spectralMassSpread)
      .def_readonly("localization", &ScaleProfileRead::localization)
      .def_readonly("localizationSpread",
                    &ScaleProfileRead::localizationSpread)
      .def_readonly("profileMaxDeviation",
                    &ScaleProfileRead::profileMaxDeviation,
                    "Max absolute per-shell deviation across the window; "
                    "NaN = unknown, never zero.")
      .def_readonly("profileShells", &ScaleProfileRead::profileShells)
      .def_readonly("colorGramDeterminant",
                    &ScaleProfileRead::colorGramDeterminant)
      .def_readonly("colorGramSpread", &ScaleProfileRead::colorGramSpread)
      .def_readonly("rotationCharacter",
                    &ScaleProfileRead::rotationCharacter)
      .def_readonly("rotationCharacterSpread",
                    &ScaleProfileRead::rotationCharacterSpread)
      .def_readonly("baryonFlux", &ScaleProfileRead::baryonFlux)
      .def_readonly("baryonFluxSpread", &ScaleProfileRead::baryonFluxSpread)
      .def_readonly("electricFlux", &ScaleProfileRead::electricFlux)
      .def_readonly("electricFluxSpread",
                    &ScaleProfileRead::electricFluxSpread)
      .def_readonly("compositeParity", &ScaleProfileRead::compositeParity)
      .def_readonly("compositeParityStable",
                    &ScaleProfileRead::compositeParityStable)
      .def_readonly("anchorScore", &ScaleProfileRead::anchorScore)
      .def_readonly("anchorScoreSpread",
                    &ScaleProfileRead::anchorScoreSpread)
      .def_readonly("physicalMass", &ScaleProfileRead::physicalMass,
                    "ALWAYS None: unknown until a physical scale is "
                    "independently established.")
      .def_readonly("stable", &ScaleProfileRead::stable)
      .def_readonly("failedCertificates",
                    &ScaleProfileRead::failedCertificates)
      .def_readonly("thresholds", &ScaleProfileRead::thresholds)
      .def_readonly("certificate", &ScaleProfileRead::certificate)
      .def("describe", &ScaleProfileRead::describe)
      .def("__repr__", &ScaleProfileRead::describe);

  py::class_<BaryonCandidateEvidence>(m, "BaryonCandidateEvidence",
      R"doc(The assembled evidence bundle of ONE #775 three-cluster
candidate: the three #773 constituent verdicts consumed VERBATIM, the
bound-supercomponent search result, the three normalized anchored color
columns (the wedge is built ONCE from them), the #774 octet bilinear read
of the bound object (the INDEPENDENT net-color-flux diagnostic), the #772
Berry-cancelled 2pi rotation character and optional Spin(d) lift, the
#780 Wick <J^2> and Var(J^2), the accepted covariance-only class's
variance reads, and the refinement-window mass-radius samples.)doc")
      .def(py::init<>())
      .def_readwrite("boundComponent",
                     &BaryonCandidateEvidence::boundComponent)
      .def_readwrite("quarks", &BaryonCandidateEvidence::quarks,
                     "The three constituents' #773 QuarkReads.  Assign the "
                     "whole list (ev.quarks = [a, b, c]): like every "
                     "std::array/std::vector binding, reading it yields a "
                     "COPY, so item assignment does not stick.")
      .def_readwrite("binding", &BaryonCandidateEvidence::binding,
                     "The boundSupercomponentSearch result.")
      .def_readwrite("colorColumns", &BaryonCandidateEvidence::colorColumns,
                     "The 3x3 matrix of normalized anchored color columns "
                     "C = [c_A c_B c_C]; the three-mode wedge is built "
                     "ONCE from it -- no extra fermion sign is multiplied "
                     "onto the color epsilon.")
      .def_readwrite("colorFlux", &BaryonCandidateEvidence::colorFlux,
                     "The bound object's OctetBilinearRead -- the "
                     "INDEPENDENT net-color-flux diagnostic.")
      .def_readwrite("rotation", &BaryonCandidateEvidence::rotation,
                     "#772 PhysicalRotation character of the closed 2pi "
                     "total-space cluster-frame cycle.")
      .def_readwrite("exchange", &BaryonCandidateEvidence::exchange,
                     "The #772 PARTICLE-EXCHANGE character, when the "
                     "exchange experiment was run.  REPORT-ONLY: neither "
                     "the ticket's proton-certificate list nor spec 16.4 "
                     "has an exchange row, so it gates nothing.")
      .def_readwrite("continuumSpinClaim",
                     &BaryonCandidateEvidence::continuumSpinClaim,
                     "When True the SO(d)->Spin(d) lift is REQUIRED; when "
                     "False it is never demanded (spec 16.4).")
      .def_readwrite("spinLift", &BaryonCandidateEvidence::spinLift,
                     "#772 spinLift decision; None = none made.")
      .def_readwrite("spinSquaredRead",
                     &BaryonCandidateEvidence::spinSquaredRead,
                     "#780 wickSpinSquaredExpectation of the carried "
                     "quasi-free state.")
      .def_readwrite("spinVarianceRead",
                     &BaryonCandidateEvidence::spinVarianceRead,
                     "#780 wickSpinSquaredVariance -- the sharp-spin "
                     "certificate.")
      .def_readwrite("classVarianceReads",
                     &BaryonCandidateEvidence::classVarianceReads,
                     "Var(J^2) of every candidate of the ACCEPTED "
                     "covariance-only class; empty/uncertified = the class "
                     "was NOT swept, so a variance failure is an unknown, "
                     "never an obstruction.")
      .def_readwrite("totalSpaceJ2", &BaryonCandidateEvidence::totalSpaceJ2,
                     "The #772 DENSE ExchangeHolonomy.totalJSquared "
                     "oracle, consulted only when the #780 Wick "
                     "expectation is absent; it never supplies a variance.")
      .def_readwrite("scaleSamples", &BaryonCandidateEvidence::scaleSamples,
                     "Refinement-window ScaleProfileSamples.")
      .def_readwrite("persistenceLifetime",
                     &BaryonCandidateEvidence::persistenceLifetime,
                     "#765 lifetime of the BOUND component (report-only).")
      .def_readwrite("lifetimeTransports",
                     &BaryonCandidateEvidence::lifetimeTransports,
                     "#770 composite transports (report-only).")
      .def_readwrite("crossingMass", &BaryonCandidateEvidence::crossingMass,
                     "The whitepaper's world-tube crossing mass for this "
                     "candidate.  None = the crossing-readouts gate passes "
                     "VACUOUSLY (applicable-gated like spin-lift); supplied, "
                     "it is ENFORCED together with crossingBaryon.")
      .def_readwrite("crossingBaryon",
                     &BaryonCandidateEvidence::crossingBaryon,
                     "The coherent one-third baryon sum for the same "
                     "candidate and level.  Must travel WITH crossingMass: a "
                     "half bundle fails the gate by name rather than grading "
                     "half a certificate.");

  py::class_<BaryonRead>(m, "BaryonRead",
      R"doc(The #775 three-quark baryon read and complete proton
certificate.  classification is one of
"no-baryon", "baryon-candidate", "certified-proton", or
"quasi-free-sharp-spin-obstruction" (the hyphenated spelling of the
spec's quasi_free_sharp_spin_obstruction).

failedCertificates vocabulary -- the two STRUCTURAL gates first (a
failure of either is "no-baryon"): "constituent-quarks",
"bound-supercomponent"; then the proton gates: "color-singlet",
"color-flux-zero", "baryon-flux-unit", "composite-parity-odd",
"flavor-uud", "electric-flux-unit", "spin-expectation", "sharp-spin",
"rotation-character", "spin-lift", "finite-radius", "profile-stability",
"crossing-readouts".

Unknown values are None/NaN/0-sign, never zero-filled; physicalMass is
ALWAYS None.)doc")
      .def(py::init<>())
      .def_readonly("quarks", &BaryonRead::quarks,
                    "The three constituents' #765 ids, in evidence order.")
      .def_readonly("boundComponent", &BaryonRead::boundComponent)
      .def_readonly("colorGramDeterminant",
                    &BaryonRead::colorGramDeterminant,
                    "det(C^dag C) = |det C|^2; NaN = no color evidence.")
      .def_readonly("colorFlux", &BaryonRead::colorFlux,
                    "The NET COLOR FLUX diagnostic (octet weight of the "
                    "bound object's color bilinear); NaN = unknown.  An "
                    "independent finite-complex diagnostic -- never on its "
                    "own a proof of confinement.")
      .def_readonly("baryonFlux", &BaryonRead::baryonFlux,
                    "B = nu/3 over the three certified windings (+1 for a "
                    "proton); None = unknown, never zero.")
      .def_readonly("electricFlux", &BaryonRead::electricFlux,
                    "Summed certified constituent Gauss fluxes (+1 for a "
                    "proton); None = unknown.")
      .def_readonly("totalJ2", &BaryonRead::totalJ2,
                    "Certified total-space <J^2> (3/4 proton, 15/4 Delta); "
                    "None = unknown.")
      .def_readonly("totalJ2Variance", &BaryonRead::totalJ2Variance,
                    "Certified Var(J^2); None = UNKNOWN, never zero and "
                    "never inferred from the expectation.")
      .def_readonly("rotationCharacter", &BaryonRead::rotationCharacter,
                    "The #772 Berry-cancelled 2pi character; None = "
                    "uncertified.")
      .def_readonly("classification", &BaryonRead::classification)
      .def_readonly("persistence", &BaryonRead::persistence)
      .def_readonly("failedCertificates", &BaryonRead::failedCertificates)
      .def_readonly("colorWedge", &BaryonRead::colorWedge,
                    "S_ABC = det[c_A c_B c_C], built ONCE.  A constituent "
                    "transposition flips this sign and leaves "
                    "colorGramDeterminant invariant.")
      .def_readonly("totalWinding", &BaryonRead::totalWinding,
                    "nu = nu_A + nu_B + nu_C (3 for a proton); None = "
                    "unknown.")
      .def_readonly("exteriorParity", &BaryonRead::exteriorParity,
                    "Exact graded product of the constituent parities; "
                    "0 = unknown.")
      .def_readonly("flavorPattern", &BaryonRead::flavorPattern,
                    "Certified isospin occupation pattern in canonical "
                    "order ('uud', ...); '' = unknown.")
      .def_readonly("totalIsospin", &BaryonRead::totalIsospin)
      .def_readonly("rotationCharacterSign",
                    &BaryonRead::rotationCharacterSign)
      .def_readonly("exchangeCharacter", &BaryonRead::exchangeCharacter,
                    "The #772 Berry-cancelled exchange character; None "
                    "unless a certified, correctly tagged exchange read "
                    "was supplied.  REPORT-ONLY.")
      .def_readonly("spinStatisticsRatio", &BaryonRead::spinStatisticsRatio,
                    "chi(exchange) * chi(2pi)^-1 (+1 on a spin-1/2 "
                    "fixture, each factor separately near -1); None unless "
                    "BOTH channels certified.  REPORT-ONLY.")
      .def_readonly("spinLiftApplicable", &BaryonRead::spinLiftApplicable)
      .def_readonly("spinLiftAccepted", &BaryonRead::spinLiftAccepted)
      .def_readonly("sharpSpin", &BaryonRead::sharpSpin)
      .def_readonly("quasiFreeClassSwept",
                    &BaryonRead::quasiFreeClassSwept,
                    "Whether the accepted covariance-only class was swept "
                    "-- the premise the obstruction verdict quantifies "
                    "over.")
      .def_readonly("classVarianceFloor", &BaryonRead::classVarianceFloor,
                    "min |Var(J^2)| over the swept class; NaN = not swept.")
      .def_readonly("radius", &BaryonRead::radius)
      .def_readonly("radiusFinite", &BaryonRead::radiusFinite)
      .def_readonly("spectralMass", &BaryonRead::spectralMass)
      .def_readonly("radiusRatio", &BaryonRead::radiusRatio)
      .def_readonly("profileMaxDeviation",
                    &BaryonRead::profileMaxDeviation)
      .def_readonly("profileStable", &BaryonRead::profileStable)
      .def_readonly("physicalMass", &BaryonRead::physicalMass,
                    "ALWAYS None (see ScaleProfileRead.physicalMass).")
      .def_readonly("crossingMassApplicable",
                    &BaryonRead::crossingMassApplicable,
                    "False when the caller supplied no world-tube crossing "
                    "evidence; the crossing-readouts gate then passed "
                    "VACUOUSLY, exactly like spin-lift.")
      .def_readonly("crossingMassValue", &BaryonRead::crossingMassValue,
                    "The whitepaper's crossing mass m_x as a difference "
                    "against M0.  UNCALIBRATED by default: ratio-only, never "
                    "a physical mass.  NaN without crossing evidence.")
      .def_readonly("crossingBaryonNumber", &BaryonRead::crossingBaryonNumber,
                    "The coherent one-third crossing sum; None when no "
                    "crossing evidence was supplied (unknown, never zero).")
      .def_readonly("crossingSignDefects", &BaryonRead::crossingSignDefects,
                    "Tubes whose crossing sign disagreed with their "
                    "determinant-line winding -- a defect signal.")
      .def_readonly("lifetimeOverlap", &BaryonRead::lifetimeOverlap)
      .def_readonly("transportCount", &BaryonRead::transportCount)
      .def_readonly("transportLeakageMax",
                    &BaryonRead::transportLeakageMax)
      .def_readonly("confidence", &BaryonRead::confidence,
                    "Passed-fraction of the fifteen certificates; 1.0 "
                    "exactly for a certified proton.")
      .def_readonly("thresholds", &BaryonRead::thresholds)
      .def_readonly("certificate", &BaryonRead::certificate)
      .def("describe", &BaryonRead::describe)
      .def("__repr__", &BaryonRead::describe)
      .def("toRecord",
           [](const BaryonRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization.")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return BaryonRead::fromRecord(pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate; rejects an unknown schema_version.");

  py::class_<ParticleClusters>(m, "ParticleClusters",
      R"doc(The #773 quark/antiquark classifier over persistent modular
spectral components (whitepaper "Quarks as modular clusters").  Composes
the merged Wave 1/2 certificates -- #765
persistence, #769 bands/tracking, #767 anchors, #770 transports and
determinant windings with recorded closures, #780 Wick parity/occupation,
and the EXISTING Gauss-flux read -- into QuarkReads; its own claim is the
exact boolean combination (StructureExact given the consumed held
certificates).

Certificate name vocabulary (failedCertificates): "persistence",
"localization", "parity-odd", "occupation-one", "color-rank-three",
"anchor", "transport-leakage", "winding", "winding-unit",
"refinement-stability" (the ten core gates), then "flavor-doublet",
"isospin", "gauss-consistency", "ud-identification" (flavor/charge gates
that never veto quark-ness -- they only leave their own fields unknown).

Read-only observable: never calls a solver, never mutates the spacetime,
and no output enters any emergence objective.  No "quark = hole", no
hard-coded u/d labels, no baryon number without determinant-winding
evidence.)doc")
      .def(py::init<ParticleClustersConfig>(),
           py::arg("config") = ParticleClustersConfig{})
      .def("config", &ParticleClusters::config,
           py::return_value_policy::reference_internal)
      .def("classifyQuark", &ParticleClusters::classifyQuark,
           py::arg("evidence"),
           "Classify one candidate from its assembled evidence: the ten "
           "core certificates, quark vs antiquark from the determinant-"
           "line orientation, B = nu/3 under the certified winding, and "
           "isospin/charge from their own independent certificates.  "
           "Missing evidence is a NAMED failed certificate, never an "
           "error.")
      .def("classifyQuarks", &ParticleClusters::classifyQuarks,
           py::arg("candidates"),
           "classifyQuark over a candidate stream, in input order.")
      .def("classifyQuarkCached", &ParticleClusters::classifyQuarkCached,
           py::arg("cache"), py::arg("evidence"),
           "classifyQuark through the #764 AnalyticCache contract (key: "
           "the color band's cell-vertex set; parameter: the evidence "
           "fingerprint).  Cached equals cold.")
      .def("evidenceFingerprint", &ParticleClusters::evidenceFingerprint,
           py::arg("evidence"),
           "Content fingerprint of the decision-relevant evidence AND the "
           "thresholds (the cache parameter).")
      .def("conjugatePair", &ParticleClusters::conjugatePair,
           py::arg("first"), py::arg("second"),
           "Verify pair conservation of a conjugate creation path from "
           "the two endpoint reads; a singular leg leaves the totals "
           "unknown.")
      .def("flavorDoubletSearch", &ParticleClusters::flavorDoubletSearch,
           py::arg("frames"),
           "Search the candidate's band enumeration across frames for a "
           "stable transported two-state subclass (certified #769 "
           "continuations, unambiguous, full length).  No dimension is "
           "ever requested; every stable rank is reported.")
      .def("gaussFluxOnSurfaces", &ParticleClusters::gaussFluxOnSurfaces,
           py::arg("st"), py::arg("field_strength"),
           py::arg("enclosed_vertex_sets"), py::arg("electric_only") = true,
           "The EXISTING Gauss-flux read "
           "(EigenstateSynthesis.gaussLawCharge) on nested enclosing "
           "surfaces, then the consistency combination.  Read-only on the "
           "spacetime.")
      .def("gaussFluxConsistency", &ParticleClusters::gaussFluxConsistency,
           py::arg("fluxes"),
           py::arg("surface_vertex_counts") = std::vector<std::size_t>{},
           py::arg("electric_only") = true,
           "Pure consistency combination over precomputed per-surface "
           "fluxes (the spacetime path delegates here).")
      .def_static("nestedEnclosures", &ParticleClusters::nestedEnclosures,
                  py::arg("st"), py::arg("seed_vertex_ids"),
                  py::arg("shells"),
                  "Nested enclosing vertex sets by breadth-first shell "
                  "growth (returns exactly `shells` sets; sets[0] = the "
                  "seed).")
      .def_static("trackCandidates", &ParticleClusters::trackCandidates,
                  py::arg("from_candidates"), py::arg("to_candidates"),
                  py::arg("overlap_threshold") = 0.5,
                  "Track candidates across scale/time by their color "
                  "bands (#769 matchFibers delegation).")
      // ---- #774 even sectors ------------------------------------------
      .def("octetBilinearRead", &ParticleClusters::octetBilinearRead,
           py::arg("state"), py::arg("color_modes"),
           "The quasi-free traceless-bilinear (octet) read of three "
           "declared color modes of a carried #780 covariance: exact Wick "
           "sums on the covariance layer (no Fock vector); the 1+8 split "
           "is DELEGATED to ColorFiber.  Throws unless exactly three "
           "distinct in-range modes are named.")
      .def("octetBilinearReadCached",
           &ParticleClusters::octetBilinearReadCached,
           py::arg("cache"), py::arg("component_vertex_ids"),
           py::arg("state"), py::arg("color_modes"),
           "octetBilinearRead through the #764 AnalyticCache contract "
           "(key: the caller's component vertex set; parameter: the "
           "covariance hash + declared modes + thresholds).  Cached "
           "equals cold; a Gamma change recomputes.")
      .def("octetFingerprint", &ParticleClusters::octetFingerprint,
           py::arg("state"), py::arg("color_modes"),
           "Content fingerprint of an octet-read request (the cache "
           "parameter).")
      .def("classifyGluon", &ParticleClusters::classifyGluon,
           py::arg("evidence"),
           "Classify one gluon candidate: "
           "certified even parity, a nonzero certified octet excitation "
           "with machine-level octet purity, accepted rank-three "
           "transports, a CERTIFIED zero total determinant winding (zero "
           "baryon flux as evidence), and persistence.  Missing evidence "
           "is a NAMED failed certificate.")
      .def("classifyMeson", &ParticleClusters::classifyMeson,
           py::arg("evidence"),
           "Classify one meson candidate: certified quark + antiquark "
           "(order-insensitive), even composite parity (exact constituent "
           "product), color-singlet pairing, zero total certified "
           "winding/flux.")
      .def("classifyDiquark", &ParticleClusters::classifyDiquark,
           py::arg("evidence"),
           "Classify one diquark candidate: two certified quarks, even "
           "composite parity, a certified anti-triplet wedge occupation, "
           "and the preserved constituent baryon flux B = 2/3 (not an "
           "antiquark).")
      .def("boundSupercomponentSearch",
           &ParticleClusters::boundSupercomponentSearch,
           py::arg("nextLevelComponents"), py::arg("candidates"),
           "The #775 bound-supercomponent search: one "
           "read per next-level component containing at least one "
           "certified quark candidate; found requires a strictly higher "
           "modular level, EXACTLY three contained certified quark "
           "candidates, full support containment, overlapping #765 "
           "lifetimes, and bounded mutual #770 transports.")
      .def_static("scaleProfileSample",
                  &ParticleClusters::scaleProfileSample, py::arg("ctx"),
                  "One refinement sample of the EXISTING #575/#566/#593 "
                  "mass-radius battery, read through the #593 context "
                  "exactly as EmergentRadius/EmergentMass read it "
                  "(RegisterContext.interiorHinges).  Read-only.")
      .def("scaleProfile", &ParticleClusters::scaleProfile,
           py::arg("samples"),
           "The #775 refinement-window certificate: a finite emergent "
           "radius plus the refinement stability of every DIMENSIONLESS "
           "channel.  Nothing here is a form factor and no dimensionful "
           "mass is ever emitted.")
      .def("classifyBaryon", &ParticleClusters::classifyBaryon,
           py::arg("evidence"),
           "Classify one three-cluster candidate and evaluate the "
           "complete proton certificate.  Returns "
           "'no-baryon', 'baryon-candidate', 'certified-proton', or "
           "'quasi-free-sharp-spin-obstruction' with every failed or "
           "unknown certificate NAMED.")
      .def("classifyBoundSupercomponents",
           &ParticleClusters::classifyBoundSupercomponents,
           py::arg("bindings"), py::arg("constituentReads"),
           py::arg("boundLifetimes") = std::vector<double>{},
           "classifyBaryon over the boundSupercomponentSearch result: one "
           "BaryonRead per binding that grouped EXACTLY three certified "
           "constituents, in bindings order.  A binding that grouped a "
           "different number emits NOTHING -- a three-cluster verdict is "
           "never assembled by padding the missing legs.  Only the "
           "binding, the three QuarkReads (quarkIndices indexes "
           "constituentReads) and the bound component's persistence "
           "lifetime travel; the colour columns, the octet flux, the #772 "
           "rotation character, the #780 spin reads, the swept "
           "covariance-only class and the refinement window are left "
           "ABSENT, so each gap is NAMED rather than presumed.");

  // ========================================
  // CrossingReadouts: the whitepaper's world-tube crossing readouts
  // ========================================
  py::class_<CrossingReadoutsConfig>(m, "CrossingReadoutsConfig",
      "Analysis parameters of the world-tube crossing readouts, echoed "
      "verbatim on every read.  kappaMass is the ONE declared mass "
      "calibration; while massCalibrated is False the crossing mass is "
      "reported in UNCALIBRATED units and only ratios are meaningful.")
      .def(py::init<>())
      .def_readwrite("kappaMass", &CrossingReadoutsConfig::kappaMass)
      .def_readwrite("massCalibrated", &CrossingReadoutsConfig::massCalibrated)
      .def_readwrite("signTolerance", &CrossingReadoutsConfig::signTolerance)
      .def_readwrite("degeneracyTolerance",
                     &CrossingReadoutsConfig::degeneracyTolerance)
      .def_readwrite("monopoleTolerance",
                     &CrossingReadoutsConfig::monopoleTolerance)
      .def("toRecord", [](const CrossingReadoutsConfig &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<TemporalFunctionRead>(m, "TemporalFunctionRead",
      "The complex Lorentzian distance tau from the incoming boundary M0, "
      "with its temporal-function certificate.  tau is intrinsic: it reads "
      "the 1-skeleton and the stored complex edge lengths, NEVER a vertex "
      "coordinate.  `certified` is True only when Re tau strictly increases "
      "along every future-directed causal edge; otherwise every failure is "
      "NAMED in failedCertificates.")
      .def(py::init<>())
      .def_readonly("vertices", &TemporalFunctionRead::vertices)
      .def_readonly("tau", &TemporalFunctionRead::tau)
      .def_readonly("layer", &TemporalFunctionRead::layer)
      .def_readonly("certified", &TemporalFunctionRead::certified)
      .def_readonly("failedCertificates",
                    &TemporalFunctionRead::failedCertificates)
      .def_readonly("minCausalIncrement",
                    &TemporalFunctionRead::minCausalIncrement)
      .def_readonly("causalEdgeCount", &TemporalFunctionRead::causalEdgeCount)
      .def_readonly("unreachableCount",
                    &TemporalFunctionRead::unreachableCount)
      .def("at", &TemporalFunctionRead::at, py::arg("vertex"),
           "tau of one vertex, or NaN when unknown.")
      .def("toRecord", [](const TemporalFunctionRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<WorldTubeInput>(m, "WorldTubeInput",
      "One persistent band tracked across cobordism frames, as the crossing "
      "readouts consume it.  `orientation` is the tube's traversal direction "
      "(+1 future-directed, -1 the REVERSED tube): reversing it flips "
      "sgn(pi_perp) and sends B = +1/3 to B = -1/3.  Only certified quark "
      "tubes enter the baryon sum; every admissible crossing enters the "
      "crossing mass.")
      .def(py::init<>())
      .def_readwrite("tubeId", &WorldTubeInput::tubeId)
      .def_readwrite("band", &WorldTubeInput::band)
      .def_readwrite("orientation", &WorldTubeInput::orientation)
      .def_readwrite("determinantWinding", &WorldTubeInput::determinantWinding)
      .def_readwrite("certifiedQuarkTube",
                     &WorldTubeInput::certifiedQuarkTube);

  py::class_<TubeCrossingRead>(m, "TubeCrossingRead",
      "One tube's crossing of one level set.  `perpendicular` is the COMPLEX "
      "pi_perp; `sign` is sgn(Re pi_perp) on an admissible crossing and 0 "
      "when UNKNOWN (an inadmissible crossing has no sign at all, never a "
      "silent zero).")
      .def(py::init<>())
      .def_readonly("tubeId", &TubeCrossingRead::tubeId)
      .def_readonly("level", &TubeCrossingRead::level)
      .def_readonly("crossingEdges", &TubeCrossingRead::crossingEdges)
      .def_readonly("density", &TubeCrossingRead::density)
      .def_readonly("perpendicular", &TubeCrossingRead::perpendicular)
      .def_readonly("sign", &TubeCrossingRead::sign)
      .def_readonly("admissible", &TubeCrossingRead::admissible)
      .def_readonly("failedCertificates",
                    &TubeCrossingRead::failedCertificates)
      .def("toRecord", [](const TubeCrossingRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<CrossingMassRead>(m, "CrossingMassRead",
      "The crossing-mass functional m_x on one level, as the DIFFERENCE "
      "against the same sum at M0.  Never a dimensionful physical mass while "
      "`calibrated` is False.")
      .def(py::init<>())
      .def_readonly("level", &CrossingMassRead::level)
      .def_readonly("crossingMass", &CrossingMassRead::crossingMass)
      .def_readonly("levelSum", &CrossingMassRead::levelSum)
      .def_readonly("referenceSum", &CrossingMassRead::referenceSum)
      .def_readonly("kappaMass", &CrossingMassRead::kappaMass)
      .def_readonly("calibrated", &CrossingMassRead::calibrated)
      .def_readonly("units", &CrossingMassRead::units)
      .def_readonly("admissibleCrossings",
                    &CrossingMassRead::admissibleCrossings)
      .def_readonly("refusedCrossings", &CrossingMassRead::refusedCrossings)
      .def("toRecord", [](const CrossingMassRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<BaryonCrossingRead>(m, "BaryonCrossingRead",
      "The coherent one-third sum over certified quark tubes, with the "
      "determinant-line cross-check.  A tube whose crossing sign DISAGREES "
      "with its certified winding sign is named in signDefects: a defect "
      "signal, reported and never silently resolved.")
      .def(py::init<>())
      .def_readonly("level", &BaryonCrossingRead::level)
      .def_readonly("baryonNumber", &BaryonCrossingRead::baryonNumber)
      .def_readonly("levelSum", &BaryonCrossingRead::levelSum)
      .def_readonly("referenceSum", &BaryonCrossingRead::referenceSum)
      .def_readonly("quarkTubes", &BaryonCrossingRead::quarkTubes)
      .def_readonly("signDefects", &BaryonCrossingRead::signDefects)
      .def_readonly("windingAgreements",
                    &BaryonCrossingRead::windingAgreements)
      .def("toRecord", [](const BaryonCrossingRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<ChargePowerProfileRead>(m, "ChargePowerProfileRead",
      "The spectral charge-power profile S(lambda) built from the EIGENSPACE "
      "PROJECTORS of the slice Laplacian (basis- and phase-invariant, "
      "degeneracies handled).  An INCOHERENT power -- the analogue of a "
      "structure factor -- and NEVER the electromagnetic form factor.  For a "
      "neutral system the monopole vanishes, the normalized profile REFUSES "
      "('neutral-system') and the unnormalized power stays reported.")
      .def(py::init<>())
      .def_readonly("level", &ChargePowerProfileRead::level)
      .def_readonly("eigenvalues", &ChargePowerProfileRead::eigenvalues)
      .def_readonly("power", &ChargePowerProfileRead::power)
      .def_readonly("normalizedPower", &ChargePowerProfileRead::normalizedPower)
      .def_readonly("monopole", &ChargePowerProfileRead::monopole)
      .def_readonly("normalized", &ChargePowerProfileRead::normalized)
      .def_readonly("failedCertificates",
                    &ChargePowerProfileRead::failedCertificates)
      .def_readonly("sliceNodes", &ChargePowerProfileRead::sliceNodes)
      .def("toRecord", [](const ChargePowerProfileRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<ElectromagneticFormFactorRead>(m,
      "ElectromagneticFormFactorRead",
      "The CONDITIONAL electromagnetic form factor G_E and the charge "
      "radius.  This tree certifies neither a conserved U(1) current nor "
      "momentum-transfer states, so this is a refusal scaffold: `available` "
      "is False and the radius is UNAVAILABLE with each missing certificate "
      "NAMED.  The spectral charge-power profile is never substituted.")
      .def(py::init<>())
      .def_readonly("available", &ElectromagneticFormFactorRead::available)
      .def_readonly("chargeRadiusSquared",
                    &ElectromagneticFormFactorRead::chargeRadiusSquared)
      .def_readonly("failedCertificates",
                    &ElectromagneticFormFactorRead::failedCertificates)
      .def_readonly("note", &ElectromagneticFormFactorRead::note)
      .def("toRecord", [](const ElectromagneticFormFactorRead &self) {
        return recordToPython(self.toRecord());
      });

  py::class_<CrossingReadouts>(m, "CrossingReadouts",
      "The whitepaper's world-tube crossing readouts (section \"Mass, "
      "charge, and form factor from world-tube crossings\").  Read-only: no "
      "solver, no facet materialization, no complex rebuild, and nothing "
      "here enters any emergence objective.")
      .def(py::init<>())
      .def_readonly_static("kSchemaVersion", &CrossingReadouts::kSchemaVersion)
      .def_static("temporalFunction", &CrossingReadouts::temporalFunction,
                  py::arg("spacetime"), py::arg("m0Vertices"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "The complex Lorentzian distance tau from M0 with its "
                  "temporal-function certificate.")
      .def_static("bandEdgeDensity",
                  [](const SpectralFiber &band) {
                    py::dict out;
                    for (const auto &entry :
                         CrossingReadouts::bandEdgeDensity(band)) {
                      out[py::make_tuple(entry.first[0], entry.first[1])] =
                          entry.second;
                    }
                    return out;
                  },
                  py::arg("band"),
                  "The band density mu on the 1-skeleton, keyed by the "
                  "endpoint pair in ascending vertex order: the projector "
                  "diagonal of P = Phi Psi^dagger W carried to edges "
                  "(gauge-invariant by left/right cancellation).")
      .def_static("crossing", &CrossingReadouts::crossing, py::arg("tube"),
                  py::arg("temporal"), py::arg("level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "One tube's crossing of the level Re tau = level.")
      .def_static("crossingMass", &CrossingReadouts::crossingMass,
                  py::arg("tubes"), py::arg("temporal"), py::arg("level"),
                  py::arg("m0Level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "m_x on `level` as the difference against `m0Level`.")
      .def_static("baryonNumber", &CrossingReadouts::baryonNumber,
                  py::arg("tubes"), py::arg("temporal"), py::arg("level"),
                  py::arg("m0Level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "B = (1/3) sum sgn(pi_perp) over certified quark tubes, as "
                  "the difference against `m0Level`.")
      .def_static("chargePowerProfile",
                  &CrossingReadouts::chargePowerProfile, py::arg("tubes"),
                  py::arg("temporal"), py::arg("level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "The spectral charge-power profile on `level`.")
      .def_static("formFactor", &CrossingReadouts::formFactor,
                  py::arg("profile"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "The conditional electromagnetic form factor: a refusal "
                  "scaffold naming the certificates this tree lacks.")
      .def_static("overlayRecord",
                  [](const std::vector<WorldTubeInput> &tubes,
                     const TemporalFunctionRead &temporal, double level,
                     double m0Level, const CrossingReadoutsConfig &cfg) {
                    return recordToPython(CrossingReadouts::overlayRecord(
                        tubes, temporal, level, m0Level, cfg));
                  },
                  py::arg("tubes"), py::arg("temporal"), py::arg("level"),
                  py::arg("m0Level"),
                  py::arg("cfg") = CrossingReadoutsConfig{},
                  "Every readout on one level as the versioned overlay "
                  "block.");

  // ── #860: the register carried by a certified cluster ───────────────
  py::class_<RegisterConjunct>(m, "RegisterConjunct",
      "The whitepaper's six fiber-acceptance conjuncts, named.  Reference "
      "these constants rather than retyping the strings: a mis-spelled "
      "literal produces a name no consumer matches.")
      .def_property_readonly_static("CLUSTER_SUPPORT",
          [](py::object) { return RegisterConjunct::kClusterSupport; })
      .def_property_readonly_static("LOCALIZED_PROJECTOR",
          [](py::object) { return RegisterConjunct::kLocalizedProjector; })
      .def_property_readonly_static("BAND_GAP",
          [](py::object) { return RegisterConjunct::kBandGap; })
      .def_property_readonly_static("NEIGHBOUR_OVERLAP",
          [](py::object) { return RegisterConjunct::kNeighbourOverlap; })
      .def_property_readonly_static("FRAME_LIFETIME",
          [](py::object) { return RegisterConjunct::kFrameLifetime; })
      .def_property_readonly_static("TRANSPORT_LEAKAGE",
          [](py::object) { return RegisterConjunct::kTransportLeakage; });

  py::class_<RegisterUnmeasured>(m, "RegisterUnmeasured",
      "Why a conjunct could not be DECIDED, as distinct from being decided "
      "against.  An unmeasured quantity is not a failed one, and neither is "
      "ever encoded as a zero.")
      .def_property_readonly_static("NO_BAND",
          [](py::object) { return RegisterUnmeasured::kNoBand; })
      .def_property_readonly_static("LOCALIZATION_UNMEASURED",
          [](py::object) { return RegisterUnmeasured::kLocalizationUnmeasured; })
      .def_property_readonly_static("BAND_GAP_UNKNOWN",
          [](py::object) { return RegisterUnmeasured::kBandGapUnknown; })
      .def_property_readonly_static("NO_FRAME_TRACK",
          [](py::object) { return RegisterUnmeasured::kNoFrameTrack; })
      .def_property_readonly_static("NO_TRANSPORT",
          [](py::object) { return RegisterUnmeasured::kNoTransport; })
      .def_property_readonly_static("SUPPORT_UNREADABLE",
          [](py::object) { return RegisterUnmeasured::kSupportUnreadable; });

  py::class_<ClusterRegisterConfig>(m, "ClusterRegisterConfig",
      "Thresholds the register is accepted under.  Analysis parameters "
      "only: none of them selects which bands or clusters exist.")
      .def(py::init<>())
      .def_readwrite("minNeighbourOverlap",
                     &ClusterRegisterConfig::minNeighbourOverlap)
      .def_readwrite("minFrameLifetime",
                     &ClusterRegisterConfig::minFrameLifetime)
      .def_readwrite("maxTransportLeakage",
                     &ClusterRegisterConfig::maxTransportLeakage);

  py::class_<RegisterRegimeReport>(m, "RegisterRegimeReport",
      "What the specification requires be reported of the band's metric "
      "regime.  A negative signature is a CERTIFICATE, never an automatic "
      "antiparticle identification.  Unmeasured values are NaN.")
      .def_readonly("regime", &RegisterRegimeReport::regime)
      .def_readonly("gramDefect", &RegisterRegimeReport::gramDefect)
      .def_readonly("positiveSignature",
                    &RegisterRegimeReport::positiveSignature)
      .def_readonly("negativeSignature",
                    &RegisterRegimeReport::negativeSignature)
      .def_readonly("neutralSignature",
                    &RegisterRegimeReport::neutralSignature)
      .def_readonly("signatureNormalizable",
                    &RegisterRegimeReport::signatureNormalizable)
      .def_readonly("eigenResidual", &RegisterRegimeReport::eigenResidual)
      .def_readonly("leftResidual", &RegisterRegimeReport::leftResidual)
      .def_readonly("frameConditionNumber",
                    &RegisterRegimeReport::frameConditionNumber);

  py::class_<ClusterRegisterRead>(m, "ClusterRegisterRead",
      "One register read: the cluster it is carried by, the fiber "
      "E_C = Ran Phi_C it is, the six conjuncts as measured, and the "
      "verdict.  Unmeasured values are NaN and unmeasured conjuncts are "
      "named; nothing is zero-filled.")
      .def_readonly("component", &ClusterRegisterRead::component)
      .def_readonly("support", &ClusterRegisterRead::support)
      .def_readonly("degree", &ClusterRegisterRead::degree)
      .def_readonly("rank", &ClusterRegisterRead::rank)
      .def_readonly("band", &ClusterRegisterRead::band)
      .def_readonly("supportConnected", &ClusterRegisterRead::supportConnected)
      .def_readonly("supportPieces", &ClusterRegisterRead::supportPieces)
      .def_readonly("localizationExcess",
                    &ClusterRegisterRead::localizationExcess)
      .def_readonly("bandGap", &ClusterRegisterRead::bandGap)
      .def_readonly("neighbourOverlap", &ClusterRegisterRead::neighbourOverlap)
      .def_readonly("frameLifetime", &ClusterRegisterRead::frameLifetime)
      .def_readonly("transportLeakage",
                    &ClusterRegisterRead::transportLeakage)
      .def_readonly("regime", &ClusterRegisterRead::regime)
      .def_readonly("failedConjuncts", &ClusterRegisterRead::failedConjuncts)
      .def_readonly("unmeasured", &ClusterRegisterRead::unmeasured)
      .def_readonly("accepted", &ClusterRegisterRead::accepted)
      .def_readonly("certificate", &ClusterRegisterRead::certificate)
      .def_readonly("thresholds", &ClusterRegisterRead::thresholds)
      .def("describe", &ClusterRegisterRead::describe)
      .def("toRecord",
           [](const ClusterRegisterRead &self) {
             return recordToPython(self.toRecord());
           },
           "Checkpoint serialization: the JSON-able record of the read "
           "(schema-versioned; unmeasured channels stay NaN).")
      .def_static("fromRecord",
                  [](const py::handle &record) {
                    return ClusterRegisterRead::fromRecord(
                        pythonToRecord(record));
                  },
                  py::arg("record"),
                  "Rehydrate from toRecord() output; rejects an unknown "
                  "schema_version (ValueError).");

  py::class_<ClusterRegister>(m, "ClusterRegister",
      "Reads the register the whitepaper's 'Recursive spectral fibers' "
      "section defines: the fiber E_C = Ran Phi_C of an isolated localized "
      "band on a persistent cluster, accepted under the six-conjunct list.  "
      "Assembles the existing observables and derives no spectrum, "
      "transport or clustering of its own.  The support's provenance is "
      "never consulted, so no proposer can veto a certified fiber.  "
      "Read-only; nothing here enters any emergence objective and no hole "
      "is required or consulted.")
      .def(py::init<ClusterRegisterConfig>(),
           py::arg("cfg") = ClusterRegisterConfig{})
      .def_property_readonly("config", &ClusterRegister::config)
      .def("read", &ClusterRegister::read, py::arg("st"), py::arg("support"),
           py::arg("band"), py::arg("track"), py::arg("externalTransports"),
           py::arg("component") = ComponentId{},
           "Read the register of one cluster.  An absent track leaves the "
           "lifetime and overlap conjuncts UNMEASURED, never satisfied; an "
           "empty transport list likewise leaves leakage unmeasured rather "
           "than small.")
      .def_static("supportConnectivity", &ClusterRegister::supportConnectivity,
                  py::arg("st"), py::arg("support"),
                  "Whether the induced one-skeleton on the support is "
                  "connected, and in how many pieces.");
}
