// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: sparse graphs and persistent
// modularity. One of the translation units that Bindings.cpp registers in
// order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_graph_modularity(py::module_ &m) {
  // ========================================
  // SparseGraph (for modularity / spectral dimension)
  // ========================================
  py::class_<SparseGraph>(m, "SparseGraph",
      R"doc(Undirected sparse graph in compressed sparse row (CSR) form.

Built from the coordinate-list (COO) output of
``Spacetime.get_dual_adjacency``.  Used by the modularity sweep to compute
spectral dimension on the dual graph.)doc")
      .def_static("from_coo", &SparseGraph::fromCOO,
                  py::arg("rows"), py::arg("cols"), py::arg("n"),
                  "Construct from coordinate-list arrays and a node count.")
      .def("n_nodes", &SparseGraph::nNodes,
           "Number of nodes.")
      .def("n_edges", &SparseGraph::nEdges,
           "Number of undirected edges.")
      .def("degree", &SparseGraph::degree, py::arg("i"),
           "Degree of node ``i`` (number of incident undirected edges).")
      .def("is_bipartite", &SparseGraph::isBipartite,
           "True iff the graph is 2-colorable (no odd cycle).")
      .def("modularity", &SparseGraph::modularity, py::arg("labels"),
           R"doc(Newman-Girvan modularity Q for a node partition.

Q = sum_c [L_c/m - (D_c/2m)^2] over communities c, where L_c is the
intra-community edge count, D_c the summed degree, and m the edge count.
``labels`` has one community id per node (length n_nodes()); distinct
values are distinct communities and need not be dense. Returns 0 for an
empty / edgeless graph; raises ValueError if len(labels) != n_nodes().)doc")
      .def("diagonal_heat_kernel",
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
           py::arg("starts"), py::arg("times"), py::arg("krylov_dim") = 30,
           R"doc(Diagonal of the heat kernel ``e^{-t L_sym}`` for each
(start, t) pair.  Returns a list of lists with shape
(len(starts), len(times)).)doc")
      .def("spectral_dimension",
           [](const SparseGraph &self, int nWalks, double maxSigma,
              std::uint64_t seed, double tailFraction, int nTimes,
              double tMin, int krylovDim) {
             std::mt19937 rng(seed);
             return self.spectralDimension(nWalks, maxSigma, &rng,
                                           tailFraction, nTimes, tMin,
                                           krylovDim);
           },
           py::arg("n_walks"), py::arg("max_sigma"), py::arg("seed") = 0,
           py::arg("tail_fraction") = 0.2, py::arg("n_times") = 40,
           py::arg("t_min") = 0.5, py::arg("krylov_dim") = 30,
           R"doc(Estimate spectral dimension at small / large diffusion times.

Returns ``(D_S_small, D_S_large)``.  Mirrors the Python implementation
in ``examples/modularity.py:Graph.spectral_dimension``.)doc")
      // Per-sigma return probability and spectral-dimension curve,
      // inherited from SpectralGraph; SparseGraph::applyLaplacian installs
      // the symmetric-normalised Laplacian L_sym.
      .def("return_probability",
           &::tessera::graph::SpectralGraph::returnProbability,
           py::arg("sigmas"), py::arg("krylov_dim") = 30,
           py::arg("m") = 0, py::arg("seed") = 0,
           R"doc(P(sigma) = (1/|V|) Tr exp(-sigma L_sym) by Krylov-Lanczos
diagonal estimation, evaluated at each diffusion time in ``sigmas``.

``m`` is the Hutchinson-style subsample of start vertices: 0 (the default)
uses ``min(n_nodes(), 3000)``; pass ``m = n_nodes()`` for the exact trace.
``seed`` controls the subset random number generator.

Feed the result to ``spectral_dimension_curve`` or
``spectral_dimension_smoothed`` to extract D_S(sigma).)doc")
      .def_static("spectral_dimension_curve",
                  &::tessera::graph::SpectralGraph::spectralDimension,
                  py::arg("sigmas"), py::arg("P"),
                  R"doc(D_S(sigma) = -2 d log P / d log sigma by centered
finite differences (one-sided at the endpoints); NaN where P <= 0 or
non-finite.

The full per-sigma curve, aligned with ``sigmas``.  The
``spectral_dimension`` instance method instead random-walk samples and
returns only the (small, large) summary pair.)doc")
      .def_static("spectral_dimension_smoothed",
                  &::tessera::graph::SpectralGraph::spectralDimensionSmoothed,
                  py::arg("sigmas"), py::arg("P"),
                  py::arg("window_size") = 5, py::arg("poly_order") = 2,
                  R"doc(Savitzky-Golay-smoothed D_S(sigma): a local polynomial of
order ``poly_order`` is fit over a centered ``window_size`` window in
(log sigma, log P) and its slope read off at each point.  ``window_size``
must be odd and >= ``polyOrder + 1``.)doc");
  // ========================================
  // PersistentModularity: label-free persistent component discovery
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
      .def("canonical_hash", &ComponentId::canonicalHash,
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
      "Which search proposes the communities. "
      "Both score the same exact Q_gamma closed "
      "form, so their slices are directly comparable; "
      "they differ only in how a partition "
      "is searched for.")
      .value("MultilevelAggregation", DiscoveryStrategy::MultilevelAggregation,
             "Multilevel aggregation from a fixed restart seed sequence, "
             "keeping the best exact score and reporting the restart spread.")
      .value("LeadingEigenvector", DiscoveryStrategy::LeadingEigenvector,
             "Newman's leading-eigenvector bisection of B_gamma = A - gamma "
             "k k^T / 2m, recursed until no group has a positive leading "
             "eigenvalue.  The community count is fixed by the spectrum "
             "rather than by a parameter, and the search carries no seed.");

  py::class_<SplitReason>(m, "SplitReason",
      "The named outcomes of one attempted leading-eigenvector bisection.  "
      "Reference these constants rather than retyping the strings; a "
      "mis-spelled literal names a reason no consumer matches.")
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
      .def_readonly("group_size", &SplitRead::groupSize)
      .def_readonly("leading_eigenvalue", &SplitRead::leadingEigenvalue,
                    "Most positive eigenvalue of B_gamma on the group, over "
                    "the complement of the all-ones vector.  NaN if not "
                    "computed.")
      .def_readonly("second_eigenvalue", &SplitRead::secondEigenvalue,
                    "Second most positive eigenvalue, by deflation.  NaN if "
                    "not computed.")
      .def_readonly("eigenvalue_gap", &SplitRead::eigenvalueGap,
                    "leadingEigenvalue - secondEigenvalue: how well "
                    "determined the bisection is.  NaN if either is "
                    "unmeasured.")
      .def_readonly("delta_q", &SplitRead::deltaQ,
                    "Exact change in total Q_gamma this split would produce, "
                    "from the class's own closed form.  NaN if no split was "
                    "evaluated.")
      .def_readonly("accepted", &SplitRead::accepted,
                    "Whether the group was actually bisected.")
      .def_readonly("resolved", &SplitRead::resolved,
                    "Whether the spectrum determined the outcome.  False "
                    "means the split was refused as not well determined, "
                    "which is distinct from a determined 'do not split'.")
      .def_readonly("reason", &SplitRead::reason,
                    "One of the SplitReason constants.")
      .def_readonly("size_a", &SplitRead::sizeA)
      .def_readonly("size_b", &SplitRead::sizeB);

  py::class_<PersistentModularityConfig>(m, "PersistentModularityConfig",
      "Configuration for the label-free multiscale component discovery.")
      .def(py::init<>())
      .def_readwrite("strategy", &PersistentModularityConfig::strategy,
                     "Which search proposes the communities. "
                     "The default is multilevel aggregation.")
      .def_readwrite("objective", &PersistentModularityConfig::objective,
                     R"doc(Which real functional of Q the search maximizes.
The default is Score on a real graph; a complex graph selects Magnitude
regardless, since Score is not an ordering there.

Setting Magnitude on a real graph pursues anti-community structure: it has
Q < 0, so maximizing Q passes it over in favour of the one-community
partition, while maximizing |Q| finds it.)doc")
      .def_readwrite("leading_eigenvalue_tolerance",
                     &PersistentModularityConfig::leadingEigenvalueTolerance,
                     "LeadingEigenvector: a group is indivisible when its "
                     "leading eigenvalue does not exceed this.")
      .def_readwrite("min_eigenvalue_gap",
                     &PersistentModularityConfig::minEigenvalueGap,
                     "LeadingEigenvector: minimum leading-to-second gap for "
                     "a bisection to count as well determined.  Below it the "
                     "split is refused with a named reason.")
      .def_readwrite("dense_eigen_solve_max_group",
                     &PersistentModularityConfig::denseEigenSolveMaxGroup,
                     "LeadingEigenvector: groups of at most this many cells "
                     "get an exact dense symmetric eigendecomposition; larger "
                     "groups fall back to shifted power iteration.  The dense "
                     "path exists because iteration is slowest exactly where "
                     "the pair is near-degenerate, which is the case the gap "
                     "certificate has to adjudicate.")
      .def_readwrite("max_power_iterations",
                     &PersistentModularityConfig::maxPowerIterations,
                     "LeadingEigenvector: hard cap on power-iteration steps "
                     "per eigenpair above denseEigenSolveMaxGroup.  "
                     "Non-convergence is reported.")
      .def_readwrite("power_iteration_tolerance",
                     &PersistentModularityConfig::powerIterationTolerance,
                     "LeadingEigenvector: relative convergence tolerance of "
                     "the Rayleigh quotient.")
      .def_readwrite("kernighan_lin_refinement",
                     &PersistentModularityConfig::kernighanLinRefinement,
                     "LeadingEigenvector: run a Kernighan-Lin local "
                     "refinement after each sign bisection.")
      .def_readwrite("resolutions", &PersistentModularityConfig::resolutions,
                     "Resolution parameters gamma, in scan order.")
      .def_readwrite("base_seed", &PersistentModularityConfig::baseSeed,
                     "Base of the fixed restart seed sequence "
                     "(restart t uses splitmix64(baseSeed + t)).")
      .def_readwrite("restarts", &PersistentModularityConfig::restarts,
                     "Deterministic restarts per resolution; best exact "
                     "score kept, spread reported.")
      .def_readwrite("overlap_threshold",
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
      .def_readonly("internal_weight", &ComponentRead::internalWeight,
                    "Sigma_in: internal weight counting both directions.")
      .def_readonly("strength", &ComponentRead::strength,
                    "S_C: summed member strength.")
      .def_readonly("conductance", &ComponentRead::conductance,
                    R"doc(cut(C)/min(vol C, vol V\C); 0 when the denominator
vanishes.  NaN on a signed graph, where a community's strength is a
difference and there is no volume for the cut to be a fraction of; left
unmeasured rather than computed by a formula that does not apply.)doc")
      .def_readonly("modularity_contribution",
                    &ComponentRead::modularityContribution,
                    R"doc(This community's exact additive Q_gamma term, under
whichever null model scores the graph.  These sum over a level to that
level's exact Q_gamma either way.)doc");

  py::class_<RestartRead>(m, "RestartRead",
      "One deterministic restart: seed and exact best score.")
      .def_readonly("seed", &RestartRead::seed)
      .def_readonly("q", &RestartRead::q,
                    "Exact Q_gamma of this restart, complex and unreduced.")
      .def_readonly("objective_value", &RestartRead::objectiveValue,
                    "The real scalar this restart was ranked on.")
      .def_readonly("communities", &RestartRead::communities);

  py::class_<ResolutionSlice>(m, "ResolutionSlice",
      R"doc(Discovery result at one resolution gamma.  ``q`` is the exact
Q_gamma of the winning partition (cold recompute): the best score across
deterministic restarts, a heuristic proposal, never the global optimum
(modularity maximization is NP-hard).  ``q_incremental`` is the
accepted-delta-Q ledger and must agree with ``q`` to double round-off.

``q`` is complex and unreduced: ``abs(q)`` is how much structure the
partition has, ``cmath.phase(q)`` which kind — 0 a community, pi an
anti-community, +-pi/2 lightlike cohesion, anything else mixed.  It is
exactly real on a real graph.  ``objective_value`` is the real scalar the
search maximized; ``objective`` says which functional that was.)doc")
      .def_readonly("gamma", &ResolutionSlice::gamma)
      .def_readonly("q", &ResolutionSlice::q)
      .def_readonly("q_incremental", &ResolutionSlice::qIncremental)
      .def_readonly("objective_value", &ResolutionSlice::objectiveValue,
                    "The real scalar the search maximized: q.real under "
                    "Score, abs(q) under Magnitude.")
      .def_readonly("objective", &ResolutionSlice::objective,
                    "Which functional that was.  Reported rather than "
                    "assumed: a complex graph selects Magnitude whatever "
                    "the config asked for, since Score is not an ordering "
                    "there.")
      .def_readonly("levels", &ResolutionSlice::levels)
      .def_readonly("sweep_recurrences", &ResolutionSlice::sweepRecurrences,
                    "Aggregation levels of the winning run whose sweeps "
                    "ended at a partition an earlier pass had ended in (the "
                    "gains around the cycle were rounding), and not at a "
                    "pass that moved no node.")
      .def_readonly("components", &ResolutionSlice::components,
                    "Final-level components, ordered by canonical hash.")
      .def_readonly("hierarchy", &ResolutionSlice::hierarchy,
                    "hierarchy[k] = communities at aggregation level k+1.")
      .def_readonly("restarts", &ResolutionSlice::restarts)
      .def_readonly("restart_spread", &ResolutionSlice::restartSpread,
                    "max - min of the restart scores, the heuristic spread. "
                    "NaN under LeadingEigenvector, which has no restarts; "
                    "unmeasured is never encoded as zero.")
      .def_readonly("strategy", &ResolutionSlice::strategy,
                    "Which search produced this slice.")
      .def_readonly("splits", &ResolutionSlice::splits,
                    "LeadingEigenvector: one SplitRead per attempted "
                    "bisection, in the order attempted — the strategy's "
                    "spectral certificate.  Empty for MultilevelAggregation.");

  py::class_<ComponentMatch>(m, "ComponentMatch",
      R"doc(Matched component pair across adjacent resolutions or cobordism
time.  ``projector_overlap`` is the value of the spectral-projector hook, or
None when no hook is installed; unknown is never encoded as zero.)doc")
      .def_readonly("from_id", &ComponentMatch::from)
      .def_readonly("to_id", &ComponentMatch::to)
      .def_readonly("from_index", &ComponentMatch::fromIndex)
      .def_readonly("to_index", &ComponentMatch::toIndex)
      .def_readonly("support_overlap", &ComponentMatch::supportOverlap,
                    "Jaccard overlap of level-0 cell supports.")
      .def_readonly("projector_overlap", &ComponentMatch::projectorOverlap);

  py::class_<PersistenceTrack>(m, "PersistenceTrack",
      R"doc(A component followed across the resolution scan by maximum
support overlap.  Lifetime, overlap and conductance are proposal
diagnostics only: they neither accept nor veto a fiber.
``weight_aware_status`` is the downstream weight-aware
gap/localization/persistence status, or None when unpopulated (unknown is
never encoded as zero).)doc")
      .def_readonly("members", &PersistenceTrack::members)
      .def_readonly("member_indices", &PersistenceTrack::memberIndices)
      .def_readonly("first_slice", &PersistenceTrack::firstSlice)
      .def_readonly("last_slice", &PersistenceTrack::lastSlice)
      .def_readonly("gamma_first", &PersistenceTrack::gammaFirst)
      .def_readonly("gamma_last", &PersistenceTrack::gammaLast)
      .def_readonly("min_adjacent_overlap",
                    &PersistenceTrack::minAdjacentOverlap)
      .def_readonly("mean_conductance", &PersistenceTrack::meanConductance)
      .def_property_readonly("weight_aware_status",
           [](const PersistenceTrack &t) {
             return recordToPython(t.weightAwareStatus);
           });

  py::class_<FrameTrack>(m, "FrameTrack",
      R"doc(A component followed across cobordism frames by maximum support
overlap.  ``frames`` is the lifetime measured in cobordism frames, a
different quantity from :class:`PersistenceTrack`, which counts modularity
resolution slices of a single frame.)doc")
      .def_readonly("members", &FrameTrack::members)
      .def_readonly("member_indices", &FrameTrack::memberIndices)
      .def_readonly("first_frame", &FrameTrack::firstFrame)
      .def_readonly("last_frame", &FrameTrack::lastFrame)
      .def_readonly("min_adjacent_overlap", &FrameTrack::minAdjacentOverlap)
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
resolution and cobordism time.

Exact identities on the nonnegative weighted undirected similarity graph:
generalized modularity Q_gamma(P) = (1/2m) sum_ij (A_ij - gamma k_i k_j/2m)
[c_i = c_j], evaluated from per-community sufficient statistics, and the
exact O(deg v) cached local move gain
dQ(v: a->b) = (w_vb - w_va)/m - gamma k_v (k_v + S_b - S_a)/(2 m^2), so one
sparse sweep is near O(|E|).  Incremental accumulations are tested against
cold recomputation at double round-off.

Global modularity maximization is NP-hard; discovery is a deterministic
multilevel aggregation from a fixed seed sequence, with the restart spread
reported.  The score is blind to signed and complex Hodge weights.
Modularity is a proposal generator only: it never enters the emergence
objective and may not veto an otherwise certified fiber; acceptance belongs
to the independent weight-aware certificates, and unknown is reported as
None, never zero.

Read-only: never calls a solver, never mutates the spacetime it reads.)doc");

  py::enum_<PersistentModularity::WeightMap>(pm, "WeightMap",
      "Documented monotone map from complex edge magnitude to similarity.")
      .value("Unit", PersistentModularity::WeightMap::Unit,
             "w = 1: the combinatorial one-skeleton, exactly the standard "
             "Newman-Girvan graph.")
      .value("ExpNegAbsLength",
             PersistentModularity::WeightMap::ExpNegAbsLength,
             "w = exp(-|l|): monotone decreasing in the complex edge "
             "magnitude (the mutual-information convention l = -log I).  "
             "Causally blind: a timelike and a spacelike edge of equal "
             "magnitude give the identical weight.")
      .value("CausalPhaseExpNegAbsLength",
             PersistentModularity::WeightMap::CausalPhaseExpNegAbsLength,
             "w = exp(-|l|) exp(i arg(l^2)): the same similarity magnitude "
             "as ExpNegAbsLength, carrying the edge's causal character as "
             "its argument.  Spacelike lands on the positive real axis, "
             "timelike on the negative, lightlike on +-i, and a generic "
             "argument stays where it is -- nothing is bucketed, so there "
             "is no indefinite case to refuse.");

  py::enum_<ModularityObjective>(m, "ModularityObjective",
      R"doc(Which real functional of the complex Q the search maximizes.
Both readings are always reported; this chooses only what is pursued.)doc")
      .value("Score", ModularityObjective::Score,
             "Maximize Q itself. Available only where Q is "
             "real, and the default. Finds community structure "
             "and passes over anti-communities, which score "
             "below the one-community partition.")
      .value("Magnitude", ModularityObjective::Magnitude,
             "Maximize |Q|.  Always available, and the only ordered choice "
             "once A is genuinely complex.  Finds community and "
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
                    "Edges with a generic arg(l^2).  Ordinary edges for the "
                    "complex weight map, which carries their argument as it "
                    "stands; a diagnostic, not a gate.")
      .def_readonly("degenerate",
                    &PersistentModularity::CausalWeightRead::degenerate);

  pm.def_static("causal_weight_availability",
                [](const std::shared_ptr<Spacetime> &st) {
                  return PersistentModularity::causalWeightAvailability(*st);
                },
                py::arg("spacetime"),
                R"doc(Census of the one-skeleton's causal characters and
whether CausalPhaseExpNegAbsLength can be read from it.  Read-only; uses
the same Edge.disposition() classifier, so a True here is exactly the
condition under which fromSpacetime will not raise.

A mixed count does not make the map unavailable: the complex weight carries
a generic argument as readily as a definite one.  Only a genuine absence
does.  The mixed fraction is a diagnostic, and falls if relaxation is
imposing causal character.)doc")
      .def_static("from_weighted_edges",
                &PersistentModularity::fromWeightedEdges,
                py::arg("src"), py::arg("tgt"), py::arg("weight"),
                py::arg("isolated_cells") = std::vector<std::uint64_t>{},
                R"doc(Build from an explicit real weighted edge list, signed
or not.  Cells are arbitrary 64-bit ids; parallel edges consolidate by
weight summation; self-loops are ignored, as are edges whose consolidated
weight is zero (a measured absence of net similarity).  Raises ValueError
on non-finite weights or mismatched lengths.

See fromComplexWeightedEdges for the general domain.)doc")
      .def_static("from_complex_weighted_edges",
                &PersistentModularity::fromComplexWeightedEdges,
                py::arg("src"), py::arg("tgt"), py::arg("weight"),
                py::arg("isolated_cells") = std::vector<std::uint64_t>{},
                R"doc(Build from an explicit complex weighted edge list.
Consolidation, self-loops and the cancel-to-zero convention are as for
fromWeightedEdges; both components must be finite.  A list that happens to
be real takes the real path and scores exactly as fromWeightedEdges would.

The adjacency is complex symmetric (A_ij = A_ji, no conjugation): a weight
is a property of the edge, and its magnitude and argument do not depend on
which end it is read from.)doc")
      .def_static("from_spacetime",
                  [](const std::shared_ptr<Spacetime> &st,
                     PersistentModularity::WeightMap map) {
                    return PersistentModularity::fromSpacetime(*st, map);
                  },
                  py::arg("spacetime"),
                  py::arg("map") =
                      PersistentModularity::WeightMap::ExpNegAbsLength,
                  R"doc(Build the similarity graph from the spacetime
one-skeleton (read-only).  With CausalPhaseExpNegAbsLength this raises
ValueError, naming the reason, when causal_weight_availability() reports the
map unreadable.  That happens only for a genuine absence: a degenerate
edge has no argument to carry, and arg(0) is not a reading of anything.  An
indefinite argument is not an absence and is carried as it stands.)doc")
      .def("n_cells", &PersistentModularity::nCells)
      .def("n_edges", &PersistentModularity::nEdges)
      .def("is_complex", &PersistentModularity::isComplex,
           R"doc(True when some edge weight has a nonzero imaginary part, so
Q is genuinely complex and Score is not an ordering.  A property of the
graph, not a setting.)doc")
      .def("is_signed", &PersistentModularity::isSigned,
           R"doc(True when some edge weight is negative or non-real, i.e.
when the graph leaves the nonnegative regime the standard modularity
formula assumes.  A property of the graph, not a setting.)doc")
      .def("total_weight2", &PersistentModularity::totalWeight2,
           R"doc(T = sum_ij |A_ij|, the real positive scale the score divides
by.  It cannot vanish while any edge exists, unlike the signed total.
Equal to 2m = sum_ij A_ij on a nonnegative graph.)doc")
      .def("total_weight_sum", &PersistentModularity::totalWeightSum,
           R"doc(SA = sum_ij A_ij, the complex total the configuration null
model redistributes.  Equal to total_weight2() on a nonnegative graph.  A
vanishing SA leaves the null model undefined and is refused by name.)doc")
      .def("cell_ids", &PersistentModularity::cellIds,
           "Cell ids in internal storage order (no convention).")
      .def("modularity_gamma", &PersistentModularity::modularityGamma,
           py::arg("labels"), py::arg("gamma"),
           R"doc(Exact generalized modularity Q_gamma of a fixed partition
(labels[i] labels cell_ids()[i]).  The fixed-partition entry point: at
gamma = 1 on a Unit-weight graph this is exactly the Newman-Girvan score.)doc")
      .def("discover",
           [](const PersistentModularity &self, double gamma,
              const PersistentModularityConfig &cfg) {
             py::gil_scoped_release release;
             return self.discover(gamma, cfg);
           },
           py::arg("gamma"), py::arg("config"),
           "Deterministic label-free discovery at one resolution.")
      .def("scan_resolutions",
           [](const PersistentModularity &self,
              const PersistentModularityConfig &cfg) {
             py::gil_scoped_release release;
             return self.scanResolutions(cfg);
           },
           py::arg("config"),
           "The configurable resolution-sequence scan with persistence "
           "tracks.")
      .def("match_components", &PersistentModularity::matchComponents,
           py::arg("a"), py::arg("b"),
           R"doc(Match components across resolution or cobordism time by
simplex-support overlap (Jaccard on level-0 cell ids over a common cell-id
universe).  When a projector-overlap hook is installed its value is
reported per match; matching decisions are support-based.)doc")
      .def("track_across_frames",
           [](const PersistentModularity &self,
              const std::vector<std::vector<ComponentRead>> &frames,
              double overlapThreshold) {
             py::gil_scoped_release release;
             return self.trackAcrossFrames(frames, overlapThreshold);
           },
           py::arg("frames"), py::arg("overlap_threshold") = 0.5,
           R"doc(Follow components across cobordism frames: frames[t] is the
component list read from frame t over a common cell-id universe.  Chains
consecutive frames with matchComponents by best support overlap, the same
rule scanResolutions applies to resolution slices.  Supplies the lifetime
measured in cobordism frames; a component seen in one frame gets a
one-frame track, which is a measured fact.)doc")
      .def("set_projector_overlap_hook",
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
           "Install (or clear with None) the spectral-projector "
           "overlap hook: hook(fromId, toId) -> float in [0, 1].  "
           "The caller supplies the "
           "projectors.")
      .def_static("invalidated_ancestry",
                  &PersistentModularity::invalidatedAncestry,
                  py::arg("report"), py::arg("touched_cells"),
                  R"doc(Components (at every hierarchy level of every slice)
whose support intersects the touched level-0 cells, plus the affected
tracks.  Siblings with disjoint support remain valid.  Pure bookkeeping —
no recomputation.)doc");
}
