// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: CobordismDAG, proton
// ingredients, surgical cones and certificates. One of the translation units
// that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_cobordism_dag_proton_certificate(py::module_ &m) {
  // === CobordismDAG: chain emergent merges, output -> input ===
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
           "Pipe each node's output fibers into the downstream input blocks the edges name.")
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

  // === ProtonSynthesis: the labelled controlled synthesis of a proton ===
  auto protonClass = py::class_<ProtonSynthesis>(m, "ProtonSynthesis",
      R"doc(Controlled synthesis of a proton, composing MultiCobordism.

This is a labelled controlled-synthesis experiment, not emergence: it pins the
colour singlet {1, w, w*w} (w = (-1 + i*sqrt(3))/2, the primitive cube root of
unity) as an output target and accepts an attempt only if the whole cobordism
carries that singlet on at least min_emergent_holes holes. Targets are
permitted only in explicitly labelled controlled synthesis, so every node this
class builds is in MultiCobordism.SimulationMode.SYNTHESIS (recorded on a
checkpoint as "synthesis"), and drive_node, build() and build_direct() refuse,
with ValueError, a node in any other mode, including either emergence
sub-mode. The emergence protocol pins no target.

A proton is THREE quarks in a colourless bound state, so it is synthesized in
TWO steps (a single merge would be physically invalid).
  * Step A (recombination, one 2->2 node): two neutral q-qbar pairs {1,-1,0},
    {1,0,-1} -> a coloured diquark {1,w} + antidiquark {1,w*w} (2-vectors).
  * Step B (formation, a separate 2->1 node): the diquark {1,w} + the third
    quark {w*w} -> the proton {1,w,w*w} (the 3-vector colour singlet).
build() grows each step from a single Delta^4 simplex seed and restarts across
distinct seeds until step B's whole cobordism carries the singlet on >= 3
holes. The accessors lazily trigger build() on first use, so
`ProtonSynthesis().block()` just works. Observable readers
(charge/mass/radius/spin) read OFF block().)doc");
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
      .def_static("omega", &ProtonSynthesis::omega,
                  "w = (-1 + i*sqrt(3))/2, the primitive cube root of unity.")
      .def_static("singlet", &ProtonSynthesis::singlet,
                  "The proton colour singlet {1, w, w*w}.")
      .def("build", &ProtonSynthesis::build, py::arg("max_restarts") = 16,
           py::arg("init_steps") = 180,
           py::arg("evolve_steps") = 60, py::arg("stage1_candidate_moves") = 8,
           py::arg("stage2_beta") = 1.0,
           py::arg("stage2_max_iters") = 10, py::arg("color_tolerance") = 0.5,
           py::arg("min_emergent_holes") = 3,
           "Restart across seeds until the whole step-B cobordism carries the singlet "
           "on >= min_emergent_holes holes. Each step runs an init pass (grow the "
           "boundary until it carries) then an evolution pass (boundary frozen), in "
           "SimulationMode.SYNTHESIS.")
      .def("recombination_node", &ProtonSynthesis::recombinationNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) Step A node in SimulationMode.SYNTHESIS: two "
           "neutral q-qbar pairs -> a diquark {1,w} + antidiquark {1,w*w}, on a single "
           "Delta^4 seed -- the exact node build() uses for recombination.")
      .def("formation_node", &ProtonSynthesis::formationNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) Step B node in SimulationMode.SYNTHESIS: the "
           "diquark {1,w} + the third quark {w*w} -> the proton singlet, on a single "
           "Delta^4 seed (output read off the whole).")
      .def("direct_node", &ProtonSynthesis::directNode, py::arg("seed"),
           "A fresh, seeded (not-yet-run) ONE-STEP node (6->1) in "
           "SimulationMode.SYNTHESIS: the three bare quarks {1}, {w}, {w*w} and their "
           "three anti-quarks (the elementwise conjugates -- three q-qbar pairs) as "
           "inputs, and the proton singlet as the single output, read off the WHOLE "
           "cobordism (the anti-baryon partner is not pinned), on a single Delta^4 "
           "seed -- the experimental single-merge alternative to the two-step "
           "synthesis. Drive it with run().")
      .def_static("drive_node",
           [](MultiCobordism &node, int initSteps, int evolveSteps,
              int stage1CandidateMoves, double stage2Beta, int stage2MaxIters,
              bool directedSurgery) {
             ProtonSynthesis::driveNode(
                 node, ProtonSynthesis::NodeDrive{initSteps, evolveSteps,
                                                  stage1CandidateMoves,
                                                  stage2Beta, stage2MaxIters,
                                                  directedSurgery});
           },
           py::arg("node"), py::arg("init_steps") = 180,
           py::arg("evolve_steps") = 60, py::arg("stage1_candidate_moves") = 8,
           py::arg("stage2_beta") = 1.0, py::arg("stage2_max_iters") = 10,
           py::arg("directed_surgery") = false,
           "Drive one synthesis node through build()'s per-node schedule: an init "
           "pass (grow the boundary), an optional directed cone-out, an evolution "
           "pass (boundary frozen), an optional directed cone-in, then run_stage2. "
           "Raises ValueError, before anything runs, when the node is not in "
           "SimulationMode.SYNTHESIS.")
      .def_static("require_synthesis_mode", &ProtonSynthesis::requireSynthesisMode,
           py::arg("node"),
           "Raise ValueError naming the node's mode unless it is "
           "SimulationMode.SYNTHESIS: the synthesis pins targets, which are "
           "permitted only in the labelled controlled-synthesis mode.")
      .def("build_direct", &ProtonSynthesis::buildDirect, py::arg("max_restarts") = 16,
           py::arg("init_steps") = 180, py::arg("evolve_steps") = 60,
           py::arg("stage1_candidate_moves") = 8, py::arg("stage2_beta") = 1.0,
           py::arg("color_tolerance") = 0.5, py::arg("min_emergent_holes") = 3,
           py::call_guard<py::gil_scoped_release>(),
           "EXPERIMENTAL one-step synthesis: drive direct_node (three q-qbar pairs in, "
           "the singlet out) with the combined run() drive -- stage-1 surgery and "
           "stage-2 relaxation interleaved in one loop -- as an init pass then an "
           "evolution pass, restarting across seeds, in SimulationMode.SYNTHESIS. "
           "Populates the same accessors as build() (diquark_residual stays 0 -- no "
           "step A). Shares build()'s once-only latch: call it BEFORE any accessor "
           "triggers the lazy two-step build().")
      .def("converged", &ProtonSynthesis::converged,
           "True iff step B's whole cobordism carries the singlet on enough holes.")
      .def("seed", &ProtonSynthesis::seed,
           "Base seed of the converged (or best) attempt.")
      .def("spacetime", &ProtonSynthesis::spacetime,
           "Step B's full relaxed complex, grown from the single Delta^4 seed.")
      .def("block", &ProtonSynthesis::block,
           "The synthesized proton: the relaxed step-B cobordism as a whole.")
      .def("emergent_holes", &ProtonSynthesis::emergentHoles,
           "The holes (MultiCobordism.emergent_holes) of the synthesized proton over "
           "which the singlet periods are read (>=3 when converged). A topological "
           "observable, not a quark count.")
      .def("color_residual", &ProtonSynthesis::colorResidual,
           "Step B's proton singlet r_state (~0 => carried).")
      .def("diquark_residual", &ProtonSynthesis::diquarkResidual,
           "Step A's r_U (small => the diquark recombination converged).");

  // === ProtonIngredients: the ingredients arm, no output pinned ===
  py::class_<ProtonIngredients>(m, "ProtonIngredients",
      R"doc(The ingredients arm of the proton experiment, whose final state is not
pinned. ProtonSynthesis is composed here unchanged; ProtonIngredients prepares the same
ingredients through the same two-step drive EXCEPT that the final state is never
pinned: step B's output-target list is EMPTY, so the objective is
F = ||grad S||^2 + gamma * sum_i r_U(input_i) and whatever the whole cobordism
comes to carry is READ afterwards, never driven. Exactly one variable differs
from ProtonSynthesis.build() (the singlet output target), so the two classes form a
clean A/B experiment. Step A is ProtonSynthesis.recombination_node, which pins the
diquark and antidiquark outputs and so runs in SimulationMode.SYNTHESIS; step B
pins no output and runs in the node's default mode, EMERGENCE (STRICT). The
seed stays uniform and all-spacelike by design: at initialization no time has
passed — causal structure marks sequences of events and may only emerge. Convergence carries no answer-shaped gate: an attempt
converges iff it is STATIONARY (stage 2 stopped on its stationarity test) and
PERSISTENT (a continued evolve+relax pass leaves holes, b_k, and F stable).
Everything physical is a post-hoc observable, including the singlet residual —
a diagnostic for comparing against the synthesis's carried level.)doc")
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
           "attempt. Same drive per node as ProtonSynthesis.build().")
      .def("recombination_node", &ProtonIngredients::recombinationNode,
           py::arg("seed"),
           "Step A verbatim: the composed ProtonSynthesis's recombination_node, in "
           "SimulationMode.SYNTHESIS.")
      .def("formation_node", &ProtonIngredients::formationNode, py::arg("seed"),
           "Step B with nothing pinned: the same ideal diquark {1,w} + third quark "
           "{w*w} inputs on the same single Delta^4 seed as ProtonSynthesis.formation_node, "
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
           "ProtonSynthesis.block).")
      .def("emergent_holes", &ProtonIngredients::emergentHoles,
           "The emergent holes on the whole — an observable, not a gate; "
           "may be any count, including zero.")
      .def("singlet_residual", &ProtonIngredients::singletResidual,
           "DIAGNOSTIC only: the singlet r_state of ProtonSynthesis.singlet() against the "
           "whole, read after the fact for comparison with the canonical build. It "
           "never steers or gates this build.")
      .def("input_residual", &ProtonIngredients::inputResidual,
           "Step B's inputs-only r_U — the whole matter term of the emergent arm.")
      .def("final_objective", &ProtonIngredients::finalObjective,
           "The kept attempt's final objective F.")
      .def("diquark_residual", &ProtonIngredients::diquarkResidual,
           "Step A's r_U — reported exactly as ProtonSynthesis reports it.");

  // ----- Gated surgical cone-out/cone-in (topology change) -----
  py::class_<SurgicalCone>(m, "SurgicalCone",
      R"doc(Gated surgical cone-out/cone-in: the topology-CHANGING move.

The genuine b_k-hole creator. Pachner
moves and the orientation-safe stellar refinement cone (T1/T2) are topology-
PRESERVING; this is not. coneOut removes one top cell (its orphaned edges, then
any isolated vertex) -- on a closed manifold this opens a manifold-with-boundary
and, for a cell disjoint from an existing hole, raises b_{d-1} by 1 (on S^3, the
color register's b_2). coneIn adds one top cell on a fresh vertex joined to d
existing vertices, lowering b_{d-1} by 1 when it caps a hole. EVERY move is gated
on ChainComplex.dualComplexIsValid (a valid manifold-with-boundary; the
n>=4 recursive check) -- surgery is allowed BECAUSE it is gated. Rejected
moves roll back bit-identically.
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
      .def("bridge", &SurgicalCone::bridge, py::arg("cell_vertices"),
           "(ok, reason): gated surgical bridge (qubit cobordism spec D1) -- "
           "create the top cell on the d+1 EXISTING vertices `cell_vertices` "
           "(no fresh apex), auto-wiring the edges it lacks with the engine's "
           "auto-wired length. Accepts only a valid manifold-with-boundary; "
           "otherwise removes the cell and every edge it alone introduced, "
           "bit-exactly. Whether the vertices split across two boundary blocks "
           "without a chord is the caller's draw, not this gate.")
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

  // ----- Analytic-first kernel and cache contract -----

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
      R"doc(Certification record attached to every analytic-first kernel result.

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
}
