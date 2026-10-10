// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: the modularity optimizer,
// volume profiles and the simplicial qubit. One of the translation units
// that Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "BindingsCommon.h"

void register_observables_modularity_volume_simplicial_qubit(py::module_ &m) {
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
      R"doc(Modularity sweep on a causal dynamical triangulation (CDT)
spacetime, driven by transactional Pachner moves with Q-direction
acceptance.

Each iteration:
  1. Picks a random move type from {add, remove, flip, iflip, shift}.
  2. Calls cdt.proposeXxx() — if no eligible target, tries another type.
  3. Snapshots Q on the spacetime 1-skeleton.
  4. Calls move.apply() to commit the move.
  5. Computes new Q.  If direction matches, keeps the move; else
     calls move.rollback().
  6. If Q crossed the next target_dq threshold, builds the dual graph
     and measures D_S.)doc")
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
current spacetime one-skeleton.  Read-only: never mutates the spacetime and
never proposes moves.  Builds the nonnegative similarity graph under
``map`` and runs PersistentModularity.scanResolutions(config).  A proposal
generator: blind to signed and complex Hodge weights, never part of the
emergence objective, and never a veto over a certified fiber.)doc");
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
  // Simplicial qubit: complex geometry and pure-gauge link phases.
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
        "A single qubit state encoded as the holomorphic line in the harmonic space of the metric "
        "Hodge Laplacian on a triangulated torus. Input: vertices 0..nV-1, edges (i, j) with i "
        "< j, consistently oriented faces (i, j, k), edge lengths (real positive, or complex), "
        "and marked cycles A, B as (edge_index, sign) lists with A.B = +1. Computed on load: the "
        "input validations; d0, d1; per-face angles, Heron areas and local layouts; cotangent "
        "weights M1 = diag(w_e) (negative weights and Delaunay violations flagged on the real "
        "locus); the harmonic space H = null_space([d1; d0.T M1]) of dimension 2; the Whitney-form "
        "L2 inner product at barycenters; the complex structure J = G^{-1} R.T by rotate-then-project "
        "with its residual ||J J + I||_F; the holomorphic line, periods, tau = P_B / P_A (in the "
        "upper half plane on the real locus); the state, Bloch vector and density matrix; and "
        "the degeneration warnings. The optional intrinsic Delaunay edge-flip pass is intrinsic_delaunay(). "
        "Alternatively read a Spacetime of dimension 2 directly: its faces are oriented by the "
        "fundamental class, reversed=True selects the other hemisphere, and its edge phases are "
        "the pure-gauge link connection.\n\n"
        "Complex geometry: off the real locus (real lengths, no phases) every formula runs over "
        "C. The real reference is the same complex with every squared length 1 (the unit equilateral "
        "reference simplex of chainhodge.WhitneyMass); angles are the principal acos, Heron areas "
        "the continuation branch of sqrt(det G)/2 (WhitneyMass.volumeOnBranch) along the straight "
        "segment in the squared lengths from the reference, pairings are the transpose (bilinear) "
        "pairing, the harmonic space a complex null space, and the eigenline is chosen by continuity "
        "from the reference (the Im tau > 0 rule holds on the real locus only). Link phases must "
        "be a pure gauge (flux or holonomy is refused by name): they twist the incidences and the "
        "Whitney pairing, taken against the kernel of the inverse links (dual_harmonic_basis()), "
        "and the periods are taken with parallel transport along the marked cycles from one base "
        "point (base_vertex()), which leaves tau, the state and the coefficient pairs in the period "
        "frame invariant. On the real locus the matrix-valued reads return real arrays, bit-identical "
        "to the real-length construction; off it complex arrays.")
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
             "The explicit constructor; lengths real positive or complex, the trivial "
             "connection. Raises ValueError when an input validation or a complex "
             "continuation fails, and RuntimeError when dim H != 2.")
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
        // ---- derived reads
        .def("harmonic_basis", [realOr](const SimplicialQubit &q) { return realOr(q, q.harmonicBasis()); },
             "H: nE x 2; real on the real locus.")
        .def("dual_harmonic_basis",
             [realOr](const SimplicialQubit &q) { return realOr(q, q.dualHarmonicBasis()); },
             "H^vee: nE x 2, the kernel twisted by the inverse links the pairings are "
             "taken against; harmonic_basis() itself on the trivial connection.")
        .def("complex_structure", [realOr](const SimplicialQubit &q) { return realOr(q, q.complexStructure()); },
             "J: 2 x 2; real on the real locus.")
        .def("j_residual", &SimplicialQubit::jResidual, "||J @ J + I||_F.")
        .def("holomorphic_form", &SimplicialQubit::holomorphicForm, "omega: nE complex.")
        .def("periods", &SimplicialQubit::periods,
             "(P_A, P_B); transported from base_vertex() under a nontrivial connection.")
        .def("tau", &SimplicialQubit::tau, "P_B / P_A.")
        .def("tau_derivative", &SimplicialQubit::tauDerivative,
             "d tau / d z_e for every edge in edge order, z_e = l_e^2: the holomorphic derivative "
             "of tau() with respect to each squared edge length, from the complex structure "
             "in the period frame and its dual (no eigen-decomposition and no null-space basis "
             "is differentiated); scale-free (sum_e z_e d tau/d z_e = 0) and gauge invariant. "
             "Raises RuntimeError when the two eigenlines of J coincide.")
        .def("intersection_number", &SimplicialQubit::intersectionNumber,
             "A . B of the marked cycles on the surface as oriented by the faces: the ordered cup "
             "product of the reference period frame on the fundamental cycle, +1 or -1 to rounding "
             "(+1 on flat_torus, -1 under reversed=True).")
        .def("period_frame", [realOr](const SimplicialQubit &q) { return realOr(q, q.periodFrame()); },
             "F: nE x 2 (real on the real locus), the period frame: the basis (f_A, f_B) of the harmonic "
             "space with periods (1, 0) and (0, 1) over the marking in force, harmonic_basis() times "
             "the inverse period matrix, in the torus's edge order. The coordinates a state is written "
             "in: holomorphic_form() == period_frame() @ (P_A, P_B), i.e. P_A * F @ (1, tau), the qubit "
             "|0> + tau|1> read as a 1-form (f_A <-> |0>, f_B <-> |1>); the frame a two-body target "
             "chi compares with the transfer in (MultiCobordism.set_input_frame). Under a nontrivial "
             "connection the periods are the transported ones from base_vertex().")
        .def("state", &SimplicialQubit::state, "(|0> + tau|1>) / sqrt(1 + |tau|^2).")
        .def("bloch", &SimplicialQubit::bloch, "The Bloch vector, a unit vector for every tau.")
        .def("density_matrix", &SimplicialQubit::densityMatrix, "rho = (I + r . sigma) / 2.")
        .def_static("flat_torus",
                    [emitWarnings](std::complex<double> tau, int nx, int ny) {
                      SimplicialQubit q = SimplicialQubit::flatTorus(tau, nx, ny);
                      emitWarnings(q);
                      return q;
                    },
                    py::arg("tau"), py::arg("nx"), py::arg("ny"),
                    "The flat torus C / (Z + tau Z): unit cell spanned by 1 and tau as an nx x ny "
                    "grid split by its diagonals, sides identified, marked by the row loop A and "
                    "the column loop B. Exact: returns tau to rounding.")
        // ---- the optional preprocessing pass
        .def("intrinsic_delaunay",
             [emitWarnings](const SimplicialQubit &q) {
               SimplicialQubit r = q.intrinsicDelaunay();
               emitWarnings(r);
               return r;
             },
             "Flip every edge with alpha_e + beta_e > pi until none remains (marked cycles "
             "rerouted); returns the qubit of the flipped triangulation. Real locus only.")
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
             "Cycle A as a closed walk of directed vertex steps; starts at base_vertex() "
             "under a nontrivial connection.")
        .def("walk_B", &SimplicialQubit::walkB, "Cycle B as a closed walk.")
        .def("base_vertex", &SimplicialQubit::baseVertex,
             "The common base point of the transported periods: the first vertex of A's walk on B's.")
        .def("degeneracy_threshold", &SimplicialQubit::degeneracyThreshold)
        .def("spacetime", &SimplicialQubit::spacetime, "The Spacetime holding vertices, edges, lengths, phases.")
        .def("on_real_locus", &SimplicialQubit::onRealLocus,
             "True when every length is real and every link is 1: the real-number construction ran, "
             "bit-identical to the real-length one.")
        .def("trivial_connection", &SimplicialQubit::trivialConnection, "True when every link is 1.")
        // ---- intermediate quantities
        .def("d0", &SimplicialQubit::d0, "nE x nV cells.")
        .def("d1", &SimplicialQubit::d1, "nF x nE cells.")
        .def("angles", [realOr](const SimplicialQubit &q) { return realOr(q, q.angles()); },
             "Per face (alpha_i, alpha_j, alpha_k); the principal acos off the real locus.")
        .def("areas", [realOrVector](const SimplicialQubit &q) { return realOrVector(q, q.areas()); },
             "Per face the Heron area; the continuation branch off the real locus.")
        .def("layout", [realOr](const SimplicialQubit &q) { return realOr(q, q.layout()); },
             "Per face (p_i, p_j, p_k) in its local frame.")
        .def("barycentric_gradients",
             [realOr](const SimplicialQubit &q) { return realOr(q, q.barycentricGradients()); },
             "Per face (grad lambda_i, grad lambda_j, grad lambda_k).")
        .def("weights", [realOrVector](const SimplicialQubit &q) { return realOrVector(q, q.weights()); },
             "The cotangent weights w_e.")
        .def("negative_weight_edges", &SimplicialQubit::negativeWeightEdges)
        .def("non_delaunay_edges", &SimplicialQubit::nonDelaunayEdges,
             "Edges with alpha_e + beta_e > pi; evaluated on the real locus only.")
        .def("gram", [realOr](const SimplicialQubit &q) { return realOr(q, q.gram()); },
             "G[a][b] = <h_a, h_b>: the transpose pairing, between the dual kernel and the "
             "kernel under a nontrivial connection.")
        .def("rotation_pairing", [realOr](const SimplicialQubit &q) { return realOr(q, q.rotationPairing()); },
             "R[a][b] = sum_t A_t rot90(W_t(h_a)) . W_t(h_b).")
        .def("marking_swapped", &SimplicialQubit::markingSwapped,
             "True when |P_A| vanished and -1/tau of the given marking is reported.")
        .def("condition_m1", &SimplicialQubit::conditionM1, "cond(M1) of the cotangent weights.")
        .def("condition_g", &SimplicialQubit::conditionG, "cond(G) of the harmonic Gram matrix.")
        .def("near_degenerate", &SimplicialQubit::nearDegenerate)
        .def("warnings", &SimplicialQubit::warnings, "Every warning the construction raised.");
    m.def("fubini_study_distance", &SimplicialQubit::fubiniStudyDistance, py::arg("q1"), py::arg("q2"),
          "arccos(|1 + conj(tau1) tau2| / sqrt((1+|tau1|^2)(1+|tau2|^2))): distinguishability, "
          "curvature +4.");
    m.def("weil_petersson_distance", &SimplicialQubit::weilPeterssonDistance, py::arg("q1"), py::arg("q2"),
          "arccosh(1 + |tau1 - tau2|^2 / (2 Im tau1 Im tau2)): the moduli distance between shapes, "
          "curvature -1.");
  }
}
