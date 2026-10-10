// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: spectral gap, harmonic
// dimension and Wilson loops. One of the translation units that Bindings.cpp
// registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_spectral_gap_wilson(py::module_ &m) {
  // ========================================
  // Hodge spectral observables: scalars over cobordism::HodgeLaplacian
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
  //   K. G. Wilson, "Confinement of quarks", Phys. Rev. D 10, 2445 (1974).
  //   Greensite, "The Confinement Problem in Lattice Gauge Theory",
  //   arXiv:hep-lat/0301023.
  // ========================================
  py::enum_<WilsonMode>(m, "WilsonMode",
      "Evaluation mode for Wilson loops.")
      .value("COMBINATORIAL", WilsonMode::COMBINATORIAL,
             "Dual-graph topology only (loop length, enclosed hinges).")
      .value("DEFICIT_ANGLE", WilsonMode::DEFICIT_ANGLE,
             "Deficit-angle based: W = ((d-2)+2cos(epsilon))/d.")
      .value("CAUSAL", WilsonMode::CAUSAL,
             "Causal-orientation changes around the loop.")
      .value("U1_CONNECTION", WilsonMode::U1_CONNECTION,
             "U(1) connection holonomy: oriented sum of Edge.phase around "
             "a 1-skeleton vertex cycle, reduced mod 2*pi. The Wilson-loop "
             "view of the cobordism.HodgeLaplacian cycle flux.");

  py::enum_<LoopType>(m, "LoopType",
      "Which loop-shape generator to use.")
      .value("HINGE", LoopType::HINGE,
             "Elementary loop around a (d-2)-simplex.")
      .value("DUAL_LATTICE", LoopType::DUAL_LATTICE,
             "Breadth-first-search loop of a target size.")
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
                    "derived view of connection_accumulation (its "
                    "residual_phase()), not a second datum.")
      .def_readonly("connection_accumulation",
                    &WilsonResult::connectionAccumulation,
                    "U1_CONNECTION mode: the complete gauge-invariant datum -- "
                    "the unreduced complex accumulation of the oriented edge "
                    "phase around the cycle. Both components are carried: "
                    "around a closed loop a gauge transformation telescopes to "
                    "zero, so the whole complex sum is gauge-invariant. Only Re "
                    "quantizes, which makes e^{-Im} a gauge-invariant real "
                    "rather than a quantum number. Never reduced mod 2*pi -- "
                    "reducing would destroy the winding. NaN in other modes: "
                    "unmeasured, never zero.")
      .def("holonomy", &WilsonResult::holonomy,
           "Derived: the holonomy exp(i * connectionAccumulation).")
      .def("holonomy_modulus", &WilsonResult::holonomyModulus,
           "Derived: |H| = exp(-Im accumulation). Exactly 1 for a purely "
           "compact connection -- a cancellation to be observed, not imposed.")
      .def("residual_phase", &WilsonResult::residualPhase,
           "Derived: Re(accumulation) mod 2*pi, in (-pi, pi].")
      .def("winding_number", &WilsonResult::windingNumber,
           "Derived: whole 2*pi turns in Re(accumulation). Recoverable only "
           "because the accumulation is stored unreduced.")
      .def_readonly("loop_size", &WilsonResult::loopSize,
                    "Number of simplices in the loop.")
      .def_readonly("enclosed_hinges", &WilsonResult::enclosedHinges,
                    "Hinges enclosed by the loop.")
      .def_readonly("contractible", &WilsonResult::contractible,
                    "Whether the loop is contractible.")
      .def_readonly("causal_winding_number", &WilsonResult::causalWindingNumber,
                    "Net time-orientation changes (causal mode).");

  py::class_<WilsonLoop, std::shared_ptr<WilsonLoop>>(m, "WilsonLoop",
      R"doc(Wilson loop observable on a triangulated spacetime.

A Wilson loop is the trace of a parallel-transport operator around a
closed path. Without an explicit gauge field the Levi-Civita holonomy
analogue is computed: closed walks on the dual graph (top-simplices as
nodes, shared facets as edges), with the loop value set by the deficit
angles of the enclosed hinges.

Evaluation modes:

* ``COMBINATORIAL`` — dual-graph topology only. ``value`` is the loop
  length; ``enclosed_hinges`` counts hinges contained in every loop
  simplex; ``contractible`` is True iff ``enclosedHinges == 0``.
* ``DEFICIT_ANGLE`` — Regge-curvature holonomy. For a hinge loop
  enclosing one hinge h:
      W = ((d-2) + 2 cos(eps_h)) / d
  For multi-hinge loops the U(1) approximation:
      W = product_{h in enclosed} cos(eps_h).
  W = 1 is a flat loop; deviation from 1 measures local curvature.
* ``CAUSAL`` — causal-orientation winding. ``causal_winding_number`` is the
  signed net change in foliation index around the loop; nonzero values
  mark loops that cross a causal dynamical triangulation (CDT) slice
  boundary.
* ``U1_CONNECTION`` — U(1) connection holonomy. The oriented sum of the
  ``Edge.phase`` carried on the primal 1-skeleton around a closed vertex
  cycle (``+phase`` along the stored source->target orientation,
  ``-phase`` reversed), reduced mod 2*pi. Evaluated via
  ``evaluate_u1_connection(cycle)``; the connection is a primal-edge, not a
  dual-graph, quantity. ``value`` carries the holonomy.

Loops come from ``hinge_loop``, ``dual_lattice_loop`` or ``geodesic_loop``.
``measure()`` and ``measure_all_hinges()`` append ``WilsonResult`` entries to
an internal list; ``get_measurements()`` returns it, ``get_average_by_size()``
aggregates by loop length, and ``reset()`` clears it.

See ``docs/source/wilson_loops.md`` for a tutorial.
)doc")
      .def(py::init<std::shared_ptr<Spacetime>>(), py::arg("spacetime"),
           "Construct a Wilson-loop calculator bound to a Spacetime.")
      .def("evaluate", &WilsonLoop::evaluate,
           py::arg("loop"), py::arg("mode"),
           R"doc(Evaluate the Wilson loop in the given mode.

Dispatches to ``evaluate_combinatorial``, ``evaluate_deficit_angle``, or
``evaluate_causal`` depending on ``mode``.
)doc")
      .def("evaluate_combinatorial", &WilsonLoop::evaluateCombinatorial,
           py::arg("loop"),
           R"doc(Evaluate using dual-graph topology only.

Returns a ``WilsonResult`` with ``value = loop_size``,
``enclosed_hinges`` = count of hinges shared by every loop simplex, and
``contractible`` = True iff no hinge is enclosed.
)doc")
      .def("evaluate_deficit_angle", &WilsonLoop::evaluateDeficitAngle,
           py::arg("loop"),
           R"doc(Evaluate using Regge deficit angles.

For a hinge loop (exactly one enclosed hinge h):
    W = ((d - 2) + 2 cos(eps_h)) / d.
For multi-hinge loops the U(1) approximation:
    W = product_{h in enclosed} cos(eps_h).
``value`` carries W; ``enclosed_hinges`` carries the count.
)doc")
      .def("evaluate_causal", &WilsonLoop::evaluateCausal,
           py::arg("loop"),
           R"doc(Evaluate using CDT causal-orientation changes.

Walks the loop and accumulates a signed winding count from the
final-time stamps of consecutive simplices. ``causal_winding_number`` is
the net winding; ``value`` carries the same number as a double.
)doc")
      .def("evaluate_u1_connection", &WilsonLoop::evaluateU1Connection,
           py::arg("cycle"),
           R"doc(U(1) connection holonomy around a closed vertex cycle.

``cycle`` is an ordered list of vertices on the primal 1-skeleton whose
consecutive pairs (with wrap-around) are joined by edges. Each edge's
``phase`` is accumulated along its stored source->target orientation
(``+phase`` forward, ``-phase`` reversed); ``value`` is the total reduced
into ``(-pi, pi]`` and ``loop_size`` is the number of edges. Returns an empty
result (``loopSize == 0``) for a degenerate (fewer than two vertices) or
open (a consecutive pair with no joining edge) cycle.

This is the same oriented phase sum as the cycle flux of the
Hermitian-weighted ``cobordism.HodgeLaplacian``. Restricted to phases in
``{0, pi}`` the holonomy lands in ``{0, pi}`` and reproduces the Z2 flux.
)doc")
      .def("hinge_loop", &WilsonLoop::hingeLoop,
           py::arg("hinge"),
           R"doc(Loop of top-simplices around a hinge, ordered cyclically.

Encloses exactly one hinge (the input). This is the natural loop for
``DEFICIT_ANGLE`` mode and is what ``measure_all_hinges`` uses internally.
)doc")
      .def("dual_lattice_loop", &WilsonLoop::dualLatticeLoop,
           py::arg("start"), py::arg("target_length"),
           R"doc(Breadth-first-search loop of approximately ``target_length``
simplices.

Not guaranteed to be exactly ``target_length``: the search may overshoot or
return a shorter loop if local connectivity does not permit closing at the
target size. Suitable for population-level scans at a fixed loop scale
(the analogue of a Wilson-loop side length in lattice gauge theory).
)doc")
      .def("geodesic_loop", &WilsonLoop::geodesicLoop,
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
      .def("measure_all_hinges", &WilsonLoop::measureAllHinges,
           py::arg("mode"),
           R"doc(Walk every (d-2)-simplex of the spacetime, generate its hinge
loop, and record the evaluation in ``mode``. Skips degenerate hinges whose
loop has fewer than 2 distinct simplices.
)doc")
      .def("reset", &WilsonLoop::reset,
           "Clear all accumulated measurements.")
      .def("get_measurements", &WilsonLoop::getMeasurements,
           "Return the full list of accumulated ``WilsonResult`` entries.")
      .def("get_average_by_size", &WilsonLoop::getAverageBySize,
           R"doc(Mean ``value`` grouped by loop size, as a ``{size: mean}`` dict.

The standard form for Creutz-ratio analyses: fix loop size L, read off the
population-averaged Wilson value at that scale.
)doc");
}
