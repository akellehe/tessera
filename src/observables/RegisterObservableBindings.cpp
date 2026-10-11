// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the observables subsystem: the live complex, register
// context, register observables and gates. One of the translation units that
// Bindings.cpp registers in order
// (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

void register_observables_register_observable(py::module_ &m) {
  // ==========================================================================
  // Emergent-proton readout battery: pure readers over a live complex, the
  // loader/transform layer, and the gauge/relabel gates.
  // ==========================================================================

  // ---- LiveComplex: the loader / transform layer (outside the readers) ----
  py::class_<LiveComplex::Relabeled>(m, "Relabeled",
      "A relabeled rebuild: the live relabeled complex + the vertex-id "
      "permutation (original id -> relabeled id).")
      .def_readonly("spacetime", &LiveComplex::Relabeled::spacetime)
      .def_readonly("vertex_map", &LiveComplex::Relabeled::vertexMap);

  py::class_<LiveComplex>(m, "LiveComplex",
      R"doc(The loader / transform layer outside the pure readers: loads a
saved combinatorial and metric description back into a live,
skeleton-complete Spacetime, and produces a relabeled copy for the relabel
gate. Never builds a spacetime of its own and never re-runs the dynamics
(those live in ProtonSynthesis / ProtonIngredients / MultiCobordism); it
reads a recorded geometry back through ``Spacetime.from_vertex_tuples``, completing
the facet skeleton with ``materialize_facets``.)doc")
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
                  "permutation (the relabel-gate transform).");

  // ---- RegisterContext: the validated read context (pure reader) ----
  py::class_<RegisterContext, std::shared_ptr<RegisterContext>>(
      m, "RegisterContext",
      R"doc(The one validated read context every emergent-proton observable
measures: a live, already-built complex, its emergent holes, the
induced-orientation signs, and the shared per-complex caches. A pure reader:
it never builds, solves, or materializes anything.)doc")
      .def(py::init([](std::shared_ptr<Spacetime> st, int count, int degree,
                       std::vector<std::complex<double>> target) {
             auto ctx = std::make_shared<RegisterContext>(
                 std::move(st), count, degree, std::move(target));
             emitSelectionWarning(*ctx);
             return ctx;
           }),
           py::arg("spacetime"), py::arg("count") = 3, py::arg("degree") = 3,
           py::arg("target") = ::tessera::cobordism::ProtonSynthesis::singlet())
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
      .def("b_k", &RegisterContext::bK)
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
      "The whole-complex singlet diagnostic (headline = singlet r_state).")
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
      "Per-output-block carry residuals (blocks are constructor provenance).")
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
      "The mass half of the relaxed 4D interior read (headline = m_shell).")
      .def(py::init<>())
      .def("masses", &EmergentMass::masses, py::arg("ctx"))
      .def("localization", &EmergentMass::localization, py::arg("ctx"));

  py::class_<EmergentRadius, RegisterObservable,
             std::shared_ptr<EmergentRadius>>(m, "EmergentRadius",
      "The radius half of the relaxed 4D interior read (headline = r_dual).")
      .def(py::init<>())
      .def("radii", &EmergentRadius::radii, py::arg("ctx"));

  py::class_<PairLoopFlavor::JointRead>(m, "PairLoopJointRead")
      .def_readonly("sigma", &PairLoopFlavor::JointRead::sigma)
      .def_readonly("r_u", &PairLoopFlavor::JointRead::rU)
      .def_readonly("w", &PairLoopFlavor::JointRead::w)
      .def_readonly("hole_intensity", &PairLoopFlavor::JointRead::holeIntensity,
                    "Per-hole boundary intensity: the squared Hodge norm of "
                    "the carried representative over the hole's boundary "
                    "facets. A norm of the representative and nothing else -- "
                    "not the charge of any current, and no Kahler-Dirac "
                    "operator exists here to give it that reading.")
      .def_readonly("loop_w", &PairLoopFlavor::JointRead::loopW)
      .def_readonly("loop_intensity", &PairLoopFlavor::JointRead::loopIntensity,
                    "Per-pair-loop boundary intensity: the same squared Hodge "
                    "norm over the union of the two holes' boundary facets.")
      .def_readonly("dual_residual", &PairLoopFlavor::JointRead::dualResidual);
  py::class_<PairLoopFlavor::Verdict>(m, "PairLoopVerdict")
      .def_readonly("odd_loop", &PairLoopFlavor::Verdict::oddLoop,
                    "The pair loop whose boundary intensity sits farthest "
                    "from the mean of the other two.")
      .def_readonly("dual_hole", &PairLoopFlavor::Verdict::dualHole)
      .def_readonly("rho", &PairLoopFlavor::Verdict::rho)
      .def_readonly("multiplicity_2_1", &PairLoopFlavor::Verdict::multiplicity21)
      .def_readonly("odd_is_diquark_loop",
                    &PairLoopFlavor::Verdict::oddIsDiquarkLoop);

  py::class_<PairLoopFlavor, RegisterObservable,
             std::shared_ptr<PairLoopFlavor>>(m, "PairLoopFlavor",
      "The pair-loop dual-basis flavor read (headline = rho).")
      .def(py::init<>())
      .def(py::init([](std::pair<int, int> diquark) {
             return std::make_shared<PairLoopFlavor>(diquark);
           }),
           py::arg("diquark_pair"))
      .def("joint_read", &PairLoopFlavor::jointRead, py::arg("ctx"))
      .def("evaluate_criteria", &PairLoopFlavor::evaluateCriteria,
           py::arg("read"))
      .def_static("odd_one_out", &PairLoopFlavor::oddOneOut,
                  py::arg("loop_intensity"))
      .def_static("complement_hole", &PairLoopFlavor::complementHole,
                  py::arg("pair"));
  m.attr("PairLoopFlavor").attr("RHO_MAX") = PairLoopFlavor::RHO_MAX;

  // ---- the self-test probes ----
  py::class_<LabelLeakProbe, RegisterObservable,
             std::shared_ptr<LabelLeakProbe>>(m, "LabelLeakProbe",
      "A deliberately label-dependent probe; the relabel gate must flag it.")
      .def(py::init<>());
  py::class_<GaugeLeakProbe, RegisterObservable,
             std::shared_ptr<GaugeLeakProbe>>(m, "GaugeLeakProbe",
      "A deliberately gauge-dependent probe; the gauge gate must flag it.")
      .def(py::init<>());

  // ---- the gauge/relabel gate harness ----
  py::class_<ObservableGates::GateResult>(m, "GateResult")
      .def_readonly("gauge_delta", &ObservableGates::GateResult::gaugeDelta)
      .def_readonly("relabel_delta", &ObservableGates::GateResult::relabelDelta)
      .def_readonly("gate_tol", &ObservableGates::GateResult::gateTol)
      .def_readonly("gauge_ok", &ObservableGates::GateResult::gaugeOk)
      .def_readonly("relabel_ok", &ObservableGates::GateResult::relabelOk);

  py::class_<ObservableGates>(m, "ObservableGates",
      "The gauge/relabel gate harness: post-hoc validation, never a loop "
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
}
