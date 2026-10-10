// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// Python bindings of the cobordism subsystem: the objective a MultiCobordism
// descends (include/cobordism/CobordismObjective.h) -- its terms, context,
// scope and direction, and the objectives Python can subclass. One of the
// translation units that Bindings.cpp registers in order (https://github.com/akellehe/tessera/issues/1453).

#include "Bindings.h"

/// # PyCobordismObjective
///
/// The trampoline that lets a Python subclass supply the functional
/// `MultiCobordism` descends. Each override forwards to the Python method of
/// the corresponding snake_case name; the three with C++ defaults fall back to
/// the base implementation when a subclass does not define them.
///
/// The firewall survives the crossing: a Python objective is handed an
/// `ObjectiveContext`, plain data carrying geometry, a region, that region's
/// targets and scalar configuration. It receives no node and no callable that
/// closes over one, so no route to a component, fiber, transport, colour,
/// charge, flavour, exchange, spin certificate or verdict.
///
/// The override macros acquire the GIL themselves, so an engine entry point
/// that released it re-enters Python safely.
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

void register_cobordism_objective(py::module_ &m) {
  py::class_<ObjectiveTerms>(m, "ObjectiveTerms",
      "The COMPLETE, enumerable term list the scalar objective is the sum of. "
      "MultiCobordism.objective_of is static over this record, so the objective "
      "provably reads nothing else -- the structural half of the no-feedback "
      "firewall.")
      .def(py::init<>())
      .def_readwrite("regge_stationarity",
                     &ObjectiveTerms::reggeStationarity)
      .def_readwrite("hodge_stationarity",
                     &ObjectiveTerms::hodgeStationarity)
      .def_readwrite("connection_stationarity",
                     &ObjectiveTerms::connectionStationarity,
                     "eta_C ||grad_phi S||^2 of the C* connection operator -- "
                     "the ONLY term with a gradient in the connection phase: the "
                     "Hodge-entropy term sees phi through h_k(z, U) but is "
                     "differentiated in z alone, so without this the phase is "
                     "a declared field no update can move.")
      .def_readwrite("register_residual",
                     &ObjectiveTerms::registerResidual)
      .def_readwrite("action_magnitude",
                     &ObjectiveTerms::actionMagnitude)
      .def_readwrite("carried_state_energy",
                     &ObjectiveTerms::carriedStateEnergy)
      .def_readwrite("moment_stiffness",
                     &ObjectiveTerms::momentStiffness);

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
                           &ObjectiveTermName::kCarriedStateEnergy)
      .def_readonly_static("MOMENT_STIFFNESS",
                           &ObjectiveTermName::kMomentStiffness);

  py::class_<ObjectiveContext>(m, "ObjectiveContext",
      "The COMPLETE set of inputs an objective may read -- the no-feedback "
      "firewall restated as an input type. Plain data: geometry, a region, "
      "that region's declared targets, configured weights, and precomputed "
      "geometric scalars. No MultiCobordism reference and deliberately no "
      "callable, since a bound callable would capture the node and smuggle "
      "back the reachability the former static objective_of denied.")
      .def(py::init<>())
      // Every field is readable and every one is data; no analysis product is
      // exposed here.
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
      .def_readwrite("fiber_residuals", &ObjectiveContext::fiberResiduals,
                     "Whether the node scores its blocks by fiber residuals (use_fiber_residuals), "
                     "whose stage-2 direction is analytic: the legacy objective then descends r_U "
                     "next to the Regge term instead of only gating on it. False by default.")
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
      .def_readwrite("moment_stiffness_weight", &ObjectiveContext::momentStiffnessWeight,
                     "beta_M, the weight of the spectral-moment stiffness of the geometric action. Zero by default.")
      .def_readwrite("moment_stiffness_degrees", &ObjectiveContext::momentStiffnessDegrees,
                     "The degrees k whose Hodge operators' local moments are held.")
      .def_readwrite("moment_stiffness_coefficients", &ObjectiveContext::momentStiffnessCoefficients,
                     "beta_j, j = 1..m, the weights of the moment orders.")
      .def_readwrite("moment_stiffness_reference", &ObjectiveContext::momentStiffnessReference,
                     "The carrier's local moments, one flat |C_k| x m array per degree.")
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
      "objective this class describes, and the only built-in that is not "
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
}
