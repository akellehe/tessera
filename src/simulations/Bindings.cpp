// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>

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
#include "simulations/InteractionSimulation.h"
#include "matter/MatterConfiguration.h"
#include "mesh/SimplexFilter.h"
#include "observables/ModularityOptimizer.h"
#include "observables/SparseGraph.h"
#include "observables/VolumeProfile.h"
#include "observables/WilsonLoop.h"
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
using namespace tessera::simulations;

// Registers all tessera::simulations classes into the `m` submodule
// (i.e. `tessera.simulations`). Called from src/bindings.cpp's
// PYBIND11_MODULE entry point.
void register_simulations(py::module_ m) {
  // ========================================
  // CDTSimulation
  // ========================================
  py::class_<CDT, std::shared_ptr<CDT> >(m, "CDTSimulation",
      R"doc(Causal Dynamical Triangulations Monte Carlo simulation.

Implements the five Pachner moves (add, remove, flip, iflip, shift) with
Metropolis-Hastings acceptance, including the combinatorial prefactors.

The Regge action is::

    S = -(k0 + 6*delta)*N0 + (k4 + 2*delta)*N41
        + (k4 + delta)*N32 + epsilon*(N41 - target)^2

Reference: Ambjorn, Jurkiewicz & Loll, "Reconstructing the Universe",
arXiv:hep-th/0505154.

Args:
    spacetime: A built Spacetime object.
    k0: Bare inverse Newton's constant.
    k4: Cosmological constant coupling (tuned to pseudo-critical value).
    delta: Asymmetry parameter between timelike and spacelike edges.
    epsilon: Volume-fixing strength.
    targetN41: Target (d,1)-type four-volume for volume-fixing.
    quadraticVolumeFix: If True (default), use epsilon*(N41 - target)^2;
        if False, use epsilon*|N41 - target|.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, double, double, double, double, std::size_t, bool>(),
           py::arg("spacetime"),
           py::arg("k0"),
           py::arg("k4"),
           py::arg("delta"),
           py::arg("epsilon"),
           py::arg("target_n41"),
           py::arg("quadratic_volume_fix") = true)
      .def("add", &CDT::add,
           R"doc(Attempt one (2,2d) vertex insertion move.

Picks a random N41 simplex, finds a spatial face with a partner, and
proposes inserting a new vertex.  Accepted via Metropolis with prefactor
N41/(N0+1).  The new vertex is relabeled uniformly on acceptance when
set_relabel_vertices(True) is set.

Returns True if accepted, False if rejected.)doc")
      .def("remove", &CDT::remove,
           R"doc(Attempt one (2d,2) vertex deletion move.

Picks a random vertex, checks whether it has order 2d (all N41-type), and
proposes removing it.  Inverse of add().

Returns True if accepted, False if rejected.)doc")
      .def("flip", &CDT::flip,
           R"doc(Attempt one (2,d) flip move.

Picks a random top simplex, picks a random facet, and proposes
replacing the 2 simplices sharing that facet with d new simplices.
dN0 = 0, dN4 = d - 2 = +2 in 4D.

Returns True if accepted, False if rejected.)doc")
      .def("iflip", &CDT::iflip,
           R"doc(Attempt one (d,2) inverse flip move.

Picks a random top simplex, picks a random edge, and proposes
replacing the d simplices sharing that edge with 2 new simplices.
dN0 = 0, dN4 = -(d - 2) = -2 in 4D.  Inverse of flip().

Returns True if accepted, False if rejected.)doc")
      .def("shift", &CDT::shift,
           R"doc(Attempt one (3,3) shift move.

Replaces 3 simplices sharing a (d-2)-face with 3 new simplices
sharing the complementary (d-2)-face.  Self-inverse: dN0 = 0, dN4 = 0.
Combinatorial prefactor is 1 (symmetric selection).

Returns True if accepted, False if rejected.)doc")
      .def("propose_add", &CDT::proposeAdd,
           R"doc(Construct a transactional AddMove bound to this
simulation's spacetime and RNG, with propose() already called.  Returns
None if there is no eligible target.  The caller drives apply()/rollback().
Does not update acceptance counters.)doc")
      .def("propose_remove", &CDT::proposeRemove,
           "Like propose_add() for the (2d,2) remove move.")
      .def("propose_flip", &CDT::proposeFlip,
           "Like propose_add() for the (2,d) flip move.")
      .def("propose_iflip", &CDT::proposeIflip,
           "Like propose_add() for the (d,2) inverse-flip move.")
      .def("propose_shift", &CDT::proposeShift,
           "Like propose_add() for the (3,3) shift move.")
      .def("ishift", &CDT::ishift,
           R"doc(Attempt one inverse shift move (same as shift, since (3,3) is self-inverse).

Returns True if accepted, False if rejected.)doc")
      .def("sweep", [](CDT &self, int nSweeps, py::object progress) {
          int total = 0;
          if (progress.is_none()) {
              // No callback — release the GIL for the entire loop so
              // multiple threads get true parallelism.
              py::gil_scoped_release release;
              for (int i = 0; i < nSweeps; i++) {
                  total += self.sweep();
              }
          } else {
              for (int i = 0; i < nSweeps; i++) {
                  int accepted;
                  {
                      py::gil_scoped_release release;
                      accepted = self.sweep();
                  }
                  total += accepted;
                  progress(i + 1, nSweeps);
              }
          }
          return total;
      }, py::arg("n_sweeps") = 1, py::arg("progress") = py::none(),
           R"doc(Run one or more Monte Carlo sweeps.

Each sweep proposes N4 moves uniformly among all 5 types
(add, remove, flip, iflip, shift).

Args:
    nSweeps: Number of sweeps to perform (default 1).
    progress: Optional callback(i, n) called after each sweep.

Returns the total number of accepted moves across all sweeps.)doc")
      .def("tune", [](CDT &self, py::object progress) {
          if (progress.is_none()) {
              py::gil_scoped_release release;
              self.tune();
          } else {
              self.tune([&](int i, int n) {
                  py::gil_scoped_acquire acquire;
                  progress(i, n);
              });
          }
      }, py::arg("progress") = py::none(),
           R"doc(Tune k4 to its pseudo-critical value.

Estimates k4 from the coupling constants, then locates the coupling at which
the four-volume drift changes sign: brackets it in doubling steps and
bisects. Measurements run with the volume-fixing term inactive; the
configured epsilon is restored before returning. Call this before sweep().

Args:
    progress: Optional callback(i, n) called once per drift measurement.)doc")
      .def("thermalize", &CDT::thermalize, py::call_guard<py::gil_scoped_release>(),
           R"doc(Thermalize the simulation until the action stabilizes.

Runs sweeps until the relative change in action is < 1%, with a minimum
of 20 sweeps.  Use after tune() to reach thermal equilibrium before
taking measurements.)doc")
      .def("compute_action", &CDT::computeAction,
           R"doc(Compute the Regge action S from the current counts.

S = -(k0 + 6*delta)*N0 + (k4 + 2*delta)*N41
    + (k4 + delta)*N32 + volume_fix_term)doc")
      .def("get_volume_profile", &CDT::getVolumeProfile, py::call_guard<py::gil_scoped_release>(),
           R"doc(Return the spatial volume profile as a list of simplex counts per time slice.

Each entry is the number of top simplices whose minimum vertex time
equals that slice.  The sum equals N4.)doc")
      .def("get_acceptance_rates", &CDT::getAcceptanceRates, py::call_guard<py::gil_scoped_release>(),
           R"doc(Return acceptance rates for each move type as a dict.

Keys: 'add', 'remove', 'flip', 'iflip', 'shift', 'ishift'.
Values: fraction of attempts accepted (0.0 to 1.0).)doc")
      .def("get_spacetime", &CDT::getSpacetime,
           "Return the underlying Spacetime object.")
      .def("get_k0", &CDT::getK0,
           "Return the bare inverse Newton's constant k0.")
      .def("get_k4", &CDT::getK4,
           "Return the current cosmological coupling k4 (may differ from initial after tune).")
      .def("get_delta", &CDT::getDelta,
           "Return the asymmetry parameter delta.")
      .def("set_relabel_vertices", &CDT::setRelabelVertices, py::arg("enabled"),
           R"doc(Enable or disable vertex relabeling after add/remove moves.

Disabled by default: in the labeled formalism the acceptance ratio already
carries the uniform-labeling requirement, and the swap walks a list whose
length grows with the four-volume.)doc")
      .def("set_seed",
           [](CDT& self, std::uint32_t seed) { self.setSeed(seed); },
           py::arg("seed"),
           R"doc(Re-seed the internal RNG.

The default constructor pulls a seed from std::random_device — fine
for production MC sweeps but flaky for tests whose outcome depends
on a specific growth pattern. Pass a fixed seed at the top of such
tests to make them reproducible.)doc");
  // ========================================
  // ReggeSolver
  // ========================================
  py::class_<ReggeSolver>(m, "ReggeSolver",
      R"doc(Regge equation solver.

Adjusts edge lengths so that the Regge equations (∂S/∂ℓ² = 0) are satisfied.
The total action is S = S_grav + S_matter where:
  S_grav = Σ_h |h| ε_h   (Regge gravitational action: hinge content × deficit)
  S_matter = -M Σ √(-ℓ²)  (proper-time action along worldlines)

Minimizes F = ||∇S||² to find stationary points of S (the discrete
Einstein equations).  F ≥ 0, and F = 0 at the solution.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, ::tessera::matter::MatterConfiguration>(),
           py::arg("spacetime"), py::arg("matter"))
      .def("dihedral_angle", &ReggeSolver::dihedralAngle,
           py::arg("sigma"), py::arg("hinge"),
           "Dihedral angle at hinge h within top-simplex sigma.")
      .def("deficit_angle", &ReggeSolver::deficitAngle,
           py::arg("hinge"),
           "Deficit angle at a hinge: 2π minus sum of dihedral angles.")
      .def_static("hinge_content", &ReggeSolver::hingeContent,
           py::arg("hinge"),
           "The (d-2)-content of a hinge, the weight of its deficit angle in the "
           "primal Regge action: the length of an edge hinge on a 3-dimensional "
           "mesh, the area of a triangular hinge on a 4-dimensional mesh, the "
           "volume of a tetrahedral hinge on a 5-dimensional mesh "
           "(Simplex.volume). Complex: a timelike hinge's content is imaginary.")
      .def("regge_action", &ReggeSolver::reggeAction,
           "Gravitational Regge action: S_grav = Σ_h |h| · ε_h, with |h| the "
           "(d-2)-content of the hinge (hingeContent) and ε_h its deficit angle.")
      .def("dual_regge_action", &ReggeSolver::dualReggeAction,
           "Dual Lorentzian Regge action S_Regge(W*) = Σ_h |*h| · ε_h: each "
           "(d-2)-hinge's circumcentric dual content (Simplex.dual_volume) times "
           "its complex Lorentzian deficit. Returns a Python complex (real = "
           "angle-defect curvature, imag = boost/light-cone content).")
      .def("hinge_faces_of_cells", &ReggeSolver::hingeFacesOfCells, py::arg("cells"),
           "The (d-2) hinge vertex-tuples that are faces of the given top d-cells "
           "— the affected-hinge index for the incremental ΔS_Regge. `cells` is a "
           "list of vertex-id tuples; returns the dedup'd sorted (d-1)-vertex "
           "sub-tuples. Build from a move's touched cells (created ∪ removed ∪ the "
           "perturbed edge's top cofaces) and reuse the same set before/after.")
      .def("dual_regge_action_over_hinges", &ReggeSolver::dualReggeActionOverHinges,
           py::arg("hinges"),
           "Localized dual Regge action Σ |*h|·ε_h over only the given (d-2) hinge "
           "tuples that are genuine (registered, with a top coface; orphans → 0). "
           "Same per-term measure as dualReggeAction. Evaluated over a fixed hinge "
           "set across a move, ΔS = after − before is exact.")
      .def("affected_edges_of_cells", &ReggeSolver::affectedEdgesOfCells,
           py::arg("cells"),
           "The edges (sorted (a,b) id pairs) whose ∂S/∂ℓ²_e a move over the given "
           "top cells can change — the affected-edge index for Δ‖∇S_Regge‖². Union "
           "the before/after evaluations for a fixed set across the move.")
      .def("gradient_norm2_over_edges", &ReggeSolver::gradientNorm2OverEdges,
           py::arg("edges"),
           "Localized squared gradient norm Σ_e |∂S/∂ℓ²_e|² (the geometry term of "
           "F = ‖∇S_Regge‖² + Γ·r_U, extremize δS=0). Each ∂S/∂ℓ²_e is the full "
           "per-edge complex gradient (e's star). Over a fixed affected-edge set "
           "across a move, Δ‖∇S‖² = after − before is exact; over all edges it "
           "equals Σ_e |action_gradient_exact()_e|².")
      .def("matter_action", &ReggeSolver::matterAction,
           "Point-particle matter action: S_matter = -M Σ √(-ℓ²) along worldlines.")
      .def("total_action", &ReggeSolver::totalAction,
           "Total action: S = S_grav + S_matter.  Stationary point = Einstein eqs.")
      .def("action_gradient_norm", &ReggeSolver::actionGradientNorm,
           "||∇S||² = Σ_e (∂S/∂ℓ²_e)².  Zero = Regge equations solved.")
      .def("action_gradient_exact", &ReggeSolver::actionGradientExact,
           "Exact analytic gradient of the complex dual (Sorkin) Regge action: "
           "∂S/∂ℓ²_e per edge (getEdgeList order), as a list of complex. "
           "Assembled from the per-hinge dualVolume/deficit analytic gradients "
           "(no finite differences); matches FD of dualReggeAction to machine "
           "precision in one pass.")
      .def("action_hessian_exact", &ReggeSolver::actionHessianExact,
           "Exact analytic Hessian ∂²S/∂ℓ²_e∂ℓ²_f of the dual Regge action: a "
           "dense |E|x|E| complex matrix (getEdgeList order). Four-term product "
           "rule over the per-hinge dualVolume/deficit Hessians + gradients (no "
           "finite differences); matches a central difference of "
           "actionGradientExact to machine precision.")
      .def("action_hessian_exact_sparse",
           [](const ReggeSolver &self) {
               // Sparse exact Hessian as a COO tuple (rows, cols, values, n).
               // Hand-rolled rather than returned via pybind11/eigen.h, whose
               // sparse binding yields an empty CSC under LTO in this build
               // (cf. quantum EmergentGraph.laplacianCOO).
               const auto H = self.actionHessianExactSparse();
               const auto nnz = static_cast<std::size_t>(H.nonZeros());
               std::vector<int> rows, cols;
               std::vector<std::complex<double>> values;
               rows.reserve(nnz);
               cols.reserve(nnz);
               values.reserve(nnz);
               for (int k = 0; k < H.outerSize(); ++k)
                   for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator
                            it(H, k); it; ++it) {
                       rows.push_back(static_cast<int>(it.row()));
                       cols.push_back(static_cast<int>(it.col()));
                       values.push_back(it.value());
                   }
               return py::make_tuple(rows, cols, values,
                                     static_cast<int>(H.rows()));
           },
           "Sparse exact analytic Hessian as a COO tuple (rows, cols, values, "
           "n), getEdgeList order — wrap with "
           "scipy.sparse.coo_matrix((values, (rows, cols)), shape=(n, n)). Same "
           "values as the dense actionHessianExact on the nonzero pattern (edge "
           "pairs sharing a hinge), assembled at O(nnz) memory instead of O(|E|²).");


  using ::tessera::simulations::InitialChargeMode;
  using ::tessera::simulations::InteractionConfig;
  using ::tessera::simulations::InteractionSimulation;

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
      .def_readwrite("n_systems",           &InteractionConfig::nSystems)
      .def_readwrite("a",                  &InteractionConfig::a)
      .def_readwrite("g",                  &InteractionConfig::g)
      .def_readwrite("m",                  &InteractionConfig::m)
      .def_readwrite("dt",                 &InteractionConfig::dt)
      .def_readwrite("beta",               &InteractionConfig::beta)
      .def_readwrite("epsilon_i",           &InteractionConfig::epsilonI)
      .def_readwrite("target_interactions",
                     &InteractionConfig::targetInteractions)
      .def_readwrite("delaunay_edges",      &InteractionConfig::delaunayEdges)
      .def_readwrite("use_charges",         &InteractionConfig::useCharges)
      .def_readwrite("feature_charges",
                     &InteractionConfig::featureCharges)
      .def_readwrite("feature_deactivate_on_annihilate",
                     &InteractionConfig::featureDeactivateOnAnnihilate)
      .def_readwrite("feature_photon_on_annihilate",
                     &InteractionConfig::featurePhotonOnAnnihilate)
      .def_readwrite("feature_qudit_basis",
                     &InteractionConfig::featureQuditBasis)
      .def_readwrite("feature_choi_sigma_ab",
                     &InteractionConfig::featureChoiSigmaAB)
      .def_readwrite("j_charge_charge",
                     &InteractionConfig::j_chargeCharge)
      .def_readwrite("j_spin_spin",
                     &InteractionConfig::j_spinSpin)
      .def_readwrite("mass_shift",
                     &InteractionConfig::massShift)
      .def_readwrite("gamma_cp_violation",
                     &InteractionConfig::gammaCpViolation)
      .def_readwrite("dt_pair",
                     &InteractionConfig::dtPair)
      .def_readwrite("cp_bias",             &InteractionConfig::cpBias)
      .def_readwrite("initial_charge_mode",
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

Mirrors tessera.CDT: the move primitives interact() / un_interact(), the
driving loop sweep() / thermalize() / tune(), and the diagnostics
compute_action() / get_spectral_dimension() / get_acceptance_rates(). The
object of the search is the beta at which the emergent spectral
dimension reaches 4.
)doc")
      .def(py::init<InteractionConfig>(), py::arg("config"))
      .def("interact",   &InteractionSimulation::interact,
           R"doc(Propose + Metropolis-accept one interaction. Returns acceptance.)doc")
      .def("un_interact", &InteractionSimulation::unInteract,
           R"doc(Propose + Metropolis-accept one un-interaction. Returns acceptance.)doc")
      .def("sweep",      &InteractionSimulation::sweep,
           R"doc(One Monte Carlo sweep; returns the number of accepted moves.)doc")
      .def("thermalize", &InteractionSimulation::thermalize,
           R"doc(Tune to the target volume, then sweep to equilibrium.)doc")
      .def("tune",       &InteractionSimulation::tune,
           py::arg("progress") = nullptr,
           R"doc(Grow the complex toward targetInteractions.)doc")
      .def("compute_action", &InteractionSimulation::computeAction,
           R"doc(The geometric Regge action S = sum_h A_h eps_h.)doc")
      .def("get_spectral_dimension",
           &InteractionSimulation::getSpectralDimension,
           py::arg("sigmas"), py::arg("krylov_dim") = 30,
           R"doc(Heat-kernel spectral dimension D_S(sigma) of the MI-weighted complex.)doc")
      .def("get_deficit_angle_distribution",
           &InteractionSimulation::getDeficitAngleDistribution,
           R"doc(Deficit angles over the interior hinges.)doc")
      .def("get_volume_profile", &InteractionSimulation::getVolumeProfile,
           R"doc(Interaction-count profile by time slice.)doc")
      .def("get_acceptance_rates",
           &InteractionSimulation::getAcceptanceRates,
           R"doc(Accepted / attempted ratio per move type.)doc")
      .def("annihilate", &InteractionSimulation::annihilate,
           R"doc(Spontaneous partial annihilation of a (+, -) frontier pair.)doc")
      .def("pair_create", &InteractionSimulation::pairCreate,
           R"doc(Spontaneous (+, -) pair creation with a Bell joint.)doc")
      .def("get_global_charge", &InteractionSimulation::getGlobalCharge,
           R"doc(Total signed charge across the complex.)doc")
      .def("get_charge_profile", &InteractionSimulation::getChargeProfile,
           R"doc(Per-time-slice (n_+, n_0, n_-, sum_q).)doc")
      .def("get_charge_correlation",
           &InteractionSimulation::getChargeCorrelation,
           py::arg("max_dist"),
           R"doc(<q_v . q_w> as a function of graph distance.)doc")
      .def("qudit_charge_of", &InteractionSimulation::quditChargeOf,
           py::arg("vertex"),
           R"doc(A single vertex's continuous charge via Tr[ρ · Q̂].

Q̂ = diag(+1, +1, -1, -1) on the {|+0⟩, |+1⟩, |−0⟩, |−1⟩} basis.
For an integer-charge eigenstate this returns ±1; for the maximally-mixed
I/4 proxy it returns 0; for an arbitrary mixed state, the value sits in
[−1, +1]. Requires ``featureQuditBasis = True``. Returns 0.0 for vertices
the simulation has no qudit state for.)doc")
      .def("qudit_state_of",
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
      .def("qudit_joint_state_for",
           &InteractionSimulation::quditJointStateFor,
           py::arg("x"), py::arg("y"),
           R"doc(16×16 joint qudit state ρ_XY for a pair.

Returns the stored correlated joint when (x, y) share an interaction
history or are initial-layer Delaunay neighbours; otherwise the
uncorrelated product ρ_x ⊗ ρ_y.)doc")
      .def("get_spacetime", &InteractionSimulation::getSpacetime,
           R"doc(The interaction-history simplicial complex (the primal).)doc")
      .def_property_readonly("interaction_count",
           &InteractionSimulation::interactionCount)
      .def_property_readonly("frontier_size",
           &InteractionSimulation::frontierSize)
      .def_property("beta", &InteractionSimulation::getBeta,
           &InteractionSimulation::setBeta)
      .def("set_seed", &InteractionSimulation::setSeed, py::arg("seed"));
}
