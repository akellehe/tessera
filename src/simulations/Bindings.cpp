// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
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
           py::arg("targetN41"),
           py::arg("quadraticVolumeFix") = true)
      .def("add", &CDT::add,
           R"doc(Attempt one (2,2d) vertex insertion move.

Picks a random N41 simplex, finds a spatial face with a partner, and
proposes inserting a new vertex.  Accepted via Metropolis with prefactor
N41/(N0+1).  The new vertex is relabeled uniformly on acceptance when
setRelabelVertices(True) is set.

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
      .def("proposeAdd", &CDT::proposeAdd,
           R"doc(Construct a transactional AddMove bound to this
simulation's spacetime and RNG, with propose() already called.  Returns
None if there is no eligible target.  The caller drives apply()/rollback().
Does not update acceptance counters.)doc")
      .def("proposeRemove", &CDT::proposeRemove,
           "Like proposeAdd() for the (2d,2) remove move.")
      .def("proposeFlip", &CDT::proposeFlip,
           "Like proposeAdd() for the (2,d) flip move.")
      .def("proposeIflip", &CDT::proposeIflip,
           "Like proposeAdd() for the (d,2) inverse-flip move.")
      .def("proposeShift", &CDT::proposeShift,
           "Like proposeAdd() for the (3,3) shift move.")
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
      }, py::arg("nSweeps") = 1, py::arg("progress") = py::none(),
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
      .def("computeAction", &CDT::computeAction,
           R"doc(Compute the Regge action S from the current counts.

S = -(k0 + 6*delta)*N0 + (k4 + 2*delta)*N41
    + (k4 + delta)*N32 + volume_fix_term)doc")
      .def("getVolumeProfile", &CDT::getVolumeProfile, py::call_guard<py::gil_scoped_release>(),
           R"doc(Return the spatial volume profile as a list of simplex counts per time slice.

Each entry is the number of top simplices whose minimum vertex time
equals that slice.  The sum equals N4.)doc")
      .def("getAcceptanceRates", &CDT::getAcceptanceRates, py::call_guard<py::gil_scoped_release>(),
           R"doc(Return acceptance rates for each move type as a dict.

Keys: 'add', 'remove', 'flip', 'iflip', 'shift', 'ishift'.
Values: fraction of attempts accepted (0.0 to 1.0).)doc")
      .def("getSpacetime", &CDT::getSpacetime,
           "Return the underlying Spacetime object.")
      .def("getK0", &CDT::getK0,
           "Return the bare inverse Newton's constant k0.")
      .def("getK4", &CDT::getK4,
           "Return the current cosmological coupling k4 (may differ from initial after tune).")
      .def("getDelta", &CDT::getDelta,
           "Return the asymmetry parameter delta.")
      .def("setRelabelVertices", &CDT::setRelabelVertices, py::arg("enabled"),
           R"doc(Enable or disable vertex relabeling after add/remove moves.

Disabled by default: in the labeled formalism the acceptance ratio already
carries the uniform-labeling requirement, and the swap walks a list whose
length grows with the four-volume.)doc")
      .def("setSeed",
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
  S_grav = Σ_h A_h ε_h   (Regge gravitational action)
  S_matter = -M Σ √(-ℓ²)  (proper-time action along worldlines)

Minimizes F = ||∇S||² to find stationary points of S (the discrete
Einstein equations).  F ≥ 0, and F = 0 at the solution.)doc")
      .def(py::init<std::shared_ptr<Spacetime>, MatterConfiguration>(),
           py::arg("spacetime"), py::arg("matter"))
      .def("dihedralAngle", &ReggeSolver::dihedralAngle,
           py::arg("sigma"), py::arg("hinge"),
           "Dihedral angle at hinge h within top-simplex sigma.")
      .def("deficitAngle", &ReggeSolver::deficitAngle,
           py::arg("hinge"),
           "Deficit angle at a hinge: 2π minus sum of dihedral angles.")
      .def_static("hingeArea", &ReggeSolver::hingeArea,
           py::arg("hinge"),
           "Area of a triangular hinge (Heron's formula).")
      .def("reggeAction", &ReggeSolver::reggeAction,
           "Gravitational Regge action: S_grav = Σ_h A_h · ε_h.")
      .def("dualReggeAction", &ReggeSolver::dualReggeAction,
           "Dual Lorentzian Regge action S_Regge(W*) = Σ_h |*h| · ε_h: each "
           "(d-2)-hinge's circumcentric dual content (Simplex.dualVolume) times "
           "its complex Lorentzian deficit. Returns a Python complex (real = "
           "angle-defect curvature, imag = boost/light-cone content).")
      .def("hingeFacesOfCells", &ReggeSolver::hingeFacesOfCells, py::arg("cells"),
           "The (d-2) hinge vertex-tuples that are faces of the given top d-cells "
           "— the affected-hinge index for the incremental ΔS_Regge. `cells` is a "
           "list of vertex-id tuples; returns the dedup'd sorted (d-1)-vertex "
           "sub-tuples. Build from a move's touched cells (created ∪ removed ∪ the "
           "perturbed edge's top cofaces) and reuse the same set before/after.")
      .def("dualReggeActionOverHinges", &ReggeSolver::dualReggeActionOverHinges,
           py::arg("hinges"),
           "Localized dual Regge action Σ |*h|·ε_h over only the given (d-2) hinge "
           "tuples that are genuine (registered, with a top coface; orphans → 0). "
           "Same per-term measure as dualReggeAction. Evaluated over a fixed hinge "
           "set across a move, ΔS = after − before is exact.")
      .def("affectedEdgesOfCells", &ReggeSolver::affectedEdgesOfCells,
           py::arg("cells"),
           "The edges (sorted (a,b) id pairs) whose ∂S/∂ℓ²_e a move over the given "
           "top cells can change — the affected-edge index for Δ‖∇S_Regge‖². Union "
           "the before/after evaluations for a fixed set across the move.")
      .def("gradientNorm2OverEdges", &ReggeSolver::gradientNorm2OverEdges,
           py::arg("edges"),
           "Localized squared gradient norm Σ_e |∂S/∂ℓ²_e|² (the geometry term of "
           "F = ‖∇S_Regge‖² + Γ·r_U, extremize δS=0). Each ∂S/∂ℓ²_e is the full "
           "per-edge complex gradient (e's star). Over a fixed affected-edge set "
           "across a move, Δ‖∇S‖² = after − before is exact; over all edges it "
           "equals Σ_e |actionGradientExact()_e|².")
      .def("matterAction", &ReggeSolver::matterAction,
           "Point-particle matter action: S_matter = -M Σ √(-ℓ²) along worldlines.")
      .def("totalAction", &ReggeSolver::totalAction,
           "Total action: S = S_grav + S_matter.  Stationary point = Einstein eqs.")
      .def("actionGradientNorm", &ReggeSolver::actionGradientNorm,
           "||∇S||² = Σ_e (∂S/∂ℓ²_e)².  Zero = Regge equations solved.")
      .def("actionGradientExact", &ReggeSolver::actionGradientExact,
           "Exact analytic gradient of the complex dual (Sorkin) Regge action: "
           "∂S/∂ℓ²_e per edge (getEdgeList order), as a list of complex. "
           "Assembled from the per-hinge dualVolume/deficit analytic gradients "
           "(no finite differences); matches FD of dualReggeAction to machine "
           "precision in one pass.")
      .def("actionHessianExact", &ReggeSolver::actionHessianExact,
           "Exact analytic Hessian ∂²S/∂ℓ²_e∂ℓ²_f of the dual Regge action: a "
           "dense |E|x|E| complex matrix (getEdgeList order). Four-term product "
           "rule over the per-hinge dualVolume/deficit Hessians + gradients (no "
           "finite differences); matches a central difference of "
           "actionGradientExact to machine precision.")
      .def("actionHessianExactSparse",
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

}
