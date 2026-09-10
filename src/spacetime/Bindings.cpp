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
#include "spacetime/topologies/SimplexBoundarySphere.h"
#include "spacetime/topologies/SolidSimplex.h"
#include "spacetime/topologies/RealProjectivePlane.h"
#include "spacetime/topologies/RealProjectiveSpace.h"
#include "spacetime/topologies/ComplexProjectivePlane.h"
#include "spacetime/topologies/SimplicialProduct.h"
#include "spacetime/topologies/SphereCircleProduct.h"
#include "spacetime/topologies/StellarSubdivision.h"
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
using namespace tessera::spacetime;

// Registers all tessera::spacetime classes into the `m` submodule
// (i.e. `tessera.spacetime`). Called from src/bindings.cpp's
// PYBIND11_MODULE entry point.
void register_spacetime(py::module_ m) {
  // ========================================
  // Topologies
  // ========================================
  py::class_<Topology, std::shared_ptr<Topology> >(m, "Topology",
      "Base class for spatial topologies (Toroid, Sphere, etc.).")
      .def("dimension", &Topology::dimension,
           R"doc(The intrinsic manifold dimension d of this topology (top cells
are d-simplices on d+1 vertices). Use it to pick the matching
Signature(d, ...) when building a fixture so its top cells register as
top-dimensional. The fixed-triangulation fixtures report their dimension;
the dimension-parametric CDT topologies (Toroid, Sphere, Cylinder) raise
RuntimeError — their dimension comes from the signature, not the
topology.)doc");

  py::class_<Sphere, Topology, std::shared_ptr<Sphere> >(m, "Sphere",
      "Spherical spatial topology S^{d-1}.")
      .def(py::init<>())
      .def("build", &Sphere::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a spherical initial triangulation with the given number of simplices.");

  py::class_<Cylinder, Topology, std::shared_ptr<Cylinder> >(m, "Cylinder",
      "Cylindrical spatial topology with open time boundaries.")
      .def(py::init<>())
      .def("build", &Cylinder::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a cylindrical triangulation with the given number of simplices.");

  py::class_<Toroid, Topology, std::shared_ptr<Toroid> >(m, "Toroid",
      R"doc(Toroidal spatial topology (periodic boundary conditions).

Uses the staircase product triangulation: each time slab contains
d*(d+1) simplices covering all CDT orientation types (d,1), (d-1,2),
..., (1,d).  For d=4 this gives 20 simplices per slab with equal
numbers of (4,1), (3,2), (2,3), and (1,4) types, enabling all five
Pachner moves (add, remove, flip, iflip, shift).)doc")
      .def(py::init<>())
      .def("build", &Toroid::build, py::arg("spacetime"), py::arg("numSimplices"),
           "Build a toroidal staircase triangulation with the given number of simplices.");

  // Exact, fixed minimal triangulations (cobordism fixtures). Unlike the CDT
  // topologies above, these build a specific pre-geometric (coordinate-free)
  // complex and ignore `numSimplices`.
  py::class_<SimplexBoundarySphere, Topology,
             std::shared_ptr<SimplexBoundarySphere> >(m, "SimplexBoundarySphere",
      "S^n = boundary of the (n+1)-simplex: the minimal n-sphere triangulation "
      "(n+2 vertices). Exact and pre-geometric; build() ignores numSimplices.")
      .def(py::init<int>(), py::arg("n"))
      .def("n", &SimplexBoundarySphere::n, "Dimension n of the sphere.")
      .def("build", &SimplexBoundarySphere::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build S^n = ∂Δ^{n+1} (numSimplices ignored).");

  py::class_<SolidSimplex, Topology, std::shared_ptr<SolidSimplex> >(
      m, "SolidSimplex",
      "Solid n-simplex Δ^n (closed n-ball; ∂ = S^{n-1}), a single top simplex on "
      "n+1 vertices. Exact and pre-geometric; build() ignores numSimplices.")
      .def(py::init<int>(), py::arg("n"))
      .def("n", &SolidSimplex::n, "Dimension n of the simplex.")
      .def("build", &SolidSimplex::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build the solid n-simplex (numSimplices ignored).");

  py::class_<RealProjectivePlane, Topology,
             std::shared_ptr<RealProjectivePlane> >(m, "RealProjectivePlane",
      "Minimal 6-vertex ℝP² (hemi-icosahedron): f=(6,15,10), χ=1, "
      "non-orientable. Exact and pre-geometric; build() ignores numSimplices.")
      .def(py::init<>())
      .def("build", &RealProjectivePlane::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build the 6-vertex RP^2 (numSimplices ignored).");

  py::class_<ComplexProjectivePlane, Topology,
             std::shared_ptr<ComplexProjectivePlane> >(m, "ComplexProjectivePlane",
      "Minimal 9-vertex CP^2 (Kühnel): f=(9,36,84,90,36), χ=3, "
      "Betti (1,0,1,0,1), orientable with |signature|=1. Exact and "
      "pre-geometric; build() ignores numSimplices.")
      .def(py::init<>())
      .def("build", &ComplexProjectivePlane::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build the 9-vertex CP^2 (numSimplices ignored).");

  py::class_<SimplicialProduct, Topology, std::shared_ptr<SimplicialProduct> >(
      m, "SimplicialProduct",
      "Product K x L of two topologies, triangulated by the staircase "
      "(Eilenberg-Zilber) construction. E.g. SimplicialProduct(S2, S2) builds "
      "S^2 x S^2. Exact and pre-geometric; build() ignores numSimplices.")
      .def(py::init<std::shared_ptr<Topology>, std::shared_ptr<Topology> >(),
           py::arg("left"), py::arg("right"))
      .def("build", &SimplicialProduct::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build the product complex (numSimplices ignored).");

  py::class_<SphereCircleProduct, Topology,
             std::shared_ptr<SphereCircleProduct> >(m, "SphereCircleProduct",
      "S^2 x S^1: the closed oriented 3-manifold ∂Δ^3 × ∂Δ^2 (12 vertices, "
      "36 tetrahedra), χ=0, Betti (1,1,1,1). The triple-cup negative control "
      "for T^3. Exact and pre-geometric; build() ignores numSimplices.")
      .def(py::init<>())
      .def("build", &SphereCircleProduct::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build S^2 x S^1 (numSimplices ignored).");

  py::class_<RealProjectiveSpace, Topology,
             std::shared_ptr<RealProjectiveSpace> >(m, "RealProjectiveSpace",
      "Minimal 11-vertex RP^3 = L(2,1) (Walkup): f=(11,51,80,40), chi=0, "
      "Betti (1,0,0,1) over Q and (1,1,1,1) over Z/2 (H_1=Z/2), closed "
      "orientable. The triple-cup positive control for T^3: the DW sign cocycle "
      "distinguishes it (Z_Sign=0 != 1=Z_Trivial). Exact and pre-geometric; "
      "build() ignores numSimplices.")
      .def(py::init<>())
      .def("build", &RealProjectiveSpace::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build the 11-vertex RP^3 (numSimplices ignored).");

  py::class_<StellarSubdivision, Topology,
             std::shared_ptr<StellarSubdivision> >(m, "StellarSubdivision",
      "Refines a base topology by one stellar 1->(n+1) subdivision (star the "
      "lexicographically smallest top simplex at a fresh vertex). Same manifold "
      "and homology as the base, but a genuinely distinct (non-isomorphic) "
      "labelled complex — e.g. StellarSubdivision(T^3) retriangulates T^3. "
      "Exact and pre-geometric; build() ignores numSimplices.")
      .def(py::init<std::shared_ptr<Topology> >(), py::arg("base"))
      .def("build", &StellarSubdivision::build, py::arg("spacetime"),
           py::arg("numSimplices") = 0,
           "Build the subdivided complex (numSimplices ignored).");
  // ========================================
  // Metric
  // ========================================
  py::class_<Metric, std::shared_ptr<Metric> >(m, "Metric",
      R"doc(Spacetime metric defining edge lengths and causal structure.

In coordinate-free mode (the default for CDT), edge squared lengths
are stored directly on the edges rather than computed from coordinates.

Args:
    coordinateFree: If True, squared lengths are stored on edges directly.
    signature: The metric signature (Lorentzian or Euclidean).)doc")
      .def(py::init<bool, Signature &>(),
           py::arg("coordinateFree"),
           py::arg("signature"))
      .def("getSquaredLength", &Metric::getSquaredLength,
           py::arg("sourceCoords"), py::arg("targetCoords"),
           "Compute the squared length of an edge from vertex coordinates.");
  // ========================================
  // Enums
  // ========================================
  py::enum_<SignatureType>(m, "SignatureType",
      "Metric signature: Lorentzian (-,+,+,...) or Euclidean (+,+,+,...).")
      .value("Lorentzian", SignatureType::Lorentzian)
      .value("Euclidean", SignatureType::Euclidean)
      .export_values();

  py::enum_<Foliation>(m, "Foliation",
      "Time foliation type for CDT spacetimes.")
      .value("PREFERRED", Foliation::PREFERRED)
      .value("NONE", Foliation::NONE)
      .export_values();
  // ========================================
  // Signature
  // ========================================
  py::class_<Signature, std::shared_ptr<Signature> >(m, "Signature",
      R"doc(Metric signature specifying dimension and type.

Args:
    dimensions: Number of spacetime dimensions d (e.g. 4 for 4D CDT).
    signatureType: Lorentzian or Euclidean.)doc")
      .def(py::init<int, SignatureType>(), py::arg("dimensions"), py::arg("signatureType"))
      .def("getDiagonal", &Signature::getDiagonal,
           "Return the diagonal entries of the metric signature tensor.");

  py::enum_<SpacetimeType>(m, "SpacetimeType",
      "Type of spacetime simulation.")
      .value("CDT", SpacetimeType::CDT)
      .value("REGGE", SpacetimeType::REGGE)
      .value("COSET", SpacetimeType::COSET)
      .value("HERMITIAN_WEIGHTED", SpacetimeType::HERMITIAN_WEIGHTED)
      .export_values();
  // ========================================
  // Spacetime
  // ========================================
  py::class_<Spacetime, std::shared_ptr<Spacetime> >(m, "Spacetime",
      R"doc(The simplicial spacetime manifold.

Holds the full simplicial complex: vertices, edges, and simplices of all
dimensions, along with the metric and topology.  Provides methods for
building the initial triangulation and manipulating the complex.

Typical construction::

    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0, tessera.PREFERRED,
                         tessera.Toroid())
    st.build(500))doc")
      .def(py::init<
             std::shared_ptr<Metric>,
             const SpacetimeType,
             std::optional<double>,
             std::optional<double>,
             Foliation,
             std::optional<std::shared_ptr<Topology> >
           >(),
           py::arg("metric"),
           py::arg("spacetimeType"),
           py::arg("alpha"),
           py::arg("a"),
           py::arg("foliation"),
           py::arg("topology"),
           R"doc(Construct a spacetime with the given metric, type, and topology.

Args:
    metric: The Metric object defining edge lengths.
    spacetimeType: CDT, REGGE, or COSET.
    alpha: Wick rotation parameter (1.0 for Lorentzian CDT).
    a: Lattice spacing parameter.
    foliation: PREFERRED for CDT (fixed time slicing).
    topology: Spatial topology (Toroid or Sphere).)doc"
      )
      .def(py::init<>(), "Create an empty spacetime with default 4D Lorentzian CDT settings.")
      .def("setSeed", &Spacetime::setSeed, py::arg("seed"),
           R"doc(Seed the spacetime's internal RNG deterministically.

The RNG drives ``getRandomVertex`` / ``getRandomSimplex`` /
``getRandomTopSimplex`` — i.e. the first-step sigma selection in
every Pachner move (ShiftMove, FlipMove, IFlipMove, AddMove,
RemoveMove). Use this in tests that need byte-identical reproducibility
across processes; otherwise the default seed comes from
``std::random_device`` and varies per construction.)doc")
      .def("getVertexList", &Spacetime::getVertexList,
           "Return the VertexList containing all vertices.")
      .def("getSimplicesWithOrientation",
           &Spacetime::getSimplicesWithOrientation,
           py::arg("orientation"),
           py::return_value_policy::reference,
           "Return all top simplices with the given CDT orientation.")
      .def("getEdgeList", &Spacetime::getEdgeList,
           "Return the EdgeList containing all edges.")
      .def("getConnectedComponents", &Spacetime::getConnectedComponents, py::return_value_policy::reference,
           "Return the connected components of the simplicial complex.")
      .def("getDualAdjacency", &Spacetime::getDualAdjacency,
           R"doc(Return the dual graph of the top-dimensional triangulation as COO arrays.

Two top simplices are adjacent when they share a (d-1)-face.  Returns
(rows, cols, N) where rows[k] and cols[k] are 0-based indices into the
internal top-simplex array and N is the number of top simplices.

This is much faster than iterating over simplices/facets/cofaces from
Python because it makes a single C++ call instead of O(N) round trips.)doc")
      .def("getDualGraph", &Spacetime::getDualGraph,
           R"doc(Return the dual graph as a SparseGraph.

Equivalent to ``SparseGraph::fromCOO(*getDualAdjacency())`` but
avoids the intermediate Python conversion.)doc")
      .def("modularityOnSkeleton", &Spacetime::modularityOnSkeleton,
           py::arg("M"),
           R"doc(Newman-Girvan modularity Q on the vertex/edge 1-skeleton.

Implicit labels: ``label(v) = v.id() % M``.  Returns 0 if M < 2,
the graph has no edges, or the spacetime has no vertices.)doc")
      .def("getSpectralDimensionOnSkeleton",
           &Spacetime::getSpectralDimensionOnSkeleton,
           py::arg("sigmas"), py::arg("krylovDim"),
           py::arg("filter"), py::arg("topK") = 4,
           py::arg("skeletonDim") = 1,
           R"doc(D_S(σ) on the weighted 1-skeleton of top simplices that
pass ``filter``.

Walks the simplex list keeping those with ``size() == topK + 1`` for
which ``filter.accept(s)`` is true, unions their edges with weights
``w_uv = I_max · exp(-sqrt(|squaredLength_uv|))``, builds the
unnormalised weighted Laplacian ``L = D - W``, and returns
``SpectralGraph.spectralDimension`` of the heat-kernel return
probability. Sits next to ``modularityOnSkeleton``.

``skeletonDim`` reserves API space for higher-k skeletons; only
``skeletonDim == 1`` is currently supported.)doc")
      .def("getTimeSlices", &Spacetime::getTimeSlices,
           "Return sorted list of integer time values in the triangulation.")
      .def("getVerticesAtTime", &Spacetime::getVerticesAtTime,
           py::arg("t"), py::return_value_policy::reference,
           "Return all vertices at integer time t.")
      .def("getSpatialSubgraph", &Spacetime::getSpatialSubgraph,
           py::arg("t"), py::return_value_policy::reference,
           "Return (vertices, spacelike_edges) at time t.")
      .def("bfsDistances", &Spacetime::bfsDistances,
           py::arg("center"), py::arg("maxDepth") = -1,
           "BFS distances from center through spacelike edges.  Returns {id: dist}.")
      .def("build", &Spacetime::build, py::arg("numSimplices") = 3, py::call_guard<py::gil_scoped_release>(),
           R"doc(Build the initial triangulation with approximately n_simplices top simplices.

Uses the topology's builder (e.g. Toroid staircase triangulation) to
create the initial simplicial complex.  The actual number of simplices
may differ slightly due to slab quantization.)doc")
      .def("metricRevisionKey", &Spacetime::metricRevisionKey,
           "Monotone metric revision: structural revision + every live edge's "
           "length and phase revision counters. Strictly increases under any "
           "mutation (creation, setLength, setPhase, simplex register/"
           "unregister, edge removal), so it can key caches and assert "
           "invariants (#692).")
      .def_static("fromCells", &Spacetime::fromCells,
           py::arg("dimensions"), py::arg("cells"),
           py::arg("weight") = 1.0,
           py::arg("phase") = std::complex<double>{0.0, 0.0},
           py::arg("vertexTimes") = std::optional<std::vector<double>>{},
           R"doc(Build a pre-geometric complex from an explicit list of top cells.

The cells-to-Spacetime factory the register/fill builders share. Creates a
coordinate-free Lorentzian ``dimensions``-D CDT spacetime, one vertex per
distinct id, one top simplex per cell (edges auto-wired), and sets the edge
geometry by one of two explicit rules:

  - Uniform Hermitian pin (vertexTimes=None): every edge is pinned to squared
    length ``weight`` and phase ``phase``.
  - Tracked metric (vertexTimes given): each vertex ``v`` carries the single
    time coordinate ``vertexTimes[v]``, so the tracked metric rule assigns
    spacelike (equal-time) and timelike (differing-time) edges automatically.
    ``weight`` and ``phase`` are ignored.

The time coordinate is always arity one — a vertex carries ``[t]`` or no
coordinate, never the length-2/3 vector that makes ``Vertex.getTime`` throw.

Args:
    dimensions: Metric/signature dimension d; pass (d+1)-vertex cells so they
        register as top simplices.
    cells: Top cells as vertex-id tuples (sorted internally).
    weight: Uniform-pin squared length (ignored when vertexTimes is given).
    phase: Uniform-pin complex C* connection phase -- Re the compact U(1)
        angle, Im the non-compact log-scale (ignored when vertexTimes is
        given).
    vertexTimes: Optional per-vertex time indexed by vertex id; its presence
        selects the tracked-metric rule. Must index every vertex id in cells.)doc")
      .def_static("prismCells", &Spacetime::prismCells,
           py::arg("cells"), py::arg("layers") = 1,
           py::arg("twist") =
               std::optional<std::unordered_map<std::uint64_t, std::uint64_t>>{},
           R"doc(The dimension-generic staircase (prism) triangulation of K x [0, layers].

For each base cell (v_0 < ... < v_{m-1}) and each layer, emits the m cells
S_j = {lo[v_0..v_j]} u {hi[v_j..v_{m-1}]}, with lo[x] = phi^l(x) + s*l and
hi[x] = phi^{l+1}(x) + s*(l+1), where s is the per-layer vertex stride (one
past the largest base id). The same rule in every dimension: m=3 gives
tetrahedra over triangles, m=4 gives 4-simplices over tetrahedra, and so on —
the single source replacing the separate 3d and 4d copies.

Args:
    cells: Base top cells as vertex-id tuples.
    layers: Number of product layers (>= 1).
    twist: Optional vertex permutation phi of the base (a dict {id: id}),
        applied cumulatively per layer to glue the top end through a symmetry
        (the mapping-torus twisted product). Identity when None; a missing key
        maps to itself.

Returns:
    The prism's top cells as sorted vertex-id tuples, uniqued and sorted.)doc")
      .def_static("symmetricStackCells", &Spacetime::symmetricStackCells,
           py::arg("baseCells"), py::arg("nApexSlices") = 1,
           R"doc(The symmetric apex stacking of a triangulated d-manifold (#413, #429).

A label-independent alternative to prismCells via coface mirroring (no
vertex-sort diagonal in d=2). Each top d-simplex t cones up to a cell-apex f_t
(up-cone t u {f_t}) and down to the top copy (down-cone = the point reflection
of the up-cone through f_t); the gap over a (d-1)-facet g shared by two cofaces
(apexes f1, f2) is [f1,f2] * boundary(g x I) -- the join of the canonical dual
edge with the worldprism boundary. In d=2 this is exactly the #413 octahedron
split on the dual edge; in d>=3 the side worldsheets take a globally consistent
staircase diagonal, giving a valid manifold on a tetrahedral S^3 base.

The apex is a point reflection (a parity+time inversion), so stacking
nApexSlices reflect-and-cap layers gives an alternating (-1)^j per-slice
chirality (bidirectional / Dirac, not a single chiral screw). IDs: primal layer
ell holds v + ell*stride; apexes start at (nApexSlices+1)*stride. nApexSlices=1
reproduces the single #413 reflection bit-for-bit.

Args:
    baseCells: Base top d-simplices as vertex-id tuples (uniform (d+1)-vertex
        cells; d is inferred and must be >= 2). A facet with one incident top
        simplex (a hole boundary) is a tube wall, not gap-filled.
    nApexSlices: Number of stacked apex (reflect-and-cap) layers (>= 1).

Returns:
    The cobordism's (d+1)-simplices as sorted vertex-id tuples, uniqued.)doc")
      // The returned Simplex handles point into this Spacetime's storage, so
      // each one must keep the Spacetime alive — otherwise
      // `Spacetime(...).getSimplices()` on a temporary frees the storage before
      // the handles are used (a use-after-free / segfault). A blanket
      // keep_alive on the result list does not work (a Python list cannot be a
      // keep-alive target), so we cast each element with reference_internal,
      // which ties its lifetime to the Spacetime (self).
      .def("getSimplices",
           [](py::object self) {
             py::list out;
             for (const auto &s : py::cast<Spacetime &>(self).getSimplices())
               out.append(py::cast(s, py::return_value_policy::reference_internal, self));
             return out;
           },
           "Return all simplices (every dimension) registered in the complex.")
      .def("getTopSimplices",
           [](py::object self) {
             py::list out;
             for (const auto &s : py::cast<Spacetime &>(self).getTopSimplices())
               out.append(py::cast(s, py::return_value_policy::reference_internal, self));
             return out;
           },
           R"doc(Return the top-dimensional simplices in dual-node order.

Element i is dual node i in getDualAdjacency(), so per-node data (vertex
sets, times, ...) can be attached to the dual COO without re-deriving the
top-simplex indexing from Python.)doc")
      .def("getExternalSimplices",
           [](py::object self) {
             py::list out;
             for (const auto &s : py::cast<Spacetime &>(self).getExternalSimplices())
               out.append(py::cast(s, py::return_value_policy::reference_internal, self));
             return out;
           },
           "Return simplices on the boundary of the complex.")
      .def("getBoundary", &Spacetime::getBoundary,
           R"doc(Return the boundary surface as codimension-one faces.

The faces (one dimension below the top simplices) that belong to exactly
one top simplex, as sorted vertex-id tuples. Computed by facet-counting
from the top simplices, so it is side-effect-free and robust to lazily
materialized facets. A closed manifold returns an empty list.

Unlike getExternalSimplices (which returns whole boundary top cells and
materializes facets as a side-effect), this returns the boundary faces
themselves and leaves the complex untouched.)doc")
      .def("materializeFacets",
           static_cast<void (Spacetime::*)() noexcept>(
               &Spacetime::materializeFacets),
           R"doc(Force lazy facet materialization to a fixpoint.

Creates and registers every face of every dimension (down to the
vertices) and wires up the coface incidence. This is the side-effect
getExternalSimplices performs internally, exposed for callers that want
the materialization without the boundary scan.)doc")
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::return_value_policy::reference,
           "Create an edge between two vertices as a NULL edge (l^2 = 0).")
      .def("createEdge",
           static_cast<EdgePtr (Spacetime::*)(const VertexPtr &, const VertexPtr &,
                                              std::complex<double>) const>(&
             Spacetime::createEdge),
           py::arg("source"),
           py::arg("target"),
           py::arg("length"),
           py::return_value_policy::reference,
           "Create an edge between two vertices with a specified complex LENGTH (pass sqrt(l2) to give it by squared value)!")
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t) const noexcept>(
             &Spacetime::createVertex),
           py::arg("id"),
           py::return_value_policy::reference,
           "Create a vertex with the given ID (auto-assigned coordinates).")
      .def("createVertex",
           static_cast<VertexPtr (Spacetime::*)(const std::uint64_t, const std::vector<double> &) const noexcept>(
             &Spacetime::createVertex),
           py::arg("id"), py::arg("coordinates"),
           py::return_value_policy::reference,
           "Create a vertex with the given ID and coordinates (first = time).")
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"),
           py::return_value_policy::reference,
           R"doc(Create a simplex from the given vertices.

Returns (simplex, created) where created is True if the simplex was
new, False if it already existed (dedup by fingerprint).)doc")
      .def("createSimplexTracked",
           [](Spacetime &self, const std::vector<VertexPtr> &vertices) {
               auto r = self.createSimplexTracked(vertices);
               return py::make_tuple(r.simplex, r.created, r.newEdges);
           },
           py::arg("vertices"),
           py::return_value_policy::reference,
           R"doc(Like createSimplex(vertices) but also returns the edges
that this call freshly inserted into the EdgeList.

Returns (simplex, created, newEdges) where:
  - simplex: SimplexPtr (existing or freshly created)
  - created: True if newly created, False if found existing
  - newEdges: list of edges this call freshly inserted (empty when
              created=False, or when all needed edges already existed)

Used by transactional Pachner moves so rollback can undo edge
insertions.)doc")
      .def("createSimplex",
           py::overload_cast<const std::vector<VertexPtr> &, const std::vector<EdgePtr> &>(
             &Spacetime::createSimplex),
           py::arg("vertices"),
           py::arg("edges"),
           py::return_value_policy::reference,
           "Create a simplex from the given vertices and edges.")
      .def("createSimplex",
           py::overload_cast<const std::tuple<uint8_t, uint8_t> &>(&Spacetime::createSimplex),
           py::arg("orientation"),
           py::return_value_policy::reference,
           R"doc(Create a seed simplex with the given orientation.

Automatically creates vertices at times 0 and 1.  Useful for building
minimal test lattices, e.g. createSimplex((1, 4)) for a (1,4) simplex.)doc")
      .def("getSimplexCount", &Spacetime::getSimplexCount,
           "Return N4 = N41 + N32, the total number of top-dimensional simplices.")
      .def("getVertexCount", &Spacetime::getVertexCount,
           "Return N0, the total number of vertices.")
      .def("getN41", &Spacetime::getN41,
           R"doc(Return N41: the count of (d,1) + (1,d) type simplices.

This is the volume-fixing target per [RU] eq. 6.)doc")
      .def("getN32", &Spacetime::getN32,
           "Return N32: the count of (d-1,2) + (2,d-1) type simplices.")
      .def("getRandomSimplex",
           static_cast<SimplexPtr (Spacetime::*)()>(&Spacetime::getRandomSimplex),
           py::return_value_policy::reference,
           py::keep_alive<0, 1>(),  // single handle (weak-referenceable) keeps the Spacetime alive
           "Return a uniformly random simplex from the complex (any dimension).")
      .def("getRandomTopSimplex",
           static_cast<SimplexPtr (Spacetime::*)()>(&Spacetime::getRandomTopSimplex),
           py::return_value_policy::reference,
           py::keep_alive<0, 1>(),
           "Return a uniformly random top-dimensional simplex.")
      .def("getTopVertexCount", &Spacetime::getTopVertexCount,
           R"doc(Return the vertex count of a top-dimensional simplex (d+1).

d is the metric signature's dimension. This is the single source of truth
for what registers as a top cell: a simplex joins topSimplicesVec (and is
seen by getBoundary/getRandomTopSimplex) exactly when its vertex count
equals this. Build a fixture with Signature(topology.dimension(), ...) so
its top cells match.)doc")
      .def("getRandomVertex",
           static_cast<VertexPtr (Spacetime::*)()>(&Spacetime::getRandomVertex),
           py::return_value_policy::reference,
           py::keep_alive<0, 1>(),
           "Return a uniformly random vertex.")
      .def("removeSimplex", &Spacetime::removeSimplex, py::arg("simplex"),
           "Remove a top-dimensional simplex from the complex.")
      .def("pruneOrphanedSimplices",
           static_cast<std::size_t (Spacetime::*)()>(
               &Spacetime::pruneOrphanedSimplices),
           "Unregister sub-simplices (facets/hinges) orphaned by a move — "
           "registered but no longer a face of any current top cell. Returns "
           "the count pruned. Restores the simplex set to the exact closure of "
           "the top cells (bit-identical across an apply/rollback round trip).")
      .def("reclaimSimplexSlots", &Spacetime::reclaimSimplexSlots,
           "Make the storage slots freed since the last call available for "
           "reuse, and return how many were released. Call it where no Pachner "
           "move is in flight: a move captures SimplexPtr in propose() and "
           "reads them in apply(), so a slot freed during the move must not be "
           "handed back out until it finishes. CDT::sweep calls this itself at "
           "the end of every sweep.")
      .def("freeSimplexSlotCount", &Spacetime::freeSimplexSlotCount,
           "Storage slots held for reuse right now.")
      .def("pendingSimplexSlotCount", &Spacetime::pendingSimplexSlotCount,
           "Slots freed since the last reclaimSimplexSlots, not yet reusable.")
      .def("simplexStorageSize", &Spacetime::simplexStorageSize,
           "Slots ever allocated, live or free. Without slot reuse this equals "
           "the number of simplices the spacetime has ever created; with it, it "
           "settles at the live count plus one sweep of churn.")
      .def("swapVertexLabels", &Spacetime::swapVertexLabels, py::arg("v1"), py::arg("v2"),
           R"doc(Swap the integer IDs of two vertices ([BGL] Sec. 2.2.1).

Atomically re-keys all dependent data structures: VertexList,
Edge fingerprints, Simplex vertex-ID maps, and all hash tables.
Handles shared simplices, transient fingerprint collisions, and
sub-simplex ownership correctly.

No-op if v1 and v2 are the same vertex.)doc")
      .def("save", [](const Spacetime &st, const std::string &path,
                       int panelSize, int layoutIters,
                       double tilt, int spin, int precession,
                       int nFrames, int delayCentiseconds) {
          renderSpacetime(st, path, panelSize, layoutIters,
                          tilt, spin, precession, nFrames, delayCentiseconds);
      }, py::arg("path"), py::arg("panelSize") = 800,
         py::arg("layoutIters") = 500,
         py::arg("tilt") = 25.0,
         py::arg("spin") = 1,
         py::arg("precession") = 1,
         py::arg("nFrames") = 36,
         py::arg("delayCs") = 15,
           R"doc(Render the spacetime to an image file.

Uses a force-directed layout (time fixed, spatial coordinates
optimized via spring + repulsion) then projects onto 2D.

If the path ends in .gif, produces an animated GIF whose rotation
is controlled by three parameters:

    tilt        – cone half-angle in degrees (default 25).
    spin        – full Y-axis rotations per loop (default 1).
    precession  – precession cycles per loop (default 1).

Per-frame rotation:
    ry = 2π · spin · t
    rx = tilt · cos(2π · precession · t)
    rz = tilt · sin(2π · precession · t)

Integer values for spin and precession guarantee a perfect loop.

For .graphml or .dot/.gv, exports the graph structure with vertex
attributes (id, time, degree) and edge attributes (squared_length,
timelike).  Layout/rotation parameters are ignored for these formats.

Otherwise produces a static PNG with four panels
(no rotation, 40° X, 40° Y, 40° Z).

The layout is computed internally and does not modify vertex state.

Args:
    path: Output file path (.png, .gif, .graphml, .dot, or .gv).
    panelSize: Pixel size of each panel (default 800).
    layoutIters: Maximum force-directed iterations (default 500).
    tilt: Precession cone half-angle in degrees (default 25).
    spin: Y-axis rotations per loop, integer for perfect loop (default 1).
    precession: Precession cycles per loop, integer for perfect loop (default 1).
    nFrames: Number of GIF frames (default 36).
    delayCs: Frame delay in centiseconds (default 7).)doc");
  // ========================================
  // PachnerMove (transactional Pachner moves)
  // ========================================
  py::enum_<PachnerMode>(m, "PachnerMode",
      R"doc(Validity regime a Pachner move runs under.

  - CDT:          the causal-dynamical-triangulations path. Every
                  proposed cell must satisfy the time-sliced CDT
                  orientation constraint and the move dimension comes
                  from the metric signature. This path is byte-identical
                  to the pre-generalization behaviour.
  - PreGeometric: the CDT orientation/time-slice guards are dropped so
                  the bistellar moves run on a coordinate-free
                  (non-time-sliced) simplicial complex. The move
                  dimension is read off the actual top cell and a
                  manifold-preservation check stands in for the
                  orientation guard.)doc")
      .value("CDT", PachnerMode::CDT)
      .value("PreGeometric", PachnerMode::PreGeometric);
      // NB: no export_values() — that would inject a module-level ``CDT``
      // that shadows ``SpacetimeType.CDT`` (``tessera.CDT``).  Access the
      // members as ``tessera.PachnerMode.CDT`` / ``.PreGeometric``.

  py::class_<PachnerMove>(m, "PachnerMove",
      R"doc(Abstract base class for transactional Pachner moves.

Each subclass (ShiftMove, FlipMove, IFlipMove, AddMove, RemoveMove)
exposes a propose / apply / rollback lifecycle:

  - propose():  read-only target selection + validation.  Returns
                False if no eligible target is found.
  - apply():    commits the move; builds an internal undo log.
  - rollback(): replays the undo log; restores the spacetime to its
                pre-apply state.  Idempotent.

After a successful propose(), the move publishes its combinatorial
deltas (dN0, dN41, dN32) and Metropolis log prefactor so callers can
plug them into action-based acceptance criteria.

See ``docs/source/modularity-plan.md`` for the design rationale.)doc")
      .def("propose", &PachnerMove::propose,
           "Pick a target and validate.  No state change.  Returns "
           "True on success.")
      .def("apply", &PachnerMove::apply,
           "Commit the proposed move; build the undo log.  Returns "
           "True on success.")
      .def("rollback", &PachnerMove::rollback,
           "Restore the spacetime to its pre-apply state.  Idempotent.")
      .def("isApplied", &PachnerMove::isApplied,
           "True iff apply() has been called and not rolled back.")
      .def("dN0", &PachnerMove::dN0,
           "Combinatorial change in vertex count.")
      .def("dN41", &PachnerMove::dN41,
           "Combinatorial change in N41-type top-simplex count.")
      .def("dN32", &PachnerMove::dN32,
           "Combinatorial change in N32-type top-simplex count.")
      .def("metropolisLogPrefactor", &PachnerMove::metropolisLogPrefactor,
           "log of the Metropolis combinatorial prefactor.")
      .def("touchedVertexIds", &PachnerMove::touchedVertexIds,
           "IDs of the vertices whose neighborhood the move re-arranges. "
           "Used for informed (community-aware) proposals.")
      .def("moveType", &PachnerMove::moveType,
           "Move-type tag: one of 'add', 'remove', 'flip', 'iflip', "
           "'shift'.")
      .def("mode", &PachnerMove::mode,
           "Validity regime: PachnerMode.CDT or PachnerMode.PreGeometric.")
      .def("boundaryFixed", &PachnerMove::boundaryFixed,
           "True iff the move is restricted to the interior (∂W fixed).");

  py::class_<AddMove, PachnerMove>(m, "AddMove",
      R"doc(Transactional (2,2d) add (vertex insertion) move.

Picks a random N41 simplex, finds its spatial face and the adjacent
simplex of opposite orientation, and inserts a new vertex at the
spatial time slice.  ``dN0 = +1``; ``dN41 = +(2d-2) = +6`` in 4D;
``dN32 = 0``.

Vertex relabeling (per [BGL] Sec. 2.2.1) is enabled by default.
Pass ``relabel=False`` to disable for tests that need stable
fingerprints across moves.)doc")
      .def(py::init<Spacetime *, std::uint64_t, bool, PachnerMode, bool>(),
           py::arg("spacetime"), py::arg("seed"),
           py::arg("relabel") = true,
           py::arg("mode") = PachnerMode::CDT,
           py::arg("boundaryFixed") = false,
           py::keep_alive<1, 2>(),
           "Construct an add move bound to ``spacetime`` with a fresh "
           "RNG seeded from ``seed``.  ``relabel`` controls whether the "
           "new vertex's ID is swap-relabeled with a random existing "
           "vertex on apply().  ``mode=PachnerMode.PreGeometric`` runs "
           "the 1->(d+1) stellar subdivision on a coordinate-free "
           "complex; ``boundaryFixed`` restricts the move to the "
           "interior (a no-op for add, which never touches ∂W).");

  py::class_<FlipMove, PachnerMove>(m, "FlipMove",
      R"doc(Transactional (2,d) flip move.

Removes 2 d-simplices sharing a (d-1)-face and creates d new
d-simplices sharing an edge.  ``dN0 = 0``; ``ΔN4 = d - 2 = +2`` in 4D.
Inverse: :class:`IFlipMove`.)doc")
      .def(py::init<Spacetime *, std::uint64_t, PachnerMode, bool>(),
           py::arg("spacetime"), py::arg("seed"),
           py::arg("mode") = PachnerMode::CDT,
           py::arg("boundaryFixed") = false,
           py::keep_alive<1, 2>(),
           "Construct a (2,d) flip move bound to ``spacetime`` with a "
           "fresh ``std::mt19937`` seeded with ``seed``.  "
           "``mode=PachnerMode.PreGeometric`` runs the 2->(d+1) bistellar "
           "flip on a coordinate-free complex (drops the CDT orientation "
           "guard, adds a manifold check); ``boundaryFixed`` keeps it "
           "interior (∂W fixed).");

  py::class_<RemoveMove, PachnerMove>(m, "RemoveMove",
      R"doc(Transactional (2d, 2) remove (vertex deletion) move.

Picks a random vertex with order 2d, removes the 2d incident
N41-type simplices and the vertex, and creates 2 replacement
simplices.  ``dN0 = -1``; ``dN41 = -(2d-2) = -6`` in 4D;
``dN32 = 0``.  Inverse: :class:`AddMove`.

Rollback recreates the deleted vertex (with original ID and
coordinates), reinserts its incident edges (with original squared
lengths), and recreates the 2d removed simplices.)doc")
      .def(py::init<Spacetime *, std::uint64_t, PachnerMode, bool>(),
           py::arg("spacetime"), py::arg("seed"),
           py::arg("mode") = PachnerMode::CDT,
           py::arg("boundaryFixed") = false,
           py::keep_alive<1, 2>(),
           "Construct a remove move bound to ``spacetime`` with a fresh "
           "RNG seeded from ``seed``.  ``mode=PachnerMode.PreGeometric`` "
           "runs the (d+1)->1 stellar weld (inverse of the pre-geometric "
           "add) on a coordinate-free complex; the move is interior by "
           "construction, so ``boundaryFixed`` adds no restriction.");

  py::class_<IFlipMove, PachnerMove>(m, "IFlipMove",
      R"doc(Transactional inverse (d, 2) flip move.

Removes d d-simplices sharing an edge and creates 2 new d-simplices
sharing a (d-1)-face.  ``dN0 = 0``; ``ΔN4 = -(d - 2) = -2`` in 4D.
Inverse: :class:`FlipMove`.

Includes a manifold-preservation check in propose() — rejects if
either new simplex would already exist in the lattice.)doc")
      .def(py::init<Spacetime *, std::uint64_t, PachnerMode, bool>(),
           py::arg("spacetime"), py::arg("seed"),
           py::arg("mode") = PachnerMode::CDT,
           py::arg("boundaryFixed") = false,
           py::keep_alive<1, 2>(),
           "Construct an inverse flip bound to ``spacetime`` with a "
           "fresh ``std::mt19937`` seeded with ``seed``.  "
           "``mode=PachnerMode.PreGeometric`` runs the (d+1)->2 bistellar "
           "flip on a coordinate-free complex; ``boundaryFixed`` rejects "
           "flips that would collapse an edge on ∂W.");

  py::class_<ShiftMove, PachnerMove>(m, "ShiftMove",
      R"doc(Transactional (3,3) shift move.

Picks a random top simplex and a random (d-2)-face.  If exactly d-1
top simplices share that face, replaces them with d-1 new simplices
sharing the complementary (d-2)-face.  Self-inverse — dN0 = 0 and
dN41 + dN32 = 0.)doc")
      .def(py::init<Spacetime *, std::uint64_t, PachnerMode, bool>(),
           py::arg("spacetime"), py::arg("seed"),
           py::arg("mode") = PachnerMode::CDT,
           py::arg("boundaryFixed") = false,
           py::keep_alive<1, 2>(),
           R"doc(Construct a shift move bound to ``spacetime``, using a
fresh ``std::mt19937`` seeded with ``seed`` for the proposal.

``mode=PachnerMode.PreGeometric`` drops the CDT orientation guard;
``boundaryFixed`` only fires shifts whose whole region is interior.

For sweeps that share a single Markov chain across many moves, drive
moves via ``CDT.proposeShift()`` instead.)doc");

  // #613: each Pachner move's canonical type name, exposed so Python callers that
  // DISPATCH on it reference the same definition C++ does rather than re-spelling
  // the literal. Tests that ASSERT the name deliberately keep their literals --
  // comparing moveType() against this constant could never fail, so it would
  // weaken the characterization rather than strengthen it.
  m.attr("AddMove").attr("MOVE_TYPE") = AddMove::kMoveType;
  m.attr("RemoveMove").attr("MOVE_TYPE") = RemoveMove::kMoveType;
  m.attr("FlipMove").attr("MOVE_TYPE") = FlipMove::kMoveType;
  m.attr("IFlipMove").attr("MOVE_TYPE") = IFlipMove::kMoveType;
  m.attr("ShiftMove").attr("MOVE_TYPE") = ShiftMove::kMoveType;
}
