// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/options.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "mesh/Fingerprint.h"
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
using namespace tessera::mesh;

// Registers all tessera::mesh classes into the `m` submodule
// (i.e. `tessera.mesh`). Called from src/bindings.cpp's
// PYBIND11_MODULE entry point.
void register_mesh(py::module_ m) {
  // ========================================
  // Fingerprint
  // ========================================
  py::class_<Fingerprint>(m, "Fingerprint",
      R"doc(Order-independent hash of a set of object identifiers.

Every object identified by its constituent vertices (a simplex, a boundary
block's region) is named by this hash, so the name does not depend on the
order the vertices were listed in.

An INSTANCE stores at most `kMax` identifiers and drops the rest silently,
which suits a simplex; `fingerprintOf` is the same hash without that limit,
for sets held in the caller's own container.)doc")
      .def_static("fingerprintOf",
           [](const std::set<IdType> &ids) {
             return Fingerprint::fingerprintOf(ids);
           },
           py::arg("ids"),
           "The fingerprint of a set of identifiers of any size — the value "
           "an instance holding them would report, without the instance's "
           "kMax limit. Depends on the set, not on the order.")
      .def_static("mix64", &Fingerprint::mix64, py::arg("x"),
           "The avalanche mixer the fingerprint is built from: a bijection on "
           "64-bit values in which flipping one input bit changes about half "
           "the output bits.")
      .def(py::init<const std::vector<IdType> &>(), py::arg("ids"),
           "An instance holding these identifiers (duplicates filtered, and "
           "everything past kMax dropped silently).")
      .def("fingerprint", &Fingerprint::fingerprint,
           "This instance's hash — the same value fingerprintOf reports for "
           "the identifiers the instance actually kept.");

  // ========================================
  // Edge
  // ========================================
  py::class_<Edge, std::unique_ptr<Edge, py::nodelete>>(m, "Edge",
      R"doc(An edge connecting two vertices in the simplicial complex.

Edges are 1-simplices linking a source and target vertex.  In CDT the
squared length determines the edge disposition: positive = spacelike,
negative = timelike, zero = lightlike.

Edges are identified by an order-independent fingerprint of their
endpoint vertex IDs, so Edge(v1, v2) == Edge(v2, v1).)doc")
      .def(
        py::init<
          const VertexPtr &,
          const VertexPtr &>(),
        py::arg("source"),
        py::arg("target"),
        "Create an edge between two vertices with a random squared length."
      )
      .def(
        py::init<
          const VertexPtr &,
          const VertexPtr &,
          std::complex<double>>(),
        py::arg("source"),
        py::arg("target"),
        py::arg("squaredLength"),
        "Create an edge with a specified (possibly complex) squared length l^2, "
        "stored exactly. The length is derived as its sqrt (real = spacelike, "
        "imaginary = timelike). A real value is a real l^2, not a length."
      )
      .def("simplices", &Edge::simplicesCopy,
        "The simplices registered on this edge -- every simplex that carries it, "
        "of any dimension. Spacetime::registerSimplex mirrors each simplex into "
        "its edges, so this is the edge's own incidence list and its length does "
        "not grow with the four-volume, unlike a vertex's."
      )
      .def("__str__", &Edge::toString)
      .def("__repr__", &Edge::toString)
      .def("__eq__", &Edge::operator==, py::arg("other"))
      .def("__hash__", &Edge::toHash)
      .def("getSource", &Edge::getSource, py::return_value_policy::reference,
           "Return the source vertex of this edge.")
      .def("getLength", &Edge::getLength,
           R"doc(Return the (possibly complex) edge length — the causal DOF.

Real for spacelike, imaginary for timelike, general complex off the
real-Lorentzian locus. This is the edge's ONE degree of freedom: l^2 is derived
by squaring and is never stored (#639), so square getLength() where you need it.
Causal character is the ARGUMENT of l^2 -- see squaredArgument() and the
predicates below -- never the Euclidean modulus abs(l). Distinct from getPhase()
(the C* connection).)doc")
      .def("squaredArgument", &Edge::squaredArgument,
           "arg(l^2) in (-pi, pi] -- the MEASURED quantity every causal predicate "
           "classifies. 0 is spacelike, +/-pi/2 lightlike, +/-pi timelike, anything "
           "else mixed. Carried so a consumer can see where an edge actually sits "
           "rather than only which bucket it fell in.")
      .def("lorentzianMagnitude", &Edge::lorentzianMagnitude,
           "Re(l^2) = x^2 - t^2 for l = x + i t. Carried for consumers that want "
           "the interval itself; it does NOT decide the disposition alone, since "
           "that would discard Im(l^2) -- which is nonzero precisely at the "
           "lightlike point.")
      .def("isTimelike", &Edge::isTimelike,
           "Timelike iff arg(l^2) ~ +/-pi, i.e. l^2 real negative.")
      .def("isSpacelike", &Edge::isSpacelike,
           "Spacelike iff arg(l^2) ~ 0, i.e. l^2 real positive.")
      .def("isNull", &Edge::isNull,
           "Null/lightlike iff arg(l^2) ~ +/-pi/2, i.e. l^2 purely imaginary and "
           "NONZERO -- the light cone, reached non-trivially at Re(l) == Im(l) != 0. "
           "Distinct from isDegenerate(): a null edge is a physical lightlike ray.")
      .def("isMixed", &Edge::isMixed,
           "A genuinely complex l^2 with no definite causal character. NOT snapped "
           "to the nearest of the three -- that would invent definiteness the "
           "geometry does not have. The common case for a uniformly drawn argument.")
      .def("isDegenerate", &Edge::isDegenerate,
           "An ABSENT edge (Euclidean modulus ~ 0), which is not a causal type. "
           "Exactly one of isSpacelike/isTimelike/isNull/isMixed/isDegenerate holds.")
      .def("getPhase", &Edge::getPhase,
           R"doc(Return the complex C* connection phase carried by this edge.

The SECOND edge field, independent of the geometry. The link variable is
U = exp(i * phase) in C*, and the reverse orientation carries its INVERSE
U**-1, not its conjugate (they agree only for a real phase). A gauge
transformation acts by U_xy -> g_x**-1 U_xy g_y and leaves the length, and
every metric weight built from it, untouched.

The phase is complex because the structure group is C* = U(1) x R+:
exp(i*phase) = exp(i*Re(phase)) * exp(-Im(phase)). Re is the compact U(1)
angle in radians -- the only part with winding, hence the only part that
quantizes and the only part a Wilson loop reads. Im is the non-compact local
scale and carries no quantum number.

It twists the hopping of the Aharonov-Bohm operator
(HodgeLaplacian.connectionLaplacian) and never rescales a metric weight: the
geometric Hodge laplacian(k) is built from the lengths alone and is blind to
it at every degree. The default of 0 leaves an untwisted CDT edge unchanged.)doc")
      .def("setLength", &Edge::setLength, py::arg("length"),
           "Set the (complex) edge LENGTH: real=spacelike, imaginary=timelike, "
           "general complex off the real-Lorentzian locus. There is no squared "
           "setter (#639) -- pass sqrt(l2) and choose the branch explicitly.")
      .def("setPhase", &Edge::setPhase, py::arg("phase"),
           "Set the complex C* connection phase carried by this edge: Re is "
           "the compact U(1) angle in radians, Im the non-compact log-scale. "
           "A real value converts and leaves the non-compact part zero.")
      .def_static("vanRaamsdonkLength", &Edge::vanRaamsdonkLength,
                  py::arg("I"), py::arg("iMax"), py::arg("epsilon") = 1e-10,
                  "Van Raamsdonk metric law: the spacelike length -log(I/iMax), "
                  "floored to a finite value when I < epsilon*iMax. Always >= 0.")
      .def("vanRaamsdonkLengthFor", &Edge::vanRaamsdonkLengthFor,
           py::arg("I"), py::arg("iMax"), py::arg("epsilon") = 1e-10,
           "Time-aware Van Raamsdonk length for this edge given the mutual "
           "information I between its endpoints: 0 (null) when the endpoints lie "
           "on different time slices (a forward-time worldline edge), else the "
           "spacelike vanRaamsdonkLength.")
      .def("getTarget", &Edge::getTarget, py::return_value_policy::reference,
           "Return the target vertex of this edge.");
  // ========================================
  // Vertex
  // ========================================
  py::class_<Vertex, std::unique_ptr<Vertex, py::nodelete>>(m, "Vertex",
      R"doc(A point in the simplicial spacetime, identified by a unique integer ID.

Each vertex carries a time coordinate (the first element of its
coordinate vector) and maintains references to its incident edges
and containing simplices.)doc")
      .def("__eq__", &Vertex::operator==, py::arg("other"))
      .def("__repr__", &Vertex::toString)
      .def("__str__", &Vertex::toString)
      .def("addInEdge", &Vertex::addInEdge, py::arg("edge"),
           "Register an incoming edge on this vertex.")
      .def("addOutEdge", &Vertex::addOutEdge, py::arg("edge"),
           "Register an outgoing edge on this vertex.")
      .def("degree", &Vertex::degree,
           "Return the total number of edges incident to this vertex (in + out).")
      .def("getCoordinates", &Vertex::getCoordinates,
           "Return the coordinate vector of this vertex.")
      .def("getEdges", &Vertex::getEdges, py::return_value_policy::copy,
           "Return all edges (both in and out) incident to this vertex.")
      .def("getId", &Vertex::getId,
           "Return the unique integer ID of this vertex.")
      .def("getInEdges", &Vertex::getInEdges, py::return_value_policy::copy,
           "Return edges where this vertex is the target.")
      .def("getOutEdges", &Vertex::getOutEdges, py::return_value_policy::copy,
           "Return edges where this vertex is the source.")
      .def("getSimplices", &Vertex::getSimplices, py::return_value_policy::copy,
           "Return all simplices (of any dimension) containing this vertex.")
      .def("getTime", &Vertex::getTime,
           "Return the time coordinate of this vertex (first coordinate).")
      .def("setTime", &Vertex::setTime, py::arg("time"),
           "Set the time coordinate (first coordinate) of this vertex; "
           "creates a 1-D coordinate when the vertex had none.")
      .def("moveEdgesTo", &Vertex::moveEdgesTo, py::arg("vertex"), py::arg("spacetime"),
           "Transfer all edges from this vertex to another vertex.")
      .def("removeInEdge", &Vertex::removeInEdge, py::arg("edge"),
           "Remove an incoming edge from this vertex and its containing simplices.")
      .def("removeOutEdge", &Vertex::removeOutEdge, py::arg("edge"),
           "Remove an outgoing edge from this vertex and its containing simplices.")
      .def("setCoordinates", &Vertex::setCoordinates, py::arg("coordinates"),
           "Set the coordinate vector of this vertex.")
      .def(py::init<std::uint64_t, std::vector<double> &>(), py::arg("id"), py::arg("coordinates"),
           "Create a vertex with the given ID and coordinates.");
  // ========================================
  // VertexList
  // ========================================
  py::class_<VertexList, std::shared_ptr<VertexList> >(m, "VertexList",
      "Container mapping vertex IDs to Vertex objects.")
      .def(py::init<>())
      .def("__getitem__", &VertexList::operator[], py::arg("vertexId"), py::return_value_policy::reference,
           "Look up a vertex by ID (operator[]).")
      .def("get", &VertexList::get, py::arg("id"), py::return_value_policy::reference,
           "Look up a vertex by its integer ID.  Raises if not found.")
      .def("add",
           py::overload_cast<const std::uint64_t, const std::vector<double> &>(&VertexList::add),
           py::arg("id"), py::arg("coordinates"),
           py::return_value_policy::reference,
           "Create and insert a vertex with the given ID and coordinates.")
      .def("add", py::overload_cast<const std::uint64_t>(&VertexList::add),
           py::arg("id"),
           py::return_value_policy::reference,
           "Create and insert a vertex with the given ID (no coordinates).")
      .def("replace", &VertexList::replace, py::arg("toRemove"), py::arg("toAdd"),
           "Replace a vertex in the list (same ID, new object).")
      .def("size", &VertexList::size,
           "Return the number of vertices in the list.")
      .def("toVector", &VertexList::toVector, py::return_value_policy::reference,
           "Return all vertices as a list.");
  // ========================================
  // EdgeList
  // ========================================
  py::class_<EdgeList, std::shared_ptr<EdgeList> >(m, "EdgeList",
      "Container storing all edges in the spacetime, keyed by fingerprint.")
      .def(py::init<>())
      .def("add", py::overload_cast<const VertexPtr &, const VertexPtr &,
                                    std::complex<double>>(&EdgeList::add),
           py::arg("source"), py::arg("target"), py::arg("length"),
           py::return_value_policy::reference,
           "Add an edge with a specified complex LENGTH (pass sqrt(l2) to give it "
           "by squared value), or return the existing one if duplicate.")
      .def("add", py::overload_cast<const VertexPtr &, const VertexPtr &>(&EdgeList::add),
           py::arg("source"), py::arg("target"),
           py::return_value_policy::reference,
           "Add an edge with auto-computed squared length, or return existing if duplicate.")
      .def("tryAdd", &EdgeList::tryAdd,
           py::arg("source"), py::arg("target"), py::arg("squaredLength"),
           py::return_value_policy::reference,
           R"doc(Insert if absent, otherwise return the existing edge.

Returns ``(edge, inserted)`` where ``inserted`` is ``True`` on a fresh
insert and ``False`` on a dedupe-hit.  Used by transactional Pachner
moves to record which edges they freshly created (so rollback knows
which to remove).)doc")
      .def("remove", py::overload_cast<const EdgePtr &>(&EdgeList::remove), py::arg("edge"),
           "Remove an edge from the list.")
      .def("size", &EdgeList::size,
           "Return the total number of edges.")
      .def("toVector", &EdgeList::toVector, py::return_value_policy::reference,
           "Return all edges as a list.");
  // ========================================
  // TemporalOrientation
  // ========================================
  py::class_<TemporalOrientation, std::shared_ptr<TemporalOrientation> >(m, "TemporalOrientation",
      R"doc(CDT simplex orientation (ti, tf) counting vertices at each time slice.

For a d-simplex spanning times t and t+1:
  - ti = number of vertices at time t
  - tf = number of vertices at time t+1
  - ti + tf = d + 1

Valid CDT orientations: (d,1), (1,d), (d-1,2), (2,d-1).
For d=4: (4,1), (1,4), (3,2), (2,3).)doc")
      .def(py::init<uint8_t, uint8_t>(), py::arg("ti"), py::arg("tf"))
      .def("getOrientation", &TemporalOrientation::getOrientation,
           "Return the (ti, tf) orientation as a pair.")
      .def("__hash__", &TemporalOrientation::hash)
      .def("__eq__", &TemporalOrientation::operator==, py::arg("other"))
      .def("__str__", &TemporalOrientation::toString)
      .def("__repr__", &TemporalOrientation::toString)
      .def("numeric", &TemporalOrientation::numeric,
           "Return the orientation as a Python tuple (ti, tf).");

  py::class_<TemporalOrientationHash, std::shared_ptr<TemporalOrientationHash> >(m, "TemporalOrientationHash")
      .def(py::init<>());
  py::class_<TemporalOrientationEq, std::shared_ptr<TemporalOrientationEq> >(m, "TemporalOrientationEq")
      .def(py::init<>());
  // ========================================
  // Simplex
  // ========================================
  py::class_<Simplex, std::unique_ptr<Simplex, py::nodelete>>(m, "Simplex",
      R"doc(A k-simplex in the simplicial complex.

A k-simplex has k+1 vertices, C(k+1,2) edges, and k+1 facets
(each a (k-1)-simplex).  Top-dimensional simplices in d-dimensional
CDT have d+1 vertices (e.g. 5 vertices for d=4).

Simplices are identified by an order-independent fingerprint of
their vertex IDs.)doc")
      .def("__repr__", &Simplex::toString)
      .def("__str__", &Simplex::toString)
      .def("__hash__", &Simplex::hash)
      .def("__eq__",
           static_cast<bool (Simplex::*)(const Simplex*) const noexcept>(&Simplex::operator==),
           py::arg("other"))
      .def("__eq__",
           static_cast<bool (Simplex::*)(const Simplex &) const noexcept>(&Simplex::operator==),
           py::arg("other"))
      .def("getCofaces", &Simplex::getCofaces,
           py::return_value_policy::reference_internal,
           "Return all simplices of one dimension higher that contain this simplex "
           "as a face. Returned by reference to the canonical Spacetime-owned "
           "simplices (not copies): driving facet/coface materialization from "
           "Python therefore registers the real cofaces, matching the C++ path "
           "(issue #261).")
      .def("getEdges", &Simplex::getEdges, py::return_value_policy::copy,
           "Return the edges (1-faces) of this simplex.")
      .def("assertSpacelikeAdmissible", &Simplex::assertSpacelikeAdmissible,
           py::arg("tol") = 1e-12,
           "Fail-loudly admissibility check for a purely-spacelike simplex: "
           "raises RuntimeError when the Gram matrix is not positive-definite "
           "(the spacelike triangle inequalities are violated). A simplex with "
           "any null/timelike (worldline) edge is skipped; fewer than two "
           "vertices is trivially admissible.")
      .def("getFacets", &Simplex::getFacets,
           py::return_value_policy::reference_internal,
           R"doc(Return the (k-1)-dimensional faces of this k-simplex.

For a top d-simplex with d+1 vertices, returns d+1 facets each with
d vertices.  Also registers coface relationships so that
facet.getCofaces() includes this simplex.

Returned by reference to the canonical Spacetime-owned facets (not copies).
Previously bound return_value_policy::copy, which handed Python detached
copies of the sub-simplices: calling getFacets() on such a copy registered
the copy (with an incomplete coface list) onto the shared vertices, so a
Python-driven materialization corrupted dualVolume(). Reference fixes it
(issue #261).)doc")
      .def("getNumberOfFaces", &Simplex::getNumberOfFaces,
           "Return the number of sub-faces at each dimension.")
      .def("getOrientation", &Simplex::getOrientation,
           "Return the CDT orientation (ti, tf) of this simplex.")
      .def("getVertexIdLookup", &Simplex::getVertexIdLookup, py::return_value_policy::copy,
           "Return the internal vertex-ID-to-index mapping.")
      .def("getVertices", &Simplex::getVertices, py::return_value_policy::reference_internal,
           "Return the vertices of this simplex.")
      .def("hasVertex", &Simplex::hasVertex, py::arg("vertex"),
           "Return True if this simplex contains the given vertex.")
      .def("isCofaceTo", &Simplex::isCofaceTo, py::arg("facet"), py::arg("shallow") = true,
           "Return True if this simplex is a coface of the given facet.")
      .def("isInitialized", &Simplex::isInitialized,
           "Return True if this simplex has been properly initialized with vertices.")
      .def("isSpatial", &Simplex::isSpatial,
           "Return True if all vertices lie on the same time slice (purely spatial simplex).")
      .def("isTimelike", &Simplex::isTimelike,
           "Deprecated: misnamed. Returns True for *spatial* simplices (all same time). Use isSpatial().")
      .def("replaceVertex", &Simplex::replaceVertex, py::arg("oldVertex"), py::arg("newVertex"),
           "Replace a vertex in this simplex (updates fingerprint and internal maps).")
      .def("validate", &Simplex::validate,
           "Run internal consistency checks on this simplex.")
      .def("gramMatrix", &Simplex::gramMatrix,
           "Gram matrix from edge lengths (flat d*d row-major), complex and always "
           "signature-aware (signed l^2). There is no Wick-rotated |l^2| mode.")
      .def("cayleyMengerMatrix", &Simplex::cayleyMengerMatrix,
           "Cayley-Menger bordered matrix (flat (d+2)*(d+2) row-major) whose "
           "cofactors give the dihedral angles. Complex, always signed l^2.")
      .def("area", &Simplex::area,
           "Area of this triangle (hinge) via Heron's formula, COMPLEX: a "
           "negative Heron radicand (every timelike triangle) gives an imaginary "
           "area, not the 0 the old real-typed clamp returned (#641).")
      .def("volume", &Simplex::volume,
           "d-content sqrt(det G)/d! on the honest geometry, COMPLEX. A "
           "Lorentzian cell with det G < 0 has an IMAGINARY content -- that is "
           "what its d-content is, not the negative real a double could hold.")
      .def("volumeGradient", &Simplex::volumeGradient,
           "Exact analytic gradient dV/dl^2_e of volume() w.r.t. each edge's "
           "squared length (edge-keyed map, complex values), via Jacobi's formula "
           "on the Gram determinant: dV = (V/2) tr(G^-1 dG). The per-degree Hodge "
           "weight gradient -- keystone for arbitrary-k.")
      .def("volumeGradientDirectionalDerivative",
           &Simplex::volumeGradientDirectionalDerivative, py::arg("direction"),
           "Exact directional SECOND derivative sum_f v_f d^2V/dl^2_e dl^2_f, "
           "edge-keyed in e, for the direction v given in the same edge-keyed "
           "shape volumeGradient() returns. G is LINEAR in l^2, so the "
           "d^2G term vanishes identically and the Hessian is closed form. "
           "Euler: contracting against l^2 itself returns (d/2 - 1) dV/dl^2_e.")
      .def("circumcenterBarycentric", &Simplex::circumcenterBarycentric,
           "Circumcenter in barycentric coordinates (sum 1), complex, intrinsic "
           "from the signature-aware edge lengths; entry i weights "
           "getVertices()[i].")
      .def("circumradiusSquared", &Simplex::circumradiusSquared,
           "Circumradius squared R^2 (intrinsic, signature-aware), complex.")
      .def("dualVolume", &Simplex::dualVolume,
           "Signed circumcentric dual cell content |*sigma| in the surrounding "
           "complex (DEC recursion over cofaces, n = top dimension), complex. "
           "The orientation sign is geometric and is retained.")
      .def("hasTopCoface", &Simplex::hasTopCoface,
           "True iff this simplex is a genuine face of a current top cell (not "
           "an orphan stranded by a Pachner move). The hinges the Regge action "
           "sums over are exactly the (d-2)-faces for which this is true.")
      .def("dualVolumeGradient", &Simplex::dualVolumeGradient,
           "Exact analytic d(dualVolume)/d(l^2_e) for each surrounding edge, as a "
           "dict {(v0,v1): complex}. Differentiates the DEC circumradius recursion "
           "(R^2 = h^T G^-1 h). (n-2)-hinge case only.")
      .def("dualVolumeHessian", &Simplex::dualVolumeHessian,
           "Exact analytic d^2(dualVolume)/d(l^2_e)d(l^2_f), as a dict "
           "{((v0,v1),(v2,v3)): complex}; symmetric. (n-2)-hinge case only.")
      .def("hodgeStar", &Simplex::hodgeStar,
           "Diagonal Hodge-star ratio |*sigma|/|sigma| (dual over primal "
           "content), complex.")
      .def("dihedralAngle", &Simplex::dihedralAngle,
           py::arg("hinge"),
           "Complex Lorentzian (Sorkin) dihedral angle at the hinge — the full "
           "m in {0,1,2} structure (#581): real for an ordinary wedge, complex "
           "(imaginary part = boost rapidity) for a same-character wedge in "
           "the boost regime, and pi/2 - i*asinh(.) for a wedge CROSSING the "
           "light cone (one facet direction spacelike, one timelike). Unlike "
           "the removed real-typed pair it is not clamped, so boosts survive.")
      .def("deficitAngle", &Simplex::deficitAngle,
           "Complex Lorentzian deficit 2π − Σ dihedralAngle over the "
           "top cells at this hinge; real for an all-spacelike neighbourhood, "
           "complex when timelike cells contribute boosts.")
      .def("deficitAngleGradient",
           &Simplex::deficitAngleGradient,
           "Exact analytic d(deficit)/d(l^2_e) for each surrounding edge, as a "
           "dict {(v0,v1): complex}. Cofactor-derivative of the Cayley-Menger "
           "dihedral with the boost-safe sin(theta) branch; matches finite "
           "differences to machine precision.")
      .def("deficitAngleHessian",
           &Simplex::deficitAngleHessian,
           "Exact analytic d^2(deficit)/d(l^2_e)d(l^2_f), as a dict "
           "{((v0,v1),(v2,v3)): complex}. One derivative beyond the gradient "
           "(cofactor second derivative + d^2theta/dr^2); symmetric.");

  py::class_<SimplexHash, std::shared_ptr<SimplexHash> >(m, "SimplexHash")
      .def(py::init<>());
  py::class_<SimplexEq, std::shared_ptr<SimplexEq> >(m, "SimplexEq")
      .def(py::init<>());
  // ========================================
  // SimplexFilter (predicate over top simplices)
  // ========================================
  //
  // Held as ``std::shared_ptr<SimplexFilter>`` so the same object can
  // be referenced from HolographyConfig.simplexFilter and survive
  // copy-by-value of the config. Python subclassing is not supported
  // in this build — extend via C++ subclass + binding instead.
  py::class_<SimplexFilter, std::shared_ptr<SimplexFilter>>(m, "SimplexFilter",
      R"doc(Predicate over top simplices (abstract base).

Selects which top simplices participate in a downstream observable
(``Spacetime.getSpectralDimensionOnSkeleton``). Use one of the
concrete subclasses below; in this build, custom filters require a
C++ subclass + binding.)doc")
      .def("accept", &SimplexFilter::accept, py::arg("simplex"),
           "Return True iff the simplex should participate in the "
           "downstream observable.")
      .def("name", &SimplexFilter::name,
           "Human-readable name; appears in JSON output for "
           "reproducibility.")
      .def("__repr__", [](SimplexFilter const& self) {
        return "<" + self.name() + ">";
      });
  py::class_<AllSimplexFilter, SimplexFilter,
             std::shared_ptr<AllSimplexFilter>>(m, "AllSimplexFilter",
      R"doc(Accepts every top simplex.

Default for the holographic-dual measurement (issue #31). Registration
via ``Spacetime.createSimplex`` already implies combinatorial
constructibility (the ``k + 1`` vertices form a complete subgraph in
the edge set), so this filter intentionally ignores edge-length
geometry.)doc")
      .def(py::init<>());
  py::class_<PositiveGramDeterminantFilter, SimplexFilter,
             std::shared_ptr<PositiveGramDeterminantFilter>>(m, "PositiveGramDeterminantFilter",
      R"doc(Accepts simplices whose Gram matrix has positive determinant.

Restricts the measurement to metrically valid (non-degenerate,
non-collapsed) Euclidean cells. Stricter alternative to the default
``AllSimplexFilter``.)doc")
      .def(py::init<>());
}
