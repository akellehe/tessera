// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/TemporalOrientation.h"
#include "spacetime/topologies/Toroid.h"
#include "spacetime/Spacetime.h"
#include "utils.h"
#include <deque>
#include <cmath>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Build the causal dynamical triangulation (CDT) as a staircase product.
///
/// Time slabs run between adjacent layers of \f$ d+1 \f$ vertices. The spatial
/// slice at each time is the boundary of the \f$ d \f$-simplex, i.e.
/// \f$ S^{d-1} \f$; at \f$ d = 4 \f$ that is \f$ S^3 \f$ with 5 vertices and
/// 5 tetrahedra.
///
/// Each slab is the staircase decomposition of \f$ S^{d-1} \times [t, t+1] \f$.
/// Every spatial (d-1)-simplex contributes d top simplices covering the CDT
/// orientation types (d,1), (d-1,2), ..., (2,d-1), (1,d), giving
/// \f$ d(d+1) \f$ simplices per slab (20 at \f$ d = 4 \f$). The (3,2) and
/// (2,3) types are what make the flip and shift moves possible.
///
/// Reference: Ambjorn, Jurkiewicz & Loll, "Reconstructing the Universe",
/// arXiv:hep-th/0505154.
void Toroid::build(Spacetime *spacetime, int nSimplices) {
  auto d = spacetime->getMetric()->getSignature()->getDimensions();
  int dPlus1 = d + 1;
  int simplicesPerSlab = d * dPlus1;  // staircase: d simplices per face, (d+1) faces
  int numSlabs = std::max(2, nSimplices / simplicesPerSlab);
  int numLayers = numSlabs + 1;

  spacetime->reserve(nSimplices);

  // Create vertices: (d+1) per time layer
  std::vector<std::vector<VertexPtr>> layers(numLayers);
  for (int t = 0; t < numLayers; ++t) {
    for (int i = 0; i < dPlus1; ++i) {
      layers[t].push_back(
        spacetime->createVertex(std::vector<double>{static_cast<double>(t)}));
    }
    if (t > 0) spacetime->incrementTime();
  }

  // One staircase-triangulated slab per time step.  For each spatial face
  // F_i (skip vertex i) the face has d vertices, and the staircase produces
  // d simplices with orientations (d,1) down to (1,d):
  //   k=d-1: {v0,...,v_{d-1}, w_{d-1}}             — (d,1)
  //   k=d-2: {v0,...,v_{d-2}, w_{d-2}, w_{d-1}}    — (d-1,2)
  //   ...
  //   k=0:   {v0, w0, w1, ..., w_{d-1}}            — (1,d)
  // where v_j are the d lower-layer vertices of F_i and w_j are their
  // upper-layer counterparts.
  for (int slab = 0; slab < numSlabs; ++slab) {
    auto &S = layers[slab];       // spatial vertices at time t
    auto &N = layers[slab + 1];   // next-time vertices at time t+1

    // For each spatial face F_i (skip vertex i from the boundary of Δ^d)
    for (int i = 0; i < dPlus1; ++i) {
      // Collect the d face vertices and their upper counterparts
      std::vector<VertexPtr> faceS, faceN;
      faceS.reserve(d);
      faceN.reserve(d);
      for (int j = 0; j < dPlus1; ++j) {
        if (j != i) {
          faceS.push_back(S[j]);
          faceN.push_back(N[j]);
        }
      }

      // Staircase: for k = d-1 down to 0, create one simplex
      // with (k+1) lower vertices and (d-k) upper vertices
      for (int k = d - 1; k >= 0; --k) {
        VertexPtrs verts;
        verts.reserve(dPlus1);
        for (int j = 0; j <= k; ++j) verts.push_back(faceS[j]);
        for (int j = k; j < d; ++j) verts.push_back(faceN[j]);
        spacetime->createSimplex(verts);
      }
    }
  }

  // Force facet computation on every top simplex so the coface relations
  // exist: the add move finds its spatial-face partner through getCofaces().
  // Iterate by index — getFacets() registers new sub-simplices and so grows
  // simplicesVec — so that only the original top simplices are processed.
  auto nBefore = spacetime->getSimplices().size();
  for (std::size_t i = 0; i < nBefore; ++i) {
    auto s = spacetime->getSimplices()[i];
    if (s->size() == static_cast<std::size_t>(dPlus1)) {
      s->getFacets();
    }
  }
}

} // namespace tessera::spacetime
