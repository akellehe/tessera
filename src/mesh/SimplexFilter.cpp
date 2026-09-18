// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// PositiveGramDeterminantFilter is the only filter with a non-trivial body;
// AllSimplexFilter is header-only.

#include "mesh/SimplexFilter.h"

#include <complex>
#include "mesh/Simplex.h"

#include <cmath>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

bool PositiveGramDeterminantFilter::accept(SimplexPtr const& simplex) const {
    if (simplex == nullptr) return false;
    const std::uint64_t k1 = simplex->size();  // k + 1 vertices for a k-simplex
    if (k1 < 2) return false;                  // 0-simplex: nothing to check

    // gramMatrix() returns a flat (d × d) row-major matrix with d = k = size − 1,
    // vertex 0 at the origin, on the honest signed Lorentzian geometry.
    //
    // The criterion is non-degeneracy, det(G) != 0, not positive-definiteness.
    // Wick-rotating to |l^2| so that det(G) > 0 could stand in for validity asks a
    // Euclidean question of a Lorentzian complex: a valid cell with timelike edges
    // has a sign-indefinite det(G) by design, and once det(G) is complex "> 0" has
    // no meaning. Any NaN or inf in the edge lengths propagates through the Gram
    // build, so such a simplex is rejected defensively.
    std::vector<std::complex<double>> gram = simplex->gramMatrix();
    const int d = static_cast<int>(k1) - 1;
    if (static_cast<int>(gram.size()) != d * d) return false;

    for (std::complex<double> const& g : gram) {
        if (!std::isfinite(g.real()) || !std::isfinite(g.imag())) return false;
    }

    const std::complex<double> det = Simplex::determinant(gram, d);
    return std::isfinite(det.real()) && std::isfinite(det.imag()) &&
           det != std::complex<double>{0.0, 0.0};
}

} // namespace tessera::mesh
