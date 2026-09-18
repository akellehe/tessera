// Implementation of the QqbarQuench operator. See
// include/quantum/Quench.hpp for the physics and the parity constraint.

#include "quantum/Quench.hpp"

#include <itensor/all.h>

#include <stdexcept>
#include <string>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;

QqbarQuench::QqbarQuench(int i0, int d, bool enforceParity) noexcept
    : i0_(i0), d_(d), enforceParity_(enforceParity) {}

itensor::MPS QqbarQuench::apply(itensor::MPS const& psi_in,
                                 itensor::SpinHalf const& sites) const {
    using namespace itensor;
    const int N = length(psi_in);
    if (i0_ < 1 || i0_ > N) {
        throw std::invalid_argument(
            "QqbarQuench::apply: i0 out of range [1, N]");
    }
    if (d_ < 1 || i0_ + d_ > N) {
        throw std::invalid_argument(
            "QqbarQuench::apply: i0 + d out of range [i0+1, N]");
    }
    if (enforceParity_) {
        if ((i0_ % 2) != 1) {
            throw std::invalid_argument(
                "QqbarQuench::apply: i0 must be in the Up sublattice (odd, "
                "1-based) for the heavy-quark vacuum; pass "
                "enforceParity=false to override");
        }
        if ((d_ % 2) != 1) {
            throw std::invalid_argument(
                "QqbarQuench::apply: d must be odd so i0 + d lands in the "
                "Dn sublattice (even, 1-based); pass enforceParity=false "
                "to override");
        }
    }

    // Apply a one-site operator to an MPS:
    //   1) bring the orthogonality center to the site;
    //   2) contract op * tensor and remove the prime op introduced;
    //   3) overwrite the site tensor.
    auto applyLocal = [&](MPS& psi, int site, std::string const& opName) {
        psi.position(site);
        auto Op = op(sites, opName, site);
        auto newLocal = noPrime(Op * psi(site));
        psi.set(site, newLocal);
    };

    MPS psi = psi_in;
    applyLocal(psi, i0_,        "S-");   // σ⁻_{i0}: Up → Dn
    applyLocal(psi, i0_ + d_,   "S+");   // σ⁺_{i0+d}: Dn → Up
    psi.normalize();
    return psi;
}

} // namespace tessera::quantum
