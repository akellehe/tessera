#include "quantum/QuantumVertex.hpp"

namespace tessera::quantum {

// QuantumVertex is header-only: the density matrix accessors are inline and
// the Van Raamsdonk law lives on Edge, where the mutual information of a pair
// is available. This translation unit exists so the class has a home for any
// future out-of-line definition.

} // namespace tessera::quantum
