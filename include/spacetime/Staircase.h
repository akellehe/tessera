// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SPACETIME_STAIRCASE_H
#define TESSERA_SPACETIME_STAIRCASE_H

#include <cstddef>
#include <type_traits>
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

/// Cell \f$ k \f$ of the staircase triangulation of the prism
/// \f$ \sigma \times I \f$ over a simplex
/// \f$ \sigma = (v_0 < \dots < v_{m-1}) \f$:
/// \f[
///   S_k = \{\mathrm{lo}(v_0), \dots, \mathrm{lo}(v_k)\} \cup
///         \{\mathrm{hi}(v_k), \dots, \mathrm{hi}(v_{m-1})\},
///   \quad k = 0 \dots m-1,
/// \f]
/// listed in that order: the \f$ k+1 \f$ lower copies, then the \f$ m-k \f$
/// upper ones. `lower(i)` and `upper(i)` give the two copies of \f$ v_i \f$,
/// as vertex ids or vertices. The \f$ m \f$ cells fill the prism, and prisms
/// over simplices that share a face split their common wall alike when their
/// vertices are ordered consistently.
///
/// The caller picks the order of \f$ k \f$ and any sorting of the cell:
/// Spacetime::prismCells and the worldprisms of
/// Spacetime::symmetricStackCells take \f$ k \f$ ascending and sort each cell,
/// and Toroid's time slabs take \f$ k \f$ descending.
template <class Lower, class Upper>
[[nodiscard]] auto staircaseCell(std::size_t m, std::size_t k,
                                 Lower const &lower, Upper const &upper) {
  std::vector<std::decay_t<decltype(lower(std::size_t{0}))>> cell;
  cell.reserve(m + 1);
  for (std::size_t i = 0; i <= k; ++i) cell.push_back(lower(i));
  for (std::size_t i = k; i < m; ++i) cell.push_back(upper(i));
  return cell;
}

} // namespace tessera::spacetime

#endif // TESSERA_SPACETIME_STAIRCASE_H
