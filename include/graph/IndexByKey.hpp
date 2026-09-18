// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Build a "key -> index" lookup table from a vector of items: the map sending
// keyFn(items[i]) to i, with the key extraction explicit at the call site.
//
// Keys are assumed unique across items. If two items share a key the first
// wins (emplace semantics) and the duplicate is silently dropped.

#pragma once

#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::graph {
using namespace ::tessera::mesh;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

template <typename IndexType = int, typename Container, typename KeyFn>
[[nodiscard]] auto
indexByKey(Container const& items, KeyFn keyFn) {
    using KeyType = std::decay_t<decltype(keyFn(*items.begin()))>;
    std::unordered_map<KeyType, IndexType> out;
    out.reserve(items.size());
    IndexType i{0};
    for (auto const& item : items) {
        out.emplace(keyFn(item), i);
        ++i;
    }
    return out;
}

} // namespace tessera::graph
