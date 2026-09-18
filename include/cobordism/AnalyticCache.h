// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_ANALYTICCACHE_H
#define TESSERA_COBORDISM_ANALYTICCACHE_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cobordism/Certificate.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::spacetime;

/// # TouchedStar
///
/// The publication record of one accepted move: the touched simplices, changed
/// edges, and created/deleted cells, all named by their vertex identifiers.
/// `AnalyticCache::publish` intersects this record with each cache entry's
/// component vertex set; entries meeting the star are invalidated, disjoint
/// siblings survive. Matching is by vertex set, not vertex order.
class TouchedStar {
  public:
    /// Record a simplex whose geometry or incidence changed.
    void addTouchedSimplex(const std::vector<std::uint64_t> &vertexIds);
    /// Record an edge whose complex length or phase changed.
    void addChangedEdge(std::uint64_t vertexA, std::uint64_t vertexB);
    /// Record a created cell (a combinatorial change).
    void addCreatedCell(const std::vector<std::uint64_t> &vertexIds);
    /// Record a deleted cell (a combinatorial change).
    void addDeletedCell(const std::vector<std::uint64_t> &vertexIds);

    /// The union of all recorded vertex identifiers (the star's support).
    [[nodiscard]] const std::unordered_set<std::uint64_t> &vertices() const noexcept {
      return vertices_;
    }
    /// Whether any cell was created or deleted (vs. a pure metric change).
    [[nodiscard]] bool structuralChange() const noexcept { return structural_; }
    /// Whether nothing was recorded.
    [[nodiscard]] bool empty() const noexcept { return vertices_.empty(); }

  private:
    std::unordered_set<std::uint64_t> vertices_{};
    bool structural_{false};
};

/// # AnalyticCache
///
/// Revision- and star-keyed cache for per-component analytic payloads (Hodge
/// blocks, factorizations, spectral projectors, transports, ...). Entries are
/// keyed by
///
///  - the **component key** — the order-independent
///    `Fingerprint::fingerprintOf` hash of the component's vertex-identifier
///    set, so the key is invariant under input ordering;
///  - a caller-chosen **kind** string (e.g. `"hodge-block"`); and
///  - an integer **parameter** (degree \f$ k \f$, band index, ...).
///
/// **Freshness contract.** Every entry is stamped with the spacetime's
/// `metricRevisionKey()` at store time; the cache also tracks the revision it
/// was last synchronized to. An entry is served when either
///
///  1. the global metric revision still equals the entry's stamp, or
///  2. every change since the stamp was published through `publish` and this
///     entry survived every intersection test.
///
/// An unpublished revision drift makes the cache serve nothing until the next
/// `publish` or `store`, so it can only cause recomputation, never a stale hit.
///
/// Payloads are opaque `shared_ptr<void>` plus the `Certificate` grading them.
///
/// Threading: not synchronized.
class AnalyticCache {
  public:
    /// Bind the cache to the spacetime whose geometry revisions gate it. The
    /// held `shared_ptr` keeps it alive.
    explicit AnalyticCache(std::shared_ptr<Spacetime> st);

    /// The order-independent component key of a vertex-identifier set
    /// (`Fingerprint::fingerprintOf`, an XOR of mixed ids).
    [[nodiscard]] static std::uint64_t componentKey(
        const std::vector<std::uint64_t> &vertexIds);

    /// The bound spacetime's current metric revision
    /// (`Spacetime::metricRevisionKey`): moves on any combinatorial change,
    /// `setLength` or `setPhase`.
    [[nodiscard]] std::uint64_t geometryRevision() const;

    /// The bound spacetime's current combinatorial revision
    /// (`Spacetime::structuralRevision`).
    [[nodiscard]] std::uint64_t structuralRevision() const;

    /// Store `payload` and `certificate` for (component vertex set, kind,
    /// parameter), stamped at the current metric revision, overwriting any entry
    /// under the same key. The vertex-id set is retained for `publish`.
    void store(const std::vector<std::uint64_t> &componentVertexIds,
               const std::string &kind, std::int64_t parameter,
               std::shared_ptr<void> payload, Certificate certificate);

    /// The cached payload, or nullptr when absent, disabled, or stale under
    /// the freshness contract above. Counts hits/misses.
    [[nodiscard]] std::shared_ptr<void> fetch(
        const std::vector<std::uint64_t> &componentVertexIds,
        const std::string &kind, std::int64_t parameter) const;

    /// The certificate stored beside a payload, or nullptr under the same
    /// conditions `fetch` returns nullptr. Does not count a hit or miss.
    [[nodiscard]] const Certificate *fetchCertificate(
        const std::vector<std::uint64_t> &componentVertexIds,
        const std::string &kind, std::int64_t parameter) const;

    /// Publish one accepted move: drop every entry whose component vertex set
    /// intersects `star.vertices()`, then mark the cache synchronized to the
    /// current metric revision. Call after the mutation, with the full record of
    /// what it touched. An empty star asserts that the revision drift touched
    /// nothing any entry depends on.
    void publish(const TouchedStar &star);

    /// Number of live entries.
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    /// Drop every entry and resynchronize to the current revision.
    void clear();

    /// Replay-mode switch: a disabled cache serves nothing (fetch returns
    /// nullptr) but keeps accepting stores.
    void setEnabled(bool enabled) noexcept { enabled_ = enabled; }
    /// Whether the cache is serving entries (see `setEnabled`).
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    /// Served-fetch count since construction/clear-counters.
    [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }
    /// Empty-fetch count (absent, stale, or disabled).
    [[nodiscard]] std::uint64_t misses() const noexcept { return misses_; }
    /// Entries dropped by `publish` intersection tests.
    [[nodiscard]] std::uint64_t invalidations() const noexcept {
      return invalidations_;
    }

  private:
    struct Entry {
      std::vector<std::uint64_t> vertexIds{};
      std::shared_ptr<void> payload{};
      Certificate certificate{};
      std::uint64_t revision{0};
    };
    struct Key {
      std::uint64_t component{0};
      std::uint64_t kind{0};
      std::int64_t parameter{0};
      bool operator==(const Key &other) const noexcept {
        return component == other.component && kind == other.kind &&
               parameter == other.parameter;
      }
    };
    struct KeyHash {
      std::size_t operator()(const Key &key) const noexcept;
    };

    /// Whether an entry may be served under the freshness contract.
    [[nodiscard]] bool fresh(const Entry &entry) const;
    [[nodiscard]] static Key makeKey(
        const std::vector<std::uint64_t> &componentVertexIds,
        const std::string &kind, std::int64_t parameter);

    std::shared_ptr<Spacetime> st_{};
    std::unordered_map<Key, Entry, KeyHash> entries_{};
    std::uint64_t syncedRevision_{0};
    bool enabled_{true};
    mutable std::uint64_t hits_{0};
    mutable std::uint64_t misses_{0};
    std::uint64_t invalidations_{0};
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_ANALYTICCACHE_H
