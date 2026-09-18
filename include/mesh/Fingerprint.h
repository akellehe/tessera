// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by Andrew Kelleher on 11/13/25.
//

#ifndef TESSERA_VERTEXFINGERPRINT_H
#define TESSERA_VERTEXFINGERPRINT_H

#include <tuple>
#include <cassert>
#include <array>
#include <iterator>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#include "Logger.h"

///
/// \file Fingerprint.h
/// \brief Order-independent hashing system for set-based object identification
///
/// Hashes an unordered set of IDs commutatively: the fingerprint of {1, 2, 3} equals
/// the fingerprint of {3, 1, 2}.
///
/// The fingerprint \f$ h \f$ of a set \f$ S = \{id_1, id_2, \ldots, id_n\} \f$ is
///
/// \f[
/// h(S) = \bigoplus_{id \in S} \text{mix64}(id)
/// \f]
///
/// where \f$ \oplus \f$ is XOR and mix64 is a bijective mixing function.
///
/// A simplex is defined by its constituent vertices, not by the order they are given in:
/// the 2-simplex {v1, v2, v3} is the 2-simplex {v3, v1, v2}. XOR gives a commutative hash
/// that runs in O(n) time and O(1) space, has good avalanche behaviour, and supports
/// incremental update without full recomputation.
///
/// Costs:
/// - add or remove an ID: O(n), with n the current ID count (typically under 10)
/// - compute the hash: O(n), lazily, only when the dirty flag is set
/// - equality: O(1) hash comparison, with an O(n²) fallback on collision
///

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

// ========================================
// Type Definitions
// ========================================

///
/// \brief Type for object identifiers (64-bit unsigned integers)
///
/// All vertices, edges, and simplices are assigned unique IDs of this type.
/// Using uint64_t allows for \f$ 2^{64} \approx 1.8 \times 10^{19} \f$ unique objects.
///
using IdType = std::uint64_t;

///
/// \brief Maximum number of IDs that can be stored in a fingerprint
///
/// A k-simplex holds k+1 IDs, so 8 covers up to a 7-simplex (a 4-simplex uses 5 slots),
/// which is ample for the simplex sizes these simulations use. The array is stored
/// inline, so each extra slot costs 8 bytes on every Fingerprint instance; at 500k
/// objects that is the difference between a compact and a cache-hostile layout.
///
inline constexpr std::size_t kMax = 8;

///
/// \brief Fixed-size array for storing IDs
///
/// A fixed-size array avoids dynamic allocation and cache misses. Only the first n_
/// elements are in use.
///
using IdArray = std::array<IdType, kMax>;

///
/// \brief FNV-1a offset basis. Unused: mix64() mixes differently.
///
inline constexpr std::uint64_t kSeed = 0xcbf29ce484222325ull;

// ========================================
// Fingerprint Class
// ========================================

///
/// \brief Order-independent hash for sets of IDs
///
/// Identifies an unordered set of IDs, which is what a simplex is: a set of constituent
/// vertices, independent of the order they were given in.
///
/// Include a `Fingerprint` as a public member in your class:
///
/// ```cpp
/// class Simplex {
///   public:
///     Fingerprint fingerprint;
///     // ... other members
/// };
/// ```
///
/// Then use the provided template functors for hash tables:
///
/// ```cpp
/// using SimplexHash = FingerprintHash<Simplex>;
/// using SimplexEq = FingerprintEq<Simplex>;
/// std::unordered_set<Simplex, SimplexHash, SimplexEq> simplices;
/// ```
///
/// Or specialize std::hash in the std namespace:
///
/// ```cpp
/// namespace std {
/// template<>
/// struct hash<tessera::mesh::Simplex> {
///   size_t operator()(const tessera::mesh::Simplex &s) const noexcept {
///     return std::hash<std::uint64_t>{}(s.fingerprint.fingerprint());
///   }
/// };
/// }
/// ```
///
/// ## Lazy evaluation
///
/// A dirty flag defers hash computation: add and remove only set dirty, reading
/// fingerprint() recomputes if dirty, and mutable members let const methods refresh the
/// cache.
///
/// ## Collision handling
///
/// Collisions are rare (probability of order \f$ n^2 / 2^{64} \f$ for n objects), but
/// operator== still compares the sets: quick reject on differing size or hash, then an
/// O(n²) set comparison when the hashes match.
///
/// ## Thread safety
///
/// Not thread-safe: the mutable dirty flag and the hash cache h_ can race. Concurrent
/// reads are safe with no concurrent writes; concurrent modification needs external
/// synchronization.
///
class Fingerprint {
  public:
    // ========================================
    // Constructors
    // ========================================

    ///
    /// \brief Default constructor creating an empty fingerprint
    ///
    /// Zero IDs and zero hash, and not dirty: the hash is already current.
    ///
    Fingerprint() noexcept : ids_({}), n_(0), h_(0), dirty_(false) {
    }

    ///
    /// \brief Construct from a vector of IDs
    /// \param ids Initial set of IDs (duplicates are automatically filtered)
    ///
    /// O(n²) in ids.size(), from the duplicate check in addId(). For small n (under 10)
    /// that beats std::unordered_set.
    ///
    explicit Fingerprint(const std::vector<IdType> &ids) noexcept : ids_({}), n_(0), h_(0), dirty_(true) {
      setIds(ids);
    }

    // ========================================
    // Hash Mixing Function
    // ========================================

    ///
    /// \brief Avalanche hash mixer based on MurmurHash3 finalizer
    /// \param x Input value to mix
    /// \return Mixed 64-bit hash value
    ///
    /// A variant of the MurmurHash3 64-bit finalizer:
    ///
    /// \f[
    /// \begin{aligned}
    /// x &\gets x + \phi \cdot 2^{64} \\
    /// x &\gets (x \oplus (x \gg 30)) \cdot c_1 \\
    /// x &\gets (x \oplus (x \gg 27)) \cdot c_2 \\
    /// x &\gets x \oplus (x \gg 31)
    /// \end{aligned}
    /// \f]
    ///
    /// with \f$ \phi \approx 1.618 \f$ the golden ratio and \f$ c_1, c_2 \f$ the mixing
    /// constants.
    ///
    /// It is bijective, so every input maps to a distinct output; it avalanches, so
    /// flipping one input bit changes about half the output bits; it is constexpr; and it
    /// costs an add plus three xor-shift-multiply steps, roughly 4-5 pipelined cycles on
    /// x86-64.
    ///
    static inline constexpr std::uint64_t mix64(IdType x) noexcept {
      x += 0x9e3779b97f4a7c15ull;  // Golden ratio * 2^64
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
      return x ^ (x >> 31);
    }

    ///
    /// \brief Fingerprint of a set of IDs held elsewhere, at any size
    /// \param first,last Iterator range over IDs (duplicates must already be filtered)
    /// \return The same hash an instance holding those IDs would report
    ///
    /// The one place this hash is computed. `fingerprint()` calls it over the instance's
    /// own IDs, and callers whose IDs live in their own container call it directly, so an
    /// instance's hash and a caller's hash of the same set agree by construction.
    ///
    /// An instance stores at most `kMax` IDs and `addId` discards the rest silently,
    /// which suits a simplex but not a set that can outgrow it: past `kMax` the
    /// instance's hash describes an arbitrary subset, so two different sets can report
    /// the same fingerprint. This static has no such limit and hashes every ID in the
    /// range, so it is the form to use for sets not bounded by `kMax`.
    ///
    /// \f[
    /// h = \bigoplus_{i} \text{mix64}(\text{id}_i)
    /// \f]
    ///
    /// XOR is commutative and associative, so the result depends on the set of IDs and
    /// not on the order they arrive in. Duplicates cancel in pairs, hence the
    /// requirement that the range is already duplicate-free, as every standard set
    /// container is.
    ///
    /// O(n) over the range, with no allocation.
    ///
    template <typename Iterator>
    [[nodiscard]] static std::uint64_t fingerprintOf(Iterator first,
                                                     Iterator last) noexcept {
      std::uint64_t hash = 0;
      for (; first != last; ++first) hash ^= mix64(static_cast<IdType>(*first));
      return hash;
    }

    ///
    /// \brief Fingerprint of a container of IDs, at any size
    /// \param ids Container of unique IDs (e.g. a std::set<IdType>)
    /// \return The same hash an instance holding those IDs would report
    ///
    /// Convenience form of the iterator-range overload; see it for the algorithm and for
    /// when to prefer this over holding a `Fingerprint`.
    ///
    template <typename Container>
    [[nodiscard]] static std::uint64_t fingerprintOf(
        const Container &ids) noexcept {
      return fingerprintOf(std::begin(ids), std::end(ids));
    }

    // ========================================
    // ID Management
    // ========================================

    ///
    /// \brief Replace all IDs with a new set
    /// \param ids New set of IDs (duplicates filtered)
    ///
    /// Clears the existing IDs, adds each new one via addId(), and marks the fingerprint
    /// dirty for lazy recomputation. O(n²) in ids.size().
    ///
    void setIds(const std::vector<IdType> &ids) noexcept {
      n_ = 0;
      for (auto id : ids) {
        addId(id);
      }
      dirty_ = true;
    }

    ///
    /// \brief Add a single ID to the set
    /// \param id ID to add (ignored if already present)
    ///
    /// Duplicates are found by linear search, O(n); for small n (under 10) that beats
    /// std::unordered_set on cache locality and allocation.
    ///
    /// Once n_ == kMax the ID is silently ignored. Simplices here rarely exceed 5
    /// vertices, so the limit is not reached in practice.
    ///
    /// The duplicate branch is marked `[[unlikely]]`, favouring the common path of
    /// unique IDs.
    ///
    void addId(IdType id) noexcept {
      // Check for duplicates using linear scan (fast for small n)
      for (std::uint8_t i = 0; i < n_; ++i) {
        if (ids_[i] == id) [[unlikely]] return; // Already present
      }

      if (n_ < kMax) [[likely]] {
        ids_[n_++] = id;
        dirty_ = true;
      }
    }

    ///
    /// \brief Remove an ID from the set
    /// \param id ID to remove (no-op if not present)
    ///
    /// Swap-and-pop: the removed element is replaced by the last one and the count is
    /// decremented, so nothing shifts and the cost stays O(n). ID order is not
    /// preserved, which is harmless because the hash is commutative.
    ///
    void removeId(IdType id) noexcept {
      for (std::uint8_t i = 0; i < n_; ++i) {
        if (ids_[i] == id) {
          // Remove by swapping with last element
          ids_[i] = ids_[--n_];
          dirty_ = true;
          return;
        }
      }
    }

    // ========================================
    // Hash Computation
    // ========================================

    ///
    /// \brief Get the cached fingerprint value (lazy evaluation)
    /// \return 64-bit hash value
    ///
    /// Recomputes only if the dirty flag is set, which amortizes repeated
    /// modification:
    ///
    /// ```cpp
    /// fp.addId(1);  // O(1) - just marks dirty
    /// fp.addId(2);  // O(1) - just marks dirty
    /// fp.addId(3);  // O(1) - just marks dirty
    /// auto h = fp.fingerprint();  // O(n) - computes once
    /// ```
    ///
    /// \f[
    /// h = \bigoplus_{i=0}^{n-1} \text{mix64}(\text{ids}_i)
    /// \f]
    ///
    /// XOR is commutative and associative, so order does not matter.
    ///
    /// The cache members are mutable so a const method can refresh them: logical const,
    /// where the observable value is unchanged and only the cache moves.
    ///
    std::uint64_t fingerprint() const noexcept {
      if (dirty_) [[unlikely]] {
        // The static is the single implementation of this hash (see
        // fingerprintOf): an instance's value and a caller's value for the
        // same set of IDs cannot drift apart.
        h_ = fingerprintOf(ids_.begin(), ids_.begin() + n_);
        dirty_ = false;
      }
      return h_;
    }

    ///
    /// \brief Force immediate hash recomputation
    ///
    /// The hash is otherwise computed lazily on access. Force it before a tight loop
    /// that reads the fingerprint repeatedly:
    ///
    /// ```cpp
    /// for (int i = 0; i < 1000000; ++i) {
    ///   auto h = fp.fingerprint();  // Redundant dirty checks
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```cpp
    /// fp.refresh();  // Compute once
    /// for (int i = 0; i < 1000000; ++i) {
    ///   auto h = fp.fingerprint();  // Fast path: already clean
    /// }
    /// ```
    ///
    void refresh() const noexcept {
      if (dirty_) fingerprint(); // Trigger lazy evaluation
    }

    ///
    /// \brief Batch update and refresh: setIds() followed by refresh().
    /// \param ids New set of IDs
    ///
    void refreshFingerprint(const std::vector<IdType> &ids) {
      setIds(ids);
      refresh();
    }

    // ========================================
    // Operators and Utilities
    // ========================================

    ///
    /// \brief Generate debug string representation
    /// \return String showing hash value and ID list
    ///
    /// Formatted as `<Fingerprint: 12345678901234567890 (1, 5, 9)>`. Refreshes first, so
    /// the hash shown is current.
    ///
    std::string toString() const {
      // Force refresh before stringifying
      refresh();

      std::stringstream ss;
      ss << "<Fingerprint: " << h_ << " (";
      for (std::uint8_t i = 0; i < n_; ++i) {
        ss << ids_[i];
        if (i < n_ - 1) ss << ", ";
      }
      ss << ")>";
      return ss.str();
    }

    ///
    /// \brief Equality comparison with collision-resistant set comparison
    /// \param o Other fingerprint to compare
    /// \return true if both fingerprints represent the same set of IDs
    ///
    /// Refreshes both sides if dirty, rejects on differing n_, rejects on differing h_,
    /// then falls back to a full set comparison so a hash collision cannot make two
    /// different sets compare equal.
    ///
    /// The birthday bound for n fingerprints is
    ///
    /// \f[
    /// P(\text{collision}) \approx 1 - e^{-n^2 / (2 \cdot 2^{64})}
    /// \f]
    ///
    /// giving \f$ P \approx 2.7 \times 10^{-11} \f$ at a billion simplices.
    ///
    /// O(1) when the hashes differ, O(n²) in the ID count on a collision.
    ///
    bool operator==(const Fingerprint &o) const noexcept {
      // Force refresh on both sides if needed
      if (dirty_) [[unlikely]] refresh();
      if (o.dirty_) [[unlikely]] o.refresh();

      if (n_ != o.n_) return false;
      if (h_ != o.h_) return false; // Fast path: hash mismatch

      // Compare sets - both are unique but unsorted
      // For small n (typically 2-5), nested loop is fastest
      for (std::uint8_t i = 0; i < n_; ++i) {
        bool found = false;
        for (std::uint8_t j = 0; j < o.n_; ++j) {
          if (ids_[i] == o.ids_[j]) {
            found = true;
            break;
          }
        }
        if (!found) return false;
      }
      return true;
    }

    ///
    /// \brief Inequality comparison
    /// \param o Other fingerprint
    /// \return true if fingerprints differ
    ///
    bool operator!=(const Fingerprint &o) const noexcept {
      return !(*this == o);
    }

  private:
    // ========================================
    // Private Members
    // ========================================

    IdArray ids_{};                ///< Unique IDs (unsorted for speed)
    std::uint8_t n_{0};            ///< Count of unique IDs [0, kMax]
    mutable std::uint64_t h_{0};   ///< Cached XOR hash - recomputed when dirty
    mutable bool dirty_{false};    ///< Needs recomputation?
};

// ========================================
// Hash and Equality Functors
// ========================================

///
/// \brief Hash functor for types with a `fingerprint` member
/// \tparam T Type that has a public `Fingerprint fingerprint` member
///
/// A generic hash functor for std::unordered_set and std::unordered_map, covering value
/// types, shared_ptr types and heterogeneous lookup.
///
/// `is_transparent` enables C++20 heterogeneous lookup:
///
/// ```cpp
/// std::unordered_set<SimplexPtr, FingerprintHash<Simplex>, FingerprintEq<Simplex>> simplices;
/// uint64_t fp = computeFingerprint();
/// auto it = simplices.find(fp);  // No temporary object created!
/// ```
///
/// Without `is_transparent` a full SimplexPtr would have to be built just to search.
///
/// Supported types:
///
/// - `T` (value type)
/// - `std::shared_ptr<T>`
/// - `std::shared_ptr<const T>`
/// - `uint64_t` (raw fingerprint value for heterogeneous lookup)
///
template<typename T>
struct FingerprintHash {
  using is_transparent = void; // enables heterogeneous lookup

  /// Hash a value type
  size_t operator()(const T &s) const noexcept {
    return static_cast<size_t>(s.fingerprint.fingerprint());
  }

  /// Hash a shared_ptr
  size_t operator()(const std::shared_ptr<T> &s) const noexcept {
    return static_cast<size_t>(s->fingerprint.fingerprint());
  }

  /// Hash a shared_ptr to const
  size_t operator()(const std::shared_ptr<const T> &s) const noexcept {
    return static_cast<size_t>(s->fingerprint.fingerprint());
  }

  /// Hash a raw fingerprint value (for heterogeneous lookup)
  size_t operator()(uint64_t fp) const noexcept {
    return static_cast<size_t>(fp);
  }
};

///
/// \brief Equality functor for types with a `fingerprint` member
/// \tparam T Type that has a public `Fingerprint fingerprint` member
///
/// Equality for std::unordered_set and std::unordered_map, comparing objects by
/// fingerprint rather than by address or full state.
///
/// Supported comparisons:
///
/// - `T == T` (delegates to T::operator==)
/// - `T == uint64_t` (compares fingerprint to raw value)
/// - `shared_ptr<T> == shared_ptr<T>` (compares fingerprints)
/// - `shared_ptr<const T> == shared_ptr<const T>` (with nullptr handling)
/// - All combinations for heterogeneous lookup
///
/// Null handling: `nullptr == nullptr` is true and `nullptr == non-null` is false, and
/// with TESSERA_ASSERTIONS defined some null cases abort instead.
///
template<typename T>
struct FingerprintEq {
  using is_transparent = void;  // enables heterogeneous lookup

  /// Compare two value types (delegates to operator==)
  bool operator()(const T &a, const T &b) const noexcept {
    return a == b;
  }

  /// Compare value type to raw fingerprint
  bool operator()(const T &a, uint64_t fp) const noexcept {
    return a.fingerprint.fingerprint() == fp;
  }

  /// Compare raw fingerprint to value type
  bool operator()(uint64_t fp, const T &a) const noexcept {
    return fp == a.fingerprint.fingerprint();
  }

  /// Compare two shared_ptr by fingerprint (not by pointer address)
  bool operator()(const std::shared_ptr<T> &a, const std::shared_ptr<T> &b) const noexcept {
    return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();
  }

  /// Compare shared_ptr to raw fingerprint
  bool operator()(const std::shared_ptr<T> &a, uint64_t fp) const noexcept {
    return a->fingerprint.fingerprint() == fp;
  }

  /// Compare raw fingerprint to shared_ptr (with nullptr check)
  bool operator()(uint64_t fp, const std::shared_ptr<T> &a) const noexcept {
    if (a == nullptr) return false;
    return fp == a->fingerprint.fingerprint();
  }

  /// Compare two const shared_ptr with explicit nullptr handling
  bool operator()(const std::shared_ptr<const T> &a, const std::shared_ptr<const T> &b) const noexcept {
    if (a == nullptr && b == nullptr) {
      return true;
    }
    if (a == nullptr || b == nullptr) {
      return false;
    }
    return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();
  }

  /// Compare const shared_ptr to raw fingerprint
  bool operator()(const std::shared_ptr<const T> &a, uint64_t fp) const noexcept {
    return a->fingerprint.fingerprint() == fp;
  }

  /// Compare raw fingerprint to const shared_ptr (aborts on nullptr with assertions)
  bool operator()(uint64_t fp, const std::shared_ptr<const T> &a) const noexcept {
#ifdef TESSERA_ASSERTIONS
    if (a == nullptr) {
      CLOG(CRITICAL_LEVEL, "Nullptr in FingerprintEq");
      std::abort();
    }
#endif
    return fp == a->fingerprint.fingerprint();
  }
};

///
/// \brief Hash functor for shared_ptr types with fingerprint member (strict nullptr checking)
/// \tparam T Shared pointer type (e.g., std::shared_ptr<Simplex>)
///
/// Unlike FingerprintHash this assumes T is already a pointer type and checks for null
/// more strictly: with TESSERA_ASSERTIONS defined it aborts on a null pointer, and
/// without assertions a null pointer is undefined behaviour.
///
template<typename T>
struct FingerprintPtrHash {
  using is_transparent = void; // enables heterogeneous lookup

  /// Hash a pointer (aborts on nullptr with assertions)
  size_t operator()(const T &s) const noexcept {
#ifdef TESSERA_ASSERTIONS
    if (s == nullptr) {
      CLOG(CRITICAL_LEVEL, "Nullptr in FingerprintPtrHash");
      std::abort();
    }
#endif
    return static_cast<size_t>(s->fingerprint.fingerprint());
  }

  /// Hash a raw fingerprint value
  size_t operator()(uint64_t fp) const noexcept {
    return static_cast<size_t>(fp);
  }
};

///
/// \brief Equality functor for shared_ptr types with fingerprint member (strict nullptr checking)
/// \tparam T Shared pointer type (e.g., std::shared_ptr<Simplex>)
///
/// With TESSERA_ASSERTIONS defined, asserts that both pointers are non-null.
///
template<typename T>
struct FingerprintPtrEq {
  using is_transparent = void; // enables heterogeneous lookup

  /// Compare two pointers by fingerprint (asserts non-null)
  bool operator()(const T &a, const T &b) const noexcept {
#ifdef TESSERA_ASSERTIONS
    assert(a != nullptr);
    assert(b != nullptr);
#endif
    return a->fingerprint.fingerprint() == b->fingerprint.fingerprint();
  }

  /// Compare pointer to raw fingerprint value
  bool operator()(const T &a, uint64_t fp) const noexcept {
#ifdef TESSERA_ASSERTIONS
    assert(a != nullptr);
#endif
    return a->fingerprint.fingerprint() == fp;
  }

  /// Compare raw fingerprint value to pointer
  bool operator()(uint64_t fp, const T &a) const noexcept {
#ifdef TESSERA_ASSERTIONS
    assert(a != nullptr);
#endif
    return fp == a->fingerprint.fingerprint();
  }
};

// ========================================
// Debugging Utilities (Assertion Builds Only)
// ========================================

#ifdef TESSERA_ASSERTIONS
///
/// \brief Debug utility for detecting hash table corruption
/// \tparam Ptr Pointer type stored in container (e.g., SimplexPtr)
/// \tparam PtrHash Hash functor for Ptr
/// \tparam PtrEq Equality functor for Ptr
///
/// With TESSERA_ASSERTIONS enabled, validates a hash container against the usual
/// corruption modes: null entries left by an incomplete removal, duplicate fingerprints
/// from a double insert, and objects sitting in the wrong bucket or missing entirely.
///
/// Call isCorrupted() or wouldDuplicate() around a critical operation:
///
/// ```cpp
/// #ifdef TESSERA_ASSERTIONS
/// using Detector = CorruptionDetector<SimplexPtr, SimplexHash, SimplexEq>;
/// if (Detector::isCorrupted(simplices)) {
///   // Log error, abort, or enter debugger
/// }
/// #endif
/// ```
///
/// The checks are O(n) and active only when TESSERA_ASSERTIONS is defined; without it
/// the whole class is compiled out.
///
template<typename Ptr, typename PtrHash, typename PtrEq>
class CorruptionDetector {
  public:
    ///
    /// \brief Check if an unordered_set contains corrupted entries
    /// \param container Set to validate
    /// \return true if corruption detected (nullptr or duplicate fingerprints)
    ///
    /// Checks that no entry is null and that all fingerprints are unique. O(n) in
    /// container.size().
    ///
    static bool isCorrupted(const std::unordered_set<Ptr, PtrHash, PtrEq> &container) {
      std::unordered_set<IdType> seen{};
      for (const auto &o : container) {
        if (o == nullptr) {
          CLOG(WARN_LEVEL, "Corruption detected (nullptr)!");
          return true;
        }
        if (seen.contains(o->fingerprint.fingerprint())) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        seen.insert(o->fingerprint.fingerprint());
      }
      return false;
    }

    ///
    /// \brief Check if an unordered_map<IdType, Ptr> contains corrupted entries
    /// \param container Map to validate
    /// \return true if corruption detected
    ///
    /// Checks for null object pointers, an invalid fingerprint member, two objects with
    /// the same fingerprint, and duplicate keys. O(n) in container.size().
    ///
    static bool isCorrupted(const std::unordered_map<IdType, Ptr, PtrHash, PtrEq> &container) {
      std::unordered_set<IdType> seen{};
      for (const auto &[id, o] : container) {
        if (o == nullptr) {
          CLOG(WARN_LEVEL, "Corruption detected (nullptr)!");
          return true;
        }
        if (o->fingerprint == nullptr) {
          CLOG(WARN_LEVEL, "Corruption detected (fp nullptr)!");
          return true;
        }
        if (seen.contains(o->fingerprint.fingerprint())) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        if (seen.contains(id)) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        seen.insert(o->fingerprint.fingerprint());
        seen.insert(id);
      }
      return false;
    }

    ///
    /// \brief Check if an ID→ID map contains duplicate keys or values
    /// \param container Map to validate
    /// \return true if corruption detected (duplicate keys or values)
    ///
    /// This overload is for plain ID mappings, without object pointers. O(n) in
    /// container.size().
    ///
    static bool isCorrupted(const std::unordered_map<IdType, IdType> &container) {
      std::unordered_set<IdType> seen{};
      for (const auto &[id, o] : container) {
        if (seen.contains(id)) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        if (seen.contains(o)) {
          CLOG(WARN_LEVEL, "Corruption detected!");
          return true;
        }
        seen.insert(id);
      }
      return false;
    }

    ///
    /// \brief Check if inserting an element would create a duplicate
    /// \param container Set to check
    /// \param newElement Element to potentially insert
    /// \return true if element already exists (by pointer or fingerprint)
    ///
    /// Call before insertion to verify uniqueness:
    ///
    /// ```cpp
    /// if (Detector::wouldDuplicate(simplices, newSimplex)) {
    ///   CLOG(ERROR_LEVEL, "Attempted to insert duplicate simplex!");
    ///   std::abort();
    /// }
    /// simplices.insert(newSimplex);
    /// ```
    ///
    /// Checks pointer equality (the same shared_ptr instance), fingerprint equality
    /// (different pointers, same content), and container membership via contains().
    /// O(n) linear scan plus an O(1) hash lookup.
    ///
    static bool wouldDuplicate(const std::unordered_set<Ptr, PtrHash, PtrEq> &container, const Ptr &newElement) {
      std::unordered_set<IdType> seen{};
      for (const auto &o : container) {
        if (o == newElement) {
          return true;
        }
        if (o->fingerprint.fingerprint() == newElement->fingerprint.fingerprint()) {
          return true;
        }
      }
      for (const auto &o : container) {
        if (o == newElement) {
          return true;
        }
      }
      return container.contains(newElement);
    }

    ///
    /// \brief Check if inserting a key-value pair would create a duplicate
    /// \param container Map to check
    /// \param newKey Key to potentially insert
    /// \param newElement Element to potentially insert
    /// \return true if key or element already exists
    ///
    /// Checks key equality, pointer equality, fingerprint equality and container
    /// membership. O(n) linear scan plus O(1) hash lookups.
    ///
    static bool wouldDuplicate(const std::unordered_map<IdType, Ptr, PtrHash, PtrEq> &container, const IdType &newKey, const Ptr &newElement) {
      std::unordered_set<IdType> seen{};
      for (const auto &[k, o] : container) {
        if (k == newKey || o == newElement) {
          return true;
        }
        if (o->fingerprint.fingerprint() == newElement->fingerprint.fingerprint()) {
          return true;
        }
      }
      return container.contains(newElement) || container.contains(newKey);
    }
};
#endif  // TESSERA_ASSERTIONS

}

#endif //TESSERA_VERTEXFINGERPRINT_H
