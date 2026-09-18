// Schmidt-spectrum extraction for contiguous-interval bipartitions of a
// matrix product state (MPS): given an MPS and a contiguous interval
// [i, j], return the Schmidt spectrum of the bipartition A = [i, j]
// against its complement.
//
// ─── What we compute ──────────────────────────────────────────────────────
//
// For an MPS |ψ⟩ = T_1 T_2 … T_N |s_1 … s_N⟩ in canonical form with
// orthogonality center brought into A, contracting sites i..j gives a
// tensor M with three index groups: a left bond α (the bond between sites
// i-1 and i), site indices (s_i, …, s_j), and a right bond β (between
// sites j and j+1). With sites outside A in canonical form, the reduced
// density matrix is
//
//     ρ_A^{ s, s' }  =  Σ_{α, β}  M_{αβ}^{s} (M^*)_{αβ}^{s'}
//
// and so the entries of the Schmidt spectrum — the eigenvalues of ρ_A —
// are the squared singular values of M reshaped as
// (sites = rows) × (bonds = cols). Equivalently they are the squares of
// the Schmidt coefficients σ_α in the decomposition
// |ψ⟩ = Σ_α σ_α |α⟩_A ⊗ |α⟩_{Ā}.
//
// The convention throughout is λ_α = σ_α²: the eigenvalues of ρ_A,
// summing to 1 for a normalized state. Majorization::posetOf expects
// spectra in this form.

#pragma once

#include <itensor/all.h>

#include <vector>

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

// 1-based contiguous interval [i, j] with i ≤ j on a chain of N sites.
struct Interval {
    int i{0};  // first site, 1-based
    int j{0};  // last  site, 1-based, j ≥ i
};

// All-contiguous-cut Schmidt spectra of an MPS, excluding the trivial
// full-chain bipartition [1, N] | ∅. This set is the cut family
// \f$\mathcal{F}\f$.
struct SchmidtSpectra {
    int N{0};                                 // chain length
    std::vector<Interval> intervals;          // labels for each spectrum
    std::vector<std::vector<double>> spectra; // spectra[k] for intervals[k]
};

// Schmidt-spectrum extraction. Stateless and not instantiable; call the
// static methods on the class.
class Schmidt {
public:
    Schmidt() = delete;
    Schmidt(Schmidt const&) = delete;
    Schmidt& operator=(Schmidt const&) = delete;

    // Schmidt spectrum across the bipartition [i, j] | rest of an MPS
    // `psi`. Returns the eigenvalues of ρ_A (= squared Schmidt
    // coefficients), sorted non-increasingly, with no zero-padding.
    //
    // Special cases:
    //   • i == j: single-site cuts.
    //   • i == 1 && j == N: the whole chain | empty; returns {1.0}.
    //
    // Throws std::invalid_argument if the interval is out of range or
    // i > j.
    //
    // Complexity: an SVD of a tensor whose dimensions are at most
    //   ( min(2^|A|, D_left · D_right) ) × ( D_left · D_right )
    // where D_* are the MPS bond dimensions adjacent to the interval.
    [[nodiscard]] static std::vector<double>
    of(itensor::MPS const& psi, int i, int j);

    // Every contiguous-cut Schmidt spectrum (1 ≤ i ≤ j ≤ N) excluding
    // the trivial full-chain cut. For an N-site MPS this is
    // N(N+1)/2 - 1 spectra.
    [[nodiscard]] static SchmidtSpectra allOf(itensor::MPS const& psi);
};

} // namespace tessera::quantum
