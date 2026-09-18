// Causal-order comparison data classes and the
// `CausalOrders::fromSnapshots` factory.
//
// The three orders on the (cut, time) label set:
//
//   1. ≼_maj - majorization order: (A, s) ≼_maj (B, t) iff the Schmidt
//      spectrum λ_A(s) is majorized by λ_B(t). Built by feeding every
//      snapshot spectrum (across cuts and times) into
//      Majorization::posetOf.
//
//   2. ≼_LR - Lieb-Robinson cone: (A, s) ≼_LR (B, t) iff s < t and the
//      shortest distance between intervals A and B is ≤ vLr · (t − s).
//
//   3. ≼_cs - causal-set order: on a regular chain this is the time
//      order, (A, s) ≼_cs (B, t) iff s < t. With a non-trivial causal
//      set (see the Causet adapter) ≼_cs is informative within a time
//      slice too.
//
// Each order is stored as a Hasse-cover Poset over the same shared
// label set. `Majorization::agreement` computes Kendall-τ, the
// discordant-pair fraction, and the Hasse-graph edit distance between
// any two of the three.
//
// The end-to-end pipeline (config → snapshots → orders → report) lives
// on SchwingerQuench (TDVPRunner.hpp); this file defines the data types
// and the orders factory.
//
// References:
//   Lieb & Robinson, "The finite group velocity of quantum spin
//     systems" (1972); Nachtergaele & Sims, arXiv:math-ph/0506030.
//   Sorkin, "Causal Sets: Discrete Gravity", arXiv:gr-qc/0309009.

#pragma once

#include "quantum/Majorization.hpp"
#include "quantum/Schmidt.hpp"

#include <string>
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

// TDVPSnapshot is defined in TDVPRunner.hpp; only its name is needed
// to declare CausalOrders::fromSnapshots.
struct TDVPSnapshot;

// One node in the (cut, time) label set.
struct LabelSpacetime {
    int    cutIdx{0};      // index into the snapshot's spectra/intervals
    int    tIdx{0};        // index into TDVPSnapshot list
    int    intervalI{0};   // the contiguous interval [intervalI, intervalJ]
    int    intervalJ{0};
    double time{0.0};       // physical time at this snapshot
};

// All three orders on the same label set, plus the labels themselves.
// Each Poset stores Hasse cover edges (transitive reduction of the
// strict order). A node `k` corresponds to `labels[k]`.
struct CausalOrders {
    std::vector<LabelSpacetime> labels;
    Poset maj;     // strict-majorization
    Poset lr;      // Lieb-Robinson cone
    Poset cs;      // causal set (time-only on a regular chain)

    // Build the cross-time majorization poset, the Lieb-Robinson cone
    // poset, and the (regular-chain) causal-set poset from a list of
    // snapshots. Snapshots must have been recorded with
    // `recordSpectra=true`.
    //
    // `predicate` selects the majorization variant for ≼_maj. nullptr
    // means classical majorization (StandardMajorization{1e-12}).
    [[nodiscard]] static CausalOrders fromSnapshots(
        std::vector<TDVPSnapshot> const& snapshots,
        double vLr,
        MajorizationPredicate const* predicate = nullptr);
};

// Result struct for SchwingerQuench::compareCausalOrders.
struct CausalComparisonReport {
    OrderAgreement majVsLr;
    OrderAgreement majVsCs;
    OrderAgreement lrVsCs;
    int nLabels{0};
    int nSnapshots{0};
    double      vLr{0.0};                  // LR velocity used to build ≼_LR
    std::string majKind{"standard"};       // MajorizationPredicate::name() for ≼_maj
};

} // namespace tessera::quantum
