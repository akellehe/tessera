// InteractionSimulation — Metropolis Monte Carlo over interaction histories,
// weighted by the geometric Regge action on the dual lattice.
//
// See docs/source/interaction-history-monte-carlo.md.
//
// ─── The construction ────────────────────────────────────────────────────
//
// The configuration is an interaction history: a simplicial complex grown
// from a Poisson-Delaunay initial layer of quantum systems by pairwise
// interaction events.
//
//   • Initial layer. N systems, each a randomized mixed state (S(ρ) > 0),
//     Delaunay-triangulated in 2D.
//
//   • Interaction event. Two frontier systems X, Y interact through a
//     two-system unitary U; the interaction product is the joint state
//     ρ_AB = U (ρ_X ⊗ ρ_Y) U†. The (2,3) cell {X, Y, X', AB, Y'} is
//     attached: X, Y leave the frontier, the marginals X', Y' and the
//     joint-state node AB join it.
//
//   • Edge lengths. ℓ = -log I. The spatial / co-existing edges are
//     ordinary mutual informations; the temporal edges close by a
//     conservation law — S(X) = I(X:X') + I(X:AB) with I(X:X') the
//     residual. Every quantity is a reduced-density-matrix computation on
//     a state that concretely exists.
//
// ─── The ensemble ────────────────────────────────────────────────────────
//
// The geometric Regge action S = Σ_h A_h ε_h on the mutual-information
// lengthed complex, sampled at inverse temperature β by Metropolis-Hastings
// with the interact / un-interact move pair:
//
//     A(C→C') = min{ 1, (N₊/N₋)·(C_C/C_{C'})·e^{-β ΔS} }.
//
// Volume is controlled by capping the interaction count (the T-cap). The
// target of the search is the β at which the emergent spectral dimension
// reaches 4. The class mirrors tessera::simulations::CDT: move primitives,
// propose* counterparts, sweep / thermalize / tune, and the computeAction /
// getAcceptanceRates / observable getters.

#pragma once

#include "simulations/Simulation.h"
#include "spacetime/Spacetime.h"

#include <Eigen/Dense>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::spacetime {}
namespace tessera::simulations {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::quantum;

// A single system's quantum state — a one-qubit density matrix.
using SystemState = Eigen::Matrix2cd;

// Initial charge assignment for the charged-model initial layer.
enum class InitialChargeMode {
    ALTERNATING,  // even-index vertex +1, odd-index −1 (exact Q = 0)
    RANDOM,       // independent ±1 per vertex (Q ≈ 0 ± √N)
};

// Flat configuration for an interaction-history Monte Carlo run.
struct InteractionConfig {
    // ─── Initial layer ──────────────────────────────────────────────────
    int nSystems{0};   // system count = Poisson-point count

    // ─── Two-site interaction unitary U = exp(-i H_XY dt) ───────────────
    // H_XY is the Schwinger two-site Hamiltonian term.
    double a{1.0};
    double g{1.0};
    double m{0.5};
    double dt{0.25};

    // ─── Regge ensemble ─────────────────────────────────────────────────
    double beta{1.0};        // inverse temperature in e^{-β S}
    double epsilonI{1e-10};  // mutual-information floor for ℓ = -log I

    // ─── Volume control (the T-cap) ─────────────────────────────────────
    std::size_t targetInteractions{0};

    // ─── Initial Poisson-Delaunay layer connectivity ────────────────────
    // Delaunay edges among the N initial systems, 0-based index pairs,
    // supplied by the caller (scipy.spatial.Delaunay).
    std::vector<std::pair<int, int>> delaunayEdges;

    // ─── Charged Monte Carlo ────────────────────────────────────────────
    // Optional features are toggled by booleans named `featureXxx`, all
    // defaulting to false, so every added behaviour is opt-in.
    bool featureCharges{false};
    bool featureDeactivateOnAnnihilate{false};
    bool featurePhotonOnAnnihilate{false};
    // Alias for featureCharges: setting either one enables the charged
    // code paths.
    bool useCharges{false};
    // CP-violation bias for pair-creation. The first vertex of a
    // spontaneously-created (+, −) pair gets charge 0.5 + δ. If cpBias
    // > 0, δ is drawn from [0, +cpBias] (mean cpBias/2 > 0, favouring
    // +); if < 0, δ from [cpBias, 0] (favouring −). Default 0 = exact
    // C/CP symmetry.
    double cpBias{0.0};
    InitialChargeMode initialChargeMode{InitialChargeMode::ALTERNATING};

    // ─── Qudit-basis vertices (4-dim Hilbert space per vertex) ──────────
    // When true, vertices carry a 4-dim ququart state with the basis
    // {|+0⟩, |+1⟩, |−0⟩, |−1⟩} so charge is intrinsic via the operator
    // Q̂ = diag(+1, +1, −1, −1). The 16×16 pair Hamiltonian is built
    // from the parameters below; the unitary is U = exp(−i H_pair · dt).
    // When this flag is on, the scalar-charge flags above are ignored.
    bool featureQuditBasis{false};
    // ─── Σ_AB as a full 256-dim Choi state ──────────────────────────────
    // When true (requires featureQuditBasis), the Σ_AB entangling-core
    // vertex carries the full Choi state J(U) = (U ⊗ I)|Ω⟩⟨Ω|(U† ⊗ I)
    // — a 256×256 pure state on the doubled Hilbert space — instead of
    // the I/4 maximally-mixed proxy. This makes Σ_AB's marginal
    // consistent with its joint correlations and removes a discrete
    // drift in the total charge. When Σ_AB is consumed by a later
    // interaction, its Choi state is reduced to 4 dimensions via
    // charge-basis projection matched to the partner.
    //
    // Default: true. The constructor clears it when featureQuditBasis is
    // off, since it has no meaning without the qudit basis.
    bool featureChoiSigmaAB{true};
    // Pair-Hamiltonian parameters. The J_c → 0 limit reduces to the
    // scalar-charge regime; experiments can scan them.
    double j_chargeCharge{1.0};    // J_c · (Q̂_A · Q̂_B)
    double j_spinSpin{0.25};       // J_s · (XX + YY)_spin
    double massShift{0.5};         // δ_m · (Q̂_A + Q̂_B)
    double gammaCpViolation{0.0};  // γ_CP · (Ĉ_A ⊗ I + I ⊗ Ĉ_B)
    double dtPair{0.25};           // exponentiation time step

    std::uint32_t seed{0};
    bool          quiet{true};
};

// Transactional interaction move — mirrors tessera::spacetime::PachnerMove. Defined
// in the .cpp; forward-declared so propose* can hand one back.
class InteractionMove;

// Metropolis Monte Carlo over interaction histories.
class InteractionSimulation : public tessera::simulations::Simulation {
  public:
    // Build the Poisson-Delaunay initial layer of randomized mixed-state
    // systems and the frontier / N₊ / N₋ bookkeeping. Throws
    // std::invalid_argument on a malformed config.
    explicit InteractionSimulation(InteractionConfig config);

    ~InteractionSimulation() override;

    // ─── Move primitives (propose + Metropolis accept, like CDT::add) ───

    // Interact a uniformly-random eligible frontier spatial edge {X, Y}:
    // form ρ_AB = U(ρ_X⊗ρ_Y)U†, attach the (2,3) cell, length its edges.
    // Returns true if the Metropolis test accepted.
    bool interact();

    // Remove a uniformly-random cell along with the future cone of its
    // products, restoring the parents it consumed to the frontier. Returns
    // true if accepted.
    bool unInteract();

    // ─── Charge moves ───────────────────────────────────────────────────

    // Spontaneous partial-annihilation: pick a uniformly-random (+, −)
    // frontier pair, cancel the matched portion of their charges, leave
    // a residual (or fully annihilate if magnitudes match). Free of
    // Regge action change; the move's Metropolis prefactor balances it
    // against pairCreate. Returns true if accepted. No-op if useCharges
    // is false.
    bool annihilate();

    // Spontaneous (+, −) pair creation on the frontier with a Bell-state
    // joint and charges drawn from the cpBias distribution. Free of
    // Regge change. Returns true if accepted. No-op if useCharges is
    // false.
    bool pairCreate();

    // ─── Transactional proposers (caller drives propose/apply/rollback) ──
    [[nodiscard]] std::unique_ptr<InteractionMove> proposeInteract();
    [[nodiscard]] std::unique_ptr<InteractionMove> proposeUnInteract();

    // ─── Driving loop (Simulation overrides + sweep) ────────────────────

    // One Monte Carlo sweep: propose ~N₊ + N₋ moves, accept/reject each.
    // Returns the number of accepted moves.
    int sweep();

    // Grow the complex toward targetInteractions — the initial-condition
    // phase, analogous to CDT::tune driving N₄.
    void tune(std::function<void(int, int)> progress = nullptr) override;

    // Run sweeps until the action stabilises (relative change < 1%).
    void thermalize() override;

    // ─── Diagnostics ────────────────────────────────────────────────────

    // The geometric Regge action S = Σ_h A_h ε_h of the current complex.
    [[nodiscard]] double computeAction() const;

    // Heat-kernel spectral dimension D_S(σ) of the MI-weighted complex
    // graph, on the supplied σ-grid.
    [[nodiscard]] std::vector<double>
    getSpectralDimension(const std::vector<double> &sigmas,
                         int krylovDim = 30) const;

    // Histogram of deficit angles over interior hinges.
    [[nodiscard]] std::vector<double> getDeficitAngleDistribution() const;

    // Interaction-count profile by generation depth.
    [[nodiscard]] std::vector<int> getVolumeProfile() const;

    // Accepted / attempted ratio per move type.
    [[nodiscard]] std::map<std::string, double> getAcceptanceRates() const;

    // ─── Charge observables ─────────────────────────────────────────────

    // Σ_v q_v over the entire complex. Exactly conserved at cpBias = 0;
    // drifts as the integrated CP-violation when cpBias ≠ 0.
    [[nodiscard]] double getGlobalCharge() const;

    // Per-time-slice counts (n_+, n_0, n_−) plus net Σq_v on that slice.
    // Each entry is a 4-tuple (n_pos, n_zero, n_neg, sum_q) indexed by
    // time slice (slice 0 = initial layer).
    [[nodiscard]] std::vector<std::array<double, 4>>
    getChargeProfile() const;

    // Charge auto-correlation ⟨q_v · q_w⟩ vs graph-distance d(v, w) for
    // 1 ≤ d ≤ maxDist. Returns a vector of length maxDist; entry [d-1]
    // is the mean q·q over vertex pairs at graph-distance exactly d on
    // the MI-weighted complex graph. Sample-averaged when the pair
    // count is large; exact otherwise.
    [[nodiscard]] std::vector<double>
    getChargeCorrelation(int maxDist) const;

    [[nodiscard]] const std::shared_ptr<tessera::spacetime::Spacetime> &
    getSpacetime() const noexcept { return spacetime_; }

    // ─── Qudit-basis observables ────────────────────────────────────────

    // Per-vertex continuous charge ⟨Q̂⟩ = Tr[ρ · Q̂], where
    // Q̂ = diag(+1, +1, -1, -1) on the {|+0⟩, |+1⟩, |−0⟩, |−1⟩} basis.
    // Returns ±1 for integer-charge eigenstates, 0 for the maximally-mixed
    // I/4 proxy. 0.0 for vertices the simulation has not associated with
    // a qudit state. Requires `featureQuditBasis = True`.
    [[nodiscard]] double quditChargeOf(tessera::mesh::VertexPtr v) const;

    // 16×16 joint qudit state ρ_XY for a pair. Stored correlated joint
    // when (x, y) share an interaction history or are initial-layer
    // Delaunay neighbours; uncorrelated product ρ_x ⊗ ρ_y otherwise.
    [[nodiscard]] Eigen::MatrixXcd
    quditJointStateFor(tessera::mesh::VertexPtr x, tessera::mesh::VertexPtr y) const;

    // Read-only access to the per-vertex 4-dim qudit states. Empty for
    // non-qudit configs and for vertices in the Choi-state path that
    // haven't materialized a single-vertex state. Used by the Python
    // `quditStateOf(vertex)` accessor for tests.
    [[nodiscard]] const std::unordered_map<
        tessera::mesh::VertexPtr, Eigen::Matrix4cd> &
    quditStateOfMap() const noexcept { return quditStateOf_; }

    [[nodiscard]] std::size_t interactionCount() const noexcept {
        return interactionCount_;
    }
    [[nodiscard]] std::size_t frontierSize() const noexcept {
        return frontier_.size();
    }
    [[nodiscard]] double getBeta() const noexcept { return config_.beta; }

    void setBeta(double beta) noexcept { config_.beta = beta; }
    void setSeed(std::uint32_t s) noexcept { rng_.seed(s); }

  private:
    // ─── Configuration + owned state ────────────────────────────────────
    InteractionConfig config_;
    std::shared_ptr<tessera::spacetime::Spacetime> spacetime_;
    std::mt19937 rng_;

    // The two-site interaction unitary U = exp(-i H_XY dt), 4×4.
    Eigen::Matrix4cd interactionU_;

    // Per-system quantum state — the one-qubit marginal. Every system ever
    // created keeps its state here; frozen systems retain theirs so
    // un-interact restores cleanly.
    std::unordered_map<tessera::mesh::VertexPtr, SystemState> stateOf_;

    // Joint states of correlated system pairs: the randomized initial
    // layer is one correlated mixed state, so Delaunay-adjacent systems
    // share genuine mutual information, and an interaction's two products
    // X', Y' inherit the joint state ρ_AB. Keyed by the sorted vertex
    // pointer pair. A pair absent here is treated as uncorrelated.
    std::map<std::pair<tessera::mesh::VertexPtr, tessera::mesh::VertexPtr>,
             Eigen::Matrix4cd>
        jointOf_;

    // ─── Frontier / move bookkeeping ────────────────────────────────────
    // The frontier — systems that have not yet interacted (no temporal
    // out-edges). Any pair of frontier vertices is an eligible interact
    // candidate, so N₊ = |frontier|·(|frontier|−1)/2 and a uniform-random
    // candidate is just (frontier_[i], frontier_[j]) for random i ≠ j.
    // Kept as a flat vector for O(1) sampling, paired with an index map
    // for O(1) swap-and-pop removal.
    std::vector<tessera::mesh::VertexPtr> frontier_;
    std::unordered_map<tessera::mesh::VertexPtr, std::size_t> frontierIdx_;

    // ─── Charge bookkeeping ─────────────────────────────────────────────
    // Continuous charge per vertex in [−1, +1]. Only populated when
    // config.useCharges is true. Worldline-product vertices inherit
    // their parent's charge; Σ_AB products are neutral (0).
    std::unordered_map<tessera::mesh::VertexPtr, double> chargeOf_;

    // ─── Qudit-basis state buffers ──────────────────────────────────────
    // Populated only when config.featureQuditBasis is on. Per-vertex 4×4
    // density matrix in the basis {|+0⟩, |+1⟩, |−0⟩, |−1⟩}. Charge is
    // derived as Tr[ρ · Q̂] where Q̂ = diag(+1, +1, −1, −1).
    std::unordered_map<tessera::mesh::VertexPtr, Eigen::Matrix4cd> quditStateOf_;
    // Per-pair 16-dim joint state in the (A ⊗ B) qudit-tensor order.
    std::map<std::pair<tessera::mesh::VertexPtr, tessera::mesh::VertexPtr>,
             Eigen::MatrixXcd> quditJointOf_;
    // The 16×16 ququart-pair unitary. Built at construction time when
    // featureQuditBasis is on.
    Eigen::MatrixXcd quditInteractionU_;
    // The 256×256 Choi state J(U) of the pair unitary, built once at
    // construction when featureChoiSigmaAB is on. Each Σ_AB vertex
    // carries a copy by reference (via membership in choiStateOf_).
    Eigen::MatrixXcd quditChoiU_;
    // Σ_AB vertices that carry the Choi state (rather than the I/4
    // proxy in quditStateOf_). Membership in this set is the
    // vertex-type marker; the state itself is the shared quditChoiU_.
    std::unordered_set<tessera::mesh::VertexPtr> choiSigmaAbSet_;
    // Frontier vertices indexed by charge sign for O(1) annihilation
    // candidate sampling. A vertex with q > 0 is in frontierPos_,
    // q < 0 in frontierNeg_, q = 0 in frontierZero_. The same
    // swap-and-pop machinery as frontier_ applies.
    std::vector<tessera::mesh::VertexPtr> frontierPos_, frontierNeg_;
    std::unordered_map<tessera::mesh::VertexPtr, std::size_t>
        frontierPosIdx_, frontierNegIdx_;
    // Counters for the charge moves.
    std::int64_t annihilateAttempts_{0}, annihilateAccepted_{0};
    std::int64_t pairCreateAttempts_{0}, pairCreateAccepted_{0};

    // Per-hinge action contribution A_h·ε_h. Updated locally per move so
    // ΔS is O(affected hinges).
    std::unordered_map<tessera::mesh::SimplexPtr, double,
                       tessera::mesh::SimplexPtrHash, tessera::mesh::SimplexPtrEq>
        hingeAction_;

    // Dependency tracking for deep un-interactions.
    //   producedByCell_[v]  = the cell whose interaction created v (a
    //                         product vertex X', AB, or Y').
    //   consumedByCell_[v]  = the cell that took v as a parent in its
    //                         interaction (if any). Each vertex is
    //                         consumed by at most one cell.
    // Together these give the dependency tree: un-interacting a cell
    // truncates its future cone via BFS through producedByCell_ →
    // consumedByCell_.
    std::unordered_map<tessera::mesh::VertexPtr, tessera::mesh::SimplexPtr>
        producedByCell_;
    std::unordered_map<tessera::mesh::VertexPtr, tessera::mesh::SimplexPtr>
        consumedByCell_;

    // Per-cell count of its products that have been consumed by a child
    // interaction. A cell is a *leaf* (its deep un-interact returns to
    // exactly the state where this cell didn't exist) iff this count is
    // zero. The Metropolis prefactor for interact uses leafCellCount_,
    // bounded as the lattice grows. Maintained incrementally.
    std::unordered_map<tessera::mesh::SimplexPtr, std::uint8_t,
                       tessera::mesh::SimplexPtrHash, tessera::mesh::SimplexPtrEq>
        consumedProductsOf_;
    std::size_t leafCellCount_{0};

    std::size_t   interactionCount_{0};
    std::uint64_t nextVertexId_{0};   // monotone; never reused after removeVertex

    // ─── Helpers ────────────────────────────────────────────────────────
    // Build the Poisson-Delaunay initial layer: randomized mixed-state
    // systems as t=0 vertices, the Delaunay edges, the initial tables.
    void buildInitialLayer();

    // O(1) frontier mutations via swap-and-pop on the flat vector.
    // When useCharges is enabled, also maintain the per-sign frontier
    // vectors (frontierPos_ / frontierNeg_) for fast annihilate
    // candidate sampling.
    void addToFrontier(tessera::mesh::VertexPtr v);
    void removeFromFrontier(tessera::mesh::VertexPtr v);
    [[nodiscard]] bool onFrontier(tessera::mesh::VertexPtr v) const noexcept {
        return frontierIdx_.find(v) != frontierIdx_.end();
    }

    // Sign-bucketed frontier helpers (no-op when useCharges is false).
    void addToSignBucket(tessera::mesh::VertexPtr v);
    void removeFromSignBucket(tessera::mesh::VertexPtr v);

    // The ten edge mutual informations of an interaction X,Y → X',AB,Y'.
    // Keyed by the cell's local-label pairs (0=X 1=Y 2=X' 3=AB 4=Y').
    // Also returns the marginal states X' = Tr_Y ρ_AB, Y' = Tr_X ρ_AB.
    struct InteractionResult {
        std::map<std::pair<int, int>, double> edgeMI;
        SystemState statePrimeX;     // X' = Tr_Y ρ_AB
        SystemState statePrimeY;     // Y' = Tr_X ρ_AB
        SystemState stateAB;         // AB carried as its X-marginal proxy
        Eigen::Matrix4cd jointAB;    // ρ_AB in (X' ⊗ Y') order
    };

    // The same construction at 4-dim qudit vertices: ρ are 4×4, joints are
    // 16×16. statePrimeX/Y are the 4-dim marginals; stateAB carries the Σ_AB
    // proxy as a 4-dim state (the A-side marginal of the qudit joint). With
    // featureChoiSigmaAB the Σ_AB vertex instead carries the 256-dim Choi
    // state.
    struct InteractionResultQudit {
        std::map<std::pair<int, int>, double> edgeMI;
        Eigen::Matrix4cd statePrimeX;
        Eigen::Matrix4cd statePrimeY;
        Eigen::Matrix4cd stateAB;
        Eigen::MatrixXcd jointAB;   // 16×16, in (X' ⊗ Y') order
    };
    // (quditChargeOf and quditJointStateFor are declared public above —
    // they're observables intended for inspection from tests/Python.)
    [[nodiscard]] InteractionResult
    computeInteraction(tessera::mesh::VertexPtr x, tessera::mesh::VertexPtr y) const;

    // Qudit analog: 4-dim qudit inputs, 16×16 interaction unitary.
    [[nodiscard]] InteractionResultQudit
    computeInteractionQudit(tessera::mesh::VertexPtr x,
                            tessera::mesh::VertexPtr y) const;

    // The joint state ρ_XY in (X ⊗ Y) order: the stored correlated pair
    // if X, Y share one (initial-layer Delaunay neighbours, or the two
    // products of one interaction), the uncorrelated product otherwise.
    [[nodiscard]] Eigen::Matrix4cd
    jointStateFor(tessera::mesh::VertexPtr x, tessera::mesh::VertexPtr y) const;

    // Metropolis-Hastings acceptance: accepts when -βΔS + logPrefactor ≥ 0,
    // else with probability e^{-βΔS + logPrefactor}.
    [[nodiscard]] bool accept(double deltaS, double logPrefactor = 0.0);

    // ─── Acceptance counters ────────────────────────────────────────────
    std::int64_t interactAttempts_{0},   interactAccepted_{0};
    std::int64_t unInteractAttempts_{0}, unInteractAccepted_{0};
};

} // namespace tessera::simulations
