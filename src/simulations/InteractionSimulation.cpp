// InteractionSimulation — implementation.
//
// See docs/source/interaction-history-monte-carlo.md.
// The construction is local: each interaction works on the participating
// systems' one-qubit density matrices, the joint state
// ρ_AB = U(ρ_X⊗ρ_Y)U†, and conservation-law bookkeeping. There is no
// matrix product state and no global wavefunction — the global correlation
// structure lives in the geometry (the accumulated edge lengths / Regge
// action).

#include "simulations/InteractionSimulation.h"

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/SimplexFilter.h"
#include "mesh/Vertex.h"
#include "observables/MIUnits.hpp"
#include "quantum/ChoiJamiolkowski.h"
#include "quantum/Holography.hpp"
#include "quantum/KoashiImoto.hpp"
#include "spacetime/Metric.h"
#include "spacetime/Spacetime.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <set>
#include <stdexcept>

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

namespace {

using cd = std::complex<double>;
constexpr cd I_UNIT{0.0, 1.0};

// Algebraic maximum mutual information between two single qubits.
// Shared with the holography path via observables/MIUnits.hpp.
using ::tessera::observables::kIMax;

// Von Neumann entropy S(ρ) = -Tr ρ log ρ, in nats.
double vonNeumannEntropy(Eigen::MatrixXcd const& rho) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho);
    double s = 0.0;
    for (double lambda : es.eigenvalues())
        if (lambda > 1e-13) s -= lambda * std::log(lambda);
    return s;
}

// A randomized correlated mixed state on n qubits: ρ = M M† / Tr(M M†)
// with M a complex-Gaussian 2ⁿ × 2ⁿ matrix. Generic — every pair of
// qubits shares genuine mutual information.
Eigen::MatrixXcd randomCorrelatedState(int n, std::mt19937& rng) {
    const int dim = 1 << n;
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::MatrixXcd m(dim, dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            m(i, j) = cd(g(rng), g(rng));
    Eigen::MatrixXcd rho = m * m.adjoint();
    return rho / rho.trace().real();
}

// Reduced density matrix of an n-qubit state, keeping the qubits in
// `keep` (0-based; qubit 0 is the most significant bit). The kept qubits
// keep their listed order, so keep = {i, j} returns the (qubit i ⊗
// qubit j) joint state in the same ordering as tensor2.
Eigen::MatrixXcd partialTrace(Eigen::MatrixXcd const& rho, int n,
                              std::vector<int> const& keep) {
    std::vector<int> traced;
    for (int b = 0; b < n; ++b)
        if (std::find(keep.begin(), keep.end(), b) == keep.end())
            traced.push_back(b);
    const int k = static_cast<int>(keep.size());
    const int t = static_cast<int>(traced.size());
    const int dimK = 1 << k;
    const int dimT = 1 << t;
    auto fullIndex = [&](int kv, int tv) {
        int idx = 0;
        for (int i = 0; i < k; ++i)             // keep[0] = most significant
            if (kv & (1 << (k - 1 - i))) idx |= 1 << (n - 1 - keep[i]);
        for (int i = 0; i < t; ++i)
            if (tv & (1 << (t - 1 - i))) idx |= 1 << (n - 1 - traced[i]);
        return idx;
    };
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(dimK, dimK);
    for (int r = 0; r < dimK; ++r)
        for (int c = 0; c < dimK; ++c)
            for (int tv = 0; tv < dimT; ++tv)
                out(r, c) += rho(fullIndex(r, tv), fullIndex(c, tv));
    return out;
}

// Sorted vertex-pointer pair, the canonical key for jointOf_.
std::pair<VertexPtr, VertexPtr> sortedPair(VertexPtr a, VertexPtr b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

// Kronecker product of two d x d operators, ordering (A (x) B): the (i,k)
// row of the result pairs A's row i with B's row k.
Eigen::MatrixXcd kronDense(Eigen::MatrixXcd const& a,
                           Eigen::MatrixXcd const& b) {
    const int d = static_cast<int>(a.rows());
    Eigen::MatrixXcd out(d * d, d * d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
            for (int k = 0; k < d; ++k)
                for (int l = 0; l < d; ++l)
                    out(d * i + k, d * j + l) = a(i, j) * b(k, l);
    return out;
}

// Swap the two d-dimensional subsystems of a (d*d) x (d*d) operator:
// (A (x) B) -> (B (x) A).
Eigen::MatrixXcd swapSubsystems(Eigen::MatrixXcd const& m, int d) {
    Eigen::MatrixXcd out(d * d, d * d);
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b)
            for (int c = 0; c < d; ++c)
                for (int e = 0; e < d; ++e)
                    out(d * b + a, d * e + c) = m(d * a + b, d * c + e);
    return out;
}

// Swap the two qubits of a two-qubit operator: (X (x) Y) -> (Y (x) X).
Eigen::Matrix4cd swapQubits(Eigen::Matrix4cd const& m) {
    return swapSubsystems(m, 2);
}

// Kronecker product of two one-qubit states, ordering (X (x) Y).
Eigen::Matrix4cd tensor2(SystemState const& a, SystemState const& b) {
    return kronDense(a, b);
}

// Partial traces of a two-qubit joint state. The index convention -- A most
// significant -- is the one quantum::partialTrace{A,B} implement, so these are
// that implementation at dimension two.
SystemState traceOutSecond(Eigen::Matrix4cd const& rho) {  // -> X marginal
    return ::tessera::quantum::partialTraceB(rho, 2, 2);
}
SystemState traceOutFirst(Eigen::Matrix4cd const& rho) {  // -> Y marginal
    return ::tessera::quantum::partialTraceA(rho, 2, 2);
}

// The Schwinger two-site interaction unitary U = exp(-i H_XY dt).
// H_XY is the local two-site Hamiltonian — hopping plus the staggered
// (Kogut-Susskind) mass term — built as a 4×4 dense matrix and
// exponentiated through a Hermitian eigendecomposition. The electric term
// L_n² is non-local in the staggered formulation and is not part of a
// two-site fragment. Reference: Kogut & Susskind, "Hamiltonian formulation
// of Wilson's lattice gauge theories", Phys. Rev. D 11, 395 (1975).
Eigen::Matrix4cd schwingerTwoSiteU(double a, double m, double dt) {
    // Pauli matrices.
    Eigen::Matrix2cd X, Y, Z, Id;
    X << 0, 1, 1, 0;
    Y << 0, -I_UNIT, I_UNIT, 0;
    Z << 1, 0, 0, -1;
    Id << 1, 0, 0, 1;
    auto kron = [](Eigen::Matrix2cd const& p, Eigen::Matrix2cd const& q) {
        Eigen::Matrix4cd r;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    for (int l = 0; l < 2; ++l)
                        r(2 * i + k, 2 * j + l) = p(i, j) * q(k, l);
        return r;
    };
    // H_hop = (1/(4a)) (XX + YY); H_m = (m/2)(-Z⊗I + I⊗Z).
    Eigen::Matrix4cd H =
        (1.0 / (4.0 * a)) * (kron(X, X) + kron(Y, Y))
        + (m / 2.0) * (-kron(Z, Id) + kron(Id, Z));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4cd> es(H);
    Eigen::Vector4cd phase;
    for (int k = 0; k < 4; ++k)
        phase(k) = std::exp(-I_UNIT * dt * es.eigenvalues()(k));
    return es.eigenvectors() * phase.asDiagonal()
           * es.eigenvectors().adjoint();
}

// Mutual information of a two-qubit joint state, in nats.
double jointMutualInformation(Eigen::Matrix4cd const& rho) {
    const double sX = vonNeumannEntropy(traceOutSecond(rho));
    const double sY = vonNeumannEntropy(traceOutFirst(rho));
    const double sXY = vonNeumannEntropy(rho);
    return std::max(sX + sY - sXY, 0.0);
}

// ─── 4-dim qudit-basis helpers ──────────────────────────────────────────
//
// Vertex Hilbert space: basis {|+0⟩, |+1⟩, |−0⟩, |−1⟩}, indexed 0..3.
// The first label is the charge sector, the second the spin / internal.
// Charge operator: Q̂ = diag(+1, +1, −1, −1). Charge-conjugation Ĉ swaps
// the charge bit while leaving spin untouched: Ĉ|q,s⟩ = |¬q, s⟩.

// Charge operator on the 4-dim Hilbert.
Eigen::Matrix4cd quditChargeOp() {
    Eigen::Matrix4cd Q = Eigen::Matrix4cd::Zero();
    Q(0,0) = 1; Q(1,1) = 1; Q(2,2) = -1; Q(3,3) = -1;
    return Q;
}

// Charge-conjugation operator on the 4-dim Hilbert.
Eigen::Matrix4cd quditChargeConj() {
    Eigen::Matrix4cd C = Eigen::Matrix4cd::Zero();
    C(0,2) = 1; C(2,0) = 1; C(1,3) = 1; C(3,1) = 1;
    return C;
}

// Tensor product of two d-dim matrices (d=4 for the ququart pair).
Eigen::MatrixXcd kron4(Eigen::Matrix4cd const& A, Eigen::Matrix4cd const& B) {
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(16, 16);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                for (int l = 0; l < 4; ++l)
                    out(4*i+k, 4*j+l) = A(i,j) * B(k,l);
    return out;
}

// X and Y of the *spin* register only — i.e., the qubit X/Y acting on
// the 2-dim spin subspace, lifted to the 4-dim qudit by ignoring the
// charge index. Layout: bit 0 = spin, bit 1 = charge → index = (q<<1)|s.
Eigen::Matrix4cd quditSpinSx() {
    Eigen::Matrix4cd X = Eigen::Matrix4cd::Zero();
    X(0,1) = 1; X(1,0) = 1; X(2,3) = 1; X(3,2) = 1;
    return X;
}
Eigen::Matrix4cd quditSpinSy() {
    Eigen::Matrix4cd Y = Eigen::Matrix4cd::Zero();
    Y(0,1) = -I_UNIT; Y(1,0) = I_UNIT;
    Y(2,3) = -I_UNIT; Y(3,2) = I_UNIT;
    return Y;
}

// Build the 16×16 pair Hamiltonian H_pair and exponentiate to U.
Eigen::MatrixXcd quditPairU(double jCharge, double jSpin,
                            double massShift, double gammaCp,
                            double dt) {
    const Eigen::Matrix4cd Q = quditChargeOp();
    const Eigen::Matrix4cd C = quditChargeConj();
    const Eigen::Matrix4cd Sx = quditSpinSx();
    const Eigen::Matrix4cd Sy = quditSpinSy();
    const Eigen::Matrix4cd Id4 = Eigen::Matrix4cd::Identity();
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(16, 16);
    // Charge-charge coupling.
    H += jCharge * kron4(Q, Q);
    // Spin-spin coupling (XX + YY) on the spin register.
    H += jSpin * (kron4(Sx, Sx) + kron4(Sy, Sy));
    // Mass-like one-body shift.
    H += massShift * (kron4(Q, Id4) + kron4(Id4, Q));
    // CP-violating coupling (Ĉ ⊗ I + I ⊗ Ĉ, Hermitian as Ĉ² = I).
    if (gammaCp != 0.0)
        H += gammaCp * (kron4(C, Id4) + kron4(Id4, C));
    // Exponentiate via Hermitian eigendecomposition.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    Eigen::VectorXcd phase(16);
    for (int k = 0; k < 16; ++k)
        phase(k) = std::exp(-I_UNIT * dt * es.eigenvalues()(k));
    return es.eigenvectors() * phase.asDiagonal()
           * es.eigenvectors().adjoint();
}

// The same partial traces at dimension four, on a 2-ququart joint: keep A and
// trace out B, or the reverse.
Eigen::Matrix4cd quditTraceOutB(Eigen::MatrixXcd const& rho) {
    return ::tessera::quantum::partialTraceB(rho, 4, 4);
}
Eigen::Matrix4cd quditTraceOutA(Eigen::MatrixXcd const& rho) {
    return ::tessera::quantum::partialTraceA(rho, 4, 4);
}

// The same Kronecker product at dimension four: a 16x16 separable joint.
Eigen::MatrixXcd quditTensor(Eigen::Matrix4cd const& a,
                             Eigen::Matrix4cd const& b) {
    return kronDense(a, b);
}

// The same subsystem swap at dimension four.
Eigen::MatrixXcd quditSwap(Eigen::MatrixXcd const& m) {
    return swapSubsystems(m, 4);
}

// Mutual information of a bipartite 16-dim joint, in nats.
double quditJointMI(Eigen::MatrixXcd const& rho) {
    const Eigen::Matrix4cd rhoA = quditTraceOutB(rho);
    const Eigen::Matrix4cd rhoB = quditTraceOutA(rho);
    const Eigen::MatrixXcd full = rho;
    const double sA  = vonNeumannEntropy(rhoA);
    const double sB  = vonNeumannEntropy(rhoB);
    const double sAB = vonNeumannEntropy(full);
    return std::max(sA + sB - sAB, 0.0);
}

// ℓ = -log(I / I_max), normalised so ℓ ≥ 0; floored when I is tiny so
// uncorrelated systems are far apart but not infinitely so.
double edgeLengthFromMI(double mi, double epsilon) {
    const double x = std::max(mi, 0.0) / kIMax;
    if (x < epsilon) return -std::log(epsilon);
    return -std::log(x);
}

// Wick-rotated to Euclidean signature: every edge contributes +ℓ². The
// spacelike/timelike flag on each cell edge is now informational only —
// the Regge action is real and well-defined everywhere with no complex
// projection step.
double signedSquaredLength(double length, bool /*spacelike*/) {
    return length * length;
}

// The ten edges of the (2,3) cell, with CDT disposition. Local labels:
// 0 = X, 1 = Y, 2 = X', 3 = AB, 4 = Y'. Same time slice -> spacelike.
struct CellEdge {
    int u, v;
    bool spacelike;
};
const CellEdge kCellEdges[10] = {
    {0, 1, true},    // X-Y     Delaunay edge (t=0 slice)
    {0, 2, false},   // X-X'    temporal residual
    {0, 3, false},   // X-AB    temporal, joint MI
    {1, 3, false},   // Y-AB    temporal, joint MI
    {1, 4, false},   // Y-Y'    temporal residual
    {2, 3, true},    // X'-AB   spatial (t=1 slice)
    {3, 4, true},    // Y'-AB   spatial
    {2, 4, true},    // X'-Y'   spatial
    {0, 4, false},   // X-Y'    closure (cross slice)
    {1, 2, false},   // Y-X'    closure
};

// Sum of the hinge contributions A_h ε_h of a single (2,3) cell with the
// given ten signed squared edge lengths. Built in a throwaway Spacetime
// so a rejected Metropolis proposal never touches the live complex.
double cellHingeAction(const double edgeSq[10]) {
    auto metric = std::make_shared<Metric>(
        /*coordinateFree=*/true, Signature(4, SignatureType::Euclidean));
    Spacetime st(metric, SpacetimeType::REGGE, 1.0, 1.0, Foliation::NONE,
                 std::nullopt);
    VertexPtr v[5];
    v[0] = st.createVertex(0, std::vector<double>{0.0});
    v[1] = st.createVertex(1, std::vector<double>{0.0});
    v[2] = st.createVertex(2, std::vector<double>{1.0});
    v[3] = st.createVertex(3, std::vector<double>{1.0});
    v[4] = st.createVertex(4, std::vector<double>{1.0});
    for (int e = 0; e < 10; ++e)
        (void)st.createEdge(v[kCellEdges[e].u], v[kCellEdges[e].v],
                            std::sqrt(std::complex<double>(edgeSq[e])));
    auto [cell, created] = st.createSimplex(
        VertexPtrs{v[0], v[1], v[2], v[3], v[4]});
    (void)created;
    double s = 0.0;
    for (SimplexPtr facet : cell->getFacets())
        for (SimplexPtr hinge : facet->getFacets())
            if (hinge->getVertices().size() == 3)
                s += (hinge->area() * hinge->deficitAngle()).real();
    return s;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────

InteractionSimulation::InteractionSimulation(InteractionConfig config)
    : config_(std::move(config)), rng_(config_.seed) {
    if (config_.nSystems < 2)
        throw std::invalid_argument(
            "InteractionConfig.nSystems must be >= 2");
    if (config_.delaunayEdges.empty())
        throw std::invalid_argument(
            "InteractionConfig.delaunayEdges must be non-empty");
    for (auto const& [i, j] : config_.delaunayEdges)
        if (i < 0 || j < 0 || i >= config_.nSystems
            || j >= config_.nSystems || i == j)
            throw std::invalid_argument(
                "InteractionConfig.delaunayEdges has an out-of-range or "
                "degenerate site-index pair");
    if (config_.a <= 0.0)
        throw std::invalid_argument("InteractionConfig.a must be > 0");

    // Reconcile the legacy `useCharges` alias with `featureCharges`:
    // setting either enables the charged-Cartan code paths.
    const bool chargesOn = config_.useCharges || config_.featureCharges;
    config_.useCharges = chargesOn;
    config_.featureCharges = chargesOn;
    // Dependent features only make sense with charges on; auto-clear
    // them otherwise so we fail safely rather than partially-on.
    if (!chargesOn) {
        config_.featureDeactivateOnAnnihilate = false;
        config_.featurePhotonOnAnnihilate = false;
    }

    auto metric = std::make_shared<Metric>(
        /*coordinateFree=*/true, Signature(4, SignatureType::Euclidean));
    spacetime_ = std::make_shared<Spacetime>(
        metric, SpacetimeType::REGGE, 1.0, 1.0, Foliation::NONE,
        std::nullopt);

    interactionU_ = schwingerTwoSiteU(config_.a, config_.m, config_.dt);

    // Build the 16×16 qudit-pair unitary if requested.
    if (config_.featureQuditBasis) {
        quditInteractionU_ = quditPairU(
            config_.j_chargeCharge, config_.j_spinSpin,
            config_.massShift, config_.gammaCpViolation,
            config_.dtPair);
    }
    // Build the 256-dim Choi state of U once. It requires
    // featureQuditBasis (a Choi state has no meaning without the U it
    // encodes), so clear the flag when the qudit basis is off.
    if (!config_.featureQuditBasis) {
        config_.featureChoiSigmaAB = false;
    }
    if (config_.featureChoiSigmaAB) {
        // J(U) = (U ⊗ I_16) |Φ⁺⟩⟨Φ⁺| (U ⊗ I_16)^H, the Choi matrix of the
        // qudit-pair unitary on the doubled ququart space (|Φ⁺⟩ = (1/4) Σ_k
        // |k,k⟩). Delegated to quantum::ChoiJamiolkowski::choiMatrix so the
        // vec(U) convention (standard, row-major) stays canonical across the
        // codebase.
        std::vector<cd> uFlat(256);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j)
                uFlat[static_cast<std::size_t>(i) * 16 + j] =
                    quditInteractionU_(i, j);
        const std::vector<cd> jFlat = ChoiJamiolkowski::choiMatrix(uFlat, 16);
        quditChoiU_ = Eigen::Map<const Eigen::Matrix<
            cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            jFlat.data(), 256, 256);
    }

    buildInitialLayer();
}

InteractionSimulation::~InteractionSimulation() = default;

void InteractionSimulation::buildInitialLayer() {
    const int n = config_.nSystems;

    // The initial layer is a set of *independent* random mixed states: no
    // mutual information between any pair of initial systems. All MI in
    // the complex is generated by interactions.
    std::vector<SystemState> initialStates(static_cast<std::size_t>(n));
    // Parallel buffer of 4-dim random mixed states confined to a chosen
    // charge sector. Built only when featureQuditBasis is on.
    std::vector<Eigen::Matrix4cd> initialQuditStates(
        static_cast<std::size_t>(n));
    {
        std::normal_distribution<double> g(0.0, 1.0);
        for (int s = 0; s < n; ++s) {
            // ρ_s = M M† / Tr(MM†) with M a 2×2 complex Gaussian — a
            // generic one-qubit mixed state, independent of the others.
            Eigen::Matrix2cd m;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    m(i, j) = cd(g(rng_), g(rng_));
            Eigen::Matrix2cd rho = m * m.adjoint();
            initialStates[static_cast<std::size_t>(s)] =
                rho / rho.trace().real();
        }
    }

    std::vector<VertexPtr> verts(static_cast<std::size_t>(n));
    std::uniform_int_distribution<int> coin(0, 1);
    for (int s = 0; s < n; ++s) {
        VertexPtr v = spacetime_->createVertex(
            nextVertexId_++, std::vector<double>{0.0});
        verts[static_cast<std::size_t>(s)] = v;
        stateOf_[v] = initialStates[static_cast<std::size_t>(s)];
        if (config_.useCharges) {
            const double q =
                (config_.initialChargeMode == InitialChargeMode::ALTERNATING)
                    ? ((s % 2 == 0) ? +1.0 : -1.0)
                    : (coin(rng_) ? +1.0 : -1.0);
            chargeOf_[v] = q;
        }
        if (config_.featureQuditBasis) {
            // Build a 4-dim random mixed state confined to one charge
            // sector: ρ_qudit = P_q · (M M†) · P_q / Tr(P_q · MM†), where
            // P_q is the projector onto the chosen charge subspace
            // (rows/cols 0,1 for +, rows/cols 2,3 for −).
            const bool isPositive =
                (config_.initialChargeMode == InitialChargeMode::ALTERNATING)
                    ? (s % 2 == 0)
                    : coin(rng_);
            std::normal_distribution<double> g4(0.0, 1.0);
            Eigen::Matrix4cd m4;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m4(i, j) = cd(g4(rng_), g4(rng_));
            Eigen::Matrix4cd rho4 = m4 * m4.adjoint();
            // Zero out the off-sector rows/cols.
            for (int i = 0; i < 4; ++i) {
                const bool iIsPositive = (i < 2);
                if (iIsPositive != isPositive) {
                    rho4.row(i).setZero();
                    rho4.col(i).setZero();
                }
            }
            const double tr = rho4.trace().real();
            if (tr > 1e-15) rho4 /= tr;
            initialQuditStates[static_cast<std::size_t>(s)] = rho4;
            quditStateOf_[v] = rho4;
        }
        addToFrontier(v);
    }

    // Delaunay edges with zero MI between independent inputs: stored at
    // the epsilon-floored length so the geometry is well-defined. No
    // jointOf_ entries — the default tensor-product fallback in
    // jointStateFor gives the correct separable joint state.
    const double zeroMiLen = edgeLengthFromMI(0.0, config_.epsilonI);
    const double zeroMiSq  = signedSquaredLength(zeroMiLen,
                                                  /*spacelike=*/true);
    for (auto const& [i, j] : config_.delaunayEdges) {
        VertexPtr a = verts[static_cast<std::size_t>(i)];
        VertexPtr b = verts[static_cast<std::size_t>(j)];
        (void)spacetime_->createEdge(a, b, std::sqrt(std::complex<double>(zeroMiSq)));
    }
}

// The joint state ρ_XY in (X ⊗ Y) qubit order — the stored correlated
// pair if X, Y share one, the uncorrelated product otherwise.
Eigen::Matrix4cd
InteractionSimulation::jointStateFor(VertexPtr x, VertexPtr y) const {
    auto it = jointOf_.find(sortedPair(x, y));
    if (it == jointOf_.end())
        return tensor2(stateOf_.at(x), stateOf_.at(y));
    return (sortedPair(x, y).first == x) ? it->second
                                         : swapQubits(it->second);
}

// ─────────────────────────────────────────────────────────────────────────
// The interaction
// ─────────────────────────────────────────────────────────────────────────

InteractionSimulation::InteractionResult
InteractionSimulation::computeInteraction(VertexPtr x, VertexPtr y) const {
    // The genuine joint input state — correlated when X, Y are Delaunay
    // neighbours of the initial layer or co-products of one interaction.
    const Eigen::Matrix4cd rhoXY = jointStateFor(x, y);

    // ρ_AB = U ρ_XY U† — the genuine joint state after the interaction.
    const Eigen::Matrix4cd rhoAB =
        interactionU_ * rhoXY * interactionU_.adjoint();

    const SystemState primeX = traceOutSecond(rhoAB);  // X' = Tr_Y ρ_AB
    const SystemState primeY = traceOutFirst(rhoAB);   // Y' = Tr_X ρ_AB

    const double sX = vonNeumannEntropy(traceOutSecond(rhoXY));
    const double sY = vonNeumannEntropy(traceOutFirst(rhoXY));
    // I(X:AB): the genuine mutual information sitting in ρ_AB — how much
    // the interaction correlated the two systems. The primary temporal
    // quantity; I(X:X') = S(X) - I(X:AB) is the residual.
    const double iJoint = jointMutualInformation(rhoAB);
    const double iInput = jointMutualInformation(rhoXY);

    InteractionResult res;
    res.statePrimeX = primeX;
    res.statePrimeY = primeY;
    res.stateAB = primeX;  // AB carried forward via its X-marginal proxy
    res.jointAB = rhoAB;   // (X' ⊗ Y') joint, inherited by the products

    // Cartan/local-frame model: under the KAK decomposition
    //   U = (K₁⊗K₂)·exp(i·c·σσ)·(K₃⊗K₄),
    // U_A ≡ K₁K₃ and V_B ≡ K₂K₄ are *local* operators on the A and B
    // worldlines, and Σ_AB ≡ exp(i·c·σσ) is the entangling core. So
    // the post-interaction A-side state (and B-side state) depend only
    // on the respective input — I(B : U_A) = I(A : V_B) = 0 — and the
    // genuine new joint information lives in Σ_AB. The bowtie has:
    //   • two worldline self-info edges A–A' and B–B' carrying S(ρ_A)
    //     and S(ρ_B) respectively (a local unitary preserves entropy);
    //   • four hub-spoke edges to AB, each carrying jointMI(ρ_AB);
    //   • the input spatial edge A–B carrying the input MI;
    //   • and three cross-worldline edges (A–B', B–A', A'–B') carrying
    //     the *input* MI — local unitaries on a single side preserve the
    //     joint, so I(A:V_B) = I(A:B)_input, symmetric for the others.
    //     When the inputs were uncorrelated these reduce to zero
    //     (initial-layer separable layer); when they carry inherited
    //     joint state they reflect that correlation transparently.
    auto key = [](int u, int v) {
        return std::make_pair(std::min(u, v), std::max(u, v));
    };
    res.edgeMI[key(0, 1)] = iInput;   // A-B   input spatial
    res.edgeMI[key(0, 2)] = sX;       // A-A'  worldline self-info
    res.edgeMI[key(1, 4)] = sY;       // B-B'  worldline self-info
    res.edgeMI[key(0, 3)] = iJoint;   // A-AB  hub spoke
    res.edgeMI[key(1, 3)] = iJoint;   // B-AB  hub spoke
    res.edgeMI[key(2, 3)] = iJoint;   // A'-AB hub spoke
    res.edgeMI[key(3, 4)] = iJoint;   // AB-B' hub spoke
    res.edgeMI[key(2, 4)] = iInput;   // A'-B' = I(A:B)_input by unitary invariance
    res.edgeMI[key(0, 4)] = iInput;   // A-B'  = I(A:B)_input
    res.edgeMI[key(1, 2)] = iInput;   // B-A'  = I(A:B)_input
    return res;
}

// ─── Qudit-basis interaction machinery ──────────────────────────────────

namespace {

// Partial trace of a 256×256 Choi state J(U) on the doubled
// (H_AB ⊗ H_A'B') Hilbert, keeping the first 16-dim factor (H_AB)
// and tracing out the second (H_A'B'). Returns a 16×16 mixed state.
// (For a maximally-entangled Choi this gives I_16/16 — uninformative
// alone; the partner-conditioning step below picks the right
// 4-dim subspace.)
Eigen::MatrixXcd traceOutChoiPrimed(Eigen::MatrixXcd const& choi) {
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(16, 16);
    // Index layout: row = a * 16 + b' where a ∈ [0,16) is H_AB,
    // b' ∈ [0,16) is H_A'B'. Trace out the b' / d' diagonal.
    for (int a = 0; a < 16; ++a)
        for (int c = 0; c < 16; ++c)
            for (int k = 0; k < 16; ++k)
                out(a, c) += choi(a * 16 + k, c * 16 + k);
    return out;
}

// Reduce a 256-dim Σ_AB Choi state to a 4-dim "single ququart" state
// in a way that's consistent with the joint correlations stored in
// quditJointOf_. The reduction depends on the partner's charge:
//
//   • Partial-trace the Choi over the "primed" doubled subsystem to
//     get a 16-dim density matrix on H_AB.
//   • The 16-dim H_AB factorises as H_A ⊗ H_B where A is the
//     worldline-continuation side (q_A determined by inputs) and B
//     is the other.
//   • For a partner with charge q_p, project the 16-dim state onto
//     the q_total = q_p subspace (so the joint conserves Q with the
//     partner), then partial-trace down to 4-dim on the "shared"
//     side.
//
// This is the conservation-preserving choice: when a Σ_AB is later
// consumed with partner p, the effective marginal of Σ_AB has
// ⟨Q⟩ = -q_p, ensuring the joint Q sums correctly.
Eigen::Matrix4cd reduceChoiAgainstPartner(Eigen::MatrixXcd const& choi,
                                          double partnerCharge) {
    // Step 1: trace out primed → 16-dim density matrix on H_AB.
    const Eigen::MatrixXcd rho_AB = traceOutChoiPrimed(choi);
    // Step 2: project onto the Q_total = −partnerCharge subspace so
    // the joint with the partner has Q = 0 (charge-conserving
    // contraction). Q_total acts on H_AB = H_A ⊗ H_B with Q̂_A and
    // Q̂_B on each ququart factor; eigenvalues of Q̂_A + Q̂_B are
    // in {−2, 0, +2} (with 0 having an 8-dim subspace, ±2 each
    // 4-dim). Round the partner's charge to the nearest integer for
    // the projection target.
    const int qTarget = -static_cast<int>(std::lround(partnerCharge));
    // Build the projector onto the Q_total = qTarget subspace of the
    // 16-dim H_AB. Basis index = 4*a + b where a, b ∈ [0,4) and
    // Q̂(a) = +1 for a ∈ {0,1}, −1 for a ∈ {2,3} (and same for b).
    auto qOfIndex = [](int idx) {
        const int a = idx / 4;
        const int b = idx % 4;
        const int qA = (a < 2) ? +1 : -1;
        const int qB = (b < 2) ? +1 : -1;
        return qA + qB;
    };
    Eigen::MatrixXcd proj = Eigen::MatrixXcd::Zero(16, 16);
    for (int i = 0; i < 16; ++i)
        if (qOfIndex(i) == qTarget) proj(i, i) = cd(1.0, 0.0);
    Eigen::MatrixXcd projected = proj * rho_AB * proj;
    // Renormalise. If the projection has zero weight (target Q
    // outside ±2), fall back to the I/4 proxy.
    const double tr = projected.trace().real();
    if (tr < 1e-12) {
        return 0.25 * Eigen::Matrix4cd::Identity();
    }
    projected /= tr;
    // Step 3: partial-trace projected (16-dim, on H_A ⊗ H_B) down
    // to H_A (4-dim) — this is the marginal of Σ_AB on the side
    // facing the partner.
    Eigen::Matrix4cd out = Eigen::Matrix4cd::Zero();
    for (int a = 0; a < 4; ++a)
        for (int c = 0; c < 4; ++c)
            for (int b = 0; b < 4; ++b)
                out(a, c) += projected(4*a + b, 4*c + b);
    return out;
}

} // namespace

double InteractionSimulation::quditChargeOf(VertexPtr v) const {
    // When v is a Choi Σ_AB, its single-vertex charge (used by
    // getGlobalCharge for frontier summation) needs care: the Choi
    // state on its own has ⟨Q̂_total⟩ = 0 (maximally entangled),
    // and we want the frontier-sum to track conserved Q. We report
    // 0 for the unconsumed Choi state — at consumption time, the
    // partner-conditioned reduction gives the right marginal.
    if (choiSigmaAbSet_.count(v)) {
        // Choi state's diagonal trace against (Q̂ ⊗ I): identically
        // zero because J(U) has uniform diagonal weight 1/16 on the
        // Q = +2, 0, −2 sectors of the 16-dim H_AB.
        return 0.0;
    }
    auto it = quditStateOf_.find(v);
    if (it == quditStateOf_.end()) return 0.0;
    // ⟨Q⟩ = Tr[ρ · Q̂] where Q̂ = diag(+1,+1,-1,-1).
    const Eigen::Matrix4cd& rho = it->second;
    return (rho(0,0) + rho(1,1) - rho(2,2) - rho(3,3)).real();
}

Eigen::MatrixXcd
InteractionSimulation::quditJointStateFor(VertexPtr x, VertexPtr y) const {
    // Choi-vertex partner-conditioned reduction: if either x or y
    // carries the Choi state, replace its "marginal" used in the
    // separable-fallback tensor product with the partner-conditioned
    // reduction. Stored quditJointOf_ entries override this.
    auto jit = quditJointOf_.find(sortedPair(x, y));
    if (jit != quditJointOf_.end()) {
        if (sortedPair(x, y).first == x) return jit->second;
        return quditSwap(jit->second);
    }
    auto materialise = [&](VertexPtr v) -> Eigen::Matrix4cd {
        if (choiSigmaAbSet_.count(v)) {
            // The Choi-typed Σ_AB has an effective single-vertex
            // marginal of I/4 (the maximally-entangled-on-doubled
            // Hilbert Choi state traces down to (1/16) I_16 on H_AB,
            // and one more partial trace gives (1/4) I_4). Q = 0 by
            // construction. The genuine quantum content of the
            // interaction lives in the (xp, yp) joint that the
            // Choi-mode interact() stored, not in joint contractions
            // through ab.
            return 0.25 * Eigen::Matrix4cd::Identity();
        }
        auto sit = quditStateOf_.find(v);
        if (sit != quditStateOf_.end()) return sit->second;
        return (Eigen::Matrix4cd)(0.25 * Eigen::Matrix4cd::Identity());
    };
    Eigen::Matrix4cd rhoX = materialise(x);
    Eigen::Matrix4cd rhoY = materialise(y);
    return quditTensor(rhoX, rhoY);
}

InteractionSimulation::InteractionResultQudit
InteractionSimulation::computeInteractionQudit(VertexPtr x,
                                               VertexPtr y) const {
    // The genuine joint input state in the qudit basis (16×16).
    const Eigen::MatrixXcd rhoXY = quditJointStateFor(x, y);

    // ρ_AB = U ρ_XY U† — the genuine joint state after the interaction.
    const Eigen::MatrixXcd rhoAB =
        quditInteractionU_ * rhoXY * quditInteractionU_.adjoint();

    const Eigen::Matrix4cd primeX = quditTraceOutB(rhoAB);
    const Eigen::Matrix4cd primeY = quditTraceOutA(rhoAB);

    // Input marginals' entropies (the worldline self-MI).
    const Eigen::Matrix4cd rhoX_in = quditTraceOutB(rhoXY);
    const Eigen::Matrix4cd rhoY_in = quditTraceOutA(rhoXY);
    const double sX = vonNeumannEntropy(rhoX_in);
    const double sY = vonNeumannEntropy(rhoY_in);
    const double iJoint = quditJointMI(rhoAB);
    const double iInput = quditJointMI(rhoXY);

    InteractionResultQudit res;
    res.statePrimeX = primeX;
    res.statePrimeY = primeY;
    // Σ_AB is the entangling-core analog of a neutral photon: use the
    // maximally-mixed 4-dim state, which has ⟨Q̂⟩ = 0 by construction
    // (Tr[I/4 · diag(+1,+1,-1,-1)] = 0). The genuine joint correlations
    // between AB and its bowtie neighbours live in jointAB, not in this
    // proxy. With featureChoiSigmaAB the caller replaces the proxy with
    // the full Choi state of U.
    res.stateAB = 0.25 * Eigen::Matrix4cd::Identity();
    res.jointAB = rhoAB;

    auto key = [](int u, int v) {
        return std::make_pair(std::min(u, v), std::max(u, v));
    };
    res.edgeMI[key(0, 1)] = iInput;
    res.edgeMI[key(0, 2)] = sX;
    res.edgeMI[key(1, 4)] = sY;
    res.edgeMI[key(0, 3)] = iJoint;
    res.edgeMI[key(1, 3)] = iJoint;
    res.edgeMI[key(2, 3)] = iJoint;
    res.edgeMI[key(3, 4)] = iJoint;
    res.edgeMI[key(2, 4)] = iInput;
    res.edgeMI[key(0, 4)] = iInput;
    res.edgeMI[key(1, 2)] = iInput;
    return res;
}

// ─────────────────────────────────────────────────────────────────────────
// Frontier bookkeeping — flat vector + index map, O(1) per mutation.
//
// Any pair of frontier vertices is an eligible interact candidate, so
// N₊ = |frontier|·(|frontier|−1)/2 and a uniform-random candidate is
// picked by drawing two distinct indices into the vector. The
// un-interact denominator uses leafCellCount_ — see consumedProductsOf_.
// ─────────────────────────────────────────────────────────────────────────

void InteractionSimulation::addToFrontier(VertexPtr v) {
    if (v == nullptr || frontierIdx_.count(v)) return;
    frontierIdx_[v] = frontier_.size();
    frontier_.push_back(v);
    if (config_.useCharges) addToSignBucket(v);
}

void InteractionSimulation::removeFromFrontier(VertexPtr v) {
    if (config_.useCharges) removeFromSignBucket(v);
    auto it = frontierIdx_.find(v);
    if (it == frontierIdx_.end()) return;
    const std::size_t pos  = it->second;
    const std::size_t last = frontier_.size() - 1;
    if (pos != last) {
        frontier_[pos] = frontier_[last];
        frontierIdx_[frontier_[pos]] = pos;
    }
    frontier_.pop_back();
    frontierIdx_.erase(it);
}

void InteractionSimulation::addToSignBucket(VertexPtr v) {
    if (v == nullptr) return;
    auto qit = chargeOf_.find(v);
    if (qit == chargeOf_.end()) return;
    const double q = qit->second;
    if (q > 0.0) {
        if (frontierPosIdx_.count(v)) return;
        frontierPosIdx_[v] = frontierPos_.size();
        frontierPos_.push_back(v);
    } else if (q < 0.0) {
        if (frontierNegIdx_.count(v)) return;
        frontierNegIdx_[v] = frontierNeg_.size();
        frontierNeg_.push_back(v);
    }
    // q == 0: no sign bucket (neutrals don't annihilate).
}

void InteractionSimulation::removeFromSignBucket(VertexPtr v) {
    if (v == nullptr) return;
    auto removeFrom = [&](std::vector<VertexPtr>& vec,
                          std::unordered_map<VertexPtr, std::size_t>& idx) {
        auto it = idx.find(v);
        if (it == idx.end()) return false;
        const std::size_t pos  = it->second;
        const std::size_t last = vec.size() - 1;
        if (pos != last) {
            vec[pos] = vec[last];
            idx[vec[pos]] = pos;
        }
        vec.pop_back();
        idx.erase(it);
        return true;
    };
    if (removeFrom(frontierPos_, frontierPosIdx_)) return;
    (void)removeFrom(frontierNeg_, frontierNegIdx_);
}

// ─────────────────────────────────────────────────────────────────────────
// Moves
// ─────────────────────────────────────────────────────────────────────────

bool InteractionSimulation::interact() {
    ++interactAttempts_;
    if (frontier_.size() < 2) return false;
    if (config_.targetInteractions != 0
        && interactionCount_ >= config_.targetInteractions)
        return false;

    // Sample a uniform-random unordered pair of distinct frontier
    // vertices. With useCharges, opposite-sign pairs are ineligible
    // for interaction (they would annihilate); reject and re-sample
    // (up to a small cap) to stay roughly uniform over the eligible
    // subset.
    std::uniform_int_distribution<std::size_t> pick(
        0, frontier_.size() - 1);
    std::size_t i = 0, j = 0;
    VertexPtr x = nullptr, y = nullptr;
    constexpr int kMaxResamples = 32;
    for (int k = 0; k < kMaxResamples; ++k) {
        i = pick(rng_);
        j = pick(rng_);
        if (j == i) continue;
        x = frontier_[i];
        y = frontier_[j];
        if (!config_.useCharges) break;
        const double qx = chargeOf_.count(x) ? chargeOf_.at(x) : 0.0;
        const double qy = chargeOf_.count(y) ? chargeOf_.at(y) : 0.0;
        if (qx * qy >= 0.0) break;  // same-sign or with-neutral: eligible
        x = y = nullptr;
    }
    if (x == nullptr) return false;

    const std::size_t nFrontier   = frontier_.size();
    const std::size_t nPlusBefore = nFrontier * (nFrontier - 1) / 2;

    // Compute the interaction result on the right Hilbert space. The
    // one-qubit result is always populated (it lengths the edges and
    // drives the accept logic); with the qudit basis on, the qudit
    // result supplies the edge MIs for the action and the 4-dim states
    // for the product seeding.
    const InteractionResult       res       = computeInteraction(x, y);
    InteractionResultQudit        resQudit;
    if (config_.featureQuditBasis)
        resQudit = computeInteractionQudit(x, y);

    // The ten signed squared edge lengths of the proposed cell.
    double edgeSq[10];
    for (int e = 0; e < 10; ++e) {
        CellEdge const& ce = kCellEdges[e];
        const auto k = std::make_pair(std::min(ce.u, ce.v),
                                       std::max(ce.u, ce.v));
        const double mi = config_.featureQuditBasis
                              ? resQudit.edgeMI.at(k)
                              : res.edgeMI.at(k);
        const double len = edgeLengthFromMI(mi, config_.epsilonI);
        edgeSq[e] = signedSquaredLength(len, ce.spacelike);
    }

    // ΔS evaluated in a throwaway complex — a rejected proposal never
    // touches the live geometry.
    const double deltaS = cellHingeAction(edgeSq);

    // Reverse-move denominator: the proposed cell will be a leaf in the
    // new state (its three products are fresh), and only leaf cells'
    // deep un-interact returns to exactly the pre-interact state.
    // Counting leaves stays bounded as T grows; using total cells would
    // crater acceptance at large T.
    auto producer = [&](VertexPtr v) -> SimplexPtr {
        auto it = producedByCell_.find(v);
        return (it == producedByCell_.end()) ? nullptr : it->second;
    };
    auto consumedCount = [&](SimplexPtr c) -> unsigned {
        auto it = consumedProductsOf_.find(c);
        return (it == consumedProductsOf_.end()) ? 0u : it->second;
    };
    const SimplexPtr cX = producer(x);
    const SimplexPtr cY = producer(y);
    std::size_t leavesAfter = leafCellCount_ + 1;  // the new cell
    if (cX != nullptr && consumedCount(cX) == 0) --leavesAfter;
    if (cY != nullptr && cY != cX && consumedCount(cY) == 0)
        --leavesAfter;
    const std::size_t nMinusAfter = std::max<std::size_t>(leavesAfter, 1);
    const double logPrefactor =
        std::log(static_cast<double>(nPlusBefore))
        - std::log(static_cast<double>(nMinusAfter));

    if (!accept(deltaS, logPrefactor)) return false;

    // Accepted — build the (2,3) cell into the live complex. Products
    // are placed one time-step after the *later* of the two inputs so
    // every product comes strictly after both its parents (the
    // causal-set indexing of events).
    const double tNext = std::max(x->getTime(), y->getTime()) + 1.0;
    VertexPtr xp = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNext});
    VertexPtr ab = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNext});
    VertexPtr yp = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNext});
    VertexPtr label[5] = {x, y, xp, ab, yp};
    for (int e = 0; e < 10; ++e)
        (void)spacetime_->createEdge(label[kCellEdges[e].u],
                                     label[kCellEdges[e].v],
                                     std::sqrt(std::complex<double>(edgeSq[e])));
    auto [cell, created] =
        spacetime_->createSimplex(VertexPtrs{x, y, xp, ab, yp});
    (void)created;
    for (SimplexPtr facet : cell->getFacets())
        for (SimplexPtr hinge : facet->getFacets())
            if (hinge->getVertices().size() == 3)
                hingeAction_[hinge] =
                    (hinge->area() * hinge->deficitAngle()).real();

    stateOf_[xp] = res.statePrimeX;
    stateOf_[ab] = res.stateAB;
    stateOf_[yp] = res.statePrimeY;
    // Cartan model joints: the entangling correlation lives on the
    // hub-spokes A'–AB and AB–B', not on A'–B'. The (A',B') pair has
    // no entry — jointStateFor falls back to the separable tensor
    // product, giving the structural-zero MI we want.
    auto putJoint = [&](VertexPtr u, VertexPtr v,
                        const Eigen::Matrix4cd& rho) {
        const auto k = sortedPair(u, v);
        jointOf_[k] = (k.first == u) ? rho : swapQubits(rho);
    };
    putJoint(xp, ab, res.jointAB);  // A'–AB joint (= ρ_AB, A' on A-side)
    putJoint(ab, yp, res.jointAB);  // AB–B' joint (= ρ_AB, B' on B-side)

    // Qudit seeding: 4-dim qudit states + 16×16 joints.
    if (config_.featureQuditBasis) {
        quditStateOf_[xp] = resQudit.statePrimeX;
        quditStateOf_[yp] = resQudit.statePrimeY;
        auto putQuditJoint = [&](VertexPtr u, VertexPtr v,
                                 const Eigen::MatrixXcd& rho) {
            const auto k = sortedPair(u, v);
            quditJointOf_[k] = (k.first == u) ? rho : quditSwap(rho);
        };
        if (config_.featureChoiSigmaAB) {
            // Σ_AB carries the full Choi state of U (marker only; the
            // shared 256-dim state is in quditChoiU_). The products'
            // correlation rhoAB is stored as a joint *between the two
            // worldline continuations* (xp ↔ yp), not duplicated as
            // (xp, ab) and (ab, yp); duplicating it double-counts and
            // lets Q drift. The Σ_AB vertex is then a neutral
            // entangling core: q_ab = 0, and any interaction involving
            // it goes through tensor(I/4, partner) as a separable
            // input, which has Q = q_partner and so conserves charge.
            choiSigmaAbSet_.insert(ab);
            putQuditJoint(xp, yp, resQudit.jointAB);
        } else {
            // I/4 proxy with the joint duplicated onto both hub spokes.
            quditStateOf_[ab] = resQudit.stateAB;
            putQuditJoint(xp, ab, resQudit.jointAB);
            putQuditJoint(ab, yp, resQudit.jointAB);
        }
    }

    // Dependency tracking for deep un-interactions.
    producedByCell_[xp] = cell;
    producedByCell_[ab] = cell;
    producedByCell_[yp] = cell;
    consumedByCell_[x] = cell;
    consumedByCell_[y] = cell;

    // Leaf bookkeeping — the new cell joins as a leaf; each of x, y that
    // was itself a product takes its producing cell out of leaf state on
    // the first consumed product.
    consumedProductsOf_[cell] = 0;
    ++leafCellCount_;
    if (cX != nullptr) {
        std::uint8_t &k = consumedProductsOf_[cX];
        if (k == 0) --leafCellCount_;
        ++k;
    }
    if (cY != nullptr) {
        std::uint8_t &k = consumedProductsOf_[cY];
        if (k == 0 && cY != cX) --leafCellCount_;
        ++k;
    }

    // Charge inheritance: worldline products carry their parent's
    // charge, the entangling-core product is neutral.
    if (config_.useCharges) {
        chargeOf_[xp] = chargeOf_.count(x) ? chargeOf_.at(x) : 0.0;
        chargeOf_[yp] = chargeOf_.count(y) ? chargeOf_.at(y) : 0.0;
        chargeOf_[ab] = 0.0;
    }

    // Incremental frontier update — O(1).
    removeFromFrontier(x);
    removeFromFrontier(y);
    addToFrontier(xp);
    addToFrontier(ab);
    addToFrontier(yp);

    ++interactionCount_;
    ++interactAccepted_;
    return true;
}

bool InteractionSimulation::unInteract() {
    ++unInteractAttempts_;
    // Eligible: every (2,3) cell. Removing a past cell truncates the
    // whole future cone of its products — each product is consumed by
    // at most one later cell, so the descendant set is well-defined.
    // Spacetime already partitions top-dimensional simplices into a
    // dedicated vector (topSimplicesVec) with O(1) swap-pop on remove,
    // so we ask it directly instead of scanning the full flat
    // simplices_ vector and filtering by vertex count.
    SimplexPtr root = spacetime_->getRandomTopSimplex();
    if (root == nullptr) return false;

    // BFS through producedByCell_ → consumedByCell_ to collect every
    // cell whose existence depends on root.
    std::vector<SimplexPtr> descendants;
    std::set<SimplexPtr> visited;
    std::vector<SimplexPtr> stack{root};
    visited.insert(root);
    while (!stack.empty()) {
        SimplexPtr c = stack.back();
        stack.pop_back();
        descendants.push_back(c);
        for (VertexPtr v : c->getVertices()) {
            auto pit = producedByCell_.find(v);
            if (pit == producedByCell_.end() || pit->second != c) continue;
            auto cit = consumedByCell_.find(v);
            if (cit == consumedByCell_.end()) continue;
            if (visited.insert(cit->second).second)
                stack.push_back(cit->second);
        }
    }

    // ΔS = −Σ A_h ε_h over the descendant cells' hinges. Read off the
    // per-hinge table; no Spacetime mutation yet.
    double deltaS = 0.0;
    std::set<SimplexPtr> descendantHinges;
    for (SimplexPtr c : descendants)
        for (SimplexPtr facet : c->getFacets())
            for (SimplexPtr h : facet->getFacets())
                if (h->getVertices().size() == 3) {
                    auto it = hingeAction_.find(h);
                    if (it != hingeAction_.end()) {
                        if (descendantHinges.insert(h).second)
                            deltaS -= it->second;
                    }
                }

    // Coarse combinatorial prefactor: N₊ / N₋ where N₋ is the eligible
    // cells (we picked from), and N₊ in the post-uninteract state is the
    // unordered-pair count over the (predicted) post-frontier. Each
    // descendant cell removes 3 product vertices from the frontier and
    // restores 2 parent vertices (its inputs).
    const std::size_t nMinusBefore   = spacetime_->getSimplexCount();
    const std::size_t nFrontierNow   = frontier_.size();
    const std::size_t nProductsLost  = 3 * descendants.size();
    const std::size_t nParentsBack   = 2 * descendants.size();
    const std::size_t nFrontierAfter =
        (nFrontierNow + nParentsBack > nProductsLost)
            ? nFrontierNow + nParentsBack - nProductsLost
            : 0;
    const std::size_t nPlusAfter =
        nFrontierAfter < 2
            ? 1
            : nFrontierAfter * (nFrontierAfter - 1) / 2;
    const double logPrefactor =
        std::log(static_cast<double>(nMinusBefore))
        - std::log(static_cast<double>(nPlusAfter));

    if (!accept(deltaS, logPrefactor)) return false;

    // Commit. Classify every vertex of the descendant set as product
    // (will be deleted) or parent (will re-frontier, unless also a
    // product of some other descendant) before any mutation.
    const std::set<SimplexPtr> descendantSet(descendants.begin(),
                                             descendants.end());
    std::vector<VertexPtr> productsToRemove;
    std::set<VertexPtr> parentsTouched;
    for (SimplexPtr c : descendants) {
        for (VertexPtr v : c->getVertices()) {
            auto pit = producedByCell_.find(v);
            if (pit != producedByCell_.end() && pit->second == c) {
                productsToRemove.push_back(v);
            } else {
                auto cit = consumedByCell_.find(v);
                if (cit != consumedByCell_.end() && cit->second == c)
                    consumedByCell_.erase(cit);
                parentsTouched.insert(v);
                // v was a product of some cell c'_v (perhaps); when v
                // got consumed by c, c'_v's consumed-product count went
                // up. If c'_v survives, that count now goes back down,
                // potentially making c'_v a leaf again.
                auto vp = producedByCell_.find(v);
                if (vp != producedByCell_.end()
                    && descendantSet.count(vp->second) == 0) {
                    auto kit = consumedProductsOf_.find(vp->second);
                    if (kit != consumedProductsOf_.end()
                        && kit->second > 0) {
                        --kit->second;
                        if (kit->second == 0) ++leafCellCount_;
                    }
                }
            }
        }
    }
    const std::set<VertexPtr> productSet(productsToRemove.begin(),
                                         productsToRemove.end());

    // Each descendant cell exits the leaf pool (if it was one) and its
    // bookkeeping entry is dropped.
    for (SimplexPtr c : descendants) {
        auto kit = consumedProductsOf_.find(c);
        if (kit != consumedProductsOf_.end()) {
            if (kit->second == 0 && leafCellCount_ > 0) --leafCellCount_;
            consumedProductsOf_.erase(kit);
        }
    }

    // 1. Remove the simplices first so vertex/edge removal doesn't
    //    dangle simplex references.
    for (SimplexPtr c : descendants) spacetime_->removeSimplex(c);

    // 2. Drop hingeAction entries for the descendants.
    for (SimplexPtr h : descendantHinges) hingeAction_.erase(h);

    // 3. Drop the product vertices' states / dependency / frontier
    //    entries, then remove them from the complex (which also removes
    //    their edges).
    for (VertexPtr v : productsToRemove) {
        removeFromFrontier(v);
        stateOf_.erase(v);
        chargeOf_.erase(v);
        producedByCell_.erase(v);
        for (auto it = jointOf_.begin(); it != jointOf_.end();) {
            if (it->first.first == v || it->first.second == v)
                it = jointOf_.erase(it);
            else
                ++it;
        }
        spacetime_->removeVertex(v);
    }

    // 4. Re-frontier the parents whose subtree just got cut and that
    //    aren't themselves slated for deletion.
    for (VertexPtr v : parentsTouched)
        if (productSet.count(v) == 0) addToFrontier(v);

    if (interactionCount_ >= descendants.size())
        interactionCount_ -= descendants.size();
    else
        interactionCount_ = 0;
    ++unInteractAccepted_;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Charge moves — annihilate / pairCreate
// ─────────────────────────────────────────────────────────────────────────

bool InteractionSimulation::annihilate() {
    if (!config_.useCharges) return false;
    ++annihilateAttempts_;
    if (frontierPos_.empty() || frontierNeg_.empty()) return false;

    std::uniform_int_distribution<std::size_t> pickPos(
        0, frontierPos_.size() - 1);
    std::uniform_int_distribution<std::size_t> pickNeg(
        0, frontierNeg_.size() - 1);
    VertexPtr p = frontierPos_[pickPos(rng_)];
    VertexPtr m = frontierNeg_[pickNeg(rng_)];

    const double qp = chargeOf_.at(p);
    const double qm = chargeOf_.at(m);  // qm < 0

    // Symmetric Metropolis prefactor: forward and reverse rates are
    // both ∝ 1 over the candidate-pool sizes. ΔS = 0.
    const std::size_t fwdCount = frontierPos_.size() * frontierNeg_.size();
    const std::size_t revCount = 1;
    const double logPrefactor =
        std::log(static_cast<double>(fwdCount))
        - std::log(static_cast<double>(revCount));
    if (!accept(/*deltaS=*/0.0, logPrefactor)) return false;

    // Three behaviours selected by feature flags:
    //   1. Default (charge-only): both vertices stay on the frontier
    //      with charges neutralised / reduced. The worldlines persist
    //      as neutral systems. Preserves cell references but breaks
    //      Q-conservation under later un-interact.
    //   2. feature_deactivate_on_annihilate: the matched portion's
    //      vertices are *removed from the frontier* (worldlines
    //      terminate) but stay in the spacetime. Their chargeOf_
    //      entries are kept as-is — they're historical charges, no
    //      longer counted in getGlobalCharge (frontier-only sum).
    //      Q is then conserved exactly under later un-interact.
    //   3. feature_photon_on_annihilate: in addition to (2), spawn a
    //      new neutral "photon" vertex on the frontier carrying away
    //      the released information (state = I/2, maximally mixed).
    const double netQ = qp + qm;
    const bool deactivate = config_.featureDeactivateOnAnnihilate;
    const bool emitPhoton = config_.featurePhotonOnAnnihilate;

    auto setCharge = [&](VertexPtr v, double q) {
        // Update charge and migrate sign-bucket membership.
        removeFromSignBucket(v);
        chargeOf_[v] = q;
        if (q != 0.0 && frontierIdx_.count(v)) addToSignBucket(v);
    };
    auto deactivateVertex = [&](VertexPtr v) {
        // Remove v from the frontier; keep chargeOf_ / stateOf_ / the
        // spacetime entry intact so historical cell references still
        // resolve. The vertex is now "inactive" — can't be picked for
        // future moves.
        removeFromFrontier(v);
    };
    auto spawnPhoton = [&](double tNew) {
        if (!emitPhoton) return;
        VertexPtr photon = spacetime_->createVertex(
            nextVertexId_++, std::vector<double>{tNew});
        stateOf_[photon] = 0.5 * Eigen::Matrix2cd::Identity();
        chargeOf_[photon] = 0.0;
        addToFrontier(photon);
    };

    if (std::abs(netQ) < 1e-12) {
        // Full annihilation — both worldlines terminate (or both
        // neutralise under the default).
        const double tNew =
            std::max(p->getTime(), m->getTime()) + 1.0;
        if (deactivate) {
            deactivateVertex(p);
            deactivateVertex(m);
        } else {
            setCharge(p, 0.0);
            setCharge(m, 0.0);
        }
        spawnPhoton(tNew);
    } else if (netQ > 0.0) {
        // |qp| > |qm|: the m-worldline fully annihilates (matched by
        // |qm| of p's charge); p survives with reduced charge.
        const double tNew = m->getTime() + 1.0;
        if (deactivate) {
            deactivateVertex(m);
        } else {
            setCharge(m, 0.0);
        }
        setCharge(p, netQ);
        spawnPhoton(tNew);
    } else {
        // |qm| > |qp|: the p-worldline fully annihilates; m survives
        // with the residual negative charge.
        const double tNew = p->getTime() + 1.0;
        if (deactivate) {
            deactivateVertex(p);
        } else {
            setCharge(p, 0.0);
        }
        setCharge(m, netQ);
        spawnPhoton(tNew);
    }
    ++annihilateAccepted_;
    return true;
}

bool InteractionSimulation::pairCreate() {
    if (!config_.useCharges) return false;
    ++pairCreateAttempts_;

    // Draw charges with CP-violation bias. ε_CP > 0 favours +
    // (δ uniform on [0, +ε_CP], mean ε_CP/2 > 0; pair net charge mean
    // ε_CP > 0); ε_CP < 0 favours − (δ uniform on [ε_CP, 0], mean
    // ε_CP/2 < 0). Default ε_CP = 0 gives δ = 0 — exactly symmetric.
    const double eps = config_.cpBias;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double delta;
    if (eps > 0.0) {
        delta = eps * u(rng_);                 // [0, +ε_CP]
    } else if (eps < 0.0) {
        delta = eps * u(rng_);                 // [ε_CP, 0]  (eps<0)
    } else {
        delta = 0.0;
    }
    const double qPos = 0.5 + delta;        // (0, 1]
    const double qNeg = -(1.0 - qPos);      // [−1, 0)

    // Metropolis prefactor: forward = 1, reverse = N_pos·N_neg AFTER.
    const std::size_t revCount =
        (frontierPos_.size() + 1) * (frontierNeg_.size() + 1);
    const std::size_t fwdCount = 1;
    const double logPrefactor =
        std::log(static_cast<double>(fwdCount))
        - std::log(static_cast<double>(revCount));
    if (!accept(/*deltaS=*/0.0, logPrefactor)) return false;

    // Build the Bell state |Φ+⟩⟨Φ+| as the 2-qubit joint of the new
    // pair, with maximally-mixed marginals I/2 on each vertex.
    Eigen::Matrix4cd bell = Eigen::Matrix4cd::Zero();
    bell(0, 0) = 0.5;  bell(0, 3) = 0.5;
    bell(3, 0) = 0.5;  bell(3, 3) = 0.5;
    SystemState halfMixed = 0.5 * Eigen::Matrix2cd::Identity();

    // Spawn the two vertices at time = current frontier max + 1 so
    // they're "born now" causally. If frontier_ is empty pick t=0.
    double tMax = 0.0;
    for (VertexPtr v : frontier_)
        tMax = std::max(tMax, v->getTime());
    const double tNew = tMax + 1.0;

    VertexPtr vPos = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNew});
    VertexPtr vNeg = spacetime_->createVertex(
        nextVertexId_++, std::vector<double>{tNew});
    stateOf_[vPos] = halfMixed;
    stateOf_[vNeg] = halfMixed;
    chargeOf_[vPos] = qPos;
    chargeOf_[vNeg] = qNeg;
    const auto key = sortedPair(vPos, vNeg);
    jointOf_[key] = (key.first == vPos) ? bell : swapQubits(bell);
    addToFrontier(vPos);
    addToFrontier(vNeg);
    ++pairCreateAccepted_;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Driving loop
// ─────────────────────────────────────────────────────────────────────────

int InteractionSimulation::sweep() {
    // Move count scales with N₊ + N₋ — frontier-pair count plus the
    // total cell count (every cell is uninteractable under deep
    // truncation).
    const std::size_t nFront = frontier_.size();
    const std::size_t nPairs = nFront < 2 ? 0 : nFront * (nFront - 1) / 2;
    const std::size_t nMoves = std::max<std::size_t>(
        1, nPairs + interactionCount_);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    int accepted = 0;
    for (std::size_t k = 0; k < nMoves; ++k) {
        if (config_.useCharges) {
            // Four-move sweep: interact / unInteract / annihilate /
            // pairCreate with equal proposal probability.
            const double r = coin(rng_);
            bool ok = false;
            if      (r < 0.25) ok = interact();
            else if (r < 0.50) ok = (interactionCount_ > 0) && unInteract();
            else if (r < 0.75) ok = annihilate();
            else               ok = pairCreate();
            if (ok) ++accepted;
        } else {
            const bool doInteract =
                interactionCount_ == 0 ? true
                : nFront < 2           ? false
                                       : coin(rng_) < 0.5;
            if (doInteract ? interact() : unInteract()) ++accepted;
        }
    }
    return accepted;
}

void InteractionSimulation::tune(std::function<void(int, int)> progress) {
    const std::size_t target = config_.targetInteractions;
    std::size_t guard = 0;
    const std::size_t guardMax = (target == 0) ? 0 : 100 * target;
    while (interactionCount_ < target && guard < guardMax) {
        interact();
        ++guard;
        if (progress)
            progress(static_cast<int>(interactionCount_),
                     static_cast<int>(target));
    }
}

void InteractionSimulation::thermalize() {
    if (config_.targetInteractions != 0) tune();

    double prevAction = computeAction();
    for (int s = 0; s < 1000; ++s) {
        sweep();
        const double action = computeAction();
        const double denom =
            std::abs(prevAction) > 1e-12 ? std::abs(prevAction) : 1.0;
        if (s > 5 && std::abs(action - prevAction) / denom < 0.01) break;
        prevAction = action;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────

double InteractionSimulation::computeAction() const {
    double s = 0.0;
    for (auto const& [hinge, contrib] : hingeAction_) {
        (void)hinge;
        s += contrib;
    }
    return s;
}

std::vector<double>
InteractionSimulation::getDeficitAngleDistribution() const {
    std::vector<double> out;
    for (SimplexPtr s : spacetime_->getSimplices()) {
        if (s->getVertices().size() != 3) continue;
        if (s->getCofaces().empty()) continue;
        out.push_back(s->deficitAngle().real());
    }
    return out;
}

std::vector<int> InteractionSimulation::getVolumeProfile() const {
    std::vector<int> profile;
    for (SimplexPtr s : spacetime_->getSimplices()) {
        if (s->getVertices().size() != 5) continue;
        double earliest = std::numeric_limits<double>::infinity();
        for (VertexPtr v : s->getVertices())
            earliest = std::min(earliest, v->getTime());
        const int slice = static_cast<int>(std::lround(earliest));
        if (static_cast<int>(profile.size()) <= slice)
            profile.resize(static_cast<std::size_t>(slice) + 1, 0);
        ++profile[static_cast<std::size_t>(slice)];
    }
    return profile;
}

std::map<std::string, double>
InteractionSimulation::getAcceptanceRates() const {
    auto rate = [](std::int64_t acc, std::int64_t att) {
        return att > 0 ? static_cast<double>(acc) / static_cast<double>(att)
                       : 0.0;
    };
    std::map<std::string, double> out{
        {"interact",   rate(interactAccepted_,   interactAttempts_)},
        {"unInteract", rate(unInteractAccepted_, unInteractAttempts_)},
    };
    if (config_.useCharges) {
        out["annihilate"] = rate(annihilateAccepted_, annihilateAttempts_);
        out["pairCreate"] = rate(pairCreateAccepted_, pairCreateAttempts_);
    }
    return out;
}

double InteractionSimulation::getGlobalCharge() const {
    // Sum over frontier (live) vertices only. Past inputs whose charge
    // has already propagated to product worldlines are excluded — their
    // charge is now carried by their descendants.
    double q = 0.0;
    if (config_.featureQuditBasis) {
        // Qudit basis: read charge from Tr[ρ · Q̂] per vertex.
        for (VertexPtr v : frontier_)
            q += quditChargeOf(v);
    } else {
        for (VertexPtr v : frontier_) {
            auto it = chargeOf_.find(v);
            if (it != chargeOf_.end()) q += it->second;
        }
    }
    return q;
}

std::vector<std::array<double, 4>>
InteractionSimulation::getChargeProfile() const {
    std::vector<std::array<double, 4>> profile;
    for (auto const& [v, qv] : chargeOf_) {
        const int slice = static_cast<int>(std::lround(v->getTime()));
        if (slice < 0) continue;
        if (static_cast<int>(profile.size()) <= slice)
            profile.resize(static_cast<std::size_t>(slice) + 1,
                           {0.0, 0.0, 0.0, 0.0});
        auto& row = profile[static_cast<std::size_t>(slice)];
        if      (qv > 0.0) row[0] += 1.0;
        else if (qv < 0.0) row[2] += 1.0;
        else               row[1] += 1.0;
        row[3] += qv;
    }
    return profile;
}

std::vector<double>
InteractionSimulation::getChargeCorrelation(int maxDist) const {
    std::vector<double> out(static_cast<std::size_t>(std::max(maxDist, 1)),
                             0.0);
    if (maxDist <= 0 || chargeOf_.empty()) return out;

    // Build a CSR-style adjacency on the MI-weighted graph (we ignore
    // edge weights here — only graph-distance matters).
    std::unordered_map<VertexPtr, int> idx;
    int n = 0;
    for (auto const& kv : chargeOf_) idx[kv.first] = n++;
    std::vector<std::vector<int>> adj(n);
    for (EdgePtr e : spacetime_->getEdgeList()->toVector()) {
        VertexPtr a = e->getSource();
        VertexPtr b = e->getTarget();
        auto ia = idx.find(a), ib = idx.find(b);
        if (ia == idx.end() || ib == idx.end()) continue;
        adj[ia->second].push_back(ib->second);
        adj[ib->second].push_back(ia->second);
    }
    std::vector<double> charges(n, 0.0);
    for (auto const& kv : chargeOf_) charges[idx.at(kv.first)] = kv.second;

    std::vector<long long> count(maxDist, 0);
    std::vector<double>   sumQQ(maxDist, 0.0);
    std::vector<int> dist(n, -1);

    // BFS from a sample of source vertices (to keep this O(n · maxDist ·
    // avg_degree) for a small sample fraction rather than O(n^2)).
    const int sampleStride = std::max(1, n / 200);
    for (int s = 0; s < n; s += sampleStride) {
        std::fill(dist.begin(), dist.end(), -1);
        dist[s] = 0;
        std::vector<int> frontier{s};
        for (int d = 1; d <= maxDist; ++d) {
            std::vector<int> next;
            for (int u : frontier)
                for (int w : adj[u])
                    if (dist[w] < 0) {
                        dist[w] = d;
                        next.push_back(w);
                        ++count[d - 1];
                        sumQQ[d - 1] += charges[s] * charges[w];
                    }
            frontier.swap(next);
            if (frontier.empty()) break;
        }
    }
    for (int d = 0; d < maxDist; ++d)
        out[d] = (count[d] > 0)
                     ? sumQQ[d] / static_cast<double>(count[d])
                     : 0.0;
    return out;
}

bool InteractionSimulation::accept(double deltaS, double logPrefactor) {
    const double exponent = -config_.beta * deltaS + logPrefactor;
    if (exponent >= 0.0) return true;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_) < std::exp(exponent);
}

// ─────────────────────────────────────────────────────────────────────────
std::vector<double> InteractionSimulation::getSpectralDimension(
    const std::vector<double>& sigmas, int krylovDim) const {
    // The dimension we want is that of the simplicial complex of
    // completed (2,3) 4-simplices. We measure its 1-skeleton: systems
    // as nodes, the cells' edges MI-weighted. Bare initial-layer edges
    // that never joined a cell are part of the primal interaction
    // lattice and are excluded by the topK == 4 size filter.
    //
    // Delegates to the shared Spacetime → SpectralGraph path so the
    // holography and interaction-history pipelines share one measurement
    // code path. AllSimplexFilter lets every 4-simplex pass.
    return spacetime_->getSpectralDimensionOnSkeleton(
        sigmas, krylovDim, ::tessera::AllSimplexFilter{},
        /*topK=*/4, /*skeletonDim=*/1);
}

} // namespace tessera::simulations
