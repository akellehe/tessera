#include "quantum/QuantumSimplex.hpp"

#include "mesh/Edge.h"
#include "mesh/Vertex.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tessera::quantum {

namespace {

// All ten pairs (i, j) with i < j over the five vertex positions of
// a 4-simplex (C(5, 2) = 10).
struct PositionPair { int u, v; };
constexpr PositionPair kAllEdges[10] = {
    {0, 1}, {0, 2}, {0, 3}, {0, 4},
    {1, 2}, {1, 3}, {1, 4},
    {2, 3}, {2, 4},
    {3, 4},
};

// ρ_u ⊗ ρ_v in (u ⊗ v) ordering.
Eigen::MatrixXcd kron(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) {
    const int dA = static_cast<int>(a.rows());
    const int dB = static_cast<int>(b.rows());
    Eigen::MatrixXcd out(dA * dB, dA * dB);
    for (int i = 0; i < dA; ++i)
        for (int j = 0; j < dA; ++j)
            for (int k = 0; k < dB; ++k)
                for (int l = 0; l < dB; ++l)
                    out(i * dB + k, j * dB + l) = a(i, j) * b(k, l);
    return out;
}

// The Van Raamsdonk law is Edge::vanRaamsdonkLength. epsilon = 0 opts out of
// its floor, so an uncorrelated pair gives the divergent +inf rather than a
// capped length: on a KI cell a vanishing mutual information means the two
// systems are unrelated, not merely weakly related.
constexpr double kNoFloor = 0.0;

double vrFromJoint(const Eigen::MatrixXcd& rhoAB,
                   const Eigen::MatrixXcd& rhoA,
                   const Eigen::MatrixXcd& rhoB,
                   double                  iMax) {
    const double I = mutualInformation(rhoAB, rhoA, rhoB);
    return ::tessera::mesh::Edge::vanRaamsdonkLength(I, iMax, kNoFloor);
}

struct SortedEig {
    Eigen::VectorXd  evals;
    Eigen::MatrixXcd evecs;
};

SortedEig descendingEig(const Eigen::MatrixXcd& rho) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(rho);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("QuantumSimplex: eigendecomposition failed");
    }
    SortedEig out;
    out.evals = es.eigenvalues().reverse();
    out.evecs = Eigen::MatrixXcd(es.eigenvectors().rows(),
                                 es.eigenvectors().cols());
    for (Eigen::Index c = 0; c < es.eigenvectors().cols(); ++c) {
        out.evecs.col(c) = es.eigenvectors().col(
            es.eigenvectors().cols() - 1 - c);
    }
    return out;
}

void requireMatchedSpectra(const Eigen::VectorXd& evalsA,
                           const Eigen::VectorXd& evalsB,
                           double                 tolerance) {
    if (evalsA.size() != evalsB.size()) {
        throw std::invalid_argument(
            "QuantumSimplex: ρ_A and ρ_B must have equal dimension "
            "for the requested factory");
    }
    for (Eigen::Index i = 0; i < evalsA.size(); ++i) {
        if (std::abs(evalsA[i] - evalsB[i]) > tolerance) {
            throw std::invalid_argument(
                "QuantumSimplex: this factory requires ρ_A and ρ_B "
                "to share an eigenvalue spectrum; spectra differ");
        }
    }
}

// |ψ⟩⟨ψ| with |ψ⟩ = Σ_i √λ_i |a_i⟩|b_i⟩ in matched eigenbases.
Eigen::MatrixXcd schmidtJointFromMarginals(const Eigen::MatrixXcd& rhoA,
                                           const Eigen::MatrixXcd& rhoB) {
    const auto eigA = descendingEig(rhoA);
    const auto eigB = descendingEig(rhoB);
    const int d = static_cast<int>(eigA.evals.size());
    Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(d * d);
    for (int a = 0; a < d; ++a) {
        for (int b = 0; b < d; ++b) {
            std::complex<double> acc(0.0, 0.0);
            for (int i = 0; i < d; ++i) {
                const double lambda = std::max(0.0, eigA.evals[i]);
                acc += eigA.evecs(a, i) * eigB.evecs(b, i)
                       * std::sqrt(lambda);
            }
            psi[a * d + b] = acc;
        }
    }
    return psi * psi.adjoint();
}

// ρ_AB = Σ_i λ_i |a_i⟩⟨a_i| ⊗ |b_i⟩⟨b_i| in matched eigenbases.
Eigen::MatrixXcd classicalJointFromMarginals(const Eigen::MatrixXcd& rhoA,
                                             const Eigen::MatrixXcd& rhoB) {
    const auto eigA = descendingEig(rhoA);
    const auto eigB = descendingEig(rhoB);
    const int d = static_cast<int>(eigA.evals.size());
    Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(d * d, d * d);
    for (int i = 0; i < d; ++i) {
        const double lambda = std::max(0.0, eigA.evals[i]);
        if (lambda <= 0.0) continue;
        Eigen::VectorXcd aOut = eigA.evecs.col(i);
        Eigen::VectorXcd bOut = eigB.evecs.col(i);
        Eigen::MatrixXcd Pa = aOut * aOut.adjoint();
        Eigen::MatrixXcd Pb = bOut * bOut.adjoint();
        for (int p = 0; p < d; ++p) {
            for (int q = 0; q < d; ++q) {
                for (int r = 0; r < d; ++r) {
                    for (int s = 0; s < d; ++s) {
                        out(p * d + r, q * d + s) +=
                            lambda * Pa(p, q) * Pb(r, s);
                    }
                }
            }
        }
    }
    return out;
}

// Build the five-vertex / ten-edge mesh::Simplex from the joint ρ_AB on
// (qva, qvb). Allocates the Σ, A', B' QuantumVertex objects in the
// spacetime's vertex list, carrying the KI core and tail states, and
// computes d_VR per edge through Edge::vanRaamsdonkLength: for (A, B) from
// the input joint's mutual information, for every other edge from a mutual
// information of zero. d_VR is stored as the edge length at edge creation.
::tessera::mesh::Simplex*
buildSimplexFromJoint(::tessera::spacetime::Spacetime& spacetime,
                      QuantumVertex*                   qva,
                      QuantumVertex*                   qvb,
                      const Eigen::MatrixXcd&          rhoAB,
                      double                           iMax,
                      const KoashiImotoTolerances&     tol) {
    if (qva == nullptr || qvb == nullptr) {
        throw std::invalid_argument(
            "QuantumSimplex: input QuantumVertex pointers must be non-null");
    }
    if (!(iMax > 0.0)) {
        throw std::invalid_argument(
            "QuantumSimplex: iMax must be positive");
    }
    const Eigen::MatrixXcd& rhoA = qva->getState();
    const Eigen::MatrixXcd& rhoB = qvb->getState();
    const int dimA = qva->stateDim();
    const int dimB = qvb->stateDim();
    if (rhoAB.rows() != rhoAB.cols()
        || rhoAB.rows() != static_cast<Eigen::Index>(dimA) * dimB) {
        throw std::invalid_argument(
            "QuantumSimplex: rhoAB dimension must equal dimA · dimB");
    }

    const KoashiImotoResult ki =
        koashiImotoDecompose(rhoAB, rhoA, rhoB, tol);

    auto vlist = spacetime.getVertexList();
    std::array<QuantumVertex*, 5> positions{};
    positions[QuantumSimplex::A] = qva;
    positions[QuantumSimplex::B] = qvb;
    {
        const auto id = spacetime.reserveVertexId();
        positions[QuantumSimplex::Sigma] =
            vlist->template addAs<QuantumVertex>(id, id, ki.getSigma());
    }
    {
        const auto id = spacetime.reserveVertexId();
        positions[QuantumSimplex::APrime] =
            vlist->template addAs<QuantumVertex>(id, id, ki.getAPrime());
    }
    {
        const auto id = spacetime.reserveVertexId();
        positions[QuantumSimplex::BPrime] =
            vlist->template addAs<QuantumVertex>(id, id, ki.getBPrime());
    }

    // d_VR per edge. (A, B) uses the input joint directly; every
    // other pair asks the endpoint vertex (product joint).
    ::tessera::mesh::Edges edges;
    edges.reserve(10);
    for (const auto& [u, v] : kAllEdges) {
        double dvr;
        if ((u == QuantumSimplex::A && v == QuantumSimplex::B)
            || (u == QuantumSimplex::B && v == QuantumSimplex::A)) {
            dvr = vrFromJoint(rhoAB, rhoA, rhoB, iMax);
        } else {
            // Every pair but (A, B) is built from marginals that carry no
            // inherited correlation, so the mutual information is zero.
            dvr = ::tessera::mesh::Edge::vanRaamsdonkLength(
                0.0, iMax, kNoFloor);
        }
        const double dvrSq = dvr * dvr;
        ::tessera::mesh::EdgePtr edge =
            spacetime.createEdge(positions[u], positions[v], std::sqrt(std::complex<double>(dvrSq)));
        if (edge != nullptr) edges.push_back(edge);
    }

    ::tessera::mesh::VertexPtrs verts;
    verts.reserve(5);
    for (int p = QuantumSimplex::A; p <= QuantumSimplex::BPrime; ++p) {
        verts.push_back(positions[p]);
    }
    auto [s, created] = spacetime.createSimplex(verts, edges);
    (void)created;
    return s;
}

} // namespace

::tessera::mesh::Simplex*
QuantumSimplex::fromSchmidtPurification(
    ::tessera::spacetime::Spacetime& spacetime,
    QuantumVertex*                   qva,
    QuantumVertex*                   qvb,
    double                           iMax,
    const KoashiImotoTolerances&     tol) {
    if (qva == nullptr || qvb == nullptr) {
        throw std::invalid_argument(
            "fromSchmidtPurification: inputs must be non-null");
    }
    const auto eigA = descendingEig(qva->getState());
    const auto eigB = descendingEig(qvb->getState());
    requireMatchedSpectra(eigA.evals, eigB.evals, tol.getEpsKiEigen());
    auto rhoAB = schmidtJointFromMarginals(qva->getState(), qvb->getState());
    return buildSimplexFromJoint(spacetime, qva, qvb, rhoAB, iMax, tol);
}

::tessera::mesh::Simplex*
QuantumSimplex::fromClassicalCorrelation(
    ::tessera::spacetime::Spacetime& spacetime,
    QuantumVertex*                   qva,
    QuantumVertex*                   qvb,
    double                           iMax,
    const KoashiImotoTolerances&     tol) {
    if (qva == nullptr || qvb == nullptr) {
        throw std::invalid_argument(
            "fromClassicalCorrelation: inputs must be non-null");
    }
    const auto eigA = descendingEig(qva->getState());
    const auto eigB = descendingEig(qvb->getState());
    requireMatchedSpectra(eigA.evals, eigB.evals, tol.getEpsKiEigen());
    auto rhoAB = classicalJointFromMarginals(qva->getState(), qvb->getState());
    return buildSimplexFromJoint(spacetime, qva, qvb, rhoAB, iMax, tol);
}

::tessera::mesh::Simplex*
QuantumSimplex::fromExplicitJoint(
    ::tessera::spacetime::Spacetime& spacetime,
    QuantumVertex*                   qva,
    QuantumVertex*                   qvb,
    const Eigen::MatrixXcd&          rhoAB,
    double                           iMax,
    const KoashiImotoTolerances&     tol) {
    return buildSimplexFromJoint(spacetime, qva, qvb, rhoAB, iMax, tol);
}

::tessera::mesh::Simplex*
QuantumSimplex::fromTargetMutualInformation(
    ::tessera::spacetime::Spacetime& spacetime,
    QuantumVertex*                   qva,
    QuantumVertex*                   qvb,
    double                           targetMI,
    double                           iMax,
    const KoashiImotoTolerances&     tol) {
    if (qva == nullptr || qvb == nullptr) {
        throw std::invalid_argument(
            "fromTargetMutualInformation: inputs must be non-null");
    }
    if (targetMI < 0.0) {
        throw std::invalid_argument(
            "fromTargetMutualInformation: targetMI must be >= 0");
    }
    const auto eigA = descendingEig(qva->getState());
    const auto eigB = descendingEig(qvb->getState());
    requireMatchedSpectra(eigA.evals, eigB.evals, tol.getEpsKiEigen());

    const Eigen::MatrixXcd rhoA    = qva->getState();
    const Eigen::MatrixXcd rhoB    = qvb->getState();
    const Eigen::MatrixXcd product = kron(rhoA, rhoB);
    const Eigen::MatrixXcd schmidt = schmidtJointFromMarginals(rhoA, rhoB);
    const double miMax = mutualInformation(schmidt, rhoA, rhoB);
    if (targetMI > miMax + tol.getEpsKiCondState()) {
        throw std::invalid_argument(
            "fromTargetMutualInformation: target exceeds the "
            "achievable maximum 2·H(λ) for the given marginals");
    }
    // Short-circuit the trivial endpoints, so the binary search cannot
    // leave a residual α > 0 on an MI = 0 target.
    if (targetMI <= tol.getEpsKiCondState()) {
        return buildSimplexFromJoint(spacetime, qva, qvb, product, iMax, tol);
    }
    if (std::abs(targetMI - miMax) <= tol.getEpsKiCondState()) {
        return buildSimplexFromJoint(spacetime, qva, qvb, schmidt, iMax, tol);
    }
    double lo = 0.0, hi = 1.0;
    Eigen::MatrixXcd rhoAB;
    for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo + hi);
        rhoAB = (1.0 - mid) * product + mid * schmidt;
        const double I = mutualInformation(rhoAB, rhoA, rhoB);
        if (std::abs(I - targetMI) < tol.getEpsKiCondState()) break;
        if (I < targetMI) lo = mid;
        else              hi = mid;
    }
    return buildSimplexFromJoint(spacetime, qva, qvb, rhoAB, iMax, tol);
}

} // namespace tessera::quantum
