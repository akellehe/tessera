// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "observables/ColorFiber.h"

#include "quantum/GradedFock.h"

#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/KroneckerProduct>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>

namespace tessera::observables {

namespace {

using tessera::quantum::ExteriorAlgebra;

/// The three-mode exterior algebra this class delegates to.  Generated once
/// as a function-local static; the object itself is a lightweight handle.
const ExteriorAlgebra& colorAlgebra() {
    static const ExteriorAlgebra algebra(3);
    return algebra;
}

/// Self-adjoint matrix modulus square root |B|^{1/2} = U |λ|^{1/2} U† of a
/// 3×3 Hermitian block, plus the Krein signature of B under `tol`.
Eigen::Matrix3cd modulusSqrt(const Eigen::Matrix3cd& block, double tol,
                             std::array<int, 3>* signature) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd> es(block);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "ColorAnchor: eigen decomposition of a restricted weight block "
            "failed");
    }
    const Eigen::Vector3d lambda = es.eigenvalues();
    Eigen::Vector3d moduliSqrt;
    int nPos = 0;
    int nZero = 0;
    int nNeg = 0;
    for (int k = 0; k < 3; ++k) {
        const double v = lambda(k);
        moduliSqrt(k) = std::sqrt(std::abs(v));
        if (v > tol) {
            ++nPos;
        } else if (v < -tol) {
            ++nNeg;
        } else {
            ++nZero;
        }
    }
    if (signature != nullptr) {
        *signature = {nPos, nZero, nNeg};
    }
    return es.eigenvectors() * moduliSqrt.asDiagonal() *
           es.eigenvectors().adjoint();
}

/// The signed restriction R_τ Φ: row k = signs[k] · Φ(edges[k], :).
Eigen::Matrix3cd signedRestriction(const Eigen::MatrixXcd& frame,
                                   const OrientedTriangle& tri) {
    Eigen::Matrix3cd out;
    for (int k = 0; k < 3; ++k) {
        out.row(k) =
            static_cast<double>(tri.signs[static_cast<std::size_t>(k)]) *
            frame.row(tri.edges[static_cast<std::size_t>(k)]);
    }
    return out;
}

void validateFrame(const Eigen::MatrixXcd& frame) {
    if (frame.cols() != 3) {
        throw std::invalid_argument(
            "ColorAnchor: the frame must have exactly three columns (a "
            "rank-three band), got " +
            std::to_string(frame.cols()));
    }
}

void validateEdgeRange(const std::vector<OrientedTriangle>& tris,
                       Eigen::Index edgeCount) {
    for (const auto& tri : tris) {
        for (const auto e : tri.edges) {
            if (e >= edgeCount) {
                throw std::invalid_argument(
                    "ColorAnchor: triangle edge index " + std::to_string(e) +
                    " out of range for " + std::to_string(edgeCount) +
                    " edges");
            }
        }
    }
}

/// General-matrix modulus |W| = U|Λ|U† of a Hermitian E×E weight.
Eigen::MatrixXcd matrixModulus(const Eigen::MatrixXcd& weight) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(weight);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "ColorAnchor: eigen decomposition of the weight matrix failed");
    }
    return es.eigenvectors() *
           es.eigenvalues().cwiseAbs().asDiagonal() *
           es.eigenvectors().adjoint();
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════
// ColorFiber — the constant three-edge algebra
// ═══════════════════════════════════════════════════════════════════════

Eigen::MatrixXcd ColorFiber::sectorProjector(std::size_t occupation) {
    return Eigen::MatrixXcd(colorAlgebra().sectorProjector(occupation));
}

Eigen::MatrixXcd ColorFiber::vacuumProjector() { return sectorProjector(0); }
Eigen::MatrixXcd ColorFiber::tripletProjector() { return sectorProjector(1); }
Eigen::MatrixXcd ColorFiber::antiTripletProjector() {
    return sectorProjector(2);
}
Eigen::MatrixXcd ColorFiber::singletProjector() { return sectorProjector(3); }

Eigen::MatrixXcd ColorFiber::creationMatrix(std::size_t mode) {
    return Eigen::MatrixXcd(colorAlgebra().creationMatrix(mode));
}

Eigen::MatrixXcd ColorFiber::annihilationMatrix(std::size_t mode) {
    return Eigen::MatrixXcd(colorAlgebra().annihilationMatrix(mode));
}

Eigen::MatrixXcd ColorFiber::hoppingMatrix(std::size_t i, std::size_t j) {
    return Eigen::MatrixXcd(colorAlgebra().creationMatrix(i) *
                            colorAlgebra().annihilationMatrix(j));
}

std::array<std::size_t, 3> ColorFiber::tripletBasisIndices() {
    // |e_i> = a_i† Ω sits at Fock index 2^i.
    return {1, 2, 4};
}

Eigen::Matrix3cd ColorFiber::restrictToTriplet(const Eigen::MatrixXcd& op) {
    if (op.rows() != 8 || op.cols() != 8) {
        throw std::invalid_argument(
            "ColorFiber::restrictToTriplet: expected an 8x8 Fock operator, "
            "got " +
            std::to_string(op.rows()) + "x" + std::to_string(op.cols()));
    }
    const auto idx = tripletBasisIndices();
    Eigen::Matrix3cd out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = op(static_cast<Eigen::Index>(
                               idx[static_cast<std::size_t>(r)]),
                           static_cast<Eigen::Index>(
                               idx[static_cast<std::size_t>(c)]));
        }
    }
    return out;
}

Eigen::Matrix3cd ColorFiber::matrixUnit(std::size_t i, std::size_t j) {
    if (i >= 3 || j >= 3) {
        throw std::invalid_argument(
            "ColorFiber::matrixUnit: indices must be in 0..2");
    }
    Eigen::Matrix3cd out = Eigen::Matrix3cd::Zero();
    out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = 1.0;
    return out;
}

Eigen::MatrixXcd ColorFiber::dGamma(const Eigen::Matrix3cd& m) {
    return Eigen::MatrixXcd(colorAlgebra().dGamma(m));
}

Eigen::Matrix3cd ColorFiber::gellMann(int a) {
    const Complex i(0.0, 1.0);
    switch (a) {
        case 1:
            return matrixUnit(0, 1) + matrixUnit(1, 0);
        case 2:
            return -i * (matrixUnit(0, 1) - matrixUnit(1, 0));
        case 3:  // H1 = E11 - E22
            return matrixUnit(0, 0) - matrixUnit(1, 1);
        case 4:
            return matrixUnit(0, 2) + matrixUnit(2, 0);
        case 5:
            return -i * (matrixUnit(0, 2) - matrixUnit(2, 0));
        case 6:
            return matrixUnit(1, 2) + matrixUnit(2, 1);
        case 7:
            return -i * (matrixUnit(1, 2) - matrixUnit(2, 1));
        case 8:  // H2 = (E11 + E22 - 2 E33) / sqrt(3)
            return (matrixUnit(0, 0) + matrixUnit(1, 1) -
                    2.0 * matrixUnit(2, 2)) /
                   std::sqrt(3.0);
        default:
            throw std::invalid_argument(
                "ColorFiber::gellMann: generator index must be in 1..8");
    }
}

Eigen::MatrixXcd ColorFiber::adjointOctetProjector() {
    Eigen::VectorXcd vecI = Eigen::VectorXcd::Zero(9);
    vecI(0) = 1.0;  // (0,0)
    vecI(4) = 1.0;  // (1,1)
    vecI(8) = 1.0;  // (2,2)
    return Eigen::MatrixXcd::Identity(9, 9) -
           (vecI * vecI.adjoint()) / 3.0;
}

Eigen::Matrix3cd ColorFiber::tracelessPart(const Eigen::Matrix3cd& m) {
    return m - (m.trace() / 3.0) * Eigen::Matrix3cd::Identity();
}

// ── The adjoint singlet/octet decomposition ────────────────────────────────

Eigen::MatrixXcd ColorFiber::adjointSingletProjector() {
    // The complement of the octet projector, so P₁ + P₈ = I₉ is a
    // bitwise-exact resolution of 3 ⊗ 3̄ = 1 ⊕ 8.
    return Eigen::MatrixXcd::Identity(9, 9) - adjointOctetProjector();
}

Eigen::MatrixXcd ColorFiber::octetBilinear(std::size_t i, std::size_t j) {
    if (i >= 3 || j >= 3) {
        throw std::invalid_argument(
            "ColorFiber::octetBilinear: indices must be in 0..2");
    }
    // The traceless even bilinear a_i†a_j − (δ_ij/3) N̂ is dΓ of the
    // traceless matrix unit.
    return dGamma(tracelessPart(matrixUnit(i, j)));
}

Eigen::MatrixXcd ColorFiber::adjointCasimirMatrix() {
    // C = Σ_a K_a², K_a = I ⊗ (λ_a/2) − (λ_a/2)ᵀ ⊗ I acting on the
    // column-major vec(M): K_a vec(M) = vec([λ_a/2, M]), by the Kronecker
    // rule vec(A M B) = (Bᵀ ⊗ A) vec(M).  Generated once as a function-local
    // static.  The commutator sum is built independently of P₈ so that the
    // cross-check against 3 P₈ in verifyConstantAlgebra is meaningful.
    static const Eigen::MatrixXcd casimir = [] {
        Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(9, 9);
        const Eigen::Matrix3cd id = Eigen::Matrix3cd::Identity();
        for (int a = 1; a <= 8; ++a) {
            const Eigen::Matrix3cd half = 0.5 * gellMann(a);
            const Eigen::MatrixXcd k =
                Eigen::kroneckerProduct(id, half) -
                Eigen::kroneckerProduct(half.transpose(), id);
            sum += k * k;
        }
        return sum;
    }();
    return casimir;
}

double ColorFiber::adjointCasimir(const Eigen::Matrix3cd& m) {
    const double norm2 = m.squaredNorm();
    if (norm2 == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Evaluated through the identity C = 3 P₈: the quadratic form
    // ⟨vec M, C vec M⟩ equals 3 ‖P₈ vec M‖² = 3 ‖M − (Tr M/3) I‖_F².
    return 3.0 * tracelessPart(m).squaredNorm() / norm2;
}

ColorFiber::Complex ColorFiber::omega() {
    // The algebraic value (−1 + i√3)/2 rather than exp(2πi/3): with these
    // components 1 + ω + ω̄ cancels exactly in floating point.
    return Complex(-0.5, std::sqrt(3.0) / 2.0);
}

Eigen::Matrix3cd ColorFiber::fourierFrame() {
    const Complex w = omega();
    const std::array<Complex, 3> table{Complex(1.0, 0.0), w, std::conj(w)};
    const double invSqrt3 = 1.0 / std::sqrt(3.0);
    Eigen::Matrix3cd f;
    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
            f(j, k) = table[static_cast<std::size_t>((j * k) % 3)] * invSqrt3;
        }
    }
    return f;
}

Eigen::Vector3cd ColorFiber::fourierBasisVector(int k) {
    if (k < 0 || k > 2) {
        throw std::invalid_argument(
            "ColorFiber::fourierBasisVector: k must be in 0..2");
    }
    return fourierFrame().col(k);
}

Eigen::Vector3cd ColorFiber::omegaPhaseState() { return fourierBasisVector(1); }

double ColorFiber::perimeter(const Eigen::Vector3cd& z) {
    double p = 0.0;
    for (int k = 0; k < 3; ++k) {
        p += std::sqrt(std::abs(z(k)));
    }
    return p;
}

Eigen::Vector3cd ColorFiber::perimeterNormalized(const Eigen::Vector3cd& z) {
    const double p = perimeter(z);
    if (p == 0.0) {
        throw std::invalid_argument(
            "ColorFiber::perimeterNormalized: zero perimeter");
    }
    return z / (p * p);
}

double ColorFiber::hilbertNorm(const Eigen::Vector3cd& z) { return z.norm(); }

Eigen::Vector3cd ColorFiber::hilbertNormalized(const Eigen::Vector3cd& z) {
    const double n = z.norm();
    if (n == 0.0) {
        throw std::invalid_argument(
            "ColorFiber::hilbertNormalized: zero Hilbert norm");
    }
    return z / n;
}

Eigen::Vector3cd ColorFiber::colorVector(const Eigen::Vector3cd& z) {
    return hilbertNormalized(z);
}

ColorFiber::Complex ColorFiber::colorWedge(const Eigen::Matrix3cd& c) {
    return c.determinant();
}

ColorFiber::Complex ColorFiber::colorWedge(const Eigen::Vector3cd& a,
                                           const Eigen::Vector3cd& b,
                                           const Eigen::Vector3cd& c) {
    Eigen::Matrix3cd m;
    m.col(0) = a;
    m.col(1) = b;
    m.col(2) = c;
    return colorWedge(m);
}

double ColorFiber::singletGram(const Eigen::Matrix3cd& c) {
    return (c.adjoint() * c).determinant().real();
}

bool ColorFiber::isSpecialUnitary(const Eigen::Matrix3cd& g, double tol) {
    const double unitary =
        (g.adjoint() * g - Eigen::Matrix3cd::Identity()).cwiseAbs().maxCoeff();
    const double special = std::abs(g.determinant() - Complex(1.0, 0.0));
    return unitary <= tol && special <= tol;
}

ColorFiber::SectorWeights ColorFiber::sectorWeights(
    const Eigen::VectorXcd& state) {
    if (state.size() != 8) {
        throw std::invalid_argument(
            "ColorFiber::sectorWeights: expected an 8-dimensional Fock "
            "vector, got size " +
            std::to_string(state.size()));
    }
    SectorWeights out;
    for (unsigned b = 0; b < 8; ++b) {
        const double w = std::norm(state(static_cast<Eigen::Index>(b)));
        switch (std::popcount(b)) {
            case 0:
                out.vacuum += w;
                break;
            case 1:
                out.quark += w;
                break;
            case 2:
                out.antiTriplet += w;
                break;
            default:
                out.singlet += w;
                break;
        }
    }
    return out;
}

ColorFiber::OctetRead ColorFiber::octetRead(const Eigen::Matrix3cd& m) {
    OctetRead out;
    out.octet = tracelessPart(m).squaredNorm();
    out.singlet = std::norm(m.trace()) / 3.0;
    return out;
}

double ColorFiber::verifyConstantAlgebra() {
    double residual = 0.0;
    const auto track = [&residual](double r) {
        residual = std::max(residual, r);
    };

    // ω algebra: 1 + ω + ω² = 0 (exact with the algebraic components) and
    // ω³ = 1 to round-off.
    const Complex w = omega();
    track(std::abs(1.0 + w + std::conj(w)));
    track(std::abs(w * w - std::conj(w)));
    track(std::abs(w * w * w - 1.0));

    // F₃†F₃ = I and |det F₃| = 1.
    const Eigen::Matrix3cd f = fourierFrame();
    track((f.adjoint() * f - Eigen::Matrix3cd::Identity())
              .cwiseAbs()
              .maxCoeff());
    track(std::abs(std::abs(f.determinant()) - 1.0));
    // The identified basis vector and its cyclic triad are F₃'s columns.
    track((omegaPhaseState() - f.col(1)).cwiseAbs().maxCoeff());

    // Gell-Mann: Hermitian, traceless, Tr(λ_a λ_b) = 2 δ_ab.
    for (int a = 1; a <= 8; ++a) {
        const Eigen::Matrix3cd la = gellMann(a);
        track((la - la.adjoint()).cwiseAbs().maxCoeff());
        track(std::abs(la.trace()));
        for (int b = 1; b <= 8; ++b) {
            const Complex tr = (la * gellMann(b)).trace();
            track(std::abs(tr - (a == b ? Complex(2.0, 0.0)
                                        : Complex(0.0, 0.0))));
        }
    }

    // gl(3): [E_ij, E_kl] = δ_jk E_il − δ_il E_kj on the 3×3 matrix units
    // and on the full 8×8 Fock bilinears.
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            for (std::size_t k = 0; k < 3; ++k) {
                for (std::size_t l = 0; l < 3; ++l) {
                    const Eigen::Matrix3cd lhs3 =
                        matrixUnit(i, j) * matrixUnit(k, l) -
                        matrixUnit(k, l) * matrixUnit(i, j);
                    Eigen::Matrix3cd rhs3 = Eigen::Matrix3cd::Zero();
                    if (j == k) rhs3 += matrixUnit(i, l);
                    if (i == l) rhs3 -= matrixUnit(k, j);
                    track((lhs3 - rhs3).cwiseAbs().maxCoeff());

                    const Eigen::MatrixXcd eij = hoppingMatrix(i, j);
                    const Eigen::MatrixXcd ekl = hoppingMatrix(k, l);
                    Eigen::MatrixXcd rhs8 =
                        Eigen::MatrixXcd::Zero(8, 8);
                    if (j == k) rhs8 += hoppingMatrix(i, l);
                    if (i == l) rhs8 -= hoppingMatrix(k, j);
                    track((eij * ekl - ekl * eij - rhs8)
                              .cwiseAbs()
                              .maxCoeff());
                }
            }
        }
    }

    // Sector projectors: idempotent, mutually orthogonal, complete, with
    // traces 1, 3, 3, 1 (Λ•C³ = 1 ⊕ 3 ⊕ 3̄ ⊕ 1).
    Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(8, 8);
    const std::array<double, 4> dims{1.0, 3.0, 3.0, 1.0};
    for (std::size_t n = 0; n < 4; ++n) {
        const Eigen::MatrixXcd p = sectorProjector(n);
        track((p * p - p).cwiseAbs().maxCoeff());
        track(std::abs(p.trace() - dims[n]));
        for (std::size_t m2 = 0; m2 < n; ++m2) {
            track((p * sectorProjector(m2)).cwiseAbs().maxCoeff());
        }
        sum += p;
    }
    track((sum - Eigen::MatrixXcd::Identity(8, 8)).cwiseAbs().maxCoeff());

    // The adjoint-octet projector: Hermitian, idempotent, rank 8, fixing
    // every λ_a and annihilating the identity.
    const Eigen::MatrixXcd p8 = adjointOctetProjector();
    track((p8 - p8.adjoint()).cwiseAbs().maxCoeff());
    track((p8 * p8 - p8).cwiseAbs().maxCoeff());
    track(std::abs(p8.trace() - Complex(8.0, 0.0)));
    for (int a = 1; a <= 8; ++a) {
        const Eigen::Matrix3cd la = gellMann(a);
        const Eigen::VectorXcd v =
            Eigen::Map<const Eigen::VectorXcd>(la.data(), 9);
        track((p8 * v - v).cwiseAbs().maxCoeff());
    }
    {
        const Eigen::Matrix3cd id = Eigen::Matrix3cd::Identity();
        const Eigen::VectorXcd v =
            Eigen::Map<const Eigen::VectorXcd>(id.data(), 9);
        track((p8 * v).cwiseAbs().maxCoeff());
    }

    // The singlet complement resolves 3 ⊗ 3̄ = 1 ⊕ 8: P₁ + P₈ = I₉ (bitwise
    // by construction), P₁ idempotent, Hermitian and rank one, and
    // P₁P₈ = P₈P₁ = 0.
    {
        const Eigen::MatrixXcd p1 = adjointSingletProjector();
        track((p1 + p8 - Eigen::MatrixXcd::Identity(9, 9))
                  .cwiseAbs()
                  .maxCoeff());
        track((p1 - p1.adjoint()).cwiseAbs().maxCoeff());
        track((p1 * p1 - p1).cwiseAbs().maxCoeff());
        track(std::abs(p1.trace() - Complex(1.0, 0.0)));
        track((p1 * p8).cwiseAbs().maxCoeff());
        track((p8 * p1).cwiseAbs().maxCoeff());
    }

    // The traceless even bilinears: T_ij = E_ij − (δ_ij/3) Σ_k E_kk, exact
    // tracelessness of the family, commutation with the fermion parity
    // (−1)^N, and the N = 1 restriction to the traceless matrix unit.
    {
        Eigen::MatrixXcd number = Eigen::MatrixXcd::Zero(8, 8);
        for (std::size_t k = 0; k < 3; ++k) number += hoppingMatrix(k, k);
        Eigen::MatrixXcd parity = Eigen::MatrixXcd::Zero(8, 8);
        for (unsigned b = 0; b < 8; ++b) {
            parity(b, b) = (std::popcount(b) % 2 == 0) ? 1.0 : -1.0;
        }
        Eigen::MatrixXcd diagonalSum = Eigen::MatrixXcd::Zero(8, 8);
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                const Eigen::MatrixXcd t = octetBilinear(i, j);
                Eigen::MatrixXcd want = hoppingMatrix(i, j);
                if (i == j) want -= number / 3.0;
                track((t - want).cwiseAbs().maxCoeff());
                track((t * parity - parity * t).cwiseAbs().maxCoeff());
                track((restrictToTriplet(t) -
                       tracelessPart(matrixUnit(i, j)))
                          .cwiseAbs()
                          .maxCoeff());
                if (i == j) diagonalSum += t;
            }
        }
        track(diagonalSum.cwiseAbs().maxCoeff());
    }

    // The adjoint quadratic Casimir equals 3 P₈: zero on the singlet,
    // C₂(adjoint) = 3 on every octet direction.
    {
        const Eigen::MatrixXcd casimir = adjointCasimirMatrix();
        track((casimir - 3.0 * p8).cwiseAbs().maxCoeff());
        for (int a = 1; a <= 8; ++a) {
            track(std::abs(adjointCasimir(gellMann(a)) - 3.0));
        }
        track(std::abs(adjointCasimir(Eigen::Matrix3cd::Identity())));
    }

    // The N = 1 identification: restrictToTriplet ∘ dΓ = id on 3×3, and
    // restrictToTriplet(E_ij) is the matrix unit.
    for (int a = 1; a <= 8; ++a) {
        const Eigen::Matrix3cd la = gellMann(a);
        track((restrictToTriplet(dGamma(la)) - la).cwiseAbs().maxCoeff());
    }
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            track((restrictToTriplet(hoppingMatrix(i, j)) - matrixUnit(i, j))
                      .cwiseAbs()
                      .maxCoeff());
        }
    }

    return residual;
}

::tessera::cobordism::Certificate ColorFiber::constantAlgebraCertificate() {
    return ::tessera::cobordism::Certificate::algebraicallyExact(
        ::tessera::cobordism::CertificateDomain::Static,
        ::tessera::cobordism::CertificateRegime::PositiveSemidefinite,
        verifyConstantAlgebra(), 1e-12);
}

#ifndef NDEBUG
namespace {
/// Debug builds verify the constant algebra once at startup.  Release builds
/// skip the check; tests exercise verifyConstantAlgebra in every build.
struct ColorFiberStartupCheck {
    ColorFiberStartupCheck() {
        const double residual = ColorFiber::verifyConstantAlgebra();
        if (!(residual <= 1e-12)) {
            std::fprintf(stderr,
                         "ColorFiber constant-algebra startup check FAILED: "
                         "max residual %.3e > 1e-12\n",
                         residual);
            std::abort();
        }
    }
};
const ColorFiberStartupCheck colorFiberStartupCheck{};
}  // namespace
#endif

// ═══════════════════════════════════════════════════════════════════════
// ColorAnchor — the calibrated weighted oriented-triangle anchor
// ═══════════════════════════════════════════════════════════════════════

void ColorAnchor::validateTriangles(
    const std::vector<OrientedTriangle>& tris) {
    if (tris.empty()) {
        throw std::invalid_argument(
            "ColorAnchor: an anchor needs at least one declared oriented "
            "triangle");
    }
    for (const auto& tri : tris) {
        if (tri.edges[0] == tri.edges[1] || tri.edges[0] == tri.edges[2] ||
            tri.edges[1] == tri.edges[2]) {
            throw std::invalid_argument(
                "ColorAnchor: a triangle's three boundary edges must be "
                "distinct");
        }
        for (const auto e : tri.edges) {
            if (e < 0) {
                throw std::invalid_argument(
                    "ColorAnchor: negative edge index");
            }
        }
        for (const auto s : tri.signs) {
            if (s != 1 && s != -1) {
                throw std::invalid_argument(
                    "ColorAnchor: incidence signs must be +1 or -1");
            }
        }
    }
}

void ColorAnchor::validateConvex(const std::vector<double>& weights,
                                 std::size_t count) {
    if (weights.size() != count) {
        throw std::invalid_argument(
            "ColorAnchor: need one convex weight per declared triangle (" +
            std::to_string(count) + "), got " +
            std::to_string(weights.size()));
    }
    double sum = 0.0;
    for (const double w : weights) {
        if (!(w >= 0.0)) {
            throw std::invalid_argument(
                "ColorAnchor: convex weights must be non-negative");
        }
        sum += w;
    }
    if (std::abs(sum - 1.0) > 1e-12) {
        throw std::invalid_argument(
            "ColorAnchor: convex weights must sum to one (got " +
            std::to_string(sum) + ")");
    }
}

ColorAnchor::ColorAnchor(std::vector<OrientedTriangle> triangles)
    : triangles_(std::move(triangles)),
      weightingId_("uniform") {
    validateTriangles(triangles_);
    weights_.assign(triangles_.size(),
                    1.0 / static_cast<double>(triangles_.size()));
    markOverlaps();
}

ColorAnchor::ColorAnchor(std::vector<OrientedTriangle> triangles,
                         std::vector<double> weights)
    : triangles_(std::move(triangles)),
      weights_(std::move(weights)),
      weightingId_("declared") {
    validateTriangles(triangles_);
    validateConvex(weights_, triangles_.size());
    markOverlaps();
}

void ColorAnchor::markOverlaps() {
    // Which declared triangles overlap: an edge row occurring in more than
    // one triangle marks every triangle carrying it as overlapping.  This is
    // the shared-edge relation, the only sharing relation an OrientedTriangle
    // atlas determines, since a triangle declares edge rows and incidence
    // signs rather than vertex identities.
    overlapping_.assign(triangles_.size(), 0);
    overlapCount_ = 0;
    std::map<Eigen::Index, std::vector<std::size_t>> byEdge;
    for (std::size_t t = 0; t < triangles_.size(); ++t)
        for (const Eigen::Index e : triangles_[t].edges)
            byEdge[e].push_back(t);
    for (const auto& entry : byEdge) {
        if (entry.second.size() < 2) continue;
        for (const std::size_t t : entry.second) overlapping_[t] = 1;
    }
    for (const char flag : overlapping_)
        if (flag != 0) ++overlapCount_;
}

void ColorAnchor::declareWeights(std::vector<double> weights) {
    if (sealed_) {
        throw std::logic_error(
            "ColorAnchor: post-hoc weight selection rejected — the convex "
            "weighting must be declared before the data are examined "
            "(evaluate() has already run)");
    }
    validateConvex(weights, triangles_.size());
    weights_ = std::move(weights);
    weightingId_ = "declared";
}

Eigen::Matrix3cd ColorAnchor::anchorMatrix(const Eigen::MatrixXcd& frame,
                                           const Eigen::VectorXd& edgeWeights,
                                           const OrientedTriangle& tri) {
    validateFrame(frame);
    if (edgeWeights.size() != frame.rows()) {
        throw std::invalid_argument(
            "ColorAnchor::anchorMatrix: edgeWeights size must match the "
            "frame's edge rows");
    }
    validateTriangles({tri});
    validateEdgeRange({tri}, frame.rows());
    // Diagonal weights: the τ-oriented restricted block S W_τ S is again
    // diagonal (the signs cancel), so |W_τ|^{1/2} = diag(|w_e|^{1/2}).
    Eigen::Matrix3cd a = signedRestriction(frame, tri);
    for (int k = 0; k < 3; ++k) {
        a.row(k) *= std::sqrt(std::abs(
            edgeWeights(tri.edges[static_cast<std::size_t>(k)])));
    }
    return a;
}

Eigen::MatrixXcd ColorAnchor::orthonormalizeFrame(
    const Eigen::MatrixXcd& frame, const Eigen::VectorXd& edgeWeights) {
    validateFrame(frame);
    if (edgeWeights.size() != frame.rows()) {
        throw std::invalid_argument(
            "ColorAnchor::orthonormalizeFrame: edgeWeights size must match "
            "the frame's edge rows");
    }
    const Eigen::MatrixXcd gram =
        frame.adjoint() *
        edgeWeights.cwiseAbs().cast<std::complex<double>>().asDiagonal() *
        frame;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(gram);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "ColorAnchor::orthonormalizeFrame: Gram eigen decomposition "
            "failed");
    }
    const double maxEig = es.eigenvalues().cwiseAbs().maxCoeff();
    if (maxEig <= 0.0 ||
        es.eigenvalues().minCoeff() <= 1e-13 * maxEig) {
        throw std::invalid_argument(
            "ColorAnchor::orthonormalizeFrame: the frame is rank-deficient "
            "in the |W| inner product");
    }
    const Eigen::MatrixXcd invSqrt =
        es.eigenvectors() *
        es.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
        es.eigenvectors().adjoint();
    return frame * invSqrt;
}

Eigen::MatrixXcd ColorAnchor::orthonormalizeFrame(
    const Eigen::MatrixXcd& frame, const Eigen::MatrixXcd& weight) {
    validateFrame(frame);
    if (weight.rows() != frame.rows() || weight.cols() != frame.rows()) {
        throw std::invalid_argument(
            "ColorAnchor::orthonormalizeFrame: the weight matrix must be "
            "square over the frame's edge rows");
    }
    const Eigen::MatrixXcd absW = matrixModulus(weight);
    const Eigen::MatrixXcd gram = frame.adjoint() * absW * frame;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(gram);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "ColorAnchor::orthonormalizeFrame: Gram eigen decomposition "
            "failed");
    }
    const double maxEig = es.eigenvalues().cwiseAbs().maxCoeff();
    if (maxEig <= 0.0 ||
        es.eigenvalues().minCoeff() <= 1e-13 * maxEig) {
        throw std::invalid_argument(
            "ColorAnchor::orthonormalizeFrame: the frame is rank-deficient "
            "in the |W| inner product");
    }
    const Eigen::MatrixXcd invSqrt =
        es.eigenvectors() *
        es.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
        es.eigenvectors().adjoint();
    return frame * invSqrt;
}

bool ColorAnchor::accepts(const AnchorProfile& profile, double minScore,
                          double minPhaseCoherence) {
    // An empty weightingId means no weighting was declared and no anchor was
    // measured, so the score is absent rather than zero.  The calibration
    // certificate must hold before either floor is consulted.
    return !profile.weightingId.empty() && profile.certificate.holds() &&
           profile.score >= minScore &&
           profile.phaseCoherence >= minPhaseCoherence;
}

AnchorGate ColorAnchor::gateFor(const AnchorProfile& profile, double minScore,
                                double minPhaseCoherence) {
    AnchorGate gate;
    gate.score = profile.score;
    gate.phaseCoherence = profile.phaseCoherence;
    gate.weightingId = profile.weightingId;
    gate.accepted = accepts(profile, minScore, minPhaseCoherence);
    if (gate.accepted) {
        gate.refusalReason.clear();
        return gate;
    }
    if (profile.weightingId.empty())
        gate.refusalReason = "triangle-anchor certificate absent";
    else if (!profile.certificate.holds())
        gate.refusalReason = "triangle-anchor certificate does not hold";
    else if (!(profile.score >= minScore))
        gate.refusalReason = "triangle-anchor atlas score below the floor";
    else
        gate.refusalReason =
            "triangle-anchor determinant-phase coherence below the floor";
    return gate;
}

AnchorProfile ColorAnchor::evaluate(const Eigen::MatrixXcd& frame,
                                    const Eigen::VectorXd& edgeWeights,
                                    double gramTolerance) {
    // The data are being examined: seal the declared weighting first.
    sealed_ = true;
    validateFrame(frame);
    if (edgeWeights.size() != frame.rows()) {
        throw std::invalid_argument(
            "ColorAnchor::evaluate: edgeWeights size (" +
            std::to_string(edgeWeights.size()) +
            ") must match the frame's edge rows (" +
            std::to_string(frame.rows()) + ")");
    }
    validateEdgeRange(triangles_, frame.rows());

    // Diagonal weights: the τ-oriented restricted block S W_τ S is again
    // diagonal (the signs cancel), so |W_τ|^{1/2} = diag(|w_e|^{1/2}) and
    // the Krein signature reads off the raw weight signs exactly, with no
    // per-triangle eigensolve on this path.
    std::vector<Eigen::Matrix3cd> sqrtBlocks;
    std::vector<std::array<int, 3>> signatures;
    sqrtBlocks.reserve(triangles_.size());
    signatures.reserve(triangles_.size());
    for (const auto& tri : triangles_) {
        Eigen::Matrix3cd sqrtBlock = Eigen::Matrix3cd::Zero();
        std::array<int, 3> sig{0, 0, 0};
        for (int k = 0; k < 3; ++k) {
            const double w =
                edgeWeights(tri.edges[static_cast<std::size_t>(k)]);
            sqrtBlock(k, k) = std::sqrt(std::abs(w));
            if (w > kreinTolerance()) {
                ++sig[0];
            } else if (w < -kreinTolerance()) {
                ++sig[2];
            } else {
                ++sig[1];
            }
        }
        sqrtBlocks.push_back(sqrtBlock);
        signatures.push_back(sig);
    }
    const Eigen::MatrixXcd gram =
        frame.adjoint() *
        edgeWeights.cwiseAbs().cast<std::complex<double>>().asDiagonal() *
        frame;
    return evaluateBlocks(frame, sqrtBlocks, signatures, gram,
                          gramTolerance, /*diagonalWeights=*/true);
}

AnchorProfile ColorAnchor::evaluate(const Eigen::MatrixXcd& frame,
                                    const Eigen::MatrixXcd& weight,
                                    double gramTolerance) {
    sealed_ = true;
    validateFrame(frame);
    if (weight.rows() != frame.rows() || weight.cols() != frame.rows()) {
        throw std::invalid_argument(
            "ColorAnchor::evaluate: the weight matrix must be square over "
            "the frame's edge rows");
    }
    const double scale = std::max(1.0, weight.cwiseAbs().maxCoeff());
    if ((weight - weight.adjoint()).cwiseAbs().maxCoeff() > 1e-12 * scale) {
        throw std::invalid_argument(
            "ColorAnchor::evaluate: the weight matrix must be Hermitian");
    }
    validateEdgeRange(triangles_, frame.rows());

    std::vector<Eigen::Matrix3cd> sqrtBlocks;
    std::vector<std::array<int, 3>> signatures;
    sqrtBlocks.reserve(triangles_.size());
    signatures.reserve(triangles_.size());
    for (const auto& tri : triangles_) {
        // The τ-oriented restricted block S_τ W_ττ S_τ = R_τ W R_τ†.
        Eigen::Matrix3cd block;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                block(r, c) =
                    static_cast<double>(
                        tri.signs[static_cast<std::size_t>(r)] *
                        tri.signs[static_cast<std::size_t>(c)]) *
                    weight(tri.edges[static_cast<std::size_t>(r)],
                           tri.edges[static_cast<std::size_t>(c)]);
            }
        }
        std::array<int, 3> sig{0, 0, 0};
        sqrtBlocks.push_back(modulusSqrt(block, kreinTolerance(), &sig));
        signatures.push_back(sig);
    }
    const Eigen::MatrixXcd gram =
        frame.adjoint() * matrixModulus(weight) * frame;
    return evaluateBlocks(frame, sqrtBlocks, signatures, gram,
                          gramTolerance, /*diagonalWeights=*/false);
}

AnchorProfile ColorAnchor::evaluateBlocks(
    const Eigen::MatrixXcd& frame,
    const std::vector<Eigen::Matrix3cd>& sqrtBlocks,
    const std::vector<std::array<int, 3>>& signatures,
    const Eigen::MatrixXcd& gram, double gramTolerance,
    bool diagonalWeights) {
    AnchorProfile profile;
    profile.weightingId = weightingId_;
    profile.weights = weights_;

    profile.frameGramResidual =
        (gram - Eigen::MatrixXcd::Identity(3, 3)).cwiseAbs().maxCoeff();
    if (profile.frameGramResidual > gramTolerance) {
        throw std::invalid_argument(
            "ColorAnchor::evaluate: the frame is not |W|-orthonormal "
            "(||Phi^dag|W|Phi - I||_max = " +
            std::to_string(profile.frameGramResidual) +
            " > " + std::to_string(gramTolerance) +
            "); use orthonormalizeFrame first — the calibration bound is "
            "undefined outside this domain");
    }

    const std::size_t nTri = triangles_.size();
    profile.terms.resize(nTri, 0.0);
    profile.detPhases.resize(
        nTri, std::numeric_limits<double>::quiet_NaN());
    profile.kreinSignatures.resize(nTri, {0, 0, 0});

    profile.overlappingTriangles = overlapCount_;

    double sumT = 0.0;
    double sumT2 = 0.0;
    double maxLambda = 0.0;
    std::complex<double> resultant(0.0, 0.0);
    double resultantWeight = 0.0;

    for (std::size_t t = 0; t < nTri; ++t) {
        const std::array<int, 3>& sig = signatures[t];
        profile.kreinSignatures[t] = sig;
        if (sig[1] != 0 || sig[2] != 0) {
            profile.positiveRegime = false;
        }

        const Eigen::Matrix3cd a =
            sqrtBlocks[t] * signedRestriction(frame, triangles_[t]);
        const std::complex<double> det = a.determinant();
        const double term = std::norm(det);
        profile.terms[t] = term;
        sumT += term;
        sumT2 += term * term;
        if (term > profile.maxTerm) {
            profile.maxTerm = term;
            profile.maxTermIndex = t;
        }
        profile.score += weights_[t] * term;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3cd> es(a.adjoint() * a);
        if (es.info() == Eigen::Success) {
            maxLambda = std::max(maxLambda, es.eigenvalues().maxCoeff());
        }

        if (det != std::complex<double>(0.0, 0.0)) {
            const double phase = std::arg(det);
            profile.detPhases[t] = phase;
            // The coherence measures agreement of the local determinant-line
            // trivializations where they overlap: only triangles sharing a
            // boundary edge with another declared triangle contribute.  An
            // isolated face has no overlap partner.
            if (overlapping_[t] != 0) {
                const double u = weights_[t] * term;
                resultant += u * std::complex<double>(std::cos(phase),
                                                      std::sin(phase));
                resultantWeight += u;
            }
        }
    }

    profile.participationRatio =
        (sumT2 > 0.0) ? (sumT * sumT) / sumT2 : 0.0;
    if (resultantWeight > 0.0) {
        profile.phaseCoherence = std::abs(resultant) / resultantWeight;
        profile.phaseDispersion = 1.0 - profile.phaseCoherence;
    } else {
        // No overlapping triangle carries a nonzero determinant, so there is
        // no overlap phase datum.  Unknown is reported as NaN, not zero; a
        // disjoint atlas lands here by construction.
        profile.phaseCoherence = std::numeric_limits<double>::quiet_NaN();
        profile.phaseDispersion = std::numeric_limits<double>::quiet_NaN();
    }
    profile.calibrationMargin = maxLambda - 1.0;

    // Attach the certification record.  The graded claim is the calibrated
    // score: closed-form given the verified |W|-orthonormal premise on a
    // diagonal weight, an eigen-modulus numerical evaluation on a general
    // Hermitian weight.
    using ::tessera::cobordism::Certificate;
    using ::tessera::cobordism::CertificateDomain;
    using ::tessera::cobordism::CertificateRegime;
    const CertificateRegime regime =
        profile.positiveRegime ? CertificateRegime::PositiveSemidefinite
                               : CertificateRegime::HermitianIndefinite;
    const double certResidual = std::max(
        profile.frameGramResidual, std::max(0.0, profile.calibrationMargin));
    profile.certificate =
        diagonalWeights
            ? Certificate::structureExact(CertificateDomain::Static, regime,
                                          certResidual,
                                          Certificate::kUnmeasured,
                                          gramTolerance)
            : Certificate::certifiedNumerical(CertificateDomain::Static,
                                              regime, certResidual,
                                              Certificate::kUnmeasured,
                                              gramTolerance);
    return profile;
}

}  // namespace tessera::observables
