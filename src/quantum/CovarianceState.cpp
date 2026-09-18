// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "quantum/CovarianceState.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "cobordism/AnalyticCache.h"
#include "mesh/Fingerprint.h"

namespace tessera::quantum {

namespace {

using cd = std::complex<double>;
using observables::Record;

constexpr int kSchemaVersion = 1;
constexpr const char* kRecordType = "covariance-state";

/// Regime-verification tolerance of the Wick-read certificates (the regime
/// is verified, never assumed).
constexpr double kRegimeTolerance = 1e-9;
/// Declared tolerance of the AlgebraicallyExact Wick-read claims.
constexpr double kReadTolerance = 1e-12;

double relativeHermiticityDefect(const Eigen::MatrixXcd& m) {
    const double scale = std::max(1.0, m.norm());
    return (m - m.adjoint()).norm() / scale;
}

std::vector<cd> matrixToFlat(const Eigen::MatrixXcd& m) {
    std::vector<cd> flat(static_cast<std::size_t>(m.rows()) *
                         static_cast<std::size_t>(m.cols()));
    for (Eigen::Index r = 0; r < m.rows(); ++r)
        for (Eigen::Index c = 0; c < m.cols(); ++c)
            flat[static_cast<std::size_t>(r) *
                     static_cast<std::size_t>(m.cols()) +
                 static_cast<std::size_t>(c)] = m(r, c);
    return flat;
}

std::vector<cd> complexListFromRecord(const Record::Map& m,
                                      const std::string& name) {
    const auto& re = m.at(name + "_re").asList();
    const auto& im = m.at(name + "_im").asList();
    if (re.size() != im.size())
        throw std::invalid_argument(
            "CovarianceState: complex list length mismatch");
    std::vector<cd> out(re.size());
    for (std::size_t i = 0; i < re.size(); ++i)
        out[i] = cd(re[i].asDouble(), im[i].asDouble());
    return out;
}

std::uint64_t chainHash(std::uint64_t seed, std::uint64_t value) {
    return mesh::Fingerprint::mix64(seed ^ value);
}

std::uint64_t stringFingerprint(const std::string& s, std::uint64_t seed) {
    std::uint64_t h = chainHash(seed, s.size());
    for (const char c : s)
        h = chainHash(h, static_cast<std::uint64_t>(
                             static_cast<unsigned char>(c)));
    return h;
}

}  // namespace

// ─── construction ─────────────────────────────────────────────────────────

CovarianceState::CovarianceState(Eigen::MatrixXcd gamma)
    : gamma_(std::move(gamma)) {
    if (gamma_.rows() != gamma_.cols())
        throw std::invalid_argument(
            "CovarianceState: the covariance matrix must be square");
}

CovarianceState CovarianceState::fromBandProjector(
    const Eigen::MatrixXcd& projector) {
    return CovarianceState(projector);
}

CovarianceState CovarianceState::fromOccupations(
    const Eigen::VectorXd& occupations) {
    Eigen::MatrixXcd gamma =
        Eigen::MatrixXcd::Zero(occupations.size(), occupations.size());
    for (Eigen::Index i = 0; i < occupations.size(); ++i)
        gamma(i, i) = cd(occupations(i), 0.0);
    return CovarianceState(std::move(gamma));
}

CovarianceState CovarianceState::fromSlaterFrame(
    const Eigen::MatrixXcd& orbitals, double rankTolerance) {
    if (orbitals.rows() == 0)
        throw std::invalid_argument(
            "CovarianceState: the Slater frame has no mode rows");
    if (orbitals.cols() == 0)  // no occupied orbitals: the vacuum, Γ = 0.
        return CovarianceState(
            Eigen::MatrixXcd::Zero(orbitals.rows(), orbitals.rows()));
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(orbitals, Eigen::ComputeThinU);
    const auto& sv = svd.singularValues();
    if (sv(sv.size() - 1) <= rankTolerance * sv(0))
        throw std::invalid_argument(
            "CovarianceState: the Slater frame is rank-deficient — occupied "
            "orbitals must be linearly independent");
    const Eigen::MatrixXcd u = svd.matrixU();
    return CovarianceState(u * u.adjoint());
}

// ─── Nambu shape ──────────────────────────────────────────────────────────

Eigen::MatrixXcd CovarianceState::pairing() const {
    return Eigen::MatrixXcd::Zero(gamma_.rows(), gamma_.cols());
}

Eigen::MatrixXcd CovarianceState::nambuCovariance() const {
    const Eigen::Index m = gamma_.rows();
    Eigen::MatrixXcd nambu = Eigen::MatrixXcd::Zero(2 * m, 2 * m);
    nambu.topLeftCorner(m, m) = gamma_;
    // Number-conserving: both anomalous blocks (F and −F̄) are zero.
    nambu.bottomRightCorner(m, m) =
        Eigen::MatrixXcd::Identity(m, m) - gamma_.transpose();
    return nambu;
}

// ─── state data ───────────────────────────────────────────────────────────

CovarianceState::Complex CovarianceState::occupation(std::size_t mode) const {
    if (mode >= modeCount())
        throw std::invalid_argument("CovarianceState: mode out of range");
    const auto i = static_cast<Eigen::Index>(mode);
    return gamma_(i, i);
}

Eigen::VectorXcd CovarianceState::occupations() const {
    return gamma_.diagonal();
}

CovarianceState::Complex CovarianceState::particleNumber() const {
    return gamma_.trace();
}

// ─── measured defects ─────────────────────────────────────────────────────

double CovarianceState::hermiticityDefect() const {
    if (hermiticityDefect_ < 0.0)
        hermiticityDefect_ = relativeHermiticityDefect(gamma_);
    return hermiticityDefect_;
}

double CovarianceState::purityDefect() const {
    if (purityDefect_ < 0.0) purityDefect_ = (gamma_ * gamma_ - gamma_).norm();
    return purityDefect_;
}

double CovarianceState::occupationSpectrumDefect() const {
    if (spectrumDefect_ < 0.0) {
        // Measured on the Hermitian part; the Hermiticity defect itself is
        // reported separately (never silently symmetrized away).
        const Eigen::MatrixXcd hermitianPart =
            0.5 * (gamma_ + gamma_.adjoint());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(hermitianPart,
                                                           false);
        double defect = 0.0;
        for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) {
            const double lambda = es.eigenvalues()(i);
            defect = std::max(defect, std::max(-lambda, lambda - 1.0));
        }
        spectrumDefect_ = std::max(0.0, defect);
    }
    return spectrumDefect_;
}

cobordism::Certificate CovarianceState::purityCertificate(
    double tolerance) const {
    const double herm = hermiticityDefect();
    const double residual = std::max(purityDefect(), herm);
    cobordism::CertificateRegime regime = cobordism::CertificateRegime::NonNormal;
    if (herm <= kRegimeTolerance)
        regime = occupationSpectrumDefect() <= kRegimeTolerance
                     ? cobordism::CertificateRegime::PositiveSemidefinite
                     : cobordism::CertificateRegime::HermitianIndefinite;
    return cobordism::Certificate::algebraicallyExact(
        cobordism::CertificateDomain::Static, regime, residual, tolerance);
}

std::uint64_t CovarianceState::matrixFingerprint(const Eigen::MatrixXcd& m,
                                                 std::uint64_t seed) {
    std::uint64_t h = chainHash(seed, static_cast<std::uint64_t>(m.rows()));
    h = chainHash(h, static_cast<std::uint64_t>(m.cols()));
    for (Eigen::Index r = 0; r < m.rows(); ++r) {
        for (Eigen::Index c = 0; c < m.cols(); ++c) {
            h = chainHash(h, std::bit_cast<std::uint64_t>(m(r, c).real()));
            h = chainHash(h, std::bit_cast<std::uint64_t>(m(r, c).imag()));
        }
    }
    return h;
}

std::string CovarianceState::covarianceHash() const {
    char buffer[17];
    std::snprintf(buffer, sizeof(buffer), "%016llx",
                  static_cast<unsigned long long>(
                      matrixFingerprint(gamma_, 0x9e3779b97f4a7c15ull)));
    return std::string(buffer);
}

void CovarianceState::invalidateDefects() noexcept {
    hermiticityDefect_ = -1.0;
    purityDefect_ = -1.0;
    spectrumDefect_ = -1.0;
}

// ─── propagation ──────────────────────────────────────────────────────────

Eigen::MatrixXcd CovarianceState::propagator(const Eigen::MatrixXcd& h,
                                             double dt,
                                             double hermitianTolerance) {
    if (h.rows() != h.cols())
        throw std::invalid_argument(
            "CovarianceState: the generator must be square");
    if (relativeHermiticityDefect(h) > hermitianTolerance)
        throw std::invalid_argument(
            "CovarianceState: the generator failed Hermiticity verification "
            "(iΓ̇ = [h, Γ] requires a Hermitian one-particle h)");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(h);
    if (es.info() != Eigen::Success)
        throw std::runtime_error(
            "CovarianceState: generator eigendecomposition failed");
    Eigen::VectorXcd phases(es.eigenvalues().size());
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i)
        phases(i) = std::exp(cd(0.0, -es.eigenvalues()(i) * dt));
    return es.eigenvectors() * phases.asDiagonal() *
           es.eigenvectors().adjoint();
}

void CovarianceState::evolve(const Eigen::MatrixXcd& h, double dt,
                             double hermitianTolerance) {
    if (h.rows() != gamma_.rows() || h.cols() != gamma_.cols())
        throw std::invalid_argument(
            "CovarianceState: generator shape does not match the mode count");
    applyTransport(propagator(h, dt, hermitianTolerance));
}

void CovarianceState::applyTransport(const Eigen::MatrixXcd& transport) {
    if (transport.rows() != gamma_.rows() || transport.cols() != gamma_.cols())
        throw std::invalid_argument(
            "CovarianceState: transport shape does not match the mode count");
    gamma_ = transport * gamma_ * transport.adjoint();
    invalidateDefects();
}

// ─── mean-field self-consistency ──────────────────────────────────────────

std::vector<MeanFieldStepRead> CovarianceState::meanFieldEvolve(
    const std::function<Eigen::MatrixXcd(const Eigen::MatrixXcd&)>& hamiltonian,
    double dt, std::size_t steps, double hermitianTolerance,
    double purityTolerance) {
    if (!hamiltonian)
        throw std::invalid_argument(
            "CovarianceState: the mean-field loop needs a generator callback");
    // The certificate path is chosen once, on entry: a pure Slater state
    // certifies purity; a mixed quasi-free state certifies the
    // covariance-spectrum constraint instead.
    const bool purePath = purityDefect() <= purityTolerance;
    std::vector<MeanFieldStepRead> reads;
    reads.reserve(steps);
    for (std::size_t step = 0; step < steps; ++step) {
        const Eigen::MatrixXcd h = hamiltonian(gamma_);
        if (h.rows() != gamma_.rows() || h.cols() != gamma_.cols())
            throw std::invalid_argument(
                "CovarianceState: mean-field generator shape does not match "
                "the mode count");
        MeanFieldStepRead read;
        read.step = step;
        read.time = static_cast<double>(step + 1) * dt;
        read.generatorHermiticityDefect = relativeHermiticityDefect(h);
        evolve(h, dt, hermitianTolerance);  // throws loudly on a bad h
        read.hermiticityDefect = hermiticityDefect();
        read.purityDefect = purityDefect();
        read.occupationSpectrumDefect = occupationSpectrumDefect();
        const double gaussianity =
            purePath ? read.purityDefect : read.occupationSpectrumDefect;
        const double residual =
            std::max({read.generatorHermiticityDefect, read.hermiticityDefect,
                      gaussianity});
        cobordism::CertificateRegime regime =
            cobordism::CertificateRegime::NonNormal;
        if (read.hermiticityDefect <= kRegimeTolerance)
            regime = read.occupationSpectrumDefect <= kRegimeTolerance
                         ? cobordism::CertificateRegime::PositiveSemidefinite
                         : cobordism::CertificateRegime::HermitianIndefinite;
        read.certificate = cobordism::Certificate::algebraicallyExact(
            cobordism::CertificateDomain::Static, regime, residual,
            purityTolerance);
        reads.push_back(std::move(read));
    }
    return reads;
}

// ─── Wick reads ───────────────────────────────────────────────────────────

WickCertificateRead CovarianceState::makeRead(Complex value,
                                              bool realByConstruction,
                                              std::string polynomialId) const {
    WickCertificateRead read;
    read.value = value;
    read.polynomialId = std::move(polynomialId);
    read.covarianceHash = covarianceHash();
    const double herm = hermiticityDefect();
    read.residual =
        realByConstruction ? std::max(herm, std::abs(value.imag())) : herm;
    cobordism::CertificateRegime regime = cobordism::CertificateRegime::NonNormal;
    if (herm <= kRegimeTolerance)
        regime = occupationSpectrumDefect() <= kRegimeTolerance
                     ? cobordism::CertificateRegime::PositiveSemidefinite
                     : cobordism::CertificateRegime::HermitianIndefinite;
    read.certificate = cobordism::Certificate::algebraicallyExact(
        cobordism::CertificateDomain::Static, regime, read.residual,
        kReadTolerance);
    return read;
}

WickCertificateRead CovarianceState::wickOccupation(std::size_t mode) const {
    return makeRead(occupation(mode), true,
                    "occupation[" + std::to_string(mode) + "]");
}

WickCertificateRead CovarianceState::wickTotalNumber() const {
    return makeRead(particleNumber(), true, "total-number");
}

WickCertificateRead CovarianceState::wickParity() const {
    const Eigen::MatrixXcd m =
        Eigen::MatrixXcd::Identity(gamma_.rows(), gamma_.cols()) -
        2.0 * gamma_;
    return makeRead(m.determinant(), true, "parity");
}

WickCertificateRead CovarianceState::wickSubsetParity(
    const std::vector<std::size_t>& modes) const {
    std::string id = "subset-parity[";
    for (std::size_t k = 0; k < modes.size(); ++k) {
        if (modes[k] >= modeCount())
            throw std::invalid_argument(
                "CovarianceState: subset-parity mode out of range");
        for (std::size_t l = 0; l < k; ++l)
            if (modes[l] == modes[k])
                throw std::invalid_argument(
                    "CovarianceState: subset-parity modes must be distinct");
        id += (k ? "," : "") + std::to_string(modes[k]);
    }
    id += "]";
    const auto p = static_cast<Eigen::Index>(modes.size());
    Eigen::MatrixXcd sub(p, p);
    for (Eigen::Index r = 0; r < p; ++r)
        for (Eigen::Index c = 0; c < p; ++c)
            sub(r, c) =
                (r == c ? cd(1.0, 0.0) : cd(0.0, 0.0)) -
                2.0 * gamma_(static_cast<Eigen::Index>(modes[r]),
                             static_cast<Eigen::Index>(modes[c]));
    return makeRead(p == 0 ? cd(1.0, 0.0) : sub.determinant(), true,
                    std::move(id));
}

WickCertificateRead CovarianceState::wickNormalOrdered(
    const std::vector<std::size_t>& creators,
    const std::vector<std::size_t>& annihilators) const {
    for (const std::size_t c : creators)
        if (c >= modeCount())
            throw std::invalid_argument(
                "CovarianceState: creator mode out of range");
    for (const std::size_t a : annihilators)
        if (a >= modeCount())
            throw std::invalid_argument(
                "CovarianceState: annihilator mode out of range");
    std::string id = "normal-ordered[c:";
    for (std::size_t k = 0; k < creators.size(); ++k)
        id += (k ? "," : "") + std::to_string(creators[k]);
    id += ";a:";
    for (std::size_t k = 0; k < annihilators.size(); ++k)
        id += (k ? "," : "") + std::to_string(annihilators[k]);
    id += "]";
    if (creators.size() != annihilators.size())
        // Number conservation: an unbalanced monomial is exactly zero.
        return makeRead(cd(0.0, 0.0), false, std::move(id));
    const auto p = static_cast<Eigen::Index>(creators.size());
    if (p == 0) return makeRead(cd(1.0, 0.0), false, std::move(id));
    // ⟨a†_{c_1}···a†_{c_p} a_{a_p}···a_{a_1}⟩ = det[Γ_{a_l c_k}]_{k,l}.
    Eigen::MatrixXcd m(p, p);
    for (Eigen::Index k = 0; k < p; ++k)
        for (Eigen::Index l = 0; l < p; ++l)
            m(k, l) = gamma_(
                static_cast<Eigen::Index>(annihilators[static_cast<std::size_t>(l)]),
                static_cast<Eigen::Index>(creators[static_cast<std::size_t>(k)]));
    return makeRead(m.determinant(), false, std::move(id));
}

WickCertificateRead CovarianceState::wickGramDeterminant(
    const Eigen::MatrixXcd& creatorFrame,
    const Eigen::MatrixXcd& annihilatorFrame) const {
    if (creatorFrame.rows() != gamma_.rows() ||
        annihilatorFrame.rows() != gamma_.rows())
        throw std::invalid_argument(
            "CovarianceState: smeared frames must have one row per mode");
    std::uint64_t fp = matrixFingerprint(creatorFrame, 0x5851f42d4c957f2dull);
    fp = matrixFingerprint(annihilatorFrame, fp);
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(fp));
    std::string id = "gram-determinant[p=" +
                     std::to_string(creatorFrame.cols()) + "," + hex + "]";
    if (creatorFrame.cols() != annihilatorFrame.cols())
        return makeRead(cd(0.0, 0.0), false, std::move(id));
    if (creatorFrame.cols() == 0)
        return makeRead(cd(1.0, 0.0), false, std::move(id));
    // ⟨a†(v_1)···a†(v_p) a(w_p)···a(w_1)⟩ = det(W† Γ V).
    const Eigen::MatrixXcd m =
        annihilatorFrame.adjoint() * gamma_ * creatorFrame;
    return makeRead(m.determinant(), false, std::move(id));
}

WickCertificateRead CovarianceState::wickColorWedgeSquared(
    const Eigen::MatrixXcd& colorColumns) const {
    if (colorColumns.rows() != gamma_.rows() || colorColumns.cols() != 3)
        throw std::invalid_argument(
            "CovarianceState: the color wedge takes exactly three color "
            "columns over the modes");
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(matrixFingerprint(
                      colorColumns, 0x2545f4914f6cdd1dull)));
    // |S_ABC|² = det(C† Γ C): Hermitian form of a Hermitian Γ — real by
    // construction.
    const Eigen::MatrixXcd m =
        colorColumns.adjoint() * gamma_ * colorColumns;
    return makeRead(m.determinant(), true,
                    std::string("color-wedge-squared[") + hex + "]");
}

CovarianceState::Complex CovarianceState::bilinearMoment(
    const std::vector<Eigen::MatrixXcd>& factors) const {
    const std::size_t n = factors.size();
    const std::size_t full = (std::size_t{1} << n) - 1;
    // B_S = A_S Γ with A_S = ∏_{k∈S ascending} A_k, built by the lowest-bit
    // recurrence A_S = A_{k0} A_{S\k0} (blocks[rest] already ends in Γ, so
    // left-multiplying by A_{k0} preserves the single trailing Γ).
    std::vector<Eigen::MatrixXcd> blocks(full + 1);
    for (std::size_t s = 1; s <= full; ++s) {
        const auto low = static_cast<std::size_t>(std::countr_zero(s));
        const std::size_t rest = s & (s - 1);  // s without its lowest bit
        blocks[s] = rest == 0
                        ? Eigen::MatrixXcd(factors[low] * gamma_)
                        : Eigen::MatrixXcd(factors[low] * blocks[rest]);
    }
    // c[T] = Σ_j (−1)^{j+1}/j · tr(W_j[T]) with
    // W_1[R] = B_R, W_j[R] = Σ_{∅≠S⊊R} B_S W_{j−1}[R\S].
    std::vector<cd> cyclic(full + 1, cd(0.0, 0.0));
    std::vector<Eigen::MatrixXcd> layer(full + 1);
    for (std::size_t r = 1; r <= full; ++r) {
        layer[r] = blocks[r];
        cyclic[r] += layer[r].trace();
    }
    for (std::size_t j = 2; j <= n; ++j) {
        std::vector<Eigen::MatrixXcd> next(full + 1);
        for (std::size_t r = 1; r <= full; ++r) {
            if (static_cast<std::size_t>(std::popcount(r)) < j) continue;
            Eigen::MatrixXcd acc =
                Eigen::MatrixXcd::Zero(gamma_.rows(), gamma_.cols());
            // Enumerate nonempty proper submasks s of r (the first part).
            for (std::size_t s = (r - 1) & r; s != 0; s = (s - 1) & r) {
                const std::size_t tail = r ^ s;
                if (static_cast<std::size_t>(std::popcount(tail)) < j - 1)
                    continue;
                if (layer[tail].size() == 0) continue;
                acc.noalias() += blocks[s] * layer[tail];
            }
            next[r] = std::move(acc);
            const double sign = (j % 2 == 1) ? 1.0 : -1.0;
            cyclic[r] += sign / static_cast<double>(j) * next[r].trace();
        }
        layer = std::move(next);
    }
    // Moment = Σ over set partitions ∏ c[T]: subset DP pinning the lowest
    // element of the remaining set into its block.
    std::vector<cd> partial(full + 1, cd(0.0, 0.0));
    partial[0] = cd(1.0, 0.0);
    for (std::size_t r = 1; r <= full; ++r) {
        const std::size_t lowBit = r & (~r + 1);
        const std::size_t rest = r ^ lowBit;
        cd acc(0.0, 0.0);
        // Blocks T containing lowBit: T = lowBit | u for submasks u of rest.
        std::size_t u = rest;
        while (true) {
            const std::size_t t = lowBit | u;
            acc += cyclic[t] * partial[r ^ t];
            if (u == 0) break;
            u = (u - 1) & rest;
        }
        partial[r] = acc;
    }
    return partial[full];
}

WickCertificateRead CovarianceState::wickBilinearMoment(
    const std::vector<Eigen::MatrixXcd>& oneParticleFactors) const {
    if (oneParticleFactors.empty())
        throw std::invalid_argument(
            "CovarianceState: the bilinear moment needs at least one factor");
    if (oneParticleFactors.size() > kMaxBilinearFactors)
        throw std::invalid_argument(
            "CovarianceState: the ordered-composition expansion is capped at "
            "kMaxBilinearFactors bilinear factors");
    std::uint64_t fp = 0x9e3779b97f4a7c15ull;
    for (const Eigen::MatrixXcd& a : oneParticleFactors) {
        if (a.rows() != gamma_.rows() || a.cols() != gamma_.cols())
            throw std::invalid_argument(
                "CovarianceState: bilinear factor shape does not match the "
                "mode count");
        fp = matrixFingerprint(a, fp);
    }
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(fp));
    return makeRead(bilinearMoment(oneParticleFactors), false,
                    "bilinear-moment[n=" +
                        std::to_string(oneParticleFactors.size()) + "," + hex +
                        "]");
}

WickCertificateRead CovarianceState::wickSpinSquaredExpectation(
    const Eigen::MatrixXcd& jx, const Eigen::MatrixXcd& jy,
    const Eigen::MatrixXcd& jz) const {
    std::uint64_t fp = 0x94d049bb133111ebull;
    for (const Eigen::MatrixXcd* j : {&jx, &jy, &jz}) {
        if (j->rows() != gamma_.rows() || j->cols() != gamma_.cols())
            throw std::invalid_argument(
                "CovarianceState: spin matrix shape does not match the mode "
                "count");
        fp = matrixFingerprint(*j, fp);
    }
    cd value(0.0, 0.0);
    for (const Eigen::MatrixXcd* j : {&jx, &jy, &jz})
        value += bilinearMoment({*j, *j});
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(fp));
    return makeRead(value, true,
                    std::string("spin-squared-expectation[") + hex + "]");
}

WickCertificateRead CovarianceState::wickSpinSquaredVariance(
    const Eigen::MatrixXcd& jx, const Eigen::MatrixXcd& jy,
    const Eigen::MatrixXcd& jz) const {
    std::uint64_t fp = 0xbf58476d1ce4e5b9ull;
    for (const Eigen::MatrixXcd* j : {&jx, &jy, &jz}) {
        if (j->rows() != gamma_.rows() || j->cols() != gamma_.cols())
            throw std::invalid_argument(
                "CovarianceState: spin matrix shape does not match the mode "
                "count");
        fp = matrixFingerprint(*j, fp);
    }
    const std::array<const Eigen::MatrixXcd*, 3> js = {&jx, &jy, &jz};
    cd second(0.0, 0.0);
    for (const Eigen::MatrixXcd* j : js) second += bilinearMoment({*j, *j});
    cd fourth(0.0, 0.0);
    for (const Eigen::MatrixXcd* ja : js)
        for (const Eigen::MatrixXcd* jb : js)
            fourth += bilinearMoment({*ja, *ja, *jb, *jb});
    const cd value = fourth - second * second;
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(fp));
    return makeRead(value, true,
                    std::string("spin-squared-variance[") + hex + "]");
}

// ─── cached Wick reads (the AnalyticCache contract) ────────────────────────────────────

WickCertificateRead CovarianceState::wickReadCached(
    cobordism::AnalyticCache& cache,
    const std::vector<std::uint64_t>& componentVertexIds,
    const std::string& polynomialId,
    const std::function<WickCertificateRead()>& compute) const {
    if (!compute)
        throw std::invalid_argument(
            "CovarianceState: the cached Wick read needs a compute callback");
    const std::string hash = covarianceHash();
    const std::uint64_t key = stringFingerprint(
        polynomialId, matrixFingerprint(gamma_, 0x9e3779b97f4a7c15ull));
    const auto parameter = static_cast<std::int64_t>(key);
    if (const auto cached =
            cache.fetch(componentVertexIds, kCacheKind, parameter)) {
        const auto read =
            std::static_pointer_cast<const WickCertificateRead>(cached);
        // A Γ change is a state change, not a geometry change: verified
        // explicitly, so a stale-Γ payload causes recomputation, never a
        // wrong serve.
        if (read->polynomialId == polynomialId && read->covarianceHash == hash)
            return *read;
    }
    WickCertificateRead fresh = compute();
    cache.store(componentVertexIds, kCacheKind, parameter,
                std::make_shared<WickCertificateRead>(fresh),
                fresh.certificate);
    return fresh;
}

// ─── checkpoint serialization ─────────────────────────────────────────────

Record CovarianceState::toRecord() const {
    Record::Map m;
    m["schema_version"] = Record(kSchemaVersion);
    m["record_type"] = Record(kRecordType);
    m["mode_count"] = Record(static_cast<std::int64_t>(modeCount()));
    m["number_conserving"] = Record(numberConserving());
    Record::splitComplex(m, "gamma", matrixToFlat(gamma_));
    // Informational channels — recomputed (never trusted) on load.
    m["hermiticity_defect"] = Record(hermiticityDefect());
    m["purity_defect"] = Record(purityDefect());
    m["occupation_spectrum_defect"] = Record(occupationSpectrumDefect());
    m["covariance_hash"] = Record(covarianceHash());
    return Record(std::move(m));
}

CovarianceState CovarianceState::fromRecord(const Record& record) {
    const Record::Map& m = record.asMap();
    const auto version = m.find("schema_version");
    if (version == m.end() ||
        version->second.asInt() != static_cast<std::int64_t>(kSchemaVersion))
        throw std::invalid_argument(
            "CovarianceState: unknown schema_version (reader rejects unknown "
            "checkpoint schemas)");
    const auto type = m.find("record_type");
    if (type == m.end() || type->second.asString() != kRecordType)
        throw std::invalid_argument(
            "CovarianceState: expected a 'covariance-state' record");
    const auto modes =
        static_cast<std::size_t>(m.at("mode_count").asInt());
    const std::vector<cd> flat = complexListFromRecord(m, "gamma");
    if (flat.size() != modes * modes)
        throw std::invalid_argument(
            "CovarianceState: gamma payload size mismatch");
    Eigen::MatrixXcd gamma(static_cast<Eigen::Index>(modes),
                           static_cast<Eigen::Index>(modes));
    for (std::size_t r = 0; r < modes; ++r)
        for (std::size_t c = 0; c < modes; ++c)
            gamma(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                flat[r * modes + c];
    return CovarianceState(std::move(gamma));
}

}  // namespace tessera::quantum
