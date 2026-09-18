// Implementation of the emergent-spectral-dimension submodule. See
// include/quantum/Holography.hpp for the architectural overview.

#include "quantum/Holography.hpp"

#include "graph/CSRBuilder.hpp"
#include "quantum/ChoiState.hpp"
#include "quantum/TDVPRunner.hpp"

#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

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

namespace detail {

// Minimal JSON formatting. The emitted schema uses only scalar and
// array primitives, so a small hand-rolled writer avoids pulling in
// nlohmann_json or rapidjson.
void writeArray(std::ostringstream& os,
                 std::vector<double> const& xs) {
    os << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ", ";
        os << xs[i];
    }
    os << "]";
}

void writeArrayInt(std::ostringstream& os,
                    std::vector<int> const& xs) {
    os << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ", ";
        os << xs[i];
    }
    os << "]";
}

std::string
serialiseResultToJson(SpectralDimensionResult const& r,
                       HolographyConfig const& cfg) {
    std::ostringstream os;
    os << std::scientific;
    os.precision(12);

    os << "{\n";
    // ── config block ────────────────────────────────────────────────
    os << "  \"config\": {\n";
    os << "    \"tdvp\": {\n";
    os << "      \"N\":              " << cfg.tdvp.N             << ",\n";
    os << "      \"a\":              " << cfg.tdvp.a             << ",\n";
    os << "      \"m\":              " << cfg.tdvp.m             << ",\n";
    os << "      \"g\":              " << cfg.tdvp.g             << ",\n";
    os << "      \"L0\":             " << cfg.tdvp.L0            << ",\n";
    os << "      \"i0\":             " << cfg.tdvp.i0            << ",\n";
    os << "      \"d\":              " << cfg.tdvp.d             << ",\n";
    os << "      \"dt\":             " << cfg.tdvp.dt            << ",\n";
    os << "      \"T\":              " << cfg.tdvp.T             << ",\n";
    os << "      \"snapshotEvery\":  " << cfg.tdvp.snapshotEvery << ",\n";
    os << "      \"maxBondDim\":     " << cfg.tdvp.maxBondDim    << ",\n";
    os << "      \"cutoff\":         " << cfg.tdvp.cutoff        << "\n";
    os << "    },\n";
    os << "    \"sigmaMin\":          " << cfg.sigmaMin          << ",\n";
    os << "    \"sigmaMax\":          " << cfg.sigmaMax          << ",\n";
    os << "    \"sigmaCount\":        " << cfg.sigmaCount        << ",\n";
    os << "    \"epsilonI\":          " << cfg.epsilonI          << ",\n";
    os << "    \"includeTemporal\":   "
       << (cfg.includeTemporal ? "true" : "false")               << ",\n";
    os << "    \"maxTemporalStride\": " << cfg.maxTemporalStride << ",\n";
    os << "    \"krylovDim\":         " << cfg.krylovDim         << ",\n";
    os << "    \"seed\":              " << cfg.seed              << "\n";
    os << "  },\n";

    // ── tdvp_summary ────────────────────────────────────────────────
    os << "  \"tdvp_summary\": {\n";
    os << "    \"snapshot_times\": "; writeArray(os, r.snapshotTimes); os << ",\n";
    os << "    \"bond_dims\":      "; writeArrayInt(os, r.snapshotBondDims); os << ",\n";
    os << "    \"energies\":       "; writeArray(os, r.snapshotEnergies); os << "\n";
    os << "  },\n";

    // ── graph diagnostics ───────────────────────────────────────────
    os << "  \"graph\": {\n";
    os << "    \"n_vertices\":   " << r.graphNVertices << ",\n";
    os << "    \"n_edges\":      " << r.graphNEdges    << ",\n";
    const double density = r.graphNVertices > 1
        ? (2.0 * r.graphNEdges) /
          (static_cast<double>(r.graphNVertices) * (r.graphNVertices - 1))
        : 0.0;
    os << "    \"edge_density\": " << density << "\n";
    os << "  },\n";

    // ── spectral dimension ──────────────────────────────────────────
    os << "  \"spectral_dimension\": {\n";
    os << "    \"sigmas\":      "; writeArray(os, r.sigmas);     os << ",\n";
    os << "    \"P\":           "; writeArray(os, r.P);          os << ",\n";
    os << "    \"D_S\":         "; writeArray(os, r.dS);         os << ",\n";
    os << "    \"D_S_smoothed\":"; writeArray(os, r.dSSmoothed); os << ",\n";
    os << "    \"fit\": {\n";
    os << "      \"D_infinity\":   " << r.dInfinity     << ",\n";
    os << "      \"C\":            " << r.C             << ",\n";
    os << "      \"B\":            " << r.B             << ",\n";
    os << "      \"chi_squared\":  " << r.fitChiSquared << "\n";
    os << "    }\n";
    os << "  },\n";

    // ── provenance ──────────────────────────────────────────────────
    os << "  \"provenance\": {\n";
#ifdef TESSERA_VERSION
    os << "    \"tessera_version\": \"" << TESSERA_VERSION << "\",\n";
#else
    os << "    \"tessera_version\": \"unknown\",\n";
#endif
    os << "    \"seed\":            " << cfg.seed << "\n";
    os << "  }\n";

    os << "}\n";
    return os.str();
}

} // namespace detail

// ─── HolographyConfig ────────────────────────────────────────────────

void HolographyConfig::validate() const {
    if (sigmaMin <= 0.0) {
        throw std::invalid_argument(
            "HolographyConfig: sigmaMin must be > 0");
    }
    if (sigmaMax <= sigmaMin) {
        throw std::invalid_argument(
            "HolographyConfig: sigmaMax must be > sigmaMin");
    }
    if (sigmaCount < 8) {
        throw std::invalid_argument(
            "HolographyConfig: sigmaCount must be >= 8 for finite-difference D_S");
    }
    if (epsilonI < 0.0) {
        throw std::invalid_argument(
            "HolographyConfig: epsilonI must be non-negative");
    }
    if (maxTemporalStride < 0) {
        throw std::invalid_argument(
            "HolographyConfig: maxTemporalStride must be >= 0");
    }
    if (krylovDim < 4) {
        throw std::invalid_argument(
            "HolographyConfig: krylovDim must be >= 4");
    }
}

// ─── MutualInformationProfile ────────────────────────────────────────

MutualInformationProfile::MutualInformationProfile(
    std::vector<TDVPSnapshot> const& snapshots,
    HolographyConfig const& config)
    : nSnapshots_(static_cast<int>(snapshots.size())),
      epsilonI_(config.epsilonI),
      vertexIds_(config.vertexIds) {
    if (nSnapshots_ == 0) {
        throw std::invalid_argument(
            "MutualInformationProfile: snapshot list is empty");
    }
    if (snapshots.front().mutualInformation.empty()) {
        throw std::invalid_argument(
            "MutualInformationProfile: snapshots have no recorded MI; "
            "set TDVPConfig::recordMutualInformation = true");
    }

    // Derive N from the size of the first snapshot's MI matrix.
    const auto miFlat0 = snapshots.front().mutualInformation.size();
    int N = 0;
    while (static_cast<std::size_t>(N) * N < miFlat0) ++N;
    if (static_cast<std::size_t>(N) * N != miFlat0) {
        throw std::invalid_argument(
            "MutualInformationProfile: snapshot MI buffer is not N×N");
    }
    nSites_ = N;

    if (!vertexIds_.empty() &&
        static_cast<int>(vertexIds_.size()) != nSites_) {
        throw std::invalid_argument(
            "MutualInformationProfile: config.vertexIds length must "
            "equal nSites or be empty");
    }

    const int nLabels = nSites_ * nSnapshots_;
    mi_.assign(static_cast<std::size_t>(nLabels) * nLabels, 0.0);

    // Copy per-snapshot N×N spatial-MI blocks onto the diagonal of
    // the global (nLabels × nLabels) matrix.
    for (int s = 0; s < nSnapshots_; ++s) {
        auto const& snap = snapshots[static_cast<std::size_t>(s)];
        if (static_cast<int>(snap.mutualInformation.size()) != N * N) {
            throw std::invalid_argument(
                "MutualInformationProfile: snapshot MI size inconsistent across snapshots");
        }
        const int base = s * nSites_;
        for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
                const std::size_t srcIdx =
                    static_cast<std::size_t>(a * N + b);
                const std::size_t dstIdx =
                    static_cast<std::size_t>((base + a) * nLabels + (base + b));
                mi_[dstIdx] = snap.mutualInformation[srcIdx];
            }
        }
    }

    // Optional: populate cross-snapshot temporal MI blocks via the
    // Choi-state propagator. Because the Schwinger Hamiltonian is
    // time-independent, U_{s→t} depends only on the duration |t − s|;
    // we compute one Choi state per unique snapshot stride.
    if (!config.includeTemporal) return;
    if (nSnapshots_ < 2) return;

    SchwingerParams p;
    p.N = config.tdvp.N;
    p.a = config.tdvp.a;
    p.m = config.tdvp.m;
    p.g = config.tdvp.g;
    p.L0 = config.tdvp.L0;
    if (p.N != N) {
        // The snapshot MI matrices and the config N must agree — they
        // come from the same TDVP run.
        throw std::invalid_argument(
            "MutualInformationProfile: snapshot N inconsistent with config.tdvp.N");
    }

    const double dtSnap =
        config.tdvp.dt * static_cast<double>(std::max(config.tdvp.snapshotEvery, 1));

    ChoiPropagator::TDVPSettings settings;
    settings.dt         = config.tdvp.dt;
    settings.maxBondDim = config.tdvp.maxBondDim;
    settings.krylovDim  = config.tdvp.krylovDim;
    settings.cutoff     = config.tdvp.cutoff;
    settings.quiet      = config.tdvp.quiet;

    const int strideCap =
        (config.maxTemporalStride <= 0)
            ? (nSnapshots_ - 1)
            : std::min(config.maxTemporalStride, nSnapshots_ - 1);

    // For each unique stride compute the Choi state once, extract the
    // N×N temporal MI matrix, and fan it out over every (s, s+stride)
    // snapshot pair. Store symmetrically so the global MI matrix stays
    // Hermitian.
    for (int stride = 1; stride <= strideCap; ++stride) {
        const double duration = stride * dtSnap;
        auto choi = ChoiPropagator::choiState(p, duration, settings);
        auto miMat = ChoiPropagator::temporalMutualInformation(choi, p.N);
        for (int s = 0; s + stride < nSnapshots_; ++s) {
            const int t = s + stride;
            const int baseS = s * nSites_;
            const int baseT = t * nSites_;
            for (int a = 0; a < N; ++a) {
                for (int b = 0; b < N; ++b) {
                    const double mi = miMat(a, b);
                    // Forward direction (s, a) → (t, b)
                    const std::size_t fwd =
                        static_cast<std::size_t>((baseS + a) * nLabels +
                                                  (baseT + b));
                    // Symmetric storage (t, b) → (s, a)
                    const std::size_t bwd =
                        static_cast<std::size_t>((baseT + b) * nLabels +
                                                  (baseS + a));
                    mi_[fwd] = mi;
                    mi_[bwd] = mi;
                }
            }
        }
    }
}

double MutualInformationProfile::at(int siteV, int snapV,
                                     int siteW, int snapW) const {
    const int v = snapV * nSites_ + siteV;
    const int w = snapW * nSites_ + siteW;
    return atFlat(v, w);
}

double MutualInformationProfile::atFlat(int v, int w) const noexcept {
    const int n = nLabels();
    if (v < 0 || w < 0 || v >= n || w >= n) return 0.0;
    return mi_[static_cast<std::size_t>(v) * n + w];
}

::tessera::graph::WeightedCOO<int, double>
MutualInformationProfile::weightedAdjacency() const {
    ::tessera::graph::WeightedCOO<int, double> coo;
    const int n = nLabels();
    coo.n = n;
    if (n == 0) return coo;

    // Heuristic reservation: each snapshot contributes at most
    // N(N-1)/2 distinct edges, and each edge is listed twice.
    coo.rows.reserve(static_cast<std::size_t>(n * nSites_));
    coo.cols.reserve(static_cast<std::size_t>(n * nSites_));
    coo.weights.reserve(static_cast<std::size_t>(n * nSites_));

    for (int v = 0; v < n; ++v) {
        for (int w = 0; w < n; ++w) {
            if (v == w) continue;
            const double wt =
                mi_[static_cast<std::size_t>(v) * n + w];
            if (wt > epsilonI_) {
                coo.rows.push_back(v);
                coo.cols.push_back(w);
                coo.weights.push_back(wt);
            }
        }
    }
    return coo;
}

// ─── EmergentGraph ───────────────────────────────────────────────────

EmergentGraph::EmergentGraph(MutualInformationProfile const& profile) {
    auto coo = profile.weightedAdjacency();
    // coo lists each undirected edge twice (v→w and w→v) already.
    buildFromCOO_(profile.nLabels(), coo.rows, coo.cols, coo.weights);
    nEdges_ = static_cast<int>(coo.rows.size()) / 2;
}

EmergentGraph
EmergentGraph::fromWeightedEdges(
    int n,
    std::vector<std::tuple<int, int, double>> const& edges) {
    std::vector<int>    rows;
    std::vector<int>    cols;
    std::vector<double> weights;
    rows.reserve(edges.size() * 2);
    cols.reserve(edges.size() * 2);
    weights.reserve(edges.size() * 2);
    for (auto const& [u, v, w] : edges) {
        if (u < 0 || v < 0 || u >= n || v >= n || u == v) {
            throw std::invalid_argument(
                "EmergentGraph::fromWeightedEdges: edge endpoints out of "
                "range or self-loop");
        }
        rows.push_back(u); cols.push_back(v); weights.push_back(w);
        rows.push_back(v); cols.push_back(u); weights.push_back(w);
    }
    EmergentGraph g;
    g.buildFromCOO_(n, rows, cols, weights);
    g.nEdges_ = static_cast<int>(edges.size());
    return g;
}

void EmergentGraph::buildFromCOO_(int n,
                                    std::vector<int> const& rows,
                                    std::vector<int> const& cols,
                                    std::vector<double> const& weights) {
    n_ = n;
    ::tessera::graph::buildCSRFromCOO<int, double, int>(
        static_cast<std::size_t>(n_), rows, cols, weights,
        indptr_, indices_, weights_);

    degrees_.assign(static_cast<std::size_t>(n_), 0.0);
    for (int v = 0; v < n_; ++v) {
        const int lo = indptr_[static_cast<std::size_t>(v)];
        const int hi = indptr_[static_cast<std::size_t>(v) + 1];
        double d = 0.0;
        for (int k = lo; k < hi; ++k) {
            d += weights_[static_cast<std::size_t>(k)];
        }
        degrees_[static_cast<std::size_t>(v)] = d;
    }
}

Eigen::SparseMatrix<double> EmergentGraph::laplacian() const {
    using Trip = Eigen::Triplet<double>;
    std::vector<Trip> trips;
    trips.reserve(static_cast<std::size_t>(2 * nEdges_ + n_));
    for (int v = 0; v < n_; ++v) {
        trips.emplace_back(v, v, degrees_[static_cast<std::size_t>(v)]);
        const int lo = indptr_[static_cast<std::size_t>(v)];
        const int hi = indptr_[static_cast<std::size_t>(v) + 1];
        for (int k = lo; k < hi; ++k) {
            trips.emplace_back(v, indices_[static_cast<std::size_t>(k)],
                                -weights_[static_cast<std::size_t>(k)]);
        }
    }
    Eigen::SparseMatrix<double> L(n_, n_);
    L.setFromTriplets(trips.begin(), trips.end());
    L.makeCompressed();
    return L;
}

std::string EmergentGraph::toGraphML() const {
    std::ostringstream os;
    os << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    os << "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n";
    os << "  <key id=\"w\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n";
    os << "  <graph id=\"emergent\" edgedefault=\"undirected\">\n";
    for (int v = 0; v < n_; ++v) {
        os << "    <node id=\"n" << v << "\"/>\n";
    }
    int eId = 0;
    for (int v = 0; v < n_; ++v) {
        const int lo = indptr_[static_cast<std::size_t>(v)];
        const int hi = indptr_[static_cast<std::size_t>(v) + 1];
        for (int k = lo; k < hi; ++k) {
            const int w = indices_[static_cast<std::size_t>(k)];
            if (w <= v) continue;  // each undirected edge once
            os << "    <edge id=\"e" << eId++
               << "\" source=\"n" << v
               << "\" target=\"n" << w << "\">\n";
            os << std::scientific << std::setprecision(8)
               << "      <data key=\"w\">"
               << weights_[static_cast<std::size_t>(k)] << "</data>\n";
            os << "    </edge>\n";
        }
    }
    os << "  </graph>\n";
    os << "</graphml>\n";
    return os.str();
}

std::string EmergentGraph::toDot() const {
    std::ostringstream os;
    os << "graph emergent {\n";
    os << "  // Weighted graph: edge label = mutual information (nats)\n";
    for (int v = 0; v < n_; ++v) {
        os << "  " << v << ";\n";
    }
    for (int v = 0; v < n_; ++v) {
        const int lo = indptr_[static_cast<std::size_t>(v)];
        const int hi = indptr_[static_cast<std::size_t>(v) + 1];
        for (int k = lo; k < hi; ++k) {
            const int w = indices_[static_cast<std::size_t>(k)];
            if (w <= v) continue;  // each undirected edge listed once
            os << std::fixed << std::setprecision(3)
               << "  " << v << " -- " << w
               << " [label=\"" << weights_[static_cast<std::size_t>(k)] << "\"];\n";
        }
    }
    os << "}\n";
    return os.str();
}

// ─── AmbjornLollFit ──────────────────────────────────────────────────

AmbjornLollFit::Result
AmbjornLollFit::fit(std::vector<double> const& sigmas,
                      std::vector<double> const& dS,
                      double sigmaFitMin,
                      double sigmaFitMax) {
    Result r;
    const int nAll = static_cast<int>(sigmas.size());
    if (nAll < 4 || static_cast<int>(dS.size()) != nAll) return r;

    // Filter to the chosen window and drop NaN points.
    std::vector<double> sFit, yFit;
    sFit.reserve(static_cast<std::size_t>(nAll));
    yFit.reserve(static_cast<std::size_t>(nAll));
    for (int i = 0; i < nAll; ++i) {
        const double s = sigmas[static_cast<std::size_t>(i)];
        const double y = dS[static_cast<std::size_t>(i)];
        if (!std::isfinite(s) || !std::isfinite(y)) continue;
        if (sigmaFitMin > 0.0 && s < sigmaFitMin) continue;
        if (sigmaFitMax > 0.0 && s > sigmaFitMax) continue;
        sFit.push_back(s);
        yFit.push_back(y);
    }
    const int n = static_cast<int>(sFit.size());
    if (n < 4) return r;

    // Initial guesses: D_∞ = max(y), the large-σ asymptote; B = σ at the
    // median point; C chosen so the curve passes through that point.
    double dInf = *std::max_element(yFit.begin(), yFit.end());
    double B    = sFit[static_cast<std::size_t>(n / 2)];
    double yMid = yFit[static_cast<std::size_t>(n / 2)];
    double C    = std::max(0.0, (dInf - yMid) * (B + sFit[static_cast<std::size_t>(n / 2)]));

    // Gauss-Newton iteration on the residual r_i = D_∞ - C/(B+σ_i) - y_i.
    // Jacobian:
    //   ∂r_i/∂D_∞ = 1
    //   ∂r_i/∂C   = -1 / (B + σ_i)
    //   ∂r_i/∂B   = C / (B + σ_i)²
    constexpr int maxIter = 100;
    constexpr double tol  = 1e-10;
    for (int iter = 0; iter < maxIter; ++iter) {
        Eigen::MatrixXd J(n, 3);
        Eigen::VectorXd resid(n);
        for (int i = 0; i < n; ++i) {
            const double s = sFit[static_cast<std::size_t>(i)];
            const double inv = 1.0 / (B + s);
            const double f = dInf - C * inv;
            resid(i) = yFit[static_cast<std::size_t>(i)] - f;
            J(i, 0) =  1.0;
            J(i, 1) = -inv;
            J(i, 2) =  C * inv * inv;
        }
        Eigen::Matrix3d JtJ = J.transpose() * J;
        Eigen::Vector3d Jtr = J.transpose() * resid;
        // Levenberg-Marquardt damping for stability.
        JtJ += 1e-8 * Eigen::Matrix3d::Identity();
        Eigen::Vector3d delta = JtJ.ldlt().solve(Jtr);
        dInf += delta(0);
        C    += delta(1);
        B    += delta(2);
        if (delta.norm() < tol * (std::abs(dInf) + std::abs(C) + std::abs(B) + 1.0)) {
            break;
        }
    }

    // Reduced χ² assuming unit variance per point.
    double chi2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double s = sFit[static_cast<std::size_t>(i)];
        const double f = dInf - C / (B + s);
        const double e = yFit[static_cast<std::size_t>(i)] - f;
        chi2 += e * e;
    }
    chi2 /= std::max(1, n - 3);

    r.dInfinity  = dInf;
    r.C          = C;
    r.B          = B;
    r.chiSquared = chi2;
    return r;
}

// ─── EmergentSpectralDimension ───────────────────────────────────────

EmergentSpectralDimension::EmergentSpectralDimension(HolographyConfig config)
    : config_(std::move(config)) {
    config_.validate();
    // The pipeline needs all-pairs MI per snapshot, so the flag is
    // forced on here rather than left to the caller.
    config_.tdvp.recordMutualInformation = true;
}

SpectralDimensionResult
EmergentSpectralDimension::computeFromSnapshots(QuenchResult const& quench) const {
    SpectralDimensionResult result;

    // (1) Build the MI profile on the (site, snapshot) label set.
    //
    // D_S is measured here on the boundary MI graph directly, not on a
    // reconstructed bulk. For a physical-bulk D_S, use the
    // interaction-history pipeline
    // (InteractionSimulation::getSpectralDimension), which routes
    // through Spacetime::getSpectralDimensionOnSkeleton.
    MutualInformationProfile profile(quench.snapshots, config_);
    EmergentGraph graph(profile);

    result.graphNVertices = graph.nVertices();
    result.graphNEdges    = graph.nEdges();

    // (2) σ-grid (log-spaced).
    const int nSig = config_.sigmaCount;
    std::vector<double> sigmas(static_cast<std::size_t>(nSig), 0.0);
    const double logMin = std::log(config_.sigmaMin);
    const double logMax = std::log(config_.sigmaMax);
    for (int i = 0; i < nSig; ++i) {
        const double frac =
            static_cast<double>(i) / static_cast<double>(nSig - 1);
        sigmas[static_cast<std::size_t>(i)] =
            std::exp(logMin + frac * (logMax - logMin));
    }
    result.sigmas = sigmas;

    // (3) P(σ) via the heat-kernel trace estimator.
    result.P  = graph.returnProbability(sigmas, config_.krylovDim);

    // (4) D_S(σ) via centered finite differences (raw) and a
    // Savitzky-Golay local-polynomial fit (smoothed). Both are
    // reported.
    result.dS         = EmergentGraph::spectralDimension(sigmas, result.P);
    result.dSSmoothed = EmergentGraph::spectralDimensionSmoothed(
        sigmas, result.P, /*windowSize=*/5, /*polyOrder=*/2);

    // (5) Ambjorn-Loll three-parameter fit on the smoothed D_S; the raw
    // finite-difference signal can latch onto grid-spacing noise.
    auto fit = AmbjornLollFit::fit(sigmas, result.dSSmoothed);
    result.dInfinity     = fit.dInfinity;
    result.C             = fit.C;
    result.B             = fit.B;
    result.fitChiSquared = fit.chiSquared;

    // (6) Snapshot summary copied across.
    const auto& snaps = quench.snapshots;
    result.snapshotTimes.reserve(snaps.size());
    result.snapshotBondDims.reserve(snaps.size());
    result.snapshotEnergies.reserve(snaps.size());
    for (auto const& s : snaps) {
        result.snapshotTimes.push_back(s.time);
        result.snapshotBondDims.push_back(s.bondDim);
        result.snapshotEnergies.push_back(s.energy);
    }
    return result;
}

SpectralDimensionResult
EmergentSpectralDimension::compute() const {
    // Run the TDVP pipeline; recordMutualInformation is forced on by
    // the constructor so the snapshots arrive with MI populated.
    const auto quench = SchwingerQuench{config_.tdvp}.evolve();
    return computeFromSnapshots(quench);
}

} // namespace tessera::quantum
