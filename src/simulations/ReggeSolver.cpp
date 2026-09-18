// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#include "simulations/ReggeSolver.h"
#include "spacetime/Spacetime.h"
#include "graph/IndexByKey.hpp"
#include "mesh/Simplex.h"
#include "mesh/Edge.h"
#include "mesh/Vertex.h"
#include "mesh/EdgeList.h"
#include "mesh/VertexList.h"
#include "mesh/Fingerprint.h"


#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numbers>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <Eigen/SparseCore>

#ifdef _OPENMP
#include <omp.h>
#endif

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

// =====================================================================
// Construction
// =====================================================================

ReggeSolver::ReggeSolver(std::shared_ptr<Spacetime> spacetime,
                         MatterConfiguration matter)
    : spacetime_(std::move(spacetime)), matter_(std::move(matter)) {
    // Materialize the facet/coface lattice down to the (d-2)-hinges, in C++.
    //
    // dualVolume() walks a hinge up through its cofaces to a top cell, so every
    // coface link from the hinge up to the top must exist. getFacets() on a
    // k-simplex creates its (k-1)-facets and registers itself as their coface,
    // so it has to run on every simplex of size >= d: the top d-cells (size d+1)
    // register the (d-1)-facets, and the (d-1)-facets (size d) register the
    // (d-2)-hinges. build() does not guarantee the (d-1)-facets exist (a freshly
    // built SolidSimplex holds only its top cells), so this starts from the tops
    // rather than assuming the facets are present.
    //
    // This has to run in C++. The Python getFacets()/getCofaces() bindings use
    // return_value_policy::copy, so driving materialization from Python would
    // register copies of the sub-simplices — each carrying an incomplete coface
    // list — onto the shared vertices; the fingerprint-keyed hasCoface() guard
    // would then block the canonical facets and dualVolume() would see half the
    // cofaces it should.
    //
    // getFacets() grows simplicesVec as it registers new sub-simplices, so the
    // loop re-reads size() each iteration rather than snapshotting it.
    const int d = spacetime_->getMetric()->getSignature()->getDimensions();
    for (std::size_t i = 0; i < spacetime_->getSimplices().size(); ++i) {
        auto s = spacetime_->getSimplices()[i];
        if (static_cast<int>(s->size()) >= d)
            (void)s->getFacets();
    }
}

// =====================================================================
// Geometry delegations to Simplex
// =====================================================================

std::complex<double> ReggeSolver::dihedralAngle(SimplexPtr sigma,
                                   SimplexPtr hinge) const {
    return sigma->dihedralAngle(hinge);
}

std::complex<double> ReggeSolver::deficitAngle(SimplexPtr hinge) const {
    return hinge->deficitAngle();
}

std::complex<double> ReggeSolver::hingeArea(SimplexPtr hinge) {
    // The signed Lorentzian area: a timelike hinge's area is imaginary, not
    // |l^2|-real. There is no Wick-rotated mode.
    return hinge->area();
}

// =====================================================================
// Collect hinges
// =====================================================================

std::vector<SimplexPtr> ReggeSolver::collectHinges() const {
    // Hinges are (d-2)-simplices. In 4D, these are triangles (3 vertices).
    // They are registered in the spacetime's simplex list (sub-simplices are
    // registered during getFacets()).
    //
    // Only genuine hinges count toward the Regge action: a (d-2)-face of at
    // least one current top (d)-cell. A Pachner move that removes a cell can
    // leave a lazily-materialised hinge registered with no surviving top coface
    // (an orphan); ``deficitAngle`` then returns a bare 2π for it while its
    // gradient maps are empty, so an unfiltered sum would let the resident
    // action and ``actionGradientExact`` disagree. Skipping orphans
    // (``hasTopCoface``) makes ``dualReggeAction`` a pure function of the
    // current top-cell set — equal to a from-scratch rebuild — so the action is
    // invariant under any move∘move⁻¹ that restores those cells. That is
    // bookkeeping over which hinges are real, not a change to the action
    // S = Σ_h |★h|·ε_h itself.
    int d = spacetime_->getMetric()->getSignature()->getDimensions();
    int hingeSize = d - 1; // (d-2)-simplex has (d-1) vertices

    std::vector<SimplexPtr> hinges;
    for (const auto &s : spacetime_->getSimplices()) {
        if (static_cast<int>(s->size()) == hingeSize && s->hasTopCoface())
            hinges.push_back(s);
    }
    return hinges;
}

// =====================================================================
// Actions
// =====================================================================

std::complex<double> ReggeSolver::reggeAction() const {
    std::complex<double> S{0.0, 0.0};
    for (const auto &h : collectHinges()) {
        S += hingeArea(h) * h->deficitAngle();
    }
    return S;
}

std::complex<double> ReggeSolver::dualReggeAction() const {
    // S_Regge(W*) = sum_h |*h| * eps_h: the circumcentric dual content of each
    // (d-2)-hinge weighted by its complex Lorentzian deficit. Hinges must be
    // registered (sub-simplices materialize via getFacets), as for reggeAction().
    std::complex<double> S(0.0, 0.0);
    for (const auto &h : collectHinges()) {
        S += h->dualVolume() * h->deficitAngle();
    }
    return S;
}

std::vector<std::vector<std::uint64_t>> ReggeSolver::hingeFacesOfCells(
    const std::vector<std::vector<std::uint64_t>> &cells) const {
    // A hinge is a (d-2)-simplex = (d-1) vertices. Its dual-action contribution
    // can change only when a move adds/removes a top coface around it, i.e. only
    // for hinges that are faces of a touched top cell — so the affected set is the
    // dedup'd (d-1)-element sub-tuples of the touched cells. Pure topology.
    const int d = spacetime_->getMetric()->getSignature()->getDimensions();
    const auto hingeSize = static_cast<std::size_t>(d - 1);
    std::set<std::vector<std::uint64_t>> seen;
    for (const auto &cell : cells) {
        std::vector<std::uint64_t> cs(cell.begin(), cell.end());
        std::sort(cs.begin(), cs.end());
        cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
        if (cs.size() < hingeSize) continue;
        // Every hingeSize-element combination of cs (sorted ⇒ combinations stay
        // sorted), via a descending-prefix selection mask.
        std::vector<bool> mask(cs.size(), false);
        std::fill(mask.begin(), mask.begin() + static_cast<std::ptrdiff_t>(hingeSize),
                  true);
        do {
            std::vector<std::uint64_t> hinge;
            hinge.reserve(hingeSize);
            for (std::size_t i = 0; i < cs.size(); ++i)
                if (mask[i]) hinge.push_back(cs[i]);
            seen.insert(std::move(hinge));
        } while (std::prev_permutation(mask.begin(), mask.end()));
    }
    return {seen.begin(), seen.end()};
}

std::complex<double> ReggeSolver::dualReggeActionOverHinges(
    const std::vector<std::vector<std::uint64_t>> &hinges) const {
    // The dual Regge action over a fixed hinge set, term-for-term equal to
    // dualReggeAction's summand: |★h|·ε_h for each genuine hinge (registered,
    // with a top coface), 0 for orphans. Resolve tuples by vertex id.
    std::unordered_map<std::uint64_t, VertexPtr> vidx;
    for (const auto &v : spacetime_->getVertexList()->toVector())
        if (v != nullptr) vidx.emplace(v->getId(), v);

    std::complex<double> S(0.0, 0.0);
    std::set<std::vector<std::uint64_t>> done;  // dedup (caller sets may overlap)
    for (const auto &h : hinges) {
        std::vector<std::uint64_t> key(h.begin(), h.end());
        std::sort(key.begin(), key.end());
        key.erase(std::unique(key.begin(), key.end()), key.end());
        if (!done.insert(key).second) continue;
        VertexPtrs vp;
        vp.reserve(key.size());
        bool ok = true;
        for (const std::uint64_t id : key) {
            const auto it = vidx.find(id);
            if (it == vidx.end()) { ok = false; break; }
            vp.push_back(it->second);
        }
        if (!ok) continue;
        const auto s = spacetime_->findSimplexByVerts(vp);
        if (s == nullptr || !s->hasTopCoface()) continue;
        S += s->dualVolume() * s->deficitAngle();
    }
    return S;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
ReggeSolver::affectedEdgesOfCells(
    const std::vector<std::vector<std::uint64_t>> &cells) const {
    // A move changes ∂S/∂ℓ²_e only where an affected hinge contributes to e, i.e.
    // for edges sharing a top cell with an affected hinge. So: affected hinges →
    // their incident top cells → those tops' edges. Top cells are reached through a
    // hinge vertex's incidences (getSimplices), so this never depends on edges /
    // hinges being materialized as registered simplices.
    std::unordered_map<std::uint64_t, VertexPtr> vidx;
    for (const auto &v : spacetime_->getVertexList()->toVector())
        if (v != nullptr) vidx.emplace(v->getId(), v);
    const int topSize =
        spacetime_->getMetric()->getSignature()->getDimensions() + 1;

    std::set<std::pair<std::uint64_t, std::uint64_t>> E;
    for (const auto &ht : hingeFacesOfCells(cells)) {
        const auto v0 = vidx.find(ht[0]);
        if (v0 == vidx.end()) continue;
        const std::set<std::uint64_t> need(ht.begin(), ht.end());
        for (auto *sigma : v0->second->getSimplices()) {
            if (static_cast<int>(sigma->size()) != topSize) continue;
            std::set<std::uint64_t> sv;
            for (const auto &v : sigma->getVertices()) sv.insert(v->getId());
            if (!std::includes(sv.begin(), sv.end(), need.begin(), need.end()))
                continue;
            const auto &tv = sigma->getVertices();
            for (std::size_t i = 0; i < tv.size(); ++i)
                for (std::size_t j = i + 1; j < tv.size(); ++j) {
                    const std::uint64_t a = tv[i]->getId(), b = tv[j]->getId();
                    E.insert({std::min(a, b), std::max(a, b)});
                }
        }
    }
    return {E.begin(), E.end()};
}

double ReggeSolver::gradientNorm2OverEdges(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> &edges) const {
    std::unordered_map<std::uint64_t, VertexPtr> vidx;
    for (const auto &v : spacetime_->getVertexList()->toVector())
        if (v != nullptr) vidx.emplace(v->getId(), v);

    std::set<std::pair<std::uint64_t, std::uint64_t>> E;
    for (const auto &e : edges)
        E.insert({std::min(e.first, e.second), std::max(e.first, e.second)});

    // Every hinge that contributes to a query edge: the (d-2)-faces of the top cells
    // containing that edge. The top cells are reached through an edge endpoint's
    // incidences (getSimplices) — edges are not registered as 1-simplices, so resolve
    // via the vertex. Summing each query edge's full per-edge gradient over these
    // hinges is exact: they are precisely e's star for every e in E.
    const int d = spacetime_->getMetric()->getSignature()->getDimensions();
    const int topSize = d + 1;
    const auto hingeSize = static_cast<std::size_t>(d - 1);
    std::set<std::vector<std::uint64_t>> hinges;
    for (const auto &e : E) {
        const auto va = vidx.find(e.first);
        if (va == vidx.end()) continue;
        for (auto *sigma : va->second->getSimplices()) {
            if (static_cast<int>(sigma->size()) != topSize) continue;
            bool hasB = false;
            for (const auto &v : sigma->getVertices())
                if (v->getId() == e.second) { hasB = true; break; }
            if (!hasB) continue;
            std::vector<std::uint64_t> cs;
            for (const auto &v : sigma->getVertices()) cs.push_back(v->getId());
            std::sort(cs.begin(), cs.end());
            cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
            if (cs.size() < hingeSize) continue;
            std::vector<bool> mask(cs.size(), false);
            std::fill(mask.begin(),
                      mask.begin() + static_cast<std::ptrdiff_t>(hingeSize), true);
            do {
                std::vector<std::uint64_t> h;
                h.reserve(hingeSize);
                for (std::size_t i = 0; i < cs.size(); ++i)
                    if (mask[i]) h.push_back(cs[i]);
                hinges.insert(std::move(h));
            } while (std::prev_permutation(mask.begin(), mask.end()));
        }
    }

    // ∂S/∂ℓ²_e = Σ_h [∂|★h|·ε_h + |★h|·∂ε_h], restricted to the query edges (the
    // full per-edge gradient for every e in E, since E's hinges are all in `hinges`).
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::complex<double>> g;
    for (const auto &ht : hinges) {
        VertexPtrs vp;
        vp.reserve(ht.size());
        bool ok = true;
        for (const std::uint64_t id : ht) {
            const auto it = vidx.find(id);
            if (it == vidx.end()) { ok = false; break; }
            vp.push_back(it->second);
        }
        if (!ok) continue;
        const auto h = spacetime_->findSimplexByVerts(vp);
        if (h == nullptr || !h->hasTopCoface()) continue;
        const std::complex<double> eps = h->deficitAngle();
        const std::complex<double> dv = h->dualVolume();
        for (const auto &[ed, dEps] : h->deficitAngleGradient()) {
            const std::pair<std::uint64_t, std::uint64_t> k{
                std::min(ed.first, ed.second), std::max(ed.first, ed.second)};
            if (E.count(k)) g[k] += dv * dEps;
        }
        for (const auto &[ed, dDv] : h->dualVolumeGradient()) {
            const std::pair<std::uint64_t, std::uint64_t> k{
                std::min(ed.first, ed.second), std::max(ed.first, ed.second)};
            if (E.count(k)) g[k] += dDv * eps;
        }
    }

    double s = 0.0;
    for (const auto &[k, v] : g) s += std::norm(v);  // |∂S/∂ℓ²_e|²
    return s;
}

double ReggeSolver::matterAction() const {
    // Point-particle action: S_matter = -M ∫ dτ. Causal character comes from
    // the canonical Edge::isTimelike() classifier (Im of the complex length),
    // not a hand-rolled sign-of-Re test. The resident ℓ² is real and signed
    // (Edge::setSquaredLength), so the proper time of a timelike step is
    // √(-Re ℓ²) = √(-ℓ²); null edges are not timelike and contribute nothing.
    double S = 0.0;
    for (const auto &wl : matter_.getWorldlines()) {
        for (std::size_t i = 0; i + 1 < wl.vertices.size(); ++i) {
            auto *v1 = wl.vertices[i];
            auto *v2 = wl.vertices[i + 1];
            // Find the edge connecting consecutive worldline vertices
            for (const auto &e : v1->getEdges()) {
                auto *other = (e->getSource()->getId() == v1->getId())
                              ? e->getTarget() : e->getSource();
                if (other->getId() == v2->getId()) {
                    if (e->isTimelike())
                        S -= wl.mass *
                             std::sqrt(-(e->getLength() * e->getLength())).real();
                    break;
                }
            }
        }
    }
    return S;
}

std::complex<double> ReggeSolver::totalAction() const {
    return reggeAction() + matterAction();
}

// =====================================================================
// Action gradient: ∂S/∂ℓ²_e for each edge (numerical)
// =====================================================================

std::vector<std::complex<double>> ReggeSolver::actionGradient() const {
    auto edgeList = spacetime_->getEdgeList();
    auto edges = edgeList->toVector();
    std::vector<std::complex<double>> g(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const std::complex<double> origSq = (edges[i]->getLength() * edges[i]->getLength());
        const double W = std::abs(origSq);              // |l^2|
        const double h = std::max(W * 1e-4, 1e-8);
        const bool tl = edges[i]->isTimelike();
        auto sqAtW = [tl](double w) {                    // signed l^2 with |l^2|=w, same character
            return tl ? std::complex<double>{-w, 0.0}
                      : std::complex<double>{w, 0.0};
        };
        // Central differences in W-space, preserving edge character; perturb l^2
        // exactly and restore the original l^2 exactly (no sqrt round-trip drift).
        edges[i]->setLength(std::sqrt(sqAtW(W + h)));
        const std::complex<double> Sp = totalAction();
        edges[i]->setLength(std::sqrt(sqAtW(std::max(W - h, 1e-12))));
        const std::complex<double> Sm = totalAction();
        g[i] = (Sp - Sm) / (2.0 * h);
        edges[i]->setLength(std::sqrt(origSq));
    }
    return g;
}

std::vector<std::complex<double>> ReggeSolver::actionGradientExact() const {
    using cd = std::complex<double>;
    const auto edges = spacetime_->getEdgeList()->toVector();
    const std::size_t E = edges.size();
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> eidx;
    for (std::size_t i = 0; i < E; ++i) {
        const std::uint64_t a = edges[i]->getSource()->getId();
        const std::uint64_t b = edges[i]->getTarget()->getId();
        eidx[{std::min(a, b), std::max(a, b)}] = i;
    }

    // dS/dl^2_e = sum_h [ d|*h|/dl^2_e * eps_h + |*h| * d eps_h/dl^2_e ].
    //
    // The per-hinge work is independent: deficitAngle/dualVolume and their
    // gradients are const reads over already-materialized cofaces (no mutable
    // members, no lazy caches), so hinges parallelize cleanly. Many hinges
    // contribute to the same edge, so writing the shared g directly would
    // contend; each thread accumulates into its own partial vector and the
    // partials are summed into g in thread-index order. With schedule(static)
    // that order is fixed, so the result is deterministic run-to-run and
    // bit-identical to the serial code at one thread. At more than one thread it
    // matches serial to floating-point round-off, since the per-thread split
    // reassociates the per-edge sum. Respects OMP_NUM_THREADS; serial no-op when
    // built without OpenMP.
    const auto hinges = collectHinges();
    const auto nH = static_cast<std::ptrdiff_t>(hinges.size());
#ifdef _OPENMP
    const int nThreads = omp_get_max_threads();
#else
    const int nThreads = 1;
#endif
    std::vector<std::vector<cd>> partials(
        static_cast<std::size_t>(nThreads), std::vector<cd>(E, cd(0.0, 0.0)));

    // An exception may not escape an OpenMP region: it calls std::terminate and
    // takes the whole process. Nothing in the hinge geometry throws in normal
    // operation, but a genuine error can (std::bad_alloc, a corrupted or empty
    // simplex), so capture the first exception and rethrow it after the join.
    std::exception_ptr pending = nullptr;
    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<cd> &gLocal = partials[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static)
        for (std::ptrdiff_t hi = 0; hi < nH; ++hi) {
            try {
                const auto &h = hinges[static_cast<std::size_t>(hi)];
                const cd eps = h->deficitAngle();
                const std::complex<double> dv = h->dualVolume();
                for (const auto &[e, dEps] : h->deficitAngleGradient()) {
                    const auto it = eidx.find(e);
                    if (it != eidx.end()) gLocal[it->second] += dv * dEps;
                }
                for (const auto &[e, dDv] : h->dualVolumeGradient()) {
                    const auto it = eidx.find(e);
                    if (it != eidx.end()) gLocal[it->second] += dDv * eps;
                }
            } catch (...) {
                #pragma omp critical(tessera_regge_grad_eptr)
                if (!pending) pending = std::current_exception();
            }
        }
    }
    if (pending) std::rethrow_exception(pending);

    std::vector<cd> g(E, cd(0.0, 0.0));
    for (const auto &p : partials)
        for (std::size_t i = 0; i < E; ++i) g[i] += p[i];
    return g;
}

std::vector<std::vector<std::complex<double>>>
ReggeSolver::actionHessianExact() const {
    using cd = std::complex<double>;
    const auto edges = spacetime_->getEdgeList()->toVector();
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> eidx;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const std::uint64_t a = edges[i]->getSource()->getId();
        const std::uint64_t b = edges[i]->getTarget()->getId();
        eidx[{std::min(a, b), std::max(a, b)}] = i;
    }
    const std::size_t E = edges.size();
    // d^2 S/dl^2_e dl^2_f = sum_h [ d2V_ef*eps + dV_e*dEps_f + dV_f*dEps_e
    //                              + V*d2Eps_ef ]. Independent per hinge, so the
    // same per-thread-partial reduction as actionGradientExact (see there for
    // the determinism / bit-identity argument) — here over the dense E x E
    // matrix. Per-thread partials cost nThreads * E^2 complex; the meshes this
    // runs on are small (E in the hundreds) and the result is already a dense
    // E x E matrix, so this is a bounded constant-factor blowup.
    const auto hinges = collectHinges();
    const auto nH = static_cast<std::ptrdiff_t>(hinges.size());
#ifdef _OPENMP
    const int nThreads = omp_get_max_threads();
#else
    const int nThreads = 1;
#endif
    std::vector<std::vector<std::vector<cd>>> partials(
        static_cast<std::size_t>(nThreads),
        std::vector<std::vector<cd>>(E, std::vector<cd>(E, cd(0.0, 0.0))));

    // Same OpenMP exception discipline as actionGradientExact: a genuine error
    // must propagate loudly, never std::terminate.
    std::exception_ptr pending = nullptr;
    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<std::vector<cd>> &Hloc =
            partials[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static)
        for (std::ptrdiff_t hi = 0; hi < nH; ++hi) {
            try {
                for (const auto &[i, j, term] : hingeHessianEntries(
                         hinges[static_cast<std::size_t>(hi)], eidx))
                    Hloc[i][j] += term;
            } catch (...) {
                #pragma omp critical(tessera_regge_hess_eptr)
                if (!pending) pending = std::current_exception();
            }
        }
    }
    if (pending) std::rethrow_exception(pending);

    std::vector<std::vector<cd>> H(E, std::vector<cd>(E, cd(0.0, 0.0)));
    for (const auto &P : partials)
        for (std::size_t i = 0; i < E; ++i)
            for (std::size_t j = 0; j < E; ++j) H[i][j] += P[i][j];
    return H;
}

std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>>
ReggeSolver::hingeHessianEntries(
    const SimplexPtr &hinge,
    const std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> &eidx)
    const {
    using cd = std::complex<double>;
    // d^2 S/dl^2_e dl^2_f = d2V_ef*eps + dV_e*dEps_f + dV_f*dEps_e + V*d2Eps_ef,
    // summed over the hinges that couple e and f. This emits one hinge's
    // contributions; both the dense and sparse assemblies sum them per (e,f).
    std::vector<std::tuple<std::size_t, std::size_t, cd>> entries;
    const cd eps = hinge->deficitAngle();
    const cd V = hinge->dualVolume();
    const auto dEps = hinge->deficitAngleGradient();
    const auto dV = hinge->dualVolumeGradient();
    const auto d2Eps = hinge->deficitAngleHessian();
    const auto d2V = hinge->dualVolumeHessian();
    entries.reserve(dV.size() * dV.size());
    for (const auto &[e, dVe] : dV) {
        const auto ie = eidx.find(e);
        if (ie == eidx.end()) continue;
        const auto dEe_it = dEps.find(e);
        const cd dEe = (dEe_it != dEps.end()) ? dEe_it->second : cd(0.0, 0.0);
        for (const auto &[f, dVf] : dV) {
            const auto if_ = eidx.find(f);
            if (if_ == eidx.end()) continue;
            const auto dEf_it = dEps.find(f);
            const cd dEf = (dEf_it != dEps.end()) ? dEf_it->second : cd(0.0, 0.0);
            cd term = dVe * dEf + dVf * dEe;       // cross terms
            const auto k = std::make_pair(e, f);
            const auto d2Vit = d2V.find(k);
            if (d2Vit != d2V.end()) term += d2Vit->second * eps;
            const auto d2Eit = d2Eps.find(k);
            if (d2Eit != d2Eps.end()) term += V * d2Eit->second;
            entries.emplace_back(ie->second, if_->second, term);
        }
    }
    return entries;
}

Eigen::SparseMatrix<std::complex<double>>
ReggeSolver::actionHessianExactSparse() const {
    using cd = std::complex<double>;
    using Trip = Eigen::Triplet<cd>;
    const auto edges = spacetime_->getEdgeList()->toVector();
    const std::size_t E = edges.size();
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t> eidx;
    for (std::size_t i = 0; i < E; ++i) {
        const std::uint64_t a = edges[i]->getSource()->getId();
        const std::uint64_t b = edges[i]->getTarget()->getId();
        eidx[{std::min(a, b), std::max(a, b)}] = i;
    }
    // ∂²S/∂ℓ²_e∂ℓ²_f is nonzero only for edge pairs that share a hinge (local
    // coupling), so the assembled Hessian is sparse. Emit each hinge's
    // contributions (hingeHessianEntries) as (e,f) triplets; setFromTriplets
    // sums the per-hinge terms for each pair, giving identical values to the
    // dense actionHessianExact on the nonzero pattern at O(nnz) memory instead
    // of O(E²). Same parallel-over-hinges structure as the dense path; the
    // per-thread triplet lists are concatenated in thread-index order before
    // assembly (deterministic), as in actionGradientExact.
    const auto hinges = collectHinges();
    const auto nH = static_cast<std::ptrdiff_t>(hinges.size());
#ifdef _OPENMP
    const int nThreads = omp_get_max_threads();
#else
    const int nThreads = 1;
#endif
    std::vector<std::vector<Trip>> partials(static_cast<std::size_t>(nThreads));

    // Same OpenMP exception discipline as actionGradientExact: a genuine error
    // must propagate loudly, never std::terminate.
    std::exception_ptr pending = nullptr;
    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<Trip> &local = partials[static_cast<std::size_t>(tid)];
        #pragma omp for schedule(static)
        for (std::ptrdiff_t hi = 0; hi < nH; ++hi) {
            try {
                for (const auto &[i, j, term] : hingeHessianEntries(
                         hinges[static_cast<std::size_t>(hi)], eidx))
                    local.emplace_back(static_cast<int>(i),
                                       static_cast<int>(j), term);
            } catch (...) {
                #pragma omp critical(tessera_regge_sparse_eptr)
                if (!pending) pending = std::current_exception();
            }
        }
    }
    if (pending) std::rethrow_exception(pending);

    std::vector<Trip> triplets;
    std::size_t total = 0;
    for (const auto &p : partials) total += p.size();
    triplets.reserve(total);
    for (const auto &p : partials)
        triplets.insert(triplets.end(), p.begin(), p.end());

    Eigen::SparseMatrix<cd> H(static_cast<int>(E), static_cast<int>(E));
    H.setFromTriplets(triplets.begin(), triplets.end());  // sums duplicate (e,f)
    return H;
}

double ReggeSolver::actionGradientNorm() const {
    auto g = actionGradient();
    double F = 0.0;
    // Sum |dS/dl^2|^2 over edges: std::norm is the squared modulus, so Re and Im
    // both constrain the objective. Summing (Re g)^2 alone would build it on Re S.
    for (const std::complex<double> &gi : g) F += std::norm(gi);
    return F;
}

} // namespace tessera::simulations
