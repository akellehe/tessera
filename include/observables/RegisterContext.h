// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_REGISTER_CONTEXT_H
#define TESSERA_OBSERVABLES_REGISTER_CONTEXT_H

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/Proton.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
// === cross-subsystem fwd-decls ===
namespace tessera::spacetime {
  class Spacetime;
}
namespace tessera::observables {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

class InteriorHinges;  // the shared 4D hinge-selection core (InteriorHinges.h)

/// # RegisterContext
///
/// The validated read context every emergent-proton observable measures: a
/// converged spacetime, its emergent holes, the induced-orientation signs, and
/// the shared per-complex caches. Composed from the readout cores
/// `cobordism::EigenstateSynthesis`, `cobordism::MultiCobordism` and
/// `cobordism::ChainComplex`.
///
///   * A reader, not a builder. The context reads an already-built, relaxed
///     spacetime: a `Proton::block()`, a `ProtonIngredients` state, a relaxed
///     `MultiCobordism` complex, or a dump the loader already rehydrated into a
///     live complex. It never builds, solves or materializes anything; the
///     emergent build lives in Proton, ProtonIngredients and MultiCobordism. The
///     facet/coface skeleton that the `dualVolume()` and `deficitAngle()` reads
///     walk must already be present on the live complex, as it is on every built
///     state. Completing a bare `Spacetime::fromCells` skeleton — for dump
///     rehydration and the RELABEL-gate rebuild — is the job of `LiveComplex`.
///   * Hole selection is validated at one entry point. The selected holes are the
///     emergent `(degree+2)`-vertex removed top cells
///     (`cobordism::MultiCobordism::emergentHoles`), in emergent-hole order. A
///     deficit throws `std::invalid_argument` naming the holes found. A surplus
///     is a recorded truncation naming the dropped holes in `selectionWarning()`
///     (the Python binding emits it as a `UserWarning`), never a silent slice.
///     Both censuses are kept (`holesUsed()` and `holesTotal()`) alongside the
///     Betti number at the register degree (`bK()`) and the
///     `holesVsBettiDivergent()` flag, since the hole count and \f$ b_k \f$ can
///     disagree (for instance 3 holes against \f$ b_3 = 2 \f$).
///   * One orientation convention. The induced-orientation signs
///     \f$ \varepsilon_h = \pm 1 \f$ come from
///     `cobordism::ChainComplex::endSignCovector`: the label-free orientation
///     under which every closed form's signed periods obey
///     \f$ \sum_h \varepsilon_h p_h = 0 \f$, determined up to one global sign
///     (the propagation root, \f$ -1 \in U(1) \f$ on the register, which is gauge
///     rather than physics). Vertices are never sorted to impose a convention;
///     holes are matched by vertex set.
///   * One cached `EigenstateSynthesis` per (spacetime, degree), plus the other
///     shared per-complex structures: Hodge metric weights, the canonical
///     \f$ k \f$-cell index, the Betti vector, and the 4D interior-hinge
///     selection. `gauged()` copies share the caches, since the gauge knob only
///     rotates the target.
///
/// The GAUGE gate transform acts on the context, not on the observables:
/// `gauged(theta)` rotates the register target by the surviving global U(1)
/// phase, which contains the \f$ \mathbb{Z}_3 \f$ cyclic recolor of the singlet
/// and the orientation flip. It builds nothing and shares the same live complex.
/// The RELABEL gate needs a rebuilt, relabeled complex; that construction lives
/// in `LiveComplex` and is orchestrated by `ObservableGates`.
class RegisterContext {
  public:
    /// Read the context over the already-built `spacetime`, selecting and
    /// validating `count` emergent holes at `degree`: a deficit throws, and a
    /// surplus is recorded in `selectionWarning()` naming the dropped holes.
    /// `target` is the register target state, one component per hole slot,
    /// defaulting to the color singlet \f$ [1, \omega, \omega^2] \f$. The GAUGE
    /// gate rotates exactly this.
    /// @throws std::invalid_argument on a hole deficit (fewer than `count`
    ///   emergent holes), an empty complex, or `count < 0` / `degree < 0`.
    explicit RegisterContext(
        std::shared_ptr<Spacetime> spacetime, int count = 3, int degree = 3,
        std::vector<std::complex<double>> target = cobordism::Proton::singlet());

    /// Build the context with an explicit hole selection, such as a build's own
    /// census or the relabel gate's matched images. Validated with the same count
    /// semantics as the selecting constructor: fewer than `count` throws, more
    /// than `count` is a recorded truncation naming the dropped holes.
    /// `holesTotal()` still reports the complex's own emergent census.
    RegisterContext(std::shared_ptr<Spacetime> spacetime,
                    const std::vector<std::vector<std::uint64_t>> &holes,
                    int count, int degree,
                    std::vector<std::complex<double>> target);

    // ---- validated register ----
    [[nodiscard]] const std::shared_ptr<Spacetime> &spacetime() const noexcept {
      return spacetime_;
    }
    /// The register degree \f$ k \f$ (holes are `(k+2)`-vertex removed top
    /// cells; the cached synthesis reads at this \f$ k \f$).
    [[nodiscard]] int degree() const noexcept { return degree_; }
    /// The register target state (rotated by `gauged()`).
    [[nodiscard]] const std::vector<std::complex<double>> &target() const noexcept {
      return target_;
    }
    /// The selected holes, in emergent-hole order (each a vertex-id
    /// tuple of a removed top cell, intrinsic order preserved).
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &holes()
        const noexcept {
      return holes_;
    }
    /// The surplus holes dropped by the validated selection (empty when none).
    [[nodiscard]] const std::vector<std::vector<std::uint64_t>> &droppedHoles()
        const noexcept {
      return droppedHoles_;
    }
    /// Holes the register reads (`holes().size()`).
    [[nodiscard]] int holesUsed() const noexcept {
      return static_cast<int>(holes_.size());
    }
    /// The complex's full emergent-hole census at the register degree.
    [[nodiscard]] int holesTotal() const noexcept { return holesTotal_; }
    /// Betti at the register degree (GF(2) ranks; 0 when the Betti vector is
    /// shorter than the degree).
    [[nodiscard]] int bK() const;
    /// The full Betti vector of the complex (cached).
    [[nodiscard]] const std::vector<int> &betti() const;
    /// True when the emergent-hole census and the Betti number at the register
    /// degree disagree.
    [[nodiscard]] bool holesVsBettiDivergent() const { return holesTotal_ != bK(); }
    /// The surplus-selection warning naming the dropped holes (empty when the
    /// selection was exact). The Python binding emits it as a `UserWarning` at
    /// construction; C++ callers consult it here.
    [[nodiscard]] const std::string &selectionWarning() const noexcept {
      return selectionWarning_;
    }
    /// The canonical spacetime dimension \f$ d \f$: the metric signature's
    /// dimension, from `getMetric()->getSignature()->getDimensions()`, the same
    /// accessor `WilsonLoop` and `ReggeSolver` read.
    [[nodiscard]] int dimensions() const noexcept { return dimensions_; }
    /// The number of top cells.
    [[nodiscard]] int topCellCount() const noexcept { return topCellCount_; }
    /// True iff any edge is non-spacelike (timelike or null). At initialization
    /// no time has passed and causal structure has yet to emerge, so
    /// all-spacelike specimens report false.
    [[nodiscard]] bool causalContent() const noexcept { return causalContent_; }

    // ---- shared per-complex caches ----
    /// The one cached `cobordism::EigenstateSynthesis(spacetime, degree)` every
    /// period readout shares (lazily built; `gauged()` copies share it).
    [[nodiscard]] cobordism::EigenstateSynthesis &synthesis() const;
    /// The induced-orientation signs \f$ \varepsilon_h = \pm 1 \f$ of the
    /// selected holes, from `cobordism::ChainComplex::endSignCovector`. Lazily
    /// built and shared across `gauged()` copies.
    [[nodiscard]] const std::vector<int> &epsilonSigns() const;
    /// The Hodge metric weights \f$ W_k \f$ at the register degree, in the
    /// canonical `ChainComplex` \f$ k \f$-cell order (lazily built, shared).
    [[nodiscard]] const std::vector<double> &hodgeWeights() const;
    /// The canonical \f$ k \f$-cell index: sorted vertex-id tuple → operator
    /// index into `synthesis().cellSimplices()` (lazily built, shared).
    [[nodiscard]] const std::map<std::vector<std::uint64_t>, std::size_t> &
    cellIndex() const;
    /// The shared 4D interior-hinge selection: `InteriorHinges` over this
    /// context's spacetime and holes, lazily built and shared. `EmergentMass` and
    /// `EmergentRadius` compose this one instance.
    /// @throws std::invalid_argument if the complex is not genuinely 4D.
    [[nodiscard]] const std::shared_ptr<InteriorHinges> &interiorHinges() const;

    // ---- GAUGE gate transform ----
    /// The GAUGE-gate variant: the same live complex and register, with the
    /// target rotated by the global U(1) phase \f$ e^{i\theta} \f$ — the
    /// register's one surviving gauge freedom, containing the
    /// \f$ \mathbb{Z}_3 \f$ cyclic recolor of the singlet and the orientation
    /// flip. Shares this context's spacetime and caches; nothing is rebuilt. The
    /// RELABEL gate needs a rebuilt, relabeled complex and lives in
    /// `LiveComplex` and `ObservableGates`.
    [[nodiscard]] std::shared_ptr<RegisterContext> gauged(double theta) const;

  private:
    /// The lazily-built shared structures. Held behind one `shared_ptr` so
    /// `gauged()` copies share every cache in both directions: a cache built by
    /// either context is visible to the other, since the complex is one object.
    struct Caches;

    /// Shared constructor tail: read the live complex's censuses and validate
    /// the hole selection. A null `explicitHoles` selects from the emergent
    /// census. Reads only; nothing is built or materialized.
    void initialize(int count,
                    const std::vector<std::vector<std::uint64_t>> *explicitHoles);

    std::shared_ptr<Spacetime> spacetime_;
    int degree_;
    std::vector<std::complex<double>> target_;
    std::vector<std::vector<std::uint64_t>> holes_;
    std::vector<std::vector<std::uint64_t>> droppedHoles_;
    int holesTotal_ = 0;
    int dimensions_ = -1;
    int topCellCount_ = 0;
    bool causalContent_ = false;
    std::string selectionWarning_;
    std::shared_ptr<Caches> caches_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_REGISTER_CONTEXT_H
