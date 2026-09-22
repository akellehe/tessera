// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "mesh/Fingerprint.h"
#include "mesh/Edge.h"
#include "mesh/EdgeKey.h"
#include "mesh/RiemannSheet.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "utils.h"

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <numbers>


// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

class Simplex;

    Edge::Edge(
      const VertexPtr &source_,
      const VertexPtr &target_,
      std::complex<double> length
    ) : source(source_), target(target_), length_(length),
        phase(0.0), fingerprint({source_->getId(), target_->getId()}) {
    }

    Edge::Edge(
      const VertexPtr &source_,
      const VertexPtr &target_
    ) : source(source_), target(target_), phase(0.0), fingerprint({source_->getId(), target_->getId()}) {
      // Fallback (CDT always provides explicit edge lengths): a random real,
      // i.e. spacelike, length.
      length_ = {random_uniform(), 0.0};
    }

    [[nodiscard]] const VertexPtr &Edge::getSource() const noexcept {
      return source;
    }

    [[nodiscard]] const VertexPtr &Edge::getTarget() const noexcept {
      return target;
    }

    [[nodiscard]] std::complex<double> Edge::getPhase() const noexcept {
      return phase;
    }

    [[nodiscard]] std::complex<double> Edge::getLength() const noexcept {
      return length_;
    }

    [[nodiscard]] double Edge::squaredArgument() const noexcept {
      const auto l = getLength();
      return principalArgument(l * l);  // in (-pi, pi]
    }

    [[nodiscard]] double Edge::declaredSquaredArgument() const noexcept {
      return squaredArgument() +
             2.0 * std::numbers::pi * static_cast<double>(squaredWinding_);
    }

    [[nodiscard]] int Edge::squaredSheet() const noexcept {
      const int parity = squaredWinding_ % 2;
      return (parity < 0) ? parity + 2 : parity;
    }

    void Edge::continueLength(std::complex<double> l) noexcept {
      // The sheet is carried by the squared length -- the quantity every
      // downstream squared-volume formula is a function of. SheetedSqrt accrues
      // the turn l^2 makes about its branch point on this step and reports the
      // running monodromy.
      SheetedSqrt root(getLength() * getLength(), squaredWinding_);
      root.advance(l * l);
      squaredWinding_ = root.winding();
      length_ = l;
      ++lengthRevision_;
    }

    namespace {
    /// The declared argument folded back into (-pi, pi], which is what a causal
    /// bucket is a function of: the two roots +/-l of one l^2 share a causal
    /// character, so the declared sheet cannot move the bucket. It fixes only
    /// which side of the cut a timelike edge sits on, and that is what
    /// `declaredSquaredArgument` keeps and this fold discards.
    [[nodiscard]] inline double foldToPi(double declared) noexcept {
      const double twoPi = 2.0 * std::numbers::pi;
      double folded = std::fmod(declared + std::numbers::pi, twoPi);
      if (folded <= 0.0) folded += twoPi;
      return folded - std::numbers::pi;
    }
    }  // namespace

    [[nodiscard]] double Edge::lorentzianMagnitude() const noexcept {
      // Re(l^2) = x^2 - t^2 for l = x + i t. Formed from the parts rather than as
      // (l*l).real() so the subtraction is the only cancellation step. Carried for
      // consumers that want the interval; it does not decide the disposition alone.
      const auto l = getLength();
      return l.real() * l.real() - l.imag() * l.imag();
    }

    [[nodiscard]] bool Edge::isDegenerate() const noexcept {
      // The Euclidean modulus, and the one place it is the right norm: an edge with
      // no extent at all is absent, not lightlike.
      return std::abs(getLength()) <= kDegenerateEpsilon;
    }

    [[nodiscard]] bool Edge::isSpacelike() const noexcept {
      // The declared argument ~ 0 (mod 2 pi): l^2 real positive. |arg| folds the
      // (-pi, pi] range so each test below is one comparison against a single
      // definite argument. Reading the declaration rather than re-deriving
      // arg(l^2) is what makes the bucket a function of the transported state
      // and not of where the principal cut happens to fall.
      if (isDegenerate()) return false;
      return std::abs(foldToPi(declaredSquaredArgument())) <= kCausalAngularEpsilon;
    }

    [[nodiscard]] bool Edge::isTimelike() const noexcept {
      // The declared argument ~ +/- pi: l^2 real negative.
      if (isDegenerate()) return false;
      return std::abs(std::abs(foldToPi(declaredSquaredArgument())) - std::numbers::pi)
             <= kCausalAngularEpsilon;
    }

    [[nodiscard]] bool Edge::isNull() const noexcept {
      // The declared argument ~ +/- pi/2: l^2 purely imaginary and nonzero -- the
      // light cone, reached non-trivially at Re(l) == Im(l) != 0. Not the same as
      // degenerate.
      if (isDegenerate()) return false;
      return std::abs(std::abs(foldToPi(declaredSquaredArgument())) - 0.5 * std::numbers::pi)
             <= kCausalAngularEpsilon;
    }

    [[nodiscard]] bool Edge::isMixed() const noexcept {
      // No definite causal character. Not snapped to the nearest of the three: a
      // generic argument is genuinely mixed, and reporting it as definite would
      // invent structure the geometry does not have.
      return !isDegenerate() && !isSpacelike() && !isTimelike() && !isNull();
    }

    [[nodiscard]] EdgeDisposition Edge::disposition() const noexcept {
      if (isDegenerate()) return EdgeDisposition::Degenerate;
      if (isSpacelike()) return EdgeDisposition::Spacelike;
      if (isTimelike()) return EdgeDisposition::Timelike;
      if (isNull()) return EdgeDisposition::Lightlike;
      return EdgeDisposition::Mixed;
    }

#ifdef TESSERA_VERBOSE
    [[nodiscard]] std::string Edge::toString() const noexcept {
      return source->toString() + "->" + target->toString();
    }
#endif

    void Edge::replaceSourceVertex(const VertexPtr &newSource) {
      fingerprint.removeId(source->getId());
      source = newSource;
      fingerprint.addId(newSource->getId());
      fingerprint.refresh();
    }

    void Edge::replaceTargetVertex(const VertexPtr &newTarget) {
      fingerprint.removeId(target->getId());
      target = newTarget;
      fingerprint.addId(newTarget->getId());
      fingerprint.refresh();
    }

    bool Edge::hasVertex(std::uint64_t vertexId) const {
      if (getSource()->getId() == vertexId || getTarget()->getId() == vertexId) return true;
      return false;
    }

    bool Edge::hasVertex(const VertexPtr &vertex) const {
      return vertex != nullptr && hasVertex(vertex->getId());
    }

    bool Edge::operator==(const Edge &other) const {
      return fingerprint.fingerprint() == other.fingerprint.fingerprint();
    }

    [[nodiscard]] std::uint64_t Edge::toHash() const {
      return fingerprint.fingerprint();
    }

    EdgeKey Edge::getKey() const noexcept {
      return {source->getId(), target->getId()};
    }

    void Edge::registerSimplex(SimplexPtr s) {
      // Append; the registry is duplicate-free by construction because
      // Simplex::addEdge / Spacetime::registerSimplex deduplicate before
      // calling. Idempotency under multiple registers would require a
      // scan we don't want in the hot path.
      simplices_.push_back(s);
    }

    void Edge::unregisterSimplex(SimplexPtr s) noexcept {
      // Swap-pop by fingerprint to keep the storage tight. The list is
      // small (≤ few simplices per edge in typical 4D builds) so the
      // linear scan is cheap; the canonical Simplex pointer is unique
      // per fingerprint so the first match suffices.
      const auto fp = s->fingerprint.fingerprint();
      for (std::size_t i = 0; i < simplices_.size(); ++i) {
        if (simplices_[i]->fingerprint.fingerprint() == fp) {
          simplices_[i] = simplices_.back();
          simplices_.pop_back();
          return;
        }
      }
    }

double Edge::vanRaamsdonkLength(double I, double iMax,
                                double epsilon) noexcept {
  const double x = (iMax > 0.0 && I > 0.0) ? (I / iMax) : 0.0;
  double dVR = (x > 0.0) ? -std::log(x)
                         : std::numeric_limits<double>::infinity();
  if (epsilon <= 0.0) {
    return dVR;  // caller opted out of the floor; +inf is the law's value
  }
  const double cap = -std::log(epsilon);  // floor on d_VR => finite length
  if (!std::isfinite(dVR) || dVR > cap) {
    dVR = cap;
  }
  return dVR;
}

double Edge::vanRaamsdonkLengthFor(double I, double iMax,
                                   double epsilon) const {
  const VertexPtr s = getSource();
  const VertexPtr t = getTarget();
  // Forward-time worldline edge (endpoints on different time slices) → null.
  if (s != nullptr && t != nullptr &&
      std::abs(s->getTime() - t->getTime()) > 1e-12) {
    return 0.0;
  }
  return vanRaamsdonkLength(I, iMax, epsilon);  // spacelike
}

}

