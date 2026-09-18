// QuantumVertex — mesh::Vertex carrying a density matrix.
//
// QuantumVertex extends mesh::Vertex with an Eigen-typed density matrix
// in its local Hilbert space. The dimension is fixed at construction
// rather than by template parameter, so QuantumVertex objects in one
// VertexList can carry states of different sizes: the A, B vertices of a
// Koashi-Imoto (KI) interaction cell carry ρ_A, ρ_B of dimension d_A,
// d_B, the Σ vertex carries ρ_Σ on the KI core, and the A', B' tail
// vertices carry ρ_{A'}, ρ_{B'}.
//
// QuantumVertex lives in tessera_quantum because density matrices
// require Eigen, and tessera_core is deliberately Eigen-free.
// VertexList stores its instances polymorphically through
// std::unique_ptr<Vertex>, so a QuantumVertex* can be downcast from
// a Vertex* via dynamic_cast or via the helper ``require``.

#pragma once

#include "mesh/Vertex.h"

#include <Eigen/Dense>

#include <stdexcept>
#include <utility>
#include <vector>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

class QuantumVertex : public ::tessera::mesh::Vertex {
  public:
    // Default constructor — empty state, id = 0. Needed so the pool
    // free-list path that placement-constructs into a slot has a
    // valid default. State must be assigned before use.
    QuantumVertex() noexcept = default;

    // Construct with id and a density matrix.
    QuantumVertex(std::uint64_t id, Eigen::MatrixXcd state) noexcept
      : Vertex(id), state_(std::move(state)) {}

    QuantumVertex(std::uint64_t id,
                  const std::vector<double>& coords,
                  Eigen::MatrixXcd state) noexcept
      : Vertex(id, coords), state_(std::move(state)) {}

    /// The vertex's quantum state ρ, in its local basis.
    [[nodiscard]] const Eigen::MatrixXcd& getState() const noexcept {
      return state_;
    }

    /// Replace the density matrix. Used by KI factories that
    /// allocate a vertex first and fill in its state once the
    /// decomposition is computed.
    void setState(Eigen::MatrixXcd state) noexcept {
      state_ = std::move(state);
    }

    [[nodiscard]] int stateDim() const noexcept {
      return static_cast<int>(state_.rows());
    }

    /// Van Raamsdonk distance d_VR = −log(I / iMax) between this vertex
    /// and ``other``. I is the mutual information of the product joint
    /// ρ_this ⊗ ρ_other, which is the right joint for a pair carrying no
    /// inherited correlation — every pair the KI factories build except
    /// the input (A, B) edge. A product joint has I = 0, so the returned
    /// distance is +∞.
    ///
    /// The KI-cell factory uses this method for the nine non-(A, B)
    /// edges and computes the (A, B) edge from its input joint ρ_AB. It
    /// then writes d_VR² into ``Edge::squaredLength``.
    ///
    /// Throws std::invalid_argument if ``other`` is not a QuantumVertex.
    [[nodiscard]] double
    vanRaamsdonkDistanceTo(const ::tessera::mesh::Vertex* other,
                           double                          iMax) const;

    /// Downcast a Vertex* to QuantumVertex*; throws
    /// std::invalid_argument if the dynamic type is wrong.
    static QuantumVertex* require(::tessera::mesh::Vertex* v) {
      auto* qv = dynamic_cast<QuantumVertex*>(v);
      if (qv == nullptr) {
        throw std::invalid_argument(
          "QuantumVertex::require: vertex is not a QuantumVertex");
      }
      return qv;
    }

  private:
    Eigen::MatrixXcd state_{};
};

} // namespace tessera::quantum
