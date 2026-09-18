// QuantumSimplex — Koashi-Imoto (KI) factories that build a 5-vertex /
// 10-edge mesh::Simplex from two QuantumVertex inputs.
//
// QuantumSimplex is not a runtime type. It is a static-only utility
// class whose only methods are the four KI factory entry points, which:
//   - take two pre-existing QuantumVertex objects (A and B) and a
//     simulation-wide ``iMax``;
//   - construct ρ_AB by one of four strategies (see the methods);
//   - run the symmetric Koashi-Imoto decomposition;
//   - allocate the Σ, A', B' QuantumVertex objects in the
//     spacetime's vertex list, carrying ρ_Σ, ρ_{A'}, ρ_{B'};
//   - write d_VR² = (−log(I / iMax))² as the squaredLength of each
//     of the ten edges;
//   - build a regular ``mesh::Simplex`` via ``spacetime.createSimplex``;
//   - return that ``mesh::Simplex*``.
//
// State distribution:
//   - per-vertex density matrices ρ live on the QuantumVertex objects
//     (which are owned by ``VertexList`` via ``unique_ptr<Vertex>``);
//   - per-edge Van Raamsdonk distance lives on ``Edge::squaredLength_``
//     (d_VR² so callers recover d_VR via sqrt and MI via
//     ``iMax · exp(−sqrt(squaredLength))``);
//   - ``iMax`` is global to the simulation and is not stored on the
//     simplex; callers track it externally;
//   - the KoashiImotoResult is computed inside the factory and
//     discarded: the vertex states plus block dimensions encode
//     everything the simplex needs at runtime.
//
// The five Position constants give canonical indices into the
// ``mesh::Simplex::getVertices()`` ordering used by the factories.

#pragma once

#include "mesh/Simplex.h"
#include "quantum/KoashiImoto.hpp"
#include "quantum/QuantumVertex.hpp"
#include "spacetime/Spacetime.h"

#include <Eigen/Dense>

namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::quantum {

class QuantumSimplex {
  public:
    // Static-only utility class — never constructed at runtime. The
    // default constructor is deleted; the destructor stays
    // compiler-generated because pybind11 registers a deallocator when
    // it binds this name into Python.
    QuantumSimplex() = delete;

    // Canonical positions in the returned ``mesh::Simplex``'s vertex
    // list. Callers downcast via ``QuantumVertex::require`` (or
    // ``dynamic_cast``) when they want the typed handle.
    enum Position : int {
        A       = 0,
        B       = 1,
        Sigma   = 2,
        APrime  = 3,
        BPrime  = 4,
    };

    // (a) Schmidt purification.
    //
    // Requires ρ_A on ``qva`` and ρ_B on ``qvb`` to share the same
    // eigenvalue spectrum. Constructs the canonical purifier
    //   |ψ⟩ = Σ_i √λ_i |a_i⟩|b_i⟩
    // and sets ρ_AB = |ψ⟩⟨ψ|. The resulting joint is pure and
    // I(A:B) = 2·H(λ). Throws ``std::invalid_argument`` if the
    // spectra do not match within ``tol.epsKiEigen``.
    [[nodiscard]] static ::tessera::mesh::Simplex*
    fromSchmidtPurification(::tessera::spacetime::Spacetime& spacetime,
                            QuantumVertex*                   qva,
                            QuantumVertex*                   qvb,
                            double                           iMax,
                            const KoashiImotoTolerances&     tol = {});

    // (b) Classical correlation (perfectly correlated diagonal joint).
    //
    //   ρ_AB = Σ_i λ_i |a_i⟩⟨a_i| ⊗ |b_i⟩⟨b_i|
    // in matched eigenbases. Separable; I(A:B) = H(λ). Throws if
    // the spectra do not match.
    [[nodiscard]] static ::tessera::mesh::Simplex*
    fromClassicalCorrelation(::tessera::spacetime::Spacetime& spacetime,
                             QuantumVertex*                   qva,
                             QuantumVertex*                   qvb,
                             double                           iMax,
                             const KoashiImotoTolerances&     tol = {});

    // (c) Explicit joint.
    //
    // Caller supplies ρ_AB directly. Its partial traces over B (resp.
    // A) must agree with ρ_A on ``qva`` (resp. ρ_B on ``qvb``).
    [[nodiscard]] static ::tessera::mesh::Simplex*
    fromExplicitJoint(::tessera::spacetime::Spacetime& spacetime,
                      QuantumVertex*                   qva,
                      QuantumVertex*                   qvb,
                      const Eigen::MatrixXcd&          rhoAB,
                      double                           iMax,
                      const KoashiImotoTolerances&     tol = {});

    // (d) Target mutual information.
    //
    // Interpolates
    //   ρ_AB(α) = (1−α)·(ρ_A ⊗ ρ_B) + α·ρ_AB^Schmidt
    // and binary-searches α so that I(A:B) ≈ targetMI. Requires
    // matched spectra (the Schmidt endpoint is only defined when
    // ρ_A and ρ_B agree). Throws if targetMI ∉ [0, 2·H(λ)].
    [[nodiscard]] static ::tessera::mesh::Simplex*
    fromTargetMutualInformation(::tessera::spacetime::Spacetime& spacetime,
                                QuantumVertex*                   qva,
                                QuantumVertex*                   qvb,
                                double                           targetMI,
                                double                           iMax,
                                const KoashiImotoTolerances&     tol = {});
};

} // namespace tessera::quantum
