// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_COBORDISM_CHARACTERISTIC_H
#define TESSERA_COBORDISM_CHARACTERISTIC_H

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "observables/Observable.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime { class Spacetime; }
namespace tessera::cobordism {
using namespace ::tessera::observables;
using namespace ::tessera::spacetime;

/// # EulerCharacteristic
///
/// Observable measuring the Euler characteristic of a triangulation: the
/// alternating count of its cells,
/// \f$ \chi = (\text{vertices}) - (\text{edges}) + (\text{triangles}) - \cdots
/// = \sum_{k} (-1)^k\, |C_k| \f$. A topological invariant, computed from the
/// chain complex's face counts.
class EulerCharacteristic : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # Signature
///
/// Observable measuring the signature of a closed, orientable 4-dimensional
/// manifold: the number of positive minus the number of negative eigenvalues of
/// the intersection form on second homology. Returns 0 when second homology is
/// trivial or the manifold is not 4-dimensional.
/// See :func:`ChainComplex::signature`.
class Signature : public Observable {
  public:
    double compute(const std::shared_ptr<Spacetime> &spacetime) override;
};

/// # Characteristic numbers
///
/// Topological invariants of a closed piecewise-linear \f$ n \f$-manifold:
/// Euler characteristic, signature, and the Stiefel–Whitney and Pontryagin
/// families.
struct CharacteristicNumbers {
  /// Euler characteristic (see EulerCharacteristic above).
  int euler{0};

  /// Signature (see Signature above). Present only for an orientable
  /// 4-manifold; left empty for a non-orientable 4-manifold or any dimension
  /// other than 4, where it is not defined.
  std::optional<int> signature{};

  /// Stiefel–Whitney numbers: mod-2 invariants detecting orientability and
  /// related twisting. One entry per partition of \f$ n \f$, keyed by the
  /// characteristic-class monomial (e.g. "w1^2"), with value in
  /// \f$ \{0, 1\} \f$. Empty when the complex is not a closed manifold, or when
  /// a needed class requires a higher Steenrod cup-\f$ i \f$ product, which
  /// :func:`ChainComplex::stiefelWhitneyNumbers` does not implement.
  std::map<std::string, int> stiefelWhitneyNumbers{};

  /// Pontryagin numbers: integer invariants defined for orientable manifolds
  /// whose dimension is a multiple of 4. In dimension 4 there is only one,
  /// keyed "p1"; by the Hirzebruch signature theorem it is three times the
  /// signature, \f$ \langle p_1, [K]\rangle = 3\sigma \f$.
  std::map<std::string, long> pontryaginNumbers{};

  /// Compute the characteristic numbers of the manifold \a K. With \a oriented
  /// set and \a K 4-dimensional, this fills in the signature and the Pontryagin
  /// number \f$ p_1 = 3\sigma \f$, leaving both out if \a K admits no
  /// fundamental class. The Stiefel–Whitney numbers are computed either way.
  static CharacteristicNumbers of(const Spacetime &K, bool oriented = true);
};

}  // namespace tessera::cobordism

#endif  // TESSERA_COBORDISM_CHARACTERISTIC_H
