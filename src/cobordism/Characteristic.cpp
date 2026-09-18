// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "cobordism/Characteristic.h"

#include <exception>

#include "cobordism/ChainComplex.h"
#include "spacetime/Spacetime.h"

namespace tessera::cobordism {

double EulerCharacteristic::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  return static_cast<double>(ChainComplex::fromSpacetime(*spacetime).eulerCharacteristic());
}

double Signature::compute(const std::shared_ptr<Spacetime> &spacetime) {
  if (spacetime == nullptr) return 0.0;
  return static_cast<double>(ChainComplex::fromSpacetime(*spacetime).signature());
}

CharacteristicNumbers CharacteristicNumbers::of(const Spacetime &K, bool oriented) {
  const ChainComplex cc = ChainComplex::fromSpacetime(K);
  CharacteristicNumbers out;
  out.euler = cc.eulerCharacteristic();
  if (oriented && cc.dimension() == 4) {
    try {
      const int signatureValue = cc.signature();
      out.signature = signatureValue;
      // Hirzebruch signature theorem: the first Pontryagin number equals three
      // times the signature.
      out.pontryaginNumbers["p1"] = 3 * static_cast<long>(signatureValue);
    } catch (const std::exception &) {
      // Non-orientable / no fundamental class: signature (hence p1) undefined.
    }
  }
  // Stiefel–Whitney numbers are unoriented invariants, computed regardless of
  // the `oriented` flag. A class needing a higher Steenrod square leaves the
  // family empty rather than failing the call.
  try {
    out.stiefelWhitneyNumbers = cc.stiefelWhitneyNumbers();
  } catch (const std::exception &) {
    // Higher cup-i square required, or not a closed manifold.
  }
  return out;
}

}  // namespace tessera::cobordism
