// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 10/23/25.
//

#ifndef TESSERA_METRIC_H
#define TESSERA_METRIC_H

#include <memory>

#include "spacetime/Signature.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;
/// # The Metric
///
/// The metric tensor \f$ g_{\mu\nu} \f$ and the signature it carries, used to turn vertex coordinates into
/// edge lengths when the complex is not coordinate-free.
///
class Metric {
  public:
    /// @param coordinateFree_ True when squared lengths are stored on the edges rather than derived from
    ///   vertex coordinates (the usual CDT setting).
    /// @param signature_ The metric signature; copied into the Metric.
    Metric(bool coordinateFree_, const Signature &signature_);

    ///
    /// The squared length of the edge between the source and target vertices, for a coordinate-carrying
    /// complex. Uses the metric \f$ g_{\mu \nu} \f$ to compute the separation of vertex \f$ i \f$ and
    /// vertex \f$ j \f$ as
    ///
    /// \f[
    /// l_{ij}^2 = g_{\mu \nu} \Delta x^{\mu} \Delta x^{\nu}
    /// \f]
    ///
    /// where
    ///
    /// \f[
    /// \Delta x^{\mu} := x_i^{\mu} - x_j^{\mu}
    /// \f]
    ///
    /// with signature (-,+,+,+).
    ///
    /// Timelike edges will have negative squared lengths, spacelike edges positive squared lengths, and null/lightlike
    /// edges zero squared lengths.
    ///
    /// Causal dynamical triangulation (CDT) uses fixed spacelike edge lengths to build and update the
    /// triangulation; Regge calculus lets edge lengths vary.
    ///
    /// Reference: Ambjorn, Goerlich, Jurkiewicz & Loll, arXiv:1203.3591
    ///
    /// @throws std::runtime_error if the metric is coordinate-free, in which case the squared length is
    ///   stored on the edge instead.
    ///
    [[nodiscard]] double getSquaredLength(
      const std::vector<double> &sourceCoords,
      const std::vector<double> &targetCoords
      ) const;

    [[nodiscard]] const std::shared_ptr<Signature> &getSignature() const noexcept;

  private:
    std::shared_ptr<Signature> signature;
    bool coordinateFree;
};
} // namespace tessera::spacetime

#endif //TESSERA_METRIC_H
