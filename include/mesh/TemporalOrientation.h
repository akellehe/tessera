// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 12/14/25.
//

#ifndef TESSERA_TEMPORALORIENTATION_H
#define TESSERA_TEMPORALORIENTATION_H

// This header does not include pybind11 on purpose: that would pull Python.h
// into every translation unit of the core mesh subsystem, including
// tessera_core, and force every consumer (test executables, tessera_quantum)
// to link Python. mesh, spacetime and observables use no pybind11; the
// bindings translation unit holds all of it.

#include <algorithm>
#include <memory>
#include <vector>

#include "mesh/ForwardDeclarations.h"

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

/// Where a simplex sits relative to the two time slices it spans: FUTURE when more of
/// its vertices lie on the later slice, PRESENT when more lie on the earlier one,
/// UNKNOWN when the counts are equal.
enum class TimeOrientation : uint8_t {
  FUTURE = 0,
  PRESENT = 1,
  UNKNOWN = 2
};

class TemporalOrientation {
  public:
    /// The temporal orientation of a simplex is how many of its vertices lie on the
    /// initial time slice and how many on the final one. It matters for Lorentzian
    /// causal dynamical triangulation (CDT) complexes, which admit only the
    /// orientations that progress forward in time and share faces without gaps.
    ///
    /// Every d-simplex splits its vertices across two adjacent slices t and t+1, so the
    /// split is \f$ (n, d + 1 - n) \f$.
    ///
    /// Reference: Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0105267
    ///
    /// @param ti_ The number of vertices on the initial time slice.
    /// @param tf_ The number of vertices on the final time slice.
    ///
    TemporalOrientation(uint8_t ti_, uint8_t tf_);
    TemporalOrientation();

    [[nodiscard]] TemporalOrientation decTf() const;
    [[nodiscard]] TemporalOrientation decTi() const;
    [[nodiscard]] TemporalOrientation flip() const;
    [[nodiscard]] TimeOrientation getOrientation() const;
    [[nodiscard]] std::pair<uint8_t, uint8_t> numeric() const;
    [[nodiscard]] std::string toString() const noexcept;
    [[nodiscard]] std::vector<TemporalOrientation> getFacialOrientations() const;
    [[nodiscard]] uint8_t getK() const; /// A k-simplex has \f$ k+1 \f$ vertices.
    [[nodiscard]] size_t hash() const;
    bool operator==(const TemporalOrientation &other) const noexcept;
    static TemporalOrientation orientationOf(const VertexPtrs &vertices);
    Fingerprint fingerprint;
  private:
    uint8_t ti{0};
    uint8_t tf{0};
    uint8_t k{0};
};

}

#endif //TESSERA_TEMPORALORIENTATION_H