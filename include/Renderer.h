// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#pragma once

#include <string>

namespace tessera::spacetime { class Spacetime; }

namespace tessera {
using namespace ::tessera::spacetime;

/// Render the spacetime to an image file.
///
/// Uses a force-directed layout: time coordinate fixed per slice,
/// spatial coordinates optimized via spring + repulsion forces.
/// The layout is computed internally and does not modify vertex state.
///
/// A .gif path produces an animated GIF. Three parameters control the
/// rotation; the animation loops seamlessly when spin and precession
/// are integers:
///
///   tilt        – cone half-angle in degrees (default 25)
///   spin        – full Y-axis rotations per loop (default 1)
///   precession  – precession cycles per loop (default 1)
///
/// Per-frame rotation:
///   ry = 2π · spin · t
///   rx = tilt · cos(2π · precession · t)
///   rz = tilt · sin(2π · precession · t)
///
/// Any other path produces a static four-panel PNG.
void renderSpacetime(const Spacetime &st, const std::string &path,
                     int panelSize = 800, int layoutIters = 500,
                     double tilt = 25.0, int spin = 1,
                     int precession = 1, int nFrames = 36,
                     int delayCentiseconds = 15);

} // namespace tessera
