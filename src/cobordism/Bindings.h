// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#pragma once

// The shared prelude of the cobordism bindings: the includes, the namespace
// aliases and the helpers that the binding translation units of this
// directory and its subdirectories use. Bindings.cpp registers them in order
// (https://github.com/akellehe/tessera/issues/1453).
// Pybind11 bindings for the cobordism subsystem. Lives outside tessera_core
// (which is pybind-free) so the static library can be reused without the
// Python dependency. Always added to _tessera's sources.

#include <limits>
#include <optional>
#include <tuple>

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cobordism/AnalyticCache.h"
#include "cobordism/BoundStatePole.h"
#include "cobordism/Certificate.h"
#include "cobordism/ChainComplex.h"
#include "cobordism/Characteristic.h"
#include "cobordism/DenseReference.h"
#include "cobordism/DressedFluctuation.h"
#include "cobordism/KuennethProduct.h"
#include "cobordism/LevelRecursion.h"
#include "cobordism/MappingCylinder.h"
#include "cobordism/SpacetimeComposition.h"
#include "cobordism/LowRankUpdate.h"
#include "cobordism/OccupationSpectra.h"
#include "cobordism/Cochain.h"
#include "cobordism/CombinatorialDimension.h"
#include "cobordism/CobordismDAG.h"
#include "cobordism/EigenstateSynthesis.h"
#include "cobordism/MultiCobordism.h"
#include "observables/SimplicialQubit.h"
#include "cobordism/PencilLayer.h"
#include "cobordism/ProtonSynthesis.h"
#include "cobordism/ProtonIngredients.h"
#include "cobordism/HodgeLaplacian.h"
#include "cobordism/HolomorphicRelaxation.h"
#include "cobordism/IntegerLinalg.h"
#include "cobordism/JointAction.h"
#include "cobordism/SelfConsistentMeanField.h"
#include "cobordism/RecursiveQuotient.h"
#include "cobordism/SurgicalCone.h"
#include "cobordism/WardFlux.h"
#include "cobordism/Spectrum.h"
#include "spacetime/Spacetime.h"  // complete type required by pybind (typeid)

namespace py = pybind11;

using namespace tessera;
using namespace tessera::cobordism;
