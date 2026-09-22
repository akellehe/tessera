# C++ API

tessera simulates lattice spacetimes built from causal sets and causal
simplicial complexes. This page is the public C++ interface. To see the
internal interface as well, edit the Doxyfile and comment out its exclusions.

<!--
  This page must list every public header under ``include/`` via a Breathe
  ``{doxygenfile}`` directive, grouped into the per-module sections below.
  The sections mirror the ``include/`` directory tree one-to-one.

  Doxygen's INPUT (see ../Doxyfile: ``INPUT = ../src ../include`` with
  ``RECURSIVE = YES``) covers the entire tree, so every header has a
  Breathe-addressable file compound — entries here only need the header's
  basename (there are no duplicate header basenames in the tree).

  When you add a header under ``include/``, add a matching ``{doxygenfile}``
  entry to the correct section here. ``tests/test_cpp_api_docs_coverage.py``
  fails if a header is missing from, or stale in, this page.
-->

## Core spacetime

```{doxygenfile} Spacetime.h
```
```{doxygenfile} Metric.h
```
```{doxygenfile} Signature.h
```
```{doxygenfile} Foliation.h
```

## Simplicial complex

```{doxygenfile} Simplex.h
```
```{doxygenfile} TemporalOrientation.h
```
```{doxygenfile} Vertex.h
```
```{doxygenfile} VertexList.h
```
```{doxygenfile} Edge.h
```
```{doxygenfile} EdgeKey.h
```
```{doxygenfile} EdgeList.h
```
```{doxygenfile} Fingerprint.h
```
```{doxygenfile} SimplexFilter.h
```
```{doxygenfile} FlatHashMap.h
```
```{doxygenfile} ForwardDeclarations.h
```

## Pachner moves

```{doxygenfile} PachnerMove.h
```
```{doxygenfile} AddMove.h
```
```{doxygenfile} RemoveMove.h
```
```{doxygenfile} FlipMove.h
```
```{doxygenfile} IFlipMove.h
```
```{doxygenfile} ShiftMove.h
```

## Topologies

```{doxygenfile} Topology.h
```
```{doxygenfile} Toroid.h
```
```{doxygenfile} Cylinder.h
```
```{doxygenfile} Sphere.h
```
```{doxygenfile} SimplicialProduct.h
```
```{doxygenfile} PeriodicKuhnGrid.h
```
```{doxygenfile} SphereCircleProduct.h
```
```{doxygenfile} SimplexBoundarySphere.h
```
```{doxygenfile} SolidSimplex.h
```
```{doxygenfile} StellarSubdivision.h
```
```{doxygenfile} RealProjectivePlane.h
```
```{doxygenfile} RealProjectiveSpace.h
```
```{doxygenfile} ComplexProjectivePlane.h
```

## Simulations

```{doxygenfile} Simulation.h
```
```{doxygenfile} CDT.h
```
```{doxygenfile} ReggeSolver.h
```
```{doxygenfile} InteractionSimulation.h
```

## Observables

```{doxygenfile} EffectiveTopology.h
```
```{doxygenfile} Observable.h
```
```{doxygenfile} VolumeProfile.h
```
```{doxygenfile} SpacetimeVolume.h
```
```{doxygenfile} Spectral.h
```
```{doxygenfile} SimplicialQubit.h
```
```{doxygenfile} WilsonLoop.h
```
```{doxygenfile} MIUnits.hpp
```
```{doxygenfile} ModularityOptimizer.h
```
```{doxygenfile} PersistentModularity.h
```
```{doxygenfile} SparseGraph.h
```
```{doxygenfile} ColorFiber.h
```
```{doxygenfile} SpectralFiber.h
```
```{doxygenfile} ExchangeHolonomy.h
```
```{doxygenfile} FiberConnection.h
```
```{doxygenfile} ClusterRegister.h
```
```{doxygenfile} ParticleClusters.h
```
```{doxygenfile} CrossingReadouts.h
```

### Emergent-proton observables

```{doxygenfile} Record.h
```
```{doxygenfile} RegisterContext.h
```
```{doxygenfile} LiveComplex.h
```
```{doxygenfile} InteriorHinges.h
```
```{doxygenfile} RegisterObservable.h
```
```{doxygenfile} SingletResidual.h
```
```{doxygenfile} BlockResiduals.h
```
```{doxygenfile} EmergentMass.h
```
```{doxygenfile} EmergentRadius.h
```
```{doxygenfile} PairLoopFlavor.h
```
```{doxygenfile} ObservableGates.h
```
```{doxygenfile} DualVolumeSigns.h
```

## Cobordism

```{doxygenfile} ChainComplex.h
```
```{doxygenfile} HodgeLaplacian.h
```
```{doxygenfile} IntegerLinalg.h
```
```{doxygenfile} EigenstateSynthesis.h
```
```{doxygenfile} Characteristic.h
```
```{doxygenfile} CombinatorialDimension.h
```
```{doxygenfile} Cochain.h
```
```{doxygenfile} Spectrum.h
```
```{doxygenfile} CobordismObjective.h
```
```{doxygenfile} MultiCobordism.h
```
```{doxygenfile} CobordismDAG.h
```
```{doxygenfile} Proton.h
```
```{doxygenfile} ProtonIngredients.h
```
```{doxygenfile} SurgicalCone.h
```
```{doxygenfile} Certificate.h
```
```{doxygenfile} AnalyticCache.h
```
```{doxygenfile} KuennethProduct.h
```
```{doxygenfile} SpacetimeComposition.h
```
```{doxygenfile} OccupationSpectra.h
```
```{doxygenfile} LowRankUpdate.h
```
```{doxygenfile} DenseReference.h
```
```{doxygenfile} RecursiveQuotient.h
```
```{doxygenfile} PencilLayer.h
```
```{doxygenfile} LevenbergMarquardt.h
```

## Whitney-form Hodge Laplacian pencil

```{doxygenfile} WhitneyMass.h
```
```{doxygenfile} ChainHodge.h
```
```{doxygenfile} LorentzianFamily.h
```
```{doxygenfile} CovariantChainHodge.h
```
```{doxygenfile} FaceAnchor.h
```
```{doxygenfile} RieszBand.h
```

```{doxygenfile} BandDerivative.h
```
```{doxygenfile} PencilSchur.h
```
```{doxygenfile} SparsePencil.h
```
```{doxygenfile} SparsePencilSolver.h
```
```{doxygenfile} SparseRank.h
```

## Reinforcement learning

```{doxygenfile} CobordismObjectiveEnv.h
```
```{doxygenfile} PpoAgent.h
```
```{doxygenfile} Trainer.h
```

## Quantum

```{doxygenfile} SchwingerModel.hpp
```
```{doxygenfile} DMRGRunner.hpp
```
```{doxygenfile} TDVPRunner.hpp
```
```{doxygenfile} TDVPIntegrator.hpp
```
```{doxygenfile} Quench.hpp
```
```{doxygenfile} ChoiJamiolkowski.h
```
```{doxygenfile} ChoiState.hpp
```
```{doxygenfile} Holography.hpp
```
```{doxygenfile} MutualInformation.hpp
```
```{doxygenfile} Majorization.hpp
```
```{doxygenfile} KoashiImoto.hpp
```
```{doxygenfile} Schmidt.hpp
```
```{doxygenfile} CausalCompare.hpp
```
```{doxygenfile} CausetChain.hpp
```
```{doxygenfile} QuantumSimplex.hpp
```
```{doxygenfile} QuantumVertex.hpp
```
```{doxygenfile} GradedFock.h
```
```{doxygenfile} LazyFock.h
```
```{doxygenfile} CovarianceState.h
```

## Graph

```{doxygenfile} DualGraph.hpp
```
```{doxygenfile} SpectralGraph.hpp
```
```{doxygenfile} WeightedCsrGraph.hpp
```
```{doxygenfile} CSRBuilder.hpp
```
```{doxygenfile} COO.hpp
```
```{doxygenfile} IndexByKey.hpp
```

## Matter

```{doxygenfile} MatterConfiguration.h
```

## Constraints

```{doxygenfile} Constraint.h
```

## GPU / CUDA acceleration

```{doxygenfile} eigenstate_cuda.h
```

## Core utilities

```{doxygenfile} Poset.h
```
```{doxygenfile} Renderer.h
```
```{doxygenfile} ForceLayout.h
```
```{doxygenfile} Logger.h
```
```{doxygenfile} utils.h
```
