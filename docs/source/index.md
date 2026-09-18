# tessera

tessera is a C++ library, with Python bindings, for simplicial manifolds and
partially ordered ("causal") sets. It builds causally oriented simplicial meshes
in Lorentzian or Euclidean signature and computes observables on them —
holonomies, cochain quantities, Regge curvature, spectral dimension. Optional
CUDA acceleration is available for the Regge solver.

The Python bindings mirror the C++ interface, so the C++ reference below also
documents the Python API.

```{toctree}
:maxdepth: 2

getting_started
theory
examples
benchmarks
causal_sets
wilson_loops
cpp_api
```
