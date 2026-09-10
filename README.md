# tessera

[![Build & Deploy Docs](https://github.com/akellehe/tessera/actions/workflows/pages.yml/badge.svg)](https://github.com/akellehe/tessera/actions/workflows/pages.yml)
[![build](https://github.com/akellehe/tessera/actions/workflows/build.yml/badge.svg)](https://github.com/akellehe/tessera/actions/workflows/build.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE.md)

**[Documentation](https://akellehe.github.io/tessera/)**

A personal sandbox for triangulated spacetimes and discrete-geometry experiments — building simplicial meshes, sampling them by Monte Carlo, solving the discrete Einstein equations, and rendering the results. Python on top, C++/CUDA underneath.

tessera bundles several discrete-geometry formulations on a shared simplicial mesh:

- **Causal Dynamical Triangulations (CDT)** -- Monte Carlo path integral over geometries with a causal foliation
- **Regge Calculus** -- discrete general relativity via deficit angles and edge-length dynamics
- **Spin Foam / GFT**, **Coset**, and **Ricci Flow** formulations (scaffolding in place)

## Installation

Requires Python 3.9+, a C++20 compiler, CMake 3.18+, and a BLAS/LAPACK backend
(Accelerate ships with macOS; on Linux install `liblapack-dev libblas-dev`).

```bash
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import tessera; print('tessera OK')"
```

CUDA GPU acceleration is auto-detected. To force it off: `TESSERA_CUDA=0 pip install -e .`

The **quantum subsystem** (Schwinger model / DMRG, ITensor-backed) is **always
built** — ITensor, Eigen3, and a BLAS/LAPACK backend are unconditional
dependencies. The ITensor submodule is fetched automatically on the first build
(a one-time network download) if it is not already present; cloning with
`git clone --recurse-submodules` avoids the fetch. A cold build is noticeably
slower because ITensor is compiled from source — `ccache` (below) helps a lot.

Builds also use [`ccache`](https://ccache.dev/) automatically when it is installed (`brew install ccache` / `apt-get install ccache`) — recommended: it makes rebuilds and CI dramatically faster.

### Reinforcement-learning subsystem (optional, libtorch)

The RL harness (`tessera.rl` — a PPO policy over `MultiCobordism.buildStep`, used
by the proton animation) is an **optional** extension built against libtorch. Because
it is compiled against torch, torch must be visible to the build interpreter — and
since torch is heavy and only the RL needs it, it lives in the `[rl]` extra rather
than as a universal build dependency. Build it with a single, ordinary (build-isolated)
install — the `TESSERA_RL` flag pulls torch into the isolated build environment on
demand:

```bash
TESSERA_RL=1 pip install -e ".[rl]"     # or: make rl
```

Without `TESSERA_RL`, the core and quantum builds are unaffected and never require
torch. (`TESSERA_RL` toggles a scikit-build-core override in `pyproject.toml` that adds
torch to `build.requires` only when set; see the `Makefile` for a fast, no-isolation
path for rapid C++ iteration.) Verify:

```bash
python -c "import tessera.rl; print('tessera.rl OK')"
```

### OpenMP (optional, speeds up large meshes)

Several CPU hot paths — the Regge action gradient/Hessian hinge loop and the
spectral-graph heat-kernel — are parallelized with OpenMP. It is **optional**:
without it the `#pragma omp` directives compile as no-ops and the code runs
serially, so the build never fails for lack of it.

- **Linux (gcc / clang):** the OpenMP runtime ships with the compiler — nothing
  to install; CMake detects it automatically.
- **macOS (Apple Clang):** Apple's compiler has no bundled OpenMP runtime —
  `brew install libomp` and CMake will pick it up.

CMake prints `OpenMP found (...)` or `OpenMP not found; ... no-ops` at configure
time, so you can confirm which path you built. Control the thread count at run
time with `OMP_NUM_THREADS` (e.g. `OMP_NUM_THREADS=16`); the parallel paths scale
near-linearly up to the core count, and the computed result is deterministic and
independent of the thread count (to floating-point round-off). The default when
unset is one thread per core.

## Quick start

Build a 4D Lorentzian spacetime, thermalize it with CDT, and export a rotating GIF:

```python
import tessera

metric = tessera.Metric(
    coordinateFree=True,
    signature=tessera.Signature(dimensions=4, signature_type=tessera.Lorentzian),
)
st = tessera.Spacetime(
    metric=metric, spacetimeType=tessera.CDT,
    alpha=1.0, a=1.0,
    foliation=tessera.PREFERRED, topology=tessera.Toroid(),
)
st.build(500)

cdt = tessera.CDTSimulation(
    spacetime=st, k0=2.2, k4=0.5, delta=0.6,
    epsilon=0.02, targetN41=st.getN41(),
)
cdt.tune()
cdt.sweep(100)

st.save("spacetime.gif", tilt=25, spin=1, precession=1)
```

## Features

### CDT Monte Carlo

Full implementation of 4D Causal Dynamical Triangulations following [Ambjorn, Jurkiewicz & Loll (2005)](https://arxiv.org/abs/hep-th/0505154). Includes all four Pachner moves (add, remove, flip, shift), Metropolis acceptance, and automatic coupling-constant tuning.

The CDT phase structure is accessible out of the box:

| Phase | Geometry | Coupling regime |
|-------|----------|-----------------|
| **A** (branched polymer) | Fractal, elongated | Large k_0 |
| **B** (crumpled) | Collapsed to 1-2 slices | Small k_0, small delta |
| **C_dS** (de Sitter) | Extended 4D, matches S^4 | Moderate k_0, nonzero delta |

```bash
python examples/phase_diagram.py        # scan the (k0, delta) plane
python examples/volume_profile_phases.py # visualize blob/crumpled/polymer shapes
```

### Regge solver

The discrete Einstein equations say that a Regge geometry's action is *stationary* in the squared edge lengths, `dS/d(l^2_e) = 0`. `ReggeSolver` evaluates that action and its exact analytic derivatives; it does not relax the geometry itself. Point-mass matter enters through the proper-time action, and the mass sources curvature around itself.

```python
matter = tessera.MatterConfiguration()
matter.setWorldlineMass(center_vertex, mass=1.0, spacetime=st)

solver = tessera.ReggeSolver(st, matter)

S = solver.totalAction()            # S_grav + S_matter
F = solver.actionGradientNorm()     # sum_e |dS/d(l^2_e)|^2, zero on a solution
```

`F` is the stationarity residual, and it rather than the action is what a relaxation drives to zero: the action is unbounded below and diverges if descended. `actionGradientExact` returns the same gradient per edge, analytically and in one pass, for the gravitational term on its own; `matterAction` carries the proper-time term. `examples/curvature_slice_gif.py` assembles both into the residual and minimizes it as a least-squares problem, bounding each edge so its `l^2` keeps its sign and no relaxation step alters the causal structure.

```bash
python examples/curvature_slice_gif.py --n-simplices 600 --seed 20260909 --mass 1.0
```

This produces a per-time-slice curvature heat-map GIF.

### Paper validation

The examples reproduce key figures from the CDT literature, primarily from:

> J. Ambjorn, J. Jurkiewicz, R. Loll, **"Reconstructing the Universe"**, Phys. Rev. D 72, 064014 (2005) [[hep-th/0505154]](https://arxiv.org/abs/hep-th/0505154)

| Example | Figures reproduced | What it shows |
|---------|-------------------|---------------|
| `volume_profile_phases.py` | Figs 4-6 | Volume profiles in phases A (polymer), B (crumpled), C (de Sitter) |
| `phase_diagram.py` | Fig 3 | Phase diagram scan over the (k_0, delta) coupling plane |
| `spectral_dimension.py` | Figs 9-10 | Spectral dimension D_S: ~1.8 at short scales, ~4.0 at long scales |
| `volume_scaling.py` | Figs 7-8, 12 | Hausdorff dimension D_H = 4 from volume-volume correlator collapse |
| `effective_action.py` | Figs 11-13 | Effective action, D_2 scaling dimension, minisuperspace comparison |
| `n32_distribution.py` | Fig 2 | N_32 distribution at fixed N_41 -- strong simplex-type coupling |

Each script includes the paper's coupling constants (k_0=2.2, delta=0.6) and prints reference values for comparison. Run any of them with `--help` to see the full parameter set.

### Topologies

Three spatial topologies for the foliated slices. Just swap the last argument to the `Spacetime` constructor:

```python
metric = tessera.Metric(
    coordinateFree=True,
    signature=tessera.Signature(dimensions=4, signature_type=tessera.Lorentzian),
)

# Toroid (T^3 x S^1) -- periodic in space and time, default
st = tessera.Spacetime(metric=metric, spacetimeType=tessera.CDT,
                     alpha=1.0, a=1.0, foliation=tessera.PREFERRED,
                     topology=tessera.Toroid())

# Sphere (S^3 x S^1) -- natural for de Sitter cosmology
st = tessera.Spacetime(metric=metric, spacetimeType=tessera.CDT,
                     alpha=1.0, a=1.0, foliation=tessera.PREFERRED,
                     topology=tessera.Sphere())

# Cylinder (Sigma x [0,T]) -- open time boundaries for transition amplitudes
st = tessera.Spacetime(metric=metric, spacetimeType=tessera.CDT,
                     alpha=1.0, a=1.0, foliation=tessera.PREFERRED,
                     topology=tessera.Cylinder())
```

### Visualization

Export spacetimes in multiple formats:

```python
st.save("spacetime.gif")      # animated rotating GIF
st.save("spacetime.png")      # static 4-panel PNG
st.save("spacetime.graphml")  # import into Gephi, yEd, etc.
st.save("spacetime.dot")      # render with Graphviz
```

The curvature-slice visualization embeds each spatial slice in 3D with a heat map of the deficit-angle curvature:

```python
from curvature_slice_gif import render_curvature_gif
render_curvature_gif(st, solver, worldline, "curvature.gif")
```

### GPU acceleration

The Regge solver optionally offloads deficit-angle and gradient computation to CUDA. This is transparent -- the same Python API works with or without a GPU. Significant speedups on triangulations with 10k+ simplices.

## Running tests

`pytest tests/` runs the **entire** suite — slow tests included. The `slow`
marker is opt-out, not opt-in: nothing is deselected unless you ask for it.

```bash
pytest tests/ -v                   # everything, including slow (>30s) tests
pytest tests/ -v -m "not slow"     # skip the slow tests
pytest tests/ -v -m slow           # run only the slow tests
```

Some tests cover the quantum subsystem (Schwinger model / DMRG). It is always
built, so these run as part of the normal suite:

```bash
pytest tests/ -v                     # includes the quantum tests
```

Build options via environment variables:

```bash
TESSERA_CUDA=0       pip install -e .     # CPU-only build
TESSERA_CCACHE=0     pip install -e .     # disable the ccache compiler cache
TESSERA_ASAN=1       pytest tests/        # AddressSanitizer + UBSan
TESSERA_VERBOSE=1    pytest tests/        # C++ logging
TESSERA_ASSERTIONS=1 pytest tests/        # extra invariant checks
```

## Building documentation

```bash
sudo apt-get install doxygen   # or: brew install doxygen
cd docs && pip install -r requirements-docs.txt && make html
open _build/html/index.html
```

## License

Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved. See [LICENSE.md](LICENSE.md).

Third-party components linked or redistributed by tessera are inventoried in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

