# Benchmarks

```{note}
Results on this page were generated with **tessera {{version}}**.
The benchmark JSON log records the exact version, platform, and Python
version for each run so results can be compared across releases.
```

Build-time benchmarks for constructing causal dynamical triangulation (CDT)
simplicial complexes in dimensions 1D through 4D over a range of target sizes.

All timings are wall-clock averages over 5 repetitions on a single core,
measured with `time.perf_counter()`.  Run the benchmark script yourself to
get numbers for your hardware:

```bash
python examples/benchmarks/build_benchmark.py --save docs/source/assets/benchmarks/
```

The script writes a structured JSON log alongside the plots
(`benchmark_results.json`) for programmatic comparison across versions.

---

## Build time vs. complex size

Build time scales roughly linearly with the number of target simplices in
every dimension.  Higher-dimensional complexes are more expensive per
simplex because each $d$-simplex carries $\binom{d+1}{2}$ edges and
$d + 1$ vertices that must be linked into the triangulation.

```{image} assets/benchmarks/build_time.png
:alt: Log-log plot of build time vs. target simplices for 1D-4D
:width: 80%
:align: center
```

---

## Build throughput

Throughput (simplices per second) is highest in 1D--2D (~200,000--500,000
simpl/s) and decreases with dimension as the per-simplex bookkeeping
grows.  4D sustains ~150,000 simpl/s at 100k simplices, so a
100k-simplex triangulation builds in under 1 second.

```{image} assets/benchmarks/build_throughput.png
:alt: Bar chart of build throughput by dimension and target size
:width: 80%
:align: center
```

---

## Full dashboard

The four-panel view combines build time, throughput, actual-vs-requested
size, and complex density (vertices and edges per simplex) across all
dimensions.

```{note}
1D CDT produces far fewer simplices than requested because the
triangulation is topologically a circle and the builder converges once the
minimal cell structure is complete.  The 1D data points are included for
completeness but are not directly comparable to 2D--4D.
```

```{image} assets/benchmarks/build_benchmarks.png
:alt: Four-panel benchmark dashboard
:width: 100%
:align: center
```

---

## Results table

```{list-table} Build times (5-run average, v0.1.0)
:header-rows: 1
:widths: 8 15 15 15 15 15

* - Dim
  - Target
  - Actual
  - Vertices
  - Edges
  - Time (s)
* - 2D
  - 500
  - 498
  - 252
  - 750
  - 0.001
* - 2D
  - 10,000
  - 9,996
  - 5,001
  - 14,997
  - 0.038
* - 2D
  - 100,000
  - 99,996
  - 50,001
  - 149,997
  - 0.509
* - 3D
  - 500
  - 492
  - 168
  - 662
  - 0.002
* - 3D
  - 10,000
  - 9,996
  - 3,336
  - 13,334
  - 0.043
* - 3D
  - 100,000
  - 99,996
  - 33,336
  - 133,334
  - 0.580
* - 4D
  - 500
  - 500
  - 130
  - 635
  - 0.003
* - 4D
  - 10,000
  - 10,000
  - 2,505
  - 12,510
  - 0.058
* - 4D
  - 100,000
  - 100,000
  - 25,005
  - 125,010
  - 0.684
```

---

## Running the benchmark

```bash
# Default: 5 sizes x 4 dimensions x 5 repeats (~1 minute)
python examples/benchmarks/build_benchmark.py --save docs/source/assets/benchmarks/

# Quick check (fewer repeats)
python examples/benchmarks/build_benchmark.py --repeats 2 --save /tmp/bench/
```

The `--save` directory receives:

| File | Contents |
|:-----|:---------|
| `benchmark_results.json` | Structured log with metadata, per-trial timings, and summary statistics |
| `build_benchmarks.png` | Four-panel dashboard |
| `build_time.png` | Build time vs. size (standalone) |
| `build_throughput.png` | Throughput bar chart (standalone) |

`compare_benchmarks.py` plots two saved runs against each other:

```bash
# Save a baseline
python examples/benchmarks/build_benchmark.py --save /tmp/baseline/

# ... make changes, rebuild ...

# Save and compare
python examples/benchmarks/build_benchmark.py --save /tmp/new/
python examples/benchmarks/compare_benchmarks.py \
    --before /tmp/baseline/benchmark_results.json \
    --after /tmp/new/benchmark_results.json \
    --save docs/source/assets/benchmarks/
```

## References

- J. Ambjorn, J. Jurkiewicz, R. Loll, *Reconstructing the Universe*,
  [arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)
- U. Pachner, *P.L. homeomorphic manifolds are equivalent by elementary
  shellings*, European J. Combin. **12** (1991) 129.
