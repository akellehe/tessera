# Qubit cobordism — the run and checks C1–C5 (#964, T5)

The construction is `qubit_cobordism_spec.md` (this directory), implemented by
T1–T4 (#960–#963, #975–#977). This note records the run of spec T5: what was
driven, the numbers behind checks C1–C5, the scan of the objective's weights,
and what remains open. Records live under `~/cobordism-runs/qubit-cobordism/t5/`
(one JSON and one GIF per run, `runs.md` the index with every command line);
every number below is read from one of those records.

## What was run

Every run is the qubit mode of `examples/cobordism/emergence_animation.py`
(`run --inputs qubit`): two flat tori `SimplicialQubit.flat_torus(τ, n, n)`
with τ_A = 0.3 + 1.1i and τ_B = −0.2 + 0.8i, their one-layer collar as the
host, each torus one input block with its holomorphic form attached as a
degree-1 fiber on the harmonic contour and its marking with the input
coefficients (1, τ_in) on the block (`set_input_marking`), χ of spec S5 as
the Choi-decomposed two-body target at the driver's J = 1 and t = 0.05
(J·t = 0.05), the node's default objective (`legacy`: the Regge stationarity
term when `--regge` is on, plus Γ·r_U with r_U the weighted sum of the two
block residuals and the two-body leak), the complex locus, the Whitney
pencil. One engine unit is one stage-1 pass (at most one committed move,
drawn from 6 candidates at lookahead depth 1) followed by 12 stage-2
iterations; `--seed 7` throughout. The exit rule is the driver's: a unit that
improves the objective by less than `--tolerance` (1e-12 by default) ends
the run.

Two knobs the CLI does not expose were needed: Γ (the `MultiCobordism`
constructor's weight on r_U, which `build_qubit_node` does not pass) and the
pure-gauge phases of run (iv) (the driver has no phase input, and
`seed_collar` writes a zero phase on every host edge — spec S2 — so phases
written on the tori before seeding do not reach the host). Both are
supplied by `t5_run.py` in the records: it registers a node factory that
calls the driver's own `build_qubit_node` and then either substitutes, in
that process only, a `MultiCobordism` subclass whose constructor supplies
`gamma`, or applies the gauge exactly as the T2-bis test does
(`tests/cobordism/test_block_residual_whole_frame_python.py`,
`gauged_collar`): one gauge function g over every host vertex, φ_e = g(target)
− g(source) on every host edge and on each torus's own edges through the id
map, the gauged tori read by `SimplicialQubit` and their holomorphic forms
re-attached as the state fibers on the same cells (the marking survives the
re-attachment). The loop, the reads, the frames and the rendering are the
driver's. The harness also writes a diagnostics sidecar the driver's record
lacks: after every unit, the whole's cell, vertex and edge counts, every
edge's length and phase, `last_stage2_stationary`, `last_stage1_lookahead`
(0 = no F-lowering move found) and the norm of the analytic r_U ascent.

| run | what | units | wall time |
|---|---|---|---|
| `seed-3x3`, `seed-4x4` | the collar seed alone (`--steps 0`) | 0 | 7.8 s, 24.2 s |
| `scan-w1e6-regge-on` | the default weights: input weight 1e6, Γ = 1, Regge on | 5 of 8 (exit on tolerance) | 14 min 45 s |
| `scan-w1e4-regge-on` | input weight 1e4, Regge on | 3 of 8 (exit on tolerance) | 7 min 40 s |
| `scan-w1e6-regge-off` | input weight 1e6, Regge off | 8 | 36 min 53 s |
| `scan-w1e4-regge-off` | input weight 1e4, Regge off | 8 | 33 min 18 s |
| `default-w1e6-regge-on-tol30` | the default weights at `--tolerance 1e-30`, diagnostics | 5 of 8 (exit on tolerance) | 14 min 50 s |
| `chosen-w1e4-regge-on-tol30` | the chosen weights at `--tolerance 1e-30`, diagnostics | 3 of 8 (exit on tolerance) | 8 min 05 s |
| `phased-w1e4-regge-on-tol30` | run (iv): the tori under a pure gauge, the chosen weights | 3 of 8 (exit on tolerance) | 7 min 31 s |
| `synth-4x4-w1e4-regge-on-tol30` | run (iii): 4×4 tori at the chosen weights, seed 7 | 2 of 4 (exit on tolerance) | 4 min 35 s |
| `synth-4x4-w1e4-regge-on-tol30-seed8` | run (iii) again with `--seed 8` | 3 of 4 (exit on tolerance) | 23 min 35 s |
| `probe-4x4-stage2` | stage 2 alone on the pristine 4×4 collar (12 accepted steps) | — | 10 min 49 s |
| `gamma10-w1e6-regge-on-tol30`, `gamma100-w1e6-regge-on-tol30` | the Γ leg of the scan at the default weight | 6 of 8, 4 of 8 (exit on tolerance) | 19 min 32 s, 3 min 22 s |
| `live-attempts.log` | run (v): `--live` under the process's backend, under webagg | 1 | see C5 |

At OMP_NUM_THREADS=8 a 3×3 unit costs 200–260 s (stage 2 dominates; the
processes run at about two cores). The tolerance-1e-30 runs reproduce the
tolerance-1e-12 runs frame for frame to the last digit (`default-…-tol30`
against `scan-w1e6-regge-on`, `chosen-…-tol30` against `scan-w1e4-regge-on`):
on this host the drive is process-deterministic, and the exits on tolerance
are exact stationarity, not a threshold effect.

## C1. The collar seed

Frame 0 of every run is the collar seed (`MultiCobordism.seed_collar`, one
layer), read by the driver's own channels (`seed-3x3.json`, `seed-4x4.json`;
τ_A = 0.3 + 1.1i, τ_B = −0.2 + 0.8i):

| | 3×3 | 4×4 |
|---|---|---|
| cells / vertices / edges | 54 / 18 / 90 | 96 / 32 / 160 |
| boundary components (Euler characteristic, faces, vertices) | 2: (0, 18, 9) in block A, (0, 18, 9) in block B | 2: (0, 32, 16), (0, 32, 16) |
| `bridge_phase_complete`, uncovered torus faces | true, 0 | true, 0 |
| Betti numbers b₀…b₃ | [1, 2, 1, 0] | [1, 2, 1, 0] |
| monodromy (rounded), rounding residual, fit residual | identity, 1.7e-15, 2.4e-16 | identity, 4.2e-15, 1.6e-16 |
| harmonic rank of the whole | 2 | 2 |

The manifold condition is the gate the collar was created through
(`ChainComplex::dualComplexIsValid` on the whole, refused by name otherwise);
the boundary read splits `getBoundary()` into exactly the two tori, so
∂W = T_A ⊔ T_B. The bit-exact bridge rollback is T1's test
(`tests/cobordism/test_bridge_move_python.py::test_bridge_rollback_is_bit_exact`),
re-run in this session (`bridge-tests.log` in the records).

The seed residuals, frame 0 (weight 1e6, Γ = 1, Regge on; the residuals
themselves do not depend on the weights):

| | 3×3 A | 3×3 B | 4×4 A | 4×4 B |
|---|---|---|---|---|
| block residual (the whole's zero mode against (1, τ_in) in the live frame) | 3.099981e-3 | 9.344559e-3 | 4.006415e-3 | 1.171689e-2 |
| coefficients of the whole's zero mode in the frame | (0.99825 + 0.00114i, 0.29847 + 1.09505i) | (0.99782 + 0.00318i, −0.20131 + 0.78407i) | (0.99822 + 0.00230i, 0.29799 + 1.09270i) | (0.99756 + 0.00556i, −0.20059 + 0.77850i) |
| max distance of the coefficients to (1, τ_in) | 5.2e-3 | 1.6e-2 | 7.6e-3 | 2.2e-2 |
| own-kernel leak of the holomorphic form (diagnostic) | 2.2e-30 | 1.3e-30 | 2.9e-30 | 2.2e-30 |
| τ̂ of the block's own metric (qubit read), d_FS, d_WP to τ_in | τ_in to 1e-15, 0, 0 | τ_in to 1e-15, 0, 0 | τ_in to 1e-15, 0, 0 | τ_in to 1e-15, 0, 0 |
| J residual, cond G, non-Delaunay edges | 2.5e-16, 5.26, 9 | 1.6e-16, 2.08, 0 | 3.2e-16, 5.26, 16 | 6.3e-16, 2.08, 0 |

The block residual on the seed equals the restricted leak of the torus's
holomorphic form in the whole's zero mode (`leaks` channel: 3.099981e-3,
9.344559e-3, 4.006415e-3, 1.171689e-2), T2-bis's and T4's numbers. The
two-body read on the seed, in the derived period frames:

| | 3×3 | 4×4 |
|---|---|---|
| transfer T (real; imaginary parts ≤ 2e-17) | [[−0.015920, −0.019358], [−0.004327, −0.005689]] | [[−0.028714, −0.039005], [−0.020535, −0.051481]] |
| Schmidt spectrum, rank | (2.606e-2, 2.612e-4), 2 | (7.302e-2, 9.274e-3), 2 |
| leak against χ | 0.526175 | 0.666443 |
| reversal residual | 2.5e-14 | 8.3e-15 |
| objective: Regge stationarity + Γ·r_U = total | 123.124 + 12445.066 = 12568.190 | 344.094 + 15723.968 = 16068.062 |

χ is spec S5's for spin-½ with ψ = (1, τ_A)/√(1+|τ_A|²), φ likewise, at the
driver's default J = 1, t = 0.05 (J·t = 0.05): χ = [[0, 0.15262 + 0.55960i],
[−0.10174 + 0.40698i, 0]], the |01⟩, |10⟩ sector only. Torus A (sheared)
carries 9 (3×3) or 16 (4×4) non-Delaunay edges with negative cotangent
weights, flagged by the qubit read; the flat-torus read is exact regardless
(`simplicial_qubit_findings.md`).

## C2. The block residuals and the weight scan

The block residual of a torus is the leak of its input coefficients
(1, τ_in), written on its edges through its live frame, in the zero mode of
the entire cobordism restricted to those edges (spec D2 as revised). Its
collar value is 3.100e-3 (A) and 9.345e-3 (B) at 3×3; synthesis drives it
down by two to five orders of magnitude in the first unit in every run, and
the coefficients of the whole's zero mode in the live frames reach (1, τ_in)
to 1e-4–1e-5:

| run | weight, Γ, Regge | units (exit) | block residuals A, B: seed → end | coefficient distance to (1, τ_in) at the end | two-body leak: seed → peak → end | Schmidt spectrum at the end | Regge term: seed → end | τ̂_A, τ̂_B at the end (d_WP to τ_in) | wall time |
|---|---|---|---|---|---|---|---|---|---|
| `scan-w1e6-regge-on` | 1e+06, 1, on | 5 (tolerance) | 3.10e-03 → 3.74e-07 / 9.34e-03 → 1.95e-07 | 6.9e-05 / 1.8e-05 | 0.5262 → 0.5906 (unit 1) → 0.5297 | (2.528e-01, 1.417e-02) | 123.12 → 66.36 | 0.0803 + 1.1233i (0.198) / -0.2956 + 0.9074i (0.169) | 14 min 45 s |
| `scan-w1e4-regge-on` | 10000, 1, on | 3 (tolerance) | 3.10e-03 → 2.60e-05 / 9.34e-03 → 2.64e-05 | 1.6e-04 / 1.7e-04 | 0.5262 → 0.5262 (unit 0) → 0.3906 | (2.348e-01, 4.990e-02) | 123.12 → 30.07 | -0.1898 + 1.1144i (0.439) / -0.4714 + 0.9939i (0.372) | 7 min 40 s |
| `scan-w1e6-regge-off` | 1e+06, 1, off | 8 (budget) | 3.10e-03 → 3.49e-08 / 9.34e-03 → 4.28e-07 | 9.2e-06 / 3.8e-05 | 0.5262 → 0.5982 (unit 8) → 0.5982 | (2.916e-01, 8.393e-03) | off | 0.3145 + 1.1040i (0.014) / -0.2798 + 0.8885i (0.141) | 36 min 53 s |
| `scan-w1e4-regge-off` | 10000, 1, off | 8 (budget) | 3.10e-03 → 3.14e-07 / 9.34e-03 → 9.48e-07 | 7.1e-05 / 2.4e-05 | 0.5262 → 0.5704 (unit 1) → 0.4912 | (3.171e-01, 2.381e-02) | off | 0.3422 + 1.0900i (0.040) / -0.2148 + 0.8222i (0.033) | 33 min 18 s |
| `default-w1e6-regge-on-tol30` | 1e+06, 1, on | 5 (tolerance) | 3.10e-03 → 3.74e-07 / 9.34e-03 → 1.95e-07 | 6.9e-05 / 1.8e-05 | 0.5262 → 0.5906 (unit 1) → 0.5297 | (2.528e-01, 1.417e-02) | 123.12 → 66.36 | 0.0803 + 1.1233i (0.198) / -0.2956 + 0.9074i (0.169) | 14 min 50 s |
| `chosen-w1e4-regge-on-tol30` | 10000, 1, on | 3 (tolerance) | 3.10e-03 → 2.60e-05 / 9.34e-03 → 2.64e-05 | 1.6e-04 / 1.7e-04 | 0.5262 → 0.5262 (unit 0) → 0.3906 | (2.348e-01, 4.990e-02) | 123.12 → 30.07 | -0.1898 + 1.1144i (0.439) / -0.4714 + 0.9939i (0.372) | 8 min 05 s |

(Coefficient distance = max |(a, b) − (1, τ_in)| over the two coefficients;
d_WP = the Weil–Petersson distance of the torus's own τ̂ to τ_in. The
tolerance-1e-30 rows are the bit-identical twins of the first two rows.)

What the scan shows:

- **The block residuals settle where the weight balances the other terms.**
  With the Regge term on, the settled level is 3.7e-7 / 2.0e-7 at weight
  1e6 and 2.6e-5 / 2.6e-5 at 1e4 (the weighted terms 0.37 + 0.20 and
  0.26 + 0.26 in the objective, next to a two-body term of 0.53 and 0.39
  and a Regge term of 66 and 30). With the Regge term off the residuals keep
  descending through the 8-unit budget (to 3.5e-8 / 4.3e-7 at 1e6, 3.1e-7 /
  9.5e-7 at 1e4).
- **The whole's coefficients hold while the tori's own τ̂ drifts.** The
  coefficients stay at (1, τ_in) to 1e-4 from unit 1 on in every run, while
  τ̂ of each torus's own metric (the qubit read on the block's live surface,
  a diagnostic) moves: at the default weights τ̂_A goes from 0.3 + 1.1i to
  0.080 + 1.123i (d_FS 0.097, d_WP 0.198) and τ̂_B from −0.2 + 0.8i to
  −0.296 + 0.907i (0.080, 0.169); at weight 1e4 with the Regge term on the
  drift is larger (d_WP 0.439 and 0.372); with the Regge term off it is
  small (d_WP 0.014–0.141). The Regge term moves the tori's own lengths; the
  block residual constrains only the whole's zero mode in the frame those
  lengths define. Correspondingly the own-kernel leaks of the input
  holomorphic forms (the T2 diagnostic) leave their floor, 2e-30 →
  1.6e-3–5.9e-3 (A) and 1.2e-2–2.4e-2 (B), and the restricted leaks of the
  unphased holomorphic forms in the whole's zero mode do not descend
  (3.1e-3 / 9.3e-3 → 1.6e-3–5.9e-3 / 1.2e-2–2.4e-2): the tori no longer
  carry the input form as their own holomorphic line, which is D2 as
  revised — the state is held at the block as coefficients in the live
  frame, not as a form.
- **The relaxed lengths are complex, none timelike.** From unit 1 on every
  one of the 90 edges has a nonzero imaginary length (max |Im ℓ²| 2.5e-2 at
  the default weights, 3.5e-2 at 1e4); Re ℓ² stays in [0.070, 1.041]
  (seed [0.076, 1]); no edge turns timelike. The two-body transfer becomes
  complex with it (|Im T| up to 0.075 at 1e6, 0.048 at 1e4, 0.107–0.124
  with the Regge term off) while its reversal residual stays at 1e-15. The
  qubit read's J residual on the relaxed tori is 1.5e-3–4.4e-3 (2e-16 on
  the flat seed) with no Delaunay violation left.
- **Stage 1 committed no move in any 3×3 run.** The whole stays at 54 cells,
  18 vertices and 90 edges through every unit of every run (the diagnostics
  of the two tolerance-1e-30 runs and of the phased run; the drawn layout's
  edge count of every other run), and `last_stage1_lookahead` is 0 at every
  unit: none of the 6 candidates drawn per unit lowered F. The Betti
  numbers and the monodromy are therefore constant because the topology
  never moved, not because a move preserved them. Synthesis at 3×3 was
  stage 2 alone.
- **The runs stop when stage 2 goes stationary.** With the Regge term on,
  stage 2 hits its 12-iteration budget in unit 1 (and unit 2 at the default
  weights) and then ends on the stationarity test; the next unit changes
  nothing and the driver exits (unit 5 at the default weights, unit 3 at
  1e4), at tolerance 1e-30 exactly as at 1e-12. The r_U ascent alone is not
  zero there (norm 1.1e3 at the default weights, 88 at 1e4): the total
  objective is stationary along every line-search trial while its Regge
  and r_U parts pull against each other.

**The deliberate choice: input weight 1e4, Γ = 1, the Regge term on.** At
the default weight 1e6 the block terms dominate the descent by three orders
of magnitude and the two-body leak is sacrificed: it rises to 0.591 in unit
1 and settles at 0.530, above its seed value 0.526, and with the Regge term
off it rises monotonically to 0.598. At 1e4 the block residuals still fall
two orders below their collar values (2.6e-5) and the coefficients still
reach (1, τ_in) to 1.7e-4, while the two-body leak descends 0.526 → 0.391
and the Regge term descends 123 → 30, and the settled objective has its
three parts of comparable size (Regge 30.07, blocks 0.26 + 0.26, two-body
0.39). Γ does not change the ratio of the block terms to the two-body term
(both sit inside r_U), so it cannot repair the default weight's balance;
the Γ leg below measures that directly. The Regge term is kept because spec
S4's objective has it, and because without it the bulk's lengths have
nothing but r_U to relax against (the Regge-off runs are the scan's
control, not a candidate).

**The Γ leg** (`gamma10-…`, `gamma100-…`: the default weight 1e6, the Regge
term on, tolerance 1e-30, the node constructed at Γ through `t5_run.py
--gamma`; the objective's `register_residual` term is Γ·r_U, 124451 at the
seed for Γ = 10):

| Γ | units (exit) | block residuals at the end | two-body leak: seed → peak → end | Regge term: seed → end | stage 2 | wall time |
|---|---|---|---|---|---|---|
| 1 | 5 (tolerance) | 3.74e-7 / 1.95e-7 | 0.5262 → 0.5906 (unit 1) → 0.5297 | 123.12 → 66.36 | budget in units 1–2, then stationary | 14 min 50 s |
| 10 | 6 (tolerance) | 5.72e-7 / 1.62e-6 | 0.5262 → 0.5837 (unit 1) → 0.5819 | 123.12 → 125.92 | stationary from unit 1 | 19 min 32 s |
| 100 | 4 (tolerance) | 1.50e-4 / 1.65e-3 | 0.5262 → 0.5985 (unit 3) → 0.5985 | 123.12 → 172.97 | stationary from unit 1, a few accepted steps per unit | 3 min 22 s |

No Γ in {1, 10, 100} makes the two-body leak descend at the default
weight: it rises in unit 1 in every case and ends above its seed value.
Raising Γ leaves the block-to-two-body ratio inside r_U untouched and only
outweighs the Regge term, which then no longer descends (125.9 and 173.0 at
the end against 66.4 at Γ = 1); at Γ = 100 the stage-2 line search goes
stationary after a few steps in every unit and the block residuals stop two
orders above the Γ = 1 level. The lever is the input weight, not Γ.

## C3. Monodromy and the transfer

The monodromy read (`MultiCobordism.monodromy` with both markings, periods
with parallel transport) is the identity at every frame of every 3×3 run —
rounding residual ≤ 5.3e-15, fit residual ≤ 3.8e-16, harmonic rank 2,
Betti numbers [1, 2, 1, 0] — and at the 4×4 seed (4.2e-15, 1.6e-16). No run
produced a drawing with monodromy M ≠ 1, because no stage-1 move was
committed (above): the clause "with monodromy M, the periods transform by
M" is not exercised by these records. It was measured on the seed by T1
(a remarking of the far torus gives the swap and sign matrices).

The transfer T in the derived period frames is not diagonal and is a
metric quantity, as C3 says:

| | seed 3×3 | after synthesis, default weights (unit 4) | after synthesis, chosen weights (unit 2) | seed 4×4 |
|---|---|---|---|---|
| T | [[−0.01592, −0.01936], [−0.00433, −0.00569]] | [[−0.12422, −0.14429], [−0.10383, −0.09833]] + i·(≤ 0.075) | [[−0.12236, −0.16609], [−0.09892, −0.04186]] + i·(≤ 0.048) | [[−0.02871, −0.03901], [−0.02054, −0.05148]] |
| Schmidt spectrum (σ₁, σ₂), σ₂/σ₁ | (2.606e-2, 2.612e-4), 0.010 | (2.528e-1, 1.417e-2), 0.056 | (2.348e-1, 4.990e-2), 0.212 | (7.302e-2, 9.274e-3), 0.127 |
| off-diagonal / diagonal | 1.22 | 1.16 | 1.36 | 0.76 |
| reversal residual | 2.5e-14 | 1.6e-15 | 1.0e-15 | 8.3e-15 |

(Off-diagonal / diagonal = max |T₀₁|, |T₁₀| over max |T₀₀|, |T₁₁| on the real parts, T3's ratio.)
Under synthesis |T| grows eightfold and its second Schmidt channel rises
from 1 % of the first to 5.6 % (default weights) and 21 % (chosen weights);
χ's ratio is 0.72.

## C4. The two-body leak against χ

χ of spec S5 for ψ = (1, τ_A)/√(1+|τ_A|²) = (0.6594, 0.1978 + 0.7253i) and
φ = (0.7715, −0.1543 + 0.6172i) at J·t = 0.05 is
χ = [[0, 0.15262 + 0.55960i], [−0.10174 + 0.40698i, 0]], singular values
(0.5800, 0.4195), Schmidt rank 2, the |01⟩, |10⟩ sector only; the
first-order amplitude −iJt·χ and the exact evolution by the magnetization
blocks are in every record's `inputs.algebra`. The projective leak of χ in
the transfer:

| run | seed | trace | end | Schmidt spectrum at the end |
|---|---|---|---|---|
| default weights (1e6, Regge on) | 0.5262 | 0.5906 → 0.5432 → 0.5298 → 0.5297 → 0.5297 | 0.5297 (above the seed) | (2.528e-1, 1.417e-2) |
| chosen weights (1e4, Regge on) | 0.5262 | 0.3912 → 0.3906 → 0.3906 | 0.3906 | (2.348e-1, 4.990e-2) |
| 1e6, Regge off | 0.5262 | 0.5977 → 0.5971 → 0.5974 → 0.5978 → 0.5980 → 0.5980 → 0.5981 → 0.5982 | 0.5982 (rising) | (2.916e-1, 8.393e-3) |
| 1e4, Regge off | 0.5262 | 0.5704 → 0.5586 → 0.5477 → 0.5350 → 0.5244 → 0.5149 → 0.5005 → 0.4912 | 0.4912 (still descending at the budget) | (3.171e-1, 2.381e-2) |

The leak decreases under synthesis at the chosen weights (C4 holds there)
and its floor on the 3×3 collar is 0.3906 — a stationary point of the whole
objective, reached in two units. At the default weights it rises in unit 1
and never recovers its seed value: the block terms, 10⁶ × 10⁻³ against a
two-body term of 0.5, own the descent direction until they are three orders
smaller, by which point the geometry is stationary. This is the behaviour
§6 records for T3's stage-2 pass ("the two-body leak rises under
Regge-dominated steps"), but the scan locates it in the weight ratio
inside r_U rather than in the Regge term: with the Regge term off the leak
still rises at 1e6 and still descends at 1e4.

## The 4×4 run

`synth-4x4-w1e4-regge-on-tol30` (run iii): 4×4 tori at the chosen weights,
4 units asked, tolerance 1e-30, 4 min 35 s. It exited after 2 units:

| unit | objective (Regge, Γ·r_U) | block residuals A, B | two-body leak | cells / vertices / edges | what happened |
|---|---|---|---|---|---|
| 0 | 501.994 (344.094, 157.899) | 4.006e-3, 1.172e-2 | 0.666443 | 96 / 32 / 160 | the collar seed |
| 1 | 499.953 (341.238, 158.715) | 3.966e-3, 1.184e-2 | 0.658172 | 97 / 33 / 163 | stage 1 committed a timelike cone-in on the torus-B face (20, 23, 24): a fresh apex 32 with three edges at ℓ² = −1, ΔF = −2.04 (the Regge term −2.86, Γ·r_U +0.82); stage 2 then accepted no step (no edge moved by more than 1e-6 in 12 iterations; stationary at 1e-30) |
| 2 | 499.953 (341.238, 158.715) | 3.966e-3, 1.184e-2 | 0.658172 | 97 / 33 / 163 | no F-lowering move among 6 candidates; stage 2 stationary; exit |

After the cone-in the boundary of W is T_A and a 34-face, 17-vertex surface
of Euler characteristic 0 that is no longer inside block B (`boundary`
channel: `block: None`), `bridge_phase_complete` is false with no uncovered
torus face, while block B's own surface — the faces it carries from seeding
— is still its torus (16 vertices, 48 edges, 32 faces), so its residual
and its frame are still read on it. The coned torus face is interior now
(two cells on it). Betti numbers [1, 2, 1, 0] and the identity monodromy
(rounding 5.1e-15, fit 3.8e-16) are unchanged by the cone-in. The τ̂ reads
are still exactly τ_in (no length moved), the transfer
[[−0.02886, −0.04006], [−0.02071, −0.05206]] + i·(≤ 0.020) with Schmidt
spectrum (7.757e-2, 9.247e-3) moved only through the new cells.

Whether stage 2 descends on the pristine 4×4 collar — before any move — is
answered by the probe `probe-4x4-stage2.log` (`probe_stage2.py`: the node
built by the driver's own `build_qubit_node` at the chosen weights, then
`run_stage2` alone, 12 iterations, tolerance 1e-30). It does: 12 accepted
steps in 10 min 49 s, objective 501.994 → 108.832 (Regge 344.094 → 103.651,
Γ·r_U 157.899 → 5.180), block residuals 4.006e-3 / 1.172e-2 → 3.217e-4 /
1.412e-4, two-body leak 0.666443 → 0.551256, max |Im ℓ²| 3.75e-2, no
timelike edge, not yet stationary. So the 4×4 collar relaxes as the 3×3
one does, and it is the timelike cone-in committed ahead of stage 2 in
unit 1 — three edges at ℓ² = −1 on a torus face — after which the line
search finds no descent at all. Whether that is the objective's landscape
with timelike edges or a gradient defect on them is not decided by these
records; a second 4×4 draw with `--seed 8` settles it for that draw.

`synth-4x4-w1e4-regge-on-tol30-seed8` (the same command with `--seed 8`,
23 min 35 s, 3 of 4 units): stage 1 committed nothing in any unit and
stage 2 relaxed the collar — unit 1 reproduces the probe to the last digit
(objective 108.832), unit 2 reaches 108.319 and unit 3 changes nothing:

| unit | objective (Regge, Γ·r_U) | block residuals A, B | coefficients A, B | two-body leak | Schmidt | τ̂_A, τ̂_B (d_WP) |
|---|---|---|---|---|---|---|
| 0 | 501.994 (344.094, 157.899) | 4.006e-3, 1.172e-2 | (0.9982 + 0.0023i, 0.2980 + 1.0927i), (0.9976 + 0.0056i, −0.2006 + 0.7785i) | 0.666443 | (7.302e-2, 9.274e-3) | τ_in, τ_in |
| 1 | 108.832 (103.651, 5.180) | 3.217e-4, 1.412e-4 | (0.9992 + 0.0003i, 0.2997 + 1.0996i), (0.9995 − 0.0006i, −0.1999 + 0.8000i) | 0.551256 | (3.300e-1, 5.215e-4) | −0.3138 + 1.1038i, −0.5388 + 0.9944i |
| 2 | 108.319 (103.344, 4.975) | 3.120e-4, 1.307e-4 | (0.9992 + 0.0003i, 0.2997 + 1.0996i), (0.9995 − 0.0005i, −0.1998 + 0.8000i) | 0.548297 | (3.300e-1, 1.302e-3) | −0.3145 + 1.1038i (0.551), −0.5399 + 0.9948i (0.436) |

The 4×4 collar behaves as the 3×3 one: the block residuals fall 13× and
90× with the coefficients at (1, τ_in) to 8e-4, the two-body leak descends
0.666 → 0.548, the Regge term 344 → 103, the lengths go complex (max
|Im ℓ²| 3.8e-2, none timelike), ∂W stays the two tori, the monodromy stays
the identity (rounding 4.4e-15), and the tori's own τ̂ drift further than at
3×3 (d_WP 0.55 and 0.44). A 4×4 unit costs 8–12 minutes.

## C5. Headless and live

Headless: every run above rendered its GIF (`--out`) and wrote its record
(`--json`) through `render`; every channel of spec S6 is present on every
frame of every record, and the channels the qubit mode does not read are
`Absent` with their reason (the neutral mode's certificate channels: "not
read in the qubit input mode: the baryon certificates are the neutral
mode's instrument"; the dual curvature: "drawn from the triangle hinges of a
4-dimensional host; the qubit host is 3-dimensional"). The existing run
mode is covered by `tests/cobordism/test_emergence_animation.py` and by
T4's `test_the_neutral_mode_is_the_default_and_keeps_its_record`, both run
in this session (`tests-required.log`: 58 passed, 39 subtests).

`--live` (`live-attempts.log`, the chosen weights, `--steps 1`):

1. Under the process's default backend — `agg`, since this session has no
   X display (`DISPLAY` empty, `WAYLAND_DISPLAY` set) and the venv has no
   GUI toolkit — the driver refuses by name before computing anything:
   "--live needs an interactive matplotlib backend; this process has 'agg',
   which renders to files and shows no window … Set MPLBACKEND to an
   interactive backend (webagg needs no display), or drop --live" (exit 1,
   0.5 s).
2. Under `MPLBACKEND=webagg`, the driver's own suggestion: the venv built
   from `[dev]` has no tornado, so matplotlib refuses ("The WebAgg backend
   requires Tornado").
3. After `pip install tornado` into the throwaway venv, under webagg: the
   WebAgg server starts ("Press Ctrl+C to stop WebAgg server", a browser
   tab is opened on the session), frame 0 is drawn, the worker thread runs
   the unit and prints its step line, and the main thread never returns:
   `drive_live`'s `plt.pause(0.001)` is, under webagg, `plt.show()`, whose
   tornado loop runs forever (the faulthandler stack in the log:
   `drive_live` → `pyplot.pause` → `pyplot.show` → `backend_webagg.start`
   → `run_forever`). No frame after the first is drawn, and `--json` and
   `--out` are not written; the run was aborted at 300 s. So the live path
   is refused correctly without a display, and the backend the refusal
   names as needing no display does not carry it: a finding on the driver,
   not worked around here.

## The phases

Run (iv): the tori of the chosen-weights run under one pure gauge over all
18 host vertices (`--gauge-seed 964`, g uniform in (−π, π]), the same seed
7, the same tolerance 1e-30. The two markings' base points are host
vertices 0 (torus A) and 9 (torus B), so the base-point factor is
exp(i(g(0) − g(9))) = 0.37536 − 0.92688i. The phased run made the same (no)
stage-1 moves and the same stage-2 trajectory as its twin, exiting after 3
units at the same objective 30.986423; per unit, the defects against the
unphased run (`compare_phased.py`):

| unit | block residuals A, B | coefficients A, B | transfer, after the factor (relative) | two-body leak | monodromy, after the inverse factor | τ̂_A, τ̂_B | Regge term |
|---|---|---|---|---|---|---|---|
| 0 | 1.2e-16, 1.7e-16 | 2.2e-15, 1.6e-15 | 3.3e-14 | 3.4e-15 | 2.1e-15 | 1.1e-15, 1.2e-15 | 0 |
| 1 | 4.9e-17, 1.8e-15 | 1.1e-14, 2.9e-14 | 1.1e-11 | 1.0e-12 | 2.9e-15 | 1.7e-13, 1.2e-12 | 1.6e-10 |
| 2 | 3.6e-15, 5.2e-17 | 5.1e-14, 1.9e-14 | 9.1e-12 | 1.4e-12 | 4.8e-15 | 1.8e-13, 1.0e-12 | 1.1e-10 |
| 3 | 3.6e-15, 5.2e-17 | 5.1e-14, 1.9e-14 | 9.1e-12 | 1.4e-12 | 4.8e-15 | 1.8e-13, 1.0e-12 | 1.1e-10 |

The block residuals, the coefficients, the two-body leak and τ̂ are gauge
invariant; the transfer and the monodromy are invariant up to the
base-point factor, as T2-bis (c) states. The phases stay on all 90 edges
through the run (stage 1 committed nothing; stage 2 does not move degree-1
phases). One consequence for the read-out: under the gauge the monodromy
read returns M = (0.3754 − 0.9269i)·I, so its integer part `rounded` is
[[0, 0], [0, 0]] with rounding residual 1.0 and the driver's stdout line
prints "monodromy [[0, 0], [0, 0]]" — the integer read is defined for two
markings whose base points share a gauge, and needs the factor divided out
otherwise. The fit residual stays at 3e-16.

## The block residual of D2 as reworded (#988, #989)

Everything above C4 was measured under the previous D2, where the block
residual held the zero mode of the *entire* cobordism at each torus's input
coefficients and the torus's own conformal structure was free (the d_WP
column of C2, up to 0.44). On 2026-09-07 the owner restored the original
reading: **each torus keeps representing its input state through the zero
mode of its own Laplacian; the whole's zero mode is the output state, read
and never held.** The engine now scores

    r = 1 − |⟨ψ(τ_in)|ψ(τ̂)⟩|² = sin²(d_FS)

with τ̂ the ratio of the transported periods of the holomorphic 1-form of the
block's own Laplacian on its live surface (`MultiCobordism::blockQubit`,
`ownStateResidual`), ψ(τ) = (1, τ)/√(1+|τ|²). The previous quantity survives
unchanged as the output-state read (`inputStateResidual`, `readInputState`,
the `output_leak` channel of the animation).

Measured on this branch (3×3 collar, `OMP_NUM_THREADS=8`):

| quantity | value |
|---|---|
| block residual on the collar seed | 2.2e-16 and 0.0 |
| τ̂ on the seed against τ_in | agrees to 1e-14 |
| output-state read's leak on the seed | 3.100e-3 and 9.345e-3 (unchanged) |
| residuals after a 5% jitter of the tori's edges | 4.856e-4 and 1.855e-4 |
| the same after three stage-2 steps (weight 1e4, no Regge term) | 8.7e-9 and 4.7e-7 |
| τ̂ after those steps | 0.29979 + 1.10001i and −0.19891 + 0.79964i |
| gradient support | exactly the 27 edges of each torus, zero on the 36 bulk edges |
| Euler identity Σ Re(z_e) g_e | below 1e-19 (τ is scale-free) |
| gradient against central differences | agrees to 3e-8, the difference's own accuracy |
| gradient under a pure gauge on every host edge | unchanged to 1e-12 |

The seed is the residual's global minimum, so C2 cannot ask it to descend
from the seed any more; what it asks is that the torus still represents its
input after synthesis, which the residual measures directly and which the
identity r = sin²(d_FS) ties to the reported qubit read.

**A pre-existing discontinuity this exposes.** The residual is a real
function of τ̂, which is holomorphic in the squared lengths, so its descent
direction always has an imaginary component. The engine's Regge stationarity
term is *discontinuous* across the real locus: measured on origin/main
(85b4a94), an imaginary displacement of 1e-12 in the squared lengths moves it
from 123.1236 to 121.6405, and the limit depends on the signs of Im z_e, not
their size. Along the new descent direction the jump is upward (+15.6), so
with the Regge term on and the input weight at 1e6 every one of the line
search's 24 halvings is rejected and the drive is stationary at the seed. At
weight 1e4 — the chosen weight of this note, now the driver's default — the
Regge direction dominates the step and the drive descends normally
(123.65 → 119.60 → 78.57 over two units, both residuals below 1e-3). The
discontinuity is engine behaviour independent of this change and is filed
separately as #991.

**The real locus avoids it.** A node built with `realSquaredLengthsOnly`
projects the imaginary part of every trial away, so the step never leaves the
real locus and never meets the jump. There the drive runs at weight 1e6 with
the Regge term on: twenty stage-2 steps take the objective from 123.6497 to
122.4027 (the Regge term 123.1236 to 121.6539), the own-state residuals stay
at 2.0e-7 and 6.2e-10, τ̂ stays within 1.0e-3 of τ_in, and the output-state
read moves only from 3.100e-3 / 9.345e-3 to 3.094e-3 / 9.302e-3. That is the
D2 reading in one run: the tori keep their own states while the bulk relaxes,
and the whole's zero mode is reported rather than held.

## Where this leaves the experiment

Answered by the records:

- C1 holds: the collar seed is a manifold with ∂W = T_A ⊔ T_B, Betti
  numbers [1, 2, 1, 0], the identity monodromy to 1e-15 at 3×3 and 4×4,
  and the bridge round trip is bit-exact (T1's test re-run here).
- C2 held under the D2 of the time: the block residuals fell from
  3.1e-3 / 9.3e-3 to 2.6e-5 / 2.6e-5 (weight 1e4) or 3.7e-7 / 2.0e-7
  (weight 1e6) and the whole's coefficients in the live frames sat at
  (1, τ_in) to 1e-4–1e-5, reported every frame, while the tori's own τ̂
  moved away from τ_in (d_WP up to 0.44 at 3×3, 0.55 at 4×4). Under the
  D2 of #988 that drift is what the residual forbids: see the section
  above, where the seed reads its residual at 2e-16 and a jittered torus
  is driven back to τ_in.
- C3 holds for the trivial monodromy: the identity at every frame. No
  drawing with monodromy M ≠ 1 arose, so the transformation of the periods
  by M was not measured beyond T1's remarking check on the seed. The
  transfer is not diagonal (max-off / max-diagonal 1.22 → 1.16–1.36 at 3×3,
  0.76 at the 4×4 seed) and grows eightfold under relaxation.
- C4: the two-body leak descends under synthesis only when the two-body
  term is not dwarfed by the block terms — weight 1e4 (0.526 → 0.391 at
  3×3, 0.666 → 0.548 at 4×4) — and rises at the default weight 1e6 for
  every Γ in {1, 10, 100} and with the Regge term off. Its floor at 3×3 is
  0.391 with Schmidt spectrum (2.35e-1, 4.99e-2) against χ's
  (0.580, 0.420): the second channel of the transfer is 21 % of the first
  where χ's is 72 %.
- C5: headless runs render every channel; `--live` refuses by name without
  an interactive backend (below).
- The pure gauge leaves the block residuals, the coefficients, the two-body
  leak and τ̂ invariant (≤ 5e-14) and the transfer and the monodromy
  invariant up to the base-point factor (≤ 1e-11), through a whole run.

Open, from the records:

- **Stage 1 committed no move on the 3×3 collar in any run** (6 candidates
  per unit, depth 1, 3–8 units) and none on the 4×4 collar with seed 8; the
  one move committed (4×4, seed 7) was a timelike cone-in on a torus face,
  after which stage 2 accepted no step at all while the same collar relaxes
  fivefold without it. The synthesized W is therefore the collar with
  relaxed lengths, its topology unmoved, and the "drawn bulk" of spec S3–S4
  did not emerge under the driver's draw. Whether a wider draw
  (`--stage-one-iterations`, `--surgical-depth`) or the objective's balance
  changes that, and whether the stall after the timelike cone-in is the
  Lorentzian landscape or a gradient defect on ℓ² = −1 edges, are for the
  next ticket.
- **The relaxation goes stationary early**: the total objective admits no
  descending line-search trial after 2–4 units (at tolerance 1e-30 exactly
  as at 1e-12), while the r_U ascent alone is far from zero (norm 88–1145
  at 3×3, 359 at 4×4). The two-body leak's floor, 0.39, is that stationary
  point's, not χ's.
- **The record has no per-frame cell count or geometry** (the driver's
  `to_json` carries neither the layout nor the lengths); the harness's
  diagnostics sidecar filled that here.
- **The monodromy's integer read under a gauge** carries the base-point
  factor of the two markings (M = e^{i(g(v_B) − g(v_A))}·I, `rounded`
  [[0, 0], [0, 0]], rounding residual 1.0), so the integer part is only
  defined once that factor is divided out or the base points share a
  gauge.
- **Two knobs are not on the CLI**: Γ and the phases of run (iv); both were
  driven through `t5_run.py` on top of the driver's own factory and loop.
- Recursion (spec S7) was not attempted.

## The commands

From the worktree root, `R=~/cobordism-runs/qubit-cobordism/t5`, the venv
`/tmp/venv-t9` (origin/main fb58820 built with `TESSERA_CUDA=0`),
`OMP_NUM_THREADS=8`, at most two runs at once; every command with its
units and wall time is in `$R/runs.md`.

    # the collar seed alone (3x3; --grid 4 for 4x4)
    python examples/cobordism/emergence_animation.py run --inputs qubit \
      --tau-a 0.3+1.1j --tau-b=-0.2+0.8j --grid 3 --steps 0 \
      --out $R/seed-3x3.gif --json $R/seed-3x3.json

    # the weight scan: --input-weight {1e4, 1e6} x {--regge, --no-regge}
    python examples/cobordism/emergence_animation.py run --inputs qubit \
      --tau-a 0.3+1.1j --tau-b=-0.2+0.8j --grid 3 --steps 8 \
      --input-weight 1e6 --regge \
      --out $R/scan-w1e6-regge-on.gif --json $R/scan-w1e6-regge-on.json

    # the same drive through the harness (diagnostics; --gamma G; --gauge-seed S)
    python $R/t5_run.py ~/qubit-cobordism-run [--gamma 10] [--gauge-seed 964] -- \
      --inputs qubit --tau-a 0.3+1.1j --tau-b=-0.2+0.8j --grid 3 --steps 8 \
      --input-weight 1e4 --regge --tolerance 1e-30 \
      --out $R/chosen-w1e4-regge-on-tol30.gif --json $R/chosen-w1e4-regge-on-tol30.json

    # the phased run against its twin
    python $R/compare_phased.py $R/chosen-w1e4-regge-on-tol30.json $R/phased-w1e4-regge-on-tol30.json

    # stage 2 alone on the pristine 4x4 collar
    python $R/probe_stage2.py ~/qubit-cobordism-run 4 1e4 12

    # the tables of this note
    python $R/scan_table.py $R/<run>.json
    python $R/note_tables.py scan $R/scan-*.json
    python $R/note_tables.py trace $R/chosen-w1e4-regge-on-tol30.json

The pinning test: `OMP_NUM_THREADS=2 python -m pytest -q
tests/cobordism/test_qubit_cobordism_run_python.py`.
