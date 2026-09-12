# Qubit cobordism — standalone spec (follow-up to #955)

This document is the single source of truth for the experiment that puts the
simplicial qubit (`simplicial_qubit_spec.md`, implemented in
`observables::SimplicialQubit`, PR #957) into the cobordism engine. It is
written to be picked up from scratch: read it before touching code, audit the
work against it before and after, and treat every "Do not" as binding. Where
it names engine facts, they were verified in the tree at `origin/main`
`d8d87ec` (2026-09-05); re-verify line-level details, not the facts.

## 1. What is being built, in one paragraph

Two qubits are two 2-tori with complex edge lengths and pure-gauge link
phases (holonomy trivial on both cycles, so the kernel keeps its two
dimensions). Each torus's degree-1 zero
mode (the kernel of its own Laplacian), normalized by its marking, is that
qubit's frame |0⟩, |1⟩; a state at a torus is a pair of coefficients in that
frame, and the torus's input state (1, τ_in) is generated once by the
flat-torus construction. The tori are the boundary of a 3-complex W. The bulk of W starts as the *collar* between them
— the minimal manifold connecting the two boundaries, created as one gated
whole (S3) — and is then synthesized by the engine's stage 1 (combinatorial
moves, the gated `bridge` move among them) and stage 2 (length relaxation)
against the algebraic two-body target.
The output state is the degree-1 zero mode of the Laplacian of the **entire**
W, bulk and boundary edges together. Throughout, each torus keeps
representing its input state through the zero mode of its **own** Laplacian
(a residual in the objective). Stage 1 preserves the declared boundary
geometry, and the driver pins its input regions during stage 2 by default so
the fitted degrees of freedom are in the bulk; `--no-pin-boundary` requests
free boundary relaxation. Pinned boundary edges remain in the whole's
Laplacian and all read-outs. The canonical qubit entrypoint is
`examples/cobordism/qubit_animation.py`. It owns the qubit construction,
configuration, read-outs, and panels while reusing the stage loop, live and
headless rendering, output handling, and other mode-neutral runtime from
`examples/cobordism/emergence_animation.py`. There is one shared runtime, not
two implementations of the drive.

## 2. Vocabulary (use these words, no others)

- **State**: a zero mode (harmonic chain/form) of a Laplacian. Never a
  coordinate vector chosen by hand, never a cochain with coefficients
  constructed outside the geometry. At a block, a state is represented by
  its coefficients in the block's frame (below).
- **Qubit torus**: a closed oriented triangulated 2-torus with real positive
  edge lengths and a marking (A, B), A·B = +1, as in the qubit spec. Its
  input state is its holomorphic line ω (the −i eigenline of the complex
  structure on its harmonic space), whose coefficients in the torus's frame
  are (1, τ), τ = P_B/P_A; the marking exists so the frame and τ are defined.
- **Input block**: a boundary component of W carrying an input qubit torus.
  A block is a vertex set plus the fiber it carries (engine `BoundaryBlock`).
- **The whole**: W with every edge, boundary included, in one Laplacian. The
  output state is its zero mode.
- **Own Laplacian of a block**: the Laplacian of the block's surface, the
  2-complex of its own triangles with the host's current lengths.
- **Frame of a block**: the two kernel elements of the block's own Laplacian
  normalized by the block's marking to periods (1, 0) and (0, 1): |0⟩ and
  |1⟩ for that block (`SimplicialQubit::periodFrame`), recomputed from the
  block's live surface as its cells and lengths move. The coefficients of a
  kernel vector in this frame are its periods over A and B. The periods that
  normalize the frame and read the coefficients are taken with parallel
  transport along A and B; the ratio τ and the coefficient pair are
  gauge-invariant.
- **State at a block**: the coefficients (a, b) of a vector written in the
  block's frame; those coefficients are the representation of the state. The
  input state of torus A is (1, τ_A).
- **Fiber**: engine `BoundaryFiber`: a degree, a list of k-cells, and image
  columns on those cells. At degree 1 the cells are edges.
- **Leak / fiber residual**: the least-squares residual of a fiber's images
  inside the zero-mode band of a pencil restricted to the fiber's cells.
- **Bridge**: a top cell whose vertices are split across two blocks
  (tetrahedra: 1+3, 2+2, 3+1), created on existing vertices only.
- **Collar**: the minimal manifold connecting the two blocks, T²×I over their
  shared triangulation (`Spacetime::prismCells`); the seed of W.
- **Gate**: the manifold check `ChainComplex::dualComplexIsValid` (facet
  coface counts in {1, 2}, ridge links paths or cycles, vertex links disks or
  spheres), the only thing that decides whether a move may be applied.

## 3. Rules from the owner (binding; quotes from 2026-09-05)

R1. The output state is the whole's zero mode: "the output/target state is the
    zero mode of the ENTIRE cobordism, bulk + boundary." Restriction to a
    boundary component is a *read-out channel*, never the definition.

R2. Fixed does not mean excluded: "the boundary … is what stays fixed. but
    staying fixed does NOT exclude it from the output state readout of the
    laplacian harmonic over the ENTIRE cobordism (bulk + boundary)."

R3. **Current fixed-boundary rule (revised by #1022).** The boundary is a
    declared input: stage 1 may not alter its facet set or geometry, and the
    driver pins its regions during stage 2 by default. `--no-pin-boundary`
    retains the earlier free-relaxation experiment. In either mode the block's
    own-Laplacian residual stays in the objective beside the bulk terms, and
    the boundary stays in the whole's Laplacian and read-outs.

R4. The bulk is drawn, not templated: "choose a vertex on one of the boundary
    blocks. cone it into 4 vertices on the other. or choose 2 and 3 or 3 and 2.
    you just have to continue asserting the manifold condition as a gate."
    (For tetrahedra the splits are 1+3, 2+2, 3+1.)

R5. Do not split implementations: reuse the fiber machinery ("why do we need
    to replace the fiber machinery?"), reuse MultiCobordism, and run the qubit
    example through the shared animation runtime from
    `emergence_animation.py` ("run this as an animation in the same way"). The
    dedicated `qubit_animation.py` entrypoint separates qubit-specific policy;
    it does not copy the drive, live worker, or renderer.

R6. States are zero modes, not cochains with hand-made coefficients ("it
    sounds like you might be slipping into using cochains (with extra
    machinery for coefficients) and skipping using the laplacian harmonic as
    the states").

R7. The zero mode, not the band above it (earlier: "i wanted the zero mode").

R8. Program-wide rules that still apply: hosts and bulks are emergent beyond
    the collar seed of S3 (no template beyond the collar between the given
    surfaces); no runtime guards or clamps in the
    dynamics, only configuration-space gates (the manifold check) and
    variational acceptance (ΔF); Lorentzian machinery is the default.

## 4. The construction, step by step

S1. **Inputs.** For each input state τ_in ∈ upper half plane, build the qubit
    torus with `SimplicialQubit.flat_torus(tau_in, n, n)` (exact: the read
    returns τ_in to rounding). Its edges, faces (counterclockwise), complex
    edge lengths and pure-gauge link phases (holonomy trivial on both cycles,
    so the kernel keeps its two dimensions), and cycles are its own data; its holomorphic form on its edges is the
    state fiber. Both tori must have Im τ > 0 (the representable hemisphere);
    the poles |0⟩, |1⟩ are limits reached by pinching, not inputs.

S2. **Host initialization.** One 3-dimensional `Spacetime` containing both tori
    as 2-simplices (their triangles, edges, vertices) with their lengths and
    zero phases, disjoint vertex id ranges, and no 3-cells yet. Each torus's
    vertex set is one input block (`seed_inputs`). Its holomorphic form's edge
    values are the degree-1 state fiber, attached with
    `attach_input_fiber` and the **harmonic contour**
    (`PencilLayer.harmonic_contour` / `band_contour(…, 0)`). Its cycles and
    input coefficients (1, τ_in) are attached as a marking with
    `set_input_marking`; the engine derives the live period frame from that
    marking at every read (D3), rather than holding a seed frame. The metric
    source is the Whitney pencil. The two-body target χ is set (S5), and
    `use_fiber_residuals(True)` puts the fiber and selected readout residuals
    in the objective.

S3. **The collar.** The bulk between the two surfaces starts as the minimal
    manifold connecting them: T²×I over the tori's shared triangulation,
    `Spacetime::prismCells` over the base faces with the surfaces' own
    lengths at the two ends and the engine's auto-wired length on the
    interior edges (`MultiCobordism::seedCollar`), created as ONE gated
    whole — the full manifold check (`ChainComplex::dualComplexIsValid`) on
    the result, refused by name if it fails; a pair of surfaces whose
    combinatorics differ is refused by name. `bridgePhaseComplete()` (every
    face of both tori has exactly one 3-cell on it and no other boundary face
    exists, so ∂W = T_A ⊔ T_B) holds by construction, and so does the no-chord
    condition: the sub-complex on each block's vertex set is exactly its
    torus. From here everything is emergent: stage 1 refines and surgers the
    interior, stage 2 relaxes every unpinned edge, and `bridge` is one more
    gated stage-1 move kind — a candidate top cell on existing vertices, k
    from one block and 4 − k from the other, applied through
    `SurgicalCone::bridge`, gated by the manifold check, scored by ΔF, offered
    while a face of a surface block is uncovered (after a cone-out dent, for
    instance). The per-cell drawing of the bulk from nothing is withdrawn
    (§6): a cell the
    gate accepts meets the complex along a disk, so such a drawing stays a
    ball and never reaches ∂W = T_A ⊔ T_B. Record the Betti numbers and the
    monodromy (S6) of the seed ([1, 2, 1, 0] and the identity for matched
    markings) and again after every frame: the topology of the synthesized W
    is emergent from there. Stage 2 relaxes the bulk edges by default; with
    `--no-pin-boundary` it relaxes the boundary edges as well.

S4. **Synthesis.** Per frame: ordinary stage 1 (adds, flips, cone-outs,
    cone-ins with fresh vertices, and `bridge` while a surface face is
    uncovered), then stage 2 on every unpinned edge. The driver pins the input
    regions by default and `--no-pin-boundary` leaves every edge free. The
    objective is the engine's: Regge stationarity plus Γ·r_U,
    where r_U under fiber residuals is the sum of each block's own-Laplacian
    fiber residual (weight `inputResidualWeight`), the whole-complex fiber
    residual if a whole-complex target is set, and the two-body residual.
    Stage 1 runs before stage 2 within a frame (a committed stage-1 move
    rebuilds the complex from a snapshot of its cells and every edge's
    length and phase).

S5. **Target.** The two-qubit XY flip-flop, mirroring the spin-3/2 experiment:
    H_int = ħJ(σ₁⁺σ₂⁻ + σ₁⁻σ₂⁺), first-order amplitude A = −iJt χ with
    χ = (σ⁻ψ)(σ⁺φ)ᵀ + (σ⁺ψ)(σ⁻φ)ᵀ in the |0⟩, |1⟩ bases of the two qubits
    (ψ = (1, τ_A)/√(1+|τ_A|²), φ likewise), exact evolution by the
    total-magnetization blocks (sizes 1, 2, 1). χ is the two-body target
    (`set_two_body_target(chi, choi_decomposed)`), compared with the transfer
    read in the tori's period frames (S6) by the engine's projective leak.

S6. **Read-outs, every frame.**
    - Per block: its own-Laplacian fiber residual; the qubit read
      (`SimplicialQubit(surface, cycle_A, cycle_B)`) on the block's surface
      with the live lengths: τ̂, d_FS and d_WP to τ_in, Delaunay flags, the
      spec's J residual. The marking is the flat torus's cycle pair; its edges
      persist under every engine move (see §6), so the read is always defined.
    - The whole: Betti numbers, boundary components, completion status, the
      rank of the degree-1 harmonic band, the coefficients (a, b) of the
      whole's zero mode in each block's live frame, next to the input
      (1, τ_in) — the output state read at that torus, the transfer T in the
      period frames
      (2×2), its projective leak against
      χ, the Schmidt spectrum and rank (Choi flag), the monodromy: the
      integer matrix relating the two markings through the whole's zero mode,
      M = P_B P_A⁻¹ from the periods of the whole's harmonic forms over the
      two tori's cycles (an element of SL(2, Z) when W is an I-bundle).
    - The objective and its terms.

S7. **Recursion.** As in `recursion_as_propagation.py`: the whole's zero mode
    read on the tori (period coordinates in their markings) is the next
    layer's input; `velocity` = fresh node per layer, `extend` = the same
    cobordism continued. Nothing new is designed here until S1–S6 run.

## 5. Engine deltas (all additive; nothing else changes)

D1. **Bridge primitive and collar seed** (`cobordism::SurgicalCone`, next to
    `coneOut`/`coneIn`; `MultiCobordism::seedCollar`, next to `seedSimplex`):
    `bridge` creates the top cell on d + 1 existing vertices, auto-wiring
    missing edges with the engine's auto-wired length, gated with
    `dualComplexIsValid` and nothing else, rolled back bit-exactly (remove the
    cell and every edge it alone introduced). The seed of W is the collar of
    S3, one gated whole; there is no drawing search. Stage 1 draws bridge
    candidates from vertex splits across the two input blocks while any
    torus face is uncovered (a move kind, scored by ΔF like `cone_out`; the
    split is taken inside the blocks' own simplices, so the no-chord
    condition of S3 holds by construction), and `bridgePhaseComplete()`
    reports ∂W = T_A ⊔ T_B. `seedFromSurfaces` seeds the bare surfaces (no
    3-cell) for the primitive's own tests.

D2. **Block surface complex.** `blockSubcomplexWithGeometry` extracts the top
    cells inside a block's vertex set, which is empty for a surface block in a
    3-complex. For a block whose fiber degree is below the host dimension, the
    block's own complex is the 2-complex of its own triangles inside its
    vertex set, with the host's lengths and phases. The block's own
    Laplacian supplies its frame (§2). The per-block fiber residual and its
    gradient then read the torus's own Laplacian, which is R3: the residual
    is 1 − |⟨ψ(τ_in)|ψ(τ̂)⟩|², where τ̂ is the ratio of the periods, taken
    with parallel transport, of the holomorphic 1-form of the block's own
    Laplacian on its live surface over the marking cycles B and A, and
    ψ(τ) = (1, τ)/√(1+|τ|²); it is zero exactly when the block's own
    Laplacian represents the input state, and its gradient is analytic
    through the dependence of that 1-form on the block's lengths and phases.
    The least-squares residual of the input 1-form's edge values in the
    block's own kernel is not this residual: that kernel contains them at
    every step, so it is near zero for every state (measured on T4: 1e-9
    while τ̂ moved).

D3. **Transfer in period frames.** `frameTransferOn` uses identity frames on
    the fibers' cells (an edge-by-edge block at degree 1). Each block supplies
    its marking, the cycles A and B. Its frame is derived by the engine from
    the marking and the block's live kernel: the kernel elements normalized
    to periods (1, 0) and (0, 1) with parallel transport, with their dual
    images under the block's own pencil, recomputed at every read (§2). The
    transfer is read in those frames (2×2), which is what χ is written in.
    The state fibers stay rank one; the marking is separate data.

D4. **Animation.** `examples/cobordism/qubit_animation.py` is the canonical
    qubit entrypoint:
    `qubit_animation.py run --tau-a --tau-b --grid --J --time`. It has no
    `--inputs qubit` mode switch. It owns the qubit node factory and config,
    uses `--pin-boundary` by default with `--no-pin-boundary` as the opt-out,
    and supplies frame channels for the S6 read-outs, panels for the residual
    traces, the two τ̂ trajectories on the upper half plane and the Bloch
    hemisphere, the transfer versus χ, and the drawn boundary highlighted in
    the layout. It reuses the mode-neutral drive, live worker, renderer, and
    output plumbing from `emergence_animation.py`, so headless and `--live`
    paths remain the same loop. Records go under
    `~/cobordism-runs/qubit-cobordism/` (never `/tmp`).

    The run document's `config` object is mode-specific after #1072: qubit
    records contain common scheduling keys and qubit keys, but not the inert
    neutral `size`, `host_seed`, `resolution`, or `edge_disposition` keys.
    Neutral records likewise omit inert qubit keys. Frame measurements, the
    qubit `inputs` object, and geometry schema 1 are unchanged.

    The driver accepts the two-torus `transfer` read and the two-torus `whole`
    read with an explicit `--output-state`. It refuses `bulk` without an
    explicit interior Choi frame. It also refuses the four-torus `whole` and
    `operator` reads: concatenating torus period frames gives a direct sum,
    while a two-qubit target uses tensor-product coordinates; equal rank four
    does not identify those spaces. The engine's paired 4x4 transfer remains a
    low-level geometric diagnostic only.

## 6. Engine facts to rely on (verified 2026-09-05)

- Fibers are degree-generic; cells are any k-cells of the live complex, edges
  included; a listed cell that does not exist gives the full leak 1.0.
- The default fiber contour is band 1 (above the zero mode); the harmonic
  contour must be set explicitly on every fiber and read (R7). A surface
  block's own-Laplacian read (D2, `MultiCobordism::blockSurfaceWithGeometry`)
  takes the zero mode of the block's OWN pencil (`harmonicContour` on its
  surface, recomputed at every read) whatever contour its fiber stores; the
  stored contour governs whole-complex reads only.
- A surface block carries its own faces from seeding (`BoundaryBlock::faces`):
  a cone-out dent uncovers a torus face and the host's orphan prune drops the
  face's registration, so the block — not the host — is what says which
  triangles are the surface, and `uncoveredInputFaces` reads them. On a
  one-layer collar no cell can be dented at all: every (3,1)/(1,3) cell's apex
  is a boundary vertex of the other torus (removing an interior triangle of
  its disk link punctures it) and every (2,2) cell sits mid-path in a torus
  edge's link; a two-layer collar's torus-adjacent cells, whose apex is
  interior, can be. The engine's moves never remove a surface vertex or edge,
  and a surface torn by any future move reads as the full leak (no own
  complex), which the ΔF acceptance rejects.
- All fiber and two-body paths require the Whitney pencil metric source.
- Under fiber residuals the engine skips the hole-forcing near-kernel term and
  demands no register count.
- Pinned regions only zero stage-2 descent on edges inside one region; they do
  not remove those edges from the whole's Laplacian or its read-outs. The
  driver registers each input torus as a pinned region by default; the engine
  remains capable of the free-boundary run selected by `--no-pin-boundary`.
- Stage 1 candidates: add, remove, flip, iflip, cone_out (random top cell),
  cone_in (fresh apex on a boundary face); the gate is the manifold check;
  Pachner moves preserve Betti numbers; a cone-out raises b₂, a cone-in
  lowers it. No existing move creates genus, which is why the tori are inputs
  and the bulk is drawn (R4).
- Under these moves a torus edge is never removed: cone-in on a face keeps
  the face's edges on the boundary; a cone-out dent keeps them; 3-2 flips act
  on interior edges; 4-1 removals act on interior vertices. So edge-indexed
  fibers and markings survive; the surface gains vertices and faces.
- A committed stage-1 move restores every surviving edge's length and
  phase; edges a move creates are auto-wired with a zero phase. Stage 1
  still runs before stage 2 in a frame.
- The zero mode of the whole restricted to the boundary is topological: half
  of the boundary's harmonic space extends into W. The bulk metric shapes the
  representative on the edges and the transfer block, not the periods. A
  collar-like drawing transmits τ unchanged; a drawing with monodromy applies
  a modular transformation. This is a fact to report, not a defect.
- The qubit read uses the spec's cotangent operator on the torus; the engine
  uses the Whitney pencil. On flat tori both harmonic spaces are the constant
  forms exactly; on a deformed torus they differ at mesh order. Report both.
- Analytic gradients exist for the fiber residuals and the transfer and
  whole-harmonic two-body readings (`BandDerivative`). Bulk and paired-operator
  readings have no analytic gradient; when either is selected, stage 2 uses
  the numerical ascent of the complete `rU` objective.
- Frames (D3, T3): a torus's period frame is `SimplicialQubit::periodFrame`
  (`harmonicBasis` times the inverse period matrix over the marking in
  force, real, n_E × 2, periods (1, 0) and (0, 1); the holomorphic form is
  P_A·F·(1, τ)ᵀ). The driver stores a `BlockMarking`; `deriveFrame` recomputes
  its `BlockFrame` from the block's live own kernel and marking at every read.
  `setInputFrame` remains the low-level alternative for an explicitly supplied
  frame, which is held until re-attaching the fiber clears it. With both input
  blocks framed by either route, `readTwoBody`, `twoBodyResidual` and the
  transfer gradient read T = (Z_A^∨)ᵀ Ã_AB Z_B in those frames
  (`TwoBodyRead::inFrames`); the transfer gradient selects the current frames
  at the evaluation point and differentiates the pencil operator. With neither
  block framed, identity frames on the cells are bit-identical to before; a
  partially framed pair is a contract error.
  `setTwoBodyTarget` checks χ's shape against the frames' ranks when both
  are present, the cell counts otherwise, once two fibers are attached. The
  dual-frame contract found in the code: `PencilSchur::transfer` pairs dual
  images against the pencil operator by the transpose and normalizes
  nothing, so the dual of a supplied frame is Z^∨ = Z·B^{-T} with
  B = Zᵀ M_1 Z the frame's pairing under the Whitney mass matrix of the
  block's OWN pencil, (Z^∨)ᵀ M_1 Z = I (`MultiCobordism::dualFrame`,
  `inputFrameDual`); under it a change of frame (g_A, g_B) sends T to
  g_A⁻¹ T g_B, the matrix of the operator block in the frames' coordinates.
  Measured on the collar seed with the period frames (C3): T is 2 × 2, real,
  and NOT diagonal — 3×3: [[−0.01592, −0.01936], [−0.00433, −0.00569]],
  Schmidt spectrum (2.6e-2, 2.6e-4); 4×4: [[−0.02871, −0.03901],
  [−0.02054, −0.05148]], (7.3e-2, 9.3e-3) — the whole's pencil-operator
  block between the two tori paired through the collar's cells, scaling as
  one inverse power of a common length scale (the projective residual is
  scale-invariant), its rows and columns permuted or negated exactly with a
  remarking of either torus. The identity monodromy is the WHOLE's zero mode
  read on both markings (a topological fact); the transfer is a metric block
  and is not the identity.
- Under the legacy objective with the Regge term on, stage 2 used to descend
  the Regge direction alone and only gate on Γ·r_U (a shortcut from the
  numerical-r_U era, kept for the period residual); under fiber residuals the
  context flag `ObjectiveContext::fiberResiduals` restores the r_U direction
  — the analytic fiber ascent at weight Γ — so the block residuals are
  descended next to the bulk term (R3, S4). The level a block residual settles
  at is the balance of `inputResidualWeight` against the Regge pull on the
  tori's edges, roughly 1/weight² (measured on the 3×3 collar after 40
  stage-2 steps in an explicit real-locus probe: 2e-9 at weight 1e6 with the Regge term
  123 → 84 and the bulk's edges moved; 1e-4–7e-4 at 1e3; 2e-2–4e-2 at
  weight 1, where the Regge term also drives torus edges timelike; T2's
  test); report the weight with every residual. The production driver uses the
  complex locus (`realSquaredLengthsOnly = false`): the torus construction
  carries complex lengths and the full complex gradient must remain available.
- GIL: `run_stage1`, `run_stage2`, `two_body_residual`, `read_two_body`,
  `whole_complex_fiber_residual`, `read_whole_complex_fiber` release it;
  `build_step`, `attach_input_fiber`, `read_output_fiber` do not.
- A bulk drawn one gated cell at a time from a single cell stays a ball: a
  tetrahedron the manifold gate accepts meets the complex along a disk (any
  two of its faces share an edge; a cell touching the complex anywhere else
  is a pinch the gate refuses), and a shellable 3-manifold is a ball, so
  ∂W = T_A ⊔ T_B is unreachable by such a drawing — even the prism cannot be
  built cell by cell under the gate. Measured on #960 before the collar was
  adopted: under the manifold gate per cell, a depth-first search over
  frontier-adjacent split cells (close-most-faces first, random ties, no
  buried vertex) completed 0 of 6 seeds within 3·10⁵ gated attempts each on
  3×3 vs 4×4 tori, and 0 of 80 greedy restarts on 3×3 vs 3×3 and 3×3 vs 4×4;
  under a facets-only gate (coface counts ≤ 2, no link checks) the same
  drawing completed 40 of 40 restarts within 50 attempts each — one cell per
  surface face and no 2+2 cell — and 0 of those 40 were manifolds (edge links
  disconnected). The collar is the seed because of this fact, not for
  convenience.

## 7. Do not

- Do not let stage 1 rewrite the declared boundary geometry, and do not exclude
  boundary edges from the whole's Laplacian. The driver pins the input regions
  during stage 2 by default; `--no-pin-boundary` is the explicit free-boundary
  experiment.
- No template beyond the collar between the given surfaces: the collar is
  the minimal manifold connecting the boundaries, and nothing more is
  templated; do not add a tube/genus move; do not glue external complexes.
- Do not add any distance-in-moduli objective other than the block residual
  of D2. The marking enters the relaxation only through the periods D2 reads.
- Do not define the output state by restriction; do not read states as
  period vectors in the relaxation.
- Do not leave a fiber on the default band-1 contour.
- Do not copy or fork the shared runtime in `emergence_animation.py`.
  Keep qubit-specific construction, configuration, reads, panels, and CLI in
  `qubit_animation.py`, and reuse the shared drive, live, rendering, and output
  machinery.
- Do not replace or bypass the fiber residual machinery, MultiCobordism's
  stages, or the two-body read.
- Do not widen core classes for one consumer beyond D1–D4; extend, never wrap.

## 8. Checks that decide whether it worked

C1. The collar seed is a manifold with ∂W = T_A ⊔ T_B, Betti numbers
    [1, 2, 1, 0], and monodromy the identity for matched markings; a bridge
    rolls back bit-exactly (a round trip leaves lengths and cells identical).

C2. Each block's residual is the projective leak of its input state against
    the holomorphic zero mode of that block's **own** surface Laplacian. It is
    at rounding on a seeded flat torus, remains there while the input boundary
    is pinned, rises under a controlled boundary-metric perturbation, and is
    driven back toward its floor when that boundary is allowed to relax. The
    whole-complex coefficients and optional leak are separate readouts,
    reported next to the inputs rather than substituted for this residual.

C3. With trivial monodromy the monodromy read is the identity; with
    monodromy M, the periods transform by M. The transfer in period
    frames is a different object — the whole's pencil-operator block between
    the two frames through the collar's cells, a metric quantity — and is NOT
    diagonal on the collar (measured on T3: off-diagonal/diagonal 1.22 on the
    3×3 collar, 0.76 on the 4×4). Record it; do not expect it to be diagonal.

C4. The two-body leak against χ decreases under synthesis; its floor and the
    Schmidt spectrum are recorded next to the χ of the algebra.

C5. `qubit_animation.py run` runs headless and `--live` with every channel
    present or `Absent(reason)`. Both paths use the shared runtime from
    `emergence_animation.py`. Its neutral drive behavior and frame/geometry
    schemas remain unchanged; D4 records the intentional config-key split.

## 9. Existing code to start from

- `examples/cobordism/two_body_xy_flip_flop.py`: single-seed growth, per-input
  fibers on frames, `set_two_body_target`, `read_two_body`, residual traces.
- `examples/cobordism/choi_encoding.py`: two prepared boundary components as
  inputs, the whole complex as output (its prism host is what R4 replaces).
- `examples/cobordism/qubit_animation.py`: the qubit node/config factory,
  qubit frame channels and panels, and the canonical qubit CLI.
- `examples/cobordism/emergence_animation.py`: the neutral emergence example
  and the shared drive, live, rendering, layout, output, and serialization
  infrastructure used by the qubit entrypoint.
- `include/cobordism/MultiCobordism.h`: `seedInputs`, `attachInputFiber`,
  `setWholeComplexFiberTarget`, `setTwoBodyTarget`, `readTwoBody`,
  `useFiberResiduals`, `runStage1`, `runStage2`, `fiberModeAscent`,
  `blockSubcomplexWithGeometry`, `blockSurfaceWithGeometry`, `FiberBand`,
  `frameTransferOn`, `seedCollar`, `seedFromSurfaces`, `blockSurface`,
  `bridgePhaseComplete`, `monodromy`, `setInputFrame`, `dualFrame`,
  `inputFrameDual`, `BlockFrame`.
- `include/cobordism/SurgicalCone.h`: `coneOut`, `coneIn`, `bridge`, rollback,
  gate.
- `include/cobordism/PencilLayer.h`: `BoundaryFiber`, `harmonicContour`,
  `bandContour`, `indicesOf`, `transfer`.
- `include/observables/SimplicialQubit.h`: `flatTorus`, `harmonicBasis`,
  `holomorphicForm`, `periods`, `tau`, `periodFrame`, the `Spacetime`
  constructor.
- `docs/design/cobordism.md` (historical MergeCobordism note): boundary
  components as the states, the whole as the output; its prism/icosahedron
  start and its period readouts are superseded by R1, R3, R4.

## 10. Ticket order

T1. (#960) Bridge primitive and collar seed (D1) with tests: gate, bit-exact
    rollback, the collar seed from two spec tori (a manifold with
    ∂W = T_A ⊔ T_B, Betti numbers [1, 2, 1, 0], identity monodromy), the
    no-chord condition, a mismatched pair refused by name, bridge on the
    collar.
T2. (#961, merged: the block surface and its frame) and T2-bis (D2 as
    revised): the block residual on the whole's zero mode in the block's live
    frame, with the frame's gradient; tests: collar value, floor after
    synthesis, coefficients reported.
T3. (#962) Transfer in period frames (D3) with tests: identity on a trivially drawn
    collar, χ comparison shape 2×2.
T4. (#963, historical implementation) Animation qubit mode (D4) with tests:
    the existing mode stayed unchanged and the qubit mode produced every
    channel headless. This originally landed as `--inputs qubit` in
    `emergence_animation.py`.
T5. (#964) The run: records, findings note `docs/design/qubit_cobordism_findings.md`,
    C1–C5 answered with numbers.
T6. (#1072) Separate the canonical qubit entrypoint into
    `examples/cobordism/qubit_animation.py` while continuing to use the shared
    runtime from the neutral `emergence_animation.py`; keep the old
    `--inputs qubit` spelling as a thin forwarding path without changing the
    experiment.
