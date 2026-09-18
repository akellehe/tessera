---
orphan: true
---

# Interaction-history Monte Carlo: emergent spacetime from mutual information

A Metropolis Monte Carlo that samples *interaction histories* of a set of
quantum systems, weighted by the geometric Regge action on the simplicial
complex those interactions build. Edge lengths come from mutual information,
`d = -log I`. The target of the search is the coupling at which the emergent
heat-kernel spectral dimension reaches 4.

The implementation is one C++ class, `InteractionSimulation`, shaped like
`CDTSimulation` and reusing the same simplicial machinery (`Spacetime`,
`Simplex`, `ReggeSolver`).

---

## 1. The construction

Quantum systems interact pairwise. Each interaction is an *event*: two
systems `A`, `B` interact and the event spawns a third worldline `AB`,
the interaction product, leaving `A'` and `B'` carried forward. The five
systems `{A, B, A', AB, B'}` are the vertices of a `(2,3)` 4-simplex —
two on the earlier time slice, three on the later one — and the history
of accepted interactions is a simplicial complex.

The mutual information between systems is an edge length, `d = -log I`, in the
van Raamsdonk sense. The geometric Regge action `S = Σ_h A_h ε_h` on the
mutual-information-lengthed complex weights the ensemble of interaction
histories. Which interactions occur is sampled from that ensemble by
Metropolis-Hastings rather than dictated.

As the inverse-temperature coupling `β` and the interaction parameters vary,
the emergent spectral dimension of the interaction-history complex passes
through a phase structure containing a locus where `D_S → 4`.

---

## 2. The initial layer

`N` quantum systems, each prepared in a known, randomized mixed state `ρ_i`
with `S(ρ_i) > 0`. The mixedness is required: the conservation law in §4 is
trivial for a pure system (`S = 0`).

The systems are Poisson-distributed in a 2D patch and Delaunay-triangulated.
The Delaunay edges are the `t = 0` spatial adjacency, and the Delaunay
triangulation is the Voronoi dual.

---

## 3. The interaction event

When `A` and `B` interact through a two-system unitary `U`, the
interaction product `AB` is the joint state of the two systems:

$$\rho_{AB} = U\,(\rho_A \otimes \rho_B)\,U^\dagger.$$

`AB` is a new node — a worldline created by the event — whose quantum content
is `ρ_AB`. There is no Choi isomorphism and no reference legs: `AB`, its
marginals, and the input states are all reduced-density-matrix quantities on
states that exist.

### 3.1 Factorizing the joint state

`ρ_AB` is factorized to extract the entanglement content. Any two-qubit
state can be written

$$\rho_{AB} = \tfrac14\Big(I + \sum_i a_i\,\sigma^A_i + \sum_j b_j\,\sigma^B_j + \sum_{ij} c_{ij}\,\sigma^A_i\sigma^B_j\Big),$$

and under local rotations the correlation matrix `c_ij` diagonalizes to
three invariants `\vec c = (c_1, c_2, c_3)`, the Cartan coordinates of the
state, which measure how much the interaction coupled the two systems. The
Bloch vectors `a_i`, `b_j` are the local content and peel off as `A'` and
`B'` (the marginals carried forward). `\vec c` is zero for a non-entangling
interaction and grows with the coupling.

---

## 4. Edge bookkeeping

The `(2,3)` cell `{A, B, A', AB, B'}` has ten edges, from two sources:
mutual informations on co-existing systems, and a conservation law for the
temporal edges.

### 4.1 Mutual informations

These are ordinary `I(X:Y) = S(X) + S(Y) - S(XY)` on systems that
co-exist in the one global state:

- `I(A:B)` — the input pair, before the interaction.
- `I(A':AB)`, `I(B':AB)`, `I(A':B')` — the output triple.
- `I(A:AB)`, `I(B:AB)` — the primary temporal quantities. `AB` is the
  joint state `ρ_AB`; `A'`, `B'` are its marginals. `I(A:AB)` is the
  mutual information in `ρ_AB` — `S(A') + S(B') − S(ρ_AB)` — how much the
  interaction correlated the two systems.

### 4.2 The conservation law

The remaining temporal edges close by conservation — information in equals
information out — on the six-edge interaction structure
`A→A'`, `A→AB`, `B→B'`, `B→AB`:

$$S(A) = I(A{:}A') + I(A{:}AB), \qquad S(B) = I(B{:}B') + I(B{:}AB).$$

`I(A:AB)` and `I(B:AB)` are the primary quantities (§4.1); `I(A:A')` and
`I(B:B')` are the residuals:

$$I(A{:}A') = S(A) - I(A{:}AB), \qquad I(B{:}B') = S(B) - I(B{:}AB).$$

No co-existence of `A` with `A'` is required: every input is a
single-system entropy or a mutual information on a state that exists. This
is how the construction avoids the no-cloning obstruction.

### 4.3 Edge lengths

Every edge length is `d = -log I` (van Raamsdonk distance), normalised
so `d ≥ 0`, with an `ε_I` floor. Same-slice edges are spacelike,
cross-slice edges timelike, in the CDT sense.

---

## 5. The Regge action and the partition function

The geometric Regge action on the mutual-information-lengthed complex is

$$S[C] = \sum_{h \in \text{hinges}} A_h\, \varepsilon_h,$$

with `A_h` the Heron hinge area and `ε_h = 2π - Σ θ` the deficit angle,
evaluated through `ReggeSolver`'s `hingeArea` and `deficitAngle` primitives.
There is no worldline matter term: the matter is in the mutual informations,
so a separate `S_matter` would double-count.

The partition function is over interaction histories reachable from the
initial layer:

$$Z = \sum_{C} \frac{1}{C_C}\, e^{-\beta S[C]},$$

with `1/C_C` the symmetry factor. `β` is the inverse-temperature coupling;
varying `β` maps the phase structure.

---

## 6. The Monte Carlo

The equilibrium ensemble is sampled with two moves:

- **`interact{X,Y}`** — pick a uniformly-random eligible frontier
  spatial edge (`X`, `Y` both on the frontier, with no out-edges), attach
  the `(2,3)` cell, spawn `AB`.
- **`unInteract`** — pick a uniformly-random *leaf* cell (all three
  products still on the frontier) and remove it.

A system may interact only while it has no out-edges, and `unInteract`
removes only leaf cells. Each system therefore interacts at most once, the
moves are reversible, and `N₊` (frontier spatial edges) and `N₋` (leaf
cells) are well-defined incremental tables.

Metropolis-Hastings acceptance:

$$A(C \to C') = \min\!\left\{1,\; \frac{N_+}{N_-}\cdot\frac{C_C}{C_{C'}}\cdot e^{-\beta\,\Delta S}\right\}.$$

`ΔS` is local — the new cell's hinge contributions — read off a
per-hinge action table. Volume is controlled by capping the interaction
count (the `T`-cap). The lifecycle is tune, then thermalize, matching
`CDTSimulation`.

---

## 7. Implementation

`InteractionSimulation` lives in `tessera::simulations` and is shaped like
`CDT`: constructed with the couplings and the initial layer, it exposes the
move primitives and their `propose*` counterparts,
`sweep` / `thermalize` / `tune`, and `computeAction` /
`getAcceptanceRates` / the observable getters.

Each system's state is a one-qubit density matrix, held in a per-vertex map
inside the class; it never crosses the language boundary. Single-system
entropies and the mutual informations of §4.1 are reduced-density-matrix
computations on those matrices and the joint state `ρ_AB` of each event.

---

## 8. Modelling choices

1. **`AB` is the joint state** `ρ_AB`, a single node, not a Choi-isomorphism
   construct. It connects through the ordinary edges `A→AB`, `B→AB`,
   `A'–AB`, `B'–AB`; `A→A'` and `B→B'` are `A`'s and `B`'s other out-edges.
   There is no internal multi-leg structure.
2. **`I(A:AB)` is the mutual information of `ρ_AB`** — `S(A') + S(B') −
   S(ρ_AB)` — and `I(A:A') = S(A) − I(A:AB)` is the residual. Every quantity
   is a reduced-density-matrix computation on a state that exists.
3. **Lifecycle: tune, then thermalize**, the `CDTSimulation` order.

---

## References

- M. Van Raamsdonk, *Building up spacetime with quantum entanglement*,
  [arXiv:1005.3035](https://arxiv.org/abs/1005.3035) — the `d ∝ -log I`
  relation.
- B. Kraus, J. I. Cirac, *Optimal creation of entanglement using a two-qubit
  gate*, [arXiv:quant-ph/0011050](https://arxiv.org/abs/quant-ph/0011050) —
  the Cartan (KAK) decomposition and entangling power.
- J. Ambjorn, J. Jurkiewicz, R. Loll, *Reconstructing the Universe*,
  [arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154) — the Regge
  action and the Metropolis machinery `CDTSimulation` mirrors.
- T. Regge, *General relativity without coordinates*, Nuovo Cimento **19**
  (1961) 558 — the Regge action and deficit angles.
