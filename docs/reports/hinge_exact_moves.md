# Hinge-exact, exactly invertible Pachner moves and cone primitives

The CDT Pachner move suite is hinge-exact and exactly invertible:

> `move ∘ move⁻¹` leaves both the simplicial complex and its Lorentzian
> (Sorkin) `dualReggeAction` — real and imaginary parts — invariant to machine
> precision.

Without that guarantee, a greedy move search would chase orphan-hinge,
lossy-rollback, and build-order artifacts rather than the geometry.

Everything described here is topology-preserving: the cone primitive is stellar
refinement only, with no topology-changing surgery.

## The four mechanisms that keep the action exact

The dual Regge action is defined as the geometric invariant
`S = Σ_h |★h|·ε_h` over the `(d-2)`-hinges of the triangulation, where `|★h|`
is the dual volume of the hinge and `ε_h` its deficit angle. Four mechanisms
make the implementation compute that invariant faithfully.

### 1. Orphan hinges are excluded from the hinge set

`Spacetime::removeSimplex` on a top cell leaves the `(d-1)` and `(d-2)`
sub-faces it had lazily materialised (`Simplex::getFacets`) registered in
`getSimplices()`. Once their last top coface is gone they are orphans:
`Simplex::deficitAngle` returns a bare `2π` for them, because there is no top
cell to subtract a dihedral angle from, while their gradient maps are empty.
Counting them in the action but not in the gradient would make the two
inconsistent.

`ReggeSolver::collectHinges` therefore keeps only genuine `(d-2)`-faces of a
current top cell, via `Simplex::hasTopCoface`. `dualReggeAction`,
`reggeAction`, `actionGradientExact`, and the Hessians all route through
`collectHinges`, so each is a pure function of the current top-cell set and is
exactly equal to a from-scratch rebuild.

### 2. Dual volume uses the ambient dimension from the metric

`Simplex::dualVolume` and its gradient and Hessian need the ambient dimension
`n`. Walking `getCofaces()[0]` up to a top cell is unreliable: a move can leave
a stale orphan facet at index `[0]`, truncating the walk and giving a hinge the
wrong `n` and an empty gradient.

`Simplex::ambientTopDimension` reads `n` directly from the metric signature
when an owning spacetime is present, falling back to the coface walk for
coordinate-free fixtures. The dual-volume recursion then sums over the genuine
up-closure; orphan facets in a coface list contribute zero because their own
recursion dead-ends.

### 3. The Lorentzian deficit angle is vertex-order independent

The dihedral-cosine cofactor formula applies a sign fix
(`if (Cii < 0) denom = -denom`) standing in for the `(-1)^d` diagonal-cofactor
parity. With the signed (non-Wick-rotated) Cayley-Menger matrix of a Lorentzian
cell, `C_ii` and `C_jj` can carry different signs, so a naive evaluation
depends on which of the two opposite vertices the cell stores first. Pachner
moves store a cell's vertices in causal rather than sorted order, so the same
geometry built by a move and built sorted would give different deficit angles:
on a fixed `S⁴ = ∂Δ⁵` geometry, sorted, reversed, and shuffled cell orders gave
actions of `-3.11`, `-1.87`, and `-2.62`.

`Simplex::dihedralAngle` evaluates in the canonical sorted-by-id frame (the
`ChainComplex` reference orientation) via `Simplex::cayleyMengerCanonical`, and
anchors the asymmetric sign fix on the lower canonical position. The deficit
angle, and therefore the whole action, is a relabelling and vertex-order
invariant.

### 4. `RemoveMove` is exactly invertible

`RemoveMove` deletes a vertex, and `rollback` recreates it with the same id but
as a fresh `Vertex` object (`createVertex(id)` after `removeIfIsolated` freed
the original). Any facet or hinge materialised before the deletion would still
point at the old object. Because sub-simplices are reused by fingerprint, and
fingerprints are id-based, the old- and new-object versions collide, so a
restored star's dual and coface walk would run over a stale, empty-list vertex,
yielding a spurious bare `2π` and a wrong dual volume. The add, flip, iflip,
and shift moves are unaffected: they never remove and recreate a vertex.

Two layers prevent this:

- `RemoveMove::apply` and `applyPreGeometric` drop the orphaned sub-simplices
  incident to the deleted vertex (`removeIncidentSubSimplices`) before removing
  it, so rollback's re-materialisation builds facets and hinges that reference
  the recreated vertex.
- `Simplex::deficitAngle`, its gradient, and `hasTopCoface` scan **all** of a
  hinge's vertices for incident top cells (`Simplex::incidentTopCells`, deduped
  by fingerprint, membership tested by id), so a single stale pointer cannot
  mask a genuine top coface. This is identical to the single-vertex scan
  whenever no vertex is stale.

## Cone primitives

The stellar `1↔(d+1)` refinement cone is the pre-geometric `AddMove` (cone-in,
`1→(d+1)`) and `RemoveMove` (cone-out, `(d+1)→1`). New cells are built in the
canonical orientation. Round-trip tests exercise both
`AddMove(PreGeometric).apply`/`.rollback` (deterministic cone-in and cone-out)
and the standalone pre-geometric `RemoveMove` (cone-out) inverted by its
rollback.

## API

| Symbol | Role |
| --- | --- |
| `Simplex::hasTopCoface()` | genuine-hinge predicate (a face of some current top cell) |
| `Simplex::incidentTopCells()` | deduped top cells containing this simplex, stale-pointer-robust |
| `Simplex::ambientTopDimension()` | ambient `n` from the metric (robust dual-volume walk) |
| `Simplex::cayleyMengerCanonical()` | bordered Cayley-Menger matrix in the sorted-by-id reference frame |
| `Spacetime::pruneOrphanedSimplices()` | restore the registry to the exact top-cell closure |
| `RemoveMove::removeIncidentSubSimplices()` | drop a deleted vertex's orphaned sub-faces |

`pruneOrphanedSimplices` is an explicit utility. The moves do not auto-prune
the wider region, so a long optimiser run may want to call it periodically. The
action never needs it, since orphans are already excluded; it exists only for
raw-set identity.

## Round-trip residuals

Measured by `tests/cobordism/test_hinge_exact_moves.py`:

- **Action invariance** `|A(move∘move⁻¹) − A|`: `≤ ~1e-15` on the minimal
  spheres, `≤ ~1e-9` on the larger CDT(250) builds, where summation
  reassociation over more hinges dominates. Both real and imaginary parts are
  asserted; the CDT toroid fixture has `Im S ≈ −35`, so imaginary-part sign
  stability under coning is a live check.
- **Complex invariance:** the top-cell set and the genuine-hinge set are
  restored identically; after `pruneOrphanedSimplices` the raw registered
  simplex set is bit-identical.
- **Coverage:** every move type (add, remove, flip, iflip, shift) plus stellar
  cone-in and cone-out, on `S³ = ∂Δ⁴`, `S⁴ = ∂Δ⁵`, `S²×S¹`, and CDT toroids of
  size 40, 120, and 250; a five-deep stack of cones inverted last-in-first-out
  with the action retraced at every level; `Im S` preserved under coning and
  flips.

## References

- T. Regge, *General relativity without coordinates*, Nuovo Cimento **19**
  (1961) 558 — the Regge action and deficit angles.
- U. Pachner, *P.L. homeomorphic manifolds are equivalent by elementary
  shellings*, European J. Combin. **12** (1991) 129 — the Pachner moves.
- R. D. Sorkin, *Lorentzian angles and trigonometry including lightlike
  vectors*, [arXiv:1908.10022](https://arxiv.org/abs/1908.10022) — dihedral and
  deficit angles in Lorentzian signature.
- S. K. Asante, B. Dittrich, J. Padua-Arguelles, *Effective spin foam models
  for Lorentzian quantum gravity*,
  [arXiv:2104.00485](https://arxiv.org/abs/2104.00485) — the Lorentzian Regge
  action with complex dihedral angles.
