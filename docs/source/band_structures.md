# Band structures

`tessera.drivers.bands` computes the band structure of a crystal from the
degree-zero covariant Whitney pencil. A crystal is a periodic cell meshed by
`tessera.PeriodicKuhnGrid`; its one-particle problem is the generalized
eigenproblem $A z = E M z$, where $M$ is the mass matrix of piecewise-linear
finite elements on the mesh and $A$ is $\hbar^2/2m$ times their stiffness
matrix plus the potential. Lengths are in angstrom and energies in electron
volts.

## The pencil at a crystal momentum

A crystal momentum $k$ is a flat U(1) connection on the mesh: the link of the
edge from $v$ to $w$ is $e^{i k \cdot \Delta x_{vw}}$ with $\Delta x_{vw}$ the
unwrapped displacement of the edge. Its curvature is 1 on every triangle and its
holonomy around the fundamental cycle along the lattice vector $a_i$ is
$e^{i k \cdot a_i}$. Because such a connection is a pure gauge on every
tetrahedron, dressing the pencil by it multiplies the entry $(v, w)$ of both
matrices by the link $U_{vw}$. `CrystalCell` assembles the matrices once and
dresses them per momentum; `CrystalCell.certify` holds that form to the C++
`CovariantChainHodge` and measures the premises the solver relies on
(allowable geometry, unimodular links, zero curvature, the prescribed holonomy,
Hermitian matrices, a positive definite mass matrix).

A scalar potential given by its vertex values enters as the weighted mass
matrix $M_0[V]$, the matrix of the bilinear form $\int V u w$ with $V$
interpolated linearly on every tetrahedron.

```python
import numpy as np
from tessera.drivers.bands.crystal import CrystalCell, k_path, richardson
from tessera.drivers.bands.potentials import ZincBlendeEPM

epm = ZincBlendeEPM.gallium_arsenide()
cell = CrystalCell(epm.conventional_lattice(), 24)
read = cell.solve((0.0, 0.0, 0.0), count=24, potential=epm.potential(cell))
assert read.certified()         # residual met, and the shift certified below the spectrum
print(read.energies)
```

`read.certified()` combines the two statements the sparse solver certifies: the
relative residual of every eigenpair met its tolerance, and the shift admitted a
Cholesky factorization, so the returned levels are the lowest of the spectrum.

## Mesh convergence

Piecewise-linear elements converge at second order in the mesh spacing $h$, and
on a uniform mesh the error expansion is even in $h$, so three meshes remove the
$h^2$ and $h^4$ terms (`richardson`). Single meshes are far from converged at
sizes a workstation can hold; the extrapolation is what makes the numbers
usable. For gallium arsenide with the Cohen–Bergstresser empirical
pseudopotential on the conventional cell:

| divisions | vertices | direct gap (eV) |
|-----------|----------|-----------------|
| 24        | 13,824   | 1.1422          |
| 32        | 32,768   | 1.2611          |
| 40        | 64,000   | 1.3172          |
| extrapolated |       | 1.4188          |
| plane waves, same form factors | | 1.4186 |

The reference is a plane-wave diagonalization with the same form factors
(`ZincBlendeEPM.plane_wave_bands`), so the comparison tests the machinery and
involves no external number.

## What the mesh does to symmetry

Every cube of the grid is cut along the same body diagonal. The mesh keeps the
threefold axis along $(1, 1, 1)$ and inversion, and loses the fourfold axes, so
a cubic triplet (the top of the valence band of a zinc-blende crystal, or the
$p$ level of a spherical well) is a doublet and a singlet, split at the order of
the mesh error. The drivers read such a level as the mean of its three members,
in which the splitting cancels to first order, and report the splitting.

The conventional cubic cell folds the three X points onto the zone centre. The
two families of states are told apart by their characters under the
face-centred translations, which are exact symmetries of the crystal and of the
mesh (`gaas.translation_characters`).

## The stages

Each module has a test file under `tests/drivers/` that holds it to a reference
with a known answer.

| Module | Subject | Reference |
|--------|---------|-----------|
| `crystal` | free electrons, certificates, extrapolation | $\lvert k + G \rvert^2$; the degree-one exact sector; gauge invariance |
| `potentials` | cosine potential, Gaussian well, nonlocal projector, empirical pseudopotential | Mathieu characteristic values; the radial equation; dense solves; plane waves |
| `fiber` | a static potential as the phase of the timelike edges of the history complex | its Klein–Gordon limit; the static route |
| `spin` | two sheets, spin–orbit coupling as the attachment block | the quartet and doublet of $L \cdot S$ |
| `coulomb` | the finite-element Coulomb kernel, Hartree–Fock on the covariance | the periodic Coulomb potential; exact diagonalization; the Wick engine; the electron-gas exchange energy |
| `screening` | gauge response, polarizability, random-phase approximation, one-shot quasiparticle correction | the Ward identity; the exact discrete Lindhard sum; two routes to the correlation energy; the second-order self-energy |
| `response` | the derivative of a band energy with respect to the squared edge lengths | Euler's identity; finite differences |
| `gaas` | gallium arsenide with the empirical pseudopotential | plane waves with the same form factors |

### The fiber-edge route

On the history $K \times [0, 1]$ of a cell the timelike edges carry the
Euclidean squared length $\tau^2$ and the potential as the non-compact part of
the connection. The library's link $U_{xy}$ carries a value at $y$ back to $x$,
so propagation forward across a tick is multiplied by $U_{yx} = e^{-\tau V_e}$
when the link read forward in time is $U_{xy} = e^{\tau V_e}$; the fields live
on the edges of a `Spacetime` (`fiber.history_spacetime`), and the slab is
assembled and reduced onto its outer levels by `PencilLayer`
(`fiber.layered_response`). The stiffness of that timelike connection on the
history, $\varphi^T \partial_2 M_2 \partial_2^T \varphi$ on a fiber-edge
pattern, equals the spatial stiffness matrix up to the fiber measure $1/\tau$
exactly (`fiber.fiber_edge_stiffness`), which is why eliminating it leaves the
Coulomb kernel $\tilde A_0^+$ of the `coulomb` module. A constant potential is then a pure
gauge and shifts every decay rate of the tick map by exactly $V$; a varying one
has curvature on the vertical triangles, the electric field. The tick map of a
stack of slabs solves a quadratic eigenproblem in the blocks of the slab's
pencil, and as $\tau \to 0$ its decay rates $E$ obey

$$ (A + m^2 W)\, u = E^2 D\, u , $$

with $D$ the lumped (diagonal) mass matrix, which is what the time stiffness of
a staircase slab produces, and $W$ the matrix the mass term is added with. With
the consistent mass matrix ($W = M$) the mismatch between $M$ and $D$ is
multiplied by $m^2$, and the non-relativistic reduction $E \approx m + L/2m + V$
requires the mesh to resolve the Compton wavelength, $m h \ll 1$, in addition to
$L \ll m^2$. With the lumped mass term ($W = D$) the reduction holds at any mesh
spacing. What remains between the fiber-edge route and the static route is then
first order in the potential and closes with the mesh: on a connection with
curvature the covariant operator transports through the base vertex of each
cell, which samples the potential one mesh step away.

### The quasiparticle correction

`screening.RandomPhase` solves the direct random-phase equations in the basis of
particle–hole pairs, which gives the screened interaction analytically at every
frequency, and evaluates the correlation self-energy without a frequency grid or
a plasmon-pole model. `screening.one_shot_gap` runs the chain end to end on a
periodic mesh: pencil, real modes, pair densities, the finite-element Coulomb
kernel, the Roothaan loop, and the quasiparticle equation on the Hartree–Fock
levels. The number of modes is the basis of the screened interaction and the
result must be converged in it. An ab initio calculation of a real crystal
additionally needs ionic pseudopotentials and a set of crystal momenta, at a
cost (one Poisson solve per pair of orbitals per pair of momenta) that belongs
on a cluster.
