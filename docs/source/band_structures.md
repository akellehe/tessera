# Band structures

`tessera.drivers.bands` computes the band structure of a crystal from the
degree-zero covariant Whitney pencil. A crystal is a periodic cell meshed by
`tessera.PeriodicKuhnGrid`; its one-particle problem is the generalized
eigenproblem $A z = E M z$, where $M$ is the mass matrix of piecewise-linear
finite elements on the mesh and $A$ is $\hbar^2/2m$ times their stiffness
matrix plus the potential. Lengths are in angstrom and energies in electron
volts.

## What these calculations are

The crystal, its potential and its two spin sheets are put in by hand: in the
terms of the theory this is targeted synthesis, not emergence, and nothing here
tests whether a geometry relaxes to a crystal. What is tested is the code path
from the declared fields (a complex squared length and a connection on every
edge, one mode per cell, second quantization) to a quasiparticle energy. The
modes are carried at degree zero, which is the exact sector of the degree-one
edge operator: the nonzero levels of the two agree, and the test suite holds
them to each other.

Every interaction is the Coulomb kernel obtained by eliminating the timelike
connection, and every mean field is its Wick contraction (Hartree and exchange).
A density functional is not an object of the theory and none is used.

A calculation of a real material is held to experimentally determined values
(`tessera.drivers.bands.reference`). A plane-wave calculation of the same model
Hamiltonian is a consistency check of that model and does not arbitrate a
discrepancy; fixtures with synthetic ions have no measured value and are
consistency checks only.

## The pencil at a crystal momentum

A crystal momentum $k$ is a flat U(1) connection on the mesh: the link of the
edge from $v$ to $w$ is $e^{i k \cdot \Delta x_{vw}}$ with $\Delta x_{vw}$ the
unwrapped displacement of the edge. Its curvature is 1 on every triangle and its
holonomy around the fundamental cycle along the lattice vector $a_i$ is
$e^{i k \cdot a_i}$. `CrystalCell.pencil` assembles the pencil at that
connection with `CovariantChainHodge.sparsePencil` and
`dressedVertexPotential`. Because such a connection is a pure gauge on every
tetrahedron, dressing the pencil by it multiplies the entry $(v, w)$ of both
matrices by the link $U_{vw}$; `CrystalCell.certify` holds that entrywise form
to the assembly and measures the premises the solver relies on
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

$$ (A + m^2 M)\, u = E^2 D\, u , $$

with $M$ the Whitney mass matrix and $D$ the lumped (diagonal) mass matrix,
which is what the time stiffness of a staircase slab produces. The mismatch
between $M$ and $D$ is multiplied by $m^2$, so the non-relativistic reduction
$E \approx m + L/2m + V$ requires the mesh to resolve the Compton wavelength,
$m h \ll 1$, in addition to $L \ll m^2$. On cells of three to six divisions at
$m = 6$ the level shift a potential induces in the tick map exceeds the static
route's by a factor of 6.7, 3.3, 2.3 and 1.9, falling toward one as the mesh is
refined: on a connection with curvature the covariant operator transports
through the base vertex of each cell, which samples the potential one mesh step
away.

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

`screening.KineticBasisScreening` is the second route to the same self-energy,
and the one that scales to momentum sets: in the eigenbasis of the kinetic
pencil the Coulomb kernel is diagonal, the dielectric matrix is inverted at
imaginary frequencies, and the self-energy follows by contour deformation, the
Lorentzian integrated in closed form against the interpolated interaction. The
response at vanishing momentum enters as one more basis function, which screens
the rest of the interaction through the mixed entries of the dielectric matrix;
`RandomPhase.set_head` does the same on the modes of the particle-hole pairs.
With the whole basis the two routes agree at every frequency, and the test suite
holds them to each other.

### Ab initio

`pseudopotential` reads norm-conserving pseudopotentials in the Unified
Pseudopotential Format and `abinitio` runs a crystal self-consistently on the
mesh and, as the reference, in plane waves with the same pseudopotentials,
momenta and conventions. The local part of an ion is split into a short-range
remainder, summed over images, and the potential of a Gaussian charge, which on
the mesh is a source of the same Coulomb kernel the electrons interact through
(the continuum kernel of the plane-wave reference leaves a uniform remainder,
`PlaneWaveCrystal.alignment`, that the mesh does not carry). Closed forms used
anywhere in these drivers are closed forms of the framework's own matrices (the
Fourier symbols of its stiffness and mass matrices, derivatives of its covariant
assembly), each held to the framework's numerical route by a test; a continuum
solution is never substituted for one. The separable nonlocal part is a term
`P D P^T` of low rank in the left-hand matrix of the pencil, with `P = M beta`
the load vectors of the projector functions; `SparsePencilSolver` applies it
through the Woodbury identity and certifies the shift by inertia. Exchange is
compressed onto the computed bands and joins the same low-rank term. The
Coulomb kernel of the grid is inverted exactly by Fourier transform, because the
stiffness matrix commutes with the grid translations.

A Coulomb kernel of zero mean leaves out the zero-momentum term of exchange and
of the screened interaction, and both are restored by the auxiliary-function
method of Gygi and Baldereschi with the kernel's own symbol as the auxiliary
function. For exchange the term is the constant
`GridCoulombKernel.zero_momentum_constant()` on the filled bands: the average of
the inverse symbol of the stiffness matrix over every momentum, minus its sum
over the wavevectors the cell supports. For the continuum kernel on a cubic cell
that constant is `2 MADELUNG / L`, which the mesh constant tends to under
refinement. For the screened interaction the term is the same constant times
the inverse dielectric function at vanishing momentum, which needs the charge
of every particle-hole pair per unit momentum. That charge is the derivative of
$1^T M_0^U[\psi_i]\, z_a(q)$ with respect to the momentum of the flat
connection, and it is taken in closed form
(`MeshCrystal.vanishing_momentum_pairs`): because the filled orbitals are
eigenvectors, first-order perturbation theory needs no linear solve,

$$ d_{ia} = 1^T (\partial M_0^U[\psi_i])\, z_a
   + \frac{\psi_i^T (\partial H - \epsilon_a\, \partial M)\, z_a}{\epsilon_a - \epsilon_i} , $$

with $\partial$ the derivative with respect to a uniform change of the link
phases: entrywise for the stiffness, mass and weighted mass matrices
(`GridMatrix.momentum_derivative`), the product rule on the projector loads, and
for exchange the derivative of the dressed weighted mass matrices and of the
Coulomb kernel, whose symbol has a closed-form gradient
(`GridCoulombKernel.potential_derivative`). The entry of the kernel at $G = 0$
tends to `strength / (V q^2)` exactly, because piecewise-linear elements
reproduce linear functions. The same response from a small finite momentum
(the Hartree-Fock pencil solved at the flat connection of momentum $q$ by
`MeshCrystal.bands_at`, pair densities loaded with the link phases of that
momentum by `momentum_pairs`) converges to the closed form at second order in
$q$, and the test suite holds the two to each other. On the energy shell the
term lowers the gap by `c (1 - 1/eps)`, which is how screening closes a
Hartree-Fock gap.

`MeshCrystal.run_hartree_fock_set` samples the covariance on a momentum set, a
uniform grid that contains the zone centre. The pencil at each momentum is
dressed by the flat connection of that momentum; the exchange operator carries
the momentum transfer $k - k'$ in its kernel, the inverse of the stiffness
matrix dressed by the transfer, with pair densities loaded between the two
momenta; and the zero-momentum constant is that of the set, which is the
constant of the supercell the set is equivalent to. A cell doubled along an
axis at its zone centre and the single cell sampled at 0 and 1/2 along that
axis give the same Hartree-Fock levels, filled and empty, and the test suite
holds them to each other.

Whether a pseudopotential can be used is decided by `pseudopotential`'s
screened, confined pseudo-atom, solved radially and on the mesh. A
pseudopotential that keeps the gallium 3d shell in the valence is not resolved
by piecewise-linear elements at any mesh a workstation holds (the 3d level is
22 eV too high at 32 divisions of a 5.64 angstrom cell and not in the asymptotic
regime); three- and five-electron pseudopotentials for gallium and arsenic are
(their 4s and 4p levels extrapolate to the radial values within 0.1 eV).
