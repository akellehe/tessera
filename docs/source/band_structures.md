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

## A mesh graded toward the ions

A level bound within a fraction of a bohr of an ion, such as the gallium $3d$
shell, is beyond a uniform mesh: the error is second order in a spacing that the
whole cell has to pay for. The geometry of the framework is one squared length
per edge, so a graded mesh is the same objects with other lengths and, where
vertices are added, another chain complex. `graded.GradedCell` is a
`CrystalCell` built in two ways that combine:

- `IonRefinement` bisects the Kuhn simplices within given radii of the ions
  (`KuhnBisection`: the conforming bisection of Maubach and Traxler, in which
  every simplex around an edge is brought to the state that cuts that edge
  before it is cut). Three bisections halve every edge and return Kuhn
  simplices of half the size, so shapes do not degenerate and no vertex hangs
  inside a neighbour's face. The vertices of the grid keep their numbers.
- `IonGrading` moves the vertices by a smooth periodic map that contracts a
  ball about every ion. It adds no vertices, so it can only borrow them from
  the shell around the core; it is refused when it folds the mesh
  (`orientation_margin`) or would move an ion.

```python
from tessera.drivers.bands.graded import GradedCell, IonRefinement

cell = GradedCell(lattice, 12, refinement=IonRefinement(ions, [(3.2, 1), (1.8, 2), (1.0, 3)]))
cell.certify(kappa).holds()          # the same premises, measured on this mesh
```

A crystal momentum is the flat connection $U_{vw} = e^{i k \cdot dx_{vw}}$ on
the true displacement of every edge, which is the connection on the grid steps
in another gauge; the constant section is then the plane wave at the vertices
wherever they are. On a well of width 0.35 in a cell of side 6 the uniform mesh
of 4,096 vertices misses the lowest level by a third of its binding energy, the
bisected mesh of 2,322 vertices by six per cent, and one more halving about the
ion divides every error by four.

What the uniform grid has in closed form through its translation invariance has
a counterpart without it. `graded.GradedCoulombKernel` applies the inverse of
the dressed stiffness matrix through a sparse factorization and names the
entries of the kernel by what they are: the $G = 0$ component of a load is its
total charge, the $G = 0$ entry of the kernel is the energy of the uniform
normalized charge of that momentum, and the auxiliary function of the
zero-momentum constant is the energy of a unit point load, averaged over every
momentum with the singular part taken analytically as on the grid. On the Kuhn
grid every method returns the value of `GridCoulombKernel`, the zero-momentum
constant included (to $10^{-12}$ Ry in the test suite). Off the grid the
constant depends on the vertex that carries the point load at second order in
the spacing, and costs one factorization per momentum of its quadrature.
`abinitio.MeshCrystal(crystal, divisions, grading=..., refinement=...)` runs the
Hartree and Hartree-Fock mean fields on such a mesh, at the zone centre and at a
finite momentum, and the quasiparticle step at the zone centre with its
closed-form head. The prolongation between meshes, the closed-form kinetic
modes of the kernel (`kinetic_modes`) and the square root of the mass matrix in
`band_fibers` are still those of the uniform grid. A bisected mesh is
assembled from its chain complex and squared lengths; it is not yet a `Topology`
of the library, so it has no `Spacetime`.

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
which is what the time stiffness of a staircase slab produces. With a potential
the baseline is the static relativistic problem

$$ (A + m^2 M)\, u = D\, (E - V)^2\, u , $$

a quadratic eigenproblem in $E$ in which nothing is expanded, neither in the
potential nor in $1/m$ (`fiber.static_levels`). The lowest decay rate of the tick
map closes on its lowest level at second order in the mesh spacing (a difference
of 0.062, 0.029, 0.016, 0.010 on cells of 3 to 6 divisions at $m = 6$), and the
response to the potential approaches the static one from below (0.74, 0.82,
0.88, 0.91); neither depends on the tick. The exact limit of the tick map with
the potential is a closed form (`fiber.tick_limit`),

$$ (S_0 + E S_1 + E^2 S_2)\,u = 0 , \qquad S_2 = -D , \quad S_1 = 2DV + C_1 , \quad
   S_0 = A + m^2 M - DV^2 + C_0 , $$

and the decay rates of the tick map converge to its levels at second order in
the tick. $C_0$ and $C_1$ are the curvature of the connection on the vertical
triangles: the covariant operator transports through the base vertex of each
cell, $b(\sigma) = \min \sigma$, which samples the potential one mesh step away.
They are sums over the spatial simplices $T$ of $|T|$ times a block on the
vertices of $T$ (`fiber.staircase_blocks`). With the vertices of $T$ in
ascending id, $i = 0, \dots, d$ (the order in which the staircase of the slab
climbs), $\Delta_a = V_a - V_i$ and $N = 4(d+1)(d+2)(d+3)$,

$$ N\,(c_1)_{ii} = 4(2d+3-i) \sum_{a<i} \Delta_a , \qquad
   N\,(c_1)_{ij} = -2 \Big[ \sum_{a<i} (V_a - V_j) + (d+3-i)(V_i - V_j) \Big] \quad (i<j) , $$

$$ N\,(c_0)_{ii} = -V_i\,N (c_1)_{ii} - (2d+3-2i) \sum_{a<i} \Delta_a^2 - \Big( \sum_{a<i} \Delta_a \Big)^2 , \qquad
   (c_0)_{ij} = -V_i\,(c_1)_{ij} \quad (i<j) , $$

both symmetric. They are the first and the second order in the tick of the time
part of the dressed stiffness $\partial_1^U M_1^U (\partial_1^{U^{-1}})^T$ on
the $d+1$ simplices of a staircase prism. Only the time components of the
gradients of the barycentric coordinates enter the Whitney mass matrix $M_1$
there, and they are $\mp 1/\tau$ on the two ends of the one vertical edge of
each simplex, so the blocks depend on the volume of $T$, on the potential at its
vertices and on the order of their ids, and on nothing else of the geometry.
Every entry is a sum of differences of the potential (the electric field through
the vertical triangles) and vanishes for a constant one. The test suite holds
the blocks to the expansion of $M_1$ with the transport convention in exact
rational arithmetic for $d = 1, \dots, 4$, and the assembled limit to the blocks
of the slab (`fiber.tick_limit_from_blocks`): reversing the tick transposes the
pencil, so the symmetric parts of the blocks are even in the tick, and one
Richardson step leaves a difference that falls like $\tau^4$, to $10^{-9}$ at
$\tau = 0.02$ on cells of 3 to 6 divisions with a potential that is not a pure
gauge. With the tick gone, the lowest level of the limit closes on the static
relativistic problem at second order in the mesh (the difference times the
square of the divisions is 0.56, 0.46, 0.39, 0.37, 0.35, 0.35 on cells of 3, 4,
5, 6, 8, 10 divisions). The non-relativistic reduction $E \approx m + L/2m + V$ is not
used anywhere: it needs the mesh to resolve the Compton wavelength, and compared
with it the same tick map appeared to over-respond by factors of 2 to 7.

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
imaginary frequencies, and the self-energy follows by contour deformation. The
interaction along the imaginary axis is carried by a Chebyshev series of degree
63 in a mapped frequency (nothing is linearized), and the Lorentzian becomes the
measure of a Gauss-Legendre quadrature. The
response at vanishing momentum enters as one more basis function, which screens
the rest of the interaction through the mixed entries of the dielectric matrix;
`RandomPhase.set_head` does the same on the modes of the particle-hole pairs.
With the whole basis the two routes agree at every frequency to 1e-11 Ry, and the
test suite holds them to each other.

No equation in these drivers is linearized: the quasiparticle equation is
iterated to its root, the tick map is the full quadratic eigenproblem, and a
series that is truncated keeps at least five terms (the refinement series of the
zero-momentum constant, the lattice sums over images, the frequency series
above). The mesh extrapolation `richardson` removes one even order per mesh
beyond the first, five with six meshes; `richardson_amplification` is the factor
by which it multiplies anything in the values that does not follow the error
model (5.6 for divisions 16, 24, 32 with two orders; 27 for 8, 12, 16, 20, 24,
32 with five), and is reported with every extrapolated number.

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
`P D P^T` of low rank in the left-hand matrix of the pencil, with `P` the load
vectors of the projector functions, $P_v = \int \beta(x)\,\lambda_v(x)\,dx$
against the vertex function $\lambda_v$. The radial functions are tabulated, so
the loads have no closed form: they are taken by a collapsed Gauss rule on every
tetrahedron (`loads.SimplexQuadrature`, exact for polynomials of degree $2n-1$
with $n$ points per direction, held to the library's mass matrix and triple
integrals by a test), summed over the images of each ion within the reach of
its table. At a crystal momentum the load of every image carries the Bloch
phase of its displacement to the vertex. `M beta`, the mass matrix on the
vertex values of the projector, is the load of the projector's interpolant and
remains available (`--projector-quadrature 0`); the local potential is
interpolated at the vertices, as the plan has it. `SparsePencilSolver` applies
the term through the Woodbury identity and certifies the shift by inertia. Exchange is
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
(`GridMatrix.momentum_derivative`), the derivative of the Bloch phases of the
projector loads (`loads.LocalLoads.derivative`), and
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

### Running the prediction

```
python -m tessera.drivers.bands.gaas ab-initio --cation Ga.UPF --anion As.UPF \
    --divisions 8 12 16 20 24 32 --bands 24 --screening-bands 200 --out gaas.json
```

runs Hartree-Fock and the quasiparticle equation (one shot, with the levels fed
back into the propagator, and into the propagator and the screening) on every
mesh, certifies each converged state as a `CovarianceState`, extrapolates over
the meshes and prints the result next to `richardson_amplification` and the
measured gap, which is the arbiter. Every approximation made for the sake of
cost is a flag (`settings.Approximations`), recorded in the output:

| flag | what it truncates | range, default |
|---|---|---|
| `--self-energy-order` | terms of the expansion of the self-energy in the screened interaction $W$; 1 is $\Sigma = iGW$, 2 adds the crossed diagram, 3 the six skeleton diagrams of third order, 4 the 49 of fourth, 5 the 542 of fifth (`diagrams`) | 1 to 5, default 3; all implemented |
| `--vertex-bands`, `--vertex-poles` | the modes nearest the gap on the internal lines of the diagrams beyond the first order, and the modes of the screened interaction kept in them; the cost of order $k$ grows as (bands/2)$^{2k-1}$ poles$^k$ | default 12 and 12 |
| `--vertex-memory` | what the diagrams hold at once, in GiB, over all worker processes; it truncates nothing: a diagram too wide for it is evaluated one value at a time of the labels of some of its interaction lines | default 8 |
| `--zero-momentum-order` | how the self-energy integrand is averaged over the momentum transfers the sampling leaves out: 1 is the closed form at vanishing momentum; k is a midpoint grid of k transfers per axis, the Hartree-Fock pencil solved at every node, the singular part averaged analytically and the bounded remainder by the grid | 1 to 5, default 3; all implemented |
| `--momenta` | the momentum set on which the covariance is sampled, a uniform grid of that many crystal momenta per axis of the cell through its zone centre (`momentum_set`); Hartree-Fock is solved at one momentum of every orbit of time reversal and of the axis permutations the crystal has, the screened interaction is built at every momentum transfer of the set with the entry at zero transfer in closed form, and the offsets of `--zero-momentum-order` surround every transfer. A cell on a set is the supercell at its zone centre, and the test suite holds every step to that identity | at least 1, default 1 (the zone centre); on a set the diagrams beyond the first order run over the states of the set nearest the gap, a mode of the screened interaction of momentum $q$ entering as two bosons with Hermitian couplings |
| `--refinement-terms` | terms of the refinement series of the zero-momentum constant | 1 to 5, default 5 |
| `--lattice-images` | periodic images per axis in the lattice sums | odd, default 5 |
| `--projector-quadrature` | Gauss points per direction of the rule that loads the projector functions on every tetrahedron; 0 loads the interpolant of the projector with the mass matrix | at least 1 (or 0), default 6 |
| `--frequency-nodes` | terms of the Chebyshev series along the imaginary frequency axis (`KineticBasisScreening`) | at least 5, default 64 |
| `--exchange-history` | earlier exchange updates of Hartree-Fock whose filled sections join the span in which the accelerator of the update minimizes the energy; it changes the number of updates and the cost of each, and leaves the converged state where it is | at least 0, default 5 |
| `--divisions` | meshes; every mesh beyond the first removes one even order of the mesh error | default six meshes, five orders |

An order that is not implemented is refused by name before anything runs; it is
never replaced by a lower one (every order of both expansions exists today). The
diagrams beyond $\Sigma = iGW$ are evaluated as sums over the orderings of their
vertex times, in closed form on the poles of $W$, with the instantaneous part of
$W$ as lines whose two vertices share a time. `diagrams.skeleton_diagrams`
enumerates them (1, 1, 6, 49, 542 at the orders 1 to 5), and the sum over the
$V!$ orderings of $V$ vertices is taken as a recursion over the sets of vertices
that have happened, $V\,2^{V-1}$ contractions, which is the same number term for
term. The test suite holds the diagrams to the closed form at first order, to
the textbook second-order exchange, to the plain frequency integrals of their
Feynman rules on the imaginary axis (every diagram through fourth order, the
fermion loops included), to the heavy-boson limit for instantaneous lines, the
recursion to the plain sum over the orderings, and the rules as a whole to the
exact diagonalization of electrons coupled to bosons: with the bare propagator
on the lines, all 238 irreducible diagrams of fourth order and all 2732 of fifth
(the skeletons among them) sum to the coefficient of that order of the exact
self-energy to $10^{-11}$.

What an order costs is set by the denominator that holds the most lines at once:
$2k-1$ fermion lines and $k$ interaction lines at order $k$, which no
factorization separates. Measured on four worker processes, one evaluation of
the third order takes 3 s at 12 bands and 12 poles; the fourth 16 s at 6 and 6
and 200 s at 8 and 8; the fifth 400 s at 4 and 4. The number of multiplications
(`SkeletonSelfEnergy.work`) puts the fourth order at 12 and 12 near 13 hours on
four processes at the rate measured at 8 and 8, and the fifth at 6 and 6 between
5 and 20 hours, so a run at these orders lowers `--vertex-bands` and
`--vertex-poles`, or takes that long; the choice is the caller's and is recorded
with the result.
On a cell of 6 bohr the correlation self-energy of the filled level
goes from -0.032 Ry at zero-momentum order 1 to -0.052 and -0.061 Ry at orders 2
and 3 (-0.071 Ry on a grid of 6): sampling the zone centre alone is a large
approximation on a small cell.

Hartree-Fock has more than one stationary state, and a loop reaches the one its
start leads to. The loop starts from the Hartree mean field when that has a
self-consistent state and from one diagonalization in the potential of the
atomic density when it does not (without exchange gallium arsenide is gapless to
0.03 eV and its filling does not converge); every mesh after the first starts
from the orbitals of the mesh before it, which are piecewise-linear functions
and are evaluated exactly on the finer vertices (`MeshCrystal.prolonged`). The
electronic energy of every run is recorded so that stationary states can be
compared.

The exchange update replaces the exchange energy, which is concave in the
covariance, by its tangent at the current state and minimizes the rest, so each
update lowers the energy; near a saddle the descent is slow enough to hold the
loop for tens of updates. Before each update the energy is therefore minimized
over the Slater frames of the span of the bands just computed, joined by the
filled sections of the last `--exchange-history` updates (`acceleration`). The
Coulomb integrals of the span are computed once through the mesh's kernel, the
state is a pure covariance on the modes of the span, the Fock operator is the
Wick contraction of `ModeInteraction`, and the minimization is by Newton steps
in a trust region, a step being kept only if the energy falls. Nothing is
extrapolated, and a stationary state of the mesh is returned unchanged, so the
fixed points are those of the plain update. A run records the energy of the
state each exchange operator was built from, which never rises, and the lowest
eigenvalue of the second variation of the energy on the last span: positive at
a minimum, negative at a saddle. A crystal of eight soft ions on which the
plain update stalls converges in six updates; gallium arsenide at 12 divisions
converges from scratch in six, where the plain update is still leaving its
saddle after a hundred.

Published inputs cannot be extended, and no flag pretends otherwise. A
pseudopotential file fixes the angular momenta of its projectors (the
Bachelet-Hamann-Schlueter files stop at $l = 1$) and carries spin-orbit data only
if it says so; the Cohen-Bergstresser form factors of the empirical subcommand
are three per series.

Whether a pseudopotential can be used is decided by `pseudopotential`'s
screened, confined pseudo-atom, solved radially and on the mesh. A
pseudopotential that keeps the gallium 3d shell in the valence is not resolved
by piecewise-linear elements at any mesh a workstation holds (the 3d level is
22 eV too high at 32 divisions of a 5.64 angstrom cell and not in the asymptotic
regime); three- and five-electron pseudopotentials for gallium and arsenic are
(their 4s and 4p levels extrapolate to the radial values within 0.1 eV).
