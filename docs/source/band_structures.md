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
| `spinorbit` | two sheets in the ab initio run: relativistic separable pseudopotentials, Hartree–Fock on two-component sections | the radial equation of each total angular momentum; dense solves; the one-sheet loop; the measured splitting |
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
0.88, 0.91); neither depends on the tick. The exact limit of the tick map with the potential is read off the slab's own
blocks (`fiber.tick_limit`): $(S_0 + E S_1 + E^2 S_2)\,u = 0$ with $S_2 = -D$,
$S_1 = 2DV$ and $S_0 = A + m^2 M - DV^2$, the last two up to a symmetric term that
vanishes for a constant potential, and the decay rates converge to its levels at
second order in the tick. What remains at a finite mesh is that term, the
curvature of the connection: on the vertical triangles the covariant operator
transports through the base vertex of each cell, which samples the potential one
mesh step away. The non-relativistic reduction $E \approx m + L/2m + V$ is not
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

### Two sheets in the ab initio run

`spinorbit` carries spin in the ab initio run as two sheets and the spin-orbit
coupling as the attachment block between them. Its input is the relativistic
separable pseudopotential of Hartwigsen, Goedecker and Hutter (Physical Review
B 58, 3641, 1998), a closed form: an error-function local part and Gaussian
projectors with two coefficient matrices per angular momentum $l$, $h^l$ for
the average over the two total angular momenta $j = l \pm 1/2$ and $k^l$ for
their difference,

$$
V_{nl} = \sum_{l} \sum_{ij} \sum_{m} \lvert p^l_i Y_{lm} \rangle h^l_{ij} \langle p^l_j Y_{lm} \rvert
       + \sum_{l \ge 1} \sum_{ij} \sum_{ms, m's'} \lvert p^l_i Y_{lm} s \rangle k^l_{ij}
         (L \cdot S)_{ms, m's'} \langle p^l_j Y_{lm'} s' \rvert .
$$

`HGH_TABLE` holds the rows of the paper's Table I for gallium (three electrons)
and arsenic (five) as printed; an element that is not held is refused, and one
is added by copying its rows. The paper gives no valence density; the one used
to start a loop and to screen the pseudo-atom is the spherical Hartree density
of the pseudo-atom, and no converged number depends on it. The matrices of $L$
on the real harmonics of `pseudopotential.real_harmonics` are exact
(`angular_momentum` applies $-i\, r \times \nabla$ to the harmonic polynomials),
and $L \cdot S$ has the eigenvalues $l/2$ and $-(l+1)/2$ of the two total
angular momenta.

The two-sheet pencil is the direct sum of the one-sheet pencil with itself
(`SparsePencilComposition.directSum`), and the separable part is one low-rank
term on the projector loads of both sheets, with the complex Hermitian core
$1 \otimes h + k \otimes L \cdot S$ (`spin_orbit_core`, `two_sheet_term`).
`SparsePencilSolver` diagonalizes the whole pencil (`solve_two_sheets`), so the
coupling is carried to all orders, and every level is a Kramers doublet
(`time_reversal_defect`). `run_hartree_fock_two_sheets` iterates Hartree-Fock on
the two-component sections: the density sums both sheets, and the exchange
operator couples them through the off-diagonal spin blocks of the density
matrix. Its first diagonalization is the two-sheet pencil of the converged
one-sheet operator with the projectors and the compressed exchange doubled per
sheet; the later ones matter because the compressed exchange is exact only on
the span of the sections it was built from, which the block rotates out of the
span of the one-sheet bands.

The inertia certificate of `SparsePencilSolver` with a low-rank term needs
$A - \sigma M$ positive definite. The strongly repulsive projectors of these
pseudopotentials leave levels of the local part below the lowest level of the
whole operator, so a shift just below the lowest level is not certified; the
converged pencil is solved once more from below the minimum of the local
potential, where the certificate applies (`certify_one_sheet`, and the same at
the end of the two-sheet loop).

The resolution test of `pseudopotential`, the screened and confined pseudo-atom,
runs on two sheets (`pseudo_atom_levels`) against the radial equation with the
coefficients $h + k \langle L \cdot S \rangle_j$ (`spinorbit.radial_levels`). A p
level is a quartet above a doublet; the mesh, which keeps one threefold axis,
splits the quartet further at the order of its error, a rank-two field that has
no trace on either multiplet, so the distance between the centres of the two
multiplets (`multiplet_splitting`) is moved at second order only. For the
published arsenic potential in a cell of 9 bohr:

| spacing (bohr) | 0.75 | 0.56 | 0.45 | 0.375 | 0.32 | radial |
|---|---|---|---|---|---|---|
| splitting of the p level (meV) | 579 | 481 | 491 | 501 | 507 | 514 |
| s level (Ry) | -0.114 | -0.164 | -0.230 | -0.271 | -0.298 | -0.388 |

The Gaussians of these potentials are 0.46 to 0.98 bohr wide, and
piecewise-linear elements do not resolve them at these spacings: the s level is
1.2 eV high at 0.32 bohr and closes more slowly than the second order of a
resolved potential, while the spin-orbit splitting, a ratio of matrix elements
on one projector channel, is within 2 % there.

```
python -m tessera.drivers.bands.spinorbit --divisions 12 16 20 24 --checkpoint state/ --out splitting.json
```

runs gallium arsenide on the conventional cell, one sheet and then two, and
reports the splitting of the top of the valence band (a triplet on one sheet, a
quartet above a doublet on two) after the first diagonalization and at
self-consistency, on every mesh and extrapolated, against the measured 0.341 eV
(`reference.GALLIUM_ARSENIDE`). The flags `--refinement-terms` and
`--lattice-images` are those of the table below; `--checkpoint` keeps the state
of every loop so that a later call continues it, and `--max-updates` bounds the
exchange updates of one call.

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
| `--self-energy-order` | terms of the expansion of the self-energy in the screened interaction $W$; 1 is $\Sigma = iGW$, 2 adds the crossed diagram, 3 the six skeleton diagrams of third order (`diagrams`) | 1 to 5, default 3; implemented: 1 to 3 |
| `--vertex-bands`, `--vertex-poles` | the modes nearest the gap on the internal lines of the diagrams beyond the first order, and the modes of the screened interaction kept in them; the cost of order $k$ grows as bands$^{2k-1}$ poles$^k$ | default 12 and 12 |
| `--zero-momentum-order` | how the self-energy integrand is averaged over the momentum transfers the sampling leaves out: 1 is the closed form at vanishing momentum; k is a midpoint grid of k transfers per axis, the Hartree-Fock pencil solved at every node, the singular part averaged analytically and the bounded remainder by the grid | 1 to 5, default 3; all implemented |
| `--refinement-terms` | terms of the refinement series of the zero-momentum constant | 1 to 5, default 5 |
| `--lattice-images` | periodic images per axis in the lattice sums | odd, default 5 |
| `--frequency-nodes` | terms of the Chebyshev series along the imaginary frequency axis (`KineticBasisScreening`) | at least 5, default 64 |
| `--divisions` | meshes; every mesh beyond the first removes one even order of the mesh error | default six meshes, five orders |

An order that is not implemented is refused by name before anything runs; it is
never replaced by a lower one (orders 4 and 5 of the expansion in the screened
interaction are). The diagrams beyond $\Sigma = iGW$ are evaluated as sums over
the orderings of their vertex times, in closed form on the poles of $W$, with
the instantaneous part of $W$ as lines whose two vertices share a time; the
test suite holds them to the closed form at first order, to the textbook
second-order exchange, to the plain frequency integrals of their Feynman rules
on the imaginary axis (the triangle loops included), and to the heavy-boson
limit for instantaneous lines. On a cell of 6 bohr the correlation self-energy of the filled level
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
