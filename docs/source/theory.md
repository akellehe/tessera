# Theory

Background reading for the rest of the docs: the lattice-spacetime ideas the
package implements.

Since Tullio Regge's work in the 1960s there has been sustained interest in
formulating general relativity so that it is compatible with quantum mechanics,
quantum field theory, and quantum chromodynamics.

Regge calculus is not computationally efficient for large simulations. Dynamical
triangulations are a later approach, largely subsumed into causal dynamical
triangulations (CDT).

CDT works with a *preferred foliation*: spacetime is sliced into spatial
hypersurfaces at integer time steps. Within a time slice, edges are spacelike;
edges connecting adjacent slices are timelike. The simplices spanning two
adjacent slices (the (4,1) and (3,2) types in 4D) contain both. Later work
explores models without a preferred foliation while fixing edge lengths or the
ratio of temporal to spatial edge lengths.

## Why the Lorentzian metric matters

Lorentzian path integrals are taken over an infinite domain and tend to diverge.
Euclidean path integrals are easier to evaluate, but they obscure causal
structure: in a Euclidean spacetime all points are effectively spacelike
separated, and unitarity is tied to the Lorentzian structure. See
[Loll, arXiv:hep-th/0011194](https://arxiv.org/abs/hep-th/0011194).

## Path integrals and Wick rotation

The path integral formulation of quantum mechanics, introduced by Richard
Feynman, gives the probability amplitude for a system to go from one state to
another. It sums over every path the system can take, weighted by the
exponential of the action in units of $\hbar$. The path integral underlies
lattice gauge theory and quantum chromodynamics:

$$
\langle x_f, t_f | x_i, t_i \rangle = \int \mathcal{D}[x(t)] e^{\frac{i}{\hbar} S[x(t)]}
$$

— the "sum over all possible histories".

In field theory $x(t)$ becomes a field $\phi(x)$ and $S[x(t)]$ becomes the
action functional $S[\phi]$:

$$
Z = \int \mathcal{D}[\phi] e^{\frac{i}{\hbar} S[\phi]}
$$

where $Z$ is the partition function. This is the Lorentzian path integral,
because $S[\phi]$ is computed with the Lorentzian metric $g_{\mu\nu}$.

The exponential oscillates rapidly for large actions and is not generally
convergent. A Wick rotation takes the integral from Lorentzian to Euclidean
signature,

$$
t \rightarrow -i \tau
$$

so that

$$
e^{iS} \rightarrow e^{-S_E}
$$

where $S_E$ is the Euclidean action. The result converges and behaves like a
statistical partition function, but it is no longer timelike: the rotation
obscures causal structure. There is no known general mapping between Euclidean
and Lorentzian metrics, or between their diffeomorphism equivalence classes.

## Quantum gravity

In quantum gravity the path integral takes the form

$$
Z = \int \frac{\mathcal{D}[g_{\mu\nu}]}{\mathrm{Diff}(M)} e^{\frac{i}{\hbar} S_{EH}[g_{\mu\nu}]}
$$

where $S_{EH}$ is the Einstein-Hilbert action,

$$
S_{EH} = \frac{1}{16 \pi G} \int d^4x \sqrt{-g} (R - 2 \Lambda).
$$

This integrates over equivalence classes of Lorentzian metrics, two metrics
being equivalent if they differ by a diffeomorphism — that is, over all
Lorentzian geometries modulo diffeomorphisms. It requires regularization or
contour deformation to be well defined.

## Lattice gravity approaches

### Regge calculus

Regge calculus lets edge lengths vary, giving a discrete approximation to
general relativity. Spacetime is a simplicial complex and curvature is
concentrated on the hinges, the $(D-2)$-dimensional simplices. The
Einstein-Hilbert action is expressed in terms of the deficit angles around those
hinges, giving a discrete gravitational action.

### Causal dynamical triangulations

CDT is a non-perturbative approach to quantum gravity that builds spacetime from
causally ordered simplices; see
[Loll, arXiv:1905.08669](https://arxiv.org/abs/1905.08669). The path integral
over geometries is approximated by a sum over causal triangulations. From
equation (5) of
[Loll, arXiv:hep-th/0011194](https://arxiv.org/abs/hep-th/0011194) the partition
function is

$$
Z(\lambda, G) = \sum_{\text{causal } T} \frac{1}{C_T} e^{iS^{\text{Regge}}}
$$

$$
e^{iS^{\text{Lor}}} \rightarrow e^{-S^{\text{Euclid}}}
$$

In Lorentzian signature, squared temporal edge lengths are negative,
$l_t^2 = -\alpha a^2$. The Wick rotation takes them positive for Euclidean
computation:

$$
l_t^2 = -\alpha a^2 \xrightarrow{\text{Wick}} l_t^2 = +\alpha a^2
$$

CDT is implemented as a Markov chain Monte Carlo that preserves causal
structure. A detailed description with source code is in
[Brunekreef, Görlich & Loll, arXiv:2310.16744](https://arxiv.org/abs/2310.16744).
The approach recovers classical spacetime at large scales while retaining
quantum effects at small scales; results appear in
[Jordan & Loll, arXiv:1305.4582](https://arxiv.org/abs/1305.4582),
[Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0604212](https://arxiv.org/abs/hep-th/0604212),
and
[Jordan & Loll, arXiv:1307.5469](https://arxiv.org/abs/1307.5469).

### Causal sets

Causal set theory posits that spacetime is fundamentally discrete: a set of
events with a partial order representing causal relationships, from which the
geometry emerges. Causal sets are locally finite — between any two events there
are finitely many intermediate events — which supplies a natural ultraviolet
cutoff.

## Connections between the three

All three discretize spacetime, in different ways.

- Regge calculus varies edge lengths in a simplicial complex and captures
  curvature through deficit angles.
- CDT builds spacetime from causally ordered simplices.
- Causal set theory abstracts spacetime to a set of events with causal
  relationships.

## Incompatibilities

Regge calculus and CDT both use simplicial complexes, but CDT imposes a strict
causal structure that Regge calculus does not require. Regge calculus allows
continuously varying edge lengths; CDT fixes them. Regge calculus allows
arbitrary triangulations; CDT restricts to causality-preserving ones, and
retriangulates by rules that preserve causality.

Causal set theory does not use a simplicial complex at all, focusing on the
causal relations between discrete events. The three therefore have distinct
mathematical frameworks, and direct comparison is difficult.

Regge calculus and CDT both avoid coordinates by discretizing spacetime
geometrically rather than analytically, which removes diffeomorphism redundancy.
Two triangulations differing only by vertex labels are the same geometry in CDT,
so the path integral becomes a sum over inequivalent triangulations rather than
an integral over $g_{\mu\nu}$ modulo diffeomorphisms.

### Path integrals

CDT's path integral, before Wick rotation, is

$$
Z = \sum_{T \in \text{Causal Triangulations}} \frac{1}{C(T)} e^{i S_{\text{Regge}}(T)}
$$

where $C(T)$ is the symmetry factor of the triangulation $T$ and
$S_{\text{Regge}}(T)$ is the Regge action, the discrete Einstein-Hilbert action.

Quantum Regge calculus instead replaces the continuum gravitational path
integral

$$
Z = \int_{\mathrm{Lor}(M)/\mathrm{Diff}(M)} \mathcal{D}[g_{\mu\nu}] e^{i S_{EH}[g_{\mu\nu}]/\hbar}
$$

with a discrete analogue integrating over edge lengths $l_{ij}$ for all
triangulations $\mathcal{T}$ of the manifold $M$:

$$
Z = \sum_{T \in \mathcal{T}} \int_{l_{ij} > 0} \prod_{i < j \in T} dl_{ij} \mu(l_{ij}) e^{i S_{\text{Regge}}([T,l_{ij}])/\hbar}
$$

where $\mu(l_{ij})$ is a measure factor on the edge lengths.

### Phase transitions

CDT fixes edge lengths and sums only over causal triangulations, enforced by the
time foliation. Both formulations sum over geometries and both yield a partition
function in the statistical-mechanics sense.

A phase transition occurs when changing a parameter produces a qualitative
change in the system's structure or large-scale behaviour. In CDT, varying the
coupling constants in the Regge action gives different phases of spacetime
geometry: one phase is an extended four-dimensional spacetime, others are
crumpled or degenerate. These transitions are central to the search for a
continuum limit.

### Observables

In Regge calculus curvature is computed from deficit angles. In CDT observables
are defined across the ensemble of triangulations. Spatial volume as a function
of discrete proper time encodes the effective curvature radius of spacetime,

$$
V_3(t) = \text{number of 3-simplices in time slice } t
$$

with ensemble average

$$
\langle V_3(t) \rangle = \frac{1}{Z} \sum_{T \in \text{Causal Triangulations}} V_3(t)_T e^{i S_{\text{Regge}}(T) / \hbar}
$$

which fits the classical Euclidean de Sitter solution in 4D:

$$
V_3(t) \propto \cos^3\left(\frac{t}{R}\right)
$$

The spectral dimension $D_S(\sigma)$ is measured by diffusion on the
triangulation: a random walk with diffusion time $\sigma$ has return probability
$P(\sigma) \sim \sigma^{-D_S/2}$, a scale-dependent measure of effective
dimensionality. In 4D CDT, $D_S$ flows from $\approx 2$ at short scales to
$\approx 4$ at large scales
([Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0505113](https://arxiv.org/abs/hep-th/0505113)).
Both observables are described in the 'Observables' section of
[Loll, arXiv:1905.08669](https://arxiv.org/abs/1905.08669).

Geodesic distance distributions are a third family of observables: measure the
volume of a geodesic ball as a function of its radius.

## References

1. T. Regge, *General relativity without coordinates*, Nuovo Cimento **19**
   (1961) 558.
2. R. Loll, *Discrete Lorentzian quantum gravity*,
   [arXiv:hep-th/0011194](https://arxiv.org/abs/hep-th/0011194)
3. J. Ambjorn, J. Jurkiewicz, R. Loll, *Emergence of a 4D world from causal
   quantum gravity*,
   [arXiv:hep-th/0404156](https://arxiv.org/abs/hep-th/0404156)
4. J. Ambjorn, J. Jurkiewicz, R. Loll, *Reconstructing the Universe*,
   [arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)
5. J. Ambjorn, J. Jurkiewicz, R. Loll, *Spectral dimension of the universe*,
   [arXiv:hep-th/0505113](https://arxiv.org/abs/hep-th/0505113)
6. J. Ambjorn, J. Jurkiewicz, R. Loll, *Quantum gravity, or the art of building
   spacetime*, [arXiv:hep-th/0604212](https://arxiv.org/abs/hep-th/0604212)
7. S. Jordan, R. Loll, *Causal dynamical triangulations without preferred
   foliation*, [arXiv:1305.4582](https://arxiv.org/abs/1305.4582)
8. S. Jordan, R. Loll, *De Sitter universe from causal dynamical triangulations
   without preferred foliation*,
   [arXiv:1307.5469](https://arxiv.org/abs/1307.5469)
9. R. Loll, *Quantum gravity from causal dynamical triangulations: a review*,
   [arXiv:1905.08669](https://arxiv.org/abs/1905.08669)
10. J. Brunekreef, A. Görlich, R. Loll, *Simulating CDT quantum gravity*,
    [arXiv:2310.16744](https://arxiv.org/abs/2310.16744)
11. L. Bombelli, J. Lee, D. Meyer, R. Sorkin, *Space-time as a causal set*,
    Phys. Rev. Lett. **59** (1987) 521.
