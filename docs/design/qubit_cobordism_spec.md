# Geometric harmonic states and operators

This specification governs `examples/cobordism/qubit_animation.py` (#1096).
It supersedes interpreting a selected-state amplitude fit as proof of an
operator correspondence. Primitive data remain simplex topology, scalar edge
lengths and scalar phases. No prescribed matrix-valued connections are allowed.

## 1. Scientific question

Can internal geometry realize an operator through the whole Laplacian's zero
modes? Test a complete input basis and unseen superpositions on frozen
geometry. Report three separate claims: an identifiable linear relation;
preservation of declared positive quantum norms; and realization of the
requested gate in the required subsystem tensor coordinates. A failed
realization is a result. Numerical evidence is not an exact existence theorem.

## 2. Binding construction rules

R1. A state is a whole harmonic cochain, boundary included. Restrictions are
    readouts, not the definition of the state.
R2. Reuse the Whitney pencil, `MultiCobordism`, fiber machinery and shared
    runtime from `emergence_animation.py`. Keep `qubit_animation.py` canonical.
R3. Preserve pinned-boundary defaults and the explicit `--no-pin-boundary`
    alternative. Pinning never removes boundary cells from the pencil.
R4. Preserve the manifold gate, collar seed, stage-1 search and stage-2 scalar
    length/phase relaxation. Never insert the target in a link, measured frame
    or assembled geometric Laplacian.
R5. Targets may condition synthesis. Frozen recovery accepts no target,
    desired outputs or fitted witnesses; only geometry and declared readouts.
R6. Preserve complex/Lorentzian geometry. A complex bilinear Whitney pairing
    is not automatically a positive quantum metric. Do not silently repair it.
R7. Read the zero kernel, not an excited band. Distinguish algebraic kernel,
    generalized zero eigenspace, and closed/coclosed space on complex pencils.

## 3. Mathematical contracts

### Harmonic space and selected state

Let k denote cochain degree, D geometric dimension, and r complex harmonic
dimension. With positive Hermitian metrics and adjoint dagger,

    Delta_k = d_(k-1) d_(k-1)^dagger + d_k^dagger d_k
    <h,Delta_k h> = ||d_(k-1)^dagger h||^2 + ||d_k h||^2.

Its kernel is the closed, coclosed space. An orthonormal frame identifies it
isometrically with C^r. A state ray additionally needs input data or a
geometric selection rule. On a flat torus with tau=a+ib, b>0,

    g_tau = [[1,a],[a,a^2+b^2]]/b
    omega_tau = dx+tau dy,  star omega_tau = -i omega_tau.

Periods (1,tau) give psi=(1,tau)/sqrt(1+|tau|^2). Its norm is the declared
identity metric on period coordinates, not assumed Whitney orthonormality.
Fixed orientation covers one hemisphere in this positive-real construction;
complex continuation has separate diagnostics in `simplicial_qubit_spec.md`.

### Identifiable whole-kernel operator

Assemble K(lambda)=A_tilde-lambda M. For Z spanning ker A_tilde at zero, set

    A=R_in Z, B=R_out Z.

An operator on every input exists precisely when A is onto and ker A is
contained in ker B. For square invertible A, T_G=B A^-1. Otherwise choose a
right inverse C, verify B(I-CA)=0, then use T_G=BC. A pseudoinverse without
these checks is not an operator certificate. Record numerical thresholds,
ranks, conditioning and full-space kernel residuals.

For basis and held-out x, reconstruct a whole h, then independently check
R_in h=x, A_tilde h=0 and R_out h=T_G x. A minimum Euclidean image-norm
witness resolves unobserved modes computationally, not physically. The
operator is invariant under Z->ZS; port-frame changes instead give
T->g_out^-1 T g_in with correspondingly transformed metrics.

### Internal geometry and composition

Partition retained/internal coordinates b,z. When K_zz(0) is invertible,

    z=-K_zz^-1 K_zb b
    S_G=K_bb-K_bz K_zz^-1 K_zb.

This is exact elimination, not a partial trace; internal coordinates remain
part of the reconstructable whole state. In positive Hermitian coordinates,
S_G=[-T,I]^dagger W[-T,I], W positive definite, implies y=Tx for b=(x,y).
The effective matrix must come from scalar geometry. A target-built block
matrix is an analytical fixture only, never successful geometric synthesis.

Gluing y=T1 x and z=T2 y gives z=T2 T1 x. Eliminating y from the two squared
constraint energies yields

    (z-T2 T1 x)^dagger (I+T2 T2^dagger)^-1 (z-T2 T1 x).

Actual geometric composition requires identified interface cells/frames and
comparison with the assembled relation. Arbitrary coupling blocks need not
multiply. Additional dimension/topology may supply internal modes, but does
not prove realizability. In particular, a connected positive scalar
degree-zero connection graph has at most one zero mode: each edge imposes
f_j=exp(i theta_ij)f_i. Higher-degree harmonic spaces are the candidates.

### Quantum norms, tensor factors and Choi representation

Declare positive port metrics Q_in,Q_out. Norm preservation requires
T^dagger Q_out T=Q_in. In orthonormal coordinates with output index first,

    vec(T)=sum_(i,j) T_ij |i>_out tensor |j>_in
    (I tensor <conjugate(x)|)vec(T)=T x.

Normalize nonzero vec(T) by ||T||_F. Its Schmidt coefficients are normalized
singular values. The input marginal is (T^dagger T)^T/||T||_F^2, equal to I/r
for a unitary. A general channel has J=sum_a vec(K_a)vec(K_a)^dagger and
Tr_out J=I in the unnormalized convention. A zero map has no normalized Choi ray.

A direct sum of torus frames is not a two-qubit tensor product. A two-qubit
state has four amplitudes, its operator is 4x4 and its operator vector has
sixteen amplitudes. Specify a geometric subsystem-factor map; dimension
matching is insufficient. Product-complex constructions must identify their
cochain bidegree and isolate the intended harmonic tensor sector.

Choi input/reference entanglement is not physical-boundary entanglement;
operator-vector entropy is not total-system entropy. A projection remainder
is not automatically an environment. Joint-state instrumentation is #1095.

## 4. Current experiment and interpretation

S1. Retain flat-torus inputs, markings, holomorphic fibers and own-state residuals.
S2. Retain scalar-geometry collar synthesis and the shared engine-unit schedule.
S3. Record the actual objective. Historical transfer/chi fitting optimizes
    selected-state amplitudes, not the recovered whole operator.
S4. Independently recover whole-kernel relations every frame through both
    transported periods and bilinear Gram contractions F^T M Z using live
    marked block frames F embedded in the whole. Periods are a topological
    control; Gram contractions are distinct metric-dependent observations.
    Identity coordinate norms for these ports are explicit external encoding
    conventions, not claims that the Whitney metric is positive or that the
    two readouts have the same physical interpretation.
S5. Historical chi is a 2x2 selected-pair amplitude matrix, not the full XY or
    named two-qubit gate. Keep it for reproducibility; never pass it to the
    frozen operator recovery routine.
S6. Publish `correspondence`: identifiability, ranks, conditions, basis and
    held-out reconstruction errors, coordinate norm preservation, normalized
    operator-vector/Choi data and a separate requested-gate verdict.
S7. The current two-torus collar's rank-two whole space supports a one-register
    transport control, not a four-dimensional two-qubit gate. Report that
    obstruction even if the historical objective converges. A higher-dimensional
    successful realization still requires a construction meeting Section 3.

## 5. Implementation and compatibility

D1. Preserve native collar/bridge gates and rollback behavior.
D2. Preserve own-surface frames and block-state residuals.
D3. Preserve existing bilinear pencil transfer as a diagnostic distinct from
    whole transport and from a certified quantum channel.
D4. Add the scientific readout and verdict in `qubit_animation.py`, using
    reusable numerical helpers where appropriate. Preserve CLI, geometry
    records, output handling and neutral-runtime behavior. Display actual
    whole-operator evidence without deleting historical measurements.

## 6. Acceptance checks

C1. Live collar: kernel residual, input coverage, unique output, basis and
    fixed-seed held-out whole-cochain reconstruction. No target in recovery.
C2. Independent analytical fixtures: complex and rectangular operators,
    hidden modes, ambiguous output, incomplete coverage, zero maps and rank
    loss. Rebase harmonic frames and check invariance.
C3. Choi contraction, marginals, norm preservation and composition identities.
    No unconditional unitary certificate for nonunitary or zero operators.
C4. Perturb native scalar geometry. Period transport stays topological on a
    fixed marked collar; Gram observations can change. Neither proves universality.
C5. Headless/live paths share records. JSON is finite-safe; rendered panels
    name the claim and obstruction; CLI/runtime regressions remain covered.
C6. An obstructed gate realization stays obstructed when the selected-state
    objective converges. Report the actual measured outcome.

## References

- [Finite element exterior calculus](https://arxiv.org/abs/0906.4325): positive
  Hodge structure, not positivity of the complex Lorentzian pencil.
- [Watrous, channel representations](https://cs.uwaterloo.ca/~watrous/TQI-notes/TQI-notes.05.pdf):
  vectorization, Choi and Kraus conventions.
- `simplicial_qubit_spec.md`, `include/chainhodge/ChainHodge.h`,
  `include/cobordism/PencilLayer.h`, `include/cobordism/MultiCobordism.h`.
