# Geometric harmonic states and operators

This specification governs `examples/cobordism/qubit_animation.py` (#1096),
including its read-only `verify` (#1092, #1103, #1109) and `theta` (#1107)
subcommands, and `tessera/quantum/theta_register.py`. It supersedes
interpreting a selected-state amplitude fit as proof of an operator
correspondence. Primitive data remain simplex topology, scalar edge lengths
and scalar phases. No prescribed matrix-valued connections are allowed.

## 1. Scientific question

Can internal geometry realize an operator through the whole Laplacian's zero
modes? Test a complete input basis and unseen superpositions on frozen
geometry. Report three separate claims: an identifiable linear relation;
preservation of declared positive quantum norms; and realization of the
requested gate in the required subsystem tensor coordinates. A failed
realization is a result. Numerical evidence is not an exact existence theorem,
except where Section 3 states a theorem of simplicial cohomology and the
numerics verify it at machine precision; Section 7 records the status of
every claim as exact, measured, or open.

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

### Restriction, monodromy and composition

For a cobordism W with boundary M_0 ⊔ M_1, markings giving each H^1(M_i;Z)
a basis, Z spanning the whole's degree-1 harmonic space, and P_A, P_B the
transported period matrices of Z over the two markings, the following are
theorems of simplicial cohomology in any signature, given dim ker L_1 = b_1:

    rank [P_A; P_B] = b_1(dW)/2, and im(H^1(W) -> H^1(dW)) is isotropic for
        Omega = J_A (+) (-det phi*) J_B, the boundary intersection form built
        from declared data (each surface's side of the bulk, and the
        orientation of the relabelling phi applied to it), never from det M;
    M = P_B P_A^-1 is independent of the basis Z, an integer matrix with
        det M = +-1 (invertible P_A), equal to phi* for a relabelled collar,
        and independent of the metric;
    gluing composes, M(W_2 o W_1) = M(W_2) M(W_1), and a disjoint union gives
        the direct sum.

T_G above is this M for R_in = P_A, R_out = P_B. The metric enters only in
(i) whether dim ker L_1 = b_1 holds, a condition on the Lorentzian locus that
is certified, not assumed (next subsection); (ii) the boundary pairings, and
hence whether M preserves them; (iii) the harmonic representative and its
frame. M on periods is topological: stage-1 topology moves change it,
stage-2 metric relaxation cannot. On the two-torus collar M ∈ GL(2,Z), so a
fixed triangulation yields a discrete set of operators; the continuous part
of any operator lives in the metric-dependent pairing, not in M.

Measured by `verify` on the seeded collars (grid 3, seed 7, tolerance 1e-12):
M rounds to an integer with residual <= 1e-14 and equals I (product collar)
or the swap (relabelled collar); the restriction subspace is isotropic for
the declared form to <= 1.1e-14 with the orientation-flipped control at 1.41
(two tori) and 2.00 (four); a 75% complex jitter of every length moves M by
<= 1.6e-14 while the canonical representative Z P_A^-1 moves by 0.49-0.77
with the difference in im d_0 to <= 4e-14; the twist group composes across
separately seeded collars to <= 1.5e-14; four tori joined through a removed
tetrahedron give the direct sum with off-block norm <= 5e-15. On this
triangulation the simplicial mapping classes are {+-I, +-swap}, abelian, so
the order of composition is not testable there, and no host with an interior
seam has been read: C1 is a group-law control, not a seam-gluing test.

### Harmonic certificates

Rank equal to b_1 is necessary and not sufficient for a contour band to be
the harmonic space on complex lengths. In the chain formulation, with Z the
band's cochain images and Phi its chain frame (Z = G^U Phi), the band is
harmonic when

    (d_2^{U^-1})^T Z = 0   (closed, on the images)
    d_1^U Phi = 0          (co-closed, on the frame),

an independent null-space read has nullity = band rank = b_1 with the same
span, and the band's own certificate (projector idempotency, rank tolerance,
singular gap, resolvent bound) is finite. A kernel-dimension mismatch is the
geometry failing dim ker L_1 = b_1 and is named as that, never repaired.
Measured before and after the jitter on every seeded host: closed and
co-closed residuals <= 2.5e-14, nullity = rank = b_1 (2 and 4), span
residuals <= 1.2e-14, idempotency <= 7e-16.

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
cochain bidegree and isolate the intended harmonic tensor sector. On the
modulus register this stands as written. The theta quantization register
(next subsection) supplies a subsystem-factor map of a different kind: it
quantizes the marking lattices, on which a direct sum is a tensor product.

Norm preservation on the modulus register is measured, not assumed: with G_i
the covariant dual-frame pairing of the boundary torus in its marking
coordinates (S4's rule, (F^vee)^T M_own F = I), the transport defect
|M^T G_B M - G_A| / |G_A| is 0.46-0.58 on the seeded hosts with generic
moduli, jitter-dependent, and 1e-14 for identical square tori on the product
collar. It is bilinear; it is quantum unitarity only where the Whitney form
is positive Hermitian, the real locus, and Krein pseudo-unitarity elsewhere.

### Theta quantization register

The marking lattice H^1(M;Z) of a boundary torus is taken as a weight
lattice and the Hilbert space as its exponential: the level-k theta functions
on the Jacobian C/(Z + tau Z),

    theta_j(z; tau) = sum_n exp(pi i k (n + j/k)^2 tau + 2 pi i k (n + j/k) z),

j in Z/k, a k-dimensional space Theta_k(tau); k = 2 is a qubit. For g tori
with period matrix Omega the characteristics are j in (Z/k)^g, and for
block-diagonal Omega the lattice sum factorises exactly, so

    Theta_k(tau_1 (+) tau_2) = Theta_k(tau_1) (x) Theta_k(tau_2).

A symplectic integer M = [[A,B],[C,D]] acts by Omega -> (A Omega + B)(C
Omega + D)^-1, z -> (C Omega + D)^-T z, and carries the theta space to
itself; its Weil matrix rho_k(M) is DETERMINED by a least-squares fit of that
transformation law over sampled z, never typed in, with the residual as the
certificate. rho_k(M) is unitary, projective, and independent of Omega; an
anti-symplectic M (M^T J M = -J, the orientation-reversing swap) factors as
M = M+ R with R = diag(I,-I) acting by Omega -> -conj(Omega) and complex
conjugation on the basis, so rho(M) = rho(M+) K is antiunitary. At k = 2:
rho(S) is the Hadamard gate, rho(T) the phase gate diag(1,i), and the
genus-2 shear Omega -> Omega + [[0,1],[1,0]] is controlled-Z. A
block-diagonal M acts as rho(M_1) (x) rho(M_2), operator Schmidt rank 1; a
monodromy that mixes the two lattices is entangling.

The register reads what the geometry already produced -- each torus's
modulus and the integer monodromy between markings -- and introduces no
matrix-valued connection: non-abelian structure would enter through a
finite symmetry group acting simplicially (a covering), tensor structure
through this factorisation, and the continuous part of an operator through
the Weil representation of the discrete monodromy. It re-identifies the
state: on the modulus register the qubit is tau, the holomorphic line in
H^1 (x) C; here tau is the polarization and the state is a section over the
Jacobian, and the two are different physical proposals. Nothing on the
modulus register is altered by it.

Measured (`theta`, tolerance 1e-12, level 2): transformation-law residuals
<= 6.8e-16 for S, T and the swap at three moduli; unitarity defect <= 1.3e-15;
group law <= 9.5e-16; Hadamard and phase gate to <= 4.4e-16; factorisation to
1.9e-16 with rho(S (+) T) = rho(S) (x) rho(T) to 6.2e-16 at Schmidt rank 1;
controlled-Z to 5.3e-16 at Schmidt rank 2; rho identical across moduli to
8.8e-16. On the seeded hosts the product collar acts as the identity, the
swap collar antiunitarily, and the four-torus join along a sphere as a
product operator of Schmidt rank 1: it cannot entangle.

Choi input/reference entanglement is not physical-boundary entanglement;
operator-vector entropy is not total-system entropy. A projection remainder
is not automatically an environment. Joint-state instrumentation is #1095.

## 4. Current experiment and interpretation

S1. Retain flat-torus inputs, markings, holomorphic fibers and own-state residuals.
S2. Retain scalar-geometry collar synthesis and the shared engine-unit schedule.
S3. Record the actual objective. Historical transfer/chi fitting optimizes
    selected-state amplitudes, not the recovered whole operator.
S4. Independently recover whole-kernel relations every frame through both
    transported periods and bilinear Gram contractions (F^vee)^T M_whole Z.
    Reuse the live marked block's native dual frame, normalized by
    (F^vee)^T M_own F = I on the block's own Whitney metric, then embed it
    in the whole. Use the existing PencilSchur.gramBlock primitive; do not
    reconstruct the Gram matrix or dual normalization independently.
    The contraction uses transpose, not adjoint, and M_whole rather than
    M_own. A primal-primal contraction F^T M_whole Z is not gauge covariant.
    Periods are a topological control; the dual-frame Gram contractions are
    distinct metric-dependent observations, not an orthogonal projection.
    Identity coordinate norms for these ports are explicit external encoding
    conventions, not claims that the Whitney metric is positive or that the
    two readouts have the same physical interpretation. The existing `gram`
    record key carries `observation_convention: dual_frame_whitney`; older
    records without that field used primal frames and are not numerically
    interchangeable, even at zero phases, because their normalization differs.
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
    On the theta register two tori do carry Theta_2 (x) Theta_2, but every
    host joined along a sphere has a block-diagonal monodromy and acts as a
    product operator (measured Schmidt rank 1, off-block norm <= 5e-15); an
    entangling gate needs a monodromy mixing the two lattices, a join along
    a circle (a 1-handle), which no current seed constructs.

## 5. Implementation and compatibility

D1. Preserve native collar/bridge gates and rollback behavior.
D2. Preserve own-surface frames and block-state residuals.
D3. Preserve existing bilinear pencil transfer as a diagnostic distinct from
    whole transport and from a certified quantum channel.
D4. Add the scientific readout and verdict in `qubit_animation.py`, using
    reusable numerical helpers where appropriate. Preserve CLI, geometry
    records, output handling and neutral-runtime behavior. Display actual
    whole-operator evidence without deleting historical measurements.
D5. `verify` and `theta` are reads of the fixed complex: they seed through
    `seed_qubit_host` (the first half of `build_qubit_node`, no node, no
    objective, no readout) and never run a stage. `MultiCobordism.restriction`
    returns one harmonic basis with every marking's periods from it, the
    band's chain frame and its certificate; `monodromy` is that read with the
    fit added. Records are written under `~/cobordism-runs/functor-verify/`
    and `~/cobordism-runs/theta-register/`, named with the host parameters
    including the moduli; each carries a `certifies` sentence stating what a
    pass establishes and what it does not. `run.sh` runs both suites.

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
    A pure-gauge phase change fixed to one at both port basepoints must leave
    both recovered operators unchanged, for real and complex lengths. With
    g(v)=exp(i phi(v)), unfixed basepoints instead give
    T' = exp(-i (phi_out-phi_in)) T. The normalized operator's Schmidt
    coefficients, marginals and Choi entropy remain unchanged in either case.
C5. Headless/live paths share records. JSON is finite-safe; rendered panels
    name the claim and obstruction; CLI/runtime regressions remain covered.
C6. An obstructed gate realization stays obstructed when the selected-state
    objective converges. Report the actual measured outcome.
C7. `verify` on two tori (product and swap collars, and identical square
    tori) and four tori (product and swap) passes every check at tolerance
    1e-12: R1-R5 (rank, integrality, declared-form isotropy on the
    orthonormalised subspace with the flipped control nonvanishing), H1-H3
    (closed, co-closed, independent kernel, band certificate; before and
    after the jitter), J1-J3 (M fixed, representative moves exactly, covariant
    Gram moves), C1, D1, U1. 17/17, 17/17, 17/17, 27/27, 27/27. Regressions
    pin the review cases: the 4x4 diag(S,S) isotropy, the (1, 1e-5)
    exactness residual, the isometric start, the 1e-7 basis rescaling, the
    zero-phase agreement of the dual-frame and primal Grams, and gauge
    invariance under U(1) and C* vertex gauges.
C8. `theta` passes T1-T8 on the same hosts: 8/8, 8/8, 10/10, 10/10.

## 7. Status of claims

Exact, as theorems verified at machine precision on the seeded collars:
the restriction subspace is Lagrangian and of rank b_1(dW)/2; the monodromy
is an integer matrix of det +-1, basis- and metric-independent, equal to the
relabelling's induced map; the twist group composes; four tori give the
direct sum; the harmonic band is closed and co-closed with an independent
kernel of dimension b_1; on the theta register a direct sum of lattices is a
tensor product, the integer monodromies act as Weil matrices (Clifford gates
at level 2, antiunitary when orientation-reversing), a block-diagonal
monodromy is a product operator and the genus-2 shear is controlled-Z.

Measured and metric-dependent, on these hosts only: the transport defect
(0.46-0.58 generic, 1e-14 for identical tori) and the motions of the
representative and its covariant Gram under the jitter (0.49-0.77,
0.18-0.45).

Open. No interior seam has been glued (C1 compares separately seeded
collars), and composition order cannot be tested on the grid torus, whose
simplicial mapping classes commute; a hexagonal-lattice torus (dihedral
symmetry) is the first host that could test both. Quantum unitarity on the
modulus register is certified only on the real locus; off it the pairing is
Krein. Tensor structure on the modulus register needs product boundaries
(Kunneth) with the register at degree 2; on the theta register an entangling
gate needs a monodromy mixing two lattices, a join along a circle, which no
seed constructs. Non-abelian levels (Weyl-alcove fusion on a triangulation
with a rank-2 Weyl group) are not built. Every measurement is a read of a
fixed complex: nothing here shows that the dynamics respects the functor.
The original hypothesis -- that the whole Laplacian's harmonic space is a
geometric proxy for states and Choi-decomposed operators, exactly analogous
to algebra -- is established at the level of cohomology and the register
built on it, and remains a program at the level of unitarity, tensor
composition and dynamics.

## References

- [Finite element exterior calculus](https://arxiv.org/abs/0906.4325): positive
  Hodge structure, not positivity of the complex Lorentzian pencil.
- [Watrous, channel representations](https://cs.uwaterloo.ca/~watrous/TQI-notes/TQI-notes.05.pdf):
  vectorization, Choi and Kraus conventions.
- `simplicial_qubit_spec.md`, `include/chainhodge/ChainHodge.h`,
  `include/cobordism/PencilLayer.h`, `include/cobordism/MultiCobordism.h`.
