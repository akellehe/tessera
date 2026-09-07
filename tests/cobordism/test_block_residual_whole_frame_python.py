# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The whole's zero mode in the block's live frame: the OUTPUT-state read
(#975, T2-bis of ``docs/design/qubit_cobordism_spec.md``; since the D2 wording
of 2026-09-07 (#988) this leak is REPORTED, not scored -- ``r_U`` scores
``own_state_residual``, the block's own-Laplacian residual of D2, tested in
``test_own_state_residual_python.py``; ``input_state_residual`` and its gradient
remain the output-state read and its derivative, exactly as measured below).

A surface block holds its MARKING with its input coefficients
(``set_input_marking(index, cycles, (a, b))``: the cycles A, B as closed walks
in host vertex ids, each ordered into one closed walk and rotated to the
common base point -- the first vertex of A's walk on B's, the
``SimplicialQubit.base_vertex`` rule). The engine DERIVES the block's frame at
every read (``derive_input_frame``): the zero mode of the block's own
covariant Whitney pencil on its live surface (lengths and phases,
``block_surface_subcomplex``), normalized to transported periods (1, 0) and
(0, 1) over the cycles from the base point (``Connection.transportedPeriod``),
with dual images from the DUAL kernel (the same zero mode under the inverse
links) so that ``F^vee.T @ M_1^U @ F == I`` on the block's own pencil -- T3's
contract, paired between the kernel and the dual kernel as the qubit spec
section 16 states. The block residual (``input_state_residual``) is the leak
of the coefficients written through that frame, ``t = F @ (a, b)``, in the
zero mode of the ENTIRE cobordism at the whole's harmonic contour restricted
to the block's edges (the whole-complex fiber residual, one target per
block), scored in r_U at ``set_input_residual_weight`` in place of the
own-kernel leak of T2 (which a frame always contains, so it is zero for every
state and stays a labelled diagnostic, ``fiber_residual_for_input_block``).
Its analytic gradient differentiates the leak with the MOVING target: the
whole's band derivative plus ``dt = (dZ Pi^-1 - F dPi Pi^-1) (a, b)`` from the
block's own band, dPi the transported periods of dZ. The two-body transfer
reads the derived frames; the monodromy read takes its periods with parallel
transport; ``read_input_state`` reports the coefficients of the whole's zero
mode in the live frame -- the transported periods of the least-squares
combination that fits the target -- next to the inputs and the residual.

The fixture is T2's and T3's: collar nodes from two ``SimplicialQubit.flat_torus``
inputs (3x3 and 4x4; tau_A = 0.3 + 1.1i, tau_B = -0.2 + 0.8i), Whitney pencil
metric source, the holomorphic forms attached as the state fibers with the
harmonic contour, plus the degree-0 tetrahedral node of the two-body tests
for bit-identity.

Observed (OMP_NUM_THREADS=8):

* (a) the derived frame on the collar seed equals ``period_frame()`` of each
  input torus through the id map to 1.7e-15 (3x3) and 1.1e-15 (4x4), its
  dual satisfies ``F^vee.T M_1 F = I`` to 5.6e-16 and equals T3's
  ``input_frame_dual`` of the period frame to 1.5e-15; base vertices 0 and 9
  (3x3), 0 and 16 (4x4) -- the tori's base vertex 0 through the id map --
  and the walks are the tori's walks; the refusals are by name;
* (b) the block residual on the seed equals T4's restricted leak of the
  holomorphic form (the leak is projective in the target): 3.099981e-3 and
  9.344559e-3 at 3x3, 4.006415e-3 and 1.171689e-2 at 4x4 -- not zero, while
  the own-kernel leak of T2 stays 2e-30; r_U is the weighted sum of the two
  block residuals plus the two-body residual; the coefficients of the whole's
  zero mode in the live frames are (0.99825 + 0.00114i, 0.29847 + 1.09505i)
  and (0.99782 + 0.00318i, -0.20131 + 0.78407i) against the inputs
  (1, 0.3 + 1.1i) and (1, -0.2 + 0.8i); the transfer in the derived frames
  equals T3's in the supplied period frames to 2.2e-16 (3x3) and 3.0e-16
  (4x4): [[-0.01592, -0.01936], [-0.00433, -0.00569]], Schmidt (2.606e-2,
  2.612e-4);
* (c) under a pure gauge on the whole (phi_e = g(target) - g(source) on every
  host edge, both tori's vertices) the derived frames equal the gauged tori's
  period frames, the coefficient pairs and the block residuals are gauge
  invariant to rounding, the transfer is invariant up to the base-point
  factor exp(i(g(v_A) - g(v_B))) of the two markings and the two-body leak
  exactly, and the monodromy is invariant up to the same factor;
* (d) the gradient of each block residual satisfies the Euler identity
  sum_e s_e dF_e = 0 to 1.9e-15 (3x3), is nonzero on the bulk edges
  (|g| 3.2e-2) and on both tori's own edges (1.7e-2, 2.0e-2; the target's
  motion), agrees with a central difference on a torus edge to 6e-10 and on
  a bulk edge to 2e-9, and ``fiber_mode_ascent`` is the weighted sum of the
  two block gradients and the two-body gradient (1e-16);
* (e) stage 2 on the 3x3 collar (Regge term on, real locus, weight 1e6, 20
  steps, 383 s): the block residuals descend 3.100e-3 -> 9.98e-5 and
  9.345e-3 -> 7.50e-5, the whole's coefficients in each frame move toward
  (1, tau_in) (largest distance 5.2e-3 -> 1.2e-3 and 1.6e-2 -> 2.0e-3),
  objective 12568 -> 299 (the block terms dominate at this weight) while the
  Regge term moves 123.12 -> 123.70 and the two-body leak against chi(S5)
  rises 0.5262 -> 0.6394 (as T3 recorded under a dominated objective; not
  asserted); the lengths stay real and spacelike; the own-kernel leaks of T2
  leave their floor (2e-30 -> 2.4e-3 and 2.8e-2) since they are no longer
  scored: the tori's own metrics move while the whole's zero mode is held at
  the inputs in their live frames, which is D2 as revised;
* (f) an ordinary degree-0 node, T3's collar with supplied frames and T2's
  unmarked collar (residuals, transfer, gradients, edges) and the monodromy
  read on zero phases are bit-identical to a saved dump of origin/main's
  build (d42c24d) in a second venv.
"""
import itertools
import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
TAU_A = complex(0.3, 1.1)
TAU_B = complex(-0.2, 0.8)
FLOOR = 1e-24
DUMP = pathlib.Path(__file__).with_name("data") / "block_residual_whole_frame_dump.json"
# T4's restricted leaks of the holomorphic forms in the whole's zero mode on
# the collar seed (3x3 and 4x4), which the block residual reproduces.
SEED_RESIDUALS = {3: (3.099981154846e-3, 9.344558825278e-3), 4: (4.006415020773e-3, 1.171688659780e-2)}


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def quiet(build):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def torus(tau, n):
    return quiet(lambda: obs.SimplicialQubit.flat_torus(tau, n, n))


def state_fiber(q, ids, contour_on=None):
    """The torus's holomorphic form on the host's edge ids as a degree-1 fiber
    with the harmonic contour of the torus's own pencil."""
    f = cob.BoundaryFiber()
    f.degree = 1
    f.cells = [sorted((ids[int(i)], ids[int(j)])) for i, j in q.edges()]
    f.images = np.asarray(q.holomorphic_form(), dtype=complex).reshape(-1, 1)
    f.contour = cob.PencilLayer.harmonic_contour(
        cob.PencilLayer.assemble([contour_on if contour_on is not None else q.spacetime()]), 1)
    return f


def host_marking(q, ids):
    """The torus's marking as cycles of directed host steps (u -> v): a step
    (edge index, sign) is the edge's (i, j) as (i -> j) for +1 and (j -> i)
    for -1, through the id map (the `monodromy` convention)."""
    edges = list(q.edges())

    def cycle(steps):
        out = []
        for e, sign in steps:
            i, j = edges[int(e)]
            out.append((ids[int(i)], ids[int(j)]) if sign > 0 else (ids[int(j)], ids[int(i)]))
        return out

    return [cycle(q.cycle_A()), cycle(q.cycle_B())]


def collar(n, weight=1e6, einstein_hilbert=False, real_squared_lengths_only=False, seed_value=0, layers=1):
    qa, qb = torus(TAU_A, n), torus(TAU_B, n)
    seed = MC.seed_collar(qa.spacetime(), qb.spacetime(), layers)
    node = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value,
              einstein_hilbert=einstein_hilbert, real_squared_lengths_only=real_squared_lengths_only)
    node.seed_inputs([sorted(ids.values()) for ids in seed.vertex_ids])
    node.use_fiber_residuals(True)
    node.set_input_residual_weight(weight)
    for i, q in enumerate((qa, qb)):
        f = state_fiber(q, seed.vertex_ids[i])
        node.attach_input_fiber(i, f, f.cells)
    return qa, qb, seed, node


def mark(node, tori, ids):
    """The tori's markings with the input coefficients (1, tau_in)."""
    markings = []
    for i, q in enumerate(tori):
        marking = host_marking(q, ids[i])
        node.set_input_marking(i, marking, [1.0 + 0j, complex(q.tau())])
        markings.append(marking)
    return markings


def frame_through_ids(q, ids, cells):
    """`period_frame()` of the torus on the host's cells (sorted pairs)."""
    F = np.asarray(q.period_frame(), dtype=complex)
    row = {tuple(int(v) for v in c): k for k, c in enumerate(cells)}
    out = np.zeros((len(cells), F.shape[1]), dtype=complex)
    for e, (i, j) in enumerate(q.edges()):
        out[row[tuple(sorted((ids[int(i)], ids[int(j)])))]] = F[e]
    return out


def own_mass_matrix(node, index, cells):
    own = MC.block_surface_subcomplex(node.inputs[index], node.spacetime())
    assembled = cob.PencilLayer.assemble([own])
    idx = cob.PencilLayer.indices_of(assembled, 1, [list(c) for c in cells])
    return np.asarray(cob.PencilLayer.pencil(assembled, 1).B)[np.ix_(idx, idx)]


def restricted_leak(st, cells, target):
    """T4's channel: the leak of `target` in the whole's zero mode (harmonic
    contour of the whole) restricted to `cells`, by PencilLayer alone."""
    assembled = cob.PencilLayer.assemble([st])
    contour = cob.PencilLayer.harmonic_contour(assembled, 1)
    read = cob.PencilLayer.read_boundary_fiber(assembled, 1, contour, [list(c) for c in cells])
    Z = np.asarray(read.images)
    c = np.linalg.lstsq(Z, target, rcond=None)[0]
    return float(np.linalg.norm(Z @ c - target) ** 2 / np.linalg.norm(target) ** 2), Z.shape[1]


def spin_half_chi(psi, phi):
    """chi of spec S5 for two spin-1/2 in the |0>, |1> bases."""
    lowering = np.array([[0, 0], [1, 0]], dtype=complex)
    raising = lowering.T
    return np.outer(lowering @ psi, raising @ phi) + np.outer(raising @ psi, lowering @ phi)


def edge_keys(st):
    return [(e.getSource().getId(), e.getTarget().getId()) for e in st.getEdgeList().toVector()]


def squared_lengths(st):
    return np.array([complex(e.getLength()) ** 2 for e in st.getEdgeList().toVector()])


def holomorphic(packed):
    """dF from the packed (2 Re dF, -2 Im dF)."""
    packed = np.asarray(packed)
    return 0.5 * (packed.real - 1j * packed.imag)


def euler_defect(st, gradient):
    s = squared_lengths(st)
    g = np.asarray(gradient)
    return abs(np.sum(s * holomorphic(g))) / (np.abs(g).max() * np.abs(s).max())


def region_masks(st, seed):
    keys = edge_keys(st)
    regions = [set(ids.values()) for ids in seed.vertex_ids]
    on = [np.array([u in r and v in r for u, v in keys]) for r in regions]
    return on[0], on[1], ~on[0] & ~on[1]


def central_difference(node, index, edge, h=1e-6):
    s0 = complex(edge.getLength()) ** 2
    edge.setLength(np.sqrt(s0 + h))
    plus = node.input_state_residual(index)
    edge.setLength(np.sqrt(s0 - h))
    minus = node.input_state_residual(index)
    edge.setLength(np.sqrt(s0))
    return (plus - minus) / (2 * h)


def gauge_function(n_vertices, seed):
    rng = np.random.default_rng(seed)
    return [complex(x) for x in rng.uniform(-math.pi, math.pi, size=n_vertices)]


def gauged_torus(q, g):
    """The torus read through its own Spacetime with the pure gauge
    phi_e = g(target) - g(source) on every edge's stored orientation."""
    st = q.spacetime()
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        edge.setPhase(g[v] - g[u])
    return st, quiet(lambda: obs.SimplicialQubit(st, list(q.cycle_A()), list(q.cycle_B())))


def hexc(z):
    z = complex(z)
    return [float(z.real).hex(), float(z.imag).hex()]


def unhex(pair):
    return complex(float.fromhex(pair[0]), float.fromhex(pair[1]))


def within_ulps(got, expected, ulps):
    return abs(got - expected) <= ulps * math.ulp(max(abs(expected), np.finfo(float).tiny))


def close_ulps(got, expected, ulps):
    got, expected = np.asarray(got), np.asarray(expected)
    scale = max(np.abs(expected).max(), np.finfo(float).tiny)
    return got.shape == expected.shape and np.abs(got - expected).max() <= ulps * np.finfo(float).eps * scale


# --------------------------------------------------------------------------- #
# (a) the derived frame on the seed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_derived_frame_equals_the_period_frame_on_the_seed(n, whitney_default):
    qa, qb, seed, node = collar(n)
    ids = seed.vertex_ids
    markings = mark(node, (qa, qb), ids)
    for index, q in enumerate((qa, qb)):
        marking = node.input_marking(index)
        assert marking is not None and marking.rank() == 2
        assert np.allclose(np.asarray(marking.coefficients), [1.0, complex(q.tau())])
        # the base point and the walks are the torus's through the id map
        assert marking.base_vertex == ids[index][int(q.base_vertex())]
        for cycle, walk in zip(marking.cycles, (q.walk_A(), q.walk_B())):
            assert [tuple(int(x) for x in step) for step in cycle] == \
                [(ids[index][int(u)], ids[index][int(v)]) for u, v in walk]
        derived = node.derive_input_frame(index)
        assert derived.derived() and derived.obstruction == "" and derived.kernel_rank == 2
        frame = derived.frame
        cells = [tuple(int(v) for v in c) for c in frame.cells]
        assert sorted(cells) == cells and len(cells) == len(q.edges()), "the block's own edges, canonical order"
        F, Fd = np.asarray(frame.images), np.asarray(frame.dual_images)
        assert F.shape == Fd.shape == (len(cells), 2) and frame.rank() == 2
        # (i) the frame IS the period frame of the input torus through the id map
        reference = frame_through_ids(q, ids[index], cells)
        agreement = np.abs(F - reference).max()
        assert agreement < 1e-12, agreement
        # (ii) periods (1, 0) and (0, 1) over the marking (plain sums on zero phases)
        row = {c: k for k, c in enumerate(cells)}
        periods = np.zeros((2, 2), dtype=complex)
        for c, cycle in enumerate(markings[index]):
            for u, v in cycle:
                periods[c] += (1 if u < v else -1) * F[row[(min(u, v), max(u, v))]]
        assert np.abs(periods - np.eye(2)).max() < 1e-12, periods
        assert np.abs(np.asarray(derived.periods) @ np.linalg.inv(np.asarray(derived.periods)) - np.eye(2)).max() < 1e-12
        # (iii) the dual: T3's contract on the block's own pencil, and T3's dual of the same frame
        M = own_mass_matrix(node, index, cells)
        contract = np.abs(Fd.T @ M @ F - np.eye(2)).max()
        assert contract < 1e-12, contract
        t3 = np.asarray(node.input_frame_dual(index, reference))
        assert np.abs(Fd - t3).max() < 1e-12
        print(f"\n[T2-bis] {n}x{n} block {index}: frame vs period_frame {agreement:.2e}, dual contract {contract:.2e}, "
              f"dual vs T3 {np.abs(Fd - t3).max():.2e}, base vertex {marking.base_vertex}")
    # the static form on the block reads the same frame
    static = MC.derive_frame(node.inputs[0], node.spacetime())
    assert np.array_equal(np.asarray(static.frame.images), np.asarray(node.derive_input_frame(0).frame.images))


def test_marking_refusals_are_by_name(whitney_default):
    qa, qb, seed, node = collar(3, weight=1.0)
    ids = seed.vertex_ids
    marking = host_marking(qa, ids[0])
    assert node.input_marking(0) is None and node.inputs[0].marking is None
    with pytest.raises(RuntimeError, match="carries no marking"):
        node.derive_input_frame(0)
    with pytest.raises(RuntimeError, match="carries no marking"):
        node.input_state_residual(0)
    with pytest.raises(RuntimeError, match="carries no marking"):
        node.read_input_state(0)
    with pytest.raises(IndexError):
        node.set_input_marking(2, marking, [1.0, TAU_A])
    with pytest.raises(ValueError, match="one coefficient per cycle"):
        node.set_input_marking(0, marking, [1.0])
    with pytest.raises(ValueError, match="all zero"):
        node.set_input_marking(0, marking, [0j, 0j])
    with pytest.raises(ValueError, match="not finite"):
        node.set_input_marking(0, marking, [1.0, complex(float("nan"), 0)])
    with pytest.raises(ValueError, match="at least one cycle"):
        node.set_input_marking(0, [], [])
    with pytest.raises(ValueError, match="self-loop"):
        node.set_input_marking(0, [[(0, 0)], marking[1]], [1.0, TAU_A])
    with pytest.raises(ValueError, match="not an edge of the live complex"):
        node.set_input_marking(0, [[(0, 5), (5, 0)], marking[1]], [1.0, TAU_A])
    other = host_marking(qb, ids[1])
    with pytest.raises(ValueError, match="outside input block 0"):
        node.set_input_marking(0, [other[0], marking[1]], [1.0, TAU_A])
    with pytest.raises(ValueError, match="does not form one closed walk"):
        node.set_input_marking(0, [marking[0][:2], marking[1]], [1.0, TAU_A])
    with pytest.raises(ValueError, match="share no vertex"):
        # A's cycle and the row loop through another vertex (a parallel loop): no common vertex
        a = marking[0]
        shifted = [(u + 1, v + 1) for u, v in a]  # the next row of the 3x3 grid, inside the torus
        node.set_input_marking(0, [a, shifted], [1.0, TAU_A])
    # a block without an attached fiber has no state to mark
    bare = MC(seed.host, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=0, einstein_hilbert=False)
    bare.seed_inputs([sorted(m.values()) for m in ids])
    with pytest.raises(RuntimeError, match="carries no attached fiber"):
        bare.set_input_marking(0, marking, [1.0, TAU_A])
    # a valid marking in any step order: the cycles come back as closed walks from the base point
    node.set_input_marking(0, [list(reversed(marking[0])), marking[1]], [1.0, TAU_A])
    walk = [tuple(int(x) for x in s) for s in node.input_marking(0).cycles[0]]
    assert walk[0][0] == node.input_marking(0).base_vertex and all(walk[k][1] == walk[k + 1][0] for k in range(len(walk) - 1))
    assert walk[-1][1] == walk[0][0]
    node.clear_input_marking(0)
    assert node.input_marking(0) is None


# --------------------------------------------------------------------------- #
# (b) the block residual on the seed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [3, 4])
def test_block_residual_on_the_seed_is_the_restricted_leak(n, whitney_default):
    qa, qb, seed, node = collar(n)
    ids = seed.vertex_ids
    own_before = [node.fiber_residual_for_input_block(i) for i in range(2)]
    mark(node, (qa, qb), ids)
    st = node.spacetime()
    residuals = [node.input_state_residual(i) for i in range(2)]
    for index, q in enumerate((qa, qb)):
        # T4's channel: the restricted leak of the holomorphic form, which is the same
        # target up to the scale P_A (the leak is projective in the target)
        leak, rank = restricted_leak(st, node.inputs[index].fiber.cells, np.asarray(q.holomorphic_form()).reshape(-1))
        assert rank == 2
        assert residuals[index] == pytest.approx(leak, rel=1e-9, abs=1e-15)
        assert residuals[index] == pytest.approx(SEED_RESIDUALS[n][index], rel=1e-6)
        assert residuals[index] > 1e-3, "not identically zero: the whole's zero mode is not the torus's"
        # the own-kernel leak of T2 is a diagnostic at its floor, not the scored residual
        assert node.fiber_residual_for_input_block(index) == own_before[index] < FLOOR
        read = node.read_input_state(index)
        assert read.block == index and read.obstruction == "" and read.harmonic_rank == 2 and read.frame_rank == 2
        assert read.residual == residuals[index] and read.weight == 1e6
        assert np.allclose(np.asarray(read.input), [1.0, complex(q.tau())])
        coefficients = np.asarray(read.coefficients)
        assert np.abs(coefficients - np.asarray(read.input)).max() < 0.03, coefficients
        assert np.abs(coefficients - np.asarray(read.input)).max() > 1e-4, "the whole's coefficients are not the inputs on the seed"
    # r_U scores the weighted OWN-state residuals of D2 (zero on the seed: each
    # torus is its own flat torus there), plus the two-body term once set; the
    # whole-frame leak above is the output-state read, not a term
    own = [node.own_state_residual(i) for i in range(2)]
    assert all(r < 1e-12 for r in own), own
    assert node.r_u(st) == pytest.approx(1e6 * sum(own), abs=1e-12)
    chi = spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state()))
    node.set_two_body_target(chi, True)
    assert node.r_u(st) == pytest.approx(1e6 * sum(own) + node.two_body_residual(), rel=1e-12)
    # the transfer in the derived frames equals T3's in the supplied period frames
    read = node.read_two_body()
    assert read.in_frames and read.derived_frames and np.asarray(read.transfer).shape == (2, 2)
    assert [r.residual for r in read.input_states] == residuals
    assert [tuple(int(v) for v in c) for c in read.cells_a] == [tuple(int(v) for v in c) for c in node.derive_input_frame(0).frame.cells]
    assert all(r < FLOOR for r in read.input_fiber_residuals)
    supplied = collar(n)[3]
    for index, q in enumerate((qa, qb)):
        cells = [list(c) for c in supplied.inputs[index].fiber.cells]
        F = np.asarray(q.period_frame()).astype(complex)
        supplied.set_input_frame(index, cells, F, np.asarray(supplied.input_frame_dual(index, F)))
    supplied.set_two_body_target(chi, True)
    T, T3 = np.asarray(read.transfer), np.asarray(supplied.read_two_body().transfer)
    assert np.abs(T - T3).max() < 1e-12 * np.abs(T3).max()
    assert node.two_body_residual() == pytest.approx(supplied.two_body_residual(), rel=1e-12)
    # a marked block next to a supplied-frame block is still framed (both in effect)
    mixed = collar(n)[3]
    mixed.set_input_marking(0, host_marking(qa, ids[0]), [1.0, complex(qa.tau())])
    with pytest.raises(RuntimeError, match="only when both"):
        mixed.read_two_body()
    cells = [list(c) for c in mixed.inputs[1].fiber.cells]
    F = np.asarray(qb.period_frame()).astype(complex)
    mixed.set_input_frame(1, cells, F, np.asarray(mixed.input_frame_dual(1, F)))
    mixed_read = mixed.read_two_body()
    assert mixed_read.in_frames and not mixed_read.derived_frames
    assert np.abs(np.asarray(mixed_read.transfer) - T3).max() < 1e-12 * np.abs(T3).max()
    print(f"\n[T2-bis] {n}x{n} seed: block residuals {residuals[0]:.6e}, {residuals[1]:.6e} (own-kernel {own_before[0]:.1e}, "
          f"{own_before[1]:.1e}); coefficients {[np.round(np.asarray(node.read_input_state(i).coefficients), 5).tolist() for i in range(2)]}; "
          f"T {np.round(T.real, 6).tolist()}, Schmidt {read.singular_values}, two-body {node.two_body_residual():.6f}")


# --------------------------------------------------------------------------- #
# (c) gauge invariance under a pure gauge on the whole
# --------------------------------------------------------------------------- #
def gauged_collar(n, seed_value=0):
    """The collar with a pure gauge phi_e = g(target) - g(source) on EVERY
    host edge (one gauge function over both tori's vertices), the gauged
    tori's holomorphic forms attached as the state fibers, and their
    markings with (1, tau_in)."""
    qa, qb = torus(TAU_A, n), torus(TAU_B, n)
    seed = MC.seed_collar(qa.spacetime(), qb.spacetime(), 1)
    st = seed.host
    ids = seed.vertex_ids
    g = {}
    reads = []
    for q, mapping, gauge_seed in zip((qa, qb), ids, (11, 12)):
        local = gauge_function(len(q.vertices()), gauge_seed)
        g.update({mapping[i]: local[i] for i in range(len(local))})
        reads.append(gauged_torus(q, local))
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        edge.setPhase(g[v] - g[u])
    node = MC(st, [[1.0 + 0j], [1.0 + 0j]], [], degrees=[1], seed=seed_value, einstein_hilbert=False)
    node.seed_inputs([sorted(m.values()) for m in ids])
    node.use_fiber_residuals(True)
    node.set_input_residual_weight(1e6)
    for i, ((gauged_st, gauged), q) in enumerate(zip(reads, (qa, qb))):
        f = cob.BoundaryFiber()
        f.degree = 1
        f.cells = [sorted((ids[i][int(a)], ids[i][int(b)])) for a, b in gauged.edges()]
        f.images = np.asarray(gauged.holomorphic_form(), dtype=complex).reshape(-1, 1)
        f.contour = cob.PencilLayer.harmonic_contour(cob.PencilLayer.assemble([gauged_st]), 1)
        node.attach_input_fiber(i, f, f.cells)
        node.set_input_marking(i, host_marking(q, ids[i]), [1.0 + 0j, complex(q.tau())])
    return (qa, qb), [read for _, read in reads], seed, node, g


def test_frames_coefficients_residuals_and_transfer_are_gauge_invariant(whitney_default):
    n = 3
    qa, qb, seed, plain = collar(n)
    ids = seed.vertex_ids
    mark(plain, (qa, qb), ids)
    tori, gauged_reads, gseed, gauged, g = gauged_collar(n)
    assert gseed.vertex_ids == ids
    chi = spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state()))
    plain.set_two_body_target(chi, True)
    gauged.set_two_body_target(chi, True)
    for index, (q, gq) in enumerate(zip(tori, gauged_reads)):
        # the gauged torus reads the same tau, and its own base vertex is the block's
        assert abs(complex(gq.tau()) - complex(q.tau())) < 1e-12
        assert gauged.input_marking(index).base_vertex == ids[index][int(gq.base_vertex())]
        # the derived frame under the gauge is the gauged torus's period frame (g_{v0} rho_1 F)
        derived = gauged.derive_input_frame(index)
        assert derived.derived(), derived.obstruction
        cells = [tuple(int(v) for v in c) for c in derived.frame.cells]
        reference = frame_through_ids(gq, ids[index], cells)
        F = np.asarray(derived.frame.images)
        assert np.abs(F - reference).max() < 1e-11 * np.abs(reference).max()
        # ... and it is not the plain frame (the phases enter)
        assert np.abs(F - np.asarray(plain.derive_input_frame(index).frame.images)).max() > 1e-3
        # the dual contract holds on the block's own covariant pencil
        M = own_mass_matrix(gauged, index, cells)
        assert np.abs(np.asarray(derived.frame.dual_images).T @ M @ F - np.eye(2)).max() < 1e-11
        # the residual and the coefficient pair are gauge invariant
        r_plain, r_gauged = plain.read_input_state(index), gauged.read_input_state(index)
        assert r_gauged.obstruction == "" and r_gauged.harmonic_rank == 2
        assert abs(r_gauged.residual - r_plain.residual) < 1e-12
        assert gauged.input_state_residual(index) == r_gauged.residual
        defect = np.abs(np.asarray(r_gauged.coefficients) - np.asarray(r_plain.coefficients)).max()
        assert defect < 1e-10, defect
        print(f"\n[T2-bis] gauge: block {index} frame defect {np.abs(F - reference).max():.2e}, residual defect "
              f"{abs(r_gauged.residual - r_plain.residual):.2e}, coefficient defect {defect:.2e}")
    # the transfer: invariant up to the base-point factor exp(i (g(v_A) - g(v_B))); the leak exactly
    T_plain, T_gauged = np.asarray(plain.read_two_body().transfer), np.asarray(gauged.read_two_body().transfer)
    factor = np.exp(1j * (g[plain.input_marking(0).base_vertex] - g[plain.input_marking(1).base_vertex]))
    assert np.abs(T_gauged * factor - T_plain).max() < 1e-11 * np.abs(T_plain).max()
    assert abs(gauged.two_body_residual() - plain.two_body_residual()) < 1e-12
    assert gauged.read_two_body().reversal_residual < 1e-8
    # the monodromy read with transported periods: invariant up to the same factor
    markings = [host_marking(q, ids[i]) for i, q in enumerate(tori)]
    m_plain = MC.monodromy(plain.spacetime(), markings[0], markings[1])
    m_gauged = MC.monodromy(gauged.spacetime(), markings[0], markings[1])
    assert m_plain.obstruction == "" and m_gauged.obstruction == ""
    assert m_plain.rounded == [[1, 0], [0, 1]] and m_plain.rounding_residual < 1e-9
    M_gauged = np.asarray(m_gauged.monodromy) * np.exp(1j * (g[plain.input_marking(1).base_vertex] - g[plain.input_marking(0).base_vertex]))
    assert np.abs(M_gauged - np.asarray(m_plain.monodromy)).max() < 1e-9
    # the gradient under the gauge is the plain gradient (the residual is invariant)
    g_plain = np.asarray(plain.input_state_residual_gradient(0)[0])
    g_gauged = np.asarray(gauged.input_state_residual_gradient(0)[0])
    assert np.abs(g_gauged - g_plain).max() < 1e-9 * np.abs(g_plain).max()
    print(f"[T2-bis] gauge: transfer defect {np.abs(T_gauged * factor - T_plain).max():.2e}, two-body defect "
          f"{abs(gauged.two_body_residual() - plain.two_body_residual()):.2e}, monodromy defect "
          f"{np.abs(M_gauged - np.asarray(m_plain.monodromy)).max():.2e}, gradient defect {np.abs(g_gauged - g_plain).max():.2e}")


# --------------------------------------------------------------------------- #
# (d) the gradient
# --------------------------------------------------------------------------- #
def test_gradient_euler_identity_support_and_sign(whitney_default):
    qa, qb, seed, node = collar(3)
    ids = seed.vertex_ids
    mark(node, (qa, qb), ids)
    st = node.spacetime()
    edges = st.getEdgeList().toVector()
    on_a, on_b, bulk = region_masks(st, seed)
    gradients = []
    for index in range(2):
        g, p = node.input_state_residual_gradient(index)
        g = np.asarray(g)
        assert np.asarray(p).size == 0, "degree 1 carries no phase gradient"
        assert g.shape == (len(edges),) and np.abs(g).max() > 0
        defect = euler_defect(st, g)
        assert defect < 1e-10, f"Euler identity violated: {defect:.3e}"
        # nonzero on the bulk (the whole's zero mode moves) and on the block's own
        # edges (the target moves), and on the other torus's edges too
        assert np.abs(g[bulk]).max() > 1e-3 and np.abs(g[on_a]).max() > 1e-3 and np.abs(g[on_b]).max() > 1e-3
        gradients.append(g)
        # one central-difference probe on the block's own edge of largest sensitivity and one bulk edge
        # (a sanity check of this test only; the engine never uses a finite difference)
        own = on_a if index == 0 else on_b
        for label, mask in (("own", own), ("bulk", bulk)):
            i = int(np.argmax(np.abs(g) * mask))
            fd = central_difference(node, index, edges[i])
            analytic = g[i].real
            assert fd != 0 and np.sign(fd) == np.sign(analytic)
            assert abs(fd - analytic) < 1e-5 * abs(analytic), (label, fd, analytic)
            print(f"\n[T2-bis] block {index} {label} edge {edge_keys(st)[i]}: analytic {analytic:.6e} central difference {fd:.6e}; "
                  f"Euler defect {defect:.2e}; |g| bulk {np.abs(g[bulk]).max():.3e} A {np.abs(g[on_a]).max():.3e} B {np.abs(g[on_b]).max():.3e}")
    # the ascent of r_U is the weighted sum of the two OWN-state gradients (D2),
    # plus the two-body gradient; the output read's gradients above are not in it
    own = [np.asarray(node.own_state_residual_gradient(i)[0]) for i in range(2)]
    total, _ = node.fiber_mode_ascent()
    expected = 1e6 * (own[0] + own[1])
    assert np.abs(np.asarray(total) - expected).max() <= 1e-12 * max(np.abs(expected).max(), 1.0)
    node.set_two_body_target(spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state())), True)
    total, _ = node.fiber_mode_ascent()
    expected = expected + np.asarray(node.two_body_residual_gradient()[0])
    assert np.abs(np.asarray(total) - expected).max() < 1e-12 * np.abs(expected).max()


# --------------------------------------------------------------------------- #
# (e) stage 2 descends the block residuals
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_stage2_descends_the_block_residuals(whitney_default):
    qa, qb, seed, node = collar(3, weight=1e6, einstein_hilbert=True, real_squared_lengths_only=True)
    ids = seed.vertex_ids
    mark(node, (qa, qb), ids)
    node.set_two_body_target(spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state())), True)

    def state():
        reads = [node.read_input_state(i) for i in range(2)]
        return dict(residuals=[r.residual for r in reads], coefficients=[np.asarray(r.coefficients) for r in reads],
                    two_body=node.two_body_residual(), objective=node.objective(), terms=node.objective_terms())

    before = state()
    assert [round(r, 6) for r in before["residuals"]] == [round(r, 6) for r in SEED_RESIDUALS[3]]
    trace = node.run_stage2(beta=1.0, max_iters=20, tolerance=1e-15)
    after = state()
    st = node.spacetime()
    s = squared_lengths(st)
    assert bool(np.all(s.real > 0) and np.all(np.abs(s.imag) < 1e-15)), "the real locus"
    assert len(trace) > 1 and trace[-1] < before["objective"]
    inputs = [np.array([1.0, TAU_A]), np.array([1.0, TAU_B])]
    for index in range(2):
        assert after["residuals"][index] < before["residuals"][index], (index, before["residuals"], after["residuals"])
        assert np.abs(after["coefficients"][index] - inputs[index]).max() < np.abs(before["coefficients"][index] - inputs[index]).max()
    print(f"\n[T2-bis] stage 2 (Regge on, real locus, weight 1e6, {len(trace) - 1} steps): objective "
          f"{before['objective']:.4f} -> {after['objective']:.4f}, Regge {before['terms'].regge_stationarity:.4f} -> "
          f"{after['terms'].regge_stationarity:.4f}, blocks {before['residuals'][0]:.3e}/{before['residuals'][1]:.3e} -> "
          f"{after['residuals'][0]:.3e}/{after['residuals'][1]:.3e}, two-body {before['two_body']:.6f} -> "
          f"{after['two_body']:.6f}, coefficients {[np.round(c, 5).tolist() for c in after['coefficients']]} "
          f"against {[np.round(c, 5).tolist() for c in inputs]}")


# --------------------------------------------------------------------------- #
# (f) ordinary nodes and the supplied-frame path are bit-identical to origin/main
# --------------------------------------------------------------------------- #
LAMBDA = np.array([math.sqrt(3.0), 2.0, math.sqrt(3.0), 0.0])


def flip_flop(psi, phi):
    D = np.zeros((4, 4), dtype=complex)
    for k in range(3):
        D[k + 1, k] = LAMBDA[k]
    return np.outer(D @ psi, D.T @ phi) + np.outer(D.T @ psi, D @ phi)


def degree0_fiber(psi):
    f = cob.BoundaryFiber()
    f.degree = 0
    f.cells = [[0], [1], [2], [3]]
    f.images = np.asarray(psi, dtype=complex).reshape(4, 1)
    return f


def tops(st):
    return sorted(tuple(sorted(v.getId() for v in s.getVertices())) for s in st.getTopSimplices())


def edge_geometry(st):
    out = {}
    for e in st.getEdgeList().toVector():
        u, v = e.getSource().getId(), e.getTarget().getId()
        phase = complex(e.getPhase())
        out[(min(u, v), max(u, v))] = (complex(e.getLength()), phase if u < v else 0j - phase)
    return out


def vector_hex(values):
    return [hexc(z) for z in np.asarray(values).ravel()]


def pristine_dump():
    """Every path this change must leave alone, on origin/main's API only:
    (i) an ordinary degree-0 node (two degree-0 fibers on disjoint tetrahedra
    of a grown simplex seed, jittered lengths, the flip-flop target) after two
    stage-2 steps -- cells, edges, r_U, the residuals and the ascent; (ii) T3's
    3x3 collar with SUPPLIED period frames -- the transfer, the two-body
    residual, its gradient and the ascent; (iii) T2's unmarked 3x3 collar on
    jittered lengths -- the own-kernel residuals and the ascent; (iv) the
    monodromy read of the collar seed on zero phases."""
    rng = np.random.default_rng(975)
    psi, phi = (rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(2))
    node = MC(MC.seed_simplex(3), [[1.0 + 0j, 0j, 0j, 0j], [1.0 + 0j, 0j, 0j, 0j]], [], degrees=[0],
              seed=0, precone=8, einstein_hilbert=False)
    for e in node.spacetime().getEdgeList().toVector():
        s = 1.0 + 0.15 * rng.uniform(-1, 1) + 1j * 0.15 * rng.uniform(-1, 1)
        e.setLength(np.sqrt(complex(s)))
    tets = [tuple(int(v) for v in t) for t in cob.ChainComplex.fromSpacetime(node.spacetime()).kSimplexVertices(3)]
    a, b = next((x, y) for x, y in itertools.combinations(tets, 2) if not set(x) & set(y))
    node.seed_inputs([0, 1])
    node.attach_input_fiber(0, degree0_fiber(psi), [[v] for v in a])
    node.attach_input_fiber(1, degree0_fiber(phi), [[v] for v in b])
    node.set_two_body_target(flip_flop(psi, phi))
    node.use_fiber_residuals(True)
    trace = node.run_stage2(beta=1.0, max_iters=2, tolerance=1e-15)
    st = node.spacetime()
    ordinary = {
        "cells": [list(c) for c in tops(st)],
        "edges": [[k[0], k[1], hexc(l), hexc(p)] for k, (l, p) in sorted(edge_geometry(st).items())],
        "trace": [float(x).hex() for x in trace],
        "r_u": float(node.r_u(st)).hex(),
        "block_residuals": [float(node.fiber_residual_for_input_block(i)).hex() for i in range(2)],
        "two_body_residual": float(node.two_body_residual()).hex(),
        "ascent": vector_hex(node.fiber_mode_ascent()[0]),
        "transfer": vector_hex(node.read_two_body().transfer),
    }
    qa, qb, seed, framed = collar(3, weight=1e6)
    for index, q in enumerate((qa, qb)):
        cells = [list(c) for c in framed.inputs[index].fiber.cells]
        F = np.asarray(q.period_frame()).astype(complex)
        framed.set_input_frame(index, cells, F, np.asarray(framed.input_frame_dual(index, F)))
    framed.set_two_body_target(spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state())), True)
    read = framed.read_two_body()
    supplied = {
        "in_frames": bool(read.in_frames),
        "transfer": vector_hex(read.transfer),
        "two_body_residual": float(framed.two_body_residual()).hex(),
        "two_body_gradient": vector_hex(framed.two_body_residual_gradient()[0]),
        "ascent": vector_hex(framed.fiber_mode_ascent()[0]),
        "block_residuals": [float(r).hex() for r in read.input_fiber_residuals],
    }
    qa, qb, seed, unmarked = collar(3, weight=1e6)
    st = unmarked.spacetime()
    for e in st.getEdgeList().toVector():
        e.setLength(complex(e.getLength()) * (1.0 + 0.05 * rng.uniform(-1, 1)))
    own = {
        "block_residuals": [float(unmarked.fiber_residual_for_input_block(i)).hex() for i in range(2)],
        "r_u": float(unmarked.r_u(st)).hex(),
        "ascent": vector_hex(unmarked.fiber_mode_ascent()[0]),
    }
    qa, qb, seed, node = collar(3)
    markings = [host_marking(q, seed.vertex_ids[i]) for i, q in enumerate((qa, qb))]
    m = MC.monodromy(node.spacetime(), markings[0], markings[1])
    monodromy = {
        "periods_a": vector_hex(m.periods_a), "periods_b": vector_hex(m.periods_b),
        "monodromy": vector_hex(m.monodromy), "rounded": [[int(x) for x in row] for row in m.rounded],
        "harmonic_rank": int(m.harmonic_rank), "obstruction": m.obstruction,
    }
    return {"ordinary": ordinary, "supplied": supplied, "unmarked": own, "monodromy": monodromy}


# The saved dump was generated from origin/main's build (d42c24d) in a second
# venv by  python tests/cobordism/test_block_residual_whole_frame_python.py <path>
# with OMP_NUM_THREADS=8. Cells, edges, ranks and the monodromy's integer
# rounding must agree exactly; the floating-point records are compared at a
# few units in the last place, the level at which the OpenMP thread count
# already moves them on origin/main.
def test_ordinary_nodes_and_supplied_frames_are_bit_identical_to_the_saved_dump(whitney_default):
    expected = json.loads(DUMP.read_text())
    got = pristine_dump()
    o, e = got["ordinary"], expected["ordinary"]
    assert o["cells"] == e["cells"]
    assert o["edges"] == e["edges"], "an ordinary node's edge differs bit for bit"
    assert len(o["trace"]) == len(e["trace"])
    for key in ("r_u", "two_body_residual"):
        assert within_ulps(float.fromhex(o[key]), float.fromhex(e[key]), 8), key
    for mine, saved in zip(o["block_residuals"], e["block_residuals"]):
        assert within_ulps(float.fromhex(mine), float.fromhex(saved), 8)
    for key in ("ascent", "transfer"):
        assert close_ulps([unhex(z) for z in o[key]], [unhex(z) for z in e[key]], 64), key
    s, t = got["supplied"], expected["supplied"]
    assert s["in_frames"] and t["in_frames"]
    assert s["transfer"] == t["transfer"], "the supplied-frame transfer differs bit for bit"
    assert within_ulps(float.fromhex(s["two_body_residual"]), float.fromhex(t["two_body_residual"]), 8)
    for key in ("two_body_gradient", "ascent"):
        assert close_ulps([unhex(z) for z in s[key]], [unhex(z) for z in t[key]], 64), key
    for mine, saved in zip(s["block_residuals"], t["block_residuals"]):
        assert within_ulps(float.fromhex(mine), float.fromhex(saved), 8)
    u, v = got["unmarked"], expected["unmarked"]
    for mine, saved in zip(u["block_residuals"], v["block_residuals"]):
        assert within_ulps(float.fromhex(mine), float.fromhex(saved), 8)
    assert within_ulps(float.fromhex(u["r_u"]), float.fromhex(v["r_u"]), 8)
    assert close_ulps([unhex(z) for z in u["ascent"]], [unhex(z) for z in v["ascent"]], 64)
    m, w = got["monodromy"], expected["monodromy"]
    assert m["rounded"] == w["rounded"] == [[1, 0], [0, 1]] and m["harmonic_rank"] == w["harmonic_rank"] == 2
    assert m["obstruction"] == w["obstruction"] == ""
    for key in ("periods_a", "periods_b", "monodromy"):
        assert m[key] == w[key], f"the monodromy read's {key} differs bit for bit on zero phases"
    print(f"\n[T2-bis] bit-identity: ordinary node {len(o['cells'])} cells, {len(o['edges'])} edges; supplied-frame "
          f"transfer {[unhex(z) for z in s['transfer']]}; monodromy periods bit-identical")


if __name__ == "__main__":
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(pristine_dump(), sort_keys=True))
    print("wrote", sys.argv[1])
