# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The block residual of the qubit cobordism spec D2 (#989): a marked input
torus keeps representing its input state through the zero mode of its OWN
Laplacian, and the zero mode of the ENTIRE cobordism is the OUTPUT state.

``own_state_residual(index)`` is ``1 - |<psi(tau_in)|psi(tau_hat)>|^2`` with
``tau_hat`` the ratio of the transported periods of the holomorphic form of
the block's own Laplacian on its live surface (``block_qubit``: the
``SimplicialQubit`` of the block's own triangles with the host's live lengths
and phases, over the block's marking, in the orientation with A.B = +1) and
``psi(tau) = (1, tau)/sqrt(1 + |tau|^2)``. It is what ``r_U`` scores for a
marked block at ``set_input_residual_weight``; ``input_state_residual`` and
``read_input_state`` -- the whole's zero mode in the block's live frame --
are the output-state read and are no longer held (spec R1, R3, D2 of
2026-09-07). Its gradient is analytic through
``SimplicialQubit.tau_derivative`` (no finite difference anywhere in the
engine), supported on the block's own edges, empty in the phases.

The fixture is T2's: collar nodes from two ``SimplicialQubit.flat_torus``
inputs (3x3; tau_A = 0.3 + 1.1i, tau_B = -0.2 + 0.8i) on the Whitney pencil.
Observed (OMP_NUM_THREADS=8):

* on the collar seed each block reads its own torus (tau_hat = tau_in to
  1e-15, A.B = +1) and the residual is zero to rounding (2e-16 and 0), while
  the output-state read's leak is T4's 3.1e-3 and 9.3e-3; r_U is the
  weighted sum of the two (zero) residuals plus the two-body term;
* with the tori's edge lengths jittered by 5% the residuals are 4.9e-4 and
  1.9e-4, the gradient is supported on exactly the 27 edges of each torus
  (zero on the 36 bulk edges), satisfies the Euler identity
  sum_e Re(z_e) g_e = 0 to 1e-19 (tau is scale-free) and agrees with central
  differences to 3e-8 (the difference's own accuracy);
* under a pure gauge on every host edge the residual, tau_hat and the
  gradient are unchanged to 1e-12;
* three stage-2 steps from the jittered collar (no Regge term, no two-body
  target) lower both residuals.
"""
import sys
import warnings

import numpy as np
import pytest

from tessera import cobordism as cob
from tessera import observables as obs

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import test_block_residual_whole_frame_python as whole  # noqa: E402  (T2's fixture)

MC = cob.MultiCobordism
HL = cob.HodgeLaplacian
TAU_A, TAU_B = whole.TAU_A, whole.TAU_B


@pytest.fixture
def whitney_default():
    previous = HL.defaultMetricSource()
    HL.setDefaultMetricSource(cob.HodgeMetricSource.WhitneyPencil)
    try:
        yield
    finally:
        HL.setDefaultMetricSource(previous)


def marked_collar(weight=1e4, **kwargs):
    qa, qb, seed, node = whole.collar(3, weight=weight, **kwargs)
    whole.mark(node, (qa, qb), seed.vertex_ids)
    return qa, qb, seed, node


def region_masks(node, seed):
    """Boolean masks over the host's EdgeList: on torus A, on torus B, bulk."""
    edges = node.spacetime().getEdgeList().toVector()
    tori = [set(seed.vertex_ids[i].values()) for i in range(2)]
    ends = [{e.getSource().getId(), e.getTarget().getId()} for e in edges]
    on_a = np.array([vs <= tori[0] for vs in ends])
    on_b = np.array([vs <= tori[1] for vs in ends])
    return on_a, on_b, ~(on_a | on_b)


def jitter_tori(node, seed, amplitude=0.05, seed_value=3):
    rng = np.random.default_rng(seed_value)
    on_a, on_b, _ = region_masks(node, seed)
    for k, edge in enumerate(node.spacetime().getEdgeList().toVector()):
        if on_a[k] or on_b[k]:
            edge.setLength(complex(edge.getLength()) * (1.0 + amplitude * rng.uniform(-1, 1)))


def squared_lengths(node):
    return np.array([complex(e.getLength()) ** 2 for e in node.spacetime().getEdgeList().toVector()])


def central_difference(node, index, edge, h=1e-6):
    """(d/dRe z, d/dIm z) of own_state_residual(index) in one host edge's
    squared length, packed as a complex number (the engine's convention); an
    approximation of order h^2, about 1e-8 here, for this test only."""
    l0 = complex(edge.getLength())
    z0 = l0 * l0
    out = []
    for dz in (h, 1j * h):
        values = []
        for sign in (+1, -1):
            root = np.sqrt(z0 + sign * dz)
            root = root if abs(root - l0) <= abs(-root - l0) else -root
            edge.setLength(complex(root))
            values.append(node.own_state_residual(index))
        out.append((values[0] - values[1]) / (2 * h))
    edge.setLength(l0)
    return complex(out[0], out[1])


def test_the_seed_represents_its_inputs_on_its_own(whitney_default):
    qa, qb, seed, node = marked_collar(weight=1e4)
    st = node.spacetime()
    own = []
    for index, q in enumerate((qa, qb)):
        read = node.block_qubit(index)
        assert abs(read.intersection_number() - 1.0) < 1e-12, "the marking fixes the orientation"
        assert abs(read.tau() - q.tau()) < 1e-14, (read.tau(), q.tau())
        assert (len(read.vertices()), len(read.edges()), len(read.faces())) == (9, 27, 18)
        residual = node.own_state_residual(index)
        assert 0.0 <= residual < 1e-14, residual
        own.append(residual)
        # the output-state read is a different number: the whole's zero mode is not the torus's
        assert node.input_state_residual(index) > 1e-3
        assert node.read_input_state(index).residual == node.input_state_residual(index)
    # r_U scores the own-state residuals at their weight, plus the two-body term once set
    assert node.r_u(st) == pytest.approx(1e4 * sum(own), abs=1e-12)
    node.set_two_body_target(whole.spin_half_chi(np.asarray(qa.state()), np.asarray(qb.state())), True)
    assert node.r_u(st) == pytest.approx(1e4 * sum(own) + node.two_body_residual(), rel=1e-12)
    # refusals by name
    with pytest.raises(IndexError):
        node.own_state_residual(2)
    plain = whole.collar(3)[3]
    with pytest.raises(RuntimeError, match="marking"):
        plain.own_state_residual(0)
    with pytest.raises(RuntimeError, match="marking"):
        plain.block_qubit(0)


def test_the_gradient_is_analytic_supported_on_the_torus_and_scale_free(whitney_default):
    qa, qb, seed, node = marked_collar(weight=1e4)
    jitter_tori(node, seed)
    on_a, on_b, bulk = region_masks(node, seed)
    edges = node.spacetime().getEdgeList().toVector()
    z = squared_lengths(node)
    rng = np.random.default_rng(0)
    for index, own in enumerate((on_a, on_b)):
        residual = node.own_state_residual(index)
        assert 1e-6 < residual < 1e-2, residual
        g, p = node.own_state_residual_gradient(index)
        g = np.asarray(g)
        assert np.asarray(p).size == 0, "tau is gauge invariant: no phase gradient"
        assert g.shape == (len(edges),)
        assert np.count_nonzero(g) == 27 and np.all(g[~own] == 0) and np.all(g[own] != 0)
        assert np.all(g[bulk] == 0), "the block's own Laplacian does not see the bulk"
        # the Euler identity: the residual is invariant under a common scale of the lengths
        euler = np.sum(z.real * g.real + z.imag * g.imag)
        assert abs(euler) < 1e-15 * np.abs(g).max() * np.abs(z).max(), euler
        for k in rng.choice(np.flatnonzero(own), 3, replace=False):
            fd = central_difference(node, index, edges[k])
            assert abs(fd - g[k]) <= 1e-6 * abs(g[k]), (k, fd, g[k])
    # the ascent of r_U is the weighted sum of the two gradients
    total, _ = node.fiber_mode_ascent()
    expected = 1e4 * sum(np.asarray(node.own_state_residual_gradient(i)[0]) for i in range(2))
    assert np.abs(np.asarray(total) - expected).max() <= 1e-12 * np.abs(expected).max()


def test_the_residual_and_its_gradient_are_gauge_invariant(whitney_default):
    plain = marked_collar(weight=1e4)
    gauged = marked_collar(weight=1e4)
    for _, _, seed, node in (plain, gauged):
        jitter_tori(node, seed)
    rng = np.random.default_rng(11)
    st = gauged[3].spacetime()
    gauge = {int(v.getId()): rng.uniform(-np.pi, np.pi) for v in st.getVertexList().toVector()}
    for edge in st.getEdgeList().toVector():
        edge.setPhase(gauge[int(edge.getTarget().getId())] - gauge[int(edge.getSource().getId())])
    for index in range(2):
        a, b = plain[3].block_qubit(index), gauged[3].block_qubit(index)
        assert a.trivial_connection() and not b.trivial_connection()
        assert abs(a.tau() - b.tau()) < 1e-12
        assert abs(plain[3].own_state_residual(index) - gauged[3].own_state_residual(index)) < 1e-12
        ga = np.asarray(plain[3].own_state_residual_gradient(index)[0])
        gb = np.asarray(gauged[3].own_state_residual_gradient(index)[0])
        assert np.abs(ga - gb).max() < 1e-11 * np.abs(ga).max()


def test_stage2_descends_the_own_state_residuals(whitney_default):
    qa, qb, seed, node = marked_collar(weight=1e4, einstein_hilbert=False)
    jitter_tori(node, seed)
    before = [node.own_state_residual(i) for i in range(2)]
    objective_before = node.objective()
    trace = node.run_stage2(beta=1.0, max_iters=3, tolerance=1e-15)
    after = [node.own_state_residual(i) for i in range(2)]
    assert len(trace) > 1 and trace[-1] < objective_before
    for index in range(2):
        assert after[index] < before[index], (index, before, after)
        read = node.block_qubit(index)
        assert abs(read.intersection_number() - 1.0) < 1e-12
    print(f"\n[D2] stage 2 ({len(trace) - 1} steps, weight 1e4, no Regge term): objective "
          f"{objective_before:.6e} -> {trace[-1]:.6e}, residuals {before[0]:.3e}/{before[1]:.3e} -> "
          f"{after[0]:.3e}/{after[1]:.3e}, tau-hat {[node.block_qubit(i).tau() for i in range(2)]}")
