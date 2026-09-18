# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The simplicial qubit over complex geometry (#976), section 16 of
docs/design/simplicial_qubit_spec.md: (a) the real locus is bit-identical to
the real-length construction; (b) a torus continued from a real reference by
a small complex perturbation has tau and its eigenline continuous in the
perturbation; (c) a pure-gauge phase assignment leaves tau, the state and the
period frame's coefficient pairs invariant, on real and complex-length tori;
(d) flux and flat holonomy are refused by name; (e) the branch rules (Heron on
WhitneyMass.volumeOnBranch, the principal acos, the transpose pairing); (f)
the Connection.transportedPeriod primitive."""
import cmath
import json
import math
import re
import pathlib
import warnings

import numpy as np

from tests._golden import RELATIVE
import pytest

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import observables as obs

SimplicialQubit = obs.SimplicialQubit
DUMP = pathlib.Path(__file__).parent / "data" / "simplicial_qubit_real_locus_dump.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def quiet(build):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def flat(tau, nx=4, ny=4):
    return quiet(lambda: SimplicialQubit.flat_torus(tau, nx, ny))


def rebuilt(q, lengths=None, cycle_A=None, cycle_B=None):
    return quiet(lambda: SimplicialQubit(
        list(q.vertices()), list(q.edges()), list(q.faces()),
        list(q.lengths()) if lengths is None else list(lengths),
        list(q.cycle_A()) if cycle_A is None else cycle_A,
        list(q.cycle_B()) if cycle_B is None else cycle_B))


def conformal_lengths(q, n, amplitude=0.3):
    """The lengths of q (an n x n grid) scaled by exp(phi) at each edge midpoint,
    phi = amplitude sin(2 pi x) cos(2 pi y): conformally flat, not flat."""
    def reduce(delta):
        return -1 if delta == n - 1 else (1 if delta == -(n - 1) else delta)

    out = []
    for (u, v), length in zip(q.edges(), q.lengths()):
        iu, ju = divmod(u, n)
        iv, jv = divmod(v, n)
        di, dj = reduce(iv - iu), reduce(jv - ju)
        x, y = (iu + 0.5 * di) / n, (ju + 0.5 * dj) / n
        out.append(length * math.exp(amplitude * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)))
    return out


def perturbed_lengths(q, epsilon, seed=3):
    """Complex lengths continued from q's real ones: l_e (1 + i epsilon delta_e),
    delta_e in [-1, 1] fixed by the seed."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-1.0, 1.0, size=len(q.edges()))
    return [complex(l) * (1 + 1j * epsilon * d) for l, d in zip(q.lengths(), delta)]


def read_spacetime(q, lengths=None, phases=None, reversed_flag=False):
    """The same torus read through the Spacetime constructor after writing
    lengths (edge order) and phases (a map edge index -> phase, the phase on the
    edge's stored source -> target orientation) onto its Spacetime's edges."""
    st = q.spacetime()
    index = {tuple(e): k for k, e in enumerate(q.edges())}
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        e = index[(min(u, v), max(u, v))]
        if lengths is not None:
            edge.setLength(complex(lengths[e]))
        edge.setPhase(complex(0.0) if phases is None else complex(phases[e]))
    return quiet(lambda: SimplicialQubit(st, list(q.cycle_A()), list(q.cycle_B()), reversed_flag))


def pure_gauge_phases(q, g):
    """phi_e = g(target) - g(source) on each edge's stored orientation: the
    connection U = 1^g, U_xy = g_x^{-1} g_y with g_x = exp(i g(x))."""
    st = q.spacetime()
    index = {tuple(e): k for k, e in enumerate(q.edges())}
    phases = [0j] * len(q.edges())
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        phases[index[(min(u, v), max(u, v))]] = g[v] - g[u]
    return phases


def gauge_function(q, seed=11, complex_scale=0.0):
    rng = np.random.default_rng(seed)
    g = rng.uniform(-math.pi, math.pi, size=len(q.vertices()))
    if complex_scale:
        g = g + 1j * complex_scale * rng.uniform(-1.0, 1.0, size=len(q.vertices()))
    return [complex(x) for x in g]


def line_overlap(a, b):
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


def proportional(a, b, tol=1e-10):
    """|a> and |b> span the same line: a x b == 0 relative to the scales."""
    a, b = np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    outer = np.outer(a, b) - np.outer(b, a)
    return np.abs(outer).max() <= tol * np.linalg.norm(a) * np.linalg.norm(b)


def hexc(z):
    z = complex(z)
    return [float(z.real).hex(), float(z.imag).hex()]


# --------------------------------------------------------------------------- #
# (a) the real locus: bit-identical to the real-length construction
# --------------------------------------------------------------------------- #
def _dump_cases():
    return {
        "square_4x4": lambda: flat(1j, 4, 4),
        "shear_4x5": lambda: flat(0.3 + 1j, 4, 5),
        "hexagonal_4x4": lambda: flat(cmath.exp(1j * math.pi / 3), 4, 4),
        "thin_4x4": lambda: flat(0.05j, 4, 4),
        "conformal_6x6": lambda: rebuilt(flat(1j, 6, 6), lengths=conformal_lengths(flat(1j, 6, 6), 6)),
    }


def unhex(pair):
    return complex(float.fromhex(pair[0]), float.fromhex(pair[1]))


def agrees(got, expected, rtol=RELATIVE):
    """Every entry of ``got`` within ``rtol`` of the quantity's scale."""
    got = np.asarray(got, dtype=complex).ravel()
    expected = np.asarray(expected, dtype=complex).ravel()
    assert got.shape == expected.shape
    scale = max(np.abs(expected).max(), 1.0)
    return np.abs(got - expected).max() <= rtol * scale


def _warning_shape(text):
    """A warning with its embedded measurements replaced by a placeholder.

    The near-degeneracy warning quotes cond(M1), which on these tori is the
    condition number of a numerically singular matrix -- about 1/eps. That
    value is not reproducible between builds (measured 9.3e15 against 1.1e16),
    so compare the warning the construction raises, not the number it quotes.
    """
    return re.sub(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", "N", text)


# The saved dump (tests/observables/data/simplicial_qubit_real_locus_dump.json)
# holds the real-length construction's outputs as hex floats, generated from
# origin/main's build. The hex encoding is kept because it round-trips exactly
# and reads well in a diff, but the comparison is numeric: the compiler's
# floating-point contraction (-O3 -march=native) of the section-8 accumulation
# changes with the code around it and with the toolchain, so the dump is not
# reproducible bit for bit. The bar is a relative tolerance on each quantity's
# scale; any change to the real path's arithmetic (an operation, a branch, an
# order) moves these values far further than that.
def _span_projector(basis):
    """Orthogonal projector onto the column span of a basis.

    Gauge-invariant: any other basis of the same subspace gives the same
    projector, so this compares the subspace the construction found without
    depending on which representative the eigensolver returned.
    """
    b = np.atleast_2d(np.asarray(basis, dtype=complex))
    q, _ = np.linalg.qr(b)
    return q @ q.conj().T


def _well_conditioned(q):
    """True when M1 is not numerically singular.

    When cond(M1) reaches 1/eps the matrix is singular to working precision and
    its null space has no determined basis: the eigensolver may return any
    representative, so the basis-dependent reads (harmonic basis, Gram,
    complex structure, periods, holomorphic form, rotation pairing) are not
    reproducible between builds, while the gauge-invariant reads (tau, state,
    Bloch vector, period frame) are. Measured on the saved dump, the
    well-conditioned cases reproduce exactly and the singular ones differ by
    order 1 in exactly those basis-dependent quantities.
    """
    return q.condition_m1() < 1e8


@pytest.mark.parametrize("name", sorted(_dump_cases()))
def test_real_locus_reproduces_the_saved_dump_to_rounding(name):
    expected = json.loads(DUMP.read_text())[name]
    q = _dump_cases()[name]()
    assert q.on_real_locus() and q.trivial_connection()
    assert q.marking_swapped() == expected["marking_swapped"]
    assert ([_warning_shape(w) for w in q.warnings()]
            == [_warning_shape(w) for w in expected["warnings"]])
    assert agrees([q.tau()], [unhex(expected["tau"])])
    # cond(M1) is the condition number of a numerically singular matrix, so its
    # value is noise; what the dump pins is that the construction still reports
    # the matrix as degenerate. cond(G) is well conditioned and is compared.
    saved_m1 = float.fromhex(expected["condition_m1"])
    assert (q.condition_m1() > 1e8) == (saved_m1 > 1e8)
    assert agrees([q.condition_g()], [float.fromhex(expected["condition_g"])])
    # ||J J + I|| is itself rounding noise on flat tori (1e-16): compare at
    # the scale of J's entries, not its own.
    assert abs(q.j_residual() - float.fromhex(expected["j_residual"])) <= RELATIVE
    # Determined regardless of which null-space basis came back.
    for key, value in (("weights", q.weights()), ("areas", q.areas()),
                       ("state", q.state()), ("bloch", q.bloch())):
        assert agrees(value, [unhex(x) for x in expected[key]]), key
    assert agrees(q.period_frame(),
                  [[unhex(x) for x in row] for row in expected["period_frame"]]), "period_frame"

    # The subspace is determined even when the basis of it is not.
    assert agrees(_span_projector(q.harmonic_basis()),
                  _span_projector([[unhex(x) for x in row]
                                   for row in expected["harmonic_basis"]])), "harmonic span"

    # These read off a chosen basis, so they only mean something against the
    # dump when that basis is determined.
    if _well_conditioned(q):
        assert agrees(list(q.periods()), [unhex(p) for p in expected["periods"]]), "periods"
        for key, value in (("holomorphic_form", q.holomorphic_form()),):
            assert agrees(value, [unhex(x) for x in expected[key]]), key
        for key, value in (("harmonic_basis", q.harmonic_basis()), ("gram", q.gram()),
                           ("rotation_pairing", q.rotation_pairing()),
                           ("complex_structure", q.complex_structure())):
            saved = [[unhex(x) for x in row] for row in expected[key]]
            assert agrees(value, saved), key
    # Real dtypes on the real locus, and the dual kernel is the kernel itself.
    assert q.harmonic_basis().dtype.kind == "f" and q.period_frame().dtype.kind == "f"
    assert all(isinstance(l, float) for l in q.lengths())
    assert np.array_equal(q.dual_harmonic_basis(), q.harmonic_basis())
    assert all(u == 1 for u in q.links())


# --------------------------------------------------------------------------- #
# (b) continuity from the real reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("deformed", [False, True], ids=["flat", "conformal"])
def test_tau_and_the_eigenline_are_continuous_in_a_complex_perturbation(deformed):
    tau0 = 0.3 + 1.2j
    base = flat(tau0, 4, 4)
    if deformed:
        base = rebuilt(base, lengths=conformal_lengths(base, 4, 0.25))
    omega0 = np.asarray(base.holomorphic_form())
    slopes, overlaps = [], []
    for epsilon in (1e-2, 1e-3, 1e-4):
        q = rebuilt(base, lengths=perturbed_lengths(base, epsilon))
        assert not q.on_real_locus() and q.trivial_connection()
        assert q.harmonic_basis().dtype.kind == "c" and any(isinstance(l, complex) for l in q.lengths())
        slopes.append(abs(q.tau() - base.tau()) / epsilon)
        overlaps.append(1.0 - line_overlap(omega0, q.holomorphic_form()))
        assert abs(np.linalg.norm(q.bloch()) - 1.0) < 1e-12
        assert abs(q.j_residual() - base.j_residual()) < 10 * epsilon
    # |tau(eps) - tau(0)| = O(eps): the ratio is bounded and settles.
    assert max(slopes) < 5.0 and abs(slopes[1] - slopes[2]) < 0.05 * max(slopes[2], 1e-3)
    # The chosen eigenline is continuous: 1 - overlap = O(eps^2) for a line moving linearly.
    assert overlaps[0] < 1e-2 and overlaps[1] < 1e-4 and overlaps[2] < 1e-6
    # The two code paths agree: an imaginary part of one part in 1e12 leaves
    # the section-16 pipeline (continuation, complex null space, transpose
    # pairing, tracked eigenline) within rounding of the real-locus result.
    tiny = rebuilt(base, lengths=perturbed_lengths(base, 1e-12))
    assert not tiny.on_real_locus()
    assert abs(tiny.tau() - base.tau()) < 1e-9
    assert line_overlap(omega0, tiny.holomorphic_form()) > 1 - 1e-9
    assert np.abs(np.asarray(tiny.period_frame()) - np.asarray(base.period_frame())).max() < 1e-8


def test_the_real_reference_is_the_unit_equilateral_torus():
    """The reference of the continuation is the torus with every squared length
    one (WhitneyMass's reference simplex); it is itself a valid, Delaunay,
    flat torus, and a torus read AT the reference agrees with the real path."""
    q = flat(1j, 4, 4)
    ones = rebuilt(q, lengths=[1.0] * len(q.edges()))
    assert ones.on_real_locus() and ones.non_delaunay_edges() == [] and ones.j_residual() < 1e-10
    near = rebuilt(q, lengths=[1.0 + 1e-13j] * len(q.edges()))
    assert not near.on_real_locus()
    assert abs(near.tau() - ones.tau()) < 1e-9


# --------------------------------------------------------------------------- #
# (c) pure-gauge link phases leave tau, the state and the frame's pairs invariant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", ["flat_real", "conformal_real", "conformal_complex", "flat_complex_gauge"])
def test_pure_gauge_phases_leave_tau_the_state_and_the_coefficient_pairs_invariant(case):
    n = 4
    base = flat(0.3 + 1.2j, n, n)
    lengths = list(base.lengths())
    if case.startswith("conformal"):
        lengths = conformal_lengths(base, n, 0.3)
    if case == "conformal_complex":
        lengths = [l * (1 + 0.05j * math.sin(3.0 * k)) for k, l in enumerate(lengths)]
    plain = read_spacetime(base, lengths=lengths)
    g = gauge_function(base, complex_scale=0.2 if case == "flat_complex_gauge" else 0.0)
    gauged = read_spacetime(base, lengths=lengths, phases=pure_gauge_phases(base, g))
    assert plain.trivial_connection() and not gauged.trivial_connection() and not gauged.on_real_locus()
    assert plain.on_real_locus() == (case != "conformal_complex")
    assert gauged.harmonic_basis().dtype.kind == "c"
    if case == "flat_complex_gauge":
        assert not gauged.connection().isUnitary()

    # tau, the state, the Bloch vector, the density matrix: invariant to rounding.
    assert abs(gauged.tau() - plain.tau()) < 1e-11
    assert np.abs(np.asarray(gauged.state()) - np.asarray(plain.state())).max() < 1e-11
    assert np.abs(np.asarray(gauged.bloch()) - np.asarray(plain.bloch())).max() < 1e-11
    assert np.abs(np.asarray(gauged.density_matrix()) - np.asarray(plain.density_matrix())).max() < 1e-11
    # The complex structure has the same spectrum (J is the same operator in
    # another basis) and the same residual.
    assert abs(gauged.j_residual() - plain.j_residual()) < 1e-10
    assert np.allclose(sorted(np.linalg.eigvals(np.asarray(gauged.complex_structure())), key=lambda z: z.imag),
                       sorted(np.linalg.eigvals(np.asarray(plain.complex_structure())), key=lambda z: z.imag),
                       atol=1e-10)

    # Both walks start at the common base point.
    v0 = gauged.base_vertex()
    assert gauged.walk_A()[0][0] == v0 and gauged.walk_B()[0][0] == v0
    assert gauged.walk_A()[-1][1] == v0 and gauged.walk_B()[-1][1] == v0
    gv = np.exp(1j * np.asarray(g))
    # The twisted kernel: the holomorphic form is rho_1 omega, rho_1 = diag(g_{b(e)}^{-1}),
    # up to the eigenvector's scale; the frame is exactly g_{v0} rho_1 F.
    rho = np.array([1.0 / gv[e[0]] for e in base.edges()])
    assert proportional(np.asarray(gauged.holomorphic_form()) / rho, np.asarray(plain.holomorphic_form()))
    F_plain = np.asarray(plain.period_frame(), dtype=complex)
    F_gauged = np.asarray(gauged.period_frame(), dtype=complex)
    assert np.abs(F_gauged - gv[v0] * rho[:, None] * F_plain).max() < 1e-10 * np.abs(F_plain).max()
    # The coefficient pair of the holomorphic form in the frame: (P_A, P_B) is
    # g_{v0}^{-1} times the untwisted pair; its ratio and the normalized pair
    # (1, tau) are invariant.
    P_A, P_B = gauged.periods()
    Q_A, Q_B = plain.periods()
    assert abs(P_B / P_A - Q_B / Q_A) < 1e-11
    coefficients = np.linalg.lstsq(F_gauged, np.asarray(gauged.holomorphic_form()), rcond=None)[0]
    assert np.abs(coefficients - np.array([P_A, P_B])).max() < 1e-10 * abs(P_A)
    assert abs(coefficients[1] / coefficients[0] - plain.tau()) < 1e-11
    # The links read back as the pure gauge U_xy = g_x^{-1} g_y.
    for (x, y), u in zip(base.edges(), gauged.links()):
        assert abs(u - gv[y] / gv[x]) < 1e-12 * abs(u)


def test_pure_gauge_with_a_marking_whose_steps_do_not_chain_in_the_given_order():
    """The modular transformation A' = B, B' = -A with -A listed in the original
    order (its steps chain only after reordering): the walk is built by
    Hierholzer's algorithm and tau -> -1/tau holds under a pure gauge."""
    tau = 0.35 + 1.1j
    base = flat(tau, 4, 4)
    A, B = list(base.cycle_A()), list(base.cycle_B())
    minus_A = [(e, -s) for (e, s) in A]
    st = base.spacetime()
    phases = pure_gauge_phases(base, gauge_function(base, seed=5))
    index = {tuple(e): k for k, e in enumerate(base.edges())}
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        edge.setPhase(complex(phases[index[(min(u, v), max(u, v))]]))
    swapped = quiet(lambda: SimplicialQubit(st, B, minus_A))
    assert not swapped.trivial_connection()
    assert abs(swapped.tau() - (-1.0 / tau)) < 1e-9
    assert len(swapped.walk_B()) == len(A) and swapped.walk_B()[0][0] == swapped.base_vertex()
    shifted = quiet(lambda: SimplicialQubit(st, A, A + B))
    assert abs(shifted.tau() - (tau + 1.0)) < 1e-9


# --------------------------------------------------------------------------- #
# (d) flux and flat holonomy are refused by name
# --------------------------------------------------------------------------- #
def test_flux_and_flat_holonomy_are_refused_by_name():
    n = 4
    base = flat(0.3 + 1.2j, n, n)
    # Flux: one phased edge puts curvature on its two faces.
    phases = [0j] * len(base.edges())
    phases[0] = 0.3
    with pytest.raises(ValueError, match="not a pure gauge: face .* carries flux"):
        read_spacetime(base, phases=phases)
    # Flat holonomy: phi_e = theta di / n with the wrapped displacement di:
    # every face closes to zero (flat) and the loop along 1 carries theta.
    theta = 0.7

    def reduce(delta):
        return -1 if delta == n - 1 else (1 if delta == -(n - 1) else delta)

    st = base.spacetime()
    index = {tuple(e): k for k, e in enumerate(base.edges())}
    flat_phases = [0j] * len(base.edges())
    for edge in st.getEdgeList().toVector():
        u, v = edge.getSource().getId(), edge.getTarget().getId()
        di = reduce(divmod(v, n)[0] - divmod(u, n)[0])
        flat_phases[index[(min(u, v), max(u, v))]] = theta * di / n
    with pytest.raises(ValueError, match="not a pure gauge: the connection is flat but has holonomy"):
        read_spacetime(base, phases=flat_phases)
    # The same connection seen by the chainhodge primitives: no curvature, a
    # Wilson loop of exp(i theta) along A.
    K = cob.ChainComplex.fromTopCells([list(f) for f in base.faces()])
    U = ch.Connection.fromSpacetime(st, K)
    for f in base.faces():
        p, q_, r = sorted(f)
        assert abs(U.curvature(p, q_, r) - 1) < 1e-12
    assert abs(U.holonomy(list(base.walk_A())) - cmath.exp(1j * theta)) < 1e-12
    assert abs(U.holonomy(list(base.walk_B())) - 1) < 1e-12
    read_spacetime(base)  # phases reset to zero: accepted again


def test_intrinsic_delaunay_refuses_off_the_real_locus():
    base = flat(cmath.exp(1j * math.pi / 3), 4, 4)
    q = rebuilt(base, lengths=perturbed_lengths(base, 1e-3))
    with pytest.raises(ValueError, match="real locus"):
        q.intrinsic_delaunay()
    assert q.non_delaunay_edges() == [] and q.negative_weight_edges() == []


# --------------------------------------------------------------------------- #
# (e) the branch rules and the transpose pairing
# --------------------------------------------------------------------------- #
def test_branch_rules_and_the_transpose_pairing():
    base = flat(0.3 + 1.2j, 4, 4)
    lengths = perturbed_lengths(base, 0.2, seed=8)
    q = rebuilt(base, lengths=lengths)
    index = {tuple(e): k for k, e in enumerate(q.edges())}
    areas = np.asarray(q.areas())
    angles = np.asarray(q.angles())
    layout = np.asarray(q.layout())
    gradients = np.asarray(q.barycentric_gradients())
    weights = np.asarray(q.weights())
    cot = {}
    for t, (i, j, k) in enumerate(q.faces()):
        a = complex(lengths[index[(min(j, k), max(j, k))]])
        b = complex(lengths[index[(min(k, i), max(k, i))]])
        c = complex(lengths[index[(min(i, j), max(i, j))]])
        sa, sb, sc = a * a, b * b, c * c
        # Heron on the continuation branch = WhitneyMass.volumeOnBranch of the
        # face's Gram matrix, bit for bit.
        gram = np.array([[sc, 0.5 * (sc + sb - sa)], [0.5 * (sc + sb - sa), sb]])
        volume, ambiguous = ch.WhitneyMass.volumeOnBranch(gram, ch.Branch.Continuation)
        assert not ambiguous and abs(areas[t] - volume) < 1e-13 * abs(volume)
        assert abs(areas[t] ** 2 - 0.25 * np.linalg.det(gram)) < 1e-12 * abs(areas[t]) ** 2
        # The principal branch of acos of the complex cosine.
        cosines = [(sb + sc - sa) / (2 * b * c), (sc + sa - sb) / (2 * c * a), (sa + sb - sc) / (2 * a * b)]
        assert np.abs(angles[t] - np.arccos(np.array(cosines))).max() < 1e-13
        assert abs(angles[t].sum() - math.pi) < 1e-12
        # The layout on the same branch: A_t = (1/2) b c sin(alpha_i) exactly,
        # p_k = (b cos alpha_i, 2 A_t / c).
        assert abs(layout[t, 4] - b * cosines[0]) < 1e-13 * abs(b)
        assert abs(0.5 * layout[t, 2] * layout[t, 5] - areas[t]) < 1e-13 * abs(areas[t])
        # The barycentric gradients: grad lambda_v . (p_v - p_u) = 1 (bilinear).
        p = [layout[t, 0:2], layout[t, 2:4], layout[t, 4:6]]
        for v in range(3):
            gv = gradients[t, 2 * v:2 * v + 2]
            for u in range(3):
                if u != v:
                    assert abs(gv @ (p[v] - p[u]) - 1) < 1e-11
        cot[(t, 0)] = (sb + sc - sa) / (4 * areas[t])
        cot[(t, 1)] = (sc + sa - sb) / (4 * areas[t])
        cot[(t, 2)] = (sa + sb - sc) / (4 * areas[t])
    # The cotangent weights on the continuation branch: w_e = (cot alpha_e + cot beta_e) / 2.
    expected = np.zeros(len(q.edges()), dtype=complex)
    for t, f in enumerate(q.faces()):
        for slot in range(3):
            u, v = f[slot], f[(slot + 1) % 3]
            expected[index[(min(u, v), max(u, v))]] += 0.5 * cot[(t, (slot + 2) % 3)]
    assert np.abs(weights - expected).max() < 1e-12 * np.abs(weights).max()
    # The harmonic space is a complex null space of [d1; d0.T M1].
    H = np.asarray(q.harmonic_basis())
    S = np.vstack([np.asarray(q.d1()), np.asarray(q.d0()).T @ np.diag(weights)])
    assert H.shape == (len(q.edges()), 2) and np.abs(S @ H).max() < 1e-12
    assert np.abs(H.imag).max() > 1e-3
    # The transpose pairing: G is complex symmetric with a nonzero imaginary
    # part (not Hermitian), J = G^{-1} R^T with the same R the spec writes.
    G, R, J = np.asarray(q.gram()), np.asarray(q.rotation_pairing()), np.asarray(q.complex_structure())
    assert np.abs(G - G.T).max() < 1e-12 * np.abs(G).max()
    assert np.abs(G.imag).max() > 1e-4 * np.abs(G).max()
    assert np.abs(G - G.conj().T).max() > 1e-4 * np.abs(G).max()
    assert np.abs(J - np.linalg.solve(G, R.T)).max() < 1e-10 * np.abs(J).max()
    assert np.linalg.norm(J @ J + np.eye(2)) == pytest.approx(q.j_residual())

    def whitney(t, omega):
        f = q.faces()[t]
        w = np.zeros(2, dtype=complex)
        for slot in range(3):
            u, v = f[slot], f[(slot + 1) % 3]
            value = (1 if u < v else -1) * omega[index[(min(u, v), max(u, v))]]
            w += value * (gradients[t, 2 * ((slot + 1) % 3):2 * ((slot + 1) % 3) + 2] - gradients[t, 2 * slot:2 * slot + 2])
        return w / 3

    def rot90(w):
        return np.array([-w[1], w[0]])

    G_expected = np.zeros((2, 2), dtype=complex)
    R_expected = np.zeros((2, 2), dtype=complex)
    for t in range(len(q.faces())):
        w = [whitney(t, H[:, a]) for a in range(2)]
        for a in range(2):
            for b in range(2):
                G_expected[a, b] += areas[t] * (w[a] @ w[b])            # bilinear, no conjugate
                R_expected[a, b] += areas[t] * (rot90(w[a]) @ w[b])
    assert np.abs(G - G_expected).max() < 1e-12 * np.abs(G).max()
    assert np.abs(R - R_expected).max() < 1e-12 * np.abs(R).max()
    # The eigenline is an eigenvector of J; the periods are the plain signed sums.
    omega = np.asarray(q.holomorphic_form())
    c = np.linalg.lstsq(H, omega, rcond=None)[0]
    assert np.abs(H @ c - omega).max() < 1e-12
    assert proportional(J @ c, c)
    P_A, P_B = q.periods()
    assert abs(sum(s * omega[e] for e, s in q.cycle_A()) - P_A) < 1e-12
    assert abs(sum(s * omega[e] for e, s in q.cycle_B()) - P_B) < 1e-12
    assert abs(q.tau() - P_B / P_A) < 1e-14
    assert abs(np.linalg.norm(q.bloch()) - 1.0) < 1e-12


def test_a_root_on_the_continuation_segment_is_refused_by_name():
    base = flat(1j, 4, 4)
    lengths = [complex(l) for l in base.lengths()]
    # A degenerate triangle at the end of the segment: a = b + c.
    i, j, k = base.faces()[0]
    index = {tuple(e): k_ for k_, e in enumerate(base.edges())}
    e_jk, e_ki, e_ij = index[(min(j, k), max(j, k))], index[(min(k, i), max(k, i))], index[(min(i, j), max(i, j))]
    lengths[e_jk] = (lengths[e_ki] + lengths[e_ij]) * (1 + 0j)
    lengths[e_ij] = lengths[e_ij] * (1 + 1e-9j)  # off the real locus
    with pytest.raises(ValueError, match="continuation of the Heron root"):
        rebuilt(base, lengths=lengths)


# --------------------------------------------------------------------------- #
# (f) Connection.transportedPeriod
# --------------------------------------------------------------------------- #
def test_transported_period_primitive():
    base = flat(0.3 + 1.2j, 4, 4)
    K = cob.ChainComplex.fromTopCells([list(f) for f in base.faces()])
    canonical = [tuple(int(v) for v in e) for e in K.kSimplexVertices(1)]
    assert sorted(canonical) == canonical
    nE = len(canonical)
    rng = np.random.default_rng(2)
    omega = rng.normal(size=nE) + 1j * rng.normal(size=nE)
    walk_A, walk_B = list(base.walk_A()), list(base.walk_B())
    assert walk_A[0][0] == walk_B[0][0] == base.base_vertex()

    def plain(walk):
        return sum((1 if u < v else -1) * omega[canonical.index((min(u, v), max(u, v)))] for u, v in walk)

    # Trivial connection: the plain signed sum, and the qubit's own periods.
    trivial = ch.Connection.trivial(K)
    assert abs(trivial.transportedPeriod(omega, walk_A) - plain(walk_A)) < 1e-13
    assert abs(trivial.holonomy(walk_A) - 1) == 0
    own = np.asarray(base.holomorphic_form())
    own_canonical = np.zeros(nE, dtype=complex)
    for e in range(nE):
        own_canonical[base.canonical_edge_index(e)] = own[e]
    P_A, P_B = base.periods()
    assert abs(trivial.transportedPeriod(own_canonical, walk_A) - P_A) < 1e-13
    assert abs(trivial.transportedPeriod(own_canonical, walk_B) - P_B) < 1e-13
    # Pure gauge U = 1^g on the twisted cochain rho_1 omega: g_{v0}^{-1} times the
    # plain period, so the ratio P_B / P_A is invariant.
    g = {v: cmath.exp(1j * x) * (1 + 0.3 * math.sin(v)) for v, x in enumerate(np.linspace(-2, 2, len(base.vertices())))}
    U = trivial.gauge(g)
    twisted = np.array([omega[k] / g[x] for k, (x, y) in enumerate(canonical)])
    v0 = walk_A[0][0]
    for walk in (walk_A, walk_B):
        assert abs(U.holonomy(walk) - 1) < 1e-12
        assert abs(U.transportedPeriod(twisted, walk) - plain(walk) / g[v0]) < 1e-12 * abs(plain(walk))
    assert abs(U.transportedPeriod(twisted, walk_B) / U.transportedPeriod(twisted, walk_A)
               - plain(walk_B) / plain(walk_A)) < 1e-12
    # The base point matters: the same walk started elsewhere is another base.
    rotated = walk_A[1:] + walk_A[:1]
    v1 = rotated[0][0]
    assert abs(U.transportedPeriod(twisted, rotated) - plain(walk_A) / g[v1]) < 1e-12 * abs(plain(walk_A))
    # Refusals by name.
    with pytest.raises(ValueError, match="does not chain"):
        U.transportedPeriod(twisted, [walk_A[0], walk_A[2]] + walk_A[1:2] + walk_A[3:])
    with pytest.raises(ValueError, match="not closed"):
        U.transportedPeriod(twisted, walk_A[:-1])
    with pytest.raises(ValueError, match="is not an edge"):
        U.holonomy([(0, 6), (6, 0)])
    with pytest.raises(ValueError, match="entries for"):
        U.transportedPeriod(twisted[:-1], walk_A)
