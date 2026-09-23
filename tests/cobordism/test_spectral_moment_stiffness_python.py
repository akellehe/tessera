# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The spectral-moment stiffness of the geometric action (#1183).

The whitepaper makes self-trapping depend on the geometric action's Hessian
being positive on the range of the Hellmann-Feynman force, which pure Regge
fails, and assigns that stiffness to the spectral-moment part of the action.
The code's spectral entropy is a functional of the normalized spectrum, whose
stiffness per degree of freedom falls as the inverse number of cells. The
stiffness here is built on the local parts of the whitepaper's power sums,
mu_j(x) = (L_k^j)_xx with tr L_k^j = sum_x mu_j(x):

    S_M(z) = 1/2 sum_j beta_j sum_x (mu_j(x; z) - mu_j(x; z_0))^2 .

These tests hold its value, gradient and Hessian-vector product to their
definitions, show that it keeps the carrier stationary, that its stiffness per
degree of freedom does not fall with the size of the complex on flat periodic
Kuhn tori (where the entropy's does), and that the objectives carry it only
when it is declared.
"""
import itertools
import math

import numpy as np
import pytest

import tessera as T

cob = T.cobordism
COEFFICIENTS = [1.0, 0.5]


def torus(n, jitter=0.0, seed=0):
    """The periodic Kuhn triangulation of the unit cubic lattice of n^3
    vertices, flat, or with its squared lengths scaled by 1 + jitter * noise."""
    index = lambda c: ((c[0] % n) * n + (c[1] % n)) * n + (c[2] % n)
    cells = []
    for corner in itertools.product(range(n), repeat=3):
        for order in itertools.permutations(range(3)):
            c = list(corner)
            path = [index(c)]
            for axis in order:
                c[axis] += 1
                path.append(index(c))
            cells.append(path)
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    coordinates = {index(c): np.array(c, dtype=float) for c in itertools.product(range(n), repeat=3)}
    rng = np.random.default_rng(seed)
    steps, midpoints = [], []
    for edge in spacetime.getEdgeList().toVector():
        a, b = coordinates[edge.getSource().getId()], coordinates[edge.getTarget().getId()]
        d = b - a
        d -= n * np.rint(d / n)
        edge.setLength(math.sqrt(float(d @ d) * (1.0 + jitter * rng.standard_normal())))
        steps.append(d)
        midpoints.append(a + 0.5 * d)
    return spacetime, np.array(steps), np.array(midpoints), coordinates


def squared_lengths(spacetime):
    return np.array([edge.getLength() ** 2 for edge in spacetime.getEdgeList().toVector()])


def set_squared_lengths(spacetime, values):
    for edge, value in zip(spacetime.getEdgeList().toVector(), values):
        edge.setLength(np.sqrt(complex(value)))


def moments(spacetime, k, m):
    return np.asarray(cob.HodgeLaplacian(spacetime).localSpectralMoments(k, m))


@pytest.mark.parametrize("k", [0, 1])
def test_the_local_moments_are_the_diagonals_of_the_powers_and_sum_to_the_power_sums(k):
    spacetime, _, _, _ = torus(3, jitter=0.05)
    hodge = cob.HodgeLaplacian(spacetime)
    L = np.asarray(hodge.laplacian(k))
    size = int(round(math.sqrt(L.size)))
    L = L.reshape(size, size)
    local = np.asarray(hodge.localSpectralMoments(k, 3)).reshape(size, 3)
    power = np.eye(size)
    for j in range(3):
        power = power @ L
        assert np.abs(local[:, j] - np.diag(power)).max() < 1e-10 * max(1.0, np.abs(np.diag(power)).max())
        assert local[:, j].sum() == pytest.approx(np.trace(power), rel=1e-12)
    with pytest.raises(ValueError):
        hodge.localSpectralMoments(k, 0)


@pytest.mark.parametrize("k", [0, 1])
def test_the_stiffness_vanishes_with_its_gradient_at_the_carrier(k):
    spacetime, _, _, _ = torus(3, jitter=0.05)
    hodge = cob.HodgeLaplacian(spacetime)
    reference = list(moments(spacetime, k, len(COEFFICIENTS)))
    assert abs(hodge.spectralMomentStiffness(k, reference, COEFFICIENTS)) < 1e-24
    assert np.abs(hodge.spectralMomentStiffnessGradient(k, reference, COEFFICIENTS)).max() < 1e-12
    with pytest.raises(ValueError):
        hodge.spectralMomentStiffness(k, reference[:-1], COEFFICIENTS)


@pytest.mark.parametrize("k", [0, 1])
def test_the_gradient_and_the_hessian_product_are_exact(k):
    """Away from the carrier, against central differences of the value and of
    the gradient, in complex squared lengths."""
    spacetime, _, _, _ = torus(3, jitter=0.05, seed=1)
    reference = list(moments(spacetime, k, len(COEFFICIENTS)))
    rng = np.random.default_rng(2)
    carrier = squared_lengths(spacetime)
    moved = carrier * (1.0 + 0.03 * rng.standard_normal(len(carrier)) + 0.01j * rng.standard_normal(len(carrier)))
    set_squared_lengths(spacetime, moved)
    hodge = cob.HodgeLaplacian(spacetime)
    gradient = np.asarray(hodge.spectralMomentStiffnessGradient(k, reference, COEFFICIENTS))
    step = 1e-6
    for e in rng.choice(len(moved), 6, replace=False):
        values = []
        for sign in (1.0, -1.0):
            shifted = moved.copy()
            shifted[e] += sign * step
            set_squared_lengths(spacetime, shifted)
            values.append(cob.HodgeLaplacian(spacetime).spectralMomentStiffness(k, reference, COEFFICIENTS))
        assert gradient[e] == pytest.approx((values[0] - values[1]) / (2.0 * step), rel=1e-6, abs=1e-9)
    direction = rng.standard_normal(len(moved)) + 1j * rng.standard_normal(len(moved))
    set_squared_lengths(spacetime, moved)
    product = np.asarray(cob.HodgeLaplacian(spacetime).spectralMomentStiffnessHessianProduct(
        k, reference, COEFFICIENTS, list(direction)))
    gradients = []
    for sign in (1.0, -1.0):
        set_squared_lengths(spacetime, moved + sign * step * direction)
        gradients.append(np.asarray(cob.HodgeLaplacian(spacetime).spectralMomentStiffnessGradient(
            k, reference, COEFFICIENTS)))
    difference = (gradients[0] - gradients[1]) / (2.0 * step)
    assert np.abs(product - difference).max() < 1e-6 * np.abs(product).max()
    with pytest.raises(RuntimeError):
        cob.HodgeLaplacian(spacetime).spectralMomentStiffnessHessianProduct(k, reference, COEFFICIENTS, [0.0])


def test_at_the_carrier_the_hessian_is_the_gauss_newton_form():
    """v . H v = sum_j beta_j sum_x (d mu_j(x) / dt)^2 along a real v, which is
    never negative: the stiffness is positive semidefinite at the carrier."""
    spacetime, _, _, _ = torus(3, jitter=0.05, seed=3)
    k = 1
    reference = list(moments(spacetime, k, len(COEFFICIENTS)))
    carrier = squared_lengths(spacetime)
    rng = np.random.default_rng(4)
    for _ in range(3):
        v = rng.standard_normal(len(carrier))
        product = np.asarray(cob.HodgeLaplacian(spacetime).spectralMomentStiffnessHessianProduct(
            k, reference, COEFFICIENTS, list(v.astype(complex))))
        step = 1e-6
        velocities = []
        for sign in (1.0, -1.0):
            set_squared_lengths(spacetime, carrier + sign * step * v)
            velocities.append(moments(spacetime, k, len(COEFFICIENTS)).reshape(-1, len(COEFFICIENTS)))
        set_squared_lengths(spacetime, carrier)
        rate = (velocities[0] - velocities[1]) / (2.0 * step)
        expected = sum(beta * np.sum(rate[:, j] ** 2) for j, beta in enumerate(COEFFICIENTS))
        assert float(np.real(v @ product)) == pytest.approx(float(np.real(expected)), rel=1e-6)
        assert float(np.real(v @ product)) > 0.0


def _rayleigh(spacetime, k, v):
    reference = list(moments(spacetime, k, len(COEFFICIENTS)))
    product = np.asarray(cob.HodgeLaplacian(spacetime).spectralMomentStiffnessHessianProduct(
        k, reference, COEFFICIENTS, list(v.astype(complex))))
    return float(np.real(v @ product)) / float(v @ v)


def _directions(n, steps, midpoints, coordinates, spacetime):
    q = 2.0 * np.pi / n
    displacement = {vid: np.array([0.0, np.sin(q * c[0]), 0.0]) for vid, c in coordinates.items()}
    return {
        "uniform traceless": steps[:, 1] ** 2 - steps[:, 2] ** 2,
        "traceless wave": np.cos(q * midpoints[:, 0]) * (steps[:, 1] ** 2 - steps[:, 2] ** 2),
        "dilation": np.sum(steps ** 2, axis=1),
        "vertex displacement": np.array([2.0 * d @ (displacement[e.getTarget().getId()]
                                                    - displacement[e.getSource().getId()])
                                         for e, d in zip(spacetime.getEdgeList().toVector(), steps)]),
    }


def test_the_stiffness_per_degree_of_freedom_does_not_fall_with_the_size_of_the_complex():
    """On flat periodic tori of n^3 vertices the Rayleigh quotient of the
    degree-one stiffness is the same at every n, along a uniform traceless
    strain, a traceless wave of the longest wavelength, a dilation and a vertex
    displacement (measured 3.47e4, 3.0e4 and 2.84e4, 4.51e5, 9.8e4 and 9.9e4
    at n = 3, 4, 5 with these coefficients), and positive along all four. The
    entropy's falls as 1/n^3 (7.7e-4 to 2.3e-4 from n = 4 to 6). Regge has no
    stiffness along the displacement and the uniform strain at all."""
    quotients = {}
    for n in (3, 4):
        spacetime, steps, midpoints, coordinates = torus(n)
        for name, v in _directions(n, steps, midpoints, coordinates, spacetime).items():
            quotients.setdefault(name, []).append(_rayleigh(spacetime, 1, v))
    for name, values in quotients.items():
        values = np.array(values)
        assert np.all(values > 0.0), name
        # Does not fall: the larger complex's quotient is within a tenth of the
        # smaller's or above it. A 1/N fall would be (4/3)^3 = 2.4. The
        # traceless wave's wavelength grows with n, and along it the quotient
        # of the covariant Whitney operator rises (2.69e4 at n = 3, 3.80e4 at
        # n = 4), so an upper bound on the ratio is not part of the property.
        assert values[1] >= values[0] / 1.1, name
    assert quotients["uniform traceless"][0] == pytest.approx(quotients["uniform traceless"][1], rel=1e-8)


def test_at_degree_zero_the_scalar_moments_leave_traceless_strains_free():
    """The local moments of the degree-zero operator are scalars at vertices,
    which a traceless strain of the symmetric lattice moves only at second
    order: those strains are zero modes of the degree-zero stiffness, while a
    dilation and a vertex displacement are stiffened. The degree-one
    operator's cells are edges of different directions, which is why it
    stiffens every direction."""
    spacetime, steps, midpoints, coordinates = torus(3)
    directions = _directions(3, steps, midpoints, coordinates, spacetime)
    assert abs(_rayleigh(spacetime, 0, directions["uniform traceless"])) < 1e-20
    assert abs(_rayleigh(spacetime, 0, directions["traceless wave"])) < 1e-20
    assert _rayleigh(spacetime, 0, directions["dilation"]) > 1.0
    assert _rayleigh(spacetime, 0, directions["vertex displacement"]) > 1.0


def test_the_objectives_carry_the_stiffness_only_when_it_is_declared():
    spacetime, _, _, _ = torus(3, jitter=0.05, seed=5)
    k = 0
    reference = moments(spacetime, k, len(COEFFICIENTS))
    carrier = squared_lengths(spacetime)
    set_squared_lengths(spacetime, carrier * (1.0 + 0.02 * np.random.default_rng(6).standard_normal(len(carrier))))
    context = cob.ObjectiveContext()
    context.spacetime = spacetime
    context.einstein_hilbert = False
    for objective in (cob.JointStationarityObjective(), cob.LegacyObjective()):
        assert objective.terms(context).moment_stiffness == 0.0
    context.moment_stiffness_weight = 2.0
    context.moment_stiffness_degrees = [k]
    context.moment_stiffness_coefficients = COEFFICIENTS
    context.moment_stiffness_reference = [list(reference)]
    value = cob.HodgeLaplacian(spacetime).spectralMomentStiffness(k, list(reference), COEFFICIENTS)
    assert value.real > 0.0
    for objective in (cob.JointStationarityObjective(), cob.LegacyObjective()):
        terms = objective.terms(context)
        assert terms.moment_stiffness == pytest.approx(2.0 * value.real, rel=1e-12)
        assert cob.CobordismObjective.total(terms) >= terms.moment_stiffness


def kuhn_ball(n=4):
    """The Kuhn triangulation of an n^3 block of unit cubes, with its boundary,
    the kind of ball the whitepaper's self-trapping computation uses."""
    index = lambda c: (c[0] * (n + 1) + c[1]) * (n + 1) + c[2]
    cells = []
    for corner in itertools.product(range(n), repeat=3):
        for order in itertools.permutations(range(3)):
            c = list(corner)
            path = [index(c)]
            for axis in order:
                c[axis] += 1
                path.append(index(c))
            cells.append(path)
    spacetime = T.Spacetime.fromVertexTuples(3, cells, 1.0, 0.0)
    coordinates = {index(c): np.array(c, dtype=float) for c in itertools.product(range(n + 1), repeat=3)}
    for edge in spacetime.getEdgeList().toVector():
        d = coordinates[edge.getTarget().getId()] - coordinates[edge.getSource().getId()]
        edge.setLength(math.sqrt(float(d @ d)))
    return spacetime


def test_the_stiffness_is_positive_on_the_range_of_the_hellmann_feynman_force():
    """Section 7's condition, for the stiffness alone: along the force
    F_e = -psi^T (dL_0 / dz_e) psi of the lowest nonconstant mode psi of the
    degree-zero operator on a Kuhn ball, F . H_M F > 0 at the carrier."""
    spacetime = kuhn_ball(3)
    hodge = cob.HodgeLaplacian(spacetime)
    L = np.asarray(hodge.laplacian(0))
    size = int(round(math.sqrt(L.size)))
    L = L.reshape(size, size).real
    values, vectors = np.linalg.eig(L)
    order = np.argsort(values.real)
    psi = np.real(vectors[:, order[1]])                             # the lowest mode above the constant
    psi /= np.linalg.norm(psi)
    force = []
    for edge in spacetime.getEdgeList().toVector():
        dL = np.asarray(hodge.laplacianGradient(0, edge.getSource().getId(), edge.getTarget().getId()))
        force.append(-float(np.real(psi @ dL.reshape(size, size) @ psi)))
    force = np.array(force)
    assert np.abs(force).max() > 1e-6
    for k in (0, 1):
        quotient = _rayleigh(spacetime, k, force)
        assert quotient > 0.0
