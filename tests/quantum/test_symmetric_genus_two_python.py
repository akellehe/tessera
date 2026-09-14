"""The Z/10-symmetric genus-2 surface (issue #1118): a closed orientable
simplicial surface of Euler characteristic -2 whose rotations are
simplicial isometries; its symplectic marking; the induced map of the
order-5 rotation is a symplectic matrix of order 5 that mixes the two
factors, and its reduction mod 2 has order 5 in Sp(4, F_2)."""
import numpy as np
import pytest

import tessera as T  # noqa: F401
from tessera.quantum import SurfacePeriods, SymmetricGenusTwo, ThetaRegister

SURFACE = SymmetricGenusTwo()
J = ThetaRegister.symplectic_form(2).astype(int)


def test_surface_is_a_closed_orientable_genus_two_complex():
    assert SURFACE.euler_characteristic == -2
    assert len(SURFACE.vertices) == 28 and len(SURFACE.edges) == 90 and len(SURFACE.faces) == 60
    read = SURFACE.periods
    assert read.harmonic_rank == 4 and read.positive and not read.orientation_flipped
    assert read.symmetry_residual <= 1e-9


def test_rotations_are_simplicial_automorphisms_of_order_ten():
    perm = SURFACE.rotation(1)
    power = list(range(28))
    for k in range(1, 11):
        power = [perm[v] for v in power]
        assert SURFACE.is_automorphism(SURFACE.rotation(k))
        assert (power == list(range(28))) == (k == 10)


def test_marking_is_symplectic_and_the_intersection_form_unimodular():
    form = SURFACE.intersection
    assert (form == -form.T).all() and abs(int(round(np.linalg.det(form)))) == 1
    assert SURFACE.intersection_residual <= 1e-6
    basis = SURFACE.symplectic_basis
    assert (basis @ form @ basis.T == J).all()
    # the marking cycles, read on the surface, intersect as declared
    cycles = SURFACE.a_cycles + SURFACE.b_cycles
    measured, residual = SURFACE.periods.intersection_form(cycles)
    assert (measured == J).all() and residual <= 1e-6


def test_flat_torus_calibrates_the_intersection_sign():
    from tessera import observables as obs
    torus = obs.SimplicialQubit.flat_torus(0.3 + 1.1j, 3, 3)
    edges = torus.edges()
    lengths = {(min(int(i), int(j)), max(int(i), int(j))): float(np.real(l)) for (i, j), l in zip(edges, torus.lengths())}
    faces = [tuple(int(v) for v in face) for face in torus.faces()]

    def cycle(steps):
        out = []
        for e, sign in steps:
            i, j = (int(v) for v in edges[int(e)])
            out.append((i, j) if sign > 0 else (j, i))
        return out
    a, b = cycle(torus.cycle_A()), cycle(torus.cycle_B())
    read = SurfacePeriods(faces, lengths, [a], [b], root_face=faces[0])
    form, residual = read.intersection_form([a, b])
    assert (form == np.array([[0, 1], [-1, 0]])).all() and residual <= 1e-9


@pytest.mark.parametrize("k,order", [(2, 5), (4, 5), (5, 2), (1, 10)])
def test_induced_maps_are_symplectic_of_the_rotation_order(k, order):
    m = SURFACE.induced_map(k)
    assert (m.T @ J @ m == J).all()
    power = np.eye(4, dtype=int)
    for n in range(1, order + 1):
        power = power @ m
        assert (power == np.eye(4, dtype=int)).all() == (n == order)
    if k == 5:
        assert (m == -np.eye(4, dtype=int)).all()


def test_order_five_map_mixes_and_has_order_five_mod_two():
    m = SURFACE.induced_map(2)
    assert (SURFACE.induced_map(4) == m @ m).all()
    off = np.abs(m[:2, 2:]).sum() + np.abs(m[2:, :2]).sum()
    assert off > 0
    mod2 = m % 2
    power = np.eye(4, dtype=int)
    for n in range(1, 6):
        power = (power @ mod2) % 2
        assert (power == np.eye(4, dtype=int)).all() == (n == 5)


def test_period_matrix_is_fixed_by_the_rotation():
    omega = SURFACE.periods.omega
    m = SURFACE.induced_map(2)
    moved = (m[2:, :2] + m[2:, 2:] @ omega) @ np.linalg.inv(m[:2, :2] + m[:2, 2:] @ omega)
    assert np.linalg.norm(moved - omega) <= 1e-9 * np.linalg.norm(omega)
