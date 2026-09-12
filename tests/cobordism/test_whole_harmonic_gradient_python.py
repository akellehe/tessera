# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The analytic gradient of the whole-complex harmonic reading (#1055).

Two checks, because neither alone is enough.

The Euler identity `Sum_e s_e dF_e = 0` is exact: the residual is projective in
the coefficient vector, so a common scaling of every squared length leaves it
alone. It is also nearly useless on its own -- it constrains ONE direction, and
a wrong form of this gradient passed it at 3.8e-16 while its imaginary part was
wrong.

So the derivation is checked against an INDEPENDENT NumPy computation that uses
nothing from tessera: a synthetic operator plays the pencil, its eigenvectors
the harmonic images under a holomorphic normalisation, a square period block
defines the live frame, and the reference is a Richardson-extrapolated
difference quotient. That is what caught the errors.

What it caught, twice:

  * `c = G^-1 Pi* p` depends on s-bar through `Pi*`, so along a real direction
    BOTH terms are present: `dc = -G^-1 (Pi* dPi) c + G^-1 dPi* (p - Pi c)`.
  * The packed gradient is the PAIR of directional derivatives, along ds = 1
    and ds = i. The `(2 Re dF, -2 Im dF)` shortcut off a single holomorphic dF
    needs everything between the coordinate and the residual to be holomorphic,
    and c is not.
  * The reported state is `B c` in the live period frame, not c in the band's
    arbitrary basis, so `d(B c) = (dB)c + B(dc)`. Omitting dB differentiates a
    different, basis-dependent read.

Where the period fit is exact the non-holomorphic fit term vanishes
algebraically, which is why a node alone does not reliably separate the first
two forms.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))

import emergence_animation as ea  # noqa: E402

TAU_A, TAU_B, GRID = 0.3 + 1.1j, -0.2 + 0.8j, 3


def holomorphic(packed):
    """dF from the packed (2 Re dF, -2 Im dF)."""
    packed = np.asarray(packed)
    return 0.5 * (packed.real - 1j * packed.imag)


@pytest.fixture(scope="module")
def seeded():
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False, pin_boundary=True,
                             readout="whole", tori=2, output_state="0.1+1.3j")
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    return held["node"]


def test_it_has_one_component_per_edge(seeded):
    edges = seeded.spacetime().getEdgeList().toVector()
    chi = seeded.two_body_target().chi
    assert np.asarray(seeded.whole_harmonic_residual_gradient(chi)).shape == (len(edges),)


def test_the_euler_identity_is_exact(seeded):
    """Necessary, and nowhere near sufficient -- see the module docstring."""
    chi = seeded.two_body_target().chi
    packed = np.asarray(seeded.whole_harmonic_residual_gradient(chi))
    squared = np.array([complex(edge.getLength()) ** 2
                        for edge in seeded.spacetime().getEdgeList().toVector()])
    numerator = abs(np.sum(squared * holomorphic(packed)))
    scale = np.abs(packed).max() * np.abs(squared).max()
    # On the exact product collar this gradient is zero; dividing two
    # round-off-sized values makes a meaningless O(1) ratio. Keep the relative
    # check away from zero and an absolute round-off allowance at the floor.
    allowance = 1e-10 * scale + 1e-12 * max(np.abs(squared).max(), 1.0)
    assert numerator < allowance, (
        "Euler identity violated: %.3e > %.3e" % (numerator, allowance))


def test_a_reading_that_refuses_has_no_direction():
    """The zero gradient where the residual is the full leak.

    No declared output state, so the 4-dimensional chi is the target and the
    two-torus harmonic space has rank 2: it cannot carry it, and a direction
    invented for it would be worse than none.
    """
    held = {}
    config = ea.build_config(steps=0, inputs=ea.InputMode.QUBIT, grid=GRID,
                             operator="xx", tau_a=TAU_A, tau_b=TAU_B,
                             input_weight=100.0, regge=False, pin_boundary=True,
                             readout="transfer", tori=2)
    ea.drive(config, progress=False, on_node=lambda node: held.update(node=node))
    node = held["node"]
    chi = node.two_body_target().chi
    assert node.whole_harmonic_residual(chi) == 1.0
    assert "rank 2" in node.whole_harmonic_obstruction
    assert np.count_nonzero(
        np.asarray(node.whole_harmonic_residual_gradient(chi))) == 0


# --------------------------------------------------------------------------- #
# the derivation itself, against an independent computation
# --------------------------------------------------------------------------- #

def _synthetic(seed=11, n=10, rank=3, cycles=6):
    rng = np.random.default_rng(seed)
    parts = {}
    for name, shape in (("A0", (n, n)), ("A1", (n, n)), ("C", (cycles, n)),
                        ("p", (cycles,)), ("w", (rank,))):
        parts[name] = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    pivots = []

    def images(s):
        """Eigenvectors normalised so a FIXED component is 1 -- holomorphic in
        s, so dZ/ds exists and the basis does not jump."""
        values, vectors = np.linalg.eig(parts["A0"] + s * parts["A1"])
        Z = vectors[:, np.argsort(np.abs(values))[:rank]]
        if not pivots:
            pivots.extend(int(np.argmax(np.abs(Z[:, a]))) for a in range(rank))
        for a in range(rank):
            Z[:, a] = Z[:, a] / Z[pivots[a], a]
        return Z

    def residual(s):
        Pi = parts["C"] @ images(s)
        c = np.linalg.solve(Pi.conj().T @ Pi, Pi.conj().T @ parts["p"])
        state = Pi[:rank] @ c
        overlap = np.vdot(state, parts["w"])
        return float(1.0 - abs(overlap) ** 2
                     / (np.vdot(state, state).real
                        * np.vdot(parts["w"], parts["w"]).real))

    def directional(s, direction, h=1e-7, *, differentiate_fit=True,
                    differentiate_frame=True):
        """The engine's chain in NumPy, with each required term switchable."""
        Z = images(s)
        dZ = (images(s + direction * h) - images(s - direction * h)) / (2 * h)
        Pi, dPi = parts["C"] @ Z, parts["C"] @ dZ
        G = Pi.conj().T @ Pi
        c = np.linalg.solve(G, Pi.conj().T @ parts["p"])
        fit = parts["p"] - Pi @ c
        dc = -np.linalg.solve(G, (Pi.conj().T @ dPi) @ c)
        if differentiate_fit:
            dc += np.linalg.solve(G, dPi.conj().T @ fit)
        B, dB = Pi[:rank], dPi[:rank]
        state = B @ c
        dstate = B @ dc
        if differentiate_frame:
            dstate += dB @ c
        overlap = np.vdot(state, parts["w"])
        ss = np.vdot(state, state).real
        ww = np.vdot(parts["w"], parts["w"]).real
        s1, s2 = np.vdot(parts["w"], dstate), np.vdot(state, dstate)
        return 2.0 * (-(overlap * s1 * ss - abs(overlap) ** 2 * s2)
                      / (ss * ss * ww)).real

    return images, residual, directional


def test_the_chain_matches_an_independent_computation():
    """The check that caught both errors. Nothing from tessera is used."""
    images, residual, directional = _synthetic()
    s0 = 0.31 + 0.17j
    images(s0)                                    # pin the pivots

    def richardson(direction, h):
        d1 = (residual(s0 + direction * h) - residual(s0 - direction * h)) / (2 * h)
        d2 = (residual(s0 + direction * h / 2) - residual(s0 - direction * h / 2)) / h
        return (4 * d2 - d1) / 3.0                # O(h^4)

    external = complex(richardson(1.0, 1e-4), richardson(1j, 1e-4))
    analytic = complex(directional(s0, 1.0), directional(s0, 1j))
    # 1e-6 is the REFERENCE's accuracy, not the formula's: its dZ is itself a
    # difference quotient. Measured 3.0e-08.
    assert abs(analytic - external) / abs(external) < 1e-6, (analytic, external)


def test_the_holomorphic_shortcut_alone_is_wrong():
    """The error this suite exists to keep out.

    Dropping the dPi* term and packing a single holomorphic dF gives a
    different answer, so a regression to it is caught rather than passing on
    the Euler identity alone.
    """
    images, residual, directional = _synthetic()
    s0 = 0.31 + 0.17j
    images(s0)

    def richardson(direction, h):
        d1 = (residual(s0 + direction * h) - residual(s0 - direction * h)) / (2 * h)
        d2 = (residual(s0 + direction * h / 2) - residual(s0 - direction * h / 2)) / h
        return (4 * d2 - d1) / 3.0

    external = complex(richardson(1.0, 1e-4), richardson(1j, 1e-4))
    correct = complex(directional(s0, 1.0), directional(s0, 1j))
    missing_fit = complex(
        directional(s0, 1.0, differentiate_fit=False),
        directional(s0, 1j, differentiate_fit=False))
    assert abs(correct - external) / abs(external) < 1e-6
    assert abs(missing_fit - external) / abs(external) > 1e-3


def test_the_live_period_frame_derivative_is_not_optional():
    """The value reads Bc, so differentiating c alone is a different read."""
    images, residual, directional = _synthetic()
    s0 = 0.31 + 0.17j
    images(s0)

    def richardson(direction, h):
        d1 = (residual(s0 + direction * h) - residual(s0 - direction * h)) / (2 * h)
        d2 = (residual(s0 + direction * h / 2) - residual(s0 - direction * h / 2)) / h
        return (4 * d2 - d1) / 3.0

    external = complex(richardson(1.0, 1e-4), richardson(1j, 1e-4))
    missing_frame = complex(
        directional(s0, 1.0, differentiate_frame=False),
        directional(s0, 1j, differentiate_frame=False))
    assert abs(missing_frame - external) / abs(external) > 1e-3
