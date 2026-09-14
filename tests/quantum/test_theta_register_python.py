"""The theta quantization register (issue #1107): T1-T7 on synthetic data,
T8 on the seeded collar hosts."""
import os
import sys

import numpy as np
import pytest

from tessera.quantum import ThetaRegister

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "cobordism"))
import qubit_animation as qa  # noqa: E402

TOL = 1e-12
REGISTER = ThetaRegister(level=2, tolerance=TOL)
GEN = REGISTER.generators()
FORMS = REGISTER.level_two_forms()
TAU = 0.3 + 1.1j


def test_t1_t2_closure_and_unitarity_at_several_moduli():
    for name in ("S", "T", "swap"):
        for tau in (TAU, -0.2 + 0.8j, -0.7 + 2.3j, 1j):
            fit = REGISTER.weil(GEN[name], tau)
            assert fit.residual <= TOL, (name, tau, fit.residual)
            assert fit.unitarity_defect <= TOL, (name, tau, fit.unitarity_defect)
            assert fit.antiunitary == (name == "swap")


def test_t3_projective_group_law():
    s, t = GEN["S"], GEN["T"]
    for first, second in ((s, s), (s, t), (t, s), (t, t)):
        direct = REGISTER.weil(first @ second, TAU)
        mid = REGISTER.siegel(second, TAU)[0]
        chained = REGISTER.weil(first, mid).compose(REGISTER.weil(second, TAU))
        assert REGISTER.projective_distance(direct.matrix, chained.matrix) <= TOL
    # (ST)^3 = S^2 in SL(2, Z): the fitted matrices agree projectively
    st = s @ t
    assert REGISTER.projective_distance(REGISTER.weil(st @ st @ st, TAU).matrix,
                                        REGISTER.weil(s @ s, TAU).matrix) <= TOL


def test_t4_level_two_closed_forms():
    assert REGISTER.projective_distance(REGISTER.weil(GEN["S"], TAU).matrix, FORMS["S"]) <= TOL
    assert REGISTER.projective_distance(REGISTER.weil(GEN["T"], TAU).matrix, FORMS["T"]) <= TOL


def test_t5_direct_sum_is_tensor_product():
    omega = np.diag([TAU, -0.2 + 0.8j])
    rng = np.random.default_rng(1)
    z = REGISTER.samples(2, rng)
    genus2 = REGISTER.basis(z, omega)
    product = np.einsum("is,js->ijs", REGISTER.basis(z[:, :1], TAU),
                        REGISTER.basis(z[:, 1:], -0.2 + 0.8j)).reshape(genus2.shape)
    assert np.linalg.norm(genus2 - product) / np.linalg.norm(genus2) <= TOL
    joint = REGISTER.weil(REGISTER.direct_sum(GEN["S"], GEN["T"]), omega)
    kron = np.kron(REGISTER.weil(GEN["S"], TAU).matrix, REGISTER.weil(GEN["T"], -0.2 + 0.8j).matrix)
    assert REGISTER.projective_distance(joint.matrix, kron) <= TOL
    assert REGISTER.schmidt_rank(joint.matrix, (2, 2)) == 1


def test_t6_the_shear_is_controlled_z_and_entangles():
    omega = np.diag([TAU, -0.2 + 0.8j])
    fit = REGISTER.weil(GEN["shear"], omega)
    assert fit.residual <= TOL
    assert REGISTER.projective_distance(fit.matrix, FORMS["shear"]) <= TOL
    assert REGISTER.schmidt_rank(fit.matrix, (2, 2)) == 2


def test_t7_polarization_independence():
    for name in ("S", "T"):
        assert REGISTER.projective_distance(REGISTER.weil(GEN[name], TAU).matrix,
                                            REGISTER.weil(GEN[name], -0.7 + 2.3j).matrix) <= TOL
    omega = np.diag([TAU, -0.2 + 0.8j])
    other = np.array([[-0.7 + 2.3j, 0.15 + 0.05j], [0.15 + 0.05j, -0.2 + 0.8j]])
    assert REGISTER.projective_distance(REGISTER.weil(GEN["shear"], omega).matrix,
                                        REGISTER.weil(GEN["shear"], other).matrix) <= TOL


def test_refusals():
    with pytest.raises(ValueError):
        REGISTER.weil(np.array([[2.0, 0.0], [0.0, 1.0]]), TAU)     # not symplectic
    with pytest.raises(ValueError):
        REGISTER.basis([[0.0]], 0.3 - 1.1j)                         # lower half-plane
    with pytest.raises(ValueError):
        ThetaRegister(level=0)


@pytest.mark.parametrize("twist", ["none", "swap"])
def test_t8_two_tori(twist):
    args = qa.build_parser().parse_args(["theta", "--collar-twist", twist])
    record = qa.theta(qa._verify_config(args))
    assert record["all_pass"], [row for row in record["checks"] if not row["pass"]]
    row = next(r for r in record["checks"] if r["id"] == "T8:A->B")
    assert row["measured"]["antiunitary"] == (twist == "swap")
    assert row["measured"]["distance_to_expected"] <= TOL


def test_t8_four_tori_is_a_product_operator():
    args = qa.build_parser().parse_args(["theta", "--tori", "4"])
    record = qa.theta(qa._verify_config(args))
    assert record["all_pass"], [row for row in record["checks"] if not row["pass"]]
    row = next(r for r in record["checks"] if r["id"] == "T8:whole")
    assert row["measured"]["schmidt_rank"] == 1
    assert row["measured"]["kron_distance"] <= TOL


def test_coherent_state_entanglement_is_the_neck():
    """The geometric state at z is a product for a diagonal Omega and is
    entangled once Omega_12 != 0; the Schmidt spectrum is a probability
    vector and the entropy is symmetric under exchanging the tori."""
    diagonal = np.diag([0.3 + 1.1j, -0.2 + 0.8j])
    coupled = diagonal + np.array([[0, 0.2 + 0.1j], [0.2 + 0.1j, 0]])
    rng = np.random.default_rng(3)
    for z in [np.zeros(2), *REGISTER.samples(2, rng)[:3]]:
        entropy, spectrum = REGISTER.entanglement_entropy(REGISTER.coherent_state(z, diagonal), (2, 2))
        assert entropy <= 1e-12
        assert abs(spectrum.sum() - 1.0) <= 1e-12
        entropy_c, spectrum_c = REGISTER.entanglement_entropy(REGISTER.coherent_state(z, coupled), (2, 2))
        assert entropy_c > 1e-6
        swapped = REGISTER.entanglement_entropy(
            REGISTER.coherent_state(z[::-1], coupled[::-1, ::-1]), (2, 2))[0]
        assert abs(swapped - entropy_c) <= 1e-9
    # more coupling, more entanglement at the origin
    weak = diagonal + np.array([[0, 0.05], [0.05, 0]])
    assert (REGISTER.entanglement_entropy(REGISTER.coherent_state(np.zeros(2), weak), (2, 2))[0]
            < REGISTER.entanglement_entropy(REGISTER.coherent_state(np.zeros(2), coupled), (2, 2))[0])


def test_entanglement_entropy_of_a_bell_vector_is_one_bit():
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    entropy, spectrum = REGISTER.entanglement_entropy(bell, (2, 2))
    assert abs(entropy - 1.0) <= 1e-12
    assert np.allclose(spectrum, [0.5, 0.5])
