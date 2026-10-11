# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Every term of the joint action traced through a relaxation (#1285).

Terms used below:

* the *joint action* is S = w_R S_Regge + S_hol + w_m tr(Gamma h)
  + sum_j xi_j (c_j - c_j*); a relaxation solves its stationarity, so each
  term's *gradient* is that term's contribution to the stationarity
  equations on the lengths (dS/dz_e) and on the links (U_e dS/dU_e);
* a *term record* is one term at one recorded point: its value and the
  Euclidean norm of its gradient on the coordinates the relaxation relaxes,
  summed over the declared edge classes as the residual is;
* the *trace* of a solve is the list of the term records at its starting
  point and at every step, recorded when the geometry declaration asks for
  it and empty otherwise.
"""
import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import recursion as R

from tests.drivers import _recursion_run_2026_09_23 as RUN

TERMS = ["regge", "holonomy", "matter"]
#: The two sums every record list ends with.
SUMS = ["constraints", "action"]


def _host(content=(1, 1, 1)):
    config = bp.default_config([1.0], [1.0], selected_contents=[content],
                               fiber_moments="r", fiber_pinning="power-sums",
                               tolerances=RUN.TOLERANCES)
    config["host_cell"] = RUN.HOST_CELLS[(0, 1, 3, 4)]
    config["held_sectors"] = R.held_sectors([[0, 1, 2, 3]], [1], 4)
    spacetime = bp.build_host(config["edge_squared"], config["host_cell"])
    # every hinge, so the Regge term is not structurally zero on the host,
    # which has no interior hinge
    declaration = bp.action_declaration(spacetime, 1.0, 1.0, "all")
    return config, spacetime, declaration


def _with_constraints(spacetime, declaration, mean_field):
    """The action with the fiber's power sums pinned (three constraints)
    and nonzero multipliers, so every term of the list is active."""
    action = cob.JointAction(spacetime, declaration)
    read = cob.BandFollower(mean_field).read(action.carrier_operator())
    declaration.covariance = list(read.covariance)
    declaration.moment_projector = list(
        sum(np.asarray(b.projector) for b in read.bands))
    declaration.moment_scale = 19.371
    constraints = []
    for order in (1, 2, 3):
        constraint = cob.SpectralMomentConstraint()
        constraint.order = order
        constraint.target = 0.1 * order
        constraint.multiplier = complex(0.3 * order, -0.1)
        constraints.append(constraint)
    declaration.moment_constraints = constraints
    return cob.JointAction(spacetime, declaration)


def test_the_terms_sum_to_the_stationarity_exactly():
    """Regge, holonomy, matter and each pinned constraint: the per-term
    length and link gradients sum to `length_stationarity` and
    `link_stationarity`, and the term values to the action's value."""
    config, spacetime, declaration = _host()
    mean_field = bp.mean_field_declaration((1, 1, 1), config, spacetime)
    action = _with_constraints(spacetime, declaration, mean_field)
    terms = action.term_gradients()
    assert [t.name for t in terms] == TERMS + ["constraint 1", "constraint 2",
                                               "constraint 3"]
    n = len(spacetime.get_edge_list().to_vector())
    lengths = sum(np.asarray(t.length_stationarity) for t in terms)
    links = sum(np.asarray(t.link_stationarity) for t in terms)
    assert all(len(t.length_stationarity) == n and
               len(t.link_stationarity) == n for t in terms)
    total_lengths = np.asarray(action.length_stationarity())
    total_links = np.asarray(action.link_stationarity())
    np.testing.assert_allclose(lengths, total_lengths, rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(total_lengths)))
    np.testing.assert_allclose(links, total_links, rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(total_links)))
    assert sum(t.value for t in terms) == pytest.approx(complex(action.value()),
                                                       rel=1e-12)
    # every term is active on this action
    assert all(np.max(np.abs(np.asarray(t.length_stationarity))) > 0
               for t in terms if t.name != "holonomy")
    assert np.max(np.abs(np.asarray(terms[1].link_stationarity))) > 0
    assert np.max(np.abs(np.asarray(terms[0].link_stationarity))) == 0
    assert terms[0].value == pytest.approx(complex(action.regge_term()))
    assert terms[1].value == pytest.approx(complex(action.holonomy_term()))
    assert terms[2].value == pytest.approx(complex(action.matter_term()))
    # every term as it stands in the action: its weight, its bare factor
    # and their product
    assert [t.label for t in terms[:3]] == [
        "(1/kappa) S_Regge", "beta S_hol", "w_m tr(Gamma h_1)"]
    assert terms[3].label == "xi_1 (c_1 - c_1*), c_1 = p_1(h_C / s)"
    assert terms[0].weight == 1.0 and terms[2].weight == 1.0
    for term in terms:
        if term.factored:
            assert term.value == pytest.approx(term.weight * term.bare,
                                               rel=1e-12, abs=1e-15)
    assert not terms[1].factored and terms[1].bare == terms[1].value
    assert terms[3].weight == complex(0.3, -0.1)
    assert terms[3].bare == pytest.approx(complex(action.moment_residuals()[0]))


def test_a_term_of_zero_weight_is_listed_with_zeros():
    _, spacetime, declaration = _host()
    declaration.gravitational_weight = 0.0
    declaration.matter_weight = 0.0
    terms = cob.JointAction(spacetime, declaration).term_gradients()
    assert [t.name for t in terms] == TERMS
    for name in ("regge", "matter"):
        term = next(t for t in terms if t.name == name)
        assert term.value == 0 and not np.any(np.asarray(
            term.length_stationarity)) and not np.any(np.asarray(
                term.link_stationarity))


def test_the_records_reduce_onto_the_relaxed_coordinates():
    """A term's recorded gradient norm is the norm of its gradient summed
    over the shared sheet classes, lengths and links, as the residual is;
    with the links held it is the length part alone."""
    config, spacetime, declaration = _host()
    mean_field = bp.mean_field_declaration((1, 1, 1), config, spacetime)
    action = _with_constraints(spacetime, declaration, mean_field)
    geometry = bp.share_sheet_geometry(bp.relaxation_declaration(config),
                                       spacetime)
    classes = list(geometry.edge_classes)
    orientations = list(geometry.edge_class_orientations)
    records = cob.action_term_records(action, geometry)
    terms = action.term_gradients()
    assert [r.name for r in records] == [t.name for t in terms] + SUMS
    count = max(classes) + 1
    for record, term in zip(records, terms):
        summed = np.zeros(2 * count, dtype=complex)
        for edge, (c, o) in enumerate(zip(classes, orientations)):
            summed[c] += term.length_stationarity[edge]
            summed[count + c] += o * term.link_stationarity[edge]
        assert record.gradient_norm == pytest.approx(np.linalg.norm(summed),
                                                     rel=1e-12)
        assert record.value == term.value
        assert (record.label, record.weight, record.bare, record.factored) == \
            (term.label, term.weight, term.bare, term.factored)
    # the sums: of the constraint terms, and of the whole action, each with
    # the norm of its summed gradient, which is the force norm for the whole
    constraints, whole = records[-2], records[-1]
    assert constraints.label == "sum_j xi_j (c_j - c_j*)" and whole.label == "S"
    assert constraints.value == pytest.approx(
        sum(t.value for t in terms if t.name.startswith("constraint")))
    assert whole.value == pytest.approx(complex(action.value()), rel=1e-12)
    summed = np.zeros(2 * count, dtype=complex)
    for term in terms:
        for edge, (c, o) in enumerate(zip(classes, orientations)):
            summed[c] += term.length_stationarity[edge]
            summed[count + c] += o * term.link_stationarity[edge]
    assert whole.gradient_norm == pytest.approx(np.linalg.norm(summed),
                                                rel=1e-12)
    geometry.relax_links = False
    for record, term in zip(cob.action_term_records(action, geometry), terms):
        summed = np.zeros(count, dtype=complex)
        for edge, c in enumerate(classes):
            summed[c] += term.length_stationarity[edge]
        assert record.gradient_norm == pytest.approx(np.linalg.norm(summed),
                                                     rel=1e-12)


def test_a_relaxation_records_the_terms_only_when_asked():
    # a relaxation moves its complex, so each run gets a fresh host
    config, spacetime, declaration = _host((0, 3, 0))
    geometry = bp.share_sheet_geometry(bp.relaxation_declaration(config),
                                       spacetime)
    geometry.maximum_iterations = 2
    geometry.held_sectors = []
    silent = cob.HolomorphicRelaxation(
        cob.JointAction(spacetime, declaration), geometry).solve()
    assert len(silent.initial_terms) == 0
    assert all(len(step.terms) == 0 for step in silent.steps)
    _, spacetime, declaration = _host((0, 3, 0))
    geometry.record_terms = True
    expected = cob.action_term_records(cob.JointAction(spacetime, declaration),
                                       geometry)
    traced = cob.HolomorphicRelaxation(
        cob.JointAction(spacetime, declaration), geometry).solve()
    assert [t.name for t in traced.initial_terms] == TERMS + SUMS
    assert len(traced.steps) >= 1
    for step in traced.steps:
        assert [t.name for t in step.terms] == TERMS + SUMS
    # the same steps either way: recording changes nothing
    assert [s.residual_norm for s in traced.steps] == \
        [s.residual_norm for s in silent.steps]
    # the norms are those of the terms at the starting point
    assert [r.gradient_norm for r in traced.initial_terms] == \
        [r.gradient_norm for r in expected]


def test_the_mean_field_solve_traces_its_terms_and_the_driver_prints_them():
    """With ``trace_terms`` in the config every iterate of a content's
    solve carries the terms, the record carries them, and the trace lines
    name every term with its value, gradient and the change since the
    previous iterate; without it the record carries none and prints
    nothing."""
    config, _, _ = _host((0, 3, 0))
    config["trace_terms"] = True
    config["mean_field_iterations"] = 2
    _, _, report = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    names = TERMS + ["constraint %d" % j for j in (1, 2, 3)] + SUMS
    assert all([t.name for t in step.terms] == names for step in report.steps)
    record = bp.relaxation_record(report)
    assert [t["name"] for t in record["trace"][0]["terms"]] == names
    lines = bp.term_trace_lines(record, "  ")
    # one header and one line per term and per sum, per iterate
    per_iterate = 1 + 3 + 1 + 3
    assert len(lines) == per_iterate * len(report.steps)
    assert lines[0].startswith("  iterate 0: S = ")
    assert "stationarity residual" in lines[0]
    assert lines[1].startswith("    (1/kappa) S_Regge = (1+0i) x (")
    assert lines[2].startswith("    beta S_hol = ") and "(beta = 1+0i)" in lines[2]
    assert lines[3].startswith("    w_m tr(Gamma h_1) = (1+0i) x (")
    assert lines[4].startswith("    sum_j xi_j (c_j - c_j*) = ")
    assert lines[5].startswith("      xi_1 (c_1 - c_1*), c_1 = p_1(h_C / s) = (")
    assert all("gradient" in line for line in lines[1:per_iterate])
    assert not any("[value" in line for line in lines[:per_iterate])
    if len(report.steps) > 1:
        second = lines[per_iterate:2 * per_iterate]
        assert second[0].startswith("  iterate 1: S = ")
        assert all("[value" in line and "improved" in line for line in second)
    config["trace_terms"] = False
    _, _, quiet = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    assert all(len(step.terms) == 0 for step in quiet.steps)
    assert bp.term_trace_lines(bp.relaxation_record(quiet), "  ") == []


def test_the_option_reaches_both_drivers_and_every_cell():
    assert not bp.build_parser().parse_args(["run"]).trace_terms
    assert bp.build_parser().parse_args(["run", "--trace-terms"]).trace_terms
    assert R.build_parser().parse_args(["run", "--trace-terms"]).trace_terms
    assert bp.default_config([1.0], [1.0])["trace_terms"] is False
    config = R.default_config(trace_terms=True)
    assert config["trace_terms"] is True
    geometry = bp.relaxation_declaration(dict(config, held_sectors=[]))
    assert geometry.record_terms
    assert not bp.relaxation_declaration(
        dict(R.default_config(), held_sectors=[])).record_terms
