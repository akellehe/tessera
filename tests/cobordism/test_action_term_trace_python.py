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
* a *step proposal* is one point of a drive (`tessera.drivers.cell_solve`)
  at which a step is formed: the starting point, every point an accepted
  update reached, and the end point once for each line search that
  accepted no trial there;
* the *trace* of a solve is the list of the term records at every step
  proposal, recorded when the geometry declaration asks for it and empty
  otherwise.
"""
import numpy as np
import pytest

from tessera import cobordism as cob
from tessera.drivers import baryon_poles as bp
from tessera.drivers import cell_solve as cs
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
    n = len(spacetime.getEdgeList().toVector())
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


def _level_relaxation(trace_terms):
    """The level relaxation of the host of (0134) as one shared base field,
    every hinge in the Regge sum and no sector held, with the term records
    at its starting point."""
    config, spacetime, declaration = _host((0, 3, 0))
    level = dict(config, kappa=1.0, beta=1.0, regge_hinges="all",
                 trace_terms=trace_terms)
    geometry = bp.share_sheet_geometry(
        bp.relaxation_declaration(dict(level, held_sectors=[])), spacetime)
    expected = cob.action_term_records(
        cob.JointAction(spacetime, declaration), geometry)
    return R.relax_level(spacetime, level, [], count=4), expected


def test_a_relaxation_records_the_terms_only_when_asked():
    """The geometric action alone has no stationary point near this host:
    twelve accepted updates each triple the squared lengths and lower the
    residual norm by the factor 3^(-1/2), from 12.67 to 9.0e-3, and at
    squared lengths 4.25e6 the Jacobian's rank at the run's rank tolerance
    falls from nine to three, the step moves no coordinate, and the drive
    stops. With ``trace_terms`` every one of the fifteen step proposals
    carries the terms; without it none does, and the drive is the same."""
    silent, _ = _level_relaxation(False)
    assert silent["term_trace"] == []
    traced, expected = _level_relaxation(True)
    assert len(traced["jacobian_ranks"]) == 15
    assert len(traced["term_trace"]) == 15
    for terms in traced["term_trace"]:
        assert [t["name"] for t in terms] == TERMS + SUMS
    # the same drive either way: recording changes nothing
    assert traced["residual_trace"] == silent["residual_trace"]
    assert traced["accepted_updates"] == silent["accepted_updates"] == 12
    assert traced["moves_committed"] == 0 and not traced["converged"]
    assert traced["stop_reason"] == cs.STOP_STATIONARY
    assert traced["jacobian_ranks"] == [9] * 12 + [3] * 3
    assert traced["initial_residual"] == pytest.approx(12.669871793382649,
                                                       rel=1e-12)
    assert traced["residual"] == pytest.approx(0.009002790053988682,
                                               rel=1e-9)
    ratios = np.divide(traced["residual_trace"][2:],
                       traced["residual_trace"][1:-1])
    np.testing.assert_allclose(ratios, 3.0 ** -0.5, rtol=1e-3)
    # the norms of the first proposal are those of the terms at the
    # starting point, and the whole action's is the residual norm there
    assert [r["gradient_norm"] for r in traced["term_trace"][0]] == \
        pytest.approx([r.gradient_norm for r in expected], rel=1e-13)
    assert traced["term_trace"][0][-1]["gradient_norm"] == pytest.approx(
        traced["initial_residual"], rel=1e-13)


def test_the_mean_field_solve_traces_its_terms_and_the_driver_prints_them():
    """With ``trace_terms`` in the config every step proposal of a content's
    solve carries the terms, the record carries them, and the trace lines
    name every term with its value, gradient and the change since the
    previous proposal; without it the record carries none and prints
    nothing. The solve of (0134, 030) with every power sum of its fiber
    pinned accepts seven updates and makes ten step proposals."""
    config, _, _ = _host((0, 3, 0))
    config["trace_terms"] = True
    _, _, report, drive = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    names = TERMS + ["constraint %d" % j for j in (1, 2, 3)] + SUMS
    steps = [update["measured"] for update in drive["objective"].updates]
    assert drive["accepted_updates"] == 7 and len(steps) == 10
    assert all([t.name for t in step.terms] == names for step in steps)
    # the read of the end point carries them as well
    assert [[t.name for t in step.terms] for step in report.steps] == [names]
    record = bp.relaxation_record(report, drive)
    assert len(record["trace"]) == 10
    assert all([t["name"] for t in entry["terms"]] == names
               for entry in record["trace"])
    lines = bp.term_trace_lines(record, "  ")
    # one header and one line per term and per sum, per step proposal
    per_iterate = 1 + 3 + 1 + 3
    assert len(lines) == per_iterate * len(steps)
    assert lines[0].startswith("  iterate 0: S = ")
    assert "stationarity residual" in lines[0]
    assert lines[1].startswith("    (1/kappa) S_Regge = (1+0i) x (")
    assert lines[2].startswith("    beta S_hol = ") and "(beta = 1+0i)" in lines[2]
    assert lines[3].startswith("    w_m tr(Gamma h_1) = (1+0i) x (")
    assert lines[4].startswith("    sum_j xi_j (c_j - c_j*) = ")
    assert lines[5].startswith("      xi_1 (c_1 - c_1*), c_1 = p_1(h_C / s) = (")
    assert all("gradient" in line for line in lines[1:per_iterate])
    assert not any("[value" in line for line in lines[:per_iterate])
    second = lines[per_iterate:2 * per_iterate]
    assert second[0].startswith("  iterate 1: S = ")
    assert all("[value" in line and "improved" in line for line in second)
    config["trace_terms"] = False
    _, _, quiet, quiet_drive = bp.relax_content((0, 3, 0), 1.0, 1.0, config)
    assert all(len(update["measured"].terms) == 0
               for update in quiet_drive["objective"].updates)
    assert all(len(step.terms) == 0 for step in quiet.steps)
    # the same drive either way: recording changes nothing
    assert quiet_drive["trace"] == drive["trace"]
    assert bp.term_trace_lines(bp.relaxation_record(quiet, quiet_drive),
                               "  ") == []


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
