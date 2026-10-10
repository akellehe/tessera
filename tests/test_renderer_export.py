"""GraphML/DOT export carries the full complex l^2 + phase and classifies
via the canonical Edge helpers (#581 scope item 5).

The exporters used the superseded sign-of-Re classifier (contradicting the
same file's PNG path, which already used ``Edge::isTimelike()``) and had no
key for ``Im l^2`` or the U(1) phase at all.  Now: ``squared_length`` stays
Re (compatibility), ``squared_length_im`` and ``phase`` are new keys, and
``timelike`` is ``Edge::isTimelike()``.
"""

import math
import re
import xml.etree.ElementTree as ET

import pytest
import cmath

try:
    import tessera
    _IMPORT_OK = True
except Exception:  # pragma: no cover
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="tessera not built")


def _host():
    """A triangle carrying one timelike edge, one Im l^2 != 0 edge, and a
    genuinely COMPLEX connection phase on the spacelike edge, so the exporter's
    non-compact component is exercised too."""
    sig = tessera.Signature(2, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.CDT, 1.0, 1.0,
                           tessera.PREFERRED, tessera.SolidSimplex(2))
    st.build()
    values = {(0, 1): (-2.0 + 0.0j, 0.0),              # timelike
              (0, 2): (1.0 + 0.25j, 0.0),              # analytically continued
              (1, 2): (1.5 + 0.0j, complex(0.7, -0.3))}  # spacelike, twisted
    for e in st.get_edge_list().to_vector():
        a, b = e.get_source().get_id(), e.get_target().get_id()
        sq, ph = values[(min(a, b), max(a, b))]
        e.set_length(cmath.sqrt(complex(sq)))
        e.set_phase(ph)
    return st, values


def _canonical_timelike(st):
    out = {}
    for e in st.get_edge_list().to_vector():
        a, b = e.get_source().get_id(), e.get_target().get_id()
        out[(min(a, b), max(a, b))] = e.is_timelike()
    return out


def test_graphml_export_roundtrip(tmp_path):
    st, values = _host()
    canon = _canonical_timelike(st)
    path = str(tmp_path / "host.graphml")
    st.save(path)

    tree = ET.parse(path)
    ns = {"g": tree.getroot().tag.split("}")[0].strip("{")}
    # key declarations exist for the new attributes
    key_names = {k.get("attr.name") for k in tree.getroot().findall("g:key", ns)}
    assert {"squared_length", "squared_length_im", "phase", "phase_im",
            "timelike"} <= key_names

    seen = {}
    for edge in tree.getroot().find("g:graph", ns).findall("g:edge", ns):
        a, b = int(edge.get("source")), int(edge.get("target"))
        data = {d.get("key"): d.text for d in edge.findall("g:data", ns)}
        seen[(min(a, b), max(a, b))] = data
    assert set(seen) == set(values)

    for k, (sq, ph) in values.items():
        d = seen[k]
        assert math.isclose(float(d["sq_length"]), sq.real, abs_tol=1e-12)
        assert math.isclose(float(d["sq_length_im"]), sq.imag, abs_tol=1e-12)
        assert math.isclose(float(d["phase"]), complex(ph).real, abs_tol=1e-12)
        assert math.isclose(float(d["phase_im"]), complex(ph).imag,
                            abs_tol=1e-12)
        assert (d["timelike"] == "true") == canon[k], (
            f"edge {k}: exported timelike={d['timelike']} disagrees with "
            f"Edge.isTimelike()={canon[k]}")
    # The Im-carrying edge (l^2 = 1 + 0.25i) has arg(l^2) ~ 0.245 rad -- a
    # generic argument, so it is MIXED: no definite causal character, and in
    # particular NOT timelike (#870). It read as timelike only while any
    # nonzero Im(l) counted as such.
    #
    # NOTE the export carries a BOOLEAN `timelike`, which cannot distinguish
    # mixed from spacelike now that the classification has five cases. That is
    # a fidelity gap in the GraphML schema rather than a defect in the
    # classifier; the assertion above still pins export against the live
    # classifier, so the two cannot silently drift.
    assert seen[(0, 2)]["timelike"] == "false"


def test_dot_export_roundtrip(tmp_path):
    st, values = _host()
    canon = _canonical_timelike(st)
    path = str(tmp_path / "host.dot")
    st.save(path)

    # The connection phase is complex, so it exports as a real/imag pair
    # exactly as the squared length does.
    pat = re.compile(
        r"^\s*(\d+)\s*--\s*(\d+)\s*\[squared_length=([-\d.e+]+), "
        r"squared_length_im=([-\d.e+]+), phase=([-\d.e+]+), "
        r"phase_im=([-\d.e+]+), timelike=(true|false)", re.M)
    with open(path) as f:
        text = f.read()
    seen = {}
    for m in pat.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        seen[(min(a, b), max(a, b))] = m
    assert set(seen) == set(values), text

    for k, (sq, ph) in values.items():
        m = seen[k]
        assert math.isclose(float(m.group(3)), sq.real, abs_tol=1e-12)
        assert math.isclose(float(m.group(4)), sq.imag, abs_tol=1e-12)
        assert math.isclose(float(m.group(5)), complex(ph).real, abs_tol=1e-12)
        assert math.isclose(float(m.group(6)), complex(ph).imag, abs_tol=1e-12)
        assert (m.group(7) == "true") == canon[k]


if __name__ == "__main__":
    import unittest
    unittest.main()
