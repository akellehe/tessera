# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Sign audit of the diagonal DEC Hodge star (observables::DualVolumeSigns, #605).

The *diagonal Discrete Exterior Calculus Hodge star* assigns each k-simplex
``sigma`` the single scalar ratio ``|*sigma| / |sigma|``: its signed circumcentric
dual cell content (``Simplex.dual_volume``) over its own signed content
(``Simplex.volume``). ``DualVolumeSigns`` reports the sign statistics of that
ratio; it measures only and changes no geometry.

Every assertion here is against an independent Python oracle that recomputes the
audit straight from the ``Simplex`` bindings, so the observable is never checked
against itself.

Two properties matter beyond raw agreement:

  * **Orphan exclusion.** A Pachner move can strand a lazily-materialised
    sub-face with no surviving top coface. Such a simplex is not part of the
    complex, and ``Simplex.has_top_coface`` is the documented filter. The oracle
    applies the same filter.
  * **Signature partition.** A negative ratio has two unrelated causes -- a
    circumcenter falling outside its simplex (Riemannian mesh degradation) versus
    a timelike circumcenter displacement (expected in Lorentzian signature). The
    observable partitions every count by all-spacelike versus mixed-signature
    cells, and that partition must be exact.

The skeleton is materialised by constructing a ``ReggeSolver``, never from
Python: the ``get_facets``/``get_cofaces`` bindings return copies, so building the
skeleton Python-side corrupts the coface lists and ``dual_volume`` then sees only
part of the star (the pitfall recorded on the mesh bindings).
"""

import unittest

import tessera
import cmath

TOLERANCE = 1e-12


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _from_simplices(num_vertices, simplices):
    """A 4D Lorentzian spacetime carrying exactly `simplices`, skeleton built."""
    sig = tessera.Signature(4, tessera.Lorentzian)
    metric = tessera.Metric(True, sig)
    st = tessera.Spacetime(metric, tessera.HERMITIAN_WEIGHTED, 1.0, 1.0,
                           tessera.PREFERRED, tessera.Toroid())
    verts = [st.create_vertex(i) for i in range(num_vertices)]
    for simplex in simplices:
        st.create_simplex([verts[i] for i in simplex])
    # Materialize the full skeleton in C++ so cofaces stay intact.
    tessera.ReggeSolver(st, tessera.MatterConfiguration())
    return st


def _pentatope():
    """A single 4-simplex -- the Delta^4 seed every emergent host grows from."""
    return _from_simplices(5, [(0, 1, 2, 3, 4)])


def _two_pentatopes():
    """Two 4-simplices glued on the shared tetrahedron (0,1,2,3)."""
    return _from_simplices(6, [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5)])


def _set_all_squared_lengths(st, value):
    for e in st.get_edge_list().to_vector():
        e.set_length(cmath.sqrt(complex(value)))


def _edge(st, a, b):
    for e in st.get_edge_list().to_vector():
        if {e.get_source().get_id(), e.get_target().get_id()} == {a, b}:
            return e
    raise KeyError((a, b))


# --------------------------------------------------------------------------- #
# Independent oracle
# --------------------------------------------------------------------------- #
def _oracle(st, tolerance=TOLERANCE):
    """Recompute the audit from the Simplex bindings, keyed by dimension.

    Mirrors the documented contract: orphans excluded via hasTopCoface, cells
    whose own content is within `tolerance` of zero counted as degenerate and
    left out of both the negative tally and the ratio statistics.
    """
    per_dimension = {}
    for s in st.get_simplices():
        if not s.has_top_coface():
            continue
        n_vertices = len(s.get_vertices())
        if n_vertices == 0:
            continue
        dimension = n_vertices - 1
        entry = per_dimension.setdefault(dimension, {
            "n_simplices": 0, "n_negative_dual_volume": 0,
            "n_degenerate_volume": 0, "n_circumcenter_outside": 0,
            "n_negative_circumradius": 0, "n_negative_star": 0,
            "n_all_spacelike": 0, "n_negative_star_all_spacelike": 0,
            "n_mixed_signature": 0, "n_negative_star_mixed_signature": 0,
            "ratios": [],
        })
        entry["n_simplices"] += 1

        all_spacelike = all(e.is_spacelike() for e in s.get_edges())
        entry["n_all_spacelike" if all_spacelike else "n_mixed_signature"] += 1

        # Complex-typed reads; the sign tallies are defined on the real part,
        # exactly as the C++ observable reads them (#640).
        dual_volume = complex(s.dual_volume())
        if dual_volume.real < 0.0:
            entry["n_negative_dual_volume"] += 1

        barycentric = [complex(b) for b in s.circumcenter_barycentric()]
        if barycentric and min(b.real for b in barycentric) < 0.0:
            entry["n_circumcenter_outside"] += 1

        if complex(s.circumradius_squared()).real < 0.0:
            entry["n_negative_circumradius"] += 1

        volume = s.volume()
        if abs(volume) <= tolerance:
            entry["n_degenerate_volume"] += 1
            continue

        ratio = (dual_volume / complex(volume)).real
        entry["ratios"].append(ratio)
        if ratio < 0.0:
            entry["n_negative_star"] += 1
            key = ("n_negative_star_all_spacelike" if all_spacelike
                   else "n_negative_star_mixed_signature")
            entry[key] += 1
    return per_dimension


class TestDualVolumeSigns(unittest.TestCase):
    """DualVolumeSigns reproduces an independent oracle and partitions exactly."""

    def _assert_matches_oracle(self, st):
        report = tessera.DualVolumeSigns(TOLERANCE).analyze(st)
        expected = _oracle(st)

        self.assertEqual(
            [d.dimension for d in report.dimensions], sorted(expected),
            "dimensions must be reported in increasing order, one per dimension "
            "present in the complex",
        )

        for entry in report.dimensions:
            want = expected[entry.dimension]
            for field in ("n_simplices", "n_negative_dual_volume",
                          "n_degenerate_volume", "n_circumcenter_outside",
                          "n_negative_circumradius", "n_negative_star",
                          "n_all_spacelike", "n_negative_star_all_spacelike",
                          "n_mixed_signature",
                          "n_negative_star_mixed_signature"):
                self.assertEqual(
                    getattr(entry, field), want[field],
                    f"dimension {entry.dimension}: {field} disagrees with oracle",
                )

            ratios = want["ratios"]
            if ratios:
                self.assertAlmostEqual(entry.min_star_ratio, min(ratios), places=12)
                self.assertAlmostEqual(entry.max_star_ratio, max(ratios), places=12)
                self.assertAlmostEqual(
                    entry.mean_star_ratio, sum(ratios) / len(ratios), places=12)
        return report

    def test_matches_oracle_on_a_pentatope(self):
        """Single Delta^4, uniform spacelike edges."""
        st = _pentatope()
        _set_all_squared_lengths(st, 1.0)
        report = self._assert_matches_oracle(st)
        self.assertGreater(report.n_simplices, 0,
                           "a materialised pentatope must audit some simplices")

    def test_matches_oracle_with_a_timelike_edge(self):
        """One timelike edge -- exercises the mixed-signature branch."""
        st = _two_pentatopes()
        _set_all_squared_lengths(st, 1.0)
        _edge(st, 0, 1).set_length(cmath.sqrt(complex(-1.0)))
        self._assert_matches_oracle(st)

    def test_signature_partition_is_exact(self):
        """All-spacelike and mixed-signature counts partition every dimension."""
        st = _two_pentatopes()
        _set_all_squared_lengths(st, 1.0)
        _edge(st, 0, 1).set_length(cmath.sqrt(complex(-1.0)))
        _edge(st, 2, 3).set_length(cmath.sqrt(complex(-1.0)))

        for entry in tessera.DualVolumeSigns().analyze(st).dimensions:
            self.assertEqual(
                entry.n_all_spacelike + entry.n_mixed_signature,
                entry.n_simplices,
                f"dimension {entry.dimension}: signature classes must partition "
                "the audited simplices",
            )
            self.assertEqual(
                entry.n_negative_star_all_spacelike
                + entry.n_negative_star_mixed_signature,
                entry.n_negative_star,
                f"dimension {entry.dimension}: the negative-star breakout must "
                "sum to the total",
            )

    def test_timelike_edge_moves_cells_to_mixed_signature(self):
        """Making an edge timelike must reclassify the cells that contain it."""
        st = _two_pentatopes()
        _set_all_squared_lengths(st, 1.0)
        all_spacelike = tessera.DualVolumeSigns().analyze(st)
        self.assertTrue(
            all(entry.n_mixed_signature == 0 for entry in all_spacelike.dimensions),
            "a uniformly spacelike complex has no mixed-signature cells",
        )

        _edge(st, 0, 1).set_length(cmath.sqrt(complex(-1.0)))
        mixed = tessera.DualVolumeSigns().analyze(st)
        self.assertGreater(
            sum(entry.n_mixed_signature for entry in mixed.dimensions), 0,
            "a timelike edge must reclassify every cell containing it",
        )

    def test_headline_is_the_negative_star_fraction(self):
        """compute() equals n_negative_star / n_simplices over the whole report."""
        st = _two_pentatopes()
        _set_all_squared_lengths(st, 1.0)
        _edge(st, 0, 1).set_length(cmath.sqrt(complex(-1.0)))

        observable = tessera.DualVolumeSigns()
        report = observable.analyze(st)
        expected = (report.n_negative_star / report.n_simplices
                    if report.n_simplices else 0.0)
        self.assertAlmostEqual(observable.compute(st), expected, places=12)
        self.assertGreaterEqual(observable.compute(st), 0.0)
        self.assertLessEqual(observable.compute(st), 1.0)

    def test_totals_sum_over_dimensions(self):
        """Report totals are the per-dimension sums, degenerate cells excluded."""
        st = _two_pentatopes()
        _set_all_squared_lengths(st, 1.0)
        report = tessera.DualVolumeSigns().analyze(st)
        self.assertEqual(
            report.n_simplices,
            sum(entry.n_simplices - entry.n_degenerate_volume
                for entry in report.dimensions),
        )
        self.assertEqual(
            report.n_negative_star,
            sum(entry.n_negative_star for entry in report.dimensions),
        )


if __name__ == "__main__":
    unittest.main()
