# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The stationarity residual of the joint action on a truncated-series path.

`cobordism.HolomorphicRelaxation` poses the stationarity system R(x) = 0 of
the joint action (`cobordism.JointAction`) in the coordinates x it relaxes:
one squared length z per edge class, one Maurer-Cartan link increment delta
per edge class (a link U of the class is moved to U exp(+-delta), the sign
the member's orientation), and one multiplier xi per spectral constraint.
`SeriesStationarity` evaluates R on a path x_0 + s(t) through the system's
current point x_0, with s(t) a truncated power series in one parameter t
(`tessera.numerics.TruncatedSeries`, order p at most ten) and s(0) = 0: it
returns the Taylor coefficients of t -> R(x_0 + s(t)) to order p.

Every term is propagated in series arithmetic from its closed form, the one
the library evaluates in numbers:

* the primal Regge term on the sheets the action carries at x_0: the hinge
  contents sqrt(det G) / (d - 2)!, the dihedral angles
  theta = 2 pi k + eps arccos(-C_ij / (sigma sqrt(C_ii) sqrt(C_jj))) from the
  cofactors C of each cell's Cayley-Menger matrix, and their derivatives in
  the squared lengths, every root and inverse cosine a series whose order-0
  value is the one on the declared sheet;
* the Villain term: the face holonomies are Laurent monomials in the links
  and the per-face derivative is D phi(F) = -beta_V DW_M(F) / W_M(F) with
  W_M(F) = 1 + sum_{m=1..M} c_m (F^m + F^-m), a Laurent polynomial of the
  declared order M;
* the matter term and the spectral constraints: the carrier operator is the
  Whitney pencil's operator on geometric images, L = Q h M with
  h = M A P B + C M' D Q built from the dressed Whitney mass matrices of
  three adjacent degrees (M_-, M, M' with P = M_-^-1 and Q = M^-1) and the
  twisted boundary maps, and tr(A dL / dx_a) is contracted in closed form
  against the derivative of each mass matrix and each link factor;
* the self-consistent state: with a mean-field declaration, the covariance
  and the pinned fiber's projectors are rebuilt along the path from the
  series Riesz projector of every occupied band
  (`TruncatedSeriesMatrix.rieszProjector`), the band being the group of
  eigenvalues the band read holds at x_0.

No difference quotient and no contour integral enters. A sum or a product of
series is formed here on the coefficient arrays (the Cauchy product); every
other operation (quotient, square root on a branch, exponential, inverse
cosine on a branch, matrix inverse, determinant, Riesz projector, reversion)
is `tessera.numerics`'.

`step(order, solve)` is the step of order p: s(1) of the reversion
(`tessera.numerics.revert`) of the residual about x_0 with the caller's
linear solve; order one is Newton's step.
"""
import cmath
import itertools
import math

import numpy as np

from tessera import chainhodge as ch
from tessera import cobordism as cob
from tessera import mesh
from tessera import numerics as nm

#: The largest turn about its branch point a root may make in one step of
#: the Regge continuation, the largest distance an angle may move, and the
#: shortest step the continuation refines to: the constants of
#: `JointAction`'s own walk of the Regge sheets, which this module repeats to
#: read the sheets the action carries.
REGGE_MAXIMUM_ROOT_TURN = math.pi / 4.0
REGGE_MAXIMUM_ANGLE_STEP = 0.25
REGGE_SHORTEST_STEP = 1.0 / (1 << 30)


# ------------------------------------------------------------------- series
#
# A series-valued array is a complex numpy array whose leading axis is the
# order: a[k] holds the coefficients of t^k.


def _constant(value, order):
    """The constant series of ``value`` (a number or an array)."""
    value = np.asarray(value, dtype=complex)
    out = np.zeros((order + 1,) + value.shape, dtype=complex)
    out[0] = value
    return out


def _mul(a, b):
    """The Cauchy product (ab)_k = sum_j a_j b_(k-j), entry by entry."""
    order = a.shape[0] - 1
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=complex)
    for k in range(order + 1):
        for j in range(k + 1):
            out[k] += a[j] * b[k - j]
    return out


def _matmul(*factors):
    """The Cauchy product of matrix series, (AB)_k = sum_j A_j B_(k-j), of
    the factors from left to right."""
    out = factors[0]
    order = out.shape[0] - 1
    for b in factors[1:]:
        product = np.zeros((order + 1, out.shape[1], b.shape[2]),
                           dtype=complex)
        for k in range(order + 1):
            for j in range(k + 1):
                product[k] += out[j] @ b[k - j]
        out = product
    return out


def _transpose(a):
    return np.swapaxes(a, 1, 2)


def _scale(scalar, array):
    """A scalar series times an array series."""
    shape = (scalar.shape[0],) + (1,) * (array.ndim - 1)
    return _mul(scalar.reshape(shape), array)


def _trace(a):
    return np.trace(a, axis1=1, axis2=2)


def _series(a):
    return nm.TruncatedSeries([complex(c) for c in a])


def _array(series):
    return np.array(series.coefficients, dtype=complex)


def _divide(a, b):
    return _array(_series(a) / _series(b))


def _sqrt(a, branch):
    return _array(nm.sqrt(_series(a), complex(branch)))


def _exp(a):
    return _array(nm.exp(_series(a)))


def _acos(a, branch):
    return _array(nm.acos(_series(a), complex(branch)))


def _matrix(a):
    return nm.TruncatedSeriesMatrix([np.array(c) for c in a])


def _inverse(a):
    return np.array(_matrix(a).inverse().coefficients, dtype=complex)


def _determinant(a):
    return _array(_matrix(a).determinant())


def _minors(matrices, rows, columns, order):
    """The determinants of N minors of size k, as series of shape
    (order + 1, N): minor n has the entry ``matrices[r][rows[n][r],
    columns[n][c]]`` in row r and column c, so that a different source may
    supply each row (the row-replacement form of a determinant's derivative).
    The Leibniz sum over the permutations of the columns; k = 0 gives one."""
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)
    count, size = rows.shape
    if size == 0:
        return _constant(np.ones(count), order)
    total = np.zeros((order + 1, count), dtype=complex)
    for permutation in itertools.permutations(range(size)):
        inversions = sum(1 for i in range(size) for j in range(i + 1, size)
                         if permutation[i] > permutation[j])
        term = None
        for r in range(size):
            factor = matrices[r][:, rows[:, r], columns[:, permutation[r]]]
            term = factor if term is None else _mul(term, factor)
        total += term if inversions % 2 == 0 else -term
    return total


# ----------------------------------------------------------------- numbers


def _positive_zero(z):
    """A complex number whose zero imaginary part is +0, the side of a cut
    the library's principal roots and inverse cosines take."""
    z = complex(z)
    return complex(z.real, 0.0) if z.imag == 0.0 else z


def _principal_sqrt(z):
    return cmath.sqrt(_positive_zero(z))


def _principal_acos(r):
    return cmath.acos(_positive_zero(r))


def _pinned_cosine(cij, root_product):
    return _positive_zero(-cij / root_product)


def _cofactors(matrix):
    """The cofactor matrix of a square complex matrix."""
    n = matrix.shape[0]
    out = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            out[i, j] = (-1) ** (i + j) * (np.linalg.det(minor)
                                           if n > 1 else 1.0)
    return out


def _subsets(top, size):
    """The ``size``-subsets of 0..top in lexicographic order."""
    return [list(c) for c in itertools.combinations(range(top + 1), size)]


def _local_edge_index(d, a, b):
    return sum(d - i for i in range(a)) + (b - a - 1)


def _pair(a, b):
    return (a, b) if a <= b else (b, a)


class SeriesStationarity:
    """The stationarity residual of a `HolomorphicRelaxation` system on a
    truncated-series path through the system's current point.

    ``action`` is the `JointAction` at the point, as
    `HolomorphicRelaxation.action` carries it (its constraints with their
    targets and multipliers, its covariance and fiber projectors);
    ``declaration`` is the system's `HolomorphicRelaxationDeclaration` (which
    blocks are relaxed, the edge classes and their orientations).

    With ``mean_field``, a `SelfConsistentMeanFieldDeclaration`, the system
    is the self-consistent one of `SelfConsistentMeanField.joint_system`: the
    covariance, the pinned fiber's projector and the pinned bands' projectors
    are rebuilt at every point of the path from the bands of the band
    operator, the multipliers are variables whenever a fiber constraint is
    pinned, and ``follower`` is the `BandFollower` whose ``read`` of the band
    operator at the point names the bands (a new follower of ``mean_field``
    when none is given, which chooses them by the declared order, as a
    system built at the point does).

    ``regge_start`` is the starting squared length of every edge, in
    `getEdgeList()` order, that the action's continued Regge sheets were
    declared at; the declaration's own `regge_start_squared_lengths` when it
    has them, and otherwise the squared lengths at the point, which is the
    start of an action constructed there.
    """

    def __init__(self, action, declaration, mean_field=None, follower=None,
                 regge_start=None):
        self._action = action
        self._declaration = action.declaration
        self._spacetime = action.spacetime
        self._mean_field = mean_field
        self._read_geometry()
        self._read_layout(declaration)
        self._read_complex()
        self._read_state(follower)
        self._read_regge(regge_start)
        self._read_villain()

    # ------------------------------------------------------------ the point

    def _read_geometry(self):
        """The stored edges with their squared lengths and links, and the
        chain complex with its canonical cells."""
        edges = self._spacetime.getEdgeList().toVector()
        self._stored = [(int(e.getSource().getId()),
                         int(e.getTarget().getId())) for e in edges]
        self._z0_stored = [complex(e.getLength()) ** 2 for e in edges]
        self._phase0_stored = [complex(e.getPhase()) for e in edges]
        self._complex = cob.ChainComplex.fromSpacetime(self._spacetime)
        self._dimension = int(self._complex.dimension())
        self._cells = [[tuple(int(v) for v in cell)
                        for cell in self._complex.kSimplexVertices(k)]
                       for k in range(self._dimension + 1)]
        self._edge_index = {cell: index
                            for index, cell in enumerate(self._cells[1])}
        #: for every stored edge, its canonical index and the sign of its
        #: stored orientation relative to the ascending one
        self._canonical = [self._edge_index[_pair(a, b)]
                           for a, b in self._stored]
        self._stored_sign = [1.0 if a < b else -1.0 for a, b in self._stored]
        count = len(self._cells[1])
        self._z0 = np.zeros(count, dtype=complex)
        self._link0 = np.ones(count, dtype=complex)
        for index, (a, b) in enumerate(self._stored):
            canonical = self._canonical[index]
            self._z0[canonical] = self._z0_stored[index]
            link = cmath.exp(1j * self._phase0_stored[index])
            self._link0[canonical] = link if a < b else 1.0 / link

    def _read_layout(self, declaration):
        """The variables and equations of the system: the edge classes, the
        relaxed blocks and their offsets, as `HolomorphicRelaxation` lays
        them out."""
        edges = len(self._stored)
        classes = [int(c) for c in declaration.edge_classes]
        orientations = [int(o) for o in declaration.edge_class_orientations]
        if not classes:
            self._members = [[(edge, 1)] for edge in range(edges)]
        else:
            if len(classes) != edges:
                raise ValueError(
                    "SeriesStationarity: %d edge classes were declared for "
                    "%d edges" % (len(classes), edges))
            self._members = [[] for _ in range(max(classes) + 1)]
            for edge, index in enumerate(classes):
                self._members[index].append(
                    (edge, orientations[edge] if orientations else 1))
        self._constraints = list(self._declaration.moment_constraints)
        self._lengths = bool(declaration.relax_lengths)
        self._links = bool(declaration.relax_links)
        pinned = (self._mean_field is not None
                  and int(self._mean_field.fiber_moments) > 0)
        self._multipliers = ((bool(declaration.relax_multipliers) or pinned)
                             and len(self._constraints) > 0)
        count = len(self._members)
        self._length_offset = 0
        self._link_offset = count if self._lengths else 0
        self._multiplier_offset = self._link_offset + (count if self._links
                                                       else 0)
        self._count = self._multiplier_offset + (
            len(self._constraints) if self._multipliers else 0)

    def _read_complex(self):
        """What the Whitney pencil is assembled from: the reference-oriented
        complex with its boundary maps and top cells, the stored orientation
        signs, the base vertex of every cell, and the volume of every top
        cell on its branch at the point."""
        d = self._dimension
        k = int(self._declaration.carrier_degree)
        self._degree = k
        self._carrier_needed = bool(
            (self._declaration.matter_weight != 0.0
             and (self._mean_field is not None
                  or len(self._declaration.covariance) > 0))
            or self._constraints) and 0 <= k <= d
        if not self._carrier_needed:
            return
        if (self._declaration.metric_source
                != cob.HodgeMetricSource.WhitneyPencil):
            raise NotImplementedError(
                "SeriesStationarity: the carrier operator is propagated for "
                "the Whitney pencil; the action declares another metric "
                "source")
        reference = ch.WhitneyMass.complexOf(self._spacetime)
        for degree in range(d + 1):
            cells = [tuple(int(v) for v in cell)
                     for cell in reference.kSimplexVertices(degree)]
            if cells != self._cells[degree]:
                raise ValueError(
                    "SeriesStationarity: the complex's cell order differs "
                    "from the canonical order at degree %d" % degree)
        self._signs = [np.array([float(s) for s in signs])
                       for signs in self._complex.orientationSigns()]
        self._boundary = {}
        for degree in (k, k + 1):
            if 1 <= degree <= d:
                matrix = np.zeros((len(self._cells[degree - 1]),
                                   len(self._cells[degree])))
                for row, column, value in reference.boundaryEntries(degree):
                    matrix[int(row), int(column)] = float(value)
                self._boundary[degree] = matrix
        self._base = [[cell[0] for cell in cells] for cells in self._cells]
        self._tops = [tuple(int(v) for v in top)
                      for top in reference.orientedTopSimplices()]
        self._cell_index = [{cell: index for index, cell in enumerate(cells)}
                            for cells in self._cells]
        self._top_edges = []
        self._top_volume = []
        for top in self._tops:
            edges = [self._edge_index[(top[a], top[b])]
                     for a in range(d + 1) for b in range(a + 1, d + 1)]
            self._top_edges.append(edges)
            gram = self._gram(_constant(self._z0[edges], 0))[0]
            volume, _ = ch.WhitneyMass.volumeOnBranch(
                gram, ch.Branch.Continuation)
            self._top_volume.append(complex(volume))
        self._degrees = [j for j in (k - 1, k, k + 1) if 0 <= j <= d]
        self._faces = {j: _subsets(d, j + 1) for j in self._degrees}
        self._minor_tables = {j: self._minor_table(j) for j in self._degrees}
        self._link_tables = {}

    def _read_state(self, follower):
        """The carried state at the point: the declared covariance and
        projectors, or, for the self-consistent system, the bands the
        rebuild continues."""
        self._bands = None
        if not self._carrier_needed:
            return
        order = len(self._cells[self._degree])
        declared = self._declaration

        def square(flat):
            flat = np.asarray(flat, dtype=complex)
            return flat.reshape(order, order) if flat.size else None

        self._covariance0 = square(declared.covariance)
        self._projector0 = square(declared.moment_projector)
        self._band_projectors0 = [square(p)
                                  for p in declared.moment_band_projectors]
        if self._mean_field is None:
            return
        mean_field = self._mean_field
        self._symmetry = [np.asarray(g, dtype=complex).reshape(order, order)
                          for g in mean_field.band_symmetry] \
            if mean_field.covariance_rule == cob.CovarianceRule.BandFilling \
            else []
        if follower is None:
            follower = cob.BandFollower(mean_field)
        operator = self._band_operator(
            _constant(np.asarray(self._action.carrier_operator(),
                                 dtype=complex).reshape(order, order), 0))[0]
        read = follower.read(list(operator.reshape(-1)))
        self._band_eigenvalues = np.asarray(read.eigenvalues, dtype=complex)
        self._bands = [{"modes": {int(m) for m in band.modes},
                        "weight": (float(band.occupation) / int(band.rank)
                                   if int(band.rank) > 0 else 0.0)}
                       for band in read.bands]
        self._fiber = int(mean_field.fiber_moments) > 0
        self._pinned_bands = (
            min(len(self._bands), int(mean_field.fiber_moments))
            if self._fiber and mean_field.fiber_constraint_form
            == cob.FiberConstraintForm.BandEigenvalues else 0)

    def _read_villain(self):
        """The Villain weight's coefficients and matched weight, and the
        stored complex's face incidences."""
        declared = self._declaration
        self._villain = None
        if not (declared.holonomy_weight > 0.0) or self._dimension < 2:
            return
        character = cob.VillainCharacter(float(declared.holonomy_weight),
                                         int(declared.villain_order))
        self._villain = {
            "weight": float(character.matched_weight),
            "coefficients": [float(c) for c in character.coefficients],
            "incidence": [(int(row), int(column), int(value)) for
                          row, column, value in
                          self._complex.boundaryEntries(2)],
            "faces": len(self._cells[2]),
        }

    # ------------------------------------------------------- the Regge sheets

    def _hinges(self):
        """The hinges of the primal Regge sum under the declared rule, as
        sorted vertex tuples: every (d-2)-cell under `ReggeHinges.All`, and
        under `ReggeHinges.Interior` those whose link closes (every
        (d-1)-face containing the hinge is shared by exactly two top
        cells)."""
        d = self._dimension
        if d < 2:
            return []
        hinges = list(self._cells[d - 2])
        if self._declaration.regge_hinges == cob.ReggeHinges.All:
            return hinges
        cofaces = {face: 0 for face in self._cells[d - 1]}
        for top in self._cells[d]:
            for face in itertools.combinations(top, d):
                cofaces[face] += 1
        kept = []
        for hinge in hinges:
            around = [face for face in self._cells[d - 1]
                      if set(hinge) <= set(face)]
            if around and all(cofaces[face] == 2 for face in around):
                kept.append(hinge)
        return kept

    def _read_regge(self, regge_start):
        """The hinges of the primal Regge sum with the sheet of every
        dihedral angle and the sign of every content root at the point, as
        `JointAction` reads them: principal under `ReggeBranch.Principal`,
        and under `ReggeBranch.Continued` continued from the Euclidean
        reference Re z_start through z_start to the point along straight
        segments."""
        declared = self._declaration
        self._regge = None
        # the Regge term enters the length equations only
        if declared.gravitational_weight == 0.0 or not self._lengths:
            return
        if declared.regge_form != cob.ReggeForm.Primal:
            raise NotImplementedError(
                "SeriesStationarity: the Regge term is propagated in its "
                "primal form; the action declares the dual form")
        d = self._dimension
        hinges = []
        for hinge in self._hinges():
            cells = []
            for top in self._cells[d]:
                if not set(hinge) <= set(top):
                    continue
                opposite = [position + 1 for position, vertex
                            in enumerate(top) if vertex not in hinge]
                if len(opposite) == 2:
                    cells.append({"cell": top, "pair": tuple(opposite),
                                  "root_sign": 1, "orientation": 1,
                                  "branch_index": 0})
            hinges.append({"hinge": hinge, "cells": cells, "content_sign": 1})
        self._regge = hinges
        now = {_pair(a, b): z for (a, b), z in zip(self._stored,
                                                  self._z0_stored)}
        if declared.regge_branch != cob.ReggeBranch.Principal and hinges:
            self._continue_regge(hinges, now, regge_start)
        # the order-0 value of every root and angle on its sheet, from the
        # cofactors at the point
        for hinge in hinges:
            for entry in hinge["cells"]:
                cofactors = _cofactors(self._cayley_menger(
                    entry["cell"], now, now, 0.0))
                i, j = entry["pair"]
                entry["roots"] = (_principal_sqrt(cofactors[i, i]),
                                  _principal_sqrt(cofactors[j, j]))
                entry["arccosine"] = _principal_acos(_pinned_cosine(
                    cofactors[i, j], entry["root_sign"] * entry["roots"][0]
                    * entry["roots"][1]))
            hinge["content"] = hinge["content_sign"] * cmath.sqrt(
                _positive_zero(self._gram_determinant(hinge["hinge"], now,
                                                      now, 0.0)))

    @staticmethod
    def _cayley_menger(ids, origin, target, t):
        """The Cayley-Menger matrix of the simplex on the sorted vertex ids
        at parameter t of the straight segment between two geometries."""
        n = len(ids) + 1
        matrix = np.zeros((n, n), dtype=complex)
        matrix[0, 1:] = 1.0
        matrix[1:, 0] = 1.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = _pair(ids[i], ids[j])
                z = origin[key] + t * (target[key] - origin[key])
                matrix[i + 1, j + 1] = matrix[j + 1, i + 1] = z
        return matrix

    @staticmethod
    def _gram_determinant(ids, origin, target, t):
        """The Gram determinant of the simplex on the sorted vertex ids at
        parameter t of the segment, the radicand of its content root."""
        size = len(ids) - 1
        if size < 1:
            return 1.0 + 0.0j

        def z(a, b):
            if a == b:
                return 0.0j
            key = _pair(ids[a], ids[b])
            return origin[key] + t * (target[key] - origin[key])
        gram = np.array([[0.5 * (z(0, i + 1) + z(0, j + 1) - z(i + 1, j + 1))
                          for j in range(size)] for i in range(size)])
        return complex(np.linalg.det(gram))

    def _continue_regge(self, hinges, now, regge_start):
        """The sheets of `ReggeBranch.Continued`, written into ``hinges``."""
        declared = self._declaration
        cayley_menger = self._cayley_menger
        gram_determinant = self._gram_determinant
        if regge_start is None:
            regge_start = list(declared.regge_start_squared_lengths)
        start = ({_pair(a, b): complex(z)
                  for (a, b), z in zip(self._stored, regge_start)}
                 if len(regge_start) else dict(now))
        reference = {key: complex(value.real, 0.0)
                     for key, value in start.items()}
        legs = [(reference, start), (start, now)]

        def walk(state, advance, copy, what):
            """Walk ``state`` along a segment, refining every step by
            bisection until it is fine."""
            t, step = 0.0, 1.0
            while t < 1.0:
                following = min(1.0, t + step)
                trial = copy(state)
                if advance(trial, following):
                    state, t, step = trial, following, min(1.0, 2.0 * step)
                elif step <= REGGE_SHORTEST_STEP:
                    raise ValueError(
                        "SeriesStationarity: the continued Regge sheets "
                        "cannot be followed: %s makes no fine step from "
                        "parameter %g of the segment at the shortest step, "
                        "2^-30 of it" % (what, t))
                else:
                    step *= 0.5
            return state

        # every cell's roots and angles, continued along the two segments
        pairs = {}
        for hinge in hinges:
            for entry in hinge["cells"]:
                pairs.setdefault(entry["cell"], [])
                if entry["pair"] not in pairs[entry["cell"]]:
                    pairs[entry["cell"]].append(entry["pair"])
        continued = {}
        for cell, cell_pairs in pairs.items():
            n = len(cell) + 1
            cofactors = _cofactors(cayley_menger(cell, reference, reference,
                                                 0.0))
            roots = [mesh.SheetedSqrt(complex(cofactors[p, p]))
                     for p in range(1, n)]
            angles = [mesh.SheetedAcos(_pinned_cosine(
                cofactors[i, j], roots[i - 1].value() * roots[j - 1].value()))
                for i, j in cell_pairs]

            def copy(state):
                return ([mesh.SheetedSqrt(r.radicand(), r.winding())
                         for r in state[0]],
                        [mesh.SheetedAcos(a.cosine(), a.branchIndex(),
                                          a.orientation())
                         for a in state[1]])

            for origin, target in legs:
                def advance(state, t, origin=origin, target=target):
                    cof = _cofactors(cayley_menger(cell, origin, target, t))
                    fine = True
                    for p in range(1, n):
                        root = state[0][p - 1]
                        root.advance(complex(cof[p, p]))
                        if abs(root.lastStep()) > REGGE_MAXIMUM_ROOT_TURN:
                            fine = False
                    for index, (i, j) in enumerate(cell_pairs):
                        angle = state[1][index]
                        angle.advance(_pinned_cosine(
                            cof[i, j], state[0][i - 1].value()
                            * state[0][j - 1].value()))
                        if angle.lastStep() > REGGE_MAXIMUM_ANGLE_STEP:
                            fine = False
                    return fine
                roots, angles = walk(
                    (roots, angles), advance, copy,
                    "the dihedral angles of the cell on vertices %s"
                    % (list(cell),))
            for index, (i, j) in enumerate(cell_pairs):
                continued[(cell, (i, j))] = (
                    roots[i - 1].value() * roots[j - 1].value(),
                    angles[index].value())

        # each continued value read as a sheet of the cofactors at the point
        for hinge in hinges:
            for entry in hinge["cells"]:
                product, angle = continued[(entry["cell"], entry["pair"])]
                cofactors = _cofactors(cayley_menger(entry["cell"], now, now,
                                                     0.0))
                i, j = entry["pair"]
                principal = (_principal_sqrt(cofactors[i, i])
                             * _principal_sqrt(cofactors[j, j]))
                entry["root_sign"] = (
                    1 if (product.conjugate() * principal).real >= 0.0
                    else -1)
                arccosine = _principal_acos(_pinned_cosine(
                    cofactors[i, j], entry["root_sign"] * principal))
                best = math.inf
                for orientation in (1, -1):
                    turns = round((angle - orientation * arccosine).real
                                  / (2.0 * math.pi))
                    candidate = 2.0 * math.pi * turns + orientation * arccosine
                    if abs(candidate - angle) < best:
                        best = abs(candidate - angle)
                        entry["orientation"] = orientation
                        entry["branch_index"] = int(turns)
            ids = hinge["hinge"]
            if len(ids) >= 2:
                content = mesh.SheetedSqrt(
                    gram_determinant(ids, reference, reference, 0.0))
                for origin, target in legs:
                    def advance(state, t, origin=origin, target=target):
                        state.advance(gram_determinant(ids, origin, target, t))
                        return abs(state.lastStep()) <= REGGE_MAXIMUM_ROOT_TURN
                    content = walk(
                        content, advance,
                        lambda s: mesh.SheetedSqrt(s.radicand(), s.winding()),
                        "the content root of the hinge on vertices %s"
                        % (list(ids),))
                volume = cmath.sqrt(gram_determinant(ids, now, now, 0.0))
                hinge["content_sign"] = (
                    1 if (content.value().conjugate() * volume).real >= 0.0
                    else -1)

    # --------------------------------------------------------------- queries

    def variable_count(self):
        """The number of variables, which is the number of equations: the
        relaxed length classes, the relaxed link classes and the relaxed
        multipliers."""
        return self._count

    def residual(self, displacement):
        """The residual on the path x_0 + s(t).

        ``displacement`` is s: one `TruncatedSeries` per variable, in the
        variable order of `HolomorphicRelaxation.jacobian`'s columns (the
        squared lengths of the classes, then their links, then the
        multipliers), all of one order p and each with a zero order-0
        coefficient. A squared-length variable moves the class's z by its
        series, a link variable multiplies every link of the class by the
        exponential of its series on the member's orientation, and a
        multiplier variable adds its series. Returns one `TruncatedSeries`
        of order p per equation, in the residual's block order."""
        displacement = list(displacement)
        if len(displacement) != self._count:
            raise ValueError(
                "SeriesStationarity.residual: %d series for %d variables"
                % (len(displacement), self._count))
        orders = {int(series.order) for series in displacement}
        if len(orders) > 1:
            raise ValueError(
                "SeriesStationarity.residual: the series of the displacement "
                "differ in order")
        order = orders.pop() if orders else 0
        steps = np.array([series.coefficients for series in displacement],
                         dtype=complex).reshape(self._count, order + 1)
        if np.any(steps[:, 0] != 0.0):
            raise ValueError(
                "SeriesStationarity.residual: the displacement has a nonzero "
                "order-0 coefficient; the path passes through the point the "
                "branches are declared at")
        values = self._evaluate(steps, order)
        return [nm.TruncatedSeries([complex(c) for c in row])
                for row in values]

    def reversion(self, order, solve):
        """The reversion of the residual about the point to ``order`` (one
        to ten), a `tessera.numerics.SeriesReversion`: the path
        s(t) = s_1 t + ... + s_p t^p with
        F(x_0 + s(t)) = (1 - t) F(x_0) + O(t^(p + 1)), each linear system
        J s_k = r handed to ``solve`` (a callable from the right-hand side to
        the solution, which decides how a rank-deficient Jacobian is
        treated). Its ``coefficients`` are the s_k and ``evaluate(t)`` is the
        path at t."""
        return nm.revert(self.residual, np.zeros(self._count, dtype=complex),
                         int(order), solve)

    def step(self, order, solve):
        """The step of order ``order``: s(1) of `reversion`. Order one is
        Newton's step with that solve. It approaches a root as the order
        grows when t = 1 lies inside the disc of convergence of the path."""
        return np.asarray(self.reversion(order, solve).step(), dtype=complex)

    # ------------------------------------------------------------- the terms

    def _evaluate(self, steps, order):
        """The residual's coefficients, one row per equation, for the
        displacement coefficients ``steps`` (one row per variable)."""
        edges = len(self._cells[1])
        zero = np.zeros(order + 1, dtype=complex)
        # the fields on the path, per canonical edge
        z = _constant(self._z0, order)
        increment = np.zeros((order + 1, edges), dtype=complex)
        for index, members in enumerate(self._members):
            for edge, orientation in members:
                canonical = self._canonical[edge]
                if self._lengths:
                    z[:, canonical] += steps[self._length_offset + index]
                if self._links:
                    increment[:, canonical] += (
                        self._stored_sign[edge] * orientation
                        * steps[self._link_offset + index])
        link = np.zeros((order + 1, edges), dtype=complex)
        inverse = np.zeros((order + 1, edges), dtype=complex)
        for edge in range(edges):
            if np.any(increment[:, edge] != 0.0):
                link[:, edge] = self._link0[edge] * _exp(increment[:, edge])
                inverse[:, edge] = (_exp(-increment[:, edge])
                                    / self._link0[edge])
            else:
                link[0, edge] = self._link0[edge]
                inverse[0, edge] = 1.0 / self._link0[edge]
        multipliers = _constant(
            np.array([complex(c.multiplier) for c in self._constraints]),
            order)
        if self._multipliers:
            for index in range(len(self._constraints)):
                multipliers[:, index] += steps[self._multiplier_offset + index]

        length = np.zeros((order + 1, edges), dtype=complex)
        canonical_link = np.zeros((order + 1, edges), dtype=complex)
        constraint_values = np.zeros((order + 1, len(self._constraints)),
                                     dtype=complex)
        if self._regge and self._lengths:
            length += self._regge_part(z, order)
        if self._villain is not None and self._links:
            canonical_link += self._villain_part(link, inverse, order)
        if self._carrier_needed:
            matter_length, matter_link, constraint_values = self._carrier_part(
                z, link, inverse, multipliers, order)
            length += matter_length
            canonical_link += matter_link

        rows = []
        if self._lengths:
            for members in self._members:
                total = zero.copy()
                for edge, _ in members:
                    total += length[:, self._canonical[edge]]
                rows.append(total)
        if self._links:
            for members in self._members:
                total = zero.copy()
                for edge, orientation in members:
                    total += (orientation * self._stored_sign[edge]
                              * canonical_link[:, self._canonical[edge]])
                rows.append(total)
        if self._multipliers:
            for index, constraint in enumerate(self._constraints):
                row = constraint_values[:, index].copy()
                row[0] -= complex(constraint.target)
                rows.append(row)
        return rows

    # -- the Villain term

    def _villain_part(self, link, inverse, order):
        """sum_tau eps_(tau e) D phi(F_tau) per canonical edge, with
        D phi(F) = -beta_V DW_M(F) / W_M(F)."""
        villain = self._villain
        faces = villain["faces"]
        holonomy = _constant(np.ones(faces), order)
        reciprocal = _constant(np.ones(faces), order)
        for row, column, value in villain["incidence"]:
            forward = link[:, row] if value > 0 else inverse[:, row]
            backward = inverse[:, row] if value > 0 else link[:, row]
            holonomy[:, column] = _mul(holonomy[:, column], forward)
            reciprocal[:, column] = _mul(reciprocal[:, column], backward)
        value = _constant(np.ones(faces), order)
        first = np.zeros((order + 1, faces), dtype=complex)
        up = _constant(np.ones(faces), order)
        down = _constant(np.ones(faces), order)
        for m, coefficient in enumerate(villain["coefficients"]):
            if m == 0:
                continue
            up = _mul(up, holonomy)
            down = _mul(down, reciprocal)
            value += coefficient * (up + down)
            first += coefficient * m * (up - down)
        derivative = np.zeros((order + 1, faces), dtype=complex)
        for face in range(faces):
            derivative[:, face] = -villain["weight"] * _divide(first[:, face],
                                                               value[:, face])
        out = np.zeros((order + 1, link.shape[1]), dtype=complex)
        for row, column, sign in villain["incidence"]:
            out[:, row] += sign * derivative[:, column]
        return out

    # -- the Regge term

    def _regge_part(self, z, order):
        """w_G sum_h (d|h|/dz_e eps_h + |h| d eps_h/dz_e) per canonical
        edge, every root and angle on its sheet at the point."""
        d = self._dimension
        weight = float(self._declaration.gravitational_weight)
        out = np.zeros((order + 1, z.shape[1]), dtype=complex)

        def squared(a, b):
            return z[:, self._edge_index[_pair(a, b)]]

        cells = {}
        for hinge in self._regge:
            for entry in hinge["cells"]:
                cell = entry["cell"]
                if cell in cells:
                    continue
                n = len(cell) + 1
                matrix = np.zeros((order + 1, n, n), dtype=complex)
                matrix[0, 0, 1:] = 1.0
                matrix[0, 1:, 0] = 1.0
                for i in range(len(cell)):
                    for j in range(i + 1, len(cell)):
                        value = squared(cell[i], cell[j])
                        matrix[:, i + 1, j + 1] = value
                        matrix[:, j + 1, i + 1] = value
                determinant = _determinant(matrix)
                inverse = _inverse(matrix)
                cells[cell] = (determinant, inverse,
                               _scale(determinant, inverse))
        for hinge in self._regge:
            ids = hinge["hinge"]
            size = len(ids) - 1
            # the content sqrt(det G) / (d - 2)! and its gradient
            gram = np.zeros((order + 1, size, size), dtype=complex)
            for i in range(size):
                for j in range(size):
                    gram[:, i, j] = 0.5 * (
                        squared(ids[0], ids[i + 1])
                        + squared(ids[0], ids[j + 1])
                        - (squared(ids[i + 1], ids[j + 1]) if i != j else 0.0))
            determinant = _determinant(gram)
            content = _sqrt(determinant, hinge["content"]) \
                / math.factorial(d - 2)
            gram_inverse = _inverse(gram)
            content_gradient = {}
            for p in range(len(ids)):
                for q in range(p + 1, len(ids)):
                    change = np.zeros((size, size))
                    for i in range(size):
                        for j in range(size):
                            def indicator(a, b):
                                return 1.0 if (a != b and {a, b} == {p, q}) \
                                    else 0.0
                            change[i, j] = 0.5 * (
                                indicator(0, i + 1) + indicator(0, j + 1)
                                - indicator(i + 1, j + 1))
                    trace = np.einsum("kij,ji->k", gram_inverse, change)
                    content_gradient[_pair(ids[p], ids[q])] = 0.5 * _mul(
                        content, trace)
            # the deficit 2 pi - sum theta and its gradient
            deficit = _constant(2.0 * math.pi, order)
            deficit_gradient = {}
            for entry in hinge["cells"]:
                cell = entry["cell"]
                determinant, inverse, cofactor = cells[cell]
                i, j = entry["pair"]
                cij, cii, cjj = (cofactor[:, i, j], cofactor[:, i, i],
                                 cofactor[:, j, j])
                denominator = entry["root_sign"] * _mul(
                    _sqrt(cii, entry["roots"][0]),
                    _sqrt(cjj, entry["roots"][1]))
                ratio = _divide(-cij, denominator)
                arccosine = _acos(ratio, entry["arccosine"])
                theta = entry["orientation"] * arccosine
                theta[0] += 2.0 * math.pi * entry["branch_index"]
                deficit -= theta
                # d theta / d r = -1 / sin(theta), with sin(theta) the root
                # of 1 - r^2 whose order-0 value is the sine of the angle
                sine = _sqrt(_constant(1.0, order) - _mul(ratio, ratio),
                             cmath.sin(theta[0]))
                slope = _divide(_constant(-1.0, order), sine)
                m = len(cell)
                for a in range(m):
                    for b in range(a + 1, m):
                        def derivative(p, q):
                            return _mul(determinant, (
                                2.0 * _mul(inverse[:, a + 1, b + 1],
                                           inverse[:, p, q])
                                - _mul(inverse[:, p, a + 1],
                                       inverse[:, b + 1, q])
                                - _mul(inverse[:, p, b + 1],
                                       inverse[:, a + 1, q])))
                        dcij, dcii, dcjj = (derivative(i, j), derivative(i, i),
                                            derivative(j, j))
                        ddenominator = 0.5 * _mul(
                            denominator,
                            _divide(dcii, cii) + _divide(dcjj, cjj))
                        dratio = -_divide(
                            _mul(dcij, denominator) - _mul(cij, ddenominator),
                            _mul(denominator, denominator))
                        key = _pair(cell[a], cell[b])
                        deficit_gradient[key] = deficit_gradient.get(
                            key, 0.0) - _mul(slope, dratio)
            for key, gradient in content_gradient.items():
                out[:, self._edge_index[key]] += weight * _mul(gradient,
                                                               deficit)
            for key, gradient in deficit_gradient.items():
                out[:, self._edge_index[key]] += weight * _mul(content,
                                                               gradient)
        return out

    # -- the carrier operator

    def _gram(self, local):
        """The Gram matrix of one top cell from its local squared lengths
        (series of shape (order + 1, edges of the cell))."""
        d = self._dimension
        order = local.shape[0] - 1

        def squared(a, b):
            if a == b:
                return np.zeros(order + 1, dtype=complex)
            a, b = min(a, b), max(a, b)
            return local[:, _local_edge_index(d, a, b)]

        gram = np.zeros((order + 1, d, d), dtype=complex)
        for i in range(1, d + 1):
            for j in range(1, d + 1):
                gram[:, i - 1, j - 1] = 0.5 * (squared(0, i) + squared(0, j)
                                               - squared(i, j))
        return gram

    def _gram_derivatives(self):
        """dG/ds_m for the local edges m of a top cell, constant matrices."""
        d = self._dimension
        out = []
        for a in range(d + 1):
            for b in range(a + 1, d + 1):
                change = np.zeros((d, d))
                if a == 0:
                    for i in range(1, d + 1):
                        change[i - 1, b - 1] += 0.5
                        change[b - 1, i - 1] += 0.5
                else:
                    change[a - 1, b - 1] -= 0.5
                    change[b - 1, a - 1] -= 0.5
                out.append(change)
        return out

    @staticmethod
    def _extend(block):
        """The extended Gamma ((d+1) x (d+1)) of a d x d block: the Gram
        matrix of the barycentric gradients from the inverse Gram matrix."""
        order, d = block.shape[0], block.shape[1]
        out = np.zeros((order, d + 1, d + 1), dtype=complex)
        out[:, 1:, 1:] = block
        columns = block.sum(axis=1)
        out[:, 0, 1:] = -columns
        out[:, 1:, 0] = -columns
        out[:, 0, 0] = block.sum(axis=(1, 2))
        return out

    def _minor_table(self, degree):
        """The expansion of the local Whitney block of a degree: for every
        entry (p, q) of the block and every pair (i, j) of removed vertices,
        the two index lists of the minor and its coefficient
        (-1)^(i+j) delta lambda (k!)^2."""
        d = self._dimension
        faces = self._faces[degree]
        weight = (math.factorial(degree) ** 2
                  / float((d + 1) * (d + 2)))
        rows, columns, coefficients, entries = [], [], [], []
        for p, sigma in enumerate(faces):
            for q, tau in enumerate(faces):
                for i in range(degree + 1):
                    for j in range(degree + 1):
                        rows.append([v for n, v in enumerate(sigma) if n != i])
                        columns.append([v for n, v in enumerate(tau)
                                        if n != j])
                        coefficients.append(
                            (1.0 if (i + j) % 2 == 0 else -1.0)
                            * (2.0 if sigma[i] == tau[j] else 1.0) * weight)
                        entries.append(p * len(faces) + q)
        return (np.array(rows, dtype=int).reshape(len(rows), degree),
                np.array(columns, dtype=int).reshape(len(columns), degree),
                np.array(coefficients), np.array(entries, dtype=int),
                len(faces))

    def _local(self, top, z, with_derivative):
        """One top cell's local geometry on the path: the extended Gamma,
        the volume on its branch, and, with ``with_derivative``, their
        derivatives in the cell's squared lengths."""
        d = self._dimension
        gram = self._gram(z[:, self._top_edges[top]])
        inverse = _inverse(gram)
        gamma = self._extend(inverse)
        determinant = _determinant(gram)
        factorial = math.factorial(d)
        volume = _sqrt(determinant, self._top_volume[top] * factorial) \
            / factorial
        out = {"gamma": gamma, "volume": volume}
        if with_derivative:
            out["dgamma"], out["dvolume"] = [], []
            for change in self._gram_derivatives():
                change = _constant(change, gram.shape[0] - 1)
                product = _matmul(inverse, change)
                out["dgamma"].append(self._extend(-_matmul(product, inverse)))
                out["dvolume"].append(0.5 * _mul(volume, _trace(product)))
        return out

    def _block(self, local, degree):
        """The local Whitney block of a degree, of shape
        (order + 1, faces, faces)."""
        rows, columns, coefficients, entries, faces = \
            self._minor_tables[degree]
        gamma, volume = local["gamma"], local["volume"]
        minors = _minors([gamma] * degree, rows, columns,
                         gamma.shape[0] - 1) * coefficients
        block = np.zeros((gamma.shape[0], faces * faces), dtype=complex)
        np.add.at(block, (slice(None), entries), minors)
        return _scale(volume, block.reshape(-1, faces, faces))

    def _block_derivative(self, local, degree, m):
        """The derivative of the local Whitney block in the cell's local
        squared length m: d(vol) det + vol d(det), the determinant's
        derivative by rows."""
        rows, columns, coefficients, entries, faces = \
            self._minor_tables[degree]
        gamma, volume = local["gamma"], local["volume"]
        dgamma, dvolume = local["dgamma"][m], local["dvolume"][m]
        order = gamma.shape[0] - 1
        minors = _minors([gamma] * degree, rows, columns, order)
        total = _scale(dvolume, minors)
        for r in range(degree):
            sources = [gamma] * degree
            sources[r] = dgamma
            total = total + _scale(volume, _minors(sources, rows, columns,
                                                   order))
        block = np.zeros((gamma.shape[0], faces * faces), dtype=complex)
        np.add.at(block, (slice(None), entries), total * coefficients)
        return block.reshape(-1, faces, faces)

    def _link_table(self, rows_degree, columns_degree):
        """For the cells of two degrees: the canonical edge joining the base
        vertices of a row cell and a column cell (-1 for none), and +1 when
        the pair runs along the edge's ascending orientation, -1 when
        against it and 0 when the base vertices coincide."""
        key = (rows_degree, columns_degree)
        if key in self._link_tables:
            return self._link_tables[key]
        rows = self._base[rows_degree]
        columns = self._base[columns_degree]
        edge = -np.ones((len(rows), len(columns)), dtype=int)
        sign = np.zeros((len(rows), len(columns)))
        for r, a in enumerate(rows):
            for c, b in enumerate(columns):
                if a == b:
                    continue
                index = self._edge_index.get(_pair(a, b))
                if index is None:
                    continue
                edge[r, c] = index
                sign[r, c] = 1.0 if a < b else -1.0
        self._link_tables[key] = (edge, sign)
        return edge, sign

    def _link_factor(self, rows_degree, columns_degree, link, inverse):
        """The link U_(b(row) b(col)) between the base vertices of every row
        cell and every column cell: one where they coincide, the edge's link
        along its ascending orientation and its inverse against it, and zero
        for a pair that no edge joins (which carries no entry)."""
        edge, sign = self._link_table(rows_degree, columns_degree)
        factor = np.zeros((link.shape[0],) + edge.shape, dtype=complex)
        equal = np.array([[a == b for b in self._base[columns_degree]]
                          for a in self._base[rows_degree]])
        factor[0][equal] = 1.0
        forward = sign > 0
        backward = sign < 0
        factor[:, forward] = link[:, edge[forward]]
        factor[:, backward] = inverse[:, edge[backward]]
        return factor

    def _dress(self, bare, rows_degree, columns_degree, link, inverse):
        """A matrix series with every entry multiplied by the link between
        the base vertices of its row and column cells."""
        factor = self._link_factor(rows_degree, columns_degree, link, inverse)
        if bare.shape[0] == 1:
            return bare[0][None] * factor
        return _mul(bare, factor)

    def _band_operator(self, operator):
        """The operator the bands are read on: the carrier, or its average
        over the declared band symmetry."""
        if self._mean_field is None or not getattr(self, "_symmetry", None):
            return operator
        total = np.zeros_like(operator)
        for element in self._symmetry:
            inverse = np.linalg.inv(element)
            total += np.einsum("ij,kjl,lm->kim", inverse, operator, element)
        return total / len(self._symmetry)

    def _state(self, operator, order):
        """The carried state on the path: the covariance, the fiber's
        projector and the pinned bands' projectors, constant when the system
        holds them fixed and rebuilt from the bands' Riesz projectors when it
        is self-consistent."""
        if self._bands is None:
            def constant(matrix):
                return None if matrix is None else _constant(matrix, order)
            return (constant(self._covariance0), constant(self._projector0),
                    [constant(p) for p in self._band_projectors0])
        band_operator = _matrix(self._band_operator(operator))
        eigenvalues = self._band_eigenvalues
        projectors = []
        for band in self._bands:
            modes = band["modes"]

            def in_group(value, modes=modes):
                return int(np.argmin(np.abs(eigenvalues - value))) in modes
            projectors.append(np.array(
                band_operator.rieszProjector(in_group).coefficients,
                dtype=complex))
        covariance = sum(band["weight"] * projector
                         for band, projector in zip(self._bands, projectors))
        if not self._fiber:
            return covariance, None, []
        return (covariance, sum(projectors),
                projectors[:self._pinned_bands])

    def _carrier_part(self, z, link, inverse, multipliers, order):
        """The matter and constraint terms of the stationarity equations per
        canonical edge, tr(A dL/dz_e) and tr(A U_e dL/dU_e) with
        A = w_M Gamma + sum_j xi_j X_j, and the constraints' values."""
        d, k = self._dimension, self._degree
        lower, upper = k >= 1, k < d
        lengths = self._lengths
        # the Whitney mass matrices of the degrees in play, and each top
        # cell's local geometry
        locals_ = [self._local(top, z, lengths)
                   for top in range(len(self._tops))]
        bare = {}
        indices = {}
        for j in self._degrees:
            size = len(self._cells[j])
            matrix = np.zeros((order + 1, size, size), dtype=complex)
            indices[j] = []
            for top, vertices in enumerate(self._tops):
                cells = np.array([self._cell_index[j][tuple(
                    vertices[v] for v in face)] for face in self._faces[j]])
                indices[j].append(cells)
                block = self._block(locals_[top], j)
                np.add.at(matrix, (slice(None), cells[:, None],
                                   cells[None, :]), block)
            bare[j] = matrix
        M = self._dress(bare[k], k, k, link, inverse)
        Q = _inverse(M)
        size = M.shape[1]
        h = np.zeros((order + 1, size, size), dtype=complex)
        if lower:
            Mm = self._dress(bare[k - 1], k - 1, k - 1, link, inverse)
            P = _inverse(Mm)
            boundary = _constant(self._boundary[k], 0)
            B = self._dress(boundary, k - 1, k, link, inverse)
            Bd = self._dress(boundary, k - 1, k, inverse, link)
            AT = _transpose(Bd)
            PB = _matmul(P, B)
            T1 = _matmul(M, AT)
            h = h + _matmul(T1, PB)
        if upper:
            Mp = self._dress(bare[k + 1], k + 1, k + 1, link, inverse)
            boundary = _constant(self._boundary[k + 1], 0)
            C = self._dress(boundary, k, k + 1, link, inverse)
            Cd = self._dress(boundary, k, k + 1, inverse, link)
            DT = _transpose(Cd)
            DQ = _matmul(DT, Q)
            T2 = _matmul(C, Mp, DQ)
            h = h + T2
        reference = _matmul(Q, h, M)
        signs = self._signs[k]
        operator = signs[None, :, None] * reference * signs[None, None, :]

        # the carried state, the constraints and the contraction matrix
        covariance, projector, band_projectors = self._state(operator, order)
        declared = self._declaration
        scale = float(declared.moment_scale)
        contraction = np.zeros((order + 1, size, size), dtype=complex)
        if declared.matter_weight != 0.0 and covariance is not None:
            contraction += float(declared.matter_weight) * covariance
        values = np.zeros((order + 1, len(self._constraints)), dtype=complex)
        identity = _constant(np.eye(size), order)
        for index, constraint in enumerate(self._constraints):
            if constraint.form == cob.SpectralConstraintForm.BandMean:
                band = band_projectors[int(constraint.band)]
                rank = complex(np.trace(band[0]))
                if rank == 0.0:
                    continue
                values[:, index] = _trace(_matmul(band, operator)) / (
                    rank * scale)
                derivative = band / (rank * scale)
            else:
                power = int(constraint.order)
                constrained = (operator if projector is None else
                               _matmul(projector, operator, projector)) / scale
                powered = identity
                for _ in range(power - 1):
                    powered = _matmul(powered, constrained)
                values[:, index] = _trace(_matmul(powered, constrained))
                if projector is not None:
                    powered = _matmul(projector, powered, projector)
                derivative = (power / scale) * powered
            contraction += _scale(multipliers[:, index], derivative)

        # tr(A dL) = sum_N tr(dN Z_N) over the matrices L is built from
        A = signs[None, :, None] * contraction * signs[None, None, :]
        W = _matmul(A, Q)
        X = {k: -_matmul(reference, W) + _matmul(W, h)}
        if lower:
            X[k] = X[k] + _matmul(AT, PB, M, W)
            X[k - 1] = -_matmul(PB, M, W, T1, P)
        if upper:
            X[k] = X[k] - _matmul(W, T2)
            X[k + 1] = _matmul(DQ, M, W, C)
        edges = z.shape[1]
        matter_length = np.zeros((order + 1, edges), dtype=complex)
        matter_link = np.zeros((order + 1, edges), dtype=complex)
        dressed = {k: M}
        if lower:
            dressed[k - 1] = Mm
        if upper:
            dressed[k + 1] = Mp
        if lengths:
            for j in self._degrees:
                # tr(dM^U X) = sum_rc dM_rc U_rc X_cr
                weight = _mul(self._link_factor(j, j, link, inverse),
                              _transpose(X[j]))
                for top in range(len(self._tops)):
                    cells = indices[j][top]
                    local = weight[:, cells[:, None], cells[None, :]]
                    for m, edge in enumerate(self._top_edges[top]):
                        block = self._block_derivative(locals_[top], j, m)
                        matter_length[:, edge] += _mul(block, local).sum(
                            axis=(1, 2))
        if self._links:
            def accumulate(matrix, partner, rows_degree, columns_degree,
                           direction):
                """Add sum over the entries whose link factor is the edge's
                link (or its inverse) of +-N_rc Z_rc."""
                edge, sign = self._link_table(rows_degree, columns_degree)
                product = _mul(matrix, partner)
                moving = edge >= 0
                np.add.at(matter_link, (slice(None), edge[moving]),
                          direction * sign[moving] * product[:, moving])
            for j in self._degrees:
                accumulate(dressed[j], _transpose(X[j]), j, j, 1.0)
            if lower:
                accumulate(B, _transpose(_matmul(M, W, T1, P)), k - 1, k, 1.0)
                accumulate(Bd, _matmul(PB, M, W, M), k - 1, k, -1.0)
            if upper:
                accumulate(C, _transpose(_matmul(Mp, DQ, M, W)), k, k + 1,
                           1.0)
                accumulate(Cd, _matmul(W, C, Mp), k, k + 1, -1.0)
        return matter_length, matter_link, values
