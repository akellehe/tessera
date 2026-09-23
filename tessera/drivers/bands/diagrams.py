# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""The expansion of the self-energy in the screened interaction beyond
Sigma = i G W, as time-ordered skeleton diagrams evaluated in closed form.

The screened interaction of the random-phase approximation is the bare kernel
plus the exchange of the particle-hole modes (`screening.RandomPhase`),

    W_{pq,rs}(w) = (pq|rs) + sum_s g^s_pq g^s_rs 2 W_s / (w^2 - W_s^2) ,

so every diagram is a diagram of electrons coupled linearly to bosons of
energies W_s plus an instantaneous interaction, with the Hartree-Fock propagator
on every fermion line. The skeleton diagrams (no self-energy insertion on a
fermion line, no polarization insertion on an interaction line, both being
carried by the propagator and by W) are one at first order, one at second (the
two interaction lines crossed), six at third (the four irreducible ways of
crossing three lines, and the two orientations of a fermion triangle that
absorbs three lines), 49 at fourth and 542 at fifth. `skeleton_diagrams`
enumerates them: every way of pairing the 2 k vertices of one open fermion line
and any number of closed loops by k interaction lines, one representative per
relabelling of the loops, kept when no two fermion lines and no one or two
interaction lines cut it in two. No diagram has a symmetry of its own (the
external line fixes the open line, and an interaction line fixes the vertex at
its other end), so each counts once.

A diagram is evaluated as the sum over the orderings of its vertex times. In an
ordering a fermion line that runs forward in time is a particle (an empty
state, energy xi above the chemical potential), one that runs backward a hole (a
filled state, energy -xi, and a factor -1); a boson is present between its two
vertices; an instantaneous line puts its two vertices at one time. Between
consecutive times the intermediate state has the energy E of everything present,
and contributes 1 / (w - E) when the external line spans it forward, 1 / (-w - E)
when it spans it backward, and 1 / (-E) otherwise. A closed fermion loop carries
-1 and the factor 2 of its spin sum. Every denominator is w minus an excitation
energy, so nothing cancels between large terms, and the expression is analytic
in w: a complex w gives the derivative (`SkeletonSelfEnergy.evaluate`) and the
value on the imaginary axis, where the test suite holds every diagram to the
plain frequency integrals of its Feynman rules.

The orderings are not visited one at a time. The lines present after a set S of
vertices has happened, and whether each is a particle or a hole, depend on S
alone, so the sum over the orderings of S of everything up to that moment is one
tensor T(S) over the labels of those lines, and

    T(S) = 1 / (z_S - E_S)  sum over v in S of  T(S - v) . (the couplings at v) .

That is V 2^(V-1) contractions for V vertices where the orderings are V!, and it
is the same number, term for term (`SkeletonSelfEnergy._orderings` is the plain
sum, kept as the reference of the test suite). What it costs is the memory of
T(S) where the most lines are present at once: every other vertex first puts
all 2 k - 1 fermion lines and all k interaction lines of an order k in one
denominator, which no factorization separates. `memory` bounds what is held at
once; beyond it the labels of interaction lines are fixed one value at a time,
which trades the memory for repetitions of the narrow part of the recursion.
"""
import functools
import heapq
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

_POOL = None
_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# What one evaluation may hold at once, in bytes, over all of its worker processes (`SkeletonSelfEnergy.memory`).
MEMORY = 8 * 2 ** 30


def _workers():
    """The number of worker processes: every core but one, or the environment's
    TESSERA_DIAGRAM_WORKERS."""
    return max(1, int(os.environ.get("TESSERA_DIAGRAM_WORKERS", (os.cpu_count() or 2) - 1)))


def _pool():
    """One pool of worker processes for the life of the interpreter: the terms of
    a diagram are thousands of small independent contractions, whose cost is
    the interpreter's and does not thread."""
    global _POOL
    import __main__
    if not hasattr(__main__, "__file__"):
        return None                                              # workers re-import the main module; without one, run serially
    if _POOL is None:
        import multiprocessing
        # The workers start fresh and single-threaded: a threaded linear-algebra library in every one of them
        # would oversubscribe the machine.
        saved = {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        os.environ.update({name: "1" for name in saved})
        try:
            _POOL = ProcessPoolExecutor(max_workers=_workers(), mp_context=multiprocessing.get_context("spawn"))
            list(_POOL.map(abs, range(_POOL._max_workers)))      # start them while the environment is set
        finally:
            for name, value in saved.items():
                os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)
    return _POOL


def _task(arguments):
    engine, items, n, frequency = arguments
    return sum(engine._diagram(diagram, kinds, n, frequency, fixed) for diagram, kinds, fixed in items)


@dataclass(frozen=True)
class Diagram:
    """`edges` are the internal fermion lines (from vertex, to vertex); the
    external line enters at `entry` and leaves at `exit`; `chords` pair the
    vertices by interaction lines; `loops` counts closed fermion loops."""
    vertices: int
    edges: tuple
    entry: int
    exit: int
    chords: tuple
    loops: int = 0


# -- the enumeration

def _partitions(count, smallest, largest=None):
    """The ways of writing `count` as a sum of parts no smaller than `smallest`, largest part first."""
    largest = count if largest is None else largest
    if count == 0:
        yield ()
    for part in range(min(count, largest), smallest - 1, -1):
        for rest in _partitions(count - part, smallest, part):
            yield (part,) + rest


def _matchings(items):
    """The ways of pairing the items."""
    if not items:
        yield ()
        return
    for j in range(1, len(items)):
        for rest in _matchings(items[1:j] + items[j + 1:]):
            yield ((items[0], items[j]),) + rest


def _connected(nodes, lines):
    parent = list(range(nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pieces = nodes
    for a, b in lines:
        a, b = find(a), find(b)
        if a != b:
            parent[a], pieces = b, pieces - 1
    return pieces == 1


def _without(lines, *dropped):
    return [line for index, line in enumerate(lines) if index not in dropped]


@functools.lru_cache(maxsize=None)
def _generate(order, skeleton):
    """The diagrams of `order` interaction lines, open line first: vertices
    0 .. m - 1 along the open line, then each closed loop in turn. A pairing and
    its image under a rotation of a loop, or an exchange of two loops of one
    length, are one diagram."""
    count, found = 2 * order, []
    for line in range(count, 1 if skeleton else 0, -1):
        for lengths in _partitions(count - line, 3 if skeleton else 2):
            edges, blocks, start = [(j, j + 1) for j in range(line - 1)], [], line
            for length in lengths:
                blocks.append(list(range(start, start + length)))
                edges += [(blocks[-1][i], blocks[-1][(i + 1) % length]) for i in range(length)]
                start += length
            relabellings = []
            for rotated in itertools.product(*[[block[r:] + block[:r] for r in range(len(block))] for block in blocks]):
                for images in itertools.permutations(range(len(blocks))):
                    if all(len(blocks[images[i]]) == len(blocks[i]) for i in range(len(blocks))):
                        relabelling = list(range(count))
                        for i, block in enumerate(blocks):
                            for source, target in zip(rotated[images[i]], block):
                                relabelling[source] = target
                        relabellings.append(relabelling)
            outside = count                                      # the rest of the world, which the external line runs through
            fermions = edges + [(outside, 0), (line - 1, outside)]
            external = (len(fermions) - 2, len(fermions) - 1)
            seen = set()
            for chords in _matchings(list(range(count))):
                name = min(tuple(sorted(tuple(sorted((r[a], r[b]))) for a, b in chords)) for r in relabellings)
                if name in seen:
                    continue
                seen.add(name)
                chords = list(chords)
                if skeleton:
                    # No tadpole, no polarization insertion, no self-energy insertion (a reducible diagram among them).
                    kept = all(_connected(count + 1, fermions + _without(chords, i, j))
                               for i in range(order) for j in range(i + 1)) \
                        and all(_connected(count + 1, _without(fermions, i, j) + chords)
                                for i in range(len(fermions)) for j in range(i) if (j, i) != external)
                else:
                    kept = _connected(count, edges + chords) \
                        and all(_connected(count, _without(edges, i) + chords) for i in range(line - 1))
                if kept:
                    found.append(Diagram(count, tuple(edges), 0, line - 1, tuple(chords), len(lengths)))
    return tuple(found)


def skeleton_diagrams(order):
    """The skeleton self-energy diagrams of `order` interaction lines: 1, 1, 6,
    49, 542 of them at the orders 1 to 5."""
    if order < 1:
        raise ValueError("the order of a diagram is at least 1")
    return list(_generate(int(order), True))


def irreducible_diagrams(order):
    """Every diagram of `order` interaction lines that cutting one fermion line
    does not separate: the skeletons, and those with insertions on their lines
    and with tadpoles. With the bare propagator on the lines their sum is the
    whole self-energy at that order, which is how the test suite holds the
    rules of this module to an exact diagonalization. A fermion line that
    closes on its own vertex is left out: it is the density of the mean field,
    which the propagator carries."""
    return list(_generate(int(order), False))


# -- the recursion over the sets of vertices that have happened

def _times(diagram, kinds):
    """The time of every vertex, 0, 1, .. in the order of the vertices, the two
    ends of an instantaneous line sharing one; None when a fermion line then
    joins two vertices of one time, which is an insertion of the mean field."""
    group = list(range(diagram.vertices))
    for (u, v), kind in zip(diagram.chords, kinds):
        if kind == "static":
            group[v] = group[u]
    names = sorted(set(group))
    time = [names.index(g) for g in group]
    return None if any(time[u] == time[v] for u, v in diagram.edges) else time


def _footprint(diagram, kinds, fixed, dimensions):
    """(the most entries the recursion holds at once, its multiplications) for
    one diagram, one assignment of lines and one choice of fixed labels, with
    `dimensions` = (particle labels, hole labels, boson labels); (0, 0) for a
    void assignment. Two consecutive layers of T(S) are alive together, with
    the denominator and the term being added."""
    time = _times(diagram, kinds)
    if time is None:
        return 0, 0
    time, count = np.array(time), max(time) + 1
    masks = np.arange(1 << count)
    inside = (masks[:, None] >> time[None, :]) & 1                           # [set, vertex]
    logarithm = np.zeros(len(masks))
    particle, hole, boson = (np.log(d) if d else -np.inf for d in dimensions)
    for u, v in diagram.edges:
        logarithm += np.where(inside[:, u] > inside[:, v], particle, 0.0) + np.where(inside[:, u] < inside[:, v], hole, 0.0)
    for c, (u, v) in enumerate(diagram.chords):
        if kinds[c] == "dynamic" and c not in fixed:
            logarithm += np.where(inside[:, u] != inside[:, v], boson, 0.0)
    sizes = np.exp(logarithm)
    population = np.array([bin(mask).count("1") for mask in masks])
    layers = [sizes[population == size] for size in range(count + 1)]
    held = max(a.sum() + b.sum() + 2.0 * b.max() for a, b in zip(layers, layers[1:]))
    # A step into a set of times multiplies over the labels present after it and those it ends, one line's worth at most
    # beyond the set's own; this is what balances the pieces between the workers.
    work = sum(layers[size].sum() * size for size in range(1, count + 1)) * max(dimensions)
    return held, work


class SkeletonSelfEnergy:
    """Sigma_nn(w) from the skeleton diagrams, for fermion levels `xi` (measured
    from a chemical potential inside the gap; the first `occupied` are filled),
    boson energies `bosons`, couplings `g[s, p, q]` and the instantaneous
    interaction `U[p, q, r, s] = (pq|rs)`. With `U = None` there are no
    instantaneous lines. At first order the instantaneous line is the exchange
    of the mean field and is left out, as it is in the Hartree-Fock levels.

    `memory` is the number of bytes an evaluation may hold at once over all of
    its worker processes (`MEMORY` by default). It does not change the value:
    a diagram whose widest set of simultaneous lines does not fit is evaluated
    one value at a time of the labels of as many retarded lines as it takes."""

    def __init__(self, xi, occupied, bosons, g, U=None, memory=None):
        self.xi, self.occupied = np.asarray(xi, dtype=float), int(occupied)
        self.bosons, self.g = np.asarray(bosons, dtype=float), np.asarray(g)
        self.U = None if U is None else np.asarray(U)
        self.filled, self.empty = np.arange(self.occupied), np.arange(self.occupied, len(self.xi))
        self.memory = MEMORY if memory is None else float(memory)
        self._paths, self._couplings, self._work = {}, {}, {}                # `_paths` serves `_orderings`

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_paths"], state["_couplings"], state["_work"] = {}, {}, {}   # found again in the worker
        return state

    def diagrams(self, order):
        """The diagrams of `order` this evaluates; a subclass of the test suite
        puts every irreducible diagram here."""
        return skeleton_diagrams(order)

    def work(self, order, processes=1):
        """The pieces an evaluation of `order` is made of, as (multiplications,
        diagram, kinds, fixed labels), each within memory / processes."""
        key = (order, processes)
        if key not in self._work:
            dimensions = (len(self.empty), len(self.filled), len(self.bosons))
            budget, items = self.memory / processes / 16.0, []               # entries, complex
            for diagram in self.diagrams(order):
                static = self.U is not None and order > 1
                for kinds in itertools.product(*[("dynamic", "static") if static else ("dynamic",)] * len(diagram.chords)):
                    retarded = [c for c, kind in enumerate(kinds) if kind == "dynamic"]
                    for count in range(len(retarded) + 1):
                        held, work = _footprint(diagram, kinds, tuple(retarded[:count]), dimensions)
                        if held <= budget:
                            break
                    else:
                        raise MemoryError(
                            f"a diagram of order {order} holds {16.0 * held / 2 ** 30:.3g} GiB with every label of its "
                            f"retarded lines fixed, and {16.0 * budget / 2 ** 30:.3g} GiB are allowed per process: "
                            "raise the memory or lower the number of bands on the lines of the diagrams")
                    if work:
                        for values in itertools.product(range(dimensions[2]), repeat=count):
                            items.append((work, diagram, kinds, tuple(zip(retarded[:count], values))))
            self._work[key] = items
        return self._work[key]

    def evaluate(self, n, frequency, order, parallel=None):
        """The sum of the diagrams of `order` for the mode n at the (complex)
        frequency. From the third order on it is spread over worker processes
        by default."""
        pool = _pool() if ((order >= 3) if parallel is None else parallel) else None
        if pool is None:
            return sum(self._diagram(diagram, kinds, n, frequency, fixed) for _, diagram, kinds, fixed in self.work(order))
        key = ("batches", order, pool._max_workers)
        if key not in self._work:
            # The longest pieces first, each to the batch that has the least so far; one batch carries the engine once.
            batches = [(0.0, index, []) for index in range(4 * pool._max_workers)]
            for work, diagram, kinds, fixed in sorted(self.work(order, pool._max_workers), key=lambda item: -item[0]):
                total, index, items = heapq.heappop(batches)
                items.append((diagram, kinds, fixed))
                heapq.heappush(batches, (total + work, index, items))
            self._work[key] = [items for _, _, items in batches if items]
        return sum(pool.map(_task, [(self, items, n, frequency) for items in self._work[key]]))

    def value_and_derivative(self, n, frequency, order, step=1e-6):
        """Sigma and d Sigma / dw at a real frequency from one complex evaluation:
        the expression is analytic, so the derivative is Im Sigma(w + i h) / h."""
        shifted = self.evaluate(n, complex(frequency, step), order)
        return float(np.real(shifted)), float(np.imag(shifted)) / step

    # -- one diagram, one assignment of instantaneous and retarded lines

    def _diagram(self, diagram, kinds, n, frequency, fixed=()):
        """The sum over the orderings of the vertex times by the recursion over
        the sets of times that have happened. `fixed` holds the labels of some
        retarded lines, as (line, label) pairs."""
        kinds, fixed = tuple(kinds), dict(fixed)
        edges, chords = diagram.edges, diagram.chords
        time = _times(diagram, kinds)
        if time is None:
            return 0.0
        count = max(time) + 1
        bit = [1 << t for t in time]
        # Every line as (bit of the time it leaves, bit of the time it enters, letter, its energies as a particle
        # or a boson, as a hole). A line with a fixed label has no letter and one energy.
        particle, hole = self.xi[self.empty], -self.xi[self.filled]
        lines = [(bit[u], bit[v], _LETTERS[e], particle, hole) for e, (u, v) in enumerate(edges)]
        for c, (u, v) in enumerate(chords):
            if kinds[c] == "dynamic":
                energy = self.bosons[fixed[c]] if c in fixed else self.bosons
                lines.append((bit[u], bit[v], "" if c in fixed else _LETTERS[len(edges) + c], energy, energy))
        # Every time as the couplings that happen at it: (which, fixed label, letters, the bits of the times at the far
        # ends of its fermion lines, outgoing and incoming alternately, None for the external line).
        outgoing, incoming = {u: e for e, (u, v) in enumerate(edges)}, {v: e for e, (u, v) in enumerate(edges)}

        def ends(vertex):
            leaves, enters = outgoing.get(vertex), incoming.get(vertex)
            return ("" if leaves is None else _LETTERS[leaves]) + ("" if enters is None else _LETTERS[enters]), \
                (None if leaves is None else bit[edges[leaves][1]], None if enters is None else bit[edges[enters][0]])

        # One coupling a time: a vertex, or both ends of an instantaneous line.
        happenings = [None] * count
        for c, (u, v) in enumerate(chords):
            if kinds[c] == "dynamic":
                for vertex in (u, v):
                    letters, far = ends(vertex)
                    letters = ("" if c in fixed else _LETTERS[len(edges) + c]) + letters
                    happenings[time[vertex]] = ("g", fixed.get(c), letters, far)
            else:
                (first, far_first), (second, far_second) = ends(u), ends(v)
                happenings[time[u]] = ("U", None, first + second, far_first + far_second)

        def present(mask):
            return [(letter, forward if mask & leaves else backward) for leaves, enters, letter, forward, backward in lines
                    if bool(mask & leaves) != bool(mask & enters)]

        full = (1 << count) - 1
        tensors, subscripts = {0: np.ones(())}, {0: ""}
        for size in range(1, count + 1):
            current = {}
            for mask in (m for m in range(1, full + 1) if bin(m).count("1") == size):
                cut = present(mask)
                subscripts[mask] = "".join(letter for letter, _ in cut)
                total = None
                for t in range(count):
                    before = mask ^ (1 << t)
                    if not mask >> t & 1:
                        continue
                    # A line is a hole when it enters a vertex that happens before the one it leaves: an outgoing line
                    # whose far end has happened, an incoming one whose far end has not. It is counted where it is
                    # made, at the vertex it enters.
                    which, label, letters, far = happenings[t]
                    natures = tuple(None if other is None else bool(before & other) == bool(side % 2)
                                    for side, other in enumerate(far))
                    holes = sum(1 for side, other in enumerate(far) if side % 2 and other is not None and not before & other)
                    term = self._contract(tensors[before], subscripts[before], self._coupling(n, which, label, natures),
                                          letters, subscripts[mask])
                    total = (-term if holes % 2 else term) if total is None else (total - term if holes % 2 else total + term)
                if cut:
                    energy = 0.0
                    for axis, energies in enumerate([energies for letter, energies in cut if letter]):
                        shape = [1] * len(subscripts[mask])
                        shape[axis] = -1
                        energy = energy + energies.reshape(shape)
                    energy = energy + sum(energies for letter, energies in cut if not letter)
                    spanned = bool(mask & bit[diagram.entry]) - bool(mask & bit[diagram.exit])
                    # The external line runs forward through this interval, backward, or not at all.
                    total = total / (spanned * frequency - energy)
                current[mask] = total
            tensors = current
        return (-2.0) ** diagram.loops * tensors[full]

    @staticmethod
    def _contract(tensor, present, coupling, letters, result):
        """T(S - t) with the coupling at t: the letters the two share are the
        lines that end at t and are summed, every other letter is a line
        present after it."""
        if tensor.size * coupling.size < 1 << 12:
            return np.einsum(f"{present},{letters}->{result}", tensor, coupling)
        shared = [letter for letter in letters if letter in present]
        product = np.tensordot(tensor, coupling, ([present.index(letter) for letter in shared],
                                                  [letters.index(letter) for letter in shared]))
        left = [letter for letter in present + letters if letter not in shared]
        return product if left == list(result) else product.transpose([left.index(letter) for letter in result])

    def _coupling(self, n, which, label, natures):
        """g[s, out, in] at one end of a retarded line (at the label s when it
        is fixed), U[out, in, out', in'] on an instantaneous one, every fermion
        axis over the empty modes (`natures` True: a particle), the filled
        ones (False: a hole), or at the external mode n (None)."""
        key = (n, which, label, natures)
        if key not in self._couplings:
            subsets = [None if nature is None else self.empty if nature else self.filled for nature in natures]
            if which == "U":
                picked = self._pick(self.U, subsets, n, leading=False)
            elif label is None:
                picked = self._pick(self.g, [None] + subsets, n)
            else:
                picked = self._pick(self.g[label], subsets, n, leading=False)
            self._couplings[key] = np.ascontiguousarray(picked)
        return self._couplings[key]

    def _orderings(self, diagram, kinds, n, frequency, part=0, parts=1):
        """The same number as `_diagram`, as the plain sum over the orderings of
        the vertex times with one contraction each: the reference that the
        recursion is held to."""
        group = list(range(diagram.vertices))                    # instantaneous lines merge their two times
        for (u, v), kind in zip(diagram.chords, kinds):
            if kind == "static":
                group[v] = group[u]
        groups = sorted(set(group))
        dynamic = [c for c, kind in enumerate(kinds) if kind == "dynamic"]
        letters = "abcdefghijklmnopqrstuvwxyz"
        edge_letter = {e: letters[i] for i, e in enumerate(range(len(diagram.edges)))}
        chord_letter = {c: letters[len(diagram.edges) + i] for i, c in enumerate(dynamic)}
        incoming, outgoing = {diagram.entry: None}, {diagram.exit: None}
        for e, (u, v) in enumerate(diagram.edges):
            outgoing[u], incoming[v] = e, e
        total = 0.0
        for position, order in enumerate(itertools.permutations(range(len(groups)))):
            if position % parts != part:
                continue
            time = {vertex: order[groups.index(group[vertex])] for vertex in range(diagram.vertices)}
            if any(time[u] == time[v] for u, v in diagram.edges):
                continue                                         # an equal-time line is a mean-field insertion
            hole = [time[u] > time[v] for u, v in diagram.edges]
            subsets = [self.filled if h else self.empty for h in hole]
            if any(len(subset) == 0 for subset in subsets):
                continue
            # One factor per interval between consecutive times, on the lines present in it only.
            operands, subscripts = [], []
            for slice_ in range(len(groups) - 1):
                present = [("edge", e) for e, (u, v) in enumerate(diagram.edges)
                           if min(time[u], time[v]) <= slice_ < max(time[u], time[v])]
                present += [("chord", c) for c in dynamic
                            if min(time[diagram.chords[c][0]], time[diagram.chords[c][1]]) <= slice_
                            < max(time[diagram.chords[c][0]], time[diagram.chords[c][1]])]
                energy = np.zeros([1] * len(present))
                for axis, (kind, index) in enumerate(present):
                    values = ((-1.0 if hole[index] else 1.0) * self.xi[subsets[index]] if kind == "edge" else self.bosons)
                    view = [1] * len(present)
                    view[axis] = len(values)
                    energy = energy + values.reshape(view)
                if time[diagram.entry] <= slice_ < time[diagram.exit]:
                    factor = 1.0 / (frequency - energy)
                elif time[diagram.exit] <= slice_ < time[diagram.entry]:
                    factor = 1.0 / (-frequency - energy)
                else:
                    factor = 1.0 / (-energy)
                operands.append(factor)
                subscripts.append("".join(edge_letter[i] if kind == "edge" else chord_letter[i] for kind, i in present))
            # The couplings: g[s, out, in] at both ends of a retarded line, U[out, in, out', in'] on an instantaneous one.

            def fermion(vertex, which):
                e = (outgoing if which == "out" else incoming)[vertex]
                return (None, None) if e is None else (edge_letter[e], subsets[e])

            for c, (u, v) in enumerate(diagram.chords):
                if kinds[c] == "dynamic":
                    for vertex in (u, v):
                        (lo, so), (li, si) = fermion(vertex, "out"), fermion(vertex, "in")
                        tensor = self._pick(self.g, [None, so, si], n)
                        operands.append(tensor)
                        subscripts.append(chord_letter[c] + (lo or "") + (li or ""))
                else:
                    picks = [fermion(u, "out"), fermion(u, "in"), fermion(v, "out"), fermion(v, "in")]
                    operands.append(self._pick(self.U, [s for _, s in picks], n, leading=False))
                    subscripts.append("".join(letter or "" for letter, _ in picks))
            expression = ",".join(subscripts) + "->"
            key = (expression, tuple(operand.shape for operand in operands))
            if key not in self._paths:                           # the contraction order is found once per term
                self._paths[key] = np.einsum_path(expression, *operands, optimize="greedy")[0]
            value = np.einsum(expression, *operands, optimize=self._paths[key])
            sign = (-1.0) ** (sum(hole) + diagram.loops) * 2.0 ** diagram.loops
            total = total + sign * value
        return total

    @staticmethod
    def _pick(tensor, subsets, n, leading=True):
        """`tensor` with every fermion axis restricted to its subset, or fixed at
        the external mode n where the subset is None. `leading` marks a first
        axis that is kept whole (the boson axis of g)."""
        picked = tensor
        axis = 1 if leading else 0
        for subset in (subsets[1:] if leading else subsets):
            if subset is None:
                picked = np.take(picked, n, axis=axis)
            else:
                picked = np.take(picked, subset, axis=axis)
                axis += 1
        return picked
