# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers.entanglement_complex -- a simplicial complex over qubits
built from their mutual information alone.

Definitions
-----------
* A density matrix rho of n qubits is a 2^n x 2^n Hermitian, positive
  semidefinite, unit-trace matrix in the computational basis. Qubit 0 is the
  most significant bit of the row index, which is the ordering of the
  Kronecker product rho_0 (x) rho_1 (x) ... (x) rho_{n-1}. Qubits 0, 1, 2, ...
  are printed as A, B, C, ...
* S(X) = -Tr rho_X ln rho_X is the von Neumann entropy of the reduced state
  rho_X of a set X of qubits, in nats (natural logarithm), as everywhere in
  tessera. The mutual information between two qubits is at most
  I_MAX = 2 ln 2.
* I(X:Y) = S(X) + S(Y) - S(XY) is the mutual information of X and Y.
* C = sum_i S(i) - S(all) is the total correlation of the network.

Pipeline
--------
1. The qubits start in input pairs (A,B), (C,D), ...: the first three pairs
   in the explicit states below, every further pair in a seeded random
   correlated state. An odd count leaves the last qubit alone in a random
   one-qubit state.
2. Round 1 applies one unitary to each input pair: explicit gates for the
   first three pairs, Haar-random unitaries for the rest.
3. Interactions apply SWAP^alpha to pairs of qubits, either on a round-robin
   schedule (every pair once per n - 1 rounds) or, with ``--timesteps T``,
   to one random pair per time slice, drawn from all pairs or from the
   nearest neighbours of an open chain.
4. Every pairwise I(X:Y) becomes an edge length (``--length``), optionally
   replaced by the shortest path through the edges (``--geodesic``).
5. The complex is the Vietoris-Rips filtration of those lengths: at scale r,
   a set of qubits is a simplex when every pairwise length within it is at
   most r. The qubits are given no coordinates. At every scale where the
   complex changes, the simplex counts and the Betti numbers (the ranks of
   the homology groups: b_0 counts connected components, b_1 independent
   loops, b_2 enclosed voids) are reported.

Reused rather than reimplemented: partial traces
(``tessera.quantum.partialTrace``), random correlated states
(``tessera.quantum.randomCorrelatedState``), entropies
(``tessera.quantum.MutualInformation.vonNeumannEntropy``), mutual information
(``tessera.quantum.mutualInformation``), the -ln(I/I_MAX) length
(``tessera.mesh.Edge.vanRaamsdonkLength``), homology
(``tessera.cobordism.ChainComplex``), the drawing palette
(``tessera.drivers.baryon_poles``), shortest paths (``scipy.sparse.csgraph``),
Haar unitaries (``scipy.stats.unitary_group``), and cliques and the circle
layout (``networkx``).
"""

import itertools
import math
import random
import warnings
from fractions import Fraction

import numpy as np

#: Largest supported qubit count: the state is a dense 2^n x 2^n complex
#: matrix (16 * 4^n bytes; 4 GB at n = 14).
MAX_QUBITS = 14
#: The maximum mutual information between two qubits, in nats.
I_MAX = 2.0 * math.log(2.0)
#: Agreement required of the step identity at every time slice (float64).
STEP_IDENTITY_TOL = 1e-9
#: Tolerance of the density-matrix checks on the explicit input states.
STATE_TOL = 1e-12


# ============================================================ input states

#: The explicit input states of the first three pairs, in the basis
#: |00>, |01>, |10>, |11>.
RHO_AB = np.array([[0.40, 0.10, 0.00, 0.05],
                   [0.10, 0.20, 0.02, 0.00],
                   [0.00, 0.02, 0.25, 0.03],
                   [0.05, 0.00, 0.03, 0.15]], dtype=complex)
#: 0.6 |Phi+><Phi+| + 0.4 I/4 (a triply degenerate spectrum).
RHO_CD = np.array([[0.40, 0.00, 0.00, 0.30],
                   [0.00, 0.10, 0.00, 0.00],
                   [0.00, 0.00, 0.10, 0.00],
                   [0.30, 0.00, 0.00, 0.40]], dtype=complex)
RHO_EF = np.array([[0.35, 0.05j, 0.00, 0.00],
                   [-0.05j, 0.25, 0.04, 0.00],
                   [0.00, 0.04, 0.25, 0.02j],
                   [0.00, 0.00, -0.02j, 0.15]], dtype=complex)
EXPLICIT_STATES = (RHO_AB, RHO_CD, RHO_EF)


# ============================================================ gates

HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
SQRT_ISWAP = np.array([[1, 0, 0, 0],
                       [0, 1 / math.sqrt(2.0), 1j / math.sqrt(2.0), 0],
                       [0, 1j / math.sqrt(2.0), 1 / math.sqrt(2.0), 0],
                       [0, 0, 0, 1]], dtype=complex)


def ry(theta):
    """The single-qubit rotation exp(-i theta Y / 2) about the Y axis."""
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def swap_power(alpha):
    """SWAP^alpha = exp(i pi alpha (SWAP - 1) / 2): alpha = 1/2 is sqrt(SWAP),
    alpha = 1 the full SWAP."""
    w = np.exp(1j * math.pi * float(alpha))
    return np.array([[1, 0, 0, 0],
                     [0, (1 + w) / 2, (1 - w) / 2, 0],
                     [0, (1 - w) / 2, (1 + w) / 2, 0],
                     [0, 0, 0, 1]], dtype=complex)


#: The round-1 unitaries of the first three pairs.
EXPLICIT_UNITARIES = (
    ("CNOT.(H x I)", CNOT @ np.kron(HADAMARD, np.eye(2))),
    ("CZ.(Ry(pi/3) x Ry(pi/5))", CZ @ np.kron(ry(math.pi / 3), ry(math.pi / 5))),
    ("sqrt(iSWAP)", SQRT_ISWAP),
)


def _require_unitary(name, U):
    U = np.asarray(U, dtype=complex)
    defect = float(np.abs(U.conj().T @ U - np.eye(len(U))).max())
    if defect > 1e-12:
        raise ValueError("%s is not unitary (defect %.3g)" % (name, defect))


def _require_density_matrix(name, rho):
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("%s must be a square matrix" % name)
    if np.abs(rho - rho.conj().T).max() > STATE_TOL:
        raise ValueError("%s is not Hermitian" % name)
    if abs(np.trace(rho) - 1) > STATE_TOL:
        raise ValueError("%s does not have unit trace" % name)
    if np.linalg.eigvalsh(rho).min() < -STATE_TOL:
        raise ValueError("%s is not positive semidefinite" % name)


# ============================================================ schedules

def round_robin(n):
    """A 1-factorisation of the complete graph on n qubits (circle method):
    a list of rounds, each a list of disjoint pairs, so that every pair of
    qubits meets exactly once. The first round is the input pairing (0,1),
    (2,3), ...; an odd count leaves one qubit out of each round."""
    m = n + (n % 2)
    items = list(range(m))
    matchings = []
    for _ in range(m - 1):
        matchings.append([(items[i], items[m - 1 - i]) for i in range(m // 2)])
        items = [items[0], items[-1]] + items[1:-1]
    perm, k = {}, 0
    for a, b in matchings[0]:
        if a < n and b < n:
            perm[a], perm[b] = 2 * k, 2 * k + 1
            k += 1
        else:
            perm[a if a < n else b] = n - 1
    if n % 2:
        perm[n] = n
    return [sorted(tuple(sorted((perm[a], perm[b]))) for a, b in matching
                   if a < n and b < n)
            for matching in matchings]


def candidate_pairs(n, pair_mode):
    """The pairs a time slice may draw: every pair ("all"), or the nearest
    neighbours of the open chain 0-1-...-(n-1) ("chain")."""
    if pair_mode == "chain":
        return [(i, i + 1) for i in range(n - 1)]
    if pair_mode == "all":
        return list(itertools.combinations(range(n), 2))
    raise ValueError("pair_mode must be 'all' or 'chain'")


# ============================================================ the network

class QubitNetwork:
    """The global state of n qubits as a (2,)*2n tensor: ket axes 0..n-1,
    bra axes n..2n-1, qubit 0 first (the most significant bit)."""

    def __init__(self, n):
        if not 2 <= n <= MAX_QUBITS:
            raise ValueError("the qubit count must be in [2, %d]" % MAX_QUBITS)
        self.n = n
        self.names = [chr(65 + i) for i in range(n)]
        self.rho = None

    def set_product(self, factors):
        """factors: (qubit indices, density matrix) pairs covering 0..n-1 in order."""
        covered, out = [], np.ones((1, 1), dtype=complex)
        for qubits, matrix in factors:
            covered.extend(qubits)
            out = np.kron(out, np.asarray(matrix, dtype=complex))
        if covered != list(range(self.n)):
            raise ValueError("factors must cover the qubits in order")
        self.rho = out.reshape((2,) * (2 * self.n))

    def apply_gate(self, U, pair):
        """rho -> U rho U^dagger with the 4 x 4 gate U on the qubits in `pair`
        (the first listed qubit is U's first factor)."""
        n = self.n
        qi, qj = pair
        u = np.asarray(U, dtype=complex).reshape(2, 2, 2, 2)
        axes = list(range(2 * n))                 # integer labels: no 26-letter limit
        new_i, new_j = 2 * n, 2 * n + 1
        out = axes.copy()
        out[qi], out[qj] = new_i, new_j
        rho = np.einsum(u, [new_i, new_j, qi, qj], self.rho, axes, out)
        out = axes.copy()
        out[n + qi], out[n + qj] = new_i, new_j
        self.rho = np.einsum(np.conjugate(u), [new_i, new_j, n + qi, n + qj], rho, axes, out)

    def matrix(self):
        """The 2^n x 2^n density matrix."""
        return self.rho.reshape(2 ** self.n, 2 ** self.n)

    def reduced(self, keep):
        """The reduced state on the qubits in `keep`, in their listed order."""
        from tessera import quantum
        return quantum.partialTrace(self.matrix(), self.n, [int(q) for q in keep])

    def entropy(self, keep):
        """S(keep) in nats."""
        from tessera import quantum
        return quantum.MutualInformation.vonNeumannEntropy(self.reduced(keep))

    def mutual_information(self, pair):
        """I(i:j) in nats for the two qubits in `pair`."""
        from tessera import quantum
        return quantum.mutualInformation(self.reduced(pair), 2, 2)

    def replace_by_marginals(self):
        """Replace the state by the product of its one-qubit marginals."""
        self.set_product([((q,), self.reduced((q,))) for q in range(self.n)])

    def trace(self):
        return float(np.trace(self.matrix()).real)


# ============================================================ simulation

def build_inputs(n, seed):
    """Input pairs, their states and round-1 unitaries, and (odd n) the
    state of the last qubit. Pairs beyond the explicit three draw their
    states from `tessera.quantum.randomCorrelatedState` and their unitaries
    from `scipy.stats.unitary_group`, seeded from `seed`."""
    from scipy.stats import unitary_group
    from tessera import quantum
    pairs = [(2 * k, 2 * k + 1) for k in range(n // 2)]
    seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(2 * len(pairs) + 1)]
    states, unitaries = {}, {}
    for k, pair in enumerate(pairs):
        if k < len(EXPLICIT_STATES):
            states[pair] = EXPLICIT_STATES[k]
            unitaries[pair] = EXPLICIT_UNITARIES[k]
        else:
            states[pair] = quantum.randomCorrelatedState(2, seeds[2 * k])
            unitaries[pair] = ("Haar(seed %d, pair %d)" % (seed, k),
                               unitary_group.rvs(4, random_state=seeds[2 * k + 1]))
    single = quantum.randomCorrelatedState(1, seeds[-1]) if n % 2 else None
    return pairs, states, unitaries, single


def interaction_label(alpha):
    return "sqrt(SWAP)" if Fraction(alpha) == Fraction(1, 2) else "SWAP^(%s)" % alpha


def _silent(_line):
    pass


def simulate(n, rounds=None, alpha="1/2", seed=0, carry="global", timesteps=None,
             pair_mode="all", regions=False, log=_silent):
    """Prepare the inputs, apply round 1, then either `rounds` round-robin
    rounds or `timesteps` random time slices; return the final state's
    entropies and pairwise mutual information. `log` receives the report
    lines (the command line passes `print`)."""
    from tessera import quantum
    net = QubitNetwork(n)
    names = net.names
    alpha = Fraction(alpha)
    interaction = swap_power(alpha)
    label = interaction_label(alpha)
    schedule = round_robin(n)
    if rounds is None:
        rounds = max(len(schedule) - 1, 0)
    if timesteps is not None and carry == "marginals":
        raise ValueError("time slices use the global state; carry='marginals' is not supported")
    pairs, states, unitaries, single = build_inputs(n, seed)
    for name, U in list(unitaries.values()) + [(label, interaction)]:
        _require_unitary(name, U)

    log("INPUT STATES (%d qubits %s..%s; entropies in nats)" % (n, names[0], names[-1]))
    S_global = 0.0
    for pair in pairs:
        rho = states[pair]
        _require_density_matrix("rho_" + "".join(names[q] for q in pair), rho)
        S_pair = quantum.MutualInformation.vonNeumannEntropy(rho)
        S_global += S_pair
        log("  rho_%s%s: eigenvalues %s   I = %.6f"
            % (names[pair[0]], names[pair[1]],
               ", ".join("%.6f" % v for v in np.linalg.eigvalsh(rho)),
               quantum.mutualInformation(rho, 2, 2)))
    if single is not None:
        S_global += quantum.MutualInformation.vonNeumannEntropy(single)
    log("  S_global = %.6f nats (constant: every step is unitary)" % S_global)

    factors = [(pair, states[pair]) for pair in pairs]
    if single is not None:
        factors.append(((n - 1,), single))
    net.set_product(factors)

    log("ROUND 1: local unitaries on the input pairs")
    for pair in pairs:
        name, U = unitaries[pair]
        net.apply_gate(U, pair)
        log("  %s%s after %s: I = %.6f" % (names[pair[0]], names[pair[1]], name,
                                          net.mutual_information(pair)))

    slices = order = None
    if timesteps is not None:
        rng = random.Random(seed + 1000)          # separate stream from the inputs
        slices, order = run_slices(net, interaction, label, timesteps, rng, S_global,
                                   pair_mode, regions, log)
        rounds = 0

    for r in range(rounds):
        matching = schedule[(r + 1) % len(schedule)]
        log("ROUND %d: %s on %s   [carry = %s]"
            % (r + 2, label, ", ".join(names[a] + names[b] for a, b in matching), carry))
        if carry == "marginals":
            net.replace_by_marginals()
        for pair in matching:
            before = net.mutual_information(pair)
            net.apply_gate(interaction, pair)
            log("  %s%s: I = %.6f -> %.6f" % (names[pair[0]], names[pair[1]], before,
                                             net.mutual_information(pair)))

    trace = net.trace()
    if abs(trace - 1.0) > 1e-10:
        raise RuntimeError("the trace drifted to %r" % trace)
    S = np.array([net.entropy((q,)) for q in range(n)])
    MI = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        MI[i, j] = MI[j, i] = net.mutual_information((i, j))
    return {"net": net, "names": names, "S": S, "MI": MI, "S_global": S_global,
            "interaction": label, "slices": slices, "order": order}


# ============================================================ time slices

def analyse_slice(net, t, pair, prev, S_global):
    """Entropies, pairwise mutual information and the entropy budget of the
    current state. Only the 2n - 3 pairs touching the interacting qubits are
    recomputed; the rest are copied from the previous slice. The step
    identity S(i') + S(j') - S(i) - S(j) = I(i':j') - I(i:j) holds exactly for
    a unitary on (i, j); the temporal residual of a participant is
    S_i(t - 1) - I(i':j'), of a bystander S_k(t - 1); the slice's event-time
    increment is the participants' mean residual."""
    n = net.n
    changed = set(range(n)) if prev is None else set(pair)
    S = dict(prev["S_dict"]) if prev else {}
    for q in changed:
        S[q] = net.entropy((q,))
    I_pairs = dict(prev["I_dict"]) if prev else {}
    for p in itertools.combinations(range(n), 2):
        if changed & set(p):
            I_pairs[p] = net.mutual_information(p)
    MI = np.zeros((n, n))
    for (i, j), v in I_pairs.items():
        MI[i, j] = MI[j, i] = v
    sl = {"t": t, "pair": pair, "S_dict": S, "I_dict": I_pairs,
          "S": np.array([S[q] for q in range(n)]), "MI": MI,
          "sumS": sum(S.values()), "sumI": sum(I_pairs.values())}
    sl["C"] = sl["sumS"] - S_global
    if prev is None:
        sl.update(residual={q: None for q in range(n)}, dtau=0.0, tau=0.0,
                  dS=None, dI=None, identity_err=None)
        return sl
    i, j = pair
    I_new, I_old = I_pairs[pair], prev["I_dict"][pair]
    residual = {q: prev["S_dict"][q] - (I_new if q in pair else 0.0) for q in range(n)}
    sl["residual"] = residual
    sl["dtau"] = (residual[i] + residual[j]) / 2.0
    sl["tau"] = prev["tau"] + sl["dtau"]
    sl["dS"] = (S[i] - prev["S_dict"][i]) + (S[j] - prev["S_dict"][j])
    sl["dI"] = I_new - I_old
    sl["identity_err"] = abs(sl["dS"] - sl["dI"])
    return sl


def run_slices(net, interaction, label, timesteps, rng, S_global, pair_mode="all",
               regions=False, log=_silent):
    """Apply `timesteps` interactions, one per slice, each to a pair drawn by
    `rng` from `candidate_pairs`; analyse every slice. Returns the slices and
    their indices ordered by event time."""
    names = net.names
    pairs = candidate_pairs(net.n, pair_mode)
    if regions:
        from tessera.drivers import entanglement_regions as reg
    log("TIME SLICES: %d interactions with %s (%s pairs: %d candidates)"
        % (timesteps, label, pair_mode, len(pairs)))
    slices = [analyse_slice(net, 0, None, None, S_global)]
    if regions:
        slices[0]["regions"] = reg.slice_summary(net, S_global)
    _log_slice(slices[0], names, S_global, log)
    for t in range(1, timesteps + 1):
        pair = rng.choice(pairs)
        net.apply_gate(interaction, pair)
        sl = analyse_slice(net, t, pair, slices[-1], S_global)
        if sl["identity_err"] > STEP_IDENTITY_TOL:
            raise RuntimeError("step identity failed at t=%d: %.3g" % (t, sl["identity_err"]))
        if regions:
            sl["regions"] = reg.slice_summary(net, S_global)
        slices.append(sl)
        _log_slice(sl, names, S_global, log)
    order = sorted(range(len(slices)), key=lambda k: (slices[k]["tau"], k))
    log("  slices ordered by event time: " + " < ".join("t=%d" % k for k in order))
    return slices, order


def _log_slice(sl, names, S_global, log):
    if sl["pair"] is None:
        head = "  slice %3d  tau = %+.6f   (prepared state)" % (sl["t"], sl["tau"])
    else:
        i, j = sl["pair"]
        head = ("  slice %3d  tau = %+.6f   event (%s,%s)   dtau = %+.6f   "
                "|dS - dI| = %.2e" % (sl["t"], sl["tau"], names[i], names[j], sl["dtau"],
                                      sl["identity_err"]))
    log(head)
    log("             sum S = %.6f   C = sum S - S_global = %.6f   sum of pairwise I = %.6f"
        % (sl["sumS"], sl["C"], sl["sumI"]))
    if "regions" in sl:
        r = sl["regions"]
        log("             C by order: " + "  ".join(
            "c_%d = %+.4f" % (k, c) for k, c in enumerate(r["shares"], 2))
            + "   MMI violated on %d/%d triples" % (r["mmi_violations"], r["mmi_triples"]))


# ============================================================ edge lengths

LENGTH_MODES = ("inverse", "log", "mi", "vi")
LENGTH_LABEL = {"inverse": "1/I(X:Y)", "log": "-ln(I(X:Y)/I_max)", "mi": "I(X:Y)",
                "vi": "S(X|Y)+S(Y|X)"}


def length_label(mode, geodesic=False):
    return ("shortest path over %s" % LENGTH_LABEL[mode]) if geodesic else LENGTH_LABEL[mode]


def target_lengths(MI, mode, floor, S=None, geodesic=False):
    """Edge lengths D and edge indicator W from the mutual information MI.

    A pair with I <= floor has no edge (W = 0). The modes:
      inverse  1/I(X:Y)
      log      -ln(I(X:Y)/I_MAX), via tessera.mesh.Edge.vanRaamsdonkLength
      mi       I(X:Y)
      vi       S(X|Y) + S(Y|X) = S(X) + S(Y) - 2 I(X:Y) (needs S, the
               one-qubit entropies); negative values, which a strongly
               entangled pair gives, are clipped to 0 with a warning
    With `geodesic`, every length is replaced by the shortest path through
    the edges, and a pair joined by some path gets an edge even when its own
    I is at or below the floor; pairs in separate components stay without
    one. The result is scaled to mean edge length 1, or is (None, W) when no
    pair has an edge.
    """
    import tessera
    MI = np.asarray(MI, dtype=float)
    n = len(MI)
    off = ~np.eye(n, dtype=bool)
    W = ((MI > floor) & off).astype(float)
    if W.sum() == 0:
        return None, W
    D = np.zeros_like(MI)
    present = W > 0
    if mode == "inverse":
        D[present] = 1.0 / MI[present]
    elif mode == "log":
        # epsilon = 0 switches off the edge's length cap: below-floor pairs
        # are already without an edge
        D[present] = [tessera.mesh.Edge.vanRaamsdonkLength(float(I), I_MAX, 0.0)
                      for I in MI[present]]
    elif mode == "mi":
        D[present] = MI[present]
    elif mode == "vi":
        if S is None:
            raise ValueError("length mode 'vi' needs the one-qubit entropies S")
        S = np.asarray(S, dtype=float)
        D[present] = (S[:, None] + S[None, :] - 2.0 * MI)[present]
        negative = (D < 0) & present
        if negative.any():
            warnings.warn("%d pair(s) have S(X|Y) + S(Y|X) < 0 (minimum %.4g); clipped to 0"
                          % (negative.sum() // 2, D[negative].min()), stacklevel=2)
            D[negative] = 0.0
    else:
        raise ValueError("length mode must be one of %s" % (LENGTH_MODES,))
    if geodesic:
        from scipy.sparse.csgraph import csgraph_from_dense, shortest_path
        # inf marks a missing edge, so zero-length edges stay edges
        G = shortest_path(csgraph_from_dense(np.where(present, D, np.inf), null_value=np.inf),
                          directed=False)
        W = (np.isfinite(G) & off).astype(float)
        D = np.where(W > 0, G, 0.0)
    return D / D[W > 0].mean(), W


def triangle_inequality_report(D, W, names):
    """Check every edge of every triangle of edges against the path through
    the triangle's third vertex. Returns (checks, violations, triples with a
    violation, worst) with worst a list of (edge / path, edge, path) sorted
    by the ratio. An edge equal to its path, as shortest paths make them, is
    no violation."""
    n = len(D)
    checks = violations = bad_triples = 0
    worst = []
    for i, j, k in itertools.combinations(range(n), 3):
        if not (W[i, j] and W[i, k] and W[j, k]):
            continue
        bad = False
        for (a, b), c in (((i, j), k), ((i, k), j), ((j, k), i)):
            path = D[a, c] + D[c, b]
            checks += 1
            if D[a, b] > path * (1 + 1e-12):
                ratio = D[a, b] / path if path > 0 else np.inf
                violations += 1
                bad = True
                worst.append((ratio, names[a] + names[b],
                              "%s%s+%s%s" % (names[a], names[c], names[c], names[b])))
        bad_triples += bad
    worst.sort(reverse=True)
    return checks, violations, bad_triples, worst


# ============================================================ the complex

def rips_filtration(D, W):
    """The Vietoris-Rips filtration of the edge lengths D on the edges W.

    A set of qubits is a simplex from the scale r at which its longest
    pairwise length appears (its birth; 0 for a single qubit), so the
    simplices are the cliques of the edge graph. For every distinct birth r
    the complex at r is passed to `tessera.cobordism.ChainComplex.fromCells`
    for its simplex counts (the f-vector: vertices, edges, triangles, ...)
    and Betti numbers over the rationals.
    """
    import networkx as nx
    from tessera import cobordism
    n = len(W)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((i, j) for i, j in itertools.combinations(range(n), 2) if W[i, j] > 0)
    cliques = [tuple(sorted(c)) for c in nx.enumerate_all_cliques(graph)]
    births = [max((D[a, b] for a, b in itertools.combinations(c, 2)), default=0.0)
              for c in cliques]
    levels = []
    for r in sorted(set(births)):
        cells = [list(c) for c, b in zip(cliques, births) if b <= r]
        complex_ = cobordism.ChainComplex.fromCells(cells)
        levels.append({"scale": float(r),
                       "f_vector": [int(x) for x in complex_.fVector()],
                       "betti": [int(x) for x in complex_.bettiNumbers()]})
    return {"cliques": cliques, "births": [float(b) for b in births], "levels": levels}


# ============================================================ figures

#: Categorical slots beyond the two of `baryon_poles.SERIES`, in a fixed order.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948")
FILL = "#cde2fb"


def _snapshot_levels(levels, count=4):
    """Up to `count` levels, preferring those where the Betti numbers change."""
    changes = [k for k, lv in enumerate(levels)
               if k == 0 or lv["betti"] != levels[k - 1]["betti"]]
    picks = changes if len(changes) >= count else sorted(
        set(changes) | set(np.linspace(0, len(levels) - 1, count).round().astype(int)))
    if len(picks) > count:
        picks = [picks[i] for i in np.linspace(0, len(picks) - 1, count).round().astype(int)]
    return [levels[k] for k in picks]


def draw_filtration(filtration, names, title, save=None):
    """Betti numbers and simplex counts against the scale, and the complex at
    a few scales drawn on a circle. The circle only displays which simplices
    exist; the positions carry no information."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import Polygon
    from tessera.drivers import baryon_poles as bp

    levels = filtration["levels"]
    scales = [lv["scale"] for lv in levels]
    snaps = _snapshot_levels(levels)
    fig = plt.figure(figsize=(14, 8.5), facecolor=bp.SURFACE)
    grid = fig.add_gridspec(2, len(snaps), height_ratios=[1.0, 1.15], hspace=0.35,
                            wspace=0.2, left=0.06, right=0.98, top=0.9, bottom=0.04)
    top = grid[0, :].subgridspec(1, 2, wspace=0.18)

    ax = fig.add_subplot(top[0])
    bp.style_axis(ax)
    top_dim = max(len(lv["betti"]) for lv in levels)
    for k in range(top_dim):
        values = [lv["betti"][k] if k < len(lv["betti"]) else 0 for lv in levels]
        ax.step(scales, values, where="post", color=CATEGORICAL[k % len(CATEGORICAL)],
                linewidth=2, label="b_%d" % k)
    ax.set_xlabel("scale r (edge length)", color=bp.INK_MUTED)
    ax.set_ylabel("Betti number", color=bp.INK_MUTED)
    ax.set_title("homology of the Vietoris-Rips complex", loc="left", color=bp.INK)
    ax.legend(frameon=False, labelcolor=bp.INK_MUTED)

    ax = fig.add_subplot(top[1])
    bp.style_axis(ax)
    top_dim = max(len(lv["f_vector"]) for lv in levels)
    for k in range(top_dim):
        values = [lv["f_vector"][k] if k < len(lv["f_vector"]) else 0 for lv in levels]
        ax.step(scales, values, where="post", color=CATEGORICAL[k % len(CATEGORICAL)],
                linewidth=2, label="%d-simplices" % k)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("scale r (edge length)", color=bp.INK_MUTED)
    ax.set_ylabel("simplices", color=bp.INK_MUTED)
    ax.set_title("simplex counts", loc="left", color=bp.INK)
    ax.legend(frameon=False, labelcolor=bp.INK_MUTED, ncol=2)

    n = len(names)
    pos = nx.circular_layout(range(n))
    xy = np.array([pos[q] for q in range(n)])
    for col, lv in enumerate(snaps):
        ax = fig.add_subplot(grid[1, col])
        ax.set_aspect("equal")
        ax.axis("off")
        cells = [c for c, b in zip(filtration["cliques"], filtration["births"])
                 if b <= lv["scale"]]
        for c in cells:
            if len(c) == 3:
                ax.add_patch(Polygon(xy[list(c)], closed=True, facecolor=FILL,
                                     edgecolor="none", alpha=0.35, zorder=1))
        for c in cells:
            if len(c) == 2:
                ax.plot(*xy[list(c)].T, color=bp.INK_MUTED, linewidth=1.2, zorder=2)
        ax.scatter(xy[:, 0], xy[:, 1], s=90, color=bp.INK, zorder=3)
        for q in range(n):
            ax.annotate(names[q], xy[q] * 1.18, ha="center", va="center", color=bp.INK)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_title("r = %.4g   b = (%s)" % (lv["scale"], ", ".join(map(str, lv["betti"]))),
                     color=bp.INK, fontsize=10)
    fig.suptitle(title, x=0.02, ha="left", color=bp.INK)
    if save:
        fig.savefig(save, dpi=150, facecolor=bp.SURFACE)
    return fig


# ============================================================ command line

def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("not JSON serialisable: %r" % type(value))


def main(argv=None):
    """Simulate the network, build the complex from mutual information alone,
    and report it: a text report on standard output, or with ``--json`` the
    record as JSON; figures with ``--save`` (and in a window unless
    ``--no-show``)."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="tessera.drivers.entanglement_complex",
        description="Simulate an n-qubit network under pairwise SWAP^alpha "
                    "interactions and build a simplicial complex over the qubits "
                    "from their mutual information alone (no embedding): the "
                    "Vietoris-Rips filtration of the edge lengths, with simplex "
                    "counts and Betti numbers at every scale. Entropies are in nats.")
    parser.add_argument("--qubits", type=int, default=6, help="number of qubits (default 6)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="round-robin rounds after round 1 (default n - 2: every pair "
                             "meets once)")
    parser.add_argument("--swap-power", default="1/2",
                        help="exponent alpha of the SWAP^alpha interaction (default 1/2)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the random inputs and of the slice events (default 0)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="instead of rounds, apply this many interactions, one random "
                             "pair per time slice")
    parser.add_argument("--pairs", choices=("all", "chain"), default="all",
                        help="with --timesteps: draw pairs from all pairs or from nearest "
                             "neighbours of an open chain")
    parser.add_argument("--carry", choices=("global", "marginals"), default="global",
                        help="carry the full state between rounds (global) or only each "
                             "qubit's reduced state (marginals)")
    parser.add_argument("--length", choices=LENGTH_MODES, default="inverse",
                        help="edge length from I(X:Y): 1/I (inverse, default), "
                             "-ln(I/I_max) (log), I (mi), or S(X)+S(Y)-2I (vi)")
    parser.add_argument("--geodesic", action="store_true",
                        help="replace every length by the shortest path through the edges; "
                             "pairs at or below the floor get an edge when a path joins them")
    parser.add_argument("--mi-floor", type=float, default=1e-12,
                        help="pairs with I(X:Y) at or below this get no edge (default 1e-12)")
    parser.add_argument("--regions", action="store_true",
                        help="also report the co-information ledger and monogamy of mutual "
                             "information over every subset of the qubits")
    parser.add_argument("--json", action="store_true",
                        help="write the record as JSON on standard output")
    parser.add_argument("--save", default=None,
                        help="write the filtration figure to this path (and, with "
                             "--regions, the ledger to <stem>_regions.png)")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args(argv)

    log = _silent if args.json else print
    if args.regions:
        from tessera.drivers import entanglement_regions as reg
        if args.qubits > reg.MAX_QUBITS:
            parser.error("--regions supports at most %d qubits" % reg.MAX_QUBITS)
    if args.timesteps is not None and args.carry == "marginals":
        parser.error("--timesteps uses the global state; --carry marginals is not supported")
    try:
        Fraction(args.swap_power)
    except (ValueError, ZeroDivisionError):
        parser.error("--swap-power must be a number such as 1/2 or 0.25")

    result = simulate(args.qubits, args.rounds, args.swap_power, args.seed, args.carry,
                      args.timesteps, args.pairs, args.regions, log)
    names, MI, S = result["names"], result["MI"], result["S"]
    n = len(names)

    log("FINAL STATE")
    for q in range(n):
        log("  S(%s) = %.6f" % (names[q], S[q]))
    log("  I(X:Y):" + "".join("%11s" % q for q in names))
    for i in range(n):
        log("  %7s" % names[i] + "".join(
            "%11s" % ("" if i == j else "%.6f" % MI[i, j]) for j in range(n)))

    D, W = target_lengths(MI, args.length, args.mi_floor, S, args.geodesic)
    label = length_label(args.length, args.geodesic)
    record = {"qubits": n, "entropy_unit": "nats", "S_global": result["S_global"],
              "S": S, "MI": MI, "interaction": result["interaction"],
              "length": {"mode": args.length, "geodesic": args.geodesic,
                         "floor": args.mi_floor, "label": label}}
    if D is None:
        log("GEOMETRY: no pair has I > %g; the complex is %d isolated vertices"
            % (args.mi_floor, n))
        filtration = {"cliques": [(q,) for q in range(n)], "births": [0.0] * n,
                      "levels": [{"scale": 0.0, "f_vector": [n], "betti": [n]}]}
        tri = (0, 0, 0, [])
    else:
        log("GEOMETRY: edge length = %s, scaled to mean 1" % label)
        for i, j in itertools.combinations(range(n), 2):
            if W[i, j] > 0:
                via = "   via shortest path (own I <= floor)" if MI[i, j] <= args.mi_floor else ""
                log("  %s%s  %.6f%s" % (names[i], names[j], D[i, j], via))
            else:
                why = "no path of pairs with I > " if args.geodesic else "I <= "
                log("  %s%s  no edge (%s%g)" % (names[i], names[j], why, args.mi_floor))
        tri = triangle_inequality_report(D, W, names)
        log("  triangle inequality: %d/%d edge-against-path checks violated; %d triples "
            "affected" % (tri[1], tri[0], tri[2]))
        for ratio, edge, path in tri[3][:8]:
            log("    %s = %.3f x (%s)" % (edge, ratio, path))
        filtration = rips_filtration(D, W)
        record["lengths"] = D
        record["edges"] = W
    record["triangle_inequality"] = {"checks": tri[0], "violations": tri[1],
                                     "triples": tri[2]}
    record["filtration"] = filtration["levels"]

    log("VIETORIS-RIPS FILTRATION (a set is a simplex once all its pairwise lengths "
        "are <= r)")
    log("  %10s   %-28s %s" % ("scale r", "simplices by dimension", "Betti numbers"))
    for lv in filtration["levels"]:
        log("  %10.6f   %-28s %s" % (lv["scale"], lv["f_vector"], lv["betti"]))

    regions = None
    if args.regions:
        regions = reg.analyse(result["net"], result["S_global"])
        reg.print_report(regions, names, log)
        record["regions"] = {"shares": regions["shares"], "C": regions["C"],
                             "mmi_violations": regions["mmi"]["violations"],
                             "mmi_triples": regions["mmi"]["triples"],
                             "mmi_worst": regions["mmi"]["worst"]}
    if result["slices"] is not None:
        record["slices"] = [{"t": s["t"], "pair": s["pair"], "tau": s["tau"], "C": s["C"],
                             "sumI": s["sumI"]} for s in result["slices"]]

    if args.json:
        json.dump(record, sys.stdout, indent=2, sort_keys=True, default=_json_default)
        sys.stdout.write("\n")
        return 0

    if args.save or not args.no_show:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        stem = args.save.rsplit(".", 1)[0] if args.save else None
        draw_filtration(filtration, names,
                        "%d qubits: Vietoris-Rips filtration of edge length = %s" % (n, label),
                        args.save)
        if regions is not None:
            reg.draw(regions, result["slices"], names,
                     "%s_regions.png" % stem if stem else None)
        if not args.no_show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
