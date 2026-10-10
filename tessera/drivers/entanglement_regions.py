# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers.entanglement_regions -- region entropies of an n-qubit
network: the co-information ledger and monogamy of mutual information.

Every subset A of the qubits gets its von Neumann entropy S(A) in nats
(2^n - 1 numbers; the whole set has S_global), from
``tessera.quantum.partial_trace`` and
``tessera.quantum.MutualInformation.von_neumann_entropy``. The mutual
information between two disjoint subsets is I(A:B) = S(A) + S(B) - S(AB), so
these numbers carry all (3^n - 2^(n+1) + 1)/2 block mutual informations.
Only mutual information is used: adding sum_{i in A} c_i to every S(A)
changes no I(A:B) and nothing below.

Subsets are bitmasks: qubit q is bit q.

Co-information ledger
    I_T = -sum_{U subset of T} (-1)^|U| S(U)   (|T| = 2: I(a:b); |T| = 3: I3)
    I(A:B) = sum over T inside A u B meeting both A and B of (-1)^|T| I_T
    C = sum_i S(i) - S_global = sum_{|T| >= 2} (-1)^|T| I_T
  c_k = (-1)^k sum_{|T| = k} I_T is the share of the total correlation C
  carried by k-body terms; the c_k carry signs and sum to C exactly.

Monogamy of mutual information (MMI)
    I3(A:B:C) = I(A:B) + I(A:C) - I(A:BC) <= 0
  checked on every unordered triple of disjoint non-empty subsets (Hayden,
  Headrick, Maloney 2013). A violation means the correlations of A with B
  and with C, counted separately, exceed its correlation with B and C
  together: B and C carry the same information about A.
"""

import math

import numpy as np

#: Largest qubit count: the monogamy check visits 4^n labellings.
MAX_QUBITS = 10
#: Agreement required of the ledger identity sum_k c_k = C.
CHECK_TOL = 1e-9
#: I3 above this counts as a violation of monogamy (float64 entropies).
MMI_TOL = 1e-10


# ============================================================ subsets as bitmasks

def members(mask, n):
    return tuple(q for q in range(n) if mask >> q & 1)


def label(mask, names):
    return "".join(names[q] for q in members(mask, len(names)))


def popcounts(n):
    return np.bitwise_count(np.arange(2 ** n, dtype=np.uint64)).astype(int)


def lowest_bit(masks):
    return masks & -masks


# ============================================================ region entropies

def region_entropies(net, S_global):
    """S[mask] for every subset of the qubits of `net` (a
    `tessera.drivers.entanglement_complex.QubitNetwork`); S[0] = 0 and
    S[all] = S_global."""
    n = net.n
    S = np.zeros(2 ** n)
    for mask in range(1, 2 ** n - 1):
        S[mask] = net.entropy(members(mask, n))
    S[-1] = float(S_global)
    return S


# ============================================================ co-information ledger

def coinformation(S, n):
    """I_T = -sum_{U subset of T} (-1)^|U| S(U) for every T (a subset-sum transform)."""
    g = -((-1.0) ** popcounts(n)) * S
    masks = np.arange(2 ** n)
    for q in range(n):
        has = masks[masks >> q & 1 == 1]
        g[has] += g[has ^ (1 << q)]
    return g


def ledger(S, n):
    """The order shares c_2..c_n of the total correlation, and C itself."""
    I_T = coinformation(S, n)
    pc = popcounts(n)
    shares = np.array([(-1) ** k * I_T[pc == k].sum() for k in range(2, n + 1)])
    C = sum(S[1 << q] for q in range(n)) - S[-1]
    if abs(shares.sum() - C) > CHECK_TOL:
        raise RuntimeError("ledger: the order shares sum to %r, not C = %r" % (shares.sum(), C))
    return shares, C


# ============================================================ block MI and monogamy

def _labellings(n, parts):
    """Every assignment of the qubits to parts 0..parts-1, as a (parts^n, n) array."""
    codes = np.arange(parts ** n)
    return (codes[:, None] // parts ** np.arange(n)) % parts


def disjoint_pairs(n):
    """(A, B) masks for every unordered pair of disjoint non-empty subsets."""
    lab = _labellings(n, 3)
    bits = 1 << np.arange(n)
    A, B = (lab == 1) @ bits, (lab == 2) @ bits
    keep = (A > 0) & (B > 0) & (lowest_bit(A) < lowest_bit(B))
    return A[keep], B[keep]


def block_mi(S, A, B):
    return S[A] + S[B] - S[A | B]


def mmi_report(S, n):
    """Violations of I3(A:B:C) <= 0 over unordered disjoint non-empty triples."""
    lab = _labellings(n, 4)
    bits = 1 << np.arange(n)
    A, B, C = ((lab == k) @ bits for k in (1, 2, 3))
    keep = ((A > 0) & (B > 0) & (C > 0)
            & (lowest_bit(A) < lowest_bit(B)) & (lowest_bit(B) < lowest_bit(C)))
    A, B, C = A[keep], B[keep], C[keep]
    I3 = S[A] + S[B] + S[C] - S[A | B] - S[A | C] - S[B | C] + S[A | B | C]
    k = int(np.argmax(I3)) if len(I3) else 0
    return {"violations": int((I3 > MMI_TOL).sum()), "triples": len(I3),
            "worst": float(I3[k]) if len(I3) else float("-inf"),
            "worst_triple": (int(A[k]), int(B[k]), int(C[k])) if len(I3) else (0, 0, 0),
            "I3": I3}


def slice_summary(net, S_global):
    """The ledger and monogamy count of one time slice."""
    S = region_entropies(net, S_global)
    shares, C = ledger(S, net.n)
    mmi = mmi_report(S, net.n)
    return {"shares": shares, "C": C, "mmi_violations": mmi["violations"],
            "mmi_triples": mmi["triples"], "mmi_worst": mmi["worst"]}


# ============================================================ report

def analyse(net, S_global):
    n = net.n
    if n > MAX_QUBITS:
        raise ValueError("region analysis supports at most %d qubits" % MAX_QUBITS)
    S = region_entropies(net, S_global)
    shares, C = ledger(S, n)
    A, _ = disjoint_pairs(n)
    return {"n": n, "S": S, "shares": shares, "C": C, "mmi": mmi_report(S, n),
            "n_pairs": len(A), "S_global": float(S_global)}


def print_report(rep, names, log=print):
    n, mmi = rep["n"], rep["mmi"]
    log("REGIONS: entropies of every subset, the co-information ledger, monogamy")
    log("  %d region entropies -> %d block mutual informations (nats)"
        % (2 ** n - 1, rep["n_pairs"]))
    log("  total correlation by order: C = sum_k c_k, c_k = (-1)^k sum_{|T|=k} I_T")
    for k, c in enumerate(rep["shares"], 2):
        share = c / rep["C"] if rep["C"] else float("nan")
        log("    k = %2d   c_k = %+.6f   (%+7.1f%% of C)" % (k, c, 100 * share))
    log("    C = sum S_i - S_global = %.6f   (at most n ln 2 - S_global = %.6f)"
        % (rep["C"], n * math.log(2.0) - rep["S_global"]))
    a, b, c = mmi["worst_triple"]
    log("  monogamy of mutual information, I3(A:B:C) <= 0, over %d disjoint triples: "
        "%d violated" % (mmi["triples"], mmi["violations"]))
    log("    largest I3 = %+.3e at (%s : %s : %s)"
        % (mmi["worst"], label(a, names), label(b, names), label(c, names)))


# ============================================================ figure

#: The ordinal blue ramp (light, few-body, to dark, many-body) for the order k.
RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")


def order_colours(n):
    """A colour per order k = 2..n, light to dark."""
    idx = np.linspace(0, len(RAMP) - 1, max(n - 1, 1)).round().astype(int)
    return {k: RAMP[i] for k, i in zip(range(2, n + 1), idx)}


def draw(rep, slices, names, save=None):
    """The ledger: the order shares c_k over the time slices (or as bars for
    the final state), with the total C and its cap n ln 2 - S_global."""
    import matplotlib.pyplot as plt
    from tessera.drivers import baryon_poles as bp

    n, mmi = rep["n"], rep["mmi"]
    col = order_colours(n)
    fig = plt.figure(figsize=(12, 7.5), facecolor=bp.SURFACE)
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.2], hspace=0.3,
                            left=0.08, right=0.98, top=0.88, bottom=0.04)
    ax = fig.add_subplot(grid[0])
    bp.style_axis(ax)
    cap = n * math.log(2.0) - rep["S_global"]
    if slices and all("regions" in s for s in slices):
        t = [s["t"] for s in slices]
        shares = np.array([s["regions"]["shares"] for s in slices])
        for i, k in enumerate(range(2, n + 1)):
            ax.plot(t, shares[:, i], color=col[k], linewidth=2, label="k = %d" % k)
        ax.plot(t, [s["regions"]["C"] for s in slices], color=bp.INK, linewidth=2,
                label="C (sum)")
        ax.set_xlabel("slice t (pair interactions)", color=bp.INK_MUTED)
        ax.legend(frameon=False, labelcolor=bp.INK_MUTED, ncol=4, loc="center right",
                  bbox_to_anchor=(1.0, 0.62))
    else:
        ks = np.arange(2, n + 1)
        ax.bar(ks, rep["shares"], width=0.6, color=[col[k] for k in ks],
               edgecolor=bp.SURFACE, linewidth=2)
        for k, c in zip(ks, rep["shares"]):
            if rep["C"] and abs(c) > 0.02 * abs(rep["C"]):
                ax.annotate("%.0f%%" % (100 * c / rep["C"]), (k, c),
                            xytext=(0, 3 if c >= 0 else -10), textcoords="offset points",
                            ha="center", fontsize=8, color=bp.INK_MUTED)
        ax.set_xticks(ks)
        ax.set_xlabel("k (k-body co-information)", color=bp.INK_MUTED)
    ax.axhline(cap, color=bp.BASELINE, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate("cap n ln 2 - S_global = %.3f" % cap, (0.0, cap),
                xycoords=("axes fraction", "data"), xytext=(4, -10),
                textcoords="offset points", ha="left", fontsize=8, color=bp.INK_MUTED)
    ax.axhline(0, color=bp.BASELINE, linewidth=0.8)
    ax.set_ylabel("nats", color=bp.INK_MUTED)
    ax.set_title("total correlation by order: C = sum_k c_k", loc="left", color=bp.INK)

    text = fig.add_subplot(grid[1])
    text.axis("off")
    a, b, c = mmi["worst_triple"]
    lines = ["%d region entropies -> %d block mutual informations (nats)"
             % (2 ** n - 1, rep["n_pairs"]),
             "monogamy I3(A:B:C) <= 0: %d/%d triples violate; largest I3 = %+.3e at (%s:%s:%s)"
             % (mmi["violations"], mmi["triples"], mmi["worst"],
                label(a, names), label(b, names), label(c, names))]
    text.text(0.0, 1.0, "\n".join(lines), transform=text.transAxes, va="top",
              family="monospace", fontsize=9, color=bp.INK)
    fig.suptitle("%d qubits: where the correlation goes" % n, x=0.02, ha="left",
                 color=bp.INK)
    if save:
        fig.savefig(save, dpi=150, facecolor=bp.SURFACE)
    return fig
