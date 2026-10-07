#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Entanglement complex: a simplicial complex over qubits built from their
mutual information alone.

An n-qubit density matrix evolves under pairwise SWAP^alpha interactions,
either on a round-robin schedule or one random pair per time slice. The
mutual information I(X:Y) = S(X) + S(Y) - S(XY) between every pair of
qubits, with S the von Neumann entropy in nats, becomes an edge length, and
the complex is the Vietoris-Rips filtration of those lengths: at scale r, a
set of qubits is a simplex when every pairwise length within it is at most
r. The qubits are given no coordinates. The script prints the simplex counts
and Betti numbers at every scale where the complex changes and draws them;
the circle in the figure only displays which simplices exist.

This script holds no logic of its own. It runs
``tessera.drivers.entanglement_complex``, whose ``--regions`` option adds the
co-information ledger and monogamy check of
``tessera.drivers.entanglement_regions``; with ``--compare-schedules`` it runs
``tessera.drivers.entanglement_schedules`` instead. Every other option passes
through to the driver; run with ``--help`` for them.

Usage:
  python examples/entanglement_complex.py --save complex.png
  python examples/entanglement_complex.py --length log --geodesic --save geodesic.png
  python examples/entanglement_complex.py --qubits 8 --timesteps 40 --regions --save t8.png
  python examples/entanglement_complex.py --compare-schedules --qubits 8 \\
      --timesteps 40 --seeds 0 1 --save schedules.png
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--compare-schedules" in argv:
        argv.remove("--compare-schedules")
        from tessera.drivers import entanglement_schedules as driver
    else:
        from tessera.drivers import entanglement_complex as driver
    # As in every example, --save writes the figures and opens no window.
    if "--save" in argv and "--no-show" not in argv:
        argv.append("--no-show")
    return driver.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
