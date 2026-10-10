#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Root lattice spacetime: the information network of a qubit network, the root
lattice its interactions translate, and the causal set the order of the
interactions makes of it, with one path on all three.

n qubits evolve under one SWAP^alpha interaction per time slice. Every qubit X
is drawn as a disc of radius S(X), its von Neumann entropy in nats, and every
pair (X, Y) with mutual information I(X:Y) above the floor as an edge of
length l(X,Y) = a ln(1 + I_0 / I(X:Y)). The interaction on (X, Y) is the pair
of ladder operators of the root e_Y - e_X of the local Cartan subalgebra's
dual: it translates one unit of Z-charge between the two qubits, so the
n(n - 1) roots e_Y - e_X form the root system A_{n-1} on the qubit labels and
generate the root lattice, drawn in the first three Fourier coordinates of
the cyclic order of the qubits (exact for four qubits, where the lattice is
the face-centred cubic lattice of su(4) and the roots a cuboctahedron; a
projection for more). Time is the order of the interactions in the sense of a
causal set: an event precedes every later event that shares a qubit with it,
and the third panel is the Hasse diagram of that order with each event at the
height of its depth, the longest chain below it. The path is the worldline of
a unit of charge that starts on one qubit and rides every interaction of its
current qubit, or, with --path geodesic, the shortest path between two qubits
in the final slice; its steps carry the same numbers and colours on all three
panels, and --animate writes one frame per slice.

This script holds no logic of its own. It runs
``tessera.drivers.root_lattice_spacetime``, which builds the network with
``tessera.drivers.entanglement_complex`` and stores the causal order in a
``tessera.quantum.Poset``; every option passes through to the driver, see --help.

Usage:
  python examples/root_lattice_spacetime/root_lattice_spacetime.py --save out.png
  python examples/root_lattice_spacetime/root_lattice_spacetime.py --qubits 6 \\
      --timesteps 24 --start B --lattice-dims 2 --save plane.png
  python examples/root_lattice_spacetime/root_lattice_spacetime.py --path geodesic \\
      --start A --end D --save geodesic.png
  python examples/root_lattice_spacetime/root_lattice_spacetime.py --animate walk.gif
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from tessera.drivers import root_lattice_spacetime as driver
    # As in every example, --save writes the figure and opens no window.
    if "--save" in argv and "--no-show" not in argv:
        argv.append("--no-show")
    return driver.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
