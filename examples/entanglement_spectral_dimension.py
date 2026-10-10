#!/usr/bin/env python3
# Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
"""
Spectral dimension of the qubit information network: the dimension a walker
diffusing on the mutual-information graph sees, slice by slice.

n qubits evolve under one SWAP^alpha interaction per time slice. In every
slice the qubits are the vertices of a graph whose edge (X, Y) carries the
mutual information I(X:Y) as its weight (or, with --walk skeleton, an edge of
weight 1 for every pair within --scale of the entanglement complex's length).
The return probability P(sigma) = (1/n) Tr exp(-sigma L) of the heat kernel
of the weighted Laplacian L gives the spectral dimension
D_S(sigma) = -2 d ln P / d ln sigma, read from the maximum of the curve and
from the Ambjorn-Loll fit D_S = D_inf - C / (B + sigma). By default the
all-pairs and the chain schedules are run on the same seeds and overlaid.

This script holds no logic of its own. It runs
``tessera.drivers.entanglement_spectral``, which builds the network with
``tessera.drivers.entanglement_complex`` and the graphs, return probabilities,
curves and fits with ``tessera.quantum.holography``; every option passes
through to the driver, see --help.

Usage:
  python examples/entanglement_spectral_dimension.py --save spectral.png
  python examples/entanglement_spectral_dimension.py --state pure --qubits 20 \\
      --timesteps 40 --save spectral20.png
  python examples/entanglement_spectral_dimension.py --walk skeleton --scale 1.2 \\
      --save skeleton.png
"""

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from tessera.drivers import entanglement_spectral as driver
    # As in every example, --save writes the figure and opens no window.
    if "--save" in argv and "--no-show" not in argv:
        argv.append("--no-show")
    return driver.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
