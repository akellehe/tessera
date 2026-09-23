# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""tessera.drivers -- research drivers for the cobordism relaxation loops.

These modules drive `MultiCobordism` and read certificates off the accepted
geometry. They live in the package, not in an example script, because the C++
API test suite imports them as a library: `emergence.build_config` and
`qubit.build_config` construct the spacetimes most cobordism tests run against.

* `emergence` -- unforced emergence on a neutral complex.
* `qubit`     -- the two-torus qubit cobordism experiment.
* `harmonic`  -- harmonic state/operator correspondence measurement.
* `fock`      -- the inductive limit of the Fock stages over a refinement
                 sequence: the compatibility defect of the vacuum embedding,
                 measured at every adjacent pair of a stated sequence.

Each module keeps a `main()` entry point, so the thin command line wrappers
kept outside this repository can call straight into it.
"""
