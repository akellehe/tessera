# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.
"""Inputs recorded by the recursion run of 2026-09-23.

The run was ``python -m tessera.drivers.recursion run --ticks 3 --live`` at
commit c86c915b, with its outputs in ``~/recursion-runs/2026-09-23/``
(``recursion.json``). The values below are copied from ``recursion.json`` with
their full double-precision representation, so that a test reads exactly the
input the run read. They are the tick-0 level (the declared two-tetrahedron fan
around the edge (0, 1)) and the two host cells whose per-cell reads failed
quark condition 2 in 12 of 16 contents.
"""

#: The tick-0 base cells.
LEVEL_ZERO_CELLS = [[0, 1, 2, 3], [0, 1, 3, 4]]

#: The tick-0 level fields, per ascending base edge, as recorded after the
#: level relaxation (which took zero Newton iterations: the declared host is
#: stationary in its held sector to a residual of 1.18e-14).
LEVEL_ZERO_SQUARED_LENGTHS = {
    (0, 1): 8.000000000000002, (0, 2): 8.000000000000002,
    (0, 3): 8.000000000000002, (1, 2): 8.000000000000002,
    (1, 3): 8.000000000000002, (2, 3): 8.000000000000002,
    (0, 4): 8.000000000000002, (1, 4): 8.000000000000002,
    (3, 4): 8.000000000000002,
}
LEVEL_ZERO_LINKS = {
    (0, 1): complex(1.0, 8.881784197001252e-16),
    (0, 2): complex(-0.4999999999999994, -0.8660254037844389),
    (0, 3): complex(1.0, 3.134022356484828e-16),
    (1, 2): complex(-0.5000000000000002, 0.8660254037844385),
    (1, 3): complex(1.0, -3.1340223564848283e-16),
    (2, 3): complex(1.0, -4.930380657631324e-32),
    (0, 4): complex(-0.4999999999999994, 0.8660254037844389),
    (1, 4): complex(-0.4999999999999994, -0.8660254037844389),
    (3, 4): complex(1.0, 2.465190328815662e-32),
}

#: The bounding cut of the tick-0 host as the run held it (run.log), and the
#: monopole number through it before and after the relaxation.
LEVEL_ZERO_BOUNDING_CUT = [[1, 2, 3], [0, 3, 2], [0, 2, 1], [1, 3, 4],
                           [0, 4, 3], [0, 1, 4]]
LEVEL_ZERO_CUT_MONOPOLE_NUMBER = 2

#: The two host cells of the per-cell reads: the six squared lengths and the
#: six links U_e on the ascending orientation of each edge, in the order of
#: `MonopoleSupport.tetrahedron(1).edges`, (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
#: of the cell's sorted vertices.
HOST_CELLS = {
    (0, 1, 2, 3): {
        "squared_lengths": [complex(8.000000000000002, 0.0)] * 6,
        "links": [complex(1.0, 8.881784197001252e-16),
                  complex(-0.4999999999999994, -0.8660254037844389),
                  complex(1.0, 3.134022356484828e-16),
                  complex(-0.5000000000000002, 0.8660254037844385),
                  complex(1.0, -3.1340223564848283e-16),
                  complex(1.0, -4.930380657631324e-32)],
    },
    (0, 1, 3, 4): {
        "squared_lengths": [complex(8.000000000000002, 0.0)] * 6,
        "links": [complex(1.0, 8.881784197001252e-16),
                  complex(1.0, 3.134022356484828e-16),
                  complex(-0.4999999999999994, 0.8660254037844389),
                  complex(1.0, -3.1340223564848283e-16),
                  complex(-0.4999999999999994, -0.8660254037844389),
                  complex(1.0, 2.465190328815662e-32)],
    },
}

#: The condition-2 evidence the run recorded for every read content, per host
#: cell: (content, sheet isomorphism held, squared-length residual, connection
#: residual, fibre lift held, largest face-holonomy disagreement between
#: sheets). The last column is computed from the recorded
#: ``relaxation.face_holonomies``; it separates the four reads whose sheets
#: differ only by a gauge transformation from the eight whose geometry
#: separated.
CONDITION_TWO_RECORD = {
    (0, 1, 2, 3): [
        ((0, 1, 2), True, 9.98e-10, 1.44e-09, True, 9.68e-11),
        ((0, 2, 1), False, 22.3, 5.6, False, 1.71),
        ((1, 0, 2), False, 19.5, 1.52e5, False, 2.00),
        ((1, 1, 1), False, 5.43e-4, 1.05e3, False, 1.58e-6),
        ((1, 2, 0), False, 66.8, 5.57, False, 1.95),
        ((2, 0, 1), True, 6.17e-10, 1.72e-09, True, 1.07e-10),
        ((2, 1, 0), False, 2.33e-08, 1.49, True, 4.83e-10),
    ],
    (0, 1, 3, 4): [
        ((0, 0, 3), False, 1.69e-08, 1.61, True, 2.85e-09),
        ((0, 1, 2), False, 3.82e-09, 0.868, True, 9.22e-10),
        ((0, 2, 1), False, 11.0, 2.12, False, 1.96),
        ((1, 0, 2), False, 7.3, 14.2, False, 1.82),
        ((1, 1, 1), False, 5.22, 3.06e4, False, 3.47e-3),
        ((1, 2, 0), False, 4.42e-09, 0.852, True, 1.71e-09),
        ((2, 0, 1), False, 1.67e4, 6.79e4, False, 1.44),
        ((2, 1, 0), True, 6.29e-09, 3.56e-09, True, 2.89e-10),
        ((3, 0, 0), True, 3.48e-10, 1.25e-09, True, 1.50e-11),
    ],
}

#: The contents the run refused on host cell (0, 1, 2, 3), each with
#: "SelfConsistentMeanField: band 2 has rank 2 and cannot hold the declared
#: occupation 3.000000".
REFUSED_ON_FIRST_CELL = [(0, 0, 3), (0, 3, 0), (3, 0, 0)]
