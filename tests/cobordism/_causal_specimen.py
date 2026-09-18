# Copyright (c) 2026 Twin Vector Labs LLC.
# All rights reserved.

"""Rebuild a causal-specimen geometry dump as a live joint-node continuation.

The #562 campaign worker dumps every attempt's final complex
(``worker.dump_geometry``, schema 1): top cells in intrinsic vertex order,
edges as ``[src, tgt, Re l^2, Im l^2]`` (Im is 0 in every specimen), and
per-vertex times — enough for ``Spacetime.fromVertexTuples`` to rebuild the state
without re-running anything. ``tests/fixtures/causal_specimens/`` carries the
campaign's first causal specimens (seeds 14001000, 11001000, 13001000 reached
``re_min < 0``) plus two all-spacelike ones, copied out of the campaign run.

The dump does NOT record the input seed vertices. The joint node seeds its
three neutral-pair inputs at the first three vertices of the freshly built
Δ⁴ seed (``ProtonIngredients::jointNode`` — ``seedVertexIds[0..2]``, ids
0, 1, 2), and those ids persist through the emergent build (every specimen
carries them). The reconstruction therefore seeds the SAME vertex ids on the
rebuilt final complex; each input block's REGION is re-derived as its seed
vertex's current cell neighbourhood (``seedInputs``) — the grown regions are
not recorded, so this is the closest faithful reconstruction the dump
supports, and the one knob on which a reconstructed ``F`` may differ from the
recorded one.
"""

import json
import os

import tessera
import cmath

cob = tessera.cobordism

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "fixtures", "causal_specimens")

# The joint node's fixed prepared content (ProtonIngredients::jointNode): the
# three Z3-symmetric neutral q-qbar pairs as inputs, nothing pinned downstream.
JOINT_PAIRS = [[complex(1), complex(-1), complex(0)],
               [complex(0), complex(1), complex(-1)],
               [complex(-1), complex(0), complex(1)]]
REGISTER_DEGREE = 3     # worker.REGISTER_DEGREE
GAMMA = 50.0            # ProtonIngredients default
INPUT_WEIGHT = 20.0     # ProtonIngredients default
SEED_VERTEX_IDS = [0, 1, 2]


def load_dump(base_seed):
    path = os.path.join(FIXTURE_DIR, f"seed_{base_seed}_geometry.json")
    with open(path) as fh:
        return json.load(fh)


def rebuild_spacetime(dump):
    """A Spacetime carrying the dumped final state: fromVertexTuples on the top
    cells, then the recorded per-vertex times and per-edge complex squared
    lengths (the analyze_attempt.py rebuild path, verbatim)."""
    st = tessera.spacetime.Spacetime.fromVertexTuples(dump["dimensions"],
                                               dump["cells"])
    vertices = st.getVertexList()
    for vid, t in dump["vertex_times"]:
        vertices.get(int(vid)).setTime(float(t))
    by_pair = {}
    for e in st.getEdgeList().toVector():
        a, b = e.getSource().getId(), e.getTarget().getId()
        by_pair[(min(a, b), max(a, b))] = e
    for u, v, re_l2, im_l2 in dump["edges"]:
        key = (min(int(u), int(v)), max(int(u), int(v)))
        by_pair[key].setLength(cmath.sqrt(complex(complex(re_l2, im_l2))))
    st.materializeFacets()
    return st


def rebuild_joint_node(dump, seed):
    """The jointNode-equivalent MultiCobordism over the REBUILT host: same
    pair targets, degrees, gamma, and input weight as
    ``ProtonIngredients::jointNode``, inputs re-seeded at the recorded seed
    vertex ids (see the module docstring for the region caveat)."""
    st = rebuild_spacetime(dump)
    node = cob.MultiCobordism(st, JOINT_PAIRS, [], degrees=[REGISTER_DEGREE],
                              gamma=GAMMA, seed=seed, precone=0)
    node.set_input_residual_weight(INPUT_WEIGHT)
    node.seed_inputs(SEED_VERTEX_IDS)
    return node
