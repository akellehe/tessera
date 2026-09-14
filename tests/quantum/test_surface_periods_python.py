"""`SurfacePeriods` (issue #1115): the period matrix of a closed triangulated
surface from its own lengths -- a flat torus reproduces `SimplicialQubit`'s
modulus, and a genus-2 surface built from two tori through a tube gives a
symmetric period matrix with positive definite imaginary part."""
import math

import numpy as np
import pytest

import tessera as T  # noqa: F401  (the driver imports the package)
from tessera import observables as obs
from tessera.quantum import SurfacePeriods


def _flat_torus_data(tau, n):
    """A flat torus as `SurfacePeriods` wants it: faces, lengths, marking."""
    torus = obs.SimplicialQubit.flat_torus(tau, n, n)
    edges = torus.edges()
    lengths = {(min(int(i), int(j)), max(int(i), int(j))): float(np.real(l))
               for (i, j), l in zip(edges, torus.lengths())}
    faces = [tuple(int(v) for v in face) for face in torus.faces()]

    def cycle(steps):
        out = []
        for e, sign in steps:
            i, j = (int(v) for v in edges[int(e)])
            out.append((i, j) if sign > 0 else (j, i))
        return out
    return torus, faces, lengths, cycle(torus.cycle_A()), cycle(torus.cycle_B())


@pytest.mark.parametrize("tau", [0.3 + 1.1j, -0.2 + 0.8j, 1j])
def test_flat_torus_reproduces_the_modulus(tau):
    torus, faces, lengths, a, b = _flat_torus_data(tau, 3)
    read = SurfacePeriods(faces, lengths, [a], [b], root_face=faces[0])
    assert read.harmonic_rank == 2
    assert read.positive and not read.orientation_flipped
    assert abs(complex(read.omega[0, 0]) - complex(torus.tau())) <= 1e-9
    assert read.j_residual <= 1e-9


def test_reversed_root_face_flips_the_orientation_and_is_repaired():
    torus, faces, lengths, a, b = _flat_torus_data(0.3 + 1.1j, 3)
    root = (faces[0][0], faces[0][2], faces[0][1])
    read = SurfacePeriods(faces, lengths, [a], [b], root_face=root)
    assert read.orientation_flipped and read.positive
    assert abs(complex(read.omega[0, 0]) - complex(torus.tau())) <= 1e-9


def _tubed_surface(tau_1, tau_2, n, waist, length, layers=2):
    """Two flat tori, one triangle removed from each, joined by the walls of a
    prism between the removed triangles: a genus-2 surface, with the two
    tori's markings. The tube's rings are the affine interpolation of the
    two triangles' layouts scaled by `waist`, `length` apart."""
    tori = [_flat_torus_data(tau_1, n), _flat_torus_data(tau_2, n)]
    offset = n * n
    faces, lengths = [], {}
    markings = []
    for index, (torus, tf, tl, a, b) in enumerate(tori):
        shift = index * offset
        faces.extend(tuple(v + shift for v in face) for face in tf)
        lengths.update({(u + shift, v + shift): l for (u, v), l in tl.items()})
        markings.append(([(u + shift, v + shift) for u, v in a], [(u + shift, v + shift) for u, v in b]))
    vid = lambda i, j: i * n + j  # noqa: E731
    hole = (vid(1, 1), vid(2, 1), vid(2, 2))
    end_a = [v for v in hole]
    end_b = [hole[0] + offset, hole[2] + offset, hole[1] + offset]   # reflected: orientation-consistent
    faces = [f for f in faces if set(f) != set(end_a) and set(f) != set(end_b)]
    # rings
    rings = [end_a] + [[2 * offset + 3 * (l - 1) + i for i in range(3)] for l in range(1, layers)] + [end_b]

    def layout(end, lens):
        c = lens[(min(end[0], end[1]), max(end[0], end[1]))]
        a = lens[(min(end[1], end[2]), max(end[1], end[2]))]
        b = lens[(min(end[2], end[0]), max(end[2], end[0]))]
        alpha = math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c))))
        return [np.array([0.0, 0.0]), np.array([c, 0.0]), np.array([b * math.cos(alpha), b * math.sin(alpha)])]
    la, lb = layout(end_a, lengths), layout(end_b, lengths)
    positions = []
    for l in range(layers + 1):
        s = l / layers
        scale = 1.0 if l in (0, layers) else waist
        ring = [(1 - s) * la[i] + s * lb[i] for i in range(3)]
        centroid = sum(ring) / 3
        positions.append([np.array([*(centroid + scale * (p - centroid)), l * length]) for p in ring])
    # prism cells over {0,1,2} as Spacetime.prismCells would produce them, as their boundary triangles
    def prism_faces():
        out = set()
        for l in range(layers):
            lo = rings[l]
            hi = rings[l + 1]
            for j in range(3):
                cell = [lo[i] for i in range(j + 1)] + [hi[i] for i in range(j, 3)]
                for skip in range(4):
                    out.add(tuple(sorted(v for m, v in enumerate(cell) if m != skip)))
        counts = {}
        for l in range(layers):
            lo, hi = rings[l], rings[l + 1]
            for j in range(3):
                cell = [lo[i] for i in range(j + 1)] + [hi[i] for i in range(j, 3)]
                for skip in range(4):
                    face = tuple(sorted(v for m, v in enumerate(cell) if m != skip))
                    counts[face] = counts.get(face, 0) + 1
        return [face for face, c in counts.items() if c == 1
                and set(face) != set(end_a) and set(face) != set(end_b)]
    slot = {v: (l, i) for l, ring in enumerate(rings) for i, v in enumerate(ring)}
    walls = prism_faces()
    for face in walls:
        for a_ in range(3):
            u, v = face[a_], face[(a_ + 1) % 3]
            key = (min(u, v), max(u, v))
            if key not in lengths:
                (lu, iu), (lv, iv) = slot[u], slot[v]
                lengths[key] = float(np.linalg.norm(positions[lu][iu] - positions[lv][iv]))
    faces.extend(walls)
    root = (vid(0, 0), vid(1, 0), vid(1, 1))
    return faces, lengths, [markings[0][0], markings[1][0]], [markings[0][1], markings[1][1]], root


@pytest.mark.parametrize("waist,length", [(1.0, 1.0), (0.5, 1.0), (1.0, 3.0)])
def test_genus_two_period_matrix(waist, length):
    faces, lengths, a, b, root = _tubed_surface(0.3 + 1.1j, -0.2 + 0.8j, 3, waist, length)
    read = SurfacePeriods(faces, lengths, a, b, root_face=root)
    assert read.harmonic_rank == 4
    assert read.positive and not read.orientation_flipped
    assert read.symmetry_residual < 0.2          # discrete: reported, not exact
    assert abs(read.omega[0, 1]) > 1e-6          # the neck couples the tori
    assert read.j_residual < 1.0


def test_genus_two_neck_pinches():
    """A thinner, longer neck couples less: |Omega_12| decreases and the
    diagonal approaches the tori's own moduli."""
    reads = []
    for waist, length in [(1.0, 0.5), (0.5, 1.0), (0.25, 2.0)]:
        faces, lengths, a, b, root = _tubed_surface(0.3 + 1.1j, -0.2 + 0.8j, 3, waist, length)
        reads.append(SurfacePeriods(faces, lengths, a, b, root_face=root))
    coupling = [abs(read.omega[0, 1]) for read in reads]
    assert coupling[0] > coupling[1] > coupling[2]


def test_refusals():
    faces, lengths, a, b, root = _tubed_surface(0.3 + 1.1j, -0.2 + 0.8j, 3, 1.0, 1.0)
    with pytest.raises(ValueError, match="not closed"):
        SurfacePeriods(faces[:-1], lengths, a, b, root_face=root)
    bad = dict(lengths)
    bad[next(iter(bad))] = 1.0 + 0.5j
    with pytest.raises(ValueError, match="complex length"):
        SurfacePeriods(faces, bad, a, b, root_face=root)
    with pytest.raises(ValueError, match="A cycles against"):
        SurfacePeriods(faces, lengths, a, b[:1], root_face=root)
