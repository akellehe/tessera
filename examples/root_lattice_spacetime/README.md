# Root lattice spacetime

`root_lattice_spacetime.py` draws, for a network of qubits evolved by pairwise
SWAP^α interactions, three views of the same data side by side and one path
traversed on all three. It is a thin script over
`tessera.drivers.root_lattice_spacetime`, which builds the network with
`tessera.drivers.entanglement_complex` and stores the order of the
interactions in a `tessera.quantum.Poset`.

```
python examples/root_lattice_spacetime/root_lattice_spacetime.py --save out.png
python examples/root_lattice_spacetime/root_lattice_spacetime.py --qubits 4 --timesteps 12 --start B --save walk.png
python examples/root_lattice_spacetime/root_lattice_spacetime.py --path geodesic --start A --end D --save geodesic.png
```

## Definitions

- The network has n qubits named A, B, C, ... in a global state. One
  SWAP^α interaction is applied per time slice t = 1..T to a pair drawn by the
  schedule (`--pairs all` draws from every pair, `--pairs chain` from the
  nearest neighbours of an open chain). S(X) is the von Neumann entropy in
  nats of the one-qubit reduced state of X, and I(X:Y) = S(X) + S(Y) − S(XY)
  is the mutual information of the pair.
- The length of a pair is l(X,Y) = a ln(1 + I_0 / I(X:Y)) for I(X:Y) above the
  floor, with the scale a (`--length-scale`) and the reference mutual
  information I_0 (`--length-reference`, default I_max = 2 ln 2). It falls as
  the mutual information grows and diverges as it vanishes; a pair at or below
  the floor has no edge.
- The local Cartan subalgebra of the network is spanned by the Pauli Z of
  every qubit. In its dual, e_X is the weight of one unit of Z-charge on X.
  The off-diagonal part of the generator of SWAP^α on (X, Y) is the pair of
  ladder operators of the root e_Y − e_X: the interaction translates one unit
  of charge between the two qubits. The n(n − 1) roots e_Y − e_X form the root
  system A_{n−1} on the qubit labels, and the lattice they generate, the
  integer vectors with zero sum, is the root lattice.
- The lattice is drawn in the Coxeter plane, where the cyclic permutation of
  the qubits in alphabetical order acts as the rotation by 2π/n: the weight
  e_X is the point at angle 2πX/n on the unit circle, the root e_Y − e_X the
  chord from X to Y. This is a projection fixed by the order of the names, not
  a fit to the lengths. For n ≥ 5 the projected lattice is dense in the plane,
  so only the points within two root steps of the origin are drawn.
- Time is the order of the interactions in the sense of a causal set. Every
  interaction is an event; an event precedes a later one when their pairs
  share a qubit, and the order is the transitive closure of that relation.
  Events on disjoint pairs are unrelated, as their generators commute. No edge
  of the lattice carries a duration.
- The path (`--path walk`, the default) is the worldline of one unit of charge
  that starts on the qubit `--start` and is carried across every event whose
  pair contains its current qubit; each step is the root of its event and has
  the length of the pair in the state after the event. With `--path geodesic`
  the path is the shortest path from `--start` to `--end` through the edges of
  the final slice, with that slice's lengths.

![The three panels for six qubits and 24 sqrt(SWAP) interactions](root_lattice_spacetime.png)

The rendering above is the default run: six qubits, 24 interactions of
sqrt(SWAP) drawn from all pairs, the walk from A.

## The three panels

1. **Information network.** The final slice. Each qubit is a disc of radius
   S(X) at its Coxeter-plane weight, on a circle large enough that
   neighbouring discs do not overlap. Each pair with I(X:Y) above the floor is
   an edge coloured by l(X,Y), solid if the pair interacted at least once and
   dashed if its correlation arose only through other qubits. The path's edges
   are bold and labelled with their lengths. The drawn chord lengths are the
   Coxeter-plane geometry, not l(X,Y).
2. **Root lattice.** The roots as arrows from the origin, the lattice points
   within two root steps, the dashed n-gon of positions the charge can occupy,
   and the path as root translations composed tip to tail from the origin,
   each labelled with its length. The path ends at e_end − e_start whatever
   route it took.
3. **Spacetime.** The Coxeter plane against the slice index. Each qubit is a
   vertical worldline, each event the chord of its pair at its slice coloured
   by l(X,Y) after the event, and the path's worldline is bold: it rises along
   its qubit's line and crosses a chord at every event it rides.

The printed report lists the entropies and lengths of the final slice, every
event with its length and the number of events in its past, the path's steps,
its total length and its displacement in the lattice, and the shortest path
between the same endpoints in the final slice for comparison.
