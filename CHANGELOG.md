# Changelog

What changed between states of this repository. Code and documentation describe
the present state; the history of how it got there lives here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Fixed

- **`CDTSimulation::tune` locates the pseudo-critical coupling by measurement.**
  It previously took \f$ k_4 \f$ from a closed form that zeroes the Regge action
  change of one add move, which prices that move's energy and ignores the entropy
  of the triangulations available at a volume. For \f$ k_0 = 2.2 \f$,
  \f$ \Delta = 0.6 \f$ that returned -0.233 where long chains put the coupling
  between 1.0 and 1.2, so every simulation ran below criticality: the
  volume-fixing term could not hold \f$ N_4^{(4,1)} \f$ at its target, which
  settled instead at a fixed 1.5 times it, and \f$ N_4^{(3,2)} \f$, which carries
  no fixing term, was left to grow. `tune` now brackets the sign change of the
  four-volume drift and bisects. (#965, PR #966)

- **Example figures written with `--save` no longer open a window.**
  `spectral_dimension.py` and `volume_profile_phases.py` called `plt.show()`
  after `savefig` unconditionally, so an unattended run blocked at the end
  holding its configurations in memory. (PR #958)

### Performance

- **The shift and inverse-flip moves propose off the edge's simplex index.**
  They walked a vertex's incidence list and filtered by `hasVertex`. That list
  grows with the four-volume -- 80 simplices per vertex at
  \f$ N_4 = 24.5\mathrm{k} \f$, 148 at \f$ 50.4\mathrm{k} \f$ -- so each proposal
  was linear in the volume and a sweep quadratic. (#970, PR #972)

- **The remove move stops counting a vertex's star once it cannot host the
  move.** It fires only on a vertex of order \f$ 2d \f$ and succeeded 1.2% of the
  time, but walked the whole star to find out. (#970, PR #972)

  Together these took a sweep at \f$ N_4 = 50\mathrm{k} \f$ from 1.130 s to
  0.544 s and its scaling exponent from 1.42 to 0.64.

- **Vertex relabeling is off by default in `CDTSimulation`.** The uniform
  labeling [BGL] Sec. 2.2.1 requires is carried by the Metropolis prefactors,
  and no part of the sweep reads a label, so the swap changed nothing the chain
  samples while costing a walk over a uniformly drawn vertex's incidence list.
  A sweep at \f$ N_4 = 50\mathrm{k} \f$ went from 0.544 s to 0.290 s and its
  exponent from 0.64 to flat. Set `setRelabelVertices(true)` to restore the
  swap. (#978, PR #1023)

### Added

- `Edge::simplices` is exposed to Python.
- `examples/benchmarks/sweep_scaling.py` times a sweep against the four-volume
  and fits the scaling exponent.
