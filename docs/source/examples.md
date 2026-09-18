# CDT examples

These examples reproduce experiments from the causal dynamical triangulations
(CDT) literature.  Each script runs a Metropolis Monte Carlo simulation using the
`tessera` C++ library and produces plots that can be compared side by side with
figures from the original papers.

All scripts accept `--save <path.png>` to write output to disk, or display
interactively by default.  Run any script with `--help` for the full set of
tunable parameters.

```{note}
The paper results were obtained on lattices of $N_4 = 10\,000$--$362\,000$
four-simplices with $T = 80$ time slices and $10^5$--$10^8$ Monte Carlo
sweeps.  The plots shown here use $N_4 \sim 800$--$1\,600$ to keep
runtimes under a few minutes.  Observables that depend on large-scale
geometry (phase boundaries, Hausdorff scaling, $\cos^3$ profile shape)
require at least $N_4 > 5\,000$ to become visible.
```

---

## Distribution of (3,2)-simplices at fixed (4,1)-volume

**Script:** `examples/n32_distribution.py` \
**Reproduces:** Figure 2, Ambjorn, Jurkiewicz & Loll,
*Reconstructing the Universe*, Phys. Rev. D **72** (2005)
\[[arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)\]

**Paper result (Fig. 2):** At fixed $\tilde{N}_4 = N_4^{(4,1)}$, the
distribution of $(3{,}2)$-simplices $P(N_4^{(3,2)})$ is sharply peaked,
with the peak narrowing and shifting to larger $N_4^{(3,2)}$ as the
target volume increases.  The volume-fixing term
$\delta S = \epsilon |N_4^{(4,1)} - \tilde{N}_4|$ therefore constrains
both simplex types simultaneously.

**Our result:** Confirmed.  The distributions are sharply peaked at each
target volume ($\tilde{N}_4 \approx 193$, $383$, $686$) and shift to
larger $N_4^{(3,2)}$ as in the paper.  The ratio
$N_4^{(3,2)} / N_4^{(4,1)} \approx 2.3$--$2.5$ is consistent with the
paper's Table 1.

```bash
python examples/n32_distribution.py \
    --target-volumes 40000 80000 160000 --n-therm 500 \
    --n-meas 500 --meas-interval 50
```

```{image} assets/cdt/n32_distribution.png
:alt: Distribution of N_4^(3,2) at three target volumes
:width: 70%
:align: center
```

---

## Spectral dimension

**Script:** `examples/spectral_dimension.py` \
**Reproduces:** Figures 9 and 10, Ambjorn, Jurkiewicz & Loll,
*Reconstructing the Universe*, Phys. Rev. D **72** (2005)
\[[arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)\]

**Paper result (Figs. 9--10):** The spectral dimension
$D_S(\sigma) = -2\, d\log P(\sigma) / d\log\sigma$ rises monotonically
from $D_S(0) = 1.80 \pm 0.25$ at short diffusion times to
$D_S(\infty) = 4.02 \pm 0.1$ at large scales.  The best three-parameter
fit is (Eq. 29):

$$
D_S(\sigma) = 4.02 - \frac{119}{54 + \sigma}
$$

**Our result:** Qualitatively confirmed.  The measured $D_S(\sigma)$
rises monotonically from near zero and approaches $\sim 3.5$ at
$\sigma = 200$, matching the shape of the paper's curve.  The absolute
values are lower than the paper's because our lattice ($N_4 \approx 1\,000$)
is much smaller than the paper's ($N_4 = 20$k--$80$k, $T = 80$); the
spectral dimension is finite-size dependent and converges upward with
increasing volume.  The paper's fit curve (red dashed) is overlaid for
reference.

```bash
python examples/spectral_dimension.py \
    --n-simplices 160000 --n-therm 500 --n-configs 50 \
    --n-walks 100 --max-sigma 500 --sweeps-between 50
```

```{image} assets/cdt/spectral_dimension.png
:alt: Spectral dimension D_S(sigma) with reference fit from the paper
:width: 100%
```

---

## Effective action and minisuperspace

**Script:** `examples/effective_action.py` \
**Reproduces:** Figures 11, 12, 13, Ambjorn, Jurkiewicz & Loll,
*Reconstructing the Universe*, Phys. Rev. D **72** (2005)
\[[arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)\]

**Paper results:**
- **Fig. 11:** The kinetic-term scaling dimension $D_2$ is estimated by
  minimising overlap error of rescaled volume-difference distributions.
  The minimum occurs at $D_2 \approx 2.12$, interpreted as $D_2 = 2$.
- **Fig. 12:** The distribution of rescaled volume differences
  $z = |\Delta N_3| / N_3^{1/D_2}$ at $D_2 = 2$ collapses onto a
  Gaussian $e^{-cz^2}$ independent of $N_3$.
- **Fig. 13:** The Monte Carlo volume-volume correlator matches the
  prediction from the minisuperspace effective action (Eq. 40).

**Our results:**
- **Scaling dimension (top left):** The overlap-error curve has its
  minimum near $D_2 \approx 2$, consistent with the paper.
- **Volume differences (top right):** The measured distribution decays.
  The Gaussian fit captures the trend but scatter is large at
  $N_4 \sim 800$.
- **Profile comparison (bottom left):** The Monte Carlo average profile
  is relatively flat compared to the $\cos^3(\pi\tau/T)$ prediction.  The
  $\cos^3$ shape emerges only at $N_4 \gg 10\,000$ with $T \geq 40$ time
  slices; our profile spans only $\sim 9$ slices.
- **Action evolution (bottom right):** The Regge action and four-volume
  track each other during equilibration, confirming the volume-fixing
  term drives $N_4$ toward the target.

```bash
python examples/effective_action.py \
    --n-simplices 160000 --n-therm 500 --n-meas 200 \
    --meas-interval 50
```

```{image} assets/cdt/effective_action.png
:alt: Effective action analysis: D_2 scaling, volume differences, minisuperspace comparison
:width: 100%
```

---

## Volume profiles in phases A, B, C

**Script:** `examples/volume_profile_phases.py` \
**Reproduces:** Figures 4, 5, 6, Ambjorn, Jurkiewicz & Loll,
*Reconstructing the Universe*, Phys. Rev. D **72** (2005)
\[[arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)\]

**Paper results:**
- **Fig. 4** (Phase A, $\kappa_0 = 5.0$, $\Delta = 0$): thin, elongated
  branched-polymer geometry with irregular maxima.
- **Fig. 5** (Phase B, $\kappa_0 = 1.6$, $\Delta = 0$): crumpled geometry
  collapsed to $\sim 2$ time slices.
- **Fig. 6** (Phase C, $\kappa_0 = 2.2$, $\Delta = 0.6$): extended
  de Sitter universe with $N_3(\tau) \propto \cos^3(\pi\tau/T)$.

**Our result:** At $N_4 \sim 800$ the three profiles are not
differentiated -- all show a roughly uniform distribution across
$\sim 9$ time slices.  The phase transitions in the paper occur at
$N_4 \sim 20$k--$45$k; at smaller volumes the system is too small to
develop the geometric structures that characterise each phase.
The dashed $\cos^3$ reference curve shows the expected shape for
phase $C_{dS}$ at large $N_4$.

To reproduce the paper's Fig. 6 shape, run with `--n-simplices 10000`
or larger (estimated runtime: several hours).

```bash
python examples/volume_profile_phases.py \
    --n-simplices 80000 --n-therm 500 --n-meas 100 \
    --meas-interval 50
```

```{image} assets/cdt/volume_profiles_surface.png
:alt: Surface-of-revolution plots of the universe in phases A, B, C
:width: 100%
```

```{image} assets/cdt/volume_profiles_profile.png
:alt: Line plot of N_3(tau) for all three phases
:width: 80%
:align: center
```

---

## Volume-volume correlator and Hausdorff dimension

**Script:** `examples/volume_scaling.py` \
**Reproduces:** Figures 7, 8, 12, Ambjorn, Jurkiewicz & Loll,
*Reconstructing the Universe*, Phys. Rev. D **72** (2005)
\[[arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)\]

**Paper results:**
- **Fig. 7:** The rescaled volume-volume correlator
  $c_{\tilde{N}_4}(x)$ with $x = \delta / (\tilde{N}_4^{\text{eff}})^{1/4}$
  collapses onto a universal peaked curve for system sizes
  $\tilde{N}_4 = 10$k, $20$k, $40$k, $80$k, $160$k.
- **Fig. 8:** The overlap error has a clear minimum at $D_H \approx 3.8$,
  interpreted as $D_H = 4$.

**Our result:** At $N_4 \sim 650$--$1\,900$ the correlator data points
(top left) do not collapse onto a single curve, and the $D_H$ estimate
(top right) shows no clear minimum at $D_H = 4$.  The paper notes that
finite-volume effects are significant below $\tilde{N}_4 = 20$k and uses
$T = 80$ time slices (vs. our $\sim 9$--$11$).  The volume-difference
distribution (bottom left) decays consistently with the Gaussian fit, in
qualitative agreement with Fig. 12.

To reproduce the scaling collapse, run with `--n-simplices 10000` or
larger and `--n-meas 100+`.

```bash
python examples/volume_scaling.py \
    --n-simplices 80000 --n-therm 500 --n-meas 200 \
    --meas-interval 50
```

```{image} assets/cdt/volume_scaling.png
:alt: Volume scaling analysis with correlator and Hausdorff dimension
:width: 100%
```

---

## Phase diagram

**Script:** `examples/phase_diagram.py` \
**Reproduces:** Figure 3, Ambjorn, Jurkiewicz & Loll,
*Reconstructing the Universe*, Phys. Rev. D **72** (2005)
\[[arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)\];
and the phase diagram from Gorlich,
*Introduction to Causal Dynamical Triangulations* (2013)

**Paper result (Fig. 3):** The $(\kappa_0, \Delta)$ plane splits into
three phases -- A (large $\kappa_0$), B (small $\kappa_0$, small $\Delta$),
C (moderate $\kappa_0$, $\Delta > 0$) -- separated by first-order
transition lines.

**Our result:** At $N_4 \sim 400$ per point the phase diagram is
uniformly phase C across the entire scanned region.  The phase
transitions are not visible because they require lattice sizes of
$N_4 > 20\,000$ to resolve: at small volumes the system does not have
enough degrees of freedom to collapse into phase B or fragment into
phase A.  The white star marks the de Sitter point $\kappa_0 = 2.2$,
$\Delta = 0.6$ used in all measurements in the paper.

To see the phase boundaries, run with `--n-simplices 20000` or larger
(estimated runtime: many hours per grid point).

```bash
python examples/phase_diagram.py \
    --n-simplices 10000 --n-sweeps 200 --grid-size 20
```

```{image} assets/cdt/phase_diagram.png
:alt: Phase diagram of 4D CDT
:width: 70%
:align: center
```

---

## Summary

| Paper figure | Observable | Status | Notes |
|:------------|:-----------|:------:|:------|
| Fig. 2 | $N_{32}$ distribution | Confirmed | Sharp peaks, correct ratio $N_{32}/N_{41} \approx 2.3$ |
| Figs. 9--10 | Spectral dimension $D_S(\sigma)$ | Qualitatively confirmed | Monotonic rise, finite-size offset |
| Fig. 11 | Kinetic dimension $D_2$ | Consistent | Minimum near $D_2 = 2$ |
| Figs. 4--6 | Volume profiles A/B/C | Below threshold | Requires $N_4 > 10\,000$ |
| Figs. 7--8 | Hausdorff dimension $D_H$ | Below threshold | Requires $N_4 > 20\,000$ |
| Fig. 3 | Phase diagram | Below threshold | Requires $N_4 > 20\,000$ |
| Figs. 12--13 | Effective action | Partially confirmed | $D_2$ minimum correct; $\cos^3$ needs larger lattice |

---

## References

1. J. Ambjorn, J. Jurkiewicz, R. Loll, *Reconstructing the Universe*,
   Phys. Rev. D **72** (2005),
   [arXiv:hep-th/0505154](https://arxiv.org/abs/hep-th/0505154)
2. J. Ambjorn, J. Jurkiewicz, R. Loll, *Spectral dimension of the universe*,
   Phys. Rev. Lett. **95** (2005),
   [arXiv:hep-th/0505113](https://arxiv.org/abs/hep-th/0505113)
3. A. Gorlich, *Introduction to Causal Dynamical Triangulations*,
   Lecture notes (Zakopane, 2013)
4. R. Loll, *Quantum Gravity from Causal Dynamical Triangulations: A Review*,
   Class. Quant. Grav. **37** (2020),
   [arXiv:1905.08669](https://arxiv.org/abs/1905.08669)
5. J. Brunekreef, A. Görlich, R. Loll, *Simulating CDT quantum gravity*,
   [arXiv:2310.16744](https://arxiv.org/abs/2310.16744)
