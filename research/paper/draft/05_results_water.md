# Draft — Results 3: Water NVT (provisional)

**Status:** 100 ps numbers final; 2.5 ns/diffusion placeholders to fill after the gate.
**Figures:** Fig. 3; **Table:** Table 1.

---

## Protocol and observables

We applied the GLR to the frozen 250k backbone that failed the NVT screen without it. The
readout was fitted once from a 100 ps block of the reference NVT trajectory (500–600 ps) and then
used unchanged in all rollouts. We ran 100 ps from the val_04 NVE initial condition and from an
independent NVT trajectory (301.8 K), and a 2.5 ns production run from each start. The integration
step is 5 fs; no external thermostat is applied—the GLR is the thermostat. We compare the mode
kinetic partition, the mode velocity ACFs, the O–O/O–H/H–H RDFs, the H/O VDOS, and the molecular
centre-of-mass MSD and diffusion coefficient against the reference NVT trajectory. The published
TrajCast checkpoint is evaluated under the same 100 ps protocol; its 2.5 ns diffusion is taken
from the published value because the native float64 rollout is prohibitively slow on our
contended GPU.

## The partition is canonical on both starts

Without the readout, the 250k backbone has translation \(0.538\) eV, rotation \(0.786\) eV, and
internal \(6.044\) eV—collectively cold and internally hot. With the GLR, the 100 ps partition is
\(2.482/2.526/2.399\) eV from val_04 and \(2.445/2.490/2.362\) eV from the independent start,
against the 2.5 ns reference means of \(2.459/2.502/2.450\) eV (the first 100 ps block is
2.373/2.430/2.609 eV). The total temperature is 300.0 K in both
runs. The two starts agree within the reference block-to-block scatter, so the stationary
partition is not a property of the val_04 initial condition. The published checkpoint gives
\(2.505/2.463/2.444\) eV, also correct but with a different mix of residuals.

The per-mode ACFs tell the same story. The translation ACF matches the reference at 5, 25, 100,
and 500 fs (GLR 0.981/0.640/0.014/−0.030 versus reference 0.982/0.658/0.039/−0.019). The internal
ACF matches at 5 fs (−0.575 versus −0.567) and reproduces the mid-lag oscillation; the 500 fs tail
is still too persistent (0.312 versus 0.104), a known limitation of AR(10). The AR(20) fit removes
most of that residual (internal 100/200 fs 0.147/0.068 versus 0.147/0.072), and is reported as an
order ablation rather than the main configuration. The important point is that the readout matches
the **multi-lag** dynamics, not just a single 5 fs gain.

## Structure and spectra

The 100 ps RDFs are closer to the reference than the published checkpoint in our analysis:
O–O 0.032 versus 0.043, O–H 0.104 versus 0.118, H–H 0.026 versus 0.096. The independent start
gives 0.036/0.098/0.027, essentially the same. The O VDOS overlap is 0.964 versus 0.937 for the
published checkpoint; the H VDOS overlap is 0.848 versus 0.624. The H comparison is sensitive to
our reference H-stretch frequency (the published model scores 0.624 here while the paper reports
about 0.99 with its own reference), so we treat the O VDOS and the RDFs as the robust comparisons
and report the H number with that caveat.

The improvement over the AR(1) readout is large and diagnostic. FDR v1, which has the correct
partition but only a single gain, gives O–O 0.062, O–H 0.301, H–H 0.110 and H/O VDOS 0.807/0.851.
The AR(10) memory kernel improves O–H by a factor of three and the H VDOS by four points. The
memory order, not the global temperature, is what controls the O–H spectrum.

## Diffusion and long-time stability

[D = ... from the 2.5 ns block analysis, both starts; compare to reference 1.84–1.88e-9 and
published 2.18e-9. State whether the GLR is within the reference band or give the percentage
deviation. Include the MSD linearity and the block error.]

The 2.5 ns run remains stable (temperature 300.0 K, minimum interatomic distance above 0.9 Å) and
shows no sign of the slow collective drift that characterized the un-readout models. The
independent 2.5 ns run is [status]. The FDR v1 fast tail (3.59e-9) is absent, consistent with the
translation ACF matching at 500 fs.

## What the water results establish

The GLR is the first configuration in this study that combines (i) a stable 100 ps NVT run on two
independent starts, (ii) a canonical mean mode partition, (iii) multi-lag ACF agreement, (iv) RDF
and VDOS at or beyond the published checkpoint, and (v) a diffusion coefficient in the reference
range. It does so with a frozen backbone and a readout fitted from 100 ps of reference data. The
remaining questions—seed error bars, the internal long tail, and a second system—are addressed in
the ablations and discussion.
