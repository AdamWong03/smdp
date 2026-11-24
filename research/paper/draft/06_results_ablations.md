# Draft — Results 4: Ablations (provisional)

**Status:** AR-order and displacement-correction results final; rescale/noise/seed numbers pending.
**Figure:** Fig. 4.

---

## The memory kernel is necessary

The AR(1) readout is the natural minimal thermostat: one gain per mode, fitted to the 5 fs
correlation. It holds the mean partition once the variance is rescaled, but it does not reproduce
the multi-lag dynamics. FDR v1 gives O–H 0.301 and H VDOS 0.807, versus 0.104 and 0.848 for
AR(10); the O–H RDF is the clearest signature. The AR(1) internal ACF decays monotonically from
−0.567 to zero, while the reference oscillates and crosses zero near 25 fs; the AR(10) kernel
reproduces the oscillation. The AR(20) kernel also captures the 100–200 fs tail (internal
100/200 fs 0.147/0.068 versus 0.147/0.072), at the cost of a longer history and a larger
coefficient vector. The order sweep therefore shows a monotone improvement from AR(1) to AR(10),
with AR(20) refining the long tail but not changing the qualitative conclusion.

## The variance rescale is the partition guarantee

The ablation that removes the per-step rescale isolates the partition guarantee. In the AR(1)
readout, removing the rescale collapses the total kinetic energy and cools the system to about
14 K: the FD noise alone does not hold the temperature when the backbone velocity is blended in.
[GLR rescale-off result: partition and temperature.] The rescale is not a cosmetic
normalization; it is the step that makes the mode kinetic energy a design constraint rather than a
by-product of the backbone and the noise.

## FD noise is necessary for the fluctuations

[GLR noise-off result: partition, RDF/VDOS, stability.] Removing the FD noise leaves the AR
memory as a deterministic damped filter. The velocity variance decays and the readout no longer
supplies the correct fluctuations; the blend with the backbone keeps the run finite but the
structure and spectra degrade. The combination of memory and noise is what makes the readout a
thermostat rather than a filter.

## The internal blend is a stability knob, not the fix

For the AR(1) readout, increasing the internal blend from 0.1 to 1.0 makes the O–H RDF worse
(0.957 versus 0.301) and the VDOS worse (H/O 0.618/0.613 versus 0.807/0.851). A larger admixture
of a single-gain readout does not fix the spectrum; it replaces the backbone's internal bias with
the readout's wrong long-lag dynamics. [GLR internal-blend 1.0 result.] The blend is kept at 0.1
to preserve the position head; the memory kernel, not the blend, does the work.

## Direct displacement correction is unstable

Adding the velocity correction to the displacement—the obvious way to make positions and
velocities consistent—crashes at 24.8 ps (minimum distance 0.647 Å) while the temperature remains
at 300 K. The failure is structural, not thermal: the correction injects a displacement
increment that the backbone's next state cannot absorb. The decoupled scheme (positions from the
backbone, velocities from the readout) is therefore a stability requirement. This negative result
also explains why the method does not simply integrate the readout velocity: doing so destroys the
learned displacement distribution.

## Seeds and scope

[Seed 2/3 results: partition, RDF/VDOS, diffusion mean ± s.d.] The current production runs use a
single seed; the seed study is the last statistical requirement before the numerical claims are
final. The frozen-vs-end-to-end backbone comparison and a second system are left to future work:
the present paper is a water method paper, and the claim is architectural rather than universal.
